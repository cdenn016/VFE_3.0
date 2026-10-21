r"""Beta-weighted activation of the routing-cost cap, per layer and head (GL(K)_attention_v2, line 838).

The relational router bounds every cost to :math:`[0, \kappa]` before the softmax,
:math:`\bar D_{ij} = \min(D_{ij} - m_i, \kappa)` with :math:`\kappa` = ``cfg.kl_max`` (non-finite costs are
also set to :math:`\kappa`) and the clamp reference :math:`m_i` of ``routing_kl_max_reference``: :math:`m_i = 0`
under ``'absolute'``, the row minimum :math:`\min_{j \in \mathcal S_i} D_{ij}` under ``'row_min'``
(:func:`vfe3.gradients.relational.clamped_routing_cost`), the query-only row constant :math:`c_i` of the cost plus
the admissible row mean of the key log-volumes under ``'query_shifted'`` (audit 2026-10-20 item 1), and :math:`c_i`
alone under ``'query_shifted_v1'``; under both the bound is from above only (audit 2026-10-17 item 2). The captured ``score_energy_pre`` is
:math:`\bar D_{ij}` either way, so an entry is *at the cap* when :math:`\bar D_{ij} \ge \kappa`; under
``'row_min'`` that is a gap to the row's best source, ``max_cost_uncapped`` is the largest uncapped gap, and
the reference is recorded in the summary (``kl_max_reference``). The unweighted
fraction of capped entries says how often the bound is reached; the routing weight on those entries,

.. math::
    m_i = \sum_{j \le i,\ \bar D_{ij} \ge \kappa} \beta_{ij},

says how much of the routing the bound touches, and bounds the clamp's effect on the row,
:math:`\mathrm{TV}(\beta_i, \beta_i^{\text{unclamped}}) \le m_i/(1 - m_i)`; ``capped_beta_mass_max`` is its row
maximum. Where the capture carries the pre-clamp row extremes ``score_raw_row_min_pre`` and
``score_raw_row_max_pre`` (the kernel readout, audit 2026-10-17 item 9), ``raw_row_min_mean``, ``raw_row_min_max``
and ``raw_row_gap_mean`` report the raw row minimum and row range of the cost before the reference and the clamp,
which separate a row just above the cap from one far above it (NaN without the capture). Two query-row populations are reported, each with its
causal sources :math:`j` (finite attention log-prior; every source with :math:`\beta_{ij} > 0` when the
model has no log-prior). Both keep only rows with at least two causal sources: a single-source row has
:math:`\beta_{ii} = 1` whatever its cost, so the cap cannot change its routing.

* ``scored``: the scored query rows of the evaluation protocol, so every position of the split counts
  exactly once, at the context it is scored with;
* ``computed``: every query row with a real input token in every evaluated window. With overlapping
  windows a position appears once per window; the early rows of a window are not scored but feed the
  scored rows through the later layers, so a cap that binds only there still changes the prediction.

:class:`KLCapObserver` consumes the per-layer captures ``score_energy_pre`` (capped costs) and
``beta_pre`` (routing weights), both ``(B, H, N, N)``, that ``forward_beliefs`` records under
``capture={"diagnostic": {}}``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import torch

POPULATIONS: Tuple[str, ...] = ("scored", "computed")
# Accumulated per (population, layer, head): slot index of each sum.
_SLOTS: Tuple[str, ...] = (
    "n_rows", "n_entries", "n_capped", "capped_beta_mass", "rows_with_cap", "capped_at_window_start",
    "n_raw_rows", "raw_row_min_sum", "raw_row_gap_sum")


def causal_support(
    beta:      torch.Tensor,                     # (B, H, N, N)
    log_prior: Optional[torch.Tensor],           # (H, N, N), (N, N) or (B, H, N, N)
) -> torch.Tensor:                               # (B, H, N, N) bool
    r"""Allowed sources: the finite entries of the attention log-prior, else :math:`\beta_{ij} > 0`."""
    if isinstance(log_prior, torch.Tensor):
        prior = log_prior
        while prior.dim() < 4:
            prior = prior.unsqueeze(0)
        return torch.isfinite(prior).expand_as(beta)
    return beta > 0.0


class KLCapObserver:
    r"""Accumulate the cap statistics over :func:`vfe3.posthoc.evaluate.nll_pass` batches."""

    def __init__(self, kl_max: float, kl_max_reference: str = "absolute") -> None:
        self.kl_max = float(kl_max)
        self.kl_max_reference = str(kl_max_reference)
        self._sums: Dict[Tuple[str, int, int], np.ndarray] = {}
        self._max_beta_capped: Dict[Tuple[str, int, int], float] = {}
        self._max_cost_uncapped: Dict[Tuple[str, int, int], float] = {}
        self._max_capped_mass: Dict[Tuple[str, int, int], float] = {}
        self._max_raw_row_min: Dict[Tuple[str, int, int], float] = {}
        self._windows_with_cap: Dict[Tuple[str, int], Set[int]] = {}
        self.n_batches = 0
        self.missing_captures = 0

    def update(
        self,
        diagnostic: Mapping[str, Any],
        valid:      torch.Tensor,                 # (B, N) scored-target mask = scored query rows
        meta:       Mapping[str, Any],            # windows, starts, input_valid (B, N) real input tokens
    ) -> None:
        captures = diagnostic.get("layer_conditional_captures") or []
        if not captures:
            self.missing_captures += 1
            return
        self.n_batches += 1
        log_prior = diagnostic.get("log_prior")
        windows = np.asarray(meta["windows"], dtype=np.int64)
        input_valid = meta.get("input_valid")
        if input_valid is None:
            input_valid = torch.ones_like(valid)
        for layer, cap in enumerate(captures):
            score, beta = cap.get("score_energy_pre"), cap.get("beta_pre")
            if score is None or beta is None:
                self.missing_captures += 1
                continue
            score, beta = score.float(), beta.float()                          # (B, H, N, N)
            B, H, N, _ = beta.shape
            support = causal_support(beta, log_prior)                           # (B, H, N, N)
            multi_source = support.sum(dim=-1) >= 2                             # (B, H, N)
            at_cap = score >= self.kl_max
            raw_low, raw_high = cap.get("score_raw_row_min_pre"), cap.get("score_raw_row_max_pre")
            if raw_low is not None and raw_high is not None:
                raw_low, raw_high = raw_low.double(), raw_high.double()          # (B, H, N)
                if raw_low.dim() == 2:
                    raw_low, raw_high = raw_low.unsqueeze(1), raw_high.unsqueeze(1)
            else:
                raw_low = raw_high = torch.full((B, H, N), float("nan"), dtype=torch.float64, device=beta.device)
            for population, row_mask in (("scored", valid), ("computed", input_valid)):
                rows = (torch.as_tensor(row_mask, device=beta.device)[:, None, :].expand(B, H, N)
                        & multi_source)                                         # (B, H, N)
                entries = support & rows[..., None]                                 # (B, H, N, N)
                capped = entries & at_cap                                           # (B, H, N, N)
                capped_mass = torch.where(capped, beta, torch.zeros_like(beta)).sum(dim=-1)   # (B, H, N)
                raw_rows = rows & torch.isfinite(raw_low)                               # (B, H, N)
                zero = torch.zeros((), dtype=torch.float64, device=beta.device)
                stacked = torch.stack([
                    rows.sum(dim=(0, 2)).double(),
                    entries.sum(dim=(0, 2, 3)).double(),
                    capped.sum(dim=(0, 2, 3)).double(),
                    capped_mass.double().sum(dim=(0, 2)),
                    capped.any(dim=-1).sum(dim=(0, 2)).double(),
                    capped[..., 0].sum(dim=(0, 2)).double(),
                    raw_rows.sum(dim=(0, 2)).double(),
                    torch.where(raw_rows, raw_low, zero).sum(dim=(0, 2)),
                    torch.where(raw_rows, raw_high - raw_low, zero).sum(dim=(0, 2)),
                ], dim=-1).cpu().numpy()                                           # (H, len(_SLOTS))
                beta_capped = torch.where(capped, beta, torch.full_like(beta, -1.0)).amax(dim=(0, 2, 3))
                cost_uncapped = torch.where(entries & ~capped, score,
                                            torch.full_like(score, -1.0)).amax(dim=(0, 2, 3))   # (H,)
                beta_capped, cost_uncapped = beta_capped.cpu().tolist(), cost_uncapped.cpu().tolist()
                mass_max = torch.where(rows, capped_mass, torch.full_like(capped_mass, -1.0)).amax(dim=(0, 2))
                raw_max = torch.where(raw_rows, raw_low, torch.full_like(raw_low, -math.inf)).amax(dim=(0, 2))
                mass_max, raw_max = mass_max.cpu().tolist(), raw_max.cpu().tolist()
                for head in range(H):
                    key = (population, layer, head)
                    self._sums[key] = self._sums.get(key, np.zeros(len(_SLOTS))) + stacked[head]
                    if beta_capped[head] >= 0.0:
                        self._max_beta_capped[key] = max(self._max_beta_capped.get(key, 0.0), beta_capped[head])
                    if cost_uncapped[head] >= 0.0:
                        self._max_cost_uncapped[key] = max(self._max_cost_uncapped.get(key, 0.0),
                                                           cost_uncapped[head])
                    if mass_max[head] >= 0.0:
                        self._max_capped_mass[key] = max(self._max_capped_mass.get(key, 0.0), mass_max[head])
                    if math.isfinite(raw_max[head]):
                        self._max_raw_row_min[key] = max(self._max_raw_row_min.get(key, -math.inf), raw_max[head])
                window_has_cap = capped.any(dim=(1, 2, 3)).cpu().numpy()            # (B,)
                if window_has_cap.any():
                    self._windows_with_cap.setdefault((population, layer), set()).update(
                        int(w) for w in windows[window_has_cap])

    def rows(self) -> List[Dict[str, Any]]:
        r"""One row per (population, layer, head): counts, fractions, capped mass and the largest costs."""
        out: List[Dict[str, Any]] = []
        for (population, layer, head), sums in sorted(self._sums.items(),
                                                      key=lambda item: (POPULATIONS.index(item[0][0]),) + item[0][1:]):
            s = dict(zip(_SLOTS, sums.tolist()))
            key = (population, layer, head)
            out.append({
                "population": population, "layer": layer, "head": head,
                "n_rows": int(s["n_rows"]), "n_entries": int(s["n_entries"]), "n_capped": int(s["n_capped"]),
                "frac_capped": s["n_capped"] / s["n_entries"] if s["n_entries"] else float("nan"),
                "rows_with_cap": int(s["rows_with_cap"]),
                "frac_rows_with_cap": s["rows_with_cap"] / s["n_rows"] if s["n_rows"] else float("nan"),
                "capped_beta_mass_mean": s["capped_beta_mass"] / s["n_rows"] if s["n_rows"] else float("nan"),
                "capped_beta_mass_total": s["capped_beta_mass"],
                "max_beta_capped": self._max_beta_capped.get(key, float("nan")),
                "max_cost_uncapped": self._max_cost_uncapped.get(key, float("nan")),
                "capped_at_window_start": int(s["capped_at_window_start"]),
                "capped_beta_mass_max": self._max_capped_mass.get(key, float("nan")),
                "raw_row_min_mean": (s["raw_row_min_sum"] / s["n_raw_rows"] if s["n_raw_rows"]
                                     else float("nan")),
                "raw_row_min_max": self._max_raw_row_min.get(key, float("nan")),
                "raw_row_gap_mean": (s["raw_row_gap_sum"] / s["n_raw_rows"] if s["n_raw_rows"]
                                     else float("nan")),
            })
        return out

    def summary(self) -> Dict[str, Any]:
        r"""Per-population, per-layer totals and the overall verdict (``cap_active``), with the cap value."""
        rows = self.rows()
        per_layer: List[Dict[str, Any]] = []
        for population in POPULATIONS:
            for layer in sorted({r["layer"] for r in rows if r["population"] == population}):
                mine = [r for r in rows if r["population"] == population and r["layer"] == layer]
                n_rows = sum(r["n_rows"] for r in mine)
                n_entries = sum(r["n_entries"] for r in mine)
                n_capped = sum(r["n_capped"] for r in mine)
                mass = sum(r["capped_beta_mass_total"] for r in mine)
                per_layer.append({
                    "population": population, "layer": layer, "n_capped": n_capped, "n_entries": n_entries,
                    "frac_capped": n_capped / n_entries if n_entries else float("nan"),
                    "capped_beta_mass_mean": mass / n_rows if n_rows else float("nan"),
                    "rows_with_cap": sum(r["rows_with_cap"] for r in mine),
                    "capped_at_window_start": sum(r["capped_at_window_start"] for r in mine),
                    "max_beta_capped": max((r["max_beta_capped"] for r in mine
                                            if np.isfinite(r["max_beta_capped"])), default=float("nan")),
                    "max_cost_uncapped": max((r["max_cost_uncapped"] for r in mine
                                              if np.isfinite(r["max_cost_uncapped"])), default=float("nan")),
                    "windows_with_cap": sorted(self._windows_with_cap.get((population, layer), set())),
                })
        return {
            "kl_max": self.kl_max,
            "kl_max_reference": self.kl_max_reference,
            "populations": {
                "scored": "scored query rows of the evaluation protocol (each position once) and their causal sources",
                "computed": "every query row with a real input token in every evaluated window, and its causal sources",
            },
            "cap_active": any(r["n_capped"] > 0 for r in rows),
            "cap_active_scored": any(r["n_capped"] > 0 for r in rows if r["population"] == "scored"),
            "n_batches": self.n_batches,
            "missing_captures": self.missing_captures,
            "per_layer": per_layer,
        }
