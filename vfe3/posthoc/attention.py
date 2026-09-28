r"""Per-head routing statistics over an evaluation split (S2 panel B of GL(K)_attention_v2).

For a scored query row :math:`i` with causal sources :math:`\mathcal S_i` (finite attention log-prior),
:math:`|\mathcal S_i| \ge 2`, and routing weights :math:`\beta_{ij}` of head :math:`h` (``beta_pre``),
the statistics are those of the training-time ``attention_stats.csv``:

.. math::
    \bar H_i = \frac{-\sum_j \beta_{ij}\log\beta_{ij}}{\log|\mathcal S_i|}, \qquad
    d_i = \sum_j \beta_{ij}\,|i - j|, \qquad
    s_i = \beta_{ii}, \qquad
    o_i = \sum_j \min(\beta_{ij}, \pi_{ij}),

with :math:`\pi_{i\cdot} = \operatorname{softmax}_j(\log\pi_{ij})` the routing the attention log-prior
alone would give (:math:`o_i = 1` when the costs change nothing; the uniform causal prior when the model
has none). The normalizer :math:`\log|\mathcal S_i|` is the causal maximum :math:`\log(i+1)` of a
window, not :math:`\log N`. Here the rows are the SCORED query rows of the evaluation protocol, so every
position of the split counts exactly once (the training CSV uses all rows of 8 fixed windows).

Means are row-weighted over the split; 95% intervals come from the same resampling-unit (article)
bootstrap as the cross-entropy. For each head the minimum-entropy row is kept with its stream position
and its most-weighted source, for the token context in the report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from vfe3.posthoc.klcap import causal_support

STATS: Tuple[str, ...] = ("entropy_norm", "distance_mean", "self_weight", "prior_overlap", "entropy_nats")
HIST_BINS: int = 200                       # normalized-entropy histogram resolution (quantiles to 1/200)
CI_RESAMPLES: int = 2_000                  # unit-bootstrap resamples per statistic


class AttentionStatsObserver:
    r"""Accumulate :data:`STATS` per (layer, head), per resampling unit, over ``nll_pass`` batches.

    ``position_units`` maps a stream position to its resampling unit (article); a query row belongs
    to the unit of the target it predicts. ``None`` uses the window index as the unit.
    """

    def __init__(self, position_units: Optional[np.ndarray] = None) -> None:
        self.position_units = (None if position_units is None
                               else torch.as_tensor(np.asarray(position_units, dtype=np.int64)))
        self.n_units = (int(self.position_units.max()) + 1) if self.position_units is not None else None
        self._unit_sums: Dict[int, torch.Tensor] = {}          # layer -> (H, U, len(STATS) + 1) float64
        self._hist: Dict[int, torch.Tensor] = {}               # layer -> (H, HIST_BINS) float64
        self._minimum: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._window_units: Dict[int, int] = {}                # window -> dense unit index (no position map)
        self.n_batches = 0
        self.missing_captures = 0

    def _row_units(self, valid: torch.Tensor, meta: Mapping[str, Any]) -> torch.Tensor:   # (B, N) int64
        B, N = valid.shape
        starts = torch.as_tensor(np.asarray(meta["starts"], dtype=np.int64))
        if self.position_units is not None:
            target_pos = (starts[:, None] + torch.arange(N)[None, :] + 1).clamp_max(
                self.position_units.numel() - 1)                                        # (B, N)
            return self.position_units[target_pos]
        dense = [self._window_units.setdefault(int(w), len(self._window_units)) for w in meta["windows"]]
        return torch.as_tensor(dense, dtype=torch.int64)[:, None].expand(B, N)

    def update(
        self,
        diagnostic: Mapping[str, Any],
        valid:      torch.Tensor,                 # (B, N) scored query rows
        meta:       Mapping[str, Any],
    ) -> None:
        captures = diagnostic.get("layer_conditional_captures") or []
        if not captures:
            self.missing_captures += 1
            return
        self.n_batches += 1
        log_prior = diagnostic.get("log_prior")
        units_cpu = self._row_units(valid.cpu(), meta)                                # (B, N)
        windows = np.asarray(meta["windows"], dtype=np.int64)
        starts = np.asarray(meta["starts"], dtype=np.int64)
        for layer, cap in enumerate(captures):
            beta = cap.get("beta_pre")
            if beta is None:
                self.missing_captures += 1
                continue
            beta = beta.float()                                                       # (B, H, N, N)
            B, H, N, _ = beta.shape
            device = beta.device
            support = causal_support(beta, log_prior)                                 # (B, H, N, N)
            if isinstance(log_prior, torch.Tensor):
                prior_logits = log_prior.float()
                while prior_logits.dim() < 4:
                    prior_logits = prior_logits.unsqueeze(0)
                prior = torch.softmax(prior_logits, dim=-1).expand_as(beta)          # (B, H, N, N)
            else:
                prior = support.float() / support.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            n_sources = support.sum(dim=-1)                                           # (B, H, N)
            rows = (n_sources >= 2) & valid.to(device)[:, None, :]                    # (B, H, N)
            entropy = -torch.special.xlogy(beta, beta).sum(dim=-1)                   # (B, H, N) nats
            entropy_norm = entropy / torch.log(n_sources.clamp_min(2).float())       # (B, H, N)
            offsets = torch.arange(N, device=device)
            distance = (beta * (offsets.view(-1, 1) - offsets.view(1, -1)).abs().float()).sum(dim=-1)
            self_weight = torch.diagonal(beta, dim1=-2, dim2=-1)                      # (B, H, N)
            overlap = torch.minimum(beta, prior).sum(dim=-1)                          # (B, H, N)
            values = torch.stack([entropy_norm.double(), distance.double(), self_weight.double(),
                                  overlap.double(), entropy.double(), torch.ones_like(entropy.double())],
                                 dim=-1)                                              # (B, H, N, S+1)
            values = torch.where(rows[..., None], values, torch.zeros_like(values))   # unscored rows add 0
            values = values.permute(1, 0, 2, 3).reshape(H, B * N, len(STATS) + 1)
            unit_index = units_cpu.reshape(-1).to(device)                             # (B*N,)
            n_units = self.n_units if self.n_units is not None else len(self._window_units)
            sums = torch.zeros(H, n_units, len(STATS) + 1, dtype=torch.float64, device=device)
            sums.index_add_(1, unit_index, values)                                    # (H, U, S+1)
            sums = sums.cpu()
            previous = self._unit_sums.get(layer)
            if previous is not None and previous.shape[1] < sums.shape[1]:
                previous = torch.cat([previous, torch.zeros(H, sums.shape[1] - previous.shape[1],
                                                            len(STATS) + 1, dtype=torch.float64)], dim=1)
            self._unit_sums[layer] = sums if previous is None else previous + sums
            hist = self._hist.setdefault(layer, torch.zeros(H, HIST_BINS, dtype=torch.float64))
            for head in range(H):
                selected = entropy_norm[:, head][rows[:, head]]
                if selected.numel():
                    hist[head] += torch.histc(selected.clamp(0.0, 1.0), bins=HIST_BINS, min=0.0,
                                              max=1.0).double().cpu()
            masked = torch.where(rows, entropy_norm, torch.full_like(entropy_norm, float("inf")))
            flat = masked.permute(1, 0, 2).reshape(H, B * N)                          # (H, B*N)
            best, where = flat.min(dim=-1)
            best, where = best.cpu().tolist(), where.cpu().tolist()
            for head in range(H):
                if not np.isfinite(best[head]):
                    continue
                key = (layer, head)
                if key in self._minimum and self._minimum[key]["entropy_norm"] <= best[head]:
                    continue
                b, i = divmod(int(where[head]), N)
                row = beta[b, head, i]                                                # (N,)
                source = int(row.argmax())
                self._minimum[key] = {
                    "entropy_norm": float(best[head]), "window": int(windows[b]), "query_pos": i,
                    "query_stream_pos": int(starts[b] + i), "target_stream_pos": int(starts[b] + i + 1),
                    "argmax_source_pos": source, "argmax_source_stream_pos": int(starts[b] + source),
                    "argmax_beta": float(row[source]), "n_sources": int(n_sources[b, head, i]),
                }

    def rows(self, *, n_resamples: int = CI_RESAMPLES, seed: int = 0) -> List[Dict[str, Any]]:
        r"""One row per (layer, head): means, unit-bootstrap intervals, entropy quantiles, minimum row."""
        from vfe3.posthoc.evaluate import bootstrap_ratio

        out: List[Dict[str, Any]] = []
        centers = (np.arange(HIST_BINS) + 0.5) / HIST_BINS
        for layer in sorted(self._unit_sums):
            sums = self._unit_sums[layer].numpy()                                    # (H, U, S+1)
            hist = self._hist[layer].numpy()
            for head in range(sums.shape[0]):
                counts = sums[head, :, -1]
                keep = counts > 0
                row: Dict[str, Any] = {"layer": int(layer), "head": int(head), "n_rows": int(counts.sum())}
                for slot, name in enumerate(STATS):
                    interval = bootstrap_ratio(sums[head, keep, slot], counts[keep],
                                               n_resamples=n_resamples, seed=seed)
                    row[name] = interval["point"]
                    row[f"{name}_lo"] = interval["lo"]
                    row[f"{name}_hi"] = interval["hi"]
                row["n_units"] = int(keep.sum())
                cdf = np.cumsum(hist[head])
                total = cdf[-1] if cdf.size else 0.0
                for q in (0.05, 0.5, 0.95):
                    row[f"entropy_norm_q{int(round(q * 100)):02d}"] = (
                        float(centers[min(HIST_BINS - 1, int(np.searchsorted(cdf, q * total)))])
                        if total > 0 else float("nan"))
                minimum = self._minimum.get((layer, head), {})
                for key, value in minimum.items():
                    row[f"min_{key}" if key != "entropy_norm" else "min_entropy_norm"] = value
                out.append(row)
        return out

    def layer_rows(self) -> List[Dict[str, Any]]:
        r"""Row-weighted means over the heads of each layer (no intervals)."""
        out: List[Dict[str, Any]] = []
        for layer in sorted(self._unit_sums):
            totals = self._unit_sums[layer].sum(dim=(0, 1)).numpy()                  # (S+1,)
            n = totals[-1]
            out.append({"layer": int(layer), "n_rows": int(n),
                        **{name: (float(totals[slot] / n) if n else float("nan"))
                           for slot, name in enumerate(STATS)}})
        return out
