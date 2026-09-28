r"""Post-hoc residual ratios and LayerNorm change per layer (the v2:899 quantities, S1 panel A endpoints).

For layer :math:`\ell` with input mean :math:`\mu_i` (the stack-entry belief at :math:`\ell = 0`, else
layer :math:`\ell-1`'s output) and the layer's attention output :math:`\mu_i + m_i` (the pre-transform
converged mean), every scored target position :math:`i` of an evaluation pass contributes

.. math::
    \rho_i = \frac{\|m_i\|}{\|\mu_i\|}, \qquad
    \delta_i = \frac{\|\mathrm{LN}(\mu_i + m_i) - \mathrm{LN}(m_i)\|}{\|\mathrm{LN}(\mu_i + m_i)\|}.

These are the definitions of the in-training probe (``vfe3.train._residual_and_attention_rows``, which
writes ``diagnostics/residual_ratio.csv``): :func:`residual_token_values` evaluates them on the same
diagnostic capture fields, and :func:`ln_change_applicability` uses the training probe's own conditions
(``vfe3.train._residual_update_is_additive``, the block norm, no head mixer or CG coupling between the
E-step and the norm). The training probe covers every position of 8 fixed validation windows at eval
cadence; here the population is the scored targets of the evaluation protocol, so every position of a
split counts exactly once. Quantiles use linear interpolation in float64 over the float32 per-token
values, which is what ``torch.quantile`` computes for :math:`\rho` in the probe; the probe's
:math:`\delta` median is ``torch.median`` (the lower middle value for an even count), while
``ln_change_q50`` here interpolates like every other quantile.

When the layer has no additive residual (``predictive_content`` readout without ``attention_residual``)
:math:`\mu_i + m_i` is not defined; :math:`\rho_i` is then the displacement
:math:`\|\mu^{\mathrm{out}}_i - \mu_i\| / \|\mu_i\|` (as in the probe) and :math:`\delta_i` is not
defined. Both facts are recorded in the result.

The routing self-weight :math:`\beta_{ii}` per layer (v2:899) is not recomputed here: the report takes it
from :class:`vfe3.posthoc.attention.AttentionStatsObserver` on the same pass (:func:`self_weight_summary`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


# ----------------------------------------------------------------------------------------------
# Definitions (shared with the in-training probe)
# ----------------------------------------------------------------------------------------------

def ln_change_applicability(model: torch.nn.Module) -> Tuple[bool, Optional[str]]:
    r"""``(defined, reason)`` for :math:`\delta_i`, with the training probe's four conditions."""
    from vfe3.train import _residual_update_is_additive

    if getattr(model, "block_norm", None) is None:
        return False, "the block has no normalization (norm_type_block='none')"
    if getattr(model, "head_mixer", None) is not None:
        return False, "a head mixer sits between the E-step and the norm, which then does not see mu + m"
    if getattr(model, "cg_coupling", None) is not None:
        return False, "a CG coupling sits between the E-step and the norm, which then does not see mu + m"
    if not _residual_update_is_additive(model):
        return False, "predictive readout without attention_residual: no additive residual to remove"
    return True, None


def residual_additive(model: torch.nn.Module) -> bool:
    r"""Whether the layer update has the residual form :math:`\mu + m` (the probe's predicate)."""
    from vfe3.train import _residual_update_is_additive

    return bool(_residual_update_is_additive(model))


def residual_token_values(
    diagnostic: Mapping[str, Any],
    layer:      int,

    *,
    block_norm: Optional[torch.nn.Module],
    ln_defined: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Per-token :math:`\rho_i` and (when defined) :math:`\delta_i` of ``layer``, each (B, N) float32.

    ``diagnostic`` is a ``forward_beliefs(capture={"diagnostic": {}})`` capture; the arithmetic is the
    in-training probe's line for line (float32 means, vector norms over the last axis).
    """
    source = diagnostic["initial_belief"] if layer == 0 else diagnostic["layer_outputs"][layer - 1]
    mu_in = source.mu.float()                                                 # (B, N, K)
    converged = diagnostic["layer_converged"][layer]
    mu_out = converged.mu.float()                                             # (B, N, K) = mu + m
    message = mu_out - mu_in                                                  # (B, N, K) m_i
    ratio = torch.linalg.vector_norm(message, dim=-1) / torch.linalg.vector_norm(mu_in, dim=-1)   # (B, N)
    if not ln_defined or block_norm is None:
        return ratio, None
    sigma = converged.sigma.float()
    full = block_norm(mu_out, sigma).float()                                  # (B, N, K) LN(mu + m)
    removed = block_norm(message, sigma).float()                              # (B, N, K) LN(m)
    change = (torch.linalg.vector_norm(full - removed, dim=-1)
              / torch.linalg.vector_norm(full, dim=-1))                       # (B, N)
    return ratio, change


def quantile_row(values: np.ndarray, prefix: str) -> Dict[str, float]:
    r"""``{prefix}_q10/q50/q90/min/max/mean`` and ``{prefix}_n`` of ``values`` (NaN when empty)."""
    values = np.asarray(values, dtype=np.float64)
    nan = float("nan")
    row: Dict[str, float] = {f"{prefix}_n": int(values.size)}
    if values.size:
        for q, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
            row[f"{prefix}_q{int(round(100 * q)):02d}"] = float(value)
        row.update({f"{prefix}_min": float(values.min()), f"{prefix}_max": float(values.max()),
                    f"{prefix}_mean": float(values.mean())})
    else:
        for q in QUANTILES:
            row[f"{prefix}_q{int(round(100 * q)):02d}"] = nan
        row.update({f"{prefix}_min": nan, f"{prefix}_max": nan, f"{prefix}_mean": nan})
    return row


# ----------------------------------------------------------------------------------------------
# Observer
# ----------------------------------------------------------------------------------------------

class ResidualObserver:
    r"""Collect :math:`\rho_i` and :math:`\delta_i` per layer over the scored targets of ``nll_pass``."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.block_norm = getattr(model, "block_norm", None)
        self.additive = residual_additive(model)
        self.ln_defined, self.ln_reason = ln_change_applicability(model)
        self._ratios: Dict[int, List[np.ndarray]] = {}
        self._changes: Dict[int, List[np.ndarray]] = {}
        self._nonfinite: Dict[str, int] = {"ratio": 0, "ln_change": 0}
        self.n_batches = 0
        self.missing_captures = 0

    def update(
        self,
        diagnostic: Mapping[str, Any],
        valid:      torch.Tensor,                   # (B, N) scored target positions
        meta:       Mapping[str, Any],
    ) -> None:
        converged = diagnostic.get("layer_converged") or []
        if not converged or "initial_belief" not in diagnostic:
            self.missing_captures += 1
            return
        self.n_batches += 1
        for layer in range(len(converged)):
            ratio, change = residual_token_values(diagnostic, layer, block_norm=self.block_norm,
                                                  ln_defined=self.ln_defined)
            scored = valid.to(ratio.device)                                   # (B, N)
            finite = torch.isfinite(ratio)
            self._nonfinite["ratio"] += int((scored & ~finite).sum())
            self._ratios.setdefault(layer, []).append(
                ratio[scored & finite].detach().float().cpu().numpy())
            if change is not None:
                finite = torch.isfinite(change)
                self._nonfinite["ln_change"] += int((scored & ~finite).sum())
                self._changes.setdefault(layer, []).append(
                    change[scored & finite].detach().float().cpu().numpy())

    def layer_values(self, layer: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        r"""All collected ``(rho, delta)`` values of ``layer`` (delta ``None`` when not defined)."""
        ratios = np.concatenate(self._ratios[layer]) if self._ratios.get(layer) else np.zeros(0, np.float32)
        changes = (np.concatenate(self._changes[layer]) if self._changes.get(layer)
                   else (np.zeros(0, np.float32) if self.ln_defined else None))
        return ratios, changes

    def result(self) -> Dict[str, Any]:
        r"""Per-layer quantiles, the all-layer extremes quoted in v2:899, and the definitions used."""
        rows: List[Dict[str, Any]] = []
        for layer in sorted(self._ratios):
            ratios, changes = self.layer_values(layer)
            row: Dict[str, Any] = {"layer": int(layer), **quantile_row(ratios, "rho")}
            row.update(quantile_row(changes if changes is not None else np.zeros(0), "ln_change"))
            rows.append(row)
        all_ratios = [r["rho_min"] for r in rows if np.isfinite(r["rho_min"])]
        all_changes = [r["ln_change_max"] for r in rows if np.isfinite(r["ln_change_max"])]
        status = "ok" if rows else "no_captures"
        return {
            "status": status,
            "reason": None if rows else "no layer captures reached the observer",
            "layers": rows,
            "rho_min_all_layers": min(all_ratios) if all_ratios else None,
            "ln_change_max_all_layers": max(all_changes) if all_changes else None,
            "residual_additive": self.additive,
            "ratio_definition": ("||m_i|| / ||mu_i||, m_i = (mu_i + m_i) - mu_i" if self.additive else
                                 "displacement ||mu_out_i - mu_i|| / ||mu_i|| (no additive residual: the "
                                 "layer replaces the mean)"),
            "ln_change": {"defined": self.ln_defined, "reason": self.ln_reason,
                          "definition": "||LN(mu_i + m_i) - LN(m_i)|| / ||LN(mu_i + m_i)|| with the layer's "
                                        "own block norm at the converged covariance"},
            "nonfinite_scored_values": dict(self._nonfinite),
            "population": "scored targets of the evaluation protocol (each position of the split once)",
            "training_probe": "vfe3.train._residual_and_attention_rows (diagnostics/residual_ratio.csv: "
                              "every position of 8 fixed validation windows, ln change median only)",
            "n_batches": self.n_batches, "missing_captures": self.missing_captures,
        }


def self_weight_summary(
    head_rows:  Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    r"""Per layer the row-weighted mean self-weight :math:`\bar\beta_{ii}` and its range over heads."""
    out: List[Dict[str, Any]] = []
    for layer_row in layer_rows:
        layer = int(layer_row["layer"])
        heads = [float(r["self_weight"]) for r in head_rows
                 if int(r["layer"]) == layer and r.get("self_weight") is not None
                 and np.isfinite(r["self_weight"])]
        out.append({"layer": layer, "self_weight_mean": float(layer_row.get("self_weight", float("nan"))),
                    "self_weight_head_min": min(heads) if heads else float("nan"),
                    "self_weight_head_max": max(heads) if heads else float("nan"),
                    "n_rows": int(layer_row.get("n_rows", 0))})
    return {"status": "ok" if out else "unavailable", "layers": out,
            "definition": "mean of beta_ii over scored query rows with >= 2 causal sources "
                          "(vfe3.posthoc.attention.AttentionStatsObserver)"}


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------

def residual_table(result: Mapping[str, Any]) -> str:
    r"""Markdown of the v2:899 numbers per layer at one set of weights."""
    from vfe3.semantics.common import format_value

    def sci(value: Any) -> str:
        if value is None or not np.isfinite(float(value)):
            return "–"
        return f"{float(value):.3g}"

    weights = {int(r["layer"]): r for r in (result.get("self_weight") or {}).get("layers", [])}
    lines = ["| Layer | Tokens | rho q10 | rho q50 | rho q90 | rho min | LN change q50 | LN change q90 | "
             "LN change max | Self-weight mean [head range] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in result.get("layers", []):
        sw = weights.get(int(row["layer"]))
        weight = (f"{format_value(sw['self_weight_mean'], 3)} [{format_value(sw['self_weight_head_min'], 3)}, "
                  f"{format_value(sw['self_weight_head_max'], 3)}]" if sw else "–")
        lines.append(f"| {row['layer']} | {row['rho_n']:,} | {sci(row['rho_q10'])} | {sci(row['rho_q50'])} | "
                     f"{sci(row['rho_q90'])} | {sci(row['rho_min'])} | {sci(row['ln_change_q50'])} | "
                     f"{sci(row['ln_change_q90'])} | {sci(row['ln_change_max'])} | {weight} |")
    ln = result.get("ln_change") or {}
    note = [f"\nrho = {result.get('ratio_definition')}; minimum over all layers {sci(result.get('rho_min_all_layers'))}.",
            f"LN change = {ln.get('definition')}; " + (
                f"maximum over all layers {sci(result.get('ln_change_max_all_layers'))}." if ln.get("defined")
                else f"not defined: {ln.get('reason')}."),
            f"Population: {result.get('population')}; {result.get('checkpoint')} on the {result.get('split')} split"
            + (" (SUBSET of windows, not the manuscript number)." if result.get("subset") else ".")]
    sw = result.get("self_weight") or {}
    if sw.get("status") != "ok":
        note.append(f"Self-weights unavailable: {sw.get('reason', 'no attention statistics on this pass')}.")
    return "\n".join(lines) + "\n" + "\n".join(note) + "\n"


def write_residual_outputs(result: Mapping[str, Any], out_dir: Path, *, figures: bool = True) -> List[Path]:
    r"""``tables/residual_<stem>.md``, ``csv/residual_<stem>.csv`` and the S1A endpoint figure."""
    from vfe3.posthoc.figures import plot_residual_ratio

    out_dir = Path(out_dir)
    stem = str(result.get("stem") or "final")
    written: List[Path] = []
    if result.get("status") != "ok":
        return written
    table = out_dir / "tables" / f"residual_{stem}.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(residual_table(result), encoding="utf-8")
    written.append(table)
    weights = {int(r["layer"]): r for r in (result.get("self_weight") or {}).get("layers", [])}
    rows = [dict(r, **{k: v for k, v in weights.get(int(r["layer"]), {}).items() if k != "layer"})
            for r in result["layers"]]
    csv_path = out_dir / "csv" / f"residual_{stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    written.append(csv_path)
    if figures:
        written += plot_residual_ratio(result, out_dir / "figures", f"s1a_residual_ratio_{stem}")
    return written
