r"""History figures of the numerics, decode-margin and block-variance diagnostics (audit 2026-10-17, item 64).

``plot_block_variance_history`` draws ``diagnostics/block_variance.csv``: per layer and stage, the
quantiles over the diagnostic windows of the local variances ``diag(Sigma_i)``, written at eval
cadence by ``vfe3.train``. ``plot_numerics_history`` draws the cumulative numerics counters of every
metrics row, the decode table's variance margin ``decode_sigma_v_min`` (which ``decode_av_precision``
asks the user to watch) and the non-finite fraction of the relational scores. Both figures read only
the columns that exist and raise :class:`vfe3.viz.figures.FigureSkipped` when nothing is left to draw;
``run_artifacts._save_figures`` calls them only when their inputs are present.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from vfe3.viz import figures as _figs

# Cumulative numerics counters on every metrics row: (column, label). Each is a running total since step 0.
NUMERICS_COUNTER_SERIES: Tuple[Tuple[str, str], ...] = (
    ("layernorm_trace_cap_elements", "LayerNorm trace-cap tokens"),
    ("covariance_ceiling_elements", "covariance-ceiling entries"),
    ("covariance_escalation_elements", "float64 covariance recomputes"),
    ("gemm_fallback_slices", "pair-KL GEMM fallback slices"),
    ("source_recovery_elements", "source-factor recoveries"),
)
DECODE_MARGIN_SERIES: Tuple[Tuple[str, str], ...] = (
    ("decode_sigma_v_min", r"$\min_{v,k}\sigma^2_{vk}$ of the decode table"),
)
SCORE_NONFINITE_SERIES: Tuple[Tuple[str, str], ...] = (
    ("relational_score_nonfinite_frac", "non-finite relational scores"),
)
# Every metrics.csv column plot_numerics_history reads (the run finalizer selects exactly these).
NUMERICS_PANEL_COLUMNS: Tuple[str, ...] = tuple(
    key for key, _ in NUMERICS_COUNTER_SERIES + DECODE_MARGIN_SERIES + SCORE_NONFINITE_SERIES)

# block_variance.csv quantile columns drawn, low tail first, with their panel titles.
BLOCK_VARIANCE_QUANTILES: Tuple[Tuple[str, str], ...] = (
    ("var_min", "minimum"),
    ("var_q001", "0.1% quantile"),
    ("var_q01", "1% quantile"),
    ("var_median", "median"),
    ("var_max", "maximum"),
)
BLOCK_VARIANCE_STAGES: Tuple[Tuple[str, str], ...] = (
    ("pre_norm", "converged belief (pre-norm)"),
    ("post_norm", "after the block norm"),
    ("block_out", "layer output"),
)


def read_block_variance(path: Union[str, Path]) -> Optional[Dict[str, List[object]]]:
    r"""``diagnostics/block_variance.csv`` as ``{column: list}``: ``stage`` kept as text, every other cell a
    float (NaN when blank or non-numeric). None when the file is absent, empty or lacks step/layer/stage."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not {"step", "layer", "stage"} <= set(fields):
            return None
        table: Dict[str, List[object]] = {field: [] for field in fields}
        for row in reader:
            for field in fields:
                raw = row.get(field)
                if field == "stage":
                    table[field].append("" if raw is None else str(raw).strip())
                    continue
                try:
                    table[field].append(float(raw) if raw not in (None, "") else float("nan"))
                except ValueError:
                    table[field].append(float("nan"))
    return table if table["step"] else None


def plot_block_variance_history(
    table: Mapping[str, Sequence[object]],   # read_block_variance output

    *,
    path:  Optional[str] = None,
) -> object:
    r"""Per-layer, per-stage variance-quantile history of ``diagnostics/block_variance.csv``.

    One row of panels per stage present (pre-norm belief, post-norm block input, layer output) and one
    column per variance quantile present (minimum, 0.1%, 1%, median, maximum of ``diag(Sigma_i)`` over the
    diagnostic windows); one line per layer, colored by depth. All panels share one log y axis, so a
    collapse of the lower tail at some depth reads directly against the median and the other layers.
    """
    plt = _figs.plt
    steps = np.asarray(table.get("step", []), dtype=float)
    layers_raw = np.asarray(table.get("layer", []), dtype=float)
    stages = np.asarray([str(s) for s in table.get("stage", [])], dtype=object)
    if steps.size == 0 or layers_raw.size != steps.size or stages.size != steps.size:
        raise _figs.FigureSkipped("block_variance.csv has no rows")
    quantiles = [(key, title) for key, title in BLOCK_VARIANCE_QUANTILES if key in table
                 and np.isfinite(np.asarray(table[key], dtype=float)).any()]
    stage_rows = [(key, title) for key, title in BLOCK_VARIANCE_STAGES if (stages == key).any()]
    if not quantiles or not stage_rows:
        raise _figs.FigureSkipped("block_variance.csv has no finite variance quantile")
    finite_layers = layers_raw[np.isfinite(layers_raw)].astype(int)
    layers = sorted(set(finite_layers.tolist()))
    colors = _figs._layer_colors(max(layers) + 1 if layers else 1)
    with _figs._compact_style():
        fig, axes = plt.subplots(len(stage_rows), len(quantiles), squeeze=False, sharex=True, sharey=True,
                                 figsize=(_figs.MAX_FIGURE_WIDTH_IN, 1.55 * len(stage_rows) + 0.7))
        drawn = 0
        for r, (stage, stage_title) in enumerate(stage_rows):
            for c, (key, q_title) in enumerate(quantiles):
                ax = axes[r][c]
                values = np.asarray(table[key], dtype=float)
                for layer in layers:
                    sel = (stages == stage) & (layers_raw == layer) & np.isfinite(steps) & np.isfinite(values)
                    sel &= values > 0                                     # log axis
                    if not sel.any():
                        continue
                    order = np.argsort(steps[sel])
                    ax.plot(steps[sel][order], values[sel][order], "o-", ms=1.8, lw=0.9,
                            color=colors[layer], label=f"layer {layer}")
                    drawn += 1
                ax.set_yscale("log")
                if r == 0:
                    ax.set_title(q_title)
                if c == 0:
                    ax.set_ylabel(stage_title, fontsize=6.5)
                if r == len(stage_rows) - 1:
                    ax.set_xlabel("training step")
        if not drawn:
            plt.close(fig)
            raise _figs.FigureSkipped("block_variance.csv has no positive finite variance")
        first = axes[0][0]
        if _figs._set_finite_xlim(first, steps):
            _figs._step_xaxis(first)
        handles, labels = [], []
        for ax_row in axes:
            for ax in ax_row:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), frameon=False,
                   fontsize=6.0)
        fig.suptitle(r"Local variances $\mathrm{diag}(\Sigma_i)$ by layer and stage (diagnostic windows)")
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        return _figs._save(fig, path)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0)


def plot_numerics_history(
    history: Mapping[str, Sequence[float]],   # step + any of NUMERICS_PANEL_COLUMNS

    *,
    spikes:  Optional[Mapping[str, Sequence[float]]] = None,   # the same columns on divergence-spike rows
    path:    Optional[str] = None,
) -> object:
    r"""Numerics counters and the decode variance margin over training.

    Panel 1: the cumulative counters (LayerNorm trace-cap tokens, covariance-ceiling entries, float64
    covariance recomputes, pair-KL GEMM fallback slices, source-factor recoveries) on a symlog axis, so
    a counter that stays at zero is drawn at zero rather than dropped. Panel 2: ``decode_sigma_v_min``,
    the smallest decode-table variance, on a log axis; its decline is what costs ``decode_av_precision``
    digits. Panel 3: the fraction of relational scores that were non-finite at the diagnostic pass.
    Only panels with a finite column are drawn. Spike rows, when given, are overlaid as crosses.
    """
    plt = _figs.plt
    step = np.asarray(history.get("step", []), dtype=float)
    spike_step = np.asarray((spikes or {}).get("step", []), dtype=float)

    def present(series: Tuple[Tuple[str, str], ...]) -> List[Tuple[str, str]]:
        return [(k, label) for k, label in series
                if k in history and np.isfinite(np.asarray(history[k], dtype=float)).any()]

    panels = [(title, cols, scale, ylabel) for title, cols, scale, ylabel in (
        ("Numerics counters (cumulative)", present(NUMERICS_COUNTER_SERIES), "symlog", "count since step 0"),
        ("Decode variance margin", present(DECODE_MARGIN_SERIES), "log", r"$\sigma^2_{\min}$"),
        ("Relational score health", present(SCORE_NONFINITE_SERIES), "linear", "fraction"),
    ) if cols]
    if not panels or step.size == 0:
        raise _figs.FigureSkipped("no numerics counter or decode-margin column")
    with _figs._compact_style():
        fig, axes = plt.subplots(1, len(panels), squeeze=False,
                                 figsize=(_figs.MAX_FIGURE_WIDTH_IN if len(panels) > 1
                                          else _figs.MAX_FIGURE_WIDTH_IN / 2.0, 3.6))
        for i, (title, cols, scale, ylabel) in enumerate(panels):
            ax = axes[0][i]
            for j, (key, label) in enumerate(cols):
                color = _figs._CB[j % len(_figs._CB)]
                values = np.asarray(history[key], dtype=float)
                keep = np.isfinite(values) & np.isfinite(step)
                if scale == "log":
                    keep &= values > 0
                if keep.any():
                    x, v = step[keep], values[keep]
                    last = f"{int(v[-1]):,}" if scale == "symlog" else f"{v[-1]:.3g}"
                    ax.plot(x, v, "o-" if v.size <= 50 else "-", ms=2.0, lw=1.1, color=color,
                            label=f"{label} (last {last})")
                if spikes is not None and key in spikes and spike_step.size:
                    sv = np.asarray(spikes[key], dtype=float)
                    sk = np.isfinite(sv) & np.isfinite(spike_step)
                    if scale == "log":
                        sk &= sv > 0
                    if sk.any():
                        ax.plot(spike_step[sk], sv[sk], ls="none", marker="x", ms=3.5, mew=0.9,
                                color=color, zorder=6)
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=1.0)
                peak = max((float(np.nanmax(np.asarray(history[key], dtype=float)))
                            for key, _ in cols), default=0.0)
                ax.set_ylim(0.0, max(10.0, 2.0 * peak))           # an all-zero counter sits on a 0..10 axis
            elif scale == "log":
                ax.set_yscale("log")
            ax.set(title=title, xlabel="training step", ylabel=ylabel)
            if step.size and _figs._set_finite_xlim(ax, step):
                _figs._step_xaxis(ax)
            # Below the axes: the counter labels are wider than a third of the figure.
            ax.legend(fontsize=6.0, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
        suptitle = "Numerics counters and decode margin"
        if spikes is not None and spike_step.size:
            suptitle += " (x: divergence-spike rows)"
        fig.suptitle(suptitle)
        fig.tight_layout()
        return _figs._save(fig, path)
