r"""Manuscript figures assembled from run directories: M3 (learning curves), S1 (numerics), S3 (scaling).

All figures are at most :data:`vfe3.viz.figures.MAX_FIGURE_WIDTH_IN` (7.2 in) wide with no text below
6 pt, drawn under the compact rc of :mod:`vfe3.viz.figures`, and written as PNG (300 dpi) and PDF. A
panel whose data a run does not record is left out and the reason is returned; nothing is drawn as a
placeholder.

* :func:`plot_learning_curves` (``fig:placeholder_learning_curves``): per arm and seed, held-out CE
  against training-token presentations (A) and device-hours (B), and the fixed-sample generalization
  gap (C). One curve per seed, colored by arm.
* :func:`plot_numerics` (``fig:placeholder_numerics``) for one run: the per-layer residual ratio (A),
  the map invariants over training (B, with an endpoint-only fallback), the stored log-variances with
  the curve implied by decoupled weight decay alone (C), and event counters with denominators (D).
* :func:`plot_scaling` (``fig:placeholder_scaling``): test CE against :math:`K` and against depth,
  mean :math:`\pm` SD over seeds, from the declared grids.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vfe3.paper.runs import (
    SCALING_AXIS_FIELD, Arm, Manifest, RunRecord, finite_float, optimizer_groups, select_checkpoint,
)

M3_LOG_X: bool = True                 # token and device-hour axes on a log scale (early evaluations)
DECAY_CURVE_POINTS: int = 2000        # samples of the decay-only curve drawn over training
LR_CROSSCHECK_RTOL: float = 1e-4      # reconstructed vs logged sigma learning rate, relative tolerance
COUNTER_SYMLOG_LINTHRESH: float = 1e-8
POSTHOC_RESIDUAL_GLOB: str = "residual_*.csv"   # vfe3.posthoc.residual endpoint tables in posthoc/csv/
SIGMA_TABLE_KEY: str = "prior_bank.sigma_log_embed"

# S1 panel D: (count column, denominator column, label, cadence) of the training-builder contract.
COUNTER_PAIRS: Tuple[Tuple[str, str, str, str], ...] = (
    ("cap_active_count", "cap_entry_count", "cost-bound activations", "per eval"),
    ("factorization_fallback_count", "factorization_count", "factorization fallbacks", "per eval"),
    ("frame_projection_count", "frame_projection_total", "frame-generator projections", "cumulative"),
    ("skipped_updates", "attempted_updates", "skipped updates", "cumulative"),
)


class FigureSkipped(RuntimeError):
    r"""A figure with no data to show; the message is the recorded reason."""


# ----------------------------------------------------------------------------------------------
# Style and output
# ----------------------------------------------------------------------------------------------

def _viz():
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    return figs


def save_figure(fig: Any, out_dir: Path, stem: str) -> List[Path]:
    r"""Write ``stem.png`` (300 dpi) and ``stem.pdf`` at the figure's own size (no tight bbox)."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(paths[0], dpi=300)
        fig.savefig(paths[1])
    plt.close(fig)
    return paths


def min_text_points(fig: Any) -> float:
    r"""Smallest font size (pt) of any visible text in ``fig`` (after a draw)."""
    from matplotlib.text import Text

    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return float(min(sizes)) if sizes else float("inf")


def _panel_label(ax: Any, label: str) -> None:
    ax.text(-0.02, 1.02, label, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


def _interp(step: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    r"""A monotone per-row column (``tokens_seen``, ``wall_clock_s``) evaluated at ``query`` steps."""
    keep = np.isfinite(step) & np.isfinite(values)
    if not keep.any():
        return np.full(query.shape, np.nan)
    xs, ys = step[keep], values[keep]
    order = np.argsort(xs)
    return np.interp(query, xs[order], ys[order], left=np.nan, right=np.nan)


# Line colors (the Wong palette without its yellow, which is illegible as a thin line on white) and
# line styles; arm i takes color i mod 7 and style i div 7.
ARM_COLORS: Tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
ARM_LINESTYLES: Tuple[str, ...] = ("-", "--", ":", "-.")


def arm_style(position: int) -> Tuple[str, str]:
    r"""``(color, linestyle)`` of the ``position``-th drawn arm (distinct for 28 arms)."""
    return (ARM_COLORS[position % len(ARM_COLORS)],
            ARM_LINESTYLES[(position // len(ARM_COLORS)) % len(ARM_LINESTYLES)])


# ----------------------------------------------------------------------------------------------
# M3: learning curves
# ----------------------------------------------------------------------------------------------

def learning_curve_series(run: RunRecord) -> Dict[str, Any]:
    r"""Held-out CE at every evaluation with its token and device-hour coordinates, and the fixed gap.

    Tokens: ``tokens_seen`` interpolated to the evaluation steps, else ``step x batch_size x
    max_seq_len`` from the saved config. Device-hours: ``wall_clock_s / 3600`` (the training loop's
    wall clock, one device), else ``summary.json wall_time_s`` spread linearly over the steps (an
    approximation that includes evaluation and artifact time; the source is recorded). Gap:
    ``generalization_gap_fixed``, else :math:`\mathrm{CE}_{\mathrm{val}} - \mathrm{CE}_{\mathrm{train\ sample}}`
    from ``val_ce`` and ``train_sample_ce`` on the rows carrying both. Absent inputs leave ``None``
    and a reason.
    """
    out: Dict[str, Any] = {"run_dir": str(run.run_dir), "reasons": {}}
    metrics = run.metrics()
    if metrics is None or "val_ce" not in metrics or "step" not in metrics:
        out["status"] = "skipped"
        out["reason"] = "metrics.csv absent or without val_ce"
        return out
    step, val = metrics["step"], metrics["val_ce"]
    rows = np.isfinite(step) & np.isfinite(val)
    if not rows.any():
        out["status"] = "skipped"
        out["reason"] = "metrics.csv has no finite val_ce"
        return out
    out["status"] = "ok"
    out["step"], out["val_ce"] = step[rows], val[rows]
    tokens_col = metrics.get("tokens_seen")
    logged_tokens = tokens_col is not None and bool(np.isfinite(tokens_col).any())

    def tokens_at(query: np.ndarray) -> Optional[np.ndarray]:
        if logged_tokens:
            return _interp(step, tokens_col, query)
        return query * float(run.tokens_per_step()) if run.tokens_per_step() else None

    out["tokens"] = tokens_at(step[rows])
    out["tokens_source"] = ("metrics.csv tokens_seen" if logged_tokens else
                            "step x batch_size x max_seq_len (config.json)" if out["tokens"] is not None else None)
    if out["tokens"] is None:
        out["reasons"]["tokens"] = "no tokens_seen column and no batch geometry in config.json"
    wall = metrics.get("wall_clock_s")
    wall_time = finite_float(run.summary.get("wall_time_s"))
    last = run.final_step()
    if wall is not None and np.isfinite(wall).any():
        out["hours"], out["hours_source"] = _interp(step, wall, step[rows]) / 3600.0, "metrics.csv wall_clock_s"
    elif wall_time is not None and last:
        out["hours"] = step[rows] / float(last) * wall_time / 3600.0
        out["hours_source"] = "summary.json wall_time_s, linear in step (approximation)"
    else:
        out["hours"], out["hours_source"] = None, None
        out["reasons"]["hours"] = "no wall_clock_s column and no summary.json wall_time_s"
    gap = metrics.get("generalization_gap_fixed")
    train = metrics.get("train_sample_ce")
    if gap is not None and np.isfinite(gap).any():
        grow = np.isfinite(step) & np.isfinite(gap)
        out["gap_step"], out["gap"], out["gap_source"] = step[grow], gap[grow], "metrics.csv generalization_gap_fixed"
    elif train is not None and np.isfinite(train + val).any():
        grow = np.isfinite(step) & np.isfinite(train) & np.isfinite(val)
        out["gap_step"], out["gap"] = step[grow], (val - train)[grow]
        out["gap_source"] = "metrics.csv val_ce - train_sample_ce"
    else:
        out["gap"] = None
        out["reasons"]["gap"] = "no generalization_gap_fixed or train_sample_ce column (runs before 2026-09-28)"
    if out["gap"] is not None:
        out["gap_tokens"] = tokens_at(out["gap_step"])
    return out
    step, val = metrics["step"], metrics["val_ce"]
    rows = np.isfinite(step) & np.isfinite(val)
    if not rows.any():
        out["status"] = "skipped"
        out["reason"] = "metrics.csv has no finite val_ce"
        return out
    out["status"] = "ok"
    out["step"], out["val_ce"] = step[rows], val[rows]
    tokens_col = metrics.get("tokens_seen")
    if tokens_col is not None and np.isfinite(tokens_col).any():
        out["tokens"], out["tokens_source"] = _interp(step, tokens_col, step[rows]), "metrics.csv tokens_seen"
    elif run.tokens_per_step():
        out["tokens"] = step[rows] * float(run.tokens_per_step())
        out["tokens_source"] = "step x batch_size x max_seq_len (config.json)"
    else:
        out["tokens"], out["tokens_source"] = None, None
        out["reasons"]["tokens"] = "no tokens_seen column and no batch geometry in config.json"
    wall = metrics.get("wall_clock_s")
    if wall is not None and np.isfinite(wall).any():
        out["hours"] = _interp(step, wall, step[rows]) / 3600.0
    else:
        out["hours"] = None
        out["reasons"]["hours"] = "no wall_clock_s column in metrics.csv"
    gap = metrics.get("generalization_gap_fixed")
    if gap is not None and np.isfinite(gap).any():
        grow = np.isfinite(step) & np.isfinite(gap)
        out["gap_step"], out["gap"] = step[grow], gap[grow]
        out["gap_tokens"] = (_interp(step, tokens_col, step[grow])
                             if out.get("tokens_source") == "metrics.csv tokens_seen"
                             else (step[grow] * float(run.tokens_per_step()) if run.tokens_per_step() else None))
    else:
        out["gap"] = None
        out["reasons"]["gap"] = "no generalization_gap_fixed column (runs before 2026-09-28)"
    return out


def plot_learning_curves(manifest: Manifest, out_dir: Path, *, stem: str = "m3_learning_curves") -> Dict[str, Any]:
    r"""Figure M3 for every declared arm and seed; returns ``{"paths", "panels", "runs", "skipped"}``."""
    figs = _viz()
    plt = figs.plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    series: List[Tuple[int, Arm, RunRecord, Dict[str, Any]]] = []
    record: Dict[str, Any] = {"runs": [], "skipped": {}}
    for index, arm in enumerate(manifest.arms):
        for run in arm.runs:
            data = learning_curve_series(run)
            record["runs"].append({"arm": arm.label, "run_dir": str(run.run_dir),
                                   "status": data["status"], "reason": data.get("reason"),
                                   "tokens_source": data.get("tokens_source"),
                                   "hours_source": data.get("hours_source"),
                                   "gap_source": data.get("gap_source"), "reasons": data["reasons"]})
            if data["status"] == "ok":
                series.append((index, arm, run, data))
    if not series:
        raise FigureSkipped("no declared run has a validation curve")
    panels = []
    if any(d.get("tokens") is not None for *_, d in series):
        panels.append("A")
    if any(d.get("hours") is not None for *_, d in series):
        panels.append("B")
    else:
        record["skipped"]["B"] = "no run records wall_clock_s"
    if any(d.get("gap") is not None and d.get("gap_tokens") is not None for *_, d in series):
        panels.append("C")
    else:
        record["skipped"]["C"] = "no run records generalization_gap_fixed"
    if "A" not in panels:
        record["skipped"]["A"] = "no run has a token axis"
    if not panels:
        raise FigureSkipped("no panel has data")
    token_sources = {d.get("tokens_source") for *_, d in series if d.get("tokens_source")}
    token_label = ("training tokens" if token_sources <= {"metrics.csv tokens_seen"}
                   else "training tokens (step x batch x context where not logged)")
    hour_sources = {d.get("hours_source") for *_, d in series if d.get("hours_source")}
    hours_label = ("device-hours (one device, wall clock)" if hour_sources <= {"metrics.csv wall_clock_s"}
                   else "device-hours (wall clock; linear in step where not logged)")
    drawn_arms = sorted({index for index, *_ in series})
    style_of = {index: arm_style(position) for position, index in enumerate(drawn_arms)}
    legend_rows = int(math.ceil(len(drawn_arms) / 3))
    with figs._compact_style():
        fig, axes = plt.subplots(1, len(panels), figsize=(figs.MAX_FIGURE_WIDTH_IN, 2.3 + 0.13 * legend_rows),
                                 squeeze=False, layout="constrained")
        axes = dict(zip(panels, axes[0]))
        for index, arm, run, data in series:
            color, linestyle = style_of[index]
            for panel, ax in axes.items():
                if panel == "A" and data.get("tokens") is not None:
                    x, y = data["tokens"], data["val_ce"]
                elif panel == "B" and data.get("hours") is not None:
                    x, y = data["hours"], data["val_ce"]
                elif panel == "C" and data.get("gap") is not None and data.get("gap_tokens") is not None:
                    x, y = data["gap_tokens"], data["gap"]
                else:
                    continue
                keep = np.isfinite(x) & np.isfinite(y) & ((x > 0) if M3_LOG_X else True)
                ax.plot(x[keep], y[keep], ls=linestyle, marker="o", ms=1.8, lw=0.9, color=color, alpha=0.9)
        for panel, ax in axes.items():
            formatter = figs._count_label if panel in ("A", "C") else (lambda v, _pos=None: f"{v:g}")
            if M3_LOG_X:
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(FuncFormatter(formatter))
            ax.set_xlabel({"A": token_label, "B": hours_label, "C": token_label}[panel])
            ax.set_ylabel(r"$\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train\ sample}}$ (nats)"
                          if panel == "C" else "held-out CE (nats/token)")
            ax.set_title({"A": "Held-out CE vs tokens", "B": "Held-out CE vs device-hours",
                          "C": "Generalization gap (fixed sample)"}[panel])
            if panel == "C":
                ax.axhline(0.0, color="#444444", lw=0.6, ls="--")
            _panel_label(ax, panel)
        handles = []
        for index in drawn_arms:
            n_seeds = sum(1 for s in series if s[0] == index)
            color, linestyle = style_of[index]
            handles.append(Line2D([], [], color=color, ls=linestyle, lw=1.2,
                                  label=f"{manifest.arms[index].label} ({n_seeds} seed{'s' if n_seeds != 1 else ''})"))
        fig.legend(handles=handles, loc="outside lower center", ncol=min(3, len(handles)), frameon=False)
        record["min_text_pt"] = min_text_points(fig)
        record["paths"] = [str(p) for p in save_figure(fig, out_dir, stem)]
    record["panels"] = panels
    if M3_LOG_X:
        record["note"] = "log x axes: evaluations at zero tokens or hours (step 0) are not drawn"
    return record


# ----------------------------------------------------------------------------------------------
# S1: numerics of one run
# ----------------------------------------------------------------------------------------------

def decay_only_log_variance(
    s0_mean:        float,
    learning_rates: np.ndarray,          # (T,) eta_k of accepted update k = 0..T-1
    weight_decay:   float,
) -> np.ndarray:
    r"""Mean stored log-variance implied by decoupled weight decay alone, after ``k = 0..T`` updates.

    The stored table is :math:`s = \log\sigma^2` (``prior_bank.sigma_log_embed``,
    :math:`\Sigma_v = \operatorname{diag}(e^{s_v})`). AdamW's decoupled decay multiplies every entry
    by :math:`1 - \eta_k\lambda` before the adaptive step, so with the adaptive step removed

    .. math::
        s_{k+1} = (1 - \eta_k \lambda)\, s_k, \qquad
        \bar s_k = \bar s_0 \prod_{j<k} (1 - \eta_j \lambda)
                 = \bar s_0 \exp\Big(\sum_{j<k} \log(1 - \eta_j\lambda)\Big),

    which pulls :math:`s` toward 0 (unit variance) from either sign. Linear maps commute with the
    mean, so the same factor carries the mean; the spread under decay alone is the initial spread
    times the same factor (0 for the constant initialization :math:`s_0 = \log\sigma_{\mathrm{init}}`).
    Returns the (T+1,) curve :math:`\bar s_0, \ldots, \bar s_T`.
    """
    eta = np.asarray(learning_rates, dtype=np.float64)
    factors = 1.0 - eta * float(weight_decay)
    if np.any(factors <= 0.0):
        raise ValueError("eta * weight_decay >= 1: the decay factor is not a contraction")
    log_prod = np.concatenate([[0.0], np.cumsum(np.log(factors))])
    return float(s0_mean) * np.exp(log_prod)


def sigma_schedule(run: RunRecord, n_updates: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    r"""Learning rates :math:`\eta_k` (k = 0..T-1) and decay :math:`\lambda` of the log-variance table.

    Base learning rate and weight decay: the executed optimizer group holding
    ``prior_bank.sigma_log_embed`` in the latest training checkpoint (``initial_lr``,
    ``weight_decay``), else the config (``m_p_sigma_lr``; ``sigma_weight_decay``, inheriting
    ``weight_decay`` when None, as :func:`vfe3.train.build_optimizer` groups it). Schedule: the
    per-update multiplier of :func:`vfe3.train._floor_lr_lambdas` (warmup, half cosine, floors) at
    the accepted-update count :math:`k` (the scheduler steps once per accepted update). The logged
    ``lr_sigma`` column (the rate of the next update after ``step`` updates) is used to check the
    reconstruction; on a mismatch beyond :data:`LR_CROSSCHECK_RTOL`, or when the schedule cannot be
    rebuilt, the logged rates are interpolated instead.
    """
    info: Dict[str, Any] = {}
    base = lam = None
    groups = optimizer_groups(run)
    if groups is not None:
        for group in groups["groups"]:
            if SIGMA_TABLE_KEY in group["names"]:
                base, lam = group["initial_lr"], group["weight_decay"]
                info["hyperparameter_source"] = f"optimizer group in {Path(groups['checkpoint']).name}"
                break
    if base is None or lam is None:
        base = finite_float(run.value("m_p_sigma_lr"))
        wd = run.value("sigma_weight_decay")
        lam = finite_float(wd if wd is not None else run.value("weight_decay"))
        info["hyperparameter_source"] = "config.json (m_p_sigma_lr; sigma_weight_decay, else weight_decay)"
    info["base_lr"], info["weight_decay"] = base, lam
    if base is None or lam is None or n_updates <= 0:
        info["reason"] = "sigma learning rate or weight decay not recorded"
        return None, info
    k = np.arange(int(n_updates))
    lrs = None
    multiplier: Optional[Callable[[int], float]] = None
    try:
        from vfe3.train import _floor_lr_lambdas
        cfg_like = run.cfg if run.cfg is not None else SimpleNamespace(
            **{key: run.raw(key) for key in ("warmup_steps", "max_steps", "min_lr_frac")})
        multiplier = _floor_lr_lambdas([base], cfg_like)[0]
        lrs = np.array([base * multiplier(int(i)) for i in k], dtype=np.float64)
        info["lr_source"] = "reconstructed schedule (vfe3.train._floor_lr_lambdas)"
    except Exception as exc:                                   # noqa: BLE001 - fall back to the log
        info["reconstruction_error"] = repr(exc)
    metrics = run.metrics()
    logged = metrics.get("lr_sigma") if metrics is not None else None
    if logged is not None and "step" in metrics:
        rows = np.isfinite(metrics["step"]) & np.isfinite(logged)
        steps_logged, lr_logged = metrics["step"][rows], logged[rows]
        if steps_logged.size and multiplier is not None:
            predicted = np.array([base * multiplier(int(s)) for s in steps_logged])
            scale = np.maximum(np.abs(lr_logged), 1e-300)
            info["lr_crosscheck_max_rel_error"] = float(np.max(np.abs(predicted - lr_logged) / scale))
            if info["lr_crosscheck_max_rel_error"] > LR_CROSSCHECK_RTOL:
                lrs = None
                info["lr_crosscheck"] = "mismatch: logged lr_sigma used"
            else:
                info["lr_crosscheck"] = "ok"
        if lrs is None and steps_logged.size >= 2:
            order = np.argsort(steps_logged)
            lrs = np.interp(k, steps_logged[order], lr_logged[order])
            info["lr_source"] = "metrics.csv lr_sigma interpolated between log rows"
    if lrs is None:
        info["reason"] = "the sigma learning-rate schedule could not be rebuilt or read"
    return lrs, info


def _checkpoint_sigma_stats(path: Path) -> Optional[Tuple[float, float]]:
    r"""Population mean and SD of the stored log-variance table of a checkpoint (mmap, float64)."""
    if not path.is_file():
        return None
    from vfe3.posthoc.maps import load_map_state
    state, _ = load_map_state(path)
    table = next((value for name, value in state.items()
                  if name.endswith(SIGMA_TABLE_KEY)), None)
    if table is None:
        return None
    values = table.detach().double()
    return float(values.mean()), float(values.std(unbiased=False))


def _residual_panel(run: RunRecord) -> Dict[str, Any]:
    figs = _viz()
    table = figs.read_long_table(run.run_dir / "diagnostics" / "residual_ratio.csv")
    needed = ("step", "layer", "q10", "q50", "q90")
    if table is not None and all(c in table for c in needed) and np.isfinite(table["q50"]).any():
        return {"status": "ok", "kind": "curves", "table": table,
                "source": "diagnostics/residual_ratio.csv"}
    parts, sources = [], []
    for path in sorted((run.run_dir / "posthoc" / "csv").glob(POSTHOC_RESIDUAL_GLOB)):
        endpoint = figs.read_long_table(path)
        if endpoint is None:
            continue
        endpoint.update({q: endpoint[f"rho_{q}"] for q in ("q10", "q50", "q90")   # vfe3.posthoc.residual
                         if q not in endpoint and f"rho_{q}" in endpoint})         # names them rho_q*
        if "layer" not in endpoint or "q50" not in endpoint:
            continue
        if "step" not in endpoint:
            step = run.summary.get("best_step") if "best" in path.stem else run.final_step()
            endpoint["step"] = np.full(endpoint["layer"].shape, float(step) if step else np.nan)
        parts.append(endpoint)
        sources.append(f"posthoc/csv/{path.name}")
    if parts:
        keys = ("step", "layer", "q10", "q50", "q90")
        table = {key: np.concatenate([part.get(key, part["q50"]) for part in parts]) for key in keys}
        return {"status": "ok", "kind": "endpoint", "table": table, "source": ", ".join(sources)}
    return {"status": "skipped",
            "reason": "no diagnostics/residual_ratio.csv (runs before 2026-09-28) and no post-hoc "
                      "residual-ratio table"}


def _map_rows_by_layer(rows: Mapping[str, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    r"""Per layer and step: head medians of the invariants, head extrema of the gains."""
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for layer in sorted({int(v) for v in rows["layer"] if math.isfinite(v)}):
        mask = rows["layer"] == layer
        steps = sorted({float(v) for v in rows["step"][mask] if math.isfinite(v)})
        entry: Dict[str, List[float]] = {k: [] for k in ("step", "scale_rms", "M_sigma_max", "gain_median",
                                                         "gain_min", "gain_max", "angle_median_deg")}
        for step in steps:
            sel = mask & (rows["step"] == step)
            entry["step"].append(step)
            for key in ("scale_rms", "M_sigma_max", "gain_median", "angle_median_deg"):
                entry[key].append(float(np.nanmedian(rows[key][sel])) if key in rows and np.isfinite(rows[key][sel]).any() else np.nan)
            entry["gain_min"].append(float(np.nanmin(rows["gain_min"][sel])) if "gain_min" in rows else np.nan)
            entry["gain_max"].append(float(np.nanmax(rows["gain_max"][sel])) if "gain_max" in rows else np.nan)
        out[layer] = {k: np.asarray(v, dtype=np.float64) for k, v in entry.items()}
    return out


def _rows_to_columns(rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    keys = sorted({k for row in rows for k in row})
    return {k: np.array([float(row.get(k, np.nan)) for row in rows], dtype=np.float64) for k in keys}


def _maps_panel(run: RunRecord) -> Dict[str, Any]:
    mode = run.value("attention_mode")
    if mode not in ("projected_predictive", "projected_gradient"):
        return {"status": "skipped",
                "reason": f"attention_mode={mode!r} is a retired architecture"}
    from vfe3.posthoc import maps as mp
    cfg_like = run.cfg_like()
    snapshots = run.run_dir / "map_snapshots"
    try:
        if mp.snapshot_files(snapshots):
            rows = mp.snapshot_curves(snapshots, cfg_like)
            if rows:
                return {"status": "ok", "kind": "curves", "layers": _map_rows_by_layer(_rows_to_columns(rows)),
                        "source": "map_snapshots/ (vfe3.posthoc.maps.snapshot_curves)"}
    except Exception as exc:                                   # noqa: BLE001 - fall back to endpoints
        run.notes.append(f"map snapshot curves failed: {exc!r}")
    figs = _viz()
    endpoint_rows: List[Dict[str, Any]] = []
    sources = []
    final_step = run.final_step()
    for checkpoint, stem, step in (("final_model.pt", "final", final_step),
                                   ("best_model.pt", "best", run.summary.get("best_step"))):
        if not isinstance(step, int) or (stem == "best" and step == final_step):
            continue                                           # best == final: one endpoint
        csv_path = run.run_dir / "posthoc" / "csv" / f"map_invariants_{stem}.csv"
        table = figs.read_long_table(csv_path)
        if table is not None and "layer" in table:
            n = table["layer"].shape[0]
            endpoint_rows += [{**{k: table[k][i] for k in table}, "step": step} for i in range(n)]
            sources.append(f"posthoc/csv/{csv_path.name}")
            continue
        if (run.run_dir / checkpoint).is_file():
            try:
                state, _ = mp.load_map_state(run.run_dir / checkpoint)
                rows = mp.map_invariants(state, cfg_like)
                endpoint_rows += [{**row, "step": step} for row in rows]
                sources.append(f"{checkpoint} (vfe3.posthoc.maps.map_invariants)")
            except Exception as exc:                           # noqa: BLE001
                run.notes.append(f"map invariants of {checkpoint} failed: {exc!r}")
    if not endpoint_rows:
        return {"status": "skipped", "reason": "no map snapshots, post-hoc map tables or readable checkpoints"}
    return {"status": "ok", "kind": "endpoint", "layers": _map_rows_by_layer(_rows_to_columns(endpoint_rows)),
            "source": "; ".join(sources)}


def _sigma_panel(run: RunRecord) -> Dict[str, Any]:
    metrics = run.metrics() or {}
    out: Dict[str, Any] = {"status": "ok"}
    mean, sd, step = metrics.get("sigma_log_mean"), metrics.get("sigma_log_sd"), metrics.get("step")
    if mean is not None and step is not None and np.isfinite(mean).any():
        rows = np.isfinite(step) & np.isfinite(mean)
        out.update(kind="curves", step=step[rows], mean=mean[rows],
                   sd=(sd[rows] if sd is not None else np.full(rows.sum(), np.nan)),
                   source="metrics.csv sigma_log_mean, sigma_log_sd")
    else:
        points = []
        final_step = run.final_step()
        for checkpoint, step_value in (("final_model.pt", final_step),
                                       ("best_model.pt", run.summary.get("best_step"))):
            if not isinstance(step_value, int) or (checkpoint == "best_model.pt" and step_value == final_step):
                continue
            try:
                stats = _checkpoint_sigma_stats(run.run_dir / checkpoint)
            except Exception as exc:                           # noqa: BLE001
                run.notes.append(f"log-variance table of {checkpoint} unreadable: {exc!r}")
                stats = None
            if stats is not None:
                points.append((step_value, stats[0], stats[1], checkpoint))
        if not points:
            return {"status": "skipped",
                    "reason": "no sigma_log_mean column and no readable log-variance table "
                              f"({SIGMA_TABLE_KEY}) in the saved weights"}
        out.update(kind="endpoint", step=np.array([p[0] for p in points], float),
                   mean=np.array([p[1] for p in points]), sd=np.array([p[2] for p in points]),
                   source=", ".join(p[3] for p in points) + " (population mean and SD)")
    # Decay-only reference: s0 from the step-0 row when logged, else log(sigma_init) (constant init).
    s0, s0_source = None, None
    if out["kind"] == "curves" and out["step"].size and out["step"][0] == 0:
        s0, s0_source = float(out["mean"][0]), "metrics.csv sigma_log_mean at step 0"
    elif finite_float(run.raw("sigma_init")) and finite_float(run.raw("sigma_init")) > 0:
        s0, s0_source = math.log(float(run.raw("sigma_init"))), "log(sigma_init) from config.json"
    accepted = run.summary.get("accepted_updates")
    n_updates = int(accepted) if isinstance(accepted, int) else (run.final_step() or 0)
    out["decay_only"] = {"s0_mean": s0, "s0_source": s0_source, "n_updates": n_updates}
    if s0 is None:
        out["decay_only"]["reason"] = "initial log-variance not recorded"
        return out
    lrs, info = sigma_schedule(run, n_updates)
    out["decay_only"].update(info)
    if lrs is not None and info.get("weight_decay") is not None:
        curve = decay_only_log_variance(s0, lrs, info["weight_decay"])
        out["decay_only"]["curve"] = curve
        out["decay_only"]["final_mean"] = float(curve[-1])
        skipped = run.summary.get("skipped_steps")
        if isinstance(skipped, int) and skipped:
            out["decay_only"]["note"] = (f"{skipped} skipped updates: the curve is indexed by accepted "
                                         "updates and drawn against the step")
    return out


def _counter_panel(run: RunRecord) -> Dict[str, Any]:
    metrics = run.metrics() or {}
    step = metrics.get("step")
    series, absent = [], []
    for count_col, denom_col, label, cadence in COUNTER_PAIRS:
        count, denom = metrics.get(count_col), metrics.get(denom_col)
        if step is not None and count is not None and denom is not None:
            rows = np.isfinite(step) & np.isfinite(count) & np.isfinite(denom) & (denom > 0)
            if rows.any():
                series.append({"label": label, "cadence": cadence, "step": step[rows],
                               "count": count[rows], "denominator": denom[rows],
                               "source": f"metrics.csv {count_col}/{denom_col}"})
                continue
        absent.append(f"{count_col}/{denom_col}")
    endpoints = []
    if "metrics.csv skipped_updates/attempted_updates" not in {s["source"] for s in series}:
        skipped, attempted = run.summary.get("skipped_steps"), run.summary.get("attempted_steps")
        if isinstance(skipped, int) and isinstance(attempted, int) and attempted > 0:
            endpoints.append({"label": "skipped updates", "step": run.final_step(), "count": skipped,
                              "denominator": attempted, "source": "summary.json skipped_steps/attempted_steps"})
    if "metrics.csv cap_active_count/cap_entry_count" not in {s["source"] for s in series}:
        klcap = (((run.posthoc or {}).get("klcap") or {}).get("validation") or {}).get("summary") or {}
        scored = [row for row in klcap.get("per_layer") or [] if row.get("population") == "scored"]
        if scored:
            endpoints.append({"label": "cost-bound activations (post hoc, validation)",
                              "step": run.final_step(),
                              "count": int(sum(int(r.get("n_capped", 0)) for r in scored)),
                              "denominator": int(sum(int(r.get("n_entries", 0)) for r in scored)),
                              "source": "posthoc.json klcap validation (scored population)"})
    if not series and not endpoints:
        return {"status": "skipped", "reason": "no counter columns (" + ", ".join(absent) + ") and no "
                                               "recorded endpoint counts"}
    return {"status": "ok", "series": series, "endpoints": endpoints, "absent": absent}


def numerics_data(run: RunRecord) -> Dict[str, Any]:
    r"""The four S1 panels of one run (each ``{"status": "ok", ...}`` or ``{"status": "skipped", "reason"}``)."""
    panels: Dict[str, Any] = {}
    for key, builder in (("A", _residual_panel), ("B", _maps_panel), ("C", _sigma_panel), ("D", _counter_panel)):
        try:
            panels[key] = builder(run)
        except Exception as exc:                               # noqa: BLE001 - one panel never aborts S1
            panels[key] = {"status": "skipped", "reason": f"failed: {type(exc).__name__}: {exc}"}
    return panels


def _step_axis(ax: Any, figs: Any, last_step: Optional[int] = None) -> None:
    r"""Training-step x axis; with ``last_step`` every S1 panel spans the same range [0, last step]."""
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(figs._count_label))
    ax.set_xlabel("training step")
    if last_step:
        ax.set_xlim(-0.03 * last_step, 1.03 * last_step)


def plot_numerics(run: RunRecord, out_dir: Path, *, stem: str = "s1_numerics") -> Dict[str, Any]:
    r"""Figure S1 for one run; returns the panel records (without arrays), paths and skip reasons."""
    figs = _viz()
    plt = figs.plt
    data = numerics_data(run)
    last_step = run.final_step()
    draw: List[Tuple[str, Callable[[Any], None]]] = []
    cmap = plt.get_cmap("viridis")

    def layer_color(layer: int, n: int) -> Any:
        return cmap(0.1 + 0.8 * layer / max(1, n - 1))

    a = data["A"]
    if a["status"] == "ok":
        def draw_a(ax: Any) -> None:
            table = a["table"]
            layers = sorted({int(v) for v in table["layer"] if math.isfinite(v)})
            for layer in layers:
                sel = (table["layer"] == layer) & np.isfinite(table["q50"])
                order = np.argsort(table["step"][sel])
                x = table["step"][sel][order]
                color = layer_color(layer, len(layers))
                if a["kind"] == "curves":
                    ax.plot(x, table["q50"][sel][order], lw=1.0, color=color, label=f"layer {layer}")
                    if "q10" in table and "q90" in table:
                        ax.fill_between(x, table["q10"][sel][order], table["q90"][sel][order],
                                        color=color, alpha=0.15, lw=0)
                else:
                    lo = table["q50"][sel][order] - table.get("q10", table["q50"])[sel][order]
                    hi = table.get("q90", table["q50"])[sel][order] - table["q50"][sel][order]
                    ax.errorbar(x, table["q50"][sel][order], yerr=[lo, hi], fmt="o", ms=3, color=color,
                                label=f"layer {layer}", capsize=1.5, lw=0.8)
            ax.set_yscale("log")
            ax.set_ylabel(r"$\|m_i\|/\|\mu_i\|$ (median, 10-90%)")
            ax.set_title("Message-to-state ratio" + (" (endpoint)" if a["kind"] == "endpoint" else ""))
            _step_axis(ax, figs, last_step)
            ax.legend(frameon=False, ncol=2)
        draw.append(("A", draw_a))

    b = data["B"]
    if b["status"] == "ok":
        layers = b["layers"]
        endpoint = b["kind"] == "endpoint"

        def style(ls: str = "-", marker: str = "o") -> Dict[str, Any]:
            return (dict(marker=marker, ms=3.5, ls="none") if endpoint
                    else dict(marker=marker, ms=1.8, lw=0.9, ls=ls))

        def draw_b1(ax: Any) -> None:
            for layer, entry in layers.items():
                color = layer_color(layer, len(layers))
                ax.plot(entry["step"], entry["scale_rms"], color=color, label=f"layer {layer}", **style())
                ax.plot(entry["step"], entry["M_sigma_max"], color=color, **style("--", "s"))
            ax.set_yscale("log")
            ax.set_ylabel(r"common scale (o), $\sigma_{\max}(O_hT_h)$ (s)")
            ax.set_title("Map scales (head median)")
            _step_axis(ax, figs, last_step)
            ax.legend(frameon=False, ncol=2, loc="center left")

        def draw_b2(ax: Any) -> None:
            for layer, entry in layers.items():
                color = layer_color(layer, len(layers))
                ax.plot(entry["step"], entry["gain_median"], color=color, **style())
                if b["kind"] == "curves":
                    ax.fill_between(entry["step"], entry["gain_min"], entry["gain_max"], color=color,
                                    alpha=0.12, lw=0)
                else:
                    ax.vlines(entry["step"], entry["gain_min"], entry["gain_max"], color=color, lw=0.8)
            ax.set_yscale("log")
            ax.set_ylabel(r"relative gain of $S_{Q,h},S_{K,h}$")
            ax.set_title("Relative gains (median, range)")
            _step_axis(ax, figs, last_step)

        def draw_b3(ax: Any) -> None:
            for layer, entry in layers.items():
                ax.plot(entry["step"], entry["angle_median_deg"], color=layer_color(layer, len(layers)),
                        **style())
            ax.set_ylabel("principal angle (degrees)")
            ax.set_title("Principal angles (head median)")
            _step_axis(ax, figs, last_step)
        draw += [("B", draw_b1), ("", draw_b2), ("", draw_b3)]

    c = data["C"]
    if c["status"] == "ok":
        def draw_c(ax: Any) -> None:
            if c["kind"] == "curves":
                ax.plot(c["step"], c["mean"], color=figs._CB[1], lw=1.1, label="mean (stored)")
                if np.isfinite(c["sd"]).any():
                    ax.fill_between(c["step"], c["mean"] - c["sd"], c["mean"] + c["sd"],
                                    color=figs._CB[1], alpha=0.18, lw=0, label=r"$\pm$ SD")
            else:
                ax.errorbar(c["step"], c["mean"], yerr=c["sd"], fmt="o", ms=3.5, color=figs._CB[1],
                            capsize=2, lw=0.8, label=r"mean $\pm$ SD (saved weights)")
            decay = c.get("decay_only") or {}
            if decay.get("curve") is not None:
                curve = decay["curve"]
                idx = np.unique(np.linspace(0, curve.size - 1, min(curve.size, DECAY_CURVE_POINTS)).astype(int))
                ax.plot(idx, curve[idx], color="#444444", ls="--", lw=1.0,
                        label=rf"weight decay alone ($\lambda={decay['weight_decay']:g}$)")
            ax.axhline(0.0, color="#999999", lw=0.5, ls=":")
            ax.set_ylabel(r"stored $\log\sigma^2$")
            ax.set_title("Stored log-variances")
            _step_axis(ax, figs, last_step)
            ax.legend(frameon=False, loc="center right")
        draw.append(("C", draw_c))

    d = data["D"]
    if d["status"] == "ok":
        def draw_d(ax: Any) -> None:
            for i, s in enumerate(d["series"]):
                frac = s["count"] / s["denominator"]
                label = (f"{s['label']} ({s['cadence']})\n{int(s['count'][-1]):,}/"
                         f"{int(s['denominator'][-1]):,}")
                ax.plot(s["step"], frac, marker="o", ms=1.8, lw=0.9, color=figs._CB[i % len(figs._CB)],
                        label=label)
            for j, e in enumerate(d["endpoints"]):
                if e["step"] is None:
                    continue
                ax.plot([e["step"]], [e["count"] / e["denominator"]], ls="none", marker="D", ms=4,
                        color=figs._CB[(len(d["series"]) + j) % len(figs._CB)],
                        label=f"{e['label']}\n{e['count']:,}/{e['denominator']:,}")
            ax.set_yscale("symlog", linthresh=COUNTER_SYMLOG_LINTHRESH)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0])
            ax.set_ylabel("count / denominator")
            ax.set_title("Event counters")
            _step_axis(ax, figs, last_step)
            ax.legend(frameon=False, loc="upper left")
        draw.append(("D", draw_d))

    record: Dict[str, Any] = {"run_dir": str(run.run_dir), "panels": _strip_arrays(data)}
    if not draw:
        raise FigureSkipped("no S1 panel has data: " + "; ".join(
            f"{k}: {v.get('reason')}" for k, v in data.items()))
    ncols = 3 if len(draw) > 4 else 2
    nrows = int(math.ceil(len(draw) / ncols))
    with figs._compact_style():
        fig = plt.figure(figsize=(figs.MAX_FIGURE_WIDTH_IN, 2.3 * nrows), layout="constrained")
        grid = fig.add_gridspec(nrows, ncols)
        for position, (label, painter) in enumerate(draw):
            row, col = divmod(position, ncols)
            last = position == len(draw) - 1
            ax = fig.add_subplot(grid[row, col:] if last else grid[row, col])   # the last panel fills its row
            painter(ax)
            if label:
                _panel_label(ax, label)
        record["min_text_pt"] = min_text_points(fig)
        record["paths"] = [str(p) for p in save_figure(fig, out_dir, stem)]
    return record


def _strip_arrays(value: Any) -> Any:
    r"""JSON-ready copy: arrays become summary tuples (first, last, size), tables are dropped."""
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)] if value.dtype.kind == "f" else value
        return {"n": int(value.size), "first": float(finite[0]) if finite.size else None,
                "last": float(finite[-1]) if finite.size else None}
    if isinstance(value, Mapping):
        return {str(k): _strip_arrays(v) for k, v in value.items() if k not in ("table",)}
    if isinstance(value, (list, tuple)):
        return [_strip_arrays(v) for v in value]
    return value


# ----------------------------------------------------------------------------------------------
# S3: scaling
# ----------------------------------------------------------------------------------------------

def scaling_points(grid: Arm, axis: str, rule: str) -> Dict[str, Any]:
    r"""Test CE by grid value of ``axis`` (``embed_dim`` or ``n_layers`` from each run's config)."""
    field = SCALING_AXIS_FIELD[axis]
    by_x: Dict[float, List[Tuple[float, str]]] = {}
    missing = []
    for run in grid.runs:
        x = finite_float(run.value(field))
        sel = select_checkpoint(run, rule)
        if x is None or sel["status"] != "ok":
            missing.append({"run_dir": str(run.run_dir), "reason": sel.get("reason") or f"no {field}"})
            continue
        by_x.setdefault(x, []).append((sel["test_ce"], run.name))
    points = []
    for x in sorted(by_x):
        values = [v for v, _ in by_x[x]]
        mean = sum(values) / len(values)
        sd = (math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else None)
        points.append({"x": x, "n": len(values), "mean": mean, "sd": sd, "values": values,
                       "runs": [name for _, name in by_x[x]]})
    other = "n_layers" if axis == "K" else "embed_dim"
    fixed = sorted({run.value(other) for run in grid.runs if run.value(other) is not None}, key=str)
    return {"axis": axis, "field": field, "points": points, "missing": missing, "fixed": {other: fixed},
            "homogeneity": grid.homogeneity}


def plot_scaling(manifest: Manifest, out_dir: Path, *, stem: str = "s3_scaling") -> Dict[str, Any]:
    r"""Figure S3 from the declared grids (panel A: :math:`K`, panel B: depth)."""
    record: Dict[str, Any] = {"panels": {}, "skipped": {}}
    grids = []
    for panel, axis in (("A", "K"), ("B", "depth")):
        grid = manifest.scaling.get(axis)
        if grid is None or not grid.runs:
            record["skipped"][panel] = f"no {axis} grid declared"
            continue
        points = scaling_points(grid, axis, manifest.selection)
        record["panels"][panel] = points
        if points["points"]:
            grids.append((panel, points))
        else:
            record["skipped"][panel] = f"no run of the {axis} grid has a selected test CE"
    if not grids:
        raise FigureSkipped("no scaling grid has data: " + "; ".join(record["skipped"].values()))
    figs = _viz()
    plt = figs.plt
    with figs._compact_style():
        fig, axes = plt.subplots(1, len(grids), figsize=(min(figs.MAX_FIGURE_WIDTH_IN, 3.6 * len(grids)), 2.5),
                                 squeeze=False, layout="constrained")
        for ax, (panel, points) in zip(axes[0], grids):
            xs = np.array([p["x"] for p in points["points"]])
            means = np.array([p["mean"] for p in points["points"]])
            sds = np.array([p["sd"] if p["sd"] is not None else np.nan for p in points["points"]])
            for p in points["points"]:
                ax.plot([p["x"]] * p["n"], p["values"], ls="none", marker="o", ms=2.5, color=figs._CB[5],
                        alpha=0.8)
            ax.errorbar(xs, means, yerr=np.where(np.isfinite(sds), sds, 0.0), fmt="-s", ms=3.5,
                        color=figs._CB[0], capsize=2, lw=1.0, label=r"mean $\pm$ SD over seeds")
            for p in points["points"]:
                ax.annotate(f"n={p['n']}", (p["x"], p["mean"]), textcoords="offset points", xytext=(3, 4),
                            fontsize=6)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{int(x)}" for x in xs])
            fixed = points["fixed"]
            symbol = {"n_layers": "$L$", "embed_dim": "$K$"}
            fixed_text = ", ".join(f"{symbol.get(k, k)}={'/'.join(map(str, v))}" for k, v in fixed.items())
            ax.set_xlabel("state dimension $K$" if points["axis"] == "K" else "depth (layers)")
            ax.set_ylabel("test CE (nats/token)")
            ax.set_title(("Test CE vs $K$" if points["axis"] == "K" else "Test CE vs depth") + f" ({fixed_text})")
            if points["homogeneity"].get("violations"):
                record.setdefault("warnings", {})[panel] = (
                    f"{points['axis']} grid differs in fields other than the varied ones: "
                    + ", ".join(v["field"] for v in points["homogeneity"]["violations"]))
            ax.legend(frameon=False)
            _panel_label(ax, panel)
        record["min_text_pt"] = min_text_points(fig)
        record["paths"] = [str(p) for p in save_figure(fig, out_dir, stem)]
    return record
