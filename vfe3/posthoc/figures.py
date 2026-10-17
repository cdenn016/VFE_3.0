r"""Publication figures of the post-hoc analyses (7.2 in maximum width, text at least 6 pt).

* :func:`plot_head_block_norms`: S2 panel A, per layer the :math:`H \times B` block-energy shares of
  :math:`S_{Q,h}, S_{K,h}, T_h, O_h` (share of :math:`\|M_h\|_F^2` on each frame block's coordinates).
* :func:`plot_map_invariants`: S1 panel B endpoints, layer-by-head maps of the common scale, the
  relative-gain spread, the median principal angle and :math:`\sigma_{\max}(O_h T_h)`.
* :func:`plot_map_invariant_curves`: the same quantities over training from map snapshots, with the
  similarity-invariant floor :math:`\rho(T_h O_h)` of :math:`\sigma_{\max}`.
* :func:`plot_link_spectrum`: per layer, the eigenvalues of :math:`T_h O_h` (the nonzero spectrum of the
  content map :math:`M_h = O_h T_h`) in the complex plane, with the unit circle and the flat-composition
  point :math:`1`.
* :func:`plot_attention_stats`: S2 panel B, layer-by-head normalized entropy
  :math:`H(\beta_{i\cdot})/\log|\mathcal S_i|`, mean distance :math:`\sum_j \beta_{ij}|i-j|`, self-weight
  :math:`\beta_{ii}` and overlap with the prior-only routing :math:`\sum_j \min(\beta_{ij}, \pi_{ij})`.
* :func:`plot_interventions`: the M5 forest plot of :math:`\Delta\mathrm{CE}` with bootstrap intervals.
* :func:`plot_residual_ratio`: S1 panel A endpoints and v2:899, per layer the quantiles of
  :math:`\|m_i\|/\|\mu_i\|`, the relative LayerNorm change and the self-weights.
* :func:`plot_covariance_gap`: v2:872, the routing-covariance gaps and the hidden temperature.
* :func:`plot_ce_by_context`: cross-entropy against context length from the per-target dumps.

Every function writes ``<name>.png`` (300 dpi) and ``<name>.pdf`` into ``out_dir`` and returns the paths.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

MAX_WIDTH_IN: float = 7.2
MIN_FONT_PT: float = 6.0
MAP_NAMES = ("S_Q", "S_K", "T", "O")
MAP_LABELS = {"S_Q": r"$S_{Q,h}$", "S_K": r"$S_{K,h}$", "T": r"$T_h$", "O": r"$O_h$"}


def _style() -> None:
    import matplotlib.pyplot as plt
    from vfe3.semantics.common import apply_style

    apply_style()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5})


def _save(fig: Any, out_dir: Path, name: str) -> List[Path]:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    try:
        for suffix in (".png", ".pdf"):
            target = out_dir / f"{name}{suffix}"
            temporary = target.with_name(f".{target.stem}.tmp{suffix}")
            fig.savefig(str(temporary), dpi=300, bbox_inches="tight")
            os.replace(temporary, target)
            written.append(target)
    finally:
        plt.close(fig)
    return written


def _title(ax: Any, letter: str, label: str) -> None:
    r"""Left-aligned panel title led by the bold panel letter."""
    ax.set_title(r"$\mathbf{" + letter + "}$  " + label, loc="left")


def _grid(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    r"""(L, H) array of ``key`` from (layer, head) rows (NaN where missing)."""
    layers = sorted({int(r["layer"]) for r in rows})
    heads = sorted({int(r["head"]) for r in rows})
    out = np.full((len(layers), len(heads)), np.nan)
    for r in rows:
        value = r.get(key)
        if value is None:
            continue
        out[layers.index(int(r["layer"])), heads.index(int(r["head"]))] = float(value)
    return out


def _heatmap(ax: Any, fig: Any, values: np.ndarray, letter: str, title: str, *, cmap: str = "viridis",
             vmin: Optional[float] = None, vmax: Optional[float] = None, fmt: str = "{:.2f}",
             annotate: bool = True, xlabel: str = "head", ylabel: str = "layer") -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    _title(ax, letter, title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(values.shape[1]))
    ax.set_yticks(range(values.shape[0]))
    if annotate and values.size <= 96:
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v):
                r, g, b, _ = image.cmap(image.norm(v))
                light = 0.299 * r + 0.587 * g + 0.114 * b > 0.5            # text contrast by luminance
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=MIN_FONT_PT,
                        color="black" if light else "white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=MIN_FONT_PT)


# ----------------------------------------------------------------------------------------------
# S2 panel A
# ----------------------------------------------------------------------------------------------

def plot_head_block_norms(
    norms:   Mapping[int, Mapping[str, np.ndarray]],
    out_dir: Path,
    name:    str = "s2a_head_block_shares",
) -> List[Path]:
    r"""Per layer (rows) and map (columns) the (H, B) block-energy shares; own blocks outlined."""
    import matplotlib.pyplot as plt
    from vfe3.posthoc.maps import block_energy_shares

    _style()
    layers = sorted(norms)
    maps = [m for m in MAP_NAMES if any(m in norms[layer] for layer in layers)]
    fig, axes = plt.subplots(len(layers), len(maps), figsize=(MAX_WIDTH_IN, 1.15 * len(layers) + 0.5),
                             squeeze=False, constrained_layout=True)
    image = None
    for i, layer in enumerate(layers):
        for j, name_map in enumerate(maps):
            ax = axes[i, j]
            if name_map not in norms[layer]:
                ax.axis("off")
                continue
            shares = block_energy_shares(norms[layer][name_map])            # (H, B)
            H, B = shares.shape
            image = ax.imshow(shares, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto",
                              interpolation="nearest")
            if H == B:
                ax.plot(range(H), range(H), linestyle="none", marker="s", markersize=5,
                        markerfacecolor="none", markeredgecolor="#56B4E9", markeredgewidth=0.6)
            ax.set_xticks(range(B))
            ax.set_yticks(range(H))
            if i == 0:
                ax.set_title(MAP_LABELS[name_map])
            if i == len(layers) - 1:
                ax.set_xlabel("frame block $b$")
            if j == 0:
                ax.set_ylabel(f"layer {layer}\nhead $h$")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.01)
        colorbar.set_label(r"share of $\|M_h\|_F^2$ on block $b$ (uniform $=1/B$)")
        colorbar.ax.tick_params(labelsize=MIN_FONT_PT)
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------------------
# S1 panel B
# ----------------------------------------------------------------------------------------------

def _invariant_panels(rows: Sequence[Mapping[str, Any]]) -> List[tuple]:
    spread = [dict(r, gain_spread=(r["gain_max"] / r["gain_min"]) if r.get("gain_min") else float("nan"),
                   log10_scale=math.log10(r["scale_rms"]) if r.get("scale_rms", 0) > 0 else float("nan"),
                   log10_sigma=(math.log10(r["M_sigma_max"])
                                if r.get("M_sigma_max") and r["M_sigma_max"] > 0 else float("nan")))
              for r in rows]
    return [
        (_grid(spread, "log10_scale"), r"$\log_{10}$ common scale", "{:.2f}"),
        (_grid(spread, "gain_spread"), r"gain spread $g_{\max}/g_{\min}$", "{:.1f}"),
        (_grid(spread, "angle_median_deg"), "median angle (deg)", "{:.0f}"),
        (_grid(spread, "log10_sigma"), r"$\log_{10}\sigma_{\max}(O_hT_h)$", "{:.2f}"),
    ]


def plot_map_invariants(
    rows:    Sequence[Mapping[str, Any]],
    out_dir: Path,
    name:    str = "s1b_map_invariants",
) -> List[Path]:
    r"""Layer-by-head heatmaps of the S1 panel B quantities at one set of weights."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 4, figsize=(MAX_WIDTH_IN, 2.1), constrained_layout=True)
    for ax, letter, (values, label, fmt) in zip(axes, "ABCD", _invariant_panels(rows)):
        _heatmap(ax, fig, values, letter, label, fmt=fmt)
    return _save(fig, out_dir, name)


def plot_map_invariant_curves(
    rows:    Sequence[Mapping[str, Any]],
    out_dir: Path,
    name:    str = "s1b_map_invariant_curves",
) -> List[Path]:
    r"""Over training: per layer, the median over heads of the S1 panel B quantities."""
    import matplotlib.pyplot as plt

    _style()
    steps = sorted({int(r["step"]) for r in rows})
    layers = sorted({int(r["layer"]) for r in rows})
    keys = [("scale_rms", "common scale", True),
            ("gain_spread", r"gain spread $g_{\max}/g_{\min}$", True),
            ("angle_median_deg", "median angle (deg)", False),
            ("M_sigma_max", r"$\sigma_{\max}(O_hT_h)$", True),
            ("link_spectral_radius", r"$\rho(T_hO_h)$", True)]
    fig, axes = plt.subplots(1, len(keys), figsize=(MAX_WIDTH_IN, 1.9), constrained_layout=True)
    for ax, letter, (key, label, log) in zip(axes, "ABCDE", keys):
        for layer in layers:
            series = []
            for step in steps:
                values = []
                for r in rows:
                    if int(r["step"]) != step or int(r["layer"]) != layer:
                        continue
                    value = (r["gain_max"] / r["gain_min"]) if key == "gain_spread" else r.get(key)
                    if value is not None and np.isfinite(value):
                        values.append(float(value))
                series.append(np.median(values) if values else np.nan)
            ax.plot(steps, series, marker="o", markersize=2, linewidth=0.9, label=f"layer {layer}")
        ax.set_xlabel("step")
        _title(ax, letter, label)
        if log:
            from matplotlib.ticker import FuncFormatter, NullFormatter

            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
    axes[-1].legend(loc="best", fontsize=MIN_FONT_PT)
    return _save(fig, out_dir, name)


def plot_link_spectrum(
    rows:    Sequence[Mapping[str, Any]],     # link_eigenvalue_rows: (layer, head, k, re, im, abs)
    out_dir: Path,
    name:    str = "s1b_link_spectrum",
) -> List[Path]:
    r"""Per layer, the eigenvalues of :math:`T_h O_h` of every head in the complex plane.

    These are the nonzero spectrum of the content map :math:`M_h = O_h T_h`, a similarity invariant of the
    content links. The dashed unit circle separates amplifying from attenuating directions, and the '+'
    marks :math:`\lambda = 1`, the flat-composition point :math:`T_h O_h = I`.
    """
    import matplotlib.pyplot as plt

    _style()
    layers = sorted({int(r["layer"]) for r in rows})
    heads = sorted({int(r["head"]) for r in rows})
    fig, axes = plt.subplots(1, len(layers), figsize=(MAX_WIDTH_IN, MAX_WIDTH_IN / max(len(layers), 1)),
                             squeeze=False, constrained_layout=True)
    circle = np.linspace(0.0, 2.0 * np.pi, 256)
    for ax, letter, layer in zip(axes[0], "ABCDEFGHIJKLMNOP", layers):
        ax.plot(np.cos(circle), np.sin(circle), linestyle="--", linewidth=0.6, color="0.6")
        ax.axhline(0.0, linewidth=0.4, color="0.85")
        ax.axvline(0.0, linewidth=0.4, color="0.85")
        ax.plot([1.0], [0.0], marker="+", markersize=7, color="0.2", linestyle="none")
        for h in heads:
            pts = [(float(r["re"]), float(r["im"])) for r in rows if int(r["layer"]) == layer and int(r["head"]) == h]
            if pts:
                re, im = zip(*pts)
                ax.scatter(re, im, s=9, label=f"head {h}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(r"Re $\lambda$")
        ax.set_ylabel(r"Im $\lambda$")
        _title(ax, letter, f"layer {layer}")
    axes[0][-1].legend(loc="best", fontsize=MIN_FONT_PT)
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------------------
# S2 panel B
# ----------------------------------------------------------------------------------------------

def plot_attention_stats(
    rows:    Sequence[Mapping[str, Any]],
    out_dir: Path,
    name:    str = "s2b_attention_stats",
) -> List[Path]:
    r"""Layer-by-head normalized entropy, mean distance, self-weight and prior overlap."""
    import matplotlib.pyplot as plt

    _style()
    panels = [
        ("entropy_norm", r"entropy / $\log|\mathcal{S}_i|$", dict(vmin=0.0, vmax=1.0), "{:.2f}"),
        ("distance_mean", r"mean distance $|i-j|$", {}, "{:.1f}"),
        ("self_weight", r"self-weight $\beta_{ii}$", dict(vmin=0.0), "{:.2f}"),
        ("prior_overlap", "overlap with prior", dict(vmin=0.0, vmax=1.0), "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(MAX_WIDTH_IN, 2.1), constrained_layout=True)
    for ax, letter, (key, label, limits, fmt) in zip(axes, "ABCD", panels):
        _heatmap(ax, fig, _grid(rows, key), letter, label, fmt=fmt, **limits)
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------------------
# M5
# ----------------------------------------------------------------------------------------------

def _bootstrap_unit_text(unit: str) -> str:
    r"""Axis wording of a resampling unit: ``'article'``, or ``'block_<n>'`` as an ``n``-token block."""
    if unit.startswith("block_") and unit[len("block_"):].isdigit():
        return f"{int(unit[len('block_'):])}-token block"
    return unit.replace("_", " ")


def plot_interventions(
    rows:    Sequence[Mapping[str, Any]],
    out_dir: Path,
    name:    str = "m5_interventions",

    *,
    unit:    str = "article",
) -> List[Path]:
    r"""Forest plot of :math:`\Delta\mathrm{CE}` (nats) per M5 row with its 95% bootstrap interval.

    A symmetric-log axis keeps effects from :math:`10^{-3}` to several nats readable; rows without a
    number carry their status as text. ``unit`` is the resampling unit of the paired bootstrap (the
    interventions report's ``unit``: the article, or a fixed token block when no article headings are
    found), named on the axis.
    """
    import matplotlib.pyplot as plt

    _style()
    rows = sorted(rows, key=lambda r: int(r["row"]))
    fig, ax = plt.subplots(figsize=(MAX_WIDTH_IN, 0.24 * len(rows) + 0.8), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    finite = [abs(float(r["delta_ce"])) for r in rows
              if r.get("delta_ce") is not None and np.isfinite(r["delta_ce"]) and r["delta_ce"] != 0]
    linthresh = max(1e-4, min(finite) / 2) if finite else 1e-3
    for yi, r in zip(y, rows):
        delta = r.get("delta_ce")
        if r.get("status") == "ok" and delta is not None and np.isfinite(delta):
            lo, hi = r.get("lo"), r.get("hi")
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [yi, yi], color="#0072B2", linewidth=1.2)
            ax.plot([delta], [yi], marker="o", color="#0072B2", markersize=3.5)
        else:
            ax.text(linthresh, yi, f"  {r.get('status', 'pending')}", va="center", ha="left",
                    fontsize=MIN_FONT_PT, color="#555555")
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xscale("symlog", linthresh=linthresh)
    ends = [float(v) for r in rows if r.get("status") == "ok"
            for v in (r.get("delta_ce"), r.get("lo"), r.get("hi")) if v is not None and np.isfinite(v)]
    if ends:                                        # a multiplicative margin so no marker is clipped
        ax.set_xlim(min(-linthresh, 2.0 * min(ends)), max(linthresh, 2.0 * max(ends)))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(r['row'])}. {r['label']}" for r in rows], fontsize=MIN_FONT_PT + 0.5)
    ax.set_xlabel(r"$\Delta$CE (nats per target), 95% " + f"{_bootstrap_unit_text(unit)}-bootstrap interval")
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------------------
# S1 panel A endpoints and v2:899 (residual ratios, LayerNorm change, self-weights)
# ----------------------------------------------------------------------------------------------

def plot_residual_ratio(
    result:  Mapping[str, Any],
    out_dir: Path,
    name:    str = "s1a_residual_ratio",
) -> List[Path]:
    r"""Per layer: :math:`\|m_i\|/\|\mu_i\|` quantiles, the relative LayerNorm change and the self-weight.

    Panel A: 10-90% band, median and minimum of :math:`\rho_i` (log axis). Panel B: median, 90% and
    maximum of :math:`\delta_i` (log axis), or the reason it is not defined. Panel C: mean self-weight
    :math:`\beta_{ii}` with its range over heads, when attention statistics were collected.
    """
    import matplotlib.pyplot as plt

    _style()
    rows = sorted(result.get("layers", []), key=lambda r: int(r["layer"]))
    layers = np.array([int(r["layer"]) for r in rows])
    weights = sorted((result.get("self_weight") or {}).get("layers", []), key=lambda r: int(r["layer"]))
    n_panels = 3 if weights else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(MAX_WIDTH_IN, 2.0), constrained_layout=True)

    def column(key: str) -> np.ndarray:
        return np.array([float(r.get(key, np.nan)) for r in rows])

    ax = axes[0]
    ax.fill_between(layers, column("rho_q10"), column("rho_q90"), color="#0072B2", alpha=0.25, linewidth=0,
                    label="10-90%")
    ax.plot(layers, column("rho_q50"), marker="o", markersize=3, color="#0072B2", linewidth=1.0, label="median")
    ax.plot(layers, column("rho_min"), marker="x", markersize=3.5, linestyle="none", color="#D55E00",
            label="minimum")
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    _title(ax, "A", r"$\|m_i\|/\|\mu_i\|$" if result.get("residual_additive", True)
           else r"$\|\mu^{\rm out}_i-\mu_i\|/\|\mu_i\|$")
    ax.legend(loc="best", fontsize=MIN_FONT_PT)

    ax = axes[1]
    ln = result.get("ln_change") or {}
    finite = np.isfinite(column("ln_change_q50"))
    if ln.get("defined") and finite.any():
        ax.plot(layers, column("ln_change_q50"), marker="o", markersize=3, color="#009E73", linewidth=1.0,
                label="median")
        ax.plot(layers, column("ln_change_q90"), marker="s", markersize=2.5, color="#009E73", linewidth=0.7,
                linestyle="--", label="90%")
        ax.plot(layers, column("ln_change_max"), marker="^", markersize=3, linestyle="none", color="#D55E00",
                label="maximum")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=MIN_FONT_PT)
    else:
        ax.text(0.5, 0.5, "not defined:\n" + str(ln.get("reason") or "no values"), transform=ax.transAxes,
                ha="center", va="center", fontsize=MIN_FONT_PT, wrap=True)
        ax.set_yticks([])
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    _title(ax, "B", r"$\|\mathrm{LN}(\mu+m)-\mathrm{LN}(m)\|/\|\mathrm{LN}(\mu+m)\|$")

    if weights:
        ax = axes[2]
        w_layers = np.array([int(r["layer"]) for r in weights])
        mean = np.array([float(r["self_weight_mean"]) for r in weights])
        lo = np.array([float(r["self_weight_head_min"]) for r in weights])
        hi = np.array([float(r["self_weight_head_max"]) for r in weights])
        ax.errorbar(w_layers, mean, yerr=np.vstack([mean - lo, hi - mean]), marker="o", markersize=3,
                    color="#CC79A7", linewidth=1.0, capsize=2, elinewidth=0.7)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(w_layers)
        ax.set_xlabel("layer")
        _title(ax, "C", r"self-weight $\beta_{ii}$ (head range)")
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------------------
# v2:872 (routing covariance gap and hidden temperature)
# ----------------------------------------------------------------------------------------------

def _ecdf(ax: Any, values: np.ndarray, label: str, color: str) -> None:
    values = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if values.size:
        ax.plot(values, np.arange(1, values.size + 1) / values.size, color=color, linewidth=1.0, label=label)


def plot_covariance_gap(
    result:  Mapping[str, Any],
    out_dir: Path,
    name:    str = "v2_872_covariance",
) -> List[Path]:
    r"""ECDFs of the relative Frobenius and log-det gaps per sample, and the per-coordinate temperature.

    Panel A: :math:`\|C_i - cG_i^{-1}\|_F/\|C_i\|_F`. Panel B:
    :math:`|\log\det C_i - \log\det(cG_i^{-1})|/d`. Panel C: histogram of
    :math:`\sigma^2_{\rm init}/\sigma^2_{v,k}` with the headline factor
    :math:`\sigma^2_{\rm init}/\overline{\sigma^2}` marked.
    """
    import matplotlib.pyplot as plt

    _style()
    rc = result.get("routing_covariance") or {}
    ht = result.get("hidden_temperature") or {}
    colors = {"occurrence": "#0072B2", "uniform_types": "#E69F00"}
    fig, axes = plt.subplots(1, 3, figsize=(MAX_WIDTH_IN, 2.0), constrained_layout=True)
    panels = ((axes[0], "A", "frobenius_gap", r"$\|C_i-cG_i^{-1}\|_F/\|C_i\|_F$"),
              (axes[1], "B", "logdet_gap", r"$|\log\det C_i-\log\det cG_i^{-1}|/d$"))
    for ax, letter, key, label in panels:
        drawn: List[np.ndarray] = []
        if rc.get("status") == "ok":
            for sample, entry in rc["samples"].items():
                values = np.asarray((entry.get("arrays") or {}).get(key, []), dtype=np.float64)
                if (values > 0).any():
                    values = np.maximum(values, values[values > 0].min())
                    _ecdf(ax, values, f"{sample} (c={entry['c']:.3g})", colors.get(sample, "#555555"))
                    drawn.append(values)
        if drawn:
            pooled = np.concatenate(drawn)
            if pooled.max() > 100.0 * pooled.min():               # log axis only over >= 2 decades
                ax.set_xscale("log")
            ax.set_ylabel("fraction of pairs")
            ax.set_ylim(0.0, 1.02)
            ax.legend(loc="best", fontsize=MIN_FONT_PT)
        else:
            text = (f"{rc.get('status')}:\n{rc.get('reason') or rc.get('error')}" if rc.get("status") != "ok"
                    else "every gap is zero")
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=MIN_FONT_PT,
                    wrap=True)
        _title(ax, letter, label)
    ax = axes[2]
    factor = (ht.get("arrays") or {}).get("per_coordinate_factor") if ht.get("status") == "ok" else None
    values = np.asarray(factor if factor is not None else [], dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size:
        lo, hi = float(np.quantile(values, 0.001)), float(np.quantile(values, 0.999))
        shown = values[(values >= lo) & (values <= hi)]
        ax.hist(shown, bins=60 if hi > lo * (1 + 1e-9) else 10, weights=np.full(shown.size, 1.0 / values.size),
                color="#009E73", alpha=0.8)
        ax.axvline(float(ht["factor"]), color="#D55E00", linewidth=1.0,
                   label=rf"$\sigma^2_{{\rm init}}/\overline{{\sigma^2}}={ht['factor']:.3g}$")
        ax.set_ylabel("fraction of table entries")
        ax.set_xlabel(r"$\sigma^2_{\rm init}/\sigma^2_{v,k}$ (central 99.8%)")
        ax.legend(loc="upper right", fontsize=MIN_FONT_PT)
    else:
        ax.text(0.5, 0.5, f"{ht.get('status')}:\n{ht.get('reason') or ht.get('error')}", transform=ax.transAxes,
                ha="center", va="center", fontsize=MIN_FONT_PT, wrap=True)
    _title(ax, "C", "hidden temperature")
    return _save(fig, out_dir, name)


# ----------------------------------------------------------------------------------------------
# Cross-entropy by context position
# ----------------------------------------------------------------------------------------------

CONTEXT_MIN_TARGETS: int = 30             # context lengths with fewer scored targets are not drawn


def ce_by_context(context: np.ndarray, nll: np.ndarray) -> List[Dict[str, float]]:
    r"""Per context length :math:`c` (tokens up to and including the query): count, mean NLL, its SE."""
    context = np.asarray(context, dtype=np.int64)
    nll = np.asarray(nll, dtype=np.float64)
    if not context.size:
        return []
    counts = np.bincount(context)
    sums = np.bincount(context, weights=nll)
    squares = np.bincount(context, weights=nll * nll)
    rows: List[Dict[str, float]] = []
    for c in np.nonzero(counts)[0]:
        n = float(counts[c])
        mean = sums[c] / n
        var = max(squares[c] / n - mean * mean, 0.0) * n / (n - 1) if n > 1 else float("nan")
        rows.append({"context": int(c), "n": int(n), "ce": float(mean),
                     "se": float(math.sqrt(var / n)) if n > 1 else float("nan")})
    return rows


def plot_ce_by_context(
    series:  Mapping[str, Sequence[Mapping[str, float]]],
    out_dir: Path,
    name:    str = "ce_by_context",
) -> List[Path]:
    r"""Mean CE (nats) against context length per labelled series, with a 95% normal band.

    Under the strided evaluation protocol only the first window of a split scores short contexts, so
    lengths with fewer than :data:`CONTEXT_MIN_TARGETS` targets are omitted (stated on the axis).
    """
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    drawn = False
    for (label, rows), color in zip(series.items(), ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):
        kept = [r for r in rows if r["n"] >= CONTEXT_MIN_TARGETS]
        if not kept:
            continue
        x = np.array([r["context"] for r in kept])
        y = np.array([r["ce"] for r in kept])
        se = np.nan_to_num(np.array([r["se"] for r in kept]))
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.2, linewidth=0)
        ax.plot(x, y, color=color, linewidth=0.9, label=label)
        drawn = True
    ax.set_xlabel(f"context length (tokens; < {CONTEXT_MIN_TARGETS} targets omitted)")
    ax.set_ylabel("CE (nats per target)")
    if drawn:
        ax.legend(loc="best", fontsize=MIN_FONT_PT)
    else:
        ax.text(0.5, 0.5, f"no context length has {CONTEXT_MIN_TARGETS} scored targets", transform=ax.transAxes,
                ha="center", va="center", fontsize=MIN_FONT_PT)
    return _save(fig, out_dir, name)
