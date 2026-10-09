r"""Figures of the minimal-agent attention readout and the prototype-agent block MLP.

History figures read the eval-cadence tables ``diagnostics/minimal_agent_heads.csv`` and
``diagnostics/prototype_templates.csv`` and the log-cadence ``diagnostics/prototype_templates_train.csv``
(:func:`vfe3.train._feature_rows`). Probe figures read the dictionaries of
:mod:`vfe3.posthoc.feature_probes`. :func:`plot_template_umap` is the only expensive one; the report runs it only
under ``generate_embedding_figures``. Every builder raises :class:`vfe3.viz.figures.FigureSkipped` when it has no
data to show.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from vfe3.viz.figures import MAX_FIGURE_WIDTH_IN, FigureSkipped, _compact_style, _layer_colors, _np, _save, plt
from vfe3.viz.text import token_label

_HEAD_PANELS = (
    ("gain_fraction", "gain fraction", False),
    ("channel_noise", "channel noise $n_h$", True),
    ("head_coupling", "head coupling", False),
    ("label_staleness", "mean-field label staleness (TV)", False),
    ("spread_share", "exact spread share", False),
    ("covariance_contraction", r"covariance contraction tr$C^+$/tr$C$", False),
)
_TEMPLATE_PANELS = (
    ("effective_templates", "effective templates in use"),
    ("dead_fraction", "dead-template fraction"),
    ("log_lambda", r"$\log\lambda$ (min, median, max)"),
    ("prior_entropy", r"prior entropy $H(\pi)$"),
    ("gain_fraction", "gain fraction"),
    ("weight_entropy", "token weight entropy (nats)"),
)
_LINESTYLES = ("-", "--", ":", "-.")


def _finite_columns(table: Mapping[str, Any], names: Sequence[str]) -> list:
    return [name for name in names
            if name in table and np.isfinite(_np(table[name]).astype(float)).any()]


def _panel_grid(n: int):
    n_cols = 2 if n > 1 else 1
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(MAX_FIGURE_WIDTH_IN if n_cols > 1 else 3.6, 2.0 * n_rows + 0.4))
    for extra in range(n, n_rows * n_cols):
        axes[extra // n_cols][extra % n_cols].set_visible(False)
    return fig, axes, n_cols


def plot_minimal_agent_heads(
    table: Mapping[str, Any],            # diagnostics/minimal_agent_heads.csv columns
    *,
    path:  Optional[str] = None,
):
    r"""Per-(layer, head) minimal-agent diagnostics over training: one line per head (color = layer, line style =
    head). Panels whose column is all NaN (staleness under ``"exact"``, spread share under ``"mean_field"``) are
    left out. Raises :class:`FigureSkipped` with fewer than two evaluation steps."""
    steps = _np(table["step"]).astype(float)
    grid = np.unique(steps[np.isfinite(steps)])
    if grid.size < 2:
        raise FigureSkipped("minimal_agent_heads.csv has fewer than two evaluation steps")
    panels = [spec for spec in _HEAD_PANELS if spec[0] in _finite_columns(table, [s[0] for s in _HEAD_PANELS])]
    layers, heads = _np(table["layer"]).astype(int), _np(table["head"]).astype(int)
    colors = _layer_colors(int(layers.max()) + 1)
    with _compact_style():
        fig, axes, n_cols = _panel_grid(len(panels))
        for index, (name, label, log_y) in enumerate(panels):
            ax = axes[index // n_cols][index % n_cols]
            values = _np(table[name]).astype(float)
            for layer in np.unique(layers):
                for head in np.unique(heads[layers == layer]):
                    mask = (layers == layer) & (heads == head)
                    order = np.argsort(steps[mask])
                    ax.plot(steps[mask][order], values[mask][order], _LINESTYLES[head % len(_LINESTYLES)],
                            lw=1.0, color=colors[layer], label=f"L{layer} h{head}")
            if log_y:
                ax.set_yscale("log")
            ax.set(xlabel="training step", title=label)
        axes[0][0].legend(ncol=2, frameon=False)
        fig.tight_layout()
        return _save(fig, path)


def plot_prototype_templates(
    table:       Optional[Mapping[str, Any]],            # diagnostics/prototype_templates.csv columns
    *,
    train_table: Optional[Mapping[str, Any]] = None,     # diagnostics/prototype_templates_train.csv columns
    path:        Optional[str] = None,
):
    r"""Template health over training, one color per layer: templates in use, dead fraction, the band of
    :math:`\log\lambda` (min to max, line at the median), :math:`H(\pi)`, gain fraction and token weight entropy.
    Log-step rows on the training batch (``train_table``) are drawn as lines; eval rows on the fixed validation
    windows (``table``) as markers. Raises :class:`FigureSkipped` when neither table has two distinct steps."""
    sources = []
    for source, style in ((train_table, {"lw": 1.0, "alpha": 0.85}),
                          (table, {"ls": "none", "marker": "o", "ms": 2.5})):
        if source is None:
            continue
        steps = _np(source["step"]).astype(float)
        if np.unique(steps[np.isfinite(steps)]).size >= 2:
            sources.append((source, steps, _np(source["layer"]).astype(int), style))
    if not sources:
        raise FigureSkipped("prototype template tables have fewer than two steps")
    colors = _layer_colors(max(int(layers.max()) for _, _, layers, _ in sources) + 1)
    with _compact_style():
        fig, axes, n_cols = _panel_grid(len(_TEMPLATE_PANELS))
        for index, (name, label) in enumerate(_TEMPLATE_PANELS):
            ax = axes[index // n_cols][index % n_cols]
            for source, steps, layers, style in sources:
                for layer in np.unique(layers):
                    mask = layers == layer
                    order = np.argsort(steps[mask])
                    x = steps[mask][order]
                    if name == "log_lambda":
                        lo, mid, hi = (_np(source[f"log_lambda_{k}"]).astype(float)[mask][order]
                                       for k in ("min", "median", "max"))
                        if source is train_table or train_table is None:
                            ax.fill_between(x, lo, hi, color=colors[layer], alpha=0.18, lw=0)
                        y = mid
                    else:
                        y = _np(source[name]).astype(float)[mask][order]
                    ax.plot(x, y, color=colors[layer], **style)
            ax.set(xlabel="training step", title=label)
        handles = [plt.Line2D([], [], color=colors[layer], lw=1.1, label=f"layer {layer}")
                   for layer in range(len(colors))]
        if len(sources) == 2:
            handles += [plt.Line2D([], [], color="0.4", lw=1.0, label="training batch (log steps)"),
                        plt.Line2D([], [], color="0.4", ls="none", marker="o", ms=2.5,
                                   label="validation windows (eval)")]
        axes[0][0].legend(handles=handles, frameon=False)
        fig.tight_layout()
        return _save(fig, path)


def plot_minimal_agent_gap(
    gap:  Mapping[str, Any],             # vfe3.posthoc.feature_probes.minimal_agent_gap
    *,
    path: Optional[str] = None,
):
    r"""Exact versus mean-field minimal agent at the trained inputs: per layer and head the label total variation
    and the exact spread share; per layer the whitened mean shift and the per-dimension log-determinant ratio
    (mean field minus exact)."""
    rows = gap.get("layers") or []
    if not rows:
        raise FigureSkipped("no minimal-agent layer")
    layers = [row["layer"] for row in rows]
    H = len(rows[0]["label_tv"])
    width = 0.8 / max(1, H)
    with _compact_style():
        fig, axes = plt.subplots(1, 3, figsize=(MAX_FIGURE_WIDTH_IN, 2.3))
        for head in range(H):
            offset = (head - (H - 1) / 2) * width
            axes[0].bar(np.arange(len(rows)) + offset, [row["label_tv"][head] for row in rows], width,
                        label=f"head {head}")
            axes[1].bar(np.arange(len(rows)) + offset, [row["spread_share"][head] for row in rows], width)
        axes[2].bar(np.arange(len(rows)) - 0.2, [row["whitened_mean_shift"] for row in rows], 0.4,
                    label="whitened mean shift")
        axes[2].bar(np.arange(len(rows)) + 0.2, [row["logdet_ratio_per_dim"] for row in rows], 0.4,
                    label="log det ratio / K (mf - exact)")
        for ax, title in zip(axes, ("label TV (exact vs mean field)", "exact spread share",
                                    "posterior moments")):
            ax.set_xticks(np.arange(len(rows)), [f"L{layer}" for layer in layers])
            ax.set_title(title)
            ax.axhline(0.0, color="0.5", lw=0.6)
        axes[0].legend(frameon=False)
        axes[2].legend(frameon=False)
        fig.suptitle(f"minimal agent, trained with {gap.get('configured_inference')}")
        fig.tight_layout()
        return _save(fig, path)


def plot_template_usage(
    probe: Mapping[str, Any],            # vfe3.posthoc.feature_probes.prototype_probe
    *,
    path:  Optional[str] = None,
):
    r"""Rank-ordered mean template weight per layer (log scale) with the uniform level :math:`1/M` and the dead
    threshold :math:`1/(10M)`."""
    rows = probe.get("layers") or []
    if not rows:
        raise FigureSkipped("no prototype_agents layer")
    colors = _layer_colors(max(row["layer"] for row in rows) + 1)
    with _compact_style():
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for row in rows:
            usage = np.sort(np.asarray(row["usage"], dtype=float))[::-1]
            ax.plot(np.arange(1, usage.size + 1), np.maximum(usage, 1e-12), lw=1.1, color=colors[row["layer"]],
                    label=f"layer {row['layer']}")
        M = len(rows[0]["usage"])
        ax.axhline(1.0 / M, color="0.4", lw=0.7, ls="--", label="uniform 1/M")
        ax.axhline(1.0 / (10 * M), color="0.6", lw=0.7, ls=":", label="dead 1/(10M)")
        ax.set(xscale="log", yscale="log", xlabel="template rank", ylabel="mean weight",
               title="template usage (final model)")
        ax.legend(frameon=False)
        fig.tight_layout()
        return _save(fig, path)


def plot_template_codebook(
    probe:  Mapping[str, Any],           # vfe3.posthoc.feature_probes.prototype_probe
    *,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    top:    int = 12,
    n_tokens: int = 4,
    path:   Optional[str] = None,
):
    r"""The ``top`` most-used templates of each layer: one panel per layer (two columns), a horizontal bar of each
    template's mean weight (%) labeled by template index on the left, and on the right the ``n_tokens`` tokens it
    wins most often (top-1 assignment over the report tokens), each in quotes, decoded to text when ``decode`` is
    given (leading spaces dropped). A template that wins no report token shows "(none)"."""
    rows = probe.get("layers") or []
    if not rows:
        raise FigureSkipped("no prototype_agents layer")
    n_cols = 2 if len(rows) > 1 else 1
    n_rows = (len(rows) + n_cols - 1) // n_cols
    k = max(1, min(top, max(len(row["codebook"]) for row in rows)))
    with _compact_style():
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, constrained_layout=True,
                                 figsize=(MAX_FIGURE_WIDTH_IN, n_rows * (0.16 * k + 0.6)))
        for extra in range(len(rows), n_rows * n_cols):
            axes[extra // n_cols][extra % n_cols].set_visible(False)
        for index, row in enumerate(rows):
            ax = axes[index // n_cols][index % n_cols]
            items = row["codebook"][:k]
            usage = [100.0 * item["usage"] for item in items]
            y = np.arange(len(items))
            ax.barh(y, usage, height=0.7, color=_layer_colors(1)[0])
            ax.set_yticks(y, [f"#{item['template']}" for item in items])
            ax.set_ylim(len(items) - 0.5, -0.5)                             # most used on top
            ax.set_xlabel("mean weight (%)")
            ax.set_title(f"layer {row['layer']}")
            words = ax.twinx()
            words.set_ylim(ax.get_ylim())
            words.set_yticks(y, ["  ".join(f"‘{token_label(token, decode=decode).strip() or token_label(token, decode=decode)}’"
                                           for token, _ in item["top_tokens"][:n_tokens])
                                 or "(none)" for item in items])
            words.tick_params(axis="y", length=0, labelsize=6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
                words.spines[side].set_visible(False)
        fig.suptitle("prototype templates: usage and most frequent tokens")
        return _save(fig, path)


def plot_template_umap(
    probe:  Mapping[str, Any],           # vfe3.posthoc.feature_probes.prototype_probe
    *,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    label_top: int = 12,
    seed:   int = 0,
    path:   Optional[str] = None,
):
    r"""UMAP of the template means :math:`\nu_a` per layer; marker area and color follow log usage, and the
    ``label_top`` most-used templates carry their most frequent token. Expensive: the report runs it only under
    ``generate_embedding_figures``."""
    rows = probe.get("layers") or []
    if not rows:
        raise FigureSkipped("no prototype_agents layer")
    import umap
    with _compact_style():
        fig, axes = plt.subplots(1, len(rows), squeeze=False, figsize=(MAX_FIGURE_WIDTH_IN, 3.0))
        for ax, row in zip(axes[0], rows):
            nu = np.asarray(row["nu"], dtype=float)
            usage = np.maximum(np.asarray(row["usage"], dtype=float), 1e-12)
            xy = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(15, len(nu) - 1)).fit_transform(nu)
            log_usage = np.log10(usage)
            scatter = ax.scatter(xy[:, 0], xy[:, 1], s=2 + 30 * (log_usage - log_usage.min())
                                 / max(1e-12, np.ptp(log_usage)), c=log_usage, cmap="viridis", lw=0)
            for item in row["codebook"][:label_top]:
                if item["top_tokens"]:
                    a = item["template"]
                    ax.annotate(token_label(item["top_tokens"][0][0], decode=decode), xy[a], fontsize=5.5)
            ax.set_title(f"layer {row['layer']}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(scatter, ax=axes[0].tolist(), label="log10 mean weight", shrink=0.8)
        fig.suptitle(r"template means $\nu_a$ (UMAP)")
        return _save(fig, path)
