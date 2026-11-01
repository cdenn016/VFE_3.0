r"""Audit 2026-10-31 remediation, package viz-figures: figure builders and their inputs.

Covers viz-figures-1 to -8 and -10, viz-pipeline-3, -4 and -6 of the October 31 ultradeep audit (the offset-law
interval of analysis-statistics-2 is in ``test_audit_20261031_viz_figures_scaling.py``). Inputs are synthetic numpy
or torch arrays and one tiny model from ``tests.profiles``; every figure is built headless and closed.
"""

import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from vfe3 import metrics  # noqa: E402
from vfe3.viz import figures as figs  # noqa: E402
from vfe3.viz import text as viz_text  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _causal_maps(shape: tuple, seed: int = 0) -> torch.Tensor:
    r"""Causal softmax maps of ``shape`` (..., N, N): exact zeros above the diagonal."""
    n = shape[-1]
    logits = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    return torch.softmax(logits.masked_fill(future, float("-inf")), dim=-1)


def _reference_head_redundancy_js(beta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""The unblocked (H, H, N, N) form of ``head_redundancy_js`` before audit 2026-10-31 viz-figures-1."""
    b = beta.clamp(min=eps)
    bh, bh2 = b.unsqueeze(1), b.unsqueeze(0)
    m = 0.5 * (bh + bh2)
    kl1 = (bh * (torch.log(bh) - torch.log(m))).sum(dim=-1)
    kl2 = (bh2 * (torch.log(bh2) - torch.log(m))).sum(dim=-1)
    return (0.5 * (kl1 + kl2)).mean(dim=-1)


class _OutputSizes(TorchDispatchMode):
    r"""Records the element count of every aten op output."""

    def __init__(self) -> None:
        super().__init__()
        self.sizes: List[int] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple = (), kwargs: Optional[dict] = None) -> Any:
        out = func(*args, **(kwargs or {}))
        for value in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(value, torch.Tensor):
                self.sizes.append(int(value.numel()))
        return out


# --------------------------------------------------------------------------- viz-figures-1 and -5

def test_head_redundancy_js_row_blocks_bound_the_transient_and_keep_the_matrix() -> None:
    beta = _causal_maps((10, 6, 12, 12)).reshape(60, 12, 12)          # (L*H, N, N) of a (10, 6) stack
    reference = _reference_head_redundancy_js(beta)
    per_row = 60 * 12 * 12                                              # one (1, L*H, N, N) block
    recorder = _OutputSizes()
    with recorder:
        js = metrics.head_redundancy_js(beta, max_block_elements=per_row)
    assert max(recorder.sizes) <= per_row                               # the unblocked form makes (L*H)^2 N^2
    assert torch.equal(js, reference)                                   # the same elementwise ops per pair


def test_head_redundancy_js_of_one_layer_is_one_block_and_bitwise_unchanged() -> None:
    beta = _causal_maps((6, 16, 16), seed=3)                            # (H, N, N) of the val diagnostics
    assert torch.equal(metrics.head_redundancy_js(beta), _reference_head_redundancy_js(beta))


def test_attention_structure_decay_panel_drops_the_legend_above_a_dozen_curves() -> None:
    beta = _causal_maps((10, 6, 8, 8), seed=1)
    fig = figs.plot_attention_structure(beta)
    decay = fig.axes[2]
    fig.canvas.draw()
    legend = decay.get_legend()
    assert legend is None or legend.get_window_extent().height <= decay.get_window_extent().height
    assert len(decay.get_lines()) == 60
    # The (layer, head) maps keep their identity through the layer color.
    assert len({tuple(np.round(matplotlib.colors.to_rgba(line.get_color()), 6)) for line in decay.get_lines()}) == 10
    js = np.asarray(fig.axes[1].images[0].get_array())
    np.testing.assert_allclose(js, _reference_head_redundancy_js(beta.reshape(60, 8, 8)).numpy(), rtol=1e-5)


def test_attention_structure_keeps_the_legend_for_a_few_curves() -> None:
    fig = figs.plot_attention_structure(_causal_maps((2, 2, 8, 8), seed=2))
    legend = fig.axes[2].get_legend()
    assert legend is not None and len(legend.get_texts()) == 4


# --------------------------------------------------------------------------- viz-figures-2

def test_sigma_stratified_error_trace_ticks_stay_distinct_at_a_one_percent_spread() -> None:
    rng = np.random.default_rng(0)
    tr = 2880.0 * np.exp(0.01 * rng.standard_normal(4000))
    ce = 3.0 + 0.1 * rng.standard_normal(4000)
    fig = figs.plot_sigma_stratified_error({"tr_sigma": tr, "ce": ce}, frozen_covariance=True, n_boot=20)
    ax = fig.axes[0]
    fig.canvas.draw()
    lo, hi = sorted(ax.get_xlim())
    labels = [label.get_text() for label, loc in zip(ax.get_xticklabels(), ax.get_xticks()) if lo <= loc <= hi]
    assert len(labels) >= 3 and len(set(labels)) == len(labels)


# --------------------------------------------------------------------------- viz-figures-3

_FROZEN_LIVE_TITLES = ("Belief conditioning", "Belief effective rank", "Belief precision", "Guard saturation",
                       "Held-out conditioning", "Held-out precision", "Held-out guard saturation")


@pytest.mark.parametrize("builder,columns", [
    (figs.plot_geometry_health, figs.GEOMETRY_HEALTH_COLUMNS),
    (figs.plot_validation_sanity, figs.VALIDATION_SANITY_COLUMNS),
], ids=("geometry_health", "validation_sanity"))
def test_frozen_covariance_dashboards_title_sigma_panels_as_token_table_statistics(
        builder: Callable[..., Any], columns: tuple) -> None:
    steps = np.arange(1, 11) * 100.0
    history = {"step": steps, **{column: 1.0 + 0.1 * np.arange(10) for column in columns}}
    live = [ax.get_title() for ax in builder(history).axes]
    assert any(title in _FROZEN_LIVE_TITLES for title in live)
    frozen = [ax.get_title() for ax in builder(history, covariance_frozen=True).axes]
    assert not any(title in _FROZEN_LIVE_TITLES or title.startswith("Belief") for title in frozen)
    assert sum("token table" in title for title in frozen) == sum(title in _FROZEN_LIVE_TITLES for title in live)


def test_save_figures_passes_the_frozen_route_to_both_dashboards(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    from tests.profiles import test_config
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts, _save_figures

    cfg = test_config(attention_mode="projected_predictive", vocab_size=16, embed_dim=4, n_heads=2,
                      max_seq_len=8, max_steps=2, warmup_steps=2, batch_size=2)
    assert metrics.covariance_frozen_at_encode(cfg)
    torch.manual_seed(0)
    art = RunArtifacts(str(tmp_path), cfg, VFEModel(cfg), dataset="synthetic", device="cpu")
    art.history = [{"step": s, "belief_cond_p95": 2.0 + s, "val_belief_cond_p95": 2.0 + s} for s in range(12)]
    seen: Dict[str, Any] = {}

    def capture(name: str) -> Callable[..., Any]:
        def fake(history: Any, **kwargs: Any):
            seen[name] = kwargs.get("covariance_frozen")
            raise figs.FigureSkipped("captured by the test")
        return fake

    monkeypatch.setattr(figs, "plot_geometry_health", capture("geometry_health"))
    monkeypatch.setattr(figs, "plot_validation_sanity", capture("validation_sanity"))
    _save_figures(art, [3.0, 2.5], logging.getLogger("test"))
    assert seen == {"geometry_health": True, "validation_sanity": True}


# --------------------------------------------------------------------------- viz-figures-6 and viz-pipeline-3

def test_gauge_equivariance_draws_a_nonfinite_residual_as_absent_not_at_the_floor() -> None:
    floor = 2.220446049250313e-16
    resid = {"residuals": {"routing_score": [1e-14, float("nan")], "routing_weights": [0.0, float("inf")]},
             "readout_mode": "predictive_content", "n_draws": 3, "unit_roundoff": floor}
    fig = figs.plot_gauge_equivariance(resid)
    ax = fig.axes[0]
    heights = [patch.get_height() for patch in ax.patches]          # layer 0 then layer 1, per quantity
    assert heights[0] == pytest.approx(1e-14) and heights[1] == pytest.approx(floor)   # exact zero at the floor
    assert not np.isfinite(heights[2]) and not np.isfinite(heights[3])
    texts = [text.get_text() for text in ax.texts]
    assert "NaN" in texts and "inf" in texts


def test_phi_numerics_panel_distinguishes_nonfinite_from_zero_residuals() -> None:
    keys = ("numerical_holonomy", "numerical_cocycle", "inverse_consistency")
    texts: Dict[str, List[str]] = {}
    for label, value in (("nan", float("nan")), ("zero", 0.0)):
        flat = {f"{key}_{precision}_rel": value for key in keys for precision in ("fp32", "fp64")}
        fig = figs.plot_phi_numerics_reference({"flatness": flat, "chart": {}})
        texts[label] = [text.get_text() for text in fig.axes[0].texts]
    assert texts["nan"] != texts["zero"]
    assert any("non-finite" in text for text in texts["nan"])
    assert any("zero" in text for text in texts["zero"]) and not any("non-finite" in t for t in texts["zero"])


def test_relative_deviation_counts_one_sided_nonfinite_entries_as_infinite() -> None:
    from vfe3.viz.extract import _relative_deviation
    inf, nan = float("inf"), float("nan")
    reference = torch.tensor([1.0, 2.0, inf], dtype=torch.float64)          # the last entry is masked in both
    assert _relative_deviation(reference, torch.tensor([1.0, 2.0, inf], dtype=torch.float64)) == 0.0
    assert _relative_deviation(reference, torch.tensor([1.0, inf, inf], dtype=torch.float64)) == inf
    assert _relative_deviation(reference, torch.tensor([nan, 2.0, inf], dtype=torch.float64)) == inf
    masked = torch.tensor([nan, inf], dtype=torch.float64)
    assert math.isnan(_relative_deviation(masked, masked))


@pytest.mark.parametrize("mode,check", [
    ("one_inf", lambda value: value == float("inf")),
    ("moved_nan", lambda value: value == float("inf")),       # finite base, NaN moved: one-sided
    ("all_nan", lambda value: math.isnan(value)),             # no comparable entry in any draw
], ids=("one_moved_entry_inf", "moved_mean_nan", "base_and_moved_nan"))
def test_equivariance_witness_reports_a_nonfinite_moved_evaluation(
        monkeypatch: pytest.MonkeyPatch, mode: str, check: Callable[[float], bool]) -> None:
    import warnings

    import vfe3.gradients.relational as relational
    from tests.profiles import test_config
    from vfe3.model.model import VFEModel
    from vfe3.viz import extract

    cfg = test_config(attention_mode="projected_predictive", n_layers=1, max_seq_len=6)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 6), generator=torch.Generator().manual_seed(2))
    original = relational.relational_predictive_readout
    calls = {"n": 0}

    def moved_nonfinite(*args: Any, **kwargs: Any) -> tuple:
        mean, cov = original(*args, **kwargs)
        index, calls["n"] = calls["n"], calls["n"] + 1
        if index > 0 or mode == "all_nan":                              # call 0 is the base evaluation
            mean = mean.clone()
            if mode == "one_inf":
                mean[..., 0, 0] = math.inf
            else:
                mean.fill_(math.nan)
        return mean, cov

    monkeypatch.setattr(relational, "relational_predictive_readout", moved_nonfinite)
    residual = extract.projected_gauge_equivariance(model, tokens, n_draws=3)["residuals"]["readout_mean"][0]
    assert check(residual)


def test_ablation_gauge_residual_keeps_a_nan_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    monkeypatch.setattr(ablation, "readout_state", lambda model, token_ids: {})
    monkeypatch.setattr(ablation, "projected_gauge_equivariance", lambda model, token_ids: {
        "residuals": {"routing_score": [1e-15, float("nan")], "readout_mean": [2e-16, 3e-16]}})
    out = ablation._cell_diagnostics(object(), None, [(torch.zeros(1, 4, dtype=torch.long),)], torch.device("cpu"))
    assert math.isnan(out["gauge_resid_in"])                            # was 1e-15: max(1e-15, nan) is 1e-15


# --------------------------------------------------------------------------- viz-figures-7

def test_reliability_bars_are_at_most_one_bin_wide_and_never_overlap() -> None:
    rel = [{"conf": c, "acc": a, "frac": f} for c, a, f in
           ((0.05, 0.04, 0.35), (0.12, 0.10, 0.25), (0.19, 0.18, 0.2), (0.26, 0.25, 0.12), (0.33, 0.35, 0.08),
            (0.39, 0.41, 0.0), (0.40, 0.42, 0.0))]
    fig = figs.plot_reliability_diagram(rel, n_bins=15)
    bars = sorted((patch for patch in fig.axes[0].patches), key=lambda patch: patch.get_x())[:len(rel)]
    widths = [bar.get_width() for bar in bars]
    assert max(widths) <= 1.0 / 15.0 + 1e-12
    assert all(bars[i].get_x() + widths[i] <= bars[i + 1].get_x() + 1e-12 for i in range(len(bars) - 1))
    assert all(0.0 <= bar.get_x() and bar.get_x() + bar.get_width() <= 1.0 for bar in bars)


# --------------------------------------------------------------------------- viz-pipeline-4

def test_token_label_falls_back_to_the_id_without_the_arabic_shaping_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    real_import = builtins.__import__

    def without_viz_extra(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in ("arabic_reshaper", "bidi"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_viz_extra)
    assert viz_text.token_label(23525, decode=lambda ids: "ال") == "23525"


def test_token_labels_escape_dollar_signs_and_render_as_ticks() -> None:
    decoded = {13702: "$$", 32382: " $$", 34206: "#$#$", 36737: "$$$$", 7: "$"}
    labels = [viz_text.token_label(tid, decode=lambda ids, t=tid: decoded[t]) for tid in decoded]
    assert labels[0] == r"\$\$" and labels[-1] == r"\$"
    fig, ax = plt.subplots()
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.canvas.draw()                                                   # unescaped pairs raised ValueError here
    joined = figs._distinct_token_labels(lambda ids: decoded.get(int(ids[0]), "x"), [13702, 13702])
    assert all("$" not in label.replace(r"\$", "") for label in joined)


# --------------------------------------------------------------------------- viz-figures-8

def test_log_axis_trajectory_trend_is_the_geometric_rolling_mean() -> None:
    rng = np.random.default_rng(0)
    steps = np.arange(1, 61) * 500.0
    ppl = 30.0 + 20000.0 * np.exp(-steps / 1500.0) * np.exp(0.05 * rng.standard_normal(60))
    fig = figs.plot_trajectory(ppl, steps, logy=True, smooth=5)
    np.testing.assert_allclose(fig.axes[0].lines[1].get_ydata(), figs._log_rolling_mean(ppl, 5))


def test_series_panel_trend_is_geometric_on_a_log_axis_and_index_columns_stay_unsmoothed() -> None:
    rng = np.random.default_rng(1)
    steps = np.arange(1, 61) * 500.0
    history = {"step": steps, "norm": np.exp(rng.standard_normal(60)),
               "layer": np.where(np.arange(60) % 2 == 0, 3.0, 9.0)}
    fig, ax = plt.subplots()
    figs._series_panel(ax, history, steps, [("norm", "norm", "#000000")], logy=True)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), figs._log_rolling_mean(history["norm"], 5))
    fig, ax = plt.subplots()
    figs._series_panel(ax, history, steps, [("layer", "layer", "#000000")], smooth=False)
    drawn = np.concatenate([np.asarray(line.get_ydata(), float) for line in ax.lines])
    assert set(np.unique(drawn)) <= {3.0, 9.0}
    location = next(panel for panel in figs._readout_panels(False, None) if panel["title"] == "Routing cap location")
    assert location.get("smooth") is False


def test_grad_norm_decomposition_trend_is_geometric_and_short_series_are_not_smoothed() -> None:
    rng = np.random.default_rng(2)
    steps = np.arange(1, 81) * 100.0
    grad = np.exp(0.1 * rng.standard_normal(80))
    grad[40] = 10.0
    fig = figs.plot_grad_norm_decomposition({"step": steps, "grad_norm_mu": grad})
    np.testing.assert_allclose(fig.axes[0].lines[1].get_ydata(), figs._log_rolling_mean(grad, 5))
    fig = figs.plot_grad_norm_decomposition({"step": steps[:10], "grad_norm_mu": grad[:10]})
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), grad[:10])


def test_short_eval_series_are_drawn_point_by_point_in_the_free_energy_relationship_figures() -> None:
    rng = np.random.default_rng(3)
    eval_steps = np.arange(1, 11) * 3000.0
    total = np.linspace(40.0, 25.0, 10) + rng.standard_normal(10)
    history = {"step": eval_steps, "self_coupling": total, "belief_coupling": np.zeros(10),
               "attention_entropy": np.zeros(10), "val_ce": np.linspace(6.0, 3.4, 10)}
    for builder in (figs.plot_canonical_free_energy_relationship, figs.plot_free_energy_relationship):
        ax = builder(history, lambda_beta=1.0).axes[0]
        bold = [line for line in ax.lines if line.get_linewidth() > 1.0][0].get_ydata()
        np.testing.assert_allclose(bold, total)


def test_log_axis_curve_bands_stay_positive() -> None:
    steps = np.arange(10.0)
    mean, sd = np.full(10, 1.0), np.full(10, 1.5)                       # heavy tailed: mean - sd < 0
    fig = figs.plot_curve_band(steps, mean, sd, logy=True)
    grid = figs.plot_curve_band_grid([{"steps": steps, "mean": mean, "sd": sd, "title": "grad_norm", "logy": True}])
    for ax in (fig.axes[0], grid.axes[0]):
        assert ax.collections
        for band in ax.collections:
            vertices = np.concatenate([path.vertices for path in band.get_paths()])
            assert np.all(vertices[:, 1] > 0.0)


# --------------------------------------------------------------------------- viz-figures-4

def test_saving_a_heatmap_adds_no_draw_pass_beyond_savefig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real_draw = matplotlib.figure.Figure.draw

    def counting_draw(self: matplotlib.figure.Figure, renderer: Any) -> Any:
        calls["n"] += 1
        return real_draw(self, renderer)

    monkeypatch.setattr(matplotlib.figure.Figure, "draw", counting_draw)
    beta = _causal_maps((8, 8)).numpy()
    figs.plot_attention_heatmap(beta, path=str(tmp_path / "saved.png"))
    through_save = calls["n"]
    calls["n"] = 0
    fig = figs.plot_attention_heatmap(beta)
    fig.savefig(str(tmp_path / "plain.png"))
    assert through_save == calls["n"]
    # The skipped pass changed nothing: a figure laid out beforehand saves the same bytes.
    fig = figs.plot_attention_heatmap(beta)
    fig.draw_without_rendering()
    figs._save(fig, str(tmp_path / "drawn.png"))
    assert (tmp_path / "drawn.png").read_bytes() == (tmp_path / "saved.png").read_bytes()


def test_tidy_figure_still_places_an_overlapping_legend() -> None:
    fig, ax = plt.subplots()
    x = np.linspace(0.0, 1.0, 50)
    for k in range(4):
        ax.plot(x, x + 0.01 * k, label=f"curve {k}")                    # the curves end in the upper right
    ax.legend(loc="upper right")
    figs.tidy_figure(fig)
    assert ax.get_legend()._loc != 1                                     # moved off the upper-right curves


# --------------------------------------------------------------------------- viz-pipeline-6

def test_template_umap_embeds_through_one_isolated_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.viz import feature_figures

    calls = []

    def fake_embed(features: Any, *, n_neighbors: int, seed: int, worker: Any = None, **kwargs: Any) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        calls.append((array.shape, n_neighbors, seed, worker))
        return np.stack([np.arange(array.shape[0], dtype=float), np.zeros(array.shape[0])], axis=1)

    monkeypatch.setattr(feature_figures, "umap_embed", fake_embed)
    monkeypatch.setitem(sys.modules, "umap", None)                      # an in-process umap-learn import fails
    rng = np.random.default_rng(4)
    rows = [{"layer": layer, "nu": rng.standard_normal((6, 4)), "usage": np.linspace(0.3, 0.05, 6),
             "codebook": [{"template": 0, "usage": 0.3, "top_tokens": [(5, 3)]}]} for layer in (0, 1)]
    feature_figures.plot_template_umap({"layers": rows}, seed=7)
    assert [call[1:3] for call in calls] == [(15, 7), (15, 7)]
    assert calls[0][3] is not None and calls[0][3] is calls[1][3]


# --------------------------------------------------------------------------- viz-figures-10

def test_figure_skipped_contract_names_the_builders_that_still_draw_a_reason() -> None:
    doc = " ".join(figs.FigureSkipped.__doc__.split())
    assert "no longer drawn" not in doc
    fig = figs.plot_context_sensitivity({"stages": {}})
    assert isinstance(fig, matplotlib.figure.Figure)
    fig = figs.plot_beta_channel_decomposition({"available": False, "reason": "no softmax"})
    assert isinstance(fig, matplotlib.figure.Figure)
    for name in ("plot_context_sensitivity", "plot_beta_channel_decomposition", "plot_ppl_offset",
                 "_history_dashboard"):
        assert name in doc
