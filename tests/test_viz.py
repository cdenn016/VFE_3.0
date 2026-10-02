from tests.profiles import test_config
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vfe3.viz.figures import (
    _fe_terms,
    _fit_power_law,
    attention_graph,
    get_figure,
    plot_ablation_forest,
    plot_attention_graph,
    plot_attention_grid,
    plot_attention_heatmap,
    plot_attention_structure,
    plot_belief_spectrum,
    plot_belief_trajectories,
    plot_capacity_scaling,
    plot_covariance_ellipses,
    plot_estep_capacity,
    plot_estep_convergence,
    plot_free_energy_codescent,
    plot_free_energy_decomposition,
    plot_gauge_equivariance,
    plot_grad_norm_decomposition,
    plot_estep_grad_norm_decomposition,
    plot_holonomy_curvature,
    plot_ln3_symmetry_breaking,
    plot_lr_grid_heatmap,
    plot_kmup_stability,
    plot_numerical_trust,
    plot_per_layer_diagnostics,
    plot_pareto_frontier,
    plot_ppl_offset,
    plot_scaling_routes,
    plot_spd_ellipses,
    plot_trajectory,
    plot_vocab_calibration,
    plot_vocab_confusion,
    plot_vocab_probability_heatmap,
    plot_decode_readout,
    set_publication_style,
)
from vfe3.model.model import VFEModel
from vfe3.viz import extract
from vfe3.viz import figures as figures_mod
from vfe3.viz.text import installed_multilingual_sans_serif


def _saved_nonempty(path):
    return path.exists() and path.stat().st_size > 0


def test_atomic_save_preserves_existing_png_when_renderer_fails_mid_write(tmp_path):
    class PartialWriteFailure:
        def savefig(self, path):
            Path(path).write_bytes(b"PARTIAL")
            raise RuntimeError("renderer failed mid-write")

    target = tmp_path / "publication.png"
    target.write_bytes(b"SENTINEL")

    with pytest.raises(RuntimeError, match="failed mid-write"):
        figures_mod._save(PartialWriteFailure(), str(target))

    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".publication.*.tmp.png"))


def test_atomic_save_rejects_empty_renderer_output_without_replacing_existing_png(tmp_path):
    class EmptyWrite:
        def savefig(self, path):
            Path(path).write_bytes(b"")

    target = tmp_path / "publication.png"
    target.write_bytes(b"SENTINEL")

    with pytest.raises(RuntimeError, match="nonempty temporary output"):
        figures_mod._save(EmptyWrite(), str(target))

    assert target.read_bytes() == b"SENTINEL"
    assert not list(tmp_path.glob(".publication.*.tmp.png"))


def test_set_publication_style_runs():
    set_publication_style()
    assert plt.rcParams["savefig.dpi"] == 300
    expected = installed_multilingual_sans_serif()
    assert list(plt.rcParams["font.family"]) == expected
    assert list(plt.rcParams["font.sans-serif"]) == expected


def test_publication_style_renders_japanese_without_missing_glyph_warning():
    set_publication_style()
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "日本語")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font.*", category=UserWarning)
        fig.canvas.draw()
    plt.close(fig)


def test_publication_style_does_not_request_uninstalled_font_families(caplog):
    caplog.set_level(logging.WARNING, logger="matplotlib.font_manager")
    set_publication_style()
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "日本語 and العربية")
    fig.canvas.draw()
    plt.close(fig)

    assert "Font family" not in caplog.text
    assert "not found" not in caplog.text


def test_token_display_wiring_preserves_japanese_and_shapes_arabic():
    # The Arabic half needs the optional 'viz' extras (arabic-reshaper + python-bidi), which are
    # absent here; the wiring it checks is exercised by the Japanese half in _tok_label's tests.
    pytest.importorskip("arabic_reshaper")
    assert figures_mod._tok_label(lambda ids: "日本語", 1) == "日本語"
    assert figures_mod._tok_label(lambda ids: "العربية", 2) != "العربية"
    assert figures_mod._tok_label(lambda ids: "�", 3) == "3"


def test_finite_xlim_pads_one_point_without_warning():
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert figures_mod._set_finite_xlim(ax, np.array([15000.0]))
    lo, hi = ax.get_xlim()
    assert lo < 15000.0 < hi
    plt.close(fig)


def test_finite_xlim_preserves_distinct_endpoints():
    fig, ax = plt.subplots()
    assert figures_mod._set_finite_xlim(ax, np.array([100.0, 200.0]))
    assert ax.get_xlim() == pytest.approx((100.0, 200.0))
    plt.close(fig)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: figures_mod.plot_trajectory([1.0], steps=[15000.0]),
        lambda: figures_mod.plot_grad_norm_decomposition({
            "step": [15000.0], "grad_norm_mu": [1.0],
        }),
        lambda: figures_mod._history_dashboard(
            {"step": [15000.0], "metric": [1.0]},
            [{"title": "metric", "ylabel": "value",
              "series": [("metric", "metric", "#0072B2")]}],
            "one-point dashboard",
            None,
        ),
    ],
    ids=("trajectory", "gradient", "dashboard"),
)
def test_one_point_step_figures_do_not_request_identical_limits(factory):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Attempting to set identical low and high xlims.*",
            category=UserWarning,
        )
        fig = factory()
    plt.close(fig)


def test_offset_fit_requires_four_distinct_sizes():
    # Four observations at only three distinct sizes do not identify the
    # three-parameter E + A*N**(-alpha) model with residual degrees of freedom.
    N = np.array([1.0e2, 1.0e3, 1.0e4, 1.0e4])
    L = np.array([4.7, 4.2, 3.9, 3.8])

    fit = _fit_power_law(N, L, with_offset=True)

    assert fit["form"] == "power_law_fallback_underdetermined"
    assert fit["n_points"] == 4
    assert fit["n_distinct_sizes"] == 3

    plain = _fit_power_law(N, L, with_offset=False)
    assert plain["form"] == "power_law"
    assert plain["n_distinct_sizes"] == 3


def test_offset_solver_failure_is_labeled(monkeypatch):
    scipy_optimize = pytest.importorskip("scipy.optimize")

    def _fail_solver(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(scipy_optimize, "curve_fit", _fail_solver)
    N = np.array([1.0e2, 1.0e3, 1.0e4, 1.0e5])
    L = np.array([4.7, 4.2, 3.9, 3.8])

    fit = _fit_power_law(N, L, with_offset=True)

    assert fit["form"] == "power_law_fallback_solver"
    assert fit["E"] == 0.0
    assert fit["n_distinct_sizes"] == 4


def test_ppl_figures_label_underdetermined_fallbacks():
    # The live grow_K route has three widths.  It may show a fallback curve,
    # but neither figure may call that two-parameter curve an offset law or
    # report a fitted irreducible floor E.
    points = [
        {"embed_dim": K, "ppl_mean": ppl, "ppl_sem": 0.1, "n_seeds": 3}
        for K, ppl in ((60, 95.0), (80, 90.0), (100, 87.0))
    ]
    fig = plot_ppl_offset(points)
    offset_text = " ".join(text.get_text() for text in fig.axes[0].get_legend().get_texts())
    assert "fallback" in offset_text.lower()
    assert "offset fit" not in offset_text.lower()
    assert "E=" not in offset_text
    assert "offset law" not in fig.axes[0].get_title().lower()
    plt.close(fig)

    routes = {"grow_K": [
        {"embed_dim": p["embed_dim"], "ppl_mean": p["ppl_mean"],
         "ppl_sem": p["ppl_sem"], "n": p["n_seeds"]}
        for p in points
    ]}
    fig = plot_kmup_stability(routes)
    kmup_text = " ".join(text.get_text() for text in fig.axes[0].get_legend().get_texts())
    assert "fallback" in kmup_text.lower()
    assert "offset" not in kmup_text.lower()
    assert "E=" not in kmup_text
    plt.close(fig)


def test_scaling_routes_validates_supplied_weights_after_filtering():
    points = [
        {"route": "a", "n_params": 100.0, "ce_seeds": [4.0, 4.2]},
        {"route": "a", "n_params": float("nan"), "ce_seeds": [3.9, 4.1]},
    ]
    with pytest.raises(ValueError, match="expected 1 after point filtering"):
        plot_scaling_routes(
            points,
            with_offset=False,
            weights_by_route={"a": np.ones(2)},
        )


def test_attention_graph_structure():
    beta = torch.softmax(torch.randn(5, 5), dim=-1)
    G = attention_graph(beta, threshold=0.0)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() <= 5 * 4                        # no self-loops


def test_plot_attention_graph_and_heatmap_save(tmp_path):
    beta = torch.softmax(torch.randn(6, 6), dim=-1)
    pg = tmp_path / "g.png"; ph = tmp_path / "h.png"
    fig1 = plot_attention_graph(beta, path=str(pg)); plt.close(fig1)
    fig2 = plot_attention_heatmap(beta, path=str(ph)); plt.close(fig2)
    assert _saved_nonempty(pg) and _saved_nonempty(ph)


def test_plot_attention_heatmap_channel_colors(tmp_path):
    # The belief (beta, magma) and model (gamma, viridis) channels render as distinct-coloured
    # per-head heatmaps through the same plotter -- cmap/symbol select the channel identity.
    beta = torch.softmax(torch.randn(6, 6), dim=-1)
    pb = tmp_path / "beta.png"; pg = tmp_path / "gamma.png"
    fig1 = plot_attention_heatmap(beta, cmap="magma", symbol=r"\beta", path=str(pb)); plt.close(fig1)
    fig2 = plot_attention_heatmap(beta, cmap="viridis", symbol=r"\gamma", path=str(pg)); plt.close(fig2)
    assert _saved_nonempty(pb) and _saved_nonempty(pg)


def test_plot_covariance_ellipses_and_trajectory_save(tmp_path):
    mu = torch.randn(8, 3); sigma = torch.rand(8, 3) + 0.5
    pe = tmp_path / "ell.png"; pt = tmp_path / "traj.png"
    fig1 = plot_covariance_ellipses(mu, sigma, path=str(pe)); plt.close(fig1)
    fig2 = plot_trajectory([3.0, 2.1, 1.7, 1.5], ylabel="CE", path=str(pt)); plt.close(fig2)
    assert _saved_nonempty(pe) and _saved_nonempty(pt)


def test_plot_trajectory_options_save(tmp_path):
    # Real step x-axis + log y + smoothing + annotations on a long heavy-tailed series.
    steps = np.arange(100, 100 * 401, 100)                         # 400 points, real "step" values
    vals = 300.0 * np.exp(-steps / 4e4) + np.random.default_rng(0).random(steps.size)
    p_log = tmp_path / "logy.png"; p_med = tmp_path / "median.png"
    fig = plot_trajectory(vals, steps, ylabel="ppl", logy=True, smooth=15, annotate="min",
                          annotate_final=True, path=str(p_log)); plt.close(fig)
    # median_line + 'max' annotate on a positive series (the holonomy treatment).
    fig = plot_trajectory(np.abs(vals) + 1e-4, steps, logy=True, median_line=True, annotate="max",
                          path=str(p_med)); plt.close(fig)
    assert _saved_nonempty(p_log) and _saved_nonempty(p_med)


def test_plot_attention_grid_saves_4d_and_degenerate(tmp_path):
    # (L, H, N, N) grid plus the L==1/H==1 degenerate inputs the squeeze=False guard must handle.
    maps = torch.softmax(torch.randn(2, 3, 5, 5), dim=-1)          # 2 layers x 3 heads
    p4 = tmp_path / "grid.png"; p3 = tmp_path / "g3.png"; p2 = tmp_path / "g2.png"
    fig = plot_attention_grid(maps, path=str(p4)); plt.close(fig)
    fig = plot_attention_grid(maps[0], path=str(p3)); plt.close(fig)   # (H, N, N) -> one layer
    fig = plot_attention_grid(maps[0, 0], path=str(p2)); plt.close(fig)  # (N, N) -> one panel
    assert _saved_nonempty(p4) and _saved_nonempty(p3) and _saved_nonempty(p2)


def test_figure_registry():
    assert callable(get_figure("attention_heatmap"))
    assert callable(get_figure("attention_grid"))
    with pytest.raises(KeyError):
        get_figure("not_a_figure")


# --- publication figures (smoke: build synthetic inputs, save a nonempty PNG) ---

def _T():
    return 5, 6, 4, 2          # T, N, K, H


def _fe_hist(n=30):
    s = np.arange(1, n + 1) * 100
    return {"step": s, "self_coupling": np.full(n, 140.0), "belief_coupling": np.linspace(30, 20, n),
            "attention_entropy": np.linspace(34, 13, n), "val_ce": np.linspace(5.7, 4.27, n)}


def test_plot_free_energy_decomposition_saves(tmp_path):
    p = tmp_path / "decomp.png"
    fig = plot_free_energy_decomposition(
        _fe_hist(), lambda_beta=1.0, path=str(p))
    labels = " ".join(text.get_text() for axis in fig.axes for text in axis.get_yticklabels())
    assert "KL" not in labels
    plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_free_energy_codescent_saves(tmp_path):
    p = tmp_path / "codescent.png"
    # per-row learned lambda_beta vector is accepted as well as a scalar
    fig = plot_free_energy_codescent(_fe_hist(), lambda_beta=np.ones(30), path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_fe_terms_total_sums_components_and_excludes_ce():
    h = _fe_hist(20)
    step, comps, total, ce = _fe_terms(h, lambda_beta=1.0)
    assert [k for k, _ in comps] == ["self", "belief", "attention_entropy"]
    assert np.allclose(total, sum(c for _, c in comps))           # F total is the sum of the shown bars
    assert np.allclose(ce, np.asarray(h["val_ce"], dtype=float))  # CE returned separately, not summed in
    h2 = dict(h); h2["val_ce"] = np.asarray(h["val_ce"], dtype=float) + 1000.0
    _, _, total2, _ = _fe_terms(h2, lambda_beta=1.0)
    assert np.allclose(total, total2)                             # total is independent of CE (no data term in F)


def test_fe_terms_belief_only_and_entropy_gate():
    _, comps, _, _ = _fe_terms(_fe_hist(10), lambda_beta=1.0)
    assert [k for k, _ in comps] == ["self", "belief", "attention_entropy"]
    _, comps_off, _, _ = _fe_terms(_fe_hist(10), lambda_beta=1.0, include_attention_entropy=False)
    assert [k for k, _ in comps_off] == ["self", "belief"]        # entropy gated out of F, matching the column


def test_plot_grad_norm_decomposition_saves(tmp_path):
    n = 60
    rng = np.random.default_rng(0)
    hist = {
        "step":            list(range(1, n + 1)),
        "grad_norm_mu":    list(np.abs(rng.standard_normal(n)) * 1e-1 + 1e-2),   # spans orders of magnitude
        "grad_norm_sigma": list(np.abs(rng.standard_normal(n)) * 1e-3 + 1e-4),
        "grad_norm_phi":   list(np.abs(rng.standard_normal(n)) * 1e-2 + 1e-3),
    }
    hist["grad_norm_mu"][5]  = float("nan")                        # NaN dropped by the finite filter
    hist["grad_norm_phi"][7] = 0.0                                 # non-positive dropped on the log axis
    p = tmp_path / "grad_decomp.png"
    fig = plot_grad_norm_decomposition(hist, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)
    assert callable(get_figure("grad_norm_decomposition"))


def test_plot_grad_norm_decomposition_partial_columns(tmp_path):
    # a run may log only a subset of the per-group norms; the figure still renders
    hist = {"step": list(range(1, 21)), "grad_norm_sigma": list(np.linspace(1e-3, 1e-2, 20))}
    p = tmp_path / "grad_decomp_partial.png"
    fig = plot_grad_norm_decomposition(hist, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_estep_grad_norm_decomposition_saves(tmp_path):
    # the E-step (inference) belief-gradient decomposition: separate figure from the M-step one,
    # reads the estep_grad_norm_* columns. phi may be identically 0 (e_phi_lr=0) -> dropped on log y.
    n = 60
    rng = np.random.default_rng(1)
    hist = {
        "step":                  list(range(1, n + 1)),
        "estep_grad_norm_mu":    list(np.abs(rng.standard_normal(n)) * 1e-1 + 1e-2),
        "estep_grad_norm_sigma": list(np.abs(rng.standard_normal(n)) * 1e-2 + 1e-3),
        "estep_grad_norm_phi":   [0.0] * n,                           # phi substep off -> all dropped
    }
    p = tmp_path / "estep_grad_decomp.png"
    fig = plot_estep_grad_norm_decomposition(hist, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)
    assert callable(get_figure("estep_grad_norm_decomposition"))


def test_plot_estep_convergence_saves(tmp_path):
    T, N, K, _ = _T()
    trace = {"mu": torch.randn(T, N, K).cumsum(0), "sigma": torch.rand(T, N, K) + 0.3,
             "phi": torch.randn(T, N, 2), "free_energy": torch.linspace(50, 30, T)}
    p = tmp_path / "f2.png"
    fig = plot_estep_convergence(trace, path=str(p))
    titles = [axis.get_title().lower() for axis in fig.axes]
    assert not any("convergence" in title or "fixed point" in title or "descent" in title
                   for title in titles)
    plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_ln3_symmetry_breaking_saves(tmp_path):
    _, N, K, H = _T()
    def arm():
        return {"step": np.arange(8), "val_ce": np.linspace(1.5, 1.1, 8),
                "omega": torch.randn(N, N, K, K), "beta": torch.softmax(torch.randn(H, N, N), dim=-1)}
    p = tmp_path / "f3.png"
    fig = plot_ln3_symmetry_breaking(arm(), arm(), period=3, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_belief_trajectories_saves(tmp_path):
    T, N, K, _ = _T()
    trace = {"mu": torch.randn(T, N, K).cumsum(0), "sigma": torch.rand(T, N, K) + 0.3, "phi": torch.randn(T, N, 2)}
    layer = {"d_ai": torch.tensor([0.0, 0.5, 0.9]), "effective_rank": torch.tensor([3.0, 2.5, 2.0])}
    p = tmp_path / "f4.png"
    fig = plot_belief_trajectories(trace, layer, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_token_category_helpers():
    from vfe3.viz.figures import _bpe_category, _funccontent_category
    assert _bpe_category(",") == 0          # punctuation
    assert _bpe_category(" 2014") == 1      # number (leading space, all-digit core)
    assert _bpe_category(" cat") == 2       # word-start lowercase
    assert _bpe_category(" Cat") == 3       # word-start Capitalized
    assert _bpe_category("ing") == 4        # continuation subword
    assert _bpe_category(" ") == 5          # whitespace/other
    assert _funccontent_category(" the") == 2   # function word
    assert _funccontent_category(" cat") == 3   # content word
    assert _funccontent_category(".") == 0      # punctuation
    assert _funccontent_category("42") == 1     # number


def test_plot_gauge_equivariance_saves(tmp_path):
    resid = {"residuals": {"routing_score": [1e-14, 3e-14], "routing_weights": [0.0, 1e-16],
                           "readout_mean": [2e-16, 4e-16], "readout_covariance": [8e-16, 1e-15]},
             "readout_mode": "predictive_content", "n_draws": 3, "unit_roundoff": 2.2e-16}
    p = tmp_path / "f6.png"
    fig = plot_gauge_equivariance(resid, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_gauge_head_specialization_is_retired():
    # It paired frame block h with head h, which no measurement supports; head_block_norms replaces it.
    assert not hasattr(figures_mod, "plot_gauge_head_specialization")
    with pytest.raises(KeyError):
        get_figure("gauge_head_specialization")
    assert callable(get_figure("head_block_norms"))


def test_plot_attention_structure_saves(tmp_path):
    _, N, _, H = _T()
    beta = torch.softmax(torch.randn(2, H, N, N), dim=-1)         # (L=2, H, N, N)
    p = tmp_path / "f8.png"
    fig = plot_attention_structure(beta, path=str(p))
    # the (layer, head) axes must KEEP their identity as L{l}H{h} -- NOT be renumbered 0..L*H-1.
    labs = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labs == [f"L{l}H{h}" for l in range(2) for h in range(H)]
    plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_attention_structure_single_layer_uses_head_labels(tmp_path):
    _, N, _, H = _T()
    beta = torch.softmax(torch.randn(H, N, N), dim=-1)           # (H, N, N) single layer
    p = tmp_path / "f8s.png"
    fig = plot_attention_structure(beta, path=str(p))
    labs = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labs == [f"h{h}" for h in range(H)]
    plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_per_layer_diagnostics_saves(tmp_path):
    L = 3
    keys = ("self_coupling", "belief_coupling", "attention_entropy", "total", "self_divergence",
            "holonomy_deviation", "holonomy_wilson", "gauge_trace_spread", "gauge_invariant_spread",
            "effective_rank", "attn_entropy", "belief_cond_median", "phi_norm_mean")
    per_layer = {k: [float(v) for v in torch.rand(L)] for k in keys}
    p = tmp_path / "f8b.png"
    fig = plot_per_layer_diagnostics(per_layer, path=str(p))
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [str(l) for l in range(L)]
    plt.close(fig)
    assert _saved_nonempty(p)
    assert get_figure("per_layer_diagnostics") is plot_per_layer_diagnostics


def test_plot_belief_spectrum_and_ellipses_save(tmp_path):
    _, N, K, _ = _T()
    sigma = torch.rand(N, K) + 0.3
    p1 = tmp_path / "f9a.png"; p2 = tmp_path / "f9b.png"
    fig = plot_belief_spectrum(sigma, eps=1e-6, sigma_max=5.0, path=str(p1)); plt.close(fig)
    full = torch.diag_embed(sigma)
    fig = plot_spd_ellipses(torch.randn(N, K), full, dims=(0, 1), path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_holonomy_curvature_saves(tmp_path):
    flat = {"per_triple": torch.rand(80) * 1e-6, "span": torch.randint(1, 6, (80,)).float()}
    regime = {"per_triple": torch.rand(80) + 0.1, "span": torch.randint(1, 6, (80,)).float()}
    p = tmp_path / "f10.png"
    fig = plot_holonomy_curvature(flat, regime, curvature=torch.rand(6, 6), path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_holonomy_curvature_title_follows_regime(tmp_path):
    # A flat (Regime-I) run has no regime_ii data: the curvature-field panel must NOT claim "regime II".
    flat = {"per_triple": torch.rand(40) * 1e-6, "span": torch.randint(1, 6, (40,)).float()}
    fig = plot_holonomy_curvature(flat, None, curvature=torch.rand(6, 6), path=str(tmp_path / "cf_flat.png"))
    titles = [a.get_title() for a in fig.axes]
    assert "Curvature field (flat)" in titles, titles
    assert "Curvature field (comparison)" not in titles, titles
    plt.close(fig)
    # With comparison data present the field is genuine curvature -> the title follows the run.
    regime = {"per_triple": torch.rand(40) + 0.1, "span": torch.randint(1, 6, (40,)).float()}
    fig = plot_holonomy_curvature(flat, regime, curvature=torch.rand(6, 6), path=str(tmp_path / "cf_rii.png"))
    assert "Curvature field (comparison)" in [a.get_title() for a in fig.axes]
    plt.close(fig)


def test_per_layer_belief_geometry_labels_cond_median(tmp_path):
    # The twin-axis condition-number marker must be named in the legend so it cannot be misread as a
    # second attention-entropy series on the left rank/nats axis.
    keys = ("self_coupling", "belief_coupling", "attention_entropy", "total",
            "holonomy_deviation", "holonomy_wilson", "gauge_trace_spread", "gauge_invariant_spread",
            "phi_norm_mean", "effective_rank", "attn_entropy", "belief_cond_median")
    per_layer = {k: [float(v) for v in torch.rand(3)] for k in keys}
    fig = plot_per_layer_diagnostics(per_layer, path=str(tmp_path / "bg.png"))
    labels = []
    for a in fig.axes:
        leg = a.get_legend()
        if leg is not None:
            labels += [t.get_text() for t in leg.get_texts()]
    assert any("cond median" in lab for lab in labels), labels
    plt.close(fig)


def test_trajectory_max_annotation_keeps_small_values(tmp_path):
    # Regression: a holonomy-scale (~1e-4) extremum must render its real value in the 'max' callout,
    # not round to "0.0" the way the old ``:.1f`` format did.
    vals = [3.0e-5, 2.3e-4, 1.1e-4, 1.7e-4]
    fig = plot_trajectory(vals, [0, 1500, 3000, 4500], logy=True, annotate="max",
                          path=str(tmp_path / "holo.png"))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any(t.startswith("max ") and "0.0002" in t for t in texts), texts
    assert not any(t.startswith("max 0.0\n") for t in texts), texts
    plt.close(fig)


def test_plot_capacity_scaling_and_estep_capacity_save(tmp_path):
    scaling = {"embed_dim": {"x": np.array([20, 40, 64]), "bits_per_token": np.array([7.5, 7.0, 6.8]),
                             "lo": np.array([7.4, 6.9, 6.7]), "hi": np.array([7.6, 7.1, 6.9])},
               "n_layers": {"x": np.array([1, 2, 3]), "bits_per_token": np.array([7.5, 7.1, 7.0])}}
    p1 = tmp_path / "f11a.png"; p2 = tmp_path / "f11b.png"
    fig = plot_capacity_scaling(scaling, path=str(p1)); plt.close(fig)
    fig = plot_estep_capacity(np.array([1, 2, 4]), np.array([7.5, 7.1, 6.9]), np.array([100., 90., 85.]),
                              n_params=12345, wall_time=np.array([10., 18., 33.]), path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_pareto_frontier_saves(tmp_path):
    points = {"bits_per_token": np.array([7.5, 7.0, 6.8, 7.2]), "n_params": np.array([1e4, 4e4, 9e4, 2e4]),
              "wall_time": np.array([10., 30., 60., 18.])}
    p = tmp_path / "f11c.png"
    fig = plot_pareto_frontier(points, path=str(p)); plt.close(fig)
    assert _saved_nonempty(p)


def test_plot_ablation_forest_and_lr_grid_save(tmp_path):
    rows = [{"label": "frozen gauge", "delta": 0.4, "lo": 0.3, "hi": 0.5},
            {"label": "surrogate F", "delta": 0.1, "lo": 0.05, "hi": 0.15},
            {"label": "uniform prior", "delta": 0.25, "lo": 0.2, "hi": 0.3}]
    grid = {"x": np.array([0.01, 0.02, 0.03]), "y": np.array([0.001, 0.002]),
            "z": np.array([[190., 185., 188.], [186., 180., 184.]]),
            "xlabel": "m_p_mu_lr", "ylabel": "m_p_sigma_lr", "baseline": (0.02, 0.002)}
    p1 = tmp_path / "f12a.png"; p2 = tmp_path / "f12b.png"
    fig = plot_ablation_forest(rows, path=str(p1))
    # PB-07: the forest x-axis is delta BITS/TOKEN (paired per-token nats / ln 2), not the
    # character-corrected BPC of the scaling figures -- pin the relabel.
    assert fig.axes[0].get_xlabel() == r"$\Delta$ bits/token vs full model"
    plt.close(fig)
    fig = plot_lr_grid_heatmap(grid, path=str(p2)); plt.close(fig)
    assert _saved_nonempty(p1) and _saved_nonempty(p2)


def test_plot_numerical_trust_saves(tmp_path):
    guard = {"sigma_floor_frac": 0.0, "sigma_ceil_frac": 0.0, "energy_klmax_frac": 0.0, "selfdiv_klmax_frac": 0.0}
    health = {"nan_mu": 0.0, "nan_sigma": 0.0, "nan_phi": 0.0, "nan_energy": 0.0, "nan_beta": 0.0, "max_condition": 12.0}
    # future_leakage is (L, H) (causal_sanity keeps the leading layer,head axes): the causal row
    # counts (layer, head) maps, so its denominator is L*H = 6. Guard rows without a recorded
    # denominator say so instead of guessing one.
    causal = {"future_leakage": torch.zeros(3, 2), "row_sum_error": torch.zeros(2), "active_set_slope": torch.ones(2)}
    p = tmp_path / "f13.png"
    fig = plot_numerical_trust(guard, health, causal, path=str(p))
    cells = {(r, c): cell.get_text().get_text()
             for table in fig.axes[0].tables for (r, c), cell in table.get_celld().items()}
    rows = {cells[(r, 0)]: (cells[(r, 1)], cells[(r, 2)]) for r in range(1, 1 + max(k[0] for k in cells))}
    assert rows["(layer, head) maps with future attention > $10^{-6}$"] == ("0", "6")
    assert rows[r"variance at the floor $\varepsilon$"] == ("n/a", "not recorded")
    plt.close(fig)
    assert _saved_nonempty(p)


# --- extractor replay fidelity (phi chart) ---

def _replay_model(**over):
    base = dict(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=2,
        gauge_group="block_glk",
        family="gaussian_full",
        pos_rotation="none",
        use_head_mixer=False, use_block_mlp=False, norm_type_block="none",
        decode_mode="full_chunked",
        pos_phi="none",
    )
    base.update(over)
    torch.manual_seed(31)
    model = VFEModel(test_config(**base))
    model.eval()
    return model


def _raw_forward_belief(model, tokens):
    capture = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    return capture["out"]


def test_phi_extractors_remain_unchanged():
    from vfe3.geometry.transport import compute_transport_operators

    model = _replay_model()
    tokens = torch.tensor([[0, 1, 2, 3]])
    forward = _raw_forward_belief(model, tokens)
    state = extract.readout_state(model, tokens)
    built = compute_transport_operators(state["phi"].unsqueeze(0), model.group)

    assert torch.allclose(state["mu"], forward.mu[0], atol=1e-5, rtol=1e-5)
    assert torch.equal(state["exp_phi"], built["exp_phi"][0])
    omega = state["omega"].to_dense_omega()
    expected = built["Omega"][0]
    diagonal = torch.eye(tokens.shape[1], dtype=torch.bool, device=omega.device)
    # The dense builder declares self-transport exactly; compact expansion computes exp(phi)exp(-phi).
    identity = torch.eye(model.cfg.embed_dim, dtype=expected.dtype, device=expected.device)
    assert torch.equal(expected[diagonal], identity.expand(tokens.shape[1], -1, -1))
    torch.testing.assert_close(omega[~diagonal], expected[~diagonal], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(omega[diagonal], expected[diagonal], atol=1e-5, rtol=1e-5)


# --- vocabulary next-token probability figures (extractors + the four arm-list plotters) ---

def _vocab_model(**kw):
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.1)
    base.update(kw)
    torch.manual_seed(0)
    return VFEModel(test_config(**base))


def _vocab_batches(bs=3, n=8, nb=2):
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 12, (bs, n), generator=g) for _ in range(nb)]


def _fake_decode(ids):                                            # leading-space word-ish gpt2-style tokens
    return " " + "".join(chr(97 + (int(i) % 26)) for i in ids)


def test_vocab_prediction_stats_shapes_and_invariants():
    st = extract.vocab_prediction_stats(_vocab_model(decode_mode="linear"), _vocab_batches(),
                                        max_rows=6, per_pos_k=2, max_positions=6)
    R, P = st["disp_probs"].shape
    assert R <= 6 and R == st["row_ids"].shape[0]
    assert P == st["disp_truth_row"].shape[0] == st["disp_target_ids"].shape[0]
    assert st["mean_pred_prob"].shape == (12,) and st["unigram"].shape == (12,)
    assert torch.isfinite(st["disp_probs"]).all()
    # mean over per-position softmaxes is a convex combination of distributions -> still sums to 1
    assert abs(float(st["mean_pred_prob"].sum()) - 1.0) < 1e-4
    assert abs(float(st["unigram"].sum()) - 1.0) < 1e-4
    assert st["mean_pred_entropy"] > 0.0 and st["unigram_entropy"] > 0.0
    assert st["true_ids"].shape == st["pred_ids"].shape


def test_decode_readout_present_only_for_linear_mode():
    assert extract.decode_readout(_vocab_model(decode_mode="diagonal_chunked")) is None
    ro = extract.decode_readout(_vocab_model(decode_mode="linear"), max_rows=5)
    assert ro is not None and ro["weight"].shape == (5, 4) and ro["row_ids"].shape == (5,)


def test_vocab_figures_single_and_two_arm_save(tmp_path):
    model = _vocab_model(decode_mode="linear")
    st = extract.vocab_prediction_stats(model, _vocab_batches(), max_rows=6, per_pos_k=2, max_positions=6)
    ro = extract.decode_readout(model, max_rows=6)
    one = [{**st, "label": "K4"}]
    two = [{**st, "label": "A"}, {**st, "label": "B"}]                            # the side-by-side comparison path
    cases = [
        ("ph1",  lambda p: plot_vocab_probability_heatmap(one, decode=_fake_decode, path=p)),
        ("ph2",  lambda p: plot_vocab_probability_heatmap(two, decode=_fake_decode, path=p)),
        ("cal",  lambda p: plot_vocab_calibration(two, decode=_fake_decode, path=p)),
        ("conf", lambda p: plot_vocab_confusion(two, decode=_fake_decode, path=p)),
        ("ro",   lambda p: plot_decode_readout([{**ro, "label": "K4"}], decode=_fake_decode, path=p)),
    ]
    for name, fn in cases:
        p = tmp_path / f"{name}.png"
        fig = fn(str(p)); plt.close(fig)
        assert _saved_nonempty(p)


def test_vocab_confusion_requires_decoder():
    st = extract.vocab_prediction_stats(_vocab_model(decode_mode="linear"), _vocab_batches(),
                                        max_rows=4, per_pos_k=2, max_positions=4)
    with pytest.raises(ValueError):
        plot_vocab_confusion([{**st, "label": "x"}], decode=None)


def test_vocab_figures_registered():
    for name in ("vocab_probability_heatmap", "vocab_calibration", "vocab_confusion", "decode_readout"):
        assert callable(get_figure(name))
