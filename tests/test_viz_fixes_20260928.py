r"""Training-time figure fixes of the k160 metrics audit (2026-09-28, Section 4 and Appendix B).

Every new or changed figure is driven from tiny synthetic ``metrics.csv`` / ``diagnostics/*.csv``
tables written here, every route gate is exercised, and each figure is checked against the page
geometry (at most 7.2 in wide, no text below 6 pt). CPU only, tiny models.
"""
from tests.profiles import test_config
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from vfe3.model.model import VFEModel
from vfe3.viz import extract, report
from vfe3.viz import figures as figs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_page(fig) -> None:
    r"""Width at most MAX_FIGURE_WIDTH_IN and every visible, nonempty text at least MIN_TEXT_PT."""
    width, _ = fig.get_size_inches()
    assert width <= figs.MAX_FIGURE_WIDTH_IN + 1e-9, width
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()
             and t.get_fontsize() < figs.MIN_TEXT_PT - 1e-9]
    assert not small, small


def _titles(fig) -> list:
    return [ax.get_title() for ax in fig.axes if ax.axison]


def _live_axes(fig) -> list:
    return [ax for ax in fig.axes if ax.axison and ax.get_label() != "<colorbar>"]


def _write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if (isinstance(v, float) and np.isnan(v)) else v for v in row])


def _write_metrics(path: Path, *, new_columns: bool = True) -> None:
    r"""Log rows every 10 steps; eval rows (val_ce, train sample) every 30 steps, NaN elsewhere."""
    header = ["step", "train_loss", "wall_clock_s", "corpus_pass", "val_ce"]
    if new_columns:
        header += ["tokens_seen", "train_sample_ce", "train_sample_ppl", "generalization_gap_fixed"]
    rows = []
    for step in range(10, 130, 10):
        is_eval = step % 30 == 0
        val = 5.0 - step / 100.0 if is_eval else float("nan")
        row = [step, 5.5 - step / 90.0, 2.0 * step, step / 50.0, val]
        if new_columns:
            train = val - 0.05 if is_eval else float("nan")
            row += [step * 64, train, float(np.exp(train)) if is_eval else float("nan"),
                    val - train if is_eval else float("nan")]
        rows.append(row)
    _write_csv(path, header, rows)


def _write_attention_stats(path: Path, steps=(30, 60, 90), n_layers=2, n_heads=3) -> None:
    rows = []
    for step in steps:
        for layer in range(n_layers):
            for head in range(n_heads):
                rows.append([step, layer, head, 0.4 + 0.1 * head, 3.0 + layer + head,
                             0.1 + 0.05 * layer, 0.6 + 0.01 * step / 30.0, 120])
    _write_csv(path, ["step", "layer", "head", "entropy_norm", "distance_mean", "self_weight",
                      "prior_overlap", "n_rows"], rows)


def _write_residual_ratio(path: Path, steps=(30, 60, 90), n_layers=3, ln=True) -> None:
    rows = []
    for step in steps:
        for layer in range(n_layers):
            q50 = 10.0 ** (1 + layer) * (1 + step / 300.0)
            rows.append([step, layer, 0.5 * q50, q50, 2.0 * q50,
                         0.01 * (layer + 1) if ln else float("nan")])
    _write_csv(path, ["step", "layer", "q10", "q50", "q90", "ln_rel_change_q50"], rows)


# ---------------------------------------------------------------------------
# (1) route gates
# ---------------------------------------------------------------------------

_ESTEP_FAMILY = {"iterate_trajectory", "belief_trajectories",
                 "estep_character", "estep_quality", "estep_grad_norm_decomposition",
                 "attention_estep"}
_SIGMA_FAMILY = {"belief_spectrum", "spd_ellipses", "belief_condition"}


def test_route_gates_flat_projected_frozen():
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     attention_mode="projected_predictive",
                     gauge_group="block_glk")
    reasons = figs.route_figure_skip_reasons(cfg)
    assert "flat" in reasons["holonomy_curvature"]
    assert _ESTEP_FAMILY <= set(reasons)
    assert "projected_predictive" in reasons["iterate_trajectory"]
    assert _SIGMA_FAMILY <= set(reasons)
    assert "frozen at encode" in reasons["belief_spectrum"]
    assert not figs.has_estep_iterates(cfg)


def test_route_gates_gradient_keeps_sigma_but_skips_legacy_objective_figures() -> None:
    cfg = SimpleNamespace(attention_mode="projected_gradient", n_e_steps=2,
                          skip_belief_sigma_update=False)
    reasons = figs.route_figure_skip_reasons(cfg)
    assert _ESTEP_FAMILY <= set(reasons)
    assert _SIGMA_FAMILY.isdisjoint(reasons)
    assert not figs.has_estep_iterates(cfg)


def test_route_gates_zero_estep_and_frozen_gradient() -> None:
    cfg = SimpleNamespace(attention_mode="projected_gradient", n_e_steps=0,
                          skip_belief_sigma_update=True)
    reasons = figs.route_figure_skip_reasons(cfg)
    assert "legacy shared-objective E-step probes do not apply" in reasons["iterate_trajectory"]
    assert _SIGMA_FAMILY <= set(reasons) and "holonomy_curvature" in reasons


def test_estep_probe_figures_skip_instead_of_drawing_placeholders(tmp_path):
    unavailable = {"available": False, "reason": "only the configured single readout is supported"}
    with pytest.raises(figs.FigureSkipped, match="single readout"):
        figs.plot_estep_character(unavailable, path=str(tmp_path / "c.png"))
    single = {"available": True, "depth_sweep_available": False,
              "points": [{"belief_depth": 1, "rel_displacement": 737.0}]}
    with pytest.raises(figs.FigureSkipped, match="residual_ratio"):
        figs.plot_estep_character(single, path=str(tmp_path / "c.png"))
    assert not list(tmp_path.iterdir())


def test_write_figure_logs_skip_reason_and_writes_nothing(tmp_path, caplog):
    def builder(path):
        plt.figure()
        raise figs.FigureSkipped("nothing to show on this route")

    with caplog.at_level(logging.INFO):
        assert report._write_figure(tmp_path, "gated", builder, logging.getLogger("t")) is None
    assert "figure 'gated' skipped (nothing to show on this route)" in caplog.text
    assert not list(tmp_path.iterdir())
    assert not plt.get_fignums()


def _projected_model() -> VFEModel:
    torch.manual_seed(71)
    return VFEModel(test_config(
        vocab_size=7, embed_dim=4, n_heads=2, batch_size=1, max_seq_len=4,
        n_layers=1, n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
        pos_phi="learned", use_block_mlp=True,
        block_mlp_expansion=1, attention_mode="projected_predictive",

    )).eval()


def test_report_gates_route_trivial_figures_and_emits_history_figures(tmp_path, caplog, monkeypatch):
    run_dir = tmp_path / "run"
    _write_metrics(run_dir / "metrics.csv")
    _write_attention_stats(run_dir / "diagnostics" / "attention_stats.csv")
    _write_residual_ratio(run_dir / "diagnostics" / "residual_ratio.csv")
    called = []
    for name in ("e_step_belief_trace", "across_layer_belief_trace"):
        monkeypatch.setattr(extract, name, lambda *a, _n=name, **k: called.append(_n))
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    with caplog.at_level(logging.DEBUG):                  # route-trivial skips log at debug
        written = report.generate_figures(run_dir, model=_projected_model(),
                                          loader=[(tokens, targets)], device=torch.device("cpu"),
                                          max_sequences=1)
    names = {path.name for path in written}
    assert called == []                          # the E-step traces are not replayed on this route
    for name in ("holonomy_curvature", "iterate_trajectory", "belief_trajectories",
                 "belief_spectrum", "spd_ellipses"):
        assert f"{name}.png" not in names
        assert f"figure '{name}' skipped (" in caplog.text
    assert "flat transport: holonomy is the identity" in caplog.text
    assert "legacy shared-objective E-step probes do not apply" in caplog.text
    assert {"head_block_norms.png", "numerical_trust.png", "per_layer_diagnostics.png",
            "sigma_ce_scatter.png", "reliability_diagram.png"} <= names
    assert {f"{name}.png" for name in report.HISTORY_FIGURES} <= names
    assert "gauge_head_specialization.png" not in names
    assert not plt.get_fignums()


# ---------------------------------------------------------------------------
# (2) head x block map norms (replaces gauge_head_specialization)
# ---------------------------------------------------------------------------

def test_head_block_norms_energy_shares_and_own_block_range(tmp_path):
    rng = np.random.default_rng(0)
    norms = {layer: {name: rng.uniform(0.5, 2.0, (3, 3)) for name in ("S_Q", "S_K", "T", "O")}
             for layer in range(2)}
    norms[1]["S_Q"] = np.eye(3) * 2.0                     # all energy on the own block
    fig = figs.plot_head_block_norms(norms, path=str(tmp_path / "hb.png"))
    images = [ax.images[0].get_array() for ax in fig.axes if ax.images]
    assert len(images) == 8
    for share in images:
        np.testing.assert_allclose(np.asarray(share).sum(axis=1), 1.0, rtol=1e-12)
    assert any("own 1.00-1.00" in title for title in _titles(fig))
    _check_page(fig)
    plt.close(fig)
    with pytest.raises(figs.FigureSkipped):
        figs.plot_head_block_norms({})


def test_head_block_norms_from_model_state_through_posthoc_maps(tmp_path):
    maps = pytest.importorskip("vfe3.posthoc.maps")
    model = _projected_model()
    state = {k: v for k, v in model.state_dict().items() if k.startswith("relational_operators")}
    norms = maps.head_block_norms(state, model.cfg)
    fig = figs.plot_head_block_norms(norms, path=str(tmp_path / "hb_model.png"))
    assert (tmp_path / "hb_model.png").stat().st_size > 0
    _check_page(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (3) grad_norm_decomposition and optimizer_geometry
# ---------------------------------------------------------------------------

def _group_history(n: int = 60) -> dict:
    step = np.arange(1, n + 1) * 10.0
    hist = {"step": step}
    for index, group in enumerate(figs.PARAMETER_GROUPS):
        hist[f"grad_norm_group_{group}"] = np.full(n, 0.1 * (index + 1))
        hist[f"param_norm_group_{group}"] = np.full(n, 10.0 * (index + 1))
        hist[f"update_ratio_group_{group}"] = np.full(n, 1e-4 * (index + 1))
    hist["sigma_log_mean"] = np.linspace(2.6, 0.04, n)
    hist["sigma_log_sd"] = np.linspace(0.5, 0.05, n)
    return hist


def test_grad_norm_decomposition_uses_named_groups(tmp_path):
    fig = figs.plot_grad_norm_decomposition(_group_history(), path=str(tmp_path / "g.png"))
    assert fig._suptitle.get_text() == "M-step gradient norm by parameter group"
    # One panel per live group, per-type tables first, under one bold header per family.
    assert _titles(fig) == [figs._GROUP_LABELS[g] for g in figs._TABLE_GROUPS + figs._CONTEXT_GROUPS]
    assert [t.get_text() for ax in fig.axes for t in ax.texts] == ["Per-type tables",
                                                                  "Context processing"]
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len({ax.get_ylim() for ax in _live_axes(fig)}) == 1          # one shared log y axis
    _check_page(fig)
    plt.close(fig)


def test_grad_norm_decomposition_legacy_roles_are_labeled_by_content(tmp_path):
    n = 40
    hist = {"step": np.arange(n) * 10.0, "grad_norm_mu": np.full(n, 0.8),
            "grad_norm_sigma": np.full(n, 1e-3), "grad_norm_phi": np.full(n, 0.2)}
    fig = figs.plot_grad_norm_decomposition(hist, path=str(tmp_path / "g_old.png"))
    title = fig.axes[0].get_title()
    assert title.startswith("M-step gradient norm by optimizer role")
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("'mu' role" in lab and "decode" in lab and "relational maps" in lab for lab in legend)
    _check_page(fig)
    plt.close(fig)


def test_optimizer_geometry_groups_sigma_moments_and_title(tmp_path):
    hist = _group_history()
    fig = figs.plot_optimizer_geometry(hist, path=str(tmp_path / "o.png"))
    assert fig._suptitle.get_text() == "Optimizer geometry"          # AdamW: no pullback claim
    titles = _titles(fig)
    assert "Parameter norm, per-type tables" in titles
    assert "Update-to-weight ratio, context processing" in titles
    assert any("Stored log-variances" in title for title in titles)
    assert all(ax.get_legend() is None for ax in fig.axes)          # lines are labeled at their ends
    labels = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "decode" in labels and r"score maps $S_Q,S_K$" in labels
    assert r"$\log\sigma^2$ table" not in labels                     # excluded from norms and ratios
    _check_page(fig)
    plt.close(fig)
    hist["phi_ridge_direction_cosine_mean"] = np.full(hist["step"].size, 0.9)
    fig = figs.plot_optimizer_geometry(hist, path=str(tmp_path / "o_pb.png"))
    assert "pullback-group" in fig._suptitle.get_text()
    plt.close(fig)


def test_optimizer_geometry_legacy_role_columns(tmp_path):
    n = 30
    hist = {"step": np.arange(n) * 10.0, "weight_norm_mu": np.full(n, 1.8e3),
            "weight_norm_sigma": np.geomspace(7334, 184, n), "weight_norm_phi": np.full(n, 50.0),
            "grad_norm_mu": np.full(n, 0.8), "grad_norm_sigma": np.full(n, 1e-3),
            "grad_norm_phi": np.full(n, 0.2)}
    fig = figs.plot_optimizer_geometry(hist, path=str(tmp_path / "o_old.png"))
    titles = _titles(fig)
    assert fig._suptitle.get_text() == "Optimizer geometry"
    assert any("Norm of the" in title for title in titles)
    ratio_axis = next(ax for ax in fig.axes if ax.get_title() == "Gradient-to-weight ratio")
    assert [t.get_text() for t in ratio_axis.texts] == ["'mu' role", "'phi' role"]   # no Sigma ratio
    assert any("'mu' role = mu table, decode" in t.get_text() for t in fig.texts)
    _check_page(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (4) validation_sanity
# ---------------------------------------------------------------------------

def test_validation_sanity_lays_out_live_panels_and_prefers_fixed_gap(tmp_path):
    step = np.arange(1, 11) * 100.0
    hist = {"step": step, "generalization_gap": np.linspace(-0.4, 0.3, 10),
            "generalization_gap_fixed": np.linspace(0.01, 0.2, 10)}
    fig = figs.plot_validation_sanity(hist, path=str(tmp_path / "v.png"))
    assert _titles(fig) == ["Generalization gap (fixed train sample)"]
    assert len(_live_axes(fig)) == 1 and len(fig.axes) == 1
    _check_page(fig)
    plt.close(fig)
    del hist["generalization_gap_fixed"]
    hist["pos_loss_first_q"] = np.full(10, 4.0)
    hist["pos_loss_last_q"] = np.full(10, 3.5)
    fig = figs.plot_validation_sanity(hist, path=str(tmp_path / "v_old.png"))
    assert _titles(fig) == ["Generalization gap (one training batch)", "Positional loss"]
    assert len(fig.axes) == 2
    _check_page(fig)
    plt.close(fig)
    assert "generalization_gap_fixed" in figs.VALIDATION_SANITY_COLUMNS


# ---------------------------------------------------------------------------
# (5) sigma_ce_scatter and reliability population labels (9)
# ---------------------------------------------------------------------------

def test_sigma_ce_scatter_states_frozen_route_and_cv(tmp_path):
    gen = torch.Generator().manual_seed(0)
    bank = {"tr_sigma": 4.0 + 0.1 * torch.rand(500, generator=gen),
            "ce": torch.rand(500, generator=gen) * 6.0}
    fig = figs.plot_sigma_ce_scatter(bank, frozen_covariance=True, vocab_cv=0.0188,
                                     population="test split, report bank, n=500 scored tokens",
                                     path=str(tmp_path / "s.png"))
    ax = fig.axes[0]
    assert ax.get_title().startswith(r"$\Sigma$ frozen at encode")
    note = ax.texts[0].get_text()
    tr = bank["tr_sigma"].double()
    expected_cv = float(tr.std(unbiased=True) / tr.mean().abs())
    assert f"over these tokens = {expected_cv:.3f}" in note
    assert "over the vocabulary = 0.019" in note
    assert "n=500" in note
    _check_page(fig)
    plt.close(fig)
    fig = figs.plot_sigma_ce_scatter(bank, path=str(tmp_path / "s_live.png"))
    assert "frozen" not in fig.axes[0].get_title()
    plt.close(fig)


def test_reliability_diagram_prints_population_and_keeps_bars_in_unit_interval(tmp_path):
    rel = [{"conf": 0.05, "acc": 0.04, "frac": 0.3}, {"conf": 0.5, "acc": 0.45, "frac": 0.4},
           {"conf": 0.99, "acc": 0.97, "frac": 0.3}]
    fig = figs.plot_reliability_diagram(
        rel, population="research.json sample (test split, finalization batches), not this report's bank",
        path=str(tmp_path / "r.png"))
    ax = fig.axes[0]
    assert "research.json sample" in ax.get_title()
    for patch in ax.patches:
        assert patch.get_x() >= -1e-12 and patch.get_x() + patch.get_width() <= 1.0 + 1e-12
    _check_page(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (6) numerical_trust counts with denominators
# ---------------------------------------------------------------------------

def test_numerical_trust_counts_with_denominators(tmp_path):
    guard = {"sigma_floor_frac": 0.25, "sigma_ceil_frac": 0.0, "energy_klmax_frac": 0.0}
    health = {"nan_mu": 0.5, "nan_beta": 0.0, "numel_mu": 4, "numel_beta": 16, "max_condition": 3.0}
    causal = {"future_leakage": torch.tensor([[0.0, 0.2], [0.0, 0.0]]),
              "row_sum_error": torch.zeros(2, 2)}
    fig = figs.plot_numerical_trust(
        guard, health, causal, guard_denominators={"sigma_floor_frac": 8, "sigma_ceil_frac": 8,
                                                   "energy_klmax_frac": 16},
        population="test split, one 4-token sequence", path=str(tmp_path / "t.png"))
    cells = {(r, c): cell.get_text().get_text()
             for (r, c), cell in fig.axes[0].tables[0].get_celld().items()}
    rows = {cells[(r, 0)]: (cells[(r, 1)], cells[(r, 2)])
            for r in range(1, 1 + max(key[0] for key in cells))}
    assert rows[r"variance at the floor $\varepsilon$"] == ("2", "8")
    assert rows[r"non-finite $\mu$"] == ("2", "4")
    assert rows[r"non-finite $\beta$"] == ("0", "16")
    assert rows["(layer, head) maps with future attention > $10^{-6}$"] == ("1", "4")
    assert rows[r"(layer, head) maps with $|{\sum}_j\beta_{ij}-1|$ > $10^{-5}$"] == ("0", "4")
    assert "test split, one 4-token sequence" in fig.axes[0].get_title()
    _check_page(fig)
    plt.close(fig)


def test_guard_denominators_diagonal_and_full():
    energy = torch.zeros(2, 5, 5)
    diag = extract.guard_denominators(torch.ones(5, 4), energy, torch.zeros(5))
    assert diag == {"sigma_floor_frac": 20, "sigma_ceil_frac": 20, "energy_klmax_frac": 50,
                    "selfdiv_klmax_frac": 5}
    full = extract.guard_denominators(torch.eye(4).expand(5, 4, 4), energy)
    assert full["sigma_floor_frac"] == 20 and "selfdiv_klmax_frac" not in full
    assert extract.guard_denominators(torch.ones(4, 4), energy, diagonal=True)["sigma_floor_frac"] == 16


def _tiny_model(**over) -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(test_config(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                               n_e_steps=1, e_q_mu_lr=0.1, **over)).eval()


def test_numerical_health_reports_entry_counts_and_vocab_sigma_trace():
    model = _tiny_model()
    health = extract.numerical_health(model, torch.randint(0, 9, (1, 5)))
    for name in ("mu", "sigma", "phi", "energy", "beta"):
        assert health[f"numel_{name}"] > 0
    traces = extract.vocabulary_sigma_trace(model, chunk=4)
    assert traces.shape == (9,) and traces.dtype == torch.float64
    direct = model.prior_bank.encode(torch.arange(9).unsqueeze(0)).sigma
    from vfe3 import metrics
    expected = metrics.sigma_trace(direct, diagonal=model.cfg.diagonal_covariance,
                                   family=model.cfg.family).reshape(-1).double()
    torch.testing.assert_close(traces, expected)


# ---------------------------------------------------------------------------
# (7) readout / per-layer diagnostics with residual_ratio.csv
# ---------------------------------------------------------------------------

def _readout_history(n: int = 12) -> dict:
    return {"step": np.arange(1, n + 1) * 10.0, "attn_entropy": np.linspace(3, 2, n),
            "attn_entropy_min": np.linspace(1, 0.5, n),
            "relational_mean_displacement_norm": np.full(n, 1.2e4),
            "effective_rank": np.full(n, 4.0), "belief_cond_median": np.full(n, 1.6)}


def test_readout_diagnostics_residual_ratio_replaces_displacement(tmp_path):
    table_path = tmp_path / "diagnostics" / "residual_ratio.csv"
    _write_residual_ratio(table_path)
    table = figs.read_long_table(table_path)
    fig = figs.plot_readout_diagnostics(_readout_history(), residual_ratio=table,
                                        covariance_frozen=True, path=str(tmp_path / "ro.png"))
    titles = _titles(fig)
    assert any("Residual ratio" in title for title in titles)
    assert not any("displacement" in title.lower() for title in titles)
    assert not any("Receiver covariance" in title for title in titles)
    _check_page(fig)
    plt.close(fig)
    fig = figs.plot_readout_diagnostics(_readout_history(), path=str(tmp_path / "ro_old.png"))
    titles = _titles(fig)
    assert any("Unnormalized displacement" in title for title in titles)
    assert any("Receiver covariance rank" in title for title in titles)
    plt.close(fig)


def test_per_layer_conditional_branch_uses_last_eval_residual_ratio(tmp_path):
    table_path = tmp_path / "rr.csv"
    _write_residual_ratio(table_path)
    per_layer = {"inference_semantics": "readout", "attn_entropy": [3.0, 2.8, 2.3],
                 "attn_entropy_min": [1.0, 0.9, 0.8],
                 "relational_mean_displacement_norm": [2e4, 1.9e4, 6e4],
                 "effective_rank": [4.0, 4.0, 4.0], "belief_cond_median": [1.6, 1.6, 1.6]}
    fig = figs.plot_per_layer_diagnostics(per_layer, residual_ratio=figs.read_long_table(table_path),
                                          covariance_frozen=True, path=str(tmp_path / "pl.png"))
    titles = _titles(fig)
    assert any("Residual ratio" in title and "step 90" in title for title in titles)
    assert not any("Receiver covariance" in title for title in titles)
    residual_axis = next(ax for ax in fig.axes if "Residual ratio" in ax.get_title())
    medians = next(line for line in residual_axis.lines if line.get_label().startswith("median"))
    np.testing.assert_allclose(medians.get_ydata()[-3:] if len(medians.get_ydata()) > 3
                               else medians.get_ydata(), [10.0 * 1.3, 100.0 * 1.3, 1000.0 * 1.3])
    _check_page(fig)
    plt.close(fig)


def test_per_layer_transported_branch_fits_page(tmp_path):
    keys = ("self_coupling", "belief_coupling", "attention_entropy", "total",
            "holonomy_deviation", "holonomy_wilson", "gauge_trace_spread", "gauge_invariant_spread",
            "phi_norm_mean", "effective_rank", "attn_entropy", "belief_cond_median")
    fig = figs.plot_per_layer_diagnostics({k: [1.0, 2.0, 3.0] for k in keys},
                                          path=str(tmp_path / "plt.png"))
    _check_page(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (8) learning curves, attention statistics, residual ratio
# ---------------------------------------------------------------------------

def test_read_metrics_columns_blank_cells_are_nan(tmp_path):
    path = tmp_path / "metrics.csv"
    _write_metrics(path)
    cols = figs.read_metrics_columns(path)
    assert np.isnan(cols["val_ce"][0]) and np.isfinite(cols["val_ce"][2])
    assert figs.read_metrics_columns(tmp_path / "absent.csv") is None


def test_learning_curves_three_panels_from_new_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    _write_metrics(path)
    fig = figs.plot_learning_curves(figs.read_metrics_columns(path), path=str(tmp_path / "lc.png"))
    axes = _live_axes(fig)
    assert [ax.get_title() for ax in axes] == [
        "CE vs training tokens", "CE vs wall-clock time", "Generalization gap (fixed sample)"]
    assert axes[0].get_xlabel() == "training tokens"
    assert axes[1].get_xlabel() == "wall-clock hours"
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert any(lab.startswith("validation") for lab in labels)
    assert any(lab.startswith("fixed training sample") for lab in labels)
    np.testing.assert_allclose(axes[2].lines[0].get_ydata(), 0.05, atol=1e-12)
    assert "corpus passes" in fig._suptitle.get_text()               # corpus_pass crosses 1 and 2
    _check_page(fig)
    plt.close(fig)


def test_learning_curves_legacy_run_uses_tokens_per_step(tmp_path):
    path = tmp_path / "metrics.csv"
    _write_metrics(path, new_columns=False)
    cols = figs.read_metrics_columns(path)
    fig = figs.plot_learning_curves(cols, tokens_per_step=2048, path=str(tmp_path / "lc_old.png"))
    axes = _live_axes(fig)
    assert len(axes) == 2                                            # no fixed gap column
    assert axes[0].get_xlabel() == "training tokens (step x tokens per step)"
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), np.array([30, 60, 90, 120]) * 2048.0)
    plt.close(fig)
    fig = figs.plot_learning_curves(cols, path=str(tmp_path / "lc_step.png"))
    assert _live_axes(fig)[0].get_xlabel() == "training step"
    plt.close(fig)
    with pytest.raises(figs.FigureSkipped):
        figs.plot_learning_curves({"step": np.arange(3.0), "val_ce": np.full(3, np.nan)})


def test_attention_stats_heatmaps_and_training_curves(tmp_path):
    path = tmp_path / "diagnostics" / "attention_stats.csv"
    _write_attention_stats(path)
    table = figs.read_long_table(path)
    fig = figs.plot_attention_stats(table, path=str(tmp_path / "as.png"))
    grids = [np.asarray(ax.images[0].get_array()) for ax in fig.axes if ax.images]
    assert len(grids) == 4 and grids[0].shape == (2, 3)
    np.testing.assert_allclose(grids[0][1], [0.4, 0.5, 0.6])        # entropy_norm by head
    assert "step 90" in fig._suptitle.get_text() and "120 query rows" in fig._suptitle.get_text()
    _check_page(fig)
    plt.close(fig)
    fig = figs.plot_attention_stats_training(table, path=str(tmp_path / "ast.png"))
    first = _live_axes(fig)[0]
    assert len(first.lines) == 2                                     # one head-mean line per layer
    np.testing.assert_allclose(first.lines[0].get_ydata(), 0.5)
    _check_page(fig)
    plt.close(fig)
    one_step = tmp_path / "one.csv"
    _write_attention_stats(one_step, steps=(30,))
    with pytest.raises(figs.FigureSkipped, match="fewer than two"):
        figs.plot_attention_stats_training(figs.read_long_table(one_step))


def test_residual_ratio_curves_single_eval_and_missing_norm(tmp_path):
    path = tmp_path / "rr.csv"
    _write_residual_ratio(path)
    fig = figs.plot_residual_ratio(figs.read_long_table(path), path=str(tmp_path / "rr.png"))
    axes = _live_axes(fig)
    assert len(axes) == 2 and axes[0].get_yscale() == "log"
    assert len(axes[0].lines) == 3                                   # one median line per layer
    _check_page(fig)
    plt.close(fig)
    single = tmp_path / "rr1.csv"
    _write_residual_ratio(single, steps=(90,), ln=False)
    fig = figs.plot_residual_ratio(figs.read_long_table(single), path=str(tmp_path / "rr1.png"))
    axes = _live_axes(fig)
    assert len(axes) == 1 and "step 90" in axes[0].get_title()      # no normalization panel
    _check_page(fig)
    plt.close(fig)


def test_generate_history_figures_writes_each_figure_only_when_its_data_exist(tmp_path, caplog):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with caplog.at_level(logging.INFO):
        assert report.generate_history_figures(run_dir) == []
    assert not (run_dir / "figures").exists()
    _write_metrics(run_dir / "metrics.csv", new_columns=False)
    (run_dir / "config.json").write_text(json.dumps({"config": {"batch_size": 8, "max_seq_len": 16}}),
                                         encoding="utf-8")
    with caplog.at_level(logging.INFO):
        written = report.generate_history_figures(run_dir)
    assert [p.name for p in written] == ["learning_curves.png"]
    assert "figure 'attention_stats' skipped (diagnostics/attention_stats.csv absent or empty)" in caplog.text
    _write_attention_stats(run_dir / "diagnostics" / "attention_stats.csv", steps=(30,))
    _write_residual_ratio(run_dir / "diagnostics" / "residual_ratio.csv")
    with caplog.at_level(logging.INFO):
        written = report.generate_history_figures(run_dir)
    assert [p.name for p in written] == ["learning_curves.png", "attention_stats.png",
                                         "residual_ratio.png"]
    assert "figure 'attention_stats_training' skipped (attention_stats.csv has fewer than two" in caplog.text
    assert all(p.stat().st_size > 0 for p in written)
    assert report._tokens_per_step(run_dir, None) == 128.0
    assert not plt.get_fignums()


# ---------------------------------------------------------------------------
# shared dashboard geometry
# ---------------------------------------------------------------------------

def test_history_dashboard_caps_width_and_columns(tmp_path):
    step = np.arange(1, 21) * 10.0
    hist = {"step": step, **{f"m{i}": np.full(20, float(i + 1)) for i in range(5)}}
    panels = [{"title": f"m{i}", "ylabel": "v", "series": [(f"m{i}", f"m{i}", "#0072B2")]}
              for i in range(5)]
    fig = figs._history_dashboard(hist, panels, "five panels", str(tmp_path / "d.png"), ncols=4)
    assert fig.axes[0].get_gridspec().ncols == 3
    _check_page(fig)
    plt.close(fig)
