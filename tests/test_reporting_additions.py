r"""Tests for the 2026-06-28 reporting build-out (surfacing already-logged diagnostics):

  * the four new registered history-dashboard figures render from synthetic history, drop absent
    panels, and mask eval-cadence NaN gaps; the pooled PPL offset figure renders;
  * run_artifacts._save_figures emits the four dashboard PNGs from a synthetic history;
  * run_artifacts._pure_path_report reports the toggle/stress state and the on-pure-path flag;
  * train() surfaces the held-out gauge/SPD/Fisher geometry columns into metrics.csv;
  * ablation._seed_aggregate / _base_label group seeds into n/mean/SD/CV;
  * scaling_analysis._write_scaling_md renders the console-only fits as a markdown report;
  * end-to-end: scaling_analysis.analyze() and ablation.main() dispatch capacity_scaling /
    pareto_frontier / ablation_forest / lr_grid_heatmap through the figure registry by NAME
    (a monkeypatched registration is what actually produces each PNG) while every legacy output
    keeps being written (PB-07 registry-completion Task 5).

Device-agnostic (CPU). Figures use the Agg backend.
"""
from tests.profiles import config_values, test_config
import csv as _csv
import hashlib
import json
import pytest
import logging
import math
import os
import types
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import ablation
import scaling_analysis
from vfe3.config import serialize_config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    RunArtifacts,
    _cost_model_fields,
    _pure_path_report,
    _save_figures,
    finalize_run,
)
from vfe3.train import train
from vfe3.viz import figures as figs

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.fixture(autouse=True)
def _local_reporting_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling_analysis, "CONFIG", dict(
        input_dir="unused-test-reporting-input", with_offset=True,
        n_bootstrap=0, min_points=2, force_accept_code_identity_drift=False))


def _hist(keys, n=30, nan_every=0):
    r"""Synthetic {step, key: [...]} history: positive decreasing series, optionally NaN-masked on
    non-eval rows (nan_every>0 keeps a finite value every nan_every steps, like an eval-cadence column)."""
    h = {"step": list(range(n))}
    base = np.linspace(1.0, 0.1, n) + 0.05
    for k in keys:
        col = base.copy()
        if nan_every:
            for i in range(n):
                if i % nan_every:
                    col[i] = float("nan")
        h[k] = col.tolist()
    return h


# --------------------------------------------------------------------------- figure renders

def test_geometry_health_renders():
    h = _hist(("holonomy_wilson", "cocycle_residual", "gauge_invariant_spread", "phi_norm_mean",
               "phi_norm_std", "belief_cond_p95", "belief_cond_max", "eff_rank_p5", "eff_rank_median",
               "eff_rank_p95", "fisher_trace_mean", "guard_sigma_floor_frac", "nonfinite_frac",
               "renyi_band_frac", "attn_entropy_min", "attn_entropy_collapsed_heads"))
    fig = figs.plot_geometry_health(h); assert fig is not None; plt.close(fig)


def test_estep_quality_renders():
    h = _hist(("estep_f_drop", "estep_f_nondecreasing_frac", "estep_r_mu_last", "estep_r_sigma_last",
               "estep_r_phi_last"))
    fig = figs.plot_estep_quality(h); assert fig is not None; plt.close(fig)


def test_validation_sanity_renders_with_eval_cadence_gaps():
    # eval-cadence columns are NaN on non-eval rows; the dashboard must mask per series and still render.
    h = _hist(("generalization_gap", "pos_loss_ratio", "val_future_leakage", "val_row_sum_error",
               "val_pos_content_r2", "val_head_redundancy_js", "val_holonomy_wilson",
               "val_cocycle_residual", "val_belief_cond_p95", "val_fisher_trace_mean",
               "val_phi_norm_mean"), nan_every=5)
    fig = figs.plot_validation_sanity(h); assert fig is not None; plt.close(fig)


def test_optimizer_geometry_renders_with_ratio_synthesis():
    h = _hist(("phi_ridge_direction_cosine_mean",
               "phi_pullback_damped_gen_cond_median", "phi_pullback_damped_gen_cond_max",
               "phi_group_trust_scale_mean", "phi_group_trust_scale_min",
               "phi_group_active_rows", "phi_group_chart_norm_max",
               "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
               "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))
    fig = figs.plot_optimizer_geometry(h); assert fig is not None; plt.close(fig)

def test_ppl_offset_renders():
    pts = [{"embed_dim": k, "ppl_mean": 50.0 / k + 12.0, "ppl_sem": 0.3, "n_seeds": 3}
           for k in (8, 16, 32, 64)]
    fig = figs.plot_ppl_offset(pts); assert fig is not None; plt.close(fig)


def test_new_figures_registered():
    for name, fn in (("geometry_health", figs.plot_geometry_health),
                     ("estep_quality", figs.plot_estep_quality),
                     ("validation_sanity", figs.plot_validation_sanity),
                     ("optimizer_geometry", figs.plot_optimizer_geometry),
                     ("ppl_offset", figs.plot_ppl_offset)):
        assert figs.get_figure(name) is fn


def test_dashboard_drops_absent_and_empty():
    # only one column present -> one panel, still renders; no columns -> the no-data fallback renders.
    fig = figs.plot_geometry_health(_hist(("holonomy_wilson",))); assert fig is not None; plt.close(fig)
    fig = figs.plot_geometry_health({"step": [0, 1, 2]}); assert fig is not None; plt.close(fig)


# --------------------------------------------------------------------------- run_artifacts wiring

def test_save_figures_emits_dashboards(tmp_path):
    torch.manual_seed(0)
    cfg = test_config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2, batch_size=2)
    model = VFEModel(cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic", device=str(DEVICE))
    keys = ("holonomy_wilson", "cocycle_residual", "fisher_trace_mean", "belief_cond_p95",
            "estep_f_drop", "estep_r_mu_last", "generalization_gap", "val_holonomy_wilson",
            "phi_ridge_direction_cosine_mean", "weight_norm_mu", "grad_norm_mu")
    art.history = [{"step": s, **{k: 1.0 / (s + 1) + 0.1 for k in keys}} for s in range(12)]
    _save_figures(art, [3.0, 2.5], logging.getLogger("test"))
    for png in ("geometry_health.png", "validation_sanity.png",
                "optimizer_geometry.png"):
        assert (tmp_path / png).exists(), f"{png} not emitted"



# ------------------------------------------------------ M4 (audit 2026-07-06): multiseed ablation figures

def _capture_ablation_figure(monkeypatch, fn_name):
    r"""Patch vfe3.viz.figures.<fn_name> (imported call-time inside the ablation collectors) to capture
    the cells it is called with instead of rendering."""
    cap = {}
    def fake(cells, path=None):
        cap["cells"] = list(cells)
        return plt.figure()
    monkeypatch.setattr(f"vfe3.viz.figures.{fn_name}", fake)
    return cap


def test_gauge_transport_figure_aggregates_seeds(monkeypatch, tmp_path):
    # M4(a): on_L1__s6 must parse (not split into 4 parts and skip); the seed cells aggregate to one
    # bar per (mode, depth) at the across-seed mean PPL.
    rows = [
        {"label": "on_L1__s6",  "primary_val_ppl": "10.0", "omega_identity_dev": "0.0"},
        {"label": "on_L1__s7",  "primary_val_ppl": "12.0", "omega_identity_dev": "0.0"},
        {"label": "off_L1__s6", "primary_val_ppl": "20.0", "omega_identity_dev": "1e-7"},
        {"label": "off_L1__s7", "primary_val_ppl": "22.0", "omega_identity_dev": "3e-7"},
    ]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d: rows)
    cap = _capture_ablation_figure(monkeypatch, "plot_gauge_transport_bars")
    ablation._plot_gauge_transport(tmp_path, tmp_path)
    assert "cells" in cap, "figure skipped: seeded on_L1__s* labels not parsed"
    assert len(cap["cells"]) == 2
    by = {(c["mode"], c["depth"]): c for c in cap["cells"]}
    assert set(by) == {("on", "L1"), ("off", "L1")}
    assert abs(by[("on", "L1")]["ppl"] - 11.0) < 1e-9      # (10 + 12) / 2
    assert abs(by[("off", "L1")]["ppl"] - 21.0) < 1e-9     # (20 + 22) / 2


def test_mu_precond_figure_aggregates_seeds(monkeypatch, tmp_path):
    # M4(b): fisher_T2__s6 must parse (int('2__s6') would ValueError and skip); aggregate to one point
    # per (precond, n_e_steps).
    rows = [
        {"label": "fisher_T2__s6", "primary_val_ppl": "30.0"},
        {"label": "fisher_T2__s7", "primary_val_ppl": "32.0"},
        {"label": "raw_T2__s6",    "primary_val_ppl": "40.0"},
        {"label": "raw_T2__s7",    "primary_val_ppl": "42.0"},
    ]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d: rows)
    cap = _capture_ablation_figure(monkeypatch, "plot_mu_precond")
    ablation._plot_mu_precond(tmp_path, tmp_path)
    assert "cells" in cap, "figure skipped: int('2__s6') ValueError on seeded labels"
    assert len(cap["cells"]) == 2
    by = {(c["precond"], c["n_e_steps"]): c for c in cap["cells"]}
    assert set(by) == {("fisher", 2), ("raw", 2)}
    assert abs(by[("fisher", 2)]["ppl"] - 31.0) < 1e-9
    assert abs(by[("raw", 2)]["ppl"] - 41.0) < 1e-9


def test_attention_entropy_figure_aggregates_seeds(monkeypatch, tmp_path):
    # M4(c): the seed cells sharing (include_attention_entropy, kappa) must collapse to one cell at the
    # mean PPL, not be passed as N per-seed cells (the figure would then plot one arbitrary seed).
    rows = [
        {"label": f"ent{e}__s{s}", "primary_val_ppl": str(p), "cov_gap": "inf",
         "overrides": {"include_attention_entropy": e, "kappa_beta": 1.0}}
        for (e, base) in ((True, 10.0), (False, 14.0)) for (s, p) in ((6, base), (7, base + 2.0))
    ]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d: rows)
    cap = _capture_ablation_figure(monkeypatch, "plot_entropy_ppl_gap")
    ablation._plot_attention_entropy(tmp_path, tmp_path)
    assert "cells" in cap
    assert len(cap["cells"]) == 2                           # one per include_attention_entropy at kappa=1
    by = {bool(c["include_attention_entropy"]): c for c in cap["cells"]}
    assert set(by) == {True, False}
    assert abs(by[True]["ppl"] - 11.0) < 1e-9              # (10 + 12) / 2
    assert abs(by[False]["ppl"] - 15.0) < 1e-9            # (14 + 16) / 2


def test_offset_power_law_honors_weights():
    # m29: the with_offset (Chinchilla) fit must honor the WLS weights callers pass, not silently
    # run unweighted OLS. Down-weighting a corrupted point must move the fitted exponent.
    import pytest
    pytest.importorskip("scipy")   # the offset branch needs scipy.optimize.curve_fit (else falls back to log-log)
    from vfe3.viz.figures import _fit_power_law
    N = np.array([1e2, 1e3, 1e4, 1e5])
    L = (2.0 + 5.0 * N ** -0.3).copy()
    L[0] += 0.5                                            # corrupt the low-N point
    fit_uniform = _fit_power_law(N, L, weights=np.ones_like(N), with_offset=True)
    fit_downwt  = _fit_power_law(N, L, weights=np.array([1e-3, 1.0, 1.0, 1.0]), with_offset=True)
    assert fit_uniform["form"] == "offset_power_law" and fit_downwt["form"] == "offset_power_law"
    assert abs(fit_uniform["alpha"] - fit_downwt["alpha"]) > 1e-4   # weights must change the fit


# --------------------------------------------------------------------------- pure-path certificate

def _cost_cfg(**over):
    base = dict(
        vocab_size=11, embed_dim=6, n_heads=2, max_seq_len=7, batch_size=2,
        n_layers=2, n_e_steps=3, max_steps=5,
        decode_mode="linear", decode_bias=False,
        diagonal_covariance=True, gauge_group="block_glk", amp_dtype="fp32", use_block_mlp=False,
        relational_mode="projected_filtering", relational_readout_mode="full_link_update",
    )
    base = {**config_values(), **base}
    base.update(over)
    base.setdefault("d_head", base["embed_dim"] // base["n_heads"])
    base.setdefault("relational_mode", "projected_filtering")
    base.setdefault("relational_readout_mode", "full_link_update")
    return types.SimpleNamespace(**base)


def _cost_model(*, K=6, n_gen=5, n_blocks=2):
    return types.SimpleNamespace(
        group=types.SimpleNamespace(
            generators=torch.zeros(n_gen, K, K),
            generator_shape=(n_gen, K, K),
            irrep_dims=[K // n_blocks] * n_blocks,
        ),
        parameters=lambda: iter(()),
    )


def test_cost_model_linear_decode_does_not_count_prior_bank_readout():
    V, K, n_gen = 11, 6, 5
    model = _cost_model(K=K, n_gen=n_gen)
    token_row = 2 * K + n_gen

    linear = _cost_model_fields(model, _cost_cfg(), n_params=123, tokens_seen=13)
    biased = _cost_model_fields(
        model,
        _cost_cfg(decode_bias=True),
        n_params=123,
        tokens_seen=13,
    )
    prior_bank = _cost_model_fields(
        model,
        _cost_cfg(decode_mode="diagonal_chunked"),
        n_params=123,
        tokens_seen=13,
    )

    assert linear["active_params_per_token"] == token_row + V * K + 4 * 2 * K * K
    assert biased["active_params_per_token"] == token_row + V * K + V + 4 * 2 * K * K
    assert prior_bank["active_params_per_token"] == token_row + 2 * V * K + 4 * 2 * K * K
    assert linear["flops_per_token_decode"] == prior_bank["flops_per_token_decode"] == 2.0 * V * K


def test_cost_model_declares_projected_estep_flops_unavailable():
    V, K, n_gen, N, L, T, n_blocks = 11, 6, 5, 7, 2, 3, 2
    tokens_seen = 13
    model = _cost_model(K=K, n_gen=n_gen, n_blocks=n_blocks)
    cfg = _cost_cfg(
        max_seq_len=N,
        n_layers=L,
        n_e_steps=T,
    )

    out = _cost_model_fields(model, cfg, n_params=123, tokens_seen=tokens_seen)

    # The linear decoder reads only W, so the report must not count a prior-bank readout here.
    expected_active = (2 * K + n_gen) + V * K + 4 * L * K * K
    assert out["active_params_per_token"] == expected_active
    assert out["flops_per_token_estep"] is None
    assert out["est_flops_analytic"] is None
    assert "projected" in out["est_flops_analytic_unavailable_reason"]


def _pure_ns(**over):
    r"""SimpleNamespace with every attribute _pure_path_report reads, at the pure values."""
    base = dict(
        include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
        decode_mode="diagonal_chunked", decode_coordinates="reference", use_head_mixer=False,
        use_priorbank_head_evidence_mixer=False, encode_mode="per_token",
        lambda_beta=1.0,
        gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
        skip_belief_sigma_update=False,
        gauge_group="block_glk", family="gaussian_full",
        gradient_mode="filtering", attention_mode="projected_gradient",
        relational_mode="projected_filtering", relational_readout_mode="full_link_update",
        embed_dim=4, n_heads=2, d_head=2,
        e_step_update="gradient", mm_damping=1.0,
        use_block_mlp=False, block_mlp_mode="coordinate", block_mlp_covariance="passthrough",
        block_mlp_expansion=4, block_mlp_covariance_floor=1e-4,
        block_mlp_activation="gelu", block_mlp_dropout=0.0,
        m_block_mlp_lr=None, m_p_mu_lr=1e-3,
        head_mixer_compatibility="disabled", head_mixer_gauge_compatible=True,
        sigma_max=None,
        norm_type_block="none", norm_type_final="none", layernorm_affine=False,
        m_phi_update_mode="adamw", m_phi_group_trust_radius=0.1,
        phi_mstep_max_matrix_norm=None, transport_chart_max_norm=12.0,
        e_phi_lr=0.0,
        beta_attention_prior="causal")
    base = {**config_values(), **base}
    base.update(over)
    base.setdefault("d_head", base["embed_dim"] // base["n_heads"])
    base.setdefault("relational_mode", "projected_filtering")
    base.setdefault("relational_readout_mode", "full_link_update")
    return types.SimpleNamespace(**base)


def test_pure_path_report_structure_and_flags():
    history = [{"step": 0, "nonfinite_frac": 0.0, "cocycle_residual": 1e-7, "holonomy_wilson": 2e-7,
                "guard_sigma_floor_frac": 0.0}]
    pure = _pure_ns()
    rep = _pure_path_report(pure, history)
    assert rep["on_pure_path"] is False
    assert rep["on_gauge_pure_path"] is True
    preexisting = {
        "on_pure_path", "pure_flags", "config_toggles", "converged_stress",
        "gauge_flags", "on_gauge_pure_path",
    }
    additions = {
        "on_causal_lm_path", "transport_exactness_status", "on_theory_pure_path",
        "causal_flags", "theory_flags",
    }
    assert preexisting <= rep.keys()
    assert additions <= rep.keys()
    assert rep["converged_stress"]["cocycle_residual"] == 1e-7
    assert rep["pure_flags"]["canonical_attention_entropy"] is False
    # Optional components retain independent flags despite the unavailable shared objective.
    impure = types.SimpleNamespace(**{**pure.__dict__, "use_head_mixer": True})
    rep2 = _pure_path_report(impure, history)
    assert rep2["on_pure_path"] is False and rep2["pure_flags"]["no_head_mixer"] is False


def test_pure_path_report_marks_priorbank_head_evidence_mixer_as_opt_in():
    rep = _pure_path_report(_pure_ns(
        use_priorbank_head_evidence_mixer=True,
        encode_mode="canonical_content_projected",
    ), [])

    assert rep["on_pure_path"] is False
    assert rep["pure_flags"]["no_priorbank_head_evidence_mixer"] is False
    assert rep["config_toggles"]["use_priorbank_head_evidence_mixer"] is True
    assert rep["config_toggles"]["encode_mode"] == "canonical_content_projected"
    assert rep["config_toggles"]["priorbank_head_evidence_role"] == "decoder_kl_irrep_weights"


def test_pure_path_report_keeps_priorbank_evidence_role_separate_from_head_mixer():
    rep = _pure_path_report(_pure_ns(
        use_head_mixer=True,
        use_priorbank_head_evidence_mixer=False,
        encode_mode="canonical_content_gauge",
    ), [])

    assert rep["pure_flags"]["no_head_mixer"] is False
    assert rep["pure_flags"]["no_priorbank_head_evidence_mixer"] is True
    assert rep["config_toggles"]["use_head_mixer"] is True
    assert rep["config_toggles"]["use_priorbank_head_evidence_mixer"] is False
    assert rep["config_toggles"]["encode_mode"] == "canonical_content_gauge"
    assert rep["config_toggles"]["priorbank_head_evidence_role"] == "inactive"


def test_gauge_purity_axis_is_independent_of_fe_axis():
    # F8 (audit 2026-07-01): the gauge axis remains independent of the shared-objective flag; flipping a gauge
    # toggle changes its own axis while projected attention remains off the shared-objective path.
    history = []
    rep = _pure_path_report(_pure_ns(gauge_transport="off"), history)
    assert rep["on_pure_path"] is False and rep["on_gauge_pure_path"] is False
    assert rep["gauge_flags"]["learned_gauge_transport"] is False
    # Right-folded RoPE (U_i R_i^T) is covariant on the projected routes (audit 2026-10-01 F11).
    rep2 = _pure_path_report(_pure_ns(pos_rotation="rope"), history)
    assert rep2["on_pure_path"] is False
    assert rep2["gauge_flags"]["positional_rotation_gauge_covariant"] is True
    # the gauge settings are surfaced in config_toggles for transparency
    for key in ("gauge_transport", "pos_rotation", "rope_full_gauge", "rope_on_value"):
        assert key in rep["config_toggles"]


def test_pure_path_marks_sigma_and_twohop():
    rep = _pure_path_report(_pure_ns(skip_belief_sigma_update=True), [])

    assert rep["on_pure_path"] is False
    assert rep["pure_flags"]["full_sigma_update"] is False
    assert rep["on_gauge_pure_path"] is True
    toggles = rep["config_toggles"]
    assert toggles["skip_belief_sigma_update"] is True


# --------------------------------------------------------------------------- figure memory guard

def _finalize_ns(**over):
    r"""SimpleNamespace cfg carrying every finalization attribute touched by these dispatch tests."""
    base = dict(
        # figure memory-guard inputs: 8*V*N*B/1e9 ~ 13.2 GB fp32 logits+probs peak > the 8 GB guard
        vocab_size=50257, max_seq_len=1024, batch_size=32,
        generate_embedding_figures=False,

        # numeric-path attrs (summary / provenance / cost model / pure-path report)
        seed=0, max_steps=2, decode_mode="full_chunked", decode_bias=False,
        use_head_mixer=False,
        min_lr=0.0, min_lr_frac=0.0,
        include_attention_entropy=True, transport_mode="flat", lambda_alpha_mode="constant",
        lambda_beta=1.0,
        gauge_transport="on", pos_rotation="none", rope_full_gauge=False, rope_on_value=True,
        skip_belief_sigma_update=False,
        embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1, diagonal_covariance=True,
        gauge_group="block_glk", family="gaussian_full",
        relational_mode="projected_filtering", relational_readout_mode="full_link_update",
        amp_dtype="fp32")
    base = {**config_values(), **base}
    base.update(over)
    base.setdefault("d_head", base["embed_dim"] // base["n_heads"])
    base.setdefault("relational_mode", "projected_filtering")
    base.setdefault("relational_readout_mode", "full_link_update")
    return types.SimpleNamespace(**base)


def test_finalize_run_forwards_disabled_embedding_policy_to_figure_driver(tmp_path, monkeypatch):
    torch.manual_seed(0)
    real_cfg = test_config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2,
                          batch_size=2)
    model = VFEModel(real_cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), real_cfg, model, dataset="synthetic", device=str(DEVICE))
    calls = []
    monkeypatch.setattr(
        "vfe3.run_artifacts._run_figures_isolated",
        lambda *a, **k: calls.append((a, k)) or True,
    )
    finalize_run(model, art, _finalize_ns(), test_loader=None)
    assert len(calls) == 1
    assert calls[0][1]["generate_embedding_figures"] is False


def test_finalize_run_forwards_enabled_embedding_policy_to_figure_driver(tmp_path, monkeypatch):
    torch.manual_seed(0)
    real_cfg = test_config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=2,
                          batch_size=2)
    model = VFEModel(real_cfg).to(DEVICE)
    art = RunArtifacts(str(tmp_path), real_cfg, model, dataset="synthetic", device=str(DEVICE))
    calls = []
    monkeypatch.setattr(
        "vfe3.run_artifacts._run_figures_isolated",
        lambda *a, **k: calls.append((a, k)) or True,
    )
    finalize_run(model, art, _finalize_ns(generate_embedding_figures=True), test_loader=None)
    assert len(calls) == 1
    assert calls[0][1]["generate_embedding_figures"] is True


# --------------------------------------------------------------------------- held-out geometry columns

def test_train_logs_held_out_geometry(tmp_path):
    torch.manual_seed(0)
    cfg = test_config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=4,
                     log_interval=2, eval_interval=2, batch_size=2)
    model = VFEModel(cfg).to(DEVICE)
    batches = [(torch.randint(0, 32, (2, 8), device=DEVICE),
                torch.randint(0, 32, (2, 8), device=DEVICE)) for _ in range(4)]
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic", device=str(DEVICE))
    train(model, batches, cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=batches, artifacts=art, device=DEVICE, generate_samples=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows and "val_holonomy_wilson" in rows[0]      # rectangular: column defined from row 0
    finite = [r for r in rows if r["val_holonomy_wilson"] not in ("", "nan", None)]
    assert not finite, "legacy geometry is unavailable on projected attention"
    assert any(r["val_relational_receiver_cost_after"] not in ("", "nan", None) for r in rows)


# --------------------------------------------------------------------------- ablation seed aggregation

def test_base_label_strips_seed_suffix():
    assert ablation._base_label("a2_on__s0") == "a2_on"
    assert ablation._base_label("a2_on__s12") == "a2_on"
    assert ablation._base_label("kappa=2.0") == "kappa=2.0"
    assert ablation._base_label("plain") == "plain"


def test_seed_aggregate_groups_seeds():
    rows = [{"label": "a__s0", "primary_val_ppl": "30.0"},
            {"label": "a__s1", "primary_val_ppl": "32.0"},
            {"label": "b", "primary_val_ppl": "40.0"},
            {"label": "c__s0", "primary_val_ppl": "inf"}]      # non-finite cell dropped
    agg = ablation._seed_aggregate(rows)
    by = {a["label"]: a for a in agg}
    assert by["a"]["n"] == 2 and abs(by["a"]["mean"] - 31.0) < 1e-9
    assert abs(by["a"]["sd"] - 2.0 ** 0.5) < 1e-9 and abs(by["a"]["cv"] - (2.0 ** 0.5) / 31.0) < 1e-9
    assert by["b"]["n"] == 1 and math.isnan(by["b"]["sd"]) and math.isnan(by["b"]["cv"])  # unknown, not 0
    assert "c" not in by                                       # no finite seed -> dropped
    assert [a["label"] for a in agg] == ["a", "b"]             # sorted by mean PPL


# --------------------------------------------------------------------------- scaling summary report


def _record_complete_scaling_cell(root, route, label, seed, scale_knob="n_params"):
    path = root / "scaling_design.json"
    design = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": 1,
        "routes": [],
        "seeds": [],
        "status": "complete",
        "cells": [],
    }
    design["routes"] = sorted(set([*design["routes"], route]))
    design["seeds"] = sorted(set([*design["seeds"], seed]))
    design["cells"].append({
        "route": route,
        "label": label,
        "seed": seed,
        "scale_knob": scale_knob,
        "run_dir": f"{route}/{label}/s{seed}",
        "status": "complete",
    })
    path.write_text(json.dumps(design), encoding="utf-8")


def _write_scaling_analysis_run(
    root,
    *,
    route,
    label,
    seed,
    n_params,
    test_ce,
    scale_knob="n_params",
    n_e_steps=1,
    estep_final_f_per_token=None,
    tokens_seen=1000,
    git_sha="git-a",
    train_sha="train-a",
    val_sha="val-a",
    test_sha="test-a",
):
    run = root / route / label / f"s{seed}"
    run.mkdir(parents=True)
    cfg = test_config(
        seed=seed,
        vocab_size=16,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        n_layers=1,
        n_e_steps=n_e_steps,
        family="gaussian_diagonal",
        gauge_group="block_glk",
    )
    config = serialize_config(cfg)
    effective_config = asdict(cfg)
    code_identity = {
        "git_sha": git_sha,
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    cell = {
        "schema_version": 2,
        "route": route,
        "scale_knob": scale_knob,
        "label": label,
        "seed": seed,
        "dataset": "synthetic",
        "config_sha256": hashlib.sha256(json.dumps(
            effective_config, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "code_identity": code_identity,
        "data_sources": {
            split: {"format": "pt", "size_bytes": len(split), "sha256": split[0] * 64}
            for split in ("train", "validation", "test")
        },
    }
    digest = hashlib.sha256(json.dumps(
        cell, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    cell["reuse_contract_sha256"] = digest
    metrics = {
        "n_params": n_params,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }
    (run / "summary.json").write_text(json.dumps({
        **metrics,
        "estep_final_f_per_token": estep_final_f_per_token,
        "scaling_reuse_contract_sha256": digest,
        "scaling_point": {
            **metrics,
            "n_learnable_params": n_params,
            "embed_dim": 4,
            "n_heads": cfg.n_heads,
            "n_gen": cfg.embed_dim * cfg.d_head,
            "gauge_group": cfg.gauge_group,
            "n_layers": 1,
            "n_e_steps": n_e_steps,
            "tokens_seen": tokens_seen,
        },
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "dataset": "synthetic",
        "config": config,
    }), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed,
        **code_identity,
        "train_data_sha256": train_sha,
        "val_data_sha256": val_sha,
        "test_data_sha256": test_sha,
        "data_sha256": test_sha,
    }), encoding="utf-8")
    _record_complete_scaling_cell(root, route, label, seed, scale_knob=scale_knob)


def test_all_scaling_tables_and_overlays_share_estimator(tmp_path, monkeypatch):
    # Two routes, four sizes each, and two seeds per cell give every table and
    # overlay enough sizes for the requested offset estimator.  With symmetric
    # seed values mean +/- 0.05, SEM is exactly 0.05 at every point.
    import pytest
    pytest.importorskip("scipy")

    for route_i, route in enumerate(("route_a", "route_b")):
        for n_params in (100, 200, 400, 800):
            mean = 2.0 + (5.0 + route_i) * n_params ** -0.3
            for seed, delta in ((1, -0.05), (2, 0.05)):
                _write_scaling_analysis_run(
                    tmp_path,
                    route=route,
                    label=f"n{n_params}",
                    seed=seed,
                    n_params=n_params,
                    test_ce=mean + delta,
                )

    real_fit = figs._fit_power_law
    calls = []

    def _record_fit(N, L, *, weights=None, with_offset=False):
        calls.append({
            "N": np.asarray(N, dtype=float).copy(),
            "L": np.asarray(L, dtype=float).copy(),
            "weights": None if weights is None else np.asarray(weights, dtype=float).copy(),
            "with_offset": with_offset,
        })
        return real_fit(N, L, weights=weights, with_offset=with_offset)

    monkeypatch.setattr(figs, "_fit_power_law", _record_fit)
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", True)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)

    scaling_analysis.analyze()

    assert len(calls) >= 8
    assert all(call["with_offset"] is True for call in calls)
    assert all(call["weights"] is not None for call in calls)
    for call in calls:
        np.testing.assert_allclose(call["weights"], (call["L"] / 0.05) ** 2, rtol=1e-12)

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"]["form"] == "offset_power_law"
    assert all(fit["form"] == "offset_power_law" for fit in summary["per_route"].values())


def test_scaling_summary_persists_code_and_data_drift(tmp_path, monkeypatch):
    _write_scaling_analysis_run(
        tmp_path,
        route="grow_K",
        label="n100",
        seed=1,
        n_params=100,
        test_ce=4.5,
        tokens_seen=1000,
        git_sha="git-a",
        train_sha="train-a",
    )
    _write_scaling_analysis_run(
        tmp_path,
        route="grow_K",
        label="n200",
        seed=1,
        n_params=200,
        test_ce=4.2,
        tokens_seen=2000,
        git_sha="git-b",
        train_sha="train-b",
    )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    provenance = summary["provenance"]
    assert provenance["git_sha"] == ["git-a", "git-b"]
    assert provenance["train_data_sha256"] == ["train-a", "train-b"]
    assert provenance["val_data_sha256"] == ["val-a"]
    assert provenance["test_data_sha256"] == ["test-a"]
    assert provenance["code_drift"] is True
    assert provenance["mixed_corpus"] is True
    assert provenance["token_budgets"] == [1000, 2000]
    assert provenance["token_budget_varies"] is True
    assert provenance["missing"] == {
        "git_sha": 0,
        "train_data_sha256": 0,
        "val_data_sha256": 0,
        "test_data_sha256": 0,
    }
    assert summary["pooled_fit_status"] == "confounded"
    assert set(summary["pooled_fit_confounds"]) == {"code_drift", "mixed_corpus", "token_budget_varies"}

    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    for value in ("git-a", "git-b", "train-a", "train-b", "1000", "2000", "confounded"):
        assert value in text


def test_divergent_routes_mark_pooled_fit_confounded(tmp_path, monkeypatch):
    for route, ce0 in (("grow_K", 4.6), ("blocks_K48_tied_2x", 4.1)):
        for i, n_params in enumerate((100, 200)):
            _write_scaling_analysis_run(
                tmp_path,
                route=route,
                label=f"n{n_params}",
                seed=1,
                n_params=n_params,
                test_ce=ce0 - 0.1 * i,
            )
    ancova = {
        "routes": {"blocks_K48_tied_2x": 2, "grow_K": 2},
        "testable": True,
        "F": 12.0,
        "df1": 2,
        "df2": 4,
        "p_value": 0.004,
        "rss_pooled": 0.4,
        "rss_full": 0.04,
        "collapses": False,
    }
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "ancova_frontier_collapse", lambda *args, **kwargs: ancova)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"] is not None
    assert summary["pooled_fit_status"] == "confounded"
    assert summary["pooled_fit_confounds"] == ["routes_diverge"]
    assert summary["frontier_collapse"] == ancova
    assert summary["route_notes"]["blocks_K48_tied_2x"] == (
        "tied structural ablation; not a strict full-covariance pure control"
    )

    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "Pooled L(N) power law" in text
    assert "confounded" in text
    assert "tied structural ablation" in text
    assert "not a strict full-covariance pure control" in text


def test_indeterminate_ancova_is_unassessed_not_divergent(tmp_path, monkeypatch, capsys):
    for route, ce0 in (("route_a", 4.6), ("route_b", 4.4)):
        for i, n_params in enumerate((100, 200)):
            _write_scaling_analysis_run(
                tmp_path,
                route=route,
                label=f"n{n_params}",
                seed=1,
                n_params=n_params,
                test_ce=ce0 - 0.1 * i,
            )
    ancova = {
        "routes": {"route_a": 2, "route_b": 2},
        "testable": True,
        "F": 1.0,
        "df1": 2,
        "df2": 4,
        "p_value": float("nan"),
        "rss_pooled": 0.2,
        "rss_full": 0.1,
        "collapses": None,
    }
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "ancova_frontier_collapse", lambda *args, **kwargs: ancova)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"] is not None
    # Two sizes per route: not confounded (the indeterminate ANCOVA is not divergence), and below
    # the four-size testability floor (audit 2026-09-23 #18) the status is 'not_testable'.
    assert summary["pooled_fit_status"] == "not_testable"
    assert summary["pooled_fit_confounds"] == []
    console = capsys.readouterr().out.lower()
    assert "indeterminate" in console
    assert "routes diverge" not in console
    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8").lower()
    assert "indeterminate" in text
    assert "routes diverge" not in text


def test_duplicate_single_n_does_not_persist_nan_fit(tmp_path, monkeypatch):
    for label, ce in (("arm_a", 4.5), ("arm_b", 4.3)):
        _write_scaling_analysis_run(
            tmp_path,
            route="grow_K_mup",
            label=label,
            seed=1,
            n_params=100,
            test_ce=ce,
        )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", True)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    raw = (tmp_path / "scaling_summary.json").read_text(encoding="utf-8")
    summary = json.loads(raw)
    assert summary["pooled_fit"] is None
    assert summary["pooled_fit_status"] == "not_fitted"
    assert summary["per_route"] == {}

    def _reject_nonfinite_json(value):
        pytest.fail(f"scaling summary persisted non-finite JSON constant {value!r}")

    assert json.loads(raw, parse_constant=_reject_nonfinite_json) == summary
    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8").lower()
    assert "status: **not_fitted**" in text
    assert "no pooled estimate is available" in text


def test_ancova_requires_distinct_sizes_per_route(tmp_path, monkeypatch):
    for route, n_params, ce0 in (("route_a", 100, 4.5), ("route_b", 200, 4.2)):
        for arm in range(3):
            _write_scaling_analysis_run(
                tmp_path,
                route=route,
                label=f"arm_{arm}",
                seed=1,
                n_params=n_params,
                test_ce=ce0 - 0.02 * arm,
            )
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["pooled_fit"] is not None
    assert summary["frontier_collapse"]["testable"] is False
    assert summary["frontier_collapse"]["routes"] == {}
    # Two distinct sizes: below the four-size testability floor (audit 2026-09-23 #18).
    assert summary["pooled_fit_status"] == "not_testable"


def test_scaling_md_persists_not_fitted_status(tmp_path):
    out = tmp_path / "SCALING_ANALYSIS.md"
    scaling_analysis._write_scaling_md(out, {
        "input_dir": str(tmp_path),
        "n_runs": 1,
        "n_param_points": 1,
        "n_inference_points": 0,
        "with_offset": True,
        "pooled_fit": None,
        "pooled_fit_status": "not_fitted",
        "pooled_fit_confounds": [],
    })

    text = out.read_text(encoding="utf-8")
    assert text.count("## Pooled L(N) power law") == 1
    assert "status: **not_fitted**" in text
    assert "no pooled estimate is available" in text


def test_write_scaling_md(tmp_path):
    summary = {
        "input_dir": str(tmp_path), "n_runs": 6, "with_offset": True,
        "n_param_points": 4, "n_inference_points": 2,
        "pooled_fit": {"form": "E + A N^-alpha", "alpha": 0.12, "alpha_ci": [0.08, 0.16],
                       "A": 3.4, "E": 2.1, "r2": 0.98, "n_points": 4},
        "per_route": {"grow_K": {"alpha": 0.11, "r2": 0.97, "n_sizes": 4}},
        "frontier_collapse": {"testable": True, "collapses": True, "df1": 2, "df2": 8,
                              "F": 0.5, "p_value": 0.62},
        "estep_structural": {"n_arms": 3, "pearson_ne_final_f": -0.95, "pearson_final_f_test_ce": 0.1},
        "provenance": {"git_sha": ["abc"], "train_data_sha256": ["train"],
                       "val_data_sha256": ["val"], "test_data_sha256": ["test"],
                       "data_sha256": ["test"], "code_drift": False, "mixed_corpus": False,
                       "token_budgets": [1000], "token_budget_varies": False, "missing": {}},
        "pooled_fit_status": "clean", "pooled_fit_confounds": [],
        "route_notes": {},
    }
    out = tmp_path / "SCALING_ANALYSIS.md"
    scaling_analysis._write_scaling_md(out, summary)
    text = out.read_text(encoding="utf-8")
    for section in ("Pooled L(N) power law", "Per-route exponents", "Frontier-collapse F-test",
                    "E-step structural-EM check", "Provenance and confounds"):
        assert section in text
    assert "expect strongly negative" not in text
    assert "expect ~0" not in text
    assert "separate objectives" in text

def test_analyze_reports_target_blind_correlations_without_expected_signs(
    tmp_path,
    monkeypatch,
    capsys,
):
    for n_e_steps, final_f, test_ce in (
        (1, 3.0, 4.0),
        (2, 2.4, 4.2),
        (3, 2.1, 4.1),
    ):
        _write_scaling_analysis_run(
            tmp_path,
            route=scaling_analysis.INFERENCE_ROUTE,
            label=f"T{n_e_steps}",
            seed=1,
            n_params=100,
            test_ce=test_ce,
            scale_knob="n_e_steps",
            n_e_steps=n_e_steps,
            estep_final_f_per_token=final_f,
        )

    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)

    scaling_analysis.analyze()

    console = capsys.readouterr().out
    assert "Pearson(n_e_steps, final F/token)" in console
    assert "Pearson(final F/token, test CE)" in console
    assert "expect strongly negative" not in console
    assert "expect ~0" not in console
    assert "target blindness alone implies no Pearson sign" in console
    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["estep_structural"]["interpretation"].endswith("no Pearson sign.")




# =============================================================================
# PB-07 registry-completion Task 5: end-to-end production-driver integration.
#
# Both tests below monkeypatch the SPECIFIC registry slot each new figure claims
# (``vfe3.viz.figures._FIGURES[name]``), not a direct-import reference. If either production
# driver ever regressed to calling its plotter by direct import (bypassing ``get_figure``), the
# stub would never run, the asserted PNG would still carry the old real-plot bytes (or be
# missing entirely), and the captured-kwargs assertion below would KeyError -- so this is a
# substitution test for "resolves by name", not just an existence check. Both drivers avoid any
# model replay: ``scaling_analysis.analyze()`` reads only persisted JSON fixtures, and
# ``ablation.main()`` runs with ``run_single`` stubbed to a fast fabricated cell.
# =============================================================================

def _write_scaling_validation_run(root, *, route, label, embed_dim, n_params, test_ce,
                                  best_val_ppl, seed=1, wall_time_s=12.5):
    r"""One persisted ``scaling_analysis`` run directory carrying BOTH the test-metric fields
    ``aggregate_points`` reads (feeding the legacy scaling_ce_vs_params.png fit) and the
    top-level ``best_val_ppl`` / ``wall_time_s`` fields ``aggregate_validation_points`` reads
    (feeding the two PB-07 registered validation figures)."""
    run = root / route / label / f"s{seed}"
    run.mkdir(parents=True)
    cfg = test_config(
        seed=seed,
        vocab_size=16,
        embed_dim=embed_dim,
        n_heads=2,
        max_seq_len=8,
        n_layers=1,
        n_e_steps=1,
        family="gaussian_diagonal",
        gauge_group="block_glk",
    )
    config = serialize_config(cfg)
    effective_config = asdict(cfg)
    code_identity = {
        "git_sha": "git-a",
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    cell = {
        "schema_version": 2,
        "route": route,
        "scale_knob": "embed_dim",
        "label": label,
        "seed": seed,
        "dataset": "synthetic",
        "config_sha256": hashlib.sha256(json.dumps(
            effective_config, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "code_identity": code_identity,
        "data_sources": {
            split: {"format": "pt", "size_bytes": len(split), "sha256": split[0] * 64}
            for split in ("train", "validation", "test")
        },
    }
    digest = hashlib.sha256(json.dumps(
        cell, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    cell["reuse_contract_sha256"] = digest
    metrics = {
        "n_params": n_params,
        "test_ce": test_ce,
        "test_ppl": math.exp(test_ce),
        "test_bits_per_token": test_ce / math.log(2.0),
        "test_bpc": None,
    }
    (run / "summary.json").write_text(json.dumps({
        **metrics,
        "best_val_ppl": best_val_ppl,
        "wall_time_s": wall_time_s,
        "scaling_reuse_contract_sha256": digest,
        "scaling_point": {
            **metrics,
            "n_learnable_params": n_params,
            "embed_dim": embed_dim,
            "n_heads": cfg.n_heads,
            "n_gen": embed_dim * cfg.d_head,
            "gauge_group": cfg.gauge_group,
            "n_layers": 1,
            "n_e_steps": 1,
            "tokens_seen": 1000,
        },
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({
        "dataset": "synthetic",
        "config": config,
    }), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed, **code_identity, "train_data_sha256": "train-a",
        "val_data_sha256": "val-a", "test_data_sha256": "test-a", "data_sha256": "test-a",
    }), encoding="utf-8")
    _record_complete_scaling_cell(root, route, label, seed, "embed_dim")


def _stub_registered_figure(name, cap):
    r"""A registered-figure stub that records the exact kwargs its caller resolved for ``name``
    and writes a real (nonempty) PNG to ``path``, standing in for the true plotter."""
    def _fn(*, path=None, **kwargs):
        cap[name] = kwargs
        fig = plt.figure()
        cap[f"{name}__figure"] = fig
        if path is not None:
            fig.savefig(path)
        return fig
    return _fn


def test_scaling_analyze_dispatches_registered_figures_by_name_and_keeps_legacy_outputs(
    tmp_path, monkeypatch,
):
    # Tiny persisted scaling fixture: 3 embed_dim points on route "grow_K" (AXIS_ROUTES maps the
    # embed_dim axis to this route), each carrying a finite test_ce (legacy fit) and best_val_ppl
    # (the PB-07 validation figures) so both the legacy and the two new figures have enough points.
    _write_scaling_validation_run(tmp_path, route="grow_K", label="K8",  embed_dim=8,
                                  n_params=100, test_ce=4.5, best_val_ppl=30.0)
    _write_scaling_validation_run(tmp_path, route="grow_K", label="K16", embed_dim=16,
                                  n_params=200, test_ce=4.2, best_val_ppl=24.0)
    _write_scaling_validation_run(tmp_path, route="grow_K", label="K32", embed_dim=32,
                                  n_params=400, test_ce=4.0, best_val_ppl=20.0)

    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 0)

    cap = {}
    monkeypatch.setitem(figs._FIGURES, "capacity_scaling", _stub_registered_figure("capacity_scaling", cap))
    monkeypatch.setitem(figs._FIGURES, "pareto_frontier", _stub_registered_figure("pareto_frontier", cap))

    scaling_analysis.analyze()

    fig_dir = tmp_path / "figures"
    # The two PB-07 figures exist ONLY because the stub registered under their name ran --
    # emit_registered_figures resolved them through get_figure("capacity_scaling" / "pareto_frontier").
    assert (fig_dir / "capacity_scaling.png").exists() and (fig_dir / "capacity_scaling.png").stat().st_size > 0
    assert (fig_dir / "pareto_frontier.png").exists() and (fig_dir / "pareto_frontier.png").stat().st_size > 0
    assert "embed_dim" in cap["capacity_scaling"]["scaling"]
    assert {"bits_per_token", "n_params"} <= set(cap["pareto_frontier"]["points"])

    # Legacy scaling outputs are unaffected: the test-metric report, csv, and headline figure.
    assert (tmp_path / "scaling_points.csv").exists()
    assert (tmp_path / "scaling_summary.json").exists()
    assert (tmp_path / "SCALING_ANALYSIS.md").exists()
    assert (fig_dir / "scaling_ce_vs_params.png").exists()


_FIXED_ABLATION_CODE_IDENTITY = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}


def _fake_ablation_source_ok(dataset, split, *, cache_dir=None):
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def test_ablation_renderer_dispatches_registered_figures_by_name_and_keeps_legacy_outputs(
    tmp_path, monkeypatch,
):
    # Stub the contract-building identity seams exactly as the existing PB-07 ablation-report
    # tests do (tests/test_ablation_reporting.py::_stub_sweep_identity), and stub run_single so no
    # model ever trains: each cell's terminal PPL is fabricated, and forest cells additionally
    # publish a real val_token_nats.pt with a correctly recomputed identity.
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_FIXED_ABLATION_CODE_IDENTITY))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_ablation_source_ok)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)

    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(
        max_seq_len=8, use_head_mixer=True, attention_mode="projected_predictive",
        family="gaussian_full", decode_mode="full_chunked", e_phi_lr=0.0,
        max_steps=2, warmup_steps=1))
    monkeypatch.setattr(ablation, "SWEEPS", {"component_ablation_forest": {
        "description": "paired-token component ablation forest",
        "configs": [{"label": "baseline"}, {"label": "head_mixer_off", "use_head_mixer": False}],
        "paired_token_bootstrap": True, "forest_baseline_label": "baseline"}})
    monkeypatch.setattr(ablation, "CONFIG", dict(
        list_only=False, sweep="component_ablation_forest", device="cpu",
        dataset="wikitext-103", max_tokens=None, max_steps=None,
        stop_after_steps=None, seed=6, target_n_params=100,
        max_param_relative_deviation=0.02, resume=False, output_dir=str(tmp_path)))
    monkeypatch.setattr(ablation, "DATA_SEED", 3)

    forest_offsets = {"baseline": 0.0, "head_mixer_off": 0.4}

    def fake_run_single(label, overrides, run_dir, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "checkpoints" / "terminal.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{label}".encode("utf-8"))
        result = {"label": label, "error_kind": None, "seed": 6,
                  "overrides": ablation._jsonable(overrides),
                  "terminal_checkpoint": str(checkpoint),
                  "_loaded_data_sources": {
                      split: _fake_ablation_source_ok("wikitext-103", split)
                      for split in ("train", "validation")
                  }}
        if kwargs.get("paired_token_bootstrap"):
            idx = list(forest_offsets).index(label)
            g = torch.Generator().manual_seed(1000 + idx)
            vec = torch.rand(48, generator=g, dtype=torch.float32) * 0.1 + 1.0 + forest_offsets[label]
            tpath = run_dir / "val_token_nats.pt"
            torch.save(vec, tpath)
            # A real cell's RunArtifacts config.json carries the evaluation window that the
            # forest's sequence-cluster bootstrap resamples (audit 2026-09-23 #19).
            (run_dir / "config.json").write_text(json.dumps({"config": {"max_seq_len": 8}}),
                                                 encoding="utf-8")
            ppl = float(vec.mean())
            result.update({
                "primary_val_ppl": ppl, "final_val_ppl": ppl,
                "val_token_nats_path":       "val_token_nats.pt",
                "val_token_nats_sha256":     hashlib.sha256(tpath.read_bytes()).hexdigest(),
                "val_token_nats_size_bytes": tpath.stat().st_size,
                "val_token_nats_numel":      int(vec.numel()),
                "val_token_nats_dtype":      str(vec.dtype),
            })
        else:
            mu, sigma = overrides["e_q_mu_lr"], overrides["e_q_sigma_lr"]
            ppl = 10.0 + (mu - 0.7) ** 2 * 5.0 + (sigma - 0.0005) ** 2 * 2000.0
            result.update({"primary_val_ppl": ppl, "final_val_ppl": ppl})
        return result

    monkeypatch.setattr(ablation, "run_single", fake_run_single)

    cap = {}
    monkeypatch.setitem(figs._FIGURES, "ablation_forest", _stub_registered_figure("ablation_forest", cap))
    monkeypatch.setitem(figs._FIGURES, "lr_grid_heatmap", _stub_registered_figure("lr_grid_heatmap", cap))

    # Only the forest sweep is exercised: the e_q_mu_sigma_lr_grid sweep and its _GRID_*_LRS
    # axes were retired in the 2026-09-18 sweep cleanup, which leaves lr_grid_heatmap without a
    # registered producer.
    fig_dir = tmp_path / "figures"
    name = "component_ablation_forest"
    ablation.run_sweep(
        name,
        tmp_path,
        dataset="wikitext-103",
        device=DEVICE,
        seed=6,
        resume=False,
    )
    ablation._render_sweep_figures(tmp_path / name, fig_dir)

    forest_path = fig_dir / "component_ablation_forest_ablation_forest.png"
    assert forest_path.exists() and forest_path.stat().st_size > 0
    assert any(
        "Gauge classification" in text.get_text()
        for text in cap["ablation_forest__figure"].texts
    )
    assert {r["label"] for r in cap["ablation_forest"]["rows"]} == set(forest_offsets)

    # Legacy per-sweep outputs (CSV, metadata, PPL figure) survive.
    sweep_dir = tmp_path / name
    assert (sweep_dir / "sweep_results.csv").exists()
    assert (sweep_dir / "sweep_meta.json").exists()
    assert (fig_dir / f"{name}.png").exists()


def test_pure_path_reports_mlp_as_optional_nonintertwiner():
    """The default route remains pure; a coordinate MLP is explicitly reported as non-pure."""
    off = _pure_path_report(_pure_ns(use_block_mlp=False), [])
    assert off["pure_flags"]["no_block_mlp"] is True
    assert off["on_pure_path"] is False
    assert off["config_toggles"]["block_mlp_structural_mode"] == "disabled"
    assert off["config_toggles"]["block_mlp_covariance_contract"] == "not_applicable"

    on = _pure_path_report(_pure_ns(use_block_mlp=True), [])
    assert on["pure_flags"]["no_block_mlp"] is False
    assert on["config_toggles"]["block_mlp_structural_mode"] == "coordinate_mean_only_nonintertwiner"
    assert on["config_toggles"]["block_mlp_covariance_contract"] == "passthrough"
    assert on["gauge_flags"]["block_mlp_intertwiner_compatible"] is False
    assert on["on_pure_path"] is False
    assert on["on_gauge_pure_path"] is False


def test_cost_model_counts_untied_block_mlp_params_and_flops():
    """MLP accounting follows the established multiply-add FLOP convention."""
    K, expansion, layers = 6, 3, 2
    model = _cost_model(K=K, n_gen=5, n_blocks=2)
    cfg = _cost_cfg(n_layers=layers, use_block_mlp=True, block_mlp_expansion=expansion)
    off_cfg = _cost_cfg(n_layers=layers, use_block_mlp=False, block_mlp_expansion=expansion)
    out = _cost_model_fields(model, cfg, n_params=123, tokens_seen=13)
    off = _cost_model_fields(model, off_cfg, n_params=123, tokens_seen=13)
    per_layer = 2 * expansion * K * K + (expansion + 1) * K
    mlp_flops = 4.0 * layers * expansion * K * K
    base_active = (2 * K + 5) + 11 * K + 4 * layers * K * K

    assert off["active_params_per_token"] == base_active
    assert out["active_params_per_token"] - base_active == layers * per_layer
    assert out["block_mlp_params"] == layers * per_layer
    assert out["flops_per_token_block_mlp"] == mlp_flops
