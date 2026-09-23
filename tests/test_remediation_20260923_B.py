r"""Group B remediation of the 2026-09-23 deep audit: run artifacts, metrics, viz, scaling analysis.

Tiny CPU models only (vocab <= 7, K = 4, two heads, one layer). Each test pins one punch-list item:
#3(e) persisted inert set, #9 frozen-covariance Sigma_q relabeling, #13 executed-route pure-path
facets, #18 scaling testability / lack of fit / non-embedding axis, #19 sequence-cluster forest
bootstrap, #2 realized decode temperature, #23 resume seed-drift wording, and the Low items
(mixed-generation provenance, t-based noise band, finalize best-weights guard).
"""

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import multiseed_analysis as ms
import scaling_analysis
from tests.test_reporting_additions import _pure_ns, _write_scaling_analysis_run
from vfe3 import metrics
from vfe3 import run_artifacts
from vfe3.config import VFE3Config
from vfe3.metrics import bootstrap_token_ce_band
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    RunArtifacts,
    _cost_model_fields,
    _decode_tau_eff,
    _pure_path_report,
    _write_provenance,
    _write_research_artifacts,
    finalize_run,
    load_checkpoint,
)
from vfe3.train import build_optimizer
from vfe3.viz import figures as figs
from vfe3.viz.run_loading import load_run_config
from vfe3.viz.sweep_adapters import ablation_forest_kwargs


def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vfe3.run_artifacts._git_code_identity",
        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})


def _cfg(**overrides: object) -> VFE3Config:
    values = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                  n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                  warmup_steps=1, max_steps=4)
    values.update(overrides)
    return VFE3Config(**values)


def _projected_cfg(**overrides: object) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1, n_e_steps=1,
                  family="gaussian_diagonal", gauge_group="block_glk",
                  skip_belief_sigma_update=False, e_phi_lr=0.0, oracle_unroll_grad=True,
                  warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0,
                  attention_mode="projected_predictive")
    values.update(overrides)
    return VFE3Config(**values)


# --------------------------------------------------------------------------- #3(e) inert set


def test_config_json_records_inactive_fields_beside_config(tmp_path: Path, monkeypatch) -> None:
    _no_git(monkeypatch)
    cfg = _cfg()
    RunArtifacts(tmp_path / "r", cfg, VFEModel(cfg), save_model_artifacts=False)
    path = tmp_path / "r" / "config.json"
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["inactive_fields"] == cfg.inactive_fields()
    assert record["inactive_fields"]                              # the tiny default has inert knobs
    assert "inactive_fields" not in record["config"]
    loaded, _ = load_run_config(tmp_path / "r")                  # reload ignores the sibling key
    assert asdict(loaded) == asdict(cfg)

    stripped = tmp_path / "stripped.json"
    stripped.write_text(json.dumps({k: v for k, v in record.items() if k != "inactive_fields"}),
                        encoding="utf-8")
    assert ms._config_fingerprint(path) == ms._config_fingerprint(stripped)


# --------------------------------------------------------------------------- #9 Sigma_q calibration


@pytest.mark.parametrize(
    ("overrides", "frozen"),
    [
        ({}, False),
        ({"skip_belief_sigma_update": True}, True),
        ({"attention_mode": "projected_predictive"}, True),
        ({"attention_mode": "projected_conditional"}, True),
        ({"attention_mode": "projected_predictive", "use_block_mlp": True,
          "block_mlp_covariance": "delta_full"}, False),
        ({"skip_belief_sigma_update": True, "use_cg_coupling": True,
          "cg_covariance_mode": "delta_full"}, False),
    ],
)
def test_covariance_frozen_at_encode_follows_the_executed_route(overrides, frozen) -> None:
    cfg = types.SimpleNamespace(**{"attention_mode": "transported",
                                   "skip_belief_sigma_update": False, **overrides})
    assert metrics.covariance_frozen_at_encode(cfg) is frozen


class _Artifacts:
    def __init__(self) -> None:
        self.saved: dict = {}

    def save_json(self, name: str, obj: dict) -> None:
        self.saved[name] = obj


def test_frozen_route_relabels_sigma_statistics_and_withholds_gate(monkeypatch) -> None:
    torch.manual_seed(0)
    cfg = _projected_cfg()
    model = VFEModel(cfg).eval()
    with torch.no_grad():                                        # distinct per-token variances
        model.prior_bank.sigma_log_embed.add_(0.3 * torch.randn_like(model.prior_bank.sigma_log_embed))
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 5]])
    loader = [(tokens, targets)]

    # The premise: on this route tr Sigma is the input token's encode-table trace.
    from vfe3.viz.extract import belief_ce_bank
    bank = belief_ce_bank(model, loader, max_batches=1)
    encoded = model.prior_bank.encode(tokens).sigma
    expected = metrics.sigma_trace(encoded, diagonal=cfg.diagonal_covariance,
                                   family=cfg.family).reshape(-1)
    assert torch.allclose(bank["tr_sigma"], expected.to(bank["tr_sigma"].dtype))

    monkeypatch.setattr(run_artifacts, "_fd_gradient_check", lambda *a, **k: float("nan"))
    artifacts = _Artifacts()
    _write_research_artifacts(model, artifacts, cfg, None, loader, torch.device("cpu"),
                              logging.getLogger("test"))
    research = artifacts.saved["research.json"]
    assert research["sigma_trace_source"] == "input_token_variance"
    assert research["sigma_trace_cv_gate_pass"] is None
    assert "frozen at encode" in research["sigma_trace_cv_gate_reason"]
    assert math.isfinite(research["sigma_trace_cv"])


def test_live_covariance_route_keeps_the_calibration_gate(monkeypatch) -> None:
    torch.manual_seed(0)
    cfg = _cfg(skip_belief_sigma_update=False)
    model = VFEModel(cfg).eval()
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]])
    loader = [(tokens, torch.roll(tokens, -1, dims=1))]
    monkeypatch.setattr(run_artifacts, "_fd_gradient_check", lambda *a, **k: float("nan"))
    artifacts = _Artifacts()
    _write_research_artifacts(model, artifacts, cfg, None, loader, torch.device("cpu"),
                              logging.getLogger("test"))
    research = artifacts.saved["research.json"]
    assert research["sigma_trace_source"] == "belief_covariance"
    assert type(research["sigma_trace_cv_gate_pass"]) is bool
    assert "sigma_trace_cv_gate_reason" not in research


def test_sigma_figures_retitle_input_token_variance() -> None:
    generator = torch.Generator().manual_seed(0)
    bank = {"tr_sigma": torch.rand(64, generator=generator), "ce": torch.rand(64, generator=generator)}
    frozen = figs.plot_sigma_stratified_error(bank, n_bins=4, n_boot=8, frozen_covariance=True)
    live = figs.plot_sigma_stratified_error(bank, n_bins=4, n_boot=8)
    scatter = figs.plot_sigma_ce_scatter(bank, frozen_covariance=True)
    try:
        assert "input-token variance" in frozen.axes[0].get_title()
        assert "input-token variance" in frozen.axes[0].get_xlabel()
        assert "belief uncertainty" in live.axes[0].get_title()
        assert "input-token variance" in scatter.axes[0].get_xlabel()
    finally:
        for fig in (frozen, live, scatter):
            plt.close(fig)


# --------------------------------------------------------------------------- #13 pure-path facets


def test_pure_path_facets_follow_the_executed_route() -> None:
    live = _pure_path_report(_projected_cfg(), [])                  # the launcher's route
    assert live["pure_flags"]["full_sigma_update"] is False
    assert live["pure_flags"]["gradient_e_step_update"] == "not_applicable"
    toggles = live["config_toggles"]
    assert toggles["fixed_covariance_surrogate"] is True
    for key in ("spd_retraction_route", "spd_retraction_exact",
                "spd_retraction_trust_projection_active",
                "spd_retraction_exponential_clipping_active"):
        assert toggles[key] == "not_applicable"
    assert live["on_pure_path"] is False                            # still off it, for its own reason

    conditional = _pure_path_report(_pure_ns(attention_mode="projected_conditional"), [])
    assert conditional["pure_flags"]["gradient_e_step_update"] is True
    assert conditional["pure_flags"]["full_sigma_update"] is False

    transported = _pure_path_report(_pure_ns(), [])
    assert transported["pure_flags"]["full_sigma_update"] is True
    assert transported["pure_flags"]["gradient_e_step_update"] is True
    assert transported["config_toggles"]["fixed_covariance_surrogate"] is False
    assert transported["config_toggles"]["spd_retraction_route"].startswith("airm")
    assert transported["on_pure_path"] is True

    mm_exact = _pure_path_report(_pure_ns(e_step_update="mm_exact"), [])
    assert mm_exact["pure_flags"]["gradient_e_step_update"] is False
    assert mm_exact["config_toggles"]["spd_retraction_route"] == "not_applicable"


# --------------------------------------------------------------------------- #18 scaling analysis


def _scaling_run(root: Path, sizes, ce_by_size, *, seeds=(1, 2), noise=0.01) -> None:
    for n_params in sizes:
        for seed, sign in zip(seeds, (-1.0, 1.0)):
            _write_scaling_analysis_run(root, route="grow_K", label=f"n{n_params}", seed=seed,
                                        n_params=n_params,
                                        test_ce=ce_by_size(n_params) * math.exp(sign * noise))


def _analyze(root: Path, monkeypatch) -> dict:
    monkeypatch.setitem(scaling_analysis.CONFIG, "input_dir", str(root))
    monkeypatch.setitem(scaling_analysis.CONFIG, "with_offset", False)
    monkeypatch.setitem(scaling_analysis.CONFIG, "n_bootstrap", 50)
    monkeypatch.setattr(scaling_analysis, "_make_figures", lambda *args, **kwargs: None)
    scaling_analysis.analyze()
    return json.loads((root / "scaling_summary.json").read_text(encoding="utf-8"))


def test_three_sizes_are_not_testable_and_withhold_r2_and_ci(tmp_path: Path, monkeypatch) -> None:
    _scaling_run(tmp_path, (100, 200, 400), lambda n: {100: 4.0, 200: 4.0, 400: 3.2}[n])
    summary = _analyze(tmp_path, monkeypatch)
    assert summary["pooled_fit_status"] == "not_testable"
    assert summary["pooled_fit"]["r2"] is None
    assert summary["pooled_fit"]["alpha_ci"] is None
    assert summary["pooled_fit"]["testable"] is False
    assert summary["per_route"]["grow_K"]["r2"] is None
    text = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "R^2 withheld" in text and "not_testable" in text


def test_four_size_power_law_passes_lack_of_fit(tmp_path: Path, monkeypatch) -> None:
    _scaling_run(tmp_path, (100, 200, 400, 800), lambda n: 2.0 * n ** -0.1)
    summary = _analyze(tmp_path, monkeypatch)
    lof = summary["pooled_fit_lack_of_fit"]
    assert lof["testable"] is True and lof["rejected"] is False
    assert (lof["df_lof"], lof["df_pe"]) == (2, 4)
    assert summary["pooled_fit"]["r2"] is not None
    assert summary["pooled_fit_status"] == "clean"


def test_four_size_curved_frontier_fails_lack_of_fit(tmp_path: Path, monkeypatch) -> None:
    _scaling_run(tmp_path, (100, 200, 400, 800),
                 lambda n: {100: 4.0, 200: 4.0, 400: 4.0, 800: 3.0}[n], noise=0.001)
    summary = _analyze(tmp_path, monkeypatch)
    assert summary["pooled_fit_lack_of_fit"]["rejected"] is True
    assert summary["pooled_fit_status"] == "lack_of_fit"


def test_single_seed_lack_of_fit_is_untestable_so_status_is_unassessed(tmp_path: Path,
                                                                       monkeypatch) -> None:
    _scaling_run(tmp_path, (100, 200, 400, 800), lambda n: 2.0 * n ** -0.1, seeds=(1,))
    summary = _analyze(tmp_path, monkeypatch)
    assert summary["pooled_fit_lack_of_fit"]["testable"] is False
    assert summary["pooled_fit_status"] == "unassessed"


def test_non_embedding_axis_is_fit_when_recorded(tmp_path: Path, monkeypatch) -> None:
    _scaling_run(tmp_path, (100, 200, 400, 800), lambda n: 2.0 * n ** -0.1)
    for summary_path in tmp_path.glob("**/summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["scaling_point"]["n_non_embedding_params"] = payload["scaling_point"]["n_params"] // 10
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = _analyze(tmp_path, monkeypatch)
    ne = summary["pooled_fit_non_embedding"]
    assert ne["x_axis"] == "n_non_embedding_params"
    assert ne["alpha"] == pytest.approx(summary["pooled_fit"]["alpha"], rel=1e-9)  # N/10: same slope


def test_cost_model_records_non_embedding_parameters() -> None:
    cfg = _cfg()
    model = VFEModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    out = _cost_model_fields(model, cfg, n_params, 16)
    pb = model.prior_bank
    tables = sum(getattr(pb, name).numel() for name in ("mu_embed", "sigma_log_embed", "phi_embed")
                 if getattr(pb, name, None) is not None)
    tables += sum(getattr(pb, name).numel() for name in
                  ("output_proj_weight", "output_proj_bias", "decode_mu_embed", "decode_sigma_log_embed")
                  if isinstance(getattr(pb, name, None), torch.nn.Parameter))
    tables += model.pos_phi_free.numel() if hasattr(model, "pos_phi_free") else 0
    assert out["n_token_table_params"] == tables
    assert out["n_non_embedding_params"] == n_params - tables > 0


# --------------------------------------------------------------------------- #19 forest bootstrap


def test_cluster_bootstrap_resamples_whole_sequences() -> None:
    generator = torch.Generator().manual_seed(0)
    n_seq, window = 32, 16
    base = torch.rand(n_seq * window, generator=generator)
    shared = (0.5 * torch.randn(n_seq, 1, generator=generator)).expand(n_seq, window).reshape(-1)
    arm = base + shared + 0.01 * torch.randn(n_seq * window, generator=generator)

    token = bootstrap_token_ce_band(arm, base, n_boot=400, seed=0)
    cluster = bootstrap_token_ce_band(arm, base, n_boot=400, seed=0, block_len=window)
    assert cluster["delta"] == pytest.approx(token["delta"])
    assert (cluster["hi"] - cluster["lo"]) > 2.0 * (token["hi"] - token["lo"])

    # One block: every replicate is the full-sample ratio (fp64), the point is the fp32 mean delta.
    one_block = bootstrap_token_ce_band(arm, base, n_boot=50, seed=0, block_len=arm.numel())
    assert one_block["lo"] == pytest.approx(one_block["delta"], abs=1e-6)
    assert one_block["hi"] == pytest.approx(one_block["delta"], abs=1e-6)
    with pytest.raises(ValueError):
        bootstrap_token_ce_band(arm, base, block_len=0)


def test_forest_withheld_without_window_and_captioned_by_seed_count(tmp_path: Path) -> None:
    from tests.test_figures_tail import _write_forest_cell

    sweep = tmp_path / "component_ablation_forest"
    baseline = torch.rand(32, dtype=torch.float32)
    cell, _ = _write_forest_cell(sweep, "baseline", baseline)
    _write_forest_cell(sweep, "head_mixer_off", baseline + 0.1)
    assert ablation_forest_kwargs(sweep, "baseline")["n_seeds"] == 1
    (cell / "config.json").unlink()
    assert ablation_forest_kwargs(sweep, "baseline") is None

    rows = [{"label": "a", "delta": 0.1, "lo": 0.0, "hi": 0.2}]
    single, multi = figs.plot_ablation_forest(rows), figs.plot_ablation_forest(rows, n_seeds=3)
    try:
        assert "single-seed" in single.axes[0].get_title()
        assert "3 seeds" in multi.axes[0].get_title()
    finally:
        plt.close(single)
        plt.close(multi)


# --------------------------------------------------------------------------- #2 decode temperature


def test_decode_tau_eff_is_the_realized_temperature() -> None:
    cfg = _cfg(decode_mode="diagonal_chunked", decode_tau=0.5)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.decode_log_scale.fill_(-1.0)
    assert _decode_tau_eff(model, cfg) == pytest.approx(0.5 * math.e)
    linear = _cfg(decode_mode="linear")
    assert _decode_tau_eff(VFEModel(linear), linear) is None


def test_finalize_run_persists_decode_tau_eff(tmp_path: Path, monkeypatch) -> None:
    _no_git(monkeypatch)
    for name in ("_write_provenance", "_write_research_artifacts", "_run_figures_isolated"):
        monkeypatch.setattr(run_artifacts, name, lambda *a, **k: None)
    cfg = _cfg(decode_mode="diagonal_chunked", decode_tau=0.25, generate_embedding_figures=False)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.decode_log_scale.fill_(0.5)
    art = RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    finalize_run(model, art, cfg, test_loader=None)
    expected = 0.25 * math.exp(-0.5)
    for name in ("summary.json", "test_results.json"):
        saved = json.loads((tmp_path / "r" / name).read_text(encoding="utf-8"))
        assert saved["decode_tau_eff"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #23 resume seed drift


@pytest.mark.parametrize(("restore_rng", "phrase"),
                         [(True, "'seed' is NOT applied"), (False, "only partly applied")])
def test_resume_seed_drift_warning_states_what_happens(tmp_path: Path, monkeypatch,
                                                       restore_rng: bool, phrase: str) -> None:
    _no_git(monkeypatch)
    cfg = _cfg(seed=6)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    path = RunArtifacts(tmp_path / "r", cfg, model).save_checkpoint(2, model, optimizer, cfg)
    with pytest.warns(UserWarning, match=r"config drift") as record:
        load_checkpoint(path, model, optimizer, cfg=_cfg(seed=23), restore_rng=restore_rng)
    message = " ".join(str(w.message) for w in record if "config drift" in str(w.message))
    assert phrase in message
    assert "use the CURRENT values" not in message                 # seed is the only drift


# --------------------------------------------------------------------------- Low items


def test_mixed_generation_process_is_not_reported_stable(tmp_path: Path, monkeypatch) -> None:
    _no_git(monkeypatch)
    cfg = _cfg()
    model = VFEModel(cfg)
    stable = RunArtifacts(tmp_path / "stable", cfg, model, save_model_artifacts=False)
    _write_provenance(stable, cfg, model, logging.getLogger("test"))
    stable_record = json.loads((tmp_path / "stable" / "provenance.json").read_text(encoding="utf-8"))
    assert stable_record["source_identity_status"] == "stable"

    monkeypatch.setattr(run_artifacts, "_PROCESS_PACKAGE_CODE_IDENTITY", "a" * 64)
    mixed = RunArtifacts(tmp_path / "mixed", cfg, model, save_model_artifacts=False)
    _write_provenance(mixed, cfg, model, logging.getLogger("test"))
    record = json.loads((tmp_path / "mixed" / "provenance.json").read_text(encoding="utf-8"))
    assert record["source_identity_status"] == "mixed_generation"
    assert record["process_code_identity_sha256"] == "a" * 64
    assert record["process_code_identity_match"] is False


def _span(patch) -> tuple:
    if hasattr(patch, "get_height"):
        return patch.get_y(), patch.get_y() + patch.get_height()
    ys = np.asarray(patch.get_xy())[:, 1]
    return float(ys.min()), float(ys.max())


def test_noise_band_is_a_t_prediction_interval_with_data_order_caption() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    agg = {"values": [20.1, 20.4, 19.8], "seeds": [6, 23, 54], "n": 3}
    agg["mean"] = float(np.mean(agg["values"]))
    agg["sd"] = float(np.std(agg["values"], ddof=1))
    fig = figs.plot_ppl_noise_band(agg, data_order_status="shared_explicit")
    try:
        ax = fig.axes[0]
        lo, hi = _span(ax.patches[0])
        half = scipy_stats.t.ppf(0.975, 2) * agg["sd"] * math.sqrt(1.0 + 1.0 / 3.0)
        assert lo == pytest.approx(agg["mean"] - half)
        assert hi == pytest.approx(agg["mean"] + half)
        assert "lower bound" in ax.get_title()
    finally:
        plt.close(fig)


def test_finalize_run_rejects_unreachable_best_metadata(tmp_path: Path, monkeypatch) -> None:
    _no_git(monkeypatch)
    cfg = _cfg(generate_embedding_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.best_val_ppl, art.best_step = 5.0, 2
    with pytest.raises(RuntimeError, match="no reachable weights"):
        finalize_run(model, art, cfg, test_loader=None)
    assert not (tmp_path / "r" / "summary.json").exists()
