r"""Cross-lane residuals of the 2026-10-16 audit fixes (the edits a lane needed in a file it did not own).

Each lane fixed its findings in the files it owned and filed the matching change in another lane's file as a
residual request; this module pins those changes after the merge. Grouped by finding:

* F37, F40, F50: ``build_optimizer`` forwards ``m_phi_group_differential_mode``, ``m_phi_group_log_fallback`` and
  ``m_phi_fisher_completion`` to the manifold optimizer, so the config values reach training.
* F14, F126, F95, F3: the metrics row carries the pullback-group halving counts, the LayerNorm-cap and GEMM-fallback
  run totals, and the rejected-row count is a whole-run total across resumes.
* F5, F3, F2: the block norm pushes the covariance only under a LayerNorm (a Mahalanobis norm ignores either
  pushforward value), the validator admits ``'pushforward_moment'`` wherever it admits ``'pushforward'``, and the
  gauge-covariant switch warns when it keeps the capped pushforward on a dense family.
* F7: the ``predictive_covariance_precision`` config field reaches every layer's operators.
* F26: the ``relational_tile_mib`` config field installs the scorer and routed-spread tile budgets; a raised spread
  budget still checkpoints a loop above the checkpoint floor.
* F9: a ConfigNotice when the certified factorization acts as ``gram_certified`` or bypasses the diagonal route.
* F10, F114, F45: the validation schema lists the new relational columns; ``numerical_health`` reads the pre-clamp
  non-finite map.
* F64: the inference bank and the logit lens decode under ``decode_covariance='encoded'``.
* F92: the E-step trace and its figure carry no free-energy series.
* F61, F59: the decision rule reads a failed seed pair as undetermined and the homogeneity check compares the
  training-window protocol; provenance records it.
* F96, F110, F122: the launcher's run-directory helpers, its zero-gradient banner line and its launch-time test split.

Tiny CPU models only (K <= 8, at most two layers, vocab <= 12).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import config_values, test_config, tiny_config
from vfe3.config import ConfigNotice, VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _quiet_and_single_threaded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _train(cfg: VFE3Config, run_dir: Path, *, n_steps: int, log_interval: int, resume_from: Any = None):
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    tokens = (torch.arange(61) * 5 + 2) % cfg.vocab_size
    loader = DataLoader(TokenWindows(tokens, cfg.max_seq_len), batch_size=cfg.batch_size, shuffle=False,
                        drop_last=True)
    torch.manual_seed(11)
    T.train(model, loader, cfg, n_steps=n_steps, grad_clip=cfg.grad_clip, log_interval=log_interval,
            eval_interval=0, device=CPU, logger=logging.getLogger("audit-integration"), artifacts=artifacts,
            generate_samples=False, resume_from=resume_from)
    return model, artifacts


def _small(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=7, embed_dim=4, n_heads=2, batch_size=2, max_seq_len=4, eval_stride=2, n_layers=1,
        n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0, checkpoint_interval=0,
        generate_embedding_figures=False, pos_phi="learned", gauge_group="block_glk", decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


# ------------------------------------------------------------------------------------------- F37, F40, F50

def test_build_optimizer_forwards_the_phi_group_settings() -> None:
    from vfe3.train import build_optimizer

    cfg = _small(m_phi_update_mode="pullback_group", m_phi_group_differential_mode="spectral")
    assert build_optimizer(VFEModel(cfg), cfg)._phi_group_differential_mode == "spectral"
    cfg = _small(m_phi_update_mode="pullback_group")
    assert build_optimizer(VFEModel(cfg), cfg)._phi_group_differential_mode == "series"

    cfg = _small(gauge_tying="per_head", m_phi_update_mode="pullback_group", m_phi_group_retraction="group_log",
                 m_phi_group_log_fallback="eigenbasis")
    assert build_optimizer(VFEModel(cfg), cfg)._phi_group_log_fallback == "eigenbasis"

    cfg = _small(pos_phi="none", m_phi_update_mode="fisher_group", m_phi_fisher_completion="stabilizer",
                 transport_chart_max_norm=6.0)
    assert build_optimizer(VFEModel(cfg), cfg)._phi_fisher_completion == "stabilizer"


def test_pure_path_report_records_the_fisher_completion() -> None:
    cfg = _small(pos_phi="none", m_phi_update_mode="fisher_group", m_phi_fisher_completion="stabilizer",
                 transport_chart_max_norm=6.0)
    report = ra._pure_path_report(cfg, [])
    found = [section.get("m_phi_fisher_completion") for section in report.values() if isinstance(section, dict)]
    assert "stabilizer" in found


# ------------------------------------------------------------------------------------- F14, F126, F95, F3

def test_metrics_rows_carry_the_new_counters_and_halving_columns(tmp_path: Path) -> None:
    cfg = _small(m_phi_update_mode="pullback_group", max_steps=4)
    _, artifacts = _train(cfg, tmp_path / "run", n_steps=4, log_interval=2)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert rows
    for row in rows:
        for column in ("layernorm_trace_cap_elements", "gemm_fallback_slices"):
            assert int(float(row[column])) == 0, (column, row[column])
        for column in ("phi_group_backtracks_mean", "phi_group_backtracks_max"):
            assert math.isfinite(float(row[column])), (column, row[column])
    for key in ("phi_rejected_rows", "gemm_fallback_slices"):
        assert key in artifacts.realized_updates


def test_gemm_fallback_counter_is_reset_and_counted_from_train_start(tmp_path: Path) -> None:
    import vfe3.relational_scoring as scoring

    scoring._GEMM_FALLBACK_SLICES = 1000                                   # left over from an earlier cell
    cfg = _small(attention_mode="projected_predictive", predictive_covariance="moment_matched",
                 decode_mode="diagonal_chunked", pair_kl_gemm_tol=1e-3, max_steps=2)
    assert "pair_kl_gemm_tol" not in cfg.inactive_fields()
    _, artifacts = _train(cfg, tmp_path / "run", n_steps=2, log_interval=1)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert int(float(rows[-1]["gemm_fallback_slices"])) == scoring.gemm_fallback_slices() < 1000
    assert artifacts.realized_updates["gemm_fallback_slices"] == scoring.gemm_fallback_slices()


def test_phi_rejected_rows_is_a_whole_run_total() -> None:
    base = {key: 0 for key in T.RUN_TOTAL_COUNTER_KEYS}
    base["phi_rejected_rows"] = 2
    totals = T._run_total_counters(base, 0, phi_rejected_rows=3)
    assert totals["phi_rejected_rows"] == 5
    assert "phi_rejected_rows" in T.RUN_TOTAL_COUNTER_KEYS and "gemm_fallback_slices" in T.RUN_TOTAL_COUNTER_KEYS
    restored = T._resumed_run_totals({**{key: 0 for key in T.RUN_TOTAL_COUNTER_KEYS}, "phi_rejected_rows": 4}, 6,
                                     logging.getLogger("audit-integration"))
    assert restored["phi_rejected_rows"] == 4


def test_resumed_pullback_run_carries_its_rejected_row_total(tmp_path: Path) -> None:
    cfg = _small(m_phi_update_mode="pullback_group", checkpoint_interval=2, max_steps=4)
    _train(cfg, tmp_path / "straight", n_steps=4, log_interval=0)
    state = torch.load(tmp_path / "straight" / "checkpoints" / "step_2.pt", map_location="cpu", weights_only=False)
    assert state["numerics_counters"]["phi_rejected_rows"] >= 0
    state["numerics_counters"]["phi_rejected_rows"] = 9                  # as if the first segment rejected 9 rows
    patched = tmp_path / "patched.pt"
    torch.save(state, patched)
    model, artifacts = _train(cfg, tmp_path / "resumed", n_steps=4, log_interval=0, resume_from=patched)
    assert artifacts.realized_updates["phi_rejected_rows"] >= 9


# ------------------------------------------------------------------------------------------ F5, F3, F2

def _block_cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(embed_dim=4, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=1,
                                  attention_mode="projected_predictive", predictive_covariance="moment_matched",
                                  content_link_chart="exponential", norm_type_block="layernorm",
                                  layernorm_affine=True, use_block_mlp=True, block_mlp_mode="coordinate",
                                  block_mlp_expansion=2)
    values.update(over)
    return tiny_config(**values)


def _loss_and_grads(cfg: VFE3Config):
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    tokens = torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(3))
    targets = torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    return loss.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("value", ["pushforward", "pushforward_moment"])
def test_a_mahalanobis_block_norm_ignores_the_pushforward(value: str) -> None:
    pushed = _block_cfg(norm_type_block="mahalanobis", layernorm_affine=False, block_norm_covariance=value)
    assert "block_norm_covariance" in pushed.inactive_fields()
    twin = _block_cfg(norm_type_block="mahalanobis", layernorm_affine=False, block_norm_covariance="passthrough")
    loss, grads = _loss_and_grads(pushed)
    loss_twin, grads_twin = _loss_and_grads(twin)
    assert torch.equal(loss, loss_twin)
    assert grads.keys() == grads_twin.keys()
    for name in grads:
        assert torch.equal(grads[name], grads_twin[name]), name


def test_pushforward_moment_is_admitted_where_pushforward_is() -> None:
    assert _block_cfg(block_norm_covariance="pushforward_moment").block_norm_covariance == "pushforward_moment"
    full = _block_cfg(family="gaussian_full", decode_mode="full_chunked", block_coordinates="reference",
                      block_norm_covariance="pushforward_moment")
    assert "block_norm_covariance" not in full.inactive_fields()
    with pytest.raises(ValueError, match="block_norm_covariance='pushforward_moment' needs"):
        _block_cfg(block_coordinates="reference", block_norm_covariance="pushforward_moment")   # gaussian_diagonal
    with pytest.raises(ValueError, match="block_norm_covariance='pushforward' needs"):
        _block_cfg(block_coordinates="reference", block_norm_covariance="pushforward")
    with pytest.raises(ValueError, match="block_norm_covariance"):
        _block_cfg(block_norm_covariance="moment")
    delta = _block_cfg(family="gaussian_full", decode_mode="full_chunked", block_mlp_covariance="delta_full",
                       block_norm_covariance="pushforward_moment")
    assert "block_norm_covariance" in delta.inactive_fields()
    # The norm pushforward reads the covariance floor, so it is live without a delta_full MLP.
    pushed = _block_cfg(block_norm_covariance="pushforward_moment")
    assert "block_mlp_covariance_floor" not in pushed.inactive_fields()
    assert "block_mlp_covariance_floor" in _block_cfg().inactive_fields()


def test_the_gauge_covariant_switch_warns_when_it_keeps_the_capped_pushforward() -> None:
    from vfe3.gauge_covariant import fully_gauge_covariant_overrides

    values = config_values(norm_type_block="layernorm", layernorm_affine=True, family="gaussian_full",
                           decode_mode="full_chunked", attention_mode="projected_predictive")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fully_gauge_covariant_overrides({**values, "block_norm_covariance": "pushforward"})
    assert any("pushforward_moment" in str(w.message) for w in caught)
    for value in ("passthrough", "pushforward_moment"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fully_gauge_covariant_overrides({**values, "block_norm_covariance": value})
        assert not any("fully_gauge_covariant" in str(w.message) for w in caught), value


# ------------------------------------------------------------------------------------------------- F7

def test_predictive_covariance_precision_reaches_every_layer() -> None:
    assert VFE3Config.__dataclass_fields__["predictive_covariance_precision"].default in ("fp32", "fp64")
    assert config_values()["predictive_covariance_precision"] == "fp32"
    cfg = tiny_config(n_layers=2, predictive_covariance_precision="fp64")
    model = VFEModel(cfg)
    assert [ops.predictive_covariance_precision for ops in model.relational_operators_by_layer] == ["fp64", "fp64"]
    with pytest.raises(ValueError, match="predictive_covariance_precision"):
        tiny_config(predictive_covariance_precision="fp16")
    assert "predictive_covariance_precision" not in cfg.inactive_fields()
    assert "predictive_covariance_precision" in tiny_config(
        predictive_covariance_precision="fp64", predictive_covariance="passthrough").inactive_fields()
    assert "predictive_covariance_precision" in tiny_config(
        predictive_covariance_precision="fp64", predictive_readout="minimal_agent",
        attention_residual=False).inactive_fields()


# ------------------------------------------------------------------------------------------------ F26

def test_relational_tile_mib_installs_both_tile_budgets() -> None:
    import vfe3.model.relational_operators as ro
    import vfe3.relational_scoring as scoring

    default = VFE3Config.__dataclass_fields__["relational_tile_mib"].default
    assert isinstance(default, int) and default >= 1 and config_values()["relational_tile_mib"] == 32
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="relational_tile_mib"):
            tiny_config(relational_tile_mib=bad)
    previous = (scoring.projection_tile_bytes(), ro.spread_tile_bytes())
    try:
        VFEModel(tiny_config(relational_tile_mib=64))
        assert scoring.projection_tile_bytes() == ro.spread_tile_bytes() == 64 * 1024 ** 2
        VFEModel(tiny_config())
        assert scoring.projection_tile_bytes() == ro.spread_tile_bytes() == 32 * 1024 ** 2
    finally:
        scoring.set_projection_tile_bytes(previous[0])
        ro.set_spread_tile_bytes(previous[1])


def test_spread_budget_setter_and_checkpoint_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.model.relational_operators as ro

    for bad in (0, True, 2.0):
        with pytest.raises(ValueError):
            ro.set_spread_tile_bytes(bad)
    g = torch.Generator().manual_seed(3)
    H, n, d = 2, 6, 3
    logits = torch.randn(1, H, n, n, generator=g, dtype=torch.float64)
    logits = logits.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    b = torch.softmax(logits, dim=-1).requires_grad_()
    values = torch.randn(1, n, H, d, generator=g, dtype=torch.float64).requires_grad_()
    mean = torch.einsum("...hij,...jhs->...ihs", b, values)
    reference = ro._centered_spread(b, values, mean)                      # one tile, below the floor
    calls: List[int] = []
    real = ro.checkpoint

    def spy(*args: Any, **kwargs: Any):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ro, "checkpoint", spy)
    previous = ro.set_spread_tile_bytes(1 << 30)
    try:
        assert ro._centered_spread(b, values, mean).shape == reference.shape and calls == []
        monkeypatch.setattr(ro, "_SPREAD_CHECKPOINT_BYTES", 64)           # every tiny loop exceeds the floor
        out = ro._centered_spread(b, values, mean)
        assert calls == [1]
        assert torch.equal(out, reference)
    finally:
        ro.set_spread_tile_bytes(previous)


# ------------------------------------------------------------------------------------------------- F9

def test_comparison_invariant_route_notes_what_the_certified_factorization_does() -> None:
    def notices(**over: Any) -> List[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tiny_config(rope_mode="comparison_invariant", relational_routing="gaussian_kl", **over)
        return [str(w.message) for w in caught if issubclass(w.category, ConfigNotice)]

    assert not any("comparison_invariant" in message for message in notices())
    assert any("acts as 'gram_certified'" in message
               for message in notices(projected_factorization="cholesky_qr2_certified"))
    # pair_kl_gemm_tol keeps the diagonal scorer since audit 2026-10-17 item 11; a certified factorization bypasses it.
    assert not any("closed-form diagonal comparison scorer" in message for message in notices(pair_kl_gemm_tol=1e-3))
    assert any("closed-form diagonal comparison scorer" in message
               for message in notices(projected_factorization="gram_certified"))


# ---------------------------------------------------------------------------------------- F10, F114, F45

def test_validation_schema_lists_the_new_relational_columns() -> None:
    for key in ("val_relational_score_nonfinite_frac", "val_relational_score_nonfinite_frac_layer_max",
                "val_relational_routing_free_energy", "val_relational_rank_one_residual_reference"):
        assert key in T._VAL_RELATIONAL_DIAG_KEYS, key


def test_numerical_health_counts_nonfinite_router_costs_before_the_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.gradients.relational as relational
    from vfe3.viz.extract import numerical_health

    torch.manual_seed(3)
    model = VFEModel(tiny_config(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=2,
                                 phi_scale=0.4)).eval()
    original = relational.relational_score
    calls: List[int] = []

    def score(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original(*args, **kwargs)                                     # (B, H, N, N)
        calls.append(1)
        if len(calls) % 2 == 0:                                             # layer 1, sequence 0
            out = out.clone()
            out[0, 0, 3, 1], out[0, 1, 4, 2] = float("nan"), float("inf")
        return out

    monkeypatch.setattr(relational, "relational_score", score)
    tokens = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
    health = numerical_health(model, tokens)
    assert health["numel_energy"] == 2 * 2 * 5 * 5
    assert health["nan_energy"] == pytest.approx(2.0 / health["numel_energy"], rel=1e-12)


# ------------------------------------------------------------------------------------------------ F64

def _encoded_model() -> VFEModel:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(n_layers=2, sigma_init=2.0, phi_scale=0.3, decode_mode="diagonal_chunked",
                                 untie_decode_bank=True, decode_covariance="encoded")).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.uniform_(-0.5, 1.0, generator=torch.Generator().manual_seed(1))
        model.prior_bank.decode_sigma_log_embed.uniform_(-0.5, 1.0, generator=torch.Generator().manual_seed(2))
    return model


def test_inference_bank_redecodes_under_the_encoded_covariance() -> None:
    from vfe3.viz.extract import _decode_inference_record, collect_inference_bank

    model = _encoded_model()
    tokens = torch.tensor([[1, 4, 2], [5, 3, 6]])
    records = collect_inference_bank(model, [(tokens, tokens)])
    logits = _decode_inference_record(model, records[0], CPU)
    with torch.no_grad():
        _, want = model.forward_beliefs(tokens, return_logits=True)
    torch.testing.assert_close(logits, want, rtol=0.0, atol=0.0)
    plain = VFEModel(tiny_config(decode_mode="diagonal_chunked"))
    assert collect_inference_bank(plain, [(tokens, tokens)])[0].get("encoded_sigma") is None


def test_logit_lens_readout_passes_the_stack_input_covariance() -> None:
    from vfe3.semantics import contextual as C

    model = _encoded_model()
    tokens = torch.tensor([[1, 4, 2], [5, 3, 6]])
    out = C._forward_layers(model, tokens, return_logits=True)
    layers, frame = out["layers"], out["canonical_frame"]
    got = C._readout(model, layers[-1].mu, layers[-1].sigma, frame, encoded_sigma=layers[0].sigma)
    torch.testing.assert_close(got, out["logits"], rtol=0.0, atol=0.0)


# ------------------------------------------------------------------------------------------------ F92

def test_belief_trace_carries_no_free_energy_series() -> None:
    from vfe3.viz.extract import e_step_belief_trace

    model = VFEModel(tiny_config())
    trace = e_step_belief_trace(model, torch.tensor([[1, 2, 3]]))
    assert "free_energy" not in trace and trace["free_energy_available"] is False


# ------------------------------------------------------------------------------------------ F61, F59

def _arm(label: str, **fields: Any):
    from vfe3.paper.runs import Arm
    return Arm(label=label, **fields)


def test_a_failed_seed_pair_leaves_the_clause_undetermined() -> None:
    from vfe3.paper.tables import _unverified_reasons

    assert _unverified_reasons(_arm("Evaluated")) == []
    reasons = _unverified_reasons(_arm("Evaluated", failed=[{"run": "x", "seed_pair": [6, 3]}]))
    assert len(reasons) == 1 and "failed" in reasons[0]


def test_manifest_rows_carry_failed_runs_and_window_protocol_notes() -> None:
    from vfe3.paper.runs import resolve_manifest
    from vfe3.paper.tables import _window_protocol_notes

    manifest = resolve_manifest({
        "arms": {"Evaluated": []},
        "paper_arms": {"rows": {"Evaluated": {
            "failed": [{"run": "r1", "reason": "test_ce is NaN"}],
            "window_protocol_mismatch": [{"run": "r0", "reason": "trained with fixed windows"}]}}},
    })
    arm = manifest.arms[0]
    assert arm.failed == [{"run": "r1", "reason": "test_ce is NaN"}]
    assert arm.window_protocol_mismatch == [{"run": "r0", "reason": "trained with fixed windows"}]
    notes = _window_protocol_notes(arm)
    assert len(notes) == 1 and "window" in notes[0]
    assert _window_protocol_notes(resolve_manifest({"arms": {"Evaluated": []}}).arms[0]) == []


def test_homogeneity_compares_the_known_training_window_protocols() -> None:
    from vfe3.paper.runs import RunRecord, check_homogeneity

    def record(name: str, mode: Any) -> Any:
        provenance = {} if mode is None else {"train_window_mode": mode}
        return RunRecord(run_dir=Path(name), config_raw={"embed_dim": 4}, provenance=provenance, dataset="d",
                         seed=ord(name), data_seed=3)

    mixed = check_homogeneity([record("a", "fixed"), record("b", "per_epoch_random")])
    assert not mixed["homogeneous"]
    assert any(v["field"] == "train_window_mode" for v in mixed["violations"])
    assert check_homogeneity([record("a", "fixed"), record("b", "fixed")])["homogeneous"]
    assert check_homogeneity([record("a", "fixed"), record("b", None)])["homogeneous"]   # unknown blocks nothing


def test_provenance_records_the_training_window_mode(tmp_path: Path) -> None:
    cfg = _small(max_steps=2)
    model, artifacts = _train(cfg, tmp_path / "run", n_steps=2, log_interval=0)
    tokens = (torch.arange(61) * 5 + 2) % cfg.vocab_size
    dataset = TokenWindows(tokens, cfg.max_seq_len)
    dataset.use_per_epoch_offsets()
    ra._write_provenance(artifacts, cfg, model, logging.getLogger("audit-integration"),
                         train_loader=DataLoader(dataset, batch_size=2))
    provenance = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["train_window_mode"] == "per_epoch_random"


# ------------------------------------------------------------------------------------- F96, F110, F122

def test_launcher_uses_the_shared_run_directory_helpers(tmp_path: Path) -> None:
    import train_vfe3
    import vfe3.experiment as experiment

    assert train_vfe3._reserve_directory is experiment.reserve_directory
    assert not hasattr(train_vfe3, "_rename_run_by_ppl") and not hasattr(train_vfe3, "_rename_run_by_validation_ppl")
    run = experiment.reserve_directory(tmp_path, "20261017-000000_arm")
    renamed = train_vfe3._rename_run_by_test_ppl(str(run), "arm", 3.25, logging.getLogger("audit-integration"))
    assert isinstance(renamed, str) and Path(renamed) == tmp_path / "3.25_arm"


def test_launcher_banner_names_zero_gradient_parameters() -> None:
    import train_vfe3

    cfg = test_config(block_mlp_mode="gauge_gate", use_block_mlp=True, family="gaussian_full")
    model = VFEModel(cfg)
    banner = train_vfe3._banner(model, cfg, "synthetic", "cpu", 1)
    assert "zero gradient at the init probe (counted live)" in banner
    assert "zero gradient at the init probe" not in train_vfe3._banner(VFEModel(test_config()), test_config(),
                                                                       "synthetic", "cpu", 1)


def test_launcher_opens_the_test_split_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import train_vfe3
    import vfe3.data.datasets as datasets_mod

    cache = tmp_path / "cache"
    cache.mkdir()
    generator = torch.Generator().manual_seed(0)
    for split, n in (("train", 60), ("validation", 24)):
        torch.save(torch.randint(0, 7, (n,), generator=generator),
                   datasets_mod.cache_path("wikitext-103", split, cache_dir=cache))
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: cache)
    monkeypatch.setattr(datasets_mod, "tokens_per_char", lambda *args, **kwargs: None)

    def reached_training(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("training started without a test split")

    values = {key: getattr(_small(), key) for key in train_vfe3.config if hasattr(VFE3Config, key)}
    values.update(vocab_size=7, embed_dim=4, n_heads=2, batch_size=2, max_seq_len=4, max_steps=2, warmup_steps=0,
                  log_interval=0, eval_interval=0, checkpoint_interval=0, generate_embedding_figures=False)
    monkeypatch.setattr(train_vfe3, "config", values)
    monkeypatch.setattr(train_vfe3, "DATASET", "wikitext-103")
    monkeypatch.setattr(train_vfe3, "DEVICE", "cpu")
    monkeypatch.setattr(train_vfe3, "MAX_TOKENS", None)
    monkeypatch.setattr(train_vfe3, "fully_gauge_covariant", False)
    monkeypatch.setattr(train_vfe3, "train", reached_training)
    runs = tmp_path / "runs"
    with pytest.raises(FileNotFoundError, match="test"):
        train_vfe3._run_once(0, logging.getLogger("audit-integration"), run_root=str(runs))
    assert not runs.exists() or not any(runs.iterdir())
