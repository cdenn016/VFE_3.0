"""Remediation checks for the October 9, 2026 audit: config pins and guards, the projected_gradient
clamp reference, and the training/run-artifact publications."""

from tests.profiles import make_relational_operators
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config
from tests.test_audit_20261008_config_train import _cfg, _rows, _train
from vfe3.config import config_from_serialized
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.full_link import build_full_link_context
from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import RelationalOperators


# --------------------------------------------------------------------------- rope_space pin

def test_rope_record_without_rope_space_reloads_as_frame_space() -> None:
    payload = asdict(test_config(attention_mode="projected_predictive", pos_rotation="rope",
                                 rope_space="comparison", rope_on_value=False))
    payload.pop("rope_space")
    restored = config_from_serialized(payload, source="pre-2026-10-07 record")
    assert restored.rope_space == "frame"


# --------------------------------------------------------------------------- delta_full guard

def test_delta_full_block_mlp_refuses_a_block_norm_without_a_jacobian() -> None:
    # LayerNorm's Jacobian is composed into the pushforward (tests/test_delta_full_layernorm_20261009.py).
    full = dict(use_block_mlp=True, block_mlp_mode="coordinate", block_mlp_covariance="delta_full",
                family="gaussian_full", decode_mode="full_chunked")
    with pytest.raises(ValueError, match="norm_type_block='mahalanobis'"):
        test_config(**full, norm_type_block="mahalanobis")
    assert test_config(**full, norm_type_block="layernorm").block_mlp_covariance == "delta_full"
    assert test_config(**full, norm_type_block="none").block_mlp_covariance == "delta_full"


# --------------------------------------------------------------------------- projected_gradient clamp reference

def _router_fixture(kl_max_reference: str) -> tuple:
    torch.manual_seed(0)
    mu = torch.tensor([[[.2, -.4, .1, .7], [1.1, .5, -.3, .8],
                        [.6, 1.2, .9, -.2]]], dtype=torch.float64)
    sigma = torch.tensor([[[.8, 1.2, .9, 1.4], [1.1, .6, 1.5, .7],
                           [.7, 1.3, .8, 1.1]]], dtype=mu.dtype)
    U = torch.eye(4, dtype=mu.dtype).expand(1, 3, 4, 4).clone()
    frame = CanonicalFrameContext(U, U.clone(), block_diagonal=False)
    ops = make_relational_operators([2, 2], 11, readout_mode="predictive_content",
                              predictive_covariance="passthrough",
                              kl_max_reference=kl_max_reference).double()
    return mu, sigma, frame, ops


@pytest.mark.parametrize("reference", ["absolute", "row_min"])
def test_projected_gradient_honors_the_routing_clamp_reference(reference: str) -> None:
    mu, sigma, frame, ops = _router_fixture(reference)
    with torch.no_grad():
        score = relational_score(mu, sigma, mu, sigma, frame, ops, None)    # (1, H, N, N)
    kl_max = 0.5 * float(score.min())     # every row fully saturated under the absolute reference
    context = build_full_link_context(mu, sigma, mu + .2, sigma * 1.2, frame, ops,
                                      kl_max=kl_max, value=.4)
    weight = torch.arange(1, 4, dtype=mu.dtype)                        # non-constant readout of beta
    (context.beta * weight).sum().backward()
    grad = ops.raw_score_q.grad
    if reference == "absolute":
        assert grad is None or float(grad.abs().max()) == 0.0
        torch.testing.assert_close(context.beta, torch.full_like(context.beta, 1 / 3))
    else:
        # Every non-argmin gap exceeds the cap here, so each row routes by its cost order with the constant
        # weights softmax(-(0, kl_max, kl_max)/tau): not uniform, and (with the row minimum differentiated,
        # audit 2026-10-09 round 2) exactly flat in the maps. A cap above some gaps leaves the router live.
        assert not torch.allclose(context.beta, torch.full_like(context.beta, 1 / 3))
        assert grad is None or float(grad.abs().max()) == 0.0
        ops.zero_grad()
        context = build_full_link_context(mu, sigma, mu + .2, sigma * 1.2, frame, ops, kl_max=1.0, value=.4)
        (context.beta * weight).sum().backward()
        assert float(ops.raw_score_q.grad.abs().max()) > 0.0


def test_projected_gradient_marks_routing_kl_max_reference_live() -> None:
    cfg = test_config(attention_mode="projected_gradient")
    assert "routing_kl_max_reference" not in cfg.inactive_fields()


# --------------------------------------------------------------------------- training publications

@pytest.fixture
def _no_side_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def test_tiny_run_publishes_source_recoveries_and_warns_once_per_dead_router_layer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
        _no_side_outputs: None) -> None:
    real_diagnostics = VFEModel.diagnostics

    def with_router_cap(self: VFEModel, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = dict(real_diagnostics(self, *args, **kwargs))
        out.update(relational_score_klmax_frac_layer_max=0.9, relational_score_klmax_frac_argmax_layer=1,
                   relational_clamped_row_frac_head_max=0.9, relational_clamped_row_frac_argmax_layer=1,
                   relational_clamped_row_frac_argmax_head=0)
        return out

    monkeypatch.setattr(VFEModel, "diagnostics", with_router_cap)
    cfg = _cfg()
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, VFEModel(cfg), save_model_artifacts=False)
    with caplog.at_level(logging.WARNING, logger="dead-router-test"):
        _train(cfg, artifacts)
    assert artifacts.realized_updates["source_recovery_elements"] == 0
    rows = _rows(run_dir / "metrics.csv")
    assert len(rows) == cfg.max_steps
    assert all(row["source_recovery_elements"] == "0" for row in rows)
    assert all(float(row["relational_score_klmax_frac_layer_max"]) == 0.9 for row in rows)
    warnings_seen = [r.getMessage() for r in caplog.records if "router layer 1 head 0" in r.getMessage()]
    assert len(warnings_seen) == 1


def test_dead_router_warning_reads_its_keys_defensively(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("dead-router-unit")
    warned: set = set()
    with caplog.at_level(logging.WARNING, logger="dead-router-unit"):
        T._warn_dead_router({}, warned, logger)                                        # keys absent
        T._warn_dead_router({"relational_clamped_row_frac_head_max": 0.5}, warned, logger)    # not above
        T._warn_dead_router({"relational_clamped_row_frac_head_max": float("nan")}, warned, logger)
        for layer in (0, 0, 2):
            T._warn_dead_router({"relational_clamped_row_frac_head_max": torch.tensor(0.75),
                                 "relational_clamped_row_frac_argmax_layer": layer,
                                 "relational_clamped_row_frac_argmax_head": 1}, warned, logger)
    assert warned == {("clamped", 0, 1), ("clamped", 2, 1)}
    assert [r.getMessage().split(":")[0].strip() for r in caplog.records] == [
        "router layer 0 head 1", "router layer 2 head 1"]


def test_evaluate_splits_ignored_targets_into_overlap_context_and_padding() -> None:
    cfg = _cfg()
    stream = (torch.arange(20) * 5 + 1) % cfg.vocab_size              # 19 transitions
    windows = TokenWindows(stream, cfg.max_seq_len, pad_final=True, stride=cfg.eval_stride)
    torch.manual_seed(0)
    m = T.evaluate(VFEModel(cfg), DataLoader(windows, batch_size=2, shuffle=False),
                   device=torch.device("cpu"))
    # 9 windows of 4: windows 1..8 mask a 2-target overlap prefix, the last pads 1 trailing target.
    assert (m["expected_targets"], m["scored_targets"], m["excluded_targets"]) == (36, 19, 17)
    assert (m["overlap_context_targets"], m["padding_targets"]) == (16, 1)
    assert m["scored_targets"] == int(windows.tokens.numel()) - 1


def test_finalize_summary_reports_the_target_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       _no_side_outputs: None) -> None:
    import tests.test_baseline_20260928 as tb_test
    monkeypatch.setattr(tb_test.tb, "_select_loader", tb_test._fake_select_loader)
    monkeypatch.setattr(tb_test.tb, "_tokens_per_char", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb_test.tb, "_tokenizer_tag", lambda dataset: None)
    run_dir = tb_test.tb.run_baseline(
        tb_test._bcfg(max_steps=2), dataset="tiny-synthetic", out_root=tmp_path, seed=3, data_seed=None,
        label="tiny", device="cpu", generate_samples=False, logger=logging.getLogger("partition-test"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["test_scored_targets"] == summary["test_stream_transitions"] == 22
    # 23 tokens in 7 windows of 6 at stride 3: 6 overlap prefixes of 3 and 2 padded final targets.
    assert (summary["test_overlap_context_targets"], summary["test_padding_targets"]) == (18, 2)
    assert summary["test_expected_targets"] == 42                                  # unchanged fields
    assert summary["test_excluded_targets"] == summary["excluded_targets"] == 20
    assert summary["source_recovery_elements"] == 0


# --------------------------------------------------------------------------- phi numerics precision

def test_phi_numerics_fp32_leg_runs_under_ieee_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3 import metrics
    seen: List[str] = []
    real = metrics.flatness_reference_statistics

    def spy(*args: Any, **kwargs: Any) -> Dict[str, float]:
        seen.append(torch.get_float32_matmul_precision())
        return real(*args, **kwargs)

    monkeypatch.setattr(metrics, "flatness_reference_statistics", spy)
    cfg = _cfg()
    model = VFEModel(cfg)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        record = ra.collect_phi_numerics(model, torch.tensor([[0, 1, 2, 3]]))
        assert torch.get_float32_matmul_precision() == "high"                 # process policy restored
    finally:
        torch.set_float32_matmul_precision(previous)
    assert seen == ["highest"]
    assert record["flatness_fp32_matmul_precision"] == "highest"
    assert record["process_fp32_matmul_precision"] == "high"
