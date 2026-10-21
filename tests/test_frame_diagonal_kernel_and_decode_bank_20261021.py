r"""``gaussian_frame_diagonal`` on the diagonal comparison route and its decode bank (audit 2026-10-20 items 45, 59).

Item 45: ``summary.json`` records which diagonal-moments kernel ran; the family reaches the diagonal route, now through
the identity-frame GEMM of item 46, so ``relational_diag_moments_kernel`` is inactive under it. Item 59: a tied decode bank pairs the encoder's local-frame mean with
the reference variance, so the validator refuses it for the non-linear decoder and the gauge switch unties it.
Tiny CPU models only.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.profiles import config_values, live_config
from vfe3 import relational_scoring as rs
from vfe3.config import VFE3Config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.model.model import VFEModel

FAMILY = "gaussian_frame_diagonal"

_FRAME_DIAGONAL_ROUTE = dict(family=FAMILY, predictive_readout="kernel", block_mlp_mode="coordinate",
                             block_mlp_covariance="passthrough", decode_mode="diagonal_chunked",
                             norm_type_block="layernorm", untie_decode_bank=True,
                             attention_mode="projected_predictive", relational_routing="gaussian_kl",
                             rope_mode="comparison_invariant", rope_fraction=1.0, projected_factorization="gram")


def _route(**over: Any) -> VFE3Config:
    return live_config(**{**_FRAME_DIAGONAL_ROUTE, **over})


# ------------------------------------------------------------------------------------- item 45

def test_kernel_field_is_inactive_under_the_frame_diagonal_family() -> None:
    # Audit 2026-10-20 item 46 supersedes item 45 here: the frame-diagonal family's moments read the identity frame
    # through one torch GEMM, (S o S) c, on every device, so the fused kernels never run under it.
    assert "relational_diag_moments_kernel" in _route().inactive_fields()
    assert "relational_diag_moments_kernel" not in _route(family="gaussian_diagonal").inactive_fields()
    assert "relational_diag_moments_kernel" in _route(projected_factorization="gram_certified").inactive_fields()


def test_frame_diagonal_model_reaches_the_diagonal_moments_route() -> None:
    torch.manual_seed(0)
    model = VFEModel(_route(relational_diag_moments_kernel="torch"))
    rs.reset_diagonal_moments_triton_calls()
    model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))
    assert rs.diagonal_moments_torch_calls() > 0
    assert rs.realized_diagonal_moments_kernel() == "torch"


@pytest.mark.parametrize("triton, torch_calls, expected", [
    (0, 0, None), (3, 0, "triton"), (0, 2, "torch"), (3, 2, "mixed")])
def test_realized_kernel_reads_both_counters(monkeypatch: pytest.MonkeyPatch, triton: int, torch_calls: int,
                                             expected: Any) -> None:
    monkeypatch.setattr(rs, "_DIAG_MOMENTS_TRITON_CALLS", triton)
    monkeypatch.setattr(rs, "_DIAG_MOMENTS_TORCH_CALLS", torch_calls)
    assert rs.realized_diagonal_moments_kernel() == expected
    rs.reset_diagonal_moments_triton_calls()
    assert rs.realized_diagonal_moments_kernel() is None


def test_summary_records_the_realized_kernel_from_the_run_start(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3 import run_artifacts as ra

    monkeypatch.setattr(ra, "_git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    for name in ("_write_provenance", "_write_research_artifacts", "_run_figures_isolated"):
        monkeypatch.setattr(ra, name, lambda *a, **k: None)
    cfg = _route(generate_embedding_figures=False)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    monkeypatch.setattr(rs, "_DIAG_MOMENTS_TRITON_CALLS", 5)                      # a previous run in this process
    art = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    assert rs.diagonal_moments_triton_calls() == 0                                 # reset at the run start
    model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))
    ra.finalize_run(model, art, cfg, test_loader=None)
    summary = json.loads((tmp_path / "r" / "summary.json").read_text(encoding="utf-8"))
    assert summary["diag_moments_kernel"] == "torch"
    assert summary["diag_moments_kernel_calls"]["triton"] == 0
    assert summary["diag_moments_kernel_calls"]["torch"] > 0


# ------------------------------------------------------------------------------------- item 59

def test_tied_decode_bank_is_refused_for_the_kl_decoder() -> None:
    with pytest.raises(ValueError, match="untie_decode_bank=False with decode_mode='diagonal_chunked'"):
        _route(untie_decode_bank=False)
    assert _route(untie_decode_bank=True).untie_decode_bank is True
    # The linear decoder reads no bank, so a tied bank stays admissible there.
    assert _route(decode_mode="linear", untie_decode_bank=False).decode_mode == "linear"


def test_gauge_switch_unties_the_bank_and_certifies_only_the_untied_route() -> None:
    raw = config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                        predictive_covariance="moment_matched", family=FAMILY, decode_mode="diagonal_chunked",
                        untie_decode_bank=False, vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                        block_mlp_expansion=2)
    overrides = fully_gauge_covariant_overrides(raw)
    assert overrides.get("untie_decode_bank") is True
    cfg = VFE3Config(**{**raw, **overrides})
    assert cfg.untie_decode_bank and gauge_covariance_failures(cfg) == []
    with pytest.raises(ValueError, match="untie_decode_bank=False"):
        VFE3Config(**{**raw, **overrides, "untie_decode_bank": False})
    linear = {**raw, "decode_mode": "linear"}
    assert "untie_decode_bank" not in fully_gauge_covariant_overrides(linear)
