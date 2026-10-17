r"""Audit 2026-10-16, lane C2 (post-hoc analysis): F65, F66, F68, F69, F120.

Synthetic strings and tiny models only (single-digit dimensions, CPU, one thread); nothing is trained.

* F65: :func:`vfe3.posthoc.calibration.text_audit` reports ``same_text`` only when every literal
  segment matched in its wildcard window and every ``<unk>`` wildcard matched characters.
* F66: M5 row 5 sets every stored covariance table to the identity, the untied decode bank's
  ``decode_sigma_log_embed`` included; ``decode=False`` keeps the earlier encode-side-only arm.
* F68: the reproduction check calls a replay matched arithmetic only when its amp dtype and its fp32
  matmul policy are the run's; an IEEE fp32 replay of a TF32 run gets the fp32-replay tolerance.
* F69: the M5 table marks a row scored on the aligned subset of the baseline's targets and prints the
  baseline CE on that subset; the forest plot names the resampling unit the intervals used.
* F120: under a comparison ``rope_mode`` M5 rows 6 and 10 score through the trained router (its
  comparison-space rotation and commutant projection included) on substituted reference moments, and
  the M4 decomposition rebuilds the rotated moments under ``rope_mode='comparison_invariant'``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch

from tests.test_posthoc_20260928 import DEVICE, _cfg, _model, _stream
from vfe3.posthoc import calibration as cal
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import report as rp
from vfe3.posthoc import streams as st


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# F65: text audit verdict
# ----------------------------------------------------------------------------------------------

def test_text_audit_rejects_a_segment_that_failed_its_wildcard_window() -> None:
    audit = cal.text_audit("The cat", " The <unk> cat\n")         # the <unk> word is absent from the BPE text
    assert audit["mismatched_segments"] == 1
    assert audit["matched_chars"] == audit["word_chars"] and audit["unmatched_bpe_chars"] == 0
    assert not audit["same_text"]
    assert audit["unmatched_unk_wildcards"] == 1


def test_text_audit_rejects_a_trailing_wildcard_with_no_bpe_tail() -> None:
    audit = cal.text_audit("The cat", " The cat <unk>\n")
    assert audit["mismatched_segments"] == 0 and audit["unmatched_bpe_chars"] == 0
    assert audit["unmatched_unk_wildcards"] == 1
    assert not audit["same_text"]


def test_text_audit_still_accepts_matched_wildcards() -> None:
    inner = cal.text_audit("The big cat", " The <unk> cat\n")
    trailing = cal.text_audit("The cat sat", " The cat <unk>\n")
    for audit in (inner, trailing):
        assert audit["same_text"] and audit["unmatched_unk_wildcards"] == 0
        assert audit["mismatched_segments"] == 0


# ----------------------------------------------------------------------------------------------
# F66: row 5 on an untied Gaussian decode bank
# ----------------------------------------------------------------------------------------------

def _untied_model(**over: Any) -> Any:
    cfg = _cfg(decode_mode="diagonal_chunked", untie_decode_bank=True, **over)
    model = _model(cfg)
    with torch.no_grad():
        model.prior_bank.decode_sigma_log_embed.normal_(0.0, 0.5)
    return model


def test_stored_covariances_identity_also_sets_the_untied_decode_bank_to_identity() -> None:
    model = _untied_model()
    pb = model.prior_bank
    assert pb.untie_decode_bank and pb._decode_sigma_log_table() is pb.decode_sigma_log_embed
    encode_before = pb.sigma_log_embed.detach().clone()
    decode_before = pb.decode_sigma_log_embed.detach().clone()
    with iv.stored_covariances_identity(model) as record:
        assert torch.count_nonzero(pb.sigma_log_embed) == 0
        assert torch.count_nonzero(pb._decode_sigma_log_table()) == 0
    assert record["tables"] == ["prior_bank.sigma_log_embed", "prior_bank.decode_sigma_log_embed"]
    assert record["decode_bank"] == "untied"
    assert torch.equal(pb.sigma_log_embed, encode_before) and torch.equal(pb.decode_sigma_log_embed, decode_before)


def test_stored_covariances_identity_encode_side_only_variant_keeps_the_decode_bank() -> None:
    model = _untied_model()
    pb = model.prior_bank
    dataset = st.eval_windows(_stream(61, 3), 8, 4)
    indices = np.arange(len(dataset))
    with iv.stored_covariances_identity(model, decode=False) as record:
        assert torch.count_nonzero(pb.sigma_log_embed) == 0
        assert torch.count_nonzero(pb.decode_sigma_log_embed) > 0
        encode_only = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert record["tables"] == ["prior_bank.sigma_log_embed"]
    with iv.stored_covariances_identity(model):
        both = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert np.abs(both.nll - encode_only.nll).max() > 1e-6


def test_stored_covariances_identity_covers_the_block_cholesky_table() -> None:
    import warnings

    from tests.profiles import tiny_config
    from vfe3.model.model import VFEModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = tiny_config(embed_dim=6, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=1, family="gaussian_block",
                          decode_mode="full_chunked", untie_decode_bank=True, block_coordinates="reference",
                          decode_coordinates="reference", norm_type_block="layernorm", layernorm_affine=True,
                          use_block_mlp=True, block_mlp_mode="coordinate", content_link_chart="exponential",
                          pos_phi="none", gauge_transport="off", prior_covariance_table="block_cholesky")
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    pb = model.prior_bank
    assert pb.sigma_log_embed is None
    with torch.no_grad():
        pb.sigma_chol_embed.normal_(0.0, 0.5)
        pb.decode_sigma_log_embed.normal_(0.0, 0.5)
    with iv.stored_covariances_identity(model) as record:
        blocks = pb._prior_covariance_blocks()                                     # (V, H, d, d)
        eye = torch.eye(blocks.shape[-1], dtype=blocks.dtype)
        torch.testing.assert_close(blocks, (1.0 + pb.eps) * eye.expand_as(blocks), rtol=0, atol=1e-7)
    assert record["tables"] == ["prior_bank.sigma_chol_embed", "prior_bank.decode_sigma_log_embed"]
    assert torch.count_nonzero(pb.sigma_chol_embed) > 0


# ----------------------------------------------------------------------------------------------
# F68: matched arithmetic includes the fp32 matmul policy
# ----------------------------------------------------------------------------------------------

def _replay(amp: Optional[str], matmul: Optional[str]) -> Any:
    return SimpleNamespace(amp_dtype=amp, cfg=SimpleNamespace(amp_dtype=amp, fp32_matmul_precision=matmul))


def test_an_ieee_fp32_replay_of_a_tf32_run_is_not_matched_arithmetic() -> None:
    run = SimpleNamespace(amp_dtype=None, fp32_matmul_precision="high")
    cuda = torch.device("cuda")
    assert not rp._replay_matches_run(_replay(None, "highest"), run, cuda)          # precision='fp32'
    assert rp._replay_matches_run(_replay(None, "high"), run, cuda)                 # precision='run'
    same = SimpleNamespace(amp_dtype=None, fp32_matmul_precision="highest")
    assert rp._replay_matches_run(_replay(None, "highest"), same, cuda)
    mixed = SimpleNamespace(amp_dtype="bf16", fp32_matmul_precision="highest")
    assert not rp._replay_matches_run(_replay(None, "highest"), mixed, cuda)
    assert rp._replay_matches_run(_replay("bf16", "highest"), mixed, cuda)
    assert not rp._replay_matches_run(_replay("bf16", "highest"), mixed, torch.device("cpu"))


def test_reproduction_check_uses_the_replay_tolerance_and_label_when_not_matched() -> None:
    reference = {"ce": 1.0, "scored_targets": None}
    result = {"ce": 1.002, "n_scored": 10}
    loose = rp._reproduction(result, reference, full=True, matched=False)
    assert loose["arithmetic"] == "fp32 replay" and loose["tolerance"] == rp.REPRODUCTION_TOL_FP32_REPLAY
    assert loose["status"] == "ok"


# ----------------------------------------------------------------------------------------------
# F69: aligned-subset rows and the bootstrap unit
# ----------------------------------------------------------------------------------------------

def _m5_rows() -> list:
    return [
        {"row": 1, "key": "baseline", "label": "None (final weights)", "status": "ok", "ce": 3.0,
         "delta_ce": 0.0, "lo": 0.0, "hi": 0.0, "ce_lo": 2.9, "ce_hi": 3.1, "n_targets": 100},
        {"row": 5, "key": "stored_covariances_identity", "label": "Stored covariances", "status": "ok",
         "ce": 3.5, "delta_ce": 0.4, "lo": 0.3, "hi": 0.5, "n_targets": 96, "n_targets_dropped": 4,
         "ce_base_aligned": 3.1},
        {"row": 7, "key": "token_frames_identity", "label": "Token frames", "status": "ok",
         "ce": 3.2, "delta_ce": 0.2, "lo": 0.1, "hi": 0.3, "n_targets": 100, "n_targets_dropped": 0},
    ]


def test_interventions_table_marks_rows_scored_on_an_aligned_subset() -> None:
    text = rp.interventions_table(_m5_rows(), n_units=8, unit="article", n_targets=100, subset=False)
    lines = {line.split(" |")[0].lstrip("| "): line for line in text.splitlines() if line.startswith("| ")}
    assert "†" in lines["Stored covariances"] and "†" not in lines["Token frames"]
    assert "96 of the baseline's 100 targets" in text and "4 dropped" in text
    assert "3.1" in text.split("†", 2)[-1]                                    # the aligned baseline CE


def test_interventions_plot_names_the_bootstrap_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.posthoc import figures as fg

    labels: list = []
    monkeypatch.setattr(fg, "_save", lambda fig, out_dir, name: labels.append(fig.axes[0].get_xlabel()) or [])
    fg.plot_interventions(_m5_rows(), tmp_path, unit="block_4096")
    fg.plot_interventions(_m5_rows(), tmp_path, unit="article")
    assert "4096-token block" in labels[0] and "article" not in labels[0]
    assert "article-bootstrap" in labels[1]


# ----------------------------------------------------------------------------------------------
# F120: rows 6 and 10 and the M4 decomposition under a comparison rope_mode
# ----------------------------------------------------------------------------------------------

# (rope_mode, embed_dim, rope_fraction): heads of 2 rotate every pair (the diagonal P(G) route of
# comparison_invariant); heads of 4 at p = 0.5 keep a two-coordinate block (the dense P(G) route).
ROPE_MODES = [("none", 4, 1.0), ("comparison", 4, 1.0), ("comparison_invariant", 4, 1.0),
              ("comparison", 8, 0.5), ("comparison_invariant", 8, 0.5)]


def _scorer_inputs(rope_mode: str, embed_dim: int, fraction: float, seed: int = 1) -> Any:
    r"""Float64 operators of a tiny two-head model and a random compact frame U_i."""
    import copy

    from vfe3.contracts import CanonicalFrameContext

    model = _model(_cfg(rope_mode=rope_mode, embed_dim=embed_dim, rope_fraction=fraction))
    ops = copy.deepcopy(model.relational_operators_by_layer[0]).double()
    g = torch.Generator().manual_seed(seed)
    B, N, H, d = 2, 5, 2, embed_dim // 2
    K = H * d
    mu = torch.randn(B, N, K, generator=g, dtype=torch.float64)
    source_mu = torch.randn(B, N, K, generator=g, dtype=torch.float64)
    sigma = torch.rand(B, N, K, generator=g, dtype=torch.float64) + 0.5          # (B, N, K) diagonal
    source_sigma = torch.rand(B, N, K, generator=g, dtype=torch.float64) + 0.5
    A = torch.eye(d, dtype=torch.float64) + 0.3 * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    frame = CanonicalFrameContext(forward=A, inverse=torch.linalg.inv(A), block_diagonal=True)
    U = torch.zeros(B, N, K, K, dtype=torch.float64)                             # (B, N, K, K) blockdiag U_i
    for h in range(H):
        U[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = A[..., h, :, :]
    return ops, frame, U, mu, source_mu, sigma, source_sigma


@pytest.mark.parametrize("rope_mode, embed_dim, fraction", ROPE_MODES)
def test_identity_covariance_score_is_the_trained_router_at_reference_covariance_identity(
        rope_mode: str, embed_dim: int, fraction: float) -> None:
    r"""Row 6 sets :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top} = I`, i.e. :math:`\Sigma_i = U_iU_i^\top` in the trained router."""
    from vfe3.gradients.relational import relational_score

    ops, frame, U, mu, source_mu, sigma, source_sigma = _scorer_inputs(rope_mode, embed_dim, fraction)
    UU = U @ U.transpose(-1, -2)                                                 # (B, N, K, K) C_i = I
    for q, k in ((mu, mu), (mu, source_mu)):
        trained = relational_score(q, UU, k, UU, frame, ops)
        got = iv._identity_covariance_score(q, sigma, k, source_sigma, frame, ops)
        torch.testing.assert_close(got, trained, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("rope_mode, embed_dim, fraction", ROPE_MODES)
def test_frame_free_covariance_score_is_the_trained_router_at_reference_covariance_sigma(
        rope_mode: str, embed_dim: int, fraction: float) -> None:
    r"""Row 10 sets :math:`C_i = \Sigma_i`, i.e. the trained router fed :math:`U_i\Sigma_iU_i^\top`."""
    from vfe3.gradients.relational import relational_score

    ops, frame, U, mu, source_mu, sigma, source_sigma = _scorer_inputs(rope_mode, embed_dim, fraction, seed=2)

    def pushed(s: torch.Tensor) -> torch.Tensor:                                # (B, N, K, K) U diag(s) U^T
        return U @ torch.diag_embed(s) @ U.transpose(-1, -2)

    trained = relational_score(mu, pushed(sigma), source_mu, pushed(source_sigma), frame, ops)
    got = iv._frame_free_covariance_score(mu, sigma, source_mu, source_sigma, frame, ops)
    torch.testing.assert_close(got, trained, rtol=1e-9, atol=1e-9)
    self_scored = iv._frame_free_covariance_score(mu, sigma, mu, sigma, frame, ops)
    torch.testing.assert_close(self_scored, relational_score(mu, pushed(sigma), mu, pushed(sigma), frame, ops),
                               rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("rope_mode", ["comparison", "comparison_invariant"])
def test_routing_covariances_identity_runs_under_comparison_rope(rope_mode: str) -> None:
    r"""Identity prior covariances and frames give :math:`C_i = I` already, so row 6 is a no-op there."""
    model = _model(_cfg(rope_mode=rope_mode))
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.zero_()
        model.prior_bank.phi_embed.zero_()
        model.pos_phi_free.zero_()
    dataset = st.eval_windows(_stream(61, 4), 8, 4)
    indices = np.arange(len(dataset))
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    with iv.routing_covariances_identity(model) as record:
        arm = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert record["intercepted_calls"] > 0
    np.testing.assert_allclose(arm.nll, base.nll, rtol=0, atol=1e-5)
    moved = _model(_cfg(rope_mode=rope_mode))
    with iv.routing_covariances_identity(moved):
        changed = ev.nll_pass(moved, dataset, indices, batch_size=2, device=DEVICE)
    assert np.abs(changed.nll - ev.nll_pass(moved, dataset, indices, batch_size=2, device=DEVICE).nll).max() > 1e-6


@pytest.mark.parametrize("embed_dim, fraction", [(4, 1.0), (8, 0.5)], ids=["diagonal_P", "dense_P"])
def test_routing_decomposition_is_exact_under_comparison_invariant_rope(embed_dim: int, fraction: float) -> None:
    from tests.test_posthoc_routing_20260928 import _cfg as routing_cfg
    from tests.test_posthoc_routing_20260928 import _model as routing_model
    from tests.test_posthoc_routing_20260928 import _run

    result = _run(routing_model(routing_cfg(rope_mode="comparison_invariant", embed_dim=embed_dim,
                                            rope_fraction=fraction)))
    assert result["status"] == "ok", result.get("reason")
    worst = result["checks_worst"]
    assert worst["identity_max_rel_error"] < 1e-12
    assert worst["model_cost_max_abs_error"] < 1e-10
    assert worst["model_logit_max_abs_error"] < 1e-10
    assert worst["model_beta_max_abs_error"] < 1e-12


def test_routing_decomposition_still_refuses_the_full_rotated_comparison_covariance() -> None:
    from tests.test_posthoc_routing_20260928 import _cfg as routing_cfg
    from tests.test_posthoc_routing_20260928 import _model as routing_model
    from vfe3.posthoc import routing as rt

    reason = rt.not_applicable_reason(routing_model(routing_cfg(rope_mode="comparison")))
    assert reason is not None and "rope_mode='comparison'" in reason
