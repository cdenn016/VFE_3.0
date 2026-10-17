r"""Audit 2026-10-16, lane C2 (post-hoc analysis): F65, F66, F68, F69, F120.

Synthetic strings and tiny models only (single-digit dimensions, CPU, one thread); nothing is trained.

* F65: :func:`vfe3.posthoc.calibration.text_audit` reports ``same_text`` only when every literal
  segment matched in its wildcard window and every ``<unk>`` wildcard matched characters.
* F66: M5 row 5 sets every stored covariance table to the identity, the untied decode bank's
  ``decode_sigma_log_embed`` included; ``decode=False`` keeps the earlier encode-side-only arm.
* F68: the reproduction check calls a replay matched arithmetic only when its amp dtype and its fp32
  matmul policy are the run's; an IEEE fp32 replay of a TF32 run gets the fp32-replay tolerance.
"""

from __future__ import annotations

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
