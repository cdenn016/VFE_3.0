"""Tiny regressions for explicit decoder memory and numerical tradeoffs."""
import math
from unittest.mock import patch

import pytest
import torch

import vfe3.model.prior_bank as bank_module
from tests.test_decode_learning_rate_20260930 import _config
from vfe3.baseline import BaselineConfig
from vfe3.config import config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank

MODES = ("diagonal_chunked", "full_chunked", "expected_likelihood_chunked")


def _bank(mode: str, **kwargs: object) -> PriorBank:
    full = mode == "full_chunked"
    return PriorBank(7, 2, 3, family="gaussian_full" if full else "gaussian_diagonal",
                     diagonal_covariance=not full, decode_mode=mode, decode_chunk_size=4,
                     decode_av_precision="fp32", decode_ranking_fp64_escalation=False, **kwargs)


def _query(mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_q = torch.full((1, 3, 2), 0.2, requires_grad=True)
    sigma_q = (torch.eye(2).expand(1, 3, 2, 2).clone() if mode == "full_chunked"
               else torch.ones_like(mu_q)).requires_grad_(True)
    return mu_q, sigma_q, torch.tensor([[0, 4, 6]])


@pytest.mark.parametrize("mode", MODES + ("linear",))
def test_aggregate_recomputation_requires_explicit_policy(mode: str) -> None:
    pb = _bank(mode, decode_ce_checkpoint="auto")
    mu_q, sigma_q, targets = _query(mode)
    args = (mu_q, targets) if mode == "linear" else (mu_q, sigma_q, targets)
    fn = getattr(pb, "decode_ce_" + ("linear_chunked" if mode == "linear" else mode))
    # Old chunks are <=96 bytes (linear <=48); aggregate storage is >=168 (linear 84).
    with patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 60 if mode == "linear" else 100):
        with patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
            legacy = fn(*args)
        assert checkpoint.call_count == 0
        pb.decode_ce_checkpoint = "auto_aggregate"
        with patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
            aggregate = fn(*args)
        assert checkpoint.call_count == 2
    inputs = (mu_q, pb.output_proj_weight) if mode == "linear" else (mu_q, sigma_q, pb.mu_embed, pb.sigma_log_embed)
    torch.testing.assert_close(aggregate, legacy, atol=0, rtol=0)
    for actual, expected in zip(torch.autograd.grad(aggregate, inputs), torch.autograd.grad(legacy, inputs)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_centered_reduction_matches_the_dense_ce(mode: str, checkpoint: str) -> None:
    pb = _bank(mode, decode_ce_checkpoint=checkpoint)
    mu_q, sigma_q, targets = _query(mode)
    fused = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets)
    dense = torch.nn.functional.cross_entropy(pb.decode(mu_q, sigma_q).reshape(-1, 7), targets.reshape(-1))
    torch.testing.assert_close(fused, dense, atol=3e-6, rtol=3e-6)
    inputs = (mu_q, sigma_q, pb.mu_embed, pb.sigma_log_embed)
    for actual, expected in zip(torch.autograd.grad(fused, inputs), torch.autograd.grad(dense, inputs)):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("mode", MODES)
def test_centered_reduction_is_exact_at_large_scores(mode: str) -> None:
    # Absolute scores of order 1e11 cancel in an uncentered logsumexp - target; the identical
    # vocabulary rows give exactly log V.
    pb = _bank(mode, decode_ce_checkpoint="never")
    with torch.no_grad():
        pb.mu_embed.zero_()
        pb.sigma_log_embed.fill_(math.log(1e-4))
    mu_q, sigma_q, targets = _query(mode)
    with torch.no_grad():
        mu_q.fill_(3000.0)
    loss = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets)
    torch.testing.assert_close(loss, loss.new_tensor(math.log(7)), atol=3e-7, rtol=0)


def test_config_roundtrip_reaches_model_and_matched_baseline() -> None:
    cfg = _config(decode_ce_checkpoint="auto_aggregate")
    restored = config_from_serialized(serialize_config(cfg), source="decoder tradeoff test")
    model = VFEModel(restored)
    assert model.prior_bank.decode_ce_checkpoint == "auto_aggregate"
    baseline = BaselineConfig(vocab_size=7, embed_dim=2, n_heads=1, n_layers=1,
                              max_seq_len=3, batch_size=1, decode_ce_checkpoint="auto_aggregate")
    assert baseline.decode_ce_checkpoint == "auto_aggregate"


@pytest.mark.parametrize("mode", MODES)
def test_aggregate_policy_does_not_checkpoint_evaluation(mode: str) -> None:
    pb = _bank(mode, decode_ce_checkpoint="auto_aggregate")
    mu_q, sigma_q, targets = _query(mode)
    with torch.no_grad(), patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 1), \
         patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
        result = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets)
    assert checkpoint.call_count == 0
    assert torch.isfinite(result)
