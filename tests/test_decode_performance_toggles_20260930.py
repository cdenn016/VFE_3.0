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

MODES = ("diagonal_chunked", "full_chunked", "expected_likelihood_chunked", "family_chunked")


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


@pytest.mark.parametrize("mode", MODES[:3] + ("linear",))
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
def test_stable_toggle_preserves_ordinary_loss_and_gradients(mode: str, checkpoint: str) -> None:
    pb = _bank(mode, decode_ce_checkpoint=checkpoint)
    mu_q, sigma_q, targets = _query(mode)
    fn = getattr(pb, "decode_ce_" + mode)
    ordinary = fn(mu_q, sigma_q, targets)
    pb.decode_ce_stable_reduction = True
    stable = fn(mu_q, sigma_q, targets)
    torch.testing.assert_close(stable, ordinary, atol=3e-6, rtol=3e-6)
    inputs = (mu_q, sigma_q, pb.mu_embed, pb.sigma_log_embed)
    for actual, expected in zip(torch.autograd.grad(stable, inputs), torch.autograd.grad(ordinary, inputs)):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("mode", MODES)
def test_default_reduction_uses_legacy_arithmetic_at_large_scores(mode: str) -> None:
    pb = _bank(mode, decode_ce_checkpoint="never")
    with torch.no_grad():
        pb.mu_embed.zero_()
        pb.sigma_log_embed.fill_(math.log(1e-4))
    mu_q, sigma_q, targets = _query(mode)
    with torch.no_grad():
        mu_q.fill_(3000.0)
    logits = pb.decode(mu_q, sigma_q)
    expected = torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets)
    torch.testing.assert_close(loss, expected.mean(), atol=0, rtol=0)
    pb.decode_ce_stable_reduction = True
    corrected = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets)
    torch.testing.assert_close(corrected, corrected.new_tensor(math.log(7)), atol=3e-7, rtol=0)


@pytest.mark.parametrize("stable", [False, True])
def test_config_roundtrip_reaches_model_and_matched_baseline(stable: bool) -> None:
    cfg = _config(decode_ce_checkpoint="auto_aggregate", decode_ce_stable_reduction=stable)
    restored = config_from_serialized(serialize_config(cfg), source="decoder tradeoff test")
    model = VFEModel(restored)
    assert model.prior_bank.decode_ce_checkpoint == "auto_aggregate"
    assert model.prior_bank.decode_ce_stable_reduction is stable
    baseline = BaselineConfig(vocab_size=7, embed_dim=2, n_heads=1, n_layers=1,
                              max_seq_len=3, batch_size=1, decode_ce_checkpoint="auto_aggregate")
    assert baseline.decode_ce_checkpoint == "auto_aggregate"


@pytest.mark.parametrize("value", [0, 1, "False", None])
def test_stable_toggle_rejects_non_boolean_config(value: object) -> None:
    with pytest.raises(ValueError, match="decode_ce_stable_reduction"):
        _config(decode_ce_stable_reduction=value)


@pytest.mark.parametrize("policy", ["auto", "auto_aggregate"])
@pytest.mark.parametrize("stable", [False, True])
def test_generic_family_keeps_aggregate_policy_and_excluded_row_gradients(
    policy: str,
    stable: bool,
) -> None:
    pb = _bank("full_chunked", decode_ce_checkpoint=policy, decode_ce_stable_reduction=stable)
    pb.decode_mode = "family_chunked"
    pb.renyi_order = 0.5  # Exercise the generic full-family route rather than the analytic delegate.
    mu_q, sigma_q, targets = _query("full_chunked")
    with torch.no_grad():
        sigma_q[:, 1] = -torch.eye(2)
    with patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 100), \
         patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
        result = pb.decode_ce_family_chunked(mu_q, sigma_q, targets, return_stats=True)
    assert checkpoint.call_count == 2
    assert int(result.excluded_tokens) == 1
    result.ce.backward()
    assert torch.isfinite(result.ce)
    assert torch.isfinite(mu_q.grad).all()
    assert torch.isfinite(sigma_q.grad).all()
    assert torch.equal(sigma_q.grad[:, 1], torch.zeros_like(sigma_q.grad[:, 1]))


@pytest.mark.parametrize("policy", ["auto", "auto_aggregate"])
def test_canonical_full_family_delegates_checkpoint_policy(policy: str) -> None:
    pb = _bank("full_chunked", decode_ce_checkpoint=policy)
    pb.decode_mode = "family_chunked"
    mu_q, sigma_q, targets = _query("full_chunked")
    with patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 100), \
         patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
        result = pb.decode_ce_family_chunked(mu_q, sigma_q, targets)
    assert checkpoint.call_count == (2 if policy == "auto_aggregate" else 0)
    result.backward()
    assert torch.isfinite(mu_q.grad).all()


@pytest.mark.parametrize("mode", MODES)
def test_aggregate_policy_does_not_checkpoint_evaluation(mode: str) -> None:
    pb = _bank(mode, decode_ce_checkpoint="auto_aggregate")
    mu_q, sigma_q, targets = _query(mode)
    with torch.no_grad(), patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 1), \
         patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
        result = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets)
    assert checkpoint.call_count == 0
    assert torch.isfinite(result)
