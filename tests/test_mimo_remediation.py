"""Tiny CPU regressions for the September 22 audit remediation."""

from tests.profiles import config_values, test_config
import json
import math

import pytest
import torch

from vfe3.config import serialize_config, config_from_serialized
from vfe3.families.base import safe_kl_clamp
from vfe3.families.gaussian import diag_kl_unclamped, diag_kl_unclamped_per_coord


def test_near_equal_diagonal_kl_preserves_small_positive_terms_and_gradients() -> None:
    mu = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    sq = torch.tensor([[1., 2., 3.], [1., 2., 3.]], dtype=torch.float64, requires_grad=True)
    sp = sq.detach() * (1 + 1e-7)
    actual = diag_kl_unclamped(mu, sq, mu.detach(), sp)
    x = (sq.detach() - sp) / sp
    expected = (x.square() * (0.5 - x / 3)).sum(-1) / 2
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-25)
    torch.testing.assert_close(actual, diag_kl_unclamped_per_coord(mu, sq, mu.detach(), sp).sum(-1))
    assert torch.autograd.gradcheck(lambda a, b: diag_kl_unclamped(a, b, mu.detach(), sp),
                                    (mu, sq), eps=1e-6, atol=1e-7)


def test_unbounded_kl_ceiling_keeps_gradient_and_json_round_trip() -> None:
    cfg = test_config(kl_max=math.inf)
    payload = json.loads(json.dumps(serialize_config(cfg), allow_nan=False))
    restored = config_from_serialized(payload, source="test")
    assert restored.kl_max == math.inf
    score = torch.tensor([200.], requires_grad=True)
    unclipped = safe_kl_clamp(score, kl_max=restored.kl_max)
    gradient, = torch.autograd.grad(unclipped.sum(), score)
    torch.testing.assert_close(gradient, torch.ones_like(gradient))
    assert safe_kl_clamp(score, kl_max=100.).item() == 100.


@pytest.mark.parametrize("ceiling", [0., -1., -math.inf, math.nan])
def test_invalid_kl_ceiling_is_rejected(ceiling: float) -> None:
    with pytest.raises(ValueError, match="kl_max"):
        test_config(kl_max=ceiling)


def test_adamw_reports_nondefault_group_trust_radius() -> None:
    with pytest.warns(UserWarning, match="m_phi_group_trust_radius"):
        test_config(m_phi_update_mode="adamw", m_phi_group_trust_radius=.2)


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fp64_decode_policy_preserves_completed_logits(mode: str) -> None:
    from vfe3.model.prior_bank import PriorBank
    full = mode == "full_chunked"
    torch.manual_seed(711)
    bank = PriorBank(5, 5, 9, decode_tau=1.7, diagonal_covariance=not full,
                     family="gaussian_full" if full else "gaussian_diagonal", decode_mode=mode,
                     decode_chunk_size=2, decode_ce_checkpoint="off", decode_unigram_prior=True,
                     unigram_kappa=0.37, irrep_dims=[2, 3])
    bank.set_unigram_log_prior(torch.tensor([2, 11, 5, 3, 17]))
    bank.decode_av_precision = "fp64"
    mu = torch.randn(2, 2, 5)
    factor = torch.randn(2, 2, 5, 5) * 0.3 + torch.eye(5)
    sigma = factor @ factor.transpose(-1, -2) if full else torch.rand(2, 2, 5) + 0.5
    logits = bank.decode(mu, sigma)
    assert logits.dtype == torch.float64
    assert torch.isfinite(logits).all()
