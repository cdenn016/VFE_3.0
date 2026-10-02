"""Centering of the predictive covariance's between-source spread (always on since 2026-10-02)."""

import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.relational_operators import RelationalOperators


def _operator() -> RelationalOperators:
    ops = RelationalOperators([2], 3, readout_mode="predictive_content", predictive_covariance="moment_matched")
    with torch.no_grad():
        ops.T.copy_(torch.eye(2).unsqueeze(0))
        ops.O.copy_(torch.eye(2).unsqueeze(0))
    return ops


def _frame(mu: torch.Tensor) -> CanonicalFrameContext:
    identity = torch.eye(2, dtype=mu.dtype).expand(1, 3, 2, 2)
    return CanonicalFrameContext(identity, identity)


def test_centered_spread_keeps_an_offset_weighted_spread() -> None:
    # Raw float32 moments lose this exactly representable weighted spread (the historical
    # uncentered arithmetic returned 1.0001); the shift by token 0's mean keeps it.
    mu = torch.tensor([[[9999., 10001.], [10000., 10000.], [10001., 9999.]]])
    sigma = torch.ones_like(mu)
    beta = torch.tensor([.25, .5, .25]).expand(1, 1, 3, 3)
    actual = _operator().predictive_covariance_update(mu, sigma, beta, _frame(mu))
    torch.testing.assert_close(actual, torch.full_like(mu, 1.5001), rtol=0, atol=1e-6)


def test_centered_spread_equals_the_raw_moments_in_float64() -> None:
    # The shift is exact in real arithmetic because every beta row sums to one.
    ops = _operator().double()
    generator = torch.Generator().manual_seed(3)
    mu = torch.randn(1, 3, 2, generator=generator, dtype=torch.float64)
    sigma = torch.rand(1, 3, 2, generator=generator, dtype=torch.float64) + .5
    beta = torch.tril(torch.rand(1, 1, 3, 3, generator=generator, dtype=torch.float64))
    beta = beta / beta.sum(-1, keepdim=True)
    actual = ops.predictive_covariance_update(mu, sigma, beta, _frame(mu))
    values, weights = mu[0], beta[0, 0]
    mean = weights @ values
    second = torch.einsum("ij,jk,jl->ikl", weights, values, values)
    spread = torch.diagonal(second - mean.unsqueeze(-1) * mean.unsqueeze(-2), dim1=-2, dim2=-1)
    pooled = weights @ sigma[0]
    floor = ops.predictive_covariance_floor * sigma[0].mean(-1, keepdim=True)
    torch.testing.assert_close(actual[0], pooled + spread + floor, rtol=1e-12, atol=1e-12)


def test_centering_gradients_through_normalized_routing() -> None:
    ops = _operator().double()
    mu = torch.tensor([[[.2, -.1], [.4, .3], [-.2, .5]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.ones_like(mu)
    logits = torch.linspace(-.4, .4, 9, dtype=torch.float64).reshape(1, 1, 3, 3).requires_grad_()
    frame = _frame(mu)

    def covariance(mu_in: torch.Tensor, logits_in: torch.Tensor) -> torch.Tensor:
        return ops.predictive_covariance_update(mu_in, sigma, logits_in.softmax(-1), frame)

    assert torch.autograd.gradcheck(covariance, (mu, logits), fast_mode=True)
