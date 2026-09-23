"""Projection roundoff must not reject a mathematically SPD covariance."""

import pytest
import torch

from vfe3.relational_scoring import projected_gaussian_kl, projected_moments


@pytest.mark.parametrize("full", [False, True])
def test_projected_covariance_roundoff_preserves_scores_and_gradients(
    device: torch.device,
    full:   bool,
) -> None:
    d, seed = (32, 0) if device.type == "cuda" else (3, 19)
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(1, 3, 2, d, generator=generator).to(device)
    sigma = (torch.rand(1, 3, 2, d, generator=generator) + 14).to(device)
    inverse = (torch.eye(d).expand(1, 3, 2, d, d)
               + .035 * torch.randn(1, 3, 2, d, d, generator=generator)).to(device)
    S = torch.eye(d, device=device).expand(2, d, d)
    covariance = torch.diag_embed(sigma) if full else sigma
    inverse.requires_grad_()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        a, G = projected_moments(mu, covariance, inverse, S)
        assert torch.equal(G, G.transpose(-1, -2))
        actual = projected_gaussian_kl(a, G, a, G)
    assert G.dtype == actual.dtype == torch.float32
    actual_gradient, = torch.autograd.grad(actual.square().sum(), inverse)
    inverse64 = inverse.detach().double().requires_grad_()
    a64, G64 = projected_moments(mu.double(), covariance.double(), inverse64, S.double())
    expected = projected_gaussian_kl(a64, G64, a64, G64)
    expected_gradient, = torch.autograd.grad(expected.square().sum(), inverse64)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=5e-6)
    torch.testing.assert_close(actual_gradient.double(), expected_gradient, rtol=3e-4, atol=5e-5)


def test_direct_scorer_still_rejects_asymmetric_covariance() -> None:
    a = torch.zeros(1, 2, 1, 2)
    G = torch.tensor([[1., .2], [0., 1.]]).expand(1, 2, 1, 2, 2)
    with pytest.raises(ValueError, match="symmetric"):
        projected_gaussian_kl(a, G, a, G)
