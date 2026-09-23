"""Ordered conditional Gaussian KL contributions, including correlations."""
import math
import pytest
import torch
from vfe3.families.gaussian import FullGaussian, DiagonalGaussian
from vfe3.free_energy import self_divergence_for_alpha
from tests.test_relational_integration import _cfg


def test_correlated_kl_coordinates_match_conditional_reference() -> None:
    mu = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
    Sigma = torch.tensor([[1., .5], [.5, 1.]], dtype=torch.float64, requires_grad=True)
    q = FullGaussian(mu, Sigma)
    p = FullGaussian(torch.zeros_like(mu), torch.eye(2, dtype=mu.dtype))
    actual = self_divergence_for_alpha(q, p, lambda_alpha_mode="state_dependent_per_coord")
    torch.testing.assert_close(actual, torch.tensor([.02, .5 * (.09 - math.log(.75))], dtype=mu.dtype))
    reference = torch.distributions.kl_divergence(
        torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma),
        torch.distributions.MultivariateNormal(p.mu, covariance_matrix=p.sigma))
    torch.testing.assert_close(actual.sum(), reference, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(actual.sum(), (mu, Sigma), retain_graph=True)
    expected_grad = torch.autograd.grad(reference, (mu, Sigma))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_coordinate_kl_diagonal_limit_and_broadcast() -> None:
    mu = torch.tensor([[.2, -.4, .7], [-.3, .1, .5]], dtype=torch.float64)
    variance = torch.tensor([[.8, 1.2, .6], [1.1, .7, 1.3]], dtype=mu.dtype)
    p_mu = torch.tensor([.1, .2, -.1], dtype=mu.dtype)
    p_var = torch.tensor([1.4, .9, .8], dtype=mu.dtype)
    actual = FullGaussian(mu, torch.diag_embed(variance)).renyi_per_coord(
        FullGaussian(p_mu, torch.diag(p_var)))
    expected = DiagonalGaussian(mu, variance).renyi_per_coord(DiagonalGaussian(p_mu, p_var))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_coordinate_kl_weighted_gradient_check() -> None:
    mu = torch.tensor([.3, -.5], dtype=torch.float64, requires_grad=True)
    root = torch.tensor([[1.1, .0], [.3, .8]], dtype=mu.dtype, requires_grad=True)
    prior = FullGaussian(torch.tensor([-.1, .2], dtype=mu.dtype),
                         torch.tensor([[.9, -.2], [-.2, 1.2]], dtype=mu.dtype))
    def objective(m: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        D = FullGaussian(m, L @ L.T).renyi_per_coord(prior)
        return torch.log1p(D).sum()
    assert torch.autograd.gradcheck(objective, (mu, root), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_full_coordinate_non_kl_renyi_fails_explicitly() -> None:
    q = FullGaussian(torch.zeros(2), torch.eye(2))
    with pytest.raises(ValueError, match="KL"):
        q.renyi_per_coord(q, alpha=.5)
    with pytest.raises(ValueError, match="KL"):
        _cfg(family="gaussian_full", relational_mode="off", renyi_order=.5,
             lambda_alpha_mode="state_dependent_per_coord")


@pytest.mark.parametrize("policy", ["fp64", "fp32_escalate", "fp32_escalate_cond"])
def test_coordinate_kl_precision_and_order(policy: str) -> None:
    mu = torch.tensor([.2, -.3], requires_grad=True)
    Sigma = torch.tensor([[1., .4], [.4, 1.2]], requires_grad=True)
    p_mu = torch.tensor([-.2, .1])
    p_sigma = torch.tensor([[1.4, -.2], [-.2, .8]])
    q = FullGaussian(mu, Sigma, _precision_policy=policy)
    p = FullGaussian(p_mu, p_sigma, _precision_policy=policy)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = q.renyi_per_coord(p)
    reference = FullGaussian(mu.double(), Sigma.double()).renyi_closed_form(
        FullGaussian(p_mu.double(), p_sigma.double()))
    torch.testing.assert_close(actual.double().sum(), reference, atol=2e-6, rtol=2e-6)
    perm = torch.tensor([1, 0])
    permuted = FullGaussian(mu[perm], Sigma[perm][:, perm]).renyi_per_coord(
        FullGaussian(p_mu[perm], p_sigma[perm][:, perm]))
    torch.testing.assert_close(actual.sum(), permuted.sum(), atol=2e-6, rtol=2e-6)
    assert not torch.allclose(actual[perm], permuted)
    for grad in torch.autograd.grad(actual.log1p().sum(), (mu, Sigma)):
        assert torch.isfinite(grad).all()
