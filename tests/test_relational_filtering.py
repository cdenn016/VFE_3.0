"""Tiny CPU checks of the actual conditional update and its outer derivative."""
import torch
import pytest

from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.transport import FactoredTransport
from vfe3.gradients.relational import compute_relational_filtering_gradients
from vfe3.gradients.qkv import qkv_conditional_cost
from vfe3.model.relational_operators import RelationalOperators


def _fixture() -> tuple:
    mu = torch.tensor([[[0.2, -0.4, 0.1, 0.7], [1.1, 0.5, -0.3, 0.8],
                        [0.6, 1.2, 0.9, -0.2]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([[[0.8, 1.2, 0.9, 1.4], [1.1, 0.6, 1.5, 0.7],
                           [0.7, 1.3, 0.8, 1.1]]], dtype=mu.dtype)
    U = torch.eye(4, dtype=mu.dtype).expand(1, 3, 4, 4).clone()
    U[..., 0, 1] = 0.3
    U[..., 2, 3] = -0.2
    V = torch.linalg.inv(U)
    frame = CanonicalFrameContext(U, V)
    omega = FactoredTransport(U, V, [2, 2], same_frame_flat_cocycle=True)
    ops = RelationalOperators([2, 2], 11).double()
    prior = torch.tensor([[0., -torch.inf, -torch.inf], [0., 0., -torch.inf],
                          [0., 0., 0.]], dtype=mu.dtype)
    return mu, sigma, omega, frame, ops, prior


def test_inner_force_matches_frozen_cost() -> None:
    mu, sigma, omega, frame, ops, prior = _fixture()
    cap = {}
    grad, covariance_grad = compute_relational_filtering_gradients(
        mu, sigma, mu.detach() + 0.2, sigma, omega, ops, frame,
        log_prior=prior, tau=0.9, capture=cap, create_graph=True)
    assert covariance_grad is None
    probe = mu.detach().clone().requires_grad_(True)
    expected, = torch.autograd.grad(qkv_conditional_cost(probe, cap['conditional_context']), probe)
    torch.testing.assert_close(grad, expected, rtol=1e-9, atol=1e-10)
    assert not cap['beta_pre'].requires_grad
    assert torch.equal(cap['beta_pre'][..., 0, 1:], torch.zeros_like(cap['beta_pre'][..., 0, 1:]))
    assert 'relational_conditional_cost_before' in cap


@pytest.mark.parametrize('parameter_name', ['raw_score_q', 'raw_score_k', 'T', 'O'])
def test_outer_parameter_finite_difference(parameter_name: str) -> None:
    mu, sigma, omega, frame, ops, prior = _fixture()
    ops = RelationalOperators([2, 2], 11).double()
    # Stay away from the strict zero-KL endpoint mask the identity content start sits on:
    # perturbing a map off zero flips 1_{0 < D < kl_max} on the self pairs, so the analytic
    # derivative jumps there while the finite difference stays continuous.
    with torch.no_grad():
        for _p in (ops.raw_score_q, ops.raw_score_k, ops.T, ops.O):
            _p.add_(.02 * torch.randn_like(_p))
    parameter = getattr(ops, parameter_name)
    def objective() -> torch.Tensor:
        gradient, _ = compute_relational_filtering_gradients(
            mu, sigma, mu.detach() + 0.2, sigma, omega, ops, frame,
            log_prior=prior, tau=0.9, create_graph=True)
        return (mu - 0.1 * sigma * gradient)[0, 2].square().sum()
    derivative, = torch.autograd.grad(objective(), parameter)
    assert derivative.abs().sum() > 1e-8
    index = tuple(0 for _ in parameter.shape)
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter[index] += 1e-6
    plus = objective().detach()
    with torch.no_grad():
        parameter.copy_(original)
        parameter[index] -= 1e-6
    minus = objective().detach()
    with torch.no_grad():
        parameter.copy_(original)
    torch.testing.assert_close(derivative[index], (plus-minus)/2e-6, rtol=1e-5, atol=1e-7)


def test_source_beliefs_detached_and_no_grad_evaluation() -> None:
    mu, sigma, omega, frame, ops, prior = _fixture()
    args = (mu, sigma, mu.detach(), sigma, omega, ops, frame)
    gradient, _ = compute_relational_filtering_gradients(*args, log_prior=prior, create_graph=True)
    state_derivative, = torch.autograd.grad(gradient[0, 2].sum(), mu)
    assert torch.equal(state_derivative[0, :2], torch.zeros_like(state_derivative[0, :2]))
    with torch.no_grad():
        evaluated, _ = compute_relational_filtering_gradients(*args, log_prior=prior, create_graph=True)
    torch.testing.assert_close(evaluated, gradient.detach())
    assert not evaluated.requires_grad
