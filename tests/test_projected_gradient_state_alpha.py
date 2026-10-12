"""Tiny CPU checks for the profiled state-dependent self-coupling on projected_gradient.

alpha*(D) = c0/(b0 + D) minimizes alpha D + b0 alpha - c0 log alpha, so the receiver energy profiles
to E = c0 log(b0 + D) + G (projected_gradient.tex, Sec. profiled self-coupling).
"""

from typing import Any

import pytest
import torch

from tests.test_full_link_gradients import _fixture
from tests.test_projected_gradient import _cfg, _tokens
from vfe3.belief import BeliefState
from vfe3.gradients.full_link import (
    FullLinkContext, build_full_link_context, full_link_context_at, full_link_cost,
    full_link_gradients, full_link_minimizer, self_coupling_divergence,
)
from vfe3.inference.e_step import e_step
from vfe3.model.model import VFEModel

NOISE, HEADS, B0, C0 = .7, (.6, 1.3), .8, 1.7
CASES = [("state_dependent", False), ("state_dependent", True), ("state_dependent_per_coord", False)]


@pytest.fixture(autouse=True)
def _single_cpu_thread() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _context(mode: str, full: bool, b0: Any = B0, c0: Any = C0) -> tuple:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=full)
    context = build_full_link_context(
        mu.detach(), sigma.detach(), mu_p, sigma_p, frame, ops, tau=1.0,
        lambda_beta=torch.tensor(HEADS, dtype=mu.dtype), receiver_noise=NOISE,
        log_prior=prior, lambda_alpha_mode=mode, b0=b0, c0=c0)
    return mu.detach(), sigma.detach(), mu_p, sigma_p, frame, ops, prior, context


def _away(mu: torch.Tensor, sigma: torch.Tensor) -> tuple:
    r_mu = mu * .7 - .1
    r_sigma = sigma * 1.3
    if r_sigma.ndim == r_mu.ndim + 1:
        r_sigma = r_sigma + .05 * torch.eye(4, dtype=mu.dtype)
    return r_mu, r_sigma


@pytest.mark.parametrize(("mode", "full"), CASES)
def test_profiled_energy_is_c0_log_b0_plus_literal_kl(mode: str, full: bool) -> None:
    mu, sigma, mu_p, sigma_p, *_, ctx = _context(mode, full)
    r_mu, r_sigma = _away(mu, sigma)
    # The channel law carries no coupling entropy: the coupling-only energy has c = 0.
    coupling = full_link_cost(r_mu, r_sigma, FullLinkContext(
        ctx.profile.Lambda, ctx.profile.eta, torch.zeros_like(ctx.profile.prior_logdet), ctx.beta, ctx.score))
    cov = lambda m, s: torch.diag_embed(s) if s.ndim == m.ndim else s
    if mode == "state_dependent_per_coord":
        literal = torch.distributions.kl_divergence(
            torch.distributions.Normal(r_mu, r_sigma.sqrt()), torch.distributions.Normal(mu_p, sigma_p.sqrt()))
    else:
        literal = torch.distributions.kl_divergence(
            torch.distributions.MultivariateNormal(r_mu, cov(r_mu, r_sigma)),
            torch.distributions.MultivariateNormal(mu_p, cov(mu_p, sigma_p)))
    torch.testing.assert_close(self_coupling_divergence(r_mu, r_sigma, ctx.profile), literal,
                               atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(full_link_cost(r_mu, r_sigma, ctx) - coupling,
                               (C0 * torch.log(B0 + literal)).sum(), atol=1e-12, rtol=1e-11)


def _fd_gradient(energy: Any, x: torch.Tensor, symmetric: bool, h: float = 1e-6) -> torch.Tensor:
    """Central differences; a full covariance is perturbed symmetrically and returned as G with
    dE = tr(G dS), so the off-diagonal entry is half the symmetric directional derivative."""
    out = torch.zeros_like(x)
    flat = out.view(-1)
    for index in range(x.numel()):
        direction = torch.zeros_like(x).view(-1)
        direction[index] = 1.0
        direction = direction.view_as(x)
        if symmetric:
            direction = torch.maximum(direction, direction.transpose(-1, -2))
        value = (energy(x + h * direction) - energy(x - h * direction)) / (2 * h)
        off_diagonal = symmetric and bool((direction != direction * torch.eye(
            x.shape[-1], dtype=x.dtype)).any())
        flat[index] = value / 2 if off_diagonal else value
    return out


@pytest.mark.parametrize(("mode", "full"), CASES)
def test_euler_force_is_the_finite_difference_gradient_of_the_profiled_energy(mode: str, full: bool) -> None:
    mu, sigma, *_, ctx = _context(mode, full)
    r_mu, r_sigma = _away(mu, sigma)
    g_mu, g_sigma = full_link_gradients(r_mu, r_sigma, full_link_context_at(ctx, r_mu, r_sigma))
    fd_mu = _fd_gradient(lambda x: full_link_cost(x, r_sigma, ctx), r_mu, False)
    fd_sigma = _fd_gradient(lambda x: full_link_cost(r_mu, x, ctx), r_sigma, full)
    assert float((g_mu - fd_mu).abs().max()) < 1e-7
    assert float((g_sigma - fd_sigma).abs().max()) < 1e-7
    # The snapshot-frozen alpha is stale at r: its force is not the profiled gradient.
    stale_mu, _ = full_link_gradients(r_mu, r_sigma, ctx)
    assert float((stale_mu - fd_mu).abs().max()) > 1e-4


def test_per_coordinate_alpha_accepts_coordinate_vectors() -> None:
    b0 = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.float64)
    c0 = torch.tensor([2., 1., .7, 1.3], dtype=torch.float64)
    mu, sigma, mu_p, sigma_p, *_, ctx = _context("state_dependent_per_coord", False, b0, c0)
    r_mu, r_sigma = _away(mu, sigma)
    at = full_link_context_at(ctx, r_mu, r_sigma)
    D = self_coupling_divergence(r_mu, r_sigma, ctx.profile)
    torch.testing.assert_close(at.c, c0 / (b0 + D))                      # channel: c_k = alpha_k
    g_mu, g_sigma = full_link_gradients(r_mu, r_sigma, at)
    torch.testing.assert_close(g_mu, _fd_gradient(lambda x: full_link_cost(x, r_sigma, ctx), r_mu, False),
                               atol=1e-7, rtol=0)
    torch.testing.assert_close(g_sigma, _fd_gradient(lambda x: full_link_cost(r_mu, x, ctx), r_sigma, False),
                               atol=1e-7, rtol=0)
    # The exact diagonal closed form at frozen alpha: s_k = c_k / Lambda_kk, mu = Lambda^{-1} eta.
    new_mu, new_sigma = full_link_minimizer(r_mu, r_sigma, at)
    torch.testing.assert_close(new_sigma, at.c / at.Lambda.diagonal(dim1=-2, dim2=-1))
    grad_mu, grad_sigma = full_link_gradients(new_mu, new_sigma, at)
    assert float(grad_mu.abs().max()) < 1e-10 and float(grad_sigma.abs().max()) < 1e-10


def _profiled_trajectory(mode: str, full: bool, damping: float, n_iter: int) -> list:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=full)
    ops.readout_mode = "full_link_update"
    ops.receiver_update, ops.receiver_damping = "closed_form", damping
    r_mu, r_sigma = _away(mu.detach(), sigma.detach())
    belief = BeliefState(r_mu, r_sigma, torch.zeros(1, 3, 2, dtype=mu.dtype))
    capture: dict = {}
    record: dict = {}
    e_step(belief, mu_p, sigma_p, None, log_prior=prior, relational_operators=ops, relational_frame=frame,
           relational_capture=capture, state_record=record, n_iter=n_iter, sigma_max=None,
           family="gaussian_full" if full else "gaussian_diagonal", lambda_beta=torch.tensor(HEADS, dtype=mu.dtype),
           lambda_alpha_mode=mode, b0=B0, c0=C0)
    ctx = capture["full_link_context"]
    costs = [float(full_link_cost(b.mu, b.sigma, ctx)) for b in record["beliefs"]]
    assert float(capture["relational_receiver_cost_before"]) == costs[0]
    assert float(capture["relational_receiver_cost_after"]) == costs[-1]
    return costs, record["beliefs"][-1], ctx


@pytest.mark.parametrize(("mode", "full"), CASES)
def test_closed_form_mm_step_never_increases_the_profiled_energy(mode: str, full: bool) -> None:
    costs, last, ctx = _profiled_trajectory(mode, full, 1.0, 8)
    assert all(b <= a + 1e-12 for a, b in zip(costs, costs[1:])), costs
    assert costs[-1] < costs[0] - 1e-3
    # The MM fixed point is a stationary point of the profiled energy.
    costs_long, last, ctx = _profiled_trajectory(mode, full, 1.0, 60)
    g_mu, g_sigma = full_link_gradients(last.mu, last.sigma, full_link_context_at(ctx, last.mu, last.sigma))
    assert float(g_mu.abs().max()) < 1e-8 and float(g_sigma.abs().max()) < 1e-8


@pytest.mark.parametrize("damping", [.3, .7])
def test_damped_mm_step_descends_on_the_full_family(damping: float) -> None:
    """rho < 1 on gaussian_full: the frozen energy is c KL(q||q*) + const and the natural-parameter
    segment decreases KL(q_rho||q*) monotonically, so the MM bound still gives descent."""
    costs, *_ = _profiled_trajectory("state_dependent", True, damping, 8)
    assert all(b <= a + 1e-12 for a, b in zip(costs, costs[1:])), costs


def test_diagonal_damping_is_not_a_descent_guarantee() -> None:
    """On gaussian_diagonal the rho < 1 interpolation contracts the mean error by a non-uniform
    diagonal, which can raise the dense quadratic: the frozen energy itself increases here, so the
    MM bound gives no descent guarantee for diagonal rho < 1 (rho = 1 is the exact minimizer)."""
    Lam = torch.tensor([[[[1., .99], [.99, 1.]]]], dtype=torch.float64)
    ctx = FullLinkContext(Lam, torch.zeros(1, 1, 2, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64),
                          None, None)
    mu, s = torch.tensor([[[5., -5.]]], dtype=torch.float64), torch.tensor([[[.5, 2.]]], dtype=torch.float64)
    before = float(full_link_cost(mu, s, ctx))
    assert float(full_link_cost(*full_link_minimizer(mu, s, ctx, damping=.5), ctx)) > before + .5
    assert float(full_link_cost(*full_link_minimizer(mu, s, ctx), ctx)) < before


def test_euler_episode_descends_with_small_steps() -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture()
    ops.readout_mode = "full_link_update"
    ops.receiver_update = "euler"
    r_mu, r_sigma = _away(mu.detach(), sigma.detach())
    record: dict = {}
    capture: dict = {}
    e_step(BeliefState(r_mu, r_sigma, torch.zeros(1, 3, 2, dtype=mu.dtype)), mu_p, sigma_p, None,
           log_prior=prior, relational_operators=ops, relational_frame=frame, relational_capture=capture,
           state_record=record, n_iter=5, e_q_mu_lr=.05, e_q_sigma_lr=.05, sigma_max=None,
           e_sigma_q_trust=50.0, lambda_alpha_mode="state_dependent_per_coord", b0=B0, c0=C0)
    ctx = capture["full_link_context"]
    costs = [float(full_link_cost(b.mu, b.sigma, ctx)) for b in record["beliefs"]]
    assert all(b < a for a, b in zip(costs, costs[1:])), costs


def test_full_family_rejects_per_coordinate_alpha_in_the_receiver() -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=True)
    with pytest.raises(ValueError, match="diagonal family"):
        build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior,
                                lambda_alpha_mode="state_dependent_per_coord")


def test_config_accepts_state_dependent_alpha_and_predictive_knobs_as_inactive() -> None:
    cfg = _cfg(lambda_alpha_mode="state_dependent", predictive_covariance="moment_matched",
               attention_residual=True)
    inactive = cfg.inactive_fields()
    for name in ("predictive_covariance", "predictive_covariance_floor", "attention_residual"):
        assert name in inactive, name
    assert "b0" not in inactive and "c0" not in inactive
    assert "lambda_alpha_mode" not in inactive
    _cfg(lambda_alpha_mode="state_dependent_per_coord", b0=[1., .5, 2., 1.], c0=1.5)
    with pytest.raises(ValueError, match="requires gaussian_diagonal"):
        _cfg(lambda_alpha_mode="state_dependent_per_coord", family="gaussian_full")
    with pytest.raises(ValueError, match="requires attention_mode='projected_predictive'"):
        _cfg(relational_routing="noisy_query")


def test_inactive_predictive_knobs_leave_the_gradient_route_unchanged() -> None:
    def run(**overrides: Any) -> tuple:
        torch.manual_seed(0)
        model = VFEModel(_cfg(n_layers=2, n_e_steps=2, **overrides))
        ids, targets = _tokens()
        _, loss, _ = model(ids, targets)
        loss.backward()
        return loss.detach(), [p.grad for p in model.parameters()]

    base_loss, base_grads = run()
    loss, grads = run(predictive_covariance="moment_matched", attention_residual=True)
    assert torch.equal(loss, base_loss)
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(grads, base_grads))


@pytest.mark.parametrize(("mode", "update"), [("state_dependent", "closed_form"), ("state_dependent", "euler"),
                                              ("state_dependent_per_coord", "closed_form"),
                                              ("state_dependent_per_coord", "euler")])
def test_model_trains_through_a_state_dependent_receiver(mode: str, update: str) -> None:
    torch.manual_seed(0)
    model = VFEModel(_cfg(n_layers=2, n_e_steps=2, lambda_alpha_mode=mode, relational_receiver_update=update,
                          b0=.7, c0=1.3))
    ids, targets = _tokens()
    _, loss, _ = model(ids, targets)
    loss.backward()
    assert torch.isfinite(loss)
    for operators in model.relational_operators_by_layer:
        for name in ("raw_score_q", "raw_score_k", "T", "O"):
            grad = getattr(operators, name).grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    diagnostics = model.diagnostics(ids, snapshot=model.build_diagnostic_snapshot(ids))
    assert diagnostics["relational_receiver_cost_after"] <= diagnostics["relational_receiver_cost_before"] + 1e-6 \
        or update == "euler"
