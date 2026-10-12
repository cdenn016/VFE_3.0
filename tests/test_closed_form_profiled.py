"""Tiny float64 CPU checks for relational_receiver_update='closed_form_profiled'.

The profiled receiver energy E(q) = c0 log(b0 + D(q)) + G(q) is stationary where alpha = c0/(b0 + D(q)) and q is
the closed-form minimizer at prior weight alpha. Iterated closed_form steps are the fixed-point iteration
a_{t+1} = h(a_t) = c0/(b0 + D(q*(a_t))); closed_form_profiled returns their limit in one solve.
"""

from typing import Any

import pytest
import torch

import vfe3.gradients.full_link as full_link
from tests.test_full_link_gradients import _fixture
from tests.test_projected_gradient import _cfg
from vfe3.belief import BeliefState
from vfe3.gradients.full_link import (
    full_link_context_at, full_link_cost, full_link_gradients, full_link_minimizer, self_coupling_divergence,
)
from vfe3.inference.e_step import e_step

NOISE, HEADS, B0, C0 = .7, (.6, 1.3), .8, 1.7
# (b0, c0): the default pair, a weak regularizer (h near c0/b0 = 60, large alpha moves) and a strong one.
COEFFICIENTS = ((B0, C0), (.05, 3.), (5., .2))


@pytest.fixture(autouse=True)
def _single_cpu_thread() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _setup(full: bool, update: str) -> tuple:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=full)
    ops.readout_mode = "full_link_update"
    ops.receiver_update = update
    r_mu, r_sigma = mu.detach() * .7 - .1, sigma.detach() * 1.3
    if full:
        r_sigma = r_sigma + .05 * torch.eye(4, dtype=mu.dtype)
    return r_mu, r_sigma, mu_p, sigma_p, frame, ops, prior


def _run(full: bool, update: str, n_iter: int, *, b0: float = B0, c0: float = C0,
         mode: str = "state_dependent", damping: float = 1.0, sigma_max: Any = None,
         skip_sigma: bool = False, inputs: Any = None, record: Any = None) -> tuple:
    r_mu, r_sigma, mu_p, sigma_p, frame, ops, prior = _setup(full, update)
    ops.receiver_damping = damping
    if inputs is not None:
        r_mu, r_sigma, mu_p, sigma_p = inputs
    capture: dict = {}
    out = e_step(BeliefState(r_mu, r_sigma, torch.zeros(1, 3, 2, dtype=r_mu.dtype)), mu_p, sigma_p, None,
                 log_prior=prior, relational_operators=ops, relational_frame=frame, relational_capture=capture,
                 state_record=record, n_iter=n_iter, sigma_max=sigma_max,
                 skip_belief_sigma_update=skip_sigma, family="gaussian_full" if full else "gaussian_diagonal",
                 lambda_beta=torch.tensor(HEADS, dtype=r_mu.dtype), lambda_alpha_mode=mode, b0=b0, c0=c0)
    return out, capture


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(("b0", "c0"), COEFFICIENTS)
def test_one_profiled_step_is_the_limit_of_iterated_closed_form(full: bool, b0: float, c0: float) -> None:
    profiled, capture = _run(full, "closed_form_profiled", 1, b0=b0, c0=c0)
    iterated, _ = _run(full, "closed_form", 400, b0=b0, c0=c0)
    one_step, _ = _run(full, "closed_form", 1, b0=b0, c0=c0)
    torch.testing.assert_close(profiled.mu, iterated.mu, atol=1e-12, rtol=0)
    torch.testing.assert_close(profiled.sigma, iterated.sigma, atol=1e-12, rtol=0)
    assert float((profiled.mu - one_step.mu).abs().max()) > 1e-3        # the solve goes beyond one MM step
    ctx = capture["full_link_context"]
    # E at the profiled point is no higher than after one closed_form (MM) step.
    assert float(full_link_cost(profiled.mu, profiled.sigma, ctx)) <= float(
        full_link_cost(one_step.mu, one_step.sigma, ctx)) + 1e-12
    assert float(capture["relational_receiver_profiled_residual"]) < 1e-12 * c0 / b0


@pytest.mark.parametrize("full", [False, True])
def test_returned_state_is_the_profiled_stationary_point(full: bool) -> None:
    out, capture = _run(full, "closed_form_profiled", 1)
    ctx = capture["full_link_context"]
    mu, sigma = out.mu.detach(), out.sigma.detach()
    # alpha = c0/(b0 + D(q)) at the returned q: the closed form frozen there reproduces q ...
    again = full_link_minimizer(mu, sigma, full_link_context_at(ctx, mu, sigma))
    torch.testing.assert_close(again[0], mu, atol=1e-13, rtol=0)
    torch.testing.assert_close(again[1], sigma, atol=1e-13, rtol=0)
    # ... and the profiled-energy force alpha*(q) dD/dq + dG/dq vanishes.
    g_mu, g_sigma = full_link_gradients(mu, sigma, full_link_context_at(ctx, mu, sigma))
    assert float(g_mu.abs().max()) < 1e-12 and float(g_sigma.abs().max()) < 1e-12
    # The root equals alpha*(q) of the returned state to rounding.
    r_mu, r_sigma, *_ = _setup(full, "closed_form_profiled")
    curve = full_link._divergence_curve(ctx.profile, r_mu, r_sigma, True, 1e-6, None)
    a0 = C0 / (B0 + self_coupling_divergence(r_mu, r_sigma, ctx.profile))
    root, *_ = full_link._profiled_root(curve, a0, B0, C0)
    torch.testing.assert_close(root, C0 / (B0 + self_coupling_divergence(mu, sigma, ctx.profile)),
                               atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("full", [False, True])
def test_implicit_gradient_matches_central_differences(full: bool, monkeypatch: Any) -> None:
    r_mu, r_sigma, mu_p, sigma_p, *_ = _setup(full, "closed_form_profiled")
    weights = torch.linspace(-1., 1., r_mu.numel(), dtype=r_mu.dtype).reshape(r_mu.shape)

    def loss(source_mu: torch.Tensor, prior_mu: torch.Tensor, prior_shift: torch.Tensor) -> torch.Tensor:
        # The prior covariance moves along its diagonal, a symmetric direction for the full family.
        prior_sigma = sigma_p + (torch.diag_embed(prior_shift) if full else prior_shift)
        out, _ = _run(full, "closed_form_profiled", 1, inputs=(source_mu, r_sigma, prior_mu, prior_sigma))
        sigma = out.sigma if not full else out.sigma.diagonal(dim1=-2, dim2=-1)
        return (weights * out.mu).sum() + (weights.flip(-1) * sigma).sum()

    inputs = tuple(x.clone().requires_grad_() for x in (r_mu, mu_p, torch.zeros_like(r_mu)))
    assert torch.autograd.gradcheck(loss, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)
    # The implicit term carries the root's derivative: an infinite g' floor detaches alpha and fails the check.
    monkeypatch.setattr(full_link, "_PROFILED_SLOPE_FLOOR", 1e300)
    assert not torch.autograd.gradcheck(loss, inputs, eps=1e-6, atol=1e-6, rtol=1e-5, raise_exception=False)


@pytest.mark.parametrize("full", [False, True])
def test_kept_covariance_and_diagonal_clamp_match_iterated_closed_form(full: bool) -> None:
    for kwargs in ({"skip_sigma": True}, {"sigma_max": .9} if not full else {"skip_sigma": True, "b0": .05}):
        profiled, _ = _run(full, "closed_form_profiled", 1, **kwargs)
        iterated, _ = _run(full, "closed_form", 400, **kwargs)
        torch.testing.assert_close(profiled.mu, iterated.mu, atol=1e-12, rtol=0)
        torch.testing.assert_close(profiled.sigma, iterated.sigma, atol=1e-12, rtol=0)
        if "sigma_max" in kwargs:
            assert bool((profiled.sigma == .9).any())                    # the clamp is active


@pytest.mark.parametrize("full", [False, True])
def test_future_tokens_leave_earlier_receivers_bitwise_unchanged(full: bool) -> None:
    r_mu, r_sigma, mu_p, sigma_p, *_ = _setup(full, "closed_form_profiled")
    first, _ = _run(full, "closed_form_profiled", 2, inputs=(r_mu, r_sigma, mu_p, sigma_p))
    changed_mu, changed_sigma = r_mu.clone(), r_sigma.clone()
    changed_mu[0, 2] += 3.
    changed_sigma[0, 2] *= 2.
    second, _ = _run(full, "closed_form_profiled", 2,
                     inputs=(changed_mu, changed_sigma, mu_p, sigma_p))
    assert torch.equal(first.mu[0, :2], second.mu[0, :2]) and torch.equal(first.sigma[0, :2], second.sigma[0, :2])
    assert not torch.equal(first.mu[0, 2], second.mu[0, 2])


def test_damping_interpolates_toward_the_profiled_point() -> None:
    target, _ = _run(True, "closed_form_profiled", 1)
    damped, capture = _run(True, "closed_form_profiled", 1, damping=.5)
    ctx = capture["full_link_context"]
    r_mu, r_sigma, *_ = _setup(True, "closed_form_profiled")
    # rho = 1/2 in natural parameters: J = (Sigma_0^{-1} + Sigma_r^{-1})/2, h = (J_0 mu_0 + J_r mu_r)/2.
    J0, Jr = torch.linalg.inv(r_sigma), torch.linalg.inv(target.sigma)
    J = .5 * (J0 + Jr)
    torch.testing.assert_close(damped.sigma, torch.linalg.inv(J), atol=1e-12, rtol=0)
    mean = torch.linalg.solve(J, .5 * ((J0 @ r_mu.unsqueeze(-1)) + (Jr @ target.mu.unsqueeze(-1)))).squeeze(-1)
    torch.testing.assert_close(damped.mu, mean, atol=1e-12, rtol=0)
    assert torch.isfinite(full_link_cost(damped.mu, damped.sigma, ctx))


def test_per_coordinate_alpha_is_refused() -> None:
    with pytest.raises(ValueError, match="state_dependent"):
        _run(False, "closed_form_profiled", 1, mode="state_dependent_per_coord")
    with pytest.raises(ValueError, match="closed_form_profiled"):
        _cfg(relational_receiver_update="closed_form_profiled", lambda_alpha_mode="state_dependent_per_coord")


def test_config_accepts_the_profiled_update_and_marks_euler_controls_inert() -> None:
    cfg = _cfg(relational_receiver_update="closed_form_profiled", lambda_alpha_mode="state_dependent")
    inactive = cfg.inactive_fields()
    assert {"e_q_mu_lr", "e_q_sigma_lr", "e_mu_q_trust", "e_sigma_q_trust"} <= set(inactive)
    assert "relational_receiver_damping" not in inactive
    with pytest.raises(ValueError, match="relational_receiver_update"):
        _cfg(relational_receiver_update="profiled")


def _single_term_curve(Q: float) -> full_link._DivergenceCurve:
    r"""K = 1, l = 0: D(a) = Q/a^2, so h(a) = c0 a^2/(b0 a^2 + Q)."""
    f64 = torch.float64
    return full_link._DivergenceCurve(
        torch.zeros(3, 1, dtype=f64), torch.full((3, 1), 2 * Q, dtype=f64),
        "fixed", torch.zeros(3, dtype=f64), None, None, 1e-6, None)


def test_root_is_the_mm_limit_when_fixed_points_are_not_unique() -> None:
    """b0 = 1, c0 = 3, Q = 1: g = a(a^2 - 3a + 1)/(a^2 + 1) has roots 0, (3 - 5^{1/2})/2, (3 + 5^{1/2})/2.
    E is not convex; the MM iteration from a0 goes up to the largest root from (0.382, 2.618) and from
    above 2.618 down to it, and from below 0.382 down to 0."""
    a0 = torch.tensor([1., 3.5, .3], dtype=torch.float64)
    root, residual, g_prime = full_link._profiled_root(_single_term_curve(1.), a0, 1., 3.)
    large = (3 + 5 ** .5) / 2
    torch.testing.assert_close(root[:2], torch.full((2,), large, dtype=torch.float64), atol=1e-14, rtol=0)
    assert float(root[2]) < 1e-12
    assert float(residual.max()) < 1e-14 and bool((g_prime[:2] > 0).all())


def test_random_curves_reach_the_mm_limit() -> None:
    generator = torch.Generator().manual_seed(0)
    f64, n, K = torch.float64, 400, 3
    l = torch.rand(n, K, dtype=f64, generator=generator).pow(3) * 2
    r2 = torch.rand(n, K, dtype=f64, generator=generator).pow(2) * torch.rand(n, 1, dtype=f64, generator=generator) * 20
    b0 = torch.rand(n, dtype=f64, generator=generator) * .5 + .02
    c0 = torch.rand(n, dtype=f64, generator=generator) * 4 + .1
    P = torch.rand(n, K, dtype=f64, generator=generator) + .2
    curves = (full_link._DivergenceCurve(l, r2, "full", torch.zeros(n, dtype=f64), None, None, 1e-6, None),
              full_link._DivergenceCurve(l, r2, "diagonal", -P.log().sum(-1) - K, l, P, 1e-6, 5.))
    multiple = 0
    for curve in curves:
        a0 = (c0 / b0) * torch.rand(n, dtype=f64, generator=generator)
        root, *_ = full_link._profiled_root(curve, a0, b0, c0)
        a = a0.clone()
        for _ in range(20000):
            a = c0 / (b0 + curve.evaluate(a)[0])
        grid = torch.linspace(1e-4, 1, 4001, dtype=f64).unsqueeze(-1) * (c0 / b0)
        g = grid - c0 / (b0 + curve.evaluate(grid)[0])
        multiple += int((((g[1:] * g[:-1]) < 0).sum(0) > 1).sum())
        converged = (c0 / (b0 + curve.evaluate(a)[0]) - a).abs() < 1e-11      # slow MM tails excluded
        assert float(((root - a).abs() / a.clamp(min=1e-3))[converged].max()) < 1e-8
    assert multiple > 0                                                  # the sweep has non-unique fixed points
