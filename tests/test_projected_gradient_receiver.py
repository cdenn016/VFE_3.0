"""Tiny CPU checks for the projected_gradient receiver law and updates (projected_gradient.tex)."""

from dataclasses import replace

from tests.profiles import make_relational_operators
from typing import Any

import pytest
import torch

from tests.profiles import test_config
from tests.test_full_link_gradients import _dense_frame, _fixture, _snapshot_alpha
from tests.test_projected_gradient import _cfg, _tokens
from vfe3.belief import BeliefState
from vfe3.config import config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.full_link import (
    _spd_precision, build_full_link_context, full_link_cost, full_link_cost_parts,
    full_link_gradients, full_link_minimizer,
)
from vfe3.inference.e_step import e_step
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import source_recovery_counter

NOISE, HEADS = .7, (.6, 1.3)


@pytest.fixture(autouse=True)
def _single_cpu_thread() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _context(full: bool = False, compact: bool = False) -> tuple:
    """The receiver context frozen at alpha*(snapshot), without the profile: fixed coefficients."""
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=full, compact=compact)
    context = build_full_link_context(
        mu, sigma, mu_p, sigma_p, frame, ops, tau=1.0, lambda_beta=torch.tensor(HEADS, dtype=mu.dtype),
        receiver_noise=NOISE, log_prior=prior)
    return mu, sigma, mu_p, sigma_p, frame, ops, replace(context, profile=None)


def _receiver(mu: torch.Tensor, sigma: torch.Tensor) -> tuple:
    """A receiver state away from the snapshot, with gradients enabled."""
    r_mu = (mu.detach() * .7 - .1).requires_grad_()
    r_sigma = (sigma.detach() * 1.3)
    if r_sigma.ndim == r_mu.ndim + 1:
        r_sigma = r_sigma + .05 * torch.eye(4, dtype=mu.dtype)
    return r_mu, r_sigma.requires_grad_()


def _literal_channel(r_mu, r_sigma, mu, sigma, mu_p, sigma_p, beta, frame, ops, alpha) -> torch.Tensor:
    r"""alpha_i KL(q_i||p_i) + sum w E_q[-log N(T_h xi_j; O_h^T xi, T_h C_j T_h^T + nu I_d)] (no constants)."""
    U = _dense_frame(frame)
    cov = lambda m, s: torch.diag_embed(s) if s.ndim == m.ndim else s
    rc, sc, pc = cov(r_mu, r_sigma), cov(mu, sigma), cov(mu_p, sigma_p)
    total = r_mu.new_zeros(())
    for i in range(3):
        q = torch.distributions.MultivariateNormal(r_mu[0, i], rc[0, i])
        p = torch.distributions.MultivariateNormal(mu_p[0, i], pc[0, i])
        total = total + alpha[0, i] * torch.distributions.kl_divergence(q, p)
        Ui = torch.linalg.inv(U[0, i])
        xi, C = Ui @ r_mu[0, i], Ui @ rc[0, i] @ Ui.T
        for h in range(2):
            P = ops.O[h].T
            for j in range(3):
                Uj = torch.linalg.inv(U[0, j])
                v = ops.T[h] @ Uj @ mu[0, j]
                R = ops.T[h] @ Uj @ sc[0, j] @ Uj.T @ ops.T[h].T + NOISE * torch.eye(2, dtype=mu.dtype)
                r = v - P @ xi
                total = total + HEADS[h] * beta[0, h, i, j] * .5 * (
                    r @ torch.linalg.solve(R, r) + torch.trace(torch.linalg.solve(R, P @ C @ P.T)))
    return total


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_channel_kernel_matches_literal_expected_likelihood(full: bool, compact: bool) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, ctx = _context(full, compact)
    beta, alpha = ctx.beta.detach(), ctx.c.detach()
    r_mu, r_sigma = _receiver(mu, sigma)
    literal = _literal_channel(r_mu, r_sigma, mu.detach(), sigma.detach(), mu_p, sigma_p, beta, frame, ops, alpha)
    g_mu, g_sigma = torch.autograd.grad(literal, (r_mu, r_sigma))
    a_mu, a_sigma = full_link_gradients(r_mu.detach(), r_sigma.detach(), ctx)
    if full:
        g_sigma = .5 * (g_sigma + g_sigma.transpose(-1, -2))
    torch.testing.assert_close(a_mu, g_mu, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(a_sigma, g_sigma, atol=1e-10, rtol=1e-9)
    # Cost differences equal literal differences (the cost omits receiver-independent constants).
    r2_mu, r2_sigma = mu.detach() + .3, sigma.detach() * .9
    literal2 = _literal_channel(r2_mu, r2_sigma, mu.detach(), sigma.detach(), mu_p, sigma_p, beta, frame, ops,
                                alpha)
    torch.testing.assert_close(
        full_link_cost(r_mu.detach(), r_sigma.detach(), ctx) - full_link_cost(r2_mu, r2_sigma, ctx),
        literal.detach() - literal2, atol=1e-9, rtol=1e-9)


def test_entropy_multiplicity_is_the_self_coupling() -> None:
    mu, sigma, mu_p, sigma_p, *_, channel = _context()
    torch.testing.assert_close(channel.c, _snapshot_alpha(mu, sigma, mu_p, sigma_p), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("full", [False, True])
def test_closed_form_is_stationary_and_damped_updates_descend(full: bool) -> None:
    mu, sigma, *_, ctx = _context(full)
    r_mu, r_sigma = (x.detach() for x in _receiver(mu, sigma))
    star_mu, star_sigma = full_link_minimizer(r_mu, r_sigma, ctx)
    g_mu, g_sigma = full_link_gradients(star_mu, star_sigma, ctx)
    assert float(g_mu.abs().max()) < 1e-10 and float(g_sigma.abs().max()) < 1e-10
    costs = [float(full_link_cost(r_mu, r_sigma, ctx))]
    for rho in (.25, .6, 1.0):
        new_mu, new_sigma = full_link_minimizer(r_mu, r_sigma, ctx, damping=rho)
        if full:
            assert float(torch.linalg.eigvalsh(new_sigma).min()) > 0
        costs.append(float(full_link_cost(new_mu, new_sigma, ctx)))
    assert all(a > b for a, b in zip(costs, costs[1:])), costs
    # Covariance kept: the mean moves along the segment to mu*.
    kept_mu, kept_sigma = full_link_minimizer(r_mu, r_sigma, ctx, damping=.5, update_sigma=False)
    torch.testing.assert_close(kept_mu, .5 * r_mu + .5 * star_mu)
    assert kept_sigma is r_sigma


def test_full_family_damping_is_the_exact_fisher_flow() -> None:
    mu, sigma, *_, ctx = _context(full=True)
    r_mu, r_sigma = (x.detach() for x in _receiver(mu, sigma))
    c = ctx.c[..., None, None]
    t = .37
    # c_i = alpha*_i differs per receiver, so the flow time t maps to one damping rho_i per receiver.
    rho = 1 - torch.exp(-ctx.c * t)                                     # (1, N)
    J0 = torch.linalg.inv(r_sigma)
    J = (1 - rho[..., None, None]) * J0 + rho[..., None, None] * ctx.Lambda / c
    steps_by_receiver = [full_link_minimizer(r_mu, r_sigma, ctx, damping=float(r)) for r in rho[0]]
    new_mu = torch.stack([step[0][0, n] for n, step in enumerate(steps_by_receiver)]).unsqueeze(0)
    new_sigma = torch.stack([step[1][0, n] for n, step in enumerate(steps_by_receiver)]).unsqueeze(0)
    torch.testing.assert_close(new_sigma, torch.linalg.inv(J), atol=1e-12, rtol=1e-10)
    # Integrate the flow d mu = -Sigma (Lambda mu - eta), d Sigma = c Sigma - Sigma Lambda Sigma by RK4.
    m, S, steps = r_mu.clone(), r_sigma.clone(), 4000
    h = t / steps
    f = lambda m, S: (-(S @ ((ctx.Lambda @ m.unsqueeze(-1)).squeeze(-1) - ctx.eta).unsqueeze(-1)).squeeze(-1),
                      c * S - S @ ctx.Lambda @ S)
    for _ in range(steps):
        k1 = f(m, S); k2 = f(m + h / 2 * k1[0], S + h / 2 * k1[1])
        k3 = f(m + h / 2 * k2[0], S + h / 2 * k2[1]); k4 = f(m + h * k3[0], S + h * k3[1])
        m = m + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        S = S + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    torch.testing.assert_close(new_mu, m, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(new_sigma, S, atol=1e-9, rtol=1e-8)


def test_channel_predictive_limit() -> None:
    """Tiled block-identity maps, equal source laws and a flat prior give the predictive head sum."""
    torch.manual_seed(3)
    mu = torch.randn(1, 3, 4, dtype=torch.float64)
    sigma = torch.full((1, 3, 4), .9, dtype=torch.float64)
    eye = torch.eye(2, dtype=mu.dtype).expand(1, 3, 2, 2, 2).clone()
    frame = CanonicalFrameContext(eye, eye, block_diagonal=True)
    ops = make_relational_operators([2, 2], 5, readout_mode="full_link_update").double()
    prior = torch.tensor([[0., -torch.inf, -torch.inf], [0., 0., -torch.inf], [0., 0., 0.]], dtype=mu.dtype)
    # c0 = 1e-12 makes alpha* = c0/(b0 + D) at most 1e-12: a vanishing self-coupling.
    ctx = build_full_link_context(mu, sigma, mu, sigma * 1e12, frame, ops, c0=1e-12, log_prior=prior)
    assert float(ctx.c.max()) <= 1e-12
    star_mu, _ = full_link_minimizer(mu, sigma, ctx)
    predictive = ops.predictive_content(mu, ctx.beta, frame)
    torch.testing.assert_close(star_mu, predictive, atol=1e-8, rtol=1e-8)


def test_channel_leaves_unobserved_directions_at_the_prior() -> None:
    """With O_h spanning only block 0 for every head, block 1 keeps the prior mean (no zero pinning)."""
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture()
    eye = torch.eye(2, dtype=mu.dtype).expand(1, 3, 2, 2, 2).clone()
    frame = CanonicalFrameContext(eye, eye, block_diagonal=True)
    with torch.no_grad():
        ops.O.zero_()
        ops.O[:, :2, :] = torch.eye(2, dtype=mu.dtype)
    ctx = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, receiver_noise=NOISE, log_prior=prior)
    star_mu, _ = full_link_minimizer(mu.detach(), sigma.detach(), ctx)
    torch.testing.assert_close(star_mu[..., 2:], mu_p[..., 2:], atol=1e-10, rtol=0)


def _episode(update: str, full: bool = False, **kwargs: Any) -> tuple:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=full)
    ops.readout_mode = "full_link_update"
    ops.receiver_update = update
    belief = BeliefState(mu.detach(), sigma.detach(), torch.zeros(1, 3, 2, dtype=mu.dtype))
    capture: dict = {}
    defaults = dict(e_q_mu_lr=.05, e_q_sigma_lr=.05, e_mu_q_trust=None, e_sigma_q_trust=50.0,
                    sigma_max=None, n_iter=1, lambda_beta=1.0)
    defaults.update(kwargs)
    out = e_step(belief, mu_p, sigma_p, None, log_prior=prior, relational_operators=ops,
                 relational_frame=frame, relational_capture=capture,
                 family="gaussian_full" if full else "gaussian_diagonal", **defaults)
    return belief, out, capture


@pytest.mark.parametrize("full", [False, True])
def test_euler_mean_step_follows_the_fisher_formula_and_descends(full: bool) -> None:
    belief, out, capture = _episode("euler", full, skip_belief_sigma_update=True)
    ctx = capture["full_link_context"]
    g_mu, _ = full_link_gradients(belief.mu, belief.sigma, ctx)
    nat = (belief.sigma @ g_mu.unsqueeze(-1)).squeeze(-1) if full else belief.sigma * g_mu
    torch.testing.assert_close(out.mu, belief.mu - .05 * nat)
    assert float(capture["relational_receiver_cost_after"]) < float(capture["relational_receiver_cost_before"])


@pytest.mark.parametrize("full", [False, True])
def test_euler_covariance_step_descends(full: bool) -> None:
    belief, out, capture = _episode("euler", full, e_q_mu_lr=0.0)
    torch.testing.assert_close(out.mu, belief.mu)
    assert float(capture["relational_receiver_covariance_cost_after"]) < float(
        capture["relational_receiver_covariance_cost_before"])


def test_closed_form_episode_reaches_the_minimizer_and_reports_it() -> None:
    # The reports read the context frozen at the episode-initial alpha; one undamped step reaches its minimizer.
    belief, out, capture = _episode("closed_form", n_iter=1)
    assert float(capture["relational_receiver_target_distance_after"]) < 1e-9
    assert float(capture["relational_receiver_target_distance_before"]) > 1e-3
    torch.testing.assert_close(capture["relational_mean_displacement_norm"], (out.mu - belief.mu).norm())
    assert int(capture["relational_covariance_repair_count"]) == 0


def test_spd_precision_retries_float32_failures_in_float64() -> None:
    generator = torch.Generator().manual_seed(1)
    B = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    target = (B @ B.T + 1e-7 * torch.eye(6, dtype=torch.float64)).float()
    assert int(torch.linalg.cholesky_ex(target)[1]) != 0, "precondition: float32 factorization fails"
    from vfe3.gradients.full_link import receiver_law_retry_counter
    before = receiver_law_retry_counter(target.device)
    recovery_before = source_recovery_counter(target.device)
    precision = _spd_precision(target)
    # Receiving-law retries have their own counter; the projected-covariance recovery count is untouched.
    assert int(receiver_law_retry_counter(target.device) - before) == 1
    assert int(source_recovery_counter(target.device) - recovery_before) == 0
    assert precision.dtype == torch.float32 and bool(torch.isfinite(precision).all())
    torch.testing.assert_close(precision.double(), torch.linalg.inv(target.double()).float().double(),
                               rtol=1e-4, atol=1e-2)


def test_full_link_gradients_never_return_half_precision() -> None:
    mu, sigma, *_, ctx = _context()
    g_mu, g_sigma = full_link_gradients(mu.detach().bfloat16(), sigma.detach().float(), ctx)
    assert g_mu.dtype == torch.float32 and g_sigma.dtype == torch.float32
    ctx32 = type(ctx)(*(x.float() for x in (ctx.Lambda, ctx.eta, ctx.c, ctx.beta, ctx.score)))
    g_mu, _ = full_link_gradients(mu.detach().bfloat16(), sigma.detach().float(), ctx32)
    assert g_mu.dtype == torch.float32


def test_cost_parts_sum_to_cost_up_to_a_constant() -> None:
    mu, sigma, *_, ctx = _context(full=True)
    states = [(mu.detach(), sigma.detach()), (mu.detach() * .5, sigma.detach() * 1.4)]
    sums = [sum(full_link_cost_parts(m, s, ctx)[:2]) for m, s in states]
    costs = [full_link_cost(m, s, ctx) for m, s in states]
    torch.testing.assert_close(sums[0] - sums[1], costs[0] - costs[1])


# Configuration and model integration.

def test_config_rejects_zero_e_steps_and_invalid_receiver_fields() -> None:
    with pytest.raises(ValueError, match="n_e_steps >= 1"):
        _cfg(n_e_steps=0)
    with pytest.raises(ValueError, match="relational_receiver_update"):
        _cfg(relational_receiver_update="other")
    for rho in (0.0, 1.5, float("nan"), True):
        with pytest.raises(ValueError, match="relational_receiver_damping"):
            _cfg(relational_receiver_update="closed_form", relational_receiver_damping=rho)


def test_inactive_fields_follow_the_receiver_update() -> None:
    closed = _cfg(relational_receiver_update="closed_form")
    inactive = closed.inactive_fields()
    assert {"e_q_mu_lr", "e_q_sigma_lr", "e_mu_q_trust", "e_sigma_q_trust"} <= set(inactive)
    assert "relational_receiver_damping" not in inactive
    euler = _cfg(relational_receiver_update="euler").inactive_fields()
    assert "relational_receiver_damping" in euler and "e_q_mu_lr" not in euler
    predictive = test_config(attention_mode="projected_predictive").inactive_fields()
    assert {"relational_receiver_update", "relational_receiver_damping"} <= set(predictive)


def test_serialized_configs_without_receiver_fields_keep_the_historical_receiver() -> None:
    # A projected_gradient config without the receiver fields ran ambient_alignment, which no longer exists.
    payload = dict(serialize_config(_cfg(relational_receiver_update="closed_form")))
    for name in ("relational_receiver_update", "relational_receiver_damping"):
        payload.pop(name)
    with pytest.raises(ValueError, match="ambient_alignment"):
        config_from_serialized(payload, source="test")
    # On the predictive route the law was inert: the historical Euler update loads.
    payload = dict(serialize_config(test_config(attention_mode="projected_predictive")))
    for name in ("relational_receiver_update", "relational_receiver_damping"):
        payload.pop(name)
    assert config_from_serialized(payload, source="test").relational_receiver_update == "euler"
    round_trip = config_from_serialized(serialize_config(_cfg(relational_receiver_update="closed_form",
                                                              relational_receiver_damping=.4)),
                                        source="test")
    assert (round_trip.relational_receiver_update, round_trip.relational_receiver_damping) == ("closed_form", .4)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_model_channel_closed_form_trains_and_reports(family: str) -> None:
    cfg = _cfg(family=family, n_e_steps=2, relational_receiver_update="closed_form_profiled")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    ids, targets = _tokens()
    _, loss, _ = model(ids, targets)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if name.startswith("relational_operators"):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert float(parameter.grad.abs().max()) > 0, name
    snapshot = model.build_diagnostic_snapshot(ids)
    states = snapshot.trace_states
    torch.testing.assert_close(states[-1].mu, states[-2].mu, atol=1e-6, rtol=1e-6)  # rho=1 is idempotent
    d = model.diagnostics(ids)
    for key in ("relational_receiver_cost_delta", "relational_receiver_mean_cost_delta",
                "relational_receiver_covariance_cost_delta", "relational_receiver_target_distance_ratio"):
        assert key in d and d[key] == d[key], key
    # The target distance reads the context frozen at the episode-initial alpha, whose minimizer one
    # undamped closed_form step reaches.
    torch.manual_seed(0)
    one_step = VFEModel(_cfg(family=family, n_e_steps=1, relational_receiver_update="closed_form"))
    assert one_step.diagnostics(ids)["relational_receiver_target_distance_ratio"] < 1e-4


# Coverage restored after PR #32 (audit 2026-10-01 F8).

def test_euler_mean_trust_region_binds_in_whitened_units() -> None:
    belief, free, _ = _episode("euler", e_q_mu_lr=5.0, skip_belief_sigma_update=True)
    _, clipped, _ = _episode("euler", e_q_mu_lr=5.0, skip_belief_sigma_update=True, e_mu_q_trust=.05)
    # Mahalanobis ball: each row's whitened step has 2-norm at most the trust radius.
    whitened = (clipped.mu - belief.mu) / belief.sigma.sqrt()
    assert float(whitened.norm(dim=-1).max()) <= .05 + 1e-12
    assert float(((free.mu - belief.mu) / belief.sigma.sqrt()).norm(dim=-1).max()) > .05


def test_euler_zero_rates_freeze_their_components() -> None:
    belief, out, _ = _episode("euler", e_q_mu_lr=.1, e_q_sigma_lr=0.0)
    torch.testing.assert_close(out.sigma, belief.sigma)
    assert not torch.equal(out.mu, belief.mu)


@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_diagnostics_run_under_rope(mode: str) -> None:
    overrides = dict(rope_mode="frame_value")
    if mode == "projected_predictive":
        cfg = test_config(attention_mode=mode, **overrides)
    else:
        cfg = _cfg(relational_receiver_update="closed_form", **overrides)
    torch.manual_seed(0)
    d = VFEModel(cfg).diagnostics(_tokens()[0])
    assert d["attn_entropy"] == d["attn_entropy"] and d["effective_rank"] == d["effective_rank"]


def test_timer_wall_clock_continues_from_a_resumed_offset() -> None:
    from vfe3.timing import TrainingTimer
    ticks = iter([10.0, 13.0, 13.0])
    timer = TrainingTimer(torch.device("cpu"), clock=lambda: next(ticks), wall_clock_offset=100.0)
    assert timer.sample_pipeline_window().wall_clock_s == 103.0
    assert timer.elapsed_wall_clock_s() == 103.0
