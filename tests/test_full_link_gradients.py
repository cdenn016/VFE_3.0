"""Tiny CPU checks against the literal channel-likelihood receiving law."""

from dataclasses import replace

from tests.profiles import make_relational_operators
import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.full_link import (
    build_full_link_context, full_link_context_at, full_link_cost, full_link_gradients,
)
from vfe3.model.relational_operators import RelationalOperators


def _fixture(full: bool = False, compact: bool = False) -> tuple:
    torch.set_num_threads(1)
    mu = torch.tensor([[[.2, -.4, .1, .7], [1.1, .5, -.3, .8],
                        [.6, 1.2, .9, -.2]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([[[.8, 1.2, .9, 1.4], [1.1, .6, 1.5, .7],
                           [.7, 1.3, .8, 1.1]]], dtype=mu.dtype)
    if full:
        sigma = torch.diag_embed(sigma)
        sigma[..., 0, 2] = sigma[..., 2, 0] = .12
        sigma[..., 1, 3] = sigma[..., 3, 1] = -.08
    sigma.requires_grad_()
    U = torch.eye(4, dtype=mu.dtype).expand(1, 3, 4, 4).clone()
    U[0, :, 0, 1] = torch.tensor([.3, -.1, .15])
    U[0, :, 2, 3] = torch.tensor([-.2, .12, -.08])
    U[0, :, 0, 0] = torch.tensor([1.1, .9, 1.2])
    V = torch.linalg.inv(U)
    if compact:
        U = torch.stack((U[..., :2, :2], U[..., 2:, 2:]), dim=-3)
        V = torch.stack((V[..., :2, :2], V[..., 2:, 2:]), dim=-3)
    frame = CanonicalFrameContext(U, V, block_diagonal=compact)
    ops = make_relational_operators(
        [2, 2], 11,
        readout_mode="predictive_content", predictive_covariance="passthrough").double()
    with torch.no_grad():
        ops.T.add_(torch.tensor([[[.02, -.03, .21, .11], [.04, .01, -.14, .16]],
                                [[-.13, .18, .03, -.02], [.12, -.17, .04, .02]]]))
        ops.O.add_(torch.tensor([[[.01, .03], [-.02, .01], [.23, -.17], [.14, .19]],
                                [[-.12, .22], [.18, .13], [.02, -.03], [.01, .04]]]))
    prior = torch.tensor([[0., -torch.inf, -torch.inf], [0., 0., -torch.inf],
                          [0., 0., 0.]], dtype=mu.dtype)
    return mu, sigma, mu.detach() + .2, sigma.detach() * 1.2, frame, ops, prior


def _dense_frame(frame: CanonicalFrameContext) -> torch.Tensor:
    if not frame.block_diagonal:
        return frame.forward
    return torch.stack([
        torch.block_diag(*frame.forward[0, n].unbind(0))
        for n in range(frame.forward.shape[1])]).unsqueeze(0)


def _literal_cost(
    receiver_mu:    torch.Tensor,
    receiver_sigma: torch.Tensor,
    source_mu:      torch.Tensor,
    source_sigma:   torch.Tensor,
    mu_p:           torch.Tensor,
    sigma_p:        torch.Tensor,
    beta:           torch.Tensor,
    frame:          CanonicalFrameContext,
    ops:            RelationalOperators,
    alpha:          torch.Tensor,           # (1, N) self-coupling per receiver
    noise:          float = .7,
    head_weights:   tuple = (.6, 1.3),
) -> torch.Tensor:
    r"""alpha_i KL(q_i||p_i) + sum w E_q[-log N(T_h xi_j; O_h^T xi_i, T_h C_j T_h^T + nu I_d)] (no constants).

    The affine-chart adjoint measurement P_h = O_h^T of the reference-coordinate receiver xi_i = U_i^{-1} mu_i.
    """
    U = _dense_frame(frame)
    source_cov = torch.diag_embed(source_sigma) if source_sigma.ndim == source_mu.ndim else source_sigma
    receiver_cov = torch.diag_embed(receiver_sigma) if receiver_sigma.ndim == receiver_mu.ndim else receiver_sigma
    prior_cov = torch.diag_embed(sigma_p) if sigma_p.ndim == mu_p.ndim else sigma_p
    d = ops.T.shape[1]
    result = receiver_mu.new_zeros(())
    for i in range(3):
        q = torch.distributions.MultivariateNormal(receiver_mu[0, i], receiver_cov[0, i])
        p = torch.distributions.MultivariateNormal(mu_p[0, i], prior_cov[0, i])
        result = result + alpha[0, i] * torch.distributions.kl_divergence(q, p)
        Ui = torch.linalg.inv(U[0, i])
        xi, C = Ui @ receiver_mu[0, i], Ui @ receiver_cov[0, i] @ Ui.T
        for h in range(2):
            P = ops.O[h].T
            for j in range(3):
                Uj = torch.linalg.inv(U[0, j])
                v = ops.T[h] @ Uj @ source_mu[0, j]
                R = ops.T[h] @ Uj @ source_cov[0, j] @ Uj.T @ ops.T[h].T + noise * torch.eye(d, dtype=v.dtype)
                r = v - P @ xi
                result = result + head_weights[h] * beta[0, h, i, j] * .5 * (
                    r @ torch.linalg.solve(R, r) + torch.trace(torch.linalg.solve(R, P @ C @ P.T)))
    return result


def _snapshot_alpha(mu: torch.Tensor, sigma: torch.Tensor, mu_p: torch.Tensor, sigma_p: torch.Tensor,
                    b0: float = 1., c0: float = 1.) -> torch.Tensor:
    r"""alpha*_i = c0/(b0 + KL(q_i||p_i)) of the snapshot, computed independently of the kernel; (1, N)."""
    cov = lambda m, s: torch.diag_embed(s) if s.ndim == m.ndim else s
    q = torch.distributions.MultivariateNormal(mu.detach(), cov(mu, sigma.detach()))
    p = torch.distributions.MultivariateNormal(mu_p, cov(mu_p, sigma_p))
    return c0 / (b0 + torch.distributions.kl_divergence(q, p))


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_receiver_derivatives_match_literal_gaussian_energy(full: bool, compact: bool) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full, compact)
    # The context frozen at alpha*(snapshot), without the profile: fixed receiving-law coefficients.
    context = replace(build_full_link_context(
        mu, sigma, mu_p, sigma_p, frame, ops, tau=.9, receiver_noise=.7,
        lambda_beta=torch.tensor([.6, 1.3], dtype=mu.dtype), log_prior=prior), profile=None)
    alpha = _snapshot_alpha(mu, sigma, mu_p, sigma_p)
    torch.testing.assert_close(context.c.detach(), alpha, atol=1e-13, rtol=1e-13)
    probe_mu = (mu.detach() + .17).requires_grad_()
    probe_sigma = (sigma.detach() * 1.07).requires_grad_()
    literal = _literal_cost(probe_mu, probe_sigma, mu, sigma, mu_p, sigma_p, context.beta, frame, ops, alpha)
    expected = torch.autograd.grad(literal, (probe_mu, probe_sigma))
    actual = full_link_gradients(probe_mu, probe_sigma, context)
    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-11, rtol=3e-11)
    cost_gradients = torch.autograd.grad(full_link_cost(probe_mu, probe_sigma, context), (probe_mu, probe_sigma))
    for actual_gradient, cost_gradient in zip(actual, cost_gradients):
        torch.testing.assert_close(actual_gradient, cost_gradient, atol=3e-11, rtol=3e-11)
    # Receiver-independent normalizers are intentionally omitted from full_link_cost.
    next_mu, next_sigma = probe_mu + .03, probe_sigma * .97
    expected_delta = _literal_cost(next_mu, next_sigma, mu, sigma, mu_p, sigma_p, context.beta, frame, ops,
                                   alpha) - literal
    actual_delta = full_link_cost(next_mu, next_sigma, context) - full_link_cost(probe_mu, probe_sigma, context)
    torch.testing.assert_close(actual_delta, expected_delta, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("full", [False, True])
def test_receiver_derivatives_finite_difference(full: bool) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full)
    context = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior)
    probe_mu, probe_sigma = mu.detach() + .12, sigma.detach() * .95
    # alpha frozen at the probe: the profiled-energy gradient alpha*(q) dD/dq + dG/dq (envelope theorem).
    gradients = full_link_gradients(probe_mu, probe_sigma, full_link_context_at(context, probe_mu, probe_sigma))
    direction_mu = torch.arange(12, dtype=mu.dtype).reshape_as(mu) / 12 - .4
    direction_sigma = torch.ones_like(sigma) * .07
    if full:
        direction_sigma[..., 0, 2] = direction_sigma[..., 2, 0] = -.09
    step = 1e-6
    for which, direction in enumerate((direction_mu, direction_sigma)):
        args_plus, args_minus = [probe_mu, probe_sigma], [probe_mu, probe_sigma]
        args_plus[which] = args_plus[which] + step * direction
        args_minus[which] = args_minus[which] - step * direction
        delta = (full_link_cost(*args_plus, context) - full_link_cost(*args_minus, context)) / (2 * step)
        torch.testing.assert_close((gradients[which] * direction).sum(), delta, atol=4e-8, rtol=4e-8)


def test_source_factorized_pooling_matches_literal_pair_precision() -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=True, compact=True)
    weights = torch.tensor([.6, 1.3], dtype=mu.dtype)
    context = build_full_link_context(
        mu, sigma, mu_p, sigma_p, frame, ops, receiver_noise=.7, lambda_beta=weights, log_prior=prior)
    alpha = _snapshot_alpha(mu, sigma, mu_p, sigma_p)
    U = _dense_frame(frame)
    identity2 = torch.eye(2, dtype=mu.dtype)
    Lambda, eta = torch.empty_like(context.Lambda), torch.empty_like(context.eta)
    for i in range(3):
        precision = torch.linalg.inv(sigma_p[0, i])
        Lambda[0, i] = alpha[0, i] * precision
        eta[0, i] = alpha[0, i] * precision @ mu_p[0, i]
        Ui = torch.linalg.inv(U[0, i])
        for h in range(2):
            for j in range(3):
                # The d x d pair precision R_hj^{-1} of the message T_h xi_j, lifted by P_h = O_h^T.
                Uj = torch.linalg.inv(U[0, j])
                R = ops.T[h] @ Uj @ sigma[0, j] @ Uj.T @ ops.T[h].T + .7 * identity2
                pair = torch.linalg.solve(R, identity2)
                lift = ops.O[h].T @ Ui                                          # (d, K)
                weight = weights[h] * context.beta[0, h, i, j]
                Lambda[0, i] += weight * lift.T @ pair @ lift
                eta[0, i] += weight * lift.T @ pair @ (ops.T[h] @ Uj @ mu[0, j])
    torch.testing.assert_close(context.Lambda, Lambda, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(context.eta, eta, atol=3e-11, rtol=3e-11)
    # The channel law counts the receiver entropy once: c = alpha.
    torch.testing.assert_close(context.c, alpha, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("full", [False, True])
def test_rank_deficient_content_is_finite_with_positive_receiving_noise(full: bool) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full, compact=True)
    with torch.no_grad():
        ops.T[0].zero_()
        ops.O[1, :, 1].copy_(ops.O[1, :, 0])
    context = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, receiver_noise=.4, log_prior=prior)
    gradients = full_link_gradients(mu, sigma, context)
    assert torch.linalg.eigvalsh(context.Lambda).min() > 0
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("parameter_name,index", [
    ("raw_score_q", (0, 0, 2)), ("raw_score_k", (0, 0, 2)),
    ("T", (0, 0, 2)), ("O", (0, 2, 0)),
])
def test_outer_map_credit_matches_finite_difference(parameter_name: str, index: tuple) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture()
    parameter = getattr(ops, parameter_name)

    def objective() -> torch.Tensor:
        context = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior)
        grad_mu, grad_sigma = full_link_gradients(mu, sigma, context)
        updated = mu - .07 * sigma * grad_mu
        return updated[0, 2].square().sum() + .02 * grad_sigma[0, 2].square().sum()

    analytic, = torch.autograd.grad(objective(), parameter)
    assert torch.isfinite(analytic).all() and analytic.abs().sum() > 1e-8
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
    torch.testing.assert_close(analytic[index], (plus - minus) / 2e-6, atol=3e-8, rtol=3e-5)


def test_source_and_router_graphs_remain_live_with_off_head_content() -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture()
    capture = {}
    context = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior, capture=capture)
    grad_mu, grad_sigma = full_link_gradients(mu.detach() + .1, sigma.detach(), context)
    credit = torch.autograd.grad(grad_mu[0, 2].square().sum() + grad_sigma[0, 2].square().sum(), (mu, sigma))
    assert all(torch.isfinite(gradient).all() and gradient[0, :2].abs().sum() > 1e-8 for gradient in credit)
    assert context.beta.requires_grad and context.score.requires_grad
    assert not capture["beta_pre"].requires_grad and not capture["score_energy_pre"].requires_grad
    # Head zero writes receiver coordinates 2/3 through O, beyond its original block.
    assert context.eta[0, 2, 2:].abs().sum() > 1e-8


@pytest.mark.parametrize("full", [False, True])
def test_causal_future_sources_leave_earlier_receiver_updates_unchanged(full: bool) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full)
    first = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior)
    changed_mu, changed_sigma = mu.detach().clone(), sigma.detach().clone()
    changed_mu[0, 2] += 8.
    changed_sigma[0, 2] *= 3.
    second = build_full_link_context(changed_mu, changed_sigma, mu_p, sigma_p, frame, ops, log_prior=prior)
    for first_value, second_value in ((first.Lambda, second.Lambda), (first.eta, second.eta), (first.c, second.c)):
        torch.testing.assert_close(first_value[0, :2], second_value[0, :2], atol=0, rtol=0)
    first_gradients = full_link_gradients(mu, sigma, first)
    second_gradients = full_link_gradients(changed_mu, changed_sigma, second)
    for first_gradient, second_gradient in zip(first_gradients, second_gradients):
        torch.testing.assert_close(first_gradient[0, :2], second_gradient[0, :2], atol=0, rtol=0)


def test_separate_value_frame_changes_channels_without_changing_router() -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(compact=True)
    identity = torch.eye(2, dtype=mu.dtype).expand_as(frame.forward).clone()
    value_frame = CanonicalFrameContext(identity, identity, block_diagonal=True)
    first = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior)
    second = build_full_link_context(
        mu, sigma, mu_p, sigma_p, frame, ops, log_prior=prior, value_frame=value_frame)
    torch.testing.assert_close(first.beta, second.beta, atol=0, rtol=0)
    assert not torch.allclose(first.Lambda, second.Lambda)
    # Independent receiver probes define filtering partials while source graphs stay live.
    probe = mu.detach().clone().requires_grad_()
    literal = _literal_cost(probe, sigma.detach(), mu, sigma, mu_p, sigma_p, second.beta, value_frame, ops,
                            second.c.detach(), noise=1., head_weights=(1., 1.))
    expected, = torch.autograd.grad(literal, probe)
    torch.testing.assert_close(full_link_gradients(mu, sigma, second)[0], expected, atol=3e-11, rtol=3e-11)


@pytest.mark.parametrize("weight", [-.1, float("nan"), torch.tensor([1., -1.]), torch.ones(3)])
def test_invalid_coupling_weights_are_rejected(weight: object) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture()
    with pytest.raises(ValueError, match="lambda_beta"):
        build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, lambda_beta=weight, log_prior=prior)


@pytest.mark.parametrize("noise", [0., -.1, float("nan")])
def test_nonpositive_or_nonfinite_receiving_noise_is_rejected(noise: float) -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture()
    with pytest.raises(ValueError, match="receiver_noise"):
        build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, receiver_noise=noise, log_prior=prior)


def test_autocast_is_disabled_and_escalated_frame_dtype_is_respected() -> None:
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(compact=True)
    ops.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = build_full_link_context(
            mu.float(), sigma.float(), mu_p.float(), sigma_p.float(), frame, ops, log_prior=prior)
        gradients = full_link_gradients(mu.float(), sigma.float(), context)
        energy = full_link_cost(mu.float(), sigma.float(), context)
    assert context.Lambda.dtype == torch.float64
    assert all(gradient.dtype == torch.float32 and torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(energy)
