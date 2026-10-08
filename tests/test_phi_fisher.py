r"""Belief-Fisher natural gradient for the stored frames: ``m_phi_update_mode='fisher_group'``.

Derivation 2026-09-28, Track 2. Tiny float64 CPU checks of the completed frame Fisher (against KL
Hessians and the Gaussian Fisher pullback), its spectrum, the closed-form type solve, the pooled
positional operator, the symmetry and split-redundancy properties of the implemented direction, a
tiny VFEModel against a dense chart-coordinate route, and the optimizer, training-step and config
wiring under both retractions.
"""

from tests.profiles import test_config

import math
import warnings

import pytest
import torch
from torch import nn
from torch.autograd.functional import hessian, jacobian

from vfe3.gauge_optim import (
    FISHER_DIAGNOSTIC_KEYS,
    GaugeManifoldAdamW,
    belief_fisher_group_direction,
    get_phi_update_policy,
)
from vfe3.geometry.groups import get_group
from vfe3.geometry.matrix_log import logm_principal
from vfe3.geometry.phi_fisher import (
    FisherRowStatistics,
    completed_fisher_operator,
    fisher_spectrum_bounds,
    fisher_type_solve,
    jacobi_cholesky_solve,
    middle_fiber_moments,
    pooled_pos_operator,
)
from vfe3.geometry.phi_preconditioner import uspace_pullback_body_direction

F64 = torch.float64


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _spd(d: int, seed: int, spread: float = 1.0) -> torch.Tensor:
    A = torch.randn(d, d, generator=_generator(seed), dtype=F64)
    return A @ A.T * spread + 0.5 * torch.eye(d, dtype=F64)


def _dense(op, d: int) -> torch.Tensor:
    r"""Row-major matrix of a linear map on ``d x d`` matrices: column ``k`` is ``vec op(E_k)``."""
    basis = torch.eye(d * d, dtype=F64).reshape(d * d, d, d)
    return torch.stack([op(E).reshape(-1) for E in basis], dim=1)


def _operator_matrix(xi: torch.Tensor, C: torch.Tensor, rotation_weight: float) -> torch.Tensor:
    return _dense(lambda X: completed_fisher_operator(
        X, xi, C, rotation_weight=rotation_weight), xi.shape[-1])


def _kl(m0, S0, m1, S1) -> torch.Tensor:
    d = m0.shape[-1]
    S1_inv = torch.linalg.inv(S1)
    diff = m1 - m0
    return 0.5 * (torch.trace(S1_inv @ S0) + diff @ S1_inv @ diff - d
                  + torch.logdet(S1) - torch.logdet(S0))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


# 1
@pytest.mark.parametrize("d", [3, 4])
def test_frame_fisher_equals_kl_hessian_and_gaussian_fisher_pullback(d: int) -> None:
    xi = torch.randn(d, generator=_generator(d), dtype=F64)
    C = _spd(d, seed=10 + d)

    def moments(x: torch.Tensor):
        E_inv = torch.linalg.matrix_exp(-x.reshape(d, d))        # U -> U e^X moves r to e^{-X} r
        return E_inv @ xi, E_inv @ C @ E_inv.T

    x0 = torch.zeros(d * d, dtype=F64)
    kl_hessian = hessian(lambda x: _kl(xi, C, *moments(x)), x0)
    J_m = jacobian(lambda x: moments(x)[0], x0)                   # (d, d^2)
    J_S = jacobian(lambda x: moments(x)[1].reshape(-1), x0)       # (d^2, d^2)
    C_inv = torch.linalg.inv(C)
    pullback = J_m.T @ C_inv @ J_m + 0.5 * J_S.T @ torch.kron(C_inv, C_inv) @ J_S
    undamped = _operator_matrix(xi, C, rotation_weight=0.0)       # c1 = c2 = 1: the bare Fisher
    assert _rel(undamped, kl_hessian) <= 1e-12
    assert _rel(undamped, pullback) <= 1e-12


# 2
@pytest.mark.parametrize("rotation_weight", [0.5, 2.0, 3.0])
def test_generalized_spectrum_matches_the_formula(rotation_weight: float) -> None:
    d = 4
    xi = torch.randn(d, generator=_generator(21), dtype=F64)
    C = _spd(d, seed=22)
    M = _operator_matrix(xi, C, rotation_weight)
    P = _dense(lambda X: torch.linalg.solve(C, X @ C), d)
    eigenvalues = torch.linalg.eigvals(torch.linalg.solve(P, M)).real.sort().values
    a = xi @ torch.linalg.solve(C, xi)
    c1, c2 = 1 + rotation_weight / 2, 1 - rotation_weight / 2
    root = math.sqrt(float(a) ** 2 + 4 * c2 * c2)
    formula = ([2 + float(a)] + [(2 * c1 + float(a) + root) / 2] * (d - 1)
               + [(2 * c1 + float(a) - root) / 2] * (d - 1) + [2.0] * (d * (d - 1) // 2)
               + [rotation_weight] * ((d - 1) * (d - 2) // 2))
    expected = torch.tensor(sorted(formula), dtype=F64)
    assert float((eigenvalues - expected).abs().max()) <= 1e-10
    lam_min, lam_max = fisher_spectrum_bounds(a, d, rotation_weight=rotation_weight)
    assert abs(float(lam_min) - float(expected[0])) <= 1e-12
    assert abs(float(lam_max) - float(expected[-1])) <= 1e-12


# 3
@pytest.mark.parametrize("d", [3, 4, 5, 6])
@pytest.mark.parametrize("mean_scale", [0.0, 1e-4, 1.5])
def test_type_solve_matches_dense_solve_and_the_kronecker_form(d: int, mean_scale: float) -> None:
    rows, H = 2, 2
    gen = _generator(100 + d)
    G_W = torch.randn(rows, H, d, d, generator=gen, dtype=F64)
    mu = mean_scale * torch.randn(rows, H, d, generator=gen, dtype=F64)
    s = torch.exp(torch.randn(rows, H, d, generator=gen, dtype=F64))
    scale = torch.tensor([3.0, 0.5], dtype=F64)
    for rotation_weight in (0.5, 2.0, 3.0):
        W = fisher_type_solve(G_W, mu, s, scale, rotation_weight=rotation_weight)
        for r in range(rows):
            for h in range(H):
                M = _operator_matrix(mu[r, h], torch.diag(s[r, h]), rotation_weight)
                dense = torch.linalg.solve(M, scale[r] * G_W[r, h].reshape(-1)).reshape(d, d)
                assert _rel(W[r, h], dense) <= 1e-10
    W2 = fisher_type_solve(G_W, mu, s, scale, rotation_weight=2.0)
    S = torch.diag_embed(s)
    kronecker = scale[:, None, None, None] * S @ G_W @ torch.linalg.inv(
        2 * S + mu[..., :, None] * mu[..., None, :])
    assert _rel(W2, kronecker) <= 1e-12
    near = fisher_type_solve(G_W, mu, s, scale, rotation_weight=2.0 + 1e-9)   # general branch
    assert _rel(near, W2) <= 1e-8


# 4
def test_spatial_solve_at_the_local_row_is_the_body_solve_at_the_middle_fiber() -> None:
    d = 4
    gen = _generator(41)
    X = 0.4 * torch.randn(1, 1, d, d, generator=gen, dtype=F64)
    mu = torch.randn(1, 1, d, generator=gen, dtype=F64)
    s = torch.exp(0.5 * torch.randn(1, 1, d, generator=gen, dtype=F64))
    G_W = torch.randn(1, 1, d, d, generator=gen, dtype=F64)
    W = fisher_type_solve(G_W, mu, s, torch.ones(1, dtype=F64), rotation_weight=2.0)[0, 0]
    E = torch.linalg.matrix_exp(X[0, 0])
    xi0, C0, _ = middle_fiber_moments(X, mu, s)
    body = torch.linalg.solve(E, W @ E)                            # Ad_{E^{-1}} W
    G_X = E.T @ G_W[0, 0] @ torch.linalg.inv(E).T                  # body covector
    applied = completed_fisher_operator(body, xi0[0, 0], C0[0, 0], rotation_weight=2.0)
    assert _rel(applied, G_X) <= 1e-12


def _body_direction(mu, Sigma, U, G_X, rotation_weight):
    r"""Dense natural body direction at the reference belief of ``(mu, Sigma, U)``."""
    U_inv = torch.linalg.inv(U)
    M = _operator_matrix(U_inv @ mu, U_inv @ Sigma @ U_inv.T, rotation_weight)
    return torch.linalg.solve(M, G_X.reshape(-1)).reshape(U.shape)


# 5
def test_direction_is_left_gauge_invariant_and_right_gl_equivariant() -> None:
    d = 3
    gen = _generator(51)
    mu = torch.randn(d, generator=gen, dtype=F64)
    Sigma = _spd(d, seed=52)
    U = torch.linalg.matrix_exp(0.5 * torch.randn(d, d, generator=gen, dtype=F64))
    G_X = torch.randn(d, d, generator=gen, dtype=F64)
    g = torch.linalg.matrix_exp(0.6 * torch.randn(d, d, generator=gen, dtype=F64))
    h = torch.linalg.matrix_exp(0.6 * torch.randn(d, d, generator=gen, dtype=F64))
    for rotation_weight in (2.0, 0.7):
        X = _body_direction(mu, Sigma, U, G_X, rotation_weight)
        left = _body_direction(g @ mu, g @ Sigma @ g.T, g @ U, G_X, rotation_weight)
        right = _body_direction(mu, Sigma, U @ h, h.T @ G_X @ torch.linalg.inv(h).T, rotation_weight)
        assert _rel(left, X) <= 1e-10
        assert _rel(right, torch.linalg.solve(h, X @ h)) <= 1e-10


# 6
def test_type_metric_does_not_depend_on_the_position_factor() -> None:
    d = 3
    gen = _generator(61)
    mu = torch.randn(d, generator=gen, dtype=F64)
    S = torch.diag(torch.exp(torch.randn(d, generator=gen, dtype=F64)))
    E = torch.linalg.matrix_exp(0.5 * torch.randn(d, d, generator=gen, dtype=F64))
    Y = torch.randn(d, d, generator=gen, dtype=F64)                   # E -> E e^Y
    E_inv = torch.linalg.inv(E)
    middle = (Y * completed_fisher_operator(Y, E_inv @ mu, E_inv @ S @ E_inv.T,
                                            rotation_weight=2.0)).sum()
    for _ in range(4):
        F = torch.linalg.matrix_exp(0.5 * torch.randn(d, d, generator=gen, dtype=F64))
        U_inv = torch.linalg.inv(E @ F)
        body = torch.linalg.solve(F, Y @ F)                           # Ad_{F^{-1}} Y
        occurrence = (body * completed_fisher_operator(
            body, U_inv @ mu, U_inv @ S @ U_inv.T, rotation_weight=2.0)).sum()
        assert abs(float(occurrence - middle)) <= 1e-12 * abs(float(middle))


# 7
@pytest.mark.parametrize("rotation_weight", [2.0, 0.7])
def test_pooled_positional_operator_is_the_mean_occurrence_kl_hessian_plus_completion(
        rotation_weight: float) -> None:
    d, B = 3, 3
    gen = _generator(71)
    X = 0.4 * torch.randn(B, 1, d, d, generator=gen, dtype=F64)
    mu = torch.randn(B, 1, d, generator=gen, dtype=F64)
    s = torch.exp(0.5 * torch.randn(B, 1, d, generator=gen, dtype=F64))
    F = torch.linalg.matrix_exp(0.4 * torch.randn(d, d, generator=gen, dtype=F64))
    xi0, C0, C0_inv = middle_fiber_moments(X, mu, s)
    operator = pooled_pos_operator(xi0[None], C0[None], C0_inv[None],
                                   rotation_weight=rotation_weight)[0, 0]
    E = torch.linalg.matrix_exp(X[:, 0])
    L = torch.linalg.cholesky(C0[:, 0])                               # (B, d, d)

    def objective(w: torch.Tensor) -> torch.Tensor:
        W = w.reshape(d, d)                                            # F -> e^W F
        total = torch.zeros((), dtype=F64)
        for b in range(B):
            U = E[b] @ F
            U_new = E[b] @ torch.linalg.matrix_exp(W) @ F
            reference = (torch.linalg.solve(U, mu[b, 0]), torch.linalg.solve(U, torch.diag(
                s[b, 0])) @ torch.linalg.inv(U).T)
            moved = (torch.linalg.solve(U_new, mu[b, 0]), torch.linalg.solve(U_new, torch.diag(
                s[b, 0])) @ torch.linalg.inv(U_new).T)
            Z = torch.linalg.solve(L[b], W @ L[b])
            total = total + _kl(*reference, *moved) + 0.25 * rotation_weight * (
                Z - Z.T).square().sum() / 2
        return total / B

    dense = hessian(objective, torch.zeros(d * d, dtype=F64))
    assert _rel(operator, dense) <= 1e-10
    g = torch.randn(d * d, generator=gen, dtype=F64)
    x, ok, residual = jacobi_cholesky_solve(operator, g)
    assert bool(ok) and float(residual) <= 1e-12
    assert _rel(x, torch.linalg.solve(dense, g)) <= 1e-10


def _type_statistics(mu, s, count, total) -> FisherRowStatistics:
    return FisherRowStatistics(kind="type", count=count, total=total, local_mean=mu, local_var=s)


# 8
def test_direction_identities_off_strip_rejection_and_descent_pairing() -> None:
    d, rows = 3, 3
    gen = _generator(81)
    X = 0.5 * torch.randn(rows, 1, d, d, generator=gen, dtype=F64)
    X[2, 0] = 0.0
    X[2, 0, 0, 1], X[2, 0, 1, 0] = -3.2, 3.2                          # eigenvalues +-3.2i: off strip
    grad = torch.randn(rows, 1, d, d, generator=gen, dtype=F64)
    mu = torch.randn(rows, 1, d, generator=gen, dtype=F64)
    s = torch.exp(0.3 * torch.randn(rows, 1, d, generator=gen, dtype=F64))
    statistics = _type_statistics(mu, s, torch.tensor([2.0, 1.0, 1.0], dtype=F64),
                                  torch.tensor(6.0, dtype=F64))
    U = torch.linalg.matrix_exp(X)
    diagnostics = {}
    direction = belief_fisher_group_direction(
        grad.reshape(rows, -1), X.reshape(rows, -1), U, collect=True, statistics=statistics,
        rotation_weight=2.0, chart_velocity=True, diagnostics_out=diagnostics)
    # Row 2 is off the strip; since 2026-10-07 (audit F21) its covector and chart velocity come from the
    # eigenbasis Psi inversion and satisfy the same identities as the on-strip rows.
    assert direction.accepted.tolist() == [True, True, True]
    assert float(torch.cat(diagnostics["offstrip_rows"]).sum()) == 1.0
    for r in range(3):
        W = U[r, 0] @ direction.xi[r, 0] @ torch.linalg.inv(U[r, 0])   # spatial direction
        v = direction.v_phi[r].reshape(d, d)
        for t in (1e-4, 1e-5):                                         # chart velocity is dexp^{-1}
            gap = torch.linalg.matrix_exp(X[r, 0] + t * v) - torch.linalg.matrix_exp(t * W) @ U[r, 0]
            # Second order in t with constant ~ |v|^2: the off-strip row sits near the exp singularity
            # (eigenvalue difference 6.4i, |Psi_R| ~ 0.018), so its chart velocity is large.
            assert float(gap.norm()) <= 50 * t * t * (1.0 + float(v.norm()) ** 2)
        pairing = (grad[r, 0] * v).sum()
        assert float(pairing) > 0.0
        G_bar = direction.covector[r, 0]
        assert abs(float((G_bar * (U[r, 0] @ direction.xi[r, 0])).sum() - pairing)) <= 1e-10 * float(pairing)
        chart = torch.autograd.functional.jacobian(
            lambda x: (G_bar * torch.linalg.matrix_exp(x.reshape(d, d))).sum(), X[r, 0].reshape(-1))
        assert _rel(chart, grad[r, 0].reshape(-1)) <= 1e-10          # G_bar is grad_U L


def _split_velocity(E, F, tokens, mu, s, loss, pullback: bool = False) -> torch.Tensor:
    r"""Realized-frame spatial velocity Omega_i of the Fisher (or Frobenius-pullback) direction."""
    V = E.shape[0]
    B, N = tokens.shape
    X = logm_principal(E)[0].requires_grad_(True)
    Y = logm_principal(F)[0].requires_grad_(True)
    grad_X, grad_Y = torch.autograd.grad(loss(torch.linalg.matrix_exp(X)[tokens]
                                              @ torch.linalg.matrix_exp(Y)), (X, Y))
    X, Y = X.detach(), Y.detach()
    total = torch.tensor(float(B * N), dtype=F64)
    if pullback:
        xi_E = uspace_pullback_body_direction(X, grad_X)[0]
        xi_F = uspace_pullback_body_direction(Y, grad_Y)[0]
    else:
        count = torch.bincount(tokens.reshape(-1), minlength=V).to(F64)
        type_direction = belief_fisher_group_direction(
            grad_X.reshape(V, -1), X.reshape(V, -1), E[:, None], collect=False,
            statistics=_type_statistics(mu, s, count, total), rotation_weight=2.0)
        xi0, C0, C0_inv = middle_fiber_moments(X[:, None], mu, s)
        gather = tokens.T                                              # (N, B)
        pos_statistics = FisherRowStatistics(
            kind="pos", count=torch.full((N,), float(B), dtype=F64), total=total,
            pooled_operator=pooled_pos_operator(xi0[gather], C0[gather], C0_inv[gather],
                                                rotation_weight=2.0))
        pos_direction = belief_fisher_group_direction(
            grad_Y.reshape(N, -1), Y.reshape(N, -1), F[:, None], collect=False,
            statistics=pos_statistics, rotation_weight=2.0)
        assert bool(type_direction.accepted.all() & pos_direction.accepted.all())
        xi_E, xi_F = type_direction.xi[:, 0], pos_direction.xi[:, 0]
    W_E = E @ xi_E @ torch.linalg.inv(E)                               # (V, d, d)
    W_F = F @ xi_F @ torch.linalg.inv(F)                               # (N, d, d)
    E_i = E[tokens]                                                    # (B, N, d, d)
    return W_E[tokens] + E_i @ W_F @ torch.linalg.inv(E_i)


# 10
def test_realized_frame_velocity_is_covariant_under_the_type_position_split() -> None:
    d, V, B, N = 3, 3, 2, 4
    gen = _generator(101)
    tokens = torch.tensor([[0, 2, 0, 1], [1, 1, 2, 0]])
    E = torch.linalg.matrix_exp(0.4 * torch.randn(V, d, d, generator=gen, dtype=F64))
    F = torch.linalg.matrix_exp(0.3 * torch.randn(N, d, d, generator=gen, dtype=F64))
    mu = torch.randn(V, 1, d, generator=gen, dtype=F64)
    s = torch.exp(0.4 * torch.randn(V, 1, d, generator=gen, dtype=F64))
    A = torch.randn(B, N, d, d, generator=gen, dtype=F64)
    A_inv = torch.randn(B, N, d, d, generator=gen, dtype=F64)

    def loss(U: torch.Tensor) -> torch.Tensor:                         # (B, N, d, d) realized frames
        return (A * U).sum() + (A_inv * torch.linalg.inv(U)).sum() + 0.1 * U.square().sum() ** 2

    h = torch.linalg.matrix_exp(0.3 * torch.randn(d, d, generator=gen, dtype=F64))
    moved = (E @ h, torch.linalg.solve(h, F))                          # (E_v h, h^{-1} F_p)
    fisher = _split_velocity(E, F, tokens, mu, s, loss)
    assert _rel(_split_velocity(*moved, tokens, mu, s, loss), fisher) <= 1e-10
    pullback = _split_velocity(E, F, tokens, mu, s, loss, pullback=True)
    assert _rel(_split_velocity(*moved, tokens, mu, s, loss, pullback=True), pullback) > 1e-3


_ACTIVE_ROUTE = dict(
    vocab_size=16, embed_dim=6, n_heads=2, max_seq_len=6, eval_stride=3, n_layers=2, batch_size=3,
    gauge_group="block_glk", family="gaussian_diagonal", attention_mode="projected_predictive",

    predictive_covariance="moment_matched", pos_phi="learned",
    decode_mode="diagonal_chunked", untie_decode_bank=True, decode_coordinates="local",
    block_coordinates="local", norm_type_block="mahalanobis", use_block_mlp=True,
    block_mlp_mode="coordinate", block_mlp_expansion=2, m_phi_update_mode="fisher_group",
    transport_chart_max_norm=6.0,
)


def _chart_jacobian(X: torch.Tensor) -> torch.Tensor:
    r"""Row-major matrix of ``delta phi -> (dexp_X delta phi) exp(-X)`` for one ``d x d`` block."""
    d = X.shape[-1]
    D = jacobian(lambda x: torch.linalg.matrix_exp(x.reshape(d, d)).reshape(-1), X.reshape(-1))
    return torch.kron(torch.eye(d, dtype=F64), torch.linalg.matrix_exp(-X).T.contiguous()) @ D


# 11
def test_tiny_model_direction_matches_a_dense_chart_route_and_descends() -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import PhiFisherStatisticsProvider

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(**_ACTIVE_ROUTE)
        torch.manual_seed(0)
        model = VFEModel(cfg).double()
    pb = model.prior_bank
    gen = _generator(4)
    with torch.no_grad():
        pb.phi_embed.copy_(0.4 * torch.randn(pb.phi_embed.shape, generator=gen, dtype=F64))
        pb.sigma_log_embed.add_(0.3 * torch.randn(pb.sigma_log_embed.shape, generator=gen, dtype=F64))
        pb.mu_embed.copy_(2.0 * torch.randn(pb.mu_embed.shape, generator=gen, dtype=F64))
        model.pos_phi_free.copy_(0.3 * torch.randn(model.pos_phi_free.shape, generator=gen, dtype=F64))
    tokens = torch.randint(0, 16, (3, 6), generator=_generator(5))
    targets = torch.randint(0, 16, (3, 6), generator=_generator(6))

    def model_loss():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model(tokens, targets, return_decode_stats=True)

    _, loss0, decode_stats = model_loss()
    loss0.backward()
    provider = PhiFisherStatisticsProvider(model, 2.0)
    provider.set_batch(tokens, decode_stats.scored_tokens)
    total = float(decode_stats.scored_tokens)
    H, d = 2, 3
    velocities, pairing = [], 0.0
    for table in (pb.phi_embed, model.pos_phi_free):
        grad = table.grad.detach()
        rows = (grad.abs().sum(dim=-1) > 0).nonzero(as_tuple=True)[0]
        statistics = provider(table, rows)
        phi = table.detach()[rows]
        U = torch.linalg.matrix_exp(phi.view(-1, H, d, d))
        direction = belief_fisher_group_direction(
            grad[rows], phi, U, collect=False, statistics=statistics, rotation_weight=2.0,
            chart_velocity=True)
        assert bool(direction.accepted.all())
        for r, row in enumerate(rows.tolist()):
            for h in range(H):
                X = phi[r].view(H, d, d)[h]
                J = _chart_jacobian(X)
                if table is pb.phi_embed:
                    M = _operator_matrix(pb.mu_embed.detach()[row].view(H, d)[h],
                                         torch.diag(statistics.local_var[r, h]), 2.0)
                    weight = float(statistics.count[r]) / total
                else:
                    blocks = []
                    for v in tokens[:, row].tolist():
                        E_inv = torch.linalg.matrix_exp(-pb.phi_embed.detach()[v].view(H, d, d)[h])
                        m = pb.mu_embed.detach()[v].view(H, d)[h]
                        S = torch.diag(provider._type_moments(torch.tensor([v]))[1][0, h])
                        blocks.append(_operator_matrix(E_inv @ m, E_inv @ S @ E_inv.T, 2.0))
                    M = torch.stack(blocks).mean(dim=0)
                    weight = tokens.shape[0] / total
                chart_metric = J.T @ (weight * M) @ J
                g = grad[row].view(H, d, d)[h].reshape(-1)
                dense = torch.linalg.solve(chart_metric, g)
                assert _rel(direction.v_phi[r].view(H, d * d)[h], dense) <= 1e-10
        velocity = torch.zeros_like(table)
        velocity[rows] = direction.v_phi.to(dtype=velocity.dtype)
        velocities.append(velocity)
        pairing += float((grad * velocity).sum())
    assert pairing > 0.0
    epsilon = 1e-3
    with torch.no_grad():
        for table, velocity in zip((pb.phi_embed, model.pos_phi_free), velocities):
            table.sub_(epsilon * velocity)
        loss1 = model_loss()[1]
    ratio = float(loss1 - loss0.detach()) / (-epsilon * pairing)
    assert abs(ratio - 1.0) <= 0.02


def _stub_provider(mu: torch.Tensor, s: torch.Tensor, counts: torch.Tensor):
    def provider(param: nn.Parameter, rows: torch.Tensor) -> FisherRowStatistics:
        return _type_statistics(mu[rows], s[rows], counts[rows], torch.tensor(8.0, dtype=F64))
    return provider


# 12 (direct optimizer)
@pytest.mark.parametrize("retraction", ["chart", "group_log"])
def test_fisher_optimizer_moves_only_active_rows_counts_rejections_and_keeps_no_state(
        retraction: str) -> None:
    group = get_group("block_glk")(6, 2)
    gen = _generator(121)
    phi = nn.Parameter((0.3 * torch.randn(5, 18, generator=gen)).float())
    with torch.no_grad():                                               # row 4, head 0 off the strip
        phi[4].zero_()
        phi[4, 1], phi[4, 3] = -3.2, 3.2
    mu = torch.randn(5, 2, 3, generator=gen, dtype=F64)
    s = torch.exp(0.3 * torch.randn(5, 2, 3, generator=gen, dtype=F64))
    counts = torch.tensor([2.0, 0.0, 3.0, 0.0, 1.0], dtype=F64)
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group, phi_group_trust_radius=1e-3, phi_chart_max_norm=5.0, phi_bch_residual_max=1e-6,
        phi_group_retraction=retraction,
        phi_statistics_provider=_stub_provider(mu, s, counts), weight_decay=0.0)
    before = phi.detach().clone()
    phi.grad = torch.zeros_like(phi)
    phi.grad[[0, 2, 4]] = torch.randn(3, 18, generator=gen).float()
    optimizer._collect_gauge_diag = True

    optimizer.step()

    after = phi.detach()
    assert torch.equal(after[[1, 3]], before[[1, 3]])                 # inactive
    # Row 4 is off the strip and moves (audit 2026-10-07, F21: no longer a permanent zero step).
    for r in (0, 2, 4):
        assert not torch.equal(after[r], before[r])
    assert optimizer.phi_rejected_rows == 0
    assert phi.grad is None and optimizer.state_dict()["state"] == {}
    for r in (0, 2, 4):                                                # trust radius on the right factor
        X_old = before[r].double().view(2, 3, 3)
        X_new = after[r].double().view(2, 3, 3)
        right = logm_principal(torch.linalg.matrix_exp(-X_old) @ torch.linalg.matrix_exp(X_new))[0]
        assert float(right.norm()) <= 1e-3 * (1 + 1e-3)
    diagnostics = optimizer._gauge_diag
    assert set(FISHER_DIAGNOSTIC_KEYS) <= set(diagnostics)
    assert diagnostics["phi_fisher_offstrip_rows"] == 1.0
    assert diagnostics["phi_group_rejected_rows"] == 0.0              # the off-strip row is solved, not rejected
    assert math.isnan(diagnostics["phi_fisher_pos_lambda_min"])      # no positional table here
    assert math.isfinite(diagnostics["phi_ridge_direction_cosine_mean"])


def _fisher_cfg(**overrides):

    values = dict(
        vocab_size=12, embed_dim=6, n_heads=2, max_seq_len=8, n_layers=1,
        gauge_group="block_glk", pos_phi="learned", m_phi_update_mode="fisher_group",
        transport_chart_max_norm=6.0, m_phi_lr=0.1,
    )
    values.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


# 12 (build_optimizer + train_step)
@pytest.mark.parametrize("retraction,accumulation", [("group_log", 1), ("chart", 2)])
def test_build_optimizer_and_train_step_route_fisher_group(retraction: str, accumulation: int) -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import PhiFisherStatisticsProvider, build_optimizer, train_step

    cfg = _fisher_cfg(m_phi_group_retraction=retraction)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
    keys_before = list(model.state_dict())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0]])
    phi_before = model.prior_bank.phi_embed.detach().clone()
    pos_before = model.pos_phi_free.detach().clone()
    optimizer._collect_gauge_diag = True
    assert isinstance(optimizer.phi_statistics_provider, PhiFisherStatisticsProvider)
    for group in optimizer.param_groups:
        if group.get("pullback_group", False):
            assert group["weight_decay"] == 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = train_step(model, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1),
                          grad_clip=1.0, grad_accum_steps=accumulation)

    assert math.isfinite(float(loss))
    phi_after = model.prior_bank.phi_embed.detach()
    assert torch.equal(phi_after[4:], phi_before[4:])                  # types absent from the batch
    assert not torch.equal(phi_after[:4], phi_before[:4])
    assert not torch.equal(model.pos_phi_free.detach(), pos_before)
    assert model.prior_bank.phi_embed not in optimizer.state
    assert model.pos_phi_free not in optimizer.state
    assert list(model.state_dict()) == keys_before
    assert optimizer.phi_statistics_provider._tokens is None          # cleared after the step
    assert set(FISHER_DIAGNOSTIC_KEYS) <= set(optimizer._gauge_diag)
    assert optimizer._gauge_diag["phi_group_rejected_rows"] == 0.0
    assert optimizer._gauge_diag["phi_fisher_pos_cond_max"] >= 1.0      # positional table present
    assert math.isnan(optimizer._gauge_diag["phi_pullback_damped_gen_cond_max"])   # not Frobenius


def test_provider_counts_scored_tokens_and_positional_columns() -> None:
    from vfe3.model.model import VFEModel
    from vfe3.numerics import bounded_variance_from_log
    from vfe3.train import PhiFisherStatisticsProvider

    cfg = _fisher_cfg()
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    provider = PhiFisherStatisticsProvider(model, 2.0)
    with pytest.raises(RuntimeError, match="without batch tokens"):
        provider(model.prior_bank.phi_embed, torch.tensor([0]))
    tokens = torch.tensor([[0, 1, 1, 5], [5, 5, 2, 1]])
    provider.set_batch(tokens, torch.tensor(7))
    type_statistics = provider(model.prior_bank.phi_embed, torch.tensor([1, 5, 7]))
    assert type_statistics.kind == "type"
    assert type_statistics.count.tolist() == [3.0, 3.0, 0.0]
    assert float(type_statistics.total) == 7.0
    pb = model.prior_bank
    s = bounded_variance_from_log(pb._prior_sigma_log_table()[5].double(), eps=pb.eps)
    torch.testing.assert_close(type_statistics.local_var[1].reshape(-1), s)
    pos_statistics = provider(model.pos_phi_free, torch.tensor([0, 2]))
    assert pos_statistics.kind == "pos" and pos_statistics.count.tolist() == [2.0, 2.0]
    X = pb.phi_embed.detach().double()[[0, 5]].view(2, 2, 3, 3)
    mu = pb._prior_mu_table().detach().double()[[0, 5]].view(2, 2, 3)
    var = bounded_variance_from_log(pb._prior_sigma_log_table().detach().double()[[0, 5]],
                                    eps=pb.eps).view(2, 2, 3)
    xi0, C0, C0_inv = middle_fiber_moments(X, mu, var)
    expected = pooled_pos_operator(xi0[None], C0[None], C0_inv[None], rotation_weight=2.0)[0]
    assert pos_statistics.pooled_operator is None                        # formed per row chunk by the consumer
    from vfe3.geometry.phi_fisher import pooled_operator_chunks
    start, stop, operator = next(pooled_operator_chunks(pos_statistics, rotation_weight=2.0))
    assert (start, stop) == (0, 2)
    torch.testing.assert_close(operator[0], expected)
    with pytest.raises(ValueError, match="phi_embed and pos_phi_free"):
        provider(nn.Parameter(torch.zeros(2, 18)), torch.tensor([0]))


# 13
def test_fisher_group_policy_config_validation_and_inactive_marks() -> None:

    policy = get_phi_update_policy("fisher_group")
    assert policy.requires_fisher_statistics and policy.requires_manifold_optimizer
    assert not policy.requires_pullback_geometry
    assert dict(policy.optimizer_group_metadata) == {"pullback_group": True, "weight_decay": 0.0}
    assert not get_phi_update_policy("pullback_group").requires_fisher_statistics
    for overrides, message in (
            (dict(encode_mode="per_token_additive"), "encode_mode='per_token'"),
            (dict(encode_mode="canonical_content_projected"), None),
            (dict(gauge_group="so_k"), "gauge_group"),
            (dict(gauge_group="tied_block_glk"), None),
            (dict(pos_phi_project_slk=True), "pos_phi_project_slk"),
            (dict(family="gaussian_frame_diagonal"), "diag"),
            (dict(transport_chart_max_norm=None), "transport_chart_max_norm"),
            (dict(transport_chart_max_norm=4.0), "transport_chart_max_norm"),
            (dict(m_phi_fisher_rotation_weight=0.0), "m_phi_fisher_rotation_weight"),
            (dict(m_phi_fisher_rotation_weight=float("nan")), "m_phi_fisher_rotation_weight")):
        with pytest.raises(ValueError, match=message):
            _fisher_cfg(**overrides)
    _fisher_cfg(family="gaussian_diagonal", attention_mode="projected_predictive")                       # layer-0 row is diag(s)
    with pytest.raises(ValueError, match="gauge_group"):
        _fisher_cfg(gauge_group="glk", embed_dim=3, n_heads=1)
    inactive = test_config(m_phi_update_mode="adamw", m_phi_fisher_rotation_weight=2.0)
    assert inactive.m_phi_fisher_rotation_weight == 2.0
    assert "m_phi_fisher_rotation_weight" in inactive.inactive_fields()
    active = _fisher_cfg()
    for name in ("m_phi_fisher_rotation_weight", "m_phi_group_trust_radius",
                 "m_phi_group_retraction"):
        assert name not in active.inactive_fields()
    with pytest.raises(ValueError, match="phi_fisher_rotation_weight"):
        GaugeManifoldAdamW([nn.Parameter(torch.zeros(1))], get_group("block_glk")(n_heads=1, K=2),
                           phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                           phi_bch_residual_max=1e-6, 
                           phi_fisher_rotation_weight=-1.0)
    with pytest.raises(ValueError, match="group_log"):
        GaugeManifoldAdamW([nn.Parameter(torch.zeros(1))], get_group("so_n")(3, group_n=3, irrep_spec=[("l1", 1)]),
                           phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                           phi_bch_residual_max=1e-6, 
                           phi_statistics_provider=lambda param, rows: None)
