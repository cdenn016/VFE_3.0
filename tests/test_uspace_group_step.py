r"""U-space pullback step: ``m_phi_group_retraction='group_log'`` (derivation 2026-09-28, Track 1).

Tiny float64 CPU checks of the matrix-log kernels, the closed-form Frobenius-pullback body direction,
the exact right-product retraction and its gates, and the optimizer/config wiring. The chart route
is covered unchanged by tests/test_gauge_optim.py and tests/test_checkpoint_resume.py.
"""
from tests.profiles import config_values, test_config

import functools
import math

import pytest
import torch
from torch import nn

import vfe3.gauge_optim as gauge_optim
import vfe3.geometry.phi_preconditioner as phi_preconditioner
from vfe3.gauge_optim import (
    GROUP_LOG_DIAGNOSTIC_KEYS,
    GaugeManifoldAdamW,
    GroupLogDirection,
    group_log_block_shape,
    pullback_group_log_direction,
    stage_group_log_candidate,
    stage_pullback_group_candidate,
)
from vfe3.geometry.groups import get_group
from vfe3.geometry.matrix_log import (
    logm_frechet_pair,
    logm_principal,
    principal_strip_certificate,
    sqrtm_denman_beavers,
)
from vfe3.geometry.phi_preconditioner import (
    pullback_group_direction,
    uspace_ambient_gradient,
    uspace_chart_velocity,
    uspace_pullback_body_direction,
    uspace_spatial_covector,
)

F64 = torch.float64
J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=F64)       # rotation generator, spec +-i
I2 = torch.eye(2, dtype=F64)


def _blocks(rows: int, d: int, norm: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(rows, d, d, generator=generator, dtype=F64)
    return X * (norm / torch.linalg.matrix_norm(X))[:, None, None]


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    r"""Largest per-matrix relative Frobenius error over the batch."""
    return float((torch.linalg.matrix_norm(a - b)
                  / torch.linalg.matrix_norm(b).clamp_min(1e-300)).max())


def _expm_frechet(X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    r"""``L_exp(X, E)``: the (1,2) block of ``exp([[X, E], [0, X]])``."""
    d = X.shape[-1]
    block = torch.cat((torch.cat((X, E), -1), torch.cat((torch.zeros_like(X), X), -1)), -2)
    return torch.linalg.matrix_exp(block)[..., :d, d:]


def _chart_covector(X: torch.Tensor, G_bar: torch.Tensor) -> torch.Tensor:
    r"""Chart covector of ``L(U) = <G_bar, U>`` at ``U = exp(X)`` by autograd."""
    X = X.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad((G_bar * torch.linalg.matrix_exp(X)).sum(), X)
    return grad


def _pullback(group):
    return functools.partial(pullback_group_log_direction, group=group)


def _fixed_direction(xi_blocks: torch.Tensor):
    r"""A direction callable with a prescribed body generator and the pullback covector ``U xi``."""
    def direction(grad_phi, phi, current_blocks, *, collect):
        xi = xi_blocks.reshape(current_blocks.shape).to(F64)
        return GroupLogDirection(
            xi=xi, covector=current_blocks @ xi,
            accepted=torch.ones(phi.shape[0], dtype=torch.bool))
    return direction


def _stage(grad, phi, group, direction, **kwargs):
    options = dict(learning_rate=0.05, trust_radius=0.1, chart_max_norm=20.0,
                   bch_residual_max=1e-6)
    options.update(kwargs)
    return stage_group_log_candidate(grad, phi, group, direction=direction, **options)


# T1
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("norm", [0.1, 1.0, 2.5])
def test_principal_log_and_exp_are_mutual_inverses_on_the_strip(d: int, norm: float) -> None:
    X = _blocks(12, d, norm, seed=10 * d)
    certified, _ = principal_strip_certificate(X, margin=0.0)
    X = X[certified]
    U = torch.linalg.matrix_exp(X)
    L, failed, _ = logm_principal(U)
    assert X.shape[0] > 0 and not bool(failed.any())
    assert _rel(L, X) <= 1e-12
    assert _rel(torch.linalg.matrix_exp(L), U) <= 1e-12
    root, root_failed = sqrtm_denman_beavers(U)
    assert not bool(root_failed.any()) and _rel(root @ root, U) <= 1e-12


# T2
def test_kernels_flag_matrices_without_a_real_principal_logarithm() -> None:
    A = torch.diag(torch.tensor([-1.0, -2.0], dtype=F64))
    _, root_failed = sqrtm_denman_beavers(A)
    L, failed, _ = logm_principal(A)
    _, L_frechet, pair_failed = logm_frechet_pair(A, I2)
    assert bool(root_failed) and bool(failed) and bool(pair_failed)
    assert bool(torch.isnan(L).all()) and bool(torch.isnan(L_frechet).all())
    nonfinite = torch.full((2, 2), float("nan"), dtype=F64)
    assert bool(logm_principal(nonfinite)[1])


# T3
@pytest.mark.parametrize("norm", [0.1, 1.0, 2.5])
def test_frechet_pair_inverts_the_exponential_differential(norm: float) -> None:
    X = _blocks(8, 3, norm, seed=3)
    E = _blocks(8, 3, 1.0, seed=4)
    log_A, L, failed = logm_frechet_pair(torch.linalg.matrix_exp(X), _expm_frechet(X, E))
    assert not bool(failed.any())
    assert _rel(L, E) <= 1e-10
    assert _rel(log_A, X) <= 1e-12


# T4
@pytest.mark.parametrize("name,kwargs,mode", [
    ("block_glk", {"K": 6, "n_heads": 2}, "pullback_per_block"),
    ("block_glk", {"K": 3, "n_heads": 1}, "pullback_per_block"),
])
def test_closed_form_direction_equals_the_series_pullback_direction(
    monkeypatch, name: str, kwargs: dict, mode: str,
) -> None:
    group = get_group(name)(**kwargs, dtype=F64)
    H, d = group_log_block_shape(group)
    n_gen = group.generator_shape[0]
    generator = torch.Generator().manual_seed(7)
    phi = 0.5 * torch.randn(5, n_gen, generator=generator, dtype=F64)
    grad = torch.randn(5, n_gen, generator=generator, dtype=F64)
    U = torch.linalg.matrix_exp(phi.view(5, H, d, d))

    def compare() -> tuple[float, float]:
        closed = pullback_group_log_direction(grad, phi, U, collect=True, group=group)
        series = pullback_group_direction(
            grad, phi, group.generators, mode=mode,
            coordinate_layout=group.phi_coordinate_layout, irrep_dims=list(group.irrep_dims),
            reject_ill_conditioned_rows=True)
        assert bool(closed.fast_rows.all()) and bool(closed.accepted.all())
        xi_error = float((closed.xi.reshape(5, n_gen) - series.xi).norm() / series.xi.norm())
        v_error = float((closed.v_phi - series.v_phi).norm() / series.v_phi.norm())
        return xi_error, v_error

    with_ridge = compare()
    monkeypatch.setattr(phi_preconditioner, "_PHI_GROUP_GRAM_RIDGE", 0.0)
    xi_error, v_error = compare()
    assert xi_error <= 1e-10 and v_error <= 1e-10
    assert max(with_ridge) <= 1e-3


# T5
def test_ambient_gradient_spatial_covector_and_chart_velocity_identities() -> None:
    X = _blocks(4, 3, 1.0, seed=11).requires_grad_(True)
    W1 = _blocks(4, 3, 1.0, seed=12)
    W2 = _blocks(4, 3, 1.0, seed=13)

    def loss(U: torch.Tensor) -> torch.Tensor:
        return (W1 * U).sum() + (W2 * torch.linalg.inv(U)).sum()

    (grad_X,) = torch.autograd.grad(loss(torch.linalg.matrix_exp(X)), X)
    U_leaf = torch.linalg.matrix_exp(X.detach()).requires_grad_(True)
    (grad_U,) = torch.autograd.grad(loss(U_leaf), U_leaf)
    X = X.detach()
    U = torch.linalg.matrix_exp(X)

    G_bar, ok = uspace_ambient_gradient(X, grad_X)
    assert bool(ok.all()) and _rel(G_bar, grad_U) <= 1e-10
    G_W, ok_W = uspace_spatial_covector(X, grad_X)
    assert bool(ok_W.all()) and _rel(G_W, grad_U @ U.transpose(-1, -2)) <= 1e-10
    W = _blocks(4, 3, 1.0, seed=14)
    v, ok_v = uspace_chart_velocity(X, W)
    assert bool(ok_v.all()) and _rel(_expm_frechet(X, v), W @ U) <= 1e-10
    # <G_W, W> = <grad_phi, v>: the spatial covector pairs with the spatial velocity.
    torch.testing.assert_close((G_W * W).sum(dim=(-2, -1)), (grad_X * v).sum(dim=(-2, -1)),
                               rtol=1e-10, atol=1e-12)
    xi, ok_xi = uspace_pullback_body_direction(X, grad_X)
    assert bool(ok_xi.all()) and _rel(U @ xi, grad_U) <= 1e-10


# T6
def test_group_log_candidate_is_the_exact_right_product_with_second_order_chart_gap(
    monkeypatch,
) -> None:
    monkeypatch.setattr(phi_preconditioner, "_PHI_GROUP_GRAM_RIDGE", 0.0)
    group = get_group("block_glk")(K=4, n_heads=2, dtype=F64)
    mode = "pullback_per_block"
    generator = torch.Generator().manual_seed(21)
    phi = 0.3 * torch.randn(3, 8, generator=generator, dtype=F64)
    grad = torch.randn(3, 8, generator=generator, dtype=F64)
    series = pullback_group_direction(grad, phi, group.generators, mode=mode,
                                      coordinate_layout=group.phi_coordinate_layout,
                                      irrep_dims=[2, 2], reject_ill_conditioned_rows=True)
    grad = grad / series.xi.norm(dim=-1, keepdim=True)          # unit ||xi|| per row (linearity)
    series = pullback_group_direction(grad, phi, group.generators, mode=mode,
                                      coordinate_layout=group.phi_coordinate_layout,
                                      irrep_dims=[2, 2], reject_ill_conditioned_rows=True)
    U = torch.linalg.matrix_exp(phi.view(3, 2, 2, 2))
    xi = series.xi.view(3, 2, 2, 2)

    def W(t: float) -> torch.Tensor:
        return logm_principal(U @ torch.linalg.matrix_exp(-t * xi))[0].reshape(3, 8)

    h = 1e-3
    second = (W(h) - 2.0 * phi + W(-h)) / h**2                  # W''(0), central difference
    for eta in (1e-3, 1e-2):
        candidate = _stage(grad, phi, group, _pullback(group), learning_rate=eta,
                           trust_radius=1e3)
        assert float(candidate.group_product_residual.max()) <= 1e-12
        assert int(candidate.backtracking_reductions.max()) == 0
        gap = candidate.candidate_phi - (phi - eta * series.v_phi)
        ratio = gap.norm(dim=-1) / (0.5 * eta**2 * second.norm(dim=-1))
        assert float((ratio - 1.0).abs().max()) <= 1e-2
    stored = _stage(grad, phi.float(), group, _pullback(group), learning_rate=1e-2)
    assert float(stored.group_product_residual.max()) <= 1e-6


# T7
def test_trust_scale_bounds_the_learning_rate_weighted_right_factor() -> None:
    group = get_group("block_glk")(K=4, n_heads=2, dtype=F64)
    generator = torch.Generator().manual_seed(5)
    phi = 0.3 * torch.randn(4, 8, generator=generator, dtype=F64)
    grad = torch.randn(4, 8, generator=generator, dtype=F64) * torch.tensor(
        [[0.01], [0.1], [1.0], [10.0]], dtype=F64)
    lr, radius = 0.5, 0.05
    candidate = _stage(grad, phi, group, _pullback(group),
                       learning_rate=lr, trust_radius=radius)
    right = lr * candidate.direction.xi.norm(dim=-1)
    torch.testing.assert_close(candidate.trust_scale, (radius / right).clamp(max=1.0))
    assert bool((candidate.trust_scale * right <= radius * (1 + 1e-12)).all())
    assert float(candidate.trust_scale.min()) < 1.0 < float(candidate.trust_scale.max()) + 1e-12


# T8
def test_trust_saturated_step_needs_no_truncation_halvings() -> None:
    group = get_group("block_glk")(K=6, n_heads=2, dtype=F64)
    generator = torch.Generator().manual_seed(8)
    phi = torch.randn(4, 18, generator=generator, dtype=F64) / 3.0     # ||X_h||_F ~ 1
    grad = torch.randn(4, 18, generator=generator, dtype=F64)
    options = dict(learning_rate=10.0, trust_radius=0.1, chart_max_norm=20.0,
                   bch_residual_max=1e-6)
    group_log = stage_group_log_candidate(grad, phi, group, direction=_pullback(group),
                                          **options)
    chart = stage_pullback_group_candidate(grad, phi, group, **options)
    assert bool((group_log.trust_scale < 1.0).all())
    assert int(group_log.backtracking_reductions.max()) == 0
    assert not bool(group_log.rejected_rows.any())
    assert int(chart.backtracking_reductions.min()) > 0


# T9
def test_off_strip_row_uses_the_series_fallback_and_is_recharted_exactly(monkeypatch) -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    X_off = (math.pi + 0.3) * J2 + 0.1 * I2
    X_in = _blocks(1, 2, 0.2, seed=9)[0]
    phi = torch.stack((X_off.reshape(4), X_in.reshape(4)))
    grad = torch.randn(2, 4, generator=torch.Generator().manual_seed(9), dtype=F64)
    seen = []

    def spy(grad_rows, phi_rows, *args, **kwargs):
        seen.append(phi_rows.clone())
        return pullback_group_direction(grad_rows, phi_rows, *args, **kwargs)

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", spy)
    candidate = _stage(grad, phi, group, _pullback(group), learning_rate=0.01,
                       collect=True)
    assert len(seen) == 1 and torch.equal(seen[0], phi[:1])
    assert candidate.fast_rows.tolist() == [False, True]
    assert not bool(candidate.rejected_rows.any())
    assert candidate.recharted_rows.tolist() == [True, False]
    new_blocks = candidate.candidate_phi.view(2, 2, 2)
    assert float(torch.linalg.eigvals(new_blocks).imag.abs().max()) < math.pi
    assert float(candidate.group_product_residual.max()) <= 1e-6
    assert candidate.direction.current_element.shape == (2, 2, 2)


def test_conditioning_rejected_off_strip_row_keeps_its_chart_without_recharting() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    # Psi_L(ad_X) is nearly singular (ad eigenvalue 2i(pi + 1e-4)): the series metric fails its
    # conditioning certificate, and Log(U) would re-chart this row by 2 pi if it were retracted.
    phi = torch.stack(((math.pi + 1e-4) * J2, 0.2 * I2)).reshape(2, 4)
    grad = torch.randn(2, 4, generator=torch.Generator().manual_seed(2), dtype=F64)
    candidate = _stage(grad, phi, group, _pullback(group), learning_rate=0.01)
    assert candidate.rejected_rows.tolist() == [True, False]
    assert torch.equal(candidate.candidate_phi[0], phi[0])
    assert candidate.recharted_rows.tolist() == [False, False]
    assert not torch.equal(candidate.candidate_phi[1], phi[1])


# T10
def test_log_failure_halves_then_accepts_or_rejects_without_raising() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    phi = ((math.pi - 0.05) * J2 + 0.1 * I2).reshape(1, 4)
    xi = -0.1 * torch.diag(torch.tensor([1.0, -1.0], dtype=F64))    # Z = diag(0.1, -0.1)
    grad = torch.ones(1, 4, dtype=F64)                                  # only its finiteness is read
    target = torch.linalg.matrix_exp(phi.view(2, 2)) @ torch.linalg.matrix_exp(-xi)
    eigenvalues = torch.linalg.eigvals(target)
    assert bool((eigenvalues.imag == 0).all() & (eigenvalues.real < 0).all())   # no real log
    options = dict(learning_rate=1.0, trust_radius=1.0)
    accepted = _stage(grad, phi, group, _fixed_direction(xi), **options)
    assert int(accepted.backtracking_reductions) == 2
    assert not bool(accepted.rejected_rows.any())
    assert float(accepted.group_product_residual) <= 1e-6
    rejected = _stage(grad, phi, group, _fixed_direction(xi), max_backtracks=0, **options)
    assert bool(rejected.rejected_rows.all())
    assert torch.equal(rejected.candidate_phi, phi)


# T11
def test_chart_bound_halves_then_rejects() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    X = _blocks(1, 2, 0.5, seed=12)[0]
    phi = X.reshape(1, 4)
    xi = -X / torch.linalg.matrix_norm(X)                   # e^X e^{t X/|X|}: chart norm |X| + t
    grad = torch.ones(1, 4, dtype=F64)
    options = dict(learning_rate=0.2, trust_radius=10.0)
    halved = _stage(grad, phi, group, _fixed_direction(xi), chart_max_norm=0.5 + 0.06, **options)
    assert int(halved.backtracking_reductions) == 2 and not bool(halved.rejected_rows.any())
    torch.testing.assert_close(halved.candidate_chart_norm, torch.tensor([0.55], dtype=F64))
    stuck = _stage(grad, phi, group, _fixed_direction(xi), chart_max_norm=0.49,
                   max_backtracks=3, **options)
    assert int(stuck.backtracking_reductions) == 3 and bool(stuck.rejected_rows.all())
    assert torch.equal(stuck.candidate_phi, phi)


def test_ascent_covector_is_rejected_by_the_descent_gate_and_keeps_the_chart() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    phi = _blocks(3, 2, 0.5, seed=41).reshape(3, 4)
    xi = _blocks(3, 2, 1.0, seed=42).view(3, 1, 2, 2)

    def ascent(grad_phi, phi_rows, current_blocks, *, collect):
        return GroupLogDirection(xi=xi, covector=-(current_blocks @ xi),
                                 accepted=torch.ones(3, dtype=torch.bool))

    candidate = _stage(torch.ones(3, 4, dtype=F64), phi, group, ascent, max_backtracks=4)
    assert bool(candidate.rejected_rows.all())
    assert candidate.backtracking_reductions.tolist() == [4, 4, 4]
    assert torch.equal(candidate.candidate_phi, phi)


def test_nonfinite_direction_on_an_accepted_row_raises() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    phi = _blocks(2, 2, 0.5, seed=43).reshape(2, 4)
    xi = _blocks(2, 2, 1.0, seed=44)
    xi[1, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite accepted row"):
        _stage(torch.ones(2, 4, dtype=F64), phi, group, _fixed_direction(xi))


def test_frechet_failure_and_ill_conditioned_frames_take_the_series_fallback(monkeypatch) -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    X_stretched = torch.diag(torch.tensor([3.6, -3.6], dtype=F64))    # kappa_2(U) = e^{7.2} > 1e3
    phi = torch.stack((_blocks(1, 2, 0.4, seed=45)[0], _blocks(1, 2, 0.4, seed=46)[0],
                       X_stretched)).reshape(3, 4)
    grad = torch.randn(3, 4, generator=torch.Generator().manual_seed(47), dtype=F64)
    U = torch.linalg.matrix_exp(phi.view(3, 1, 2, 2))
    seen = []

    def spy(grad_rows, phi_rows, *args, **kwargs):
        seen.append(phi_rows.clone())
        return pullback_group_direction(grad_rows, phi_rows, *args, **kwargs)

    frechet = phi_preconditioner.logm_frechet_pair

    def fail_first_row(A, E):
        log_A, L, failed = frechet(A, E)
        failed = failed.clone()
        failed[0] = True
        return log_A, L, failed

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", spy)
    monkeypatch.setattr(phi_preconditioner, "logm_frechet_pair", fail_first_row)
    proposal = pullback_group_log_direction(grad, phi, U, collect=False, group=group)
    assert proposal.fast_rows.tolist() == [False, True, False]
    assert float(proposal.frame_condition[2]) > 1e3 >= float(proposal.frame_condition[:2].max())
    assert len(seen) == 1 and torch.equal(seen[0], phi[[0, 2]])
    series = pullback_group_direction(
        grad[[0, 2]], phi[[0, 2]], group.generators, mode="pullback_per_block",
        coordinate_layout=group.phi_coordinate_layout, irrep_dims=[2],
        reject_ill_conditioned_rows=True)
    assert proposal.accepted[[0, 2]].tolist() == series.accepted_rows.reshape(-1).tolist()


# T12
@pytest.mark.parametrize("d", [2, 3, 4])
def test_bendixson_certificate_never_certifies_a_block_the_exact_test_rejects(d: int) -> None:
    generator = torch.Generator().manual_seed(d)
    samples = []
    for norm in (0.5, 2.0, 3.0, 5.0):
        gauss = torch.randn(64, d, d, generator=generator, dtype=F64)
        skew = gauss - gauss.transpose(-1, -2) + 0.1 * gauss
        near_symmetric = gauss + gauss.transpose(-1, -2) + 0.1 * gauss
        for X in (gauss, skew, near_symmetric):
            samples.append(X * (norm / torch.linalg.matrix_norm(X))[:, None, None])
    X = torch.cat(samples)
    certified, bendixson_margin = principal_strip_certificate(X, margin=0.25)
    exact = torch.linalg.eigvals(X).imag.abs().amax(dim=-1) < math.pi - 0.25
    assert torch.equal(certified, exact)
    assert bool(exact[bendixson_margin > 0.25].all())
    assert bool((bendixson_margin <= 0.25).any()) and bool((bendixson_margin > 0.25).any())


# T13
def test_positional_model_recovers_both_factor_covectors_and_first_order_velocity() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=F64)
    V, N = 3, 4
    tokens = torch.tensor([0, 2, 0, 1])
    X = _blocks(V, 2, 0.6, seed=31)
    Y = _blocks(N, 2, 0.4, seed=32)
    W = _blocks(N, 2, 1.0, seed=33)
    W_inv = _blocks(N, 2, 1.0, seed=34)
    C = torch.randn(N, N, 2, 2, generator=torch.Generator().manual_seed(35), dtype=F64)

    def loss(E: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        U = E[tokens] @ F                                                  # (N, 2, 2)
        U_inv = torch.linalg.inv(U)
        pair = U[:, None] @ U_inv[None, :]                                 # (N, N, 2, 2)
        return (W * U).sum() + (W_inv * U_inv).sum() + (C * pair).sum() + U.square().sum() ** 2

    X_leaf, Y_leaf = X.clone().requires_grad_(True), Y.clone().requires_grad_(True)
    grad_X, grad_Y = torch.autograd.grad(
        loss(torch.linalg.matrix_exp(X_leaf), torch.linalg.matrix_exp(Y_leaf)), (X_leaf, Y_leaf))
    E, F = torch.linalg.matrix_exp(X), torch.linalg.matrix_exp(Y)
    E_leaf, F_leaf = E.clone().requires_grad_(True), F.clone().requires_grad_(True)
    grad_E, grad_F = torch.autograd.grad(loss(E_leaf, F_leaf), (E_leaf, F_leaf))
    G_E, ok_E = uspace_ambient_gradient(X, grad_X)
    G_F, ok_F = uspace_ambient_gradient(Y, grad_Y)
    active = grad_X.flatten(1).abs().sum(-1) > 0
    assert active.tolist() == [True, True, True]
    assert bool(ok_E.all() & ok_F.all())
    assert _rel(G_E, grad_E) <= 1e-10 and _rel(G_F, grad_F) <= 1e-10

    predicted_velocity = -(G_E[tokens] @ F + E[tokens] @ G_F)             # first-order dU_i / d eta
    errors = []
    for eta in (1e-3, 1e-4):
        new = []
        for table, grad in ((X, grad_X), (Y, grad_Y)):
            candidate = _stage(grad.reshape(-1, 4), table.reshape(-1, 4), group,
                               _pullback(group), learning_rate=eta,
                               trust_radius=1e3)
            new.append(torch.linalg.matrix_exp(candidate.candidate_phi.view(-1, 2, 2)))
        realized = new[0][tokens] @ new[1] - E[tokens] @ F
        errors.append(float((realized - eta * predicted_velocity).norm()
                            / (eta * predicted_velocity).norm()))
    assert errors[1] <= 0.2 * errors[0] and errors[1] <= 2e-2      # relative error O(eta)


# T14
def test_group_log_optimizer_moves_only_active_rows_and_equals_the_stager() -> None:
    group = get_group("block_glk")(n_heads=1, K=2)
    torch.manual_seed(0)
    phi = nn.Parameter(0.2 * torch.randn(5, 4))
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group,
        phi_group_trust_radius=0.1,
        phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6,
        phi_group_retraction="group_log",
        weight_decay=0.0,
    )
    before = phi.detach().clone()
    phi.grad = torch.zeros_like(phi)
    phi.grad[1] = torch.tensor([1.0, 0.0, -0.5, 0.2])
    phi.grad[3, 3] = -2.0
    expected = stage_group_log_candidate(
        phi.grad[[1, 3]], before[[1, 3]], group, direction=_pullback(group),
        learning_rate=0.05, trust_radius=0.1, chart_max_norm=5.0, bch_residual_max=1e-6,
    ).candidate_phi.to(dtype=phi.dtype)
    optimizer._collect_gauge_diag = True

    optimizer.step()

    assert torch.equal(phi.detach()[[0, 2, 4]], before[[0, 2, 4]])
    assert torch.equal(phi.detach()[[1, 3]], expected)
    assert not torch.equal(phi.detach()[1], before[1])
    assert phi.grad is None
    assert phi not in optimizer.state and optimizer.state_dict()["state"] == {}
    diagnostics = optimizer._gauge_diag
    assert set(GROUP_LOG_DIAGNOSTIC_KEYS) <= set(diagnostics)
    assert diagnostics["phi_group_fast_rows"] == 2.0
    assert diagnostics["phi_group_fallback_rows"] == 0.0
    assert diagnostics["phi_group_log_backtracks_max"] == 0.0
    assert math.isfinite(diagnostics["phi_ridge_direction_cosine_mean"])
    assert math.isnan(diagnostics["phi_pullback_damped_gen_cond_max"])   # no metric on fast rows
    assert 1.0 <= diagnostics["phi_group_fast_frame_cond_max"] <= 1e3


def _group_log_model_cfg(**overrides):
    from vfe3.config import VFE3Config

    values = dict(
        vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
        gauge_group="block_glk", pos_phi="none", m_phi_update_mode="pullback_group",
        transport_chart_max_norm=6.0, m_phi_lr=0.1,
        m_phi_group_retraction="group_log",
    )
    values.update(overrides)
    return test_config(**values)


# T15
def test_build_optimizer_routes_group_log_and_train_step_moves_phi_embed() -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = _group_log_model_cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    phi_before = model.prior_bank.phi_embed.detach().clone()
    optimizer._collect_gauge_diag = True

    loss = train_step(model, optimizer, scheduler, torch.randint(0, 12, (4, 8)),
                      torch.randint(0, 12, (4, 8)), grad_clip=1.0)

    assert isinstance(optimizer, GaugeManifoldAdamW)
    assert optimizer._phi_group_retraction == "group_log"
    assert torch.isfinite(torch.tensor(float(loss)))
    assert not torch.equal(model.prior_bank.phi_embed.detach(), phi_before)
    assert model.prior_bank.phi_embed not in optimizer.state
    assert set(GROUP_LOG_DIAGNOSTIC_KEYS) <= set(optimizer._gauge_diag)


# T16
def test_group_log_resume_is_step_exact_and_stateless(tmp_path) -> None:
    from tests.test_checkpoint_resume import _params, _pullback_resume_cfg, _sequential_loader
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import train

    cfg = _pullback_resume_cfg(m_phi_group_retraction="group_log")
    torch.manual_seed(0)
    uninterrupted = VFEModel(cfg)
    phi_initial = uninterrupted.prior_bank.phi_embed.detach().clone()
    train(uninterrupted, _sequential_loader(), cfg, n_steps=4)
    assert not torch.equal(phi_initial, uninterrupted.prior_bank.phi_embed)

    torch.manual_seed(0)
    partial = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "group_log", cfg, partial)
    train(partial, _sequential_loader(), cfg, n_steps=2, artifacts=artifacts)
    checkpoint = artifacts.ckpt_dir / "step_2.pt"
    optimizer_state = torch.load(checkpoint, weights_only=True)["optimizer_state"]
    phi_ids = {pid for group in optimizer_state["param_groups"]
               if group.get("pullback_group", False) for pid in group["params"]}
    assert phi_ids and all(not optimizer_state["state"].get(pid) for pid in phi_ids)

    resumed = VFEModel(cfg)
    train(resumed, _sequential_loader(), cfg, n_steps=4, resume_from=checkpoint)
    for expected, actual in zip(_params(uninterrupted), _params(resumed)):
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)


# T17
def test_retraction_config_validation_inactive_marks_and_serialized_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect
    from vfe3.config import VFE3Config, config_from_serialized, serialize_config

    with pytest.raises(ValueError, match="m_phi_group_retraction"):
        test_config(m_phi_group_retraction="bogus")
    default = test_config(m_phi_update_mode="adamw", m_phi_group_retraction="chart")
    assert default.m_phi_group_retraction == "chart"
    assert "m_phi_group_retraction" in default.inactive_fields()
    active = _group_log_model_cfg()
    assert "m_phi_group_retraction" not in active.inactive_fields()
    payload = serialize_config(_group_log_model_cfg(m_phi_group_retraction="chart"))
    payload.pop("m_phi_group_retraction")
    parameters = list(inspect.signature(VFE3Config.__init__).parameters)[1:]
    values = config_values()
    monkeypatch.setattr(VFE3Config.__init__, "__defaults__", tuple(values[name] for name in parameters))
    assert config_from_serialized(payload, source="pre-retraction run").m_phi_group_retraction == "chart"
    with pytest.raises(ValueError, match="phi_group_retraction"):
        GaugeManifoldAdamW([nn.Parameter(torch.zeros(1))], get_group("block_glk")(n_heads=1, K=2),
                           phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                           phi_bch_residual_max=1e-6, 
                           phi_group_retraction="bogus")
    with pytest.raises(ValueError, match="group_log"):
        GaugeManifoldAdamW([nn.Parameter(torch.zeros(1))], get_group("so_n")(3, group_n=3, irrep_spec=[("l1", 1)]),
                           phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                           phi_bch_residual_max=1e-6, 
                           phi_group_retraction="group_log")


# T19
@pytest.mark.parametrize("side", ["left", "right"])
def test_frobenius_step_is_orthogonally_but_not_gl_equivariant(side: str) -> None:
    X = _blocks(1, 3, 0.3, seed=41)
    G_bar = _blocks(1, 3, 1.0, seed=42)
    eta = 1e-2

    def realized_step(X: torch.Tensor, G_bar: torch.Tensor) -> torch.Tensor:
        xi, ok = uspace_pullback_body_direction(X, _chart_covector(X, G_bar))
        assert bool(ok.all())
        return torch.linalg.matrix_exp(X) @ torch.linalg.matrix_exp(-eta * xi)

    def defect(g: torch.Tensor) -> float:
        U = torch.linalg.matrix_exp(X)
        if side == "left":       # E -> g E, G_bar -> g^{-T} G_bar
            X_moved = logm_principal(g @ U)[0]
            G_moved = torch.linalg.solve(g.transpose(-1, -2), G_bar)
            expected = g @ realized_step(X, G_bar)
        else:                    # E -> E h, G_bar -> G_bar h^{-T}
            X_moved = logm_principal(U @ g)[0]
            G_moved = torch.linalg.solve(g, G_bar.transpose(-1, -2)).transpose(-1, -2)
            expected = realized_step(X, G_bar) @ g
        displacement = (expected - (g @ U if side == "left" else U @ g)).norm()
        return float((realized_step(X_moved, G_moved) - expected).norm() / displacement)

    A = _blocks(1, 3, 0.4, seed=43)
    rotation = torch.linalg.matrix_exp(A - A.transpose(-1, -2))
    general = torch.linalg.matrix_exp(A)
    assert defect(rotation) <= 1e-10
    assert defect(general) > 1e-3
