import gc
import math
import weakref

import pytest
import torch

import vfe3.geometry.phi_preconditioner as phi_preconditioner
from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.phi_preconditioner import (
    PullbackGroupDirectionResult,
    _PHI_GROUP_DIRECTIONS,
    pullback_group_direction,
)


def _pullback_single_block(grad_phi, phi, generators, **kwargs):
    """Strict pullback direction on one block (the retired ambient 'pullback' mode's live equivalent)."""
    kwargs.setdefault("irrep_dims", [generators.shape[-1]])
    return pullback_group_direction(grad_phi, phi, generators, mode="pullback_per_block", **kwargs)


def test_pullback_group_registry_holds_only_the_per_block_direction():
    assert set(_PHI_GROUP_DIRECTIONS) == {"pullback_per_block"}


def _remove_test_phi_group_direction(name: str) -> None:
    _PHI_GROUP_DIRECTIONS.pop(name, None)
    for support_name in (
        "_PHI_GROUP_DIRECTION_ACCEPTS_COORDINATE_LAYOUT",
        "_PHI_GROUP_DIRECTION_REQUIRES_SOURCE_BASIS",
    ):
        support = getattr(phi_preconditioner, support_name, None)
        if support is not None:
            support.pop(name, None)


def test_pullback_group_dispatch_preserves_legacy_registered_callable_without_layout_keyword():
    name = "test_legacy_without_coordinate_layout"
    seen = {}

    @phi_preconditioner.register_phi_group_direction(name)
    def legacy_direction(grad_phi, phi, generators, *, irrep_dims=None):
        seen["irrep_dims"] = irrep_dims
        return phi_preconditioner._pullback_group_direction_per_block(
            grad_phi,
            phi,
            generators,
            irrep_dims=irrep_dims,
        )

    try:
        result = pullback_group_direction(
            torch.ones(4),
            torch.zeros(4),
            generate_glk_multihead(2, 1),
            mode=name,
            coordinate_layout="block_head_row_major",
            irrep_dims=[2],
        )
    finally:
        _remove_test_phi_group_direction(name)

    assert seen["irrep_dims"] == [2]
    assert torch.isfinite(result.v_phi).all()


def test_pullback_group_dispatch_forwards_layout_to_kwargs_callable_and_override():
    name = "test_kwargs_coordinate_layout"
    seen = []

    @phi_preconditioner.register_phi_group_direction(name)
    def initial_direction(grad_phi, phi, generators, *, irrep_dims=None):
        raise AssertionError("override did not replace initial registered direction")

    @phi_preconditioner.register_phi_group_direction(name, override=True)
    def kwargs_direction(grad_phi, phi, generators, *, irrep_dims=None, **kwargs):
        seen.append(kwargs.get("coordinate_layout"))
        return phi_preconditioner._pullback_group_direction_per_block(
            grad_phi,
            phi,
            generators,
            irrep_dims=irrep_dims,
        )

    try:
        result = pullback_group_direction(
            torch.ones(4),
            torch.zeros(4),
            generate_glk_multihead(2, 1),
            mode=name,
            coordinate_layout="block_head_row_major",
            irrep_dims=[2],
        )
    finally:
        _remove_test_phi_group_direction(name)

    assert seen == ["block_head_row_major"]
    assert torch.isfinite(result.v_phi).all()


def test_pullback_group_dispatch_does_not_keyword_call_positional_only_layout():
    name = "test_positional_only_coordinate_layout"
    seen = []

    @phi_preconditioner.register_phi_group_direction(name)
    def positional_layout_direction(
        grad_phi,
        phi,
        generators,
        coordinate_layout=None,
        /,
        *,
        irrep_dims=None,
    ):
        seen.append(coordinate_layout)
        return phi_preconditioner._pullback_group_direction_per_block(
            grad_phi,
            phi,
            generators,
            irrep_dims=irrep_dims,
        )

    try:
        result = pullback_group_direction(
            torch.ones(4),
            torch.zeros(4),
            generate_glk_multihead(2, 1),
            mode=name,
            coordinate_layout="block_head_row_major",
            irrep_dims=[2],
        )
    finally:
        _remove_test_phi_group_direction(name)

    assert seen == [None]
    assert torch.isfinite(result.v_phi).all()


def test_pullback_group_direction_returns_float64_gram_relative_zero_chart_solution():
    generators = generate_glk_multihead(2, 1)
    generators = generators * torch.tensor([0.5, 1.0, 2.0, 3.0]).view(-1, 1, 1)
    grad = torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.float32)
    phi = torch.zeros(4, dtype=torch.float32)
    input_dtypes = (grad.dtype, phi.dtype, generators.dtype)

    result = _pullback_single_block(grad, phi, generators)

    assert isinstance(result, PullbackGroupDirectionResult)
    gram = torch.einsum("aij,bij->ab", generators.double(), generators.double())
    expected = torch.linalg.solve((1.0 + 1e-6) * gram, grad.double())
    assert torch.allclose(result.v_phi, expected, rtol=1e-10, atol=1e-12)
    assert torch.allclose(result.xi, result.v_phi, rtol=1e-10, atol=1e-12)
    for value in (
        result.v_phi,
        result.xi,
        result.min_undamped_generalized_eigenvalue,
        result.undamped_generalized_condition,
        result.damped_generalized_condition,
        result.scaled_solve_residual,
    ):
        assert value.dtype == torch.float64
    assert (grad.dtype, phi.dtype, generators.dtype) == input_dtypes


def _augmented_matrix_exp_differential(
    X: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Float64 oracle for D exp_X[H] from an augmented matrix exponential."""
    zeros = torch.zeros_like(X)
    augmented = torch.cat(
        (
            torch.cat((X, H), dim=-1),
            torch.cat((zeros, X), dim=-1),
        ),
        dim=-2,
    )
    return torch.linalg.matrix_exp(augmented)[:X.shape[-1], X.shape[-1]:]


def _pullback_oracle(
    phi:       torch.Tensor,
    generators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return J, J^T J, and the left-trivialized coordinate differential."""
    X = torch.einsum("a,aij->ij", phi, generators)
    U = torch.linalg.matrix_exp(X)
    differential = torch.stack(
        [_augmented_matrix_exp_differential(X, H) for H in generators],
        dim=0,
    )
    metric = torch.einsum("aij,bij->ab", differential, differential)
    left_matrices = torch.linalg.solve(U, differential)
    gram = torch.einsum("aij,bij->ab", generators, generators)
    pairings = torch.einsum("bij,aij->ba", generators, left_matrices)
    left_coordinates = torch.linalg.solve(gram, pairings)
    return differential, metric, left_coordinates


def _strict_series_pullback(
    phi:       torch.Tensor,
    generators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the strict kernel's actual right differential and pullback metric."""
    structure, _ = phi_preconditioner._strict_structure_constants(generators)
    ad = torch.einsum("a,abc->cb", phi, structure)
    psi_right, _, _ = phi_preconditioner._adaptive_phi_differentials(ad)
    pushed = torch.einsum("ca,cij->aij", psi_right, generators)
    X = torch.einsum("a,aij->ij", phi, generators)
    differential = torch.einsum("aij,jk->aik", pushed, torch.linalg.matrix_exp(X))
    metric = torch.einsum("aij,bij->ab", differential, differential)
    return differential, metric


def _gl_chart_fixture(
    kind: str,
    norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "symmetric":
        X = torch.tensor(
            [[0.7, 0.2, -0.1], [0.2, -0.4, 0.15], [-0.1, 0.15, -0.3]],
            dtype=torch.float64,
        )
    elif kind == "random_nonnormal":
        X = torch.tensor(
            [[0.3, 0.8, -0.2], [-0.1, -0.2, 0.5], [0.05, -0.4, -0.1]],
            dtype=torch.float64,
        )
    elif kind == "jordan_like":
        X = torch.tensor(
            [[0.2, 1.0, 0.0], [0.0, 0.2, 1.0], [0.0, 0.0, -0.4]],
            dtype=torch.float64,
        )
    elif kind == "traceless_diagonal":
        X = torch.diag(torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0], dtype=torch.float64))
    else:
        raise AssertionError(f"unknown chart fixture {kind!r}")
    if norm == 0.0:
        X = torch.zeros_like(X)
    else:
        X = X * (norm / torch.linalg.matrix_norm(X))
    return X.reshape(-1), generate_glk_multihead(X.shape[-1], 1).double()


@pytest.mark.parametrize(
    "kind",
    ("symmetric", "random_nonnormal", "jordan_like", "traceless_diagonal"),
)
@pytest.mark.parametrize("norm", (0.0, 1.0, 3.0, 5.0))
def test_pullback_group_differentials_match_augmented_exp_oracle(kind, norm):
    phi, generators = _gl_chart_fixture(kind, norm)
    differential, metric, left_coordinates = _pullback_oracle(phi, generators)
    actual_differential, actual_metric = _strict_series_pullback(phi, generators)
    differential_error = (
        torch.linalg.matrix_norm(
            (actual_differential - differential).reshape(generators.shape[0], -1)
        )
        / torch.linalg.matrix_norm(differential.reshape(generators.shape[0], -1)).clamp_min(1.0)
    )
    direct_metric_error = (
        torch.linalg.matrix_norm(actual_metric - metric)
        / torch.linalg.matrix_norm(metric).clamp_min(1.0)
    )
    assert differential_error <= 1e-9
    assert direct_metric_error <= 1e-9

    gram = torch.einsum("aij,bij->ab", generators, generators)
    damped = metric + 1e-6 * gram
    n_gen = generators.shape[0]
    batched_phi = phi.expand(n_gen, -1).clone()

    inverse_result = _pullback_single_block(
        torch.eye(n_gen, dtype=torch.float64),
        batched_phi,
        generators,
    )
    recovered_damped = torch.linalg.solve(
        inverse_result.v_phi,
        torch.eye(n_gen, dtype=torch.float64),
    )
    recovered_metric = recovered_damped - 1e-6 * gram
    metric_error = (
        torch.linalg.matrix_norm(recovered_metric - metric)
        / torch.linalg.matrix_norm(metric).clamp_min(1.0)
    )
    assert metric_error <= 1e-9

    differential_result = _pullback_single_block(
        damped,
        batched_phi,
        generators,
    )
    direction_error = torch.linalg.matrix_norm(
        differential_result.v_phi - torch.eye(n_gen, dtype=torch.float64)
    ) / math.sqrt(n_gen)
    assert direction_error <= 1e-8
    left_error = (
        torch.linalg.matrix_norm(differential_result.xi.transpose(-1, -2) - left_coordinates)
        / torch.linalg.matrix_norm(left_coordinates).clamp_min(1.0)
    )
    assert left_error <= 1e-9
    assert torch.isfinite(differential).all()


def test_pullback_group_adaptive_series_accepts_traceless_gl5_norm_five():
    phi, generators = _gl_chart_fixture("traceless_diagonal", 5.0)
    result = _pullback_single_block(
        torch.linspace(-0.5, 0.5, generators.shape[0], dtype=torch.float64),
        phi,
        generators,
    )
    # Candidate grid moved from (40, step 8) to (8, step 4) -- audit 2026-08-07. The certificate
    # that bounds the error is unchanged; only how early it is TESTED moved.
    assert result.series_order in range(8, 129, 4)
    assert all(
        torch.isfinite(value).all()
        for value in (
            result.v_phi,
            result.xi,
            result.min_undamped_generalized_eigenvalue,
            result.undamped_generalized_condition,
            result.damped_generalized_condition,
            result.scaled_solve_residual,
        )
    )


def test_pullback_group_adaptive_series_rejects_uncertified_chart():
    generators = generate_glk_multihead(2, 1).double()
    phi = torch.tensor([100.0, 0.0, 0.0, -100.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="series|certificate|certif"):
        _pullback_single_block(
            torch.ones(4, dtype=torch.float64),
            phi,
            generators,
        )


def test_pullback_group_direction_matches_cholesky_oracle_and_residual():
    generators = generate_glk_multihead(2, 1).double()
    generators = generators * torch.tensor([0.5, 1.0, 1.5, 2.0]).view(-1, 1, 1)
    phi = torch.tensor([0.1, -0.2, 0.15, -0.05], dtype=torch.float64)
    grad = torch.tensor([0.7, -0.3, 0.2, 0.9], dtype=torch.float64)
    _, metric, _ = _pullback_oracle(phi, generators)
    gram = torch.einsum("aij,bij->ab", generators, generators)
    damped = 0.5 * (metric + metric.transpose(-1, -2)) + 1e-6 * gram
    factor, info = torch.linalg.cholesky_ex(damped)
    assert not bool((info != 0).any())
    expected = torch.cholesky_solve(grad.unsqueeze(-1), factor).squeeze(-1)

    result = _pullback_single_block(grad, phi, generators)

    relative_error = (
        torch.linalg.vector_norm(result.v_phi - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1.0)
    )
    assert relative_error <= 1e-8
    assert result.scaled_solve_residual <= 1e-10


def _gl2_rotation_chart(theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    generators = generate_glk_multihead(2, 1).double()
    phi = torch.tensor([0.0, -theta, theta, 0.0], dtype=torch.float64)
    return phi, generators


def test_pullback_group_rejects_gl2_rotation_pi_singularity_before_ridge():
    phi, generators = _gl2_rotation_chart(math.pi)
    with pytest.raises(ValueError, match="undamped generalized eigenvalue"):
        _pullback_single_block(
            torch.ones(4, dtype=torch.float64),
            phi,
            generators,
        )


def test_pullback_group_undamped_regularity_threshold_is_stable():
    failing_phi, generators = _gl2_rotation_chart(math.pi - 3e-4)
    with pytest.raises(ValueError, match="undamped generalized eigenvalue"):
        _pullback_single_block(
            torch.ones(4, dtype=torch.float64),
            failing_phi,
            generators,
        )

    passing_phi, _ = _gl2_rotation_chart(math.pi - 4.5e-4)
    result = _pullback_single_block(
        torch.ones(4, dtype=torch.float64),
        passing_phi,
        generators,
    )
    assert result.min_undamped_generalized_eigenvalue >= 1e-8
    assert result.damped_generalized_condition <= 1e6


def test_pullback_group_direction_rejects_damped_condition_above_limit():
    generators = generate_glk_multihead(2, 1).double()
    phi = torch.tensor(
        [5.0 / math.sqrt(2.0), 0.0, 0.0, -5.0 / math.sqrt(2.0)],
        dtype=torch.float64,
    )
    _, metric, _ = _pullback_oracle(phi, generators)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (metric + metric.transpose(-1, -2)))
    assert eigenvalues[0] >= 1e-8
    assert (eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6) > 1e6
    with pytest.raises(ValueError, match="damped generalized condition"):
        _pullback_single_block(
            torch.ones(4, dtype=torch.float64),
            phi,
            generators,
        )


def test_pullback_group_direction_rejects_scaled_solve_residual_above_limit(monkeypatch):
    original = torch.cholesky_solve
    call_count = 0

    def inaccurate_solve(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = original(*args, **kwargs)
        if call_count == 2:
            result = result + 1e-4
        return result

    monkeypatch.setattr(torch, "cholesky_solve", inaccurate_solve)
    with pytest.raises(ValueError, match="scaled solve residual"):
        _pullback_single_block(
            torch.ones(4, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            generate_glk_multihead(2, 1).double(),
        )


def test_pullback_group_left_trivialization_satisfies_u_xi_equals_dexp_v():
    generators = generate_glk_multihead(2, 1).double()
    phi = torch.tensor([0.25, -0.4, 0.15, -0.2], dtype=torch.float64)
    grad = torch.tensor([0.6, -0.2, 0.3, 0.8], dtype=torch.float64)
    result = _pullback_single_block(grad, phi, generators)
    X = torch.einsum("a,aij->ij", phi, generators)
    H_v = torch.einsum("a,aij->ij", result.v_phi, generators)
    U = torch.linalg.matrix_exp(X)
    target = _augmented_matrix_exp_differential(X, H_v)
    actual = U @ torch.einsum("a,aij->ij", result.xi, generators)
    opposite_sign = U @ torch.einsum("a,aij->ij", -result.xi, generators)
    relative_residual = (
        torch.linalg.matrix_norm(actual - target)
        / torch.linalg.matrix_norm(target).clamp_min(1.0)
    )
    opposite_residual = (
        torch.linalg.matrix_norm(opposite_sign - target)
        / torch.linalg.matrix_norm(target).clamp_min(1.0)
    )
    assert relative_residual <= 1e-9
    assert opposite_residual > 1e-3


def test_pullback_group_per_block_matches_two_independent_gl5_solves_without_full_structure_constants(
    monkeypatch,
):
    torch.manual_seed(20260717)
    generators = generate_glk_multihead(10, 2).double()
    grad = torch.randn(50, dtype=torch.float64)
    phi = 0.03 * torch.randn(50, dtype=torch.float64)
    local_generators = generate_glk_multihead(5, 1).double()
    expected = [
        _pullback_single_block(
            grad[start:start + 25],
            phi[start:start + 25],
            local_generators,
        )
        for start in (0, 25)
    ]
    original = phi_preconditioner._strict_structure_constants
    seen_dimensions = []

    def reject_full_k10(local_basis, **kwargs):
        dimension = local_basis.shape[-1]
        seen_dimensions.append(dimension)
        if dimension == 10:
            raise AssertionError("full K10 structure constants must not be constructed")
        return original(local_basis, **kwargs)

    monkeypatch.setattr(phi_preconditioner, "_strict_structure_constants", reject_full_k10)
    result = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[5, 5],
    )

    assert seen_dimensions == [5, 5]
    assert torch.allclose(result.v_phi, torch.cat([item.v_phi for item in expected]), atol=1e-10)
    assert torch.allclose(result.xi, torch.cat([item.xi for item in expected]), atol=1e-10)
    min_undamped = torch.stack(
        [item.min_undamped_generalized_eigenvalue for item in expected]
    ).min(dim=0).values
    max_undamped = torch.stack(
        [
            item.min_undamped_generalized_eigenvalue
            * item.undamped_generalized_condition
            for item in expected
        ]
    ).max(dim=0).values
    min_damped = torch.stack(
        [item.min_undamped_generalized_eigenvalue + 1e-6 for item in expected]
    ).min(dim=0).values
    max_damped = torch.stack(
        [
            (item.min_undamped_generalized_eigenvalue + 1e-6)
            * item.damped_generalized_condition
            for item in expected
        ]
    ).max(dim=0).values
    assert torch.allclose(result.min_undamped_generalized_eigenvalue, min_undamped)
    assert torch.allclose(result.undamped_generalized_condition, max_undamped / min_undamped)
    assert torch.allclose(result.damped_generalized_condition, max_damped / min_damped)
    assert torch.allclose(
        result.scaled_solve_residual,
        torch.stack([item.scaled_solve_residual for item in expected]).max(dim=0).values,
    )
    assert result.series_order == max(item.series_order for item in expected)


def test_pullback_group_rejects_nonclosed_generator_span():
    generators = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="bracket-closed"):
        _pullback_single_block(
            torch.ones(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            generators,
        )


def test_pullback_group_direction_disables_autocast_in_strict_float64_island(monkeypatch):
    original = phi_preconditioner._strict_structure_constants
    autocast_states = []

    def record_autocast_state(local_basis, **kwargs):
        autocast_states.append(torch.is_autocast_enabled(local_basis.device.type))
        assert local_basis.dtype == torch.float64
        return original(local_basis, **kwargs)

    monkeypatch.setattr(phi_preconditioner, "_strict_structure_constants", record_autocast_state)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _pullback_single_block(
            torch.ones(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
            generate_glk_multihead(2, 1),
        )
    assert autocast_states == [False]
    assert result.v_phi.dtype == torch.float64


def _two_recurrence_differential_oracle(
    ad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Frozen pre-optimization right/left differential recurrence."""
    n_gen = ad.shape[-1]
    eye = torch.eye(n_gen, dtype=ad.dtype, device=ad.device).expand_as(ad).clone()
    psi_right = eye.clone()
    psi_left = eye.clone()
    term_right = eye.clone()
    term_left = eye.clone()
    norm_one = torch.linalg.matrix_norm(ad, ord=1, dim=(-2, -1))
    norm_inf = torch.linalg.matrix_norm(ad, ord=float("inf"), dim=(-2, -1))
    use_one = norm_one <= norm_inf
    alpha = torch.where(use_one, norm_one, norm_inf)
    bound_k = torch.ones_like(alpha)
    for k in range(1, phi_preconditioner._PHI_GROUP_MAX_SERIES_ORDER):
        term_right = torch.matmul(term_right, ad) / float(k + 1)
        term_left = -torch.matmul(term_left, ad) / float(k + 1)
        psi_right = psi_right + term_right
        psi_left = psi_left + term_left
        bound_k = bound_k * alpha / float(k + 1)
        order = k + 1
        if order < phi_preconditioner._PHI_GROUP_MIN_SERIES_ORDER or (
            order % phi_preconditioner._PHI_GROUP_SERIES_ORDER_STEP
        ):
            continue
        first_omitted = bound_k * alpha / float(order + 1)
        ratio = alpha / float(order + 2)
        tail = torch.where(
            ratio < 1.0,
            first_omitted / (1.0 - ratio),
            torch.full_like(ratio, torch.inf),
        )
        right_one = torch.linalg.matrix_norm(psi_right, ord=1, dim=(-2, -1))
        right_inf = torch.linalg.matrix_norm(psi_right, ord=float("inf"), dim=(-2, -1))
        left_one = torch.linalg.matrix_norm(psi_left, ord=1, dim=(-2, -1))
        left_inf = torch.linalg.matrix_norm(psi_left, ord=float("inf"), dim=(-2, -1))
        right_norm = torch.where(use_one, right_one, right_inf)
        left_norm = torch.where(use_one, left_one, left_inf)
        right_ok = tail <= phi_preconditioner._PHI_GROUP_TAIL_TOL * torch.maximum(
            torch.ones_like(tail), right_norm
        )
        left_ok = tail <= phi_preconditioner._PHI_GROUP_TAIL_TOL * torch.maximum(
            torch.ones_like(tail), left_norm
        )
        if bool((right_ok & left_ok & torch.isfinite(tail)).all()):
            return psi_right, psi_left, order
    raise AssertionError("oracle series did not converge")


def test_pullback_group_differentials_reuse_one_positive_power_recurrence(monkeypatch, device):
    phi, generators = _gl_chart_fixture("random_nonnormal", 3.0)
    phi = phi.to(device)
    generators = generators.to(device)
    structure, _ = phi_preconditioner._strict_structure_constants(generators)
    ad = torch.einsum("a,abc->cb", phi, structure)
    expected_right, expected_left, expected_order = _two_recurrence_differential_oracle(ad)
    original = torch.matmul
    matmul_calls = 0

    def count_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "matmul", count_matmul)
    actual_right, actual_left, actual_order = phi_preconditioner._adaptive_phi_differentials(ad)

    assert actual_order == expected_order
    assert torch.equal(actual_right, expected_right)
    assert torch.equal(actual_left, expected_left)
    assert matmul_calls == actual_order - 1


@pytest.mark.parametrize("orthonormal", (True, False))
def test_pullback_group_solve_scale_reuses_eigenvalue_only_for_identity_gram(
    monkeypatch,
    orthonormal,
    device,
):
    phi_preconditioner._STRICT_PULLBACK_PREP_CACHE.clear()
    generators = generate_glk_multihead(2, 1, device=device).double()
    if not orthonormal:
        generators = generators * torch.tensor(
            [0.5, 1.0, 1.5, 2.0],
            device=device,
        ).view(-1, 1, 1)
    grad = torch.tensor([0.7, -0.3, 0.2, 0.9], dtype=torch.float64, device=device)
    phi = torch.tensor([0.1, -0.2, 0.15, -0.05], dtype=torch.float64, device=device)
    original = torch.linalg.matrix_norm
    original_solve = torch.cholesky_solve
    spectral_calls = 0
    solve_calls = 0

    def count_spectral(*args, **kwargs):
        nonlocal spectral_calls
        if kwargs.get("ord") == 2:
            spectral_calls += 1
        return original(*args, **kwargs)

    def perturb_direction_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        result = original_solve(*args, **kwargs)
        if solve_calls == 2:
            perturbation = torch.linspace(
                1.0,
                2.0,
                result.numel(),
                dtype=result.dtype,
                device=result.device,
            ).reshape_as(result)
            result = result + 1e-12 * perturbation
        return result

    monkeypatch.setattr(torch.linalg, "matrix_norm", count_spectral)
    monkeypatch.setattr(torch, "cholesky_solve", perturb_direction_solve)
    result = _pullback_single_block(grad, phi, generators)

    _, metric = _strict_series_pullback(phi, generators)
    gram = torch.einsum("aij,bij->ab", generators, generators)
    damped = 0.5 * (metric + metric.transpose(-1, -2)) + 1e-6 * gram
    residual = torch.linalg.vector_norm(damped @ result.v_phi - grad)
    expected_scale = (
        original(damped, ord=2)
        * torch.linalg.vector_norm(result.v_phi)
        + torch.linalg.vector_norm(grad)
    )
    expected = residual / expected_scale
    assert torch.allclose(result.scaled_solve_residual, expected, rtol=1e-12, atol=1e-18)
    assert spectral_calls == (0 if orthonormal else 1)


def test_pullback_group_strict_preparation_cache_hits_and_invalidates_on_mutation(monkeypatch):
    phi_preconditioner._STRICT_PULLBACK_PREP_CACHE.clear()
    generators = generate_glk_multihead(2, 1)
    original = phi_preconditioner._strict_structure_constants
    calls = 0

    def count_builds(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(phi_preconditioner, "_strict_structure_constants", count_builds)
    grad = torch.ones(4)
    phi = torch.zeros(4)
    _pullback_single_block(grad, phi, generators)
    _pullback_single_block(grad, phi, generators)
    generators.add_(0.0)
    _pullback_single_block(grad, phi, generators)

    assert calls == 2


def test_pullback_group_strict_preparation_cache_is_bounded_and_weak():
    phi_preconditioner._STRICT_PULLBACK_PREP_CACHE.clear()
    generators_live = [
        generate_glk_multihead(1, 1)
        for _ in range(phi_preconditioner._STRICT_PULLBACK_PREP_CACHE_MAXSIZE + 3)
    ]
    for generators in generators_live:
        _pullback_single_block(
            torch.ones(1),
            torch.zeros(1),
            generators,
        )
    assert len(phi_preconditioner._STRICT_PULLBACK_PREP_CACHE) == (
        phi_preconditioner._STRICT_PULLBACK_PREP_CACHE_MAXSIZE
    )
    cached_sources = [
        generators_ref()
        for generators_ref, _ in phi_preconditioner._STRICT_PULLBACK_PREP_CACHE.values()
    ]
    assert all(source is not generators_live[0] for source in cached_sources)
    assert any(source is generators_live[-1] for source in cached_sources)
    generators_refs = [weakref.ref(source) for source in generators_live]
    del cached_sources
    del generators
    generators_live.clear()
    gc.collect()
    assert all(generators_ref() is None for generators_ref in generators_refs)
    assert not phi_preconditioner._STRICT_PULLBACK_PREP_CACHE


def test_pullback_group_strict_preparation_cache_bypasses_requires_grad(monkeypatch):
    phi_preconditioner._STRICT_PULLBACK_PREP_CACHE.clear()
    generators = generate_glk_multihead(2, 1).requires_grad_()
    original = phi_preconditioner._strict_structure_constants
    calls = 0

    def count_builds(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(phi_preconditioner, "_strict_structure_constants", count_builds)
    for _ in range(2):
        result = _pullback_single_block(
            torch.ones(4),
            torch.tensor([0.1, -0.2, 0.15, -0.05]),
            generators,
        )
    assert calls == 2
    assert not phi_preconditioner._STRICT_PULLBACK_PREP_CACHE
    objective = result.v_phi.square().sum() + 0.25 * result.xi.square().sum()
    objective.backward()
    assert generators.grad is not None
    assert torch.isfinite(generators.grad).all()
    assert torch.count_nonzero(generators.grad) > 0


def test_pullback_group_block_head_layout_avoids_generic_membership_scan(monkeypatch, device):
    generators = generate_glk_multihead(4, 2, device=device)
    grad = torch.linspace(-0.5, 0.5, generators.shape[0], device=device)
    phi = torch.linspace(-0.1, 0.1, generators.shape[0], device=device)
    expected = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
    )
    monkeypatch.setattr(
        phi_preconditioner,
        "_generator_block_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic scan called")),
    )
    actual = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
        coordinate_layout="block_head_row_major",
    )
    assert torch.equal(actual.v_phi, expected.v_phi)
    assert torch.equal(actual.xi, expected.xi)
    assert actual.series_order == expected.series_order


def _assert_pullback_group_result_equal(
    actual:   PullbackGroupDirectionResult,
    expected: PullbackGroupDirectionResult,
) -> None:
    assert torch.equal(actual.v_phi, expected.v_phi)
    assert torch.equal(actual.xi, expected.xi)
    assert torch.equal(
        actual.min_undamped_generalized_eigenvalue,
        expected.min_undamped_generalized_eigenvalue,
    )
    assert torch.equal(
        actual.undamped_generalized_condition,
        expected.undamped_generalized_condition,
    )
    assert torch.equal(
        actual.damped_generalized_condition,
        expected.damped_generalized_condition,
    )
    assert torch.equal(actual.scaled_solve_residual, expected.scaled_solve_residual)
    assert actual.series_order == expected.series_order


def test_pullback_group_equal_block_layout_batches_one_exact_strict_solve(monkeypatch, device):
    generators = generate_glk_multihead(4, 2, device=device)
    grad = torch.linspace(-0.5, 0.5, 3 * generators.shape[0], device=device).reshape(3, -1)
    phi = torch.linspace(-0.1, 0.1, 3 * generators.shape[0], device=device).reshape(3, -1)
    expected = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
    )
    original = phi_preconditioner._full_pullback_group_direction
    strict_solve_calls = 0

    def count_strict_solves(*args, **kwargs):
        nonlocal strict_solve_calls
        strict_solve_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        phi_preconditioner,
        "_full_pullback_group_direction",
        count_strict_solves,
    )
    actual = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        coordinate_layout="block_head_row_major",
        irrep_dims=[2, 2],
    )

    _assert_pullback_group_result_equal(actual, expected)
    assert strict_solve_calls == 1


def test_pullback_group_layout_nonidentical_local_bases_fall_back_to_independent_solves(
    monkeypatch,
    device,
):
    generators = generate_glk_multihead(4, 2, device=device)
    permutation = torch.tensor([0, 1, 2, 3, 5, 4, 6, 7], device=device)
    generators = generators[permutation]
    grad = torch.linspace(-0.5, 0.5, generators.shape[0], device=device)
    phi = torch.linspace(-0.1, 0.1, generators.shape[0], device=device)
    expected = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
    )
    original = phi_preconditioner._full_pullback_group_direction
    strict_solve_calls = 0

    def count_strict_solves(*args, **kwargs):
        nonlocal strict_solve_calls
        strict_solve_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        phi_preconditioner,
        "_full_pullback_group_direction",
        count_strict_solves,
    )
    actual = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        coordinate_layout="block_head_row_major",
        irrep_dims=[2, 2],
    )

    _assert_pullback_group_result_equal(actual, expected)
    assert strict_solve_calls == 2


def test_pullback_group_equal_blocks_with_mixed_series_orders_fall_back_exactly(
    monkeypatch,
    device,
):
    generators = generate_glk_multihead(4, 2, device=device).double()
    block_40 = torch.tensor([0.0, -1.0, 1.0, 0.0], dtype=torch.float64, device=device)
    block_48 = torch.tensor([0.0, -4.5, 4.5, 0.0], dtype=torch.float64, device=device)
    phi_row = torch.cat((block_40, block_48))
    phi = torch.stack((phi_row, phi_row), dim=0)
    grad = torch.linspace(-0.5, 0.5, 2 * generators.shape[0], device=device).reshape(2, -1)
    expected = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
    )
    # 44 on the (8, step 4) candidate grid; was 48 when the grid started at 40 in steps of 8.
    # The point of the fixture is unchanged: the two blocks certify at DIFFERENT orders.
    assert expected.series_order == 44
    original = phi_preconditioner._full_pullback_group_direction
    strict_solve_calls = 0

    def count_strict_solves(*args, **kwargs):
        nonlocal strict_solve_calls
        strict_solve_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        phi_preconditioner,
        "_full_pullback_group_direction",
        count_strict_solves,
    )
    actual = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        coordinate_layout="block_head_row_major",
        irrep_dims=[2, 2],
    )

    _assert_pullback_group_result_equal(actual, expected)
    assert strict_solve_calls == 3


def test_pullback_group_equal_block_batch_preserves_sequential_exception_precedence(device):
    generators = generate_glk_multihead(4, 2, device=device).double()
    first_fails_regularity = torch.tensor(
        [0.0, -math.pi, math.pi, 0.0],
        dtype=torch.float64,
        device=device,
    )
    later_fails_series = torch.tensor(
        [0.0, -30.0, 30.0, 0.0],
        dtype=torch.float64,
        device=device,
    )
    phi = torch.cat((first_fails_regularity, later_fails_series)).unsqueeze(0)
    grad = torch.ones_like(phi)
    with pytest.raises(ValueError) as expected_error:
        pullback_group_direction(
            grad,
            phi,
            generators,
            mode="pullback_per_block",
            irrep_dims=[2, 2],
        )
    with pytest.raises(ValueError) as actual_error:
        pullback_group_direction(
            grad,
            phi,
            generators,
            mode="pullback_per_block",
            coordinate_layout="block_head_row_major",
            irrep_dims=[2, 2],
        )
    assert str(actual_error.value) == str(expected_error.value)


def test_pullback_group_equal_block_batch_bypasses_requires_grad_basis(monkeypatch, device):
    source = generate_glk_multihead(4, 2, device=device).double()
    grad = torch.linspace(-0.5, 0.5, 2 * source.shape[0], device=device).reshape(2, -1)
    phi = torch.linspace(-0.1, 0.1, 2 * source.shape[0], device=device).reshape(2, -1)
    generic_generators = source.clone().requires_grad_()
    expected = pullback_group_direction(
        grad,
        phi,
        generic_generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
    )
    expected_objective = expected.v_phi.square().sum() + 0.25 * expected.xi.square().sum()
    expected_objective.backward()
    expected_gradient = generic_generators.grad.detach().clone()

    layout_generators = source.clone().requires_grad_()
    original = phi_preconditioner._full_pullback_group_direction
    strict_solve_calls = 0

    def count_strict_solves(*args, **kwargs):
        nonlocal strict_solve_calls
        strict_solve_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        phi_preconditioner,
        "_full_pullback_group_direction",
        count_strict_solves,
    )
    actual = pullback_group_direction(
        grad,
        phi,
        layout_generators,
        mode="pullback_per_block",
        coordinate_layout="block_head_row_major",
        irrep_dims=[2, 2],
    )
    actual_objective = actual.v_phi.square().sum() + 0.25 * actual.xi.square().sum()
    actual_objective.backward()

    _assert_pullback_group_result_equal(actual, expected)
    assert strict_solve_calls == 2
    assert layout_generators.grad is not None
    assert torch.equal(layout_generators.grad, expected_gradient)


def test_pullback_group_equal_block_batch_falls_back_for_unmatched_leading_shapes(device):
    generators = generate_glk_multihead(4, 2, device=device)
    grad = torch.linspace(-0.5, 0.5, 3 * generators.shape[0], device=device).reshape(3, -1)
    phi = torch.linspace(-0.1, 0.1, generators.shape[0], device=device)
    expected = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        irrep_dims=[2, 2],
    )
    actual = pullback_group_direction(
        grad,
        phi,
        generators,
        mode="pullback_per_block",
        coordinate_layout="block_head_row_major",
        irrep_dims=[2, 2],
    )
    _assert_pullback_group_result_equal(actual, expected)
