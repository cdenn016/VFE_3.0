"""Exact common-frame Gaussian scores preserve the public coupling contract."""

from __future__ import annotations

from collections.abc import Iterator
from math import prod
from typing import Any

import pytest
import torch

from vfe3.families.gaussian import FullGaussian
from vfe3.geometry.transport import (
    CompactFactoredTransport, FactoredTransport,
    set_full_cov_congruence_certification, set_full_cov_congruence_precision,
)


@pytest.fixture(autouse=True)
def _tiny_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    previous_precision = set_full_cov_congruence_precision("fp64")
    previous_certification = set_full_cov_congruence_certification(True)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    set_full_cov_congruence_precision(previous_precision)
    set_full_cov_congruence_certification(previous_certification)


def _inputs(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(704)

    def _random(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return scale * torch.randn(shape, generator=generator, dtype=dtype)

    def _covariance() -> torch.Tensor:
        raw = _random((1, 3, 4, 4), 0.25)
        return (raw @ raw.transpose(-1, -2) + 0.8 * torch.eye(4, dtype=dtype)).requires_grad_()

    return (
        _random((1, 3, 4), 0.3).requires_grad_(),
        _covariance(),
        _random((1, 3, 4), 0.3).requires_grad_(),
        _covariance(),
        _random((1, 3, 2, 2, 2), 0.08).requires_grad_(),
    )


def _dense_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Assemble per-token block frames without a pairwise axis."""
    zero = torch.zeros_like(blocks[..., 0, :, :])
    return torch.cat((
        torch.cat((blocks[..., 0, :, :], zero), dim=-1),
        torch.cat((zero, blocks[..., 1, :, :]), dim=-1),
    ), dim=-2)


def _transport(
    phi:           torch.Tensor,
    compact:       bool = True,
    trusted:       bool = True,
    detach_key:    bool = False,
) -> CompactFactoredTransport | FactoredTransport:
    frames = torch.matrix_exp(phi)
    inverse = torch.linalg.inv(frames.detach() if detach_key else frames)
    if compact:
        return CompactFactoredTransport(frames, inverse, 4, same_frame_flat_cocycle=trusted)
    return FactoredTransport(_dense_blocks(frames), _dense_blocks(inverse), [2, 2],
                             same_frame_flat_cocycle=trusted)


def _score(
    inputs:        tuple[torch.Tensor, ...],
    backend:       str = "common_frame",
    compact:       bool = True,
    **kwargs:      Any,
) -> torch.Tensor:
    mu_q, Sigma_q, mu_k, Sigma_k, phi = inputs
    options = dict(irrep_dims=[2, 2], precision_policy="fp64", diagonal_out=False)
    options.update(kwargs)
    omega = options.pop("omega", None)
    if omega is None:
        omega = _transport(phi, compact=compact)
    return FullGaussian.coupling_energy(mu_q, Sigma_q, mu_k, Sigma_k, omega,
                                        score_backend=backend, **options)


def _weighted(energy: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.2, 1.1, energy.numel(), dtype=energy.dtype).reshape_as(energy)
    return (weights * energy).sum()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("compact", [True, False])
def test_common_frame_backend_matches_pairwise_score_and_first_gradients(
    dtype:         torch.dtype,
    compact:       bool,
) -> None:
    """Compare complete KL, including receiver covariance entropy derivatives."""
    inputs = _inputs(dtype)
    reference = _score(inputs, "pairwise", compact)
    actual = _score(inputs, "common_frame", compact)
    tolerance = 2e-9 if dtype == torch.float64 else 2e-5
    assert actual.shape == (1, 2, 3, 3)
    assert actual.dtype == reference.dtype == dtype
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    grad_reference = torch.autograd.grad(_weighted(reference), inputs)
    grad_actual = torch.autograd.grad(_weighted(actual), inputs)
    for observed, expected in zip(grad_actual, grad_reference):
        torch.testing.assert_close(observed, expected, rtol=tolerance, atol=tolerance)


def test_common_frame_preserves_hessian_vector_through_covariances_and_frames() -> None:
    inputs = _inputs()
    generator = torch.Generator().manual_seed(19)
    vectors = tuple(torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in inputs)
    hessians = []
    for backend in ("pairwise", "common_frame"):
        gradients = torch.autograd.grad(_weighted(_score(inputs, backend)), inputs, create_graph=True)
        directional = sum((gradient * vector).sum() for gradient, vector in zip(gradients, vectors))
        hessians.append(torch.autograd.grad(directional, inputs))
    for observed, expected in zip(hessians[1], hessians[0]):
        torch.testing.assert_close(observed, expected, rtol=3e-8, atol=3e-8)


def test_common_frame_gradient_matches_symmetric_finite_difference() -> None:
    """Check the continuous SPD/Lie chart, independently of legacy autodiff."""
    inputs = _inputs()
    vectors = []
    generator = torch.Generator().manual_seed(821)
    for index, value in enumerate(inputs):
        direction = torch.randn(value.shape, dtype=value.dtype, generator=generator) * 0.1
        if index in (1, 3):
            direction = 0.5 * (direction + direction.transpose(-1, -2))
        vectors.append(direction)
    gradients = torch.autograd.grad(_weighted(_score(inputs)), inputs)
    analytic = sum((gradient * vector).sum() for gradient, vector in zip(gradients, vectors))
    step = 1e-5
    plus = tuple(value + step * vector for value, vector in zip(inputs, vectors))
    minus = tuple(value - step * vector for value, vector in zip(inputs, vectors))
    numerical = (_weighted(_score(plus)) - _weighted(_score(minus))) / (2 * step)
    torch.testing.assert_close(analytic, numerical, rtol=2e-7, atol=2e-7)


def test_common_frame_preserves_detached_filtering_keys() -> None:
    inputs = _inputs()
    mu_q, Sigma_q, mu_k, Sigma_k, phi = inputs
    detached_inputs = (mu_q, Sigma_q, mu_k.detach(), Sigma_k.detach(), phi)
    gradients = []
    for backend in ("pairwise", "common_frame"):
        energy = _score(detached_inputs, backend, omega=_transport(phi, detach_key=True))
        gradients.append(torch.autograd.grad(_weighted(energy), inputs, allow_unused=True))
    assert gradients[0][2:4] == gradients[1][2:4] == (None, None)
    for index in (0, 1, 4):
        torch.testing.assert_close(gradients[1][index], gradients[0][index], rtol=2e-9, atol=2e-9)


def test_common_frame_preserves_kl_cap_and_saturated_gradients() -> None:
    inputs = _inputs()
    outputs = []
    gradients = []
    for backend in ("pairwise", "common_frame"):
        value = _score(inputs, backend, kl_max=0.15)
        outputs.append(value)
        gradients.append(torch.autograd.grad(_weighted(value), inputs))
    assert torch.any(outputs[1] == 0.15)
    assert torch.any(outputs[1] < 0.15)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-9, atol=2e-9)
    for observed, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(observed, expected, rtol=2e-9, atol=2e-9)


def test_common_frame_support_masks_energy_and_gradient() -> None:
    inputs = _inputs()
    support = torch.ones(3, 3, dtype=torch.bool).tril().expand(1, 2, 3, 3)
    reference = _score(inputs, "pairwise", pair_support=support)
    actual = _score(inputs, "common_frame", pair_support=support)
    torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-9)
    assert torch.count_nonzero(actual[~support]) == 0
    gradients = torch.autograd.grad(actual[~support].sum(), inputs)
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


@pytest.mark.parametrize("compact", [True, False])
def test_common_frame_factors_only_token_matrices(
    compact:       bool,
    monkeypatch:   pytest.MonkeyPatch,
) -> None:
    """Record actual linear-algebra operand shapes without changing any results."""
    inputs = _inputs()
    omega = _transport(inputs[-1], compact=compact)
    calls: list[tuple[str, tuple[int, ...]]] = []

    def _record(name: str, operation: Any) -> Any:
        def _wrapped(matrix: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            calls.append((name, tuple(matrix.shape)))
            return operation(matrix, *args, **kwargs)
        return _wrapped

    for name in ("cholesky", "cholesky_ex", "solve", "solve_ex", "solve_triangular", "inv", "inv_ex", "slogdet"):
        operation = getattr(torch.linalg, name)
        monkeypatch.setattr(torch.linalg, name, _record(name, operation))
    output = _score(inputs, compact=compact, omega=omega)
    assert torch.isfinite(output).all()
    assert calls
    assert any("cholesky" in name for name, _ in calls)
    for name, shape in calls:
        assert prod(shape[:-2]) <= 6, f"{name} factored a pair grid: {shape}"


@pytest.mark.parametrize("case", ["untrusted", "dense", "renyi", "other_functional", "head_partition"])
def test_ineligible_common_frame_calls_preserve_pairwise_fallback(case: str) -> None:
    inputs = _inputs()
    options: dict[str, Any] = {}
    if case == "untrusted":
        options["omega"] = _transport(inputs[-1], trusted=False)
    elif case == "dense":
        options["omega"] = _transport(inputs[-1]).to_dense_omega()
    elif case == "renyi":
        options["alpha"] = 0.7
    elif case == "other_functional":
        options["divergence_family"] = "jeffreys"
    elif case == "head_partition":
        options["irrep_dims"] = [4]
    reference = _score(inputs, "pairwise", **options)
    actual = _score(inputs, "common_frame", **options)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_common_frame_preserves_independent_receiver_and_sender_frame_roles() -> None:
    inputs = _inputs()
    phi_q = inputs[-1]
    phi_k = phi_q.detach().clone().requires_grad_()
    leaves = (*inputs, phi_k)
    gradients = []
    for backend in ("pairwise", "common_frame"):
        omega = CompactFactoredTransport(
            torch.matrix_exp(phi_q), torch.linalg.inv(torch.matrix_exp(phi_k)),
            4, same_frame_flat_cocycle=True)
        energy = _score(inputs, backend, omega=omega)
        gradients.append(torch.autograd.grad(_weighted(energy), leaves))
    for observed, expected in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(observed, expected, rtol=2e-9, atol=2e-9)


def test_common_frame_numerical_fallback_preserves_unused_future_pair_support() -> None:
    from tests.test_causal_congruence_support_20260904 import _future_pair_fixture

    mu, Sigma, omega, prior = _future_pair_fixture(torch.device("cpu"))
    support = ~torch.isneginf(prior)
    inputs = (mu, Sigma, omega.exp_blocks, omega.inv_blocks)
    previous = set_full_cov_congruence_precision("fp32_escalate")
    energies, gradients, outer_gradients = [], [], []
    try:
        for backend in ("pairwise", "common_frame"):
            energy = FullGaussian.coupling_energy(
                mu, Sigma, mu.detach(), Sigma.detach(), omega,
                irrep_dims=[2, 2], diagonal_out=False, precision_policy="fp32_escalate",
                score_backend=backend, pair_support=support)
            energies.append(energy)
            gradient = torch.autograd.grad(energy.sum(), inputs, create_graph=True)
            gradients.append(gradient)
            outer_gradients.append(torch.autograd.grad(
                sum(value.square().sum() for value in gradient), inputs))
    finally:
        set_full_cov_congruence_precision(previous)
    torch.testing.assert_close(energies[1], energies[0], rtol=0, atol=0)
    assert torch.count_nonzero(energies[1][~support.expand_as(energies[1])]) == 0
    for observed, expected in zip((*gradients[1], *outer_gradients[1]),
                                  (*gradients[0], *outer_gradients[0])):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_common_frame_numerical_fallback_keeps_supported_bad_pair_certificate() -> None:
    from tests.test_causal_congruence_support_20260904 import _future_pair_fixture

    mu, Sigma, omega, prior = _future_pair_fixture(torch.device("cpu"))
    previous = set_full_cov_congruence_precision("fp32_escalate")
    try:
        for backend in ("pairwise", "common_frame"):
            with pytest.raises(FloatingPointError, match="conditioning bounds"):
                FullGaussian.coupling_energy(
                    mu, Sigma, mu.detach(), Sigma.detach(), omega,
                    irrep_dims=[2, 2], diagonal_out=False, precision_policy="fp32_escalate",
                    score_backend=backend, pair_support=torch.ones_like(prior, dtype=torch.bool))
    finally:
        set_full_cov_congruence_precision(previous)


def test_common_frame_cond_precision_policy_preserves_exact_fallback() -> None:
    inputs = _inputs(torch.float32)
    reference = _score(inputs, "pairwise", precision_policy="fp32_escalate_cond")
    actual = _score(inputs, "common_frame", precision_policy="fp32_escalate_cond")
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    gradients_reference = torch.autograd.grad(_weighted(reference), inputs)
    gradients_actual = torch.autograd.grad(_weighted(actual), inputs)
    for observed, expected in zip(gradients_actual, gradients_reference):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
