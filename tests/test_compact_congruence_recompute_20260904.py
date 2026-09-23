"""CPU regressions for rebuilding compact pair products during precision escalation."""

from collections.abc import Iterator

import pytest
import torch

from vfe3.geometry import transport as T


@pytest.fixture(autouse=True)
def _restore_policy() -> Iterator[None]:
    precision = T.set_full_cov_congruence_precision("fp32_escalate")
    certification = T.set_full_cov_congruence_certification(True)
    yield
    T.set_full_cov_congruence_precision(precision)
    T.set_full_cov_congruence_certification(certification)


@pytest.mark.parametrize("route", ("head_blocks", "marginal_dense", "full_dense"))
def test_compact_fallback_rebuilds_cancelled_pair_product(route: str) -> None:
    r"""Exactly inverse GL+(2) factors require float64 to recover U U^{-1} = I."""
    exp = torch.tensor([[8192., 8191.], [8193., 8192.]])
    inv = torch.tensor([[8192., -8191.], [-8193., 8192.]])
    assert torch.linalg.det(exp.double()) == 1.0
    assert torch.equal(exp.double() @ inv.double(), torch.eye(2, dtype=torch.float64))
    factored = T.CompactFactoredTransport(
        exp.expand(2, 2, 2, 2).clone().requires_grad_(),
        inv.expand(2, 2, 2, 2).clone().requires_grad_(),
        4,
        same_frame_flat_cocycle=True,
    )
    assert torch.count_nonzero(T._compact_pair_blocks(factored)) == 0
    scale = torch.tensor(1.0, requires_grad=True)
    sigma = scale * torch.eye(4).expand(2, 4, 4)
    if route == "head_blocks":
        result = T.transport_covariance_head_blocks(
            factored, sigma, diagonal_out=False, marginal_blocks=[2, 2])
        expected = torch.eye(2, dtype=torch.float64).expand(2, 2, 2, 2, 2)
    else:
        result = T.transport_covariance(
            factored, sigma, diagonal_out=False,
            marginal_blocks=([2, 2] if route == "marginal_dense" else None))
        expected = torch.eye(4, dtype=torch.float64).expand(2, 2, 4, 4)
    assert result is not None
    assert result.dtype is torch.float64
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    # Use a non-self pair: self links are restored separately by the public boundary.
    inputs = (scale, factored.exp_blocks, factored.inv_blocks)
    gradients = torch.autograd.grad(result[0, 1].square().sum(), inputs)
    query = torch.block_diag(*factored.exp_blocks[0].double().unbind())
    key = torch.block_diag(*factored.inv_blocks[1].double().unbind())
    pair_reference = query @ key
    covariance_reference = (
        pair_reference @ (scale.double() * torch.eye(4, dtype=torch.float64))
        @ pair_reference.T
    )
    reference_gradients = torch.autograd.grad(covariance_reference.square().sum(), inputs)
    for actual, expected_gradient in zip(gradients, reference_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected_gradient, rtol=0, atol=0)
    step = 1e-4
    reference = torch.eye(4, dtype=torch.float64)
    finite_difference = (
        ((1 + step) * reference).square().sum()
        - ((1 - step) * reference).square().sum()
    ) / (2 * step)
    torch.testing.assert_close(gradients[0].double(), finite_difference, rtol=1e-10, atol=1e-10)


def test_uncertifiable_congruence_reports_failed_matrix_and_condition_bound() -> None:
    """A real condition failure stays visible and reports the actual failed certificate."""
    omega = torch.diag(torch.tensor([1.0, 1e-8])).reshape(1, 1, 2, 2)
    sigma = torch.eye(2).reshape(1, 2, 2)
    with pytest.raises(FloatingPointError) as caught:
        T.transport_covariance(omega, sigma, diagonal_out=False)
    message = str(caught.value)
    assert "float64 recomputation could not be certified" in message
    assert "failed_matrices=1/1" in message
    assert "first_failed_index=(0, 0)" in message
    assert "finite=True" in message
    assert "condition=" in message and "condition_limit=" in message
    assert "symmetry_residual=" in message and "factor_residual=" in message
