"""Audit 2026-10-16 F27: predictive_content saves no batch-expanded copy of T.

The values ``T_h xi_j`` were computed as ``T @ xi.unsqueeze(-2).unsqueeze(-1)``, whose broadcast
lowering in ``at::matmul`` expands T over the (..., N) batch, reshapes it into a contiguous
``(B N n_r, d_r, K)`` copy and saves it for backward (256 MiB per layer at the launcher shape).
The contraction is now one einsum over the head rows of T; these tests pin that no such copy is
saved and that values and both gradients equal the old broadcast form in float64, for equal and
mixed heads.
"""
from tests.profiles import make_relational_operators

import math
from typing import List, Tuple

import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.relational_operators import RelationalOperators


def _inputs(operators: RelationalOperators, compact: bool, B: int = 2, N: int = 4,
            ) -> Tuple[torch.Tensor, torch.Tensor, CanonicalFrameContext]:
    r"""Random means, row-stochastic routing and a random well-conditioned frame (dense or compact)."""
    K, H = operators.input_dim, len(operators.irrep_dims)
    mu = torch.randn(B, N, K, dtype=torch.float64).requires_grad_()
    beta = torch.randn(B, H, N, N, dtype=torch.float64).softmax(-1).requires_grad_()
    if compact:
        d = operators.irrep_dims[0]
        forward = torch.eye(d, dtype=torch.float64) + 0.1 * torch.randn(B, N, H, d, d, dtype=torch.float64)
    else:
        forward = torch.eye(K, dtype=torch.float64) + 0.1 * torch.randn(B, N, K, K, dtype=torch.float64)
    return mu, beta, CanonicalFrameContext(forward, torch.linalg.inv(forward), compact)


def _broadcast_reference(operators: RelationalOperators, mu: torch.Tensor, beta: torch.Tensor,
                         frame: CanonicalFrameContext) -> torch.Tensor:
    r"""The pre-fix route: m_i = U_i sum_h O'_h sum_j beta_ij^h (T_h @ xi_j) with the broadcast matvec."""
    coordinates = operators.canonical_mean(mu, frame).unsqueeze(-2)          # (..., N, 1, K)
    body = None
    for run, (T, O) in enumerate(zip(operators._stored_runs("T"), operators._receive_runs(None))):
        values = (T @ coordinates.unsqueeze(-1)).squeeze(-1)                  # (..., N, n_r, d_r)
        pooled = torch.einsum("...hij,...jhs->...ihs", operators._head_slice(beta, run, -3), values)
        term = torch.einsum("hks,...nhs->...nk", O, pooled)                    # (..., N, K)
        body = term if body is None else body + term
    if frame.block_diagonal:
        body = body.reshape(*mu.shape[:-1], len(operators.irrep_dims), operators.irrep_dims[0])
    return (frame.forward @ body.unsqueeze(-1)).squeeze(-1).reshape_as(mu)


@pytest.mark.parametrize("irrep_dims,compact", [([3, 3], True), ([3, 3], False), ([2, 2, 4], False)])
def test_predictive_content_saves_no_batch_expanded_copy_of_T(irrep_dims: List[int], compact: bool) -> None:
    torch.manual_seed(271)
    operators = make_relational_operators(irrep_dims, 5, readout_mode="predictive_content").double()
    mu, beta, frame = _inputs(operators, compact)
    T_shapes = [tuple(T.shape) for T in operators._stored_runs("T")]
    saved: List[Tuple[int, ...]] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        operators.predictive_content(mu, beta, frame)
    # A batch-expanded copy of run r's T has trailing dims (d_r, K) and more elements than T itself.
    copies = [shape for shape in saved for T_shape in T_shapes
              if shape[-2:] == T_shape[-2:] and math.prod(shape) > math.prod(T_shape)]
    assert copies == [], f"predictive_content saved batch-expanded copies of T: {copies}"


@pytest.mark.parametrize("irrep_dims,compact", [([3, 3], True), ([3, 3], False), ([2, 2, 4], False)])
def test_predictive_content_matches_broadcast_form_in_float64(irrep_dims: List[int], compact: bool) -> None:
    torch.manual_seed(272)
    operators = make_relational_operators(irrep_dims, 6, readout_mode="predictive_content").double()
    with torch.no_grad():
        for O in operators._stored_runs("O"):
            O.add_(0.1 * torch.randn_like(O))
    mu, beta, frame = _inputs(operators, compact)
    actual = operators.predictive_content(mu, beta, frame)
    expected = _broadcast_reference(operators, mu, beta, frame)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    variables = (mu, beta, *operators._stored_runs("T"), *operators._stored_runs("O"))
    for a, b in zip(torch.autograd.grad(actual.square().sum(), variables),
                    torch.autograd.grad(expected.square().sum(), variables)):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
