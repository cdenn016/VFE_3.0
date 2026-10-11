"""Frame applications, projected factorizations and the healthy pair-KL GEMM run at IEEE fp32.

U^{-1} and U round trips amplify rounding by kappa(U) (means) or kappa(U)^2 (covariances, Gram
matrices), and the factorization certificates use the fp32 epsilon. Under fp32_matmul_precision="high"
these token-linear pieces are pinned to "highest". The healthy pair KL, one centered GEMM per (batch,
head) since the 2026-10-09 audit, is pinned too, and so are the recovery-path factor tiles (2026-10-10). TF32
does not exist on the CPU, so these tests record the policy each operation sees and check the restore.
Tiny CPU shapes only.
"""
from typing import Any, Dict, List, Set

import pytest
import torch
from torch.overrides import TorchFunctionMode

from vfe3.model.frame_actions import frame_matvec, reference_covariance, reference_variances
from vfe3.numerics import cholesky_qr2, validated_cholesky_solve
from vfe3.relational_scoring import projected_gaussian_kl, projected_moments


class _Record(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.seen: Dict[str, Set[str]] = {}

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = getattr(func, "__name__", "")
        self.seen.setdefault(name, set()).add(torch.get_float32_matmul_precision())
        return func(*args, **(kwargs or {}))


@pytest.fixture
def high_precision_policy():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _blocks(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.eye(2) + 0.2 * torch.randn(1, 3, 2, 2, 2, generator=generator)   # (B, N, H, d, d)


_MATMULS = ("matmul", "__matmul__", "einsum", "bmm")


def _matmul_policies(recorder: _Record) -> Set[str]:
    return set().union(*(recorder.seen.get(name, set()) for name in _MATMULS))


@pytest.mark.parametrize("call", ["frame_matvec", "reference_variances", "reference_covariance"])
def test_frame_actions_run_ieee_fp32(high_precision_policy: None, call: str) -> None:
    blocks = _blocks()
    frame = type("Frame", (), {"forward": blocks, "inverse": torch.linalg.inv(blocks),
                               "block_diagonal": True})()
    mu, sigma = torch.randn(1, 3, 4), torch.rand(1, 3, 4) + 0.5
    with _Record() as recorder:
        if call == "frame_matvec":
            frame_matvec(frame.inverse, mu, True)
        elif call == "reference_variances":
            reference_variances(mu, sigma, frame)
        else:
            reference_covariance(mu, sigma, frame)
    assert _matmul_policies(recorder) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


def test_cholesky_helpers_run_ieee_fp32(high_precision_policy: None) -> None:
    X = torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(1))
    A = X @ X.transpose(-1, -2) + torch.eye(2)
    with _Record() as recorder:
        cholesky_qr2(X, shift=True)
        validated_cholesky_solve(A)
    assert _matmul_policies(recorder) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


def test_projected_moments_factors_and_pair_gemm_run_ieee_fp32(
        high_precision_policy: None) -> None:
    generator = torch.Generator().manual_seed(2)
    blocks = _blocks(3)
    inverse = torch.linalg.inv(blocks)
    mu_h = torch.randn(1, 3, 2, 2, generator=generator)
    sigma_h = torch.rand(1, 3, 2, 2, generator=generator) + 0.5
    S = torch.randn(2, 2, 2, generator=generator)
    with _Record() as moments:
        a, G = projected_moments(mu_h, sigma_h, inverse, S)
    assert _matmul_policies(moments) == {"highest"}
    projection = (sigma_h, inverse, S, mu_h)
    with _Record() as kl:
        projected_gaussian_kl(a, G, a, G, projection_q=projection, projection_k=projection)
    factor_ops = set().union(*(kl.seen.get(n, set()) for n in
                               ("linalg_cholesky_ex", "cholesky_ex", "cholesky_inverse")))
    assert factor_ops == {"highest"}
    assert _matmul_policies(kl) == {"highest"}                  # the healthy pair GEMM is pinned
    assert torch.get_float32_matmul_precision() == "high"
