"""The moment-matched covariance update runs at IEEE fp32 whatever the global matmul precision.

diag(U C_att U^T) with C_i = U^{-1} diag(sigma) U^{-T} passes through U^{-1} and back through U, so its
rounding is amplified by about kappa(U)^2. Under fp32_matmul_precision="high" (TF32, rounding 2^-10)
small output variances came out nonpositive on ill-conditioned frames and the next layer's projected
factorization rejected them. TF32 does not exist on the CPU, so these tests record the policy every
contraction sees, on both routes and through the checkpointed recompute, and check the restore.
Tiny CPU shapes only.
"""
from typing import Any, List

import pytest
import torch
from torch.overrides import TorchFunctionMode

from tests.test_blockwise_moment_matched_20261004 import _case

_RECORDED = ("matmul", "__matmul__", "einsum", "bmm")


class _RecordMatmulPrecision(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.seen: List[str] = []

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if getattr(func, "__name__", "") in _RECORDED:
            self.seen.append(torch.get_float32_matmul_precision())
        return func(*args, **(kwargs or {}))


@pytest.fixture
def high_precision_policy():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("route", ["blockwise", "dense"])
def test_moment_matched_update_runs_ieee_fp32_and_restores_the_policy(
        high_precision_policy: None, route: str) -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, True, (2,), torch.float32, seed=1)
    if route == "dense":
        frame = type(frame)(torch.block_diag(*frame.forward[0, 0]).expand(2, 5, 4, 4).clone(),
                            torch.block_diag(*frame.inverse[0, 0]).expand(2, 5, 4, 4).clone())
    with _RecordMatmulPrecision() as recorder:
        ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=sigma)
    assert recorder.seen and set(recorder.seen) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


def test_checkpointed_recompute_runs_ieee_fp32(high_precision_policy: None) -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, True, (2,), torch.float32, seed=2)
    ops.predictive_covariance_checkpoint = True
    body = ops._predictive_covariance_update_blockwise
    seen: List[str] = []

    def recorded(*args: Any) -> torch.Tensor:                  # the policy at each (re)computation
        seen.append(torch.get_float32_matmul_precision())
        return body(*args)

    ops._predictive_covariance_update_blockwise = recorded
    mu = mu.requires_grad_(True)
    out = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=sigma)
    out.sum().backward()                                       # recomputes the checkpointed forward
    assert seen == ["highest", "highest"] and mu.grad is not None
    assert torch.get_float32_matmul_precision() == "high"
