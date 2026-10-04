"""The stored frame pair is built at IEEE fp32 whatever the global fp32 matmul precision.

Under ``fp32_matmul_precision="high"`` the CUDA GEMMs that compose the frame pair,
:math:`(U E, E^{-1} U^{-1})`, and fold RoPE into it would run in TF32, whose rounding
(:math:`2^{-10}` against :math:`2^{-23}`) breaks the ``CanonicalFrameContext`` inverse contract
for ill-conditioned frames. TF32 does not exist on the CPU, so these tests record the policy every
matmul and matrix exponential sees and check that the caller's policy is restored, also on error.
Tiny CPU shapes only (K=4, two heads).
"""
from typing import Any, List

import pytest
import torch
from torch.overrides import TorchFunctionMode

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import build_factored_transport, fold_rope_into_frame

_RECORDED = ("matmul", "__matmul__", "bmm", "linalg_matrix_exp", "matrix_exp")


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


def _frames() -> tuple:
    group = get_group("block_glk")(4, 2)
    generator = torch.Generator().manual_seed(3)
    phi = 0.3 * torch.randn(2, 3, group.generators.shape[0], generator=generator)
    right = 0.3 * torch.randn(2, 3, group.generators.shape[0], generator=generator)
    return group, phi, right


@pytest.mark.parametrize("compact", [True, False])
def test_frame_pair_composition_runs_ieee_fp32_and_restores_the_policy(
        high_precision_policy: None, compact: bool) -> None:
    group, phi, right = _frames()
    with _RecordMatmulPrecision() as recorder:
        build_factored_transport(phi, group, right_phi=right, compact_blocks=compact)
    assert recorder.seen and set(recorder.seen) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.parametrize("compact", [True, False])
def test_rope_fold_runs_ieee_fp32_and_restores_the_policy(
        high_precision_policy: None, compact: bool) -> None:
    group, phi, right = _frames()
    base = build_factored_transport(phi, group, right_phi=right, compact_blocks=compact)
    rope = torch.eye(4).expand(2, 3, 4, 4).contiguous()
    with _RecordMatmulPrecision() as recorder:
        fold_rope_into_frame(base, rope)
    assert recorder.seen and set(recorder.seen) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


def test_policy_is_restored_when_the_frame_build_raises(high_precision_policy: None) -> None:
    group, phi, right = _frames()
    with pytest.raises(ValueError, match="validity bound exceeded"):
        build_factored_transport(1e3 * phi, group, right_phi=right, compact_blocks=True)
    assert torch.get_float32_matmul_precision() == "high"
