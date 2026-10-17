r"""Audit 2026-10-16 F89: the diagonal LayerNorm pushforward keeps no float64 autograd graph.

Below float64, :func:`vfe3.model.block_mlp.layernorm_diagonal_pushforward` also forms the float64 result and selects
it where the working value is not positive, without a host read (audit 2026-10-10 round 2, item 57). The select was a
``torch.where`` over both branches, so every call kept the float64 branch in the graph: about half the forward ops
and half the backward ops of the closed form ran in float64 although no entry was selected. The float64 result is now
formed without a graph and selected by value; the gradient reaches the working-precision expression at every entry.
Values are bitwise the old ones. Gradients are bitwise the old ones where nothing is selected; at selected entries
they are the working-precision derivative of the same closed form instead of the float64 one.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vfe3.model import block_mlp as bm


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


class _Float64Ops(TorchDispatchMode):
    r"""Count dispatched ops, and those producing a float64 tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.total, self.float64 = 0, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Tuple[Any, ...] = (),
                           kwargs: Dict[str, Any] | None = None) -> Any:
        out = func(*args, **(kwargs or {}))
        self.total += 1
        outs = out if isinstance(out, (tuple, list)) else (out,)
        self.float64 += any(isinstance(o, torch.Tensor) and o.dtype == torch.float64 for o in outs)
        return out


def _inputs(K: int, rows: int = 512, seed: int = 13) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(rows, K, generator=g).requires_grad_()
    sigma = (0.1 + torch.rand(rows, K, generator=g)).requires_grad_()
    weight = (1.0 + 0.2 * torch.randn(K, generator=g)).requires_grad_()
    return u, sigma, weight


def _old_select(u: torch.Tensor, sigma: torch.Tensor, weight: torch.Tensor, floor: float) -> torch.Tensor:
    r"""The pre-change select: both branches in the graph, gradient through float64 where it is selected."""
    pushed = bm._layernorm_diagonal_pushforward_at(u, sigma, 1e-6, floor, weight)
    high = bm._layernorm_diagonal_pushforward_at(u.double(), sigma.double(), 1e-6, floor, weight.double())
    return torch.where(pushed <= 0, high.float(), pushed)


def test_float32_pushforward_has_no_float64_backward_or_saved_tensor() -> None:
    u, sigma, weight = _inputs(8)
    saved = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = bm.layernorm_diagonal_pushforward(u, sigma, eps=1e-6, covariance_floor=1e-4, weight=weight)
    assert out.dtype == torch.float32 and torch.float64 not in saved
    with _Float64Ops() as backward:
        out.square().sum().backward()
    assert backward.total > 0 and backward.float64 == 0


@pytest.mark.parametrize("K,floor", [(8, 1e-4), (2, 1e-12)])
def test_values_and_gradients_match_the_old_select(K: int, floor: float) -> None:
    r"""K = 8: nothing is selected, values and gradients bitwise. K = 2 with a tiny floor saturates the norm, so
    the float32 bracket rounds to non-positive values: values bitwise, gradients within 1e-12 of the gradient scale."""
    u, sigma, weight = _inputs(K)
    w = torch.randn(u.shape, generator=torch.Generator().manual_seed(5))
    selected = bm._layernorm_diagonal_pushforward_at(u, sigma, 1e-6, floor, weight) <= 0
    assert bool(selected.any()) == (K == 2)
    old = _old_select(u, sigma, weight, floor)
    new = bm.layernorm_diagonal_pushforward(u, sigma, eps=1e-6, covariance_floor=floor, weight=weight)
    assert torch.equal(new, old)
    grads_old = torch.autograd.grad((old * w).sum(), (u, sigma, weight))
    grads_new = torch.autograd.grad((new * w).sum(), (u, sigma, weight))
    for a, b in zip(grads_new, grads_old):
        if K == 2:
            assert float((a - b).abs().max()) <= 1e-12 * float(b.abs().max())
        else:
            assert torch.equal(a, b)
