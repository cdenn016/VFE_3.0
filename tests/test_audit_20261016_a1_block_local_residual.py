r"""Block-local groups certify the chart retraction blockwise (audit 2026-10-16, F29).

``stage_pullback_group_candidate`` checks every backtracking attempt against the exact right product,
:math:`\|e^{\varphi'} - e^{\varphi} e^{\delta}\|_F / \|e^{\varphi} e^{\delta}\|_F \le 10^{-6}`. Only the
per-head ``block_glk`` basis read the chart as stacked head blocks; every block-local group (``block_so``,
``block_co``, ``block_sl``, ``block_borel``, ``block_unipotent``, ``block_glmc``) copied the dense
(H m, K, K) basis to float64 on every attempt and exponentiated K x K matrices. The block-local route embeds
:math:`M_h = \sum_a \varphi_{(h,a)} L_a` through the cached (m, d, d) local basis instead. Pins, at K = 8 in
float64 and float32: the blockwise residual and current element equal the dense formulas of a twin whose local
certificate is withdrawn; staging gives the same candidate as the twin; and no attempt reads the dense basis
after the direction call.
"""
from __future__ import annotations

import pytest
import torch

import vfe3.gauge_optim as gauge_optim
import vfe3.geometry.groups as groups_module
from vfe3.gauge_optim import (
    _dense_group_element,
    _pullback_group_current_element,
    _pullback_group_product_residual,
    stage_pullback_group_candidate,
)
from vfe3.geometry.groups import get_group

GROUPS = ("block_so", "block_co", "block_sl", "block_borel", "block_unipotent", "block_glmc")
F64 = torch.float64


def _pair(name: str, n_heads: int, dtype: torch.dtype):
    r"""A lazy block-local group and a dense twin whose replaced basis withdraws the local certificate."""
    lazy = get_group(name)(8, n_heads, dtype=dtype)
    dense = get_group(name)(8, n_heads, dtype=dtype)
    dense.generators = dense.generators.clone()
    assert lazy.certified_block_local_basis() is not None
    assert dense.certified_block_local_basis() is None
    return lazy, dense


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("n_heads", [1, 2])
@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_blockwise_residual_equals_the_dense_formula(name: str, n_heads: int, dtype: torch.dtype) -> None:
    lazy, dense = _pair(name, n_heads, dtype)
    n_gen, K = lazy.generator_shape[0], lazy.generator_shape[-1]
    seed = torch.Generator().manual_seed(7)
    phi = 0.3 * torch.randn(6, n_gen, generator=seed, dtype=F64)
    delta = 1e-2 * torch.randn(6, n_gen, generator=seed, dtype=F64)
    candidate = phi + delta + 1e-5 * torch.randn(6, n_gen, generator=seed, dtype=F64)

    current_lazy = _pullback_group_current_element(phi, lazy)
    current_dense = _pullback_group_current_element(phi, dense)                   # (6, K, K)
    expected_shape = (6, K, K) if n_heads == 1 else (6, n_heads, K // n_heads, K // n_heads)
    assert tuple(current_lazy.shape) == expected_shape
    torch.testing.assert_close(_dense_group_element(current_lazy, K), current_dense, rtol=1e-13, atol=1e-14)

    residual_lazy = _pullback_group_product_residual(candidate, current_lazy, delta, lazy)
    residual_dense = _pullback_group_product_residual(candidate, current_dense, delta, dense)
    assert residual_lazy.shape == (6,)
    torch.testing.assert_close(residual_lazy, residual_dense, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("n_heads", [1, 2])
def test_staging_matches_the_dense_twin(name: str, n_heads: int) -> None:
    lazy, dense = _pair(name, n_heads, F64)
    n_gen = lazy.generator_shape[0]
    seed = torch.Generator().manual_seed(3)
    phi = 0.2 * torch.randn(5, n_gen, generator=seed, dtype=F64)
    grad = torch.randn(5, n_gen, generator=seed, dtype=F64)
    kwargs = dict(learning_rate=0.01, trust_radius=0.1, chart_max_norm=5.0)
    a = stage_pullback_group_candidate(grad, phi, lazy, **kwargs)
    b = stage_pullback_group_candidate(grad, phi, dense, **kwargs)
    assert torch.equal(a.backtracking_reductions, b.backtracking_reductions)
    torch.testing.assert_close(a.candidate_phi, b.candidate_phi, rtol=0, atol=0)
    torch.testing.assert_close(a.group_product_residual, b.group_product_residual, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("name", GROUPS)
def test_no_dense_basis_read_after_the_direction(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    group = get_group(name)(8, 2)                                                # float32 storage, as built
    n_gen = group.generator_shape[0]
    seed = torch.Generator().manual_seed(11)
    phi = 0.3 * torch.randn(5, n_gen, generator=seed)
    grad = torch.randn(5, n_gen, generator=seed)
    kwargs = dict(learning_rate=0.01, trust_radius=0.1, chart_max_norm=5.0)
    stage_pullback_group_candidate(grad, phi, group, **kwargs)                   # warm the one-time caches

    state = {"after_direction": False, "reads": 0}
    original_get = groups_module._GeneratorTensor.__get__
    original_direction = gauge_optim.pullback_group_direction

    def counting_get(self, instance, owner=None):
        if instance is group and state["after_direction"]:
            state["reads"] += 1
        return original_get(self, instance, owner)

    def flagging_direction(*args, **kwargs):
        result = original_direction(*args, **kwargs)
        state["after_direction"] = True
        return result

    monkeypatch.setattr(groups_module._GeneratorTensor, "__get__", counting_get)
    monkeypatch.setattr(gauge_optim, "pullback_group_direction", flagging_direction)
    candidate = stage_pullback_group_candidate(grad, phi, group, **kwargs)
    assert state["after_direction"]
    assert int(candidate.backtracking_reductions.max()) >= 1                     # several attempts ran
    assert state["reads"] == 0
