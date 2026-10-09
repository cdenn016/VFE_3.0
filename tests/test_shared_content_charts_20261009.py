"""Every layer's content-link chart O' in one batched float64 evaluation per forward (audit 2026-10-09).

The exponential chart O' = T^{-1} + O phi1(TO) was evaluated per layer per forward, one 2K x 2K matrix_exp
each. ``shared_content_charts`` evaluates all layers' stacks in one batched call and hands each operator
its slice. Checked here: the values equal the per-layer evaluation (bit for bit on the CPU), the gradients
with respect to every layer's (T, O) agree, the float64 gradcheck of the batched kernel, an ill-conditioned
T, the restore on exit, and an unchanged model loss. Tiny CPU shapes only.
"""
from tests.profiles import make_relational_operators
from contextlib import nullcontext
from typing import List

import pytest
import torch

import vfe3.model.model as model_module
import vfe3.model.relational_operators as relational_operators
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import (
    RelationalOperators, _chart_receive, shared_content_charts)


def _layers(chart: str, irrep_dims: List[int], n: int = 3, seed: int = 0) -> List[RelationalOperators]:
    g = torch.Generator().manual_seed(seed)
    layers = []
    for layer in range(n):
        ops = make_relational_operators(irrep_dims, 10 + layer, content_link_chart=chart).double()
        with torch.no_grad():
            for T in ops._stored_runs("T"):
                T.add_(.3 * torch.randn(T.shape, generator=g, dtype=torch.float64))
            for O in ops._stored_runs("O"):
                O.add_(.5 * torch.randn(O.shape, generator=g, dtype=torch.float64))
        layers.append(ops)
    return layers


@pytest.mark.parametrize("chart", ["first_order", "exponential"])
@pytest.mark.parametrize("irrep_dims", [[2, 2], [1, 1, 2]])
def test_shared_charts_equal_the_per_layer_values_and_gradients(chart: str, irrep_dims: List[int]) -> None:
    layers = _layers(chart, irrep_dims)
    expected = [ops.content_factor_runs()[1] for ops in layers]
    with shared_content_charts(layers + layers[:1]):                 # a repeated operator is evaluated once
        shared = [ops.content_factor_runs()[1] for ops in layers]
    for want, got in zip(expected, shared):
        for w, s in zip(want, got):
            assert torch.equal(w, s)
    params = [p for ops in layers for p in ops.parameters()]
    loss_per_layer = sum((O.square().sum() for runs in expected for O in runs), torch.zeros((), dtype=torch.float64))
    loss_shared = sum((O.square().sum() for runs in shared for O in runs), torch.zeros((), dtype=torch.float64))
    for a, b in zip(torch.autograd.grad(loss_per_layer, params, allow_unused=True),
                    torch.autograd.grad(loss_shared, params, allow_unused=True)):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    assert all(ops.__dict__.get("_shared_receive_runs") is None for ops in layers)


def test_batched_chart_gradcheck_and_ill_conditioned_T() -> None:
    g = torch.Generator().manual_seed(1)
    K = 3
    T = (torch.eye(K, dtype=torch.float64) + .3 * torch.randn(2, K, K, generator=g, dtype=torch.float64))
    O = .4 * torch.randn(2, K, K, generator=g, dtype=torch.float64)
    T.requires_grad_(True)
    O.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t, o: _chart_receive(t, o, "exponential")[1], (T, O))
    # cond(T) = 1e6: the batched value still equals the per-matrix one, which is e^{OT} T^{-1}.
    Q, _ = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=torch.float64))
    T_ill = (Q * torch.tensor([1., 1e-3, 1e-6], dtype=torch.float64)) @ Q.T
    stack = torch.stack((T_ill, T.detach()[0]))
    batched = _chart_receive(stack, O.detach(), "exponential")[1]
    single = _chart_receive(T_ill, O.detach()[0], "exponential")[1]
    assert torch.equal(batched[0], single)
    product = torch.linalg.matrix_exp(O.detach()[0] @ T_ill) @ torch.linalg.inv(T_ill)
    assert float((single - product).abs().max() / product.abs().max()) < 1e-9


def test_model_loss_is_unchanged_by_the_shared_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(n_layers=2, content_link_chart="exponential"))
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    calls = []
    real_phi1 = relational_operators._phi1
    monkeypatch.setattr(relational_operators, "_phi1", lambda A: calls.append(A.shape) or real_phi1(A))
    _, shared, _ = model(tokens, targets=tokens)
    assert calls == [torch.Size([2, 4, 4])]                          # one batched call for both layers
    shared.backward()
    shared_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(model_module, "shared_content_charts", lambda operators: nullcontext())
    _, per_layer, _ = model(tokens, targets=tokens)
    per_layer.backward()
    assert torch.equal(shared, per_layer)
    for a, b in zip(shared_grads, [p.grad for p in model.parameters() if p.grad is not None]):
        assert torch.equal(a, b)
