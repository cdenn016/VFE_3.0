r"""Blockwise moment-matched covariance (2026-10-04).

For the diagonal family with block-diagonal frames, C_i = U_i^{-1} diag(sigma_i) U_i^{-T} is block
diagonal and only diag(U_i C_i^att U_i^T) is returned, so the update needs the H diagonal d x d
blocks of C_i and C_i^att alone. The blockwise route must equal the dense K x K route (kept for the
full family and dense frames) to rounding, in value and in gradient. The independent enumerated
kernel of ``tests/test_predictive_covariance.py`` checks both against the declared law.
"""
from tests.profiles import make_relational_operators
from typing import Tuple

import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.relational_operators import RelationalOperators


def _case(
    H:        int,
    d:        int,
    residual: bool,
    batch:    Tuple[int, ...],
    dtype:    torch.dtype,
    seed:     int = 0,
) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    K, N = H * d, 5
    ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=residual,
                              predictive_covariance="moment_matched", predictive_covariance_floor=1e-3)
    ops = ops.to(dtype)
    with torch.no_grad():
        ops.T.copy_(randn(*ops.T.shape))
        ops.O.copy_(randn(*ops.O.shape))
    blocks = torch.eye(d, dtype=torch.float64) + 0.3 * randn(*batch, N, H, d, d)
    frame = CanonicalFrameContext(blocks.to(dtype), torch.linalg.inv(blocks).to(dtype), block_diagonal=True)
    mu = randn(*batch, N, K).to(dtype)
    sigma = (torch.rand(*batch, N, K, generator=generator, dtype=torch.float64) + 0.2).to(dtype)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(randn(*batch, H, N, N).masked_fill(~causal, -torch.inf), dim=-1).to(dtype)
    receive = randn(H, K, d).to(dtype)                       # a generic effective O'_h
    return ops, mu, sigma, beta, frame, receive


@pytest.mark.parametrize("H,d", [(1, 3), (2, 2), (3, 2), (4, 3)])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_blockwise_equals_dense_in_float64(H: int, d: int, residual: bool, batch: Tuple[int, ...]) -> None:
    ops, mu, sigma, beta, frame, receive = _case(H, d, residual, batch, torch.float64)
    for maps in (None, receive):                            # the stored chart, then a generic O'
        blockwise = ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame, maps)
        dense = ops._predictive_covariance_update_dense(mu, sigma, beta, frame, maps)
        assert blockwise.shape == sigma.shape and blockwise.dtype == sigma.dtype
        torch.testing.assert_close(blockwise, dense, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("residual", [False, True])
def test_blockwise_equals_dense_in_float32(residual: bool) -> None:
    ops, mu, sigma, beta, frame, receive = _case(4, 3, residual, (2,), torch.float32, seed=3)
    blockwise = ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame, receive)
    dense = ops._predictive_covariance_update_dense(mu, sigma, beta, frame, receive)
    assert blockwise.dtype == torch.float32
    torch.testing.assert_close(blockwise, dense, rtol=1e-4, atol=1e-4 * float(dense.abs().max()))


@pytest.mark.parametrize("residual", [False, True])
def test_blockwise_gradients_equal_dense(residual: bool) -> None:
    ops, mu, sigma, beta, frame, receive = _case(3, 2, residual, (2,), torch.float64, seed=5)
    weights = torch.randn(sigma.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    grads = []
    for route in (ops._predictive_covariance_update_blockwise, ops._predictive_covariance_update_dense):
        leaves = [x.detach().clone().requires_grad_(True)
                  for x in (mu, sigma, beta, frame.forward, frame.inverse, receive)]
        leaf_frame = CanonicalFrameContext(leaves[3], leaves[4], block_diagonal=True)
        ops.zero_grad(set_to_none=True)
        (route(leaves[0], leaves[1], leaves[2], leaf_frame, leaves[5]) * weights).sum().backward()
        grads.append([leaf.grad for leaf in leaves] + [ops.T.grad])
    for blockwise, dense in zip(*grads):
        torch.testing.assert_close(blockwise, dense, rtol=1e-10, atol=1e-10)


def test_public_update_dispatches_on_family_and_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, True, (1,), torch.float64)
    calls = []
    for name in ("_predictive_covariance_update_blockwise", "_predictive_covariance_update_dense"):
        original = getattr(ops, name)
        monkeypatch.setattr(ops, name, lambda *a, _o=original, _n=name, **k: calls.append(_n) or _o(*a, **k))
    ops.predictive_covariance_update(mu, sigma, beta, frame, receive)          # diagonal, compact
    ops.predictive_covariance_update(mu, torch.diag_embed(sigma), beta, frame, receive)   # full family
    dense_frame = CanonicalFrameContext(torch.block_diag(*frame.forward[0, 0]).expand(1, 5, 4, 4).clone(),
                                        torch.block_diag(*frame.inverse[0, 0]).expand(1, 5, 4, 4).clone())
    ops.predictive_covariance_update(mu, sigma, beta, dense_frame, receive)    # dense frame
    assert calls == ["_predictive_covariance_update_blockwise", "_predictive_covariance_update_dense",
                     "_predictive_covariance_update_dense"]
