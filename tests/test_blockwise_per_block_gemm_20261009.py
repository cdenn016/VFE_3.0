r"""Per-block GEMMs of the blockwise moment-matched covariance update (2026-10-09).

On CUDA ``_per_block_bmm`` runs the head-block GEMMs one block per call so their weight gradients get a
split reduction; on CPU it is ``torch.bmm``. Each block keeps the reduction order of the batched call, so
forcing the per-block form on CPU must reproduce the batched values and gradients bitwise in float32.
"""
from typing import List

import pytest
import torch

from tests.profiles import make_relational_operators
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import relational_operators


def _per_block_everywhere(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.bmm(x[None], y[None])[0] for x, y in zip(left.unbind(0), right.unbind(0))])


def _update_and_grads(block: bool, residual: bool) -> List[torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    H, d, N, batch = 3, 2, 5, (2,)
    K = H * d
    ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=residual,
                                    predictive_covariance="moment_matched", predictive_covariance_floor=1e-3)
    with torch.no_grad():
        ops.T.copy_(randn(*ops.T.shape))
        ops.O.copy_(randn(*ops.O.shape))
    blocks = torch.eye(d, dtype=torch.float64) + 0.3 * randn(*batch, N, H, d, d)
    mu = randn(*batch, N, K)
    if block:
        A = randn(*batch, N, H, d, d)
        sigma = torch.einsum("...hab,hg->...hagb", A @ A.transpose(-1, -2) + 0.2 * torch.eye(d, dtype=torch.float64),
                             torch.eye(H, dtype=torch.float64)).reshape(*batch, N, K, K)
    else:
        sigma = torch.rand(*batch, N, K, generator=generator, dtype=torch.float64) + 0.2
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(randn(*batch, H, N, N).masked_fill(~causal, -torch.inf), dim=-1)
    leaves = [x.float().requires_grad_(True) for x in (mu, sigma, beta, blocks, torch.linalg.inv(blocks))]
    frame = CanonicalFrameContext(leaves[3], leaves[4], block_diagonal=True)
    out = ops._predictive_covariance_update_blockwise(leaves[0], leaves[1], leaves[2], frame, None)
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(9))
    (out * weights).sum().backward()
    return [out.detach()] + [leaf.grad for leaf in leaves] + [ops.T.grad, ops.O.grad]


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_per_block_gemms_equal_batched_bitwise(monkeypatch: pytest.MonkeyPatch, block: bool, residual: bool) -> None:
    batched = _update_and_grads(block, residual)
    monkeypatch.setattr(relational_operators, "_per_block_bmm", _per_block_everywhere)
    per_block = _update_and_grads(block, residual)
    for reference, value in zip(batched, per_block):
        assert torch.equal(reference, value)
