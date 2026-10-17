r"""Precision of the kernel moment-matched covariance (audit 2026-10-16, findings F7 and F41).

F7. On block frames the diagonal :math:`\mathrm{diag}(U_iC_i^{\mathrm{att}}U_i^\top)` passes through
:math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}` and back, so float32 storage of :math:`C_i` (before any pooling) is amplified by
about :math:`\kappa(U_i)^2`. At :math:`\kappa = 50` and variances over :math:`[4\cdot10^{-4}, 340]` in each head block the
float32 update returns variances that are positive but more than 10 percent wrong; the sign-only float64 escalation
does not see them. ``predictive_covariance_precision='fp64'`` (the requested opt-in value; the constructor keyword
here) runs the whole covariance body, the floor and the ceiling in float64 from the float32 beliefs and casts the
result back, which removes the error up to the rounding of the float32 inputs. ``'fp32'`` is the unchanged update.

F41. The float64 escalation recomputes a failing sequence whole, but writes back only its failing receivers: a
receiver's update reads its own routing row and its sources :math:`j \le i`, so an earlier position keeps its float32
value bit for bit when a later token fails, and the update stays causal.

Tiny CPU shapes only (H = 2, d = 4, N = 6).
"""
import math
from typing import Tuple

import pytest
import torch

from tests.profiles import make_relational_operators
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import relational_operators
from vfe3.model.relational_operators import covariance_escalation_elements, reset_covariance_escalation_elements

torch.set_num_threads(1)
F64 = torch.float64
H, D, N, B = 2, 4, 6, 2
K = H * D


def _case(kappa: float, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    r"""Per-sequence block frames of condition ``kappa`` shared by every token, zero means (no spread), identity
    content maps, the same log-spaced variances :math:`[4\cdot10^{-4}, 340]` in each head block and a causal router.

    Then :math:`C_i^{\mathrm{att}} = C_i` and the exact update is :math:`\Sigma_i + f\Sigma_p`, so every float32 error is
    the round trip through :math:`U_i^{-1}` and :math:`U_i`."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *shape: torch.randn(*shape, generator=g, dtype=F64)
    orthogonal = lambda *shape: torch.linalg.qr(rn(*shape))[0]
    singular = torch.logspace(-math.log10(kappa) / 2, math.log10(kappa) / 2, D, dtype=F64)
    blocks = (orthogonal(B, 1, H, D, D) @ torch.diag_embed(singular.expand(B, 1, H, D))
              @ orthogonal(B, 1, H, D, D).transpose(-1, -2)).expand(B, N, H, D, D).contiguous()
    inverse = torch.linalg.inv(blocks)
    mu = torch.zeros(B, N, K, dtype=F64)
    sigma = torch.logspace(math.log10(4e-4), math.log10(340.0), D, dtype=F64).repeat(H).expand(B, N, K).contiguous()
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(rn(B, H, N, N).masked_fill(~causal, -torch.inf), dim=-1)
    receive = torch.eye(K, dtype=F64).reshape(K, H, D).permute(1, 0, 2).contiguous()      # (H, K, d) O'_h
    prior = torch.full_like(sigma, 5.0)                                                  # Sigma_p
    return blocks, inverse, mu, sigma, beta, receive, prior


def _ops(dtype: torch.dtype, **overrides: object):
    ops = make_relational_operators([D] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                    predictive_covariance="moment_matched", predictive_covariance_floor=1e-4,
                                    **overrides).to(dtype)
    with torch.no_grad():
        ops.T.copy_(torch.eye(K, dtype=F64).reshape(H, D, K))
    return ops


def _float32(kappa: float):
    blocks, inverse, mu, sigma, beta, receive, prior = _case(kappa)
    frame32 = CanonicalFrameContext(blocks.float(), inverse.float(), block_diagonal=True)
    with torch.no_grad():
        reference = _ops(F64).predictive_covariance_update(
            mu, sigma, beta, CanonicalFrameContext(blocks, inverse, block_diagonal=True), receive, floor_anchor=prior)
    return frame32, mu.float(), sigma.float(), beta.float(), receive.float(), prior.float(), reference


def _relative(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    return (got.detach().double() - want).abs() / want.abs()


def test_float32_update_returns_positive_but_wrong_variances_the_sign_escalation_misses() -> None:
    r"""The defect at :math:`\kappa = 50`: every variance positive, some more than 10 percent wrong, escalation inert."""
    frame, mu, sigma, beta, receive, prior, reference = _float32(50.0)
    ops = _ops(torch.float32)
    assert ops.predictive_covariance_precision == "fp32"
    out = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior)
    assert bool((out > 0).all()) and float(_relative(out, reference).max()) > 0.1
    ops.predictive_covariance_fp64_escalation = True
    reset_covariance_escalation_elements()
    assert torch.equal(ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior), out)
    assert covariance_escalation_elements() == 0


@pytest.mark.parametrize("kappa", [50.0, 200.0])
@pytest.mark.parametrize("ceiling", [None, 30.0])
def test_fp64_precision_removes_the_round_trip_error(kappa: float, ceiling: object) -> None:
    r"""Under ``'fp64'`` the float32 output is the float64 update of the float32 inputs (frame, beliefs, router, maps)
    cast back, bit for bit, and lies within :math:`10^{-4}` of the unrounded float64 update (measured: 8e-7 at
    :math:`\kappa = 50`, 2e-6 at 200, from the rounding of the frame pair); the float32 route is wrong by 50 percent
    and more at the same inputs."""
    frame, mu, sigma, beta, receive, prior, reference = _float32(kappa)
    ops = _ops(torch.float32, predictive_covariance_ceiling=ceiling)
    if ceiling is not None:
        reference = ops.ceiling_term(reference, prior.double(), diagonal=True)
    fp32 = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior)
    ops.predictive_covariance_precision = "fp64"
    out = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior)
    assert out.dtype == torch.float32 and bool((out > 0).all())
    wide = ops.predictive_covariance_update(mu.double(), sigma.double(), beta.double(), frame, receive.double(),
                                            floor_anchor=prior.double())
    assert torch.equal(out, wide.float())
    assert float(_relative(out, reference).max()) < 1e-4 < 0.5 < float(_relative(fp32, reference).max())


def test_fp64_precision_carries_float32_gradients_equal_to_the_float64_update() -> None:
    r"""The cast is the only difference: with float32-exact output weights the gradients equal, bit for bit, the
    float64 update's gradients cast to float32."""
    frame, mu, sigma, beta, receive, prior, _ = _float32(50.0)
    weights = torch.randn(sigma.shape, generator=torch.Generator().manual_seed(3)).double()
    ops = _ops(torch.float32)
    ops.predictive_covariance_precision = "fp64"
    leaves = [x.clone().requires_grad_(True) for x in (mu, sigma, beta)]
    (ops.predictive_covariance_update(*leaves, frame, receive, floor_anchor=prior).double() * weights).sum().backward()
    wide = [x.double().requires_grad_(True) for x in (mu, sigma, beta)]
    (ops.predictive_covariance_update(*wide, frame, receive.double(), floor_anchor=prior.double())
     * weights).sum().backward()
    for leaf, ref in zip(leaves, wide):
        assert leaf.grad.dtype == torch.float32 and bool(torch.isfinite(leaf.grad).all())
        assert torch.equal(leaf.grad, ref.grad.float())


def test_fp64_precision_runs_the_dense_route_in_float64() -> None:
    r"""The full family (dense route) takes the same float64 body."""
    frame, mu, sigma, beta, receive, prior, _ = _float32(50.0)
    ops = _ops(torch.float32)
    ops.predictive_covariance_precision = "fp64"
    full, anchor = torch.diag_embed(sigma), torch.diag_embed(prior)
    out = ops.predictive_covariance_update(mu, full, beta, frame, receive, floor_anchor=anchor)
    wide = ops.predictive_covariance_update(mu.double(), full.double(), beta.double(), frame, receive.double(),
                                            floor_anchor=anchor.double())
    assert out.dtype == torch.float32 and torch.equal(out, wide.float())


def test_precision_values_and_refusal() -> None:
    assert relational_operators.PREDICTIVE_COVARIANCE_PRECISIONS == ("fp32", "fp64")
    with pytest.raises(ValueError, match="predictive_covariance_precision"):
        _ops(torch.float32, predictive_covariance_precision="float64")


# ----------------------------------------------------------------------------------------------
# F41: the escalation writes back only the failing receivers
# ----------------------------------------------------------------------------------------------

def _poison_last_token(ops) -> None:
    r"""Wrap the update so a float32 call returns one nonpositive variance at the last token of sequence 0."""
    original = ops._predictive_covariance_update

    def update(mu, sigma, *args):
        out = original(mu, sigma, *args)
        if out.dtype == torch.float32:
            out = out.clone()
            out[0, -1, 1] = -1.0
        return out
    ops._predictive_covariance_update = update


@pytest.mark.parametrize("batched", [True, False])
def test_escalation_keeps_earlier_positions_bit_for_bit(batched: bool) -> None:
    r"""A failure at the last token changes no earlier position of its sequence (and no other sequence); the failing
    receiver takes the float64 value. ``batched=False`` drops the batch axis, the whole-call recompute."""
    frame, mu, sigma, beta, receive, prior, _ = _float32(10.0)
    if not batched:
        frame = CanonicalFrameContext(frame.forward[0], frame.inverse[0], block_diagonal=True)
        mu, sigma, beta, prior = mu[0], sigma[0], beta[0], prior[0]
    ops = _ops(torch.float32)
    healthy = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior)
    wide = ops.predictive_covariance_update(mu.double(), sigma.double(), beta.double(), frame, receive.double(),
                                            floor_anchor=prior.double()).float()
    ops.predictive_covariance_fp64_escalation = True
    if batched:
        _poison_last_token(ops)
    else:
        _poison_unbatched(ops)
    reset_covariance_escalation_elements()
    out = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior)
    if batched:
        assert torch.equal(out[0, :-1], healthy[0, :-1]) and torch.equal(out[1], healthy[1])
        torch.testing.assert_close(out[0, -1], wide[0, -1])        # the float64 value (floor rounded in float32)
        assert covariance_escalation_elements() == N                  # the recomputed sequence, every token
    else:
        assert torch.equal(out[:-1], healthy[:-1])
        torch.testing.assert_close(out[-1], wide[-1])
        assert covariance_escalation_elements() == N


def _poison_unbatched(ops) -> None:
    original = ops._predictive_covariance_update

    def update(mu, sigma, *args):
        out = original(mu, sigma, *args)
        if out.dtype == torch.float32:
            out = out.clone()
            out[-1, 1] = -1.0
        return out
    ops._predictive_covariance_update = update


def test_escalation_of_a_full_covariance_writes_back_only_the_failing_receiver() -> None:
    frame, mu, sigma, beta, receive, prior, _ = _float32(10.0)
    full, anchor = torch.diag_embed(sigma), torch.diag_embed(prior)
    ops = _ops(torch.float32)
    healthy = ops.predictive_covariance_update(mu, full, beta, frame, receive, floor_anchor=anchor)
    wide = ops.predictive_covariance_update(mu.double(), full.double(), beta.double(), frame, receive.double(),
                                            floor_anchor=anchor.double()).float()
    ops.predictive_covariance_fp64_escalation = True
    original = ops._predictive_covariance_update

    def update(mu_, sigma_, *args):
        out = original(mu_, sigma_, *args)
        if out.dtype == torch.float32:
            out = out.clone()
            out[1, 2, 0, 0] = -1.0                                    # sequence 1, token 2 is not PD
        return out
    ops._predictive_covariance_update = update
    out = ops.predictive_covariance_update(mu, full, beta, frame, receive, floor_anchor=anchor)
    keep = torch.ones(B, N, dtype=torch.bool)
    keep[1, 2] = False
    assert torch.equal(out[keep], healthy[keep])
    torch.testing.assert_close(out[1, 2], wide[1, 2])


def test_escalated_receivers_carry_gradients_through_both_branches() -> None:
    frame, mu, sigma, beta, receive, prior, _ = _float32(10.0)
    ops = _ops(torch.float32)
    ops.predictive_covariance_fp64_escalation = True
    _poison_last_token(ops)
    mu, sigma = mu.clone().requires_grad_(True), sigma.clone().requires_grad_(True)
    ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=prior).sum().backward()
    assert sigma.grad.dtype == torch.float32 and bool(torch.isfinite(sigma.grad).all())
    assert bool(torch.isfinite(mu.grad).all())
