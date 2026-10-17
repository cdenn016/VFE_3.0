r"""Output screen of the minimal-agent posterior covariance (audit 2026-10-16, finding F8).

The float32 local posterior covariance :math:`\Sigma_i^+ = \Sigma_i - \sum_hZ_i^hM_i^hZ_i^{h\top} + f\Sigma_{p,i}` is a
Kalman downdate computed by cancellation; at block frames of condition :math:`\kappa_F \approx 20` and above, local
variances :math:`e^{4z}` and channel noise :math:`10^{-3}` it returned nonpositive variances with no factorization
failure, so the float64 recompute (keyed to the readout status alone) never fired. Two changes:

1. The floored output is screened on the device (finite and positive variances; a Cholesky factorization of a full
   covariance) and the flag joins the readout status, so the one host read of the call also triggers the float64
   recompute on it. A float64 output that fails the screen raises ``ValueError`` naming the posterior covariance.
   Output-screen recomputes are counted in ``covariance_escalation_elements``, as on the kernel route.
2. The first term is :math:`U_iC_iU_i^\top` recomputed from the :math:`C_i` the gains used, not the input
   :math:`\Sigma_i`, so :math:`\Sigma_i^+ = U_i(C_i - \sum_h\mathsf K^hM^h\mathsf K^{h\top})U_i^\top + f\Sigma_{p,i}`
   holds for the stored frame pair. The float64 recompute keeps the float32 pair, whose inverse is exact only to about
   :math:`\varepsilon_{32}\kappa`; with the input :math:`\Sigma_i` it returned a negative variance at the same case.

Tiny CPU shapes only (N = 8, H = 2, d = 2).
"""
import math
from typing import Tuple

import pytest
import torch

from tests.profiles import make_relational_operators
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import minimal_agent as ma
from vfe3.model.relational_operators import covariance_escalation_elements, reset_covariance_escalation_elements
from vfe3.relational_scoring import set_projected_condition_recovery, source_recovery_counter

torch.set_num_threads(1)
F64 = torch.float64
N, H, D = 8, 2, 2
K = H * D
INFERENCE = ("exact", "mean_field")


def _rn(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=F64)


def _case(seed: int = 43) -> Tuple[CanonicalFrameContext, torch.Tensor, ...]:
    r"""The adjudication generator: block frames of condition :math:`\kappa_2` log-uniform in :math:`[10, 10^4]`
    (seed 43: 785.7, below ``frame_pair_escalation_tol = 1e-4``), local variances :math:`e^{4z}`, maps
    :math:`I + 0.3\,\mathrm{randn}`, a causal prior and a positive prior covariance anchor; float64 masters."""
    g = torch.Generator().manual_seed(seed)
    kappa = float(10 ** (torch.rand(1, generator=g, dtype=F64) * 3 + 1))
    singular = torch.logspace(-math.log10(kappa) / 2, math.log10(kappa) / 2, D, dtype=F64)
    orthogonal = lambda: torch.linalg.qr(_rn(g, 1, N, H, D, D))[0]
    blocks = orthogonal() @ torch.diag_embed(singular.expand(1, N, H, D)) @ orthogonal().transpose(-1, -2)
    frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    mu = _rn(g, 1, N, K) + 1.0
    sigma = torch.exp(4.0 * _rn(g, 1, N, K))
    prior = torch.rand(1, N, K, generator=g, dtype=F64) + 0.4
    S = torch.eye(K, dtype=F64).reshape(H, D, K) + 0.3 * _rn(g, H, D, K)
    T_ = torch.eye(K, dtype=F64).reshape(H, D, K) + 0.3 * _rn(g, H, D, K)
    lp = torch.zeros(N, N, dtype=F64).masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -math.inf)
    return frame, mu, sigma, prior, S, T_, lp


def _ops(inference: str, S: torch.Tensor, T_: torch.Tensor, dtype: torch.dtype):
    ops = make_relational_operators([D] * H, 3, predictive_covariance="moment_matched", predictive_covariance_floor=1e-4,
                                    predictive_readout="minimal_agent", minimal_agent_inference=inference).to(dtype)
    with torch.no_grad():
        ops.raw_score_q.copy_(S.to(dtype))
        ops.T.copy_(T_.to(dtype))
        ops.channel_log_noise.fill_(math.log(1e-3))
    return ops


def _float32(inference: str, seed: int = 43):
    frame, mu, sigma, prior, S, T_, lp = _case(seed)
    frame32 = CanonicalFrameContext(frame.forward.float(), frame.inverse.float(), block_diagonal=True)
    ops32 = _ops(inference, S, T_, torch.float32)
    # The float64 readout of exactly the float32 inputs, parameters and frame pair.
    rounded = _ops(inference, ops32.raw_score_q.detach().double(), ops32.T.detach().double(), F64)
    return ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("certify", [False, True])
def test_nonpositive_float32_output_is_recomputed_in_float64(inference: str, certify: bool) -> None:
    r"""The float32 body returns a nonpositive variance with no factorization failure (certificate off); the call
    returns the float64 recompute instead, positive, the float64 readout of the float32 inputs, and counts
    its N tokens as a covariance escalation. The source-recovery counter grows only when the certificate trips."""
    ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp = _float32(inference)
    args32 = (mu.float(), sigma.float(), frame32, lp.float())
    with torch.no_grad():
        floor = ops32.floor_term(prior.float(), args32[1], frame32, diagonal=True)
        _, body, _, status = ops32._minimal_agent_body(*args32, True, False, floor)
    assert bool((body <= 0).any()) and status.tolist() == [False, False, True]       # the screen alone flags it
    previous = set_projected_condition_recovery(certify)
    try:
        reset_covariance_escalation_elements()
        before = int(source_recovery_counter(mu.device))
        with torch.no_grad():
            mean32, cov32 = ops32.minimal_agent_update(*args32, return_covariance=True, floor_anchor=prior.float())
        recovered = int(source_recovery_counter(mu.device)) - before
        with torch.no_grad():
            mean64, cov64 = rounded.minimal_agent_update(mu.float().double(), sigma.float().double(), frame32, lp,
                                                         return_covariance=True, floor_anchor=prior.float().double())
    finally:
        set_projected_condition_recovery(previous)
    assert cov32.dtype == torch.float32 and bool((cov32 > 0).all())
    assert torch.equal(mean32, mean64.float())
    # The escalated call floors with f Sigma_p formed in float32 (``floor.double()``), the reference in float64.
    torch.testing.assert_close(cov32, cov64.float(), rtol=1e-5, atol=0.0)
    assert covariance_escalation_elements() == N
    assert (recovered > 0) == certify


@pytest.mark.parametrize("inference", INFERENCE)
def test_float64_recompute_with_the_float32_frame_pair_stays_positive(inference: str) -> None:
    r"""With the float32 frame pair the float64 readout returned -0.519 at this case (the input :math:`\Sigma_i` and
    :math:`U_iC_iU_i^\top` differ by about :math:`\varepsilon_{32}\kappa`); built from :math:`C_i` it is positive and
    within 1 percent of the float64 readout of the unrounded inputs (measured below 1e-4 at the flipped entry)."""
    ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp = _float32(inference)
    with torch.no_grad():
        _, cov = rounded.minimal_agent_update(mu.float().double(), sigma.float().double(), frame32, lp,
                                              return_covariance=True, floor_anchor=prior.float().double())
        _, exact = _ops(inference, S, T_, F64).minimal_agent_update(mu, sigma, frame, lp, return_covariance=True,
                                                                   floor_anchor=prior)
    assert bool((cov > 0).all())
    assert float(((cov - exact).abs() / exact).max()) < 1e-2


@pytest.mark.parametrize("diagonal", [True, False])
def test_local_posterior_covariance_reads_the_reference_covariance(diagonal: bool) -> None:
    r""":math:`U_i(C_i - \sum_h\mathsf K^hM^h\mathsf K^{h\top})U_i^\top + f\Sigma_p` for a frame pair that is not an exact
    inverse pair; without ``reference`` the input :math:`\Sigma_i` is used, which differs."""
    g = torch.Generator().manual_seed(5)
    blocks = torch.eye(D, dtype=F64) + 0.3 * _rn(g, 1, N, H, D, D)
    inverse = torch.linalg.inv(blocks) + 1e-3 * _rn(g, 1, N, H, D, D)              # an inconsistent pair
    U = torch.block_diag(*[torch.eye(D, dtype=F64)] * H).expand(1, N, K, K).clone()
    U_inv = U.clone()
    for h in range(H):
        U[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = blocks[..., h, :, :]
        U_inv[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = inverse[..., h, :, :]
    sigma = torch.rand(1, N, K, generator=g, dtype=F64) + 0.3
    if diagonal:
        C = (inverse * sigma.reshape(1, N, H, D).unsqueeze(-2)) @ inverse.transpose(-1, -2)   # (1, N, H, d, d)
        C_dense = torch.zeros(1, N, K, K, dtype=F64)
        for h in range(H):
            C_dense[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = C[..., h, :, :]
        local = sigma
    else:
        local = torch.diag_embed(sigma) + 0.1
        C_dense = C = U_inv @ local @ U_inv.transpose(-1, -2)
    gains = 0.3 * _rn(g, 1, N, H, K, D)
    M = ma._symmetrize(0.2 * _rn(g, 1, N, H, D, D)) + torch.eye(D, dtype=F64)
    floor = torch.full_like(local, 1e-3)
    want = U @ (C_dense - torch.einsum("...nhks,...nhst,...nhlt->...nkl", gains, M, gains)) @ U.transpose(-1, -2)
    want = (torch.diagonal(want, dim1=-2, dim2=-1) if diagonal else want) + floor
    got = ma.local_posterior_covariance(local, blocks, True, gains, M, floor, reference=C)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    plain = ma.local_posterior_covariance(local, blocks, True, gains, M, floor)
    assert float((plain - got).abs().max()) > 1e-4


def _poison(monkeypatch: pytest.MonkeyPatch, also_float64: bool) -> None:
    r"""Make the returned covariance fail the screen at token 0 (a negative variance, or a negative diagonal entry of a
    full covariance) in float32, and optionally in float64."""
    original = ma.local_posterior_covariance

    def poisoned(sigma, *args, **kwargs):
        out = original(sigma, *args, **kwargs)
        if out.dtype == torch.float32 or also_float64:
            out = out.clone()
            if out.dim() == sigma.dim() and sigma.dim() == args[2].dim() - 2:     # diagonal (..., N, K)
                out[..., 0, 1] = -1.0
            else:
                out[..., 0, 1, 1] = -1.0
        return out
    monkeypatch.setattr(ma, "local_posterior_covariance", poisoned)


@pytest.mark.parametrize("full", [False, True])
def test_screened_full_and_diagonal_outputs_take_the_float64_values(full: bool,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp = _float32("exact", seed=3)
    sigma32, prior32 = sigma.float(), prior.float()
    if full:
        sigma32, prior32 = torch.diag_embed(sigma32), torch.diag_embed(prior32)
    with torch.no_grad():
        _, want = rounded.minimal_agent_update(mu.float().double(), sigma32.double(), frame32, lp,
                                               return_covariance=True, floor_anchor=prior32.double())
    _poison(monkeypatch, also_float64=False)
    reset_covariance_escalation_elements()
    with torch.no_grad():
        _, got = ops32.minimal_agent_update(mu.float(), sigma32, frame32, lp.float(), return_covariance=True,
                                            floor_anchor=prior32)
    torch.testing.assert_close(got, want.float(), rtol=1e-5, atol=1e-8)
    assert covariance_escalation_elements() == N


def test_float64_failure_of_the_screen_raises_naming_the_covariance(monkeypatch: pytest.MonkeyPatch) -> None:
    ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp = _float32("mean_field", seed=3)
    _poison(monkeypatch, also_float64=True)
    with pytest.raises(ValueError, match="posterior covariance.*float32 output failed first"):
        ops32.minimal_agent_update(mu.float(), sigma.float(), frame32, lp.float(), return_covariance=True,
                                   floor_anchor=prior.float())
    with pytest.raises(ValueError, match="posterior covariance") as info:             # a float64 call: no recompute
        rounded.minimal_agent_update(mu, sigma, frame, lp, return_covariance=True, floor_anchor=prior)
    assert "float32" not in str(info.value)


@pytest.mark.parametrize("inference", INFERENCE)
def test_healthy_float32_call_is_not_recomputed(inference: str) -> None:
    ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp = _float32(inference, seed=3)
    reset_covariance_escalation_elements()
    before = int(source_recovery_counter(mu.device))
    with torch.no_grad():
        _, cov32 = ops32.minimal_agent_update(mu.float(), sigma.float(), frame32, lp.float(), return_covariance=True,
                                              floor_anchor=prior.float())
    assert bool((cov32 > 0).all()) and covariance_escalation_elements() == 0
    assert int(source_recovery_counter(mu.device)) == before


def test_screened_call_carries_float32_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    ops32, rounded, frame, frame32, mu, sigma, prior, S, T_, lp = _float32("exact", seed=3)
    _poison(monkeypatch, also_float64=False)
    mu32, sigma32 = mu.float().requires_grad_(True), sigma.float().requires_grad_(True)
    mean, cov = ops32.minimal_agent_update(mu32, sigma32, frame32, lp.float(), return_covariance=True,
                                           floor_anchor=prior.float())
    (mean.sum() + cov.sum()).backward()
    assert sigma32.grad.dtype == torch.float32 and bool(torch.isfinite(sigma32.grad).all())
    assert bool(torch.isfinite(mu32.grad).all())


@pytest.mark.parametrize("diagonal", [True, False])
def test_reference_path_gradients_match_finite_differences(diagonal: bool) -> None:
    r"""float64 finite-difference check of :func:`vfe3.model.minimal_agent.local_posterior_covariance` with
    ``reference`` in every differentiable input (frame blocks, reference covariance, gains, M)."""
    g = torch.Generator().manual_seed(9)
    n = 3
    blocks = (torch.eye(D, dtype=F64) + 0.3 * _rn(g, n, H, D, D)).requires_grad_(True)
    if diagonal:
        A = _rn(g, n, H, D, D)
        C = (A @ A.transpose(-1, -2) + torch.eye(D, dtype=F64)).requires_grad_(True)       # (n, H, d, d) blocks
        sigma, floor = torch.ones(n, K, dtype=F64), torch.full((n, K), 1e-3, dtype=F64)
    else:
        A = _rn(g, n, K, K)
        C = (A @ A.transpose(-1, -2) + torch.eye(K, dtype=F64)).requires_grad_(True)       # (n, K, K)
        sigma, floor = torch.eye(K, dtype=F64).expand(n, K, K), torch.full((n, K, K), 1e-3, dtype=F64)
    gains = (0.3 * _rn(g, n, H, K, D)).requires_grad_(True)
    M = (torch.eye(D, dtype=F64) + 0.1 * _rn(g, n, H, D, D)).requires_grad_(True)

    def f(blocks_, C_, gains_, M_):
        return ma.local_posterior_covariance(sigma, blocks_, True, gains_, M_, floor, reference=C_)
    assert torch.autograd.gradcheck(f, (blocks, C, gains, M), eps=1e-6, atol=1e-7, rtol=1e-5)
