"""Decode ranking guard by candidate refinement (optimization_plan.md WS5), 2026-09-24.

When a float32 ranking interval overlaps the winner's, the guard recomputes in float64 only the
overlapping candidates of the uncertain rows. Every non-candidate is certified below the winner, so
the argmax equals a complete float64 decode's. The fused CE never reads the ranking, so it keeps the
streamed float32 CE. Candidate refinement is the only resolution since 2026-10-02.
"""
from tests.profiles import make_prior_bank
import copy

import pytest
import torch

from vfe3.model.prior_bank import PriorBank


def _near_tie_bank(decode_mode: str, full: bool) -> PriorBank:
    r"""Vocabulary rows 1-3 are row 0 plus 1e-6 perturbations, so their scores tie in float32."""
    generator = torch.Generator().manual_seed(7)
    V, K = 12, 4
    kwargs = dict(decode_mode=decode_mode, decode_ranking_fp64_escalation=True)
    if full:
        kwargs.update(diagonal_covariance=False, family="gaussian_full")
    pb = make_prior_bank(V, K, 9, **kwargs)
    with torch.no_grad():
        mu = torch.randn(V, K, generator=generator)
        mu[1:4] = mu[0] + 1e-6 * torch.randn(3, K, generator=generator)
        log_sigma = .1 * torch.randn(V, K, generator=generator)
        log_sigma[1:4] = log_sigma[0] + 1e-6 * torch.randn(3, K, generator=generator)
        pb._decode_mu_table().copy_(mu)
        pb._decode_sigma_log_table().copy_(log_sigma)
    return pb


def _queries(pb: PriorBank, full: bool) -> tuple:
    generator = torch.Generator().manual_seed(8)
    mu_v = pb._decode_mu_table().detach()
    mu_q = mu_v[0] + .05 * torch.randn(2, 5, mu_v.shape[-1], generator=generator)
    sigma_q = torch.rand(2, 5, mu_v.shape[-1], generator=generator) + .5
    if full:
        A = .1 * torch.randn(2, 5, mu_v.shape[-1], mu_v.shape[-1], generator=generator)
        sigma_q = torch.diag_embed(sigma_q) + A @ A.transpose(-1, -2)
    return mu_q, sigma_q


@pytest.mark.parametrize("decode_mode,full", [("diagonal_chunked", False), ("full_chunked", True)])
def test_candidate_refinement_reproduces_the_float64_argmax(decode_mode: str, full: bool) -> None:
    pb = _near_tie_bank(decode_mode, full)
    mu_q, sigma_q = _queries(pb, full)
    oracle = copy.deepcopy(pb).double().decode(mu_q.double(), sigma_q.double())
    refined = pb.decode(mu_q, sigma_q)
    assert refined.dtype == torch.float64                      # the guard fired
    assert torch.equal(refined.argmax(-1), oracle.argmax(-1))
    # Candidates carry float64 values; everything else is the float32 decode, upcast.
    unguarded = _near_tie_bank(decode_mode, full)
    unguarded.decode_ranking_fp64_escalation = False
    float32 = unguarded.decode(mu_q, sigma_q)
    changed = refined != float32.double()
    assert bool(changed.any())
    torch.testing.assert_close(refined[changed], oracle[changed], rtol=1e-12, atol=1e-9)
    assert bool(changed[..., :4].any()) and not bool(changed[..., 4:].any())


def test_candidate_refinement_is_differentiable() -> None:
    pb = _near_tie_bank("diagonal_chunked", False)
    mu_q, sigma_q = _queries(pb, False)
    mu_q.requires_grad_(True)
    torch.log_softmax(pb.decode(mu_q, sigma_q), dim=-1)[..., 0].sum().backward()
    assert torch.isfinite(mu_q.grad).all() and bool(mu_q.grad.abs().sum() > 0)


@pytest.mark.parametrize("full", [False, True])
def test_guard_keeps_the_streamed_ce(full: bool) -> None:
    decode_mode = "full_chunked" if full else "diagonal_chunked"
    mu_q, sigma_q = _queries(_near_tie_bank(decode_mode, full), full)
    targets = torch.randint(0, 12, mu_q.shape[:-1], generator=torch.Generator().manual_seed(9))
    fused = "decode_ce_full_chunked" if full else "decode_ce_diagonal_chunked"
    streamed_bank = _near_tie_bank(decode_mode, full)
    streamed_bank.decode_ranking_fp64_escalation = False
    streamed = getattr(streamed_bank, fused)(mu_q, sigma_q, targets)
    guarded = getattr(_near_tie_bank(decode_mode, full), fused)(mu_q, sigma_q, targets)
    assert guarded.dtype == torch.float32
    assert torch.equal(guarded, streamed)
