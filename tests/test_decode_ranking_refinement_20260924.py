"""Decode ranking guard by candidate refinement (optimization_plan.md WS5), 2026-09-24.

When a float32 ranking interval overlaps the winner's, ``decode_ranking_refinement="candidates"``
recomputes in float64 only the overlapping candidates of the uncertain rows. Every non-candidate is
certified below the winner, so the argmax equals the full float64 recompute's. The fused CE never
reads the ranking, so under "candidates" it keeps the streamed float32 CE.
"""
import pytest
import torch

import vfe3.model.prior_bank as prior_bank_module
from vfe3.model.prior_bank import PriorBank


def _near_tie_bank(decode_mode: str, refinement: str, full: bool, evidence: bool = False) -> PriorBank:
    r"""Vocabulary rows 1-3 are row 0 plus 1e-6 perturbations, so their scores tie in float32."""
    generator = torch.Generator().manual_seed(7)
    V, K = 12, 4
    kwargs = dict(decode_mode=decode_mode, decode_ranking_fp64_escalation=True,
                  decode_ranking_refinement=refinement)
    if full:
        kwargs.update(diagonal_covariance=False, family="gaussian_full")
    if evidence:
        kwargs.update(irrep_dims=[2, 2], use_priorbank_head_evidence_mixer=True)
    pb = PriorBank(V, K, 9, **kwargs)
    with torch.no_grad():
        if evidence:
            pb.head_evidence_logits.copy_(torch.tensor([.4, -.3]))
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


@pytest.mark.parametrize("evidence", [False, True])
@pytest.mark.parametrize("decode_mode,full", [("diagonal", False), ("full_chunked", True)])
def test_candidate_refinement_reproduces_the_full_promotion_argmax(monkeypatch, decode_mode: str,
                                                                    full: bool, evidence: bool) -> None:
    full_bank = _near_tie_bank(decode_mode, "full", full, evidence)
    candidate_bank = _near_tie_bank(decode_mode, "candidates", full, evidence)
    mu_q, sigma_q = _queries(full_bank, full)
    promoted = full_bank.decode(mu_q, sigma_q)
    assert promoted.dtype == torch.float64                     # the guard fired
    calls = []
    original = prior_bank_module._promoted_expanded_decode
    monkeypatch.setattr(prior_bank_module, "_promoted_expanded_decode",
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    refined = candidate_bank.decode(mu_q, sigma_q)
    assert not calls                                           # no (B, N, V) float64 recompute
    assert refined.dtype == torch.float64
    assert torch.equal(refined.argmax(-1), promoted.argmax(-1))
    # Candidates carry float64 values; everything else is the float32 decode, upcast.
    unguarded = _near_tie_bank(decode_mode, "full", full, evidence)
    unguarded.decode_ranking_fp64_escalation = False
    float32 = unguarded.decode(mu_q, sigma_q)
    changed = refined != float32.double()
    assert bool(changed.any())
    torch.testing.assert_close(refined[changed], promoted[changed], rtol=1e-12, atol=1e-9)
    assert bool(changed[..., :4].any()) and not bool(changed[..., 4:].any())


def test_candidate_refinement_is_differentiable() -> None:
    pb = _near_tie_bank("diagonal", "candidates", False)
    mu_q, sigma_q = _queries(pb, False)
    mu_q.requires_grad_(True)
    torch.log_softmax(pb.decode(mu_q, sigma_q), dim=-1)[..., 0].sum().backward()
    assert torch.isfinite(mu_q.grad).all() and bool(mu_q.grad.abs().sum() > 0)


@pytest.mark.parametrize("full", [False, True])
def test_candidates_keep_the_streamed_ce(monkeypatch, full: bool) -> None:
    decode_mode = "full_chunked" if full else "diagonal_chunked"
    mu_q, sigma_q = _queries(_near_tie_bank(decode_mode, "full", full), full)
    targets = torch.randint(0, 12, mu_q.shape[:-1], generator=torch.Generator().manual_seed(9))
    fused = "decode_ce_full_chunked" if full else "decode_ce_diagonal_chunked"
    calls = []
    original = prior_bank_module._promoted_expanded_decode
    monkeypatch.setattr(prior_bank_module, "_promoted_expanded_decode",
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    getattr(_near_tie_bank(decode_mode, "full", full), fused)(mu_q, sigma_q, targets)
    assert calls                                               # "full" promotes the CE
    calls.clear()
    streamed_bank = _near_tie_bank(decode_mode, "full", full)
    streamed_bank.decode_ranking_fp64_escalation = False
    streamed = getattr(streamed_bank, fused)(mu_q, sigma_q, targets)
    candidate = getattr(_near_tie_bank(decode_mode, "candidates", full), fused)(mu_q, sigma_q, targets)
    assert not calls
    assert torch.equal(candidate, streamed)


def test_refinement_config_default_validated_and_inactive_without_the_guard() -> None:
    from tests.profiles import tiny_config
    assert tiny_config(decode_ranking_refinement="full").decode_ranking_refinement == "full"
    with pytest.raises(ValueError, match="decode_ranking_refinement"):
        tiny_config(decode_ranking_refinement="some")
    assert "decode_ranking_refinement" in tiny_config().inactive_fields()      # guard off in profile
    with pytest.raises(ValueError, match="decode ranking refinement"):
        PriorBank(4, 2, 9, decode_ranking_refinement="some")
