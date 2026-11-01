r"""Audit 2026-10-31 remediation, package ``decode``: the refactored KL decode is bitwise the previous one (CPU).

dead-code-6 (A329) moved the diagonal and full chunk loops into one ``PriorBank._kl_decode_ce``, prior-bank-decode-2
(A096) moved the eps clamp of the query variance into the trace term as well, gap-checkpoint-recompute-semantics-4
(A181) stopped the RNG stash of the chunk checkpoints, prior-bank-decode-1 (A028) moved the fp64 island to the
inputs, and numerics-1 (A007) split the cross-term GEMM from the bias add. On the default fp32 route with query
variances at or above eps none of these may change a number (brief rule 2). The references below are the bodies of
``_decode_av``, ``_prepare_decode_bank_factors``, ``_refined_candidate_decode``, ``_decode_diagonal`` and the torch
routes of ``decode_ce_diagonal_chunked`` and ``decode_ce_full_chunked`` as they stood at 5bb0a57, copied without
their docstrings and without the fused Triton branch (CPU tensors never take it), so each comparison is against the
previous arithmetic and not against the new code. Tiny models only.
"""
from __future__ import annotations

import dataclasses
import math
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch
import torch.utils.checkpoint as _checkpoint

import vfe3.model.prior_bank as bank_module
from tests.profiles import live_config, make_prior_bank, reference_k8_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import (
    PriorBank,
    _decode_analytic_kl_logits,
    _decode_av_lhs,
    _decode_ce_chunk_activation_bytes,
    _decode_ce_chunk_summaries,
    _decode_ce_combine_summaries,
    _decode_ce_result,
    _decode_ce_should_checkpoint,
    _final_ranking_uncertain,
    _outward_lower,
    _outward_upper,
    _pairwise_decode_av,
)
from vfe3.numerics import _count_decode_logdet_fallback, bounded_variance_from_log
from vfe3.runtime import ieee_fp32_matmul


@pytest.fixture(autouse=True)
def _restore_policy_and_threads() -> Iterator[None]:
    policy, threads = torch.get_float32_matmul_precision(), torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    yield
    torch.set_float32_matmul_precision(policy)
    torch.set_num_threads(threads)


# ------------------------------------------------------------------------------------- 5bb0a57 references

def _head_decode_av(
    sq: torch.Tensor, mc_q: torch.Tensor, mc_v: torch.Tensor, inv_v: torch.Tensor, lsum: torch.Tensor,
    *, lhs: Optional[torch.Tensor] = None, return_error_bound: bool = False, precision: Optional[str] = None,
    bank_factors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Any:
    precision = bank_module._DECODE_AV_PRECISION if precision is None else precision
    if precision == "fp64" and sq.dtype is not torch.float64:
        return _head_decode_av(
            sq.double(), mc_q.double(), mc_v.double(), inv_v.double(), lsum.double(),
            lhs=None, return_error_bound=return_error_bound, precision=precision, bank_factors=bank_factors)
    if lhs is None or lhs.dtype is not sq.dtype:
        lhs = _decode_av_lhs(sq, mc_q)
    if bank_factors is not None and bank_factors[0].dtype == lhs.dtype:
        rhs, bias = bank_factors
    else:
        rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)             # (Vc, 2K)
        bias = (mc_v ** 2 * inv_v).sum(-1) + lsum
    with (ieee_fp32_matmul() if lhs.dtype is torch.float32 else nullcontext()):
        out = lhs @ rhs.transpose(-1, -2) + bias                     # (..., N, Vc)
        if return_error_bound and sq.dtype is torch.float32 and precision == "fp32":
            absolute_sum = lhs.abs() @ rhs.abs().transpose(-1, -2) + bias.abs()
    if not return_error_bound:
        return out
    if sq.dtype is torch.float32 and precision == "fp32":
        gamma = ((2.0 * lhs.shape[-1] + 4.0) * torch.finfo(out.dtype).eps)
        error_bound = gamma * absolute_sum
    else:
        error_bound = torch.zeros_like(out)
    return out, error_bound


def _head_bank_factors(pb: PriorBank) -> Dict[str, torch.Tensor]:
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)
    mu_v = pb._decode_mu_table()
    inv_v = 1.0 / sigma_v
    c = mu_v.mean(dim=0, keepdim=True)
    mc_v = mu_v - c
    lsum = torch.log(sigma_v).sum(-1)
    work_mc_v, work_inv_v, work_lsum = mc_v, inv_v, lsum
    if pb.decode_av_precision == "fp64":
        work_mc_v, work_inv_v, work_lsum = mc_v.double(), inv_v.double(), lsum.double()
    rhs = torch.cat([work_inv_v, work_mc_v * work_inv_v], dim=-1)
    bias = (work_mc_v.square() * work_inv_v).sum(-1) + work_lsum
    return dict(c=c, mc_v=mc_v, inv_v=inv_v, lsum=lsum, rhs=rhs, bias=bias)


def _head_refined_candidate_decode(pb: PriorBank, mu_q: torch.Tensor, sigma_q: torch.Tensor, tau_eff: torch.Tensor,
                                   ranking_logits: torch.Tensor, ranking_bound: torch.Tensor,
                                   kernel_logits: torch.Tensor) -> torch.Tensor:
    winner = ranking_logits.argmax(dim=-1, keepdim=True)
    winner_lower = _outward_lower(ranking_logits.gather(-1, winner), ranking_bound.gather(-1, winner))
    candidates = _outward_upper(ranking_logits, ranking_bound) >= winner_lower
    candidates &= candidates.sum(dim=-1, keepdim=True) > 1
    b, n, v = candidates.nonzero(as_tuple=True)
    mu_v = pb._decode_mu_table().double()
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table().double(), eps=pb.eps)
    c = mu_v.mean(dim=0, keepdim=True)
    mc_v, sigma_v = (mu_v - c)[v], sigma_v[v]
    inv_v = 1.0 / sigma_v
    mc_q = mu_q[b, n].double() - c
    sigma = sigma_q[b, n].double()
    sq = sigma
    per_pos = pb.K + torch.log(sigma.clamp(min=pb.eps)).sum(-1)
    a_v = _pairwise_decode_av(sq, mc_q, mc_v, inv_v, torch.log(sigma_v).sum(-1))
    refined = kernel_logits.double()
    refined[b, n, v] = _decode_analytic_kl_logits(a_v, per_pos, tau_eff.double(), torch.float64)
    return refined


def _head_decode_diagonal(pb: PriorBank, mu_q: torch.Tensor, sigma_q: torch.Tensor,
                          tau_eff: torch.Tensor) -> torch.Tensor:
    factors = _head_bank_factors(pb)
    c, mc_v, inv_v = factors["c"], factors["mc_v"], factors["inv_v"]
    mc_q = mu_q - c
    ranking_guard = pb.decode_ranking_fp64_escalation
    a_result = _head_decode_av(
        sigma_q, mc_q, mc_v, inv_v, factors["lsum"],
        return_error_bound=ranking_guard, precision=pb.decode_av_precision,
        bank_factors=(factors["rhs"], factors["bias"]))
    if ranking_guard:
        a_v, a_bound = a_result
    else:
        a_v, a_bound = a_result, None
    per_pos = pb.K + torch.log(sigma_q.clamp(min=pb.eps)).sum(-1, keepdim=True)
    if ranking_guard:
        unigram = pb._unigram_bias() if pb.decode_unigram_prior else None
        ranking_logits, ranking_bound = _decode_analytic_kl_logits(
            a_v, per_pos, tau_eff, mu_q.dtype, a_error_bound=a_bound,
            unigram_bias=unigram, return_error_bound=True)
        if _final_ranking_uncertain(ranking_logits, ranking_bound):
            return _head_refined_candidate_decode(
                pb, mu_q, sigma_q, tau_eff, ranking_logits, ranking_bound,
                _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype))
    return _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype)


def _head_chunk_loop(pb: PriorBank, lhs: torch.Tensor, per_pos: torch.Tensor, sq: torch.Tensor, mc_q: torch.Tensor,
                     mu_q: torch.Tensor, mu_v_all: torch.Tensor, sigma_v_all: torch.Tensor, c: torch.Tensor,
                     u_all: Optional[torch.Tensor], tau_eff: torch.Tensor, targets: torch.Tensor,
                     chunk: int) -> torch.Tensor:
    r"""The chunk loop both 5bb0a57 KL decode CEs ran on the torch route; returns the (B, N) CE rows."""
    V = pb.vocab_size

    def _chunk_summaries(lhs_: torch.Tensor, per_pos_: torch.Tensor, mu_v_c: torch.Tensor, inv_v_c: torch.Tensor,
                         lsum_c: torch.Tensor, in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                         u_c: Optional[torch.Tensor], sq_: torch.Tensor,
                         mc_q_: torch.Tensor) -> 'tuple[Optional[torch.Tensor], ...]':
        a_v = _head_decode_av(sq_, mc_q_, mu_v_c, inv_v_c, lsum_c, lhs=lhs_, precision=pb.decode_av_precision)
        logit_chunk = _decode_analytic_kl_logits(a_v, per_pos_, tau_eff, lhs_.dtype)
        if u_c is not None:
            logit_chunk = logit_chunk + u_c.to(logit_chunk.dtype)
        return _decode_ce_chunk_summaries(logit_chunk, in_chunk_f, local_idx)

    lse_chunks, max_chunks, target_chunks, in_chunks = [], [], [], []
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        mc_v_c = (mu_v_all[v0:v1] - c)
        inv_v_c = 1.0 / sigma_v_all[v0:v1]
        lsum_c = torch.log(sigma_v_all[v0:v1]).sum(-1)
        u_c = u_all[v0:v1] if u_all is not None else None
        in_chunk = (targets >= v0) & (targets < v1)
        in_chunk_f = in_chunk.to(mu_q.dtype)
        local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)
        grad_active = torch.is_grad_enabled() and any(
            value is not None and value.requires_grad for value in
            (lhs, per_pos, mu_v_all, sigma_v_all, tau_eff))
        activation_bytes = _decode_ce_chunk_activation_bytes(
            lhs, V if pb.decode_ce_checkpoint == "auto_aggregate" else v1 - v0, inner=2,
            workspace_bytes_per_scalar=(8 if pb.decode_ce_checkpoint == "auto_aggregate"
                                        and pb.decode_av_precision == "fp64" else None))
        if _decode_ce_should_checkpoint(pb.decode_ce_checkpoint, grad_active, activation_bytes):
            chunk_result = _checkpoint.checkpoint(
                _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx, u_c, sq, mc_q,
                use_reentrant=False)
        else:
            chunk_result = _chunk_summaries(lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx, u_c,
                                            sq, mc_q)
        lse_chunk, contrib, maximum = chunk_result[:3]
        lse_chunks.append(lse_chunk)
        max_chunks.append(maximum)
        target_chunks.append(contrib)
        in_chunks.append(in_chunk)
    return _decode_ce_combine_summaries(lse_chunks, max_chunks, target_chunks, in_chunks)


_HEAD_CALLS: List[str] = []


def _head_decode_ce_diagonal_chunked(
    self: PriorBank, mu_q: torch.Tensor, sigma_q: torch.Tensor, targets: torch.Tensor,
    *, tau: Optional[float] = None, chunk_size: Optional[int] = None, ignore_index: int = -100,
    return_stats: bool = False, canonical_frame: Optional[CanonicalFrameContext] = None,
) -> Any:
    _HEAD_CALLS.append("diagonal_chunked")
    mu_q, sigma_q = self._query_in_decode_frame(mu_q, sigma_q, canonical_frame)
    has_ignored = self._validate_fused_ce_targets(targets, ignore_index=ignore_index,
                                                  expected_shape=mu_q.shape[:-1])
    original_valid = targets != ignore_index
    if has_ignored:
        mu_q = mu_q[original_valid].unsqueeze(0)
        sigma_q = sigma_q[original_valid].unsqueeze(0)
        targets = targets[original_valid].unsqueeze(0)
    tau_eff = self._tau_eff(tau)
    chunk = self.decode_chunk_size if chunk_size is None else chunk_size
    sigma_v_all = bounded_variance_from_log(self._decode_sigma_log_table(), eps=self.eps)
    mu_v_all = self._decode_mu_table()
    c = mu_v_all.mean(dim=0, keepdim=True)
    u_all = self._unigram_bias() if self.decode_unigram_prior else None
    mc_q = mu_q - c
    lhs = _decode_av_lhs(sigma_q, mc_q)
    per_pos = self.K + torch.log(sigma_q.clamp(min=self.eps)).sum(-1, keepdim=True)
    valid = targets != ignore_index
    assert not lhs.is_cuda                                               # the reference copies the torch route only
    ce_per_pos = _head_chunk_loop(self, lhs, per_pos, sigma_q, mc_q, mu_q, mu_v_all, sigma_v_all, c, u_all,
                                  tau_eff, targets, chunk)
    ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
    return _decode_ce_result(ce, original_valid, return_stats)


def _head_decode_ce_full_chunked(
    self: PriorBank, mu_q: torch.Tensor, sigma_q: torch.Tensor, targets: torch.Tensor,
    *, tau: Optional[float] = None, chunk_size: Optional[int] = None, ignore_index: int = -100,
    canonical_frame: Optional[CanonicalFrameContext] = None, return_stats: bool = False,
) -> Any:
    _HEAD_CALLS.append("full_chunked")
    mu_q, sigma_q, factor_q, spd_ok = self._query_in_decode_frame_for_full_decode(mu_q, sigma_q, canonical_frame)
    has_ignored = self._validate_fused_ce_targets(targets, ignore_index=ignore_index,
                                                  expected_shape=mu_q.shape[:-1])
    original_valid = targets != ignore_index
    if has_ignored:
        mu_q = mu_q[original_valid].unsqueeze(0)
        sigma_q = sigma_q[original_valid].unsqueeze(0)
        factor_q = factor_q[original_valid].unsqueeze(0)
        spd_ok = spd_ok[original_valid].unsqueeze(0)
        targets = targets[original_valid].unsqueeze(0)
    tau_eff = self._tau_eff(tau)
    chunk = self.decode_chunk_size if chunk_size is None else chunk_size
    sigma_v_all = bounded_variance_from_log(self._decode_sigma_log_table(), eps=self.eps)
    mu_v_all = self._decode_mu_table()
    c = mu_v_all.mean(dim=0, keepdim=True)
    u_all = self._unigram_bias() if self.decode_unigram_prior else None
    diag_sq, logdet_q, spd_ok = self._full_cov_query_invariants_from_prepared(sigma_q, factor_q, spd_ok)
    mc_q = mu_q - c
    lhs = _decode_av_lhs(diag_sq, mc_q)
    per_pos = self.K + logdet_q.unsqueeze(-1)
    final_ok = spd_ok
    _count_decode_logdet_fallback(final_ok)
    valid = (targets != ignore_index) & final_ok
    scored_valid = original_valid.clone()
    if has_ignored:
        scored_valid[original_valid] = valid.flatten()
    else:
        scored_valid = valid
    assert not lhs.is_cuda
    ce_per_pos = _head_chunk_loop(self, lhs, per_pos, diag_sq, mc_q, mu_q, mu_v_all, sigma_v_all, c, u_all,
                                  tau_eff, targets, chunk)
    ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
    return _decode_ce_result(ce, scored_valid, return_stats)


_HEAD_CE = {"diagonal_chunked": _head_decode_ce_diagonal_chunked, "full_chunked": _head_decode_ce_full_chunked}


def _use_head_fused_ce(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    registration = bank_module._DECODERS[mode]
    monkeypatch.setitem(bank_module._DECODERS, mode, dataclasses.replace(registration, fused_ce=_HEAD_CE[mode]))


def _same(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))


# ------------------------------------------------------------------------------------- live route, whole model

_PROFILES = {
    # The launcher's route at tiny scale (followed when it decodes through diagonal_chunked), and the frozen
    # reference route, which does not depend on the launcher.
    "live": lambda: live_config(),
    "reference_k8": lambda: reference_k8_config(),
}


def _model_step(cfg: Any) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], Optional[str]]:
    torch.manual_seed(0)
    model = VFEModel(cfg)
    g = torch.Generator().manual_seed(1)
    if cfg.decode_unigram_prior:
        model.prior_bank.set_unigram_log_prior(torch.randint(1, 30, (cfg.vocab_size,), generator=g).float())
    tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
    targets[0, 1] = -100                                                 # one ignored position: the packed rows
    torch.manual_seed(2)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    return loss.detach(), [p.grad for p in model.parameters()], model.prior_bank.realized_decode_ce_kernel


@pytest.mark.parametrize("profile", sorted(_PROFILES))
def test_live_route_loss_and_every_gradient_are_bitwise_the_previous_decode(
        profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""One forward and backward of the tiny model: loss and every parameter gradient are ``torch.equal`` under the
    current decode CE and under the 5bb0a57 copy."""
    cfg = _PROFILES[profile]()
    if cfg.decode_mode != "diagonal_chunked" or cfg.decode_av_precision != "fp32":
        pytest.skip(f"the launcher decodes through {cfg.decode_mode!r} at {cfg.decode_av_precision!r}; "
                    "reference_k8 pins the fp32 diagonal_chunked route")
    assert not cfg.decode_backward_ieee_fp32
    loss, grads, kernel = _model_step(cfg)
    assert kernel == "torch"
    _HEAD_CALLS.clear()
    _use_head_fused_ce(monkeypatch, "diagonal_chunked")
    head_loss, head_grads, _ = _model_step(cfg)
    assert _HEAD_CALLS == ["diagonal_chunked"]
    assert torch.equal(loss, head_loss), (loss, head_loss)
    assert len(grads) == len(head_grads) and any(g is not None for g in grads)
    assert all(_same(a, b) for a, b in zip(grads, head_grads))


# ------------------------------------------------------------------------------------- the CE kernels directly

def _bank(mode: str, checkpoint: str, *, escalation: bool = False, flat_unigram: bool = False) -> PriorBank:
    full = mode == "full_chunked"
    torch.manual_seed(0)
    pb = make_prior_bank(23, 4, 3, decode_mode=mode, family="gaussian_full" if full else "gaussian_diagonal",
                         diagonal_covariance=not full, decode_tau=0.3, mu_init_std=0.7, decode_chunk_size=7,
                         decode_ce_checkpoint=checkpoint, decode_unigram_prior=True, untie_decode_bank=True,
                         decode_ranking_fp64_escalation=escalation, irrep_dims=[2, 2])
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        pb.decode_sigma_log_embed.copy_(0.6 * torch.randn(23, 4, generator=g))
        pb.decode_sigma_log_embed[0, 0] = math.log(pb.eps) - 1.0        # a table entry at the eps floor
    counts = torch.randint(1, 30, (23,), generator=g).float()
    pb.set_unigram_log_prior(torch.full_like(counts, 5.0) if flat_unigram else counts)
    return pb


def _query(mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(2)
    mu = 0.8 * torch.randn(2, 5, 4, generator=g)
    if mode == "full_chunked":
        a = 0.4 * torch.randn(2, 5, 4, 4, generator=g)
        sigma = a @ a.transpose(-1, -2) + 0.2 * torch.eye(4)
    else:
        sigma = 0.3 + torch.rand(2, 5, 4, generator=g)
        sigma[0, 0, 0] = 1e-6                                           # exactly at eps: the clamp is the identity
    targets = torch.randint(0, 23, (2, 5), generator=g)
    targets[1, 3] = -100
    return mu, sigma, targets


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
@pytest.mark.parametrize("checkpoint", ["never", "always"])
def test_kl_decode_ce_value_counts_and_gradients_are_bitwise_the_previous_loops(mode: str, checkpoint: str) -> None:
    results = []
    for impl in ("current", "head"):
        pb = _bank(mode, checkpoint)
        mu0, sigma0, targets = _query(mode)
        mu, sigma = mu0.clone().requires_grad_(), sigma0.clone().requires_grad_()
        fn = getattr(pb, "decode_ce_" + mode) if impl == "current" else _HEAD_CE[mode].__get__(pb)
        result = fn(mu, sigma, targets, return_stats=True)
        leaves = [mu, sigma, pb.decode_mu_embed, pb.decode_sigma_log_embed, pb.decode_log_scale]
        grads = torch.autograd.grad(result.ce, leaves, allow_unused=True)
        results.append((result, grads))
    (current, grads), (head, head_grads) = results
    assert torch.equal(current.ce, head.ce), (current.ce, head.ce)
    assert torch.equal(current.scored_tokens, head.scored_tokens)
    assert torch.equal(current.excluded_tokens, head.excluded_tokens)
    assert all(_same(a, b) for a, b in zip(grads, head_grads))


@pytest.mark.parametrize("escalation", [False, True])
def test_dense_diagonal_decode_logits_are_bitwise_the_previous_decode(escalation: bool,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The inference logits (``decode``), with and without the float32 ranking guard. Queries at the midpoint of two
    templates of one variance under a flat unigram prior are exact ties, which the guard refines in float64."""
    pb = _bank("diagonal_chunked", "never", escalation=escalation, flat_unigram=True)
    mu, sigma, _ = _query("diagonal_chunked")
    with torch.no_grad():
        pb.decode_sigma_log_embed[:10].fill_(0.3)
        mu_v = pb._decode_mu_table().detach()
        mu[1] = 0.5 * (mu_v[:5] + mu_v[5:10])                          # (5, K) midpoints: ties in exact arithmetic
    refined: List[int] = []
    original = bank_module._refined_candidate_decode
    monkeypatch.setattr(bank_module, "_refined_candidate_decode",
                        lambda *a, **k: (refined.append(1), original(*a, **k))[1])
    with torch.no_grad():
        got = pb.decode(mu, sigma)
        want = _head_decode_diagonal(pb, mu, sigma, pb._tau_eff(None)) + pb._unigram_bias()
    assert len(refined) == (1 if escalation else 0)
    assert got.dtype == want.dtype
    assert torch.equal(got, want)
