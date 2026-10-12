r"""PriorBank for VFE_3.0: learnable Gaussian vocab priors + the KL decode boundary.

Holds the routed per-vocabulary prior pi_v = N(mu_v, Sigma_v) with gauge frame phi_v as
PARAMETER TABLES (nn.Parameter -- priors, not neural maps; the no-NN rule bans
nn.Linear/MLP/activations, not learnable parameters). encode(token_ids) looks the active tables
up into the initial belief (q = p); decode(mu_q, sigma_q) scores the posterior against every
active prior as logits = -KL(q || pi_v)/tau_eff (the divergence seam), replacing a linear output
projection.

Modularity:
    encode_mode registry -- ``per_token`` (table lookup, default) and implemented
        experimental encoders.
    decode_mode registry -- ``diagonal_chunked`` (fused decode+CE; inference logits from the
        dense diagonal closed form); ``full_chunked`` (full-cov KL via the diagonal-prior closed
        form); ``expected_likelihood_chunked`` (log N(mu_q; mu_v, Sigma_q + Sigma_v)
        Gaussian-convolution scoring, diagonal only); and the ``linear`` readout kernel. The dense
        ``_decode_diagonal`` / ``_decode_full`` / ``_decode_family`` helpers are unregistered private
        oracles (historical ``diagonal``/``full`` configs migrate to the chunked modes, and the
        retired generic ``family``/``family_chunked`` modes to the kernel of the family's rank).

Decode seam (PB-14): the authoritative ``reference_decode`` scores logits =
-D_configured(q || pi_v)/tau_eff through the configured family (``self.family``) and divergence
functional (``self.divergence_family`` at ``self.renyi_order``). The fast diagonal/full kernels are
the OPTIMIZED gaussian_* + renyi(alpha=1) implementations; config requires a ``family_consistent``
decoder for any non-Gaussian family on a prior-bank decoder. The registry seam is
honored at the COVARIANCE-STRUCTURE granularity (``DecodeRegistration.covariance_kinds``): a new
covariance structure or a new family-consistent readout is added by writing-and-registering a decode
kernel, never by editing a call site. The full kernels score a full q against the intentionally
DIAGONAL vocabulary-prior table (promoted with diag_embed only when the family is full).
``reference_decode`` is the slow per-V seam-call cross-check the fused canonical kernels are pinned
to EXACTLY (and under ``log_softmax``) on the canonical path.
"""

import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional, Protocol, Tuple

import torch
import torch.utils.checkpoint as _checkpoint
from torch import nn

from vfe3.config import _REFERENCE_DECODE_MODES, VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.frame_actions import frame_matvec, reference_covariance, reference_variances

from vfe3.belief import BeliefState
from vfe3.divergence import get_family, get_functional, kl
from vfe3.families.base import _logdet_chol
from vfe3.runtime import ieee_fp32_matmul
from vfe3.numerics import (
    _count_decode_logdet_fallback,
    bounded_variance_from_log,
    safe_cholesky,
)


# --- decode a_v working precision (audit 2026-08-06 F32) -------------------------------------
# Every decode kernel needs
#     a_v = sum_k [ (sq_k + (mc_q_k - mc_v_k)^2) / sigma_v_k ] + sum_k log sigma_v_k,
# evaluated as mc_q^2 - 2 mc_q mc_v + mc_v^2 so the v-dependent part is a single (2K)-inner-product
# GEMM over the vocabulary -- the reason the chunked kernels are cheap. F32 filed this as a
# CANCELLATION defect in that expanded square and proposed differencing before squaring.
#
# MEASURED, and the attribution is wrong (K=20, V=64, uniform sigma_v, vs a float64 reference):
#
#   sigma_v   expanded fp32   exact fp32   expanded fp64   exact fp64
#   4e+00        5.20e-06      4.49e-06        1.42e-14      0.00e+00
#   1e-02        2.50e-03      1.52e-03        5.46e-12      0.00e+00
#   1e-04        2.22e-01      1.26e-01        4.66e-10      0.00e+00
#
# Differencing before squaring buys a factor of ~1.8. Doing the SAME expanded algebra in float64
# buys nine decades. So the defect is not the factorization, it is accumulating terms of size
# 1/sigma_v (2e5 at sigma_v=1e-4, against fp32's ~7 significant digits) and then dividing the
# result by 2*tau = 0.016. The exact form was implemented, measured, and dropped: a 1.8x accuracy
# gain does not pay for giving up the GEMM and materializing a (B, N, Vc, K) transient per chunk.
#
# "fp64" therefore keeps the expanded algebra and raises the working precision, and the island must
# extend THROUGH the logit -- a_v itself is large, so rounding it back to float32 before subtracting
# per_pos and dividing by tau would throw the accuracy straight back away. Callers preserve fp64 for the finished
# O(1) logit, not a_v. The severity is latent either way: sigma_log_embed trains freely against only
# the eps=1e-6 floor, so this is harmless at init (2.4e-4 induced logit error) and about one order
# of magnitude of shrinkage from material. Default "fp32" is bit-identical to every run on disk.
_DECODE_AV_PRECISIONS = ("fp32", "fp64")
_DECODE_AV_PRECISION: str = "fp32"

# Prior covariance table layouts (VFE3Config.prior_covariance_table): the (V, K) log-variance table, or the
# frame-free per-head Cholesky table (V, H, d(d+1)/2) of PriorBank._prior_covariance_blocks.
PRIOR_COVARIANCE_TABLES = ("diagonal", "block_cholesky")


def packed_cholesky_covariance(
    packed: torch.Tensor,                # (..., d(d+1)/2) tril_indices order: l on the diagonal, N below
    d:      int,
    *,
    eps:    float,                       # eigenvalue floor, the prior bank's eps
) -> torch.Tensor:                       # (..., d, d) C = L L^T + eps I
    r"""SPD block from its packed row-scaled Cholesky factor (``prior_covariance_table='block_cholesky'``).

    ``packed`` holds, in ``torch.tril_indices(d, d)`` order, a log-variance :math:`\ell_k` in each diagonal slot and
    the strictly lower unit-triangular part :math:`N` below it:

    .. math::
        L = \operatorname{diag}(e^{\ell/2})(I + N),\qquad C = LL^\top + \epsilon I,\qquad
        C_{kk} = e^{\ell_k}\Bigl(1 + \sum_{j<k}N_{kj}^2\Bigr) + \epsilon.

    The factor is a bijection of :math:`(\ell, N)` onto lower-triangular matrices with positive diagonal, so
    :math:`LL^\top` ranges over SPD(d) with :math:`d(d+1)/2` parameters. The coordinates match the diagonal table's
    optimizer meaning (audit 2026-10-09 round 2, #17): :math:`\ell_k` is in log-variance units (a unit step moves
    :math:`\log C_{kk}` by one unit, as on ``sigma_log_embed``; the previous :math:`\log L_{kk}` moved it by two),
    and :math:`N` is dimensionless, so the correlations of :math:`LL^\top` depend on :math:`N` only and a step on
    :math:`N` has the same effect at every variance scale (the previous absolute off-diagonal entries of :math:`L`
    stepped correlations as :math:`1/\sqrt{\sigma}`). :math:`\epsilon I` is the eigenvalue floor matching the diagonal
    table's ``eps`` (:func:`vfe3.numerics.bounded_variance_from_log` clamps each variance at :math:`\epsilon`; here
    the floor is added, which keeps the map smooth). :math:`\ell` is clamped at 80 before exponentiation, the
    log-variance table's ``max_log``, only to keep float32 finite.
    """
    rows, cols = torch.tril_indices(d, d, device=packed.device)
    if packed.shape[-1] != rows.numel():
        raise ValueError(f"packed Cholesky entries must have d(d+1)/2 = {rows.numel()} entries, "
                         f"got {packed.shape[-1]}")
    on_diagonal = rows == cols                                                       # (P,)
    unit = torch.where(on_diagonal, torch.ones_like(packed), packed)                 # (..., P) I + N entries
    factor = packed.new_zeros(*packed.shape[:-1], d * d).index_copy(
        -1, rows * d + cols, unit).reshape(*packed.shape[:-1], d, d)                 # (..., d, d) I + N
    log_variance = packed[..., on_diagonal].clamp(max=80.0)                          # (..., d) l, diag slots in row order
    factor = torch.exp(0.5 * log_variance).unsqueeze(-1) * factor                    # diag(e^{l/2}) (I + N)
    eye = torch.eye(d, dtype=packed.dtype, device=packed.device)
    return factor @ factor.transpose(-1, -2) + eps * eye


def set_decode_av_precision(precision: str) -> str:
    r"""Set the process-wide decode ``a_v`` working precision; returns the previous value."""
    global _DECODE_AV_PRECISION
    if precision not in _DECODE_AV_PRECISIONS:
        raise ValueError(
            f"decode_av_precision must be one of {_DECODE_AV_PRECISIONS}, got {precision!r}")
    previous = _DECODE_AV_PRECISION
    _DECODE_AV_PRECISION = precision
    return previous


def decode_av_precision() -> str:
    r"""Return the active decode ``a_v`` working precision."""
    return _DECODE_AV_PRECISION


DECODE_CE_CHECKPOINT_AUTO_BYTES: int = 2 * 1024 ** 3   # 2 GiB; see decode_ce_checkpoint in config.py

def _decode_ce_chunk_summaries(
    logits:       torch.Tensor,          # (B, N, Vc) final scores
    in_chunk:     torch.Tensor,          # (B, N) target-membership mask
    local_idx:    torch.Tensor,          # (B, N) safe chunk-local target indices
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return max-centered LSE/target summaries.

    With the detached chunk maximum m = max_v(logit_v), returns L = logsumexp_v(logit_v - m), the
    centered target logit and m, so the CE never subtracts two large absolute scores. Detaching m
    retains the CE derivative (the CE is invariant to the shift).
    """
    maximum = logits.amax(dim=-1).detach()
    center = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    shifted = logits - center.unsqueeze(-1)
    lse = torch.logsumexp(shifted, dim=-1)
    gathered = shifted.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
    target = torch.where(in_chunk > 0, gathered, torch.zeros_like(gathered))
    return lse, target, maximum


def _decode_ce_combine_summaries(
    lse_chunks:    List[torch.Tensor],
    max_chunks:    List[torch.Tensor],
    target_chunks: List[torch.Tensor],
    in_chunks:     List[torch.Tensor],
) -> torch.Tensor:                       # (B, N) CE rows
    r"""Compute logsumexp_c(L_c) - target relative to the global maximum M = max_c m_c."""
    maxima = torch.stack(max_chunks)
    maximum = maxima.amax(dim=0)
    any_finite = torch.isfinite(maximum)
    maximum = torch.where(any_finite, maximum, torch.zeros_like(maximum))
    shifts = torch.where(torch.isfinite(maxima), maxima - maximum, torch.zeros_like(maxima))
    lse = torch.stack(lse_chunks) + shifts
    # A generic family may exclude rows with no finite class; keep their backward finite.
    lse = torch.where(any_finite.unsqueeze(0), lse, torch.zeros_like(lse))
    target_shifts = torch.where(torch.stack(in_chunks), shifts, torch.zeros_like(shifts))
    return torch.logsumexp(lse, dim=0) - (torch.stack(target_chunks) + target_shifts).sum(dim=0)


def _decode_ce_workspace_scalar_bytes(
    ref: torch.Tensor,
    workspace_bytes_per_scalar: Optional[int],
) -> int:
    r"""Resolve the optional scalar-byte override while keeping byte arithmetic integral."""
    scalar_bytes = ref.element_size() if workspace_bytes_per_scalar is None else workspace_bytes_per_scalar
    if type(scalar_bytes) is not int or scalar_bytes <= 0:
        raise ValueError("workspace_bytes_per_scalar must be a positive integer")
    return scalar_bytes


def _decode_ce_chunk_activation_bytes(
    ref:         torch.Tensor,            # (B, N, ...) tensor sharing the closure's (B, N) and dtype
    chunk_width: int,                     # Vc for this iteration
    inner:       int                = 1,   # extra trailing-dim multiplier (e.g. K or K*K) the
                                            # closure's largest workspace carries beyond (B, N, Vc)
    *,
    workspace_bytes_per_scalar: Optional[int] = None,
) -> int:
    r"""Bytes of the largest (B, N, chunk_width[, inner]) tensor a decode_ce_*_chunked kernel's
    ``_chunk_summaries`` closure actually materializes for THIS chunk, at ``ref``'s dtype -- the
    vocabulary width callers select for transient or aggregate-retention accounting.
    Computed from the real shapes/dtype in play (batch, positions, chunk_width, dtype.itemsize),
    not a guessed formula; ``inner`` lets callers whose chunk workspace carries an extra (K) axis
    (``decode_ce_expected_likelihood_chunked``) report that workspace's true size rather than
    under-counting it as (B, N, Vc).
    """
    scalar_bytes = _decode_ce_workspace_scalar_bytes(ref, workspace_bytes_per_scalar)
    batch, positions = ref.shape[0], ref.shape[1]
    return batch * positions * chunk_width * inner * scalar_bytes


def _decode_ce_should_checkpoint(
    checkpoint_mode:  str,                # "always" | "never" | "auto" | "auto_aggregate"
    grad_active:      bool,               # torch.is_grad_enabled() and the checkpointed input requires_grad
    activation_bytes: int,                # _decode_ce_chunk_activation_bytes(...) for this chunk
) -> bool:
    r"""Central dispatch for every ``decode_ce_*_chunked`` kernel's per-chunk gradient checkpoint.

    Grad-inactive calls (eval / ``no_grad``, or a checkpointed input that itself carries no grad)
    never checkpoint in any mode -- there is nothing for a checkpoint to save memory against, and
    checkpointing it anyway would only pay a pointless recompute. Under grad: "always" reproduces
    the pre-2026-08-07 unconditional behavior (checkpoint every chunk); "never" always keeps the
    chunk workspace live for backward. Both automatic modes compare the caller's estimate with
    ``DECODE_CE_CHECKPOINT_AUTO_BYTES``: "auto" keeps historical route-specific accounting;
    "auto_aggregate" opts into whole-vocabulary retention accounting and extra recomputation.
    """
    if not grad_active:
        return False
    if checkpoint_mode == "always":
        return True
    if checkpoint_mode == "never":
        return False
    return activation_bytes > DECODE_CE_CHECKPOINT_AUTO_BYTES               # automatic modes


def _decode_av_lhs(
    sq:          torch.Tensor,                   # (..., N, K) query variances
    mc_q:        torch.Tensor,                   # (..., N, K) centered query means
) -> torch.Tensor:                        # (..., N, 2K)
    r"""The v-independent left factor, hoisted out of the chunk loop."""
    return torch.cat([sq + mc_q ** 2, -2.0 * mc_q], dim=-1)


# The a_v interval certificate bounds the fp32 inner product of length n = 2K by
# gamma = (2n + 4) u sum |l_k r_k| with u = torch.finfo(float32).eps. That bound holds only for IEEE
# fp32 multiplication, so the a_v GEMM runs under the shared IEEE fp32 pin.
_ieee_fp32_matmul = ieee_fp32_matmul


def _decode_av(
    sq:    torch.Tensor,                  # (..., N, K) query variances (diag(Sigma_q) for the full family)
    mc_q:  torch.Tensor,                  # (..., N, K) centered query means
    mc_v:  torch.Tensor,                  # (Vc, K) centered prior means
    inv_v: torch.Tensor,                  # (Vc, K) 1/sigma_v
    lsum:  torch.Tensor,                  # (Vc,) sum_k log sigma_v

    *,
    lhs:         Optional[torch.Tensor] = None,  # (..., N, 2K) precomputed _decode_av_lhs
    return_error_bound: bool = False,
    precision: Optional[str] = None,
    bank_factors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> 'torch.Tensor | tuple[torch.Tensor, torch.Tensor]':
    r"""``a_v`` under the active working precision -- the ONE place this algebra is written.

    Shared by all four decode kernels (diagonal/full x chunked-CE/logits) so the policy cannot apply
    to some and not others, which would let the fused CE and the logits it is pinned against
    disagree. Returns float64 under ``"fp64"`` ON PURPOSE: the caller must keep the island open
    through ``-0.5 * (a_v - per_pos) / tau_eff`` and retain fp64 logits for ranking, because a_v is the
    large quantity and rounding it here would discard the entire gain. See the module note.
    """
    precision = _DECODE_AV_PRECISION if precision is None else precision
    if precision == "fp64" and sq.dtype is not torch.float64:
        return _decode_av(
            sq.double(), mc_q.double(), mc_v.double(), inv_v.double(), lsum.double(),
            lhs=None,
            return_error_bound=return_error_bound, precision=precision, bank_factors=bank_factors)
    if lhs is None or lhs.dtype is not sq.dtype:
        lhs = _decode_av_lhs(sq, mc_q)
    if bank_factors is not None and bank_factors[0].dtype == lhs.dtype:
        rhs, bias = bank_factors
    else:
        rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)             # (Vc, 2K)
        bias = (mc_v ** 2 * inv_v).sum(-1) + lsum
    # The fp32 GEMMs run under IEEE fp32 whatever the process-global matmul policy, so the eps-based
    # certificate below describes the arithmetic that actually ran. fp64 GEMMs ignore the policy.
    with (_ieee_fp32_matmul() if lhs.dtype is torch.float32 else nullcontext()):
        out = lhs @ rhs.transpose(-1, -2) + bias                                  # (..., N, Vc)
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


def _decode_analytic_kl_logits(
    a_v:          torch.Tensor,
    per_pos:      torch.Tensor,
    tau_eff:      torch.Tensor,
    output_dtype: torch.dtype,
    *,
    a_error_bound: Optional[torch.Tensor] = None,
    unigram_bias: Optional[torch.Tensor] = None,
    return_error_bound: bool = False,
) -> 'torch.Tensor | tuple[torch.Tensor, torch.Tensor]':
    r"""Apply every final-score transform and propagate a conservative absolute error interval."""
    centered = 0.5 * (a_v - per_pos)
    kl_v = centered.clamp(min=0.0)
    # A final fp32 cast can collapse distinct near-tie scores after escalation.
    result_dtype = torch.float64 if kl_v.dtype is torch.float64 else output_dtype
    logits = (-kl_v / tau_eff).to(result_dtype)
    if not return_error_bound:
        return logits

    eps = torch.finfo(a_v.dtype).eps
    av_bound = torch.zeros_like(a_v) if a_error_bound is None else a_error_bound
    per_bound = 8.0 * eps * (per_pos.abs() + 1.0)
    # clamp is 1-Lipschitz, so the input interval remains conservative through max(x, 0).
    bound = 0.5 * (av_bound + per_bound) + eps * (a_v.abs() + per_pos.abs())
    tau_abs = tau_eff.abs().clamp_min(torch.finfo(tau_eff.dtype).tiny)
    tau_bound = 4.0 * eps * (tau_abs + 1.0)
    bound = (
        bound / tau_abs
        + kl_v.abs() * tau_bound / tau_abs.square()
        + eps * logits.abs()
    )
    ranking_logits = logits
    if unigram_bias is not None:
        bias = unigram_bias.to(logits.dtype)
        ranking_logits = logits + bias
        bias_bound = eps * (bias.abs() + 1.0)
        bound = bound + bias_bound + eps * (logits.abs() + bias.abs())
    positive_inf = torch.full(
        (), float("inf"), dtype=bound.dtype, device=bound.device)
    return ranking_logits, torch.nextafter(bound, positive_inf)


def _outward_lower(logits: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    r"""Lower interval endpoint with one explicit outward ulp."""
    negative_inf = torch.full(
        (), float("-inf"), dtype=logits.dtype, device=logits.device)
    return torch.nextafter(logits - bounds, negative_inf)


def _outward_upper(logits: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    r"""Upper interval endpoint with one explicit outward ulp."""
    positive_inf = torch.full(
        (), float("inf"), dtype=logits.dtype, device=logits.device)
    return torch.nextafter(logits + bounds, positive_inf)


def _final_ranking_uncertain(logits: torch.Tensor, bounds: torch.Tensor) -> bool:
    r"""Whether any competitor interval overlaps the nominal winner interval."""
    if logits.shape[-1] < 2 or logits.dtype is torch.float64:
        return False
    winner = logits.argmax(dim=-1, keepdim=True)
    winner_score = logits.gather(-1, winner).squeeze(-1)
    winner_bound = bounds.gather(-1, winner).squeeze(-1)
    winner_lower = _outward_lower(winner_score, winner_bound)
    upper = _outward_upper(logits, bounds)
    competitor_upper = upper.scatter(
        -1, winner, torch.full_like(winner, float("-inf"), dtype=upper.dtype)
    ).amax(dim=-1)
    # Only constant-size (B,N) summaries promote; no vocabulary-sized fp64 scan.
    return bool((winner_lower.double() <= competitor_upper.double()).any())


def _pairwise_decode_av(
    sq:          torch.Tensor,                   # (P, K) query variances
    mc_q:        torch.Tensor,                   # (P, K) centered query means
    mc_v:        torch.Tensor,                   # (P, K) centered prior means
    inv_v:       torch.Tensor,                   # (P, K) 1/sigma_v
    lsum:        torch.Tensor,                   # (P,)   sum_k log sigma_v
) -> torch.Tensor:                               # (P,)
    r"""The expanded-form ``a_v`` of :func:`_decode_av` for P explicit (query, prior) pairs.

    Same algebra, one prior per query row instead of a shared vocabulary table:
    :math:`a = \sum_k [(s_k + m_k^2) / \sigma_k - 2 m_k \bar m_k / \sigma_k
    + \bar m_k^2 / \sigma_k] + \ell`.
    """
    return (((sq + mc_q ** 2) - 2. * mc_q * mc_v + mc_v ** 2) * inv_v).sum(-1) + lsum


def _refined_candidate_decode(
    pb:             'PriorBank',
    mu_q:           torch.Tensor,   # (B, N, K) posterior means
    sigma_q:        torch.Tensor,   # (B, N, K) variances, or (B, N, K, K) prepared covariances
    tau_eff:        torch.Tensor,
    ranking_logits: torch.Tensor,   # (B, N, V) float32 final ranking scores
    ranking_bound:  torch.Tensor,   # (B, N, V) their certified absolute error bounds
    kernel_logits:  torch.Tensor,   # (B, N, V) float32 kernel logits (no unigram bias)
    *,
    full_covariance: bool,
) -> torch.Tensor:                  # (B, N, V) float64 kernel logits
    r"""Resolve an uncertain float32 ranking by recomputing only its candidates in float64.

    A candidate of row (b, n) is a v whose outward interval upper end reaches the nominal winner's
    outward lower end. Every non-candidate j therefore has exact score at most
    :math:`u_j < \ell_w \le s^*_w \le \max_v s^*_v`, so the exact argmax is a candidate, and the
    float64 candidate values resolve it as the full float64 recompute would. The returned kernel
    logits are the float32 values upcast (exact) with the candidates of uncertain rows replaced by
    float64 values. The unigram bias added by :meth:`PriorBank.decode` in float64 keeps every
    non-candidate inside its certified interval, so the final argmax matches a complete float64 recompute.
    Non-candidate VALUES keep float32 accuracy, exactly as when no row is uncertain; only the
    ranking is certified. Cost O(P K) float64 for P candidates, instead of the O(B N V K)
    recompute. The row and candidate selection reads one host index list, on the escalation path.
    """
    winner = ranking_logits.argmax(dim=-1, keepdim=True)
    winner_lower = _outward_lower(ranking_logits.gather(-1, winner), ranking_bound.gather(-1, winner))
    candidates = _outward_upper(ranking_logits, ranking_bound) >= winner_lower         # (B, N, V)
    candidates &= candidates.sum(dim=-1, keepdim=True) > 1                             # uncertain rows only
    b, n, v = candidates.nonzero(as_tuple=True)
    mu_v = pb._decode_mu_table().double()
    sigma_v = bounded_variance_from_log(pb._decode_sigma_log_table().double(), eps=pb.eps)
    c = mu_v.mean(dim=0, keepdim=True)                         # the full-vocabulary shift, as expanded
    mc_v, sigma_v = (mu_v - c)[v], sigma_v[v]                  # (P, K)
    inv_v = 1.0 / sigma_v
    mc_q = mu_q[b, n].double() - c                             # (P, K)
    sigma = sigma_q[b, n].double()
    if full_covariance:
        # The full-covariance helpers take (B, N, K, K) rows; the P candidates form one row batch.
        sq, logdet, _ = pb._full_cov_query_invariants(sigma.unsqueeze(0))
        sq, logdet = sq[0], logdet[0]
        per_pos = pb.K + logdet
    else:
        sq = sigma
        per_pos = pb.K + torch.log(sigma.clamp(min=pb.eps)).sum(-1)
    a_v = _pairwise_decode_av(sq, mc_q, mc_v, inv_v, torch.log(sigma_v).sum(-1))
    refined = kernel_logits.double()
    refined[b, n, v] = _decode_analytic_kl_logits(a_v, per_pos, tau_eff.double(), torch.float64)
    return refined


# Registries: mode name -> callable. Variants swap by config; add a variant by
# writing-and-registering it, never by editing call sites.
#   encode: fn(pb, token_ids) -> BeliefState
#   decode: fn(pb, mu_q, sigma_q, tau_eff) -> logits (B, N, V)
# ---------------------------------------------------------------------------
_ENCODERS: 'Dict[str, EncodeCallable]' = {}


@dataclass(frozen=True)
class EncodeRegistration:
    """An encode callable and registration-owned prior-table routing capabilities."""

    callable:               'EncodeCallable'
    can_omit_base_mean:     bool = False
    can_omit_base_variance: bool = False


_ENCODER_REGISTRATIONS: Dict[str, EncodeRegistration] = {}


@dataclass(frozen=True)
class DecodeRegistration:
    """A decode callable and all routing capabilities attached to that callable.

    ``covariance_kinds`` is the resolved set of family covariance structures the decoder scores
    ("diagonal" and/or "full"); config validates ``family_cov_kind(cfg.family) in covariance_kinds``
    rather than treating ``supports_full`` as an exclusive rank bit, so a dual-rank decoder (e.g.
    ``family``) accepts BOTH a diagonal and a full family. ``supports_full`` is retained (public,
    read by legacy callers) and stays coherent with the set: it is ``"full" in covariance_kinds``.
    ``family_consistent`` flags a decoder that scores logits = -D_configured(q||p_v)/tau_eff through
    the CONFIGURED family AND divergence functional (as opposed to the fast kernels' hardcoded
    gaussian alpha=1 KL); config requires a family_consistent decoder for any non-Gaussian family or
    noncanonical divergence on a prior-bank decoder.

    Direct construction ``DecodeRegistration(callable, supports_full, supports_chunked, fused_ce)``
    stays source-compatible: later capability fields default, and ``__post_init__`` derives the
    legacy singleton ``covariance_kinds`` from ``supports_full`` when it is not supplied.
    ``fused_ce_supports_stats`` is an explicit opt-in: only such registrations receive the new
    ``return_stats`` keyword, preserving legacy custom fused-callable signatures.
    """

    callable:          'DecodeCallable'
    supports_full:     bool
    supports_chunked:  bool
    fused_ce:          'Optional[FusedCECallable]'
    family_consistent: bool                        = False
    covariance_kinds:  'Optional[FrozenSet[str]]'  = None
    can_omit_base_mean: bool                       = False
    can_omit_base_variance: bool                   = False
    fused_ce_supports_stats: bool                  = False

    def __post_init__(self) -> None:
        if not callable(self.callable):
            raise TypeError("decoder callable must be callable")
        for field_name in (
            "supports_full",
            "supports_chunked",
            "family_consistent",
            "can_omit_base_mean",
            "can_omit_base_variance",
            "fused_ce_supports_stats",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"decoder {field_name} declaration must be bool")
        # Resolve the covariance-kind set. Omitted -> the legacy singleton derived from
        # supports_full (a frozen dataclass, so the resolved value is written via object.__setattr__).
        if self.covariance_kinds is None:
            resolved_kinds = frozenset({"full"} if self.supports_full else {"diagonal"})
        else:
            resolved_kinds = frozenset(self.covariance_kinds)
        if not resolved_kinds or not resolved_kinds <= {"diagonal", "full"}:
            raise ValueError(
                "decoder covariance_kinds must be a nonempty subset of "
                f"{{'diagonal', 'full'}}, got {sorted(resolved_kinds)}"
            )
        resolved_full = "full" in resolved_kinds
        if self.supports_full != resolved_full:
            raise ValueError(
                "decoder has contradictory rank metadata: "
                f"supports_full={self.supports_full} but covariance_kinds="
                f"{sorted(resolved_kinds)} implies supports_full={resolved_full}"
            )
        if self.fused_ce is not None and not callable(self.fused_ce):
            raise TypeError("decoder fused_ce must be callable or None")
        if self.supports_chunked != (self.fused_ce is not None):
            raise ValueError(
                "decoder has contradictory chunk metadata: supports_chunked must be true "
                "exactly when fused_ce is provided"
            )
        if self.fused_ce_supports_stats and self.fused_ce is None:
            raise ValueError("decoder cannot support fused CE stats without fused_ce")
        object.__setattr__(self, "covariance_kinds", resolved_kinds)


_DECODERS: Dict[str, DecodeRegistration] = {}


@dataclass(frozen=True)
class DecodeCEResult:
    """Opt-in fused-decoder result with the exact, unclamped reduction denominator."""

    ce:            torch.Tensor
    scored_tokens: torch.Tensor
    excluded_tokens: Optional[torch.Tensor] = None


def _decode_ce_result(
    ce:           torch.Tensor,
    valid:        torch.Tensor,
    return_stats: bool,
) -> 'torch.Tensor | DecodeCEResult':
    """Keep scalar CE as the default while exposing a device-side int64 count on request."""
    if not return_stats:
        return ce
    scored = valid.sum(dtype=torch.int64)
    excluded = valid.numel() - scored                      # no host-to-device copy of the count
    return DecodeCEResult(ce=ce, scored_tokens=scored, excluded_tokens=excluded)

# Once-per-process guard for the decode_unigram_prior=True-with-unset-table warning
# (the decode then degenerates to the current uniform-prior behavior).
_WARNED_UNIGRAM_UNSET: bool = False


def register_encode(
    name: str,

    *,
    override:               bool           = False,
    can_omit_base_mean:     bool = False,
    can_omit_base_variance: bool = False,
) -> 'Callable[[EncodeCallable], EncodeCallable]':
    """Decorator registering an encode kernel under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: 'EncodeCallable') -> 'EncodeCallable':
        if name in _ENCODERS and not override:
            raise KeyError(f"encode mode {name!r} already registered; pass override=True to replace")
        # Capabilities belong to this NAME registration, never to the callable object. An alias or
        # functools.wraps wrapper is therefore conservative unless this invocation declares otherwise.
        if type(can_omit_base_mean) is not bool or type(can_omit_base_variance) is not bool:
            raise TypeError("encoder base-table omission declarations must be bools")
        _ENCODERS[name] = fn
        _ENCODER_REGISTRATIONS[name] = EncodeRegistration(
            callable=fn,
            can_omit_base_mean=can_omit_base_mean,
            can_omit_base_variance=can_omit_base_variance,
        )
        return fn
    return _wrap


def get_encode(name: str) -> 'EncodeCallable':
    """Return the registered encode kernel for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    return _ENCODERS[name]


def get_encode_registration(name: str) -> EncodeRegistration:
    """Return the registration-owned encode capabilities for ``name`` (KeyError if absent)."""
    if name not in _ENCODERS or name not in _ENCODER_REGISTRATIONS:
        raise KeyError(
            f"no encode mode registered under {name!r}; available: {sorted(_ENCODERS)}"
        )
    registration = _ENCODER_REGISTRATIONS[name]
    if registration.callable is not _ENCODERS[name]:
        raise RuntimeError(f"encode mode {name!r} registry record is inconsistent")
    return registration


def register_decode(
    name: str,

    *,
    supports_full:     Optional[bool]              = None,
    supports_chunked:  bool                        = False,
    override:          bool                        = False,
    family_consistent: bool                        = False,
    fused_ce:          'Optional[FusedCECallable]'      = None,
    covariance_kinds:  'Optional[FrozenSet[str]]'       = None,
    can_omit_base_mean:     bool                        = False,
    can_omit_base_variance: bool                        = False,
    fused_ce_supports_stats: bool                       = False,
) -> 'Callable[[DecodeCallable], DecodeCallable]':
    """Decorator registering a decode kernel under ``name``.

    ``covariance_kinds`` is the resolved set of family covariance structures the decoder scores.
    OMITTED -> derive the legacy singleton from ``supports_full`` (``{"full"}`` when True, else
    ``{"diagonal"}``); SUPPLIED -> derive ``supports_full`` from membership (``"full" in kinds``) and
    reject an explicitly contradictory legacy ``supports_full``. Every existing
    ``register_decode(..., supports_full=True|False)`` call therefore keeps its old behavior.
    ``family_consistent`` marks a decoder that reads logits out through the CONFIGURED family and
    divergence functional. ``supports_chunked`` advertises a fused chunked-CE training path, whose
    callable is ``fused_ce``. ``fused_ce_supports_stats=True`` declares that callable's opt-in
    ``return_stats`` contract; the default is false so existing callables receive exactly their
    historical keyword set. The callable and all capabilities are replaced atomically, so an
    override cannot retain stale routing metadata from the prior registration.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    if covariance_kinds is None:
        resolved_full  = bool(supports_full) if supports_full is not None else False
        resolved_kinds = frozenset({"full"} if resolved_full else {"diagonal"})
    else:
        resolved_kinds = frozenset(covariance_kinds)
        if not resolved_kinds or not resolved_kinds <= {"diagonal", "full"}:
            raise ValueError(
                f"decode mode {name!r} covariance_kinds must be a nonempty subset of "
                f"{{'diagonal', 'full'}}, got {sorted(resolved_kinds)}"
            )
        resolved_full = "full" in resolved_kinds
        if supports_full is not None and bool(supports_full) != resolved_full:
            raise ValueError(
                f"decode mode {name!r} has contradictory metadata: supports_full={supports_full} "
                f"but covariance_kinds={sorted(resolved_kinds)} implies supports_full={resolved_full}"
            )

    def _wrap(fn: 'DecodeCallable') -> 'DecodeCallable':
        if name in _DECODERS and not override:
            raise KeyError(f"decode mode {name!r} already registered; pass override=True to replace")
        if supports_chunked != (fused_ce is not None):
            raise ValueError(
                f"decode mode {name!r} must declare supports_chunked=True exactly when fused_ce "
                f"is provided"
            )
        if type(fused_ce_supports_stats) is not bool:
            raise TypeError("decoder fused-CE stats declaration must be bool")
        if fused_ce_supports_stats and fused_ce is None:
            raise ValueError(
                f"decode mode {name!r} cannot declare fused-CE stats support without fused_ce"
            )
        # Capabilities belong only to this registration record. Callable identity, aliases, and
        # functools.wraps attributes cannot confer omission rights on a new registered name.
        if type(can_omit_base_mean) is not bool or type(can_omit_base_variance) is not bool:
            raise TypeError("decoder base-table omission declarations must be bools")
        _DECODERS[name] = DecodeRegistration(
            callable=fn,
            supports_full=resolved_full,
            supports_chunked=supports_chunked,
            fused_ce=fused_ce,
            family_consistent=family_consistent,
            covariance_kinds=resolved_kinds,
            can_omit_base_mean=can_omit_base_mean,
            can_omit_base_variance=can_omit_base_variance,
            fused_ce_supports_stats=fused_ce_supports_stats,
        )
        return fn
    return _wrap


def get_decode_registration(name: str) -> DecodeRegistration:
    """Return the complete registration record for ``name`` (KeyError if absent)."""
    if name not in _DECODERS:
        raise KeyError(
            f"no decode mode registered under {name!r}; available: {sorted(_DECODERS)}"
        )
    return _DECODERS[name]


def get_decode(name: str) -> 'DecodeCallable':
    """Return the registered decode kernel for ``name`` (KeyError if absent)."""
    return get_decode_registration(name).callable


class PriorBank(nn.Module):
    r"""Learnable Gaussian vocab priors; encode (lookup) and decode (-KL/tau_eff).

    The routed mean/variance tables and ``phi_embed`` (V, n_gen) parameterize
    pi_v = N(mu_v, exp(sigma_log_v)) with gauge frame phi_v. ``mu_embed`` and ``sigma_log_embed``
    are always allocated and are the ONLY belief prior. They are PRIORS (nn.Parameter), not a
    neural map: there is no nn.Linear/MLP/activation anywhere in this module. The
    scalar ``decode_log_scale`` sets the decode temperature; ``VFEModel`` always freezes it,
    fixing tau_eff at its initialized (or loaded) value.
    """

    output_proj_weight: Optional[nn.Parameter]   # (V, K) linear-decode weight; None for prior-bank modes
    output_proj_bias:   Optional[nn.Parameter]   # (V,) linear-decode log-unigram bias (None off linear)
    mu_embed:           nn.Parameter             # (V, K) token-prior mean
    sigma_log_embed:    Optional[nn.Parameter]   # (V, K) token-prior log variance; None under 'block_cholesky'

    def __init__(
        self,
        vocab_size:   int,
        K:            int,
        n_gen:        int,

        *,
        mu_init_std:         float = 0.02,
        sigma_init:          float = 1.0,
        phi_scale:           float = 0.01,
        decode_tau:          float = 1.0,
        eps:                 float = 1e-6,
        diagonal_covariance: bool  = True,
        family:              str   = "gaussian_diagonal",
        divergence_family:   str   = "renyi",
        renyi_order:         float = 1.0,
        encode_mode:         str   = "per_token",
        decode_mode:         str   = "diagonal_chunked",
        decode_chunk_size:   int   = 8192,
        decode_av_precision: Optional[str] = None,
        decode_ce_checkpoint: str  = "auto",
        decode_ranking_fp64_escalation: bool = VFE3Config.decode_ranking_fp64_escalation,

        unigram_kappa:        float = 1.0,
        decode_unigram_prior: bool  = False,
        untie_decode_bank:    bool  = False,

        irrep_dims:             Optional[List[int]] = None,
        gauge_group_name:       Optional[str]       = None,
        decode_coordinates:     str                 = "local",
        prior_covariance_table: str                 = "diagonal",
    ) -> None:
        super().__init__()
        if prior_covariance_table not in PRIOR_COVARIANCE_TABLES:
            raise ValueError(f"prior_covariance_table must be one of {PRIOR_COVARIANCE_TABLES}, "
                             f"got {prior_covariance_table!r}")
        if prior_covariance_table == "block_cholesky":
            problems = []
            if not irrep_dims or len(set(irrep_dims)) != 1:
                problems.append(f"equal head blocks (irrep_dims={irrep_dims})")
            if diagonal_covariance:
                problems.append("a full-layout family (gaussian_block or gaussian_full)")
            if decode_mode != "linear" and not untie_decode_bank:
                problems.append("untie_decode_bank=True (the decode bank keeps its own diagonal tables)")
            if problems:
                raise ValueError("prior_covariance_table='block_cholesky' requires " + ", ".join(problems))
        # 'block_cholesky' replaces the (V, K) log-variance table by per-head Cholesky factors (sigma_chol_embed).
        self.prior_covariance_table = prior_covariance_table
        if decode_coordinates not in ("local", "reference"):
            raise ValueError(f"decode_coordinates must be 'local' or 'reference', got {decode_coordinates!r}")
        if decode_coordinates == "reference" and decode_mode not in _REFERENCE_DECODE_MODES:
            raise ValueError(
                f"decode_coordinates='reference' requires decode_mode in {_REFERENCE_DECODE_MODES}; "
                f"got decode_mode={decode_mode!r}")
        # 'reference' decodes (xi, C) = (U^{-1} mu, U^{-1} Sigma U^{-T}) through the realized frame that
        # every decode boundary receives as ``canonical_frame`` (see ``_reference_query``).
        self.decode_coordinates = decode_coordinates
        self.decode_av_precision = (_DECODE_AV_PRECISION
                                    if decode_av_precision is None else decode_av_precision)
        if self.decode_av_precision not in _DECODE_AV_PRECISIONS:
            raise ValueError(f"invalid decode_av_precision {self.decode_av_precision!r}")
        if not isinstance(decode_ranking_fp64_escalation, bool):
            raise ValueError(
                "decode_ranking_fp64_escalation must be bool, got "
                f"{decode_ranking_fp64_escalation!r}")
        self.vocab_size = vocab_size
        self.K = K
        self.n_gen = n_gen
        self.decode_tau = decode_tau
        self.eps = eps
        self.diagonal_covariance = diagonal_covariance
        # The vocabulary prior and decode variance tables stay diagonal in EVERY family (PB-11).
        self.family = family
        # divergence_family / renyi_order drive the reference decode (reference_decode): logits =
        # -D_configured(q||p_v)/tau_eff scored through get_functional(divergence_family) at
        # alpha=renyi_order. The fast Gaussian kernels hardcode alpha=1 KL. The defaults are the KL
        # readout every projected route uses.
        self.divergence_family = divergence_family
        self.renyi_order = renyi_order
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode
        self.decode_chunk_size = decode_chunk_size
        # Autocast dtype for the linear decode's logits matmul (VFE3Config.decode_logits_amp); None
        # keeps it in the fp32 decode island. Set by the model after construction.
        self.decode_logits_amp_dtype: Optional[torch.dtype] = None
        self.decode_ce_checkpoint = decode_ce_checkpoint
        # The kernel that last ran decode_ce_expected_likelihood_chunked ('torch' | 'triton'; None before
        # the first call), recorded for summary.json (audit 2026-10-10 round 2, #42).
        self.realized_expected_likelihood_kernel: Optional[str] = None
        self.decode_ranking_fp64_escalation = decode_ranking_fp64_escalation
        self.unigram_kappa = unigram_kappa
        self.decode_unigram_prior = decode_unigram_prior
        self.gauge_group_name = gauge_group_name
        self.irrep_dims = irrep_dims
        self.n_gen = n_gen
        # Untying applies only to prior-bank decoders; linear is already independent.
        self.untie_decode_bank = untie_decode_bank and decode_mode != "linear"

        sigma_log_init = float(torch.log(torch.tensor(sigma_init)))
        # Resolve both kernels up front so an unregistered or registry-inconsistent encode/decode
        # mode fails at construction rather than at the first encode()/decode() call.
        get_encode_registration(encode_mode)
        get_decode_registration(decode_mode)
        # The routed token-prior tables ARE the belief prior, so both are always live Parameters.
        self.mu_embed = nn.Parameter(mu_init_std * torch.randn(vocab_size, K))
        if prior_covariance_table == "block_cholesky":
            # Frame-free per-type covariance C_v = blockdiag_h(L_vh L_vh^T + eps I), L = diag(e^{l/2})(I + N):
            # packed (V, H, d(d+1)/2) in torch.tril_indices(d, d) order, l (log-variance units) on the diagonal
            # slots and N below (packed_cholesky_covariance). Initialized at l = log sigma_init, N = 0, i.e.
            # L = sqrt(sigma_init) I, the covariance the diagonal table starts at (plus the eps floor); no RNG draw.
            n_heads, d = len(irrep_dims), irrep_dims[0]
            rows, cols = torch.tril_indices(d, d)
            packed = torch.where(rows == cols, torch.tensor(sigma_log_init), torch.tensor(0.0))
            self.register_parameter("sigma_log_embed", None)
            self.sigma_chol_embed = nn.Parameter(packed.expand(vocab_size, n_heads, packed.numel()).clone())
        else:
            self.sigma_log_embed = nn.Parameter(
                torch.full((vocab_size, K), sigma_log_init), requires_grad=True)
        # Draw the historical phi initialization so every later RNG-owned table keeps the same seed
        # position.
        initial_phi = phi_scale * torch.randn(vocab_size, n_gen)
        # One frame table serves both attention roles: the relational transport is
        # Omega_ij = U_i U_j^{-1} from this single table. There is no separate key frame.
        self.phi_embed = nn.Parameter(initial_phi)
        self.decode_log_scale = nn.Parameter(torch.zeros(1))

        # Arm-2 control (encode_mode='per_token_additive'): a NON-structural use of the SAME learned
        # (V, n_gen) phi table. A FROZEN random readout R (K, n_gen) maps each token's n_gen-dim code
        # to an additive K-dim mean shift, and encode returns phi=0 so Omega = exp(phi.G) = I (no gl(g)
        # transport). Isolates raw phi-table CAPACITY (V*n_gen learned params, matched to the gauge
        # cell) from the gl(g) generator STRUCTURE. R is a buffer (not a Parameter, so learned-param
        # count is unchanged), seeded for reproducibility, scaled 1/sqrt(n_gen) so the per-dim shift
        # std matches phi_scale at init. Deliberately breaks gauge equivariance -- that IS the control.
        if encode_mode == "per_token_additive":
            _r_gen = torch.Generator().manual_seed(0)
            self.register_buffer(
                "additive_R",
                torch.randn(K, n_gen, generator=_r_gen) / (float(n_gen) ** 0.5),
            )

        # decode_mode='linear': decode is a plain linear projection
        # logits = mu_q @ W^T through a learned (V, K) weight, the single authorized neural
        # exception (a lone linear output readout; see CLAUDE.md). Realized as a raw nn.Parameter
        # matmul -- NOT an nn.Linear/MLP -- so no neural-layer class enters the module. Created
        # only on the linear path so prior-bank modes carry no extra weight.
        # Xavier-uniform init (PyTorch's nn.Linear default), no bias (a constant shift in
        # V that softmax/cross-entropy absorbs). Encode stays the prior-bank lookup either way.
        if decode_mode != "linear":
            self.output_proj_weight = None
            self.output_proj_bias   = None
        else:
            self.output_proj_weight = nn.Parameter(torch.empty(vocab_size, K))
            nn.init.xavier_uniform_(self.output_proj_weight)
            # Per-vocab bias (always on; the decode_bias toggle was removed 2026-10-12): a
            # *per-class* bias is NOT a softmax-invariant constant shift -- it is a learned
            # log-unigram prior, and a 50k Zipfian vocab is the opposite of balanced, so the
            # bias-free map can represent token base rates only by spending rank-K mean capacity.
            # Zero-init -> logits bit-identical to a bias-free map at construction (drawn AFTER the
            # weight, so the weight's RNG is unchanged); the CE gradient drives it toward
            # log p(token). Routed to a weight-decay-free optimizer group in build_optimizer
            # (decaying a unigram prior toward zero biases it to flat).
            self.output_proj_bias = nn.Parameter(torch.zeros(vocab_size))

        # Unigram log-prior decode table (decode_unigram_prior=True): a non-trainable (V,) buffer
        # log pi_v holding the smoothed corpus unigram log-frequencies, added to EVERY decode
        # path's logits as kappa * log pi_v (the Bayes class prior; a DATA statistic set by
        # set_unigram_log_prior, not a learned parameter). Created only on the toggled path
        # (matching additive_R) so the default state_dict is byte-identical.
        # Init zeros = the current implicit uniform prior; decode warns once per process while
        # the table is still unset.
        if decode_unigram_prior:
            self.register_buffer("unigram_log_prior", torch.zeros(vocab_size))
            self._unigram_set = False                                   # flips on set_unigram_log_prior
        # Untied decode bank (untie_decode_bank=True, prior-bank modes only): decode reads its
        # OWN (V, K) tables decode_mu_embed / decode_sigma_log_embed, cloned from the token prior
        # tables the shared route would read (mu_embed / sigma_log_embed) so step 0 is
        # byte-identical, then trained separately. Encode and the alpha-KL self-coupling target keep
        # the original tables. Cloning draws no RNG, so the default path's table init is byte-unchanged.
        if self.untie_decode_bank:
            self.decode_mu_embed        = nn.Parameter(self._prior_mu_table().clone().detach())
            # Under 'block_cholesky' there is no diagonal prior table to clone; the decode bank starts at the
            # value that table would have held (sigma_init everywhere).
            decode_sigma_log = (torch.full((vocab_size, K), sigma_log_init) if self.sigma_log_embed is None
                                else self._prior_sigma_log_table())
            self.decode_sigma_log_embed = nn.Parameter(decode_sigma_log.clone().detach())

    def encode(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
    ) -> BeliefState:
        r"""Look up the per-token Gaussian prior as the initial belief (q = p)."""
        belief = get_encode(self.encode_mode)(self, token_ids)
        return belief

    def _prior_mu_table(self) -> torch.Tensor:
        r"""The (V, K) mean prior table feeding p_i: the token table ``mu_embed``.

        Routed through ONE accessor so encode (q_i(0)=p_i), the E-step self-coupling target
        alpha*KL(q_i||p_i), and the decode per-vocab readout -KL(q||p_v) all consume the SAME prior.
        """
        return self.mu_embed

    def _prior_sigma_log_table(self) -> torch.Tensor:
        r"""The (V, K) log-variance prior table feeding p_i; the sigma sibling of
        ``_prior_mu_table`` (see there): the token table ``sigma_log_embed`` (None under
        ``prior_covariance_table='block_cholesky'``, whose prior covariance is :meth:`_prior_covariance_blocks`)."""
        return self.sigma_log_embed

    def _prior_covariance_blocks(
        self,
        index: Optional[torch.Tensor] = None,    # (...) integer token ids; None reads every type
    ) -> torch.Tensor:                           # (..., H, d, d) C_v = L_v L_v^T per head
        r"""Per-head prior covariance blocks of ``prior_covariance_table='block_cholesky'``
        (:func:`packed_cholesky_covariance` of the gathered ``sigma_chol_embed`` rows, eigenvalue floor ``eps``)."""
        packed = self.sigma_chol_embed if index is None else self.sigma_chol_embed[index]   # (..., H, P)
        return packed_cholesky_covariance(packed, self.irrep_dims[0], eps=self.eps)

    def _diagonal_decoder_dispersion(
        self,
        mu_q:    torch.Tensor,                   # (..., N, K)
        sigma_q: Optional[torch.Tensor],         # (..., N, K) or (..., N, K, K)
    ) -> Optional[torch.Tensor]:
        r""":math:`\mathrm{diag}\,\Sigma_q` when a diagonal-bank Gaussian decoder receives a full-layout covariance.

        Reached by ``gaussian_block`` with ``decode_mode='diagonal_chunked'``. Against a diagonal bank
        :math:`p_v = \mathcal N(\mu_v, \mathrm{diag}\,s_v)`,

        .. math::
            2\,\mathrm{KL}(q\|p_v) = \sum_k\frac{\Sigma_{kk} + (\mu_k-\mu_{vk})^2}{s_{vk}} - K + \sum_k\log s_{vk}
              - \log\det\Sigma,

        so the vocabulary-dependent part reads only :math:`\mathrm{diag}\,\Sigma`, and the diagonal kernel's
        logits differ from :math:`-\mathrm{KL}/\tau` by the per-position constant
        :math:`\tfrac{1}{2\tau}(\log\det\Sigma - \sum_k\log\Sigma_{kk})`: the softmax and cross-entropy are exact.
        Every other decoder, and every diagonal covariance, is returned unchanged.
        """
        if (sigma_q is not None and sigma_q.dim() == mu_q.dim() + 1 and self.decode_mode != "linear"
                and not get_decode_registration(self.decode_mode).supports_full):
            return torch.diagonal(sigma_q, dim1=-2, dim2=-1)
        return sigma_q

    def _decode_mu_table(self) -> torch.Tensor:
        r"""The (V, K) mean table the DECODE boundary scores against: the untied decode table
        decode_mu_embed when untie_decode_bank created it, else the shared prior table
        (_prior_mu_table). Encode and the E-step self-coupling target always read the prior
        table, so the untie toggle splits ONLY the decode readout. On a tied bank
        (untie_decode_bank=False; the default is untied) this returns the identical tensor
        _prior_mu_table does -- byte-identical.
        """
        return self.decode_mu_embed if self.untie_decode_bank else self._prior_mu_table()

    @torch.no_grad()
    def decode_sigma_v_min(self) -> float:
        r"""The smallest decode prior variance in the table -- the margin behind ``a_v``'s accuracy.

        ``a_v`` accumulates terms of size ``1/sigma_v`` and is then divided by ``2*tau``, so the
        working precision has to carry roughly ``log10(1/(sigma_v * tau))`` digits before the logits
        start moving (audit 2026-08-06 F32). This is the one number that says how much headroom is
        left: at the ``sigma_init``-scale table the induced logit error is ~2e-4, at 1e-3 it is ~1.5
        logits, and at 1e-4 ~12. ``sigma_log_embed`` trains freely against only the ``eps`` floor, so
        the margin is not structurally bounded -- it has to be watched. Switch
        ``decode_av_precision="fp64"`` if this falls far enough to matter.
        """
        return float(bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps).min())

    def _decode_sigma_log_table(self) -> torch.Tensor:
        r"""The (V, K) log-variance decode table; the sigma sibling of _decode_mu_table (see there).

        Both tables are genuine log-variances, so this is a plain selection between the untied decode
        table and the shared token prior table.
        """
        if self.untie_decode_bank:
            return self.decode_sigma_log_embed
        return self._prior_sigma_log_table()

    @torch.no_grad()
    def set_unigram_log_prior(
        self,
        counts: torch.Tensor,            # (V,) corpus unigram COUNTS (integer or float, >= 0)
    ) -> None:
        r"""Fill the unigram decode table with add-one-smoothed log-frequencies (IN PLACE).

            log pi_v = log((counts_v + 1) / (sum_v counts_v + V)),
        the Laplace (add-one) smoothed unigram log-prior: every token gets one pseudo-count, so
        zero-count tokens carry a finite log pi_v = -log(total + V) instead of -inf, and
        sum_v pi_v = 1 exactly. Requires construction with decode_unigram_prior=True (the buffer
        exists only on the toggled path).
        """
        if not self.decode_unigram_prior:
            raise RuntimeError(
                "set_unigram_log_prior requires decode_unigram_prior=True at construction "
                "(the unigram_log_prior buffer exists only on the toggled path)."
            )
        if counts.shape != (self.vocab_size,):
            raise ValueError(
                f"counts must have shape ({self.vocab_size},), got {tuple(counts.shape)}"
            )
        counts_f = counts.to(dtype=self.unigram_log_prior.dtype,
                             device=self.unigram_log_prior.device)          # (V,)
        self.unigram_log_prior.copy_(
            torch.log((counts_f + 1.0) / (counts_f.sum() + float(self.vocab_size)))
        )
        self._unigram_set = True

    def _unigram_bias(self) -> torch.Tensor:
        r"""The (V,) additive decode bias kappa * log pi_v (decode_unigram_prior=True only).

        Warns ONCE PER PROCESS while the table is still all-zero (never set): the decode then
        degenerates to the pre-toggle uniform prior (kappa * 0 = 0, a value no-op). A table
        restored nonzero through load_state_dict counts as set.
        """
        global _WARNED_UNIGRAM_UNSET
        if not self._unigram_set:
            if bool((self.unigram_log_prior != 0.0).any()):
                self._unigram_set = True                                 # restored via state_dict
            elif not _WARNED_UNIGRAM_UNSET:
                _WARNED_UNIGRAM_UNSET = True
                warnings.warn(
                    "decode_unigram_prior=True but the unigram_log_prior table is unset "
                    "(all-zero): the decode degenerates to the uniform prior. Call "
                    "PriorBank.set_unigram_log_prior(counts) with the (V,) corpus counts.",
                    UserWarning, stacklevel=3,
                )
        return self.unigram_kappa * self.unigram_log_prior

    def _tau_eff(
        self,
        tau: Optional[float] = None,     # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:
        r"""Effective decode temperature tau_eff = tau * exp(-clamp(decode_log_scale, -3, 3))."""
        base_tau = self.decode_tau if tau is None else tau
        return base_tau * torch.exp(-self.decode_log_scale.clamp(-3.0, 3.0))

    def _prepare_full_covariance_for_decode(
        self,
        sigma_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Sanitize and factor full-covariance rows without recording a fallback.

        Nonfinite rows are replaced before Cholesky, diagonals, matrix products, or divergence
        kernels can observe them. The returned validity is the conjunction of original finiteness
        and Cholesky success; failed factors are replaced by identity so downstream log-determinant
        arithmetic and backward remain finite. Scoring boundaries own the exactly-once counter.
        """
        if sigma_q.dim() < 4 or sigma_q.shape[-1] != sigma_q.shape[-2]:
            raise ValueError(
                "full-covariance decode requires (..., K, K) covariance rows, got "
                f"{tuple(sigma_q.shape)}"
            )
        finite = torch.isfinite(sigma_q).all(dim=(-2, -1))
        eye = torch.eye(sigma_q.shape[-1], dtype=sigma_q.dtype, device=sigma_q.device)
        safe_sigma = torch.where(
            finite[..., None, None], sigma_q, eye.expand_as(sigma_q))
        factor, chol_ok = safe_cholesky(safe_sigma, eps=self.eps, rounds=5)
        ok = finite & chol_ok
        safe_sigma = torch.where(
            ok[..., None, None], safe_sigma, eye.expand_as(safe_sigma))
        safe_factor = torch.where(ok[..., None, None], factor, eye.expand_as(factor))
        return safe_sigma, safe_factor, ok

    def _reference_query(
        self,
        mu_q:            torch.Tensor,                           # (..., N, K) local means
        sigma_q:         Optional[torch.Tensor],                 # (..., N, K) or (..., N, K, K) local covariance
        canonical_frame: Optional[CanonicalFrameContext],        # realized U_i: (..., N, K, K) or (..., N, G, d, d)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""The query in reference coordinates for ``decode_coordinates='reference'``.

        .. math::
            \xi_i = U_i^{-1}\mu_i,\qquad C_i = U_i^{-1}\Sigma_iU_i^{-\top}.

        A full-covariance decoder receives :math:`C_i`; a diagonal Gaussian decoder receives
        :math:`\operatorname{diag}C_i`, which against a diagonal bank :math:`p_v` gives
        :math:`\mathrm{KL}(\mathcal N(\xi_i, C_i)\|p_v)` up to the per-position constant
        :math:`\tfrac12(\log\det\operatorname{diag}C_i - \log\det C_i)`, so the softmax is exact; the
        linear decoder discards the covariance and receives it unchanged. Frame actions run with
        autocast off (float32, float64 for an escalated frame) and return the query's dtype.
        """
        if canonical_frame is None:
            raise ValueError(
                "decode_coordinates='reference' requires the realized relational frame of the same "
                "forward at every decode boundary (canonical_frame); none was supplied")
        if not isinstance(canonical_frame, CanonicalFrameContext):
            raise ValueError(
                "canonical_frame must be a CanonicalFrameContext captured from the same forward "
                f"query, got {type(canonical_frame).__name__}")
        K = mu_q.shape[-1]
        inverse = canonical_frame.inverse
        # A compact frame carries one more axis than a dense one; infer it so a context rebuilt
        # without its block flag (a CPU-offloaded record) still acts blockwise.
        compact = canonical_frame.block_diagonal or inverse.dim() == mu_q.dim() + 2
        if compact:
            valid = (inverse.dim() == mu_q.dim() + 2 and tuple(inverse.shape[:-3]) == tuple(mu_q.shape[:-1])
                     and inverse.shape[-3] * inverse.shape[-1] == K)
        else:
            valid = tuple(inverse.shape) == (*mu_q.shape[:-1], K, K)
        if not valid:
            raise ValueError(
                "canonical_frame shape must match the decode query, got inverse="
                f"{tuple(inverse.shape)} for a query of shape {tuple(mu_q.shape)}")
        if canonical_frame.inverse.device != mu_q.device:
            raise ValueError(
                f"canonical_frame device {canonical_frame.inverse.device} differs from the query's {mu_q.device}")
        frame = (canonical_frame if compact == canonical_frame.block_diagonal
                 else CanonicalFrameContext(canonical_frame.forward, canonical_frame.inverse, compact))
        xi = frame_matvec(frame.inverse, mu_q, compact)                          # (..., N, K)
        if sigma_q is None or self.decode_mode == "linear":
            return xi, sigma_q
        if get_decode_registration(self.decode_mode).supports_full:
            return xi, reference_covariance(mu_q, sigma_q, frame)                # (..., N, K, K)
        return xi, reference_variances(mu_q, sigma_q, frame)                     # (..., N, K)

    def _reference_mean(
        self,
        mu_q:            torch.Tensor,                           # (..., N, K) local means
        canonical_frame: Optional[CanonicalFrameContext],
    ) -> torch.Tensor:                                           # (..., N, K) xi = U^{-1} mu
        r"""The linear decode's query :math:`\xi_i = U_i^{-1}\mu_i` (``decode_coordinates='reference'``)."""
        if self.decode_coordinates != "reference":
            raise ValueError("canonical_frame is only valid for decode_coordinates='reference'")
        return self._reference_query(mu_q, None, canonical_frame)[0]

    def _query_in_decode_frame_for_full_decode(
        self,
        mu_q:            torch.Tensor,
        sigma_q:         torch.Tensor,
        canonical_frame: Optional[CanonicalFrameContext],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sanitize before frame arithmetic, then return one prepared full-covariance query."""
        if self.decode_coordinates == "reference":
            # Sanitize the local rows, pull back to C = U^{-1} Sigma U^{-T}, THEN factor C, so the
            # log-determinant and the validity mask describe the covariance the decoder scores.
            if sigma_q.dim() >= 4 and sigma_q.shape[-1] == sigma_q.shape[-2]:
                input_ok = torch.isfinite(sigma_q).all(dim=(-2, -1))
                eye = torch.eye(sigma_q.shape[-1], dtype=sigma_q.dtype, device=sigma_q.device)
                sigma_q = torch.where(input_ok[..., None, None], sigma_q, eye.expand_as(sigma_q))
            else:
                input_ok = torch.isfinite(sigma_q).all(dim=-1)
                sigma_q = torch.where(input_ok[..., None], sigma_q, torch.ones_like(sigma_q))
            mu_q, sigma_q = self._reference_query(mu_q, sigma_q, canonical_frame)   # (xi, C)
            sigma_q, factor_q, output_ok = self._prepare_full_covariance_for_decode(sigma_q)
            ok = input_ok & output_ok
            eye = torch.eye(sigma_q.shape[-1], dtype=sigma_q.dtype, device=sigma_q.device)
            sigma_q = torch.where(ok[..., None, None], sigma_q, eye.expand_as(sigma_q))
            factor_q = torch.where(ok[..., None, None], factor_q, eye.expand_as(factor_q))
            return mu_q, sigma_q, factor_q, ok
        sigma_q, factor_q, input_ok = self._prepare_full_covariance_for_decode(sigma_q)
        mu_q, sigma_q = self._query_in_decode_frame(mu_q, sigma_q, canonical_frame)
        return mu_q, sigma_q, factor_q, input_ok

    @staticmethod
    def _full_cov_query_invariants_from_prepared(
        safe_sigma: torch.Tensor,
        safe_factor: torch.Tensor,
        ok: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read query-only KL invariants from one already-sanitized factorization."""
        diag_sq = torch.diagonal(safe_sigma, dim1=-2, dim2=-1)
        logdet_q = _logdet_chol(safe_factor)
        return diag_sq, logdet_q, ok

    def _query_in_decode_frame(
        self,
        mu_q:           torch.Tensor,
        sigma_q:        torch.Tensor,
        canonical_frame: Optional[CanonicalFrameContext],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return the query in the coordinates owned by the active decode bank.

        Local-coordinate decoders score the query as given and reject a frame context, so a stale or
        accidentally threaded frame cannot change their values. ``decode_coordinates='reference'``
        requires the realized relational frame and scores the reference moments
        (``_reference_query``).
        """
        if self.decode_coordinates == "reference":
            return self._reference_query(mu_q, sigma_q, canonical_frame)
        if canonical_frame is not None:
            raise ValueError("canonical_frame is only valid for decode_coordinates='reference'")
        return mu_q, self._diagonal_decoder_dispersion(mu_q, sigma_q)

    def decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:             Optional[float] = None,  # override decode_tau; None -> self.decode_tau
        canonical_frame: Optional[CanonicalFrameContext] = None,
    ) -> torch.Tensor:                   # (B, N, V) logits
        r"""Decode logits through the kernel selected by ``decode_mode``.

        ``linear`` computes mu_q @ W^T and ignores sigma_q/tau_eff. Every other admitted mode
        reads out against the prior bank with its registered covariance/divergence contract.

        Under ``decode_unigram_prior=True`` the unigram log-prior bias kappa * log pi_v is added
        HERE, after the registered kernel, so every decode mode (linear included) gets it from
        one seam; toggle off adds nothing (byte-identical)."""
        mu_q, sigma_q = self._query_in_decode_frame(mu_q, sigma_q, canonical_frame)
        logits = get_decode(self.decode_mode)(self, mu_q, sigma_q, self._tau_eff(tau))
        if self.decode_unigram_prior:
            logits = logits + self._unigram_bias()                       # (B, N, V) + (V,)
        return logits

    def reference_decode(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances

        *,
        tau:     Optional[float] = None,  # override decode_tau; None -> self.decode_tau
    ) -> torch.Tensor:                   # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
        r"""Authoritative reference decode: -D_configured(q_i || pi_v)/tau_eff via the seam.

        Dispatches through the CONFIGURED family (``self.family``) and divergence functional
        (``self.divergence_family`` at ``self.renyi_order``), broadcasting the seam over the
        vocabulary V in one shot (general but slow, O(B*N*V*K)). This is the same computation the
        registered ``family`` kernel performs, so it stays the oracle for the fast canonical kernels:
        for a canonical gaussian + renyi + alpha=1 config it equals the fused ``diagonal``/``full``
        kernels exactly (and under log-softmax); for a non-Gaussian family or a noncanonical
        divergence it reads the belief out under the SAME geometry the E-step minimized.

        The seam is invoked with ``kl_max=inf``: a DECODE must preserve the full divergence ranking
        over the vocabulary, so the saturation policy (default ``kl_max=100``, which flattens every
        distant prior to a single -100 logit and destroys the argmax) is disabled here. The full q is
        scored against the intentionally DIAGONAL vocabulary-prior table (promoted with diag_embed
        only for a full family). (``nan_to_num`` inside ``safe_kl_clamp`` still maps NaN/+inf from
        degenerate pairs to +inf -> -inf logits.)
        """
        tau_eff = self._tau_eff(tau)
        logits = _decode_family(self, mu_q, sigma_q, tau_eff)           # configured family/divergence
        if self.decode_unigram_prior:
            logits = logits + self._unigram_bias()                       # same seam as decode()
        return logits

    def _prepare_decode_bank_factors(self) -> Dict[str, torch.Tensor]:
        r"""Prepare fixed vocabulary factors for A_v = lhs @ rhs^T + bias."""
        sigma_v = bounded_variance_from_log(self._decode_sigma_log_table(), eps=self.eps)
        mu_v = self._decode_mu_table()
        inv_v = 1.0 / sigma_v
        c = mu_v.mean(dim=0, keepdim=True)
        mc_v = mu_v - c
        lsum = torch.log(sigma_v).sum(-1)
        work_mc_v, work_inv_v, work_lsum = mc_v, inv_v, lsum
        if self.decode_av_precision == "fp64":
            work_mc_v, work_inv_v, work_lsum = mc_v.double(), inv_v.double(), lsum.double()
        rhs = torch.cat([work_inv_v, work_mc_v * work_inv_v], dim=-1)
        bias = (work_mc_v.square() * work_inv_v).sum(-1) + work_lsum
        return dict(c=c, mc_v=mc_v, inv_v=inv_v, lsum=lsum, rhs=rhs, bias=bias)

    def _decode_bank_factors(self) -> Dict[str, torch.Tensor]:
        """Read a scoped fixed bank, or prepare fresh differentiable training factors."""
        cache = getattr(self, "_fixed_decode_bank_cache", None)
        if cache is None or torch.is_grad_enabled():
            return self._prepare_decode_bank_factors()
        if not cache:
            cache.update(self._prepare_decode_bank_factors())
        return cache

    @contextmanager
    def fixed_decode_bank(self) -> Iterator[None]:
        """Reuse vocabulary factors during a no-grad scope with fixed bank parameters.

        Generation owns this scope and performs no parameter updates. Preparation is lazy;
        the transient dictionary never enters state_dict and is released on every exit.
        Calls with gradients enabled always prepare fresh factors.
        """
        if torch.is_grad_enabled():
            yield
            return
        previous = getattr(self, "_fixed_decode_bank_cache", None)
        self._fixed_decode_bank_cache = {} if previous is None else previous
        try:
            yield
        finally:
            if previous is None:
                del self._fixed_decode_bank_cache
            else:
                self._fixed_decode_bank_cache = previous

    def _validate_fused_ce_targets(
        self,
        targets: torch.Tensor,           # (B, N) next-token ids

        *,
        ignore_index: int = -100,
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """Check exact target shape, then consume the loader's one-use ID validation."""
        if expected_shape is not None and tuple(targets.shape) != tuple(expected_shape):
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match query rows {tuple(expected_shape)}")
        validated = targets.__dict__.pop("_vfe3_host_target_validation", None)
        if (validated is not None
                and validated[:4] == (id(targets), targets._version, self.vocab_size, ignore_index)):
            return validated[4]
        counted = targets != ignore_index
        invalid = counted & ((targets < 0) | (targets >= self.vocab_size))
        has_invalid, has_ignored = torch.stack((invalid.any(), (~counted).any())).tolist()
        if has_invalid:
            invalid_target = int(targets[invalid][0].item())
            raise IndexError(f"Target {invalid_target} is out of bounds.")
        return has_ignored

    def decode_ce_diagonal_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
        return_stats:  bool            = False,
        canonical_frame: Optional[CanonicalFrameContext] = None,   # decode_coordinates='reference' only
    ) -> 'torch.Tensor | DecodeCEResult':
        r"""Fused chunked-vocab cross-entropy: the ``diagonal`` decode CE WITHOUT a (B, N, V) tensor.

        Iterates the vocabulary in chunks ``[v0, v1)``, computing each chunk's logits with the SAME
        closed form (and the SAME global centering offset ``c = mean_v(mu_v)``) as ``_decode_diagonal``,
        reducing each chunk to its per-position ``logsumexp`` and gathering the target-token logit, so
        the full ``(B, N, V)`` logit tensor is never materialized. Per position the cross-entropy is
        ``logsumexp_v(logit_v) - logit_target`` (= -log-softmax at the target); the loss is the mean
        over non-ignored positions, exactly matching ``F.cross_entropy(decode(...), targets, ignore_index)``.

        The offset ``c`` is a per-coordinate ``(1, K)`` mean over ALL V, computed in one ``O(V*K)``
        pass with no big tensor, so it is IDENTICAL to the full path (the closed form is
        offset-invariant: ``(mu_q - c) - (mu_v - c) == mu_q - mu_v``). The V-axis reduction (the
        chunk ``logsumexp`` and the target gather) happens INSIDE a gradient-checkpointed function
        that returns only the three ``(B, N)`` centered per-chunk summaries, so the ``(B, N, Vc)`` chunk logit
        is born and dies inside the checkpoint -- it is recomputed in backward and never crosses the
        boundary (without this the downstream ``logsumexp``/``exp``/``gather`` would save it and the
        peak would stay ``(B, N, V)``). Recompute is deterministic (no RNG here), so value and
        gradient match the full path exactly.

        ``decode_unigram_prior=True`` adds the chunk slice of kappa * log pi_v to each chunk's
        logits BEFORE its logsumexp/gather, so the streamed CE equals the dense CE over the
        shifted logits.

        Under ``decode_coordinates='reference'`` the query is first mapped to
        :math:`(\xi_i, \operatorname{diag}C_i)` through ``canonical_frame`` (``_reference_query``). A full-layout
        covariance (``gaussian_block``) is read as its diagonal (``_diagonal_decoder_dispersion``).
        """
        mu_q, sigma_q = self._query_in_decode_frame(mu_q, sigma_q, canonical_frame)
        has_ignored = self._validate_fused_ce_targets(
            targets, ignore_index=ignore_index, expected_shape=mu_q.shape[:-1])
        original_valid = targets != ignore_index
        if has_ignored:
            # Belief/frame production remains full-context; only readout rows are packed.
            mu_q = mu_q[original_valid].unsqueeze(0)       # (1, S, K)
            sigma_q = sigma_q[original_valid].unsqueeze(0) # (1, S, K[, K])
            targets = targets[original_valid].unsqueeze(0)

        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        c = mu_v_all.mean(dim=0, keepdim=True)                              # (1, K) global v-independent shift
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        mc_q = mu_q - c                                                     # (B, N, K) centered query means
        lhs = _decode_av_lhs(sigma_q, mc_q)                                 # (B, N, 2K) expanded-form left factor
        # Per-position, v-INDEPENDENT term of -KL/tau_eff: it cancels in the CE difference
        # (logsumexp - target_logit) but is carried so each chunk's logits equal _decode_diagonal's.
        per_pos = self.K + torch.log(sigma_q.clamp(min=self.eps)).sum(-1, keepdim=True)  # (B, N, 1)

        def _chunk_summaries(lhs_:    torch.Tensor, per_pos_:        torch.Tensor,
                             mu_v_c:  torch.Tensor, inv_v_c:         torch.Tensor,
                             lsum_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor,
                             local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor],
                             sq_:     torch.Tensor,
                             mc_q_:   torch.Tensor) -> 'tuple[Optional[torch.Tensor], ...]':
            r"""Reduce final scores to centered LSE/target/max summaries, each (B, N).

            logit_{i,v} = -0.5(a_v - per_pos)/tau_eff over the chunk (see _decode_diagonal). The
            full (B, N, Vc) chunk logit lives only here so checkpointing frees it after forward.
            ``in_chunk_f`` is a 0/1 (B, N) mask selecting positions whose target falls in this chunk.

            ``sq_``/``mc_q_`` carry the UNEXPANDED query pieces alongside ``lhs_`` so the "exact"
            a_v form can difference before squaring (audit 2026-08-06 F32); the default expanded
            form uses ``lhs_`` and ignores them.
            """
            a_v = _decode_av(
                sq_, mc_q_, mu_v_c, inv_v_c, lsum_c, lhs=lhs_, precision=self.decode_av_precision)
            logit_chunk = _decode_analytic_kl_logits(a_v, per_pos_, tau_eff, lhs_.dtype)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c.to(logit_chunk.dtype)
            return _decode_ce_chunk_summaries(logit_chunk, in_chunk_f, local_idx)

        valid = targets != ignore_index                                    # (B, N) bool
        scored_valid = original_valid
        lse_chunks = []
        max_chunks = []
        target_chunks = []
        in_chunks = []

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mc_v_c = (mu_v_all[v0:v1] - c)                                  # (Vc, K) centered prior means
            inv_v_c = 1.0 / sigma_v_all[v0:v1]                             # (Vc, K)
            lsum_c = torch.log(sigma_v_all[v0:v1]).sum(-1)                 # (Vc,)
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            # Target gather indices: positions whose target lands in [v0, v1). Ignored positions have
            # target < 0 < v0, so they never match -> target_logit stays 0 for them and `valid` excludes
            # them from the mean. local_idx is clamped to a safe range for the out-of-window rows.
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and any(
                value is not None and value.requires_grad for value in
                (lhs, per_pos, mu_v_all, sigma_v_all, tau_eff))
            activation_bytes = _decode_ce_chunk_activation_bytes(
                lhs, V if self.decode_ce_checkpoint == "auto_aggregate" else v1 - v0, inner=2,
                workspace_bytes_per_scalar=(8 if self.decode_ce_checkpoint == "auto_aggregate"
                                             and self.decode_av_precision == "fp64" else None))
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                chunk_result = _checkpoint.checkpoint(
                    _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, lsum_c,
                    in_chunk_f, local_idx,
                    u_c, sigma_q, mc_q,
                    use_reentrant=False,
                )
            else:
                chunk_result = _chunk_summaries(
                    lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx, u_c,
                    sigma_q, mc_q,
                )
            lse_chunk, contrib, maximum = chunk_result[:3]
            lse_chunks.append(lse_chunk)
            max_chunks.append(maximum)
            target_chunks.append(contrib)
            in_chunks.append(in_chunk)

        ce_per_pos = _decode_ce_combine_summaries(
            lse_chunks, max_chunks, target_chunks, in_chunks)                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        # (Matches the full path, whose F.cross_entropy mean over zero counted tokens would be NaN.)
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        return _decode_ce_result(ce, scored_valid, return_stats)

    def decode_degenerate_positions(
        self,
        sigma_q: torch.Tensor,           # (B, N, K) diagonal or (B, N, K, K) full posterior dispersion

        *,
        canonical_frame: Optional[CanonicalFrameContext] = None,
    ) -> Optional[torch.Tensor]:         # (B, N) True where the decode cannot score, else None
        r"""Positions the full-covariance decode cannot score, for a CE consumer holding its own targets.

        The fused chunked kernels own their ignore mask and fold this in directly
        (``decode_ce_full_chunked``). The DENSE branch instead hands ``(B, N, V)`` logits to
        ``F.cross_entropy`` in the model, which has no way to learn that a position is degenerate --
        so it asks here and marks those positions ``ignore_index``, keeping the two paths in parity
        on the exclusion contract (audit 2026-08-06 F31).

        ``None`` means "nothing to exclude": a diagonal dispersion is scored without a Cholesky and
        has no failure mode to report. Does NOT touch the fallback counter -- the decode kernel this
        accompanies has already counted the same event.

        Under ``decode_coordinates='reference'`` a diagonal decoder scores ``diag C`` (nothing to
        factor), and a full decoder factors ``C = U^{-1} Sigma U^{-T}`` when the frame is supplied.
        Without the frame the local rows answer the same question, since ``Sigma`` is positive
        definite iff ``C`` is.
        """
        if self.decode_coordinates == "reference":
            if not get_decode_registration(self.decode_mode).supports_full:
                return None                                   # diag C: no factorization
            if canonical_frame is None:
                if sigma_q.dim() < 4 or sigma_q.shape[-1] != sigma_q.shape[-2]:
                    return None
                safe_sigma, _, ok = self._prepare_full_covariance_for_decode(sigma_q)
            else:
                is_full = sigma_q.dim() >= 4 and sigma_q.shape[-1] == sigma_q.shape[-2]
                mu_placeholder = (
                    torch.zeros_like(sigma_q[..., 0]) if is_full else torch.zeros_like(sigma_q))
                _, safe_sigma, _, ok = self._query_in_decode_frame_for_full_decode(
                    mu_placeholder, sigma_q, canonical_frame)
        elif canonical_frame is not None:
            raise ValueError("canonical_frame is only valid for decode_coordinates='reference'")
        else:
            if not get_decode_registration(self.decode_mode).supports_full:
                return None                                   # diag Sigma: no factorization
            if sigma_q.dim() < 4 or sigma_q.shape[-1] != sigma_q.shape[-2]:
                return None                                   # diagonal dispersion: no factorization
            safe_sigma, _, ok = self._prepare_full_covariance_for_decode(sigma_q)
        return ~ok

    def _full_cov_query_invariants(
        self,
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # diag_sq, logdet_q, SPD-ok mask
        r"""Per-position, v-INDEPENDENT pieces of the full-cov KL against a DIAGONAL prior.

        Scoring a full q = N(mu_q, Sigma_q) against a diagonal prior pi_v = N(mu_v, diag(sigma_v))
        is the gaussian_full KL with a DIAGONAL second covariance, which collapses every per-pair
        (K, K) Cholesky into matmuls over V PLUS one per-position log|Sigma_q|. This returns the two
        pieces that depend on q only (not on the vocabulary):
            diag_sq = diag(Sigma_q)                    (B, N, K)  -- the raw query variances
            logdet_q = log|Sigma_q|                    (B, N)
        with the same round-zero-first factorization as the gaussian_full closed form
        (families/gaussian.py renyi_closed_form), so the diagonal-prior closed form is value-equal
        to ``_decode_full`` (the per-pair Cholesky seam) without ever forming a (B, N, V, K, K)
        workspace. ``safe_cholesky`` (jittered, never raises) yields a finite log-det where its
        ``ok`` mask is True. Nonfinite or non-PD rows are replaced by identity before these
        invariants are read; ``ok`` retains their original invalidity so the scoring boundary can
        exclude them. The SPD retraction is expected to keep ``ok`` all-True on the pure path.

        RESOLVED (audit 2026-08-06 F31): a degenerate position is EXCLUDED from the cross-entropy,
        not scored. It used to take ``logdet_q = -inf``, pinned as a cross-path parity contract by
        test_fullcov_alpha_roadmap_2026_06_13::test_full_cov_chunked_matches_dense_on_non_pd (audit
        2026-07-01 F6) -- but an all--inf logit row is not merely an -inf score, it is NaN
        downstream: ``log_softmax`` of it is NaN (on the DENSE path too), and in the chunked
        reduction ``logsumexp_v - target_logit`` is ``-inf - (-inf)``. So the pinned contract was
        itself the NaN generator, and one degenerate position NaN'd the scalar CE for the whole
        batch on BOTH paths.

        A position whose ``Sigma_q`` is not positive definite has no valid likelihood, so scoring it
        is a fiction under any sentinel; the honest reading is that it contributes nothing and its
        absence is visible in the token count. ``ok`` is therefore returned for the CE seams to fold
        into their ignore mask, and ``logdet_q`` gets a FINITE placeholder so nothing NaNs before the
        exclusion applies (the value is free: ``per_pos`` is v-INDEPENDENT and cancels exactly in
        every logit difference, so it cannot bias a surviving position). This helper does not touch
        the fallback counter: each scoring boundary combines every validity mask and records the
        final invalid set exactly once, so a run can tell how many tokens the denominator lost.
        """
        safe_sigma, safe_factor, ok = self._prepare_full_covariance_for_decode(sigma_q)
        return self._full_cov_query_invariants_from_prepared(
            safe_sigma, safe_factor, ok)

    def decode_ce_full_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K, K) posterior covariances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
        canonical_frame: Optional[CanonicalFrameContext] = None,
        return_stats:  bool            = False,
    ) -> 'torch.Tensor | DecodeCEResult':
        r"""Fused chunked-vocab cross-entropy for the FULL-covariance KL decode WITHOUT the dense
        (B, N, V) logits OR the (B, N, V, K, K) per-pair Cholesky workspace ``_decode_full`` builds.

        The prior table is DIAGONAL (sigma_log_embed), so KL(q_full || pi_v_diag) needs no per-pair
        (K, K) Cholesky: the v-dependent trace and Mahalanobis terms are matmuls over V and the only
        (K, K) work is ONE log|Sigma_q| per position (``_full_cov_query_invariants``). This is the
        full-cov twin of ``decode_ce_diagonal_chunked`` -- same streaming logsumexp + target gather
        inside a gradient checkpoint, same global centering offset c = mean_v(mu_v) for fp32
        stability -- with the diagonal query variance replaced by diag(Sigma_q), the prior kept at
        its existing variance floor, and the per-position v-independent term K + sum_k log sigma_q
        replaced by K + log|Sigma_q|. Value-equal to F.cross_entropy(_decode_full(...)) to the
        decode's atol-1e-3 (tests/test_fullcov_alpha_roadmap_2026_06_13.py). The unigram-prior
        chunk-slice add follows ``decode_ce_diagonal_chunked`` exactly.
        """
        mu_q, sigma_q, factor_q, spd_ok = self._query_in_decode_frame_for_full_decode(
            mu_q, sigma_q, canonical_frame)
        has_ignored = self._validate_fused_ce_targets(
            targets, ignore_index=ignore_index, expected_shape=mu_q.shape[:-1])
        original_valid = targets != ignore_index
        if has_ignored:
            # Belief/frame production remains full-context; only readout rows are packed.
            mu_q = mu_q[original_valid].unsqueeze(0)       # (1, S, K)
            sigma_q = sigma_q[original_valid].unsqueeze(0) # (1, S, K[, K])
            factor_q = factor_q[original_valid].unsqueeze(0)
            spd_ok = spd_ok[original_valid].unsqueeze(0)
            targets = targets[original_valid].unsqueeze(0)

        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        c = mu_v_all.mean(dim=0, keepdim=True)                              # (1, K) global v-independent shift
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        diag_sq, logdet_q, spd_ok = self._full_cov_query_invariants_from_prepared(
            sigma_q, factor_q, spd_ok)
        mc_q = mu_q - c                                                     # (B, N, K) centered query means
        lhs = _decode_av_lhs(diag_sq, mc_q)                                 # (B, N, 2K) expanded-form left factor
        # v-INDEPENDENT term of -KL/tau_eff (cancels in the CE difference, carried so each chunk's
        # logits equal _decode_full's): K + log|Sigma_q| (the full-cov analogue of K + sum_k log sigma_q).
        per_pos = self.K + logdet_q.unsqueeze(-1)                          # (B, N, 1)
        final_ok = spd_ok
        _count_decode_logdet_fallback(final_ok)

        def _chunk_summaries(lhs_:    torch.Tensor, per_pos_:        torch.Tensor,
                             mu_v_c:  torch.Tensor, inv_v_c:         torch.Tensor,
                             lsum_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor,
                             local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor],
                             sq_:     torch.Tensor,
                             mc_q_:   torch.Tensor) -> 'tuple[Optional[torch.Tensor], ...]':
            r"""Reduce final scores to centered LSE/target/max summaries, each (B, N).

            a_v = sum_k[(diag(Sigma_q) + (mc_q-mc_v)^2)/sigma_v] + sum_k log sigma_v
                = trace_term + mahalanobis + log|diag(sigma_v)|, the gaussian_full KL with a
            diagonal prior; logit = -0.5(a_v - per_pos)/tau_eff. The (B, N, Vc) chunk logit lives
            only here so checkpointing frees it after forward.

            ``sq_``/``mc_q_`` carry the UNEXPANDED query pieces alongside ``lhs_`` so the "exact"
            a_v form can difference before squaring (audit 2026-08-06 F32).
            """
            a_v = _decode_av(
                sq_, mc_q_, mu_v_c, inv_v_c, lsum_c, lhs=lhs_, precision=self.decode_av_precision)
            logit_chunk = _decode_analytic_kl_logits(a_v, per_pos_, tau_eff, lhs_.dtype)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c.to(logit_chunk.dtype)
            return _decode_ce_chunk_summaries(logit_chunk, in_chunk_f, local_idx)

        # A position whose Sigma_q is not PD is EXCLUDED, exactly like an ignore_index token, rather
        # than scored (audit 2026-08-06 F31); it leaves both the numerator and the denominator, and
        # _count_decode_logdet_fallback has already recorded it. The dense path drops the same
        # positions via PriorBank.decode_degenerate_positions, so the two stay in parity.
        valid = (targets != ignore_index) & final_ok                       # (B, N) bool
        scored_valid = original_valid.clone()
        if has_ignored:
            scored_valid[original_valid] = valid.flatten()
        else:
            scored_valid = valid
        lse_chunks = []
        max_chunks = []
        target_chunks = []
        in_chunks = []

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mc_v_c = (mu_v_all[v0:v1] - c)                                  # (Vc, K) centered prior means
            inv_v_c = 1.0 / sigma_v_all[v0:v1]                             # (Vc, K) = 1/sigma_v
            lsum_c = torch.log(sigma_v_all[v0:v1]).sum(-1)                 # (Vc,) = sum_k log sigma_v
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and any(
                value is not None and value.requires_grad for value in
                (lhs, per_pos, mu_v_all, sigma_v_all, tau_eff))
            activation_bytes = _decode_ce_chunk_activation_bytes(
                lhs, V if self.decode_ce_checkpoint == "auto_aggregate" else v1 - v0, inner=2,
                workspace_bytes_per_scalar=(8 if self.decode_ce_checkpoint == "auto_aggregate"
                                             and self.decode_av_precision == "fp64" else None))
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                chunk_result = _checkpoint.checkpoint(
                    _chunk_summaries, lhs, per_pos, mc_v_c, inv_v_c, lsum_c,
                    in_chunk_f, local_idx, u_c, diag_sq, mc_q,
                    use_reentrant=False,
                )
            else:
                chunk_result = _chunk_summaries(
                    lhs, per_pos, mc_v_c, inv_v_c, lsum_c, in_chunk_f, local_idx, u_c,
                    diag_sq, mc_q,
                )
            lse_chunk, contrib, maximum = chunk_result[:3]
            lse_chunks.append(lse_chunk)
            max_chunks.append(maximum)
            target_chunks.append(contrib)
            in_chunks.append(in_chunk)

        ce_per_pos = _decode_ce_combine_summaries(
            lse_chunks, max_chunks, target_chunks, in_chunks)                            # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        return _decode_ce_result(ce, scored_valid, return_stats)

    def decode_ce_linear_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        chunk_size:    Optional[int]         = None,  # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int                   = -100,
        return_stats:  bool                  = False,
        canonical_frame: Optional[CanonicalFrameContext] = None,   # decode_coordinates='reference' only
        sigma_q:       Optional[torch.Tensor] = None,  # (B, N, K) diagonal variances: second-order term
    ) -> 'torch.Tensor | DecodeCEResult':
        r"""Fused chunked-vocab cross-entropy for ``decode_mode='linear'``.

        The ``_decode_linear`` CE -- ``logits = x @ W^T (+ b)`` -> ``F.cross_entropy`` -- WITHOUT
        the (B, N, V) logit tensor (plus cross_entropy's same-size log-softmax copy, both retained
        for backward on the dense path; the dominant decode VRAM at large B, vram audit 2026-06-10).
        Same streaming contract as ``decode_ce_diagonal_chunked``: each vocab chunk's logits are
        born and die inside a gradient-checkpointed reduction that returns only the (B, N) chunk
        logsumexp and target-logit summaries; recompute is deterministic, so value and gradient (to
        mu_q, W, and b) match the dense path exactly. The unigram-prior chunk-slice add follows
        ``decode_ce_diagonal_chunked``. Ignored positions are gathered out before projection;
        the original mask still supplies scored and excluded token counts. Under
        ``decode_coordinates='reference'`` the logits read :math:`\xi_i = U_i^{-1}\mu_i` through
        ``canonical_frame``, pulled back before any position is gathered out.

        With ``sigma_q`` (``decode_second_order``; local coordinates, diagonal covariance) each
        position's loss is the second-order expected log-loss under
        :math:`q = \mathcal{N}(\mu, \Sigma)`, :math:`\Sigma = \mathrm{diag}(\sigma)`. Write
        :math:`\ell(x) = \mathrm{LSE}(Wx + c) - w_y^\top x - c_y` with :math:`c` the bias plus unigram
        prior. The target term is linear in :math:`x`, so the Hessian is that of the log-sum-exp,
        :math:`\nabla^2 \ell = W^\top(\mathrm{diag}\, p - p p^\top) W` at
        :math:`p = \mathrm{softmax}(W\mu + c)`. Taylor expansion about :math:`\mu` with
        :math:`\varepsilon = x - \mu`, :math:`\mathbb{E}\varepsilon = 0` and vanishing odd Gaussian
        moments gives
        :math:`\mathbb{E}_q \ell = \ell(\mu) + \tfrac12 \mathrm{tr}(\Sigma\, \mathrm{Cov}_{v\sim p}(w_v))
        + O(\|\Sigma\|^2)`, which for diagonal :math:`\Sigma` is

        .. math::
            \mathrm{CE}(\mu) + \tfrac12 \Big[\sum_v p_v \big((W \circ W)\sigma\big)_v
            - \sum_k \sigma_k (W^\top p)_k^2\Big].

        Each chunk returns, besides its summaries, the chunk-normalized moments
        :math:`\sum_{v \in c} e^{\ell_v - \mathrm{lse}_c} [w_v, w_v \circ w_v]` from one
        :math:`(B, N, V_c) \times (V_c, 2K)` GEMM in the caller's precision (outside the
        ``decode_logits_amp`` autocast); the chunks are recombined with weights
        :math:`e^{\mathrm{lse}_c - \mathrm{LSE}}`. No (B, N, V, K) tensor is formed, and gradients reach
        :math:`\mu`, :math:`\sigma`, :math:`W` and the bias. The full-covariance analogue
        :math:`\tfrac12\mathrm{tr}(\Sigma W^\top \mathrm{diag}(p) W) - \tfrac12 (W^\top p)^\top \Sigma (W^\top p)`
        needs a per-position :math:`(K, K)` moment (K times the logits GEMM) and is not implemented.
        """
        if sigma_q is not None and (canonical_frame is not None or self.decode_coordinates == "reference"):
            raise ValueError("the second-order linear decode term needs local coordinates: the "
                             "reference covariance U^-1 Sigma U^-T is not diagonal")
        if canonical_frame is not None or self.decode_coordinates == "reference":
            mu_q = self._reference_mean(mu_q, canonical_frame)             # (B, N, K) xi = U^{-1} mu
        has_ignored = self._validate_fused_ce_targets(
            targets, ignore_index=ignore_index, expected_shape=mu_q.shape[:-1])
        original_valid = targets != ignore_index
        if has_ignored:
            # Retain the (batch, positions, K) checkpoint contract with only S scored rows.
            mu_q = mu_q[original_valid].unsqueeze(0)       # (1, S, K)
            targets = targets[original_valid].unsqueeze(0) # (1, S)
            if sigma_q is not None:
                sigma_q = sigma_q[original_valid].unsqueeze(0)  # (1, S, K)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size
        W = self.output_proj_weight                                        # (V, K)
        bias = self.output_proj_bias                                       # (V,) or None
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        def _chunk_logits(mu_: torch.Tensor, w_c: torch.Tensor,
                          b_c: Optional[torch.Tensor], u_c: Optional[torch.Tensor]) -> torch.Tensor:
            r"""One vocab chunk's logits :math:`x W_c^\top + b_c + u_c`, (B, N, Vc)."""
            logit_chunk = _linear_logits(self, mu_, w_c)                   # (B, N, Vc)
            if b_c is not None:
                if torch.promote_types(logit_chunk.dtype, b_c.dtype) == logit_chunk.dtype:
                    logit_chunk.add_(b_c)  # fresh logits; the bias slice cannot expand shape
                else:
                    logit_chunk = logit_chunk + b_c
            if u_c is not None:
                if torch.promote_types(logit_chunk.dtype, u_c.dtype) == logit_chunk.dtype:
                    logit_chunk.add_(u_c)  # fresh logits; the bias slice cannot expand shape
                else:
                    logit_chunk = logit_chunk + u_c
            return logit_chunk

        def _chunk_reduce(logit_chunk: torch.Tensor, in_chunk_f: torch.Tensor,
                          local_idx: torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""(lse_chunk, target_contrib), both (B, N), from one chunk's logits."""
            lse_chunk = torch.logsumexp(logit_chunk, dim=-1)               # (B, N)
            gathered = logit_chunk.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)  # (B, N)
            # SELECT rather than multiply (audit 2026-08-06 F31): `gathered * in_chunk_f` is
            # `-inf * 0.0` = NaN in every chunk that does NOT contain the target, so a degenerate
            # position poisoned target_logit through chunks it has no business contributing to.
            # Byte-identical for finite logits (x*1.0 == x, x*0.0 == 0.0).
            return lse_chunk, torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))

        def _chunk_summaries(mu_:     torch.Tensor, w_c:       torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             b_c:     Optional[torch.Tensor],
                             u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor]':
            r"""Reduce one vocab chunk to (lse_chunk, target_contrib), both (B, N), on the inside."""
            return _chunk_reduce(_chunk_logits(mu_, w_c, b_c, u_c), in_chunk_f, local_idx)

        def _chunk_summaries_second_order(
                mu_:     torch.Tensor, w_c:       torch.Tensor,
                in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                b_c:     Optional[torch.Tensor],
                u_c:     Optional[torch.Tensor]) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
            r"""As ``_chunk_summaries`` plus the chunk-normalized moments
            :math:`\sum_{v \in c} e^{\ell_v - \mathrm{lse}_c} [w_v, w_v \circ w_v]`, (B, N, 2K)."""
            logit_chunk = _chunk_logits(mu_, w_c, b_c, u_c)                # (B, N, Vc)
            lse_chunk, contrib = _chunk_reduce(logit_chunk, in_chunk_f, local_idx)
            p_c = torch.exp(logit_chunk - lse_chunk.unsqueeze(-1))         # (B, N, Vc) chunk-normalized
            w_moments = torch.cat((w_c, w_c * w_c), dim=-1).to(p_c.dtype)  # (Vc, 2K) [w_v, w_v o w_v]
            return lse_chunk, contrib, p_c @ w_moments                     # (B, N, 2K)

        valid = targets != ignore_index                                    # (B, N) bool
        second_order = sigma_q is not None
        lse_chunks = []
        moment_chunks = []                                                 # (B, N, 2K) per chunk (second order)
        target_logit = torch.zeros(mu_q.shape[:-1], device=mu_q.device, dtype=mu_q.dtype)  # (B, N)

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            w_c = W[v0:v1]                                                 # (Vc, K)
            b_c = bias[v0:v1] if bias is not None else None                # (Vc,) or None
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and (mu_q.requires_grad or W.requires_grad)
            activation_bytes = _decode_ce_chunk_activation_bytes(
                mu_q, V if self.decode_ce_checkpoint == "auto_aggregate" else v1 - v0)
            summaries = _chunk_summaries_second_order if second_order else _chunk_summaries
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                chunk_result = _checkpoint.checkpoint(
                    summaries, mu_q, w_c, in_chunk_f, local_idx, b_c, u_c,
                    use_reentrant=False,
                )
            else:
                chunk_result = summaries(mu_q, w_c, in_chunk_f, local_idx, b_c, u_c)
            lse_chunk, contrib = chunk_result[:2]
            if second_order:
                moment_chunks.append(chunk_result[2])
            lse_chunks.append(lse_chunk)
            target_logit = target_logit + contrib                          # exactly one chunk contributes per valid pos

        lse_stack = torch.stack(lse_chunks, dim=0)                         # (C, B, N)
        logsumexp_v = torch.logsumexp(lse_stack, dim=0)                    # (B, N)
        ce_per_pos = logsumexp_v - target_logit                           # (B, N) = -log-softmax at target
        if second_order:
            # Global moments E_p[w], E_p[w o w]: chunk weights exp(lse_c - LSE) sum to 1 per position.
            chunk_weight = torch.exp(lse_stack - logsumexp_v)              # (C, B, N)
            moments = (chunk_weight.unsqueeze(-1) * torch.stack(moment_chunks, dim=0)).sum(0)  # (B, N, 2K)
            K = mu_q.shape[-1]
            mean_w, mean_w_sq = moments[..., :K], moments[..., K:]         # (B, N, K) each
            # 1/2 sum_k sigma_k Var_p(w_k) = 1/2 tr(Sigma Cov_p(w)), the delta-method term.
            ce_per_pos = ce_per_pos + 0.5 * (sigma_q * (mean_w_sq - mean_w * mean_w)).sum(-1)  # (B, N)
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        return _decode_ce_result(ce, original_valid, return_stats)

    def decode_ce_expected_likelihood_chunked(
        self,
        mu_q:    torch.Tensor,           # (B, N, K) posterior means
        sigma_q: torch.Tensor,           # (B, N, K) posterior variances
        targets: torch.Tensor,           # (B, N) next-token ids (-100 = ignore)

        *,
        tau:           Optional[float] = None,   # override decode_tau; None -> self.decode_tau
        chunk_size:    Optional[int]   = None,   # vocab-chunk width; None -> self.decode_chunk_size
        ignore_index:  int             = -100,
        return_stats:  bool            = False,
    ) -> 'torch.Tensor | DecodeCEResult':
        r"""Fused chunked-vocab cross-entropy for the EXPECTED-LIKELIHOOD decode (diagonal only).

        The fused-CE twin of ``decode_mode='expected_likelihood_chunked'`` (see
        ``_decode_expected_likelihood_chunked`` for the scoring math): the same streaming contract
        as ``decode_ce_diagonal_chunked`` -- each chunk's (B, N, Vc) logits are born and die inside
        a gradient-checkpointed reduction returning only the (B, N) chunk logsumexp and target
        summaries, so the (B, N, V) tensor is never materialized. The couplings sigma_q + sigma_v
        block the diagonal kernel's single-matmul trick, so each chunk broadcasts a (B, N, Vc, K)
        workspace instead (bounded by the chunk width; freed by the checkpoint). The unigram-prior
        chunk-slice add follows ``decode_ce_diagonal_chunked``.
        """
        has_ignored = self._validate_fused_ce_targets(
            targets, ignore_index=ignore_index, expected_shape=mu_q.shape[:-1])
        original_valid = targets != ignore_index
        if has_ignored:
            # Belief/frame production remains full-context; only readout rows are packed.
            mu_q = mu_q[original_valid].unsqueeze(0)       # (1, S, K)
            sigma_q = sigma_q[original_valid].unsqueeze(0) # (1, S, K[, K])
            targets = targets[original_valid].unsqueeze(0)

        tau_eff = self._tau_eff(tau)
        chunk = self.decode_chunk_size if chunk_size is None else chunk_size
        V = self.vocab_size

        sigma_v_all = bounded_variance_from_log(
            self._decode_sigma_log_table(), eps=self.eps,
        )                                                                             # (V, K)
        mu_v_all = self._decode_mu_table()                                  # (V, K) decode table (untied if set)
        u_all = self._unigram_bias() if self.decode_unigram_prior else None  # (V,) kappa*log pi_v or None

        # CUDA tensors always take the fused Triton kernel (the expected_likelihood_kernel selector
        # was removed 2026-10-12): it streams (i, v) register tiles and never forms (B, N, Vc, K);
        # same formula, rounding-level differences listed in vfe3/kernels/expected_likelihood.py.
        # The Triton kernel is CUDA-only, so CPU tensors run the chunked torch kernel below.
        fused = mu_q.is_cuda
        self.realized_expected_likelihood_kernel = "triton" if fused else "torch"
        if fused:
            from vfe3.kernels.expected_likelihood import expected_likelihood_ce
            ce = expected_likelihood_ce(
                mu_q.reshape(-1, mu_q.shape[-1]), sigma_q.reshape(-1, sigma_q.shape[-1]),
                mu_v_all, sigma_v_all, u_all, tau_eff, targets.reshape(-1),
            )                                                               # () mean over packed rows
            return _decode_ce_result(ce, original_valid, return_stats)

        def _chunk_summaries(mu_q_:   torch.Tensor, sigma_q_:   torch.Tensor,
                             mu_v_c:  torch.Tensor, sigma_v_c:  torch.Tensor,
                             in_chunk_f: torch.Tensor, local_idx: torch.Tensor,
                             u_c:     Optional[torch.Tensor]) -> 'tuple[Optional[torch.Tensor], ...]':
            r"""Reduce final scores to centered LSE/target/max summaries, each (B, N)."""
            d = mu_q_.unsqueeze(-2) - mu_v_c                               # (B, N, Vc, K)
            s = sigma_q_.unsqueeze(-2) + sigma_v_c                         # (B, N, Vc, K) convolved variances
            logit_chunk = -0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff   # (B, N, Vc)
            if u_c is not None:
                logit_chunk = logit_chunk + u_c                            # unigram log-prior chunk slice
            return _decode_ce_chunk_summaries(logit_chunk, in_chunk_f, local_idx)

        valid = targets != ignore_index                                    # (B, N) bool
        scored_valid = original_valid
        lse_chunks = []
        max_chunks = []
        target_chunks = []
        in_chunks = []

        for v0 in range(0, V, chunk):
            v1 = min(v0 + chunk, V)
            mu_v_c = mu_v_all[v0:v1]                                       # (Vc, K)
            sigma_v_c = sigma_v_all[v0:v1]                                 # (Vc, K)
            u_c = u_all[v0:v1] if u_all is not None else None              # (Vc,) or None
            in_chunk = (targets >= v0) & (targets < v1)                    # (B, N) bool
            in_chunk_f = in_chunk.to(mu_q.dtype)                           # (B, N) 0/1, carried into the checkpoint
            local_idx = (targets - v0).clamp(min=0, max=v1 - v0 - 1)       # (B, N) safe gather index
            grad_active = torch.is_grad_enabled() and any(
            value.requires_grad for value in (mu_q, sigma_q, mu_v_all, sigma_v_all, tau_eff))
            # This closure's largest workspace is (B, N, Vc, K) (the `d`/`s` broadcast the diagonal
            # kernel's single-matmul trick cannot use, per the docstring above), so the byte estimate
            # carries the extra K factor a bare (B, N, Vc) count would miss.
            aggregate = self.decode_ce_checkpoint == "auto_aggregate"
            activation_bytes = _decode_ce_chunk_activation_bytes(
                mu_q, V if aggregate else v1 - v0, inner=self.K * (2 if aggregate else 1))
            if _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes):
                lse_chunk, contrib, maximum = _checkpoint.checkpoint(
                    _chunk_summaries, mu_q, sigma_q, mu_v_c, sigma_v_c, in_chunk_f, local_idx,
                    u_c, use_reentrant=False,
                )
            else:
                lse_chunk, contrib, maximum = _chunk_summaries(
                    mu_q, sigma_q, mu_v_c, sigma_v_c, in_chunk_f, local_idx, u_c
                )
            lse_chunks.append(lse_chunk)
            max_chunks.append(maximum)
            target_chunks.append(contrib)
            in_chunks.append(in_chunk)

        ce_per_pos = _decode_ce_combine_summaries(
            lse_chunks, max_chunks, target_chunks, in_chunks)                           # (B, N) = -log-softmax at target
        # Device-side masked mean: clamp the denominator so an all-ignore microbatch yields a finite
        # grad-connected 0 (the numerator is then 0) without a host sync to branch on valid.sum() == 0.
        ce = (ce_per_pos * valid).sum() / valid.sum().clamp_min(1)
        return _decode_ce_result(ce, scored_valid, return_stats)

EncodeCallable = Callable[[PriorBank, torch.Tensor], BeliefState]
DecodeCallable = Callable[
    [PriorBank, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


class GeometricFusedCECallable(Protocol):
    """Fused CE contract for covariance-aware geometric decoders."""

    def __call__(
        self,
        pb:            PriorBank,
        mu_q:          torch.Tensor,
        sigma_q:       torch.Tensor,
        targets:       torch.Tensor,

        *,
        tau:           Optional[float] = None,
        chunk_size:    Optional[int]   = None,
        ignore_index:  int             = -100,
        return_stats:  bool            = False,
    ) -> 'torch.Tensor | DecodeCEResult':
        ...


class FrameAwareGeometricFusedCECallable(Protocol):
    """Geometric fused CE contract for the projected full-covariance decode boundary."""

    def __call__(
        self,
        pb:              PriorBank,
        mu_q:            torch.Tensor,
        sigma_q:         torch.Tensor,
        targets:         torch.Tensor,

        *,
        tau:             Optional[float]                 = None,
        chunk_size:      Optional[int]                   = None,
        ignore_index:    int                             = -100,
        canonical_frame: Optional[CanonicalFrameContext] = None,
        return_stats:    bool                            = False,
    ) -> 'torch.Tensor | DecodeCEResult':
        ...


class LinearFusedCECallable(Protocol):
    """Fused CE contract for the mean-only linear decoder."""

    def __call__(
        self,
        pb:            PriorBank,
        mu_q:          torch.Tensor,
        targets:       torch.Tensor,

        *,
        chunk_size:    Optional[int] = None,
        ignore_index:  int           = -100,
        return_stats:  bool          = False,
    ) -> 'torch.Tensor | DecodeCEResult':
        ...


FusedCECallable = (
    GeometricFusedCECallable
    | FrameAwareGeometricFusedCECallable
    | LinearFusedCECallable
)


def _encode_prior_sigma(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> torch.Tensor:
    """Look up the token-table belief prior covariance at the configured family rank."""
    if pb.prior_covariance_table == "block_cholesky":
        blocks = pb._prior_covariance_blocks(token_ids)                      # (B, N, H, d, d)
        eye_h = torch.eye(blocks.shape[-3], dtype=blocks.dtype, device=blocks.device)
        return torch.einsum("...hab,hg->...hagb", blocks, eye_h).reshape(
            *token_ids.shape, pb.K, pb.K)                                    # (B, N, K, K) head-block diagonal
    variance = bounded_variance_from_log(pb._prior_sigma_log_table(), eps=pb.eps,
                                         index=token_ids)                 # (B, N, K)
    if pb.diagonal_covariance:
        return variance
    return torch.diag_embed(variance)


@register_encode("per_token", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_per_token(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Per-token table lookup: token_ids -> (mu_v, sigma_v, phi_v) as the belief q = p.

    Diagonal family: sigma is the (B, N, K) variance vector. A full token-table prior promotes its
    diagonal variances to (B, N, K, K). The mean and gauge tables are shared across families.
    """
    mu = pb._prior_mu_table()[token_ids]                                     # (B, N, K) token prior
    sigma = _encode_prior_sigma(pb, token_ids)                               # (B,N,K) or (B,N,K,K)
    phi = pb.phi_embed[token_ids]                                            # (B, N, n_gen)
    return BeliefState(mu=mu, sigma=sigma, phi=phi)


@register_encode("per_token_additive", can_omit_base_mean=True, can_omit_base_variance=True)
def _encode_per_token_additive(
    pb:        PriorBank,
    token_ids: torch.Tensor,             # (B, N) integer token ids
) -> BeliefState:
    r"""Arm-2 control: the SAME learned (V, n_gen) phi table used NON-structurally.

    Each token's phi code is mapped by the FROZEN readout ``pb.additive_R`` (K, n_gen) to an additive
    mean shift ``mu += phi @ R^T``, and the returned phi is ZERO so the transport
    ``Omega = exp(phi.G) exp(-phi.G) = I`` (no gl(g) congruence). The learned parameter count is the
    gauge cell's (``V*n_gen`` in ``phi_embed``; ``R`` is a frozen buffer), so this isolates raw phi-table
    CAPACITY from the gl(g) generator STRUCTURE -- the capacity-vs-structure control for the blocks_K48
    REMAND (docs/2026-07-05-blocks-k48-followup-experiment-spec.md, Arm 2a). Deliberately NOT gauge
    equivariant; use with ``transport_mode='flat'`` and ``pos_phi='none'`` so no other channel transports.
    """
    mu = pb._prior_mu_table()[token_ids]                                     # (B, N, K) token prior
    sigma = _encode_prior_sigma(pb, token_ids)                               # (B,N,K) or (B,N,K,K)
    phi_code = pb.phi_embed[token_ids]                                       # (B, N, n_gen) learned table
    mu = mu + phi_code @ pb.additive_R.t()                                   # (B, N, K) structure-free shift
    phi = torch.zeros_like(phi_code)                                         # Omega = I: no gl(g) transport
    return BeliefState(mu=mu, sigma=sigma, phi=phi)


def _decode_diagonal(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact diagonal -KL/tau_eff in closed form via a single fused matmul.

        KL = 0.5[ sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) - K + sum_k log(sigma_v/sigma_q) ]
    The v-dependent part A_v expands the Mahalanobis/trace terms into one matmul:
        lhs = [sigma_q + mc_q^2, -2 mc_q]            (B, N, 2K)
        rhs = [1/sigma_v,        mc_v/sigma_v]       (V, 2K)
        A_v = lhs @ rhs^T + sum_k(mc_v^2/sigma_v + log sigma_v)
            == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
            == 2 KL + K + sum_k log sigma_q.
    The per-position (-K - sum_k log sigma_q) is v-INDEPENDENT (drops under softmax) but
    is KEPT so logits == -KL/tau_eff EXACTLY.

    NUMERICS: the Mahalanobis term ``(mu_q - mu_v)^2`` is reconstructed by the matmul as
    ``mc_q^2 - 2 mc_q mc_v + mc_v^2``, a subtraction of large near-equal quantities that
    catastrophically cancels in float32 once the means carry a large common offset (the
    error grows like eps * mu^2 / sigma_v and breaks the atol-1e-3 seam pin at modest
    |mu| / tight sigma_v). We remove the common offset BEFORE the matmul by subtracting
    the v-independent shift ``c = mean_v(mu_v)`` (per dim) from both means; since
    ``(mu_q - c) - (mu_v - c) == mu_q - mu_v`` the closed form is unchanged exactly while
    the canceled magnitude collapses to the residual spread of the means.
    """
    factors = pb._decode_bank_factors()
    c, mc_v, inv_v = factors["c"], factors["mc_v"], factors["inv_v"]
    mc_q = mu_q - c
    ranking_guard = pb.decode_ranking_fp64_escalation
    a_result = _decode_av(
        sigma_q, mc_q, mc_v, inv_v, factors["lsum"],
        return_error_bound=ranking_guard, precision=pb.decode_av_precision,
        bank_factors=(factors["rhs"], factors["bias"]))
    if ranking_guard:
        a_v, a_bound = a_result
    else:
        a_v, a_bound = a_result, None
    # == sum_k[(sigma_q + mc_q^2 - 2 mc_q mc_v)/sigma_v] + sum_k(mc_v^2/sigma_v + log sigma_v)
    # under the default expanded form; see _decode_av for the "exact" alternative (F32).
    # a_v == sum_k(sigma_q/sigma_v + (mc_q-mc_v)^2/sigma_v) + sum_k log sigma_v
    #     == sum_k(sigma_q/sigma_v + (mu_q-mu_v)^2/sigma_v) + sum_k log sigma_v = 2 KL + K + sum_k log sigma_q
    per_pos = pb.K + torch.log(sigma_q.clamp(min=pb.eps)).sum(-1, keepdim=True)   # (B, N, 1) = K + sum_k log sigma_q
    # .to() after the clamp keeps the fp64 a_v island open through the subtraction (F32); it is an
    # identity no-op under the default fp32 policy.
    if ranking_guard:
        unigram = pb._unigram_bias() if pb.decode_unigram_prior else None
        ranking_logits, ranking_bound = _decode_analytic_kl_logits(
            a_v, per_pos, tau_eff, mu_q.dtype, a_error_bound=a_bound,
            unigram_bias=unigram, return_error_bound=True)
        if _final_ranking_uncertain(ranking_logits, ranking_bound):
            return _refined_candidate_decode(
                pb, mu_q, sigma_q, tau_eff, ranking_logits, ranking_bound,
                _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype),
                full_covariance=False)
    return _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype)


@register_decode(
    "diagonal_chunked",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_diagonal_chunked,
    fused_ce_supports_stats=True,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_diagonal_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='diagonal_chunked'``: full diagonal logits.

    The chunked mode's training memory win is the FUSED decode+CE in ``decode_ce_diagonal_chunked``
    (it never forms ``(B, N, V)``). When ``decode`` is called for logits (sampling / generation /
    inference), correctness is what matters, so this delegates to the exact dense
    ``_decode_diagonal`` closed form.
    """
    return _decode_diagonal(pb, mu_q, sigma_q, tau_eff)


def _decode_full(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Exact full-covariance decode logits_{i,v} = -KL(q_i || pi_v)/tau_eff via Cholesky.

    Scores the full-covariance posterior q_i = N(mu_q, Sigma_q) against every vocab prior
    pi_v through the ``gaussian_full`` divergence seam (Cholesky KL). The prior table is
    diagonal (sigma_log_embed), embedded as a diagonal full covariance diag(exp(sigma_log_v))
    so a full q is scored against it. As in ``reference_decode`` the seam is invoked with
    ``kl_max=inf`` so the full KL ranking over the vocabulary is preserved (decode must not
    saturate distant priors to a single logit). General but O(B*N*V*K^3) (per-pair Cholesky):
    the theoretically pure full-covariance path, not the fast diagonal kernel.
    """
    sigma_q, factor_q, spd_ok = pb._prepare_full_covariance_for_decode(sigma_q)
    mu_v = pb._decode_mu_table()                                         # (V, K) decode table (untied if set)
    sigma_v = torch.diag_embed(
        bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)
    )                                                                                       # (V, K, K) diagonal-as-full
    mu_q_b = mu_q.unsqueeze(-2)                                          # (B, N, 1, K)
    sigma_q_b = sigma_q.unsqueeze(-3)                                    # (B, N, 1, K, K)
    full = get_family("gaussian_full")
    kl_v = kl(
        full(mu_q_b, sigma_q_b),
        full(mu_v, sigma_v),
        kl_max=float("inf"),
        eps=pb.eps,
    )                                                                       # (B, N, V)
    diag_sq, _, spd_ok = pb._full_cov_query_invariants_from_prepared(
        sigma_q, factor_q, spd_ok)
    final_ok = spd_ok
    _count_decode_logdet_fallback(final_ok)
    # Same exclusion contract as the chunked twin (audit 2026-08-06 F31). Here the -inf arrived
    # indirectly -- the family seam maps a failed Cholesky to NaN and ``kl_max=inf`` maps that to
    # inf -- so the row has to be neutralized after the fact. One extra (B, N) Cholesky against this
    # kernel's own O(B*N*V*K^3) per-pair factorization is not measurable.
    kl_v = torch.where(final_ok.unsqueeze(-1), kl_v, torch.zeros_like(kl_v))
    return (-kl_v / tau_eff).to(mu_q.dtype)


@register_decode(
    "full_chunked",
    supports_full=True,
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_full_chunked,
    fused_ce_supports_stats=True,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_full_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K, K) posterior covariances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -KL(q || pi_v)/tau_eff
    r"""Inference (targets=None) decode for ``decode_mode='full_chunked'``: full-cov KL logits via
    the DIAGONAL-prior closed form -- NO per-pair (K, K) Cholesky.

    The training memory win is the fused decode+CE in ``decode_ce_full_chunked`` (never forms
    (B, N, V)); for logits (sampling / generation) this materializes (B, N, V) -- inherent to
    producing every vocab logit -- but still avoids the (B, N, V, K, K) Cholesky/solve workspace
    that ``_decode_full`` builds, by exploiting the diagonal prior (see ``_full_cov_query_invariants``).
    Value-equal to ``decode_mode='full'`` to atol-1e-3 (tests/test_fullcov_alpha_roadmap_2026_06_13.py).
    """
    sigma_q, factor_q, spd_ok = pb._prepare_full_covariance_for_decode(sigma_q)
    factors = pb._decode_bank_factors()
    c, mc_v, inv_v = factors["c"], factors["mc_v"], factors["inv_v"]
    diag_sq, logdet_q, spd_ok = pb._full_cov_query_invariants_from_prepared(
        sigma_q, factor_q, spd_ok)
    mc_q = mu_q - c
    ranking_guard = pb.decode_ranking_fp64_escalation
    a_result = _decode_av(
        diag_sq, mc_q, mc_v, inv_v, factors["lsum"],
        return_error_bound=ranking_guard, precision=pb.decode_av_precision,
        bank_factors=(factors["rhs"], factors["bias"]))
    if ranking_guard:
        a_v, a_bound = a_result
    else:
        a_v, a_bound = a_result, None
    per_pos = pb.K + logdet_q.unsqueeze(-1)                              # (B, N, 1) = K + log|Sigma_q|
    # .to() after the clamp keeps the fp64 a_v island open through the subtraction (F32); it is an
    # identity no-op under the default fp32 policy.
    final_ok = spd_ok
    _count_decode_logdet_fallback(final_ok)
    if ranking_guard:
        unigram = pb._unigram_bias() if pb.decode_unigram_prior else None
        ranking_logits, ranking_bound = _decode_analytic_kl_logits(
            a_v, per_pos, tau_eff, mu_q.dtype, a_error_bound=a_bound,
            unigram_bias=unigram, return_error_bound=True)
        if _final_ranking_uncertain(ranking_logits, ranking_bound):
            logits = _refined_candidate_decode(
                pb, mu_q, sigma_q, tau_eff, ranking_logits, ranking_bound,
                _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype),
                full_covariance=True)
        else:
            logits = _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype)
    else:
        logits = _decode_analytic_kl_logits(a_v, per_pos, tau_eff, mu_q.dtype)
    # A non-PD Sigma_q has no valid likelihood: emit an INFORMATIONLESS uniform row rather than a
    # score built on a placeholder log-det (audit 2026-08-06 F31). log_softmax of a uniform row is
    # -log V, finite; the all--inf row this used to produce maps to NaN. The CE seams exclude the
    # position outright -- see decode_ce_full_chunked and decode_degenerate_positions.
    return torch.where(final_ok.unsqueeze(-1), logits, torch.zeros_like(logits))


def _family_logits(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Body of the dense family-consistent LOGITS oracle ``_decode_family`` (audit 2026-08-07).

    Degenerate positions are handled exactly as ``_decode_full_chunked`` handles them (audit
    2026-08-06 F31): a non-PD Sigma_q has no valid likelihood, so it is scored on a substituted
    benign covariance and then overwritten with an INFORMATIONLESS uniform row, rather than emitting
    the NaN that ``kl_max=inf`` would otherwise produce and that would propagate through any
    downstream ``log_softmax`` or sampling step.
    """
    family_cls = get_family(pb.family)
    is_full = family_cls.cov_kind == "full"
    functional = get_functional(pb.divergence_family)

    degenerate = None
    if is_full:
        sigma_q, _, full_ok = pb._prepare_full_covariance_for_decode(sigma_q)
        degenerate = ~full_ok
        _count_decode_logdet_fallback(full_ok)

    q = family_cls(
        mu_q.unsqueeze(-2),
        sigma_q.unsqueeze(-3 if is_full else -2),
    )
    p_sigma_all = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps
    )                                                                       # (V, K) diagonal prior variances
    mu_v_all = pb._decode_mu_table()                                        # (V, K) decode table (untied if set)
    p = family_cls(
        mu_v_all,
        torch.diag_embed(p_sigma_all) if is_full else p_sigma_all,
    )
    energy = functional(q, p, alpha=pb.renyi_order,
                        kl_max=float("inf"), eps=pb.eps)                    # (B, N, V)
    logits = -energy / tau_eff                                              # (B, N, V)

    if degenerate is not None:
        logits = torch.where(degenerate.unsqueeze(-1), torch.zeros_like(logits), logits)
    return logits


def _decode_family(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) or (B, N, K, K) posterior (co)variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) logits = -D_configured(q || pi_v)/tau_eff
    r"""Family/divergence-consistent decode (PB-14): logits = -D_configured(q_i || pi_v)/tau_eff.

    Scores the posterior q_i against every vocabulary prior pi_v through the CONFIGURED belief family
    ``pb.family`` and divergence functional ``pb.divergence_family`` at ``alpha=pb.renyi_order``, so
    the readout matches the E-step geometry rather than a hardcoded gaussian alpha=1 KL. As in the
    other decode kernels the seam is invoked with ``kl_max=inf`` (a DECODE must preserve the full
    divergence ranking over the vocabulary). The vocabulary prior table is intentionally DIAGONAL in
    every family (PB-11); a full family promotes it with ``diag_embed`` so a full q is scored against
    a diagonal-as-full prior. Broadcasting the functional over V materializes a (B, N, V) energy
    (a full family a (B, N, V, K, K) workspace): general but O(B*N*V*...), so it serves only as the
    oracle of ``reference_decode``. For a canonical gaussian + renyi + alpha=1 config this equals the
    fast ``diagonal_chunked``/``full_chunked`` kernels."""
    return _family_logits(pb, mu_q, sigma_q, tau_eff)


@register_decode(
    "expected_likelihood_chunked",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_expected_likelihood_chunked,
    fused_ce_supports_stats=True,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_expected_likelihood_chunked(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances
    tau_eff: torch.Tensor,               # () effective temperature
) -> torch.Tensor:                       # (B, N, V) expected-likelihood logits
    r"""Expected-likelihood decode: logits from the exact Gaussian-convolution marginal.

        E_{x~q_i}[N(x; mu_v, Sigma_v)] = N(mu_q_i; mu_v, Sigma_q_i + Sigma_v)
    (the Gaussian convolution identity: the expectation of one Gaussian density under another
    integrates to a Gaussian in the mean difference with the SUMMED covariances). Taking the log,
    dropping the v-independent constant -(K/2) log(2 pi), and tempering by tau_eff (diagonal
    family):

        logit_{i,v} = -1/(2 tau_eff) * sum_k [ (mu_q - mu_v)^2 / (sigma_q + sigma_v)
                                               + log(sigma_q + sigma_v) ].

    Unlike the -KL readout this scores q as an OBSERVATION model marginal (Bayes-exact up to the
    dropped constant), so a diffuse prior pi_v is penalized through log(sigma_q + sigma_v) rather
    than rewarded through the KL's 1/sigma_v flattening. DIAGONAL family only (registered without
    is_full, so the config rank cross-check pairs it with diagonal families by construction).
    Chunked over the vocabulary: the couplings sigma_q + sigma_v block the diagonal kernel's
    single-matmul trick, so each chunk broadcasts a (B, N, Vc, K) workspace; the (B, N, V) output
    is inherent to producing every logit (the training memory win is the fused CE twin
    ``decode_ce_expected_likelihood_chunked``).
    """
    sigma_v_all = bounded_variance_from_log(
        pb._decode_sigma_log_table(), eps=pb.eps,
    )                                                                         # (V, K)
    mu_v_all = pb._decode_mu_table()                                     # (V, K) decode table (untied if set)
    chunk = pb.decode_chunk_size
    V = pb.vocab_size

    logit_chunks = []
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        d = mu_q.unsqueeze(-2) - mu_v_all[v0:v1]                         # (B, N, Vc, K)
        s = sigma_q.unsqueeze(-2) + sigma_v_all[v0:v1]                   # (B, N, Vc, K) convolved variances
        logit_chunks.append(-0.5 * (d ** 2 / s + torch.log(s)).sum(-1) / tau_eff)  # (B, N, Vc)
    return torch.cat(logit_chunks, dim=-1)                               # (B, N, V)


def _linear_logits(
    pb: PriorBank,
    x:  torch.Tensor,                    # (..., K) decode input
    W:  torch.Tensor,                    # (V, K) output weight (or a vocab chunk of it)
) -> torch.Tensor:                       # (..., V) logits x W^T, fp32 dtype when autocast ran
    r"""The linear decode's logits matmul :math:`x W^\top`.

    With ``pb.decode_logits_amp_dtype`` set it runs under that autocast dtype (low-precision
    operands, fp32 accumulation) and is cast to fp32, so the bias, logsumexp and cross-entropy
    that follow run in fp32; otherwise it is the plain matmul in the caller's precision. The
    autocast GEMM stores its output in the autocast dtype, so the returned logits carry that dtype's
    precision (bf16: unit roundoff :math:`2^{-8}`, a rounding error up to 0.125 for logits between 32 and 64); the fp32 cast
    restores no bits, and per-token cross-entropy and ranking metrics see the rounded values.
    """
    dtype = getattr(pb, "decode_logits_amp_dtype", None)
    if dtype is None:
        return x @ W.transpose(-1, -2)
    with torch.autocast(device_type=x.device.type, dtype=dtype):
        logits = x @ W.transpose(-1, -2)
    return logits.float()


@register_decode(
    "linear",
    supports_chunked=True,
    fused_ce=PriorBank.decode_ce_linear_chunked,
    fused_ce_supports_stats=True,
    can_omit_base_mean=True,
    can_omit_base_variance=True,
)
def _decode_linear(
    pb:      PriorBank,
    mu_q:    torch.Tensor,               # (B, N, K) posterior means
    sigma_q: torch.Tensor,               # (B, N, K) posterior variances (DISCARDED)
    tau_eff: torch.Tensor,               # () effective temperature (DISCARDED)
) -> torch.Tensor:                       # (B, N, V) logits = mu_q @ W^T (+ b)
    r"""Linear-projection decode: logits = mu_q @ W^T (+ b).

    The one authorized neural exception: a single learned (V, K) output weight applied to the
    converged mean, with NO KL geometry at the decode boundary (the decode temperature is discarded;
    only encode + the E-step remain gauge-aware). Realized as a raw nn.Parameter matmul, not an
    nn.Linear module. A learned per-vocab log-unigram bias ``b`` (V,) is added (see __init__; a
    bank without one, such as a bias-free baseline decoder, skips it). Prior-bank KL readout remains available through the non-linear modes; this is
    the opt-in ablation the user uses to compare with/without the prior-bank decode.
    """
    x = mu_q                                                           # bare converged mean
    logits = _linear_logits(pb, x, pb.output_proj_weight)              # (B, N, V)
    if pb.output_proj_bias is not None:
        logits = logits + pb.output_proj_bias                           # learned log-unigram prior
    return logits
