r"""Triton kernels for the fused KL-to-prior decode cross-entropy (see ``kl_decode.py``).

This module imports ``triton`` at import time, so only ``kl_decode.py`` imports it, and only when the fused
route launches. Both kernels read one vocabulary chunk of the cuBLAS cross term :math:`A = \mathrm{lhs}\,
\mathrm{rhs}^\top` (row-major (S, Vc), the only (S, Vc) tensor of the route) and own BI rows each, walking the
chunk in (BI, BV) register tiles; the per-row state is carried across chunks through (S,) buffers, so every
output element has one owner program and no atomics are used.

Kernels:
  ``kl_ce_fwd_kernel``  online max / sum-exp / target logit across the chunk; updates M, L, T in place.
  ``kl_ce_bwd_kernel``  dCE/dA for the chunk into G (may alias A); accumulates the per-row sums for
                        dCE/dper_pos and dCE/dtau in place.
"""
from typing import Dict

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Captured at import, the moment ``triton.jit`` below decides between compiling and interpreting.
INTERPRETED: bool = bool(triton.knobs.runtime.interpret)


@triton.jit
def _exp(x, ACCURATE_EXP: tl.constexpr):
    r"""exp(x): libdevice's accurate exp when compiled (``tl.exp`` is ``ex2.approx``); ``tl.exp`` interpreted."""
    if ACCURATE_EXP:
        return libdevice.exp(x)
    else:
        return tl.exp(x)


@triton.jit
def _score_tile(A, BIAS, U, row, mi, offs_j, mj, pp, tau,
                HAS_U: tl.constexpr, PRECISE_DIV: tl.constexpr):
    r"""(centered, kl, logit) of one (BI, BV) tile, in the unfused route's operation order.

    a = A + bias, centered = 0.5 (a - per_pos), kl = clamp(centered, min=0), logit = (-kl) / tau (+ u). The
    clamp is written as a select so a NaN score stays NaN, as ``torch.clamp`` keeps it.
    """
    m2 = mi[:, None] & mj[None, :]
    a = tl.load(A + row[:, None] + offs_j[None, :], mask=m2, other=0.0)
    a = a + tl.load(BIAS + offs_j, mask=mj, other=0.0)[None, :]
    centered = 0.5 * (a - pp[:, None])
    kl = tl.where(centered < 0.0, 0.0, centered)
    if PRECISE_DIV:
        logit = tl.math.div_rn(-kl, tau)                            # IEEE division, as torch's
    else:
        logit = -kl / tau
    if HAS_U:
        logit += tl.load(U + offs_j, mask=mj, other=0.0)[None, :]
    return centered, kl, logit


@triton.jit
def kl_ce_fwd_kernel(A, BIAS, PP, U, TGT, TAU, M, L, T, S, VC, V0,
                     BI: tl.constexpr, BV: tl.constexpr,
                     HAS_U: tl.constexpr, PRECISE_DIV: tl.constexpr, ACCURATE_EXP: tl.constexpr):
    pid = tl.program_id(0)
    offs_i = pid * BI + tl.arange(0, BI)
    mi = offs_i < S
    row = offs_i.to(tl.int64) * VC                                   # (BI,) row offsets into A
    tau = tl.load(TAU)
    pp = tl.load(PP + offs_i, mask=mi, other=0.0)
    t = tl.load(TGT + offs_i, mask=mi, other=-1) - V0               # chunk-local target (out of range: no hit)
    m = tl.load(M + offs_i, mask=mi, other=float("-inf"))           # running max
    l = tl.load(L + offs_i, mask=mi, other=0.0)                     # running sum exp(logit - m)
    lt = tl.load(T + offs_i, mask=mi, other=0.0)                    # target logit
    for j0 in range(0, VC, BV):
        offs_j = j0 + tl.arange(0, BV)
        mj = offs_j < VC
        _, _, logit = _score_tile(A, BIAS, U, row, mi, offs_j, mj, pp, tau, HAS_U, PRECISE_DIV)
        logit = tl.where(mj[None, :], logit, float("-inf"))
        # A target of a later chunk can fall on a padded column of this tile, so the hit needs mj.
        hit = (offs_j[None, :] == t[:, None]) & mj[None, :]
        lt += tl.sum(tl.where(hit, logit, 0.0), 1)
        m_new = tl.maximum(m, tl.max(logit, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)        # an all -inf prefix contributes 0, not NaN
        l = l * _exp(m - m_ref, ACCURATE_EXP) + tl.sum(_exp(logit - m_ref[:, None], ACCURATE_EXP), 1)
        m = m_new
    tl.store(M + offs_i, m, mask=mi)
    tl.store(L + offs_i, l, mask=mi)
    tl.store(T + offs_i, lt, mask=mi)


@triton.jit
def kl_ce_bwd_kernel(A, G, BIAS, PP, U, TGT, TAU, M, LOGL, GROW, DPP, DTAU, S, VC, V0,
                     BI: tl.constexpr, BV: tl.constexpr, HAS_U: tl.constexpr,
                     PRECISE_DIV: tl.constexpr, ACCURATE_EXP: tl.constexpr, NEED_TAU: tl.constexpr):
    r"""G = dCE/dA = 0.5 [centered >= 0] (-(w / tau)), w = g_i (p_iv - [v = t_i]); DPP -= sum_v G, DTAU += sum_v w kl."""
    pid = tl.program_id(0)
    offs_i = pid * BI + tl.arange(0, BI)
    mi = offs_i < S
    row = offs_i.to(tl.int64) * VC
    tau = tl.load(TAU)
    pp = tl.load(PP + offs_i, mask=mi, other=0.0)
    t = tl.load(TGT + offs_i, mask=mi, other=-1) - V0
    m = tl.load(M + offs_i, mask=mi, other=0.0)                     # lse = m + log_l, kept apart
    log_l = tl.load(LOGL + offs_i, mask=mi, other=0.0)
    g = tl.load(GROW + offs_i, mask=mi, other=0.0)
    dt = BIAS.dtype.element_ty
    acc_pp = tl.zeros([BI], dt)                                      # sum_v G
    acc_tau = tl.zeros([BI], dt)                                     # sum_v w kl
    for j0 in range(0, VC, BV):
        offs_j = j0 + tl.arange(0, BV)
        mj = offs_j < VC
        m2 = mi[:, None] & mj[None, :]
        centered, kl, logit = _score_tile(A, BIAS, U, row, mi, offs_j, mj, pp, tau, HAS_U, PRECISE_DIV)
        p = _exp((logit - m[:, None]) - log_l[:, None], ACCURATE_EXP)
        hit = (offs_j[None, :] == t[:, None]) & mj[None, :]
        w = g[:, None] * (p - tl.where(hit, 1.0, 0.0))
        if PRECISE_DIV:
            dkl = -tl.math.div_rn(w, tau)
        else:
            dkl = -(w / tau)
        da = tl.where(m2 & (centered >= 0.0), 0.5 * dkl, 0.0)       # clamp passes the gradient where input >= 0
        tl.store(G + row[:, None] + offs_j[None, :], da, mask=m2)
        acc_pp += tl.sum(da, 1)
        if NEED_TAU:
            acc_tau += tl.sum(tl.where(m2, w * kl, 0.0), 1)
    tl.store(DPP + offs_i, tl.load(DPP + offs_i, mask=mi, other=0.0) - acc_pp, mask=mi)
    if NEED_TAU:
        tl.store(DTAU + offs_i, tl.load(DTAU + offs_i, mask=mi, other=0.0) + acc_tau, mask=mi)


def launch_config() -> Dict[str, Dict[str, int]]:
    r"""Tile and launch parameters per kernel (shared by the launcher and the sm_120 compile test).

    The kernels are bandwidth-bound streams over the (S, Vc) chunk; a short row tile keeps enough programs in
    flight (S = 8192 rows give 4096 programs) and a wide column tile keeps each load coalesced. On the idle RTX
    5090 at S = 8192, V = 50257, Vc = 8192 this tile ran the forward over the whole vocabulary in 0.99 ms (1.7 TB/s
    of A read) and the backward in 1.74 ms, against 1.24 and 1.79 ms for an (8, 256) tile and 2.4 and 3.9 ms for
    (8, 512) at eight warps.
    """
    return {
        "fwd": dict(BI=2, BV=512, num_warps=4, num_stages=2),
        "bwd": dict(BI=2, BV=512, num_warps=4, num_stages=2),
    }
