r"""Triton kernels for the fused expected-likelihood cross-entropy (see ``expected_likelihood.py``).

This module imports ``triton`` at import time, so only ``expected_likelihood.py`` imports it, and only
when the fused route launches. Each kernel walks one axis of the (S, V) pair grid in register tiles;
the (S, V, K) workspace exists only as a (BI, BV) tile per feature k inside the dynamic ``range(K)``
loop (the dynamic loop keeps ptxas from hoisting all K feature loads, the spill source of a static
unroll). Tables arrive transposed, (K, V), so each feature row loads coalesced over v.

Kernels (one owner per output element, so no atomics and no partial buffers):
  ``el_fwd_kernel``       program = BI rows; online logsumexp over V; writes lse_i and CE_i.
  ``el_bwd_rows_kernel``  program = BI rows; writes dCE/dmu_q, dCE/dsigma_q and sum_v w_iv E_iv.
  ``el_bwd_cols_kernel``  program = BV vocabulary entries; writes dCE/dmu_v, dCE/dsigma_v, dCE/du.
"""
from typing import Dict

import triton
import triton.language as tl

# Captured at import, the moment ``triton.jit`` below decides between compiling and interpreting.
INTERPRETED: bool = bool(triton.knobs.runtime.interpret)

_LN2: tl.constexpr = tl.constexpr(0.6931471805599453)


@triton.jit
def _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V,
                 K: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr, FAST_LOG: tl.constexpr):
    r"""E_iv = \sum_k [ (mu_q,ik - mu_v,vk)^2 / s_ivk + log s_ivk ],  s = sigma_q,ik + sigma_v,vk."""
    dt = MQ.dtype.element_ty
    Q = tl.zeros([BI, BV], dt)                                       # (BI, BV) sum_k d^2 / s
    L = tl.zeros([BI, BV], dt)                                       # (BI, BV) sum_k log s (log2 s if FAST_LOG)
    for k in range(K):
        a = tl.load(MQ + offs_i * K + k, mask=mi, other=0.0)        # (BI,)
        b = tl.load(SQ + offs_i * K + k, mask=mi, other=1.0)        # (BI,)
        mv = tl.load(MVT + k * V + offs_v, mask=mv_ok, other=0.0)   # (BV,)
        sv = tl.load(SVT + k * V + offs_v, mask=mv_ok, other=1.0)   # (BV,)
        d = a[:, None] - mv[None, :]
        s = b[:, None] + sv[None, :]
        Q += d * d / s
        if FAST_LOG:
            L += tl.inline_asm_elementwise("lg2.approx.f32 $0, $1;", "=r,r", [s],
                                           dtype=tl.float32, is_pure=True, pack=1)
        else:
            L += tl.log(s)
    if FAST_LOG:
        L = L * _LN2
    return Q + L


@triton.jit
def _pair_weights(E, c, U, offs_v, mv_ok, mi, t, lse, scale, HAS_U: tl.constexpr):
    r"""w_iv = (p_iv - [v = t_i]) * scale with p_iv = exp(logit_iv - lse_i); zero off the grid."""
    logit = -c * E
    if HAS_U:
        logit += tl.load(U + offs_v, mask=mv_ok, other=0.0)[None, :]
    p = tl.exp(logit - lse[:, None])
    hit = offs_v[None, :] == t[:, None]
    w = (p - tl.where(hit, 1.0, 0.0)) * scale
    return tl.where(mi[:, None] & mv_ok[None, :], w, 0.0)


@triton.jit
def el_fwd_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, LSE, CE, S, V,
                  K: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
                  HAS_U: tl.constexpr, FAST_LOG: tl.constexpr):
    pid = tl.program_id(0)
    offs_i = pid * BI + tl.arange(0, BI)
    mi = offs_i < S
    c = 0.5 / tl.load(TAU)
    t = tl.load(TGT + offs_i, mask=mi, other=-1)
    dt = MQ.dtype.element_ty
    m = tl.full([BI], float("-inf"), dt)                             # running max
    l = tl.zeros([BI], dt)                                           # running sum exp(logit - m)
    lt = tl.zeros([BI], dt)                                          # target logit
    for v0 in range(0, V, BV):
        offs_v = v0 + tl.arange(0, BV)
        mv_ok = offs_v < V
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        logit = -c * E
        if HAS_U:
            logit += tl.load(U + offs_v, mask=mv_ok, other=0.0)[None, :]
        logit = tl.where(mv_ok[None, :], logit, float("-inf"))
        lt += tl.sum(tl.where(offs_v[None, :] == t[:, None], logit, 0.0), 1)
        m_new = tl.maximum(m, tl.max(logit, 1))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(logit - m_new[:, None]), 1)
        m = m_new
    lse = m + tl.log(l)
    tl.store(LSE + offs_i, lse, mask=mi)
    tl.store(CE + offs_i, lse - lt, mask=mi)


@triton.jit
def el_bwd_rows_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, LSE, SCALE, GMQ, GSQ, GE, S, V,
                       K: tl.constexpr, KP: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
                       HAS_U: tl.constexpr, FAST_LOG: tl.constexpr):
    pid = tl.program_id(0)
    offs_i = pid * BI + tl.arange(0, BI)
    mi = offs_i < S
    c = 0.5 / tl.load(TAU)
    scale = tl.load(SCALE)
    t = tl.load(TGT + offs_i, mask=mi, other=-1)
    lse = tl.load(LSE + offs_i, mask=mi, other=0.0)
    kk = tl.arange(0, KP)
    dt = MQ.dtype.element_ty
    acc_m = tl.zeros([BI, KP], dt)                                   # sum_v w r
    acc_s = tl.zeros([BI, KP], dt)                                   # sum_v w (r^2 - 1/s)
    acc_e = tl.zeros([BI], dt)                                       # sum_v w E
    for v0 in range(0, V, BV):
        offs_v = v0 + tl.arange(0, BV)
        mv_ok = offs_v < V
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        w = _pair_weights(E, c, U, offs_v, mv_ok, mi, t, lse, scale, HAS_U)
        acc_e += tl.sum(w * E, 1)
        for k in range(K):
            a = tl.load(MQ + offs_i * K + k, mask=mi, other=0.0)
            b = tl.load(SQ + offs_i * K + k, mask=mi, other=1.0)
            mv = tl.load(MVT + k * V + offs_v, mask=mv_ok, other=0.0)
            sv = tl.load(SVT + k * V + offs_v, mask=mv_ok, other=1.0)
            inv = 1.0 / (b[:, None] + sv[None, :])
            r = (a[:, None] - mv[None, :]) * inv
            acc_m += tl.where(kk[None, :] == k, tl.sum(w * r, 1)[:, None], 0.0)
            acc_s += tl.where(kk[None, :] == k, tl.sum(w * (r * r - inv), 1)[:, None], 0.0)
    km = mi[:, None] & (kk[None, :] < K)
    tl.store(GMQ + offs_i[:, None] * K + kk[None, :], -2.0 * c * acc_m, mask=km)
    tl.store(GSQ + offs_i[:, None] * K + kk[None, :], c * acc_s, mask=km)
    tl.store(GE + offs_i, acc_e, mask=mi)


@triton.jit
def el_bwd_cols_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, LSE, SCALE, GMV, GSV, GU, S, V,
                       K: tl.constexpr, KP: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
                       HAS_U: tl.constexpr, FAST_LOG: tl.constexpr):
    pid = tl.program_id(0)
    offs_v = pid * BV + tl.arange(0, BV)
    mv_ok = offs_v < V
    c = 0.5 / tl.load(TAU)
    scale = tl.load(SCALE)
    kk = tl.arange(0, KP)
    dt = MQ.dtype.element_ty
    acc_m = tl.zeros([KP, BV], dt)                                   # sum_i w r
    acc_s = tl.zeros([KP, BV], dt)                                   # sum_i w (r^2 - 1/s)
    acc_u = tl.zeros([BV], dt)                                       # sum_i w
    for i0 in range(0, S, BI):
        offs_i = i0 + tl.arange(0, BI)
        mi = offs_i < S
        t = tl.load(TGT + offs_i, mask=mi, other=-1)
        lse = tl.load(LSE + offs_i, mask=mi, other=0.0)
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        w = _pair_weights(E, c, U, offs_v, mv_ok, mi, t, lse, scale, HAS_U)
        acc_u += tl.sum(w, 0)
        for k in range(K):
            a = tl.load(MQ + offs_i * K + k, mask=mi, other=0.0)
            b = tl.load(SQ + offs_i * K + k, mask=mi, other=1.0)
            mv = tl.load(MVT + k * V + offs_v, mask=mv_ok, other=0.0)
            sv = tl.load(SVT + k * V + offs_v, mask=mv_ok, other=1.0)
            inv = 1.0 / (b[:, None] + sv[None, :])
            r = (a[:, None] - mv[None, :]) * inv
            acc_m += tl.where(kk[:, None] == k, tl.sum(w * r, 0)[None, :], 0.0)
            acc_s += tl.where(kk[:, None] == k, tl.sum(w * (r * r - inv), 0)[None, :], 0.0)
    km = (kk[:, None] < K) & mv_ok[None, :]
    tl.store(GMV + offs_v[None, :] * K + kk[:, None], 2.0 * c * acc_m, mask=km)
    tl.store(GSV + offs_v[None, :] * K + kk[:, None], c * acc_s, mask=km)
    tl.store(GU + offs_v, acc_u, mask=mv_ok)


def launch_config(K: int) -> Dict[str, Dict[str, int]]:
    r"""Tile and launch parameters per kernel at feature width K (shared by the launcher and tests).

    The backward kernels hold two (BI, KP) or (KP, BV) gradient accumulators with KP the padded
    power of two >= K, so the owned tile edge shrinks as K grows to keep them in registers. At K=20
    these are the configurations that compiled for sm_120a with zero spill bytes.
    """
    KP = max(16, triton.next_power_of_2(K))
    owned = 64 if KP <= 32 else max(16, 2048 // KP)
    return {
        "fwd":  dict(KP=KP, BI=32, BV=64, num_warps=4, num_stages=1),
        "rows": dict(KP=KP, BI=owned, BV=64, num_warps=8, num_stages=1),
        "cols": dict(KP=KP, BI=64, BV=owned, num_warps=8, num_stages=1),
    }
