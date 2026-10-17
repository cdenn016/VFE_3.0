r"""Triton kernels for the fused expected-likelihood cross-entropy (see ``expected_likelihood.py``).

This module imports ``triton`` at import time, so only ``expected_likelihood.py`` imports it, and only
when the fused route launches. Each kernel walks one axis of the (S, V) pair grid in register tiles;
the (S, V, K) workspace exists only as a (BI, BV) tile per feature k inside the dynamic loops over k
(dynamic loops keep ptxas from hoisting all K feature loads, the spill source of a static unroll; the
energy sums k in blocks of 16, two levels, see ``_energy_tile``). Tables arrive transposed, (K, V), so
each feature row loads coalesced over v.

Kernels (one owner per output element, so no atomics and no partial buffers):
  ``el_fwd_kernel``       program = BI rows; online logsumexp over V; writes m_i, log l_i and CE_i.
                          lse_i = m_i + log l_i is never formed: rounding it to the ulp of m_i (the logits
                          are large at a small tau) put an error into every softmax weight of the row.
  ``el_bwd_rows_kernel``  program = BI rows; writes dCE/dmu_q, dCE/dsigma_q and sum_v w_iv E_iv.
  ``el_bwd_cols_kernel``  program = BV vocabulary entries; writes dCE/dmu_v, dCE/dsigma_v, dCE/du.
"""
from typing import Dict

import triton
import triton.language as tl

# Captured at import, the moment ``triton.jit`` below decides between compiling and interpreting.
INTERPRETED: bool = bool(triton.knobs.runtime.interpret)

_LN2: tl.constexpr = tl.constexpr(0.6931471805599453)
# Features per inner partial sum of the energy (two-level accumulation over k; audit 2026-10-10 round 2, item 56).
_KB: tl.constexpr = tl.constexpr(16)


@triton.jit
def _energy_terms(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, k,
                  K: tl.constexpr, FAST_LOG: tl.constexpr):
    r"""Feature k's terms (d^2 / s, log s) of the energy, (BI, BV) each (log2 s if FAST_LOG)."""
    a = tl.load(MQ + offs_i * K + k, mask=mi, other=0.0)            # (BI,)
    b = tl.load(SQ + offs_i * K + k, mask=mi, other=1.0)            # (BI,)
    mv = tl.load(MVT + k * V + offs_v, mask=mv_ok, other=0.0)       # (BV,)
    sv = tl.load(SVT + k * V + offs_v, mask=mv_ok, other=1.0)       # (BV,)
    d = a[:, None] - mv[None, :]
    s = b[:, None] + sv[None, :]
    if FAST_LOG:
        lg = tl.inline_asm_elementwise("lg2.approx.f32 $0, $1;", "=r,r", [s],
                                       dtype=tl.float32, is_pure=True, pack=1)
    else:
        lg = tl.log(s)
    return d * d / s, lg


@triton.jit
def _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V,
                 K: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr, FAST_LOG: tl.constexpr):
    r"""E_iv = \sum_k [ (mu_q,ik - mu_v,vk)^2 / s_ivk + log s_ivk ],  s = sigma_q,ik + sigma_v,vk.

    Two-level accumulation: partial sums over blocks of _KB = 16 consecutive features, added into the running
    sums once per block, so the worst-case fp32 accumulation bound is about (16 + K/16) u instead of K u. In a
    float32 emulation of this order at tau = 0.1 the per-row CE error fell from 3.8e-4 to 1.4e-4 at K = 128 and
    from 3.6e-3 to 1.1e-3 at K = 512 (the per-term rounding remains). All three kernels form E here, so the forward
    and every gradient read the same energy.
    """
    dt = MQ.dtype.element_ty
    Q = tl.zeros([BI, BV], dt)                                       # (BI, BV) sum_k d^2 / s
    L = tl.zeros([BI, BV], dt)                                       # (BI, BV) sum_k log s (log2 s if FAST_LOG)
    for k0 in range(0, (K // _KB) * _KB, _KB):                       # full blocks of _KB features
        q = tl.zeros([BI, BV], dt)
        lb = tl.zeros([BI, BV], dt)
        for j in range(_KB):
            dq, dl = _energy_terms(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, k0 + j, K, FAST_LOG)
            q += dq
            lb += dl
        Q += q
        L += lb
    q = tl.zeros([BI, BV], dt)                                       # the K mod _KB remaining features
    lb = tl.zeros([BI, BV], dt)
    for k in range((K // _KB) * _KB, K):
        dq, dl = _energy_terms(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, k, K, FAST_LOG)
        q += dq
        lb += dl
    Q += q
    L += lb
    if FAST_LOG:
        L = L * _LN2
    return Q + L


@triton.jit
def _pair_weights(E, c, U, offs_v, mv_ok, mi, t, m, log_l, scale, HAS_U: tl.constexpr):
    r"""w_iv = (p_iv - [v = t_i]) * scale with p_iv = exp((logit_iv - m_i) - log l_i); zero off the grid.

    The leading difference logit - m is exact for the logits that carry the mass (Sterbenz), so p carries
    the rounding of log l rather than the ulp of m that lse = m + log l would add.
    """
    logit = -c * E
    if HAS_U:
        logit += tl.load(U + offs_v, mask=mv_ok, other=0.0)[None, :]
    p = tl.exp((logit - m[:, None]) - log_l[:, None])
    hit = offs_v[None, :] == t[:, None]
    w = (p - tl.where(hit, 1.0, 0.0)) * scale
    return tl.where(mi[:, None] & mv_ok[None, :], w, 0.0)


@triton.jit
def el_fwd_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, M, LOGL, CE, S, V,
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
    log_l = tl.log(l)
    tl.store(M + offs_i, m, mask=mi)
    tl.store(LOGL + offs_i, log_l, mask=mi)
    tl.store(CE + offs_i, (m - lt) + log_l, mask=mi)                 # lse - lt with m and log l kept apart


@triton.jit
def el_bwd_rows_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, M, LOGL, SCALE, GMQ, GSQ, GE, S, V,
                       K: tl.constexpr, KP: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
                       HAS_U: tl.constexpr, FAST_LOG: tl.constexpr):
    pid = tl.program_id(0)
    offs_i = pid * BI + tl.arange(0, BI)
    mi = offs_i < S
    c = 0.5 / tl.load(TAU)
    scale = tl.load(SCALE)
    t = tl.load(TGT + offs_i, mask=mi, other=-1)
    m = tl.load(M + offs_i, mask=mi, other=0.0)                     # lse = m + log_l, kept apart
    log_l = tl.load(LOGL + offs_i, mask=mi, other=0.0)
    kk = tl.arange(0, KP)
    dt = MQ.dtype.element_ty
    acc_m = tl.zeros([BI, KP], dt)                                   # sum_v w r
    acc_s = tl.zeros([BI, KP], dt)                                   # sum_v w (r^2 - 1/s)
    acc_e = tl.zeros([BI], dt)                                       # sum_v w E
    for v0 in range(0, V, BV):
        offs_v = v0 + tl.arange(0, BV)
        mv_ok = offs_v < V
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        w = _pair_weights(E, c, U, offs_v, mv_ok, mi, t, m, log_l, scale, HAS_U)
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
def el_bwd_cols_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, M, LOGL, SCALE, GMV, GSV, GU, S, V,
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
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        # The row state is loaded after the energy tile so that it is not live across it: loaded before, the
        # second row vector (log l, F48) took the kernel from 4 to 300 bytes of sm_120a spill stores; here 0.
        t = tl.load(TGT + offs_i, mask=mi, other=-1)
        m = tl.load(M + offs_i, mask=mi, other=0.0)
        log_l = tl.load(LOGL + offs_i, mask=mi, other=0.0)
        w = _pair_weights(E, c, U, offs_v, mv_ok, mi, t, m, log_l, scale, HAS_U)
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
