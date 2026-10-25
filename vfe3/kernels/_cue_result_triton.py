r"""Triton kernels for the fused cue/result memory on diagonal records (see ``cue_result.py``).

This module imports ``triton`` at import time, so only ``cue_result.py`` imports it, and only when the fused route
launches. Every kernel works on one record row :math:`i` at a time and walks a (BA, BK) register tile of components
:math:`a` by coordinates :math:`k`; the (rows, M, K) workspace of the torch route exists only as that tile. The
per-pair quantities with a rank index (:math:`t_{ia,r}`, :math:`G_{ia,rs}` and their adjoints) are kept in planes of
shape (P, rows, M), one plane per rank index, and the rank loops are unrolled (``RANK`` and ``RP``, the rank padded to
a power of two, are constexprs).

Per element (:math:`W = c_{ik} + \lambda_{ak}`, :math:`\delta = \xi_{ik} - \nu_{ak}`, gain :math:`= \lambda/W`,
:math:`g = \mathrm{gain}\,\delta`, :math:`p = \mathrm{gain}\,c`):

.. math::
    \ell_{ia} = \log\hat\pi_{ia} - \tfrac12\sum_k(\delta^2/W + \log W) - \tfrac K2\log 2\pi,\qquad
    t_{ia,r} = \sum_k g V_{akr},\qquad G_{ia,rs} = \sum_k V_{akr}\,p\,V_{aks},

    m^Y_{iak} = \omega_{ak} + g + \sum_r t_{ia,r}U_{akr},\qquad
    v^Y_{iak} = p(1 + a)^2 + \max\Big(\sum_{rs}U_{akr}G_{ia,rs}U_{aks} - pa^2, 0\Big) + q_{ak},
    \qquad a = \sum_rU_{akr}V_{akr},

the regression variance as a sum of squares with the result noise added last (audit 2026-10-25 F02).

Kernels (one owner per output element, no atomics):
  ``cr_fwd_weights_kernel``   program = (row, BA components); k loop; writes the unnormalized log weights and the
                              t and G planes.
  ``cr_fwd_moments_kernel``   program = (row, BK coordinates); loop over component tiles with a weighted Chan merge;
                              writes the result mean and variances.
  ``cr_bwd_pairs_kernel``     program = (row, BA components); two k loops: recomputes t and G, then accumulates
                              wbar = dL/dw, tbar and Gbar; writes their planes and the t and G planes.
  ``cr_bwd_rows_kernel``      program = (row, BK coordinates); loop over component tiles; writes dL/dxi and dL/dc.
  ``cr_bwd_cols_kernel``      program = (BA components, BK coordinates, row split); loop over the split's rows;
                              writes one partial of dL/dnu, dL/dlambda, dL/domega, dL/dq, dL/dU and dL/dV.
"""
from typing import Dict

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Captured at import, the moment ``triton.jit`` below decides between compiling and interpreting.
INTERPRETED: bool = bool(triton.knobs.runtime.interpret)


@triton.jit
def _exp(x, ACC: tl.constexpr):
    r"""exp: libdevice's accurate exp when compiled (``tl.exp`` is ``ex2.approx``); ``tl.exp`` interpreted."""
    if ACC:
        return libdevice.exp(x)
    else:
        return tl.exp(x)


@triton.jit
def _log(x, ACC: tl.constexpr):
    r"""log: libdevice's accurate log when compiled; ``tl.log`` interpreted."""
    if ACC:
        return libdevice.log(x)
    else:
        return tl.log(x)


@triton.jit
def _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K):
    r"""The (BA, BK) element tile of row ``row``: mask, table offsets, c, lambda, 1/W, delta, gain, g, p."""
    m2 = ma[:, None] & mk[None, :]
    x = tl.load(X + row * K + offs_k, mask=mk, other=0.0)                      # (BK,)
    c = tl.load(C + row * K + offs_k, mask=mk, other=1.0)                      # (BK,)
    idx = offs_a[:, None].to(tl.int64) * K + offs_k[None, :]                   # (BA, BK) into (M, K)
    nu = tl.load(NU + idx, mask=m2, other=0.0)
    lam = tl.load(LAM + idx, mask=m2, other=1.0)
    W = c[None, :] + lam
    inv = 1.0 / W
    delta = x[None, :] - nu
    gain = lam * inv
    g = tl.where(m2, gain * delta, 0.0)
    p = tl.where(m2, gain * c[None, :], 0.0)
    return m2, idx, c, lam, W, inv, delta, gain, g, p


@triton.jit
def _load_planes(P, RM, row, M, offs_a, ma, offs_p, valid_p):
    r"""(BA, NP) tile of planes offs_p of a (P, rows, M) buffer at row ``row``."""
    ptr = P + offs_p[None, :].to(tl.int64) * RM + row * M + offs_a[:, None]
    return tl.load(ptr, mask=ma[:, None] & valid_p[None, :], other=0.0)


@triton.jit
def _col(tile, offs_p, j):
    r"""Column j of a (BA, NP) register tile, as (BA, 1)."""
    return tl.sum(tile * (offs_p == j)[None, :], 1)[:, None]


@triton.jit
def _result_moments(U, V, t, Gm, offs_r, offs_rr, idx, m2, OMEGA, Q, g, p,
                    RANK: tl.constexpr, RP: tl.constexpr):
    r"""m^Y and v^Y of the tile from the (BA, RP) t and (BA, RP*RP) G tiles, uv = a = sum_r U_r V_r, and the
    indicator ind = [quad - p a^2 >= 0] of the unclamped remainder.

    v^Y = p (1 + a)^2 + max(quad - p a^2, 0) + q: the regression variance as diag(F P F^T)_k = p_k (1 + a_k)^2 plus the
    sum of squares over l != k, clamped to its sign, with the result noise added last (audit 2026-10-25 F02; as the
    torch route's ``_diagonal_tile``)."""
    omega = tl.load(OMEGA + idx, mask=m2, other=0.0)
    q = tl.load(Q + idx, mask=m2, other=0.0)
    mY = omega + g
    uv = tl.zeros_like(g)
    quad = tl.zeros_like(g)
    for r in tl.static_range(RANK):
        u_r = tl.load(U + idx * RANK + r, mask=m2, other=0.0)
        v_r = tl.load(V + idx * RANK + r, mask=m2, other=0.0)
        mY += _col(t, offs_r, r) * u_r
        uv += u_r * v_r
        for s in tl.static_range(RANK):
            u_s = tl.load(U + idx * RANK + s, mask=m2, other=0.0)
            quad += _col(Gm, offs_rr, r * RP + s) * u_r * u_s
    rem = quad - p * uv * uv
    ind = tl.where(rem >= 0.0, 1.0, 0.0)                                       # torch.clamp_min passes x >= 0
    vY = (p * (1.0 + uv) * (1.0 + uv) + rem * ind) + q
    return mY, vY, uv, ind


@triton.jit
def _pair_sums(V, idx, m2, g, p, offs_r, offs_rr, BA: tl.constexpr, RANK: tl.constexpr, RP: tl.constexpr):
    r"""The tile's contributions to t (BA, RP) and G (BA, RP*RP)."""
    t = tl.zeros([BA, RP], dtype=g.dtype)
    Gm = tl.zeros([BA, RP * RP], dtype=g.dtype)
    for r in tl.static_range(RANK):
        v_r = tl.load(V + idx * RANK + r, mask=m2, other=0.0)
        t += tl.sum(g * v_r, 1)[:, None] * (offs_r == r)[None, :]
        for s in tl.static_range(RANK):
            v_s = tl.load(V + idx * RANK + s, mask=m2, other=0.0)
            Gm += tl.sum(v_r * p * v_s, 1)[:, None] * (offs_rr == r * RP + s)[None, :]
    return t, Gm


@triton.jit
def _store_planes(P, RM, row, M, offs_a, ma, vals, offs_p, valid_p):
    r"""Store a (BA, NP) tile into planes offs_p of a (P, rows, M) buffer."""
    ptr = P + offs_p[None, :].to(tl.int64) * RM + row * M + offs_a[:, None]
    tl.store(ptr, vals, mask=ma[:, None] & valid_p[None, :])


@triton.jit
def cr_fwd_weights_kernel(X, C, NU, LAM, V, LP, LOGW, T, G, R, M, K, LP_STRIDE, HALF_K_LOG2PI,
                          BA: tl.constexpr, BK: tl.constexpr, RANK: tl.constexpr, RP: tl.constexpr,
                          ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs_a = tl.program_id(1) * BA + tl.arange(0, BA)
    ma = offs_a < M
    RM = R.to(tl.int64) * M
    offs_r = tl.arange(0, RP)
    offs_rr = tl.arange(0, RP * RP)
    dt = X.dtype.element_ty
    s_acc = tl.zeros([BA], dtype=dt)
    t = tl.zeros([BA, RP], dtype=dt)
    Gm = tl.zeros([BA, RP * RP], dtype=dt)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < K
        m2, idx, c, lam, W, inv, delta, gain, g, p = _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K)
        s_acc += tl.sum(tl.where(m2, delta * delta * inv + _log(W, ACC), 0.0), 1)
        dt_, dG = _pair_sums(V, idx, m2, g, p, offs_r, offs_rr, BA, RANK, RP)
        t += dt_
        Gm += dG
    lp = tl.load(LP + row * LP_STRIDE + offs_a, mask=ma, other=0.0)
    logw = lp - 0.5 * s_acc - HALF_K_LOG2PI
    tl.store(LOGW + row * M + offs_a, tl.where(ma, logw, float("-inf")), mask=ma)
    _store_planes(T, RM, row, M, offs_a, ma, t, offs_r, offs_r < RANK)
    _store_planes(G, RM, row, M, offs_a, ma, Gm, offs_rr, ((offs_rr // RP) < RANK) & ((offs_rr % RP) < RANK))


@triton.jit
def cr_fwd_moments_kernel(X, C, NU, LAM, OMEGA, Q, U, V, LOGWN, T, G, MEAN, VAR, R, M, K,
                          BA: tl.constexpr, BK: tl.constexpr, RANK: tl.constexpr, RP: tl.constexpr,
                          ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs_k = tl.program_id(1) * BK + tl.arange(0, BK)
    mk = offs_k < K
    RM = R.to(tl.int64) * M
    offs_r = tl.arange(0, RP)
    offs_rr = tl.arange(0, RP * RP)
    valid_r = offs_r < RANK
    valid_rr = ((offs_rr // RP) < RANK) & ((offs_rr % RP) < RANK)
    dt = X.dtype.element_ty
    w_run = tl.zeros([1], dtype=dt)                                             # running weight mass
    mean = tl.zeros([BK], dtype=dt)
    m2acc = tl.zeros([BK], dtype=dt)                                            # running sum w (v + (m - mean)^2)
    for a0 in range(0, M, BA):
        offs_a = a0 + tl.arange(0, BA)
        ma = offs_a < M
        m2, idx, c, lam, W, inv, delta, gain, g, p = _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K)
        t = _load_planes(T, RM, row, M, offs_a, ma, offs_r, valid_r)
        Gm = _load_planes(G, RM, row, M, offs_a, ma, offs_rr, valid_rr)
        mY, vY, uv, ind = _result_moments(U, V, t, Gm, offs_r, offs_rr, idx, m2, OMEGA, Q, g, p, RANK, RP)
        w = tl.where(ma, _exp(tl.load(LOGWN + row * M + offs_a, mask=ma, other=float("-inf")), ACC), 0.0)
        bw = tl.sum(w, 0)
        bw_safe = tl.where(bw > 0.0, bw, 1.0)
        bmean = tl.sum(w[:, None] * mY, 0) / bw_safe                            # (BK,)
        dev = mY - bmean[None, :]
        bm2 = tl.sum(w[:, None] * (vY + dev * dev), 0)                          # (BK,)
        total = w_run + bw
        frac = bw / tl.where(total > 0.0, total, 1.0)
        d = bmean - mean
        mean += d * frac
        m2acc += bm2 + d * d * w_run * frac
        w_run = total
    var = m2acc / tl.where(w_run > 0.0, w_run, 1.0)
    tl.store(MEAN + row * K + offs_k, mean, mask=mk)
    tl.store(VAR + row * K + offs_k, var, mask=mk)


@triton.jit
def cr_bwd_pairs_kernel(X, C, NU, LAM, OMEGA, Q, U, V, LOGWN, MEAN, GM, GV, T, G, WBAR, TBAR, GBAR, R, M, K,
                        BA: tl.constexpr, BK: tl.constexpr, RANK: tl.constexpr, RP: tl.constexpr,
                        ACC: tl.constexpr):
    r"""wbar_ia = sum_k gm m^Y + gv (v^Y + (m^Y - mean)^2); tbar_r = sum_k mbar U_r; Gbar_rs = sum_k vbar U_r U_s."""
    row = tl.program_id(0).to(tl.int64)
    offs_a = tl.program_id(1) * BA + tl.arange(0, BA)
    ma = offs_a < M
    RM = R.to(tl.int64) * M
    offs_r = tl.arange(0, RP)
    offs_rr = tl.arange(0, RP * RP)
    valid_r = offs_r < RANK
    valid_rr = ((offs_rr // RP) < RANK) & ((offs_rr % RP) < RANK)
    dt = X.dtype.element_ty
    t = tl.zeros([BA, RP], dtype=dt)
    Gm = tl.zeros([BA, RP * RP], dtype=dt)
    for k0 in range(0, K, BK):                                                  # recompute t and G
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < K
        m2, idx, c, lam, W, inv, delta, gain, g, p = _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K)
        dt_, dG = _pair_sums(V, idx, m2, g, p, offs_r, offs_rr, BA, RANK, RP)
        t += dt_
        Gm += dG
    _store_planes(T, RM, row, M, offs_a, ma, t, offs_r, valid_r)
    _store_planes(G, RM, row, M, offs_a, ma, Gm, offs_rr, valid_rr)
    w = tl.where(ma, _exp(tl.load(LOGWN + row * M + offs_a, mask=ma, other=float("-inf")), ACC), 0.0)
    wbar = tl.zeros([BA], dtype=dt)
    tbar = tl.zeros([BA, RP], dtype=dt)
    Gbar = tl.zeros([BA, RP * RP], dtype=dt)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < K
        m2, idx, c, lam, W, inv, delta, gain, g, p = _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K)
        mY, vY, uv, ind = _result_moments(U, V, t, Gm, offs_r, offs_rr, idx, m2, OMEGA, Q, g, p, RANK, RP)
        mean = tl.load(MEAN + row * K + offs_k, mask=mk, other=0.0)
        gm = tl.load(GM + row * K + offs_k, mask=mk, other=0.0)
        gv = tl.load(GV + row * K + offs_k, mask=mk, other=0.0)
        dev = mY - mean[None, :]
        wbar += tl.sum(tl.where(m2, gm[None, :] * mY + gv[None, :] * (vY + dev * dev), 0.0), 1)
        mbar = tl.where(m2, w[:, None] * (gm[None, :] + 2.0 * gv[None, :] * dev), 0.0)
        vbar = tl.where(m2, w[:, None] * gv[None, :], 0.0)
        for r in tl.static_range(RANK):
            u_r = tl.load(U + idx * RANK + r, mask=m2, other=0.0)
            tbar += tl.sum(mbar * u_r, 1)[:, None] * (offs_r == r)[None, :]
            for s in tl.static_range(RANK):
                u_s = tl.load(U + idx * RANK + s, mask=m2, other=0.0)
                Gbar += tl.sum(vbar * ind * u_r * u_s, 1)[:, None] * (offs_rr == r * RP + s)[None, :]
    tl.store(WBAR + row * M + offs_a, wbar, mask=ma)
    _store_planes(TBAR, RM, row, M, offs_a, ma, tbar, offs_r, valid_r)
    _store_planes(GBAR, RM, row, M, offs_a, ma, Gbar, offs_rr, valid_rr)


@triton.jit
def _chain(U, V, t, Gm, tbar, Gbar, offs_r, offs_rr, row, M, offs_a, ma, idx, m2, OMEGA, Q, LOGWN, LBAR,
           mean, gm, gv, c, lam, inv, delta, gain, g, p, RANK: tl.constexpr, RP: tl.constexpr, ACC: tl.constexpr):
    r"""The element adjoints of one tile: mbar, vbar, and dL/d(delta), dL/dlambda, dL/dc (module docstring of
    ``cue_result.py``)."""
    mY, vY, uv, ind = _result_moments(U, V, t, Gm, offs_r, offs_rr, idx, m2, OMEGA, Q, g, p, RANK, RP)
    w = tl.where(ma, _exp(tl.load(LOGWN + row * M + offs_a, mask=ma, other=float("-inf")), ACC), 0.0)
    lbar = tl.load(LBAR + row * M + offs_a, mask=ma, other=0.0)
    mbar = tl.where(m2, w[:, None] * (gm[None, :] + 2.0 * gv[None, :] * (mY - mean[None, :])), 0.0)
    vbar = tl.where(m2, w[:, None] * gv[None, :], 0.0)
    gbar = mbar
    vv = tl.zeros_like(g)
    for r in tl.static_range(RANK):
        v_r = tl.load(V + idx * RANK + r, mask=m2, other=0.0)
        gbar += _col(tbar, offs_r, r) * v_r
        for s in tl.static_range(RANK):
            v_s = tl.load(V + idx * RANK + s, mask=m2, other=0.0)
            vv += _col(Gbar, offs_rr, r * RP + s) * v_r * v_s
    pbar = vbar * ((1.0 + uv) * (1.0 + uv) - ind * uv * uv) + vv               # Gbar already carries ind
    dbar = gbar * gain - lbar[:, None] * delta * inv
    gainbar = gbar * delta + pbar * c[None, :]
    Wbar = -gainbar * lam * inv * inv - 0.5 * lbar[:, None] * (inv - delta * delta * inv * inv)
    lambar = gainbar * inv + Wbar
    cbar = Wbar + pbar * gain
    zero = tl.zeros_like(g)
    return (mbar, vbar, tl.where(m2, dbar, zero), tl.where(m2, lambar, zero), tl.where(m2, cbar, zero), uv, ind)


@triton.jit
def cr_bwd_rows_kernel(X, C, NU, LAM, OMEGA, Q, U, V, LOGWN, MEAN, GM, GV, T, G, TBAR, GBAR, LBAR, DX, DC, R, M, K,
                       BA: tl.constexpr, BK: tl.constexpr, RANK: tl.constexpr, RP: tl.constexpr,
                       ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs_k = tl.program_id(1) * BK + tl.arange(0, BK)
    mk = offs_k < K
    RM = R.to(tl.int64) * M
    offs_r = tl.arange(0, RP)
    offs_rr = tl.arange(0, RP * RP)
    valid_r = offs_r < RANK
    valid_rr = ((offs_rr // RP) < RANK) & ((offs_rr % RP) < RANK)
    dt = X.dtype.element_ty
    mean = tl.load(MEAN + row * K + offs_k, mask=mk, other=0.0)
    gm = tl.load(GM + row * K + offs_k, mask=mk, other=0.0)
    gv = tl.load(GV + row * K + offs_k, mask=mk, other=0.0)
    dx = tl.zeros([BK], dtype=dt)
    dc = tl.zeros([BK], dtype=dt)
    for a0 in range(0, M, BA):
        offs_a = a0 + tl.arange(0, BA)
        ma = offs_a < M
        m2, idx, c, lam, W, inv, delta, gain, g, p = _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K)
        t = _load_planes(T, RM, row, M, offs_a, ma, offs_r, valid_r)
        Gm = _load_planes(G, RM, row, M, offs_a, ma, offs_rr, valid_rr)
        tbar = _load_planes(TBAR, RM, row, M, offs_a, ma, offs_r, valid_r)
        Gbar = _load_planes(GBAR, RM, row, M, offs_a, ma, offs_rr, valid_rr)
        mbar, vbar, dbar, lambar, cbar, uv, ind = _chain(U, V, t, Gm, tbar, Gbar, offs_r, offs_rr, row, M, offs_a, ma, idx,
                                                m2, OMEGA, Q, LOGWN, LBAR, mean, gm, gv, c, lam, inv, delta, gain,
                                                g, p, RANK, RP, ACC)
        dx += tl.sum(dbar, 0)
        dc += tl.sum(cbar, 0)
    tl.store(DX + row * K + offs_k, dx, mask=mk)
    tl.store(DC + row * K + offs_k, dc, mask=mk)


@triton.jit
def cr_bwd_cols_kernel(X, C, NU, LAM, OMEGA, Q, U, V, LOGWN, MEAN, GM, GV, T, G, TBAR, GBAR, LBAR,
                       DNU, DLAM, DOMEGA, DQ, DU, DV, R, M, K, ROWS_PER_SPLIT,
                       BA: tl.constexpr, BK: tl.constexpr, RANK: tl.constexpr, RP: tl.constexpr,
                       ACC: tl.constexpr):
    r"""One row split's partial parameter gradients; DNU ... DV are (splits, M, K[, RANK]) buffers."""
    offs_a = tl.program_id(0) * BA + tl.arange(0, BA)
    offs_k = tl.program_id(1) * BK + tl.arange(0, BK)
    split = tl.program_id(2)
    ma = offs_a < M
    mk = offs_k < K
    m2 = ma[:, None] & mk[None, :]
    RM = R.to(tl.int64) * M
    MK = M.to(tl.int64) * K
    offs_r = tl.arange(0, RP)
    offs_rr = tl.arange(0, RP * RP)
    valid_r = offs_r < RANK
    valid_rr = ((offs_rr // RP) < RANK) & ((offs_rr % RP) < RANK)
    dt = X.dtype.element_ty
    dnu = tl.zeros([BA, BK], dtype=dt)
    dlam = tl.zeros([BA, BK], dtype=dt)
    domega = tl.zeros([BA, BK], dtype=dt)
    dq = tl.zeros([BA, BK], dtype=dt)
    du = tl.zeros([BA, BK, RP], dtype=dt)
    dv = tl.zeros([BA, BK, RP], dtype=dt)
    r0 = split.to(tl.int64) * ROWS_PER_SPLIT
    r1 = tl.minimum(r0 + ROWS_PER_SPLIT, R.to(tl.int64))
    for row in range(r0, r1):
        m2, idx, c, lam, W, inv, delta, gain, g, p = _elem(X, C, NU, LAM, row, offs_a, offs_k, ma, mk, K)
        mean = tl.load(MEAN + row * K + offs_k, mask=mk, other=0.0)
        gm = tl.load(GM + row * K + offs_k, mask=mk, other=0.0)
        gv = tl.load(GV + row * K + offs_k, mask=mk, other=0.0)
        t = _load_planes(T, RM, row, M, offs_a, ma, offs_r, valid_r)
        Gm = _load_planes(G, RM, row, M, offs_a, ma, offs_rr, valid_rr)
        tbar = _load_planes(TBAR, RM, row, M, offs_a, ma, offs_r, valid_r)
        Gbar = _load_planes(GBAR, RM, row, M, offs_a, ma, offs_rr, valid_rr)
        mbar, vbar, dbar, lambar, cbar, uv, ind = _chain(U, V, t, Gm, tbar, Gbar, offs_r, offs_rr, row, M, offs_a, ma, idx,
                                                m2, OMEGA, Q, LOGWN, LBAR, mean, gm, gv, c, lam, inv, delta, gain,
                                                g, p, RANK, RP, ACC)
        dnu -= dbar
        dlam += lambar
        domega += mbar
        dq += vbar
        for r in tl.static_range(RANK):
            u_r = tl.load(U + idx * RANK + r, mask=m2, other=0.0)
            v_r = tl.load(V + idx * RANK + r, mask=m2, other=0.0)
            t_r = _col(t, offs_r, r)
            tb_r = _col(tbar, offs_r, r)
            gu = tl.zeros_like(g)                                               # sum_s G_rs U_s
            gbv = tl.zeros_like(g)                                              # sum_s Gbar_rs V_s
            for s in tl.static_range(RANK):
                gu += _col(Gm, offs_rr, r * RP + s) * tl.load(U + idx * RANK + s, mask=m2, other=0.0)
                gbv += _col(Gbar, offs_rr, r * RP + s) * tl.load(V + idx * RANK + s, mask=m2, other=0.0)
            coef = 1.0 + uv - ind * uv                                          # dv/da / (2p) = (1 + a) - ind a
            u_bar = mbar * t_r + vbar * (2.0 * p * coef * v_r + 2.0 * ind * gu)
            v_bar = g * tb_r + 2.0 * p * (vbar * coef * u_r + gbv)
            sel = (offs_r == r)[None, None, :]
            du += tl.where(m2, u_bar, 0.0)[:, :, None] * sel
            dv += tl.where(m2, v_bar, 0.0)[:, :, None] * sel
    base = split.to(tl.int64) * MK
    idx = offs_a[:, None].to(tl.int64) * K + offs_k[None, :]
    tl.store(DNU + base + idx, dnu, mask=m2)
    tl.store(DLAM + base + idx, dlam, mask=m2)
    tl.store(DOMEGA + base + idx, domega, mask=m2)
    tl.store(DQ + base + idx, dq, mask=m2)
    ridx = (base + idx)[:, :, None] * RANK + offs_r[None, None, :]
    rmask = m2[:, :, None] & (offs_r < RANK)[None, None, :]
    tl.store(DU + ridx, du, mask=rmask)
    tl.store(DV + ridx, dv, mask=rmask)


def launch_config() -> Dict[str, Dict[str, int]]:
    r"""Tile and launch parameters per kernel (shared by the launcher and the sm_120 compile test).

    Not tuned on the GPU yet. (BA, BK) = (32, 32) at four warps compiles without spills for ranks 2 and 4 on sm_120
    except the column backward, whose six parameter accumulators spilled at that tile (544 bytes at rank 2); it takes
    (16, 32).
    """
    tile = dict(BA=32, BK=32, num_warps=4, num_stages=2)
    return {"fwd_weights": dict(tile), "fwd_moments": dict(tile), "bwd_pairs": dict(tile),
            "bwd_rows": dict(tile), "bwd_cols": dict(tile, BA=16)}
