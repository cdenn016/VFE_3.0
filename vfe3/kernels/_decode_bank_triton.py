r"""Triton kernels of the fused KL decode bank preparation (see ``decode_bank.py``).

This module imports ``triton`` at import time, so only ``decode_bank.py`` imports it, and only when the fused route
launches. Each program owns BV vocabulary rows and walks the K coordinates in BK-wide register tiles; every output
element has one owner program, so no atomics are used.

Kernels:
  ``bank_fwd_kernel``  rhs = [1/sigma_v, mc_v/sigma_v] (V, 2K) and b_v = sum_k mc_v^2/sigma_v + sum_k log sigma_v.
  ``bank_bwd_kernel``  d/dmu_v (V, K) and d/dlog sigma_v (V, K) from d rhs and d b, and per-program column sums of
                       d/dmu_v (the centering offset's adjoint).
"""
from typing import Dict

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Captured at import, the moment ``triton.jit`` below decides between compiling and interpreting.
INTERPRETED: bool = bool(triton.knobs.runtime.interpret)


@triton.jit
def _bank_tile(MU, L, C, EPS, rows, mv, offs_k, mk, K,
               MAX_LOG: tl.constexpr, ACCURATE: tl.constexpr, PRECISE_DIV: tl.constexpr):
    r"""(mask, offsets, s, inv, mc, passes) of one (BV, BK) tile.

    s = max(exp(min(l, MAX_LOG)), eps), inv = 1/s and mc = mu - c. The clamps are selects, so a NaN log-variance stays
    NaN as ``torch.clamp`` keeps it, and ``passes`` is the pass mask of their gradients (``torch.clamp`` passes the
    gradient at the bounds). ``exp`` is libdevice's when compiled (``tl.exp`` is ``ex2.approx``), and the compiled
    float32 division is IEEE ``div_rn`` as torch's.
    """
    m2 = mv[:, None] & mk[None, :]
    idx = rows[:, None] * K + offs_k[None, :]
    mu = tl.load(MU + idx, mask=m2, other=0.0)
    l = tl.load(L + idx, mask=m2, other=0.0)
    c = tl.load(C + offs_k, mask=mk, other=0.0)
    eps = tl.load(EPS)
    l_cl = tl.where(l > MAX_LOG, MAX_LOG, l)
    if ACCURATE:
        e = libdevice.exp(l_cl)
    else:
        e = tl.exp(l_cl)
    s = tl.where(e < eps, eps, e)
    if PRECISE_DIV:
        inv = tl.math.div_rn(tl.zeros_like(s) + 1.0, s)
    else:
        inv = 1.0 / s
    passes = (l <= MAX_LOG) & (e >= eps)
    return m2, idx, s, inv, mu - c[None, :], passes


@triton.jit
def bank_fwd_kernel(MU, L, C, EPS, RHS, BIAS, V, K,
                    BV: tl.constexpr, BK: tl.constexpr, MAX_LOG: tl.constexpr,
                    ACCURATE: tl.constexpr, PRECISE_DIV: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BV + tl.arange(0, BV)
    mv = rows < V
    rows = rows.to(tl.int64)
    acc_q = tl.zeros([BV], BIAS.dtype.element_ty)                      # sum_k mc^2 / s
    acc_l = tl.zeros([BV], BIAS.dtype.element_ty)                      # sum_k log s
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < K
        m2, idx, s, inv, mc, _ = _bank_tile(MU, L, C, EPS, rows, mv, offs_k, mk, K, MAX_LOG, ACCURATE, PRECISE_DIV)
        minv = mc * inv
        out = rows[:, None] * (2 * K) + offs_k[None, :]
        tl.store(RHS + out, inv, mask=m2)
        tl.store(RHS + out + K, minv, mask=m2)
        acc_q += tl.sum(tl.where(m2, mc * minv, 0.0), 1)
        if ACCURATE:
            log_s = libdevice.log(s)
        else:
            log_s = tl.log(s)
        acc_l += tl.sum(tl.where(m2, log_s, 0.0), 1)
    tl.store(BIAS + rows, acc_q + acc_l, mask=mv)


@triton.jit
def bank_bwd_kernel(MU, L, C, EPS, GRHS, GBIAS, GMU, GL, PART, V, K,
                    BV: tl.constexpr, BK: tl.constexpr, MAX_LOG: tl.constexpr,
                    ACCURATE: tl.constexpr, PRECISE_DIV: tl.constexpr):
    r"""g_mc = g2 inv + 2 gb mc inv, g_l = [passes] (gb - (g1 + g2 mc + gb mc^2) inv); PART[pid] = sum_rows g_mc."""
    pid = tl.program_id(0)
    rows = pid * BV + tl.arange(0, BV)
    mv = rows < V
    rows = rows.to(tl.int64)
    gb = tl.load(GBIAS + rows, mask=mv, other=0.0)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < K
        m2, idx, s, inv, mc, passes = _bank_tile(MU, L, C, EPS, rows, mv, offs_k, mk, K, MAX_LOG, ACCURATE,
                                                 PRECISE_DIV)
        out = rows[:, None] * (2 * K) + offs_k[None, :]
        g1 = tl.load(GRHS + out, mask=m2, other=0.0)
        g2 = tl.load(GRHS + out + K, mask=m2, other=0.0)
        g_inv = g1 + g2 * mc + gb[:, None] * (mc * mc)
        g_mc = g2 * inv + 2.0 * gb[:, None] * (mc * inv)
        g_l = tl.where(passes, gb[:, None] - g_inv * inv, 0.0)
        tl.store(GMU + idx, g_mc, mask=m2)
        tl.store(GL + idx, g_l, mask=m2)
        tl.store(PART + pid.to(tl.int64) * K + offs_k, tl.sum(tl.where(m2, g_mc, 0.0), 0), mask=mk)


def launch_config() -> Dict[str, int]:
    r"""Tile and launch parameters of both kernels: BV rows per program, BK coordinates per register tile.

    Both kernels stream the (V, K) tables once; a short row tile keeps about 3,000 programs in flight at V = 50,257,
    and BK = 64 covers K = 192 in three coalesced tiles. Not tuned on a GPU.
    """
    return dict(BV=16, BK=64, num_warps=4, num_stages=2)
