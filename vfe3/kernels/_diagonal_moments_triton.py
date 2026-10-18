r"""Triton kernels for the fused diagonal comparison moments (see ``diagonal_moments.py``).

This module imports ``triton`` at import time, so only ``diagonal_moments.py`` imports it, and only when the fused
route launches. Every kernel views one block :math:`m` of the frame as a GEMM whose column axis runs over TB tokens
at once,

.. math::
    Y^{(m)}_{s,(t,e)} = \sum_c S_{s,(m,c)} U_{t,m,c,e}, \qquad X^{(m)}_{s,(t,e)} = Y^{(m)}_{s,(t,e)} \rho_{t,(m,e)},

so a (BR, BD) tile of the stacked comparison maps S multiplies a (BD, TB BD) tile of TB consecutive tokens' inverse
frame blocks, and the source root X lives only as that (BR, TB BD) register tile. BD is the frame block width d
padded to a power of two of at least 16 (the masked rows and columns load as zeros and contribute nothing).

Kernels:
  ``diag_moments_fwd_kernel``  program = (TB tokens, BR rows); g_{t,s} = sum_m sum_e X^2 over the M blocks.
  ``diag_moments_bwd_tok_kernel``  program = TB tokens; per block m, loops over all rows and writes the owned
                               dU_{t,m} and drho_{t,(m,.)} (no atomics: the row sum stays in the program).
  ``diag_moments_bwd_maps_kernel``  program = (BR rows, block m, token split); loops over the split's tokens and
                               writes one partial dS tile per split; the launcher sums the splits in torch.

The dots run at ``input_precision="ieee"`` (no TF32), the precision the torch route pins with ``ieee_fp32_matmul``.
"""
from typing import Dict

import triton
import triton.language as tl

# Captured at import, the moment ``triton.jit`` below decides between compiling and interpreting.
INTERPRETED: bool = bool(triton.knobs.runtime.interpret)


@triton.jit
def _load_maps(S, rows, row_ok, m, offs_c, c_ok, K, d):
    r"""The (BR, BD) tile S[rows, m d + c] of the row-major (ROWS, K) stacked maps."""
    ptr = S + rows.to(tl.int64)[:, None] * K + (m * d + offs_c)[None, :]
    return tl.load(ptr, mask=row_ok[:, None] & c_ok[None, :], other=0.0)


@triton.jit
def _load_frames(U, t0, m, offs_c, c_ok, cols, T, M, d, BD: tl.constexpr):
    r"""The (BD, TB BD) tile U[t0 + col // BD, m, c, col % BD] of the (T, M, d, d) inverse frame blocks."""
    tok = t0 + cols // BD
    e = cols % BD
    col_ok = (tok < T) & (e < d)
    base = (tok.to(tl.int64) * M + m) * d * d + e                                    # (TB BD,)
    ptr = U + base[None, :] + offs_c.to(tl.int64)[:, None] * d
    return tl.load(ptr, mask=c_ok[:, None] & col_ok[None, :], other=0.0)


@triton.jit
def _load_roots(ROOT, t0, m, cols, T, K, d, BD: tl.constexpr):
    r"""The (TB BD,) root vector rho[t0 + col // BD, m d + col % BD] of the (T, K) head-shared roots."""
    tok = t0 + cols // BD
    e = cols % BD
    col_ok = (tok < T) & (e < d)
    return tl.load(ROOT + tok.to(tl.int64) * K + m * d + e, mask=col_ok, other=0.0), tok, e, col_ok


@triton.jit
def diag_moments_fwd_kernel(S, U, ROOT, G, T, ROWS, K, M, d,
                            BR: tl.constexpr, BD: tl.constexpr, TB: tl.constexpr):
    r"""g_{t,s} = \sum_m \sum_e (Y^{(m)}_{s,(t,e)} \rho_{t,(m,e)})^2 for TB tokens and BR rows."""
    t0 = tl.program_id(0) * TB
    rows = tl.program_id(1) * BR + tl.arange(0, BR)
    row_ok = rows < ROWS
    offs_c = tl.arange(0, BD)
    c_ok = offs_c < d
    cols = tl.arange(0, TB * BD)
    acc = tl.zeros([BR, TB], S.dtype.element_ty)                                    # (BR, TB) partial g
    for m in range(0, M):
        s_tile = _load_maps(S, rows, row_ok, m, offs_c, c_ok, K, d)                 # (BR, BD)
        u_tile = _load_frames(U, t0, m, offs_c, c_ok, cols, T, M, d, BD)            # (BD, TB BD)
        root, _, _, _ = _load_roots(ROOT, t0, m, cols, T, K, d, BD)                 # (TB BD,)
        x = tl.dot(s_tile, u_tile, input_precision="ieee") * root[None, :]          # (BR, TB BD) source root
        acc += tl.sum(tl.reshape(x * x, (BR, TB, BD)), 2)
    toks = t0 + tl.arange(0, TB)
    ptr = G + toks.to(tl.int64)[None, :] * ROWS + rows[:, None]
    tl.store(ptr, acc, mask=row_ok[:, None] & (toks < T)[None, :])


@triton.jit
def diag_moments_bwd_tok_kernel(S, U, ROOT, GBAR, DU, DROOT, T, ROWS, K, M, d,
                                BR: tl.constexpr, BD: tl.constexpr, TB: tl.constexpr,
                                NEED_U: tl.constexpr, NEED_ROOT: tl.constexpr):
    r"""dU_{t,m} = \sum_s S_{s,(m,\cdot)}^\top \bar Y_s and d\rho_{t,(m,e)} = \sum_s \bar X_{s,(t,e)} Y_{s,(t,e)}.

    With :math:`\bar X = 2 \bar g X` and :math:`\bar Y = \bar X \rho`, for TB tokens; every row is visited inside
    the program, so both outputs have one owner.
    """
    t0 = tl.program_id(0) * TB
    offs_c = tl.arange(0, BD)
    c_ok = offs_c < d
    cols = tl.arange(0, TB * BD)
    dt = S.dtype.element_ty
    for m in range(0, M):
        u_tile = _load_frames(U, t0, m, offs_c, c_ok, cols, T, M, d, BD)            # (BD, TB BD)
        root, tok, e, col_ok = _load_roots(ROOT, t0, m, cols, T, K, d, BD)          # (TB BD,)
        acc_u = tl.zeros([BD, TB * BD], dt)                                          # (BD, TB BD) dU tile
        acc_r = tl.zeros([TB * BD], dt)                                              # (TB BD,) drho
        for r0 in range(0, ROWS, BR):
            rows = r0 + tl.arange(0, BR)
            row_ok = rows < ROWS
            s_tile = _load_maps(S, rows, row_ok, m, offs_c, c_ok, K, d)             # (BR, BD)
            y = tl.dot(s_tile, u_tile, input_precision="ieee")                      # (BR, TB BD)
            gbar = tl.load(GBAR + tok.to(tl.int64)[None, :] * ROWS + rows[:, None],
                           mask=row_ok[:, None] & col_ok[None, :], other=0.0)       # (BR, TB BD) gbar_{t,s}
            x_bar = 2.0 * gbar * (y * root[None, :])                                 # dg/dX
            if NEED_ROOT:
                acc_r += tl.sum(x_bar * y, 0)
            if NEED_U:
                acc_u += tl.dot(tl.trans(s_tile), x_bar * root[None, :], input_precision="ieee")
        if NEED_ROOT:
            tl.store(DROOT + tok.to(tl.int64) * K + m * d + e, acc_r, mask=col_ok)
        if NEED_U:
            base = (tok.to(tl.int64) * M + m) * d * d + e
            tl.store(DU + base[None, :] + offs_c.to(tl.int64)[:, None] * d, acc_u,
                     mask=c_ok[:, None] & col_ok[None, :])


@triton.jit
def diag_moments_bwd_maps_kernel(S, U, ROOT, GBAR, DS_PART, T, ROWS, K, M, d, GROUPS_PER_SPLIT,
                                 BR: tl.constexpr, BD: tl.constexpr, TB: tl.constexpr):
    r"""Partial dS_{s,(m,c)} = \sum_t \sum_e \bar Y_{s,(t,e)} U_{t,m,c,e} over one split of token groups."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    row_ok = rows < ROWS
    m = tl.program_id(1)
    split = tl.program_id(2)
    offs_c = tl.arange(0, BD)
    c_ok = offs_c < d
    cols = tl.arange(0, TB * BD)
    s_tile = _load_maps(S, rows, row_ok, m, offs_c, c_ok, K, d)                     # (BR, BD)
    acc = tl.zeros([BR, BD], S.dtype.element_ty)                                     # (BR, BD) dS tile
    first = split * GROUPS_PER_SPLIT
    for group in range(first, first + GROUPS_PER_SPLIT):
        t0 = group * TB
        u_tile = _load_frames(U, t0, m, offs_c, c_ok, cols, T, M, d, BD)            # (BD, TB BD), zero past T
        root, tok, _, col_ok = _load_roots(ROOT, t0, m, cols, T, K, d, BD)
        y = tl.dot(s_tile, u_tile, input_precision="ieee")                          # (BR, TB BD)
        gbar = tl.load(GBAR + tok.to(tl.int64)[None, :] * ROWS + rows[:, None],
                       mask=row_ok[:, None] & col_ok[None, :], other=0.0)
        y_bar = 2.0 * gbar * (y * root[None, :]) * root[None, :]                     # dg/dY
        acc += tl.dot(y_bar, tl.trans(u_tile), input_precision="ieee")
    ptr = (DS_PART + split.to(tl.int64) * ROWS * K + rows.to(tl.int64)[:, None] * K
           + (m * d + offs_c)[None, :])
    tl.store(ptr, acc, mask=row_ok[:, None] & c_ok[None, :])


def launch_config(d: int, fp64: bool) -> Dict[str, int]:
    r"""Tile and launch parameters at frame block width ``d`` and the operand dtype (shared by the launcher and the
    sm_120 compile test).

    BD pads d to a power of two of at least 16 (the smallest ``tl.dot`` operand edge) and TB packs TB = COLS / BD
    tokens into the column axis, so a frame tile is (BD, COLS): COLS = 128 in float32 up to BD = 32, and 64 for
    float64 or BD = 64. The forward and the dS kernel carry one (BR, COLS) source-root tile; the token kernel also
    carries the (BD, COLS) dU accumulator, hence its shorter row tile. These are the tiles that compiled for sm_120a
    with zero spill bytes at d = 4 to 64 in both dtypes (tests/test_diag_moments_triton_20261018.py).
    """
    BD = max(16, triton.next_power_of_2(d))
    cols = 128 if not fp64 and BD <= 32 else 64
    wide64 = fp64 and BD >= 64
    return dict(BD=BD, TB=max(1, cols // BD), BR_FWD=16 if wide64 else 32, BR_TOK=16,
                BR_MAPS=16 if wide64 else 32, num_warps=8, num_stages=1)
