r"""Fused Triton diagonal comparison moments (config ``relational_diag_moments_kernel = "triton"``).

On the diagonal comparison route (``diagonal_only`` of :func:`vfe3.relational_scoring.projected_moments`) the
scorer reads only the diagonal of the projected covariance :math:`G_i = X_i X_i^\top`, where the source root of a
diagonal covariance is

.. math::
    X_i = S U_i^{-1} \operatorname{diag}(\sigma_i)^{1/2}, \qquad
    g_{i,s} = (G_i)_{ss} = \sum_k X_{i,sk}^2 .

With the stacked maps :math:`S` of shape (ROWS, K), ROWS = R H r (roles, heads, comparison rank), the inverse frame
in M compact blocks :math:`U_{t,m}` of width d (K = M d) and the head-shared root
:math:`\rho_t = \sigma_t^{1/2}` (the zero-safe root of :func:`vfe3.relational_scoring._diagonal_root_moments`),

.. math::
    g_{t,s} = \sum_m \sum_e \Big( \rho_{t,(m,e)} \sum_c S_{s,(m,c)} U_{t,m,c,e} \Big)^2 .

The torch route builds X, of shape (T, ROWS, K), tile by tile in checkpointed token tiles and rebuilds it in
backward. The kernels keep X in registers and never store it (``_diagonal_moments_triton.py``). The source-root
formula is kept: the quadratic form :math:`S U^{-1} \operatorname{diag}(\sigma) U^{-\top} S^\top` would square the
conditioning of the source root. With :math:`Y = S U` (per block), :math:`\bar X = 2 \bar g X` and
:math:`\bar Y = \bar X \rho`, the hand-written backward is

.. math::
    \bar U_{t,m} = \sum_s S_{s,(m,\cdot)}^\top \bar Y_{s,(t,m,\cdot)}, \qquad
    \bar\rho_{t,(m,e)} = \sum_s \bar X_{s,(t,m,e)} Y_{s,(t,m,e)}, \qquad
    \bar S_{s,(m,\cdot)} = \sum_t \bar Y_{s,(t,m,\cdot)} U_{t,m}^\top ,

and recomputes Y from S and U in each of its two kernels instead of saving it. Only S, U and the root are saved.
The gradient reaches :math:`\sigma` through the torch root (zero gradient at a zero variance, as on the torch
route). All dots are IEEE fp32 (or fp64) with fp32 (fp64) accumulation; the differences from the torch route are
the summation order over k (block by block here, one reduction there) and of the dS token sum (per-split
partial sums added by a deterministic torch reduction). The kernels use no atomics, so results are bitwise
reproducible run to run for a fixed GPU, shape and dtype. ``triton`` is imported only when this route launches.
"""
from typing import Any, Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

_KERNELS: Any = None

# Largest frame block width the kernels take; a wider block falls back to the torch route (its (BD, 128) frame
# tile and dU accumulator would spill).
MAX_BLOCK_WIDTH = 64


def _kernels() -> Any:
    r"""Import the Triton kernel module once; fail loudly rather than fall back."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from vfe3.kernels import _diagonal_moments_triton as kernels
        except ImportError as exc:
            raise RuntimeError(
                "relational_diag_moments_kernel='triton' on CUDA tensors runs the fused Triton diagonal moments, "
                f"which need the triton package (triton-windows on Windows), and importing it failed: {exc}. "
                "Install triton, or set relational_diag_moments_kernel='torch'.") from exc
        _KERNELS = kernels
    return _KERNELS


def _map_splits(tk: Any, device: torch.device, n_tiles: int, n_groups: int) -> int:
    r"""Token splits of the dS kernel: about four programs per SM over the (row tile, block) grid, at most one split
    per token group. A fixed function of the device and shape, so the partial-sum order is reproducible."""
    if tk.INTERPRETED or device.type != "cuda":
        target = 3                                                       # several splits, so the tests sum them
    else:
        target = -(-4 * torch.cuda.get_device_properties(device).multi_processor_count // max(1, n_tiles))
    return max(1, min(n_groups, target))


class DiagonalRootSquareSums(torch.autograd.Function):
    r"""g = sum over k of X^2 for X = S U^{-1} diag(rho) per token; see the module docstring.

    The backward is ``once_differentiable``: it is a set of Triton writes autograd does not track, so a
    ``create_graph=True`` gradient raises instead of returning a truncated second derivative.
    """

    @staticmethod
    def forward(
        ctx:  Any,
        S:    torch.Tensor,  # (ROWS, K) stacked comparison maps, rows (R, H, r)
        U:    torch.Tensor,  # (T, M, d, d) inverse frame blocks, K = M d
        root: torch.Tensor,  # (T, K) head-shared source roots rho = sigma^{1/2}
    ) -> torch.Tensor:       # (T, ROWS) g
        tk = _kernels()
        T, M, d, _ = U.shape
        ROWS, K = S.shape
        cfg = tk.launch_config(d, S.dtype == torch.float64)
        g = torch.empty(T, ROWS, dtype=S.dtype, device=S.device)
        if T > 0 and ROWS > 0:
            grid = (-(-T // cfg["TB"]), -(-ROWS // cfg["BR_FWD"]))
            tk.diag_moments_fwd_kernel[grid](
                S, U, root, g, T, ROWS, K, M, d, BR=cfg["BR_FWD"], BD=cfg["BD"], TB=cfg["TB"],
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        ctx.save_for_backward(S, U, root)
        return g

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_g: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        tk = _kernels()
        S, U, root = ctx.saved_tensors
        need_S, need_U, need_root = ctx.needs_input_grad[:3]
        T, M, d, _ = U.shape
        ROWS, K = S.shape
        cfg = tk.launch_config(d, S.dtype == torch.float64)
        grad_g = grad_g.contiguous()                                      # (T, ROWS)
        dS = dU = droot = None
        if need_U:
            dU = torch.zeros_like(U) if T == 0 or ROWS == 0 else torch.empty_like(U)       # (T, M, d, d)
        if need_root:
            droot = torch.zeros_like(root) if T == 0 or ROWS == 0 else torch.empty_like(root)  # (T, K)
        if (need_U or need_root) and T > 0 and ROWS > 0:
            tk.diag_moments_bwd_tok_kernel[(-(-T // cfg["TB"]),)](
                S, U, root, grad_g, dU if need_U else U, droot if need_root else root, T, ROWS, K, M, d,
                BR=cfg["BR_TOK"], BD=cfg["BD"], TB=cfg["TB"], NEED_U=bool(need_U), NEED_ROOT=bool(need_root),
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        if need_S:
            if T == 0 or ROWS == 0:
                dS = torch.zeros_like(S)
            else:
                row_tiles = -(-ROWS // cfg["BR_MAPS"])
                n_groups = -(-T // cfg["TB"])
                splits = _map_splits(tk, S.device, row_tiles * M, n_groups)
                per_split = -(-n_groups // splits)
                splits = -(-n_groups // per_split)                       # no empty trailing split
                partial = torch.empty(splits, ROWS, K, dtype=S.dtype, device=S.device)   # (P, ROWS, K)
                tk.diag_moments_bwd_maps_kernel[(row_tiles, M, splits)](
                    S, U, root, grad_g, partial, T, ROWS, K, M, d, per_split,
                    BR=cfg["BR_MAPS"], BD=cfg["BD"], TB=cfg["TB"],
                    num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
                dS = partial.sum(0)
        return dS, dU, droot


def diagonal_root_square_sums(
    S:    torch.Tensor,  # (ROWS, K) stacked comparison maps
    U:    torch.Tensor,  # (T, M, d, d) inverse frame blocks, K = M d
    root: torch.Tensor,  # (T, K) head-shared roots
) -> torch.Tensor:       # (T, ROWS)
    r"""Validate the operands and run the fused forward/backward (see the module docstring).

    One float dtype (float32 or float64) across the operands. CUDA tensors only, except under ``TRITON_INTERPRET=1``
    where the Triton interpreter runs on CPU tensors (the test route).
    """
    tk = _kernels()
    if S.ndim != 2 or U.ndim != 4 or root.ndim != 2:
        raise ValueError(f"fused diagonal moments need S (ROWS, K), U (T, M, d, d), root (T, K); got "
                         f"{tuple(S.shape)}, {tuple(U.shape)}, {tuple(root.shape)}")
    T, M, d, d2 = U.shape
    K = S.shape[1]
    if d != d2 or M * d != K or root.shape != (T, K):
        raise ValueError(f"inconsistent fused diagonal moment operands: S {tuple(S.shape)}, U {tuple(U.shape)}, "
                         f"root {tuple(root.shape)}")
    if d > MAX_BLOCK_WIDTH:
        raise ValueError(f"fused diagonal moments take frame blocks of width <= {MAX_BLOCK_WIDTH}, got {d}")
    if S.dtype not in (torch.float32, torch.float64) or U.dtype != S.dtype or root.dtype != S.dtype:
        raise TypeError(f"fused diagonal moments need one float32 or float64 dtype; got "
                        f"{S.dtype}, {U.dtype}, {root.dtype}")
    devices = {S.device, U.device, root.device}
    if len(devices) != 1 or (S.device.type != "cuda" and not tk.INTERPRETED):
        raise ValueError(f"fused diagonal moments need all operands on one CUDA device; got {devices}")
    return DiagonalRootSquareSums.apply(S.contiguous(), U.contiguous(), root.contiguous())
