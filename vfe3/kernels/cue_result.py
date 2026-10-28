r"""Fused Triton route of the cue/result memory on diagonal records with a low-rank regression (``block_memory_triton``).

:func:`vfe3.model.cue_result_memory.cue_result_diagonal_posterior` computes, for record rows :math:`i` and components
:math:`a`, the exact result posterior of one switching linear-Gaussian step on a diagonal record and its diagonal
M-projection. The torch route materializes (rows, M, K) tensors per tile; this route runs the same formulas in Triton
register tiles (``_cue_result_triton.py``, whose docstring lists the per-element quantities), so the only (rows, M)
buffers are the log weights and a few planes per rank index. The torch route is unchanged and remains the reference.

Forward. ``cr_fwd_weights_kernel`` writes the unnormalized log weights :math:`\ell_{ia}` and the planes
:math:`t_{ia,r}` and :math:`G^{(j)}_{ia,rs}`, the Gram partial of each coordinate tile :math:`j` (``NKT`` tiles of the
kernels' common BK), from which the moment and backward kernels assemble the leave-one-out Gram
:math:`G^{(-k)} = \sum_{l\ne k}p_lV_lV_l^\top` without subtraction (``_cue_result_triton.py``, audit 2026-10-28 F01);
torch normalizes, :math:`\log w_{ia} = \ell_{ia} - \operatorname{LSE}_b\ell_{ib}`;
``cr_fwd_moments_kernel`` returns :math:`\bar m_{ik} = \sum_a w_{ia}m^Y_{iak}` and
:math:`\bar v_{ik} = \sum_a w_{ia}[v^Y_{iak} + (m^Y_{iak} - \bar m_{ik})^2]` by a weighted Chan merge over component
tiles, so the variance is a sum of non-negative terms with no cancellation.

The log weights are formed relative to the row maximum, :math:`(\ell_{ia} - s_i) - \log l_i`, never by subtracting
the absolute log evidence (audit 2026-10-25 F01).

Backward, for incoming :math:`\bar g^m, \bar g^v` (rows, K), :math:`\bar g^{\log w}` (rows, M) and
:math:`\bar g^{\log Z}` (rows,): with :math:`\bar m = w(\bar g^m + 2\bar g^v(m^Y - \bar m_{ik}))` and
:math:`\bar v = w\bar g^v` per element,

.. math::
    \bar w_{ia} = \sum_k \bar g^m_{ik}m^Y + \bar g^v_{ik}(v^Y + (m^Y - \bar m_{ik})^2),\quad
    \bar t_{ia,r} = \sum_k \bar m\,U_{akr},\quad \bar G_{ia,rs} = \sum_k \bar v\,U_{akr}U_{aks}

(``cr_bwd_pairs_kernel``, over t and Gram partial planes that the weights kernel rewrites for the backward's rows),
then in torch the log-weight adjoint

.. math::
    \bar\ell_{ia} = w_{ia}\Big(\bar w_{ia} - \sum_b w_{ib}\bar w_{ib}\Big) + \bar g^{\log w}_{ia}
                    - w_{ia}\sum_b\bar g^{\log w}_{ib} + w_{ia}\bar g^{\log Z}_i,

which is also :math:`\partial L/\partial\log\hat\pi_{ia}`. With :math:`a = \sum_rU_rV_r` and the indicator
:math:`\iota = [U^\top G^{(-k)}U \ge 0]` of the unclamped off-axis sum of
:math:`v^Y = p(1 + a)^2 + \max(U^\top G^{(-k)}U, 0) + q` (``_cue_result_triton.py``), and
:math:`\bar G_{ia,rs} = \sum_k\iota\,\bar v\,U_{akr}U_{aks}`, per element :math:`\bar g = \bar m + \sum_r\bar t_rV_r`,
:math:`\bar p = \bar v((1 + a)^2 - \iota a^2) + \sum_{rs}\bar G_{rs}V_rV_s`,
:math:`\bar\delta = \bar g\,\mathrm{gain} - \bar\ell\,\delta/W`,
:math:`\overline{\mathrm{gain}} = \bar g\,\delta + \bar p\,c`,
:math:`\bar W = -\overline{\mathrm{gain}}\,\lambda/W^2 - \tfrac12\bar\ell(1/W - \delta^2/W^2)`, and

.. math::
    \bar\xi = \sum_a\bar\delta,\quad \bar c = \sum_a(\bar W + \bar p\,\mathrm{gain}),\quad
    \bar\nu = -\sum_i\bar\delta,\quad \bar\lambda = \sum_i(\overline{\mathrm{gain}}/W + \bar W),\quad
    \bar\omega = \sum_i\bar m,\quad \bar q = \sum_i\bar v,

    \bar U_r = \sum_i\Big[\bar m\,t_r + \bar v\Big(2p(1 + a)V_r + 2\iota\sum_sG^{(-k)}_{rs}U_s\Big)\Big],\quad
    \bar V_r = \sum_i\Big[g\,\bar t_r + 2p\Big(\bar v(1 + a - \iota a)U_r + \sum_s\bar G_{rs}V_s\Big)\Big]

The :math:`\bar p` and :math:`\bar V` forms are the leave-one-out adjoints written with the full :math:`\bar G` minus
the coordinate's own term :math:`\iota\bar vU_rU_s`; their rounding error is relative to the self terms they contain,
not to the variance. (``cr_bwd_rows_kernel`` owns :math:`\bar\xi, \bar c`; ``cr_bwd_cols_kernel`` owns the parameter sums, one partial per
row split, summed in torch). Rows are processed in chunks of ``row_chunk`` so the planes stay bounded. The kernels use
no atomics, so results are bitwise reproducible for a fixed GPU, shape and dtype. ``triton`` is imported only when
this route launches; CPU tensors run the torch route.
"""
from typing import Any, Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

_KERNELS: Any = None
MAX_RANK = 4                                     # ranks above this run the torch route's register budget out


def _kernels() -> Any:
    r"""Import the Triton kernel module once; fail loudly rather than fall back."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from vfe3.kernels import _cue_result_triton as kernels
        except ImportError as exc:
            raise RuntimeError(
                "block_memory_triton=True on CUDA tensors runs the fused Triton cue/result kernels, which need the "
                f"triton package (triton-windows on Windows), and importing it failed: {exc}. Install triton, or set "
                "block_memory_triton=False.") from exc
        _KERNELS = kernels
    return _KERNELS


def fused_route_active(xi: torch.Tensor) -> bool:
    r"""Whether the fused route runs for these tensors: CUDA only (the interpreter tests replace this)."""
    return xi.is_cuda


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _rank_pad(rank: int) -> int:
    return 1 << max(0, (rank - 1).bit_length())


def _tile_k() -> int:
    r"""The coordinate tile width BK shared by every kernel (the Gram partials are stored per tile of this width)."""
    widths = {cfg["BK"] for cfg in _kernels().launch_config().values()}
    if len(widths) != 1:
        raise RuntimeError(f"the fused cue/result kernels must share one BK, got {sorted(widths)}")
    return widths.pop()


def plane_count(rank: int, K: int, bk: int = 32) -> int:
    r"""(rows, M)-sized buffers per row the route holds at once: in the backward, t and tbar (RP each), the Gram
    partials (RP^2 per coordinate tile of width ``bk``) and Gbar (RP^2), with a factor two of slack, plus the log
    weights, their scratch and wbar."""
    RP = _rank_pad(rank)
    return 2 * (2 * RP + RP * RP * (_cdiv(K, bk) + 1)) + 3


class CueResultDiagonalLowRank(torch.autograd.Function):
    r"""The fused exact posterior on diagonal records with :math:`F_a = I + U_aV_a^\top`; see the module docstring.

    The backward is ``once_differentiable``: its planes are Triton writes autograd does not track.
    """

    @staticmethod
    def forward(
        ctx:       Any,
        x:         torch.Tensor,             # (R, K) record values xi
        c:         torch.Tensor,             # (R, K) record variances
        nu:        torch.Tensor,             # (M, K) cue means
        omega:     torch.Tensor,             # (M, K) result means
        lam:       torch.Tensor,             # (M, K) cue variances lambda
        q:         torch.Tensor,             # (M, K) result noise variances
        lp:        torch.Tensor,             # (M,) or (R, M) normalized log prior
        U:         torch.Tensor,             # (M, K, rank)
        V:         torch.Tensor,             # (M, K, rank)
        row_chunk: int,                      # rows per launch group
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tk = _kernels()
        R, K = x.shape
        M, rank = nu.shape[0], U.shape[-1]
        RP = _rank_pad(rank)
        per_token = lp.dim() == 2
        dtype, device = x.dtype, x.device
        acc = not tk.INTERPRETED
        cfg = tk.launch_config()
        NKT = _cdiv(K, _tile_k())
        half_k_log2pi = 0.5 * K * float(torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64)))
        mean = torch.empty(R, K, dtype=dtype, device=device)
        var = torch.empty(R, K, dtype=dtype, device=device)
        logwn = torch.empty(R, M, dtype=dtype, device=device)
        log_ev = torch.empty(R, dtype=dtype, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            for r0 in range(0, R, row_chunk):
                r1 = min(r0 + row_chunk, R)
                Rc = r1 - r0
                xs, cs = x[r0:r1], c[r0:r1]
                lps = lp[r0:r1] if per_token else lp
                # Every plane element a reader consumes is written first (readers mask the padded rank planes), so
                # no initialization (docs/perf-investigation_2026-10-28.md, rank 3).
                T = torch.empty(RP, Rc, M, dtype=dtype, device=device)          # (RP, Rc, M) t planes
                GP = torch.empty(RP * RP * NKT, Rc, M, dtype=dtype, device=device)   # (RP^2 NKT, Rc, M) Gram partials
                logw = torch.empty(Rc, M, dtype=dtype, device=device)
                f = cfg["fwd_weights"]
                tk.cr_fwd_weights_kernel[(Rc, _cdiv(M, f["BA"]))](
                    xs, cs, nu, lam, V, lps, logw, T, GP, Rc, M, K, NKT, M if per_token else 0, half_k_log2pi,
                    BA=f["BA"], BK=f["BK"], RANK=rank, RP=RP, ACC=acc, WRITE_LOGW=True,
                    num_warps=f["num_warps"], num_stages=f["num_stages"])
                # Shift and scaled mass kept apart: subtracting the absolute log evidence rounds its log l increment
                # away at large log densities (audit 2026-10-25 F01).
                shift = logw.amax(dim=-1)
                shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
                logw.sub_(shift.unsqueeze(-1))                                  # l - s, formed once (rank 15)
                log_l = torch.log(torch.exp(logw).sum(-1))                                  # (Rc,)
                torch.sub(logw, log_l.unsqueeze(-1), out=logwn[r0:r1])
                log_ev[r0:r1] = shift + log_l
                f = cfg["fwd_moments"]
                tk.cr_fwd_moments_kernel[(Rc, _cdiv(K, f["BK"]))](
                    xs, cs, nu, lam, omega, q, U, V, logwn[r0:r1], T, GP, mean[r0:r1], var[r0:r1], Rc, M, K, NKT,
                    BA=f["BA"], BK=f["BK"], RANK=rank, RP=RP, ACC=acc,
                    num_warps=f["num_warps"], num_stages=f["num_stages"])
        ctx.save_for_backward(x, c, nu, omega, lam, q, lp, U, V, logwn, mean)
        # Unused outputs (the log weights and evidence without a carried regime) keep None gradients rather than
        # zero tensors, and the backward skips their terms (docs/perf-investigation_2026-10-28.md, rank 11).
        ctx.set_materialize_grads(False)
        ctx.row_chunk = row_chunk
        ctx.half_k_log2pi = half_k_log2pi
        return mean, var, logwn, log_ev

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, g_mean: Optional[torch.Tensor], g_var: Optional[torch.Tensor],
                 g_logw: Optional[torch.Tensor], g_lev: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        tk = _kernels()
        x, c, nu, omega, lam, q, lp, U, V, logwn, mean = ctx.saved_tensors
        R, K = x.shape
        M, rank = nu.shape[0], U.shape[-1]
        RP = _rank_pad(rank)
        per_token = lp.dim() == 2
        dtype, device = x.dtype, x.device
        acc = not tk.INTERPRETED
        cfg = tk.launch_config()
        NKT = _cdiv(K, _tile_k())
        zeros = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)       # noqa: E731
        g_mean = zeros(R, K) if g_mean is None else g_mean.contiguous()
        g_var = zeros(R, K) if g_var is None else g_var.contiguous()
        g_logw = None if g_logw is None else g_logw.contiguous()
        g_lev = None if g_lev is None else g_lev.contiguous()
        dx, dc = torch.empty_like(x), torch.empty_like(c)
        dnu, dlam, domega, dq = zeros(M, K), zeros(M, K), zeros(M, K), zeros(M, K)
        dU, dV = zeros(M, K, rank), zeros(M, K, rank)
        dlp = torch.zeros_like(lp)
        with torch.autocast(device_type=device.type, enabled=False):
            for r0 in range(0, R, ctx.row_chunk):
                r1 = min(r0 + ctx.row_chunk, R)
                Rc = r1 - r0
                args_rows = (x[r0:r1], c[r0:r1], nu, lam, omega, q, U, V, logwn[r0:r1], mean[r0:r1],
                             g_mean[r0:r1], g_var[r0:r1])
                empty = lambda *shape: torch.empty(*shape, dtype=dtype, device=device)   # noqa: E731, rank 3
                T, GP = empty(RP, Rc, M), empty(RP * RP * NKT, Rc, M)
                TB, GB = empty(RP, Rc, M), empty(RP * RP, Rc, M)
                wbar = torch.empty(Rc, M, dtype=dtype, device=device)
                # The weights kernel rewrites the t and Gram partial planes of these rows; it writes no log weights
                # (the saved normalized ones are used; rank 13), so LOGW is an unused pointer.
                f = cfg["fwd_weights"]
                tk.cr_fwd_weights_kernel[(Rc, _cdiv(M, f["BA"]))](
                    x[r0:r1], c[r0:r1], nu, lam, V, lp[r0:r1] if per_token else lp, wbar, T, GP, Rc, M, K, NKT,
                    M if per_token else 0, ctx.half_k_log2pi,
                    BA=f["BA"], BK=f["BK"], RANK=rank, RP=RP, ACC=acc, WRITE_LOGW=False,
                    num_warps=f["num_warps"], num_stages=f["num_stages"])
                f = cfg["bwd_pairs"]
                tk.cr_bwd_pairs_kernel[(Rc, _cdiv(M, f["BA"]))](
                    *args_rows, T, GP, wbar, TB, GB, Rc, M, K, NKT,
                    BA=f["BA"], BK=f["BK"], RANK=rank, RP=RP, ACC=acc,
                    num_warps=f["num_warps"], num_stages=f["num_stages"])
                w = logwn[r0:r1].exp()                                           # (Rc, M)
                lbar = w * (wbar - (w * wbar).sum(-1, keepdim=True))
                # An absent gradient's terms are zeros; adding +0.0 where they were added keeps the sign of zero of
                # the materialized expression ((A + g) - w sum g) + w g_lev bit for bit.
                if g_logw is not None:
                    glw = g_logw[r0:r1]
                    lbar = lbar + glw - w * glw.sum(-1, keepdim=True)
                else:
                    lbar = lbar + 0.0
                if g_lev is not None:
                    lbar = lbar + w * g_lev[r0:r1].unsqueeze(-1)
                elif g_logw is not None:
                    lbar = lbar + 0.0
                lbar = lbar.contiguous()
                if per_token:
                    dlp[r0:r1] = lbar
                else:
                    dlp += lbar.sum(0)
                f = cfg["bwd_rows"]
                tk.cr_bwd_rows_kernel[(Rc, _cdiv(K, f["BK"]))](
                    *args_rows, T, GP, TB, GB, lbar, dx[r0:r1], dc[r0:r1], Rc, M, K, NKT,
                    BA=f["BA"], BK=f["BK"], RANK=rank, RP=RP, ACC=acc,
                    num_warps=f["num_warps"], num_stages=f["num_stages"])
                f = cfg["bwd_cols"]
                splits = max(1, min(32, _cdiv(Rc, 256)))
                per_split = _cdiv(Rc, splits)
                # Every split stores every (a, k) of its partials, an empty split its zero accumulators (rank 3).
                parts = [empty(splits, M, K) for _ in range(4)] + [empty(splits, M, K, rank) for _ in range(2)]
                tk.cr_bwd_cols_kernel[(_cdiv(M, f["BA"]), _cdiv(K, f["BK"]), splits)](
                    *args_rows, T, GP, TB, GB, lbar, *parts, Rc, M, K, NKT, per_split,
                    BA=f["BA"], BK=f["BK"], RANK=rank, RP=RP, ACC=acc,
                    num_warps=f["num_warps"], num_stages=f["num_stages"])
                for total, part in zip((dnu, dlam, domega, dq, dU, dV), parts):
                    total += part.sum(0)
        return dx, dc, dnu, domega, dlam, dq, dlp, dU, dV, None


def cue_result_diagonal_lowrank_fused(
    x:         torch.Tensor,                 # (R, K) record values
    c:         torch.Tensor,                 # (R, K) record variances
    nu:        torch.Tensor,                 # (M, K)
    omega:     torch.Tensor,                 # (M, K)
    lam:       torch.Tensor,                 # (M, K) cue variances
    q:         torch.Tensor,                 # (M, K) result noise variances
    lp:        torch.Tensor,                 # (M,) or (R, M) normalized log prior
    U:         torch.Tensor,                 # (M, K, rank)
    V:         torch.Tensor,                 # (M, K, rank)
    *,
    row_chunk: int,                          # rows per launch group (bounds the (P, rows, M) planes)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:   # mean (R, K), var (R, K), log w (R, M), log Z (R,)
    r"""Validate the operands and run the fused forward/backward (module docstring). One float dtype (float32 or
    float64); CUDA tensors, or CPU tensors under ``TRITON_INTERPRET=1`` (the test route)."""
    tk = _kernels()
    R, K = x.shape
    M, rank = nu.shape[0], U.shape[-1]
    if not 1 <= rank <= MAX_RANK:
        raise ValueError(f"the fused cue/result route takes regression ranks 1..{MAX_RANK}, got {rank}")
    shapes_ok = (c.shape == (R, K) and nu.shape == omega.shape == lam.shape == q.shape == (M, K)
                 and U.shape == V.shape == (M, K, rank) and lp.shape in ((M,), (R, M)))
    if not shapes_ok:
        raise ValueError("inconsistent fused cue/result operands")
    floats = (x, c, nu, omega, lam, q, lp, U, V)
    if x.dtype not in (torch.float32, torch.float64) or any(t.dtype != x.dtype for t in floats):
        raise TypeError(f"the fused cue/result route needs one float32 or float64 dtype; got {[t.dtype for t in floats]}")
    devices = {t.device for t in floats}
    if len(devices) != 1 or (x.device.type != "cuda" and not tk.INTERPRETED):
        raise ValueError(f"the fused cue/result route needs all operands on one CUDA device; got {devices}")
    return CueResultDiagonalLowRank.apply(*(t.contiguous() for t in floats), max(1, int(row_chunk)))
