r"""Fused Triton elementwise work of the chunked KL-to-prior decode cross-entropy (``decode_ce_triton``).

``decode_mode='diagonal_chunked'`` and ``'full_chunked'`` score packed rows i against every vocabulary entry v as

.. math::
    a_{iv} = \mathrm{lhs}_i \cdot \mathrm{rhs}_v + b_v, \qquad
    \ell_{iv} = -\frac{\max\big(\tfrac12 (a_{iv} - c_i), 0\big)}{\tau} + u_v, \qquad
    \mathrm{CE}_i = \mathrm{LSE}_v(\ell_{iv}) - \ell_{i t_i},

with the expanded-form factors of ``vfe3.model.prior_bank._decode_av`` (lhs = [sq + mc_q^2, -2 mc_q],
rhs = [1/sigma_v, mc_v/sigma_v], b_v = sum_k mc_v^2/sigma_v + log sigma_v), the per-row term c_i = per_pos and
the unigram bias u. The cross term lhs rhs^T stays a cuBLAS GEMM per vocabulary chunk, run under the IEEE fp32
pin exactly as the torch route runs it, so A is bitwise the torch route's GEMM output. Everything after it (the
bias add, the clamp, the temperature, the unigram bias, the online max and log-sum-exp across chunks and the
target gather) runs in one Triton pass per chunk that reads A once and writes three (S,) row states. The backward
recomputes the scores tile by tile and writes

.. math::
    G_{iv} = \frac{\partial \mathrm{CE}}{\partial a_{iv}}
           = -\frac{[\tfrac12(a_{iv}-c_i) \ge 0]}{2} \frac{w_{iv}}{\tau}, \qquad
    w_{iv} = g_i \big(p_{iv} - [v = t_i]\big),

into one (S, Vc) buffer that feeds the two cuBLAS backward GEMMs (dlhs = G rhs, [drhs, db] = G^T [lhs, 1]), and
accumulates :math:`\partial/\partial c_i = -\sum_v G_{iv}` and :math:`\partial/\partial\tau = \tau^{-2}\sum_{iv}
w_{iv}\,\mathrm{kl}_{iv}` per row. The backward GEMMs run under the caller's fp32 matmul policy, as the torch
route's autograd backward does. When the torch route would keep its chunk workspace for backward
(``decode_ce_checkpoint`` accounting), A is kept per chunk and G goes to a separate buffer; when it would
checkpoint, A is recomputed by the same GEMM in backward and G overwrites it.

The logits are the torch route's bitwise (same operations in the same order, fp32 division by IEEE ``div_rn``; the
compiled division flushes denormal results to zero; pinned by the CUDA test). The torch route is unchanged and
remains the reference; the differences are rounding-level and none changes the formula:

* the log-sum-exp is one online pass over BV-wide tiles instead of a per-chunk logsumexp combined across chunks,
  and the softmax weight of the backward is one ``exp((logit - m) - log l)`` instead of the product of the two
  nested logsumexp backwards. The running maximum m and log l are kept apart: rounding m + log l to the ulp of m
  multiplied the gradient error by up to 20 at tau = 0.01;
* compiled kernels take ``exp`` from libdevice, since ``tl.exp`` (``ex2.approx``) tripled the temperature-gradient
  error at tau = 1; the interpreter uses numpy;
* the row and column sums of the backward (dper_pos, dtau, db) are tile-ordered or GEMM-ordered sums.

The kernels use no atomics, so results are bitwise reproducible run to run for a fixed GPU, shape and dtype.
``triton`` is imported only when this route launches.
"""
from typing import Any, List, Optional, Tuple

import torch

from vfe3.runtime import ieee_fp32_matmul

_KERNELS: Any = None


def _kernels() -> Any:
    r"""Import the Triton kernel module once; fail loudly rather than fall back."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from vfe3.kernels import _kl_decode_triton as kernels
        except ImportError as exc:
            raise RuntimeError(
                "decode_ce_triton=True on CUDA tensors runs the fused Triton decode kernels, which need the "
                f"triton package (triton-windows on Windows), and importing it failed: {exc}. Install triton, "
                "or set decode_ce_triton=False.") from exc
        _KERNELS = kernels
    return _KERNELS


def _launch(kernel: Any, n_rows: int, cfg: dict, *args: Any, **constexprs: Any) -> None:
    r"""Launch ``kernel`` over the row tiles of ``n_rows`` rows with the tile constexprs in ``cfg``."""
    kernel[(-(-n_rows // cfg["BI"]),)](*args, **constexprs, BI=cfg["BI"], BV=cfg["BV"],
                                       num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def _cross_term(lhs: torch.Tensor, rhs_c: torch.Tensor) -> torch.Tensor:
    r"""The (S, Vc) cuBLAS cross term lhs rhs_c^T, IEEE fp32 as in ``_decode_av``."""
    with ieee_fp32_matmul():
        return lhs @ rhs_c.transpose(-1, -2)


class KLDecodeCE(torch.autograd.Function):
    r"""Per-row KL-decode cross-entropy over packed rows; see the module docstring."""

    @staticmethod
    def forward(
        ctx:     Any,
        lhs:     torch.Tensor,           # (S, 2K) expanded-form left factor
        per_pos: torch.Tensor,           # (S,) per-row v-independent term
        rhs:     torch.Tensor,           # (V, 2K) expanded-form right factor
        bias:    torch.Tensor,           # (V,) b_v
        u:       Optional[torch.Tensor], # (V,) unigram bias or None (never differentiated)
        tau:     torch.Tensor,           # (1,) effective temperature
        targets: torch.Tensor,           # (S,) int64 target ids in [0, V)
        chunk:   int,                    # vocabulary chunk width
        keep:    bool,                   # keep A per chunk for backward (else recompute)
    ) -> torch.Tensor:                   # (S,) CE rows
        tk = _kernels()
        S, V = lhs.shape[0], rhs.shape[0]
        dtype, device = lhs.dtype, lhs.device
        has_u = u is not None
        u_arg = u if has_u else bias                                       # dummy pointer, never read
        precise = dtype == torch.float32 and not tk.INTERPRETED
        m = torch.full((S,), float("-inf"), dtype=dtype, device=device)    # (S,) running max
        l = torch.zeros(S, dtype=dtype, device=device)                     # (S,) running sum exp
        t = torch.zeros(S, dtype=dtype, device=device)                     # (S,) target logit
        kept: List[torch.Tensor] = []
        cfg = tk.launch_config()["fwd"]
        with torch.autocast(device_type=device.type, enabled=False):
            for v0 in range(0, V, chunk):
                v1 = min(v0 + chunk, V)
                A = _cross_term(lhs, rhs[v0:v1])                           # (S, Vc)
                if S > 0:
                    _launch(tk.kl_ce_fwd_kernel, S, cfg, A, bias[v0:v1], per_pos, u_arg[v0:v1], targets,
                            tau, m, l, t, S, v1 - v0, v0, HAS_U=has_u, PRECISE_DIV=precise,
                            ACCURATE_EXP=not tk.INTERPRETED)
                if keep:
                    kept.append(A)
            log_l = torch.log(l)                                           # (S,) lse = m + log_l
        # m and log_l stay separate: at a small tau the logits are large, and m + log_l would round lse to the
        # ulp of m, an error every softmax weight of the row would inherit (exp(logit - m - log_l) is exact in
        # the leading difference for the logits that carry the mass).
        ctx.save_for_backward(lhs, per_pos, rhs, bias, u_arg, tau, targets, m, log_l, *kept)
        ctx.has_u, ctx.precise, ctx.chunk, ctx.keep = has_u, precise, chunk, keep
        return (m - t) + log_l

    @staticmethod
    def backward(ctx: Any, grad_rows: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        tk = _kernels()
        lhs, per_pos, rhs, bias, u_arg, tau, targets, m, log_l, *kept = ctx.saved_tensors
        need = ctx.needs_input_grad
        S, V = lhs.shape[0], rhs.shape[0]
        dtype, device = lhs.dtype, lhs.device
        want_lhs, want_pp, want_cols, want_tau = need[0], need[1], need[2] or need[3], need[5]
        g_lhs = torch.zeros_like(lhs) if want_lhs else None                  # (S, 2K)
        g_cols = torch.empty(V, lhs.shape[1] + 1, dtype=dtype, device=device) if want_cols else None
        d_pp = torch.zeros(S, dtype=dtype, device=device)                    # (S,) -sum_v G
        d_tau = torch.zeros(S if want_tau else 1, dtype=dtype, device=device)   # (S,) sum_v w kl
        lhs_one = torch.cat([lhs, torch.ones(S, 1, dtype=dtype, device=device)], dim=1) if want_cols else None
        grad_rows = grad_rows.contiguous()
        cfg = tk.launch_config()["bwd"]
        scratch = (torch.empty(S * min(ctx.chunk, V), dtype=dtype, device=device)
                   if ctx.keep and S > 0 else None)                          # G when A is kept
        with torch.autocast(device_type=device.type, enabled=False):
            for index, v0 in enumerate(range(0, V, ctx.chunk)):
                v1 = min(v0 + ctx.chunk, V)
                rhs_c = rhs[v0:v1]
                if S == 0:
                    if want_cols:
                        g_cols[v0:v1].zero_()
                    continue
                if ctx.keep:
                    A = kept[index]
                    G = scratch[:S * (v1 - v0)].view(S, v1 - v0)
                else:
                    A = _cross_term(lhs, rhs_c)                              # recompute; G overwrites it
                    G = A
                _launch(tk.kl_ce_bwd_kernel, S, cfg, A, G, bias[v0:v1], per_pos, u_arg[v0:v1], targets, tau,
                        m, log_l, grad_rows, d_pp, d_tau, S, v1 - v0, v0,
                        HAS_U=ctx.has_u, PRECISE_DIV=ctx.precise, ACCURATE_EXP=not tk.INTERPRETED,
                        NEED_TAU=want_tau)
                if want_lhs:
                    g_lhs.addmm_(G, rhs_c)                                   # (S, 2K) += G rhs_c
                if want_cols:
                    torch.mm(G.t(), lhs_one, out=g_cols[v0:v1])              # (Vc, 2K+1) = G^T [lhs, 1]
        g_rhs = g_cols[:, :-1] if (want_cols and need[2]) else None
        g_bias = g_cols[:, -1] if (want_cols and need[3]) else None
        g_tau = (d_tau.sum() / (tau * tau)).reshape(tau.shape) if want_tau else None
        return (g_lhs, d_pp if want_pp else None, g_rhs, g_bias, None, g_tau, None, None, None)


def kl_decode_ce_rows(
    lhs:     torch.Tensor,               # (S, 2K) expanded-form left factor
    per_pos: torch.Tensor,               # (S,) per-row v-independent term
    rhs:     torch.Tensor,               # (V, 2K) expanded-form right factor
    bias:    torch.Tensor,               # (V,) b_v
    u:       Optional[torch.Tensor],     # (V,) unigram bias or None
    tau_eff: torch.Tensor,               # (1,) or () effective temperature
    targets: torch.Tensor,               # (S,) target ids in [0, V)
    *,
    chunk:   int,                        # vocabulary chunk width
    keep:    bool,                       # keep the (S, Vc) cross terms for backward instead of recomputing
) -> torch.Tensor:                       # (S,) CE rows
    r"""Validate the packed operands and run the fused forward/backward (see the module docstring).

    One float dtype (float32 or float64) across the float operands. CUDA tensors only, except under
    ``TRITON_INTERPRET=1`` where the Triton interpreter runs on CPU tensors (the test route). ``u`` must not
    require grad (the unigram bias is a buffer).
    """
    tk = _kernels()
    S, two_k = lhs.shape
    V = rhs.shape[0]
    if rhs.shape != (V, two_k) or bias.shape != (V,) or per_pos.shape != (S,) or targets.shape != (S,):
        raise ValueError(f"inconsistent fused decode operands: lhs {tuple(lhs.shape)}, rhs {tuple(rhs.shape)}, "
                         f"bias {tuple(bias.shape)}, per_pos {tuple(per_pos.shape)}, targets {tuple(targets.shape)}")
    if u is not None and (u.shape != (V,) or u.requires_grad):
        raise ValueError("u must be a (V,) tensor that does not require grad")
    floats = [lhs, per_pos, rhs, bias, tau_eff] + ([u] if u is not None else [])
    if lhs.dtype not in (torch.float32, torch.float64) or any(x.dtype != lhs.dtype for x in floats):
        raise TypeError(f"the fused decode CE needs one float32 or float64 dtype; got {[x.dtype for x in floats]}")
    if tau_eff.numel() != 1:
        raise ValueError(f"tau_eff must hold one value, got shape {tuple(tau_eff.shape)}")
    devices = {x.device for x in floats + [targets]}
    if len(devices) != 1 or (lhs.device.type != "cuda" and not tk.INTERPRETED):
        raise ValueError(f"the fused decode CE needs all operands on one CUDA device; got {devices}")
    return KLDecodeCE.apply(
        lhs.contiguous(), per_pos.contiguous(), rhs.contiguous(), bias.contiguous(),
        u.contiguous() if u is not None else None, tau_eff.reshape(1).contiguous(),
        targets.to(torch.int64).contiguous(), int(chunk), bool(keep))
