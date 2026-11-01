r"""Fused Triton preparation of the KL decode bank factors (``decode_bank_triton``).

The fused chunked KL decode cross-entropy (``vfe3/kernels/kl_decode.py``) scores packed rows against the
expanded-form right factor and bias of the diagonal decode bank :math:`p_v = \mathcal N(\mu_v, \operatorname{diag}
\sigma_v)`,

.. math::
    \sigma_v = \max\big(e^{\min(\ell_v, \ell_{\max})}, \epsilon\big), \qquad mc_v = \mu_v - c, \qquad
    \mathrm{rhs}_v = [\iota_v, mc_v \circ \iota_v], \qquad
    b_v = \sum_k mc_{vk}^2\,\iota_{vk} + \sum_k \log\sigma_{vk},

with :math:`\iota = 1/\sigma`, :math:`\ell` the (V, K) log-variance table (``bounded_variance_from_log`` with
:math:`\ell_{\max} = 80`) and :math:`c = \mathrm{mean}_v\,\mu_v` the centering offset. The torch route forms these
with about a dozen vocabulary-table-sized kernels and differentiates them with about thirty more (audit 2026-10-31
perf-live-path-3). Here one kernel writes rhs and b, and one writes their closed-form derivatives: with incoming
:math:`\bar r = [\bar r^{(1)}, \bar r^{(2)}]` and :math:`\bar b`,

.. math::
    \bar\iota_{vk} = \bar r^{(1)}_{vk} + \bar r^{(2)}_{vk}\,mc_{vk} + \bar b_v\,mc_{vk}^2, \qquad
    \bar\mu_{vk} = \bar r^{(2)}_{vk}\,\iota_{vk} + 2\bar b_v\,mc_{vk}\,\iota_{vk}, \qquad
    \bar\ell_{vk} = \pi_{vk}\big(\bar b_v - \bar\iota_{vk}\,\iota_{vk}\big),

since :math:`\partial\iota/\partial\ell = -\iota` and :math:`\partial\log\sigma/\partial\ell = 1` where neither clamp
is active, and :math:`\pi_{vk} = [\ell_{vk} \le \ell_{\max}]\,[e^{\min(\ell_{vk}, \ell_{\max})} \ge \epsilon]` is the
pass mask of the two clamps (``torch.clamp`` passes the gradient at a bound). :math:`c` is an input: the Function
returns :math:`\bar c = -\sum_v \bar\mu_v` (column partial sums per row tile, added by a deterministic torch
reduction), and autograd adds the broadcast term :math:`\bar c/V` of :math:`c = \mathrm{mean}_v\,\mu_v` to every row.
In units of one (V, K) table, the forward reads two and writes two (rhs), and the backward reads four (the tables
and the rhs gradient) and writes two, against about forty table-sized torch kernels.

Rounding-level differences from the torch preparation (``PriorBank._decode_ce_triton_rows``), which is unchanged and
remains the reference: the k-sums of :math:`b` are tile-ordered and :math:`mc^2\iota` is formed as
:math:`mc\,(mc\,\iota)`; :math:`\bar\ell` uses :math:`-\iota` where autograd forms :math:`-\iota^2 e^{\ell}`, and
:math:`\bar b` where autograd forms :math:`(\bar b/\sigma) e^{\ell}`. Compiled kernels take ``exp`` and ``log`` from
libdevice and divide by IEEE ``div_rn`` in float32; the interpreter uses numpy. The kernels use no atomics, so results
are bitwise reproducible for a fixed GPU, shape and dtype. ``triton`` is imported only when this route launches.
"""
from typing import Any, Optional, Tuple

import torch

from vfe3.kernels._autograd import strict_once_differentiable

_KERNELS: Any = None

# max_log of vfe3.numerics.bounded_variance_from_log: the log-variance is clamped there before exponentiation.
DECODE_BANK_MAX_LOG: float = 80.0


def _kernels() -> Any:
    r"""Import the Triton kernel module once; fail loudly rather than fall back."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from vfe3.kernels import _decode_bank_triton as kernels
        except ImportError as exc:
            raise RuntimeError(
                "decode_bank_triton=True on CUDA tensors runs the fused Triton decode-bank preparation, which needs "
                f"the triton package (triton-windows on Windows), and importing it failed: {exc}. Install triton, "
                "or set decode_bank_triton=False.") from exc
        _KERNELS = kernels
    return _KERNELS


class DecodeBankFactors(torch.autograd.Function):
    r"""(rhs, b) of the decode bank from (mu_v, log sigma_v, c); see the module docstring.

    The backward is once differentiable (its gradients are Triton writes autograd does not track), and refuses a
    double backward toward any upstream leaf (``strict_once_differentiable``). Saves only its inputs.
    """

    @staticmethod
    def forward(
        ctx:         Any,
        mu_v:        torch.Tensor,       # (V, K) decode mean table
        log_sigma_v: torch.Tensor,       # (V, K) decode log-variance table
        c:           torch.Tensor,       # (1, K) centering offset mean_v(mu_v)
        eps:         float,              # variance floor
    ) -> Tuple[torch.Tensor, torch.Tensor]:   # (V, 2K) rhs, (V,) bias
        tk = _kernels()
        V, K = mu_v.shape
        dtype, device = mu_v.dtype, mu_v.device
        rhs = torch.empty(V, 2 * K, dtype=dtype, device=device)
        bias = torch.empty(V, dtype=dtype, device=device)
        eps_t = torch.full((1,), eps, dtype=dtype, device=device)          # the floor at the tables' precision
        cfg = tk.launch_config()
        accurate = not tk.INTERPRETED
        if V > 0 and K > 0:
            tk.bank_fwd_kernel[(-(-V // cfg["BV"]),)](
                mu_v, log_sigma_v, c, eps_t, rhs, bias, V, K, BV=cfg["BV"], BK=cfg["BK"],
                MAX_LOG=DECODE_BANK_MAX_LOG, ACCURATE=accurate, PRECISE_DIV=accurate and dtype == torch.float32,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        elif V > 0:
            bias.zero_()
        ctx.save_for_backward(mu_v, log_sigma_v, c, eps_t)
        return rhs, bias

    @staticmethod
    @strict_once_differentiable
    def backward(ctx: Any, g_rhs: torch.Tensor, g_bias: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        tk = _kernels()
        mu_v, log_sigma_v, c, eps_t = ctx.saved_tensors
        V, K = mu_v.shape
        dtype, device = mu_v.dtype, mu_v.device
        cfg = tk.launch_config()
        accurate = not tk.INTERPRETED
        n_programs = -(-V // cfg["BV"])
        g_mu = torch.empty_like(mu_v)                                     # (V, K) d/dmu_v at fixed c
        g_l = torch.empty_like(log_sigma_v)                               # (V, K)
        partial = torch.empty(n_programs, K, dtype=dtype, device=device)  # (P, K) column sums of g_mu per tile
        if V > 0 and K > 0:
            tk.bank_bwd_kernel[(n_programs,)](
                mu_v, log_sigma_v, c, eps_t, g_rhs.contiguous(), g_bias.contiguous(), g_mu, g_l, partial, V, K,
                BV=cfg["BV"], BK=cfg["BK"], MAX_LOG=DECODE_BANK_MAX_LOG, ACCURATE=accurate,
                PRECISE_DIV=accurate and dtype == torch.float32,
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
            g_c = -partial.sum(0, keepdim=True)                            # (1, K) mc_v = mu_v - c
        else:
            g_mu.zero_()
            g_l.zero_()
            g_c = torch.zeros_like(c)
        need = ctx.needs_input_grad
        return (g_mu if need[0] else None, g_l if need[1] else None, g_c if need[2] else None, None)


def decode_bank_factors(
    mu_v:        torch.Tensor,           # (V, K) decode mean table
    log_sigma_v: torch.Tensor,           # (V, K) decode log-variance table
    c:           torch.Tensor,           # (1, K) centering offset, mean_v(mu_v) (computed by the caller)

    *,
    eps:         float,                  # variance floor of bounded_variance_from_log
) -> Tuple[torch.Tensor, torch.Tensor]:  # (V, 2K) rhs, (V,) bias
    r"""Validate the operands and run the fused bank preparation (see the module docstring).

    One float dtype (float32 or float64). CUDA tensors only, except under ``TRITON_INTERPRET=1`` where the Triton
    interpreter runs on CPU tensors (the test route). The caller keeps the overflow warning of
    ``bounded_variance_from_log`` (``vfe3.numerics._check_max_log``).
    """
    tk = _kernels()
    if mu_v.dim() != 2 or log_sigma_v.shape != mu_v.shape or c.shape != (1, mu_v.shape[1]):
        raise ValueError(f"inconsistent decode bank operands: mu_v {tuple(mu_v.shape)}, log_sigma_v "
                         f"{tuple(log_sigma_v.shape)}, c {tuple(c.shape)}")
    if mu_v.dtype not in (torch.float32, torch.float64) or log_sigma_v.dtype != mu_v.dtype or c.dtype != mu_v.dtype:
        raise TypeError(f"the fused decode bank needs one float32 or float64 dtype; got "
                        f"{[mu_v.dtype, log_sigma_v.dtype, c.dtype]}")
    devices = {mu_v.device, log_sigma_v.device, c.device}
    if len(devices) != 1 or (mu_v.device.type != "cuda" and not tk.INTERPRETED):
        raise ValueError(f"the fused decode bank needs all operands on one CUDA device; got {devices}")
    return DecodeBankFactors.apply(mu_v.contiguous(), log_sigma_v.contiguous(), c.contiguous(), float(eps))
