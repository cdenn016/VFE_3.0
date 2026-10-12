r"""Fused Triton cross-entropy for the expected-likelihood decode (every CUDA tensor; CPU tensors run torch).

For packed rows i (the scored positions) with diagonal posteriors (mu_q, sigma_q) and diagonal decode
tables (mu_v, sigma_v), the logits of ``decode_mode='expected_likelihood_chunked'`` are

.. math::
    \ell_{iv} = -c E_{iv} + u_v, \qquad c = \frac{1}{2\tau}, \qquad
    E_{iv} = \sum_k \Big[ \frac{d_{ivk}^2}{s_{ivk}} + \log s_{ivk} \Big],
    \quad d_{ivk} = \mu_{q,ik} - \mu_{v,vk}, \quad s_{ivk} = \sigma_{q,ik} + \sigma_{v,vk},

and the cross-entropy is :math:`\mathrm{CE} = S^{-1} \sum_i [\mathrm{LSE}_v(\ell_{iv}) - \ell_{i t_i}]`.
The coupling :math:`1/(\sigma_{q,ik} + \sigma_{v,vk})` is a Cauchy kernel in (i, v), which has full rank
for distinct entries, so no term separates into a GEMM. The kernels instead stream (i, v) register
tiles and never form the (S, V, K) or (S, V) tensors. With
:math:`w_{iv} = g (p_{iv} - [v = t_i]) / S` (g the upstream gradient, p the softmax) and
:math:`r_{ivk} = d_{ivk} / s_{ivk}`, the analytic backward is

.. math::
    \frac{\partial \mathrm{CE}}{\partial \mu_{q,ik}} = -2c \sum_v w_{iv} r_{ivk}, \qquad
    \frac{\partial \mathrm{CE}}{\partial \sigma_{q,ik}} = c \sum_v w_{iv} \big(r_{ivk}^2 - s_{ivk}^{-1}\big),

.. math::
    \frac{\partial \mathrm{CE}}{\partial \mu_{v,vk}} = 2c \sum_i w_{iv} r_{ivk}, \qquad
    \frac{\partial \mathrm{CE}}{\partial \sigma_{v,vk}} = c \sum_i w_{iv} \big(r_{ivk}^2 - s_{ivk}^{-1}\big), \qquad
    \frac{\partial \mathrm{CE}}{\partial u_v} = \sum_i w_{iv}, \qquad
    \frac{\partial \mathrm{CE}}{\partial \tau} = \frac{c}{\tau} \sum_{iv} w_{iv} E_{iv}.

The target term :math:`[v = t_i]` is folded into :math:`w` inside the tiles, so no ``index_add`` is
needed. Gradients reach the parameters through ordinary autograd outside the Function
(:math:`\sigma_v` = ``bounded_variance_from_log`` of the log table, :math:`\tau` = ``_tau_eff``).

Rounding-level differences from the torch route (``PriorBank.decode_ce_expected_likelihood_chunked``),
which is unchanged and remains the reference. None of them changes the formula.

* Summation order. E sums over k in one register accumulator per pair, the logsumexp is online over
  BV-wide vocabulary tiles rather than per ``decode_chunk_size`` chunk, and the row and column
  gradients are tile-ordered sums. The torch route sums the d^2/s and log s terms together; here they
  are summed separately and added once per pair.
* Scaling. Logits use :math:`-c E` with :math:`c = 0.5/\tau` instead of :math:`-0.5 E / \tau`.
* Compiled fp32 arithmetic. Division compiles to ``div.full.f32`` (at most 2 ulp), ``exp`` to
  ``ex2.approx``, and the per-pair logarithm to ``lg2.approx.f32`` times :math:`\ln 2` (absolute error
  about :math:`2^{-22}` per term). The torch route uses IEEE division and accurate ``exp``/``log``.
  The interpreter (``TRITON_INTERPRET=1``) and float64 use ``tl.log`` and IEEE division.

Every output element has a single owner program, and the kernels use no atomics. Results are
therefore bitwise reproducible run to run for a fixed GPU, shape and dtype; the only host reductions
(``sum`` over the (S,) rows) are deterministic torch reductions. ``triton`` is imported only when
this route launches, so CPU-only machines never import it.
"""
from typing import Any, Optional, Tuple

import torch

_KERNELS: Any = None


def _kernels() -> Any:
    r"""Import the Triton kernel module once; fail loudly rather than fall back."""
    global _KERNELS
    if _KERNELS is None:
        try:
            from vfe3.kernels import _expected_likelihood_triton as kernels
        except ImportError as exc:
            raise RuntimeError(
                "decode_mode='expected_likelihood_chunked' on CUDA tensors runs the fused Triton "
                "kernel, which needs the triton package (triton-windows on Windows), and importing "
                f"it failed: {exc}. Install triton, or run the decoder on CPU tensors."
            ) from exc
        _KERNELS = kernels
    return _KERNELS


def _grid_launch(kernel: Any, n_programs: int, cfg: dict, *args: Any, **constexprs: Any) -> None:
    r"""Launch ``kernel`` over ``n_programs`` programs with the tile constexprs in ``cfg``."""
    kernel[(n_programs,)](*args, **constexprs, BI=cfg["BI"], BV=cfg["BV"],
                          num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


class ExpectedLikelihoodCE(torch.autograd.Function):
    r"""Mean expected-likelihood cross-entropy over packed rows; saves only lse_i (S,) besides inputs."""

    @staticmethod
    def forward(
        ctx:     Any,
        mu_q:    torch.Tensor,           # (S, K) packed posterior means
        sigma_q: torch.Tensor,           # (S, K) packed posterior variances
        mu_v:    torch.Tensor,           # (V, K) decode mean table
        sigma_v: torch.Tensor,           # (V, K) decode variance table
        u:       Optional[torch.Tensor], # (V,) unigram log-prior bias or None
        tau:     torch.Tensor,           # (1,) effective temperature
        targets: torch.Tensor,           # (S,) int64 target ids in [0, V)
    ) -> torch.Tensor:                   # () mean CE
        tk = _kernels()
        S, K = mu_q.shape
        V = mu_v.shape[0]
        mu_v_t = mu_v.t().contiguous()                                  # (K, V) coalesced over v
        sigma_v_t = sigma_v.t().contiguous()                            # (K, V)
        has_u = u is not None
        u_arg = u if has_u else mu_v_t                                  # dummy pointer, never read
        fast_log = mu_q.dtype == torch.float32 and not tk.INTERPRETED
        lse = torch.empty(S, dtype=mu_q.dtype, device=mu_q.device)      # (S,)
        ce_rows = torch.empty(S, dtype=mu_q.dtype, device=mu_q.device)  # (S,)
        cfgs = tk.launch_config(K)
        if S > 0:
            cfg = cfgs["fwd"]
            _grid_launch(tk.el_fwd_kernel, -(-S // cfg["BI"]), cfg,
                         mu_q, sigma_q, mu_v_t, sigma_v_t, u_arg, targets, tau, lse, ce_rows, S, V,
                         K=K, HAS_U=has_u, FAST_LOG=fast_log)
        ctx.save_for_backward(mu_q, sigma_q, mu_v_t, sigma_v_t, u_arg, tau, targets, lse)
        ctx.has_u, ctx.fast_log = has_u, fast_log
        return ce_rows.sum() / max(S, 1)

    @staticmethod
    def backward(ctx: Any, grad_ce: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        tk = _kernels()
        mu_q, sigma_q, mu_v_t, sigma_v_t, u_arg, tau, targets, lse = ctx.saved_tensors
        S, K = mu_q.shape
        V = mu_v_t.shape[1]
        need = ctx.needs_input_grad
        want_rows = need[0] or need[1] or need[5]
        want_cols = need[2] or need[3] or (ctx.has_u and need[4])
        cfgs = tk.launch_config(K)
        scale = (grad_ce.reshape(1) / max(S, 1)).to(mu_q.dtype).contiguous()   # (1,) g / S, on device
        common = (mu_q, sigma_q, mu_v_t, sigma_v_t, u_arg, targets, tau, lse, scale)
        consts = dict(K=K, KP=cfgs["rows"]["KP"], HAS_U=ctx.has_u, FAST_LOG=ctx.fast_log)

        g_mu_q = g_sigma_q = g_mu_v = g_sigma_v = g_u = g_tau = None
        if want_rows:
            g_mu_q = torch.zeros_like(mu_q)                             # (S, K)
            g_sigma_q = torch.zeros_like(sigma_q)                       # (S, K)
            w_energy = torch.zeros(S, dtype=mu_q.dtype, device=mu_q.device)   # (S,) sum_v w E
            if S > 0:
                cfg = cfgs["rows"]
                _grid_launch(tk.el_bwd_rows_kernel, -(-S // cfg["BI"]), cfg,
                             *common, g_mu_q, g_sigma_q, w_energy, S, V, **consts)
            if need[5]:
                g_tau = (w_energy.sum() * (0.5 / tau) / tau).reshape(tau.shape)
        if want_cols:
            g_mu_v = torch.zeros(V, K, dtype=mu_q.dtype, device=mu_q.device)      # (V, K)
            g_sigma_v = torch.zeros(V, K, dtype=mu_q.dtype, device=mu_q.device)   # (V, K)
            g_u_all = torch.zeros(V, dtype=mu_q.dtype, device=mu_q.device)        # (V,)
            if S > 0:
                cfg = cfgs["cols"]
                _grid_launch(tk.el_bwd_cols_kernel, -(-V // cfg["BV"]), cfg,
                             *common, g_mu_v, g_sigma_v, g_u_all, S, V, **consts)
            g_u = g_u_all if ctx.has_u else None
        return (g_mu_q if need[0] else None, g_sigma_q if need[1] else None,
                g_mu_v if need[2] else None, g_sigma_v if need[3] else None,
                g_u if need[4] else None, g_tau, None)


def expected_likelihood_ce(
    mu_q:    torch.Tensor,               # (S, K) packed posterior means
    sigma_q: torch.Tensor,               # (S, K) packed posterior variances
    mu_v:    torch.Tensor,               # (V, K) decode mean table
    sigma_v: torch.Tensor,               # (V, K) decode variance table
    u:       Optional[torch.Tensor],     # (V,) unigram log-prior bias or None
    tau_eff: torch.Tensor,               # (1,) or () effective temperature
    targets: torch.Tensor,               # (S,) target ids in [0, V) (ignored rows already packed out)
) -> torch.Tensor:                       # () mean CE over the S rows (0 when S == 0)
    r"""Validate the packed operands and run the fused forward/backward (see the module docstring).

    CUDA tensors only, except under ``TRITON_INTERPRET=1`` where the Triton interpreter runs on CPU
    tensors (the test route). Raises rather than falling back when triton is missing.
    """
    tk = _kernels()
    if mu_q.dim() != 2 or sigma_q.shape != mu_q.shape:
        raise ValueError(f"mu_q and sigma_q must both be (S, K); got {tuple(mu_q.shape)} and "
                         f"{tuple(sigma_q.shape)} (the expected-likelihood decode is diagonal only)")
    S, K = mu_q.shape
    if mu_v.shape != (mu_v.shape[0], K) or sigma_v.shape != mu_v.shape:
        raise ValueError(f"mu_v and sigma_v must both be (V, {K}); got {tuple(mu_v.shape)} and "
                         f"{tuple(sigma_v.shape)}")
    V = mu_v.shape[0]
    if u is not None and u.shape != (V,):
        raise ValueError(f"u must be ({V},); got {tuple(u.shape)}")
    if tau_eff.numel() != 1 or targets.shape != (S,):
        raise ValueError(f"tau_eff must hold one value and targets must be ({S},); got "
                         f"{tuple(tau_eff.shape)} and {tuple(targets.shape)}")
    floats = [mu_q, sigma_q, mu_v, sigma_v, tau_eff] + ([u] if u is not None else [])
    if mu_q.dtype not in (torch.float32, torch.float64) or any(x.dtype != mu_q.dtype for x in floats):
        raise TypeError("the fused expected-likelihood CE needs one float32 or float64 dtype across "
                        f"mu_q, sigma_q, mu_v, sigma_v, u and tau_eff; got {[x.dtype for x in floats]}")
    devices = {x.device for x in floats + [targets]}
    if len(devices) != 1 or (mu_q.device.type != "cuda" and not tk.INTERPRETED):
        raise ValueError(f"the fused expected-likelihood CE needs all operands on one CUDA device; got {devices}")
    return ExpectedLikelihoodCE.apply(
        mu_q.contiguous(), sigma_q.contiguous(), mu_v.contiguous(), sigma_v.contiguous(),
        u.contiguous() if u is not None else None, tau_eff.reshape(1).contiguous(),
        targets.to(torch.int64).contiguous())
