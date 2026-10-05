r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free (the block owns no nn.Parameters): on the pure default path the learnable capacity
is the PriorBank's prior tables, but the block also receives the opt-in learned exceptions
(CG coupling) as arguments and applies them; the block
runs the iterative E-step (Phase 6) and an optional gauge-equivariant norm on the
mean. The belief handoff across blocks lives in stack.py.
"""

from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    from vfe3.model.relational_operators import RelationalOperators

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext, EStepGradientRecord, MStepCapture
from vfe3.geometry.groups import GaugeGroup
from vfe3.free_energy import attention_tau
from vfe3.inference.e_step import e_step
from vfe3.model.cg_coupling import cg_moment_energy_rows


def _as_coeff(v: 'float | list | tuple', device: torch.device) -> 'float | torch.Tensor':
    r"""Pass a scalar through unchanged; turn a list or tuple into a float32 tensor on device."""
    return torch.as_tensor(v, dtype=torch.float32, device=device) if isinstance(v, (list, tuple)) else v


def e_step_shared_kwargs(
    cfg:    VFE3Config,
    device: torch.device,
) -> dict:
    r"""The cfg-derived knob bag ``vfe_block`` spreads into ``e_step``.

    It holds exactly the configuration the projected receiver reads (``e_step`` takes no
    ``**kwargs``, so a stale knob raises ``TypeError``). Runtime objects (tau, lambda_beta,
    log_prior, relational operators and frames) stay per-call-site. ``device`` places a per-coordinate
    (list-valued) b0/c0 for the state-dependent self-coupling.
    """
    return dict(
        value=cfg.lambda_alpha, kl_max=cfg.kl_max, eps=cfg.eps,
        sigma_max=cfg.sigma_max, e_sigma_q_trust=cfg.e_sigma_q_trust,
        e_mu_q_trust=cfg.e_mu_q_trust,
        family=cfg.family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        b0=_as_coeff(cfg.b0, device), c0=_as_coeff(cfg.c0, device),
        skip_belief_sigma_update=cfg.skip_belief_sigma_update,
    )


def vfe_block(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) prior means
    sigma_p:    torch.Tensor,             # (N, K) prior variances
    group:      GaugeGroup,
    cfg:        VFE3Config,

    *,
    log_prior:       Optional[torch.Tensor]                  = None,
    block_norm:      Optional[Callable[..., torch.Tensor]]   = None,      # cached norm instance (None -> off)
    block_mlp:       Optional[Callable]                      = None,      # selected residual moment transform (None -> off)
    block_mlp_frame: Optional[CanonicalFrameContext]         = None,      # realized frame for canonical_frame mode
    cg_coupling:     Optional[Callable[..., 'tuple']]        = None,      # opt-in CG cross-type coupling (None -> off)
    lambda_beta:     'float | torch.Tensor'                  = 1.0,       # belief-coupling weight (cfg.lambda_beta)
    tau:             'Optional[float | torch.Tensor]'        = None,      # softmax temperature

    capture:         Optional[MStepCapture]        = None,   # out-param: stashes the CONVERGED (pre-transform) belief under 'converged'
    grad_record:     Optional[EStepGradientRecord] = None,   # diag out-param: E-step belief-grad norms (None -> no capture)
    state_record:    Optional[dict]                = None,   # diag out-param: E-step belief/F trace (None -> no capture)

    relational_operators: Optional["RelationalOperators"] = None,
    relational_frame: Optional[CanonicalFrameContext] = None,
    relational_value_frame: Optional[CanonicalFrameContext] = None,
    relational_capture: Optional[dict] = None,
    hidden_bias:     Optional[torch.Tensor]                  = None,      # (B, N, w) per-type hidden bias (None -> off)
) -> BeliefState:
    r"""Run n_e_steps of the E-step from ``belief`` toward the prior, then optional norm.

    ``lambda_beta`` is the belief-coupling weight (the constant cfg.lambda_beta); 1.0 is the pure F.
    The realized frames (with any RoPE fold) reach the receiver through ``relational_frame`` and
    ``relational_value_frame``. ``tau`` is the softmax
    temperature (kappa * sqrt(dim_h)); when None it is computed here from cfg.kappa_beta and
    group.irrep_dims. Pass a precomputed value from vfe_stack to avoid recomputing each layer.
    ``hidden_bias`` (``per_type_hidden_bias``) is this layer's ``(B, N, w)`` slice of the per-type table,
    handed to the block MLP's ``forward_moments(..., hidden_bias=...)``. Under
    ``cfg.block_coordinates='reference'`` the norm and MLP act in reference coordinates through
    ``relational_frame`` (:func:`_reference_coordinate_block`)."""
    if hidden_bias is not None and block_mlp is None:
        raise ValueError("hidden_bias (per_type_hidden_bias) needs a block MLP")
    if tau is None:
        tau = attention_tau(_as_coeff(cfg.kappa_beta, belief.mu.device), group.irrep_dims,
                            device=belief.mu.device)
    out = e_step(
        belief, mu_p, sigma_p, group,
        n_iter=cfg.n_e_steps, tau=tau,
        e_q_mu_lr=cfg.e_q_mu_lr, e_q_sigma_lr=cfg.e_q_sigma_lr,
        lambda_beta=lambda_beta,
        relational_operators=relational_operators,
        relational_frame=relational_frame,
        relational_value_frame=relational_value_frame,
        relational_capture=relational_capture,
        grad_record=grad_record, state_record=state_record,
        log_prior=log_prior,
        **e_step_shared_kwargs(cfg, belief.mu.device),
    )
    if capture is not None:
        # The CONVERGED variational belief q* -- what the E-step's F was minimized over,
        # BEFORE the post-inference transforms below. The manuscript's self-coupling
        # alpha*KL(q||p) is pinned to THIS belief (audit 2026-06-09 overnight F19); the
        # M-step regularizer and the diagnostics F self-term read it from here. With the
        # transforms off (the pure path) it is the SAME object the block returns.
        capture["converged"] = out
    if cg_coupling is not None:              # opt-in CG cross-type coupling: after the E-step, before norm
        # CG moment-energy participation (PB-13): the CG lists live in ``capture`` ONLY when
        # cfg.cg_energy_weight>0, so their presence is the sole trigger. A capture allocated only for
        # M-step self-coupling (no lists) or a captureless scorer/diagnostics call falls to the plain
        # mean update below and never reads a CG list.
        if capture is not None and "cg_moment_energy_rows" in capture:
            # The moment result AND its D(q_post||q_pre) rows are grad-connected.
            pre_mu, pre_sigma = out.mu, out.sigma
            res = cg_coupling.forward_moments(pre_mu, pre_sigma)
            capture["cg_moment_energy_rows"].append(cg_moment_energy_rows(
                pre_mu, pre_sigma, res.mu, res.sigma,
                kl_max=cfg.kl_max, eps=cfg.eps, family=cfg.family,
                full_cov_kl_precision=cfg.full_cov_kl_precision))
            out = out._replace(mu=res.mu, sigma=res.sigma)
        else:
            mu_cg, sigma_cg = cg_coupling(out.mu, out.sigma)
            out = out._replace(mu=mu_cg, sigma=sigma_cg)   # _replace preserves phi/reflection
    if getattr(cfg, "block_coordinates", "local") == "reference":
        return _reference_coordinate_block(
            out, block_norm=block_norm, block_mlp=block_mlp, frame=relational_frame,
            capture=capture, hidden_bias=hidden_bias)
    if block_norm is not None:               # cached parameter-free norm (audit 2d/4f)
        out = out._replace(mu=block_norm(out.mu, out.sigma))   # _replace preserves phi/reflection
    if capture is not None:
        capture["block_mlp_input"] = out     # exact immediate seam after CG/norm
    if block_mlp is not None:                # default-off residual moment transform, after norm before handoff
        if hasattr(block_mlp, "forward_moments"):
            extra = {} if hidden_bias is None else {"hidden_bias": hidden_bias}
            result = block_mlp.forward_moments(
                out.mu, out.sigma, frame_context=block_mlp_frame, **extra)
            out = out._replace(mu=result.mu, sigma=result.sigma)
        else:                                # legacy callable compatibility: coordinate mean only
            if hidden_bias is not None:
                raise TypeError("a legacy block-MLP callable cannot take hidden_bias")
            out = out._replace(mu=block_mlp(out.mu))
    if capture is not None:
        capture["block_mlp_output"] = out
    return out


def _reference_coordinate_block(
    out:         BeliefState,                              # post-attention belief, local coordinates
    *,
    block_norm:  Optional[Callable[..., torch.Tensor]],
    block_mlp:   Optional[Callable],
    frame:       Optional[CanonicalFrameContext],          # realized U_i: (B, N, K, K) or (B, N, G, d, d)
    capture:     Optional[MStepCapture],
    hidden_bias: Optional[torch.Tensor],                   # (B, N, w) or None
) -> BeliefState:
    r"""Norm and MLP in reference coordinates, one conjugation per block (``block_coordinates='reference'``).

    With the realized frame :math:`U_i` (fixed through the stack on the projected route),

    .. math::
        \xi_i = U_i^{-1}\mu_i^{\mathrm{att}},\qquad
        v_i = \mathrm{LN}_{\gamma,b}(\xi_i),\qquad
        \xi_i^+ = v_i + \mathrm{MLP}(v_i),\qquad
        \mu_i^{(\ell+1)} = U_i\xi_i^+,\qquad \Sigma_i^{(\ell+1)} = \Sigma_i^{\mathrm{att}},

    and a Gaussian-template MLP reads the reference variances :math:`c_i = \operatorname{diag}C_i`,
    :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}`. Under the passive action
    :math:`(\mu, \Sigma, U) \to (g\mu, g\Sigma g^\top, gU)` the pair :math:`(\xi, C)` is invariant and
    :math:`\mu^{(\ell+1)} \to g\mu^{(\ell+1)}`. The frame actions run with autocast off (float32, float64
    for an escalated frame); the norm and MLP keep the ambient autocast. The captured
    ``block_mlp_input``/``block_mlp_output`` stay in local coordinates (:math:`U_iv_i`, :math:`U_i\xi_i^+`).
    """
    from vfe3.model.block_mlp import GaussianTemplateBlockMLP
    from vfe3.model.frame_actions import frame_matvec, reference_variances

    if frame is None:
        raise RuntimeError(
            "block_coordinates='reference' requires the realized relational frame U_i; this block "
            "received none (every reference-coordinate forward goes through forward_beliefs)")
    bd = frame.block_diagonal
    xi = frame_matvec(frame.inverse, out.mu, bd)                             # (B, N, K) xi = U^{-1} mu
    template = isinstance(block_mlp, GaussianTemplateBlockMLP)
    c = reference_variances(out.mu, out.sigma, frame) if template else out.sigma   # diag C, or unread
    if block_norm is not None:
        xi = block_norm(xi, c)                                               # LN acts on xi only
    if capture is not None:
        capture["block_mlp_input"] = out._replace(mu=frame_matvec(frame.forward, xi, bd))
    if block_mlp is not None:
        if hasattr(block_mlp, "forward_moments"):
            extra = {} if hidden_bias is None else {"hidden_bias": hidden_bias}
            xi = block_mlp.forward_moments(xi, c, **extra).mu                # passthrough: covariance kept
        else:
            if hidden_bias is not None:
                raise TypeError("a legacy block-MLP callable cannot take hidden_bias")
            xi = block_mlp(xi)
    out = out._replace(mu=frame_matvec(frame.forward, xi, bd))               # (B, N, K) mu = U xi^+
    if capture is not None:
        capture["block_mlp_output"] = out
    return out
