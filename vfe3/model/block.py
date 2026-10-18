r"""A single VFE block for VFE_3.0: E-step belief inference + optional norm.

Parameter-free (the block owns no nn.Parameters): on the pure default path the learnable capacity
is the PriorBank's prior tables, but the block also receives the opt-in learned exceptions
(CG coupling) as arguments and applies them; the block
runs the attention step and the configured block norm (the default ``layernorm`` is not
gauge-equivariant; ``none`` and ``mahalanobis`` are) on the mean, and under ``block_norm_covariance='pushforward'``
its pushforward on the covariance (first order plus the radial second-order term), or under ``'pushforward_moment'``
the same at the effective scale of the input's expected LayerNorm variance. The belief handoff across blocks
lives in stack.py.
"""

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Callable, Optional, Sequence

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
        kl_max=cfg.kl_max, eps=cfg.eps,
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
    link_state:      Optional[dict]                          = None,      # content_link_loop_weight > 0: cross-layer link state
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
    ``relational_frame`` (:func:`_reference_coordinate_block`). ``link_state`` carries the previous
    layer's inferred links to the loop prior of ``content_link_loop_weight`` (passed only when it is on)."""
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
        **({"link_state": link_state} if link_state is not None else {}),
    )
    if capture is not None:
        # The belief q* the E-step returns (on projected_predictive the predictive readout; no supported route
        # minimizes F, vfe3.free_energy.free_energy), BEFORE the post-inference transforms below. The manuscript's self-coupling
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
                kl_max=cfg.kl_max, eps=cfg.eps, family=cfg.family))
            out = out._replace(mu=res.mu, sigma=res.sigma)
        else:
            mu_cg, sigma_cg = cg_coupling(out.mu, out.sigma)
            out = out._replace(mu=mu_cg, sigma=sigma_cg)   # _replace preserves phi/reflection
    mlp_delta_full = getattr(block_mlp, "covariance_contract", None) == "delta_full"
    # block_norm_covariance='pushforward' (audit 2026-10-10, item 9) or 'pushforward_moment' (audit 2026-10-16, F3):
    # each norm application also carries Sigma to post-norm units; inert under a delta_full MLP, whose Jacobian
    # composes the norm's, and under a norm other than the LayerNorm, the only norm with that pushforward (the
    # validator admits either value there and inactive_fields marks it inert; audit 2026-10-16 F5).
    from vfe3.model.block_mlp import BLOCK_NORM_PUSHFORWARD_MODES
    norm_covariance = cfg.block_norm_covariance
    push_norm = (norm_covariance in BLOCK_NORM_PUSHFORWARD_MODES and not mlp_delta_full
                 and cfg.norm_type_block == "layernorm")
    from vfe3.families.base import get_family
    frame_diagonal = get_family(cfg.family).frame_diagonal_covariance
    if cfg.block_coordinates == "reference":
        return _reference_coordinate_block(
            out, block_norm=block_norm, block_mlp=block_mlp, frame=relational_frame,
            capture=capture, hidden_bias=hidden_bias,
            readout=lambda pushed: _head_block_readout(pushed, out.sigma, group, cfg),
            push_norm=push_norm, covariance_floor=cfg.block_mlp_covariance_floor,
            norm_covariance=norm_covariance if push_norm else "pushforward",
            trace_cap=cfg.block_norm_trace_cap,
            cap_moment_fallback=cfg.block_norm_cap_moment_fallback,
            frame_diagonal=frame_diagonal)
    # gaussian_frame_diagonal in local coordinates: a covariance pushforward (the LayerNorm's or a delta_full MLP's)
    # acts on the local Sigma_i = U_i diag(c_i) U_i^T, and its result S is read back as the family's M-projection
    # c_i' = diag(U_i^{-1} S U_i^{-T}); a passthrough covariance keeps c_i exactly (_frame_diagonal_local_block).
    if frame_diagonal and (mlp_delta_full or (push_norm and block_norm is not None)):
        return _frame_diagonal_local_block(
            out, block_norm=block_norm, block_mlp=block_mlp, group=group, cfg=cfg, frame=relational_frame,
            block_mlp_frame=block_mlp_frame, capture=capture, hidden_bias=hidden_bias, push_norm=push_norm,
            mlp_delta_full=mlp_delta_full)
    return _local_coordinate_block(
        out, block_norm=block_norm, block_mlp=block_mlp, group=group, cfg=cfg, block_mlp_frame=block_mlp_frame,
        capture=capture, hidden_bias=hidden_bias, push_norm=push_norm, mlp_delta_full=mlp_delta_full)


def _frame_diagonal_local_block(
    out:            BeliefState,                       # post-attention belief: local mu, reference variances c
    *,
    block_norm:     Optional[Callable[..., torch.Tensor]],
    block_mlp:      Optional[Callable],
    group:          GaugeGroup,
    cfg:            VFE3Config,
    frame:          Optional[CanonicalFrameContext],   # realized U_i: (B, N, K, K) or (B, N, G, d, d)
    block_mlp_frame: Optional[CanonicalFrameContext],
    capture:        Optional[MStepCapture],
    hidden_bias:    Optional[torch.Tensor],
    push_norm:      bool,
    mlp_delta_full: bool,
) -> BeliefState:
    r"""The local-coordinate block of ``gaussian_frame_diagonal`` when a covariance pushforward is active.

    The block runs on the local covariance :math:`\Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top` (``(B, N, K, K)``,
    :func:`vfe3.model.frame_actions.frame_diagonal_local_covariance`) exactly as on the full family, and a returned
    covariance :math:`S_i` is read back as :math:`c_i' = \operatorname{diag}(U_i^{-1}S_iU_i^{-\top})`, the M-projection
    :math:`\arg\min_c\mathrm{KL}(\mathcal N(\mu, S_i)\,\|\,\mathcal N(\mu, U_i\operatorname{diag}(c)U_i^\top))`. The
    captured ``block_mlp_input`` and ``block_mlp_output`` carry the family's layout (reference variances).
    """
    from vfe3.model.frame_actions import frame_diagonal_local_covariance, reference_variances
    if frame is None:
        raise RuntimeError("gaussian_frame_diagonal reads its local covariance through the realized relational frame "
                           "U_i; this block received none (every projected forward goes through forward_beliefs)")
    c_in = out.sigma
    local = frame_diagonal_local_covariance(c_in, frame, diagonal=False)            # (B, N, K, K) U diag(c) U^T
    result = _local_coordinate_block(
        out._replace(sigma=local), block_norm=block_norm, block_mlp=block_mlp, group=group, cfg=cfg,
        block_mlp_frame=block_mlp_frame, capture=capture, hidden_bias=hidden_bias,
        push_norm=push_norm, mlp_delta_full=mlp_delta_full)

    def readout(belief: BeliefState) -> BeliefState:
        if belief.sigma is local:                                                   # passthrough: c exactly
            return belief._replace(sigma=c_in)
        return belief._replace(sigma=reference_variances(belief.mu, belief.sigma, frame).to(c_in.dtype))

    if capture is not None:
        for key in ("block_mlp_input", "block_mlp_output"):
            if key in capture:
                capture[key] = readout(capture[key])
    return readout(result)


def _local_coordinate_block(
    out:            BeliefState,
    *,
    block_norm:     Optional[Callable[..., torch.Tensor]],
    block_mlp:      Optional[Callable],
    group:          GaugeGroup,
    cfg:            VFE3Config,
    block_mlp_frame: Optional[CanonicalFrameContext],
    capture:        Optional[MStepCapture],
    hidden_bias:    Optional[torch.Tensor],
    push_norm:      bool,
    mlp_delta_full: bool,
) -> BeliefState:
    r"""The block norm and block MLP in local coordinates (``block_coordinates='local'``), after the E-step and CG."""
    norm_linearization = norm_radial = norm_cap = None
    if block_norm is not None:               # cached parameter-free norm (audit 2d/4f)
        if mlp_delta_full:
            # The norm maps the mean only; the delta_full pushforward composes its Jacobian at the pre-norm
            # mean, J = J_mlp J_LN, against the pre-norm covariance (audit 2026-10-09, variational), with the
            # floor in post-norm units through the norm's scale D = diag(gamma)/s (round 2, #13) and the norm's
            # radial second-order variance carried through J_mlp (2026-10-10 round 2, #11).
            from vfe3.model.block_mlp import block_norm_linearization, block_norm_radial_term, block_norm_trace_cap
            norm_linearization = block_norm_linearization(block_norm, out.mu)   # (B, N, K, K), (B, N, K)
            norm_radial = block_norm_radial_term(block_norm, out.mu, out.sigma)  # (B, N, 1), (B, N, K)
            # The composite is linear in the norm-stage covariance, so the norm's sphere-bound cap scales it
            # (audit 2026-10-12, item 1).
            norm_cap = (block_norm_trace_cap(block_norm, out.mu, out.sigma, cfg.block_mlp_covariance_floor)   # (B, N, 1)
                        if cfg.block_norm_trace_cap else None)
        out = _apply_block_norm(out, block_norm, group, cfg, push_covariance=push_norm)
    if capture is not None:
        capture["block_mlp_input"] = out     # exact immediate seam after CG/norm
    if block_mlp is not None:                # default-off residual moment transform, after norm before handoff
        context, mu_in = _block_mlp_precision(out.mu)
        if hasattr(block_mlp, "forward_moments"):
            extra = {} if hidden_bias is None else {"hidden_bias": hidden_bias}
            if norm_linearization is not None:
                extra["input_jacobian"] = norm_linearization.jacobian
                extra["input_scale"] = norm_linearization.scale
                extra["input_radial"] = norm_radial
                extra["input_cap"] = norm_cap
            if capture is not None and "block_mlp_em_rows" in capture:   # block_mlp_em_weight > 0 only
                extra["em_rows"] = capture["block_mlp_em_rows"]
            with context:
                result = block_mlp.forward_moments(
                    mu_in, out.sigma, frame_context=block_mlp_frame, **extra)
            out = out._replace(mu=result.mu, sigma=_head_block_readout(result.sigma, out.sigma, group, cfg))
        else:                                # legacy callable compatibility: coordinate mean only
            if hidden_bias is not None:
                raise TypeError("a legacy block-MLP callable cannot take hidden_bias")
            with context:
                out = out._replace(mu=block_mlp(mu_in))
    if capture is not None:
        capture["block_mlp_output"] = out
    return out


def _apply_block_norm(
    out:             BeliefState,
    block_norm:      Callable[..., torch.Tensor],
    group:           GaugeGroup,
    cfg:             VFE3Config,
    *,
    push_covariance: bool,                    # a block_norm_covariance pushforward outside a delta_full MLP
) -> BeliefState:
    r"""The block norm on the mean, and under ``push_covariance`` its pushforward of the covariance.

    The pushforward is taken at the pre-norm mean :math:`u`:
    :math:`\Sigma \mapsto J_{\mathrm{LN}}\Sigma J_{\mathrm{LN}}^\top + fD\Sigma D + \rho gg^\top`,
    :math:`D = \operatorname{diag}(\gamma)/s`, :math:`f` = ``block_mlp_covariance_floor``, with the radial second-order
    term :math:`\rho gg^\top` (:func:`norm_covariance_pushforward`; under ``'pushforward_moment'`` at the effective
    scale :math:`s_{\mathrm{eff}}`). Without ``push_covariance`` the covariance object passes through unchanged.
    """
    if not push_covariance:
        return out._replace(mu=block_norm(out.mu, out.sigma))   # _replace preserves phi/reflection
    return out._replace(mu=block_norm(out.mu, out.sigma),
                        sigma=norm_covariance_pushforward(block_norm, out.mu, out.sigma, group, cfg))


def norm_covariance_pushforward(
    norm:  Callable[..., torch.Tensor],       # the LayerNorm block norm
    mu:    torch.Tensor,                      # (B, N, K) local pre-norm means
    sigma: torch.Tensor,                      # (B, N, K) or (B, N, K, K) local covariance at mu
    group: GaugeGroup,
    cfg:   VFE3Config,
) -> torch.Tensor:                            # sigma's layout, read by the family
    r"""The covariance through the LayerNorm block norm under ``block_norm_covariance`` 'pushforward' or
    'pushforward_moment', read by the family.

    :func:`vfe3.model.block_mlp.block_norm_covariance_pushforward` at the pre-norm mean in local coordinates; the
    diagonal family keeps its diagonal and ``gaussian_block`` its head blocks, each the M-projection of the pushed
    Gaussian onto the family. The pushed Gaussian approximates the LayerNorm output's moments: under 'pushforward' by
    the delta method, valid while :math:`\Sigma` is small against :math:`s^2` and capped at the sphere bound beyond,
    where it keeps the delta method's radial shape (audit 2026-10-16 F3, F2); under 'pushforward_moment' by the delta
    method at the effective scale :math:`s_{\mathrm{eff}}^2 = s^2 + \operatorname{tr}(\Pi\Sigma\Pi)/K`, which tracks the
    Monte Carlo moments at every :math:`\Sigma/s^2`. Reference coordinates run their own pushforward on
    :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}` in :func:`_reference_coordinate_block`.
    """
    from vfe3.model.block_mlp import block_norm_covariance_pushforward
    return block_norm_covariance_pushforward(norm, mu, sigma, cfg.block_mlp_covariance_floor, _head_dims(group, cfg),
                                             mode=cfg.block_norm_covariance,
                                             trace_cap=cfg.block_norm_trace_cap,
                                             cap_moment_fallback=cfg.block_norm_cap_moment_fallback)


def _head_dims(group: GaugeGroup, cfg: VFE3Config) -> Optional[Sequence[int]]:
    r"""The head block sizes a head-block family (``gaussian_block``) keeps, else None."""
    from vfe3.families.base import get_family
    return tuple(group.irrep_dims) if get_family(cfg.family).head_block_covariance else None


def _block_mlp_precision(
    mu: torch.Tensor,                         # (B, N, K) block-MLP input mean
) -> 'tuple[AbstractContextManager, torch.Tensor]':
    r"""The block MLP's precision context and input mean (audit 2026-10-10 round 2, #29).

    Autocast disabled and the mean at ``promote_types(mu.dtype, float32)``, so under ``amp_dtype`` the MLP residual,
    otherwise the only reduced-precision input of the fp32 router, is computed in fp32 (hardcoded 2026-10-12;
    the ``block_mlp_autocast`` toggle that kept the ambient autocast was removed).
    """
    return (torch.autocast(device_type=mu.device.type, enabled=False),
            mu.to(torch.promote_types(mu.dtype, torch.float32)))


def _head_block_readout(
    pushed: torch.Tensor,                     # (B, N, K, K) covariance returned by the block MLP or norm pushforward
    sigma:  torch.Tensor,                     # its input covariance
    group:  GaugeGroup,
    cfg:    VFE3Config,
) -> torch.Tensor:
    r"""The family's readout of a covariance the block MLP or the block norm pushforward produced.

    For a head-block family (``gaussian_block``) a new covariance :math:`\Sigma'` (``delta_full``, the
    prototype ``posterior`` or a ``block_norm_covariance`` pushforward) is read through :func:`vfe3.families.gaussian.head_block_projection`,
    :math:`\bigoplus_h\Sigma'_{hh}`, the exact M-projection :math:`\arg\min_q\mathrm{KL}(\mathcal N(\mu,\Sigma')\|q)` onto
    the family over the heads ``group.irrep_dims``. Its head blocks are principal submatrices of an SPD matrix, so
    they stay SPD. A passthrough covariance (the same object) and every other family return ``pushed`` unchanged.
    """
    if pushed is sigma or pushed.dim() != sigma.dim() or pushed.dim() < 2 or pushed.shape[-1] != pushed.shape[-2]:
        return pushed
    from vfe3.families.base import get_family
    if not get_family(cfg.family).head_block_covariance:
        return pushed
    from vfe3.families.gaussian import head_block_projection
    return head_block_projection(pushed, group.irrep_dims)


def _reference_coordinate_block(
    out:         BeliefState,                              # post-attention belief, local coordinates
    *,
    block_norm:  Optional[Callable[..., torch.Tensor]],
    block_mlp:   Optional[Callable],
    frame:       Optional[CanonicalFrameContext],          # realized U_i: (B, N, K, K) or (B, N, G, d, d)
    capture:     Optional[MStepCapture],
    hidden_bias: Optional[torch.Tensor],                   # (B, N, w) or None
    readout:     Callable[[torch.Tensor], torch.Tensor] = lambda pushed: pushed,   # the family's covariance readout
    push_norm:   bool = False,                             # a block_norm_covariance pushforward (no delta_full MLP)
    covariance_floor: float = 1e-4,                        # block_mlp_covariance_floor of the pushforwards
    norm_covariance: str = "pushforward",                  # push_norm only: the block_norm_covariance mode
    trace_cap:   bool = True,                              # block_norm_trace_cap
    cap_moment_fallback: bool = False,                     # block_norm_cap_moment_fallback
    frame_diagonal: bool = False,                          # gaussian_frame_diagonal: out.sigma holds c_i
) -> BeliefState:
    r"""Norm and MLP in reference coordinates, one conjugation per block (``block_coordinates='reference'``).

    With the realized frame :math:`U_i` (fixed through the stack on the projected route),

    .. math::
        \xi_i = U_i^{-1}\mu_i^{\mathrm{att}},\qquad
        v_i = \mathrm{LN}_{\gamma,b}(\xi_i),\qquad
        \xi_i^+ = v_i + \mathrm{MLP}(v_i),\qquad
        \mu_i^{(\ell+1)} = U_i\xi_i^+,\qquad \Sigma_i^{(\ell+1)} = \Sigma_i^{\mathrm{att}},

    and a Gaussian-template MLP reads the reference variances :math:`c_i = \operatorname{diag}C_i`,
    :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}`. A prototype-agent MLP reads the whole :math:`C_i`
    (blocks ``(B, N, G, d, d)`` for diagonal beliefs on a block frame, else dense ``(B, N, K, K)``,
    :func:`vfe3.model.frame_actions.reference_covariance_compact`), and under
    ``block_mlp_covariance='posterior'`` its :math:`C_i^+` returns as :math:`\Sigma_i^{(\ell+1)} = U_iC_i^+U_i^\top`
    (its diagonal for diagonal beliefs), so that block is invariant under the passive action. Under the
    passive action :math:`(\mu, \Sigma, U) \to (g\mu, g\Sigma g^\top, gU)` the pair :math:`(\xi, C)` is invariant
    and :math:`\mu^{(\ell+1)} \to g\mu^{(\ell+1)}`.

    Covariance transport (``gaussian_full`` and ``gaussian_block``; audit 2026-10-10 round 2, #51): under
    ``push_norm`` the norm also maps the dense :math:`C_i` by
    :func:`vfe3.model.block_mlp.block_norm_covariance_pushforward` at :math:`\xi_i` (mode ``norm_covariance``), and a
    coordinate ``delta_full``
    MLP pushes :math:`C_i` through the composite :math:`J_{\mathrm{MLP}}J_{\mathrm{LN}}` at :math:`\xi_i`, with the norm's
    floor scale and radial term, as in local coordinates. Either result returns as
    :math:`\Sigma_i^{(\ell+1)} = U_iC_i'U_i^\top` through the family's readout, so :math:`\Sigma^{(\ell+1)} \to
    g\Sigma^{(\ell+1)}g^\top` under the passive action; the template and prototype MLPs read the reference covariance
    of the pushed belief the family keeps.
    The frame actions and the MLP run with autocast off (float32, float64 for an escalated frame); the norm keeps
    the ambient autocast. The captured
    ``block_mlp_input``/``block_mlp_output`` stay in local coordinates (:math:`U_iv_i`, :math:`U_i\xi_i^+`).

    Frame-diagonal family (``frame_diagonal``, ``gaussian_frame_diagonal``): ``out.sigma`` is already
    :math:`c_i = \operatorname{diag}C_i`, so every covariance action runs through the identity frame
    (:func:`vfe3.model.frame_actions.identity_frame`): the pushforwards read :math:`C_i = \operatorname{diag}(c_i)` and
    a pushed :math:`C_i'` returns as its diagonal, the family's M-projection in reference coordinates, which the
    passive action leaves fixed. The mean still moves through :math:`U_i`.
    """
    from vfe3.model.block_mlp import (GaussianTemplateBlockMLP, PrototypeAgentBlockMLP,
                                      block_norm_covariance_pushforward, block_norm_linearization,
                                      block_norm_radial_term, block_norm_trace_cap)
    from vfe3.model.frame_actions import (frame_matvec, identity_frame, local_covariance, reference_covariance,
                                          reference_covariance_compact, reference_variances)

    if frame is None:
        raise RuntimeError(
            "block_coordinates='reference' requires the realized relational frame U_i; this block "
            "received none (every reference-coordinate forward goes through forward_beliefs)")
    bd = frame.block_diagonal
    xi = frame_matvec(frame.inverse, out.mu, bd)                             # (B, N, K) xi = U^{-1} mu
    # The frame of the covariance actions: U_i, or the identity when the family stores reference variances.
    cov_frame = identity_frame(frame) if frame_diagonal else frame
    template = isinstance(block_mlp, GaussianTemplateBlockMLP)
    prototype = isinstance(block_mlp, PrototypeAgentBlockMLP)
    delta_full = getattr(block_mlp, "covariance_contract", None) == "delta_full"
    push = push_norm and block_norm is not None and not delta_full
    sigma = out.sigma
    linearization = radial = cap = C = None
    if push or delta_full:
        C = reference_covariance(out.mu, out.sigma, cov_frame)               # (B, N, K, K) invariant C
        if delta_full and block_norm is not None:
            linearization = block_norm_linearization(block_norm, xi)         # at the pre-norm xi
            radial = block_norm_radial_term(block_norm, xi, C)
            cap = (block_norm_trace_cap(block_norm, xi, C, covariance_floor)   # the norm's sphere-bound scale
                   if trace_cap else None)
        if push:                                                             # Sigma' = U C' U^T, C' at the pre-norm xi
            pushed = block_norm_covariance_pushforward(block_norm, xi, C, covariance_floor, mode=norm_covariance,
                                                       trace_cap=trace_cap, cap_moment_fallback=cap_moment_fallback)
            sigma = readout(local_covariance(out.mu, pushed, cov_frame,
                                             diagonal=frame_diagonal).to(out.sigma.dtype))
    c = (C if delta_full                                                     # the composite reads the pre-norm C
         else reference_variances(out.mu, sigma, cov_frame) if template      # diag C (of the pushed belief)
         else reference_covariance_compact(out.mu, sigma, cov_frame) if prototype   # C blocks or dense
         else sigma)                                                         # unread
    if block_norm is not None:
        xi = block_norm(xi, c)                                               # LN acts on xi only
    if capture is not None:
        capture["block_mlp_input"] = out._replace(mu=frame_matvec(frame.forward, xi, bd), sigma=sigma)
    if block_mlp is not None:
        context, xi = _block_mlp_precision(xi)
        if hasattr(block_mlp, "forward_moments"):
            extra = {} if hidden_bias is None else {"hidden_bias": hidden_bias}
            if linearization is not None:
                extra.update(input_jacobian=linearization.jacobian, input_scale=linearization.scale,
                             input_radial=radial, input_cap=cap)
            if capture is not None and "block_mlp_em_rows" in capture:   # block_mlp_em_weight > 0 only
                extra["em_rows"] = capture["block_mlp_em_rows"]      # the templates read (xi, diag C)
            with context:
                result = block_mlp.forward_moments(xi, c, **extra)
            xi = result.mu                                                   # passthrough: covariance kept
            if delta_full:
                sigma = readout(local_covariance(out.mu, result.sigma, cov_frame,  # Sigma' = U C' U^T
                                                 diagonal=frame_diagonal).to(out.sigma.dtype))
            elif prototype and block_mlp.covariance_contract == "posterior":
                sigma = readout(local_covariance(out.mu, result.sigma, cov_frame,  # Sigma^+ = U C^+ U^T
                                                 diagonal=out.sigma.dim() == out.mu.dim()).to(out.sigma.dtype))
        else:
            if hidden_bias is not None:
                raise TypeError("a legacy block-MLP callable cannot take hidden_bias")
            with context:
                xi = block_mlp(xi)
    out = out._replace(mu=frame_matvec(frame.forward, xi, bd), sigma=sigma)  # (B, N, K) mu = U xi^+
    if capture is not None:
        capture["block_mlp_output"] = out
    return out
