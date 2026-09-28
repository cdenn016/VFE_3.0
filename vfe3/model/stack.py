r"""The VFE block stack for VFE_3.0: L blocks with the belief handoff mu_q -> mu_p.

After each block the updated belief becomes (a blend toward) the next block's prior:
mu_p_next = (1 - rho) mu_p + rho mu_q (rho = prior_handoff_rho); sigma_p frozen at the
embedding by default; phi flows through the belief, not the prior. Uncaptured
all-predictive stacks omit this arithmetic when no prior consumer is enabled.

Placement note (audit 2026-06-09 overnight F23): the opt-in head_mixer / cg_coupling run
inside each block after its E-step and before its norm; the opt-in block MLP runs after
that norm. All three transforms precede handoff, so the belief
handed off above is the POST-transform belief — at n_layers > 1 the transforms recurse
into every subsequent block's prior. The manuscript places the mixer in the single
W_O-readout slot (Manuscripts-Theory/GL(K)_attention.tex) and concedes genuine
cross-head capacity at depth > 1, but does not state this per-block prior-handoff
recursion; the pre-transform converged belief and the live prior entering the final
block stay available via the ``capture`` out-param (the M-step self-coupling reads
them from there).
"""

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

if TYPE_CHECKING:
    from vfe3.model.relational_operators import RelationalOperators

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext, EStepGradientRecord, MStepCapture
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.transport import TransportState
from vfe3.free_energy import attention_tau
from vfe3.model.block import _as_coeff, vfe_block


def _can_omit_prior_handoff(
    cfg:                           VFE3Config,
    capture:                       Optional[MStepCapture],
    grad_record:                   Optional[EStepGradientRecord],
    relational_capture:            Optional[dict],
    relational_operators:          Optional["RelationalOperators"],
    relational_operators_by_layer: Optional[Sequence["RelationalOperators"]],
) -> bool:
    """Admit only unobserved predictive stacks with no prior-dependent consumer."""
    from vfe3.model.relational_operators import RelationalOperators

    if (capture is not None or grad_record is not None or relational_capture is not None
            or cfg.mstep_self_coupling_weight != 0.0 or cfg.cg_energy_weight != 0.0
            or cfg.e_step_update != "gradient" or cfg.e_phi_lr != 0.0):
        return False
    layers = (relational_operators_by_layer if relational_operators_by_layer is not None
              else (relational_operators,) * cfg.n_layers)
    return len(layers) == cfg.n_layers and all(
        type(operator) is RelationalOperators and operator.readout_mode == "predictive_content"
        for operator in layers)


def vfe_stack(
    belief:     BeliefState,
    mu_p:       torch.Tensor,             # (N, K) initial prior means
    sigma_p:    torch.Tensor,             # (N, K) initial prior variances
    group:      GaugeGroup,
    cfg:        VFE3Config,

    *,
    log_prior:       Optional[torch.Tensor]                  = None,
    block_norm:      Optional[Callable[..., torch.Tensor]]   = None,      # cached norm instance (None -> off)
    block_mlps:      Optional[Sequence[Callable]]            = None,      # one selected residual transform per block
    block_mlp_frame: Optional[CanonicalFrameContext]         = None,      # shared fixed vertex frame for canonical mode
    head_mixer:      Optional[Callable[..., 'tuple']]        = None,      # opt-in Schur head mixer (None -> off)
    cg_coupling:     Optional[Callable[..., 'tuple']]        = None,      # opt-in CG cross-type coupling (None -> off)
    lambda_beta:     'float | torch.Tensor'                  = 1.0,       # belief-coupling weight (cfg.lambda_beta)
    transport_state: Optional[TransportState]                = None,      # registry-owned trainable transport state
    e_step_gradient: str                                     = "unroll",  # E-step backward estimator
    rope:            Optional[torch.Tensor]                  = None,      # (N, K, K) gauge-RoPE rotation (None -> off)
    rope_on_cov:     bool                                    = False,     # full-gauge: rotate covariance too
    rope_on_value:   bool                                    = True,      # False -> value aggregation uses the un-rotated base
    rope_insertion:  str                                     = "right",   # "right" (pure, folded frame) | "left" (legacy)
    _need_final_covariance: bool                             = True,      # private ordinary loss; earlier handoffs need sigma

    capture:         Optional[MStepCapture]        = None,   # out-param: LAST block's converged belief + live prior
    grad_record:     Optional[EStepGradientRecord] = None,   # diag out-param: LAST block's E-step belief-grad norms (None -> no capture)
    transport_status: Optional[dict]               = None,   # run-sticky covariant-feature status

    prebuilt_transport: Optional[object]       = None,   # internally prebuilt flat transport shared across blocks (valid: e_phi_lr==0 + flat, phi loop-invariant)
    relational_operators: Optional["RelationalOperators"] = None,
    relational_operators_by_layer: Optional[Sequence["RelationalOperators"]] = None,
    relational_frame: Optional[CanonicalFrameContext] = None,
    relational_value_frame: Optional[CanonicalFrameContext] = None,
    relational_capture: Optional[dict] = None,
    block_mlp_hidden_biases: Optional[Sequence[Optional[torch.Tensor]]] = None,   # per layer (B, N, w_l) or None
) -> BeliefState:
    r"""Run L = cfg.n_layers blocks, handing the belief mean off to the next prior.

    ``transport_state`` is the selected registration's complete trainable state mapping; it is empty
    on the pure flat path. ``e_step_gradient`` is the E-step
    backward estimator forwarded to the E-step ('unroll' default keeps the second-order trajectory
    gradient, 'straight_through' detaches the per-iteration tangent; both share the forward value).
    'detach' is handled by the caller's no_grad wrapper, so here it behaves like 'unroll'.
    ``rope`` is the precomputed block-diagonal positional rotation R(theta) (None = off, the pure
    path); ``rope_on_cov`` enables the full-gauge covariance sandwich rotation.
    ``block_mlp_hidden_biases`` (``per_type_hidden_bias``) holds one ``(B, N, w_l)`` slice of the
    per-type table per layer, added to that layer's block-MLP hidden pre-activation."""
    if block_mlp_hidden_biases is not None and len(block_mlp_hidden_biases) != cfg.n_layers:
        raise ValueError(
            f"block_mlp_hidden_biases needs one entry per layer ({cfg.n_layers}), "
            f"got {len(block_mlp_hidden_biases)}")
    rho = cfg.prior_handoff_rho
    rho_s = cfg.prior_handoff_sigma
    need_prior_handoff = not _can_omit_prior_handoff(
        cfg, capture, grad_record, relational_capture,
        relational_operators, relational_operators_by_layer)
    # The model runs the detach estimator under torch.no_grad(), but the outer M-step still learns
    # the prior. Track the same handoff values in a narrow grad-enabled recurrence: the real stack
    # input stays connected, while every E-step output is fixed state for the M-step contract.
    detach_capture = capture is not None and e_step_gradient == "detach"
    capture_mu_p, capture_sigma_p = mu_p, sigma_p
    diagnostic_capture = capture.get("diagnostic") if capture is not None else None
    if diagnostic_capture is not None:
        diagnostic_capture.setdefault("layer_priors", [])
        diagnostic_capture.setdefault("layer_converged", [])
        diagnostic_capture.setdefault("layer_outputs", [])
        diagnostic_capture.setdefault("layer_block_mlp_inputs", [])
        diagnostic_capture.setdefault("layer_block_mlp_outputs", [])
    # Hoist the loop-invariant temperature computation out of the per-layer vfe_block call.
    # attention_tau depends only on the kappa, group.irrep_dims, and device -- all constant across
    # layers -- so computing it once here and passing it as tau avoids L redundant calls.
    tau = attention_tau(_as_coeff(cfg.kappa_beta, belief.mu.device), group.irrep_dims)
    for layer_index in range(cfg.n_layers):
        if diagnostic_capture is not None:
            diagnostic_capture["layer_priors"].append((mu_p, sigma_p))
        tau_b = tau
        if capture is not None and layer_index == cfg.n_layers - 1:
            if detach_capture:
                with torch.enable_grad():
                    capture["final_block_prior"] = (
                        capture_mu_p.clone(),
                        capture_sigma_p.clone(),
                    )
            else:
                capture["final_block_prior"] = (mu_p.clone(), sigma_p.clone())
            # Capture the exact base temperature used by the final block so reflection/two-hop
            # scoring reuses the forward value. The scorer reads it under no_grad.
            capture["final_block_tau"] = tau_b
        hidden_bias = (block_mlp_hidden_biases[layer_index]
                       if block_mlp_hidden_biases is not None else None)
        block_extra = {} if hidden_bias is None else {"hidden_bias": hidden_bias}
        belief = vfe_block(belief, mu_p, sigma_p, group, cfg, log_prior=log_prior,
                           block_norm=block_norm, head_mixer=head_mixer, cg_coupling=cg_coupling,
                           block_mlp=(block_mlps[layer_index] if block_mlps is not None else None),
                           block_mlp_frame=block_mlp_frame,
                           lambda_beta=lambda_beta,
                           relational_operators=(relational_operators_by_layer[layer_index]
                                                 if relational_operators_by_layer is not None
                                                 else relational_operators),
                           relational_frame=relational_frame,
                           relational_value_frame=relational_value_frame,
                           relational_capture=relational_capture,
                           transport_state=transport_state,
                           e_step_gradient=e_step_gradient, rope=rope, rope_on_cov=rope_on_cov,
                           rope_on_value=rope_on_value, rope_insertion=rope_insertion,
                           tau=tau_b,
                           _need_mixer_covariance=(
                               _need_final_covariance or layer_index < cfg.n_layers - 1),
                           capture=capture, grad_record=grad_record,   # each block overwrites; last wins
                           transport_status=transport_status,
                           state_record=(diagnostic_capture.setdefault(
                               "e_step_trace", {"sequence_index": 0})
                                         if diagnostic_capture is not None and layer_index == 0 else None),
                           prebuilt_transport=prebuilt_transport,      # phi is loop-invariant when e_phi_lr==0
                           **block_extra)
        if diagnostic_capture is not None:
            if relational_capture is not None:
                # The shared capture is overwritten by the next block. Preserve each
                # block's actual pre-readout router and covariance diagnostics.
                diagnostic_capture.setdefault("layer_conditional_captures", []).append(dict(relational_capture))
            diagnostic_capture["layer_converged"].append(capture["converged"])
            diagnostic_capture["layer_outputs"].append(belief)
            diagnostic_capture["layer_block_mlp_inputs"].append(capture["block_mlp_input"])
            diagnostic_capture["layer_block_mlp_outputs"].append(capture["block_mlp_output"])
        if need_prior_handoff and layer_index < cfg.n_layers - 1:
            mu_p = (1.0 - rho) * mu_p + rho * belief.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
        if detach_capture and layer_index < cfg.n_layers - 1:
            with torch.enable_grad():
                # Values equal the live recurrence; detaching the inferred state prevents this
                # capture path from backpropagating through the no-grad E-step prefix.
                capture_mu_p = (1.0 - rho) * capture_mu_p + rho * belief.mu.detach()
                capture_sigma_p = (
                    (1.0 - rho_s) * capture_sigma_p
                    + rho_s * belief.sigma.detach()
                )
    return belief
