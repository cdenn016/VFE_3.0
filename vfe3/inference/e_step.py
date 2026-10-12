r"""Projected full-input/full-output attention inference.

Predictive content directly replaces the receiving moments. Full-link gradient
inference freezes incoming routing and source statistics for an episode, retains
their outer graph, and takes Gaussian Fisher steps with the configured trust
regions and SPD retraction. The flat transport builders here also produce the realized frames
the model and its diagnostics read.
"""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from vfe3.model.relational_operators import RelationalOperators

from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext, EStepGradientRecord
from vfe3.gradients.full_link import (
    FULL_LINK_ALPHA_MODES, FullLinkContext, build_full_link_context, full_link_context_at,
    full_link_cost, full_link_cost_parts, full_link_gradients, full_link_minimizer, receiver_law_retry_counter,
)
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.families.base import get_family
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.retraction import retract_spd_affine
from vfe3.numerics import apply_mu_trust_region
from vfe3.relational_scoring import source_recovery_counter
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    TransportState,
    build_factored_transport,
    get_transport_registration,
)


def _transport(
    phi:                torch.Tensor,             # (N, n_gen) or (B, N, n_gen)
    group:              GaugeGroup,

    *,
    transport_mode:         str   = "flat",     # connection-regime registry key (default = flat)
    gauge_mode:             str   = "learned",  # 'learned' | 'trivial' (forwarded to the builder)
    materialize:            bool = True,         # compatibility/diagnostic boundary

    validity_max_norm:      Optional[float]          = None,  # fail-closed chart radius (None -> VFE3Config.transport_chart_max_norm default)
    mu:                     Optional[torch.Tensor]   = None,  # (N,K) or (B,N,K) means
    sigma:                  Optional[torch.Tensor]   = None,  # variances for covariant features
    transport_state:        Optional[TransportState] = None,  # registry-owned trainable state
    mu_key:                 Optional[torch.Tensor]   = None,  # detached key-slot means
    sigma_key:              Optional[torch.Tensor]   = None,  # detached key-slot variances
    right_phi:              Optional[torch.Tensor] = None,  # exact right positional factor
    exactness_out:          Optional[dict]         = None,  # run-sticky feature status sink
) -> 'torch.Tensor | CompactFactoredTransport | FactoredTransport':
    r"""Build the pairwise transport Omega_ij via the connection-regime registry.

    The build is config-selected through ``get_transport(transport_mode)``; 'flat' is the
    phi-cocycle Omega_ij = exp(phi_i) exp(-phi_j) (byte-identical to a direct
    ``compute_transport_operators`` call). ``gauge_mode`` is
    forwarded to the builder (audit 2026-06-10 F6: previously dropped, so the builders always saw
    the 'learned' default).

    Rank-aware: a 2-D (N, n_gen) frame (the unbatched diagnostics / trajectory path) is transported
    as a batch of one and stripped back to (N, N, K, K); a 3-D (B, N, n_gen) frame (the batched
    forward) flows straight through. ``mu``/``mu_key`` are unsqueezed to match so a registered
    belief-dependent builder always sees batched (B, N, K) means."""
    # A None state is the pure (stateless flat) path: normalize once here, since the builders are
    # handed the mapping as keyword arguments and the registration declares its own state keys.
    transport_state = dict(transport_state or {})
    if right_phi is not None and transport_mode == "flat":
        built = build_factored_transport(
            phi, group,
            gauge_mode=gauge_mode,
            validity_max_norm=validity_max_norm,
            right_phi=right_phi,
        )
        return built.to_dense_omega() if materialize else built
    registration = get_transport_registration(transport_mode)
    build = registration.callable
    if phi.dim() == 2:
        mu_b = mu.unsqueeze(0) if mu is not None else None
        mu_kb = mu_key.unsqueeze(0) if mu_key is not None else None
        sig_b = sigma.unsqueeze(0) if sigma is not None else None
        sig_kb = sigma_key.unsqueeze(0) if sigma_key is not None else None
        right_b = right_phi.unsqueeze(0) if right_phi is not None else None
        omega = build(phi.unsqueeze(0), group, gauge_mode=gauge_mode, mu=mu_b, mu_key=mu_kb,
                      sigma=sig_b, sigma_key=sig_kb,
                      materialize=False,
                      validity_max_norm=validity_max_norm, exactness_out=exactness_out,
                      right_phi=right_b,
                      **transport_state)["Omega"]
        # The unbatched compatibility/diagnostic path builds a batch of one -> strip that leading
        # batch axis back off unless the builder already returns (N, N, K, K).
        built = omega if registration.batch_independent else omega[0]
    else:
        built = build(phi, group, gauge_mode=gauge_mode, mu=mu, mu_key=mu_key,
                      sigma=sigma, sigma_key=sigma_key,
                      materialize=False,
                      validity_max_norm=validity_max_norm, exactness_out=exactness_out,
                      right_phi=right_phi,
                      **transport_state)["Omega"]
    return built


def _can_fuse_flat(transport_mode: str, group: GaugeGroup) -> bool:
    r"""Whether the forward belief-transport may skip the dense (B,N,N,K,K) Omega (P0 #2).

    The fused factored route is valid ONLY when the connection is flat
    (its Omega is mu-independent, so it need not stay dense) AND
    the group is genuinely block-diagonal with EQUAL blocks (``len(irrep_dims) > 1`` and all blocks
    the same size -- block_glk / tied_block_glk). Single-block (glk, so_k, cross-coupled block_glk
    all report ``irrep_dims=[K]``) keeps the dense path.
    """
    return (
        transport_mode == "flat"
        and len(group.irrep_dims) > 1
        and len(set(group.irrep_dims)) == 1
    )


def build_belief_transport(
    phi:                torch.Tensor,             # (B, N, n_gen) batched gauge frames
    group:              GaugeGroup,

    *,
    transport_mode:              str                                         = "flat",   # connection-regime registry key
    gauge_mode:                  str                                         = "learned", # 'learned' | 'trivial' (forwarded to the builder)
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    compact_phi_block_transport: bool                                        = False,    # compact head-block factors (equal head-block groups)
    require_vertex_factors:       bool                                        = False,    # exact flat vertex representation even for a single block
    validity_max_norm:           Optional[float]                             = None,     # fail-closed chart radius (None -> VFE3Config.transport_chart_max_norm default)
    mu:                          Optional[torch.Tensor]                      = None,     # belief-dependent builder reads these
    sigma:                       Optional[torch.Tensor]                      = None,     # belief-dependent builder reads these
    transport_state:             Optional[TransportState]                    = None,     # registry-owned trainable state mapping
    mu_key:                      Optional[torch.Tensor]                      = None,     # KEY-slot means (None -> mu)
    sigma_key:                   Optional[torch.Tensor]                      = None,     # KEY-slot variances (None -> sigma)
    rope:                        Optional[torch.Tensor]                      = None,     # (N, K, K) gauge-RoPE rotation (None -> off)
    right_phi:                   Optional[torch.Tensor]                      = None,     # exact right positional factor exp(Y)
    exactness_out:               Optional[dict]                              = None,     # run-sticky covariant-feature status sink
    _content_token_ids:          Optional[torch.Tensor]                      = None,     # private built-in shared-frame lookup
) -> 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport':
    r"""Build the FORWARD belief-transport for one E-step iteration (the hot path, P0 #2).

    On the flat + block-diagonal-with-equal-blocks path (``_can_fuse_flat``) returns a
    :class:`FactoredTransport` -- or, with ``compact_phi_block_transport`` on a group whose generators live
    on equal head blocks, a :class:`CompactFactoredTransport` -- containing per-token exps only, NEVER the dense
    (B,N,N,K,K) Omega. Single-block groups and the trivial gauge fall back to ``_transport``'s
    DENSE Omega; that fallback is REGISTRY-driven (belief tensors gated on the registration's
    ``needs_mu`` / ``needs_sigma`` flags, never on literal mode names).

    When ``rope`` is given the built transport is wrapped in a :class:`RopeTransport` before being
    returned. Every flat build certifies its float32 frame pair and escalates it to float64 when
    the certificate fails (``stable_matrix_exp_pair``).
    """
    if transport_mode == "flat" and (
        require_vertex_factors or right_phi is not None or _can_fuse_flat(transport_mode, group)
    ):
        built = build_factored_transport(phi, group, gauge_mode=gauge_mode,
                                         validity_max_norm=validity_max_norm,
                                         right_phi=right_phi,
                                         _content_token_ids=_content_token_ids,
                                         compact_blocks=compact_phi_block_transport)
    else:
        # Registry-driven routing (audit 2026-07-01 round-3, punch 12b): forward ALL state kwargs
        # once, gating the belief tensors on the registration's flags (the add-by-registering
        # contract) instead of matching literal mode names; builders tolerate unused kwargs.
        registration = get_transport_registration(transport_mode)
        built = _transport(phi, group, transport_mode=transport_mode, gauge_mode=gauge_mode,
                           mu=(mu if registration.needs_mu else None),
                           mu_key=(mu_key if registration.needs_mu else None),
                           sigma=(sigma if registration.needs_sigma else None),
                           sigma_key=(sigma_key if registration.needs_sigma else None),
                           transport_state=transport_state,
                           materialize=False,
                           validity_max_norm=validity_max_norm, right_phi=right_phi,
                           exactness_out=exactness_out)
    if rope is None:
        return built
    return RopeTransport(
        base=built,
        rope=rope,
        insertion=rope_insertion,
        same_frame_flat_cocycle=(
            isinstance(built, (CompactFactoredTransport, FactoredTransport))
            and built.same_frame_flat_cocycle
        ),
    )


def e_step_iteration(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K)
    sigma_p:                   torch.Tensor,        # (N, K)
    group:                     Optional[GaugeGroup],

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    e_q_mu_lr:                 float = 0.1,
    e_q_sigma_lr:              float = 0.1,
    value:                     float = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure)
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 Optional[float] = 100.0,  # matches VFE3Config.sigma_max (None = no ceiling)
    e_sigma_q_trust:           float = 5.0,
    e_mu_q_trust:              Optional[float] = None,   # Mahalanobis mean trust radius (sigma units); None = unbounded
    skip_belief_sigma_update:  bool = False,             # freeze the belief sigma through this iteration (no grad, no retraction)
    family:                    str  = "gaussian_diagonal",
    lambda_alpha_mode:         str  = "constant",
    b0:                        'float | torch.Tensor' = 1.0,   # state-dependent alpha* = c0/(b0 + D)
    c0:                        'float | torch.Tensor' = 1.0,

    log_prior:                 Optional[torch.Tensor]        = None,
    grad_record:               Optional[EStepGradientRecord] = None,   # diag out-param: stashes ||grad_mu/sigma/phi|| (None -> no capture)
    relational_operators: Optional["RelationalOperators"] = None,
    relational_frame:         Optional[CanonicalFrameContext] = None,
    relational_value_frame:   Optional[CanonicalFrameContext] = None,
    relational_capture:       Optional[dict] = None,
    link_state:               Optional[dict] = None,   # content_link_loop_weight > 0: cross-layer link state

    _full_link_context:         Optional[FullLinkContext] = None,
) -> BeliefState:
    r"""Update a projected receiver while retaining its incoming outer graph.

    Predictive content assigns its mean and optional moment-matched covariance.
    Full-link inference either applies the exact frozen-context natural-parameter update
    (``receiver_update='closed_form'``, damping rho) or the explicit step
    :math:`(\Sigma\nabla_\mu F, 2\Sigma\nabla_\Sigma F\Sigma)` through the family Fisher metric
    with an SPD retraction (``'euler'``). The realized frame is fixed throughout the episode, the
    source rows are frozen (filtering), the trajectory is unrolled, and the mean trust region is
    the Mahalanobis ball. A state-dependent ``lambda_alpha_mode`` recomputes alpha*(q) = c0/(b0 + D)
    from the current receiver and the layer prior at every iteration (full_link_context_at): the
    Euler force is then the exact gradient of the profiled energy, and the undamped closed form
    (rho = 1) is its majorize-minimize step; a damped diagonal-family closed form is not guaranteed
    to descend (see full_link_minimizer).
    """
    if (relational_operators is None
            or relational_operators.readout_mode not in ("predictive_content", "full_link_update")):
        raise ValueError(
            "transported and conditional attention are retired; projected inference requires "
            "predictive_content or full_link_update relational operators")
    if relational_frame is None:
        raise ValueError("projected inference requires the realized forward frame")
    full_link = relational_operators.readout_mode == "full_link_update"
    if full_link and (
            lambda_alpha_mode not in FULL_LINK_ALPHA_MODES
            or family not in ("gaussian_diagonal", "gaussian_full")
            or (lambda_alpha_mode == "state_dependent_per_coord" and family != "gaussian_diagonal")):
        raise ValueError("full-link filtering requires Gaussian gradient inference, a constant or "
                         "state-dependent alpha (per coordinate only on gaussian_diagonal), "
                         "and unrolled Fisher steps")

    recovery_before = (source_recovery_counter(belief.mu.device)
                       if relational_capture is not None else None)
    if not full_link:
        moment_matched = relational_operators.predictive_covariance == "moment_matched"
        readout = relational_predictive_readout(
            belief.mu, belief.sigma, relational_frame, relational_operators,
            tau=tau, kl_max=kl_max, log_prior=log_prior,
            value_frame=relational_value_frame, capture=relational_capture,
            **({"return_covariance": True, "floor_anchor": sigma_p} if moment_matched else {}),
            **({"link_state": link_state} if link_state is not None else {}))
        mu, sigma = readout if moment_matched else (readout, belief.sigma)
        if grad_record is not None:
            for role in ("mu", "sigma", "phi"):
                grad_record[role] = None
    else:
        if _full_link_context is None:
            _full_link_context = build_full_link_context(
                belief.mu, belief.sigma, mu_p, sigma_p, relational_frame, relational_operators,
                tau=tau, lambda_beta=lambda_beta, value=value,
                receiver_noise=relational_operators.receiver_noise, kl_max=kl_max,
                value_frame=relational_value_frame, log_prior=log_prior, capture=relational_capture,
                lambda_alpha_mode=lambda_alpha_mode, b0=b0, c0=c0)
            if relational_capture is not None:
                with torch.no_grad():
                    relational_capture["relational_receiver_cost_before"] = full_link_cost(
                        belief.mu, belief.sigma, _full_link_context).detach()
        # Constant alpha: the episode context itself. State-dependent: alpha frozen at this iterate.
        iterate_context = full_link_context_at(_full_link_context, belief.mu, belief.sigma)
        grad_mu, grad_sigma = full_link_gradients(belief.mu, belief.sigma, iterate_context)
        closed_form = relational_operators.receiver_update == "closed_form"
        if grad_record is not None:
            grad_record["mu"] = grad_mu.detach().pow(2).sum().sqrt()
            grad_record["sigma"] = (grad_sigma.detach().pow(2).sum().sqrt()
                                    if grad_sigma is not None else grad_mu.new_zeros(()))
            grad_record["phi"] = grad_mu.new_zeros(())
        if closed_form:
            mu, sigma = full_link_minimizer(
                belief.mu, belief.sigma, iterate_context,
                damping=relational_operators.receiver_damping,
                update_sigma=not skip_belief_sigma_update, eps=eps, sigma_max=sigma_max)
        else:
            nat_mu, nat_sigma = get_family(family)(belief.mu, belief.sigma).natural_gradient(
                grad_mu, grad_sigma, eps=eps)
            delta_mu = e_q_mu_lr * nat_mu
            if e_mu_q_trust is not None:
                delta_mu = apply_mu_trust_region(
                    delta_mu, belief.sigma, trust=e_mu_q_trust,
                    is_diagonal=(belief.sigma.dim() == belief.mu.dim()), eps=eps,
                    family=family)
            mu = belief.mu - delta_mu
            if skip_belief_sigma_update or nat_sigma is None:
                sigma = belief.sigma
            else:
                sigma = retract_spd_affine(
                    belief.sigma, -e_q_sigma_lr * nat_sigma, belief.mu.dim(),
                    trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max)

    if recovery_before is not None:
        relational_capture["relational_covariance_repair_count"] = (
            source_recovery_counter(belief.mu.device) - recovery_before)
    if full_link and relational_capture is not None:
        with torch.no_grad():
            relational_capture["full_link_context"] = _full_link_context
            relational_capture["relational_receiver_cost_after"] = full_link_cost(
                mu, sigma, _full_link_context).detach()
            relational_capture["relational_mean_displacement_norm"] = (mu - belief.mu).norm().detach()
    return belief._replace(mu=mu, sigma=sigma)


def e_step(
    belief:            BeliefState,
    mu_p:              torch.Tensor,        # (N, K)
    sigma_p:           torch.Tensor,        # (N, K)
    group:             Optional[GaugeGroup],

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    e_q_mu_lr:                 float = 0.1,
    e_q_sigma_lr:              float = 0.1,
    n_iter:                    int = 1,
    value:                     float = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 Optional[float] = 100.0,
    e_sigma_q_trust:           float = 5.0,
    e_mu_q_trust:              Optional[float] = None,
    skip_belief_sigma_update:  bool = False,
    family:                    str  = "gaussian_diagonal",
    lambda_alpha_mode:         str  = "constant",
    b0:                        'float | torch.Tensor' = 1.0,   # state-dependent alpha* = c0/(b0 + D)
    c0:                        'float | torch.Tensor' = 1.0,

    grad_record:                 Optional[EStepGradientRecord] = None,   # diag out-param: LAST iteration's belief-grad norms
    state_record:                Optional[dict]                = None,   # diag out-param: every executed receiver iterate
    log_prior:                   Optional[torch.Tensor]        = None,
    relational_operators: Optional["RelationalOperators"] = None,
    relational_frame:         Optional[CanonicalFrameContext] = None,
    relational_value_frame:   Optional[CanonicalFrameContext] = None,
    relational_capture:       Optional[dict] = None,
    link_state:               Optional[dict] = None,   # content_link_loop_weight > 0: cross-layer link state
) -> BeliefState:
    r"""Run projected receiver inference for one incoming-state episode.

    Full-link routing and source-law statistics keep fixed values during receiver
    iterations, while retaining outer gradient credit. State records capture the
    executed beliefs; these projected routes do not expose a global-F trajectory. Only the
    parameters the projected receiver reads are accepted, so a stale knob raises ``TypeError``.
    """
    if (relational_operators is None
            or relational_operators.readout_mode not in ("predictive_content", "full_link_update")):
        raise ValueError(
            "transported and conditional attention are retired; projected inference requires "
            "predictive_content or full_link_update relational operators")
    if relational_frame is None:
        raise ValueError("projected inference requires the realized forward frame")

    full_link_context = None
    episode_start = belief
    episode_recovery = None
    if relational_operators.readout_mode == "full_link_update":
        if relational_capture is not None:
            # Open the repair window before the context build, which runs the routing score.
            episode_recovery = source_recovery_counter(belief.mu.device)
            episode_law_retry = receiver_law_retry_counter(belief.mu.device)
        full_link_context = build_full_link_context(
            belief.mu, belief.sigma, mu_p, sigma_p, relational_frame, relational_operators,
            tau=tau, lambda_beta=lambda_beta, value=value,
            receiver_noise=relational_operators.receiver_noise, kl_max=kl_max,
            value_frame=relational_value_frame, log_prior=log_prior, capture=relational_capture,
            lambda_alpha_mode=lambda_alpha_mode, b0=b0, c0=c0)
        if relational_capture is not None:
            with torch.no_grad():
                relational_capture["relational_receiver_cost_before"] = full_link_cost(
                    belief.mu, belief.sigma, full_link_context).detach()

    if state_record is not None:
        sequence_index = state_record.get("sequence_index")
        state_record.clear()
        state_record.update(
            sequence_index=sequence_index, beliefs=[belief], free_energy=[],
            semantics=relational_operators.inference_semantics)
    for _ in range(n_iter):
        belief = e_step_iteration(
            belief, mu_p, sigma_p, group, tau=tau,
            e_q_mu_lr=e_q_mu_lr, e_q_sigma_lr=e_q_sigma_lr, value=value, lambda_beta=lambda_beta,
            kl_max=kl_max, eps=eps, sigma_max=sigma_max, e_sigma_q_trust=e_sigma_q_trust,
            e_mu_q_trust=e_mu_q_trust, skip_belief_sigma_update=skip_belief_sigma_update,
            family=family, lambda_alpha_mode=lambda_alpha_mode, b0=b0, c0=c0,
            grad_record=grad_record, log_prior=log_prior,
            relational_operators=relational_operators,
            relational_frame=relational_frame,
            relational_value_frame=relational_value_frame,
            relational_capture=relational_capture,
            _full_link_context=full_link_context,
            **({"link_state": link_state} if link_state is not None else {}))
        if state_record is not None:
            state_record["beliefs"].append(belief)
    if full_link_context is not None and relational_capture is not None:
        # Episode-level records: they hold for any n_iter, including zero iterations.
        with torch.no_grad():
            relational_capture["full_link_context"] = full_link_context
            relational_capture["relational_receiver_cost_after"] = full_link_cost(
                belief.mu, belief.sigma, full_link_context).detach()
            relational_capture["relational_covariance_repair_count"] = (
                source_recovery_counter(belief.mu.device) - episode_recovery)
            relational_capture["relational_receiver_law_retry_count"] = (
                receiver_law_retry_counter(belief.mu.device) - episode_law_retry)
            relational_capture["relational_mean_displacement_norm"] = (
                belief.mu - episode_start.mu).norm().detach()
            # Both tags read the one context frozen at the episode-initial alpha (the built context),
            # so the part deltas are deltas of a single functional and the distances share one mu*.
            # Under a state-dependent alpha that functional is the episode-initial majorizer.
            for tag, state in (("before", episode_start), ("after", belief)):
                mean_part, covariance_part, distance = full_link_cost_parts(
                    state.mu, state.sigma, full_link_context)
                relational_capture[f"relational_receiver_mean_cost_{tag}"] = mean_part.detach()
                relational_capture[f"relational_receiver_covariance_cost_{tag}"] = covariance_part.detach()
                relational_capture[f"relational_receiver_target_distance_{tag}"] = distance.detach()
    return belief
