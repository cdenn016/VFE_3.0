r"""Projected full-input/full-output attention inference.

Predictive content directly replaces the receiving moments. Full-link gradient
inference freezes incoming routing and source statistics for an episode, retains
their outer graph, and takes Gaussian Fisher steps with the configured trust
regions and SPD retraction. Transport and energy helpers remain available for
independent geometric diagnostics.
"""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from vfe3.model.relational_operators import RelationalOperators

from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext, EStepGradientRecord
from vfe3.gradients.full_link import (
    FullLinkContext, build_full_link_context, full_link_cost, full_link_cost_parts,
    full_link_gradients, full_link_minimizer,
)
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.families.base import get_family
from vfe3.families.policy import coupling_energy as _coupling_energy, family_instance as _family_instance
from vfe3.free_energy import attention_weights, free_energy, reduced_free_energy, self_divergence_for_alpha
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


_E_STEP_UPDATE_ALIASES = {"gradient": "gradient"}


def canonical_e_step_update(name: str) -> str:
    """Return the supported projected receiver updater."""
    try:
        return _E_STEP_UPDATE_ALIASES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported e_step_update {name!r}; projected attention requires 'gradient'"
        ) from exc


def _transport(
    phi:                torch.Tensor,             # (N, n_gen) or (B, N, n_gen)
    group:              GaugeGroup,

    *,
    transport_mode:         str   = "flat",     # connection-regime registry key (default = flat)
    gauge_mode:             str   = "learned",  # 'learned' | 'trivial' (forwarded to the builder)

    exp_fp64_norm_threshold: float = 5.0,        # 'norm': max clamped block ||M||_F upcast threshold

    clamp_monitor:          bool = False,        # opt-in: warn when the exp Frobenius clamp fires
    transport_mean_per_head: bool = False,       # factored transport_mean contracts per gauge block
    congruence_cond_escalation: bool = False,    # opt-in conditioning trigger for the fp64 congruence escalation
    materialize:            bool = True,         # compatibility/diagnostic boundary

    validity_max_norm:      Optional[float]          = None,  # fail-closed pre-clamp chart bound
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
            exp_fp64_norm_threshold=exp_fp64_norm_threshold,
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
            mean_per_head=transport_mean_per_head,
            cond_escalation=congruence_cond_escalation,
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
                      clamp_monitor=clamp_monitor,
                      exp_fp64_norm_threshold=exp_fp64_norm_threshold,
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
                      clamp_monitor=clamp_monitor,
                      exp_fp64_norm_threshold=exp_fp64_norm_threshold,
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
    clamp_monitor:               bool                                        = False,     # opt-in: warn when the exp Frobenius clamp fires (host sync)
    rope_on_cov:                 bool                                        = False,     # rotate the covariance too (full-gauge)
    rope_on_value:               bool                                        = True,      # False -> value aggregation uses the un-rotated base
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    exp_fp64_norm_threshold:     float                                       = 5.0,      # 'norm': upcast only when max clamped block ||M||_F >= this
    transport_mean_per_head:     bool                                        = False,    # factored transport_mean contracts per gauge block (fused path only)
    congruence_cond_escalation:  bool                                        = False,    # opt-in conditioning trigger for the fp64 congruence escalation
    compact_phi_block_transport: bool                                        = False,    # packed phi factors for canonical flat block_glk
    require_vertex_factors:       bool                                        = False,    # exact flat vertex representation even for a single block
    validity_max_norm:           Optional[float]                             = None,     # opt-in fail-closed pre-clamp chart bound
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
    :class:`FactoredTransport` -- or, for the opt-in canonical block_glk P1 route, a
    :class:`CompactFactoredTransport` -- containing per-token exps only, NEVER the dense
    (B,N,N,K,K) Omega --
    which ``transport_mean`` / ``transport_covariance`` consume on a fused fast path (the mean is an
    exact reassociation; the diagonal covariance factors per head). Single-block / cross-coupled
    groups and the trivial gauge fall back to ``_transport``'s DENSE Omega exactly as before, so
    those numerics are
    unchanged; that fallback is REGISTRY-driven (belief tensors gated on the registration's
    ``needs_mu`` / ``needs_sigma`` flags, never on literal mode names). The factored container flows
    OPAQUELY through ``belief_gradients`` (kernel + oracle), which only ever forward it to
    ``transport_mean`` / ``transport_covariance``; the oracle's full-cov / smoothing routes rebuild
    the dense Omega from the factors on demand (byte-identical).

    When ``rope`` is given the built transport is wrapped in a :class:`RopeTransport` before being
    returned; ``transport_mean`` / ``transport_covariance`` consume it opaquely, so no downstream
    changes are needed.

    Tier-1 perf toggles (default OFF, byte-identical): ``transport_mean_per_head`` marks the
    factored container so ``transport_mean`` contracts per gauge block (fp32 reassociation only;
    inert on the dense fallback, whose mean has no factored form -- the RoPE wrapper recurses into
    the container, so the fused RoPE path is covered). ``exp_fp64_norm_threshold`` controls
    the maximum clamped block norm at which the flat builders compute in float64 on both routes.
    """
    if transport_mode == "flat" and (
        require_vertex_factors or right_phi is not None or _can_fuse_flat(transport_mode, group)
    ):
        built = build_factored_transport(phi, group, gauge_mode=gauge_mode, clamp_monitor=clamp_monitor,
                                         exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                                         mean_per_head=transport_mean_per_head,
                                         cond_escalation=congruence_cond_escalation,
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
                           clamp_monitor=clamp_monitor,
                           exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                           materialize=False,
                           validity_max_norm=validity_max_norm, right_phi=right_phi,
                           exactness_out=exactness_out)
    if rope is None:
        return built
    return RopeTransport(
        base=built,
        rope=rope,
        on_cov=rope_on_cov,
        on_value=rope_on_value,
        insertion=rope_insertion,
        same_frame_flat_cocycle=(
            isinstance(built, (CompactFactoredTransport, FactoredTransport))
            and built.same_frame_flat_cocycle
        ),
    )


def _transport_qk(
    query_phi: torch.Tensor,         # (N, n_gen) current query frames phi_i
    key_phi:   torch.Tensor,         # (N, n_gen) frozen key frames phi_j
    group:     GaugeGroup,

    preserve_factors:  bool                   = False,

    query_right_phi:   Optional[torch.Tensor] = None,
    key_right_phi:     Optional[torch.Tensor] = None,
    validity_max_norm: Optional[float]        = None,
) -> 'torch.Tensor | FactoredTransport':
    r"""Mixed-frame transport for the FILTERED objective: the query frame phi_i is current
    (belief) and the key frame phi_j is frozen (keys). Reduces to ``_transport`` exactly when
    query_phi == key_phi (the global / keys-None case)."""
    query = build_factored_transport(
        query_phi,
        group,
        right_phi=query_right_phi,
        validity_max_norm=validity_max_norm,
    )
    key = build_factored_transport(
        key_phi,
        group,
        right_phi=key_right_phi,
        validity_max_norm=validity_max_norm,
    )
    if not isinstance(query, FactoredTransport) or not isinstance(key, FactoredTransport):
        raise TypeError("mixed-frame transport requires dense FactoredTransport factors")
    # The two builds escalate to float64 independently (audit 2026-09-23 #12); this operator is a
    # mixed-frame product, not a stored inverse pair, so promotion is all it needs.
    common = torch.promote_types(query.exp_phi.dtype, key.exp_neg_phi.dtype)
    exp_q = query.exp_phi.to(common)
    exp_neg_k = key.exp_neg_phi.to(common)
    if preserve_factors:
        return FactoredTransport(
            exp_q,
            exp_neg_k,
            list(group.irrep_dims),
            mean_per_head=query.mean_per_head,
            same_frame_flat_cocycle=False,
            cond_escalation=query.cond_escalation or key.cond_escalation,
        )
    return torch.einsum("ikl,jlm->ijkm", exp_q, exp_neg_k)


def free_energy_value(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K) prior means
    sigma_p:                   torch.Tensor,        # (N, K) prior variances
    group:                     GaugeGroup,

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    renyi_order:               float = 1.0,
    value:                     float = 1.0,
    b0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord for state_dependent_per_coord
    c0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure)
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 Optional[float] = 100.0,  # matches VFE3Config.sigma_max; accepted-and-ignored iteration-only knob
    bch_residual_max:          Optional[float] = None,   # accepted-and-ignored iteration-only knob
    e_sigma_q_trust:           float = 5.0,            # accepted-and-ignored iteration-only knob
    e_mu_q_trust:              Optional[float] = None, # accepted-and-ignored iteration-only knob
    mu_trust_mode:             str  = "box",           # accepted-and-ignored iteration-only knob
    e_step_mu_precond:         str  = "fisher",        # accepted-and-ignored iteration-only knob
    e_step_update:             str  = "gradient",      # accepted-and-ignored iteration-only knob
    mass_phi:                  float = 0.0,            # HONORED: adds (mass_phi/2) ||phi||^2 to the reported F
    mm_damping:                float = 1.0,            # accepted-and-ignored iteration-only knob

    exp_fp64_norm_threshold:   float = 5.0,            # HONORED: forwarded to the global-F _transport float64-island keying

    include_attention_entropy: bool = True,
    skip_belief_sigma_update:  bool = False,           # accepted-and-ignored iteration-only knob
    reuse_pairwise_kl_stats:   bool = False,           # accepted-and-ignored iteration-only knob
    transport_mean_per_head:   bool = False,           # HONORED: factored transport_mean contracts per gauge block
    congruence_cond_escalation:    bool = False,   # opt-in conditioning trigger for the fp64 congruence escalation
    compact_phi_block_transport: bool = False,         # HONORED by the compact diagnostic transport
    rope_on_cov:               bool = False,           # full-gauge: rotate the covariance sandwich too
    rope_on_value:             bool = True,            # False -> value aggregation uses the un-rotated base
    family:                    str  = "gaussian_diagonal",
    full_cov_kl_precision:     Optional[str] = None,
    full_cov_kl_backend:       str  = "pairwise",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    gradient_mode:             str  = "filtering",     # accepted-and-ignored iteration-only knob
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    phi_precond_mode:          str  = "none",          # accepted-and-ignored iteration-only knob
    phi_retract_mode:          str  = "euclidean",     # accepted-and-ignored iteration-only knob
    transport_mode:            str  = "flat",          # HONORED for global F; raises for frozen keys
    clamp_monitor:             bool = False,           # HONORED for the global-F transport build (exp clamp diagnostic)

    rope:                      Optional[torch.Tensor]   = None,   # (N, K, K) gauge-RoPE rotation (None -> off)
    log_prior:                 Optional[torch.Tensor]   = None,
    transport_state:           Optional[TransportState] = None,  # registry-owned trainable state mapping
    keys:                      Optional[BeliefState]    = None,   # None -> global F; else keys frozen at `keys`
    transport_chart_max_norm:  Optional[float]        = None,   # fail-closed pre-clamp chart bound
    transport_status:          Optional[dict]         = None,   # run-sticky covariant-feature status
) -> torch.Tensor:                   # scalar F
    r"""Scalar free energy of a belief. ``keys=None`` -> global F (keys = the belief);
    ``keys`` given -> F with the transported keys frozen at ``keys`` (the F_filt objective).

        F = Sum_i [ alpha_i D(q_i||p_i) (+ R(alpha_i))
                  + Sum_j beta_ij E_ij + tau Sum_j beta_ij log(beta_ij/pi_ij) ],
        E_ij = D(q_i || Omega_ij q_j),  beta = softmax_j(log_prior - E/tau).

    The step-size / E-step-only knobs (gradient_mode, phi_precond_mode, phi_retract_mode,
    sigma_max, bch_residual_max, e_sigma_q_trust, reuse_pairwise_kl_stats) are declared here as
    EXPLICIT accept-and-ignore
    parameters (not a blanket ``**kwargs`` sink) so direct diagnostic callers can reuse
    the config-derived knob bag while a misspelled real parameter still raises
    ``TypeError`` here instead of being silently swallowed. ``transport_mode`` IS honored for the
    global (``keys=None``) F to select its standalone transported objective. ``rope``/``rope_on_cov`` are honored too (audit
    2026-06-09 PP6): the built transport is wrapped in :class:`RopeTransport` exactly as on the
    frame representation, so the scalar evaluates the configured gauge-RoPE objective.
    ``transport_mean_per_head`` is forwarded to the phi-path factored builders. Compact builds retain
    the configured metadata; the legacy noncompact diagnostic seam still materializes its dense
    operator. The exp-island threshold (``exp_fp64_norm_threshold``) is HONORED for
    the global (``keys=None``) F: it is forwarded to the ``_transport`` build so the fp64 island keys
    identically to frame construction (audit PB-12; the two-hop scorer reuses this evaluator,
    so its numerics match the active objective, with the same norm threshold).
    """
    # keys=None -> global F (query = key = belief). keys given -> filtered F: the transport
    # Omega_ij uses the CURRENT query frame phi_i (belief) and the FROZEN key frame phi_j (keys),
    # and the transported key beliefs come from `keys`; only the key side is frozen.
    key_belief = belief if keys is None else keys
    if keys is None:
        # Build Omega under the ACTIVE connection regime so the logged global F matches the objective
        # the beliefs actually descend.
        registration = get_transport_registration(transport_mode)
        transport_kwargs = {
            "transport_mode": transport_mode,
            "right_phi": belief.right_phi,
            "mu": (belief.mu if registration.needs_mu else None),
            "sigma": (belief.sigma if registration.needs_sigma else None),
            "transport_state": transport_state,
            "clamp_monitor": clamp_monitor,
            "transport_mean_per_head": transport_mean_per_head,
            "congruence_cond_escalation": congruence_cond_escalation,
            "exp_fp64_norm_threshold": exp_fp64_norm_threshold,
            "validity_max_norm": transport_chart_max_norm,
            "exactness_out": transport_status,
        }
        if compact_phi_block_transport:
            omega = build_belief_transport(
                belief.phi,
                group,
                compact_phi_block_transport=True,
                **transport_kwargs,
            )
        else:
            omega = _transport(belief.phi, group, **transport_kwargs)
    else:
        omega = _transport_qk(
            belief.phi,
            keys.phi,
            group,
            query_right_phi=belief.right_phi,
            key_right_phi=keys.right_phi,
            validity_max_norm=transport_chart_max_norm,
        )
    fam = get_family(family)
    # Both energies go through the family coupling seam (BeliefParams.coupling_energy), so the logged
    # The standalone F uses the selected family's transported divergence.
    base_omega = None
    if rope is not None:
        # Mirror the model path: R_i Omega_ij R_j^T on the means (and the covariance sandwich
        # under the full gauge). The Rope einsums are rank-agnostic, so no unsqueeze dance.
        effective = RopeTransport(
            base=omega,
            rope=rope,
            on_cov=rope_on_cov,
            on_value=rope_on_value,
            insertion=rope_insertion,
            same_frame_flat_cocycle=(
                keys is None
                and isinstance(omega, (CompactFactoredTransport, FactoredTransport))
                and omega.same_frame_flat_cocycle
            ),
        )
        q_mu, q_sigma = belief.mu, belief.sigma
        k_mu, k_sigma = key_belief.mu, key_belief.sigma
        batched = False
        if not rope_on_value:
            # Diagnostic fidelity: log the DECOUPLED F the beliefs actually descend -- beta from the
            # rotated SCORE energy, coupling sum from the UN-rotated base VALUE energy.
            base_omega = effective.base
    else:
        # Unbatched diagnostic transport: lift every argument to a leading batch axis exactly as the
        # transported moments were lifted before, and drop it again after the energy.
        effective = omega.unsqueeze(0)
        q_mu, q_sigma = belief.mu.unsqueeze(0), belief.sigma.unsqueeze(0)
        k_mu, k_sigma = key_belief.mu.unsqueeze(0), key_belief.sigma.unsqueeze(0)
        batched = True
    sd = self_divergence_for_alpha(
                                   _family_instance(belief.mu, belief.sigma, fam, full_cov_kl_precision),
                                   _family_instance(mu_p, sigma_p, fam, full_cov_kl_precision), alpha=renyi_order, kl_max=kl_max,
                                   eps=eps, divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    alpha, reg = self_coupling_alpha(sd, value=value, mode=lambda_alpha_mode, b0=b0, c0=c0)
    energy = _coupling_energy(fam, q_mu, q_sigma, k_mu, k_sigma, effective,
                               log_prior=log_prior,
                               precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
                               alpha=renyi_order, kl_max=kl_max, eps=eps,
                               divergence_family=divergence_family, irrep_dims=group.irrep_dims)
    coupling_energy = None
    if base_omega is not None:
        coupling_energy = _coupling_energy(
            fam, q_mu, q_sigma, k_mu, k_sigma, base_omega,
            log_prior=log_prior,
            precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
            alpha=renyi_order, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family, irrep_dims=group.irrep_dims)
    if batched:
        energy = energy[0]
    F = free_energy(
        sd, energy, alpha, tau=tau, lambda_beta=lambda_beta,
        include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if lambda_alpha_mode != "constant" else None),
        coupling_energy=coupling_energy,
    )
    # D-08: mass_phi was accepted-and-ignored here, so a mass_phi>0 sweep logged an F that omitted
    # the very term the phi substep descends (phi_alignment_loss:mass). Same expression and same
    # per-sequence sum as that substep, so the reported F and the descended objective agree.
    if mass_phi > 0.0 and getattr(belief, "phi", None) is not None:
        F = F + 0.5 * mass_phi * (belief.phi ** 2).sum()
    return F


def _detach_key_frames(
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',
) -> 'CompactFactoredTransport | FactoredTransport | RopeTransport':
    r"""Receiver-live, sender-detached copy of a vertex-factored flat transport.

    For :math:`\Omega_{ij} = U_i U_j^{-1}` the receiver factor :math:`U_i` keeps its graph and
    the sender factor :math:`U_j^{-1}` is detached, so any energy built from the copy
    differentiates to the receiver-row derivative only. Values are bit-identical. The copy is no
    longer ONE differentiable vertex table, so it is marked non-cocycle: a consumer that exploits
    :math:`\Omega_{ij}^{-1} = \Omega_{ji}` or a common frame would otherwise route the derivative
    to the wrong endpoint, and the certified self-link restore no longer applies (the self pair
    carries only roundoff, and its first-order energy change vanishes because every divergence is
    minimized at :math:`\Omega_{ii} = I`).
    """
    if isinstance(omega, RopeTransport):
        return RopeTransport(
            base=_detach_key_frames(omega.base),
            rope=omega.rope,
            on_cov=omega.on_cov,
            on_value=omega.on_value,
            insertion=omega.insertion,
            same_frame_flat_cocycle=False,
        )
    if isinstance(omega, CompactFactoredTransport):
        return CompactFactoredTransport(
            omega.exp_blocks, omega.inv_blocks.detach(), omega.K,
            mean_per_head=omega.mean_per_head,
            same_frame_flat_cocycle=False,
            cond_escalation=omega.cond_escalation,
        )
    if isinstance(omega, FactoredTransport):
        return FactoredTransport(
            exp_phi=omega.exp_phi,
            exp_neg_phi=omega.exp_neg_phi.detach(),
            irrep_dims=omega.irrep_dims,
            mean_per_head=omega.mean_per_head,
            same_frame_flat_cocycle=False,
            cond_escalation=omega.cond_escalation,
        )
    raise TypeError(
        "the filtering phi objective detaches the sender frames of a vertex-factored flat "
        f"transport; got {type(omega).__name__!r}, which exposes no per-vertex factors")


def phi_alignment_loss(
    mu:        torch.Tensor,             # (N, K)
    sigma:     torch.Tensor,             # (N, K)
    phi:       torch.Tensor,             # (N, n_gen) -- the differentiated variable
    group:     GaugeGroup,

    *,
    tau:       'float | torch.Tensor' = 1.0,
    renyi_order: float = 1.0,
    kl_max:    float = 100.0,
    eps:       float = 1e-6,
    mass_phi:  float = 0.0,
    lambda_beta: 'float | torch.Tensor' = 1.0,        # weight on the belief-coupling block (1.0 = pure)
    family:    str   = "gaussian_diagonal",
    full_cov_kl_precision: Optional[str] = None,
    full_cov_kl_backend:       str  = "pairwise",
    divergence_family: str = "renyi",

    include_attention_entropy: bool  = True,
    transport_mode:            str   = "flat",        # connection-regime registry key (default flat)
    clamp_monitor:             bool  = False,         # opt-in: warn when the exp Frobenius clamp fires
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    exp_fp64_norm_threshold:       float = 5.0,
    transport_mean_per_head:       bool  = False,
    congruence_cond_escalation:    bool = False,   # opt-in conditioning trigger for the fp64 congruence escalation
    compact_phi_block_transport:   bool  = False,
    rope_on_cov:                   bool  = False,  # gauge-RoPE: rotate the covariance sandwich too
    rope_on_value:                 bool  = True,   # False -> value aggregation uses the un-rotated base
    detach_key_frames:             bool  = False,  # filtering: receiver-row phi gradient only

    rope:                     Optional[torch.Tensor]   = None,  # (N,K,K) gauge-RoPE rotation (None -> off)
    right_phi:                Optional[torch.Tensor]   = None,  # exact right positional factor exp(Y)
    log_prior:                Optional[torch.Tensor]   = None,
    transport_state:          Optional[TransportState] = None,  # registry-owned trainable state mapping
    transport_chart_max_norm: Optional[float]          = None,  # fail-closed pre-clamp chart bound
    transport_status:         Optional[dict]           = None,  # run-sticky covariant-feature status
) -> torch.Tensor:
    r"""Canonical belief-coupling block as a function of phi (mu, sigma fixed), plus the
    gauge-frame penalty (manuscript Algorithm 1, line for nabla_phi F):

        L(phi) = lambda_beta Sum_ij [ beta_ij E_ij + tau beta_ij log(beta_ij/pi_ij) ] + (mass_phi/2) ||phi||^2,
        E_ij = D(q_i || Omega_ij(phi) q_j),  beta = softmax_j(log_prior - E/tau).
    Both roles of phi flow (Omega_ij depends on phi_i and phi_j); autograd gives the envelope
    phi-gradient. ``lambda_beta`` (1.0 = pure) scales the coupling block but NOT the ``mass_phi``
    penalty, so the effective phi step is e_phi_lr * lambda_beta * nabla (lambda_beta and e_phi_lr
    interact). The ``mass_phi`` term makes the phi E-step descend the PENALIZED
    objective during inference (distinct from the outer M-step ||phi||^2 on the learned prior
    table). The coherent canonical branch reuses ``reduced_free_energy``, the -tau log Z envelope
    form. When RoPE is present with ``on_value=False``, beta instead comes from the rotated score
    energy while the coupling sum uses the unrotated base-transport value energy; beta remains live
    because it is not stationary for that decoupled block.

    ``detach_key_frames=True`` is the filtering objective (audit 2026-09-23 #6): the sender frames
    :math:`U_j` of :math:`\Omega_{ij} = U_i U_j^{-1}` are detached, so
    :math:`\partial L / \partial \phi_i = \partial F_i / \partial \phi_i` is receiver row
    :math:`i` alone, and under a causal prior no later row :math:`i > j` reaches :math:`\phi_j`.
    The value of ``L`` is unchanged. A ``'coboundary'`` family cancels the frames exactly, so its
    energy carries no frame dependence and its certified transport is kept.
    """
    # Build the standalone alignment transport under the selected connection regime.
    # Only phi varies in this objective. The
    # factored-when-fusable dispatch (audit 2026-06-09 PE3) builds the per-token exps FROM THE
    # LIVE phi leaf -- the factored container is differentiable in phi, so the envelope
    # phi-gradient is preserved while the per-iteration dense (N,N,K,K) Omega disappears on the
    # flat equal-blocks path.
    compact_phi_block_transport = (
        compact_phi_block_transport
        and transport_mode == "flat"
        and group.phi_coordinate_layout == "block_head_row_major"
    )
    fam = get_family(family)
    detach_key_frames = detach_key_frames and fam.transport_requirement != "coboundary"
    omega = build_belief_transport(phi, group, transport_mode=transport_mode, mu=mu, sigma=sigma,
                                   right_phi=right_phi,
                                   transport_state=transport_state,
                                   clamp_monitor=clamp_monitor,
                                   exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                                   transport_mean_per_head=transport_mean_per_head,
                                   congruence_cond_escalation=congruence_cond_escalation,
                                   compact_phi_block_transport=compact_phi_block_transport,
                                   require_vertex_factors=detach_key_frames,
                                   validity_max_norm=transport_chart_max_norm,
                                   exactness_out=transport_status,
                                   rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value, rope_insertion=rope_insertion)
    if detach_key_frames:
        omega = _detach_key_frames(omega)
    # The family coupling seam owns the whole transport + divergence composition (see
    # BeliefParams.coupling_energy), so this standalone objective follows the selected family,
    # including one that evaluates the transported divergence exactly.
    score_energy = _coupling_energy(
        fam, mu, sigma, mu, sigma, omega, precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
        log_prior=log_prior,
        alpha=renyi_order, kl_max=kl_max, eps=eps, divergence_family=divergence_family,
        irrep_dims=group.irrep_dims)
    mass = 0.5 * mass_phi * (phi ** 2).sum() if mass_phi > 0.0 else 0.0
    # value_energy is the coupling-sum energy grid: the SCORE energy on the coherent default, the
    # UN-rotated base VALUE energy under the decoupled RoPE gauge. Both the base coupling block and
    # the two-hop block sum over it (mirroring free_energy's coupling_energy selection).
    value_energy = score_energy
    has_decoupled_value = isinstance(omega, RopeTransport) and not omega.on_value
    if has_decoupled_value:
        value_energy = _coupling_energy(
            fam, mu, sigma, mu, sigma, omega.base, precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
            log_prior=log_prior,
            alpha=renyi_order, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family, irrep_dims=group.irrep_dims)
        zero = score_energy.new_zeros(score_energy.shape[:-1])
        base = free_energy(
            zero, score_energy, zero,
            tau=tau, lambda_beta=lambda_beta,
            include_attention_entropy=include_attention_entropy,
            log_prior=log_prior, coupling_energy=value_energy,
        )
    elif include_attention_entropy:
        base = lambda_beta * reduced_free_energy(score_energy, tau=tau, log_prior=log_prior).sum()
    else:
        beta = attention_weights(score_energy, tau=tau, log_prior=log_prior)
        base = lambda_beta * (beta * score_energy).sum()
    return base + mass


def e_step_iteration(
    belief:                    BeliefState,
    mu_p:                      torch.Tensor,        # (N, K)
    sigma_p:                   torch.Tensor,        # (N, K)
    group:                     GaugeGroup,

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    e_q_mu_lr:                 float = 0.1,
    e_q_sigma_lr:              float = 0.1,
    e_phi_lr:                  float = 0.0,
    renyi_order:               float = 1.0,
    value:                     float = 1.0,
    b0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord for state_dependent_per_coord
    c0:                        'float | torch.Tensor' = 1.0,   # scalar, or (K,) per-coord
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure)
    kl_max:                    float = 100.0,
    eps:                       float = 1e-6,
    sigma_max:                 Optional[float] = 100.0,  # matches VFE3Config.sigma_max (None = no ceiling)
    bch_residual_max:          Optional[float] = None,   # opt-in fail-closed BCH residual gate on the phi retraction
    e_sigma_q_trust:           float = 5.0,
    e_mu_q_trust:              Optional[float] = None,   # mean trust radius (sigma units); None = unbounded
    mu_trust_mode:             str  = "box",             # "box" | "ball" (only when e_mu_q_trust is not None)
    e_step_mu_precond:         str  = "fisher",          # "fisher" (nat-grad mean) | "raw" (B3/EXP-14 mu-arm)
    e_step_update:             str  = "gradient",        # projected receiver updater
    mass_phi:                  float = 0.0,
    mm_damping:                float = 1.0,              # inactive compatibility field in the shared config bag

    include_attention_entropy: bool = True,
    skip_belief_sigma_update:  bool = False,             # freeze the belief sigma through this iteration (no grad, no retraction)
    reuse_pairwise_kl_stats:   bool = False,             # opt-in P3 statistics reuse
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    full_cov_kl_precision:     Optional[str] = None,
    full_cov_kl_backend:       str  = "pairwise",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    phi_precond_mode:          str  = "none",
    phi_retract_mode:          str  = "euclidean",
    transport_mode:            str  = "flat",
    e_step_gradient:           str  = "unroll",               # projected receiver backward estimator
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    oracle_unroll_grad:        bool = False,                  # opt-in: oracle returns a differentiable grad (unroll)
    clamp_monitor:             bool = False,                   # opt-in: warn when the exp Frobenius clamp fires (host sync)
    exp_fp64_norm_threshold:       float = 5.0,    # Tier-1 'norm': max clamped block ||M||_F upcast threshold
    transport_mean_per_head:       bool  = False,  # Tier-1: factored transport_mean contracts per gauge block
    congruence_cond_escalation:    bool = False,   # opt-in conditioning trigger for the fp64 congruence escalation
    compact_phi_block_transport:   bool  = False,  # P1: packed canonical flat block_glk phi factors

    log_prior:                 Optional[torch.Tensor]        = None,
    transport_state:           Optional[TransportState]      = None,   # registry-owned trainable state mapping
    rope:                      Optional[torch.Tensor]        = None,   # (N, K, K) gauge-RoPE rotation
    rope_on_cov:               bool                          = False,  # full-gauge: rotate covariance too
    rope_on_value:             bool                          = True,   # False -> value aggregation uses the un-rotated base
    grad_record:               Optional[EStepGradientRecord] = None,   # diag out-param: stashes ||grad_mu/sigma/phi|| (None -> no capture)
    transport_chart_max_norm: Optional[float]               = None,   # fail-closed pre-clamp chart bound
    transport_status:         Optional[dict]                = None,   # run-sticky covariant-feature status
    relational_operators: Optional["RelationalOperators"] = None,
    relational_frame:         Optional[CanonicalFrameContext] = None,
    relational_value_frame:   Optional[CanonicalFrameContext] = None,
    relational_capture:       Optional[dict] = None,

    _full_link_context:         Optional[FullLinkContext] = None,
) -> BeliefState:
    r"""Update a projected receiver while retaining its incoming outer graph.

    Predictive content assigns its mean and optional moment-matched covariance.
    Full-link inference either applies the exact frozen-context natural-parameter update
    (``receiver_update='closed_form'``, damping rho) or the explicit step
    :math:`(\Sigma\nabla_\mu F, 2\Sigma\nabla_\Sigma F\Sigma)` through the family Fisher metric
    with an SPD retraction (``'euler'``). The realized frame is fixed throughout the episode.
    """
    e_step_update = canonical_e_step_update(e_step_update)
    if (relational_operators is None
            or relational_operators.readout_mode not in ("predictive_content", "full_link_update")):
        raise ValueError(
            "transported and conditional attention are retired; projected inference requires "
            "predictive_content or full_link_update relational operators")
    if relational_frame is None:
        raise ValueError("projected inference requires the realized forward frame")
    if (e_phi_lr != 0.0 or gradient_mode != "filtering" or e_step_gradient != "unroll"
            or transport_mode != "flat"):
        raise ValueError("projected inference requires fixed flat frames, filtering, and unroll")
    full_link = relational_operators.readout_mode == "full_link_update"
    if full_link and (
            lambda_alpha_mode != "constant" or e_step_mu_precond != "fisher"
            or family not in ("gaussian_diagonal", "gaussian_full")):
        raise ValueError("full-link filtering requires Gaussian gradient inference, constant alpha, "
                         "and unrolled Fisher steps")

    recovery_before = (source_recovery_counter(belief.mu.device)
                       if relational_capture is not None else None)
    if not full_link:
        moment_matched = relational_operators.predictive_covariance == "moment_matched"
        readout = relational_predictive_readout(
            belief.mu, belief.sigma, relational_frame, relational_operators,
            tau=tau, kl_max=kl_max, log_prior=log_prior,
            value_frame=relational_value_frame, capture=relational_capture,
            **({"return_covariance": True} if moment_matched else {}))
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
                value_frame=relational_value_frame, log_prior=log_prior, capture=relational_capture)
            if relational_capture is not None:
                with torch.no_grad():
                    relational_capture["relational_receiver_cost_before"] = full_link_cost(
                        belief.mu, belief.sigma, _full_link_context).detach()
        grad_mu, grad_sigma = full_link_gradients(belief.mu, belief.sigma, _full_link_context)
        closed_form = relational_operators.receiver_update == "closed_form"
        if grad_record is not None:
            grad_record["mu"] = grad_mu.detach().pow(2).sum().sqrt()
            grad_record["sigma"] = (grad_sigma.detach().pow(2).sum().sqrt()
                                    if grad_sigma is not None else grad_mu.new_zeros(()))
            grad_record["phi"] = grad_mu.new_zeros(())
        if closed_form:
            mu, sigma = full_link_minimizer(
                belief.mu, belief.sigma, _full_link_context,
                damping=relational_operators.receiver_damping,
                update_sigma=not skip_belief_sigma_update, eps=eps, sigma_max=sigma_max)
        else:
            nat_mu, nat_sigma = get_family(family)(belief.mu, belief.sigma).natural_gradient(
                grad_mu, grad_sigma, eps=eps)
            delta_mu = e_q_mu_lr * nat_mu
            if e_mu_q_trust is not None:
                delta_mu = apply_mu_trust_region(
                    delta_mu, belief.sigma, trust=e_mu_q_trust, mode=mu_trust_mode,
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
    group:             GaugeGroup,

    *,
    tau:                       'float | torch.Tensor' = 1.0,

    e_q_mu_lr:                 float = 0.1,
    e_q_sigma_lr:              float = 0.1,
    e_phi_lr:                  float = 0.0,
    exp_fp64_norm_threshold:   float = 5.0,       # 'norm': max clamped block ||M||_F upcast threshold

    n_iter:                    int = 1,

    e_step_gradient:           str = "unroll",
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    # The model realizes frames before this projected receiver call.

    return_trajectory:           bool = False,
    oracle_unroll_grad:          bool = False,    # inactive shared-knob compatibility field
    transport_mean_per_head:     bool = False,    # factored transport_mean contracts per gauge block
    congruence_cond_escalation:    bool = False,   # opt-in conditioning trigger for the fp64 congruence escalation
    compact_phi_block_transport: bool = False,    # packed canonical flat block_glk phi factors
    rope_on_cov:                 bool = False,
    rope_on_value:               bool = True,

    grad_record:                 Optional[EStepGradientRecord] = None,   # diag out-param: LAST iteration's belief-grad norms
    state_record:                Optional[dict]                = None,   # diag out-param: every executed receiver iterate
    rope:                        Optional[torch.Tensor]        = None,
    log_prior:                   Optional[torch.Tensor]        = None,
    transport_chart_max_norm:   Optional[float]               = None,
    transport_status:           Optional[dict]                = None,
    relational_operators: Optional["RelationalOperators"] = None,
    relational_frame:         Optional[CanonicalFrameContext] = None,
    relational_value_frame:   Optional[CanonicalFrameContext] = None,
    relational_capture:       Optional[dict] = None,

    prebuilt_transport: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport | None' = None,   # caller-prebuilt forward transport for the eligible fixed-flat route
    **kwargs,
) -> BeliefState:
    r"""Run projected receiver inference for one incoming-state episode.

    Full-link routing and source-law statistics keep fixed values during receiver
    iterations, while retaining outer gradient credit. State records capture the
    executed beliefs; these projected routes do not expose a global-F trajectory.
    """
    if (relational_operators is None
            or relational_operators.readout_mode not in ("predictive_content", "full_link_update")):
        raise ValueError(
            "transported and conditional attention are retired; projected inference requires "
            "predictive_content or full_link_update relational operators")
    canonical_e_step_update(kwargs.get("e_step_update", "gradient"))
    if return_trajectory:
        raise ValueError("Global-F trajectories are unavailable for projected attention")
    if relational_frame is None:
        raise ValueError("projected inference requires the realized forward frame")

    full_link_context = None
    episode_start = belief
    episode_recovery = None
    if relational_operators.readout_mode == "full_link_update":
        if relational_capture is not None:
            # Open the repair window before the context build, which runs the routing score.
            episode_recovery = source_recovery_counter(belief.mu.device)
        full_link_context = build_full_link_context(
            belief.mu, belief.sigma, mu_p, sigma_p, relational_frame, relational_operators,
            tau=tau, lambda_beta=kwargs.get("lambda_beta", 1.0), value=kwargs.get("value", 1.0),
            receiver_noise=relational_operators.receiver_noise, kl_max=kwargs.get("kl_max", 100.0),
            value_frame=relational_value_frame, log_prior=log_prior, capture=relational_capture)
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
            e_q_mu_lr=e_q_mu_lr, e_q_sigma_lr=e_q_sigma_lr, e_phi_lr=e_phi_lr,
            e_step_gradient=e_step_gradient, oracle_unroll_grad=oracle_unroll_grad,
            exp_fp64_norm_threshold=exp_fp64_norm_threshold,
            transport_mean_per_head=transport_mean_per_head,
            congruence_cond_escalation=congruence_cond_escalation,
            compact_phi_block_transport=compact_phi_block_transport,
            transport_chart_max_norm=transport_chart_max_norm,
            transport_status=transport_status,
            grad_record=grad_record,
            log_prior=log_prior, rope=rope, rope_on_cov=rope_on_cov,
            rope_on_value=rope_on_value, rope_insertion=rope_insertion,
            relational_operators=relational_operators,
            relational_frame=relational_frame,
            relational_value_frame=relational_value_frame,
            relational_capture=relational_capture,
            _full_link_context=full_link_context,
            **kwargs)
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
            relational_capture["relational_mean_displacement_norm"] = (
                belief.mu - episode_start.mu).norm().detach()
            for tag, state in (("before", episode_start), ("after", belief)):
                mean_part, covariance_part, distance = full_link_cost_parts(
                    state.mu, state.sigma, full_link_context)
                relational_capture[f"relational_receiver_mean_cost_{tag}"] = mean_part.detach()
                relational_capture[f"relational_receiver_covariance_cost_{tag}"] = covariance_part.detach()
                relational_capture[f"relational_receiver_target_distance_{tag}"] = distance.detach()
    return belief
