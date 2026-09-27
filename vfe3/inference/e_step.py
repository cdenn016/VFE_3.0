r"""The E-step for VFE_3.0: an iterative natural-gradient descent on F over the
Gaussian belief (mu, sigma, phi).

One inner iteration (all positions in parallel, updates sequential):
  transport Omega(phi) -> belief_gradients (envelope kernel / oracle) -> Fisher
  preconditioner -> retract mu (Euclidean) + sigma (SPD) -> phi (autograd of the
  canonical belief-coupling block -> precondition -> Lie retraction).
Decoupled learning rates and trust regions. Parallel mean-field updates are not
guaranteed monotone per iteration; F-descent holds as a DIRECTION property
(filtering descends F_filt; smoothing and the phi step descend global F).
"""

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from vfe3.model.relational_operators import RelationalOperators

from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext, EStepGradientRecord
from vfe3.gradients.qkv import qkv_conditional_cost
from vfe3.gradients.relational import compute_relational_filtering_gradients, relational_predictive_readout
from vfe3.families.base import get_family
from vfe3.families.policy import coupling_energy as _coupling_energy, family_instance as _family_instance
from vfe3.families.gaussian import FullGaussian
from vfe3.free_energy import attention_weights, free_energy, reduced_free_energy, self_divergence_for_alpha
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import _bch_chart_direction, precondition_phi_gradient
from vfe3.geometry.retraction import retract_phi, retract_spd_affine
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
from vfe3.gradients.kernels import (
    belief_gradients, mm_damped_precision_blend_full, mm_exact_update, uses_kernel_route,
)


_E_STEP_UPDATE_ALIASES = {
    "gradient": "gradient",
    "mm_exact": "mm_exact",
}


def canonical_e_step_update(name: str) -> str:
    r"""Return the executable updater name while preserving config serialization.

    ``mm_exact`` names the closed-form minimizer of the frozen-attention,
    strict-pair-masked surrogate; it does not claim a fixed point of the live
    self-consistent E-step.
    """
    try:
        return _E_STEP_UPDATE_ALIASES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_E_STEP_UPDATE_ALIASES))
        raise ValueError(f"unknown e_step_update {name!r}; expected one of: {choices}") from exc


def _belief_update_consumes_omega(
    e_step_update: str,
    relational_operators: Optional["RelationalOperators"] = None,
) -> bool:
    r"""True when the belief-update branch reads the forward transport ``omega``.

    ``mm_exact`` consumes ``omega`` in ``mm_exact_update``. The ``predictive_content``
    readout replaces ``mu`` (and ``sigma`` under ``predictive_covariance='moment_matched'``,
    otherwise it keeps ``sigma``) and never reads ``omega``. The filtering
    and ``belief_gradients`` branches consume ``omega`` / ``omega_builder``. The phi
    sub-step builds its own transport and is unaffected either way.
    """
    if e_step_update == "mm_exact":
        return True
    if (relational_operators is not None
            and relational_operators.readout_mode == "predictive_content"):
        return False
    return True


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
    compile_pair_kernel:       bool = False,           # accepted-and-ignored iteration-only knob
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
    parameters (not a blanket ``**kwargs`` sink) so the common ``e_step`` call site may forward one knob
    bag to both this and ``e_step_iteration`` while a MISSPELLED real parameter still raises
    ``TypeError`` here instead of being silently swallowed. ``transport_mode`` IS honored for the
    global (``keys=None``) F so the logged trajectory matches
    the objective the beliefs descend. ``rope``/``rope_on_cov`` are honored too (audit
    2026-06-09 PP6): the built transport is wrapped in :class:`RopeTransport` exactly as on the
    model path, so under gauge-RoPE the logged F is the F being descended, not the un-rotated one.
    ``transport_mean_per_head`` is forwarded to the phi-path factored builders. Compact builds retain
    the configured metadata; the legacy noncompact diagnostic seam still materializes its dense
    operator. The exp-island threshold (``exp_fp64_norm_threshold``) is HONORED for
    the global (``keys=None``) F: it is forwarded to the ``_transport`` build so the fp64 island keys
    identically to the model E-step (audit PB-12; the two-hop scorer reuses this evaluator,
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
    # F is built from the SAME transported divergence the beliefs descend under any family.
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
    # Build Omega under the ACTIVE connection regime so the phi step descends the SAME objective as
    # the mu/sigma step. The transport is held constant w.r.t. the phi gradient (only phi varies). The
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
    # BeliefParams.coupling_energy), so the phi step descends the SAME energy the belief step does
    # under any family, including one that evaluates the transported divergence exactly.
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
    e_phi_lr:                  float = 0.1,
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
    e_step_update:             str  = "gradient",        # "gradient" (pure) | "mm_exact" (closed-form MM fusion)
    mass_phi:                  float = 0.0,
    mm_damping:                float = 1.0,              # mm_exact: natural-coordinate damping eta (1 = exact minimizer)

    include_attention_entropy: bool = True,
    skip_belief_sigma_update:  bool = False,             # freeze the belief sigma through this iteration (no grad, no retraction)
    compile_pair_kernel:       bool = False,             # torch.compile the closed-form pair kernel (lazy cache)
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
    e_step_gradient:           str  = "unroll",               # backward estimator: unroll | straight_through | detach
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

    _prebuilt_omega:           'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport | None' = None,   # PRIVATE: forward-transport cache from e_step
) -> BeliefState:
    r"""One inner E-step iteration: mu, sigma (Fisher natgrad + SPD retraction) then phi
    (autograd of the alignment block + preconditioner + Lie retraction).

    The belief-transport build is config-selected through the connection-regime registry
    (``transport_mode``); 'flat' is the phi-cocycle (byte-identical). Because the flat Omega depends
    only on ``belief.phi`` it is rebuilt from the CURRENT frame each iteration here.

    ``e_step_update='mm_exact'`` (kernel route only) replaces the mu/sigma gradient + retraction
    with the closed-form precision fusion at frozen beta (``mm_exact_update``), damped by
    ``mm_damping`` in natural coordinates; the phi sub-step is untouched either way."""
    transport_state = dict(transport_state or {})   # None = the pure stateless flat path
    e_step_update = canonical_e_step_update(e_step_update)
    compact_phi_block_transport = (
        compact_phi_block_transport
        and transport_mode == "flat"
        and group.phi_coordinate_layout == "block_head_row_major"
    )
    # Build the forward belief-transport (P0 #2): on the flat + block-diagonal-with-equal-blocks
    # path this is a FactoredTransport (the per-token exps only, NO dense (B,N,N,K,K) Omega), which
    # the belief-gradient kernel / oracle consume opaquely through transport_mean / covariance;
    # single-block / cross-coupled groups keep the dense Omega exactly as before.
    #
    # A registration whose Omega depends on mu or sigma is NOT pre-built here. The hand kernel is a
    # frozen-transport gradient, so every belief-dependent registration routes to the autograd
    # oracle, which must rebuild the transport from its own differentiation leaves for the gradient
    # value to carry d Omega/d belief in every regime. The builder closure binds phi/state/rope and
    # receives (mu_query, sigma_query,
    # mu_key, sigma_key) from the oracle -- the key slots are the oracle's detached keys under
    # filtering (query-side coordinate ascent) and the shared live leaves under smoothing (full
    # gradient). Pre-building here as well would only double the most expensive build in the
    # codebase.
    transport_registration = get_transport_registration(transport_mode)
    consumes_omega = _belief_update_consumes_omega(e_step_update, relational_operators)
    if not consumes_omega:
        # Predictive-content replaces mu and keeps sigma and never reads omega; the phi
        # sub-step and the post-phi chart check build their own transports. Skip this dead
        # build and leave ``transport_status`` unfilled. On the live projected route the
        # model already builds ``shared_omega`` for the relational frame and reuses it via
        # ``_prebuilt_omega``, so this gate only helps un-hoisted callers.
        omega, omega_builder = None, None
    elif transport_registration.needs_mu or transport_registration.needs_sigma:
        omega = None

        def _omega_builder(mu_q: torch.Tensor, sigma_q: torch.Tensor,
                           mu_k: torch.Tensor, sigma_k: torch.Tensor):
            return build_belief_transport(
                belief.phi, group, transport_mode=transport_mode,
                mu=mu_q, mu_key=mu_k, transport_state=transport_state,
                # d Omega/d sigma on every path, not only the live unroll (audit 2026-07-01 C4).
                sigma=sigma_q, sigma_key=sigma_k,
                clamp_monitor=clamp_monitor,
                validity_max_norm=transport_chart_max_norm,
                exactness_out=transport_status,
                rope=rope, rope_on_cov=rope_on_cov,
                rope_on_value=rope_on_value, rope_insertion=rope_insertion,
            )
        omega_builder = _omega_builder
    else:
        # Use the hoisted transport when the caller pre-built it (flat path, e_phi_lr==0 so
        # belief.phi is iteration-invariant). Fall back to building here for direct calls or
        # when phi is live (e_phi_lr > 0 means the previous iteration updated phi).
        if _prebuilt_omega is not None:
            omega = _prebuilt_omega
        else:
            omega = build_belief_transport(
                belief.phi, group, transport_mode=transport_mode,
                right_phi=belief.right_phi,
                mu=belief.mu, transport_state=transport_state,
                clamp_monitor=clamp_monitor,

                exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                transport_mean_per_head=transport_mean_per_head,
                congruence_cond_escalation=congruence_cond_escalation,
                compact_phi_block_transport=compact_phi_block_transport,
                validity_max_norm=transport_chart_max_norm,
                exactness_out=transport_status,
                rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value, rope_insertion=rope_insertion,
            )
        omega_builder = None
    # Runtime truncation warning (audit 2026-06-09 P8): under the 'unroll' estimator with a LIVE
    # belief, a non-kernel route is served by the autograd oracle, which returns a DETACHED
    # tangent unless oracle_unroll_grad=True -- silently severing the unrolled-through-inference
    # signal to the prior tables. The config-time warning covers only the learnable-parameter
    # cases; this fires when the truncation actually happens (default warnings filter: once).
    # ``e_step_update != 'mm_exact'`` (audit 2026-08-07): under mm_exact the belief update NEVER
    # calls belief_gradients/the oracle at all (mm_exact_update's own graph stays live under
    # 'unroll' regardless of oracle_unroll_grad -- see its docstring), so this warning's "routes to
    # the DETACHED oracle" claim is a false positive there; excluded rather than inherited from the
    # GRADIENT-route predicate below.
    if (e_step_gradient == "unroll" and not oracle_unroll_grad and belief.mu.requires_grad
            and e_step_update != "mm_exact"
            and not uses_kernel_route(
                renyi_order=renyi_order, gradient_mode=gradient_mode, family=family,
                divergence_family=divergence_family,
                include_attention_entropy=include_attention_entropy,
                transport_mode=transport_mode,
                decoupled_value_gauge=(rope is not None and not rope_on_value),
                route="gradient")):
        import warnings
        warnings.warn(
            "e_step_gradient='unroll' is being served by the autograd ORACLE with "
            "oracle_unroll_grad=False: the belief gradient is a detached tangent, so the "
            "unrolled-through-inference signal to the prior tables is truncated. Set "
            "oracle_unroll_grad=True for the differentiable oracle (or use the kernel route).",
            UserWarning,
            stacklevel=2,
        )
    # Per-iteration count of float64 source-factor recoveries in the relational scorer, reduced
    # on the device (no host sync); published by the relational diagnostics (audit 2026-09-23 low).
    recovery_before = (source_recovery_counter(belief.mu.device)
                       if relational_capture is not None and relational_operators is not None
                       else None)
    if e_step_update == "mm_exact":
        # Closed-form MM/coordinate-exact update: precision fusion at frozen beta (kernels.py,
        # ``mm_exact_update``). Derived from the diagonal-KL filtering kernel, so it is
        # kernel-route ONLY; config __post_init__ rejects oracle-routed configs, and direct
        # callers get the same rejection here.
        if not math.isfinite(mm_damping) or not (0.0 < mm_damping <= 1.0):
            raise ValueError(
                f"mm_damping must be finite and in (0, 1], got {mm_damping}"
            )
        if not uses_kernel_route(
                renyi_order=renyi_order, gradient_mode=gradient_mode, family=family,
                divergence_family=divergence_family,
                include_attention_entropy=include_attention_entropy,
                transport_mode=transport_mode,
                decoupled_value_gauge=(rope is not None and not rope_on_value),
                route="mm_exact"):
            raise ValueError(
                "e_step_update='mm_exact' requires a family that registers the mm_exact closed-form "
                "capability (filtering + renyi order 1 + attention entropy, flat transport, coupled "
                "value gauge); this call routes the belief gradient to the autograd oracle (or has no "
                "closed form at all)."
            )
        mu_star, sigma_star = mm_exact_update(
            belief.mu, belief.sigma, mu_p, sigma_p, omega,
            tau=tau, b0=b0, c0=c0, lambda_beta=lambda_beta,
            kl_max=kl_max, eps=eps, value=value,
            lambda_alpha_mode=lambda_alpha_mode, family=family,
            full_cov_kl_precision=full_cov_kl_precision,
            full_cov_kl_backend=full_cov_kl_backend,
            divergence_family=divergence_family,
            need_sigma_update=(not skip_belief_sigma_update),
            reuse_pairwise_kl_stats=reuse_pairwise_kl_stats,
            irrep_dims=group.irrep_dims, log_prior=log_prior,
        )
        # The MM forward blend is shared by all backward estimators. Straight-through
        # attaches the live incoming-state chain after the complete blend and bounds.
        if grad_record is not None:
            # mm has no raw gradient; record the DISPLACEMENT norms ||theta* - theta|| instead
            # (same keys, detached 0-dim tensors; phi overwritten below iff phi steps).
            grad_record["mu"]    = (mu_star - belief.mu).detach().pow(2).sum().sqrt()
            grad_record["sigma"] = (sigma_star - belief.sigma).detach().pow(2).sum().sqrt()
            grad_record["phi"]   = belief.mu.new_zeros(())
        # Damped step in NATURAL coordinates (mirror descent on the Gaussian natural parameters):
        #   Lambda <- (1-eta) Lambda + eta Lambda*,  (Lambda mu) <- (1-eta) Lambda mu + eta Lambda* mu*,
        # eta = mm_damping (eta=1 -> the exact coordinate minimizer in one step).
        eta = mm_damping
        if skip_belief_sigma_update:
            # sigma frozen: the natural-parameter blend is undefined with Lambda held fixed, so
            # damp the mean in MEAN coordinates instead (eta=1 still lands exactly on mu*).
            mu    = (1.0 - eta) * belief.mu + eta * mu_star
            sigma = belief.sigma
        elif belief.sigma.dim() == belief.mu.dim() + 1:
            # Full covariance (audit 2026-08-07): the SAME natural-parameter blend, generalized from
            # the scalar 1/sigma below to the K x K precision matrix; see
            # kernels.mm_damped_precision_blend_full for the derivation and the eta=0/eta=1 endpoint
            # pins. The elementwise [eps, sigma_max] clamp below becomes an eigenvalue projection
            # (the SAME one retract_spd_full applies for its own ceiling).
            mu, sigma = mm_damped_precision_blend_full(
                belief.mu, belief.sigma, mu_star, sigma_star, eta, eps=eps, sigma_max=sigma_max,
            )
        else:
            lam_old = 1.0 / belief.sigma.clamp(min=eps)
            lam_new = (1.0 - eta) * lam_old + eta / sigma_star           # sigma_star >= eps (kernel floor)
            mu      = ((1.0 - eta) * lam_old * belief.mu + eta * mu_star / sigma_star) / lam_new
            sigma   = 1.0 / lam_new
            # the existing SPD bounds, applied AFTER the blend (mirrors the retraction's clamps)
            sigma   = sigma.clamp(min=eps) if sigma_max is None else sigma.clamp(min=eps, max=sigma_max)
        if e_step_gradient == "straight_through":
            # Parenthesized zero carriers preserve the computed forward bits.
            mu = mu.detach() + (belief.mu - belief.mu.detach())
            sigma = sigma.detach() + (belief.sigma - belief.sigma.detach())
    elif (relational_operators is not None
          and relational_operators.readout_mode == "predictive_content"):
        if relational_frame is None:
            raise ValueError("predictive content requires the realized query/key frames")
        # 'moment_matched' replaces both moments by those of the predictive kernel;
        # 'passthrough' replaces the mean and keeps the incoming covariance.
        moment_matched = getattr(relational_operators, "predictive_covariance",
                                 "passthrough") == "moment_matched"
        readout = relational_predictive_readout(
            belief.mu, belief.sigma, relational_frame, relational_operators,
            tau=tau, kl_max=kl_max, log_prior=log_prior,
            value_frame=relational_value_frame, capture=relational_capture,
            **({"return_covariance": True} if moment_matched else {}))
        mu, sigma = readout if moment_matched else (readout, belief.sigma)
        if grad_record is not None:
            # No comparison-belief gradient is taken in the predictive branch.
            for role in ("mu", "sigma", "phi"):
                grad_record[role] = None
    else:
        if relational_operators is not None:
            if relational_frame is None:
                raise ValueError("relational filtering requires the realized forward frame")
            grad_mu, grad_sigma = compute_relational_filtering_gradients(
                belief.mu, belief.sigma, mu_p, sigma_p, omega, relational_operators,
                relational_frame,
                family=family, value_frame=relational_value_frame,
                full_cov_kl_precision=full_cov_kl_precision,
                tau=tau, lambda_beta=lambda_beta, kl_max=kl_max, eps=eps,
                b0=b0, c0=c0, value=value, lambda_alpha_mode=lambda_alpha_mode,
                log_prior=log_prior, create_graph=(oracle_unroll_grad and torch.is_grad_enabled()),
                capture=relational_capture,
            )
        else:
            grad_mu, grad_sigma = belief_gradients(
                belief.mu, belief.sigma, mu_p, sigma_p, omega,
                tau=tau, renyi_order=renyi_order, value=value, b0=b0, c0=c0, lambda_beta=lambda_beta,
                kl_max=kl_max, eps=eps,
                include_attention_entropy=include_attention_entropy, gradient_mode=gradient_mode,
                family=family, full_cov_kl_precision=full_cov_kl_precision,
                full_cov_kl_backend=full_cov_kl_backend,
                divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode,
                transport_mode=transport_mode, omega_builder=omega_builder,
                irrep_dims=group.irrep_dims, log_prior=log_prior,
                # skip_belief_sigma_update: the kernel skips the (B,N,N,K) sigma pair contraction
                # entirely and returns grad_sigma=None (the oracle fallback still computes it; the
                # sigma update below is skipped either way).
                need_sigma_grad=(not skip_belief_sigma_update),
                compile_pair_kernel=compile_pair_kernel,
                reuse_pairwise_kl_stats=reuse_pairwise_kl_stats,
                # Opt-in unrolled-oracle: make the autograd oracle (non-kernel families) return a
                # differentiable belief gradient so the through-inference signal reaches the prior, matching
                # the hand kernel. Gated on the explicit oracle_unroll_grad toggle (default OFF preserves the
                # detached oracle); only meaningful under the 'unroll' estimator (straight_through detaches
                # downstream, detach runs under no_grad).
                create_graph=(oracle_unroll_grad and e_step_gradient == "unroll"),
            )
        if grad_record is not None:
            # Diagnostic: the RAW belief-gradient L2 norms ||grad_mu||, ||grad_sigma|| (pre-natural-grade,
            # the E-step analogue of the M-step's raw per-group grad_norm). Detached 0-dim tensors (no host
            # sync here, no graph retained); phi defaults to 0 and is overwritten below iff phi steps.
            grad_record["mu"]    = grad_mu.detach().pow(2).sum().sqrt()
            grad_record["sigma"] = (grad_sigma.detach().pow(2).sum().sqrt()
                                    if grad_sigma is not None else grad_mu.new_zeros(()))
            grad_record["phi"]   = grad_mu.new_zeros(())
        # Fisher preconditioner is FAMILY-KEYED (add-by-registering): each BeliefParams owns its Fisher
        # metric, so a non-Gaussian family (e.g. laplace_diagonal, I_mu=I_b=1/b^2) is descended in its
        # own geometry instead of the hardcoded Gaussian Fisher. The Gaussian families delegate to the
        # pinned geometry kernel (byte-identical); 'family' is the same key passed to belief_gradients.
        # When both sigma is frozen and the mean uses the raw Euclidean arm, no natural-gradient
        # component is consumed. Bypass the family preconditioner entirely instead of computing and
        # discarding its sigma arm. The natural-mean route still delegates to the family metric.
        if grad_sigma is None and e_step_mu_precond == "raw":
            nat_mu, nat_sigma = grad_mu, None
        else:
            nat_mu, nat_sigma = get_family(family)(belief.mu, belief.sigma).natural_gradient(
                grad_mu, grad_sigma, eps=eps)

        # STRAIGHT-THROUGH (manuscript Algorithm 1, GL(K)_attention.tex:2050): detach the per-iteration
        # tangent so only the ADDITIVE chain stays live -- the belief is rebuilt as mu_prev + delta and
        # retract(sigma_prev, delta) below, giving d belief_next/d belief_prev = I WITHOUT the
        # second-order d delta/d belief_prev term the unrolled path keeps. The tangent is detached AFTER
        # natural_gradient (which reintroduces a live belief.sigma dependence), not at grad_mu/grad_sigma,
        # so no second-order term leaks through the Fisher metric. 'unroll' (and the no_grad-wrapped
        # 'detach') leave nat_mu/nat_sigma untouched, so those lines are byte-identical to before and the
        # forward VALUE is unchanged (detach never alters a number). This mirrors the phi step, which is
        # already straight-through (fresh detached leaf, create_graph=False).
        if e_step_gradient == "straight_through":
            nat_mu = nat_mu.detach()
            if nat_sigma is not None:
                nat_sigma = nat_sigma.detach()

        # B3/EXP-14 mean-arm ablation: descend the Fisher natural gradient nat_mu (= Sigma*grad_mu for a
        # diagonal Gaussian) by default, or the raw Euclidean grad_mu under e_step_mu_precond='raw'. The
        # SPD sigma retraction below is unchanged either way, isolating the MEAN preconditioner; grad_mu is
        # detached under straight_through to match nat_mu's per-iteration tangent severance.
        if e_step_mu_precond == "raw":
            mu_grad = grad_mu.detach() if e_step_gradient == "straight_through" else grad_mu
        else:
            mu_grad = nat_mu
        delta_mu = e_q_mu_lr * mu_grad
        # E-step MEAN trust region (default OFF). When e_mu_q_trust is set, bound the
        # per-iteration mean step in sigma-whitened units before the additive update; None reproduces the
        # bare mu = belief.mu - e_q_mu_lr*nat_mu bit-for-bit. is_diagonal mirrors the SPD-retraction rank
        # rule below (full cov iff sigma.dim() == mu.dim() + 1).
        if e_mu_q_trust is not None:
            # m11: under straight_through the whitening sigma must be detached, else where the box/ball
            # clamp binds the output +-trust*sqrt(sigma) leaks a live d/d(belief.sigma) gradient into mu
            # -- the per-iteration tangent the ST estimator severs everywhere else. Values unchanged.
            trust_sigma = belief.sigma.detach() if e_step_gradient == "straight_through" else belief.sigma
            delta_mu = apply_mu_trust_region(
                delta_mu, trust_sigma, trust=e_mu_q_trust, mode=mu_trust_mode,
                is_diagonal=(belief.sigma.dim() == belief.mu.dim()), eps=eps,
                family=family,
            )
        mu = belief.mu - delta_mu
        if skip_belief_sigma_update or nat_sigma is None:
            # skip_belief_sigma_update: sigma passes through UNCHANGED (no retraction, no clamp).
            # nat_sigma is None whenever the gradient route produced no sigma arm -- relational
            # filtering always does (its return type is Tuple[Tensor, None]), and belief_gradients
            # does when need_sigma_grad=False. Retracting with a None step is a TypeError and there
            # is no step to take, so freeze either way.
            sigma = belief.sigma
        else:
            # The affine SPD retraction owns the diagonal-vs-full rank decision internally
            # (full covariance iff sigma.dim() == mu.dim() + 1).
            sigma = retract_spd_affine(
                belief.sigma, -e_q_sigma_lr * nat_sigma, belief.mu.dim(),
                trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
            )
    if recovery_before is not None:
        relational_capture["relational_covariance_repair_count"] = (
            source_recovery_counter(belief.mu.device) - recovery_before)

    if relational_capture is not None and "conditional_context" in relational_capture:
        with torch.no_grad():
            relational_capture["relational_conditional_cost_after"] = qkv_conditional_cost(
                mu, relational_capture["conditional_context"]).detach()
            relational_capture["relational_mean_displacement_norm"] = (mu - belief.mu).norm().detach()

    phi = belief.phi
    if e_phi_lr > 0.0:
        # Sequential (Gauss-Seidel) coordinate descent, by design (audit 2026-06-10 F15): the phi
        # substep is evaluated at the UPDATED (mu, sigma) with the pre-update phi -- each substep
        # descends F at the current point of the sweep. This mixed state is the standard
        # coordinate-descent sweep, not a frozen joint state.
        # The phi gradient fundamentally requires autograd (autograd.grad on a
        # fresh requires_grad leaf), so it must run under an enable_grad island even when
        # the caller wraps the stack in no_grad (the detached fixed-point regime).
        # create_graph defaults to False, so grad_phi is detached from the outer graph and
        # acts as a constant tangent there; on the default unrolled path enable_grad is a
        # no-op and the phi-graph connection still flows belief.phi -> retract_phi -> omega.
        with torch.enable_grad():
            phi_g = belief.phi.detach().clone().requires_grad_(True)
            L = phi_alignment_loss(
                mu, sigma, phi_g, group, tau=tau, renyi_order=renyi_order, kl_max=kl_max, eps=eps,
                mass_phi=mass_phi, lambda_beta=lambda_beta,
                family=family, divergence_family=divergence_family,
                full_cov_kl_precision=full_cov_kl_precision,
                full_cov_kl_backend=full_cov_kl_backend,
                include_attention_entropy=include_attention_entropy, log_prior=log_prior,
                transport_mode=transport_mode,
                # gauge-RoPE: the phi step must descend the SAME rotated belief-coupling block as the
                # mu/sigma step, else under pos_rotation='rope' + e_phi_lr>0 phi optimizes a different
                # free energy than mu/sigma (audit 2026-06-17 round 2 id15). None/off -> byte-identical.
                rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value, rope_insertion=rope_insertion,
                right_phi=belief.right_phi,
                # INTENTIONAL asymmetry (audit 2026-06-09 D3): registry-owned transport state is
                # detached here, so it trains only through the mu/sigma belief path, never through
                # the phi-step autograd island. Removing the detach would leak second-order phi-step
                # terms into transport-state gradients.
                transport_state={key: value.detach() for key, value in transport_state.items()},
                clamp_monitor=clamp_monitor,
                exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                transport_mean_per_head=transport_mean_per_head,
                congruence_cond_escalation=congruence_cond_escalation,
                compact_phi_block_transport=compact_phi_block_transport,
                transport_chart_max_norm=transport_chart_max_norm,
                transport_status=transport_status,
                # Filtering is query-side coordinate ascent for phi too (audit 2026-09-23 #6):
                # detach the sender frames so phi_j receives no rows i > j. Smoothing keeps the
                # full envelope gradient through both endpoints.
                detach_key_frames=(gradient_mode == "filtering"),
            )
            grad_phi = torch.autograd.grad(L, phi_g)[0]
        if grad_record is not None:                      # RAW phi-gradient norm (pre-precondition)
            grad_record["phi"] = grad_phi.detach().pow(2).sum().sqrt()
        grad_phi = precondition_phi_gradient(
            grad_phi, belief.phi, group.generators, mode=phi_precond_mode, irrep_dims=group.irrep_dims,
        )
        if phi_retract_mode == "bch" and phi_precond_mode in ("pullback", "pullback_per_block"):
            grad_phi = _bch_chart_direction(
                grad_phi, belief.phi, group.generators,
                irrep_dims=(group.irrep_dims if phi_precond_mode == "pullback_per_block" else None),
            )
        phi = retract_phi(
            belief.phi,
            -grad_phi,
            group,
            step_size=e_phi_lr,
            mode=phi_retract_mode,
            compact_blocks=compact_phi_block_transport,
            residual_max=bch_residual_max,
        )
        if transport_chart_max_norm is not None:
            # The phi objective validates the pre-step frame, but a large retraction can leave the
            # opt-in validity chart on the final iteration. Validate the actual returned transport
            # immediately; there may be no later iteration or diagnostic to catch it.
            with torch.no_grad():
                build_belief_transport(
                    phi,
                    group,
                    transport_mode=transport_mode,
                    mu=mu,
                    sigma=sigma,
                    transport_state=transport_state,


                    clamp_monitor=clamp_monitor,
                    exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                    transport_mean_per_head=transport_mean_per_head,
                    congruence_cond_escalation=congruence_cond_escalation,
                    compact_phi_block_transport=compact_phi_block_transport,
                    validity_max_norm=transport_chart_max_norm,
                    right_phi=belief.right_phi,
                    exactness_out=transport_status,
                    rope=rope,
                    rope_on_cov=rope_on_cov,
                    rope_on_value=rope_on_value, rope_insertion=rope_insertion,
                )

    # _replace (not a fresh BeliefState): carry the untouched channels (right_phi) through the
    # rebuild, so the next iteration's transport build still sees them.
    return belief._replace(mu=mu, sigma=sigma, phi=phi)


def e_step(
    belief:            BeliefState,
    mu_p:              torch.Tensor,        # (N, K)
    sigma_p:           torch.Tensor,        # (N, K)
    group:             GaugeGroup,

    *,
    tau:                       'float | torch.Tensor' = 1.0,

    e_q_mu_lr:                 float = 0.1,
    e_q_sigma_lr:              float = 0.1,
    e_phi_lr:                  float = 0.1,
    exp_fp64_norm_threshold:   float = 5.0,       # 'norm': max clamped block ||M||_F upcast threshold

    n_iter:                    int = 1,

    e_step_gradient:           str = "unroll",
    rope_insertion:              str = "right",   # "right" (pure, folded frame) | "left" (legacy)
    # Tier-1 transport perf toggles (2026-07-05; explicit, off the F_diag bag -- that diagnostic
    # deliberately keeps the default transport numerics). All default OFF/byte-identical.

    return_trajectory:           bool = False,
    oracle_unroll_grad:          bool = False,    # explicit (not in kwargs): keep it off the F_diag bag
    transport_mean_per_head:     bool = False,    # factored transport_mean contracts per gauge block
    congruence_cond_escalation:    bool = False,   # opt-in conditioning trigger for the fp64 congruence escalation
    compact_phi_block_transport: bool = False,    # packed canonical flat block_glk phi factors
    rope_on_cov:                 bool = False,
    rope_on_value:               bool = True,

    grad_record:                 Optional[EStepGradientRecord] = None,   # diag out-param: LAST iteration's belief-grad norms
    state_record:                Optional[dict]                = None,   # diag out-param: every belief iterate + matching F tensor
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
) -> 'BeliefState | Tuple[BeliefState, List[float]]':
    r"""Iterate ``e_step_iteration`` ``n_iter`` times (parallel mean-field). Optionally
    returns the global-F trajectory (a DIAGNOSTIC; parallel updates are not guaranteed
    monotone per iteration). ``state_record`` captures the same initial/updated beliefs and
    free-energy tensors without changing the return type; it is used only by the one-forward
    diagnostic snapshot path.

    ``e_step_gradient`` is the backward estimator forwarded to each iteration: 'unroll' (default,
    full second-order trajectory gradient) or 'straight_through' (per-iteration tangent detached,
    additive chain live). It is an EXPLICIT keyword (not in ``**kwargs``) so it binds here and does
    NOT ride the forwarded knob bag into the diagnostic ``free_energy_value`` (which rejects unknown
    kwargs). 'detach' is realized by the caller wrapping this in no_grad, so it is treated as
    'unroll' here (no_grad already severs every gradient).

    The loop always executes ``n_iter`` iterations, with backward behavior selected
    by ``e_step_gradient`` and the caller's autograd context.
    """
    traj: List[float] = []

    # Hoist the flat transport when phi is frozen across iterations (e_phi_lr==0).  On the flat
    # path the transport depends only on belief.phi, which the phi sub-step never updates when
    # e_phi_lr==0, so it is iteration-invariant.  Build ONCE here and pass it into every
    # e_step_iteration call via _prebuilt_omega, eliminating the redundant per-iteration rebuild.
    # When e_phi_lr > 0 phi changes each iteration so we leave _prebuilt_omega=None and let
    # e_step_iteration rebuild as before.
    transport_mode_kw: str = kwargs.get("transport_mode", "flat")
    compact_phi_block_transport = (
        compact_phi_block_transport
        and transport_mode_kw == "flat"
        and group.phi_coordinate_layout == "block_head_row_major"
    )
    _hoisted_omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport | None' = None
    e_step_update_kw = canonical_e_step_update(kwargs.get("e_step_update", "gradient"))
    consumes_omega = _belief_update_consumes_omega(e_step_update_kw, relational_operators)
    if consumes_omega and e_phi_lr == 0.0 and transport_mode_kw == "flat":
        if prebuilt_transport is not None:
            # The caller built the eligible flat transport ONCE from the SAME phi (both channels
            # hold it fixed at e_phi_lr=0) and shared it across E-step calls, skipping this redundant
            # per-call matrix-exp build. Consumed ONLY inside the
            # hoist guard above (e_phi_lr==0 + flat): on any other route the kwarg is ignored and
            # the per-iteration rebuild stays authoritative.
            _hoisted_omega = prebuilt_transport
        else:
            _hoisted_omega = build_belief_transport(
                belief.phi, group,
                transport_mode="flat",
                right_phi=belief.right_phi,
                gauge_mode="learned",   # not an e_step kwarg (no **kwargs sink downstream); literal, not a dead kwargs read (m14)
                clamp_monitor=kwargs.get("clamp_monitor", False),
                exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                transport_mean_per_head=transport_mean_per_head,
                congruence_cond_escalation=congruence_cond_escalation,
                compact_phi_block_transport=compact_phi_block_transport,
                validity_max_norm=transport_chart_max_norm,
                exactness_out=transport_status,
                rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value, rope_insertion=rope_insertion,
            )

    record_sequence_index = (
        state_record.get("sequence_index") if state_record is not None else None)

    def _f_diag(
        b: BeliefState,
        sequence_index: Optional[int] = None,
    ) -> torch.Tensor:
        # Diagnostic scalar: under no_grad so the logged trajectory never enters the
        # training graph. The legacy float trajectory performs its explicit .item() below;
        # diagnostic snapshots retain the scalar tensor and avoid that host transfer.
        # rope/rope_on_cov are forwarded explicitly (audit PP6): the logged F carries the same
        # RoPE-wrapped transport the iterations descend.
        # NOTE (audit 2026-06-13): under a non-'constant' lambda_alpha_mode the forwarded kwargs
        # carry the regularizer into free_energy_value (alpha_reg=R), so the logged F is the AUGMENTED
        # objective F+R the iterations actually descend (the envelope-correct value), not the bare
        # divergence-weighted F. At the constant default alpha_reg=None, so the trajectory is the pure F.
        if relational_operators is not None:
            raise ValueError("Global-F trajectories are unavailable for relational conditional filtering")
        b_eval, mu_eval, sigma_eval, prior_eval, tau_eval = b, mu_p, sigma_p, log_prior, tau
        if sequence_index is not None and b.mu.dim() >= 3:
            b_eval = b._replace(
                mu=b.mu[sequence_index], sigma=b.sigma[sequence_index], phi=b.phi[sequence_index],
                right_phi=(b.right_phi[sequence_index]
                           if b.right_phi is not None and b.right_phi.dim() == b.phi.dim()
                           else b.right_phi))
            mu_eval = mu_p[sequence_index]
            sigma_eval = sigma_p[sequence_index]
            if isinstance(tau, torch.Tensor) and tau.dim() >= 2 and tau.shape[0] == b.mu.shape[0]:
                tau_eval = tau[sequence_index]
            if log_prior is not None:
                unbatched_rank = 2 if len(group.irrep_dims) == 1 else 3
                if log_prior.dim() == unbatched_rank + 1 and log_prior.shape[0] == b.mu.shape[0]:
                    prior_eval = log_prior[sequence_index]
        with torch.no_grad():
            return free_energy_value(b_eval, mu_eval, sigma_eval, group, tau=tau_eval, log_prior=prior_eval,
                                     rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value, rope_insertion=rope_insertion,
                                     compact_phi_block_transport=compact_phi_block_transport,
                                     transport_mean_per_head=transport_mean_per_head,
                                     congruence_cond_escalation=congruence_cond_escalation,
                                     transport_chart_max_norm=transport_chart_max_norm,
                                     transport_status=transport_status,
                                     **kwargs).detach()

    if state_record is not None:
        state_record.clear()
        state_record["sequence_index"] = record_sequence_index
        state_record["beliefs"] = []
        state_record["free_energy"] = []

    def _record_diagnostic_state(b: BeliefState) -> None:
        if return_trajectory:
            traj.append(_f_diag(b).item())
        if state_record is not None:
            state_record["beliefs"].append(b)
            if relational_operators is None:
                state_record["free_energy"].append(_f_diag(b, record_sequence_index))
            else:
                state_record["semantics"] = relational_operators.inference_semantics

    if return_trajectory or state_record is not None:
        _record_diagnostic_state(belief)
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
            grad_record=grad_record,                       # last iteration overwrites -> converged-ish grad
            log_prior=log_prior, rope=rope, rope_on_cov=rope_on_cov, rope_on_value=rope_on_value, rope_insertion=rope_insertion,
            relational_operators=relational_operators,
            relational_frame=relational_frame,
            relational_value_frame=relational_value_frame,
            relational_capture=relational_capture,
            _prebuilt_omega=_hoisted_omega,
            **kwargs,
        )
        if return_trajectory or state_record is not None:
            _record_diagnostic_state(belief)
    return (belief, traj) if return_trajectory else belief
