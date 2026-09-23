r"""Autograd belief-gradient oracle for VFE_3.0 (the correctness source of truth).

The reduced free energy F_red is differentiated w.r.t. the Gaussian belief
(mu, sigma) by torch.autograd. Two modes for the belief-coupling term, in which a
token appears both as the query (first KL argument) and the transported key
(second argument):
  filtering  query-side only: keys are a DETACHED copy of the belief, so only the
             first-argument (row) gradient flows -- the mean-field coordinate-ascent
             default (holding other beliefs fixed).
  smoothing  full gradient: keys share the belief leaf, so the second-argument
             (column) gradient flows back through the transport (Omega^T pullback)
             -- the theoretically pure d F_red.
Reference for every family / divergence / mode; the hand kernels are pinned to the
FILTERING oracle.
"""

from typing import Callable, List, Optional, Tuple

import torch

from vfe3.alpha_i import self_coupling_alpha
from vfe3.families.base import get_family
from vfe3.families.policy import coupling_energy as _coupling_energy, family_instance as _family_instance
from vfe3.families.gaussian import FullGaussian
from vfe3.free_energy import free_energy, self_divergence_for_alpha
# Forward transport containers are named in the `omega` forward-ref annotation below; import them
# at runtime (alongside the transport helpers this module already imports) so
# typing.get_type_hints resolves the annotation. transport.py does not import this module, so there
# is no import cycle.
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
)


def _transport_to_float(
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',
) -> 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport':
    r"""Cast only an oracle island's transport tensors to fp32, preserving its representation."""
    if isinstance(omega, torch.Tensor):
        return omega.float()
    # cond_escalation MUST be carried on every rebuild (audit 2026-08-05): it defaults to False, so
    # dropping it here silently disabled the opt-in conditioning guard INSIDE the oracle island while
    # the rest of the forward still had it on -- the gradient and the reported objective would then
    # be computed under different congruence policies. Every other reconstruction site in
    # transport.py propagates it; these three did not.
    if isinstance(omega, CompactFactoredTransport):
        return CompactFactoredTransport(
            omega.exp_blocks.float(), omega.inv_blocks.float(), omega.K,
            mean_per_head=omega.mean_per_head,
            same_frame_flat_cocycle=omega.same_frame_flat_cocycle,
            cond_escalation=omega.cond_escalation,
        )
    if isinstance(omega, FactoredTransport):
        return FactoredTransport(
            exp_phi=omega.exp_phi.float(),
            exp_neg_phi=omega.exp_neg_phi.float(),
            irrep_dims=omega.irrep_dims,
            mean_per_head=omega.mean_per_head,
            same_frame_flat_cocycle=omega.same_frame_flat_cocycle,
            cond_escalation=omega.cond_escalation,
        )
    if isinstance(omega, RopeTransport):
        return RopeTransport(
            base=_transport_to_float(omega.base),
            rope=omega.rope.float(),
            on_cov=omega.on_cov,
            on_value=omega.on_value,
            insertion=omega.insertion,
            same_frame_flat_cocycle=omega.same_frame_flat_cocycle,
        )
    raise TypeError(f"unsupported transport type {type(omega).__name__!r}")


# The belief update is part of the model FORWARD (iterative belief minimization), so this
# oracle must produce a gradient even when the caller runs the forward under no_grad -- the
# eval() / diagnostics() / generate() regime (evaluate is @torch.no_grad) and the detached
# E-step. autograd.grad needs grad enabled, so the oracle carries its own enable_grad island,
# exactly as the phi step does (e_step.py).
#
# ``create_graph`` mirrors the hand kernel's behavior on the UNROLLED E-step (e_step_gradient=
# 'unroll'): with create_graph=True the query leaf is the LIVE belief (not a detached clone) and the
# returned grads keep their grad_fn, so the unrolled-through-inference signal reaches the prior
# tables for non-kernel families (smoothing / gaussian_full / renyi_order!=1) -- exactly as the closed-
# form kernel already does. With create_graph=False (the default, used for detach / straight-through /
# diagnostics / any no_grad caller) the oracle clones a detached leaf and .detach()-es the result, a
# constant tangent that leaks no graph -- byte-identical to the previous behavior. Either way the
# returned VALUES are the same (autograd.grad gives the same numbers); only connectivity differs.
@torch.enable_grad()
def belief_gradients_autograd(
    mu:           torch.Tensor,           # (N, K) belief means (the variable)
    sigma:        torch.Tensor,           # (N, K) belief variances
    mu_p:         torch.Tensor,           # (N, K) prior means
    sigma_p:      torch.Tensor,           # (N, K) prior variances
    omega:        'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',

    *,
    tau:           'float | torch.Tensor' = 1.0,
    lambda_beta:   'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure F)

    renyi_order:   float = 1.0,
    kl_max:        float = 100.0,
    eps:           float = 1e-6,
    b0:            float = 1.0,
    c0:            float = 1.0,
    value:         float = 1.0,

    include_attention_entropy: bool = True,
    create_graph:              bool = False,   # True (unroll): live leaf + differentiable grad to prior
    need_sigma_grad:           bool = True,    # False -> return (grad_mu, None)
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    full_cov_kl_precision:     Optional[str] = None,
    full_cov_kl_backend:       str  = "pairwise",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",

    irrep_dims:                Optional[List[int]]    = None,
    log_prior:                 Optional[torch.Tensor] = None,
    omega_builder:             Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                                                 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport']] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:   # (grad_mu, grad_sigma), each (N, K)
    r"""Autograd of canonical F_red w.r.t. (mu, sigma). See module docstring for modes.

    ``irrep_dims`` (when multi-block) routes the per-head energy through the family coupling seam;
    autograd then yields the correct per-head belief gradient with no special-casing here.

    ``omega_builder`` (audit 2026-06-10 F1/F2; None = unchanged pre-built ``omega``): a callable
    ``(mu_query, sigma_query, mu_key, sigma_key) -> transport`` for a belief-DEPENDENT connection.
    The transport is rebuilt here from the DIFFERENTIATION
    leaves, so the returned gradient VALUE carries the d Omega/d mu (and, for the covariant
    builder, d Omega/d sigma -- audit 2026-07-01 C4) term in every regime (train, eval, detached
    E-step) -- a pre-built omega is constant w.r.t. the local leaves and silently drops it. The
    key-role semantics carry over exactly: under filtering the builder receives detached key
    slots (query-side d delta/d mu_i only -- mean-field coordinate ascent); under smoothing both
    slots share the live leaves (full gradient, the stationary point of the global F).

    Under an outer AMP context, only this inner objective/derivative construction is re-entered with
    autocast disabled and fp32 belief/prior inputs. The surrounding E-step remains under its caller's
    autocast context, preserving mixed-precision transport, iteration, and state-update operations.
    """
    if torch.is_autocast_enabled(mu.device.type):
        # The outer training autocast scales only the final loss. Inner derivative construction has
        # already happened by then, so a bf16/fp16 transport or objective here cannot be recovered by
        # GradScaler. Re-enter only this oracle under an explicit fp32 island; recursion terminates
        # immediately because autocast is disabled in the nested call. Tensor casts retain the graph
        # for create_graph=True, while the caller resumes its outer autocast context after return.
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return belief_gradients_autograd(
                mu.float(), sigma.float(), mu_p.float(), sigma_p.float(),
                _transport_to_float(omega),
                tau=(tau.float() if isinstance(tau, torch.Tensor) else tau),
                lambda_beta=(lambda_beta.float()
                             if isinstance(lambda_beta, torch.Tensor) else lambda_beta),
                renyi_order=renyi_order,
                kl_max=kl_max,
                eps=eps,
                b0=b0,
                c0=c0,
                value=value,
                include_attention_entropy=include_attention_entropy,
                create_graph=create_graph,
                need_sigma_grad=need_sigma_grad,
                gradient_mode=gradient_mode,
                family=family,
                full_cov_kl_precision=full_cov_kl_precision,
                full_cov_kl_backend=full_cov_kl_backend,
                divergence_family=divergence_family,
                lambda_alpha_mode=lambda_alpha_mode,
                irrep_dims=irrep_dims,
                log_prior=(log_prior.float() if log_prior is not None else None),
                omega_builder=omega_builder,
            )
    # Live leaves (keep the unrolled chain) only when create_graph is requested AND the belief
    # carries grad upstream. Otherwise use a detached clone (autograd.grad
    # needs a grad leaf, and there is no unrolled signal to preserve for a no_grad caller).
    use_live = create_graph and mu.requires_grad and sigma.requires_grad
    if use_live:
        mu_q, sigma_q = mu, sigma
    else:
        mu_q = mu.detach().clone().requires_grad_(True)
        sigma_q = sigma.detach().clone().requires_grad_(True)

    if gradient_mode == "filtering":
        mu_k, sigma_k = mu_q.detach(), sigma_q.detach()       # key role frozen
    elif gradient_mode == "smoothing":
        mu_k, sigma_k = mu_q, sigma_q                          # shared leaf -> full grad
    else:
        raise ValueError(f"gradient_mode must be 'filtering' or 'smoothing', got {gradient_mode!r}")

    if omega_builder is not None:
        # belief-dependent connection: rebuild the transport
        # from the differentiation leaves so autograd sees d Omega/d mu AND d Omega/d sigma
        # (query slots live; key slots follow the filtering/smoothing key-role split above).
        omega = omega_builder(mu_q, sigma_q, mu_k, sigma_k)
    fam = get_family(family)
    # The whole transported pair grid dispatches through the FAMILY's coupling seam, so a family that
    # evaluates the transported divergence differently (the frame-intrinsic identity transport, the
    # exact-congruence backward pull) is honored here by its registration alone. The default seam
    # composes transport_location + transport_dispersion + pairwise_energy exactly as this call site
    # used to inline them, so every family that does not override it is byte-identical.
    #
    # diagonal_out from the BELIEF shape (diagonal iff sigma has the same rank as mu) -- family-agnostic
    # (covers laplace_diagonal etc., not just gaussian_diagonal), so a batch-collapsed (N,N,K,K)
    # omega routes correctly against a batched diagonal sigma (its rank gap vs the omega
    # would otherwise misinfer the full sandwich); behavior-identical for a batched dense omega.
    sd = self_divergence_for_alpha(
                                   _family_instance(mu_q, sigma_q, fam, full_cov_kl_precision),
                                   _family_instance(mu_p, sigma_p, fam, full_cov_kl_precision), alpha=renyi_order, kl_max=kl_max, eps=eps,
                                   divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    alpha, reg = self_coupling_alpha(sd, mode=lambda_alpha_mode, value=value, b0=b0, c0=c0)
    energy = _coupling_energy(
        fam, mu_q, sigma_q, mu_k, sigma_k, omega,
        log_prior=log_prior,
        precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
        alpha=renyi_order, kl_max=kl_max, eps=eps,
        divergence_family=divergence_family, irrep_dims=irrep_dims,
        diagonal_out=(sigma_k.dim() == mu_k.dim()))
    # Value-gauge decoupling (RopeTransport.on_value=False): beta comes from the rotated SCORE energy
    # above, but the coupling sum the belief descends uses the UN-rotated base transport -- RoPE's
    # position-independent value aggregation (GL(K)_attention.tex:1909). None on the coherent default
    # path (byte-identical). Autograd carries the extra d beta/d mu term the broken envelope leaves.
    coupling_energy = None
    if isinstance(omega, RopeTransport) and not omega.on_value:
        coupling_energy = _coupling_energy(
            fam, mu_q, sigma_q, mu_k, sigma_k, omega.base,
            log_prior=log_prior,
            precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
            alpha=renyi_order, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family, irrep_dims=irrep_dims,
            diagonal_out=(sigma_k.dim() == mu_k.dim()))
    F = free_energy(
        sd, energy, alpha, tau=tau, lambda_beta=lambda_beta,
        include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=(reg if lambda_alpha_mode != "constant" else None),
        coupling_energy=coupling_energy,
    )
    if need_sigma_grad:
        grad_mu, grad_sigma = torch.autograd.grad(F, [mu_q, sigma_q], create_graph=use_live)
    else:
        grad_mu, = torch.autograd.grad(F, [mu_q], create_graph=use_live)
        grad_sigma = None
    if use_live:
        return grad_mu, grad_sigma                         # differentiable -> unrolled signal to prior
    return grad_mu.detach(), None if grad_sigma is None else grad_sigma.detach()
