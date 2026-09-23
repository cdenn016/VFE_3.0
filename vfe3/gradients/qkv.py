"""Relational conditional filtering, with a live outer routing graph."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from vfe3.alpha_i import (
    alpha_constant, alpha_gradient_coefficient, alpha_is_per_coord,
    alpha_state_dependent, alpha_state_dependent_per_coord, get_alpha, self_coupling_alpha,
)
from vfe3.contracts import CanonicalFrameContext
from vfe3.divergence import get_functional, get_functional_per_coord, renyi, renyi_per_coord
from vfe3.families.base import BeliefParams, get_family, safe_kl_clamp
from vfe3.families.gaussian import DiagonalGaussian, diag_kl_unclamped, diag_kl_unclamped_per_coord
from vfe3.free_energy import _broadcast_tau, attention_weights, pairwise_energy, self_divergence_for_alpha
from vfe3.geometry.transport import (
    CompactFactoredTransport, FactoredTransport, RopeTransport, transport_covariance,
    transport_covariance_head_blocks, transport_mean,
)
from vfe3.gradients.oracle import _transport_to_float
from vfe3.gradients.kernels import _pair_contract
from vfe3.gradients.pairwise_stats import _reduce_coordinate_term

if TYPE_CHECKING:
    from vfe3.model.relational_operators import RelationalOperators

Transport = Union[torch.Tensor, CompactFactoredTransport, FactoredTransport, RopeTransport]
Coefficient = Union[float, torch.Tensor]
_DIAGONAL_KL = DiagonalGaussian.renyi_closed_form
_DIAGONAL_KL_PER_COORD = DiagonalGaussian.renyi_per_coord


def _uses_analytic_qkv(lambda_alpha_mode: str) -> bool:
    """Keep replacement divergence/alpha registrations on the probe-derivative path."""
    return (
        get_functional("renyi") is renyi
        and get_functional_per_coord("renyi") is renyi_per_coord
        and DiagonalGaussian.renyi_closed_form is _DIAGONAL_KL
        and DiagonalGaussian.renyi_per_coord is _DIAGONAL_KL_PER_COORD
        and get_alpha(lambda_alpha_mode) in (
            alpha_constant, alpha_state_dependent, alpha_state_dependent_per_coord)
    )


def _shared_covariance_energy(
    delta:      torch.Tensor,               # (..., N, N, K), mapped source minus receiver
    sigma_t:    torch.Tensor,               # (..., N, N, K), already floored
    trace:      torch.Tensor,               # (..., H, N, N), or (..., N, N) for one head
    logdet:     torch.Tensor,
    kl_max:     float,
    irrep_dims: List[int],
) -> torch.Tensor:
    r"""KL = (trace + sum(delta^2 / sigma_t) - d + logdet)/2, in the baseline order."""
    mahal = _reduce_coordinate_term(delta.square() / sigma_t, irrep_dims=irrep_dims)
    return safe_kl_clamp(0.5 * (trace + mahal - irrep_dims[0] + logdet), kl_max=kl_max)


@dataclass(frozen=True)
class QKVConditionalContext:
    """Detached pre-step templates used for both conditional-cost endpoints."""

    sigma:             torch.Tensor
    mu_p:              torch.Tensor
    sigma_p:           torch.Tensor
    beta:              torch.Tensor
    value_templates:   torch.Tensor
    sigma_t:           torch.Tensor
    irrep_dims:        List[int]
    lambda_beta:       Coefficient
    b0:                Coefficient
    c0:                Coefficient
    value:             float
    kl_max:            float
    eps:               float
    lambda_alpha_mode: str
    family:            str = "gaussian_diagonal"
    pair_family:       str = "gaussian_diagonal"
    full_cov_kl_precision: Optional[str] = None


def _strict_energy(
    energy: torch.Tensor,
    raw:    torch.Tensor,
    kl_max: float,
) -> torch.Tensor:
    r"""Keep the clipped value and legacy strict derivative 1_{0 < D < kl_max} dD."""
    mask = ((raw.detach() > 0.0) & (raw.detach() < kl_max)).to(energy.dtype)
    return energy.detach() + mask * (energy - energy.detach())


def _family_params(
    mu:        torch.Tensor,
    sigma:     torch.Tensor,
    family:    str,
    precision: Optional[str],
) -> BeliefParams:
    """Construct the selected law with the caller's full-covariance precision policy."""
    cls = get_family(family)
    return cls(mu, sigma, _precision_policy=precision) if cls.cov_kind == "full" else cls(mu, sigma)


def _self_cost(
    mu:                torch.Tensor,
    sigma:             torch.Tensor,
    mu_p:              torch.Tensor,
    sigma_p:           torch.Tensor,
    b0:                Coefficient,
    c0:                Coefficient,
    value:             float,
    kl_max:            float,
    eps:               float,
    lambda_alpha_mode: str,
    strict:            bool,
    family:            str = "gaussian_diagonal",
    full_cov_kl_precision: Optional[str] = None,
) -> torch.Tensor:
    r"""Sum A(mu) = alpha(D) D + R(alpha(D)) once over tokens and coordinates."""
    sd = self_divergence_for_alpha(
        _family_params(mu, sigma, family, full_cov_kl_precision),
        _family_params(mu_p, sigma_p, family, full_cov_kl_precision),
        kl_max=kl_max, eps=eps, lambda_alpha_mode=lambda_alpha_mode)
    if strict and family == "gaussian_diagonal":
        raw_function = diag_kl_unclamped_per_coord if alpha_is_per_coord(lambda_alpha_mode) else diag_kl_unclamped
        raw = raw_function(mu, sigma, mu_p, sigma_p, eps=eps)
        sd = _strict_energy(sd, raw, kl_max)
    alpha, regularizer = self_coupling_alpha(
        sd, mode=lambda_alpha_mode, value=value, b0=b0, c0=c0)
    terms = alpha * sd
    if lambda_alpha_mode != "constant":
        terms = terms + regularizer
    return terms.sum()


def _pair_energy(
    mu:         torch.Tensor,
    sigma:      torch.Tensor,
    templates:  torch.Tensor,
    sigma_t:    torch.Tensor,
    kl_max:     float,
    eps:        float,
    irrep_dims: List[int],
    family:     str = "gaussian_diagonal",
    full_cov_kl_precision: Optional[str] = None,
) -> torch.Tensor:
    r"""Return per-head KL with mapped means and family-specific covariance."""
    # Construct distributions directly: distinct role means invalidate same-law certificates.
    if family == "gaussian_full" and sigma.ndim == mu.ndim:
        sigma = torch.diag_embed(sigma)
    if family == "gaussian_full" and sigma_t.ndim == templates.ndim + 2:
        # Packed transported head marginals: (..., Nq, Nk, H, d, d).
        # Avoid scattering them into a mostly unread (..., Nq, Nk, K, K) grid.
        H, d = len(irrep_dims), irrep_dims[0]
        query_sigma = torch.stack(
            [sigma[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
        q = _family_params(mu.reshape(*mu.shape[:-1], H, d).unsqueeze(-3),
                           query_sigma.unsqueeze(-4), family, full_cov_kl_precision)
        k = _family_params(templates.reshape(*templates.shape[:-1], H, d), sigma_t,
                           family, full_cov_kl_precision)
        return get_functional("renyi")(q, k, alpha=1.0, kl_max=kl_max, eps=eps).movedim(-1, -3)
    return pairwise_energy(
        _family_params(mu, sigma, family, full_cov_kl_precision),
        _family_params(templates, sigma_t, family, full_cov_kl_precision),
        kl_max=kl_max, eps=eps, irrep_dims=irrep_dims)


def qkv_conditional_cost(
    mu:      torch.Tensor,
    context: QKVConditionalContext,
) -> torch.Tensor:
    r"""Evaluate C(mu; beta_pre, V_pre) using one detached pre-update context."""
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work = mu.to(context.sigma.dtype)
        self_cost = _self_cost(
            mu_work, context.sigma, context.mu_p, context.sigma_p,
            context.b0, context.c0, context.value, context.kl_max, context.eps,
            context.lambda_alpha_mode, False, context.family, context.full_cov_kl_precision)
        energy = _pair_energy(mu_work, context.sigma, context.value_templates,
                              context.sigma_t, context.kl_max, context.eps, context.irrep_dims,
                              context.pair_family, context.full_cov_kl_precision)
        return self_cost + (_broadcast_tau(context.lambda_beta, energy) * context.beta * energy).sum()


def _cast_coefficient(value: Coefficient, dtype: torch.dtype) -> Coefficient:
    return value.to(dtype) if isinstance(value, torch.Tensor) else value


def _detach_coefficient(value: Coefficient) -> Coefficient:
    return value.detach() if isinstance(value, torch.Tensor) else value


def qkv_filtering_gradients(
    mu:                torch.Tensor,             # (..., N, K)
    sigma:             torch.Tensor,             # (..., N, K) diagonal or (..., N, K, K) full covariance
    mu_p:              torch.Tensor,
    sigma_p:           torch.Tensor,
    omega:             Transport,
    frame:             CanonicalFrameContext,
    *,
    tau:               Coefficient = 1.0,
    lambda_beta:       Coefficient = 1.0,
    b0:                Coefficient = 1.0,
    c0:                Coefficient = 1.0,
    kl_max:            float       = 100.0,
    eps:               float       = 1e-6,
    value:             float       = 1.0,
    create_graph:      bool        = False,
    lambda_alpha_mode: str         = "constant",
    family:            str         = "gaussian_diagonal",
    full_cov_kl_precision: Optional[str] = None,
    value_frame:       Optional[CanonicalFrameContext] = None,
    log_prior:         Optional[torch.Tensor]  = None,
    capture:           Optional[Dict[str, Any]] = None,
    relational_operators: Optional["RelationalOperators"] = None,
) -> Tuple[torch.Tensor, None]:
    r"""Compute partial_z C(z; beta(mu), V(stop_gradient(mu))) at z = mu.

    The analytic receiver derivative excludes inner router differentiation, while
    beta and mapped templates retain their outer training graphs. Strict endpoint
    masks reproduce the legacy derivative. Custom registrations retain the probe path.
    """
    if relational_operators is None:
        raise ValueError("conditional filtering requires the relational operator module")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("conditional filtering does not support inference_mode; use torch.no_grad() instead")
    outer_grad_enabled = torch.is_grad_enabled()
    training_graph = bool(create_graph and outer_grad_enabled)
    dtype = torch.float64 if mu.dtype == torch.float64 else torch.float32
    public_dtype = mu.dtype
    analytic = family == "gaussian_diagonal" and _uses_analytic_qkv(lambda_alpha_mode)
    pair_family = ("gaussian_full" if family in (
        "gaussian_full", "gaussian_diagonal_exact", "gaussian_frame_diagonal") else family)
    value_frame = frame if value_frame is None else value_frame
    with torch.set_grad_enabled(training_graph or not analytic), torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work, sigma_work = mu.to(dtype), sigma.to(dtype)
        prior_mu, prior_sigma = mu_p.to(dtype), sigma_p.to(dtype)
        omega_work = omega if dtype == torch.float64 else _transport_to_float(omega)
        # apply_mean casts canonical factors and Delta coherently without detaching.
        source_mu, source_sigma = mu_work.detach(), sigma_work.detach()
        from vfe3.gradients.relational import relational_score
        dims = relational_operators.irrep_dims
        intrinsic_covariance = (family == "gaussian_frame_diagonal"
                                and not (isinstance(omega_work, RopeTransport) and omega_work.on_cov))
        if family == "gaussian_frame_diagonal":
            # Stored locations/covariances are intrinsic; only the relative RoPE
            # frame acts on them. With RoPE off this is exactly the identity.
            omega_work = (CompactFactoredTransport(
                value_frame.forward, value_frame.inverse, sum(dims))
                if value_frame.block_diagonal else FactoredTransport(
                    value_frame.forward, value_frame.inverse, dims))
        # Value gauge: under RoPE with on_value=False the value link is the UN-rotated base
        # transport, values_ij = U_i O T U_j^{-1} mu_j, exactly as the predictive route and the
        # transported oracle's value-gauge coupling energy (gradients/oracle.py) use it; the router
        # score below still reads the rotated frame. Mean and covariance share this value transport.
        value_omega = (omega_work.base
                       if isinstance(omega_work, RopeTransport) and not omega_work.on_value
                       else omega_work)
        values = transport_mean(value_omega, relational_operators.apply_content(source_mu, value_frame))  # (..., N, N, K)
        transport_sigma = (torch.diag_embed(source_sigma)
                           if pair_family == "gaussian_full" and source_sigma.ndim == source_mu.ndim
                           else source_sigma)
        if intrinsic_covariance:
            # No covariance rotation under means-only RoPE; the base frame
            # cancels in intrinsic coordinates. Broadcast a view over receivers.
            H, d = len(dims), dims[0]
            source_blocks = torch.diag_embed(source_sigma.reshape(*source_mu.shape[:-1], H, d))
            sigma_t = source_blocks.unsqueeze(-5).expand(
                *source_mu.shape[:-2], mu.shape[-2], source_mu.shape[-2], H, d, d)
        else:
            sigma_t = (transport_covariance_head_blocks(
                value_omega, transport_sigma, diagonal_out=False, marginal_blocks=dims,
                retain_full_precision=(full_cov_kl_precision == "fp64"))
                if pair_family == "gaussian_full" else None)
        if sigma_t is None:
            sigma_t = transport_covariance(
                value_omega, transport_sigma, diagonal_out=(pair_family != "gaussian_full"),
                marginal_blocks=dims)
        if analytic:
            sigma_t_safe = sigma_t.clamp(min=eps)
            sigma_q_safe = sigma_work.clamp(min=eps).unsqueeze(-2)
            trace = _reduce_coordinate_term(sigma_q_safe / sigma_t_safe, irrep_dims=dims)
            logdet = _reduce_coordinate_term(
                torch.log(sigma_t_safe) - torch.log(sigma_q_safe), irrep_dims=dims)
        score = safe_kl_clamp(relational_score(
            mu_work, sigma_work, source_mu, source_sigma, frame, relational_operators, capture),
            kl_max=kl_max)
        tau_work = _cast_coefficient(tau, dtype)
        beta = attention_weights(score, tau=tau_work,
                                 log_prior=None if log_prior is None else log_prior.to(dtype))
        beta_scale = _cast_coefficient(lambda_beta, dtype)
        b0_work, c0_work = _cast_coefficient(b0, dtype), _cast_coefficient(c0, dtype)
        if analytic:
            delta = values - mu_work.unsqueeze(-2)                  # (..., N, N, K)
            energy = _shared_covariance_energy(delta, sigma_t_safe, trace, logdet, kl_max, dims)
            pair_mask = ((energy > 0.0) & (energy < kl_max)).to(dtype)
            pair_weight = _broadcast_tau(beta_scale, energy) * beta * pair_mask
            grad_pair = _pair_contract(pair_weight, -delta / sigma_t_safe, dims)
            # Use the strict self mask in BOTH the coefficient graph and the force.
            per_coord = alpha_is_per_coord(lambda_alpha_mode)
            raw_fn = diag_kl_unclamped_per_coord if per_coord else diag_kl_unclamped
            raw = raw_fn(mu_work, sigma_work, prior_mu, prior_sigma, eps=eps)
            sd = _strict_energy(safe_kl_clamp(raw, kl_max=kl_max), raw, kl_max)
            coefficient = alpha_gradient_coefficient(
                sd, value=value, b0=b0_work, c0=c0_work, mode=lambda_alpha_mode)
            self_mask = ((raw > 0.0) & (raw < kl_max)).to(dtype)
            if not per_coord:
                coefficient, self_mask = coefficient.unsqueeze(-1), self_mask.unsqueeze(-1)
            grad_mu = self_mask * coefficient * (mu_work - prior_mu) / prior_sigma.clamp(min=eps) + grad_pair
            if capture is not None:
                cost = _self_cost(mu_work, sigma_work, prior_mu, prior_sigma, b0_work,
                                  c0_work, value, kl_max, eps, lambda_alpha_mode, False)
                cost = cost + (_broadcast_tau(beta_scale, energy) * beta * energy).sum()
        else:
            # Clone, not alias: router and templates have no dependency on this node.
            probe = (mu_work.clone() if training_graph else mu_work.detach().clone()).requires_grad_(True)
            self_cost = _self_cost(probe, sigma_work, prior_mu, prior_sigma, b0_work,
                                  c0_work, value, kl_max, eps, lambda_alpha_mode, True, family, full_cov_kl_precision)
            energy = _pair_energy(probe, sigma_work, values, sigma_t, kl_max, eps, dims,
                                  pair_family, full_cov_kl_precision)
            value_energy = _strict_energy(energy, energy, kl_max)
            cost = self_cost + (_broadcast_tau(beta_scale, energy) * beta * value_energy).sum()
            grad_mu, = torch.autograd.grad(cost, probe, create_graph=training_graph)
        if capture is not None:
            context = QKVConditionalContext(
                sigma_work.detach(), prior_mu.detach(), prior_sigma.detach(), beta.detach(),
                values.detach(), sigma_t.detach(), list(dims),
                _detach_coefficient(beta_scale), _detach_coefficient(b0_work),
                _detach_coefficient(c0_work), value, kl_max, eps, lambda_alpha_mode,
                family, pair_family, full_cov_kl_precision)
            prefix = "relational"
            capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                           value_energy_pre=energy.detach(), conditional_context=context,
                           **{f"{prefix}_conditional_cost_before": cost.detach()})
        grad_mu = grad_mu.to(public_dtype)
    return (grad_mu if training_graph else grad_mu.detach()), None
