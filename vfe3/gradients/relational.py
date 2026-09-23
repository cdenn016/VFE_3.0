"""Exact projected Gaussian routing and family-aware conditional content forces."""

from typing import Any, Dict, Optional, Tuple

import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.qkv import (
    Coefficient, Transport, _DIAGONAL_KL, _DIAGONAL_KL_PER_COORD,
    qkv_filtering_gradients,
)
from vfe3.divergence import get_functional, get_functional_per_coord, renyi, renyi_per_coord
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import projected_moments, projected_gaussian_kl


def relational_score(
    mu:       torch.Tensor,
    sigma:    torch.Tensor,
    source_mu: torch.Tensor,
    source_sigma: torch.Tensor,
    frame:    CanonicalFrameContext,
    operators: RelationalOperators,
    capture: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    r"""Compute KL(S_Q U_Qi^-1 q_i || S_K U_Kj^-1 q_j), retaining full reduced covariance."""
    H, d = len(operators.irrep_dims), operators.irrep_dims[0]
    inverse = frame.inverse
    if not frame.block_diagonal:
        inverse = torch.stack(
            [inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
    # Both roles read the one shared frame: there is no separate key-side inverse.
    key_inverse = inverse
    if operators.input_scope == "full":
        # Broadcast the same whole belief to readout heads; gauge blocks stay compact.
        mu_h, source_mu_h = (
            x.unsqueeze(-2).expand(*x.shape[:-1], H, H*d) for x in (mu, source_mu))
        sigma_h, source_sigma_h = (
            x.unsqueeze(-3).expand(*x.shape[:-2], H, H*d, H*d)
            if x.ndim == mu.ndim + 1 else x.unsqueeze(-2).expand(*x.shape[:-1], H, H*d)
            for x in (sigma, source_sigma))
    else:
        mu_h, source_mu_h = (
            x.reshape(*x.shape[:-1], H, d) for x in (mu, source_mu))
        sigma_h, source_sigma_h = (
            torch.stack([x[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
            if x.ndim == mu.ndim + 1 else x.reshape(*x.shape[:-1], H, d)
            for x in (sigma, source_sigma))
    S_q, S_k = operators.comparison_pair()
    a_q, G_q = projected_moments(mu_h, sigma_h, inverse, S_q)
    a_k, G_k = projected_moments(source_mu_h, source_sigma_h, key_inverse, S_k)
    if capture is not None:
        capture["projected_covariance_pre"] = G_q.detach()
        capture["projected_key_covariance_pre"] = G_k.detach()
    return projected_gaussian_kl(
        a_q, G_q, a_k, G_k,
        projection_q=(sigma_h, inverse, S_q, mu_h),
        projection_k=(source_sigma_h, key_inverse, S_k, source_mu_h))


def compute_relational_filtering_gradients(
    mu:                torch.Tensor,             # (..., N, K)
    sigma:             torch.Tensor,             # (..., N, K) diagonal or (..., N, K, K) full covariance
    mu_p:              torch.Tensor,
    sigma_p:           torch.Tensor,
    omega:             Transport,
    operators:         RelationalOperators,
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
    log_prior:         Optional[torch.Tensor] = None,
    capture:           Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, None]:
    r"""Return partial_z C(z; beta_pre, U_i O T xi_j), evaluated at z=mu.

    The score uses exact canonical Gaussian marginals. Conditional values retain
    the family transport covariance (diagonal for the legacy diagonal family,
    full for full/exact Gaussian families), not an OT covariance pushforward.
    Outer gradients through maps and routing remain live.
    """
    if (get_functional("renyi") is not renyi
            or get_functional_per_coord("renyi") is not renyi_per_coord
            or DiagonalGaussian.renyi_closed_form is not _DIAGONAL_KL
            or DiagonalGaussian.renyi_per_coord is not _DIAGONAL_KL_PER_COORD):
        raise NotImplementedError("relational filtering requires the built-in Gaussian KL registration")
    return qkv_filtering_gradients(
        mu, sigma, mu_p, sigma_p, omega, frame,
        tau=tau, lambda_beta=lambda_beta, b0=b0, c0=c0, kl_max=kl_max,
        eps=eps, value=value, create_graph=create_graph,
        lambda_alpha_mode=lambda_alpha_mode, log_prior=log_prior,
        capture=capture, relational_operators=operators,
        family=family, value_frame=value_frame, full_cov_kl_precision=full_cov_kl_precision)


def relational_predictive_readout(
    mu:        torch.Tensor,
    sigma:     torch.Tensor,
    frame:     CanonicalFrameContext,
    operators: RelationalOperators,
    *,
    tau:       Coefficient = 1.0,
    kl_max:    float = 100.0,
    log_prior:   Optional[torch.Tensor] = None,
    value_frame: Optional[CanonicalFrameContext] = None,
    capture:     Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    r"""Return the conditional predictive mean with KL routing and live source graphs.

    This is a separate content mixture, not a Fisher step on the comparison belief.
    Its covariance is not estimated here. No source, router, or map is detached.
    """
    from vfe3.families.base import safe_kl_clamp
    from vfe3.free_energy import attention_weights
    dtype = torch.float64 if mu.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work = mu.to(dtype)
        score = safe_kl_clamp(relational_score(
            mu_work, sigma.to(dtype), mu_work, sigma.to(dtype), frame, operators, capture), kl_max=kl_max)
        beta = attention_weights(score, tau=tau,
                                 log_prior=None if log_prior is None else log_prior.to(dtype))
        result = operators.predictive_content(mu_work, beta, value_frame or frame).to(mu.dtype)
        if operators.attention_residual:
            result = mu + result
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                       relational_mean_displacement_norm=(result-mu).norm().detach())
    return result


