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
from vfe3.relational_scoring import projected_moments, projected_gaussian_kl, projected_noisy_query_cost


def relational_score(
    mu:           torch.Tensor,                        # (..., N, K) receiver (query) means
    sigma:        torch.Tensor,                        # (..., N, K) diagonal or (..., N, K, K) full covariance
    source_mu:    torch.Tensor,                        # (..., N, K) source (key) means
    source_sigma: torch.Tensor,                        # (..., N, K) or (..., N, K, K), same layout as sigma
    frame:        CanonicalFrameContext,               # (..., N, H, d, d) compact or (..., N, K, K) dense
    operators:    RelationalOperators,
    capture:      Optional[Dict[str, Any]] = None,     # out-param: pre-update projected covariances
) -> torch.Tensor:                                     # (..., H, N, N) projected routing costs
    r"""Compute KL(S_Q U_i^{-1} q_i || S_K U_j^{-1} q_j) or the noisy-query cost, retaining full reduced covariance.

    ``operators.routing`` selects the cost: ``"gaussian_kl"`` (projected_gaussian_kl) or
    ``"noisy_query"`` (projected_noisy_query_cost, the expected-likelihood source posterior).

    The comparison reads each head's canonical coordinates through the head blocks of
    :math:`U^{-1}`, which equal the full canonical coordinates only when :math:`U^{-1}` is
    block-diagonal. A dense-layout frame is therefore admitted only when its inverse has no
    off-diagonal head block; a genuinely dense frame is rejected rather than silently truncated, since
    the content side (``canonical_mean`` / ``predictive_content``) always applies the full inverse.
    """
    H, d = len(operators.irrep_dims), operators.irrep_dims[0]
    inverse = frame.inverse
    if not frame.block_diagonal:
        if H > 1:
            with torch.no_grad():
                block_mask = torch.block_diag(
                    *(torch.ones(d, d, dtype=torch.bool, device=inverse.device) for _ in range(H)))
                if bool((inverse.detach()[..., ~block_mask] != 0).any()):       # one host sync
                    raise ValueError(
                        "relational_score requires a head-block-diagonal frame: the dense frame "
                        "inverse couples gauge heads, and scoring its diagonal blocks would disagree "
                        "with the full-inverse content readout")
        inverse = torch.stack(
            [inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)     # (..., N, H, d, d)
    # Both roles read the one shared frame: there is no separate key-side inverse.
    key_inverse = inverse
    if operators.input_scope == "full":
        # Broadcast the same whole belief to readout heads; gauge blocks stay compact.
        mu_h, source_mu_h = (                                                   # (..., N, H, K)
            x.unsqueeze(-2).expand(*x.shape[:-1], H, H*d) for x in (mu, source_mu))
        sigma_h, source_sigma_h = (                                             # (..., N, H, K[, K])
            x.unsqueeze(-3).expand(*x.shape[:-2], H, H*d, H*d)
            if x.ndim == mu.ndim + 1 else x.unsqueeze(-2).expand(*x.shape[:-1], H, H*d)
            for x in (sigma, source_sigma))
    else:
        mu_h, source_mu_h = (                                                   # (..., N, H, d)
            x.reshape(*x.shape[:-1], H, d) for x in (mu, source_mu))
        sigma_h, source_sigma_h = (                                             # (..., N, H, d[, d])
            torch.stack([x[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
            if x.ndim == mu.ndim + 1 else x.reshape(*x.shape[:-1], H, d)
            for x in (sigma, source_sigma))
    S_q, S_k = operators.comparison_pair()                                     # each (H, d, input_dim)
    if mu is source_mu and sigma is source_sigma and inverse is key_inverse:
        # Predictive readout scores a belief against itself: one batched projection over both
        # comparison maps. S_q and S_k still differ, so two L = S U^-1 maps are formed, but the
        # covariance contraction G = L Sigma L^T is batched across the role axis.
        a, G = projected_moments(mu_h, sigma_h, inverse, torch.stack((S_q, S_k)))
        a_q, G_q = a[0], G[0]
        a_k, G_k = a[1], G[1]
    else:
        a_q, G_q = projected_moments(mu_h, sigma_h, inverse, S_q)              # (..., N, H, d), (..., N, H, d, d)
        a_k, G_k = projected_moments(source_mu_h, source_sigma_h, key_inverse, S_k)
    if capture is not None:
        capture["projected_covariance_pre"] = G_q.detach()
        capture["projected_key_covariance_pre"] = G_k.detach()
    score = projected_noisy_query_cost if operators.routing == "noisy_query" else projected_gaussian_kl
    return score(
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
    r"""Return the conditional predictive mean with the configured projected routing cost and live source graphs.

    This is a separate content mixture, not a Fisher step on the comparison belief.
    Its covariance is not estimated here. No source, router, or map is detached.
    """
    from vfe3.families.base import safe_kl_clamp
    from vfe3.free_energy import attention_weights
    dtype = torch.float64 if mu.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work = mu.to(dtype)
        sigma_work = sigma.to(dtype)
        score = safe_kl_clamp(relational_score(
            mu_work, sigma_work, mu_work, sigma_work, frame, operators, capture), kl_max=kl_max)
        beta = attention_weights(score, tau=tau,
                                 log_prior=None if log_prior is None else log_prior.to(dtype))
        result = operators.predictive_content(mu_work, beta, value_frame or frame).to(mu.dtype)
        if operators.attention_residual:
            result = mu + result
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                       relational_mean_displacement_norm=(result-mu).norm().detach())
    return result


