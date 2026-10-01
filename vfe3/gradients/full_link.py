r"""Frozen, fully connected Gaussian receiver forces.

.. math::
    L_{ij}^{(h)}=U_iO_hT_hU_j^{-1}.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.families.base import safe_kl_clamp
from vfe3.free_energy import attention_weights
from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import _compute_dtype


@dataclass(frozen=True)
class FullLinkContext:
    r"""Episode coefficients of the receiver energy.

    .. math::
        F_i = \frac{1}{2}\mu_i^\top\Lambda_i\mu_i-\eta_i^\top\mu_i
            +\frac{1}{2}\operatorname{tr}(\Lambda_i\Sigma_i)
            -\frac{c_i}{2}\log\det\Sigma_i+\mathrm{constant}.

    Sources and routing have fixed values during the receiver episode. Their tensors
    retain the incoming outer graph for supervised differentiation through inference.
    """

    Lambda: torch.Tensor       # (..., N, K, K), local pooled precision
    eta:    torch.Tensor       # (..., N, K), local precision-weighted target mean
    c:      torch.Tensor       # (..., N), prior plus coupling mass
    beta:   torch.Tensor       # (..., H, N, N), fixed-value projected Q/K routing
    score:  torch.Tensor       # (..., H, N, N), routing cost, distinct from receiving energy


def _head_weights(
    beta:        torch.Tensor,
    lambda_beta: 'float | torch.Tensor',
) -> torch.Tensor:
    r"""Return scalar or head-vector coefficient-weighted routing.

    .. math::
        \lambda_h\beta_{ij}^{(h)}.
    """
    coefficient = torch.as_tensor(lambda_beta, dtype=beta.dtype, device=beta.device)
    if coefficient.ndim > 1 or (coefficient.ndim == 1 and coefficient.shape[0] != beta.shape[-3]):
        raise ValueError("lambda_beta must be a scalar or a vector with one value per head")
    if not bool(torch.isfinite(coefficient).all()) or bool((coefficient < 0).any()):
        raise ValueError("lambda_beta must be finite and nonnegative")
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(coefficient.shape[0], 1, 1)
    return beta * coefficient


def _source_channel(
    mu:        torch.Tensor,                # (..., N, K)
    sigma:     torch.Tensor,                # (..., N, K) or (..., N, K, K)
    frame:     CanonicalFrameContext,
    operators: RelationalOperators,
    noise:     float,
    dtype:     torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return reference precision and natural means.

    .. math::
        R_{hj}=O_hT_hC_jT_h^\top O_h^\top+\nu I,
        m_{hj}=O_hT_h\xi_j.

    Only head/source targets (..., N, H, K, K) are materialized. Receiving noise
    keeps R positive definite when either content map has deficient rank. No
    Cholesky factor of the possibly singular reduced source covariance is needed.
    """
    H, d = len(operators.irrep_dims), operators.irrep_dims[0]
    K = H * d
    T, O = operators.T.to(dtype), operators.O.to(dtype)
    inverse = frame.inverse.to(dtype)
    if frame.block_diagonal:
        # Each full T reads all compact gauge blocks, (..., N, H, d, K).
        action = torch.einsum(
            "hsgu,...nguv->...nhsgv", T.reshape(H, d, H, d), inverse)
        action = action.reshape(*mu.shape[:-1], H, d, K)
    else:
        action = torch.einsum("hsk,...nkm->...nhsm", T, inverse)
    values = torch.einsum("...nhsk,...nk->...nhs", action, mu.to(dtype))
    if sigma.ndim == mu.ndim:
        reduced = torch.einsum(
            "...nhsk,...nk,...nhtk->...nhst", action, sigma.to(dtype), action)
    else:
        reduced = action @ sigma.to(dtype).unsqueeze(-3) @ action.transpose(-1, -2)
    target = torch.einsum("hks,...nhst,hmt->...nhkm", O, reduced, O)
    target = .5 * (target + target.transpose(-1, -2))
    target = target + noise * torch.eye(K, dtype=dtype, device=mu.device)
    factor = torch.linalg.cholesky(target)
    precision = torch.cholesky_inverse(factor)                            # (..., N, H, K, K)
    means = torch.einsum("hks,...nhs->...nhk", O, values)                  # (..., N, H, K)
    natural_mean = (precision @ means.unsqueeze(-1)).squeeze(-1)
    return precision, natural_mean


def _local_pullback(
    precision: torch.Tensor,                # (..., N, K, K), pooled reference precision
    eta:       torch.Tensor,                # (..., N, K), pooled reference natural mean
    frame:     CanonicalFrameContext,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Pull reference information into local coordinates.

    .. math::
        \Lambda_i=U_i^{-\top}P_iU_i^{-1},
        \eta_i=U_i^{-\top}b_i.
    """
    inverse = frame.inverse.to(precision.dtype)
    if frame.block_diagonal:
        H, d = inverse.shape[-3], inverse.shape[-1]
        blocks = precision.reshape(*precision.shape[:-2], H, d, H, d)
        local = torch.einsum("...ngak,...ngahb,...nhbl->...ngkhl", inverse, blocks, inverse)
        local = local.reshape_as(precision)
        eta_h = eta.reshape(*eta.shape[:-1], H, d)
        local_eta = (inverse.transpose(-1, -2) @ eta_h.unsqueeze(-1)).squeeze(-1).reshape_as(eta)
    else:
        local = inverse.transpose(-1, -2) @ precision @ inverse
        local_eta = (inverse.transpose(-1, -2) @ eta.unsqueeze(-1)).squeeze(-1)
    return .5 * (local + local.transpose(-1, -2)), local_eta


def build_full_link_context(
    mu:             torch.Tensor,            # (..., N, K), incoming source/query snapshot
    sigma:          torch.Tensor,            # (..., N, K) or (..., N, K, K)
    mu_p:           torch.Tensor,
    sigma_p:        torch.Tensor,
    frame:          CanonicalFrameContext,
    operators:      RelationalOperators,
    *,
    tau:            'float | torch.Tensor' = 1.0,
    lambda_beta:    'float | torch.Tensor' = 1.0,
    value:          float                 = 1.0,
    receiver_noise: float                 = 1.0,
    kl_max:         float                 = 100.0,
    log_prior:      Optional[torch.Tensor] = None,
    value_frame:    Optional[CanonicalFrameContext] = None,
    capture:        Optional[Dict[str, Any]] = None,
) -> FullLinkContext:
    r"""Snapshot beta and source laws once for receiver filtering with noisy full L links.

    Routing uses the incoming Q/K comparison and frame. Channels use
    value_frame or frame and Q_i = receiver_noise U_i U_i.T. With alpha=value
    and weights lambda_beta beta_ij^h, compute

    .. math::
        \Lambda_i=\alpha\Sigma_{p,i}^{-1}
            +\sum_{h,j}\lambda_h\beta_{ij}^{(h)}R_{ij}^{(h)-1},
        \eta_i=\alpha\Sigma_{p,i}^{-1}\mu_{p,i}
            +\sum_{h,j}\lambda_h\beta_{ij}^{(h)}R_{ij}^{(h)-1}m_{ij}^{(h)},
        c_i=\alpha+\sum_{h,j}\lambda_h\beta_{ij}^{(h)}.

    Neither source nor router is detached. Reuse this context with independent
    receiver arguments to get inner partials while preserving the outer graph.
    relational_score retains its block_glk frame contract.
    """
    if operators.input_scope != "full" or operators.output_scope != "full" or operators.T is None:
        raise ValueError("full-link inference requires learned full input/output content maps")
    if operators.routing != "gaussian_kl":
        raise ValueError("full-link inference requires gaussian_kl routing")
    if not math.isfinite(receiver_noise) or receiver_noise <= 0:
        raise ValueError("receiver_noise must be finite and positive")
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be a finite nonnegative prior coefficient")
    channel_frame = value_frame or frame
    dtype = _compute_dtype(mu, sigma, mu_p, sigma_p, frame.inverse,
                           channel_frame.inverse, operators.T, operators.O)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work, sigma_work = mu.to(dtype), sigma.to(dtype)
        score = safe_kl_clamp(relational_score(
            mu_work, sigma_work, mu_work, sigma_work, frame, operators, capture), kl_max=kl_max)
        beta = attention_weights(score, tau=tau, log_prior=(
            None if log_prior is None else log_prior.to(dtype)))
        weights = _head_weights(beta, lambda_beta)                       # (..., H, I, J)
        source_precision, source_eta = _source_channel(
            mu_work, sigma_work, channel_frame, operators, receiver_noise, dtype)
        pooled_precision = torch.einsum("...hij,...jhkm->...ikm", weights, source_precision)
        pooled_eta = torch.einsum("...hij,...jhk->...ik", weights, source_eta)
        Lambda, eta = _local_pullback(pooled_precision, pooled_eta, channel_frame)
        if sigma_p.ndim == mu_p.ndim:
            prior_precision = torch.diag_embed(sigma_p.to(dtype).reciprocal())
        else:
            prior_precision = torch.cholesky_inverse(torch.linalg.cholesky(sigma_p.to(dtype)))
        Lambda = Lambda + value * prior_precision
        eta = eta + value * (prior_precision @ mu_p.to(dtype).unsqueeze(-1)).squeeze(-1)
        c = value + weights.sum(dim=(-3, -1))                            # (..., I)
        if bool((c <= 0).any()):
            raise ValueError("prior and lambda_beta must give positive total receiver mass")
        context = FullLinkContext(Lambda, eta, c, beta, score)
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                       full_link_context=context)
    return context


def full_link_gradients(
    mu:      torch.Tensor,                   # (..., N, K), fast receiver means
    sigma:   torch.Tensor,                   # (..., N, K) or (..., N, K, K), receiver covariance
    context: FullLinkContext,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return partial receiver derivatives.

    .. math::
        g_{\mu_i}=\Lambda_i\mu_i-\eta_i,
        G_{\Sigma_i}=\frac{1}{2}(\Lambda_i-c_i\Sigma_i^{-1}).

    A diagonal recognition family varies only the diagonal covariance entries,
    giving G_sigma,k = (Lambda_kk-c/sigma_k)/2. The dense precision still enters
    the mean force; no head partition or diagonal target projection is applied.
    """
    dtype = _compute_dtype(mu, sigma, context.Lambda)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        Lambda, eta, c = context.Lambda.to(dtype), context.eta.to(dtype), context.c.to(dtype)
        grad_mu = (Lambda @ mu.to(dtype).unsqueeze(-1)).squeeze(-1) - eta
        if sigma.ndim == mu.ndim:
            grad_sigma = .5 * (Lambda.diagonal(dim1=-2, dim2=-1) - c.unsqueeze(-1) / sigma.to(dtype))
        else:
            receiver_precision = torch.cholesky_inverse(torch.linalg.cholesky(sigma.to(dtype)))
            grad_sigma = .5 * (Lambda - c[..., None, None] * receiver_precision)
    return grad_mu.to(mu.dtype), grad_sigma.to(sigma.dtype)


def full_link_cost(
    mu:      torch.Tensor,
    sigma:   torch.Tensor,
    context: FullLinkContext,
) -> torch.Tensor:
    r"""Return the receiver-dependent energy only.

    .. math::
        F=\sum_i\left[\frac{1}{2}\mu_i^\top\Lambda_i\mu_i-\eta_i^\top\mu_i
            +\frac{1}{2}\operatorname{tr}(\Lambda_i\Sigma_i)
            -\frac{c_i}{2}\log\det\Sigma_i\right].

    Target/prior normalizers and beta entropy are omitted. This scalar checks
    receiver descent with a fixed context; it is not a normalized VFE or the
    full parameter-dependent training objective.
    """
    dtype = _compute_dtype(mu, sigma, context.Lambda)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mean, covariance = mu.to(dtype), sigma.to(dtype)
        Lambda, eta, c = context.Lambda.to(dtype), context.eta.to(dtype), context.c.to(dtype)
        quadratic = (mean * (Lambda @ mean.unsqueeze(-1)).squeeze(-1)).sum(-1)
        linear = (mean * eta).sum(-1)
        if sigma.ndim == mu.ndim:
            trace = (Lambda.diagonal(dim1=-2, dim2=-1) * covariance).sum(-1)
            logdet = covariance.log().sum(-1)
        else:
            trace = (Lambda * covariance.transpose(-1, -2)).sum(dim=(-2, -1))
            factor = torch.linalg.cholesky(covariance)
            logdet = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return (.5 * quadratic - linear + .5 * trace - .5 * c * logdet).sum()
