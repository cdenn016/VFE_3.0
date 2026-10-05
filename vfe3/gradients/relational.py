"""Full-input projected Gaussian routing and predictive content readout."""

from typing import Any, Dict, Optional, Tuple

import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import (
    _compute_dtype, projected_moments, projected_gaussian_kl, projected_noisy_query_cost,
)


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

    Every head projects the full canonical state through its own Q/K maps. The
    inverse frame retains its compact gauge blocks. A dense-layout frame must
    represent the same block-diagonal structure; cross-head frame coupling is
    outside the supported projected architecture.
    """
    dims = operators.irrep_dims
    inverse = frame.inverse
    if not frame.block_diagonal:
        if len(dims) > 1:
            with torch.no_grad():
                block_mask = torch.block_diag(
                    *(torch.ones(d, d, dtype=torch.bool, device=inverse.device) for d in dims))
                if bool((inverse.detach()[..., ~block_mask] != 0).any()):       # one host sync
                    raise ValueError(
                        "relational_score requires a head-block-diagonal frame: the dense frame "
                        "inverse couples gauge heads, and scoring its diagonal blocks would disagree "
                        "with the full-inverse content readout")
        if operators.mixed_heads:
            # Every head reads the whole belief through the whole frame, L_h = S_h U^{-1}. Blocks of
            # different sizes have no compact stack, so each run reads the dense inverse as one
            # full-width block, (..., N, 1, K, K).
            inverse_runs = [inverse.unsqueeze(-3)] * len(operators.runs)
        else:
            d = dims[0]
            inverse_runs = [torch.stack(                                       # (..., N, H, d, d)
                [inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(len(dims))], dim=-3)]
    elif operators.mixed_heads:
        raise ValueError("a compact (..., N, H, d, d) frame cannot carry heads of different sizes")
    else:
        inverse_runs = [inverse]
    comparison = operators.comparison_runs()                                   # per run (n_r, d_r, K) pairs
    # An escalated float64 frame (audit 2026-09-23 #12) runs the whole projection and score in
    # float64; the results are returned at the dtype a lower-precision frame would have produced.
    public = _compute_dtype(mu, sigma, source_mu, source_sigma, *(S for pair in comparison for S in pair))
    score = projected_noisy_query_cost if operators.routing == "noisy_query" else projected_gaussian_kl
    scores, captured_q, captured_k = [], [], []
    # One scoring call per run: the heads of a run share their rank d_r. Each call ends in its own
    # host transfer, so a tower of R runs pays R of them per call (one for equal heads).
    for (S_q, S_k), inverse in zip(comparison, inverse_runs):
        H, K = S_q.shape[0], S_q.shape[-1]
        # Both roles read the one shared frame: there is no separate key-side inverse.
        key_inverse = inverse
        # Each head reads the whole belief; gauge blocks stay compact.
        mu_h, source_mu_h = (                                                   # (..., N, H, K)
            x.unsqueeze(-2).expand(*x.shape[:-1], H, K) for x in (mu, source_mu))
        sigma_h, source_sigma_h = (                                             # (..., N, H, K[, K])
            x.unsqueeze(-3).expand(*x.shape[:-2], H, K, K)
            if x.ndim == mu.ndim + 1 else x.unsqueeze(-2).expand(*x.shape[:-1], H, K)
            for x in (sigma, source_sigma))
        if mu is source_mu and sigma is source_sigma and inverse is key_inverse:
            # Predictive readout scores a belief against itself: one batched projection over both
            # comparison maps. S_q and S_k still differ, so two L = S U^-1 maps are formed, but the
            # covariance contraction G = L Sigma L^T is batched across the role axis.
            a, G = projected_moments(mu_h, sigma_h, inverse, torch.stack((S_q, S_k)))
            a_q, G_q = a[0], G[0]
            a_k, G_k = a[1], G[1]
        else:
            a_q, G_q = projected_moments(mu_h, sigma_h, inverse, S_q)          # (..., N, H, d), (..., N, H, d, d)
            a_k, G_k = projected_moments(source_mu_h, source_sigma_h, key_inverse, S_k)
        if capture is not None:
            captured_q.append(G_q.detach().to(public))
            captured_k.append(G_k.detach().to(public))
        scores.append(score(
            a_q, G_q, a_k, G_k,
            projection_q=(sigma_h, inverse, S_q, mu_h),
            projection_k=(source_sigma_h, key_inverse, S_k, source_mu_h)).to(public))
    if capture is not None:
        # One (..., N, H, r, r) tensor for equal heads; a list with one per run for mixed head sizes.
        capture["projected_covariance_pre"] = captured_q[0] if len(captured_q) == 1 else captured_q
        capture["projected_key_covariance_pre"] = captured_k[0] if len(captured_k) == 1 else captured_k
    return scores[0] if len(scores) == 1 else torch.cat(scores, dim=-3)      # (..., H, N, N)


def relational_predictive_readout(
    mu:        torch.Tensor,
    sigma:     torch.Tensor,
    frame:     CanonicalFrameContext,
    operators: RelationalOperators,
    *,
    tau:       'float | torch.Tensor' = 1.0,
    kl_max:    float = 100.0,
    log_prior:   Optional[torch.Tensor] = None,
    value_frame: Optional[CanonicalFrameContext] = None,
    capture:     Optional[Dict[str, Any]] = None,
    return_covariance: bool = False,
) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor]':
    r"""Return the conditional predictive mean with the configured projected routing cost and live source graphs.

    This is a separate content mixture, not a Fisher step on the comparison belief. With
    ``return_covariance`` it also returns ``(mean, sigma_plus)``, where sigma_plus is the
    moment-matched covariance of the same predictive kernel
    (``RelationalOperators.predictive_covariance_update``), computed from the same routing weights
    and value frame as the mean. No source, router, or map is detached.
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
        _, receive = operators.content_factor_runs()         # O' once per layer for mean and covariance
        result = operators.predictive_content(mu_work, beta, value_frame or frame, receive).to(mu.dtype)
        if operators.attention_residual:
            result = mu + result
        sigma_plus = (operators.predictive_covariance_update(mu_work, sigma_work, beta, value_frame or frame,
                                                             receive).to(sigma.dtype)
                      if return_covariance else None)
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                       relational_mean_displacement_norm=(result-mu).norm().detach())
    if return_covariance:
        return result, sigma_plus
    return result
