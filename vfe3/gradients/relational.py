"""Full-input projected Gaussian routing and predictive content readout."""

from typing import Any, Dict, Optional, Tuple

import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.edge_links import edge_diagnostic_capture, receive_heads as edge_receive_heads
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import _compute_dtype, projected_moments, routing_cost
from vfe3.runtime import ieee_fp32_matmul


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
    ``"noisy_query"`` (projected_noisy_query_cost, the expected-likelihood source posterior), or
    ``"mean_distance"`` (projected_mean_distance_cost, the covariance-blind control).

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
    score = routing_cost(operators.routing)                                  # ROUTING_COSTS registry
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
        projection_q = (sigma_h, inverse, S_q, mu_h)
        projection_k = (source_sigma_h, key_inverse, S_k, source_mu_h)
        rope_base = getattr(operators, "comparison_rope_base", None)
        if rope_base is not None:
            # rope_space='comparison' (audit 2026-10-07, F28): rotate token i's projected moments by R_i in the
            # r-dimensional comparison space, a -> R_i a, G -> R_i G R_i^T, for query and key alike. A Gaussian KL
            # (and the noisy-query and mean-distance costs) is invariant under a common orthogonal map, so the
            # pair (i, j) cost equals the cost of (a_i, G_i) against (R_{j-i} b_j, R_{j-i} G_j R_{j-i}^T): a function
            # of positions only through j - i. The rotated moments are those of the per-token comparison map
            # R_i S, so the source tuples carry R_i S and the source-root recovery and factor route factor
            # X_i = R_i S U_i^{-1} Sigma_i^{1/2} exactly as on the unrotated route (audit 2026-10-08 P1-2).
            # The rotation runs under the same IEEE-fp32 pin as the moments it rotates (P1-1).
            r = S_q.shape[-2]
            R_q = operators.comparison_rope_rotation(a_q.shape[-3], r, a_q.device, a_q.dtype)  # (N, r, r)
            R_k = R_q if a_k.shape[-3] == a_q.shape[-3] else operators.comparison_rope_rotation(
                a_k.shape[-3], r, a_k.device, a_k.dtype)
            with ieee_fp32_matmul():
                a_q = torch.einsum("nst,...nht->...nhs", R_q, a_q)
                a_k = torch.einsum("nst,...nht->...nhs", R_k, a_k)
                G_q = torch.einsum("nst,...nhtu,nvu->...nhsv", R_q, G_q, R_q)
                G_k = torch.einsum("nst,...nhtu,nvu->...nhsv", R_k, G_k, R_k)
                S_q_rot = torch.einsum("nst,htk->nhsk", R_q.to(S_q.dtype), S_q)   # (N, H, r, K)
                S_k_rot = torch.einsum("nst,htk->nhsk", R_k.to(S_k.dtype), S_k)
            G_q, G_k = 0.5 * (G_q + G_q.transpose(-1, -2)), 0.5 * (G_k + G_k.transpose(-1, -2))
            projection_q = (sigma_h, inverse, S_q_rot, mu_h)
            projection_k = (source_sigma_h, key_inverse, S_k_rot, source_mu_h)
        scores.append(score(
            a_q, G_q, a_k, G_k,
            projection_q=projection_q,
            projection_k=projection_k).to(public))
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
    link_state:  Optional[dict] = None,
) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor]':
    r"""Return the conditional predictive mean with the configured projected routing cost and live source graphs.

    This is a separate content mixture, not a Fisher step on the comparison belief. With
    ``return_covariance`` it also returns ``(mean, sigma_plus)``, where sigma_plus is the
    moment-matched covariance of the same predictive kernel
    (``RelationalOperators.predictive_covariance_update``), computed from the same routing weights
    and value frame as the mean. No source, router, or map is detached.
    Under ``content_link_edges`` the edge context (factors, :math:`\Phi_{ij}` and the float64 chart
    stacks) is computed once per layer and shared by the mean, the covariance and the capture.
    Under ``content_link_latent`` its receiver factors are inferred from the source covariances of
    ``sigma`` (whatever ``predictive_covariance`` is), and a positive loop weight reads and writes
    ``link_state`` (``{"previous": ..., "current": ...}``, threaded across layers by ``vfe_stack``).

    Under ``predictive_readout='minimal_agent'`` the routing, the content and the residual are replaced by the
    exact Gaussian conditioning of :meth:`RelationalOperators.minimal_agent_update`, which reads the ROUTING frame
    for query, key, content and write-back (a distinct ``value_frame`` with different matrices is refused: the tied
    key is the content). ``tau`` and ``kl_max`` are not read (the posterior has temperature 1 and no clamp); the
    result is the posterior mean, with no residual added, and ``return_covariance`` gives the posterior covariance
    plus the covariant floor.
    """
    if getattr(operators, "predictive_readout", "kernel") == "minimal_agent":
        if value_frame is not None and value_frame is not frame and not (
                value_frame.block_diagonal == frame.block_diagonal
                and torch.equal(value_frame.forward, frame.forward)
                and torch.equal(value_frame.inverse, frame.inverse)):
            raise ValueError("predictive_readout='minimal_agent' reads key and content in one frame; the value frame "
                             "differs from the routing frame (pos_rotation='rope' with rope_on_value=False)")
        mean, sigma_plus = operators.minimal_agent_update(
            mu, sigma, frame, log_prior, return_covariance=return_covariance, capture=capture)
        return (mean, sigma_plus) if return_covariance else mean
    from vfe3.families.base import safe_kl_clamp
    from vfe3.free_energy import attention_weights
    dtype = torch.float64 if mu.dtype == torch.float64 else torch.float32
    edges = None
    latent = getattr(operators, "content_link_latent", False)
    latent_out: Optional[Dict[str, Any]] = {} if (latent and capture is not None) else None
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work = mu.to(dtype)
        sigma_work = sigma.to(dtype)
        score = relational_score(mu_work, sigma_work, mu_work, sigma_work, frame, operators, capture)
        if getattr(operators, "kl_max_reference", "absolute") == "row_min":
            # Clamp the gap to the row's smallest admissible finite cost (audit 2026-10-08 P1-4). The
            # softmax is row-shift invariant, so beta is unchanged wherever a row keeps an entry below
            # the cap, and a row whose every cost exceeds kl_max still routes by its cost order.
            with torch.no_grad():
                admissible = torch.isfinite(score)
                if log_prior is not None:
                    admissible = admissible & torch.isfinite(log_prior.to(dtype))
                row_min = score.masked_fill(~admissible, float("inf")).amin(dim=-1, keepdim=True)
                row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
            score = score - row_min                                             # (..., H, N, N)
        score = safe_kl_clamp(score, kl_max=kl_max)
        beta = attention_weights(score, tau=tau,
                                 log_prior=None if log_prior is None else log_prior.to(dtype))
        if getattr(operators, "content_link_edges", "off") != "off":
            inference = ({} if not latent else dict(
                sigma=sigma_work, log_prior=None if log_prior is None else log_prior.to(dtype),
                link_state=link_state, latent_out=latent_out))
            edges = operators.edge_context(mu_work, beta, value_frame or frame, **inference)
            receive = [edge_receive_heads(edges, len(operators.irrep_dims))]
        else:
            _, receive = operators.content_factor_runs()         # O' once per layer for mean and covariance
        shared = {} if edges is None else {"edges": edges}
        result = operators.predictive_content(mu_work, beta, value_frame or frame, receive,
                                              **shared).to(mu.dtype)
        if operators.attention_residual:
            result = mu + result
        sigma_plus = (operators.predictive_covariance_update(mu_work, sigma_work, beta, value_frame or frame,
                                                             receive, **shared).to(sigma.dtype)
                      if return_covariance else None)
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                       relational_mean_displacement_norm=(result-mu).norm().detach())
        if edges is not None:
            capture.update(edge_diagnostic_capture(edges, beta))
        if latent_out is not None:
            capture.update(latent_out)
    if return_covariance:
        return result, sigma_plus
    return result
