r"""Frozen, fully connected Gaussian receiver forces.

.. math::
    L_{ij}^{(h)}=U_iO_hT_hU_j^{-1}.

Two receiver laws share one quadratic energy family (projected_gradient.tex, Eq. quadratic_family).
``ambient_alignment`` aligns the receiver to K-dimensional noisy receiving laws and counts the
receiver entropy c = alpha + sum_h lambda_h times. ``channel_likelihood`` scores the receiver's
partial law O_h^T xi_i against the sender's d-dimensional message as an expected likelihood and
counts it once (c = alpha).

A state-dependent self-coupling alpha*(D) = c0/(b0 + D), D = KL(q_i||p_i) (summed per token, or per
local coordinate on the diagonal family), is profiled out (projected_gradient.tex, Sec. profiled
self-coupling): the receiver energy becomes c0 log(b0 + D) + G(q), its exact force is alpha*(q) dD/dq
+ dG/dq, and freezing alpha at the current state gives a majorizer whose exact minimizer is the
existing closed form.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from vfe3.alpha_i import alpha_gradient_coefficient
from vfe3.contracts import CanonicalFrameContext
from vfe3.families.base import safe_kl_clamp
from vfe3.free_energy import attention_weights
from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import _compute_dtype, _count_source_recovery


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
    c:      torch.Tensor       # (..., N), entropy multiplicity: alpha + sum_h lambda_h (ambient) or alpha (channel)
    beta:   torch.Tensor       # (..., H, N, N), fixed-value projected Q/K routing
    score:  torch.Tensor       # (..., H, N, N), routing cost, distinct from receiving energy
    profile: Optional["SelfCouplingProfile"] = None   # state-dependent alpha; None = constant alpha


@dataclass(frozen=True)
class SelfCouplingProfile:
    r"""Coupling-only receiver coefficients and the layer prior for a profiled self-coupling.

    .. math::
        E_i(q)=c_0\log(b_0+D_i(q))+G_i(q),\qquad D_i=\mathrm{KL}(q_i\|p_i),

    with G the receiver energy at alpha = 0 (coefficients Lambda_c, eta_c and entropy multiplicity
    m = sum_h lambda_h under ambient_alignment, 0 under channel_likelihood). ``per_coord`` replaces
    c0 log(b0+D) by sum_k c0_k log(b0_k + D_ik) over local coordinates of the diagonal family.
    """

    mode:            str                      # 'state_dependent' | 'state_dependent_per_coord'
    b0:              'float | torch.Tensor'   # scalar, or (K,) per coordinate
    c0:              'float | torch.Tensor'
    mu_p:            torch.Tensor             # (..., N, K), layer prior mean (local coordinates)
    sigma_p:         torch.Tensor             # (..., N, K) or (..., N, K, K)
    prior_precision: torch.Tensor             # (..., N, K, K)
    prior_eta:       torch.Tensor             # (..., N, K), Sigma_p^{-1} mu_p
    prior_logdet:    torch.Tensor             # (..., N), log det Sigma_p
    Lambda:          torch.Tensor             # (..., N, K, K), coupling-only precision
    eta:             torch.Tensor             # (..., N, K), coupling-only natural mean
    mass:            torch.Tensor             # (..., N), coupling-only entropy multiplicity

    @property
    def per_coord(self) -> bool:
        return self.mode == "state_dependent_per_coord"


_PROFILED_MODES = ("state_dependent", "state_dependent_per_coord")
FULL_LINK_ALPHA_MODES = ("constant",) + _PROFILED_MODES


def self_coupling_divergence(
    mu:      torch.Tensor,                   # (..., N, K)
    sigma:   torch.Tensor,                   # (..., N, K) or (..., N, K, K)
    profile: SelfCouplingProfile,
) -> torch.Tensor:
    r"""Exact, unclamped KL(q_i||p_i), per token (..., N) or per local coordinate (..., N, K).

    .. math::
        D_i=\tfrac12\bigl[\operatorname{tr}(\Sigma_p^{-1}\Sigma)+(\mu-\mu_p)^\top\Sigma_p^{-1}(\mu-\mu_p)
            -K+\log\det\Sigma_p-\log\det\Sigma\bigr],\qquad
        D_{ik}=\tfrac12\Bigl[\frac{s_k}{p_k}+\frac{(\mu_k-\mu_{p,k})^2}{p_k}-1-\log\frac{s_k}{p_k}\Bigr].

    No kl_max ceiling: a clamp would zero the force beyond it and break the envelope identity.
    """
    dtype = _compute_dtype(mu, sigma, profile.prior_precision)
    mean, cov = mu.to(dtype), sigma.to(dtype)
    delta = mean - profile.mu_p.to(dtype)
    if profile.per_coord:
        sigma_p = profile.sigma_p.to(dtype)
        ratio = cov / sigma_p
        return .5 * (ratio + delta.square() / sigma_p - 1 - ratio.log())
    P = profile.prior_precision.to(dtype)
    quadratic = (delta * (P @ delta.unsqueeze(-1)).squeeze(-1)).sum(-1)
    if cov.ndim == mean.ndim:
        trace = (P.diagonal(dim1=-2, dim2=-1) * cov).sum(-1)
        logdet = cov.log().sum(-1)
    else:
        trace = (P * cov.transpose(-1, -2)).sum(dim=(-2, -1))
        logdet = 2 * torch.linalg.cholesky(cov).diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return .5 * (trace + quadratic - mean.shape[-1] + profile.prior_logdet.to(dtype) - logdet)


def _coeff(value: 'float | torch.Tensor', like: torch.Tensor) -> 'float | torch.Tensor':
    return value.to(device=like.device, dtype=like.dtype) if torch.is_tensor(value) else value


def full_link_context_at(
    context: FullLinkContext,
    mu:      torch.Tensor,                   # (..., N, K), current receiver state
    sigma:   torch.Tensor,
) -> FullLinkContext:
    r"""Freeze a state-dependent alpha at the current receiver and return the frozen-alpha context.

    .. math::
        \alpha_i=\frac{c_0}{b_0+D_i(q)},\quad
        \Lambda_i=\Lambda_{c,i}+\alpha_i\Sigma_{p,i}^{-1},\quad
        \eta_i=\eta_{c,i}+\alpha_i\Sigma_{p,i}^{-1}\mu_{p,i},\quad c_i=\alpha_i+m_i.

    Per coordinate, alpha_i Sigma_p^{-1} becomes diag(alpha_ik / p_ik) and c_i the vector
    alpha_ik + m_i. The returned context's force alpha* dD/dq + dG/dq is the exact gradient of the
    profiled energy (envelope theorem), and its minimizer is the MM step. alpha keeps its graph: the
    force value is unchanged, and the unrolled outer derivative of the force map needs d alpha*/d theta.
    A constant-alpha context is returned unchanged.
    """
    profile = context.profile
    if profile is None:
        return context
    with torch.autocast(device_type=mu.device.type, enabled=False):
        D = self_coupling_divergence(mu, sigma, profile)                 # (..., N) or (..., N, K)
        alpha = alpha_gradient_coefficient(
            D, b0=_coeff(profile.b0, D), c0=_coeff(profile.c0, D), mode=profile.mode)
        if profile.per_coord:
            sigma_p = profile.sigma_p.to(D.dtype)
            Lambda = profile.Lambda + torch.diag_embed(alpha / sigma_p)
            eta = profile.eta + alpha * profile.mu_p.to(D.dtype) / sigma_p
            c = alpha + profile.mass.unsqueeze(-1)                       # (..., N, K)
        else:
            Lambda = profile.Lambda + alpha[..., None, None] * profile.prior_precision
            eta = profile.eta + alpha.unsqueeze(-1) * profile.prior_eta
            c = alpha + profile.mass                                     # (..., N)
    return FullLinkContext(Lambda, eta, c, context.beta, context.score, profile)


def _diagonal_weight(c: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    r"""Broadcast an entropy multiplicity to diagonal variances: (..., N) -> (..., N, 1); (..., N, K) kept."""
    return c if c.ndim == mu.ndim else c.unsqueeze(-1)


def _scalar_weight(c: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    if c.ndim == mu.ndim:
        raise ValueError("per-coordinate entropy weights exist only on the diagonal family")
    return c


def _head_weights(
    beta:        torch.Tensor,
    lambda_beta: 'float | torch.Tensor',
) -> torch.Tensor:
    r"""Return scalar or head-vector coefficient-weighted routing.

    .. math::
        \lambda_h\beta_{ij}^{(h)}.
    """
    # Validate on the host value (config-validated floats or CPU tensors) so a device-resident
    # context pays no synchronization; a device tensor is trusted as validated upstream.
    host = lambda_beta if not torch.is_tensor(lambda_beta) or lambda_beta.device.type == "cpu" else None
    coefficient = torch.as_tensor(lambda_beta, dtype=beta.dtype, device=beta.device)
    if coefficient.ndim > 1 or (coefficient.ndim == 1 and coefficient.shape[0] != beta.shape[-3]):
        raise ValueError("lambda_beta must be a scalar or a vector with one value per head")
    if host is not None:
        host = torch.as_tensor(host, dtype=torch.float64)
        if not bool(torch.isfinite(host).all()) or bool((host < 0).any()):
            raise ValueError("lambda_beta must be finite and nonnegative")
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(coefficient.shape[0], 1, 1)
    return beta * coefficient


def _spd_precision(target: torch.Tensor) -> torch.Tensor:
    r"""Return target^{-1} from a zero-jitter Cholesky factor, retrying a failed batch in float64.

    A float32 factorization fails near cond(target) ~ 1/eps_32. The retry recomputes the batch in
    float64 (one host sync on the info tensor), counts the failed entries in the shared
    source-recovery counter, and raises only if float64 also fails.
    """
    factor, info = torch.linalg.cholesky_ex(target)
    failed = info != 0
    if bool(failed.any()):                                                  # one host sync
        if target.dtype == torch.float64:
            raise torch.linalg.LinAlgError("receiving-law covariance is not positive definite")
        _count_source_recovery(failed)
        factor64, info64 = torch.linalg.cholesky_ex(target.to(torch.float64))
        if bool((info64 != 0).any()):
            raise torch.linalg.LinAlgError("receiving-law covariance is not positive definite in float64")
        return torch.cholesky_inverse(factor64).to(target.dtype)
    return torch.cholesky_inverse(factor)


def _source_channel(
    mu:        torch.Tensor,                # (..., N, K)
    sigma:     torch.Tensor,                # (..., N, K) or (..., N, K, K)
    frame:     CanonicalFrameContext,
    operators: RelationalOperators,
    noise:     float,
    dtype:     torch.dtype,
    content:   Optional[Tuple[torch.Tensor, torch.Tensor]] = None,   # precomputed (T, O'); None evaluates
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return per-source, per-head precisions and natural means.

    ``ambient_alignment`` lifts each message to the K-dimensional law

    .. math::
        R_{hj}=O_hT_hC_jT_h^\top O_h^\top+\nu I_K,\qquad m_{hj}=O_hT_h\xi_j,

    returning (..., N, H, K, K) and (..., N, H, K); the noise is required for finiteness when d<K.
    ``channel_likelihood`` keeps the message in the d-dimensional value space,

    .. math::
        R_{hj}=T_hC_jT_h^\top+\nu I_d,\qquad v_{hj}=T_h\xi_j,

    returning (..., N, H, d, d) and (..., N, H, d); there nu is a declared channel floor.

    Both are returned as lists with one entry per run of equal-size heads (``operators.runs``),
    the run's heads in place of H and its size in place of d; one entry for equal heads.
    """
    T_runs, O_runs = content or operators.content_factor_runs()            # O' of the chart
    inverse = frame.inverse.to(dtype)
    channels = [_source_channel_run(mu, sigma, inverse, frame.block_diagonal, T.to(dtype), O.to(dtype),
                                    operators.receiver_law, noise, dtype)
                for T, O in zip(T_runs, O_runs)]
    return [precision for precision, _ in channels], [eta for _, eta in channels]


def _source_channel_run(
    mu:             torch.Tensor,           # (..., N, K)
    sigma:          torch.Tensor,           # (..., N, K) or (..., N, K, K)
    inverse:        torch.Tensor,           # (..., N, H, d, d) compact or (..., N, K, K) dense
    block_diagonal: bool,
    T:              torch.Tensor,           # (H, d, K) one run's sending maps
    O:              torch.Tensor,           # (H, K, d) one run's effective receiving maps
    receiver_law:   str,
    noise:          float,
    dtype:          torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""One run of :func:`_source_channel` (heads of a common size d)."""
    H, d, K = T.shape
    if block_diagonal:
        # Each full T reads all compact gauge blocks, (..., N, H, d, K).
        action = torch.einsum(
            "hsgu,...nguv->...nhsgv", T.reshape(H, d, H, d), inverse)
        action = action.reshape(*mu.shape[:-1], H, d, K)
    else:
        action = torch.einsum("hsk,...nkm->...nhsm", T, inverse)
    values = torch.einsum("...nhsk,...nk->...nhs", action, mu.to(dtype))  # (..., N, H, d)
    if sigma.ndim == mu.ndim:
        reduced = torch.einsum(
            "...nhsk,...nk,...nhtk->...nhst", action, sigma.to(dtype), action)
    else:
        reduced = action @ sigma.to(dtype).unsqueeze(-3) @ action.transpose(-1, -2)
    if receiver_law == "channel_likelihood":
        target = .5 * (reduced + reduced.transpose(-1, -2))
        target = target + noise * torch.eye(d, dtype=dtype, device=mu.device)
        precision = _spd_precision(target)                                 # (..., N, H, d, d)
        return precision, (precision @ values.unsqueeze(-1)).squeeze(-1)    # (..., N, H, d)
    target = torch.einsum("hks,...nhst,hmt->...nhkm", O, reduced, O)
    target = .5 * (target + target.transpose(-1, -2))
    target = target + noise * torch.eye(K, dtype=dtype, device=mu.device)
    precision = _spd_precision(target)                                     # (..., N, H, K, K)
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
    lambda_alpha_mode: str                = "constant",
    b0:             'float | torch.Tensor' = 1.0,
    c0:             'float | torch.Tensor' = 1.0,
) -> FullLinkContext:
    r"""Snapshot beta and source laws once for receiver filtering with noisy full L links.

    Routing uses the incoming Q/K comparison and frame. Channels use value_frame or frame.
    With alpha=value and weights w = lambda_beta beta_ij^h, ``ambient_alignment`` uses
    Q_i = receiver_noise U_i U_i.T and computes

    .. math::
        \Lambda_i=\alpha\Sigma_{p,i}^{-1}
            +\sum_{h,j}\lambda_h\beta_{ij}^{(h)}R_{ij}^{(h)-1},
        \eta_i=\alpha\Sigma_{p,i}^{-1}\mu_{p,i}
            +\sum_{h,j}\lambda_h\beta_{ij}^{(h)}R_{ij}^{(h)-1}m_{ij}^{(h)},
        c_i=\alpha+\sum_{h,j}\lambda_h\beta_{ij}^{(h)}.

    ``channel_likelihood`` pools d x d blocks W_ih = sum_j w R_hj^{-1}, lifts them through the tied
    receiving measurement P_h = O_h^T, and counts the entropy once:

    .. math::
        \Lambda_i=\alpha\Sigma_{p,i}^{-1}+U_i^{-\top}\Bigl(\sum_hO_hW_{ih}O_h^\top\Bigr)U_i^{-1},
        \qquad c_i=\alpha.

    Neither source nor router is detached. Reuse this context with independent
    receiver arguments to get inner partials while preserving the outer graph.
    relational_score retains its head-block frame contract (block_glk heads or so_n irrep blocks).

    A state-dependent ``lambda_alpha_mode`` ignores ``value``: the context carries a
    SelfCouplingProfile (coupling-only coefficients and the layer prior) and is returned frozen at
    alpha*(q) of the snapshot (mu, sigma), which is the episode's initial receiver state;
    full_link_context_at refreshes alpha at later states. ``state_dependent_per_coord`` requires the
    diagonal family with a diagonal prior, where KL(q_i||p_i) separates over local coordinates.
    """
    if lambda_alpha_mode not in FULL_LINK_ALPHA_MODES:
        raise ValueError(f"full-link inference supports lambda_alpha_mode in {FULL_LINK_ALPHA_MODES}, "
                         f"got {lambda_alpha_mode!r}")
    profiled = lambda_alpha_mode != "constant"
    if lambda_alpha_mode == "state_dependent_per_coord" and (
            sigma.ndim != mu.ndim or sigma_p.ndim != mu_p.ndim):
        raise ValueError("state_dependent_per_coord on the full-link receiver requires the diagonal family "
                         "and a diagonal prior: KL(q_i||p_i) separates over local coordinates only there")
    if operators.input_scope != "full" or operators.output_scope != "full" or operators.T is None:
        raise ValueError("full-link inference requires learned full input/output content maps")
    if operators.routing != "gaussian_kl":
        raise ValueError("full-link inference requires gaussian_kl routing")
    if not math.isfinite(receiver_noise) or receiver_noise <= 0:
        raise ValueError("receiver_noise must be finite and positive")
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be a finite nonnegative prior coefficient")
    channel = operators.receiver_law == "channel_likelihood"
    if channel and value <= 0 and not profiled:
        raise ValueError("channel_likelihood requires a positive prior coefficient (its only entropy term)")
    channel_frame = value_frame or frame
    dtype = _compute_dtype(mu, sigma, mu_p, sigma_p, frame.inverse,
                           channel_frame.inverse, *operators._content_parameters())
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work, sigma_work = mu.to(dtype), sigma.to(dtype)
        score = safe_kl_clamp(relational_score(
            mu_work, sigma_work, mu_work, sigma_work, frame, operators, capture), kl_max=kl_max)
        beta = attention_weights(score, tau=tau, log_prior=(
            None if log_prior is None else log_prior.to(dtype)))
        weights = _head_weights(beta, lambda_beta)                       # (..., H, I, J)
        content = operators.content_factor_runs()                  # per-run (T, O') once per context build
        source_precision, source_eta = _source_channel(
            mu_work, sigma_work, channel_frame, operators, receiver_noise, dtype, content)
        # Heads pool run by run (one run for equal heads); the pooled K x K precision and K-vector sum.
        pooled_precision = pooled_eta = None
        for run, (precision_r, eta_r, O) in enumerate(zip(source_precision, source_eta, content[1])):
            weights_r = operators._head_slice(weights, run, -3)                 # (..., n_r, I, J)
            if channel:
                pooled_W = torch.einsum("...hij,...jhst->...ihst", weights_r, precision_r)  # (..., I, n_r, d_r, d_r)
                pooled_w = torch.einsum("...hij,...jhs->...ihs", weights_r, eta_r)          # (..., I, n_r, d_r)
                O = O.to(dtype)                                                  # O' of content_link_chart
                precision_term = torch.einsum("hks,...ihst,hmt->...ikm", O, pooled_W, O)    # (..., I, K, K)
                eta_term = torch.einsum("hks,...ihs->...ik", O, pooled_w)                   # (..., I, K)
            else:
                precision_term = torch.einsum("...hij,...jhkm->...ikm", weights_r, precision_r)
                eta_term = torch.einsum("...hij,...jhk->...ik", weights_r, eta_r)
            pooled_precision = precision_term if pooled_precision is None else pooled_precision + precision_term
            pooled_eta = eta_term if pooled_eta is None else pooled_eta + eta_term
        Lambda, eta = _local_pullback(pooled_precision, pooled_eta, channel_frame)
        if sigma_p.ndim == mu_p.ndim:
            prior_precision = torch.diag_embed(sigma_p.to(dtype).reciprocal())
        else:
            prior_precision = torch.cholesky_inverse(torch.linalg.cholesky(sigma_p.to(dtype)))
        if profiled:
            mass = weights.sum(dim=(-3, -1))                             # (..., I)
            if sigma_p.ndim == mu_p.ndim:
                prior_logdet = sigma_p.to(dtype).log().sum(-1)
            else:
                prior_logdet = 2 * torch.linalg.cholesky(sigma_p.to(dtype)).diagonal(
                    dim1=-2, dim2=-1).log().sum(-1)
            profile = SelfCouplingProfile(
                lambda_alpha_mode, b0, c0, mu_p.to(dtype), sigma_p.to(dtype), prior_precision,
                (prior_precision @ mu_p.to(dtype).unsqueeze(-1)).squeeze(-1), prior_logdet,
                Lambda, eta, torch.zeros_like(mass) if channel else mass)
            context = full_link_context_at(
                FullLinkContext(Lambda, eta, mass, beta, score, profile), mu_work, sigma_work)
            if capture is not None:
                capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(),
                               full_link_context=context)
            return context
        Lambda = Lambda + value * prior_precision
        eta = eta + value * (prior_precision @ mu_p.to(dtype).unsqueeze(-1)).squeeze(-1)
        mass = weights.sum(dim=(-3, -1))                                 # (..., I)
        c = torch.full_like(mass, value) if channel else value + mass
        # Every causal row keeps j = i, so the mass is sum_h lambda_h; check it on the host.
        if value <= 0 and not channel and (
                not torch.is_tensor(lambda_beta) or lambda_beta.device.type == "cpu") and not bool(
                (torch.as_tensor(lambda_beta, dtype=torch.float64) > 0).any()):
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
    giving G_sigma,k = (Lambda_kk-c/sigma_k)/2 (c_k per coordinate under a per-coordinate alpha).
    The dense precision still enters the mean force; no head partition or diagonal target
    projection is applied. For a profiled context, pass full_link_context_at(context, mu, sigma):
    its partials are the exact profiled-energy gradient alpha*(q) dD/dq + dG/dq.
    """
    dtype = _compute_dtype(mu, sigma, context.Lambda)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        Lambda, eta, c = context.Lambda.to(dtype), context.eta.to(dtype), context.c.to(dtype)
        grad_mu = (Lambda @ mu.to(dtype).unsqueeze(-1)).squeeze(-1) - eta
        if sigma.ndim == mu.ndim:
            grad_sigma = .5 * (Lambda.diagonal(dim1=-2, dim2=-1) - _diagonal_weight(c, mu) / sigma.to(dtype))
        else:
            receiver_precision = torch.cholesky_inverse(torch.linalg.cholesky(sigma.to(dtype)))
            grad_sigma = .5 * (Lambda - _scalar_weight(c, mu)[..., None, None] * receiver_precision)
    # Never round the float32 force down to a half-precision incoming mean.
    return (grad_mu.to(torch.promote_types(mu.dtype, torch.float32)),
            grad_sigma.to(torch.promote_types(sigma.dtype, torch.float32)))


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
    full parameter-dependent training objective. A profiled context returns the profiled energy

    .. math::
        E=\sum_i\bigl[c_0\log(b_0+D_i)+G_i\bigr]

    (per coordinate: sum_ik c0_k log(b0_k + D_ik)), with G the coupling-only cost; the value does
    not depend on the alpha snapshot the context was frozen at.
    """
    if context.profile is not None:
        profile = context.profile
        coupling = full_link_cost(mu, sigma, FullLinkContext(
            profile.Lambda, profile.eta, profile.mass, context.beta, context.score))
        with torch.autocast(device_type=mu.device.type, enabled=False):
            D = self_coupling_divergence(mu, sigma, profile)
            self_term = _coeff(profile.c0, D) * torch.log(_coeff(profile.b0, D) + D)
        return coupling + self_term.sum()
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


def full_link_minimizer(
    mu:      torch.Tensor,                   # (..., N, K), current receiver means
    sigma:   torch.Tensor,                   # (..., N, K) or (..., N, K, K)
    context: FullLinkContext,
    *,
    damping:          float = 1.0,
    update_sigma:     bool  = True,
    eps:              float = 1e-6,
    sigma_max:        Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exact frozen-context receiver update in natural parameters (projected_gradient.tex, Prop. flow).

    .. math::
        J^+=(1-\rho)\Sigma^{-1}+\rho\Lambda/c,\qquad
        h^+=(1-\rho)\Sigma^{-1}\mu+\rho\eta/c,\qquad
        \mu^\star=\Lambda^{-1}\eta,\ \Sigma^\star=c\Lambda^{-1}.

    For gaussian_full this is the exact Fisher flow at rho = 1 - exp(-c t): SPD and non-increasing
    in the frozen energy for every rho in (0, 1], and the minimizer at rho = 1. A diagonal family
    uses the same interpolation with J = diag(1/s) and J* = diag(Lambda)/c, which reaches the
    restricted minimizer s* = c / Lambda_kk at rho = 1 (partial damping is a declared choice).
    With ``update_sigma=False`` the covariance is kept and the mean moves to
    (1 - rho) mu + rho mu*. Autograd through the solves gives the implicit outer derivative.

    A per-coordinate entropy weight c_k (diagonal family, state_dependent_per_coord) gives
    J*_k = Lambda_kk / c_k: the diagonal-family energy separates its variance terms per coordinate,
    so this remains the exact restricted minimizer. For a profiled context frozen at alpha*(q), the
    rho = 1 update is a majorize-minimize step on the profiled energy.
    """
    dtype = _compute_dtype(mu, sigma, context.Lambda)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        Lambda, eta, c = context.Lambda.to(dtype), context.eta.to(dtype), context.c.to(dtype)
        mean, cov = mu.to(dtype), sigma.to(dtype)
        factor = torch.linalg.cholesky(Lambda)
        mu_star = torch.cholesky_solve(eta.unsqueeze(-1), factor).squeeze(-1)
        diagonal = sigma.ndim == mu.ndim
        if not update_sigma:
            new_mu = mu_star if damping == 1.0 else (1 - damping) * mean + damping * mu_star
            return new_mu.to(torch.promote_types(mu.dtype, torch.float32)), sigma
        if diagonal:
            J_star = Lambda.diagonal(dim1=-2, dim2=-1) / _diagonal_weight(c, mu)
            if damping == 1.0:
                new_mu, new_sigma = mu_star, J_star.reciprocal()
            else:
                J = (1 - damping) / cov + damping * J_star
                new_sigma = J.reciprocal()
                new_mu = new_sigma * ((1 - damping) * mean / cov + damping * J_star * mu_star)
            new_sigma = new_sigma.clamp(min=eps, max=sigma_max) if sigma_max is not None else new_sigma.clamp(min=eps)
        else:
            c = _scalar_weight(c, mu)
            if damping == 1.0:
                new_mu = mu_star
                new_sigma = c[..., None, None] * torch.cholesky_inverse(factor)
            else:
                J0 = torch.cholesky_inverse(torch.linalg.cholesky(cov))
                J = (1 - damping) * J0 + damping * Lambda / c[..., None, None]
                J = .5 * (J + J.transpose(-1, -2))
                J_factor = torch.linalg.cholesky(J)
                h = (1 - damping) * (J0 @ mean.unsqueeze(-1)) + damping * (eta / c.unsqueeze(-1)).unsqueeze(-1)
                new_mu = torch.cholesky_solve(h, J_factor).squeeze(-1)
                new_sigma = torch.cholesky_inverse(J_factor)
            new_sigma = .5 * (new_sigma + new_sigma.transpose(-1, -2))
    return (new_mu.to(torch.promote_types(mu.dtype, torch.float32)),
            new_sigma.to(torch.promote_types(sigma.dtype, torch.float32)))


def full_link_cost_parts(
    mu:      torch.Tensor,
    sigma:   torch.Tensor,
    context: FullLinkContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Split the frozen receiver energy and report the distance to its mean minimizer.

    .. math::
        F_\mu=\sum_i\tfrac12(\mu_i-\mu_i^\star)^\top\Lambda_i(\mu_i-\mu_i^\star),\qquad
        F_\Sigma=\sum_i\Bigl[\tfrac12\operatorname{tr}(\Lambda_i\Sigma_i)-\tfrac{c_i}{2}\log\det\Sigma_i\Bigr].

    F_mu + F_Sigma equals full_link_cost up to a receiver-independent constant. Returns
    (F_mu, F_Sigma, sum_i ||mu_i - mu_i*||). For a profiled self-coupling, pass the context frozen at
    the evaluated state (full_link_context_at); the parts then split that frozen-alpha energy.
    """
    dtype = _compute_dtype(mu, sigma, context.Lambda)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mean, covariance = mu.to(dtype), sigma.to(dtype)
        Lambda, eta, c = context.Lambda.to(dtype), context.eta.to(dtype), context.c.to(dtype)
        mu_star = torch.linalg.solve(Lambda, eta.unsqueeze(-1)).squeeze(-1)
        delta = mean - mu_star
        mean_part = .5 * (delta * (Lambda @ delta.unsqueeze(-1)).squeeze(-1)).sum()
        if sigma.ndim == mu.ndim:
            trace = (Lambda.diagonal(dim1=-2, dim2=-1) * covariance).sum(-1)
            if c.ndim == mu.ndim:                                        # per-coordinate weights
                logdet, c = (c * covariance.log()).sum(-1), 1.0
            else:
                logdet = covariance.log().sum(-1)
        else:
            trace = (Lambda * covariance.transpose(-1, -2)).sum(dim=(-2, -1))
            logdet = torch.logdet(covariance)
        covariance_part = (.5 * trace - .5 * c * logdet).sum()
        return mean_part, covariance_part, delta.norm(dim=-1).sum()
