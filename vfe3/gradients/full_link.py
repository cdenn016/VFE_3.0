r"""Frozen, fully connected Gaussian receiver forces.

.. math::
    L_{ij}^{(h)}=U_iO_hT_hU_j^{-1}.

The receiver law is the channel likelihood (projected_gradient.tex, Eq. quadratic_family): it scores
the receiver's partial law P_h xi_i against the sender's d-dimensional message as an expected likelihood
and counts the receiver entropy once (c = alpha). The measurement P_h is the head-h rows of O'^{-1}
under ``receiver_measurement='inverse'``, or the adjoint O'_h^T.

The state-dependent self-coupling alpha*(D) = c0/(b0 + D), D = KL(q_i||p_i) (summed per token, or per
local coordinate on the diagonal family), is profiled out (projected_gradient.tex, Sec. profiled
self-coupling): the receiver energy becomes c0 log(b0 + D) + G(q), its exact force is alpha*(q) dD/dq
+ dG/dq, and freezing alpha at the current state gives a majorizer whose exact minimizer is the
existing closed form.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vfe3.alpha_i import alpha_gradient_coefficient
from vfe3.contracts import CanonicalFrameContext, host_check
from vfe3.free_energy import attention_weights
from vfe3.gradients.relational import clamped_routing_cost, relational_score
from vfe3.model.relational_operators import RelationalOperators, _chart_operands, _content_link_exp
from vfe3.relational_scoring import _compute_dtype
from vfe3.runtime import ieee_fp32_matmul

# Certificate of the inverted chart factor of the inverse measurement: eps_64 kappa_F(A) above the float32
# unit roundoff means the float64 inverse no longer certifies P O' = I to float32 accuracy.
_MEASUREMENT_CONDITION_TOL: float = torch.finfo(torch.float32).eps

# Receiving-law float64 retries of _spd_precision, per device. They are kept apart from the
# projected-covariance source-recovery counter, whose window the training loop reads as the
# G_q/G_k recovery numerator (audit 2026-10-08).
_RECEIVER_LAW_RETRIES: Dict[str, torch.Tensor] = {}


def _count_receiver_law_retry(failed: torch.Tensor) -> None:
    r"""Accumulate retried receiving-law entries per device (async add, no host sync)."""
    key = str(failed.device)
    counter = _RECEIVER_LAW_RETRIES.get(key)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=failed.device)
        _RECEIVER_LAW_RETRIES[key] = counter
    counter += failed.sum()


def receiver_law_retry_counter(device: torch.device) -> torch.Tensor:
    r"""On-device snapshot of the cumulative receiving-law retry count; differences count a window."""
    counter = _RECEIVER_LAW_RETRIES.get(str(device))
    if counter is None:
        return torch.zeros((), dtype=torch.int64, device=device)
    return counter.clone()


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
    c:      torch.Tensor       # (..., N) or (..., N, K), entropy multiplicity alpha
    beta:   torch.Tensor       # (..., H, N, N), fixed-value projected Q/K routing
    score:  torch.Tensor       # (..., H, N, N), routing cost, distinct from receiving energy
    profile: Optional["SelfCouplingProfile"] = None   # profiled alpha; None = fixed coefficients


@dataclass(frozen=True)
class SelfCouplingProfile:
    r"""Coupling-only receiver coefficients and the layer prior for a profiled self-coupling.

    .. math::
        E_i(q)=c_0\log(b_0+D_i(q))+G_i(q),\qquad D_i=\mathrm{KL}(q_i\|p_i),

    with G the receiver energy at alpha = 0 (coefficients Lambda_c, eta_c; the channel likelihood
    carries no entropy term of its own). ``per_coord`` replaces c0 log(b0+D) by
    sum_k c0_k log(b0_k + D_ik) over local coordinates of the diagonal family.
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

    @property
    def per_coord(self) -> bool:
        return self.mode == "state_dependent_per_coord"


FULL_LINK_ALPHA_MODES = ("state_dependent", "state_dependent_per_coord")


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
        \eta_i=\eta_{c,i}+\alpha_i\Sigma_{p,i}^{-1}\mu_{p,i},\quad c_i=\alpha_i.

    Per coordinate, alpha_i Sigma_p^{-1} becomes diag(alpha_ik / p_ik) and c_i the vector alpha_ik. The returned context's force alpha* dD/dq + dG/dq is the exact gradient of the
    profiled energy (envelope theorem), and its minimizer is the MM step. alpha keeps its graph: the
    force value is unchanged, and the unrolled outer derivative of the force map needs d alpha*/d theta.
    A context without a profile (fixed coefficients) is returned unchanged.
    """
    profile = context.profile
    if profile is None:
        return context
    with torch.autocast(device_type=mu.device.type, enabled=False):
        D = self_coupling_divergence(mu, sigma, profile)                 # (..., N) or (..., N, K)
        alpha = alpha_gradient_coefficient(
            D, b0=_coeff(profile.b0, D), c0=_coeff(profile.c0, D), mode=profile.mode)
        return _context_at_alpha(context, alpha)


def _context_at_alpha(
    context: FullLinkContext,
    alpha:   torch.Tensor,                   # (..., N), or (..., N, K) per coordinate
) -> FullLinkContext:
    r"""The profiled context frozen at a given self-coupling alpha (see :func:`full_link_context_at`)."""
    profile = context.profile
    if profile.per_coord:
        sigma_p = profile.sigma_p.to(alpha.dtype)
        Lambda = profile.Lambda + torch.diag_embed(alpha / sigma_p)
        eta = profile.eta + alpha * profile.mu_p.to(alpha.dtype) / sigma_p
    else:
        Lambda = profile.Lambda + alpha[..., None, None] * profile.prior_precision
        eta = profile.eta + alpha.unsqueeze(-1) * profile.prior_eta
    return FullLinkContext(Lambda, eta, alpha, context.beta, context.score, profile)   # c = alpha


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
    float64 (one host sync on the info tensor), counts the failed entries in the receiving-law
    retry counter (:func:`receiver_law_retry_counter`, not the projected-covariance recovery
    counter), and raises only if float64 also fails.
    """
    factor, info = torch.linalg.cholesky_ex(target)
    failed = info != 0
    if bool(failed.any()):                                                  # one host sync
        if target.dtype == torch.float64:
            raise torch.linalg.LinAlgError("receiving-law covariance is not positive definite")
        _count_receiver_law_retry(failed)
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
    r"""Return per-source, per-head precisions and natural means of the d-dimensional messages.

    .. math::
        R_{hj}=T_hC_jT_h^\top+\nu I_d,\qquad v_{hj}=T_h\xi_j,

    returning (..., N, H, d, d) and (..., N, H, d); nu is a declared channel floor.

    Both are returned as lists with one entry per run of equal-size heads (``operators.runs``),
    the run's heads in place of H and its size in place of d; one entry for equal heads.
    """
    T_runs, _ = content or operators.content_factor_runs()
    inverse = frame.inverse.to(dtype)
    channels = [_source_channel_run(mu, sigma, inverse, frame.block_diagonal, T.to(dtype), noise, dtype)
                for T in T_runs]
    return [precision for precision, _ in channels], [eta for _, eta in channels]


@ieee_fp32_matmul()
def _source_channel_run(
    mu:             torch.Tensor,           # (..., N, K)
    sigma:          torch.Tensor,           # (..., N, K) or (..., N, K, K)
    inverse:        torch.Tensor,           # (..., N, H, d, d) compact or (..., N, K, K) dense
    block_diagonal: bool,
    T:              torch.Tensor,           # (H, d, K) one run's sending maps
    noise:          float,
    dtype:          torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""One run of :func:`_source_channel` (heads of a common size d).

    Runs at IEEE fp32 whatever ``fp32_matmul_precision`` is: the products :math:`T_hU_j^{-1}` and
    :math:`T_hC_jT_h^\top` are :math:`\kappa(U)`-sensitive, as on the predictive route.
    """
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
    target = .5 * (reduced + reduced.transpose(-1, -2))
    target = target + noise * torch.eye(d, dtype=dtype, device=mu.device)
    precision = _spd_precision(target)                                     # (..., N, H, d, d)
    return precision, (precision @ values.unsqueeze(-1)).squeeze(-1)        # (..., N, H, d)


@ieee_fp32_matmul()
def _local_pullback(
    precision: torch.Tensor,                # (..., N, K, K), pooled reference precision
    eta:       torch.Tensor,                # (..., N, K), pooled reference natural mean
    frame:     CanonicalFrameContext,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Pull reference information into local coordinates.

    .. math::
        \Lambda_i=U_i^{-\top}P_iU_i^{-1},
        \eta_i=U_i^{-\top}b_i.

    Pinned to IEEE fp32: the sandwich amplifies TF32 rounding by about :math:`\kappa(U)^2`.
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


def _inverse_measurement_runs(
    T_runs: Sequence[torch.Tensor],         # per run (n_r, d_r, K) sending maps
    O_runs: Sequence[torch.Tensor],         # per run (n_r, K, d_r) stored receiving factors O_h
    chart:  str,                            # content_link_chart
    dtype:  torch.dtype,
) -> List[torch.Tensor]:                    # per run (n_r, d_r, K) measurements P_h
    r"""Dual receiving measurements: the head-h rows of the inverse of the column stack O' of the chart.

    .. math::
        O'=[O'_1,\dots,O'_H]\in\mathbb{R}^{K\times K},\qquad
        P=\begin{bmatrix}P_1\\\vdots\\P_H\end{bmatrix}=O'^{-1},\qquad P_hO'_g=\delta_{hg}I_{d_h}.

    P is formed in float64 from the chart's factors and cast to ``dtype``, never by inverting the float64 O'
    (audit 2026-10-10 round 2, item 14). The non-affine charts build O' from :math:`T^{-1}`, so inverting it again
    compounds the conditioning of T with that of O' and returns finite but wrong values at trained condition
    numbers (at K = 4 and cond(T) = 1e12, relative error 7e-6 in P against 6e-16 from the factors; the audit
    measured 1.0 in the K = 128 minimizer). From :func:`vfe3.model.relational_operators.chart_receive_runs`,

    .. math::
        O'_{\mathrm{exponential}} = e^{OT}T^{-1} \Rightarrow P = Te^{-OT},\qquad
        O'_{\mathrm{first\ order}} = T^{-1} + O = (I + OT)T^{-1} \Rightarrow P = T(I + OT)^{-1},\qquad
        O'_{\mathrm{affine}} = O \Rightarrow P = O^{-1}.

    The exponential form inverts nothing. The first-order and affine forms invert one matrix of stored parameters,
    :math:`A = I + OT` or :math:`A = O`, by ``inv_ex`` (no host sync), and certify it: a call is refused when
    ``inv_ex`` reports A singular, when P is not finite, or when :math:`\varepsilon_{64}\lVert A\rVert_F
    \lVert A^{-1}\rVert_F` exceeds the float32 unit roundoff, where the float64 inverse no longer certifies
    :math:`PO' = I` to float32 accuracy (:math:`\kappa_F(A) > 5.4\times10^8`). A singular or near-singular A does
    not by itself give non-finite values, so the certificate replaces the former finiteness assumption. The verdict
    only raises and goes through :func:`vfe3.contracts.host_check`, so inside
    :func:`vfe3.contracts.deferred_host_checks` it joins the next mandatory host read.
    """
    T_stack, O_stack = _chart_operands(T_runs, O_runs)                     # (K, K) float64, rows / columns by head
    K = T_stack.shape[-1]
    with torch.autocast(device_type=T_stack.device.type, enabled=False):
        if chart == "exponential":
            inverse = T_stack @ _content_link_exp(-(O_stack @ T_stack))     # (K, K) T e^{-OT}, rows by head
            flags = (~torch.isfinite(inverse).all()).reshape(1)
        else:
            eye = torch.eye(K, dtype=T_stack.dtype, device=T_stack.device)
            A = O_stack if chart == "affine" else eye + O_stack @ T_stack    # (K, K) O or I + OT
            A_inverse, info = torch.linalg.inv_ex(A)                         # (K, K)
            inverse = A_inverse if chart == "affine" else T_stack @ A_inverse
            with torch.no_grad():
                condition = A.norm() * A_inverse.norm()                      # kappa_F(A)
                flags = torch.stack((info != 0, ~torch.isfinite(inverse).all(),
                                     ~(torch.finfo(A.dtype).eps * condition <= _MEASUREMENT_CONDITION_TOL)))

    def _verdict(host: List[bool]) -> None:
        if any(host):
            raise ValueError(
                f"inverse receiver measurement refused ({chart} chart): the inverted chart factor is singular, "
                "non-finite or too ill-conditioned for its float64 inverse to certify P O' = I to float32 accuracy "
                f"(eps_64 kappa_F > {_MEASUREMENT_CONDITION_TOL:.3g}); use relational_receiver_measurement='adjoint'")

    host_check(flags, _verdict)
    inverse = inverse.to(dtype)                                            # (K, K), rows by head
    out, start = [], 0
    for O in O_runs:
        n, d = O.shape[0], O.shape[2]
        out.append(inverse[start:start + n * d].reshape(n, d, K))          # (n_r, d_r, K)
        start += n * d
    return out


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
    receiver_noise: float                 = 1.0,
    kl_max:         float                 = 100.0,
    log_prior:      Optional[torch.Tensor] = None,
    value_frame:    Optional[CanonicalFrameContext] = None,
    capture:        Optional[Dict[str, Any]] = None,
    lambda_alpha_mode: str                = "state_dependent",
    b0:             'float | torch.Tensor' = 1.0,
    c0:             'float | torch.Tensor' = 1.0,
) -> FullLinkContext:
    r"""Snapshot beta and source laws once for receiver filtering with noisy full L links.

    Routing uses the incoming Q/K comparison and frame. Channels use value_frame or frame.
    With weights w = lambda_beta beta_ij^h, the channel likelihood pools d x d blocks
    W_ih = sum_j w R_hj^{-1} and natural means w_ih = sum_j w R_hj^{-1} v_hj, lifts them through
    the receiving measurement P_h (d x K) of xi_i, and counts the entropy once:

    .. math::
        \Lambda_i=\alpha\Sigma_{p,i}^{-1}+U_i^{-\top}\Bigl(\sum_hP_h^\top W_{ih}P_h\Bigr)U_i^{-1},
        \qquad \eta_i=\alpha\Sigma_{p,i}^{-1}\mu_{p,i}+U_i^{-\top}\sum_hP_h^\top w_{ih},
        \qquad c_i=\alpha.

    ``operators.receiver_measurement`` selects P_h. ``"inverse"`` (the config default) is the dual
    measurement, the head-h rows of O'^{-1} for the square column stack O' (sum_h d_h = K), so
    P_h O'_g = delta_hg I. ``"adjoint"`` is P_h = O_h'^T, the adjoint of the effective receiving map
    O'_h of ``content_link_chart``. The two agree only for orthogonal O'. With equal channel precisions
    W_ih = w I, the inverse measurement's minimizer tends as alpha -> 0 to the kernel mean
    U_i sum_h O'_h sum_j beta_ij^h T_h xi_j, and it is invariant under (U, T, S, O') -> (U h, T h, S h,
    h^{-1} O'); the adjoint's minimizer is neither.

    Neither source nor router is detached. Reuse this context with independent
    receiver arguments to get inner partials while preserving the outer graph.
    relational_score retains its head-block frame contract (block_glk heads or so_n irrep blocks).

    The self-coupling alpha*(D) = c0/(b0 + D) is profiled: the context carries a
    SelfCouplingProfile (coupling-only coefficients and the layer prior) and is returned frozen at
    alpha*(q) of the snapshot (mu, sigma), which is the episode's initial receiver state;
    full_link_context_at refreshes alpha at later states. ``state_dependent_per_coord`` requires the
    diagonal family with a diagonal prior, where KL(q_i||p_i) separates over local coordinates.
    """
    if lambda_alpha_mode not in FULL_LINK_ALPHA_MODES:
        raise ValueError(f"full-link inference supports lambda_alpha_mode in {FULL_LINK_ALPHA_MODES}, "
                         f"got {lambda_alpha_mode!r}")
    if lambda_alpha_mode == "state_dependent_per_coord" and (
            sigma.ndim != mu.ndim or sigma_p.ndim != mu_p.ndim):
        raise ValueError("state_dependent_per_coord on the full-link receiver requires the diagonal family "
                         "and a diagonal prior: KL(q_i||p_i) separates over local coordinates only there")
    if operators.input_scope != "full" or operators.output_scope != "full" or operators.T is None:
        raise ValueError("full-link inference requires learned full input/output content maps")
    if getattr(operators, "content_link_edges", "off") != "off":
        raise ValueError("full-link inference reads the shared content maps; content_link_edges operators "
                         "(edge links G_ij = g_i e^L E_ij) are supported by the predictive_content readout only")
    if getattr(operators, "predictive_readout", "kernel") != "kernel":
        raise ValueError("full-link inference reads the content maps (T, O); predictive_readout='minimal_agent' "
                         "operators store no O (their receiving map is derived) and run only through "
                         "relational_predictive_readout")
    if operators.routing != "gaussian_kl":
        raise ValueError("full-link inference requires gaussian_kl routing")
    if not math.isfinite(receiver_noise) or receiver_noise <= 0:
        raise ValueError("receiver_noise must be finite and positive")
    channel_frame = value_frame or frame
    dtype = _compute_dtype(mu, sigma, mu_p, sigma_p, frame.inverse,
                           channel_frame.inverse, *operators._content_parameters())
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work, sigma_work = mu.to(dtype), sigma.to(dtype)
        routing_aux: Dict[str, torch.Tensor] = {}
        raw_score = relational_score(
            mu_work, sigma_work, mu_work, sigma_work, frame, operators, capture,
            routing_aux=routing_aux)                                             # (..., H, N, N) D_ij
        # Non-finite raw costs, which the clamp maps to kl_max (audit 2026-10-16 F10).
        score_nonfinite = None if capture is None else ~torch.isfinite(raw_score.detach())   # (..., H, N, N)
        score = clamped_routing_cost(raw_score, operators,
                                     kl_max=kl_max, log_prior=None if log_prior is None else log_prior.to(dtype),
                                     key_log_volume=routing_aux.get("key_log_volume"))
        beta = attention_weights(score, tau=operators.routing_temperature(tau, dtype), log_prior=(
            None if log_prior is None else log_prior.to(dtype)))
        weights = _head_weights(beta, lambda_beta)                       # (..., H, I, J)
        content = operators.content_factor_runs()                  # per-run (T, O') once per context build
        source_precision, source_eta = _source_channel(
            mu_work, sigma_work, channel_frame, operators, receiver_noise, dtype, content)
        # Dual measurement: per-run head rows of O'^{-1} from the chart factors, once per context build.
        measurement = (_inverse_measurement_runs(content[0], operators._stored_runs("O"),
                                                 operators.content_link_chart, dtype)
                       if getattr(operators, "receiver_measurement", "adjoint") == "inverse" else None)
        # Heads pool run by run (one run for equal heads); the pooled K x K precision and K-vector sum.
        pooled_precision = pooled_eta = None
        for run, (precision_r, eta_r, O) in enumerate(zip(source_precision, source_eta, content[1])):
            weights_r = operators._head_slice(weights, run, -3)                 # (..., n_r, I, J)
            pooled_W = torch.einsum("...hij,...jhst->...ihst", weights_r, precision_r)  # (..., I, n_r, d_r, d_r)
            pooled_w = torch.einsum("...hij,...jhs->...ihs", weights_r, eta_r)          # (..., I, n_r, d_r)
            if measurement is None:                                              # adjoint: P_h = O'_h^T
                O = O.to(dtype)                                                  # O' of content_link_chart
                precision_term = torch.einsum("hks,...ihst,hmt->...ikm", O, pooled_W, O)    # (..., I, K, K)
                eta_term = torch.einsum("hks,...ihs->...ik", O, pooled_w)                   # (..., I, K)
            else:                                                                # inverse: rows of O'^{-1}
                P = measurement[run]                                             # (n_r, d_r, K)
                precision_term = torch.einsum("hsk,...ihst,htm->...ikm", P, pooled_W, P)    # (..., I, K, K)
                eta_term = torch.einsum("hsk,...ihs->...ik", P, pooled_w)                   # (..., I, K)
            pooled_precision = precision_term if pooled_precision is None else pooled_precision + precision_term
            pooled_eta = eta_term if pooled_eta is None else pooled_eta + eta_term
        Lambda, eta = _local_pullback(pooled_precision, pooled_eta, channel_frame)
        if sigma_p.ndim == mu_p.ndim:
            prior_precision = torch.diag_embed(sigma_p.to(dtype).reciprocal())
        else:
            prior_precision = torch.cholesky_inverse(torch.linalg.cholesky(sigma_p.to(dtype)))
        if sigma_p.ndim == mu_p.ndim:
            prior_logdet = sigma_p.to(dtype).log().sum(-1)
        else:
            prior_logdet = 2 * torch.linalg.cholesky(sigma_p.to(dtype)).diagonal(
                dim1=-2, dim2=-1).log().sum(-1)
        profile = SelfCouplingProfile(
            lambda_alpha_mode, b0, c0, mu_p.to(dtype), sigma_p.to(dtype), prior_precision,
            (prior_precision @ mu_p.to(dtype).unsqueeze(-1)).squeeze(-1), prior_logdet, Lambda, eta)
        context = full_link_context_at(
            FullLinkContext(Lambda, eta, torch.zeros_like(prior_logdet), beta, score, profile), mu_work, sigma_work)
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(), score_nonfinite_pre=score_nonfinite,
                       full_link_context=context)
    return context


@ieee_fp32_matmul()                       # receiver force and solves at IEEE fp32
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


@ieee_fp32_matmul()                       # receiver force and solves at IEEE fp32
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
            profile.Lambda, profile.eta, torch.zeros_like(profile.prior_logdet), context.beta, context.score))
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


@ieee_fp32_matmul()                       # receiver force and solves at IEEE fp32
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
    restricted minimizer s* = c / Lambda_kk at rho = 1 (partial damping is a declared choice). For
    rho < 1 that diagonal step is not a descent step: its mean moves to mu* + W (mu - mu*) with the
    diagonal weight W_k = ((1 - rho)/s_k) / J_k, which need not contract the Lambda-norm when Lambda
    is dense and the variances are anisotropic, so the frozen energy can rise.
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


# Fixed iteration count of the certified one-sided Newton solve for the profiled self-coupling: no host
# sync and no early exit. Quadratic convergence needs a handful; a fold (g' -> 0) converges linearly.
_PROFILED_ROOT_ITERATIONS: int = 48
# Log-spaced trial lengths per iteration, from the MM step to the domain width, evaluated in one batch.
_PROFILED_TRIAL_LADDER: int = 16
# Floor on g'(alpha) = 1 - h'(alpha) in the implicit derivative: g' = 0 is a fold of the fixed-point set,
# where the root is not a differentiable function of the parameters.
_PROFILED_SLOPE_FLOOR: float = 1e-6


@dataclass(frozen=True)
class _DivergenceCurve:
    r"""D(a) = KL(q^\star(a) \| p) along the closed-form minimizers at prior weight a (float64, detached).

    With the prior precision P = LL^T, A = L^{-1}\Lambda_cL^{-\top} = V\operatorname{diag}(l)V^\top and
    r = V^\top L^{-1}(\eta_c - \Lambda_c\mu_p), the minimizer \mu^\star(a) = (\Lambda_c + aP)^{-1}(\eta_c + aP\mu_p) gives

    .. math::
        D_\mu(a)=\tfrac12\sum_k\frac{r_k^2}{(l_k+a)^2},\qquad
        D_\Sigma^{\mathrm{full}}(a)=\tfrac12\sum_k(u_k-1-\log u_k),\quad u_k=\frac{a}{l_k+a},

    from \Sigma^\star = a(\Lambda_c+aP)^{-1}. The diagonal family's variances
    s_k = \operatorname{clamp}(a/(\Lambda_{c,kk}+aP_{kk}), \epsilon, \sigma_{\max}) give
    D_\Sigma^{\mathrm{diag}} = \tfrac12[\sum_k(P_{kk}s_k-\log s_k)+\log\det\Sigma_p-K]; a kept covariance
    (``update_sigma=False``) gives a constant. Every term has slope -dD_k/da >= 0 that is nonincreasing in a
    (on the clamp-free interval of a diagonal coordinate), which :meth:`slope_floor` uses.
    """

    l:     torch.Tensor                      # (..., N, K) generalized eigenvalues of (Lambda_c, P), >= 0
    r2:    torch.Tensor                      # (..., N, K) squared rotated mean offsets
    kind:  str                               # 'full' | 'diagonal' | 'fixed'
    const: torch.Tensor                      # (..., N) variance constant (diagonal, fixed)
    Lc_diag: Optional[torch.Tensor]          # (..., N, K) Lambda_c,kk (diagonal)
    P_diag:  Optional[torch.Tensor]          # (..., N, K) P_kk (diagonal)
    eps:   float
    sigma_max: Optional[float]

    def evaluate(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Return D(a) (..., N), the mean slope (..., N), and the per-coordinate variance slopes and
        clamp-free mask (..., N, K)."""
        a_k = a.unsqueeze(-1)
        shifted = self.l + a_k                                           # (..., N, K)
        D = .5 * (self.r2 / shifted.square()).sum(-1)
        slope_mean = (self.r2 / shifted.pow(3)).sum(-1)
        weight = a_k                                                     # (..., N, 1), the entropy weight a
        if self.kind == "full":
            u = weight / shifted
            D = D + .5 * (u - 1 - u.log()).sum(-1)
            var_slope = self.l.square() / (2 * shifted.square() * weight)
            free = torch.ones_like(var_slope, dtype=torch.bool)
        elif self.kind == "diagonal":
            denominator = self.Lc_diag + a_k * self.P_diag
            s = weight / denominator
            s_c = s.clamp(min=self.eps, max=self.sigma_max)
            D = D + .5 * ((self.P_diag * s_c - s_c.log()).sum(-1) + self.const)
            free = (s > self.eps) if self.sigma_max is None else (s > self.eps) & (s < self.sigma_max)
            var_slope = torch.where(free, self.Lc_diag.square()
                                    / (2 * weight * denominator.square()), torch.zeros_like(s))
        else:
            D = D + self.const
            var_slope = torch.zeros_like(shifted)
            free = torch.ones_like(var_slope, dtype=torch.bool)
        return D, slope_mean, var_slope, free

    @staticmethod
    def slope_floor(left: Tuple[torch.Tensor, ...], right: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Lower bound of -D' on [x, y] from evaluations at x (``left``) and y (``right``).

        Each slope term decreases in a, so its value at y bounds it below; a diagonal variance term is
        kept only when clamp-free at both ends (its clamp-free set is an interval), else bounded by 0.
        """
        _, slope_mean, var_slope, free = right
        return slope_mean + torch.where(left[3] & free, var_slope, torch.zeros_like(var_slope)).sum(-1)


def _divergence_curve(
    profile:      SelfCouplingProfile,
    mu:           torch.Tensor,              # (..., N, K)
    sigma:        torch.Tensor,              # (..., N, K) or (..., N, K, K)
    update_sigma: bool,
    eps:          float,
    sigma_max:    Optional[float],
) -> _DivergenceCurve:
    r"""Diagonalize the frozen-alpha family once per receiver (float64, no graph); see :class:`_DivergenceCurve`."""
    f64 = torch.float64
    P = profile.prior_precision.to(f64)                                  # (..., N, K, K)
    Lc = profile.Lambda.to(f64)
    Lc = .5 * (Lc + Lc.transpose(-1, -2))
    L = torch.linalg.cholesky(P)
    half = torch.linalg.solve_triangular(L, Lc, upper=False)             # L^{-1} Lambda_c
    A = torch.linalg.solve_triangular(L, half.transpose(-1, -2), upper=False)   # L^{-1} Lambda_c L^{-T}
    l, V = torch.linalg.eigh(.5 * (A + A.transpose(-1, -2)))             # (..., N, K), (..., N, K, K)
    offset = profile.eta.to(f64) - (Lc @ profile.mu_p.to(f64).unsqueeze(-1)).squeeze(-1)
    r = (V.transpose(-1, -2) @ torch.linalg.solve_triangular(L, offset.unsqueeze(-1), upper=False)).squeeze(-1)
    const = torch.zeros_like(r[..., 0])                                  # (..., N)
    Lc_diag = P_diag = None
    if not update_sigma:
        kind = "fixed"
        mu_p = torch.broadcast_to(profile.mu_p.to(f64), mu.shape)
        const = self_coupling_divergence(mu_p, sigma.to(f64), profile)  # variance part only
    elif sigma.ndim == mu.ndim:
        kind = "diagonal"
        Lc_diag, P_diag = Lc.diagonal(dim1=-2, dim2=-1), P.diagonal(dim1=-2, dim2=-1)
        const = profile.prior_logdet.to(f64) - mu.shape[-1]
    else:
        kind = "full"
    return _DivergenceCurve(l.clamp(min=0), r.square(), kind, const, Lc_diag, P_diag, eps, sigma_max)


def _profiled_root(
    curve: _DivergenceCurve,
    a0:    torch.Tensor,                     # (..., N) alpha*(q_t), float64
    b0:    'float | torch.Tensor',
    c0:    'float | torch.Tensor',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fixed point of h(a) = c0/(b0 + D(a)) reached by the closed-form MM iteration a_{t+1} = h(a_t) from a0.

    h is nondecreasing (the minimizers of aD + E have D nonincreasing in a) and h <= c0/b0, so the MM
    sequence is monotone and converges to r, the nearest fixed point of g(a) = a - h(a) on the side
    sign(h(a0) - a0); g keeps the sign of g(a0) strictly between a0 and r. From the current point s the
    step is certified not to cross r: with \underline{h}' = c_0\,\underline{S}/(b_0+D_{\mathrm{left}})^2 a
    lower bound of h' on the trial interval (:meth:`_DivergenceCurve.slope_floor`, D is largest at its
    left end), g(s \pm t) keeps its sign for t < |g(s)|/(1 - \underline{h}'). Each iteration certifies the
    Newton step |g|/(1 - h'(s)) and a log-spaced ladder of trial lengths from the MM step |g| to the domain
    bound (c0/b0 above, half of s below) in one batch, and takes the longest certified step. Steps are never
    shorter than the MM step, the ladder crosses near-tangent stretches where h' ~ 1 (where MM and Newton
    both crawl), and the Newton trial converges quadratically at a simple root. Returns
    (r, |g(r)|, g'(r) = 1 - h'(r)).
    """
    tiny = torch.finfo(a0.dtype).tiny
    cap = c0 / b0
    D, *_ = curve.evaluate(a0)
    up = c0 / (b0 + D) > a0                                              # (..., N), MM direction
    ladder = torch.linspace(0, 1, _PROFILED_TRIAL_LADDER, dtype=a0.dtype, device=a0.device).reshape(
        -1, *(1,) * a0.ndim)                                             # (J, 1, ..., 1) log-spaced exponents
    s = a0
    for _ in range(_PROFILED_ROOT_ITERATIONS):
        at_s = curve.evaluate(s)
        D_s = at_s[0]
        gap = torch.where(up, c0 / (b0 + D_s) - s, s - c0 / (b0 + D_s)).clamp(min=0)   # |g(s)| toward r
        h_prime = c0 * (at_s[1] + at_s[2].sum(-1)) / (b0 + D_s).square()
        newton = torch.where(h_prime < 1, gap / (1 - h_prime).clamp(min=tiny), gap)
        width = torch.where(up, cap - s, .5 * s)
        # Trials: the Newton step and a log-spaced ladder from the MM step |g| to the domain width.
        log_gap = gap.clamp(min=tiny).log()
        spread = (width.clamp(min=tiny).log() - log_gap).clamp(min=0)
        trials = torch.cat((newton.unsqueeze(0), (log_gap + ladder * spread).exp()))     # (J + 1, ..., N)
        trials = torch.maximum(torch.minimum(trials, width), gap)
        y = torch.where(up, s + trials, s - trials)
        at_y = curve.evaluate(y)
        slope = torch.where(up, _DivergenceCurve.slope_floor(at_s, at_y), _DivergenceCurve.slope_floor(at_y, at_s))
        floor = c0 * slope / (b0 + torch.where(up, D_s, at_y[0])).square()   # h' >= floor on each trial interval
        certified = torch.where(floor < 1, torch.minimum(trials, gap / (1 - floor).clamp(min=tiny)), trials)
        step = torch.where(gap > 0, torch.maximum(certified.amax(0), gap), gap)     # g(s) = 0: stay
        s = torch.where(up, s + step, s - step)
    D, slope_mean, var_slope, _ = curve.evaluate(s)
    h = c0 / (b0 + D)
    return s, (h - s).abs(), 1 - c0 * (slope_mean + var_slope.sum(-1)) / (b0 + D).square()


@ieee_fp32_matmul()                       # receiver force and solves at IEEE fp32
def full_link_profiled_minimizer(
    mu:      torch.Tensor,                   # (..., N, K), current receiver means
    sigma:   torch.Tensor,                   # (..., N, K) or (..., N, K, K)
    context: FullLinkContext,
    *,
    damping:          float = 1.0,
    update_sigma:     bool  = True,
    eps:              float = 1e-6,
    sigma_max:        Optional[float] = None,
    capture:          Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Stationary point of the profiled receiver energy reached by the closed-form MM iteration.

    .. math::
        E(q)=c_0\log(b_0+D(q))+G(q),\qquad
        q^\star(a)=\arg\min_q\,aD(q)+G(q),\qquad
        r=h(r),\quad h(a)=\frac{c_0}{b_0+D(q^\star(a))}.

    Stationarity of E is \alpha^\star(q) = c0/(b0 + D(q)) with q = q*(alpha^\star), the scalar fixed point
    r = h(r) per receiver. Iterated ``closed_form`` steps are the fixed-point iteration a_{t+1} = h(a_t) from
    a_0 = alpha*(q_t); this returns their limit q*(r) in one solve (:func:`_profiled_root`). Along the curve
    a -> q*(a), dE/da = -g(a) dD/da with g = a - h, so E decreases monotonically from the one-step MM point
    q*(a_0) to q*(r): E(q*(r)) <= E(q*(a_0)) <= E(q_t). E need not be convex, and several fixed points can
    exist; r is the one the MM iteration from q_t converges to, not necessarily the global minimizer.

    The root is solved without graph in float64. Its derivative is attached by one implicit Newton step,
    alpha = r + (h(r; theta) - sg[h(r; theta)])/g'(r), which leaves the value r and has derivative
    d alpha/d theta = (dh/d theta)/(1 - h'), the implicit-function derivative of g(alpha; theta) = 0
    (g' floored at _PROFILED_SLOPE_FLOOR; at a fold of the fixed-point set no derivative exists).

    ``damping`` rho interpolates the natural parameters toward q*(r) exactly as :func:`full_link_minimizer`
    does toward its frozen target (it is that function, applied to the context frozen at alpha). At rho = 1
    the descent E(q*(r)) <= E(q_t) holds on both families. For rho < 1 no descent of E is guaranteed on
    either family: the frozen energy at r majorizes E with contact at q*(r), not at q_t.
    ``update_sigma=False`` keeps the covariance and solves the mean-only fixed point.
    ``state_dependent_per_coord`` is refused: its fixed points couple across coordinates through the
    dense mean precision.
    """
    profile = context.profile
    if profile.per_coord:
        raise ValueError("relational_receiver_update='closed_form_profiled' supports lambda_alpha_mode="
                         "'state_dependent' only: per coordinate, mu*(alpha) = Lambda(alpha)^{-1} eta(alpha) "
                         "couples the coordinate fixed points through the dense coupling precision")
    dtype = context.Lambda.dtype
    with torch.autocast(device_type=mu.device.type, enabled=False):
        with torch.no_grad():
            f64 = torch.float64
            mu64, sigma64 = mu.detach().to(f64), sigma.detach().to(f64)
            b0, c0 = _coeff(profile.b0, mu64), _coeff(profile.c0, mu64)
            curve = _divergence_curve(profile, mu64, sigma64, update_sigma, eps, sigma_max)
            a0 = c0 / (b0 + self_coupling_divergence(mu64, sigma64, profile))             # (..., N)
            root, residual, g_prime = _profiled_root(curve, a0, b0, c0)
        if capture is not None:
            # Diagnostic captures only (audit 2026-10-12 round 2, item 16 and M6): the largest root residual
            # |h(r) - r| and the receivers whose slope 1 - h'(r) is floored at _PROFILED_SLOPE_FLOOR, where the
            # implicit derivative 1/(1 - h') is capped (near a fold of the fixed-point set). Device tensors; the
            # diagnostics read them.
            capture["relational_receiver_profiled_residual"] = residual.max().to(dtype)
            capture["relational_receiver_profiled_floored_slopes"] = (g_prime < _PROFILED_SLOPE_FLOOR).sum()
        alpha = root.to(dtype)
        if torch.is_grad_enabled():
            at_root = full_link_minimizer(mu, sigma, _context_at_alpha(context, alpha), damping=1.0,
                                          update_sigma=update_sigma, eps=eps, sigma_max=sigma_max)
            D = self_coupling_divergence(*at_root, profile)
            h = _coeff(profile.c0, D) / (_coeff(profile.b0, D) + D)
            alpha = alpha + (h - h.detach()) / g_prime.clamp(min=_PROFILED_SLOPE_FLOOR).to(dtype)
        return full_link_minimizer(mu, sigma, _context_at_alpha(context, alpha), damping=damping,
                                   update_sigma=update_sigma, eps=eps, sigma_max=sigma_max)


@ieee_fp32_matmul()                       # receiver force and solves at IEEE fp32
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
