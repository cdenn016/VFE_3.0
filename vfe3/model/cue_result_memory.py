r"""Cue/result generative memory: exact conditioning of a switching linear-Gaussian component model.

Implementation plan: ``docs/implementation/generative-memory-plan-2026-10-24.md``, Section 3.2. The construction
is the joint cue/result model of ``docs/research/2026-10-20-neural-free-continuation/prototypes.md``. Component
:math:`a` (one of :math:`M`) carries a cue law, a linear-Gaussian result law and a prior weight; the incoming
working belief enters as a Gaussian record of the cue,

.. math::
    A\sim\operatorname{Cat}(\hat\pi),\qquad X\mid A=a\sim\mathcal N(\nu_a, L_a),\qquad
    Y\mid X, A=a\sim\mathcal N\bigl(\omega_a + F_a(X - \nu_a), Q_a\bigr),\qquad O\mid X\sim\mathcal N(X, C),

with :math:`L_a = \operatorname{diag}e^{\log\lambda_a}`, :math:`Q_a = \operatorname{diag}e^{\log q_a}`,
:math:`\hat\pi = \operatorname{softmax}(\log\pi)` and :math:`C` the declared record covariance (side information,
not a generated variable). Every factor is normalized, and the Schur complement of the component's joint
:math:`(X, Y)` covariance is :math:`Q_a \succ 0` by construction. Conditioning on :math:`O = \xi` with
:math:`W_a = L_a + C` gives

.. math::
    \log w_a = \log\hat\pi_a + \log\mathcal N(\xi;\nu_a, W_a) - \log Z,\qquad
    m_{X,a} = \nu_a + L_aW_a^{-1}(\xi - \nu_a),\qquad P_{X,a} = CW_a^{-1}L_a = (C^{-1} + L_a^{-1})^{-1},

.. math::
    m_{Y,a} = \omega_a + F_a(m_{X,a} - \nu_a),\qquad P_{Y,a} = Q_a + F_aP_{X,a}F_a^\top,

and the result marginal is the mixture :math:`\sum_aw_a\mathcal N(m_{Y,a}, P_{Y,a})`, whose first two moments are
returned. Read as dynamics, a component is a regime of a switching linear-Gaussian transition from cue to result,
selected by Bayes' rule from the cue record (Ghahramani and Hinton 2000; Xu, Jordan and Hinton 1995 for the
generative gate), and the moment match is one assumed-density-filtering collapse (Minka 2001).

Regression structure. ``regression_offset`` (M, K, K) gives a dense :math:`F_a = I + D_a`; ``regression_left`` and
``regression_right`` (M, K, r) give :math:`F_a = I + U_aV_a^\top`. Exactly one representation is passed. Both
start at :math:`F_a = I` and both put the identity outside the learned parameters, so weight decay pulls
:math:`F_a` toward the identity.

Numerics. :math:`W_a` is factored once per (token, component) by Cholesky; the evidence, the gain
:math:`W_a^{-1}(\xi - \nu_a)` and :math:`W_a^{-1}L_a` are triangular solves against that factor. The covariances are
positive sums (:math:`CW_a^{-1}L_a` symmetrized, plus :math:`Q_a` and the between-component spread), never a
difference of two large matrices. Components are processed in chunks bounded by ``tile_bytes``, and chunk moments are
merged by the law of total covariance with log-sum-exp weights, so no (tokens, M, K, K) tensor is retained beyond one
chunk. A factor that fails reads NaN. Autocast is disabled, GEMMs run in IEEE fp32, and the compute dtype is float64
when an input is float64 and float32 otherwise.

Diagonal record. :func:`cue_result_diagonal_posterior` is the same exact posterior for a diagonal record covariance
:math:`C = \operatorname{diag}c` (the reference variances of ``gaussian_frame_diagonal``). Then :math:`W_a` is diagonal,
:math:`m_{X,a} - \nu_a = \lambda_a/(\lambda_a + c)\odot(\xi - \nu_a)` and
:math:`P_{X,a} = \operatorname{diag}(\lambda_ac/(\lambda_a + c))` elementwise, and with :math:`F_a = I + U_aV_a^\top`

.. math::
    \operatorname{diag}(F_aP_{X,a}F_a^\top)_k = p_k + 2p_k\sum_rU_{kr}V_{kr} + \sum_{r,s}U_{kr}U_{ks}(V_a^\top P_{X,a}V_a)_{rs},

so a (token, component) pair costs :math:`O(Kr^2)`, or :math:`O(K^2)` for a dense :math:`F_a`, where
:math:`\operatorname{diag}(FPF^\top)_k = \sum_lF_{kl}^2p_l`. It returns the result mean and the exact diagonal of the
result mixture's covariance, :math:`\sum_aw_a[\operatorname{diag}P_{Y,a} + (m_{Y,a} - \bar m)^{\odot2}]`: the M-projection of
the exact result law onto Gaussians with diagonal covariance in the reference chart. The off-diagonal covariance, within
a component (from :math:`F_a`) and between components, is discarded by that projection, not approximated. Rows and
components are tiled by ``tile_bytes``; with gradients on and more than one tile, each tile runs under a non-reentrant
checkpoint and is recomputed in backward.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from vfe3.runtime import ieee_fp32_matmul

CUE_RESULT_TILE_BYTES = 32 * 1024 ** 2          # bound on one chunk's (tokens, m, K, K) tensors, not total memory


class CueResultPosterior(NamedTuple):
    """Moments of the exact result marginal, the component log weights and the record log evidence."""

    mean:         torch.Tensor                  # (..., K) result mean sum_a w_a m_{Y,a}
    covariance:   torch.Tensor                  # (..., K, K), or (..., K) variances on a diagonal record; with the spread
    log_weights:  torch.Tensor                  # (..., M) log w_a
    log_evidence: torch.Tensor                  # (...) log p(xi | C)


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _shifted_mass(
    log_w: torch.Tensor,                        # (r, m) unnormalized log weights
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Row maximum :math:`s` (detached; 0 where a row has no finite entry), scaled mass
    :math:`l = \sum_a e^{\ell_a - s}` and normalized weights :math:`w_a = e^{\ell_a - s}/l`.

    The absolute log mass :math:`s + \log l` is never subtracted from :math:`\ell_a`: at large log densities its
    :math:`\log l` increment rounds away (float32 spacing is 2 near :math:`-2.5\cdot10^7`), which made two identical
    components carry weight one each (audit 2026-10-25 F01). The shift cancels exactly in :math:`w`, so detaching it
    leaves every gradient unchanged."""
    s = log_w.detach().amax(dim=-1)
    s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    e = torch.exp(log_w - s.unsqueeze(-1))
    l = e.sum(-1)
    return s, l, e / l.unsqueeze(-1)


def _merge_scales(
    s_a: torch.Tensor, l_a: torch.Tensor,
    s_b: torch.Tensor, l_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Merge two (shift, scaled mass) pairs: the common shift, the merged scaled mass and the two mixing fractions
    :math:`\alpha = l_ae^{s_a - s}/l`, :math:`\beta = l_be^{s_b - s}/l`, all relative to the common shift."""
    s = torch.maximum(s_a, s_b)
    a = l_a * torch.exp(s_a - s)
    b = l_b * torch.exp(s_b - s)
    l = a + b
    return s, l, (a / l).unsqueeze(-1), (b / l).unsqueeze(-1)


def _normalized_outputs(
    log_w: torch.Tensor,                        # (R, M) unnormalized log weights
    s: torch.Tensor,                            # (R,) shift
    l: torch.Tensor,                            # (R,) scaled mass relative to s
) -> Tuple[torch.Tensor, torch.Tensor]:         # normalized log weights (R, M), log evidence (R,)
    r"""Normalized log weights :math:`(\ell_a - s) - \log l` and the log evidence :math:`s + \log l`, formed only as an
    output (F01)."""
    log_l = torch.log(l)
    return (log_w - s.unsqueeze(-1)) - log_l.unsqueeze(-1), s + log_l


def _check_inputs(
    xi: torch.Tensor,
    C: torch.Tensor,
    nu: torch.Tensor,
    omega: torch.Tensor,
    log_lambda: torch.Tensor,
    log_result_noise: torch.Tensor,
    log_pi: torch.Tensor,
    regression_offset: Optional[torch.Tensor],
    regression_left: Optional[torch.Tensor],
    regression_right: Optional[torch.Tensor],
    diagonal: bool = False,
) -> None:
    K = xi.shape[-1]
    if diagonal:
        if C.shape != xi.shape:
            raise ValueError(f"the diagonal cue/result memory needs record variances (..., K) matching xi "
                             f"{tuple(xi.shape)}, got {tuple(C.shape)}")
    elif C.dim() != xi.dim() + 1 or C.shape[-2:] != (K, K) or C.shape[:-2] != xi.shape[:-1]:
        raise ValueError(f"cue/result memory needs a dense record covariance (..., K, K) matching xi {tuple(xi.shape)}, "
                         f"got {tuple(C.shape)}")
    M = nu.shape[0]
    for name, t in (("nu", nu), ("omega", omega), ("log_lambda", log_lambda), ("log_result_noise", log_result_noise)):
        if t.shape != (M, K):
            raise ValueError(f"{name} must have shape ({M}, {K}), got {tuple(t.shape)}")
    if log_pi.shape != (M,) and log_pi.shape != (*xi.shape[:-1], M):
        raise ValueError(f"log_pi must have shape ({M},) or, per token, {(*xi.shape[:-1], M)}; got {tuple(log_pi.shape)}")
    dense = regression_offset is not None
    low_rank = regression_left is not None or regression_right is not None
    if dense == low_rank:
        raise ValueError("pass exactly one regression representation: regression_offset, or regression_left and "
                         "regression_right")
    if dense and regression_offset.shape != (M, K, K):
        raise ValueError(f"regression_offset must have shape ({M}, {K}, {K}), got {tuple(regression_offset.shape)}")
    if low_rank:
        if regression_left is None or regression_right is None:
            raise ValueError("the low-rank regression needs both regression_left and regression_right")
        if (regression_left.dim() != 3 or regression_left.shape[:2] != (M, K)
                or regression_right.shape != regression_left.shape):
            raise ValueError(f"regression_left and regression_right must both have shape ({M}, {K}, r), got "
                             f"{tuple(regression_left.shape)} and {tuple(regression_right.shape)}")


def _apply_regression(
    v: torch.Tensor,                            # (..., m, K)
    offset: Optional[torch.Tensor],             # (m, K, K) D_a, F_a = I + D_a
    left: Optional[torch.Tensor],               # (m, K, r) U_a
    right: Optional[torch.Tensor],              # (m, K, r) V_a, F_a = I + U_a V_a^T
) -> torch.Tensor:                              # (..., m, K) F_a v_a
    if offset is not None:
        return v + torch.einsum("akl,...al->...ak", offset, v)
    return v + torch.einsum("akr,...ar->...ak", left, torch.einsum("akr,...ak->...ar", right, v))


def _sandwich_regression(
    P: torch.Tensor,                            # (..., m, K, K) symmetric
    offset: Optional[torch.Tensor],
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
) -> torch.Tensor:                              # (..., m, K, K) F_a P_a F_a^T
    if offset is not None:
        FP = P + torch.einsum("akl,...alj->...akj", offset, P)                          # (I + D) P
        return FP + torch.einsum("...akl,ajl->...akj", FP, offset)                      # FP (I + D)^T
    VtP = torch.einsum("akr,...akj->...arj", right, P)                                   # V^T P   (..., m, r, K)
    FP = P + torch.einsum("akr,...arj->...akj", left, VtP)                               # (I + U V^T) P
    FPV = torch.einsum("...akl,alr->...akr", FP, right)                                  # FP V    (..., m, K, r)
    return FP + torch.einsum("...akr,ajr->...akj", FPV, left)                            # FP (I + V U^T)


@ieee_fp32_matmul()
def _chunk_moments(
    xi: torch.Tensor,                           # (T, K) flattened tokens
    C: torch.Tensor,                            # (T, K, K)
    nu: torch.Tensor,                           # (m, K)
    omega: torch.Tensor,                        # (m, K)
    lam: torch.Tensor,                          # (m, K) cue variances
    q: torch.Tensor,                            # (m, K) result noise variances
    log_pi_hat: torch.Tensor,                   # (m,) or (T, m) normalized log prior
    offset: Optional[torch.Tensor],
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""One (tokens x components) tile: log weights (T, m), the tile's shift and scaled mass (T,) (:func:`_shifted_mass`),
    and its normalized result mean and covariance."""
    K = xi.shape[-1]
    W = C.unsqueeze(-3) + torch.diag_embed(lam)                                          # (T, m, K, K) L_a + C
    L_W, info = torch.linalg.cholesky_ex(W)
    failed = info != 0                                                                   # (T, m)
    delta = (xi.unsqueeze(-2) - nu).unsqueeze(-1)                                        # (T, m, K, 1) xi - nu_a
    z = torch.linalg.solve_triangular(L_W, delta, upper=False)                           # (T, m, K, 1)
    maha = z.square().sum(dim=(-2, -1))                                                  # (T, m)
    logdet = 2.0 * torch.diagonal(L_W, dim1=-2, dim2=-1).log().sum(-1)                   # (T, m)
    log_w = log_pi_hat - 0.5 * (maha + logdet + K * math.log(2.0 * math.pi))             # (T, m) unnormalized
    log_w = log_w.masked_fill(failed, math.nan)
    gain = torch.cholesky_solve(delta, L_W).squeeze(-1)                                  # (T, m, K) W^-1 (xi - nu)
    WinvL = torch.cholesky_solve(torch.diag_embed(lam).expand_as(W), L_W)                # (T, m, K, K) W^-1 L_a
    P_X = C.unsqueeze(-3) @ WinvL                                                        # C W^-1 L_a
    P_X = 0.5 * (P_X + P_X.transpose(-2, -1))
    m_Y = omega + _apply_regression(lam * gain, offset, left, right)                     # (T, m, K) omega + F L W^-1 d
    P_Y = torch.diag_embed(q) + _sandwich_regression(P_X, offset, left, right)           # (T, m, K, K)
    s, l, w = _shifted_mass(log_w)                                                       # (T,), (T,), (T, m)
    mean = (w.unsqueeze(-1) * m_Y).sum(-2)                                               # (T, K)
    centered = m_Y - mean.unsqueeze(-2)                                                  # (T, m, K)
    cov = (w[..., None, None] * (P_Y + centered.unsqueeze(-1) * centered.unsqueeze(-2))).sum(-3)
    return log_w, s, l, mean, cov


def _merge(
    s_a: torch.Tensor, l_a: torch.Tensor, mean_a: torch.Tensor, cov_a: torch.Tensor,
    s_b: torch.Tensor, l_b: torch.Tensor, mean_b: torch.Tensor, cov_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Law of total covariance for two normalized mixtures with (shift, scaled mass) pairs: a positive sum."""
    s, l, alpha, beta = _merge_scales(s_a, l_a, s_b, l_b)                                # alpha, beta (T, 1)
    diff = mean_a - mean_b                                                               # (T, K)
    mean = alpha * mean_a + beta * mean_b
    cov = (alpha.unsqueeze(-1) * cov_a + beta.unsqueeze(-1) * cov_b
           + (alpha * beta).unsqueeze(-1) * diff.unsqueeze(-1) * diff.unsqueeze(-2))
    return s, l, mean, cov


@ieee_fp32_matmul()
def cue_result_exact_posterior(
    xi:                torch.Tensor,            # (..., K) record value (the working mean in the declared chart)
    C:                 torch.Tensor,            # (..., K, K) record covariance
    nu:                torch.Tensor,            # (M, K) cue means
    omega:             torch.Tensor,            # (M, K) result means
    log_lambda:        torch.Tensor,            # (M, K) cue log variances
    log_result_noise:  torch.Tensor,            # (M, K) result-noise log variances
    log_pi:            torch.Tensor,            # (M,) or (..., M) per token: unnormalized log prior weights
    *,
    regression_offset: Optional[torch.Tensor] = None,   # (M, K, K) D_a, F_a = I + D_a
    regression_left:   Optional[torch.Tensor] = None,   # (M, K, r) U_a
    regression_right:  Optional[torch.Tensor] = None,   # (M, K, r) V_a, F_a = I + U_a V_a^T
    tile_bytes:        int = CUE_RESULT_TILE_BYTES,
) -> CueResultPosterior:
    r"""The moments of :math:`p(Y\mid O=\xi, C)` for the model in the module docstring, with the component log
    weights and :math:`\log p(\xi\mid C) = \log\sum_a\hat\pi_a\mathcal N(\xi;\nu_a, L_a + C)`. A ``log_pi`` of
    shape (..., M) gives every token its own prior :math:`\hat\pi_i = \operatorname{softmax}(\log\pi_i)`."""
    _check_inputs(xi, C, nu, omega, log_lambda, log_result_noise, log_pi,
                  regression_offset, regression_left, regression_right)
    batch, K, M = xi.shape[:-1], xi.shape[-1], nu.shape[0]
    reg = (regression_offset, regression_left, regression_right)
    dtype = _compute_dtype(xi, C, nu, omega, log_lambda, log_result_noise, log_pi,
                           *(t for t in reg if t is not None))
    device_type = xi.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        cast = lambda t: None if t is None else t.to(dtype)                              # noqa: E731
        xi_f = xi.to(dtype).reshape(-1, K)                                               # (T, K)
        C_f = C.to(dtype).reshape(-1, K, K)                                              # (T, K, K)
        nu_c, omega_c = cast(nu), cast(omega)
        lam, q = log_lambda.to(dtype).exp(), log_result_noise.to(dtype).exp()
        log_pi_hat = torch.log_softmax(log_pi.to(dtype), dim=-1)
        if log_pi_hat.dim() > 1:
            log_pi_hat = log_pi_hat.reshape(-1, M)                                       # (T, M) per token
        offset, left, right = (cast(t) for t in reg)
        T = xi_f.shape[0]
        per_token = log_pi_hat.dim() > 1
        # About six (K, K) tensors live per (token, component) pair. Tokens and components are both tiled, so the
        # budget bounds the transient whenever one pair fits; with gradients on and more than one tile each tile is
        # checkpointed, so backward recomputes its factors instead of retaining them (audit 2026-10-25 F15).
        per_pair = 6 * K * K * (torch.finfo(dtype).bits // 8)
        pairs = max(1, int(tile_bytes) // per_pair)
        m_step = min(M, pairs)
        r_step = max(1, min(max(1, T), pairs // m_step))
        tiled = T > r_step or M > m_step
        grad = torch.is_grad_enabled()
        log_w_rows, s_rows, l_rows, mean_rows, cov_rows = [], [], [], [], []
        for i in range(0, T, r_step):
            running = None
            chunks = []
            for j in range(0, M, m_step):
                sl = slice(j, j + m_step)
                prior = log_pi_hat[i:i + r_step, sl] if per_token else log_pi_hat[sl]
                args = (xi_f[i:i + r_step], C_f[i:i + r_step], nu_c[sl], omega_c[sl], lam[sl], q[sl], prior,
                        None if offset is None else offset[sl], None if left is None else left[sl],
                        None if right is None else right[sl])
                if tiled and grad and any(t is not None and t.requires_grad for t in args):
                    log_w, s, l, mean, cov = checkpoint(_chunk_moments, *args, use_reentrant=False,
                                                        preserve_rng_state=False)
                else:
                    log_w, s, l, mean, cov = _chunk_moments(*args)
                chunks.append(log_w)
                running = (s, l, mean, cov) if running is None else _merge(*running, s, l, mean, cov)
            s_rows.append(running[0])
            l_rows.append(running[1])
            mean_rows.append(running[2])
            cov_rows.append(running[3])
            log_w_rows.append(torch.cat(chunks, dim=-1))
        mean, cov = torch.cat(mean_rows), torch.cat(cov_rows)                            # (T, K), (T, K, K)
        log_weights, log_evidence = _normalized_outputs(torch.cat(log_w_rows), torch.cat(s_rows), torch.cat(l_rows))
        cov = 0.5 * (cov + cov.transpose(-2, -1))
    return CueResultPosterior(mean.reshape(*batch, K), cov.reshape(*batch, K, K),
                              log_weights.reshape(*batch, M), log_evidence.reshape(batch))


@ieee_fp32_matmul()
def _diagonal_tile(
    xi:         torch.Tensor,                   # (r, K)
    c:          torch.Tensor,                   # (r, K) record variances
    nu:         torch.Tensor,                   # (m, K)
    omega:      torch.Tensor,                   # (m, K)
    lam:        torch.Tensor,                   # (m, K) cue variances
    q:          torch.Tensor,                   # (m, K) result noise variances
    log_pi_hat: torch.Tensor,                   # (m,) or (r, m) normalized log prior
    offset:     Optional[torch.Tensor],         # (m, K, K) D_a
    left:       Optional[torch.Tensor],         # (m, K, rank) U_a
    right:      Optional[torch.Tensor],         # (m, K, rank) V_a
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""One (rows x components) tile, elementwise: log weights (r, m), the tile's shift and scaled mass (r,)
    (:func:`_shifted_mass`), and its normalized result mean and variances (r, K).

    Low-rank result variances. With :math:`a_k = \sum_rU_{kr}V_{kr}` and :math:`G = V^\top PV`,
    :math:`\operatorname{diag}(FPF^\top)_k = p_k(1 + a_k)^2 + \sum_{l\ne k}p_l(\sum_rU_{kr}V_{lr})^2`, and the second
    sum is formed as :math:`[UGU^\top]_{kk} - p_ka_k^2`, a sum of squares clamped to its sign, so the regression
    variance is non-negative and the result noise :math:`q_k` is added last: the variance is never below :math:`q_k`.
    The expanded form :math:`q + p + 2pa + [UGU^\top]_{kk}` cancelled a positive :math:`q` against :math:`p` where
    :math:`F` cancels a coordinate (audit 2026-10-25 F02). The clamp acts only on a rounding residue; where the
    remainder is positive the gradients are those of the expanded form. The absolute rounding error of the regression
    variance stays of order :math:`\epsilon(p_ka_k^2 + [UGU^\top]_{kk})`."""
    K = xi.shape[-1]
    W = c.unsqueeze(-2) + lam                                                            # (r, m, K) diag(L_a + C)
    delta = xi.unsqueeze(-2) - nu                                                        # (r, m, K) xi - nu_a
    log_w = log_pi_hat - 0.5 * ((delta.square() / W).sum(-1) + W.log().sum(-1) + K * math.log(2.0 * math.pi))
    gain = lam / W                                                                       # (r, m, K) L W^-1
    g = gain * delta                                                                     # (r, m, K) m_X - nu
    p = gain * c.unsqueeze(-2)                                                           # (r, m, K) diag P_X = L C / W
    if offset is not None:
        F = offset + torch.eye(K, dtype=offset.dtype, device=offset.device)              # (m, K, K) I + D_a
        m_Y = omega + torch.einsum("akl,ial->iak", F, g)
        v_Y = torch.einsum("akl,ial->iak", F.square(), p) + q                            # sum_l F_kl^2 p_l, then Q
    else:
        t = torch.einsum("iak,akr->iar", g, right)                                       # (r, m, rank) V^T g
        m_Y = omega + g + torch.einsum("iar,akr->iak", t, left)
        a = (left * right).sum(-1)                                                       # (m, K) a_k = sum_r U_kr V_kr
        G = torch.einsum("akr,iak,aks->iars", right, p, right)                           # (r, m, rank, rank) V^T P V
        quad = torch.einsum("akr,iars,aks->iak", left, G, left)                          # diag U G U^T
        off = torch.clamp_min(quad - p * a.square(), 0.0)                                # sum_{l != k} p_l (U_k V_l)^2
        v_Y = (p * (1.0 + a).square() + off) + q                                         # diag F P F^T, then Q
    s, l, w = _shifted_mass(log_w)                                                       # (r,), (r,), (r, m)
    mean = (w.unsqueeze(-1) * m_Y).sum(-2)                                               # (r, K)
    var = (w.unsqueeze(-1) * (v_Y + (m_Y - mean.unsqueeze(-2)).square())).sum(-2)       # (r, K)
    return log_w, s, l, mean, var


def _merge_diagonal(
    s_a: torch.Tensor, l_a: torch.Tensor, mean_a: torch.Tensor, var_a: torch.Tensor,
    s_b: torch.Tensor, l_b: torch.Tensor, mean_b: torch.Tensor, var_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Law of total variance for two normalized mixtures with (shift, scaled mass) pairs, coordinatewise: a positive
    sum."""
    s, l, alpha, beta = _merge_scales(s_a, l_a, s_b, l_b)                                # alpha, beta (r, 1)
    return (s, l, alpha * mean_a + beta * mean_b,
            alpha * var_a + beta * var_b + alpha * beta * (mean_a - mean_b).square())


@ieee_fp32_matmul()
def cue_result_diagonal_posterior(
    xi:                torch.Tensor,            # (..., K) record value (reference working mean)
    c:                 torch.Tensor,            # (..., K) record variances (reference, diagonal)
    nu:                torch.Tensor,            # (M, K) cue means
    omega:             torch.Tensor,            # (M, K) result means
    log_lambda:        torch.Tensor,            # (M, K) cue log variances
    log_result_noise:  torch.Tensor,            # (M, K) result-noise log variances
    log_pi:            torch.Tensor,            # (M,) or (..., M) per token: unnormalized log prior weights
    *,
    regression_offset: Optional[torch.Tensor] = None,   # (M, K, K) D_a, F_a = I + D_a
    regression_left:   Optional[torch.Tensor] = None,   # (M, K, r) U_a
    regression_right:  Optional[torch.Tensor] = None,   # (M, K, r) V_a, F_a = I + U_a V_a^T
    tile_bytes:        int = CUE_RESULT_TILE_BYTES,
    fused:             bool = False,            # the fused Triton route on CUDA tensors (vfe3.kernels.cue_result)
) -> CueResultPosterior:
    r"""The exact result posterior for a diagonal record covariance, with the diagonal M-projection of its covariance
    (module docstring, Diagonal record): ``covariance`` holds the (..., K) result variances. With ``fused`` and a
    low-rank regression of rank at most :data:`vfe3.kernels.cue_result.MAX_RANK`, CUDA tensors run the same posterior
    in fused Triton kernels (``tile_bytes`` then bounds their per-row planes); everything else runs the torch tiles."""
    _check_inputs(xi, c, nu, omega, log_lambda, log_result_noise, log_pi,
                  regression_offset, regression_left, regression_right, diagonal=True)
    if tile_bytes < 1:
        raise ValueError(f"tile_bytes must be a positive byte count, got {tile_bytes!r}")
    batch, K, M = xi.shape[:-1], xi.shape[-1], nu.shape[0]
    rows = math.prod(batch)
    reg = (regression_offset, regression_left, regression_right)
    dtype = _compute_dtype(xi, c, nu, omega, log_lambda, log_result_noise, log_pi,
                           *(t for t in reg if t is not None))
    if fused and regression_left is not None:
        from vfe3.kernels import cue_result as fused_kernels
        if regression_left.shape[-1] <= fused_kernels.MAX_RANK and fused_kernels.fused_route_active(xi):
            rank = regression_left.shape[-1]
            planes = 2 * (2 * fused_kernels._rank_pad(rank) + 2 * fused_kernels._rank_pad(rank) ** 2) + 2
            row_chunk = max(1, int(tile_bytes) // ((torch.finfo(dtype).bits // 8) * planes * M))
            with torch.autocast(device_type=xi.device.type, enabled=False):
                log_pi_hat = torch.log_softmax(log_pi.to(dtype), dim=-1)
                if log_pi_hat.dim() > 1:
                    log_pi_hat = log_pi_hat.reshape(rows, M)                             # (R, M) per token
                mean, var, log_weights, log_evidence = fused_kernels.cue_result_diagonal_lowrank_fused(
                    xi.to(dtype).reshape(rows, K), c.to(dtype).reshape(rows, K), nu.to(dtype), omega.to(dtype),
                    log_lambda.to(dtype).exp(), log_result_noise.to(dtype).exp(), log_pi_hat,
                    regression_left.to(dtype), regression_right.to(dtype), row_chunk=row_chunk)
            return CueResultPosterior(mean.reshape(*batch, K), var.reshape(*batch, K),
                                      log_weights.reshape(*batch, M), log_evidence.reshape(batch))
    with torch.autocast(device_type=xi.device.type, enabled=False):
        cast = lambda t: None if t is None else t.to(dtype)                              # noqa: E731
        x, cv = xi.to(dtype).reshape(rows, K), c.to(dtype).reshape(rows, K)
        nu_c, omega_c = cast(nu), cast(omega)
        lam, q = log_lambda.to(dtype).exp(), log_result_noise.to(dtype).exp()
        log_pi_hat = torch.log_softmax(log_pi.to(dtype), dim=-1)
        per_token = log_pi_hat.dim() > 1
        if per_token:
            log_pi_hat = log_pi_hat.reshape(rows, M)                                     # (R, M) per token
        offset, left, right = (cast(t) for t in reg)
        rank = 0 if left is None else left.shape[-1]
        # About eight (r, m, K) tensors per tile, plus (r, m, K, K)-sized work for a dense F or (r, m, rank^2).
        per_pair = K * (8 + (K if offset is not None else 0)) + rank * rank
        pairs = max(1, int(tile_bytes) // ((torch.finfo(dtype).bits // 8) * per_pair))
        m_step = min(M, pairs)
        r_step = max(1, min(rows, pairs // m_step))
        tiled = rows > r_step or M > m_step
        grad = torch.is_grad_enabled()
        log_w_rows, mean_rows, var_rows, s_rows, l_rows = [], [], [], [], []
        for i in range(0, rows, r_step):
            running = None
            chunks = []
            for j in range(0, M, m_step):
                sl = slice(j, j + m_step)
                prior = log_pi_hat[i:i + r_step, sl] if per_token else log_pi_hat[sl]
                args = (x[i:i + r_step], cv[i:i + r_step], nu_c[sl], omega_c[sl], lam[sl], q[sl], prior,
                        None if offset is None else offset[sl], None if left is None else left[sl],
                        None if right is None else right[sl])
                if tiled and grad and any(t is not None and t.requires_grad for t in args):
                    log_w, s, l, mean, var = checkpoint(_diagonal_tile, *args, use_reentrant=False,
                                                        preserve_rng_state=False)
                else:
                    log_w, s, l, mean, var = _diagonal_tile(*args)
                chunks.append(log_w)
                running = (s, l, mean, var) if running is None else _merge_diagonal(*running, s, l, mean, var)
            s_rows.append(running[0])
            l_rows.append(running[1])
            mean_rows.append(running[2])
            var_rows.append(running[3])
            log_w_rows.append(torch.cat(chunks, dim=-1))
        mean, var = torch.cat(mean_rows), torch.cat(var_rows)                            # (R, K)
        log_weights, log_evidence = _normalized_outputs(torch.cat(log_w_rows), torch.cat(s_rows),
                                                        torch.cat(l_rows))              # (R, M), (R,)
    return CueResultPosterior(mean.reshape(*batch, K), var.reshape(*batch, K),
                              log_weights.reshape(*batch, M), log_evidence.reshape(batch))
