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
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch

from vfe3.runtime import ieee_fp32_matmul

CUE_RESULT_TILE_BYTES = 32 * 1024 ** 2          # bound on one chunk's (tokens, m, K, K) tensors, not total memory


class CueResultPosterior(NamedTuple):
    """Moments of the exact result marginal, the component log weights and the record log evidence."""

    mean:         torch.Tensor                  # (..., K) result mean sum_a w_a m_{Y,a}
    covariance:   torch.Tensor                  # (..., K, K) result covariance with the between-component spread
    log_weights:  torch.Tensor                  # (..., M) log w_a
    log_evidence: torch.Tensor                  # (...) log p(xi | C)


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


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
) -> None:
    K = xi.shape[-1]
    if C.dim() != xi.dim() + 1 or C.shape[-2:] != (K, K) or C.shape[:-2] != xi.shape[:-1]:
        raise ValueError(f"cue/result memory needs a dense record covariance (..., K, K) matching xi {tuple(xi.shape)}, "
                         f"got {tuple(C.shape)}")
    M = nu.shape[0]
    for name, t in (("nu", nu), ("omega", omega), ("log_lambda", log_lambda), ("log_result_noise", log_result_noise)):
        if t.shape != (M, K):
            raise ValueError(f"{name} must have shape ({M}, {K}), got {tuple(t.shape)}")
    if log_pi.shape != (M,):
        raise ValueError(f"log_pi must have shape ({M},), got {tuple(log_pi.shape)}")
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
    log_pi_hat: torch.Tensor,                   # (m,) normalized log prior
    offset: Optional[torch.Tensor],
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""One chunk: log weights (T, m), the chunk's log mass (T,), and its normalized result mean and covariance."""
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
    log_mass = torch.logsumexp(log_w, dim=-1)                                            # (T,)
    w = torch.exp(log_w - log_mass.unsqueeze(-1))                                        # (T, m) within-chunk weights
    mean = (w.unsqueeze(-1) * m_Y).sum(-2)                                               # (T, K)
    centered = m_Y - mean.unsqueeze(-2)                                                  # (T, m, K)
    cov = (w[..., None, None] * (P_Y + centered.unsqueeze(-1) * centered.unsqueeze(-2))).sum(-3)
    return log_w, log_mass, mean, cov


def _merge(
    log_mass_a: torch.Tensor, mean_a: torch.Tensor, cov_a: torch.Tensor,
    log_mass_b: torch.Tensor, mean_b: torch.Tensor, cov_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Law of total covariance for two normalized mixtures with log masses: a positive sum."""
    log_mass = torch.logaddexp(log_mass_a, log_mass_b)
    alpha = torch.exp(log_mass_a - log_mass).unsqueeze(-1)                               # (T, 1)
    beta = torch.exp(log_mass_b - log_mass).unsqueeze(-1)
    diff = mean_a - mean_b                                                               # (T, K)
    mean = alpha * mean_a + beta * mean_b
    cov = (alpha.unsqueeze(-1) * cov_a + beta.unsqueeze(-1) * cov_b
           + (alpha * beta).unsqueeze(-1) * diff.unsqueeze(-1) * diff.unsqueeze(-2))
    return log_mass, mean, cov


@ieee_fp32_matmul()
def cue_result_exact_posterior(
    xi:                torch.Tensor,            # (..., K) record value (the working mean in the declared chart)
    C:                 torch.Tensor,            # (..., K, K) record covariance
    nu:                torch.Tensor,            # (M, K) cue means
    omega:             torch.Tensor,            # (M, K) result means
    log_lambda:        torch.Tensor,            # (M, K) cue log variances
    log_result_noise:  torch.Tensor,            # (M, K) result-noise log variances
    log_pi:            torch.Tensor,            # (M,) unnormalized log prior weights
    *,
    regression_offset: Optional[torch.Tensor] = None,   # (M, K, K) D_a, F_a = I + D_a
    regression_left:   Optional[torch.Tensor] = None,   # (M, K, r) U_a
    regression_right:  Optional[torch.Tensor] = None,   # (M, K, r) V_a, F_a = I + U_a V_a^T
    tile_bytes:        int = CUE_RESULT_TILE_BYTES,
) -> CueResultPosterior:
    r"""The moments of :math:`p(Y\mid O=\xi, C)` for the model in the module docstring, with the component log
    weights and :math:`\log p(\xi\mid C) = \log\sum_a\hat\pi_a\mathcal N(\xi;\nu_a, L_a + C)`."""
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
        offset, left, right = (cast(t) for t in reg)
        T = xi_f.shape[0]
        per_component = max(1, T) * K * K * torch.finfo(dtype).bits // 8 * 6             # ~6 (T, K, K) tensors live
        chunk = max(1, min(M, int(tile_bytes) // max(1, per_component)))
        log_ws = []
        running = None
        for start in range(0, M, chunk):
            sl = slice(start, min(M, start + chunk))
            log_w, log_mass, mean, cov = _chunk_moments(
                xi_f, C_f, nu_c[sl], omega_c[sl], lam[sl], q[sl], log_pi_hat[sl],
                None if offset is None else offset[sl], None if left is None else left[sl],
                None if right is None else right[sl])
            log_ws.append(log_w)
            running = (log_mass, mean, cov) if running is None else _merge(*running, log_mass, mean, cov)
        log_evidence, mean, cov = running
        log_weights = torch.cat(log_ws, dim=-1) - log_evidence.unsqueeze(-1)             # (T, M) normalized
        cov = 0.5 * (cov + cov.transpose(-2, -1))
    return CueResultPosterior(mean.reshape(*batch, K), cov.reshape(*batch, K, K),
                              log_weights.reshape(*batch, M), log_evidence.reshape(batch))
