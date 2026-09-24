r"""Gaussian exponential families (diagonal and full covariance) for VFE_3.0.

The closed-form Renyi/KL kernels are ported verbatim from the legacy ``divergence.py``
moment forms, so the live numerics are byte-identical; natural-parameter and log-partition
maps are added so the generic Bregman/Renyi-from-A path can be pinned against them.
"""

import math
from typing import List, Optional, Tuple

import torch

from vfe3.families.base import (
    BeliefParams, register_family, safe_kl_clamp, _logdet_chol,
)
from vfe3.numerics import safe_cholesky, validated_cholesky_solve


# fp32 catastrophic-cancellation band around the alpha->1 (KL) limit of the Renyi logdet term.
# The |alpha-1| < 1e-6 KL switch in each kernel is exact in float64, but in float32 the three
# nearly-equal logs in the logdet term cancel and are divided by a tiny (alpha-1), losing ~1%
# accuracy out to roughly |alpha-1| ~ 1e-3. Inside this band the closed forms evaluate the logdet
# term in float64 (the same dtype-guarded headroom the geometry Gram pseudo-inverse already uses)
# and cast back; outside it the float32 quotient is accurate and used verbatim (byte-identical).
_RENYI_KL_BAND: float = 1e-2


def _full_gaussian_public_dtype(*dtypes: torch.dtype) -> torch.dtype:
    return torch.float64 if torch.float64 in dtypes else torch.float32


# --- full-covariance KL precision policy (audit 2026-08-05) --------------------------------
# "fp64": the historical unconditional float64 island -- the default, so every existing caller,
#         test, and viz path is byte-identical to the pre-policy build.
# "fp32_escalate": evaluate the pair grid in float32 and retain it only when Task 4's
#         zero-jitter residual/conditioning certificate passes; otherwise recompute and return
#         float64. "fp32_escalate_cond" adds its legacy pivot-ratio trigger to that shared gate.
# The module-level value remains the standalone-call default. Model-owned callers pass an immutable
# instance policy instead, so constructing another model cannot alter an existing model's numerics.
_FULL_COV_KL_PRECISIONS = ("fp64", "fp32_escalate", "fp32_escalate_cond")
# Pivot-ratio floor at which "fp32_escalate_cond" promotes the grid to float64.
#
# CALIBRATED, NOT DERIVED. The proxy is min(diag L)^2 / max(diag L)^2, which is a lower bound on
# 1/cond and under-reports true conditioning by 2-3 decades, so the threshold was measured rather
# than reasoned (K=20, 20 draws per row, random orthogonal eigenbasis):
#
#   true cond   pivot ratio   fp32 relative KL error
#   1e2         1.04e-1       4.1e-07
#   1e3         1.98e-2       1.3e-06
#   1e4         3.63e-3       2.0e-05
#   1e5         4.35e-4       1.4e-04
#   1e6         5.69e-5       8.8e-04      <- error becomes material here
#   1e7         9.44e-6       9.0e-03
#   1e8         1.12e-6       1.3e-01
#   1e9         1.85e-5       9.8e-01      <- NOT monotone: the jitter ladder has ridged the
#                                             matrix, so the pivots describe the RIDGED input
#
# 1e-4 escalates from true cond ~1e6 (ratio 5.7e-5) and leaves cond 1e5 alone (4.35e-4), i.e. it
# fires exactly where the float32 error crosses 1e-3 relative. The trained operating point (cond
# median 2.7e3 / p95 2.1e4, config.py:854-856) sits at ratio ~2e-2..1.5e-3, one to two decades
# clear, so this does not fire during normal training. The 1e9 row is non-monotone but still below
# the floor, so the trigger holds there too.
_FULL_COV_KL_COND_FLOOR: float = 1e-4
_FULL_COV_KL_PRECISION: str = "fp64"


def validate_full_cov_kl_precision(policy: str) -> str:
    """Validate and return one supported full-covariance KL precision policy."""
    if policy not in _FULL_COV_KL_PRECISIONS:
        raise ValueError(
            f"full_cov_kl_precision must be one of {_FULL_COV_KL_PRECISIONS}, got {policy!r}")
    return policy


def set_full_cov_kl_precision(policy: str) -> str:
    r"""Set the process-wide full-covariance KL precision policy; returns the previous value."""
    global _FULL_COV_KL_PRECISION
    validate_full_cov_kl_precision(policy)
    previous = _FULL_COV_KL_PRECISION
    _FULL_COV_KL_PRECISION = policy
    return previous


def full_cov_kl_precision() -> str:
    r"""Return the active full-covariance KL precision policy."""
    return _FULL_COV_KL_PRECISION


def _full_cov_kl_compute_dtype(
    result_dtype: torch.dtype,
    precision_policy: Optional[str] = None,
) -> torch.dtype:
    r"""Working dtype for the full-covariance KL under an explicit or standalone policy.

    A float64 PUBLIC family always computes in float64 regardless of policy -- the policy only
    governs whether a float32 family is promoted, which is what the historical island did."""
    policy = full_cov_kl_precision() if precision_policy is None else validate_full_cov_kl_precision(precision_policy)
    if policy == "fp64" or result_dtype == torch.float64:
        return torch.float64
    return torch.float32


def _inv_cond_from_chol(factor: torch.Tensor) -> torch.Tensor:
    r"""``min(diag L)^2 / max(diag L)^2`` -- a free lower bound on ``1 / cond(L L^T)``."""
    diag = torch.diagonal(factor, dim1=-2, dim2=-1).abs()
    lo = diag.amin(dim=-1)
    hi = diag.amax(dim=-1).clamp_min(torch.finfo(diag.dtype).tiny)
    return (lo / hi) ** 2


def _full_gaussian_kl_terms(
    mu_q:    torch.Tensor,             # (..., K) query means
    sigma_q: torch.Tensor,             # (..., K, K) query covariances
    mu_t:    torch.Tensor,             # (..., K) transported key means
    sigma_t: torch.Tensor,             # (..., K, K) transported key covariances
    K:       int,
    eps:     float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # KL, ok mask, 1/cond estimate
    r"""Exact full-covariance KL(q||t) and its per-element factorization mask.

    Lifted verbatim out of ``FullGaussian.renyi_closed_form``'s alpha==1 branch so the float64
    escalation can re-evaluate the identical arithmetic at a higher precision rather than
    duplicating it (audit 2026-08-05). Every operand must already share one dtype.
    """
    L_p, ok_p = safe_cholesky(sigma_t, eps=eps, rounds=5)
    L_q, ok_q = safe_cholesky(sigma_q, eps=eps, rounds=5)
    Y = torch.linalg.solve_triangular(L_p, sigma_q, upper=False)
    Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
    trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)
    delta_mu = mu_t - mu_q
    v = torch.linalg.solve_triangular(
        L_p, delta_mu.unsqueeze(-1), upper=False
    ).squeeze(-1)
    mahal_term = (v ** 2).sum(dim=-1)
    logdet_p = _logdet_chol(L_p)
    logdet_q = _logdet_chol(L_q)
    div = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
    # Free conditioning estimate from the factors already computed: for Sigma = L L^T the squared
    # Cholesky pivots bracket the spectrum, so min(diag L)^2 / max(diag L)^2 is a lower bound on
    # 1/cond(Sigma). Costs two reductions over an existing tensor -- no extra factorization.
    inv_cond = torch.minimum(_inv_cond_from_chol(L_p), _inv_cond_from_chol(L_q))
    return div, ok_p & ok_q, inv_cond



def _full_gaussian_kl_coordinate_terms(
    mu_q:    torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p:    torch.Tensor,
    sigma_p: torch.Tensor,
    eps:     float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Return D_k=E_q[KL(q(x_k|x_<k)||p(x_k|x_<k))] in stored coordinate order.

    For T=L_p^{-1}L_q and v=L_p^{-1}(mu_q-mu_p),
    D_k=(sum_j T_kj^2 + v_k^2 - 1 - 2 log T_kk)/2.
    The off-diagonal row norm retains the conditional regression mismatch.
    """
    L_q, ok_q = safe_cholesky(sigma_q, eps=eps, rounds=5)
    L_p, ok_p = safe_cholesky(sigma_p, eps=eps, rounds=5)
    T = torch.linalg.solve_triangular(L_p, L_q, upper=False)
    v = torch.linalg.solve_triangular(L_p, (mu_q - mu_p).unsqueeze(-1), upper=False).squeeze(-1)
    pivot = T.diagonal(dim1=-2, dim2=-1)
    delta = pivot - 1
    variance_cost = delta.square() + 2 * (delta - torch.log1p(delta))
    terms = .5 * (variance_cost + T.tril(-1).square().sum(-1) + v.square())
    inv_cond = torch.minimum(_inv_cond_from_chol(L_q), _inv_cond_from_chol(L_p))
    return terms, ok_q & ok_p, inv_cond


def diag_kl_unclamped(
    mu_q:    torch.Tensor,             # (..., N, K) query means
    sigma_q: torch.Tensor,             # (..., N, K) query variances
    mu_p:    torch.Tensor,             # (..., N, K) prior means
    sigma_p: torch.Tensor,             # (..., N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (..., N) UNCLAMPED KL(q||p)
    r"""Unclamped diagonal KL(q||p) = 0.5 Sum_k (s_k/t_k + (mu_p - mu_q)^2/t_k - 1 + log(t_k/s_k)).

    Returns the raw divergence scalar (summed over the last coordinate axis) WITHOUT clamping,
    so callers that need the pre-clamp value for saturation masking (e.g. ``_diag_kl_filtering_kernel``)
    can inspect it before applying ``safe_kl_clamp``.
    """
    return diag_kl_unclamped_per_coord(mu_q, sigma_q, mu_p, sigma_p, eps=eps).sum(-1)



def diag_kl_unclamped_per_coord(
    mu_q:    torch.Tensor,             # (..., N, K) query means
    sigma_q: torch.Tensor,             # (..., N, K) query variances
    mu_p:    torch.Tensor,             # (..., N, K) prior means
    sigma_p: torch.Tensor,             # (..., N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (..., N, K) UNCLAMPED per-coordinate KL D^(k)(q||p)
    r"""Unclamped per-coordinate diagonal KL D^(k) = 0.5 (s_k/t_k + (mu_p - mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The per-coordinate analog of ``diag_kl_unclamped``: the -K of the summed form becomes -1 per
    coordinate.  Summing over k recovers ``diag_kl_unclamped`` (no coordinate saturates the clamp).
    """
    sq = sigma_q.clamp(min=eps)
    sp = sigma_p.clamp(min=eps)
    delta = (sq - sp) / sp
    # Near equality, evaluate x-log(1+x) without subtracting its linear term.
    near = delta.abs() < 1e-3
    x = torch.where(near, delta, torch.zeros_like(delta))
    series = x.square() * (0.5 + x * (-1.0 / 3.0 + x * (0.25 - x / 5.0)))
    covariance = torch.where(near, series, sq / sp - 1.0 + torch.log(sp) - torch.log(sq))
    return 0.5 * (covariance + ((mu_p - mu_q) ** 2) / sp)


@register_family("gaussian_diagonal")
class DiagonalGaussian(BeliefParams):
    r"""Diagonal Gaussian: mu (..., K), sigma (..., K) variances.

    Natural theta = (mu/sigma, -1/(2 sigma)); A(theta) = sum_k [ -t1^2/(4 t2) - 1/2 log(-2 t2) ];
    E[T] = (mu, mu^2 + sigma).
    """

    cov_kind = "diagonal"
    dispersion_is_covariance = True
    # Scale-normalize the participation ratio (audit 2026-07-27). ``effective_rank`` floors its
    # denominator ``sum lam^2`` at the family value, and the Gaussian families returned the bare
    # ``eps`` -- a covariance^1 quantity guarding a covariance^2 denominator. The ratio
    # ``(sum lam)^2 / sum lam^2`` is homogeneous of degree 0, so it must equal K for ANY isotropic
    # spectrum at ANY scale; with the mismatched floor it collapsed instead. Measured at K=20 on a
    # flat spectrum (true value 20 everywhere): 20.000000 / 19.999998 / 4.000000 / 0.040000 /
    # 0.000400 as sigma fell 1e0 -> 1e-6. At the live K=210, eps=1e-6 the floor binds below
    # sigma ~ 7e-5, INSIDE the admissible [eps, sigma_max] band, so eff_rank_* reported a
    # maximally FLAT covariance as sub-rank-1 collapse -- the opposite of what it exists to
    # detect. ``laplace_diagonal`` already opted in; these did not.
    effective_rank_rescale = True
    gaussian_pointwise_algebra = True

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu
        self.sigma = sigma

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    @classmethod
    def from_covariance(cls, mu: torch.Tensor, covariance: torch.Tensor) -> "DiagonalGaussian":
        r"""Read a full covariance out as its variances -- the diagonal family's projection.

        This is precisely why the diagonal family is NOT GL(K)-admissible: re-diagonalizing
        ``g Sigma g^T`` discards the off-diagonal mass a non-diagonal ``g`` creates.
        """
        return cls(mu, torch.diagonal(covariance, dim1=-2, dim2=-1))

    def block(self, start: int, end: int) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "DiagonalGaussian":
        return DiagonalGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    @classmethod
    def stack(cls, parts: List["DiagonalGaussian"], *, dim: int = 0) -> "DiagonalGaussian":
        return DiagonalGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    @classmethod
    def covariance_diagonal(
        cls,
        dispersion: torch.Tensor,                    # (..., K) coordinate variances

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        return dispersion

    @classmethod
    def covariance_floor(cls, eps: float) -> float:
        return eps

    @classmethod
    def effective_rank_floor(cls, eps: float) -> float:
        r"""Preserve the historical Gaussian participation-ratio stabilizer ``eps``."""
        return eps

    @classmethod
    def mean_fisher_precision(
        cls,
        dispersion: torch.Tensor,                    # (..., K) coordinate variances

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        return dispersion.clamp(min=eps).reciprocal()

    @classmethod
    def mean_fisher_quadratic(
        cls,
        mu:         torch.Tensor,                    # (..., K) mean coordinates
        dispersion: torch.Tensor,                    # (..., K) coordinate variances

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        return mu ** 2 / dispersion.clamp(min=eps)

    @classmethod
    def trust_region_scale(
        cls,
        dispersion: torch.Tensor,                    # (..., K) coordinate variances

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        return dispersion.clamp(min=eps).sqrt()

    @classmethod
    def mix_dispersion(
        cls,
        dispersion: torch.Tensor,                    # (..., n, d) independent variances
        mixing:    torch.Tensor,                     # (m, n) component mixer
    ) -> torch.Tensor:
        return torch.einsum("mn,...nd->...md", mixing.square(), dispersion)

    @classmethod
    def diagnostic_labels(cls) -> dict[str, str]:
        return {
            "dispersion":             "Gaussian variance",
            "covariance_spectrum":    "covariance variance",
            "half_mean_fisher_trace": r"Half Fisher trace $\frac{1}{2}\sum_k \sigma_k^{-1}$",
        }

    def natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.sigma.clamp(min=1e-12)
        return (self.mu / s, -1.0 / (2.0 * s))

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        return (-(t1 ** 2) / (4.0 * t2) - 0.5 * torch.log(-2.0 * t2)).sum(dim=-1)

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.mu, self.mu ** 2 + self.sigma)

    def entropy(self) -> torch.Tensor:
        return 0.5 * (
            torch.log(self.sigma.clamp(min=1e-12)) + math.log(2.0 * math.pi * math.e)
        ).sum(dim=-1)

    def renyi_closed_form(
        self,
        other:   "DiagonalGaussian",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Closed-form diagonal Gaussian Renyi/KL (ported verbatim from
        ``divergence._gaussian_diagonal_renyi``; mu_q=self, mu_t=other).

        ``eps`` is the variance floor for the log/division terms and MUST equal the upstream belief
        sigma floor (``cfg.eps``, default 1e-6; the SPD retraction clamps every sigma to
        [eps, sigma_max], so on the live pipeline every input sigma is already >= cfg.eps and this
        clamp is inert defense-in-depth, never the binding floor). This is a DIFFERENT, looser floor
        than the hardcoded 1e-12 in ``natural()``/``entropy()`` (the generic A-path): the closed form
        tracks the pipeline floor, while the natural-parameter maps only need positivity for
        log(-2 t2)."""
        compute_dtype = (torch.float64
                         if torch.float64 in (self.mu.dtype, self.sigma.dtype,
                                              other.mu.dtype, other.sigma.dtype)
                         else torch.float32)
        mu_q = self.mu.to(compute_dtype)
        sigma_q = self.sigma.to(compute_dtype).clamp(min=eps)
        mu_t = other.mu.to(compute_dtype)
        sigma_t = other.sigma.to(compute_dtype).clamp(min=eps)
        if abs(alpha - 1.0) < 1e-6:
            div = diag_kl_unclamped(mu_q, sigma_q, mu_t, sigma_t, eps=eps)
        else:
            # alpha in (0,1): the blend is a convex combination of positive variances, so it is
            # always > 0 and the mask below is inert (byte-identical to the old clamp-only path).
            # alpha > 1 leaves the convex regime: a coordinate's blend can go non-positive, which
            # makes the divergence undefined. clamp(min=eps) here only guards log/division on the
            # GOOD coordinates; a non-PD coordinate maps the whole pair to NaN -> kl_max below,
            # mirroring the full-cov safe_cholesky mask (instead of emitting a wrong finite value).
            raw_blend   = (1.0 - alpha) * sigma_q + alpha * sigma_t
            sigma_blend = raw_blend.clamp(min=eps)
            delta       = mu_t - mu_q
            mahal_term  = (alpha * (delta ** 2) / sigma_blend).sum(dim=-1)
            if abs(alpha - 1.0) < _RENYI_KL_BAND:
                # fp32 cancellation band: evaluate the logdet term in float64, then cast back.
                sq64 = sigma_q.double()
                st64 = sigma_t.double()
                sb64 = ((1.0 - alpha) * sq64 + alpha * st64).clamp(min=eps)
                logdet_term = (
                    ((1.0 - alpha) * torch.log(sq64) + alpha * torch.log(st64) - torch.log(sb64)).sum(dim=-1)
                    / (alpha - 1.0)
                ).to(sigma_q.dtype)
            else:
                logdet_per_dim = (
                    (1.0 - alpha) * torch.log(sigma_q)
                    + alpha * torch.log(sigma_t)
                    - torch.log(sigma_blend)
                )
                logdet_term = logdet_per_dim.sum(dim=-1) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
            ok  = (raw_blend > 0.0).all(dim=-1)            # any non-PD coordinate -> NaN -> kl_max
            div = torch.where(ok, div, div.new_tensor(float("nan")))
        return safe_kl_clamp(div, kl_max=kl_max)

    def renyi_per_coord(
        self,
        other:   "DiagonalGaussian",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Per-coordinate diagonal Gaussian Renyi/KL: the diagonal Renyi/KL coordinate terms
        left UNSUMMED, each clamped independently. Summing over k recovers ``renyi_closed_form``
        ONLY when no per-coordinate term saturates ``kl_max``: this routine clamps each coordinate
        to kl_max independently, whereas the closed form clamps the single summed scalar once, so a
        coordinate that individually exceeds kl_max diverges between the two. The independent
        per-coordinate clamp is the intended behavior here; only the recovery identity is conditional."""
        compute_dtype = (torch.float64
                         if torch.float64 in (self.mu.dtype, self.sigma.dtype,
                                              other.mu.dtype, other.sigma.dtype)
                         else torch.float32)
        mu_q = self.mu.to(compute_dtype)
        sigma_q = self.sigma.to(compute_dtype).clamp(min=eps)
        mu_t = other.mu.to(compute_dtype)
        sigma_t = other.sigma.to(compute_dtype).clamp(min=eps)
        delta = mu_t - mu_q
        if abs(alpha - 1.0) < 1e-6:
            per_coord = diag_kl_unclamped_per_coord(mu_q, sigma_q, mu_t, sigma_t, eps=eps)
        else:
            # See renyi_closed_form: alpha in (0,1) blend is always > 0 (mask inert, byte-identical);
            # for alpha > 1 a non-positive-blend coordinate is masked to NaN -> kl_max PER COORDINATE
            # (the per-coord twin of the summed mask), so a bad coordinate is gated without killing
            # its in-bounds neighbors.
            raw_blend   = (1.0 - alpha) * sigma_q + alpha * sigma_t
            sigma_blend = raw_blend.clamp(min=eps)
            mahal       = alpha * (delta ** 2) / sigma_blend
            if abs(alpha - 1.0) < _RENYI_KL_BAND:
                # fp32 cancellation band: evaluate the per-coord logdet term in float64 (see
                # renyi_closed_form / _RENYI_KL_BAND), then cast back.
                sq64   = sigma_q.double()
                st64   = sigma_t.double()
                sb64   = ((1.0 - alpha) * sq64 + alpha * st64).clamp(min=eps)
                logdet = (
                    ((1.0 - alpha) * torch.log(sq64) + alpha * torch.log(st64) - torch.log(sb64))
                    / (alpha - 1.0)
                ).to(sigma_q.dtype)
            else:
                logdet  = (
                    (1.0 - alpha) * torch.log(sigma_q)
                    + alpha * torch.log(sigma_t)
                    - torch.log(sigma_blend)
                ) / (alpha - 1.0)
            per_coord = 0.5 * (mahal + logdet)
            per_coord = torch.where(raw_blend > 0.0, per_coord, per_coord.new_tensor(float("nan")))
        return safe_kl_clamp(per_coord, kl_max=kl_max)

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else grad wrt variance

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Diagonal-Gaussian Fisher preconditioner ``(sigma*grad_mu, 2 sigma^2 grad_sigma)``.
        Delegates to the pinned geometry kernel so the live numerics stay byte-identical to the
        golden-tested ``retraction.natural_gradient`` (local import avoids a families<-geometry
        import edge)."""
        from vfe3.geometry.retraction import natural_gradient as _gaussian_natural_gradient
        return _gaussian_natural_gradient(grad_mu, grad_sigma, self.sigma, eps=eps)


@register_family("gaussian_full")
class FullGaussian(BeliefParams):
    r"""Full-covariance Gaussian: mu (..., K), sigma (..., K, K) SPD covariance.

    Natural theta = (Sigma^{-1} mu, -1/2 Sigma^{-1}); A(theta) = -1/4 t1^T t2^{-1} t1 - 1/2 log|-2 t2|.

    External construction retains the any-operand float64 result policy, including mixed public
    mean/covariance dtypes. ``from_transported`` records the source covariance dtype while the
    family transport seam retains an internal float64 covariance, so this numerical representation
    does not promote the public divergence result.
    """

    cov_kind = "full"
    conditional_kl_coordinates = True
    dispersion_is_covariance = True
    # Scale-normalize the participation ratio (audit 2026-07-27). ``effective_rank`` floors its
    # denominator ``sum lam^2`` at the family value, and the Gaussian families returned the bare
    # ``eps`` -- a covariance^1 quantity guarding a covariance^2 denominator. The ratio
    # ``(sum lam)^2 / sum lam^2`` is homogeneous of degree 0, so it must equal K for ANY isotropic
    # spectrum at ANY scale; with the mismatched floor it collapsed instead. Measured at K=20 on a
    # flat spectrum (true value 20 everywhere): 20.000000 / 19.999998 / 4.000000 / 0.040000 /
    # 0.000400 as sigma fell 1e0 -> 1e-6. At the live K=210, eps=1e-6 the floor binds below
    # sigma ~ 7e-5, INSIDE the admissible [eps, sigma_max] band, so eff_rank_* reported a
    # maximally FLAT covariance as sub-rank-1 collapse -- the opposite of what it exists to
    # detect. ``laplace_diagonal`` already opted in; these did not.
    effective_rank_rescale = True
    gaussian_pointwise_algebra = True

    @classmethod
    def from_covariance(cls, mu: torch.Tensor, covariance: torch.Tensor) -> "FullGaussian":
        r"""The full covariance IS this family's stored dispersion -- an exact readout."""
        return cls(mu, covariance)

    def __init__(
        self,
        mu:    torch.Tensor,
        sigma: torch.Tensor,

        *,
        _public_dtype: Optional[torch.dtype] = None,
        _precision_policy: Optional[str] = None,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self._public_dtype = (
            _public_dtype
            if _public_dtype is not None
            else _full_gaussian_public_dtype(mu.dtype, sigma.dtype)
        )
        self._precision_policy = (
            None if _precision_policy is None else validate_full_cov_kl_precision(_precision_policy)
        )

    @classmethod
    def from_transported(
        cls,
        mu:                torch.Tensor,         # transported means
        dispersion:        torch.Tensor,         # retained full covariance
        source_dispersion: torch.Tensor,         # pre-transport covariance and public dtype source
        *,
        _precision_policy: Optional[str] = None,
    ) -> "FullGaussian":
        return cls(
            mu,
            dispersion,
            _public_dtype=_full_gaussian_public_dtype(mu.dtype, source_dispersion.dtype),
            _precision_policy=_precision_policy,
        )

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "FullGaussian":
        return FullGaussian(
            self.mu[..., start:end],
            self.sigma[..., start:end, start:end],
            _public_dtype=self._public_dtype,
            _precision_policy=self._precision_policy,
        )

    def broadcast_over_keys(self) -> "FullGaussian":
        return FullGaussian(
            self.mu.unsqueeze(-2),
            self.sigma.unsqueeze(-3),
            _public_dtype=self._public_dtype,
            _precision_policy=self._precision_policy,
        )

    @classmethod
    def stack(cls, parts: List["FullGaussian"], *, dim: int = 0) -> "FullGaussian":
        public_dtype = _full_gaussian_public_dtype(*(part._public_dtype for part in parts))
        explicit_policies = {part._precision_policy for part in parts if part._precision_policy is not None}
        if len(explicit_policies) > 1:
            raise ValueError(
                "cannot stack FullGaussian values with conflicting explicit precision policies")
        precision_policy = next(iter(explicit_policies), None)
        return FullGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
            _public_dtype=public_dtype,
            _precision_policy=precision_policy,
        )

    @classmethod
    def covariance_diagonal(
        cls,
        dispersion: torch.Tensor,                    # (..., K, K) covariance

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        return torch.diagonal(dispersion, dim1=-2, dim2=-1)

    @classmethod
    def covariance_floor(cls, eps: float) -> float:
        return eps

    @classmethod
    def effective_rank_floor(cls, eps: float) -> float:
        r"""Preserve the historical Gaussian participation-ratio stabilizer ``eps``."""
        return eps

    @classmethod
    def mean_fisher_precision(
        cls,
        dispersion: torch.Tensor,                    # (..., K, K) covariance

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        symmetric = 0.5 * (dispersion + dispersion.transpose(-1, -2))
        eye = torch.eye(
            dispersion.shape[-1],
            device=dispersion.device,
            dtype=dispersion.dtype,
        )
        factor, ok = safe_cholesky(symmetric, eps=eps, rounds=5)
        safe_factor = torch.where(
            ok.unsqueeze(-1).unsqueeze(-1),
            factor,
            eye.expand_as(factor),
        )
        precision = torch.cholesky_inverse(safe_factor)
        if not bool(ok.all()):
            precision = torch.where(
                ok.unsqueeze(-1).unsqueeze(-1),
                precision,
                torch.linalg.pinv(symmetric),
            )
        return 0.5 * (precision + precision.transpose(-1, -2))

    @classmethod
    def trust_region_scale(
        cls,
        dispersion: torch.Tensor,                    # (..., K, K) covariance

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        factor, _ = safe_cholesky(dispersion, eps=eps, rounds=0)
        return factor

    @classmethod
    def mix_dispersion(
        cls,
        dispersion: torch.Tensor,                    # (..., K, K) covariance
        mixing:    torch.Tensor,                     # (K, K) dense linear map
    ) -> torch.Tensor:
        return mixing @ dispersion @ mixing.transpose(-1, -2)

    @classmethod
    def transport_dispersion(
        cls,
        dispersion: torch.Tensor,         # (..., N, K, K) full covariance
        omega:      object,               # dense/factored/direct-link/RoPE transport container

        *,
        diagonal_out:    Optional[bool]      = None,
        marginal_blocks: Optional[List[int]] = None,
        precision_policy: Optional[str] = None,
    ) -> torch.Tensor:
        r"""Retain the float64 full-covariance congruence through SPD factorization.

        A dense sandwich can be positive definite in float64 but indefinite after float32 storage.
        The public divergence retains the source-family dtype; only this transported covariance is
        kept in float64 until the FullGaussian KL/Renyi computation has consumed it.

        Retention is keyed to whether the KL will ACTUALLY consume float64 (audit 2026-08-06 F2).
        Under ``full_cov_kl_precision="fp32_escalate"`` :meth:`renyi_closed_form` immediately does
        ``sigma_t.to(compute_dtype)`` with ``compute_dtype = float32``, so retaining produced a
        float64 pairwise ``(B, N, N, K, K)`` tensor -- 629 MB at the live shape -- and rounded it
        straight back. Keying off the policy drops that round trip.

        BIT-IDENTICAL under both policies, and the reason is specific: every full-covariance
        congruence in ``transport.py`` hard-codes float64 ARITHMETIC independently of this flag, so
        the flag chooses only the OUTPUT dtype, and float32 -> float64 promotion is exact.
        Round-early vs round-late measured 0/50 mismatches across the compact, blocks-only and dense
        routes. (Do NOT instead key ``compute_dtype`` on ``other.sigma.dtype`` -- that silently
        disables the ``full_cov_kl_precision`` knob entirely.)

        This is the family the ``marginal_blocks`` promise pays off in -- the only one whose
        transported dispersion is a full (K, K) congruence with cross-head blocks to skip.
        """
        from vfe3.geometry.transport import transport_covariance
        policy = full_cov_kl_precision() if precision_policy is None else validate_full_cov_kl_precision(precision_policy)
        return transport_covariance(
            omega,
            dispersion,
            retain_full_precision=(policy == "fp64"),
            diagonal_out=diagonal_out,
            marginal_blocks=marginal_blocks,
        )

    @classmethod
    def coupling_energy(
        cls,
        mu_q:         torch.Tensor,       # (..., N, K) query locations
        dispersion_q: torch.Tensor,       # (..., N, K, K) query covariance
        mu_k:         torch.Tensor,       # (..., N, K) key locations, PRE-transport
        dispersion_k: torch.Tensor,       # (..., N, K, K) key covariance, PRE-transport
        omega:        object,             # dense/factored/direct-link/RoPE transport container

        *,
        alpha:             float = 1.0,
        kl_max:            float = 100.0,
        eps:               float = 1e-6,
        divergence_family: str   = "renyi",

        irrep_dims:       Optional[List[int]]                   = None,
        diagonal_out:     Optional[bool]                        = None,
        precision_policy: Optional[str]                         = None,
        pair_support:     Optional[torch.Tensor]                = None,  # (..., H, Nq, Nk)
        score_backend:    str                                  = "pairwise",
    ) -> torch.Tensor:                    # (..., N, N) or (..., H, N, N)
        r"""Full-covariance coupling energy, taking the head blocks directly when it can.

        Identical in value to :meth:`BeliefParams.coupling_energy`, which it falls back to whenever
        the fast route does not apply. The fast route exists because this family is the only one
        whose transported dispersion is a full ``(K, K)`` congruence, and because the base seam's own
        ``marginal_blocks`` promise means the cross-head output blocks are provably never read:
        ``transport_covariance`` still has to hand back a dense ``(..., N, N, K, K)`` (every other
        caller is typed against it), so it scatters the head blocks into a 629 MB zero tensor at the
        live shape and ``pairwise_energy`` slices them straight back out. Asking the geometry for the
        ``(..., N, N, H, d, d)`` blocks instead removes that round trip entirely
        (audit 2026-08-06 B2/F14).

        BIT-IDENTICAL, and the reason is narrow enough to state: the blocks are the SAME einsum on
        the SAME operands the dense route already runs -- the scatter and the re-slice are pure data
        movement -- and the ``.contiguous()`` below reproduces the exact strides ``torch.stack``
        produced, so nothing downstream sees a different memory layout and reassociates.

        The rank guard replicates :func:`~vfe3.free_energy._stackable_for_batching`, which the dense
        route applies AFTER transport. Checking it here on the pre-transport inputs is equivalent:
        transport adds one key axis to the mean and one to the covariance, and
        ``broadcast_over_keys`` adds one to each, so ``sigma.dim() - mu.dim()`` is invariant along
        the whole chain. Failing it falls back rather than stacking a spurious broadcast.

        For compact equal-head transport, ``pair_support`` restricts the energy read set: excluded
        head pairs use constant SPD placeholders and return zero energy and gradient. It must include
        every pair used by the caller, including any two-hop terms. Other structural/projection
        routes retain their existing complete-grid evaluation.
        """
        policy = full_cov_kl_precision() if precision_policy is None else validate_full_cov_kl_precision(precision_policy)
        if score_backend not in ("pairwise", "common_frame", "common_frame_factored"):
            raise ValueError(f"Unsupported full covariance score backend: {score_backend!r}")
        if (score_backend in ("common_frame", "common_frame_factored")
                and cls is FullGaussian and not diagonal_out):
            from vfe3.families.base import get_functional, renyi
            # Registry identity matters: a same-name replacement must keep its own semantics.
            if alpha == 1.0 and get_functional(divergence_family) is renyi:
                from vfe3.families.full_gaussian_score import common_frame_kl_energy
                energy = common_frame_kl_energy(
                    mu_q, dispersion_q, mu_k, dispersion_k, omega,
                    kl_max=kl_max, precision_policy=policy, irrep_dims=irrep_dims,
                    pair_support=pair_support, factored=(score_backend == "common_frame_factored"),
                )
                if energy is not None:
                    return energy
        # Private provenance kwargs are an implementation detail of the exact built-in class.
        # Registry replacements and downstream subclasses historically implement the two-argument
        # constructor / three-argument from_transported contract, so preserve that public surface.
        def _make(mu: torch.Tensor, dispersion: torch.Tensor) -> "FullGaussian":
            if cls is FullGaussian:
                return cls(mu, dispersion, _precision_policy=policy)
            return cls(mu, dispersion)

        def _from_transported(
            mu: torch.Tensor, dispersion: torch.Tensor, source: torch.Tensor,
        ) -> "FullGaussian":
            if cls is FullGaussian:
                return cls.from_transported(mu, dispersion, source, _precision_policy=policy)
            return cls.from_transported(mu, dispersion, source)
        blocks = None
        head_support = None
        if (irrep_dims is not None and len(irrep_dims) > 1
                and len(set(irrep_dims)) == 1
                and dispersion_q.dim() == mu_q.dim() + 1
                and dispersion_k.dim() == mu_k.dim() + 1):
            from vfe3.geometry.transport import transport_covariance_head_blocks
            if pair_support is not None:
                batch_shape = torch.broadcast_shapes(mu_q.shape[:-2], mu_k.shape[:-2])
                energy_shape = (*batch_shape, len(irrep_dims), mu_q.shape[-2], mu_k.shape[-2])
                head_support = torch.broadcast_to(pair_support, energy_shape).movedim(-3, -1)
            blocks = transport_covariance_head_blocks(       # None unless the compact route applies
                omega, dispersion_k,
                retain_full_precision=(policy == "fp64"),
                diagonal_out=diagonal_out,
                marginal_blocks=irrep_dims,
                **({"pair_support": head_support} if head_support is not None else {}),
            )
        if blocks is None:
            from vfe3.free_energy import pairwise_energy
            mu_t = cls.transport_location(mu_k, omega)
            transport_kwargs = dict(
                diagonal_out=diagonal_out,
                marginal_blocks=irrep_dims,
            )
            if cls is FullGaussian:
                transport_kwargs["precision_policy"] = policy
            dispersion_t = cls.transport_dispersion(dispersion_k, omega, **transport_kwargs)
            return pairwise_energy(
                _make(mu_q, dispersion_q),
                _from_transported(mu_t, dispersion_t, dispersion_k),
                alpha=alpha, kl_max=kl_max, eps=eps,
                divergence_family=divergence_family, irrep_dims=irrep_dims,
            )

        from vfe3.free_energy import head_stacked_energy
        H, d = len(irrep_dims), irrep_dims[0]
        mu_t = cls.transport_location(mu_k, omega)                     # (..., N, N, K)
        mu_blocks = mu_t.reshape(*mu_t.shape[:-1], H, d)
        if head_support is not None:
            mu_blocks = torch.where(head_support[..., None], mu_blocks, 0.0)
        key_stack = _from_transported(
            mu_blocks.movedim(-2, 0).contiguous(),                         # (H,...,N,N,d)
            blocks.movedim(-3, 0).contiguous(),                                 # (H,...,N,N,d,d)
            dispersion_k,
        )
        energy = head_stacked_energy(
            _make(mu_q, dispersion_q).broadcast_over_keys(), key_stack,
            alpha=alpha, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family, irrep_dims=irrep_dims,
        )
        if head_support is not None:
            energy = energy.masked_fill(~head_support.movedim(-1, -3), 0.0)
        return energy

    @classmethod
    def diagnostic_labels(cls) -> dict[str, str]:
        return {
            "dispersion":             "Gaussian covariance",
            "covariance_spectrum":    "covariance eigenvalue",
            "half_mean_fisher_trace": r"Half Fisher trace $\mathrm{tr}(\Sigma^{-1})/2$",
        }

    def natural(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ridge the covariance before inverting to the precision (the natural parameter); the
        # 1e-6 jitter matches the eps the closed-form full kernel uses for SPD safety. NOT routed
        # through ``safe_spd_inverse``: that helper's Cholesky-inverse diverges from this LU solve by
        # ~1.5e-4 once cond(Sigma) >= 1e3 (reachable here, eps=1e-6 floor / sigma_max=5 cap give
        # cond up to ~5e6), so centralizing would CHANGE the live result -- pinned by
        # tests/test_fix_numerics_audit.py::test_natural_inverse_not_routed_solve_vs_safe_spd_diverges.
        mu = self.mu if self.mu.dtype == self._public_dtype else self.mu.to(self._public_dtype)
        sigma = self.sigma if self.sigma.dtype == self._public_dtype else self.sigma.to(self._public_dtype)
        eye = torch.eye(mu.shape[-1], device=mu.device, dtype=self._public_dtype)
        prec = torch.linalg.solve(sigma + 1e-6 * eye, eye.expand_as(sigma))
        t1 = (prec @ mu.unsqueeze(-1)).squeeze(-1)
        return (t1, -0.5 * prec)

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t1, t2 = theta
        neg2t2 = -2.0 * t2
        L, ok = safe_cholesky(neg2t2, rounds=5)        # never raises; ok=False on a non-PD -2*t2
        inv_neg2t2 = torch.cholesky_inverse(L)
        quad = (t1.unsqueeze(-2) @ inv_neg2t2 @ t1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        out = 0.5 * quad - 0.5 * _logdet_chol(L)
        return torch.where(ok, out, out.new_tensor(float("nan")))

    def expected_statistic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        outer = self.mu.unsqueeze(-1) * self.mu.unsqueeze(-2)
        return (self.mu, self.sigma + outer)

    def entropy(self) -> torch.Tensor:
        K = self.mu.shape[-1]
        L, ok = safe_cholesky(self.sigma, rounds=5)
        entropy = 0.5 * _logdet_chol(L) + 0.5 * K * math.log(2.0 * math.pi * math.e)
        return torch.where(ok, entropy, entropy.new_tensor(float("nan")))

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else grad wrt covariance

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Full-Gaussian Fisher preconditioner ``(Sigma grad_mu, 2 Sigma grad_sigma Sigma)``
        (symmetrized). Delegates to the pinned geometry kernel, which selects the full-covariance
        branch by rank (``sigma.dim() == grad_mu.dim() + 1``), so the numerics stay byte-identical
        to the golden-tested ``retraction.natural_gradient``."""
        from vfe3.geometry.retraction import natural_gradient as _gaussian_natural_gradient
        return _gaussian_natural_gradient(grad_mu, grad_sigma, self.sigma, eps=eps)

    def renyi_per_coord(
        self,
        other:   "FullGaussian",
        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Ordered conditional KL contributions with sum_k D_k = KL(q||p) before clipping.

        Correlations contribute through conditional regressions, not marginal KLs.
        Individual contributions depend on coordinate order; the unclipped sum
        does not. Each term is clipped separately, matching diagonal alpha weights.
        General Renyi orders have no such KL chain rule and remain unsupported.
        """
        if alpha != 1.0:
            raise ValueError("full-Gaussian per-coordinate self-coupling requires KL (alpha=1)")
        if (self._precision_policy is not None and other._precision_policy is not None
                and self._precision_policy != other._precision_policy):
            raise ValueError("FullGaussian operands have conflicting explicit precision policies")
        policy = self._precision_policy or other._precision_policy or full_cov_kl_precision()
        public_dtype = _full_gaussian_public_dtype(self._public_dtype, other._public_dtype)
        compute_dtype = _full_cov_kl_compute_dtype(public_dtype, policy)
        with torch.autocast(device_type=self.mu.device.type, enabled=False):
            args = tuple(x.to(compute_dtype) for x in (self.mu, self.sigma, other.mu, other.sigma))
            terms, ok, inv_cond = _full_gaussian_kl_coordinate_terms(*args, eps)
            if compute_dtype != torch.float64:
                escalate = ~ok
                if policy == "fp32_escalate_cond":
                    escalate = (escalate | (inv_cond < _FULL_COV_KL_COND_FLOOR)
                                | ~validated_cholesky_solve(args[1]).certified
                                | ~validated_cholesky_solve(args[3]).certified)
                if bool(escalate.any()):
                    args = tuple(x.double() for x in args)
                    terms, ok, _ = _full_gaussian_kl_coordinate_terms(*args, eps)
            terms = torch.where(ok.unsqueeze(-1), terms, terms.new_tensor(float("nan")))
            return safe_kl_clamp(terms, kl_max=kl_max).to(public_dtype)

    def renyi_closed_form(
        self,
        other:   "FullGaussian",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
        precision_policy: Optional[str] = None,
    ) -> torch.Tensor:
        r"""Closed-form full-covariance Gaussian Renyi/KL (ported verbatim from
        ``divergence._gaussian_full_renyi``; mu_q=self, mu_t=other)."""
        K = self.mu.shape[-1]
        # Preserve the public dtype policy (float64 for a public float64 family, otherwise float32),
        # but evaluate every full-SPD factorization and cancellation in float64. Near the admitted
        # cond(Sigma)~1e6 ceiling, float32 can turn the exact self-KL and shared derivative into
        # O(1e-2) and O(1e4) artifacts, respectively.
        # External construction retains the any-operand float64 policy. Family-owned transport
        # records the source public dtype before retaining an internal float64 covariance, and the
        # constructor plus shape combinators carry that provenance to this boundary.
        result_dtype = (torch.float64
                        if torch.float64 in (self._public_dtype, other._public_dtype)
                        else torch.float32)
        return_dtype = result_dtype
        # Precision policy (audit 2026-08-05). Historically this was an unconditional
        # `compute_dtype = torch.float64` -- unlike the `result_dtype` line directly above it and
        # unlike DiagonalGaussian.renyi_closed_form, which keys on its operand dtypes. On the live
        # gaussian_full pair grid that meant every cholesky / solve_triangular / logdet ran in
        # float64, measured at 15.7x the float32 cost on a consumer GPU whose float64 throughput is
        # heavily reduced. The policy is process-global ON PURPOSE: `coupling_energy` has 15 call
        # sites (oracle, phi objective, reported F, gamma coupling, viz), and threading a kwarg
        # through them invites exactly the desynchronisation that lost `cond_escalation` in
        # `_transport_to_float` -- a gradient computed at one precision against an objective
        # reported at another. One policy, one read site, no way for two paths to disagree.
        if precision_policy is not None:
            policy = validate_full_cov_kl_precision(precision_policy)
        elif (self._precision_policy is not None and other._precision_policy is not None
              and self._precision_policy != other._precision_policy):
            raise ValueError("FullGaussian operands have conflicting explicit precision policies; pass precision_policy explicitly")
        else:
            policy = self._precision_policy or other._precision_policy
        compute_dtype = _full_cov_kl_compute_dtype(result_dtype, policy)
        mu_q = self.mu.to(compute_dtype)
        sigma_q = self.sigma.to(compute_dtype)
        mu_t = other.mu.to(compute_dtype)
        sigma_t = other.sigma.to(compute_dtype)
        # NO unconditional eps ridge (audit 2026-07-05 m1): the old ``sigma + eps*eye`` scored
        # N(mu, Sigma + eps I) ALWAYS -- a standing 1e-6 bias in values and gradients on the pure
        # full-covariance path, unlike the diagonal path's clamp (inert whenever the SPD retraction
        # keeps sigma >= eps). Robustness against a numerically non-PD covariance is owned by
        # safe_cholesky below: round 0 adds ZERO jitter (valid-SPD inputs are byte-identical to
        # torch.linalg.cholesky), failures get an escalating eps ridge, and a pair failing every
        # round -> NaN -> safe_kl_clamp -> kl_max.
        if abs(alpha - 1.0) < 1e-6:
            # safe_cholesky (jittered cholesky_ex, never raises) hardens the KL against a
            # numerically non-PD prior/posterior covariance -- reachable once training shifts the
            # prior, and routed here for any full-cov / non-kernel config via the E-step oracle.
            # This matches the robustness the alpha != 1 branch already has. Round 0 adds zero
            # jitter, so valid-SPD inputs stay byte-identical to torch.linalg.cholesky; an element
            # that fails every round -> NaN -> safe_kl_clamp -> kl_max (mirroring that branch).
            div, ok, inv_cond = _full_gaussian_kl_terms(mu_q, sigma_q, mu_t, sigma_t, K, eps)
            # Data-keyed escalation: under a float32 policy a pair is recomputed in float64 rather
            # than being written off to kl_max. The whole grid is escalated (not the failed subset)
            # so the returned tensor keeps one dtype and one autograd graph; the branch is skipped
            # entirely on the float64 policy, which is why that path stays bit-identical to the
            # pre-policy build.
            #
            # The base fp32_escalate policy is the production hot path: reuse the Cholesky result
            # above and promote only on an actual failed row. The opt-in *_cond policy adds Task 4's
            # zero-jitter residual/spectral certificate plus the older pivot-ratio trigger. That
            # strict check is deliberately gated because a full pair grid can contain millions of
            # matrices; running a second Cholesky and eigvalsh on every one caused a 3x baseline
            # regression and a whole-grid float64 cliff once a single row crossed the condition
            # limit. The strict route retains that behavior for accuracy-sensitive experiments.
            #
            # The default limit is conservative and dimension/dtype dependent; at K=20 float32 it
            # is about 6.55e3, so upper-tail rows may promote even when Cholesky succeeds.
            # That is the explicit opt-in accuracy policy, not a failure fallback.
            if compute_dtype is not torch.float64:
                active_policy = full_cov_kl_precision() if policy is None else policy
                escalate_mask = ~ok
                if active_policy == "fp32_escalate_cond":
                    q_certificate = validated_cholesky_solve(sigma_q)
                    t_certificate = validated_cholesky_solve(sigma_t)
                    escalate_mask = (
                        escalate_mask
                        | ~q_certificate.certified
                        | ~t_certificate.certified
                        | (inv_cond < _FULL_COV_KL_COND_FLOOR)
                    )
                if bool(escalate_mask.any()):
                    div, ok, _ = _full_gaussian_kl_terms(
                        mu_q.double(), sigma_q.double(), mu_t.double(), sigma_t.double(), K, eps)
                    return_dtype = torch.float64
            div = torch.where(ok, div, div.new_tensor(float("nan")))
        else:
            # alpha > 1 leaves the convex regime: the blend can be indefinite for some
            # (i,j) pairs. safe_cholesky factors per element (cholesky_ex, never raises),
            # tries an escalating eps ridge on failures, and returns an `ok` mask; a pair
            # that fails ALL rounds is set to NaN so safe_kl_clamp maps it to kl_max while
            # good pairs in the same batch keep their finite divergence. Round 0 adds zero
            # extra jitter, so valid-SPD inputs stay byte-identical to torch.linalg.cholesky.
            sigma_blend = (1.0 - alpha) * sigma_q + alpha * sigma_t
            sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))
            L_blend, _ = safe_cholesky(sigma_blend, eps=eps, rounds=5)  # factor for mahal_term only
            L_q, ok_q = safe_cholesky(sigma_q, eps=eps, rounds=5)
            L_t, ok_t = safe_cholesky(sigma_t, eps=eps, rounds=5)
            delta_mu = mu_t - mu_q
            v = torch.linalg.solve_triangular(
                L_blend, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = alpha * (v ** 2).sum(dim=-1)
            if abs(alpha - 1.0) < _RENYI_KL_BAND:
                # The three logdets nearly cancel before division by (alpha - 1). Retain the
                # established slogdet cancellation route inside the float64 island.
                sq64 = sigma_q.double()
                st64 = sigma_t.double()
                sb64 = (1.0 - alpha) * sq64 + alpha * st64
                sb64 = 0.5 * (sb64 + sb64.transpose(-1, -2))
                logdet_term = (
                    ((1.0 - alpha) * torch.linalg.slogdet(sq64).logabsdet
                     + alpha * torch.linalg.slogdet(st64).logabsdet
                     - torch.linalg.slogdet(sb64).logabsdet)
                    / (alpha - 1.0)
                ).to(sigma_q.dtype)
            else:
                logdet_q = _logdet_chol(L_q)
                logdet_t = _logdet_chol(L_t)
                logdet_blend = _logdet_chol(L_blend)
                logdet_term = (
                    (1.0 - alpha) * logdet_q + alpha * logdet_t - logdet_blend
                ) / (alpha - 1.0)
            div = 0.5 * (mahal_term + logdet_term)
            # safe_cholesky's escalating ridge can factor an INDEFINITE blend as PD (its ok mask
            # True) even though alpha>1 left the convex regime and the Renyi divergence is undefined
            # there -- the fp64 slogdet then drops the sign and the value collapses to ~0 instead of
            # kl_max. Gate on the SIGN of the (symmetrized) blend so a non-PD blend -> NaN -> kl_max
            # regardless of the ridge (audit 2026-06-17). sigma_q/sigma_t are SPD on the live
            # pipeline (SPD retraction); safe_cholesky owns the off-pipeline non-PD handling.
            #
            # SCOPE (audit 2026-08-07). This gate was unscoped -- it ran for every alpha != 1 -- but
            # its justification above holds only ABOVE alpha=1. For alpha in (0,1) the blend is a
            # strict convex combination of two SPD matrices, hence SPD, so the test could only ever
            # return True: measured min-eigenvalue 5.2e-3 over 2000 pairs x alpha in {.25,.5,.75},
            # and 0/4000 fp32 non-PD verdicts at every cond from 1e2 to 1e10. Not merely inert --
            # at cond 1e12 it FALSELY rejected 25/4000 blends whose exact fp64 spectrum is strictly
            # positive, converting a valid divergence into a gradient-dead kl_max.
            #
            # It is also the ONLY batch-limited op in this branch. torch.linalg.eigvalsh routes to
            # cuSOLVER syevBatched for n <= 32, which rejects the CALL past ~2.6e4-3.2e4 matrices
            # (K=8: 29915/29916, K=20: 26305/26306; IDENTICAL in fp32 and fp64, so it is a shape-only
            # parameter rejection, not a workspace-byte limit -- PyTorch 2.8.0 regression
            # syevjBatched_bufferSize -> xsyevBatched_bufferSize, pytorch/pytorch#166004). That put a
            # hard ceiling on the whole full-covariance non-KL surface: squared_hellinger and
            # bhattacharyya both hardcode alpha=0.5, and the E-step pair grid H*B*N^2 crosses it at
            # batch_size=1 (2*64*128^2 = 2.1e6 at the live config, 71.7x over). cholesky_ex carries
            # no such ceiling (verified to 2,097,152), reproduces the spectral verdict 20000/20000 at
            # alpha in {1.5,2.0,5.0}, and costs ~K^3/3 flops against eigvalsh's ~9K^3. Measured 8.0x
            # end-to-end on family_chunked decode, with max relative value diff 0.000e+00.
            if alpha > 1.0:
                # info == 0 IS the numerical PD predicate, and NO ridge is added here (unlike
                # safe_cholesky above), so an indefinite blend is reported as such.
                _, blend_info = torch.linalg.cholesky_ex(sigma_blend)
                blend_pd = blend_info == 0
                ok = blend_pd & ok_q & ok_t        # non-PD blend or failed factor -> NaN -> kl_max
            else:
                ok = ok_q & ok_t                   # convex blend of SPD is SPD; only factors can fail
            div = torch.where(ok, div, div.new_tensor(float("nan")))
        return safe_kl_clamp(div, kl_max=kl_max).to(return_dtype)
