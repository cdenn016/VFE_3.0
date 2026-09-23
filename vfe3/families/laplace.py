r"""Diagonal (factorized) Laplace belief family for VFE_3.0 -- the first non-Gaussian family.

Per coordinate ``p(x_k) = (1 / 2 b_k) exp(-|x_k - mu_k| / b_k)`` (location ``mu``, scale ``b``).
The belief tuple ``(mu, sigma)`` is reused verbatim: ``mu`` is the location and ``sigma`` is the
per-coordinate scale ``b`` (>0, floored by the SPD retraction exactly as a variance is). Genuinely
non-Gaussian: exponential (L1) tails, excess kurtosis 3, a non-differentiable cusp at the mode; the
fixed-location sufficient statistic is ``|x - mu|`` (not the Gaussian ``(x, x^2)``), so no
reparameterization maps a Laplace marginal to a Gaussian one. It is the canonical heavy-tailed /
sparse (LASSO) prior counterpart to the Gaussian.

NOT a natural exponential family when the location varies: ``|x - mu|`` depends on the parameter
``mu``, so the log-density is not affine in a fixed sufficient statistic and there is NO joint
log-partition ``A(theta)`` over ``(mu, b)``. The generic Bregman/Renyi-from-``A`` path
(``families/base.py``) is therefore unavailable here; ``natural`` / ``log_partition_at`` raise, and
the family ships a mandatory closed-form ``renyi_closed_form`` (the divergence dispatch in
``base.renyi`` prefers it whenever present). The Laplace IS an EF only in the degenerate scale-only
slice (``eta = -1/b``, ``T = |x - mu|`` with ``mu`` known), which the live varying-location belief is
not, so the override is required rather than optional.

DIVERGENCES (per coordinate; summed over k by ``renyi_closed_form``). With ``s = |mu_q - mu_p|``:

  KL (alpha = 1):  D = log(b_p / b_q) + s / b_p + (b_q / b_p) exp(-s / b_q) - 1   (>= 0, 0 iff q == p).

  Renyi (alpha in (0,1]):  D_alpha = 1/(alpha - 1) * log( int q^alpha p^{1-alpha} dx ). With
  c_q = alpha / b_q, c_p = (1-alpha) / b_p (both > 0 on (0,1)), the two-region exponential integral is

      int = (1 / 2) [ (e^{-c_p s} + e^{-c_q s}) / (c_q + c_p)
                      + (e^{-c_q s} - e^{-c_p s}) / (c_p - c_q) ],

  with a REMOVABLE singularity at ``c_p == c_q`` i.e. ``alpha* = b_q / (b_p + b_q)`` (the second term
  -> ``s exp(-c s)`` in the limit). The singularity is per-coordinate and data-dependent (each k has
  its own ``alpha*_k``), unlike the Gaussian's fixed band around ``alpha = 1``; the alpha != 1 branch
  is evaluated in float64 (catastrophic cancellation otherwise loses ~all fp32 digits within ~1e-6 of
  ``alpha*``), with a ``where`` limit branch at ``|c_p - c_q| < 1e-12``. The float64 island also
  covers the outer ``log(int) / (alpha - 1)`` cancellation as ``alpha -> 1``. For ``alpha > 1`` the
  blend rate can go non-positive (``c_q + c_p <= 0``), the integral diverges, and that coordinate maps
  to NaN -> ``kl_max`` -- mirroring the Gaussian non-convex-regime policy.

  Validated to ~1e-10 against deterministic float64 trapezoidal integration (KL, Renyi on (0,1),
  the singularity limit, and self-divergence) before this module was written.

TRANSPORT (honest scope). The family-selected dispersion seam transports the scale with degree one.
For a signed diagonal/permutation element this is exact: ``b'_k = |Omega_kk| b_k``. Under a general
mixing operator a linear combination of independent Laplace variables is not factorized Laplace, so
the live diagonal family uses the explicit variance-matching marginal projection
``b'_k = sqrt(sum_l Omega_kl^2 b_l^2)``. This preserves dimensional homogeneity and exactness on the
family-preserving subgroup without claiming distributional closure under rotations or a general
non-compact GL(K) action. The full-covariance Gaussian remains the exact congruence family off that
subgroup.
"""

import math
from typing import List, Optional, Tuple

import torch

from vfe3.families.base import BeliefParams, register_family, safe_kl_clamp


@register_family("laplace_diagonal")
class DiagonalLaplace(BeliefParams):
    r"""Factorized Laplace: ``mu`` (..., K) location, ``sigma`` (..., K) per-coordinate scale ``b``."""

    cov_kind = "diagonal"
    dispersion_is_covariance = False
    effective_rank_rescale = True

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu = mu                                   # (..., K) location
        self.sigma = sigma                             # (..., K) scale b (the belief sigma slot)
        self._public_dtype = torch.promote_types(mu.dtype, sigma.dtype)

    def coordinate_dim(self) -> int:
        return self.mu.shape[-1]

    def block(self, start: int, end: int) -> "DiagonalLaplace":
        return DiagonalLaplace(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "DiagonalLaplace":
        return DiagonalLaplace(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    @classmethod
    def stack(cls, parts: List["DiagonalLaplace"], *, dim: int = 0) -> "DiagonalLaplace":
        return DiagonalLaplace(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    @classmethod
    def covariance_diagonal(
        cls,
        dispersion: torch.Tensor,                    # (..., K) Laplace scale b

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        r"""``diag(Cov[X]) = 2 b^2`` for independent Laplace coordinates."""
        return 2.0 * dispersion.clamp(min=eps).square()

    @classmethod
    def covariance_floor(cls, eps: float) -> float:
        r"""Covariance floor ``2 eps^2`` induced by the Laplace scale floor ``eps``."""
        return 2.0 * eps ** 2

    @classmethod
    def effective_rank_floor(cls, eps: float) -> float:
        r"""Squared covariance-unit stabilizer for the Laplace participation ratio."""
        return cls.covariance_floor(eps) ** 2

    @classmethod
    def mean_fisher_precision(
        cls,
        dispersion: torch.Tensor,                    # (..., K) Laplace scale b

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        r"""Mean-block Fisher precision ``I_mu = 1 / b^2``."""
        return dispersion.clamp(min=eps).square().reciprocal()

    @classmethod
    def mean_fisher_quadratic(
        cls,
        mu:         torch.Tensor,                    # (..., K) mean coordinates
        dispersion: torch.Tensor,                    # (..., K) Laplace scale b

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        r"""Mean Fisher quadratic contributions ``mu_k^2 / b_k^2``."""
        return mu ** 2 / dispersion.clamp(min=eps).square()

    @classmethod
    def trust_region_scale(
        cls,
        dispersion: torch.Tensor,                    # (..., K) Laplace scale b

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:
        r"""Fisher-whitening scale ``I_mu^{-1/2} = b``."""
        return dispersion.clamp(min=eps)

    @classmethod
    def mix_dispersion(
        cls,
        dispersion: torch.Tensor,                    # (..., n, d) independent scales b
        mixing:    torch.Tensor,                     # (m, n) component mixer
    ) -> torch.Tensor:
        r"""Moment-matched scale ``b'_m = sqrt(sum_n A_mn^2 b_n^2)``."""
        mixed_b2 = torch.einsum("mn,...nd->...md", mixing.square(), dispersion.square())
        mixed_b2 = mixed_b2.clamp_min(0.0)
        positive = mixed_b2 > 0.0
        safe_b2 = torch.where(positive, mixed_b2, torch.ones_like(mixed_b2))
        return torch.where(positive, safe_b2.sqrt(), torch.zeros_like(safe_b2))

    @classmethod
    def diagnostic_labels(cls) -> dict[str, str]:
        return {
            "dispersion":             "Laplace scale b",
            "covariance_spectrum":    "marginal covariance variance",
            "half_mean_fisher_trace": r"Half Fisher trace $\frac{1}{2}\sum_k b_k^{-2}$",
        }

    @classmethod
    def transport_dispersion(
        cls,
        dispersion: torch.Tensor,         # (..., N, K) marginal scale b
        omega:      object,               # dense/factored/direct-link/RoPE transport container

        *,
        diagonal_out:    Optional[bool]      = True,
        marginal_blocks: Optional[List[int]] = None,
    ) -> torch.Tensor:
        r"""Degree-one scale action, exact on signed diagonal/permutation gauges.

        Off that subgroup the result is the variance-matching factorized-Laplace projection, not
        an assertion that the pushed-forward joint law remains factorized Laplace.

        ``marginal_blocks`` is accepted for interface symmetry with the base seam and intentionally
        unused: the marginal scale ``b`` carries no cross-head block to skip.
        """
        from vfe3.geometry.transport import transport_scale
        return transport_scale(dispersion, omega, diagonal_out=diagonal_out)

    def natural(self) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "DiagonalLaplace is not a natural exponential family when the location varies "
            "(the sufficient statistic |x - mu| depends on the parameter mu), so it has no "
            "natural parameterization theta. It carries a closed-form renyi_closed_form instead; "
            "the generic Bregman/Renyi-from-A divergence path is not available for this family."
        )

    @classmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError(
            "DiagonalLaplace has no joint log-partition A(theta) over (mu, b): it is a natural "
            "exponential family only in the degenerate fixed-location, scale-only slice. Use its "
            "closed-form renyi_closed_form (the divergence dispatch prefers it); the generic "
            "Renyi-from-A path cannot serve this family."
        )

    def entropy(self) -> torch.Tensor:
        # H = sum_k log(2 b_k e) = sum_k [ log(2 b_k) + 1 ].
        sigma = self.sigma.to(self._public_dtype)
        return (torch.log(2.0 * sigma.clamp(min=1e-12)) + 1.0).sum(dim=-1)

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu (location)
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else grad wrt scale b

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:   # (nat_mu, nat_b) = Fisher^{-1} grad
        r"""Diagonal-Laplace Fisher preconditioner. The Laplace(mu, b) Fisher information is
        DIAGONAL and EQUAL on both coordinates, ``I_mu = I_b = 1 / b^2`` (a location-scale family;
        verified symbolically), so the natural gradient is ``b^2 * grad`` on BOTH the location and
        the scale. This differs from the Gaussian preconditioner ``(sigma*grad_mu, 2 sigma^2
        grad_sigma)``: routing the Laplace belief through the Gaussian Fisher mis-scales the mean by
        a state-dependent ``1/b`` (a wrong direction on the product manifold, not a rescaled LR).
        ``b^2 > 0`` strictly, so the step is sign-preserving and ``grad=0 -> step=0`` (every
        stationary point of F preserved). ``eps`` floors ``b`` (the belief sigma floor, ``cfg.eps``)
        exactly as the divergence does."""
        result_dtype = torch.promote_types(self._public_dtype, grad_mu.dtype)
        if grad_sigma is not None:
            result_dtype = torch.promote_types(result_dtype, grad_sigma.dtype)
        compute_dtype = torch.float64 if result_dtype is torch.float64 else torch.float32
        with torch.amp.autocast(self.sigma.device.type, enabled=False):
            b2 = self.sigma.to(compute_dtype).clamp(min=eps).square()
            nat_mu = b2 * grad_mu.to(compute_dtype)
            nat_sigma = None if grad_sigma is None else b2 * grad_sigma.to(compute_dtype)
        return (nat_mu.to(result_dtype),
                None if nat_sigma is None else nat_sigma.to(result_dtype))

    def _renyi_terms(
        self,
        other:  "DiagonalLaplace",

        *,
        alpha:  float,
        eps:    float,
    ) -> torch.Tensor:                                 # (..., K) UNSUMMED per-coordinate divergence
        r"""Per-coordinate Renyi/KL D^(k)(self || other), unclamped. KL (|alpha-1|<1e-6) uses the
        exact discrete form in float32; alpha != 1 uses the float64 integral form (covers the
        per-coordinate removable singularity at c_p==c_q and the alpha->1 outer cancellation)."""
        result_dtype = torch.promote_types(self._public_dtype, other._public_dtype)
        compute_dtype = torch.float64 if result_dtype is torch.float64 else torch.float32
        mu_q = self.mu.to(compute_dtype)
        b_q = self.sigma.to(compute_dtype).clamp(min=eps)
        mu_p = other.mu.to(compute_dtype)
        b_p = other.sigma.to(compute_dtype).clamp(min=eps)
        s = (mu_q - mu_p).abs()
        if abs(alpha - 1.0) < 1e-6:
            return torch.log(b_p) - torch.log(b_q) + s / b_p + (b_q / b_p) * torch.exp(-s / b_q) - 1.0
        # alpha != 1: float64 island (no byte-identity constraint for a new family).
        mu_q64 = mu_q.double()
        b_q64  = b_q.double()
        mu_p64 = mu_p.double()
        b_p64  = b_p.double()
        s64    = (mu_q64 - mu_p64).abs()
        c_q  = alpha / b_q64
        c_p  = (1.0 - alpha) / b_p64
        csum = c_q + c_p
        # Split the integral at the two locations. Each interval is positive, so its log can
        # be combined without materializing interval exponentials, which underflow for
        # alpha < 1 or overflow for alpha > 1 at large separation. Divergent tail blends are replaced before any branch
        # algebra, keeping the documented NaN -> kl_max policy without poisoning gradients.
        convergent = csum > 0.0
        c_q_safe    = torch.where(convergent, c_q, torch.ones_like(c_q))
        c_p_safe    = torch.where(convergent, c_p, torch.zeros_like(c_p))
        csum_safe   = torch.where(convergent, csum, torch.ones_like(csum))
        s_safe      = torch.where(convergent, s64, torch.zeros_like(s64))

        c_mid = 0.5 * (c_p_safe + c_q_safe)
        x     = 0.5 * (c_p_safe - c_q_safe) * s_safe
        small = x.abs() < 1e-3
        x_s   = torch.where(small, x, torch.zeros_like(x))
        log_sinhc_small = torch.log1p(x_s.square() / 6.0 + x_s.pow(4) / 120.0)

        x_l = torch.where(small, torch.ones_like(x), x.abs())
        log_sinhc_large = (
            x_l + torch.log1p(-torch.exp(-2.0 * x_l))
            - math.log(2.0) - torch.log(x_l)
        )
        log_sinhc = torch.where(small, log_sinhc_small, log_sinhc_large)

        tiny       = torch.finfo(s_safe.dtype).tiny
        log_middle = -c_mid * s_safe + torch.log(s_safe.clamp_min(tiny)) + log_sinhc
        log_left   = -c_p_safe * s_safe - torch.log(csum_safe)
        log_right  = -c_q_safe * s_safe - torch.log(csum_safe)
        log_integral = torch.logaddexp(torch.logaddexp(log_left, log_middle), log_right)
        log_int = (log_integral - math.log(2.0)
                   - alpha * torch.log(b_q64) - (1.0 - alpha) * torch.log(b_p64))
        div = log_int / (alpha - 1.0)
        div = torch.where(convergent, div, div.new_tensor(float("nan")))
        return div.to(result_dtype)

    def renyi_closed_form(
        self,
        other:   "DiagonalLaplace",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Closed-form factorized-Laplace Renyi/KL D_alpha(self || other), summed over coordinates.

        ``eps`` floors the scale ``b`` (the belief sigma floor, ``cfg.eps``). The summed scalar is
        clamped once (any NaN coordinate from a divergent alpha>1 blend -> kl_max for the whole pair,
        mirroring the Gaussian non-PD policy)."""
        terms = self._renyi_terms(other, alpha=alpha, eps=eps)        # (..., K)
        return safe_kl_clamp(terms.sum(dim=-1), kl_max=kl_max)

    def renyi_per_coord(
        self,
        other:   "DiagonalLaplace",

        *,
        alpha:   float = 1.0,
        kl_max:  float = 100.0,
        eps:     float = 1e-6,
    ) -> torch.Tensor:
        r"""Per-coordinate factorized-Laplace Renyi/KL: the unsummed (..., K) terms, each clamped
        independently. The factorized divergence decomposes coordinate-wise, so summing over k
        recovers ``renyi_closed_form`` when no coordinate saturates ``kl_max`` (the independent
        per-coordinate clamp is intended; only the recovery identity is conditional, as for the
        diagonal Gaussian)."""
        terms = self._renyi_terms(other, alpha=alpha, eps=eps)        # (..., K)
        return safe_kl_clamp(terms, kl_max=kl_max)
