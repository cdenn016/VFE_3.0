r"""Normalization registry for VFE_3.0 belief means.

The gauge-pure options are ``none`` (identity, trivially equivariant) and ``mahalanobis``.
MahalanobisNorm rescales mu by the gauge-invariant Mahalanobis length:
    mu_norm = mu * sqrt(K / (mu^T Sigma^-1 mu + eps)).
Since mu^T Sigma^-1 mu is invariant under mu->g mu, Sigma->g Sigma g^T, the scale is
gauge-invariant and mu_norm transforms as a vector. Pure math, no parameters.

``layernorm`` (the ``norm_type_block`` default) is a NON-gauge-equivariant baseline: standard transformer LayerNorm
standardization of the mean over the belief dimension (see LayerNorm). With ``layernorm_affine=True``
the builder returns the AffineLayerNorm variant, which adds a learned per-feature affine (gamma/beta)
on top of the standardization -- a SANCTIONED learned-scalar exception, still non-equivariant. Both are ablations / baselines against the gauge-pure paths, which
remain ``none``/``mahalanobis``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vfe3.families.base import get_family
from vfe3.numerics import safe_cholesky

_NORMS: Dict[str, Callable] = {}
_NORM_GAUGE_EQUIVARIANCE: Dict[str, tuple[Callable, bool]] = {}


@dataclass(frozen=True)
class NormRegistration:
    """One norm builder and its full-family GL(K)-equivariance declaration."""

    callable: Callable
    gauge_equivariant: bool


def register_norm(
    name: str, *, gauge_equivariant: Optional[bool] = None, override: bool = False,
) -> Callable:
    """Decorator registering a norm builder under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _NORMS and not override:
            raise KeyError(f"norm {name!r} already registered; pass override=True to replace")
        resolved = (
            bool(getattr(fn, "_vfe3_gauge_equivariant", False))
            if gauge_equivariant is None else bool(gauge_equivariant)
        )
        setattr(fn, "_vfe3_gauge_equivariant", resolved)
        _NORMS[name] = fn
        _NORM_GAUGE_EQUIVARIANCE[name] = (fn, resolved)
        return fn
    return _wrap


def get_norm(name: str) -> Callable:
    """Return the registered norm builder (KeyError if absent)."""
    if name not in _NORMS:
        raise KeyError(f"no norm registered under {name!r}; available: {sorted(_NORMS)}")
    return _NORMS[name]


def get_norm_registration(name: str) -> NormRegistration:
    """Return the active builder plus fail-closed gauge metadata."""
    fn = get_norm(name)
    record = _NORM_GAUGE_EQUIVARIANCE.get(name)
    if record is not None and record[0] is fn:
        _, gauge_equivariant = record
    else:
        gauge_equivariant = bool(getattr(fn, "_vfe3_gauge_equivariant", False))
    return NormRegistration(fn, gauge_equivariant)


class MahalanobisNorm:
    r"""mu_norm = mu * sqrt(K / (mu^T Sigma^-1 mu + eps)).

    The Mahalanobis scalar ``s2 = mu^T Sigma^-1 mu`` is gauge-invariant under the FULL-covariance
    congruence: with mu->g mu, Sigma->g Sigma g^T it maps to
    ``mu^T g^T (g Sigma g^T)^-1 g mu = mu^T Sigma^-1 mu``. The exact SPD solve is evaluated in a
    float64 island, so the scale ``sqrt(K/s2)`` is invariant up to storage precision and ``mu_norm``
    transforms as a vector. If exact factorization fails, an explicitly approximate regularized
    Cholesky/pseudo-inverse fallback keeps the forward finite; that fallback is not gauge-invariant
    because an isotropic ridge does not transform by congruence. The DIAGONAL branch
    (``sum(mu^2 / sigma)``) is the Mahalanobis form only for a diagonal Sigma: it is invariant under
    the diagonal-scaling subgroup, NOT a general non-diagonal g in GL(K) -- consistent with the
    gaussian_diagonal family being declared non-GL(K)-invariant (groups.check_admissible). Pure math,
    no parameters.
    """

    def __init__(
        self,
        K: int,

        *,
        family: str   = "gaussian_diagonal",
        eps:    float = 1e-6,
    ) -> None:
        self.K = K
        self.eps = eps
        self.family = get_family(family)

    def __call__(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariances
    ) -> torch.Tensor:                   # (..., K) rescaled means
        r"""Rescale ``mu`` by the gauge-invariant Mahalanobis length."""
        if sigma.dim() == mu.dim() + 1:        # full covariance (..., K, K)
            # The pure path solves the unmodified SPD covariance in float64. Adding eps*I
            # unconditionally would change the mathematical object and break exact GL(K)
            # covariance even in infinite precision. Only elements whose exact Cholesky fails
            # enter the explicitly approximate jitter/pseudo-inverse fallback.
            sigma64 = 0.5 * (sigma.double() + sigma.double().transpose(-1, -2))
            mu64 = mu.double()
            factor, ok = safe_cholesky(sigma64, rounds=0)
            exact = torch.cholesky_solve(mu64.unsqueeze(-1), factor).squeeze(-1)
            if bool(ok.all()):
                sig_inv_mu = exact
            else:
                factor_reg, ok_reg = safe_cholesky(sigma64, eps=self.eps, rounds=5)
                regularized = torch.cholesky_solve(
                    mu64.unsqueeze(-1),
                    factor_reg,
                ).squeeze(-1)
                eye = torch.eye(self.K, device=sigma.device, dtype=torch.float64)
                pseudo = (
                    torch.linalg.pinv(sigma64 + self.eps * eye)
                    @ mu64.unsqueeze(-1)
                ).squeeze(-1)
                fallback = torch.where(ok_reg.unsqueeze(-1), regularized, pseudo)
                sig_inv_mu = torch.where(ok.unsqueeze(-1), exact, fallback)
            s2 = (mu64 * sig_inv_mu).sum(dim=-1, keepdim=True).to(mu.dtype)
        else:                                   # diagonal variances (..., K)
            quadratic = self.family.mean_fisher_quadratic(mu, sigma, eps=self.eps)
            s2 = quadratic.sum(dim=-1, keepdim=True)
        return mu * torch.sqrt(self.K / s2.clamp(min=self.eps))


def _ln_standardize(
    mu:  torch.Tensor,               # (..., K) means
    eps: float,                      # variance floor added inside the sqrt
) -> torch.Tensor:                   # (..., K) standardized means
    r"""LayerNorm standardization over the last dim: ``(mu - E[mu]) / sqrt(Var[mu] + eps)``.

    Biased variance (divide by K, not K-1) with eps added inside the sqrt, matching
    ``torch.nn.functional.layer_norm(mu, (K,), weight=None, bias=None, eps)``. Shared by the
    parameter-free ``LayerNorm`` and the learned-affine ``AffineLayerNorm``.
    Float16/bfloat16 inputs are promoted to float32, including the output; float64 is preserved.
    """
    # Keep centering, variance, and normalization in fp32 for low-precision inputs.
    # Retain fp32 output so the subsequent affine does not amplify rounding error.
    with torch.autocast(device_type=mu.device.type, enabled=False):
        work = mu.float() if mu.dtype in (torch.float16, torch.bfloat16) else mu
        return F.layer_norm(work, (mu.shape[-1],), eps=eps)


class LayerNorm:
    r"""Standard (parameter-free) LayerNorm over the belief mean.

    ``mu_norm = (mu - mean(mu)) / sqrt(var(mu) + eps)``, the mean and (biased) variance taken over
    the last (belief) dimension -- the conventional transformer LayerNorm standardization (Ba,
    Kiros and Hinton 2016) applied to the belief MEAN mu. ``sigma`` is ignored (as in the ``none``
    norm): LayerNorm acts on the point mu, not the covariance, and never inspects ``sigma``'s shape
    (diagonal or full-covariance sigma pass through untouched). The standardization matches
    ``torch.nn.LayerNorm(elementwise_affine=False)`` -- biased variance (divide by K, not K-1) with
    eps added inside the sqrt; the default ``eps`` (1e-6) follows the sibling MahalanobisNorm rather
    than torch's 1e-5 default (eps changes the output only where the feature variance is ~0).

    NON-gauge-equivariant by construction: the mean-subtraction and per-coordinate variance are
    taken in the FIXED coordinate basis, so ``LN(g mu) != g LN(mu)`` for a general g in GL(K)
    (contrast MahalanobisNorm's gauge-invariant scale). This is an OPT-IN ablation / baseline norm;
    the gauge-pure options remain ``none`` (identity) and ``mahalanobis`` (gauge-invariant scale).
    Parameter-free -- no learned affine (gamma/beta); the AffineLayerNorm sibling adds the affine.
    """

    def __init__(
        self,
        K:   int,

        *,
        eps: float = 1e-6,
    ) -> None:
        self.K = K
        self.eps = eps

    def __call__(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) or (..., K, K); ignored (LN acts on mu)
    ) -> torch.Tensor:                   # (..., K) standardized means
        r"""Standardize ``mu`` over the last dim: ``(mu - E[mu]) / sqrt(Var[mu] + eps)``."""
        return _ln_standardize(mu, self.eps)


class AffineLayerNorm(nn.Module):
    r"""Standard LayerNorm with a learned per-feature affine: ``mu_norm = gamma * LN(mu) + beta``.

    Applies the same standardization as ``LayerNorm`` (see :func:`_ln_standardize`), then a learned
    per-feature scale ``gamma`` (``weight``) and shift ``beta`` (``bias``), each of shape ``(K,)``,
    exactly ``torch.nn.LayerNorm(K, elementwise_affine=True)`` on the belief MEAN (``sigma`` ignored).
    ``gamma`` inits to ones and ``beta`` to zeros, so at construction the affine is the identity and
    the output is BYTE-IDENTICAL to the parameter-free ``LayerNorm``.

    SANCTIONED LEARNED-SCALAR EXCEPTION (default OFF via ``layernorm_affine``): ``gamma``/``beta`` are raw ``nn.Parameter`` diagonal scale/shift tables,
    not an ``nn.Linear`` / MLP / activation, so they carry no hidden network. They are still
    NON-gauge-equivariant -- a per-coordinate affine in the fixed basis does not commute with a
    general g in GL(K); the affine sits on the same non-gauge-pure path LayerNorm already occupies,
    adding a diagonal scale/shift on top of its centering (an opt-in baseline, not a gauge-pure path;
    those remain ``none``/``mahalanobis``). As an ``nn.Module`` its parameters
    register on the owning model and must be placed in an M-step optimizer group (train.build_optimizer
    groups them at ``m_p_mu_lr``, ``weight_decay=0``, ``role='mu'``). When used as the BLOCK norm the
    affine is applied to the belief VALUE inside the stack, so (unlike learnable_kappa, which enters
    only the E-step tangent) it trains under ``'unroll'`` AND ``'straight_through'`` and is frozen
    ONLY by the fully-detached E-step (effective ``'detach'``, which no_grads the whole stack;
    model.__init__ warns). As the FINAL norm it is post-stack and trains under any estimator.
    """

    def __init__(
        self,
        K:   int,

        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.K = K
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(K))                    # (K,) gamma, init 1 -> identity affine
        self.bias   = nn.Parameter(torch.zeros(K))                   # (K,) beta,  init 0

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) or (..., K, K); ignored (LN acts on mu)
    ) -> torch.Tensor:                   # (..., K) standardized + affine means
        r"""``gamma * LN(mu) + beta`` (standardize over the last dim, then per-feature affine)."""
        return _ln_standardize(mu, self.eps) * self.weight + self.bias


@register_norm("none", gauge_equivariant=True)
def _norm_none(K: int, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Identity norm (no rescaling)."""
    def _identity(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu
    return _identity


@register_norm("mahalanobis", gauge_equivariant=True)
def _norm_mahalanobis(
    K: int,

    *,
    family: str   = "gaussian_diagonal",
    eps:    float = 1e-6,
    **kwargs,
) -> MahalanobisNorm:
    """MahalanobisNorm builder."""
    return MahalanobisNorm(K, family=family, eps=eps)


@register_norm("layernorm")
def _norm_layernorm(K: int, *, eps: float = 1e-6, affine: bool = False, **kwargs):
    """Standard LayerNorm builder (opt-in, non-gauge-equivariant baseline).

    ``affine=False`` (default) -> parameter-free ``LayerNorm``; ``affine=True`` (config
    ``layernorm_affine``) -> the learned per-feature ``AffineLayerNorm``.
    """
    return AffineLayerNorm(K, eps=eps) if affine else LayerNorm(K, eps=eps)
