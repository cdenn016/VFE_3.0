r"""Normalization registry for VFE_3.0 belief means.

The gauge-pure options are ``none`` (identity, trivially equivariant) and ``mahalanobis``.
MahalanobisNorm rescales mu by the gauge-invariant Mahalanobis length:
    mu_norm = mu * sqrt(K / max(mu^T Sigma^-1 mu, eps)),
the quadratic form clamped below at eps (not shifted by it). Since mu^T Sigma^-1 mu is invariant under
mu->g mu, Sigma->g Sigma g^T, the scale is gauge-invariant and mu_norm transforms as a vector. Pure math,
no parameters.

``layernorm`` (the ``norm_type_block`` default) is NON-gauge-equivariant: standard transformer LayerNorm
standardization of the mean over the belief dimension (see LayerNorm). With ``layernorm_affine=True`` (the
default) the builder returns the AffineLayerNorm variant, which adds a learned per-feature affine (gamma/beta)
on top of the standardization, a SANCTIONED learned-scalar exception that is still non-equivariant. The gauge-pure
paths remain ``none``/``mahalanobis``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

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
    r"""mu_norm = mu * sqrt(K / max(mu^T Sigma^-1 mu, eps)).

    The Mahalanobis scalar ``s2 = mu^T Sigma^-1 mu`` is gauge-invariant under the FULL-covariance
    congruence: with mu->g mu, Sigma->g Sigma g^T it maps to
    ``mu^T g^T (g Sigma g^T)^-1 g mu = mu^T Sigma^-1 mu``. The exact SPD solve is evaluated in a
    float64 island, so the scale ``sqrt(K/s2)`` is invariant up to storage precision and ``mu_norm``
    transforms as a vector. ``s2`` is clamped below at ``eps`` (not shifted by it), so for ``s2 >= eps`` the output has
    Mahalanobis length ``sqrt(K)``, and below ``eps`` the map is linear, ``mu * sqrt(K / eps)``. If exact
    factorization fails, an explicitly approximate regularized
    Cholesky/pseudo-inverse fallback keeps the forward finite; that fallback is not gauge-invariant
    because an isotropic ridge does not transform by congruence. The DIAGONAL branch
    (``sum(mu^2 / sigma)``) is the Mahalanobis form only for a diagonal Sigma: it is invariant under
    the diagonal-scaling subgroup, NOT a general non-diagonal g in GL(K) -- consistent with the
    gaussian_diagonal family being declared non-GL(K)-invariant (groups.check_admissible). Pure math,
    no parameters.

    ``head_dims`` (equal head widths; the head-block family ``gaussian_block``, whose covariance is
    :math:`\bigoplus_h\Sigma_{hh}` with zero off-head blocks) evaluates
    :math:`s^2 = \sum_h\mu_h^\top\Sigma_{hh}^{-1}\mu_h`, one :math:`d\times d` solve per head instead of the
    :math:`K\times K` one, the same scalar on that family (audit 2026-10-31 A185). Only the head blocks are read, so
    the derivative of :math:`s^2` on the off-head blocks is zero, where the dense solve gives
    :math:`\partial s^2/\partial\Sigma_{hg} = -x_hx_g^\top` (:math:`x = \Sigma^{-1}\mu`); the two agree on the family's
    tangent directions, since every producer of the family holds those blocks at zero. A head whose factor fails
    sends the call to the dense solve and its fallback, so the fallback is the one without ``head_dims``. The exact
    path reads one host flag per call.
    """

    def __init__(
        self,
        K: int,

        *,
        family:    str                     = "gaussian_diagonal",
        eps:       float                   = 1e-6,
        head_dims: Optional[Sequence[int]] = None,    # equal head widths of a head-block covariance, or None
    ) -> None:
        self.K = K
        self.eps = eps
        self.family = get_family(family)
        if head_dims is not None:
            head_dims = tuple(int(dim) for dim in head_dims)
            if sum(head_dims) != K or len(set(head_dims)) != 1:
                raise ValueError(f"MahalanobisNorm head_dims must be equal head widths summing to K={K}, "
                                 f"got {head_dims}")
        self.head_dims = head_dims

    def _solve(
        self,
        sigma64: torch.Tensor,           # (..., n, n) symmetric float64
        mu64:    torch.Tensor,           # (..., n) float64
    ) -> torch.Tensor:                   # (..., n) Sigma^-1 mu
        r"""``Sigma^-1 mu`` by the unmodified Cholesky factor, with the approximate fallback on failed elements only."""
        # The pure path solves the unmodified SPD covariance. Adding eps*I unconditionally would change the
        # mathematical object and break exact GL(K) covariance even in infinite precision. Only elements whose
        # exact Cholesky fails enter the explicitly approximate jitter/pseudo-inverse fallback.
        factor, info = torch.linalg.cholesky_ex(sigma64)
        ok = (info == 0) & torch.isfinite(factor).all(dim=(-2, -1))
        if bool(ok.all()):                                          # the one host read of the exact path
            return torch.cholesky_solve(mu64.unsqueeze(-1), factor).squeeze(-1)
        factor, ok = safe_cholesky(sigma64, rounds=0)               # identity factor on the failed elements
        exact = torch.cholesky_solve(mu64.unsqueeze(-1), factor).squeeze(-1)
        factor_reg, ok_reg = safe_cholesky(sigma64, eps=self.eps, rounds=5)
        regularized = torch.cholesky_solve(
            mu64.unsqueeze(-1),
            factor_reg,
        ).squeeze(-1)
        eye = torch.eye(sigma64.shape[-1], device=sigma64.device, dtype=torch.float64)
        pseudo = (
            torch.linalg.pinv(sigma64 + self.eps * eye)
            @ mu64.unsqueeze(-1)
        ).squeeze(-1)
        fallback = torch.where(ok_reg.unsqueeze(-1), regularized, pseudo)
        return torch.where(ok.unsqueeze(-1), exact, fallback)

    def __call__(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariances
    ) -> torch.Tensor:                   # (..., K) rescaled means
        r"""Rescale ``mu`` by the gauge-invariant Mahalanobis length."""
        if sigma.dim() == mu.dim() + 1:        # full covariance (..., K, K)
            mu64 = mu.double()
            if self.head_dims is None:
                sigma64 = 0.5 * (sigma.double() + sigma.double().transpose(-1, -2))   # (..., K, K)
                sig_inv_mu = self._solve(sigma64, mu64)                                # (..., K)
            else:
                H, d = len(self.head_dims), self.head_dims[0]
                blocks = torch.stack([sigma[..., h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(H)],
                                     dim=-3).double()                                  # (..., H, d, d)
                blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
                factor, info = torch.linalg.cholesky_ex(blocks)                        # (..., H, d, d)
                if bool(((info == 0) & torch.isfinite(factor).all(dim=(-2, -1))).all()):   # the one host read
                    rhs = mu64.reshape(*mu64.shape[:-1], H, d, 1)                      # (..., H, d, 1)
                    sig_inv_mu = torch.cholesky_solve(rhs, factor).reshape(mu64.shape)
                else:                         # a failed head: the dense solve and its fallback, as without head_dims
                    sigma64 = 0.5 * (sigma.double() + sigma.double().transpose(-1, -2))
                    sig_inv_mu = self._solve(sigma64, mu64)
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
    (contrast MahalanobisNorm's gauge-invariant scale). ``layernorm`` is the ``norm_type_block`` default
    (with ``layernorm_affine=True``, the default, the block norm is the AffineLayerNorm sibling); the gauge-pure
    options remain ``none`` (identity) and ``mahalanobis`` (gauge-invariant scale).
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

    SANCTIONED LEARNED-SCALAR EXCEPTION (on by default, ``layernorm_affine=True``): ``gamma``/``beta`` are raw
    ``nn.Parameter`` diagonal scale/shift tables, not an ``nn.Linear`` / MLP / activation, so they carry no hidden
    network. They are still NON-gauge-equivariant: a per-coordinate affine in the fixed basis does not commute with
    a general g in GL(K); the affine sits on the same non-gauge-pure path LayerNorm already occupies, adding a diagonal
    scale/shift on top of its centering (the gauge-pure norms remain ``none``/``mahalanobis``). As an ``nn.Module`` its
    parameters register on the owning model, in an M-step optimizer group of their own (``train.build_optimizer``:
    rate ``m_layernorm_lr``, which falls back to ``m_p_mu_lr`` when None, ``weight_decay=0``, ``role='mu'``). One
    instance is the block norm of every layer, applied to the belief mean inside each block; it trains through the
    unrolled stack like the rest of the block. No norm runs between the stack and the decoder.

    ``fused`` (``layernorm_fused``, default False) evaluates ``F.layer_norm(mu, (K,), gamma, beta, eps)`` in one call,
    one forward kernel and one fused backward in place of the separate multiply and add and their gamma and beta
    reductions, after the same float32 promotion. It agrees with the default standardize-then-affine form to float32
    rounding; its gradients are not bitwise equal to the default's (the CPU tests found the forward bitwise equal and
    the gradients different in their last bits), which is why it is opt-in (audit 2026-10-31 A150).
    """

    def __init__(
        self,
        K:     int,

        *,
        eps:   float = 1e-6,
        fused: bool  = False,                                        # layernorm_fused: one F.layer_norm call
    ) -> None:
        super().__init__()
        self.K = K
        self.eps = eps
        self.fused = bool(fused)
        self.weight = nn.Parameter(torch.ones(K))                    # (K,) gamma, init 1 -> identity affine
        self.bias   = nn.Parameter(torch.zeros(K))                   # (K,) beta,  init 0

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) means
        sigma: torch.Tensor,             # (..., K) or (..., K, K); ignored (LN acts on mu)
    ) -> torch.Tensor:                   # (..., K) standardized + affine means
        r"""``gamma * LN(mu) + beta`` (standardize over the last dim, then per-feature affine)."""
        if self.fused:
            with torch.autocast(device_type=mu.device.type, enabled=False):
                work = mu.float() if mu.dtype in (torch.float16, torch.bfloat16) else mu
                dtype = torch.promote_types(work.dtype, self.weight.dtype)          # the unfused result's dtype
                return F.layer_norm(work.to(dtype), (mu.shape[-1],), self.weight.to(dtype), self.bias.to(dtype),
                                    eps=self.eps)
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
    family:    str                     = "gaussian_diagonal",
    eps:       float                   = 1e-6,
    head_dims: Optional[Sequence[int]] = None,    # head-block family: solve per head
    **kwargs,
) -> MahalanobisNorm:
    """MahalanobisNorm builder."""
    return MahalanobisNorm(K, family=family, eps=eps, head_dims=head_dims)


@register_norm("layernorm")
def _norm_layernorm(K: int, *, eps: float = 1e-6, affine: bool = False, fused: bool = False, **kwargs):
    """Standard LayerNorm builder (the ``norm_type_block`` default; not gauge-equivariant).

    ``affine=False`` -> parameter-free ``LayerNorm``; ``affine=True`` (config ``layernorm_affine``, the
    default) -> the learned per-feature ``AffineLayerNorm``, as one fused ``F.layer_norm`` call under
    ``fused=True`` (config ``layernorm_fused``).
    """
    return AffineLayerNorm(K, eps=eps, fused=fused) if affine else LayerNorm(K, eps=eps)
