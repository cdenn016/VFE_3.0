import math
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch
from torch import nn

from vfe3.contracts import BlockMLPBuildMetadata, CanonicalFrameContext
from vfe3.model.prototype_agents import (
    PROTOTYPE_COVARIANCE_CONTRACTS,
    PROTOTYPE_INFERENCE_MODES,
    PROTOTYPE_TILE_BYTES,
    ExactPrototypePosterior,
    MeanFieldPrototypePosterior,
    covariance_diagonal,
    covariance_layout,
    prototype_elbo,
    prototype_exact_em_objective,
    prototype_exact_posterior,
    prototype_mean_field_posterior,
    template_kl_logits,
)
from vfe3.model.cue_result_memory import CUE_RESULT_TILE_BYTES, CueResultPosterior, cue_result_exact_posterior
from vfe3.numerics import validated_cholesky_solve
from vfe3.runtime import ieee_fp32_matmul


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


_BLOCK_MLP_COVARIANCE_CONTRACTS = frozenset({"passthrough", "delta_full"})
# Tensor layout of the belief covariance the block reads: (..., K, K) or its diagonal (..., K).
_COVARIANCE_LAYOUTS = frozenset({"full", "diagonal"})


def _validate_block_mlp_covariance_contract(covariance_contract: str) -> str:
    if covariance_contract not in _BLOCK_MLP_COVARIANCE_CONTRACTS:
        raise ValueError(f"unknown BlockMLP covariance contract {covariance_contract!r}")
    return covariance_contract


def _validate_covariance_layout(covariance_layout: str) -> str:
    if covariance_layout not in _COVARIANCE_LAYOUTS:
        raise ValueError(
            f"covariance_layout must be one of {sorted(_COVARIANCE_LAYOUTS)}, "
            f"got {covariance_layout!r}")
    return covariance_layout


def _require_canonical_frame_context(
    frame_context: Optional[CanonicalFrameContext],
) -> CanonicalFrameContext:
    if frame_context is None:
        raise ValueError("canonical_frame block MLP requires a realized canonical frame")
    if not isinstance(frame_context, CanonicalFrameContext):
        raise TypeError(
            "canonical_frame block MLP frame_context must be a CanonicalFrameContext")
    return frame_context


def _escalated_frame(frame: CanonicalFrameContext, mu: torch.Tensor) -> bool:
    r"""Whether a float64 frame (escalated pair, audit 2026-09-23 #12) acts on lower-precision beliefs."""
    return frame.forward.dtype == torch.float64 and mu.dtype != torch.float64


def _frame_action(
    frame: CanonicalFrameContext,
    mu:    torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""The map ``(M, x) -> M x`` used for this frame's vertex factors.

    An escalated frame applies each factor at float64 and returns ``x.dtype``, so the MLP and every
    downstream tensor keep their dtypes. Any other frame contracts with autocast disabled at
    ``promote_types(x.dtype, float32)``: an ambient bf16 autocast would otherwise store the belief
    mean in bf16 (the MLP's own residual then adds its bf16 increment to a float32 mean). The
    contraction is pinned to IEEE fp32 (:func:`vfe3.runtime.ieee_fp32_matmul`), as
    :func:`vfe3.model.frame_actions.frame_matvec` is: the round trip U F(U^{-1} mu) amplifies
    TF32 rounding by about kappa(U).
    """
    if _escalated_frame(frame, mu):
        return lambda M, x: torch.einsum("...ij,...j->...i", M, x.double()).to(x.dtype)

    def apply(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(torch.promote_types(x.dtype, M.dtype), torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False), ieee_fp32_matmul():
            return torch.einsum("...ij,...j->...i", M.to(dtype), x.to(dtype))
    return apply



def block_mlp_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown block MLP activation {name!r}") from exc


def _activation_derivative(name: str, value: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return (value > 0).to(dtype=value.dtype)
    if name == "silu":
        sigmoid = torch.sigmoid(value)
        return sigmoid * (1.0 + value * (1.0 - sigmoid))
    if name == "gelu":
        inv_sqrt_two = value.new_tensor(2.0).rsqrt()
        inv_sqrt_two_pi = value.new_tensor(2.0 * torch.pi).rsqrt()
        return 0.5 * (1.0 + torch.erf(value * inv_sqrt_two)) \
            + value * torch.exp(-0.5 * value.square()) * inv_sqrt_two_pi
    raise RuntimeError(f"unsupported block MLP activation {name!r}")


def _dropout_scale(dropout: nn.Dropout, reference: torch.Tensor) -> torch.Tensor:
    if not dropout.training or dropout.p == 0.0:
        return torch.ones_like(reference)
    return torch.nn.functional.dropout(
        torch.ones_like(reference), p=dropout.p, training=True)


def _add_hidden_bias(hidden_pre: torch.Tensor, hidden_bias: torch.Tensor) -> torch.Tensor:
    r"""Add a per-token bias to the first ``w`` units of the fc1 pre-activation.

    .. math::
        h_{ij} = (W_1 u_i + b_1)_j + E_{x_i j}\,[j < w],

    with ``hidden_pre`` (..., H) and ``hidden_bias`` (..., w), identical leading axes and w <= H. The
    bias is cast to the pre-activation dtype, as fc1's own bias is under autocast, so the hidden path
    keeps the dtype of the unbiased path; its gradient still reaches the stored table at full precision.
    """
    hidden_width = hidden_pre.shape[-1]
    if hidden_bias.dim() != hidden_pre.dim() or hidden_bias.shape[:-1] != hidden_pre.shape[:-1]:
        raise ValueError(
            "hidden_bias must be (..., w) with the leading axes of the hidden pre-activation "
            f"{tuple(hidden_pre.shape[:-1])}, got {tuple(hidden_bias.shape)}")
    width = hidden_bias.shape[-1]
    if width > hidden_width:
        raise ValueError(
            f"hidden_bias width w={width} exceeds the fc1 hidden width {hidden_width}")
    bias = hidden_bias.to(hidden_pre.dtype)                               # (..., w)
    if width == hidden_width:
        return hidden_pre + bias
    # Units w..H-1 are passed through untouched (not added to zero), so they stay bit-identical.
    return torch.cat([hidden_pre[..., :width] + bias, hidden_pre[..., width:]], dim=-1)


def _reject_hidden_bias(hidden_bias: Optional[torch.Tensor], reason: str) -> None:
    if hidden_bias is not None:
        raise ValueError(f"hidden_bias does not apply: {reason}")


def _is_diagonal_layout(mu: torch.Tensor, sigma: torch.Tensor) -> bool:
    r"""Whether ``sigma`` is the diagonal layout (..., K) of Sigma rather than the full (..., K, K)."""
    return sigma.shape == mu.shape


class BlockMLPMomentResult(NamedTuple):
    """Post-MLP belief moments and, when requested, the mean-map Jacobian."""

    mu: torch.Tensor
    sigma: torch.Tensor
    jacobian: Optional[torch.Tensor]


def layernorm_jacobian(
    mu:     torch.Tensor,                             # (..., K) pre-norm means
    *,
    eps:    float,
    weight: Optional[torch.Tensor] = None,            # (K,) affine gamma, or None for the plain norm
) -> torch.Tensor:                                    # (..., K, K) d LN(mu) / d mu
    r"""Exact Jacobian of the block LayerNorm :math:`y = \gamma\odot z + b` at ``mu``.

    With :math:`\bar u = \tfrac1K\sum_k u_k`, the biased variance :math:`v = \tfrac1K\sum_k(u_k-\bar u)^2`,
    :math:`s = \sqrt{v+\epsilon}` and :math:`z = (u - \bar u\mathbf 1)/s` (the standardization of
    :func:`vfe3.geometry.norms._ln_standardize`, eps inside the square root),
    :math:`\partial s/\partial u_j = (u_j - \bar u)/(Ks)`, so

    .. math::
        J = \operatorname{diag}(\gamma)\,\frac{1}{s}\Bigl(I - \tfrac1K\mathbf 1\mathbf 1^\top - \tfrac1K zz^\top\Bigr),

    exact for every :math:`\epsilon` (:math:`z^\top z = Kv/(v+\epsilon) < K`). The shift :math:`b` does not enter.
    Evaluated with autocast off at ``promote_types(mu.dtype, float32)``, the precision the norm itself runs at.
    """
    dtype = torch.promote_types(mu.dtype, torch.float32)
    K = mu.shape[-1]
    with torch.autocast(device_type=mu.device.type, enabled=False):
        work = mu.to(dtype)
        centered = work - work.mean(dim=-1, keepdim=True)                         # (..., K)
        s = (centered.square().mean(dim=-1, keepdim=True) + eps).sqrt()         # (..., 1)
        z = centered / s                                                        # (..., K)
        eye = torch.eye(K, dtype=dtype, device=mu.device)
        J = (eye - 1.0 / K - z.unsqueeze(-1) * z.unsqueeze(-2) / K) / s.unsqueeze(-1)   # (..., K, K)
        if weight is not None:
            J = weight.to(dtype).unsqueeze(-1) * J                              # diag(gamma) J
    return J


def layernorm_scale(
    mu:     torch.Tensor,                             # (..., K) pre-norm means
    *,
    eps:    float,
    weight: Optional[torch.Tensor] = None,            # (K,) affine gamma, or None for the plain norm
) -> torch.Tensor:                                    # (..., K) gamma / s
    r"""The scale part :math:`D = \operatorname{diag}(\gamma)/s` of the LayerNorm Jacobian of :func:`layernorm_jacobian`.

    :math:`J_{\mathrm{LN}} = D(I - \tfrac1K\mathbf 1\mathbf 1^\top - \tfrac1K zz^\top)`; ``delta_full`` expresses its
    covariance floor in post-norm units through :math:`D` (:func:`_delta_covariance`). Same precision as the Jacobian.
    """
    dtype = torch.promote_types(mu.dtype, torch.float32)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        work = mu.to(dtype)
        centered = work - work.mean(dim=-1, keepdim=True)                         # (..., K)
        s = (centered.square().mean(dim=-1, keepdim=True) + eps).sqrt()         # (..., 1)
        scale = (1.0 / s).expand_as(work)                                       # (..., K)
        if weight is not None:
            scale = weight.to(dtype) * scale
    return scale


class LayerNormRadialTerm(NamedTuple):
    r"""The second-order covariance :math:`\rho\,gg^\top` of a LayerNorm output along its radial direction.

    ``direction`` is :math:`g = \gamma\odot z` at the norm, or its image :math:`J_{\mathrm{MLP}}g` once carried
    through a block MLP (:func:`_push_radial`); see :func:`layernorm_radial_term`.
    """

    rho:       torch.Tensor                           # (..., 1)
    direction: torch.Tensor                           # (..., K)


def _layernorm_radial_coefficient(
    centered: torch.Tensor,                           # (..., K) c = u - mean(u) at the working dtype
    s2:       torch.Tensor,                           # (..., 1) s^2 = v + eps
    sigma:    torch.Tensor,                           # (..., K) diagonal or (..., K, K) full input covariance
    eps:      float,
) -> torch.Tensor:                                    # (..., 1) rho = T / (2 K^2 s^4)
    r"""Coefficient :math:`\rho` of the radial second-order term, O(K) (diagonal) or O(K^2) (full) per token.

    With :math:`\hat z = c/\lVert c\rVert`, :math:`e = \mathbf 1/\sqrt K`, :math:`\kappa = 3\epsilon/s^2` and
    :math:`\tilde H = I - ee^\top - (1 - \kappa)\hat z\hat z^\top`, :math:`T = \operatorname{tr}(\tilde H\Sigma\tilde H\Sigma)`
    is expanded through :math:`W = [e, \hat z]`, :math:`\Lambda = \operatorname{diag}(1, 1 - \kappa)`:

    .. math::
        T = \lVert\Sigma\rVert_F^2 - 2\operatorname{tr}\bigl(\Lambda(\Sigma W)^\top\Sigma W\bigr)
            + \operatorname{tr}(\Lambda A\Lambda A),\qquad A = W^\top\Sigma W,

    and is clamped at zero (it is a sum of squares, :math:`\lVert\Sigma^{1/2}\tilde H\Sigma^{1/2}\rVert_F^2`, that the
    expansion can round below zero). A constant input (:math:`c = 0`) has :math:`g = 0`, so the term vanishes.
    """
    K = centered.shape[-1]
    zhat = centered / centered.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(centered.dtype).tiny)   # (..., K)
    lam = 1.0 - 3.0 * eps / s2                                                      # (..., 1) 1 - kappa
    root_k = math.sqrt(K)
    if sigma.dim() == centered.dim():                                               # diagonal sigma (..., K)
        frob = sigma.square().sum(dim=-1, keepdim=True)                             # ||Sigma||_F^2
        n0 = frob / K                                                               # ||Sigma e||^2
        n1 = (sigma.square() * zhat.square()).sum(dim=-1, keepdim=True)            # ||Sigma zhat||^2
        a00 = sigma.sum(dim=-1, keepdim=True) / K                                   # e^T Sigma e
        a01 = (sigma * zhat).sum(dim=-1, keepdim=True) / root_k                     # zhat^T Sigma e
        a11 = (sigma * zhat.square()).sum(dim=-1, keepdim=True)                     # zhat^T Sigma zhat
    else:                                                                           # full sigma (..., K, K)
        frob = sigma.square().sum(dim=(-2, -1)).unsqueeze(-1)
        Se = sigma.sum(dim=-1) / root_k                                             # (..., K) Sigma e
        Sz = torch.einsum("...kl,...l->...k", sigma, zhat)                          # (..., K) Sigma zhat
        n0 = Se.square().sum(dim=-1, keepdim=True)
        n1 = Sz.square().sum(dim=-1, keepdim=True)
        a00 = Se.sum(dim=-1, keepdim=True) / root_k
        a01 = (zhat * Se).sum(dim=-1, keepdim=True)
        a11 = (zhat * Sz).sum(dim=-1, keepdim=True)
    T = frob - 2.0 * (n0 + lam * n1) + a00.square() + 2.0 * lam * a01.square() + lam.square() * a11.square()
    return T.clamp_min(0.0) / (2.0 * K * K * s2.square())


def layernorm_radial_term(
    mu:     torch.Tensor,                             # (..., K) pre-norm means
    sigma:  torch.Tensor,                             # (..., K) diagonal or (..., K, K) full covariance at mu
    *,
    eps:    float,
    weight: Optional[torch.Tensor] = None,            # (K,) affine gamma, or None for the plain norm
) -> LayerNormRadialTerm:
    r"""Second-order variance of the block LayerNorm along its radial direction (audit 2026-10-10 round 2, #11).

    The Jacobian :math:`J_{\mathrm{LN}} = D(I - \tfrac1K\mathbf 1\mathbf 1^\top - \tfrac1Kzz^\top)` has eigenvalue
    :math:`\epsilon/(v + \epsilon)` on :math:`z`, so the first-order pushforward gives the token-dependent radial
    direction almost no variance. With :math:`c = u - \bar u\mathbf 1`, :math:`r_0 = \lVert c\rVert = \sqrt{Kv}` and
    :math:`\delta = \Pi(u - \mu) = \alpha\hat z + \delta_\perp`, the radial component of :math:`z` is
    :math:`f = \sqrt K(r_0 + \alpha)/\sqrt{(r_0 + \alpha)^2 + \lVert\delta_\perp\rVert^2 + K\epsilon}`; it depends on
    :math:`\delta_\perp` only through :math:`\lVert\delta_\perp\rVert^2`, so its Hessian on the centered subspace is
    :math:`f_{\perp\perp}(\tilde H)`, :math:`\tilde H = I - ee^\top - (1 - \kappa)\hat z\hat z^\top`, with
    :math:`f_{\perp\perp} = -\sqrt v/(\sqrt Ks^3)` and :math:`\kappa = 3\epsilon/s^2` (the ratio
    :math:`f_{\alpha\alpha}/f_{\perp\perp}`). For :math:`u \sim \mathcal N(\mu, \Sigma)` the second-order term
    :math:`\tfrac12\delta^\top H\delta` is uncorrelated with every first-order term (odd Gaussian moments), and

    .. math::
        q = \operatorname{Var}\bigl(\tfrac12\delta^\top H\delta\bigr) = \tfrac12\operatorname{tr}(H\Sigma H\Sigma)
          = \frac{v\,T}{2Ks^6},\qquad T = \operatorname{tr}(\tilde H\Sigma\tilde H\Sigma),

    which at :math:`\epsilon = 0` is :math:`\operatorname{tr}(\Sigma_\perp^2)/(2Kv^2)`. The output covariance gains
    :math:`q(\gamma\odot\hat z)(\gamma\odot\hat z)^\top = \rho\,gg^\top` with :math:`g = \gamma\odot z` and, since
    :math:`\lVert z\rVert^2 = Kv/s^2`, :math:`\rho = T/(2K^2s^4)` (:func:`_layernorm_radial_coefficient`). It is the
    leading variance along :math:`\hat z`, where the first-order term is :math:`O(\epsilon/v)`; second-order terms in
    the other directions are a relative :math:`O(\Sigma/v)` correction to a nonzero first-order variance and are not
    added. Evaluated with autocast off and IEEE fp32 GEMMs at ``promote_types(mu.dtype, sigma.dtype, float32)``.
    """
    dtype = _pushforward_dtype(mu, sigma)
    with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
        work = mu.to(dtype)
        centered = work - work.mean(dim=-1, keepdim=True)                         # (..., K)
        s2 = centered.square().mean(dim=-1, keepdim=True) + eps                   # (..., 1)
        rho = _layernorm_radial_coefficient(centered, s2, sigma.to(dtype), eps)    # (..., 1)
        g = centered / s2.sqrt()                                                  # (..., K) z
        if weight is not None:
            g = weight.to(dtype) * g                                              # gamma o z
    return LayerNormRadialTerm(rho, g)


def block_norm_radial_term(
    norm:  Callable[..., torch.Tensor],
    mu:    torch.Tensor,                              # (..., K) pre-norm means
    sigma: torch.Tensor,                              # (..., K) or (..., K, K) pre-norm covariance
) -> LayerNormRadialTerm:
    r"""The radial second-order term (:func:`layernorm_radial_term`) of a block LayerNorm seam at ``(mu, sigma)``."""
    eps, weight = _layernorm_arguments(norm)
    return layernorm_radial_term(mu, sigma, eps=eps, weight=weight)


def layernorm_diagonal_pushforward(
    mu:               torch.Tensor,                   # (..., K) pre-norm means
    sigma:            torch.Tensor,                   # (..., K) diagonal covariance at mu, sigma_k > 0
    *,
    eps:              float,
    covariance_floor: float,
    weight:           Optional[torch.Tensor] = None,  # (K,) affine gamma, or None for the plain norm
    effective_scale:  bool                   = False, # block_norm_covariance='pushforward_moment': at s_eff
) -> torch.Tensor:                                    # (..., K)
    r"""Diagonal of the LayerNorm pushforward :math:`J\operatorname{diag}(\sigma)J^\top + fD\operatorname{diag}(\sigma)D + \rho gg^\top`, O(K) per token.

    With :math:`J_{kl} = (\gamma_k/s)\bigl(\delta_{kl} - (1 + z_kz_l)/K\bigr)` (:func:`layernorm_jacobian`, eps inside
    :math:`s`), :math:`D = \operatorname{diag}(\gamma)/s` and the moments :math:`S_m = \sum_l\sigma_lz_l^m`,

    .. math::
        \bigl[J\operatorname{diag}(\sigma)J^\top\bigr]_{kk} = \Bigl(\frac{\gamma_k}{s}\Bigr)^2\Bigl[
            \sigma_k\Bigl(1 - \frac{2(1 + z_k^2)}{K}\Bigr) + \frac{S_0 + 2z_kS_1 + z_k^2S_2}{K^2}\Bigr],

    so the result is :math:`(\gamma_k/s)^2\bigl([\cdot] + f\sigma_k\bigr) + \rho\gamma_k^2z_k^2` without forming :math:`J`,
    the last term the radial second-order variance (:func:`layernorm_radial_term`). It is the diagonal of the dense
    delta-method pushforward, that Gaussian's M-projection onto the diagonal family. The delta method itself
    approximates the LayerNorm output's moments only while :math:`\Sigma` is small against :math:`s^2`
    (:func:`layernorm_trace_cap`). The bracket equals the sum of
    squares :math:`\sum_l(\delta_{kl} - (1 + z_kz_l)/K)^2\sigma_l` but is evaluated with signed terms, so where the norm
    saturates (row :math:`k` of :math:`J` near zero) its rounding is of order :math:`u\max_l\sigma_l(\gamma_k/s)^2`; the
    floor keeps the result positive unless that exceeds :math:`f\sigma_k(\gamma_k/s)^2`. Below float64 the float64 result
    is also formed and selected where the working value is not positive: an O(K) elementwise pass with no host read
    of the failure mask (audit 2026-10-10 round 2, item 57). The float64 result carries no autograd graph; the
    gradient at every entry is that of the working-precision closed form (:class:`_SelectWorkingGradient`, audit
    2026-10-16 F89). Evaluated with autocast off at
    ``promote_types(mu.dtype, sigma.dtype, float32)``, as :func:`_delta_covariance` is.

    ``effective_scale`` (``block_norm_covariance='pushforward_moment'``, audit 2026-10-16 F3) evaluates the same
    expression at the effective scale of :func:`_layernorm_effective_scale`, a moment estimate valid at every
    :math:`\Sigma/s^2`; it is the diagonal of the full layout's (:func:`layernorm_full_pushforward`) at
    :math:`\operatorname{diag}(\sigma)`.
    """
    return _layernorm_diagonal_pushforward_pair(mu, sigma, eps=eps, covariance_floor=covariance_floor, weight=weight,
                                                effective_scale=effective_scale)[0]


def _layernorm_diagonal_pushforward_pair(
    mu:               torch.Tensor,                   # (..., K) pre-norm means
    sigma:            torch.Tensor,                   # (..., K) diagonal covariance at mu
    *,
    eps:              float,
    covariance_floor: float,
    weight:           Optional[torch.Tensor] = None,
    effective_scale:  bool                   = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:     # (working result, its float64 evaluation without graph or None)
    r""":func:`layernorm_diagonal_pushforward` and, below float64, the float64 evaluation it selects from (no graph),
    which the trace cap reads on tokens whose working-precision result overflowed (audit 2026-10-17 item 32)."""
    dtype = _pushforward_dtype(mu, sigma)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        pushed = _layernorm_diagonal_pushforward_at(
            mu.to(dtype), sigma.to(dtype), eps, covariance_floor, None if weight is None else weight.to(dtype),
            effective_scale)
        if dtype == torch.float64:
            return pushed, None
        with torch.no_grad():                    # no float64 graph: the select below routes no gradient here
            high = _layernorm_diagonal_pushforward_at(
                mu.double(), sigma.double(), eps, covariance_floor, None if weight is None else weight.double(),
                effective_scale)
        return _SelectWorkingGradient.apply(pushed, high.to(dtype), pushed <= 0), high


class _SelectWorkingGradient(torch.autograd.Function):
    r"""``torch.where(selected, high, pushed)`` in value, with the gradient of ``pushed`` at every entry.

    The float64 rescue of :func:`layernorm_diagonal_pushforward` replaces the value of the working-precision closed
    form where it rounds to a nonpositive variance. Its derivative is that of the same closed form, so the gradient
    is taken from the working-precision graph everywhere and the float64 result is formed without one; before audit
    2026-10-16 F89 the ``torch.where`` kept the float64 graph alive on every call, about half the forward and
    backward ops of the closed form, although entries are selected only where the norm saturates.
    """

    @staticmethod
    def forward(ctx: Any, pushed: torch.Tensor, high: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        return torch.where(selected, high, pushed)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        return grad, None, None


def _layernorm_effective_scale(
    s2:    torch.Tensor,                              # (..., 1) s^2 = v + eps at the pre-norm mean
    noise: torch.Tensor,                              # (..., 1) T = tr(Pi Sigma Pi), the centered input variance
    K:     int,
) -> torch.Tensor:                                    # (..., 1) s_eff^2 = s^2 + T / K
    r"""The effective LayerNorm scale :math:`s_{\mathrm{eff}}^2 = s^2 + \operatorname{tr}(\Pi\Sigma\Pi)/K` of the
    moment pushforward (``block_norm_covariance='pushforward_moment'``, audit 2026-10-16 F3).

    With :math:`\Pi = I - \mathbf 1\mathbf 1^\top/K`, :math:`u \sim \mathcal N(\mu, \Sigma)` and
    :math:`c = \Pi u = c_\mu + \delta`, :math:`\delta \sim \mathcal N(0, \Pi\Sigma\Pi)`, the LayerNorm variance is

    .. math::
        v(u) + \epsilon = \bigl(\lVert c_\mu\rVert^2 + 2c_\mu^\top\delta + \lVert\delta\rVert^2\bigr)/K + \epsilon,
        \qquad E\lVert\delta\rVert^2 = T = \operatorname{tr}(\Pi\Sigma\Pi),

    so its mean is :math:`s_{\mathrm{eff}}^2`. Replacing :math:`\lVert\delta\rVert^2` by :math:`T` leaves the map
    :math:`\Gamma(c_\mu + \delta)/\sqrt{s_{\mathrm{eff}}^2 + 2c_\mu^\top\delta/K}`, whose only nonlinearity is the
    scalar :math:`2c_\mu^\top\delta/K`, of standard deviation at most :math:`s_{\mathrm{eff}}^2/\sqrt{K-1}` for
    :math:`\Sigma = \sigma I` (AM-GM on :math:`\lVert c_\mu\rVert^2 + (K-1)\sigma`) at every :math:`\Sigma/s^2`. Its delta
    method at :math:`\mu` is the LayerNorm's with :math:`\epsilon` replaced by :math:`\epsilon + T/K`: Jacobian
    :math:`D(I - \tfrac1K\mathbf 1\mathbf 1^\top - \tfrac1Kz z^\top)`, :math:`D = \operatorname{diag}(\gamma)/s_{\mathrm{eff}}`,
    :math:`z = c_\mu/s_{\mathrm{eff}}`. The fluctuation of :math:`\lVert\delta\rVert^2` about :math:`T` is the radial
    term :math:`\rho gg^\top` at the same scale, :math:`\rho_{\mathrm{eff}} = \rho(s/s_{\mathrm{eff}})^4` and
    :math:`g = \gamma\odot z` (:func:`layernorm_radial_term`, of which it is the small-noise limit). The estimate is the
    delta method as :math:`\Sigma/s^2 \to 0`; for :math:`\Sigma = \sigma I` as :math:`\sigma/s^2 \to \infty` it is
    :math:`\tfrac{K}{K-1}\Gamma\Pi\Gamma`, the covariance of the uniform distribution on the centered sphere of
    radius :math:`\sqrt K`, plus its floors :math:`\tfrac{K}{K-1}f\Gamma^2 + f\Gamma\mathbf 1\mathbf 1^\top\Gamma/K`
    (below); in between its error is of order :math:`1/K_{\mathrm{eff}}`,
    :math:`K_{\mathrm{eff}} = T^2/\operatorname{tr}((\Pi\Sigma\Pi)^2)`, the concentration of :math:`\lVert\delta\rVert^2`
    (Monte Carlo at :math:`\Sigma/s^2` from :math:`10^{-2}` to :math:`10^3`: within 5% per coordinate at K = 128,
    10% at K = 32 and 22% at K = 8).

    Floor. The LayerNorm output has no variance along :math:`\Gamma^{-1}\mathbf 1` (its coordinates are centered),
    where :math:`fD\Sigma D` alone fills the null direction of :math:`P` with :math:`f` of the input's variance along
    :math:`\mathbf 1`. Behind another such seam that input variance is itself of order :math:`f` of the scale, so the
    floor compounds to order :math:`f^\ell` after :math:`\ell` seams (condition number :math:`4\cdot10^7` at the second
    layer at K = 8, audit 2026-10-16 F2). The moment pushforward therefore adds the span floor
    :math:`f(T/K)\,DWW^\top D` on :math:`\operatorname{span}\{\mathbf 1, z\}`, which :math:`P = I - WW^\top` annihilates
    or shrinks: :math:`f` times the mean pushed variance :math:`T/(Ks_{\mathrm{eff}}^2)` there, so the condition number
    stays about :math:`1/f` times the input's own anisotropy at every depth. The sphere bound
    (:func:`layernorm_trace_cap`) is read net of both floor terms (:func:`_layernorm_moment_floor_trace`).
    """
    return s2 + noise / K


def _centered_trace(
    sigma:    torch.Tensor,                           # (..., K) diagonal or (..., K, K) full covariance
    diagonal: bool,
) -> torch.Tensor:                                    # (..., 1) T = tr(Pi Sigma Pi), Pi = I - 1 1^T / K
    K = sigma.shape[-1]
    if diagonal:                                                                    # S0 (K - 1) / K
        return sigma.sum(dim=-1, keepdim=True) * ((K - 1) / K)
    return (torch.diagonal(sigma, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)        # tr Sigma - 1^T Sigma 1 / K
            - sigma.sum(dim=(-2, -1)).unsqueeze(-1) / K)


def _layernorm_moment_floor_trace(
    mu:               torch.Tensor,                   # (..., K) pre-norm means
    sigma:            torch.Tensor,                   # (..., K) diagonal or (..., K, K) full covariance at mu
    *,
    eps:              float,
    covariance_floor: float,
    weight:           Optional[torch.Tensor] = None,  # (K,) affine gamma, or None for the plain norm
) -> torch.Tensor:                                    # (..., 1) at promote_types(dtype, float32)
    r"""Trace of the moment pushforward's floor terms, :math:`f\sum_kd_k^2\bigl(\Sigma_{kk} + (T/K)(1 + z_k^2)/K\bigr)`.

    :math:`fD\Sigma D + f(T/K)DWW^\top D` with :math:`D = \operatorname{diag}(\gamma)/s_{\mathrm{eff}}`,
    :math:`z = c_\mu/s_{\mathrm{eff}}` and :math:`\operatorname{diag}(WW^\top)_k = (1 + z_k^2)/K`
    (:func:`_layernorm_effective_scale`), O(K) per token. The trace guard of ``'pushforward_moment'`` allows this on
    top of the sphere bound, which bounds the LayerNorm output's covariance and not the regularizing floor, so the
    guard binds only where the estimate net of its floor exceeds the bound. Head-block projection keeps the diagonal,
    so the trace is that of the ``gaussian_block`` result as well.
    """
    dtype = _pushforward_dtype(mu, sigma)
    diagonal = _is_diagonal_layout(mu, sigma)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        work, cov = mu.to(dtype), sigma.to(dtype)
        K = work.shape[-1]
        centered = work - work.mean(dim=-1, keepdim=True)                           # (..., K)
        s2 = centered.square().mean(dim=-1, keepdim=True) + eps                     # (..., 1)
        noise = _centered_trace(cov, diagonal)                                      # (..., 1) T
        scale2 = _layernorm_effective_scale(s2, noise, K)                           # (..., 1) s_eff^2
        variances = cov if diagonal else torch.diagonal(cov, dim1=-2, dim2=-1)     # (..., K) Sigma_kk
        d2 = 1.0 / scale2 if weight is None else weight.to(dtype).square() / scale2  # (..., 1) or (..., K)
        span = (noise / K) * (1.0 + centered.square() / scale2) / K                 # (..., K) (T/K) diag(W W^T)
        return covariance_floor * (d2 * (variances + span)).sum(dim=-1, keepdim=True)


def _layernorm_diagonal_pushforward_at(
    mu:               torch.Tensor,                   # (..., K) pre-norm means at the working dtype
    sigma:            torch.Tensor,                   # (..., K) diagonal covariance at the working dtype
    eps:              float,
    covariance_floor: float,
    weight:           Optional[torch.Tensor],         # (K,) gamma at the working dtype, or None
    effective_scale:  bool = False,                   # pushforward_moment: s^2 -> s_eff^2 = s^2 + tr(Pi Sigma Pi)/K
) -> torch.Tensor:                                    # (..., K)
    K = mu.shape[-1]
    centered = mu - mu.mean(dim=-1, keepdim=True)                                   # (..., K)
    s2 = centered.square().mean(dim=-1, keepdim=True) + eps                         # (..., 1) s^2
    noise = _centered_trace(sigma, True) if effective_scale else None               # (..., 1) T
    scale2 = _layernorm_effective_scale(s2, noise, K) if effective_scale else s2    # (..., 1) s_eff^2 or s^2
    z = centered / scale2.sqrt()                                                    # (..., K)
    z2 = z.square()
    S0 = sigma.sum(dim=-1, keepdim=True)                                            # (..., 1)
    S1 = (sigma * z).sum(dim=-1, keepdim=True)                                      # (..., 1)
    S2 = (sigma * z2).sum(dim=-1, keepdim=True)                                     # (..., 1)
    bracket = sigma * (1.0 - 2.0 * (1.0 + z2) / K) + (S0 + 2.0 * z * S1 + z2 * S2) / (K * K)   # (..., K)
    d2 = 1.0 / scale2 if weight is None else weight.square() / scale2               # (..., 1) or (..., K): D^2
    rho = _layernorm_radial_coefficient(centered, s2, sigma, eps)                   # (..., 1)
    floor = covariance_floor * sigma                                                # (..., K) f diag(Sigma)
    if effective_scale:
        rho = rho * (s2 / scale2).square()                                          # rho (s / s_eff)^4
        floor = floor + covariance_floor * (noise / K) * (1.0 + z2) / K             # span floor f (T/K) diag(W W^T)
    g2 = z2 if weight is None else weight.square() * z2                             # (..., K) (gamma o z)^2
    return d2 * (bracket + floor) + rho * g2


class BlockNormLinearization(NamedTuple):
    r"""The block norm's Jacobian :math:`J_{\mathrm{LN}}` at the pre-norm mean and its scale :math:`D = \operatorname{diag}(\gamma)/s`."""

    jacobian: torch.Tensor                            # (..., K, K)
    scale: torch.Tensor                               # (..., K)


def _layernorm_arguments(norm: Callable[..., torch.Tensor]) -> 'tuple[float, Optional[torch.Tensor]]':
    from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
    if isinstance(norm, AffineLayerNorm):
        return norm.eps, norm.weight
    if isinstance(norm, LayerNorm):
        return norm.eps, None
    raise ValueError(f"the delta_full and block-norm covariance pushforwards have the norm Jacobian of the "
                     f"LayerNorm seams only, got {type(norm).__name__}")


def block_norm_jacobian(
    norm: Callable[..., torch.Tensor],
    mu:   torch.Tensor,                               # (..., K) pre-norm means
) -> torch.Tensor:                                    # (..., K, K)
    r"""Jacobian of a block LayerNorm seam (plain or affine) at ``mu``; see :func:`layernorm_jacobian`.

    Only the LayerNorm seams have this closed form here; the Mahalanobis norm reads the covariance and is
    refused, so ``block_mlp_covariance='delta_full'`` never pushes a covariance through a norm whose
    Jacobian it omits.
    """
    eps, weight = _layernorm_arguments(norm)
    return layernorm_jacobian(mu, eps=eps, weight=weight)


def block_norm_linearization(
    norm: Callable[..., torch.Tensor],
    mu:   torch.Tensor,                               # (..., K) pre-norm means
) -> BlockNormLinearization:
    r"""Jacobian and scale :math:`D` of a block LayerNorm seam at ``mu`` (:func:`block_norm_jacobian`, :func:`layernorm_scale`)."""
    eps, weight = _layernorm_arguments(norm)
    return BlockNormLinearization(layernorm_jacobian(mu, eps=eps, weight=weight),
                                  layernorm_scale(mu, eps=eps, weight=weight))


# The block_norm_covariance values that push the covariance through the block LayerNorm.
#   pushforward         the delta method at the pre-norm mean, capped at the sphere bound (valid while Sigma << s^2)
#   pushforward_moment  the delta method at the effective scale s_eff^2 = s^2 + tr(Pi Sigma Pi)/K (audit 2026-10-16 F3)
BLOCK_NORM_PUSHFORWARD_MODES = ("pushforward", "pushforward_moment")


def block_norm_covariance_pushforward(
    norm:             Callable[..., torch.Tensor],
    mu:               torch.Tensor,                   # (..., K) pre-norm means
    sigma:            torch.Tensor,                   # (..., K) diagonal or (..., K, K) full covariance at mu
    covariance_floor: float,
    head_dims:        Optional[Sequence[int]] = None, # full layout: keep only these head blocks (gaussian_block)
    *,
    mode:             str                     = "pushforward",   # a BLOCK_NORM_PUSHFORWARD_MODES value
    trace_cap:        bool                    = True,            # block_norm_trace_cap; False returns it uncapped
    cap_moment_fallback: bool                 = False,           # block_norm_cap_moment_fallback (pushforward only)
) -> torch.Tensor:                                    # sigma's layout, at promote_types(dtype, float32)
    r"""The covariance through a LayerNorm seam at the pre-norm mean (``block_norm_covariance`` = ``mode``).

    Full layout: :math:`J_{\mathrm{LN}}\Sigma J_{\mathrm{LN}}^\top + fD\Sigma D + \rho gg^\top`,
    :math:`D = \operatorname{diag}(\gamma)/s`, in the O(K^2) rank-two form (:func:`layernorm_full_pushforward`), with
    the radial second-order term (:func:`layernorm_radial_term`); ``head_dims`` returns its head-block projection.
    Diagonal layout: its diagonal in closed form (:func:`layernorm_diagonal_pushforward`). Both are capped at the
    sphere bound (:func:`layernorm_trace_cap`).

    ``mode='pushforward'`` is the delta method at :math:`s`, a moment estimate only while :math:`\Sigma` is small
    against :math:`s^2`; where its cap binds the result keeps the delta method's radial shape rather than the
    LayerNorm output's (audit 2026-10-16 F3, F2). ``mode='pushforward_moment'`` evaluates it at the effective scale
    :math:`s_{\mathrm{eff}}^2 = s^2 + \operatorname{tr}(\Pi\Sigma\Pi)/K` (:func:`_layernorm_effective_scale`) with a span
    floor that keeps its conditioning from compounding across seams; it tracks the Monte Carlo moments at every
    :math:`\Sigma/s^2`, and net of its floor terms it is within the sphere bound in every tested regime, so there the
    cap, read net of the floor, is a guard whose counter reads zero.

    ``trace_cap=False`` (audit 2026-10-17 item 7) returns the uncapped delta method; the tokens whose trace exceeds
    the sphere bound are still counted (:func:`layernorm_trace_cap_elements`) and the largest ratio
    :math:`\operatorname{tr}P/B` is kept on the device, so the counter read warns once per run when it exceeds
    :data:`UNCAPPED_TRACE_RATIO_WARN` (at the launcher initialization the uncapped route overflows within a few seams
    and decodes uniformly).

    ``cap_moment_fallback`` (``block_norm_cap_moment_fallback``, ``mode='pushforward'`` with the cap only; audit
    2026-10-17 item 8) returns the moment estimate of ``'pushforward_moment'`` on the tokens where the cap binds,
    where the capped delta method keeps the radial shape :math:`\gamma_k^2z_k^2` and is off by up to about
    :math:`10^3` per coordinate on anisotropic input, and the capped delta method elsewhere. The tokens are counted
    as capped.

    Below float64 the scale :math:`\lambda` is computed in float64 from a max-scaled trace (audit 2026-10-17 item
    32), so a trace that overflows the working precision no longer gives :math:`\lambda = 0` (all-zero output) or
    :math:`0\cdot\infty` (NaN); on the diagonal layout a token whose working-precision pushforward overflowed takes
    its float64 evaluation, capped in float64 and cast back, with no gradient there.
    """
    if mode not in BLOCK_NORM_PUSHFORWARD_MODES:
        raise ValueError(f"block norm covariance pushforward mode must be one of {BLOCK_NORM_PUSHFORWARD_MODES}, "
                         f"got {mode!r}")
    moment = mode == "pushforward_moment"
    diagonal = _is_diagonal_layout(mu, sigma)
    eps, weight = _layernorm_arguments(norm)
    high = None
    if diagonal:
        pushed, high = _layernorm_diagonal_pushforward_pair(mu, sigma, eps=eps, covariance_floor=covariance_floor,
                                                            weight=weight, effective_scale=moment)
    else:
        pushed = _uncapped_block_norm_pushforward(norm, mu, sigma, covariance_floor, head_dims, effective_scale=moment)
    if not trace_cap:
        _record_uncapped_excess(pushed, weight, diagonal)
        return pushed
    allowance = (_layernorm_moment_floor_trace(mu, sigma, eps=eps, covariance_floor=covariance_floor, weight=weight)
                 if moment else None)                                        # (..., 1) the floor's trace, or None
    overflow = None
    if high is not None:                                                     # tokens whose working value overflowed
        overflow = ~torch.isfinite(pushed).all(dim=-1, keepdim=True)         # (..., 1)
        pushed = torch.where(overflow, torch.zeros_like(pushed), pushed)     # no inf * 0 in the graph
    lam = layernorm_trace_cap(pushed, weight, diagonal, allowance)           # (..., 1) float64
    capped = _apply_trace_cap(pushed, lam, diagonal)
    if overflow is not None:
        with torch.no_grad():
            bound = _sphere_bound(weight, high.dtype, allowance, K=high.shape[-1])
            lam_high = torch.clamp(bound / high.sum(dim=-1, keepdim=True), max=1.0)
            _count_capped(overflow & (lam_high < 1.0))
            capped = torch.where(overflow, (high * lam_high).to(capped.dtype), capped)
            lam = torch.where(overflow, lam_high, lam)
    if cap_moment_fallback and not moment:
        fallback = block_norm_covariance_pushforward(norm, mu, sigma, covariance_floor, head_dims,
                                                     mode="pushforward_moment", trace_cap=True)
        binds = lam < 1.0 if diagonal else (lam < 1.0).unsqueeze(-1)         # (..., 1) or (..., 1, 1)
        capped = torch.where(binds, fallback.to(capped.dtype), capped)
    return capped


# Ratio tr P / B above which the uncapped pushforward (block_norm_trace_cap=False) warns once per run.
UNCAPPED_TRACE_RATIO_WARN = 1e2


def _sphere_bound(
    weight:    Optional[torch.Tensor],                # (K,) affine gamma, or None
    dtype:     torch.dtype,
    allowance: Optional[torch.Tensor] = None,         # (..., 1)
    K:         Optional[int] = None,
) -> 'torch.Tensor | float':                          # B = K max gamma^2 (+ allowance)
    K = weight.shape[-1] if (weight is not None and K is None) else K
    bound = float(K) if weight is None else K * weight.to(dtype).square().amax()
    return bound if allowance is None else bound + allowance.to(dtype)


def _count_capped(capped: torch.Tensor) -> None:      # bool, any shape
    with torch.no_grad():
        key = str(capped.device)
        count = capped.sum()
        _LAYERNORM_TRACE_CAPS[key] = (count if key not in _LAYERNORM_TRACE_CAPS
                                      else _LAYERNORM_TRACE_CAPS[key] + count)


def _record_uncapped_excess(
    pushed:   torch.Tensor,                           # (..., K) or (..., K, K) uncapped LayerNorm pushforward
    weight:   Optional[torch.Tensor],
    diagonal: bool,
) -> None:
    r"""Count the uncapped tokens above the sphere bound and keep the largest ratio :math:`\operatorname{tr}P/B`
    on the device (no host synchronization; ``block_norm_trace_cap=False``, audit 2026-10-17 item 7)."""
    with torch.no_grad():
        K = pushed.shape[-1]
        work = pushed.detach().double()
        trace = work.sum(dim=-1) if diagonal else torch.diagonal(work, dim1=-2, dim2=-1).sum(dim=-1)   # (...,)
        ratio = trace / _sphere_bound(None if weight is None else weight.detach(), torch.float64, K=K)
        ratio = torch.where(torch.isnan(ratio), torch.full_like(ratio, float("inf")), ratio)
        _count_capped(ratio > 1.0)
        key = str(pushed.device)
        top = ratio.amax() if ratio.numel() else torch.zeros((), dtype=torch.float64, device=pushed.device)
        _LAYERNORM_UNCAPPED_RATIO[key] = (top if key not in _LAYERNORM_UNCAPPED_RATIO
                                          else torch.maximum(_LAYERNORM_UNCAPPED_RATIO[key], top))


def _uncapped_block_norm_pushforward(
    norm:             Callable[..., torch.Tensor],
    mu:               torch.Tensor,
    sigma:            torch.Tensor,
    covariance_floor: float,
    head_dims:        Optional[Sequence[int]] = None,
    effective_scale:  bool                    = False,
) -> torch.Tensor:
    eps, weight = _layernorm_arguments(norm)
    if _is_diagonal_layout(mu, sigma):
        return layernorm_diagonal_pushforward(mu, sigma, eps=eps, covariance_floor=covariance_floor, weight=weight,
                                              effective_scale=effective_scale)
    return layernorm_full_pushforward(mu, sigma, eps=eps, covariance_floor=covariance_floor, weight=weight,
                                      head_dims=head_dims, effective_scale=effective_scale)


# Tokens whose LayerNorm covariance pushforward trace exceeded the sphere bound, per device, counted on the device
# (no host synchronization); read by the training loop through layernorm_trace_cap_elements(). With the cap on these
# are the capped tokens; under block_norm_trace_cap=False the uncapped tokens above the bound (audit 2026-10-17 item 7).
_LAYERNORM_TRACE_CAPS: Dict[str, torch.Tensor] = {}
# Largest tr P / B of an uncapped pushforward since the last reset, per device, and whether this run has warned.
_LAYERNORM_UNCAPPED_RATIO: Dict[str, torch.Tensor] = {}
_UNCAPPED_WARNED = [False]


def layernorm_trace_cap(
    pushed:   torch.Tensor,                           # (..., K) diagonal or (..., K, K) full LayerNorm pushforward
    weight:   Optional[torch.Tensor],                 # (K,) affine gamma, or None for the plain norm
    diagonal: bool,
    allowance: Optional[torch.Tensor] = None,         # (..., 1) trace allowed above B (pushforward_moment: its floor)
) -> torch.Tensor:                                    # (..., 1) lambda in (0, 1]
    r"""Scale :math:`\lambda = \min(1, B/\operatorname{tr}P)` that caps a LayerNorm pushforward at the sphere bound.

    The exact LayerNorm output is :math:`\gamma\odot z + b` with :math:`z = c/\sqrt{v + \epsilon}`, so
    :math:`\lVert z\rVert^2 = Kv/(v + \epsilon) < K` for every input, and for any input distribution

    .. math::
        \operatorname{tr}\operatorname{Cov}(\gamma\odot z) \le E\lVert\gamma\odot z\rVert^2
            \le B = K\max_k\gamma_k^2

    (:math:`B = K` for the plain norm). Restricting :math:`z` to the centered sphere would tighten :math:`B` to
    :math:`K\lambda_{\max}(\Pi\Gamma^2\Pi)`, :math:`\Pi = I - \mathbf 1\mathbf 1^\top/K`, by at most the factor
    :math:`K/(K-1)`, which does not pay for an on-device eigenvalue solve on every norm call.
    :math:`B` carries its gradient with respect to :math:`\gamma` (the subgradient of ``amax``), so where the cap binds
    the output :math:`BP/\operatorname{tr}P` differentiates exactly (audit 2026-10-12 round 2, item 3). The delta
    method is valid only while the input variance is small against :math:`s^2`: its first-order term grows as
    :math:`\Sigma/s^2` and its radial term as :math:`(\Sigma/s^2)^2`, and at :math:`\Sigma/s^2 \approx 4\cdot 10^3` it
    gave a trace of :math:`7\cdot 10^6` against the Monte Carlo value :math:`K = 20`, reaching NaN by the third layer (audit 2026-10-12, item 1). Scaling the pushforward :math:`P` by
    :math:`\lambda` keeps it positive semidefinite, leaves it unchanged wherever its trace is within the bound, and
    otherwise makes its trace exactly :math:`B`. It fixes the trace, not the shape: where it binds the capped
    delta method is :math:`BP/\operatorname{tr}P`, dominated by the radial term :math:`\rho gg^\top`, so its diagonal
    follows :math:`\gamma_k^2z_k^2` and its full layout is near rank one, while the LayerNorm output tends to the
    uniform distribution on the centered sphere (audit 2026-10-16 F3, F2;
    ``block_norm_covariance='pushforward_moment'`` is the moment estimate there). The capped tokens are counted
    (:func:`layernorm_trace_cap_elements`).

    ``allowance`` raises the bound to :math:`B + a` per token. The moment pushforward passes the trace of its floor
    terms (:func:`_layernorm_moment_floor_trace`): the sphere bound holds for the LayerNorm output's covariance, not
    for the regularizing floor added to it, so there the cap is a guard on the estimate net of its floor.
    """
    K = pushed.shape[-1]
    wide = torch.promote_types(pushed.dtype, torch.float64)
    bound = _sphere_bound(weight, wide, allowance, K=K)                                       # carries d/d gamma
    entries = pushed.to(wide) if diagonal else torch.diagonal(pushed, dim1=-2, dim2=-1).to(wide)   # (..., K)
    # Max-scaled float64 trace (audit 2026-10-17 item 32): lambda = (B / m) / sum_k(P_kk / m), m = max_k |P_kk|.
    m = entries.detach().abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(wide).tiny)   # (..., 1)
    lam = torch.clamp((bound / m) / (entries / m).sum(dim=-1, keepdim=True), max=1.0)       # (..., 1) float64
    _count_capped(lam < 1.0)
    return lam


def _apply_trace_cap(
    covariance: torch.Tensor,                         # (..., K) or (..., K, K)
    lam:        Optional[torch.Tensor],               # (..., 1) from layernorm_trace_cap, or None
    diagonal:   bool,
) -> torch.Tensor:
    if lam is None:
        return covariance
    wide = torch.promote_types(covariance.dtype, lam.dtype)                 # the product at lambda's float64
    scaled = covariance.to(wide) * (lam if diagonal else lam.unsqueeze(-1))
    return scaled.to(covariance.dtype)


def block_norm_trace_cap(
    norm:             Callable[..., torch.Tensor],
    mu:               torch.Tensor,                   # (..., K) pre-norm means
    sigma:            torch.Tensor,                   # (..., K) or (..., K, K) pre-norm covariance
    covariance_floor: float,
) -> torch.Tensor:                                    # (..., 1)
    r"""The sphere-bound scale :math:`\lambda` of a block LayerNorm seam (:func:`layernorm_trace_cap`) for the
    ``delta_full`` composite: its pushforward is linear in the norm-stage covariance
    :math:`J_{\mathrm{LN}}\Sigma J_{\mathrm{LN}}^\top + fD\Sigma D + \rho gg^\top`, so capping that stage by :math:`\lambda`
    scales the whole composite by :math:`\lambda` (``input_cap`` of ``forward_moments``)."""
    pushed = _uncapped_block_norm_pushforward(norm, mu, sigma, covariance_floor)
    return layernorm_trace_cap(pushed, _layernorm_arguments(norm)[1], _is_diagonal_layout(mu, sigma))


def layernorm_trace_cap_elements() -> int:
    r"""LayerNorm-pushforward tokens above the sphere bound since the last reset, over all devices: capped with the
    cap on, passed through uncapped under ``block_norm_trace_cap=False``. Host-syncs.

    Under ``block_norm_trace_cap=False`` the first read after the largest uncapped ratio :math:`\operatorname{tr}P/B`
    exceeds :data:`UNCAPPED_TRACE_RATIO_WARN` warns once per run (audit 2026-10-17 item 7)."""
    if _LAYERNORM_UNCAPPED_RATIO and not _UNCAPPED_WARNED[0]:
        top = layernorm_uncapped_trace_ratio_max()
        if not top <= UNCAPPED_TRACE_RATIO_WARN:
            _UNCAPPED_WARNED[0] = True
            warnings.warn(
                f"block_norm_trace_cap=False: an uncapped LayerNorm covariance pushforward reached {top:.3g} times "
                f"the sphere bound K max(gamma^2) (warning above {UNCAPPED_TRACE_RATIO_WARN:g}); the delta method is "
                "outside its regime (Sigma >> s^2) and the covariance can overflow within a few layers. Use the cap "
                "or block_norm_covariance='pushforward_moment'.", RuntimeWarning, stacklevel=2)
    return int(sum(int(count) for count in _LAYERNORM_TRACE_CAPS.values()))


def layernorm_uncapped_trace_ratio_max() -> float:
    r"""Largest :math:`\operatorname{tr}P/B` of an uncapped pushforward since the last reset (0 when none ran;
    inf for a non-finite trace). Host-syncs."""
    return max((float(value) for value in _LAYERNORM_UNCAPPED_RATIO.values()), default=0.0)


def reset_layernorm_trace_cap_elements() -> None:
    r"""Zero the per-device LayerNorm trace-cap counters, the uncapped ratio and the once-per-run warning."""
    _LAYERNORM_TRACE_CAPS.clear()
    _LAYERNORM_UNCAPPED_RATIO.clear()
    _UNCAPPED_WARNED[0] = False


def layernorm_full_pushforward(
    mu:               torch.Tensor,                   # (..., K) pre-norm means
    sigma:            torch.Tensor,                   # (..., K, K) SPD covariance at mu
    *,
    eps:              float,
    covariance_floor: float,
    weight:           Optional[torch.Tensor] = None,  # (K,) affine gamma, or None for the plain norm
    head_dims:        Optional[Sequence[int]] = None, # keep only these head blocks (gaussian_block), sum = K
    effective_scale:  bool                    = False, # block_norm_covariance='pushforward_moment': at s_eff
) -> torch.Tensor:                                    # (..., K, K)
    r"""The full-layout LayerNorm pushforward in O(K^2) per token (audit 2026-10-10 round 2, item 50).

    :math:`J_{\mathrm{LN}} = DP` with :math:`P = I - WW^\top`, :math:`W = [\mathbf 1, z]/\sqrt K` (:func:`layernorm_jacobian`;
    the columns are orthogonal and :math:`P` has eigenvalue :math:`\epsilon/(v + \epsilon)` on :math:`z`), so

    .. math::
        J_{\mathrm{LN}}\Sigma J_{\mathrm{LN}}^\top + fD\Sigma D + \rho gg^\top = D\bigl(P\Sigma P + f\Sigma\bigr)D + \rho gg^\top,
        \qquad P\Sigma P = \Sigma - W(\Sigma W)^\top - (\Sigma W)W^\top + W(W^\top\Sigma W)W^\top,

    a rank-two correction of :math:`\Sigma` instead of the dense Jacobian, its Cholesky factor, :math:`JL` and their
    Gram (about :math:`2.3K^3` per token). Below float64 the result is checked by a Cholesky factorization, of
    each head block when ``head_dims`` is given (only those blocks are kept), and tokens that fail are recomputed in
    float64 and cast back, as in :func:`_delta_covariance`; the cast is not re-checked, so a float64 result whose
    condition number approaches :math:`1/u_{32}` can still fail to factor at the working dtype (the capped delta
    method at :math:`\Sigma \gg s^2`, audit 2026-10-16 F2). Autocast off and IEEE fp32 GEMMs at
    ``promote_types(dtype, float32)``.

    The delta method approximates the LayerNorm output's moments only while :math:`\Sigma` is small against
    :math:`s^2`; beyond, its radial term dominates and the capped result is near rank one (condition number about
    :math:`10^7` at the launcher's initialization, audit 2026-10-16 F2). ``effective_scale``
    (``block_norm_covariance='pushforward_moment'``) evaluates the same expression at the effective scale
    :math:`s_{\mathrm{eff}}^2 = s^2 + \operatorname{tr}(\Pi\Sigma\Pi)/K` (:func:`_layernorm_effective_scale`), where
    :math:`P` has eigenvalue :math:`(\epsilon + T/K)/(v + \epsilon + T/K)` on :math:`z` and the floor
    :math:`fD\Sigma D` stays about :math:`f` of the pushed scale.
    """
    dtype = _pushforward_dtype(mu, sigma)
    with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
        pushed = _layernorm_full_pushforward_at(
            mu.to(dtype), sigma.to(dtype), eps, covariance_floor, None if weight is None else weight.to(dtype),
            head_dims, effective_scale)
        if dtype == torch.float64:
            return pushed
        failed = _cholesky_failures(pushed, head_dims)                              # (...,)
        if bool(failed.any()):
            high = _layernorm_full_pushforward_at(
                mu.double(), sigma.double(), eps, covariance_floor, None if weight is None else weight.double(),
                head_dims, effective_scale)
            pushed = torch.where(failed.unsqueeze(-1).unsqueeze(-1), high.to(dtype), pushed)
    return pushed


def _layernorm_full_pushforward_at(
    mu:               torch.Tensor,                   # (..., K) at the working dtype
    sigma:            torch.Tensor,                   # (..., K, K) at the working dtype
    eps:              float,
    covariance_floor: float,
    weight:           Optional[torch.Tensor],         # (K,) at the working dtype, or None
    head_dims:        Optional[Sequence[int]],
    effective_scale:  bool = False,                   # pushforward_moment: s^2 -> s_eff^2 = s^2 + tr(Pi Sigma Pi)/K
) -> torch.Tensor:                                    # (..., K, K) symmetrized
    K = mu.shape[-1]
    centered = mu - mu.mean(dim=-1, keepdim=True)                                   # (..., K)
    s2 = centered.square().mean(dim=-1, keepdim=True) + eps                         # (..., 1) s^2
    noise = _centered_trace(sigma, False) if effective_scale else None              # (..., 1) T
    scale2 = _layernorm_effective_scale(s2, noise, K) if effective_scale else s2    # (..., 1) s_eff^2 or s^2
    z = centered / scale2.sqrt()                                                    # (..., K)
    d = (1.0 / scale2.sqrt()).expand_as(z) if weight is None else weight / scale2.sqrt()   # (..., K) D
    W = torch.stack([torch.ones_like(z), z], dim=-1) / math.sqrt(K)                 # (..., K, 2)
    SW = sigma @ W                                                                  # (..., K, 2) Sigma W
    A = W.transpose(-1, -2) @ SW                                                    # (..., 2, 2) W^T Sigma W
    cross = W @ SW.transpose(-1, -2)                                                # (..., K, K) W (Sigma W)^T
    core = sigma - cross - cross.transpose(-1, -2) + W @ A @ W.transpose(-1, -2) + covariance_floor * sigma
    if effective_scale:                                                             # span floor f (T/K) W W^T
        core = core + (covariance_floor * noise / K).unsqueeze(-1) * (W @ W.transpose(-1, -2))
    g = z if weight is None else weight * z                                         # (..., K) gamma o z
    rho = _layernorm_radial_coefficient(centered, s2, sigma, eps)                   # (..., 1)
    if effective_scale:
        rho = rho * (s2 / scale2).square()                                          # rho (s / s_eff)^4
    pushed = d.unsqueeze(-1) * core * d.unsqueeze(-2) + rho.unsqueeze(-1) * (g.unsqueeze(-1) * g.unsqueeze(-2))
    pushed = 0.5 * (pushed + pushed.transpose(-1, -2))
    if head_dims is None:
        return pushed
    from vfe3.families.gaussian import head_block_projection
    return head_block_projection(pushed, head_dims)


def _cholesky_failures(
    pushed:    torch.Tensor,                          # (..., K, K) symmetric
    head_dims: Optional[Sequence[int]],               # factor only these diagonal blocks, or None for the whole matrix
) -> torch.Tensor:                                    # (...,) bool
    if head_dims is None:
        return torch.linalg.cholesky_ex(pushed).info != 0
    failed = torch.zeros(pushed.shape[:-2], dtype=torch.bool, device=pushed.device)
    start = 0
    for dim in head_dims:
        stop = start + int(dim)
        failed = failed | (torch.linalg.cholesky_ex(pushed[..., start:stop, start:stop]).info != 0)
        start = stop
    return failed


def _pushforward_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = torch.float32
    for tensor in tensors:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def _compose_input_jacobian(
    jacobian:       torch.Tensor,                     # (..., K, K) J of the MLP at its input
    input_jacobian: Optional[torch.Tensor],           # (..., K, K) J of the map feeding the MLP, or None
) -> torch.Tensor:                                    # (..., K, K) chain rule J J_in
    r"""Chain rule :math:`J_{\mathrm{MLP}}J_{\mathrm{in}}` for a map (the block norm) applied before the MLP.

    Evaluated with autocast off and IEEE fp32 GEMMs at ``promote_types(dtype, float32)``: :math:`J_{\mathrm{LN}}` has
    rank :math:`K - 1`, and a bf16 or TF32 product leaks rounding into its null direction (audit 2026-10-09 round 2, #2).
    """
    if input_jacobian is None:
        return jacobian
    dtype = _pushforward_dtype(jacobian, input_jacobian)
    with torch.autocast(device_type=jacobian.device.type, enabled=False), ieee_fp32_matmul():
        return torch.einsum("...ij,...jk->...ik", jacobian.to(dtype), input_jacobian.to(dtype))


def _reject_input_jacobian(input_jacobian: Optional[torch.Tensor], contract: str) -> None:
    if input_jacobian is not None:
        raise ValueError(f"input_jacobian composes the delta_full pushforward; covariance contract "
                         f"{contract!r} pushes no covariance")


def _check_input_linearization(
    input_jacobian: Optional[torch.Tensor],
    input_scale:    Optional[torch.Tensor],
    input_radial:   Optional[LayerNormRadialTerm] = None,
    input_cap:      Optional[torch.Tensor] = None,
) -> None:
    if (input_jacobian is None) != (input_scale is None):
        raise ValueError("input_jacobian and input_scale come together from block_norm_linearization: the floor "
                         "of a pushforward through the block norm is expressed in post-norm units")
    if input_radial is not None and input_jacobian is None:
        raise ValueError("input_radial is the block norm's radial second-order term (block_norm_radial_term) and "
                         "needs the norm's input_jacobian and input_scale")
    if input_cap is not None and input_jacobian is None:
        raise ValueError("input_cap is the block norm's sphere-bound scale (block_norm_trace_cap) and needs the "
                         "norm's input_jacobian and input_scale")


def _push_radial(
    jacobian: torch.Tensor,                           # (..., K, K) J of the MLP at the post-norm mean
    radial:   Optional[LayerNormRadialTerm],          # the block norm's rho g g^T, or None
) -> Optional[LayerNormRadialTerm]:
    r"""The norm's radial term carried through the MLP: :math:`\rho(J_{\mathrm{MLP}}g)(J_{\mathrm{MLP}}g)^\top`.

    Autocast off and IEEE fp32 GEMMs at ``promote_types(dtype, float32)``, as :func:`_compose_input_jacobian`.
    """
    if radial is None:
        return None
    dtype = _pushforward_dtype(jacobian, radial.direction)
    with torch.autocast(device_type=jacobian.device.type, enabled=False), ieee_fp32_matmul():
        direction = torch.einsum("...ij,...j->...i", jacobian.to(dtype), radial.direction.to(dtype))   # (..., K)
    return LayerNormRadialTerm(radial.rho.to(dtype), direction)


def _covariance_cholesky(sigma: torch.Tensor) -> torch.Tensor:   # (..., K, K) SPD -> lower factor L
    r"""Lower Cholesky factor :math:`L` of the SPD belief covariance, :math:`\Sigma = LL^\top`.

    No jitter. Matrices whose factor fails at the working precision are refactored in float64; a float64 failure
    raises, because :math:`\Sigma` is then not numerically SPD and the pushforward has no Gram factor.
    """
    factor, info = torch.linalg.cholesky_ex(sigma)
    failed = info != 0
    if bool(failed.any()):
        if sigma.dtype == torch.float64:
            raise FloatingPointError("delta_full: the belief covariance is not numerically SPD (float64 Cholesky failed)")
        high, high_info = torch.linalg.cholesky_ex(sigma.double())
        if bool((high_info[failed] != 0).any()):
            raise FloatingPointError("delta_full: the belief covariance is not numerically SPD after float64 escalation")
        factor = torch.where(failed.unsqueeze(-1).unsqueeze(-1), high.to(sigma.dtype), factor)
    return factor


def _delta_covariance(
    jacobian:         torch.Tensor,                   # (..., K, K) J of the (composite) mean map
    sigma:            torch.Tensor,                   # (..., K, K) SPD input covariance, or its diagonal (..., K)
    covariance_floor: float,
    floor_scale:      Optional[torch.Tensor] = None,  # (..., K) D = diag(gamma)/s of a block norm, or None
    radial:           Optional[LayerNormRadialTerm] = None,   # the norm's radial term through the MLP, or None
    cap:              Optional[torch.Tensor] = None,  # (..., 1) the norm's sphere-bound scale, or None
) -> torch.Tensor:                                    # sigma's layout
    r"""Delta-method pushforward :math:`(JL)(JL)^\top + \rho hh^\top + fD\Sigma D` with :math:`\Sigma = LL^\top`.

    :math:`\rho hh^\top`, :math:`h = J_{\mathrm{MLP}}(\gamma\odot z)`, is the block norm's radial second-order variance
    carried through the MLP (:func:`layernorm_radial_term`, :func:`_push_radial`; absent without a block norm).
    A diagonal layout :math:`\sigma` (``gaussian_diagonal``) returns the diagonal
    :math:`(J\circ J)\sigma + \rho h^2 + fD^2\sigma` of the dense pushforward, its M-projection onto the diagonal
    family (:func:`_diagonal_delta_covariance`).

    The first term is formed as a Gram matrix of the factor :math:`JL` rather than as the triple product
    :math:`J\Sigma J^\top`, so it is positive semidefinite up to the rounding of one GEMM, relative to
    :math:`\lVert JL\rVert^2` rather than :math:`\lVert J\rVert^2\lVert\Sigma\rVert`; the floor makes it definite. Below
    float64 the result is certified by a Cholesky factorization and failed matrices are recomputed in float64.
    Without a block norm (``floor_scale=None``) :math:`D = I` and the floor is :math:`f\Sigma`, the covariant floor.
    Through the block LayerNorm, :math:`J = J_{\mathrm{MLP}}J_{\mathrm{LN}}` with
    :math:`J_{\mathrm{LN}} = D(I - \tfrac1K\mathbf 1\mathbf 1^\top - \tfrac1K zz^\top)` and :math:`D = \operatorname{diag}(\gamma)/s`,
    and the floor :math:`fD\Sigma D` is the input covariance carried to post-norm units by the scale part of the
    norm. :math:`J\Sigma J^\top` scales as :math:`1/s^2`, and so does this floor, so its share of the pushforward is
    independent of :math:`s` (the pre-norm floor :math:`f\Sigma` had a share growing as :math:`s^2`; audit 2026-10-09
    round 2, #13). Everything runs with autocast off and IEEE fp32 GEMMs at ``promote_types(dtype, float32)``: under
    bf16 autocast the rank-deficient :math:`J_{\mathrm{LN}}` product returned indefinite covariances (#2).
    With ``cap`` (:func:`block_norm_trace_cap`) the whole result, linear in the norm-stage covariance, is scaled by the
    norm's sphere-bound :math:`\lambda` (audit 2026-10-12, item 1).
    """
    if sigma.dim() == jacobian.dim() - 1:
        return _apply_trace_cap(_diagonal_delta_covariance(jacobian, sigma, covariance_floor, floor_scale, radial),
                                cap, True)
    dtype = _pushforward_dtype(jacobian, sigma)
    with torch.autocast(device_type=jacobian.device.type, enabled=False), ieee_fp32_matmul():
        pushed = _gram_pushforward(jacobian.to(dtype), sigma.to(dtype), covariance_floor,
                                   None if floor_scale is None else floor_scale.to(dtype), radial)
        if dtype == torch.float64:
            return _apply_trace_cap(pushed, cap, False)
        # The GEMM rounding of the Gram is bounded by about K u ||J L||^2, which exceeds the floor's smallest
        # eigenvalue f lambda_min(D Sigma D) once cond(Sigma) passes about f / (K u) (u the working roundoff). The
        # result is therefore certified by a Cholesky factorization at the working precision, and matrices that
        # fail it are recomputed in float64 and rounded back (still indefinite only if the float64 result itself
        # cannot be represented as SPD at the working precision).
        _, info = torch.linalg.cholesky_ex(pushed)
        failed = info != 0
        if bool(failed.any()):
            high = _gram_pushforward(jacobian.double(), sigma.double(), covariance_floor,
                                     None if floor_scale is None else floor_scale.double(), radial)
            pushed = torch.where(failed.unsqueeze(-1).unsqueeze(-1), high.to(dtype), pushed)
        return _apply_trace_cap(pushed, cap, False)


def _diagonal_delta_covariance(
    jacobian:         torch.Tensor,                   # (..., K, K) J of the (composite) mean map
    sigma:            torch.Tensor,                   # (..., K) diagonal input covariance, sigma_k > 0
    covariance_floor: float,
    floor_scale:      Optional[torch.Tensor] = None,  # (..., K) D = diag(gamma)/s of a block norm, or None
    radial:           Optional[LayerNormRadialTerm] = None,   # rho, h = J_mlp (gamma o z), or None
) -> torch.Tensor:                                    # (..., K)
    r"""Diagonal delta method :math:`\sum_lJ_{kl}^2\sigma_l + \rho h_k^2 + fD_k^2\sigma_k`, the diagonal of :func:`_delta_covariance`.

    Every term is non-negative and the floor is positive, so the result needs no Cholesky factor or certificate.
    :math:`D = I` without a block norm. Autocast off and IEEE fp32 GEMMs at ``promote_types(dtype, float32)``.
    """
    dtype = _pushforward_dtype(jacobian, sigma)
    with torch.autocast(device_type=jacobian.device.type, enabled=False), ieee_fp32_matmul():
        work = sigma.to(dtype)                                                      # (..., K)
        pushed = torch.einsum("...kl,...l->...k", jacobian.to(dtype).square(), work)   # (J o J) sigma
        if radial is not None:
            pushed = pushed + radial.rho.to(dtype) * radial.direction.to(dtype).square()   # rho h^2
        floor = work if floor_scale is None else floor_scale.to(dtype).square() * work
        return pushed + covariance_floor * floor


def _gram_pushforward(
    J:                torch.Tensor,                   # (..., K, K)
    S:                torch.Tensor,                   # (..., K, K) SPD
    covariance_floor: float,
    d:                Optional[torch.Tensor],         # (..., K) floor scale D, or None for D = I
    radial:           Optional[LayerNormRadialTerm] = None,   # rho (..., 1), h (..., K), or None
) -> torch.Tensor:                                    # (..., K, K) symmetrized (JL)(JL)^T + rho h h^T + f D S D
    L = _covariance_cholesky(S)                                                     # (..., K, K)
    JL = torch.einsum("...ij,...jk->...ik", J, L)                                   # (..., K, K)
    gram = torch.einsum("...ik,...lk->...il", JL, JL)                               # (JL)(JL)^T
    if radial is not None:
        h = radial.direction.to(J.dtype)                                            # (..., K)
        gram = gram + radial.rho.to(J.dtype).unsqueeze(-1) * (h.unsqueeze(-1) * h.unsqueeze(-2))
    if d is None:
        floor = covariance_floor * S
    else:
        floor = covariance_floor * (d.unsqueeze(-1) * S * d.unsqueeze(-2))          # f D Sigma D
    pushed = gram + floor
    return 0.5 * (pushed + pushed.transpose(-1, -2))


class BlockMLP(nn.Module):
    """Ordinary coordinate residual MLP, retained as the backward-compatible default."""

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        activation: str,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        covariance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.activation_name = activation
        self.covariance_contract = _validate_block_mlp_covariance_contract(covariance_contract)
        self.covariance_floor = float(covariance_floor)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _hidden_pre(
        self,
        mu:          torch.Tensor,                    # (..., K)
        hidden_bias: Optional[torch.Tensor],          # (..., w) or None
    ) -> torch.Tensor:                                # (..., H)
        hidden_pre = self.fc1(mu)
        if hidden_bias is None:
            return hidden_pre
        return _add_hidden_bias(hidden_pre, hidden_bias)

    def forward(
        self,
        mu: torch.Tensor,
        *,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Residual update mu + W_2 act(W_1 mu + b_1 + E) + b_2, E = [hidden_bias, 0] (absent when None)."""
        return mu + self.dropout(self.fc2(self.activation(self._hidden_pre(mu, hidden_bias))))

    def _coordinate_update_with_jacobian(
        self,
        mu: torch.Tensor,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        hidden_pre = self._hidden_pre(mu, hidden_bias)
        residual = self.fc2(self.activation(hidden_pre))
        dropout_scale = _dropout_scale(self.dropout, residual)
        mu_out = mu + dropout_scale * residual
        # The Jacobian feeds the covariance pushforward, which runs at promote_types(dtype, float32) with
        # autocast off (see _delta_covariance); the mean path keeps the ambient autocast.
        dtype = _pushforward_dtype(mu, hidden_pre, self.fc1.weight)
        with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
            residual_jacobian = torch.einsum(
                "oh,...h,hi->...oi",
                self.fc2.weight.to(dtype),
                _activation_derivative(self.activation_name, hidden_pre.to(dtype)),
                self.fc1.weight.to(dtype),
            )                                                                       # (..., K, K)
            residual_jacobian = dropout_scale.to(dtype).unsqueeze(-1) * residual_jacobian
            identity = torch.eye(mu.shape[-1], dtype=dtype, device=mu.device)
            return mu_out, identity + residual_jacobian

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,    # (..., w), w <= H: per-token fc1 bias
        input_jacobian: Optional[torch.Tensor] = None, # (..., K, K) Jacobian of the block norm before the MLP
        input_scale: Optional[torch.Tensor] = None,    # (..., K) its scale D = diag(gamma)/s (with input_jacobian)
        input_radial: Optional[LayerNormRadialTerm] = None,   # its radial second-order term (with input_jacobian)
        input_cap: Optional[torch.Tensor] = None,      # (..., 1) the norm's sphere-bound scale (block_norm_trace_cap)
    ) -> BlockMLPMomentResult:
        r"""Mean update and covariance under the contract; ``delta_full`` gives :math:`J\Sigma J^\top + f\Sigma`.

        With ``input_jacobian`` :math:`J_{\mathrm{in}}` (the block norm's Jacobian at the pre-norm mean) the
        Jacobian is the composite :math:`J = J_{\mathrm{MLP}}J_{\mathrm{in}}` and ``sigma`` is the pre-norm
        covariance, so the pushforward is that of MLP∘norm, with the floor :math:`fD\Sigma D` in post-norm units
        and, with ``input_radial``, the norm's radial term carried through the MLP (:func:`_delta_covariance`).
        A diagonal ``sigma`` (..., K) returns the diagonal of the pushforward.
        """
        del frame_context
        _check_input_linearization(input_jacobian, input_scale, input_radial, input_cap)
        if self.covariance_contract == "passthrough":
            _reject_input_jacobian(input_jacobian, self.covariance_contract)
            return BlockMLPMomentResult(self.forward(mu, hidden_bias=hidden_bias), sigma, None)
        mu_out, jacobian = self._coordinate_update_with_jacobian(mu, hidden_bias)
        radial = _push_radial(jacobian, input_radial)
        jacobian = _compose_input_jacobian(jacobian, input_jacobian)
        return BlockMLPMomentResult(
            mu_out, _delta_covariance(jacobian, sigma, self.covariance_floor, input_scale, radial, input_cap), jacobian)


class GaugeGateBlockMLP(nn.Module):
    """Gauge-equivariant residual gate built from Mahalanobis invariants.

    The deterministic/evaluation path and every fixed realization of the scalar per-block
    dropout mask are exactly equivariant. Independent dropout draws are equivariant in
    distribution, not identical sample by sample.

    For every declared irrep block h, s_h = mu_h^T Sigma_hh^-1 mu_h is invariant under the
    corresponding invertible frame action. A scalar MLP maps the invariant vector s to one
    gate per block, and that scalar multiplies every coordinate in the block.

    Sigma may be given in the full layout (..., K, K) or the diagonal layout (..., K). On the
    diagonal layout the invariant is s_h = sum_{k in h} mu_k^2 / sigma_k, and the gate equals the
    full-layout gate on diag_embed(sigma). The operator is covariant under the block frame group
    prod_h GL(d_h) (not under a dense GL(K)): the diagonal family is not closed under that action,
    but the image g diag(sigma) g^T is block-diagonal and the full-layout gate evaluates it.
    delta_full (J Sigma J^T is dense) requires the full layout.
    """

    def __init__(
        self,
        irrep_dims: Sequence[int],
        expansion: int,
        activation: str,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        covariance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.irrep_dims = tuple(int(dim) for dim in irrep_dims)
        if not self.irrep_dims or any(dim < 1 for dim in self.irrep_dims):
            raise ValueError(f"irrep_dims must contain positive dimensions, got {irrep_dims!r}")
        n_blocks = len(self.irrep_dims)
        hidden_dim = n_blocks * expansion
        self.activation_name = activation
        self.covariance_contract = _validate_block_mlp_covariance_contract(covariance_contract)
        self.fc1 = nn.Linear(n_blocks, hidden_dim)
        self.covariance_floor = float(covariance_floor)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, n_blocks)
        self.dropout = nn.Dropout(dropout)
        # Identity at step zero keeps this opt-in transform from perturbing a baseline before its
        # invariant gate has learned evidence for a nonzero update.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _invariants(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> 'tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]':
        mu_blocks = []
        expected_dim = sum(self.irrep_dims)
        if mu.shape[-1] != expected_dim:
            raise ValueError(
                "irrep_dims must sum to the mean dimension, got "
                f"sum(irrep_dims)={expected_dim} and mean dimension={mu.shape[-1]}"
            )
        if _is_diagonal_layout(mu, sigma):
            return self._diagonal_invariants(mu, sigma)
        if sigma.shape[-2:] != (expected_dim, expected_dim):
            raise ValueError(
                "gauge gate covariance axes must match sum(irrep_dims), got "
                f"{tuple(sigma.shape[-2:])} and expected {(expected_dim, expected_dim)}"
            )
        sigma_solutions = []
        invariants = []
        start = 0
        for dim in self.irrep_dims:
            stop = start + dim
            mu_h = mu[..., start:stop]
            sigma_hh = sigma[..., start:stop, start:stop]
            rhs = mu_h.unsqueeze(-1)
            checked = validated_cholesky_solve(sigma_hh, rhs)
            if bool(checked.certified.all()):
                assert checked.solution is not None
                solved = checked.solution.squeeze(-1)
            elif sigma_hh.dtype is torch.float64:
                raise FloatingPointError(
                    "GaugeGate Mahalanobis system could not be certified in float64")
            else:
                high = validated_cholesky_solve(sigma_hh.double(), rhs.double())
                if not bool(high.certified.all()):
                    raise FloatingPointError(
                        "GaugeGate Mahalanobis system could not be certified after float64 "
                        "escalation")
                assert checked.solution is not None and high.solution is not None
                solved = torch.where(
                    checked.certified.unsqueeze(-1).unsqueeze(-1),
                    checked.solution.double(), high.solution,
                ).squeeze(-1)
            mu_blocks.append(mu_h)
            sigma_solutions.append(solved)
            invariants.append((mu_h.to(solved.dtype) * solved).sum(dim=-1))
            start = stop
        return torch.stack(invariants, dim=-1), mu_blocks, sigma_solutions

    def _diagonal_invariants(
        self,
        mu:    torch.Tensor,                          # (..., K)
        sigma: torch.Tensor,                          # (..., K) = diag Sigma, sigma_k > 0
    ) -> 'tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]':
        r"""Block invariants of a diagonal covariance layout.

        .. math::
            s_b = \mu_b^\top \Sigma_{bb}^{-1} \mu_b = \sum_{k \in b} \frac{\mu_k^2}{\sigma_k},

        evaluated at ``promote_types(mu, sigma, float32)`` (at least fp32; float64 stays float64). No
        clamp: sigma is positive through its log parameterization, and a clamp would break the exact
        agreement with the full-layout path on ``diag_embed(sigma)``.
        """
        compute_dtype = torch.promote_types(
            torch.promote_types(mu.dtype, sigma.dtype), torch.float32)
        mu_work = mu.to(compute_dtype)                # (..., K)
        sigma_work = sigma.to(compute_dtype)          # (..., K)
        mu_blocks, solutions, invariants = [], [], []
        start = 0
        for dim in self.irrep_dims:
            stop = start + dim
            mu_h = mu_work[..., start:stop]                               # (..., d_b)
            solved = mu_h / sigma_work[..., start:stop]                   # Sigma_bb^{-1} mu_b
            mu_blocks.append(mu[..., start:stop])
            solutions.append(solved)
            invariants.append((mu_h * solved).sum(dim=-1))                # (...,)
            start = stop
        return torch.stack(invariants, dim=-1), mu_blocks, solutions      # (..., B)

    def _gate_values(
        self,
        invariants: torch.Tensor,
        *,
        need_jacobian: bool,
    ) -> 'tuple[torch.Tensor, Optional[torch.Tensor]]':
        hidden_pre = torch.nn.functional.linear(
            invariants,
            self.fc1.weight.to(invariants.dtype),
            self.fc1.bias.to(invariants.dtype),
        )
        raw_gates = torch.nn.functional.linear(
            self.activation(hidden_pre),
            self.fc2.weight.to(hidden_pre.dtype),
            self.fc2.bias.to(hidden_pre.dtype),
        )
        if not need_jacobian:
            return self.dropout(raw_gates), None
        dropout_scale = _dropout_scale(self.dropout, raw_gates)
        dgate_ds = torch.einsum(
            "oh,...h,hi->...oi",
            self.fc2.weight.to(hidden_pre.dtype),
            _activation_derivative(self.activation_name, hidden_pre),
            self.fc1.weight.to(hidden_pre.dtype),
        )
        return dropout_scale * raw_gates, dropout_scale.unsqueeze(-1) * dgate_ds

    def mean_update(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        invariants, mu_blocks, _ = self._invariants(mu, sigma)
        gates, _ = self._gate_values(invariants, need_jacobian=False)
        return torch.cat([
            (1.0 + gates[..., index:index + 1]) * mu_h
            for index, mu_h in enumerate(mu_blocks)
        ], dim=-1)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
        input_jacobian: Optional[torch.Tensor] = None,  # (..., K, K) block-norm Jacobian (delta_full only)
        input_scale: Optional[torch.Tensor] = None,     # (..., K) its scale D = diag(gamma)/s
        input_radial: Optional[LayerNormRadialTerm] = None,   # its radial second-order term
        input_cap: Optional[torch.Tensor] = None,      # (..., 1) the norm's sphere-bound scale (block_norm_trace_cap)
    ) -> BlockMLPMomentResult:
        del frame_context
        _check_input_linearization(input_jacobian, input_scale, input_radial, input_cap)
        _reject_hidden_bias(
            hidden_bias,
            "the gauge_gate fc1 acts on the per-block Mahalanobis invariants, not on a coordinate "
            "hidden layer")
        need_jacobian = self.covariance_contract == "delta_full"
        if not need_jacobian:
            _reject_input_jacobian(input_jacobian, self.covariance_contract)
        if need_jacobian and _is_diagonal_layout(mu, sigma):
            raise ValueError(
                "gauge_gate covariance_contract='delta_full' needs the full (..., K, K) covariance "
                "layout: the pushed covariance J Sigma J^T is dense, so a diagonal layout cannot "
                "carry it")
        invariants, mu_blocks, sigma_solutions = self._invariants(mu, sigma)
        gates, dgate_ds = self._gate_values(invariants, need_jacobian=need_jacobian)
        mu_out = torch.cat([
            (1.0 + gates[..., index:index + 1]) * mu_h
            for index, mu_h in enumerate(mu_blocks)
        ], dim=-1)
        if not need_jacobian:
            return BlockMLPMomentResult(mu_out, sigma, None)

        K = mu.shape[-1]
        jacobian = torch.zeros(
            *mu.shape[:-1], K, K, dtype=gates.dtype, device=mu.device)
        row_start = 0
        for out_index, (out_dim, mu_h) in enumerate(zip(self.irrep_dims, mu_blocks)):
            row_stop = row_start + out_dim
            jacobian[..., row_start:row_stop, row_start:row_stop] = \
                (1.0 + gates[..., out_index]).unsqueeze(-1).unsqueeze(-1) \
                * torch.eye(out_dim, dtype=gates.dtype, device=mu.device)
            col_start = 0
            for in_index, (in_dim, solved) in enumerate(
                    zip(self.irrep_dims, sigma_solutions)):
                col_stop = col_start + in_dim
                cross = torch.einsum(
                    "...i,...,...j->...ij",
                    mu_h.to(jacobian.dtype),
                    dgate_ds[..., out_index, in_index],
                    2.0 * solved,
                )
                jacobian[..., row_start:row_stop, col_start:col_stop] = \
                    jacobian[..., row_start:row_stop, col_start:col_stop] + cross
                col_start = col_stop
            row_start = row_stop
        # The gate reads sigma as a fixed parameter of its invariants; the chain rule covers the mean map only.
        radial = _push_radial(jacobian, input_radial)
        jacobian = _compose_input_jacobian(jacobian, input_jacobian)
        return BlockMLPMomentResult(
            mu_out,
            _delta_covariance(jacobian, sigma, self.covariance_floor, input_scale, radial, input_cap),
            jacobian,
        )


class CanonicalFrameBlockMLP(BlockMLP):
    """Ordinary residual MLP in a realized, frame-fixed vertex coordinate system."""

    def forward(
        self,
        mu: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.mean_update(mu, frame_context, hidden_bias=hidden_bias)

    def mean_update(
        self,
        mu: torch.Tensor,
        frame: Optional[CanonicalFrameContext],
        *,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""mu -> U F(U^{-1} mu); a hidden_bias enters F's fc1 pre-activation in canonical coordinates."""
        frame = _require_canonical_frame_context(frame)
        apply = _frame_action(frame, mu)
        canonical_mu = apply(frame.inverse, mu)
        canonical_out = super().forward(canonical_mu, hidden_bias=hidden_bias)
        return apply(frame.forward, canonical_out)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
        input_jacobian: Optional[torch.Tensor] = None,  # (..., K, K) block-norm Jacobian (delta_full only)
        input_scale: Optional[torch.Tensor] = None,     # (..., K) its scale D = diag(gamma)/s
        input_radial: Optional[LayerNormRadialTerm] = None,   # its radial second-order term
        input_cap: Optional[torch.Tensor] = None,      # (..., 1) the norm's sphere-bound scale (block_norm_trace_cap)
    ) -> BlockMLPMomentResult:
        frame_context = _require_canonical_frame_context(frame_context)
        _check_input_linearization(input_jacobian, input_scale, input_radial, input_cap)
        apply = _frame_action(frame_context, mu)
        canonical_mu = apply(frame_context.inverse, mu)
        if self.covariance_contract == "passthrough":
            _reject_input_jacobian(input_jacobian, self.covariance_contract)
            canonical_out = super().forward(canonical_mu, hidden_bias=hidden_bias)
            mu_out = apply(frame_context.forward, canonical_out)
            return BlockMLPMomentResult(mu_out, sigma, None)
        canonical_out, canonical_jacobian = self._coordinate_update_with_jacobian(
            canonical_mu, hidden_bias)
        mu_out = apply(frame_context.forward, canonical_out)
        if not _escalated_frame(frame_context, mu):
            dtype = _pushforward_dtype(canonical_jacobian, frame_context.forward)
            # kappa(U)-sensitive conjugation U J U^{-1}: IEEE fp32 with autocast off, as the pushforward
            with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
                realized_jacobian = (
                    frame_context.forward.to(dtype) @ canonical_jacobian.to(dtype)
                    @ frame_context.inverse.to(dtype)
                )
        else:
            realized_jacobian = (
                frame_context.forward @ canonical_jacobian.double() @ frame_context.inverse
            ).to(canonical_jacobian.dtype)
        radial = _push_radial(realized_jacobian, input_radial)
        realized_jacobian = _compose_input_jacobian(realized_jacobian, input_jacobian)
        # A diagonal sigma (..., K) is local, like the realized Jacobian: it reads the diagonal of the local pushforward.
        return BlockMLPMomentResult(
            mu_out,
            _delta_covariance(realized_jacobian, sigma, self.covariance_floor, input_scale, radial, input_cap),
            realized_jacobian,
        )

class GaussianTemplateBlockMLP(nn.Module):
    r"""Residual block computed as Gibbs routing to M = expansion * K learned Gaussian templates.

    Each template a carries a diagonal Gaussian p_a = N(nu_a, diag(lambda_a)), a log prior weight
    log pi_a, and a value w_a. Under a mixture prior p(k) = sum_a pi_a p_a, the optimal variational
    responsibilities of the belief q_i = N(mu_i, Sigma_i) are

    .. math::
        r_{ia} \propto \pi_a \exp\bigl(-D_{\mathrm{KL}}(q_i \,\|\, p_a)\bigr),

    because the entropy H(q_i) is template-independent. Dropping every a-independent term,

    .. math::
        \log r_{ia} = \log\pi_a - \tfrac12 \sum_k \Bigl[
            \frac{(\Sigma_i)_{kk} + (\mu_{ik} - \nu_{ak})^2}{\lambda_{ak}} + \log\lambda_{ak}
        \Bigr] + \mathrm{const}_i,

    and the mean update is mu_i + sum_a r_ia w_a (Gaussian mixture regression with zero regression
    blocks). Only diag(Sigma_i) enters, so full and diagonal covariance layouts give the same
    logits. With shared isotropic covariances the logits are affine in mu_i and the block is a
    width-M MLP whose GELU gate is replaced by a softmax over templates. The covariance passes
    through unchanged; the block acts in the same local coordinates as the coordinate MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
    ) -> None:
        super().__init__()
        if covariance_contract != "passthrough":
            raise ValueError(
                "gaussian_template block MLP supports only covariance_contract='passthrough', "
                f"got {covariance_contract!r}")
        n_templates = embed_dim * expansion
        self.covariance_contract = covariance_contract
        # Initialization mirrors BlockMLP: template means match fc1 rows, values match fc2 columns.
        bound_in = embed_dim ** -0.5
        bound_out = n_templates ** -0.5
        self.nu = nn.Parameter(torch.empty(n_templates, embed_dim).uniform_(-bound_in, bound_in))
        self.log_lambda = nn.Parameter(torch.zeros(n_templates, embed_dim))    # lambda_a = 1 at init
        self.log_pi = nn.Parameter(torch.zeros(n_templates))                   # uniform prior at init
        self.values = nn.Parameter(
            torch.empty(n_templates, embed_dim).uniform_(-bound_out, bound_out))
        self.dropout = nn.Dropout(dropout)

    def responsibilities(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        r"""Return r_ia, shape (..., M), from mu (..., K) and sigma (..., K) or (..., K, K).

        The expanded square below reaches magnitudes of order K (Sigma_kk + mu_k^2) / lambda_ak and
        cancels terms of that size, so it is evaluated with autocast disabled at
        ``promote_types(mu, float32)`` (float64 stays float64); r is returned in ``mu.dtype``. Under
        bf16 autocast the rounded logits otherwise moved r by a mean total variation of 0.05 at
        variance ~1 and 0.26 at variance ~15 (CPU autocast, K = 256, M = 2048, 2026-09-28).
        """
        sigma_diag = sigma if sigma.dim() == mu.dim() else torch.diagonal(sigma, dim1=-2, dim2=-1)
        dtype = torch.promote_types(mu.dtype, torch.float32)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            mu_work = mu.to(dtype)                                             # (..., K)
            nu = self.nu.to(dtype)                                             # (M, K)
            precision = torch.exp(-self.log_lambda.to(dtype))                  # (M, K)
            # sum_k (Sigma_kk + mu_k^2 - 2 mu_k nu_ak + nu_ak^2) / lambda_ak, contracted as matmuls
            quadratic = (
                (sigma_diag.to(dtype) + mu_work.square()) @ precision.T        # (..., M)
                - 2.0 * mu_work @ (nu * precision).T
                + (nu.square() * precision).sum(dim=-1)
            )
            logits = self.log_pi.to(dtype) - 0.5 * (
                quadratic + self.log_lambda.to(dtype).sum(dim=-1))
            responsibilities = torch.softmax(logits, dim=-1)                   # (..., M)
        return responsibilities.to(mu.dtype)

    def mean_update(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        message = self.responsibilities(mu, sigma) @ self.values.to(mu.dtype)  # (..., K)
        return mu + self.dropout(message)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.mean_update(mu, sigma)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> BlockMLPMomentResult:
        del frame_context
        _reject_hidden_bias(
            hidden_bias,
            "the gaussian_template block routes to templates and has no fc1 hidden layer")
        return BlockMLPMomentResult(self.mean_update(mu, sigma), sigma, None)


class PrototypeAgentBlockMLP(nn.Module):
    r"""Residual block as the minimal agent model with :math:`M = eK` static prototype agents (``prototype_agents``).

    Template :math:`a` carries the belief :math:`p_a = \mathcal N(\nu_a, \Lambda_a)`,
    :math:`\Lambda_a = \operatorname{diag}\lambda_a`, :math:`\lambda_a = e^{\log\lambda_a}`, and the prior weight
    :math:`\pi_a = \operatorname{softmax}(\log\pi)_a`; the token's working belief :math:`\mathcal N(\xi_i, C_i)` is
    the receiver. Query and sending maps are the identity (:math:`S = T = I`, full rank) and the channel noise
    is absorbed into :math:`\Lambda_a`, so the record covariance is :math:`W_{ia} = C_i + \Lambda_a`
    (``minimal_agent_model.tex``, Definition ma:def:model, Proposition ma:prop:minimal (a), (d)). ``inference``:

    * ``"exact"``: :math:`w_{ia} \propto \pi_a\mathcal N(\nu_a;\xi_i, C_i + \Lambda_a)`,
      :math:`\xi_i^+ = \xi_i + C_iu_i`, :math:`C_i^+ = C_i - C_i(\bar\Lambda_i - R_i)C_i`, the moments of the exact
      mixture posterior (:func:`vfe3.model.prototype_agents.prototype_exact_posterior`).
    * ``"mean_field"``: the first structured-mean-field step of the same record model started at the token's
      prior (Proposition ma:prop:mean_field). Its label step
      :math:`r_{ia} \propto \pi_a\exp(-D_{\mathrm{KL}}(\mathcal N(\xi_i, C_i)\,\|\,p_a))` is exactly
      :meth:`GaussianTemplateBlockMLP.responsibilities`; its receiver step is the power posterior
      :math:`P_i = (C_i^{-1} + \operatorname{diag}D_i)^{-1}`, :math:`D_i = \sum_a r_{ia}/\lambda_a`,
      :math:`\xi_i^+ = P_i(C_i^{-1}\xi_i + \sum_a r_{ia}\nu_a/\lambda_a)`, :math:`C_i^+ = P_i`, which omits the
      between-template spread (:func:`vfe3.model.prototype_agents.prototype_mean_field_posterior`).

    The update is a precision-weighted combination of the token's own mean and the template means: the
    values are tied to the template means (there is no free value matrix) and the residual is derived, the
    coefficient of :math:`\xi_i` being :math:`I - C_i\bar\Lambda_i` (exact) or :math:`P_iC_i^{-1}` (mean field),
    not the identity. It is a precision-weighted modern-Hopfield-type retrieval with tied keys and values: for
    shared isotropic :math:`C_i = cI` and :math:`\Lambda_a = \lambda I` the exact weights are
    :math:`\operatorname{softmax}_a(\beta\nu_a^\top\xi_i - \beta|\nu_a|^2/2 + \log\pi_a)`, :math:`\beta = 1/(c + \lambda)`,
    and :math:`\xi_i^+ = (\lambda\xi_i + c\sum_aw_{ia}\nu_a)/(c + \lambda)`, which at :math:`\lambda \to 0` is the
    retrieval :math:`\xi^{\mathrm{new}} = \sum_a\operatorname{softmax}_a(\beta\nu_a^\top\xi + b_a)\nu_a` of Ramsauer
    et al. (2020, arXiv:2008.02217) with stored patterns :math:`\nu_a` as both keys and values. Dropout (the
    block's ``dropout``) applies to the mean increment :math:`\xi^+ - \xi` only.

    Coordinates and layouts: ``sigma`` may be diagonal ``(..., K)`` or dense ``(..., K, K)`` (local coordinates,
    like the coordinate MLP), or compact blocks ``(..., G, d, d)`` (the reference covariance of diagonal beliefs
    on a block frame, passed by ``block_coordinates='reference'``); the covariance is returned in the same
    layout. ``"passthrough"`` returns the input covariance object; ``"posterior"`` returns :math:`C^+`, whose
    diagonal (diagonal blocks) is the moment-matched member of a diagonal (block) family. The exact path on a
    block or dense layout factors one :math:`b \times b` matrix per (token, template, block): an extreme
    opt-in path (cost in ``docs/implementation/prototype-agents-results-2026-10-06.md``); ``tile_bytes`` bounds
    its pair tensors.

    Initialization mirrors :class:`GaussianTemplateBlockMLP`'s template means,
    :math:`\nu \sim U(-K^{-1/2}, K^{-1/2})` from the global stream, with :math:`\log\lambda = \log\lambda_0`
    (``template_init_variance``) and :math:`\log\pi = 0`. At :math:`\lambda_0 = 10` and token variances near 1
    the initial pull toward the templates is :math:`c/(c + \lambda_0) \approx 9` percent per layer. The
    template means :math:`\nu` train in the block-MLP optimizer group under the global ``weight_decay``;
    :math:`\log\lambda` and :math:`\log\pi` train at the same rate with ``weight_decay=0``
    (:func:`vfe3.train.build_optimizer`), because decoupled decay would pull :math:`\lambda` toward 1 and
    :math:`\pi` toward uniform.

    ``cavi_steps`` (mean field only) is the number of coordinate-ascent sweeps of the mean-field posterior
    (:func:`vfe3.model.prototype_agents.prototype_mean_field_posterior`; 1 is the first step). With an
    ``em_rows`` list, :meth:`forward_moments` also appends the per-token EM M-step objective
    (:meth:`em_objective`), which ``block_mlp_em_weight`` adds to the training loss.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        inference: str = "mean_field",
        template_init_variance: float = 10.0,
        cavi_steps: int = 1,
    ) -> None:
        super().__init__()
        if covariance_contract not in PROTOTYPE_COVARIANCE_CONTRACTS:
            raise ValueError(
                f"prototype_agents block MLP supports covariance_contract in {PROTOTYPE_COVARIANCE_CONTRACTS}, "
                f"got {covariance_contract!r}")
        if inference not in PROTOTYPE_INFERENCE_MODES:
            raise ValueError(
                f"prototype_agents inference must be one of {PROTOTYPE_INFERENCE_MODES}, got {inference!r}")
        if (isinstance(template_init_variance, bool) or not isinstance(template_init_variance, (int, float))
                or not math.isfinite(float(template_init_variance)) or float(template_init_variance) <= 0.0):
            raise ValueError(
                f"template_init_variance must be a finite real value > 0, got {template_init_variance!r}")
        if isinstance(cavi_steps, bool) or not isinstance(cavi_steps, int) or cavi_steps < 1:
            raise ValueError(f"cavi_steps must be an int >= 1, got {cavi_steps!r}")
        n_templates = embed_dim * expansion
        self.covariance_contract = covariance_contract
        self.inference = inference
        self.cavi_steps = cavi_steps                                 # mean-field coordinate-ascent sweeps (not state)
        self.tile_bytes = PROTOTYPE_TILE_BYTES                       # exact-path pair-tile budget (not state)
        bound_in = embed_dim ** -0.5
        self.nu = nn.Parameter(torch.empty(n_templates, embed_dim).uniform_(-bound_in, bound_in))   # (M, K)
        self.log_lambda = nn.Parameter(
            torch.full((n_templates, embed_dim), math.log(float(template_init_variance))))       # (M, K)
        self.log_pi = nn.Parameter(torch.zeros(n_templates))                                       # (M,)
        self.dropout = nn.Dropout(dropout)

    def posterior(
        self,
        mu:    torch.Tensor,                          # (..., K) working means
        sigma: torch.Tensor,                          # (..., K), (..., K, K) or (..., G, d, d)
        *,
        need_covariance: Optional[bool] = None,       # None: as the covariance contract requires
        need_precision:  bool = False,
    ) -> 'ExactPrototypePosterior | MeanFieldPrototypePosterior':
        r"""The template posterior of the selected ``inference`` at the working belief ``(mu, sigma)``."""
        if need_covariance is None:
            need_covariance = self.covariance_contract == "posterior"
        if self.inference == "exact":
            return prototype_exact_posterior(
                mu, sigma, self.nu, self.log_lambda, self.log_pi, need_covariance=need_covariance,
                need_precision=need_precision, tile_bytes=self.tile_bytes)
        return prototype_mean_field_posterior(
            mu, sigma, self.nu, self.log_lambda, self.log_pi, need_covariance=need_covariance,
            cavi_steps=self.cavi_steps)

    @torch.no_grad()
    def calibrate_templates(
        self,
        mu:        torch.Tensor,                      # (..., K) working means this block receives
        sigma:     torch.Tensor,                      # (..., K), (..., K, K) or (..., G, d, d)
        *,
        pull:      float,                             # target initial pull p in (0, 1)
        generator: torch.Generator,                   # CPU generator for the template draw
    ) -> Dict[str, float]:
        r"""Data-dependent template initialization (``block_mlp_template_init='working_variance'``).

        With :math:`\bar c` the mean diagonal working variance of the inputs, every template variance is set to
        :math:`\lambda_0 = \bar c(1 - p)/p`, so that at uniform routing the gain :math:`\bar c/(\bar c + \lambda_0)`
        is the target pull :math:`p`, and the template means are :math:`M` input means drawn without replacement
        (with replacement when there are fewer than :math:`M`), so they sit at the scale of the block input. The
        absolute default placed :math:`\nu` at scale :math:`K^{-1/2}` and used a fixed :math:`\lambda_0`, which gave
        a 12 to 44 percent initial pull and near-uniform routing at ``sigma_init=15`` (audit 2026-10-07, F77/F78).
        :math:`\lambda` sets both the pull and the routing temperature, so a small pull still implies soft initial
        routing unless the spread of the inputs exceeds :math:`\lambda_0`. Returns the measured
        :math:`\bar c` and the chosen :math:`\lambda_0`.
        """
        K = self.nu.shape[-1]
        if sigma.dim() == mu.dim():
            diag = sigma
        elif sigma.dim() == mu.dim() + 1:
            diag = sigma.diagonal(dim1=-2, dim2=-1)
        else:
            diag = sigma.diagonal(dim1=-2, dim2=-1).reshape(*sigma.shape[:-3], K)
        c_bar = float(diag.double().mean())
        lam0 = c_bar * (1.0 - pull) / pull
        rows = mu.reshape(-1, K)
        M, n = self.nu.shape[0], rows.shape[0]
        index = (torch.randperm(n, generator=generator)[:M] if n >= M
                 else torch.randint(0, n, (M,), generator=generator))
        self.nu.copy_(rows[index.to(rows.device)].to(self.nu.dtype))
        self.log_lambda.fill_(math.log(lam0))
        return {"working_variance": c_bar, "lambda0": lam0}

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
        em_rows: Optional[list[torch.Tensor]] = None,  # block_mlp_em_weight > 0: receives this block's (...,) EM rows
    ) -> BlockMLPMomentResult:
        r""":math:`\mu \mapsto \mu + \mathrm{dropout}(\xi^+ - \mu)`; the covariance is passed through or replaced by :math:`C^+`.

        With ``em_rows`` it also appends :meth:`em_objective` under the posterior this block ran (the mean field then
        forms :math:`C^+` for its expectations; the mean update is unchanged)."""
        del frame_context
        _reject_hidden_bias(
            hidden_bias,
            "the prototype_agents block conditions the belief on template agents and has no fc1 hidden layer")
        em_moments = em_rows is not None and self.inference == "mean_field"
        posterior = self.posterior(mu, sigma, need_covariance=True if em_moments else None)
        if em_rows is not None:
            em_rows.append(self.em_objective(mu, sigma, posterior))
        mu_out = mu + self.dropout(posterior.delta_xi.to(mu.dtype))
        if self.covariance_contract == "passthrough":
            return BlockMLPMomentResult(mu_out, sigma, None)
        return BlockMLPMomentResult(mu_out, posterior.C_plus.to(sigma.dtype), None)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.forward_moments(mu, sigma).mu

    def em_objective(
        self,
        mu:        torch.Tensor,                      # (..., K) block-input means, in the coordinates the templates read
        sigma:     torch.Tensor,                      # (..., K), (..., K, K) or (..., G, d, d)
        posterior: 'ExactPrototypePosterior | MeanFieldPrototypePosterior',   # the posterior the block ran
    ) -> torch.Tensor:                                # (...) Q_i
        r"""EM M-step objective of the templates under the posterior the template weights came from (audit
        2026-10-10 round 2, item 15).

        The model is :math:`a_i \sim \hat\pi`, :math:`X_i \mid a_i \sim \mathcal N(\nu_a, \Lambda_a)`,
        :math:`\xi_i \mid X_i \sim \mathcal N(X_i, C_i)`, :math:`\hat\pi = \operatorname{softmax}(\log\pi)`, whose evidence
        is :math:`p(\xi_i \mid \theta) = \sum_a\hat\pi_a\mathcal N(\nu_a;\xi_i, C_i + \Lambda_a)`. With a posterior
        :math:`q(a_i, X_i)` held fixed (detached),

        .. math::
            Q_i(\theta) = \sum_aq(a)\Bigl[\log\hat\pi_a - \tfrac12\sum_k\Bigl(\log 2\pi\lambda_{ak}
                + \frac{(m_{iak} - \nu_{ak})^2 + p_{iak}}{\lambda_{ak}}\Bigr)\Bigr]
            = E_q[\log\hat\pi_a + \log\mathcal N(X_i;\nu_a,\Lambda_a)],

        with :math:`(m_{ia}, \operatorname{diag}P_{ia} = p_{ia})` the moments of :math:`q(X_i \mid a)`:

        * ``"exact"``: :math:`q(a) = w_{ia}` and the component posteriors :math:`m_{ia} = \xi_i + C_iW_{ia}^{-1}(\nu_a - \xi_i)`,
          :math:`P_{ia} = C_iW_{ia}^{-1}\Lambda_a` at the current templates
          (:func:`vfe3.model.prototype_agents.prototype_exact_em_objective`). By Fisher's identity the gradient at the
          current templates is :math:`\nabla_\theta\log p(\xi_i \mid \theta)`, and its fixed point is the maximum of the
          evidence.
        * ``"mean_field"``: :math:`q(a) = r_{ia}` of the last sweep and :math:`q(X_i) = \mathcal N(\xi_i^+, C_i^+)`, the
          posterior that sweep formed from those responsibilities, so the gradient is that of the mean-field ELBO in
          :math:`\theta` at the final coordinate-ascent state (:math:`m_{ia} = \xi_i^+`, :math:`p_{ia} = \operatorname{diag}C_i^+`
          for every :math:`a`; the bracket is :func:`vfe3.model.prototype_agents.template_kl_logits` at
          :math:`(\xi^+, \operatorname{diag}C^+)` and the normalized log prior, minus :math:`\tfrac K2\log 2\pi`).

        The previous term paired these weights with expectations under the block-input moments
        :math:`(\xi_i, \operatorname{diag}C_i)`: the gradient of the router's reduced free energy for one mean-field sweep,
        but under exact inference it missed :math:`\nabla\log p` and its fixed point put :math:`\lambda` at :math:`V + c`
        instead of :math:`V - c`. Only :math:`(\nu, \log\lambda, \log\pi)` receive a gradient.
        """
        log_pi_hat = torch.log_softmax(self.log_pi, dim=-1)                  # (M,) log softmax(log pi)
        if self.inference == "exact":
            return prototype_exact_em_objective(mu.detach(), sigma.detach(), posterior.weights.detach(), self.nu,
                                                self.log_lambda, log_pi_hat, tile_bytes=self.tile_bytes)
        layout = covariance_layout(mu, sigma)
        bracket = template_kl_logits(posterior.xi_plus.detach(), covariance_diagonal(posterior.C_plus.detach(), layout),
                                     self.nu, self.log_lambda, log_pi_hat)   # (..., M)
        K = mu.shape[-1]
        return ((posterior.responsibilities.detach().to(bracket.dtype) * bracket).sum(dim=-1)
                - 0.5 * K * math.log(2.0 * math.pi))

    @torch.no_grad()
    def weights_and_gain(
        self,
        mu:    torch.Tensor,                          # (..., K) working means
        sigma: torch.Tensor,                          # (..., K), (..., K, K) or (..., G, d, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:           # (..., M) template weights, (...) removed variance
        r"""Template weights (exact :math:`w_{ia}`, mean-field responsibilities :math:`r_{ia}`) and the removed
        working variance :math:`\operatorname{tr}(C_i\bar\Lambda_i)` (exact) or :math:`\operatorname{tr}(P_i\operatorname{diag}D_i)`
        (mean field), detached; :meth:`statistics` divides it by :math:`K`."""
        weights, gain, _ = self._template_statistics(mu, sigma, staleness=False)
        return weights, gain

    @torch.no_grad()
    def weights_gain_and_staleness(
        self,
        mu:    torch.Tensor,                          # (..., K) working means
        sigma: torch.Tensor,                          # (..., K), (..., K, K) or (..., G, d, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # (..., M), (...), (...)
        r""":meth:`weights_and_gain` and the label staleness (audit 2026-10-10 round 2, item 54).

        Mean field: the total variation :math:`\tfrac12\sum_a|r^+_{ia} - r_{ia}|` between the last sweep's
        responsibilities and the label update at that sweep's posterior,
        :math:`r^+_i = \operatorname{softmax}(\text{template\_kl\_logits}(\xi_i^+, \operatorname{diag}C_i^+))`: the move
        one further coordinate-ascent sweep would make, near 0 at the fixed point (the prototype analogue of
        ``relational_minimal_agent_label_staleness``), computed in float64. Exact inference: NaN."""
        return self._template_statistics(mu, sigma, staleness=True)

    @torch.no_grad()
    def elbo_and_evidence(
        self,
        mu:    torch.Tensor,                          # (..., K) working means this block receives
        sigma: torch.Tensor,                          # (..., K), (..., K, K) or (..., G, d, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:           # (...) ELBO, (...) log evidence, float64 nats
        r"""Evidence lower bound of the posterior this block's ``inference`` forms and the exact log evidence of the
        record model (``diagnose_agent_elbo``, audit 2026-10-12 T1).

        :func:`vfe3.model.prototype_agents.prototype_elbo` at :math:`q = N(\xi^+, C^+)\,q(a)` (mean field: the last
        sweep's responsibilities and power posterior; exact: the template posterior :math:`w` and the moment-matched
        :math:`C^+` in the input layout) and ``log_evidence`` :math:`\log\sum_a\hat\pi_aN(\nu_a;\xi_i, C_i + \Lambda_a)` of
        :func:`vfe3.model.prototype_agents.prototype_exact_posterior`. The posterior is recomputed in float64 from the
        detached inputs (dropout is not applied: it acts on the mean increment after the posterior); the mean field
        runs the exact path once more for the evidence, at :math:`O(MK)` per token on the diagonal layout and one
        :math:`b\times b` factor per (token, template, block) otherwise. A diagnostic: never part of the loss.
        """
        mu64, sigma64 = mu.detach().double(), sigma.detach().double()
        posterior = self.posterior(mu64, sigma64, need_covariance=True)
        if self.inference == "exact":
            labels, evidence = posterior.weights, posterior.log_evidence
        else:
            labels = posterior.responsibilities
            evidence = prototype_exact_posterior(mu64, sigma64, self.nu, self.log_lambda, self.log_pi,
                                                 need_covariance=False, tile_bytes=self.tile_bytes).log_evidence
        elbo = prototype_elbo(mu64, sigma64, posterior.xi_plus, posterior.C_plus, labels, self.nu, self.log_lambda,
                              self.log_pi)
        return elbo, evidence

    def _template_statistics(
        self,
        mu:        torch.Tensor,                      # (..., K) working means
        sigma:     torch.Tensor,                      # (..., K), (..., K, K) or (..., G, d, d)
        *,
        staleness: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        layout = covariance_layout(mu, sigma)
        if self.inference == "exact":
            posterior = self.posterior(mu, sigma, need_covariance=False, need_precision=True)
            Lambda_bar = posterior.Lambda_bar
            axes = {"diagonal": 1, "dense": 2, "blocks": 3}[layout]
            gain = (sigma.to(Lambda_bar.dtype) * Lambda_bar).sum(dim=tuple(range(-axes, 0)))   # tr(C Lambda_bar)
            return posterior.weights, gain, (torch.full_like(gain, math.nan) if staleness else None)
        posterior = self.posterior(mu, sigma, need_covariance=True)
        c_plus = covariance_diagonal(posterior.C_plus, layout)                              # (..., K)
        gain = (c_plus * posterior.D).sum(dim=-1)                                           # tr(P diag D)
        if not staleness:
            return posterior.responsibilities, gain, None
        f64 = torch.float64
        relabeled = torch.softmax(template_kl_logits(posterior.xi_plus.to(f64), c_plus.to(f64), self.nu,
                                                     self.log_lambda, self.log_pi), dim=-1)   # (..., M)
        stale = 0.5 * (relabeled - posterior.responsibilities.to(f64)).abs().sum(dim=-1)     # (...)
        return posterior.responsibilities, gain, stale

    def statistics(
        self,
        mu:    torch.Tensor,                          # (..., K) working means
        sigma: torch.Tensor,                          # (..., K), (..., K, K) or (..., G, d, d)
    ) -> Dict[str, float]:
        r"""Detached token means of the template posterior at ``(mu, sigma)``.

        ``prototype_weight_entropy``: :math:`H(w_i) = -\sum_aw_{ia}\log w_{ia}` (nats; responsibilities under
        mean field) and ``prototype_effective_templates`` its exponential. ``prototype_gain_fraction``: the share
        of the working variance the block removes, :math:`\operatorname{tr}(C_i\bar\Lambda_i)/K` (exact, before the
        spread is added back) or :math:`\operatorname{tr}(P_i\operatorname{diag}D_i)/K = 1 - \operatorname{tr}(P_iC_i^{-1})/K`
        (mean field), both in :math:`[0, 1)`; about 0.09 at initialization (:math:`\lambda = 10`, unit
        variances). ``prototype_template_variance``: the geometric mean of :math:`\lambda`.
        ``prototype_prior_entropy``: :math:`H(\pi)`. ``prototype_label_staleness``: the mean-field label staleness of
        :meth:`weights_gain_and_staleness` (NaN under exact inference).
        """
        K = mu.shape[-1]
        weights, gain, staleness = self.weights_gain_and_staleness(mu, sigma)
        entropy = -torch.special.xlogy(weights, weights).sum(dim=-1)
        prior = torch.softmax(self.log_pi.double(), dim=-1)
        return {
            "prototype_weight_entropy": float(entropy.mean()),
            "prototype_effective_templates": float(entropy.exp().mean()),
            "prototype_gain_fraction": float((gain / K).mean()),
            "prototype_template_variance": float(self.log_lambda.double().mean().exp()),
            "prototype_prior_entropy": float(-torch.special.xlogy(prior, prior).sum()),
            "prototype_label_staleness": float(staleness.mean()),
        }


class CueResultMemoryBlockMLP(nn.Module):
    r"""Block as exact conditioning of a cue/result component model (``block_mlp_mode='cue_result_memory'``).

    The model and its posterior are in :mod:`vfe3.model.cue_result_memory`: component :math:`a` has a cue law
    :math:`\mathcal N(\nu_a, L_a)`, a result law :math:`Y\mid X\sim\mathcal N(\omega_a + F_a(X - \nu_a), Q_a)` and a
    prior weight :math:`\hat\pi_a`; the working belief :math:`(\xi_i, C_i)` is a Gaussian record of the cue. The block
    returns the first two moments of the exact result marginal, so the new belief is
    :math:`\mathcal N(\sum_aw_am_{Y,a},\ \sum_aw_a[P_{Y,a} + (m_{Y,a} - \bar m)(m_{Y,a} - \bar m)^\top])`. There is no
    additional residual: at :math:`\omega_a = \nu_a`, :math:`F_a = I` the result mean is the same-state posterior mean
    of the cue, and :math:`F_a` lets a component rotate or reverse the response, which same-state conditioning cannot
    (its whitened Jacobian is positive definite). Read as a transition, a component is one regime of a switching
    linear-Gaussian step with Bayesian regime selection, collapsed to one Gaussian.

    Scope (enforced by :class:`vfe3.config.VFE3Config`): reference coordinates on ``gaussian_full``, so the block reads the
    dense invariant :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}` and its covariance returns as :math:`U_iC_i^+U_i^\top`;
    exact inference only, covariance contract ``'posterior'`` only, no dropout and no EM term.

    Parameters, in registration order: ``nu``, ``log_lambda`` (M, K), ``log_pi`` (M,), ``omega``, ``log_result_noise``
    (M, K), then ``regression_offset`` (M, K, K) when ``regression_rank`` is None, or ``regression_left`` and
    ``regression_right`` (M, K, r). Initialization: ``nu``, ``log_lambda`` and ``log_pi`` as in
    :class:`PrototypeAgentBlockMLP`, ``omega = nu``, :math:`Q_a = \rho L_a` with :math:`\rho` = ``result_noise_ratio``,
    ``regression_offset = 0`` or ``regression_left = 0`` and ``regression_right`` normal with standard deviation
    :math:`0.01/\sqrt K`, so :math:`F_a = I` with nonzero initial gradients for both factors. ``log_lambda``, ``log_pi``
    and ``log_result_noise`` train without weight decay (:func:`vfe3.train.build_optimizer`); decay on the regression
    parameters pulls :math:`F_a` toward the identity.
    """

    covariance_contract = "posterior"

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        dropout: float,
        *,
        covariance_contract: str = "posterior",
        inference: str = "exact",
        template_init_variance: float = 10.0,
        regression_rank: Optional[int] = None,
        result_noise_ratio: float = 0.01,
    ) -> None:
        super().__init__()
        if covariance_contract != "posterior":
            raise ValueError(f"cue_result_memory supports only covariance_contract='posterior', got {covariance_contract!r}")
        if inference != "exact":
            raise ValueError(f"cue_result_memory supports only inference='exact', got {inference!r}")
        if float(dropout) != 0.0:
            raise ValueError(f"cue_result_memory takes no dropout (the output is a posterior), got {dropout!r}")
        if (isinstance(template_init_variance, bool) or not isinstance(template_init_variance, (int, float))
                or not math.isfinite(float(template_init_variance)) or float(template_init_variance) <= 0.0):
            raise ValueError(f"template_init_variance must be a finite real value > 0, got {template_init_variance!r}")
        if regression_rank is not None and (isinstance(regression_rank, bool) or not isinstance(regression_rank, int)
                                            or not 1 <= regression_rank <= embed_dim):
            raise ValueError(f"regression_rank must be None or an int in [1, {embed_dim}], got {regression_rank!r}")
        if (isinstance(result_noise_ratio, bool) or not isinstance(result_noise_ratio, (int, float))
                or not math.isfinite(float(result_noise_ratio)) or float(result_noise_ratio) <= 0.0):
            raise ValueError(f"result_noise_ratio must be a finite real value > 0, got {result_noise_ratio!r}")
        M, K = embed_dim * expansion, embed_dim
        self.regression_rank = regression_rank
        self.result_noise_ratio = float(result_noise_ratio)
        self.tile_bytes = CUE_RESULT_TILE_BYTES                                    # chunk budget (not state)
        bound_in = embed_dim ** -0.5
        self.nu = nn.Parameter(torch.empty(M, K).uniform_(-bound_in, bound_in))                     # (M, K)
        self.log_lambda = nn.Parameter(torch.full((M, K), math.log(float(template_init_variance))))  # (M, K)
        self.log_pi = nn.Parameter(torch.zeros(M))                                                  # (M,)
        self.omega = nn.Parameter(self.nu.detach().clone())                                        # (M, K)
        self.log_result_noise = nn.Parameter(self.log_lambda.detach() + math.log(self.result_noise_ratio))  # (M, K)
        if regression_rank is None:
            self.regression_offset = nn.Parameter(torch.zeros(M, K, K))                             # (M, K, K) D_a
        else:
            self.regression_left = nn.Parameter(torch.zeros(M, K, regression_rank))                 # (M, K, r) U_a
            self.regression_right = nn.Parameter(
                torch.randn(M, K, regression_rank) * (0.01 / math.sqrt(K)))                         # (M, K, r) V_a

    def undecayed_parameters(self) -> Tuple[nn.Parameter, ...]:
        r"""Log variances and log weights: decay would pull them toward 1 and uniform, a prior, not a penalty."""
        return (self.log_lambda, self.log_pi, self.log_result_noise)

    def _regression(self) -> Dict[str, torch.Tensor]:
        if self.regression_rank is None:
            return {"regression_offset": self.regression_offset}
        return {"regression_left": self.regression_left, "regression_right": self.regression_right}

    def posterior(
        self,
        mu:    torch.Tensor,                          # (..., K) record values (reference working means)
        sigma: torch.Tensor,                          # (..., K, K) dense record covariance
    ) -> CueResultPosterior:
        r"""The exact result posterior at the working belief (:func:`vfe3.model.cue_result_memory.cue_result_exact_posterior`)."""
        return cue_result_exact_posterior(mu, sigma, self.nu, self.omega, self.log_lambda, self.log_result_noise,
                                          self.log_pi, tile_bytes=self.tile_bytes, **self._regression())

    @torch.no_grad()
    def calibrate_templates(
        self,
        mu:        torch.Tensor,                      # (..., K) working means this block receives
        sigma:     torch.Tensor,                      # (..., K, K) dense working covariance
        *,
        pull:      float,                             # target initial pull p in (0, 1)
        generator: torch.Generator,                   # CPU generator for the template draw
    ) -> Dict[str, float]:
        r"""``block_mlp_template_init='working_variance'``: the cue bank as in :meth:`PrototypeAgentBlockMLP.calibrate_templates`
        (same draw), then :math:`\omega = \nu` and :math:`Q_a = \rho L_a`, so the calibrated block still starts at
        :math:`F_a = I` with the result centered on the cue."""
        K = self.nu.shape[-1]
        diag = sigma.diagonal(dim1=-2, dim2=-1)
        c_bar = float(diag.double().mean())
        lam0 = c_bar * (1.0 - pull) / pull
        rows = mu.reshape(-1, K)
        M, n = self.nu.shape[0], rows.shape[0]
        index = (torch.randperm(n, generator=generator)[:M] if n >= M
                 else torch.randint(0, n, (M,), generator=generator))
        self.nu.copy_(rows[index.to(rows.device)].to(self.nu.dtype))
        self.log_lambda.fill_(math.log(lam0))
        self.omega.copy_(self.nu)
        self.log_result_noise.copy_(self.log_lambda + math.log(self.result_noise_ratio))
        return {"working_variance": c_bar, "lambda0": lam0}

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
        em_rows: Optional[list[torch.Tensor]] = None,
    ) -> BlockMLPMomentResult:
        r""":math:`(\xi, C) \mapsto` the result posterior moments :math:`(\bar m_Y, \bar P_Y)`."""
        del frame_context
        _reject_hidden_bias(hidden_bias, "the cue_result_memory block conditions on components and has no fc1 hidden layer")
        if em_rows is not None:
            raise ValueError("cue_result_memory has no EM objective (block_mlp_em_weight must be 0)")
        if sigma.dim() != mu.dim() + 1:
            raise ValueError("cue_result_memory reads a dense (..., K, K) covariance (reference coordinates on "
                             f"gaussian_full); got sigma {tuple(sigma.shape)} for mu {tuple(mu.shape)}")
        result = self.posterior(mu, sigma)
        return BlockMLPMomentResult(result.mean.to(mu.dtype), result.covariance.to(sigma.dtype), None)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.forward_moments(mu, sigma).mu


BlockMLPFactory = Callable[..., nn.Module]
BlockMLPReportLabel = Callable[[str], str]
BlockMLPWidth = Callable[[int, Sequence[int]], int]
BlockMLPExtraFlops = Callable[[int, Sequence[int]], float]


@dataclass(frozen=True)
class BlockMLPRegistration:
    """Factory and every structural capability owned by one BlockMLP mode."""

    factory: BlockMLPFactory
    requires_frame_context: bool
    covariance_kinds: frozenset[str]
    gauge_class: str
    intertwiner_compatible: bool
    width: BlockMLPWidth
    extra_flops_per_layer: BlockMLPExtraFlops
    report_label: BlockMLPReportLabel
    # (width, expansion) -> per-layer count, for modes whose parameters are not a two-layer MLP.
    parameter_count_per_layer: Optional[Callable[[int, int], int]] = None
    # (width, expansion) -> per-layer matmul FLOPs replacing the two-layer 4 * expansion * width^2.
    matmul_flops_per_layer: Optional[Callable[[int, int], float]] = None
    # extra_flops_per_layer for the diagonal covariance layout (..., K), for modes whose extra cost
    # depends on the layout; None means the layout does not change the cost.
    diagonal_extra_flops_per_layer: Optional[BlockMLPExtraFlops] = None
    # Keyword options beyond the shared construction arguments that this mode's factory reads;
    # build_block_mlp forwards only these, so every other factory is called exactly as before.
    factory_options: frozenset[str] = frozenset()
    # (width, expansion, regression_rank) -> per-layer count, for modes whose count depends on the regression rank
    # (cue_result_memory); takes precedence over parameter_count_per_layer.
    rank_parameter_count_per_layer: Optional[Callable[[int, int, Optional[int]], int]] = None

    def parameter_count(
        self,
        *,
        embed_dim: int,
        irrep_dims: Sequence[int],
        expansion: int,
        n_layers: int = 1,
        regression_rank: Optional[int] = None,
    ) -> int:
        width = self.width(embed_dim, irrep_dims)
        if self.rank_parameter_count_per_layer is not None:
            return int(n_layers) * self.rank_parameter_count_per_layer(width, expansion, regression_rank)
        if self.parameter_count_per_layer is not None:
            return int(n_layers) * self.parameter_count_per_layer(width, expansion)
        per_layer = 2 * expansion * width * width + (expansion + 1) * width
        return int(n_layers) * per_layer

    def flops_per_token(
        self,
        *,
        embed_dim: int,
        irrep_dims: Sequence[int],
        expansion: int,
        covariance_contract: str,
        n_layers: int,
        covariance_layout: str = "full",
    ) -> float:
        width = self.width(embed_dim, irrep_dims)
        per_layer = (
            self.matmul_flops_per_layer(width, expansion)
            if self.matmul_flops_per_layer is not None
            else 4.0 * expansion * width * width
        )
        layout = _validate_covariance_layout(covariance_layout)
        extra_flops = self.extra_flops_per_layer
        if layout == "diagonal" and self.diagonal_extra_flops_per_layer is not None:
            extra_flops = self.diagonal_extra_flops_per_layer
        per_layer +=extra_flops(embed_dim, irrep_dims)
        if covariance_contract == "delta_full" and layout == "diagonal":
            # The diagonal (J o J) sigma: the squares J_kl^2 (K^2) and the product with sigma (2 K^2), plus the floor
            # f D^2 sigma (4 K); no Cholesky.
            per_layer += 3.0 * embed_dim ** 2 + 4.0 * embed_dim
        elif covariance_contract == "delta_full":
            # Cholesky of Sigma (K^3/3), the factor J L (2 K^3), its Gram (J L)(J L)^T (2 K^3) and the
            # certifying Cholesky of the result (K^3/3); a float64 recompute of failed matrices is not booked.
            per_layer += (4.0 + 2.0 / 3.0) * embed_dim ** 3
        elif covariance_contract == "posterior" and layout == "full":
            # prototype_agents, the only mode admitting 'posterior': P = D^{-1/2} B^{-1} D^{1/2} C is a
            # Cholesky solve with K right-hand sides (2 K^3) plus two diagonal scalings and the
            # symmetrization (4 K^2). On the diagonal layout P is already formed for the mean update.
            per_layer += 2.0 * embed_dim ** 3 + 4.0 * embed_dim ** 2
        return float(n_layers) * per_layer


def _coordinate_factory(**kwargs) -> nn.Module:
    kwargs.pop("irrep_dims")
    return BlockMLP(**kwargs)


def _gauge_gate_factory(**kwargs) -> nn.Module:
    kwargs.pop("embed_dim")
    return GaugeGateBlockMLP(**kwargs)


def _canonical_frame_factory(**kwargs) -> nn.Module:
    kwargs.pop("irrep_dims")
    return CanonicalFrameBlockMLP(**kwargs)


def _gaussian_template_factory(**kwargs) -> nn.Module:
    for unused in ("irrep_dims", "activation", "covariance_floor"):
        kwargs.pop(unused)
    return GaussianTemplateBlockMLP(**kwargs)


def _prototype_agents_factory(**kwargs) -> nn.Module:
    for unused in ("irrep_dims", "activation", "covariance_floor"):
        kwargs.pop(unused)
    return PrototypeAgentBlockMLP(**kwargs)


def _cue_result_memory_factory(**kwargs) -> nn.Module:
    for unused in ("irrep_dims", "activation", "covariance_floor"):
        kwargs.pop(unused)
    return CueResultMemoryBlockMLP(**kwargs)


def _cue_result_parameter_count(width: int, expansion: int, regression_rank: Optional[int]) -> int:
    M = expansion * width
    regression = M * width * width if regression_rank is None else 2 * M * width * int(regression_rank)
    return 4 * M * width + M + regression          # nu, log_lambda, omega, log_result_noise; log_pi; F_a


def _coordinate_report_label(covariance: str) -> str:
    return (
        "coordinate_mean_only_nonintertwiner"
        if covariance == "passthrough"
        else "coordinate_delta_full_covariant_floor_nonintertwiner"
    )


def _prototype_agents_report_label(covariance: str) -> str:
    return (
        "prototype_agents_mean_only_nonintertwiner"
        if covariance == "passthrough"
        else "prototype_agents_posterior_covariance_nonintertwiner"
    )


BLOCK_MLP_REGISTRATIONS: Mapping[str, BlockMLPRegistration] = MappingProxyType({
    "coordinate": BlockMLPRegistration(
        factory=_coordinate_factory,
        requires_frame_context=False,
        covariance_kinds=_BLOCK_MLP_COVARIANCE_CONTRACTS,
        gauge_class="coordinate_nonintertwiner",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda _embed_dim, _irrep_dims: 0.0,
        report_label=_coordinate_report_label,
    ),
    "gauge_gate": BlockMLPRegistration(
        factory=_gauge_gate_factory,
        requires_frame_context=False,
        covariance_kinds=_BLOCK_MLP_COVARIANCE_CONTRACTS,
        gauge_class="invariant_scalar_intertwiner",
        intertwiner_compatible=True,
        width=lambda _embed_dim, irrep_dims: len(irrep_dims),
        extra_flops_per_layer=lambda _embed_dim, irrep_dims: float(sum(
            2.0 * dim ** 3 + 2.0 * dim ** 2 for dim in irrep_dims
        )),
        report_label=lambda _covariance: "invariant_scalar_gate",
        # Diagonal layout: s_b = sum_k mu_k * (mu_k / sigma_k), a divide, a multiply and an add per
        # coordinate, in place of the per-block Cholesky solve.
        diagonal_extra_flops_per_layer=lambda _embed_dim, irrep_dims: 3.0 * float(sum(irrep_dims)),
    ),
    "canonical_frame": BlockMLPRegistration(
        factory=_canonical_frame_factory,
        requires_frame_context=True,
        covariance_kinds=_BLOCK_MLP_COVARIANCE_CONTRACTS,
        gauge_class="left_equivariant_right_fixed",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda embed_dim, _irrep_dims: float(4 * embed_dim ** 2),
        report_label=lambda _covariance: "canonical_frame_left_equivariant_right_fixed",
    ),
    # nu, log_lambda, values: (M, K) each; log_pi: (M,); M = expansion * K.
    "gaussian_template": BlockMLPRegistration(
        factory=_gaussian_template_factory,
        requires_frame_context=False,
        covariance_kinds=frozenset({"passthrough"}),
        gauge_class="coordinate_nonintertwiner",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda _embed_dim, _irrep_dims: 0.0,
        report_label=lambda _covariance: "gaussian_template_mean_only_nonintertwiner",
        parameter_count_per_layer=lambda width, expansion: (
            3 * expansion * width * width + expansion * width),
        matmul_flops_per_layer=lambda width, expansion: 6.0 * expansion * width * width,
    ),
    # nu, log_lambda: (M, K) each; log_pi: (M,); M = expansion * K. The FLOP metadata is the mean-field
    # path in local coordinates: four (K x M) matmuls (two for the label logits, D and eta), plus a dense
    # K x K Cholesky and three K^2 products per token on the full layout, or about 7 K elementwise
    # operations on the diagonal layout; flops_per_token adds 2 K^3 + 4 K^2 for the posterior covariance on
    # the full layout. It does not cover the exact path or reference coordinates (block or dense working
    # covariance on a diagonal family).
    "prototype_agents": BlockMLPRegistration(
        factory=_prototype_agents_factory,
        requires_frame_context=False,
        covariance_kinds=frozenset(PROTOTYPE_COVARIANCE_CONTRACTS),
        gauge_class="coordinate_nonintertwiner",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda embed_dim, _irrep_dims: embed_dim ** 3 / 3.0 + 6.0 * embed_dim ** 2,
        report_label=_prototype_agents_report_label,
        parameter_count_per_layer=lambda width, expansion: (
            2 * expansion * width * width + expansion * width),
        matmul_flops_per_layer=lambda width, expansion: 8.0 * expansion * width * width,
        diagonal_extra_flops_per_layer=lambda embed_dim, _irrep_dims: 7.0 * embed_dim,
        factory_options=frozenset({"inference", "template_init_variance", "cavi_steps"}),
    ),
    # nu, log_lambda, omega, log_result_noise: (M, K) each; log_pi: (M,); regression_offset (M, K, K) or
    # regression_left/right (M, K, r); M = expansion * K. Exact inference in reference coordinates on a dense C only.
    # The FLOP metadata is the dense-regression exact path per token: per component a K x K Cholesky (K^3/3), the
    # solve W^-1 L (2 K^3), C W^-1 L (2 K^3) and F P F^T (4 K^3), about 25/3 K^3; the low-rank form is cheaper and
    # is booked at the same bound.
    "cue_result_memory": BlockMLPRegistration(
        factory=_cue_result_memory_factory,
        requires_frame_context=False,
        covariance_kinds=frozenset({"posterior"}),
        gauge_class="coordinate_nonintertwiner",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda _embed_dim, _irrep_dims: 0.0,
        report_label=lambda _covariance: "cue_result_memory_posterior_covariance_nonintertwiner",
        matmul_flops_per_layer=lambda width, expansion: (25.0 / 3.0) * expansion * width ** 4,
        factory_options=frozenset({"inference", "template_init_variance", "regression_rank", "result_noise_ratio"}),
        rank_parameter_count_per_layer=_cue_result_parameter_count,
    ),
})


def get_block_mlp_registration(mode: str) -> BlockMLPRegistration:
    try:
        return BLOCK_MLP_REGISTRATIONS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown block_mlp_mode {mode!r}; available: {sorted(BLOCK_MLP_REGISTRATIONS)}"
        ) from exc


def block_mlp_build_metadata(
    *,
    enabled: bool,
    mode: str,
    covariance: str,
    expansion: int,
    activation: str,
    dropout: float,
    covariance_floor: float,
    embed_dim: int,
    irrep_dims: Sequence[int],
    n_layers: int,
    covariance_layout: str = "full",
    regression_rank: Optional[int] = None,
    tied_layers: bool = False,
) -> BlockMLPBuildMetadata:
    r"""Executable build record; ``covariance_layout`` ('full' | 'diagonal') selects layout-dependent FLOPs,
    ``regression_rank`` the cue_result_memory parameter count (other modes ignore it), and ``tied_layers`` (one
    module shared by every layer) counts the parameters once while the FLOPs stay per layer."""
    registration = get_block_mlp_registration(mode)
    if covariance not in registration.covariance_kinds:
        raise ValueError(f"unknown BlockMLP covariance contract {covariance!r}")
    realized_irrep_dims = tuple(int(dim) for dim in irrep_dims)
    realized_parameter_count = registration.parameter_count(
        embed_dim=embed_dim,
        irrep_dims=realized_irrep_dims,
        expansion=expansion,
        n_layers=1 if tied_layers else n_layers,
        regression_rank=regression_rank,
    )
    realized_flops_per_token = registration.flops_per_token(
        embed_dim=embed_dim,
        irrep_dims=realized_irrep_dims,
        expansion=expansion,
        covariance_contract=covariance,
        n_layers=n_layers,
        covariance_layout=covariance_layout,
    )
    covariance_report = (
        "not_applicable" if not enabled else
        "delta_full_plus_covariant_floor" if covariance == "delta_full" else covariance
    )
    return BlockMLPBuildMetadata(
        enabled=bool(enabled),
        mode=mode,
        covariance=covariance,
        expansion=int(expansion),
        activation=activation,
        dropout=float(dropout),
        covariance_floor=float(covariance_floor),
        embed_dim=int(embed_dim),
        irrep_dims=realized_irrep_dims,
        n_layers=int(n_layers),
        parameter_count=(realized_parameter_count if enabled else 0),
        flops_per_token=(realized_flops_per_token if enabled else 0.0),
        report_label=(registration.report_label(covariance) if enabled else "disabled"),
        covariance_report_label=covariance_report,
        gauge_class=(registration.gauge_class if enabled else "disabled"),
        intertwiner_compatible=(registration.intertwiner_compatible if enabled else True),
    )


def build_block_mlp(
    mode: str,
    *,
    embed_dim: int,
    irrep_dims: Sequence[int],
    expansion: int,
    activation: str,
    dropout: float,
    covariance_contract: str,
    covariance_floor: float,
    inference: str = "mean_field",
    template_init_variance: float = 10.0,
    cavi_steps: int = 1,
    regression_rank: Optional[int] = None,
    result_noise_ratio: float = 0.01,
) -> nn.Module:
    r"""Build one block MLP; ``inference``, ``template_init_variance``, ``cavi_steps``, ``regression_rank`` and
    ``result_noise_ratio`` reach only the modes that read them."""
    registration = get_block_mlp_registration(mode)
    if covariance_contract not in registration.covariance_kinds:
        raise ValueError(f"unknown BlockMLP covariance contract {covariance_contract!r}")
    options = {"inference": inference, "template_init_variance": template_init_variance, "cavi_steps": cavi_steps,
               "regression_rank": regression_rank, "result_noise_ratio": result_noise_ratio}
    return registration.factory(
        embed_dim=embed_dim,
        irrep_dims=irrep_dims,
        expansion=expansion,
        activation=activation,
        dropout=dropout,
        covariance_contract=covariance_contract,
        covariance_floor=covariance_floor,
        **{name: value for name, value in options.items() if name in registration.factory_options},
    )
