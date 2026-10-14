r"""Prior-anchored Loewner ceiling of the moment-matched covariance (``predictive_covariance_ceiling``).

The moment-matched readouts already floor their covariance at :math:`f\Sigma_{p,i}`
(:meth:`vfe3.model.relational_operators.RelationalOperators.floor_term`). The ceiling bounds it above by
:math:`c\Sigma_{p,i}` in the Loewner order, so every layer's output lies in the band

.. math::
    f\Sigma_{p,i} \preceq \Sigma_i^+ \preceq c\Sigma_{p,i}.

With :math:`\Sigma_{p,i} = L_iL_i^\top` and the whitened covariance :math:`W_i = L_i^{-1}\Sigma_i^+L_i^{-\top}`
the bound is the spectral clamp

.. math::
    \Sigma_i^+ \mapsto L_iV_i\min(\Lambda_i, c)V_i^\top L_i^\top, \qquad W_i = V_i\Lambda_iV_i^\top.

It involves no frame, so like the floor it is exactly invariant under a common reference change, and it bounds
:math:`\kappa(\Sigma_i^+) \le (c/f)\kappa(\Sigma_{p,i})`. It is a clamp: wherever it binds it changes the
moment-matched kernel, so it is off by default (``None``) and the clamped entries are counted
(:func:`covariance_ceiling_elements`).

The clamp is differentiated by the Daleckii--Krein formula for the matrix function :math:`g(\lambda) = \min(\lambda, c)`,

.. math::
    \bar W = V\bigl(F \circ V^\top \operatorname{sym}(\bar G) V\bigr)V^\top, \qquad
    F_{ab} = \frac{g(\lambda_a) - g(\lambda_b)}{\lambda_a - \lambda_b}, \quad F_{aa} = g'(\lambda_a),

which stays finite on the degenerate eigenvalues the clamp creates, where the ``eigh`` backward divides by
:math:`\lambda_a - \lambda_b` (Bhatia, *Matrix Analysis*, Section V.3). Since :math:`g` is monotone and 1-Lipschitz,
:math:`F_{ab} \in [0, 1]`.
"""
from typing import Dict, Optional, Sequence

import torch


# Entries (tokens, or token head blocks under head_dims) whose covariance the ceiling clamped, per device, counted on
# the device; read by the training loop through covariance_ceiling_elements().
_CEILING_COUNTS: Dict[str, torch.Tensor] = {}


def _count(clamped: torch.Tensor) -> None:
    key = str(clamped.device)
    total = clamped.sum()
    _CEILING_COUNTS[key] = total if key not in _CEILING_COUNTS else _CEILING_COUNTS[key] + total


def covariance_ceiling_elements() -> int:
    r"""Covariance entries clamped by the ceiling since the last reset, over all devices. Host-syncs."""
    return int(sum(int(count) for count in _CEILING_COUNTS.values()))


def reset_covariance_ceiling_elements() -> None:
    r"""Zero the per-device ceiling counters."""
    _CEILING_COUNTS.clear()


class _SpectralCeiling(torch.autograd.Function):
    r""":math:`W \mapsto V\min(\Lambda, c)V^\top` for symmetric :math:`W`, with the Daleckii--Krein backward."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, ceiling: float) -> torch.Tensor:   # (n, r, r) -> (n, r, r)
        lam, V = torch.linalg.eigh(W)                                   # (n, r), (n, r, r)
        g = lam.clamp(max=ceiling)                                      # (n, r)
        ctx.save_for_backward(lam, g, V)
        ctx.ceiling = ceiling
        return (V * g.unsqueeze(-2)) @ V.mT

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        lam, g, V = ctx.saved_tensors
        diff = lam.unsqueeze(-1) - lam.unsqueeze(-2)                    # (n, r, r) lambda_a - lambda_b
        num = g.unsqueeze(-1) - g.unsqueeze(-2)                         # (n, r, r) g(lambda_a) - g(lambda_b)
        slope = (lam < ctx.ceiling).to(lam.dtype)                       # (n, r) g'(lambda)
        tie = diff.abs() <= torch.finfo(lam.dtype).eps * lam.abs().amax(-1, keepdim=True).unsqueeze(-1)
        F = torch.where(tie, 0.5 * (slope.unsqueeze(-1) + slope.unsqueeze(-2)),
                        (num / torch.where(tie, torch.ones_like(diff), diff)).clamp(0.0, 1.0))
        G = 0.5 * (grad + grad.mT)
        return V @ (F * (V.mT @ G @ V)) @ V.mT, None


def _dense_ceiling(
    sigma:   torch.Tensor,   # (..., r, r) SPD covariance
    anchor:  torch.Tensor,   # (..., r, r) SPD anchor Sigma_p, broadcastable to sigma
    ceiling: float,
) -> torch.Tensor:           # (..., r, r), entries above the ceiling clamped
    r"""Clamp the entries of ``sigma`` not certified :math:`\preceq c\Sigma_p` (one host read per call).

    The certificate is a Cholesky test of :math:`c\Sigma_p - \Sigma`; only failing entries are whitened and clamped,
    in float64. A false failure is harmless: the clamp leaves a matrix below the ceiling unchanged up to rounding.
    """
    anchor = torch.broadcast_to(anchor.to(sigma.dtype), sigma.shape)
    with torch.no_grad():
        _, info = torch.linalg.cholesky_ex(ceiling * anchor - sigma)
        clamp = info != 0                                               # (...)
    _count(clamp)
    if not bool(clamp.any()):
        return sigma
    S = sigma[clamp].double()                                           # (n, r, r)
    L = torch.linalg.cholesky(anchor[clamp].double())                   # (n, r, r) Sigma_p = L L^T
    Y = torch.linalg.solve_triangular(L, S, upper=False)                # L^{-1} S
    W = torch.linalg.solve_triangular(L, Y.mT, upper=False)             # L^{-1} S L^{-T}
    W = 0.5 * (W + W.mT)
    clamped = L @ _SpectralCeiling.apply(W, float(ceiling)) @ L.mT      # (n, r, r)
    out = sigma.clone()
    out[clamp] = (0.5 * (clamped + clamped.mT)).to(sigma.dtype)
    return out


def covariance_ceiling(
    sigma:     torch.Tensor,                       # (..., K) diagonal or (..., K, K) covariance Sigma^+
    anchor:    torch.Tensor,                       # sigma's layout (broadcastable): the local prior covariance Sigma_p
    ceiling:   float,                              # c > 0
    *,
    diagonal:  bool,
    head_dims: Optional[Sequence[int]] = None,     # head-block family: clamp each diagonal head block on its own
) -> torch.Tensor:                                 # sigma's shape and dtype
    r"""Loewner ceiling :math:`\Sigma^+ \preceq c\Sigma_p` (module docstring).

    A diagonal layout is clamped elementwise, :math:`\min(\sigma^+, c\sigma_p)`, the same bound for diagonal
    matrices. ``head_dims`` (``gaussian_block``) clamps each diagonal head block against the matching block of the
    anchor and keeps the zero off-head blocks, so the result stays in the head-block family.
    """
    if diagonal:
        bound = ceiling * anchor.to(sigma.dtype)
        with torch.no_grad():
            _count((sigma > bound).any(dim=-1))
        return torch.minimum(sigma, bound)
    if head_dims is None:
        return _dense_ceiling(sigma, anchor, ceiling)
    anchor = torch.broadcast_to(anchor.to(sigma.dtype), sigma.shape)
    out = sigma.clone()
    start = 0
    for d in head_dims:
        block = slice(start, start + d)
        out[..., block, block] = _dense_ceiling(sigma[..., block, block], anchor[..., block, block], ceiling)
        start += d
    return out
