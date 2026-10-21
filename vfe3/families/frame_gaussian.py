r"""The frame-diagonal Gaussian family ``gaussian_frame_diagonal`` (revived 2026-10-18; audit 2026-10-17 item 3).

A belief of token :math:`i` is :math:`q_i = \mathcal N(\mu_i, \Sigma_i)` in the token's local coordinates with

.. math::
    \Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top,\qquad
    C_i = U_i^{-1}\Sigma_iU_i^{-\top} = \operatorname{diag}(c_i),

so the covariance is diagonal in REFERENCE coordinates, where :math:`U_i` is the realized frame of the token (the
relational frame of the projected routes). The model stores the local mean :math:`\mu_i` in the ``mu`` slot, as
every family does, and the reference variances :math:`c_i` in the ``sigma`` slot, ``(..., N, K)``. Local quantities
are derived only where a consumer needs them (:mod:`vfe3.model.frame_actions`): the local diagonal
:math:`(\Sigma_i)_{kk} = \sum_m (U_i)_{km}^2c_{i,m}` for a local-coordinate decoder, and the dense
:math:`U_i\operatorname{diag}(c_i)U_i^\top` for a local-coordinate covariance pushforward.

Under the passive action :math:`(\mu_i, \Sigma_i, U_i) \mapsto (g_i\mu_i, g_i\Sigma_ig_i^\top, g_iU_i)` the stored
:math:`c_i` is invariant, so the family is closed under the action of every structure group, and averaging is convex
in it: the moment-matched readout's M-projection onto the family is :math:`c_i^+ = \operatorname{diag}C_i^{\mathrm{att}}`
in reference coordinates, which does not depend on the frame. The local-diagonal family ``gaussian_diagonal`` instead
projects onto :math:`\operatorname{diag}(U_iC_i^{\mathrm{att}}U_i^\top)`, which is not closed under the action and
inflates :math:`\operatorname{tr}C` by up to about :math:`\kappa(U_i)^2` per layer on ill-conditioned frames.

Parameter objects of this class hold the REFERENCE moments :math:`(\xi_i, c_i)`, :math:`\xi_i = U_i^{-1}\mu_i`. Two
beliefs that share a frame (a token's belief and its prior, or :math:`q_i` against the transported
:math:`\Omega_{ij}q_j` with :math:`\Omega_{ij} = U_iU_j^{-1}`, whose covariance is :math:`U_i\operatorname{diag}(c_j)U_i^\top`)
then have the KL of the diagonal family, since a KL is invariant under a common invertible map:

.. math::
    \mathrm{KL}(q_i\|p_i) = \tfrac12\sum_k\Bigl[\frac{c_{q,k} + (\xi_{p,k} - \xi_{q,k})^2}{c_{p,k}} - 1
        + \log\frac{c_{p,k}}{c_{q,k}}\Bigr].

All divergence, natural-parameter and Fisher maps are therefore :class:`~vfe3.families.gaussian.DiagonalGaussian`'s,
evaluated on reference moments. A caller holding local means converts them first. Two beliefs in DIFFERENT frames
have the closed form :func:`frame_diagonal_kl`.
"""

from typing import Dict, List

import torch

from vfe3.families.base import register_family, safe_kl_clamp
from vfe3.families.gaussian import DiagonalGaussian


@register_family("gaussian_frame_diagonal")
class FrameDiagonalGaussian(DiagonalGaussian):
    r"""Frame-diagonal Gaussian in its reference moments: mu (..., K) holds :math:`\xi = U^{-1}\mu`, sigma (..., K)
    holds :math:`c = \operatorname{diag}(U^{-1}\Sigma U^{-\top})`.

    ``cov_kind`` is ``'diagonal'`` (the stored layout); ``frame_diagonal_covariance`` declares that the diagonal is in
    reference coordinates, which every consumer that reads a local covariance must honor (config validation scopes
    the family to the routes that do).
    """

    frame_diagonal_covariance = True

    def block(self, start: int, end: int) -> "FrameDiagonalGaussian":
        return FrameDiagonalGaussian(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "FrameDiagonalGaussian":
        return FrameDiagonalGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    @classmethod
    def stack(cls, parts: List["DiagonalGaussian"], *, dim: int = 0) -> "FrameDiagonalGaussian":
        return FrameDiagonalGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    @classmethod
    def from_covariance(cls, mu: torch.Tensor, covariance: torch.Tensor) -> "FrameDiagonalGaussian":
        r"""Read a REFERENCE-coordinate mean and full covariance as the family's M-projection :math:`(\xi, \operatorname{diag}C)`.

        Without a frame there is nothing to co-transform, so under a frameless congruence :math:`C \to gCg^\top`
        this readout is not invariant (``check_admissible`` reports it, as for the local diagonal family); the
        family's invariance comes from the frame, :math:`U \to gU`, which leaves :math:`c` fixed.
        """
        return cls(mu, torch.diagonal(covariance, dim1=-2, dim2=-1))

    @classmethod
    def diagnostic_labels(cls) -> Dict[str, str]:
        return {
            "dispersion":             "reference-coordinate Gaussian variance",
            "covariance_spectrum":    "reference-coordinate variance",
            "half_mean_fisher_trace": r"Half reference Fisher trace $\frac{1}{2}\sum_k c_k^{-1}$",
        }


def frame_diagonal_kl(
    mu_q: torch.Tensor,                # (..., K) local mean of q
    c_q:  torch.Tensor,                # (..., K) reference variances of q, Sigma_q = U_q diag(c_q) U_q^T
    U_q:  torch.Tensor,                # (..., K, K) dense frame of q
    mu_p: torch.Tensor,                # (..., K) local mean of p
    c_p:  torch.Tensor,                # (..., K) reference variances of p, Sigma_p = U_p diag(c_p) U_p^T
    U_p:  torch.Tensor,                # (..., K, K) dense frame of p

    *,
    kl_max: float = float("inf"),
) -> torch.Tensor:                     # (...,) KL(q || p)
    r"""Exact KL between two frame-diagonal Gaussians in possibly different frames.

    Test oracle: no production code calls it; the live routes compare beliefs that share a frame and use the
    diagonal KL of their reference moments.

    With :math:`W = U_p^{-1}U_q` and :math:`\delta = U_p^{-1}(\mu_q - \mu_p)` (both in :math:`p`'s reference
    coordinates, where :math:`\Sigma_p` is :math:`\operatorname{diag}(c_p)` and :math:`\Sigma_q` is
    :math:`W\operatorname{diag}(c_q)W^\top`),

    .. math::
        \mathrm{KL}(q\|p) = \tfrac12\Bigl[\sum_k\frac{\sum_m W_{km}^2c_{q,m} + \delta_k^2}{c_{p,k}} - K
            + \sum_k\log c_{p,k} - \sum_k\log c_{q,k} - 2\log\lvert\det W\rvert\Bigr].

    In a shared frame (:math:`W = I`) it is the diagonal KL of the reference moments. Computed in the inputs'
    precision without a variance floor; ``kl_max`` (default off) clamps as :func:`~vfe3.families.base.safe_kl_clamp`.
    """
    K = mu_q.shape[-1]
    W = torch.linalg.solve(U_p, U_q)                                              # (..., K, K) U_p^{-1} U_q
    delta = torch.linalg.solve(U_p, (mu_q - mu_p).unsqueeze(-1)).squeeze(-1)    # (..., K)
    trace = (W.square() @ c_q.unsqueeze(-1)).squeeze(-1)                         # (..., K) diag(W diag(c_q) W^T)
    logabsdet = torch.linalg.slogdet(W)[1]                                       # (...,)
    div = 0.5 * (((trace + delta.square()) / c_p).sum(dim=-1) - K
                 + torch.log(c_p).sum(dim=-1) - torch.log(c_q).sum(dim=-1) - 2.0 * logabsdet)
    return safe_kl_clamp(div, kl_max=kl_max)
