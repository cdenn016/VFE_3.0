r"""Frame-intrinsic Gaussian family for VFE_3.0 (default OFF; opt in via ``family``).

``gaussian_diagonal`` stores a covariance that is diagonal in the FIXED basis. The transport is
a congruence, and diagonal covariance is not closed under a general :math:`\mathrm{GL}(K)`
congruence, so :func:`vfe3.geometry.transport.transport_covariance` forms the sandwich and keeps
only its diagonal. The discarded off-diagonal mass grows with the conditioning of the frames.

This family declares the covariance diagonal in the agent's OWN fiber frame instead,

.. math::
    \Sigma_i = U_i \operatorname{diag}(\sigma_i) U_i^{\top},

which IS closed under the frame action. For the pure Regime-I coboundary
:math:`\Omega_{ij} = U_i U_j^{-1}` the sender frame cancels from the congruence and the receiver
frame cancels from the location,

.. math::
    \Omega_{ij}\Sigma_j\Omega_{ij}^{\top}
        = U_i \operatorname{diag}(\sigma_j) U_i^{\top},
    \qquad
    U_i^{-1}\Omega_{ij}\mu_j = U_j^{-1}\mu_j = a_j,

so, writing :math:`a = U^{-1}\mu` for the frame-intrinsic location and using invariance of a
divergence under a common invertible pushforward, the edge energy is EXACTLY the diagonal
divergence between :math:`(a_i, \sigma_i)` and :math:`(a_j, \sigma_j)`:

.. math::
    D\bigl(q_i \,\|\, \Omega_{ij\#}q_j\bigr)
        = D_{\mathrm{diag}}\bigl((a_i, \sigma_i) \,\|\, (a_j, \sigma_j)\bigr).

Both transports therefore reduce to a broadcast over the query axis, and the pair energy is the
exact full-covariance divergence evaluated at diagonal cost with no sandwich materialized.

CONSEQUENCE, stated plainly: the relative frame :math:`\Omega_{ij}` cancels out of the pair
energy. Under this family the gauge no longer shapes the belief coupling; it acts only through
each agent's own frame-intrinsic location. That is the coboundary triviality of Regime I
(``vfe4_whitepaper/03_bundle_geometry.tex``, ``eq:regime-i-loop-holonomy``) made explicit, and it
is a genuinely DIFFERENT model rather than a numerical refinement of ``gaussian_diagonal``. It is
default OFF for that reason. The parameters stored in the ``mu`` slot are read as the
frame-intrinsic coordinates :math:`a`, not fixed-basis means.

The exact-congruence variant, which removes the same diagonal truncation while keeping the gauge
ACTIVE in the energy (it pulls the query back by :math:`\Omega^{-1}` instead of cancelling the
frame), lives in :mod:`vfe3.families.exact_congruence` as ``family="gaussian_diagonal_exact"``.
"""

from typing import List, Optional

import torch

from vfe3.families.base import register_family
from vfe3.families.gaussian import DiagonalGaussian


def _broadcast_over_queries(
    parameter: torch.Tensor,             # (..., N, K) per-agent parameter
) -> torch.Tensor:                       # (..., N, N, K) query-broadcast view
    r"""Insert the query axis so entry ``[..., i, j, :]`` is ``parameter[..., j, :]``.

    Returns an expanded VIEW (no copy): the frame-intrinsic transport is the identity in the
    receiver frame, so the transported key parameter does not depend on the query index ``i``.
    """
    n_agents = parameter.shape[-2]
    return parameter.unsqueeze(-3).expand(
        *parameter.shape[:-2], n_agents, n_agents, parameter.shape[-1]
    )


@register_family("gaussian_frame_diagonal")
class FrameDiagonalGaussian(DiagonalGaussian):
    r"""Diagonal Gaussian in the agent's own fiber frame: a (..., K), sigma (..., K).

    Inherits every divergence, natural-parameter and log-partition map from
    :class:`~vfe3.families.gaussian.DiagonalGaussian` -- the frame-intrinsic coordinates obey the
    same diagonal exponential-family algebra. Only the two transport seams differ, because the
    frame action cancels from both.
    """

    cov_kind = "diagonal"
    dispersion_is_covariance = True
    # Pointwise algebra is DiagonalGaussian's, unchanged: only the two transport seams are overridden
    # (audit 2026-07-26 E-04).
    gaussian_pointwise_algebra = True
    # ``_require_coboundary`` below is this declaration's runtime half; config validation reads the
    # declaration so a non-flat transport_mode is rejected at construction rather than at the first
    # forward (audit 2026-07-26 E-03).
    transport_requirement = "coboundary"

    def block(self, start: int, end: int) -> "FrameDiagonalGaussian":
        return FrameDiagonalGaussian(self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "FrameDiagonalGaussian":
        return FrameDiagonalGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    @classmethod
    def stack(
        cls,
        parts: List["FrameDiagonalGaussian"],

        *,
        dim:   int = 0,
    ) -> "FrameDiagonalGaussian":
        return FrameDiagonalGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    @staticmethod
    def _require_coboundary(omega: object, seam: str) -> None:
        r"""Fail closed unless the frame cancellation this family is built on actually holds.

        The identity transport below is valid ONLY for a flat coboundary
        :math:`\Omega_{ij} = U_iU_j^{-1}`, where :math:`U_i^{-1}\Omega_{ij}U_j = I`. Any transport
        carrying an edge factor between the frames leaves a residual that is not the identity
        (measured 3.37 at K=4 against 2.4e-07 for flat, audit 2026-07-26 A-01). Returning
        the flat answer there would silently discard the entire connection: the pair energy would come
        out bit-identical to the flat cocycle while the run record reported a different transport as
        active.

        The sibling exact-congruence family guards the same contract; this one did not.
        """
        from vfe3.geometry.transport import _certifies_same_frame_flat_cocycle

        if not _certifies_same_frame_flat_cocycle(omega):
            raise TypeError(
                f"family='gaussian_frame_diagonal' cannot {seam} through transport "
                f"{type(omega).__name__!r}: the frame-intrinsic identity needs a certified "
                "same-frame flat cocycle (Omega_ij = U_i U_j^{-1}), because the receiver frame "
                "cancels only for that coboundary. A query/key-split or otherwise non-coboundary "
                "transport leaves a residual that this family "
                "would silently drop. Use "
                "family='gaussian_diagonal' / 'gaussian_diagonal_exact'.")

    @classmethod
    def transport_location(
        cls,
        mu:    torch.Tensor,             # (..., N, K) frame-intrinsic locations a_j
        omega: object,                   # transport container; must certify a flat cocycle
    ) -> torch.Tensor:                   # (..., N, N, K) transported locations
        r"""Identity transport: :math:`U_i^{-1}\Omega_{ij}\mu_j = U_j^{-1}\mu_j = a_j`.

        ``omega`` is not read for its values -- the receiver frame cancels exactly for the Regime-I
        coboundary -- but it IS checked, because that cancellation is a precondition rather than an
        identity (see :meth:`_require_coboundary`).
        """
        cls._require_coboundary(omega, "transport a location")
        return _broadcast_over_queries(mu)

    @classmethod
    def transport_dispersion(
        cls,
        dispersion:   torch.Tensor,      # (..., N, K) frame-intrinsic variances
        omega:        object,            # transport container; unused, the frame cancels

        *,
        diagonal_out:    Optional[bool]      = None,
        marginal_blocks: Optional[List[int]] = None,
    ) -> torch.Tensor:                   # (..., N, N, K) transported variances
        r"""Identity transport: :math:`\Omega_{ij}\Sigma_j\Omega_{ij}^{\top}
        = U_i \operatorname{diag}(\sigma_j) U_i^{\top}`, i.e. ``sigma_j`` in the receiver frame.

        ``diagonal_out`` and ``marginal_blocks`` are accepted for interface symmetry with the base
        seam and are intentionally unused: this family is diagonal by declaration and the sandwich
        is never formed, so there is no cross-head block to skip either. ``omega`` is not read for
        its values but IS checked -- the cancellation is a precondition
        (see :meth:`_require_coboundary`).
        """
        cls._require_coboundary(omega, "transport a dispersion")
        return _broadcast_over_queries(dispersion)
