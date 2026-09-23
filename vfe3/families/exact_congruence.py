r"""Exact-congruence diagonal Gaussian for VFE_3.0 (default OFF; opt in via ``family``).

Same DISTRIBUTION family as ``gaussian_diagonal`` -- a diagonal Gaussian in the fixed basis, with
the same self-divergence, natural parameters, decode and retraction. What differs is the
belief-COUPLING energy.

The default path pushes the key FORWARD,
:math:`\Omega_{ij}\Sigma_j\Omega_{ij}^{\top}`, which is a dense congruence that the diagonal family
cannot represent, so :func:`vfe3.geometry.transport.transport_covariance` keeps only its diagonal.
The energy is then the divergence to that PROJECTED key, and the discarded off-diagonal mass grows
with the conditioning of the frames.

Pulling the QUERY BACK instead removes the truncation entirely. A divergence is invariant under a
common invertible pushforward, so applying :math:`\Omega_{ij}^{-1}` to both arguments gives

.. math::
    \mathrm{KL}\bigl(\mathcal{N}(\mu_i, \operatorname{diag}s_i)
        \,\|\, \mathcal{N}(\Omega\mu_j, \Omega\operatorname{diag}(s_j)\Omega^{\top})\bigr)
    = \mathrm{KL}\bigl(\mathcal{N}(\Omega^{-1}\mu_i, \Omega^{-1}\operatorname{diag}(s_i)\Omega^{-\top})
        \,\|\, \mathcal{N}(\mu_j, \operatorname{diag}s_j)\bigr),

and on the right the SECOND argument is exactly diagonal while the first enters only through its
diagonal (the trace term) and its determinant (available in closed form). Writing
:math:`a_{ij} = \Omega_{ij}^{-1}\mu_i` and
:math:`\tilde{s}_{ij} = \operatorname{diag}(\Omega_{ij}^{-1}\operatorname{diag}(s_i)\Omega_{ij}^{-\top})`,

.. math::
    2 E_{ij} = \sum_k \frac{\tilde{s}_{ijk}}{s_{jk}}
             + \sum_k \frac{(\mu_{jk} - a_{ijk})^2}{s_{jk}}
             - K + \sum_k \log s_{jk} - \sum_k \log s_{ik}
             + 2\log\lvert\det\Omega_{ij}\rvert,

which is EXACT and costs exactly what the truncated form costs: one location transport, one
diagonal-of-congruence, and a rank-one determinant difference.

Two routes produce :math:`(a, \tilde{s}, \log\lvert\det\Omega\rvert)`.

FAST (certified flat cocycle, factored or compact vertex factors). For a coboundary
:math:`\Omega_{ij} = U_i U_j^{-1}` the inverse is the index TRANSPOSE,
:math:`\Omega_{ij}^{-1} = \Omega_{ji}`, so the backward-transported query is the forward transport
of the query array read at the swapped pair index. Every existing fast path (block-factored
diagonal sandwich, compact blocks, per-head mean, certified self links) is reused unchanged, and
:math:`\log\lvert\det\Omega_{ij}\rvert = \ell_i - \ell_j` needs only the per-vertex block
log-determinants.

REFERENCE (dense pairwise ``omega``). :math:`\Omega_{ij}^{-1}` is formed explicitly and the pair
log-determinants are taken per block. This assumes nothing about the connection, so it is the
correctness oracle the fast route is pinned against, but it is :math:`O(N^2K^3)` and only sensible
where the caller has already materialized the dense operator. A vertex-factored container that does
NOT certify a same-frame cocycle (a query/key-split frame, e.g. the filtering phi objective's
sender-detached transport) takes this route through its exact dense operator, the product of its
stored query and key factors, because the transpose identity is unavailable for it.

RoPE (audit 2026-09-23 #21). A right-insertion :class:`~vfe3.geometry.transport.RopeTransport`
is scored through its folded vertex frame :math:`V_i = U_i R_i^{\top}` for BOTH moments, i.e. the
coherent :math:`\mathrm{KL}(q_i \| \Omega^{V}_{ij\#} q_j)` with
:math:`\Omega^{V}_{ij} = V_i V_j^{-1}`. The pullback identity requires a single operator, so the
means-only split (mean under :math:`\Omega^V`, covariance under the un-rotated base) has no exact
diagonal form here; the former fast route mixed the two operators and matched neither energy. A
left-insertion wrapper has no vertex frame and still raises.

SCOPE. The identity above is specific to KL: for a Renyi order other than one the blend
:math:`(1-\alpha)\Sigma_1 + \alpha\Sigma_2` mixes a dense covariance with a diagonal one and does
not reduce. A non-KL divergence and a transport that exposes neither a dense operator nor vertex
factors raise rather than silently falling back to the truncated energy.
"""

from typing import List, Optional, Tuple

import torch

from vfe3.families.base import register_family, safe_kl_clamp
from vfe3.families.gaussian import DiagonalGaussian


def _linalg_dtype(dtype: torch.dtype) -> torch.dtype:
    r"""The dtype the LU/determinant routines run in: fp32 at minimum.

    ``torch.linalg`` rejects bfloat16/float16 outright, and the belief transport arrives at the
    autocast dtype under AMP. Promoting only these decompositions is also the numerically right
    call: a ``K x K`` log-determinant in bfloat16 carries ~3 decimal digits, and it enters the
    energy additively rather than as a rounding of it. The surrounding contractions keep whatever
    precision the caller's autocast context gave them, exactly as the default family's do.
    """
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def _slogdet(matrix: torch.Tensor) -> torch.Tensor:   # (...,) log|det| of the trailing matrix axes
    r"""``log|det|`` evaluated, and RETURNED, at fp32 or better.

    Deliberately not cast back to a reduced input dtype: the caller assembles the energy at fp32,
    so a bfloat16 round trip here would discard the precision this promotion exists to buy. fp32
    and fp64 inputs pass through untouched.
    """
    return torch.linalg.slogdet(matrix.to(_linalg_dtype(matrix.dtype)))[1]


def _blocking(
    irrep_dims: Optional[List[int]],      # energy block sizes; None/[K] -> one full-K block
    K:          int,                      # total belief dimension
) -> List[int]:
    r"""Normalize the energy's block partition to an explicit list summing to ``K``."""
    if irrep_dims is None or len(irrep_dims) == 0:
        return [K]
    if sum(irrep_dims) != K:
        raise ValueError(
            f"irrep_dims {list(irrep_dims)} sum to {sum(irrep_dims)}, not the belief dimension {K}")
    return list(irrep_dims)


def _vertex_log_abs_det(
    omega:      object,                   # certified factored/compact flat cocycle
    irrep_dims: List[int],                # energy block partition
    K:          int,                      # total belief dimension
) -> torch.Tensor:                        # (..., N, H) per-vertex per-block log|det U_i|
    r"""Per-vertex block log-determinants :math:`\ell_i^{(h)} = \log\lvert\det U_i^{(h)}\rvert`.

    The pair value follows from the coboundary, :math:`\log\lvert\det\Omega_{ij}^{(h)}\rvert
    = \ell_i^{(h)} - \ell_j^{(h)}`, so no pairwise determinant is ever formed. When the energy is
    computed over one full-K block the per-transport-block values are summed
    (:math:`\det` of a block-diagonal matrix is the product of its blocks).
    """
    from vfe3.geometry.lie_ops import _equal_diag_blocks
    from vfe3.geometry.transport import (
        CompactFactoredTransport,
        FactoredTransport,
        RopeTransport,
    )

    # A gauge-RoPE wrapper rotates by an ORTHOGONAL R (rope.py builds it block-diagonally from 2x2
    # rotations), so |det(R_i Omega_ij R_j^T)| = |det Omega_ij| and the per-vertex ell_i - ell_j
    # identity is unchanged (audit 2026-07-26 E-02). `_pullback_query` now hands this helper the
    # folded frame itself; a wrapper that still reaches here is unwrapped to its base.
    if isinstance(omega, RopeTransport):
        omega = omega.base

    if isinstance(omega, CompactFactoredTransport):
        blocks = omega.exp_blocks                                   # (..., N, H_t, d, d)
        transport_block_dims = [int(blocks.shape[-1])] * int(blocks.shape[-3])
    elif isinstance(omega, FactoredTransport):
        transport_dims = list(omega.irrep_dims)
        if len(set(transport_dims)) != 1:
            raise ValueError(
                "exact-congruence coupling needs equal transport blocks to slice the vertex "
                f"determinants; got irrep_dims={transport_dims}")
        blocks = _equal_diag_blocks(                                # (..., N, H_t, d, d)
            omega.exp_phi, len(transport_dims), transport_dims[0])
        transport_block_dims = list(transport_dims)
    else:
        raise TypeError(f"no vertex factors on transport {type(omega).__name__!r}")

    ell = _slogdet(blocks)                                          # (..., N, H_t)
    if len(irrep_dims) == 1:
        return ell.sum(dim=-1, keepdim=True)                        # (..., N, 1) full-K block
    # Compare the block SIZES, not just how many there are (audit 2026-07-26 D-05). Counting alone
    # let transport blocks [2,2] pair with energy blocks [1,3]: the guard passed, and each block's
    # log-determinant was then matched to the wrong coordinate slice, returning a silently wrong
    # divergence (measured max error 11.5) against this family's fail-closed contract. The dense
    # route already rejected the identical input.
    if list(irrep_dims) != list(transport_block_dims):
        raise ValueError(
            f"exact-congruence coupling needs the energy block partition {list(irrep_dims)} to "
            f"match the transport block partition {list(transport_block_dims)}; mixed partitions "
            "are not supported because each vertex determinant is sliced per block")
    return ell                                                      # (..., N, H)


def _pullback_query(
    mu_q:       torch.Tensor,             # (..., N, K) query locations (the live leaf)
    sigma_q:    torch.Tensor,             # (..., N, K) query variances
    omega:      object,                   # transport container
    irrep_dims: List[int],                # energy block partition
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``(a, s_tilde, log_abs_det)`` for the backward-transported query.

    ``a[..., i, j, :]`` is :math:`\Omega_{ij}^{-1}\mu_i`, ``s_tilde[..., i, j, :]`` is
    :math:`\operatorname{diag}(\Omega_{ij}^{-1}\operatorname{diag}(s_i)\Omega_{ij}^{-\top})`, and
    ``log_abs_det[..., i, j, h]`` is :math:`\log\lvert\det\Omega_{ij}^{(h)}\rvert`.
    """
    from vfe3.geometry.lie_ops import _equal_diag_blocks
    from vfe3.geometry.transport import (
        CompactFactoredTransport,
        FactoredTransport,
        RopeTransport,
        _certifies_same_frame_flat_cocycle,
        transport_covariance,
        transport_mean,
    )

    if isinstance(omega, RopeTransport):
        # Coherent RoPE (audit 2026-09-23 #21): pull BOTH moments back through the one operator the
        # score transports under, the folded frame V_i = U_i R_i^T. The former route pulled the mean
        # back through V and the covariance through the un-rotated base U, an energy that equals
        # neither KL(q_i || Omega^V# q_j) nor the means-only hybrid.
        certified = _certifies_same_frame_flat_cocycle(omega)
        folded = omega.score_operator()
        if folded is None:
            raise TypeError(
                "exact-congruence coupling cannot invert a left-insertion RoPE transport: it has no "
                "vertex frame, so neither the pair-transpose inverse nor a single dense operator is "
                "available. Use rope_insertion='right' or family='gaussian_diagonal'.")
        omega = folded if certified else folded.to_dense_omega()
    if isinstance(omega, (CompactFactoredTransport, FactoredTransport)) and (
            not omega.same_frame_flat_cocycle):
        # A query/key-split vertex frame has no pair-transpose inverse; invert its exact dense
        # operator on the reference route instead of failing (audit 2026-09-23 #6).
        omega = omega.to_dense_omega()

    if isinstance(omega, torch.Tensor):
        # Reference route: invert the dense operator outright, assuming nothing about the connection.
        if omega.shape[-4] != omega.shape[-3]:
            raise ValueError(
                "exact-congruence coupling needs a square token transport, got "
                f"{tuple(omega.shape)}")
        # Keep the inverse at linalg precision (audit 2026-07-26 C-06). Casting it back to the
        # caller's dtype before `inverse ** 2` squares the rounding error: measured s_tilde relative
        # error 9.4e-03 with the cast-back against 2.2e-07 without, and up to 0.235 nats of energy
        # error. `_slogdet` deliberately returns at fp32 for the same reason.
        work = _linalg_dtype(omega.dtype)
        inverse = torch.linalg.inv(omega.to(work))                  # (..., N, N, K, K)
        a = torch.einsum("...ijkl,...il->...ijk", inverse, mu_q.to(work))
        s_tilde = torch.einsum("...ijkl,...il->...ijk", inverse ** 2, sigma_q.to(work))
        if len(irrep_dims) == 1:
            log_abs_det = _slogdet(omega).unsqueeze(-1)                         # (..., N, N, 1)
        else:
            if len(set(irrep_dims)) != 1:
                raise ValueError(
                    "exact-congruence coupling on a dense transport needs equal energy blocks to "
                    f"slice the pair determinants; got irrep_dims={irrep_dims}")
            coordinates = torch.arange(omega.shape[-1], device=omega.device) // irrep_dims[0]
            off_block = coordinates[:, None] != coordinates[None, :]
            if bool((omega[..., off_block] != 0).any()):
                raise ValueError("exact-congruence multi-head energy requires block-diagonal dense transport")
            pair_blocks = _equal_diag_blocks(omega, len(irrep_dims), irrep_dims[0])
            log_abs_det = _slogdet(pair_blocks)                                 # (..., N, N, H)
        return a, s_tilde, log_abs_det

    if not _certifies_same_frame_flat_cocycle(omega):
        raise TypeError(
            f"exact-congruence coupling cannot invert transport {type(omega).__name__!r}: the fast "
            "route needs a certified same-frame flat cocycle (Omega_ij = U_i U_j^{-1}, whose "
            "inverse is the pair transpose Omega_ji). A learned-connection, direct-link or "
            "RoPE-wrapped transport has no such inverse; pass a dense pairwise Omega to use the "
            "reference route, or select family='gaussian_diagonal'.")

    # Fast route. Omega_ij^{-1} = Omega_ji, so the backward-transported QUERY at pair (i, j) is the
    # ordinary forward transport of the query array read at the swapped pair index. Both calls reuse
    # the established factored/compact fast paths verbatim.
    a = transport_mean(omega, mu_q)                                 # (..., N, N, K): Omega_ij mu_j
    s_tilde = transport_covariance(omega, sigma_q, diagonal_out=True)
    if a.shape[-3] != a.shape[-2]:
        raise ValueError(
            "exact-congruence coupling needs a square token transport, got transported shape "
            f"{tuple(a.shape)}")
    a = a.transpose(-3, -2)                                         # (..., N, N, K): Omega_ij^{-1} mu_i
    s_tilde = s_tilde.transpose(-3, -2)
    ell = _vertex_log_abs_det(omega, irrep_dims, mu_q.shape[-1])    # (..., N, H)
    log_abs_det = ell.unsqueeze(-2) - ell.unsqueeze(-3)             # (..., N, N, H): ell_i - ell_j
    return a, s_tilde, log_abs_det


@register_family("gaussian_diagonal_exact")
class ExactCongruenceDiagonalGaussian(DiagonalGaussian):
    r"""Diagonal Gaussian whose transported pair energy is the EXACT congruence divergence.

    Identical to :class:`~vfe3.families.gaussian.DiagonalGaussian` in every respect except
    :meth:`coupling_energy`: same parameters, same self-divergence, same natural/log-partition
    algebra, same decode. Only the belief-coupling grid changes, from the divergence to the
    diagonally-projected transported key to the divergence to its exact pushforward.
    """

    cov_kind = "diagonal"
    dispersion_is_covariance = True
    # Pointwise algebra is DiagonalGaussian's, unchanged: only coupling_energy is overridden, so the
    # Gaussian moment-matched barycenter and the fixed alpha=1 KL decode kernels are exact here
    # (audit 2026-07-26 E-04) -- inherited True, restated because the scope declarations below are the
    # point of this block.
    gaussian_pointwise_algebra = True
    # The module SCOPE section, declared for config validation (audit 2026-07-26 E-03): the pullback
    # identity is KL-specific, and it needs Omega_ij^{-1} -- from the pair transpose of a certified
    # cocycle, or by inverting a dense operator. A direct-link container offers neither, and reached
    # `_pullback_query`'s TypeError only at the first forward.
    requires_kl_divergence = True
    transport_requirement = "pair_inverse"

    def block(self, start: int, end: int) -> "ExactCongruenceDiagonalGaussian":
        return ExactCongruenceDiagonalGaussian(
            self.mu[..., start:end], self.sigma[..., start:end])

    def broadcast_over_keys(self) -> "ExactCongruenceDiagonalGaussian":
        return ExactCongruenceDiagonalGaussian(self.mu.unsqueeze(-2), self.sigma.unsqueeze(-2))

    @classmethod
    def stack(
        cls,
        parts: List["ExactCongruenceDiagonalGaussian"],

        *,
        dim:   int = 0,
    ) -> "ExactCongruenceDiagonalGaussian":
        return ExactCongruenceDiagonalGaussian(
            torch.stack([p.mu for p in parts], dim=dim),
            torch.stack([p.sigma for p in parts], dim=dim),
        )

    @classmethod
    def coupling_energy(
        cls,
        mu_q:         torch.Tensor,       # (..., N, K) query locations
        dispersion_q: torch.Tensor,       # (..., N, K) query variances
        mu_k:         torch.Tensor,       # (..., N, K) key locations, PRE-transport
        dispersion_k: torch.Tensor,       # (..., N, K) key variances, PRE-transport
        omega:        object,             # transport container

        *,
        alpha:             float = 1.0,
        kl_max:            float = 100.0,
        eps:               float = 1e-6,
        divergence_family: str   = "renyi",

        irrep_dims:        Optional[List[int]] = None,
        diagonal_out:      Optional[bool]      = None,
    ) -> torch.Tensor:                    # (..., N, N) or (..., H, N, N)
        r"""Exact :math:`\mathrm{KL}(q_i \| \Omega_{ij\#}q_j)` at diagonal cost.

        See the module docstring for the derivation. The clamp is applied ONCE to the assembled
        per-block divergence, exactly as ``renyi_closed_form`` clamps its summed scalar, so the
        saturation semantics the attention weights and the pair masks depend on are unchanged.
        """
        if divergence_family != "renyi" or abs(alpha - 1.0) >= 1e-9:
            raise ValueError(
                f"family='gaussian_diagonal_exact' implements the exact congruence for KL only "
                f"(divergence_family='renyi', renyi_order=1.0); got "
                f"divergence_family={divergence_family!r}, alpha={alpha}. For a Renyi order other "
                f"than one the covariance blend mixes the dense pushforward with a diagonal "
                f"covariance and does not reduce to a diagonal form. Use family="
                f"'gaussian_diagonal' for a noncanonical divergence.")
        if dispersion_q.dim() != mu_q.dim() or dispersion_k.dim() != mu_k.dim():
            raise ValueError(
                "family='gaussian_diagonal_exact' is a diagonal family; got dispersion ranks "
                f"{dispersion_q.dim()}/{dispersion_k.dim()} against location ranks "
                f"{mu_q.dim()}/{mu_k.dim()}")

        K = mu_q.shape[-1]
        blocks = _blocking(irrep_dims, K)
        a, s_tilde, log_abs_det = _pullback_query(mu_q, dispersion_q, omega, blocks)

        compute_dtype = (torch.float64
                         if torch.float64 in (mu_q.dtype, dispersion_q.dtype,
                                              mu_k.dtype, dispersion_k.dtype)
                         else torch.float32)
        s_back = s_tilde.to(compute_dtype).clamp(min=eps)                     # (..., N, N, K)
        s_key = dispersion_k.to(compute_dtype).clamp(min=eps).unsqueeze(-3)   # (..., 1, N, K)
        s_self = dispersion_q.to(compute_dtype).clamp(min=eps).unsqueeze(-2)  # (..., N, 1, K)
        delta = mu_k.to(compute_dtype).unsqueeze(-3) - a.to(compute_dtype)    # (..., N, N, K)
        # Per-coordinate terms of 2*KL, less the -K and the determinant of the pulled-back query
        # (both of which are per BLOCK, added after the coordinate reduction below).
        per_coord = (
            s_back / s_key
            + (delta ** 2) / s_key
            + torch.log(s_key)
            - torch.log(s_self)
        )                                                                     # (..., N, N, K)
        log_abs_det = log_abs_det.to(compute_dtype)                           # (..., N, N, H)

        if len(blocks) == 1:
            energy = 0.5 * (per_coord.sum(dim=-1) - K + 2.0 * log_abs_det[..., 0])
            return safe_kl_clamp(energy, kl_max=kl_max)                       # (..., N, N)

        if len(set(blocks)) == 1:
            # Equal blocks (the block_glk default): reduce every head at once, then move the head
            # axis to -3 to match pairwise_energy's (..., H, N, N) layout exactly.
            d = blocks[0]
            reduced = per_coord.reshape(*per_coord.shape[:-1], len(blocks), d).sum(dim=-1)
            energy = 0.5 * (reduced - d + 2.0 * log_abs_det)                  # (..., N, N, H)
            return safe_kl_clamp(energy.movedim(-1, -3), kl_max=kl_max)       # (..., H, N, N)

        energies = []
        start = 0
        for h, d in enumerate(blocks):
            end = start + d
            reduced = per_coord[..., start:end].sum(dim=-1)                   # (..., N, N)
            energies.append(0.5 * (reduced - d + 2.0 * log_abs_det[..., h]))
            start = end
        return safe_kl_clamp(torch.stack(energies, dim=-3), kl_max=kl_max)    # (..., H, N, N)
