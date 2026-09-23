r"""Reusable sufficient statistics for the canonical diagonal-Gaussian KL pair grid."""

from dataclasses import dataclass
from typing import List, Optional

import torch

from vfe3.families.base import safe_kl_clamp


@dataclass(frozen=True)
class DiagonalKLPairStats:
    """One graph-live diagonal-KL pair bundle for attention and filtering updates."""

    energy:        torch.Tensor
    pair_mask:     torch.Tensor
    inv_sigma_t:   torch.Tensor
    delta_tq:      torch.Tensor


def _validate_irrep_dims(
    K: int,

    *,
    irrep_dims: Optional[List[int]] = None,
) -> None:
    """Require a nonempty positive partition of all K coordinates when blocks are supplied."""
    if irrep_dims is None:
        return
    if not irrep_dims:
        raise ValueError("irrep_dims must be nonempty when supplied")
    if any(d <= 0 for d in irrep_dims):
        raise ValueError(f"irrep_dims entries must be positive, got {irrep_dims}")
    if sum(irrep_dims) != K:
        raise ValueError(f"irrep_dims must sum to K={K}, got {irrep_dims}")


def _reduce_coordinate_term(
    term: torch.Tensor,                     # (..., N, N, K) one diagonal-KL coordinate term

    *,
    irrep_dims: Optional[List[int]] = None,
) -> torch.Tensor:                          # (..., N, N) or (..., H, N, N)
    r"""Reduce one diagonal-KL coordinate term in the generic per-head layout."""
    if irrep_dims is None or len(irrep_dims) == 1:
        return term.sum(dim=-1)

    H = len(irrep_dims)
    if len(set(irrep_dims)) == 1:
        d = irrep_dims[0]
        shape = (*term.shape[:-1], H, d)
        return term.reshape(shape).sum(dim=-1).movedim(-1, -3)

    reduced = []
    start = 0
    for d in irrep_dims:
        end = start + d
        reduced.append(term[..., start:end].sum(dim=-1))
        start = end
    return torch.stack(reduced, dim=-3)


def diagonal_kl_pair_stats(
    mu_q:    torch.Tensor,                  # (..., N, K) query means
    sigma_q: torch.Tensor,                  # (..., N, K) query variances
    mu_t:    torch.Tensor,                  # (..., N, N, K) transported key means
    sigma_t: torch.Tensor,                  # (..., N, N, K) transported key variances

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,

    irrep_dims: Optional[List[int]] = None,
) -> DiagonalKLPairStats:
    r"""Build the canonical diagonal-KL energy and its reusable pair statistics.

    The three coordinate reductions remain separate:

        KL(q_i || t_ij) = 0.5 [sum(sigma_q / sigma_t)
                               + sum((mu_t - mu_q)^2 / sigma_t)
                               - K + sum(log sigma_t - log sigma_q)].

    Inputs are promoted to float32 before the shared variance clamp and difference. The energy
    preserves the generic KL path's division order exactly; the reusable reciprocal is computed
    only after the energy and mask. All returned tensors remain attached to their input graph.
    """
    mu_q_f       = mu_q.float()
    sigma_q_safe = sigma_q.float().clamp(min=eps)
    mu_t_f       = mu_t.float()
    sigma_t_safe = sigma_t.float().clamp(min=eps)
    K = mu_q_f.shape[-1]
    _validate_irrep_dims(K, irrep_dims=irrep_dims)

    delta_tq = mu_t_f - mu_q_f.unsqueeze(-2)

    sigma_q_reduce = sigma_q_safe.double()
    sigma_t_reduce = sigma_t_safe.double()
    delta_reduce = delta_tq.double()
    trace_term = _reduce_coordinate_term(
        sigma_q_reduce.unsqueeze(-2) / sigma_t_reduce,
        irrep_dims=irrep_dims,
    )
    mahal_term = _reduce_coordinate_term(
        (delta_reduce ** 2) / sigma_t_reduce,
        irrep_dims=irrep_dims,
    )
    logdet_term = _reduce_coordinate_term(
        torch.log(sigma_t_reduce).sub_(torch.log(sigma_q_reduce).unsqueeze(-2)),
        irrep_dims=irrep_dims,
    )
    if irrep_dims is None or len(irrep_dims) == 1:
        coordinate_dim: 'int | torch.Tensor' = K
    elif len(set(irrep_dims)) == 1:
        coordinate_dim = irrep_dims[0]
    else:
        head_shape = (1,) * (trace_term.dim() - 3) + (len(irrep_dims), 1, 1)
        coordinate_dim = trace_term.new_tensor(irrep_dims).reshape(head_shape)
    # Only the fresh reduction output is mutated; preserve the energy operation order.
    raw_energy = trace_term.add_(mahal_term).sub_(coordinate_dim).add_(logdet_term).mul_(0.5)
    raw_energy = raw_energy.to(mu_q_f.dtype)
    energy = safe_kl_clamp(raw_energy, kl_max=kl_max)
    pair_mask = ((energy > 0.0) & (energy < kl_max)).to(energy.dtype)
    inv_sigma_t = 1.0 / sigma_t_safe
    return DiagonalKLPairStats(
        energy=energy,
        pair_mask=pair_mask,
        inv_sigma_t=inv_sigma_t,
        delta_tq=delta_tq,
    )
