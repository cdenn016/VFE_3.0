"""Exact flat full-Gaussian KL contractions with token-sized matrix workspaces."""

from typing import List, Optional, Tuple

import torch

from vfe3.families.base import _logdet_chol, safe_kl_clamp
from vfe3.numerics import safe_eigvalsh, validated_cholesky_solve


def _frame_blocks(
    omega:      object,
    K:          int,
    irrep_dims: Optional[List[int]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
    r"""Extract certified flat factors, preserving the caller's covariance marginals."""
    from vfe3.geometry.transport import CompactFactoredTransport, FactoredTransport

    dims = [K] if irrep_dims is None else irrep_dims
    if not dims or sum(dims) != K or len(set(dims)) != 1:
        return None
    H, d = len(dims), dims[0]
    if isinstance(omega, CompactFactoredTransport):
        if not omega.same_frame_flat_cocycle or omega.irrep_dims != dims:
            return None
        return omega.exp_blocks, omega.inv_blocks, H, d
    if not isinstance(omega, FactoredTransport) or not omega.same_frame_flat_cocycle:
        return None
    if omega.exp_phi.shape[-2:] != (K, K) or omega.exp_neg_phi.shape[-2:] != (K, K):
        return None
    if H == 1:
        return omega.exp_phi.unsqueeze(-3), omega.exp_neg_phi.unsqueeze(-3), H, d
    if omega.irrep_dims != dims:
        return None
    # A dense factor must actually preserve each scored head, not merely name a partition.
    labels = torch.arange(K, device=omega.exp_phi.device) // d
    off_block = labels[:, None] != labels[None, :]
    with torch.no_grad():
        if bool((omega.exp_phi[..., off_block] != 0).any()
                | (omega.exp_neg_phi[..., off_block] != 0).any()):
            return None
    U = torch.stack([omega.exp_phi[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
    V = torch.stack([omega.exp_neg_phi[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
    return U, V, H, d


def common_frame_kl_energy(
    mu_q:             torch.Tensor,       # (..., Nq, K)
    Sigma_q:          torch.Tensor,       # (..., Nq, K, K)
    mu_k:             torch.Tensor,       # (..., Nk, K)
    Sigma_k:          torch.Tensor,       # (..., Nk, K, K)
    omega:            object,

    *,
    kl_max:           float,
    precision_policy: str,
    irrep_dims:       Optional[List[int]] = None,
    pair_support:     Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:              # (..., H, Nq, Nk), headless when H=1; None -> legacy
    r"""Compute the complete KL in common coordinates, or request the legacy scorer.

    With x_i=U_i^{-1}mu_i, C_i=U_i^{-1}Sigma_i U_i^{-T}, H_j=C_j^{-1},
    D_ij = (tr((C_i+x_i x_i^T)H_j)-2 x_i^T H_j x_j
            +x_j^T H_j x_j-d+logdet(C_j)-logdet(C_i))/2.
    Every matrix factorization has one token axis. Pair work is scalar or GEMM.
    In particular, the receiver log determinant is retained before the KL cap.

    Admission uses zero-jitter token certificates and conservative condition/scale
    bounds for the unmaterialized transported covariances. Uncertain inputs return
    None so the original local-coordinate jitter, certification and escalation
    policy owns their result. This is algebraic equality, not bitwise equality.
    No detached statistics enter the differentiable score; only the common mean
    offset and numerical admission decisions are detached.
    """
    if Sigma_q.dim() != mu_q.dim() + 1 or Sigma_k.dim() != mu_k.dim() + 1:
        return None
    K = mu_q.shape[-1]
    if mu_k.shape[-1] != K or Sigma_q.shape[-2:] != (K, K) or Sigma_k.shape[-2:] != (K, K):
        return None
    frames = _frame_blocks(omega, K, irrep_dims)
    if frames is None:
        return None
    U, V, H, d = frames
    Nq, Nk = mu_q.shape[-2], mu_k.shape[-2]
    # The legacy certified self-link restoration requires this same token table.
    if Nq != Nk or U.shape[-4] != Nq or V.shape[-4] != Nk:
        return None
    dtypes = (mu_q.dtype, Sigma_q.dtype, mu_k.dtype, Sigma_k.dtype)
    if any(dtype not in (torch.float32, torch.float64) for dtype in (*dtypes, U.dtype, V.dtype)):
        return None
    public_dtype = torch.float64 if torch.float64 in dtypes else torch.float32
    # Escalated float64 frames (audit 2026-09-23 #12) are contracted at float64, not re-rounded.
    work = (torch.float64 if precision_policy == "fp64" or public_dtype == torch.float64
            or torch.float64 in (U.dtype, V.dtype) else torch.float32)
    # Preserve the stricter pair-pivot policy rather than substituting a token proxy.
    if precision_policy == "fp32_escalate_cond":
        return None
    batch = torch.broadcast_shapes(mu_q.shape[:-2], mu_k.shape[:-2], U.shape[:-4], V.shape[:-4])
    shape = (*batch, H, Nq, Nk)
    if pair_support is None:
        support = torch.ones(shape, device=mu_q.device, dtype=torch.bool)
    else:
        if pair_support.dtype != torch.bool:
            raise ValueError("pair_support must be boolean")
        if H == 1 and pair_support.dim() <= len(batch) + 2:
            pair_support = pair_support.unsqueeze(-3)
        support = torch.broadcast_to(pair_support, shape)
    q_used = support.any(dim=-1).movedim(-2, -1)  # (..., Nq, H)
    k_used = support.any(dim=-2).movedim(-2, -1)  # (..., Nk, H)
    eye = torch.eye(d, device=mu_q.device, dtype=work)

    with torch.autocast(device_type=mu_q.device.type, enabled=False):
        U = torch.where(q_used[..., None, None], U.to(work), eye)
        V = torch.where(k_used[..., None, None], V.to(work), eye)
        mq = torch.where(q_used[..., None], mu_q.to(work).reshape(*mu_q.shape[:-1], H, d), 0.)
        mk = torch.where(k_used[..., None], mu_k.to(work).reshape(*mu_k.shape[:-1], H, d), 0.)
        Sq = torch.stack([Sigma_q[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3).to(work)
        Sk = torch.stack([Sigma_k[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3).to(work)
        Sq = torch.where(q_used[..., None, None], Sq, eye)
        Sk = torch.where(k_used[..., None, None], Sk, eye)
        if not bool(torch.isfinite(U).all() & torch.isfinite(V).all()
                    & torch.isfinite(mq).all() & torch.isfinite(mk).all()):
            return None
        local_q = validated_cholesky_solve(Sq)
        local_k = validated_cholesky_solve(Sk)
        if not bool(local_q.certified.all() & local_k.certified.all()):
            return None
        # Solve against the RECEIVER factor. Reusing source V would change partial
        # derivatives when filtering gives query/key frame roles separate graphs.
        inverse_q, info = torch.linalg.solve_ex(U, eye.expand_as(U))
        if bool((info != 0).any()) or not bool(torch.isfinite(inverse_q).all()):
            return None
        xq = (inverse_q @ mq.unsqueeze(-1)).squeeze(-1)
        xk = (V @ mk.unsqueeze(-1)).squeeze(-1)
        Cq = inverse_q @ Sq @ inverse_q.transpose(-1, -2)
        Ck = V @ Sk @ V.transpose(-1, -2)
        cq = validated_cholesky_solve(Cq)
        ck = validated_cholesky_solve(Ck)
        if not bool(cq.certified.all() & ck.certified.all()):
            return None
        # cond(U_i C_j U_i^T) <= cond(U_i)^2 cond(C_j). Check only read pairs.
        # Also bound eigenvalue scales to avoid a legacy under/overflow boundary.
        with torch.no_grad():
            singular = torch.linalg.svdvals(U.detach())
            spectral_k = safe_eigvalsh(Ck.detach())
            hi = singular[..., 0].movedim(-1, -2)
            lo = singular[..., -1].movedim(-1, -2)
            cond_bound = (hi / lo).square().unsqueeze(-1) * ck.condition.movedim(-1, -2).unsqueeze(-2)
            limit = 1. / (64. * d * torch.finfo(work).eps)
            min_bound = lo.square().unsqueeze(-1) * spectral_k[..., 0].movedim(-1, -2).unsqueeze(-2)
            max_bound = hi.square().unsqueeze(-1) * spectral_k[..., -1].movedim(-1, -2).unsqueeze(-2)
            admissible = (torch.isfinite(cond_bound) & (cond_bound <= limit)
                          & (min_bound > torch.finfo(work).tiny * (64. * d))
                          & (max_bound < torch.finfo(work).max / (64. * d)))
            if not bool((admissible | ~support).all()):
                return None
        Hk = torch.cholesky_inverse(ck.factor)
        # Common translation reduces cancellation for large shared means. Its
        # derivative is identically zero in the exact KL, so the offset is fixed.
        offset = xk.detach().mean(dim=-3, keepdim=True)
        xq, xk = xq - offset, xk - offset
        second_q = Cq + xq.unsqueeze(-1) * xq.unsqueeze(-2)
        eta_k = (Hk @ xk.unsqueeze(-1)).squeeze(-1)
        trace = torch.einsum("...ihab,...jhab->...hij", second_q, Hk)
        cross = torch.einsum("...ihd,...jhd->...hij", xq, eta_k)
        key_term = ((xk * eta_k).sum(-1) + _logdet_chol(ck.factor)).movedim(-1, -2)
        query_term = _logdet_chol(cq.factor).movedim(-1, -2)
        energy = .5 * (trace - 2. * cross + key_term.unsqueeze(-2) - query_term.unsqueeze(-1) - d)
        # The existing flat path restores identity self maps exactly, including
        # zero frame derivatives even for independent query/key factor leaves.
        delta = mq - mk
        precision_local_k = torch.cholesky_inverse(local_k.factor)
        self_energy = .5 * ((precision_local_k * Sq.transpose(-1, -2)).sum((-1, -2))
                           + (delta * (precision_local_k @ delta.unsqueeze(-1)).squeeze(-1)).sum(-1)
                           - d + _logdet_chol(local_k.factor) - _logdet_chol(local_q.factor))
        diagonal = torch.eye(Nq, device=energy.device, dtype=torch.bool)
        energy = torch.where(diagonal, self_energy.movedim(-1, -2).unsqueeze(-1), energy)
        if work == torch.float32:
            # Bound rounding in the expanded quadratic, including loss of Cq when
            # added to the mean outer product. Identity pairs are restored above.
            with torch.no_grad():
                magnitude = (trace.abs() + 2. * cross.abs() + key_term.abs().unsqueeze(-2)
                             + query_term.abs().unsqueeze(-1) + d)
                roundoff = (8. * d * torch.finfo(work).eps) * magnitude
                tolerance = 1e-3 * energy.abs().clamp_min(1.)
                if bool(((roundoff > tolerance) & support & ~diagonal).any()):
                    return None
        # Nonfinite assembled scores use the legacy numerical policy, not a new repair.
        if not bool((torch.isfinite(energy) | ~support).all()):
            return None
        energy = safe_kl_clamp(energy, kl_max=kl_max).masked_fill(~support, 0.)
        energy = energy.to(public_dtype)
        return energy.squeeze(-3) if H == 1 else energy
