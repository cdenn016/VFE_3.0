r"""Exact projected Gaussian scores and transported mixture moments.

The caller owns categorical support, priors, temperature, and KL clipping.
Projected covariances are symmetrized after contraction to remove roundoff.
Successful factorization always uses zero jitter and no spectral repair.
"""

import math
from typing import Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


# Budget for one projected map L; its covariance product can require a second.
_PROJECTION_TILE_BYTES = 32 * 1024 ** 2


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(tensor.dtype == torch.float64 for tensor in tensors) else torch.float32


def _projected_linear_map(
    S:              torch.Tensor,
    inverse_blocks: torch.Tensor,
) -> torch.Tensor:
    r"""Contract L=S U^{-1}, retaining compact blocks for whole-belief readouts."""
    width = S.shape[-1]
    if inverse_blocks.shape[-2:] == (width, width):
        return S @ inverse_blocks
    G, d, _ = inverse_blocks.shape[-3:]
    # (..., H, r, G, d) times (..., G, d, d), without dense K by K frames.
    return torch.einsum("...hrgd,...gde->...hrge",
                        S.reshape(*S.shape[:-1], G, d), inverse_blocks).flatten(-2)


def projected_moments(
    mu_h:           torch.Tensor,  # (..., N, H, d)
    Sigma_h:        torch.Tensor,  # (..., N, H, d) or (..., N, H, d, d)
    inverse_blocks: torch.Tensor,  # (..., N, H, d, d)
    S:              torch.Tensor,  # (H, r, d)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return a_i = S U_i^{-1} mu_i and G_i = S U_i^{-1} Sigma_i U_i^{-T} S^T.

    Diagonal input covariance is contracted directly into a full reduced
    covariance. The mathematically symmetric result is averaged with its
    transpose to remove multiplication roundoff without adding jitter.
    Large projected maps use checkpointed token tiles to bound saved map
    intermediates; the output moments and source-factor recovery stay unchanged.
    Returns (..., N, H, r) and (..., N, H, r, r).
    """
    if mu_h.ndim < 3 or S.ndim != 3 or S.shape[::2] != mu_h.shape[-2:]:
        raise ValueError("comparison maps must have (H, r, d) matching mean (..., N, H, d)")
    H, d = mu_h.shape[-2:]
    if not 1 <= S.shape[-2] <= d:
        raise ValueError("comparison rank must lie between one and the head dimension")
    compact_full = (inverse_blocks.shape[-3] * inverse_blocks.shape[-1] == d
                    and inverse_blocks.shape[-2] == inverse_blocks.shape[-1])
    if inverse_blocks.shape[-3:] != (H, d, d) and not compact_full:
        raise ValueError("inverse frame blocks must match mean heads and coordinate dimension")
    diagonal = Sigma_h.shape == mu_h.shape
    full = Sigma_h.shape == (*mu_h.shape, d)
    if not diagonal and not full:
        raise ValueError("covariance must match the mean as diagonal entries or full head matrices")
    dtype = _compute_dtype(mu_h, Sigma_h, inverse_blocks, S)
    tokens = mu_h.shape[-3]
    batch = torch.broadcast_shapes(mu_h.shape[:-3], inverse_blocks.shape[:-4])
    bytes_per_token = math.prod(batch) * H * S.shape[-2] * d * (torch.finfo(dtype).bits // 8)
    # Small projections keep the single contraction. Large projections checkpoint
    # token tiles: tiling alone would retain every L and weighted L for backward.
    if tokens * bytes_per_token > _PROJECTION_TILE_BYTES:
        step = max(1, min(tokens, 32, _PROJECTION_TILE_BYTES // bytes_per_token))
        means, covariances = [], []
        for start in range(0, tokens, step):
            inverse = (inverse_blocks[..., start:start+step, :, :, :]
                       if inverse_blocks.ndim >= 4 and inverse_blocks.shape[-4] == tokens
                       else inverse_blocks)
            sigma = (Sigma_h[..., start:start+step, :, :] if diagonal
                     else Sigma_h[..., start:start+step, :, :, :])
            args = (mu_h[..., start:start+step, :, :], sigma, inverse, S)
            if torch.is_grad_enabled() and any(x.requires_grad for x in args):
                a, G = checkpoint(_projected_moments_tile, *args,
                                  use_reentrant=False, preserve_rng_state=False)
            else:
                a, G = _projected_moments_tile(*args)
            means.append(a)
            covariances.append(G)
        return torch.cat(means, dim=-3), torch.cat(covariances, dim=-4)
    return _projected_moments_tile(mu_h, Sigma_h, inverse_blocks, S)


def _projected_moments_tile(
    mu_h:           torch.Tensor,
    Sigma_h:        torch.Tensor,
    inverse_blocks: torch.Tensor,
    S:              torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute a=L mu and G=L Sigma L^T for one token tile."""
    diagonal = Sigma_h.shape == mu_h.shape
    dtype = _compute_dtype(mu_h, Sigma_h, inverse_blocks, S)
    with torch.autocast(device_type=mu_h.device.type, enabled=False):
        mu = mu_h.to(dtype)
        Sigma = Sigma_h.to(dtype)
        L = _projected_linear_map(S.to(dtype), inverse_blocks.to(dtype))
        a = (L @ mu.unsqueeze(-1)).squeeze(-1)
        if diagonal:
            G = (L * Sigma.unsqueeze(-2)) @ L.transpose(-1, -2)
        else:
            G = L @ Sigma @ L.transpose(-1, -2)
        # The two off-diagonal contractions can round differently near zero.
        # Keep the covariance exactly symmetric for strict zero-jitter Cholesky.
        G = .5 * G + .5 * G.transpose(-1, -2)
    return a, G


def _cholesky_status(G: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    r"""Bound eps ||G||_F ||G^{-1}||_F using trace and determinant before inversion.

    For SPD G of order r>1, lambda_min >= det(G)/(tr(G)/(r-1))^(r-1).
    Thus ||G^{-1}||_F <= sqrt(r)/lambda_min. Unlike pivot spread, this
    conservative bound detects shear. All decisions remain on the device.
    """
    chol, info = torch.linalg.cholesky_ex(G, check_errors=False)
    with torch.no_grad():
        failed = (info != 0) | ~torch.isfinite(chol).all(dim=(-2, -1))
        screen = torch.zeros_like(failed)
        r = G.shape[-1]
        if G.dtype == torch.float32 and r > 1:
            logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
            trace = G.diagonal(dim1=-2, dim2=-1).sum(-1)
            log_bound = (math.log(torch.finfo(G.dtype).eps) + G.norm(dim=(-2, -1)).log()
                         + .5 * math.log(r) + (r - 1) * (trace / (r - 1)).log() - logdet)
            screen = (~failed) & (log_bound > math.log(1e-4))
        tolerance = 32 * torch.finfo(G.dtype).eps
        status = torch.stack((torch.isfinite(G).all(),
                              torch.isclose(G, G.transpose(-1, -2), rtol=tolerance, atol=0.).all(),
                              (failed | screen).any(), failed.any()))
    return chol, failed, screen, status


def _strict_cholesky(
    G:          torch.Tensor,
    projection: Optional[Tuple[torch.Tensor, ...]] = None,
    prepared:   Optional[tuple] = None,
) -> torch.Tensor:
    r"""Factor G with zero jitter, optionally recovering its factor from source moments.

    A failed rounded Gram matrix can lose SPD although its source measurement
    still has full row rank. For failed or poorly conditioned float32 entries, reconstruct
    X = S U^{-1} Sigma^{1/2} in float64 and factor X^T = QR, giving C = R^T.
    No jitter, covariance floor, or projection constraint is introduced.
    """
    if prepared is None:
        chol, failed, screen, status = _cholesky_status(G)
        decisions = status.tolist()
    else:
        chol, failed, screen, decisions = prepared
    finite, symmetric, suspicious_any, failed_any = decisions
    ill_conditioned = torch.zeros_like(failed)
    recover_any = failed_any
    if finite and symmetric and suspicious_any:
        with torch.no_grad():
            suspicious = failed | screen
            eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
            safe_factor = torch.where(failed[..., None, None], eye, chol)
            inverse_norm = torch.cholesky_inverse(safe_factor[suspicious]).norm(dim=(-2, -1))
            certificate = torch.finfo(G.dtype).eps * G[suspicious].norm(dim=(-2, -1)) * inverse_norm > 1e-4
            ill_conditioned[suspicious] = certificate
            recover_any = bool((failed | ill_conditioned).any())
    if not finite:
        raise ValueError("projected covariance must be finite for zero-jitter Cholesky")
    if not symmetric:
        raise ValueError("projected covariance must be symmetric for zero-jitter Cholesky")
    if recover_any:
        if projection is None:
            if failed_any:
                raise ValueError("projected covariance is not positive definite: zero-jitter Cholesky failed; no repair applied")
            return _strict_cholesky(G.double())
        failed = failed | ill_conditioned
        Sigma, inverse, S = projection[:3]
        batch, r, d = G.shape[:-2], G.shape[-1], S.shape[-1]
        # Index before promotion: rebuild only the flagged source square roots.
        with torch.autocast(device_type=G.device.type, enabled=False):
            selected_S = S.expand(*batch, r, d)[failed].double()
            if inverse.shape[-2:] == (d, d):
                L = selected_S @ inverse.expand(*batch, d, d)[failed].double()
            else:
                G_blocks, d_block, _ = inverse.shape[-3:]
                selected_inverse = inverse.unsqueeze(-4).expand(
                    *batch, G_blocks, d_block, d_block)[failed].double()
                L = _projected_linear_map(selected_S.unsqueeze(-3), selected_inverse).squeeze(-3)
            if Sigma.shape == (*batch, d):
                variance = Sigma[failed].double()
                if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
                    raise ValueError("source covariance must be finite and positive for projected factorization")
                root = L * variance.sqrt().unsqueeze(-2)
            elif Sigma.shape == (*batch, d, d):
                root = L @ _strict_cholesky(Sigma[failed].double())
            else:
                raise ValueError("source covariance shape does not match projected token/head dimensions")
            if not bool(torch.isfinite(root).all()):
                raise ValueError("projected source square root must be finite")
            _, R = torch.linalg.qr(root.transpose(-1, -2), mode="reduced")
            pivots = R.diagonal(dim1=-2, dim2=-1)
            tolerance = torch.finfo(root.dtype).eps * max(r, d) * root.norm(dim=(-2, -1))
            if bool((pivots.abs() <= tolerance.unsqueeze(-1)).any()):
                raise ValueError("projected measurement lost numerical row rank; no covariance repair applied")
            signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
            recovered = R.transpose(-1, -2) * signs.unsqueeze(-2)
        # Never backpropagate through a failed Cholesky, even with zero cotangent.
        safe = torch.where(failed[..., None, None], torch.eye(r, device=G.device, dtype=G.dtype), G)
        chol = _strict_cholesky(safe).double()
        chol[failed] = recovered
    return chol


def projected_gaussian_kl(
    a_q: torch.Tensor,  # (..., Nq, H, r)
    G_q: torch.Tensor,  # (..., Nq, H, r, r)
    a_k: torch.Tensor,  # (..., Nk, H, r)
    G_k: torch.Tensor,  # (..., Nk, H, r, r)

    *,
    projection_q: Optional[Tuple[torch.Tensor, ...]] = None,
    projection_k: Optional[Tuple[torch.Tensor, ...]] = None,
) -> torch.Tensor:      # (..., H, Nq, Nk)
    r"""Compute raw KL(q_i || k_j) = (tr(G_j^{-1}G_i) + delta^T G_j^{-1}delta - r + logdet(G_j/G_i))/2.

    Cholesky factors and inverses have only one token axis. Bounded pair tiles
    preserve direct differences without a full vector pair grid. Checkpointed
    tiles also bound saved backward intermediates. Poorly conditioned factors
    use float64 triangular solves instead of cancelling covariance/precision products. No floor, cap,
    symmetrization, jitter, or repair is applied. Optional (Sigma,U^-1,S[,mu])
    source tuples allow inaccurate rounded covariances and projected means to
    be recomputed from their original sources; covariance-only non-SPD inputs fail explicitly.
    """
    if a_q.ndim < 3 or a_k.ndim < 3 or a_q.shape[-2:] != a_k.shape[-2:]:
        raise ValueError("projected means must have matching head and comparison dimensions")
    r = a_q.shape[-1]
    if G_q.shape != (*a_q.shape, r) or G_k.shape != (*a_k.shape, r):
        raise ValueError("projected covariances must match their means with one additional matrix axis")
    dtype = _compute_dtype(a_q, G_q, a_k, G_k)
    with torch.autocast(device_type=a_q.device.type, enabled=False):
        aq, ak = a_q.to(dtype), a_k.to(dtype)
        Gq, Gk = G_q.to(dtype), G_k.to(dtype)
        q_status, k_status = _cholesky_status(Gq), _cholesky_status(Gk)
        # One host transfer for both roles and the means on the healthy path.
        decisions = torch.cat(((torch.isfinite(aq).all() & torch.isfinite(ak).all()).reshape(1),
                               q_status[-1], k_status[-1])).tolist()
        if not decisions[0]:
            raise ValueError("projected means must contain only finite values")
        chol_q = _strict_cholesky(Gq, projection_q, (*q_status[:3], decisions[1:5]))
        chol_k = _strict_cholesky(Gk, projection_k, (*k_status[:3], decisions[5:9]))
        factor_trace = chol_q.dtype == torch.float64 or chol_k.dtype == torch.float64
        if factor_trace:
            # Preserve source accuracy through means, triangular solves and the final scalar.
            for role, projection in enumerate((projection_q, projection_k)):
                if projection is not None and len(projection) == 4:
                    _, inverse, S, mu = projection
                    mean = (_projected_linear_map(S.double(), inverse.double())
                            @ mu.double().unsqueeze(-1)).squeeze(-1)
                    if role == 0:
                        aq = mean
                    else:
                        ak = mean
            aq, ak = aq.double(), ak.double()
            q_matrix, k_matrix = chol_q.double(), chol_k.double()
        else:
            q_matrix, k_matrix = Gq, torch.cholesky_inverse(chol_k)
        aq, ak = aq.movedim(-2, -3), ak.movedim(-2, -3)  # (..., H, N, r)
        q_matrix = q_matrix.movedim(-3, -4)              # (..., H, Nq, r, r)
        k_matrix = k_matrix.movedim(-3, -4)              # (..., H, Nk, r, r)
        logdet_q = 2 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2)
        logdet_k = 2 * chol_k.diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2)
        nq, nk = aq.shape[-2], ak.shape[-2]
        batch_heads = aq.numel() // (nq * r)
        per_pair = r * r if factor_trace else r
        pair_budget = max(1, (32 * 1024 ** 2) // (aq.element_size() * batch_heads * per_pair))
        q_step = max(1, min(nq, 32, pair_budget))
        k_step = max(1, min(nk, pair_budget // q_step))
        rows = []
        for i in range(0, nq, q_step):
            columns = []
            for j in range(0, nk, k_step):
                args = (aq[..., i:i+q_step, :], ak[..., j:j+k_step, :],
                        q_matrix[..., i:i+q_step, :, :], k_matrix[..., j:j+k_step, :, :],
                        logdet_q[..., i:i+q_step], logdet_k[..., j:j+k_step])
                if torch.is_grad_enabled() and (nq > q_step or nk > k_step) and any(x.requires_grad for x in args):
                    value = checkpoint(_projected_kl_tile, *args, factor_trace,
                                       use_reentrant=False, preserve_rng_state=False)
                else:
                    value = _projected_kl_tile(*args, factor_trace)
                columns.append(value)
            rows.append(torch.cat(columns, dim=-1))
        return torch.cat(rows, dim=-2).to(dtype)


def _projected_kl_tile(
    aq:       torch.Tensor,  # (..., H, I, r)
    ak:       torch.Tensor,  # (..., H, J, r)
    q_matrix: torch.Tensor,  # covariance or lower factor (..., H, I, r, r)
    k_matrix: torch.Tensor,  # precision or lower factor (..., H, J, r, r)
    logdet_q: torch.Tensor,
    logdet_k: torch.Tensor,
    factor_trace: bool,
) -> torch.Tensor:
    r"""Compute one KL tile; tr(Gk^-1 Gq) = ||Lk^-1 Lq||_F^2 on the factor route."""
    delta = aq.unsqueeze(-2) - ak.unsqueeze(-3)
    if factor_trace:
        Lk = k_matrix.unsqueeze(-4)
        relative = torch.linalg.solve_triangular(Lk, q_matrix.unsqueeze(-3), upper=False)
        trace = relative.square().sum(dim=(-2, -1))
        whitened = torch.linalg.solve_triangular(Lk, delta.unsqueeze(-1), upper=False)
        quadratic = whitened.square().sum(dim=(-2, -1))
    else:
        trace = torch.einsum("...iab,...jab->...ij", q_matrix, k_matrix)
        quadratic = torch.einsum("...ijr,...jrs,...ijs->...ij", delta, k_matrix, delta)
    return .5 * (trace + quadratic - aq.shape[-1]
                 + logdet_k.unsqueeze(-2) - logdet_q.unsqueeze(-1))


def relational_mixture_mean(
    xi:             torch.Tensor,  # (..., Nk, H, d)
    forward_blocks: torch.Tensor,  # (..., Nq, H, d, d)
    beta:           torch.Tensor,  # (..., H, Nq, Nk)
    T:              torch.Tensor,  # (H, s, d)
    O:              torch.Tensor,  # (H, d, s)
) -> torch.Tensor:                  # (..., Nq, H, d)
    r"""Return m_i = U_i O sum_j beta_ij T xi_j; identity content uses T=O=I.

    This is a mixture mean, not an E-step or a Gaussian covariance pushforward.
    Beta is supplied by the caller, including any support and normalization.
    """
    if xi.ndim < 3 or T.ndim != 3 or O.ndim != 3:
        raise ValueError("content requires headwise means and (H,s,d)/(H,d,s) factors")
    H, d = xi.shape[-2:]
    if T.shape[0] != H or T.shape[-1] != d or O.shape != (H, d, T.shape[-2]):
        raise ValueError("content factors do not match canonical mean heads")
    if forward_blocks.shape[-3:] != (H, d, d) or beta.shape[-3:] != (H, forward_blocks.shape[-4], xi.shape[-3]):
        raise ValueError("mixture weights and frame blocks must match receiver/source/head dimensions")
    dtype = _compute_dtype(xi, forward_blocks, beta, T, O)
    with torch.autocast(device_type=xi.device.type, enabled=False):
        content = torch.einsum("hsd,...jhd->...jhs", T.to(dtype), xi.to(dtype))
        mixed = torch.einsum("...hij,...jhs->...ihs", beta.to(dtype), content)
        body = torch.einsum("hds,...ihs->...ihd", O.to(dtype), mixed)
        return (forward_blocks.to(dtype) @ body.unsqueeze(-1)).squeeze(-1)
