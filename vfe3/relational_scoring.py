r"""Exact projected Gaussian scores and transported mixture moments.

The caller owns categorical support, priors, temperature, and KL clipping.
Projected covariances are symmetrized after contraction to remove roundoff.
Successful factorization always uses zero jitter and no spectral repair.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


# Budget for one projected map L; its covariance product can require a second.
_PROJECTION_TILE_BYTES = 32 * 1024 ** 2

# Cumulative (token, head) covariance entries whose Cholesky factor was rebuilt by the float64
# source-factor recovery, per device. Updated asynchronously on the recovery path only, so the
# healthy path pays nothing (audit 2026-09-23 low: the recovery had no activation counter).
_SOURCE_RECOVERY_COUNTS: Dict[str, torch.Tensor] = {}


def _count_source_recovery(recovered: torch.Tensor) -> None:
    r"""Accumulate recovered entries per device (async add, no host sync)."""
    key = str(recovered.device)
    counter = _SOURCE_RECOVERY_COUNTS.get(key)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=recovered.device)
        _SOURCE_RECOVERY_COUNTS[key] = counter
    counter += recovered.sum()


def source_recovery_counter(device: torch.device) -> torch.Tensor:
    r"""On-device snapshot of the cumulative recovered-entry count (no host sync).

    Differences of two snapshots count the recoveries between them.
    """
    counter = _SOURCE_RECOVERY_COUNTS.get(str(device))
    if counter is None:
        return torch.zeros((), dtype=torch.int64, device=device)
    return counter.clone()


def source_recovery_elements() -> int:
    r"""Recovered covariance entries since the last reset, over all devices. Host-syncs."""
    return int(sum(int(count) for count in _SOURCE_RECOVERY_COUNTS.values()))


def reset_source_recovery_elements() -> None:
    r"""Zero the per-device source-recovery counters."""
    for count in _SOURCE_RECOVERY_COUNTS.values():
        count.zero_()


# Conditioning-triggered float64 source recovery (config ``projected_cov_condition_recovery``,
# installed by VFEModel). Off, only entries whose zero-jitter float32 Cholesky FAILS are rebuilt
# from source; entries that factor but fail the certificate eps_32 ||G||_F ||G^{-1}||_F <= 1e-4
# keep their float32 factor, and the certificate itself is not evaluated.
_CONDITION_RECOVERY: bool = False


def set_projected_condition_recovery(enabled: bool) -> bool:
    r"""Set conditioning-triggered float64 source recovery; returns the previous value."""
    if not isinstance(enabled, bool):
        raise ValueError(f"projected condition recovery must be bool, got {enabled!r}")
    global _CONDITION_RECOVERY
    previous = _CONDITION_RECOVERY
    _CONDITION_RECOVERY = enabled
    return previous


def projected_condition_recovery() -> bool:
    r"""Return whether factored but ill-conditioned float32 entries take source recovery."""
    return _CONDITION_RECOVERY


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
    S:              torch.Tensor,  # (H, r, d) or (R, H, r, d)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return a_i = S U_i^{-1} mu_i and G_i = S U_i^{-1} Sigma_i U_i^{-T} S^T.

    Diagonal input covariance is contracted directly into a full reduced
    covariance. The mathematically symmetric result is averaged with its
    transpose to remove multiplication roundoff without adding jitter.
    Large projected maps use checkpointed token tiles to bound saved map
    intermediates; the output moments and source-factor recovery stay unchanged.
    A role-stacked ``S`` of shape ``(R, H, r, d)`` returns a leading ``R`` axis,
    ``(R, ..., N, H, r)`` and ``(R, ..., N, H, r, r)``, sharing one frame map and
    one covariance contraction across the roles.
    Returns (..., N, H, r) and (..., N, H, r, r) for a single map.
    """
    if mu_h.ndim < 3 or S.ndim not in (3, 4):
        raise ValueError("comparison maps must have (H, r, d) or (R, H, r, d) matching mean (..., N, H, d)")
    if S.ndim == 3 and S.shape[::2] != mu_h.shape[-2:]:
        raise ValueError("comparison maps must have (H, r, d) matching mean (..., N, H, d)")
    if S.ndim == 4 and S.shape[1::2] != mu_h.shape[-2:]:
        raise ValueError("role-stacked comparison maps must have (R, H, r, d) matching mean (..., N, H, d)")
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
    roles = S.shape[0] if S.ndim == 4 else 1
    bytes_per_token = math.prod(batch) * roles * H * S.shape[-2] * d * (torch.finfo(dtype).bits // 8)
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
    r"""Compute a=L mu and G=L Sigma L^T for one token tile.

    ``S`` is ``(H, r, d)`` for one comparison map or ``(R, H, r, d)`` for a
    role-stacked pair; the stacked form returns a leading ``R`` axis and batches
    the covariance contraction across roles while keeping one frame map.
    """
    diagonal = Sigma_h.shape == mu_h.shape
    dtype = _compute_dtype(mu_h, Sigma_h, inverse_blocks, S)
    with torch.autocast(device_type=mu_h.device.type, enabled=False):
        mu = mu_h.to(dtype)
        Sigma = Sigma_h.to(dtype)
        inverse = inverse_blocks.to(dtype)
        if S.ndim == 3:
            L = _projected_linear_map(S.to(dtype), inverse)  # (..., N, H, r, d)
            a = (L @ mu.unsqueeze(-1)).squeeze(-1)
            if diagonal:
                G = (L * Sigma.unsqueeze(-2)) @ L.transpose(-1, -2)
            else:
                G = L @ Sigma @ L.transpose(-1, -2)
        else:
            # Role-stacked maps: one map per role, batched into a single G contraction.
            L = torch.stack(
                [_projected_linear_map(S[role].to(dtype), inverse) for role in range(S.shape[0])],
                dim=0)  # (R, ..., N, H, r, d)
            a = (L @ mu.unsqueeze(-1).unsqueeze(0)).squeeze(-1)  # (R, ..., N, H, r)
            if diagonal:
                G = (L * Sigma.unsqueeze(0).unsqueeze(-2)) @ L.transpose(-1, -2)
            else:
                G = L @ Sigma.unsqueeze(0) @ L.transpose(-1, -2)
        # The two off-diagonal contractions can round differently near zero.
        # Keep the covariance exactly symmetric for strict zero-jitter Cholesky.
        G = .5 * G + .5 * G.transpose(-1, -2)
    return a, G


def _cholesky_status(G: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    r"""Factor G and certify eps ||G||_F ||G^{-1}||_F <= 1e-4 for every float32 entry on device.

    The certificate is evaluated EXACTLY for every successfully factored float32 entry from its
    own factor (``cholesky_inverse``), so ``screen`` marks exactly the ill-conditioned entries and
    the verdict joins the caller's single host transfer. It replaces the former AM-GM
    trace/determinant screen, :math:`\lambda_{\min} \ge \det G / (\operatorname{tr} G/(r-1))^{r-1}`,
    which was valid but about 4.5e4 times loose at the live condition, so almost every call paid a
    second host synchronization and masked gathers to evaluate this same certificate on the flagged
    entries (audit 2026-09-23 low). The recovery decision is unchanged: the screen was a necessary
    condition for the certificate, and the certificate alone decided recovery.

    The certificate is evaluated only under :func:`projected_condition_recovery`. Otherwise
    ``screen`` is all False and only failed factorizations are flagged for recovery.
    """
    chol, info = torch.linalg.cholesky_ex(G, check_errors=False)
    with torch.no_grad():
        failed = (info != 0) | ~torch.isfinite(chol).all(dim=(-2, -1))
        screen = torch.zeros_like(failed)
        r = G.shape[-1]
        if G.dtype == torch.float32 and r > 1 and _CONDITION_RECOVERY:
            eye = torch.eye(r, device=G.device, dtype=G.dtype)
            safe_factor = torch.where(failed[..., None, None], eye, chol)
            inverse_norm = torch.cholesky_inverse(safe_factor).norm(dim=(-2, -1))
            certificate = torch.finfo(G.dtype).eps * G.norm(dim=(-2, -1)) * inverse_norm > 1e-4
            screen = (~failed) & certificate
        tolerance = 32 * torch.finfo(G.dtype).eps
        status = torch.stack((torch.isfinite(G).all(),
                              torch.isclose(G, G.transpose(-1, -2), rtol=tolerance, atol=0.).all(),
                              (failed | screen).any(), failed.any()))
    return chol, failed, screen, status


def _strict_cholesky(
    G:          torch.Tensor,
    projection: Optional[Tuple[torch.Tensor, ...]] = None,
    prepared:   Optional[tuple] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Factor G with zero jitter, optionally recovering its factor from source moments.

    A failed rounded Gram matrix can lose SPD although its source measurement
    still has full row rank. For failed float32 entries, and for poorly conditioned ones under
    :func:`projected_condition_recovery`, reconstruct
    X = S U^{-1} Sigma^{1/2} in float64 and factor X^T = QR, giving C = R^T.
    No jitter, covariance floor, or projection constraint is introduced.

    Returns ``(chol, recover_mask)``. ``chol`` stays in ``G.dtype`` on the healthy
    path. When source recovery is applied the whole factor tensor is promoted to
    float64 so the rebuilt entries keep their QR accuracy, and ``recover_mask``
    marks exactly those entries so the pair path can escalate only the tiles that
    touch them. ``recover_mask`` is ``None`` on the healthy path.
    """
    if prepared is None:
        chol, failed, screen, status = _cholesky_status(G)
        decisions = status.tolist()
    else:
        chol, failed, screen, decisions = prepared
    finite, symmetric, suspicious_any, failed_any = decisions
    # ``screen`` is the exact conditioning certificate (see _cholesky_status), so the recovery
    # decision is already on the host: no second synchronization and no masked gathers.
    ill_conditioned = screen
    recover_any = suspicious_any
    if not finite:
        raise ValueError("projected covariance must be finite for zero-jitter Cholesky")
    if not symmetric:
        raise ValueError("projected covariance must be symmetric for zero-jitter Cholesky")
    if recover_any:
        if projection is None:
            if failed_any:
                raise ValueError("projected covariance is not positive definite: zero-jitter Cholesky failed; no repair applied")
            chol64, _ = _strict_cholesky(G.double())
            # Whole-tensor float64 refactor: every entry takes the factor route.
            recovered_all = torch.ones_like(failed)
            _count_source_recovery(recovered_all)
            return chol64, recovered_all
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
                root = L @ _strict_cholesky(Sigma[failed].double())[0]
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
        chol = _strict_cholesky(safe)[0].double()
        chol[failed] = recovered
        _count_source_recovery(failed)
        return chol, failed
    return chol, None


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

    The float64 factor route is selected PER PAIR (audit 2026-09-23 low): a pair (b, h, i, j) takes
    it only when its own query entry (b, i, h) or key entry (b, j, h) was recovered. Every other
    pair is the working-dtype value from the healthy tile geometry, bit-identical to the call in
    which no entry anywhere in the batch needed recovery, so an ill-conditioned token in another
    sequence or at a later position cannot change a healthy pair's arithmetic. The selection uses
    the recovery masks already on the device and the host flags the recovery path already reads.
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
        chol_q, rec_q = _strict_cholesky(Gq, projection_q, (*q_status[:3], decisions[1:5]))
        chol_k, rec_k = _strict_cholesky(Gk, projection_k, (*k_status[:3], decisions[5:9]))
        # Only tiles that touch a recovered query or key entry need the float64
        # triangular solves. Healthy-healthy tiles keep the cheap einsum path in the
        # working dtype; the healthy factors round-trip bit-exactly through the cast.
        factor_trace_any = rec_q is not None or rec_k is not None
        # Healthy tiles read the working-dtype means; only factor-route tiles use float64 means.
        aq_healthy, ak_healthy = aq.movedim(-2, -3), ak.movedim(-2, -3)  # (..., H, N, r)
        if factor_trace_any:
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
            q_factor, k_factor = chol_q, chol_k                 # float64 where recovered
            q_cov_healthy = Gq                                  # working dtype
            k_prec_healthy = torch.cholesky_inverse(chol_k.to(dtype))
            # Fold token-level recover flags into one host transfer, recovery path only.
            def _token_recover_flags(mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
                return mask.movedim(-2, 0).reshape(n_tokens, -1).any(dim=-1)
            nq_flags, nk_flags = Gq.shape[-3], Gk.shape[-3]
            q_tok = (_token_recover_flags(rec_q, nq_flags) if rec_q is not None
                     else torch.zeros(nq_flags, dtype=torch.bool, device=Gq.device))
            k_tok = (_token_recover_flags(rec_k, nk_flags) if rec_k is not None
                     else torch.zeros(nk_flags, dtype=torch.bool, device=Gk.device))
            flags = torch.cat((q_tok, k_tok)).tolist()
            q_needs, k_needs = flags[:nq_flags], flags[nq_flags:]
        else:
            q_cov_healthy, k_prec_healthy = Gq, torch.cholesky_inverse(chol_k)
            q_needs = k_needs = None
        aq, ak = aq.movedim(-2, -3), ak.movedim(-2, -3)  # (..., H, N, r)
        q_cov_healthy = q_cov_healthy.movedim(-3, -4)   # (..., H, Nq, r, r)
        k_prec_healthy = k_prec_healthy.movedim(-3, -4)  # (..., H, Nk, r, r)
        if factor_trace_any:
            q_factor = q_factor.movedim(-3, -4)           # (..., H, Nq, r, r) float64
            k_factor = k_factor.movedim(-3, -4)
        logdet_q = 2 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2)
        logdet_k = 2 * chol_k.diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2)
        if factor_trace_any:
            # Healthy factors round-trip bit-exactly through the cast, so the working-dtype
            # log-determinants repeat the no-recovery arithmetic exactly.
            logdet_q_healthy = (
                2 * chol_q.to(dtype).diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2))
            logdet_k_healthy = (
                2 * chol_k.to(dtype).diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2))
        else:
            logdet_q_healthy, logdet_k_healthy = logdet_q, logdet_k
        nq, nk = aq.shape[-2], ak.shape[-2]
        batch_heads = aq.numel() // (nq * r)

        def _tile_steps(element_size: int, per_pair: int) -> Tuple[int, int]:
            pair_budget = max(1, (32 * 1024 ** 2) // (element_size * batch_heads * per_pair))
            q_step = max(1, min(nq, 32, pair_budget))
            return q_step, max(1, min(nk, pair_budget // q_step))

        def _tile(args: tuple, ft: bool, q_step: int, k_step: int) -> torch.Tensor:
            if torch.is_grad_enabled() and (nq > q_step or nk > k_step) and any(x.requires_grad for x in args):
                return checkpoint(_projected_kl_tile, *args, ft,
                                  use_reentrant=False, preserve_rng_state=False)
            return _projected_kl_tile(*args, ft)

        # Working-dtype grid in the healthy tile geometry: exactly the no-recovery computation.
        q_step, k_step = _tile_steps(aq_healthy.element_size(), r)
        rows = []
        for i in range(0, nq, q_step):
            columns = []
            for j in range(0, nk, k_step):
                args = (aq_healthy[..., i:i+q_step, :], ak_healthy[..., j:j+k_step, :],
                        q_cov_healthy[..., i:i+q_step, :, :], k_prec_healthy[..., j:j+k_step, :, :],
                        logdet_q_healthy[..., i:i+q_step], logdet_k_healthy[..., j:j+k_step])
                columns.append(_tile(args, False, q_step, k_step))
            rows.append(torch.cat(columns, dim=-1))
        healthy = torch.cat(rows, dim=-2)
        if not factor_trace_any:
            return healthy.to(dtype)
        # Factor route only for pairs that touch a recovered entry, selected per (b, h, i, j).
        q_rec = (rec_q.movedim(-1, -2) if rec_q is not None                       # (..., H, Nq)
                 else torch.zeros(Gq.shape[:-4] + (Gq.shape[-3], Gq.shape[-4]),
                                  dtype=torch.bool, device=Gq.device))
        k_rec = (rec_k.movedim(-1, -2) if rec_k is not None                       # (..., H, Nk)
                 else torch.zeros(Gk.shape[:-4] + (Gk.shape[-3], Gk.shape[-4]),
                                  dtype=torch.bool, device=Gk.device))
        pair_rec = q_rec.unsqueeze(-1) | k_rec.unsqueeze(-2)                      # (..., H, Nq, Nk)
        q_step, k_step = _tile_steps(aq.element_size(), r * r)
        rows = []
        for i in range(0, nq, q_step):
            columns = []
            i_rec = any(q_needs[i:i+q_step])
            for j in range(0, nk, k_step):
                value = healthy[..., i:i+q_step, j:j+k_step]
                if i_rec or any(k_needs[j:j+k_step]):
                    args = (aq[..., i:i+q_step, :], ak[..., j:j+k_step, :],
                            q_factor[..., i:i+q_step, :, :], k_factor[..., j:j+k_step, :, :],
                            logdet_q[..., i:i+q_step], logdet_k[..., j:j+k_step])
                    factor_value = _tile(args, True, q_step, k_step).to(dtype)
                    value = torch.where(pair_rec[..., i:i+q_step, j:j+k_step], factor_value, value)
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


def projected_noisy_query_cost(
    a_q: torch.Tensor,  # (..., Nq, H, r)
    G_q: torch.Tensor,  # (..., Nq, H, r, r)
    a_k: torch.Tensor,  # (..., Nk, H, r)
    G_k: torch.Tensor,  # (..., Nk, H, r, r)

    *,
    projection_q: Optional[Tuple[torch.Tensor, ...]] = None,
    projection_k: Optional[Tuple[torch.Tensor, ...]] = None,
) -> torch.Tensor:      # (..., H, Nq, Nk)
    r"""Compute D_ij = -log N(a_i; b_j, W_ij) - log det(2 pi G_q,i)/2 with W_ij = G_q,i + G_k,j.

    With delta_ij = a_i - b_j,

        D_ij = (delta_ij^T W_ij^{-1} delta_ij + log det W_ij - log det G_q,i) / 2 >= 0.

    softmax_j(-D_ij + log pi_ij) is the exact source posterior when the projected query mean is a
    noisy measurement, with covariance G_q,i, of a draw from the key law N(b_j, G_k,j);
    N(a_i; b_j, W_ij) is the expected likelihood kernel of the two laws (Jebara, Kondor and
    Howard, 2004). The log det G_q,i term is a row constant: the softmax ignores it, and it makes
    D_ij nonnegative because W_ij > G_q,i. Each pair needs its own r x r Cholesky factor of W_ij.

    Factorization and recovery follow projected_gaussian_kl: zero jitter, no covariance repair,
    float64 source recovery for flagged entries, bounded checkpointed pair tiles, and the float64
    route selected per pair, so healthy pairs are bit-identical to a call without recovery.
    Recovered entries enter the working-dtype grid as identity placeholders, so no factorization
    of a numerically non-SPD sum runs even for a pair whose value the factor route replaces.
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
        chol_q, rec_q = _strict_cholesky(Gq, projection_q, (*q_status[:3], decisions[1:5]))
        chol_k, rec_k = _strict_cholesky(Gk, projection_k, (*k_status[:3], decisions[5:9]))
        eye = torch.eye(r, device=Gq.device, dtype=dtype)
        # Working-dtype inputs; recovered entries become identity placeholders.
        Gq_work = Gq if rec_q is None else torch.where(rec_q[..., None, None], eye, Gq)
        Gk_work = Gk if rec_k is None else torch.where(rec_k[..., None, None], eye, Gk)
        # Healthy factors round-trip bit-exactly through the cast.
        logdet_q_work = 2 * chol_q.to(dtype).diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2)
        work = (aq.movedim(-2, -3), ak.movedim(-2, -3),             # (..., H, N, r)
                Gq_work.movedim(-3, -4), Gk_work.movedim(-3, -4),   # (..., H, N, r, r)
                logdet_q_work)                                      # (..., H, Nq)
        nq, nk = work[0].shape[-2], work[1].shape[-2]
        batch_heads = work[0].numel() // (nq * r)

        def _tile_steps(element_size: int) -> Tuple[int, int]:
            # Each pair holds W_ij and its factor: two r x r matrices.
            pair_budget = max(1, (32 * 1024 ** 2) // (element_size * batch_heads * 2 * r * r))
            q_step = max(1, min(nq, 32, pair_budget))
            return q_step, max(1, min(nk, pair_budget // q_step))

        def _slice(route: tuple, i: int, j: int, q_step: int, k_step: int) -> tuple:
            return (route[0][..., i:i+q_step, :], route[1][..., j:j+k_step, :],
                    route[2][..., i:i+q_step, :, :], route[3][..., j:j+k_step, :, :],
                    route[4][..., i:i+q_step])

        def _tile(args: tuple, q_step: int, k_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if torch.is_grad_enabled() and (nq > q_step or nk > k_step) and any(x.requires_grad for x in args):
                return checkpoint(_noisy_query_tile, *args, use_reentrant=False, preserve_rng_state=False)
            return _noisy_query_tile(*args)

        failures = []
        # Working-dtype grid in the healthy tile geometry: exactly the no-recovery computation.
        q_step, k_step = _tile_steps(aq.element_size())
        rows = []
        for i in range(0, nq, q_step):
            columns = []
            for j in range(0, nk, k_step):
                value, failed = _tile(_slice(work, i, j, q_step, k_step), q_step, k_step)
                columns.append(value)
                failures.append(failed)
            rows.append(torch.cat(columns, dim=-1))
        score = torch.cat(rows, dim=-2)                             # (..., H, Nq, Nk)
        if rec_q is not None or rec_k is not None:
            # Factor route only for pairs that touch a recovered entry, selected per (b, h, i, j),
            # from float64 moments rebuilt from the (possibly recovered) factors and sources.
            means = [aq, ak]
            for role, projection in enumerate((projection_q, projection_k)):
                if projection is not None and len(projection) == 4:
                    _, inverse, S, mu = projection
                    means[role] = (_projected_linear_map(S.double(), inverse.double())
                                   @ mu.double().unsqueeze(-1)).squeeze(-1)
            factor_q, factor_k = chol_q.double(), chol_k.double()
            exact = (means[0].double().movedim(-2, -3), means[1].double().movedim(-2, -3),
                     (factor_q @ factor_q.transpose(-1, -2)).movedim(-3, -4),
                     (factor_k @ factor_k.transpose(-1, -2)).movedim(-3, -4),
                     2 * factor_q.diagonal(dim1=-2, dim2=-1).log().sum(-1).transpose(-1, -2))
            q_rec = (rec_q.movedim(-1, -2) if rec_q is not None                       # (..., H, Nq)
                     else torch.zeros(Gq.shape[:-4] + (Gq.shape[-3], Gq.shape[-4]),
                                      dtype=torch.bool, device=Gq.device))
            k_rec = (rec_k.movedim(-1, -2) if rec_k is not None                       # (..., H, Nk)
                     else torch.zeros(Gk.shape[:-4] + (Gk.shape[-3], Gk.shape[-4]),
                                      dtype=torch.bool, device=Gk.device))
            pair_rec = q_rec.unsqueeze(-1) | k_rec.unsqueeze(-2)                      # (..., H, Nq, Nk)
            # Fold token-level recovery flags into one host transfer, recovery path only.
            flags = torch.cat((q_rec.movedim(-1, 0).reshape(nq, -1).any(dim=-1),
                               k_rec.movedim(-1, 0).reshape(nk, -1).any(dim=-1))).tolist()
            q_needs, k_needs = flags[:nq], flags[nq:]
            q_step, k_step = _tile_steps(torch.finfo(torch.float64).bits // 8)
            rows = []
            for i in range(0, nq, q_step):
                columns = []
                for j in range(0, nk, k_step):
                    value = score[..., i:i+q_step, j:j+k_step]
                    if any(q_needs[i:i+q_step]) or any(k_needs[j:j+k_step]):
                        exact_value, failed = _tile(_slice(exact, i, j, q_step, k_step), q_step, k_step)
                        value = torch.where(pair_rec[..., i:i+q_step, j:j+k_step],
                                            exact_value.to(dtype), value)
                        failures.append(failed)
                    columns.append(value)
                rows.append(torch.cat(columns, dim=-1))
            score = torch.cat(rows, dim=-2)
        # One device-side reduction and one host check for every tile factorization.
        if bool(torch.stack(failures).any()):
            raise ValueError("noisy-query covariance G_q + G_k is not positive definite: "
                             "zero-jitter Cholesky failed; no repair applied")
        return score


def _noisy_query_tile(
    aq:       torch.Tensor,  # (..., H, I, r)
    ak:       torch.Tensor,  # (..., H, J, r)
    Gq:       torch.Tensor,  # (..., H, I, r, r)
    Gk:       torch.Tensor,  # (..., H, J, r, r)
    logdet_q: torch.Tensor,  # (..., H, I)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute one tile of D_ij = (delta^T W_ij^{-1} delta + log det W_ij - log det G_q,i)/2, W_ij = G_q,i + G_k,j."""
    delta = aq.unsqueeze(-2) - ak.unsqueeze(-3)                    # (..., H, I, J, r)
    W = Gq.unsqueeze(-3) + Gk.unsqueeze(-4)                        # (..., H, I, J, r, r), exactly symmetric
    chol, info = torch.linalg.cholesky_ex(W, check_errors=False)
    whitened = torch.linalg.solve_triangular(chol, delta.unsqueeze(-1), upper=False)
    quadratic = whitened.square().sum(dim=(-2, -1))                # (..., H, I, J)
    logdet_w = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)   # (..., H, I, J)
    return .5 * (quadratic + logdet_w - logdet_q.unsqueeze(-1)), (info != 0).any()


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
