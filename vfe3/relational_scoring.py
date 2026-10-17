r"""Exact projected Gaussian scores and transported mixture moments.

The caller owns categorical support, priors, temperature, and KL clipping.
Projected covariances are symmetrized after contraction to remove roundoff.
Successful factorization always uses zero jitter and no spectral repair.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint

from vfe3.contracts import host_check, host_read
from vfe3.runtime import ieee_fp32_matmul
from vfe3.numerics import (
    FACTOR_CONDITION_TOL, cholesky_qr2, guarded_cholesky, triangular_condition_estimate,
)


# Byte budget of one scorer tile, set by set_projection_tile_bytes: a token tile of the projected maps in
# projected_moments and _projected_root_factor (one map L, whose covariance product can require a second), a
# GEMM-fallback direct pair tile, and a mean-distance query-row tile. Larger tiles dispatch fewer kernels and hold
# more transient memory; the values change only by float32 reduction order (audit 2026-10-16 F26). The factor-route
# recovery tiles and the noisy-query pair tiles keep their own fixed budgets.
_PROJECTION_TILE_BYTES = 32 * 1024 ** 2

# A tile loop whose whole transient exceeds this is checkpointed under autograd also when a raised
# _PROJECTION_TILE_BYTES leaves it a single tile: the budget sets the tile count, never what autograd saves.
# It equals the default budget, so at the default every loop tiles and checkpoints exactly as before the knob.
_CHECKPOINT_BYTES = 32 * 1024 ** 2

# Byte budget of one factor-route recovery pair tile of projected_gaussian_kl: one r x r triangular-solve operand
# per pair in each gathered (batch, head) slice that holds a recovered entry (audit 2026-10-16 F28).
_RECOVERY_PAIR_TILE_BYTES = 32 * 1024 ** 2


def set_projection_tile_bytes(nbytes: int) -> int:
    r"""Set the scorer tile budget ``_PROJECTION_TILE_BYTES``, in bytes; returns the previous value."""
    if type(nbytes) is not int or nbytes < 1:
        raise ValueError(f"projection tile budget must be a positive int of bytes, got {nbytes!r}")
    global _PROJECTION_TILE_BYTES
    previous = _PROJECTION_TILE_BYTES
    _PROJECTION_TILE_BYTES = nbytes
    return previous


def projection_tile_bytes() -> int:
    r"""Return the scorer tile budget ``_PROJECTION_TILE_BYTES``, in bytes."""
    return _PROJECTION_TILE_BYTES

# Cumulative (token, head) covariance entries whose Cholesky factor was rebuilt from source by the
# recovery path (float32 shifted CholeskyQR3, or float64 QR where the float32 retry fails, the input is
# float64, or the route is cholesky_qr2_certified), per device. Updated asynchronously on the recovery
# path only, so the healthy path pays nothing (audit 2026-09-23 low: the recovery had no activation counter).
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


# Suggested tolerance, in nats, for the opt-in GEMM fallback (``gemm_expansion_tol`` of
# :func:`projected_gaussian_kl`, config ``pair_kl_gemm_tol``) on the a-priori forward-error estimate of
# the centered pair-KL GEMM (:func:`_gemm_expansion_error`). A (batch, head) slice whose estimate exceeds
# it is scored by the direct-difference tiles instead (audit 2026-10-09 #7). 1e-3 nats is a logit error of 1e-3 / tau:
# below the bf16 resolution of a unit-scale logit (2^-8) and two orders under the routing gaps
# that decide a softmax, while the direct tiles' own error on the audit sweep was 3e-6 to 8e-5.
_GEMM_EXPANSION_TOL: float = 1e-3

# Cumulative (batch, head) slices routed from the GEMM to the direct tiles. The verdict is already on
# the host (it joins the scorer's single decision read), so a host integer suffices.
_GEMM_FALLBACK_SLICES: int = 0


def gemm_fallback_slices() -> int:
    r"""(batch, head) slices whose pair KL took the direct tiles since the last reset."""
    return _GEMM_FALLBACK_SLICES


def reset_gemm_fallback_slices() -> None:
    r"""Zero the GEMM fallback slice counter."""
    global _GEMM_FALLBACK_SLICES
    _GEMM_FALLBACK_SLICES = 0


# Conditioning-triggered source recovery (on under config ``projected_factorization`` "gram_certified"
# and "cholesky_qr2_certified", installed by VFEModel). Off, only entries whose zero-jitter float32
# Cholesky FAILS are rebuilt from source; entries that factor but fail the certificate
# eps_32 ||G||_F ||G^{-1}||_F <= tol keep their float32 factor, and the certificate itself is not
# evaluated. The Gram-route tolerance is GRAM_CONDITION_TOL (1e-4, i.e. kappa_F(G) > 838.9 in float32).
GRAM_CONDITION_TOL: float = 1e-4
_CONDITION_RECOVERY: bool = False

# Working precision of a source recovery: fp32 on the Gram routes and the uncertified factor route. The flagged
# factors are rebuilt by float32 shifted CholeskyQR3 of the source root (backward stable for kappa_2(X) <~ 1/eps_32;
# Fukaya et al., SIAM J. Sci. Comput. 42 (2020) A477), the factor stays in the working dtype, and the touched pairs
# are evaluated by triangular-solve tiles in the working dtype. A call whose float32 retry fails, a float64 input,
# and every rebuild on the certified factor route (cholesky_qr2_certified, where a float32 rebuild would keep the
# O(eps_32 kappa) error its certificate rejected) take the float64 recovery: float64 QR, the factor tensor promoted
# to float64, and float64 triangular-solve tiles for every pair touching a recovered entry.


def projected_gram_condition_tol() -> float:
    r"""Return the Gram-route certificate tolerance GRAM_CONDITION_TOL."""
    return GRAM_CONDITION_TOL


def set_projected_condition_recovery(enabled: bool) -> bool:
    r"""Set conditioning-triggered source recovery; returns the previous value."""
    if not isinstance(enabled, bool):
        raise ValueError(f"projected condition recovery must be bool, got {enabled!r}")
    global _CONDITION_RECOVERY
    previous = _CONDITION_RECOVERY
    _CONDITION_RECOVERY = enabled
    return previous


def projected_condition_recovery() -> bool:
    r"""Return whether factored but ill-conditioned float32 entries take source recovery."""
    return _CONDITION_RECOVERY


# Factor route of projected_gaussian_kl ("cholesky_qr2" under config projected_factorization=
# "cholesky_qr2_certified", installed by VFEModel;
# optimization_plan.md WS1/WS2). "gram" factors the rounded Gram G = fl(X X^T), whose factor carries
# whitened error O(eps kappa(X)^2). "cholesky_qr2" factors the source root X = S U^{-1} Sigma^{1/2}
# by CholeskyQR2: the same Gaussian, whitened error O(eps kappa(X)). Its recovery certificate is
# eps kappa_1(C) > FACTOR_CONDITION_TOL on the factor itself (see _factor_status); flagged and failed factors
# are rebuilt in float64, by QR of the float64 source root, and their pairs take the float64 tiles
# (see _strict_cholesky).
_FACTOR_ROUTES = ("gram", "cholesky_qr2")
_FACTOR_ROUTE: str = "gram"


def set_projected_factor_route(route: str) -> str:
    r"""Set the projected Gaussian-KL factor route; returns the previous value."""
    if route not in _FACTOR_ROUTES:
        raise ValueError(f"projected factor route must be one of {_FACTOR_ROUTES}, got {route!r}")
    global _FACTOR_ROUTE
    previous = _FACTOR_ROUTE
    _FACTOR_ROUTE = route
    return previous


def projected_factor_route() -> str:
    r"""Return the active projected Gaussian-KL factor route."""
    return _FACTOR_ROUTE


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


@ieee_fp32_matmul()
def projected_moments(
    mu_h:           torch.Tensor,  # (..., N, H, d)
    Sigma_h:        torch.Tensor,  # (..., N, H, d) or (..., N, H, d, d)
    inverse_blocks: torch.Tensor,  # (..., N, H, d, d)
    S:              torch.Tensor,  # (H, r, d) or (R, H, r, d)

    *,
    head_dims:      Optional[Tuple[int, ...]] = None,  # caller-declared equal diagonal blocks of a full Sigma
    diagonal_only:  bool = False,                      # return diag(G_i), (..., N, H, r), instead of G_i
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
    ``head_dims`` is the caller's declaration that a full ``Sigma_h`` is block-diagonal with these equal
    diagonal blocks (the ``gaussian_block`` family); the covariance contraction then reads only the
    blocks (:func:`_head_block_covariance`) instead of multiplying the dense K x K matrix once per head
    and role. The structure is declared, never inferred from values; ``None`` keeps the dense product,
    and a diagonal ``Sigma_h`` ignores it. Source-root recovery still reads the dense covariance.
    ``diagonal_only=True`` returns the diagonal of :math:`G_i` in place of the matrix, for a caller that scores only
    it (the diagonal route of :func:`projected_gaussian_kl`): from the source root of a diagonal covariance it is
    :math:`\operatorname{diag}(G_i) = \sum_k X_{i,\cdot k}^2`, with no :math:`XX^\top` product, and from a full
    covariance it is the diagonal of the same :math:`G_i`.
    Returns (..., N, H, r) and (..., N, H, r, r) for a single map (the second (..., N, H, r) under ``diagonal_only``).
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
    if head_dims is not None:
        head_dims = tuple(head_dims)
        if (not head_dims or any(type(m) is not int or m < 1 for m in head_dims)
                or len(set(head_dims)) != 1 or sum(head_dims) != d):
            raise ValueError(f"head_dims must be equal positive block sizes summing to {d}, got {head_dims!r}")
        if diagonal:
            head_dims = None
    dtype = _compute_dtype(mu_h, Sigma_h, inverse_blocks, S)
    tokens = mu_h.shape[-3]
    batch = torch.broadcast_shapes(mu_h.shape[:-3], inverse_blocks.shape[:-4])
    roles = S.shape[0] if S.ndim == 4 else 1
    bytes_per_token = math.prod(batch) * roles * H * S.shape[-2] * d * (torch.finfo(dtype).bits // 8)
    # Small projections keep the single contraction. Large projections checkpoint
    # token tiles: tiling alone would retain every L and weighted L for backward.
    if tokens * bytes_per_token > min(_PROJECTION_TILE_BYTES, _CHECKPOINT_BYTES):
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
                a, G = checkpoint(_projected_moments_tile, *args, head_dims, diagonal_only,
                                  use_reentrant=False, preserve_rng_state=False)
            else:
                a, G = _projected_moments_tile(*args, head_dims, diagonal_only)
            means.append(a)
            covariances.append(G)
        return torch.cat(means, dim=-3), torch.cat(covariances, dim=-3 if diagonal_only else -4)
    return _projected_moments_tile(mu_h, Sigma_h, inverse_blocks, S, head_dims, diagonal_only)


@ieee_fp32_matmul()
def _projected_moments_tile(
    mu_h:           torch.Tensor,
    Sigma_h:        torch.Tensor,
    inverse_blocks: torch.Tensor,
    S:              torch.Tensor,
    head_dims:      Optional[Tuple[int, ...]] = None,
    diagonal_only:  bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute a=L mu and G=L Sigma L^T for one token tile.

    ``S`` is ``(H, r, d)`` for one comparison map or ``(R, H, r, d)`` for a
    role-stacked pair; the stacked form returns a leading ``R`` axis and batches
    the covariance contraction across roles while keeping one frame map.
    ``head_dims`` (validated by :func:`projected_moments`) routes a full covariance through
    :func:`_head_block_covariance`. A diagonal covariance takes the source root
    (:func:`_diagonal_root_moments`). The tile is pinned to IEEE fp32 itself, so a checkpointed
    recompute in backward runs under the same matmul policy as the forward (audit 2026-10-09 #15).
    ``diagonal_only`` returns diag(G) (see :func:`projected_moments`).
    """
    diagonal = Sigma_h.shape == mu_h.shape
    dtype = _compute_dtype(mu_h, Sigma_h, inverse_blocks, S)
    with torch.autocast(device_type=mu_h.device.type, enabled=False):
        inverse = inverse_blocks.to(dtype)
        if diagonal:
            # G = X X^T with X = S U^{-1} diag(sigma)^{1/2}; neither L = S U^{-1} nor L Sigma is formed
            # (audit 2026-10-10 #28). Role-stacked maps are one (R H, r, K) contraction sharing xi and W.
            maps = (S if S.ndim == 4 else S.unsqueeze(0)).to(dtype)                    # (R, H, r, K)
            a, X = _diagonal_root_moments(_head_shared(mu_h).to(dtype), _head_shared(Sigma_h).to(dtype),
                                          inverse, maps)                               # (..., N, R, H, r[, K])
            if diagonal_only:
                g = X.square().sum(-1)                                                  # (..., N, R, H, r) diag(X X^T)
                if S.ndim == 3:
                    return a.squeeze(-3), g.squeeze(-3)                                 # (..., N, H, r)
                return a.movedim(-3, 0).contiguous(), g.movedim(-3, 0).contiguous()    # (R, ..., N, H, r)
            G = X @ X.transpose(-1, -2)                                                 # (..., N, R, H, r, r)
            if S.ndim == 3:
                a, G = a.squeeze(-3), G.squeeze(-4)                                     # (..., N, H, r[, r])
            else:
                # The layout of a role-stacked contraction, (R, ..., N, H, r[, r]).
                a, G = a.movedim(-3, 0).contiguous(), G.movedim(-4, 0).contiguous()
        elif S.ndim == 3:
            mu = mu_h.to(dtype)
            Sigma = Sigma_h.to(dtype)
            L = _projected_linear_map(S.to(dtype), inverse)  # (..., N, H, r, d)
            a = (L @ mu.unsqueeze(-1)).squeeze(-1)
            if head_dims is not None:
                G = _head_block_covariance(L, Sigma, len(head_dims), S.ndim == 4)
            else:
                G = L @ Sigma @ L.transpose(-1, -2)
        else:
            mu = mu_h.to(dtype)
            Sigma = Sigma_h.to(dtype)
            # Role-stacked maps: one map per role, batched into a single G contraction.
            L = torch.stack(
                [_projected_linear_map(S[role].to(dtype), inverse) for role in range(S.shape[0])],
                dim=0)  # (R, ..., N, H, r, d)
            a = (L @ mu.unsqueeze(-1).unsqueeze(0)).squeeze(-1)  # (R, ..., N, H, r)
            if head_dims is not None:
                G = _head_block_covariance(L, Sigma, len(head_dims), S.ndim == 4)
            else:
                G = L @ Sigma.unsqueeze(0) @ L.transpose(-1, -2)
        if diagonal_only:
            return a, G.diagonal(dim1=-2, dim2=-1)                                      # the symmetrized diagonal
        # The two off-diagonal contractions can round differently near zero.
        # Keep the covariance exactly symmetric for strict zero-jitter Cholesky.
        G = .5 * G + .5 * G.transpose(-1, -2)
    return a, G


def _head_shared(x: torch.Tensor) -> torch.Tensor:  # (..., N, H, K) -> (..., N, 1, K) when head-broadcast
    r"""The head-0 slice of a head-broadcast tensor (stride 0 on the head axis, as ``expand`` builds it)."""
    return x[..., :1, :] if x.stride(-2) == 0 else x


def _diagonal_root_moments(
    mu:      torch.Tensor,             # (..., N, h, K), h = 1 for a head-shared mean or H
    sigma:   torch.Tensor,             # (..., N, h, K) diagonal variances, h = 1 or H
    inverse: torch.Tensor,             # (..., N, H', K, K) dense or (..., N, M, d, d) compact U^{-1} blocks
    S:       torch.Tensor,             # (R, H, r, K) comparison maps, one per role
) -> Tuple[torch.Tensor, torch.Tensor]:  # a (..., N, R, H, r), X (..., N, R, H, r, K)
    r"""Projected means and source roots of a diagonal covariance for every role's comparison map.

    With :math:`\xi_i = U_i^{-1}\mu_i` and :math:`W_i = U_i^{-1}\operatorname{diag}(\sigma_i)^{1/2}`,

    .. math::
        a_i = S\xi_i, \qquad X_i = S W_i, \qquad
        X_i X_i^\top = S U_i^{-1}\operatorname{diag}(\sigma_i) U_i^{-\top} S^\top = G_i,

    so neither :math:`L = S U^{-1}` nor :math:`L\operatorname{diag}(\sigma)` is formed, and :math:`W` keeps
    the block structure of compact :math:`U^{-1}` blocks. A head-shared mean or variance (``h = 1``, the
    head-0 slice of :func:`_head_shared`) gives :math:`\xi` or :math:`W` once per token instead of once per
    head, and the roles then contract as one stacked ``(R H, r, K)`` map.

    A zero variance takes the root 0 with zero gradient (audit 2026-10-10 round 2, item 40): the derivative
    :math:`1/(2\sqrt{\sigma})` of the plain root is infinite there and its product with the zero cotangent of the
    root would be NaN, while :math:`L\operatorname{diag}(\sigma)L^\top` has a finite derivative. Every other value,
    a negative or NaN variance included, is the plain ``sigma.sqrt()`` bit for bit, so an invalid variance still
    reaches the scorer's finiteness and positivity checks.
    """
    R, H, r, K = S.shape
    zero = sigma == 0
    root = torch.where(zero, 0., torch.where(zero, 1., sigma).sqrt())              # (..., N, h, K)
    if inverse.shape[-2:] == (K, K):
        # Dense blocks (..., N, H', K, K): W = U^{-1} diag(sigma)^{1/2} column-scales each block.
        xi = (inverse @ mu.unsqueeze(-1)).squeeze(-1)                              # (..., N, h, K)
        W = inverse * root.unsqueeze(-2)                                           # (..., N, h, K, K)
        X = (_projected_linear_map(S.flatten(0, 1), W).unflatten(-3, (R, H)) if W.shape[-3] == 1
             else _projected_linear_map(S, W.unsqueeze(-4)))                       # (..., N, R, H, r, K)
    else:
        M, d = inverse.shape[-3], inverse.shape[-1]
        blocks = inverse.unsqueeze(-4)                                             # (..., N, 1, M, d, d)
        xi = (blocks @ mu.unflatten(-1, (M, d)).unsqueeze(-1)).flatten(-3)         # (..., N, h, K)
        W = blocks * root.unflatten(-1, (M, d)).unsqueeze(-2)                      # (..., N, h, M, d, d)
        X = (_projected_linear_map(S.flatten(0, 1), W.squeeze(-4)).unflatten(-3, (R, H)) if W.shape[-4] == 1
             else torch.einsum("Rhrmd,...hmde->...Rhrme", S.unflatten(-1, (M, d)), W).flatten(-2))
    if xi.shape[-2] == 1:
        # One (tokens, K) x (K, R H r) GEMM, not a broadcast batched matvec per (token, role, head).
        a = (xi.squeeze(-2) @ S.flatten(0, 2).transpose(0, 1)).unflatten(-1, (R, H, r))
    else:
        a = torch.einsum("Rhrk,...hk->...Rhr", S, xi)                              # (..., N, R, H, r)
    return a, X


def _head_block_covariance(
    L:        torch.Tensor,  # (..., N, H, r, K), optionally with a leading role axis R
    Sigma:    torch.Tensor,  # (..., N, H, K, K), block-diagonal with n_blocks equal d x d blocks
    n_blocks: int,
    stacked:  bool,          # L carries a leading role axis R
) -> torch.Tensor:           # (..., N, H, r, r), or (R, ..., N, H, r, r) when stacked
    r"""Return :math:`G_h = \sum_m L_h^{(m)} \Sigma_m L_h^{(m)\top}` for a head-block-diagonal covariance.

    With :math:`\Sigma = \operatorname{diag}(\Sigma_1, \ldots, \Sigma_M)` and :math:`L_h` split into the
    column blocks :math:`L_h^{(m)}` of width :math:`d = K / M`, :math:`L_h \Sigma L_h^\top` reduces to the
    sum above, which costs :math:`O(H r K d)` per token instead of :math:`O(H r K^2)`. The diagonal blocks
    are read through a strided view (no copy), and a head-broadcast covariance (stride zero on the head
    axis, as built by ``expand``) is read once per token rather than once per head and role; its
    contraction makes one copy of L in block-major order and never forms a dense K x K product. The caller
    declares the block structure; off-block entries of ``Sigma`` are never read, so their gradient is zero.
    """
    d = Sigma.shape[-1] // n_blocks
    blocks = (Sigma.unflatten(-1, (n_blocks, d)).unflatten(-3, (n_blocks, d))
              .diagonal(dim1=-4, dim2=-2).movedim(-1, -3))                    # (..., N, H, M, d, d) view
    if Sigma.stride(-3) != 0:
        # Per-head covariances: contract each head's own blocks.
        L_blocks = L.unflatten(-1, (n_blocks, d))                             # (..., N, H, r, M, d)
        weighted = torch.einsum("...hrmd,...hmde->...hrme", L_blocks, blocks)  # (..., N, H, r, M, d)
        return torch.einsum("...hrme,...hsme->...hrs", weighted, L_blocks)    # (..., N, H, r, r)
    # Head-broadcast covariance: block axis first, so each block's (role, head, rank) rows are one
    # contiguous slab that multiplies its d x d block with no broadcast of Sigma over heads or roles.
    shared = blocks.select(-4, 0).movedim(-3, 0)                              # (M, ..., N, d, d)
    rows = L.unflatten(-1, (n_blocks, d)).movedim(-2, 0)                      # (M, [R,] ..., N, H, r, d)
    if stacked:
        rows = rows.movedim(1, -4)                                            # (M, ..., N, R, H, r, d)
    batch, r = rows.shape[1:-4 if stacked else -3], rows.shape[-2]
    rows = rows.reshape(n_blocks, *batch, -1, d)                              # (M, ..., N, [R] H r, d): one copy
    shared = shared.reshape(n_blocks, *(1,) * (rows.ndim - shared.ndim), *shared.shape[1:])
    weighted = rows @ shared                                                  # (M, ..., N, [R] H r, d)
    G = None
    for m in range(n_blocks):
        term = (weighted[m].unflatten(-2, (-1, r))
                @ rows[m].unflatten(-2, (-1, r)).transpose(-1, -2))           # (..., N, [R] H, r, r)
        G = term if G is None else G.add_(term)
    if stacked:
        G = G.unflatten(-3, (L.shape[0], -1)).movedim(-4, 0)                  # (R, ..., N, H, r, r)
    return G


class _GramPrecision(torch.autograd.Function):
    r"""Zero-jitter Cholesky :math:`G = C C^\top` with :math:`P = G^{-1}` and :math:`\log\det G`, closed-form backward.

    Returns ``(chol, failed, P, logdet)``. The forward is bitwise the unfused computation: ``cholesky_ex``,
    the failure mask of :func:`_cholesky_status`, ``cholesky_inverse`` with failed entries reading the
    identity factor, and :math:`\log\det G = 2\sum_a \log C_{aa}`. For symmetric :math:`G`,
    :math:`dP = -P\,dG\,P` and :math:`d\log\det G = \operatorname{tr}(P\,dG)` give

    .. math::
        \bar G = -P \operatorname{sym}(\bar P) P + \bar\ell P + \bar G_C,
        \qquad \operatorname{sym}(X) = \tfrac12 (X + X^\top),

    two r x r products per entry instead of the two batched triangular solves of the Cholesky backward
    (audit 2026-10-10 #27). :math:`\bar G_C = C^{-\top} \Phi(C^\top \bar C) C^{-1}` is the Cholesky backward
    of a factor cotangent :math:`\bar C` (Murray, arXiv:1602.07527, the formula autograd applies to
    ``cholesky_ex``), present only when a recovery path reads the factor itself. Every term is symmetric,
    the convention of torch's Cholesky backward. Failed entries receive zero gradient. The forward and
    the backward run at IEEE fp32.
    """

    @staticmethod
    def forward(ctx, G: torch.Tensor) -> Tuple[torch.Tensor, ...]:  # G: (..., r, r)
        with ieee_fp32_matmul():
            chol, info = torch.linalg.cholesky_ex(G, check_errors=False)
            failed = (info != 0) | ~torch.isfinite(chol).all(dim=(-2, -1))                 # (...)
            eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
            P = torch.cholesky_inverse(torch.where(failed[..., None, None], eye, chol))      # (..., r, r)
        logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)                         # (...)
        ctx.save_for_backward(chol, P, failed)
        ctx.mark_non_differentiable(failed)
        ctx.set_materialize_grads(False)
        return chol, failed, P, logdet

    @staticmethod
    def backward(
        ctx,
        g_chol:   Optional[torch.Tensor],  # (..., r, r)
        g_failed: Optional[torch.Tensor],
        g_P:      Optional[torch.Tensor],  # (..., r, r)
        g_logdet: Optional[torch.Tensor],  # (...)
    ) -> Optional[torch.Tensor]:           # (..., r, r) symmetric
        chol, P, failed = ctx.saved_tensors
        grad = None
        with ieee_fp32_matmul():
            if g_P is not None:
                grad = -(P @ (.5 * (g_P + g_P.transpose(-1, -2))) @ P)
            if g_logdet is not None:
                term = g_logdet[..., None, None] * P
                grad = term if grad is None else grad + term
            if g_chol is not None:
                phi = (chol.transpose(-1, -2) @ g_chol).tril()
                phi = .5 * (phi + phi.tril(-1).transpose(-1, -2))
                term = torch.linalg.solve_triangular(chol.transpose(-1, -2), phi, upper=True, left=True)
                term = torch.linalg.solve_triangular(chol, term, upper=False, left=False)
                grad = term if grad is None else grad + term
        return None if grad is None else grad.masked_fill(failed[..., None, None], 0.)


class _IEEEFP32Recompute(torch.autograd.Function):
    r"""``body(*inputs)`` with the GEMMs of its forward AND its backward at IEEE fp32 (audit 2026-10-10 round 2, item 2).

    :func:`vfe3.runtime.ieee_fp32_matmul` pins only the GEMMs it encloses; the autograd backward of a pinned
    forward runs later under the caller's policy, so under ``fp32_matmul_precision="high"`` the backward of the
    expanded pair-KL GEMM rounds its operands to TF32 (:math:`2^{-10}`). That backward forms
    :math:`\partial L/\partial P_j` from terms that cancel to :math:`\sum_i g_{ij}(G_i + \delta_{ij}\delta_{ij}^\top)`,
    so its relative error grows as the comparison variance falls against the squared mean spread (O(1) in a TF32
    emulation once the ratio is below about 1e-2). Here the forward evaluates ``body`` without autograd and saves
    only its inputs; the backward re-evaluates ``body`` from them with autograd and differentiates it, both inside
    :func:`ieee_fp32_matmul`, as :class:`_GramPrecision` pins its own backward. The recomputation repeats the
    forward's operations on the same inputs, so values and gradients are those of plain autograd through ``body``
    at IEEE fp32 (bit for bit on the CPU, where the policy has no effect), and only the token-linear inputs are
    held for backward, as a checkpoint would hold them. Double backward is not supported.
    """

    @staticmethod
    def forward(ctx, body: Callable[..., torch.Tensor], *inputs: torch.Tensor) -> torch.Tensor:
        ctx.body = body
        ctx.save_for_backward(*inputs)
        with ieee_fp32_matmul():
            return body(*inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        needs = ctx.needs_input_grad[1:]
        leaves = [x.detach().requires_grad_(need) for x, need in zip(ctx.saved_tensors, needs)]
        with torch.enable_grad(), ieee_fp32_matmul():
            grads = iter(torch.autograd.grad(ctx.body(*leaves), [x for x in leaves if x.requires_grad], grad,
                                             allow_unused=True))
        return (None, *(next(grads) if need else None for need in needs))


def _ieee_fp32_autograd(body: Callable[..., torch.Tensor], *inputs: torch.Tensor) -> torch.Tensor:
    r"""``body(*inputs)`` through :class:`_IEEEFP32Recompute` under autograd, or directly when no input needs a gradient."""
    if torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
        return _IEEEFP32Recompute.apply(body, *inputs)
    return body(*inputs)


@ieee_fp32_matmul()
def _cholesky_status(
    G:         torch.Tensor,
    symmetric: bool = False,
    factor:    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,  # (chol, failed, P) of G
) -> Tuple[torch.Tensor, ...]:
    r"""Factor G and certify eps ||G||_F ||G^{-1}||_F <= tol for every float32 entry on device.

    ``symmetric=True`` is the caller's certificate that G is bitwise symmetric (built as
    0.5 (G + G^T)), and skips the elementwise symmetry check; externally supplied G keeps it.
    ``factor`` is the ``(chol, failed, P)`` of G from :class:`_GramPrecision`: the factorization is then
    not repeated, and the certificate reads that ``P``, bitwise the inverse it would otherwise form.

    ``tol`` is ``GRAM_CONDITION_TOL`` (1e-4).

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
    if factor is None:
        chol, info = torch.linalg.cholesky_ex(G, check_errors=False)
    else:
        chol, failed, inverse = factor
    with torch.no_grad():
        if factor is None:
            failed = (info != 0) | ~torch.isfinite(chol).all(dim=(-2, -1))
        screen = torch.zeros_like(failed)
        r = G.shape[-1]
        if G.dtype == torch.float32 and r > 1 and _CONDITION_RECOVERY:
            if factor is None:
                eye = torch.eye(r, device=G.device, dtype=G.dtype)
                inverse = torch.cholesky_inverse(torch.where(failed[..., None, None], eye, chol))
            inverse_norm = inverse.norm(dim=(-2, -1))
            certificate = (torch.finfo(G.dtype).eps * G.norm(dim=(-2, -1)) * inverse_norm
                           > GRAM_CONDITION_TOL)
            screen = (~failed) & certificate
        status = torch.stack((torch.isfinite(G).all(), _symmetry_status(G, symmetric),
                              (failed | screen).any(), failed.any()))
    return chol, failed, screen, status


def _symmetry_status(G: torch.Tensor, symmetric: bool) -> torch.Tensor:
    r"""On-device symmetry verdict: G^T = G to 32 eps relative, or True when the caller certifies it."""
    if symmetric:
        return torch.ones((), dtype=torch.bool, device=G.device)
    tolerance = 32 * torch.finfo(G.dtype).eps
    return torch.isclose(G, G.transpose(-1, -2), rtol=tolerance, atol=0.).all()


@ieee_fp32_matmul()
def _projected_root_factor_tile(
    Sigma:    torch.Tensor,   # (..., n, H, d) diagonal or (..., n, H, d, d) full
    inverse:  torch.Tensor,   # (..., n, H, d_b, d_b) compact or full-width inverse frame blocks
    S:        torch.Tensor,   # (H, r, d), or (n, H, r, d) per token
    dtype:    torch.dtype,
    diagonal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""CholeskyQR2 factor of X X^T for X = S U^{-1} Sigma^{1/2}, one token tile.

    Pinned to IEEE fp32 itself, so a checkpointed recompute in backward factors under the forward's policy.
    """
    with torch.autocast(device_type=S.device.type, enabled=False):
        L = _projected_linear_map(S.to(dtype), inverse.to(dtype))                # (..., n, H, r, d)
        if diagonal:
            C, failed = cholesky_qr2(L * Sigma.to(dtype).sqrt().unsqueeze(-2))
            return C, failed
        Sigma = Sigma.to(dtype)
        if Sigma.stride(-3) == 0:
            # A head-broadcast full covariance (full input scope) is factored once per token.
            root, root_failed = guarded_cholesky(Sigma[..., :1, :, :])
            root, root_failed = root.expand_as(Sigma), root_failed.expand(Sigma.shape[:-2])
        else:
            root, root_failed = guarded_cholesky(Sigma)
        C, failed = cholesky_qr2(L @ root)
        return C, failed | root_failed


def _projected_root_factor(
    projection: Tuple[torch.Tensor, ...],   # (Sigma_h, U^{-1} blocks, S[, mu_h])
    batch:      torch.Size,                 # (..., N, H), the covariance batch shape
    dtype:      torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:     # (..., N, H, r, r), (..., N, H)
    r"""Lower factor C of G = X X^T straight from the source root, by CholeskyQR2.

    X = S U^{-1} Sigma^{1/2}: diagonal Sigma scales the columns of L = S U^{-1} by sqrt(sigma), and
    full Sigma right-multiplies L by its zero-jitter Cholesky factor. No rounded Gram is factorized
    as the answer, so C, its inverse and its log-determinant carry error O(eps kappa(X)).
    Token tiles are checkpointed under the projection budget, as in :func:`projected_moments`.
    ``failed`` marks entries whose covariance or root factorization failed; their factor is finite
    with finite gradients so the recovery path may overwrite it.
    """
    Sigma, inverse, S = projection[:3]
    r, width = S.shape[-2:]
    diagonal = Sigma.shape == (*batch, width)
    if not diagonal and Sigma.shape != (*batch, width, width):
        raise ValueError("source covariance shape does not match projected token/head dimensions")
    tokens = batch[-2]
    bytes_per_token = math.prod(batch[:-2]) * batch[-1] * r * width * (torch.finfo(dtype).bits // 8)
    if tokens * bytes_per_token <= min(_PROJECTION_TILE_BYTES, _CHECKPOINT_BYTES):
        return _projected_root_factor_tile(Sigma, inverse, S, dtype, diagonal)
    step = max(1, min(tokens, 32, _PROJECTION_TILE_BYTES // bytes_per_token))
    factors, failures = [], []
    for start in range(0, tokens, step):
        sigma = (Sigma[..., start:start+step, :, :] if diagonal
                 else Sigma[..., start:start+step, :, :, :])
        tile_inverse = (inverse[..., start:start+step, :, :, :]
                        if inverse.ndim >= 4 and inverse.shape[-4] == tokens else inverse)
        # A per-token comparison map (N, H, r, d) (comparison-space RoPE, R_i S) is tiled with the tokens.
        tile_S = S[start:start+step] if S.ndim == 4 else S
        args = (sigma, tile_inverse, tile_S)
        if torch.is_grad_enabled() and any(x.requires_grad for x in args):
            C, failed = checkpoint(_projected_root_factor_tile, *args, dtype, diagonal,
                                   use_reentrant=False, preserve_rng_state=False)
        else:
            C, failed = _projected_root_factor_tile(*args, dtype, diagonal)
        factors.append(C)
        failures.append(failed)
    return torch.cat(factors, dim=-4), torch.cat(failures, dim=-2)


@ieee_fp32_matmul()
def _factor_status(
    G:          torch.Tensor,                # (..., N, H, r, r) explicit Gram, working dtype
    projection: Tuple[torch.Tensor, ...],
    symmetric:  bool = False,                # caller certifies G bitwise symmetric
) -> Tuple[torch.Tensor, ...]:
    r"""Factor-route analogue of :func:`_cholesky_status`: (chol, failed, screen, status).

    The factor is the CholeskyQR2 factor of the SOURCE root, not of the rounded G. Its certificate
    (only under :func:`projected_condition_recovery`) is

    .. math::
        \varepsilon \, \hat\kappa_1(C) > \mathtt{\_FACTOR\_CONDITION\_TOL} = 3 \times 10^{-4},

    with :math:`\hat\kappa_1` the Hager-Higham estimate (O(r^2) per entry, no eigensolver). The
    factor's whitened error is at most :math:`c\,\varepsilon\,\kappa_2(X)`, measured c <= 2.7, and
    :math:`\kappa_2 \le \hat\kappa_1` on the saved runs (the 1-norm reads 5-9 times higher at
    r = 32). To first order every pair KL touching an entry then moves by at most
    :math:`\varepsilon (\kappa_{q,i} + \kappa_{k,j}) (\mathrm{tr}\,P_k G_q + \delta^T P_k \delta + r)`,
    a model that bounded every measured Phase 0 pair error with a factor of six to spare. A
    certificate that guaranteed a logit error of 1e-3 for every pair up to ``kl_max`` would need
    :math:`\varepsilon(\kappa_q + \kappa_k) \le 5 \times 10^{-6}` and would trip in the tail of the
    observed query conditioning while the measured error sits twenty times inside that guarantee.
    The tolerance is set just above the observed maximum instead (see optimization_plan.md,
    Phase 0 results). G's finiteness and symmetry are still checked, since the trace product reads it.
    """
    chol, failed = _projected_root_factor(projection, G.shape[:-2], G.dtype)
    with torch.no_grad():
        screen = torch.zeros_like(failed)
        if G.dtype == torch.float32 and G.shape[-1] > 1 and _CONDITION_RECOVERY:
            estimate = triangular_condition_estimate(chol)
            screen = (~failed) & (torch.finfo(G.dtype).eps * estimate > FACTOR_CONDITION_TOL)
        status = torch.stack((torch.isfinite(G).all(), _symmetry_status(G, symmetric),
                              (failed | screen).any(), failed.any()))
    return chol, failed, screen, status


def _source_qr_factor(root: torch.Tensor) -> torch.Tensor:
    r"""Lower factor R^T (positive diagonal) of root root^T from a reduced QR of root^T."""
    r, d = root.shape[-2:]
    _, R = torch.linalg.qr(root.transpose(-1, -2), mode="reduced")
    pivots = R.diagonal(dim1=-2, dim2=-1)
    tolerance = torch.finfo(root.dtype).eps * max(r, d) * root.norm(dim=(-2, -1))
    if bool((pivots.abs() <= tolerance.unsqueeze(-1)).any()):
        raise ValueError("projected measurement lost numerical row rank; no covariance repair applied")
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    return R.transpose(-1, -2) * signs.unsqueeze(-2)


def _commutant_root_factor(
    root:    torch.Tensor,   # (n, r, d) source roots X of the rebuilt entries
    n_pairs: int,            # rotating planes of the comparison block
    shift:   bool,           # shifted CholeskyQR3 in root's dtype; False: reduced QR (float64 recovery)
) -> Tuple[torch.Tensor, torch.Tensor]:   # (n, r, r) lower factor of P(X X^T), (n,) failed
    r"""Lower factor of the RoPE-commutant projection :math:`P(XX^\top)` straight from the source root.

    With the rotating planes :math:`p = (2k, 2k+1)`, :math:`k <` ``n_pairs``, and the non-rotating rows
    :math:`X_0 = X_{m:}`, :math:`m = 2\,` ``n_pairs`` (:func:`vfe3.geometry.rope.rope_commutant_projection`),

    .. math::
        P(XX^\top) = \bigoplus_p \tfrac12\bigl(\lVert x_{2k}\rVert^2 + \lVert x_{2k+1}\rVert^2\bigr) I_2
            \oplus X_0X_0^\top,

    so its lower factor is the diagonal of plane standard deviations beside the factor of :math:`X_0X_0^\top`.
    ``shift`` takes that factor by shifted CholeskyQR3 (:func:`vfe3.numerics.cholesky_qr2`) and flags a plane whose
    deviation is not finite and positive; otherwise it takes the reduced QR of :math:`X_0^\top`
    (:func:`_source_qr_factor`) and refuses a plane at or below the same rank tolerance. The plane deviations are
    sums of squares, accurate to the working precision, and the non-rotating block keeps the error
    :math:`O(\varepsilon\kappa(X_0))` of the source-root factorizations; no rounded Gram of X is factored.
    """
    n, r, d = root.shape
    m = 2 * n_pairs
    plane = root[:, :m, :].square().sum(-1).unflatten(-1, (n_pairs, 2)).mean(-1)          # (n, m/2)
    deviation = plane.sqrt().repeat_interleave(2, dim=-1)                                 # (n, m)
    if shift:
        with torch.no_grad():
            failed = ~(torch.isfinite(deviation) & (deviation > 0)).all(dim=-1)            # (n,)
    else:
        tolerance = torch.finfo(root.dtype).eps * max(r, d) * root.norm(dim=(-2, -1))    # (n,)
        if bool((deviation <= tolerance.unsqueeze(-1)).any()):
            raise ValueError("projected measurement lost numerical row rank; no covariance repair applied")
        failed = torch.zeros(n, dtype=torch.bool, device=root.device)
    planes = torch.diag_embed(deviation)                                                  # (n, m, m)
    if m == r:
        return planes, failed
    if shift:
        fixed, fixed_failed = cholesky_qr2(root[:, m:, :], shift=True)                    # (n, r - m, r - m)
        failed = failed | fixed_failed
    else:
        fixed = _source_qr_factor(root[:, m:, :])
    zeros = root.new_zeros(n, m, r - m)
    factor = torch.cat((torch.cat((planes, zeros), dim=-1),
                        torch.cat((zeros.transpose(-1, -2), fixed), dim=-1)), dim=-2)    # (n, r, r)
    return factor, failed


@ieee_fp32_matmul()
def _source_root(
    G:          torch.Tensor,                # (..., r, r) working-dtype projected covariance
    projection: Tuple[torch.Tensor, ...],    # (Sigma_h, U^{-1} blocks, S[, mu_h])
    mask:       torch.Tensor,                # (...) entries whose source square root is rebuilt
) -> torch.Tensor:                           # (n, r, d) float64 roots of the masked entries
    r"""Float64 source square root X = S U^{-1} Sigma^{1/2} of the masked entries (G = X X^T).

    Index before promotion: only the flagged source square roots are formed.
    """
    Sigma, inverse, S = projection[:3]
    batch, r, d = G.shape[:-2], G.shape[-1], S.shape[-1]
    with torch.autocast(device_type=G.device.type, enabled=False):
        selected_S = S.expand(*batch, r, d)[mask].double()
        if inverse.shape[-2:] == (d, d):
            L = selected_S @ inverse.expand(*batch, d, d)[mask].double()
        else:
            G_blocks, d_block, _ = inverse.shape[-3:]
            selected_inverse = inverse.unsqueeze(-4).expand(
                *batch, G_blocks, d_block, d_block)[mask].double()
            L = _projected_linear_map(selected_S.unsqueeze(-3), selected_inverse).squeeze(-3)
        if Sigma.shape == (*batch, d):
            variance = Sigma[mask].double()
            if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
                raise ValueError("source covariance must be finite and positive for projected factorization")
            root = L * variance.sqrt().unsqueeze(-2)
        elif Sigma.shape == (*batch, d, d):
            root = L @ _strict_cholesky(Sigma[mask].double())[0]
        else:
            raise ValueError("source covariance shape does not match projected token/head dimensions")
        if not bool(torch.isfinite(root).all()):
            raise ValueError("projected source square root must be finite")
    return root


@ieee_fp32_matmul()
def _strict_cholesky(
    G:          torch.Tensor,
    projection: Optional[Tuple[torch.Tensor, ...]] = None,
    prepared:   Optional[tuple] = None,
    factored:   bool = False,
    commutant_pairs: Optional[int] = None,   # G = P(X X^T), the RoPE-commutant projection with this many planes
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Factor G with zero jitter, optionally recovering its factor from source moments.

    A failed rounded Gram matrix can lose SPD although its source measurement
    still has full row rank. For failed float32 entries, and for poorly conditioned ones under
    :func:`projected_condition_recovery`, reconstruct the source root X = S U^{-1} Sigma^{1/2}
    (formed in float64, then rounded to the working dtype) and factor it by shifted CholeskyQR3 in
    the working dtype, whose factor C satisfies C C^T = (X + dX)(X + dX)^T with
    ||dX|| <= c eps ||X|| for kappa_2(X) <~ 1/eps (Fukaya et al. 2020); the factor tensor stays in
    the working dtype. No jitter, covariance floor, or projection constraint is introduced.

    ``factored`` marks a ``prepared`` factor from :func:`_factor_status` (CholeskyQR2 of the source
    root); its healthy entries keep their prepared factor bitwise. Without the certificate
    (``cholesky_qr2``) only its outright failures are rebuilt, as above. Under
    :func:`projected_condition_recovery` (``cholesky_qr2_certified``) a float32 rebuild would carry the
    same :math:`O(\varepsilon_{32}\kappa(X))` whitened error the certificate rejected, so every rebuilt
    entry, certificate-flagged or failed (whose CholeskyQR2 failure already implies
    :math:`\varepsilon_{32}\kappa(X)^2 \gtrsim 1`), takes the float64 recovery below: the float64 QR factor of
    its float64 source root, whose pairs then take the float64 triangular-solve tiles instead of the
    centered GEMM. Without source tuples no rebuild is possible, so flagged entries keep their factor.

    A failed working-dtype retry, a float64 input, and every rebuild on the certified factor route take
    the float64 recovery for the whole call: the root is factored by float64 QR, X^T = QR, giving C = R^T.

    ``commutant_pairs`` declares that G is the projection :math:`P(XX^\top)` of the source covariance onto the
    commutant of the RoPE torus with that many rotating planes (``rope_mode='comparison_invariant'``,
    where the source tuple carries the rotated map :math:`R_iS`). A rebuilt entry then factors
    :math:`P(XX^\top)` from the source root (:func:`_commutant_root_factor`), by shifted CholeskyQR3 or in float64
    as above, instead of :math:`XX^\top` (audit 2026-10-10 round 2, item 12). The factor route never runs with it.

    Returns ``(chol, recover_mask)``. ``chol`` stays in ``G.dtype`` on the healthy
    path. When float64 source recovery is applied the whole factor tensor is promoted to
    float64 so the rebuilt entries keep their QR accuracy, and ``recover_mask``
    marks exactly those entries so the pair path can escalate only the tiles that
    touch them. ``recover_mask`` is ``None`` on the healthy path.
    """
    if prepared is None:
        chol, failed, screen, status = _cholesky_status(G)
        decisions = host_read(status)
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
        working_recovery = G.dtype != torch.float64
        # cholesky_qr2_certified: a float32 rebuild carries the O(eps_32 kappa) error the certificate rejected,
        # so every rebuilt entry (flagged or failed) is the float64 QR factor of its float64 source root.
        float64_rebuild = factored and working_recovery and _CONDITION_RECOVERY
        r = G.shape[-1]
        if projection is None:
            if failed_any:
                raise ValueError("projected covariance is not positive definite: zero-jitter Cholesky failed; no repair applied")
            if working_recovery:
                return chol, None
            chol64, _ = _strict_cholesky(G.double())
            # Whole-tensor float64 refactor: every entry takes the factor route.
            recovered_all = torch.ones_like(failed)
            _count_source_recovery(recovered_all)
            return chol64, recovered_all
        if working_recovery and not float64_rebuild:
            # Gram routes, and the uncertified factor route (``ill_conditioned`` empty: only failures rebuilt).
            rebuild = failed | ill_conditioned
            root = _source_root(G, projection, rebuild)
            with torch.autocast(device_type=G.device.type, enabled=False):
                shifted, shifted_failed = (
                    cholesky_qr2(root.to(G.dtype), shift=True) if commutant_pairs is None
                    else _commutant_root_factor(root.to(G.dtype), commutant_pairs, shift=True))
            if not bool(shifted_failed.any()):
                if factored:
                    # Failed CholeskyQR2 entries already factor the identity, so the prepared factor is
                    # safe to differentiate and is overwritten only where rebuilt.
                    base = chol
                else:
                    # Never backpropagate through a failed Cholesky, even with zero cotangent.
                    safe = torch.where(rebuild[..., None, None],
                                       torch.eye(r, device=G.device, dtype=G.dtype), G)
                    base = _strict_cholesky(safe)[0]
                _count_source_recovery(rebuild)
                return base.index_put((rebuild,), shifted.to(base.dtype)), rebuild
        failed = failed | ill_conditioned
        root = _source_root(G, projection, failed)
        with torch.autocast(device_type=G.device.type, enabled=False):
            if float64_rebuild:
                recovered = _source_qr_factor(root)
            elif factored:
                # Uncertified factor route (or a float64 input, where nothing is flagged): outright
                # CholeskyQR2 failures retry shifted CholeskyQR3 in the working dtype (valid to
                # kappa(X) ~ 1/eps), and failed retries take the float64 QR.
                shifted, shifted_failed = cholesky_qr2(root.to(G.dtype), shift=True)
                recovered = torch.where(shifted_failed[..., None, None],
                                        torch.zeros((), dtype=root.dtype, device=root.device),
                                        shifted.to(root.dtype))
                if bool(shifted_failed.any()):
                    recovered = recovered.index_put((shifted_failed,), _source_qr_factor(root[shifted_failed]))
            elif commutant_pairs is not None:
                recovered = _commutant_root_factor(root, commutant_pairs, shift=False)[0]
            else:
                recovered = _source_qr_factor(root)
        if factored:
            # Failed factor-route entries already factor the identity (guarded_cholesky), so the
            # prepared factor is safe to differentiate and is overwritten only where recovered.
            chol = chol.double()
        else:
            # Never backpropagate through a failed Cholesky, even with zero cotangent.
            safe = torch.where(failed[..., None, None], torch.eye(r, device=G.device, dtype=G.dtype), G)
            chol = _strict_cholesky(safe)[0].double()
        # Out of place: for float64 inputs .double() returns the factor autograd saved, so an in-place write
        # would fail its backward (audit 2026-10-10 remediation).
        chol = chol.index_put((failed,), recovered)
        _count_source_recovery(failed)
        return chol, failed
    return chol, None


def _same_source(
    projection_q: Optional[Tuple[torch.Tensor, ...]],
    projection_k: Optional[Tuple[torch.Tensor, ...]],
) -> bool:
    r"""Whether two source tuples read the same elements: one tuple, or per position one tensor or two
    views of one base with equal geometry (the head expansions ``relational_score`` builds per role)."""
    if projection_q is projection_k:
        return True
    if projection_q is None or projection_k is None or len(projection_q) != len(projection_k):
        return False
    return all(x is y or (x._base is not None and x._base is y._base and x.dtype == y.dtype
                          and x.shape == y.shape and x.stride() == y.stride()
                          and x.storage_offset() == y.storage_offset())
               for x, y in zip(projection_q, projection_k))


def _factor_logdet(
    chol:            torch.Tensor,            # (..., N, H, r, r) factor the caller scores with
    prepared_factor: torch.Tensor,            # the prepared factor of its role
    prepared_logdet: Optional[torch.Tensor],  # (..., N, H) from _GramPrecision, or None
) -> torch.Tensor:                            # (..., N, H)
    r""":math:`\log\det G = 2\sum_a \log C_{aa}`; the prepared value of :class:`_GramPrecision` (bitwise the
    same, closed-form backward) when ``chol`` is the prepared factor itself."""
    if prepared_logdet is not None and chol is prepared_factor:
        return prepared_logdet
    return 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def projected_gaussian_kl(
    a_q: torch.Tensor,  # (..., Nq, H, r)
    G_q: torch.Tensor,  # (..., Nq, H, r, r)
    a_k: torch.Tensor,  # (..., Nk, H, r)
    G_k: torch.Tensor,  # (..., Nk, H, r, r)

    *,
    projection_q: Optional[Tuple[torch.Tensor, ...]] = None,
    projection_k: Optional[Tuple[torch.Tensor, ...]] = None,
    symmetric_inputs: bool = False,
    query_row_constant_grad: bool = True,
    gemm_expansion_tol: Optional[float] = None,
    commutant_pairs: Optional[int] = None,
    diagonal_covariance: bool = False,
) -> torch.Tensor:      # (..., H, Nq, Nk)
    r"""Compute raw KL(q_i || k_j) = (tr(G_j^{-1}G_i) + delta^T G_j^{-1}delta - r + logdet(G_j/G_i))/2.

    Cholesky factors and inverses have only one token axis. Healthy pairs are one centered
    expanded GEMM per (batch, head) at IEEE fp32 (:func:`_projected_kl_gemm`), with no pair
    difference grid; its backward also runs at IEEE fp32 and saves only the token-linear inputs
    (:class:`_IEEEFP32Recompute`). With ``gemm_expansion_tol`` set (default None: off), a (batch, head)
    slice whose GEMM error estimate (:func:`_gemm_expansion_error`) exceeds it in nats is scored instead
    by the direct-difference tiles at IEEE fp32 and counted in :func:`gemm_fallback_slices`; the verdict
    joins the existing decision read. Recovered factors use bounded checkpointed pair tiles of triangular solves,
    over only the (batch, head) slices that hold a recovered entry (``_RECOVERY_PAIR_TILE_BYTES`` per tile),
    instead of cancelling covariance/precision products, in the working dtype, or in float64 where the
    recovery fell back to float64 (:func:`_strict_cholesky`). No floor, cap, symmetrization, jitter, or repair is applied. Optional (Sigma,U^-1,S[,mu])
    source tuples allow inaccurate rounded covariances and projected means to
    be recomputed from their original sources; covariance-only non-SPD inputs fail explicitly.

    Under :func:`projected_factor_route` ``"cholesky_qr2"`` (with both source tuples) the factors
    come from CholeskyQR2 of the source roots X = S U^{-1} Sigma^{1/2} instead of from the rounded
    Gram (whitened error O(eps kappa(X)) instead of O(eps kappa(X)^2)); the key precision and both
    log-determinants read those factors, and the query trace product still reads the explicit G_q,
    whose pair-stage cancellation sum |P_ab G_ab| / tr(P G) measured at most 18 on the saved runs.
    Under :func:`projected_condition_recovery` (``cholesky_qr2_certified``) an entry whose CholeskyQR2 factor
    fails the certificate of :func:`_factor_status`, or fails outright, is rebuilt in float64 as the QR factor of
    its float64 source root (:func:`_strict_cholesky`), and is counted in :func:`source_recovery_elements`. Its
    pairs take the float64 factor-route tiles of the float64 recovery below, with the means recomputed from the
    source in float64, and are rounded to the working dtype only at the end, so a touched pair's error falls to
    the working dtype's rounding of the result plus its healthy partner's :math:`O(\varepsilon\kappa)`.

    The factor route is selected PER PAIR (audit 2026-09-23 low): a pair (b, h, i, j) takes
    it only when its own query entry (b, i, h) or key entry (b, j, h) was recovered. Every other
    pair is the working-dtype value of the healthy GEMM, bit-identical to the call in
    which no entry anywhere in the batch needed recovery, so an ill-conditioned token in another
    sequence or at a later position cannot change a healthy pair's arithmetic. The selection uses
    the recovery masks already on the device and the host flags the recovery path already reads.

    ``symmetric_inputs=True`` certifies that G_q and G_k are bitwise symmetric (the output of
    :func:`projected_moments`, or a rotated covariance re-symmetrized as 0.5 (G + G^T)) and skips
    the elementwise symmetry check; the default keeps it for externally supplied covariances.

    ``query_row_constant_grad=False`` is the caller's declaration that its use of the cost is invariant
    to a constant added to each query row (the kernel router under ``routing_kl_max_reference='row_min'``,
    whose clamp and softmax read only the gaps :math:`D_{ij} - \min_{j'} D_{ij'}`). On the healthy path
    the query factor then feeds only :math:`-\tfrac12 \log\det G_i`, a row constant whose exact derivative
    through that use is zero, so the query factorization and its certificate run without autograd. The
    returned values, the certificate and the recovery decision are bitwise those of the default. When
    either role needs recovery, the query factor also enters the factor-route trace
    :math:`\lVert L_j^{-1} L_i \rVert_F^2`, so it is recomputed with autograd before recovery.

    ``G_q is G_k`` (``tie_comparison_maps`` scoring a belief against itself; on the factor route the two
    source tuples must also read the same tensors) factors the covariance once and both roles read that
    factorization; the values and recovery decisions are those of two factorizations.

    ``commutant_pairs`` declares that G_q and G_k are RoPE-commutant projections :math:`P(XX^\top)` of the
    source covariances with that many rotating planes (``rope_mode='comparison_invariant'``; the
    source tuples then carry the rotated maps :math:`R_iS`). Recovery rebuilds :math:`P(XX^\top)` from the
    source (:func:`_strict_cholesky`) and the factor route, which factors :math:`XX^\top`, is not taken.

    ``diagonal_covariance=True`` declares that G_q and G_k are structurally diagonal and passes their diagonals,
    shaped like the means: the diagonal route (:func:`_projected_diagonal_kl`; audit 2026-10-10 round 2,
    item 30), which forms no Gram, factorization or :math:`r^2 + r` GEMM. It has no GEMM fallback
    (``gemm_expansion_tol`` must be None) and no source recovery (the source tuples and ``commutant_pairs`` are not
    read): its covariance needs no factorization, so the certificate is that every variance is finite and positive.

    When no recovery can act (no source tuples, conditioning-triggered recovery off) and ``gemm_expansion_tol`` is
    None, every verdict can only raise, so it goes through :func:`vfe3.contracts.host_check` instead of a host
    read: inside :func:`vfe3.contracts.deferred_host_checks` it joins the next mandatory host read and the call
    takes the healthy branch without a synchronization (audit 2026-10-10 round 2, item 55); outside it raises
    at once, as before.
    """
    if a_q.ndim < 3 or a_k.ndim < 3 or a_q.shape[-2:] != a_k.shape[-2:]:
        raise ValueError("projected means must have matching head and comparison dimensions")
    r = a_q.shape[-1]
    if diagonal_covariance:
        if G_q.shape != a_q.shape or G_k.shape != a_k.shape:
            raise ValueError("diagonal projected covariances must match their means")
        if gemm_expansion_tol is not None:
            raise ValueError("the diagonal route has no GEMM fallback; gemm_expansion_tol must be None")
        return _projected_diagonal_kl(a_q, G_q, a_k, G_k, query_row_constant_grad)
    if G_q.shape != (*a_q.shape, r) or G_k.shape != (*a_k.shape, r):
        raise ValueError("projected covariances must match their means with one additional matrix axis")
    if commutant_pairs is not None and (type(commutant_pairs) is not int or not 0 <= 2 * commutant_pairs <= r):
        raise ValueError(f"commutant_pairs must be an int with 0 <= 2 * commutant_pairs <= {r}, got {commutant_pairs!r}")
    dtype = _compute_dtype(a_q, G_q, a_k, G_k)
    with torch.autocast(device_type=a_q.device.type, enabled=False):
        aq, ak = a_q.to(dtype), a_k.to(dtype)
        Gq, Gk = G_q.to(dtype), G_k.to(dtype)
        # The factor route factors the source roots; without both source tuples it has nothing to
        # factor and the Gram route runs. A commutant projection P(X X^T) has no such root factor.
        factored = (_FACTOR_ROUTE == "cholesky_qr2" and commutant_pairs is None
                    and projection_q is not None and projection_k is not None)
        def _status(G: torch.Tensor, projection: Optional[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
            return (_factor_status(G, projection, symmetric_inputs) if factored
                    else _cholesky_status(G, symmetric_inputs))

        # One covariance for both roles (tie_comparison_maps scoring a belief against itself; on the
        # factor route also one source root): the key factorization serves the query role too, detached
        # under query_row_constant_grad=False (audit 2026-10-10 #33). The Gram-route status reads G only.
        tied = G_q is G_k and (not factored or _same_source(projection_q, projection_k))
        if tied:
            Gk = Gq
        query_grad = torch.is_grad_enabled() and query_row_constant_grad
        if not tied:
            with torch.set_grad_enabled(query_grad):
                q_status = _status(Gq, projection_q)
        # The key precision from the prepared factors (failed entries read the identity) serves the
        # estimate and, on the healthy path, the GEMM: one token-linear inverse either way.
        if factored:
            k_status = _status(Gk, projection_k)
            with ieee_fp32_matmul():
                k_prec_prepared = torch.cholesky_inverse(torch.where(
                    k_status[1][..., None, None], torch.eye(r, device=Gk.device, dtype=dtype), k_status[0]))
            k_logdet_prepared = None
        else:
            # Gram route: one factorization returns the key factor, P_j and log det G_j with the closed-form
            # backward of _GramPrecision; the healthy path reads only P_j and log det G_j (audit 2026-10-10 #27).
            chol_prepared, failed_prepared, k_prec_prepared, k_logdet_prepared = _GramPrecision.apply(Gk)
            k_status = _cholesky_status(Gk, symmetric_inputs,
                                        factor=(chol_prepared, failed_prepared, k_prec_prepared))
        if tied:
            q_status, q_logdet_prepared = ((k_status, k_logdet_prepared) if query_grad else
                                           ((k_status[0].detach(), *k_status[1:]),
                                            None if k_logdet_prepared is None else k_logdet_prepared.detach()))
        else:
            q_logdet_prepared = None
        # Optional per-(batch, head) GEMM error estimate; its verdict joins the single decision read below.
        slice_shape = torch.broadcast_shapes(aq.shape[:-3], ak.shape[:-3]) + aq.shape[-2:-1]  # (..., H)
        verdicts = [(torch.isfinite(aq).all() & torch.isfinite(ak).all()).reshape(1), q_status[-1], k_status[-1]]
        if gemm_expansion_tol is not None:
            # Key entries the recovery will rebuild leave the GEMM (their pairs take the factor route), so
            # they leave the estimate too: failed or certificate-flagged entries, on either route.
            recovered_k = k_status[1] | k_status[2]
            fallback = ~(_gemm_expansion_error(aq, ak, k_prec_prepared, recovered_k) <= gemm_expansion_tol)
            verdicts.append(fallback.expand(slice_shape).reshape(-1))
        if (projection_q is None and projection_k is None and not _CONDITION_RECOVERY
                and gemm_expansion_tol is None):
            # No recovery can act and no fallback is estimated, so every verdict can only raise
            # (_strict_cholesky raises on a failed factor without a source tuple, and nothing is
            # certificate-flagged with condition recovery off): queue it and take the healthy branch.
            host_check(torch.cat(verdicts), _healthy_kl_verdict)
            fallback_slices = []
            chol_q, rec_q = q_status[0], None
            chol_k, rec_k = k_status[0], None
        else:
            # One host transfer for both roles and the means on the healthy path.
            # host_read also resolves raise-only checks deferred by vfe3.contracts.deferred_host_checks.
            decisions = host_read(torch.cat(verdicts))
            fallback_slices = [s for s, flag in enumerate(decisions[9:]) if flag]
            decisions = decisions[:9]
            if not decisions[0]:
                raise ValueError("projected means must contain only finite values")
            if not query_row_constant_grad and torch.is_grad_enabled() and (decisions[3] or decisions[7]):
                # Recovery reads the query factor beyond its log-determinant: refactor it with autograd.
                # The factorization is deterministic, so the factor, masks and decisions are unchanged.
                if tied:
                    q_status, q_logdet_prepared = k_status, k_logdet_prepared
                else:
                    q_status = _status(Gq, projection_q)
            chol_q, rec_q = _strict_cholesky(Gq, projection_q, (*q_status[:3], decisions[1:5]), factored,
                                             commutant_pairs)
            chol_k, rec_k = _strict_cholesky(Gk, projection_k, (*k_status[:3], decisions[5:9]), factored,
                                             commutant_pairs)
        # Only tiles that touch a recovered query or key entry need the factor-route
        # triangular solves (float64 after a float64 recovery). Healthy pairs keep the centered GEMM
        # in the working dtype; the healthy factors round-trip bit-exactly through the cast.
        factor_trace_any = rec_q is not None or rec_k is not None
        # Working-dtype recovery keeps both factors in the working dtype, so its factor tiles do too.
        factor_fp64 = chol_q.dtype == torch.float64 or chol_k.dtype == torch.float64
        # Healthy tiles read the working-dtype means; only float64 factor-route tiles use float64 means.
        aq_healthy, ak_healthy = aq.movedim(-2, -3), ak.movedim(-2, -3)  # (..., H, N, r)
        if factor_trace_any:
            if factor_fp64:
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
                # Both factors at float64, so the tile's triangular solves (which take the
                # triangular operand's dtype) run in float64 also when only one role was recovered.
                chol_q, chol_k = chol_q.double(), chol_k.double()
            q_factor, k_factor = chol_q, chol_k                 # float64 after a float64 recovery
            q_cov_healthy = Gq                                  # working dtype
            with ieee_fp32_matmul():                            # token-linear key precision
                k_prec_healthy = torch.cholesky_inverse(chol_k.to(dtype))
            # Fold token-level recover flags into one host transfer, recovery path only.
            def _token_recover_flags(mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
                return mask.movedim(-2, 0).reshape(n_tokens, -1).any(dim=-1)
            nq_flags, nk_flags = Gq.shape[-3], Gk.shape[-3]
            q_tok = (_token_recover_flags(rec_q, nq_flags) if rec_q is not None
                     else torch.zeros(nq_flags, dtype=torch.bool, device=Gq.device))
            k_tok = (_token_recover_flags(rec_k, nk_flags) if rec_k is not None
                     else torch.zeros(nk_flags, dtype=torch.bool, device=Gk.device))
            # (batch, head) slices holding a recovered entry: only they enter the factor-route tiles.
            slice_rec = torch.zeros(slice_shape, dtype=torch.bool, device=Gq.device)       # (..., H)
            for rec in (rec_q, rec_k):
                if rec is not None:
                    slice_rec = slice_rec | rec.any(dim=-2)
            flags = torch.cat((q_tok, k_tok, slice_rec.reshape(-1))).tolist()
            q_needs, k_needs = flags[:nq_flags], flags[nq_flags:nq_flags + nk_flags]
            recovered_slices = [s for s, flag in enumerate(flags[nq_flags + nk_flags:]) if flag]
        else:
            # Healthy path: chol_k is the prepared factor, so this is its precision bit for bit.
            q_cov_healthy, k_prec_healthy = Gq, k_prec_prepared
            q_needs = k_needs = None
        aq, ak = aq.movedim(-2, -3), ak.movedim(-2, -3)  # (..., H, N, r)
        q_cov_healthy = q_cov_healthy.movedim(-3, -4)   # (..., H, Nq, r, r)
        k_prec_healthy = k_prec_healthy.movedim(-3, -4)  # (..., H, Nk, r, r)
        if factor_trace_any:
            q_factor = q_factor.movedim(-3, -4)           # (..., H, Nq, r, r)
            k_factor = k_factor.movedim(-3, -4)
        logdet_q = _factor_logdet(chol_q, q_status[0], q_logdet_prepared).transpose(-1, -2)
        logdet_k = _factor_logdet(chol_k, k_status[0], k_logdet_prepared).transpose(-1, -2)
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

        def _tile_steps(element_size: int, per_pair: int, slices: int) -> Tuple[int, int]:
            pair_budget = max(1, _RECOVERY_PAIR_TILE_BYTES // (element_size * slices * per_pair))
            q_step = max(1, min(nq, 32, pair_budget))
            return q_step, max(1, min(nk, pair_budget // q_step))

        def _tile(args: tuple, ft: bool, q_step: int, k_step: int) -> torch.Tensor:
            if torch.is_grad_enabled() and (nq > q_step or nk > k_step) and any(x.requires_grad for x in args):
                return checkpoint(_projected_kl_tile, *args, ft,
                                  use_reentrant=False, preserve_rng_state=False)
            return _projected_kl_tile(*args, ft)

        # Working-dtype grid by one centered GEMM per (batch, head): exactly the no-recovery computation.
        # Only its token-linear inputs are saved; the expanded operands are rebuilt in backward.
        healthy = _projected_kl_gemm(aq_healthy, ak_healthy, q_cov_healthy, k_prec_healthy,
                                     logdet_q_healthy, logdet_k_healthy)
        if fallback_slices:
            # Slices whose GEMM error estimate exceeds gemm_expansion_tol take the direct tiles.
            global _GEMM_FALLBACK_SLICES
            _GEMM_FALLBACK_SLICES += len(fallback_slices)
            index = torch.tensor(fallback_slices, dtype=torch.long, device=healthy.device)
            grid = healthy.shape                                                # (..., H, Nq, Nk)

            def _select(t: torch.Tensor, tail: int) -> torch.Tensor:          # (n_fallback, *tail)
                trailing = t.shape[t.ndim - tail:]
                return t.expand(*grid[:-2], *trailing).reshape(-1, *trailing).index_select(0, index)

            selected = (_select(aq_healthy, 2), _select(ak_healthy, 2), _select(q_cov_healthy, 3),
                        _select(k_prec_healthy, 3), _select(logdet_q_healthy, 1), _select(logdet_k_healthy, 1))
            # A direct tile holds three (n, I, J, r) pair tensors (delta, P delta, their product).
            per_pair = aq_healthy.element_size() * len(fallback_slices) * 3 * r
            pair_budget = max(1, _PROJECTION_TILE_BYTES // per_pair)
            q_step = max(1, min(nq, 32, pair_budget))
            k_step = max(1, min(nk, pair_budget // q_step))
            checkpointed = nq > q_step or nk > k_step or nq * nk * per_pair > _CHECKPOINT_BYTES
            rows = []
            for i in range(0, nq, q_step):
                columns = []
                for j in range(0, nk, k_step):
                    tile_args = (selected[0][:, i:i+q_step], selected[1][:, j:j+k_step],
                                 selected[2][:, i:i+q_step], selected[3][:, j:j+k_step],
                                 selected[4][:, i:i+q_step], selected[5][:, j:j+k_step])
                    if (torch.is_grad_enabled() and checkpointed
                            and any(x.requires_grad for x in tile_args)):
                        columns.append(checkpoint(_projected_kl_direct_tile, *tile_args,
                                                  use_reentrant=False, preserve_rng_state=False))
                    else:
                        columns.append(_projected_kl_direct_tile(*tile_args))
                rows.append(torch.cat(columns, dim=-1))
            direct = torch.cat(rows, dim=-2)                                    # (n_fallback, Nq, Nk)
            healthy = (healthy.reshape(-1, nq, nk)
                       .index_copy(0, index, direct.to(healthy.dtype)).reshape(grid))
        if not factor_trace_any or not recovered_slices:
            return healthy.to(dtype)
        # Factor route only for pairs that touch a recovered entry, selected per (b, h, i, j). Only the (batch, head)
        # slices holding one are gathered, and the tile size follows their count (audit 2026-10-16 F28); every
        # triangular solve is per matrix, so a recovered slice scores as in a batch that holds it alone.
        q_rec = (rec_q.movedim(-1, -2) if rec_q is not None                       # (..., H, Nq)
                 else torch.zeros(Gq.shape[:-4] + (Gq.shape[-3], Gq.shape[-4]),
                                  dtype=torch.bool, device=Gq.device))
        k_rec = (rec_k.movedim(-1, -2) if rec_k is not None                       # (..., H, Nk)
                 else torch.zeros(Gk.shape[:-4] + (Gk.shape[-3], Gk.shape[-4]),
                                  dtype=torch.bool, device=Gk.device))
        grid = healthy.shape                                                      # (..., H, Nq, Nk)
        index = torch.unravel_index(torch.tensor(recovered_slices, dtype=torch.long, device=healthy.device),
                                    grid[:-2])

        def _gather(t: torch.Tensor, tail: int) -> torch.Tensor:                 # (n_recovered, *tail)
            trailing = t.shape[t.ndim - tail:]
            return t.expand(*grid[:-2], *trailing)[index]

        pair_rec = _gather(q_rec, 1).unsqueeze(-1) | _gather(k_rec, 1).unsqueeze(-2)   # (n, Nq, Nk)
        route = (_gather(aq, 2), _gather(ak, 2), _gather(q_factor, 3), _gather(k_factor, 3),
                 _gather(logdet_q, 1), _gather(logdet_k, 1))
        current = _gather(healthy, 2)                                             # (n, Nq, Nk)
        q_step, k_step = _tile_steps(aq.element_size(), r * r, len(recovered_slices))
        rows = []
        for i in range(0, nq, q_step):
            columns = []
            i_rec = any(q_needs[i:i+q_step])
            for j in range(0, nk, k_step):
                value = current[:, i:i+q_step, j:j+k_step]
                if i_rec or any(k_needs[j:j+k_step]):
                    args = (route[0][:, i:i+q_step], route[1][:, j:j+k_step],
                            route[2][:, i:i+q_step], route[3][:, j:j+k_step],
                            route[4][:, i:i+q_step], route[5][:, j:j+k_step])
                    factor_value = _tile(args, True, q_step, k_step).to(dtype)
                    value = torch.where(pair_rec[:, i:i+q_step, j:j+k_step], factor_value, value)
                columns.append(value)
            rows.append(torch.cat(columns, dim=-1))
        return healthy.index_put(index, torch.cat(rows, dim=-2).to(healthy.dtype)).to(dtype)


def _healthy_kl_verdict(host: List[bool]) -> None:
    r"""The raises of :func:`projected_gaussian_kl`'s decision read when no recovery can act, in its order:
    the means, then the query role's and the key role's finiteness, symmetry and factorization."""
    if not host[0]:
        raise ValueError("projected means must contain only finite values")
    for finite, symmetric, _, failed_any in (host[1:5], host[5:9]):
        if not finite:
            raise ValueError("projected covariance must be finite for zero-jitter Cholesky")
        if not symmetric:
            raise ValueError("projected covariance must be symmetric for zero-jitter Cholesky")
        if failed_any:
            raise ValueError("projected covariance is not positive definite: zero-jitter Cholesky failed; "
                             "no repair applied")


def _projected_kl_gemm(
    aq:       torch.Tensor,  # (..., H, Nq, r) query means
    ak:       torch.Tensor,  # (..., H, Nk, r) key means
    q_cov:    torch.Tensor,  # (..., H, Nq, r, r) query covariances G_i
    k_prec:   torch.Tensor,  # (..., H, Nk, r, r) key precisions P_j = G_j^{-1}
    logdet_q: torch.Tensor,  # (..., H, Nq)
    logdet_k: torch.Tensor,  # (..., H, Nk)
) -> torch.Tensor:           # (..., H, Nq, Nk)
    r"""Healthy pair KL grid as one centered expanded GEMM per (batch, head), forward and backward at IEEE fp32.

    The value is :func:`_projected_kl_expansion`. Under autograd it runs through :class:`_IEEEFP32Recompute`,
    which saves only these token-linear inputs and differentiates a recomputation of the expansion under
    :func:`vfe3.runtime.ieee_fp32_matmul`, so the backward GEMMs, whose terms cancel to
    :math:`\sum_i g_{ij}(G_i + \delta_{ij}\delta_{ij}^\top)` in :math:`\partial L/\partial P_j`, no longer run under
    the caller's TF32 policy (audit 2026-10-10 round 2, item 2).
    """
    return _ieee_fp32_autograd(_projected_kl_expansion, aq, ak, q_cov, k_prec, logdet_q, logdet_k)


@ieee_fp32_matmul()
def _projected_kl_expansion(
    aq:       torch.Tensor,  # (..., H, Nq, r) query means
    ak:       torch.Tensor,  # (..., H, Nk, r) key means
    q_cov:    torch.Tensor,  # (..., H, Nq, r, r) query covariances G_i
    k_prec:   torch.Tensor,  # (..., H, Nk, r, r) key precisions P_j = G_j^{-1}
    logdet_q: torch.Tensor,  # (..., H, Nq)
    logdet_k: torch.Tensor,  # (..., H, Nk)
) -> torch.Tensor:           # (..., H, Nq, Nk)
    r"""Healthy pair KL grid as one centered expanded GEMM per (batch, head), at IEEE fp32.

    With a common shift :math:`c`, :math:`\alpha_i = a_i - c` and :math:`b_j = a_{k,j} - c`,

    .. math::
        2\,\mathrm{KL}_{ij} = \langle G_i + \alpha_i\alpha_i^\top, P_j\rangle_F
        - \alpha_i^\top (P_j + P_j^\top) b_j + b_j^\top P_j b_j + \log\det G_j - r - \log\det G_i,

    which is :math:`\operatorname{tr}(P_jG_i) + \delta_{ij}^\top P_j\delta_{ij} - r + \log\det(G_j/G_i)`
    for every :math:`c` (the :math:`(P_j + P_j^\top)` form does not assume a bitwise-symmetric precision).
    The first two terms are one ``(Nq, r^2 + r) @ (r^2 + r, Nk)`` product. The shift is the first key
    mean, detached (the value is independent of :math:`c`, so its gradient is zero), which keeps
    :math:`\alpha` and :math:`b` at the scale of the means' spread instead of their offset and bounds
    the expansion's cancellation; a first-key shift also keeps every row independent of later keys.
    The forward GEMMs run at IEEE fp32 (``vfe3.runtime.ieee_fp32_matmul``); TF32 would round the
    operands to :math:`2^{-10}`. :func:`_projected_kl_gemm` runs their backward at IEEE fp32 too.
    """
    r = aq.shape[-1]
    c = ak[..., :1, :].detach()                                                  # (..., H, 1, r)
    alpha, b = aq - c, ak - c                                                    # (..., H, N, r)
    Pb = (k_prec @ b.unsqueeze(-1)).squeeze(-1)                                  # (..., H, Nk, r)
    PTb = (k_prec.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)               # (..., H, Nk, r)
    second = q_cov + alpha.unsqueeze(-1) * alpha.unsqueeze(-2)                   # (..., H, Nq, r, r)
    query = torch.cat((second.flatten(-2), alpha), dim=-1)                       # (..., H, Nq, r^2 + r)
    key = torch.cat((k_prec.flatten(-2), -(Pb + PTb)), dim=-1)                   # (..., H, Nk, r^2 + r)
    cross = query @ key.transpose(-1, -2)                                        # (..., H, Nq, Nk)
    key_constant = (b * Pb).sum(-1) + logdet_k                                   # (..., H, Nk)
    return .5 * (cross + key_constant.unsqueeze(-2) - r - logdet_q.unsqueeze(-1))


def _projected_diagonal_kl(
    a_q: torch.Tensor,             # (..., Nq, H, r)
    g_q: torch.Tensor,             # (..., Nq, H, r) diagonal query covariances
    a_k: torch.Tensor,             # (..., Nk, H, r)
    g_k: torch.Tensor,             # (..., Nk, H, r) diagonal key covariances
    query_row_constant_grad: bool,
) -> torch.Tensor:                 # (..., H, Nq, Nk)
    r"""Diagonal route of :func:`projected_gaussian_kl` for structurally diagonal comparison covariances.

    With :math:`G_i = \operatorname{diag}(g_i)` and :math:`G_j = \operatorname{diag}(g_j)`,

    .. math::
        2\,\mathrm{KL}_{ij} = \sum_k \Bigl[\frac{g_{ik}}{g_{jk}} + \frac{(a_{ik} - b_{jk})^2}{g_{jk}}
            + \log g_{jk} - \log g_{ik} - 1\Bigr],

    which is the dense Gaussian KL of the diagonal matrices exactly. No Gram, Cholesky factorization or
    :math:`r^2 + r` GEMM runs: the pair grid is one GEMM of inner dimension :math:`2r` per (batch, head)
    (:func:`_diagonal_kl_expansion`), forward and backward at IEEE fp32 through :class:`_IEEEFP32Recompute`, and
    no (..., H, Nq, Nk, r) tensor is formed. This is the route of ``rope_mode='comparison_invariant'``
    whenever every comparison coordinate but at most one rotates (audit 2026-10-10 round 2, item 30).

    The diagonal analogue of the zero-jitter positivity certificate is that every variance is finite and positive;
    a diagonal covariance needs no factorization, so no recovery has anything to rebuild and the verdict can only
    raise. It goes through :func:`vfe3.contracts.host_check`: inside :func:`vfe3.contracts.deferred_host_checks` it
    joins the next mandatory host read, and outside it raises at once. Under ``query_row_constant_grad=False`` the
    query log-determinant, a row constant, carries no gradient, as on the dense route; the query variances still
    reach the trace term with their gradient.
    """
    dtype = _compute_dtype(a_q, g_q, a_k, g_k)
    with torch.autocast(device_type=a_q.device.type, enabled=False):
        aq, ak = a_q.to(dtype).movedim(-2, -3), a_k.to(dtype).movedim(-2, -3)   # (..., H, N, r)
        gq, gk = g_q.to(dtype).movedim(-2, -3), g_k.to(dtype).movedim(-2, -3)   # (..., H, N, r)
        with torch.no_grad():
            flags = torch.stack((torch.isfinite(aq).all() & torch.isfinite(ak).all(),
                                 torch.isfinite(gq).all(), (gq > 0).all(),
                                 torch.isfinite(gk).all(), (gk > 0).all()))
        host_check(flags, _diagonal_kl_verdict)
        with torch.set_grad_enabled(torch.is_grad_enabled() and query_row_constant_grad):
            logdet_q = gq.log().sum(-1)                                          # (..., H, Nq)
        return _ieee_fp32_autograd(_diagonal_kl_expansion, aq, ak, gq, gk, logdet_q)


def _diagonal_kl_verdict(host: List[bool]) -> None:
    r"""The raises of the diagonal route, in the dense route's order: means, query role, key role."""
    means, q_finite, q_positive, k_finite, k_positive = host
    if not means:
        raise ValueError("projected means must contain only finite values")
    for finite, positive in ((q_finite, q_positive), (k_finite, k_positive)):
        if not finite:
            raise ValueError("projected covariance must be finite for the diagonal route")
        if not positive:
            raise ValueError("projected covariance is not positive definite: a diagonal comparison variance is "
                             "not positive; no repair applied")


@ieee_fp32_matmul()
def _diagonal_kl_expansion(
    aq:       torch.Tensor,  # (..., H, Nq, r) query means
    ak:       torch.Tensor,  # (..., H, Nk, r) key means
    gq:       torch.Tensor,  # (..., H, Nq, r) query variances g_i
    gk:       torch.Tensor,  # (..., H, Nk, r) key variances g_j
    logdet_q: torch.Tensor,  # (..., H, Nq) sum_k log g_ik
) -> torch.Tensor:           # (..., H, Nq, Nk)
    r"""Diagonal pair KL grid as one centered GEMM of inner dimension 2r per (batch, head), at IEEE fp32.

    With the detached first-key shift :math:`c` of :func:`_projected_kl_expansion`, :math:`\alpha_i = a_i - c` and
    :math:`b_j = a_{k,j} - c`,

    .. math::
        2\,\mathrm{KL}_{ij} = \bigl\langle [g_i + \alpha_i\circ\alpha_i,\ \alpha_i],\
            [g_j^{-1},\ -2\,b_j\circ g_j^{-1}]\bigr\rangle
            + \sum_k \Bigl(\frac{b_{jk}^2}{g_{jk}} + \log g_{jk}\Bigr) - r - \sum_k \log g_{ik},

    the diagonal case of the dense expansion (:math:`P_j = \operatorname{diag}(g_j^{-1})`).
    """
    r = aq.shape[-1]
    c = ak[..., :1, :].detach()                                                  # (..., H, 1, r)
    alpha, b = aq - c, ak - c                                                    # (..., H, N, r)
    inverse = gk.reciprocal()                                                    # (..., H, Nk, r)
    scaled = b * inverse                                                         # (..., H, Nk, r) b_j / g_j
    query = torch.cat((gq + alpha * alpha, alpha), dim=-1)                       # (..., H, Nq, 2r)
    key = torch.cat((inverse, -2 * scaled), dim=-1)                              # (..., H, Nk, 2r)
    cross = query @ key.transpose(-1, -2)                                        # (..., H, Nq, Nk)
    key_constant = (b * scaled).sum(-1) + gk.log().sum(-1)                       # (..., H, Nk)
    return .5 * (cross + key_constant.unsqueeze(-2) - r - logdet_q.unsqueeze(-1))


@torch.no_grad()
def _gemm_expansion_error(
    aq:       torch.Tensor,  # (..., Nq, H, r) query means
    ak:       torch.Tensor,  # (..., Nk, H, r) key means
    k_prec:   torch.Tensor,  # (..., Nk, H, r, r) key precisions P_j
    excluded: torch.Tensor,  # (..., Nk, H) key entries the recovery rebuilds (their pairs leave the GEMM)
) -> torch.Tensor:           # (..., H) estimated max pair error of the GEMM, in nats
    r"""A-priori estimate of the centered GEMM's excess forward error, maximized per (batch, head).

    With :math:`\alpha_i = a_i - c`, :math:`b_j = a_{k,j} - c` and :math:`c` the first key mean, the
    GEMM forms the mean part of :math:`2\,\mathrm{KL}_{ij}` as the expansion
    :math:`\alpha_i^\top P_j\alpha_i - \alpha_i^\top (P_j + P_j^\top) b_j + b_j^\top P_j b_j` of
    :math:`\delta_{ij}^\top P_j\delta_{ij}`. Standard dot-product analysis bounds its rounding error by
    :math:`\gamma\,(|\alpha_i| + |b_j|)^\top |P_j| (|\alpha_i| + |b_j|)`, while the direct tile, which
    forms :math:`\delta_{ij}` first, pays :math:`\gamma\,|\delta_{ij}|^\top|P_j||\delta_{ij}|`; the trace
    term :math:`\langle G_i, P_j\rangle_F` carries the same error on both routes and is excluded. With
    :math:`\rho_j = \max(\lVert P_j\rVert_1, \lVert P_j\rVert_\infty) \ge \lVert |P_j| \rVert_2`,
    :math:`A = \max_i \lVert\alpha_i\rVert_2` and :math:`B_j = \lVert b_j\rVert_2`, the slice maximum is

    .. math::
        E = \tfrac12\,\gamma\,\max_j \rho_j (A + B_j)^2,
        \qquad \gamma = \sqrt{r}\,u,

    exact as a maximum of the majorant over pairs since it increases in :math:`A`, and computed in
    O(N r^2) per slice with no pair grid. The worst-case constant would be
    :math:`\gamma_{r^2+r} \approx (r^2 + r) u`; :math:`\sqrt{r}\,u` is the probabilistic scale of the
    accumulated rounding (Higham and Mary, SIAM J. Sci. Comput. 41 (2019) A2815), which on the
    audit sweep (r = 8 and 32; spread 1 to 4000; cond 1 and 1e3; an outlying key 0) majorized every
    near-minimum pair error by a factor of at least three. It is an estimate, not a certificate.
    Key entries the recovery rebuilds are excluded, since their pairs take the factor route; a healthy
    slice's verdict therefore does not depend on an ill-conditioned token elsewhere in it.
    """
    r = aq.shape[-1]
    u = torch.finfo(aq.dtype).eps / 2
    c = ak[..., :1, :, :]
    A = (aq - c).norm(dim=-1).amax(dim=-2)                                       # (..., H)
    B = (ak - c).norm(dim=-1)                                                    # (..., Nk, H)
    absolute = k_prec.abs()
    rho = torch.maximum(absolute.sum(-1).amax(-1), absolute.sum(-2).amax(-1))   # (..., Nk, H)
    rho = rho.masked_fill(excluded, 0.)
    return .5 * math.sqrt(r) * u * (rho * (A.unsqueeze(-2) + B).square()).amax(dim=-2)


@ieee_fp32_matmul()
def _projected_kl_direct_tile(
    aq:       torch.Tensor,  # (n, I, r)
    ak:       torch.Tensor,  # (n, J, r)
    q_cov:    torch.Tensor,  # (n, I, r, r)
    k_prec:   torch.Tensor,  # (n, J, r, r)
    logdet_q: torch.Tensor,  # (n, I)
    logdet_k: torch.Tensor,  # (n, J)
) -> torch.Tensor:           # (n, I, J)
    r"""Direct-difference KL tile for GEMM fallback slices, pinned to IEEE fp32 also on checkpoint recompute."""
    return _projected_kl_tile(aq, ak, q_cov, k_prec, logdet_q, logdet_k, False)


@ieee_fp32_matmul()
def _projected_kl_tile(
    aq:       torch.Tensor,  # (..., H, I, r)
    ak:       torch.Tensor,  # (..., H, J, r)
    q_matrix: torch.Tensor,  # covariance or lower factor (..., H, I, r, r)
    k_matrix: torch.Tensor,  # precision or lower factor (..., H, J, r, r)
    logdet_q: torch.Tensor,
    logdet_k: torch.Tensor,
    factor_trace: bool,
) -> torch.Tensor:
    r"""Compute one KL tile; tr(Gk^-1 Gq) = ||Lk^-1 Lq||_F^2 on the factor route.

    Pinned to IEEE fp32 itself, so the factor-route tiles and their checkpointed recompute run under the
    same matmul policy as the other scorer tiles (audit 2026-10-10 #45).
    """
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
    symmetric_inputs: bool = False,
    commutant_pairs: Optional[int] = None,
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
    source recovery for flagged entries, bounded checkpointed pair tiles, and the float64
    route selected per pair, so healthy pairs are bit-identical to a call without recovery. The
    recovered factors come from float32 shifted CholeskyQR3 (float64 QR on its fallback), and these
    pair tiles run in float64.
    Recovered entries enter the working-dtype grid as identity placeholders, so no factorization
    of a numerically non-SPD sum runs even for a pair whose value the factor route replaces.
    ``symmetric_inputs`` is the bitwise-symmetry certificate of :func:`projected_gaussian_kl`, and
    ``commutant_pairs`` its declaration that the covariances are RoPE-commutant projections, which recovery
    rebuilds as such.
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
        q_status, k_status = _cholesky_status(Gq, symmetric_inputs), _cholesky_status(Gk, symmetric_inputs)
        # One host transfer for both roles and the means on the healthy path.
        # host_read also resolves raise-only checks deferred by vfe3.contracts.deferred_host_checks.
        decisions = host_read(torch.cat(((torch.isfinite(aq).all() & torch.isfinite(ak).all()).reshape(1),
                                         q_status[-1], k_status[-1])))
        if not decisions[0]:
            raise ValueError("projected means must contain only finite values")
        chol_q, rec_q = _strict_cholesky(Gq, projection_q, (*q_status[:3], decisions[1:5]), False, commutant_pairs)
        chol_k, rec_k = _strict_cholesky(Gk, projection_k, (*k_status[:3], decisions[5:9]), False, commutant_pairs)
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


def projected_noisy_key_cost(
    a_q: torch.Tensor,        # (..., Nq, H, r)
    G_q: torch.Tensor,        # (..., Nq, H, r, r)
    a_k: torch.Tensor,        # (..., Nk, H, r)
    G_k: torch.Tensor,        # (..., Nk, H, r, r)

    *,
    key_noise: torch.Tensor,  # (H,) n_h > 0
    projection_q: Optional[Tuple[torch.Tensor, ...]] = None,
    projection_k: Optional[Tuple[torch.Tensor, ...]] = None,
    symmetric_inputs: bool = False,
    commutant_pairs: Optional[int] = None,
    **kl_options: Any,
) -> torch.Tensor:            # (..., H, Nq, Nk)
    r"""Noisy-key routing cost :math:`D_{ij} = \mathrm{KL}(N(a_i, G_{q,i})\,\|\,N(b_j, G_{k,j} + n_hI_r))`.

    Each key law is broadened by a per-head isotropic noise :math:`n_h` (audit 2026-10-12, T6). The router applies
    it at temperature 1 (:meth:`vfe3.model.relational_operators.RelationalOperators.routing_temperature`), so the
    routing objective :math:`\sum_j\beta_{ij}D_{ij} + \mathrm{KL}(\beta_i\,\|\,\pi_i)` is
    :math:`\mathrm{KL}(\mathbb Q_{i,\beta}\,\|\,\mathbb P_i)` of the declared joint
    :math:`\mathbb P_i(y, J=j) = \pi_{ij}N(y; b_j, G_{k,j} + n_hI)` and the factorized recognition law
    :math:`\mathbb Q_{i,\beta}(y, J=j) = \beta_{ij}N(y; a_i, G_{q,i})` (GL(K) v2, Eq. (mixture_free_energy)) with
    no temperature :math:`\kappa\sqrt d`: the noise sets the sharpness. At :math:`n_h = 0` it is
    :func:`projected_gaussian_kl` at temperature 1.

    The cost is :func:`projected_gaussian_kl` of the broadened key covariance. The key source tuple is not passed:
    :math:`G_{k,j} + n_hI` is not the Gram matrix of a source root, so no key recovery or factor route applies (the
    Gram route runs, and :math:`G_{k,j} + n_hI \succeq n_hI` needs none). The query keeps its source tuple and
    recovery. ``kl_options`` (``query_row_constant_grad``, ``gemm_expansion_tol``) pass through. ``key_noise`` is
    required, so a caller that does not supply it fails instead of scoring the noise-free Gaussian KL.
    """
    r = a_k.shape[-1]
    if G_k.shape != (*a_k.shape, r):
        raise ValueError("projected covariances must match their means with one additional matrix axis")
    if key_noise.shape != (a_k.shape[-2],):
        raise ValueError(f"key_noise must have one entry per head, shape ({a_k.shape[-2]},), "
                         f"got {tuple(key_noise.shape)}")
    dtype = _compute_dtype(a_q, G_q, a_k, G_k, key_noise)
    eye = torch.eye(r, dtype=dtype, device=G_k.device)
    # G_k + n_h I is exactly symmetric when G_k is: the shift touches only the diagonal.
    G_noisy = G_k.to(dtype) + key_noise.to(dtype)[:, None, None] * eye           # (..., Nk, H, r, r)
    return projected_gaussian_kl(a_q, G_q, a_k, G_noisy, projection_q=projection_q, projection_k=None,
                                 symmetric_inputs=symmetric_inputs, commutant_pairs=commutant_pairs, **kl_options)


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


def projected_mean_distance_cost(
    a_q: torch.Tensor,  # (..., Nq, H, r)
    G_q: torch.Tensor,  # (..., Nq, H, r, r)
    a_k: torch.Tensor,  # (..., Nk, H, r)
    G_k: torch.Tensor,  # (..., Nk, H, r, r)

    *,
    projection_q: Optional[Tuple[torch.Tensor, ...]] = None,
    projection_k: Optional[Tuple[torch.Tensor, ...]] = None,
    symmetric_inputs: bool = False,
    commutant_pairs: Optional[int] = None,
) -> torch.Tensor:      # (..., H, Nq, Nk)
    r"""Covariance-blind control cost D_ij = ||a_i - b_j||^2 / 2 of the projected means.

    This is the Gaussian KL of Eq. (projected_full_kl) with both projected covariances replaced by
    I_r, equivalently dot-product attention a_i^T b_j with the key-norm bias -||b_j||^2/2 (the row
    constant ||a_i||^2/2 drops out of the softmax). It is the mean-only retraining control that
    isolates covariance-aware routing: the projected covariances are accepted for signature parity
    with projected_gaussian_kl (as are ``symmetric_inputs`` and ``commutant_pairs``) and are not read, so
    they receive no gradient through routing.
    D_ij >= 0, so the [0, kl_max] cost bound applies unchanged.

    The direct differences are kept (no |a|^2 + |b|^2 - 2 a.b expansion, which cancels for nearby
    means); query rows are tiled to the projection byte budget and checkpointed under autograd, so
    the (..., H, Nq, Nk, r) difference grid is never materialized or saved whole.
    """
    if a_q.ndim < 3 or a_k.ndim < 3 or a_q.shape[-2:] != a_k.shape[-2:]:
        raise ValueError("projected means must have matching head and comparison dimensions")
    dtype = _compute_dtype(a_q, a_k)
    with torch.autocast(device_type=a_q.device.type, enabled=False):
        aq = a_q.to(dtype).movedim(-2, -3)                                     # (..., H, Nq, r)
        ak = a_k.to(dtype).movedim(-2, -3)                                     # (..., H, Nk, r)
        nq = aq.shape[-2]
        per_row = ak.numel() * ak.element_size()                              # one query row's (..., H, Nk, r)
        step = max(1, min(nq, _PROJECTION_TILE_BYTES // max(1, per_row)))
        checkpointed = step < nq or nq * per_row > _CHECKPOINT_BYTES
        rows = []
        for i in range(0, nq, step):
            tile = aq[..., i:i + step, :]
            if torch.is_grad_enabled() and checkpointed and (tile.requires_grad or ak.requires_grad):
                rows.append(checkpoint(_mean_distance_tile, tile, ak, use_reentrant=False, preserve_rng_state=False))
            else:
                rows.append(_mean_distance_tile(tile, ak))
        return rows[0] if len(rows) == 1 else torch.cat(rows, dim=-2)          # (..., H, Nq, Nk)


def _mean_distance_tile(
    aq: torch.Tensor,  # (..., H, I, r)
    ak: torch.Tensor,  # (..., H, Nk, r)
) -> torch.Tensor:     # (..., H, I, Nk)
    r"""One query-row tile of :func:`projected_mean_distance_cost`, :math:`\|a_i-b_j\|^2/2`."""
    delta = aq.unsqueeze(-2) - ak.unsqueeze(-3)                                # (..., H, I, Nk, r)
    return 0.5 * delta.pow(2).sum(-1)


# Routing cost per relational_routing value; every cost takes (a_q, G_q, a_k, G_k, *, projection_q,
# projection_k, symmetric_inputs, commutant_pairs) and returns (..., H, Nq, Nk) nonnegative costs.
ROUTING_COSTS = {
    "gaussian_kl":   projected_gaussian_kl,
    "noisy_query":   projected_noisy_query_cost,
    "noisy_key":     projected_noisy_key_cost,       # also requires key_noise (H,): the router passes it
    "mean_distance": projected_mean_distance_cost,
}


def routing_cost(routing: str):
    r"""The projected routing-cost function selected by ``routing`` (a key of :data:`ROUTING_COSTS`)."""
    try:
        return ROUTING_COSTS[routing]
    except KeyError:
        raise ValueError(f"relational routing must be one of {tuple(ROUTING_COSTS)}, got {routing!r}") from None
