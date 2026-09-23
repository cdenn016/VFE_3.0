"""Lie-bracket closure: iteratively extend a generator basis until it is closed under [.,.]."""
import logging
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _sign_pin_and_order(
    U_cols:         torch.Tensor,    # (K2, k_new) kept left-singular vectors (columns)
    S_kept:         torch.Tensor,    # (k_new,) corresponding singular values (descending)

    degenerate_tol: float,
) -> torch.Tensor:
    r"""
    Canonicalize SVD columns: sign-pin each, then lex-sort degenerate groups.

    Sign rule: flip so the largest-:math:`|\cdot|` entry is positive; ties on
    equal :math:`|\cdot|` resolved by the lowest flattened index. Order rule:
    within each maximal run of singular values equal to within ``degenerate_tol``
    (relative to the leading singular value), sort the (already sign-pinned)
    columns lexicographically by their flattened entries (ascending).

    Returns ``(k_new, K2)`` canonicalized directions as ROWS.
    """
    dirs = U_cols.transpose(-1, -2).contiguous()    # (k_new, K2) rows = directions
    k_new = dirs.shape[0]
    if k_new == 0:
        return dirs

    # 1. SIGN PINNING --------------------------------------------------------
    # argmax over |value|; torch.argmax returns the FIRST max on ties => the
    # lowest flattened index, matching the documented tie-break.
    abs_dirs = dirs.abs()                            # (k_new, K2)
    pivot_idx = torch.argmax(abs_dirs, dim=-1)       # (k_new,)
    pivot_val = dirs.gather(-1, pivot_idx.unsqueeze(-1)).squeeze(-1)  # (k_new,)
    signs = torch.where(pivot_val < 0.0, -1.0, 1.0).to(dirs.dtype)    # +1 if pivot==0
    dirs = dirs * signs.unsqueeze(-1)

    # 2. DEGENERATE-GROUP LEX ORDER -----------------------------------------
    # Group contiguous singular values equal within an absolute threshold scaled
    # by the leading singular value. S_kept is descending, so groups are runs.
    s0 = float(S_kept[0]) if S_kept.numel() > 0 else 1.0
    thresh = degenerate_tol * (s0 if s0 > 0.0 else 1.0)
    group_ids = torch.zeros(k_new, dtype=torch.long)
    gid = 0
    for i in range(1, k_new):
        if (S_kept[i - 1] - S_kept[i]).abs().item() > thresh:
            gid += 1
        group_ids[i] = gid

    out = dirs.clone()
    start = 0
    for i in range(1, k_new + 1):
        if i == k_new or group_ids[i] != group_ids[start]:
            if i - start > 1:
                grp = dirs[start:i]                  # (g, K2) sign-pinned rows
                order = _lexsort_rows(grp)
                out[start:i] = grp[order]
            start = i
    return out


def _lexsort_rows(rows: torch.Tensor) -> torch.Tensor:
    r"""
    Return indices that sort ``rows`` ((g, K2)) lexicographically (ascending) by
    flattened entries: compare column 0 first, then column 1 on ties, and so on.
    Implemented as a stable sort applied from the LAST key to the FIRST.
    """
    g = rows.shape[0]
    order = torch.arange(g)
    # Stable sort by each column from last to first => composite lexicographic.
    for col in range(rows.shape[1] - 1, -1, -1):
        keys = rows[order, col]
        perm = torch.argsort(keys, stable=True)
        order = order[perm]
    return order


def close_under_brackets(
    generators:     'torch.Tensor',         # (n_gen, K, K) initial basis (array-like accepted)

    max_iter:       int = 10,

    max_dim:        Optional[int] = None,
    tol:            float = 1e-6,
    degenerate_tol: float = 1e-9,
) -> 'Tuple[torch.Tensor, Dict]':
    r"""
    Iteratively extend a generator basis until it is closed under ``[.,.]``.

    Each iteration projects every pairwise commutator out of the current span and
    appends the residual directions (Frobenius norm ``> tol``) as the left-
    singular vectors of the stacked residuals, sign-pinned and degeneracy-ordered
    by :func:`_sign_pin_and_order` for reproducibility (see module docstring).
    The returned basis includes the input generators VERBATIM as its first
    ``n_gen`` rows (orientation and sqrt(2) so(N) norm preserved); appended
    directions are unit-norm. Closure math runs in float64 and casts back to the
    input dtype.

    Args:
        generators:     Initial generators ``(n_gen, K, K)`` (numpy/torch/list).
        max_iter:       Maximum closure iterations.
        max_dim:        Cap on final basis size (defaults to ``K**2``). A warning
                        fires if the walk hits it (sparsity intent defeated).
        tol:            Singular-value / residual threshold for accepting a new
                        direction.
        degenerate_tol: Relative threshold for grouping equal singular values
                        when canonicalizing the appended-vector order.

    Returns:
        closed_generators: ``(n_closed, K, K)`` tensor, with
                           ``closed_generators[:n_gen] == generators``.
        info:              Dict with ``n_iters``, ``n_added``, ``final_dim``,
                           ``initial_dim``, ``converged``, ``hit_max_dim``.
    """
    G_in = torch.as_tensor(generators)
    if G_in.ndim != 3 or G_in.shape[1] != G_in.shape[2]:
        raise ValueError(f"generators must have shape (n_gen, K, K); got {tuple(G_in.shape)}")
    in_dtype = G_in.dtype
    device = G_in.device

    G = G_in.to(torch.float64).clone()                      # float64 internal
    n_gen0, K, _ = G.shape
    if max_dim is None:
        max_dim = K * K

    n_added = 0
    converged = False
    hit_max_dim = False
    iters = 0

    for it in range(max_iter):
        iters = it + 1
        n_cur = G.shape[0]
        G_flat = G.reshape(n_cur, K * K)                    # (n_cur, K²)

        # Orthonormal projector for the current span via thin SVD of G_flat.T.
        # Used only as Q @ Q.T (gauge-invariant) — no sign-pinning needed.
        U_basis, S_basis, _ = torch.linalg.svd(G_flat.transpose(-1, -2), full_matrices=False)
        lead = S_basis[0] if S_basis.numel() > 0 else torch.tensor(1.0, dtype=G.dtype)
        keep_basis = S_basis > tol * lead
        Q_basis = U_basis[:, keep_basis]                    # (K², r)

        # All commutators C_ab = [G_a, G_b], a < b, projected out of the span.
        residuals = []
        for a in range(n_cur):
            for b in range(a + 1, n_cur):
                C = G[a] @ G[b] - G[b] @ G[a]
                C_flat = C.reshape(K * K)
                if float(torch.linalg.norm(C_flat)) < 1e-30:
                    continue
                C_perp = C_flat - Q_basis @ (Q_basis.transpose(-1, -2) @ C_flat)
                if float(torch.linalg.norm(C_perp)) > tol:
                    residuals.append(C_perp)

        if not residuals:
            converged = True
            break

        R = torch.stack(residuals, dim=0)                   # (n_res, K²)
        # Thin SVD on residuals; left singular vectors = orthonormal residual basis.
        U_res, S_res, _ = torch.linalg.svd(R.transpose(-1, -2), full_matrices=False)
        keep_res = S_res > tol * S_res[0]
        U_kept = U_res[:, keep_res]                         # (K², k_new) columns
        S_kept = S_res[keep_res]                            # (k_new,)
        n_new = U_kept.shape[1]
        if n_new == 0:
            converged = True
            break

        # Cap at max_dim.
        room = max_dim - G.shape[0]
        if room <= 0:
            hit_max_dim = True
            logger.warning(
                "close_under_brackets: hit max_dim=%d before convergence; "
                "basis at iteration %d has %d generators with %d residual directions. "
                "Closure walking toward full gl(K) defeats the sparsity intent — "
                "consider redesigning the cross-head coupling pattern.",
                max_dim, iters, G.shape[0], n_new,
            )
            break
        if n_new > room:
            U_kept = U_kept[:, :room]
            S_kept = S_kept[:room]
            hit_max_dim = True

        # Deterministic gauge: sign-pin + lex-order the appended directions.
        new_dirs_flat = _sign_pin_and_order(U_kept, S_kept, degenerate_tol)  # (k_new, K²)
        new_blocks = new_dirs_flat.reshape(-1, K, K)
        G = torch.cat([G, new_blocks], dim=0)
        n_added += new_blocks.shape[0]

    if not converged and not hit_max_dim:
        logger.warning(
            "close_under_brackets: did not converge in %d iterations; "
            "current dim=%d. Increase max_iter or inspect generator structure.",
            max_iter, G.shape[0],
        )

    info = {
        'n_iters': iters,
        'n_added': n_added,
        'final_dim': G.shape[0],
        'initial_dim': n_gen0,
        'converged': converged,
        'hit_max_dim': hit_max_dim,
    }
    return G.to(device=device, dtype=in_dtype), info
