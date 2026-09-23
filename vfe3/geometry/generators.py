r"""Lie-algebra generator construction for VFE_3.0 gauge groups.

Elementary block generators are built directly in the requested dtype. Other bases use
float64 construction before casting when their coefficient arithmetic requires it.
Conventions:
  gl(K)            : full K^2 basis E_ij (1 at (i,j)), row-major.
  block GL(d_head) : per-head gl(d_head) embedded in the head's diagonal block.
  cross-head       : diagonal blocks + off-diagonal E_ij blocks per coupling.
  so(N)            : skew L_ij = E_ij - E_ji for i < j.
"""

import logging
import math
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def _dedup_cross_couplings(
    pairs: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int]:
    r"""Drop exact duplicate directed pairs, preserving first-seen order.

    Directed: (a, b) and (b, a) are distinct. Returns (deduped, n_removed).
    """
    seen:    set = set()
    out:     List[Tuple[int, int]] = []
    removed: int = 0
    for a, b in pairs:
        key = (int(a), int(b))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(key)
    if removed:
        logger.warning(
            "_dedup_cross_couplings dropped %d duplicate pair(s); kept %s",
            removed, out,
        )
    return out, removed


def generate_glk(
    K:                int,

    *,
    include_identity: bool                            = True,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""gl(K) generators (full K^2 basis E_ij), or sl(K) if include_identity=False.

    Always returns ``(K^2, K, K)``. With ``include_identity=False`` the
    normalized identity (trace) direction is projected out of each generator,
    yielding an overcomplete spanning set for sl(K) (K^2 matrices spanning a
    rank K^2-1 space), not a minimal basis.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1 for GL(K), got K={K}")

    n_generators = K * K
    G = torch.zeros(n_generators, K, K, dtype=torch.float64)

    idx = 0
    for i in range(K):
        for j in range(K):
            G[idx, i, j] = 1.0
            idx += 1

    if not include_identity:
        I_K       = torch.eye(K, dtype=torch.float64)
        trace_dir = I_K / math.sqrt(K)
        projected = []
        for g in range(n_generators):
            overlap = torch.sum(G[g] * trace_dir)
            G_proj  = G[g] - overlap * trace_dir
            if torch.linalg.norm(G_proj) > 1e-8:
                projected.append(G_proj)
        G = torch.stack(projected, dim=0)

    return G.to(dtype).to(device)


def generate_glk_multihead(
    K:                int,
    n_heads:          int,

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""Block-diagonal gl(d_head) generators: GL(d_head)^H subset of GL(K).

    d_head = K // n_heads. Returns (n_heads * d_head^2, K, K).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")

    d_head         = K // n_heads
    n_gen_per_head = d_head * d_head
    n_generators   = n_heads * n_gen_per_head

    G = torch.zeros(n_generators, K, K, dtype=dtype)
    for h in range(n_heads):
        start      = h * d_head
        gen_offset = h * n_gen_per_head
        idx        = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    return G.to(device)


def generate_glk_multihead_tied(
    K:                int,
    n_heads:          int,

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""TIED block-diagonal gl(d_head) generators: one shared gl(d_head) acting on every head.

    Each generator is ``kron(I_{n_heads}, E_ij)`` for the d_head x d_head elementary basis E_ij,
    so the SAME d_head x d_head algebra element acts identically in every diagonal block. With one
    per-token gauge coordinate vector phi (n_gen = d_head^2), the group element is
    exp(sum_a phi_a kron(I, E_a)) = kron(I, exp(sum_a phi_a E_a)) -- the SAME GL(d_head) frame in
    every block (a TIED gauge across heads). Contrast ``generate_glk_multihead`` (n_heads * d_head^2
    generators, an INDEPENDENT gl(d_head) per head -- an untied gauge). The tied gauge is the one
    under which the Schur-commutant head mixer kron(A, I_d) is exactly gauge-equivariant.

    Returns (d_head^2, K, K).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")

    d_head       = K // n_heads
    n_generators = d_head * d_head
    eye_n        = torch.eye(n_heads, dtype=torch.float64)

    G = torch.zeros(n_generators, K, K, dtype=torch.float64)
    idx = 0
    for i in range(d_head):
        for j in range(d_head):
            block = torch.zeros(d_head, d_head, dtype=torch.float64)
            block[i, j] = 1.0
            G[idx] = torch.kron(eye_n, block)                # same d_head x d_head element in every block
            idx += 1

    return G.to(dtype).to(device)


def generate_glk_cross_head(
    K:                int,
    n_heads:          int,
    cross_couplings:  List[Tuple[int, int]],

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""Block-diagonal gl(d_head) plus off-diagonal coupling blocks.

    For each directed pair (a, b), d_head^2 elementary matrices map head a's
    rows into head b's columns. Returns
    (n_heads * d_head^2 + len(dedup(cross)) * d_head^2, K, K).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} not divisible by n_heads={n_heads}")

    cross_couplings, _ = _dedup_cross_couplings(list(cross_couplings))

    d_head      = K // n_heads
    n_gen_diag  = n_heads * d_head * d_head
    n_gen_cross = len(cross_couplings) * d_head * d_head
    n_gen_total = n_gen_diag + n_gen_cross

    G = torch.zeros(n_gen_total, K, K, dtype=torch.float64)

    for h in range(n_heads):
        start      = h * d_head
        gen_offset = h * d_head * d_head
        idx        = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, start + i, start + j] = 1.0
                idx += 1

    for pair_idx, (a, b) in enumerate(cross_couplings):
        if a == b:
            raise ValueError(f"Self-coupling ({a},{a}) not allowed")
        if not (0 <= a < n_heads and 0 <= b < n_heads):
            raise ValueError(f"Head indices ({a},{b}) out of range [0, {n_heads})")
        a_start    = a * d_head
        b_start    = b * d_head
        gen_offset = n_gen_diag + pair_idx * d_head * d_head
        idx        = 0
        for i in range(d_head):
            for j in range(d_head):
                G[gen_offset + idx, a_start + i, b_start + j] = 1.0
                idx += 1

    return G.to(dtype).to(device)


def generate_sp(
    K:                int,

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""sp(2m,R) generators (real symplectic Lie algebra). Returns (m(2m+1), 2m, 2m).

    With the standard symplectic form J = [[0, I_m], [-I_m, 0]],
    sp(2m,R) = {A : J A + A^T J = 0}, equivalently A = [[B, C], [D, -B^T]] with B an
    arbitrary m x m block and C = C^T, D = D^T symmetric m x m blocks. A spanning basis
    of dimension m^2 + 2 * m(m+1)/2 = m(2m+1) is emitted as three groups:
      B-block (m^2): [[E_ij, 0], [0, -E_ij^T]] = [[E_ij, 0], [0, -E_ji]], for i, j in [0, m).
      C-block (m(m+1)/2): [[0, S], [0, 0]] with S symmetric -- S = E_ii, and E_ij + E_ji
        for i < j (top-right block).
      D-block (m(m+1)/2): [[0, 0], [S, 0]] with S symmetric (bottom-left block).
    """
    if K < 2 or K % 2 != 0:
        raise ValueError(f"K must be even and >= 2 for Sp(2m,R), got K={K}")

    m            = K // 2
    n_generators = m * (2 * m + 1)
    G            = torch.zeros(n_generators, K, K, dtype=torch.float64)

    idx = 0
    # B-block: A = [[E_ij, 0], [0, -E_ji]]
    for i in range(m):
        for j in range(m):
            G[idx, i, j]         = 1.0
            G[idx, m + j, m + i] = -1.0
            idx += 1
    # C-block: A = [[0, S], [0, 0]], S symmetric (top-right block, columns m..2m-1)
    for i in range(m):
        for j in range(i, m):
            G[idx, i, m + j] = 1.0
            G[idx, j, m + i] = 1.0
            idx += 1
    # D-block: A = [[0, 0], [S, 0]], S symmetric (bottom-left block, rows m..2m-1)
    for i in range(m):
        for j in range(i, m):
            G[idx, m + i, j] = 1.0
            G[idx, m + j, i] = 1.0
            idx += 1

    return G.to(dtype).to(device)

# Irreps of SO(N) as heads (spin towers etc.) live in vfe3/geometry/irreps.py; the
# 'so_n'/'sp_n' groups in groups.py assemble blockdiag direct sums of those images.
def generate_son(
    N:                int,

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""so(N) generators L_ij = E_ij - E_ji for i < j. Returns (N(N-1)/2, N, N)."""
    if N < 2:
        raise ValueError(f"N must be >= 2 for SO(N), got N={N}")

    n_generators = N * (N - 1) // 2
    G = torch.zeros(n_generators, N, N, dtype=torch.float64)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            G[idx, i, j] = 1.0
            G[idx, j, i] = -1.0
            idx += 1

    return G.to(dtype).to(device)


