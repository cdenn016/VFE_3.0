r"""Lie-algebra generator construction for VFE_3.0 gauge groups.

Elementary block generators are built directly in the requested dtype. Other bases use
float64 construction before casting when their coefficient arithmetic requires it.
Conventions:
  block GL(d_head) : per-head gl(d_head) embedded in the head's diagonal block, row-major.
  so(N)            : skew L_ij = E_ij - E_ji for i < j.
"""

import torch


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


# Irreps of SO(N) as heads (spin towers etc.) live in vfe3/geometry/irreps.py; the
# 'so_n' group in groups.py assembles blockdiag direct sums of those images.
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




# ---------------------------------------------------------------- block-local bases
#
# A block-local group acts on each head's d-dimensional block by its own element of one local
# Lie subgroup of GL(d). Its basis is a Frobenius-orthonormal basis of the local algebra, copied
# into every head block (generate_block_local). Every local basis below is built in float64 in a
# fixed order, so the assembled generators are deterministic and their Gram is the identity.

def local_basis_so(d: int) -> torch.Tensor:
    r"""Orthonormal so(d): :math:`(E_{ij}-E_{ji})/\sqrt2` for :math:`i<j`. Returns (d(d-1)/2, d, d) float64."""
    if d < 2:
        raise ValueError(f"so(d) needs d >= 2, got d={d}")
    return generate_son(d, dtype=torch.float64) / 2.0 ** 0.5


def local_basis_co(d: int) -> torch.Tensor:
    r"""Orthonormal co(d) = so(d) + R I: the so(d) basis, then :math:`I/\sqrt d`. Returns (d(d-1)/2 + 1, d, d)."""
    if d < 1:
        raise ValueError(f"co(d) needs d >= 1, got d={d}")
    scale = (torch.eye(d, dtype=torch.float64) / d ** 0.5).unsqueeze(0)
    return scale if d == 1 else torch.cat([local_basis_so(d), scale])


def local_basis_sl(d: int) -> torch.Tensor:
    r"""Orthonormal sl(d): off-diagonal :math:`E_{ij}` row-major, then the Helmert traceless diagonals.

    The k-th Helmert element is :math:`\mathrm{diag}(1,\dots,1,-k,0,\dots,0)/\sqrt{k(k+1)}` with k ones,
    k = 1, ..., d - 1. Returns (d^2 - 1, d, d) float64.
    """
    if d < 2:
        raise ValueError(f"sl(d) needs d >= 2, got d={d}")
    basis = []
    for i in range(d):
        for j in range(d):
            if i != j:
                E = torch.zeros(d, d, dtype=torch.float64)
                E[i, j] = 1.0
                basis.append(E)
    for k in range(1, d):
        diagonal = torch.zeros(d, dtype=torch.float64)
        diagonal[:k] = 1.0
        diagonal[k] = -float(k)
        basis.append(torch.diag(diagonal) / (k * (k + 1)) ** 0.5)
    return torch.stack(basis)


def local_basis_borel(d: int) -> torch.Tensor:
    r"""Orthonormal upper-triangular algebra b+(d): :math:`E_{ij}` for :math:`i\le j`, row-major. Returns (d(d+1)/2, d, d)."""
    if d < 1:
        raise ValueError(f"b+(d) needs d >= 1, got d={d}")
    basis = []
    for i in range(d):
        for j in range(i, d):
            E = torch.zeros(d, d, dtype=torch.float64)
            E[i, j] = 1.0
            basis.append(E)
    return torch.stack(basis)


def generate_block_local(
    K:                int,
    n_heads:          int,
    local:            torch.Tensor,                    # (m, d, d) local algebra basis, d = K // n_heads

    *,
    device:           'torch.device | str | None'     = None,
    dtype:            torch.dtype                      = torch.float32,
) -> torch.Tensor:
    r"""Copy a local basis into every head block: generator (h, a) is ``local[a]`` on block h.

    Head-major order (all m local generators of head 0, then head 1, ...), with heads laid out
    contiguously, block h covering coordinates [h d, (h + 1) d). Returns (n_heads * m, K, K).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")
    d = K // n_heads
    if local.ndim != 3 or tuple(local.shape[1:]) != (d, d):
        raise ValueError(f"local basis must have shape (m, {d}, {d}) for K={K}, n_heads={n_heads}; "
                         f"got {tuple(local.shape)}")
    m = local.shape[0]
    G = torch.zeros(n_heads * m, K, K, dtype=torch.float64)
    for h in range(n_heads):
        G[h * m:(h + 1) * m, h * d:(h + 1) * d, h * d:(h + 1) * d] = local.to(torch.float64)
    return G.to(device=device, dtype=dtype)
