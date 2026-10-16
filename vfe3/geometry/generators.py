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




def generate_sun(N: int) -> torch.Tensor:
    r"""su(N) basis :math:`X_a = i\lambda_a` (generalized Gell-Mann). Returns (N^2 - 1, N, N) complex128.

    Order: the N(N-1)/2 antisymmetric :math:`X = E_{jk} - E_{kj}` (j < k, exactly :func:`generate_son`'s
    :math:`L_{jk}`, so phi carries so(N) as its leading slice), then the symmetric :math:`i(E_{jk} + E_{kj})`
    (j < k), then the N - 1 diagonal :math:`i\sqrt{2/(l(l+1))}\,\mathrm{diag}(1, \dots, 1, -l, 0, \dots)`. Every
    element is anti-Hermitian and traceless, :math:`\operatorname{tr}(X_a^\dagger X_b) = 2\delta_{ab}`, and the
    structure constants are real.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2 for SU(N), got N={N}")
    basis = [generate_son(N, dtype=torch.float64).to(torch.complex128)]
    sym = []
    for j in range(N):
        for k in range(j + 1, N):
            X = torch.zeros(N, N, dtype=torch.complex128)
            X[j, k] = X[k, j] = 1j
            sym.append(X)
    diag = []
    for l in range(1, N):
        entries = torch.zeros(N, dtype=torch.complex128)
        entries[:l] = 1.0
        entries[l] = -float(l)
        diag.append(1j * (2.0 / (l * (l + 1))) ** 0.5 * torch.diag(entries))
    return torch.cat([basis[0], torch.stack(sym), torch.stack(diag)])


def realify(X: torch.Tensor) -> torch.Tensor:
    r"""Real form :math:`\begin{pmatrix}A & -B\\ B & A\end{pmatrix}` of :math:`X = A + iB`, (..., n, n) -> (..., 2n, 2n).

    A Lie-algebra homomorphism :math:`\mathfrak{gl}(n, \mathbb{C}) \to \mathfrak{gl}(2n, \mathbb{R})`; an
    anti-Hermitian X maps to a skew matrix.
    """
    A, B = X.real, X.imag
    return torch.cat([torch.cat([A, -B], dim=-1), torch.cat([B, A], dim=-1)], dim=-2)


def generate_sun_real(N: int) -> torch.Tensor:
    r"""The realified su(N) basis :func:`realify` (:func:`generate_sun`), (N^2 - 1, 2N, 2N) float64, skew.

    A faithful real representation, so the real structure-constant solve and the irrep registry run on it
    exactly as on so(N).
    """
    return realify(generate_sun(N)).to(torch.float64)


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


def local_basis_glc(d: int) -> torch.Tensor:
    r"""Realified gl(m, C) on R^{2m}, d = 2m: :math:`E_{ij}\otimes I_2` (row-major), then :math:`E_{ij}\otimes J`.

    Complex coordinate k is the real pair (2k, 2k + 1) and :math:`J = [[0, -1], [1, 0]]` on each pair,
    so :math:`A + iB\in\mathfrak{gl}(m,\mathbb{C})` realifies to
    :math:`\sum_{ij} A_{ij} E_{ij}\otimes I_2 + B_{ij} E_{ij}\otimes J`: the coordinates are exactly
    (Re, Im) of the complex matrix, in (re/im, i, j) order. Entries are 0 and +-1, every generator
    commutes with :math:`I_m\otimes J`, and the Frobenius Gram is :math:`2I`. Returns (2 m^2, d, d) float64.
    """
    if d < 2 or d % 2 != 0:
        raise ValueError(f"gl(m, C) realified needs an even d = 2m >= 2, got d={d}")
    m = d // 2
    eye2 = torch.eye(2, dtype=torch.float64)
    J = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    basis = []
    for unit in (eye2, J):
        for i in range(m):
            for j in range(m):
                E = torch.zeros(m, m, dtype=torch.float64)
                E[i, j] = 1.0
                basis.append(torch.kron(E, unit))
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
