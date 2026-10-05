r"""Irreducible-representation registry for VFE_3.0 gauge groups (heads = irreps).

Builds the generator images rho(X_a) of a structure algebra g on an irrep block, so a
gauge group can act on the K-dim embedding through a DIRECT SUM of irreps,

    G_a = blockdiag_b rho_b(X_a),        a = 1, ..., n_gen = dim g,

with ONE shared per-token phi -- a TIED gauge: the same group element acts in every
block through that block's representation, so attention heads can be DIFFERENT irreps
of one structure group (possibly with unequal dimensions). Registered family:

    so : label 'l<p>'   -- symmetric-traceless rank-p tensors over R^N,
                           dim = C(N+p-1, p) - C(N+p-3, p-2); for N = 3 this is the
                           spin-p tower (dim 2p + 1). Built in real orthonormal bases,
                           so every generator image is exactly skew.

Two bases are registered for the 'so' family (``basis``):

* ``"canonical"`` (the default) is deterministic. For N = 3 the spin-l block is built in
  closed form from the angular-momentum ladder, J_z |l,m> = m |l,m> and
  J_+ |l,m> = sqrt(l(l+1) - m(m+1)) |l,m+1> (Varshalovich, Moskalev and Khersonskii,
  *Quantum Theory of Angular Momentum*, 1988), carried to the real spherical-harmonic basis
  (Blanco, Florez and Bermejo, J. Mol. Struct. THEOCHEM 419 (1997) 19-27), for any l at
  O(l^2) cost; l = 1 is the defining representation itself. For N != 3 the tensor-power
  construction below runs on a fixed monomial basis of Sym^p with a fixed-order Gram-Schmidt
  for the traceless subspace, so no eigensolver chooses the basis.
* ``"eigh"`` is the original tensor-power construction, whose basis is whatever ``eigh``
  returns inside a degenerate eigenspace. It is kept so checkpoints trained in it still load.

Tensor-power construction: the rank-p tensor power carries the Leibniz action

    rho_p(X) = sum_{k=0}^{p-1} I^{(k)} (x) X (x) I^{(p-1-k)}      on (R^N)^{(x)p},

and the irrep is its restriction rho(X) = Q^T rho_p(X) Q to an orthonormal basis Q of
the invariant subspace (symmetric, and traceless for so). In both bases the homomorphism
property [rho(X), rho(Y)] = rho([X, Y]) is VERIFIED at build time against the structure
constants of the defining basis (raise, not warn, on residual). All construction runs in
float64; callers cast.

A new irrep family (Young-tableau highest-weight reps, spinors, ...) slots in by
``register_irrep`` -- never by editing call sites (the modularity contract).
"""

import itertools
import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch

IRREP_BASES: Tuple[str, ...] = ("canonical", "eigh")
DEFAULT_IRREP_BASIS: str = "canonical"

# key 'algebra:prefix' -> (dim_fn(N, p) -> int, build_fn(G_def, p, basis) -> (n_gen, d, d))
_IRREPS: Dict[str, Tuple[Callable[[int, int], int],
                         Callable[[torch.Tensor, int, str], torch.Tensor]]] = {}


def register_irrep(
    key:      str,                                               # 'algebra:prefix', e.g. 'so:l'
    dim_fn:   Callable[[int, int], int],                         # (N, rank) -> block dimension
    build_fn: Callable[[torch.Tensor, int, str], torch.Tensor],  # (G_def, rank, basis) -> (n_gen, d, d)

    *,
    override: bool = False,
) -> None:
    """Register an irrep family under ``key`` (label grammar: '<prefix><rank>').

    ``build_fn`` receives the basis name (one of ``IRREP_BASES``) and must raise for a basis it
    does not implement. Duplicate keys fail closed (audit 2026-07-01 round-3): a second
    registration under an existing key silently shadowed the first. Pass ``override=True`` to
    replace deliberately.
    """
    if key in _IRREPS and not override:
        raise KeyError(f"irrep family {key!r} already registered; pass override=True to replace")
    _IRREPS[key] = (dim_fn, build_fn)


def _check_basis(basis: str) -> str:
    if basis not in IRREP_BASES:
        raise ValueError(f"irrep basis must be one of {IRREP_BASES}, got {basis!r}")
    return basis


def _parse_label(algebra: str, label: str) -> Tuple[str, int]:
    """Split 'l2' -> ('so:l', 2); raise with the registered keys on an unknown label."""
    prefix = label.rstrip("0123456789")
    digits = label[len(prefix):]
    key = f"{algebra}:{prefix}"
    if not digits or key not in _IRREPS:
        raise ValueError(
            f"unknown irrep label {label!r} for algebra {algebra!r}; registered families: "
            f"{sorted(_IRREPS)} (label grammar '<family><rank>', e.g. 'l2')"
        )
    return key, int(digits)


def irrep_dim(
    N:       int,                          # defining-rep dimension (N of SO(N))

    *,
    algebra: str,                          # 'so'
    label:   str,                          # e.g. 'l2'
) -> int:
    """Block dimension of ``label`` over the rank-N algebra (closed form, no tensors built)."""
    key, p = _parse_label(algebra, label)
    return _IRREPS[key][0](N, p)


def irrep_generators(
    G_def:   torch.Tensor,                 # (n_gen, N, N) defining-rep algebra basis (float64)

    *,
    algebra: str,                          # 'so'
    label:   str,                          # e.g. 'l2'
    basis:   str = DEFAULT_IRREP_BASIS,    # 'canonical' | 'eigh'
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Build one irrep's generator images (the registry's public single-label entry point)."""
    key, p = _parse_label(algebra, label)
    return _IRREPS[key][1](G_def, p, _check_basis(basis))


def check_irrep_constructible(
    N:       int,                          # defining-rep dimension

    *,
    algebra: str,                          # 'so'
    label:   str,                          # e.g. 'l8'
    basis:   str = DEFAULT_IRREP_BASIS,
) -> None:
    """Raise, before any tensor is built, if ``label`` cannot be constructed in ``basis``."""
    _, p = _parse_label(algebra, label)
    if algebra == "so" and N == 3 and _check_basis(basis) == "canonical":
        return                             # closed-form ladder construction: any rank
    if p >= 2:
        _guard_cost(N, p)


def _guard_cost(N: int, p: int) -> None:
    """The tensor-power construction works on N^p x N^p matrices with p! symmetrizer terms."""
    if p > 6 or N ** p > 60000:
        raise ValueError(
            f"rank-{p} tensor-power irrep over R^{N} exceeds the supported construction size "
            f"(rank <= 6 and N^rank <= 60000); for SO(3) use irrep_basis='canonical', which "
            f"builds any spin in closed form; higher ranks for N != 3 await the Young-tableau "
            f"buildout."
        )


def _leibniz_action(
    X: torch.Tensor,                       # (N, N) algebra element in the defining rep
    p: int,                                # tensor-power rank (>= 1)
) -> torch.Tensor:                         # (N^p, N^p) action on the full tensor power
    r"""rho_p(X) = sum_k I^{(k)} (x) X (x) I^{(p-1-k)} on (R^N)^{(x)p}."""
    N = X.shape[-1]
    eye = torch.eye(N, dtype=torch.float64)
    out = torch.zeros(N ** p, N ** p, dtype=torch.float64)
    for k in range(p):
        M = torch.eye(1, dtype=torch.float64)
        for slot in range(p):
            M = torch.kron(M, X if slot == k else eye)
        out += M
    return out


def _symmetrizer(N: int, p: int) -> torch.Tensor:
    r"""Projector onto Sym^p(R^N) inside (R^N)^{(x)p}: average of the p! slot permutations."""
    dim = N ** p
    T = torch.eye(dim, dtype=torch.float64).reshape([N] * p + [dim])
    S = torch.zeros(dim, dim, dtype=torch.float64)
    for perm in itertools.permutations(range(p)):
        S += T.permute(*perm, p).reshape(dim, dim)
    return S / math.factorial(p)


def _contraction(N: int, p: int) -> torch.Tensor:
    r"""Trace over slots (0, 1): (R^N)^{(x)p} -> (R^N)^{(x)(p-2)}. On SYMMETRIC tensors any
    slot pair gives the same contraction, so one pair suffices for the traceless condition."""
    dim_out = N ** (p - 2)
    C = torch.zeros(dim_out, N ** p, dtype=torch.float64)
    for rest in range(dim_out):
        for i in range(N):
            C[rest, (i * N + i) * dim_out + rest] = 1.0
    return C


def _monomial_sym_basis(N: int, p: int) -> torch.Tensor:   # (N^p, C(N+p-1, p))
    r"""Orthonormal basis of Sym^p(R^N) indexed by multisets in lexicographic order.

    Column alpha is the normalized sum of e_{i_sigma} over the DISTINCT orderings of the
    multiset alpha = (i_1 <= ... <= i_p); distinct multisets have disjoint supports, so the
    columns are orthonormal by construction and no eigensolver is involved.
    """
    multisets = list(itertools.combinations_with_replacement(range(N), p))
    B = torch.zeros(N ** p, len(multisets), dtype=torch.float64)
    for col, alpha in enumerate(multisets):
        orderings = set(itertools.permutations(alpha))
        value = 1.0 / math.sqrt(len(orderings))
        for idx in orderings:
            flat = 0
            for i in idx:
                flat = flat * N + i
            B[flat, col] = value
    return B


def _ordered_gram_schmidt(
    V:   torch.Tensor,                     # (n, m) candidate columns, taken in order
    tol: float = 1e-9,                     # relative residual below which a column is dependent
) -> torch.Tensor:                         # (n, r) orthonormal basis of span(V), r = rank
    """Deterministic orthonormal basis of a column span: Gram-Schmidt in column order, twice."""
    basis: List[torch.Tensor] = []
    scale = float(V.norm(dim=0).max()) if V.numel() else 0.0
    for j in range(V.shape[1]):
        v = V[:, j].clone()
        for _ in range(2):                                    # reorthogonalize once for stability
            for q in basis:
                v -= (q @ v) * q
        norm = float(v.norm())
        if norm > tol * max(scale, 1.0):
            basis.append(v / norm)
    return torch.stack(basis, dim=1) if basis else V.new_zeros(V.shape[0], 0)


def _canonical_invariant_basis(
    N:         int,                        # defining-rep dimension
    p:         int,                        # tensor-power rank

    *,
    traceless: bool,                       # additionally impose the trace-zero condition (so)
) -> torch.Tensor:                         # (N^p, d) deterministic orthonormal basis
    r"""Monomial basis of Sym^p(R^N); for the traceless subspace, the ordered Gram-Schmidt of the
    columns of its orthogonal projector P = I - M^+ M (unique as a subspace, so the basis is fixed
    by the column order alone)."""
    if p == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    if p == 1:
        return torch.eye(N, dtype=torch.float64)
    B = _monomial_sym_basis(N, p)
    if not traceless:
        return B
    M = _contraction(N, p) @ B                                # trace map in Sym^p coordinates
    P = torch.eye(B.shape[1], dtype=torch.float64) - torch.linalg.pinv(M) @ M
    return B @ _ordered_gram_schmidt(P)


def _invariant_basis(
    N:         int,                        # defining-rep dimension
    p:         int,                        # tensor-power rank

    *,
    traceless: bool,                       # additionally impose the trace-zero condition (so)
) -> torch.Tensor:                         # (N^p, d) orthonormal basis of the invariant subspace
    """Orthonormal basis of Sym^p(R^N) (optionally traceless) inside the tensor power ('eigh')."""
    if p == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    if p == 1:
        return torch.eye(N, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(_symmetrizer(N, p))
    B = evecs[:, evals > 0.5]                                 # Sym^p eigenspace (projector eval 1)
    if traceless:
        M = _contraction(N, p) @ B                            # trace map in Sym^p coordinates
        _, sv, Vh = torch.linalg.svd(M, full_matrices=True)
        rank = int((sv > 1e-10).sum())
        B = B @ Vh[rank:].T                                   # null space: traceless subspace
        B, _ = torch.linalg.qr(B)                             # numerical re-orthonormalization
    return B


def _tensor_power_irrep(
    G_def:     torch.Tensor,               # (n_gen, N, N) defining-rep algebra basis (float64)
    p:         int,                        # tensor-power rank (label's trailing integer)

    *,
    traceless: bool,                       # so: True (sym-traceless)
    basis:     str = "eigh",               # 'eigh' | 'canonical' (deterministic invariant basis)
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Restrict the Leibniz action to the invariant subspace: rho(X) = Q^T rho_p(X) Q."""
    n_gen, N, _ = G_def.shape
    if p == 0:
        return torch.zeros(n_gen, 1, 1, dtype=torch.float64)  # trivial rep: zero generators
    _guard_cost(N, p)
    if basis == "canonical":
        Q = _canonical_invariant_basis(N, p, traceless=traceless)
    else:
        Q = _invariant_basis(N, p, traceless=traceless)
    return torch.stack([Q.T @ _leibniz_action(G_def[a], p) @ Q for a in range(n_gen)])


# Standard rotation generators of R^3 in the (x, y, z) basis, (A_a)_{jk} = -epsilon_{ajk}, so that
# [A_x, A_y] = A_z cyclically: the bracket of A_a = -i J_a for angular-momentum operators J_a.
_SO3_STANDARD = torch.tensor([[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
                              [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],
                              [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]], dtype=torch.float64)


def _so3_ladder_real(
    l: int,                                # spin (block dimension 2l + 1)
) -> torch.Tensor:                         # (3, 2l+1, 2l+1) images of A_x, A_y, A_z
    r"""Spin-l images of the standard rotation generators in the real spherical-harmonic basis.

    Complex basis |l, m>, m = -l..l: J_z = diag(m), <m+1|J_+|m> = sqrt(l(l+1) - m(m+1)),
    J_- = J_+^T, J_x = (J_+ + J_-)/2, J_y = (J_+ - J_-)/(2i), and A_a = -i J_a. The real basis is
    S_0 = |0>, S_m = ((-1)^m |m> + |-m>)/sqrt 2 and S_{-m} = i(|-m> - (-1)^m |m>)/sqrt 2 for
    m > 0, in which every A_a is real antisymmetric (asserted).
    """
    d = 2 * l + 1
    m = [float(k) for k in range(-l, l + 1)]
    J_plus = torch.zeros(d, d, dtype=torch.complex128)
    for k in range(d - 1):                                    # <m+1|J_+|m>
        J_plus[k + 1, k] = math.sqrt(l * (l + 1) - m[k] * (m[k] + 1))
    J_minus = J_plus.conj().T
    J = (0.5 * (J_plus + J_minus), (J_plus - J_minus) / 2j,
         torch.diag(torch.tensor(m, dtype=torch.float64)).to(torch.complex128))
    U = torch.zeros(d, d, dtype=torch.complex128)             # rows: real index m; columns: |m>
    s = 1.0 / math.sqrt(2.0)
    for r in range(d):
        mr = r - l
        if mr == 0:
            U[r, l] = 1.0
        elif mr > 0:
            U[r, l + mr] = (-1) ** mr * s
            U[r, l - mr] = s
        else:
            a = -mr
            U[r, l - a] = 1j * s
            U[r, l + a] = -1j * (-1) ** a * s
    images = []
    for J_a in J:
        R = U @ (-1j * J_a) @ U.conj().T
        if float(R.imag.abs().max()) > 1e-12:
            raise RuntimeError(f"spin-{l} real-basis transform left an imaginary part")
        images.append(R.real.contiguous())
    return torch.stack(images)


def _so3_canonical_irrep(
    G_def: torch.Tensor,                   # (n_gen, 3, 3) caller's so(3) basis (float64)
    p:     int,                            # spin
) -> torch.Tensor:                         # (n_gen, 2p+1, 2p+1)
    """Spin-p block of so(3) in the canonical basis, expressed on the caller's algebra basis."""
    if p == 0:
        return torch.zeros(G_def.shape[0], 1, 1, dtype=torch.float64)
    if p == 1:
        return G_def.clone()                                  # the defining representation itself
    # G_def[k] = sum_a coeff[k, a] A_a. The A_a are Frobenius-orthogonal with ||A_a||_F^2 = 2, so the
    # coefficients are exact projections (deterministic, unlike a least-squares solve).
    coeff = torch.einsum("kij,aij->ka", G_def, _SO3_STANDARD) / 2.0               # (n_gen, 3)
    return torch.einsum("ka,aij->kij", coeff, _so3_ladder_real(p))


def structure_constants(
    G_def: torch.Tensor,                   # (n_gen, N, N) algebra basis (float64)
) -> torch.Tensor:                         # (n_gen, n_gen, n_gen) f with [Ga, Gb] = f[a,b,c] Gc
    """Structure constants of the defining basis via least squares (basis-agnostic)."""
    n = G_def.shape[0]
    A = G_def.reshape(n, -1).T.contiguous()                   # (N^2, n) flattened basis
    f = torch.zeros(n, n, n, dtype=torch.float64)
    for a in range(n):
        for b in range(a + 1, n):
            br = (G_def[a] @ G_def[b] - G_def[b] @ G_def[a]).reshape(-1, 1)
            sol = torch.linalg.lstsq(A, br).solution.squeeze(-1)
            f[a, b] = sol
            f[b, a] = -sol
    return f


def _assert_homomorphism(
    rho:   torch.Tensor,                   # (n_gen, d, d) candidate generator images
    f:     torch.Tensor,                   # (n_gen, n_gen, n_gen) defining structure constants

    *,
    atol:  float = 1e-8,
    label: str   = "",
) -> None:
    r"""Verify [rho(Xa), rho(Xb)] = f_{ab}^c rho(Xc); raise on residual (build-time guard)."""
    n = rho.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            lhs = rho[a] @ rho[b] - rho[b] @ rho[a]
            rhs = torch.einsum("c,cij->ij", f[a, b], rho)
            res = (lhs - rhs).abs().max().item()
            if res > atol:
                raise RuntimeError(
                    f"irrep {label!r} bracket-homomorphism residual {res:.3e} exceeds {atol:.0e} "
                    f"at generator pair ({a}, {b}); the constructed block is not a representation"
                )


def direct_sum_generators(
    G_def:      torch.Tensor,                       # (n_gen, N, N) defining-rep basis (float64)

    *,
    algebra:    str,                                # registry namespace: 'so'
    irrep_spec: Sequence[Tuple[str, int]],          # (label, multiplicity) pairs, in block order
    basis:      str = DEFAULT_IRREP_BASIS,          # 'canonical' | 'eigh'
) -> Tuple[torch.Tensor, List[int]]:                # ((n_gen, K, K) float64, per-block dims)
    """Assemble blockdiag_b rho_b(X_a) over the spec; each distinct label is built once,
    verified (dimension formula + bracket homomorphism), then tiled by its multiplicity."""
    N = G_def.shape[-1]
    n_gen = G_def.shape[0]
    f = structure_constants(G_def)

    built: Dict[str, torch.Tensor] = {}
    blocks: List[torch.Tensor] = []
    dims:   List[int] = []
    for label, mult in irrep_spec:
        if label not in built:
            key, p = _parse_label(algebra, label)
            dim_fn, build_fn = _IRREPS[key]
            rho = build_fn(G_def, p, _check_basis(basis))
            d_expect = dim_fn(N, p)
            if rho.shape[-1] != d_expect:
                raise RuntimeError(
                    f"irrep {label!r} over R^{N} built dimension {rho.shape[-1]} != closed-form "
                    f"{d_expect}; the invariant-subspace construction is inconsistent"
                )
            _assert_homomorphism(rho, f, label=label)
            built[label] = rho
        blocks.extend([built[label]] * mult)
        dims.extend([built[label].shape[-1]] * int(mult))

    K = sum(dims)
    G = torch.zeros(n_gen, K, K, dtype=torch.float64)
    start = 0
    for blk in blocks:
        d = blk.shape[-1]
        G[:, start:start + d, start:start + d] = blk
        start += d
    return G, dims


def _son_sym_traceless_dim(N: int, p: int) -> int:
    r"""dim = C(N+p-1, p) - C(N+p-3, p-2) (p >= 2); N for p = 1; 1 for p = 0. N=3: 2p+1."""
    if p == 0:
        return 1
    if p == 1:
        return N
    return math.comb(N + p - 1, p) - math.comb(N + p - 3, p - 2)


def _son_sym_traceless_build(G_def: torch.Tensor, p: int, basis: str) -> torch.Tensor:
    if basis == "canonical" and G_def.shape[-1] == 3:
        return _so3_canonical_irrep(G_def, p)
    return _tensor_power_irrep(G_def, p, traceless=True, basis=basis)


register_irrep("so:l",   _son_sym_traceless_dim, _son_sym_traceless_build)
