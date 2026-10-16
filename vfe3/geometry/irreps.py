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

Every block is built in one deterministic (canonical) basis. For N = 3 the spin-l block is
built in closed form from the angular-momentum ladder, J_z |l,m> = m |l,m> and
J_+ |l,m> = sqrt(l(l+1) - m(m+1)) |l,m+1> (Varshalovich, Moskalev and Khersonskii,
*Quantum Theory of Angular Momentum*, 1988), carried to the real spherical-harmonic basis
(Blanco, Florez and Bermejo, J. Mol. Struct. THEOCHEM 419 (1997) 19-27), for any l at
O(l^2) cost; l = 1 is the defining representation itself. For N != 3 the tensor-power
construction below runs on a fixed monomial basis of Sym^p with a fixed-order Gram-Schmidt
for the traceless subspace, so no eigensolver chooses the basis. The original eigh-based
tensor-power basis (the retired irrep_basis="eigh") was removed 2026-10-12.

Tensor-power construction: the rank-p tensor power carries the Leibniz action

    rho_p(X) = sum_{k=0}^{p-1} I^{(k)} (x) X (x) I^{(p-1-k)}      on (R^N)^{(x)p},

and the irrep is its restriction rho(X) = Q^T rho_p(X) Q to an orthonormal basis Q of
the invariant subspace (symmetric, and traceless for so). The homomorphism
property [rho(X), rho(Y)] = rho([X, Y]) is VERIFIED at build time against the structure
constants of the defining basis (raise, not warn, on residual). All construction runs in
float64; callers cast.

A new irrep family (Young-tableau highest-weight reps, spinors, ...) slots in by
``register_irrep`` -- never by editing call sites (the modularity contract).
"""

import itertools
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

# key 'algebra:prefix' -> (dim_fn(N, p) -> int, build_fn(G_def, p) -> (n_gen, d, d))
_IRREPS: Dict[str, Tuple[Callable[[int, int], int],
                         Callable[[torch.Tensor, int], torch.Tensor]]] = {}
# key -> label arity: 1 for '<prefix><p>' (rank p), 2 for '<prefix><p>_<q>' (rank (p, q)).
_IRREP_ARITY: Dict[str, int] = {}


def register_irrep(
    key:      str,                                               # 'algebra:prefix', e.g. 'so:l'
    dim_fn:   Callable[[int, int], int],                         # (N, rank) -> block dimension
    build_fn: Callable[[torch.Tensor, int], torch.Tensor],       # (G_def, rank) -> (n_gen, d, d)

    *,
    override: bool = False,
    arity:    int  = 1,                                          # 1: '<prefix><p>'; 2: '<prefix><p>_<q>'
) -> None:
    """Register an irrep family under ``key`` (label grammar: '<prefix><rank>', or '<prefix><p>_<q>' at arity 2).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second
    registration under an existing key silently shadowed the first. Pass ``override=True`` to
    replace deliberately.
    """
    if key in _IRREPS and not override:
        raise KeyError(f"irrep family {key!r} already registered; pass override=True to replace")
    if arity not in (1, 2):
        raise ValueError(f"irrep label arity must be 1 or 2, got {arity!r}")
    _IRREPS[key] = (dim_fn, build_fn)
    _IRREP_ARITY[key] = arity


def _parse_label(algebra: str, label: str) -> 'Tuple[str, int | Tuple[int, int]]':
    """Split 'l2' -> ('so:l', 2), or 'pq1_1' -> ('su:pq', (1, 1)) for an arity-2 family; raise on an unknown label."""
    two = re.fullmatch(r"([A-Za-z]+)(\d+)_(\d+)", label)
    if two is not None and _IRREP_ARITY.get(f"{algebra}:{two.group(1)}") == 2:
        return f"{algebra}:{two.group(1)}", (int(two.group(2)), int(two.group(3)))
    prefix = label.rstrip("0123456789")
    digits = label[len(prefix):]
    key = f"{algebra}:{prefix}"
    if not digits or key not in _IRREPS or _IRREP_ARITY.get(key, 1) != 1:
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
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Build one irrep's generator images (the registry's public single-label entry point)."""
    key, p = _parse_label(algebra, label)
    return _IRREPS[key][1](G_def, p)


def check_irrep_constructible(
    N:       int,                          # defining-rep dimension

    *,
    algebra: str,                          # 'so'
    label:   str,                          # e.g. 'l8'
) -> None:
    """Raise, before any tensor is built, if ``label`` cannot be constructed."""
    key, p = _parse_label(algebra, label)
    _IRREPS[key][0](N, p)                  # the family's dimension formula refuses an undefined (N, rank)
    if key == "su:pq":
        if tuple(p) not in ((0, 0), (1, 1)) and sum(p) >= 2:
            _guard_cost(N, sum(p))           # the tensor construction works on N^(p+q) x N^(p+q) matrices
        return
    if key in _WEDGE_FAMILIES:
        return                             # direct construction on the p-vector basis: no tensor power
    if algebra == "so" and N == 3:
        return                             # closed-form ladder construction: any rank
    if p >= 2:
        _guard_cost(N, p)


def _guard_cost(N: int, p: int) -> None:
    """The tensor-power construction works on N^p x N^p matrices with p! symmetrizer terms."""
    if p > 6 or N ** p > 60000:
        raise ValueError(
            f"rank-{p} tensor-power irrep over R^{N} exceeds the supported construction size "
            f"(rank <= 6 and N^rank <= 60000); SO(3) builds any spin in closed form; higher "
            f"ranks for N != 3 await the Young-tableau buildout."
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


def _tensor_power_irrep(
    G_def:     torch.Tensor,               # (n_gen, N, N) defining-rep algebra basis (float64)
    p:         int,                        # tensor-power rank (label's trailing integer)

    *,
    traceless: bool,                       # so: True (sym-traceless)
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Restrict the Leibniz action to the invariant subspace: rho(X) = Q^T rho_p(X) Q."""
    n_gen, N, _ = G_def.shape
    if p == 0:
        return torch.zeros(n_gen, 1, 1, dtype=torch.float64)  # trivial rep: zero generators
    _guard_cost(N, p)
    Q = _canonical_invariant_basis(N, p, traceless=traceless)
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
) -> Tuple[torch.Tensor, List[int]]:                # ((n_gen, K, K) float64, per-block dims)
    """Assemble blockdiag_b rho_b(X_a) over the spec; each distinct label is built once,
    verified (dimension formula + bracket homomorphism), then tiled by its multiplicity.
    For 'su' the defining basis is the realified 2N x 2N one, so N is half its width."""
    N = G_def.shape[-1] // 2 if algebra == "su" else G_def.shape[-1]
    n_gen = G_def.shape[0]
    f = structure_constants(G_def)

    built: Dict[str, torch.Tensor] = {}
    blocks: List[torch.Tensor] = []
    dims:   List[int] = []
    for label, mult in irrep_spec:
        if label not in built:
            key, p = _parse_label(algebra, label)
            dim_fn, build_fn = _IRREPS[key]
            rho = build_fn(G_def, p)
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


def _son_sym_traceless_build(G_def: torch.Tensor, p: int) -> torch.Tensor:
    if G_def.shape[-1] == 3:
        return _so3_canonical_irrep(G_def, p)
    return _tensor_power_irrep(G_def, p, traceless=True)


register_irrep("so:l",   _son_sym_traceless_dim, _son_sym_traceless_build)


# ---------------------------------------------------------------- exterior powers (wedge tensors)
#
# Lambda^p R^N under SO(N). For 2 <= p < N/2 it is irreducible over R, dim C(N, p). Lambda^0 and
# Lambda^1 are 'l0' and 'l1', and Lambda^p = Lambda^{N-p} by the Hodge star, so the 'w' family takes
# 2 <= p <= N/2 only. At p = m = N/2 the Hodge star commutes with SO(N) and squares to (-1)^m:
#   m odd  (N = 2 mod 4): no real eigenspaces; Lambda^m is irreducible over R (complex type), 'w<m>';
#   m even (N = 0 mod 4): Lambda^m = Lambda^m_+ (+) Lambda^m_-, the self-dual 'wp<m>' and
#                         anti-self-dual 'wm<m>' irreps, each of dim C(N, m)/2; 'w<m>' is refused.
# Every construction is a fixed integer recipe on the orthonormal basis e_I (I an increasing p-tuple
# in lexicographic order), so the basis is deterministic.

_WEDGE_FAMILIES = ("so:w", "so:wp", "so:wm")


def _wedge_dim(N: int, p: int) -> int:
    r"""dim Lambda^p = C(N, p) for the 'w' family: 2 <= p < N/2, or p = N/2 with N = 2 mod 4."""
    if p < 2 or 2 * p > N:
        raise ValueError(
            f"wedge irrep 'w{p}' over R^{N} needs 2 <= p <= N/2 (Lambda^0, Lambda^1 are 'l0', 'l1'; "
            f"Lambda^p = Lambda^(N-p) by the Hodge star)")
    if 2 * p == N and p % 2 == 0:
        raise ValueError(
            f"Lambda^{p} over R^{N} splits into self-dual and anti-self-dual irreps; use 'wp{p}' and 'wm{p}'")
    return math.comb(N, p)


def _wedge_half_dim(N: int, p: int) -> int:
    r"""dim Lambda^m_(+/-) = C(N, m)/2 for m = N/2 even ('wp' / 'wm')."""
    if 2 * p != N or p % 2 != 0 or p < 2:
        raise ValueError(
            f"self-dual wedge irreps 'wp{p}' / 'wm{p}' need p = N/2 even (N = 0 mod 4), got N={N}, p={p}")
    return math.comb(N, p) // 2


def _wedge_action(G_def: torch.Tensor, p: int) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
    r"""rho(X) on Lambda^p as the derivation X(e_I) = sum_k e_{i_1} ^ ... ^ X e_{i_k} ^ ... ^ e_{i_p}.

    Returns the (n_gen, C(N, p), C(N, p)) float64 images on the orthonormal basis e_I and the basis
    tuples I. Skew X gives skew images (the induced inner product is SO(N)-invariant).
    """
    import itertools

    N = G_def.shape[-1]
    tuples = list(itertools.combinations(range(N), p))
    index = {I: n for n, I in enumerate(tuples)}
    rho = torch.zeros(G_def.shape[0], len(tuples), len(tuples), dtype=torch.float64)
    G = G_def.to(torch.float64)
    for a in range(G.shape[0]):
        rows, cols = torch.nonzero(G[a], as_tuple=True)                # X e_i = sum_j X[j, i] e_j
        entries = [(int(j), int(i), float(G[a, j, i])) for j, i in zip(rows, cols)]
        for col, I in enumerate(tuples):
            for k, i in enumerate(I):
                for j, i_src, value in entries:
                    if i_src != i or (j in I and j != i):
                        continue
                    J = list(I)
                    J[k] = j
                    # Sign of the permutation sorting J (one entry moved): count the entries it passes.
                    sign = (-1) ** sum(1 for m, other in enumerate(J) if m != k and (other < j) != (m < k))
                    rho[a, index[tuple(sorted(J))], col] += sign * value
    return rho, tuples


def _wedge_build(G_def: torch.Tensor, p: int) -> torch.Tensor:
    _wedge_dim(G_def.shape[-1], p)
    return _wedge_action(G_def, p)[0]


def _hodge_half(G_def: torch.Tensor, p: int, sign: int) -> torch.Tensor:
    r"""rho on Lambda^m_(sign) for m = N/2 even: the basis (e_I + sign * star e_I)/sqrt 2 over I containing 0.

    star e_I = eps(I, I^c) e_{I^c}, with eps the sign of the permutation (I, I^c) of (0, ..., N-1).
    Every pair {I, I^c} has exactly one member containing 0, so these vectors are an orthonormal
    basis of the (sign)-eigenspace of star.
    """
    N = G_def.shape[-1]
    _wedge_half_dim(N, p)
    rho, tuples = _wedge_action(G_def, p)
    index = {I: n for n, I in enumerate(tuples)}
    columns = []
    for I in tuples:
        if 0 not in I:
            continue
        complement = tuple(i for i in range(N) if i not in I)
        order = list(I) + list(complement)
        inversions = sum(1 for x in range(N) for y in range(x + 1, N) if order[x] > order[y])
        v = torch.zeros(len(tuples), dtype=torch.float64)
        v[index[I]] = 1.0
        v[index[complement]] += sign * (-1) ** inversions
        columns.append(v / 2.0 ** 0.5)
    P = torch.stack(columns, dim=1)                                    # (C(N, m), C(N, m)/2)
    return P.T @ rho @ P


def _wedge_plus_build(G_def: torch.Tensor, p: int) -> torch.Tensor:
    return _hodge_half(G_def, p, +1)


def _wedge_minus_build(G_def: torch.Tensor, p: int) -> torch.Tensor:
    return _hodge_half(G_def, p, -1)


register_irrep("so:w",   _wedge_dim,      _wedge_build)
register_irrep("so:wp",  _wedge_half_dim, _wedge_plus_build)
register_irrep("so:wm",  _wedge_half_dim, _wedge_minus_build)



# ---------------------------------------------------------------- SU(N) irreps (p, q)
#
# Label 'pq<p>_<q>': the irrep with highest weight p (fundamental) + q (antifundamental), the traceless
# tensors in Sym^p(C^N) (x) Sym^q(conj C^N). The defining basis is the realified su(N) basis
# (generators.generate_sun_real), a faithful real representation, so the structure constants are the
# real f_abc of X_a = i lambda_a. The adjoint 'pq1_1' is ad(X_a)_{cb} = f_abc (real, skew, real type). Every
# other (p, q) is built on the traceless tensors of Sym^p (x) Sym^q (_su_tensor_irrep): realified to real
# dimension 2D when p != q (complex type), and given a real form of dimension D when p = q (real type).


def _su_pq_dim(N: int, pq: Tuple[int, int]) -> int:
    r"""Real dimension of the su(N) irrep (p, q): the complex dimension
    :math:`D = \binom{p+N-1}{p}\binom{q+N-1}{q} - \binom{p+N-2}{p-1}\binom{q+N-2}{q-1}` for the real type p = q,
    2D for the complex type p != q (realified). N >= 3 and p >= q; (q, p) is R-equivalent to (p, q)."""
    p, q = pq
    if N < 3:
        raise ValueError(f"su_n needs group_n >= 3 (SU(2) is so_n with group_n=3, 'l<p>' = spin p), got N={N}")
    if p < 0 or q < 0 or p < q:
        raise ValueError(f"su({N}) irrep labels need p >= q >= 0 ((q, p) is R-equivalent to (p, q)), got ({p}, {q})")

    def comb(n: int, k: int) -> int:
        return math.comb(n, k) if k >= 0 else 0

    D = comb(p + N - 1, p) * comb(q + N - 1, q) - comb(p + N - 2, p - 1) * comb(q + N - 2, q - 1)
    return D if p == q else 2 * D


def su_casimir(N: int, p: int, q: int) -> float:
    r"""Quadratic Casimir :math:`C_2(p,q) = \tfrac12\bigl[(p^2+q^2)\tfrac{N-1}{N} + \tfrac{2pq}{N} + (p+q)(N-1)\bigr]`
    in the normalization :math:`\sum_a T_a^2` with :math:`T_a = \lambda_a/2`; for :math:`X_a = i\lambda_a` the
    irrep satisfies :math:`-\sum_a\rho(X_a)^2 = 4C_2 I`. Fundamental: (N^2 - 1)/(2N); adjoint: N."""
    return 0.5 * ((p * p + q * q) * (N - 1) / N + 2 * p * q / N + (p + q) * (N - 1))


def _su_complex_basis(G_def: torch.Tensor) -> torch.Tensor:          # (n_gen, N, N) complex128
    r"""Undo :func:`vfe3.geometry.generators.realify`: X = A + iB from the realified [[A, -B], [B, A]]."""
    N = G_def.shape[-1] // 2
    return torch.complex(G_def[:, :N, :N], G_def[:, N:, :N])


def _mixed_leibniz(X: torch.Tensor, p: int, q: int) -> torch.Tensor:   # (N^(p+q), N^(p+q)) complex
    r"""Action on :math:`(\mathbb{C}^N)^{\otimes p}\otimes(\overline{\mathbb{C}^N})^{\otimes q}`: X on the p
    fundamental slots, :math:`\bar X` on the q antifundamental slots (Leibniz rule)."""
    N = X.shape[-1]
    eye = torch.eye(N, dtype=X.dtype)
    out = torch.zeros(N ** (p + q), N ** (p + q), dtype=X.dtype)
    for k in range(p + q):
        M = torch.eye(1, dtype=X.dtype)
        for slot in range(p + q):
            M = torch.kron(M, (X if k < p else X.conj()) if slot == k else eye)
        out += M
    return out


def _mixed_trace_basis(N: int, p: int, q: int) -> torch.Tensor:      # (N^(p+q), D) float64
    r"""Real orthonormal basis of the traceless tensors in :math:`\mathrm{Sym}^p\otimes\mathrm{Sym}^q`.

    The monomial bases of the two symmetric powers are real, and so is the fund-antifund contraction of slot 0
    with slot p (on symmetric tensors every such pair gives the same contraction), so the invariant subspace has
    a real basis: ordered Gram-Schmidt of the projector off the contraction, as for so(N).
    """
    S = torch.kron(_canonical_invariant_basis(N, p, traceless=False),
                   _canonical_invariant_basis(N, q, traceless=False))     # (N^(p+q), m)
    if p == 0 or q == 0:
        return S
    rest = N ** (p + q - 2)
    C = torch.zeros(rest, N ** (p + q), dtype=torch.float64)
    for r in range(rest):
        digits = []
        x = r
        for _ in range(p + q - 2):
            digits.append(x % N)
            x //= N
        digits = digits[::-1]                                         # the p-1 + q-1 untouched slots
        for i in range(N):
            idx = [i] + digits[:p - 1] + [i] + digits[p - 1:]
            flat = 0
            for v in idx:
                flat = flat * N + v
            C[r, flat] = 1.0
    M = C @ S
    P = torch.eye(S.shape[1], dtype=torch.float64) - torch.linalg.pinv(M) @ M
    return S @ _ordered_gram_schmidt(P)


def _slot_swap(N: int, p: int) -> torch.Tensor:                        # (N^(2p), N^(2p)) real permutation
    r"""Permutation exchanging the p fundamental slots with the p antifundamental slots (p = q)."""
    n = N ** (2 * p)
    perm = torch.zeros(n, n, dtype=torch.float64)
    for flat in range(n):
        digits, x = [], flat
        for _ in range(2 * p):
            digits.append(x % N)
            x //= N
        digits = digits[::-1]
        swapped = digits[p:] + digits[:p]
        target = 0
        for v in swapped:
            target = target * N + v
        perm[target, flat] = 1.0
    return perm


def _su_tensor_irrep(G_def: torch.Tensor, p: int, q: int) -> torch.Tensor:   # (n_gen, d, d) float64
    r"""su(N) irrep (p, q) in a real orthonormal basis, verified at build time.

    On the traceless tensors W (basis Q, real) the complex images are :math:`\rho_{\mathbb C}(X) = Q^\top L(X) Q`,
    anti-Hermitian. For p != q (complex type) the real form is the realification, dimension 2D. For p = q (real
    type) the antilinear :math:`J = P\circ\overline{\,\cdot\,}` (P the slot swap) commutes with the action and
    squares to 1; in the Q coordinates it is :math:`a \mapsto M\bar a` with :math:`M = Q^\top P Q` a real symmetric
    involution, so the real subspace is spanned by the +1 eigenvectors of M and i times the -1 eigenvectors, a
    real orthonormal basis B of dimension D in which :math:`B^\dagger\rho_{\mathbb C}B` is real. Checks (raise): the
    imaginary residual, skewness, and the Casimir :math:`-\sum_a\rho_a^2 = 4C_2(p,q)I`.

    Built single-threaded: a threaded matmul changes the reduction order and the last bit of the images (seen
    for (2, 2) at four threads), and the tower fingerprint hashes the bytes.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _su_tensor_irrep_unthreaded(G_def, p, q)
    finally:
        torch.set_num_threads(threads)


def _su_tensor_irrep_unthreaded(G_def: torch.Tensor, p: int, q: int) -> torch.Tensor:   # (n_gen, d, d) float64
    """The body of :func:`_su_tensor_irrep`, run under its single-thread guard."""
    X = _su_complex_basis(G_def)
    N = X.shape[-1]
    Q = _mixed_trace_basis(N, p, q).to(torch.complex128)
    rho_c = torch.stack([Q.T @ _mixed_leibniz(X[a], p, q) @ Q for a in range(X.shape[0])])
    if p != q:
        A, B = rho_c.real, rho_c.imag
        rho = torch.cat([torch.cat([A, -B], dim=-1), torch.cat([B, A], dim=-1)], dim=-2)
    else:
        Mswap = (Q.T @ _slot_swap(N, p).to(torch.complex128) @ Q).real
        if float((Mswap @ Mswap - torch.eye(Mswap.shape[0], dtype=torch.float64)).abs().max()) > 1e-9:
            raise RuntimeError(f"su({N}) irrep ({p}, {p}): the slot swap is not an involution on the irrep")
        eye = torch.eye(Mswap.shape[0], dtype=torch.float64)
        plus = _ordered_gram_schmidt(0.5 * (eye + Mswap))
        minus = _ordered_gram_schmidt(0.5 * (eye - Mswap))
        if plus.shape[1] + minus.shape[1] != Mswap.shape[0]:
            raise RuntimeError(f"su({N}) irrep ({p}, {p}): the swap eigenspaces do not span the irrep")
        basis = torch.cat([plus.to(torch.complex128), 1j * minus.to(torch.complex128)], dim=1)
        rho_full = basis.conj().T @ rho_c @ basis
        imag = float(rho_full.imag.abs().max())
        if imag > 1e-9:
            raise RuntimeError(f"su({N}) irrep ({p}, {p}) real form has imaginary residual {imag:.3e}")
        rho = rho_full.real
    skew = float((rho + rho.transpose(-1, -2)).abs().max())
    if skew > 1e-9:
        raise RuntimeError(f"su({N}) irrep ({p}, {q}) images are not skew (residual {skew:.3e})")
    rho = 0.5 * (rho - rho.transpose(-1, -2))                         # exactly skew
    d = rho.shape[-1]
    casimir = -(rho @ rho).sum(0)
    target = 4.0 * su_casimir(N, p, q) * torch.eye(d, dtype=torch.float64)
    if float((casimir - target).abs().max()) > 1e-8 * max(1.0, float(target.max())):
        raise RuntimeError(f"su({N}) irrep ({p}, {q}) Casimir differs from 4 C_2 = {4 * su_casimir(N, p, q):.6g}")
    return rho.contiguous()


def _su_pq_build(G_def: torch.Tensor, pq: Tuple[int, int]) -> torch.Tensor:
    r"""Generator images of the su(N) irrep (p, q) on a real orthonormal basis: 'pq0_0' is zero (1 x 1), 'pq1_1' is
    the adjoint :math:`\mathrm{ad}(X_a)_{cb} = f_{abc}` of the defining basis (kept from M1, so its fingerprint is
    unchanged), and every other (p, q) comes from :func:`_su_tensor_irrep`."""
    n_gen = G_def.shape[0]
    if tuple(pq) == (0, 0):
        return torch.zeros(n_gen, 1, 1, dtype=torch.float64)
    if tuple(pq) == (1, 1):
        # The realified su(N) basis is Frobenius-orthogonal with one common weight, so the structure constants
        # are the exact projections f_abc = <R_c, [R_a, R_b]>_F / ||R_c||_F^2: deterministic products only (a
        # least-squares solve is not bit-reproducible, and the basis fingerprint must be).
        gram = torch.einsum("aij,bij->ab", G_def, G_def)
        weight = gram.diagonal()
        if float((gram - torch.diag(weight)).abs().max()) > 1e-12 * float(weight.max()):
            raise RuntimeError("the su(N) defining basis must be Frobenius-orthogonal")
        brackets = torch.einsum("aij,bjk->abik", G_def, G_def)
        brackets = brackets - brackets.transpose(0, 1)                  # (n, n, 2N, 2N) [R_a, R_b]
        f = torch.einsum("abij,cij->abc", brackets, G_def) / weight     # [R_a, R_b] = f[a, b, c] R_c
        rho = f.transpose(1, 2)                                         # rho[a, c, b] = f[a, b, c]
        # f is totally antisymmetric in this basis; antisymmetrize the rounding so every image is exactly skew.
        return (0.5 * (rho - rho.transpose(1, 2))).contiguous()
    return _su_tensor_irrep(G_def, int(pq[0]), int(pq[1]))


register_irrep("su:pq", _su_pq_dim, _su_pq_build, arity=2)


def _multiset_counts(
    dims:    List[int],                    # candidate block dimensions, one per label
    n_heads: int,
    K:       int,
) -> List[List[List[int]]]:                # count[i][h][k]
    r"""count[i][h][k]: multisets of h labels from dims[i:] with total dimension k (exact integers)."""
    m = len(dims)
    count = [[[0] * (K + 1) for _ in range(n_heads + 1)] for _ in range(m + 1)]
    count[m][0][0] = 1
    for i in range(m - 1, -1, -1):
        d = dims[i]
        for h in range(n_heads + 1):
            for k in range(K + 1):
                total = 0
                for t in range(h + 1):
                    if t * d > k:
                        break
                    total += count[i + 1][h - t][k - t * d]
                count[i][h][k] = total
    return count


def random_irrep_spec(
    N:        int,                         # defining-rep dimension (N of SO(N))
    K:        int,                         # embed_dim: the block dimensions must sum to K
    n_heads:  int,                         # number of irrep blocks (one per head)

    *,
    algebra:  str           = "so",
    seed:     int           = 0,
    min_rank: int           = 1,           # 1 skips the trivial rank-0 block (no generators)
    max_rank: Optional[int] = None,        # None: every rank whose block fits and is constructible
) -> List[Tuple[str, int]]:
    r"""Draw an irrep tower of exactly ``n_heads`` blocks whose dimensions sum to ``K``.

    The candidates are the labels of every family registered for ``algebra`` with rank in
    [``min_rank``, ``max_rank``] that fit in K and are constructible. The draw is
    uniform over the multisets of candidate labels with ``n_heads`` elements and total dimension
    K, sampled exactly from the counts by dynamic programming with ``random.Random(seed)``. The
    result is in ascending (family, rank) order with multiplicities, so equal blocks are
    contiguous and the operator storage has the fewest runs. When no such multiset exists the
    error lists the candidate dimensions (for SO(3) every block is odd, so K and n_heads must have
    the same parity; when K / n_heads equals the smallest candidate dimension the draw is forced).
    """
    import random

    if type(n_heads) is not int or n_heads < 1 or type(K) is not int or K < n_heads:
        raise ValueError(f"random irrep spec needs integers K >= n_heads >= 1, got K={K!r}, n_heads={n_heads!r}")
    if (type(min_rank) is not int or min_rank < 0
            or (max_rank is not None and (type(max_rank) is not int or max_rank < min_rank))):
        raise ValueError(f"irrep rank bounds need integers 0 <= min_rank <= max_rank, got "
                         f"min_rank={min_rank!r}, max_rank={max_rank!r}")
    top = K if max_rank is None else max_rank
    candidates: List[Tuple[str, int]] = []                     # (label, dim) in draw order
    for key in sorted(k for k in _IRREPS if k.startswith(f"{algebra}:") and _IRREP_ARITY.get(k, 1) == 1):
        prefix = key.split(":", 1)[1]
        for p in range(min_rank, top + 1):
            label = f"{prefix}{p}"
            try:
                d = irrep_dim(N, algebra=algebra, label=label)
            except ValueError:
                continue                                       # the family is undefined at this (N, rank)
            if d > K:
                if N >= 3:                                     # dims grow with rank for N >= 3
                    break
                continue
            try:
                check_irrep_constructible(N, algebra=algebra, label=label)
            except ValueError:
                break                                          # higher ranks cost more still
            candidates.append((label, d))
    if not candidates:
        raise ValueError(f"no {algebra}({N}) irrep with rank in [{min_rank}, {top}] fits K={K}")

    count = _multiset_counts([d for _, d in candidates], n_heads, K)
    if count[0][n_heads][K] == 0:
        dims = sorted({d for _, d in candidates})
        # Nearest feasible totals for the same head count, over every rank that fits in 2K.
        wide = []
        for key in sorted(k for k in _IRREPS if k.startswith(f"{algebra}:")):
            prefix = key.split(":", 1)[1]
            for p in range(min_rank, (2 * K if max_rank is None else max_rank) + 1):
                try:
                    d = irrep_dim(N, algebra=algebra, label=f"{prefix}{p}")
                    check_irrep_constructible(N, algebra=algebra, label=f"{prefix}{p}")
                except ValueError:
                    continue
                if d <= 2 * K:
                    wide.append(d)
        reach = _multiset_counts(sorted(wide), n_heads, 2 * K)[0][n_heads] if wide else []
        below = max((k for k in range(K) if reach[k]), default=None)
        above = min((k for k in range(K + 1, len(reach)) if reach[k]), default=None)
        raise ValueError(
            f"no {n_heads} {algebra}({N}) irreps with rank in [{min_rank}, {top}] have dimensions "
            f"summing to K={K}: the candidate block dimensions are {dims} (labels "
            f"{[label for label, _ in candidates]}). Nearest feasible embed_dim for n_heads={n_heads}: "
            f"{below} below, {above} above (None: none up to 2K). Change K, n_heads or the rank bounds.")

    rng = random.Random(seed)
    spec: List[Tuple[str, int]] = []
    h, k = n_heads, K
    for i, (label, d) in enumerate(candidates):
        r = rng.randrange(count[i][h][k])                      # index of the multiset still to draw
        t = 0
        while True:
            ways = count[i + 1][h - t][k - t * d]
            if r < ways:
                break
            r -= ways
            t += 1
        if t > 0:
            spec.append((label, t))
        h, k = h - t, k - t * d
    return spec
