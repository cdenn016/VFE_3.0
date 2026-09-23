r"""Irreducible-representation registry for VFE_3.0 gauge groups (heads = irreps).

Builds the generator images rho(X_a) of a structure algebra g on an irrep block, so a
gauge group can act on the K-dim embedding through a DIRECT SUM of irreps,

    G_a = blockdiag_b rho_b(X_a),        a = 1, ..., n_gen = dim g,

with ONE shared per-token phi -- a TIED gauge: the same group element acts in every
block through that block's representation, so attention heads can be DIFFERENT irreps
of one structure group (possibly with unequal dimensions). Registered families:

    so : label 'l<p>'   -- symmetric-traceless rank-p tensors over R^N,
                           dim = C(N+p-1, p) - C(N+p-3, p-2); for N = 3 this is the
                           spin-p tower (dim 2p + 1). Built in real orthonormal bases,
                           so every generator image is exactly skew.
    sp : label 'sym<p>' -- Sym^p of the defining 2m-dim rep, dim = C(2m+p-1, p).
                           Irreducible because contraction with the ANTI-symmetric
                           symplectic form annihilates symmetric tensors. Not skew.

Construction (no per-irrep sign conventions to get wrong): the rank-p tensor power
carries the Leibniz action

    rho_p(X) = sum_{k=0}^{p-1} I^{(k)} (x) X (x) I^{(p-1-k)}      on (R^N)^{(x)p},

and the irrep is its restriction rho(X) = Q^T rho_p(X) Q to an orthonormal basis Q of
the invariant subspace (symmetric, and traceless for so). The homomorphism property
[rho(X), rho(Y)] = rho([X, Y]) is then automatic, and is VERIFIED at build time against
the structure constants of the defining basis (raise, not warn, on residual). All
construction runs in float64; callers cast.

A new irrep family (Young-tableau highest-weight reps, spinors, ...) slots in by
``register_irrep`` -- never by editing call sites (the modularity contract).
"""

import itertools
import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch

# key 'algebra:prefix' -> (dim_fn(N, p) -> int, build_fn(G_def, p) -> (n_gen, d, d))
_IRREPS: Dict[str, Tuple[Callable[[int, int], int],
                         Callable[[torch.Tensor, int], torch.Tensor]]] = {}


def register_irrep(
    key:      str,                                          # 'algebra:prefix', e.g. 'so:l'
    dim_fn:   Callable[[int, int], int],                    # (N, rank) -> block dimension
    build_fn: Callable[[torch.Tensor, int], torch.Tensor],  # (G_def, rank) -> (n_gen, d, d)

    *,
    override: bool = False,
) -> None:
    """Register an irrep family under ``key`` (label grammar: '<prefix><rank>').

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing key silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    if key in _IRREPS and not override:
        raise KeyError(f"irrep family {key!r} already registered; pass override=True to replace")
    _IRREPS[key] = (dim_fn, build_fn)


def _parse_label(algebra: str, label: str) -> Tuple[str, int]:
    """Split 'l2' -> ('so:l', 2); raise with the registered keys on an unknown label."""
    prefix = label.rstrip("0123456789")
    digits = label[len(prefix):]
    key = f"{algebra}:{prefix}"
    if not digits or key not in _IRREPS:
        raise ValueError(
            f"unknown irrep label {label!r} for algebra {algebra!r}; registered families: "
            f"{sorted(_IRREPS)} (label grammar '<family><rank>', e.g. 'l2', 'sym1')"
        )
    return key, int(digits)


def irrep_dim(
    N:       int,                          # defining-rep dimension (N of SO(N); 2m of Sp(2m))

    *,
    algebra: str,                          # 'so' | 'sp'
    label:   str,                          # e.g. 'l2' (so), 'sym3' (sp)
) -> int:
    """Block dimension of ``label`` over the rank-N algebra (closed form, no tensors built)."""
    key, p = _parse_label(algebra, label)
    return _IRREPS[key][0](N, p)


def irrep_generators(
    G_def:   torch.Tensor,                 # (n_gen, N, N) defining-rep algebra basis (float64)

    *,
    algebra: str,                          # 'so' | 'sp'
    label:   str,                          # e.g. 'l2' (so), 'sym3' (sp)
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Build one irrep's generator images (the registry's public single-label entry point)."""
    key, p = _parse_label(algebra, label)
    return _IRREPS[key][1](G_def, p)


def _guard_cost(N: int, p: int) -> None:
    """The tensor-power construction works on N^p x N^p matrices with p! symmetrizer terms."""
    if p > 6 or N ** p > 60000:
        raise ValueError(
            f"rank-{p} tensor-power irrep over R^{N} exceeds the supported construction size "
            f"(rank <= 6 and N^rank <= 60000); higher ranks await the Young-tableau buildout."
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


def _invariant_basis(
    N:         int,                        # defining-rep dimension
    p:         int,                        # tensor-power rank

    *,
    traceless: bool,                       # additionally impose the trace-zero condition (so)
) -> torch.Tensor:                         # (N^p, d) orthonormal basis of the invariant subspace
    """Orthonormal basis of Sym^p(R^N) (optionally traceless) inside the tensor power."""
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
    traceless: bool,                       # so: True (sym-traceless); sp: False (plain Sym^p)
) -> torch.Tensor:                         # (n_gen, d, d) generator images on the irrep
    """Restrict the Leibniz action to the invariant subspace: rho(X) = Q^T rho_p(X) Q."""
    n_gen, N, _ = G_def.shape
    if p == 0:
        return torch.zeros(n_gen, 1, 1, dtype=torch.float64)  # trivial rep: zero generators
    _guard_cost(N, p)
    Q = _invariant_basis(N, p, traceless=traceless)
    return torch.stack([Q.T @ _leibniz_action(G_def[a], p) @ Q for a in range(n_gen)])


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
    algebra:    str,                                # registry namespace: 'so' | 'sp'
    irrep_spec: Sequence[Tuple[str, int]],          # (label, multiplicity) pairs, in block order
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
    return _tensor_power_irrep(G_def, p, traceless=True)


def _sp_sym_dim(N: int, p: int) -> int:
    r"""dim Sym^p(R^{2m}) = C(2m+p-1, p) (irreducible for sp(2m): the symplectic contraction
    of a symmetric tensor vanishes)."""
    return math.comb(N + p - 1, p) if p > 0 else 1


def _sp_sym_build(G_def: torch.Tensor, p: int) -> torch.Tensor:
    return _tensor_power_irrep(G_def, p, traceless=False)


register_irrep("so:l",   _son_sym_traceless_dim, _son_sym_traceless_build)
register_irrep("sp:sym", _sp_sym_dim,            _sp_sym_build)
