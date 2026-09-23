r"""Numerical Clebsch-Gordan intertwiners for VFE_3.0 irrep towers.

For irrep labels a, b, c of one structure algebra, an intertwiner is a map
C: V_a (x) V_b -> V_c with C rho_{a(x)b}(X) = rho_c(X) C for every algebra basis
element X, where rho_{a(x)b}(X) = rho_a(X) (x) I + I (x) rho_b(X) (the Leibniz action
on the tensor product). The solution space is computed NUMERICALLY: accumulate the
Gram matrix of the stacked Sylvester operators over the basis and take its null space
(eigh), so no symbol tables and no per-family sign conventions exist. Each null vector
is one independent intertwiner (multiplicity slot), orthonormal in Frobenius norm, and
is verified by an equivariance-residual assert (raise, not warn) before caching.

Row-major vec convention (torch.reshape): vec(C rho) = kron(I_dc, rho^T) vec(C) and
vec(rho_c C) = kron(rho_c, I_D) vec(C).

Multiplicity counting rests on the spectral gap of the Gram matrix (measured at 13+
orders of magnitude in practice); the absolute `atol` cut is required rather than a
relative one because all-trivial triples such as l0 (x) l0 -> l0 have an identically
zero Gram, making a relative threshold undefined. The equivariance assert catches
spurious slots (false positives from numerical drift) but is structurally blind to
dropped slots (false negatives) — the gap is what protects those. Within a multiplicity
slot (n_mult > 1) the eigh-defined basis is arbitrary up to an orthogonal rotation but
is stable for the process lifetime, so any persisted downstream weights depending on
this basis are tied to a fixed torch build's eigh ordering.

`cg_intertwiners` returns a clone of the cached tensor on every call (both cache-hit
and first-build paths). This protects the process-global `_CG_CACHE` from in-place
caller mutations.
"""

import warnings
from typing import Dict, List, Tuple

import torch

from vfe3.geometry.generators import generate_son, generate_sp
from vfe3.geometry.irreps import irrep_dim, irrep_generators

_CG_CACHE: Dict[Tuple[str, int, str, str, str, float], torch.Tensor] = {}


def clear_cg_cache() -> None:
    """Drop every cached intertwiner (process-global; entries are otherwise kept for the
    process lifetime, bounded only by the distinct (algebra, N, triple, atol) keys built)."""
    _CG_CACHE.clear()


def _defining(N: int, algebra: str) -> torch.Tensor:
    if algebra == "so":
        return generate_son(N, dtype=torch.float64)
    if algebra == "sp":
        return generate_sp(N, dtype=torch.float64)
    raise ValueError(f"unknown algebra {algebra!r}; registered: 'so', 'sp'")


def cg_intertwiners(
    N:       int,                          # defining-rep dimension (N of SO(N); 2m of Sp(2m))

    *,
    algebra: str,                          # 'so' | 'sp'
    label_a: str,                          # first source irrep label
    label_b: str,                          # second source irrep label
    label_c: str,                          # target irrep label
    atol:    float = 1e-8,
) -> torch.Tensor:                         # (n_mult, d_c, d_a * d_b) float64; n_mult may be 0
    """All independent intertwiners V_a (x) V_b -> V_c (empty leading axis if none)."""
    # atol is part of the key: it sets the null-space cut, so two calls with different
    # tolerances are different solves and must not alias (audit 2026-06-09 overnight CR1).
    key = (algebra, N, label_a, label_b, label_c, float(atol))
    if key in _CG_CACHE:
        return _CG_CACHE[key].clone()
    da = irrep_dim(N, algebra=algebra, label=label_a)
    db = irrep_dim(N, algebra=algebra, label=label_b)
    dc = irrep_dim(N, algebra=algebra, label=label_c)
    D = da * db
    if dc * D > 5000:
        raise ValueError(
            f"CG solve for ({label_a}, {label_b}) -> {label_c} over R^{N} exceeds the supported "
            f"construction size (d_c * d_a * d_b = {dc * D} > 5000); larger products await a "
            f"matrix-free solver."
        )
    G_def = _defining(N, algebra)
    ra = irrep_generators(G_def, algebra=algebra, label=label_a)
    rb = irrep_generators(G_def, algebra=algebra, label=label_b)
    rc = irrep_generators(G_def, algebra=algebra, label=label_c)
    I_a = torch.eye(da, dtype=torch.float64)
    I_b = torch.eye(db, dtype=torch.float64)
    I_c = torch.eye(dc, dtype=torch.float64)
    I_D = torch.eye(D, dtype=torch.float64)
    gram = torch.zeros(dc * D, dc * D, dtype=torch.float64)
    rho_ab = []
    for a in range(G_def.shape[0]):
        r = torch.kron(ra[a], I_b) + torch.kron(I_a, rb[a])        # (D, D) Leibniz action
        rho_ab.append(r)
        op = torch.kron(I_c, r.T.contiguous()) - torch.kron(rc[a], I_D)
        gram += op.T @ op
    evals, evecs = torch.linalg.eigh(gram)
    null_mask = evals < atol
    # Runtime spectral-gap monitor (audit 2026-06-09 overnight F12): the multiplicity
    # count is trustworthy only while the Gram spectrum splits cleanly across the atol
    # cut (13+ orders of magnitude in practice). The equivariance assert below catches
    # SPURIOUS slots but is structurally blind to DROPPED ones, so warn when a nonzero
    # eigenvalue sits within two decades ABOVE the cut (a possible dropped slot). The
    # lower side is left to the assert: at tight atol the eigh's machine-eps null
    # eigenvalues legitimately approach the cut from below. The all-zero Gram
    # (all-trivial triples) and the empty null space skip — no cut is straddled there.
    if bool(null_mask.any()) and bool((~null_mask).any()):
        gap_lo = float(evals[null_mask].max())         # largest "zero" eigenvalue
        gap_hi = float(evals[~null_mask].min())        # smallest nonzero eigenvalue
        if gap_hi < 100.0 * atol:
            warnings.warn(
                f"CG multiplicity cut for ({label_a}, {label_b}) -> {label_c} over R^{N} "
                f"is thin: Gram eigenvalues straddle atol={atol:.1e} with largest-null "
                f"{gap_lo:.3e} / smallest-nonnull {gap_hi:.3e}. The multiplicity count "
                f"may be wrong (a DROPPED slot is not caught by the equivariance "
                f"assert); adjust atol to restore a clean spectral gap.",
                stacklevel=2,
            )
    null = evecs[:, null_mask]                                     # (dc*D, n_mult), orthonormal
    C = null.T.reshape(-1, dc, D).contiguous()
    # The verify gate scales with the null-space cut: a looser atol legitimately admits
    # vectors with proportionally larger residual (residual^2 ~ eigenvalue), so a fixed
    # 1e-7 would spuriously reject them (audit 2026-06-09 overnight PP2). At the default
    # atol=1e-8 the gate stays exactly 1e-7. Note the gate (and the float64 construction)
    # bounds the BUILD residual; the fp32 runtime cast adds its own ~1e-7-scale epsilon.
    verify_gate = max(1e-7, 10.0 * float(atol))
    for a in range(G_def.shape[0]):                                # build-time verification
        res = (C @ rho_ab[a] - torch.einsum("ij,mjk->mik", rc[a], C)).abs().max() \
            if C.shape[0] else torch.tensor(0.0)
        if float(res) > verify_gate:
            raise RuntimeError(
                f"CG intertwiner ({label_a}, {label_b}) -> {label_c} equivariance residual "
                f"{float(res):.3e} exceeds {verify_gate:.1e} at generator {a}"
            )
    _CG_CACHE[key] = C
    return C.clone()


def cg_selection(
    N:       int,                          # defining-rep dimension

    *,
    algebra: str,                          # 'so' | 'sp'
    labels:  List[str],                    # the spec's irrep labels (duplicates allowed)
    atol:    float = 1e-8,                  # null-space cut; MUST match the buffer build's atol
) -> List[Tuple[str, str, str, int]]:      # admissible (a, b, c, n_mult), a <= b, n_mult > 0
    """Enumerate admissible CG triples among the spec's labels (unordered source pairs:
    swapped duplicates are not independent bilinear maps, so a <= b canonically).

    ``atol`` is threaded into the per-triple ``cg_intertwiners`` null-space solve so the
    enumerated ``n_mult`` uses the SAME tolerance as a later buffer build (audit 2026-06-13 L20):
    a mismatch at a thin Gram gap would otherwise disagree on the multiplicity / buffer leading dim.
    """
    uniq = sorted(set(labels))
    out: List[Tuple[str, str, str, int]] = []
    for i, a in enumerate(uniq):
        for b in uniq[i:]:
            for c in uniq:
                # Pre-check the construction-size guard via the closed-form dims so one
                # oversize triple skips (with a warning) instead of raising mid-enumeration
                # and blocking an otherwise-valid tower (audit 2026-06-09 overnight F25).
                da = irrep_dim(N, algebra=algebra, label=a)
                db = irrep_dim(N, algebra=algebra, label=b)
                dc = irrep_dim(N, algebra=algebra, label=c)
                if dc * da * db > 5000:
                    warnings.warn(
                        f"cg_selection skipping ({a}, {b}) -> {c} over R^{N}: solve size "
                        f"d_c*d_a*d_b = {dc * da * db} > 5000 exceeds the supported "
                        f"construction guard, so this coupling path is OMITTED from the "
                        f"tower (larger products await a matrix-free solver).",
                        stacklevel=2,
                    )
                    continue
                n = cg_intertwiners(N, algebra=algebra, label_a=a, label_b=b,
                                    label_c=c, atol=atol).shape[0]
                if n > 0:
                    out.append((a, b, c, n))
    return out
