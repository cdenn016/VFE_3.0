r"""Lie-algebra retraction primitives for VFE_3.0 gauge frames (Gaussian-specialized).

The gauge frame phi lives in a Lie algebra g (a vector space) as coordinates in a
generator basis {G_a}: the algebra element is embed(phi) = sum_a phi^a G_a. The
group element U = exp(embed(phi)) lies in GL+(K) (det>0) or SO(N). This module
supplies: coordinate<->matrix maps, the Lie bracket, a composition registry
(euclidean step or BCH chart correction), the GL(K)/SO(N) retractions, and
determinant control. Pure: operates on a generator TENSOR, not a GaugeGroup.
"""

import hashlib
import math
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch


def embed_phi(
    phi:        torch.Tensor,             # (..., n_gen) Lie-algebra coordinates
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> torch.Tensor:                        # (..., K, K) matrix sum_a phi^a G_a
    r"""Coordinates -> algebra element: embed(phi) = sum_a phi^a G_a."""
    return torch.einsum("...a,aij->...ij", phi, generators)


def gram_pinv(
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    rcond:      float = 1e-10,
) -> torch.Tensor:                        # (n_gen, n_gen) pseudo-inverse of Gram
    r"""Pseudo-inverse of the Frobenius Gram matrix Gram_ab = <G_a, G_b>_F.

    pinv (not inv) so overcomplete / rank-deficient spanning sets (e.g. the
    trace-projected sl(K) spanning set) are handled. The Gram
    and its pinv are formed in float64 because the overcomplete spanning set has
    a true nullspace whose float32 eigenvalue is O(1e-7) -- larger than rcond, so
    a float32 pinv would treat the null direction as invertible and inject noise.
    """
    gram = torch.einsum("aij,bij->ab", generators.double(), generators.double())
    return torch.linalg.pinv(gram, rcond=rcond).to(generators.dtype)


def extract_phi(
    matrix:     torch.Tensor,             # (..., K, K) element of span{G_a}
    generators: torch.Tensor,             # (n_gen, K, K) basis

    *,
    gram_pinv_: Optional[torch.Tensor] = None,   # cached gram_pinv(generators)
) -> torch.Tensor:                        # (..., n_gen) min-norm coordinates
    r"""Algebra element -> coordinates by least squares against the Gram matrix.

    Solves Gram c = g with g_b = <G_b, matrix>_F, c = Gram^+ g (min-norm solution
    when the basis is overcomplete). For an orthonormal basis Gram = I and
    c_a = <G_a, matrix>_F.
    """
    gp = gram_pinv(generators) if gram_pinv_ is None else gram_pinv_
    g = torch.einsum("aij,...ij->...a", generators, matrix)
    return torch.einsum("...a,ab->...b", g, gp)


def lie_bracket_matrix(
    A: torch.Tensor,                      # (..., K, K)
    B: torch.Tensor,                      # (..., K, K)
) -> torch.Tensor:                        # (..., K, K) [A,B] = AB - BA
    r"""Matrix commutator [A, B] = AB - BA (sign convention AB - BA)."""
    return A @ B - B @ A


def lie_bracket_coords(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    gram_pinv_: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # (..., n_gen) coords of [embed phi1, embed phi2]
    r"""Bracket in coordinates: extract([embed(phi1), embed(phi2)])."""
    A = embed_phi(phi1, generators)
    B = embed_phi(phi2, generators)
    return extract_phi(lie_bracket_matrix(A, B), generators, gram_pinv_=gram_pinv_)


_COMPOSE: Dict[str, Callable[..., torch.Tensor]] = {}


def register_compose(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a composition rule phi1,phi2 -> composed coords.

    Duplicate keys fail closed (audit 2026-07-01 F12): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _COMPOSE and not override:
            raise KeyError(f"composition rule {name!r} already registered; pass override=True to replace")
        _COMPOSE[name] = fn
        return fn
    return _wrap


def get_compose(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered composition rule (KeyError if absent)."""
    if name not in _COMPOSE:
        raise KeyError(f"no composition rule {name!r}; available: {sorted(_COMPOSE)}")
    return _COMPOSE[name]


@register_compose("euclidean")
def compose_euclidean(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K) (unused; kept for a uniform seam)

    *,
    order:          int                    = 0,
    compact_blocks: bool                   = False,  # unused; uniform seam with compose_bch
    gram_pinv_:     Optional[torch.Tensor] = None,
    block_dims:     Optional[List[int]]    = None,   # unused; uniform seam with compose_bch
) -> torch.Tensor:
    r"""Plain Lie-algebra step phi1 + phi2 (exact iff [phi1, phi2] = 0).

    The manuscript working/default update: g is a vector space, so the tangent
    step is the sum of coordinates (GL(K)_supplementary.tex ll. 550-557).
    """
    return phi1 + phi2


# Bracket closure depends only on basis values. Cache scalar diagnostics by a stable CPU
# signature so equal-value copies share work and no caller tensor (especially CUDA storage)
# remains owned by the cache. All LRU tables are bounded because diagnostics may see many
# transient bases in sweeps and tests.
_BRACKET_CLOSURE_CACHE_MAXSIZE: int = 32
_BRACKET_CLOSURE_IDENTITIES: OrderedDict = OrderedDict()  # identity/version -> (weak ref, value signature)
_BRACKET_CLOSURE_RES: OrderedDict = OrderedDict()       # value signature -> max residual
_BRACKET_CLOSURE_WARNED: OrderedDict = OrderedDict()    # (value signature, where) -> None


def _basis_value_signature(
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> tuple:
    """Stable ``(shape, dtype, CPU value hash)`` signature without retaining the basis."""
    basis_cpu = generators.detach().contiguous().cpu()
    value_hash = hashlib.sha256(basis_cpu.view(torch.uint8).numpy().tobytes()).digest()
    return tuple(generators.shape), generators.dtype, value_hash


def _bounded_lru_store(
    cache: OrderedDict,
    key:   tuple,
    value: object,
) -> None:
    """Insert one diagnostic entry and evict the least-recently-used overflow."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _BRACKET_CLOSURE_CACHE_MAXSIZE:
        cache.popitem(last=False)


def _cached_basis_value_signature(
    generators: torch.Tensor,             # (n_gen, K, K) basis
) -> tuple:
    """Resolve the value signature through a weak, mutation-aware identity cache."""
    identity_key = (
        id(generators),
        generators._version,
        tuple(generators.shape),
        generators.dtype,
        generators.device,
    )
    cached = _BRACKET_CLOSURE_IDENTITIES.get(identity_key)
    if cached is not None:
        basis_ref, value_signature = cached
        if basis_ref() is generators:
            _BRACKET_CLOSURE_IDENTITIES.move_to_end(identity_key)
            return value_signature
        del _BRACKET_CLOSURE_IDENTITIES[identity_key]

    value_signature = _basis_value_signature(generators)

    def _drop_dead_identity(basis_ref: weakref.ReferenceType) -> None:
        current = _BRACKET_CLOSURE_IDENTITIES.get(identity_key)
        if current is not None and current[0] is basis_ref:
            del _BRACKET_CLOSURE_IDENTITIES[identity_key]

    _bounded_lru_store(
        _BRACKET_CLOSURE_IDENTITIES,
        identity_key,
        (weakref.ref(generators, _drop_dead_identity), value_signature),
    )
    return value_signature


def warn_if_basis_not_closed(
    generators:  torch.Tensor,            # (n_gen, K, K) fixed gauge generator basis

    *,
    where:        str,                     # call-site label (also the per-site warn-once key)
    closure_tol:  float                  = 1e-4,
    eps:          float                  = 1e-12,
    max_elements: int                    = 50_000_000,  # skip the O(n_gen^2 K^2) scan above this
    gram_pinv_:   Optional[torch.Tensor] = None,
) -> None:
    r"""Warn once (per call site) if ``generators`` is not closed under the Lie bracket.

    Measures the max relative out-of-span residual
    :math:`\max_{a,b} \lVert [G_a,G_b] - \mathrm{embed}(\mathrm{extract}([G_a,G_b]))\rVert_F /
    (\lVert [G_a,G_b]\rVert_F + \epsilon)`. On a closed (default direct-sum) basis this is ~0 and
    nothing is warned; on a non-closed custom basis it is O(1) and the span projection in BCH
    composition / structure-constants silently truncates the out-of-span part.
    The result depends only on generator values, so a bounded value-signature cache avoids repeating
    the bracket scan without retaining the caller's tensor storage.

    The bracket scan materializes an ``(n_gen, n_gen, K, K)`` tensor; for large bases (e.g.
    block_glk at K=140: n_gen=2800 -> ~1.5e11 elements) this would OOM. Such large bases here are the
    direct-sum block groups, which are bracket-closed BY CONSTRUCTION, so when ``n_gen^2 * K^2``
    exceeds ``max_elements`` the scan is skipped and the basis is treated as closed (cached, no
    warning).

    LIMITATION (audit 2026-06-13 L5): the size-skip ASSUMES a too-large basis is closed-by-
    construction, which holds for the shipped direct-sum groups (block_glk, so_n) but is NOT verified
    for a large custom registered basis; there the scan is silently skipped.
    """
    key = _cached_basis_value_signature(generators)
    max_res = _BRACKET_CLOSURE_RES.get(key)
    if max_res is None:
        n_gen, K = generators.shape[0], generators.shape[-1]
        if n_gen * n_gen * K * K > max_elements:                  # too large to scan; closed by construction
            _bounded_lru_store(_BRACKET_CLOSURE_RES, key, 0.0)
            return
        with torch.no_grad():
            G    = generators                                                   # (n_gen, K, K)
            brak = torch.einsum("aij,bjk->abik", G, G) - torch.einsum("bij,ajk->abik", G, G)  # [G_a,G_b]
            proj = embed_phi(extract_phi(brak, G, gram_pinv_=gram_pinv_), G)     # span projection of each bracket
            res  = (brak - proj).norm(dim=(-2, -1))                             # (n_gen, n_gen)
            den  = brak.norm(dim=(-2, -1)) + eps                                # (n_gen, n_gen)
            max_res = float((res / den).max())
        _bounded_lru_store(_BRACKET_CLOSURE_RES, key, max_res)
    else:
        _BRACKET_CLOSURE_RES.move_to_end(key)
    wkey = (key, where)
    if max_res > closure_tol and wkey not in _BRACKET_CLOSURE_WARNED:
        _bounded_lru_store(_BRACKET_CLOSURE_WARNED, wkey, None)
        warnings.warn(
            f"{where}: gauge generator basis is not closed under the Lie bracket (max relative "
            f"out-of-span residual {max_res:.3e} > {closure_tol:.1e}); BCH composition / structure "
            f"constants truncate the out-of-span [G_a,G_b] terms. Build the group with "
            f"close_basis=True for cross-coupled (3+-head chain) bases.",
            UserWarning,
            stacklevel=3,
        )


def _bch_dynkin_correction(
    X:     torch.Tensor,                  # (..., d, d) embedded left element
    Y:     torch.Tensor,                  # (..., d, d) embedded right element

    order: int,                           # Dynkin truncation order (>= 1)
) -> torch.Tensor:                        # (..., d, d) Z - (X + Y), the commutator series
    r"""The commutator part of the symmetric Dynkin series (shapes are whatever the
    caller passes -- full (K, K) matrices or per-block (H, d, d) stacks; the series is
    pure matmul algebra either way)."""
    br = lie_bracket_matrix
    XY = br(X, Y)
    C = 0.5 * XY
    if order >= 2:
        C = C + (1.0 / 12.0) * (br(X, XY) - br(Y, XY))
    if order >= 3:
        C = C - (1.0 / 24.0) * br(Y, br(X, XY))
    if order >= 4:
        YX  = br(Y, X)
        YYX = br(Y, YX); YYYX = br(Y, YYX)
        XXY = br(X, XY); XXXY = br(X, XXY)
        C = C - (1.0 / 720.0) * (br(Y, YYYX) + br(X, XXXY))
        C = C + (1.0 / 360.0) * (br(X, YYYX) + br(Y, XXXY))
        C = C + (1.0 / 120.0) * (br(Y, br(X, br(Y, XY))) + br(X, br(Y, br(X, YX))))
    return C


def _equal_diag_blocks(
    matrix: torch.Tensor,                 # (..., K, K) block-diagonal, H equal blocks
    n_blocks: int,
    block_dim: int,
) -> torch.Tensor:                        # (..., H, d, d) the diagonal blocks
    r"""Gather the H equal diagonal blocks (the same view trick as _blockwise_matrix_exp)."""
    m5 = matrix.reshape(*matrix.shape[:-2], n_blocks, block_dim, n_blocks, block_dim)
    return torch.diagonal(m5, dim1=-4, dim2=-2).movedim(-1, -3).contiguous()


def _from_equal_diag_blocks(
    blocks: torch.Tensor,                 # (..., H, d, d) per-block matrices
    K:      int,                          # full dimension H * d
) -> torch.Tensor:                        # (..., K, K) block-diagonal embedding (zeros off-block)
    r"""Scatter per-block matrices back onto the block diagonal of a zero (K, K) matrix."""
    H, d = blocks.shape[-3], blocks.shape[-1]
    out = torch.zeros(*blocks.shape[:-3], K, K, dtype=blocks.dtype, device=blocks.device)
    o5 = out.reshape(*out.shape[:-2], H, d, H, d)
    torch.diagonal(o5, dim1=-4, dim2=-2).copy_(blocks.movedim(-3, -1))
    return out


def _fail_if_bch_residual_exceeds(
    phi1:       torch.Tensor,
    phi2:       torch.Tensor,
    composed:   torch.Tensor,
    generators: torch.Tensor,

    *,
    eps:          float,
    residual_max: Optional[float] = None,
) -> None:
    r"""Fail closed when finite BCH coordinates miss the exact group product beyond a bound."""
    if residual_max is None:
        return
    with torch.no_grad(), torch.amp.autocast(phi1.device.type, enabled=False):
        generators64 = generators.detach().double()
        left64 = embed_phi(phi1.detach().double(), generators64)
        right64 = embed_phi(phi2.detach().double(), generators64)
        composed64 = embed_phi(composed.detach().double(), generators64)
        exact = torch.linalg.matrix_exp(left64) @ torch.linalg.matrix_exp(right64)
        approximate = torch.linalg.matrix_exp(composed64)
        residual = torch.linalg.matrix_norm(approximate - exact)
        reference = torch.linalg.matrix_norm(exact).clamp(min=eps)
        relative = residual / reference
        finite = torch.stack((
            torch.isfinite(exact).all(),
            torch.isfinite(approximate).all(),
            torch.isfinite(residual).all(),
            torch.isfinite(relative).all(),
        )).all()
        if not bool(finite):
            raise ValueError(
                "BCH residual validity bound encountered a nonfinite exact product, "
                "approximation, or residual"
            )
        if bool((relative > residual_max).any()):
            observed = float(relative.max())
            raise ValueError(
                "BCH residual validity bound exceeded: "
                f"relative residual {observed:.6g} > {residual_max:.6g}"
            )


@dataclass
class CompactBlockElement:
    r"""A compact equal-block group element without a dense :math:`K\times K` allocation.

    ``blocks`` stores ``(..., H, d, d)`` independent blocks, or ``(..., d, d)`` when ``tied=True``.
    The represented element has ``K = H d`` and exactly zero off-block entries. Structural slicing,
    reshaping, expansion, cloning, and detaching preserve the compact representation;
    callers that genuinely require a dense compatibility tensor must request :meth:`to_dense`
    explicitly.
    """

    blocks: torch.Tensor
    K:      int
    tied:   bool = False

    def __post_init__(self) -> None:
        if type(self.K) is not int or self.K <= 0:
            raise ValueError(f"K must be a positive int, got {type(self.K).__name__}: {self.K!r}")
        if type(self.tied) is not bool:
            raise ValueError(
                f"tied must be a bool, got {type(self.tied).__name__}: {self.tied!r}")
        min_rank = 2 if self.tied else 3
        if self.blocks.dim() < min_rank or self.blocks.shape[-1] != self.blocks.shape[-2]:
            layout = "(..., d, d)" if self.tied else "(..., H, d, d)"
            raise ValueError(
                f"CompactBlockElement blocks must have square layout {layout}, got "
                f"shape {tuple(self.blocks.shape)}")
        d = self.blocks.shape[-1]
        if self.K % d != 0:
            raise ValueError(f"K={self.K} must be divisible by block dimension d={d}")
        if not self.tied and self.blocks.shape[-3] * d != self.K:
            raise ValueError(
                f"untied blocks have H={self.blocks.shape[-3]} and d={d}, but H*d != K={self.K}")

    @property
    def block_dim(self) -> int:
        return self.blocks.shape[-1]

    @property
    def n_blocks(self) -> int:
        return self.K // self.block_dim

    @property
    def shape(self) -> torch.Size:
        leading = self.blocks.shape[:-2] if self.tied else self.blocks.shape[:-3]
        return torch.Size((*leading, self.K, self.K))

    @property
    def device(self) -> torch.device:
        return self.blocks.device

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks.dtype

    @property
    def requires_grad(self) -> bool:
        return self.blocks.requires_grad

    def dim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def expanded_blocks(self) -> torch.Tensor:
        r"""Return ``(..., H, d, d)`` blocks, broadcasting the tied block without densifying."""
        if not self.tied:
            return self.blocks
        return self.blocks.unsqueeze(-3).expand(
            *self.blocks.shape[:-2], self.n_blocks, self.block_dim, self.block_dim)

    def to_dense(self) -> torch.Tensor:
        r"""Explicit compatibility conversion to the represented dense ``(..., K, K)`` element."""
        return _from_equal_diag_blocks(self.expanded_blocks(), self.K)

    def clone(self) -> 'CompactBlockElement':
        return CompactBlockElement(self.blocks.clone(), self.K, self.tied)

    def detach(self) -> 'CompactBlockElement':
        return CompactBlockElement(self.blocks.detach(), self.K, self.tied)

    def _storage_index(self, index: object) -> tuple:
        r"""Translate logical indexing while leaving the represented matrix axes intact."""
        logical_rank = self.dim()
        n_leading = logical_rank - 2
        raw = index if isinstance(index, tuple) else (index,)
        if any(isinstance(item, torch.Tensor) and item.dtype == torch.bool for item in raw):
            raise ValueError(
                "CompactBlockElement does not support boolean advanced indexing because a mask "
                "may consume multiple logical axes; index .blocks at an internal compact call "
                "site or call to_dense() for logical tensor indexing")
        if sum(item is Ellipsis for item in raw) > 1:
            raise IndexError("an index can only have a single ellipsis")
        consumed = sum(item is not None and item is not Ellipsis for item in raw)
        if consumed > logical_rank:
            raise IndexError(
                f"too many indices for CompactBlockElement of dimension {logical_rank}")
        missing = logical_rank - consumed
        expanded = []
        for item in raw:
            if item is Ellipsis:
                expanded.extend([slice(None)] * missing)
            else:
                expanded.append(item)
        if not any(item is Ellipsis for item in raw):
            expanded.extend([slice(None)] * missing)

        storage_index = []
        logical_axis = 0
        for item in expanded:
            if item is None:
                if logical_axis > n_leading:
                    raise ValueError(
                        "CompactBlockElement indexing may only insert leading axes; call "
                        "to_dense() before inserting a matrix axis")
                storage_index.append(item)
                continue
            if logical_axis < n_leading:
                storage_index.append(item)
            elif not (isinstance(item, slice)
                      and item.start is None and item.stop is None and item.step is None):
                raise ValueError(
                    "CompactBlockElement indexing must leave both logical matrix axes intact; "
                    "call to_dense() before slicing a matrix axis")
            logical_axis += 1
        return tuple(storage_index)

    def __getitem__(self, index: object) -> 'CompactBlockElement':
        return CompactBlockElement(self.blocks[self._storage_index(index)], self.K, self.tied)

    def __setitem__(self, index: object, value: 'CompactBlockElement') -> None:
        if not isinstance(value, CompactBlockElement):
            raise TypeError(
                "compact element assignment requires CompactBlockElement; call to_dense() for "
                "dense compatibility operations")
        if value.K != self.K or value.tied != self.tied:
            raise ValueError("compact element assignment requires matching K and tied layout")
        self.blocks[self._storage_index(index)] = value.blocks

    def unsqueeze(self, dim: int) -> 'CompactBlockElement':
        n_leading = self.dim() - 2
        normalized = dim if dim >= 0 else dim + self.dim() + 1
        if normalized < 0 or normalized > n_leading:
            raise ValueError(
                "CompactBlockElement.unsqueeze may only add a leading batch/token axis; call "
                "to_dense() before inserting a matrix axis")
        return CompactBlockElement(self.blocks.unsqueeze(normalized), self.K, self.tied)

    def reshape(self, *shape: int) -> 'CompactBlockElement':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        if len(shape) < 2 or tuple(shape[-2:]) != (self.K, self.K):
            raise ValueError(
                f"compact reshape must preserve trailing dense shape ({self.K}, {self.K}), got {shape}")
        suffix = (self.block_dim, self.block_dim)
        if not self.tied:
            suffix = (self.n_blocks, *suffix)
        return CompactBlockElement(self.blocks.reshape(*shape[:-2], *suffix), self.K, self.tied)

    def expand(self, *shape: int) -> 'CompactBlockElement':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = tuple(shape[0])
        if len(shape) < 2 or tuple(shape[-2:]) != (self.K, self.K):
            raise ValueError(
                f"compact expand must preserve trailing dense shape ({self.K}, {self.K}), got {shape}")
        suffix = (self.block_dim, self.block_dim)
        if not self.tied:
            suffix = (self.n_blocks, *suffix)
        return CompactBlockElement(self.blocks.expand(*shape[:-2], *suffix), self.K, self.tied)

@register_compose("bch")
def compose_bch(
    phi1:        torch.Tensor,            # (..., n_gen)
    phi2:        torch.Tensor,            # (..., n_gen)
    generators:  torch.Tensor,            # (n_gen, K, K)

    *,
    closure_tol:    float                  = 1e-4,
    eps:            float                  = 1e-12,
    order:          int                    = 4,
    compact_blocks: bool                   = False,  # canonical block_glk: omit dense X/Y/Z embedding
    _autocast_guarded: bool                = False,
    gram_pinv_:     Optional[torch.Tensor] = None,
    block_dims:     Optional[List[int]]    = None,   # group irrep_dims; >1 equal blocks -> blocked brackets
    residual_max:   Optional[float]        = None,   # opt-in fail-closed group-product residual bound
) -> torch.Tensor:
    r"""BCH chart correction: coords of log(exp(embed phi1) exp(embed phi2)).

    Symmetric Dynkin series (matrix space, extracted once). Terms by `order`:
      order>=1: + 1/2 [X,Y]
      order>=2: + 1/12 ([X,[X,Y]] - [Y,[X,Y]])
      order>=3: - 1/24 [Y,[X,[X,Y]]]
      order>=4: - 1/720 ([Y,[Y,[Y,[Y,X]]]] + [X,[X,[X,[X,Y]]]])
                + 1/360 ([X,[Y,[Y,[Y,X]]]] + [Y,[X,[X,[X,Y]]]])
                + 1/120 ([Y,[X,[Y,[X,Y]]]] + [X,[Y,[X,[Y,X]]]])
    Truncation error is O(||X||^{order+2} + ||Y||^{order+2}).

    MEMORY (vram audit 2026-06-10). Two structural costs of the naive cascade dominated the
    training-step VRAM: (a) when one operand is unbatched (the (N, n_gen) positional element
    composed into a (B, N, n_gen) frame) every bracket matmul broadcast-materialized its own
    full-batch copy of the SAME embedded Y -- 22 saved (B, N, K, K) copies at order 4; the
    operands are now broadcast to a common shape ONCE so all brackets share one storage
    (bit-identical values). (b) ``block_dims`` (the group's irrep_dims, passed by callers that
    hold the group): when the embedding is block-diagonal with >1 EQUAL blocks -- the same
    invariant the per-block matrix_exp relies on; every multi-entry ``irrep_dims`` basis
    (block_glk / so_n towers) embeds block-diagonally for EVERY phi --
    each commutator is block-diagonal, so the whole Dynkin cascade runs on the (..., H, d, d)
    diagonal-block stacks and is scattered back once. Identical values (the off-block entries
    are exactly zero throughout), 1/H the bracket memory and H^2 fewer matmul FLOPs. ``None``
    (single block / cross-coupled / unequal towers) keeps the dense cascade unchanged.

    On a basis NOT closed under the Lie bracket the Dynkin commutator terms push
    :math:`Z` out of :math:`\mathrm{span}\{G_a\}`, and the final
    :math:`\mathrm{extract\_phi}(Z)` least-squares projection silently discards that
    out-of-span component. A diagnostic guard measures the max relative residual
    :math:`\max \lVert Z - \mathrm{embed}(\mathrm{extract}(Z))\rVert_F /
    (\lVert Z\rVert_F + \epsilon)` and warns once if it exceeds ``closure_tol``; on a
    closed (default direct-sum) basis the residual is ~0 so the guard is silent and the
    returned phi is unchanged.
    """
    if residual_max is not None and (
        not math.isfinite(residual_max) or residual_max <= 0.0
    ):
        raise ValueError(
            f"residual_max must be None or finite and positive, got {residual_max!r}"
        )

    if (not _autocast_guarded
            and phi1.dtype == torch.float32
            and phi2.dtype == torch.float32
            and torch.is_autocast_enabled(phi1.device.type)):
        with torch.amp.autocast(phi1.device.type, enabled=False):
            return compose_bch(
                phi1,
                phi2,
                generators,
                closure_tol=closure_tol,
                eps=eps,
                order=order,
                compact_blocks=compact_blocks,
                _autocast_guarded=True,
                gram_pinv_=gram_pinv_,
                block_dims=block_dims,
                residual_max=residual_max,
            )

    blocked = (
        order >= 1
        and block_dims is not None
        and len(block_dims) > 1
        and len(set(block_dims)) == 1
    )
    if (compact_blocks
            and blocked
            and generators.shape[0] == len(block_dims) * block_dims[0] * block_dims[0]):
        H, d = len(block_dims), block_dims[0]
        Xb = phi1.reshape(*phi1.shape[:-1], H, d, d)
        Yb = phi2.reshape(*phi2.shape[:-1], H, d, d)
        if Xb.shape != Yb.shape:
            Xb, Yb = torch.broadcast_tensors(Xb, Yb)
            Xb = Xb.contiguous()
            Yb = Yb.contiguous()
        Zb = Xb + Yb + _bch_dynkin_correction(Xb, Yb, order)
        phi = Zb.reshape(*Zb.shape[:-3], generators.shape[0])
        # The explicit block_head_row_major capability is set only by the canonical direct-sum
        # gl(d)^H builder, whose elementary matrix basis is bracket-closed by construction.
        # Skipping the generic closure scan here also avoids its dense Gram projection.
        _fail_if_bch_residual_exceeds(
            phi1, phi2, phi, generators, residual_max=residual_max, eps=eps,
        )
        return phi

    X = embed_phi(phi1, generators)
    Y = embed_phi(phi2, generators)
    if X.shape != Y.shape:
        # Broadcast ONCE so every bracket matmul saves the same storage instead of each
        # materializing its own full-batch copy of the unbatched operand. contiguous() is a
        # no-op on the side that already carries the batch.
        X, Y = torch.broadcast_tensors(X, Y)
        X = X.contiguous()
        Y = Y.contiguous()
    Z = X + Y
    if blocked:
        H, d = len(block_dims), block_dims[0]
        Xb = _equal_diag_blocks(X, H, d)
        Yb = _equal_diag_blocks(Y, H, d)
        Z = Z + _from_equal_diag_blocks(_bch_dynkin_correction(Xb, Yb, order), X.shape[-1])
    elif order >= 1:
        Z = Z + _bch_dynkin_correction(X, Y, order)
    phi = extract_phi(Z, generators, gram_pinv_=gram_pinv_)

    # Diagnostic (cached, phi-independent): warn once if the generator basis is not bracket-closed,
    # in which case the Dynkin commutators leave span{G_a} and the extract_phi projection above drops
    # that part. Closure depends only on the fixed generators and never touches Z's autograd graph;
    # the bounded cache owns only a CPU value signature and scalar residual, not the caller's basis.
    # The scan is size-gated (max_elements): large direct-sum bases (e.g. block_glk K=140, the prior
    # O(K^4) OOM) are skipped as closed-by-construction.
    warn_if_basis_not_closed(generators, where="compose_bch",
                             closure_tol=closure_tol, eps=eps, gram_pinv_=gram_pinv_)
    _fail_if_bch_residual_exceeds(
        phi1, phi2, phi, generators, residual_max=residual_max, eps=eps,
    )
    return phi


def compose_phi(
    phi1:       torch.Tensor,             # (..., n_gen)
    phi2:       torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    order:          int                    = 4,
    mode:           str                    = "euclidean",
    compact_blocks: bool                   = False,  # canonical block_glk packed-coordinate BCH
    gram_pinv_:     Optional[torch.Tensor] = None,
    block_dims:     Optional[List[int]]    = None,   # group irrep_dims (compose_bch blocked brackets)
    residual_max:   Optional[float]        = None,   # opt-in BCH-to-group-product residual gate
) -> torch.Tensor:
    r"""Dispatch to the registered composition rule `mode`."""
    compose = get_compose(mode)
    kwargs = {
        "order": order,
        "gram_pinv_": gram_pinv_,
        "block_dims": block_dims,
    }
    if residual_max is not None and mode == "bch":
        kwargs["residual_max"] = residual_max
    if compact_blocks:
        kwargs["compact_blocks"] = True
    return compose(phi1, phi2, generators, **kwargs)


def _retract_core(
    phi:          torch.Tensor,           # (..., n_gen) current frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step direction

    *,
    step_size:      float                  = 1.0,
    eps:            float                  = 1e-6,
    order:          int                    = 4,
    mode:           str                    = "euclidean",
    compact_blocks: bool                   = False,
    trust_region:   Optional[float]        = 0.1,   # None / <=0 disables the trust-region clamp
    max_norm:       Optional[float]        = 5.0,   # None / <=0 disables the max-norm clamp
    generators:     Optional[torch.Tensor] = None,
    gram_pinv_:     Optional[torch.Tensor] = None,
    block_dims:     Optional[List[int]]    = None,
    residual_max:   Optional[float]        = None,  # opt-in BCH-to-group-product residual gate
) -> torch.Tensor:
    r"""Shared retraction: scale -> trust-region clamp -> compose -> step-bounding norm cap.

      update   = clamp_||.|| ( step_size * delta_phi , trust_region )
      phi_new  = compose(phi, update; mode, order)
      c        = max( max_norm , ||phi|| )                    (per row)
      phi_new <- clamp_||.|| ( phi_new , c )
    Trust region and cap are applied to the coordinate-vector norm.

    The cap bounds the STEP, not the base point (audit 2026-09-23 #7). With the former cap
    ``c = max_norm``, a frame already outside the ball was rescaled onto it at ANY step size, so
    :math:`R_\phi(0) \ne \phi` and the retraction axiom (Absil, Mahony and Sepulchre 2008,
    Def. 4.1.1) failed. With ``c = max(max_norm, ||phi||)`` a zero step returns ``phi`` (exactly
    for the euclidean and compact-block BCH compositions; to the ``extract_phi`` projection
    roundoff for the dense BCH composition), rows inside the ball keep ``dR(0) = id``, and a row
    outside the ball can move tangentially or inward but never outward. The retractions have no
    caller that relies on the former point projection: the only production caller is the E-step
    phi retraction, and the public ``retract_glk``/``retract_son`` contracts (a step from inside the
    ball saturates at ``max_norm``) are unchanged.
    """
    update = step_size * delta_phi
    if trust_region is not None and trust_region > 0:
        u_norm = update.norm(dim=-1, keepdim=True)
        update = update * (trust_region / (u_norm + eps)).clamp(max=1.0)
    if compact_blocks:
        phi_new = compose_phi(
            phi,
            update,
            generators,
            order=order,
            mode=mode,
            gram_pinv_=gram_pinv_,
            block_dims=block_dims,
            compact_blocks=True,
            residual_max=residual_max,
        )
    else:
        phi_new = compose_phi(phi, update, generators, order=order, mode=mode,
                              gram_pinv_=gram_pinv_, residual_max=residual_max)
    if max_norm is not None and max_norm > 0:
        n_norm = phi_new.norm(dim=-1, keepdim=True)
        cap = phi.norm(dim=-1, keepdim=True).clamp(min=max_norm)          # max(max_norm, ||phi||)
        outside = n_norm > cap
        # The denominator is n_norm > cap > 0 on the rescaled rows and cap on the others, so both
        # torch.where branches stay finite (no eps, whose bias would move a zero-step exterior row).
        phi_new = torch.where(outside, phi_new * (cap / torch.where(outside, n_norm, cap)), phi_new)
    return phi_new


def retract_glk(
    phi:          torch.Tensor,           # (..., n_gen) current GL(K) frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step

    generators:   torch.Tensor,           # (n_gen, K, K)

    *,
    step_size:      float                  = 1.0,
    trust_region:   float                  = 0.1,    # tighter than SO(N): GL(K) is non-compact
    max_norm:       float                  = 10.0,   # bounds singular values to ~[e^-5, e^5]
    eps:            float                  = 1e-6,
    order:          int                    = 4,
    mode:           str                    = "euclidean",
    compact_blocks: bool                   = False,
    gram_pinv_:     Optional[torch.Tensor] = None,
    block_dims:     Optional[List[int]]    = None,
    residual_max:   Optional[float]        = None,   # opt-in BCH-to-group-product residual gate
) -> torch.Tensor:
    r"""GL(K) retraction (no det control here; the dispatcher applies it)."""
    return _retract_core(
        phi, delta_phi, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        compact_blocks=compact_blocks, generators=generators, gram_pinv_=gram_pinv_,
        block_dims=block_dims, residual_max=residual_max,
    )


def retract_son(
    phi:          torch.Tensor,           # (..., n_gen) current SO(N) frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step

    generators:   torch.Tensor,           # (n_gen, K, K) skew so(N) basis

    *,
    step_size:    float = 1.0,
    trust_region: float = 0.3,            # compact group
    max_norm:     float = math.pi,        # bounds principal angles
    eps:          float = 1e-6,
    order:        int   = 4,
    mode:         str   = "euclidean",
    gram_pinv_:   Optional[torch.Tensor] = None,
    residual_max: Optional[float]        = None,   # opt-in BCH-to-group-product residual gate
) -> torch.Tensor:
    r"""SO(N) retraction. det(exp) = 1 automatic (skew generators)."""
    return _retract_core(
        phi, delta_phi, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        generators=generators, gram_pinv_=gram_pinv_, residual_max=residual_max,
    )


def _block_trace_vectors(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    eps:        float = 1e-12,
) -> torch.Tensor:                        # (n_blocks, n_gen) V[h,a] = tr(G_a|block h)
    r"""Per-block trace functionals V[h,a] = tr(G_a restricted to block h)."""
    rows, start = [], 0
    for d in irrep_dims:
        end = start + d
        rows.append(generators[:, start:end, start:end].diagonal(dim1=-2, dim2=-1).sum(-1))
        start = end
    return torch.stack(rows, dim=0)                       # (n_blocks, n_gen)


def project_phi_to_slk(
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K
) -> torch.Tensor:                        # (..., n_gen) per-block trace-free
    r"""Hard projection to sl(K) per block: remove the trace component so

        det(Omega_h) = exp(tr(embed(phi)|block h)) = 1.

    Orthogonal projection of phi off the span of the per-block trace functionals V_h:
    ``phi <- phi - (phi V^T) (V V^T)^+ V``. The JOINT Gram solve ``(V V^T)^+`` is required:
    for a tied basis (one generator acting in every block, kron(I_n, E_ij)) the V_h coincide, so the
    per-block-independent form ``s / ||V_h||^2`` would over-subtract by a factor of n_heads. For an
    untied gauge the V_h have disjoint support, ``V V^T`` is diagonal, and the pseudo-inverse reduces
    to ``1/||V_h||^2`` (so the untied result is unchanged); a fully traceless basis (so_n) gives
    ``V = 0`` and the projection is a no-op.

    Runs in an ``autocast(enabled=False)`` island: under AMP the ``V V^T`` matmul would be downcast
    and ``linalg.pinv`` has no bf16/fp16 kernel. It is also a correctness island -- ``det(Omega_h) = 1``
    is an exact algebraic constraint, and a bf16 trace solve would leave an O(1e-2) relative residual.
    The island is a no-op without AMP (fp32 in, fp32 out), so the default path is byte-identical.
    """
    with torch.amp.autocast(phi.device.type, enabled=False):
        # bf16/fp16 -> fp32 (neither pinv nor the mixed-dtype matmul below has a reduced-precision
        # kernel); fp32/fp64 are returned unchanged by promote_types, so nothing is ever downcast.
        phi_c = phi.to(torch.promote_types(phi.dtype, torch.float32))
        V = _block_trace_vectors(generators, irrep_dims)      # (H, n_gen)
        gram_pinv = torch.linalg.pinv(V @ V.transpose(-1, -2))   # (H, H); diag -> 1/||V_h||^2
        s = phi_c @ V.transpose(-1, -2)                       # (..., H)
        coeffs = s @ gram_pinv                                # (..., H) joint solve, not per-block
        out = phi_c - torch.einsum("...h,hg->...g", coeffs, V)
    return out.to(phi.dtype)


def clamp_phi_trace(
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    trace_max:  float = 5.0,              # soft cap T on |tr(embed(phi)|block h)|
) -> torch.Tensor:                        # (..., n_gen) with |s_h| <= T
    r"""Soft per-block trace clamp: rescale only the trace component so |s_h| <= T,

    bounding log|det(Omega_h)|. Off-trace (sl(K)) directions are untouched. Realizes the clamped
    trace via the JOINT Gram solve ``delta = (s_clamped - s) (V V^T)^+`` so the tied gauge (coinciding
    V_h) is corrected once, not n_heads times; reduces to ``(s_clamped - s)/||V_h||^2`` for an
    orthogonal (untied) basis.

    Runs in the same ``autocast(enabled=False)`` fp32 island as ``project_phi_to_slk`` (``linalg.pinv``
    has no bf16/fp16 kernel, and the bound |s_h| <= T must hold in the precision it is asserted in).
    The island is a no-op without AMP, so the default path is byte-identical.
    """
    with torch.amp.autocast(phi.device.type, enabled=False):
        phi_c = phi.to(torch.promote_types(phi.dtype, torch.float32))   # bf16/fp16 -> fp32
        V = _block_trace_vectors(generators, irrep_dims)      # (H, n_gen)
        gram_pinv = torch.linalg.pinv(V @ V.transpose(-1, -2))   # (H, H); diag -> 1/||V_h||^2
        s = phi_c @ V.transpose(-1, -2)                       # (..., H)
        s_clamped = s.clamp(min=-trace_max, max=trace_max)
        delta = (s_clamped - s) @ gram_pinv                   # (..., H) joint solve, not per-block
        out = phi_c + torch.einsum("...h,hg->...g", delta, V)
    return out.to(phi.dtype)
