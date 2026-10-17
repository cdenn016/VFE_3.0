r"""Lie-algebra primitives for VFE_3.0 gauge frames (Gaussian-specialized).

The gauge frame phi lives in a Lie algebra g (a vector space) as coordinates in a
generator basis {G_a}: the algebra element is embed(phi) = sum_a phi^a G_a. The
group element U = exp(embed(phi)) lies in GL+(K) (det>0) or SO(N). This module
supplies the coordinate<->matrix maps, the Lie bracket, the compact equal-block group
element, and determinant control. Pure: operates on a generator TENSOR, not a GaugeGroup.

The model consumes only :class:`CompactBlockElement` and :func:`project_phi_to_slk` (and the
per-block trace functionals behind it). The coordinate maps :func:`embed_phi`, :func:`extract_phi`,
:func:`gram_pinv` and :func:`lie_bracket_coords` have no production caller and are kept as test and
diagnostic utilities. The positional frame composes as the exact group product
:math:`U_i = \exp(X_i)\exp(Y_n)`, the ``right_phi`` factor of ``build_factored_transport``, with no
truncation error; the BCH composition registry and its bracket-closure guard that once approximated
that product were removed (audit 2026-10-16, F91).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:                       # the annotation of project_phi_to_slk only (audit 2026-10-16 F97)
    from vfe3.geometry.groups import GaugeGroup


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
    generators: 'torch.Tensor | GaugeGroup',  # (n_gen, K, K) basis, or the group (its block_trace_vectors)
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
        V = (generators.block_trace_vectors() if hasattr(generators, "block_trace_vectors")
             else _block_trace_vectors(generators, irrep_dims)).to(device=phi_c.device)   # (H, n_gen)
        gram_pinv = torch.linalg.pinv(V @ V.transpose(-1, -2))   # (H, H); diag -> 1/||V_h||^2
        s = phi_c @ V.transpose(-1, -2)                       # (..., H)
        coeffs = s @ gram_pinv                                # (..., H) joint solve, not per-block
        out = phi_c - torch.einsum("...h,hg->...g", coeffs, V)
    return out.to(phi.dtype)
