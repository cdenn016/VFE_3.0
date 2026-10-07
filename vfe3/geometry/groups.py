r"""Gauge-group registry for VFE_3.0 (structure-group axis of geometry).

A GaugeGroup bundles the Lie-algebra generators with the metadata transport
needs (block/irrep structure, skew flag) and declares the families whose
divergence is invariant under its representation (admissibility). Groups are
config-selected by name so variants swap without editing call sites.

Admissibility: a (family, group) pair is valid iff the family's divergence is
invariant under common pushforward by the group's representation,
D(rho(g) q || rho(g) p) = D(q || p). For the Gaussian family with the GL(K)
congruence action (mu -> g mu, Sigma -> g Sigma g^T) this holds for every
g in G <= GL(K), so every group here is admissible for "gaussian_full" -- the
one registered family that keeps the whole pushed covariance. A diagonal family
re-diagonalizes g Sigma g^T and is admissible only for the diagonal-scaling
subgroup; check_admissible measures which is which, family by family.
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.generators import (
    generate_block_local, generate_glk_multihead, generate_son, local_basis_co, local_basis_glc,
    local_basis_so,
)
from vfe3.geometry.irreps import DEFAULT_IRREP_BASIS, direct_sum_generators


@dataclass(frozen=True)
class _ElementaryBlockBasis:
    """Immutable metadata for the built-in uncoupled row-major elementary basis."""

    K:       int
    n_heads: int
    dtype:   torch.dtype
    device:  torch.device

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.n_heads * (self.K // self.n_heads) ** 2, self.K, self.K)


class _GeneratorTensor:
    """Keep public Tensor access lazy; callers receive the same mutable Tensor thereafter."""

    def __get__(self, instance: Optional["GaugeGroup"], owner: Optional[type] = None) -> torch.Tensor:
        if instance is None:
            raise AttributeError("generators is a required constructor argument")
        source = instance._generator_source
        if isinstance(source, _ElementaryBlockBasis):
            source = generate_glk_multihead(
                source.K, source.n_heads, dtype=source.dtype, device=source.device)
            instance._generator_source = source
            instance._elementary_certificate = instance._generator_cache_key()
        elif instance._elementary_layout is not None and instance.elementary_block_shape is None:
            instance.phi_coordinate_layout = None
            instance.frobenius_gram_diagonal = False
            instance.frobenius_gram_uniform = None
        return source

    def __set__(self, instance: "GaugeGroup", value: 'torch.Tensor | _ElementaryBlockBasis') -> None:
        previous = getattr(instance, "_generator_source", None)
        if isinstance(previous, _ElementaryBlockBasis) or getattr(instance, "_elementary_certificate", None) is not None:
            # Replacement withdraws the built-in certificate, including same-shape replacements.
            instance.frobenius_gram_diagonal = False
            instance.frobenius_gram_uniform = None
            instance.phi_coordinate_layout = None
        instance._generator_source = value
        instance._elementary_certificate = None
        instance._elementary_layout = (
            (value.n_heads, value.K // value.n_heads)
            if isinstance(value, _ElementaryBlockBasis) else None)


@dataclass
class GaugeGroup:
    """A structure group plus the metadata the transport layer consumes."""

    name:                 str
    generators:           'torch.Tensor | _ElementaryBlockBasis' = _GeneratorTensor()  # constructor union; descriptor reads Tensor
    irrep_dims:           List[int]                          # block sizes; sum == K
    # NOT a transpose-for-inverse fast path: no such path exists (audit 2026-07-25 F21). Every
    # inverse routes through _checked_group_inverse's float64 solve. What this flag actually gates:
    # retraction defaults. (The transport chart radius applies to skew groups as well.)
    skew_symmetric:       bool                               # algebra is skew (exp lands in O(K))
    pullback_capability:  Optional[str]          = None      # certified strict M-step route, if any
    invariant_families:   Tuple[str, ...]        = ()        # registry-declared exact families; empty fails closed
    irrep_labels:         Optional[List[str]]    = None      # per-block label ('l1', 'sym2', ...); None = label-less
    algebra:              Optional[str]          = None      # irrep-registry algebra key ('so'); None = label-less
    phi_coordinate_layout: Optional[str]         = None      # explicit packed-coordinate capability; None = unknown
    frobenius_gram_diagonal: bool                = False     # certified <G_a,G_b>_F=0 for a != b
    frobenius_gram_uniform: Optional[float]      = None      # known common diagonal weight; None = nonuniform/unknown
    basis_fingerprint:    Optional[str]          = None      # sha256 of a rebuilt-at-load float64 basis; None = deterministic
    # Frame tying across heads: 'per_head' (an independent element per head block) or 'shared'
    # (one element acting on every block); None = a group without head structure or tying axis.
    tying:                Optional[str]          = None
    # The (m, d, d) local algebra basis of a block-local group (copied into every head block);
    # None for every other group.
    block_local_basis:    Optional[torch.Tensor] = field(default=None, repr=False, compare=False)

    # Cached pseudo-inverse of the generator Frobenius Gram (computed once, off __init__/eq/repr).
    _gram_pinv_cache:     Optional[torch.Tensor] = field(default=None, init=False, repr=False, compare=False)
    _gram_pinv_key:       Optional[tuple]        = field(default=None, init=False, repr=False, compare=False)
    _gram_diagonal_cache: Optional[torch.Tensor] = field(default=None, init=False, repr=False, compare=False)
    _gram_diagonal_key:   Optional[tuple]        = field(default=None, init=False, repr=False, compare=False)
    _gram_uniform_cache:  Optional[float]        = field(default=None, init=False, repr=False, compare=False)
    _gram_uniform_key:    Optional[tuple]        = field(default=None, init=False, repr=False, compare=False)
    _gram_uniform_ready:  bool                   = field(default=False, init=False, repr=False, compare=False)

    @property
    def generator_shape(self) -> Tuple[int, int, int]:
        """Basis dimensions without requesting the public dense Tensor."""
        return tuple(self._generator_source.shape)

    @property
    def generator_device(self) -> torch.device:
        return self._generator_source.device

    @property
    def generator_dtype(self) -> torch.dtype:
        return self._generator_source.dtype

    def _generator_cache_key(self) -> tuple:
        """Metadata-only key that invalidates cached basis properties after mutation or replacement."""
        source = self._generator_source
        return (id(source), tuple(source.shape), source.device, source.dtype,
                source._version if isinstance(source, torch.Tensor) else None)

    @property
    def elementary_block_shape(self) -> Optional[Tuple[int, int]]:
        """Certified (head count, block width), withdrawn after public basis mutation."""
        source = self._generator_source
        layout = self._elementary_layout
        if layout is None or self.irrep_dims != [layout[1]] * layout[0]:
            return None
        if isinstance(source, _ElementaryBlockBasis):
            return layout
        if not source.requires_grad and self._elementary_certificate == self._generator_cache_key():
            return layout
        return None

    def apply_generators(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Move basis metadata with the model, materializing only an already-dense source."""
        source = self._generator_source
        if isinstance(source, _ElementaryBlockBasis):
            moved = fn(torch.empty(0, device=source.device, dtype=source.dtype))
            self._generator_source = _ElementaryBlockBasis(
                source.K, source.n_heads, moved.dtype, moved.device)
        else:
            canonical = self.elementary_block_shape is not None
            self._generator_source = fn(source)
            if canonical:
                self._elementary_certificate = self._generator_cache_key()

    def embed_algebra(
        self,
        phi:   torch.Tensor,                         # (..., n_gen)
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:                               # (..., K, K)
        r"""Compute $M = \sum_a \phi_a G_a$ without retaining a dense elementary basis.

        Dense output and downstream reduction order are retained for diagnostic consumers.
        Differentiable dense embeddings and nonfinite inputs keep the original contraction,
        including zero-times-infinity behavior in both values and cotangents. The active
        compact transport embeds differentiable coordinates by its existing block reshape.
        """
        basis_dtype = self.generator_dtype if dtype is None else dtype
        layout = self.elementary_block_shape
        if (layout is None or phi.dtype != basis_dtype or phi.device != self.generator_device
                or (torch.is_grad_enabled() and phi.requires_grad)
                or not bool(torch.isfinite(phi).all())):
            basis = self.generators if dtype is None else self.generators.to(dtype=dtype)
            return torch.einsum("...a,aij->...ij", phi, basis)
        H, d = layout
        blocks = phi.reshape(*phi.shape[:-1], H, d, d)
        matrix = phi.new_zeros(*phi.shape[:-1], H * d, H * d)
        for h in range(H):
            matrix[..., h*d:(h+1)*d, h*d:(h+1)*d] = blocks[..., h, :, :]
        if phi.dtype != torch.float64 and torch.is_autocast_enabled(phi.device.type):
            matrix = matrix.to(torch.get_autocast_dtype(phi.device.type))
        return matrix

    def generator_traces(self) -> torch.Tensor:
        r"""Return $\operatorname{tr}(G_a)$ in coordinate order without a dense basis."""
        layout = self.elementary_block_shape
        if layout is not None:
            H, d = layout
            return torch.eye(d, device=self.generator_device, dtype=self.generator_dtype).expand(H, d, d).reshape(-1)
        return self.generators.diagonal(dim1=-2, dim2=-1).sum(-1)

    def gram_diagonal(self) -> Optional[torch.Tensor]:
        r"""Return cached weights ``||G_a||_F^2`` for a certified diagonal generator Gram."""
        if (not self.frobenius_gram_diagonal
                or (self._elementary_layout is not None and self.elementary_block_shape is None)):
            return None
        key = self._generator_cache_key()
        if self._gram_diagonal_cache is None or self._gram_diagonal_key != key:
            first_read = self._gram_diagonal_key is None
            if self.elementary_block_shape is not None or (first_read and self.frobenius_gram_uniform is not None):
                self._gram_diagonal_cache = torch.full(
                    (self.generator_shape[0],),
                    self.frobenius_gram_uniform,
                    device=self.generator_device,
                    dtype=self.generator_dtype,
                )
            else:
                self._gram_diagonal_cache = torch.einsum(
                    "aij,aij->a", self.generators, self.generators
                ).detach()
            self._gram_diagonal_key = key
            if self.elementary_block_shape is not None or (first_read and self.frobenius_gram_uniform is not None):
                self._gram_uniform_cache = self.frobenius_gram_uniform
                self._gram_uniform_key = key
                self._gram_uniform_ready = True
            else:
                self._gram_uniform_cache = None
                self._gram_uniform_key = None
                self._gram_uniform_ready = False
        return self._gram_diagonal_cache

    def gram_diagonal_uniform(self) -> Optional[float]:
        r"""Return the common diagonal weight when all certified weights are exactly equal."""
        diagonal = self.gram_diagonal()
        if diagonal is None or diagonal.numel() == 0:
            return None
        if self.frobenius_gram_uniform is None:
            return None
        key = self._generator_cache_key()
        if not self._gram_uniform_ready or self._gram_uniform_key != key:
            self._gram_uniform_cache = (
                float(diagonal[0])
                if bool(torch.all(diagonal == diagonal[0]))
                else None
            )
            self._gram_uniform_key = key
            self._gram_uniform_ready = True
        return self._gram_uniform_cache

    def phi_norm_route(self) -> str:
        """Name the exact embedded-norm implementation enabled by this basis certificate."""
        return ("diagonal_gram" if self.frobenius_gram_diagonal
                and not (self._elementary_layout is not None and self.elementary_block_shape is None)
                else "dense_fallback")

    def gram_pinv(self) -> torch.Tensor:
        r"""Cached ``gram_pinv(generators)`` -- the Gram pseudo-inverse ``extract_phi`` projects with.

        ``extract_phi`` recovers algebra coordinates by solving ``Gram c = g`` with
        ``c = Gram^+ g``; ``Gram_ab = <G_a, G_b>_F`` depends ONLY on the fixed generator basis.
        The BCH positional-composition hot path (``compose_bch -> extract_phi``) recomputed a dense
        ``(n_gen, n_gen)`` float64 ``pinv`` every forward (for the orthonormal block_glk basis the
        Gram is exactly ``I``, so the recompute was pure waste). Caching is value-identical -- the
        returned tensor equals ``gram_pinv(self.generators)`` -- and recomputed after device/dtype
        moves, replacement, or an in-place basis mutation.
        """
        if (self._gram_pinv_cache is None
                or self._gram_pinv_key != self._generator_cache_key()):
            from vfe3.geometry.lie_ops import gram_pinv as _gram_pinv
            with torch.no_grad():
                self._gram_pinv_cache = _gram_pinv(self.generators)
            self._gram_pinv_key = self._generator_cache_key()
        return self._gram_pinv_cache

    def __post_init__(self) -> None:
        K = self.generator_shape[-1]
        if sum(self.irrep_dims) != K:
            raise ValueError(
                f"sum(irrep_dims)={sum(self.irrep_dims)} must equal K={K}; "
                f"irrep_dims={self.irrep_dims}"
            )
        if self.irrep_labels is not None and len(self.irrep_labels) != len(self.irrep_dims):
            raise ValueError(
                f"irrep_labels has {len(self.irrep_labels)} entries but there are "
                f"{len(self.irrep_dims)} irrep blocks"
            )

    def invariant_for(self, family: str) -> bool:
        """Whether the divergence of ``family`` is invariant under this group.

        The cheap declaration lookup; :func:`check_admissible` is its executable certificate, and
        ``tests/test_admissibility_verifier.py`` pins the two against each other for every registered
        (group, family) pair, so the declaration cannot drift from the measured behavior.
        """
        return family in self.invariant_families


_GROUPS: Dict[str, Callable[..., GaugeGroup]] = {}


@dataclass(frozen=True)
class GroupTraits:
    r"""Registration-time facts about a group, readable without building it (:func:`group_traits`).

    ``algebra`` names the irrep registry of an irrep-tower group (``"so"`` for ``so_n``): heads are
    labeled irrep blocks from ``group_n``/``irrep_spec``, possibly of different sizes. ``None`` means
    heads are the ``n_heads`` equal blocks of ``embed_dim``. ``tyings`` are the supported
    ``gauge_tying`` values and ``native_tying`` the one an unset field resolves to.
    ``contains_pair_rotations`` says whether every adjacent-pair rotation of a head block lies in
    the group, which is what keeps ``pos_rotation='rope'`` inside it. ``min_head_dim`` is the
    smallest head block the builder accepts, and every head block must be a multiple of
    ``head_dim_multiple``.
    """

    algebra:                 Optional[str]   = None
    tyings:                  Tuple[str, ...] = ("per_head",)
    native_tying:            str             = "per_head"
    contains_pair_rotations: bool            = True
    min_head_dim:            int             = 1
    head_dim_multiple:       int             = 1
# Frame tying across heads (config field gauge_tying): one element per head block, or one shared.
GAUGE_TYINGS: Tuple[str, ...] = ("per_head", "shared")
# Registered family names whose stored-parameter divergence is exactly GL(K)-congruence invariant.
# The legacy "gaussian" key was DEAD DATA (audit 2026-07-26 E-05): no such family is registered, so it
# could never match `cfg.family` at the one production consumer (`run_artifacts._pure_path_report`)
# and only ever masked the fact that the declaration was untested against the registry. The full
# Gaussian keeps the whole congruence; every diagonal family re-diagonalizes `g Sigma g^T` and is
# admissible only for the diagonal-scaling subgroup. `check_admissible` verifies this list executably,
# and `test_admissibility_verifier` pins declaration against verifier for every registered pair.
_GAUSSIAN_INVARIANT_FAMILIES = ("gaussian_full",)


def register_group(
    name: str,

    *,
    override:             bool                   = False,
    invariant_families:   Tuple[str, ...]        = (),

    pullback_capability:  Optional[str]          = None,
    pullback_tying:       Optional[str]          = None,
    traits:               GroupTraits            = GroupTraits(),
) -> Callable:
    """Decorator registering a GaugeGroup builder under ``name``.

    ``pullback_capability`` names
    a certified strict M-step implementation; omission fails closed. With ``pullback_tying`` set the
    capability is certified only for groups built with that frame tying, and other tyings carry
    None. ``traits`` (:class:`GroupTraits`) is attached to the builder for config validation.
    ``invariant_families`` declares
    exact divergence invariance. The wrapper writes every declaration onto itself and each returned
    :class:`GaugeGroup`, so an override replaces the callable and metadata atomically.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an existing
    name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., GaugeGroup]) -> Callable[..., GaugeGroup]:
        if name in _GROUPS and not override:
            raise KeyError(f"gauge group {name!r} already registered; pass override=True to replace")
        if traits.native_tying not in traits.tyings or not set(traits.tyings) <= set(GAUGE_TYINGS):
            raise ValueError(f"gauge group {name!r}: native_tying {traits.native_tying!r} must be one of "
                             f"its tyings {traits.tyings}, a subset of {GAUGE_TYINGS}")
        families = tuple(invariant_families)

        @wraps(fn)
        def _coherent_builder(*args, **kwargs) -> GaugeGroup:
            group = fn(*args, **kwargs)
            if not isinstance(group, GaugeGroup):
                raise TypeError(
                    f"gauge group builder {name!r} returned {type(group).__name__}, expected GaugeGroup"
                )
            group.pullback_capability = (pullback_capability
                                         if pullback_tying is None or group.tying == pullback_tying
                                         else None)
            group.invariant_families = families
            return group

        setattr(_coherent_builder, "pullback_capability", pullback_capability)
        setattr(_coherent_builder, "pullback_tying", pullback_tying)
        setattr(_coherent_builder, "traits", traits)
        setattr(_coherent_builder, "invariant_families", families)
        _GROUPS[name] = _coherent_builder
        return _coherent_builder
    return _wrap


def group_traits(name: str) -> GroupTraits:
    r"""The :class:`GroupTraits` declared when ``name`` was registered (KeyError if absent)."""
    return getattr(get_group(name), "traits", GroupTraits())


def get_group(name: str) -> Callable[..., GaugeGroup]:
    """Return the registered GaugeGroup builder for ``name`` (KeyError if absent)."""
    if name not in _GROUPS:
        raise KeyError(
            f"no gauge group registered under {name!r}; available: {sorted(_GROUPS)}"
        )
    return _GROUPS[name]


@register_group(
    "block_glk",
    pullback_capability="pullback_per_block",
    invariant_families=_GAUSSIAN_INVARIANT_FAMILIES,
    traits=GroupTraits(tyings=GAUGE_TYINGS),
)
def _build_block_glk(
    K:               int,
    n_heads:         int,

    *,
    tying:           str                             = "per_head",
    dtype:           torch.dtype                     = torch.float32,
    device:          'torch.device | str | None'     = None,
) -> GaugeGroup:
    r"""Block-diagonal GL(K) = GL(d_head)^n_heads with the row-major elementary basis.

    The basis stays lazy (``_ElementaryBlockBasis``); the dense generators are built only when a
    consumer reads ``group.generators``.

    ``tying='shared'`` builds the diagonal subgroup :math:`\{I_H\otimes g : g\in GL(d)\}`, one
    GL(d_head) element acting identically on every head: generators :math:`I_H\otimes E_{ab}` in
    row-major (a, b) order, n_gen = d_head^2, Frobenius Gram :math:`H\,I`. No generator lives on
    one block, so the per-block metrics and the per-head compact fast paths do not apply; the
    group carries no ``phi_coordinate_layout`` and takes the dense transport route.
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")
    if tying not in GAUGE_TYINGS:
        raise ValueError(f"block_glk tying must be one of {GAUGE_TYINGS}, got {tying!r}")
    d_head = K // n_heads
    placement = torch.empty(0, device=device, dtype=dtype)
    if tying == "shared":
        elementary = torch.eye(d_head * d_head, dtype=placement.dtype).reshape(d_head * d_head, d_head, d_head)
        eye_h = torch.eye(n_heads, dtype=placement.dtype)
        generators = torch.stack([torch.kron(eye_h, E) for E in elementary]).to(placement.device)
        return GaugeGroup(
            name="block_glk",
            generators=generators,                                  # (d^2, K, K) I_H (x) E_ab
            irrep_dims=[d_head] * n_heads,
            skew_symmetric=False,
            frobenius_gram_diagonal=True,
            frobenius_gram_uniform=float(n_heads),
            tying="shared",
        )
    return GaugeGroup(
        name="block_glk",
        generators=_ElementaryBlockBasis(K, n_heads, placement.dtype, placement.device),
        irrep_dims=[d_head] * n_heads,
        skew_symmetric=False,
        phi_coordinate_layout="block_head_row_major" if n_heads > 1 else None,
        frobenius_gram_diagonal=True,
        frobenius_gram_uniform=1.0,
        tying="per_head",
    )


def declared_invariant_families(group_name: str) -> Tuple[str, ...]:
    r"""The ``invariant_families`` declaration of the group registered under ``group_name``.

    Reads the declaration WITHOUT building the group (the builder carries it as an attribute), so a
    run-artifact purity report can consult it without materializing generators. The single production
    reader of the declaration; :meth:`GaugeGroup.invariant_for` is the same lookup on a built group.
    """
    return tuple(getattr(get_group(group_name), "invariant_families", ()))


def check_admissible(
    group:      GaugeGroup,
    family:     str   = "gaussian_full",

    *,
    functional: str   = "renyi",
    alpha:      float = 1.0,
    n_samples:  int   = 8,
    batch:      int   = 5,
    scale:      float = 0.2,
    atol:       float = 1e-3,
    rtol:       float = 1e-3,
    seed:       int   = 0,
) -> bool:
    r"""Executably verify that ``functional`` is invariant under ``group``'s representation on ``family``.

    Draws ``n_samples`` random group elements ``g = exp(sum_a c_a G_a)`` (coefficients ~ ``scale`` *
    N(0,1)) and a random Gaussian belief PAIR, pushes the pair forward by the family's representation,
    and asserts ``D(rho(g) q || rho(g) p) == D(q || p)`` to tolerance, where ``D`` is the registered
    divergence ``functional`` (renyi/squared_hellinger). Returns ``True`` iff invariant for every
    sample, else ``False`` -- so it turns ``GaugeGroup.invariant_for``'s string declaration into a
    verified invariant and catches a wrongly-declared ``invariant_families``.

    The congruence itself is family-INDEPENDENT; what varies is how the family reads the pushed
    covariance back into its own dispersion, so the pushforward dispatches through the family
    registry's ``from_covariance`` hook (audit 2026-07-26 E-05). The FULL Gaussian keeps
    ``g Sigma g^T`` intact and its divergence is invariant for every ``g in GL(K)``. A DIAGONAL family
    re-diagonalizes it, which is NOT invariant under a non-diagonal ``g`` (the verifier returns
    ``False``, correctly) -- those families are admissible only for the diagonal-scaling subgroup. A
    family with no full-covariance readout raises ``NotImplementedError`` (the extension point).

    Note the SCOPE this measures: the family's stored-parameter divergence under a global gauge, not
    the gauge equivariance of its coupling energy.

    USAGE (audit 2026-06-13 L6): this is a VERIFICATION TOOL, exercised by the admissibility tests
    (test_admissibility_verifier.py), not a runtime guard wired into model/config construction --
    the per-build random-sample congruence draws would tax every model build for a contract that
    holds for all shipped (group, family) pairs. Run it in tests/CI when adding a group or declaring
    a new ``invariant_families`` entry; ``invariant_for`` is the cheap string-membership check used at
    runtime.
    """
    from vfe3.families.base import get_family, get_functional     # local import: avoid an import cycle

    fam = get_family(family)
    fn = get_functional(functional)
    G = group.generators
    K = sum(group.irrep_dims)
    dev, dt = G.device, G.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    eye = torch.eye(K, device=dev, dtype=dt)

    def _div(mu_q, S_q, mu_p, S_p):
        # The family's own readout of the full second moment; kl_max high so no clamp masks invariance.
        return fn(fam.from_covariance(mu_q, S_q), fam.from_covariance(mu_p, S_p),
                  alpha=alpha, kl_max=1e12)

    for _ in range(n_samples):
        coeff = scale * torch.randn(G.shape[0], generator=gen, device=dev, dtype=dt)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, G))
        mu_q = torch.randn(batch, K, generator=gen, device=dev, dtype=dt)
        mu_p = torch.randn(batch, K, generator=gen, device=dev, dtype=dt)
        Aq = torch.randn(batch, K, K, generator=gen, device=dev, dtype=dt)
        Ap = torch.randn(batch, K, K, generator=gen, device=dev, dtype=dt)
        S_q = Aq @ Aq.transpose(-1, -2) + eye                    # random SPD
        S_p = Ap @ Ap.transpose(-1, -2) + eye
        base = _div(mu_q, S_q, mu_p, S_p)
        mu_q2 = torch.einsum("kl,nl->nk", g, mu_q)
        mu_p2 = torch.einsum("kl,nl->nk", g, mu_p)
        S_q2 = g @ S_q @ g.transpose(-1, -2)                     # congruence Sigma -> g Sigma g^T
        S_p2 = g @ S_p @ g.transpose(-1, -2)
        moved = _div(mu_q2, S_q2, mu_p2, S_p2)
        if not torch.allclose(base, moved, atol=atol, rtol=rtol):
            return False
    return True


@register_group(
    "so_n",
    pullback_capability="pullback_per_block",
    invariant_families=_GAUSSIAN_INVARIANT_FAMILIES,
    # RoPE's coordinate-pair rotation is generally not in the irrep image rho(SO(N)).
    traits=GroupTraits(algebra="so", tyings=GAUGE_TYINGS, native_tying="shared",
                       contains_pair_rotations=False),
)
def _build_so_n(
    K:          int,

    *,
    group_n:    Optional[int]                   = None,
    irrep_spec: Optional[List[Tuple[str, int]]] = None,
    irrep_basis: str                            = DEFAULT_IRREP_BASIS,
    tying:      str                             = "shared",
    dtype:      torch.dtype                     = torch.float32,
    device:     'torch.device | str | None'     = None,
) -> GaugeGroup:
    r"""SO(N) acting on R^K through a DIRECT SUM of irreps (tied gauge by default; heads = irreps).

    The structure group is SO(N) with N = ``group_n`` DECOUPLED from K. ``irrep_spec`` lists
    (label, multiplicity) pairs; label 'l<p>' is the symmetric-traceless rank-p tensor irrep
    (for N = 3 the spin-p block, dim 2p + 1), laid out contiguously in spec order with
    sum(mult * dim) == K. One per-token phi coordinate vector (n_gen = N(N-1)/2) drives EVERY
    block through that block's irrep image -- the SAME group element in different irreps, a
    TIED gauge -- so the embedded generators are block-diagonal but do NOT partition per block
    (per-block phi preconditioners are undefined; config rejects them). All blocks are built
    in real orthonormal bases, so every embedded generator is exactly skew
    (skew_symmetric=True: exp(-M) = exp(M)^T transport fast path; det Omega = 1 structurally).
    'l0' (trivial) blocks carry ZERO generators: those coordinates are gauge-invariant by
    construction. Unequal block dims are supported end-to-end (per-block energy loop,
    per-head tau = kappa_h sqrt(d_h)); equal-dims specs (one label, mult > 1) additionally
    keep the fused factored-transport fast path.

    ``irrep_basis`` selects the block bases (:mod:`vfe3.geometry.irreps`): 'canonical' is
    deterministic and builds SO(3) spins of any order; 'eigh' is the original construction, kept
    for checkpoints trained in it. The basis is rebuilt at every load, so the builder records a
    sha256 ``basis_fingerprint`` of the float64 generators for the checkpoint loader to compare.
    The generator Frobenius Gram is certified diagonal with one common weight: in the orthogonal
    L_jk basis the trace form of any representation of the simple algebra so(N) (N >= 3) is a
    multiple of the Killing form, so <G_a, G_b>_F = (sum_b ||rho_b(L)||_F^2) delta_ab; the build
    verifies this numerically and leaves the certificate off if it fails.

    ``tying='per_head'`` (untied) gives every block its own SO(N) element: block b carries its own
    copy of the generators :math:`\rho_b(L_a)`, supported on that block alone, in block-major
    order, so n_gen = dim so(N) times the number of nontrivial blocks ('l0' blocks carry none).
    Each head's frame is then :math:`\rho_b(g_b)` for an independent :math:`g_b`. The Gram stays
    diagonal, with weight :math:`\|\rho_b(L)\|_F^2` on block b's coordinates, so it is uniform
    only when every block has the same spin.
    """
    if tying not in GAUGE_TYINGS:
        raise ValueError(f"so_n tying must be one of {GAUGE_TYINGS}, got {tying!r}")
    if group_n is None or irrep_spec is None:
        raise ValueError(
            "gauge_group='so_n' requires group_n (the N of SO(N)) and irrep_spec "
            "([(label, mult), ...], labels 'l<p>'); set both in the config"
        )
    G_def = generate_son(group_n, dtype=torch.float64)
    G, dims = direct_sum_generators(G_def, algebra="so", irrep_spec=irrep_spec, basis=irrep_basis)
    if sum(dims) != K:
        raise ValueError(
            f"irrep_spec blocks {dims} sum to {sum(dims)} != K={K} (group_n={group_n})"
        )
    labels = [lab for lab, mult in irrep_spec for _ in range(int(mult))]
    if tying == "per_head":
        untied, start = [], 0
        for d in dims:
            block = torch.zeros_like(G)
            block[:, start:start + d, start:start + d] = G[:, start:start + d, start:start + d]
            if bool(block.abs().amax() > 0):                          # 'l0' blocks carry no generators
                untied.append(block)
            start += d
        if not untied:
            raise ValueError("an untied so_n tower needs at least one nontrivial irrep block")
        G = torch.cat(untied)                                         # (n_blocks' * dim so(N), K, K)
    gram = torch.einsum("aij,bij->ab", G, G)                          # (n_gen, n_gen) float64
    diag = gram.diagonal()
    scale = max(float(diag.abs().max()), 1.0)
    diagonal = float((gram - torch.diag(diag)).abs().max()) <= 1e-12 * scale
    uniform = diagonal and float((diag - diag.mean()).abs().max()) <= 1e-12 * scale
    return GaugeGroup(
        name="so_n",
        generators=G.to(dtype).to(device),
        irrep_dims=dims,
        skew_symmetric=True,
        irrep_labels=labels,
        algebra="so",
        frobenius_gram_diagonal=diagonal,
        frobenius_gram_uniform=float(diag.mean()) if uniform else None,
        basis_fingerprint=generator_fingerprint(G),
        tying=tying,
    )


def _build_block_local(
    name:    str,
    K:       int,
    n_heads: int,
    local:   torch.Tensor,                   # (m, d, d) local algebra basis, Gram = gram_weight * I

    *,
    skew:    bool,                           # every local generator is skew (local group compact)
    gram_weight: float                   = 1.0,
    dtype:   torch.dtype                 = torch.float32,
    device:  'torch.device | str | None' = None,
) -> GaugeGroup:
    r"""A block-local group :math:`G_{\mathrm{loc}}^H`: an independent element of one local subgroup of
    GL(d) on each of the ``n_heads`` equal head blocks.

    The generators are ``local`` copied into every block in head-major order
    (:func:`vfe3.geometry.generators.generate_block_local`), built in float64 and cast. ``local``
    must have Frobenius Gram ``gram_weight`` times the identity (orthonormal by default), which the
    build verifies, so the generator Gram is ``gram_weight * I``.
    The group carries no ``phi_coordinate_layout`` and takes the dense transport route; registered
    block-local groups use :func:`_build_block_local` from their own builder.
    """
    local = local.to(torch.float64)
    m = local.shape[0]
    gram = torch.einsum("aij,bij->ab", local, local)
    if float((gram - gram_weight * torch.eye(m, dtype=torch.float64)).abs().max()) > 1e-12:
        raise ValueError(f"{name}: the local basis must have Frobenius Gram {gram_weight} * I (orthonormal at 1)")
    if skew and float((local + local.transpose(-1, -2)).abs().max()) > 0.0:
        raise ValueError(f"{name}: skew=True needs an exactly skew local basis")
    generators = generate_block_local(K, n_heads, local, dtype=torch.float64)
    return GaugeGroup(
        name=name,
        generators=generators.to(device=device, dtype=dtype),
        irrep_dims=[K // n_heads] * n_heads,
        skew_symmetric=skew,
        frobenius_gram_diagonal=True,
        frobenius_gram_uniform=float(gram_weight),
        tying="per_head",
        block_local_basis=local,
    )


@register_group(
    "block_so",
    pullback_capability="pullback_per_block",
    invariant_families=_GAUSSIAN_INVARIANT_FAMILIES,
    traits=GroupTraits(min_head_dim=2),
)
def _build_block_so(
    K:       int,
    n_heads: int,

    *,
    dtype:   torch.dtype                 = torch.float32,
    device:  'torch.device | str | None' = None,
) -> GaugeGroup:
    r"""Untied rotations SO(d)^H: an independent SO(d_head) element on each head block.

    Generators :math:`(E_{ij}-E_{ji})/\sqrt2` (i < j) per head, head-major, Frobenius-orthonormal,
    n_gen = H d(d-1)/2. Compact with det = 1 per head: the "rotations only" rung below
    ``block_co`` and ``block_glk``. The same group as untied ``so_n`` with ``[("l1", H)]``
    (``group_n = d_head``) in a basis scaled by :math:`1/\sqrt2`, with equal-head layout
    validation and RoPE inside the group (pair rotations lie in each head's SO(d)).
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")
    return _build_block_local("block_so", K, n_heads, local_basis_so(K // n_heads), skew=True,
                              dtype=dtype, device=device)


@register_group(
    "block_co",
    pullback_capability="pullback_per_block",
    invariant_families=_GAUSSIAN_INVARIANT_FAMILIES,
    traits=GroupTraits(),
)
def _build_block_co(
    K:       int,
    n_heads: int,

    *,
    dtype:   torch.dtype                 = torch.float32,
    device:  'torch.device | str | None' = None,
) -> GaugeGroup:
    r"""Conformal rotations CO(d)^H = (R+ x SO(d))^H: on each head a positive scale times a rotation.

    Generators: the ``block_so`` basis, then :math:`I_d/\sqrt{d}`, per head (head-major,
    Frobenius-orthonormal, n_gen = H (d(d-1)/2 + 1)). Every head frame is :math:`\lambda R` with
    :math:`\lambda>0`, :math:`R\in SO(d)`, so cond(U_h) = 1 although the group is non-compact; it
    keeps angles and eigenvalue ratios and adds one per-token, per-head gain. exp is surjective onto
    the identity component, which is the whole group.
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")
    return _build_block_local("block_co", K, n_heads, local_basis_co(K // n_heads), skew=False,
                              dtype=dtype, device=device)


@register_group(
    "block_glmc",
    pullback_capability="pullback_per_block",
    invariant_families=_GAUSSIAN_INVARIANT_FAMILIES,
    traits=GroupTraits(min_head_dim=2, head_dim_multiple=2),
)
def _build_block_glmc(
    K:       int,
    n_heads: int,

    *,
    dtype:   torch.dtype                 = torch.float32,
    device:  'torch.device | str | None' = None,
) -> GaugeGroup:
    r"""Complex frames GL(m, C)^H, realified: an independent GL(m, C) element per head, d_head = 2m.

    Complex coordinate k of a head is the real pair (2k, 2k + 1), with the complex structure
    :math:`J = I_m\otimes[[0,-1],[1,0]]` on each head block. Generators
    :math:`E_{ij}\otimes I_2` then :math:`E_{ij}\otimes J` per head (head-major; the coordinates
    are (Re, Im) of the complex algebra element), entries +-1, Frobenius Gram :math:`2I`, n_gen =
    2 H m^2: a real subgroup of ``block_glk`` at the same d_head with half its generators. The
    exponential is the real ``matrix_exp`` of the realified block, which equals the realification of
    the complex exponential. RoPE rotates exactly the pairs (2k, 2k + 1), so each RoPE block is
    :math:`\mathrm{diag}(e^{i\theta_k})\in U(1)^m\subset GL(m,\mathbb{C})` and stays inside the group.

    Scale: with +-1 entries a coordinate norm corresponds to a :math:`\sqrt2` larger matrix norm
    than in ``block_glk``, and the initial matrix energy at equal coordinate scale matches
    ``block_glk`` (half the coordinates, each of weight 2). The trace functionals of the realified
    blocks see only Re tr, so ``pos_phi_project_slk`` / the trace clamp remove the R+ scale and fix
    :math:`|\det_{\mathbb{C}}| = 1` while keeping the U(1) phase.
    """
    if K % n_heads != 0:
        raise ValueError(f"K={K} must be divisible by n_heads={n_heads}")
    if (K // n_heads) % 2 != 0:
        raise ValueError(f"block_glmc needs an even head size d_head = 2m; got d_head={K // n_heads}")
    return _build_block_local("block_glmc", K, n_heads, local_basis_glc(K // n_heads), skew=False,
                              gram_weight=2.0, dtype=dtype, device=device)


def generator_fingerprint(
    generators: torch.Tensor,              # (n_gen, K, K) float64 construction-precision basis
) -> str:
    """sha256 hex digest of a float64 generator basis (shape and bytes)."""
    import hashlib
    G = generators.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256(str(tuple(G.shape)).encode("utf-8"))
    digest.update(G.numpy().tobytes())
    return digest.hexdigest()
