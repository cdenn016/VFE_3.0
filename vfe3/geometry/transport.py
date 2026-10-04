r"""Gauge transport for VFE_3.0 (flat phi-cocycle, Gaussian / location-scale specific).

The flat transport parameterization:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
The projected scorers read the vertex frames U_i directly; retractions are a separate module.
Gauge-RoPE folds a positional rotation into the transport via :class:`RopeTransport`.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.runtime import ieee_fp32_matmul

if TYPE_CHECKING:
    from vfe3.config import VFE3Config


@dataclass
class FactoredTransport:
    r"""The flat phi-cocycle transport in FACTORED form: the per-token (B, N, K, K) vertex
    exponentials, NOT the dense (B, N, N, K, K) pairwise Omega.

    The dense Omega_ij = exp(phi_i) exp(-phi_j) is never materialized; consumers read the vertex
    factors (the realized frames) and :meth:`to_dense_omega` is the explicit compatibility
    boundary. ``irrep_dims`` records the gauge block sizes.
    """

    exp_phi:                 torch.Tensor  # (..., N, K, K) exp(phi_i . G)
    exp_neg_phi:             torch.Tensor  # (..., N, K, K) exp(-phi_j . G)
    irrep_dims:              List[int]     # equal block sizes; sum == K, len > 1
    same_frame_flat_cocycle: bool = False  # one vertex table supplies U_i and true inverse U_j^-1

    def __post_init__(self) -> None:
        if type(self.same_frame_flat_cocycle) is not bool:
            raise ValueError(
                "same_frame_flat_cocycle must be a bool, got "
                f"{type(self.same_frame_flat_cocycle).__name__}: "
                f"{self.same_frame_flat_cocycle!r}")

    def to_dense_omega(self) -> torch.Tensor:
        r"""Rebuild the dense Omega_ij = exp(phi_i) exp(-phi_j) (..., N, N, K, K).

        Byte-identical to ``compute_transport_operators``'s Omega einsum (same factors, same
        ``ikl,jlm->ijkm`` contraction); used to keep the FULL-covariance sandwich and any
        consumer that needs the explicit operator on the existing dense code path. Rank-agnostic
        via the leading ellipsis (an optional batch axis flows through; the unbatched call matches).
        """
        with torch.autocast(device_type=self.exp_phi.device.type, enabled=False):
            return torch.einsum("...ikl,...jlm->...ijkm", self.exp_phi, self.exp_neg_phi)


TransportDict = Dict[str, 'torch.Tensor | None']


@dataclass
class CompactFactoredTransport:
    r"""Flat equal-block transport whose vertex factors stay ``(..., N, H, d, d)``.

    ``exp_blocks`` and ``inv_blocks`` are the stored vertex elements and their true per-block
    inverses. :meth:`to_dense_omega` is the explicit compatibility boundary for legacy consumers
    that require a dense pairwise operator.
    """

    exp_blocks:              torch.Tensor  # (..., N, H, d, d) stored U_i blocks
    inv_blocks:              torch.Tensor  # (..., N, H, d, d) true U_j^{-1} blocks
    K:                       int
    same_frame_flat_cocycle: bool = False

    def __post_init__(self) -> None:
        compatible = (
            self.exp_blocks.dim() == self.inv_blocks.dim()
            and self.exp_blocks.shape[:-4] == self.inv_blocks.shape[:-4]
            and self.exp_blocks.shape[-3:] == self.inv_blocks.shape[-3:]
        )
        if not compatible:
            raise ValueError(
                "compact transport factors must share leading batch and trailing block shapes "
                "(their query/key token counts may differ), got "
                f"{tuple(self.exp_blocks.shape)} and {tuple(self.inv_blocks.shape)}")
        if self.exp_blocks.dim() < 4 or self.exp_blocks.shape[-1] != self.exp_blocks.shape[-2]:
            raise ValueError(
                "compact transport factors must have (..., N, H, d, d) square-block layout, got "
                f"{tuple(self.exp_blocks.shape)}")
        H, d = self.exp_blocks.shape[-3], self.exp_blocks.shape[-1]
        if type(self.K) is not int or self.K <= 0 or H * d != self.K:
            raise ValueError(f"compact transport has H={H}, d={d}, but H*d != K={self.K!r}")
        if type(self.same_frame_flat_cocycle) is not bool:
            raise ValueError(
                "same_frame_flat_cocycle must be a bool, got "
                f"{type(self.same_frame_flat_cocycle).__name__}: "
                f"{self.same_frame_flat_cocycle!r}")

    @property
    def n_blocks(self) -> int:
        return self.exp_blocks.shape[-3]

    @property
    def block_dim(self) -> int:
        return self.exp_blocks.shape[-1]

    @property
    def irrep_dims(self) -> List[int]:
        return [self.block_dim] * self.n_blocks

    @property
    def device(self) -> torch.device:
        return self.exp_blocks.device

    @property
    def dtype(self) -> torch.dtype:
        return self.exp_blocks.dtype

    def unsqueeze(self, dim: int) -> 'CompactFactoredTransport':
        r"""Insert a leading batch axis without materializing either pairwise or dense matrices."""
        n_leading = self.exp_blocks.dim() - 4
        logical_rank = n_leading + 4                    # conceptual (..., Nq, Nk, K, K) transport
        normalized = dim if dim >= 0 else dim + logical_rank + 1
        if normalized < 0 or normalized > n_leading:
            raise ValueError(
                "CompactFactoredTransport.unsqueeze may only add a leading axis before "
                "the conceptual (Nq, Nk, K, K) transport axes")
        return CompactFactoredTransport(
            self.exp_blocks.unsqueeze(normalized),
            self.inv_blocks.unsqueeze(normalized),
            self.K,
            same_frame_flat_cocycle=self.same_frame_flat_cocycle,
        )

    def to_dense_omega(self) -> torch.Tensor:
        r"""Explicit compatibility conversion to dense ``(..., N, N, K, K)`` transport."""
        exp_dense = CompactBlockElement(self.exp_blocks, self.K).to_dense()
        inv_dense = CompactBlockElement(self.inv_blocks, self.K).to_dense()
        with torch.autocast(device_type=exp_dense.device.type, enabled=False):
            return torch.einsum("...ikl,...jlm->...ijkm", exp_dense, inv_dense)


@dataclass
class RopeTransport:
    r"""A built transport wrapped with a gauge-RoPE positional rotation R(theta).

    ``base`` is the un-rotated transport (a dense (N,N,K,K) Omega OR a FactoredTransport). The
    effective operator is Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T.

    ``on_value`` (default True = the coherent single-gauge path) factors the transport into an
    ATTENTION gauge and a VALUE gauge (GL(K)_attention.tex:1909): the attention score
    D_KL(q_i || R_i Omega_ij R_j^T q_j) always carries the rotation, but with ``on_value=False`` the
    value aggregation mu_hat_i = sum_j beta_ij Omega_ij mu_j uses the UN-rotated base transport --
    exactly RoPE's "position-dependent attention, position-independent values" asymmetry.
    """

    base:                    'torch.Tensor | FactoredTransport | CompactFactoredTransport'
    rope:                    torch.Tensor  # (N, K, K) block-diagonal orthogonal rotation
    on_value:                bool = True   # False -> value aggregation uses the UN-rotated base (RoPE Q/K only)
    same_frame_flat_cocycle: bool = False  # trusted same-table RoPE around an already certified base
    insertion:               str  = "right"  # "right": V_i = U_i R_i^T (pure) | "left": R_i Omega R_j^T (legacy)

    def score_operator(self):
        r"""The operator the ATTENTION SCORE transports under, per ``insertion``.

        RIGHT (the pure path) folds the rotation into the vertex frame, so the score operator is an
        ordinary flat coboundary of :math:`V_i = U_i R_i^{T}` and every downstream contraction, fast
        path and cocycle certification applies to it unchanged. LEFT keeps the shipped wrapper
        semantics, where the rotation is applied around ``base`` at each contraction site instead.
        Folded once and memoized: the fold is two matmuls on the vertex factors, but it is on the
        per-iteration E-step path and there is no reason to repeat it.
        """
        if self.insertion == "left":
            return None                                       # callers keep the wrapper arithmetic
        cached = getattr(self, "_folded", None)
        if cached is None:
            if not isinstance(self.base,
                              (FactoredTransport, CompactFactoredTransport)):
                raise ValueError(
                    "rope_insertion='right' folds the rotation into the vertex frame, which needs a "
                    f"base that carries one; got {type(self.base).__name__}. A dense Omega exposes "
                    "only the composed operator, from which the per-vertex frame cannot be "
                    "recovered. Use rope_insertion='left' for those transports.")
            cached = fold_rope_into_frame(self.base, self.rope)
            object.__setattr__(self, "_folded", cached)
        return cached

    def __post_init__(self) -> None:
        if self.insertion not in ROPE_INSERTIONS:
            raise ValueError(
                f"RopeTransport insertion must be one of {ROPE_INSERTIONS}, got {self.insertion!r}")
        if type(self.same_frame_flat_cocycle) is not bool:
            raise ValueError(
                "same_frame_flat_cocycle must be a bool, got "
                f"{type(self.same_frame_flat_cocycle).__name__}: "
                f"{self.same_frame_flat_cocycle!r}")
        if isinstance(self.base, CompactFactoredTransport):
            n_query = self.base.exp_blocks.shape[-4]
            n_key = self.base.inv_blocks.shape[-4]
            K = self.base.K
        elif isinstance(self.base, FactoredTransport):
            if self.base.exp_phi.dim() < 3 or self.base.exp_neg_phi.dim() < 3:
                raise ValueError("RopeTransport factored base must have (..., N, K, K) factors")
            if (self.base.exp_phi.shape[-2] != self.base.exp_phi.shape[-1]
                    or self.base.exp_neg_phi.shape[-2] != self.base.exp_neg_phi.shape[-1]):
                raise ValueError(
                    "RopeTransport factored base factors must each end in square K x K matrix "
                    f"axes; got {tuple(self.base.exp_phi.shape)} and "
                    f"{tuple(self.base.exp_neg_phi.shape)}")
            if self.base.exp_phi.shape[-1] != self.base.exp_neg_phi.shape[-1]:
                raise ValueError(
                    "RopeTransport factored base factors must use the same K; got "
                    f"Kq={self.base.exp_phi.shape[-1]}, Kk={self.base.exp_neg_phi.shape[-1]}")
            if self.base.exp_phi.shape[:-3] != self.base.exp_neg_phi.shape[:-3]:
                raise ValueError(
                    "RopeTransport factored base factors must have matching leading batch shapes; "
                    f"got {tuple(self.base.exp_phi.shape[:-3])} and "
                    f"{tuple(self.base.exp_neg_phi.shape[:-3])}")
            n_query = self.base.exp_phi.shape[-3]
            n_key = self.base.exp_neg_phi.shape[-3]
            K = self.base.exp_phi.shape[-1]
        else:
            if self.base.dim() < 4:
                raise ValueError("RopeTransport dense base must have (..., Nq, Nk, K, K) layout")
            if self.base.shape[-2] != self.base.shape[-1]:
                raise ValueError(
                    "RopeTransport dense base must end in square K x K matrix axes; got "
                    f"shape {tuple(self.base.shape)}")
            n_query = self.base.shape[-4]
            n_key = self.base.shape[-3]
            K = self.base.shape[-1]
        if n_query != n_key:
            raise ValueError(
                "RopeTransport requires a square token transport because one rope tensor is "
                f"shared by query and key rotations; got Nq={n_query}, Nk={n_key}")
        if (self.rope.dim() < 3 or self.rope.shape[-1] != K
                or self.rope.shape[-2] != K):
            raise ValueError(
                f"RopeTransport rope must have (..., N, K, K) with K={K}, got "
                f"{tuple(self.rope.shape)}")
        if self.rope.shape[-3] != n_query:
            raise ValueError(
                "RopeTransport rope token length must match both transport token axes; "
                f"got rope N={self.rope.shape[-3]}, Nq=Nk={n_query}")


ROPE_INSERTIONS: Tuple[str, ...] = ("right", "left")


@ieee_fp32_matmul()
def fold_rope_into_frame(
    base: 'FactoredTransport | CompactFactoredTransport',
    rope: torch.Tensor,                   # (..., N, K, K) block-diagonal orthogonal rotation
) -> 'FactoredTransport | CompactFactoredTransport':
    r"""RIGHT insertion: return the transport whose vertex frame is :math:`V_i = U_i R_i^{T}`.

    The resulting operator is :math:`\Omega_{ij} = U_i R_i^{T} R_j U_j^{-1} = U_i R(\theta_j-\theta_i)
    U_j^{-1}` -- the relative rotation sits BETWEEN the query and key content factors, which is what
    ``GL(K)_attention.tex`` specifies and what Su et al. 2021 Eq (16) requires (their sign: the score
    cross term becomes :math:`Q_i^{T} R(\theta_j-\theta_i) K_j`).

    Contrast the shipped LEFT insertion :math:`R_i \Omega_{ij} R_j^{T}`, the coboundary of
    :math:`W_i = R_i U_i`, where the rotations sit OUTSIDE the content operator and never meet. That
    conjugates the transport by the query's ABSOLUTE angle. Measured on the 2026-07-26 K=20 rope
    checkpoint (audit R-4): 69% of the pair energy's spread is absolute-position contamination under
    LEFT and exactly 0 under RIGHT, and gauge covariance ``Omega^g = g Omega g^-1`` holds to 1.4e-06
    under RIGHT against a residual of 5.6 under LEFT.

    Because :math:`V_i V_j^{-1}` is an ordinary flat coboundary, the folded transport is a plain
    factored container: every downstream contraction, the cocycle certification, and the fast paths
    all apply unchanged. That is the point of folding rather than wrapping.

    An escalated float64 base (audit 2026-09-23 #12) folds in the rotation re-orthogonalized at
    float64 (:func:`_rope_at_frame_precision`), so the folded pair stays one float64 group-inverse
    pair. The two fold products run at IEEE fp32 whatever ``fp32_matmul_precision`` is
    (:func:`vfe3.runtime.ieee_fp32_matmul`): each is rounded on its own, so TF32 would leave the
    pair inverse only to :math:`2^{-10}\kappa`.
    """
    rope = _rope_at_frame_precision(rope, _factor_dtype(base))
    rope_t = rope.transpose(-1, -2)
    if isinstance(base, CompactFactoredTransport):
        H, d = base.exp_blocks.shape[-3], base.exp_blocks.shape[-1]
        # The rotation is block-diagonal on equal blocks, so its per-head blocks are the diagonal
        # (d, d) sub-matrices; slicing them keeps the compact representation compact.
        blocks = torch.stack([rope_t[..., h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(H)],
                             dim=-3)                                      # (..., N, H, d, d)
        inverse = blocks.transpose(-1, -2)                                # orthogonal -> inverse is transpose
        return CompactFactoredTransport(
            exp_blocks=base.exp_blocks @ blocks,
            inv_blocks=inverse @ base.inv_blocks,
            K=base.K,
            same_frame_flat_cocycle=base.same_frame_flat_cocycle,
        )
    return FactoredTransport(
        exp_phi=base.exp_phi @ rope_t,
        exp_neg_phi=rope @ base.exp_neg_phi,
        irrep_dims=list(base.irrep_dims),
        same_frame_flat_cocycle=base.same_frame_flat_cocycle,
    )


def _rope_at_frame_precision(
    rope:  torch.Tensor,                  # (..., N, K, K) block-diagonal orthogonal rotation
    dtype: torch.dtype,                   # storage dtype of the frame pair it meets
) -> torch.Tensor:                        # (..., N, K, K) rotation at the frames' precision
    r"""The positional rotation at an escalated float64 frame pair's precision (audit 2026-09-23 #12).

    The model builds :math:`R` at the generator dtype (float32), whose rounded cosines and sines
    leave an orthogonality defect :math:`E_0 = R^{T} R - I` of about :math:`4\times10^{-8}`. A cast
    to float64 keeps that defect, the folded pair :math:`(U R^{T}, R U^{-1})` inherits it as its
    inverse residual, and the float64 canonical-frame contract (tolerance
    :math:`64 K \varepsilon_{64} \kappa \approx 10^{-13}` on a healthy block) rejects it. Two
    Newton--Schulz polar steps (Higham, *Functions of Matrices*, SIAM 2008, ch. 8)

    .. math::
        R_{k+1} = \tfrac{1}{2} R_k \left(3I - R_k^{T} R_k\right), \qquad
        E_{k+1} = -\tfrac{3}{4} E_k^{2} + O(E_k^{3}),

    take the defect to float64 rounding. The iteration converges to the orthogonal polar factor of
    the float32 rotation, its nearest orthogonal matrix, and keeps the structural block-diagonal
    zeros exact. Any other pairing returns ``rope`` itself, so the float32 path is untouched and a
    rotation that is already float64 is not perturbed.
    """
    if dtype != torch.float64 or rope.dtype == torch.float64:
        return rope
    R = rope.double()
    eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device)
    for _ in range(2):
        R = 0.5 * (R @ (3.0 * eye - R.transpose(-1, -2) @ R))
    return R


# -- connection-regime registry --
# The CONNECTION REGIME is a registry-backed modular axis on equal footing with the structure group
# (clean-room spec sec. 4.2): config-selected, added by
# writing-and-registering, never editing call sites. The regime chooses whether the connection
# is flat at all, and the flat phi-cocycle is the sole registered regime: 'flat'
# (:func:`_build_flat`, the no-NN pure path).
# What a built transport container offers a family's coupling energy, weakest first (audit
# 2026-07-26 E-03). 'opaque' contracts into means and covariances but exposes no pairwise operator to
# invert. 'pair_inverse' is a dense (..., N, N, K, K) tensor,
# which can be inverted outright. 'coboundary' is the certified same-frame flat cocycle
# Omega_ij = U_i U_j^{-1}, whose inverse is the pair transpose and whose receiver frame cancels.
# Each level subsumes the ones before it, so a requirement is satisfiable by index comparison. The
# flat cocycle sits at the strongest level and so satisfies every requirement a family can declare;
# the ladder is kept so that requirement is still validated fail-closed against the registry rather
# than a mode-name literal.
PAIR_TRANSPORT_KINDS: Tuple[str, ...] = ("opaque", "pair_inverse", "coboundary")


@dataclass(frozen=True)
class TransportRegistration:
    """A transport builder and every routing/reporting declaration attached to it."""

    callable:                   Callable[..., TransportDict]
    needs_mu:                   bool
    needs_sigma:                bool
    batch_independent:          bool
    covariance_class:           str
    pair_transport_kind:        str
    rope_right_foldable:        bool
    state_builder:              'Optional[TransportStateBuilder]'
    serialization_keys:         Tuple[str, ...]
    offdiag_serialization_keys: Tuple[str, ...]
    gauge_equivariant:          bool
    runtime_exactness_key:      Optional[str]

    def satisfies(self, requirement: str) -> bool:
        r"""Whether this transport meets a family's ``transport_requirement``."""
        if requirement == "any":
            return True
        return (PAIR_TRANSPORT_KINDS.index(self.pair_transport_kind)
                >= PAIR_TRANSPORT_KINDS.index(requirement))


TransportState = Mapping[str, torch.Tensor]
TrainableTransportState = Dict[str, nn.Parameter]
TransportStateBuilder = Callable[['VFE3Config', GaugeGroup], TrainableTransportState]


_TRANSPORTS: Dict[str, TransportRegistration] = {}
_TRANSPORT_BUILDER_RESERVED_STATE_KEYS = frozenset({
    "phi",
    "group",
    "gauge_mode",
    "mu",
    "mu_key",
    "sigma",
    "sigma_key",
    "materialize",
    "validity_max_norm",
    "exactness_out",
    "right_phi",
})


def register_transport(
    name: str,

    *,
    covariance_class:           str,
    needs_mu:                   bool                              = False,
    needs_sigma:                bool                              = False,
    batch_independent:          bool                              = False,
    override:                   bool                              = False,
    pair_transport_kind:        str                               = "opaque",
    rope_right_foldable:        bool                              = False,
    state_builder:              'Optional[TransportStateBuilder]' = None,
    serialization_keys:         Tuple[str, ...]                   = (),
    offdiag_serialization_keys: Tuple[str, ...]                   = (),
    gauge_equivariant:          Optional[bool]                    = None,
    runtime_exactness_key:      Optional[str]                     = None,
) -> Callable:
    """Decorator registering a transport (connection-regime) builder under ``name``.

    ``covariance_class`` is the exact covariance/equivariance label emitted in run artifacts. It is
    mandatory so every config-selectable transport is reportable without a second literal dispatch
    table. ``needs_mu``/``needs_sigma`` are state-routing metadata: they declare which belief fields the
    regime's Omega builder consumes, so callers feed mu/sigma by querying the registry rather than
    matching literal mode names. Declaring them here keeps the add-by-registering contract -- a new
    stateful regime advertises its requirements at registration, not at every call site.

    ``state_builder`` optionally constructs the transport's complete trainable state from the active
    config and gauge group. ``serialization_keys`` declares the stable top-level model parameter names
    for that state. The model validates the builder result against this declaration and registers each
    parameter directly under its declared name, preserving external checkpoint keys without a container
    prefix. A stateful transport is therefore added entirely at registration; model and E-step routing
    consume one generic mapping and require no transport-specific branch. The optional
    ``offdiag_serialization_keys`` declaration identifies edge tables whose first two axes have a
    meaningful off-diagonal norm; shape alone cannot distinguish them from square coefficient tables.

    ``pair_transport_kind`` declares what the built container offers a family's coupling energy (see
    :data:`PAIR_TRANSPORT_KINDS`), so a family whose energy needs an invertible or coboundary operator
    is validated against the registry instead of a mode-name literal. It defaults to the WEAKEST level
    so a new registration must claim a stronger capability deliberately.

    ``batch_independent`` declares that the builder's ``Omega`` does NOT depend on the batch (it is a
    function of a model parameter only), so the builder
    returns a batch-collapsed ``(N, N, K, K)`` Omega that broadcasts across the batch (the D3
    memory collapse). ``_transport`` reads this flag to skip the
    per-sequence ``[0]`` strip it applies to ordinary ``(B, N, N, K, K)`` builders.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an existing
    name silently shadowed the first, so a config-selected seam could dispatch to an unintended
    implementation. Pass ``override=True`` to replace deliberately. The callable and all declarations
    are installed together as one frozen record, so stale routing flags cannot survive an override.
    """
    if not isinstance(covariance_class, str) or not covariance_class:
        raise ValueError("transport covariance_class must be a nonempty string")
    if pair_transport_kind not in PAIR_TRANSPORT_KINDS:
        raise ValueError(
            f"transport pair_transport_kind must be one of {PAIR_TRANSPORT_KINDS}, got "
            f"{pair_transport_kind!r}")
    if state_builder is None and serialization_keys:
        raise ValueError("transport serialization_keys require a state_builder")
    if state_builder is not None and not serialization_keys:
        raise ValueError("a transport state_builder requires declared serialization_keys")
    if len(set(serialization_keys)) != len(serialization_keys):
        raise ValueError("transport serialization_keys must be unique")
    if any(not isinstance(key, str) or not key or "." in key for key in serialization_keys):
        raise ValueError("transport serialization_keys must be nonempty top-level parameter names")
    reserved_state_keys = set(serialization_keys) & _TRANSPORT_BUILDER_RESERVED_STATE_KEYS
    if reserved_state_keys:
        raise ValueError(
            "transport serialization_keys contain reserved transport-builder keyword(s): "
            f"{sorted(reserved_state_keys)}"
        )
    if not set(offdiag_serialization_keys).issubset(serialization_keys):
        raise ValueError("transport offdiag_serialization_keys must be declared serialization_keys")
    if runtime_exactness_key is not None and (
        not isinstance(runtime_exactness_key, str) or not runtime_exactness_key
    ):
        raise ValueError("transport runtime_exactness_key must be None or a nonempty string")
    if gauge_equivariant is None:
        # Compatibility for complete-record restore helpers written before this additive field.
        # Unknown/custom covariance labels remain fail-closed.
        gauge_equivariant = covariance_class in {"covariant", "covariant (flat)"}

    def _wrap(fn: Callable[..., TransportDict]) -> Callable[..., TransportDict]:
        if name in _TRANSPORTS and not override:
            raise KeyError(f"transport mode {name!r} already registered; pass override=True to replace")
        registration = TransportRegistration(
            callable=fn,
            needs_mu=needs_mu,
            needs_sigma=needs_sigma,
            batch_independent=batch_independent,
            covariance_class=covariance_class,
            pair_transport_kind=pair_transport_kind,
            rope_right_foldable=rope_right_foldable,
            state_builder=state_builder,
            serialization_keys=serialization_keys,
            offdiag_serialization_keys=offdiag_serialization_keys,
            gauge_equivariant=bool(gauge_equivariant),
            runtime_exactness_key=runtime_exactness_key,
        )
        _TRANSPORTS[name] = registration
        return fn
    return _wrap


def get_transport_registration(name: str) -> TransportRegistration:
    """Return the complete registration record for ``name`` (KeyError if absent)."""
    if name not in _TRANSPORTS:
        raise KeyError(f"no transport {name!r}; available: {sorted(_TRANSPORTS)}")
    return _TRANSPORTS[name]


def get_transport(name: str) -> Callable[..., TransportDict]:
    """Return the registered transport builder (KeyError-with-available-list if absent)."""
    return get_transport_registration(name).callable


@register_transport(
    "flat",
    covariance_class="covariant (flat)",
    gauge_equivariant=True,
    rope_right_foldable=True,
    pair_transport_kind="coboundary",   # builds Omega_ij = U_i U_j^{-1} from ONE vertex table
)
def _build_flat(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:              str             = "learned",   # 'learned' (Regime I flat) or 'trivial'
    validity_max_norm:       Optional[float] = None,        # fail-closed chart radius (None -> default 20.0)
    **kwargs,                             # tolerated (a future non-flat builder shares this shape)
) -> TransportDict:
    r"""Flat (Regime I) phi-cocycle transport: the registered default.

    A thin adapter forwarding verbatim to :func:`compute_transport_operators`
    (Omega_ij = exp(phi_i) exp(-phi_j) in GL+(K)); bit-identical to calling it directly. Extra
    keyword args are tolerated and ignored so a future stateful builder can share this call shape
    without editing the registry call sites.
    """
    return compute_transport_operators(phi, group, gauge_mode=gauge_mode,
                                       validity_max_norm=validity_max_norm)


# Fail-closed chart radius of every transport matrix exponential (VFE3Config.transport_chart_max_norm,
# default 20.0). A raw Lie-algebra matrix with ||M||_F above it is rejected, never rescaled: the
# former Frobenius clamp substituted the surrogate exp(M*scale) for exp(M) and is retired
# (2026-10-02). Low-level callers that pass no radius use the config default.
DEFAULT_TRANSPORT_CHART_MAX_NORM: float = VFE3Config.transport_chart_max_norm


# Measured-condition escalation of the stored frame pair (audit 2026-09-23 #12). The pair (U, U^-1)
# is stored in float32. With the escalation enabled (VFE3Config.frame_pair_fp64_escalation, installed
# by VFEModel), a producer call whose stored pair cannot stay mutually inverse at float32 is redone in
# float64 as a whole, and every relational consumer of that frame then runs in float64. It was
# unconditional from 2026-10-02 to 2026-10-04 and is opt-in again: one escalated call costs about
# 7x a float32 step on the RTX 5090 (FP64 at 1/64 rate). Disabled, no certificate is formed and the
# stored pair keeps the inverse residual <= eps_32 kappa_F(U); CanonicalFrameContext still fails
# closed far above that. The tolerance (VFE3Config.frame_pair_condition_tol) is read at call time.
# The counter is a host int incremented from the already-read decision, so it adds no device work
# and no synchronization.
_FRAME_PAIR_FP64_ESCALATION: bool = False
_FRAME_PAIR_CONDITION_TOL: float = 1e-4
_FRAME_PAIR_FP64_ESCALATIONS: int = 0


def _resolve_chart_max_norm(
    validity_max_norm: Optional[float],   # caller radius; None -> DEFAULT_TRANSPORT_CHART_MAX_NORM
) -> float:
    r"""Return the validated fail-closed chart radius (positive and finite)."""
    radius = DEFAULT_TRANSPORT_CHART_MAX_NORM if validity_max_norm is None else validity_max_norm
    if (isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not math.isfinite(radius) or radius <= 0.0):
        raise ValueError(
            f"transport chart radius validity_max_norm must be finite and positive, got {radius!r}")
    return float(radius)


def set_frame_pair_fp64_escalation(enabled: bool) -> bool:
    r"""Enable or disable the condition-keyed float64 frame-pair escalation; returns the previous value."""
    if not isinstance(enabled, bool):
        raise ValueError(f"frame-pair float64 escalation must be bool, got {enabled!r}")
    global _FRAME_PAIR_FP64_ESCALATION
    previous = _FRAME_PAIR_FP64_ESCALATION
    _FRAME_PAIR_FP64_ESCALATION = enabled
    return previous


def frame_pair_fp64_escalation() -> bool:
    r"""Return whether producers certify the float32 frame pair and escalate it to float64."""
    return _FRAME_PAIR_FP64_ESCALATION


def set_frame_pair_condition_tol(tol: float) -> float:
    r"""Set the escalation tolerance :math:`\tau` of :func:`_frame_pair_kappa`; returns the previous value."""
    if (isinstance(tol, bool) or not isinstance(tol, (int, float))
            or not math.isfinite(tol) or not 0.0 < tol < 1.0):
        raise ValueError(f"frame-pair condition tolerance must lie in (0, 1), got {tol!r}")
    global _FRAME_PAIR_CONDITION_TOL
    previous = _FRAME_PAIR_CONDITION_TOL
    _FRAME_PAIR_CONDITION_TOL = float(tol)
    return previous


def frame_pair_fp64_escalations() -> int:
    r"""Producer calls whose frame pair was escalated to float64 since the last reset.

    One call is one ``stable_matrix_exp_pair`` / compact-pair build, so a composed
    ``build_factored_transport`` can count two. Diagnostic and evaluation builds count too.
    """
    return _FRAME_PAIR_FP64_ESCALATIONS


def reset_frame_pair_fp64_escalations() -> None:
    r"""Zero the frame-pair float64 escalation counter (per-run accounting)."""
    global _FRAME_PAIR_FP64_ESCALATIONS
    _FRAME_PAIR_FP64_ESCALATIONS = 0


def _count_frame_pair_escalation() -> None:
    global _FRAME_PAIR_FP64_ESCALATIONS
    _FRAME_PAIR_FP64_ESCALATIONS += 1


def _frame_pair_kappa(
    right:  Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (..., d, d) right pair (E, E^-1)
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # unique -> occurrence rows
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Build the Frobenius condition of the pair a producer call will actually store.

    For each trailing ``(d, d)`` block,

    .. math::
        \kappa_F(U_b) = \lVert U_b \rVert_F \lVert U_b^{-1} \rVert_F \ge \kappa_2(U_b),

    and the stored pair's consistency obeys
    :math:`\lVert \mathrm{fl}(U)\,\mathrm{fl}(U^{-1}) - I \rVert \lesssim \kappa(U)\,\varepsilon`.
    The producer escalates iff :math:`\max_b \kappa_F(U_b)\,\varepsilon_{32} > \tau` with
    :math:`\tau = 10^{-4}` (``_FRAME_PAIR_CONDITION_TOL``). Note :math:`\kappa_F \ge d`.

    With ``right`` the stored pair is the composition :math:`(U E, E^{-1} U^{-1})`, so the condition
    is taken of that product; ``gather`` first expands deduplicated token rows to the occurrence
    layout the right factor is indexed by. The returned callable receives the float64 forward
    elements and the float64 inverses the certified inverse has already formed.
    """
    def kappa(forward64: torch.Tensor, inverse64: torch.Tensor) -> torch.Tensor:
        if gather is not None:
            forward64, inverse64 = gather(forward64), gather(inverse64)
        if right is not None:
            forward64 = forward64 @ right[0].double()                    # (..., d, d) U E
            inverse64 = right[1].double() @ inverse64                    # (..., d, d) E^-1 U^-1
        return (torch.linalg.matrix_norm(forward64)
                * torch.linalg.matrix_norm(inverse64)).amax()
    return kappa


def stable_matrix_exp_pair(
    matrix:                  torch.Tensor,       # (..., d, d) Lie-algebra matrices

    *,
    only_forward:            bool                = False,
    block_dims:              Optional[List[int]] = None,     # per-block sizes (sum==d) for a block-diagonal M
    validity_max_norm:       Optional[float]     = None,     # fail-closed chart radius (None -> default 20.0)
    _right_pair:             Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # certify (U E, E^-1 U^-1)
    _force_fp64:             bool                = False,    # skip the float32 attempt (joint escalation)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Exact exp(M) on the transport chart and optionally its float64-computed inverse.

    The paired negative factor is the inverse of the represented forward factor, not a separately
    rounded evaluation of exp(-M), so the stored pair shares one finite-precision group element.

    Domain: the chart radius :math:`r` (``validity_max_norm``, the configured
    ``transport_chart_max_norm``) is fail-closed. Any raw :math:`\lVert M\rVert_F > r` or a
    nonfinite norm raises ``ValueError``; the matrix is never rescaled, so the returned factor is
    always :math:`\exp(M)` itself. The retired Frobenius clamp returned the surrogate
    :math:`\exp(M \cdot 20/\lVert M\rVert_F)` instead (2026-10-02). The bound applies to every
    group, including the compact skew-symmetric ``so_n`` towers.

    ``block_dims`` (audit 4b): when M is block-diagonal with these blocks (e.g. block_glk's
    GL(d_head)^H), exp(M) is exactly block-diagonal with the per-block exponentials, so each
    d_head x d_head block is exponentiated independently -- an O(H * d_head^3) cost instead of
    O(K^3) for the full K x K. ``None`` (a single block, a cross-coupled basis, or a skew group)
    takes the full-matrix path.

    Precision: the exponential is evaluated at the storage precision (float32, or float64 for a
    float64 input). Under :func:`frame_pair_fp64_escalation`, a float32 pair failing
    :func:`_frame_pair_kappa`'s certificate, :math:`\max_b \kappa_F(U_b)\,\varepsilon_{32} > \tau`
    (``frame_pair_condition_tol``, default :math:`10^{-4}`), is returned in float64 for the whole
    call, recomputed from the same ``M`` and inverted from the float64 ``exp(M)`` (audit 2026-09-23
    #12). The verdict rides the inverse's existing host read; ``only_forward`` builds no inverse and
    is never certified. With the escalation disabled no certificate is formed.
    """
    validity_max_norm = _resolve_chart_max_norm(validity_max_norm)
    # Both chart decisions are reduced on the device and read in ONE host transfer.
    with torch.no_grad():
        raw_mat_norm = matrix.norm(dim=(-2, -1))
        decisions = torch.stack(((~torch.isfinite(raw_mat_norm)).any(),
                                 (raw_mat_norm > validity_max_norm).any()))
    nonfinite, outside_chart = decisions.tolist()
    if nonfinite:
        raise ValueError(
            "transport chart validity bound encountered a nonfinite matrix norm"
        )
    if outside_chart:
        observed = float(raw_mat_norm.max())
        raise ValueError(
            "transport chart validity bound exceeded before the matrix exponential: "
            f"observed ||M||_F={observed:.6g} > transport_chart_max_norm={validity_max_norm:.6g}"
        )

    # float32 storage floor, matching _stable_compact_glk_exp_pair (audit 2026-07-25 F2). The dense
    # route reaches here with an autocast-eligible einsum(phi, generators) result, so without this the
    # dense and compact representations would only agree on their (reduced) storage dtype by both
    # being wrong. Parity between the two paths is preserved -- at float32 rather than at bf16.
    orig_dtype = matrix.dtype if matrix.dtype in (torch.float32, torch.float64) else torch.float32

    # Keyed to the TENSOR's device (audit 2026-07-05 m10): the old 'cuda' literal left the island
    # open under a CPU autocast context (torch.amp.autocast('cpu', bf16)), which _amp_context
    # deliberately supports -- matrix_exp would then run in bf16 on CPU-AMP runs.
    with torch.amp.autocast(matrix.device.type, enabled=False):
        matrix_up = matrix.to(orig_dtype).contiguous()
        blocked = block_dims is not None and len(block_dims) > 1

        def _exp(m: torch.Tensor) -> torch.Tensor:
            return _blockwise_matrix_exp(m, block_dims) if blocked else torch.linalg.matrix_exp(m)

        exp_pos = _exp(matrix_up)
        if only_forward:
            return exp_pos, None
        if orig_dtype == torch.float64:
            escalate = False
            exp_neg = (_blockwise_group_inverse(exp_pos, block_dims) if blocked
                       else _checked_group_inverse(exp_pos))
        elif _force_fp64:
            escalate = True
        elif blocked:
            exp_neg, escalate = _blockwise_group_inverse_certified(
                exp_pos, block_dims, certify=_FRAME_PAIR_FP64_ESCALATION, right_pair=_right_pair)
        else:
            exp_neg, escalate = _checked_group_inverse_certified(
                exp_pos, kappa=(_frame_pair_kappa(_right_pair)
                                if _FRAME_PAIR_FP64_ESCALATION else None))
        if escalate:
            exp_pos = _exp(matrix.to(torch.float64).contiguous())
            exp_neg = (_blockwise_group_inverse(exp_pos, block_dims) if blocked
                       else _checked_group_inverse(exp_pos))
            _count_frame_pair_escalation()
    return exp_pos, exp_neg


def _blockwise_matrix_exp(
    matrix:     torch.Tensor,             # (..., d, d) block-diagonal Lie-algebra matrix
    block_dims: List[int],                # block sizes; sum == d
) -> torch.Tensor:                        # (..., d, d) block-diagonal exp
    r"""exp of a block-diagonal matrix = block-diagonal of the blocks' exps (audit 4b).

    Exact for a block-diagonal M (off-block entries are zero, so the blocks commute trivially
    and exp does not mix them; Higham, Functions of Matrices, Sec 10.3). Off-block entries of the
    output are left at zero -- matching the full exp, whose off-block entries are exactly zero for
    a block-diagonal input.

    When the blocks are EQUAL size (block_glk's GL(d_head)^H), the H diagonal blocks are stacked
    into one batched ``matrix_exp`` (a single call instead of H sequential ones -- the
    launch-bound pattern a GPU is starved by); ``matrix_exp`` evaluates each (d, d) block
    independently, so this is bit-identical to the per-block loop (pinned at 1e-12 by
    tests/test_perf_equivalence.py::test_per_block_exp_is_bit_equivalent_to_full_exp). Unequal
    block sizes (a general block-diagonal M) fall back to the per-block loop.
    """
    out = torch.zeros_like(matrix)
    if len(set(block_dims)) == 1 and len(block_dims) > 1:
        # Diagonal-block gather/scatter without the H-iteration Python loops (audit
        # 2026-06-09 overnight F3): viewing (..., H*d, H*d) as (..., H, d, H, d), the H
        # diagonal blocks are torch.diagonal over the two H axes -- one view each way --
        # so the read is a single copy and the write-back a single in-place copy_ into the
        # diagonal view of the zero output (same kernel-level semantics as the former
        # slice assignments, H+H fewer launches).
        H, d = len(block_dims), block_dims[0]
        batch = matrix.shape[:-2]
        m5 = matrix.reshape(*batch, H, d, H, d)
        blocks = torch.diagonal(m5, dim1=-4, dim2=-2).movedim(-1, 0).contiguous()  # (H, ..., d, d)
        exps = torch.linalg.matrix_exp(blocks)                  # one batched call
        out5 = out.reshape(*batch, H, d, H, d)
        torch.diagonal(out5, dim1=-4, dim2=-2).copy_(exps.movedim(0, -1))
        return out
    start = 0
    for dim in block_dims:
        end = start + dim
        blk = matrix[..., start:end, start:end].contiguous()
        out[..., start:end, start:end] = torch.linalg.matrix_exp(blk)
        start = end
    return out


def _blockwise_group_inverse(
    matrix:     torch.Tensor,             # (..., d, d) represented block-diagonal group element
    block_dims: List[int],                # block sizes; sum == d
) -> torch.Tensor:                        # (..., d, d) inverse with structural off-block zeros
    r"""Invert represented diagonal blocks in float64 without materializing a full dense solve."""
    return _blockwise_group_inverse_certified(matrix, block_dims)[0]


def _blockwise_group_inverse_certified(
    matrix:     torch.Tensor,             # (..., d, d) represented block-diagonal group element
    block_dims: List[int],                # block sizes; sum == d

    *,
    certify:    bool = False,             # also return the frame-pair escalation verdict
    right_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (..., d, d) block-diagonal (E, E^-1)
) -> Tuple[torch.Tensor, bool]:           # ((..., d, d) inverse, escalate to float64)
    r""":func:`_blockwise_group_inverse` plus the per-block frame-pair verdict (audit 2026-09-23 #12).

    Equal blocks certify the stacked ``(..., H, d, d)`` blocks inside their one inverse read;
    unequal blocks OR the verdicts of their existing per-block reads. A block-diagonal right factor
    composes block by block, so the certificate uses its matching diagonal blocks.
    """
    out = torch.zeros_like(matrix)
    escalate = False
    if len(set(block_dims)) == 1 and len(block_dims) > 1:
        H, d = len(block_dims), block_dims[0]
        batch = matrix.shape[:-2]

        def _diag_blocks(t: torch.Tensor) -> torch.Tensor:          # (..., H, d, d)
            return torch.diagonal(
                t.reshape(*t.shape[:-2], H, d, H, d),
                dim1=-4,
                dim2=-2,
            ).movedim(-1, -3)

        blocks = _diag_blocks(matrix).contiguous()
        kappa = None
        if certify:
            right = (None if right_pair is None
                     else (_diag_blocks(right_pair[0]), _diag_blocks(right_pair[1])))
            kappa = _frame_pair_kappa(right)
        inverses, escalate = _checked_group_inverse_certified(blocks, kappa=kappa)
        out_view = out.reshape(*batch, H, d, H, d)
        torch.diagonal(out_view, dim1=-4, dim2=-2).copy_(inverses.movedim(-3, -1))
        return out, escalate

    start = 0
    for dim in block_dims:
        end = start + dim
        block = matrix[..., start:end, start:end].contiguous()
        kappa = None
        if certify:
            right = (None if right_pair is None
                     else (right_pair[0][..., start:end, start:end],
                           right_pair[1][..., start:end, start:end]))
            kappa = _frame_pair_kappa(right)
        inverse, block_escalate = _checked_group_inverse_certified(block, kappa=kappa)
        out[..., start:end, start:end] = inverse
        escalate = escalate or block_escalate
        start = end
    return out, escalate


def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:              str             = "learned",   # 'learned' (Regime I flat) or 'trivial'
    validity_max_norm:       Optional[float] = None,        # fail-closed chart radius (None -> default 20.0)
) -> TransportDict:
    r"""phi/exp transport Omega_ij = exp(phi_i) @ exp(-phi_j) in GL+(K).

    Flat (Regime I) transport operator construction. 'trivial' returns Omega = I.
    Returns 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K), 'Omega' (B,N,N,K,K).
    The 'constant' gauge mode is intentionally NOT supported (it would require a
    per-head learned Omega parameter, which this no-NN design does not have);
    'constant' raises ValueError.

    Conditioning note (audit 2026-06-09 overnight F26): for the NON-COMPACT groups
    (block_glk; skew_symmetric=False) Omega is not orthogonal and cond(Omega)
    grows like exp(2 ||phi_matrix||); at the phi retraction's default max_norm=5.0 a
    draw can reach cond ~1e7-1e10; compact so towers give orthogonal Omega (cond = 1).
    Omega is built at the working dtype here, so for extreme draws prefer a compact group or
    bound phi via the retraction max_norm.
    """
    B, N, _ = phi.shape
    K = group.generator_shape[-1]
    dtype = phi.dtype
    device = phi.device

    if gauge_mode == "trivial":
        eye_K = torch.eye(K, device=device, dtype=dtype)
        return {
            "exp_phi":     eye_K.expand(B, N, K, K).contiguous(),
            "exp_neg_phi": eye_K.expand(B, N, K, K).contiguous(),
            "Omega":       eye_K.expand(B, N, N, K, K).contiguous(),
        }
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    phi_matrix = _embed_algebra_fp32_floor(phi, group, pattern="bna,aij->bnij")
    # Per-block exp when the group is genuinely block-diagonal (block_glk without cross-couplings
    # -> irrep_dims [d_head]*H); single-block ([K]: one-head block_glk, a single so_n irrep) takes the full path.
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, block_dims=block_dims, validity_max_norm=validity_max_norm)
    # fp32 island on the pair product (audit 2026-07-26 C-05). The vertex factors above are already
    # pinned to float32 by _embed_algebra_fp32_floor + the exp's own upcasts, but einsum is
    # autocast-eligible, so under AMP the dense Omega was rebuilt in bf16 -- measured
    # ||Omega_ii - I||_inf = 4.76e-03 against 1.19e-07 at fp32. Omega_ii = I is the identity that
    # makes the structural self energy E_ii = 0 and keeps the self pair out of pair_mask, so this is
    # not a rounding nuisance. The compact/factored siblings never lost it: they carry their own
    # autocast(enabled=False) islands. Reachable on the flat path, where _can_fuse_flat returns
    # False for a single-block group and this dense operator is what the E-step consumes.
    with torch.amp.autocast(phi.device.type, enabled=False):
        omega = torch.einsum("bikl,bjlm->bijkm", exp_phi, exp_neg_phi)
        # This constructor certifies a flat same-frame cocycle: its self link is
        # exactly identity, including its zero derivative with respect to phi.
        omega.diagonal(dim1=1, dim2=2).copy_(
            torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype).unsqueeze(-1))
    return {"exp_phi": exp_phi, "exp_neg_phi": exp_neg_phi, "Omega": omega}


def _embed_algebra_fp32_floor(
    phi:        torch.Tensor,               # (..., n_gen) gauge-frame coordinates
    generators: 'torch.Tensor | GaugeGroup', # (n_gen, K, K) basis or lazy group

    *,
    pattern:    str = "...na,aij->...nij",  # einsum pattern for the embedding contraction
) -> torch.Tensor:
    r"""Embed ``phi`` in the Lie algebra at float32 minimum, outside any autocast context.

    ``einsum`` is autocast-eligible, so under AMP this contraction otherwise returns a bf16/fp16
    Lie-algebra matrix and the matrix exponential can only ever recover a reduced-precision operator,
    however carefully its own interior upcasts (audit 2026-07-25 F2). Pinning the embedding also
    restores exact dense/compact parity: the compact route reshapes ``phi`` with no contraction at
    all, so only the dense route was being rounded. float64 inputs are preserved.
    """
    with torch.amp.autocast(phi.device.type, enabled=False):
        phi_f = phi if phi.dtype in (torch.float32, torch.float64) else phi.float()
        if isinstance(generators, GaugeGroup):
            basis_dtype = generators.generator_dtype
            if basis_dtype not in (torch.float32, torch.float64):
                basis_dtype = torch.float32
            return generators.embed_algebra(phi_f, dtype=basis_dtype)
        gen_f = (generators if generators.dtype in (torch.float32, torch.float64)
                 else generators.float())
        return torch.einsum(pattern, phi_f, gen_f)


def _checked_group_inverse(
    omega:        torch.Tensor,             # (..., d, d) stored full elements or compact blocks

    *,
    validate:     bool            = True,   # strict public/offline check; False is the async hot path
    residual_tol: Optional[float] = None,
) -> torch.Tensor:
    r"""True float64 inverse with nonfinite/singular failure and dtype restoration.

    The strict (``validate=True``) route pays exactly ONE host synchronization per call (audit
    2026-09-23 #5): the finiteness of ``omega``, the LU status of the float64 inverse, the finiteness
    of ``inverse64`` and of the cast ``inverse`` (plus the optional residual verdict) are reduced on
    the device into one flag vector and read once. ``torch.linalg.inv_ex`` replaces
    ``torch.linalg.inv`` on that route because the latter checks its LU status with its own host
    read. The detailed error message is built only after a failure, in the original precedence.
    ``validate=False`` keeps ``torch.linalg.inv`` and performs no additional check.

    When ``residual_tol`` is supplied, the STORED pair is additionally verified against
    :math:`\lVert U U^{-1} - I \rVert_\infty \le \texttt{residual\_tol}`. The finiteness checks alone
    cannot see precision loss: the inverse is exact in float64, so every way this pair stops being a
    group-inverse pair is a rounding effect of the storage dtype, which stays perfectly finite
    (audit 2026-07-25 F2). The check costs one reduction, which joins the same single host read, so
    it is opt-in and off on the training hot path.
    """
    return _checked_group_inverse_certified(
        omega, validate=validate, residual_tol=residual_tol)[0]


def _checked_group_inverse_certified(
    omega:        torch.Tensor,             # (..., d, d) stored full elements or compact blocks

    *,
    validate:     bool            = True,
    residual_tol: Optional[float] = None,
    kappa:        Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, bool]:           # (inverse at omega.dtype, escalate to float64)
    r""":func:`_checked_group_inverse` plus the frame-pair escalation verdict (audit 2026-09-23 #12).

    ``kappa`` (see :func:`_frame_pair_kappa`) maps the float64 elements and their float64 inverses
    to :math:`\max_b \kappa_F`; the verdict :math:`\kappa_F\,\varepsilon(\text{omega.dtype}) > \tau`
    joins the strict route's ONE host read, so certifying costs no extra synchronization. When it
    escalates, the caller discards the cast inverse and rebuilds the pair in float64, so a cast that
    overflowed ``omega.dtype`` is not an error on that call. Without ``kappa`` the verdict is False.
    """
    if not validate:
        try:
            with torch.amp.autocast(omega.device.type, enabled=False):
                inverse64 = torch.linalg.inv(omega.double())
        except RuntimeError as exc:
            raise ValueError("omega group element is singular and cannot be inverted") from exc
        return inverse64.to(omega.dtype), False
    with torch.amp.autocast(omega.device.type, enabled=False):
        omega64 = omega.double()
        inverse64, info = torch.linalg.inv_ex(omega64)
    inverse = inverse64.to(omega.dtype)
    with torch.no_grad():
        flags = [
            ~torch.isfinite(omega).all(),
            (info != 0).any(),
            ~torch.isfinite(inverse64).all(),
            ~torch.isfinite(inverse).all(),
        ]
        residual = None
        if residual_tol is not None:
            eye = torch.eye(omega.shape[-1], device=omega.device, dtype=torch.float64)
            residual = (omega.double() @ inverse.double() - eye).abs().amax()
            flags.append(~(residual <= residual_tol))
        if kappa is not None:
            flags.append(kappa(omega64, inverse64) * torch.finfo(omega.dtype).eps
                         > _FRAME_PAIR_CONDITION_TOL)
        # The single host read for the whole strict certificate.
        host = torch.stack(flags).tolist()
    nonfinite_omega, singular, nonfinite_64, nonfinite_cast = host[:4]
    residual_bad = residual_tol is not None and host[4]
    escalate = kappa is not None and bool(host[-1])
    if nonfinite_omega:
        raise FloatingPointError(
            "omega group element contains nonfinite values before inversion")
    if singular:
        raise ValueError("omega group element is singular and cannot be inverted")
    if nonfinite_64:
        raise FloatingPointError(
            "omega group-element inverse is nonfinite; the element is singular or "
            "numerically unrepresentable")
    if nonfinite_cast and not escalate:
        raise FloatingPointError(
            f"omega group-element inverse is nonfinite after conversion to {omega.dtype}")
    if residual_bad:
        raise FloatingPointError(
            "stored omega group-element pair violates the inverse identity: "
            f"||U U^-1 - I||_inf={float(residual):.6g} > residual_tol={residual_tol:.6g} "
            f"(storage dtype {omega.dtype}); the represented element and its stored "
            "inverse no longer form a group-inverse pair, so the transport cocycle is "
            "not exact"
        )
    return inverse, escalate


def _stable_compact_glk_exp_pair(
    blocks:                  torch.Tensor,       # (..., H, d, d) block-diagonal gl(d)^H element

    *,
    validity_max_norm:       Optional[float] = None,     # fail-closed chart radius (None -> default 20.0)
    _content_token_ids:      Optional[torch.Tensor] = None,
    _right_pair:             Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # certify (U E, E^-1 U^-1)
    _force_fp64:             bool            = False,     # skip the float32 attempt (joint escalation)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exponentiate packed blocks and invert their represented values in float64.

    Private token IDs certify a built-in table lookup. Pre-exp decisions still see every
    occurrence; only exp/inverse work is shared, then gathered before composition. The
    table gradient is equivalent in real arithmetic, with a different reduction order.

    Under :func:`frame_pair_fp64_escalation`, float32 storage is certified by
    :func:`_frame_pair_kappa` inside the inverse's single host read (audit 2026-09-23 #12); with the
    escalation disabled no certificate is formed. A call that fails it returns the whole pair in float64:
    ``U = exp(M)`` recomputed in float64 from the same blocks and ``U^-1`` inverted from
    that float64 ``U`` itself, never the float64 inverse of the rounded float32 ``U`` (which is
    no more consistent with a float64 ``U`` than the float32 pair was). ``_right_pair`` certifies
    the composed pair instead, and ``_force_fp64`` goes straight to float64.
    """
    validity_max_norm = _resolve_chart_max_norm(validity_max_norm)
    # The chart radius bounds the Frobenius norm of the whole block-diagonal element, as on the
    # dense route; both decisions share one host read.
    with torch.no_grad():
        raw_mat_norm = blocks.square().sum(dim=(-3, -2, -1)).sqrt()
        decisions = torch.stack(((~torch.isfinite(raw_mat_norm)).any(),
                                 (raw_mat_norm > validity_max_norm).any()))
    nonfinite, outside_chart = decisions.tolist()
    if nonfinite:
        raise ValueError(
            "transport chart validity bound encountered a nonfinite matrix norm"
        )
    if outside_chart:
        observed = float(raw_mat_norm.max())
        raise ValueError(
            "transport chart validity bound exceeded before the matrix exponential: "
            f"observed ||M||_F={observed:.6g} > transport_chart_max_norm={validity_max_norm:.6g}"
        )

    # Represented fp32 storage floor; the exponential is evaluated at the storage precision.
    store_dtype = blocks.dtype if blocks.dtype in (torch.float32, torch.float64) else torch.float32

    inverse_index = None
    occurrence_shape = blocks.shape
    if _content_token_ids is not None:
        ids = _content_token_ids.reshape(-1)
        unique_ids, inverse_index = torch.unique(ids, return_inverse=True)
        if unique_ids.numel() < ids.numel():
            first = torch.full_like(unique_ids, ids.numel())
            first.scatter_reduce_(0, inverse_index, torch.arange(ids.numel(), device=ids.device),
                                  reduce="amin", include_self=True)
            blocks = blocks.reshape(-1, *blocks.shape[-3:]).index_select(0, first)
        else:
            inverse_index = None
    with torch.amp.autocast(blocks.device.type, enabled=False):
        exp_pos = torch.linalg.matrix_exp(blocks.to(store_dtype).contiguous())
        if store_dtype == torch.float64:
            exp_neg, escalate = _checked_group_inverse(exp_pos), False
        elif _force_fp64:
            escalate = True
        else:
            gather = None
            if inverse_index is not None and _right_pair is not None:
                index, shape = inverse_index, occurrence_shape

                def gather(t: torch.Tensor) -> torch.Tensor:
                    return t.index_select(0, index).reshape(shape)
            exp_neg, escalate = _checked_group_inverse_certified(
                exp_pos, kappa=(_frame_pair_kappa(_right_pair, gather)
                                if _FRAME_PAIR_FP64_ESCALATION else None))
        if escalate:
            exp_pos = torch.linalg.matrix_exp(blocks.to(torch.float64).contiguous())
            exp_neg = _checked_group_inverse(exp_pos)
            _count_frame_pair_escalation()
    if inverse_index is not None:
        exp_pos = exp_pos.index_select(0, inverse_index).reshape(occurrence_shape)
        exp_neg = exp_neg.index_select(0, inverse_index).reshape(occurrence_shape)
    return exp_pos, exp_neg


@ieee_fp32_matmul()
def build_factored_transport(
    phi:        torch.Tensor,             # (..., N, n_gen) gauge frames (optional leading batch axis)
    group:      GaugeGroup,               # block-diagonal with equal blocks (len(irrep_dims) > 1)

    *,
    gauge_mode:              str                    = "learned",   # 'learned' (Regime I flat) or 'trivial'
    compact_blocks:          bool                   = False,       # canonical block_glk: retain (..., N, H, d, d) factors
    validity_max_norm:       Optional[float]        = None,        # fail-closed chart radius (None -> default 20.0)
    right_phi:               Optional[torch.Tensor] = None,        # (..., N, n_gen) exact right factor exp(Y)
    _content_token_ids:      Optional[torch.Tensor] = None,        # private certified table-lookup identity
) -> 'CompactFactoredTransport | FactoredTransport':
    r"""Flat phi-cocycle transport in FACTORED form, skipping the dense (..., N, N, K, K) Omega.

    Builds only the per-token vertex exponentials exp(phi_i), exp(-phi_j) (the same factors
    ``compute_transport_operators`` builds) and the ``ikl,jlm->ijkm`` Omega einsum is NEVER run.
    With ``compact_blocks=True`` on canonical uncoupled ``block_glk``, both the algebra embedding
    and the returned factors retain ``(..., N, H, d, d)`` storage; every other group and the
    default keep the legacy dense-vertex :class:`FactoredTransport` representation.
    Consumers read the vertex factors directly (P0 #2); here it only requires the exps, which the
    block-diagonal exp machinery already produces. Rank-agnostic
    via the leading ellipsis: a (B, N, n_gen) frame (batched forward) and a (N, n_gen) frame (the
    unbatched block / diagnostics path) both flow through.

    The exponentials and the composition :math:`(U E, E^{-1} U^{-1})` run at IEEE fp32 whatever
    ``fp32_matmul_precision`` is (:func:`vfe3.runtime.ieee_fp32_matmul`). The two composition
    products are rounded independently, so under TF32 the stored pair would be inverse only to
    :math:`2^{-10}\kappa`, which breaks the ``CanonicalFrameContext`` contract for ill-conditioned
    frames. These are small block GEMMs; the caller's policy still governs every consumer.
    """
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    compact = (
        compact_blocks
        and group.elementary_block_shape is not None
        and block_dims is not None
        and len(set(block_dims)) == 1
        and group.generator_shape[0] == len(block_dims) * block_dims[0] * block_dims[0]
    )
    if gauge_mode == "trivial":
        # Trivial gauge: exp = I. Build the same per-token factors the dense path would (the
        # caller's guard normally excludes trivial, but keep the container well-formed).
        K = group.generator_shape[-1]
        if compact:
            H, d = len(block_dims), block_dims[0]
            eye_d = torch.eye(d, device=phi.device, dtype=phi.dtype)
            eye = eye_d.expand(*phi.shape[:-1], H, d, d).contiguous()
            return CompactFactoredTransport(
                exp_blocks=eye, inv_blocks=eye, K=K, same_frame_flat_cocycle=True)
        eye_K = torch.eye(K, device=phi.device, dtype=phi.dtype)
        eye = eye_K.expand(*phi.shape[:-1], K, K).contiguous()
        return FactoredTransport(exp_phi=eye, exp_neg_phi=eye, irrep_dims=list(group.irrep_dims),
                                 same_frame_flat_cocycle=True)
    if gauge_mode != "learned":
        raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got {gauge_mode!r}")

    if compact:
        H, d = len(block_dims), block_dims[0]
        # generate_glk_multihead orders the canonical uncoupled block_glk coordinates as
        # H consecutive row-major d x d matrices. The reshape is therefore the exact algebra
        # embedding with the structural zero off-block entries omitted.
        phi_blocks = phi.reshape(*phi.shape[:-1], H, d, d)
        # The former autocast-dtype downcast here (dtype parity with the dense
        # einsum(phi, generators) route) is removed by audit 2026-07-25 F2: it rounded the ALGEBRA
        # element to bf16 before exponentiation, so the fp32 store floor in the exp helper could not
        # recover the already-lost input precision. Parity with the dense route is now maintained at
        # float32, because stable_matrix_exp_pair carries the same floor.
        # The right factor is built FIRST (audit 2026-09-23 #12): consumers apply the composed pair
        # (U E, E^-1 U^-1), so the left call certifies that product inside its own single read, and
        # an escalation of either factor rebuilds the other in float64 so the composition is one
        # consistent float64 pair rather than a float64 factor times a float32-rounded inverse.
        right_pair = None
        if right_phi is not None:
            right_blocks = right_phi.reshape(*right_phi.shape[:-1], H, d, d)

            def _build_right(force_fp64: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
                return _stable_compact_glk_exp_pair(
                    right_blocks,
                    validity_max_norm=validity_max_norm,
                    _force_fp64=force_fp64,
                )
            right_pair = _build_right()
        exp_blocks, inv_blocks = _stable_compact_glk_exp_pair(
            phi_blocks,
            validity_max_norm=validity_max_norm,
            _content_token_ids=_content_token_ids,
            _right_pair=right_pair,
            _force_fp64=right_pair is not None and right_pair[0].dtype == torch.float64,
        )
        if right_pair is not None:
            right_exp, right_inv = (right_pair if right_pair[0].dtype == exp_blocks.dtype
                                    else _build_right(force_fp64=True))
            # Preserve the exponentials' fp32/fp64 storage floor through composition.
            # Autocast rounding here destroys the mutual-inverse frame contract.
            with torch.amp.autocast(phi.device.type, enabled=False):
                exp_blocks = exp_blocks @ right_exp
                inv_blocks = right_inv @ inv_blocks
        return CompactFactoredTransport(
            exp_blocks=exp_blocks, inv_blocks=inv_blocks, K=group.generator_shape[-1],
            same_frame_flat_cocycle=True)

    # Right factor first, as on the compact route above (audit 2026-09-23 #12).
    right_pair = None
    if right_phi is not None:
        right_matrix = _embed_algebra_fp32_floor(right_phi, group)

        def _build_right(force_fp64: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            return stable_matrix_exp_pair(
                right_matrix,
                block_dims=block_dims,
                validity_max_norm=validity_max_norm,
                _force_fp64=force_fp64,
            )
        right_pair = _build_right()
    phi_matrix = _embed_algebra_fp32_floor(phi, group)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, block_dims=block_dims,
        validity_max_norm=validity_max_norm,
        _right_pair=right_pair,
        _force_fp64=right_pair is not None and right_pair[0].dtype == torch.float64,
    )
    if right_pair is not None:
        right_exp, right_inv = (right_pair if right_pair[0].dtype == exp_phi.dtype
                                else _build_right(force_fp64=True))
        with torch.amp.autocast(phi.device.type, enabled=False):
            exp_phi = exp_phi @ right_exp
            exp_neg_phi = right_inv @ exp_neg_phi
    return FactoredTransport(exp_phi=exp_phi, exp_neg_phi=exp_neg_phi, irrep_dims=list(group.irrep_dims),
                             same_frame_flat_cocycle=True)


def _factor_dtype(
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',
) -> torch.dtype:
    r"""Storage dtype of a transport's vertex factors, or of a dense operator; RoPE reports its base."""
    if isinstance(omega, RopeTransport):
        return _factor_dtype(omega.base)
    if isinstance(omega, CompactFactoredTransport):
        return omega.exp_blocks.dtype
    if isinstance(omega, FactoredTransport):
        return omega.exp_phi.dtype
    return omega.dtype


