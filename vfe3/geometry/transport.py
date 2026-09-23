r"""Gauge transport for VFE_3.0 (flat phi-cocycle, Gaussian / location-scale specific).

The flat transport parameterization:
  phi (exp):    Omega_ij = exp(phi_i . G) exp(-phi_j . G) in GL+(K) (det>0).
Belief action: mu -> Omega @ mu, Sigma -> Omega @ Sigma @ Omega^T (sandwich;
diagonal-covariance fast path plus an exact full-covariance congruence); retractions are a separate
module. Gauge-RoPE folds a positional rotation into the transport via :class:`RopeTransport`.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import CompactBlockElement, _equal_diag_blocks
from vfe3.numerics import ValidatedCholeskySolve, validated_cholesky_solve

if TYPE_CHECKING:
    from vfe3.config import VFE3Config


@dataclass
class FactoredTransport:
    r"""The flat phi-cocycle transport in FACTORED form: the per-token (B, N, K, K) vertex
    exponentials, NOT the dense (B, N, N, K, K) pairwise Omega.

    On the flat + block-diagonal-with-equal-blocks path the dense Omega_ij = exp(phi_i) exp(-phi_j)
    is never materialized; instead ``transport_mean`` / ``transport_covariance`` consume this
    container on a fast path that fuses the exps into the contraction (P0 #2, perf doc
    docs/perf/2026-05-31-speedup-opportunities.md): the mean is an EXACT reassociation, and the
    DIAGONAL sandwich factors per head by block-diagonality (a (d, d) operation per head, never
    the full K x K square). A FULL-covariance input rebuilds the dense Omega from the factors
    (byte-identical to ``compute_transport_operators``), so the unfused sandwich is unchanged.

    ``irrep_dims`` (equal blocks, length > 1) drives the per-head slicing of the diagonal cov.

    ``mean_per_head`` (Tier-1 perf toggle, default False = byte-identical): when set,
    ``transport_mean`` contracts each gauge block separately (``_factored_per_head_mean``, the
    mean twin of the per-head diagonal cov) instead of the dense full-K einsum -- the same sum
    with the exactly-zero off-block terms dropped, equal to fp32 reassociation.
    """

    exp_phi:                 torch.Tensor  # (..., N, K, K) exp(phi_i . G)
    exp_neg_phi:             torch.Tensor  # (..., N, K, K) exp(-phi_j . G)
    irrep_dims:              List[int]     # equal block sizes; sum == K, len > 1
    mean_per_head:           bool = False  # transport_mean contracts per gauge block
    same_frame_flat_cocycle: bool = False  # one vertex table supplies U_i and true inverse U_j^-1
    cond_escalation:         bool = False  # opt-in conditioning trigger for the fp64 congruence escalation

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
    inverses. Mean, diagonal-covariance, and full-covariance contractions consume these factors
    without forming a vertex or pairwise ``K x K`` matrix. :meth:`to_dense_omega` is the explicit
    compatibility boundary for legacy consumers that require a dense pairwise operator.
    """

    exp_blocks:              torch.Tensor  # (..., N, H, d, d) stored U_i blocks
    inv_blocks:              torch.Tensor  # (..., N, H, d, d) true U_j^{-1} blocks
    K:                       int
    mean_per_head:           bool = False
    same_frame_flat_cocycle: bool = False
    cond_escalation:         bool = False  # opt-in conditioning trigger for the fp64 congruence escalation

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
        if type(self.mean_per_head) is not bool:
            raise ValueError(
                "mean_per_head must be a bool, got "
                f"{type(self.mean_per_head).__name__}: {self.mean_per_head!r}")
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
            mean_per_head=self.mean_per_head,
            same_frame_flat_cocycle=self.same_frame_flat_cocycle,
            cond_escalation=self.cond_escalation,
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
    effective operator is Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T. ``transport_mean``
    always applies the rotation; ``transport_covariance`` applies it only when ``on_cov`` (the
    means+covariance "full-gauge" regime, which the config gates to full covariance). Means-only
    (``on_cov=False``) leaves the covariance sandwich on the un-rotated ``base`` -- numerically
    identical to no RoPE for the covariance tensor itself, so the diagonal-covariance path stays
    valid. NOTE: under means-only the mean transports under R_i Omega_ij R_j^T but the covariance
    under the bare Omega_ij, so the transported (mu_t, Sigma_t) is NOT a single coherent congruence
    image -- affine/Mahalanobis invariants (e.g. mu^T Sigma^{-1} mu, norms.MahalanobisNorm) are not
    preserved for that belief. The coherent pure path is ``on_cov=True`` (rope_full_gauge), where
    both transform under the same rotated operator.

    ``on_value`` (default True = the coherent single-gauge path) factors the transport into an
    ATTENTION gauge and a VALUE gauge (GL(K)_attention.tex:1909): the attention score
    D_KL(q_i || R_i Omega_ij R_j^T q_j) always carries the rotation, but with ``on_value=False`` the
    value aggregation mu_hat_i = sum_j beta_ij Omega_ij mu_j uses the UN-rotated base transport --
    exactly RoPE's "position-dependent attention, position-independent values" asymmetry. The flag is
    consumed at the GRADIENT layer: ``gradients/oracle.py`` builds the value-gauge coupling energy from
    ``base`` while beta comes from the rotated score energy, and ``gradients/kernels.py`` routes the
    decoupled case to the oracle (beta is no longer the coupling sum's stationary point, so the
    closed-form envelope kernel does not apply). ``transport_mean`` / ``transport_covariance`` always
    honor the rotation (the score path); the value path transports on ``base`` directly.
    """

    base:                    'torch.Tensor | FactoredTransport | CompactFactoredTransport'
    rope:                    torch.Tensor  # (N, K, K) block-diagonal orthogonal rotation
    on_cov:                  bool = False
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
    """
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
            mean_per_head=base.mean_per_head,
            same_frame_flat_cocycle=base.same_frame_flat_cocycle,
            cond_escalation=base.cond_escalation,
        )
    return FactoredTransport(
        exp_phi=base.exp_phi @ rope_t,
        exp_neg_phi=rope @ base.exp_neg_phi,
        irrep_dims=list(base.irrep_dims),
        mean_per_head=base.mean_per_head,
        same_frame_flat_cocycle=base.same_frame_flat_cocycle,
        cond_escalation=base.cond_escalation,
    )


def _rope_dense_omega(
    base: 'torch.Tensor | FactoredTransport | CompactFactoredTransport',
    rope: torch.Tensor,
) -> torch.Tensor:
    r"""Effective dense Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T (full-gauge / dense path).

    LEFT insertion only. The RIGHT insertion's dense form comes from
    :meth:`RopeTransport.score_operator`'s folded container via its own ``to_dense_omega``.
    """
    omega = (
        base.to_dense_omega()
        if isinstance(base, (FactoredTransport, CompactFactoredTransport)) else base
    )                                                                               # (...,N,N,K,K)
    # R_i Omega_ij R_j^T: contract R on the left of the i-axis output and the right (transposed) of j.
    rot = torch.einsum("...ikl,...ijlm,...jnm->...ijkn", rope, omega, rope)
    return rot


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
    "clamp_monitor",
    "exp_fp64_mode",
    "exp_fp64_norm_threshold",
    "materialize",
    "validity_max_norm",
    "exactness_out",
})
_TRANSPORT_NEEDS_MU:    set = set()   # regimes whose Omega builder reads the belief means mu
_TRANSPORT_NEEDS_SIGMA: set = set()   # regimes whose Omega builder reads the belief covariance sigma
_TRANSPORT_BATCH_INDEPENDENT: set = set()   # regimes whose Omega is the SAME for every sequence in the
#                                             batch (depends only on a model parameter, not phi/mu/sigma),
#                                             so the builder returns a batch-collapsed (N,N,K,K) Omega that
#                                             broadcasts downstream instead of a dense (B,N,N,K,K).


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
    returns a batch-collapsed ``(N, N, K, K)`` Omega that ``transport_mean`` / ``transport_covariance``
    broadcast across the batch (the D3 memory collapse). ``_transport`` reads this flag to skip the
    per-sequence ``[0]`` strip it applies to ordinary ``(B, N, N, K, K)`` builders.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an existing
    name silently shadowed the first, so a config-selected seam could dispatch to an unintended
    implementation. Pass ``override=True`` to replace deliberately. The callable and all declarations
    are installed together as one frozen record; the legacy routing sets are then derived from that
    complete record, so stale membership cannot survive an override.
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
        # Compatibility views for existing hot-path membership checks. Their values are derived only
        # from the just-installed complete record; none is inherited from an overridden registration.
        _TRANSPORT_NEEDS_MU.discard(name)
        _TRANSPORT_NEEDS_SIGMA.discard(name)
        _TRANSPORT_BATCH_INDEPENDENT.discard(name)
        if registration.needs_mu:
            _TRANSPORT_NEEDS_MU.add(name)
        if registration.needs_sigma:
            _TRANSPORT_NEEDS_SIGMA.add(name)
        if registration.batch_independent:
            _TRANSPORT_BATCH_INDEPENDENT.add(name)
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
    exp_fp64_mode:           str             = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float           = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool            = False,       # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:       Optional[float] = None,        # opt-in fail-closed pre-clamp chart bound
    **kwargs,                             # tolerated (a future non-flat builder shares this shape)
) -> TransportDict:
    r"""Flat (Regime I) phi-cocycle transport: the registered default.

    A thin adapter forwarding verbatim to :func:`compute_transport_operators`
    (Omega_ij = exp(phi_i) exp(-phi_j) in GL+(K)); bit-identical to calling it directly. Extra
    keyword args are tolerated and ignored so a future stateful builder can share this call shape
    without editing the registry call sites.
    """
    return compute_transport_operators(phi, group, gauge_mode=gauge_mode,
                                       exp_fp64_mode=exp_fp64_mode,
                                       exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                                       clamp_monitor=clamp_monitor,
                                       validity_max_norm=validity_max_norm)


# Frobenius-norm clamp for stable_matrix_exp_pair: above this the returned factor is the surrogate
# exp(max_norm*M/||M||_F), NOT exp(M). Exported so the M-step drift monitor
# (train._warn_phi_transport_clamp) trips at the SAME norm the clamp fires at -- the two cannot
# diverge (audit 2026-07-06 M2).
TRANSPORT_CLAMP_MAX_NORM: float = 20.0

# --- full-covariance CONGRUENCE precision policy (audit 2026-08-06 C1/F3) --------------------
# The full-covariance sandwich Omega Sigma Omega^T hard-coded ``.double()`` at every site with no
# dtype parameter, no module global and no autocast guard -- unlike the KL (full_cov_kl_precision)
# and unlike the diagonal sibling's ``_escalate_if_negative``, this stage had no policy at all.
# It is the largest stage in the step at the live shape: the blocks_only einsum is
# 2*2*B*N^2*H*d^3 = 1.573 GFLOP of float64 over ~1.57 GB, arithmetic intensity 0.83 FLOP/byte, on
# a card whose float64 rate is ~1/64 of float32.
#
# "fp64": the historical unconditional float64 island -- THE DEFAULT, so every run on disk stays
#         bit-reproducible and nothing changes until the user opts in.
# "fp32_escalate": contract in float32 and accept only a finite, symmetric, zero-jitter
#         Cholesky-valid result within the shared residual/condition bounds. Any failed fast
#         certificate recomputes the complete affected structural congruence in float64 and retains
#         float64; an uncertified float64 result raises. The congruence SQUARES cond(Omega), so the
#         certificate is the required accuracy boundary rather than a finiteness-only guard.
_FULL_COV_CONGRUENCE_PRECISIONS = ("fp64", "fp32_escalate")
_FULL_COV_CONGRUENCE_PRECISION: str = "fp64"
_FULL_COV_CONGRUENCE_CERTIFICATION: bool = True


def set_full_cov_congruence_precision(policy: str) -> str:
    r"""Set the process-wide full-covariance congruence precision; returns the previous value."""
    global _FULL_COV_CONGRUENCE_PRECISION
    if policy not in _FULL_COV_CONGRUENCE_PRECISIONS:
        raise ValueError(
            f"full_cov_congruence_precision must be one of {_FULL_COV_CONGRUENCE_PRECISIONS}, "
            f"got {policy!r}")
    previous = _FULL_COV_CONGRUENCE_PRECISION
    _FULL_COV_CONGRUENCE_PRECISION = policy
    return previous


def full_cov_congruence_precision() -> str:
    r"""Return the active full-covariance congruence precision policy."""
    return _FULL_COV_CONGRUENCE_PRECISION


def set_full_cov_congruence_certification(enabled: bool) -> bool:
    r"""Set strict post-congruence certification; returns the previous value."""
    if not isinstance(enabled, bool):
        raise ValueError(f"full covariance congruence certification must be bool, got {enabled!r}")
    global _FULL_COV_CONGRUENCE_CERTIFICATION
    previous = _FULL_COV_CONGRUENCE_CERTIFICATION
    _FULL_COV_CONGRUENCE_CERTIFICATION = enabled
    return previous


def full_cov_congruence_certification() -> bool:
    r"""Return whether full pair-covariance fields receive the strict SPD certificate."""
    return _FULL_COV_CONGRUENCE_CERTIFICATION


def _congruence_failure_details(
    matrix:      torch.Tensor,              # (..., d, d) failed working representation
    certificate: ValidatedCholeskySolve,
) -> str:
    r"""Describe the first failed matrix and bounds, only on the terminal failure path."""
    with torch.no_grad():
        failed = ~certificate.certified
        index = tuple(failed.nonzero(as_tuple=False)[0].tolist())
        dimension = matrix.shape[-1]
        residual_tol = 64.0 * dimension * torch.finfo(matrix.dtype).eps
        return (
            f"dtype={matrix.dtype}, shape={tuple(matrix.shape)}, "
            f"failed_matrices={int(failed.sum())}/{failed.numel()}, "
            f"first_failed_index={index}, "
            f"finite={bool(torch.isfinite(matrix[index]).all())}, "
            f"condition={float(certificate.condition[index]):.6g}, "
            f"condition_limit={1.0 / residual_tol:.6g}, "
            f"symmetry_residual={float(certificate.symmetry_residual[index]):.6g}, "
            f"factor_residual={float(certificate.factor_residual[index]):.6g}, "
            f"residual_tol={residual_tol:.6g}"
        )


def _certify_full_congruence(
    out:     torch.Tensor,               # contraction result at the working dtype
    work:    torch.dtype,                # the dtype it was contracted in
    recompute: "Callable[[torch.dtype], torch.Tensor]",
    *,
    source_dtype: torch.dtype,
    retain_full_precision: bool = False,
) -> torch.Tensor:
    r"""Authoritative post-congruence boundary for every full-covariance structural route.

    Certification is zero-jitter: finiteness alone cannot accept a symmetric-but-indefinite or
    accuracy-uncertain covariance. A working result may return at source dtype only when the cast
    candidate independently certifies there. If a certified float64 working result loses its
    certificate on cast, float64 is retained. Reduced dtypes outside the certificate API's
    float32/float64 domain are treated as uncertifiable and retain the certified working tensor.
    A failed fast representation recomputes and retains float64.
    """
    if not _FULL_COV_CONGRUENCE_CERTIFICATION:
        if retain_full_precision or out.dtype == source_dtype:
            return out
        return out.to(source_dtype)
    certificate = validated_cholesky_solve(out)
    if bool(certificate.certified.all()):
        if retain_full_precision or out.dtype == source_dtype:
            return out
        cast_candidate = out.to(source_dtype)
        if cast_candidate.dtype not in (torch.float32, torch.float64):
            # The shared validator intentionally rejects reduced dtypes. Do not weaken it or call
            # it outside its contract; retain the already certified working representation.
            return out
        cast_certificate = validated_cholesky_solve(cast_candidate)
        if bool(cast_certificate.certified.all()):
            return cast_candidate
        # ``out`` itself is already certified. A failed lower-precision representation must not
        # replace it; retaining this tensor also preserves the direct high-precision autograd path.
        if out.dtype is torch.float64:
            return out
    if work is torch.float64:
        raise FloatingPointError(
            "float64 congruence could not be certified as finite, symmetric, zero-jitter SPD "
            "within the residual and conditioning bounds; "
            + _congruence_failure_details(out, certificate))
    high_precision = recompute(torch.float64)
    high_certificate = validated_cholesky_solve(high_precision)
    if not bool(high_certificate.certified.all()):
        raise FloatingPointError(
            "congruence failed fast validation and its float64 recomputation could not be "
            "certified as finite, symmetric, zero-jitter SPD within the residual and "
            "conditioning bounds; "
            + _congruence_failure_details(high_precision, high_certificate))
    return high_precision


def _congruence_compute_dtype(source_dtype: torch.dtype) -> torch.dtype:
    r"""Working dtype for a full-covariance congruence under the active policy.

    A float64 SOURCE always contracts in float64 regardless of policy -- the policy governs only
    whether a float32 source is promoted, which is what the historical island did unconditionally.
    """
    if _FULL_COV_CONGRUENCE_PRECISION == "fp64" or source_dtype == torch.float64:
        return torch.float64
    return torch.float32

# Process-wide "the exponential clamp has fired at least once" latch (audit 2026-07-25 F2). The clamp
# returns a surrogate operator rather than exp(M), so its first occurrence is reported even when the
# per-build clamp_monitor is off; the latch keeps the default path from paying a host sync forever.
_CLAMP_SURROGATE_WARNED: bool = False

# Latch for the diagonal congruence's float64 escalation (audit 2026-07-25 F3): report the first
# escalation, then stay quiet so a persistently ill-conditioned run does not flood the log.
_DIAG_CONGRUENCE_NEGATIVE_WARNED: bool = False


def _escalate_if_negative(
    out:        torch.Tensor,                          # (..., N, N, H, d) working-dtype congruence
    reduce:     'Callable[[torch.dtype], torch.Tensor]',  # recompute the same reduction at a given dtype

    *,
    exp_blocks: Optional[torch.Tensor] = None,         # (..., H, d, d) vertex blocks, for the accuracy trigger

) -> torch.Tensor:
    r"""Recompute a diagonal congruence in float64 iff the working-dtype result went negative.

    The factored regrouping trades the manifestly nonnegative ``sum_l Omega_kl^2 sigma_l`` for a
    quadratic form with mixed-sign intermediates, so cancellation can drive an output variance below
    zero. That is not a rounding nuisance: a negative variance reaches the divergence kernel's
    ``clamp(min=eps)``, which inverts that key's precision weight by ~6 orders of magnitude, saturates
    ``E_ij`` at ``kl_max``, and silently drops the pair from the coupling via ``pair_mask``.

    Measured, the working dtype is not the problem in the common case -- float32's median relative
    error against a float64 sum-of-squares reference stays ~1.5e-07 at EVERY conditioning tested. It
    is a tail effect: min ``sigma_t`` is comfortably positive up to cond(U) ~ 1e2 and reaches
    -1.51e+03 at cond 3.5e3 and -1.27e+07 at cond 7.1e4. Since the nonnegativity test is a single
    reduction the floor needs regardless, the precision decision is made from the DATA rather than
    from a static toggle the caller would have to guess: pay float32 speed in the common case, and pay
    float64 exactly on the calls that actually lose nonnegativity. This mirrors the escalating-ridge
    ladder :func:`safe_cholesky` already uses rather than upcasting unconditionally.

    **Positivity is necessary but NOT sufficient** (audit 2026-07-27). Cancellation destroys accuracy
    well before it destroys sign, so a sign-only test passes results that are order-unity wrong with
    every entry strictly positive. Measured on trained production weights (K=210, H=7, d=30) with the
    sigma dynamic range set to the run's own recorded ``belief_cond``: the guard read ``amin``
    between +0.69 and +1.00 -- comfortably positive, not marginal -- while the worst transported
    variance was 64% wrong. It was also perverse in both directions: it escalated on a case already
    accurate to 8e-6 and passed a 9.2%-wrong entry at ``amin = +1.04e-4``.

    So the trigger is now ``amin < 0`` OR a conditioning proxy. For a quadratic form
    ``sum_l Omega_kl^2 sigma_l`` the relevant backward-error scale is ``cond(Omega)^2 * eps_work``
    (Higham 2002, section 3.1), and ``cond(Omega_ij) <= cond(exp_phi_i)^2``, so the per-block vertex
    condition number bounds it. This matters most where the codebase is heading rather than where it
    is: the same tail-selection procedure gives a worst relative error of 9.9e-04 at K=20 (d=10) and
    6.4e-01 at K=210 (d=30) -- three orders of magnitude over one decade of K, on a path swept toward
    K=300. ``exp_blocks`` is optional so direct callers that do not hold the vertex factors keep the
    old sign-only behavior rather than failing.
    """
    global _DIAG_CONGRUENCE_NEGATIVE_WARNED
    with torch.no_grad():
        worst = float(out.amin())
        ill_conditioned = False
        if worst >= 0.0 and exp_blocks is not None and out.dtype != torch.float64:
            # svdvals on the (H, d, d) vertex blocks only -- O(H d^3) once, not per pair.
            sv = torch.linalg.svdvals(exp_blocks.detach().float())
            cond = (sv[..., 0] / sv[..., -1].clamp(min=torch.finfo(torch.float32).tiny)).amax()
            # cond(Omega)^2 ~ cond(exp_phi)^4; escalate once the predicted relative error clears 1e-3.
            ill_conditioned = bool(cond ** 4 * torch.finfo(out.dtype).eps > 1e-3)
    if worst >= 0.0 and not ill_conditioned:
        return out
    if ill_conditioned and worst >= 0.0:
        return reduce(torch.float64)                  # accurate-but-positive path: escalate silently
    if not _DIAG_CONGRUENCE_NEGATIVE_WARNED:
        import warnings
        _DIAG_CONGRUENCE_NEGATIVE_WARNED = True
        warnings.warn(
            f"diagonal congruence lost nonnegativity at {out.dtype} (min={worst:.6g}); recomputing "
            "this congruence in float64. The factored quadratic form has mixed-sign intermediates, so "
            "this indicates an ill-conditioned transport -- reduce ||phi|| or set "
            "transport_chart_max_norm if it persists. Warned once for this process.",
            RuntimeWarning, stacklevel=3,
        )
    return reduce(torch.float64)


def stable_matrix_exp_pair(
    matrix:                  torch.Tensor,       # (..., d, d) Lie-algebra matrices

    *,
    exp_fp64_mode:           str                 = "dim",    # float64-island keying: 'dim' (dimension rule) | 'norm'
    max_norm:                float               = TRANSPORT_CLAMP_MAX_NORM,
    exp_fp64_norm_threshold: float               = 5.0,      # 'norm' mode: upcast when max clamped block ||M||_F >= this
    dim_threshold:           int                 = 20,
    skew_symmetric:          bool                = False,
    only_forward:            bool                = False,
    clamp_monitor:           bool                = False,    # opt-in: warn when the Frobenius clamp fires (host sync)
    block_dims:              Optional[List[int]] = None,     # per-block sizes (sum==d) for a block-diagonal M
    exp_dim:                 Optional[int]       = None,     # dimension for the float64-island decision (None -> d)
    validity_max_norm:       Optional[float]     = None,     # opt-in fail-closed pre-clamp chart bound
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Represented exp(M) and optionally its float64-computed inverse.

    Frobenius-norm clamp + float64 upcast keep matrix_exp stable for large ||M||.
    The paired negative factor is the inverse of the represented forward factor, not a separately
    rounded evaluation of exp(-M), so the stored pair shares one finite-precision group element.

    SAFEGUARD, NOT THE EXACT OPERATOR: when ``||M||_F > max_norm`` the matrix is rescaled to
    ``max_norm``, so the returned factor is ``exp(max_norm * M/||M||_F)``, NOT ``exp(M)`` -- the
    singular values / determinant of the returned operator differ from the true exponential. This
    is a stability clamp on extreme inputs only; keep ||phi|| below
    ``max_norm`` to stay exact. The per-call runtime monitor is OFF by default: detecting
    activation needs a tensor reduction (a host sync) on this hot path, which the perf budget
    avoids. Pass ``clamp_monitor=True`` (opt-in diagnostic) to accept that sync and emit a
    RuntimeWarning whenever the clamp fires -- the returned factor is then a surrogate, not exp(M).

    ``block_dims`` (audit 4b): when M is block-diagonal with these blocks (e.g. block_glk's
    GL(d_head)^H), exp(M) is exactly block-diagonal with the per-block exponentials, so each
    d_head x d_head block is exponentiated independently -- an O(H * d_head^3) cost instead of
    O(K^3) for the full K x K. The result is BIT-equivalent to the full exp (the global
    Frobenius clamp is applied to the WHOLE matrix first, and each block keeps the dtype the
    full-K path would pick, so neither the scale nor the precision changes). ``None`` (a single
    block, a cross-coupled basis, or a skew group) takes the full-matrix path unchanged.

    ``exp_dim`` (audit 2026-06-10 F8c, default None = unchanged): an explicit override of the
    dimension the float64-island decision keys on. By DEFAULT the per-block path keeps the
    full-K dtype so blocking never changes precision (the bit-equivalence pin above). A caller
    whose conditioning argument lives at the BLOCK scale may pass ``exp_dim=max(block_dims)`` to run
    small blocks in fp32 instead of upcasting the whole batch to float64 at K >= dim_threshold.

    ``exp_fp64_mode`` (Tier-1 toggle; default 'dim' = the long-standing dimension rule above,
    untouched): 'norm' keys the float64 island on the max CLAMPED block Frobenius norm instead of
    the dimension -- the conditioning argument for matrix_exp is a NORM argument (fp32 matrix_exp
    is ~1e-7 accurate at the small block norms the phi retraction guarantees, at any block dim),
    so small-norm blocks stay fp32 while the fp64 island stays REACHABLE for genuinely large
    norms (upcast when max ||M_block||_F >= ``exp_fp64_norm_threshold``). Costs one host sync (a
    scalar norm compare) per call; the clamp / monitor behavior above is identical in both modes.
    """
    # Global Frobenius clamp on the FULL matrix (one scale for all blocks) -- identical to the
    # un-blocked path, so block slicing below cannot change the operator. The norm/scale is kept
    # OUT of the autograd graph (no_grad): the clamp is a numerical SAFEGUARD, not part of the
    # modeled operator, and differentiating through the rescale (a) biases the gradient toward
    # the clamped surrogate where the clamp is active and (b) makes the norm's DOUBLE-backward
    # NaN on exactly-zero matrices (a zero gauge frame, reached by
    # the unrolled oracle's create_graph path). Where the clamp is inactive (scale == 1.0 --
    # any ||phi|| < max_norm) multiplying by the detached
    # constant 1.0 is byte-identical to the previous through-graph multiply.
    if validity_max_norm is not None and (
        not math.isfinite(validity_max_norm) or validity_max_norm <= 0.0
    ):
        raise ValueError(
            "validity_max_norm must be None or finite and positive, got "
            f"{validity_max_norm!r}"
        )

    with torch.no_grad():
        raw_mat_norm = matrix.norm(dim=(-2, -1), keepdim=True)
        if validity_max_norm is not None and bool((~torch.isfinite(raw_mat_norm)).any()):
            raise ValueError(
                "transport chart validity bound encountered a nonfinite matrix norm"
            )
        if validity_max_norm is not None and bool((raw_mat_norm > validity_max_norm).any()):
            observed = float(raw_mat_norm.max())
            raise ValueError(
                "transport chart validity bound exceeded before matrix-exponential clamp: "
                f"observed ||M||_F={observed:.6g} > {validity_max_norm:.6g}"
            )
        mat_norm = raw_mat_norm.clamp(min=1e-8)
        scale = (max_norm / mat_norm).clamp(max=1.0)
        # The clamp substitutes exp(M * scale) for exp(M) -- a DIFFERENT operator, not a rounding of
        # the intended one (measured ~98% relative error at ||M||_F = 30). Before audit 2026-07-25 F2
        # this was announced only under the opt-in clamp_monitor, so the default configuration could
        # model a surrogate transport with no signal whatsoever. The verbose per-build report stays
        # gated on clamp_monitor; the FIRST occurrence is now always reported. The reduction costs one
        # host sync per build (one per forward on the internally shared transport route), so the
        # default path stops paying it once it has fired. transport_chart_max_norm remains the
        # fail-closed alternative and its default is deliberately unchanged: every value the
        # pullback_group validator admits (5.0 < bound < 20.0) would hard-abort runs whose observed
        # ||M||_F already reaches 18, so failing closed stays the caller's explicit choice.
        global _CLAMP_SURROGATE_WARNED
        if clamp_monitor or not _CLAMP_SURROGATE_WARNED:
            frac = (scale < 1.0).float().mean()
            if bool(frac > 0):
                import warnings
                _CLAMP_SURROGATE_WARNED = True
                once = ("" if clamp_monitor else
                        " Warned once for this process; set transport_clamp_monitor=True to report "
                        "every occurrence, or transport_chart_max_norm to fail closed instead.")
                warnings.warn(
                    f'stable_matrix_exp_pair: Frobenius clamp active on {float(frac):.1%} of matrices '
                    f'(max_norm={max_norm}); the returned factor is a SURROGATE exp(M*scale), NOT '
                    f'exp(M).{once}',
                    RuntimeWarning, stacklevel=2,
                )
    matrix = matrix * scale

    d = matrix.shape[-1]
    # float32 storage floor, matching _stable_compact_glk_exp_pair (audit 2026-07-25 F2). The dense
    # route reaches here with an autocast-eligible einsum(phi, generators) result, so without this the
    # dense and compact representations would only agree on their (reduced) storage dtype by both
    # being wrong. Parity between the two paths is preserved -- at float32 rather than at bf16.
    orig_dtype = matrix.dtype if matrix.dtype in (torch.float32, torch.float64) else torch.float32
    if exp_fp64_mode == "norm":
        # Norm-keyed float64 island (Tier-1 toggle): upcast ONLY when the max CLAMPED block
        # Frobenius norm reaches the threshold. Computed from the already-clamped matrices (the
        # clamp above ran first), so the keying norm is the norm actually exponentiated; the
        # island stays reachable for genuinely large norms. One host sync (the bool compare),
        # opt-in by mode.
        with torch.no_grad():
            if block_dims is not None and len(block_dims) > 1:
                start = 0
                key_norm = matrix.new_zeros(())
                for blk in block_dims:
                    end = start + blk
                    key_norm = torch.maximum(
                        key_norm, matrix[..., start:end, start:end].norm(dim=(-2, -1)).max())
                    start = end
            else:
                key_norm = (mat_norm * scale).max()      # clamped full-matrix norm (single block)
        up_dtype = torch.float64 if bool(key_norm >= exp_fp64_norm_threshold) else torch.float32
    elif exp_fp64_mode == "dim":
        # 'dim': the long-standing dimension rule. The full-K path's dtype choice; the per-block
        # path forces the SAME dtype so a small block (d_head < dim_threshold) does not silently
        # drop to float32 and drift from the full exp. exp_dim (when given) overrides the keying
        # dimension -- see the docstring.
        d_eff = exp_dim if exp_dim is not None else d
        with torch.no_grad():
            large_skew = skew_symmetric and bool(
                (mat_norm * scale).max() >= exp_fp64_norm_threshold
            )
        up_dtype = torch.float64 if d_eff >= dim_threshold or large_skew else torch.float32
    else:
        raise ValueError(f"exp_fp64_mode must be 'dim' or 'norm', got {exp_fp64_mode!r}")

    # Keyed to the TENSOR's device (audit 2026-07-05 m10): the old 'cuda' literal left the island
    # open under a CPU autocast context (torch.amp.autocast('cpu', bf16)), which _amp_context
    # deliberately supports -- matrix_exp would then run in bf16 on CPU-AMP runs.
    with torch.amp.autocast(matrix.device.type, enabled=False):
        matrix_up = matrix.to(up_dtype).contiguous()

        if block_dims is not None and len(block_dims) > 1:
            exp_pos = _blockwise_matrix_exp(matrix_up, block_dims).to(orig_dtype)
            if only_forward:
                exp_neg = None
            else:
                exp_neg = _blockwise_group_inverse(exp_pos, block_dims)
            return exp_pos, exp_neg

        exp_pos = torch.linalg.matrix_exp(matrix_up).to(orig_dtype)
        if only_forward:
            exp_neg = None
        else:
            exp_neg = _checked_group_inverse(exp_pos)
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
    out = torch.zeros_like(matrix)
    if len(set(block_dims)) == 1 and len(block_dims) > 1:
        H, d = len(block_dims), block_dims[0]
        batch = matrix.shape[:-2]
        matrix_view = matrix.reshape(*batch, H, d, H, d)
        blocks = torch.diagonal(
            matrix_view,
            dim1=-4,
            dim2=-2,
        ).movedim(-1, -3).contiguous()
        inverses = _checked_group_inverse(blocks)
        out_view = out.reshape(*batch, H, d, H, d)
        torch.diagonal(out_view, dim1=-4, dim2=-2).copy_(inverses.movedim(-3, -1))
        return out

    start = 0
    for dim in block_dims:
        end = start + dim
        block = matrix[..., start:end, start:end].contiguous()
        out[..., start:end, start:end] = _checked_group_inverse(block)
        start = end
    return out


def compute_transport_operators(
    phi:        torch.Tensor,             # (B, N, n_gen) gauge frames
    group:      GaugeGroup,               # supplies generators, skew flag, irrep_dims

    *,
    gauge_mode:              str             = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str             = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float           = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool            = False,       # opt-in: warn when the exp Frobenius clamp fires
    validity_max_norm:       Optional[float] = None,        # opt-in fail-closed pre-clamp chart bound
) -> TransportDict:
    r"""phi/exp transport Omega_ij = exp(phi_i) @ exp(-phi_j) in GL+(K).

    Flat (Regime I) transport operator construction. 'trivial' returns Omega = I.
    Returns 'exp_phi' (B,N,K,K), 'exp_neg_phi' (B,N,K,K), 'Omega' (B,N,N,K,K).
    The 'constant' gauge mode is intentionally NOT supported (it would require a
    per-head learned Omega parameter, which this no-NN design does not have);
    'constant' raises ValueError.

    Conditioning note (audit 2026-06-09 overnight F26): for the NON-COMPACT groups
    (glk/block_glk/sp_n; skew_symmetric=False) Omega is not orthogonal and cond(Omega)
    grows like exp(2 ||phi_matrix||); at the phi retraction's default max_norm=5.0 a
    draw can reach cond ~1e7-1e10, and the full-covariance sandwich Omega Sigma Omega^T
    SQUARES it. ``transport_covariance`` evaluates that full-covariance sandwich in a
    float64 island (audit 2026-06-13 M4) so the squared conditioning no longer loses all
    fp32 digits; compact so towers give orthogonal Omega (cond = 1) and are unaffected.
    Omega itself is still built at the working dtype here, so for extreme draws prefer a
    compact group / diagonal family or bound phi via the retraction max_norm.
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
    # -> irrep_dims [d_head]*H); single-block ([K]: glk, so_k, cross-coupled) takes the full path.
    # exp_dim keys the float64-island decision on the dimension actually exponentiated -- the
    # per-head block (audit F8c) -- and the always-fp32
    # per-block exp at d_head < 20. The conditioning argument lives at the block scale: the
    # retraction bounds ||phi|| (coords) by max_norm=5.0, so each block's Frobenius norm is far
    # inside fp32 matrix_exp's exact regime; without the override every flat run at K >= 20 paid
    # a (B, N, K, K) float64 upcast (vram audit 2026-06-10: ~0.4 GB of f64 transients per build
    # plus fp64-throughput matrix_exp on a consumer GPU) for blocks that never needed it.
    block_dims = group.irrep_dims if len(group.irrep_dims) > 1 else None
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric, block_dims=block_dims,
        exp_dim=(max(block_dims) if block_dims is not None else None),
        exp_fp64_mode=exp_fp64_mode, exp_fp64_norm_threshold=exp_fp64_norm_threshold,
        # m16: matrix_exp of a skew matrix is exactly orthogonal at ANY norm, so the Frobenius clamp
        # only gratuitously shortens the rotation on the pure so_n/so_k tower path (whose retraction
        # caps ||phi|| in COORDINATES, under-bounding the embedded norm). Disable it for skew; the
        # non-compact groups (glk/block_glk/sp_n) keep the clamp as a genuine exp-overflow safeguard.
        max_norm=(float("inf") if group.skew_symmetric else TRANSPORT_CLAMP_MAX_NORM),
        clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm,
    )
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
    r"""True float64 inverse with immediate nonfinite/singular failure and dtype restoration.

    When ``residual_tol`` is supplied, the STORED pair is additionally verified against
    :math:`\lVert U U^{-1} - I \rVert_\infty \le \texttt{residual\_tol}`. The finiteness checks alone
    cannot see precision loss: the inverse is exact in float64, so every way this pair stops being a
    group-inverse pair is a rounding effect of the storage dtype, which stays perfectly finite
    (audit 2026-07-25 F2). The check costs one reduction and a host sync, so it is opt-in and off on
    the training hot path.
    """
    if validate:
        with torch.no_grad():
            if not bool(torch.isfinite(omega).all()):
                raise FloatingPointError(
                    "omega group element contains nonfinite values before inversion")
    try:
        with torch.amp.autocast(omega.device.type, enabled=False):
            inverse64 = torch.linalg.inv(omega.double())
    except RuntimeError as exc:
        raise ValueError("omega group element is singular and cannot be inverted") from exc
    if validate:
        with torch.no_grad():
            if not bool(torch.isfinite(inverse64).all()):
                raise FloatingPointError(
                    "omega group-element inverse is nonfinite; the element is singular or "
                    "numerically unrepresentable")
    inverse = inverse64.to(omega.dtype)
    if validate:
        with torch.no_grad():
            if not bool(torch.isfinite(inverse).all()):
                raise FloatingPointError(
                    f"omega group-element inverse is nonfinite after conversion to {omega.dtype}")
            if residual_tol is not None:
                eye = torch.eye(omega.shape[-1], device=omega.device, dtype=torch.float64)
                residual = float(
                    (omega.double() @ inverse.double() - eye).abs().amax()
                )
                if not (residual <= residual_tol):
                    raise FloatingPointError(
                        "stored omega group-element pair violates the inverse identity: "
                        f"||U U^-1 - I||_inf={residual:.6g} > residual_tol={residual_tol:.6g} "
                        f"(storage dtype {omega.dtype}); the represented element and its stored "
                        "inverse no longer form a group-inverse pair, so the transport cocycle is "
                        "not exact"
                    )
    return inverse


def _stable_compact_glk_exp_pair(
    blocks:                  torch.Tensor,       # (..., H, d, d) block-diagonal gl(d)^H element

    *,
    exp_fp64_mode:           str             = "dim",     # float64-island keying: 'dim' | 'norm'
    exp_fp64_norm_threshold: float           = 5.0,
    clamp_monitor:           bool            = False,
    validity_max_norm:       Optional[float] = None,
    _content_token_ids:      Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exponentiate packed blocks and invert their represented values in float64.

    Private token IDs certify a built-in table lookup. Pre-exp decisions still see every
    occurrence; only exp/inverse work is shared, then gathered before composition. The
    table gradient is equivalent in real arithmetic, with a different reduction order.
    """
    if validity_max_norm is not None and (
        not math.isfinite(validity_max_norm) or validity_max_norm <= 0.0
    ):
        raise ValueError(
            "validity_max_norm must be None or finite and positive, got "
            f"{validity_max_norm!r}"
        )
    # Keep the existing scalar predicates and reduction precision, but transfer them together.
    decisions = {}
    with torch.no_grad():
        raw_mat_norm = blocks.square().sum(dim=(-3, -2, -1), keepdim=True).sqrt()
        if validity_max_norm is not None:
            decisions["nonfinite"] = (~torch.isfinite(raw_mat_norm)).any()
            decisions["outside_chart"] = (raw_mat_norm > validity_max_norm).any()
        mat_norm = raw_mat_norm.clamp(min=1e-8)
        scale = (TRANSPORT_CLAMP_MAX_NORM / mat_norm).clamp(max=1.0)
        if clamp_monitor:
            decisions["clamp_fraction"] = (scale < 1.0).float().mean()
    blocks = blocks * scale
    if exp_fp64_mode == "norm":
        with torch.no_grad():
            key_norm = blocks.norm(dim=(-2, -1)).max()
            decisions["use_fp64"] = key_norm >= exp_fp64_norm_threshold
    host = dict(zip(decisions, torch.stack(tuple(decisions.values())).tolist())) if decisions else {}
    # Preserve the original exception/warning precedence after the single healthy-path read.
    if host.get("nonfinite", False):
        raise ValueError(
            "transport chart validity bound encountered a nonfinite matrix norm"
        )
    if host.get("outside_chart", False):
        observed = float(raw_mat_norm.max())
        raise ValueError(
            "transport chart validity bound exceeded before matrix-exponential clamp: "
            f"observed ||M||_F={observed:.6g} > {validity_max_norm:.6g}"
        )
    if host.get("clamp_fraction", 0.0) > 0:
        import warnings
        warnings.warn(
            f'stable_matrix_exp_pair: Frobenius clamp active on {host["clamp_fraction"]:.1%} of matrices '
            f'(max_norm={TRANSPORT_CLAMP_MAX_NORM}); returned factor is a surrogate, not exp(M).',
            RuntimeWarning, stacklevel=2,
        )

    d = blocks.shape[-1]
    # Preserve both the represented fp32 storage floor and the configured interior precision.
    store_dtype = blocks.dtype if blocks.dtype in (torch.float32, torch.float64) else torch.float32
    if exp_fp64_mode == "norm":
        up_dtype = torch.float64 if host["use_fp64"] else torch.float32
    elif exp_fp64_mode == "dim":
        up_dtype = torch.float64 if d >= 20 else torch.float32
    else:
        raise ValueError(f"exp_fp64_mode must be 'dim' or 'norm', got {exp_fp64_mode!r}")

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
        blocks_up = blocks.to(up_dtype).contiguous()
        exp_pos = torch.linalg.matrix_exp(blocks_up).to(store_dtype)
        exp_neg = _checked_group_inverse(exp_pos)
    if inverse_index is not None:
        exp_pos = exp_pos.index_select(0, inverse_index).reshape(occurrence_shape)
        exp_neg = exp_neg.index_select(0, inverse_index).reshape(occurrence_shape)
    return exp_pos, exp_neg


def build_factored_transport(
    phi:        torch.Tensor,             # (..., N, n_gen) gauge frames (optional leading batch axis)
    group:      GaugeGroup,               # block-diagonal with equal blocks (len(irrep_dims) > 1)

    *,
    gauge_mode:              str                    = "learned",   # 'learned' (Regime I flat) or 'trivial'
    exp_fp64_mode:           str                    = "dim",       # stable_matrix_exp_pair float64-island keying ('dim' | 'norm')
    exp_fp64_norm_threshold: float                  = 5.0,         # 'norm' mode: max clamped block ||M||_F upcast threshold
    clamp_monitor:           bool                   = False,       # opt-in: warn when the exp Frobenius clamp fires
    mean_per_head:           bool                   = False,       # container flag: transport_mean contracts per gauge block
    cond_escalation:         bool                   = False,       # container flag: opt-in conditioning trigger for the fp64 congruence escalation
    compact_blocks:          bool                   = False,       # canonical block_glk: retain (..., N, H, d, d) factors
    validity_max_norm:       Optional[float]        = None,        # opt-in fail-closed pre-clamp chart bound
    right_phi:               Optional[torch.Tensor] = None,        # (..., N, n_gen) exact right factor exp(Y)
    _content_token_ids:      Optional[torch.Tensor] = None,        # private certified table-lookup identity
) -> 'CompactFactoredTransport | FactoredTransport':
    r"""Flat phi-cocycle transport in FACTORED form, skipping the dense (..., N, N, K, K) Omega.

    Builds only the per-token vertex exponentials exp(phi_i), exp(-phi_j) (the same factors
    ``compute_transport_operators`` builds) and the ``ikl,jlm->ijkm`` Omega einsum is NEVER run.
    With ``compact_blocks=True`` on canonical uncoupled ``block_glk``, both the algebra embedding
    and the returned factors retain ``(..., N, H, d, d)`` storage; every other group and the
    default keep the legacy dense-vertex :class:`FactoredTransport` representation.
    The pairwise contraction is deferred into ``transport_mean`` / ``transport_covariance``'s fast
    path (P0 #2). Caller guards this to the flat + block-diagonal-with-equal-blocks path; here it
    only requires the exps, which the block-diagonal exp machinery already produces. Rank-agnostic
    via the leading ellipsis: a (B, N, n_gen) frame (batched forward) and a (N, n_gen) frame (the
    unbatched block / diagnostics path) both flow through.
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
                exp_blocks=eye, inv_blocks=eye, K=K, mean_per_head=mean_per_head,
                same_frame_flat_cocycle=True, cond_escalation=cond_escalation)
        eye_K = torch.eye(K, device=phi.device, dtype=phi.dtype)
        eye = eye_K.expand(*phi.shape[:-1], K, K).contiguous()
        return FactoredTransport(exp_phi=eye, exp_neg_phi=eye, irrep_dims=list(group.irrep_dims),
                                 mean_per_head=mean_per_head, same_frame_flat_cocycle=True,
                                 cond_escalation=cond_escalation)
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
        exp_blocks, inv_blocks = _stable_compact_glk_exp_pair(
            phi_blocks,
            exp_fp64_mode=exp_fp64_mode,
            exp_fp64_norm_threshold=exp_fp64_norm_threshold,
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
            _content_token_ids=_content_token_ids,
        )
        if right_phi is not None:
            right_blocks = right_phi.reshape(*right_phi.shape[:-1], H, d, d)
            right_exp, right_inv = _stable_compact_glk_exp_pair(
                right_blocks,
                exp_fp64_mode=exp_fp64_mode,
                exp_fp64_norm_threshold=exp_fp64_norm_threshold,
                clamp_monitor=clamp_monitor,
                validity_max_norm=validity_max_norm,
            )
            # Preserve the exponentials' fp32/fp64 storage floor through composition.
            # Autocast rounding here destroys the mutual-inverse frame contract.
            with torch.amp.autocast(phi.device.type, enabled=False):
                exp_blocks = exp_blocks @ right_exp
                inv_blocks = right_inv @ inv_blocks
        return CompactFactoredTransport(
            exp_blocks=exp_blocks, inv_blocks=inv_blocks, K=group.generator_shape[-1],
            mean_per_head=mean_per_head, same_frame_flat_cocycle=True,
            cond_escalation=cond_escalation)

    phi_matrix = _embed_algebra_fp32_floor(phi, group)
    # exp_dim: same block-scale float64-island keying as compute_transport_operators (see the
    # comment there) -- the factored hot path is exactly where the (B, N, K, K) f64 upcast hurt.
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(
        phi_matrix, skew_symmetric=group.skew_symmetric, block_dims=block_dims,
        exp_dim=(max(block_dims) if block_dims is not None else None),
        exp_fp64_mode=exp_fp64_mode, exp_fp64_norm_threshold=exp_fp64_norm_threshold,
        # m16: matrix_exp of a skew matrix is exactly orthogonal at ANY norm, so the Frobenius clamp
        # only gratuitously shortens the rotation on the pure so_n/so_k tower path (whose retraction
        # caps ||phi|| in COORDINATES, under-bounding the embedded norm). Disable it for skew; the
        # non-compact groups (glk/block_glk/sp_n) keep the clamp as a genuine exp-overflow safeguard.
        max_norm=(float("inf") if group.skew_symmetric else TRANSPORT_CLAMP_MAX_NORM),
        clamp_monitor=clamp_monitor,
        validity_max_norm=validity_max_norm,
    )
    if right_phi is not None:
        right_matrix = _embed_algebra_fp32_floor(right_phi, group)
        right_exp, right_inv = stable_matrix_exp_pair(
            right_matrix,
            skew_symmetric=group.skew_symmetric,
            block_dims=block_dims,
            exp_dim=(max(block_dims) if block_dims is not None else None),
            exp_fp64_mode=exp_fp64_mode,
            exp_fp64_norm_threshold=exp_fp64_norm_threshold,
            max_norm=(float("inf") if group.skew_symmetric else TRANSPORT_CLAMP_MAX_NORM),
            clamp_monitor=clamp_monitor,
            validity_max_norm=validity_max_norm,
        )
        with torch.amp.autocast(phi.device.type, enabled=False):
            exp_phi = exp_phi @ right_exp
            exp_neg_phi = right_inv @ exp_neg_phi
    return FactoredTransport(exp_phi=exp_phi, exp_neg_phi=exp_neg_phi, irrep_dims=list(group.irrep_dims),
                             mean_per_head=mean_per_head, same_frame_flat_cocycle=True,
                             cond_escalation=cond_escalation)


def _certifies_same_frame_flat_cocycle(
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',
) -> bool:
    r"""Whether a transport certifies an analytically identity self map."""
    if isinstance(omega, (CompactFactoredTransport, FactoredTransport)):
        return omega.same_frame_flat_cocycle
    if isinstance(omega, RopeTransport):
        return (
            omega.same_frame_flat_cocycle
            and _certifies_same_frame_flat_cocycle(omega.base)
        )
    return False


def _restore_certified_self_links_(
    output: torch.Tensor,  # (..., N, N, *event) already allocated transported output
    source: torch.Tensor,  # (..., N, *event) source values
    omega:  'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',

    *,
    event_ndim: int,
) -> torch.Tensor:
    r"""Overwrite the diagonal view for a certified same-frame flat cocycle in place."""
    if not _certifies_same_frame_flat_cocycle(omega):
        return output
    query_axis = -(event_ndim + 2)
    key_axis = -(event_ndim + 1)
    source_token_axis = -(event_ndim + 1)
    if output.shape[query_axis] != output.shape[key_axis]:
        return output
    if source.shape[source_token_axis] != output.shape[key_axis]:
        return output
    diagonal = output.diagonal(offset=0, dim1=query_axis, dim2=key_axis)
    diagonal = diagonal.movedim(-1, source_token_axis)
    if diagonal.shape != source.shape:
        return output
    diagonal.copy_(source)
    return output


def transport_mean(
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',
    mu:    torch.Tensor,                                          # (..., N, K) source (key, index j) means
) -> torch.Tensor:
    r"""Gauge action on means: mu_t[i,j] = Omega_ij @ mu_j. Returns (..., N, N, K).

    Dense path (rank-agnostic via the leading ellipsis): an optional batch axis (B,N,N,K,K)+(B,N,K)
    flows through unchanged, and the unbatched (N,N,K,K)+(N,K) call is identical -- so the
    same primitive serves the batched forward and the unbatched diagnostics path.

    FACTORED path (``omega`` is a :class:`FactoredTransport`, the flat + block fast route): the exps
    are fused into the contraction -- compute m_j = exp(-phi_j) @ mu_j ONCE (B,N,K), then
    mu_t[i,j] = exp(phi_i) @ m_j -- an EXACT reassociation of the dense einsum (round-off level),
    never forming (B,N,N,K,K). Autograd-safe (differentiates through the live exps), so it survives
    the smoothing-mode oracle if a container ever reaches it. When the container carries
    ``mean_per_head=True`` (Tier-1 toggle, default False) the contraction runs per gauge block
    instead (:func:`_factored_per_head_mean`, fp32 reassociation only); a RoPE-wrapped factored
    base recurses through this same dispatch, so the toggle covers it too.

    ROPETRANSPORT path (``omega`` is a :class:`RopeTransport`): the gauge-RoPE rotation R(theta) is
    applied as R_i Omega_ij R_j^T mu_j -- pre-rotate the key mean by R_j^T, transport on the
    un-rotated base, post-rotate by R_i.
    """
    if isinstance(omega, RopeTransport):
        folded = omega.score_operator()
        if folded is not None:                                        # RIGHT: an ordinary coboundary
            out = transport_mean(folded, mu)
            return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
        # LEFT: mu_t[i,j] = R_i Omega_ij R_j^T mu_j -- pre-rotate the key mean by R_j^T, transport
        # on the un-rotated base, post-rotate the result by R_i. R_j^T mu_j = sum_l R[j,l,k] mu[j,l].
        m = torch.einsum("...jlk,...jl->...jk", omega.rope, mu)        # (..., N, K)
        t = transport_mean(omega.base, m)                             # (..., N, N, K)
        out = torch.einsum("...ikl,...ijl->...ijk", omega.rope, t)    # post-rotate by R_i
        return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
    if isinstance(omega, CompactFactoredTransport):
        out = _compact_factored_mean(omega, mu)
        return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
    if isinstance(omega, FactoredTransport):
        if omega.mean_per_head:
            out = _factored_per_head_mean(omega, mu)
        else:
            m = torch.einsum("...jlp,...jp->...jl", omega.exp_neg_phi, mu)  # (..., N, K): exp(-phi_j) @ mu_j
            out = torch.einsum("...ikl,...jl->...ijk", omega.exp_phi, m)    # (..., N, N, K): exp(phi_i) @ m_j
        return _restore_certified_self_links_(out, mu, omega, event_ndim=1)
    return torch.einsum("...ijkl,...jl->...ijk", omega, mu)


def transport_covariance(
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',
    sigma: torch.Tensor,                                          # (..., N, K) diagonal OR (..., N, K, K) full

    *,
    retain_full_precision: bool = False,

    diagonal_out:    Optional[bool]      = None,
    marginal_blocks: Optional[List[int]] = None,
) -> torch.Tensor:
    r"""Sandwich action Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T.

    Full input (...,N,K,K) -> full (...,N,N,K,K). Diagonal input (...,N,K) -> the
    diagonal approximation (...,N,N,K), Sigma_t[i,j,k] = sum_l Omega_ijkl^2
    sigma_jl (the diagonal of the full sandwich). Rank-agnostic via the leading
    ellipsis (optional batch axis); diagonal vs full is detected by the rank gap
    ``sigma.dim() == omega.dim() - 2``, which holds with or without the batch axis.

    FACTORED path (``omega`` is a :class:`FactoredTransport`): a DIAGONAL sigma is sandwiched
    per head -- by block-diagonality the (d, d) block Omega^(h) = exp(phi_i)^(h) exp(-phi_j)^(h)
    is the only nonzero part on head h's coordinates, so Sigma_t[i,j,k] = sum_l (Omega^(h)_ijkl)^2
    sigma_jl runs over head h only (the off-block Omega entries are exactly 0.0, so the full-l sum
    equals the in-block sum). This materializes H * d^2 per pair, never the full K^2 square -- the
    square sits inside the l-sum, so it does NOT factor by squaring the full exps; it factors by
    block-diagonality. A FULL sigma rebuilds the dense Omega (byte-identical) and runs the unchanged
    sandwich, so full covariance is never the round-off factoring.

    ROPETRANSPORT path (``omega`` is a :class:`RopeTransport`): means-only (``on_cov=False``) uses
    the un-rotated base covariance; full-gauge (``on_cov=True``) applies the RoPE congruence. A
    compact-factored base stays factored through that congruence; every other base retains its
    established dense dispatch.

    MEANS-ONLY IS A HYBRID, NOT A PUSHFORWARD (audit 2026-07-26 A-02, the half the challenge tier
    left open). At ``on_cov=False`` the mean transports under :math:`R_i\Omega_{ij}R_j^{\top}` while
    the covariance transports under the bare :math:`\Omega_{ij}`, so the resulting pair
    :math:`(\mu_t, \Sigma_t)` is the pushforward of the key under NO single linear operator. Measured
    at K=4, N=4, two 2x2 blocks: the transported mean equals the fully-rotated one to 0.0 and differs
    from the un-rotated one by 5.63, while the transported covariance equals the UN-rotated one to
    0.0 and differs from the rotated one by 2.06. It is not the diagonal truncation of the rotated
    pushforward either -- against ``diag(R_i Omega Sigma Omega^T R_i^T)`` the shipped covariance is
    off by a relative 6.33, so it is a different object rather than a projection of the right one.

    This is deliberate and forced, not an oversight: for a diagonal family :math:`R\Sigma R^{\top}`
    is dense and unrepresentable, which is why ``rope_full_gauge=True`` requires
    ``family='gaussian_full'``. What survives intact is the structural contract the machinery depends
    on -- at :math:`i=j` both :math:`\Omega_{ii}` and :math:`R_iR_i^{\top}` are the identity, so
    ``mu_t[i,i] == mu_i`` and ``sigma_t[i,i] == sigma_i`` EXACTLY (measured 0.0 for both) and the
    structural :math:`E_{ii}=0` self-pair, with the ``pair_mask`` that keys off it, is unaffected.
    What does NOT survive is the literal reading of the coupling-energy contract
    :math:`E_{ij}=D(q_i\|\Omega_{ij\#}q_j)`: under means-only RoPE there is no such
    :math:`\Omega_{ij}`. Runs record which of the two regimes they used through
    ``rope_pair_energy_exactness`` in the run report.

    ``retain_full_precision=True`` keeps only a full-covariance sandwich in float64 after its
    existing float64 contraction. The default remains the source dtype, and every diagonal route
    ignores this opt-in so the default and compact hot paths remain unchanged.

    ``marginal_blocks`` (audit 2026-08-05) is the irrep partition the CONSUMER will read as
    head-diagonal marginals -- a promise about the downstream read, not a request for a different
    object. It is honored ONLY on the compact-factored full-covariance route and ONLY when it
    matches that transport's own equal-block partition exactly, in which case the cross-head output
    blocks are provably unread and :func:`_compact_factored_full_covariance` skips them (H^2 -> H
    block products). Every other route -- dense, factored, and every diagonal sibling
    -- ignores it, so an unrecognized partition silently falls back to the full congruence rather
    than to a wrong answer. The head-diagonal blocks it does compute are the SAME algebra as the
    full route (``Omega_h Sigma^(h,h) Omega_h^T``), equal up to float64 reassociation, so this is
    exactness on the read region, not a bitwise identity.
    """
    if isinstance(omega, RopeTransport):
        if not omega.on_cov:
            return transport_covariance(
                omega.base,
                sigma,
                retain_full_precision=retain_full_precision,
                diagonal_out=diagonal_out,
                marginal_blocks=marginal_blocks,
            )   # mu-only
        folded = omega.score_operator()
        if folded is not None:                                # RIGHT: an ordinary coboundary
            return transport_covariance(
                folded,
                sigma,
                retain_full_precision=retain_full_precision,
                diagonal_out=diagonal_out,
                marginal_blocks=marginal_blocks,
            )
        if isinstance(omega.base, CompactFactoredTransport):
            blocks = _equal_diag_blocks(
                omega.rope, omega.base.n_blocks, omega.base.block_dim)
            exp_blocks = torch.einsum(
                "...nhkl,...nhlm->...nhkm", blocks, omega.base.exp_blocks)
            inv_blocks = torch.einsum(
                "...nhkl,...nhml->...nhkm", omega.base.inv_blocks, blocks)
            rotated = CompactFactoredTransport(
                exp_blocks, inv_blocks, omega.base.K,
                mean_per_head=omega.base.mean_per_head,
                same_frame_flat_cocycle=_certifies_same_frame_flat_cocycle(omega),
                cond_escalation=omega.base.cond_escalation)
            return transport_covariance(
                rotated,
                sigma,
                retain_full_precision=retain_full_precision,
                diagonal_out=diagonal_out,
                marginal_blocks=marginal_blocks,
            )
        # Other full-gauge bases use the established rotated dense operator.
        out = transport_covariance(
            _rope_dense_omega(omega.base, omega.rope),
            sigma,
            retain_full_precision=retain_full_precision,
            diagonal_out=diagonal_out,
            marginal_blocks=marginal_blocks,
        )
        if isinstance(omega.base, FactoredTransport):
            is_diag = (
                sigma.dim() == omega.base.exp_phi.dim() - 1
                if diagonal_out is None else diagonal_out
            )
            return _restore_certified_self_links_(
                out, sigma, omega, event_ndim=(1 if is_diag else 2))
        return out
    if isinstance(omega, CompactFactoredTransport):
        is_diag = (
            sigma.dim() == omega.exp_blocks.dim() - 2
            if diagonal_out is None else diagonal_out
        )
        if is_diag:
            out = _compact_factored_diagonal_covariance(omega, sigma)
        else:
            sigma_work = sigma.double() if retain_full_precision else sigma
            out = _compact_factored_full_covariance(
                omega, sigma_work,
                blocks_only=_reads_only_head_marginals(omega, marginal_blocks))
        return _restore_certified_self_links_(
            out,
            sigma if is_diag or not retain_full_precision else sigma.double(),
            omega,
            event_ndim=(1 if is_diag else 2),
        )
    if isinstance(omega, FactoredTransport):
        # Diagonal sigma is (..., N, K) -> same rank as exp_phi minus the trailing K axis; a full
        # sigma is (..., N, K, K) -> same rank as exp_phi (the dense-Omega rank-gap is +1 here
        # because the factored exps carry one fewer N axis than the dense (..., N, N, K, K)).
        is_diag = sigma.dim() == omega.exp_phi.dim() - 1 if diagonal_out is None else diagonal_out
        if is_diag:
            out = _factored_diagonal_covariance(omega, sigma)
        else:
            sigma_work = sigma.double() if retain_full_precision else sigma
            out = _factored_full_covariance(omega, sigma_work)
        return _restore_certified_self_links_(
            out,
            sigma if is_diag or not retain_full_precision else sigma.double(),
            omega,
            event_ndim=(1 if is_diag else 2),
        )
    is_diag = sigma.dim() == omega.dim() - 2 if diagonal_out is None else diagonal_out
    if is_diag:
        # fp32 island, matching the compact/factored diagonal siblings (audit 2026-07-26 C-05): this
        # einsum is autocast-eligible, so under AMP the whole congruence was evaluated in bf16 --
        # measured max relative error 4.63e-03 against the fp64 reference. Unlike those siblings the
        # dense form is the manifestly nonnegative sum_l Omega_kl^2 sigma_l, so it needs no
        # escalation or floor; only the working precision was missing.
        with torch.amp.autocast(sigma.device.type, enabled=False):
            work = sigma.dtype if sigma.dtype in (torch.float32, torch.float64) else torch.float32
            out = torch.einsum("...ijkl,...ijkl,...jl->...ijk",
                               omega.to(work), omega.to(work), sigma.to(work))
        return out.to(sigma.dtype)
    # Rank-gap dispatch hardening (audit 2026-07-05 m7): a batch-INDEPENDENT (N, N, K, K) omega with
    # a BATCHED diagonal (B, N, K) sigma satisfies sigma.dim() == omega.dim() - 1 and previously fell
    # through to the full-covariance einsum -- a shape error at best, a silent mis-broadcast when
    # B == N. A genuine full sigma is (..., N, K, K): trailing SQUARE pair matching omega's K and
    # exactly one rank below omega. Validate before contracting; ambiguous callers must pass
    # ``diagonal_out`` explicitly (as the kernel/oracle call sites already do).
    if diagonal_out is None and (
        sigma.dim() != omega.dim() - 1
        or sigma.shape[-1] != sigma.shape[-2]
        or sigma.shape[-1] != omega.shape[-1]
    ):
        raise ValueError(
            f"transport_covariance: sigma shape {tuple(sigma.shape)} is neither a diagonal "
            f"(..., N, K) (rank omega.dim()-2) nor a full (..., N, K, K) (rank omega.dim()-1, "
            f"trailing square K={omega.shape[-1]}) match for omega shape {tuple(omega.shape)}. "
            f"Pass diagonal_out=True/False explicitly to disambiguate (e.g. a batched diagonal "
            f"sigma against a batch-independent omega)."
        )
    # Full-covariance congruence sandwich Omega Sigma Omega^T SQUARES cond(Omega) (audit 2026-06-13
    # M4). Evaluate the contraction in a float64 island (like the matrix-exp upcast). By default it
    # casts back, correctly rounding the sandwich and removing fp32 sum-over-l,m accumulation error.
    # FullGaussian opts into retaining the float64 result because a correctly rounded fp32 matrix can
    # still leave the SPD cone. NOTE this does not rescue the EXTREME regime: for the non-compact
    # groups (glk/block_glk/sp_n) cond(Omega) ~
    # exp(2||phi||) can reach ~1e6 at the retraction's default max_norm=5, and the squared sandwich
    # (~1e12) is then unrepresentable in fp32 STORAGE regardless of compute precision -- bound ||phi||
    # or use a compact group / diagonal family there. Reached only on the full-covariance path
    # (family='gaussian_full'); the diagonal default above and the compact (orthogonal Omega, cond=1)
    # groups are untouched, so the hot path is unchanged.
    work = _congruence_compute_dtype(sigma.dtype)          # fp64 by default; see the policy above
    out = torch.einsum("...ijkl,...jlm,...ijnm->...ijkn",
                       omega.to(work), sigma.to(work), omega.to(work))
    return _certify_full_congruence(
        out, work,
        lambda hi: torch.einsum("...ijkl,...jlm,...ijnm->...ijkn",
                                omega.to(hi), sigma.to(hi), omega.to(hi)),
        source_dtype=sigma.dtype,
        retain_full_precision=retain_full_precision,
    )


def transport_scale(
    scale: torch.Tensor,   # (..., N, K) independent location-scale-family marginal scales
    omega: 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',

    *,
    diagonal_out: Optional[bool] = True,
) -> torch.Tensor:         # (..., N, N, K) variance-matching marginal scales
    r"""Degree-one transport for a factorized location-scale family's marginal scale.

    A signed diagonal/permutation operator maps independent scales exactly as
    ``b'_k = |Omega_kk| b_k``. For a general mixing operator the push-forward of independent
    Laplace coordinates is not factorized Laplace, so this seam returns the explicit
    variance-matching marginal projection

        b'_k = sqrt(sum_l Omega_kl^2 b_l^2).

    This projection is homogeneous of degree one and reduces to the exact ``|Omega| b`` law on
    the family-preserving subgroup. It is not a claim of distributional closure off that subgroup.
    The covariance transport owns every dense/factored/RoPE dispatch, so applying it
    to the variance proxy ``b^2`` keeps those optimized paths shared without treating ``b`` itself
    as a degree-two variance.
    """
    variance_proxy = transport_covariance(
        omega,
        scale.square(),
        diagonal_out=diagonal_out,
    )
    variance_proxy = variance_proxy.clamp_min(0.0)
    positive = variance_proxy > 0.0
    safe_variance = torch.where(positive, variance_proxy, torch.ones_like(variance_proxy))
    return torch.where(positive, safe_variance.sqrt(), torch.zeros_like(safe_variance))


def _compact_factored_mean(
    factored: CompactFactoredTransport,
    mu:       torch.Tensor,               # (..., N, K) source means
) -> torch.Tensor:                        # (..., N, N, K) transported means
    r"""Mean transport over ``H`` compact ``d x d`` blocks, never a dense ``K x K`` factor."""
    H, d = factored.n_blocks, factored.block_dim
    mu_blocks = mu.reshape(*mu.shape[:-1], H, d)                         # (..., N, H, d)
    key = torch.einsum(
        "...jhlp,...jhp->...jhl", factored.inv_blocks, mu_blocks)
    out = torch.einsum(
        "...ihkl,...jhl->...ijhk", factored.exp_blocks, key)
    return out.reshape(*out.shape[:-2], factored.K)


def _compact_pair_blocks(
    factored: CompactFactoredTransport,
    dtype:    Optional[torch.dtype] = None,
) -> torch.Tensor:                        # (..., N, N, H, d, d) pairwise block operators
    r"""Build ``U_i^(h) U_j^(-1,h)``, promoting factors before the product when requested.

    A precision retry must rebuild this product: casting a rounded or singular pair block to
    float64 cannot recover cancellation that already happened in float32.
    """
    exp = factored.exp_blocks if dtype is None else factored.exp_blocks.to(dtype)
    inv = factored.inv_blocks if dtype is None else factored.inv_blocks.to(dtype)
    return torch.einsum("...ihkl,...jhlm->...ijhkm", exp, inv)


def _compact_factored_diagonal_covariance(
    factored: CompactFactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K) diagonal variances
) -> torch.Tensor:                        # (..., N, N, K) diagonal sandwich
    r"""Diagonal congruence over compact blocks without a dense ``K x K`` operator.

    First form each key-side second moment
    ``C_j = U_j^-1 diag(sigma_j) U_j^-T`` and contract it with each query factor. This avoids the
    ``(..., N_q, N_k, H, d, d)`` pair-block allocation for every block and sequence shape.

    Precision (audit 2026-07-25 F3): this two-stage regrouping replaces the manifestly nonnegative
    ``sigma_t[k] = sum_l Omega_kl^2 sigma_l`` with a quadratic form whose intermediate terms have
    MIXED SIGN, so cancellation can drive an output variance negative -- measured minimum -1.51e+03 at
    cond(U) 3.5e3, and -4.52 even in float32 at the clamp boundary where the dense reference returns
    +4.00. Nothing detected it: a negative variance was mapped straight to ``clamp(min=eps)`` = 1e-6 by
    the divergence kernel, inverting that key's precision weight by ~6 orders of magnitude, pushing
    E_ij to kl_max and thereby zeroing the pair gradient through pair_mask. The reduction therefore
    runs in float64, matching the precision policy the full-covariance sibling already applies to the
    same congruence (which SQUARES cond(Omega)), and the result is floored at zero, which is a no-op
    for any mathematically valid value.
    """
    H, d = factored.n_blocks, factored.block_dim
    sigma_blocks = sigma.reshape(*sigma.shape[:-1], H, d)                 # (..., N, H, d)

    def _reduce(dtype: torch.dtype) -> torch.Tensor:
        inv = factored.inv_blocks.to(dtype)
        exp = factored.exp_blocks.to(dtype)
        key_second = torch.einsum(                                        # (...,Nk,H,d,d) C_j
            "...jhlp,...jhmp,...jhp->...jhlm", inv, inv, sigma_blocks.to(dtype))
        return torch.einsum("...ihkl,...jhlm,...ihkm->...ijhk", exp, key_second, exp)

    with torch.amp.autocast(sigma.device.type, enabled=False):
        work = sigma.dtype if sigma.dtype in (torch.float32, torch.float64) else torch.float32
        out = _reduce(work)
        if work is not torch.float64:
            out = _escalate_if_negative(
                out, _reduce,
                exp_blocks=(factored.exp_blocks if factored.cond_escalation else None))
        out = out.clamp(min=0.0).to(sigma.dtype)
    return out.reshape(*out.shape[:-2], factored.K)


def _reads_only_head_marginals(
    factored:        CompactFactoredTransport,
    marginal_blocks: Optional[List[int]],
) -> bool:
    r"""Is the consumer's declared read partition exactly this transport's own head partition?

    The safety predicate for the ``blocks_only`` congruence (audit 2026-08-05). It must hold that
    every coordinate the consumer reads lies INSIDE one head block, so that zeroing the cross-head
    output blocks cannot change a value that is read. That is true iff the consumer's partition is
    the transport's partition -- ``[block_dim] * n_blocks``.

    A MISALIGNED partition is the failure this guards: ``irrep_dims=[4, 6]`` against a transport
    with two 5-wide heads slices ``[0:4]`` and ``[4:10]``, and the second slice straddles the head
    boundary at 5, so it reads the (0,1) and (1,0) cross-blocks the fast path zeroes. Fails closed
    to the full congruence.

    ``len < 2`` also fails closed, mirroring :func:`~vfe3.free_energy.pairwise_energy`, which for
    ``irrep_dims`` None or a single block computes the FULL-K divergence and therefore reads every
    off-block. (At ``n_blocks == 1`` the two routes coincide anyway, so nothing is lost.)
    """
    if marginal_blocks is None or len(marginal_blocks) < 2:
        return False
    return (len(marginal_blocks) == factored.n_blocks
            and all(int(b) == factored.block_dim for b in marginal_blocks))


def _head_diagonal_blocks(
    matrix:    torch.Tensor,              # (..., K, K) with K == n_blocks * block_dim
    n_blocks:  int,
    block_dim: int,
) -> torch.Tensor:                        # (..., n_blocks, block_dim, block_dim)
    r"""The ``H`` head-DIAGONAL ``(d, d)`` sub-blocks of a ``(K, K)`` matrix.

    ``torch.diagonal`` appends the diagonal axis last, so it is moved back into the head slot.
    """
    blocked = matrix.reshape(*matrix.shape[:-2], n_blocks, block_dim, n_blocks, block_dim)
    return torch.diagonal(blocked, dim1=-4, dim2=-2).movedim(-1, -3)


def _compact_head_block_congruence(
    factored:     CompactFactoredTransport,
    sigma:        torch.Tensor,                  # (..., N, K, K) full SPD covariances
    pair_support: Optional[torch.Tensor] = None,  # (..., N, N, H) included head pairs
) -> torch.Tensor:                        # (..., N, N, H, d, d) head-diagonal congruence blocks
    r"""``Omega_h Sigma^(h,h) Omega_h^T`` for each head, in the BLOCK layout, never densified.

    This is the arithmetic that :func:`_compact_factored_full_covariance`'s ``blocks_only`` branch
    performs; that branch then scatters the result into a dense ``(K, K)`` zero tensor, of which
    half is structurally zero and never read. Callers that consume only head marginals can take the
    blocks directly through :func:`transport_covariance_head_blocks` and skip the round trip.
    """
    H, d = factored.n_blocks, factored.block_dim
    work = _congruence_compute_dtype(sigma.dtype)          # fp64 by default; see the policy above
    omega_blocks = _compact_pair_blocks(factored).to(work)
    sigma_diag = _head_diagonal_blocks(sigma, H, d)                # (..., N, H, d, d)
    def _contract(pair: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if pair_support is not None:
            pair = torch.where(pair_support[..., None, None], pair, 0.0)
        out = torch.einsum(                                      # (..., Ni, Nj, H, d, d)
            "...ijhkl,...jhlm,...ijhnm->...ijhkn", pair, sigma_diag.to(dtype), pair)
        if pair_support is not None:
            # Excluded pairs have no objective contribution. Give downstream SPD operations a
            # finite constant, with no gradient through their unused congruence.
            out = torch.where(
                pair_support[..., None, None], out,
                torch.eye(d, device=out.device, dtype=out.dtype))
        return out

    out = _contract(omega_blocks, work)

    def _recompute(dtype: torch.dtype) -> torch.Tensor:
        return _contract(_compact_pair_blocks(factored, dtype=dtype), dtype)

    return _certify_full_congruence(
        out, work, _recompute, source_dtype=sigma.dtype,
    )


def _resolve_compact_for_head_blocks(
    omega: object,
) -> Optional[CompactFactoredTransport]:
    r"""Unwrap ``omega`` to the ``CompactFactoredTransport`` its congruence actually runs on.

    Mirrors :func:`transport_covariance`'s own container dispatch for exactly the routes that reach
    the compact full-covariance branch, and returns ``None`` for every other container so the caller
    falls back to the dense route rather than to a different answer. The RoPE-on-covariance case
    rebuilds the same rotated container that :func:`transport_covariance` builds, INCLUDING its
    same-frame-flat-cocycle certification, so the self-link restore downstream sees what it would
    have seen on the dense route.
    """
    if isinstance(omega, RopeTransport):
        if not omega.on_cov:
            return _resolve_compact_for_head_blocks(omega.base)    # mu-only; covariance untouched
        folded = omega.score_operator()
        if folded is not None:                                     # RIGHT: an ordinary coboundary
            return _resolve_compact_for_head_blocks(folded)
        if isinstance(omega.base, CompactFactoredTransport):
            blocks = _equal_diag_blocks(
                omega.rope, omega.base.n_blocks, omega.base.block_dim)
            return CompactFactoredTransport(
                torch.einsum("...nhkl,...nhlm->...nhkm", blocks, omega.base.exp_blocks),
                torch.einsum("...nhkl,...nhml->...nhkm", omega.base.inv_blocks, blocks),
                omega.base.K,
                mean_per_head=omega.base.mean_per_head,
                same_frame_flat_cocycle=_certifies_same_frame_flat_cocycle(omega),
                cond_escalation=omega.base.cond_escalation)
        return None
    if isinstance(omega, CompactFactoredTransport):
        return omega
    return None


def transport_covariance_head_blocks(
    omega: object,                        # dense/factored/RoPE transport container
    sigma: torch.Tensor,                  # (..., N, K, K) full SPD covariances

    *,
    retain_full_precision: bool                = False,
    diagonal_out:          Optional[bool]          = None,
    marginal_blocks:       Optional[List[int]]     = None,
    pair_support:          Optional[torch.Tensor] = None,  # (..., N, N, H), compact route only
) -> Optional[torch.Tensor]:              # (..., N, N, H, d, d), or None when not applicable
    r"""The head-diagonal congruence blocks, or ``None`` when this transport cannot supply them.

    The block-layout twin of :func:`transport_covariance` for a consumer that reads ONLY head
    marginals (audit 2026-08-06 B2/F14). ``transport_covariance`` must return a dense
    ``(..., N, N, K, K)`` because that is what every one of its callers is typed against, so its
    ``blocks_only`` fast path pays a ``new_zeros`` + scatter to get back into that layout -- at
    B=48, N=64, H=2, d=10 a 629 MB allocation, half of it structurally zero, which the consumer then
    slices apart again. This entry point hands the ``(..., N, N, H, d, d)`` blocks over directly.

    ``None`` means "not applicable, use :func:`transport_covariance`" and is returned BEFORE any
    arithmetic, so probing costs nothing. The conditions are the same ones the dense route uses to
    choose ``blocks_only``: a container that resolves to a ``CompactFactoredTransport``, a
    full-covariance (not diagonal) ``sigma``, and a ``marginal_blocks`` partition equal to the
    transport's own -- see :func:`_reads_only_head_marginals` for why that predicate is what makes
    skipping the cross-head blocks sound.

    EQUAL ON THE READ REGION, not bitwise equal as a tensor: the head-diagonal blocks are the same
    arithmetic on the same operands as the dense route computes, but the cross-head blocks the dense
    route materializes as zeros (and, at ``i == j`` under a certified cocycle, as the source's own
    cross-head entries) are simply absent here. That is the whole point, and it is sound only
    because the caller has promised not to read them. On the compact route, ``pair_support``
    additionally identifies head pairs the objective reads. Excluded pairs are constant identities
    before certification and stay constant through self-link restoration; callers must exclude
    their mean/energy contribution too. None retains the complete pair grid.
    """
    factored = _resolve_compact_for_head_blocks(omega)
    if factored is None or not _reads_only_head_marginals(factored, marginal_blocks):
        return None
    is_diag = (sigma.dim() == factored.exp_blocks.dim() - 2
               if diagonal_out is None else diagonal_out)
    if is_diag:
        return None
    sigma_work = sigma.double() if retain_full_precision else sigma
    out = (_compact_head_block_congruence(factored, sigma_work) if pair_support is None else
           _compact_head_block_congruence(factored, sigma_work, pair_support=pair_support))
    source_blocks = _head_diagonal_blocks(sigma_work, factored.n_blocks, factored.block_dim)
    if pair_support is not None:
        # Preserve the constant placeholder on masked self links when the cocycle restore runs.
        support = torch.broadcast_to(pair_support, out.shape[:-2])
        self_support = support.diagonal(dim1=-3, dim2=-2).movedim(-1, -2)
        source_blocks = torch.where(
            self_support[..., None, None], source_blocks,
            torch.eye(factored.block_dim, device=out.device, dtype=out.dtype))
    return _restore_certified_self_links_(
        out, source_blocks, factored,
        event_ndim=3,                     # (H, d, d) is one element's trailing shape here
    )


def _compact_factored_full_covariance(
    factored:    CompactFactoredTransport,
    sigma:       torch.Tensor,            # (..., N, K, K) full SPD covariances
    blocks_only: bool = False,            # True -> compute ONLY the h == g head-diagonal blocks
) -> torch.Tensor:                        # (..., N, N, K, K) full congruence output
    r"""Full congruence from compact blocks; the output is full but no dense operator is built.

    ``out[h,k,g,n] = sum_{l,m} Omega^(h)[k,l] Sigma^(h,g)[l,m] Omega^(g)[n,m]``, i.e. the (h,g)
    output block is ``Omega_h Sigma^(h,g) Omega_g^T``.

    ``blocks_only`` (audit 2026-08-05) computes only the head-DIAGONAL blocks and leaves the
    cross-head blocks exactly zero, which drops the pair contraction from H^2 to H block products.
    It is a semantic change on general input, so it is opt-in and defaults off: it is exact only
    when the caller either (a) supplies a block-diagonal ``sigma`` -- then the cross-head outputs
    are identically zero anyway, since ``Sigma^(h,g) = 0`` for ``h != g`` -- or (b) consumes only
    the head marginals. ``pairwise_energy`` is case (b): it slices ``block(h*d, (h+1)*d)`` per
    irrep and never reads an off-block.

    CORRECTNESS RESTS ENTIRELY ON (b) (audit 2026-08-06 A8). Justification (a) also happens to hold
    under the live gaussian_full config -- the token prior enters ``diag_embed``-promoted and
    nothing writes a cross-head entry -- but only to eigh round-off: ``retract_spd_full`` on the
    degenerate ``4I`` spectrum returns head-MIXING eigenvectors, and the mixing cancels only because
    ``f(lambda)`` is constant across the degenerate cluster (measured cross-head leak in
    ``Sigma^(1/2)``: 7.3e-8 on exact ``4I``). Do NOT simplify this function by leaning on (a); it is
    a property of one config's initialization, not an invariant. (b) is enforced structurally by
    ``_reads_only_head_marginals`` and is airtight.
    """
    H, d = factored.n_blocks, factored.block_dim
    if blocks_only:
        # The head-diagonal congruence itself lives in _compact_head_block_congruence so that
        # transport_covariance_head_blocks can return it WITHOUT this scatter (audit 2026-08-06 B2).
        # Casting to sigma.dtype before the scatter rather than after is exact: the scatter is pure
        # data movement and an exact zero is an exact zero in either dtype.
        diag_out = _compact_head_block_congruence(factored, sigma)  # (..., Ni, Nj, H, d, d)
        out = diag_out.new_zeros((*diag_out.shape[:-3], H, d, H, d))
        for h in range(H):                                        # H is the head count (2 here)
            out[..., h, :, h, :] = diag_out[..., h, :, :]
        return out.reshape(*out.shape[:-4], factored.K, factored.K)
    sigma_blocks = sigma.reshape(*sigma.shape[:-3], sigma.shape[-3], H, d, H, d)
    work = _congruence_compute_dtype(sigma.dtype)          # fp64 by default; see the policy above
    omega_blocks = _compact_pair_blocks(factored).to(work)
    out = torch.einsum(
        "...ijhkl,...jhlgm,...ijgnm->...ijhkgn",
        omega_blocks, sigma_blocks.to(work), omega_blocks)
    out = out.reshape(*out.shape[:-4], factored.K, factored.K)

    def _recompute(dtype: torch.dtype) -> torch.Tensor:
        pair = _compact_pair_blocks(factored, dtype=dtype)
        return torch.einsum(
            "...ijhkl,...jhlgm,...ijgnm->...ijhkgn",
            pair, sigma_blocks.to(dtype), pair,
        ).reshape(*out.shape)

    return _certify_full_congruence(
        out, work, _recompute, source_dtype=sigma.dtype,
    )


def _factored_per_head_mean(
    factored: FactoredTransport,
    mu:       torch.Tensor,               # (..., N, K) source (key, index j) means
) -> torch.Tensor:                        # (..., N, N, K) transported means
    r"""Per-head mean transport from the factored exps (the mean twin of
    ``_factored_diagonal_covariance``; active production transport numerics).

    For each head h on coordinates [start:end] the block Omega^(h)_ij = exp(phi_i)^(h) exp(-phi_j)^(h)
    is the only nonzero part of Omega on head h's rows (the off-block entries are exactly 0.0), so

        mu_t[i,j]^(h) = exp(phi_i)^(h) ( exp(-phi_j)^(h) mu_j^(h) ),

    and the full-K contraction equals the concatenation of the per-head (d, d) contractions -- the
    same sum with the exactly-zero off-block terms dropped, ~H x fewer FLOPs on the dominant pair
    GEMM. Equal to the dense-K einsum up to fp32 reassociation (pinned allclose atol 1e-6 by
    tests/test_tier12_transport.py). Rank-agnostic via the leading ellipsis.
    """
    parts: List[torch.Tensor] = []
    start = 0
    for d in factored.irrep_dims:
        end    = start + d
        ep     = factored.exp_phi[..., start:end, start:end]       # (..., N, d, d) exp(phi_i)^(h)
        en     = factored.exp_neg_phi[..., start:end, start:end]   # (..., N, d, d) exp(-phi_j)^(h)
        mu_blk = mu[..., start:end]                                # (..., N, d)
        m = torch.einsum("...jlp,...jp->...jl", en, mu_blk)        # (..., N, d): exp(-phi_j)^(h) mu_j^(h)
        parts.append(torch.einsum("...ikl,...jl->...ijk", ep, m))  # (..., N, N, d): exp(phi_i)^(h) m_j
        start = end
    return torch.cat(parts, dim=-1)                                # (..., N, N, K)


def _factored_diagonal_covariance(
    factored: FactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K) diagonal variances
) -> torch.Tensor:                        # (..., N, N, K) diagonal sandwich
    r"""Per-head diagonal sandwich from the factored exps (P0 #2 covariance route).

    For each head h on coordinates [start:end] the block Omega^(h)_ij = exp(phi_i)^(h) exp(-phi_j)^(h)
    is a (d, d) operator and the diagonal sandwich is the quadratic form

        Sigma_t[i,j,k] = sum_l (Omega^(h)_ijkl)^2 sigma_jl
                       = sum_{m,n} ep_i[k,m] G_j[m,n] ep_i[k,n],   G_j = en_j diag(sigma_j) en_j^T,

    so the per-PAIR (..., N, N, d, d) block Omega never needs to exist (audit 2026-06-09 P3): the
    key-side second moment G_j is (..., N, d, d), the query-side outer product ep_i[k,m] ep_i[k,n]
    is (..., N, d, d, d), and the contraction lands directly on the (..., N, N, d) output -- peak
    memory N d^3 + N^2 d instead of N^2 d^2, at identical flop count. Exact w.r.t. the dense
    diagonal sandwich because the dense Omega is block-diagonal (off-block entries exactly 0.0);
    the regrouping is algebraically exact. Its ROUNDING is not: the ``d <= n_tokens`` regrouping
    below has mixed-sign intermediates and can therefore return a NEGATIVE variance, so audit
    2026-07-25 F3 escalates that branch to float64 exactly when it loses nonnegativity, matching the
    compact sibling (see :func:`_escalate_if_negative` for the measurements and the rationale for
    deciding from the data instead of a static toggle). The tall-block branch needs no escalation: its
    squared form is manifestly nonnegative. Rank-agnostic via the leading ellipsis.
    """
    parts: List[torch.Tensor] = []
    start = 0
    n_tokens = sigma.shape[-2]
    out_dtype = sigma.dtype
    work = out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32
    # Autocast-disabled island, matching the compact sibling (audit 2026-07-25 F3): these einsums are
    # autocast-eligible, so under AMP the congruence would otherwise be built in bf16 -- reintroducing
    # exactly the cancellation the float32 working dtype and the escalation below exist to control.
    with torch.amp.autocast(sigma.device.type, enabled=False):
      for d in factored.irrep_dims:
          end = start + d
          ep = factored.exp_phi[..., start:end, start:end]                # (..., N, d, d) exp(phi_i)^(h)
          en = factored.exp_neg_phi[..., start:end, start:end]            # (..., N, d, d) exp(-phi_j)^(h)
          sig_blk = sigma[..., start:end]                                 # (..., N, d)
          if d <= n_tokens:
              def _reduce(dtype: torch.dtype, ep=ep, en=en, sig_blk=sig_blk) -> torch.Tensor:
                  e, n_, s = ep.to(dtype), en.to(dtype), sig_blk.to(dtype)
                  g = torch.einsum("...jml,...jnl,...jl->...jmn", n_, n_, s)  # (..., N, d, d) G_j
                  e2 = e.unsqueeze(-1) * e.unsqueeze(-2)                 # (..., N, d, d, d)
                  return torch.einsum("...ikmn,...jmn->...ijk", e2, g)   # (..., N, N, d)
              blk = _reduce(work)
              if work is not torch.float64:
                  blk = _escalate_if_negative(
                      blk, _reduce,
                      exp_blocks=(ep if factored.cond_escalation else None))
              # Back to the common working dtype so torch.cat below sees one dtype across blocks. An
              # escalated block keeps its CORRECTED value here; only its extra precision is dropped,
              # which is the whole point -- float64 was used to avoid the cancellation, not to be stored.
              parts.append(blk.to(work))
          else:
              # Tall-block regime d > N (audit 2026-06-09 overnight F4): the query-side outer
              # product N d^3 would EXCEED the per-pair dense block's N^2 d^2 there, so rebuild
              # the (exactly equivalent) per-pair block Omega and run the squared diagonal
              # sandwich; for d <= N (the usual case) the factored route above stays the
              # memory winner (N d^3 + N^2 d < N^2 d^2).
              omega_blk = torch.einsum("...ikm,...jml->...ijkl", ep, en)  # (..., N, N, d, d)
              parts.append(torch.einsum("...ijkl,...ijkl,...jl->...ijk",
                                        omega_blk, omega_blk, sig_blk))
          start = end
      # Floored at zero for the same reason as the compact sibling: a no-op for any mathematically
      # valid variance, and it stops a cancellation-induced negative from reaching the divergence
      # kernel's clamp(min=eps), which would invert that key's precision weight (audit 2026-07-25 F3).
    return torch.cat(parts, dim=-1).clamp(min=0.0).to(out_dtype)       # (..., N, N, K)


def _factored_full_covariance(
    factored: FactoredTransport,
    sigma:    torch.Tensor,               # (..., N, K, K) full SPD covariances
) -> torch.Tensor:                        # (..., N, N, K, K) full congruence sandwich
    r"""Per-head full-covariance congruence sandwich Sigma_t[i,j] = Omega_ij Sigma_j Omega_ij^T from
    the factored exps, WITHOUT materializing the dense (..., N, N, K, K) Omega (the full-cov twin of
    ``_factored_diagonal_covariance``).

    For a block-diagonal gauge (block_glk: Omega = exp(phi_i) exp(-phi_j) is block-diagonal with
    EQUAL d x d blocks Omega^(h)_ij = exp(phi_i)^(h) exp(-phi_j)^(h)), the (h, h') output block is

        Sigma_t[i,j]^(h,h') = Omega^(h)_ij  Sigma_j^(h,h')  (Omega^(h')_ij)^T,

    because the off-block entries of Omega are exactly 0 (Higham, block-diagonal product), so only the
    per-head (d, d) blocks of Omega ever multiply. The OUTPUT is still the full (..., N, N, K, K)
    sandwich (a full Sigma's off-diagonal blocks survive, mapped by Omega^(h) on the left and
    Omega^(h') on the right -- exactly what the dense path computes). For equal block sizes, one
    batched float64 einsum carries both head axes and fuses the vertex factors, avoiding H^2 Python
    dispatches and the dense (..., N, N, K, K) Omega. Heterogeneous irrep dimensions retain the
    exact block-pair fallback because their rectangular blocks cannot share one head tensor.
    Value-equal to the dense ``transport_covariance(to_dense_omega(), sigma)``; pinned by
    tests/test_fullcov_alpha_roadmap_2026_06_13.py.
    """
    dims = factored.irrep_dims
    K = sum(dims)
    batch = sigma.shape[:-3]
    N = sigma.shape[-3]

    def _reduce(dtype: torch.dtype) -> torch.Tensor:
        if len(set(dims)) == 1:
            H, d = len(dims), dims[0]
            exp_phi = _equal_diag_blocks(factored.exp_phi, H, d).to(dtype)
            exp_neg = _equal_diag_blocks(factored.exp_neg_phi, H, d).to(dtype)
            sigma_blocks = sigma.reshape(*batch, N, H, d, H, d).to(dtype)
            # One batched contraction over both explicit head axes. Algebraically this is
            # Omega_h Sigma_(h,g) Omega_g^T with Omega_h = exp_phi_h exp_neg_h, but fusing the
            # vertex factors avoids both the H^2 Python loop and H separate pair-operator builds.
            out = torch.einsum(
                "...ihax,...jhxb,...jhbgd,...igcy,...jgyd->...ijhagc",
                exp_phi, exp_neg, sigma_blocks, exp_phi, exp_neg)
            return out.reshape(*batch, N, N, K, K)

        out = torch.zeros(
            *batch, N, N, K, K, dtype=dtype, device=sigma.device)
        blocks = []
        start = 0
        for d in dims:
            end = start + d
            ep = factored.exp_phi[..., start:end, start:end].to(dtype)
            en = factored.exp_neg_phi[..., start:end, start:end].to(dtype)
            omega_h = torch.einsum("...ikl,...jlm->...ijkm", ep, en)
            blocks.append((start, end, omega_h))
            start = end
        for s1, e1, oh in blocks:
            for s2, e2, oh2 in blocks:
                sig_blk = sigma[..., s1:e1, s2:e2].to(dtype)
                out[..., s1:e1, s2:e2] = torch.einsum(
                    "...ijkl,...jlm,...ijnm->...ijkn", oh, sig_blk, oh2)
        return out

    work = _congruence_compute_dtype(sigma.dtype)
    return _certify_full_congruence(
        _reduce(work), work, _reduce, source_dtype=sigma.dtype)
