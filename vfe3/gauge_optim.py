r"""Mixed AdamW and stateless local-group updates for stored gauge frames.

Ordinary parameter groups retain standard AdamW. A stored phi-factor group selected by the
``pullback_group`` policy is staged in float64 through the strict pullback direction, trust scaling,
a first-order-correct chart retraction, and exact right-product validation after storage rounding. Every phi candidate is validated before any phi
table is mutated; committed gradients are consumed so base AdamW cannot apply a second update.
A row whose pullback metric fails the conditioning certificate, or whose candidate cannot be
backtracked inside the chart bound, is rejected individually (zero update, counted on the
optimizer) instead of aborting the whole step (audit 2026-09-23 #10); nonfinite state still raises.
The ``fisher_group`` policy uses the same staging with the belief-Fisher natural-gradient direction
(:func:`belief_fisher_group_direction`, derivation 2026-09-28 Track 2) in place of the pullback.
"""

import functools
import math
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.matrix_log import (
    logm_frechet_pair,
    logm_principal,
    principal_strip_certificate,
)
from vfe3.geometry.phi_fisher import (
    FisherRowStatistics,
    fisher_type_solve,
    jacobi_cholesky_solve,
    pooled_operator_chunks,
    stabilizer_leak,
    type_metric_residual,
    type_row_spectrum,
    whitened_condition,
)
from vfe3.geometry.phi_preconditioner import (
    PullbackGroupDirectionResult,
    pullback_group_direction,
    phi_group_differential_mode,
    uspace_ambient_gradient,
    uspace_chart_velocity,
    uspace_pullback_body_direction,
)


_PHI_PROJECTION_TEMPORARY_BYTES = 64 * 1024 * 1024
# U-space (group_log) retraction, derivation 2026-09-28 Track 1. The margin keeps certified blocks
# 0.25 inside the principal strip |Im lambda| < pi; it was chosen from sampling (0 log failures in
# 4346 in-margin rows under a trust-saturated step), not from a proof.
_UGROUP_STRIP_MARGIN = 0.25
# Closed-form rows also need max_h kappa_2(U_h) <= this bound; larger ones take the series fallback,
# whose 1e6 damped-condition certificate then decides. For normal X_h with real spectrum the
# singular values of the exponential differential are divided differences of exp, so
# kappa(D) <= kappa(U_h) and kappa(G_0) = kappa(D)^2 <= 1e6, the chart route's gate. The closed
# form amplifies relative covector noise by about kappa(U_h) (review 2026-09-28).
_UGROUP_FAST_FRAME_COND_MAX = 1e3
_UGROUP_RETRACTIONS = ("chart", "group_log")
# Optimizer diagnostics added only when the group_log retraction is active (fixed order).
GROUP_LOG_DIAGNOSTIC_KEYS = (
    "phi_group_fast_rows",
    "phi_group_fallback_rows",
    "phi_group_rechart_rows",
    "phi_group_log_backtracks_max",
    "phi_group_strip_margin_min",
    "phi_group_fast_frame_cond_max",
)
# Optimizer diagnostics added only when the belief-Fisher direction (fisher_group) is active.
FISHER_DIAGNOSTIC_KEYS = (
    "phi_fisher_a_median",
    "phi_fisher_kernel_leak_mean",
    "phi_fisher_condP_max",
    "phi_fisher_pos_lambda_min",
    "phi_fisher_offstrip_rows",
    "phi_fisher_pos_cond_max",
)
# (param, active row indices (R,)) -> statistics of those rows; see vfe3.train.PhiFisherStatisticsProvider.
PhiStatisticsProvider = Callable[[torch.nn.Parameter, torch.Tensor], FisherRowStatistics]


def _nan_amax(values: torch.Tensor) -> torch.Tensor:
    r"""Maximum over non-NaN entries; NaN when every entry is NaN (torch has no ``nanmax``)."""
    nan = values.isnan()
    maximum = torch.where(nan, torch.full_like(values, -math.inf), values).amax()
    return torch.where(nan.all(), torch.full_like(maximum, math.nan), maximum)


@dataclass(frozen=True)
class PhiUpdatePolicy:
    optimizer_group_metadata: Mapping[str, object]
    requires_manifold_optimizer: bool
    requires_pullback_geometry: bool = False
    # The optimizer needs a batch-statistics provider (belief-Fisher direction).
    requires_fisher_statistics: bool = False


_PHI_UPDATE_POLICIES: Dict[str, PhiUpdatePolicy] = {}


def register_phi_update_policy(
    name:   str,
    policy: PhiUpdatePolicy,

    *,
    override: bool = False,
) -> None:
    """Register immutable phi-update routing metadata under a duplicate-safe key."""
    if name in _PHI_UPDATE_POLICIES and not override:
        raise KeyError(
            f"phi update policy {name!r} already registered; pass override=True to replace"
        )
    _PHI_UPDATE_POLICIES[name] = PhiUpdatePolicy(
        optimizer_group_metadata=MappingProxyType(dict(policy.optimizer_group_metadata)),
        requires_manifold_optimizer=bool(policy.requires_manifold_optimizer),
        requires_pullback_geometry=bool(policy.requires_pullback_geometry),
        requires_fisher_statistics=bool(policy.requires_fisher_statistics),
    )


def get_phi_update_policy(name: str) -> PhiUpdatePolicy:
    """Return immutable routing metadata for the selected phi update."""
    if name not in _PHI_UPDATE_POLICIES:
        raise KeyError(
            f"no phi update policy {name!r}; available: {sorted(_PHI_UPDATE_POLICIES)}"
        )
    policy = _PHI_UPDATE_POLICIES[name]
    return PhiUpdatePolicy(
        optimizer_group_metadata=MappingProxyType(dict(policy.optimizer_group_metadata)),
        requires_manifold_optimizer=policy.requires_manifold_optimizer,
        requires_pullback_geometry=policy.requires_pullback_geometry,
        requires_fisher_statistics=policy.requires_fisher_statistics,
    )


register_phi_update_policy(
    "adamw",
    PhiUpdatePolicy(
        optimizer_group_metadata={},
        requires_manifold_optimizer=False,
    ),
)
register_phi_update_policy(
    "pullback_group",
    PhiUpdatePolicy(
        optimizer_group_metadata={"pullback_group": True, "weight_decay": 0.0},
        requires_manifold_optimizer=True,
        requires_pullback_geometry=True,
    ),
)
# Belief-Fisher natural gradient for the stored frames (derivation 2026-09-28, Track 2). It shares the
# stateless manifold staging; the direction replaces the Frobenius pullback, so no pullback geometry
# (structure constants, series metric) is required.
register_phi_update_policy(
    "fisher_group",
    PhiUpdatePolicy(
        optimizer_group_metadata={"pullback_group": True, "weight_decay": 0.0},
        requires_manifold_optimizer=True,
        requires_pullback_geometry=False,
        requires_fisher_statistics=True,
    ),
)


def embedded_phi_frobenius_norm(
    phi:   torch.Tensor,                   # (..., n_gen) algebra coordinates
    group: GaugeGroup,

    *,
    warn_fallback: bool = True,
) -> torch.Tensor:                        # (...) ||sum_a phi_a G_a||_F
    r"""Exact embedded Frobenius norm through a certified diagonal Gram or dense fallback.

    If ``group`` certifies ``<G_a,G_b>_F = 0`` for ``a != b``, then
    ``||sum_a phi_a G_a||_F^2 = sum_a phi_a^2 ||G_a||_F^2``. Uncertified bases retain the exact
    dense embedding; they are correct but can be prohibitively expensive for a per-step global
    projection.
    """
    if phi.shape[-1] != group.generator_shape[0]:
        raise ValueError(
            "embedded phi norm requires full generator coordinates; got coordinate width "
            f"{phi.shape[-1]} and {group.generator_shape[0]} generators"
        )
    diagonal_fn = getattr(group, "gram_diagonal", None)
    diagonal = diagonal_fn() if diagonal_fn is not None else None
    if diagonal is not None:
        uniform = group.gram_diagonal_uniform()
        if uniform is not None:
            return torch.linalg.vector_norm(phi, dim=-1) * math.sqrt(uniform)
        weights = diagonal.to(device=phi.device, dtype=phi.dtype)
        return (phi.square() * weights).sum(dim=-1).clamp_min(0.0).sqrt()

    if warn_fallback and not getattr(group, "_phi_norm_fallback_warned", False):
        warnings.warn(
            f"gauge group {getattr(group, 'name', '<custom>')!r} has no diagonal "
            "generator-Gram certificate; "
            "phi matrix norms use the exact dense_fallback route, which can be expensive",
            RuntimeWarning,
            stacklevel=2,
        )
        group._phi_norm_fallback_warned = True
    basis = group.generators.to(device=phi.device, dtype=phi.dtype)
    embedded = torch.einsum("...a,aij->...ij", phi, basis)
    return torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1))


@dataclass(frozen=True)
class PullbackGroupCandidate:
    candidate_phi:           torch.Tensor
    trust_scale:             torch.Tensor
    backtracking_reductions: torch.Tensor
    candidate_chart_norm:    torch.Tensor
    group_product_residual:  torch.Tensor
    direction:               PullbackGroupDirectionResult
    # (active,) bool: rows whose update was rejected (candidate == current chart), either by the
    # direction's conditioning certificate or because no backtracked step stayed inside the chart.
    rejected_rows:           Optional[torch.Tensor] = None
    # group_log only (None on the chart route). (active,) bool rows served by a closed-form direction;
    # (active,) bool rows whose accepted candidate moved a block by more than pi in the chart
    # (re-charted onto the principal logarithm); (active,) min over blocks of pi - ||skew X_h||_F,
    # formed only on diagnostic steps; (active,) max_h kappa_2(U_h) where the direction formed it.
    fast_rows:               Optional[torch.Tensor] = None
    recharted_rows:          Optional[torch.Tensor] = None
    strip_margin:            Optional[torch.Tensor] = None
    frame_condition:         Optional[torch.Tensor] = None


def _require_positive_finite(value: float, name: str) -> float:
    """Return a validated positive finite scalar."""
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return converted


def _pullback_group_compact_blocks(group: GaugeGroup) -> bool:
    r"""Whether the residual check can read the chart as stacked equal square blocks.

    Reads the basis size from ``group.generator_shape``, so the lazy elementary basis is never
    materialized (audit 2026-10-08, hotspot-4#5: a float64 copy of the dense basis was built on every
    backtracking attempt only to read its first dimension).
    """
    return (
        group.name == "block_glk"
        and len(group.irrep_dims) > 1
        and len(set(group.irrep_dims)) == 1
        and group.generator_shape[0]
        == len(group.irrep_dims) * group.irrep_dims[0] * group.irrep_dims[0]
    )


def _pullback_group_current_element(
    phi:   torch.Tensor,                     # (active, n_gen) current chart
    group: GaugeGroup,
) -> torch.Tensor:                           # exp(phi), blocked or embedded per the group layout
    r"""``exp(phi)`` for the current chart -- LOOP-INVARIANT across backtracking attempts.

    Hoisted out of :func:`_pullback_group_product_residual` (audit 2026-08-07). ``phi`` does not
    change while the trust factor is halved; only ``delta`` and ``candidate_phi`` do. Recomputing
    this exponential inside the residual check cost up to ``max_backtracks`` redundant
    ``matrix_exp`` calls on the whole active batch per M-step.
    """
    if _pullback_group_compact_blocks(group):
        n_blocks, block_dim = len(group.irrep_dims), group.irrep_dims[0]
        return torch.linalg.matrix_exp(phi.reshape(-1, n_blocks, block_dim, block_dim))
    generators = group.generators.to(device=phi.device, dtype=torch.float64)
    return torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi, generators))


def _dense_group_element(
    element: torch.Tensor,                   # (..., K, K) dense, or (..., H, d, d) equal direct-sum blocks
    K:       int,
) -> torch.Tensor:                           # (..., K, K)
    r"""A dense group element, assembling direct-sum blocks when the per-block direction reported them.

    A group whose basis is not the elementary block basis (block_so, block_co, block_glmc) is staged densely
    while its per-block direction reports :math:`\exp(\varphi)` as blocks (audit 2026-10-07, F47).
    """
    if element.shape[-1] == K and element.shape[-2] == K:
        return element
    dense = element.new_zeros(*element.shape[:-3], K, K)
    d = element.shape[-1]
    for h in range(element.shape[-3]):
        dense[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = element[..., h, :, :]
    return dense


def _pullback_group_product_residual(
    candidate_phi:   torch.Tensor,           # (active, n_gen) stored-precision chart candidate
    current_element: torch.Tensor,           # exp(phi) from _pullback_group_current_element
    delta:           torch.Tensor,           # (active, n_gen) accepted right factor
    group:           GaugeGroup,
) -> torch.Tensor:                           # (active,) relative exact-product residual
    r"""Compare ``exp(candidate)`` with the exact right product ``exp(phi) exp(delta)``.

    ``current_element`` is supplied by the caller rather than recomputed, so a backtracking loop
    pays one ``exp(phi)`` instead of one per attempt (audit 2026-08-07). Value-identical.
    """
    if _pullback_group_compact_blocks(group):
        n_blocks, block_dim = len(group.irrep_dims), group.irrep_dims[0]

        def _blocks(coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates.reshape(-1, n_blocks, block_dim, block_dim)

        candidate_element = torch.linalg.matrix_exp(_blocks(candidate_phi))
        right_element = torch.linalg.matrix_exp(_blocks(delta))
        reference = current_element @ right_element
        error_sq = (candidate_element - reference).square().sum(dim=(-3, -2, -1))
        reference_sq = reference.square().sum(dim=(-3, -2, -1))
        return error_sq.sqrt() / reference_sq.sqrt().clamp_min(torch.finfo(torch.float64).tiny)

    generators = group.generators.to(device=candidate_phi.device, dtype=torch.float64)

    def _embedded(coordinates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...a,aij->...ij", coordinates, generators)

    candidate_element = torch.linalg.matrix_exp(_embedded(candidate_phi))
    right_element = torch.linalg.matrix_exp(_embedded(delta))
    current_element = _dense_group_element(current_element, candidate_element.shape[-1])
    reference = current_element @ right_element
    error = torch.linalg.matrix_norm(candidate_element - reference, dim=(-2, -1))
    scale = torch.linalg.matrix_norm(reference, dim=(-2, -1))
    return error / scale.clamp_min(torch.finfo(torch.float64).tiny)


def _validated_staging_scalars(
    grad_phi: torch.Tensor,             # (active, n_gen) processed covector
    phi:      torch.Tensor,             # (active, n_gen) current chart
    group:    GaugeGroup,

    *,
    learning_rate:    float,
    trust_radius:     float,
    chart_max_norm:   float,
    bch_residual_max: float,
    max_backtracks:   int,
) -> Tuple[float, float, float, float]:
    r"""Validate staging inputs; return ``(learning_rate, trust_radius, chart_max_norm, bch_residual_max)``."""
    if not isinstance(group, GaugeGroup):
        raise TypeError(f"group must be a GaugeGroup, got {type(group).__name__}")
    if grad_phi.ndim != 2 or phi.ndim != 2 or grad_phi.shape != phi.shape:
        raise ValueError(
            "pullback group staging requires matching (active, n_gen) grad/phi tensors; "
            f"got {tuple(grad_phi.shape)} and {tuple(phi.shape)}"
        )
    if grad_phi.shape[-1] != group.generator_shape[0]:
        raise ValueError(
            "pullback group staging requires full generator coordinates; got width "
            f"{grad_phi.shape[-1]} and {group.generator_shape[0]} generators"
        )
    if grad_phi.device != phi.device:
        raise ValueError(
            f"pullback group grad/phi devices differ: {grad_phi.device} and {phi.device}"
        )
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError(
            f"learning_rate must be finite and nonnegative, got {learning_rate!r}"
        )
    trust_radius = _require_positive_finite(trust_radius, "trust_radius")
    chart_max_norm = _require_positive_finite(chart_max_norm, "chart_max_norm")
    bch_residual_max = _require_positive_finite(bch_residual_max, "bch_residual_max")
    if type(max_backtracks) is not int or max_backtracks < 0:
        raise ValueError(
            f"max_backtracks must be a nonnegative int, got {max_backtracks!r}"
        )
    if grad_phi.shape[0] == 0:
        raise ValueError("pullback group staging requires at least one active row")
    return learning_rate, trust_radius, chart_max_norm, bch_residual_max


@torch.no_grad()
def stage_pullback_group_candidate(
    grad_phi: torch.Tensor,             # (active, n_gen) processed covector
    phi:      torch.Tensor,             # (active, n_gen) current chart
    group:    GaugeGroup,

    *,
    learning_rate:    float,
    trust_radius:     float,
    chart_max_norm:   float,
    bch_residual_max: float,
    max_backtracks:   int = 10,
    differential_mode: Optional[str] = None,
    direction_fn:     Optional[Callable[[torch.Tensor, torch.Tensor],
                                        PullbackGroupDirectionResult]] = None,
) -> PullbackGroupCandidate:
    r"""Stage the chart retraction ``phi_new = phi - eta v_phi`` without mutation.

    ``direction_fn(grad64, phi64)``, when given, replaces the Frobenius-pullback direction (the
    belief-Fisher route passes one); ``None`` runs the historical call unchanged.

    Its derivative at eta=0 is the Frobenius-pullback Riemannian direction at the CURRENT chart, or
    the supplied direction when ``direction_fn`` is given.
    Trust scaling uses ``xi = Psi_L(ad_phi) v_phi``. Backtracking bounds the difference
    from ``exp(phi) exp(-eta xi)``; this is a chart retraction, not exact group
    multiplication. The legacy ``bch_residual_max`` keyword retains that product
    tolerance. Certification uses values rounded to the input chart's storage dtype.

    Per-row rejection (audit 2026-09-23 #10): a row whose pullback metric fails the direction
    solver's conditioning certificate (undamped-eigenvalue floor or ``1e6`` damped condition)
    takes a zero step, and a row whose candidate leaves ``chart_max_norm`` is backtracked like a
    residual failure; if no backtracked step returns inside the chart, that row alone keeps its
    current chart. Both are marked in ``rejected_rows``. Nonfinite inputs, candidates, residuals
    and certificates of accepted rows still raise, as does a product-residual or descent failure
    that survives every backtrack, except on rows the direction marks ``soft_reject_rows`` (the
    fisher_group chart route's off-strip rows), which keep their current chart and are counted in
    ``rejected_rows``.
    """
    learning_rate, trust_radius, chart_max_norm, bch_residual_max = _validated_staging_scalars(
        grad_phi, phi, group,
        learning_rate=learning_rate,
        trust_radius=trust_radius,
        chart_max_norm=chart_max_norm,
        bch_residual_max=bch_residual_max,
        max_backtracks=max_backtracks,
    )

    with torch.amp.autocast(grad_phi.device.type, enabled=False):
        grad64 = grad_phi.to(dtype=torch.float64)
        phi64 = phi.to(dtype=torch.float64)
        # ONE host sync, not two (audit 2026-08-07). Every ``bool(tensor)`` on CUDA is a
        # ``cudaStreamSynchronize``; measured 33 of them per staging call. Reducing on device and
        # reading the answer once is value-identical -- the same predicate raises the same error for
        # the same input -- and the `.all()` reductions it stops short-circuiting are negligible
        # beside the stall they were costing.
        if not bool((torch.isfinite(grad64).all() & torch.isfinite(phi64).all())):
            raise FloatingPointError("pullback group staging received a nonfinite grad or chart")
        direction = pullback_group_direction(
            grad64,
            phi64,
            group.generators,
            mode=group.pullback_capability,
            differential_mode=differential_mode,
            coordinate_layout=group.phi_coordinate_layout,
            irrep_dims=list(group.irrep_dims),
            reject_ill_conditioned_rows=True,
        ) if direction_fn is None else direction_fn(grad64, phi64)
        # Rows the direction solver rejected on conditioning carry v_phi = xi = 0; a direction
        # rule without per-row rejection keeps its own whole-batch fail-closed raise.
        rejected = (
            torch.zeros(grad64.shape[0], dtype=torch.bool, device=grad64.device)
            if direction.accepted_rows is None
            else ~direction.accepted_rows.reshape(grad64.shape[0])
        )
        direction_tensors = (
            direction.v_phi,
            direction.xi,
            direction.min_undamped_generalized_eigenvalue,
            direction.undamped_generalized_condition,
            direction.damped_generalized_condition,
            direction.scaled_solve_residual,
        )
        # Six separate ``bool()`` calls collapsed into one device-side reduction (audit 2026-08-07).
        # A rejected row's certificates may be nonfinite (its metric is what failed); its step is 0.
        certificates_finite = direction_tensors[0].new_tensor(True, dtype=torch.bool)
        for value in direction_tensors:
            finite = torch.isfinite(value)
            if finite.dim() > 1:
                finite = finite.all(dim=-1)
            certificates_finite = certificates_finite & (finite | rejected).all()
        if not bool(certificates_finite):
            raise FloatingPointError("pullback group direction returned a nonfinite certificate")

        right_factor = -learning_rate * direction.xi
        right_norm = embedded_phi_frobenius_norm(
            right_factor,
            group,
            warn_fallback=False,
        )
        tiny = torch.finfo(torch.float64).tiny
        trust_scale = (trust_radius / right_norm.clamp_min(tiny)).clamp(max=1.0)
        factor_scale = trust_scale.clone()
        reductions = torch.zeros(
            grad64.shape[0],
            dtype=torch.long,
            device=grad64.device,
        )
        residual_limit = min(1e-6, bch_residual_max)
        current_element = direction.current_element
        if current_element is None:
            current_element = _pullback_group_current_element(phi64, group)
        candidate_phi = torch.empty_like(phi64)
        candidate_chart_norm = torch.empty_like(right_norm)
        group_product_residual = torch.empty_like(right_norm)
        remaining = torch.arange(phi64.shape[0], device=phi64.device)
        for attempt in range(max_backtracks + 1):
            scales = factor_scale[remaining].unsqueeze(-1)
            delta = scales * right_factor[remaining]
            # A finite BCH polynomial has the wrong linearization at a finite chart.
            # Additive chart retraction has derivative -v_phi exactly. Round BEFORE
            # checking bounds/residuals so commit cannot invalidate the certificate.
            proposed = phi64[remaining] - learning_rate * scales * direction.v_phi[remaining]
            proposed = proposed.to(dtype=phi.dtype).to(dtype=torch.float64)
            chart_norm = embedded_phi_frobenius_norm(proposed, group, warn_fallback=False)
            if not bool(torch.isfinite(proposed).all() & torch.isfinite(chart_norm).all()):
                raise FloatingPointError("pullback group staging produced a nonfinite candidate")
            # A chart-bound violation is a per-row backtracking failure, not a whole-step abort.
            outside_chart = chart_norm > chart_max_norm
            residual = _pullback_group_product_residual(
                proposed, current_element[remaining], delta, group,
            )
            # Rounding must not turn the preconditioned chart descent into ascent.
            slope = (grad64[remaining] * (proposed - phi64[remaining])).sum(dim=-1)
            residual_finite = torch.isfinite(residual)
            retraction_failed = (residual > residual_limit) | ~residual_finite | (slope > 0)
            failures = retraction_failed | outside_chart
            candidate_phi[remaining] = proposed
            candidate_chart_norm[remaining] = chart_norm
            group_product_residual[remaining] = residual
            if not bool(failures.any()):
                return PullbackGroupCandidate(
                    candidate_phi=candidate_phi,
                    trust_scale=trust_scale,
                    backtracking_reductions=reductions,
                    candidate_chart_norm=candidate_chart_norm,
                    group_product_residual=group_product_residual,
                    direction=direction,
                    rejected_rows=rejected,
                )
            if not bool(residual_finite.all()):
                raise FloatingPointError(
                    "pullback group staging produced a nonfinite group-product residual"
                )
            if attempt == max_backtracks:
                if bool(retraction_failed.any()):
                    # Rows the direction marked soft (off-strip fisher_group chart rows) are rejected
                    # per row; any other surviving failure still raises for the whole step.
                    soft = (direction.soft_reject_rows.reshape(-1)[remaining]
                            if direction.soft_reject_rows is not None
                            else torch.zeros_like(retraction_failed))
                    if bool((retraction_failed & ~soft).any()):
                        raise FloatingPointError(
                            "pullback group stored chart retraction failed its product residual "
                            f"({float(residual.max()):.3e} versus {residual_limit:.1e}) or descent "
                            f"check after {max_backtracks} reductions"
                        )
                # Chart-bound and soft-rejected failures remain: those rows keep their current chart.
                stuck = remaining[failures]
                candidate_phi[stuck] = phi64[stuck]
                candidate_chart_norm[stuck] = embedded_phi_frobenius_norm(
                    phi64[stuck], group, warn_fallback=False)
                group_product_residual[stuck] = 0.0
                rejected[stuck] = True
                return PullbackGroupCandidate(
                    candidate_phi=candidate_phi,
                    trust_scale=trust_scale,
                    backtracking_reductions=reductions,
                    candidate_chart_norm=candidate_chart_norm,
                    group_product_residual=group_product_residual,
                    direction=direction,
                    rejected_rows=rejected,
                )
            # Accepted rows are retained; only unresolved rows pay for subsequent exponentials.
            remaining = remaining[failures]
            factor_scale[remaining] *= 0.5
            reductions[remaining] += 1

    raise AssertionError("unreachable pullback group staging path")


@dataclass(frozen=True)
class GroupLogDirection:
    r"""Body direction consumed by :func:`stage_group_log_candidate`.

    ``xi`` is the left-trivialized body generator: the frame moves as
    :math:`U \mapsto U e^{-\eta \xi}`. ``covector`` is a U-space covector representative
    :math:`\bar G`, so that the first-order loss change of :math:`U \mapsto U + \delta U` is
    :math:`\langle \bar G, \delta U \rangle_F`; the stager's descent gate reads it. Both are stacks of
    blocks, ``(A, H, d, d)``, or ``(A, K, K)`` for glk. Rows with ``accepted`` False take no step.
    The optional certificate fields fill the returned :class:`PullbackGroupDirectionResult` and are
    NaN where absent.
    """
    xi:        torch.Tensor                            # (A, H, d, d) or (A, K, K)
    covector:  torch.Tensor                            # (A, H, d, d) or (A, K, K)
    accepted:  torch.Tensor                            # (A,) bool
    v_phi:     Optional[torch.Tensor] = None           # (A, n_gen) chart velocity; NaN where not formed
    min_undamped_generalized_eigenvalue: Optional[torch.Tensor] = None   # (A,)
    undamped_generalized_condition:      Optional[torch.Tensor] = None   # (A,)
    damped_generalized_condition:        Optional[torch.Tensor] = None   # (A,)
    scaled_solve_residual:               Optional[torch.Tensor] = None   # (A,)
    series_order: int = 0
    fast_rows:    Optional[torch.Tensor] = None        # (A,) bool rows served by a closed form
    frame_condition: Optional[torch.Tensor] = None     # (A,) max_h kappa_2(U_h)
    offstrip_rows: Optional[torch.Tensor] = None       # (A,) bool rows with a block off the principal strip


# direction(grad_phi (A, n_gen) f64, phi (A, n_gen) f64, current_blocks (A, H, d, d) f64, *, collect)
GroupLogDirectionFn = Callable[..., GroupLogDirection]


def group_log_block_shape(group: GaugeGroup) -> Tuple[int, int]:
    r"""``(H, d)`` such that a chart row reshapes exactly to ``H`` blocks ``X_h`` of size ``d x d``.

    Certified for block_glk with the built-in row-major elementary basis (any head count). Every
    other basis raises: the group_log route reads the chart through this reshape.
    """
    shape = group.elementary_block_shape
    if shape is not None:
        return shape
    raise ValueError(
        "m_phi_group_retraction='group_log' requires block_glk with the built-in "
        f"row-major elementary basis; got gauge group {group.name!r}"
    )


@torch.no_grad()
def pullback_group_log_direction(
    grad_phi:       torch.Tensor,               # (A, n_gen) float64 processed covector
    phi:            torch.Tensor,               # (A, n_gen) float64 current chart
    current_blocks: torch.Tensor,               # (A, H, d, d) float64 exp(X_h)

    *,
    collect:           bool,
    group:             GaugeGroup,
    differential_mode: Optional[str] = None,
) -> GroupLogDirection:
    r"""Closed-form Frobenius-pullback body direction with the series direction off the strip.

    Rows whose blocks all satisfy :math:`\max|\mathrm{Im}\,\lambda(X_h)| < \pi - 0.25` (Bendixson,
    then exact eigenvalues) and :math:`\max_h \kappa_2(U_h) \le 10^3` use
    :math:`\xi = U^{-1} L_{\log}(U^\top, M)` blockwise (:func:`uspace_pullback_body_direction`).
    The other rows, and rows whose Frechet log fails, use :func:`pullback_group_direction` with
    per-row conditioning rejection; rows it rejects are rejected. The frame-condition bound keeps
    the chart route's acceptance semantics: the closed form has no metric certificate, and it
    amplifies relative covector noise by about :math:`\kappa_2(U_h)`. The covector is :math:`U \xi = P_T \bar G`, which for gl and block gl is the ambient
    gradient itself (with the Gram ridge on fallback rows). With ``collect`` the chart velocity
    :math:`v = L_{\log}(U, U \xi)` is formed for the cosine diagnostic; fast rows carry NaN
    conditioning fields because no metric is assembled for them.
    """
    rows, n_gen = grad_phi.shape
    n_blocks, block_dim = current_blocks.shape[-3], current_blocks.shape[-1]
    X = phi.view(rows, n_blocks, block_dim, block_dim)                   # (A, H, d, d)
    M = grad_phi.view(rows, n_blocks, block_dim, block_dim)              # (A, H, d, d), Gram B = I
    U = current_blocks
    certified, _ = principal_strip_certificate(X, margin=_UGROUP_STRIP_MARGIN)   # (A, H)
    sigma = torch.linalg.svdvals(U)                                      # (A, H, d) descending
    frame_condition = (sigma[..., 0] / sigma[..., -1].clamp_min(
        torch.finfo(torch.float64).tiny)).amax(dim=-1)                   # (A,) max_h kappa_2(U_h)
    fast = certified.all(dim=-1) & (frame_condition <= _UGROUP_FAST_FRAME_COND_MAX)   # (A,)
    xi = torch.zeros_like(X)
    accepted = torch.ones(rows, dtype=torch.bool, device=phi.device)
    nan_rows = torch.full((rows,), float("nan"), dtype=torch.float64, device=phi.device)
    certificates = [nan_rows.clone() for _ in range(4)]
    v_phi = torch.full_like(phi, float("nan")) if collect else None
    series_order = 0
    fast_index = fast.nonzero(as_tuple=True)[0]
    if fast_index.numel() > 0:
        xi_fast, ok = uspace_pullback_body_direction(
            X[fast_index], M[fast_index], U=U[fast_index], certified=certified[fast_index])
        xi[fast_index] = xi_fast
        fast[fast_index] = ok.all(dim=-1)                                # a log failure falls back
    fallback_index = (~fast).nonzero(as_tuple=True)[0]
    if fallback_index.numel() > 0:
        series = pullback_group_direction(
            grad_phi[fallback_index],
            phi[fallback_index],
            group.generators,
            mode=group.pullback_capability,
            differential_mode=differential_mode,
            coordinate_layout=group.phi_coordinate_layout,
            irrep_dims=list(group.irrep_dims),
            reject_ill_conditioned_rows=True,
        )
        xi[fallback_index] = series.xi.reshape(-1, n_blocks, block_dim, block_dim)
        if series.accepted_rows is not None:
            accepted[fallback_index] = series.accepted_rows.reshape(-1)
        for target, value in zip(certificates, (
                series.min_undamped_generalized_eigenvalue,
                series.undamped_generalized_condition,
                series.damped_generalized_condition,
                series.scaled_solve_residual)):
            target[fallback_index] = value.reshape(-1)
        if v_phi is not None:
            v_phi[fallback_index] = series.v_phi.reshape(-1, n_gen)
        series_order = series.series_order
    if v_phi is not None:
        fast_index = fast.nonzero(as_tuple=True)[0]
        if fast_index.numel() > 0:
            U_fast = U[fast_index]
            _, v_fast, failed = logm_frechet_pair(U_fast, U_fast @ xi[fast_index])  # (f, H, d, d)
            v_phi[fast_index] = torch.where(
                failed.any(dim=-1, keepdim=True), float("nan"), v_fast.reshape(-1, n_gen))
    return GroupLogDirection(
        xi=xi,
        covector=U @ xi,
        accepted=accepted,
        v_phi=v_phi,
        min_undamped_generalized_eigenvalue=certificates[0],
        undamped_generalized_condition=certificates[1],
        damped_generalized_condition=certificates[2],
        scaled_solve_residual=certificates[3],
        series_order=series_order,
        fast_rows=fast,
        frame_condition=frame_condition,
    )


@torch.no_grad()
def belief_fisher_group_direction(
    grad_phi:       torch.Tensor,               # (A, n_gen) float64 processed covector
    phi:            torch.Tensor,               # (A, n_gen) float64 current chart
    current_blocks: torch.Tensor,               # (A, H, d, d) float64 exp(X_h)

    *,
    collect:         bool,
    statistics:      FisherRowStatistics,
    rotation_weight: float,
    chart_velocity:  bool = False,
    diagnostics_out: Optional[Dict[str, List[torch.Tensor]]] = None,
) -> GroupLogDirection:
    r"""Belief-Fisher natural-gradient body direction of the stored factor rows (Track 2).

    Per head block, on the principal strip (the same certificate as the closed-form pullback):

    .. math::
        \bar G = L_{\log}(U^\top, M), \qquad G_W = \bar G U^\top, \qquad
        \mathcal M(W) = \frac{N}{n}\,G_W, \qquad \xi = U^{-1} W U,

    with :math:`\mathcal M` the completed belief Fisher in spatial coordinates: the closed form
    :func:`~vfe3.geometry.phi_fisher.fisher_type_solve` at the layer-0 table row for
    ``statistics.kind='type'``, the Jacobi-scaled Cholesky solve of the pooled column operator for
    ``'pos'``. The count factor :math:`N/n` is ``total / count``. The frame moves as
    :math:`U \mapsto U e^{-\eta\xi} = e^{-\eta W} U`, and
    :math:`\langle \bar G, U\xi \rangle_F = \langle G_W, W \rangle_F > 0`. The covector for the
    descent gate is :math:`\bar G`.

    A block off the strip (within the 0.25 margin, or beyond |Im lambda| = pi) recovers its covector and chart
    velocity by inverting Psi in the eigenbasis of its chart block, exact wherever Psi is invertible; it is
    rejected only when that solve fails its conditioning gate (audit 2026-10-07, F21). Rows whose positional
    solve fails its residual gate, whose count is zero, or
    whose outputs are nonfinite are rejected as well. With ``chart_velocity`` (the chart
    retraction) the chart velocity :math:`v = L_{\log}(U, W U) = \Psi_R(\mathrm{ad}_X)^{-1} W` is the
    step and a failure rejects the row; with only ``collect`` it feeds the cosine diagnostic and a
    failure leaves NaN. ``diagnostics_out`` accumulates per-block statistics for
    :data:`FISHER_DIAGNOSTIC_KEYS`. The conditioning fields carry :math:`\mathrm{cond}_{\mathcal P}`
    (type rows) or the Jacobi-scaled operator condition (positional rows, computed only with
    ``collect``, zero otherwise); they are not Frobenius-pullback conditions, so the optimizer logs
    them under ``phi_fisher_condP_max`` / ``phi_fisher_pos_cond_max`` and leaves the
    ``phi_pullback_damped_gen_cond_*`` columns NaN.
    """
    rows, n_gen = grad_phi.shape
    n_blocks, block_dim = current_blocks.shape[-3], current_blocks.shape[-1]
    X = phi.view(rows, n_blocks, block_dim, block_dim)                   # (A, H, d, d)
    M = grad_phi.view(rows, n_blocks, block_dim, block_dim)              # (A, H, d, d), Gram B = I
    U = current_blocks
    certified, _ = principal_strip_certificate(X, margin=_UGROUP_STRIP_MARGIN)   # (A, H)
    G_bar, covector_ok = uspace_ambient_gradient(X, M, U=U, certified=certified, offstrip=True)
    G_W = G_bar @ U.transpose(-1, -2)                                    # (A, H, d, d) spatial covector
    count = statistics.count.to(dtype=torch.float64)
    scale = statistics.total.to(dtype=torch.float64) / count.clamp_min(1.0)      # (A,) N / n
    accepted = covector_ok.all(dim=-1) & (count > 0)
    if statistics.kind == "type":
        mu, s = statistics.local_mean, statistics.local_var              # (A, H, d)
        W = fisher_type_solve(G_W, mu, s, scale, rotation_weight=rotation_weight)
        a, lam_min, cond = type_row_spectrum(mu, s, rotation_weight=rotation_weight)   # (A, H)
        residual = type_metric_residual(W, G_W, mu, s, scale, rotation_weight=rotation_weight)
    elif statistics.kind == "pos":
        # The (A, H, d*d, d*d) operators are formed and solved a row chunk at a time under a byte
        # budget (audit 2026-10-08 P2-22); every row's solve is independent, so this is exact.
        g = G_W.reshape(rows, n_blocks, block_dim * block_dim)
        solution = torch.zeros_like(g)
        solved = torch.zeros(rows, n_blocks, dtype=torch.bool, device=g.device)
        solve_residual = torch.zeros(rows, n_blocks, dtype=g.dtype, device=g.device)
        # The spectrum is diagnostic only: one eigvalsh per block on collect steps, zeros otherwise
        # (the chart stager reads these fields only for finiteness).
        lam_min, cond = G_W.new_zeros(rows, n_blocks), G_W.new_zeros(rows, n_blocks)   # (A, H)
        for start, stop, operator in pooled_operator_chunks(statistics, rotation_weight=rotation_weight):
            solution[start:stop], solved[start:stop], solve_residual[start:stop] = jacobi_cholesky_solve(
                operator, g[start:stop])
            if collect:
                lam_min[start:stop], cond[start:stop] = whitened_condition(operator)
        W = scale[:, None, None, None] * solution.view(rows, n_blocks, block_dim, block_dim)
        accepted = accepted & solved.all(dim=-1)
        residual = solve_residual.amax(dim=-1)
    else:
        raise ValueError(f"unknown Fisher statistics kind {statistics.kind!r}")
    xi = torch.linalg.solve(U, W @ U)                                    # (A, H, d, d) U^{-1} W U
    accepted = (accepted & torch.isfinite(W).flatten(1).all(dim=-1)
                & torch.isfinite(xi).flatten(1).all(dim=-1))
    v_phi = None
    if chart_velocity or collect:
        v, velocity_ok = uspace_chart_velocity(X, W, U=U, certified=certified, offstrip=True)
        velocity_ok = velocity_ok.all(dim=-1)
        if chart_velocity:
            accepted = accepted & velocity_ok
        v_phi = torch.where(velocity_ok[:, None], v.reshape(rows, n_gen), float("nan"))
        v_phi = torch.where(accepted[:, None], v_phi, torch.zeros_like(v_phi))
    keep = accepted[:, None, None, None]
    if diagnostics_out is not None:
        diagnostics_out.setdefault("offstrip_rows", []).append(
            (~certified.all(dim=-1)).sum().to(dtype=torch.float64).reshape(1))
        if statistics.kind == "type":
            r = s.sqrt()                                                 # whitening L = S^{1/2}
            leak = stabilizer_leak(G_W * (r[..., :, None] / r[..., None, :]), mu / r)   # (A, H)
            diagnostics_out.setdefault("a", []).append(a.reshape(-1))
            diagnostics_out.setdefault("leak", []).append(leak.reshape(-1))
            diagnostics_out.setdefault("condP", []).append(cond.reshape(-1))
        else:
            diagnostics_out.setdefault("pos_lambda_min", []).append(lam_min.reshape(-1))
            diagnostics_out.setdefault("pos_cond", []).append(cond.reshape(-1))
    return GroupLogDirection(
        xi=torch.where(keep, xi, torch.zeros_like(xi)),
        covector=torch.where(keep, G_bar, torch.zeros_like(G_bar)),
        accepted=accepted,
        v_phi=v_phi,
        min_undamped_generalized_eigenvalue=lam_min.amin(dim=-1),
        undamped_generalized_condition=cond.amax(dim=-1),
        damped_generalized_condition=cond.amax(dim=-1),
        scaled_solve_residual=residual,
        offstrip_rows=~certified.all(dim=-1),
    )


def _fisher_chart_direction(
    grad_phi: torch.Tensor,                     # (A, n_gen) float64
    phi:      torch.Tensor,                     # (A, n_gen) float64

    *,
    group:     GaugeGroup,
    direction: GroupLogDirectionFn,
) -> PullbackGroupDirectionResult:
    r"""Adapt a body direction with chart velocity to the chart stager's direction contract.

    Used by ``fisher_group`` with ``m_phi_group_retraction='chart'``: the additive step
    :math:`\varphi - \eta v_\varphi` with :math:`v_\varphi = \Psi_R(\mathrm{ad}_\varphi)^{-1} W`.
    That route's :math:`10^{-6}` product-residual certificate is quadratic in the step and caps
    chart steps near :math:`7\times10^{-3}` per block at trained norms (measured at d=6), so
    ``group_log`` is the recommended retraction. Rows with a block off the principal strip are marked
    ``soft_reject_rows``: near the exp singularity their velocity grows like :math:`1/|\Psi_R|`, so a
    product-residual failure that survives backtracking rejects that row instead of the whole step
    (audit 2026-10-08, geometer#1).
    """
    rows, n_gen = grad_phi.shape
    n_blocks, block_dim = group_log_block_shape(group)
    U = torch.linalg.matrix_exp(phi.view(rows, n_blocks, block_dim, block_dim))
    proposal = direction(grad_phi, phi, U)
    return PullbackGroupDirectionResult(
        v_phi=proposal.v_phi,
        xi=proposal.xi.reshape(rows, n_gen),
        min_undamped_generalized_eigenvalue=proposal.min_undamped_generalized_eigenvalue,
        undamped_generalized_condition=proposal.undamped_generalized_condition,
        damped_generalized_condition=proposal.damped_generalized_condition,
        scaled_solve_residual=proposal.scaled_solve_residual,
        series_order=proposal.series_order,
        current_element=U if n_blocks > 1 else U.reshape(rows, block_dim, block_dim),
        accepted_rows=proposal.accepted,
        soft_reject_rows=proposal.offstrip_rows,
    )


@torch.no_grad()
def stage_group_log_candidate(
    grad_phi: torch.Tensor,             # (active, n_gen) processed covector
    phi:      torch.Tensor,             # (active, n_gen) current chart
    group:    GaugeGroup,

    *,
    direction:        GroupLogDirectionFn,
    learning_rate:    float,
    trust_radius:     float,
    chart_max_norm:   float,
    bch_residual_max: float,
    max_backtracks:   int  = 10,
    collect:          bool = False,
) -> PullbackGroupCandidate:
    r"""Stage the exact group retraction :math:`\varphi_{\rm new} = \mathrm{coords}\,\mathrm{Log}(e^X e^{-\eta s \xi})`.

    ``direction`` supplies the body generator :math:`\xi` (any rule: the closed-form pullback, or a
    Fisher direction). Per row, :math:`s = \min(1, R / \|\eta \xi\|_F)` with the embedded norm over
    all blocks, and :math:`Z = -\eta s \xi`. The candidate is the principal logarithm of the exact
    right product :math:`U e^{Z}`, rounded to the chart's storage dtype before any gate. A nonfinite
    ``xi`` or covector on a row the direction accepted raises. Gates: log
    failure, nonfinite candidate, product residual
    :math:`\|e^{\varphi_{\rm new}} - U e^Z\|_F / \|U e^Z\|_F > \min(10^{-6}, \mathrm{bch\_residual\_max})`,
    positive first-order slope :math:`\langle \bar G, e^{\varphi_{\rm new}} - U \rangle_F > 0`, and
    the chart bound. A failing row halves its scale; there is no halving for retraction truncation,
    since the product is exact up to the log's accuracy. A row still failing after
    ``max_backtracks`` halvings, or rejected by the direction, keeps its chart exactly (no
    re-charting) and is marked rejected. Nonfinite inputs raise.

    The returned :class:`PullbackGroupCandidate` has the chart route's contract, so the optimizer's
    commit path is shared. ``direction.current_element`` is ``(A, H, d, d)``, or ``(A, K, K)`` when
    ``H = 1``, matching :func:`_pullback_group_current_element`.
    """
    learning_rate, trust_radius, chart_max_norm, bch_residual_max = _validated_staging_scalars(
        grad_phi, phi, group,
        learning_rate=learning_rate,
        trust_radius=trust_radius,
        chart_max_norm=chart_max_norm,
        bch_residual_max=bch_residual_max,
        max_backtracks=max_backtracks,
    )
    n_blocks, block_dim = group_log_block_shape(group)
    rows, n_gen = grad_phi.shape
    tiny = torch.finfo(torch.float64).tiny

    with torch.amp.autocast(grad_phi.device.type, enabled=False):
        grad64 = grad_phi.to(dtype=torch.float64)
        phi64 = phi.to(dtype=torch.float64)
        if not bool((torch.isfinite(grad64).all() & torch.isfinite(phi64).all())):
            raise FloatingPointError("group_log staging received a nonfinite grad or chart")
        X = phi64.view(rows, n_blocks, block_dim, block_dim)             # (A, H, d, d)
        U = torch.linalg.matrix_exp(X)                                   # (A, H, d, d)
        proposal = direction(grad64, phi64, U, collect=collect)
        block_shape = (rows, n_blocks, block_dim, block_dim)
        xi = proposal.xi.to(dtype=torch.float64).reshape(block_shape)
        covector = proposal.covector.to(dtype=torch.float64).reshape(block_shape)
        accepted = proposal.accepted.reshape(rows)
        # As on the chart route, a nonfinite output on a row the direction accepted is a kernel
        # failure, not a geometric rejection: raise instead of counting it as rejected.
        finite_rows = (torch.isfinite(xi).flatten(1).all(dim=-1)
                       & torch.isfinite(covector).flatten(1).all(dim=-1))       # (A,)
        if bool((accepted & ~finite_rows).any()):
            raise FloatingPointError("group_log direction returned a nonfinite accepted row")
        xi = torch.where(accepted[:, None, None, None], xi, torch.zeros_like(xi))
        covector = torch.where(accepted[:, None, None, None], covector, torch.zeros_like(covector))

        right_norm = torch.linalg.vector_norm((learning_rate * xi).reshape(rows, -1), dim=-1)
        trust_scale = (trust_radius / right_norm.clamp_min(tiny)).clamp(max=1.0)   # (A,)
        factor_scale = trust_scale.clone()
        reductions = torch.zeros(rows, dtype=torch.long, device=phi64.device)
        residual_limit = min(1e-6, bch_residual_max)
        # Rejected and unresolved rows keep the current chart exactly.
        candidate_phi = phi64.clone()
        candidate_chart_norm = embedded_phi_frobenius_norm(phi64, group, warn_fallback=False)
        group_product_residual = torch.zeros(rows, dtype=torch.float64, device=phi64.device)
        recharted = torch.zeros(rows, dtype=torch.bool, device=phi64.device)
        remaining = accepted.nonzero(as_tuple=True)[0]
        for attempt in range(max_backtracks + 1):
            if remaining.numel() == 0:
                break
            current = U[remaining]                                                   # (r, H, d, d)
            scales = (learning_rate * factor_scale[remaining])[:, None, None, None]
            target = current @ torch.linalg.matrix_exp(-scales * xi[remaining])      # U e^Z
            X_new, log_failed, _ = logm_principal(target)
            # Round BEFORE any gate so the commit cannot invalidate the certificate.
            proposed = X_new.reshape(-1, n_gen).to(dtype=phi.dtype).to(dtype=torch.float64)
            finite = torch.isfinite(proposed).all(dim=-1)
            safe = torch.where(finite[:, None], proposed, phi64[remaining])
            proposed_blocks = safe.view(-1, n_blocks, block_dim, block_dim)
            element = torch.linalg.matrix_exp(proposed_blocks)                       # (r, H, d, d)
            residual = ((element - target).square().sum(dim=(-3, -2, -1)).sqrt()
                        / target.square().sum(dim=(-3, -2, -1)).sqrt().clamp_min(tiny))
            slope = (covector[remaining] * (element - current)).sum(dim=(-3, -2, -1))
            chart_norm = embedded_phi_frobenius_norm(safe, group, warn_fallback=False)
            passed = (~log_failed.any(dim=-1) & finite
                      & (residual <= residual_limit)
                      & (slope <= 0.0)
                      & (chart_norm <= chart_max_norm))
            done = remaining[passed]
            candidate_phi[done] = proposed[passed]
            candidate_chart_norm[done] = chart_norm[passed]
            group_product_residual[done] = residual[passed]
            recharted[done] = torch.linalg.matrix_norm(
                proposed_blocks[passed] - X[done]).amax(dim=-1) > math.pi
            remaining = remaining[~passed]
            if attempt < max_backtracks:
                factor_scale[remaining] *= 0.5
                reductions[remaining] += 1
        rejected = ~accepted
        rejected[remaining] = True

        strip_margin = None
        if collect:
            strip_margin = (math.pi - torch.linalg.matrix_norm(
                0.5 * (X - X.transpose(-1, -2)))).amin(dim=-1)                       # (A,)
        nan_rows = torch.full((rows,), float("nan"), dtype=torch.float64, device=phi64.device)

        def _row_field(value: Optional[torch.Tensor]) -> torch.Tensor:
            return nan_rows if value is None else value.to(dtype=torch.float64).reshape(rows)

        direction_result = PullbackGroupDirectionResult(
            v_phi=(torch.full_like(phi64, float("nan")) if proposal.v_phi is None
                   else proposal.v_phi.to(dtype=torch.float64).reshape(rows, n_gen)),
            xi=xi.reshape(rows, n_gen),
            min_undamped_generalized_eigenvalue=_row_field(
                proposal.min_undamped_generalized_eigenvalue),
            undamped_generalized_condition=_row_field(proposal.undamped_generalized_condition),
            damped_generalized_condition=_row_field(proposal.damped_generalized_condition),
            scaled_solve_residual=_row_field(proposal.scaled_solve_residual),
            series_order=proposal.series_order,
            current_element=U if n_blocks > 1 else U.reshape(rows, block_dim, block_dim),
            accepted_rows=~rejected,
        )
        return PullbackGroupCandidate(
            candidate_phi=candidate_phi,
            trust_scale=trust_scale,
            backtracking_reductions=reductions,
            candidate_chart_norm=candidate_chart_norm,
            group_product_residual=group_product_residual,
            direction=direction_result,
            rejected_rows=rejected,
            fast_rows=(None if proposal.fast_rows is None else proposal.fast_rows.reshape(rows)),
            recharted_rows=recharted,
            strip_margin=strip_margin,
            frame_condition=(None if proposal.frame_condition is None
                             else proposal.frame_condition.reshape(rows)),
        )


def phi_projection_chunk_rows(
    coordinate_width: int,
    matrix_width:     int,
    element_size:     int,
    temporary_bytes:  int = _PHI_PROJECTION_TEMPORARY_BYTES,

    *,
    dense_fallback:   bool = False,
) -> int:
    """Rows whose norm and scale temporaries fit within the requested approximate byte budget."""
    if temporary_bytes < 1:
        raise ValueError(f"temporary_bytes must be positive, got {temporary_bytes}")
    working_elements = 2 * coordinate_width
    if dense_fallback:
        working_elements += matrix_width * matrix_width
    return max(1, temporary_bytes // max(working_elements * element_size, 1))


def _trainable_phi_tables(model: torch.nn.Module) -> List[torch.Tensor]:
    """The distinct trainable frame tables: the token frames and, when learned, the positional table."""
    unique = []
    seen = set()
    for table in (getattr(model.prior_bank, "phi_embed", None), getattr(model, "pos_phi_free", None)):
        if table is not None and table.requires_grad and id(table) not in seen:
            unique.append(table)
            seen.add(id(table))
    return unique


@torch.no_grad()
def project_phi_parameter_rows_(
    model:           torch.nn.Module,
    max_matrix_norm: float,

    *,
    temporary_bytes: int  = _PHI_PROJECTION_TEMPORARY_BYTES,
    collect_stats:   bool = True,
    chunk_rows:      Optional[int] = None,
    count_out:       Optional[torch.Tensor] = None,   # (2,) int64 on the generator device, or None
) -> Dict[str, float]:
    r"""Project every trainable phi-table row to an embedded Frobenius-norm ball.

    Projection rescales algebra coordinates along their current ray and deliberately leaves
    optimizer moments unchanged. It covers belief token frames, independent model-channel frames,
    and the learned positional table when present. A table frozen by design (``requires_grad=False``,
    e.g. the zero token frames under ``gauge_transport='off'``) is never stepped, so it is skipped:
    neither checked nor counted in ``count_out`` or the returned row totals.

    ``count_out`` (training counters, 2026-09-28): when given, the number of rows this call
    rescales (scale :math:`< 1`) and the number of rows it checks are added to ``count_out[0]`` and
    ``count_out[1]`` on the device, with no host synchronization. It only reads ``scale``; the rows
    written are the same with or without it, and ``None`` runs the unchanged code path.
    """
    if not math.isfinite(max_matrix_norm) or max_matrix_norm <= 0.0:
        raise ValueError(
            f"max_matrix_norm must be finite and positive, got {max_matrix_norm}"
        )
    if chunk_rows is not None and (type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows!r}")
    if type(temporary_bytes) is not int or temporary_bytes < 1:
        raise ValueError(f"temporary_bytes must be a positive int, got {temporary_bytes!r}")
    if count_out is not None and (
            not isinstance(count_out, torch.Tensor) or tuple(count_out.shape) != (2,)
            or count_out.dtype != torch.long or count_out.device != model.group.generator_device):
        raise ValueError("count_out must be a (2,) int64 tensor on the generator device")
    unique = _trainable_phi_tables(model)

    group = model.group
    total_rows = 0
    projected_rows = torch.zeros((), device=group.generator_device, dtype=torch.long)
    norm_max_before = torch.zeros((), device=group.generator_device, dtype=group.generator_dtype)
    minimum_scale = torch.ones((), device=group.generator_device, dtype=group.generator_dtype)
    for table in unique:
        rows = table.reshape(-1, table.shape[-1])
        if rows.shape[-1] != group.generator_shape[0]:
            raise ValueError(
                "phi chart projection requires full generator coordinates; got table width "
                f"{rows.shape[-1]} and {group.generator_shape[0]} generators"
            )
        total_rows += rows.shape[0]
        rows_per_chunk = chunk_rows or phi_projection_chunk_rows(
            rows.shape[-1],
            group.generator_shape[-1],
            rows.element_size(),
            temporary_bytes,
            dense_fallback=group.phi_norm_route() == "dense_fallback",
        )
        for start in range(0, rows.shape[0], rows_per_chunk):
            chunk = rows[start : start + rows_per_chunk]
            norm = embedded_phi_frobenius_norm(chunk, group)
            scale = (max_matrix_norm / norm.clamp(min=1e-12)).clamp(max=1.0)
            if collect_stats:
                projected_rows.add_((scale < 1.0).sum())
                norm_max_before.copy_(torch.maximum(norm_max_before, norm.max()))
                minimum_scale.copy_(torch.minimum(minimum_scale, scale.min()))
            if count_out is not None:
                count_out[0].add_((scale < 1.0).sum())
            chunk.mul_(scale.unsqueeze(-1))
    if count_out is not None:
        count_out[1].add_(total_rows)
    if not collect_stats:
        return {}
    projected_rows_value, norm_max_value, minimum_scale_value = torch.stack((
        projected_rows.to(dtype=group.generator_dtype),
        norm_max_before,
        minimum_scale,
    )).cpu().tolist()
    return {
        "phi_chart_projected_rows":     projected_rows_value,
        "phi_chart_total_rows":         float(total_rows),
        "phi_chart_projected_fraction": projected_rows_value / max(total_rows, 1),
        "phi_chart_preproject_max":     norm_max_value,
        "phi_chart_projection_scale_min": minimum_scale_value,
    }


# Groups whose head-block algebra the spread projection maps into itself: gl(d) on every head
# (block_glk, per-head or shared) and the realified gl(m, C) (block_glmc), where the clamp is a
# functional calculus of the realified Hermitian part and so stays realified. Frames of the inert
# groups are orthogonal or conformal on every head (S_0 = 0), so cond_2 = 1 and nothing is projected.
FRAME_CONDITION_GROUPS:       Tuple[str, ...] = ("block_glk", "block_glmc")
FRAME_CONDITION_INERT_GROUPS: Tuple[str, ...] = ("block_so", "block_co", "so_n")
_SPREAD_BISECTION_STEPS = 64


def _bounded_traceless_spectrum(
    lam: torch.Tensor,             # (..., d) ascending eigenvalues of a traceless symmetric S_0
    c:   float,
) -> torch.Tensor:                 # (..., d) zero-sum spectrum with lambda_max - lambda_min <= 2c
    r"""Clamp a zero-sum spectrum into a window of width :math:`2c` that keeps the sum at zero.

    Returns :math:`x_i = \operatorname{clamp}(\lambda_i, t - c, t + c) - \bar{x}`, where :math:`t` solves
    :math:`g(t) = \sum_i \operatorname{clamp}(\lambda_i, t - c, t + c) = 0`. The function :math:`g` is continuous
    and nondecreasing, :math:`g(\lambda_{\min} + c) \le \sum_i\lambda_i = 0 \le g(\lambda_{\max} - c)`, so a root
    lies in that bracket whenever the spread exceeds :math:`2c`; it is found by bisection. The final centering only
    removes the bisection residual and leaves the spread unchanged. The result is the Euclidean projection of
    :math:`\lambda` onto :math:`\{x : \max_i x_i - \min_i x_i \le 2c\}`: for a fixed window
    :math:`[a, a + 2c]` the nearest point is the clamp, and stationarity in :math:`a` is
    :math:`\sum_i (\operatorname{clamp}_i - \lambda_i) = 0`, which is :math:`g(t) = 0` for a zero-sum input.
    A spectrum whose spread is already at most :math:`2c` is not meaningful input here: the caller masks
    those blocks out.
    """
    lo = lam[..., :1] + c                                                  # (..., 1)
    hi = lam[..., -1:] - c                                                 # (..., 1)
    for _ in range(_SPREAD_BISECTION_STEPS):
        t = 0.5 * (lo + hi)
        below = torch.clamp(lam, t - c, t + c).sum(-1, keepdim=True) < 0.0
        lo = torch.where(below, t, lo)
        hi = torch.where(below, hi, t)
    t = 0.5 * (lo + hi)
    x = torch.clamp(lam, t - c, t + c)
    return x - x.mean(-1, keepdim=True)


@torch.no_grad()
def bound_phi_frame_condition_(
    model:     torch.nn.Module,
    kappa_max: float,

    *,
    temporary_bytes: int = _PHI_PROJECTION_TEMPORARY_BYTES,
) -> torch.Tensor:                 # () int64 on the generator device: rows changed (no host read)
    r"""Bound the spectral condition number of every composed frame by ``kappa_max`` (audit 2026-10-12 T4).

    For :math:`X\in\mathfrak{gl}(d)` let :math:`S = \tfrac12(X + X^\top)`. The logarithmic norm
    (Dahlquist; Lozinskii) gives :math:`\|e^{X}\|_2 \le e^{\lambda_{\max}(S)}` and
    :math:`\|e^{-X}\|_2 \le e^{-\lambda_{\min}(S)}`, so

    .. math::
        \kappa_2(e^{X}) \le \exp\big(\lambda_{\max}(S) - \lambda_{\min}(S)\big),

    and :math:`\kappa_2` is submultiplicative over products. The trace part of :math:`S` shifts both
    eigenvalues equally and the skew part :math:`\tfrac12(X - X^\top)` does not enter, so only the traceless
    symmetric part :math:`S_0 = S - \tfrac{\operatorname{tr} S}{d} I` matters. Each head block whose spread
    :math:`\lambda_{\max}(S_0) - \lambda_{\min}(S_0)` exceeds :math:`2c` is replaced by
    :math:`X \leftarrow X - S_0 + S_0'`, with :math:`S_0' = Q\operatorname{diag}(x)Q^\top` built from
    :math:`S_0 = Q\operatorname{diag}(\lambda)Q^\top` and :math:`x` from :func:`_bounded_traceless_spectrum`:
    the eigenvalues are clamped into a width-:math:`2c` window placed so that :math:`\operatorname{tr} S_0' = 0`.
    (A plain clamp to :math:`[-c, c]` would change the trace of an asymmetric spectrum.) The skew part, and
    therefore the rotation, and the trace, and therefore :math:`\det e^{X}`, are unchanged, and
    :math:`\kappa_2(e^{X}) \le e^{2c}`. Since :math:`\mathfrak{gl}(d)` splits Frobenius-orthogonally into skew,
    scalar and traceless symmetric parts, and the nearest symmetric matrix under a spectral constraint shares
    the eigenvectors (Hoffman-Wielandt), :math:`X'` is the Frobenius-nearest point of
    :math:`\{X : \operatorname{spread}(S_0) \le 2c\}`; in particular :math:`\|X'\|_F \le \|X\|_F`, so the bound
    never leaves the ``phi_mstep_max_matrix_norm`` ball. Blocks already within the bound, and rows with no
    such block, are left bitwise unchanged; the projection is idempotent up to roundoff.

    A frame is the product :math:`U_i = e^{X_i} e^{Y_i}` of ``n_factors`` learned factors (the token frame
    ``phi_embed`` and, under ``pos_phi='learned'``, the positional row ``pos_phi_free``;
    :func:`vfe3.geometry.transport.build_factored_transport`). With :math:`c = \log\kappa_{\max} / (2\,n)` every
    composed frame satisfies :math:`\kappa_2(U_i) \le \kappa_{\max}` per head block, and since the frames are
    block-diagonal, for the whole :math:`K\times K` frame. A RoPE rotation folded on the right is orthogonal and
    does not change this. Not guaranteed: the pair transport :math:`\Omega_{ij} = U_i U_j^{-1}` is bounded only
    by :math:`\kappa_{\max}^2`, and a fixed ``pos_phi='frozen'`` factor is not a parameter and is not counted
    or bounded (the composed bound is then :math:`\kappa_{\max}\,\kappa_2(e^{Y_i})`).

    The coordinates map to the per-head :math:`d\times d` blocks exactly: the canonical per-head ``block_glk``
    coordinates are the row-major blocks themselves; any other supported basis (shared ``block_glk``,
    ``block_glmc``) embeds as :math:`M_h = \sum_a \phi_a G_a^{(h)}` and maps back through its diagonal Frobenius
    Gram, :math:`\Delta\phi_a = \sum_h \langle G_a^{(h)}, \Delta M_h\rangle_F / \|G_a\|_F^2`, which is exact
    because :math:`\Delta M` lies in the algebra. The eigendecomposition runs in float64 and the result is cast
    back to the table dtype. Optimizer moments are left unchanged, as in :func:`project_phi_parameter_rows_`.
    """
    if isinstance(kappa_max, bool) or not math.isfinite(kappa_max) or kappa_max <= 1.0:
        raise ValueError(f"kappa_max must be a finite real > 1, got {kappa_max!r}")
    if type(temporary_bytes) is not int or temporary_bytes < 1:
        raise ValueError(f"temporary_bytes must be a positive int, got {temporary_bytes!r}")
    group = model.group
    if group.name not in FRAME_CONDITION_GROUPS:
        raise ValueError(
            f"the frame condition bound is implemented for gauge groups {FRAME_CONDITION_GROUPS}, "
            f"got {group.name!r}")
    dims = group.irrep_dims
    if len(set(dims)) != 1:
        raise ValueError(f"the frame condition bound needs equal head blocks, got {dims}")
    H, d = len(dims), dims[0]
    changed = torch.zeros((), device=group.generator_device, dtype=torch.long)
    tables = _trainable_phi_tables(model)
    if not tables:
        return changed
    c = math.log(kappa_max) / (2 * len(tables))

    elementary = group.elementary_block_shape is not None
    basis = gram = None
    if not elementary:
        basis = group.head_block_generators()                                  # (n_gen, H, d, d)
        if basis is None and H == 1:
            basis = group.generators.unsqueeze(1)                              # (n_gen, 1, K, K)
        if basis is None or not group.frobenius_gram_diagonal:
            raise ValueError(
                f"gauge group {group.name!r}: the frame condition bound needs a head-block basis with a "
                "diagonal Frobenius Gram")
        basis = basis.detach().to(torch.float64)
        gram = basis.square().sum(dim=(1, 2, 3))                               # (n_gen,)
    # Roughly six float64 (H, d, d) temporaries per row (X, S_0, Q, the update and two products).
    rows_per_chunk = max(1, temporary_bytes // (6 * 8 * H * d * d))
    for table in tables:
        rows = table.reshape(-1, table.shape[-1])
        if rows.shape[-1] != group.generator_shape[0]:
            raise ValueError(
                "the frame condition bound requires full generator coordinates; got table width "
                f"{rows.shape[-1]} and {group.generator_shape[0]} generators")
        for start in range(0, rows.shape[0], rows_per_chunk):
            chunk = rows[start : start + rows_per_chunk]
            phi64 = chunk.to(torch.float64)                                    # (r, n_gen)
            if elementary:
                X = phi64.reshape(-1, H, d, d)                                 # (r, H, d, d)
            else:
                X = torch.einsum("ra,ahij->rhij", phi64, basis)                # (r, H, d, d)
            S = 0.5 * (X + X.transpose(-1, -2))
            trace_mean = S.diagonal(dim1=-2, dim2=-1).mean(-1)                 # (r, H)
            S0 = S - trace_mean[..., None, None] * torch.eye(d, dtype=S.dtype, device=S.device)
            lam, Q = torch.linalg.eigh(S0)                                     # (r, H, d), (r, H, d, d)
            over = (lam[..., -1] - lam[..., 0]) > 2.0 * c                     # (r, H)
            shift = torch.where(over[..., None], _bounded_traceless_spectrum(lam, c) - lam, 0.0)
            delta = (Q * shift[..., None, :]) @ Q.transpose(-1, -2)            # (r, H, d, d) S_0' - S_0
            if elementary:
                dphi = delta.reshape(phi64.shape)
            else:
                dphi = torch.einsum("rhij,ahij->ra", delta, basis) / gram
            row_over = over.any(-1)                                            # (r,)
            chunk.copy_(torch.where(row_over[:, None], (phi64 + dphi).to(chunk.dtype), chunk))
            changed.add_(row_over.sum())
    return changed


def _fisher_diagnostics(accumulated: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    r"""Reduce per-block belief-Fisher statistics to :data:`FISHER_DIAGNOSTIC_KEYS` (NaN when absent)."""
    reductions = (
        ("a", torch.median),
        ("leak", torch.mean),
        ("condP", torch.amax),
        ("pos_lambda_min", torch.amin),
        ("offstrip_rows", torch.sum),
        ("pos_cond", torch.amax),
    )
    if not accumulated:
        return {key: float("nan") for key in FISHER_DIAGNOSTIC_KEYS}
    device = next(iter(accumulated.values()))[0].device
    values = []
    for name, reduce in reductions:
        parts = accumulated.get(name)
        values.append(reduce(torch.cat(parts)).to(dtype=torch.float64) if parts
                      else torch.tensor(float("nan"), dtype=torch.float64, device=device))
    return dict(zip(FISHER_DIAGNOSTIC_KEYS, torch.stack(values).cpu().tolist()))   # one host read


class GaugeManifoldAdamW(torch.optim.AdamW):
    r"""AdamW for ordinary groups plus stateless phi group retractions.

    A parameter group marked ``pullback_group=True`` is staged on active rows only, through
    :func:`stage_pullback_group_candidate` (``phi_group_retraction='chart'``) or through
    :func:`stage_group_log_candidate` with :func:`pullback_group_log_direction`
    (``'group_log'``). With ``phi_statistics_provider`` (policy ``fisher_group``) the direction is
    :func:`belief_fisher_group_direction` under either retraction; the provider supplies the batch
    belief statistics of the active rows of each table. The update creates no optimizer state for
    the phi table.
    """

    # Signature-convention exception: torch.optim.Optimizer's contract REQUIRES params as the
    # first positional argument (super().__init__(params, ...)), so ``group`` follows it.
    def __init__(
        self,
        params,
        group: GaugeGroup,

        *,
        phi_group_trust_radius: float,
        phi_chart_max_norm:     float,
        phi_bch_residual_max:   float,
        phi_group_retraction:   str = "chart",
        phi_statistics_provider:    Optional[PhiStatisticsProvider] = None,
        phi_fisher_rotation_weight: float = 2.0,
        **kwargs,
    ) -> None:
        if not isinstance(group, GaugeGroup):
            raise TypeError(f"group must be a GaugeGroup, got {type(group).__name__}")
        if phi_group_retraction not in _UGROUP_RETRACTIONS:
            raise ValueError(
                f"phi_group_retraction must be one of {_UGROUP_RETRACTIONS}, got "
                f"{phi_group_retraction!r}"
            )
        if phi_group_retraction == "group_log" or phi_statistics_provider is not None:
            group_log_block_shape(group)                                # fail before any step
        phi_fisher_rotation_weight = _require_positive_finite(
            phi_fisher_rotation_weight,
            "phi_fisher_rotation_weight",
        )
        phi_group_trust_radius = _require_positive_finite(
            phi_group_trust_radius,
            "phi_group_trust_radius",
        )
        phi_chart_max_norm = _require_positive_finite(
            phi_chart_max_norm,
            "phi_chart_max_norm",
        )
        phi_bch_residual_max = _require_positive_finite(
            phi_bch_residual_max,
            "phi_bch_residual_max",
        )
        super().__init__(params, **kwargs)
        self._group                  = group
        self._phi_group_differential_mode = phi_group_differential_mode()
        self._phi_group_trust_radius = phi_group_trust_radius
        self._phi_chart_max_norm     = phi_chart_max_norm
        self._phi_bch_residual_max   = phi_bch_residual_max
        self._phi_group_retraction   = phi_group_retraction
        self.phi_statistics_provider = phi_statistics_provider
        self._phi_fisher_rotation_weight = phi_fisher_rotation_weight
        # Training-time diagnostics are gated by train.py to log/eval steps. The silent hot path
        # performs no diagnostic tensor work or host reduction. Pullback groups report the fixed-
        # coordinate ridge-direction cosine, reused generalized-condition certificate, accepted
        # right-factor scale, active-row count, and accepted chart norm.
        self._collect_gauge_diag = False
        self._gauge_diag: dict   = {}
        # Cumulative pullback-group rows whose update was rejected (zero step), kept on the device
        # so the hot path never synchronizes for it; read through ``phi_rejected_rows``.
        self._phi_rejected_rows: Optional[torch.Tensor] = None

    @property
    def phi_rejected_rows(self) -> int:
        r"""Cumulative count of pullback-group row updates rejected since construction.

        Counts rows rejected by the direction's conditioning certificate or by the chart bound
        (audit 2026-09-23 #10). Reading it synchronizes with the device.
        """
        return 0 if self._phi_rejected_rows is None else int(self._phi_rejected_rows)

    def __setstate__(self, state) -> None:                             # type: ignore[override]
        r"""Restore generically, running Adam's step migration only where ``"step"`` exists.

        ``Optimizer.load_state_dict`` dispatches to ``__setstate__``; ``Adam.__setstate__`` assumes
        every non-empty per-parameter state carries ``"step"``. Restore through the base
        ``Optimizer`` and migrate only states that actually carry an Adam ``"step"`` slot.
        """
        torch.optim.Optimizer.__setstate__(self, state)
        for s in self.state.values():
            step = s.get("step")
            if step is not None and not torch.is_tensor(step):
                s["step"] = torch.tensor(float(step))

    def state_dict(self) -> Dict[str, object]:                         # type: ignore[override]
        r"""Serialize AdamW state plus the phi differential mode of the pullback route."""
        state = super().state_dict()
        state["phi_group_differential_mode"] = self._phi_group_differential_mode
        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, object],
    ) -> None:                                                        # type: ignore[override]
        r"""Restore AdamW state and the saved phi differential mode."""
        core_state = dict(state_dict)
        saved_mode = core_state.pop("phi_group_differential_mode", None)
        if saved_mode is not None and saved_mode not in ("series", "spectral"):
            raise ValueError(f"unsupported saved phi differential mode: {saved_mode!r}")
        if saved_mode is None:
            saved_mode = "series"
            if any(item.get("pullback_group", False) for item in self.param_groups):
                warnings.warn(
                    "Checkpoint omits phi_group_differential_mode; restoring the historical "
                    "series default. A prior process-global spectral choice is not recoverable.",
                    UserWarning, stacklevel=2,
                )
        super().load_state_dict(core_state)
        self._phi_group_differential_mode = saved_mode

    @torch.no_grad()
    def step(self, closure=None):                                      # type: ignore[override]
        if closure is not None:
            # The manifold groups are stepped and their grads consumed (set to None) BELOW, before the
            # trailing super().step(). A closure that re-evaluates the loss would repopulate those grads
            # and let base AdamW step the frame a SECOND time. No caller passes a closure (GradScaler.step
            # and a bare optimizer.step() both call with closure=None), so reject it rather than risk the
            # double-step.
            raise NotImplementedError(
                "GaugeManifoldAdamW does not support closure-based steps: a manifold gradient is "
                "consumed before super().step(), so a closure that re-backpropagates would "
                "double-step the frame. Call step() with no closure."
            )
        # Fused AdamW sets _step_supports_amp_scaling, so GradScaler.step skips unscale_ and instead
        # attaches grad_scale (the loss scale) and found_inf, which only the fused kernel in
        # super().step() reads. The manifold phi step is built from p.grad here, so it must apply the
        # same contract itself: skip the commit on an overflow step, else divide out the loss scale
        # (audit 2026-10-07, F43). grad_scale is None when the caller already unscaled.
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", None)
        if grad_scale is not None or found_inf is not None:
            manifold_params = [p for group in self.param_groups if group.get("pullback_group", False)
                               for p in group["params"] if p.grad is not None]
            if found_inf is not None and bool(found_inf):              # fp16 path only; GradScaler skips the step
                for p in manifold_params:
                    p.grad = None
                if self._collect_gauge_diag:
                    self._gauge_diag = {}
                return super().step()
            if grad_scale is not None:
                for p in manifold_params:
                    p.grad.div_(grad_scale.to(device=p.grad.device, dtype=p.grad.dtype))
        collect = self._collect_gauge_diag                             # gated: True only on log/eval steps
        if collect:
            self._gauge_diag = {}                                      # never expose a prior attempted step
        cos_acc: List[torch.Tensor] = []
        cond_acc: List[torch.Tensor] = []
        trust_acc: List[torch.Tensor] = []
        chart_acc: List[torch.Tensor] = []
        chart_displacement_acc: List[torch.Tensor] = []
        frame_displacement_acc: List[torch.Tensor] = []
        covector_norm_acc: List[torch.Tensor] = []
        direction_norm_acc: List[torch.Tensor] = []
        effective_scale_acc: List[torch.Tensor] = []
        rejected_acc: List[torch.Tensor] = []
        phi_active_rows = 0
        group_log = self._phi_group_retraction == "group_log"
        fast_acc: List[Optional[torch.Tensor]] = []
        rechart_acc: List[torch.Tensor] = []
        backtrack_acc: List[torch.Tensor] = []
        margin_acc: List[torch.Tensor] = []
        frame_cond_acc: List[Optional[torch.Tensor]] = []
        provider = self.phi_statistics_provider
        fisher_acc: Optional[Dict[str, List[torch.Tensor]]] = (
            {} if collect and provider is not None else None)

        # Reject every nonfinite processed phi covector before deriving an active-row mask or
        # staging any phi candidate. In particular, NaN must not compare false against zero and
        # fall through the inactive-row path, which would otherwise permit finite siblings to move
        # and consume the complete parameter gradient.
        for phi_group in self.param_groups:
            if not phi_group.get("pullback_group", False):
                continue
            for p in phi_group["params"]:
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError(
                        "pullback group step received a nonfinite phi covector"
                    )

        # Stage every stored phi-factor candidate before committing any manifold update. The processed
        # gradient is exactly the tensor present after GradScaler unscale and clipping at optimizer.step.
        # No pullback parameter touches ``self.state``; the route is stateless by construction.
        phi_plans: Dict[int, tuple] = {}
        for phi_group in self.param_groups:
            if not phi_group.get("pullback_group", False):
                continue
            pending_updates = []
            pending_zero_gradients = []
            for p in phi_group["params"]:
                if p.grad is None:
                    continue
                if phi_group["lr"] == 0.0:
                    pending_zero_gradients.append(p)
                    continue
                gradient = p.grad
                flat_gradient = gradient.reshape(-1, gradient.shape[-1])
                flat_phi = p.data.reshape(-1, p.data.shape[-1])
                active = flat_gradient.abs().sum(dim=-1) > 0
                if not bool(active.any()):
                    pending_zero_gradients.append(p)
                    continue
                fisher = None
                if provider is not None:
                    statistics = provider(p, active.nonzero(as_tuple=True)[0])
                    fisher = functools.partial(
                        belief_fisher_group_direction,
                        statistics=statistics,
                        rotation_weight=self._phi_fisher_rotation_weight,
                        diagnostics_out=fisher_acc,
                    )
                if group_log:
                    candidate = stage_group_log_candidate(
                        flat_gradient[active],
                        flat_phi[active],
                        self._group,
                        direction=fisher if fisher is not None else functools.partial(
                            pullback_group_log_direction,
                            group=self._group,
                            differential_mode=self._phi_group_differential_mode,
                        ),
                        learning_rate=phi_group["lr"],
                        trust_radius=self._phi_group_trust_radius,
                        chart_max_norm=self._phi_chart_max_norm,
                        bch_residual_max=self._phi_bch_residual_max,
                        collect=collect,
                    )
                else:
                    candidate = stage_pullback_group_candidate(
                        flat_gradient[active],
                        flat_phi[active],
                        self._group,
                        learning_rate=phi_group["lr"],
                        trust_radius=self._phi_group_trust_radius,
                        chart_max_norm=self._phi_chart_max_norm,
                        bch_residual_max=self._phi_bch_residual_max,
                        differential_mode=self._phi_group_differential_mode,
                        direction_fn=None if fisher is None else functools.partial(
                            _fisher_chart_direction,
                            group=self._group,
                            direction=functools.partial(
                                fisher, collect=collect, chart_velocity=True),
                        ),
                    )
                if collect and group_log:
                    fast_acc.append(candidate.fast_rows)
                    rechart_acc.append(candidate.recharted_rows)
                    backtrack_acc.append(candidate.backtracking_reductions)
                    margin_acc.append(candidate.strip_margin)
                    frame_cond_acc.append(
                        None if candidate.frame_condition is None or candidate.fast_rows is None
                        else torch.where(candidate.fast_rows, candidate.frame_condition,
                                         float("nan")))
                if collect:
                    cosine = torch.nn.functional.cosine_similarity(
                        candidate.direction.v_phi,
                        flat_gradient[active].to(dtype=torch.float64),
                        dim=-1,
                        eps=torch.finfo(torch.float64).tiny,
                    )
                    cos_acc.append(cosine.reshape(-1))
                    cond_acc.append(
                        candidate.direction.damped_generalized_condition.reshape(-1)
                    )
                    accepted_scale = candidate.trust_scale / torch.exp2(
                        candidate.backtracking_reductions.to(dtype=torch.float64)
                    )
                    if candidate.rejected_rows is not None:
                        accepted_scale = accepted_scale * (~candidate.rejected_rows).to(
                            dtype=accepted_scale.dtype)
                    trust_acc.append(accepted_scale.reshape(-1))
                    chart_acc.append(candidate.candidate_chart_norm.reshape(-1))
                    previous = flat_phi[active].to(dtype=torch.float64)
                    chart_displacement_acc.append(torch.linalg.vector_norm(
                        candidate.candidate_phi - previous, dim=-1))
                    current_element = candidate.direction.current_element
                    if current_element is None:
                        current_element = _pullback_group_current_element(previous, self._group)
                    committed_element = _pullback_group_current_element(
                        candidate.candidate_phi, self._group)
                    if committed_element.shape != current_element.shape:
                        K = self._group.generators.shape[-1]
                        current_element = _dense_group_element(current_element, K)
                        committed_element = _dense_group_element(committed_element, K)
                    frame_displacement_acc.append(torch.linalg.vector_norm(
                        (committed_element - current_element).reshape(previous.shape[0], -1), dim=-1))
                    covector_norm_acc.append(torch.linalg.vector_norm(
                        flat_gradient[active].to(dtype=torch.float64), dim=-1))
                    direction_norm_acc.append(torch.linalg.vector_norm(candidate.direction.v_phi, dim=-1))
                    effective_scale_acc.append(phi_group["lr"] * accepted_scale)
                    phi_active_rows += candidate.candidate_phi.shape[0]
                pending_updates.append((p, flat_phi, active, candidate))
            phi_plans[id(phi_group)] = (pending_updates, pending_zero_gradients)

        for group in self.param_groups:
            if group.get("pullback_group", False):
                pending_updates, pending_zero_gradients = phi_plans[id(group)]
                for p, flat_phi, active, candidate in pending_updates:
                    committed = candidate.candidate_phi.to(dtype=p.dtype)
                    flat_phi[active] = committed
                    p.grad = None
                    if candidate.rejected_rows is not None:
                        rejected = candidate.rejected_rows.sum()
                        rejected_acc.append(rejected)
                        self._phi_rejected_rows = (
                            rejected if self._phi_rejected_rows is None
                            else self._phi_rejected_rows + rejected)
                for p in pending_zero_gradients:
                    p.grad = None
                continue
        if collect:
            if cos_acc:
                all_cos = torch.cat(cos_acc)
                allc = torch.cat(cond_acc)
                if provider is not None:
                    # Fisher conditions are logged under FISHER_DIAGNOSTIC_KEYS, not as pullback ones.
                    allc = torch.full_like(allc, float("nan"))
                all_trust = torch.cat(trust_acc)
                all_chart = torch.cat(chart_acc)
                chart_displacement = torch.cat(chart_displacement_acc)
                frame_displacement = torch.cat(frame_displacement_acc)
                effective_scale = torch.cat(effective_scale_acc)
                direction_norm = torch.cat(direction_norm_acc)
                # group_log: closed-form rows carry NaN conditioning (no metric is assembled), and a
                # failed chart-velocity log leaves a NaN cosine; reduce those keys NaN-aware.
                statistics = [
                    all_cos.nanmean() if group_log else all_cos.mean(),
                    allc.nanmedian() if group_log else allc.median(),
                    _nan_amax(allc) if group_log else allc.max(),
                    all_trust.mean(),
                    all_trust.min(),
                    all_chart.max(),
                    chart_displacement.mean(),
                    chart_displacement.max(),
                    frame_displacement.mean(),
                    frame_displacement.max(),
                    torch.cat(covector_norm_acc).mean(),
                    direction_norm.nanmean() if group_log else direction_norm.mean(),
                    effective_scale.mean(),
                    effective_scale.min(),
                    (torch.stack(rejected_acc).sum().to(dtype=effective_scale.dtype)
                     if rejected_acc else effective_scale.new_zeros(())),
                ]
                if group_log:
                    nan = effective_scale.new_tensor(float("nan"))
                    fast = (torch.cat(fast_acc) if all(f is not None for f in fast_acc)
                            else None)
                    statistics += [
                        nan if fast is None else fast.sum().to(dtype=nan.dtype),
                        nan if fast is None else (~fast).sum().to(dtype=nan.dtype),
                        torch.cat(rechart_acc).sum().to(dtype=nan.dtype),
                        torch.cat(backtrack_acc).amax().to(dtype=nan.dtype),
                        torch.cat(margin_acc).amin(),
                        (nan if any(c is None for c in frame_cond_acc)
                         else _nan_amax(torch.cat(frame_cond_acc))),
                    ]
                values = torch.stack(statistics).cpu().tolist()
                self._gauge_diag.update({
                    "phi_ridge_direction_cosine_mean":         values[0],
                    "phi_pullback_damped_gen_cond_median":     values[1],
                    "phi_pullback_damped_gen_cond_max":        values[2],
                    "phi_group_trust_scale_mean":              values[3],
                    "phi_group_trust_scale_min":               values[4],
                    "phi_group_active_rows":                   float(phi_active_rows),
                    "phi_group_chart_norm_max":                values[5],
                    "phi_chart_displacement_mean":             values[6],
                    "phi_chart_displacement_max":              values[7],
                    "phi_frame_displacement_mean":             values[8],
                    "phi_frame_displacement_max":              values[9],
                    "phi_covector_norm_mean":                  values[10],
                    "phi_conditioned_direction_norm_mean":     values[11],
                    "phi_effective_step_scale_mean":           values[12],
                    "phi_effective_step_scale_min":            values[13],
                    "phi_group_rejected_rows":                 values[14],
                })
                if group_log:
                    self._gauge_diag.update(dict(zip(GROUP_LOG_DIAGNOSTIC_KEYS, values[15:])))
            if fisher_acc is not None:
                self._gauge_diag.update(_fisher_diagnostics(fisher_acc))
        return super().step()                                          # closure already rejected above
