r"""Mixed AdamW and stateless local-group updates for stored gauge frames.

Ordinary parameter groups retain standard AdamW. A stored phi-factor group selected by the
``pullback_group`` policy is staged in float64 through the strict pullback direction, trust scaling,
a first-order-correct chart retraction, and exact right-product validation after storage rounding. Every phi candidate is validated before any phi
table is mutated; committed gradients are consumed so base AdamW cannot apply a second update.
"""

import math
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.phi_preconditioner import (
    PullbackGroupDirectionResult,
    pullback_group_direction,
    phi_group_differential_mode,
)


_PHI_PROJECTION_TEMPORARY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PhiUpdatePolicy:
    optimizer_group_metadata: Mapping[str, object]
    requires_manifold_optimizer: bool
    requires_pullback_geometry: bool = False


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


def _require_positive_finite(value: float, name: str) -> float:
    """Return a validated positive finite scalar."""
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return converted


def _pullback_group_compact_blocks(group: GaugeGroup, generators: torch.Tensor) -> bool:
    r"""Whether the residual check can read the chart as stacked equal square blocks."""
    return (
        group.name == "block_glk"
        and len(group.irrep_dims) > 1
        and len(set(group.irrep_dims)) == 1
        and generators.shape[0]
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
    generators = group.generators.to(device=phi.device, dtype=torch.float64)
    if _pullback_group_compact_blocks(group, generators):
        n_blocks, block_dim = len(group.irrep_dims), group.irrep_dims[0]
        return torch.linalg.matrix_exp(phi.reshape(-1, n_blocks, block_dim, block_dim))
    return torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi, generators))


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
    generators = group.generators.to(device=candidate_phi.device, dtype=torch.float64)
    if _pullback_group_compact_blocks(group, generators):
        n_blocks, block_dim = len(group.irrep_dims), group.irrep_dims[0]

        def _blocks(coordinates: torch.Tensor) -> torch.Tensor:
            return coordinates.reshape(-1, n_blocks, block_dim, block_dim)

        candidate_element = torch.linalg.matrix_exp(_blocks(candidate_phi))
        right_element = torch.linalg.matrix_exp(_blocks(delta))
        reference = current_element @ right_element
        error_sq = (candidate_element - reference).square().sum(dim=(-3, -2, -1))
        reference_sq = reference.square().sum(dim=(-3, -2, -1))
        return error_sq.sqrt() / reference_sq.sqrt().clamp_min(torch.finfo(torch.float64).tiny)

    def _embedded(coordinates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...a,aij->...ij", coordinates, generators)

    candidate_element = torch.linalg.matrix_exp(_embedded(candidate_phi))
    right_element = torch.linalg.matrix_exp(_embedded(delta))
    reference = current_element @ right_element
    error = torch.linalg.matrix_norm(candidate_element - reference, dim=(-2, -1))
    scale = torch.linalg.matrix_norm(reference, dim=(-2, -1))
    return error / scale.clamp_min(torch.finfo(torch.float64).tiny)


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
    phi_precond_mode: str,
    max_backtracks:   int = 10,
    differential_mode: Optional[str] = None,
) -> PullbackGroupCandidate:
    r"""Stage the chart retraction ``phi_new = phi - eta v_phi`` without mutation.

    Its derivative at eta=0 is the pullback natural direction at the CURRENT chart.
    Trust scaling uses ``xi = Psi_L(ad_phi) v_phi``. Backtracking bounds the difference
    from ``exp(phi) exp(-eta xi)``; this is a chart retraction, not exact group
    multiplication. The legacy ``bch_residual_max`` keyword retains that product
    tolerance. Certification uses values rounded to the input chart's storage dtype.
    """
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
            mode=phi_precond_mode,
            differential_mode=differential_mode,
            coordinate_layout=group.phi_coordinate_layout,
            irrep_dims=list(group.irrep_dims),
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
        certificates_finite = direction_tensors[0].new_tensor(True, dtype=torch.bool)
        for value in direction_tensors:
            certificates_finite = certificates_finite & torch.isfinite(value).all()
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
            candidate_ok = (
                torch.isfinite(proposed).all()
                & torch.isfinite(chart_norm).all()
                & (chart_norm <= chart_max_norm).all()
            )
            if not bool(candidate_ok):
                if not bool(torch.isfinite(proposed).all() & torch.isfinite(chart_norm).all()):
                    raise FloatingPointError("pullback group staging produced a nonfinite candidate")
                raise FloatingPointError(
                    "pullback group candidate chart norm "
                    f"{float(chart_norm.max()):.6g} exceeds bound {chart_max_norm:.6g}"
                )
            residual = _pullback_group_product_residual(
                proposed, current_element[remaining], delta, group,
            )
            # Rounding must not turn the preconditioned chart descent into ascent.
            slope = (grad64[remaining] * (proposed - phi64[remaining])).sum(dim=-1)
            residual_finite = torch.isfinite(residual)
            failures = (residual > residual_limit) | ~residual_finite | (slope > 0)
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
                )
            if not bool(residual_finite.all()):
                raise FloatingPointError(
                    "pullback group staging produced a nonfinite group-product residual"
                )
            if attempt == max_backtracks:
                raise FloatingPointError(
                    "pullback group stored chart retraction failed its product residual "
                    f"({float(residual.max()):.3e} versus {residual_limit:.1e}) or descent "
                    f"check after {max_backtracks} reductions"
                )
            # Accepted rows are retained; only unresolved rows pay for subsequent exponentials.
            remaining = remaining[failures]
            factor_scale[remaining] *= 0.5
            reductions[remaining] += 1

    raise AssertionError("unreachable pullback group staging path")


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


@torch.no_grad()
def project_phi_parameter_rows_(
    model:           torch.nn.Module,
    max_matrix_norm: float,

    *,
    temporary_bytes: int  = _PHI_PROJECTION_TEMPORARY_BYTES,
    collect_stats:   bool = True,
    chunk_rows:      Optional[int] = None,
) -> Dict[str, float]:
    r"""Project every trainable phi-table row to an embedded Frobenius-norm ball.

    Projection rescales algebra coordinates along their current ray and deliberately leaves
    optimizer moments unchanged. It covers belief token frames, independent model-channel frames,
    and the learned positional table when present.
    """
    if not math.isfinite(max_matrix_norm) or max_matrix_norm <= 0.0:
        raise ValueError(
            f"max_matrix_norm must be finite and positive, got {max_matrix_norm}"
        )
    if chunk_rows is not None and (type(chunk_rows) is not int or chunk_rows < 1):
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows!r}")
    if type(temporary_bytes) is not int or temporary_bytes < 1:
        raise ValueError(f"temporary_bytes must be a positive int, got {temporary_bytes!r}")
    tables = [
        getattr(model.prior_bank, "phi_embed", None),
        getattr(model, "pos_phi_free", None),
    ]
    unique = []
    seen = set()
    for table in tables:
        if table is not None and id(table) not in seen:
            unique.append(table)
            seen.add(id(table))

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
            chunk.mul_(scale.unsqueeze(-1))
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


class GaugeManifoldAdamW(torch.optim.AdamW):
    r"""AdamW for ordinary groups plus stateless phi group retractions.

    A parameter group marked ``pullback_group=True`` is staged through
    :func:`stage_pullback_group_candidate` on active rows only. The update creates no optimizer
    state for the phi table.
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
        phi_precond_mode:       str,
        **kwargs,
    ) -> None:
        if not isinstance(group, GaugeGroup):
            raise TypeError(f"group must be a GaugeGroup, got {type(group).__name__}")
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
        self._generators             = group.generators
        self._precond_mode           = phi_precond_mode
        self._phi_group_differential_mode = phi_group_differential_mode()
        self._phi_group_trust_radius = phi_group_trust_radius
        self._phi_chart_max_norm     = phi_chart_max_norm
        self._phi_bch_residual_max   = phi_bch_residual_max
        # Training-time diagnostics are gated by train.py to log/eval steps. The silent hot path
        # performs no diagnostic tensor work or host reduction. Pullback groups report the fixed-
        # coordinate ridge-direction cosine, reused generalized-condition certificate, accepted
        # right-factor scale, active-row count, and accepted chart norm.
        self._collect_gauge_diag = False
        self._gauge_diag: dict   = {}

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
        phi_active_rows = 0

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
                candidate = stage_pullback_group_candidate(
                    flat_gradient[active],
                    flat_phi[active],
                    self._group,
                    learning_rate=phi_group["lr"],
                    trust_radius=self._phi_group_trust_radius,
                    chart_max_norm=self._phi_chart_max_norm,
                    bch_residual_max=self._phi_bch_residual_max,
                    phi_precond_mode=self._precond_mode,
                    differential_mode=self._phi_group_differential_mode,
                )
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
                for p in pending_zero_gradients:
                    p.grad = None
                continue
        if collect:
            if cos_acc:
                all_cos = torch.cat(cos_acc)
                allc = torch.cat(cond_acc)
                all_trust = torch.cat(trust_acc)
                all_chart = torch.cat(chart_acc)
                chart_displacement = torch.cat(chart_displacement_acc)
                frame_displacement = torch.cat(frame_displacement_acc)
                effective_scale = torch.cat(effective_scale_acc)
                values = torch.stack((
                    all_cos.mean(),
                    allc.median(),
                    allc.max(),
                    all_trust.mean(),
                    all_trust.min(),
                    all_chart.max(),
                    chart_displacement.mean(),
                    chart_displacement.max(),
                    frame_displacement.mean(),
                    frame_displacement.max(),
                    torch.cat(covector_norm_acc).mean(),
                    torch.cat(direction_norm_acc).mean(),
                    effective_scale.mean(),
                    effective_scale.min(),
                )).cpu().tolist()
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
                })
        return super().step()                                          # closure already rejected above
