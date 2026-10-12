"""Shared type contracts for mutable runtime dictionaries."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple, TypedDict

import torch

from vfe3.belief import BeliefState


# Raise-only on-device verdicts waiting for the next mandatory host read; None outside a deferral scope.
_DEFERRED_CHECKS: Optional[List[Tuple[torch.Tensor, Callable[[List[bool]], None]]]] = None


@contextmanager
def deferred_host_checks() -> Iterator[None]:
    r"""Defer raise-only device verdicts to the next mandatory host read inside this scope.

    A check registered through :func:`host_check` whose verdict can only raise (it selects no
    computation) is queued instead of synchronizing. :func:`host_read` (the per-layer recovery
    decision of the projected scorer or the minimal-agent readout) resolves every queued verdict in its
    own transfer, in registration order and before its own values, so each check raises its original
    exception before any later
    check or computation reads its result. Whatever is still queued at a normal scope exit is resolved
    there in one transfer, so the scope must close before anything consumes the scoped computation
    (``train_step`` closes it after the forward and before ``backward``). An exception leaving the
    scope discards the queue. Nested scopes join the outer one; outside any scope checks run at once.
    """
    global _DEFERRED_CHECKS
    if _DEFERRED_CHECKS is not None:
        yield
        return
    _DEFERRED_CHECKS = []
    try:
        yield
        host_read(None)
    finally:
        _DEFERRED_CHECKS = None


def host_check(flags: torch.Tensor, verdict: Callable[[List[bool]], None]) -> None:
    r"""Run ``verdict(flags.tolist())`` now, or queue it inside :func:`deferred_host_checks`.

    ``flags`` is a 1-D boolean tensor and ``verdict`` may only raise; a check whose result selects
    later computation must read its flags itself.
    """
    if _DEFERRED_CHECKS is None:
        verdict(flags.tolist())
        return
    _DEFERRED_CHECKS.append((flags, verdict))


def host_read(values: Optional[torch.Tensor]) -> Optional[list]:
    r"""``values.tolist()`` (a 1-D boolean tensor) in one transfer that also resolves every queued check.

    Queued :func:`host_check` verdicts run first, in registration order, so a deferred check raises
    before the caller acts on its own values. Queued flags on another device take their own transfer.
    """
    pending = _DEFERRED_CHECKS[:] if _DEFERRED_CHECKS else []
    if not pending:
        return None if values is None else values.tolist()
    _DEFERRED_CHECKS.clear()
    device = pending[0][0].device if values is None else values.device
    joined = [flags for flags, _ in pending if flags.device == device]
    host = torch.cat([*joined, *(() if values is None else (values,))]).tolist()
    offset = 0
    for flags, verdict in pending:
        if flags.device == device:
            verdict(host[offset:offset + flags.numel()])
            offset += flags.numel()
        else:
            verdict(flags.tolist())
    return None if values is None else host[offset:]


# Attribute naming the unique-row source of a gathered frame pair (see mark_gathered_frame_pair).
_GATHER_SOURCE = "_vfe3_frame_gather_source"


def mark_gathered_frame_pair(
    forward:        torch.Tensor,  # (n, H, d, d) occurrence rows, (U, H, d, d) gathered by a row index
    inverse:        torch.Tensor,  # (n, H, d, d)
    unique_forward: torch.Tensor,  # (U, H, d, d) every unique row, each gathered at least once
    unique_inverse: torch.Tensor,  # (U, H, d, d)
) -> None:
    r"""Record that a producer built ``(forward, inverse)`` by gathering rows of a unique pair.

    The producer certifies that every occurrence row is a bitwise copy of a unique row and that every
    unique row occurs, so every maximum and every finiteness verdict of the
    :class:`CanonicalFrameContext` residual check is the same over the unique rows as over the
    occurrences. The record holds only for these exact tensor objects at their current version: any
    view, slice, product or in-place write produces a tensor the context validates in full.
    """
    record = (unique_forward, unique_inverse, forward._version, inverse._version)
    setattr(forward, _GATHER_SOURCE, record)
    setattr(inverse, _GATHER_SOURCE, record)


def _gathered_source(forward: torch.Tensor, inverse: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    r"""The unique-row pair recorded by :func:`mark_gathered_frame_pair`, if it still describes this pair."""
    record = getattr(forward, _GATHER_SOURCE, None)
    if record is None or getattr(inverse, _GATHER_SOURCE, None) is not record:
        return None
    unique_forward, unique_inverse, forward_version, inverse_version = record
    if (forward._version != forward_version or inverse._version != inverse_version
            or unique_forward.shape[1:] != forward.shape[-3:] or unique_forward.dtype != forward.dtype
            or unique_inverse.shape != unique_forward.shape or unique_inverse.dtype != inverse.dtype):
        return None
    return unique_forward, unique_inverse


@dataclass(frozen=True)
class CanonicalFrameContext:
    """Realized vertex factors: (..., K, K), or (..., H, d, d) when block_diagonal."""

    forward:        torch.Tensor
    inverse:        torch.Tensor
    block_diagonal: bool = False

    @torch.no_grad()
    def __post_init__(self) -> None:
        if type(self.block_diagonal) is not bool:
            raise ValueError("canonical block_diagonal must be a bool")
        if self.block_diagonal and (self.forward.dim() < 3 or self.forward.shape[-3] < 1):
            raise ValueError("compact canonical frames require a nonempty head axis")
        if self.forward.shape != self.inverse.shape:
            raise ValueError(
                "canonical forward and inverse factors must have identical shapes, got "
                f"{tuple(self.forward.shape)} and {tuple(self.inverse.shape)}")
        if self.forward.dim() < 2 or self.forward.shape[-2] != self.forward.shape[-1]:
            raise ValueError(
                "canonical frame factors must have square trailing matrix axes, got "
                f"{tuple(self.forward.shape)}")
        if self.forward.dtype != self.inverse.dtype:
            raise ValueError(
                "canonical forward and inverse factors must have the same dtype, got "
                f"{self.forward.dtype} and {self.inverse.dtype}")
        if self.forward.device != self.inverse.device:
            raise ValueError(
                "canonical forward and inverse factors must be on the same device, got "
                f"{self.forward.device} and {self.inverse.device}")
        if not self.forward.is_floating_point() or not self.inverse.is_floating_point():
            raise ValueError("canonical frame factors must use a floating-point dtype")
        # Meta tensors carry shape/dtype/device contracts but no materialized values. Let callers
        # reject a meta frame at their explicit device boundary instead of forcing Tensor.item()
        # while attempting a data-dependent finiteness/inverse check here.
        if self.forward.device.type == "meta":
            return
        # Every verdict is computed on device and read back in ONE host transfer (audit 2026-10-07
        # F02/F60); the checks still fail closed in their original order with their original messages.
        validation_dtype = torch.float64
        # A pair its producer gathered from unique token rows (mark_gathered_frame_pair) is validated
        # on those rows: the residual and scale maxima and the product finiteness are the same values,
        # at the unique-token instead of the occurrence count (audit 2026-10-09 #41).
        source = _gathered_source(self.forward, self.inverse) if self.block_diagonal else None
        validated_forward, validated_inverse = (self.forward, self.inverse) if source is None else source
        validation_forward = validated_forward.to(dtype=validation_dtype)
        validation_inverse = validated_inverse.to(dtype=validation_dtype)
        block_dim = self.forward.shape[-1]
        dimension = block_dim * self.forward.shape[-3] if self.block_diagonal else block_dim
        identity = torch.eye(
            block_dim,
            dtype=validation_dtype,
            device=self.forward.device,
        )
        forward_inverse = validation_forward @ validation_inverse
        inverse_forward = validation_inverse @ validation_forward
        residual = torch.maximum(
            (forward_inverse - identity).abs().amax(dim=(-2, -1)),
            (inverse_forward - identity).abs().amax(dim=(-2, -1)),
        )
        forward_scale = validation_forward.abs().sum(dim=-1).amax(dim=-1)
        inverse_scale = validation_inverse.abs().sum(dim=-1).amax(dim=-1)
        if self.block_diagonal:
            # Dense block-diagonal infinity norms and residual use the maximum across heads.
            # Retain the full K tolerance, rather than giving each head a different contract.
            residual = residual.amax(dim=-1)
            forward_scale = forward_scale.amax(dim=-1)
            inverse_scale = inverse_scale.amax(dim=-1)
        conditioning_scale = torch.clamp(forward_scale * inverse_scale, min=1.0)
        unbounded_tolerance = (
            64.0
            * float(dimension)
            * torch.finfo(self.forward.dtype).eps
            * conditioning_scale
        )
        tolerance = torch.clamp(unbounded_tolerance, max=1e-2)
        flags = torch.stack((
            torch.isfinite(self.forward).all() & torch.isfinite(self.inverse).all(),
            torch.isfinite(forward_inverse).all() & torch.isfinite(inverse_forward).all(),
            (torch.isfinite(residual).all() & torch.isfinite(conditioning_scale).all()
             & torch.isfinite(unbounded_tolerance).all()),
            torch.all(residual <= tolerance),
        ))
        # The verdict only raises, so inside deferred_host_checks it joins the next mandatory host read.
        policy = torch.get_float32_matmul_precision()
        host_check(flags, lambda host: self._frame_verdict(host, residual, tolerance, policy))

    def _frame_verdict(self, host: List[bool], residual: torch.Tensor, tolerance: torch.Tensor,
                       policy: str) -> None:
        """Raise the first failed frame contract of ``__post_init__`` from its host-side flags."""
        finite_factors, finite_products, finite_validation, within_tolerance = host
        if not finite_factors:
            raise ValueError("canonical frame factors must contain only finite values")
        if not finite_products:
            raise ValueError("canonical frame factors must produce finite products")
        if not finite_validation:
            raise ValueError(
                "canonical frame inverse validation must have finite residuals and tolerances"
            )
        if not within_tolerance:
            hint = ""
            if self.forward.dtype == torch.float32:
                hint = (f"; fp32_matmul_precision is {policy!r}, so a producer that composed this "
                        "pair outside vfe3.runtime.ieee_fp32_matmul rounded it at TF32/bf16 "
                        "precision" if policy != "highest" else
                        "; a float32 frame this ill-conditioned needs "
                        "VFE3Config.frame_pair_escalation_tol (e.g. 1e-4)")
            raise ValueError(
                "canonical forward and inverse factors must be mutual inverses within a "
                "dtype-scaled residual; got max residual "
                f"{float(residual.max().detach().cpu()):.6g} and tolerance "
                f"{float(tolerance.max().detach().cpu()):.6g}{hint}"
            )


@dataclass(frozen=True)
class BlockMLPBuildMetadata:
    """Immutable executable BlockMLP structure captured when a model is built.

    ``flops_per_token`` is the registry's analytic per-token proxy in local coordinates; for
    ``prototype_agents`` it is the mean-field cost and does not represent ``block_mlp_inference='exact'``
    or ``block_coordinates='reference'``.
    """

    enabled: bool
    mode: str
    covariance: str
    expansion: int
    activation: str
    dropout: float
    covariance_floor: float
    embed_dim: int
    irrep_dims: Tuple[int, ...]
    n_layers: int
    parameter_count: int
    flops_per_token: float
    report_label: str
    covariance_report_label: str
    gauge_class: str
    intertwiner_compatible: bool


@dataclass(frozen=True)
class ExecutableBuildMetadata:
    """Small immutable description of components actually constructed by VFEModel."""

    block_mlp: BlockMLPBuildMetadata


# Model families the shared training loop, finalize and the paper assemblers know. A model or config
# without a ``model_family`` attribute (every VFEModel and VFE3Config) is the VFE family, so the VFE
# paths never read the tag and stay exactly as they were.
VFE_MODEL_FAMILY: str = "vfe"


def model_family_of(obj: object) -> str:
    r"""The model family of a model, a config, or a serialized config mapping (``"vfe"`` when untagged)."""
    if isinstance(obj, Mapping):
        family = obj.get("model_family", VFE_MODEL_FAMILY)
    else:
        family = getattr(obj, "model_family", VFE_MODEL_FAMILY)
    return VFE_MODEL_FAMILY if family is None else str(family)


def is_vfe_model(model: object) -> bool:
    r"""Whether ``model`` is a VFE model (the gate of every VFE-specific call in the shared loop)."""
    return model_family_of(model) == VFE_MODEL_FAMILY


def is_vfe_config(cfg: object) -> bool:
    r"""Whether ``cfg`` (a config object or a serialized mapping) belongs to the VFE family."""
    return model_family_of(cfg) == VFE_MODEL_FAMILY


class MStepCapture(TypedDict, total=False):
    """Mutable intermediates captured for the M-step self-coupling term."""

    block_mlp_input:    BeliefState
    block_mlp_output:   BeliefState
    converged:          BeliefState
    final_block_prior:  Tuple[torch.Tensor, torch.Tensor]
    final_block_tau:    'float | torch.Tensor'
    prior:              BeliefState
    out:                BeliefState
    canonical_frame:    CanonicalFrameContext
    relational_capture: dict
    # CG moment-energy participation (PB-13). Present ONLY when cfg.cg_energy_weight>0. Each block
    # appends its D(q_post||q_pre) rows to ``cg_moment_energy_rows``. A capture allocated only for
    # M-step self-coupling never carries the key.
    cg_moment_energy_rows: List[torch.Tensor]
    # Prototype-template EM M-step term. Present ONLY when cfg.block_mlp_em_weight>0 on a prototype_agents
    # block MLP; each block appends its (B, N) rows Q_i (PrototypeAgentBlockMLP.em_objective).
    block_mlp_em_rows: List[torch.Tensor]


class EStepGradientRecord(TypedDict, total=False):
    """Detached gradient norms; None denotes an unavailable predictive gradient.

    Missing keys retain the historical unrecorded zero at the public output boundary.
    Explicit None values become Python NaNs there without allocating device scalars.
    """

    mu:    Optional[torch.Tensor]
    sigma: Optional[torch.Tensor]
    phi:   Optional[torch.Tensor]


class EStepGradientOutput(TypedDict, total=False):
    """Host float-valued E-step gradient norms exposed by the model API."""

    mu:    float
    sigma: float
    phi:   float


class DataStateBuffer(TypedDict, total=False):
    """Load-time buffer populated only when a checkpoint carries iterator state."""

    epoch_start_generator_state: Optional[torch.Tensor]
    batches_consumed:            int
    epoch:                       int
    data_identity:               Dict[str, object]
    window_offset:               int        # train_window_offset='per_epoch_random': the epoch's origin


class DataState(TypedDict):
    """Required iterator state written into a resumable checkpoint."""

    epoch_start_generator_state: Optional[torch.Tensor]
    batches_consumed:            int
    epoch:                       int
    data_identity:               Dict[str, object]
