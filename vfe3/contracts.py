"""Shared type contracts for mutable runtime dictionaries."""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, TypedDict

import torch

from vfe3.belief import BeliefState


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
        if not torch.isfinite(self.forward).all() or not torch.isfinite(self.inverse).all():
            raise ValueError("canonical frame factors must contain only finite values")

        validation_dtype = torch.float64
        validation_forward = self.forward.to(dtype=validation_dtype)
        validation_inverse = self.inverse.to(dtype=validation_dtype)
        block_dim = self.forward.shape[-1]
        dimension = block_dim * self.forward.shape[-3] if self.block_diagonal else block_dim
        identity = torch.eye(
            block_dim,
            dtype=validation_dtype,
            device=self.forward.device,
        )
        forward_inverse = validation_forward @ validation_inverse
        inverse_forward = validation_inverse @ validation_forward
        if (
            not torch.isfinite(forward_inverse).all()
            or not torch.isfinite(inverse_forward).all()
        ):
            raise ValueError("canonical frame factors must produce finite products")
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
        if (
            not torch.isfinite(residual).all()
            or not torch.isfinite(conditioning_scale).all()
            or not torch.isfinite(unbounded_tolerance).all()
        ):
            raise ValueError(
                "canonical frame inverse validation must have finite residuals and tolerances"
            )
        tolerance = torch.clamp(unbounded_tolerance, max=1e-2)
        if not bool(torch.all(residual <= tolerance)):
            raise ValueError(
                "canonical forward and inverse factors must be mutual inverses within a "
                "dtype-scaled residual; got max residual "
                f"{float(residual.max().detach().cpu()):.6g} and tolerance "
                f"{float(tolerance.max().detach().cpu()):.6g}"
            )


@dataclass(frozen=True)
class BlockMLPBuildMetadata:
    """Immutable executable BlockMLP structure captured when a model is built."""

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


class DataState(TypedDict):
    """Required iterator state written into a resumable checkpoint."""

    epoch_start_generator_state: Optional[torch.Tensor]
    batches_consumed:            int
    epoch:                       int
    data_identity:               Dict[str, object]
