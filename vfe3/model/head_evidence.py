"""Pure normalized weights for the opt-in PriorBank head-evidence mixer."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadEvidenceWeights:
    """Per-gauge-block and expanded coordinate weights derived from one logit per block."""

    head: Tensor
    coordinate: Tensor

    def minus_identity(self) -> "HeadEvidenceWeights":
        """Return the exact baseline-plus-delta coefficients ``w - 1``."""
        return HeadEvidenceWeights(head=self.head - 1.0, coordinate=self.coordinate - 1.0)


def normalized_head_evidence_weights(
    logits: Tensor,
    irrep_dims: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> HeadEvidenceWeights:
    """Return mean-one, overflow-safe head weights and their coordinate expansion."""
    if logits.ndim != 1 or logits.numel() != len(irrep_dims):
        raise ValueError("head-evidence logits must contain one scalar per gauge block")
    if len(irrep_dims) < 2 or any(int(dim) <= 0 for dim in irrep_dims):
        raise ValueError("head evidence requires at least two positive gauge blocks")
    work = logits.to(device=device, dtype=dtype)
    raw = (work - work.max()).exp()
    head = raw / raw.mean()
    repeats = torch.as_tensor(tuple(int(dim) for dim in irrep_dims), device=device)
    return HeadEvidenceWeights(head=head, coordinate=torch.repeat_interleave(head, repeats))
