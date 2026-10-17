r"""The rank-one residual of the reference means the content mixture reads (audit 2026-10-16 F45).

``relational_rank_one_residual`` is the Dong et al. (2021) residual :math:`r(X)` of a layer's output means in local
coordinates, :math:`\mu_i = U_i\xi_i`. The per-token frame :math:`U_i` moves under the passive gauge action, so token
spread that comes from the frames hides a collapse of the mixed content. ``relational_rank_one_residual_reference``
is :math:`r` of :math:`\xi_i = U_i^{-1}\mu_i` in the content (value) frame, the coordinates the predictive kernel
averages. Tiny CPU models only (K = 8, two heads of 4, N = 5, vocab 9).
"""
import math
import warnings
from typing import Any

import pytest
import torch

from tests.profiles import tiny_config
from vfe3 import metrics
from vfe3.model.model import VFEModel

TOKENS = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
KEY = "relational_rank_one_residual_reference"


def _model(**overrides: Any) -> VFEModel:
    values = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=3, phi_scale=0.4)
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).eval()


def _content_frame_inverse(model: VFEModel, snapshot: Any) -> torch.Tensor:   # (B, N, K, K) float64
    r"""Dense float64 :math:`U_i^{-1} = e^{-Y_i\cdot G}e^{-\phi_i\cdot G}`, with :math:`R_i` folded in only under
    ``rope_mode='frame_value'`` (the content frame of ``'frame'`` is the un-rotated :math:`U_i`)."""
    group, start = model.group, snapshot.initial_belief
    inverse = torch.linalg.matrix_exp(-group.embed_algebra(start.phi.double(), dtype=torch.float64))
    if start.right_phi is not None:
        inverse = torch.linalg.matrix_exp(
            -group.embed_algebra(start.right_phi.double(), dtype=torch.float64)) @ inverse
    if model.cfg.rope_mode == "frame_value":
        inverse = snapshot.rope.double() @ inverse
    return inverse


def _residuals(model: VFEModel) -> tuple:
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    last = model.diagnostics(TOKENS, snapshot=snapshot)
    inverse = _content_frame_inverse(model, snapshot)
    reference = [float(metrics.rank_one_residual((inverse @ out.mu.double().unsqueeze(-1)).squeeze(-1)).mean())
                 for out in snapshot.layer_outputs]
    local = [float(metrics.rank_one_residual(out.mu.double()).mean()) for out in snapshot.layer_outputs]
    return per_layer, last, reference, local


@pytest.mark.parametrize("rope_mode", ["none", "frame", "frame_value"])
def test_reference_residual_reads_the_content_frame(rope_mode: str) -> None:
    model = _model(rope_mode=rope_mode)
    per_layer, last, reference, local = _residuals(model)
    assert per_layer[KEY] == pytest.approx(reference, rel=1e-5)
    assert last[KEY] == per_layer[KEY][-1]
    assert per_layer["relational_rank_one_residual"] == pytest.approx(local, rel=1e-12)
    # The frames are active, so the local-coordinate residual is a different number.
    assert max(abs(a - b) for a, b in zip(reference, local)) > 1e-3


def test_reference_residual_is_the_local_one_without_frames() -> None:
    model = _model(gauge_transport="off")
    per_layer, _, reference, local = _residuals(model)
    assert per_layer[KEY] == pytest.approx(local, rel=1e-6)
    assert all(math.isfinite(value) for value in per_layer[KEY])
