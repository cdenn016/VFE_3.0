r"""The variance guard columns report only clamps the route applies (audit 2026-10-16 F52).

``guard_sigma_floor_frac`` and ``guard_sigma_ceil_frac`` count converged variances within a relative window of
``eps`` and ``sigma_max``. The ceiling is the sigma E-step's SPD bound, inert wherever ``inactive_fields`` marks
``sigma_max``; a moment-matched readout adds the prior-anchored floor :math:`f\Sigma_p` and clamps nothing at
``eps``. On those routes the columns are NaN. Tiny CPU models only (K = 8, two heads of 4, N = 5, vocab 9).
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


def _model(**overrides: Any) -> VFEModel:
    values = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=2, phi_scale=0.4)
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).eval()


def _guard(model: VFEModel) -> dict:
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    sigma = snapshot.layer_converged[-1].sigma.double()
    empty = sigma.new_empty(0)
    return {"diagnostics": model.diagnostics(TOKENS, snapshot=snapshot),
            "direct": metrics.guard_saturation(sigma, empty, empty, diagonal=model.cfg.diagonal_covariance,
                                               eps=model.cfg.eps, sigma_max=model.cfg.sigma_max)}


def test_moment_matched_route_reports_neither_guard() -> None:
    model = _model()
    assert model.cfg.attention_mode == "projected_predictive"
    assert model.cfg.predictive_covariance == "moment_matched"
    assert metrics.variance_guard_activity(model.cfg) == (False, False)
    d = _guard(model)["diagnostics"]
    assert math.isnan(d["guard_sigma_floor_frac"]) and math.isnan(d["guard_sigma_ceil_frac"])


def test_predictive_passthrough_keeps_the_encode_floor_and_drops_the_ceiling() -> None:
    model = _model(predictive_covariance="passthrough")
    assert "sigma_max" in model.cfg.inactive_fields()
    assert metrics.variance_guard_activity(model.cfg) == (True, False)
    out = _guard(model)
    assert out["diagnostics"]["guard_sigma_floor_frac"] == out["direct"]["sigma_floor_frac"]
    assert math.isnan(out["diagnostics"]["guard_sigma_ceil_frac"])


@pytest.mark.parametrize("sigma_max", [100.0, None], ids=["bounded", "unbounded"])
def test_sigma_e_step_route_reports_its_clamps(sigma_max: Any) -> None:
    model = _model(attention_mode="projected_gradient", predictive_covariance="passthrough",
                   sigma_max=sigma_max)
    assert "sigma_max" not in model.cfg.inactive_fields()
    assert metrics.variance_guard_activity(model.cfg) == (True, sigma_max is not None)
    out = _guard(model)
    assert out["diagnostics"]["guard_sigma_floor_frac"] == out["direct"]["sigma_floor_frac"]
    ceiling = out["diagnostics"]["guard_sigma_ceil_frac"]
    assert math.isnan(ceiling) if sigma_max is None else ceiling == out["direct"]["sigma_ceil_frac"]
