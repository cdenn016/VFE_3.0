"""Opt-in centering of the predictive covariance's between-source spread."""

import inspect

import pytest
import torch

from tests.profiles import config_values, tiny_config
from vfe3.config import ConfigNotice, VFE3Config, config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators


def _config(**overrides: object) -> VFE3Config:
    # Pin this feature independently of the launcher defaults and the ongoing test overhaul.
    values = dict(attention_mode="projected_predictive", predictive_covariance="moment_matched",
                  family="gaussian_diagonal", decode_mode="linear", gradient_mode="filtering",
                  e_step_update="gradient", lambda_alpha_mode="constant", attention_residual=False,
                  )
    values.update(overrides)
    return tiny_config(**values)


def _operator(centered: bool) -> RelationalOperators:
    ops = RelationalOperators(
        [2], 3, readout_mode="predictive_content",
        predictive_covariance="moment_matched", predictive_covariance_centering=centered,
    )
    with torch.no_grad():
        ops.T.copy_(torch.eye(2).unsqueeze(0))
        ops.O.copy_(torch.eye(2).unsqueeze(0))
    return ops


def _frame(mu: torch.Tensor) -> CanonicalFrameContext:
    identity = torch.eye(2, dtype=mu.dtype).expand(1, 3, 2, 2)
    return CanonicalFrameContext(identity, identity)


@pytest.mark.parametrize("centered", [False, True])
def test_config_wires_centering_to_every_layer(centered: bool) -> None:
    cfg = _config(n_layers=2, predictive_covariance_centering=centered)
    model = VFEModel(cfg).cpu()
    operators = [model.relational_operators, *model.relational_operators_extra]
    assert all(ops.predictive_covariance_centering is centered for ops in operators)
    belief, _ = model.forward_beliefs(torch.tensor([[1, 2, 3]]), return_logits=False)
    assert torch.isfinite(belief.sigma).all()
    restored = config_from_serialized(serialize_config(cfg), source="centering test")
    assert restored.predictive_covariance_centering is centered


def test_missing_centering_field_uses_local_historical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = list(inspect.signature(VFE3Config.__init__).parameters)[1:]
    values = config_values(predictive_covariance_centering=False)
    monkeypatch.setattr(VFE3Config.__init__, "__defaults__", tuple(values[name] for name in parameters))
    payload = serialize_config(_config(predictive_covariance_centering=True))
    payload.pop("predictive_covariance_centering")
    restored = config_from_serialized(payload, source="legacy centering test")
    assert restored.predictive_covariance_centering is False


@pytest.mark.parametrize("centered", [False, True])
def test_centering_selects_covariance_calculation(centered: bool) -> None:
    # The historical float32 raw moments lose this exactly representable weighted spread.
    mu = torch.tensor([[[9999., 10001.], [10000., 10000.], [10001., 9999.]]])
    sigma = torch.ones_like(mu)
    beta = torch.tensor([.25, .5, .25]).expand(1, 1, 3, 3)
    actual = _operator(centered).predictive_covariance_update(mu, sigma, beta, _frame(mu))
    expected = torch.full_like(mu, 1.0001 + (0.5 if centered else 0.0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("centered", [False, True])
def test_centering_gradients_through_normalized_routing(centered: bool) -> None:
    ops = _operator(centered).double()
    mu = torch.tensor([[[.2, -.1], [.4, .3], [-.2, .5]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.ones_like(mu)
    logits = torch.linspace(-.4, .4, 9, dtype=torch.float64).reshape(1, 1, 3, 3).requires_grad_()
    frame = _frame(mu)

    def covariance(mu_in: torch.Tensor, logits_in: torch.Tensor) -> torch.Tensor:
        return ops.predictive_covariance_update(mu_in, sigma, logits_in.softmax(-1), frame)

    assert torch.autograd.gradcheck(covariance, (mu, logits), fast_mode=True)


def test_centering_requires_a_boolean() -> None:
    with pytest.raises(ValueError, match="predictive_covariance_centering must be a plain bool"):
        _config(predictive_covariance_centering="false")
    with pytest.raises(ValueError, match="predictive_covariance_centering must be a bool"):
        _operator(1)


def test_centering_is_inactive_under_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    # The notice compares an explicit override with the declared test-local default.
    monkeypatch.setattr(VFE3Config.__dataclass_fields__["predictive_covariance_centering"], "default", False)
    with pytest.warns(ConfigNotice, match="predictive_covariance_centering"):
        cfg = _config(predictive_covariance="passthrough", predictive_covariance_centering=True)
    assert "predictive_covariance_centering" in cfg.inactive_fields()
    live = _config(predictive_covariance_centering=True)
    assert "predictive_covariance_centering" not in live.inactive_fields()
