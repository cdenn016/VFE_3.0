"""Focused regressions for the VFE 3.0 transport cleanup."""
import pytest
import torch

from vfe3.geometry import transport
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import _transport, build_belief_transport


def test_batch_independent_transport_preserves_both_token_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "cleanup_batch_independent"
    monkeypatch.setattr(transport, "_TRANSPORTS", dict(transport._TRANSPORTS))
    group = get_group("block_glk")(2, 1)
    expected = torch.eye(2).expand(3, 3, 2, 2).clone()
    @transport.register_transport(name, covariance_class="test", batch_independent=True)
    def build(phi: torch.Tensor, group: object, **kwargs: object) -> dict:
        return {"Omega": expected}
    actual = _transport(torch.zeros(3, 4), group, transport_mode=name)
    assert actual.shape == (3, 3, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_right_phi_cannot_be_a_registered_state_key() -> None:
    with pytest.raises(ValueError, match="reserved.*right_phi"):
        transport.register_transport(
            "cleanup_right_phi", covariance_class="test", serialization_keys=("right_phi",),
            state_builder=lambda cfg, group: {})


def test_overridden_transport_routes_current_registration_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "cleanup_override"
    monkeypatch.setattr(transport, "_TRANSPORTS", dict(transport._TRANSPORTS))
    group = get_group("block_glk")(2, 1)
    phi = torch.zeros(1, 3, 4)
    mu = torch.randn(1, 3, 2)
    sigma = torch.ones_like(mu)
    seen = []
    def build(phi: torch.Tensor, group: object, **kwargs: object) -> dict:
        seen.append((kwargs.get("mu"), kwargs.get("sigma")))
        return {"Omega": torch.eye(2).expand(1, 3, 3, 2, 2)}
    transport.register_transport(name, covariance_class="test", needs_mu=True)(build)
    build_belief_transport(phi, group, transport_mode=name, mu=mu, sigma=sigma)
    transport.register_transport(name, covariance_class="test", needs_sigma=True, override=True)(build)
    build_belief_transport(phi, group, transport_mode=name, mu=mu, sigma=sigma)
    assert seen[0][0] is mu and seen[0][1] is None
    assert seen[1][0] is None and seen[1][1] is sigma


@pytest.mark.parametrize("compact", [False, True])
def test_float64_input_is_exponentiated_and_stored_at_float64(
    monkeypatch: pytest.MonkeyPatch, compact: bool,
) -> None:
    matrix = torch.tensor([[0.04, 0.02], [-0.03, -0.01]], dtype=torch.float64)
    reference = torch.linalg.matrix_exp(matrix)
    seen = []
    original = torch.linalg.matrix_exp
    def recorded(value: torch.Tensor) -> torch.Tensor:
        seen.append(value.dtype)
        return original(value)
    monkeypatch.setattr(torch.linalg, "matrix_exp", recorded)
    if compact:
        actual, inverse = transport._stable_compact_glk_exp_pair(matrix.reshape(1, 1, 1, 2, 2))
        actual, inverse = actual[0, 0, 0], inverse[0, 0, 0]
    else:
        actual, inverse = transport.stable_matrix_exp_pair(matrix)
    assert seen == [torch.float64]
    assert actual.dtype == inverse.dtype == torch.float64
    torch.testing.assert_close(actual @ inverse, torch.eye(2, dtype=torch.float64), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_predictive_gradient_capture_uses_host_nan_and_preserves_real_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import math
    from tests.profiles import tiny_config
    from vfe3.model.model import VFEModel
    module = importlib.import_module("vfe3.inference.e_step")
    original = module.e_step_iteration
    records = []
    def recorded(*args: object, **kwargs: object) -> object:
        output = original(*args, **kwargs)
        if kwargs.get("grad_record") is not None:
            records.append(dict(kwargs["grad_record"]))
        return output
    monkeypatch.setattr(module, "e_step_iteration", recorded)
    cfg = tiny_config(attention_mode="projected_predictive", n_layers=6,

                      skip_belief_sigma_update=True)
    model = VFEModel(cfg)
    out = {}
    model.forward_beliefs(torch.tensor([[0, 1, 2]]), estep_grad_out=out)
    assert len(records) == 6
    assert all(record == {"mu": None, "sigma": None, "phi": None} for record in records)
    assert all(type(out[key]) is float and math.isnan(out[key]) for key in ("mu", "sigma", "phi"))
    records.clear()
    ordinary = VFEModel(tiny_config(
        attention_mode="projected_gradient",
         attention_residual=False,
        predictive_covariance="passthrough",
    ))
    ordinary.forward_beliefs(torch.tensor([[0, 1, 2]]), estep_grad_out=out)
    assert records and isinstance(records[-1]["mu"], torch.Tensor)
    assert all(type(out[key]) is float and math.isfinite(out[key]) for key in ("mu", "sigma", "phi"))
