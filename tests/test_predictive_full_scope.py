"""Predictive projections are fixed full/full; inference scopes remain separate."""
from dataclasses import asdict

import pytest
import torch

from tests.profiles import tiny_config
from tests.test_relational_integration import _cfg
from vfe3.config import config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import _relational_artifact_fields


def test_predictive_operator_defaults_match_existing_full_initialization() -> None:
    actual = RelationalOperators([2, 2], 11, readout_mode="predictive_content")
    expected = RelationalOperators([2, 2], 11, input_scope="full", output_scope="full",
                                   readout_mode="predictive_content")
    assert actual.input_scope == actual.output_scope == "full"
    assert actual.raw_score_q.shape == actual.raw_score_k.shape == actual.T.shape == (2, 2, 4)
    assert actual.O.shape == (2, 4, 2)
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["relational_input_scope", "relational_output_scope"])
def test_predictive_config_rejects_legacy_head_architecture(field: str) -> None:
    with pytest.raises(ValueError, match="predictive.*full"):
        tiny_config(**{field: "head"})
    payload = serialize_config(tiny_config())
    payload[field] = "head"
    with pytest.raises(ValueError, match="predictive.*full"):
        config_from_serialized(payload, source="legacy predictive head fixture")


@pytest.mark.parametrize("field", ["input_scope", "output_scope"])
def test_predictive_operator_rejects_explicit_head_architecture(field: str) -> None:
    with pytest.raises(ValueError, match="predictive.*full"):
        RelationalOperators([2, 2], 11, readout_mode="predictive_content", **{field: "head"})


def test_predictive_model_wiring_does_not_read_conditional_scope_overrides() -> None:
    cfg = tiny_config(n_layers=2, predictive_covariance="passthrough")
    cfg.relational_input_scope = cfg.relational_output_scope = "head"
    model = VFEModel(cfg)
    for ops in (model.relational_operators, *model.relational_operators_extra):
        assert ops.input_scope == ops.output_scope == "full"
        assert ops.T.shape == (2, 2, 4)
        assert ops.O.shape == (2, 4, 2)
    metadata = _relational_artifact_fields(cfg)["relational_operators"]
    assert metadata["input_scope"] == metadata["output_scope"] == "full"
    assert metadata["score_shape"] == [2, 2, 4]
    assert metadata["content_shapes"] == [[2, 2, 4], [2, 4, 2]]
    assert metadata["total_parameters"] == sum(p.numel() for ops in
        (model.relational_operators, *model.relational_operators_extra) for p in ops.parameters())


def test_full_predictive_config_roundtrip_and_conditional_scopes() -> None:
    cfg = tiny_config()
    assert asdict(config_from_serialized(serialize_config(cfg), source="full predictive fixture")) == asdict(cfg)
    for scope in ("head", "full"):
        conditional = _cfg(relational_input_scope=scope)
        ops = VFEModel(conditional).relational_operators
        assert ops.input_scope == scope
        assert ops.output_scope == "head"
    transported = _cfg(attention_mode="transported", e_step_update="mm_exact")
    assert VFEModel(transported).relational_operators is None
