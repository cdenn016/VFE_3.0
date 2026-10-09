"""All supported attention projections have fixed full input and output."""
from tests.profiles import make_relational_operators
from dataclasses import asdict

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.config import config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import _relational_artifact_fields


@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_supported_attention_has_fixed_full_shapes_and_metadata(mode: str) -> None:
    cfg = tiny_config(attention_mode=mode, n_layers=2, predictive_covariance="passthrough",
                      attention_residual=mode == "projected_predictive")
    model = VFEModel(cfg)
    for ops in model.relational_operators_by_layer:
        assert ops.input_scope == ops.output_scope == "full"
        assert ops.raw_score_q.shape == ops.raw_score_k.shape == ops.T.shape == (2, 2, 4)
        assert ops.O.shape == (2, 4, 2)
    metadata = _relational_artifact_fields(cfg)["relational_operators"]
    assert metadata["input_scope"] == metadata["output_scope"] == "full"
    assert metadata["score_shape"] == [2, 2, 4]
    assert metadata["content_shapes"] == [[2, 2, 4], [2, 4, 2]]
    assert metadata["total_parameters"] == sum(p.numel() for ops in
        model.relational_operators_by_layer for p in ops.parameters())


@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_old_full_serialized_scopes_migrate_without_parameter_changes(mode: str) -> None:
    cfg = tiny_config(attention_mode=mode, predictive_covariance="passthrough",
                      attention_residual=mode == "projected_predictive")
    payload = serialize_config(cfg)
    payload.update(relational_input_scope="full", relational_output_scope="full",
                   relational_identity_content=False)
    restored = config_from_serialized(payload, source="old full attention fixture")
    assert asdict(restored) == asdict(cfg)
    source, target = VFEModel(cfg), VFEModel(restored)
    target.load_state_dict(source.state_dict(), strict=True)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["relational_input_scope", "relational_output_scope"])
def test_old_head_serialized_scopes_are_rejected(field: str) -> None:
    payload = serialize_config(tiny_config())
    payload[field] = "head"
    with pytest.raises(ValueError, match="retired|full"):
        config_from_serialized(payload, source="old head attention fixture")


@pytest.mark.parametrize("field", ["input_scope", "output_scope", "identity_content"])
def test_operator_does_not_accept_removed_architecture_options(field: str) -> None:
    with pytest.raises(TypeError, match=field):
        make_relational_operators([2, 2], 11, **{field: "head"})
