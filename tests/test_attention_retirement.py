"""Contracts for the two-mode full-state attention architecture."""
from dataclasses import fields
from inspect import signature
from typing import Any

import pytest
import torch

from tests.profiles import config_values
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.legacy_config import lenient_config_from_serialized
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators

def _values(**overrides: Any) -> dict[str, Any]:
    """Use an explicit tiny configuration, independent of constructor defaults."""
    values = config_values(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                           eval_stride=3, n_layers=2, n_e_steps=1,
                           attention_mode="projected_predictive", attention_residual=False,
                           predictive_covariance="passthrough", lambda_alpha_mode="constant",
                           norm_type_block="none", norm_type_final="none", use_block_mlp=False,
                           block_mlp_expansion=2, amp_dtype=None, e_phi_lr=0.)
    for name in ("relational_input_scope", "relational_output_scope", "relational_identity_content"):
        values.pop(name, None)
    values.update(overrides)
    return values

def test_scope_and_conditional_identity_are_not_config_fields() -> None:
    names = {field.name for field in fields(VFE3Config)}
    assert not names.intersection({"relational_input_scope", "relational_output_scope",
                                   "relational_identity_content"})

@pytest.mark.parametrize("mode", ["transported", "projected_conditional"])
def test_retired_modes_are_explicitly_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="retired attention"):
        VFE3Config(**_values(attention_mode=mode))

@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_all_layers_use_full_maps_without_scope_arguments(mode: str) -> None:
    cfg = VFE3Config(**_values(attention_mode=mode))
    model = VFEModel(cfg)
    for ops in model.relational_operators_by_layer:
        assert ops.raw_score_q.shape == ops.raw_score_k.shape == ops.T.shape == (2, 2, 4)
        assert ops.O.shape == (2, 4, 2)
    assert not {"input_scope", "output_scope", "identity_content"}.intersection(
        signature(RelationalOperators).parameters)
    assert not {"relational_input_scope", "relational_output_scope",
                "relational_identity_content"}.intersection(serialize_config(cfg))

@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_historical_full_metadata_migrates_without_forward_change(mode: str) -> None:
    cfg = VFE3Config(**_values(attention_mode=mode))
    payload = serialize_config(cfg)
    historical = {**payload, "relational_input_scope": "full",
                  "relational_output_scope": "full", "relational_identity_content": False}
    restored = config_from_serialized(historical, source="historical full fixture")
    assert serialize_config(restored) == payload
    original, reloaded = VFEModel(cfg).eval(), VFEModel(restored).eval()
    reloaded.load_state_dict(original.state_dict(), strict=True)
    ids = torch.tensor([[0, 1, 2]], device="cpu")
    with torch.no_grad():
        before, _ = original.forward_beliefs(ids)
        after, _ = reloaded.forward_beliefs(ids)
    torch.testing.assert_close(before.mu, after.mu, rtol=0, atol=0)
    torch.testing.assert_close(before.sigma, after.sigma, rtol=0, atol=0)

@pytest.mark.parametrize("field", ["relational_input_scope", "relational_output_scope"])
@pytest.mark.parametrize("lenient", [False, True])
def test_historical_head_metadata_is_rejected_even_by_lenient_loader(
    field: str, lenient: bool,
) -> None:
    payload = serialize_config(VFE3Config(**_values()))
    payload[field] = "head"
    loader = lenient_config_from_serialized if lenient else config_from_serialized
    with pytest.raises(ValueError, match="head.*migration|migration.*head"):
        loader(payload, source="historical head fixture")

def test_historical_identity_content_requires_explicit_migration() -> None:
    payload = serialize_config(VFE3Config(**_values()))
    payload["relational_identity_content"] = True
    with pytest.raises(ValueError, match="identity.*migration|migration.*identity"):
        config_from_serialized(payload, source="historical identity fixture")
