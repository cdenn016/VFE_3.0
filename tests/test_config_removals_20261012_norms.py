"""Removal of the final norm, the post-MLP norm, the block-MLP autocast toggle and the non-causal attention priors
(October 12, 2026, second pass).

``norm_type_final`` is fixed at none, ``block_mlp_post_norm`` at False and ``block_mlp_autocast`` at False (the block
MLP's mean path runs in an fp32 island under ``amp_dtype``). A saved payload that names the removed fields loads to
the current config when the saved value is the hardcoded one (any value of the precision policy
``block_mlp_autocast``) and is refused otherwise; a saved non-causal attention prior is refused.
"""
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from tests.profiles import config_values, test_config
from tests.test_config_retirements_20261012 import _assert_migrates, _old_payload
from vfe3.attention_prior import _PRIORS, get_prior_registration
from vfe3.config import RETIRED_CONFIG_KEYS, VFE3Config, config_from_serialized
from vfe3.model.model import VFEModel

REMOVED = ("norm_type_final", "block_mlp_post_norm", "block_mlp_autocast")
REMOVED_PRIORS = ("uniform", "alibi", "windowed")


def test_fields_are_gone_and_registered_as_retired() -> None:
    names = {field.name for field in fields(VFE3Config)}
    for name in REMOVED:
        assert name not in names
        assert name not in config_values()
        assert name in RETIRED_CONFIG_KEYS
        with pytest.raises(TypeError):
            test_config(**{name: "none" if name == "norm_type_final" else False})


def test_model_has_no_final_norm() -> None:
    model = VFEModel(test_config())
    assert not hasattr(model, "final_norm")
    assert not any(name.startswith("final_norm") for name, _ in model.named_parameters())


@pytest.mark.parametrize("old", [
    dict(norm_type_final="none"),
    dict(block_mlp_post_norm=False),
    dict(block_mlp_post_norm="false"),
    dict(block_mlp_autocast=True),
    dict(block_mlp_autocast=False),
    dict(norm_type_final="none", block_mlp_post_norm=False, block_mlp_autocast=True),
])
def test_hardcoded_values_load_silently(old: dict[str, Any], tmp_path: Path) -> None:
    cfg = test_config()
    _assert_migrates(_old_payload(cfg, **old), cfg, tmp_path)


@pytest.mark.parametrize("old, message", [
    (dict(norm_type_final="layernorm"), "norm_type_final"),
    (dict(norm_type_final="mahalanobis"), "norm_type_final"),
    (dict(block_mlp_post_norm=True), "block_mlp_post_norm"),
    (dict(block_mlp_post_norm="true"), "block_mlp_post_norm"),
])
def test_forward_changing_values_are_refused(old: dict[str, Any], message: str) -> None:
    payload = _old_payload(test_config(), **old)
    with pytest.raises(ValueError, match=f"retired {message}"):
        config_from_serialized(payload, source="test")


def test_non_causal_priors_are_gone_and_every_registered_prior_is_causal() -> None:
    for name in REMOVED_PRIORS:
        assert name not in _PRIORS
        with pytest.raises(ValueError, match="beta_attention_prior"):
            test_config(beta_attention_prior=name)
    cfg = test_config()
    assert all(get_prior_registration(name).is_causal_for_next_token_scoring(cfg) for name in _PRIORS)


@pytest.mark.parametrize("prior", REMOVED_PRIORS)
def test_saved_non_causal_prior_is_refused(prior: str) -> None:
    payload = _old_payload(test_config(), beta_attention_prior=prior)
    with pytest.raises(ValueError, match="retired beta_attention_prior"):
        config_from_serialized(payload, source="test")
