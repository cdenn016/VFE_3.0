r"""Configuration migration and validator consistency (audit 2026-10-16, F1 and F5).

F1. A serialized config that predates a field takes the introduction-time value of that field, not today's default
(``vfe3.config._HISTORICAL_DEFAULT_PINS``). ``block_norm_covariance`` was introduced at 89eedfb (2026-10-10) as
``"passthrough"``, so a record saved before then (the 36.41 paper base among them) ran the mean-only block norm and
must reload as ``"passthrough"`` even once the default becomes ``"pushforward"``. ``predictive_covariance_floor_mode``
(322843b, 2026-10-16) selects a floor that changed earlier, at 5f5d484 (2026-10-12): a record lacking the field is
from either era, and 5f5d484 also added ``phi_frame_condition_max``, so a record lacking both ran the relative floor
and one carrying ``phi_frame_condition_max`` ran the prior-anchored floor.

F5. ``inactive_fields`` marks ``block_norm_covariance`` inert under a block norm other than ``'layernorm'``; the
validator must then not refuse either of its values there. Under a LayerNorm in reference coordinates on the
diagonal family the field is read (and only ``"passthrough"`` is supported), so it is not marked inert there.

Config level, except one tiny CPU forward under ``norm_type_block='none'``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest
import torch

from tests.profiles import test_config, tiny_config
import vfe3.config as config_module
from vfe3.config import _HISTORICAL_DEFAULT_PINS, VFE3Config, config_from_serialized, serialize_config
from vfe3.legacy_config import lenient_config_from_serialized


@pytest.fixture(autouse=True)
def _quiet():
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@dataclass
class _PushforwardDefaultConfig(VFE3Config):
    r"""``VFE3Config`` with ``block_norm_covariance`` defaulting to ``"pushforward"`` (the pending default flip)."""
    block_norm_covariance: str = "pushforward"


def historical_default_pins(payload: dict) -> dict:
    r"""The values the loader fills into ``payload`` (:func:`vfe3.config.historical_default_pins`)."""
    return config_module.historical_default_pins(payload)


def _loaders(payload: dict) -> list[VFE3Config]:
    r"""``payload`` through the strict and the lenient loader."""
    strict = config_from_serialized(payload, source="test")
    lenient, dropped = lenient_config_from_serialized(payload, source="test")
    assert dropped == {}
    return [strict, lenient]


def _moment_matched(**overrides) -> VFE3Config:
    return test_config(attention_mode="projected_predictive", predictive_readout="kernel",
                       predictive_covariance="moment_matched", **overrides)


# ----------------------------------------------------------------------------------------------
# F1: block_norm_covariance
# ----------------------------------------------------------------------------------------------

def test_block_norm_covariance_is_pinned_at_its_introduction_value() -> None:
    assert _HISTORICAL_DEFAULT_PINS["block_norm_covariance"] == "passthrough"


@pytest.mark.parametrize("norm", ["layernorm", "none", "mahalanobis"])
def test_a_record_predating_block_norm_covariance_reloads_as_passthrough(
        norm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = serialize_config(test_config(norm_type_block=norm))
    del payload["block_norm_covariance"]
    monkeypatch.setattr("vfe3.config.VFE3Config", _PushforwardDefaultConfig)
    for cfg in _loaders(payload):
        assert cfg.block_norm_covariance == "passthrough"
    assert historical_default_pins(payload)["block_norm_covariance"] == "passthrough"


def test_a_record_carrying_block_norm_covariance_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vfe3.config.VFE3Config", _PushforwardDefaultConfig)
    for saved in ("passthrough", "pushforward"):
        payload = serialize_config(test_config(block_norm_covariance=saved))
        for cfg in _loaders(payload):
            assert cfg.block_norm_covariance == saved
        assert "block_norm_covariance" not in historical_default_pins(payload)


# ----------------------------------------------------------------------------------------------
# F1: predictive_covariance_floor_mode, keyed on the 5f5d484 era marker
# ----------------------------------------------------------------------------------------------

def test_a_record_from_before_the_floor_relocation_reloads_with_the_relative_floor() -> None:
    payload = serialize_config(_moment_matched())
    del payload["predictive_covariance_floor_mode"], payload["phi_frame_condition_max"]
    for cfg in _loaders(payload):
        assert cfg.predictive_covariance_floor_mode == "relative"
        assert "predictive_covariance_floor_mode" not in cfg.inactive_fields()     # the floor is live here
        assert cfg.phi_frame_condition_max is None
    assert historical_default_pins(payload)["predictive_covariance_floor_mode"] == "relative"


@pytest.mark.parametrize("phi_frame_condition_max", [None, 1e4])
def test_a_record_from_after_the_floor_relocation_keeps_the_prior_floor(phi_frame_condition_max) -> None:
    r"""2026-10-12 to 2026-10-16 saves lack the floor mode but carry ``phi_frame_condition_max``; they ran ``prior``."""
    payload = serialize_config(_moment_matched(gauge_group="block_glk", m_phi_update_mode="adamw",
                                               phi_frame_condition_max=phi_frame_condition_max))
    del payload["predictive_covariance_floor_mode"]
    for cfg in _loaders(payload):
        assert cfg.predictive_covariance_floor_mode == "prior"
    assert "predictive_covariance_floor_mode" not in historical_default_pins(payload)


@pytest.mark.parametrize("mode", ["prior", "relative"])
def test_a_recorded_floor_mode_is_never_rewritten(mode: str) -> None:
    payload = serialize_config(_moment_matched(predictive_covariance_floor_mode=mode))
    del payload["phi_frame_condition_max"]
    for cfg in _loaders(payload):
        assert cfg.predictive_covariance_floor_mode == mode
    assert "predictive_covariance_floor_mode" not in historical_default_pins(payload)


def test_a_current_record_round_trips_unchanged() -> None:
    cfg = _moment_matched()
    payload = serialize_config(cfg)
    for loaded in _loaders(payload):
        assert loaded == cfg
    assert historical_default_pins(payload) == {}


# ----------------------------------------------------------------------------------------------
# F5: the validator agrees with inactive_fields
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["none", "mahalanobis"])
@pytest.mark.parametrize("value", ["passthrough", "pushforward"])
def test_block_norm_covariance_is_accepted_wherever_it_is_marked_inert(norm: str, value: str) -> None:
    cfg = test_config(norm_type_block=norm, block_norm_covariance=value)
    assert "block_norm_covariance" in cfg.inactive_fields()
    if norm == "none":                                   # nothing to conjugate: reference coordinates too
        ref = test_config(norm_type_block=norm, block_norm_covariance=value, attention_mode="projected_predictive",
                          predictive_readout="kernel", block_coordinates="reference")
        assert "block_norm_covariance" in ref.inactive_fields()


def test_the_norm_free_pushforward_is_inert_in_fact() -> None:
    r"""Under ``norm_type_block='none'`` no block norm runs, so the two values give the same loss and gradients."""
    tokens = torch.tensor([[0, 3, 5]])
    results = []
    for value in ("passthrough", "pushforward"):
        from vfe3.model.model import VFEModel
        torch.manual_seed(0)
        model = VFEModel(tiny_config(norm_type_block="none", use_block_mlp=True, block_mlp_mode="coordinate",
                                     block_norm_covariance=value))
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        results.append((loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    (loss_a, grads_a), (loss_b, grads_b) = results
    assert torch.equal(loss_a, loss_b)
    assert len(grads_a) == len(grads_b) and all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))


def test_the_layernorm_pushforward_keeps_its_live_refusals() -> None:
    r"""Under a LayerNorm the field is read; reference coordinates on the diagonal family support only passthrough."""
    reference = dict(attention_mode="projected_predictive", predictive_readout="kernel", block_coordinates="reference",
                     norm_type_block="layernorm")
    with pytest.raises(ValueError, match="block_norm_covariance='pushforward' needs block_coordinates='local'"):
        test_config(block_norm_covariance="pushforward", **reference)
    assert "block_norm_covariance" not in test_config(**reference).inactive_fields()
    assert "block_norm_covariance" not in test_config(norm_type_block="layernorm").inactive_fields()
