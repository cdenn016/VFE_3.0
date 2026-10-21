"""Configuration removals of October 12, 2026 (second pass): the projected_gradient receiver and self-coupling.

``relational_receiver_law`` is gone (the channel likelihood is the only receiver law), the ``constant`` alpha form
and ``lambda_alpha`` are gone, and ``relational_receiver_measurement`` defaults to ``"inverse"`` with a historical
pin of ``"adjoint"`` for configs saved before the field existed. The loader refuses a saved ``ambient_alignment``
on ``projected_gradient`` and a saved ``lambda_alpha_mode="constant"``; every other occurrence of the removed keys
was inert and is dropped.
"""
from dataclasses import fields
from typing import Any

import pytest

from tests.profiles import test_config
from vfe3.alpha_i import _ALPHAS
from vfe3.config import (
    RETIRED_CONFIG_KEYS, VFE3Config, _HISTORICAL_DEFAULT_PINS, config_from_serialized, serialize_config,
)
from vfe3.legacy_config import lenient_config_from_serialized

_REMOVED = ("relational_receiver_law", "lambda_alpha")
_PREDICTIVE = dict(attention_mode="projected_predictive")


def _payload(cfg: VFE3Config, removed: tuple[str, ...] = (), **old: Any) -> dict[str, Any]:
    """The serialized ``cfg`` with ``removed`` keys dropped and the historical keys ``old`` reinserted."""
    payload = dict(serialize_config(cfg))
    for name in removed:
        payload.pop(name)
    payload.update(old)
    return payload


def _load(payload: dict[str, Any]) -> VFE3Config:
    return config_from_serialized(payload, source="test")


# ---------------------------------------------------------------- the fields are gone

def test_removed_fields_are_not_config_fields_and_are_registered_as_retired() -> None:
    names = {field.name for field in fields(VFE3Config)}
    for name in _REMOVED:
        assert name not in names, name
        assert name in RETIRED_CONFIG_KEYS, name
        with pytest.raises(TypeError, match=name):
            VFE3Config(**{name: None})
    assert "relational_receiver_measurement" in names and "lambda_alpha_mode" in names


def test_constant_alpha_form_is_unregistered_and_refused() -> None:
    assert set(_ALPHAS) == {"state_dependent", "state_dependent_per_coord"}
    with pytest.raises(ValueError, match="lambda_alpha_mode"):
        test_config(lambda_alpha_mode="constant")


# ---------------------------------------------------------------- receiver-law migration

def test_saved_ambient_alignment_is_refused_on_the_gradient_route() -> None:
    payload = _payload(test_config(), relational_receiver_law="ambient_alignment")
    with pytest.raises(ValueError, match="retired relational_receiver_law='ambient_alignment'"):
        _load(payload)


def test_gradient_config_predating_the_receiver_fields_ran_ambient_alignment_and_is_refused() -> None:
    # relational_receiver_law and relational_receiver_update were introduced together, so a payload
    # with neither predates both and ran ambient_alignment.
    payload = _payload(test_config(), ("relational_receiver_update",))
    with pytest.raises(ValueError, match="ambient_alignment"):
        _load(payload)


@pytest.mark.parametrize("update", ["euler", "closed_form"])
def test_saved_channel_likelihood_loads_and_a_current_payload_round_trips(update: str) -> None:
    cfg = test_config(relational_receiver_update=update)
    assert _load(_payload(cfg, relational_receiver_law="channel_likelihood")) == cfg
    # A payload written by this build carries the update but no law: it ran the channel likelihood.
    assert "relational_receiver_law" not in serialize_config(cfg)
    assert _load(_payload(cfg)) == cfg


@pytest.mark.parametrize("law", ["ambient_alignment", "channel_likelihood", None])
def test_receiver_law_was_inert_on_the_predictive_route(law: Any) -> None:
    cfg = test_config(**_PREDICTIVE)
    if law is None:                                   # predates the receiver fields: the Euler update loads
        payload = _payload(cfg, ("relational_receiver_update",))
        assert _load(payload) == test_config(**_PREDICTIVE, relational_receiver_update="euler")
    else:
        assert _load(_payload(cfg, relational_receiver_law=law)) == cfg


def test_invalid_saved_receiver_law_is_refused() -> None:
    with pytest.raises(ValueError, match="invalid relational_receiver_law"):
        _load(_payload(test_config(**_PREDICTIVE), relational_receiver_law="other"))


def test_lenient_loader_migrates_the_removed_keys_without_reporting_them_dropped() -> None:
    cfg = test_config()
    lenient, dropped = lenient_config_from_serialized(
        _payload(cfg, relational_receiver_law="channel_likelihood", lambda_alpha=1.0), source="test")
    assert lenient == cfg and dropped == {}


# ---------------------------------------------------------------- constant-alpha migration

@pytest.mark.parametrize("route", [{}, _PREDICTIVE])
def test_saved_constant_alpha_is_refused(route: dict) -> None:
    payload = _payload(test_config(**route), lambda_alpha_mode="constant", lambda_alpha=1.0)
    with pytest.raises(ValueError, match="retired lambda_alpha_mode='constant'"):
        _load(payload)


@pytest.mark.parametrize("mode", ["state_dependent", "state_dependent_per_coord"])
@pytest.mark.parametrize("value", [0.0, 1.0, 7.5])
def test_saved_lambda_alpha_is_dropped_under_a_state_dependent_form(mode: str, value: float) -> None:
    cfg = test_config(lambda_alpha_mode=mode)
    assert _load(_payload(cfg, lambda_alpha=value)) == cfg


# ---------------------------------------------------------------- receiver measurement default and pin

def test_measurement_default_is_inverse_with_a_historical_adjoint_pin() -> None:
    # The pin is fixed history; the VFE3Config default is the user's to choose and only has to be admitted.
    default = {field.name: field.default for field in fields(VFE3Config)}
    assert default["relational_receiver_measurement"] in ("inverse", "adjoint")
    assert _HISTORICAL_DEFAULT_PINS["relational_receiver_measurement"] == "adjoint"


@pytest.mark.parametrize("measurement", ["adjoint", "inverse"])
def test_saved_measurement_loads_unchanged_and_an_absent_one_loads_adjoint(measurement: str) -> None:
    cfg = test_config(relational_receiver_measurement=measurement)
    assert _load(_payload(cfg)) == cfg
    predates = _load(_payload(cfg, ("relational_receiver_measurement",)))
    assert predates.relational_receiver_measurement == "adjoint"
