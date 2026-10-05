"""Serialized configs written before the 2026-10-05 removal of both head mixers."""
from tests.profiles import test_config

import pytest

from vfe3.config import config_from_serialized, serialize_config


def _payload(**extra: object) -> dict:
    return {**serialize_config(test_config()), **extra}


def test_disabled_mixers_and_their_rates_load_and_are_dropped() -> None:
    payload = _payload(use_head_mixer=False, use_priorbank_head_evidence_mixer=False,
                       m_head_mixer_lr=0.025, m_head_evidence_lr=None)
    cfg = config_from_serialized(payload, source="test")
    for name in ("use_head_mixer", "use_priorbank_head_evidence_mixer",
                 "m_head_mixer_lr", "m_head_evidence_lr"):
        assert not hasattr(cfg, name)


@pytest.mark.parametrize("name", ["use_head_mixer", "use_priorbank_head_evidence_mixer"])
def test_enabled_mixer_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=f"retired {name}=True"):
        config_from_serialized(_payload(**{name: True}), source="test")


@pytest.mark.parametrize("name", ["use_head_mixer", "use_priorbank_head_evidence_mixer",
                                  "m_head_mixer_lr", "m_head_evidence_lr"])
def test_live_config_rejects_the_removed_fields(name: str) -> None:
    with pytest.raises(TypeError):
        test_config(**{name: False if name.startswith("use_") else 0.01})
