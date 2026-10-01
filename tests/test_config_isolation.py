"""Test profiles own their settings even when experiment defaults change."""
from dataclasses import asdict, fields
import inspect
from collections.abc import Callable

import pytest

from tests.profiles import test_config, tiny_config
from vfe3.config import VFE3Config


@pytest.mark.parametrize("factory", [test_config, tiny_config])
@pytest.mark.parametrize("changes", [
    dict(family="gaussian_full", decode_mode="full_chunked", gauge_transport="off",
         predictive_covariance="passthrough", sigma_init=0.4, pos_phi="frozen"),
    dict(e_step_update="mm_exact", e_step_gradient="detach", lambda_alpha_mode="constant",
         m_decode_mu_lr=None, m_decode_sigma_lr=None, m_pos_phi_lr=None,
         frame_pair_fp64_escalation=True, phi_weight_decay=0.0),
])
def test_profile_ignores_changed_experiment_defaults(
    factory:     Callable[..., VFE3Config],
    changes:     dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = asdict(factory())
    constructor = VFE3Config.__init__
    names = list(inspect.signature(constructor).parameters)[1:]
    defaults = list(constructor.__defaults__)
    for name, value in changes.items():
        defaults[names.index(name)] = value
        monkeypatch.setattr(VFE3Config, name, value)
        monkeypatch.setattr(VFE3Config.__dataclass_fields__[name], "default", value)
    monkeypatch.setattr(constructor, "__defaults__", tuple(defaults))
    assert asdict(factory()) == before


def test_profile_declares_every_config_field() -> None:
    from tests.profiles import config_values

    assert {field.name for field in fields(VFE3Config)} == (
        set(config_values()) - {"_allow_legacy_dense_decode"}
    )


def test_profile_overrides_do_not_leak_between_configs() -> None:
    from tests.profiles import config_values

    couplings = [(0, 1)]
    first = config_values(cross_couplings=couplings)
    first["cross_couplings"].append((1, 0))
    first["family"] = "gaussian_full"
    assert couplings == [(0, 1)]
    assert config_values()["cross_couplings"] is None
    assert config_values()["family"] == "gaussian_diagonal"
