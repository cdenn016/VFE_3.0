"""Test profiles own their settings even when experiment defaults change."""
from dataclasses import asdict, fields
import inspect
from collections.abc import Callable

import pytest

from tests.profiles import live_config, test_config, tiny_config
from vfe3.config import VFE3Config


@pytest.mark.parametrize("factory", [test_config, tiny_config, live_config])
@pytest.mark.parametrize("changes", [
    dict(family="gaussian_full", decode_mode="full_chunked", gauge_transport="off",
         predictive_covariance="passthrough", sigma_init=0.4, pos_phi="frozen"),
    dict(relational_receiver_update="closed_form", skip_belief_sigma_update=False, lambda_alpha_mode="constant",
         m_decode_mu_lr=None, m_decode_sigma_lr=None, m_pos_phi_lr=None,
         projected_factorization="gram_certified", phi_weight_decay=0.0),
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

    assert {field.name for field in fields(VFE3Config)} == set(config_values())


def test_profile_overrides_do_not_leak_between_configs() -> None:
    from tests.profiles import config_values

    kappa = [1.0, 2.0]
    first = config_values(kappa_beta=kappa)
    first["kappa_beta"].append(3.0)
    first["family"] = "gaussian_full"
    assert kappa == [1.0, 2.0]
    assert config_values()["kappa_beta"] == 1.0
    assert config_values()["family"] == "gaussian_diagonal"
