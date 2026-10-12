"""Removals of the round-2 audit (October 12, 2026) and their load-time rules.

The ``t5_relative_bias`` attention prior and its three fields (``t5_num_buckets``, ``t5_max_distance``,
``t5_learnable_bias``), the ``mass_phi`` frame-norm penalty and ``mu_trust_cholesky_rounds`` were removed.
A saved config naming the T5 prior is refused and its fields are dropped otherwise; ``mass_phi`` loads only
as 0; any saved round count is dropped, since the ridge ladder is the constant
``numerics.MU_TRUST_CHOLESKY_ROUNDS``. The family-consistency capability flags left by the retired generic
family decode (``DecodeRegistration.family_consistent``, ``BeliefParams.gaussian_pointwise_algebra``) are gone.
"""
from dataclasses import fields
from typing import Any

import pytest
import torch

from tests.profiles import test_config
from vfe3.config import RETIRED_CONFIG_KEYS, VFE3Config, config_from_serialized, serialize_config
from vfe3.families.base import BeliefParams
from vfe3.legacy_config import lenient_config_from_serialized
from vfe3.model.prior_bank import DecodeRegistration
from vfe3.numerics import MU_TRUST_CHOLESKY_ROUNDS, safe_cholesky

_REMOVED = ("t5_num_buckets", "t5_max_distance", "t5_learnable_bias", "mass_phi", "mu_trust_cholesky_rounds")


def _loads_to(payload: dict[str, Any], expected: VFE3Config) -> None:
    assert config_from_serialized(payload, source="test") == expected
    lenient, dropped = lenient_config_from_serialized(payload, source="test")
    assert lenient == expected and dropped == {}


def test_fields_are_gone_and_registered_as_retired_keys() -> None:
    names = {field.name for field in fields(VFE3Config)}
    for name in _REMOVED:
        assert name not in names, name
        assert name in RETIRED_CONFIG_KEYS, name
        with pytest.raises(TypeError):
            test_config(**{name: 0})


def test_t5_fields_drop_when_the_prior_is_not_t5() -> None:
    cfg = test_config()
    payload = {**serialize_config(cfg), "t5_num_buckets": 32, "t5_max_distance": 128, "t5_learnable_bias": True}
    _loads_to(payload, cfg)


def test_t5_prior_is_refused() -> None:
    payload = {**serialize_config(test_config()), "beta_attention_prior": "t5_relative_bias",
               "t5_num_buckets": 32, "t5_max_distance": 128, "t5_learnable_bias": False}
    with pytest.raises(ValueError, match="retired beta_attention_prior='t5_relative_bias'"):
        config_from_serialized(payload, source="test")


def test_mass_phi_zero_drops_and_positive_is_refused() -> None:
    cfg = test_config()
    for value in (0, 0.0):
        _loads_to({**serialize_config(cfg), "mass_phi": value}, cfg)
    for value in (0.1, True):
        with pytest.raises(ValueError, match="retired mass_phi"):
            config_from_serialized({**serialize_config(cfg), "mass_phi": value}, source="test")


def test_mu_trust_cholesky_rounds_drops_any_saved_value() -> None:
    cfg = test_config()
    for value in (0, 3, 7):
        _loads_to({**serialize_config(cfg), "mu_trust_cholesky_rounds": value}, cfg)


def test_mu_trust_rounds_constant_rescues_a_rounding_indefinite_covariance() -> None:
    # lambda_min = -100 u lambda_max (u the fp32 unit roundoff) needs three ridge rounds (1e-6, 1e-5, 1e-4).
    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    eigs = 15.0 * torch.logspace(0, -3, 4, dtype=torch.float64)
    eigs[-1] = -100 * 6e-8 * 15.0
    sigma = ((Q * eigs) @ Q.T).to(torch.float32)
    assert MU_TRUST_CHOLESKY_ROUNDS == 4
    assert not bool(safe_cholesky(sigma, eps=1e-6, rounds=2)[1])
    assert bool(safe_cholesky(sigma, eps=1e-6, rounds=MU_TRUST_CHOLESKY_ROUNDS)[1])


def test_family_consistency_flags_are_gone() -> None:
    assert "family_consistent" not in {field.name for field in fields(DecodeRegistration)}
    assert not hasattr(BeliefParams, "gaussian_pointwise_algebra")
