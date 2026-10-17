r"""The ``fully_gauge_covariant`` switch on routes without reference coordinates (audit 2026-10-16, F31 and F33).

F31. ``block_coordinates='reference'`` exists only on ``projected_predictive``, and ``decode_coordinates='reference'``
only for the linear and KL prior-bank decoders. The switch used to emit reference block coordinates whatever the
route and block, and to rewrite only ``diagonal_chunked``, so it produced configurations the validator refuses,
including ones with a covariant reading (no block MLP under an equivariant norm is covariant in local
coordinates). It now leaves the block coordinates alone when nothing frame-dependent runs in the block, and raises
a targeted error where no covariant reading keeps the chosen block or decoder.

F33. The launcher's override log printed ``config.get(key)``, so a key the launcher omits showed as
``None -> None``; it now prints the value the switch compared, the launcher literal or the ``VFE3Config`` default.

Config level only (no model is built).
"""
from __future__ import annotations

import warnings

import pytest

from tests.profiles import config_values
from vfe3.config import VFE3Config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _predictive(**overrides) -> dict:
    return config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                         predictive_covariance="moment_matched", **overrides)


def _gradient(**overrides) -> dict:
    return config_values(attention_mode="projected_gradient", predictive_covariance="passthrough", **overrides)


@pytest.mark.parametrize("case", [
    _gradient(use_block_mlp=False, norm_type_block="none"),
    _gradient(use_block_mlp=False, norm_type_block="mahalanobis"),
    _gradient(use_block_mlp=True, block_mlp_mode="gauge_gate", norm_type_block="none"),
    _predictive(use_block_mlp=False, norm_type_block="mahalanobis"),
    _predictive(use_block_mlp=False, norm_type_block="none"),
], ids=["gradient_no_block", "gradient_mahalanobis", "gradient_gauge_gate", "predictive_mahalanobis",
        "predictive_no_block"])
def test_a_block_with_nothing_frame_dependent_keeps_its_coordinates(case: dict) -> None:
    before = VFE3Config(**case)
    overrides = fully_gauge_covariant_overrides(case)
    after = VFE3Config(**{**case, **overrides})              # the validator accepts the switched configuration
    assert gauge_covariance_failures(before)
    assert gauge_covariance_failures(after) == []
    if case["use_block_mlp"]:
        assert overrides.get("block_coordinates", case["block_coordinates"]) == "local"
    else:
        assert "block_coordinates" not in overrides and "use_cg_coupling" not in overrides
        assert after.block_coordinates == before.block_coordinates


@pytest.mark.parametrize("case", [
    _gradient(use_block_mlp=True, block_mlp_mode="coordinate", norm_type_block="none"),
    _gradient(use_block_mlp=False, norm_type_block="layernorm"),
])
def test_a_block_needing_reference_coordinates_is_refused_off_projected_predictive(case: dict) -> None:
    with pytest.raises(ValueError, match="attention_mode='projected_gradient'"):
        fully_gauge_covariant_overrides(case)


@pytest.mark.parametrize("case", [
    _predictive(decode_mode="expected_likelihood_chunked"),
    _predictive(decode_mode="expected_likelihood_chunked", use_block_mlp=False, norm_type_block="none"),
])
def test_a_decoder_without_a_reference_reading_is_refused(case: dict) -> None:
    VFE3Config(**case)                                        # valid before the switch
    with pytest.raises(ValueError, match="decode_mode='expected_likelihood_chunked'"):
        fully_gauge_covariant_overrides(case)


def test_the_override_log_shows_the_resolved_value_of_an_omitted_key() -> None:
    from vfe3.gauge_covariant import describe_fully_gauge_covariant_overrides

    config = _predictive()
    del config["sigma_max"]                                   # the launcher omits it; the default is 100.0
    assert VFE3Config(**config).sigma_max == 100.0
    overrides = fully_gauge_covariant_overrides(config)
    assert overrides["sigma_max"] is None
    line = describe_fully_gauge_covariant_overrides(config, overrides)
    assert "sigma_max=100.0 -> None" in line
    assert "None -> None" not in line
    assert line.count(" -> ") == len(overrides)
    assert describe_fully_gauge_covariant_overrides(config, {}) == ""
