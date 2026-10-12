"""Removal of four configuration fields on October 12, 2026 (second pass) and their load-time rules.

``expected_likelihood_kernel`` (CUDA tensors run the fused Triton kernel, CPU tensors the chunked torch
kernel), ``full_cov_kl_precision`` (hardcoded to the former ``fp32_escalate_cond``), ``irrep_basis``
(canonical only) and ``decode_bias`` (the linear decoder always carries its bias). The two numerical
policies computed one quantity to rounding, so any saved value is dropped. A saved so_n config in the
eigh basis, or without the field (saved before 2026-10-05), is refused, as is a linear decoder saved
without its bias.
"""
from dataclasses import fields
from typing import Any

import pytest
import torch

from tests.profiles import test_config
from vfe3.config import RETIRED_CONFIG_KEYS, VFE3Config, config_from_serialized, serialize_config
from vfe3.families.gaussian import FullGaussian
from vfe3.legacy_config import lenient_config_from_serialized
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank

_REMOVED = ("expected_likelihood_kernel", "full_cov_kl_precision", "irrep_basis", "decode_bias")
_SO_N = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l2", 2)], embed_dim=10, n_heads=2)


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
            test_config(**{name: None})


def test_numerical_policies_drop_any_saved_value() -> None:
    cfg = test_config()
    for name, values in (("expected_likelihood_kernel", ("torch", "triton")),
                         ("full_cov_kl_precision", ("fp64", "fp32_escalate", "fp32_escalate_cond"))):
        for value in values:
            _loads_to({**serialize_config(cfg), name: value}, cfg)


def test_irrep_basis_drops_for_canonical_so_n_and_any_other_group() -> None:
    glk = test_config()
    for value in ("canonical", "eigh"):
        _loads_to({**serialize_config(glk), "irrep_basis": value}, glk)
    so_n = test_config(**_SO_N)
    payload = serialize_config(so_n)
    assert payload["irrep_basis"] == "canonical"                       # the basis marker of an so_n save
    _loads_to(payload, so_n)


@pytest.mark.parametrize("saved", ["eigh", None])
def test_so_n_eigh_or_unmarked_payload_is_refused(saved: Any) -> None:
    payload = serialize_config(test_config(**_SO_N))
    if saved is None:
        payload.pop("irrep_basis")
    else:
        payload["irrep_basis"] = saved
    with pytest.raises(ValueError, match="eigh irrep basis"):
        config_from_serialized(payload, source="test")


def test_decode_bias_drops_unless_a_linear_decoder_saved_it_off() -> None:
    linear = test_config(decode_mode="linear")
    _loads_to({**serialize_config(linear), "decode_bias": True}, linear)
    prior_bank = test_config(decode_mode="diagonal_chunked")
    for value in (True, False):
        _loads_to({**serialize_config(prior_bank), "decode_bias": value}, prior_bank)
    with pytest.raises(ValueError, match="retired decode_bias=False"):
        config_from_serialized({**serialize_config(linear), "decode_bias": False}, source="test")


def test_prior_bank_no_longer_accepts_the_removed_arguments() -> None:
    for name, value in (("expected_likelihood_kernel", "torch"), ("decode_bias", True),
                        ("full_cov_precision_policy", "fp64")):
        with pytest.raises(TypeError):
            PriorBank(7, 3, 3, **{name: value})


def test_full_gaussian_has_no_precision_policy() -> None:
    with pytest.raises(TypeError):
        FullGaussian(torch.zeros(2), torch.eye(2), _precision_policy="fp64")


def test_linear_model_always_builds_the_zero_bias() -> None:
    model = VFEModel(test_config(decode_mode="linear", vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=4))
    bias = model.prior_bank.output_proj_bias
    assert bias is not None and torch.equal(bias, torch.zeros(9))
