"""Loader migration of the selectors retired on October 2, 2026.

Each retired key is popped by ``config_from_serialized``: a saved value equal to the one the
projected routes pinned (or any value where the field was inert) loads silently, and any other value
is refused as retired. The lenient loader hands every retired key to that rule instead of dropping it.
"""
from typing import Any

import pytest

from tests.profiles import test_config
from vfe3.config import RETIRED_CONFIG_KEYS, VFE3Config, config_from_serialized, serialize_config
from vfe3.legacy_config import MIGRATED_HISTORICAL_KEYS, lenient_config_from_serialized


def _payload(**updates: Any) -> dict[str, Any]:
    payload = serialize_config(test_config())
    payload.update(updates)
    return payload


def _load(**updates: Any) -> VFE3Config:
    return config_from_serialized(_payload(**updates), source="test")


# (key, accepted value, refused value). The accepted value is the pinned one.
PINNED = [
    ("divergence_family", "renyi", "jeffreys"),
    ("renyi_order", 1.0, 0.5),
    ("include_attention_entropy", True, False),
    ("gradient_mode", "filtering", "smoothing"),
    ("e_step_update", "gradient", "mm_exact"),
    ("e_step_gradient", "unroll", "straight_through"),
    ("oracle_unroll_grad", True, False),
    ("e_phi_lr", 0.0, 0.1),
    ("transport_mode", "flat", "regime_ii"),
]


@pytest.mark.parametrize(("key", "accepted", "refused"), PINNED)
def test_pinned_retired_key_loads_at_its_pinned_value_and_refuses_otherwise(
    key: str, accepted: object, refused: object,
) -> None:
    assert _load(**{key: accepted}) == test_config()
    with pytest.raises(ValueError, match=rf"retired {key}="):
        _load(**{key: refused})


def test_pinned_numeric_key_rejects_a_bool_stand_in() -> None:
    with pytest.raises(ValueError, match="retired renyi_order="):
        _load(renyi_order=True)
    assert _load(renyi_order=1) == test_config()


@pytest.mark.parametrize(("key", "value"), [
    ("mm_damping", 0.5),
    ("full_cov_kl_backend", "common_frame_factored"),
    ("full_cov_congruence_precision", "fp64"),
    ("full_cov_congruence_certification", True),
    ("congruence_cond_escalation", True),
    ("rope_full_gauge", True),
    ("compile_pair_kernel", True),
    ("_allow_legacy_dense_decode", True),
])
def test_inert_retired_key_loads_with_any_value(key: str, value: object) -> None:
    assert _load(**{key: value}) == test_config()


@pytest.mark.parametrize("value", [None, []])
def test_cross_couplings_accepts_none_or_empty(value: object) -> None:
    assert _load(cross_couplings=value) == test_config()


def test_cross_couplings_refuses_a_coupling() -> None:
    with pytest.raises(ValueError, match="retired cross_couplings="):
        _load(cross_couplings=[[0, 1]])


def test_mu_trust_mode_box_is_refused_only_where_the_trust_region_was_read() -> None:
    # test_config is projected_gradient + euler + e_mu_q_trust=2: the box region was live.
    with pytest.raises(ValueError, match="retired mu_trust_mode='box'"):
        _load(mu_trust_mode="box")
    assert _load(mu_trust_mode="ball") == test_config()
    assert _load(mu_trust_mode="box", e_mu_q_trust=None) == test_config(e_mu_q_trust=None)
    closed = _load(mu_trust_mode="box", relational_receiver_update="closed_form")
    assert closed == test_config(relational_receiver_update="closed_form")
    predictive = dict(attention_mode="projected_predictive", relational_receiver_law="channel_likelihood",
                      relational_receiver_update="closed_form")
    assert _load(mu_trust_mode="box", **predictive) == test_config(**predictive)


def test_e_step_mu_precond_raw_is_inert_on_predictive_and_refused_on_gradient() -> None:
    with pytest.raises(ValueError, match="retired e_step_mu_precond='raw'"):
        _load(e_step_mu_precond="raw")
    assert _load(e_step_mu_precond="fisher") == test_config()
    predictive = dict(attention_mode="projected_predictive", relational_receiver_law="channel_likelihood",
                      relational_receiver_update="closed_form")
    assert _load(e_step_mu_precond="raw", **predictive) == test_config(**predictive)


def test_m_decode_lr_migrates_into_unset_split_rates() -> None:
    cfg = _load(m_decode_lr=0.3)                                   # linear: mean-role rate only
    assert (cfg.m_decode_mu_lr, cfg.m_decode_sigma_lr) == (0.3, None)
    bank = dict(decode_mode="diagonal_chunked", decode_temperature_learned=False)
    cfg = _load(m_decode_lr=0.3, **bank)
    assert (cfg.m_decode_mu_lr, cfg.m_decode_sigma_lr) == (0.3, 0.3)
    cfg = _load(m_decode_lr=0.3, m_decode_mu_lr=0.01, **bank)
    assert (cfg.m_decode_mu_lr, cfg.m_decode_sigma_lr) == (0.01, 0.3)
    assert _load(m_decode_lr=None) == test_config()


def test_m_decode_lr_is_refused_where_it_set_a_learned_temperature_rate() -> None:
    with pytest.raises(ValueError, match="retired m_decode_lr="):
        _load(m_decode_lr=0.3, decode_mode="diagonal_chunked", decode_temperature_learned=True)
    cfg = _load(m_decode_lr=0.3, decode_mode="diagonal_chunked", decode_temperature_learned=False)
    assert cfg.m_decode_mu_lr == 0.3


@pytest.mark.parametrize(("dense", "chunked"), [
    ("diagonal", "diagonal_chunked"), ("full", "full_chunked"), ("family", "family_chunked"),
])
def test_dense_decode_modes_migrate_to_their_chunked_kernels(dense: str, chunked: str) -> None:
    family = "gaussian_full" if dense == "full" else "gaussian_diagonal"
    cfg = _load(decode_mode=dense, family=family, _allow_legacy_dense_decode=True)
    assert cfg.decode_mode == chunked
    assert "_allow_legacy_dense_decode" not in serialize_config(cfg)


def test_gaussian_diagonal_exact_migrates_to_gaussian_diagonal() -> None:
    assert _load(family="gaussian_diagonal_exact").family == "gaussian_diagonal"


def test_gaussian_frame_diagonal_is_refused_as_retired() -> None:
    with pytest.raises(ValueError, match="retired family='gaussian_frame_diagonal'"):
        _load(family="gaussian_frame_diagonal")


@pytest.mark.parametrize("mode", ["canonical_content_gauge", "canonical_content_projected"])
def test_canonical_content_encoders_are_refused_as_retired(mode: str) -> None:
    with pytest.raises(ValueError, match=f"retired encode_mode='{mode}'"):
        _load(encode_mode=mode)


@pytest.mark.parametrize("group", ["glk", "tied_block_glk", "so_k", "sp", "sp_n"])
def test_retired_gauge_groups_are_refused_on_load_and_construction(group: str) -> None:
    with pytest.raises(ValueError, match="retired"):
        _load(gauge_group=group)
    with pytest.raises(ValueError, match="retired"):
        test_config(gauge_group=group)


def test_retired_fields_are_not_constructor_arguments() -> None:
    for key in ("e_step_update", "mu_trust_mode", "renyi_order", "transport_mode", "m_decode_lr",
                "_allow_legacy_dense_decode"):
        with pytest.raises(TypeError):
            test_config(**{key: None})


def test_lenient_loader_routes_every_retired_key_through_the_strict_rule() -> None:
    assert MIGRATED_HISTORICAL_KEYS == RETIRED_CONFIG_KEYS
    assert "compile_pair_kernel" in MIGRATED_HISTORICAL_KEYS
    cfg, dropped = lenient_config_from_serialized(
        _payload(e_step_update="gradient", compile_pair_kernel=True, mm_damping=0.5, unknown_knob=1),
        source="test")
    assert cfg == test_config()
    assert dropped == {"unknown_knob": 1}
    with pytest.raises(ValueError, match="retired e_step_update='mm_exact'"):
        lenient_config_from_serialized(_payload(e_step_update="mm_exact"), source="test")
