"""Load-time migration of the configuration retirements of October 12, 2026 (audit rows C1, C4, C8, C10).

Each retirement rewrites a saved payload into the current schema. A payload built from
``serialize_config(test_config(...))`` with the old keys or values reinserted must load to the same
config as the current spelling, fingerprint identically after migration (so finished ablation cells
stay current), and keep a finished scaling cell harvestable.
"""
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import ablation
import scaling_analysis as sa
from tests.profiles import test_config, tiny_config
from tests.test_audit_20261008_stats import _write_scaling_run
from vfe3.artifact_utils import canonical_json_sha256
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.legacy_config import lenient_config_from_serialized
from vfe3.run_artifacts import semantic_config_fingerprint


def _old_payload(cfg: VFE3Config, removed: tuple[str, ...] = (), **old: Any) -> dict[str, Any]:
    """The serialized ``cfg`` with ``removed`` keys dropped and the historical keys ``old`` reinserted."""
    payload = serialize_config(cfg)
    for name in removed:
        payload.pop(name)
    payload.update(old)
    return payload


def _assert_migrates(payload: dict[str, Any], expected: VFE3Config, tmp_path: Path) -> None:
    """The old payload loads to ``expected`` and keeps every reuse fingerprint of the current form."""
    loaded = config_from_serialized(payload, source="test")
    assert loaded == expected
    lenient, dropped = lenient_config_from_serialized(payload, source="test")
    assert lenient == expected and dropped == {}
    assert semantic_config_fingerprint(asdict(loaded)) == semantic_config_fingerprint(asdict(expected))

    # An ablation cell whose stored contract fingerprinted the old schema is re-derived as current.
    run_dir = tmp_path / "ablation_cell"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"config": payload}), encoding="utf-8")
    expected_contract = {"semantic_config_fingerprint": semantic_config_fingerprint(asdict(expected))}
    stale = {"semantic_config_fingerprint": semantic_config_fingerprint(payload)}
    assert ablation._schema_migrated_contract(run_dir, stale, expected_contract) == expected_contract


def _assert_scaling_cell_survives(
    tmp_path:  Path,
    current:   dict[str, Any],
    removed:   tuple[str, ...] = (),
    **old:     Any,
) -> None:
    """A finished scaling cell whose config.json and digest predate the migration is still harvested.

    The cell is written for the current overrides ``current``; its saved config then carries the old
    keys instead, with the digest the producer computed from that saved form.
    """
    _write_scaling_run(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0, **current)
    expected = sa.harvest(tmp_path)[0]["effective_run_key"]
    run = tmp_path / "width" / "K8" / "s1"
    record = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for name in removed:
        record["config"].pop(name)
    record["config"].update(old)
    (run / "config.json").write_text(json.dumps(record), encoding="utf-8")
    cell = json.loads((run / "scaling_cell.json").read_text(encoding="utf-8"))
    cell.pop("reuse_contract_sha256")
    cell["config_sha256"] = canonical_json_sha256(record["config"])
    digest = canonical_json_sha256(cell)
    cell["reuse_contract_sha256"] = digest
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["scaling_reuse_contract_sha256"] = digest
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = sa.harvest(tmp_path)
    assert len(rows) == 1 and rows[0]["effective_run_key"] == expected


# =============================================================================
# C1: the generic family decode and the Laplace family
# =============================================================================

@pytest.mark.parametrize(("family", "old_mode", "new_mode"), [
    ("gaussian_diagonal", "family_chunked", "diagonal_chunked"),
    ("gaussian_full", "family_chunked", "full_chunked"),
    ("gaussian_diagonal", "family", "diagonal_chunked"),
])
def test_family_decode_migrates_to_the_kernel_of_the_family_rank(
    family: str, old_mode: str, new_mode: str, tmp_path: Path,
) -> None:
    expected = test_config(family=family, decode_mode=new_mode)
    _assert_migrates(_old_payload(expected, decode_mode=old_mode), expected, tmp_path)


def test_family_decode_scaling_cell_survives(tmp_path: Path) -> None:
    _assert_scaling_cell_survives(tmp_path, {"decode_mode": "diagonal_chunked"}, decode_mode="family_chunked")


def test_laplace_family_is_refused_as_retired() -> None:
    payload = _old_payload(test_config(), family="laplace_diagonal", decode_mode="family_chunked")
    with pytest.raises(ValueError, match="retired family='laplace_diagonal'"):
        config_from_serialized(payload, source="test")


def test_family_decode_with_a_block_family_is_refused() -> None:
    payload = _old_payload(test_config(), family="gaussian_block", decode_mode="family_chunked")
    with pytest.raises(ValueError, match="retired decode_mode='family_chunked'"):
        config_from_serialized(payload, source="test")


# =============================================================================
# C4: the five projected-factorization fields merge into projected_factorization
# =============================================================================

_FACTORIZATION_FIELDS = ("projected_factorization",)


@pytest.mark.parametrize(("recovery", "route", "precision", "factorization"), [
    (False, "gram", "fp64", "gram"),
    (False, "gram", "fp32", "gram"),
    (True, "gram", "fp64", "gram_certified"),
    (True, "gram", "fp32", "gram_certified"),
    (True, "cholesky_qr2", "fp32", "cholesky_qr2_certified"),
    (False, "cholesky_qr2", "fp64", "cholesky_qr2_certified"),
])
def test_projected_factor_tuples_migrate(
    recovery: bool, route: str, precision: str, factorization: str, tmp_path: Path,
) -> None:
    expected = test_config(projected_factorization=factorization)
    payload = _old_payload(
        expected, _FACTORIZATION_FIELDS, projected_cov_condition_recovery=recovery,
        projected_factor_route=route, projected_cov_recovery_precision=precision,
        projected_gram_condition_tol=1e-4, projected_factor_condition_tol=3e-4)
    _assert_migrates(payload, expected, tmp_path)


def test_a_payload_predating_the_recovery_precision_migrates(tmp_path: Path) -> None:
    expected = test_config()
    payload = _old_payload(expected, _FACTORIZATION_FIELDS, projected_cov_condition_recovery=False,
                           projected_factor_route="gram")
    _assert_migrates(payload, expected, tmp_path)


def test_projected_factor_scaling_cell_survives(tmp_path: Path) -> None:
    _assert_scaling_cell_survives(
        tmp_path, {}, _FACTORIZATION_FIELDS, projected_cov_condition_recovery=False,
        projected_factor_route="gram", projected_cov_recovery_precision="fp32",
        projected_gram_condition_tol=1e-4, projected_factor_condition_tol=3e-4)


def test_inert_tolerances_load_and_live_off_default_tolerances_are_refused() -> None:
    base = _old_payload(test_config(), _FACTORIZATION_FIELDS, projected_factor_route="gram")
    assert config_from_serialized(
        {**base, "projected_cov_condition_recovery": False, "projected_gram_condition_tol": 2e-4},
        source="test") == test_config()
    with pytest.raises(ValueError, match="retired projected_gram_condition_tol=0.0002"):
        config_from_serialized(
            {**base, "projected_cov_condition_recovery": True, "projected_gram_condition_tol": 2e-4},
            source="test")


def test_old_and_new_factorization_selectors_conflict() -> None:
    payload = {**serialize_config(test_config()), "projected_factor_route": "gram"}
    with pytest.raises(ValueError, match="conflicting projected-factorization"):
        config_from_serialized(payload, source="test")


# =============================================================================
# C8: pos_rotation, rope_space, rope_on_value and rope_comparison_covariance merge into rope_mode
# =============================================================================

_ROPE_FIELDS = ("rope_mode",)


def _rope_config(mode: str) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # comparison_invariant at head size 2: P(G) diagonal
        return tiny_config(rope_mode=mode)


# Every combination the old validator admitted, each mapped to exactly one mode.
@pytest.mark.parametrize(("rotation", "space", "on_value", "covariance", "mode"), [
    ("none", "frame", True, "full", "none"),
    ("none", "comparison", False, "rotation_invariant", "none"),
    ("rope", "frame", False, "full", "frame"),
    ("rope", "frame", False, "rotation_invariant", "frame"),
    ("rope", "frame", True, "full", "frame_value"),
    ("rope", "frame", True, "rotation_invariant", "frame_value"),
    ("rope", "comparison", False, "full", "comparison"),
    ("rope", "comparison", False, "rotation_invariant", "comparison_invariant"),
])
def test_rope_combinations_migrate(
    rotation: str, space: str, on_value: bool, covariance: str, mode: str, tmp_path: Path,
) -> None:
    expected = _rope_config(mode)
    payload = _old_payload(expected, _ROPE_FIELDS, pos_rotation=rotation, rope_space=space,
                           rope_on_value=on_value, rope_comparison_covariance=covariance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_migrates(payload, expected, tmp_path)


def test_a_rope_record_predating_rope_space_migrates_as_frame_space(tmp_path: Path) -> None:
    expected = _rope_config("frame")
    payload = _old_payload(expected, _ROPE_FIELDS, pos_rotation="rope", rope_on_value=False)
    _assert_migrates(payload, expected, tmp_path)


def test_rope_scaling_cell_survives(tmp_path: Path) -> None:
    _assert_scaling_cell_survives(tmp_path, {}, _ROPE_FIELDS, pos_rotation="none", rope_space="frame",
                                  rope_on_value=True, rope_comparison_covariance="full")


def test_comparison_rope_on_the_value_frame_stays_refused() -> None:
    payload = _old_payload(_rope_config("none"), _ROPE_FIELDS, pos_rotation="rope", rope_space="comparison",
                           rope_on_value=True, rope_comparison_covariance="full")
    with pytest.raises(ValueError, match="retired rope_on_value=True"):
        config_from_serialized(payload, source="test")


def test_old_and_new_rope_selectors_conflict() -> None:
    payload = {**serialize_config(test_config()), "pos_rotation": "none"}
    with pytest.raises(ValueError, match="conflicting RoPE selectors"):
        config_from_serialized(payload, source="test")


# =============================================================================
# C10: relational_output_init and bch_residual_max hardcoded; one frame-pair escalation tolerance
# =============================================================================

_C10_FIELDS = ("frame_pair_escalation_tol",)
_C10_DEFAULTS = dict(relational_output_init="identity", bch_residual_max=None,
                     frame_pair_fp64_escalation=False, frame_pair_condition_tol=1e-4)


@pytest.mark.parametrize(("old", "tol"), [
    ({}, None),
    ({"bch_residual_max": 1e-6}, None),
    ({"bch_residual_max": 1e-3}, None),                     # min(1e-6, 1e-3) is the fixed 1e-6 gate
    ({"frame_pair_condition_tol": 3e-4}, None),             # inert while the escalation was off
    ({"frame_pair_fp64_escalation": True}, 1e-4),
    ({"frame_pair_fp64_escalation": True, "frame_pair_condition_tol": 3e-4}, 3e-4),
])
def test_c10_fields_migrate(old: dict[str, Any], tol: Any, tmp_path: Path) -> None:
    expected = test_config(frame_pair_escalation_tol=tol)
    payload = _old_payload(expected, _C10_FIELDS, **{**_C10_DEFAULTS, **old})
    _assert_migrates(payload, expected, tmp_path)


def test_c10_scaling_cell_survives(tmp_path: Path) -> None:
    _assert_scaling_cell_survives(tmp_path, {}, _C10_FIELDS, **_C10_DEFAULTS)


def test_non_default_hardcoded_values_are_refused_where_they_were_read() -> None:
    base = _old_payload(test_config(), _C10_FIELDS, **_C10_DEFAULTS)
    with pytest.raises(ValueError, match="retired relational_output_init='depth_scaled'"):
        config_from_serialized({**base, "relational_output_init": "depth_scaled"}, source="test")
    manifold = _old_payload(test_config(m_phi_update_mode="pullback_group"), _C10_FIELDS, **_C10_DEFAULTS)
    assert config_from_serialized({**manifold, "bch_residual_max": 1e-6}, source="test") == test_config(
        m_phi_update_mode="pullback_group")
    with pytest.raises(ValueError, match="retired bch_residual_max=1e-08"):
        config_from_serialized({**manifold, "bch_residual_max": 1e-8}, source="test")


def test_non_default_hardcoded_values_load_where_they_were_inert() -> None:
    assert config_from_serialized({**_old_payload(test_config(), _C10_FIELDS, **_C10_DEFAULTS),
                                   "bch_residual_max": 1e-8}, source="test") == test_config()   # adamw phi step
    agent = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                 attention_residual=False, predictive_covariance="passthrough")
    expected = test_config(**agent)
    payload = {**_old_payload(expected, _C10_FIELDS, **_C10_DEFAULTS), "relational_output_init": "depth_scaled"}
    assert config_from_serialized(payload, source="test") == expected


def test_old_and_new_frame_pair_selectors_conflict() -> None:
    payload = {**serialize_config(test_config()), "frame_pair_fp64_escalation": False}
    with pytest.raises(ValueError, match="conflicting frame-pair escalation"):
        config_from_serialized(payload, source="test")
