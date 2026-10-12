"""Load-time migration of the configuration retirements of October 12, 2026 (audit rows C1, C4, C8, C10).

Each retirement rewrites a saved payload into the current schema. A payload built from
``serialize_config(test_config(...))`` with the old keys or values reinserted must load to the same
config as the current spelling, fingerprint identically after migration (so finished ablation cells
stay current), and keep a finished scaling cell harvestable.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import ablation
import scaling_analysis as sa
from tests.profiles import test_config
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
