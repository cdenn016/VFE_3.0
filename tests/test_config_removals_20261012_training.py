"""Training and runtime policies removed or hardcoded on October 12, 2026 (second pass).

``divergence_abort`` is gone (the divergence monitor only warns and forces metrics rows),
``fill_uninitialized_memory`` is fixed at False, ``train_window_offset`` at per-epoch random origins and
``matmul_precision_retry`` at the replay ladder. A saved config or baseline config carrying any of these keys,
at either value, loads to the current config.
"""
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest

import paper_arms
from tests.profiles import test_config
from vfe3.baseline import BaselineConfig
from vfe3.config import RETIRED_CONFIG_KEYS, VFE3Config, config_from_serialized, serialize_config
from vfe3.legacy_config import lenient_config_from_serialized
from vfe3.paper import tables as pt
from vfe3.paper.runs import RunRecord
from vfe3.run_artifacts import semantic_config_fingerprint

REMOVED = ("divergence_abort", "fill_uninitialized_memory", "train_window_offset", "matmul_precision_retry")
OLD_VALUES = (
    ("divergence_abort", False), ("divergence_abort", True),
    ("fill_uninitialized_memory", True), ("fill_uninitialized_memory", False),
    ("train_window_offset", "fixed"), ("train_window_offset", "per_epoch_random"),
    ("matmul_precision_retry", False), ("matmul_precision_retry", True),
)


def test_the_fields_are_gone() -> None:
    names = {f.name for f in fields(VFE3Config)}
    baseline = {f.name for f in fields(BaselineConfig)}
    for name in REMOVED:
        assert name not in names and name in RETIRED_CONFIG_KEYS
        with pytest.raises(TypeError):
            test_config(**{name: None})
    assert not {"train_window_offset", "fill_uninitialized_memory"} & baseline
    assert "train_window_offset" not in paper_arms.BASELINE_PROTOCOL_FIELDS


@pytest.mark.parametrize("name, value", OLD_VALUES)
def test_a_saved_value_is_dropped_on_load(name: str, value: Any) -> None:
    expected = test_config()
    payload = {**serialize_config(expected), name: value}
    loaded = config_from_serialized(payload, source="test")
    assert loaded == expected
    lenient, dropped = lenient_config_from_serialized(payload, source="test")
    assert lenient == expected and dropped == {}
    assert semantic_config_fingerprint(asdict(loaded)) == semantic_config_fingerprint(asdict(expected))


@pytest.mark.parametrize("name, value", [item for item in OLD_VALUES
                                         if item[0] in ("train_window_offset", "fill_uninitialized_memory")])
def test_a_saved_baseline_value_is_dropped_on_load(name: str, value: Any) -> None:
    expected = BaselineConfig()
    assert BaselineConfig.from_serialized({**expected.serialize(), name: value}) == expected


def test_methods_contract_reports_replays_of_a_run_without_the_retired_toggle() -> None:
    run = RunRecord(run_dir=Path("run"), config_raw={"fp32_matmul_precision": "high"},
                    summary={"matmul_precision_replays": 7})
    numerics = pt.render_items(pt.methods_contract_items(run, "best")["Numerical realization"], False)
    assert "stricter matmul precision" in numerics and "7" in numerics
    quiet = RunRecord(run_dir=Path("run"), config_raw={"fp32_matmul_precision": "highest"}, summary={})
    assert "stricter matmul precision" not in pt.render_items(
        pt.methods_contract_items(quiet, "best")["Numerical realization"], False)
