r"""Audit 2026-10-17, lane L6 (experiment drivers and statistics).

Synthetic fixtures only: no sweep, no training run beyond a tiny in-test model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from vfe3.paper import tables as paper_tables
from vfe3.paper.runs import Arm, Manifest, RunRecord, check_homogeneity


# ---- item 27: code identity is a note, run cadence is ignored ----------------------------------

def _probe_record(name: str, seed: int, checkpoint_interval: int, tree: str) -> RunRecord:
    return RunRecord(run_dir=Path(name),
                     config_raw={"embed_dim": 4, "seed": seed, "checkpoint_interval": checkpoint_interval,
                                 "log_interval": 100 if name == "base" else 50,
                                 "resume_from": None if name == "base" else "x/ckpt.pt"},
                     provenance={"source_archive": {"available": True, "tree_sha256": tree}},
                     seed=seed, data_seed=3)


def _base_plus_replicates() -> List[RunRecord]:
    return [_probe_record("base", 6, 900, "688f"),
            _probe_record("rep_s7", 7, 1200, "97ab"),
            _probe_record("rep_s8", 8, 1200, "97ab")]


def test_base_plus_replicates_is_homogeneous_with_code_identity_note() -> None:
    report = check_homogeneity(_base_plus_replicates())
    assert report["homogeneous"], report["violations"]
    assert report["violations"] == []
    assert [d["field"] for d in report["code_identity_differences"]] == ["provenance:code_identity"]
    arm = Arm(label="Evaluated", runs=_base_plus_replicates(), homogeneity=report)
    assert paper_tables._unverified_reasons(arm) == []
    notes = paper_tables._code_identity_notes(arm)
    assert len(notes) == 1 and "2 different code identities" in notes[0]


def test_homogeneity_still_flags_a_real_config_difference() -> None:
    runs = _base_plus_replicates()
    runs[2].config_raw["embed_dim"] = 8
    report = check_homogeneity(runs)
    assert not report["homogeneous"]
    assert [v["field"] for v in report["violations"]] == ["embed_dim"]


# ---- item 84: realized one-sided level of the 2 SE margin ----------------------------------------

def test_decision_note_states_realized_margin_level(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"Evaluated": {(1, 3): 3.0, (2, 3): 3.1, (4, 3): 3.05},
              "cmp": {(1, 3): 3.2, (2, 3): 3.25, (4, 3): 3.3},
              "frz": {(1, 3): 3.2, (2, 3): 3.25, (4, 3): 3.3},
              "cap": {(1, 3): 3.2, (2, 3): 3.25, (4, 3): 3.3}}
    monkeypatch.setattr(paper_tables, "_selected_by_pair", lambda arm, rule: (values[arm.label], []))
    manifest = Manifest(arms=[Arm(label=k) for k in values],
                        decision={"evaluated": "Evaluated", "comparator": "cmp", "frozen": "frz", "capacity": "cap"})
    _, md, record = paper_tables.decision_rule_table(manifest)
    alpha = record["clauses"]["a"]["margin_alpha_one_sided"]
    assert alpha == pytest.approx(0.5 * (1.0 - 2.0 / math.sqrt(6.0)), abs=1e-9)
    assert "realized level alpha = 0.092" in md
