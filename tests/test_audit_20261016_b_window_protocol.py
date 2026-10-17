r"""Audit 2026-10-16 lane B, F59: the training-window protocol of a saved run.

``train_window_offset`` existed from 2026-10-09 (default ``'fixed'``) until 2026-10-12, when the per-epoch
random window origin was hardcoded and every config loader began dropping the saved field silently. A run
trained before 2026-10-09, or with ``'fixed'``, saw other training windows than a run trained now, yet the
loaded configs were identical, so paper_arms reuse and the multi-seed homogeneity check treated the two
protocols as one. :func:`vfe3.experiment.saved_train_window_mode` recovers the protocol from the saved
field, or else from the run's commit relative to the retirement commit; paper_arms flags a reused run of
another protocol in the plan and the manifest, and multi-seed aggregation refuses seeds whose known
protocols differ.

Every fixture is a tiny JSON run directory (configs K = 4); nothing is trained.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import multiseed_analysis as ms
import paper_arms as pa
import vfe3.experiment as experiment
import vfe3.paper.runs as pr
from tests.test_multiseed import _write_full_run, _write_request
from tests.test_paper_arms_20260928 import _tiny_cfg, _write_base

REPO = Path(__file__).resolve().parent.parent
# 2026-10-09: the commit that introduced train_window_offset (a commit that predates the retirement).
INTRODUCTION_COMMIT = "4421982bafdf70b4341cbf0a2431a7fb796d8aef"


def _commit_known(sha: str) -> bool:
    try:
        return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=str(REPO),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_history = pytest.mark.skipif(
    not (_commit_known(experiment.TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT) and _commit_known(INTRODUCTION_COMMIT)),
    reason="the repository history with the 2026-10-09 and 2026-10-12 commits is not available")


# =============================================================================
# saved_train_window_mode
# =============================================================================

def test_a_recorded_train_window_offset_is_the_protocol():
    assert experiment.saved_train_window_mode({"train_window_offset": "fixed"}, {})[0] == "fixed"
    mode, evidence = experiment.saved_train_window_mode({"train_window_offset": "per_epoch_random"}, {})
    assert mode == "per_epoch_random" and "config.json" in evidence
    assert experiment.saved_train_window_mode({"train_window_offset": "sideways"}, {})[0] == "unknown"


@needs_history
def test_an_unrecorded_offset_is_placed_by_the_run_commit():
    retired = experiment.TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT
    assert experiment.saved_train_window_mode({}, {"git_sha": retired})[0] == "per_epoch_random"
    mode, evidence = experiment.saved_train_window_mode({}, {"git_sha": INTRODUCTION_COMMIT})
    assert mode == "fixed" and INTRODUCTION_COMMIT[:10] in evidence
    assert experiment.CURRENT_TRAIN_WINDOW_MODE == "per_epoch_random"


def test_a_recorded_provenance_mode_decides_before_the_commit():
    mode, evidence = experiment.saved_train_window_mode({}, {"train_window_mode": "fixed", "git_sha": "f" * 40})
    assert mode == "fixed" and "provenance.json" in evidence
    # the saved config field still decides first
    assert experiment.saved_train_window_mode({"train_window_offset": "per_epoch_random"},
                                              {"train_window_mode": "fixed"})[0] == "per_epoch_random"


def test_an_unplaceable_run_is_unknown():
    assert experiment.saved_train_window_mode({}, {})[0] == "unknown"
    assert experiment.saved_train_window_mode({}, {"git_sha": "f" * 40})[0] == "unknown"


# =============================================================================
# multiseed_analysis: seeds of different known protocols are not one group
# =============================================================================

def _group(root: Path, offsets: Dict[int, Optional[str]], git_sha: Optional[str] = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_request(root, sorted(offsets))
    for i, (seed, offset) in enumerate(sorted(offsets.items())):
        run = root / f"run_s{seed}"
        _write_full_run(run, ppl=10.0 + i, seed=seed)
        if offset is not None:
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            (run / "config.json").write_text(json.dumps({**config, "train_window_offset": offset}),
                                             encoding="utf-8")
        if git_sha is not None:
            provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
            (run / "provenance.json").write_text(json.dumps({**provenance, "git_sha": git_sha}),
                                                 encoding="utf-8")


def test_seeds_of_different_window_protocols_are_unverifiable(tmp_path: Path):
    _group(tmp_path, {1: "fixed", 2: "per_epoch_random"})
    design = ms._requested_seed_design(tmp_path)
    assert design["complete"] is False
    assert [cell["status"] for cell in design["cells"]] == ["unverifiable", "unverifiable"]
    assert design["train_window_mode"]["mixed"] is True
    assert design["train_window_mode"]["per_seed"] == {"1": "fixed", "2": "per_epoch_random"}


def test_one_protocol_or_an_unknown_one_still_aggregates(tmp_path: Path):
    _group(tmp_path / "same", {1: "fixed", 2: "fixed"})
    same = ms._requested_seed_design(tmp_path / "same")
    assert same["complete"] is True and same["train_window_mode"]["modes"] == ["fixed"]
    _group(tmp_path / "unknown", {1: "fixed", 2: None})        # seed 2: no field, an unplaceable commit
    unknown = ms._requested_seed_design(tmp_path / "unknown")
    assert unknown["complete"] is True
    assert unknown["train_window_mode"]["per_seed"] == {"1": "fixed", "2": "unknown"}


# =============================================================================
# paper_arms: a reused run of another protocol is flagged in the plan and the manifest
# =============================================================================

@pytest.fixture()
def _fixed_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "current_code_identity",
                        lambda kind: {"git_sha": "head", "git_dirty": False, "git_dirty_fingerprint": None})


def _base(root: Path, provenance: Dict[str, Any], **over: Any) -> pa.Base:
    run = _write_base(root, _tiny_cfg(seed=6, **over))
    (run / "provenance.json").write_text(json.dumps({"seed": 6, "data_seed": 3, **provenance}))
    (run / "summary.json").write_text(json.dumps({"n_steps": 100, "wall_time_s": 20.0, "test_ce": 3.5}))
    return pa.load_base("tiny", run)


@needs_history
def test_a_fixed_window_base_is_reused_and_flagged(tmp_path: Path, _fixed_code_identity: None):
    base = _base(tmp_path, {"git_sha": INTRODUCTION_COMMIT, "git_dirty": False, "git_dirty_fingerprint": None},
                 max_steps=8, eval_interval=2, checkpoint_interval=8)
    assert base.train_window_mode == "fixed"
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(6, 3), (23, 23)], labels=[pa.EVALUATED],
                          counts=False)
    plan = plans[0]
    assert plan.reused == [(6, 3)]                          # the reuse itself is unchanged
    assert any("fixed training windows" in note and "kept and flagged" in note for note in plan.notes)
    assert "training windows: fixed" in pa.format_plan(base, plans, budget="base_protocol")
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    manifest = pa.build_manifest(exp, plans, base=base, budget="base_protocol", seeds=[(6, 3), (23, 23)])
    row = manifest["paper_arms"]["rows"][pa.EVALUATED]
    assert row["train_window_mode"] == {str(base.run_dir): "fixed"}
    assert [entry["run"] for entry in row["window_protocol_mismatch"]] == [str(base.run_dir)]
    assert "per_epoch_random" in row["window_protocol_mismatch"][0]["reason"]


@needs_history
def test_a_current_protocol_base_is_not_flagged(tmp_path: Path, _fixed_code_identity: None):
    base = _base(tmp_path, {"git_sha": experiment.TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT, "git_dirty": False,
                            "git_dirty_fingerprint": None},
                 max_steps=8, eval_interval=2, checkpoint_interval=8)
    assert base.train_window_mode == "per_epoch_random"
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(6, 3)], labels=[pa.EVALUATED], counts=False)
    assert not any("training windows" in note for note in plans[0].notes)
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    manifest = pa.build_manifest(exp, plans, base=base, budget="base_protocol", seeds=[(6, 3)])
    assert manifest["paper_arms"]["rows"][pa.EVALUATED]["window_protocol_mismatch"] == []
