r"""Audit 2026-10-16, lane B batch 4: how ``paper_arms.py`` finds, reuses and refuses runs on disk.

F19: the base run serves as the evaluated arm's replicate at its own ``(seed, data_seed)``. The budget writes
one training checkpoint at the end (``checkpoint_interval = max_steps``) while the base recorded its own
cadence (the 36.41 base: 90,000 of 120,000 steps), so the plan claimed the reuse while the launch rejected
the base as stale and refused to start. The cadence does not change the trained weights, so the base
fallback ignores it, and the plan claims the reuse only when the launch's check accepts it.
F60: of several complete runs of one seed pair, the lexicographically first folder (prefixed by its test PPL)
was selected and the duplicates never reached the decision rule; now no run is selected, every duplicate is
reported as stale (the decision rule then reports the clause undetermined) and a launch refuses to start.
F61: a finished run whose test CE is not finite counted as never run, so a relaunch silently re-rolled the
seed pair; now it is a recorded failure: listed under the manifest row's ``failed``, never retrained.
F63: a ``BASES`` entry whose run directory moved crashed the whole launch in ``load_base`` with a bare
``FileNotFoundError`` before any plan printed; now ``load_base`` names the base and its path, and ``main``
reports that base as unavailable and plans the others.

Every config is tiny (K = 4, two heads, two layers, vocabulary 7); nothing is trained (``run_vfe`` is
replaced by a recorder).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import paper_arms as pa
import vfe3.experiment as experiment
import vfe3.paper.runs as pr
from tests.test_paper_arms_20260928 import (  # noqa: F401 - fixtures used by name
    _cpu_one_thread, _fake_run, _tiny_cfg, _write_base,
)

LOGGER = logging.getLogger("audit-20261016-b-paper-arms-runs")


@pytest.fixture(autouse=True)
def _fixed_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "current_code_identity",
                        lambda kind: {"git_sha": "head", "git_dirty": False, "git_dirty_fingerprint": None})


def _base(root: Path, **over: Any) -> pa.Base:
    r"""A complete base run at model seed 6 and data seed 3 (the paper bases' pair)."""
    run = _write_base(root, _tiny_cfg(seed=6, **over))
    (run / "provenance.json").write_text(json.dumps({"seed": 6, "data_seed": 3}))
    (run / "summary.json").write_text(json.dumps({"n_steps": 100, "wall_time_s": 20.0, "test_ce": 3.5}))
    return pa.load_base("tiny", run)


def _recorder(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[int, int]]:
    trained: List[Tuple[int, int]] = []

    def run_vfe(cfg: Any, **kwargs: Any) -> Path:
        trained.append((kwargs["seed"], kwargs["data_seed"]))
        return Path(kwargs["out_root"]) / "never_written"

    monkeypatch.setattr(experiment, "run_vfe", run_vfe)
    return trained


# ---- F19: the base run's own checkpoint cadence ------------------------------------------------

def test_a_base_with_its_own_checkpoint_cadence_is_reused_by_the_plan_and_the_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base(tmp_path, max_steps=8, eval_interval=2, checkpoint_interval=6)
    seeds = [(6, 3), (23, 23)]
    plans = pa.build_plan(base, budget="base_protocol", seeds=seeds, labels=[pa.EVALUATED], counts=False)
    plan = plans[0]
    assert (base.cfg.checkpoint_interval, plan.cfg.checkpoint_interval) == (6, 8)
    assert plan.reused == [(6, 3)]
    assert any("checkpoint_interval" in note and "ignored" in note for note in plan.notes)
    # An experiment-directory run is still checked on every field, the cadence included.
    strict = pa.reuse_mismatch(base.run_dir, plan.cfg, dataset=plan.dataset, seed=6, data_seed=3)
    assert strict is not None and "checkpoint_interval" in strict

    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    stale: List[Dict[str, str]] = []
    assert pa.completed_run(exp, pa.EVALUATED, 6, 3, base=base, budget="base_protocol", cfg=plan.cfg,
                            dataset=plan.dataset, stale=stale) == base.run_dir
    assert stale == []
    trained = _recorder(monkeypatch)
    manifest_path = pa.run_plan(base, plans, budget="base_protocol", out_root=tmp_path / "out", device="cpu",
                                logger=LOGGER)
    assert trained == [(23, 23)]                                   # the base pair is reused, not retrained
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["arms"][pa.EVALUATED]["runs"] == [str(base.run_dir)]
    row = manifest["paper_arms"]["rows"][pa.EVALUATED]
    assert row["stale"] == [] and row["reuse_verified"] is True


def test_the_compute_matched_rule_reads_a_base_with_its_own_checkpoint_cadence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import train_baseline  # noqa: F401
    import vfe3.baseline as api

    class Reached(Exception):
        pass

    def budget(run_dir: Any) -> Dict[str, Any]:
        raise Reached(str(run_dir))

    base = _base(tmp_path, max_steps=8, eval_interval=2, checkpoint_interval=6)
    plan = pa.build_plan(base, budget="base_protocol", seeds=[(6, 3)], counts=False,
                         labels=["Matched measured compute"])[0]
    assert plan.status == "deferred"
    monkeypatch.setattr(api, "measured_train_device_hours", budget)
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    # Past the deferral: the reused base is the evaluated run whose device-hours set the budget.
    with pytest.raises(Reached, match="base_run"):
        pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER,
                                  base=base, budget="base_protocol")


def test_a_base_that_differs_beyond_the_cadence_is_not_claimed_and_the_launch_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(pa.BUDGETS, "base_protocol", lambda cfg, dataset: {
        "max_steps": 4, "eval_interval": 2, "checkpoint_interval": 4})
    base = _base(tmp_path, max_steps=40)
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(6, 3)], labels=[pa.EVALUATED], counts=False)
    plan = plans[0]
    assert plan.reused == []
    assert any("not reused" in note and "max_steps" in note for note in plan.notes)
    trained = _recorder(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        pa.run_plan(base, plans, budget="base_protocol", out_root=tmp_path / "out", device="cpu", logger=LOGGER)
    message = str(raised.value)
    assert "max_steps" in message and "the base run itself" in message and "leave it in place" in message
    assert trained == []


# ---- F60: duplicate complete runs of one seed pair ---------------------------------------------

MLP_REMOVED = "MLP removed"


def _mlp_removed(root: Path) -> Tuple[pa.Base, List[pa.ArmPlan], Path]:
    base = pa.load_base("tiny", _write_base(root, _tiny_cfg()))
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(23, 23)], labels=[MLP_REMOVED], counts=False)
    assert plans[0].status == "ready"
    return base, plans, pa.experiment_dir(root / "out", "tiny", "base_protocol")


def test_duplicate_complete_runs_of_a_seed_pair_select_none_and_block_the_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.paper import tables as pt
    from vfe3.paper.runs import resolve_manifest

    base, plans, exp = _mlp_removed(tmp_path)
    plan = plans[0]
    arm_dir = exp / plan.slug
    only = _fake_run(arm_dir, "36.80_no_mlp_s23_d23", plan.cfg, test_ce=3.60, seed=23, data_seed=23)
    assert pa.completed_run(exp, MLP_REMOVED, 23, 23, cfg=plan.cfg, dataset=plan.dataset) == only
    # A rerun of the same pair with a better test CE sorts first by name; neither is selected.
    rerun = _fake_run(arm_dir, "35.90_no_mlp_s23_d23_2", plan.cfg, test_ce=3.58, seed=23, data_seed=23)
    stale: List[Dict[str, str]] = []
    assert pa.completed_run(exp, MLP_REMOVED, 23, 23, cfg=plan.cfg, dataset=plan.dataset, stale=stale) is None
    assert sorted(entry["run"] for entry in stale) == sorted([str(only), str(rerun)])
    assert all("2 complete runs of seed pair (23, 23)" in entry["reason"] for entry in stale)
    assert pa.completed_run(exp, MLP_REMOVED, 23, 23) is None                  # the unchecked lookup too
    assert pa.find_complete_run(arm_dir, pa.run_label(plan.slug, 23, 23)) is None

    manifest = pa.build_manifest(exp, plans, base=base, budget="base_protocol", seeds=[(23, 23)])
    assert manifest["arms"][MLP_REMOVED]["runs"] == []
    assert len(manifest["paper_arms"]["rows"][MLP_REMOVED]["stale"]) == 2
    arm = next(a for a in resolve_manifest(json.loads(json.dumps(manifest))).arms if a.label == MLP_REMOVED)
    assert pt._unverified_reasons(arm)                                         # the decision rule sees them
    trained = _recorder(monkeypatch)
    with pytest.raises(SystemExit, match="duplicate a seed pair"):
        pa.run_plan(base, plans, budget="base_protocol", out_root=tmp_path / "out", device="cpu", logger=LOGGER)
    assert trained == []


# ---- F61: a finished run without a finite test CE ----------------------------------------------

def test_a_finished_run_with_a_nonfinite_test_ce_is_a_recorded_failure_not_retrained(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base, plans, exp = _mlp_removed(tmp_path)
    plan = plans[0]
    arm_dir = exp / plan.slug
    # A diverged run keeps its timestamped name (rename_run_by_test_ppl skips a nonfinite PPL).
    diverged = _fake_run(arm_dir, "20261017-101010_no_mlp_s23_d23", plan.cfg, test_ce=float("nan"),
                         seed=23, data_seed=23)
    assert pa.failed_run_reason(diverged) is not None and not pa.is_complete(diverged)
    failed: List[Dict[str, str]] = []
    stale: List[Dict[str, str]] = []
    assert pa.completed_run(exp, MLP_REMOVED, 23, 23, cfg=plan.cfg, dataset=plan.dataset, stale=stale,
                            failed=failed) is None
    assert stale == [] and [entry["run"] for entry in failed] == [str(diverged)]
    assert "nan" in failed[0]["reason"].lower()

    trained = _recorder(monkeypatch)
    manifest_path = pa.run_plan(base, plans, budget="base_protocol", out_root=tmp_path / "out", device="cpu",
                                logger=LOGGER)
    assert trained == []                                                       # the pair is not re-rolled
    row = json.loads(manifest_path.read_text(encoding="utf-8"))["paper_arms"]["rows"][MLP_REMOVED]
    assert [entry["run"] for entry in row["failed"]] == [str(diverged)] and row["complete"] == 0

    # A later success of the same pair is listed, and the failure stays on record beside it.
    success = _fake_run(arm_dir, "3.61_no_mlp_s23_d23", plan.cfg, test_ce=1.28, seed=23, data_seed=23)
    manifest = pa.build_manifest(exp, plans, base=base, budget="base_protocol", seeds=[(23, 23)])
    assert manifest["arms"][MLP_REMOVED]["runs"] == [str(success)]
    assert [entry["run"] for entry in manifest["paper_arms"]["rows"][MLP_REMOVED]["failed"]] == [str(diverged)]


def test_an_unfinished_run_or_a_failure_of_another_config_does_not_block_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base, plans, exp = _mlp_removed(tmp_path)
    plan = plans[0]
    arm_dir = exp / plan.slug
    unfinished = _fake_run(arm_dir, "20261017-090000_no_mlp_s23_d23", plan.cfg, test_ce=None, seed=23,
                           data_seed=23)                                      # no summary.json: interrupted
    other = pa.rebuild(plan.cfg, {"m_p_mu_lr": float(plan.cfg.m_p_mu_lr) * 2}, "test")
    _fake_run(arm_dir, "20261017-091111_no_mlp_s23_d23", other, test_ce=float("inf"), seed=23, data_seed=23)
    assert pa.failed_run_reason(unfinished) is None
    failed: List[Dict[str, str]] = []
    assert pa.completed_run(exp, MLP_REMOVED, 23, 23, cfg=plan.cfg, dataset=plan.dataset, failed=failed) is None
    assert failed == []
    trained = _recorder(monkeypatch)
    pa.run_plan(base, plans, budget="base_protocol", out_root=tmp_path / "out", device="cpu", logger=LOGGER)
    assert trained == [(23, 23)]


# ---- F63: a base whose run directory is gone ---------------------------------------------------

def test_a_missing_base_is_reported_unavailable_and_the_other_bases_still_plan(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    gone = tmp_path / "moved_away"
    with pytest.raises(pa.BaseUnavailableError, match="moved_away"):
        pa.load_base("gone", gone)
    assert issubclass(pa.BaseUnavailableError, FileNotFoundError)
    present = _write_base(tmp_path, _tiny_cfg())
    config = {**pa.CONFIG, "bases": ["gone", "tiny"], "arms": [MLP_REMOVED], "budget": "base_protocol",
              "seeds": [(1, 1)], "dry_run": True}
    assert pa.main(config, bases={"gone": str(gone), "tiny": str(present)}) == 1
    out = capsys.readouterr().out
    assert "base gone: unavailable" in out and str(gone) in out
    assert f"base tiny: {present}" in out and MLP_REMOVED in out
