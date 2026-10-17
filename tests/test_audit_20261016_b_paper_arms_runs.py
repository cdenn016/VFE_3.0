r"""Audit 2026-10-16, lane B batch 4: how ``paper_arms.py`` finds, reuses and refuses runs on disk.

F19: the base run serves as the evaluated arm's replicate at its own ``(seed, data_seed)``. The budget writes
one training checkpoint at the end (``checkpoint_interval = max_steps``) while the base recorded its own
cadence (the 36.41 base: 90,000 of 120,000 steps), so the plan claimed the reuse while the launch rejected
the base as stale and refused to start. The cadence does not change the trained weights, so the base
fallback ignores it, and the plan claims the reuse only when the launch's check accepts it.

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
