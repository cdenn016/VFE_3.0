r"""Audit 2026-10-08 paper-pipeline items (P2-1, P2-10, P2-11, P2-12, P2-13, P2-26).

Tiny synthetic bases and hand-written run directories only (K = 4, two heads, two layers, vocabulary
7); nothing is trained. Covered: the evaluated row's label on a base without stored-covariance decay
and the ``n/a`` rows, the word-level budget under ``base_protocol``, run reuse bound to the plan's
config, code identity in the arm homogeneity check, the frozen compute-matched width rule, the
calibration config and device checks of both launchers, the registered seed-paired decision rule,
training-only GPU-hours in M2, and the methods-contract numerics, clipping and training-sample items.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

import paper_arms as pa
from tests.test_paper_arms_20260928 import (  # noqa: F401 - fixtures used by name
    N_TOKENS, WORD, _cpu_one_thread, _fake_run, _tiny_cfg, _write_base, caches,
)
from vfe3.paper import tables as pt
from vfe3.paper.runs import Arm, Manifest, RunRecord, check_homogeneity, resolve_manifest

LOGGER = logging.getLogger("audit-20261008-paper")


# ---- P2-10: evaluated label, inapplicable rows, word-level budget ---------------------------------

def test_evaluated_row_is_labeled_by_the_base_decay_and_inapplicable_rows_print_na(
        tmp_path: Path, caches: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(sigma_weight_decay=0.0)))
    seeds = [(1, 1)]
    plans = pa.build_plan(base, budget="one_epoch", seeds=seeds, counts=False)
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "one_epoch")
    evaluated_cfg = next(p.cfg for p in plans if p.label == pa.EVALUATED)
    run = _fake_run(exp / "evaluated", "3.21_evaluated_s1_d1", evaluated_cfg)
    manifest = pa.build_manifest(exp, plans, base=base, budget="one_epoch", seeds=seeds)

    assert pa.evaluated_label(base.cfg) == pa.EVALUATED_NO_DECAY
    assert pa.EVALUATED not in manifest["arms"]
    assert list(manifest["arms"])[0] == pa.EVALUATED_NO_DECAY
    assert manifest["arms"][pa.EVALUATED_NO_DECAY]["runs"] == [str(run)]
    assert manifest["decision"]["evaluated"] == pa.EVALUATED_NO_DECAY
    no_decay = manifest["arms"]["Learned frames, stored covariance without decay"]
    assert no_decay["runs"] == [] and "sigma_weight_decay=0" in no_decay["inapplicable"]

    resolved = resolve_manifest(json.loads(json.dumps(manifest)))
    latex, _, rows = pt.main_results_table(resolved, read_checkpoints=False)
    assert "Evaluated: learned frames, decayed" not in latex
    assert pa.EVALUATED_NO_DECAY in latex
    na_row = next(line for line in latex.splitlines() if line.startswith("Identity frames, decayed"))
    assert na_row.count("n/a") == 5 and "pending" not in na_row
    assert next(r for r in rows if r["arm"].startswith("Identity frames, decayed"))["inapplicable"]
    # A decayed base keeps the original label.
    assert pa.evaluated_label(_tiny_cfg(sigma_weight_decay=0.2)) == pa.EVALUATED


def test_word_level_rows_train_one_epoch_under_the_base_protocol(tmp_path: Path, caches: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(max_steps=40, eval_interval=5)))
    plans = {p.label: p for p in pa.build_plan(
        base, budget="base_protocol", seeds=[(1, 1)], counts=False,
        labels=[pa.EVALUATED, "Gaussian attention", "Conventional transformer, matched"])}
    assert plans[pa.EVALUATED].cfg.max_steps == 40
    epoch = math.ceil(N_TOKENS["train"] / (2 * 4))
    assert plans["Gaussian attention"].cfg.max_steps == epoch
    assert any("one_epoch" in note for note in plans["Gaussian attention"].notes)
    assert plans["Conventional transformer, matched"].cfg.max_steps == epoch


# ---- P2-11: reuse bound to the plan's config ----------------------------------------------------

def test_a_complete_run_with_another_config_is_stale_not_reused(tmp_path: Path, caches: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    seeds = [(1, 1)]
    plans = pa.build_plan(base, budget="one_epoch", seeds=seeds, counts=False, labels=["MLP removed"])
    plan = plans[0]
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "one_epoch")
    changed = pa.rebuild(plan.cfg, {"m_p_mu_lr": float(plan.cfg.m_p_mu_lr) * 2}, "test")
    old = _fake_run(exp / "no_mlp", "3.50_no_mlp_s1_d1", changed)

    stale: List[Dict[str, str]] = []
    assert pa.completed_run(exp, "MLP removed", 1, 1, cfg=plan.cfg, dataset=plan.dataset, stale=stale) is None
    assert stale and stale[0]["run"] == str(old) and "m_p_mu_lr" in stale[0]["reason"]
    assert pa.completed_run(exp, "MLP removed", 1, 1) == old            # unchecked lookup still finds it
    manifest = pa.build_manifest(exp, plans, base=base, budget="one_epoch", seeds=seeds)
    assert manifest["arms"]["MLP removed"]["runs"] == []
    assert manifest["paper_arms"]["rows"]["MLP removed"]["stale"][0]["run"] == str(old)
    with pytest.raises(SystemExit, match="do not match the plan"):
        pa.run_plan(base, plans, budget="one_epoch", out_root=tmp_path / "out", device="cpu", logger=LOGGER)

    # The plan's own config (another seed is not a config difference) is reused.
    good = _fake_run(exp / "no_mlp", "3.40_no_mlp_s1_d1_2", plan.cfg)
    assert pa.completed_run(exp, "MLP removed", 1, 1, cfg=plan.cfg, dataset=plan.dataset) == good
    assert pa.reuse_mismatch(good, plan.cfg, dataset=plan.dataset, seed=1, data_seed=2) is not None
    assert pa.reuse_mismatch(good, plan.cfg, dataset=WORD, seed=1, data_seed=1) is not None


# ---- P2-12: code identity in the homogeneity check ----------------------------------------------

def _record(name: str, provenance: Dict[str, Any]) -> RunRecord:
    seed = ord(name[0])                                   # distinct seed pairs per run
    return RunRecord(run_dir=Path(name), config_raw={"embed_dim": 4}, provenance=provenance,
                     seed=seed, data_seed=seed)


def test_homogeneity_reports_mixed_code_identity() -> None:
    same = [_record("a", {"git_sha": "abc", "git_dirty": False}), _record("b", {"git_sha": "abc", "git_dirty": False})]
    assert check_homogeneity(same)["homogeneous"]
    mixed = [_record("a", {"git_sha": "891cb1a", "git_dirty": True}),
             _record("b", {"git_sha": "a81a4aa", "git_dirty": False})]
    report = check_homogeneity(mixed)
    assert not report["homogeneous"]
    assert [v["field"] for v in report["violations"]] == ["provenance:code_identity"]
    # The source-archive content hash wins over the git triple when every run records it.
    archived = [_record(n, {"git_sha": sha, "source_archive": {"available": True, "tree_sha256": "t"}})
                for n, sha in (("a", "1"), ("b", "2"))]
    assert check_homogeneity(archived)["homogeneous"]


# ---- P2-10 / P2-13: compute-matched rule and calibrations ---------------------------------------

def _write_metrics(run: Path, rows: Sequence[Dict[str, float]]) -> None:
    columns = sorted({key for row in rows for key in row})
    lines = [",".join(columns)] + [",".join(str(row.get(c, "")) for c in columns) for row in rows]
    (run / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _calibration_run(root: Path, cfg: Any, width: int, *, device_name: Optional[str], **change: Any) -> Path:
    run = root / f"calibration_K{width}_s1"
    run.mkdir(parents=True)
    wide = dataclasses.replace(cfg.with_width(width), seed=1, max_steps=4, warmup_steps=0, log_interval=1,
                               eval_interval=0, checkpoint_interval=0, **change)
    (run / "config.json").write_text(json.dumps({"config": wide.serialize()}), encoding="utf-8")
    _write_metrics(run, [{"step": s, "train_step_tokens_per_s": 1000.0 * (8 / width), "train_step_ms_mean": 5.0}
                         for s in (1, 2, 3, 4)])
    if device_name is not None:
        (run / "calibration.json").write_text(json.dumps({"device_name": device_name}), encoding="utf-8")
    return run


def test_train_baseline_compute_match_validates_config_and_device(tmp_path: Path) -> None:
    import vfe3.baseline  # noqa: F401
    import train_baseline as tb
    from vfe3.baseline import BaselineConfig, calibration_mismatch

    cfg = BaselineConfig.matched_to(_tiny_cfg())
    evaluated = tmp_path / "evaluated"
    evaluated.mkdir()
    _write_metrics(evaluated, [{"step": s, "train_step_ms_mean": 100.0} for s in (10, 20)])
    (evaluated / "provenance.json").write_text(json.dumps({"device_name": "cpu"}), encoding="utf-8")
    calib = {w: _calibration_run(tmp_path / "cal", cfg, w, device_name="cpu") for w in (4, 8)}
    rule = tb.compute_matched_config(cfg, evaluated_run=evaluated, calibration_runs=calib,
                                     unique_train_tokens=1000)          # passed before only with provenance
    assert rule["chosen_width"] in (4, 8) and rule["budget"]["device_name"] == "cpu"

    other = _calibration_run(tmp_path / "gpu", cfg, 8, device_name="NVIDIA GeForce RTX 5090")
    with pytest.raises(ValueError, match="throughput must share a device"):
        tb.compute_matched_config(cfg, evaluated_run=evaluated, calibration_runs={4: calib[4], 8: other},
                                  unique_train_tokens=1000)
    drifted = _calibration_run(tmp_path / "rope", cfg, 8, device_name="cpu", pos_rotation="rope")
    assert calibration_mismatch(cfg, drifted, 8) == ["pos_rotation"]
    with pytest.raises(ValueError, match="differs from the matched arm"):
        tb.compute_matched_config(cfg, evaluated_run=evaluated, calibration_runs={4: calib[4], 8: drifted},
                                  unique_train_tokens=1000)
    # paper_arms skips a calibration of another configuration instead of reusing it.
    assert pa._find_calibration(tmp_path / "rope", 8, 1, 4, cfg=cfg) is None
    assert pa._find_calibration(tmp_path / "cal", 8, 1, 4, cfg=cfg) == calib[8]


def test_compute_matched_width_rule_is_frozen_after_its_first_launch(
        tmp_path: Path, caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.baseline  # noqa: F401
    import train_baseline as tb

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a frozen width rule needs no calibration")

    monkeypatch.setattr(tb, "calibrate_widths", refuse)
    monkeypatch.setitem(pa.BUDGETS, "tiny", lambda cfg, dataset: {
        "max_steps": 4, "eval_interval": 2, "checkpoint_interval": 4})
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(vocab_size=9), dataset=WORD))
    plan = next(p for p in pa.build_plan(base, budget="tiny", seeds=[(1, 1)], counts=False,
                                         labels=["Matched measured compute"]))
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "tiny")
    frozen = dataclasses.replace(plan.cfg.with_width(8), max_steps=11, eval_interval=1, checkpoint_interval=11)
    rule_path = exp / plan.slug / pa.WIDTH_RULE_FILE
    rule_path.parent.mkdir(parents=True)
    rule_path.write_text(json.dumps({"config": frozen.serialize(), "evaluated_runs": ["gone"]}), encoding="utf-8")
    before = rule_path.read_bytes()
    cfg = pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER)
    assert cfg == frozen and rule_path.read_bytes() == before
    assert pa._expected_cfg(exp, plan) == frozen

    rule_path.write_text(json.dumps({"config": dataclasses.replace(frozen, lr=frozen.lr * 2).serialize()}),
                         encoding="utf-8")
    with pytest.raises(RuntimeError, match="another matched configuration"):
        pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER)


# ---- P2-26: the registered decision rule ---------------------------------------------------------

def _arm(label: str, ces: Sequence[float], pairs: Sequence[tuple], group: str = "g") -> Arm:
    runs = [RunRecord(run_dir=Path(f"{label[:8]}_{s}_{d}"), summary={"test_ce": ce, "n_steps": 10},
                      seed=s, data_seed=d) for ce, (s, d) in zip(ces, pairs)]
    return Arm(label=label, group=group, runs=runs)


def test_decision_rule_uses_paired_se_and_the_003_nat_margin() -> None:
    pairs = [(6, 3), (23, 23), (54, 54)]
    evaluated = [3.00, 3.10, 3.05]
    comparator = [3.10, 3.25, 3.12]                  # d = 0.10, 0.15, 0.07
    frozen = [3.01, 3.10, 3.07]                      # d = 0.01, 0.00, 0.02
    manifest = Manifest(arms=[
        _arm(pa.EVALUATED_NO_DECAY, evaluated, pairs, group="retraining"),
        _arm(pt.DECISION_FROZEN, frozen, pairs, group="retraining"),
        _arm("Per-type table of the frame table's size", [3.2, 3.3, 3.2], pairs, group="baselines"),
        _arm(pt.DECISION_COMPARATOR, comparator + [3.0], pairs + [(7, 7)], group="confounds"),
    ], selection="final")
    record = pt.decision_rule(manifest)
    a, b = record["clauses"]["a"], record["clauses"]["b"]
    d = [0.10, 0.15, 0.07]
    mean = sum(d) / 3
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / 2) / math.sqrt(3)
    assert a["n"] == 3 and a["mean"] == pytest.approx(mean) and a["se"] == pytest.approx(se)
    assert a["margin"] == pytest.approx(max(2 * se, 0.03)) and a["holds"] is True
    assert a["unmatched_seed_pairs"] == [[7, 7]]
    assert b["mean"] == pytest.approx(0.01) and b["margin"] == pytest.approx(0.03) and b["holds"] is False
    assert record["verdict"] == "retracted"
    assert record["labels"]["evaluated"] == pa.EVALUATED_NO_DECAY
    assert record["other_read_table_arms"] == ["Per-type table of the frame table's size"]
    latex, markdown, _ = pt.decision_rule_table(manifest)
    assert "retracted" in markdown and r"\begin{tabular}{lccccc}" in latex
    assert all(line.count("&") == 5 for line in latex.splitlines() if line.endswith(r"\\"))

    one_seed = Manifest(arms=[_arm(pa.EVALUATED, [3.0], pairs[:1]), _arm(pt.DECISION_COMPARATOR, [3.2], pairs[:1])],
                        selection="final")
    assert pt.decision_rule(one_seed)["clauses"]["a"]["holds"] is None
    assert pt.decision_rule(one_seed)["verdict"] == "undetermined"


# ---- P2-10 GPU-hours and P2-1 methods contract ----------------------------------------------------

def test_m2_gpu_hours_are_training_only_with_wall_time_beside(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metrics(run_dir, [{"step": 1000, "train_step_ms_mean": 360.0}, {"step": 2000, "train_step_ms_mean": 360.0}])
    (run_dir / "summary.json").write_text(json.dumps({"test_ce": 3.0, "n_steps": 2000, "wall_time_s": 1080.0}))
    (run_dir / "config.json").write_text(json.dumps({"config": {"embed_dim": 4}}))
    resolved = resolve_manifest({"arms": {"Arm": [str(run_dir)]}, "selection": "final"})
    latex, markdown, rows = pt.main_results_table(resolved, read_checkpoints=False)
    assert rows[0]["device_hours"] == [pytest.approx(0.2)] and rows[0]["wall_device_hours"] == [pytest.approx(0.3)]
    assert "$0.2$" in latex and "0.20 / 0.30" in markdown and "training-only" in markdown


def test_methods_contract_reports_replays_precision_toggles_per_role_clip_and_sample_caps() -> None:
    run = RunRecord(run_dir=Path("run"), config_raw={
        "amp_dtype": "bf16", "fp32_matmul_precision": "high", "matmul_precision_retry": True,
        "decode_logits_amp": False, "grad_clip": 1.0, "grad_clip_per_role": True},
        summary={"matmul_precision_replays": 266})
    items = pt.methods_contract_items(run, "best")
    numerics = pt.render_items(items["Numerical realization"], False)
    assert "matmul_precision_retry=True" in numerics and "decode_logits_amp=False" in numerics
    assert "266" in numerics
    optimization = pt.render_items(items["Optimization"], False)
    assert "per optimizer role" in optimization and "grad_clip_per_role" not in optimization
    evaluation = pt.render_items(items["Evaluation"], False)
    assert "32,768" in evaluation and "262,144" in evaluation
    unrecorded = RunRecord(run_dir=Path("run"), config_raw={"matmul_precision_retry": True})
    assert "not recorded" in pt.render_items(pt.methods_contract_items(unrecorded, "best")["Numerical realization"],
                                             False)
