r"""Audit 2026-10-31, package paper-tables-baseline: the paper assemblers and the paper-arms run records.

Synthetic run directories only (summary.json, provenance.json, config.json, posthoc.json, metrics.csv and
resources.json written by the test into a temporary folder); the S1 label test reuses the tiny model state of
``tests/test_paper_20260928.py`` (vocabulary 6, K = 4, two heads, one layer). Nothing is trained.

Covered: the registered decision rule makes a clause undetermined when a listed run has no selected test value
or a planned seed pair has no selected run (A046, paper-claims-1), and ``paper_arms.failed_run_reason`` keys a
failure to the manifest's selection rule (paper-arms-harness-4); a data or evaluation protocol mismatch between
a clause arm and the evaluated arm makes the clause undetermined and is noted in M2 (A367, paper-claims-7); a
post-hoc pass whose ``evaluate()`` cross-check failed is not admitted as a selected value (A368, paper-claims-8);
M2 marks frozen tables, prints sub-hour GPU-hours to two significant figures and ``count_short`` rounds before
it picks the suffix, and S1 panel B names the head transfer :math:`M_h` (A366, paper-claims-6); and
``paper_artifacts.py`` takes no run key from its CONFIG beside a JSON manifest and records every key's origin
(A365, paper-claims-10).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

import paper_arms as pa
from tests.test_paper_20260928 import _new_metrics, trained  # noqa: F401 - module fixture used by name
from tests.test_paper_20260928 import _write_run as write_trained_run
from tests.test_paper_arms_20260928 import (  # noqa: F401 - fixtures used by name
    _cpu_one_thread, _fake_run, _tiny_cfg, _write_base,
)
from vfe3.paper import tables as pt
from vfe3.paper.runs import load_run, resolve_manifest, select_checkpoint

EVAL = "Evaluated: learned frames"
COMP = "Per-type read table, ALiBi (as run)"
FROZ = "Frames frozen at their random initialization"
CAP = "Identity frames, per-type additive table of equal size"
SEEDS: List[Tuple[int, int]] = [(6, 3), (23, 23), (54, 54)]
TEST_SHA = "a" * 64


def _write_run(
    root:        Path,
    name:        str,
    seed:        int,
    data_seed:   int,

    *,
    test_ce:     Optional[float],
    test_best_ce: Optional[float],
    best_step:   int = 12000,
    status:      Optional[str] = None,
    provenance:  Optional[Dict[str, Any]] = None,
    config:      Optional[Dict[str, Any]] = None,
    posthoc:     Optional[Dict[str, Any]] = None,
    scored:      int = 280541,
) -> Path:
    r"""A finished run of 15,000 steps whose headline test is at the final weights."""
    run = root / name
    run.mkdir(parents=True)
    summary = {"n_steps": 15000, "evaluated_step": 15000, "evaluation_weights": "final", "reloaded_best": False,
               "best_step": best_step, "best_val_ppl": math.exp(3.45), "test_ce": test_ce,
               "test_best_ce": test_best_ce, "test_best_step": best_step, "test_scored_targets": scored,
               "test_best_status": status or ("evaluated" if test_best_ce is not None else None)}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    prov = {"seed": seed, "data_seed": data_seed, "tokenizer_tag": "tiktoken", "max_tokens": None,
            "val_data_sha256": "v" * 64, "val_data_n_tokens": 245568,
            "test_data_sha256": TEST_SHA, "test_data_n_tokens": 280542, **(provenance or {})}
    (run / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    cfg = {"max_seq_len": 128, "eval_stride": 32, "eval_interval": 1500, "max_steps": 15000, "batch_size": 32,
           "seed": seed, **(config or {})}
    (run / "config.json").write_text(json.dumps({"config": cfg, "dataset": "wikitext-103"}), encoding="utf-8")
    if posthoc is not None:
        (run / "posthoc").mkdir()
        (run / "posthoc" / "posthoc.json").write_text(json.dumps(posthoc), encoding="utf-8")
    return run


def _decision_manifest(
    root:       Path,
    comparator: Callable[[int], Dict[str, Any]],

    *,
    n_comparator: int = 3,
    planned:      Optional[Sequence[Tuple[int, int]]] = SEEDS,
) -> Dict[str, Any]:
    r"""Four decision arms at three seed pairs; every run is listed under ``runs`` (a hand-written manifest)."""
    arms: Dict[str, Any] = {}
    for label, prefix, offset in ((EVAL, "eval", 3.50), (FROZ, "froz", 3.70), (CAP, "cap", 3.65)):
        arms[label] = {"group": "Retraining", "runs": [
            str(_write_run(root, f"{prefix}_s{s}", s, d, test_ce=offset + 0.01 * i + 0.01,
                           test_best_ce=offset + 0.01 * i)) for i, (s, d) in enumerate(SEEDS)]}
    arms[COMP] = {"group": "Confounds", "runs": [
        str(_write_run(root, f"comp_s{s}", s, d, **comparator(i))) for i, (s, d) in enumerate(SEEDS[:n_comparator])]}
    manifest: Dict[str, Any] = {"arms": arms, "selection": "best",
                                "decision": {"evaluated": EVAL, "comparator": COMP, "frozen": FROZ, "capacity": CAP}}
    if planned is not None:
        manifest["paper_arms"] = {"seeds": [list(pair) for pair in planned], "rows": {}}
    return manifest


def _good(i: int) -> Dict[str, Any]:
    return {"test_ce": 3.87 + 0.01 * i, "test_best_ce": 3.86 + 0.01 * i}


# ---- A046 (paper-claims-1, paper-arms-harness-4): pairs dropped from the decision ---------------

def test_the_control_manifest_stands(tmp_path: Path) -> None:
    record = pt.decision_rule(resolve_manifest(_decision_manifest(tmp_path, _good)))
    assert [record["clauses"][c]["n"] for c in "abc"] == [3, 3, 3]
    assert all(not record["clauses"][c]["undetermined"] for c in "abc")
    assert record["verdict"] == "stands"


def test_a_listed_run_without_a_selected_value_makes_its_clause_undetermined(tmp_path: Path) -> None:
    def comparator(i: int) -> Dict[str, Any]:
        if i == 2:                       # finite final-weights CE below every evaluated run, no best-weights value
            return {"test_ce": 3.40, "test_best_ce": None,
                    "status": "failed: RuntimeError('best_model.pt bundle unreadable')"}
        return _good(i)

    manifest = _decision_manifest(tmp_path, comparator)
    lost = Path(manifest["arms"][COMP]["runs"][2])
    assert select_checkpoint(load_run(lost), "best")["status"] == "unavailable"
    record = pt.decision_rule(resolve_manifest(manifest))
    a = record["clauses"]["a"]
    assert a["n"] == 2 and a["mean"] is not None                       # statistics still reported
    assert a["holds"] is None and record["verdict"] == "undetermined"
    assert any("no selected test value" in reason and "comp_s54" in reason for reason in a["undetermined"])
    assert not record["clauses"]["b"]["undetermined"]                   # the other clauses do not read the arm
    _, markdown, _ = pt.decision_rule_table(resolve_manifest(manifest))
    assert "Clause (a) undetermined" in markdown


def test_a_planned_seed_pair_without_a_run_makes_its_clause_undetermined(tmp_path: Path) -> None:
    manifest = _decision_manifest(tmp_path, _good, n_comparator=2)
    resolved = resolve_manifest(manifest)
    assert resolved.planned_seed_pairs == SEEDS
    record = pt.decision_rule(resolved)
    a = record["clauses"]["a"]
    assert a["n"] == 2 and a["holds"] is None and record["verdict"] == "undetermined"
    assert any("planned seed pair" in reason and "[54, 54]" in reason for reason in a["undetermined"])
    # Without a planned design (a hand-written manifest) the surplus pair is reported, not decisive.
    unplanned = pt.decision_rule(resolve_manifest(_decision_manifest(tmp_path / "u", _good, n_comparator=2,
                                                                     planned=None)))
    assert unplanned["clauses"]["a"]["holds"] is True


def test_malformed_planned_seeds_are_a_manifest_problem(tmp_path: Path) -> None:
    manifest = _decision_manifest(tmp_path, _good, planned=None)
    manifest["paper_arms"] = {"seeds": [[6, 3], "x"], "rows": {}}
    resolved = resolve_manifest(manifest)
    assert resolved.planned_seed_pairs == [] and any("paper_arms.seeds" in p for p in resolved.problems)


def test_failed_run_reason_follows_the_manifest_selection_rule(tmp_path: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(23, 23)], labels=["MLP removed"], counts=False)
    plan = plans[0]
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    run = _fake_run(exp / plan.slug, "3.40_no_mlp_s23_d23", plan.cfg, test_ce=3.40, seed=23, data_seed=23)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary.update(test_best_ce=None, best_step=2, test_best_status="failed: RuntimeError('locked')")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert pa.is_complete(run)                                           # a finished training run
    reason = pa.failed_run_reason(run)
    assert reason is not None and "'best'" in reason and "locked" in reason
    failed: List[Dict[str, str]] = []
    assert pa.completed_run(exp, "MLP removed", 23, 23, cfg=plan.cfg, dataset=plan.dataset, failed=failed) is None
    assert [entry["run"] for entry in failed] == [str(run)]
    manifest = pa.build_manifest(exp, plans, base=base, budget="base_protocol", seeds=[(23, 23)])
    assert manifest["arms"]["MLP removed"]["runs"] == []
    assert manifest["paper_arms"]["rows"]["MLP removed"]["failed"][0]["run"] == str(run)
    assert manifest["selection"] == pa.MANIFEST_SELECTION == "best"
    # A post-hoc pass on best_model.pt supplies the registered value, and the run is listed again.
    (run / "posthoc").mkdir()
    (run / "posthoc" / "posthoc.json").write_text(json.dumps({"passes": [
        {"checkpoint": "best_model.pt", "split": "test", "status": "ok", "subset": False, "ce": 3.45}]}))
    assert pa.failed_run_reason(run) is None
    assert pa.completed_run(exp, "MLP removed", 23, 23, cfg=plan.cfg, dataset=plan.dataset) == run


# ---- A367 (paper-claims-7): paired arms on one data and evaluation protocol --------------------

def test_a_protocol_mismatch_with_the_evaluated_arm_makes_the_clause_undetermined(tmp_path: Path) -> None:
    def comparator(i: int) -> Dict[str, Any]:
        return {**_good(i), "provenance": {"test_data_sha256": "b" * 64, "test_data_n_tokens": 245021},
                "scored": 245020}

    manifest = resolve_manifest(_decision_manifest(tmp_path, comparator))
    assert all(arm.homogeneity["homogeneous"] for arm in manifest.arms)   # each arm is homogeneous
    record = pt.decision_rule(manifest)
    a = record["clauses"]["a"]
    assert a["n"] == 3 and a["holds"] is None and record["verdict"] == "undetermined"
    reasons = " ".join(a["undetermined"])
    assert "provenance:test_data_sha256" in reasons and "summary:test_scored_targets" in reasons
    assert not record["clauses"]["b"]["undetermined"]
    _, markdown, rows = pt.main_results_table(manifest, read_checkpoints=False)
    comparator_row = next(row for row in rows if row["arm"] == COMP)
    assert any("protocol differs from the evaluated arm" in note for note in comparator_row["notes"])
    assert not any("protocol differs" in note for row in rows if row["arm"] != COMP for note in row["notes"])


def test_budget_context_and_unrecorded_fields(tmp_path: Path) -> None:
    def comparator(i: int) -> Dict[str, Any]:
        return {**_good(i), "config": {"max_steps": 30000, "max_seq_len": 256}}

    record = pt.decision_rule(resolve_manifest(_decision_manifest(tmp_path / "a", comparator)))
    reasons = " ".join(record["clauses"]["a"]["undetermined"])
    assert "training_tokens" in reasons and "max_seq_len" in reasons
    # A field one arm does not record cannot be compared and is not a mismatch.
    unrecorded = resolve_manifest(_decision_manifest(tmp_path / "b", _good))
    for arm in unrecorded.arms:
        if arm.label == COMP:
            for run in arm.runs:
                run.provenance.pop("test_data_sha256")
                run.config_raw.pop("eval_stride")
    assert pt.decision_rule(unrecorded)["verdict"] == "stands"


# ---- A368 (paper-claims-8): a failed evaluate() cross-check --------------------------------------

def _posthoc(ok: Optional[bool], ce: float = 3.30) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"checkpoint": "best_model.pt", "split": "test", "status": "ok", "subset": False,
                             "ce": ce}
    if ok is not None:
        entry["cross_check"] = {"evaluate_ce": 3.52, "delta_ce": ce - 3.52, "tolerance": 1e-4, "ok": ok}
    return {"passes": [entry]}


def test_a_posthoc_pass_with_a_failed_cross_check_is_not_a_selected_value(tmp_path: Path) -> None:
    failed = load_run(_write_run(tmp_path, "failed", 6, 3, test_ce=3.55, test_best_ce=None,
                                 status="best_model.pt was not saved", posthoc=_posthoc(False)))
    selection = select_checkpoint(failed, "best")
    assert selection["status"] == "unavailable" and selection["test_ce"] is None
    assert "cross-check failed" in selection["reason"]
    from vfe3.paper.runs import posthoc_pass
    assert posthoc_pass(failed, "best_model.pt", "test") is None
    for ok in (True, None):                             # a passed or absent cross-check is admitted as before
        run = load_run(_write_run(tmp_path, f"ok_{ok}", 6, 3, test_ce=3.55, test_best_ce=None, posthoc=_posthoc(ok)))
        chosen = select_checkpoint(run, "best")
        assert chosen["status"] == "ok" and chosen["test_ce"] == pytest.approx(3.30)
        assert chosen["test_source"] == "posthoc.json best_model.pt test (full split)"


# ---- A366 (paper-claims-6): M2 cells and the S1 label --------------------------------------------

def test_count_short_rounds_before_it_picks_the_suffix() -> None:
    assert pt.count_short(999_950) == "1.00M"
    assert pt.count_short(999_999_999) == "1.00B"
    assert pt.count_short(999.6) == "1.00k"
    assert pt.count_short(99_996) == "100.0k"
    for value, text in ((5252608, "5.25M"), (456654162, "456.7M"), (1.2e9, "1.20B"), (999, "999"),
                        (1000, "1.00k"), (12_000, "12.00k"), (0, "0")):
        assert pt.count_short(value) == text, value


def test_hours_print_at_least_two_significant_figures() -> None:
    assert [pt.hours_digits(h) for h in (0.1333, 0.0183, 0.2, 1.383, 13.83, 0.0)] == [2, 3, 2, 1, 1, 1]


def _metrics_run(root: Path, name: str, seed: int, minutes: float, trainable: int, frozen: int = 0) -> Path:
    run = _write_run(root, name, seed, seed, test_ce=3.55, test_best_ce=3.55, best_step=15000)
    context = 2_000_000
    breakdown = {"per_type_tables": {"total": trainable - context}, "positional_frames": {"total": 0},
                 "context_processing": {"total": context}, "other": {"total": 0},
                 "frozen": ({"prior_bank.phi_embed": frozen, "total": frozen} if frozen else {"total": 0}),
                 "total": trainable, "total_including_frozen": trainable + frozen}
    (run / "resources.json").write_text(json.dumps({"parameter_breakdown": breakdown}), encoding="utf-8")
    step_ms = minutes * 60 * 1000 / 15000
    lines = ["step,train_step_ms_mean,val_ce"] + [f"{s},{step_ms},{3.5 if s % 3000 == 0 else ''}"
                                                   for s in range(500, 15001, 500)]
    (run / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run


def test_m2_marks_frozen_tables_and_prints_sub_hour_gpu_hours(tmp_path: Path) -> None:
    arms = {"long": {"group": "g", "runs": [str(_metrics_run(tmp_path, "long", 6, 83, 233_600_000))]},
            "frozen": {"group": "g", "runs": [str(_metrics_run(tmp_path, "frozen", 6, 75, 27_700_000, 205_900_000))]},
            "short8": {"group": "g", "runs": [str(_metrics_run(tmp_path, "short8", 6, 8, 14_900_000))]}}
    latex, markdown, _ = pt.main_results_table(resolve_manifest({"arms": arms, "selection": "best"}),
                                               read_checkpoints=False)
    rows = {line.split(" & ")[0]: line for line in latex.splitlines() if " & " in line}
    assert "2.00M / 27.70M (+205.9M frozen)" in rows["frozen"]
    assert "frozen" not in rows["long"]
    assert rows["short8"].endswith("& $0.13$ \\\\") and rows["long"].endswith("& $1.4$ \\\\")
    assert "0.13 / " in markdown and "frozen" in markdown.split("Selection rule", 1)[1]


def test_s1_panel_b_names_the_head_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           trained: Dict[str, Any]) -> None:
    import matplotlib.axes

    from vfe3.paper import figures as pf

    labels: List[str] = []
    original = matplotlib.axes.Axes.set_ylabel

    def record(self: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        labels.append(str(label))
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", record)
    run = load_run(write_trained_run(tmp_path, "new", trained, weights=True, checkpoint=True, snapshots=True,
                                     residual=True, metrics={**_new_metrics(), "lr_sigma": [float("nan")] * 5}))
    pf.plot_numerics(run, tmp_path / "fig")
    scale = [label for label in labels if "common scale" in label]
    assert scale and all("M_h" in label and "O_hT_h" not in label for label in scale)


# ---- A365 (paper-claims-10): paper_artifacts.py beside a JSON manifest ---------------------------

def test_paper_artifacts_takes_no_run_key_from_config_beside_a_json_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import paper_artifacts
    import vfe3.paper

    seen: Dict[str, Any] = {}

    def fake_assemble(config: Dict[str, Any], *, out_dir: Any, log: Any, origin: Any = None) -> Dict[str, Any]:
        seen.update(config=dict(config), origin=dict(origin or {}))
        return {"tables": {}, "figures": {}, "out_dir": str(out_dir), "problems": []}

    monkeypatch.setattr(vfe3.paper, "assemble", fake_assemble)
    monkeypatch.setitem(paper_artifacts.CONFIG, "final_run", "D:/elsewhere/final_run")
    monkeypatch.setitem(paper_artifacts.CONFIG, "numerics_run", "D:/elsewhere/numerics_run")
    monkeypatch.setitem(paper_artifacts.CONFIG, "out_dir", str(tmp_path / "out"))
    path = tmp_path / "m2_manifest.json"
    path.write_text(json.dumps({"arms": {"A": []}, "evaluated_run": None, "selection": "best"}), encoding="utf-8")
    assert paper_artifacts.main(["paper_artifacts.py", str(path)]) == 0
    config, origin = seen["config"], seen["origin"]
    assert "final_run" not in config and "numerics_run" not in config and "scaling" not in config
    assert config["arms"] == {"A": []} and config["out_dir"] == str(tmp_path / "out")
    assert config["read_checkpoints"] == paper_artifacts.CONFIG["read_checkpoints"]
    assert origin["arms"] == "manifest" and origin["read_checkpoints"] == "CONFIG"
    assert origin["final_run"].startswith("absent from the manifest")
    # Without a manifest file every key is CONFIG's.
    assert paper_artifacts.main(["paper_artifacts.py"]) == 0
    assert seen["config"]["final_run"] == "D:/elsewhere/final_run"
    assert set(seen["origin"].values()) == {"CONFIG"}


def test_assemble_records_the_key_origin(tmp_path: Path) -> None:
    from vfe3.paper.assemble import assemble, manifest_config

    config, origin = manifest_config({"arms": {"X": []}, "final_run": "D:/x", "read_checkpoints": False},
                                     {"arms": {"A": []}, "selection": "final"})
    assert config == {"arms": {"A": []}, "selection": "final", "read_checkpoints": False}
    report = assemble(config, out_dir=tmp_path / "out", log=lambda message: None, origin=origin)
    written = json.loads((tmp_path / "out" / "paper_artifacts.json").read_text(encoding="utf-8"))
    assert written["config_origin"] == report["config_origin"] == origin
    assert written["config_origin"]["read_checkpoints"] == "CONFIG"
    index = (tmp_path / "out" / "PAPER_ARTIFACTS.md").read_text(encoding="utf-8")
    assert "read_checkpoints" in index and "final_run" in index
