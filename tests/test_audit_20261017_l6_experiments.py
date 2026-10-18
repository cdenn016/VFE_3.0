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


# ---- item 28: failed cells are terminal under resume; attempts are archived and recorded ---------

_FIXED_CODE = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}


def _fake_source(dataset: str, split: str, *, cache_dir: Any = None) -> Dict[str, Any]:
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def _stub_ablation(monkeypatch: pytest.MonkeyPatch, crash: Dict[str, int],
                   configs: Any = None) -> Dict[str, int]:
    import ablation
    from tests.profiles import config_values

    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_FIXED_CODE))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(max_seq_len=8, max_steps=2, warmup_steps=1))
    monkeypatch.setattr(ablation, "SWEEPS", {"probe": {
        "description": "attempt probe",
        "configs": configs or [{"label": "baseline"}, {"label": "bad", "lambda_beta": 0.5}]}})
    monkeypatch.setattr(ablation, "DATA_SEED", 3)
    calls: Dict[str, int] = {}

    def fake_run_single(label: str, overrides: Dict[str, Any], run_dir: Path, **kwargs: Any) -> Dict[str, Any]:
        calls[label] = calls.get(label, 0) + 1
        if crash.get(label, 0) >= calls[label]:
            raise RuntimeError("synthetic divergence")
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / "checkpoints" / "terminal.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{label}".encode("utf-8"))
        return {"label": label, "error_kind": None, "seed": 6, "overrides": ablation._jsonable(overrides),
                "terminal_checkpoint": str(checkpoint), "primary_val_ppl": 10.0, "final_val_ppl": 10.0,
                "skipped_steps": 3 if label == "bad" else 0,
                "accepted_update_frac": 0.5 if label == "bad" else 1.0,
                "_loaded_data_sources": {s: _fake_source("wikitext-103", s) for s in ("train", "validation")}}

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    return calls


def test_resume_keeps_failed_cell_terminal_and_records_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ablation
    import torch

    calls = _stub_ablation(monkeypatch, crash={"bad": 1})
    kwargs: Dict[str, Any] = dict(dataset="wikitext-103", device=torch.device("cpu"), seed=6)
    ablation.run_sweep("probe", tmp_path, resume=True, **kwargs)
    assert calls == {"baseline": 1, "bad": 1}
    meta = json.loads((tmp_path / "probe" / "sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["cell_attempts"]["bad"]["n_failed"] == 1

    # A second resumed invocation reports the failure and does not retrain either cell.
    ablation.run_sweep("probe", tmp_path, resume=True, **kwargs)
    assert calls == {"baseline": 1, "bad": 1}
    out = capsys.readouterr().out
    assert "FAILED earlier, not retried" in out and "bad: 1 attempt(s), 1 failed" in out
    meta = json.loads((tmp_path / "probe" / "sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "incomplete" and meta["cell_attempts"]["bad"]["last_status"] == "failed"

    # retry_failed retrains it; the failed generation is archived and both attempts are recorded.
    ablation.run_sweep("probe", tmp_path, resume=True, retry_failed=True, **kwargs)
    assert calls == {"baseline": 1, "bad": 2}
    ledger = json.loads((tmp_path / "probe" / "cell_attempts.json").read_text(encoding="utf-8"))
    assert [a["status"] for a in ledger["bad"]] == ["failed", "success"]
    archive = tmp_path / "probe" / ledger["bad"][0]["archive"]
    archived = json.loads((archive / "ablation_result.json").read_text(encoding="utf-8"))
    assert archived["error_kind"] == "train" and archived["failure_class"] == "run"
    meta = json.loads((tmp_path / "probe" / "sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete"
    assert (meta["cell_attempts"]["bad"]["n_attempts"], meta["cell_attempts"]["bad"]["n_failed"]) == (2, 1)


def test_contract_failure_is_not_terminal() -> None:
    import ablation

    marker = {"status": "failed", "error_kind": "code_identity_drift", "final_val_ppl": 10.0,
              "label": "a", "seed": 6, "overrides": {}}
    assert ablation._cell_failure_class(marker) == "contract"
    assert not ablation._is_terminal_cell_failure(marker, "a", 6, {})
    crashed = {**marker, "error_kind": "train", "final_val_ppl": float("inf")}
    assert ablation._is_terminal_cell_failure(crashed, "a", 6, {})
    assert not ablation._is_terminal_cell_failure(crashed, "a", 6, {"lambda_beta": 0.5})


def test_run_vfe_crash_writes_failure_record_and_paper_arms_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import paper_arms
    from tests.profiles import tiny_config
    from vfe3 import experiment

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise FloatingPointError("loss became NaN")

    monkeypatch.setattr(experiment, "_train_and_finalize", boom)
    monkeypatch.setattr(experiment, "_loader", lambda *a, **k: None)
    monkeypatch.setattr("vfe3.data.datasets.tokens_per_char", lambda *a, **k: None)
    with pytest.raises(FloatingPointError):
        experiment.run_vfe(tiny_config(), dataset="wikitext-103", out_root=tmp_path, seed=1, data_seed=3,
                           label="probe_s1_d3", device="cpu")
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    record = experiment.read_failure_record(run_dir)
    assert record is not None and record["error_type"] == "FloatingPointError"
    assert record["error"] == "loss became NaN" and "step" in record
    reason = paper_arms.failed_run_reason(run_dir)
    assert reason is not None and "FloatingPointError" in reason
    # A directory without summary.json or failure.json is still unfinished, not failed.
    (tmp_path / "other").mkdir()
    assert paper_arms.failed_run_reason(tmp_path / "other") is None


class _StopBeforeTraining(Exception):
    pass


def test_scaling_failed_cell_is_terminal_unless_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scaling

    route = next(iter(scaling.ROUTES))
    cell = scaling.ROUTES[route][0]
    run_dir = scaling._trusted_scaling_run_dir(tmp_path, route, cell["label"], 6)
    failure = {"label": cell["label"], "route": route, "seed": 6, "status": "failed",
               "error_kind": "train", "error": "synthetic divergence"}
    for name in ("scaling_result.json", "scaling_failure.json"):
        (run_dir / name).write_text(json.dumps(failure), encoding="utf-8")

    def stop(*args: Any, **kwargs: Any) -> Any:
        raise _StopBeforeTraining

    monkeypatch.setattr(scaling, "predict_n_params", stop)
    monkeypatch.setitem(scaling.CONFIG, "resume", True)
    monkeypatch.setitem(scaling.CONFIG, "retry_failed", False)
    res = scaling.run_cell(cell, run_dir, 6, dataset="wikitext-103", device=None, max_steps=None)
    assert res["terminal_failure"] and res["error_kind"] == "train" and res["preserve_existing"]
    assert scaling._scaling_result_status(res) == "failed"
    history = scaling._scaling_attempt_history(run_dir, 6)
    assert (history["n_attempts"], history["n_failed"]) == (1, 1)

    # retry_failed archives the failed attempt and proceeds to train (stopped here before the model build).
    monkeypatch.setitem(scaling.CONFIG, "retry_failed", True)
    with pytest.raises(_StopBeforeTraining):
        scaling.run_cell(cell, run_dir, 6, dataset="wikitext-103", device=None, max_steps=None)
    history = scaling._scaling_attempt_history(run_dir, 6)
    assert history["n_attempts"] == 1 and history["attempts"][0]["archive"] == "s6__attempt_0001"
    assert history["attempts"][0]["error_kind"] == "train"


def test_scaling_analysis_prints_design_attempts(capsys: pytest.CaptureFixture[str]) -> None:
    import scaling_analysis

    scaling_analysis.print_design_attempts({"cells": [
        {"route": "width", "label": "K20", "seed": 6, "status": "complete", "n_attempts": 2,
         "n_failed_attempts": 1, "attempts": [{"attempt": 1, "status": "failed", "error_kind": "train"},
                                              {"attempt": 2, "status": "complete", "error_kind": None}]},
        {"route": "width", "label": "K40", "seed": 6, "status": "complete", "n_attempts": 1,
         "n_failed_attempts": 0, "attempts": []}]})
    out = capsys.readouterr().out
    assert "width/K20/s6: 2 attempt(s), 1 failed" in out and "K40" not in out


# ---- item 29: realized-update counts reach every comparison table --------------------------------

def test_finalize_validation_run_returns_realized_update_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import vfe3.train as vt
    from tests.test_audit_20261012_round2_recording import _cfg, _loader, _reject_second_step
    from vfe3 import run_artifacts as ra
    from vfe3.model.model import VFEModel

    monkeypatch.setattr(ra, "_git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *a, **k: None)
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    _reject_second_step(model)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model)
    val_loader = _loader(40, shuffle=False)
    returned: Dict[str, Any] = {}

    def finalize(state: Any, losses: List[float]) -> None:
        returned.update(ra.finalize_validation_run(model, artifacts, cfg, val_loader, losses=losses,
                                                   terminal_state=state, device=torch.device("cpu")))

    vt.train(model, _loader(64, shuffle=True), cfg, n_steps=3, val_loader=val_loader, artifacts=artifacts,
             terminal_callback=finalize, generate_samples=False)
    assert returned["skipped_steps"] == 1
    assert returned["accepted_update_frac"] == pytest.approx(2 / 3)


def test_skip_counts_reach_sweep_csv_scaling_points_and_leaderboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ablation
    import parameter_optimization as po
    import scaling_analysis

    ablation._write_sweep_csv(tmp_path, [{"label": "a", "primary_val_ppl": 10.0, "skipped_steps": 3,
                                          "accepted_update_frac": 0.9},
                                         {"label": "b", "primary_val_ppl": 11.0, "skipped_steps": 0,
                                          "accepted_update_frac": 1.0}])
    rows = {r["label"]: r for r in ablation._read_sweep_csv(tmp_path)}
    assert rows["a"]["skipped_steps"] == "3" and rows["a"]["skipped_updates_flag"] == "True"
    assert rows["b"]["skipped_updates_flag"] == "False"
    for r in rows.values():
        r["_ppl"] = float(r["primary_val_ppl"])
    ablation._print_ablation_rows(list(rows.values()), parameter_matched=False)
    out = capsys.readouterr().out
    assert "[UNSTABLE: 3 skipped updates]" in out.splitlines()[2]
    assert {"skipped_steps", "accepted_update_frac", "skipped_updates_flag"} <= set(scaling_analysis._CSV_COLUMNS)

    candidates = [{"label": "c0", "overrides": {}}, {"label": "c1", "overrides": {}}]
    results = [{"label": "c0", "primary_val_ppl": 10.0, "skipped_steps": 5, "accepted_update_frac": 0.5},
               {"label": "c1", "primary_val_ppl": 12.0, "skipped_steps": 0, "accepted_update_frac": 1.0}]
    board = {row["label"]: row for row in po._leaderboard_rows(candidates, results, 1)}
    assert board["c0"]["skipped_updates_flag"] and board["c0"]["max_skipped_steps"] == 5
    assert not board["c1"]["skipped_updates_flag"]
    po._write_leaderboard(tmp_path / "leaderboard.csv", list(board.values()))
    assert "max_skipped_steps" in (tmp_path / "leaderboard.csv").read_text(encoding="utf-8").splitlines()[0]


# ---- item 30: interaction ranking against an additive fit and a permutation null ------------------

def _search(n_params: int) -> Dict[str, Any]:
    return {"parameters": {f"p{k}": {"enabled": True, "bounds": [0.0, 1.0], "scale": "linear"}
                           for k in range(n_params)}}


def _sobol_design(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def _interaction_inputs(units: np.ndarray, y: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = [{"label": f"c{i}", "overrides": {f"p{k}": float(u[k]) for k in range(units.shape[1])}}
                  for i, u in enumerate(units)]
    rows = [{"label": f"c{i}", "status": "complete", "mean_primary_val_ppl": float(v)} for i, v in enumerate(y)]
    return candidates, rows


def test_additive_response_has_no_large_interaction() -> None:
    import parameter_optimization as po

    rng = np.random.default_rng(3)
    units = _sobol_design(64, 8, seed=11)
    beta = rng.normal(0.0, 5.0, size=8)                     # sd 5 PPL per unit coordinate
    y = 100.0 + units @ beta + rng.normal(0.0, 1.0, size=64)
    candidates, rows = _interaction_inputs(units, y)
    interactions = po._interaction_rows(rows, candidates, _search(8), n_permutations=100)
    top = interactions[0]
    # The audit measured a median top-pair RMS of 2.85 PPL here under the old ranking and 0.6 to 0.63 for
    # pure noise; after the all-coordinate additive fit the top pair stays at the noise level.
    assert top["interaction_rms_ppl"] < 1.2
    assert sum(1 for item in interactions if item["exceeds_null"]) <= 4     # of 28 pairs, ~1.4 expected at 5%


def test_true_interaction_exceeds_the_permutation_null() -> None:
    import parameter_optimization as po

    rng = np.random.default_rng(4)
    units = _sobol_design(64, 4, seed=5)
    y = (100.0 + 5.0 * units[:, 2] + 40.0 * (units[:, 0] - 0.5) * (units[:, 1] - 0.5)
         + rng.normal(0.0, 0.5, size=64))
    candidates, rows = _interaction_inputs(units, y)
    top = po._interaction_rows(rows, candidates, _search(4), n_permutations=100)[0]
    assert {top["parameter_x"], top["parameter_y"]} == {"p0", "p1"}
    assert top["exceeds_null"] and top["permutation_p"] < 0.05


# ---- item 52: one run-folder resolver --------------------------------------------------------------

def _run_folder(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def test_run_folder_resolver_finds_unique_names_and_refuses_ambiguity(tmp_path: Path) -> None:
    from vfe3.paper import runs

    archive = tmp_path / "archive"
    moved = _run_folder(archive / "10-10-2026_experiments" / "36.41_wikitext-103_K128")
    roots = [tmp_path / "vfe3_runs", archive]
    found = runs.find_run_folder("vfe3_runs/36.41_wikitext-103_K128", base=tmp_path, roots=roots)
    assert found == moved
    assert runs.find_run_folder(moved, base=tmp_path, roots=roots) == moved           # a valid path wins
    _run_folder(tmp_path / "vfe3_runs" / "old" / "36.41_wikitext-103_K128")
    with pytest.raises(runs.AmbiguousRunFolder):
        runs.find_run_folder("D:/gone/36.41_wikitext-103_K128", base=tmp_path, roots=roots)
    with pytest.raises(runs.RunFolderNotFound):
        runs.find_run_folder("vfe3_runs/no_such_run", base=tmp_path, roots=roots)


def test_drivers_use_the_resolver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import paper_arms
    import train_baseline
    from vfe3.paper import runs

    monkeypatch.setattr(runs, "RUN_SEARCH_ROOTS", (str(tmp_path / "archive"),))
    with pytest.raises(paper_arms.BaseUnavailableError):
        paper_arms.load_base("gone", tmp_path / "missing" / "no_such_base_run")
    with pytest.raises(train_baseline.BaseUnavailableError):
        train_baseline.resolve_base_dir(str(tmp_path / "missing" / "no_such_base_run"))
    moved = _run_folder(tmp_path / "archive" / "dated" / "some_base_run")
    assert train_baseline.resolve_base_dir(str(tmp_path / "missing" / "some_base_run")) == moved
    paths, unresolved = runs.resolve_run_paths([str(tmp_path / "elsewhere" / "some_base_run")])
    assert paths == [moved] and unresolved == []


# ---- items 58, 59, 61: ablation tables and the shared baseline ------------------------------------

def _write_rows(sweep_dir: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    import ablation

    sweep_dir.mkdir(parents=True, exist_ok=True)
    ablation._write_sweep_csv(sweep_dir, rows)
    (sweep_dir / "sweep_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_best_is_picked_on_seed_means() -> None:
    import ablation

    rows = [{"label": f"armA__s{s}", "seed": s, "primary_val_ppl": v} for s, v in ((6, 100.0), (23, 90.0), (54, 110.0))]
    rows += [{"label": f"armB__s{s}", "seed": s, "primary_val_ppl": v} for s, v in ((6, 95.0), (23, 96.0), (54, 97.0))]
    best = ablation._best_by_seed_mean(rows)
    assert best["label"] == "armB" and best["mean"] == pytest.approx(96.0)


def test_paired_differences_on_ce_and_data_order_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import ablation

    rows = [{"label": f"ref__s{s}", "seed": s, "primary_val_ppl": v} for s, v in ((6, 100.0), (23, 100.0))]
    rows += [{"label": f"arm__s{s}", "seed": s, "primary_val_ppl": v} for s, v in ((6, 110.0), (23, 5000.0))]
    (ce,) = ablation._paired_seed_differences(rows, "ref", scale="ce")
    assert ce["mean"] == pytest.approx(0.5 * (math.log(1.1) + math.log(50.0)))
    (ppl,) = ablation._paired_seed_differences(rows, "ref")
    assert ppl["mean"] == pytest.approx(2455.0)
    _write_rows(tmp_path / "relational_features", [
        {"label": f"relational_reference__s{s}", "seed": s, "primary_val_ppl": 100.0 + s / 100} for s in (6, 23, 54)
    ] + [{"label": f"relational_frozen_score__s{s}", "seed": s, "primary_val_ppl": 101.0 + s / 50} for s in (6, 23, 54)],
        {"sweep_name": "relational_features", "aggregation_contract": {"data_seed_override": 3}})
    ablation.analyze_sweep(tmp_path / "relational_features")
    out = capsys.readouterr().out
    assert "shared batch order (data seed 3)" in out
    assert "on CE = log PPL (nats" in out


def test_requires_shown_and_sweeps_grouped_by_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ablation

    monkeypatch.setitem(ablation.SWEEPS, "probe_req", {"description": "x", "param": "m_p_mu_lr",
                                                        "values": [0.01, 0.025],
                                                        "requires": {"m_table_update_mode": "adamw"}})
    monkeypatch.setitem(ablation.SWEEPS, "probe_plain", {"description": "y", "param": "m_phi_lr",
                                                          "values": [0.001, 0.003]})
    _write_rows(tmp_path / "probe_req", [{"label": "m_p_mu_lr=0.025", "seed": 6, "primary_val_ppl": 50.0}],
                {"sweep_name": "probe_req"})
    _write_rows(tmp_path / "probe_plain", [{"label": "m_phi_lr=0.003", "seed": 6, "primary_val_ppl": 60.0}],
                {"sweep_name": "probe_plain"})
    monkeypatch.setattr(ablation, "_sweep_matches_cohort", lambda d, identity: True)
    ablation.summarize_sweeps(tmp_path, cohort_identity={})
    out = capsys.readouterr().out
    assert "another base: on base + m_table_update_mode=adamw" in out
    shared, other = out.index("shared baseline"), out.index("another base")
    assert shared < out.index("probe_plain") < other < out.index("probe_req")
    ablation.analyze_sweep(tmp_path / "probe_req")
    assert "[on base + m_table_update_mode=adamw]" in capsys.readouterr().out


def test_sensitivity_uses_paired_best_minus_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ablation

    monkeypatch.setitem(ablation.SWEEPS, "probe_sens", {"description": "x", "param": "m_phi_lr",
                                                         "values": [0.001, 0.003, 0.009]})
    monkeypatch.setitem(ablation.BASELINE_CONFIG, "m_phi_lr", 0.003)
    rows = []
    for s, (lo, base, hi) in ((6, (99.0, 100.0, 140.0)), (23, (98.0, 100.0, 150.0))):
        rows += [{"label": f"m_phi_lr={v}__s{s}", "seed": s, "primary_val_ppl": p}
                 for v, p in ((0.001, lo), (0.003, base), (0.009, hi))]
    _write_rows(tmp_path / "probe_sens", rows, {"sweep_name": "probe_sens"})
    (row,) = ablation.sensitivity_rows([tmp_path / "probe_sens"])
    assert row["reference"] == "m_phi_lr=0.003" and row["best"] == "m_phi_lr=0.001"
    assert row["difference"] == pytest.approx(0.5 * (math.log(0.99) + math.log(0.98)))
    assert row["n"] == 2 and row["se"] is not None


def test_sweep_without_a_base_cell_gets_one(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    monkeypatch.setitem(ablation.SWEEPS, "probe_nobase", {"description": "x", "param": "phi_scale",
                                                           "values": [0.01, 0.05]})
    monkeypatch.setitem(ablation.BASELINE_CONFIG, "phi_scale", 0.02)
    labels = [label for label, _ in ablation.make_run_overrides("probe_nobase")]
    assert labels == ["phi_scale=0.01", "phi_scale=0.05", "phi_scale=0.02"]
    assert ablation._paired_reference_label("probe_nobase") == "phi_scale=0.02"
    monkeypatch.setitem(ablation.SWEEPS, "probe_hasbase", {"description": "x", "param": "phi_scale",
                                                            "values": [0.01, 0.02]})
    assert len(ablation.make_run_overrides("probe_hasbase")) == 2


def test_cohort_duplicate_cell_is_reused_not_retrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ablation
    import torch

    calls = _stub_ablation(monkeypatch, crash={}, configs=[{"label": "baseline"}, {"label": "arm", "lambda_beta": 0.5}])
    monkeypatch.setitem(ablation.SWEEPS, "probe2", {"description": "second sweep",
                                                     "configs": [{"label": "baseline"},
                                                                 {"label": "arm2", "lambda_beta": 0.25}]})
    kwargs: Dict[str, Any] = dict(dataset="wikitext-103", device=torch.device("cpu"), seed=6, resume=True)
    ablation.run_sweep("probe", tmp_path, **kwargs)
    ablation.run_sweep("probe2", tmp_path, **kwargs)
    assert calls == {"baseline": 1, "arm": 1, "arm2": 1}
    assert "[REUSED] baseline" in capsys.readouterr().out
    marker = json.loads((tmp_path / "probe2" / ablation._sanitize("baseline") / "ablation_result.json")
                        .read_text(encoding="utf-8"))
    assert marker["reused_from"].startswith("probe/") and marker["sweep"] == "probe2"
    assert {r["label"] for r in ablation._collect_sweep_results(tmp_path / "probe2")} == {"baseline", "arm2"}
    meta = json.loads((tmp_path / "probe2" / "sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete"


def test_baseline_noise_floor(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import ablation

    for sweep, seed, ppl in (("a", 6, 100.0), ("b", 6, 100.0), ("a", 23, 102.0), ("b", 54, 98.0)):
        cell = tmp_path / sweep / f"baseline__s{seed}"
        cell.mkdir(parents=True)
        (cell / "ablation_result.json").write_text(json.dumps(
            {"status": "success", "seed": seed, "overrides": {}, "primary_val_ppl": ppl}), encoding="utf-8")
    floor = ablation._baseline_noise_floor([tmp_path / "a", tmp_path / "b"])
    assert floor["seeds"] == [6, 23, 54] and floor["n"] == 3
    ces = [math.log(100.0), math.log(102.0), math.log(98.0)]
    assert floor["sd_ce"] == pytest.approx(float(np.std(ces, ddof=1)))
    ablation._print_noise_floor(floor)
    assert "baseline noise floor: 3 seeds" in capsys.readouterr().out


# ---- item 60: short-horizon learning-rate warning ---------------------------------------------------

def test_short_horizon_warning_names_the_runner_up() -> None:
    import train_baseline

    sweep = {"best_lr": 5e-3, "max_steps": 6000, "runner_up_lr": 7e-3, "runner_up_best_val_ppl": 100.76}
    text = train_baseline.short_horizon_warning(sweep, 150_000)
    assert text is not None and "lr=0.005 won at 6000 steps" in text and "runner-up lr=0.007" in text
    assert train_baseline.short_horizon_warning({**sweep, "max_steps": 150_000}, 150_000) is None


def test_transfer_printout_publishes_only_the_exponent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ablation

    rows = []
    for k, h in ((20, 2), (40, 4), (80, 8)):
        for m in (0.25, 0.5, 1.0, 2.0, 4.0):
            best = {20: 2.0, 40: 1.0, 80: 0.5}[k]
            rows.append({"label": f"K={k},H={h},m_phi_lr=x{m}", "final_val_ppl": 50.0 + abs(math.log(m / best)),
                         "error": "", "skipped_steps": 2 if (k, m) == (20, 4.0) else 0})
    _write_rows(tmp_path / "lr_width_transfer", rows,
                {"sweep_name": "lr_width_transfer", "aggregation_contract": {}})
    monkeypatch.setitem(ablation.BASELINE_CONFIG, "m_phi_lr", 0.003)
    report = ablation.analyze_lr_width_transfer(tmp_path / "lr_width_transfer")
    out = capsys.readouterr().out
    assert "exponent -1.00 at" in out and "lr(K=160)" not in out
    width20 = report["fields"]["m_phi_lr"]["widths"][0]
    assert width20["skipped_update_multipliers"] == [4.0]


# ---- item 10: per-cell router cap and dead-head records ---------------------------------------------

def test_per_head_dim_router_cap_and_dead_heads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scaling

    cell = scaling.route_vary_block_fixed_k(48, [3, 24])
    monkeypatch.setitem(scaling.CONFIG, "kl_max_per_head_dim", None)
    default = [scaling._cell_cfg_dict(c["overrides"], 6, None).get("kl_max") for c in cell]
    assert default == [scaling.BASELINE.get("kl_max")] * 2                  # today's caps are preserved
    monkeypatch.setitem(scaling.CONFIG, "kl_max_per_head_dim", 4.0)
    capped = [scaling._cell_cfg_dict(c["overrides"], 6, None)["kl_max"] for c in cell]
    assert capped == [12.0, 96.0]                                          # 4 nats x d_head (3, 24)
    assert scaling._cell_cfg_dict({**cell[0]["overrides"], "kl_max": 7.0}, 6, None)["kl_max"] == 7.0

    run_dir = tmp_path / "cell"
    assert scaling._dead_router_heads(run_dir)["available"] is False
    (run_dir / "diagnostics").mkdir(parents=True)
    (run_dir / "diagnostics" / "cap_clamped_rows.csv").write_text(
        "step,layer,head,clamped_rows,rows\n"
        "100,0,0,90,100\n100,0,1,0,100\n"
        "200,0,0,10,100\n200,0,1,80,100\n200,1,0,51,100\n", encoding="utf-8")
    dead = scaling._dead_router_heads(run_dir)
    assert dead["available"] and dead["step"] == 200 and dead["heads_recorded"] == 3
    assert dead["dead"] == [[0, 1], [1, 0]] and dead["dead_heads"] == 2
