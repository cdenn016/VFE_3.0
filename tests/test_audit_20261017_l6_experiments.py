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
