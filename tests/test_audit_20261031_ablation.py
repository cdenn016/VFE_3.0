r"""Audit 2026-10-31, remediation package ``ablation``: the ablation runner (``ablation.py``).

A037  a locked sweep view (sweep_results.csv, cell_attempts.json) no longer ends the sweep; the terminal
      publication waits out the lock, then stays strict.
A038  the collector leaves out rows trained under an earlier ``requires`` regime; the CSV carries the overrides.
A040  the noise-floor line prints the t-based half-width of a single-seed difference, not a two-SD rule.
A106  an environment failure inside a cell (OSError, CUDA out of memory) is a contract failure, retried on resume.
A107  a complete sweep of a second comparison cohort is analyzed and rendered; only the cross-sweep outputs are
      restricted to one cohort.
A108  startup validation builds the executed cell configuration at the run's horizon.
A207  the learning-rate width transfer trains one shared multiplier-one cell per width.
A208  ``CONFIG["save_model_artifacts"]`` reaches ``run_sweep``.
A346  startup prints the explicit baseline settings no active path reads.
A347  best by seed-mean CE, the noise floor from admitted rows, sensitivity bars only with a paired SE, and a
      parabolic refinement of each width's optimum.
A348  projected-gradient sweeps pin their routing, the exact-route sweeps run at depth >= 2 with one memory kernel
      path for the rank sweep, and a component forest without a component arm is refused when scheduled.

Synthetic fixtures only: every cell is stubbed (no model is trained), and configuration comes from tests/profiles.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import pytest
import torch

import ablation
import vfe3.run_artifacts as run_artifacts
from tests.profiles import config_values


_FIXED_CODE = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}
_RUN: Dict[str, Any] = dict(dataset="wikitext-103", device=torch.device("cpu"), seed=6)
_TINY_RUN = dict(max_seq_len=8, max_steps=2, warmup_steps=1)


def _fake_source(dataset: str, split: str, *, cache_dir: Any = None) -> Dict[str, Any]:
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def _stub_cells(
    monkeypatch: pytest.MonkeyPatch,
    sweeps:      Dict[str, Dict[str, Any]],

    *,
    hook:        Optional[Callable[[str, Dict[str, Any], Path, Dict[str, Any]], None]] = None,
    baseline:    Optional[Dict[str, Any]] = None,
    ppl:         Optional[Callable[[str, int], float]] = None,
) -> List[str]:
    r"""Isolate ablation from the click-to-run state and stub ``run_single`` with a successful cell.

    ``hook(label, overrides, run_dir, kwargs)`` runs first and may raise; ``ppl(label, seed)`` sets the PPL."""
    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_FIXED_CODE))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)       # the driver switches are the user's
    monkeypatch.setattr(ablation, "principled_exact_route", False)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", baseline if baseline is not None else config_values(**_TINY_RUN))
    monkeypatch.setattr(ablation, "SWEEPS", sweeps)
    monkeypatch.setattr(ablation, "DATA_SEED", 3)
    calls: List[str] = []

    def fake_run_single(label: str, overrides: Dict[str, Any], run_dir: Path, **kwargs: Any) -> Dict[str, Any]:
        calls.append(label)
        if hook is not None:
            hook(label, overrides, run_dir, kwargs)
        run_dir.mkdir(parents=True, exist_ok=True)
        save = kwargs.get("save_model_artifacts", True)
        value = 10.0 if ppl is None else float(ppl(label, kwargs["seed"]))
        result: Dict[str, Any] = {
            "label": label, "error_kind": None, "seed": kwargs["seed"], "overrides": ablation._jsonable(overrides),
            "primary_val_ppl": value, "final_val_ppl": value, "save_model_artifacts": save,
            "_loaded_data_sources": {s: _fake_source("wikitext-103", s) for s in ("train", "validation")},
            "terminal_checkpoint": None,
        }
        if save:
            checkpoint = run_dir / "checkpoints" / "terminal.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"checkpoint:{label}".encode("utf-8"))
            result["terminal_checkpoint"] = str(checkpoint)
        return result

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    return calls


def _lock(monkeypatch: pytest.MonkeyPatch, failures: Dict[str, int]) -> Dict[str, int]:
    r"""Make the atomic replace of ``name`` fail ``failures[name]`` more times, as a Windows open-handle lock does.

    The mapping is live: a test edits it to lock or release a file mid-sweep (``math.inf`` locks until released)."""
    original = run_artifacts._atomic_replace

    def replace(final: Path, tmp: Path, **kwargs: Any) -> None:
        name = Path(final).name
        if failures.get(name, 0) > 0:
            failures[name] -= 1
            raise PermissionError(13, "the file is held open by another process", str(final))
        original(final, tmp, **kwargs)

    monkeypatch.setattr(ablation, "_atomic_replace", replace)          # _write_sweep_csv
    monkeypatch.setattr(run_artifacts, "_atomic_replace", replace)     # _write_json_atomic (ledger, metadata)
    return failures


def _meta(sweep_dir: Path) -> Dict[str, Any]:
    return json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))


_LAMBDA_SWEEP = {"description": "x", "param": "lambda_beta", "values": [0.25, 0.5, 0.75]}
_LAMBDA_CELLS = ["lambda_beta=0.25", "lambda_beta=0.5", "lambda_beta=0.75", "lambda_beta=1.0"]   # + the base cell


# ---- A037: the sweep view is refreshed best effort; the terminal publication waits, then stays strict ---------

def test_a_locked_csv_view_does_not_end_the_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failures: Dict[str, int] = {}

    def hook(label: str, overrides: Dict[str, Any], run_dir: Path, kwargs: Dict[str, Any]) -> None:
        if label == _LAMBDA_CELLS[0]:
            failures["sweep_results.csv"] = math.inf                   # a reader opens the CSV during cell 1
        if label == _LAMBDA_CELLS[-1]:
            failures["sweep_results.csv"] = 0                          # and closes it during the last cell

    calls = _stub_cells(monkeypatch, {"probe": dict(_LAMBDA_SWEEP)}, hook=hook)
    _lock(monkeypatch, failures)
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    assert calls == _LAMBDA_CELLS                                      # every requested cell was attempted
    assert _meta(tmp_path / "probe")["status"] == "complete"
    assert [row["label"] for row in ablation._read_sweep_csv(tmp_path / "probe")] == sorted(_LAMBDA_CELLS)


def test_a_locked_ledger_is_republished_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    calls = _stub_cells(monkeypatch, {"probe": dict(_LAMBDA_SWEEP)})
    monkeypatch.setattr(ablation, "_TERMINAL_PUBLICATION_RETRY_SECONDS", 0.0)
    # Four per-cell refreshes and the first terminal attempt fail; the terminal backoff then succeeds.
    _lock(monkeypatch, {ablation._CELL_ATTEMPTS_FILENAME: len(_LAMBDA_CELLS) + 1})
    with caplog.at_level(logging.WARNING, logger="ablation"):
        ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    assert calls == _LAMBDA_CELLS
    ledger = json.loads((tmp_path / "probe" / ablation._CELL_ATTEMPTS_FILENAME).read_text(encoding="utf-8"))
    assert sorted(ledger) == sorted(_LAMBDA_CELLS)
    assert _meta(tmp_path / "probe")["status"] == "complete"
    assert "not refreshed" in caplog.text and "is locked" in caplog.text


def test_a_lock_held_through_the_end_raises_after_the_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_cells(monkeypatch, {"probe": dict(_LAMBDA_SWEEP)})
    monkeypatch.setattr(ablation, "_TERMINAL_PUBLICATION_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(ablation, "_TERMINAL_PUBLICATION_RETRY_SECONDS", 0.02)
    _lock(monkeypatch, {"sweep_results.csv": math.inf})
    with pytest.raises(PermissionError):
        ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    assert calls == _LAMBDA_CELLS
    assert _meta(tmp_path / "probe")["status"] == "running"           # never reported complete over a stale view


@pytest.mark.skipif(os.name != "nt", reason="an open handle blocks os.replace only on Windows")
def test_a_csv_held_open_on_windows_does_not_end_the_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handles: List[Any] = []

    def hook(label: str, overrides: Dict[str, Any], run_dir: Path, kwargs: Dict[str, Any]) -> None:
        if label == _LAMBDA_CELLS[0]:
            handles.append(open(tmp_path / "probe" / "sweep_results.csv", encoding="utf-8"))
        if label == _LAMBDA_CELLS[-1]:
            handles.pop().close()

    calls = _stub_cells(monkeypatch, {"probe": dict(_LAMBDA_SWEEP)}, hook=hook)
    try:
        ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    finally:
        for handle in handles:
            handle.close()
    assert calls == _LAMBDA_CELLS
    assert _meta(tmp_path / "probe")["status"] == "complete"


# ---- A038: rows from an earlier requires regime are left out; the CSV carries the overrides --------------------

def test_rows_from_an_earlier_requires_regime_are_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    sweeps = {"probe": dict(_LAMBDA_SWEEP)}
    calls = _stub_cells(monkeypatch, sweeps)
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    # The sweep acquires a requirement; two earlier values leave the list but stay on disk (the tack-on design).
    sweeps["probe"] = {"description": "x", "param": "lambda_beta", "values": [0.25], "requires": {"weight_decay": 0.01}}
    with caplog.at_level(logging.WARNING, logger="ablation"):
        ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    assert calls == _LAMBDA_CELLS + ["lambda_beta=0.25", "lambda_beta=1.0"]   # the contract change re-ran both
    admitted = {row["label"] for row in ablation._collect_sweep_results(tmp_path / "probe")}
    assert admitted == {"lambda_beta=0.25", "lambda_beta=1.0"}
    rows = ablation._read_sweep_csv(tmp_path / "probe")
    assert {row["label"] for row in rows} == admitted
    assert all(json.loads(row["overrides"])["weight_decay"] == 0.01 for row in rows)
    assert "lambda_beta=0.5" in caplog.text and "earlier 'requires' regime" in caplog.text
    assert _meta(tmp_path / "probe")["status"] == "complete"


def test_tack_on_rows_of_the_current_regime_stay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweeps = {"probe": {"description": "x", "param": "lambda_beta", "values": [0.25, 0.5]}}
    calls = _stub_cells(monkeypatch, sweeps)
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    # A requirement at the baseline's own value changes no cell: rows whose overrides lack the key stay.
    sweeps["probe"] = {"description": "x", "param": "lambda_beta", "values": [0.25],
                       "requires": {"weight_decay": config_values()["weight_decay"]}}
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    assert calls == ["lambda_beta=0.25", "lambda_beta=0.5", "lambda_beta=1.0"]   # nothing re-ran
    admitted = {row["label"] for row in ablation._collect_sweep_results(tmp_path / "probe")}
    assert admitted == {"lambda_beta=0.25", "lambda_beta=0.5", "lambda_beta=1.0"}


def test_an_arm_may_override_a_requirement_but_a_retired_arm_must_meet_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweeps = {"probe": {"description": "x", "requires": {"weight_decay": 0.01}, "configs": [
        {"label": "base"},
        {"label": "wd_arm", "weight_decay": 0.02},                   # the arm replaces the required value
        {"label": "old_arm", "lambda_beta": 0.5},                    # trained at the required 0.01
        {"label": "old_wd_arm", "lambda_beta": 0.75, "weight_decay": 0.03},
    ]}}
    _stub_cells(monkeypatch, sweeps)
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    sweeps["probe"] = {"description": "x", "requires": {"weight_decay": 0.03}, "configs": [
        {"label": "base"}, {"label": "wd_arm", "weight_decay": 0.02}]}
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    admitted = {row["label"] for row in ablation._collect_sweep_results(tmp_path / "probe")}
    # wd_arm keeps its own value under the new requirement; old_arm was trained under the old one; old_wd_arm
    # (retired, but at 0.03) meets the current requirement.
    assert admitted == {"base", "wd_arm", "old_wd_arm"}


# ---- A040: the noise-floor line prints the t-based half-width -------------------------------------------------

def test_noise_floor_line_prints_the_t_based_half_width(capsys: pytest.CaptureFixture[str]) -> None:
    import multiseed_analysis

    floor = {"seeds": [6, 23, 54], "n": 3, "mean_ce": 4.6, "sd_ce": 0.01, "sd_ppl": 1.0}
    ablation._print_noise_floor(floor)
    out = capsys.readouterr().out
    half = multiseed_analysis._student_t_quantile_975(2) * math.sqrt(2.0) * 0.01
    assert half == pytest.approx(4.302652729749462 * math.sqrt(2.0) * 0.01, rel=1e-12)
    assert f"+/-{half:.4f} nats" in out and "t_{2,0.975} sqrt(2) SD" in out
    assert "twice" not in out


# ---- A106: environment failures are contract failures, retried on resume ---------------------------------------

def test_an_environment_failure_is_retried_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    attempts = {"bad": 0}

    def hook(label: str, overrides: Dict[str, Any], run_dir: Path, kwargs: Dict[str, Any]) -> None:
        if label == "bad":
            attempts["bad"] += 1
            if attempts["bad"] == 1:
                raise FileNotFoundError("tokenized cache missing: wikitext-103/train.pt")

    sweeps = {"probe": {"description": "x", "configs": [{"label": "baseline"}, {"label": "bad", "lambda_beta": 0.5}]}}
    calls = _stub_cells(monkeypatch, sweeps, hook=hook)
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)
    marker = json.loads((tmp_path / "probe" / ablation._sanitize("bad") / "ablation_result.json")
                        .read_text(encoding="utf-8"))
    assert (marker["status"], marker["error_kind"], marker["failure_class"]) == ("failed", "environment", "contract")
    assert _meta(tmp_path / "probe")["status"] == "incomplete"
    ablation.run_sweep("probe", tmp_path, resume=True, **_RUN)     # retry_failed stays False
    assert calls == ["baseline", "bad", "bad"]
    assert "earlier attempt failed (environment) -> retrying" in capsys.readouterr().out
    assert _meta(tmp_path / "probe")["status"] == "complete"


def test_environment_failure_classifier() -> None:
    assert ablation._is_environment_failure(FileNotFoundError("cache missing"))
    assert ablation._is_environment_failure(PermissionError(13, "locked"))
    assert ablation._is_environment_failure(OSError(28, "No space left on device"))
    assert ablation._is_environment_failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None:
        assert ablation._is_environment_failure(cuda_oom("CUDA out of memory"))
    assert not ablation._is_environment_failure(RuntimeError("synthetic divergence"))
    assert not ablation._is_environment_failure(FloatingPointError("loss became NaN"))
    assert not ablation._is_environment_failure(ValueError("bad shape"))


# ---- A107, A346: a second-cohort sweep is analyzed and rendered; startup reports inactive settings -------------

def _main_config(tmp_path: Path, **overrides: Any) -> Dict[str, Any]:
    return {**ablation.CONFIG, "list_only": False, "sweep": None, "device": "cpu", "dataset": "wikitext-103",
            "output_dir": str(tmp_path), "seed": 6, "resume": True, "retry_failed": False, "max_tokens": None,
            "max_steps": None, "stop_after_steps": None, "save_model_artifacts": True, **overrides}


def test_a_complete_sweep_in_a_second_cohort_is_analyzed_and_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    sweeps = {"single": {"description": "one seed", "param": "lambda_beta", "values": [0.5]},
              "multi": {"description": "two seeds", "param": "lambda_beta", "values": [0.5], "seeds": [6, 7]}}
    _stub_cells(monkeypatch, sweeps)
    monkeypatch.setattr(ablation, "SWEEP_ORDER", ["single", "multi"])
    monkeypatch.setattr(ablation, "CONFIG", _main_config(tmp_path))
    monkeypatch.setitem(ablation.LR_WIDTH_TRANSFER, "enabled", False)
    monkeypatch.setattr(ablation, "_print_baseline_default_divergence", lambda: None)
    figures: List[Any] = []

    def figure_process(output_dir: Path, *, scope: str, invalidate: bool = False,
                       cohort_identity: Optional[Dict[str, Any]] = None) -> bool:
        figures.append((scope, invalidate, cohort_identity))
        return True

    monkeypatch.setattr(ablation, "_run_ablation_figures_isolated", figure_process)
    analyzed: List[str] = []
    analyze = ablation.analyze_sweep
    monkeypatch.setattr(ablation, "analyze_sweep", lambda d: (analyzed.append(d.name), analyze(d))[1])
    assert ablation.main() == 0
    assert analyzed == ["single", "multi"]
    assert [scope for scope, invalidate, _ in figures if not invalidate] == ["single", "multi", "__sensitivity__"]
    cohort = ablation._cross_sweep_cohort_identity(_meta(tmp_path / "single")["aggregation_contract"])
    assert [identity for scope, invalidate, identity in figures if scope == "__sensitivity__" and not invalidate] \
        == [cohort]
    status = json.loads((tmp_path / "ablation_run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["incomplete_sweeps"] == []
    out = capsys.readouterr().out
    best = out.split("BEST PER SWEEP")[1]
    assert "single" in best and "multi" not in best                  # the summary keeps the first cohort only
    assert "BASELINE_CONFIG settings read by no active path" in out    # A346: printed at startup


def test_inactive_report_is_restricted_to_explicit_baseline_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = config_values(**_TINY_RUN, rope_mode="none")
    del baseline["rope_fraction"]                                      # omitted: inert too, but not a baseline key
    _stub_cells(monkeypatch, {}, baseline=baseline)
    monkeypatch.setattr(ablation, "CONFIG", {**ablation.CONFIG, "seed": 6, "max_steps": None})
    ablation._print_inactive_baseline_settings()
    out = capsys.readouterr().out
    assert "rope_base=100.0: only read by a rope_mode other than 'none'" in out
    assert "rope_fraction" not in out


# ---- A108: startup validation builds the executed cell at the run's horizon -----------------------------------

def test_startup_validation_builds_the_executed_cell_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cells(monkeypatch, {"probe": {"description": "x", "param": "lambda_beta", "values": [0.5]}},
                baseline=config_values(max_steps=15000, warmup_steps=250))
    ablation.validate_sweeps(["probe"])                               # the baseline's own horizon constructs
    for horizon in (100, {"probe": 100}):
        with pytest.raises(ValueError, match=r"warmup_steps \(250\) must not exceed max_steps \(100\)"):
            ablation.validate_sweeps(["probe"], max_steps=horizon)
    monkeypatch.setattr(ablation, "CONFIG", {**ablation.CONFIG, "list_only": False, "sweep": "probe",
                                             "device": "cpu", "max_steps": 100})
    monkeypatch.setitem(ablation.LR_WIDTH_TRANSFER, "enabled", False)
    with pytest.raises(ValueError, match="warmup_steps"):
        ablation.main()                                                # refused before any cell trains


def test_validation_applies_the_cell_stride_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""run_single drops a stride longer than an arm's context to None; validation now builds the same cell."""
    _stub_cells(monkeypatch, {"probe": {"description": "x", "param": "max_seq_len", "values": [4, 8]}},
                baseline=config_values(max_seq_len=8, eval_stride=6))
    ablation.validate_sweeps(["probe"])


# ---- A207: one shared multiplier-one cell per width ------------------------------------------------------------

def test_width_transfer_trains_one_multiplier_one_cell_per_width() -> None:
    settings = {"widths": [(8, 2), (16, 4)], "lr_fields": ["m_phi_lr", "m_block_mlp_lr"],
                "multipliers": [0.5, 1.0, 2.0], "max_steps": 10, "enabled": True}
    arms = ablation._lr_width_transfer_arms(settings, {"m_phi_lr": 0.004, "m_block_mlp_lr": 0.0025})
    labels = [arm["label"] for arm in arms]
    assert len(arms) == 2 * (1 + 2 * 2)                                # per width: x1 once, 2 rates x 2 multipliers
    assert len(set(labels)) == len(labels) and not any(label.endswith("=x1") for label in labels)
    assert {"label": "K=8,H=2,baseline", "embed_dim": 8, "n_heads": 2} in arms
    configurations = {json.dumps({k: v for k, v in arm.items() if k != "label"}, sort_keys=True) for arm in arms}
    assert len(configurations) == len(arms)                           # no width carries copies of one cell


def test_width_transfer_fit_reads_the_shared_cell_for_every_rate() -> None:
    widths, fields = [(20, 2), (40, 4)], ("m_phi_lr", "m_block_mlp_lr")
    rows: List[Dict[str, Any]] = []
    for k, h in widths:
        rows.append({"label": ablation._lr_width_transfer_base_label(k, h), "final_val_ppl": 40.0, "error": ""})
        for field in fields:
            rows += [{"label": ablation._lr_width_transfer_label(k, h, field, m), "final_val_ppl": 50.0, "error": ""}
                     for m in (0.5, 2.0)]
    baseline = {"m_phi_lr": 0.004, "m_block_mlp_lr": 0.0025}
    fit = ablation.fit_lr_width_transfer(rows, baseline)["fields"]
    for field in fields:
        for entry in fit[field]["widths"]:
            assert entry["ppl_by_multiplier"] == {"0.5": 50.0, "1": 40.0, "2": 50.0}
            assert entry["best_multiplier"] == 1.0 and not entry["at_grid_edge"]
    # A requested shared cell with no row is diverged at x1 for every rate at that width.
    missing = ablation._lr_width_transfer_base_label(40, 4)
    requested = [row["label"] for row in rows]
    fit = ablation.fit_lr_width_transfer([row for row in rows if row["label"] != missing], baseline,
                                         requested)["fields"]
    for field in fields:
        assert fit[field]["widths"][1]["diverged_multipliers"] == [1.0]


# ---- A208: CONFIG['save_model_artifacts'] reaches run_sweep ------------------------------------------------------

@pytest.mark.parametrize("configured,expected", [(False, False), (True, True), (None, True)])
def test_config_save_model_artifacts_reaches_run_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: Optional[bool], expected: bool,
) -> None:
    captured: List[Any] = []

    def fake_run_sweep(name: str, output_dir: Path, **kwargs: Any) -> list:
        captured.append(kwargs["save_model_artifacts"])
        directory = output_dir / ablation._sweep_output_scope(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sweep_meta.json").write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
        return []

    config = _main_config(tmp_path, sweep="probe")
    if configured is None:
        del config["save_model_artifacts"]                            # an older CONFIG without the key
    else:
        config["save_model_artifacts"] = configured
    _stub_cells(monkeypatch, {"probe": {"description": "x", "param": "lambda_beta", "values": [0.5]}})
    monkeypatch.setattr(ablation, "CONFIG", config)
    monkeypatch.setitem(ablation.LR_WIDTH_TRANSFER, "enabled", False)
    monkeypatch.setattr(ablation, "validate_sweeps", lambda *a, **k: None)
    monkeypatch.setattr(ablation, "_print_baseline_default_divergence", lambda: None)
    monkeypatch.setattr(ablation, "_run_ablation_figures_isolated", lambda *a, **k: True)
    monkeypatch.setattr(ablation, "run_sweep", fake_run_sweep)
    assert ablation.main() == 1
    assert captured == [expected]


# ---- A347: CE ranking, the floor from admitted rows, bars with a paired SE, refined optima ----------------------

def test_best_is_ranked_by_seed_mean_ce() -> None:
    rows = [{"label": f"armA__s{s}", "seed": s, "primary_val_ppl": v} for s, v in ((6, 10.0), (23, 1000.0))]
    rows += [{"label": f"armB__s{s}", "seed": s, "primary_val_ppl": v} for s, v in ((6, 200.0), (23, 200.0))]
    best = ablation._best_by_seed_mean(rows)                          # mean PPL ranks armB (200 < 505)
    assert best["label"] == "armA"                                     # mean CE ranks armA (log 100 < log 200)
    assert best["mean_ce"] == pytest.approx(math.log(100.0)) and best["geomean_ppl"] == pytest.approx(100.0)
    assert best["mean"] == pytest.approx(505.0)


def test_noise_floor_reads_only_admitted_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for seed, ppl in ((6, 100.0), (23, 102.0)):                        # raw markers without any contract
        cell = tmp_path / "a" / f"baseline__s{seed}"
        cell.mkdir(parents=True)
        (cell / "ablation_result.json").write_text(json.dumps(
            {"status": "success", "seed": seed, "overrides": {}, "primary_val_ppl": ppl}), encoding="utf-8")
    assert ablation._baseline_noise_floor([tmp_path / "a"]) is None    # the collector admits none of them
    admitted = [{"status": "success", "seed": seed, "overrides": {}, "primary_val_ppl": ppl}
                for seed, ppl in ((6, 100.0), (23, 102.0))]
    monkeypatch.setattr(ablation, "_collect_sweep_results", lambda d, **_: admitted)
    floor = ablation._baseline_noise_floor([tmp_path / "a"])
    assert floor is not None and floor["seeds"] == [6, 23]


def test_sensitivity_bars_need_a_paired_se() -> None:
    rows = [{"sweep": "seeded", "difference": -0.02, "se": 0.005, "n": 3},
            {"sweep": "single", "difference": -0.05, "se": None, "n": 1},
            {"sweep": "no_reference", "difference": None, "se": None, "n": 0}]
    bars, no_reference, single_seed = ablation._sensitivity_bars(rows)
    assert [row["sweep"] for row in bars] == ["seeded"]
    assert no_reference == ["no_reference"] and single_seed == ["single"]


def test_parabolic_refinement_recovers_an_off_grid_exponent() -> None:
    r"""CE exactly quadratic in log rate, optimum 1.2 (K/20)^-0.7 between grid points: the grid optima are x1, x1,
    x0.5 (a slope of -0.5); the three-point vertices are exact, so the fit recovers -0.7."""
    widths, grid, exponent = [(20, 2), (40, 4), (80, 8)], (0.25, 0.5, 1.0, 2.0, 4.0), -0.7
    rows = []
    for k, h in widths:
        optimum = math.log(1.2 * (k / 20) ** exponent)
        rows += [{"label": ablation._lr_width_transfer_label(k, h, "m_phi_lr", m), "error": "",
                  "final_val_ppl": math.exp(4.0 + 0.5 * (math.log(m) - optimum) ** 2)} for m in grid]
    fit = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]
    assert [entry["best_multiplier"] for entry in fit["widths"]] == [1.0, 1.0, 0.5]
    for (k, _h), entry in zip(widths, fit["widths"]):
        assert entry["refined_multiplier"] == pytest.approx(1.2 * (k / 20) ** exponent, rel=1e-9)
        assert entry["refined_lr"] == pytest.approx(0.004 * entry["refined_multiplier"], rel=1e-12)
    assert fit["exponent"] == pytest.approx(exponent, abs=1e-9)


def test_parabolic_refinement_stays_within_half_a_grid_step() -> None:
    grid = [(0.5, 60.0), (1.0, 50.0), (2.0, 50.0), (4.0, 70.0)]       # a flat pair: the vertex is their midpoint
    assert ablation._parabolic_log_optimum(grid, 1.0) == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert ablation._parabolic_log_optimum(grid, 0.5) is None          # a grid edge has no bracket
    diverged = [(0.5, float("inf")), (1.0, 50.0), (2.0, 55.0)]
    assert ablation._parabolic_log_optimum(diverged, 1.0) is None      # a diverged neighbor gives no curvature


# ---- A348: registry entries that the baseline refused, and a forest without a component arm ---------------------

_PROJECTED_GRADIENT_SWEEPS = ("e_q_mu_lr", "e_q_sigma_lr", "lambda_beta", "sigma_max",
                              "relational_receiver_noise", "relational_receiver_damping")


def test_projected_gradient_sweeps_pin_their_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROJECTED_GRADIENT_SWEEPS:
        requires = ablation.SWEEPS[name]["requires"]
        assert requires["attention_mode"] == "projected_gradient" and requires["relational_routing"] == "gaussian_kl"
    # A predictive baseline on noisy_key routing: projected_gradient admits only gaussian_kl.
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(
        attention_mode="projected_predictive", relational_routing="noisy_key",
        relational_receiver_measurement="inverse", relational_receiver_update="closed_form"))
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    monkeypatch.setattr(ablation, "principled_exact_route", False)
    ablation.validate_sweeps(list(_PROJECTED_GRADIENT_SWEEPS))


def test_exact_route_requirements_floor_the_depth() -> None:
    assert ablation._exact_route_requires(config_values(n_layers=1))["n_layers"] == 2
    assert ablation._exact_route_requires(config_values(n_layers=4))["n_layers"] == 4
    for name, sweep in ablation.SWEEPS.items():
        if name.startswith("exact_route"):
            assert sweep["requires"]["n_layers"] >= 2, name
    rank = ablation.SWEEPS["exact_route_regression_rank"]
    assert max(rank["values"]) > 4 and rank["requires"]["block_memory_triton"] is False


def test_component_forest_needs_a_component_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(**_TINY_RUN))
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    monkeypatch.setattr(ablation, "principled_exact_route", False)
    forest = dict(ablation.SWEEPS["component_ablation_forest"])
    one_arm = {**forest, "configs": [{"label": "baseline"}]}
    monkeypatch.setitem(ablation.SWEEPS, "component_ablation_forest", one_arm)
    with pytest.raises(ValueError, match="needs its baseline arm 'baseline' and at least one component arm"):
        ablation.validate_sweeps(["component_ablation_forest"])
    ablation.validate_sweeps(["component_ablation_forest"], require_construction=False)   # advisory only
    with_arm = {**forest, "configs": [{"label": "baseline"}, {"label": "half_coupling", "lambda_beta": 0.5}]}
    monkeypatch.setitem(ablation.SWEEPS, "component_ablation_forest", with_arm)
    ablation.validate_sweeps(["component_ablation_forest"])


# ---- A276: the corrected runner documentation ------------------------------------------------------------------

def test_runner_documentation_states_the_executable_state() -> None:
    doc = ablation.__doc__ or ""
    assert "at the bottom" not in doc and "K=20" not in doc and "run-for-run" not in doc
    for name in ("m_pos_phi_lr", "m_layernorm_lr"):
        description = ablation.SWEEPS[name]["description"]
        assert "arm is the" not in description and "None inherits" in description
    assert "K=20" not in (ablation._cell_diagnostics.__doc__ or "")
    assert "independent replicates" not in (ablation._lr_width_transfer_arms.__doc__ or "")
