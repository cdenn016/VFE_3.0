r"""Manuscript assemblers (:mod:`vfe3.paper`) and the multiseed_analysis additions on synthetic runs.

Every run directory here is written by the test: a tiny projected-predictive model (vocabulary 6,
K = 4, two heads, one layer) supplies real state dicts, optimizer groups and map snapshots, and the
JSON/CSV artifacts are written by hand in the formats the training loop and ``posthoc_eval.py``
produce, including the pre-2026-09-28 layouts (no ``tokens_seen``, no ``resources.json``, a retired
config field) and synthetic counter columns of the training-builder contract.
"""

from __future__ import annotations

from tests.profiles import test_config

import csv
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config, serialize_config
from vfe3.paper import figures as pf
from vfe3.paper import runs as pr
from vfe3.paper import tables as pt
from vfe3.paper.assemble import assemble

V, K = 6, 4
MANUSCRIPT_HEADERS = {
    "main": (r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xccccc@{}}",
             r"Model arm & Seeds & Parameters & Test CE & Test PPL & GPU-hours \\"),
    "per_seed": (r"\begin{tabular}{lccccc}",
                 r"Model arm & Seeds & Selected step & Validation CE & Test CE & Test PPL \\"),
    "relational": (r"\begin{tabular}{lrr}", r"Evaluation & Step & Perplexity \\"),
    "resources": (r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xcc@{}}",
                  r"Parameter role or resource & Evaluated run & Final run \\"),
    "methods": (r"\begin{tabular}{p{0.30\linewidth}p{0.62\linewidth}}",
                r"Reporting field & Final-run value or artifact identifier \\"),
}


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values = dict(vocab_size=V, embed_dim=K, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=1,
                  batch_size=2, attention_mode="projected_predictive", block_mlp_expansion=2,
                  max_steps=20, warmup_steps=2, min_lr_frac=0.1, m_p_sigma_lr=0.01, sigma_weight_decay=0.2,
                  sigma_init=2.0, seed=0)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


@pytest.fixture(scope="module")
def trained() -> Dict[str, Any]:
    r"""A tiny model, its AdamW groups and scheduler, as the training loop builds them."""
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import _optimizer_parameter_manifest
    from vfe3.train import _floor_lr_lambdas, build_optimizer

    torch.manual_seed(0)
    cfg = _cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.normal_(0.3, 0.1)
    optimizer = build_optimizer(model, cfg)
    base = [group["lr"] for group in optimizer.param_groups]
    torch.optim.lr_scheduler.LambdaLR(optimizer, _floor_lr_lambdas(base, cfg))
    state = optimizer.state_dict()
    manifest = _optimizer_parameter_manifest(model, optimizer, state)
    model_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
    return {"cfg": cfg, "state": model_state, "optimizer_state": state, "manifest": manifest}


def _write_metrics(path: Path, columns: Mapping[str, Any]) -> None:
    names = list(columns)
    n = len(next(iter(columns.values())))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for i in range(n):
            row = []
            for name in names:
                value = columns[name][i]
                row.append("" if value is None or (isinstance(value, float) and math.isnan(value)) else value)
            writer.writerow(row)


def _write_run(
    root:       Path,
    name:       str,
    trained:    Mapping[str, Any],

    *,
    seed:       int = 0,
    data_seed:  Optional[int] = 3,
    summary:    Optional[Dict[str, Any]] = None,
    metrics:    Optional[Mapping[str, Any]] = None,
    resources:  Optional[Dict[str, Any]] = None,
    posthoc:    Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    raw_config: Optional[Dict[str, Any]] = None,
    config_extra: Optional[Dict[str, Any]] = None,
    weights:    bool = True,
    checkpoint: bool = False,
    snapshots:  bool = False,
    residual:   bool = False,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    config = serialize_config(trained["cfg"])
    config["seed"] = seed
    config.update(raw_config or {})
    record = {"config": config, "dataset": "synthetic", "n_params": 1000,
              "relational_operators": {"attention_mode": "projected_predictive", "readout_mode": "predictive_content",
                                       "score_rank_per_head": 2, "content_rank_per_head": 2,
                                       "score_initialization": "independent_orthonormal_rows",
                                       "factorization_policy": "zero_jitter_no_covariance_repairs"},
              **(config_extra or {})}
    (run / "config.json").write_text(json.dumps(record), encoding="utf-8")
    prov = {"seed": seed, "data_seed": data_seed, "tokenizer_tag": "tiktoken", "device_name": "cpu",
            "git_sha": "a" * 40, "git_dirty": False, "max_tokens": None,
            "train_data_sha256": "b" * 64, "train_data_n_tokens": 400, "val_data_sha256": "c" * 64,
            "val_data_n_tokens": 50, "test_data_sha256": "d" * 64, "test_data_n_tokens": 60}
    prov.update(provenance or {})
    (run / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    base = {"n_steps": 20, "attempted_steps": 20, "accepted_updates": 20, "skipped_steps": 0,
            "best_val_ppl": math.exp(1.5), "best_step": 10, "evaluation_weights": "final",
            "evaluated_step": 20, "final_val_ce": 1.6, "test_ce": 1.7, "test_ppl": math.exp(1.7),
            "wall_time_s": 7200.0, "n_params": 1000,
            "test_scored_targets": 50, "test_expected_targets": 100, "test_excluded_targets": 50,
            "scheduler": {"kind": "warmup_half_cosine_with_floor", "groups": [{"role": "mu", "base_lr": 0.01}]}}
    base.update(summary or {})
    (run / "summary.json").write_text(json.dumps(base), encoding="utf-8")
    if metrics is not None:
        _write_metrics(run / "metrics.csv", metrics)
    if resources is not None:
        (run / "resources.json").write_text(json.dumps(resources), encoding="utf-8")
    if posthoc is not None:
        (run / "posthoc").mkdir()
        (run / "posthoc" / "posthoc.json").write_text(json.dumps(posthoc), encoding="utf-8")
    if weights:
        torch.save({"model_state": trained["state"]}, run / "final_model.pt")
        torch.save({"model_state": trained["state"]}, run / "best_model.pt")
    if checkpoint:
        (run / "checkpoints").mkdir()
        torch.save({"step": 20, "model_state": trained["state"], "optimizer_state": trained["optimizer_state"],
                    "optimizer_parameter_manifest": trained["manifest"], "config": config},
                   run / "checkpoints" / "step_20.pt")
    if snapshots:
        (run / "map_snapshots").mkdir()
        maps = {k: v for k, v in trained["state"].items() if k.startswith("relational_operators")}
        for step, scale in ((0, 1.0), (10, 2.0), (20, 3.0)):
            torch.save({"step": step, "state": {k: v.double() * scale for k, v in maps.items()}},
                       run / "map_snapshots" / f"step_{step:07d}.pt")
    if residual:
        (run / "diagnostics").mkdir()
        _write_metrics(run / "diagnostics" / "residual_ratio.csv", {
            "step": [0, 0, 10, 10, 20, 20], "layer": [0, 1, 0, 1, 0, 1],
            "q10": [1, 2, 10, 20, 100, 200], "q50": [2, 3, 20, 30, 200, 300],
            "q90": [3, 4, 30, 40, 300, 400], "ln_rel_change_q50": [0.1] * 6})
    return run


def _new_metrics(offset: float = 0.0, *, counters: bool = True) -> Dict[str, Any]:
    r"""metrics.csv of a run trained on or after 2026-09-28 (log rows every 5 steps, eval rows 0/10/20)."""
    nan = float("nan")
    steps = [0, 5, 10, 15, 20]
    cols: Dict[str, Any] = {
        "step": steps,
        "val_ce": [2.0 + offset, nan, 1.5 + offset, nan, 1.6 + offset],
        "train_sample_ce": [2.0, nan, 1.2, nan, 1.0],
        "generalization_gap_fixed": [0.0 + offset, nan, 0.3 + offset, nan, 0.6 + offset],
        "tokens_seen": [0, 80, 160, 240, 320],
        "wall_clock_s": [0.0, 900.0, 1800.0, 2700.0, 3600.0],
        "sigma_log_mean": [math.log(2.0), 0.6, 0.5, 0.45, 0.4],
        "sigma_log_sd": [0.0, 0.01, 0.02, 0.03, 0.04],
    }
    if counters:
        cols.update({"cap_active_count": [0, nan, 3, nan, 1], "cap_entry_count": [100, nan, 100, nan, 100],
                     "frame_projection_count": [0, 1, 2, 2, 3], "frame_projection_total": [0, 10, 20, 30, 40],
                     "skipped_updates": [0, 0, 0, 1, 1], "attempted_updates": [0, 5, 10, 15, 20]})
    return cols


def _manifest(arms: Mapping[str, Any], **extra: Any) -> pr.Manifest:
    return pr.resolve_manifest({"arms": arms, **extra})


def _check_figure(paths, *, record: Mapping[str, Any]) -> None:
    from PIL import Image
    png, pdf = (Path(p) for p in paths)
    assert png.is_file() and pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size[0] <= round(7.2 * 300) + 1                     # at most 7.2 in at 300 dpi
    assert record["min_text_pt"] >= 6.0


# ----------------------------------------------------------------------------------------------
# multiseed_analysis
# ----------------------------------------------------------------------------------------------

def test_multiseed_analysis_new_scalar_keys_curves_and_token_axis(tmp_path: Path, monkeypatch) -> None:
    import multiseed_analysis as ms

    for key in ("test_best_ce", "test_best_ppl", "test_best_step", "train_sample_ce_final",
                "train_sample_ce_best"):
        assert key in ms.SCALAR_KEYS
    curve_columns = [column for column, _ in ms.CURVE_SPECS]
    assert "train_sample_ce" in curve_columns and "generalization_gap_fixed" in curve_columns
    assert "generalization_gap" in curve_columns                        # existing behavior kept
    (tmp_path / "multiseed_request.json").write_text(json.dumps({
        "schema_version": 1, "status": "complete", "seeds": [1, 2],
        "cells": [{"seed": s, "status": "complete"} for s in (1, 2)]}))
    for seed, best in ((1, 1.5), (2, 1.7)):
        run = tmp_path / f"run_s{seed}"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"config": {"attention_mode": "projected_predictive", "seed": seed}}))
        (run / "summary.json").write_text(json.dumps({"test_ppl": 5.0, "test_best_ce": best}))
        (run / "provenance.json").write_text(json.dumps({
            "seed": seed, "git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None,
            "train_data_sha256": "b" * 64, "train_data_n_tokens": 100, "val_data_sha256": "c" * 64,
            "val_data_n_tokens": 20, "test_data_sha256": "d" * 64, "test_data_n_tokens": 20,
            "data_seed": 3, "max_tokens": None, "tokenizer_tag": "synthetic"}))
        (run / "metrics.csv").write_text(
            f"step,tokens_seen,val_ce,generalization_gap_fixed\n10,160,{best},0.1\n20,320,{best - 0.1},0.2\n")
    by_tokens = ms.aggregate_seed_curves(tmp_path, x="tokens_seen", columns=["val_ce", "generalization_gap_fixed"])
    np.testing.assert_allclose(by_tokens["val_ce"]["steps"], [160, 320])
    np.testing.assert_allclose(by_tokens["val_ce"]["mean"], [1.6, 1.5])
    scalar = ms.aggregate_scalar(tmp_path, "test_best_ce")
    assert scalar["n"] == 2 and scalar["mean"] == pytest.approx(1.6)

    monkeypatch.setitem(ms.CONFIG, "curve_x", "epoch")
    with pytest.raises(ValueError, match="curve_x"):
        ms._curve_x()
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setitem(ms.CONFIG, "curve_x", "tokens_seen")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl", "test_best_ce"])
    emitted = []
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))
    assert ms.main() == 0
    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["curve_x"] == "tokens_seen"
    assert summary["curves_final_step"]["val_ce"]["step"] == 320.0
    assert summary["scalars"]["test_best_ce"]["mean"] == pytest.approx(1.6)

    monkeypatch.undo()
    from vfe3.viz import figures as vf
    monkeypatch.setitem(ms.CONFIG, "curve_x", "tokens_seen")
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    labels = []
    save = vf._save
    monkeypatch.setattr(vf, "_save", lambda fig, path: (labels.append(fig.axes[0].get_xlabel()), save(fig, path))[1])
    curves = ms.aggregate_seed_curves(tmp_path, x="tokens_seen", columns=["generalization_gap_fixed"])
    ms._emit_figures(tmp_path, {}, curves, {}, None)
    assert (tmp_path / "figures" / "curve_band__generalization_gap_fixed.png").is_file()
    assert "training tokens" in labels


# ----------------------------------------------------------------------------------------------
# Manifest, homogeneity, selection
# ----------------------------------------------------------------------------------------------

def test_resolve_manifest_globs_roots_and_unresolved(tmp_path: Path, trained) -> None:
    for seed in (1, 2):
        _write_run(tmp_path / "armA", f"run_s{seed}", trained, seed=seed, weights=False)
    _write_run(tmp_path, "single", trained, seed=5, weights=False)
    manifest = _manifest({"root": [str(tmp_path / "armA")], "glob": [str(tmp_path / "arm*" / "run_s1")],
                          "missing": [str(tmp_path / "nope")], "empty": []},
                         evaluated_run=str(tmp_path / "single"))
    by_label = {arm.label: arm for arm in manifest.arms}
    assert [r.seed for r in by_label["root"].runs] == [1, 2]
    assert [r.seed for r in by_label["glob"].runs] == [1]
    assert by_label["missing"].runs == [] and by_label["missing"].unresolved
    assert by_label["empty"].runs == [] and not by_label["empty"].unresolved
    assert any("nope" in problem for problem in manifest.problems)
    assert manifest.evaluated.seed == 5 and manifest.numerics is manifest.evaluated
    assert by_label["root"].runs[0] is by_label["glob"].runs[0]           # one record per directory
    with pytest.raises(ValueError, match="selection"):
        pr.resolve_manifest({"arms": {}, "selection": "median"})


def test_homogeneity_reports_config_and_seed_violations(tmp_path: Path, trained) -> None:
    runs = [pr.load_run(_write_run(tmp_path, f"s{seed}", trained, seed=seed, data_seed=seed + 10,
                                   weights=False)) for seed in (1, 2)]
    ok = pr.check_homogeneity(runs)
    assert ok["homogeneous"] and ok["violations"] == []
    assert ok["model_seed_varied"] and ok["data_seed_varied"]
    odd = pr.load_run(_write_run(tmp_path, "odd", trained, seed=1, data_seed=11, weights=False,
                                 raw_config={"m_p_sigma_lr": 0.5},
                                 provenance={"train_data_sha256": "e" * 64}))
    report = pr.check_homogeneity(runs + [odd])
    fields = {v["field"] for v in report["violations"]}
    assert fields == {"m_p_sigma_lr", "provenance:train_data_sha256"}
    assert report["duplicate_seed_pairs"] == [[1, 11]] and not report["homogeneous"]
    allowed = pr.check_homogeneity(runs + [odd], allowed=("m_p_sigma_lr",))
    assert {v["field"] for v in allowed["violations"]} == {"provenance:train_data_sha256"}
    assert allowed["allowed_differences"][0]["field"] == "m_p_sigma_lr"


def test_legacy_config_is_read_leniently_with_dropped_fields(tmp_path: Path, trained) -> None:
    run_dir = _write_run(tmp_path, "legacy", trained, weights=False,
                         raw_config={"qk_frame_mode": "tied", "__vfe3_config_schema__": 25})
    run = pr.load_run(run_dir)
    assert run.config_load == "lenient"
    assert set(run.dropped_config_fields) == {"qk_frame_mode", "__vfe3_config_schema__"}
    assert run.cfg.embed_dim == K
    strict = pr.load_run(run_dir, drop_unknown_config_fields=False)
    assert strict.cfg is None and strict.config_load.startswith("failed")
    assert strict.value("embed_dim") == K                                # raw values still read
    other = pr.load_run(_write_run(tmp_path, "legacy2", trained, weights=False,
                                   raw_config={"qk_frame_mode": "untied"}))
    report = pr.check_homogeneity([run, other])
    assert "retired:qk_frame_mode" in {v["field"] for v in report["violations"]}


def test_selection_rule_sources(tmp_path: Path, trained) -> None:
    new = pr.load_run(_write_run(tmp_path, "new", trained, weights=False,
                                 summary={"test_best_ce": 1.55, "test_best_step": 10}))
    best = pr.select_checkpoint(new, "best")
    assert best["test_ce"] == pytest.approx(1.55) and best["step"] == 10
    assert best["val_ce"] == pytest.approx(1.5) and best["test_source"] == "summary.json test_best_ce"
    assert best["test_ppl"] == pytest.approx(math.exp(1.55))
    final = pr.select_checkpoint(new, "final")
    assert final["test_ce"] == pytest.approx(1.7) and final["step"] == 20 and final["val_ce"] == pytest.approx(1.6)

    posthoc = {"passes": [
        {"checkpoint": "best_model.pt", "split": "test", "status": "ok", "subset": True, "ce": 9.0},
        {"checkpoint": "best_model.pt", "split": "test", "status": "ok", "subset": False, "ce": 1.58}]}
    legacy = pr.load_run(_write_run(tmp_path, "legacy", trained, weights=False, posthoc=posthoc))
    assert pr.select_checkpoint(legacy, "best")["test_ce"] == pytest.approx(1.58)   # subset pass ignored

    same = pr.load_run(_write_run(tmp_path, "same", trained, weights=False, summary={"best_step": 20}))
    assert pr.select_checkpoint(same, "best")["test_source"].endswith("(best step is the final step)")
    missing = pr.load_run(_write_run(tmp_path, "missing", trained, weights=False))
    out = pr.select_checkpoint(missing, "best")
    assert out["status"] == "unavailable" and out["test_ce"] is None and "posthoc_eval" in out["reason"]
    reloaded = pr.load_run(_write_run(tmp_path, "reloaded", trained, weights=False,
                                      summary={"reloaded_best": True, "evaluation_weights": "best"}))
    assert pr.select_checkpoint(reloaded, "best")["test_ce"] == pytest.approx(1.7)
    assert pr.select_checkpoint(reloaded, "final")["status"] == "unavailable"


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------

def _header_ok(latex: str, key: str) -> None:
    spec, header = MANUSCRIPT_HEADERS[key]
    lines = latex.splitlines()
    assert lines[0] == spec and lines[1] == r"\toprule" and lines[2] == header and lines[3] == r"\midrule"
    assert lines[-2] == r"\bottomrule" and lines[-1].startswith(r"\end{tabular")


def test_main_results_and_per_seed_tables(tmp_path: Path, trained) -> None:
    ces = (1.50, 1.56, 1.62)
    for seed, ce in zip((1, 2, 3), ces):
        _write_run(tmp_path / "armA", f"s{seed}", trained, seed=seed, data_seed=seed + 10, weights=False,
                   summary={"test_best_ce": ce, "wall_time_s": 3600.0 * seed},
                   resources={"parameter_breakdown": {"total": 456_654_162,
                                                      "context_processing": {"total": 5_252_608}}})
    _write_run(tmp_path, "armB", trained, seed=7, weights=True, summary={"test_best_ce": 2.0},
               raw_config={"seed": 7})
    manifest = _manifest({"Arm A": {"group": "Retraining", "runs": [str(tmp_path / "armA")]},
                          "Arm B": {"group": "Retraining", "runs": [str(tmp_path / "armB")]},
                          "Arm C": {"group": "Baselines", "runs": []}})
    latex, markdown, rows = pt.main_results_table(manifest)
    _header_ok(latex, "main")
    assert r"\multicolumn{6}{@{}l}{\emph{Retraining}}\\" in latex
    assert r"\midrule" + "\n" + r"\multicolumn{6}{@{}l}{\emph{Baselines}}\\" in latex
    mean, sd = np.mean(ces), np.std(ces, ddof=1)
    assert f"Arm A & 3 & 5.25M / 456.7M & ${mean:.4f} \\pm {sd:.4f}$" in latex
    ppl = [math.exp(c) for c in ces]
    assert f"${np.mean(ppl):.2f} \\pm {np.std(ppl, ddof=1):.2f}$ & $2.0$" in latex        # mean 2 device-hours
    assert "Arm C & pending & pending & pending & pending & pending" in latex
    arm_b = rows[1]
    assert arm_b["parameters"]["source"].startswith("final_model.pt")   # breakdown from the weights
    assert arm_b["test_ce"] == [2.0] and "$2.0000$" in latex
    assert "Arm A" in markdown and "± " in markdown

    latex, markdown, rows = pt.per_seed_table(manifest)
    _header_ok(latex, "per_seed")
    assert "Arm A & 1 / 11 & $10$ & $1.5000$ & $1.5000$ &" in latex
    assert "\n & 2 / 12 & $10$" in latex                                # arm label only on its first row
    assert "Arm C & pending" in latex and len(rows) == 4


def test_main_results_marks_heterogeneous_arm_and_missing_selection(tmp_path: Path, trained) -> None:
    _write_run(tmp_path / "arm", "s1", trained, seed=1, weights=False, summary={"test_best_ce": 1.5})
    _write_run(tmp_path / "arm", "s2", trained, seed=2, weights=False, raw_config={"m_p_mu_lr": 0.5})
    manifest = _manifest({"Mixed": [str(tmp_path / "arm")]})
    latex, markdown, rows = pt.main_results_table(manifest)
    assert "% Mixed: config differs across runs in m_p_mu_lr" in latex
    # The difference is also visible in the compiled table (audit 2026-10-16 F67): a marker and a note row.
    assert "Mixed$^{\\ddagger}$ & 1 of 2" in latex
    assert "Config differs across runs in: Mixed (m\\_p\\_mu\\_lr)." in latex.split("\\bottomrule", 1)[1]
    assert "1 of 2 runs have a test value" in markdown and "m_p_mu_lr" in markdown


def test_relational_run_table_from_summary_and_posthoc(tmp_path: Path, trained) -> None:
    new = pr.load_run(_write_run(tmp_path, "new", trained, weights=False,
                                 summary={"test_best_ce": 1.55, "train_sample_ce_final": 1.1}))
    latex, markdown, rows = pt.relational_run_table(new)
    _header_ok(latex, "relational")
    assert f"Best recorded validation & $10$ & ${math.exp(1.5):.4f}$" in latex
    assert f"Test, best-validation weights & $10$ & ${math.exp(1.55):.4f}$" in latex
    assert f"Training-split sample, evaluation protocol & $20$ & ${math.exp(1.1):.4f}$" in latex
    posthoc = {"passes": [{"checkpoint": "final_model.pt", "split": "train_sample", "status": "ok",
                           "ce": 1.2}]}
    legacy = pr.load_run(_write_run(tmp_path, "legacy", trained, weights=False, posthoc=posthoc))
    rows = pt.relational_run_rows(legacy)
    assert rows[4]["source"] == "posthoc.json final_model.pt train_sample"
    assert rows[3]["ppl"] is None
    latex, _, _ = pt.relational_run_table(legacy)
    assert "Test, best-validation weights & $10$ & pending" in latex
    latex, _, _ = pt.relational_run_table(None)
    assert latex.count("pending") == 10


def test_resources_table_sources_and_pending_column(tmp_path: Path, trained) -> None:
    resources = {"parameter_breakdown": {"per_type_tables": {"total": 900, "share": 0.9},
                                         "positional_frames": {"total": 40, "share": 0.04},
                                         "context_processing": {"total": 60, "share": 0.06},
                                         "total": 1000, "parameters_read_per_token": 123},
                 "optimizer_state_bytes": {"total": 2_000_000_000}, "peak_gpu_memory_mb": 1500.0,
                 "throughput_tokens_per_s": {"tokens_per_s": {"q50": 4321.0}}, "device_hours": 3.5,
                 "device_name": "GPU"}
    evaluated = pr.load_run(_write_run(tmp_path, "new", trained, weights=False, resources=resources))
    latex, markdown, records = pt.resources_table(evaluated, None)
    _header_ok(latex, "resources")
    assert r"$900$ ($90.00\%$) & pending" in latex
    assert "2.00 GB optimizer state; peak 1.50 GB & pending" in latex
    assert "median 4,321 tokens/s; 3.50 wall-clock device-hours (evaluation included)" in latex
    assert "Inference latency at declared batch and context & not recorded & pending" in latex
    bare = pr.load_run(_write_run(tmp_path, "bare", trained, weights=True, checkpoint=True))
    record = pt.resource_record(bare)
    assert record["parameters"]["source"].startswith("final_model.pt")
    assert record["optimizer_state_source"] == "step_20.pt"
    assert record["device_hours"] == pytest.approx(2.0)                  # summary wall_time_s / 3600


def test_methods_contract_new_and_legacy_runs(tmp_path: Path, trained) -> None:
    provenance = {"source_archive": {"available": True, "path": "source_archive.zip", "sha256": "1" * 64,
                                     "tree_sha256": "2" * 64, "n_files": 42},
                  "tokenizer_manifest": {"status": "recorded", "kind": "tiktoken", "encoding": "gpt2",
                                         "library_version": "0.9", "ranks_sha256": "3" * 64}}
    new = pr.load_run(_write_run(tmp_path, "new", trained, weights=False, checkpoint=True,
                                 provenance=provenance, config_extra={"inactive_fields": {"m_phi_key_lr": "x"}},
                                 resources={"tokens_seen": 320}))
    items = pt.methods_contract_items(new, "best")
    flat = {name: " ; ".join(f"{label} {value}" for label, value in entries) for name, entries in items.items()}
    assert "source archive source_archive.zip" in flat["Source and configuration"]
    assert "1" * 64 in flat["Source and configuration"]
    assert "attention_mode=projected_predictive" in flat["Source and configuration"]
    assert "accepted but inactive selectors m_phi_key_lr" in flat["Source and configuration"]
    assert "tiktoken/gpt2" in flat["Data and tokenizer"] and "passes over the unique training tokens 0.80" in flat["Data and tokenizer"]
    assert "prior_bank.sigma_log_embed" in flat["Optimization"] and "wd=0.2" in flat["Optimization"]
    assert "embed_dim=4" in flat["Architecture"] and "score_rank_per_head=2" in flat["Architecture"]
    assert "token presentations 320 (resources.json tokens_seen)" in flat["Training budget"]
    assert "checkpoint-selection rule best" in flat["Evaluation"]
    latex, markdown, record = pt.methods_contract_table(new, "best")
    _header_ok(latex, "methods")
    assert r"\texttt{" + "1" * pt.HASH_CHARS_LATEX + "}" in latex and "1" * 13 not in latex
    assert "`" + "1" * 64 + "`" in markdown
    assert r"wd=\allowbreak 0.2" in latex                                   # no break inside numbers
    assert re.search(r"\d\.\\allowbreak \d", latex) is None

    legacy = pr.load_run(_write_run(tmp_path, "legacy", trained, weights=False))
    flat = {name: " ; ".join(f"{label} {value}" for label, value in entries)
            for name, entries in pt.methods_contract_items(legacy, "final").items()}
    assert "source archive not recorded" in flat["Source and configuration"]
    assert "tokenizer manifest not recorded" in flat["Data and tokenizer"]
    assert "per-table weight decay not recorded" in flat["Optimization"]
    assert "accepted but inactive selectors not recorded" in flat["Source and configuration"]
    latex, _, _ = pt.methods_contract_table(None, "best")
    assert latex.count("pending") == 8


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def test_decay_only_curve_closed_form_and_schedule(tmp_path: Path, trained) -> None:
    curve = pf.decay_only_log_variance(2.0, np.full(10, 0.1), 0.5)
    np.testing.assert_allclose(curve, 2.0 * 0.95 ** np.arange(11), rtol=1e-12)
    with pytest.raises(ValueError):
        pf.decay_only_log_variance(1.0, np.array([2.0]), 1.0)
    from vfe3.train import _floor_lr_lambdas
    cfg = trained["cfg"]
    f = _floor_lr_lambdas([cfg.m_p_sigma_lr], cfg)[0]
    steps = [0, 5, 10, 15, 20]
    logged = {"step": steps, "lr_sigma": [cfg.m_p_sigma_lr * f(s) for s in steps]}
    run = pr.load_run(_write_run(tmp_path, "sched", trained, weights=False, checkpoint=True, metrics=logged))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["hyperparameter_source"].startswith("optimizer group in step_20.pt")
    assert info["weight_decay"] == pytest.approx(0.2) and info["lr_crosscheck"] == "ok"
    np.testing.assert_allclose(lrs, [cfg.m_p_sigma_lr * f(k) for k in range(20)])
    wrong = {"step": steps, "lr_sigma": [0.5] * 5}
    run = pr.load_run(_write_run(tmp_path, "wrong", trained, weights=False, metrics=wrong))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["lr_crosscheck"].startswith("mismatch") and np.allclose(lrs, 0.5)
    assert info["hyperparameter_source"].startswith("config.json")


def test_learning_curves_figure_new_and_legacy_runs(tmp_path: Path, trained) -> None:
    for seed, offset in ((1, 0.0), (2, 0.05)):
        _write_run(tmp_path / "armA", f"s{seed}", trained, seed=seed, weights=False,
                   metrics=_new_metrics(offset))
    legacy_metrics = {"step": [5, 10, 20], "val_ce": [1.9, 1.7, 1.65]}
    _write_run(tmp_path, "legacy", trained, weights=False, metrics=legacy_metrics)
    manifest = _manifest({"New": [str(tmp_path / "armA")], "Legacy": [str(tmp_path / "legacy")], "Empty": []})
    series = pf.learning_curve_series(manifest.arms[1].runs[0])
    np.testing.assert_allclose(series["tokens"], [80, 160, 320])     # step x batch 2 x context 8
    np.testing.assert_allclose(series["hours"], [0.5, 1.0, 2.0])     # wall_time_s 7200 linear in step
    assert series["hours_source"].startswith("summary.json") and series["gap"] is None
    derived = {"step": [10, 20], "val_ce": [1.7, 1.6], "train_sample_ce": [1.5, 1.2]}
    gap_run = pr.load_run(_write_run(tmp_path, "gap", trained, weights=False, metrics=derived,
                                     summary={"wall_time_s": None}))
    gap_series = pf.learning_curve_series(gap_run)
    np.testing.assert_allclose(gap_series["gap"], [0.2, 0.4])
    assert gap_series["gap_source"] == "metrics.csv val_ce - train_sample_ce"
    assert gap_series["hours"] is None and "wall_time_s" in gap_series["reasons"]["hours"]
    new = pf.learning_curve_series(manifest.arms[0].runs[0])
    np.testing.assert_allclose(new["tokens"], [0, 160, 320])
    np.testing.assert_allclose(new["hours"], [0.0, 0.5, 1.0])
    record = pf.plot_learning_curves(manifest, tmp_path / "fig")
    assert record["panels"] == ["A", "B", "C"]
    _check_figure(record["paths"], record=record)


def test_numerics_figure_all_panels_from_training_records(tmp_path: Path, trained) -> None:
    run = pr.load_run(_write_run(tmp_path, "new", trained, weights=True, checkpoint=True, snapshots=True,
                                 residual=True, metrics={**_new_metrics(), "lr_sigma": [float("nan")] * 5}))
    data = pf.numerics_data(run)
    assert [data[k]["status"] for k in "ABCD"] == ["ok"] * 4
    assert data["A"]["kind"] == "curves" and data["B"]["kind"] == "curves"
    layer0 = data["B"]["layers"][0]
    np.testing.assert_allclose(layer0["step"], [0, 10, 20])
    np.testing.assert_allclose(layer0["scale_rms"] / layer0["scale_rms"][0], [1, 2, 3], rtol=1e-9)
    decay = data["C"]["decay_only"]
    assert decay["s0_source"] == "metrics.csv sigma_log_mean at step 0"
    assert decay["final_mean"] < math.log(2.0) and decay["weight_decay"] == pytest.approx(0.2)
    labels = {s["label"] for s in data["D"]["series"]}
    assert labels == {"cost-bound activations", "frame-generator projections", "skipped updates"}
    assert "factorization_fallback_count/factorization_count" in data["D"]["absent"]
    record = pf.plot_numerics(run, tmp_path / "fig")
    _check_figure(record["paths"], record=record)


def test_numerics_figure_endpoint_fallbacks_and_route_skip(tmp_path: Path, trained) -> None:
    legacy = pr.load_run(_write_run(tmp_path, "legacy", trained, weights=True,
                                    metrics={"step": [5, 10], "val_ce": [1.8, 1.6]}))
    data = pf.numerics_data(legacy)
    assert data["A"]["status"] == "skipped" and "residual_ratio.csv" in data["A"]["reason"]
    assert data["B"]["kind"] == "endpoint" and set(data["B"]["layers"][0]["step"]) == {10.0, 20.0}
    assert data["C"]["kind"] == "endpoint"
    table = trained["state"]["prior_bank.sigma_log_embed"].double()
    assert data["C"]["mean"][0] == pytest.approx(float(table.mean()))
    assert data["C"]["decay_only"]["s0_source"] == "log(sigma_init) from config.json"
    assert data["D"]["series"] == [] and data["D"]["endpoints"][0]["label"] == "skipped updates"
    record = pf.plot_numerics(legacy, tmp_path / "fig")
    assert record["panels"]["A"]["status"] == "skipped"
    _check_figure(record["paths"], record=record)

    with_posthoc = _write_run(tmp_path, "with_posthoc", trained, weights=False, posthoc={"passes": []})
    (with_posthoc / "posthoc" / "csv").mkdir()
    for stem, scale in (("final", 1.0), ("best", 2.0)):                  # vfe3.posthoc.residual layout
        _write_metrics(with_posthoc / "posthoc" / "csv" / f"residual_{stem}.csv",
                       {"layer": [0, 1], "rho_q10": [scale, scale], "rho_q50": [2 * scale, 3 * scale],
                        "rho_q90": [4 * scale, 5 * scale], "rho_n": [10, 10]})
    panel = pf.numerics_data(pr.load_run(with_posthoc))["A"]
    assert panel["kind"] == "endpoint" and sorted(set(panel["table"]["step"])) == [10.0, 20.0]
    np.testing.assert_allclose(sorted(panel["table"]["q50"]), [2, 3, 4, 6])

    transported = pr.load_run(_write_run(tmp_path, "transported", trained, weights=False,
                                         raw_config={"attention_mode": "transported"},
                                         metrics={"step": [5], "val_ce": [1.0]}))
    data = pf.numerics_data(transported)
    assert data["B"]["status"] == "skipped" and "transported" in data["B"]["reason"]
    assert data["C"]["status"] == "skipped" and data["D"]["status"] == "ok"     # summary skipped_steps
    record = pf.plot_numerics(transported, tmp_path / "fig2")
    assert record["panels"]["B"]["status"] == "skipped"
    empty = pr.load_run(_write_run(tmp_path, "empty", trained, weights=False,
                                   summary={"skipped_steps": None, "attempted_steps": None}))
    with pytest.raises(pf.FigureSkipped):
        pf.plot_numerics(empty, tmp_path / "fig3")


def test_scaling_figure_grid_mean_sd(tmp_path: Path, trained) -> None:
    for k_value, ces in ((4, (2.0, 2.2)), (8, (1.8, 1.9))):
        for seed, ce in zip((1, 2), ces):
            _write_run(tmp_path / "K", f"K{k_value}_s{seed}", trained, seed=seed, weights=False,
                       raw_config={"embed_dim": k_value}, summary={"test_best_ce": ce})
    manifest = pr.resolve_manifest({"arms": {}, "scaling": {"K": [str(tmp_path / "K")]}})
    points = pf.scaling_points(manifest.scaling["K"], "K", "best")
    assert [p["x"] for p in points["points"]] == [4.0, 8.0]
    assert points["points"][0]["mean"] == pytest.approx(2.1)
    assert points["points"][0]["sd"] == pytest.approx(np.std((2.0, 2.2), ddof=1))
    assert manifest.scaling["K"].homogeneity["violations"] == []        # embed_dim is varied on purpose
    record = pf.plot_scaling(manifest, tmp_path / "fig")
    assert "B" in record["skipped"]
    _check_figure(record["paths"], record=record)
    with pytest.raises(pf.FigureSkipped):
        pf.plot_scaling(pr.resolve_manifest({"arms": {}}), tmp_path / "fig3")


def test_assemble_end_to_end(tmp_path: Path, trained) -> None:
    for seed in (1, 2):
        _write_run(tmp_path / "runs" / "armA", f"s{seed}", trained, seed=seed, data_seed=seed,
                   weights=True, checkpoint=seed == 1, metrics=_new_metrics(0.01 * seed),
                   summary={"test_best_ce": 1.5 + 0.01 * seed})
    evaluated = str(tmp_path / "runs" / "armA" / "s1")
    config = {"arms": {"A": [str(tmp_path / "runs" / "armA")], "Pending": []},
              "evaluated_run": evaluated, "final_run": None, "selection": "best",
              "scaling": {"depth": [str(tmp_path / "runs" / "armA")]}}
    report = assemble(config, out_dir=tmp_path / "out", log=lambda message: None)
    out = tmp_path / "out"
    for name in ("main_results", "per_seed", "relational_run", "resources", "methods_contract"):
        assert (out / "tables" / f"{name}.tex").is_file() and (out / "tables" / f"{name}.md").is_file()
    assert report["figures"]["m3_learning_curves"]["status"] == "ok"
    assert report["figures"]["s1_numerics"]["status"] == "ok"
    assert report["figures"]["s3_scaling"]["status"] == "ok"
    saved = json.loads((out / "paper_artifacts.json").read_text(encoding="utf-8"))
    assert saved["arms"][0]["homogeneity"]["homogeneous"] is True
    assert saved["tables"]["main_results"]["rows"][0]["n_selected"] == 2
    index = (out / "PAPER_ARTIFACTS.md").read_text(encoding="utf-8")
    assert "## Arms" in index and "Pending" in index
