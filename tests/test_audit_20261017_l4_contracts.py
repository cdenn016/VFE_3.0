r"""Audit 2026-10-17, lane L4: metrics contracts, readers and visualization.

Item 5: divergence-spike rows are classified once (``vfe3.metrics_schema``) and every reader drops them
or overlays them as markers, for runs with and without the ``row_kind`` column. Item 31: the column
registry, the ``log_metrics`` cell rule and the reader-tuple enumeration. Item 64: the numerics and
block-variance history figures. Items 65 to 67: the ``DataState`` key, the figure-pass guards and the
absolute ``run_dir``. Item 79: the single-pass word-cache encoder. Every fixture is synthetic and CPU only.
"""
from __future__ import annotations

import ast
import csv
import json
import logging
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pytest
import torch

import build_wikitext103_word_cache as builder
import make_figures
import multiseed_analysis as ms
from tests.profiles import test_config
from vfe3 import metrics_schema as schema
from vfe3 import run_artifacts as ra
from vfe3 import run_resources
from vfe3.config import serialize_config
from vfe3.contracts import DataState
from vfe3.data.datasets import cache_path
from vfe3.data.word_vocab import WordVocabulary, tokenize_row, vocabulary_from_counts, vocabulary_path
from vfe3.paper import figures as paper_figures
from vfe3.viz import diagnostic_figures as diag
from vfe3.viz import figures as figs
from vfe3.viz.figure_worker import _history_from_csv

LOG, EVAL = 10, 30
# (step, divergence_spike, grad_norm, tokens_per_s): regular rows on the 10/30 lattice, spikes at 13 and 47
# (forced rows), and a spike flagged on the log step 20 (a regular row that also reports the spike).
_ROWS = [
    (10, 0.0, 0.40, 1000.0),
    (13, 1.0, 9.00, 5000.0),
    (20, 1.0, 0.45, 1010.0),
    (30, 0.0, 0.42, 900.0),
    (40, 0.0, 0.41, 1005.0),
    (47, 1.0, 7.00, 6000.0),
    (50, 0.0, 0.39, 995.0),
    (60, 0.0, 0.43, 905.0),
]
_SPIKE_STEPS = {13, 47}


def _kind(step: int) -> str:
    if step in _SPIKE_STEPS:
        return "spike"
    return "eval" if step % EVAL == 0 else "log"


def _metrics_text(*, with_kind: bool, extra: Dict[str, Sequence[object]] | None = None) -> str:
    header = ["step", "divergence_spike", "grad_norm", "tokens_per_s", "train_step_tokens_per_s",
              "train_step_ms_mean", "peak_mem_mb", "val_ppl", "grad_norm_mu", "grad_norm_sigma",
              "grad_norm_phi", "weight_norm_mu", "weight_norm_phi", "train_ce"]
    extra = extra or {}
    header += list(extra)
    if with_kind:
        header.append("row_kind")
    lines = [",".join(header)]
    for index, (step, flag, norm, tps) in enumerate(_ROWS):
        cells = [step, flag, norm, tps, tps * 1.2, 10.0, 100.0,
                 (50.0 - step / 10.0) if step % EVAL == 0 else "",
                 norm * 0.5, norm * 0.3, norm * 0.2, 3.0, 2.0, 5.0 - step / 100.0]
        cells += [values[index] for values in extra.values()]
        if with_kind:
            cells.append(_kind(step))
        lines.append(",".join(str(c) for c in cells))
    return "\n".join(lines) + "\n"


def _write_run(run_dir: Path, *, with_kind: bool, seed: int = 1,
               extra: Dict[str, Sequence[object]] | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        serialize_config(test_config(seed=seed, log_interval=LOG, eval_interval=EVAL))))
    (run_dir / "metrics.csv").write_text(_metrics_text(with_kind=with_kind, extra=extra))
    return run_dir


# ------------------------------------------------------------------------------------------------
# Item 5: one row-kind helper, both formats
# ------------------------------------------------------------------------------------------------

def test_infer_row_kind_from_flag_and_cadence() -> None:
    kinds = schema.row_kinds([{"step": s, "divergence_spike": f} for s, f, _, _ in _ROWS],
                             intervals=(LOG, EVAL))
    assert kinds == [_kind(s) for s, *_ in _ROWS]


def test_declared_row_kind_wins_and_text_cells_are_read() -> None:
    rows = [{"step": "13", "divergence_spike": "1.0", "row_kind": "log"},
            {"step": "14", "divergence_spike": "1.0", "row_kind": ""},
            {"step": "15", "divergence_spike": "", "row_kind": None}]
    assert schema.row_kinds(rows, intervals=(LOG, EVAL)) == ["log", "spike", "log"]


def test_gcd_fallback_without_config() -> None:
    rows = [{"step": s, "divergence_spike": f} for s, f, _, _ in _ROWS]
    # The regular steps 10, 30, ..., 60 have gcd 10, so the forced rows at 13 and 47 are still recognized
    # (the eval rows read as log rows when no config gives eval_interval).
    assert schema.row_kinds(rows, cfg=None) == ["spike" if s in _SPIKE_STEPS else "log" for s, *_ in _ROWS]


def test_config_intervals_accept_every_config_shape() -> None:
    cfg = test_config(log_interval=LOG, eval_interval=EVAL)
    assert schema.config_intervals(cfg) == (LOG, EVAL)
    assert schema.config_intervals(serialize_config(cfg)) == (LOG, EVAL)
    assert schema.config_intervals({"config": {"log_interval": 0, "eval_interval": None}}) == (None, None)
    assert schema.config_intervals(SimpleNamespace(log_interval=5.0)) == (5, None)


@pytest.mark.parametrize("with_kind", [False, True])
def test_spike_mask_is_identical_in_both_formats(tmp_path: Path, with_kind: bool) -> None:
    run = _write_run(tmp_path / "run", with_kind=with_kind)
    mask = schema.metrics_csv_spike_mask(run / "metrics.csv")
    assert mask == [s in _SPIKE_STEPS for s, *_ in _ROWS]


@pytest.mark.parametrize("with_kind", [False, True])
def test_throughput_summary_drops_spike_rows(tmp_path: Path, with_kind: bool) -> None:
    run = _write_run(tmp_path / "run", with_kind=with_kind)
    out = run_resources.throughput_summary(run)
    regular = [tps for s, _, _, tps in _ROWS if s not in _SPIKE_STEPS]
    assert out["spike_rows_excluded"] == 2
    assert out["throughput_tokens_per_s"]["tokens_per_s"]["median"] == pytest.approx(float(np.median(regular)))


def test_resources_record_throughput_drops_spike_rows(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "run", with_kind=False)
    history = _history_from_csv(run / "metrics.csv")
    artifacts = SimpleNamespace(history=history, optimizer_state_bytes=None)
    cfg = test_config(log_interval=LOG, eval_interval=EVAL)
    record = ra._resources_record(torch.nn.Linear(2, 2), artifacts, cfg,
                                  device=torch.device("cpu"), wall_time=None)
    regular = [tps for s, _, _, tps in _ROWS if s not in _SPIKE_STEPS]
    assert record["throughput_tokens_per_s"]["tokens_per_s"]["q50"] == pytest.approx(float(np.median(regular)))


@pytest.mark.parametrize("with_kind", [False, True])
def test_aggregate_seed_curves_drops_spike_rows(tmp_path: Path, with_kind: bool) -> None:
    (tmp_path / "multiseed_request.json").write_text(json.dumps({
        "schema_version": 1, "status": "complete", "seeds": [1, 2],
        "cells": [{"seed": 1, "status": "complete"}, {"seed": 2, "status": "complete"}]}))
    for seed in (1, 2):
        run = _write_run(tmp_path / f"s{seed}", with_kind=with_kind, seed=seed)
        (run / "summary.json").write_text(json.dumps({"test_ppl": 1.0}))
        (run / "provenance.json").write_text(json.dumps({
            "seed": seed, "git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None,
            "train_data_sha256": "b" * 64, "train_data_n_tokens": 100, "val_data_sha256": "c" * 64,
            "val_data_n_tokens": 20, "test_data_sha256": "d" * 64, "test_data_n_tokens": 20,
            "data_seed": 3, "max_tokens": None, "tokenizer_tag": "synthetic"}))
    # Spike steps of one seed only: shift seed 2's forced rows so the grids disagree, as on real runs.
    path = tmp_path / "s2" / "metrics.csv"
    path.write_text(path.read_text().replace("\n13,", "\n17,").replace("\n47,", "\n43,"))
    withheld: Dict[str, str] = {}
    curves = ms.aggregate_seed_curves(tmp_path, columns=["train_ce", "grad_norm"], withheld=withheld)
    assert withheld == {}
    regular_steps = [s for s, *_ in _ROWS if s not in _SPIKE_STEPS]
    np.testing.assert_allclose(curves["grad_norm"]["steps"], regular_steps)
    assert float(np.max(curves["grad_norm"]["mean"])) < 1.0


def _spy(monkeypatch: pytest.MonkeyPatch, module: object, name: str) -> List[Dict[str, object]]:
    calls: List[Dict[str, object]] = []
    original = getattr(module, name)

    def wrapper(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


@pytest.mark.parametrize("with_kind", [False, True])
def test_save_figures_plots_regular_rows_and_marks_spikes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_kind: bool) -> None:
    run = _write_run(tmp_path / "run", with_kind=with_kind)
    trajectory = _spy(monkeypatch, figs, "plot_trajectory")
    decomposition = _spy(monkeypatch, figs, "plot_grad_norm_decomposition")
    geometry = _spy(monkeypatch, figs, "plot_optimizer_geometry")
    artifacts = SimpleNamespace(run_dir=run, cfg=SimpleNamespace(log_interval=LOG, eval_interval=EVAL),
                                history=_history_from_csv(run / "metrics.csv"))
    ra._save_figures(artifacts, None, logging.getLogger("l4"))
    grad = [c for c in trajectory if c["kwargs"].get("title") == "Gradient norm"]
    assert len(grad) == 1
    values, steps = grad[0]["args"]
    assert list(steps) == [s for s, *_ in _ROWS if s not in _SPIKE_STEPS]
    assert max(values) < 1.0
    marker_steps, marker_values = grad[0]["kwargs"]["markers"]
    assert list(marker_steps) == sorted(_SPIKE_STEPS) and list(marker_values) == [9.0, 7.0]
    assert (run / "grad_norm.png").is_file()
    hist = decomposition[0]["args"][0]
    assert 13 not in hist["step"] and decomposition[0]["kwargs"]["spikes"]["step"] == [13, 47]
    assert geometry[0]["kwargs"]["spikes"]["step"] == [13, 47]
    assert (run / "optimizer_geometry.png").is_file()


# ------------------------------------------------------------------------------------------------
# Item 31: registry, cell rule, reader tuples
# ------------------------------------------------------------------------------------------------

def _bare_artifacts(tmp_path: Path) -> ra.RunArtifacts:
    artifacts = object.__new__(ra.RunArtifacts)
    artifacts.csv_path = tmp_path / "metrics.csv"
    artifacts._fieldnames = None
    artifacts.history = []
    return artifacts


def test_log_metrics_converts_scalars_explicitly(tmp_path: Path) -> None:
    artifacts = _bare_artifacts(tmp_path)
    artifacts.log_metrics({"step": np.int64(10), "grad_norm": torch.tensor(0.5),
                           "train_ce": np.float32("nan"), "divergence_spike": True,
                           "row_kind": "log", "relational_semantics": "relational_predictive_content",
                           "peak_mem_mb": None})
    with artifacts.csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row == {"step": "10", "grad_norm": "0.5", "train_ce": "", "divergence_spike": "1",
                   "row_kind": "log", "relational_semantics": "relational_predictive_content", "peak_mem_mb": ""}
    assert all(type(v) in (int, float, str) for v in artifacts.history[0].values())


@pytest.mark.parametrize("cell", [torch.ones(2), np.ones(3), [1.0], {"a": 1}, object()])
def test_log_metrics_rejects_unwritable_cells(tmp_path: Path, cell: object) -> None:
    artifacts = _bare_artifacts(tmp_path)
    with pytest.raises(TypeError):
        artifacts.log_metrics({"step": 1, "grad_norm": cell})
    assert not artifacts.csv_path.exists() and artifacts.history == []


def test_log_metrics_rejects_text_in_a_numeric_column(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="declared float"):
        _bare_artifacts(tmp_path).log_metrics({"step": 1, "grad_norm": "tensor(0.5)"})


def _reader_tuples() -> Dict[str, tuple]:
    r"""Every metrics.csv column tuple a reader reads, by reader."""
    from vfe3.run_resources import _TELEMETRY_COLUMNS, _THROUGHPUT_COLUMNS
    return {
        "viz grad_norm_decomposition": figs.GRAD_NORM_DECOMPOSITION_COLUMNS,
        "viz optimizer_geometry": figs.OPTIMIZER_GEOMETRY_COLUMNS,
        "viz geometry_health": figs.GEOMETRY_HEALTH_COLUMNS,
        "viz validation_sanity": figs.VALIDATION_SANITY_COLUMNS,
        "viz numerics_history": diag.NUMERICS_PANEL_COLUMNS,
        "run_artifacts trajectories": ("val_ppl", "grad_norm", "belief_cond_median", "steps_per_epoch"),
        "run_artifacts free energy": ("self_coupling", "belief_coupling", "attention_entropy", "val_ce",
                                      "self_divergence", "val_self_coupling", "val_belief_coupling",
                                      "val_attention_entropy", "val_self_divergence"),
        "run_artifacts resources": ("tokens_per_s", "train_step_tokens_per_s", "peak_mem_mb", "tokens_seen"),
        "run_resources throughput": ("step", "peak_mem_mb", "train_step_ms_mean", *_THROUGHPUT_COLUMNS,
                                     *_TELEMETRY_COLUMNS),
        "multiseed curves": tuple(key for key, _ in ms.CURVE_SPECS),
        "multiseed grid": tuple(ms.GRID_COLS),
        "multiseed x axis": tuple(ms.CURVE_X_OPTIONS),
        "paper S1 panel D": tuple(key for pair in paper_figures.COUNTER_PAIRS for key in pair[:2]),
        "paper S1 panel D totals": tuple(key for key, _ in paper_figures.COUNTER_TOTALS),
        "row kinds": ("step", "divergence_spike", schema.ROW_KIND_COLUMN),
    }


def test_every_reader_tuple_is_registered_and_every_registered_column_is_read() -> None:
    readers = _reader_tuples()
    missing = {name: schema.unregistered_columns(keys) for name, keys in readers.items()}
    assert not any(missing.values()), missing
    read = {key for keys in readers.values() for key in keys}
    unread_registered = sorted(set(schema.METRIC_COLUMNS) - read)
    assert not unread_registered, f"registered columns no reader tuple reads: {unread_registered}"


def _written_columns() -> set:
    r"""Columns ``vfe3.train`` writes to metrics.csv that are statically enumerable: its column tuples and the
    literal keys of the row ``train()`` builds (``row = {...}``, ``row["..."] = ...`` and the ``for _dk/_gk/_pk in (...)``
    loops). Keys added from diagnostic dicts at run time (relational_*, qkv_*) are not enumerable."""
    import vfe3.train as train

    written = set(train.DIVERGENCE_COLUMNS + train.NUMERICS_EVAL_COLUMNS + train.NUMERICS_CUMULATIVE_COLUMNS
                  + train.DECODE_VARIANCE_COLUMNS + train.LOG_TIMING_COLUMNS + train._GROUP_COLUMNS
                  + train._STEP_LR_COLUMNS + train._TRAIN_SAMPLE_COLUMNS)
    module = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
    tree = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_train_loop")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "row" and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)):
                    written.add(target.slice.value)
                if (isinstance(target, ast.Name) and target.id == "row" and isinstance(node.value, ast.Dict)):
                    written.update(k.value for k in node.value.keys
                                   if isinstance(k, ast.Constant) and isinstance(k.value, str))
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id in ("_dk", "_gk", "_pk") and isinstance(node.iter, ast.Tuple)):
            written.update(e.value for e in node.iter.elts
                           if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return written


def test_report_written_columns_no_reader_uses(capsys: pytest.CaptureFixture[str]) -> None:
    written = _written_columns()
    assert {"train_loss", "grad_norm", "divergence_spike", "layernorm_trace_cap_elements"} <= written
    read = {key for keys in _reader_tuples().values() for key in keys}
    unread = sorted(written - read)
    print(f"{len(unread)} of {len(written)} enumerable written columns have no reader tuple: {unread}")
    # Every registered every-row column is in fact written on every row by train.py.
    import vfe3.train as train
    for name in train.NUMERICS_CUMULATIVE_COLUMNS:
        assert schema.METRIC_COLUMNS[name].cadence == "cumulative"
    for name in train.NUMERICS_EVAL_COLUMNS:
        assert schema.METRIC_COLUMNS[name].cadence == "eval"
    assert "divergence_spike" in train.DIVERGENCE_COLUMNS


# ------------------------------------------------------------------------------------------------
# Item 64: numerics and block-variance history figures
# ------------------------------------------------------------------------------------------------

def _block_variance_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["step,layer,stage,var_min,var_q001,var_q01,var_median,var_max,mean_rms_median"]
    for step in (30, 60):
        for layer in (0, 1):
            for stage in ("pre_norm", "post_norm", "block_out"):
                base = 0.1 * (layer + 1) * (step / 30)
                lines.append(f"{step},{layer},{stage},{base},{2 * base},{3 * base},{10 * base},{100 * base},0.3")
    path.write_text("\n".join(lines) + "\n")


def test_numerics_and_block_variance_figures(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    extra = {
        "layernorm_trace_cap_elements": [0, 0, 5, 9, 12, 12, 20, 25],
        "covariance_ceiling_elements": [0] * 8,
        "decode_sigma_v_min": [1e-3, "", 9e-4, 8e-4, 8e-4, 7e-4, 7e-4, 6e-4],
        "relational_score_nonfinite_frac": [0.0] * 8,
    }
    run = _write_run(tmp_path / "run", with_kind=True, extra=extra)
    _block_variance_csv(run / "diagnostics" / "block_variance.csv")
    artifacts = SimpleNamespace(run_dir=run, cfg=SimpleNamespace(log_interval=LOG, eval_interval=EVAL),
                                history=_history_from_csv(run / "metrics.csv"))
    with caplog.at_level(logging.INFO):
        ra._save_figures(artifacts, None, logging.getLogger("l4"))
    assert (run / "numerics_history.png").stat().st_size > 0
    assert (run / "block_variance_history.png").stat().st_size > 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text


def test_numerics_figures_skip_cleanly_when_columns_are_absent(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    run = _write_run(tmp_path / "run", with_kind=False)
    artifacts = SimpleNamespace(run_dir=run, cfg=None, history=_history_from_csv(run / "metrics.csv"))
    with caplog.at_level(logging.INFO):
        ra._save_figures(artifacts, None, logging.getLogger("l4"))
    assert not (run / "numerics_history.png").exists()
    assert not (run / "block_variance_history.png").exists()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
    with pytest.raises(figs.FigureSkipped):
        diag.plot_numerics_history({"step": [1, 2]})
    assert diag.read_block_variance(run / "diagnostics" / "block_variance.csv") is None


def test_paper_counter_panel_reads_the_cumulative_totals() -> None:
    metrics = {"step": np.array([10.0, 20.0]),
               "layernorm_trace_cap_elements": np.array([0.0, 4.0]),
               "gemm_fallback_slices": np.array([np.nan, np.nan])}
    run = SimpleNamespace(metrics=lambda: metrics, summary={}, posthoc=None, final_step=lambda: 20)
    panel = paper_figures._counter_panel(run)
    assert panel["status"] == "ok" and panel["series"] == [] and panel["endpoints"] == []
    assert [t["label"] for t in panel["totals"]] == ["LayerNorm trace-cap tokens"]
    assert "gemm_fallback_slices" in panel["absent"]


# ------------------------------------------------------------------------------------------------
# Items 65 to 67
# ------------------------------------------------------------------------------------------------

def test_data_state_declares_window_offset_optional() -> None:
    assert DataState.__optional_keys__ == frozenset({"window_offset"})
    assert DataState.__required_keys__ == frozenset(
        {"epoch_start_generator_state", "batches_consumed", "epoch", "data_identity"})


def test_figure_pass_survives_text_cells_under_numeric_keys(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "run", with_kind=False)
    history = _history_from_csv(run / "metrics.csv")
    history[0]["val_ppl"] = "n/a"
    history[1]["grad_norm"] = "n/a"
    history[2]["estep_grad_norm_mu"] = "n/a"
    history[3]["estep_grad_norm_mu"] = 0.5
    artifacts = SimpleNamespace(run_dir=run, cfg=None, history=history)
    ra._save_figures(artifacts, None, logging.getLogger("l4"))
    assert (run / "grad_norm.png").is_file() and (run / "optimizer_geometry.png").is_file()


def test_make_figures_passes_an_absolute_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    seen: Dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess:
        request = json.loads(Path(kwargs["env"]["VFE3_FIGURE_REQUEST"]).read_text(encoding="utf-8"))
        seen.update(request)
        return subprocess.CompletedProcess(command, 1, "", "stop")

    monkeypatch.setattr(make_figures, "run_process_tree", fake_run)
    with pytest.raises(RuntimeError, match="exited with code 1"):
        make_figures._generate_figures_isolated(
            Path("runs") / "r", device="cpu", split="test", max_sequences=None, n_e_steps=None,
            generate_embedding_figures=False)
    assert Path(str(seen["run_dir"])).is_absolute()
    assert Path(str(seen["run_dir"])) == run.resolve()
    assert Path(str(seen["result_path"])).parent == run.resolve()


# ------------------------------------------------------------------------------------------------
# Item 79: single-pass word-cache encoding
# ------------------------------------------------------------------------------------------------

_SHARDS = {
    "train|0": [" the cat sat ", "", " = Head = ", " the dog . "],
    "train|1": [" the cat , the dog . ", " zebra the "],
    "validation|0": [" the cat ", " unseen words "],
    "test|0": [" dog . ", ""],
}


@pytest.fixture()
def fake_corpus(monkeypatch: pytest.MonkeyPatch) -> Dict[str, int]:
    reads: Dict[str, int] = {}

    def read_rows(file_name: str) -> List[str]:
        reads[file_name] = reads.get(file_name, 0) + 1
        return list(_SHARDS[file_name])

    monkeypatch.setattr(builder, "SPLIT_FILES",
                        {"train": ("train|0", "train|1"), "validation": ("validation|0",), "test": ("test|0",)})
    monkeypatch.setattr(builder, "_read_rows", read_rows)
    return reads


def _reference_encode(split: str, vocabulary: WordVocabulary) -> np.ndarray:
    r"""The pre-2026-10-17 encoder: size each shard by a tokenizing pass, then encode in a second one."""
    shards = []
    for file_name in builder.SPLIT_FILES[split]:
        rows = builder._read_rows(file_name)
        n_tokens = sum(len(tokenize_row(row, eos=vocabulary.eos)) for row in rows)
        out = np.empty(n_tokens, dtype=np.int64)
        cursor = 0
        for row in rows:
            tokens = tokenize_row(row, eos=vocabulary.eos)
            if tokens:
                ids = vocabulary.encode(tokens)
                out[cursor:cursor + len(ids)] = ids
                cursor += len(ids)
        shards.append(out)
    return shards[0] if len(shards) == 1 else np.concatenate(shards)


@pytest.mark.parametrize("eos", [False, True])
def test_single_pass_encoder_matches_the_two_pass_reference(fake_corpus: Dict[str, int], eos: bool) -> None:
    _, _, _, counts = builder._count_split_tokens("train", eos=eos)
    vocabulary = WordVocabulary(vocabulary_from_counts(counts, min_count=0, eos=eos), min_count=0, eos=eos)
    for split in ("train", "validation", "test"):
        ids, n_rows, n_chars = builder._encode_split(split, vocabulary)
        reference = _reference_encode(split, vocabulary)
        assert ids.dtype == np.int64 and np.array_equal(ids, reference)
        rows = [row for name in builder.SPLIT_FILES[split] for row in _SHARDS[name]]
        assert n_rows == len(rows) and n_chars == sum(len(row) for row in rows)


def test_build_reads_each_training_shard_twice(tmp_path: Path, fake_corpus: Dict[str, int]) -> None:
    summary = builder.build(min_count=1, cache_dir=tmp_path, dataset=builder.WORD_DATASET)
    assert fake_corpus["train|0"] == 2 and fake_corpus["train|1"] == 2          # count, then encode
    assert fake_corpus["validation|0"] == 1 and fake_corpus["test|0"] == 1
    vocabulary = WordVocabulary.load(vocabulary_path(builder.WORD_DATASET, tag="word", cache_dir=tmp_path))
    for split in ("train", "validation", "test"):
        stored = torch.load(cache_path(builder.WORD_DATASET, split, suffix="pt", cache_dir=tmp_path),
                            weights_only=True)
        assert stored.tolist() == _reference_encode(split, vocabulary).tolist()
        assert summary["splits"][split]["n_tokens"] == stored.numel()
    assert math.isclose(summary["splits"]["train"]["n_chars_source"],
                        sum(len(r) for name in ("train|0", "train|1") for r in _SHARDS[name]))
