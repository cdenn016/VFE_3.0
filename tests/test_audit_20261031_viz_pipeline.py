r"""Remediation of the October 31, 2026 audit, package ``viz-pipeline``.

Report assembly, the figure worker, the sweep adapters and ``make_figures.py``: the feature-probe summary stays out
of the returned figure paths (A054), the validation Pareto frontier excludes training-budget cells (A055), the
equivariance witness evaluates the frame-diagonal self term the M-step trains (A115), the multi-seed forest pairs
each cell with its own seed's baseline (A135), the report bank keeps no phi (A166), the loader path of
``belief_ce_bank`` runs one forward per batch (A167), the report rebuild loads its state on the host and releases it
(A168), ``n_e_steps`` is documented as ignored (A308), the ablation figure process refuses edited definitions (A349),
raw-record namespaces carry the values a record ran (A383), the history figures survive a refused replay (A385),
saved probes are rendered once (A386), and a report removes the figures it did not write and streams its child's
output (A387). Tiny CPU models only.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import textwrap
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

import make_figures
from tests.profiles import live_config, reference_k8_config, test_config, tiny_config
from vfe3.config import serialize_config
from vfe3.metrics import bootstrap_token_ce_band
from vfe3.model.model import VFEModel
from vfe3.viz import extract, figure_worker, report, sweep_adapters

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _quiet_two_threads():
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _feature_model() -> VFEModel:
    r"""The minimal-agent readout with a prototype-agent block MLP (tests/test_feature_diagnostics_20261006.py)."""
    cfg = test_config(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                      minimal_agent_inference="exact", block_mlp_mode="prototype_agents",
                      block_mlp_inference="exact", block_mlp_expansion=2, n_layers=2, max_seq_len=6,
                      attention_residual=False, predictive_covariance="moment_matched",
                      block_mlp_covariance="posterior")
    torch.manual_seed(0)
    return VFEModel(cfg)


def _plain_model(**overrides: Any) -> VFEModel:
    r"""The projected predictive route with a covariance frozen at encode (no feature probes)."""
    torch.manual_seed(0)
    return VFEModel(test_config(attention_mode="projected_predictive", n_layers=1, max_seq_len=6, **overrides))


def _batch(model: VFEModel, rows: int = 2, length: int = 6, seed: int = 1) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.cfg.vocab_size, (rows, length), generator=generator)
    targets = torch.randint(0, model.cfg.vocab_size, (rows, length), generator=generator)
    return ids, targets


def _report_request(run_dir: Path, **fields: Any) -> Path:
    result = run_dir / ".figure_result.json"
    result.write_text("{}", encoding="utf-8")
    request = run_dir / ".figure_request.json"
    request.write_text(json.dumps({
        "mode": "report", "run_dir": str(run_dir), "result_path": str(result), "split": "validation",
        "device": "cpu", "max_sequences": None, "n_e_steps": None, "generate_embedding_figures": False,
        **fields,
    }), encoding="utf-8")
    return request


# ----------------------------------------------------------------------------------- A054 viz-pipeline-1

def test_feature_probe_figures_return_figure_paths_and_log_the_summary(tmp_path: Path) -> None:
    model = _feature_model()
    ids, _ = _batch(model)
    figdir = tmp_path / "figures"
    figdir.mkdir()
    written = report.generate_feature_probe_figures(model, [ids], tmp_path, figdir, CPU)
    assert {path.name for path in written} == {"minimal_agent_gap.png", "template_usage.png",
                                               "template_codebook.png"}
    assert all(path.parent == figdir for path in written)
    summary = json.loads((tmp_path / "feature_probes.json").read_text(encoding="utf-8"))
    assert "nu" not in summary["prototype"]["layers"][0]                  # the record is still written


def test_make_figures_accepts_the_report_of_a_feature_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""make_figures' published-path validation on the tiny feature route, the worker run in-process."""
    from vfe3.run_artifacts import _serialize_run_config

    model = _feature_model()
    run_dir = tmp_path / "feature"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"config": _serialize_run_config(model.cfg), "dataset": ""}, default=str), encoding="utf-8")
    loader = [_batch(model)]
    real_generate = report.generate_figures

    def bound_generate(run_dir_arg: Path, **kwargs: Any) -> List[Path]:
        return real_generate(run_dir_arg, model=model, loader=loader, device=CPU, split=kwargs["split"],
                             max_sequences=kwargs["max_sequences"], prepare_saved_probes=kwargs["prepare_saved_probes"],
                             generate_embedding_figures=kwargs["generate_embedding_figures"],
                             n_e_steps=kwargs["n_e_steps"], logger=kwargs["logger"])

    def in_process_worker(command: List[str], **kwargs: Any) -> tuple:
        monkeypatch.setenv("VFE3_FIGURE_REQUEST", kwargs["env"]["VFE3_FIGURE_REQUEST"])
        return figure_worker.main(), ""

    monkeypatch.setattr(report, "generate_figures", bound_generate)
    monkeypatch.setattr(make_figures, "_run_figure_worker", in_process_worker)
    paths = make_figures._generate_figures_isolated(
        run_dir, device="cpu", split="validation", max_sequences=2, n_e_steps=None,
        generate_embedding_figures=False)
    figure_root = (run_dir / "figures").resolve()
    assert {"minimal_agent_gap.png", "template_usage.png", "template_codebook.png"} <= {p.name for p in paths}
    assert all(figure_root in path.parents for path in paths)
    assert (run_dir / "feature_probes.json").is_file()


# ----------------------------------------------------------------------------------- A055 viz-pipeline-2

def test_pareto_frontier_excludes_training_budget_cells() -> None:
    import scaling_analysis

    rows = [
        {"route": "width", "label": "K8", "scale_knob": "embed_dim", "seed": 0, "best_val_ppl": 50.0,
         "wall_time_s": 10.0, "n_params": 1000, "embed_dim": 8},
        {"route": "width", "label": "K16", "scale_knob": "embed_dim", "seed": 0, "best_val_ppl": 40.0,
         "wall_time_s": 20.0, "n_params": 3000, "embed_dim": 16},
        {"route": "train_steps", "label": "steps1000", "scale_knob": "max_steps", "seed": 0,
         "best_val_ppl": 80.0, "wall_time_s": 5.0, "n_params": 1000},
        {"route": "train_steps", "label": "steps8000", "scale_knob": "max_steps", "seed": 0,
         "best_val_ppl": 30.0, "wall_time_s": 40.0, "n_params": 1000},
    ]
    points = sweep_adapters.aggregate_validation_points(rows)
    pareto = sweep_adapters.pareto_frontier_kwargs(points)
    assert pareto["points"]["label"] == ["K8", "K16"]
    assert pareto["points"]["n_params"].tolist() == [1000.0, 3000.0]
    assert sweep_adapters.NON_PARAMETER_KNOBS == scaling_analysis.NON_PARAMETER_KNOBS
    # Budget cells alone leave fewer than two parameter points: the figure is withheld.
    assert sweep_adapters.pareto_frontier_kwargs([p for p in points if p["scale_knob"] == "max_steps"]) is None


# ------------------------------------------------------------------------- A115 gap-mstep-self-coupling-handoff-5

def _frame_diagonal_model(**overrides: Any) -> VFEModel:
    cfg = reference_k8_config(family="gaussian_frame_diagonal", predictive_readout="kernel",
                              block_mlp_mode="coordinate", block_mlp_covariance="passthrough",
                              decode_mode="diagonal_chunked", norm_type_block="layernorm", pos_phi="none",
                              untie_decode_bank=True, predictive_covariance_floor_mode="relative",
                              mstep_self_coupling_weight=0.5, n_layers=1, **overrides)
    torch.manual_seed(0)
    return VFEModel(cfg).double()


@pytest.mark.parametrize("mode", ["state_dependent", "state_dependent_per_coord"])
def test_witness_self_term_on_the_frame_diagonal_family_is_at_roundoff(mode: str) -> None:
    model = _frame_diagonal_model(lambda_alpha_mode=mode)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 4), generator=torch.Generator().manual_seed(0))
    residuals = extract.projected_gauge_equivariance(model, ids)["residuals"]
    assert max(residuals["self_coupling"]) < 1e-9, residuals["self_coupling"]
    for key, values in residuals.items():
        assert max(values) < 1e-9, (key, values)


def test_witness_self_term_on_the_frame_diagonal_family_detects_an_unmoved_frame(
        monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Negative control: moving the means but not the frames breaks U^-1 mu, so the self term must not stay fixed."""
    model = _frame_diagonal_model(lambda_alpha_mode="state_dependent_per_coord")
    ids = torch.randint(0, model.cfg.vocab_size, (1, 4), generator=torch.Generator().manual_seed(0))
    monkeypatch.setattr(extract, "_act_on_frame", lambda g, frame: frame)
    residuals = extract.projected_gauge_equivariance(model, ids, n_draws=1)["residuals"]
    assert residuals["self_coupling"][0] > 1e-3


def test_witness_self_term_reads_the_trained_family_and_the_mstep_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.free_energy as free_energy
    from vfe3.families.frame_gaussian import FrameDiagonalGaussian
    from vfe3.families.gaussian import FullGaussian

    seen: List[tuple] = []
    original = free_energy.self_divergence_for_alpha

    def recording(q: Any, p: Any, **kwargs: Any) -> torch.Tensor:
        seen.append((type(q), type(p), kwargs["kl_max"]))
        return original(q, p, **kwargs)

    monkeypatch.setattr(free_energy, "self_divergence_for_alpha", recording)
    model = _frame_diagonal_model(lambda_alpha_mode="state_dependent_per_coord", self_kl_max=5.0)
    ids = torch.randint(0, model.cfg.vocab_size, (1, 4), generator=torch.Generator().manual_seed(0))
    extract.projected_gauge_equivariance(model, ids, n_draws=2)
    assert seen and all(entry == (FrameDiagonalGaussian, FrameDiagonalGaussian, 5.0) for entry in seen)

    seen.clear()
    torch.manual_seed(0)
    full = VFEModel(tiny_config(family="gaussian_full", decode_mode="full_chunked", mstep_self_coupling_weight=0.5,
                                lambda_alpha_mode="state_dependent", self_kl_max=5.0)).double().eval()
    extract.projected_gauge_equivariance(full, torch.tensor([[1, 4, 2]]), n_draws=1)
    assert seen and all(entry == (FullGaussian, FullGaussian, 5.0) for entry in seen)


# ----------------------------------------------------------------------------------- A135 viz-pipeline-5

_WINDOW = 4


def _forest_cell(sweep: Path, label: str, seed: int, tensor: torch.Tensor) -> Dict[str, Any]:
    cell = sweep / label
    cell.mkdir(parents=True)
    path = cell / "val_token_nats.pt"
    torch.save(tensor, path)
    marker = {"label": label, "status": "success", "error_kind": None, "final_val_ppl": 10.0, "seed": seed,
              "paired_token_bootstrap": True, "val_token_nats_path": "val_token_nats.pt",
              "val_token_nats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "val_token_nats_numel": int(tensor.numel()), "val_token_nats_dtype": str(tensor.dtype)}
    (cell / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")
    (cell / "config.json").write_text(json.dumps({"config": {"max_seq_len": _WINDOW}}), encoding="utf-8")
    return {**marker, "_cell_dir": str(cell)}


def _seeded_forest(tmp_path: Path) -> tuple:
    sweep = tmp_path / "forest_sweep"
    generator = torch.Generator().manual_seed(0)
    vectors = {label: torch.rand(24, generator=generator, dtype=torch.float64) + 1.0
               for label in ("baseline__s0", "baseline__s1", "arm__s0", "arm__s1")}
    rows = [_forest_cell(sweep, label, int(label[-1]), vector) for label, vector in vectors.items()]
    return sweep, vectors, rows


def test_ablation_forest_pairs_each_seeded_cell_with_its_own_seed_baseline(tmp_path: Path) -> None:
    sweep, vectors, rows = _seeded_forest(tmp_path)
    for admitted in (None, rows):
        out = sweep_adapters.ablation_forest_kwargs(sweep, "baseline", admitted_rows=admitted)
        assert out is not None and out["n_seeds"] == 2
        got = {row["label"]: row for row in out["rows"]}
        assert set(got) == set(vectors)
        for label, vector in vectors.items():
            reference = vectors[f"baseline__s{label[-1]}"]                   # the same seed's baseline cell
            band = bootstrap_token_ce_band(vector, reference, seed=0, block_len=_WINDOW)
            for key in ("delta", "lo", "hi"):
                assert got[label][key] == pytest.approx(band[key] / math.log(2.0)), (label, key)


def test_ablation_forest_withholds_a_seed_without_its_baseline(tmp_path: Path,
                                                               caplog: pytest.LogCaptureFixture) -> None:
    sweep, _vectors, rows = _seeded_forest(tmp_path)
    admitted = [row for row in rows if row["label"] != "baseline__s1"]
    with caplog.at_level(logging.WARNING, logger="vfe3.viz.sweep_adapters"):
        assert sweep_adapters.ablation_forest_kwargs(sweep, "baseline", admitted_rows=admitted) is None
    assert "baseline__s1" in caplog.text


# ----------------------------------------------------------------------------------- A166 viz-pipeline-7

def test_report_bank_keeps_no_phi_and_its_consumers_are_unchanged() -> None:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(max_seq_len=6)).eval()
    tokens, targets = _batch(model)
    records = extract.collect_inference_bank(model, [(tokens, targets)])
    record = records[0]
    assert record["out"].phi is None and record["out"].right_phi is None
    assert record["belief"].phi is None and record["belief"].right_phi is None
    retained = {id(tensor) for tensor in extract._inference_bank_tensors(records)}
    expected = {id(record["tokens"]), id(record["targets"]), id(record["out"].mu), id(record["out"].sigma),
                id(record["belief"].mu)}
    assert retained == expected
    from_bank = extract.belief_ce_bank(model, [], inference_bank=records)
    from_loader = extract.belief_ce_bank(model, [(tokens, targets)])
    for key in ("tr_sigma", "ce", "token_ids", "conf", "correct"):
        assert torch.equal(from_bank[key], from_loader[key]), key


# ----------------------------------------------------------------------------------- A167 viz-pipeline-8

def _two_pass_ce_bank(model: VFEModel, loader: List[tuple]) -> Dict[str, torch.Tensor]:
    r"""The former loader path: ``model(tokens)`` for the logits, then a captured forward for the covariance."""
    from vfe3 import metrics
    tr, ce, ids, conf, correct = [], [], [], [], []
    with torch.no_grad():
        for tokens, targets in loader:
            logits = model(tokens).reshape(-1, model.cfg.vocab_size).float()
            capture: dict = {}
            model.forward_beliefs(tokens, capture=capture)
            trs = metrics.sigma_trace(capture["out"].sigma, diagonal=model.cfg.diagonal_covariance,
                                      family=model.cfg.family).reshape(-1)
            tgt = targets.reshape(-1)
            per = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=-100, reduction="none")
            top, pred = logits.softmax(dim=-1).max(dim=-1)
            valid = tgt != -100
            tr.append(trs[valid]); ce.append(per[valid]); ids.append(tgt[valid])
            conf.append(top[valid]); correct.append((pred[valid] == tgt[valid]).float())
    return {"tr_sigma": torch.cat(tr), "ce": torch.cat(ce), "token_ids": torch.cat(ids),
            "conf": torch.cat(conf), "correct": torch.cat(correct)}


@pytest.mark.parametrize("profile", ["tiny", "live"])
def test_belief_ce_bank_loader_path_runs_one_forward_per_batch(profile: str,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(max_seq_len=6) if profile == "tiny" else live_config()).eval()
    length = model.cfg.max_seq_len
    loader = [_batch(model, length=length, seed=seed) for seed in (1, 2)]
    loader[1][1][0, 0] = -100                                               # an ignored target
    expected = _two_pass_ce_bank(model, loader)
    calls: List[int] = []
    original = model.forward_beliefs

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_beliefs", counting)
    got = extract.belief_ce_bank(model, loader)
    assert len(calls) == len(loader)
    for key in ("tr_sigma", "ce", "token_ids", "conf", "correct"):
        assert torch.equal(got[key], expected[key]), key


# ----------------------------------------------------------------------------------- A168 viz-pipeline-9

class _TrackedState(dict):
    r"""A state mapping a weak reference can follow."""


class _StopReport(BaseException):
    r"""Ends the report right after the rebuild (escapes ``_safe``, which catches ``Exception`` only)."""


def test_report_rebuild_loads_on_the_host_and_releases_the_loaded_state(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = test_config(attention_mode="projected_predictive", n_layers=1, max_seq_len=6)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "final_model.pt").write_bytes(b"placeholder; the loader is replaced")
    torch.manual_seed(0)
    source = VFEModel(cfg).state_dict()
    seen: Dict[str, Any] = {}

    def load_state(path: Path, cfg_arg: Any, expected: Any, code: Any, data: Any, *, map_location: object) -> dict:
        seen["map_location"] = map_location
        state = _TrackedState({key: value.clone() for key, value in source.items()})
        seen["state"] = weakref.ref(state)
        return state

    def stop_at_the_replay(*args: Any, **kwargs: Any) -> None:
        gc.collect()
        seen["released"] = seen["state"]() is None
        raise _StopReport

    monkeypatch.setattr(report, "_load_config", lambda path: (cfg, ""))
    monkeypatch.setattr(report, "_load_run_figure_contract", lambda path: ("0" * 64, None))
    monkeypatch.setattr(report, "_load_figure_model_state", load_state)
    monkeypatch.setattr(extract, "collect_inference_bank", stop_at_the_replay)
    with pytest.raises(_StopReport):
        report.generate_figures(run_dir, loader=[(torch.zeros(1, 6, dtype=torch.long),) * 2], device=CPU,
                                max_sequences=1)
    assert seen["map_location"] == "cpu" and isinstance(seen["map_location"], str)
    assert seen["released"] is True


# ----------------------------------------------------------------------------------- A308 viz-pipeline-10

def test_report_worker_says_n_e_steps_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(_report_request(run_dir, n_e_steps=3)))
    monkeypatch.setattr(report, "generate_figures", lambda *args, **kwargs: [])
    monkeypatch.setattr(figure_worker, "_load_worker_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(figure_worker, "_render_persisted_run_figures", lambda *args, **kwargs: None)
    with caplog.at_level(logging.INFO):
        assert figure_worker.main() == 0
    assert "n_e_steps=3 is ignored" in caplog.text


# ----------------------------------------------------------------------------------- A349 ablation-harness-13

def test_ablation_figure_request_carries_the_definition_fingerprints(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    seen: Dict[str, Any] = {}

    def fake_run(command: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        seen.update(json.loads(Path(kwargs["env"]["VFE3_FIGURE_REQUEST"]).read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ablation, "run_process_tree", fake_run)
    assert ablation._run_ablation_figures_isolated(tmp_path, scope="component_ablation_forest") is True
    assert seen["definition_fingerprints"] == ablation._figure_definition_fingerprints()
    assert set(seen["definition_fingerprints"]) == {"baseline", "sweeps"}


def test_definition_fingerprints_follow_the_baseline_and_the_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    before = ablation._figure_definition_fingerprints()
    monkeypatch.setitem(ablation.SWEEPS, "zz_added_after_import", {"description": "x", "param": "seed",
                                                                   "values": [1, 2]})
    after_sweep = ablation._figure_definition_fingerprints()
    assert after_sweep["sweeps"] != before["sweeps"] and after_sweep["baseline"] == before["baseline"]
    monkeypatch.setitem(ablation.BASELINE_CONFIG, "seed", int(ablation.BASELINE_CONFIG.get("seed", 0)) + 1)
    after_baseline = ablation._figure_definition_fingerprints()
    assert after_baseline["baseline"] != before["baseline"]


def test_definition_fingerprints_are_identical_in_a_fresh_process() -> None:
    import ablation

    script = "import json, ablation; print(json.dumps(ablation._figure_definition_fingerprints()))"
    environment = {**os.environ, "PYTHONHASHSEED": "123"}
    completed = subprocess.run([sys.executable, "-c", script], cwd=str(Path(ablation.__file__).resolve().parent),
                               env=environment, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == ablation._figure_definition_fingerprints()


def test_figure_worker_refuses_a_scope_whose_definitions_changed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    current = ablation._figure_definition_fingerprints()
    stale = {**current, "sweeps": "0" * 64}
    with pytest.raises(ValueError, match="no longer matches"):
        figure_worker._render_ablation_request(
            {"scope": "component_ablation_forest", "definition_fingerprints": stale}, tmp_path)
    assert not (tmp_path / "figures").exists()                              # refused before any output
    with pytest.raises(ValueError, match="definition fingerprints"):
        figure_worker._render_ablation_request({"scope": "component_ablation_forest"}, tmp_path)
    # Matching definitions pass the check and reach the scope's own validation.
    with pytest.raises(ValueError, match="comparison cohort"):
        figure_worker._render_ablation_request(
            {"scope": "__sensitivity__", "definition_fingerprints": current}, tmp_path)
    # Invalidation reads no definition, so it needs none.
    figure_worker._render_ablation_request({"scope": "component_ablation_forest", "invalidate": True}, tmp_path)


# ----------------------------------------------------------------------------------- A383 type-contracts-7

def _refused_run(run_dir: Path, *, removed: tuple = (), metrics: bool = False) -> dict:
    run_dir.mkdir(parents=True)
    raw = {**serialize_config(test_config()), "lambda_alpha_mode": "constant"}      # a value this build refuses
    for name in removed:
        raw.pop(name)
    (run_dir / "config.json").write_text(json.dumps({"config": raw, "dataset": ""}), encoding="utf-8")
    if metrics:
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "val_ce", "tokens_seen", "wall_clock_s"])
            writer.writeheader()
            for step in range(1, 5):
                writer.writerow({"step": 10 * step, "val_ce": 5.0 - 0.3 * step, "tokens_seen": 640 * step,
                                 "wall_clock_s": 3.0 * step})
    return raw


def test_raw_record_namespace_carries_the_values_the_record_ran(tmp_path: Path) -> None:
    from vfe3.config_history import FIELD_INTRODUCTION_VALUES
    from vfe3.paper.runs import load_run

    removed = ("attention_residual", "pos_phi", "relational_receiver_update", "kl_max")
    raw = _refused_run(tmp_path / "run", removed=removed)
    cfg = figure_worker._load_worker_config(tmp_path / "run")
    assert isinstance(cfg, SimpleNamespace)
    assert cfg.attention_residual is FIELD_INTRODUCTION_VALUES["attention_residual"] is True
    assert cfg.pos_phi == FIELD_INTRODUCTION_VALUES["pos_phi"] == "learned"
    assert cfg.relational_receiver_update == "euler"                       # the strict loader's pre-field receiver
    assert cfg.kl_max == 100.0                                              # the historical pin
    assert cfg.lambda_alpha_mode == "constant" and cfg.embed_dim == raw["embed_dim"]   # saved values kept
    assert vars(load_run(tmp_path / "run").cfg_like()) == vars(cfg)


def test_raw_record_namespace_leaves_a_missing_attention_selector_missing(tmp_path: Path) -> None:
    _refused_run(tmp_path / "run", removed=("attention_mode",))
    cfg = figure_worker._load_worker_config(tmp_path / "run")
    assert not hasattr(cfg, "attention_mode")


# ----------------------------------------------------------------------------------- A385 viz-figures-9

def test_report_mode_writes_the_history_figures_when_the_replay_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    _refused_run(run_dir, metrics=True)
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(_report_request(run_dir)))
    with pytest.raises(ValueError):
        figure_worker.main()
    assert (run_dir / "figures" / "learning_curves.png").stat().st_size > 0


def test_report_mode_draws_the_history_figures_once_after_a_successful_replay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: List[str] = []
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(_report_request(run_dir)))
    monkeypatch.setattr(report, "generate_figures", lambda *args, **kwargs: calls.append("replay") or [])
    monkeypatch.setattr(report, "generate_history_figures", lambda *args, **kwargs: calls.append("history") or [])
    monkeypatch.setattr(figure_worker, "_load_worker_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(figure_worker, "_render_persisted_run_figures", lambda *args, **kwargs: None)
    assert figure_worker.main() == 0
    assert calls == ["replay"]                                    # generate_figures draws them itself


# ----------------------------------------------------------------------------------- A386 viz-pipeline-11

def test_saved_probe_preparation_persists_records_and_the_worker_renders_them_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.viz import figures as figs

    counts = {"plot_beta_channel_decomposition": 0, "plot_context_sensitivity": 0}
    for name in counts:
        original = getattr(figs, name)

        def counted(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(figs, name, counted)
    torch.manual_seed(71)                                       # the probe model of tests/test_predictive_reporting.py
    model = VFEModel(test_config(
        vocab_size=7, embed_dim=4, n_heads=2, batch_size=1, max_seq_len=4, n_layers=1, n_e_steps=1,
        family="gaussian_diagonal", gauge_group="block_glk", skip_belief_sigma_update=True, pos_phi="learned",
        use_block_mlp=True, block_mlp_expansion=1, attention_mode="projected_predictive")).eval()
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    logger = logging.getLogger("viz_pipeline_11")
    report._prepare_missing_saved_probes(model, [(tokens, targets)], tmp_path, CPU, logger)
    for name in ("beta_channel_decomposition", "context_sensitivity"):
        assert json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))["report_replay"]["split"]
        assert not (tmp_path / f"{name}.png").exists()
    assert counts == {"plot_beta_channel_decomposition": 0, "plot_context_sensitivity": 0}
    written = figure_worker._render_saved_probe_figures(tmp_path, logger)
    assert counts == {"plot_beta_channel_decomposition": 1, "plot_context_sensitivity": 1}
    assert {"beta_channel_decomposition.png", "context_sensitivity.png"} <= {path.name for path in written}


# ----------------------------------------------------------------------------------- A387 viz-pipeline-12

def test_report_removes_the_figures_this_invocation_did_not_write(tmp_path: Path) -> None:
    from vfe3.viz import figures as figs

    model = _plain_model()
    assert {"belief_spectrum", "holonomy_curvature"} <= set(figs.route_figure_skip_reasons(model.cfg))
    run_dir = tmp_path / "run"
    figdir = run_dir / "figures"
    figdir.mkdir(parents=True)
    # Route-gated, feature-probe and history names this invocation cannot write, and one it rewrites.
    stale = ("belief_spectrum.png", "holonomy_curvature.png", "template_usage.png", "learning_curves.png")
    placeholder = b"an earlier invocation's figure"
    for name in (*stale, "numerical_trust.png"):
        (figdir / name).write_bytes(placeholder)
    (figdir / "notes.txt").write_text("not a figure this driver owns", encoding="utf-8")
    written = report.generate_figures(run_dir, model=model, loader=[_batch(model)], device=CPU, max_sequences=2)
    assert written and all(path.is_file() for path in written)
    assert not {path.name for path in written} & set(stale)
    for name in stale:
        assert not (figdir / name).exists(), name
    assert (figdir / "numerical_trust.png").read_bytes() != placeholder        # rewritten, not removed
    assert (figdir / "notes.txt").is_file()


def test_figure_worker_output_reaches_the_driver_while_the_child_runs(tmp_path: Path) -> None:
    r"""The child waits for a flag the driver creates on receiving its first line: captured output would time out."""
    flag = tmp_path / "first_line_seen"
    child = textwrap.dedent(f"""
        import pathlib, sys, time
        print("first", flush=True)
        flag = pathlib.Path({str(flag)!r})
        deadline = time.monotonic() + 60.0
        while not flag.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        print("second", flush=True)
        sys.exit(0 if flag.exists() else 3)
    """)
    lines: List[str] = []

    def on_line(line: str) -> None:
        lines.append(line)
        if line == "first":
            flag.write_text("seen", encoding="utf-8")

    code, tail = make_figures._run_figure_worker([sys.executable, "-c", child], cwd=str(tmp_path),
                                                 env=dict(os.environ), timeout=120.0, on_line=on_line)
    assert code == 0
    assert lines == ["first", "second"]
    assert tail.splitlines()[-2:] == ["first", "second"]


def test_figure_worker_timeout_terminates_the_child(tmp_path: Path) -> None:
    child = "import time; print('started', flush=True); time.sleep(120)"
    with pytest.raises(subprocess.TimeoutExpired):
        make_figures._run_figure_worker([sys.executable, "-c", child], cwd=str(tmp_path), env=dict(os.environ),
                                        timeout=3.0, on_line=lambda line: None)


def test_make_figures_streams_the_real_worker_and_quotes_its_failure(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    r"""The shipped worker process on a run without a checkpoint: its lines are logged and its error is quoted."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"config": serialize_config(test_config()), "dataset": ""}), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="make_figures"):
        with pytest.raises(RuntimeError, match="exited with code 1") as failure:
            make_figures._generate_figures_isolated(run_dir, device="cpu", split="validation", max_sequences=1,
                                                    n_e_steps=None, generate_embedding_figures=False)
    assert "no model checkpoint" in str(failure.value)
    assert any("no model checkpoint" in record.getMessage() for record in caplog.records
               if record.name == "make_figures")
