"""Training-time diagnostics, finalize additions and run archive (audit 2026-09-28, Section 2).

Tiny CPU models only (K=4, two heads, N=4, vocab 7). Covers: the fixed training-sample columns, the
eval_interval-only evaluation schedule, per-named-group gradient/parameter/update norms, the
residual-ratio and attention-statistics CSVs, map snapshots, the conditional-route structure
columns, exception isolation of every new diagnostic, finalize's best-weight test and
resources.json, the corrected active-parameter count, the source archive and the tokenizer manifest.
"""
from tests.profiles import test_config
import copy
import csv
import hashlib
import io
import json
import logging
import math
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3 import run_archive
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

B, N, V = 2, 4, 7


def _cfg(route: str = "conditional", **overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=2, n_e_steps=1, warmup_steps=0, max_steps=12, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True,
        generate_embedding_figures=False, pos_phi="learned",
        use_block_mlp=True, block_mlp_expansion=1, beta_attention_prior="causal_alibi",
        norm_type_block="none", layernorm_affine=False)      # pinned: independent of the config defaults
    if route == "conditional":
        values.update(family="gaussian_diagonal", gauge_group="block_glk",
                      attention_mode="projected_predictive", attention_residual=True,
                      norm_type_block="layernorm", layernorm_affine=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(41) % V, N)                     # 10 windows = 5 full batches
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    test_ds = TokenWindows((torch.arange(19) * 5 + 1) % V, N, pad_final=True, stride=cfg.eval_stride)
    return (DataLoader(train_ds, batch_size=B, shuffle=True),              # global-RNG permutations
            DataLoader(val_ds, batch_size=2, shuffle=False),
            DataLoader(test_ds, batch_size=2, shuffle=False))


def _model(cfg: VFE3Config, seed: int = 5) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: str) -> float:
    return float("nan") if value == "" else float(value)


def _train(model: VFEModel, cfg: VFE3Config, loaders: Tuple[DataLoader, ...], *,
           artifacts: Any = None, log_interval: int = 0, eval_interval: int = 0,
           seed: int = 11) -> List[float]:
    torch.manual_seed(seed)
    return T.train(model, loaders[0], cfg, n_steps=cfg.max_steps, grad_clip=1.0,
                   log_interval=log_interval, eval_interval=eval_interval,
                   val_loader=loaders[1], device=torch.device("cpu"),
                   logger=logging.getLogger("diag-test"), artifacts=artifacts,
                   generate_samples=False)


@pytest.fixture(autouse=True)
def _no_heatmap_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def test_parameter_groups_by_name() -> None:
    expected = {
        "prior_bank.mu_embed": "mu_embed", "prior_bank.sigma_log_embed": "sigma_log_embed",
        "prior_bank.phi_embed": "phi_embed", "pos_phi_free": "pos_phi",
        "prior_bank.output_proj_weight": "decode", "prior_bank.output_proj_bias": "decode",
        "prior_bank.decode_log_scale": "decode",
        "relational_operators.raw_score_q": "relational_score",
        "relational_operators_extra.3.raw_score_k": "relational_score",
        "relational_operators.T": "relational_content",
        "relational_operators_extra.0.O": "relational_content",
        "block_mlps.2.fc1.weight": "block_mlp", "block_norm.weight": "norm",
        "final_norm.bias": "norm", "_orig_mod.block_norm.bias": "norm", "head_mixer.W": "other",
    }
    assert {name: T._parameter_group(name) for name in expected} == expected
    model = _model(_cfg())
    groups = {T._parameter_group(name) for name, _ in model.named_parameters()}
    assert "other" not in groups and {"relational_score", "norm", "pos_phi"} <= groups


def test_diagnostics_do_not_change_the_training_trajectory(tmp_path: Path) -> None:
    cfg = _cfg()
    silent_model = _model(cfg)
    silent = _train(silent_model, cfg, _loaders(cfg))
    diag_model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, diag_model, save_model_artifacts=False)
    loud = _train(diag_model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2,
                  eval_interval=4)
    assert loud == silent                                               # bitwise loss history
    for (name, a), (_, b) in zip(silent_model.state_dict().items(),
                                 diag_model.state_dict().items()):
        assert torch.equal(a, b), name
    assert artifacts.diagnostics_failures == {}


def test_conditional_route_columns_and_files(tmp_path: Path) -> None:
    cfg = _cfg()
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2,
                    eval_interval=4)
    assert len(losses) == cfg.max_steps
    rows = _rows(run_dir / "metrics.csv")
    steps = [int(row["step"]) for row in rows]
    assert steps == [2, 4, 6, 8, 10, 12]
    evals = {4, 8, 12}
    for column in (*T._TRAIN_SAMPLE_COLUMNS, *T._GROUP_COLUMNS, "sigma_log_mean",
                   "sigma_log_sd", "tokens_seen", "diagnostics_failures", "generalization_gap"):
        assert column in rows[0], column
    present = {"mu_embed", "sigma_log_embed", "phi_embed", "pos_phi", "decode",
               "relational_score", "relational_content", "block_mlp", "norm"}
    for row in rows:
        step = int(row["step"])
        assert int(row["tokens_seen"]) == step * B * N
        assert int(row["diagnostics_failures"]) == 0
        assert math.isfinite(_num(row["sigma_log_mean"])) and _num(row["sigma_log_sd"]) >= 0.0
        for group in present:
            assert math.isfinite(_num(row[f"param_norm_group_{group}"])), (step, group)
        assert math.isnan(_num(row["param_norm_group_other"]))
        for group in present:
            assert math.isfinite(_num(row[f"grad_norm_group_{group}"])), (step, group)
            ratio = _num(row[f"update_ratio_group_{group}"])
            if _num(row[f"param_norm_group_{group}"]) > 0.0:
                assert ratio >= 0.0, (step, group)
            else:                              # zero-initialized table: ratio undefined
                assert math.isnan(ratio), (step, group)
        assert _num(row["update_ratio_group_relational_score"]) > 0.0
        if step in evals:
            val_ce, sample_ce = _num(row["val_ce"]), _num(row["train_sample_ce"])
            assert math.isfinite(val_ce) and math.isfinite(sample_ce)
            assert _num(row["train_sample_ppl"]) == pytest.approx(math.exp(sample_ce))
            assert _num(row["generalization_gap_fixed"]) == pytest.approx(val_ce - sample_ce)
            for column in ("val_head_redundancy_js", "val_future_leakage", "val_row_sum_error",
                           "val_attn_entropy_min_norm", "pos_loss_first_q", "pos_loss_last_q",
                           "pos_loss_ratio", "val_effective_rank"):
                assert math.isfinite(_num(row[column])), (step, column)
            assert _num(row["val_future_leakage"]) < 1e-6                 # causal prior
        else:
            assert math.isnan(_num(row["val_ce"])) and math.isnan(_num(row["train_sample_ce"]))
    assert set(artifacts.train_sample_by_step) == evals

    residual = _rows(run_dir / "diagnostics" / "residual_ratio.csv")
    assert list(residual[0]) == list(T.RESIDUAL_RATIO_FIELDS)
    assert sorted({int(r["step"]) for r in residual}) == sorted(evals)
    assert len(residual) == len(evals) * cfg.n_layers
    for r in residual:
        q10, q50, q90 = _num(r["q10"]), _num(r["q50"]), _num(r["q90"])
        assert 0.0 <= q10 <= q50 <= q90
        assert 0.0 <= _num(r["ln_rel_change_q50"])                     # LayerNorm + residual
    attention = _rows(run_dir / "diagnostics" / "attention_stats.csv")
    assert list(attention[0]) == list(T.ATTENTION_STATS_FIELDS)
    assert len(attention) == len(evals) * cfg.n_layers * cfg.n_heads
    for a in attention:
        assert 0.0 <= _num(a["entropy_norm"]) <= 1.0 + 1e-6
        assert 0.0 <= _num(a["self_weight"]) <= 1.0
        assert 0.0 <= _num(a["prior_overlap"]) <= 1.0 + 1e-6
        assert 0.0 <= _num(a["distance_mean"]) <= N - 1
        assert int(a["n_rows"]) == 8 * (N - 1)                        # 8 windows, rows with >= 2 keys

    snapshots = sorted((run_dir / "map_snapshots").glob("step_*.pt"))
    assert [p.name for p in snapshots] == [f"step_{s:07d}.pt" for s in sorted(evals)]
    last = torch.load(snapshots[-1], weights_only=True)
    assert last["step"] == 12
    assert set(last["state"]) == set(T._map_snapshot_state(model))
    assert "pos_phi_free" in last["state"] and any(
        key.startswith("relational_operators") for key in last["state"])
    for name, tensor in last["state"].items():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, model.state_dict()[name].float(), rtol=0, atol=0)


def test_legacy_route_residual_and_attention_files(tmp_path: Path) -> None:
    cfg = _cfg("legacy", max_steps=6)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, eval_interval=3)
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [3, 6]
    assert all(math.isfinite(_num(row["train_sample_ce"])) for row in rows)
    residual = _rows(run_dir / "diagnostics" / "residual_ratio.csv")
    assert len(residual) == 2 * cfg.n_layers
    assert all(r["ln_rel_change_q50"] == "" for r in residual)  # no block normalization: NaN
    attention = _rows(run_dir / "diagnostics" / "attention_stats.csv")
    assert len(attention) == 2 * cfg.n_layers * cfg.n_heads
    assert all(0.0 <= _num(a["prior_overlap"]) <= 1.0 + 1e-6 for a in attention)


def test_no_validation_before_the_first_eval_interval_step(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=4)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2, eval_interval=1000)
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [2, 4]            # log rows only, no step-0 row
    assert all(math.isnan(_num(row["val_ce"])) for row in rows)
    assert artifacts.train_sample_by_step == {}
    assert not (run_dir / "map_snapshots").exists()


def test_long_run_evaluates_only_on_eval_interval_multiples(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1500 real optimizer steps take ~15 s on CPU; the schedule is decided by the loop, not the
    # step, so the update is replaced by a no-op that reports an accepted step (status_out's only key).
    def no_update(model: Any, optimizer: Any, scheduler: Any, tokens: torch.Tensor,
                  targets: torch.Tensor, *, grad_clip: Any, grad_accum_steps: int, scaler: Any,
                  metrics_out: Any, status_out: Dict[str, bool], **kwargs: Any) -> float:
        status_out["did_step"] = True
        return 0.0

    monkeypatch.setattr(T, "train_step", no_update)
    cfg = _cfg(max_steps=1500)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, eval_interval=1500)
    assert [int(row["step"]) for row in _rows(run_dir / "metrics.csv")] == [1500]
    assert [p.name for p in sorted((run_dir / "map_snapshots").glob("step_*.pt"))] == [
        "step_0001500.pt"]
    assert set(artifacts.train_sample_by_step) == {1500}


def test_failing_diagnostics_are_counted_and_training_continues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic diagnostic failure")

    cfg = _cfg(max_steps=8)
    reference = _model(cfg)
    expected = _train(reference, cfg, _loaders(cfg))
    monkeypatch.setattr(T, "_residual_and_attention_rows", boom)
    monkeypatch.setattr(T, "_map_snapshot_state", boom)
    monkeypatch.setattr(T, "_group_gradient_norms", boom)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    with caplog.at_level(logging.WARNING, logger="diag-test"):
        losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2,
                        eval_interval=4)
    assert losses == expected                                   # training unaffected
    assert artifacts.diagnostics_failures == {
        "residual_attention": 2, "map_snapshot": 2, "group_gradient_norms": 4}
    assert sum("'residual_attention' failed" in message for message in caplog.messages) == 1
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["diagnostics_failures"]) for row in rows] == [1, 4, 5, 8]
    assert all(row["grad_norm_group_mu_embed"] == "" for row in rows)
    assert all(math.isfinite(_num(row["update_ratio_group_mu_embed"])) for row in rows)
    assert math.isfinite(_num(rows[-1]["train_sample_ce"]))    # isolated from the failures
    assert not (run_dir / "diagnostics").exists() and not (run_dir / "map_snapshots").exists()


def test_finalize_best_weights_resources_and_provenance(tmp_path: Path) -> None:
    cfg = _cfg()
    loaders = _loaders(cfg)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, dataset="tiny")
    _train(model, cfg, loaders, artifacts=artifacts, log_interval=2, eval_interval=4)
    # A distinct validation-selected checkpoint at a non-eval step (2), so the best-weight branch
    # must load it, score test and the training sample on it, and restore the final weights.
    best_model = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in best_model.parameters():
            parameter.add_(0.05)
    artifacts.maybe_save_best(2, best_model, 1.0)
    final_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    final_validation = T.evaluate(model, loaders[1])
    results = ra.finalize_run(
        model, artifacts, cfg, final_validation=final_validation, train_loader=loaders[0],
        val_loader=loaders[1], test_loader=loaders[2], losses=[1.0],
        device=torch.device("cpu"), wall_time=2.0, logger=logging.getLogger("diag-test"))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, final_state[name]), name        # bitwise restore
    summary = json.loads((run_dir / "summary.json").read_text())
    sample, _ = T._build_fixed_train_sample(loaders[0], loaders[1], cfg)
    assert summary["test_ce"] == pytest.approx(results["test_ce"])
    assert summary["test_best_step"] == 2 and summary["test_best_status"] == "evaluated"
    assert summary["test_best_ce"] == pytest.approx(T.evaluate(best_model, loaders[2])["ce"])
    assert summary["test_best_ppl"] == pytest.approx(math.exp(summary["test_best_ce"]))
    assert summary["test_best_ce"] != pytest.approx(summary["test_ce"])
    assert summary["train_sample_ce_final"] == artifacts.train_sample_by_step[12]
    assert summary["train_sample_ce_final"] == pytest.approx(T.evaluate(model, sample)["ce"])
    assert summary["train_sample_ce_best"] == pytest.approx(T.evaluate(best_model, sample)["ce"])
    assert summary["alibi_slopes"] == pytest.approx([2.0 ** -4, 2.0 ** -8])
    assert summary["diagnostics_failures"] == {}
    resources = json.loads((run_dir / "resources.json").read_text())
    assert resources["device_name"] == "cpu" and resources["peak_gpu_memory_mb"] is None
    assert resources["parameter_breakdown"]["total"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    assert resources["optimizer_state_bytes"]["total"] > 0
    assert resources["throughput_tokens_per_s"]["tokens_per_s"]["n"] >= 1
    assert resources["tokens_seen"] == cfg.max_steps * B * N and resources["wall_time_s"] == 2.0
    provenance = json.loads((run_dir / "provenance.json").read_text())
    archive = provenance["source_archive"]
    assert archive["available"] and archive["path"] == run_archive.ARCHIVE_NAME
    assert archive["sha256"] == hashlib.sha256((run_dir / archive["path"]).read_bytes()).hexdigest()
    assert provenance["source_archive_drift"] in (True, False)   # other edits may land mid-test
    assert provenance["tokenizer_manifest"]["status"] == "unavailable"   # no tokenizer_tag


def test_best_equal_final_reuses_final_test(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=4)
    loaders = _loaders(cfg)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    _train(model, cfg, loaders, artifacts=artifacts, eval_interval=4)
    fields = ra._best_weight_and_train_sample_evaluation(
        model, artifacts, cfg, logging.getLogger("diag-test"), final_step=4,
        final_test={"test_ce": 1.5, "test_ppl": math.exp(1.5)}, train_loader=loaders[0],
        val_loader=loaders[1], test_loader=loaders[2], tokens_per_char=None,
        device=torch.device("cpu"))
    assert artifacts.best_step == 4
    assert fields["test_best_ce"] == 1.5 and fields["test_best_status"].startswith("best weights are")
    assert fields["train_sample_ce_best"] == fields["train_sample_ce_final"] == \
        artifacts.train_sample_by_step[4]


def test_active_params_per_token_counts_positional_row_and_norm() -> None:
    cfg = _cfg()
    model = _model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    full = ra._cost_model_fields(model, cfg, n_params, 100)["active_params_per_token"]
    stripped = copy.deepcopy(model)
    stripped.pos_phi_free = None
    stripped.block_norm = None
    base = ra._cost_model_fields(stripped, cfg, n_params, 100)["active_params_per_token"]
    assert full - base == model.pos_phi_free.shape[1] + sum(
        p.numel() for p in model.block_norm.parameters())


def test_source_archive_is_hashed_and_reproducible(tmp_path: Path) -> None:
    cfg = _cfg()
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, _model(cfg), save_model_artifacts=False)
    record = artifacts.source_archive
    assert record["available"], record
    path = tmp_path / "run" / record["path"]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        entries = [(name, archive.read(name)) for name in names]
    assert names == sorted(names) and len(names) == record["n_files"]
    assert {"vfe3/train.py", "vfe3/run_archive.py", "train_vfe3.py", "pyproject.toml"} <= set(names)
    assert not any("__pycache__" in name for name in names)
    assert run_archive._tree_sha256(entries) == record["tree_sha256"]
    assert isinstance(record["git_head"], str) and len(record["git_head"]) == 40
    assert isinstance(record["dirty_files"], list)


def test_source_archive_drift_writes_second_archive(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "vfe3" / "sub").mkdir(parents=True)
    (root / "vfe3" / "__init__.py").write_bytes(b"")
    (root / "vfe3" / "sub" / "a.py").write_bytes(b"x = 1\n")
    (root / "driver.py").write_bytes(b"print(1)\n")
    (root / "pyproject.toml").write_bytes(b"[project]\nname = 'tiny'\n")
    (root / "notes.txt").write_bytes(b"not archived\n")
    run_dir = tmp_path / "run"
    git_info = {"git_head": None, "dirty_files": None}
    start = run_archive.write_source_archive(run_dir, root=root, git_info=git_info)
    again = run_archive.write_source_archive(tmp_path / "run2", root=root, git_info=git_info)
    assert start["sha256"] == again["sha256"] and start["n_files"] == 4   # deterministic zip bytes
    unchanged = run_archive.verify_source_archive(run_dir, start, root=root)
    assert unchanged["drift"] is False and unchanged["source_archive_end"] is None
    assert not (run_dir / run_archive.ARCHIVE_END_NAME).exists()
    (root / "vfe3" / "sub" / "a.py").write_bytes(b"x = 2\n")
    drifted = run_archive.verify_source_archive(run_dir, start, root=root)
    assert drifted["drift"] is True
    end = drifted["source_archive_end"]
    assert end["available"] and end["tree_sha256"] == drifted["tree_sha256_end"]
    with zipfile.ZipFile(run_dir / run_archive.ARCHIVE_END_NAME) as archive:
        assert archive.read("vfe3/sub/a.py") == b"x = 2\n"
    with zipfile.ZipFile(run_dir / run_archive.ARCHIVE_NAME) as archive:
        assert archive.read("vfe3/sub/a.py") == b"x = 1\n"


def test_tokenizer_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    encoding = SimpleNamespace(_mergeable_ranks={b"a": 0, b"b": 1, b"ab": 2},
                               _special_tokens={"<|endoftext|>": 3}, n_vocab=4)
    requested: List[str] = []
    fake = SimpleNamespace(__version__="0.0-test",
                           get_encoding=lambda name: requested.append(name) or encoding)
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    first = run_archive.tokenizer_manifest("tiktoken")
    second = run_archive.tokenizer_manifest("tiktoken")
    assert requested == ["gpt2", "gpt2"] and first == second
    assert first["status"] == "recorded" and first["kind"] == "tiktoken"
    assert first["encoding"] == "gpt2" and first["library_version"] == "0.0-test"
    assert first["vocab_size"] == 4 and len(first["ranks_sha256"]) == 64
    assert run_archive.tokenizer_manifest("tiktoken_cl100k")["encoding"] == "cl100k_base"
    unknown = run_archive.tokenizer_manifest(None)
    assert unknown["status"] == "unavailable" and "reason" in unknown


def test_optimizer_state_bytes_and_sigma_statistics() -> None:
    cfg = _cfg()
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    assert T._optimizer_state_bytes(optimizer)["total"] == 0          # no state before a step
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.copy_(
            torch.arange(model.prior_bank.sigma_log_embed.numel(), dtype=torch.float32)
            .reshape(model.prior_bank.sigma_log_embed.shape))
    values = model.prior_bank.sigma_log_embed.detach().double()
    stats = T._sigma_log_statistics(model)
    assert stats["sigma_log_mean"] == pytest.approx(float(values.mean()))
    assert stats["sigma_log_sd"] == pytest.approx(float(values.std(unbiased=False)))


def test_diagnostic_rows_reject_a_changed_header(tmp_path: Path) -> None:
    cfg = _cfg()
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, _model(cfg), save_model_artifacts=False)
    artifacts.append_diagnostic_rows("x.csv", ("step", "v"), [{"step": 1, "v": float("nan")}])
    artifacts.append_diagnostic_rows("x.csv", ("step", "v"), [{"step": 2, "v": 0.5}])
    text = (tmp_path / "run" / "diagnostics" / "x.csv").read_text()
    assert list(csv.reader(io.StringIO(text))) == [["step", "v"], ["1", ""], ["2", "0.5"]]
    with pytest.raises(ValueError):
        artifacts.append_diagnostic_rows("x.csv", ("step", "w"), [{"step": 3, "w": 1.0}])
