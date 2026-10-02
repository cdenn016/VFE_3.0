"""S1 panel D numerical-event counters recorded during training (GL(K)_attention_v2, S1 panel D).

Tiny CPU models only (K=4, two heads, N=4, vocab 7). Covers the per-evaluation cost-cap and
reduced-covariance factorization counts on the fixed diagnostic windows (and their agreement with
the post-hoc cap observer), the cumulative frame-projection, skipped-update and attempted-update
counters, diagnostics/cap_activation.csv, summary.json's numerics_counters,
checkpoint carry-over on resume, exception isolation, and bitwise invariance of the training
trajectory.
"""
from tests.profiles import test_config
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.relational_scoring as rs
import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.gauge_optim import embedded_phi_frobenius_norm, project_phi_parameter_rows_
from vfe3.model.model import VFEModel
from vfe3.posthoc.klcap import KLCapObserver

B, N, V = 2, 4, 7
RADIUS = 0.05                       # small chart radius: the post-update projection binds


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=2, n_e_steps=1, warmup_steps=0, max_steps=12, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True,
        generate_embedding_figures=False, pos_phi="learned",
        use_block_mlp=True, block_mlp_expansion=1, beta_attention_prior="causal_alibi",
        phi_mstep_max_matrix_norm=RADIUS)
    values.update(family="gaussian_diagonal", gauge_group="block_glk",
                  attention_mode="projected_predictive", attention_residual=True,
                  norm_type_block="layernorm", layernorm_affine=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _loaders(cfg: VFE3Config, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(41) % V, N)
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    test_ds = TokenWindows((torch.arange(19) * 5 + 1) % V, N, pad_final=True, stride=cfg.eval_stride)
    return (DataLoader(train_ds, batch_size=B, shuffle=shuffle),
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


def _int(value: str) -> int:
    assert value != "" and float(value) == int(float(value)), value
    return int(value)


def _train(model: VFEModel, cfg: VFE3Config, loaders: Tuple[DataLoader, ...], *,
           artifacts: Any = None, log_interval: int = 0, eval_interval: int = 0,
           seed: int = 11, **kwargs: Any) -> List[float]:
    torch.manual_seed(seed)
    return T.train(model, loaders[0], cfg, n_steps=kwargs.pop("n_steps", cfg.max_steps),
                   grad_clip=1.0, log_interval=log_interval, eval_interval=eval_interval,
                   val_loader=loaders[1], device=torch.device("cpu"),
                   logger=logging.getLogger("counter-test"), artifacts=artifacts,
                   generate_samples=False, **kwargs)


def _phi_rows(model: VFEModel) -> int:
    tables = {id(t): t for t in (model.prior_bank.phi_embed, model.pos_phi_free) if t is not None}
    return sum(t.reshape(-1, t.shape[-1]).shape[0] for t in tables.values())


@pytest.fixture(autouse=True)
def _cpu_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


@pytest.fixture
def early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "EARLY_EVAL_STEPS", (0, 1, 3))
    monkeypatch.setattr(T, "EARLY_EVAL_MIN_INTERVAL", 1)


def test_projection_counter_is_an_exact_no_op_and_matches_the_stats() -> None:
    model = _model(_cfg())
    with torch.no_grad():
        generator = torch.Generator().manual_seed(3)
        model.prior_bank.phi_embed.copy_(torch.randn(
            model.prior_bank.phi_embed.shape, generator=generator) * 0.02)
    start = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    def _run(**kwargs: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model.load_state_dict(start)
        stats = project_phi_parameter_rows_(model, RADIUS, **kwargs)
        return {name: tensor.clone() for name, tensor in model.state_dict().items()}, stats

    plain, stats = _run(collect_stats=True)
    counter = torch.zeros(2, dtype=torch.long)
    counted, counted_stats = _run(collect_stats=True, count_out=counter)
    silent_counter = torch.zeros(2, dtype=torch.long)
    silent, _ = _run(collect_stats=False, count_out=silent_counter)
    for name in plain:
        assert torch.equal(plain[name], counted[name]) and torch.equal(plain[name], silent[name]), name
    assert counted_stats == stats
    assert 0 < stats["phi_chart_projected_rows"] < stats["phi_chart_total_rows"]
    assert counter.tolist() == silent_counter.tolist() == [
        int(stats["phi_chart_projected_rows"]), int(stats["phi_chart_total_rows"])]
    model.load_state_dict(start)
    for bad in (torch.zeros(2, dtype=torch.float32), torch.zeros(3, dtype=torch.long)):
        with pytest.raises(ValueError, match="count_out"):
            project_phi_parameter_rows_(model, RADIUS, count_out=bad)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, start[name]), name          # rejected before any mutation


def test_projected_route_counters_columns_files_and_summary(
        tmp_path: Path, early: None, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    loaders = _loaders(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    # Independent count of the rows each projection call rescales, read before the call.
    expected_projected = [0]
    real_project = T.project_phi_parameter_rows_

    def spy(target: VFEModel, radius: float, **kwargs: Any) -> Dict[str, float]:
        assert kwargs["count_out"] is not None
        for table in {id(t): t for t in (target.prior_bank.phi_embed, target.pos_phi_free)}.values():
            rows = table.detach().reshape(-1, table.shape[-1])
            norm = embedded_phi_frobenius_norm(rows, target.group)
            expected_projected[0] += int(((radius / norm.clamp(min=1e-12)).clamp(max=1.0) < 1.0).sum())
        return real_project(target, radius, **kwargs)

    monkeypatch.setattr(T, "project_phi_parameter_rows_", spy)
    _train(model, cfg, loaders, artifacts=artifacts, log_interval=2, eval_interval=4)
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [0, 1, 2, 3, 4, 6, 8, 10, 12]
    evals = {0, 1, 3, 4, 8, 12}
    windows = T._diagnostic_windows(loaders[1])
    W, L, H = int(windows.shape[0]), cfg.n_layers, cfg.n_heads
    n_rows = _phi_rows(model)
    previous = (0, 0)
    for row in rows:
        step = int(row["step"])
        for column in (*T.NUMERICS_EVAL_COLUMNS, *T.NUMERICS_CUMULATIVE_COLUMNS):
            assert column in row, column
        if step in evals:
            capped, entries = _int(row["cap_active_count"]), _int(row["cap_entry_count"])
            fallback, attempted = (_int(row["factorization_fallback_count"]),
                                   _int(row["factorization_count"]))
            # causal prior, N = 4: rows 1..3 have 2, 3, 4 sources in every window, layer and head
            assert entries == W * L * H * (2 + 3 + 4)
            assert 0 <= capped <= entries
            assert attempted == 2 * W * N * H * L                     # query and key role per layer
            assert 0 <= fallback <= attempted
        else:
            assert all(row[column] == "" for column in T.NUMERICS_EVAL_COLUMNS), step
        assert _int(row["attempted_updates"]) == step and _int(row["skipped_updates"]) == 0
        changed, checked = _int(row["frame_projection_count"]), _int(row["frame_projection_total"])
        assert checked == step * n_rows and 0 <= changed <= checked
        assert changed >= previous[0] and checked >= previous[1]
        previous = (changed, checked)
    assert previous[0] == expected_projected[0] > 0                    # the chart bound binds

    cap_rows = _rows(run_dir / "diagnostics" / "cap_activation.csv")
    assert list(cap_rows[0]) == list(T.CAP_ACTIVATION_FIELDS)
    assert len(cap_rows) == len(evals) * L * H
    by_step: Dict[int, List[int]] = {}
    for r in cap_rows:
        assert 0 <= _int(r["capped"]) <= _int(r["entries"]) == W * 9
        total = by_step.setdefault(_int(r["step"]), [0, 0])
        total[0] += _int(r["capped"])
        total[1] += _int(r["entries"])
    for row in rows:
        if int(row["step"]) in evals:
            assert by_step[int(row["step"])] == [_int(row["cap_active_count"]),
                                                 _int(row["cap_entry_count"])]
    last = rows[-1]
    summary = artifacts.numerics_counters
    assert summary["eval_step"] == 12 and summary["diagnostic_windows"] == W
    for column in (*T.NUMERICS_EVAL_COLUMNS, *T.NUMERICS_CUMULATIVE_COLUMNS):
        assert summary[column] == _int(last[column]), column
    assert summary["frame_projection_since_step"] == 0
    assert summary["skip_reasons"] == {
        "cap_activation": None, "factorization": None, "frame_projection": None}
    assert artifacts.diagnostics_failures == {}


def test_summary_json_carries_numerics_counters(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=4)
    loaders = _loaders(cfg)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, dataset="tiny")
    _train(model, cfg, loaders, artifacts=artifacts, log_interval=2, eval_interval=4)
    ra.finalize_run(
        model, artifacts, cfg, final_validation=T.evaluate(model, loaders[1]),
        train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2], losses=[1.0],
        device=torch.device("cpu"), wall_time=1.0, logger=logging.getLogger("counter-test"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["numerics_counters"] == json.loads(json.dumps(artifacts.numerics_counters))
    record = summary["numerics_counters"]
    assert record["eval_step"] == 4 and record["attempted_updates"] == 4
    assert record["skipped_updates"] == 0 and record["frame_projection_total"] == 4 * _phi_rows(model)
    assert 0 <= record["cap_active_count"] <= record["cap_entry_count"]


def test_forced_cap_is_counted_and_matches_the_posthoc_observer(tmp_path: Path) -> None:
    cfg = _cfg(kl_max=1e-6, max_steps=4)                      # nearly every routing cost binds
    model = _model(cfg)
    loaders = _loaders(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    _train(model, cfg, loaders, artifacts=artifacts, eval_interval=4)
    row = _rows(run_dir / "metrics.csv")[-1]
    assert 0 < _int(row["cap_active_count"]) <= _int(row["cap_entry_count"])

    # Same capture, same threshold: the in-training count equals the post-hoc "computed" population.
    probe = _model(_cfg())                                              # default cap: costs spread
    probe.eval()
    ids = T._diagnostic_windows(loaders[1])[:3]
    capture: dict = {"diagnostic": {}}
    with torch.no_grad():
        probe.forward_beliefs(ids, capture=capture)
    diagnostic = capture["diagnostic"]
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    scores = torch.cat([c["score_energy_pre"][..., causal].reshape(-1)
                        for c in diagnostic["layer_conditional_captures"]])
    threshold = float(scores.median())                                 # a mix of capped and free
    cap, factorization = T._capture_numerics_counts(diagnostic, threshold)
    observer = KLCapObserver(threshold)
    ones = torch.ones(ids.shape, dtype=torch.bool)
    observer.update(diagnostic, ones, {"windows": list(range(ids.shape[0])), "input_valid": ones})
    reference = {(r["layer"], r["head"]): (r["n_capped"], r["n_entries"])
                 for r in observer.rows() if r["population"] == "computed"}
    ours = {(layer, head): counts for layer, heads in enumerate(cap)
            for head, counts in enumerate(heads)}
    assert ours == reference
    assert 0 < sum(c for c, _ in ours.values()) < sum(e for _, e in ours.values())
    assert factorization == [(0, 2 * ids.shape[0] * N * cfg.n_heads)] * cfg.n_layers


def test_factorization_fallback_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    real_status = rs._cholesky_status

    def forced(G: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        chol, failed, screen, status = real_status(G)
        first = G.reshape(-1, G.shape[-2], G.shape[-1])[0]
        if torch.equal(first, torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)):
            return chol, failed, screen, status                        # the recovery's own refactor
        failed = failed.clone()
        failed.reshape(-1)[0] = True                                   # one entry takes the fallback
        status = status.clone()
        status[2] = True
        status[3] = True
        return chol, failed, screen, status

    monkeypatch.setattr(rs, "_cholesky_status", forced)
    windows = T._diagnostic_windows(_loaders(cfg)[1])
    accumulator: Dict[str, Any] = {}
    T._residual_and_attention_rows(model, windows, 1, torch.device("cpu"),
                                   counters_out=accumulator)
    columns, _ = T._numerics_eval_columns(accumulator, 1)
    chunks = math.ceil(windows.shape[0] / T.DIAGNOSTIC_CHUNK_WINDOWS)
    assert columns["factorization_fallback_count"] == 2 * cfg.n_layers * chunks   # q and k role
    assert columns["factorization_count"] == 2 * windows.shape[0] * N * cfg.n_heads * cfg.n_layers




def test_counters_do_not_change_the_training_trajectory(tmp_path: Path, early: None) -> None:
    cfg = _cfg()
    silent_model = _model(cfg)
    silent = _train(silent_model, cfg, _loaders(cfg))
    counted_model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, counted_model, save_model_artifacts=False)
    counted = _train(counted_model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2,
                     eval_interval=4)
    assert counted == silent                                            # bitwise loss history
    for (name, a), (_, b) in zip(silent_model.state_dict().items(),
                                 counted_model.state_dict().items()):
        assert torch.equal(a, b), name
    assert artifacts.numerics_counters["frame_projection_count"] > 0


def test_raising_counters_are_counted_and_training_continues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic counter failure")

    cfg = _cfg(max_steps=8)
    expected = _train(_model(cfg), cfg, _loaders(cfg))
    monkeypatch.setattr(T, "_capture_numerics_counts", boom)
    monkeypatch.setattr(T, "_cumulative_counter_columns", boom)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    with caplog.at_level(logging.WARNING, logger="counter-test"):
        losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2,
                        eval_interval=4)
    assert losses == expected                                           # training unaffected
    assert artifacts.diagnostics_failures == {"numerics_counters": 2, "numerics_cumulative": 4}
    assert sum("'numerics_counters' failed" in message for message in caplog.messages) == 1
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["diagnostics_failures"]) for row in rows] == [1, 3, 4, 6]
    for row in rows:
        assert all(row[column] == "" for column in
                   (*T.NUMERICS_EVAL_COLUMNS, *T.NUMERICS_CUMULATIVE_COLUMNS))
    assert len(_rows(run_dir / "diagnostics" / "residual_ratio.csv")) == 2 * cfg.n_layers
    assert not (run_dir / "diagnostics" / "cap_activation.csv").exists()


def test_skipped_updates_are_counted(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=8)
    model = _model(cfg)
    calls = [0]

    def poison(grad: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        return torch.full_like(grad, float("nan")) if calls[0] in (3, 4) else grad

    model.prior_bank.mu_embed.register_hook(poison)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2, eval_interval=4)
    assert calls[0] == cfg.max_steps                                     # one backward per step
    rows = _rows(run_dir / "metrics.csv")
    assert [(int(r["step"]), _int(r["skipped_updates"]), _int(r["attempted_updates"]))
            for r in rows] == [(2, 0, 2), (4, 2, 4), (6, 2, 6), (8, 2, 8)]
    n_rows = _phi_rows(model)
    assert [_int(r["frame_projection_total"]) for r in rows] == [
        2 * n_rows, 2 * n_rows, 4 * n_rows, 6 * n_rows]                  # accepted updates only
    assert artifacts.numerics_counters["skipped_updates"] == 2
    assert artifacts.realized_updates["skipped_steps"] == 2


def test_frame_projection_counts_survive_resume(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=6, checkpoint_interval=3, m_phi_lr=0.05)
    straight_model = _model(cfg)
    straight = ra.RunArtifacts(tmp_path / "straight", cfg, straight_model)
    _train(straight_model, cfg, _loaders(cfg, shuffle=False), artifacts=straight, log_interval=3)

    first_model = _model(cfg)
    first = ra.RunArtifacts(tmp_path / "first", cfg, first_model)
    _train(first_model, cfg, _loaders(cfg, shuffle=False), artifacts=first, log_interval=3,
           n_steps=3)
    checkpoint = tmp_path / "first" / "checkpoints" / "step_3.pt"
    carried = torch.load(checkpoint, weights_only=True)["numerics_counters"]
    assert carried == {"frame_projection_count": first.numerics_counters["frame_projection_count"],
                       "frame_projection_total": 3 * _phi_rows(first_model),
                       "frame_projection_since_step": 0}

    resumed_model = _model(cfg, seed=99)
    resumed = ra.RunArtifacts(tmp_path / "resumed", cfg, resumed_model)
    _train(resumed_model, cfg, _loaders(cfg, shuffle=False), artifacts=resumed, log_interval=3,
           resume_from=checkpoint)
    assert resumed.resumed_numerics_counters == carried
    final_straight = _rows(tmp_path / "straight" / "metrics.csv")[-1]
    final_resumed = _rows(tmp_path / "resumed" / "metrics.csv")[-1]
    assert int(final_resumed["step"]) == int(final_straight["step"]) == 6
    for column in T.NUMERICS_CUMULATIVE_COLUMNS:
        assert _int(final_resumed[column]) == _int(final_straight[column]), column
    assert resumed.numerics_counters["frame_projection_since_step"] == 0
    assert ra._checkpoint_numerics_counters({"frame_projection_count": -1}) is None
    assert ra._checkpoint_numerics_counters(None) is None
