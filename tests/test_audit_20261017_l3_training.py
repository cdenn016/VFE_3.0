r"""Training-loop remediation of the 2026-10-17 deep audit, lane L3 (items 4, 5, 14-17, 21, 34, 35, 47-49, 68, 70).

Item 4: an exception in a step or an evaluation writes ``diagnostics/crash.json``, the failing batch and a resumable
checkpoint, and the run logs to ``train.log``; a deferred scorer verdict names the offending entries. Item 5: every
metrics row carries ``row_kind``. Item 14: eigenvalue and table-step collapse alarms and ``warnings.jsonl``. Item 15:
the divergence monitor records every event and reports an escalation inside the cooldown (the 200/260/330 probe).
Items 16 and 17: opt-in failed-step skips, the consecutive-skip abort and the loss-spike update rejection. Item 21:
the frame-class diagnostics on head blocks and distinct rows equal the dense computation to rounding. Item 34: every
dead-router head is named. Item 35: the new ``block_variance.csv`` columns. Item 47: the table Fisher step survives
1e17 and 1e35 gradients. Item 48: checkpoint pruning, timing and the end-of-run checkpoint. Item 49: the training-only
peak. Item 68: the background best-model save and figure jobs. Item 70: the clip multiply is skipped bit-identically.

Tiny CPU models only (K = 4, at most two heads, N = 4, vocab 7, a handful of steps).
"""
import csv
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import deferred_host_checks, describe_offending_entries, host_check
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.table_optim import TableFisherAdamW, TableFisherStatistics

B, N, V = 2, 4, 7
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=None,
        n_layers=1, n_e_steps=1, warmup_steps=0, max_steps=8, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True, generate_embedding_figures=False,
        use_block_mlp=False, family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _loaders() -> tuple:
    train = TokenWindows((torch.arange(61) * 5 + 2) % V, N)                 # 15 windows: 7 batches per epoch
    val = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True)
    return (DataLoader(train, batch_size=B, shuffle=False, drop_last=True),
            DataLoader(val, batch_size=2, shuffle=False))


def _run(cfg: VFE3Config, run_dir: Path, *, n_steps: int, log_interval: int = 0, eval_interval: int = 0,
         resume_from: Any = None, model_hook: Any = None) -> tuple:
    torch.manual_seed(5)
    model = VFEModel(cfg)
    if model_hook is not None:
        model_hook(model)
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    train_loader, val_loader = _loaders()
    torch.manual_seed(11)
    losses = T.train(model, train_loader, cfg, n_steps=n_steps, grad_clip=cfg.grad_clip,
                     log_interval=log_interval, eval_interval=eval_interval, val_loader=val_loader, device=CPU,
                     logger=logging.getLogger("audit-l3"), artifacts=artifacts, generate_samples=False,
                     resume_from=resume_from)
    return losses, artifacts, model


def _fail_forward_at(calls: List[int], *, fail_on: set) -> Any:
    r"""A model hook whose training forward raises the audit's ValueError on the given 0-based call numbers."""
    def hook(model: VFEModel) -> None:
        original = model.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if len(calls) - 1 in fail_on:
                raise ValueError("projected covariance must be finite for the diagonal route")
            return original(*args, **kwargs)
        model.forward = forward
    return hook


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------------------------------------ item 4

def test_injected_forward_failure_writes_crash_record_batch_and_a_checkpoint_that_resumes(tmp_path: Path) -> None:
    cfg = _cfg()
    straight, _, _ = _run(cfg, tmp_path / "straight", n_steps=6)
    calls: List[int] = []
    with pytest.raises(ValueError, match="projected covariance must be finite"):
        _run(cfg, tmp_path / "crashed", n_steps=6, model_hook=_fail_forward_at(calls, fail_on={3}))
    run_dir = tmp_path / "crashed"
    record = json.loads((run_dir / "diagnostics" / T.CRASH_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["step"] == 4 and record["phase"] == "step" and not record["optimizer_update_started"]
    assert record["exception"]["type"] == "ValueError" and record["batches_consumed"] == 4
    assert record["recent_losses"] == pytest.approx(straight[:3], rel=0, abs=0)
    batch = torch.load(record["failed_batch"], weights_only=True)
    assert batch["step"] == 4 and batch["tokens"].shape == (B, N)            # the loader's host batch
    assert torch.equal(batch["targets"][:, :-1], batch["tokens"][:, 1:])
    assert record["checkpoint_step"] == 3 and Path(record["checkpoint"]).name == "step_3.pt"
    log = (run_dir / T.RUN_LOG_NAME).read_text(encoding="utf-8")
    assert "training stopped at step 4" in log
    resumed, _, _ = _run(cfg, tmp_path / "resumed", n_steps=6, resume_from=record["checkpoint"])
    assert resumed == straight[3:]                                         # the emergency checkpoint is exact


def test_evaluation_failure_checkpoints_the_post_step_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    straight, _, _ = _run(cfg, tmp_path / "straight", n_steps=6)
    original = T.evaluate
    seen: List[int] = []

    def failing(*args: Any, **kwargs: Any) -> Any:
        if isinstance(args[1], DataLoader):                                 # the validation evaluation only
            seen.append(1)
            if len(seen) == 2:
                raise RuntimeError("injected evaluation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(T, "evaluate", failing)
    with pytest.raises(RuntimeError, match="injected evaluation failure"):
        _run(cfg, tmp_path / "crashed", n_steps=6, eval_interval=2)
    record = json.loads((tmp_path / "crashed" / "diagnostics" / T.CRASH_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["phase"] == "post_step" and record["step"] == 4 and record["checkpoint_step"] == 4
    monkeypatch.setattr(T, "evaluate", original)
    resumed, _, _ = _run(cfg, tmp_path / "resumed", n_steps=6, resume_from=record["checkpoint"])
    assert resumed == straight[4:]


def test_deferred_scorer_verdict_names_the_offending_entry() -> None:
    from vfe3.relational_scoring import _projected_diagonal_kl
    a = torch.zeros(1, 3, 2, 2)                                            # (B, N, H, r)
    g = torch.ones(1, 3, 2, 2)
    bad = g.clone()
    bad[0, 1, 0, 1] = -35.7                                                # token 1, head 0, coordinate 1
    layer_index = 2                                                        # the stack's loop local
    assert layer_index == 2
    with pytest.raises(ValueError, match="not positive") as caught:
        with deferred_host_checks():
            _projected_diagonal_kl(a, g, a, bad, True)
    notes = " ".join(caught.value.__notes__)
    assert "registered at layer 2" in notes
    assert "key variances: 1 offending entry" in notes and "batch 0, head 0, token 1, coordinate 1" in notes
    assert "value -35.7" in notes
    text = describe_offending_entries(torch.tensor([[1.0, float("nan")], [2.0, 3.0]]), label="x", positive=False,
                                      axes=("row", "column"))
    assert "1 offending entry of 4" in text and "row 0, column 1" in text


def test_successful_host_check_never_calls_the_detail() -> None:
    called: List[int] = []
    with deferred_host_checks():
        host_check(torch.tensor([True]), lambda host: None, detail=lambda: called.append(1) or "x")
    assert called == []


# ------------------------------------------------------------------------------------------------ items 5, 15

def test_metrics_rows_carry_their_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_DECAY", 0.0)
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 1)
    monkeypatch.setattr(T, "DIVERGENCE_LOSS_SPIKE_MARGIN", 0.0)
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_COOLDOWN", 1)
    _run(_cfg(), tmp_path / "run", n_steps=8, log_interval=2, eval_interval=4)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    kinds = {int(row["step"]): row["row_kind"] for row in rows}
    assert kinds[4] == "eval" and kinds[8] == "eval" and kinds[2] == "log" and kinds[6] == "log"
    assert set(kinds.values()) <= {"log", "eval", "spike"}
    assert all(kind == "spike" for step, kind in kinds.items() if step % 2)


def _probe_monitor(monitor: "T._DivergenceMonitor") -> tuple:
    reported, events = [], []
    for step in range(1, 401):
        loss = {200: 5.9, 260: 1136.0, 330: 27.8}.get(step, 4.3 + 0.01 * math.sin(step))
        status = {"ce": loss, "role_grad_norms": {}, "clip_coefs": {}}
        if monitor.observe(step, loss, status) is not None:
            reported.append(step)
        if monitor.event is not None:
            events.append(step)
    return reported, events


def test_monitor_reports_the_worst_spike_inside_the_cooldown_and_bounds_its_ema() -> None:
    monitor = T._DivergenceMonitor()
    reported, events = _probe_monitor(monitor)
    assert reported == [200, 260]                                           # 260 is 10x the severity of 200
    assert events == [200, 260, 330]                                        # 330 is recorded, not reported
    assert monitor._loss_ema < 4.4                                          # min(loss, EMA + margin) on spikes
    state = monitor.state_dict(interval_ends=False)
    restored = T._DivergenceMonitor()
    assert restored.load_state_dict(state) and restored._last_severity == monitor._last_severity
    legacy = dict(state)
    legacy.pop("last_severity")
    assert T._DivergenceMonitor().load_state_dict(legacy)                   # an older checkpoint still loads


def test_every_spike_event_reaches_the_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 1)
    monkeypatch.setattr(T, "DIVERGENCE_NORM_SPIKE_RATIO", 1e-9)          # every step's role norm spikes
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_COOLDOWN", 100)
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_ESCALATION", 1e30)
    _run(_cfg(), tmp_path / "run", n_steps=8, log_interval=4)
    path = tmp_path / "run" / "diagnostics" / T.DIVERGENCE_SPIKE_LOG
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(record["reported"] for record in records) == 1               # one reported inside the cooldown
    assert [record["step"] for record in records] == list(range(2, 9))     # every event after the warmup step


# ------------------------------------------------------------------------------------------------ item 14

def test_eigenvalue_and_table_step_alarms_warn_once_and_reach_the_sink(tmp_path: Path,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    records: List[Dict[str, Any]] = []
    warned: set = set()
    logger = logging.getLogger("audit-l3-warn")
    d = {"belief_lam_min": 2.0e-14, "belief_lam_max": 1.0,
         "relational_projected_cov_min_eigenvalue_layer_min": 6.6e-13,
         "relational_projected_cov_min_eigenvalue_argmin_layer": 1.0}
    with caplog.at_level(logging.WARNING, logger="audit-l3-warn"):
        for _ in range(2):
            T._warn_eigenvalue_collapse(d, warned, logger, belief_layer=2, floor_reference=1e-6,
                                        sink=records.append)
        assert T._warn_table_step_collapse(0.136, 7.2e-6, logger, step=1000, sink=records.append)
        assert not T._warn_table_step_collapse(0.1, 0.05, logger, step=1100, sink=records.append)
    assert [r["kind"] for r in records] == ["belief_eigenvalue_ratio", "projected_floor", "table_step_collapse"]
    assert records[0]["layer"] == 2 and records[1]["layer"] == 1
    healthy = {"belief_lam_min": 1e-2, "belief_lam_max": 1.0}
    T._warn_eigenvalue_collapse(healthy, set(), logger, belief_layer=0, sink=records.append)
    assert len(records) == 3
    path = T._append_run_warning(tmp_path, {**records[0], "value": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["value"] is None


def test_floor_reference_reads_the_prior_table() -> None:
    cfg = _cfg(predictive_covariance="moment_matched", predictive_covariance_floor_mode="prior")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    floor = T._covariance_floor_reference(model)
    if floor is None:
        pytest.skip("this tiny route has no diagonal prior table")
    from vfe3.numerics import bounded_variance_from_log
    bank = model.prior_bank
    expected = cfg.predictive_covariance_floor * float(
        bounded_variance_from_log(bank._prior_sigma_log_table().double(), eps=bank.eps).min())
    assert floor == pytest.approx(expected)
    assert T._covariance_floor_reference(VFEModel(_cfg(predictive_covariance_floor_mode="relative",
                                                       predictive_covariance="moment_matched"))) is None


# ------------------------------------------------------------------------------------------------ items 16, 17

def test_failed_step_is_skipped_counted_and_saved_under_the_opt_in(tmp_path: Path) -> None:
    cfg = _cfg(max_failed_step_skips=1)
    calls: List[int] = []
    losses, artifacts, model = _run(cfg, tmp_path / "run", n_steps=5, log_interval=1,
                                    model_hook=_fail_forward_at(calls, fail_on={2}))
    assert math.isnan(losses[2]) and all(math.isfinite(value) for i, value in enumerate(losses) if i != 2)
    assert artifacts.realized_updates["accepted_updates"] == 4
    assert (tmp_path / "run" / "diagnostics" / "failed_steps" / "step_3.pt").is_file()
    rows = {int(row["step"]): row for row in _rows(tmp_path / "run" / "metrics.csv")}
    assert rows[3]["failed_step_skips"] == "1" and rows[2]["failed_step_skips"] == "0"
    warnings = [json.loads(line) for line in
                (tmp_path / "run" / T.RUN_WARNINGS_LOG).read_text(encoding="utf-8").splitlines()]
    assert any(w["kind"] == "failed_step_skip" and w["step"] == 3 for w in warnings)
    calls.clear()
    with pytest.raises(ValueError):                                         # the second failure exceeds the budget
        _run(cfg, tmp_path / "over", n_steps=5, model_hook=_fail_forward_at(calls, fail_on={1, 3}))


def test_default_still_raises_on_a_failed_step(tmp_path: Path) -> None:
    calls: List[int] = []
    with pytest.raises(ValueError):
        _run(_cfg(), tmp_path / "run", n_steps=3, model_hook=_fail_forward_at(calls, fail_on={1}))


def test_consecutive_skips_abort_through_the_crash_path(tmp_path: Path) -> None:
    cfg = _cfg(max_failed_step_skips=5, max_consecutive_skips=2)
    calls: List[int] = []
    with pytest.raises(T.ConsecutiveSkipAbort):
        _run(cfg, tmp_path / "run", n_steps=6, model_hook=_fail_forward_at(calls, fail_on={1, 2}))
    record = json.loads((tmp_path / "run" / "diagnostics" / T.CRASH_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["exception"]["type"] == "ConsecutiveSkipAbort" and record["checkpoint_step"] == 3


def test_failed_evaluation_reports_nan_validation_and_keeps_the_best(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    original = T.evaluate
    seen: List[int] = []

    def failing(*args: Any, **kwargs: Any) -> Any:
        if isinstance(args[1], DataLoader):                                 # the validation evaluation only
            seen.append(1)
            if len(seen) == 2:
                raise RuntimeError("injected evaluation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(T, "evaluate", failing)
    _, artifacts, _ = _run(_cfg(max_failed_step_skips=1), tmp_path / "run", n_steps=4, eval_interval=2)
    rows = {int(row["step"]): row for row in _rows(tmp_path / "run" / "metrics.csv")}
    assert rows[4]["val_ce"] == "" and rows[2]["val_ce"] != ""
    assert artifacts.best_step == 2


def _step_state(model: VFEModel) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_loss_spike_threshold_rejects_the_update_and_is_inert_above_the_loss() -> None:
    cfg = _cfg()
    tokens, targets = next(iter(_loaders()[0]))
    results = []
    for threshold in (None, 1e9, 0.0):
        torch.manual_seed(5)
        model = VFEModel(cfg)
        optimizer = T.build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        before = _step_state(model)
        status: Dict[str, Any] = {}
        torch.manual_seed(11)
        loss = T.train_step(model, optimizer, scheduler, tokens, targets, status_out=status,
                            **({} if threshold is None else {"loss_skip_threshold": threshold}))
        results.append((loss, status, before, _step_state(model), scheduler.last_epoch))
    (_, plain, _, after_plain, _), (_, high, _, after_high, _), (loss, low, before, after_low, epoch) = results
    assert not high["loss_spike_skip"] and high["did_step"] and plain["did_step"]
    assert all(torch.equal(after_plain[name], after_high[name]) for name in after_plain)   # inert above the loss
    assert low["loss_spike_skip"] and not low["did_step"] and not low["nonfinite_skip"] and loss > 0.0
    assert all(torch.equal(before[name], after_low[name]) for name in before) and epoch == 0


def test_loss_spike_rejections_are_counted_in_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    thresholds: List[float] = []

    def threshold(self: Any, margin: float) -> Optional[float]:            # every step after the second spikes
        thresholds.append(margin)
        return 0.0 if len(thresholds) > 2 else None
    monkeypatch.setattr(T._DivergenceMonitor, "loss_skip_threshold", threshold)
    _, artifacts, _ = _run(_cfg(loss_spike_skip_nats=0.5), tmp_path / "run", n_steps=6, log_interval=3)
    assert set(thresholds) == {0.5}
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [row["loss_spike_rejected_updates"] for row in rows] == ["1", "4"]
    assert artifacts.realized_updates["accepted_updates"] == 2
    _, plain, _ = _run(_cfg(), tmp_path / "plain", n_steps=3, log_interval=3)
    assert len(thresholds) == 6                                             # the default never asks for a threshold


# ------------------------------------------------------------------------------------------------ item 21

def _dense_frame_class(model: VFEModel, belief: BeliefState) -> Dict[str, float]:
    r"""The pre-remediation dense float64 computation of ``_frame_class_diagnostics``."""
    from vfe3 import metrics
    group = model.group
    U = torch.linalg.matrix_exp(group.embed_algebra(belief.phi.double(), dtype=torch.float64))
    if belief.right_phi is not None:
        U = U @ torch.linalg.matrix_exp(group.embed_algebra(belief.right_phi.double(), dtype=torch.float64))
    U = U.reshape(-1, *U.shape[-2:])
    offsets = [0, *itertools.accumulate(group.irrep_dims)]
    head_cond = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        singular = torch.linalg.svdvals(U[:, start:stop, start:stop])
        head_cond.append(singular[..., 0] / singular[..., -1])
    vertex = torch.stack(head_cond, dim=-1).amax(-1)
    invariant = metrics.group_gauge_invariant(U, group)
    heads = metrics.per_head_gauge_invariants(U, group.irrep_dims)
    out = {"vertex_cond_median": float(torch.quantile(vertex, 0.5)), "vertex_cond_p95": float(torch.quantile(vertex, 0.95)),
           "vertex_cond_p99": float(torch.quantile(vertex, 0.99)), "vertex_cond_max": float(vertex.max()),
           "gauge_invariant_mean": float(invariant.mean()), "gauge_invariant_spread": float(invariant.std(unbiased=False)),
           "gauge_head_aniso_mean": float(heads["anisotropy"].mean()),
           "gauge_head_logdet_spread": float(heads["logdet"].std(unbiased=False))}
    if group.name == "block_glmc":
        out["gauge_head_phase_abs_mean"] = float(metrics.per_head_complex_phase(U, group.irrep_dims).abs().mean())
    return out


@pytest.mark.parametrize("overrides", [dict(gauge_group="block_glk"),
                                       dict(gauge_group="block_glmc", embed_dim=8)])
@pytest.mark.parametrize("right", [False, True])
def test_frame_class_diagnostics_equal_the_dense_float64_computation(overrides: dict, right: bool) -> None:
    torch.manual_seed(0)
    model = VFEModel(_cfg(**overrides))
    n_gen = model.group.generator_shape[0]
    K = sum(model.group.irrep_dims)
    rows = 0.3 * torch.randn(5, n_gen)
    pick = torch.tensor([[0, 1, 2, 1], [3, 0, 4, 0], [2, 2, 1, 3]])                 # repeated frame rows
    phi = rows[pick]                                                                # (3, 4, n_gen)
    right_phi = (0.2 * torch.randn(5, n_gen))[pick] if right else None
    belief = BeliefState(mu=torch.randn(3, 4, K), sigma=torch.rand(3, 4, K) + 0.5, phi=phi, right_phi=right_phi)
    got = model._frame_class_diagnostics(belief)
    want = _dense_frame_class(model, belief)
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-11), key


# ------------------------------------------------------------------------------------------------ item 34

def test_dead_router_names_every_head_above_threshold(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("audit-l3-router")
    d = {"relational_self_weight_head_max": 1.0, "relational_self_lock_argmax_layer": 1.0,
         "relational_self_lock_argmax_head": 0.0, "relational_self_lock_unit_distance": 0.01}
    tables = {"self_weight": torch.tensor([[0.2, 0.3], [1.0, 0.95], [0.5, 1.0]]),
              "unit_distance": torch.tensor([[0.0, 0.0], [0.01, 0.02], [0.0, 0.03]])}
    records: List[Dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="audit-l3-router"):
        T._warn_dead_router(d, set(), logger, head_tables=tables, sink=records.append)
    assert [(r["layer"], r["head"]) for r in records] == [(1, 0), (1, 1), (2, 1)]
    assert "0.02" in records[1]["message"]
    torch.manual_seed(0)
    model = VFEModel(_cfg(n_layers=2))
    model.diagnostics(next(iter(_loaders()[0]))[0])
    assert model.last_router_head_tables["self_weight"].shape == (2, 2)
    assert model.last_router_head_tables["unit_distance"].shape[0] == 2


# ------------------------------------------------------------------------------------------------ item 35

def test_block_variance_rows_count_nonfinite_and_report_scale_and_trace_ratio() -> None:
    torch.manual_seed(0)
    model = VFEModel(_cfg(norm_type_block="layernorm"))
    windows = torch.randint(0, V, (3, N), generator=torch.Generator().manual_seed(2))
    variances: dict = {}
    T._residual_and_attention_rows(model, windows, 7, CPU, variance_out=variances)
    rows = T._block_variance_rows(variances, 7)
    assert all(set(row) == set(T.BLOCK_VARIANCE_FIELDS) for row in rows)
    assert all(row["n_entries"] == 3 * N * 4 and row["n_nonfinite"] == 0 for row in rows)
    assert all(row["scale_median"] > 0 and row["trace_ratio_median"] > 0 for row in rows)
    mu = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
    sigma = torch.tensor([[0.5, 1.0, 2.0, float("inf")]])
    slot: dict = {}
    capture = {"layer_converged": [BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(1, 1))]}
    T._collect_block_variances(capture, slot)
    (row,) = T._block_variance_rows(slot, 1)
    assert row["n_entries"] == 4 and row["n_nonfinite"] == 1 and row["var_max"] == 2.0
    centered = mu - mu.mean()
    s = float(centered.square().mean().sqrt())
    assert row["scale_median"] == pytest.approx(s)
    finite_sigma = torch.tensor([[0.5, 1.0, 2.0, 4.0]])
    slot = {}
    T._collect_block_variances({"layer_converged": [BeliefState(mu=mu, sigma=finite_sigma, phi=torch.zeros(1, 1))]},
                               slot)
    (row,) = T._block_variance_rows(slot, 1)
    projected = float(finite_sigma.sum() * (1 - 1 / 4))
    assert row["trace_ratio_median"] == pytest.approx(projected / (4 * s * s))


def test_block_variance_at_log_writes_log_step_rows(tmp_path: Path) -> None:
    _run(_cfg(block_variance_at_log=True), tmp_path / "run", n_steps=4, log_interval=2, eval_interval=4)
    steps = {int(row["step"]) for row in _rows(tmp_path / "run" / "diagnostics" / "block_variance.csv")}
    assert steps == {2, 4}


# ------------------------------------------------------------------------------------------------ item 47

class _Bank(torch.nn.Module):
    def __init__(self, log_s: torch.Tensor) -> None:
        super().__init__()
        self.log_s, self.eps = log_s, 1e-12

    def _prior_sigma_log_table(self) -> torch.Tensor:
        return self.log_s


@pytest.mark.parametrize("magnitude", [1e17, 1e35])
def test_table_fisher_step_survives_huge_finite_gradients(magnitude: float) -> None:
    V_, K_, radius = 3, 4, 0.5
    mu = torch.nn.Parameter(torch.zeros(V_, K_))
    log_s = torch.full((V_, K_), math.log(5.0))
    optimizer = TableFisherAdamW([{"params": [mu], "lr": 1.0, "weight_decay": 0.0, "table_fisher": "mu"}],
                                 TableFisherStatistics(_Bank(log_s)), table_fisher_lr=1.0,
                                 table_fisher_trust_radius=radius)
    optimizer._collect_table_diag = True
    grad = torch.zeros(V_, K_)
    grad[0] = magnitude                                                     # row 0: one occurrence of N = 8192
    grad[1] = 1e-3                                                          # row 1: an ordinary gradient
    mu.grad = grad
    tokens = torch.tensor([[0, 1]])
    optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(8192))
    optimizer.step()
    moved = mu.detach()
    assert bool(torch.isfinite(moved).all())
    fisher_norm = float((moved[0].square() / 5.0).sum().sqrt())
    assert fisher_norm == pytest.approx(radius, rel=1e-5)                   # the full trust step, not frozen
    assert math.isfinite(optimizer._table_diag["table_fisher_step_norm_mean"])
    # The ordinary row takes exactly the historical float32 step.
    step = -1.0 * (8192 / 1) * 5.0 * 1e-3
    norm = abs(step) / math.sqrt(5.0) * math.sqrt(K_)
    expected = torch.full((K_,), step * min(1.0, radius / norm))
    assert torch.allclose(moved[1], expected, rtol=1e-6)


def test_parameter_nonfinite_count() -> None:
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
        model.bias[1] = float("inf")
    assert T._parameter_nonfinite_count(model) == 2


# ------------------------------------------------------------------------------------------------ item 48

def test_checkpoints_are_pruned_timed_and_written_at_the_end(tmp_path: Path) -> None:
    cfg = _cfg(checkpoint_interval=2, checkpoint_keep_last=2)
    _run(cfg, tmp_path / "run", n_steps=8, log_interval=2)
    names = sorted(path.name for path in (tmp_path / "run" / "checkpoints").glob("step_*.pt"))
    assert names == ["step_6.pt", "step_8.pt"]
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert all(float(row["checkpoint_ms"]) >= 0.0 for row in rows)
    straight, _, _ = _run(_cfg(), tmp_path / "straight", n_steps=6)
    ended, _, _ = _run(_cfg(checkpoint_at_end=True), tmp_path / "end", n_steps=3)
    assert ended == straight[:3]                                            # the checkpoint changes no step
    end = tmp_path / "end" / "checkpoints" / "step_3.pt"
    resumed, _, _ = _run(_cfg(), tmp_path / "resumed", n_steps=6, resume_from=end)
    assert resumed == straight[3:]


# ------------------------------------------------------------------------------------------------ item 49

def test_peak_memory_split_separates_the_non_training_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    peak = {"value": 0.0}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: peak["value"] * 1e6)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: peak.update(value=0.0))
    memory = T._PeakMemory(torch.device("cuda"))
    peak["value"] = 100.0                                                    # training steps
    memory.begin_other()
    peak["value"] = 300.0                                                    # the diagnostics forward
    memory.end_other()
    peak["value"] = 120.0                                                    # the train step
    assert memory.sample() == (300.0, 120.0)
    assert all(math.isnan(value) for value in T._PeakMemory(CPU).sample())


# ------------------------------------------------------------------------------------------------ item 68

def test_background_best_save_publishes_after_the_wait(tmp_path: Path) -> None:
    cfg = _cfg()
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    assert artifacts.maybe_save_best(3, model, 10.0, background=True)
    artifacts.wait_background()
    assert artifacts.best_step == 3 and artifacts.best_val_ppl == 10.0 and artifacts.best_path.is_file()
    bundle = torch.load(artifacts.best_path, weights_only=False)
    for name, tensor in model.state_dict().items():
        assert torch.equal(bundle["model_state"][name], tensor)
    order: List[str] = []
    artifacts.submit_figure_work("a", lambda: order.append("a"))
    artifacts.submit_figure_work("b", lambda: order.append("b"))
    artifacts.wait_background()
    assert order == ["a", "b"]


# ------------------------------------------------------------------------------------------------ item 70

def test_clip_multiply_is_skipped_only_when_the_coefficient_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    tokens, targets = next(iter(_loaders()[0]))
    calls: List[int] = []
    original = torch.nn.utils.clip_grads_with_norm_

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", counting)
    after = {}
    for clip in (None, 1e6, 1e-6):
        torch.manual_seed(5)
        model = VFEModel(cfg)
        optimizer = T.build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        calls.clear()
        torch.manual_seed(11)
        T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=clip, status_out={})
        after[clip] = (_step_state(model), len(calls))
    assert after[1e6][1] == 0 and after[1e-6][1] > 0
    assert all(torch.equal(after[None][0][name], after[1e6][0][name]) for name in after[None][0])


def test_one_foreach_norm_gives_the_per_tensor_finiteness_verdict() -> None:
    grads = [torch.randn(3), torch.randn(2, 2), torch.empty(0)]
    assert bool(T._gradients_finite(grads))
    grads[1][0, 1] = float("inf")
    assert not bool(T._gradients_finite(grads))
    assert not bool(T._gradients_finite([torch.tensor([float("nan")])]))
