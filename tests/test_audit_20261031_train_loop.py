r"""Training-loop remediation of the 2026-10-31 ultradeep audit, package ``train-loop``.

A034 (train-loop-lifecycle-1): after ``LOSS_SPIKE_REBASELINE_REJECTIONS`` consecutive ``loss_spike_skip_nats``
rejections the next update is accepted and its loss re-baselines the loss EMA, so a level shift left by an accepted
update no longer rejects every later update; the run length travels with the checkpoint. A067
(train-loop-lifecycle-4): a failure of the eval-protocol training CE is a counted diagnostic failure with a blank
``generalization_gap``. A076 (baseline-generation-6): the reported-spike forward probe runs on a model without beliefs,
and the spike attribution lists ``per_type_table`` rows. A098 (train-loop-lifecycle-3): a consecutive-skip abort
checkpoints a monitor that observed the aborted step. A100 (training-diagnostics-3): the projected-floor reference is
read at every row. A170 (train-loop-lifecycle-5): table row sums are formed in bounded blocks, and a sustained shift
attributes only its reported and record-breaking events. A202 (train-loop-lifecycle-6): a terminal-callback cell
writes its terminal checkpoint once. A317 (perf-live-path-5): the timed step is split into forward, backward,
optimizer, post-update and remaining time, by marks that leave the step bitwise unchanged. A337
(gap-mixed-precision-routes-6): the precision ladder replays a raised step under an enabled GradScaler.

Tiny CPU models only (K = 4 or 8, two heads, N = 4, vocabulary 7 or 11, a handful of steps).
"""
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import live_config, test_config
from vfe3.baseline import BaselineConfig, BaselineTransformer
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.timing import STEP_TAIL_PHASE, TrainingTimer

B, N, V = 2, 4, 7
CPU = torch.device("cpu")
LOGGER = logging.getLogger("audit-20261031-train-loop")


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


def _run(cfg: VFE3Config, run_dir: Optional[Path], *, n_steps: int, log_interval: int = 0, eval_interval: int = 0,
         resume_from: Any = None, terminal_callback: Any = None) -> tuple:
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model) if run_dir is not None else None
    train_loader, val_loader = _loaders()
    callback = (None if terminal_callback is None
                else terminal_callback(model, artifacts, cfg, train_loader, val_loader))
    torch.manual_seed(11)
    losses = T.train(model, train_loader, cfg, n_steps=n_steps, grad_clip=cfg.grad_clip,
                     log_interval=log_interval, eval_interval=eval_interval, val_loader=val_loader, device=CPU,
                     logger=LOGGER, artifacts=artifacts, generate_samples=False, resume_from=resume_from,
                     terminal_callback=callback)
    return losses, artifacts, model


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


# ------------------------------------------------------------------------------------------------ A034

def _shift_level_after(step_index: int, monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    r"""Wrap the loop's step: after the accepted update of 0-based step ``step_index`` the readout bias moves by
    20 standard normals (an accepted update that raises the loss level by tens of nats). Records every step."""
    record: Dict[str, List[Any]] = {"step": [], "did_step": [], "loss_spike_skip": [], "after_shift": []}
    real = T._train_step_with_precision_retry

    def wrapped(model: Any, optimizer: Any, scheduler: Any, tokens: Any, targets: Any, *, step: int,
                **kwargs: Any) -> Any:
        out = real(model, optimizer, scheduler, tokens, targets, step=step, **kwargs)
        status = kwargs["status_out"]
        record["step"].append(step)
        record["did_step"].append(bool(status["did_step"]))
        record["loss_spike_skip"].append(bool(status.get("loss_spike_skip")))
        if step == step_index:
            with torch.no_grad():
                bias = dict(model.named_parameters())["prior_bank.output_proj_bias"]
                bias.add_(torch.randn(bias.shape, generator=torch.Generator().manual_seed(0)) * 20.0)
            record["after_shift"].append(_parameters(model))
        return out

    monkeypatch.setattr(T, "_train_step_with_precision_retry", wrapped)
    return record


def test_a_level_shift_ends_in_an_accepted_rebaselining_update(monkeypatch: pytest.MonkeyPatch,
                                                               caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 3)
    monkeypatch.setattr(T, "LOSS_SPIKE_REBASELINE_REJECTIONS", 4)
    record = _shift_level_after(5, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        losses, _, model = _run(_cfg(loss_spike_skip_nats=1.0, max_steps=14), None, n_steps=14)
    assert record["did_step"][:6] == [True] * 6
    assert record["loss_spike_skip"][6:10] == [True] * 4                    # the shift is rejected four times
    assert record["did_step"][10] and not record["loss_spike_skip"][10]     # then accepted without the threshold
    assert sum(record["did_step"][6:]) >= 2                                 # and later updates pass the new EMA
    after = record["after_shift"][0]
    assert any(not torch.equal(after[name], p.detach()) for name, p in model.named_parameters())
    assert any("re-baselined" in r.getMessage() and "step 11" in r.getMessage() for r in caplog.records)
    assert all(math.isfinite(value) for value in losses)


def test_rebaseline_sets_the_loss_ema_and_the_next_threshold() -> None:
    monitor = T._DivergenceMonitor()
    for step in range(1, 61):
        monitor.observe(step, 2.0, {"ce": 2.0, "role_grad_norms": {}, "clip_coefs": {}})
    assert monitor.loss_skip_threshold(1.0) == pytest.approx(3.0)
    monitor.observe(61, 30.0, {"ce": 30.0, "role_grad_norms": {}, "clip_coefs": {}})
    assert monitor.loss_skip_threshold(1.0) == pytest.approx(3.0)            # frozen on the spike
    monitor.rebaseline_loss(30.0)
    assert monitor.loss_skip_threshold(1.0) == 31.0 and monitor._loss_seen == 61
    monitor.rebaseline_loss(float("nan"))                                   # a non-finite loss changes nothing
    assert monitor.loss_skip_threshold(1.0) == 31.0


def test_the_rejection_run_is_checkpointed_and_a_resume_continues_it(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 3)
    monkeypatch.setattr(T, "LOSS_SPIKE_REBASELINE_REJECTIONS", 4)
    cfg = _cfg(loss_spike_skip_nats=1.0, max_steps=14, checkpoint_interval=8)
    record = _shift_level_after(5, monkeypatch)
    straight, _, _ = _run(cfg, tmp_path / "straight", n_steps=14)
    straight_did = list(record["did_step"])
    checkpoint = tmp_path / "straight" / "checkpoints" / "step_8.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["divergence_monitor"]["consecutive_loss_spike_rejections"] == 2   # steps 7 and 8 rejected
    record["did_step"].clear()
    resumed, _, _ = _run(cfg, tmp_path / "resumed", n_steps=14, resume_from=checkpoint)
    assert resumed == straight[8:]
    assert record["did_step"] == straight_did[8:] and record["did_step"][2]          # step 11 accepted in both


def test_a_resume_after_a_consecutive_skip_abort_makes_the_accepted_update(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 3)
    cfg = _cfg(loss_spike_skip_nats=8.0, max_consecutive_skips=2, max_steps=12)   # the abort bound is the smaller
    record = _shift_level_after(5, monkeypatch)
    with pytest.raises(T.ConsecutiveSkipAbort):
        _run(cfg, tmp_path / "aborted", n_steps=12)
    assert record["loss_spike_skip"][6:] == [True, True]                     # aborted at the second rejection
    checkpoint = tmp_path / "aborted" / "checkpoints" / "step_8.pt"
    assert torch.load(checkpoint, weights_only=True)["divergence_monitor"]["consecutive_loss_spike_rejections"] == 2
    record["did_step"].clear()
    record["loss_spike_skip"].clear()
    resumed, _, _ = _run(cfg, tmp_path / "resumed", n_steps=12, resume_from=checkpoint)
    assert len(resumed) == 4 and record["did_step"][0] and not record["loss_spike_skip"][0]


def test_the_rejection_run_record_is_tolerant_of_older_checkpoints() -> None:
    monitor = T._DivergenceMonitor()
    record = T._divergence_checkpoint_record(monitor, interval_ends=False, skipped_steps=3, consecutive_skips=3,
                                             last_group_norms={}, loss_spike_rejections=3,
                                             consecutive_loss_spike_rejections=3)
    assert T._checkpoint_divergence_state(record)["consecutive_loss_spike_rejections"] == 3
    legacy = {key: value for key, value in record.items() if key != "consecutive_loss_spike_rejections"}
    assert T._checkpoint_divergence_state(legacy)["consecutive_loss_spike_rejections"] == 0
    assert T._checkpoint_divergence_state({**record, "consecutive_loss_spike_rejections": -1}) is not None


# ------------------------------------------------------------------------------------------------ A098

def test_a_consecutive_skip_abort_checkpoints_a_monitor_that_observed_the_aborted_step(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def threshold(self: Any, margin: float) -> Optional[float]:            # step 4's loss is above it: rejected
        calls["n"] += 1
        return 0.0 if calls["n"] == 4 else None

    monkeypatch.setattr(T._DivergenceMonitor, "loss_skip_threshold", threshold)
    with pytest.raises(T.ConsecutiveSkipAbort):
        _run(_cfg(max_consecutive_skips=1, loss_spike_skip_nats=0.5), tmp_path / "run", n_steps=6)
    crash = json.loads((tmp_path / "run" / "diagnostics" / T.CRASH_RECORD_NAME).read_text(encoding="utf-8"))
    assert crash["exception"]["type"] == "ConsecutiveSkipAbort" and crash["checkpoint_step"] == 4
    payload = torch.load(tmp_path / "run" / "checkpoints" / "step_4.pt", weights_only=True)
    record = payload["divergence_monitor"]
    assert payload["step"] == 4 and payload["data_state"]["batches_consumed"] == 4
    assert record["skipped_steps"] == 1 and record["consecutive_skips"] == 1
    assert record["monitor"]["loss_seen"] == 4 and record["monitor"]["interval"]["steps"] == 4


# ------------------------------------------------------------------------------------------------ A067

def test_a_failed_eval_protocol_train_ce_is_counted_and_leaves_the_gap_blank(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: Any, **kwargs: Any) -> float:
        raise RuntimeError("injected eval-protocol forward failure")

    monkeypatch.setattr(T, "_eval_protocol_train_ce", failing)
    losses, artifacts, _ = _run(_cfg(eval_stride=2), tmp_path / "run", n_steps=4, log_interval=2,
                                eval_interval=2)
    assert len(losses) == 4 and all(math.isfinite(value) for value in losses)
    assert artifacts.diagnostics_failures["eval_protocol_train_ce"] == 2
    rows = {int(row["step"]): row for row in _rows(tmp_path / "run" / "metrics.csv")}
    for step in (2, 4):
        assert rows[step]["val_ce"] != "" and rows[step]["generalization_gap"] == "", step


# ------------------------------------------------------------------------------------------------ A076

def _baseline(**overrides: Any) -> BaselineTransformer:
    values: Dict[str, Any] = dict(vocab_size=11, embed_dim=8, n_heads=2, n_layers=1, mlp_expansion=2, max_seq_len=6,
                                  batch_size=2, max_steps=2, warmup_steps=0, log_interval=0, eval_interval=0,
                                  decode_unigram_prior=False)
    values.update(overrides)
    torch.manual_seed(0)
    return BaselineTransformer(BaselineConfig(**values))


def test_the_spike_forward_probe_reports_the_ce_of_a_model_without_beliefs() -> None:
    model = _baseline()
    tokens = torch.randint(0, 11, (2, 6), generator=torch.Generator().manual_seed(1))
    targets = torch.randint(0, 11, (2, 6), generator=torch.Generator().manual_seed(2))
    targets[1, -1] = -100
    rng = torch.get_rng_state()
    state = T._spike_forward_state(model, tokens, targets)
    assert torch.equal(torch.get_rng_state(), rng) and model.training
    assert state["sigma_q_max"] is None and state["sigma_q_at_ce_max"] is None
    model.eval()
    with torch.no_grad():
        _, logits = model.forward_beliefs(tokens, return_logits=True)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 11).float(), targets.reshape(-1), ignore_index=-100,
                                           reduction="none").reshape(2, 6)
    ce[targets == -100] = float("-inf")
    assert state["ce_max"] == pytest.approx(float(ce.max()), rel=1e-5)
    assert state["ce_max_at"] == [int(i) for i in divmod(int(ce.reshape(-1).argmax()), 6)]
    report = {"roles": {}, "top_parameters": [], "causes": [], "forward_state": state}
    summary = T._spike_summary(report)
    assert "max per-position CE" in summary and "sigma_q" not in summary


def test_the_spike_attribution_lists_per_type_table_rows() -> None:
    model = _baseline(per_type_table="block_read")
    cfg = model.cfg
    optimizer = model.build_optimizer(cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    tokens = torch.randint(0, 11, (2, 6), generator=torch.Generator().manual_seed(1))
    targets = torch.randint(0, 11, (2, 6), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        model.type_table.normal_(generator=torch.Generator().manual_seed(3))   # away from the identity start
    status: Dict[str, Any] = {}
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=None, status_out=status)
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("phi",), loss_spiked=False,
                                  replayed=False, last_group_norms={}, targets=targets)
    assert [entry["name"] for entry in report["top_parameters"]] == ["type_table"]
    entry = report["table_rows"]["type_table"]
    counts = torch.bincount(tokens.reshape(-1), minlength=11)
    assert entry["indexed_by"] == "token"
    assert entry["batch_counts"] == [int(counts[row]) for row in entry["rows"]]
    assert all(counts[row] > 0 for row in entry["rows"][:entry["rows_with_gradient"]])


# ------------------------------------------------------------------------------------------------ A100

def test_the_projected_floor_reference_is_read_at_every_row(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    references = iter([1e-6, 5e-7, 2.5e-7, 1.25e-7])
    calls: List[float] = []
    seen: List[Optional[float]] = []

    def reference(model: Any) -> Optional[float]:
        calls.append(next(references))
        return calls[-1]

    real_warn = T._warn_eigenvalue_collapse

    def warn(*args: Any, floor_reference: Optional[float] = None, **kwargs: Any) -> None:
        seen.append(floor_reference)
        return real_warn(*args, floor_reference=floor_reference, **kwargs)

    monkeypatch.setattr(T, "_covariance_floor_reference", reference)
    monkeypatch.setattr(T, "_warn_eigenvalue_collapse", warn)
    _run(_cfg(), tmp_path / "run", n_steps=3, log_interval=1)
    assert seen == [1e-6, 5e-7, 2.5e-7] and len(calls) == 3


def test_a_lowered_prior_table_lowers_the_next_rows_reference(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(predictive_covariance="moment_matched", predictive_covariance_floor_mode="prior")
    torch.manual_seed(5)
    if T._covariance_floor_reference(VFEModel(cfg)) is None:
        pytest.skip("this tiny route has no diagonal prior table")
    seen: List[float] = []
    real_warn = T._warn_eigenvalue_collapse

    def warn(*args: Any, floor_reference: Optional[float] = None, **kwargs: Any) -> None:
        seen.append(floor_reference)
        return real_warn(*args, floor_reference=floor_reference, **kwargs)

    real_step = T.train_step
    calls = {"n": 0}

    def lowering(model: Any, *args: Any, **kwargs: Any) -> float:
        loss = real_step(model, *args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:                                                 # between the rows of steps 1 and 2
            with torch.no_grad():
                model.prior_bank.sigma_log_embed.sub_(3.0)                  # every prior variance falls
        return loss

    monkeypatch.setattr(T, "_warn_eigenvalue_collapse", warn)
    monkeypatch.setattr(T, "train_step", lowering)
    _, _, model = _run(cfg, tmp_path / "run", n_steps=2, log_interval=1)
    assert len(seen) == 2 and seen[1] < 0.5 * seen[0]
    assert seen[1] == pytest.approx(T._covariance_floor_reference(model), rel=1e-12)   # the table at row 2


# ------------------------------------------------------------------------------------------------ A170

def test_row_square_sums_match_the_full_table_in_bounded_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    grad = torch.randn(7, 3, 5, generator=torch.Generator().manual_seed(0))       # (V, ...) rows of 15 entries
    expected = grad.flatten(1).to(torch.float64).square().sum(dim=1)
    for chunk in (1 << 22, 32, 4):                          # one block; two rows and a remainder; one row each
        monkeypatch.setattr(T, "UPDATE_NORM_CHUNK_ELEMENTS", chunk)
        rows = T._row_square_sums(grad)
        assert rows.dtype == torch.float64 and rows.shape == (7,)
        assert torch.allclose(rows, expected, rtol=1e-12, atol=0.0), chunk
    assert T._row_square_sums(torch.zeros(0, 4)).shape == (0,)


def test_a_sustained_shift_attributes_only_reported_and_record_breaking_events() -> None:
    monitor = T._DivergenceMonitor()
    events = reported = attributed = 0
    for step in range(1, 961):                                              # the lane's 20x phi-norm shift
        norm = 1.0 if step <= 60 else 20.0
        spike = monitor.observe(step, 5.0, {"ce": 5.0, "role_grad_norms": {"mu": 1.0, "phi": norm}, "clip_coefs": {}})
        events += monitor.event is not None
        reported += spike is not None
        attributed += monitor.event_attributed
    assert events == 900 and reported == 9 and attributed == 9
    rising = T._DivergenceMonitor()
    flags = []
    for step in range(1, 81):
        norm = 1.0 if step <= 60 else 20.0 * (step - 59)                    # every event a new maximum
        rising.observe(step, 5.0, {"ce": 5.0, "role_grad_norms": {"phi": norm}, "clip_coefs": {}})
        if step > 60:
            flags.append(rising.event_attributed)
    assert all(flags)


def test_the_attribution_peak_round_trips_and_older_records_restart_it() -> None:
    monitor = T._DivergenceMonitor()
    for step in range(1, 70):
        norm = 1.0 if step <= 60 else 30.0 + step
        monitor.observe(step, 5.0, {"ce": 5.0, "role_grad_norms": {"phi": norm}, "clip_coefs": {}})
    state = monitor.state_dict(interval_ends=False)
    assert state["attributed_peak"] > state["last_severity"]
    restored = T._DivergenceMonitor()
    assert restored.load_state_dict(state) and restored.state_dict(interval_ends=False) == state
    legacy = {key: value for key, value in state.items() if key != "attributed_peak"}
    older = T._DivergenceMonitor()
    assert older.load_state_dict(legacy) and older._attributed_peak == state["last_severity"]
    assert not T._DivergenceMonitor().load_state_dict({**state, "attributed_peak": "x"})


def test_unattributed_events_are_recorded_without_an_attribution(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 1)
    monkeypatch.setattr(T, "DIVERGENCE_NORM_SPIKE_RATIO", 1e-9)          # every step's role norm spikes
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_COOLDOWN", 100)
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_ESCALATION", 1e30)
    real = T._spike_attribution
    attributed_steps: List[int] = []

    def counting(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        attributed_steps.append(int(kwargs["step"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(T, "_spike_attribution", counting)
    _run(_cfg(), tmp_path / "run", n_steps=8, log_interval=4)
    path = tmp_path / "run" / "diagnostics" / T.DIVERGENCE_SPIKE_LOG
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["step"] for record in records] == list(range(2, 9))     # every event is still recorded
    peak = -math.inf
    for record in records:
        due = record["reported"] or record["severity"] > peak
        if due:
            peak = record["severity"]                                       # a report restarts the burst peak
        assert ("causes" in record) == due, record["step"]
        if not due:
            assert record["attribution"].startswith("skipped")
    assert attributed_steps == [record["step"] for record in records if "causes" in record]


# ------------------------------------------------------------------------------------------------ A202

def _finalizer(model: Any, artifacts: Any, cfg: VFE3Config, train_loader: Any, val_loader: Any) -> Any:
    def finalize(state: T.TrainingTerminalState, losses: List[float]) -> None:
        ra.finalize_validation_run(model, artifacts, cfg, val_loader, losses=losses, train_loader=train_loader,
                                   terminal_state=state, device=CPU, logger=LOGGER)
    return finalize


def test_a_terminal_callback_cell_writes_its_terminal_checkpoint_once(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    real = ra.RunArtifacts.save_checkpoint

    def recording(self: ra.RunArtifacts, step: int, *args: Any, **kwargs: Any) -> Path:
        calls.append(int(step))
        return real(self, step, *args, **kwargs)

    monkeypatch.setattr(ra.RunArtifacts, "save_checkpoint", recording)
    cfg = _cfg(max_steps=4, checkpoint_at_end=True, checkpoint_keep_last=1)  # the ablation cell's settings
    _run(cfg, tmp_path / "cell", n_steps=4, terminal_callback=_finalizer)
    assert calls == [4]
    payload = torch.load(tmp_path / "cell" / "checkpoints" / "step_4.pt", weights_only=True)
    assert payload["best_model_bundle"] is not None and payload["best_step"] == 4   # the finalizer's checkpoint
    calls.clear()
    _run(cfg, tmp_path / "plain", n_steps=4)                                  # no callback: the loop still saves
    assert calls == [4]


# ------------------------------------------------------------------------------------------------ A317

def test_the_cpu_timer_splits_each_step_into_marked_segments() -> None:
    ticks = iter([0.0,                                                      # construction
                  10.0, 11.0, 13.0, 16.0, 20.0, 25.0,                       # step 1: start, 4 marks, finish
                  30.0, 31.0, 33.0, 34.0,                                   # step 2: start, forward, backward, finish
                  40.0])                                                    # the pipeline sample
    timer = TrainingTimer(CPU, clock=lambda: next(ticks))
    timer.start_step()
    for phase in ("forward", "backward", "optimizer", "post_update"):
        timer.mark(phase)
    timer.finish_step(n_tokens=8)
    timer.start_step()
    timer.mark("forward")
    timer.mark("backward")
    timer.finish_step(n_tokens=8)
    sample = timer.sample_train_window()
    assert sample.train_step_ms_mean == pytest.approx(9500.0)              # (15 + 4) s over 2 steps
    assert sample.phase_ms_mean == pytest.approx({"forward": 1000.0, "backward": 2000.0, "optimizer": 1500.0,
                                                  "post_update": 2000.0, STEP_TAIL_PHASE: 3000.0})
    assert sum(sample.phase_ms_mean.values()) == pytest.approx(sample.train_step_ms_mean)
    timer.mark("forward")                                                   # outside a step: no clock read
    assert timer.sample_pipeline_window().wall_clock_s == 40.0


class _FakeEvent:
    r"""A CUDA event stand-in on a shared host counter: ``record`` stamps the counter, ``synchronize`` is counted."""
    clock = {"t": 0.0, "syncs": 0}

    def __init__(self, enable_timing: bool = False) -> None:
        self.stamp: Optional[float] = None

    def record(self, stream: object = None) -> None:
        self.stamp = self.clock["t"]

    def synchronize(self) -> None:
        self.clock["syncs"] += 1

    def elapsed_time(self, end: "_FakeEvent") -> float:
        return float(end.stamp - self.stamp)


def test_the_cuda_timer_reads_its_marks_only_at_the_window_sample() -> None:
    _FakeEvent.clock.update(t=0.0, syncs=0)
    timer = TrainingTimer(torch.device("cuda"), cuda_event_factory=_FakeEvent, cuda_stream_factory=lambda device: None)
    for step in range(2):
        _FakeEvent.clock["t"] = 100.0 * step
        timer.start_step()
        for phase, at in (("forward", 3.0), ("backward", 8.0), ("optimizer", 9.0), ("post_update", 10.0)):
            _FakeEvent.clock["t"] = 100.0 * step + at
            timer.mark(phase)
        _FakeEvent.clock["t"] = 100.0 * step + 12.0
        timer.finish_step(n_tokens=4)
    assert _FakeEvent.clock["syncs"] == 0                                   # marking never synchronizes
    sample = timer.sample_train_window()
    assert _FakeEvent.clock["syncs"] == 1
    assert sample.train_step_ms_mean == pytest.approx(12.0)
    assert sample.phase_ms_mean == pytest.approx({"forward": 3.0, "backward": 5.0, "optimizer": 1.0,
                                                  "post_update": 1.0, STEP_TAIL_PHASE: 2.0})


def test_phase_marks_leave_the_live_step_bitwise_unchanged() -> None:
    cfg = live_config()
    tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len),
                           generator=torch.Generator().manual_seed(1))
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len),
                            generator=torch.Generator().manual_seed(2))
    results = []
    marks: List[str] = []
    for marked in (False, True):
        torch.manual_seed(5)
        model = VFEModel(cfg)
        optimizer = T.build_optimizer(model, cfg)
        T._set_successful_update_count(optimizer, 0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        status: Dict[str, Any] = {}
        timer = TrainingTimer(CPU)
        timer.start_step()
        torch.manual_seed(11)
        loss = T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=cfg.grad_clip,
                            grad_clip_scope=cfg.grad_clip_scope, status_out=status,
                            **({"phase_mark": lambda phase: (marks.append(phase), timer.mark(phase))}
                               if marked else {}))
        grads = {name: status["grads"][id(p)].clone() for name, p in model.named_parameters()
                 if id(p) in status.get("grads", {})}
        results.append((loss, _parameters(model), grads))
    (loss_a, params_a, grads_a), (loss_b, params_b, grads_b) = results
    assert loss_a == loss_b
    assert params_a.keys() == params_b.keys() and all(torch.equal(params_a[k], params_b[k]) for k in params_a)
    assert grads_a and grads_a.keys() == grads_b.keys() and all(torch.equal(grads_a[k], grads_b[k]) for k in grads_a)
    assert marks == ["forward", "backward", "optimizer", "post_update"]


def test_metrics_rows_carry_the_step_phase_split(tmp_path: Path) -> None:
    _run(_cfg(), tmp_path / "run", n_steps=4, log_interval=2)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [2, 4]
    for row in rows:
        values = [float(row[column]) for column in T.TRAIN_STEP_PHASE_COLUMNS.values()]
        assert all(value >= 0.0 for value in values) and values[0] > 0.0 and values[1] > 0.0
        assert sum(values) == pytest.approx(float(row["train_step_ms_mean"]), rel=1e-6)


def test_phase_marks_leave_the_loop_bitwise_unchanged() -> None:
    cfg = live_config(max_steps=4, phi_type_dropout=0.0)                    # no frame-drop draw at log steps
    loader = DataLoader(TokenWindows((torch.arange(61) * 5 + 2) % cfg.vocab_size, cfg.max_seq_len),
                        batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    runs = []
    for log_interval in (0, 2):                                             # silent, and timed with phase marks
        torch.manual_seed(5)
        model = VFEModel(cfg)
        torch.manual_seed(11)
        losses = T.train(model, loader, cfg, n_steps=4, grad_clip=cfg.grad_clip, log_interval=log_interval,
                         eval_interval=0, device=CPU, logger=LOGGER, generate_samples=False)
        runs.append((losses, _parameters(model)))
    assert runs[0][0] == runs[1][0]
    assert all(torch.equal(runs[0][1][name], runs[1][1][name]) for name in runs[0][1])


@pytest.mark.skipif(os.environ.get("VFE3_TEST_DEVICE") != "cuda" or not torch.cuda.is_available(),
                    reason="CUDA lane only (VFE3_TEST_DEVICE=cuda)")
def test_cuda_phase_marks_add_no_host_synchronization() -> None:
    device = torch.device("cuda")
    timer = TrainingTimer(device)
    x = torch.randn(64, 64, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        timer.start_step()
        y = x @ x
        timer.mark("forward")
        y = y @ x
        timer.mark("backward")
        timer.finish_step(n_tokens=1)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    sample = timer.sample_train_window()
    assert sum(sample.phase_ms_mean.values()) == pytest.approx(sample.train_step_ms_mean, rel=1e-3)
    del y


# ------------------------------------------------------------------------------------------------ A337

def _ladder_model() -> tuple:
    cfg = _cfg()
    torch.manual_seed(5)
    model = VFEModel(cfg)
    optimizer = T.build_optimizer(model, cfg)
    T._set_successful_update_count(optimizer, 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    tokens, targets = next(iter(_loaders()[0]))
    return cfg, model, optimizer, scheduler, tokens, targets


def test_the_ladder_replays_a_step_that_raised_after_unscale_under_an_enabled_scaler(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, model, optimizer, scheduler, tokens, targets = _ladder_model()
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    scale = scaler.get_scale()
    real_clip = torch.nn.utils.clip_grads_with_norm_
    policies: List[str] = []

    def flaky_clip(*args: Any, **kwargs: Any) -> Any:                     # after unscale_, before any update
        policies.append(torch.get_float32_matmul_precision())
        if policies[-1] != "highest":
            raise RuntimeError("frame-pair inverse residual certificate failed (injected)")
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", flaky_clip)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        replays_log: List[Dict[str, object]] = []
        loss, replays = T._train_step_with_precision_retry(
            model, optimizer, scheduler, tokens, targets, logger=LOGGER, step=0, replay_log=replays_log,
            grad_clip=1e-9, scaler=scaler, status_out={})
    finally:
        torch.set_float32_matmul_precision(previous)
    assert policies[0] == "high" and policies[-1] == "highest" and replays == 1
    assert math.isfinite(loss) and scaler.get_scale() == scale
    assert T._successful_update_count(optimizer) == 1
    assert replays_log[0]["policy"] == "high" and replays_log[0]["replay_policy"] == "highest"


def test_a_non_finite_skip_is_replayed_only_without_an_enabled_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, model, optimizer, scheduler, tokens, targets = _ladder_model()
    attempts: List[str] = []

    def skipping(*args: Any, status_out: Dict[str, Any], **kwargs: Any) -> float:
        attempts.append(torch.get_float32_matmul_precision())
        status_out.update(did_step=False, nonfinite_skip=True)
        return float("nan")

    monkeypatch.setattr(T, "train_step", skipping)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        for enabled, expected in ((True, ["high"]), (False, ["high", "highest"])):
            attempts.clear()
            loss, replays = T._train_step_with_precision_retry(
                model, optimizer, scheduler, tokens, targets, logger=LOGGER, step=0,
                scaler=torch.amp.GradScaler(device="cpu", enabled=enabled), status_out={})
            assert attempts == expected and math.isnan(loss) and replays == len(expected) - 1
    finally:
        torch.set_float32_matmul_precision(previous)
