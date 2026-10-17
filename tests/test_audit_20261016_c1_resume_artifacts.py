r"""Resume exactness of the run artifacts, 2026-10-16 deep audit, lane C1 (findings 77, 81, 123 and 128).

F77: ``train()`` built a fresh divergence monitor on every call, so after a resume the loss and role-norm
EMAs, their warmup counts, the spike cooldown and the open interval statistics restarted. The checkpoint now
carries the monitor's host state (with the skip counters and the last logged group norms) and a resumed run
flags the same spikes and writes the same divergence columns as the uninterrupted run.

F81: only the periodic checkpoint set the wall clock, and only from the optional timer, so the terminal
checkpoint of a terminal-callback cell (and any checkpoint of a run without log or eval rows) carried None
and a resume from it reported the continuation's wall time alone. Every checkpoint now carries the
cumulative training wall clock.

Tiny CPU models only (K=4, two heads, N=4, vocab 7; the live-route test runs ``tests.profiles.live_config``).
"""
import csv
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.test_audit2_20261011_training import _cfg
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

B, N, V = 2, 4, 7
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _spikes(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "diagnostics" / T.DIVERGENCE_SPIKE_LOG
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _loader(cfg: VFE3Config) -> DataLoader:
    tokens = (torch.arange(61) * 5 + 2) % V                               # 15 windows: 7 batches per epoch
    return DataLoader(TokenWindows(tokens, N), batch_size=B, shuffle=False, drop_last=True)


def _run(cfg: VFE3Config, run_dir: Path, *, n_steps: int, log_interval: int,
         resume_from: Any = None) -> Tuple[List[float], ra.RunArtifacts]:
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    torch.manual_seed(11)
    losses = T.train(model, _loader(cfg), cfg, n_steps=n_steps, grad_clip=cfg.grad_clip,
                     log_interval=log_interval, eval_interval=0, device=CPU,
                     logger=logging.getLogger("audit-c1-resume"), artifacts=artifacts,
                     generate_samples=False, resume_from=resume_from)
    return losses, artifacts


# ------------------------------------------------------------------------------------------------ F77

@pytest.mark.parametrize("log_interval", [4, 3])        # the checkpoint step 6 is between rows / is a row
def test_resumed_run_flags_the_same_spikes_and_divergence_columns(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_interval: int) -> None:
    # Loose thresholds so a tiny run spikes at all: any loss above the previous one.
    monkeypatch.setattr(T, "DIVERGENCE_EMA_DECAY", 0.0)                   # the EMA is the previous value
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 1)
    monkeypatch.setattr(T, "DIVERGENCE_LOSS_SPIKE_MARGIN", 0.0)
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_COOLDOWN", 3)
    cfg = _cfg(checkpoint_interval=6, max_steps=12)
    straight, _ = _run(cfg, tmp_path / "straight", n_steps=12, log_interval=log_interval)
    checkpoint = tmp_path / "straight" / "checkpoints" / "step_6.pt"
    resumed, _ = _run(cfg, tmp_path / "resumed", n_steps=12, log_interval=log_interval, resume_from=checkpoint)
    assert resumed == straight[6:]                                         # the weights path is exact
    expected_spikes = [r for r in _spikes(tmp_path / "straight") if r["step"] > 6]
    assert expected_spikes, "the loosened thresholds must produce a spike after the resume"
    got_spikes = _spikes(tmp_path / "resumed")
    assert [r["step"] for r in got_spikes] == [r["step"] for r in expected_spikes]
    assert [r["spike"] for r in got_spikes] == [r["spike"] for r in expected_spikes]
    assert [r["causes"] for r in got_spikes] == [r["causes"] for r in expected_spikes]
    expected_rows = [row for row in _rows(tmp_path / "straight" / "metrics.csv") if int(row["step"]) > 6]
    got_rows = _rows(tmp_path / "resumed" / "metrics.csv")
    assert [row["step"] for row in got_rows] == [row["step"] for row in expected_rows]
    for expected, got in zip(expected_rows, got_rows):
        for column in T.DIVERGENCE_COLUMNS:
            assert got[column] == expected[column], (expected["step"], column)


def test_checkpoint_divergence_state_is_tolerant_and_round_trips() -> None:
    monitor = T._DivergenceMonitor()
    for step, loss in enumerate((2.0, 2.5, 1.5), start=1):
        monitor.observe(step, loss, {"ce": loss, "role_grad_norms": {"mu": 0.5 * step, "phi": 1.0},
                                     "clip_coefs": {"mu": 0.5}})
    record = T._divergence_checkpoint_record(monitor, interval_ends=False, skipped_steps=2, consecutive_skips=1,
                                             last_group_norms={"mu_embed": 0.25, "decode": float("nan")})
    normalized = T._checkpoint_divergence_state(record)
    assert normalized is not None
    assert normalized["monitor"] == record["monitor"] and normalized["skipped_steps"] == 2
    assert normalized["consecutive_skips"] == 1 and normalized["last_group_norms"]["mu_embed"] == 0.25
    restored = T._DivergenceMonitor()
    assert restored.load_state_dict(record["monitor"])
    assert restored.state_dict(interval_ends=False) == record["monitor"]
    status = {"clip_coefs": {"mu": 0.5}}
    assert (json.dumps(restored.columns(status, flagged=False), sort_keys=True)          # NaN-aware
            == json.dumps(monitor.columns(status, flagged=False), sort_keys=True))
    ended = T._divergence_checkpoint_record(monitor, interval_ends=True, skipped_steps=0, consecutive_skips=0,
                                            last_group_norms={})
    assert ended["monitor"]["interval"]["steps"] == 0 and ended["monitor"]["loss_seen"] == 3
    fresh = T._DivergenceMonitor().state_dict(interval_ends=False)
    assert ended["monitor"]["interval"] == fresh["interval"]
    bad_monitor = dict(record["monitor"])
    for malformed in (None, 3, {}, {**record, "schema": 99}, {**record, "skipped_steps": -1},
                      {**record, "last_group_norms": {"mu_embed": "x"}},
                      {**record, "monitor": {**bad_monitor, "loss_seen": -1}},
                      {**record, "monitor": {**bad_monitor, "norm_ema": {"mu": "x"}}},
                      {**record, "monitor": {**bad_monitor, "interval": {"steps": 1}}}):
        assert T._checkpoint_divergence_state(malformed) is None
    assert not T._DivergenceMonitor().load_state_dict({**bad_monitor, "loss_seen": -1})


# ------------------------------------------------------------------------------------------------ F81

def _finalized_run(cfg: VFE3Config, run_dir: Path, *, n_steps: int) -> Tuple[ra.RunArtifacts, float]:
    r"""A terminal-callback run (log/eval/checkpoint cadence off, so train() builds no timer) and its wall time."""
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    val_loader = DataLoader(TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride),
                            batch_size=2, shuffle=False)

    def finalize(state: Any, losses: List[float]) -> None:
        ra.finalize_validation_run(model, artifacts, cfg, val_loader, losses=losses, terminal_state=state,
                                   device=CPU, logger=logging.getLogger("audit-c1-resume"))

    torch.manual_seed(11)
    start = time.perf_counter()
    T.train(model, _loader(cfg), cfg, n_steps=n_steps, grad_clip=cfg.grad_clip, log_interval=0,
            eval_interval=0, val_loader=val_loader, device=CPU, logger=logging.getLogger("audit-c1-resume"),
            artifacts=artifacts, generate_samples=False, terminal_callback=finalize)
    return artifacts, time.perf_counter() - start


def test_terminal_checkpoint_carries_the_run_wall_clock(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=4)
    _, elapsed = _finalized_run(cfg, tmp_path / "cell", n_steps=4)
    payload = torch.load(tmp_path / "cell" / "checkpoints" / "step_4.pt", weights_only=True)
    wall = payload["wall_clock_s"]
    assert isinstance(wall, float) and 0.0 < wall <= elapsed
    assert T._checkpoint_divergence_state(payload["divergence_monitor"]) is not None
    # A resume from the terminal checkpoint carries the cell's wall time into the continuation.
    _, resumed = _run(cfg, tmp_path / "resumed", n_steps=6, log_interval=1,
                      resume_from=tmp_path / "cell" / "checkpoints" / "step_4.pt")
    assert resumed.resumed_wall_clock_s == wall
    rows = _rows(tmp_path / "resumed" / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [5, 6]
    assert all(float(row["wall_clock_s"]) > wall for row in rows)


def test_periodic_checkpoint_without_a_timer_carries_the_wall_clock(tmp_path: Path) -> None:
    cfg = _cfg(checkpoint_interval=2, max_steps=4)
    start = time.perf_counter()
    _run(cfg, tmp_path / "run", n_steps=4, log_interval=0)          # no log or eval row: no timer
    elapsed = time.perf_counter() - start
    walls = [torch.load(tmp_path / "run" / "checkpoints" / f"step_{step}.pt", weights_only=True)["wall_clock_s"]
             for step in (2, 4)]
    assert all(isinstance(wall, float) for wall in walls)
    assert 0.0 < walls[0] < walls[1] <= elapsed
