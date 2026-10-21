r"""Opt-in weight EMA (``weight_ema_decay``; :mod:`vfe3.weight_ema`; audit 2026-10-20 item 8, part 2).

Pinned here: the decay math with and without an update interval, the start step, that a rejected update neither
updates nor counts, that the evaluation swap restores the raw weights bit for bit (also on an exception and with a
host-memory shadow), the checkpoint record and its validation, resume continuity against an uninterrupted run, an
older checkpoint without the record, the metrics columns and the finalize fields, and that the EMA leaves the raw
training trajectory bitwise unchanged. Tiny CPU models only (K = 4, two heads, N = 4, vocab 7, a few steps).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.test_audit_20261017_l3_training import _cfg, _fail_forward_at, _loaders, _rows, _run
from vfe3.weight_ema import WEIGHT_EMA_CHECKPOINT_KEY, WeightEMA, best_ema_validation, weight_ema_start_step


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("scale", torch.ones(2))


def _set(model: _Tiny, value: List[float]) -> None:
    with torch.no_grad():
        model.w.copy_(torch.tensor(value))


# ------------------------------------------------------------------------------------------------ unit

def test_decay_math_start_and_rejected_updates() -> None:
    model = _Tiny()
    ema = WeightEMA(model, decay=0.9, start_step=2)
    assert ema.names == ["w"]                                              # frozen params and buffers excluded
    _set(model, [1.0, 2.0, 3.0])
    assert not ema.observe(1, True) and not ema.active                     # before the start step
    _set(model, [2.0, 0.0, 4.0])
    assert ema.observe(2, True) and ema.initialized_step == 2
    torch.testing.assert_close(ema.shadow[0], torch.tensor([2.0, 0.0, 4.0]), rtol=0, atol=0)
    _set(model, [12.0, 10.0, -6.0])
    assert ema.observe(3, True)
    expected = 0.9 * torch.tensor([2.0, 0.0, 4.0]) + 0.1 * torch.tensor([12.0, 10.0, -6.0])
    torch.testing.assert_close(ema.shadow[0], expected, rtol=1e-6, atol=1e-6)
    _set(model, [100.0, 100.0, 100.0])
    assert not ema.observe(4, False)                                       # a rejected update changes nothing
    torch.testing.assert_close(ema.shadow[0], expected, rtol=0, atol=0)
    assert (ema.updates, ema.accepted) == (2, 1)


def test_update_interval_applies_the_decay_of_k_steps_every_k_accepted_updates() -> None:
    model = _Tiny()
    ema = WeightEMA(model, decay=0.8, start_step=1, update_interval=2)
    _set(model, [1.0, 1.0, 1.0])
    ema.observe(1, True)
    _set(model, [3.0, 3.0, 3.0])
    assert not ema.observe(2, True)                                        # first accepted of the interval
    assert not ema.observe(3, False)                                       # rejected: does not count
    assert ema.observe(4, True)
    torch.testing.assert_close(ema.shadow[0], torch.full((3,), 0.64 * 1.0 + 0.36 * 3.0), rtol=1e-6, atol=1e-6)


def test_start_step_from_the_fraction_of_max_steps() -> None:
    assert weight_ema_start_step(0.5, 8) == 4
    assert weight_ema_start_step(0.55, 10) == 6
    assert weight_ema_start_step(0.0, 8) == 1
    assert weight_ema_start_step(1.0, 8) == 8
    cfg = _cfg(weight_ema_decay=0.99, weight_ema_start_frac=0.25, weight_ema_update_interval=3)
    ema = WeightEMA.from_config(_Tiny(), cfg)
    assert (ema.start_step, ema.update_interval, ema.decay) == (2, 3, 0.99)
    assert WeightEMA.from_config(_Tiny(), _cfg()) is None


@pytest.mark.parametrize("device", ["same", "cpu"])
def test_the_swap_restores_the_raw_weights_exactly(device: str) -> None:
    model = _Tiny()
    _set(model, [0.1, 0.2, 0.3])
    ema = WeightEMA(model, decay=0.5, start_step=1, device=device)
    ema.observe(1, True)
    _set(model, [1.0 / 3.0, -7.25, 1e-8])
    ema.observe(2, True)
    raw = {name: t.clone() for name, t in model.state_dict().items()}
    with ema.applied():
        torch.testing.assert_close(model.w.detach(), ema.shadow[0], rtol=0, atol=0)
        assert torch.equal(model.frozen, raw["frozen"]) and torch.equal(model.scale, raw["scale"])
    assert all(torch.equal(model.state_dict()[name], value) for name, value in raw.items())
    with pytest.raises(RuntimeError, match="boom"):
        with ema.applied():
            raise RuntimeError("boom")
    assert all(torch.equal(model.state_dict()[name], value) for name, value in raw.items())


def test_state_dict_round_trip_and_validation() -> None:
    model = _Tiny()
    ema = WeightEMA(model, decay=0.9, start_step=1)
    _set(model, [1.0, 2.0, 3.0])
    ema.observe(1, True)
    _set(model, [2.0, 2.0, 2.0])
    ema.observe(2, True)
    state = ema.state_dict()
    other = WeightEMA(_Tiny(), decay=0.9, start_step=1)
    other.load_state_dict(state)
    assert torch.equal(other.shadow[0], ema.shadow[0]) and (other.updates, other.accepted) == (2, 1)
    with pytest.raises(RuntimeError, match="different parameter set"):
        other.validate_state({**state, "names": ["v"]})
    with pytest.raises(RuntimeError, match="does not match"):
        other.validate_state({**state, "shadow": [torch.zeros(4)]})
    with pytest.raises(RuntimeError, match="schema-1"):
        other.validate_state({"ema": 1})


def test_best_ema_validation_reads_the_metrics_rows() -> None:
    rows = [{"step": 2, "val_ema_ppl": float("nan")}, {"step": 4, "val_ema_ppl": 30.0, "val_ema_ce": 3.4,
                                                       "weight_ema_updates": 3},
            {"step": 6, "val_ema_ppl": 31.0}, {"step": 7, "train_ce": 1.0}]
    assert best_ema_validation(rows) == {"best_val_ema_ppl": 30.0, "best_val_ema_ce": 3.4, "best_val_ema_step": 4,
                                         "best_val_ema_updates": 3}
    assert best_ema_validation(None)["best_val_ema_step"] is None


def test_validator_and_inactive_fields() -> None:
    for bad in (dict(weight_ema_decay=1.0), dict(weight_ema_decay=0.0), dict(weight_ema_start_frac=1.5),
                dict(weight_ema_update_interval=0), dict(weight_ema_device="cuda")):
        with pytest.raises(ValueError, match="weight_ema"):
            _cfg(**bad)
    assert "weight_ema_start_frac" in _cfg().inactive_fields()
    assert "weight_ema_start_frac" not in _cfg(weight_ema_decay=0.99).inactive_fields()


# ------------------------------------------------------------------------------------------------ training loop

def _params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_ema_leaves_the_raw_training_trajectory_bitwise_unchanged(tmp_path: Path) -> None:
    off_losses, off_artifacts, off_model = _run(_cfg(), tmp_path / "off", n_steps=6, log_interval=2,
                                                eval_interval=2)
    cfg = _cfg(weight_ema_decay=0.5, weight_ema_start_frac=0.25)          # starts at step 2 of max_steps = 8
    on_losses, on_artifacts, on_model = _run(cfg, tmp_path / "on", n_steps=6, log_interval=2, eval_interval=2)
    assert on_losses == off_losses
    off, on = _params(off_model), _params(on_model)
    assert all(torch.equal(off[name], on[name]) for name in off)
    assert off_artifacts.best_step == on_artifacts.best_step
    off_rows, on_rows = _rows(tmp_path / "off" / "metrics.csv"), _rows(tmp_path / "on" / "metrics.csv")
    assert "val_ema_ce" not in off_rows[0] and "weight_ema_updates" not in off_rows[0]
    for a, b in zip(off_rows, on_rows):
        assert a["val_ce"] == b["val_ce"]                                  # raw validation unchanged
    by_step = {int(row["step"]): row for row in on_rows}
    assert by_step[2]["val_ema_ce"] != "" and by_step[2]["weight_ema_updates"] == "1"
    assert by_step[6]["weight_ema_updates"] == "5"
    ema = on_artifacts.weight_ema
    assert ema.updates == 5 and ema.initialized_step == 2
    assert float(by_step[2]["val_ema_ce"]) == pytest.approx(float(by_step[2]["val_ce"]), abs=1e-6)  # shadow = raw
    assert by_step[6]["val_ema_ce"] != by_step[6]["val_ce"]


def test_rejected_updates_do_not_update_the_ema_in_training(tmp_path: Path) -> None:
    cfg = _cfg(max_failed_step_skips=1, weight_ema_decay=0.9, weight_ema_start_frac=0.0)
    calls: List[int] = []
    _, artifacts, _ = _run(cfg, tmp_path / "run", n_steps=5, model_hook=_fail_forward_at(calls, fail_on={2}))
    assert artifacts.realized_updates["accepted_updates"] == 4
    assert artifacts.weight_ema.updates == 4 and artifacts.weight_ema.initialized_step == 1


def test_checkpoint_round_trip_and_resume_continue_the_shadow(tmp_path: Path) -> None:
    cfg = _cfg(weight_ema_decay=0.7, weight_ema_start_frac=0.25, checkpoint_interval=3)
    straight, artifacts, _ = _run(cfg, tmp_path / "straight", n_steps=6)
    checkpoint = tmp_path / "straight" / "checkpoints" / "step_3.pt"
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)[WEIGHT_EMA_CHECKPOINT_KEY]
    assert record["updates"] == 2 and record["names"] == artifacts.weight_ema.names
    resumed, resumed_artifacts, _ = _run(cfg, tmp_path / "resumed", n_steps=6, resume_from=checkpoint)
    assert resumed == straight[3:]
    a, b = artifacts.weight_ema, resumed_artifacts.weight_ema
    assert (b.updates, b.initialized_step) == (a.updates, a.initialized_step) == (5, 2)
    assert all(torch.equal(x, y) for x, y in zip(a.shadow, b.shadow))


def test_a_checkpoint_without_the_record_resumes_and_restarts_the_ema(tmp_path: Path,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    plain = _cfg(checkpoint_interval=3)
    straight, _, _ = _run(plain, tmp_path / "plain", n_steps=6)
    checkpoint = tmp_path / "plain" / "checkpoints" / "step_3.pt"
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    bundle.pop(WEIGHT_EMA_CHECKPOINT_KEY, None)                            # an older checkpoint
    torch.save(bundle, checkpoint)
    cfg = _cfg(weight_ema_decay=0.7, weight_ema_start_frac=0.25, checkpoint_interval=3)
    with caplog.at_level(logging.WARNING):
        resumed, artifacts, _ = _run(cfg, tmp_path / "resumed", n_steps=6, resume_from=checkpoint)
    assert resumed == straight[3:]
    assert artifacts.weight_ema.initialized_step == 4 and artifacts.weight_ema.updates == 3
    assert any("carries no weight EMA" in record.message for record in caplog.records)


def test_finalize_fields_score_the_ema_weights_and_restore_the_raw_ones(tmp_path: Path) -> None:
    cfg = _cfg(weight_ema_decay=0.5, weight_ema_start_frac=0.25)
    _, artifacts, model = _run(cfg, tmp_path / "run", n_steps=4)
    raw = _params(model)
    _, val_loader = _loaders()
    fields = ra._weight_ema_evaluation(model, artifacts, logging.getLogger("ema"), val_loader=val_loader,
                                       test_loader=val_loader, tokens_per_char=None, device=torch.device("cpu"))
    assert fields["weight_ema_status"] == "evaluated" and fields["weight_ema_updates"] == 3
    assert fields["test_ema_ce"] == fields["val_ema_ce"] and math.isfinite(fields["test_ema_ce"])
    with artifacts.weight_ema.applied():
        expected = T.evaluate(model, val_loader, device=torch.device("cpu"))["ce"]
    assert fields["val_ema_ce"] == float(expected)
    assert all(torch.equal(raw[name], p.detach()) for name, p in model.named_parameters())
    off = ra._weight_ema_evaluation(model, type("A", (), {"weight_ema": None})(), logging.getLogger("ema"),
                                    val_loader=val_loader, test_loader=None, tokens_per_char=None,
                                    device=torch.device("cpu"))
    assert off == {}
