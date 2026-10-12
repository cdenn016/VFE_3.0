r"""Audit 2026-10-12 round 2: what the training loop records (items 7, 12 and 16).

Item 7: ``train()`` builds the realized-update block and the diagnostic-failure counts before the terminal callback,
so both the launcher's :func:`finalize_run` and the ablation cell's :func:`finalize_validation_run` publish the
accepted and skipped update counts. Item 12: under ``decode_second_order`` ``train_ce``/``train_ppl`` are the plug-in
cross-entropy that validation scores, and the second-order objective the step trains on is logged as
``train_second_order_ce``. Item 16: a training stream shorter than :math:`L + s` tokens draws its window origin from
the origins that fit, and ``pos_loss_ratio`` is the ratio of the averaged quarters. Tiny models only.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.train as vt
from tests.profiles import test_config
from vfe3 import run_artifacts as ra
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

SEQ = 8


def _cfg(**overrides: Any) -> Any:
    values = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=SEQ, n_layers=1, n_e_steps=1,
                  warmup_steps=1, max_steps=3, batch_size=2, checkpoint_interval=0,
                  generate_embedding_figures=False)
    values.update(overrides)
    return test_config(**values)


def _loader(n: int, *, shuffle: bool) -> DataLoader:
    tokens = torch.arange(3).repeat(n // 3 + 2)[:n].long()
    generator = torch.Generator().manual_seed(7) if shuffle else None
    return DataLoader(TokenWindows(tokens, SEQ), batch_size=2, shuffle=shuffle, drop_last=shuffle,
                      generator=generator)


@pytest.fixture
def _fast_finalize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ra, "_git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *a, **k: None)


def _reject_second_step(model: VFEModel) -> None:
    r"""Make the second training forward return a NaN loss, which the step rejects."""
    forward, calls = model.forward, []

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        out = forward(*args, **kwargs)
        if model.training and isinstance(out, tuple):
            calls.append(None)
            if len(calls) == 2:
                return (out[0], out[1] * float("nan"), out[2])
        return out

    model.forward = poisoned


def _assert_realized_counts(summary: Dict[str, Any]) -> None:
    assert summary["attempted_steps"] == 3
    assert summary["accepted_updates"] == 2 and summary["skipped_steps"] == 1
    assert summary["accepted_update_frac"] == pytest.approx(2 / 3)
    assert isinstance(summary["diagnostics_failures"], dict)


def test_launcher_finalizer_publishes_the_realized_counts(tmp_path: Path, _fast_finalize: None) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    _reject_second_step(model)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    val_loader = _loader(40, shuffle=False)
    losses = vt.train(model, _loader(64, shuffle=True), cfg, n_steps=3, val_loader=val_loader,
                      artifacts=artifacts, generate_samples=False)
    ra.finalize_run(model, artifacts, cfg, final_validation=vt.evaluate(model, val_loader),
                    val_loader=val_loader, test_loader=_loader(40, shuffle=False), losses=losses)
    _assert_realized_counts(json.loads((tmp_path / "r" / "summary.json").read_text(encoding="utf-8")))


def test_ablation_finalizer_publishes_the_realized_counts(tmp_path: Path, _fast_finalize: None) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    _reject_second_step(model)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model)
    val_loader = _loader(40, shuffle=False)

    def finalize(state: Any, losses: List[float]) -> None:
        ra.finalize_validation_run(model, artifacts, cfg, val_loader, losses=losses, terminal_state=state,
                                   device=torch.device("cpu"))

    vt.train(model, _loader(64, shuffle=True), cfg, n_steps=3, val_loader=val_loader, artifacts=artifacts,
             terminal_callback=finalize, generate_samples=False)
    _assert_realized_counts(json.loads((tmp_path / "r" / "summary.json").read_text(encoding="utf-8")))


# ---- item 12 -----------------------------------------------------------------------------------

def _pair(**overrides: Any) -> tuple:
    torch.manual_seed(0)
    second = VFEModel(_cfg(decode_second_order=True, **overrides))
    torch.manual_seed(0)
    plug_in = VFEModel(_cfg(**overrides))
    plug_in.load_state_dict(second.state_dict())
    return second, plug_in


def test_second_order_forward_reports_the_plug_in_ce_and_trains_on_the_objective() -> None:
    second, plug_in = _pair()
    tokens, targets = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]]), torch.tensor([[1, 2, 3, 4, 5, 0, 1, 2]])
    second.train(), plug_in.train()
    _, loss, stats = second(tokens, targets, return_decode_stats=True)
    _, plug_loss, plug_stats = plug_in(tokens, targets, return_decode_stats=True)
    torch.testing.assert_close(stats.ce, plug_stats.ce, rtol=0.0, atol=1e-6)
    assert plug_stats.objective is None and stats.objective is not None
    assert float(stats.objective) > float(stats.ce)                  # 1/2 tr(Sigma Cov_p(w)) > 0
    torch.testing.assert_close(loss.detach(), stats.objective, rtol=0.0, atol=0.0)
    _, _, ce = second(tokens, targets)                               # the plain three-tuple keeps the plug-in CE
    torch.testing.assert_close(ce, stats.ce, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_train_step_logs_plug_in_ce_and_the_second_order_ce(grad_accum_steps: int) -> None:
    second, plug_in = _pair()
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1], [2, 3, 4, 5, 0, 1, 2, 3]])
    targets = torch.roll(tokens, -1, dims=1)
    logged: Dict[str, Dict[str, float]] = {}
    for name, model in (("second", second), ("plug_in", plug_in)):
        optimizer = vt.build_optimizer(model, model.cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        logged[name] = {}
        vt.train_step(model, optimizer, scheduler, tokens, targets, grad_accum_steps=grad_accum_steps,
                      metrics_out=logged[name])
    assert logged["second"]["train_ce"] == pytest.approx(logged["plug_in"]["train_ce"], abs=1e-6)
    assert "train_second_order_ce" not in logged["plug_in"]
    assert logged["second"]["train_second_order_ce"] > logged["second"]["train_ce"]


def test_metrics_rows_carry_the_second_order_ce_beside_the_plug_in_ce(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = _cfg(decode_second_order=True, max_steps=2)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    vt.train(model, _loader(64, shuffle=True), cfg, n_steps=2, artifacts=artifacts, log_interval=1,
             generate_samples=False)
    rows = [row for row in artifacts.history if "train_ce" in row]
    assert rows and all(row["train_second_order_ce"] > row["train_ce"] for row in rows)
    assert all(row["train_ppl"] == pytest.approx(math.exp(row["train_ce"])) for row in rows)


# ---- item 16 -----------------------------------------------------------------------------------

def test_a_stream_shorter_than_two_windows_draws_from_the_origins_that_fit() -> None:
    tokens = torch.arange(SEQ + 3, dtype=torch.long) % 6                # T = L + 3 < L + s
    windows = TokenWindows(tokens, SEQ)
    windows.use_per_epoch_offsets()
    assert len(windows) == 1 and windows.offset_bound == 3
    generator = torch.Generator().manual_seed(0)
    drawn = {windows.draw_window_offset(generator) for _ in range(50)}
    assert drawn == {0, 1, 2}
    for origin in drawn:
        windows.window_offset = origin
        x, y = windows[0]
        assert torch.equal(x, tokens[origin:origin + SEQ]) and torch.equal(y, tokens[origin + 1:origin + SEQ + 1])
    long_stream = TokenWindows(torch.arange(5 * SEQ + 1) % 6, SEQ)
    long_stream.use_per_epoch_offsets()
    assert long_stream.offset_bound == SEQ and len(long_stream) == 4    # unchanged once T >= L + s


def test_a_smoke_run_on_a_short_stream_trains() -> None:
    torch.manual_seed(0)
    cfg = _cfg(max_steps=2, batch_size=1)
    loader = DataLoader(TokenWindows(torch.arange(SEQ + 3) % 6, SEQ), batch_size=1, shuffle=True,
                        generator=torch.Generator().manual_seed(1))
    losses = vt.train(VFEModel(cfg), loader, cfg, n_steps=2, log_interval=0, generate_samples=False)
    assert len(losses) == 2 and all(math.isfinite(loss) for loss in losses)


def test_pos_loss_ratio_is_the_ratio_of_the_averaged_quarters(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A sliding-window batch whose first quarter is masked contributes to the last quarter only."""
    per_batch = iter([
        {"pos_loss_first_q": float("nan"), "pos_loss_last_q": 2.0, "pos_loss_ratio": float("nan")},
        {"pos_loss_first_q": 1.0, "pos_loss_last_q": 4.0, "pos_loss_ratio": 4.0},
    ])
    monkeypatch.setattr(vt, "_val_diagnostics_one",
                        lambda *a, **k: vt.ValidationDiagnostics(next(per_batch), torch.zeros(1, SEQ), None))
    out = vt._val_diagnostics(None, [0, 1], torch.device("cpu"), batches=2).metrics
    assert out["pos_loss_first_q"] == 1.0 and out["pos_loss_last_q"] == 3.0
    assert out["pos_loss_ratio"] == pytest.approx(out["pos_loss_last_q"] / out["pos_loss_first_q"])
