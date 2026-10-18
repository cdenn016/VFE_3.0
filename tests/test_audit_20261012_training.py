"""Training-loop and finalizer fixes of the October 12 audit (items 11, 12, 13, 15).

Tiny CPU models only (K=4, two heads, one layer, N<=8, vocab <= 7). Covers: the logged ``lr_*`` columns are the
rate the logged step used, on an accepted and on a skipped step (#11); a terminal validation that falls on an eval
step is scored once and written as one metrics row, and a terminal step that is not an eval step still gets its
validation row (#12); the ``pos_loss_*`` columns ignore positions without a scored target (#13); finalize-time
probe failures are counted in summary.json ``diagnostics_failures`` (#15).
"""
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

B, N, V = 2, 4, 7
CPU = torch.device("cpu")
LOGGER = logging.getLogger("audit-20261012")


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    monkeypatch.setattr(ra, "_git_code_identity", lambda *args, **kwargs: {
        "git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    torch.set_num_threads(1)


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=1, n_e_steps=1, warmup_steps=0, max_steps=4, log_interval=0, eval_interval=0,
        checkpoint_interval=0, generate_embedding_figures=False, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(41) % V, N)
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    return (DataLoader(train_ds, batch_size=B, shuffle=False, drop_last=True),
            DataLoader(val_ds, batch_size=2, shuffle=False))


def _model(cfg: VFE3Config, seed: int = 5) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------------- #11: logged learning rate

def test_logged_lr_is_the_rate_the_logged_step_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(warmup_steps=4, max_steps=8, min_lr_frac=0.0)
    model = _model(cfg)
    real_forward = model.forward
    calls = {"n": 0}

    def forward(*args: Any, **kwargs: Any) -> Any:
        out = real_forward(*args, **kwargs)
        if kwargs.get("return_decode_stats"):
            calls["n"] += 1
            if calls["n"] == 3:                                         # the third train step: a NaN loss
                return (out[0], out[1] * float("nan"), *out[2:])
        return out

    monkeypatch.setattr(model, "forward", forward)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    loaders = _loaders(cfg)
    torch.manual_seed(11)
    T.train(model, loaders[0], cfg, n_steps=4, grad_clip=cfg.grad_clip, log_interval=1, eval_interval=0,
            val_loader=loaders[1], device=CPU, logger=LOGGER, artifacts=artifacts, generate_samples=False)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [row["step_skipped"] for row in rows] == ["0.0", "0.0", "1.0", "0.0"]
    for column in ("lr_mu", "lr_phi"):
        lr = [float(row[column]) for row in rows]
        # Warmup multipliers: step 1 uses lambda(0)=0, step 2 lambda(1)=1/4, step 3 (skipped) lambda(2)=2/4, and
        # step 4 reuses lambda(2) because the skipped step did not advance the scheduler.
        assert lr[0] == 0.0, column
        assert lr[2] == pytest.approx(2.0 * lr[1]), column
        assert lr[3] == pytest.approx(lr[2]), column


# ------------------------------------------------------------------------- #12: one terminal validation

def _run_with_finalizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_interval: int
                        ) -> Tuple[List[Dict[str, str]], Dict[str, Any], int]:
    cfg = _cfg(max_steps=4)
    model = _model(cfg)
    loaders = _loaders(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    real_evaluate = T.evaluate
    evaluations = {"n": 0}

    def evaluate(eval_model: Any, loader: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        evaluations["n"] += int(loader is loaders[1])                   # validation passes only
        return real_evaluate(eval_model, loader, *args, **kwargs)

    monkeypatch.setattr(T, "evaluate", evaluate)

    def callback(state: T.TrainingTerminalState, losses: List[float]) -> None:
        ra.finalize_validation_run(model, artifacts, cfg, loaders[1], losses=losses, train_loader=loaders[0],
                                   terminal_state=state, device=CPU, logger=LOGGER)

    torch.manual_seed(11)
    T.train(model, loaders[0], cfg, n_steps=4, grad_clip=cfg.grad_clip, log_interval=0,
            eval_interval=eval_interval, val_loader=loaders[1], device=CPU, logger=LOGGER,
            artifacts=artifacts, generate_samples=False, terminal_callback=callback)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    return _rows(tmp_path / "run" / "metrics.csv"), summary, evaluations["n"]


def test_terminal_eval_step_is_scored_once_and_logged_once(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    rows, summary, evaluations = _run_with_finalizer(tmp_path, monkeypatch, eval_interval=2)
    assert evaluations == 2                                             # steps 2 and 4; no finalizer pass
    assert [int(row["step"]) for row in rows] == [2, 4]
    assert float(rows[-1]["val_ce"]) == pytest.approx(summary["final_val_ce"])
    assert summary["n_steps"] == 4


def test_terminal_step_off_the_eval_cadence_keeps_its_validation_row(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    rows, summary, evaluations = _run_with_finalizer(tmp_path, monkeypatch, eval_interval=3)
    assert evaluations == 2                                             # step 3, then the finalizer at step 4
    assert [int(row["step"]) for row in rows] == [3, 4]
    assert float(rows[-1]["val_ce"]) == pytest.approx(summary["final_val_ce"])


# ------------------------------------------------------------------------- #13: position loss on scored targets

def test_position_loss_ignores_unscored_positions() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 8, V)
    targets = torch.randint(0, V, (1, 8))
    full = T._position_loss_metrics(logits, targets)
    per = torch.nn.functional.cross_entropy(logits[0], targets[0], reduction="none")
    assert full["pos_loss_first_q"] == pytest.approx(float(per[:2].mean()))
    assert full["pos_loss_ratio"] == pytest.approx(float(per[-2:].mean() / per[:2].mean()))

    prefix_masked = targets.clone()
    prefix_masked[:, :6] = -100                                         # a stride-2 window's overlap prefix
    out = T._position_loss_metrics(logits, prefix_masked)
    assert math.isnan(out["pos_loss_first_q"]) and math.isnan(out["pos_loss_ratio"])
    assert out["pos_loss_last_q"] == pytest.approx(full["pos_loss_last_q"])

    partly = targets.clone()
    partly[:, 0] = -100                                                 # one of the two first-quarter positions
    assert T._position_loss_metrics(logits, partly)["pos_loss_first_q"] == pytest.approx(float(per[1]))


def test_multi_batch_position_loss_skips_masked_prefix_windows() -> None:
    cfg = _cfg(max_seq_len=8, eval_stride=2)
    model = _model(cfg)
    # batch_size=1: the second batch is window 1, whose first six targets are the masked overlap prefix.
    val = DataLoader(TokenWindows((torch.arange(31) * 3) % V, 8, pad_final=True, stride=2),
                     batch_size=1, shuffle=False)
    one = T._val_diagnostics(model, val, CPU, batches=1).metrics
    two = T._val_diagnostics(model, val, CPU, batches=2).metrics
    assert two["pos_loss_first_q"] == pytest.approx(one["pos_loss_first_q"])
    assert two["pos_loss_ratio"] < 1e3


# ------------------------------------------------------------------------- #15: finalize probe failures

def test_finalize_probe_failures_reach_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    loaders = _loaders(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    artifacts.diagnostics_failures["val_diagnostics"] = 2              # a training-time failure is kept

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("probe failure")

    probes = ("estep_character", "beta_channel_decomposition",
              "context_sensitivity", "phi_numerics")
    for probe in probes:
        monkeypatch.setattr(ra, f"collect_{probe}", fail)
    monkeypatch.setattr(ra, "_write_research_artifacts", lambda *args, **kwargs: None)
    ra.finalize_run(model, artifacts, cfg, final_validation=T.evaluate(model, loaders[1], device=CPU),
                    train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[1], losses=[1.0],
                    device=CPU, wall_time=1.0, logger=LOGGER)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics_failures"] == {"val_diagnostics": 2, **{probe: 1 for probe in probes}}
