"""Regression tests for the October 25, 2026 training-lifecycle audit findings F06 and F07.

F06: an output-disabled resume (``artifacts=None``) restores the checkpoint's divergence-monitor record, so the
loss-spike threshold and skip budgets that decide update acceptance match the artifacts-enabled resume.
F07: when publication of an improving terminal best is blocked, the validation finalizer's primary score is the
retained selected score and the unpublished terminal score is recorded with an explicit qualification.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from torch.utils.data import DataLoader

from tests.profiles import test_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, load_checkpoint

T = importlib.import_module("vfe3.train")
R = importlib.import_module("vfe3.run_artifacts")


def _resume_config() -> Any:
    r"""The tiny K=4, H=2, V=5 configuration of the audit's resume-monitor probe, with loss-spike rejection on."""
    return test_config(vocab_size=5, embed_dim=4, n_heads=2, n_layers=1,
                       n_e_steps=1, batch_size=1, max_seq_len=2, eval_stride=None,
                       max_steps=3, warmup_steps=0, min_lr_frac=0.0,
                       grad_clip=None, log_interval=0, eval_interval=0,
                       checkpoint_interval=1, checkpoint_at_end=False,
                       decode_mode="linear", decode_unigram_prior=False,
                       attention_mode="projected_predictive", family="gaussian_diagonal",
                       use_block_mlp=False, skip_belief_sigma_update=True,
                       m_table_update_mode="adamw", m_phi_update_mode="adamw",
                       amp_dtype=None, fp32_matmul_precision="highest",
                       loss_spike_skip_nats=1.0, generate_embedding_figures=False)


def _resume_loader() -> DataLoader:
    r"""A deterministic unshuffled loader over a 15-token stream of windows of length 2."""
    return DataLoader(TokenWindows(torch.arange(15) % 5, 2), batch_size=1,
                      shuffle=False, drop_last=True)


def _saved_checkpoint(
    directory: Path,
    cfg: Any,
) -> Path:
    r"""Train one ordinary step with artifacts and return its ``checkpoints/step_1.pt``."""
    torch.manual_seed(17)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(directory / "source", cfg, model)
    T.train(model, _resume_loader(), cfg, n_steps=1, grad_clip=cfg.grad_clip,
            log_interval=0, eval_interval=0, artifacts=artifacts,
            device=torch.device("cpu"), logger=logging.getLogger("audit-20261025"),
            generate_samples=False)
    return directory / "source" / "checkpoints" / "step_1.pt"


def _spiked_resume(
    directory: Path,
    cfg: Any,
    checkpoint: Path,
    use_artifacts: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    r"""Resume ``checkpoint`` for one step with a constant 10-nat loss offset and record the acceptance decision."""
    torch.manual_seed(29)
    resumed = VFEModel(cfg)
    original_forward = resumed.forward

    def spiked_forward(*args: Any, **kwargs: Any) -> Any:
        logits, loss, statistics = original_forward(*args, **kwargs)
        return logits, loss + 10.0, statistics

    resumed.forward = spiked_forward
    observed: Dict[str, Any] = {}
    original_step = T.train_step

    def observe_step(*args: Any, **kwargs: Any) -> float:
        observed["loss_skip_threshold"] = kwargs.get("loss_skip_threshold")
        loss = original_step(*args, **kwargs)
        status = kwargs["status_out"]
        observed.update(did_step=status["did_step"], loss_spike_skip=status["loss_spike_skip"])
        return loss

    monkeypatch.setattr(T, "train_step", observe_step)
    destination = (RunArtifacts(directory / "resumed", cfg, resumed) if use_artifacts else None)
    try:
        T.train(resumed, _resume_loader(), cfg, n_steps=2, grad_clip=cfg.grad_clip,
                log_interval=0, eval_interval=0, artifacts=destination,
                device=torch.device("cpu"), logger=logging.getLogger("audit-20261025"),
                generate_samples=False, resume_from=checkpoint)
    finally:
        monkeypatch.setattr(T, "train_step", original_step)
    return observed


def test_f06_artifact_free_resume_keeps_the_saved_loss_spike_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The same checkpoint rejects the same spiked update with and without artifacts (audit 2026-10-25 F06)."""
    torch.set_num_threads(1)
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 1)
    cfg = _resume_config()
    checkpoint = _saved_checkpoint(tmp_path, cfg)
    with_artifacts = _spiked_resume(tmp_path, cfg, checkpoint, True, monkeypatch)
    without_artifacts = _spiked_resume(tmp_path, cfg, checkpoint, False, monkeypatch)
    assert with_artifacts["loss_skip_threshold"] is not None
    assert with_artifacts["loss_spike_skip"] is True and with_artifacts["did_step"] is False
    assert without_artifacts["loss_skip_threshold"] == pytest.approx(with_artifacts["loss_skip_threshold"])
    assert without_artifacts["loss_spike_skip"] is True and without_artifacts["did_step"] is False


def test_f06_load_checkpoint_returns_divergence_state_without_artifacts(
    tmp_path: Path,
) -> None:
    r"""``load_checkpoint(..., artifacts=None)`` fills the divergence buffer with the saved skip counters, and an
    older bundle without the record still loads and leaves the buffer empty."""
    torch.set_num_threads(1)
    cfg = _resume_config()
    checkpoint = _saved_checkpoint(tmp_path, cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    record = payload["divergence_monitor"]
    record["failed_step_skips"] = 2
    record["consecutive_skips"] = 1
    record["skipped_steps"] = 3
    modified = tmp_path / "with_counters.pt"
    torch.save(payload, modified)
    identity = payload["data_state"]["data_identity"]
    buffer: Dict[str, object] = {"stale": True}
    load_checkpoint(modified, VFEModel(cfg), artifacts=None, divergence_state=buffer, restore_rng=False,
                    expected_data_identity=identity)
    assert "stale" not in buffer
    assert buffer["failed_step_skips"] == 2
    assert buffer["consecutive_skips"] == 1
    assert buffer["skipped_steps"] == 3
    assert buffer["monitor"]["loss_seen"] == record["monitor"]["loss_seen"]

    del payload["divergence_monitor"]
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    legacy_buffer: Dict[str, object] = {}
    assert load_checkpoint(legacy, VFEModel(cfg), artifacts=None, divergence_state=legacy_buffer,
                           restore_rng=False, expected_data_identity=identity) == 1
    assert legacy_buffer == {}


def test_f07_blocked_terminal_publication_keeps_primary_on_the_selected_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Earlier best 20, improving terminal 18 whose ``best_model.pt`` publication is locked: primary stays 20 and the
    terminal score is recorded as unpublished (audit 2026-10-25 F07)."""
    torch.set_num_threads(1)
    cfg = test_config(vocab_size=5, embed_dim=4, n_heads=2, n_layers=1,
                      batch_size=1, max_seq_len=2, eval_stride=None, max_steps=2, warmup_steps=0,
                      use_block_mlp=False, decode_mode="linear",
                      decode_unigram_prior=False, generate_embedding_figures=False)
    model = VFEModel(cfg)
    loader = DataLoader(TokenWindows(torch.arange(9) % 5, 2, pad_final=True),
                        batch_size=1, shuffle=False)
    artifacts = RunArtifacts(tmp_path, cfg, model)
    artifacts.bind_selection_data_identity(T._loader_data_identity(loader, cfg.vocab_size))
    assert artifacts.maybe_save_best(1, model, 20.0)
    original_replace = R._atomic_replace

    def locked_replace(final: Path, temporary: Path, **kwargs: Any) -> None:
        if final.name == "best_model.pt":
            raise PermissionError("simulated ordinary Windows reader lock")
        original_replace(final, temporary, **kwargs)

    def terminal_metrics(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ce": math.log(18.0), "ppl": 18.0,
                "bits_per_token": math.log2(18.0), "bpc": None,
                "expected_targets": 8, "scored_targets": 8, "excluded_targets": 0}

    monkeypatch.setattr(R, "_atomic_replace", locked_replace)
    monkeypatch.setattr(T, "evaluate", terminal_metrics)
    monkeypatch.setattr(R, "_write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(R, "_pure_path_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(R, "_run_figures_isolated", lambda *args, **kwargs: True)
    result = R.finalize_validation_run(model, artifacts, cfg, loader, device=torch.device("cpu"))
    assert result["best_val_ppl"] == 20.0 and result["best_step"] == 1
    assert result["primary_val_ppl"] == result["best_val_ppl"]
    assert result["terminal_selection_published"] is False
    assert result["unpublished_terminal_val_ppl"] == 18.0
    assert result["final_val_ppl"] == 18.0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["primary_val_ppl"] == 20.0 and summary["best_step"] == 1
    assert summary["terminal_selection_published"] is False
    assert summary["unpublished_terminal_val_ppl"] == 18.0


def test_f07_published_terminal_improvement_is_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Without a lock the improving terminal score is published, selected and primary."""
    torch.set_num_threads(1)
    cfg = test_config(vocab_size=5, embed_dim=4, n_heads=2, n_layers=1,
                      batch_size=1, max_seq_len=2, eval_stride=None, max_steps=2, warmup_steps=0,
                      use_block_mlp=False, decode_mode="linear",
                      decode_unigram_prior=False, generate_embedding_figures=False)
    model = VFEModel(cfg)
    loader = DataLoader(TokenWindows(torch.arange(9) % 5, 2, pad_final=True),
                        batch_size=1, shuffle=False)
    artifacts = RunArtifacts(tmp_path, cfg, model)
    artifacts.bind_selection_data_identity(T._loader_data_identity(loader, cfg.vocab_size))
    assert artifacts.maybe_save_best(1, model, 20.0)

    def terminal_metrics(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ce": math.log(18.0), "ppl": 18.0,
                "bits_per_token": math.log2(18.0), "bpc": None,
                "expected_targets": 8, "scored_targets": 8, "excluded_targets": 0}

    monkeypatch.setattr(T, "evaluate", terminal_metrics)
    monkeypatch.setattr(R, "_write_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(R, "_pure_path_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(R, "_run_figures_isolated", lambda *args, **kwargs: True)
    result = R.finalize_validation_run(model, artifacts, cfg, loader, device=torch.device("cpu"))
    assert result["primary_val_ppl"] == 18.0 == result["best_val_ppl"]
    assert result["best_step"] == cfg.max_steps
    assert result["terminal_selection_published"] is True
    assert result["unpublished_terminal_val_ppl"] is None
