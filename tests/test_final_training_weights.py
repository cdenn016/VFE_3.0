"""Small end-to-end checks for final validation, saved weights, and test selection."""
from tests.profiles import test_config
import csv
from dataclasses import asdict
import importlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

import train_vfe3 as launcher
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.viz import report


def test_launcher_saves_and_tests_all_training_without_periodic_checkpoints(
    tmp_path:    Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = test_config(
        vocab_size=7, embed_dim=4, n_heads=2, batch_size=1, max_seq_len=4,
        n_layers=1, n_e_steps=1, max_steps=3, warmup_steps=0,
        eval_interval=2, log_interval=0, checkpoint_interval=0,
        skip_belief_sigma_update=True,
        generate_embedding_figures=False,
    )
    loader = DataLoader(TokenWindows(torch.arange(9) % 7, 4), batch_size=1)    # one window at every origin
    run_dir = tmp_path / "run"
    artifacts = importlib.import_module("vfe3.run_artifacts")
    training = importlib.import_module("vfe3.train")
    datasets = importlib.import_module("vfe3.data.datasets")
    monkeypatch.setattr(launcher, "config", asdict(cfg))
    monkeypatch.setattr(launcher, "DEVICE", "cpu")
    monkeypatch.setattr(launcher, "MAX_TOKENS", None)
    monkeypatch.setattr(launcher, "_select_loader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(launcher, "_run_dir", lambda *args, **kwargs: run_dir)
    monkeypatch.setattr(launcher, "_rename_run_by_test_ppl", lambda path, *args: path)
    monkeypatch.setattr(launcher, "_banner", lambda *args, **kwargs: "tiny final-state fixture")
    monkeypatch.setattr(datasets, "tokens_per_char", lambda *args: 0.5)
    monkeypatch.setattr(training, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(artifacts, "_run_figures_isolated", lambda *args, **kwargs: True)
    final_states: list[dict[str, torch.Tensor]] = []
    finalizer = artifacts.finalize_run

    def capture_final(model: VFEModel, *args: Any, **kwargs: Any) -> dict[str, object]:
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        final_states.append(expected)
        result = finalizer(model, *args, **kwargs)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert result["test_ce"] == pytest.approx(kwargs["final_validation"]["ce"])
        return result

    monkeypatch.setattr(artifacts, "finalize_run", capture_final)
    launcher._run_once(61, logging.getLogger("final-training-fixture"))
    assert len(final_states) == 1
    summary = json.loads((run_dir / "summary.json").read_text())
    final_val = json.loads((run_dir / "final_validation.json").read_text())
    assert summary["evaluated_step"] == final_val["step"] == 3
    assert summary["evaluation_weights"] == "final" and summary["reloaded_best"] is False
    assert summary["final_val_ce"] == pytest.approx(final_val["ce"])
    assert summary["final_val_ppl"] == pytest.approx(final_val["ppl"])
    with (run_dir / "metrics.csv").open(newline="") as handle:
        periodic = list(csv.DictReader(handle))
    assert summary["best_val_ppl"] == min(float(periodic[-1]["val_ppl"]), final_val["ppl"])
    assert not list((run_dir / "checkpoints").glob("step_*.pt"))
    bundle = torch.load(run_dir / "final_model.pt", weights_only=True)
    for name, value in bundle["model_state"].items():
        torch.testing.assert_close(value, final_states[0][name], rtol=0, atol=0)

    class LoadedFinalWeights(Exception):
        pass

    load_final = report._load_figure_model_state

    def inspect_offline_load(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert path == run_dir / "final_model.pt"
        state = load_final(path, *args, **kwargs)
        for name, value in state.items():
            torch.testing.assert_close(value, final_states[0][name], rtol=0, atol=0)
        raise LoadedFinalWeights

    monkeypatch.setattr(report, "_load_figure_model_state", inspect_offline_load)
    with pytest.raises(LoadedFinalWeights):
        report.generate_figures(run_dir, loader=loader, device=torch.device("cpu"))
