"""CUDA gates under a build that sees zero devices (2026-09-29).

With ``CUDA_VISIBLE_DEVICES=""`` the cu128 torch build reports ``torch.cuda.is_available()`` True
while ``torch.cuda.device_count()`` is 0. Every gate that selects a device or touches the CUDA RNG
must key on ``device_count() > 0``. The CUDA queries are monkeypatched, so the outcome does not
depend on the host, and the RNG stubs count calls rather than raise (several callers isolate
exceptions). Tiny CPU models only (K=4, two heads, N=4, vocab 7).
"""
import logging
from pathlib import Path
from typing import Any, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.experiment as experiment
import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.test_training_diagnostics_20260928 import _cfg, _loaders, _model, _train
from vfe3.model.model import VFEModel
from vfe3.posthoc.evaluate import resolve_device


class _Stop(Exception):
    """Sentinel raised after the device has been chosen."""


@pytest.fixture
def zero_devices(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """is_available True, device_count 0; returns the log of CUDA RNG state calls."""
    calls: List[str] = []

    def recorder(name: str, result: Any) -> Any:
        def stub(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return result
        return stub

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    # AdamW also queries graph capture when is_available is mocked true; keep that host query
    # inside this simulated zero-device environment instead of initializing the real CUDA runtime.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", recorder("get_rng_state_all", []))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", recorder("set_rng_state_all", None))
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    return calls


def test_resolve_device_auto_is_cpu(zero_devices: List[str]) -> None:
    assert resolve_device("auto") == torch.device("cpu")


def test_run_vfe_defaults_to_cpu(zero_devices: List[str], tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    chosen: List[torch.device] = []

    def record_to(self: VFEModel, device: Any = None, *args: Any, **kwargs: Any) -> VFEModel:
        chosen.append(torch.device(device))
        return self

    def stop(*args: Any, **kwargs: Any) -> None:
        raise _Stop

    monkeypatch.setattr(VFEModel, "to", record_to)
    monkeypatch.setattr(experiment, "_loader", stop)
    with pytest.raises(_Stop):
        experiment.run_vfe(_cfg(), dataset="tiny", out_root=tmp_path, seed=0, data_seed=0,
                           label="t")
    assert chosen == [torch.device("cpu")]


def test_run_training_defaults_to_cpu(zero_devices: List[str],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    chosen: List[torch.device] = []

    def record_to(self: VFEModel, device: Any = None, *args: Any, **kwargs: Any) -> VFEModel:
        chosen.append(torch.device(device))
        return self

    def stop(*args: Any, **kwargs: Any) -> None:
        raise _Stop

    monkeypatch.setattr(VFEModel, "to", record_to)
    monkeypatch.setattr(T, "make_dataloader", stop)
    with pytest.raises(_Stop):
        T.run_training(_cfg(), n_steps=1)
    assert chosen == [torch.device("cpu")]


def test_validate_rng_state_accepts_cpu_only_checkpoint(zero_devices: List[str]) -> None:
    ra._validate_rng_state({"cpu": torch.get_rng_state(), "cuda": None})


def test_validate_rng_state_accepts_gpu_checkpoint_on_zero_devices(
        zero_devices: List[str]) -> None:
    state = torch.get_rng_state()
    ra._validate_rng_state({"cpu": state, "cuda": [state.clone(), state.clone()]})


def test_restore_rng_state_skips_cuda_leg(zero_devices: List[str]) -> None:
    state = torch.get_rng_state()
    ra._restore_rng_state({"cpu": state, "cuda": [state.clone()]})
    assert zero_devices == []


def test_periodic_generation_leaves_cuda_rng_alone(zero_devices: List[str]) -> None:
    model = _model(_cfg())
    prompt = torch.tensor([[1, 2]])
    T._periodic_generation(model, prompt, 2, lambda ids: " ".join(map(str, ids)))
    assert zero_devices == []


def test_train_checkpoint_and_finalize_leave_cuda_rng_alone(
        zero_devices: List[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "EARLY_EVAL_STEPS", (0, 1, 3))
    monkeypatch.setattr(T, "EARLY_EVAL_MIN_INTERVAL", 1)
    cfg = _cfg(checkpoint_interval=2, max_steps=4)
    loaders = _loaders(cfg)
    loaders = (DataLoader(loaders[0].dataset, batch_size=loaders[0].batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(3)),) + loaders[1:]
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, dataset="tiny")
    _train(model, cfg, loaders, artifacts=artifacts, log_interval=2, eval_interval=2)
    assert (tmp_path / "run" / "checkpoints").is_dir()
    ra.finalize_run(
        model, artifacts, cfg, final_validation=T.evaluate(model, loaders[1]),
        train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2], losses=[1.0],
        device=torch.device("cpu"), wall_time=1.0, logger=logging.getLogger("zero-device"))
    assert zero_devices == []
