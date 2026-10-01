"""Device-name fields under a CUDA build that sees zero devices (2026-09-28).

With ``CUDA_VISIBLE_DEVICES=""`` the cu128 torch build reports ``torch.cuda.is_available()`` True
while ``torch.cuda.device_count()`` is 0, and ``torch.cuda.get_device_name(0)`` raises 'Invalid
device id'. The provenance and cost-model writers must report the actual CPU model device even when
CUDA inventory is present. The CUDA queries are monkeypatched, so no device is touched. Tiny CPU models only (K=4, two heads, N=4).
"""
import json
import logging
from pathlib import Path
from typing import Callable, Dict

import pytest
import torch

import vfe3.run_artifacts as ra
from vfe3.baseline import BaselineConfig, BaselineTransformer
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

V, K, H, L, N, B = 7, 4, 2, 1, 4, 2


def _vfe() -> VFEModel:
    cfg = VFE3Config(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, n_layers=L,
        n_e_steps=1, warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0,
        checkpoint_interval=0, generate_embedding_figures=False)
    torch.manual_seed(0)
    return VFEModel(cfg)


def _provenance_device_name(tmp_path: Path) -> object:
    model = _vfe()
    artifacts = ra.RunArtifacts(tmp_path / "run", model.cfg, model, save_model_artifacts=False)
    ra._write_provenance(artifacts, model.cfg, model, logging.getLogger("test_cuda_zero_devices"))
    record = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    return record["device_name"]


def _cost_model_device_name(tmp_path: Path) -> object:
    model = _vfe()
    return ra._cost_model_fields(model, model.cfg, 1, 0)["device_name"]


def _baseline_device_name(tmp_path: Path) -> object:
    cfg = BaselineConfig(vocab_size=V, embed_dim=K, n_heads=H, n_layers=L, mlp_expansion=1,
                         max_seq_len=N, batch_size=B, max_steps=2, log_interval=0,
                         eval_interval=0, checkpoint_interval=0)
    torch.manual_seed(0)
    return BaselineTransformer(cfg).cost_model_fields(1, 0)["device_name"]


SITES: Dict[str, Callable[[Path], object]] = {
    "provenance": _provenance_device_name,
    "cost_model": _cost_model_device_name,
    "baseline_cost_model": _baseline_device_name,
}


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("n_devices, expected", [(0, "cpu"), (1, "cpu")])
def test_device_name_follows_cpu_model_independently_of_cuda_inventory(
        site: str, n_devices: int, expected: str, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    def device_name(device: object = None) -> str:
        if n_devices == 0:
            raise RuntimeError("Invalid device id")
        return "Fake GPU"

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n_devices)
    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)
    assert SITES[site](tmp_path) == expected
