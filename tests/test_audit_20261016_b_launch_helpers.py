r"""Shared launch helpers and the launch-time test-split preflight (audit 2026-10-16 F96, F122).

F96: ``train_baseline`` reserves and renames its run directories through
:func:`vfe3.experiment.reserve_directory` and :func:`vfe3.experiment.rename_run_by_test_ppl` instead of
private copies; the shared rename takes the ``val`` prefix a learning-rate sweep arm needs.

F122: :func:`vfe3.experiment.run_vfe` and :func:`train_baseline.run_baseline` open the test split with the
train and validation splits, so a missing test cache fails at launch, before training and before a run
directory exists, instead of after the whole training budget. A sweep arm (``score_test=False``) never
reads the test split and still trains without one.
"""

from __future__ import annotations

from tests.profiles import test_config

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

import train_baseline as tb
import vfe3.data.datasets as datasets_mod
import vfe3.experiment as experiment
import vfe3.train as T
from vfe3.baseline import BaselineConfig
from vfe3.config import VFE3Config

DATASET = "wikitext-103"
V = 7
N_TOKENS = {"train": 60, "validation": 24, "test": 24}


class _TrainCalled(Exception):
    r"""Raised by the stubbed training loop: the launch got past its preflight into training."""


def _train_stub(*args: Any, **kwargs: Any) -> None:
    raise _TrainCalled


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _caches(root: Path, monkeypatch: pytest.MonkeyPatch, splits: tuple) -> Path:
    r"""Tiny ``wikitext-103`` caches of ``splits`` in ``root`` (the test split is left out on request)."""
    root.mkdir()
    generator = torch.Generator().manual_seed(0)
    for split in splits:
        torch.save(torch.randint(0, V, (N_TOKENS[split],), generator=generator),
                   datasets_mod.cache_path(DATASET, split, cache_dir=root))
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: root)
    monkeypatch.setattr(datasets_mod, "tokens_per_char", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_tokens_per_char", lambda *args, **kwargs: None)
    return root


def _vfe_cfg() -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=2, max_seq_len=4, eval_stride=2, n_layers=1,
        max_steps=2, warmup_steps=0, log_interval=0, eval_interval=0, checkpoint_interval=0,
        generate_embedding_figures=False, pos_phi="learned", seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _baseline_cfg() -> BaselineConfig:
    return BaselineConfig(vocab_size=V, embed_dim=4, n_heads=2, n_layers=1, mlp_expansion=2, max_seq_len=4,
                          batch_size=2, eval_stride=2, max_steps=2, warmup_steps=0, lr=1e-3, log_interval=0,
                          eval_interval=0, checkpoint_interval=0, seed=0)


# ---- F96: one copy of the run-directory helpers ------------------------------------------------

def test_train_baseline_uses_the_shared_run_directory_helpers() -> None:
    assert tb.reserve_directory is experiment.reserve_directory
    assert tb.rename_run_by_test_ppl is experiment.rename_run_by_test_ppl
    assert not hasattr(tb, "_reserve_directory") and not hasattr(tb, "_rename_by_test_ppl")


def test_shared_rename_takes_a_metric_prefix(tmp_path: Path) -> None:
    logger = logging.getLogger("launch-helpers-test")
    run = experiment.reserve_directory(tmp_path, "20261017-000000_arm")
    clash = experiment.reserve_directory(tmp_path, "val12.50_arm")
    renamed = experiment.rename_run_by_test_ppl(run, "arm", 12.5, logger, prefix="val")
    assert renamed == tmp_path / "val12.50_arm_2" and renamed.is_dir() and clash.is_dir()
    plain = experiment.rename_run_by_test_ppl(experiment.reserve_directory(tmp_path, "t_arm"), "arm", 3.25, logger)
    assert plain == tmp_path / "3.25_arm"
    kept = experiment.reserve_directory(tmp_path, "t_nan")
    assert experiment.rename_run_by_test_ppl(kept, "nan", float("nan"), logger, prefix="val") == kept


# ---- F122: the test split is opened at launch -----------------------------------------------------

def test_run_vfe_without_a_test_cache_fails_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _caches(tmp_path / "cache", monkeypatch, ("train", "validation"))
    monkeypatch.setattr(T, "train", _train_stub)
    out_root = tmp_path / "runs"
    with pytest.raises(FileNotFoundError, match="test"):
        experiment.run_vfe(_vfe_cfg(), dataset=DATASET, out_root=out_root, seed=0, data_seed=0, label="arm",
                           device="cpu")
    assert not out_root.exists() or not any(out_root.iterdir())


def test_run_vfe_with_every_split_reaches_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _caches(tmp_path / "cache", monkeypatch, ("train", "validation", "test"))
    monkeypatch.setattr(T, "train", _train_stub)
    with pytest.raises(_TrainCalled):
        experiment.run_vfe(_vfe_cfg(), dataset=DATASET, out_root=tmp_path / "runs", seed=0, data_seed=0,
                           label="arm", device="cpu")


def test_run_baseline_without_a_test_cache_fails_before_training(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _caches(tmp_path / "cache", monkeypatch, ("train", "validation"))
    monkeypatch.setattr(tb, "train", _train_stub)
    out_root = tmp_path / "runs"
    with pytest.raises(FileNotFoundError, match="test"):
        tb.run_baseline(_baseline_cfg(), dataset=DATASET, out_root=out_root, seed=0, label="arm", device="cpu",
                        generate_samples=False)
    assert not out_root.exists() or not any(out_root.iterdir())


def test_baseline_sweep_arm_trains_without_a_test_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _caches(tmp_path / "cache", monkeypatch, ("train", "validation"))
    monkeypatch.setattr(tb, "train", _train_stub)
    with pytest.raises(_TrainCalled):
        tb.run_baseline(_baseline_cfg(), dataset=DATASET, out_root=tmp_path / "runs", seed=0, label="arm",
                        device="cpu", generate_samples=False, score_test=False)
