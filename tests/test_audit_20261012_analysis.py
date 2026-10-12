r"""Audit 2026-10-12: analysis readers against the real finalize writers.

Item 7 / punch-list #29: the scored-target keys read by ``vfe3.posthoc.report``, ``vfe3.paper.tables``
and the multi-seed aggregates must exist in a ``summary.json`` written by the real finalize path,
both the launcher's :func:`finalize_run` and the ablation cell's :func:`finalize_validation_run`.
Item 8: the multi-seed noise floor is :math:`t_{n-1, 0.975} \sqrt{2} s`.
Item 16: figure titles use the headline :func:`vfe3.metrics.spearman_rho`.
"""
from tests.profiles import test_config
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import multiseed_analysis as ms
from vfe3 import run_artifacts as ra
from vfe3.data.datasets import TokenWindows
from vfe3.metrics import spearman_rho
from vfe3.model.model import VFEModel
from vfe3.paper import tables
from vfe3.paper.runs import load_run
from vfe3.paper.tables import NOT_RECORDED
from vfe3.posthoc import report
from vfe3.train import TrainingTerminalState, build_optimizer, evaluate, train
from vfe3.viz import figures as figs


def _cfg() -> Any:
    return test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, n_e_steps=1,
                       gauge_group="block_glk", warmup_steps=1, max_steps=2, log_interval=1,
                       eval_interval=1000, checkpoint_interval=0, generate_embedding_figures=False)


def _loader(n: int, *, shuffle: bool) -> DataLoader:
    tokens = torch.arange(3).repeat(n // 3 + 2)[:n].long()
    generator = torch.Generator().manual_seed(7) if shuffle else None
    return DataLoader(TokenWindows(tokens, 8), batch_size=4, shuffle=shuffle, drop_last=shuffle,
                      generator=generator)


@pytest.fixture
def _fast_finalize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ra, "_git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *a, **k: None)


def _scored_target_contract(run_dir: Path, *, has_test: bool) -> Dict[str, Any]:
    r"""Assert every scored-target read resolves on the written summary; return the summary."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    info = SimpleNamespace(summary=summary)
    validation = report._recorded_reference(info, "final_model.pt", "validation")
    assert validation is not None and isinstance(validation["scored_targets"], int)
    assert validation["scored_targets"] > 0
    best = report._recorded_reference(info, "best_model.pt", "validation")
    assert best is not None and best["scored_targets"] == validation["scored_targets"]
    items = dict((label, str(value)) for label, value in tables._evaluation_items(load_run(run_dir), "final")
                 if label)
    assert items["validation scored targets"] != NOT_RECORDED
    assert items["validation scored targets"].startswith(f"{validation['scored_targets']:,}")
    if has_test:
        test = report._recorded_reference(info, "final_model.pt", "test")
        assert test is not None and isinstance(test["scored_targets"], int) and test["scored_targets"] > 0
        assert items["test scored targets"].startswith(f"{test['scored_targets']:,}")
    return summary


def test_finalize_run_summary_carries_every_scored_target_key_the_readers_use(tmp_path: Path,
                                                                             _fast_finalize: None) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    val_loader = _loader(48, shuffle=False)
    losses = train(model, _loader(64, shuffle=True), cfg, n_steps=cfg.max_steps, val_loader=val_loader,
                   artifacts=artifacts)
    final_validation = evaluate(model, val_loader)                 # the launchers' fallback pass
    ra.finalize_run(model, artifacts, cfg, final_validation=final_validation, tokens_per_char=0.25,
                    val_loader=val_loader, test_loader=_loader(40, shuffle=False), losses=losses)

    summary = _scored_target_contract(tmp_path / "r", has_test=True)
    # The multi-seed headline scalars that summary.json is the source for.
    for key in ("test_ppl", "best_val_ppl", "test_ce", "test_bits_per_token", "test_bpc", "wall_time_s"):
        assert key in ms.SCALAR_KEYS and key in summary


def test_validation_finalizer_summary_carries_the_validation_scored_target_key(tmp_path: Path,
                                                                               _fast_finalize: None) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model)
    raw = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    state = TrainingTerminalState(step=cfg.max_steps, optimizer=build_optimizer(model, cfg), scaler=None,
                                  data_state=None, raw_model_state=raw,
                                  rng_state={"cpu": torch.get_rng_state().clone(), "cuda": None})
    ra.finalize_validation_run(model, artifacts, cfg, _loader(48, shuffle=False), losses=[1.0, 0.9],
                               terminal_state=state, device=torch.device("cpu"))

    summary = _scored_target_contract(tmp_path / "r", has_test=False)
    assert "final_val_scored_targets" not in summary and "test_ce" not in summary


def test_noise_floor_is_the_single_seed_difference_band(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy", None)                   # aggregation needs no scipy
    agg = ms._summarize([10.0, 11.0, 12.0])
    expected = 4.302652729749464 * math.sqrt(2.0) * 1.0               # t(2, 0.975) = sqrt(2) c / sqrt(1 - c^2), c = 0.95
    assert "two_sd" not in agg
    assert agg["noise_floor"] == pytest.approx(expected, rel=1e-13)  # ~6.09 s at n = 3, not 2 s
    assert math.isnan(ms._summarize([10.0])["noise_floor"])


@pytest.mark.parametrize("nu", [*range(1, 41), 99, 100, 1000])
def test_scipy_free_t_quantile_matches_scipy(nu: int) -> None:
    t_dist = pytest.importorskip("scipy.stats").t
    assert ms._student_t_quantile_975(nu) == pytest.approx(float(t_dist.ppf(0.975, nu)), rel=1e-12)


def test_figure_spearman_is_the_headline_average_rank_statistic() -> None:
    tr = np.array([1.0, 1.0, 1.0, 2.0, 2.0, np.nan])                 # frozen-covariance ties + a NaN
    ce = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2])
    headline = spearman_rho(torch.as_tensor(tr), torch.as_tensor(ce))
    assert figs._spearman(tr, ce) == headline
    assert not hasattr(figs, "_spearman_np")
