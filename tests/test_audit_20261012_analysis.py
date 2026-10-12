r"""Audit 2026-10-12: analysis readers against the real finalize writers.

Item 7 / punch-list #29 (completed by round 2, item 13): every scalar key that the analysis readers take from
``metrics.csv`` and ``summary.json`` -- the run figures (``GEOMETRY_HEALTH_COLUMNS``,
``VALIDATION_SANITY_COLUMNS``), the multi-seed aggregates (``CURVE_SPECS``, ``GRID_COLS``, the ``summary.json``
members of ``SCALAR_KEYS``), the paper tables and S1 panel D counters, and ``vfe3.posthoc.report`` -- must be
written by a real :func:`vfe3.train.train` run on the live route profile with nonzero log and evaluation intervals,
finalized by the launcher's :func:`finalize_run` or the ablation cell's :func:`finalize_validation_run`. Only the
figure subprocess is stubbed; the metrics rows and diagnostics are real.
Item 8: the multi-seed noise floor is :math:`t_{n-1, 0.975} \sqrt{2} s`.
Item 16: figure titles use the headline :func:`vfe3.metrics.spearman_rho`.
"""
from tests.profiles import live_config
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import multiseed_analysis as ms
from vfe3 import run_artifacts as ra
from vfe3.data.datasets import TokenWindows
from vfe3.metrics import spearman_rho
from vfe3.model.model import VFEModel
from vfe3.paper import figures as paper_figures
from vfe3.paper import tables
from vfe3.paper.runs import load_run
from vfe3.paper.tables import NOT_RECORDED
from vfe3.posthoc import report
from vfe3.train import evaluate, train
from vfe3.viz import figures as figs

# Keys a reader may read that this run does not produce, each with the reason.
_CONDITIONAL_COLUMNS = {
    "phi_chart_projected_fraction": "phi_mstep_max_matrix_norm is None",
    "phi_chart_projection_scale_min": "phi_mstep_max_matrix_norm is None",
    "phi_chart_projection_ms": "phi_mstep_max_matrix_norm is None",
}
# The multi-seed scalars that come from research.json (vfe3.posthoc), not from summary.json.
_RESEARCH_SCALARS = {"ece", "overall_ce", "sigma_trace_cv", "sigma_ce_spearman", "fd_gradient_worst_rel_error",
                     "corpus_freq_strata_ce.rare", "corpus_freq_strata_ce.mid", "corpus_freq_strata_ce.frequent"}


def _cfg() -> Any:
    return live_config(vocab_size=6, max_seq_len=8, max_steps=2, warmup_steps=1, log_interval=1,
                       eval_interval=1, checkpoint_interval=0, generate_embedding_figures=False,
                       phi_frame_condition_max=1.5)


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


def _metrics_rows(run_dir: Path) -> List[Dict[str, str]]:
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(rows: List[Dict[str, str]], key: str) -> bool:
    for row in rows:
        try:
            if math.isfinite(float(row.get(key) or "nan")):
                return True
        except ValueError:
            pass
    return False


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


def _metrics_contract(run_dir: Path) -> None:
    r"""Every metrics.csv column a reader reads exists; the geometry-health and held-out geometry columns that
    the relational diagnostics produce carry a finite value."""
    rows = _metrics_rows(run_dir)
    assert rows
    columns = set(rows[0])
    readers = {
        "geometry_health": figs.GEOMETRY_HEALTH_COLUMNS,
        "validation_sanity": figs.VALIDATION_SANITY_COLUMNS,
        "multiseed curves": tuple(key for key, _ in ms.CURVE_SPECS),
        "multiseed grid": tuple(ms.GRID_COLS),
        "paper S1 panel D": tuple(key for pair in paper_figures.COUNTER_PAIRS for key in pair[:2]),
    }
    for reader, keys in readers.items():
        missing = [key for key in keys if key not in columns and key not in _CONDITIONAL_COLUMNS]
        assert not missing, f"{reader} reads columns no run writes: {missing}"
    never_finite = [key for key in figs.GEOMETRY_HEALTH_COLUMNS
                    if key not in _CONDITIONAL_COLUMNS and not _finite(rows, key)]
    assert not never_finite, f"geometry-health columns never finite: {never_finite}"
    held_out = [key for key in figs.VALIDATION_SANITY_COLUMNS
                if key.startswith("val_") and not _finite(rows, key)]
    assert not held_out, f"held-out geometry columns never finite: {held_out}"
    for key in ("frame_projection_count", "frame_projection_total", "skipped_updates", "attempted_updates",
                "train_ce", "fisher_trace_mean"):
        assert _finite(rows, key), key


def _summary_contract(summary: Dict[str, Any]) -> None:
    for key in ("attempted_steps", "accepted_updates", "skipped_steps", "accepted_update_frac",
                "diagnostics_failures", "numerics_counters"):
        assert key in summary, key
    assert summary["attempted_steps"] == 2 and summary["accepted_updates"] + summary["skipped_steps"] == 2
    assert summary["numerics_counters"]["frame_projection_total"] > 0


def test_finalize_run_writes_every_key_the_readers_use(tmp_path: Path, _fast_finalize: None) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    val_loader = _loader(48, shuffle=False)
    losses = train(model, _loader(64, shuffle=True), cfg, n_steps=cfg.max_steps, val_loader=val_loader,
                   artifacts=artifacts, log_interval=cfg.log_interval,
                   eval_interval=cfg.eval_interval, generate_samples=False)
    final_validation = evaluate(model, val_loader)                 # the launchers' fallback pass
    ra.finalize_run(model, artifacts, cfg, final_validation=final_validation, tokens_per_char=0.25,
                    val_loader=val_loader, test_loader=_loader(40, shuffle=False), losses=losses)

    summary = _scored_target_contract(tmp_path / "r", has_test=True)
    _summary_contract(summary)
    _metrics_contract(tmp_path / "r")
    scalars = [key for key in ms.SCALAR_KEYS if key not in _RESEARCH_SCALARS]
    assert not [key for key in scalars if key not in summary], scalars
    tables.methods_contract_items(load_run(tmp_path / "r"), "final")


def test_validation_finalizer_writes_every_key_the_readers_use(tmp_path: Path, _fast_finalize: None) -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "r", cfg, model)
    val_loader = _loader(48, shuffle=False)

    def finalize(state: Any, losses: List[float]) -> None:
        ra.finalize_validation_run(model, artifacts, cfg, val_loader, losses=losses, terminal_state=state,
                                   device=torch.device("cpu"))

    train(model, _loader(64, shuffle=True), cfg, n_steps=cfg.max_steps, val_loader=val_loader,
          artifacts=artifacts, terminal_callback=finalize, log_interval=cfg.log_interval,
          eval_interval=cfg.eval_interval, generate_samples=False)

    summary = _scored_target_contract(tmp_path / "r", has_test=False)
    assert "final_val_scored_targets" not in summary and "test_ce" not in summary
    _summary_contract(summary)
    _metrics_contract(tmp_path / "r")


def test_noise_floor_is_the_single_seed_difference_band_and_drives_the_flags() -> None:
    from scipy.stats import t as t_dist
    agg = ms._summarize([10.0, 11.0, 12.0])
    expected = float(t_dist.ppf(0.975, 2)) * math.sqrt(2.0) * 1.0
    assert "two_sd" not in agg
    assert agg["noise_floor"] == pytest.approx(expected)              # ~6.09 s at n = 3, not 2 s
    cells = {"inside": 10.0 + 0.99 * expected, "outside": 10.0 + 1.01 * expected}
    assert ms.flag_noise_dominated(cells, agg["sd"], baseline=10.0, n_sd_seeds=agg["n"]) == ["inside"]
    assert math.isnan(ms._summarize([10.0])["noise_floor"])


def test_figure_spearman_is_the_headline_average_rank_statistic() -> None:
    tr = np.array([1.0, 1.0, 1.0, 2.0, 2.0, np.nan])                 # frozen-covariance ties + a NaN
    ce = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2])
    headline = spearman_rho(torch.as_tensor(tr), torch.as_tensor(ce))
    assert figs._spearman(tr, ce) == headline
    assert not hasattr(figs, "_spearman_np")
