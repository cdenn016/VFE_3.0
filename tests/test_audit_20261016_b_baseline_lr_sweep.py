r"""Audit 2026-10-16 F118: the baseline learning-rate sweep flags a winner at the edge of its grid.

``train_baseline.lr_sweep`` adopts the learning rate with the lowest best validation PPL. A winner at the lowest or
highest swept rate does not bracket the optimum, which may lie outside the grid; the sweep result (written to
``lr_sweep.json``) now records ``at_grid_edge`` and ``main`` warns. A diverged rate above the winner brackets it
from above, so the edge is that of the requested grid. No model is trained: ``run_baseline`` is replaced by a stub
that writes ``summary.json``.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import train_baseline as tb
from vfe3.baseline import BaselineConfig


def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ppl_by_lr: Dict[float, Optional[float]]) -> None:
    def run_baseline(cfg: BaselineConfig, *, label: str, **_: Any) -> Path:
        run_dir = tmp_path / label
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps({"best_val_ppl": ppl_by_lr[cfg.lr]}), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(tb, "run_baseline", run_baseline)


def _sweep(ppl_by_lr: Dict[float, Optional[float]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Any]:
    _stub(monkeypatch, tmp_path, ppl_by_lr)
    return tb.lr_sweep(BaselineConfig(), list(ppl_by_lr), dataset="synthetic", out_root=tmp_path, seed=6,
                       max_steps=10)


def test_a_winner_at_the_top_of_the_grid_is_flagged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep = _sweep({1e-3: 60.0, 3e-3: 55.0, 6e-3: 50.0}, monkeypatch, tmp_path)
    assert sweep["best_lr"] == 6e-3
    assert sweep["at_grid_edge"] is True


def test_a_winner_at_the_bottom_of_the_grid_is_flagged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep = _sweep({3e-3: 55.0, 1e-3: 50.0, 6e-3: 60.0}, monkeypatch, tmp_path)   # unsorted grid
    assert sweep["best_lr"] == 1e-3
    assert sweep["at_grid_edge"] is True


def test_an_interior_winner_is_not_flagged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep = _sweep({1e-3: 60.0, 3e-3: 50.0, 6e-3: None}, monkeypatch, tmp_path)   # top rate diverged
    assert sweep["best_lr"] == 3e-3
    assert sweep["at_grid_edge"] is False
    json.dumps(sweep)                                                               # lr_sweep.json stays writable


def test_no_finite_winner_has_no_edge_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep = _sweep({1e-3: None, 3e-3: None}, monkeypatch, tmp_path)
    assert sweep["best_lr"] is None
    assert sweep["at_grid_edge"] is None
