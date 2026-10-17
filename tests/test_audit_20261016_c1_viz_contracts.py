r"""Visualization and metric contracts, 2026-10-16 deep audit, lane C1 (findings 82 to 86).

F82: ``readout_diagnostics.png`` never drew its two canonical-F panels. ``_save_figures`` passed only the
``relational_*`` columns and four routing scalars to the dashboard, and the canonical panel read a ``total``
column that ``metrics.csv`` writes as ``inner_alignment_energy_total``. The readout history now carries the
canonical F terms and router columns, with ``total`` read from ``inner_alignment_energy_total``.

F83: the canonical free-energy figures paired the training-batch F of the log step with the validation CE in
one Pearson correlation, while the held-out ``val_*`` F columns of the same evaluation rows went unplotted; the
held-out chart-radius and vertex-conditioning columns were selected for ``validation_sanity.png`` and drawn by no
panel. The figures now read the held-out columns when they exist and name the population they plot, and the
held-out dashboard draws the chart radius and vertex conditioning.

F84: report mode ran the strict ``generate_figures`` replay before the lenient history re-render, so a run whose
saved config this build refuses lost its history dashboards with the replay. The history now renders whether
or not the replay succeeds.

F85: ``plot_scaling_law`` printed :math:`R^2` for the offset law at four sizes, which leave its three parameters
one residual degree of freedom; the gate now uses ``scaling_min_testable_sizes`` of the fitted form.

F86: ``lr_grid_heatmap_kwargs`` stored a ``None`` baseline that the heatmap then unpacked; the inference-bank
copy dropped ``CanonicalFrameContext.block_diagonal``; ``summary.json`` wrote three ``estep_final_*`` keys that
nothing produces, so they were always null.

Tiny CPU models only (K=4, two heads, N=4, vocab 7).
"""
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
import vfe3.viz.figures as figs
from tests.profiles import test_config
from tests.test_audit2_20261011_training import _cfg
from vfe3.config import serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows
from vfe3.metric_contracts import scaling_min_testable_sizes
from vfe3.model.model import VFEModel
from vfe3.viz import extract, figure_worker, sweep_adapters

B, N, V = 2, 4, 7
CPU = torch.device("cpu")
LOGGER = logging.getLogger("audit-c1-viz-contracts")


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _titles(fig: Any) -> List[str]:
    return [ax.get_title() for ax in fig.axes]


def _texts(fig: Any) -> str:
    parts = [text.get_text() for text in fig.texts]
    for ax in fig.axes:
        parts.append(ax.get_title())
        legend = ax.get_legend()
        if legend is not None:
            parts.extend(text.get_text() for text in legend.get_texts())
    return "\n".join(parts)


def _canonical_row(index: int, *, held_out: bool = True, eval_row: bool = True) -> Dict[str, float]:
    r"""One metrics row of a ``diagnose_canonical_free_energy`` run: train-batch F terms on every row, the
    held-out ``val_*`` F terms and ``val_ce`` on the evaluation rows only."""
    nan = float("nan")
    row: Dict[str, float] = {
        "step": float(10 * (index + 1)),
        "relational_legacy_global_f_available": 1.0,
        "relational_score_klmax_frac": 0.01 * index,
        "attn_entropy": 1.2,
        "self_coupling": 20.0 + index,
        "self_divergence": 6.0 + 0.1 * index,
        "belief_coupling": 3.0,
        "attention_entropy": 1.0,
        "inner_alignment_energy_total": 24.0 + index,
        "canonical_total_output": 23.0 + index,
        "canonical_total_delta": -1.0,
        "canonical_beta_tv": 0.6 - 0.01 * index,
        "canonical_self_weight": 0.3,
        "val_ce": (5.0 - 0.1 * index) if eval_row else nan,
    }
    if held_out:
        row.update({
            "val_self_coupling": (2.0 + 0.1 * index) if eval_row else nan,
            "val_self_divergence": 0.5 if eval_row else nan,
            "val_belief_coupling": 0.3 if eval_row else nan,
            "val_attention_entropy": 0.1 if eval_row else nan,
            "val_inner_alignment_energy_total": (2.4 + 0.1 * index) if eval_row else nan,
            "val_canonical_beta_tv": 0.2 if eval_row else nan,
            "val_canonical_self_weight": 0.4 if eval_row else nan,
        })
    return row


# ------------------------------------------------------------------------------------------------ F82

def test_readout_diagnostics_draws_the_canonical_panels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}
    original = figs.plot_readout_diagnostics

    def spy(history: Dict, **kwargs: Any) -> Any:
        fig = original(history, **kwargs)
        captured.update(history=history, titles=_titles(fig))
        return fig

    monkeypatch.setattr(figs, "plot_readout_diagnostics", spy)
    history = [_canonical_row(index) for index in range(4)]
    ra._save_figures(SimpleNamespace(run_dir=tmp_path, cfg=None, history=history), None, LOGGER)
    assert "Canonical F at the captured beliefs (diagnostic)" in captured["titles"]
    assert "Model router against the canonical router" in captured["titles"]
    readout = captured["history"]
    assert readout["total"] == [row["inner_alignment_energy_total"] for row in history]
    for key in ("canonical_total_output", "self_coupling", "self_divergence", "belief_coupling",
                "attention_entropy", "canonical_beta_tv", "canonical_self_weight"):
        assert readout[key] == [row[key] for row in history], key
    assert (tmp_path / "readout_diagnostics.png").is_file()


def test_readout_diagnostics_keeps_no_canonical_panel_without_the_diagnostic(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}
    original = figs.plot_readout_diagnostics

    def spy(history: Dict, **kwargs: Any) -> Any:
        fig = original(history, **kwargs)
        captured.update(history=history, titles=_titles(fig))
        return fig

    monkeypatch.setattr(figs, "plot_readout_diagnostics", spy)
    nan = float("nan")
    history = [{**_canonical_row(index, held_out=False),
                **{key: nan for key in ("self_coupling", "self_divergence", "belief_coupling",
                                        "attention_entropy", "inner_alignment_energy_total")}}
               for index in range(4)]
    for row in history:                                   # the canonical columns exist only under the toggle
        for key in ("canonical_total_output", "canonical_total_delta", "canonical_beta_tv",
                    "canonical_self_weight"):
            del row[key]
    ra._save_figures(SimpleNamespace(run_dir=tmp_path, cfg=None, history=history), None, LOGGER)
    assert "Canonical F at the captured beliefs (diagnostic)" not in captured["titles"]
    assert "total" not in captured["history"]
