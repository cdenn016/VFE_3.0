r"""Log-cadence prototype-template rows and the persisted working-variance calibration (2026-10-09).

The eval-cadence ``prototype_templates.csv`` missed a template collapse that finished inside the first eval
interval (``docs/experiments/2026-10-09_prototype-K128-workvar-collapse.md``). Tiny CPU model, one thread.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.test_audit_20261007_opus_fixes import B, N, V, _prototype_cfg
from vfe3.data.datasets import TokenWindows
from vfe3.model.block_mlp import PrototypeAgentBlockMLP
from vfe3.model.model import VFEModel

torch.set_num_threads(1)


def test_log_steps_write_template_rows_and_calibration_is_saved(tmp_path: Path) -> None:
    cfg = _prototype_cfg(block_mlp_template_init="working_variance", max_steps=3, warmup_steps=3, eval_stride=None)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    loader = DataLoader(TokenWindows(torch.arange(41) % V, N), batch_size=B, shuffle=False, drop_last=True)
    T.train(model, loader, cfg, n_steps=3, log_interval=1, eval_interval=0, device=torch.device("cpu"),
            logger=logging.getLogger("prototype-log-test"), generate_samples=False, artifacts=artifacts)

    with (artifacts.run_dir / "diagnostics" / "prototype_templates_train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r["step"]), int(r["layer"])) for r in rows] == [
        (s, l) for s in (1, 2, 3) for l in range(cfg.n_layers)]
    assert all(1.0 <= float(r["effective_templates"]) and int(r["n_tokens"]) == B * N for r in rows)
    assert all(type(mlp).forward_moments is PrototypeAgentBlockMLP.forward_moments
               and "forward_moments" not in vars(mlp) for mlp in model.block_mlps)       # wrappers removed

    calibration = json.loads((artifacts.run_dir / "prototype_calibration.json").read_text())
    assert [entry["layer"] for entry in calibration["layers"]] == list(range(cfg.n_layers))
    assert all(entry["lambda0"] > 0 and entry["working_variance"] > 0 for entry in calibration["layers"])

    from vfe3.viz import report
    written = {p.name for p in report.generate_history_figures(artifacts.run_dir, cfg=cfg)}
    assert "prototype_templates.png" in written                       # drawn from the log-cadence table alone


def test_template_figure_overlays_training_and_eval_tables(tmp_path: Path) -> None:
    import numpy as np

    from vfe3.viz import feature_figures

    def table(steps):
        n = len(steps) * 2
        cols = {"step": np.repeat(steps, 2).astype(float), "layer": np.tile([0, 1], len(steps)).astype(float)}
        for name in T.PROTOTYPE_TEMPLATE_FIELDS[2:]:
            cols[name] = np.linspace(0.1, 1.0, n)
        return cols

    path = tmp_path / "templates.png"
    feature_figures.plot_prototype_templates(table([10, 20]), train_table=table([1, 5, 10, 15, 20]), path=str(path))
    assert path.stat().st_size > 0
    feature_figures.plot_prototype_templates(None, train_table=table([1, 5]), path=str(path))
