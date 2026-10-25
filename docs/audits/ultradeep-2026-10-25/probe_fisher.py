"""Tiny CPU Fisher chart-retraction probe; no model construction."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
torch.set_num_threads(1)
from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_fisher import FisherRowStatistics, fisher_spectrum_bounds
from vfe3.gauge_optim import GaugeManifoldAdamW
rows = []
for mean_scale in (0., 1e-6, 1e-8, 1e-9):
    p = torch.nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
    group = get_group("block_glk")(K=2, n_heads=1)
    stats = FisherRowStatistics(kind="type", count=torch.ones(1, dtype=torch.float64),
                               total=torch.tensor(1., dtype=torch.float64),
                               local_mean=torch.tensor([[[mean_scale, 0.]]], dtype=torch.float64),
                               local_var=torch.ones(1, 1, 2, dtype=torch.float64))
    def provider(param, selected_rows):
        return stats
    opt = GaugeManifoldAdamW([dict(params=[p], pullback_group=True, lr=.01, weight_decay=0)], group,
                            lr=.01, phi_group_trust_radius=.1, phi_chart_max_norm=5.,
                            phi_group_retraction="chart", phi_statistics_provider=provider,
                            phi_fisher_completion="stabilizer", weight_decay=0)
    p.grad = torch.eye(2, dtype=torch.float64).reshape(1, 4)
    lo, hi = fisher_spectrum_bounds(torch.tensor([mean_scale**2], dtype=torch.float64), 2,
                                   rotation_weight=2., completion="stabilizer")
    row = {"mean_scale": mean_scale, "spectrum_min": lo.tolist(), "spectrum_max": hi.tolist()}
    try:
        opt.step()
        row["updated_parameter"] = p.detach().tolist()
    except Exception as exc:
        row.update(error=type(exc).__name__, message=str(exc))
    rows.append(row)
Path(__file__).with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rows, indent=2))
