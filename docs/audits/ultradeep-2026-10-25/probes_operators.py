"""Tiny finite-input numerical probes; Triton runs only in its CPU interpreter."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRITON_INTERPRET"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
result: dict[str, Any] = {}
from vfe3.geometry.retraction import retract_spd_diagonal, retract_spd_full
rows = []
for magnitude in (1e18, 1e19, 2e19, 1e20):
    for dtype in (torch.float32, torch.float64):
        diag = retract_spd_diagonal(torch.ones(2, dtype=dtype), torch.full((2,), magnitude, dtype=dtype),
                                    trust_region=1, sigma_max=None)
        full = retract_spd_full(torch.eye(2, dtype=dtype), magnitude * torch.eye(2, dtype=dtype),
                               trust_region=1, sigma_max=None)
        rows.append({"magnitude": magnitude, "dtype": str(dtype), "diagonal": diag.tolist(), "full": full.tolist()})
result["retraction_overflow"] = rows
try:
    from vfe3.kernels.expected_likelihood import expected_likelihood_ce
    from vfe3.kernels import _expected_likelihood_triton as tk
    mq, sq = torch.zeros(1, 1), torch.full((1, 1), 0.5)
    mv, sv = torch.full((65, 1), 3e19), torch.full((65, 1), 0.5)
    mv[-1] = 0
    logits = -0.5 * (((mq - mv).square() / (sq + sv)) + (sq + sv).log()).sum(-1)
    reference = torch.logsumexp(logits, 0) - logits[-1]
    fused = expected_likelihood_ce(mq, sq, mv, sv, None, torch.tensor(1.), torch.tensor([64]))
    result["el_empty_prefix"] = {"cpu_interpreted": tk.INTERPRETED, "reference": float(reference),
                                  "fused": float(fused), "fused_finite": bool(torch.isfinite(fused))}
except Exception as exc:
    result["el_empty_prefix"] = {"error": type(exc).__name__, "message": str(exc)}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
