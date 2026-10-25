"""Resolve launcher configuration without constructing a model or starting training."""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from vfe3.config import VFE3Config, serialize_config
from vfe3.exact_route import principled_exact_route_overrides, principled_exact_route_failures
from vfe3.gauge_covariant import fully_gauge_covariant_overrides
import train_vfe3 as launch
raw = dict(launch.config)
overrides = (principled_exact_route_overrides(raw) if launch.principled_exact_route
             else fully_gauge_covariant_overrides(raw) if launch.fully_gauge_covariant else {})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cfg = VFE3Config(**{**raw, **overrides, "seed": launch.SEEDS[0]})
record = {"principled_exact_route": launch.principled_exact_route, "fully_gauge_covariant": launch.fully_gauge_covariant,
          "raw": raw, "overrides": overrides, "resolved": serialize_config(cfg),
          "exact_route_failures": principled_exact_route_failures(cfg), "inactive": cfg.inactive_fields()}
Path(__file__).with_name("live-config.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
print(json.dumps({key: record[key] for key in ("principled_exact_route", "fully_gauge_covariant", "overrides", "exact_route_failures")}, indent=2))
