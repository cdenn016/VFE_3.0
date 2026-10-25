"""Additional tiny CPU audit probes, with all runtime mutations process-local."""
from __future__ import annotations
import dataclasses
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from tests.profiles import test_config
from vfe3.model.cg_coupling import CGCoupling
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior
result: dict[str, Any] = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result["inactive_activation"] = {
        str(depth): "block_mlp_activation" in test_config(n_layers=depth, use_block_mlp=True,
                      block_mlp_mode="prototype_agents").inactive_fields()
        for depth in (1, 2)}
try:
    c = CGCoupling(3, "so", [1, 3], ["l0", "l1"])
    with torch.no_grad():
        c.path_weights.fill_(0.1)
    mu, sigma = torch.arange(4, dtype=torch.float32), torch.ones(4)
    with torch.inference_mode():
        c(mu, sigma)
    c(mu.requires_grad_(), sigma)[0].sum().backward()
    result["cg_inference_cache"] = {"backward": "passed"}
except Exception as exc:
    result["cg_inference_cache"] = {"error": type(exc).__name__, "message": str(exc)}

tile_rows = []
for dense in (False, True):
    for tile in (1, 1000000):
        x, c, z = torch.full((1, 1), 1e4), torch.ones(1, 1), torch.zeros(2, 1)
        fn = cue_result_exact_posterior if dense else cue_result_diagonal_posterior
        p = fn(x, torch.diag_embed(c) if dense else c, z, z, z, z, torch.zeros(2),
               regression_offset=torch.zeros(2, 1, 1), tile_bytes=tile)
        tile_rows.append({"dense": dense, "tile_bytes": tile, "mean": p.mean.tolist(),
                          "covariance": p.covariance.tolist(), "weight_sum": p.log_weights.exp().sum(-1).tolist()})
result["cue_logmass_tiling"] = tile_rows
try:
    from scaling import predict_n_params
    from vfe3.model.model import VFEModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=2, block_mlp_expansion=2,
                          use_block_mlp=True, block_mlp_mode="coordinate", per_type_hidden_bias=True)
        rows = []
        for enabled in (False, True):
            cc = dataclasses.replace(cfg, per_type_hidden_bias=enabled)
            rows.append({"per_type_hidden_bias": enabled, "predicted": predict_n_params(cc),
                         "actual": sum(p.numel() for p in VFEModel(cc).parameters())})
    result["hidden_bias_parameter_count"] = rows
except Exception as exc:
    result["hidden_bias_parameter_count"] = {"error": type(exc).__name__, "message": str(exc)}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
