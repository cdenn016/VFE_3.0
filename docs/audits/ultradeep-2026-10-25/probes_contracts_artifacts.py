"""Coordinator-only tiny CPU checks of mode transitions and tied-bank reporting."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback
import warnings
from typing import Any, Callable

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VFE3_TEST_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tests.profiles import test_config
from vfe3.model.model import VFEModel
from vfe3.run_resources import parameter_breakdown, frozen_parameter_names
from vfe3.free_energy import attention_tau, _SQRT_D_CACHE


def attempt(action: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = action()
        return {"status": "completed", "value": value}
    except Exception as error:
        return {"status": "error", "type": type(error).__name__, "message": str(error),
                "trace": traceback.format_exc().splitlines()[-12:]}


def model_inference(tokens: list[int], mode: str) -> None:
    model = VFEModel(test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1,
                                max_seq_len=3, use_block_mlp=False)).eval()
    scope = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with scope:
        model(torch.tensor([tokens]))


def temperature_transition() -> None:
    _SQRT_D_CACHE.clear()
    with torch.inference_mode():
        attention_tau(1.0, [1, 3])
    attention_tau(torch.ones(2, requires_grad=True), [1, 3]).sum().backward()


def rope_transition() -> None:
    model = VFEModel(test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1,
                                max_seq_len=3, use_block_mlp=False, rope_mode="frame"))
    with torch.inference_mode():
        rotation = model._rope_rotation(3, torch.device("cpu"))
    cached = model._rope_rotation(3, torch.device("cpu"))
    assert rotation is cached
    (cached @ torch.ones_like(cached, requires_grad=True)).sum().backward()


def parameter_counts() -> dict[str, Any]:
    rows = []
    for tied in (False, True):
        cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=3, max_seq_len=3,
                          attention_mode="projected_predictive",
                          block_mlp_mode="cue_result_memory", block_mlp_tie_layers=tied,
                          family="gaussian_full", decode_mode="full_chunked",
                          predictive_readout="minimal_agent", minimal_agent_inference="exact",
                          block_mlp_inference="exact", block_mlp_covariance="posterior",
                          block_coordinates="reference", norm_type_block="none",
                          block_memory_regression_rank=2, block_mlp_expansion=2)
        model = VFEModel(cfg)
        state = model.state_dict()
        named = parameter_breakdown(dict(model.named_parameters()), cfg)
        saved = parameter_breakdown(state, cfg, frozen=frozen_parameter_names(cfg, state))
        rows.append({"tied": tied, "named_total": named["total"], "state_total": saved["total"],
                     "named_bank": named["context_processing"]["block_mlp"],
                     "state_bank": saved["context_processing"]["block_mlp"],
                     "bank_numel": sum(p.numel() for p in model.block_mlps[0].parameters()),
                     "buffers": {n: t.numel() for n, t in model.named_buffers()}})
    return {"counts": rows}


def tied_prototype_probe() -> Any:
    from vfe3.posthoc.feature_probes import prototype_probe
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=2, max_seq_len=3,
                      block_mlp_mode="prototype_agents", block_mlp_tie_layers=True,
                      block_mlp_expansion=2)
    model = VFEModel(cfg)
    return prototype_probe(model, [], torch.device("cpu"), gap_tokens=0)


def main() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    torch.manual_seed(7)
    results = {"model_modes": {f"{mode}_{tokens}": attempt(lambda t=tokens, m=mode: model_inference(t, m))
                               for mode in ("no_grad", "inference")
                               for tokens in ([1, 1, 2], [1, 2, 3])},
               "temperature_cache": attempt(temperature_transition),
               "rope_cache": attempt(rope_transition),
               "parameter_counts": attempt(parameter_counts),
               "tied_prototype_probe": attempt(tied_prototype_probe)}
    result_path = Path(__file__).with_suffix(".json")
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
