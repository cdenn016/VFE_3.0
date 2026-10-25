"""Small, serial CPU reproductions for the October 25 audit; no training or GPU use."""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch

torch.set_num_threads(1)
RESULTS: dict[str, Any] = {}


def record(name: str, fn: Callable[[], Any]) -> None:
    try:
        RESULTS[name] = fn()
    except Exception as exc:
        RESULTS[name] = {"error": type(exc).__name__, "message": str(exc)}


def cue_cancellation() -> dict[str, Any]:
    from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior
    x, c, z = torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1)
    args = dict(nu=z, omega=z, log_lambda=z, log_result_noise=torch.full((1, 1), -25.), log_pi=torch.zeros(1))
    low = cue_result_diagonal_posterior(x, c, **args, regression_left=torch.ones(1, 1, 1),
                                      regression_right=-torch.ones(1, 1, 1))
    dense = cue_result_diagonal_posterior(x, c, **args, regression_offset=-torch.ones(1, 1, 1))
    full = cue_result_exact_posterior(x, torch.diag_embed(c), **args, regression_offset=-torch.ones(1, 1, 1))
    return {"expected_noise": args["log_result_noise"].exp().tolist(), "lowrank": low.covariance.tolist(),
            "dense_regression": dense.covariance.tolist(), "dense_record": full.covariance.tolist()}


def cue_large_logmass() -> list[dict[str, Any]]:
    from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior
    out = []
    for dtype in (torch.float32, torch.float64):
        for dense in (False, True):
            x, c, z = torch.full((1, 1), 1e4, dtype=dtype), torch.ones(1, 1, dtype=dtype), torch.zeros(2, 1, dtype=dtype)
            fn = cue_result_exact_posterior if dense else cue_result_diagonal_posterior
            posterior = fn(x, torch.diag_embed(c) if dense else c, z, z, z, z, torch.zeros(2, dtype=dtype),
                           regression_offset=torch.zeros(2, 1, 1, dtype=dtype))
            out.append({"dtype": str(dtype), "dense": dense, "mean": posterior.mean.tolist(),
                        "covariance": posterior.covariance.tolist(),
                        "weight_sum": posterior.log_weights.exp().sum(-1).tolist()})
    return out


def cue_dense_budget() -> dict[str, Any]:
    import vfe3.model.cue_result_memory as memory
    calls = []
    original = memory._chunk_moments
    def measured(*args: Any, **kwargs: Any) -> Any:
        xi, _, nu = args[:3]
        calls.append({"tokens": xi.shape[0], "components": nu.shape[0],
                      "estimated_tile_bytes": 6 * xi.shape[0] * nu.shape[0] * xi.shape[-1] ** 2 * xi.element_size()})
        return original(*args, **kwargs)
    x, C, z = torch.zeros(8, 2), torch.eye(2).expand(8, 2, 2), torch.zeros(3, 2)
    budget = 6 * 2 * 2 * 4
    memory._chunk_moments = measured
    try:
        memory.cue_result_exact_posterior(x, C, z, z, z, z, torch.zeros(3),
                                          regression_offset=torch.zeros(3, 2, 2), tile_bytes=budget)
    finally:
        memory._chunk_moments = original
    return {"requested_bytes": budget, "calls": calls}


def cg_train_eval() -> dict[str, Any]:
    from vfe3.model.cg_coupling import CGCoupling
    c = CGCoupling(3, "so", [1, 3], ["l0", "l1"], cg_covariance_mode="delta_full", cg_covariance_floor=0.5).double()
    mu, sigma = torch.tensor([1., 2., 3., 4.], dtype=torch.float64), torch.eye(4, dtype=torch.float64)
    training = c(mu, sigma)
    with torch.no_grad():
        evaluation = c(mu, sigma)
    return {"grad_enabled_covariance_diagonal": training[1].diagonal().tolist(),
            "no_grad_covariance_diagonal": evaluation[1].diagonal().tolist(),
            "max_difference": (training[1] - evaluation[1]).abs().max().item()}


def local_ridge_noise() -> dict[str, Any]:
    from vfe3.local_learner.core import init_params
    from vfe3.local_learner.learners import Stats, m_step
    st = Stats.zeros(1, 1, 2)
    st.pp.fill_(1.0)
    st.zp.fill_(1.0)
    st.zz.fill_(2.0)
    st.cnt.fill_(0.5)
    prm = init_params(1, 2, 0)
    prm.G.fill_(0.5)
    prm.E.zero_()
    prm.Q.fill_(1.25)
    result = m_step(st, prm, ridge=1.0, q_floor=1e-4)
    W = torch.cat([result.G, result.E.transpose(-1, -2)], -1)
    Sxx = torch.diag_embed(torch.tensor([[1.0, 0.5, 0.5]], dtype=torch.float64))
    Szx = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    residual = st.zz - W @ Szx.transpose(-1, -2) - Szx @ W.transpose(-1, -2) + W @ Sxx @ W.transpose(-1, -2)
    def energy(q: float) -> float:
        return 0.5 * (math.log(q) + residual.item() / q)
    return {"fitted_G": result.G.item(), "computed_Q": result.Q.item(), "expected_residual_Q": residual.item(),
            "conditional_noise_energy_before": energy(prm.Q.item()),
            "conditional_noise_energy_after": energy(result.Q.item())}


def test_module_dtype() -> dict[str, Any]:
    before = torch.get_default_dtype()
    importlib.import_module("tests.test_local_learner_20261021")
    after = torch.get_default_dtype()
    torch.set_default_dtype(before)
    return {"before_import": str(before), "after_import": str(after)}


record("cue_lowrank_cancellation", cue_cancellation)
record("cue_large_logmass", cue_large_logmass)
record("cue_dense_budget", cue_dense_budget)
record("cg_train_eval", cg_train_eval)
record("local_ridge_noise", local_ridge_noise)
record("test_collection_default_dtype", test_module_dtype)
RESULTS["environment"] = {"torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
target = Path(__file__).with_name("probes_initial.json")
target.write_text(json.dumps(RESULTS, indent=2) + "\n", encoding="utf-8")
print(json.dumps(RESULTS, indent=2))
