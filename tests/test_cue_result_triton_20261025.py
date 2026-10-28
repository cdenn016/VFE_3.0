r"""Fused Triton route of the cue/result memory on diagonal records (``block_memory_triton``, 2026-10-25).

The fused route (vfe3/kernels/cue_result.py) runs the exact posterior of ``cue_result_diagonal_posterior`` and its
backward in Triton register tiles for a low-rank regression. These tests pin:

* the field is a bool, is inactive off the memory, is refused off ``gaussian_frame_diagonal`` and without a low-rank
  regression of rank at most ``MAX_RANK``, sits in both launchers, and reaches every memory module;
* CPU tensors run the torch tiles whatever the toggle (bitwise the same loss and gradients) and never import triton;
* under the Triton interpreter (CPU, ``TRITON_INTERPRET=1``) the fused kernels reproduce the torch route's mean,
  variances, log weights, log evidence and every gradient (float64 to 1e-11; float32 as close to a float64 reference
  as the torch route's float32 within a factor 4) on ragged shapes, ranks 1 to 4, shared and per-token priors and
  several row chunks, and the tied model with a carried regime gives the torch route's loss and gradients;
* the five kernels compile offline for the RTX 5090 target (sm_120) at ranks 2 and 4, with no register spills at
  rank 2 (at rank 4 the column backward spilled 212 bytes when written; the counts are recorded).

The interpreter and compile checks run in subprocesses, so the test process never imports triton. Nothing runs on the
GPU here.
"""
import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from tests.profiles import config_values, test_config
from vfe3.model.block_mlp import CueResultMemoryBlockMLP
from vfe3.model.model import VFEModel

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton (or ptxas) unavailable

_ROUTE = dict(attention_mode="projected_predictive", predictive_readout="kernel", family="gaussian_frame_diagonal",
              decode_mode="linear", norm_type_block="none", block_coordinates="reference", use_block_mlp=True,
              block_mlp_mode="cue_result_memory", block_mlp_inference="exact", block_mlp_covariance="posterior",
              block_mlp_dropout=0.0, block_mlp_expansion=2, n_layers=2, max_seq_len=4,
              block_memory_regression_rank=2)


def _cfg(**overrides: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**{**_ROUTE, **overrides})


def _subprocess_env(**extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", **extra)
    return env


def _run_script(script: str, env: Dict[str, str]) -> Any:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton or ptxas unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------- config and wiring

def test_field_is_validated_inactive_off_the_memory_and_refused_off_its_route() -> None:
    assert config_values()["block_memory_triton"] is False
    with pytest.raises(ValueError, match="block_memory_triton must be"):
        _cfg(block_memory_triton=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert "block_memory_triton" in test_config(block_memory_triton=True).inactive_fields()
    assert "block_memory_triton" not in _cfg(block_memory_triton=True).inactive_fields()
    with pytest.raises(ValueError, match="low-rank"):
        _cfg(block_memory_triton=True, block_memory_regression_rank=None)
    with pytest.raises(ValueError, match="gaussian_frame_diagonal"):
        _cfg(block_memory_triton=True, family="gaussian_full")


@pytest.mark.parametrize("launcher", ["train_vfe3.py", "ablation.py"])
def test_launchers_carry_the_field_with_its_options(launcher: str) -> None:
    line = next(line for line in (REPO / launcher).read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("block_memory_triton"))
    assert "True | False" in line


@pytest.mark.parametrize("over", (dict(), dict(block_mlp_tie_layers=True, block_memory_regime_transition="sticky")))
def test_field_reaches_every_memory_module(over: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_cfg(block_memory_triton=True, **over))
    assert all(isinstance(m, CueResultMemoryBlockMLP) and m.fused_kernel is True for m in model.block_mlps)


def test_cpu_tensors_run_the_torch_tiles_whatever_the_toggle() -> None:
    results = []
    for fused in (False, True):
        cfg = _cfg(block_memory_triton=fused, block_mlp_tie_layers=True, block_memory_regime_transition="sticky")
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg)
            tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(3))
            model.train()
            _, loss, _ = model(tokens, tokens)
        loss.backward()
        results.append((loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (loss_a, g_a), (loss_b, g_b) = results
    assert torch.equal(loss_a, loss_b) and g_a.keys() == g_b.keys()
    assert all(torch.equal(g_a[n], g_b[n]) for n in g_a)


def test_cpu_route_never_imports_triton() -> None:
    script = """
    import json, sys, warnings, torch
    warnings.simplefilter("ignore")
    from tests.test_cue_result_triton_20261025 import _cfg
    from vfe3.model.model import VFEModel
    cfg = _cfg(block_memory_triton=True)
    model = VFEModel(cfg)
    _, loss, _ = model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))
    loss.backward()
    print(json.dumps({"triton": "triton" in sys.modules, "kernels": "vfe3.kernels._cue_result_triton" in sys.modules}))
    """
    assert _run_script(script, _subprocess_env()) == {"triton": False, "kernels": False}


# ------------------------------------------------------------------------------------- Triton interpreter

_INTERPRETER_SCRIPT = """
import json, sys, warnings
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
warnings.simplefilter("ignore")
from vfe3.kernels import _cue_result_triton as tk
from vfe3.kernels import cue_result as fused_kernels
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior
assert tk.INTERPRETED
fused_kernels.fused_route_active = lambda xi: True        # the interpreter runs the fused route on CPU tensors

def operands(R, M, K, rank, per_token, seed):
    g = torch.Generator().manual_seed(seed)
    d = torch.float64
    ops = {"xi": torch.randn(R, K, generator=g, dtype=d), "c": 0.3 + torch.rand(R, K, generator=g, dtype=d),
           "nu": 1.5 * torch.randn(M, K, generator=g, dtype=d), "omega": torch.randn(M, K, generator=g, dtype=d),
           "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=d),
           "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=d) - 1.0,
           "log_pi": torch.randn(*((R,) if per_token else ()), M, generator=g, dtype=d),
           "regression_left": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d),
           "regression_right": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d)}
    cot = [torch.randn(R, K, generator=g, dtype=d), torch.randn(R, K, generator=g, dtype=d),
           torch.randn(R, M, generator=g, dtype=d), torch.randn(R, generator=g, dtype=d)]
    return {k: v.float().double() for k, v in ops.items()}, [t.float().double() for t in cot]

def run(ops, cot, dtype, fused, tile):
    leaves = {k: v.to(dtype).requires_grad_(True) for k, v in ops.items()}
    out = cue_result_diagonal_posterior(leaves["xi"], leaves["c"], leaves["nu"], leaves["omega"],
                                        leaves["log_lambda"], leaves["log_result_noise"], leaves["log_pi"],
                                        regression_left=leaves["regression_left"],
                                        regression_right=leaves["regression_right"], tile_bytes=tile, fused=fused)
    outs = (out.mean, out.covariance, out.log_weights, out.log_evidence)
    loss = sum((o * c.to(dtype)).sum() for o, c in zip(outs, cot))
    grads = torch.autograd.grad(loss, list(leaves.values()))
    return {**dict(zip(("mean", "var", "log_weights", "log_evidence"), outs)), **dict(zip(leaves, grads))}

def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300))

results = []
for R, M, K, rank, per_token, tile in ((5, 7, 3, 1, False, 10 ** 9), (6, 37, 35, 2, True, 6000),
                                       (3, 33, 5, 3, False, 10 ** 9), (4, 9, 6, 4, True, 2000)):
    ops, cot = operands(R, M, K, rank, per_token, seed=R + M)
    ref = run(ops, cot, torch.float64, False, 10 ** 9)
    entry = {"shape": [R, M, K, rank, per_token, tile]}
    for name, dtype, fused in (("fused64", torch.float64, True), ("torch32", torch.float32, False),
                               ("fused32", torch.float32, True)):
        got = run(ops, cot, dtype, fused, tile)
        entry[name] = {k: rel(got[k], ref[k]) for k in ref}
    results.append(entry)

# the tied model with a carried regime, fused against torch, float64
from tests.test_cue_result_triton_20261025 import _cfg
from vfe3.model.model import VFEModel
cfg = _cfg(block_memory_triton=True, block_mlp_tie_layers=True, block_memory_regime_transition="sticky", n_layers=3)
tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(9))
model_runs = []
for fused in (False, True):
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    model.block_mlps[0].fused_kernel = fused
    model.train()
    _, loss, _ = model(tokens, tokens)
    loss.backward()
    model_runs.append((float(loss), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
(l0, g0), (l1, g1) = model_runs
model_entry = {"loss": abs(l1 - l0) / abs(l0), "same_keys": sorted(g0) == sorted(g1),
               "grad": max(rel(g1[n], g0[n]) for n in g0 if g0[n].abs().max() > 0)}
print(json.dumps({"kernels": results, "model": model_entry}))
"""


def test_triton_interpreter_matches_the_torch_route(record_property: Any) -> None:
    report = _run_script(_INTERPRETER_SCRIPT, _subprocess_env(TRITON_INTERPRET="1"))
    for entry in report["kernels"]:
        record_property(f"interp_{entry['shape']}", json.dumps(entry))
        for name, err in entry["fused64"].items():
            assert err <= 1e-11, (name, entry)
        for name, err in entry["fused32"].items():           # as close to float64 as the torch route itself
            assert err <= 4.0 * entry["torch32"][name] + 1e-6, (name, entry)
    model = report["model"]
    record_property("interp_model", json.dumps(model))
    assert model["same_keys"] and model["loss"] <= 1e-12 and model["grad"] <= 1e-10, model


# ------------------------------------------------------------------------------------- sm_120 compile

_COMPILE_SCRIPT = """
import json, os, re, subprocess, sys, tempfile
try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    ptxas = get_ptxas().path
except Exception as exc:
    print("triton/ptxas unavailable:", exc)
    sys.exit(77)
from vfe3.kernels import _cue_result_triton as tk
assert not tk.INTERPRETED
cfgs = tk.launch_config()
kernels = {"fwd_weights": tk.cr_fwd_weights_kernel, "fwd_moments": tk.cr_fwd_moments_kernel,
           "bwd_pairs": tk.cr_bwd_pairs_kernel, "bwd_rows": tk.cr_bwd_rows_kernel, "bwd_cols": tk.cr_bwd_cols_kernel}
ints = {"R", "M", "K", "NKT", "LP_STRIDE", "ROWS_PER_SPLIT"}
out = {}
for rank in (2, 4):
    for key, kernel in kernels.items():
        cfg = cfgs[key]
        consts = {"BA": cfg["BA"], "BK": cfg["BK"], "RANK": rank, "RP": 1 << (rank - 1).bit_length(), "ACC": True}
        if "WRITE_LOGW" in kernel.arg_names:                    # the forward's variant (it writes the log weights)
            consts["WRITE_LOGW"] = True
        sig = {n: ("constexpr" if n in consts else "i32" if n in ints else "fp32" if n == "HALF_K_LOG2PI"
                   else "*fp32") for n in kernel.arg_names}
        compiled = triton.compile(ASTSource(fn=kernel, signature=sig, constexprs=consts),
                                  target=GPUTarget("cuda", 120, 32),
                                  options={"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]})
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "k.ptx")
            with open(src, "w") as f:
                f.write(compiled.asm["ptx"])
            log = subprocess.run([ptxas, "-v", "--gpu-name=sm_120a", src, "-o", src + ".o"],
                                 capture_output=True, text=True)
        text = log.stdout + log.stderr
        spill = re.search(r"(\\d+) bytes spill stores, (\\d+) bytes spill loads", text)
        out[f"{key}_rank{rank}"] = {"registers": int(re.search(r"Used (\\d+) registers", text).group(1)),
                                    "spill_stores": int(spill.group(1)), "spill_loads": int(spill.group(2))}
print(json.dumps(out))
"""


def test_kernels_compile_for_sm120(tmp_path: Path, record_property: Any) -> None:
    report = _run_script(_COMPILE_SCRIPT, _subprocess_env(TRITON_CACHE_DIR=str(tmp_path / "triton_cache")))
    assert len(report) == 10
    for key, r in report.items():
        record_property(f"sm120_{key}", json.dumps(r))
        assert r["registers"] <= 255, (key, r)
        if key.endswith("rank2"):                               # the launcher's rank: no spills in any kernel
            assert r["spill_stores"] == 0 and r["spill_loads"] == 0, (key, r)


# ------------------------------------------------------------------------------------- CUDA (opt-in lane)

requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")


@requires_cuda
def test_cuda_kernels_match_a_float64_reference_and_record_timing(record_property: Any) -> None:
    import time

    from vfe3.model.cue_result_memory import cue_result_diagonal_posterior

    R, M, K, rank = 512, 384, 96, 2
    g = torch.Generator().manual_seed(0)
    ops = {"xi": torch.randn(R, K, generator=g), "c": 0.3 + torch.rand(R, K, generator=g),
           "nu": 1.5 * torch.randn(M, K, generator=g), "omega": torch.randn(M, K, generator=g),
           "log_lambda": 0.5 * torch.randn(M, K, generator=g),
           "log_result_noise": 0.5 * torch.randn(M, K, generator=g) - 1.0,
           "log_pi": torch.randn(R, M, generator=g),
           "regression_left": 0.7 * torch.randn(M, K, rank, generator=g),
           "regression_right": 0.7 * torch.randn(M, K, rank, generator=g)}
    cot = [torch.randn(R, K, generator=g), torch.randn(R, K, generator=g), torch.randn(R, M, generator=g),
           torch.randn(R, generator=g)]

    def run(device: str, dtype: torch.dtype, fused: bool) -> Dict[str, torch.Tensor]:
        leaves = {k: v.to(device, dtype).requires_grad_(True) for k, v in ops.items()}
        out = cue_result_diagonal_posterior(*(leaves[k] for k in ("xi", "c", "nu", "omega", "log_lambda",
                                                                   "log_result_noise", "log_pi")),
                                            regression_left=leaves["regression_left"],
                                            regression_right=leaves["regression_right"],
                                            tile_bytes=1024 ** 3, fused=fused)
        outs = (out.mean, out.covariance, out.log_weights, out.log_evidence)
        loss = sum((o * c.to(device, dtype)).sum() for o, c in zip(outs, cot))
        grads = torch.autograd.grad(loss, list(leaves.values()))
        return {**dict(zip(("mean", "var", "log_weights", "log_evidence"), outs)), **dict(zip(leaves, grads))}

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.double().cpu() - b).abs().max() / b.abs().max().clamp_min(1e-300))

    ref = run("cpu", torch.float64, False)
    timing = {}
    for name, fused in (("torch", False), ("fused", True)):
        run("cuda", torch.float32, fused)                                  # compile / warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        got = run("cuda", torch.float32, fused)
        torch.cuda.synchronize()
        timing[name] = time.perf_counter() - t0
        timing[name + "_err"] = {k: rel(got[k], ref[k]) for k in ref}
    record_property("cuda_timing", json.dumps(timing))
    for k, err in timing["fused_err"].items():
        assert err <= 4.0 * timing["torch_err"][k] + 1e-6, (k, timing)
