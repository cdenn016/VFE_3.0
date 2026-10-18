r"""Fused Triton expected-likelihood cross-entropy (2026-10-10; the only CUDA route since 2026-10-12).

The fused route (vfe3/kernels/expected_likelihood.py) replaces the (B, N, Vc, K) broadcast of
``PriorBank.decode_ce_expected_likelihood_chunked`` on every CUDA tensor; CPU tensors run the chunked
torch kernel (the expected_likelihood_kernel selector was removed 2026-10-12). These tests pin:

* CPU tensors run the torch kernel, record it as the realized kernel, and never import triton;
* under the Triton interpreter (CPU, ``TRITON_INTERPRET=1``) the kernels reproduce the torch route's CE
  and every gradient on ragged shapes with ignored targets and the unigram bias (fp32 and fp64);
* the kernels compile offline for the RTX 5090 target (sm_120) with zero register-spill bytes;
* on CUDA (``CUDA_TESTS``) the compiled kernels match the float64 CPU torch route, are bitwise
  reproducible, and their timing is recorded.

The interpreter and compile checks run in subprocesses, so the test process never imports triton.
"""
from tests.profiles import make_prior_bank
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

from vfe3.model.prior_bank import PriorBank

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton (or ptxas) unavailable

# A CPU-lane run (VFE3_TEST_DEVICE=cpu) never touches the device, whatever torch.cuda.is_available() reports.
requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")


def _bank(dtype: torch.dtype, *, V: int, K: int, unigram: bool, seed: int = 0,
          chunk: int = 16) -> PriorBank:
    r"""A tiny expected-likelihood bank with non-degenerate tables (several vocabulary chunks)."""
    torch.manual_seed(seed)
    pb = make_prior_bank(V, K, 3, decode_mode="expected_likelihood_chunked", decode_chunk_size=chunk,
                   decode_unigram_prior=unigram, unigram_kappa=0.7, decode_tau=0.9).to(dtype)
    with torch.no_grad():
        pb.sigma_log_embed.normal_(0.0, 0.5)
        pb.mu_embed.normal_(0.0, 1.0)
        pb.decode_log_scale.fill_(0.2)
    if unigram:
        pb.set_unigram_log_prior(torch.randint(0, 50, (V,), generator=torch.Generator().manual_seed(seed)).to(dtype))
    return pb


def _ce_and_grads(pb: PriorBank, mu_q: torch.Tensor, sigma_q: torch.Tensor, targets: torch.Tensor,
                  **kwargs: Any) -> Tuple[Any, List[torch.Tensor]]:
    mu_q = mu_q.detach().clone().requires_grad_()
    sigma_q = sigma_q.detach().clone().requires_grad_()
    leaves = [mu_q, sigma_q] + [p for p in pb.parameters() if p.requires_grad]
    out = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets, **kwargs)
    ce = out.ce if hasattr(out, "ce") else out
    grads = torch.autograd.grad(ce, leaves, allow_unused=True)
    return out, [g for g in grads if g is not None]


def _subprocess_env(**extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), **extra)
    return env


def _run_script(script: str, env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton or ptxas unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_cpu_tensors_run_and_record_the_torch_kernel() -> None:
    V, K = 37, 5
    torch.manual_seed(1)
    mu_q = torch.randn(2, 7, K, dtype=torch.float64)
    sigma_q = 0.2 + torch.rand(2, 7, K, dtype=torch.float64)
    targets = torch.randint(0, V, (2, 7))
    targets[0, 2] = -100
    targets[1, 6] = -100
    pb = _bank(torch.float64, V=V, K=K, unigram=True)
    assert pb.realized_expected_likelihood_kernel is None              # nothing ran yet
    out, grads = _ce_and_grads(pb, mu_q, sigma_q, targets, return_stats=True)
    assert pb.realized_expected_likelihood_kernel == "torch"
    assert int(out.scored_tokens) == int((targets != -100).sum())
    assert torch.isfinite(out.ce) and len(grads) == 5


def test_cpu_route_never_imports_triton() -> None:
    script = """
    import json, sys, torch
    from tests.profiles import make_prior_bank
    pb = make_prior_bank(11, 3, 3, decode_mode="expected_likelihood_chunked")
    mu_q = torch.randn(1, 4, 3, requires_grad=True)
    ce = pb.decode_ce_expected_likelihood_chunked(mu_q, 0.5 + torch.rand(1, 4, 3), torch.tensor([[1, 2, -100, 4]]))
    ce.backward()
    print(json.dumps({"triton": "triton" in sys.modules,
                      "kernel_module": "vfe3.kernels.expected_likelihood" in sys.modules}))
    """
    loaded = _run_script(script, _subprocess_env())
    assert loaded == {"triton": False, "kernel_module": False}


_INTERPRETER_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
from tests.profiles import make_prior_bank
from vfe3.model.prior_bank import PriorBank
from vfe3.numerics import bounded_variance_from_log
from vfe3.kernels.expected_likelihood import expected_likelihood_ce
from vfe3.kernels import _expected_likelihood_triton as tk
assert tk.INTERPRETED

def case(dtype, B, N, V, K, unigram, ignore, seed):
    torch.manual_seed(seed)
    pb = make_prior_bank(V, K, 3, decode_mode="expected_likelihood_chunked", decode_chunk_size=16,
                   decode_unigram_prior=unigram, unigram_kappa=0.7, decode_tau=0.9).to(dtype)
    with torch.no_grad():
        pb.sigma_log_embed.normal_(0.0, 0.5); pb.mu_embed.normal_(0.0, 1.0); pb.decode_log_scale.fill_(0.2)
    if unigram:
        pb.set_unigram_log_prior(torch.randint(0, 50, (V,)).to(dtype))
    mu_q = torch.randn(B, N, K, dtype=dtype, requires_grad=True)
    sigma_q = (0.2 + torch.rand(B, N, K, dtype=dtype)).requires_grad_()
    targets = torch.randint(0, V, (B, N))
    if ignore == "some":
        targets[0, 3] = -100; targets[-1, :5] = -100
    elif ignore == "all":
        targets[:] = -100
    leaves = [mu_q, sigma_q] + [p for p in pb.parameters() if p.requires_grad]
    ce_t = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets)
    g_t = torch.autograd.grad(ce_t, leaves, allow_unused=True)
    valid = targets != -100
    ce_k = expected_likelihood_ce(
        mu_q[valid], sigma_q[valid], pb._decode_mu_table(),
        bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps),
        pb._unigram_bias() if unigram else None, pb._tau_eff(), targets[valid])
    g_k = torch.autograd.grad(ce_k, leaves, allow_unused=True)
    names = ["mu_q", "sigma_q"] + [n for n, p in pb.named_parameters() if p.requires_grad]
    rel = {}
    for n, a, b in zip(names, g_t, g_k):
        if a is None and b is None:
            continue
        rel[n] = float((a - b).norm() / a.norm().clamp_min(1e-300)) if a.norm() > 0 else float(b.abs().max())
    ce_rel = abs(float(ce_k) - float(ce_t)) / max(abs(float(ce_t)), 1e-300)
    return {"dtype": str(dtype), "shape": [B, N, V, K], "unigram": unigram, "ignore": ignore,
            "rows": int(valid.sum()), "ce_rel": ce_rel, "grad_rel": rel}

results = [
    case(torch.float32, 2, 37, 150, 20, True, "some", 0),   # 68 rows: 3 fwd programs, ragged V tiles
    case(torch.float32, 1, 9, 70, 3, False, "none", 1),     # no unigram, K below the KP=16 pad
    case(torch.float64, 2, 37, 150, 20, True, "some", 2),
    case(torch.float64, 1, 9, 70, 3, False, "none", 3),
    case(torch.float64, 1, 4, 20, 3, True, "all", 4),       # every target ignored: zero rows
]
print(json.dumps(results))
"""


def test_triton_interpreter_matches_the_torch_route(record_property: Any) -> None:
    results = _run_script(_INTERPRETER_SCRIPT, _subprocess_env(TRITON_INTERPRET="1"))
    for r in results:
        record_property(f"interp_{r['dtype']}_{'x'.join(map(str, r['shape']))}_{r['ignore']}",
                        json.dumps({"ce_rel": r["ce_rel"], **r["grad_rel"]}))
        tol = 1e-5 if r["dtype"] == "torch.float32" else 1e-12
        assert set(r["grad_rel"]) >= {"mu_q", "sigma_q", "mu_embed", "sigma_log_embed", "decode_log_scale"}
        assert r["ce_rel"] <= tol, r
        for name, err in r["grad_rel"].items():
            assert err <= tol, (name, r)
    assert results[-1]["rows"] == 0


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
from vfe3.kernels import _expected_likelihood_triton as tk
assert not tk.INTERPRETED
K = 20
cfgs = tk.launch_config(K)
out = {}
for key, kernel in (("fwd", tk.el_fwd_kernel), ("rows", tk.el_bwd_rows_kernel), ("cols", tk.el_bwd_cols_kernel)):
    cfg = cfgs[key]
    consts = {"K": K, "BI": cfg["BI"], "BV": cfg["BV"], "HAS_U": True, "FAST_LOG": True}
    if "KP" in kernel.arg_names:
        consts["KP"] = cfg["KP"]
    sig = {n: ("constexpr" if n in consts else "i32" if n in ("S", "V") else "*i64" if n == "TGT" else "*fp32")
           for n in kernel.arg_names}
    compiled = triton.compile(ASTSource(fn=kernel, signature=sig, constexprs=consts),
                              target=GPUTarget("cuda", 120, 32),
                              options={"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]})
    ptx = compiled.asm["ptx"]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "w") as f:
            f.write(ptx)
        log = subprocess.run([ptxas, "-v", "--gpu-name=sm_120a", src, "-o", src + ".o"],
                             capture_output=True, text=True)
    text = log.stdout + log.stderr
    spill = re.search(r"(\\d+) bytes spill stores, (\\d+) bytes spill loads", text)
    out[key] = {"registers": int(re.search(r"Used (\\d+) registers", text).group(1)),
                "spill_stores": int(spill.group(1)), "spill_loads": int(spill.group(2)),
                "lg2_approx": ptx.count("lg2.approx.f32"), "config": cfg}
print(json.dumps(out))
"""


def test_kernels_compile_for_sm120_without_register_spills(tmp_path: Path, record_property: Any) -> None:
    report = _run_script(_COMPILE_SCRIPT, _subprocess_env(TRITON_CACHE_DIR=str(tmp_path / "triton_cache")))
    for key, r in report.items():
        record_property(f"sm120_{key}", json.dumps(r))
        assert r["spill_stores"] == 0 and r["spill_loads"] == 0, (key, r)
        assert r["registers"] <= 255
        assert r["lg2_approx"] > 0, key                       # the compiled fp32 route uses the fast log


def _time_cuda_ms(fn: Any, repeats: int = 5) -> float:
    fn()                                                       # compile / warm up
    times = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return sorted(times)[len(times) // 2]


@requires_cuda
def test_cuda_triton_route_matches_cpu_torch_and_is_reproducible(record_property: Any) -> None:
    pytest.importorskip("triton")
    device = torch.device("cuda")
    V, K, B, N = 2085, 20, 4, 64
    torch.manual_seed(5)
    mu_q = torch.randn(B, N, K, device=device)
    sigma_q = 0.2 + torch.rand(B, N, K, device=device)
    targets = torch.randint(0, V, (B, N), device=device)
    targets[0, :7] = -100
    triton_bank = _bank(torch.float32, V=V, K=K, unigram=True, chunk=8192).to(device)   # the default chunk

    # CPU tensors run the chunked torch kernel: the float64 reference.
    reference_bank = _bank(torch.float64, V=V, K=K, unigram=True)
    reference_bank.load_state_dict({k: v.double().cpu() for k, v in triton_bank.state_dict().items()})
    ref, g_ref = _ce_and_grads(reference_bank, mu_q.double().cpu(), sigma_q.double().cpu(), targets.cpu())
    assert reference_bank.realized_expected_likelihood_kernel == "torch"

    first, g_first = _ce_and_grads(triton_bank, mu_q, sigma_q, targets, return_stats=True)
    assert triton_bank.realized_expected_likelihood_kernel == "triton"
    second, g_second = _ce_and_grads(triton_bank, mu_q, sigma_q, targets, return_stats=True)
    assert int(first.scored_tokens) == int((targets != -100).sum())
    assert torch.equal(first.ce, second.ce)                    # bitwise run-to-run reproducibility
    for a, b in zip(g_first, g_second):
        assert torch.equal(a, b)
    ce_rel = abs(float(first.ce) - float(ref)) / abs(float(ref))
    grad_rel = [float((a.double().cpu() - b).norm() / b.norm()) for a, b in zip(g_first, g_ref)]
    record_property("cuda_ce_rel_vs_fp64_torch", ce_rel)
    record_property("cuda_grad_rel_vs_fp64_torch", json.dumps(grad_rel))
    assert ce_rel <= 1e-5
    assert len(grad_rel) == 5 and max(grad_rel) <= 1e-4

    triton_ms = _time_cuda_ms(lambda: _ce_and_grads(triton_bank, mu_q, sigma_q, targets))
    record_property("cuda_ms_triton_route", triton_ms)
    print(f"expected-likelihood CE fwd+bwd at B={B} N={N} V={V} K={K}: triton {triton_ms:.2f} ms")
