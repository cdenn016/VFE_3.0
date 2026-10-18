r"""Fused Triton elementwise work of the diagonal_chunked / full_chunked decode CE (``decode_ce_triton``, 2026-10-12).

The fused route (vfe3/kernels/kl_decode.py) keeps the cross-term GEMMs in cuBLAS and runs the per-chunk score
assembly, clamp, temperature, unigram bias, online log-sum-exp, target gather and their backward in Triton. These
tests pin:

* the field defaults off, is marked inactive off the two KL decoders, sits in the launcher at its default, reaches
  the prior bank, and the kernel that ran is recorded in summary.json;
* CPU tensors run the torch loop whatever the toggle (bitwise the same CE and gradients) and never import triton;
* under the Triton interpreter (CPU, ``TRITON_INTERPRET=1``) the fused route reproduces the torch route's CE and
  every gradient on ragged chunks with ignored targets, the unigram bias, small and large temperatures and
  variance scales, tied and untied decode tables, and both the kept and the recomputed cross term;
* the kernels compile offline for the RTX 5090 target (sm_120) with zero register-spill bytes;
* on CUDA (``CUDA_TESTS``) the fused logits are bitwise the torch route's, the CE and every gradient are as close
  to a float64 reference as the torch route's, results are bitwise reproducible, and peak memory is not higher.

The interpreter and compile checks run in subprocesses, so the test process never imports triton.
"""
from tests.profiles import make_prior_bank
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton (or ptxas) unavailable

requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")


def _bank(mode: str, dtype: torch.dtype, *, V: int, K: int, tau: float = 0.3, sigma_init: float = 1.0,
          untie: bool = True, unigram: bool = True, triton: bool = True, chunk: int = 16,
          seed: int = 0) -> PriorBank:
    r"""A tiny KL-decode bank with non-degenerate tables (several vocabulary chunks, a ragged last one).

    Built and perturbed in float32, then cast, so a float64 bank holds exactly the float32 bank's values.
    """
    torch.manual_seed(seed)
    full = mode == "full_chunked"
    pb = make_prior_bank(V, K, 3, decode_mode=mode, family="gaussian_full" if full else "gaussian_diagonal",
                   diagonal_covariance=not full, decode_tau=tau, sigma_init=sigma_init, mu_init_std=0.5,
                   decode_unigram_prior=unigram, unigram_kappa=0.7, untie_decode_bank=untie,
                   decode_chunk_size=chunk, decode_ce_triton=triton)
    with torch.no_grad():
        for name, p in pb.named_parameters():
            if "sigma_log" in name:
                p.add_(0.3 * torch.randn_like(p))
        pb.decode_log_scale.fill_(0.1)
    if unigram:
        pb.set_unigram_log_prior(torch.randint(1, 50, (V,), generator=torch.Generator().manual_seed(seed)).float())
    return pb.to(dtype)


def _inputs(mode: str, dtype: torch.dtype, *, B: int, N: int, V: int, K: int, sigma_init: float = 1.0,
            seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mu_q = torch.randn(B, N, K, generator=g, dtype=torch.float64)
    if mode == "full_chunked":
        a = 0.3 * torch.randn(B, N, K, K, generator=g, dtype=torch.float64)
        sigma_q = sigma_init * (a @ a.transpose(-1, -2) + torch.eye(K, dtype=torch.float64))
    else:
        sigma_q = sigma_init * (0.5 + torch.rand(B, N, K, generator=g, dtype=torch.float64))
    targets = torch.randint(0, V, (B, N), generator=g)
    targets[0, 1] = -100
    targets[-1, -3:] = -100
    targets[0, 2] = V - 1                                 # a target in the ragged last chunk
    # float32-representable values, so a float64 reference sees exactly the float32 run's inputs
    return mu_q.float().to(dtype), sigma_q.float().to(dtype), targets


def _ce_and_grads(pb: PriorBank, mode: str, mu_q: torch.Tensor, sigma_q: torch.Tensor,
                  targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mu_q = mu_q.detach().clone().requires_grad_()
    sigma_q = sigma_q.detach().clone().requires_grad_()
    names = ["mu_q", "sigma_q"] + [n for n, _ in pb.named_parameters()]
    leaves = [mu_q, sigma_q] + list(pb.parameters())
    fn = pb.decode_ce_full_chunked if mode == "full_chunked" else pb.decode_ce_diagonal_chunked
    ce = fn(mu_q, sigma_q, targets)
    grads = torch.autograd.grad(ce, leaves, allow_unused=True)
    return ce, {n: g for n, g in zip(names, grads) if g is not None}


def _subprocess_env(**extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), **extra)
    return env


def _run_script(script: str, env: Dict[str, str]) -> Any:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton or ptxas unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------- config and records

def test_field_is_a_bool_and_is_inactive_off_the_kl_decoders() -> None:
    assert isinstance(VFE3Config.decode_ce_triton, bool)     # the default is pinned by the default snapshot test
    for mode in ("diagonal_chunked", "full_chunked"):
        family = "gaussian_full" if mode == "full_chunked" else "gaussian_diagonal"
        assert "decode_ce_triton" not in tiny_config(decode_mode=mode, family=family).inactive_fields()
    for mode in ("linear", "expected_likelihood_chunked"):
        assert "decode_ce_triton" in tiny_config(decode_mode=mode).inactive_fields()


def test_launcher_carries_a_valid_value_with_its_options() -> None:
    r"""The launcher's value is the user's; only its validity and the valid-values comment are asserted (audit
    2026-10-17, item 38)."""
    import train_vfe3

    assert isinstance(train_vfe3.config["decode_ce_triton"], bool)
    line = next(line for line in (REPO / "train_vfe3.py").read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("decode_ce_triton"))
    assert "True | False" in line


def test_model_passes_the_toggle_to_the_prior_bank() -> None:
    model = VFEModel(tiny_config(decode_mode="diagonal_chunked", decode_ce_triton=True))
    assert model.prior_bank.decode_ce_triton is True
    assert model.prior_bank.realized_decode_ce_kernel is None
    with pytest.raises(ValueError, match="decode_ce_triton must be bool"):
        make_prior_bank(7, 4, 3, decode_ce_triton=1)                       # type: ignore[arg-type]


def test_realized_kernel_is_recorded_in_summary_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3 import run_artifacts as ra
    from vfe3.paper import tables

    monkeypatch.setattr(ra, "_git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    for name in ("_write_provenance", "_write_research_artifacts", "_run_figures_isolated"):
        monkeypatch.setattr(ra, name, lambda *a, **k: None)
    cfg = tiny_config(decode_mode="diagonal_chunked", decode_ce_triton=True, generate_embedding_figures=False)
    model = VFEModel(cfg)
    model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))                   # CPU tensors: the torch loop
    assert model.prior_bank.realized_decode_ce_kernel == "torch"
    art = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    ra.finalize_run(model, art, cfg, test_loader=None)
    summary = json.loads((tmp_path / "r" / "summary.json").read_text(encoding="utf-8"))
    assert summary["decode_ce_kernel"] == "torch"
    run = SimpleNamespace(config_raw={}, summary=summary, provenance={}, config_record={},
                          raw=lambda key, default=None: default)
    items = tables._numerics_items(run)
    assert ("KL decode CE kernel that ran", "torch") in [(label, str(value)) for label, value in items]


# ------------------------------------------------------------------------------------- CPU routing

@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_cpu_tensors_run_the_torch_loop_whatever_the_toggle(mode: str) -> None:
    V, K = 37, 4
    mu_q, sigma_q, targets = _inputs(mode, torch.float64, B=2, N=7, V=V, K=K)
    results = []
    for triton in (False, True):
        pb = _bank(mode, torch.float64, V=V, K=K, triton=triton)
        results.append(_ce_and_grads(pb, mode, mu_q, sigma_q, targets))
        assert pb.realized_decode_ce_kernel == "torch"
    (ce_a, g_a), (ce_b, g_b) = results
    assert torch.equal(ce_a, ce_b) and g_a.keys() == g_b.keys()
    assert all(torch.equal(g_a[n], g_b[n]) for n in g_a)


def test_cpu_route_never_imports_triton() -> None:
    script = """
    import json, sys, torch
    from tests.profiles import make_prior_bank
    pb = make_prior_bank(11, 3, 3, decode_mode="diagonal_chunked", decode_ce_triton=True)
    mu_q = torch.randn(1, 4, 3, requires_grad=True)
    ce = pb.decode_ce_diagonal_chunked(mu_q, 0.5 + torch.rand(1, 4, 3), torch.tensor([[1, 2, -100, 4]]))
    ce.backward()
    print(json.dumps({"triton": "triton" in sys.modules, "kernel_module": "vfe3.kernels.kl_decode" in sys.modules,
                      "kernel": pb.realized_decode_ce_kernel}))
    """
    loaded = _run_script(script, _subprocess_env())
    assert loaded == {"triton": False, "kernel_module": False, "kernel": "torch"}


# ------------------------------------------------------------------------------------- Triton interpreter

_INTERPRETER_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
from tests.test_kl_decode_triton_20261012 import _bank, _ce_and_grads, _inputs
from vfe3.kernels import _kl_decode_triton as tk
from vfe3.model.prior_bank import PriorBank
assert tk.INTERPRETED

def forced(self, lhs, u_all):                       # the interpreter runs the fused route on CPU tensors
    self.realized_decode_ce_kernel = "triton" if self.decode_ce_triton else "torch"
    return self.decode_ce_triton
PriorBank._decode_ce_triton_active = forced

def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))

def case(mode, dtype, tau, sigma_init, untie, unigram, policy, all_ignored=False, seed=0):
    B, N, V, K = 2, 13, 150, 5                       # chunks of 64, 64, 22: masked tiles and a ragged chunk
    mu_q, sigma_q, targets = _inputs(mode, torch.float64, B=B, N=N, V=V, K=K, sigma_init=sigma_init, seed=seed)
    if all_ignored:
        targets[:] = -100
    out = {"mode": mode, "dtype": str(dtype), "tau": tau, "sigma": sigma_init, "untie": untie,
           "unigram": unigram, "policy": policy, "rows": int((targets != -100).sum())}
    ref = _bank(mode, torch.float64, V=V, K=K, tau=tau, sigma_init=sigma_init, untie=untie, unigram=unigram,
                triton=False, chunk=64, seed=seed)
    ce_ref, g_ref = _ce_and_grads(ref, mode, mu_q, sigma_q, targets)
    for name, triton in (("torch", False), ("triton", True)):
        pb = _bank(mode, dtype, V=V, K=K, tau=tau, sigma_init=sigma_init, untie=untie, unigram=unigram,
                   triton=triton, chunk=64, seed=seed)
        pb.decode_ce_checkpoint = policy             # "never": A kept for backward; "always": recomputed
        ce, g = _ce_and_grads(pb, mode, mu_q.to(dtype), sigma_q.to(dtype), targets)
        assert pb.realized_decode_ce_kernel == name
        assert set(g) == set(g_ref), (sorted(g), sorted(g_ref))
        out[name] = {"ce": abs(float(ce) - float(ce_ref)) / max(abs(float(ce_ref)), 1e-300),
                     **{n: rel(g[n], g_ref[n]) for n in g_ref if g_ref[n].norm() > 0}}
    return out

results = [
    case("full_chunked", torch.float32, 0.01, 1.0, True, True, "never"),
    case("full_chunked", torch.float32, 1.0, 15.0, False, True, "always"),
    case("diagonal_chunked", torch.float32, 0.01, 15.0, False, True, "always"),
    case("diagonal_chunked", torch.float32, 1.0, 1.0, True, False, "never"),
    case("full_chunked", torch.float64, 0.01, 15.0, True, True, "always"),
    case("diagonal_chunked", torch.float64, 1.0, 1.0, False, True, "never"),
    case("diagonal_chunked", torch.float64, 0.3, 1.0, True, True, "never", all_ignored=True),
]
print(json.dumps(results))
"""


def test_triton_interpreter_matches_the_torch_route(record_property: Any) -> None:
    results = _run_script(_INTERPRETER_SCRIPT, _subprocess_env(TRITON_INTERPRET="1"))
    for r in results:
        record_property(f"interp_{r['mode']}_{r['dtype']}_tau{r['tau']}_s{r['sigma']}_{r['policy']}",
                        json.dumps({"torch": r["torch"], "triton": r["triton"]}))
        if r["rows"] == 0:
            continue                                  # CE 0 with zero gradients on both routes
        assert {"mu_q", "sigma_q", "decode_log_scale"} <= set(r["triton"])
        for name, err in r["triton"].items():
            if r["dtype"] == "torch.float64":
                assert err <= 1e-11, (name, r)
            else:                                     # as close to float64 as the torch route itself
                assert err <= 4.0 * r["torch"][name] + 5e-7, (name, r)
    assert results[-1]["rows"] == 0


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
from vfe3.kernels import _kl_decode_triton as tk
assert not tk.INTERPRETED
cfgs = tk.launch_config()
out = {}
for key, kernel in (("fwd", tk.kl_ce_fwd_kernel), ("bwd", tk.kl_ce_bwd_kernel)):
    for dt in ("fp32", "fp64"):
        cfg = cfgs[key]
        consts = {"BI": cfg["BI"], "BV": cfg["BV"], "HAS_U": True, "PRECISE_DIV": dt == "fp32", "ACCURATE_EXP": True}
        if "NEED_TAU" in kernel.arg_names:
            consts["NEED_TAU"] = True
        sig = {n: ("constexpr" if n in consts else "i32" if n in ("S", "VC", "V0") else "*i64" if n == "TGT"
                   else "*" + dt) for n in kernel.arg_names}
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
        out[key + "_" + dt] = {"registers": int(re.search(r"Used (\\d+) registers", text).group(1)),
                               "spill_stores": int(spill.group(1)), "spill_loads": int(spill.group(2)),
                               "div_rn": ptx.count("div.rn."), "config": cfg}
print(json.dumps(out))
"""


def test_kernels_compile_for_sm120_without_register_spills(tmp_path: Path, record_property: Any) -> None:
    report = _run_script(_COMPILE_SCRIPT, _subprocess_env(TRITON_CACHE_DIR=str(tmp_path / "triton_cache")))
    assert set(report) == {"fwd_fp32", "fwd_fp64", "bwd_fp32", "bwd_fp64"}
    for key, r in report.items():
        record_property(f"sm120_{key}", json.dumps(r))
        assert r["spill_stores"] == 0 and r["spill_loads"] == 0, (key, r)
        assert r["registers"] <= 255
        assert r["div_rn"] > 0, key                          # IEEE division, as the torch route divides


# ------------------------------------------------------------------------------------- CUDA

def _peak_bytes(fn: Any) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def _time_cuda_ms(fn: Any, repeats: int = 5) -> float:
    fn()
    times: List[float] = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return sorted(times)[len(times) // 2]


@requires_cuda
@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_cuda_fused_route_matches_float64_like_the_torch_route(mode: str, record_property: Any) -> None:
    pytest.importorskip("triton")
    from vfe3.kernels import _kl_decode_triton as tk
    from vfe3.model.prior_bank import _decode_analytic_kl_logits
    from vfe3.runtime import ieee_fp32_matmul

    device = torch.device("cuda")
    B, N, V, K, chunk = 4, 64, 3001, 20, 1000          # 1000 = 512 + 488: every chunk ends in a padded tile
    mu_q, sigma_q, targets = _inputs(mode, torch.float64, B=B, N=N, V=V, K=K)
    reference = _bank(mode, torch.float64, V=V, K=K, tau=0.05, triton=False, chunk=chunk)
    ce_ref, g_ref = _ce_and_grads(reference, mode, mu_q, sigma_q, targets)
    runs = {}
    for name, triton in (("torch", False), ("triton", True)):
        pb = _bank(mode, torch.float32, V=V, K=K, tau=0.05, triton=triton, chunk=chunk).to(device)
        args = (mu_q.float().to(device), sigma_q.float().to(device), targets.to(device))
        ce, g = _ce_and_grads(pb, mode, *args)
        assert pb.realized_decode_ce_kernel == name
        runs[name] = (pb, args, ce, g)
    errors = {}
    for name, (pb, args, ce, g) in runs.items():
        errors[name] = {"ce": abs(float(ce) - float(ce_ref)) / abs(float(ce_ref)),
                        **{n: float((g[n].double().cpu() - g_ref[n]).norm() / g_ref[n].norm()) for n in g_ref}}
    record_property(f"cuda_rel_err_vs_fp64_{mode}", json.dumps(errors))
    for n, err in errors["triton"].items():
        assert err <= 4.0 * errors["torch"][n] + 5e-7, (n, errors)

    pb, args, ce, g = runs["triton"]
    ce_again, g_again = _ce_and_grads(pb, mode, *args)                    # bitwise run-to-run reproducibility
    assert torch.equal(ce, ce_again) and all(torch.equal(g[n], g_again[n]) for n in g)

    # The fused logits are the torch route's bitwise: the forward kernel's target logit and row maximum over one
    # chunk spanning the vocabulary equal the torch logits' gather and amax.
    S = 300
    gen = torch.Generator(device="cpu").manual_seed(3)
    lhs = torch.randn(S, 2 * K, generator=gen).to(device)
    rhs = torch.rand(V, 2 * K, generator=gen).to(device)
    bias = (5.0 * torch.randn(V, generator=gen)).to(device)
    per_pos = (10.0 * torch.randn(S, generator=gen)).to(device)
    u = torch.randn(V, generator=gen).to(device)
    tau = torch.tensor([0.05], device=device)
    tgt = torch.randint(0, V, (S,), generator=gen).to(device)
    with ieee_fp32_matmul():
        A = lhs @ rhs.t()
    logits = _decode_analytic_kl_logits(A + bias, per_pos.unsqueeze(-1), tau, torch.float32) + u
    m = torch.full((S,), float("-inf"), device=device)
    l, t = torch.zeros(S, device=device), torch.zeros(S, device=device)
    cfg = tk.launch_config()["fwd"]
    tk.kl_ce_fwd_kernel[(-(-S // cfg["BI"]),)](A, bias, per_pos, u, tgt, tau, m, l, t, S, V, 0, BI=cfg["BI"],
                                               BV=cfg["BV"], HAS_U=True, PRECISE_DIV=True, ACCURATE_EXP=True,
                                               num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    assert torch.equal(t, logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
    assert torch.equal(m, logits.amax(-1))

    # Memory and time at the default chunk on a wider batch: the fused route never peaks above the torch loop.
    big = {}
    for name, triton in (("torch", False), ("triton", True)):
        pb = _bank(mode, torch.float32, V=V, K=K, tau=0.05, triton=triton, chunk=8192).to(device)
        mq = torch.randn(16, 128, K, device=device)
        sq = (sigma_q[:1, :1].float().to(device).expand(16, 128, *sigma_q.shape[2:]).contiguous())
        tg = torch.randint(0, V, (16, 128), device=device)

        def step(pb: PriorBank = pb) -> None:
            _ce_and_grads(pb, mode, mq, sq, tg)

        big[name] = (_peak_bytes(step), _time_cuda_ms(step))
    record_property(f"cuda_peak_bytes_{mode}", json.dumps({k: v[0] for k, v in big.items()}))
    record_property(f"cuda_ms_{mode}", json.dumps({k: v[1] for k, v in big.items()}))
    assert big["triton"][0] <= big["torch"][0]
