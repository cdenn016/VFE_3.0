r"""Fused Triton diagonal comparison moments (``relational_diag_moments_kernel``, audit 2026-10-17 item 20).

On the diagonal comparison route :func:`vfe3.relational_scoring.projected_moments` returns
:math:`g = \operatorname{diag}(XX^\top) = \sum_k X^2` with the source root :math:`X = S U^{-1}\Sigma^{1/2}`. The
``"triton"`` kernel (vfe3/kernels/diagonal_moments.py) computes g without storing X. These tests pin:

* the field defaults to ``"torch"``, refuses other values, is marked inactive off the diagonal comparison route,
  sits in the launcher with its options, and VFEModel installs it;
* CPU tensors run the torch route whatever the toggle (bitwise the same moments and gradients) and never import
  triton;
* under the Triton interpreter (CPU, ``TRITON_INTERPRET=1``) the fused moments and the gradients with respect to the
  mean, the variance, the inverse frame and the comparison maps match the torch route in float64 to 1e-12 relative,
  over H in {1, 2, 4}, r in {4, 8, 32}, compact and single dense frames, token counts that are not multiples of the
  tiles, a broadcast frame batch and both role layouts; float32 stays as close to float64 as the torch route; a zero
  variance keeps a zero gradient and a negative one reaches g as NaN on both routes; and a tiny model on the diagonal
  comparison route has the torch route's loss and parameter gradients;
* the kernels compile offline for the RTX 5090 target (sm_120) with zero register-spill bytes at d = 4 to 64;
* on CUDA (``CUDA_TESTS``) the fused route matches a float64 CPU torch route on a live-shaped layer and on the tiny
  live profile moved onto the diagonal comparison route, and is bitwise reproducible; at the reference layer shape
  (K = 160, H = 5, 32 x 128 tokens) it matches as well and the forward plus backward time of both kernels is recorded.

The interpreter and compile checks run in subprocesses, so the test process never imports triton.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

from tests.profiles import live_config, tiny_config
from vfe3 import relational_scoring as rs
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton (or ptxas) unavailable

requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")


@pytest.fixture(autouse=True)
def _restore_kernel() -> Any:
    previous = rs.diagonal_moments_kernel()
    yield
    rs.set_diagonal_moments_kernel(previous)


def _operands(B: int, N: int, H: int, M: int, d: int, r: int, roles: int, *, dtype: torch.dtype = torch.float64,
              inverse_batch: int = 0, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    r"""Head-shared mean and variance (expanded over heads, as ``relational_score`` builds them), well-conditioned
    compact inverse frame blocks (M blocks of width d, K = M d), and comparison maps (H, r, K) or (2, H, r, K).

    Built in float32 and cast, so a float64 run sees exactly the float32 run's values.
    """
    g = torch.Generator().manual_seed(seed)
    K = M * d
    mu = torch.randn(B, N, K, generator=g)
    sigma = 0.2 + torch.rand(B, N, K, generator=g)
    frame_b = inverse_batch or B
    inverse = torch.eye(d) + 0.3 * torch.randn(frame_b, N, M, d, d, generator=g)
    S = torch.randn(*((2,) if roles == 2 else ()), H, r, K, generator=g) / K ** 0.5
    return tuple(x.to(dtype) for x in (mu, sigma, inverse, S))


def _moments(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor, S: torch.Tensor, H: int,
             kernel: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""(a, g) through projected_moments under ``kernel`` and the gradients of a fixed functional of both."""
    leaves = [x.detach().clone().requires_grad_() for x in (mu, sigma, inverse, S)]
    mu_l, sigma_l, inverse_l, S_l = leaves
    K = mu.shape[-1]
    mu_h = mu_l.unsqueeze(-2).expand(*mu.shape[:-1], H, K)
    sigma_h = sigma_l.unsqueeze(-2).expand(*sigma.shape[:-1], H, K)
    previous = rs.set_diagonal_moments_kernel(kernel)
    try:
        a, g = rs.projected_moments(mu_h, sigma_h, inverse_l, S_l, diagonal_only=True)
    finally:
        rs.set_diagonal_moments_kernel(previous)
    gen = torch.Generator().manual_seed(11)
    wa = torch.randn(a.shape, generator=gen, dtype=torch.float64).to(a)
    wg = torch.randn(g.shape, generator=gen, dtype=torch.float64).to(g)
    loss = (a * wa).sum() + (g * wg).sum()
    grads = torch.autograd.grad(loss, leaves)
    return torch.cat([a.detach().flatten(), g.detach().flatten()]), dict(zip(("mu", "sigma", "inverse", "S"), grads))


def _subprocess_env(**extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO),
               KMP_DUPLICATE_LIB_OK="TRUE", **extra)
    return env


def _run_script(script: str, env: Dict[str, str]) -> Any:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton or ptxas unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# The tiny live profile moved onto the diagonal comparison route (the worktree launcher may sit elsewhere).
_DIAGONAL_ROUTE = dict(attention_mode="projected_predictive", predictive_readout="kernel",
                       relational_routing="gaussian_kl", rope_mode="comparison_invariant", rope_fraction=1.0,
                       family="gaussian_diagonal", projected_factorization="gram")


def _diagonal_route_config(**overrides: Any) -> VFE3Config:
    return live_config(**{**_DIAGONAL_ROUTE, **overrides})


# ------------------------------------------------------------------------------------- config and wiring

def test_field_defaults_to_torch_and_refuses_other_values() -> None:
    assert VFE3Config.relational_diag_moments_kernel == "torch"
    assert tiny_config(relational_diag_moments_kernel="triton").relational_diag_moments_kernel == "triton"
    for bad in ("cuda", "Triton", None, True):
        with pytest.raises(ValueError, match="relational_diag_moments_kernel"):
            tiny_config(relational_diag_moments_kernel=bad)


def test_field_is_inactive_off_the_diagonal_comparison_route() -> None:
    assert "relational_diag_moments_kernel" not in _diagonal_route_config().inactive_fields()
    for off in (dict(rope_mode="none"), dict(predictive_readout="minimal_agent", rope_mode="comparison"),
                dict(projected_factorization="gram_certified"), dict(relational_routing="mean_distance")):
        assert "relational_diag_moments_kernel" in _diagonal_route_config(**off).inactive_fields(), off


def test_launcher_carries_a_valid_value_with_its_options() -> None:
    import train_vfe3

    assert train_vfe3.config["relational_diag_moments_kernel"] in ("torch", "triton")
    line = next(line for line in (REPO / "train_vfe3.py").read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("relational_diag_moments_kernel"))
    assert '"torch" | "triton"' in line


def test_model_installs_the_kernel_selection() -> None:
    VFEModel(tiny_config(relational_diag_moments_kernel="triton"))
    assert rs.diagonal_moments_kernel() == "triton"
    VFEModel(tiny_config(relational_diag_moments_kernel="torch"))
    assert rs.diagonal_moments_kernel() == "torch"
    with pytest.raises(ValueError, match="diagonal moments kernel"):
        rs.set_diagonal_moments_kernel("cuda")


# ------------------------------------------------------------------------------------- CPU routing

def test_cpu_tensors_run_the_torch_route_whatever_the_toggle() -> None:
    operands = _operands(2, 5, 2, 2, 4, 4, 2)
    rs.reset_diagonal_moments_triton_calls()
    v_torch, g_torch = _moments(*operands, H=2, kernel="torch")
    v_flag, g_flag = _moments(*operands, H=2, kernel="triton")
    assert rs.diagonal_moments_triton_calls() == 0
    assert torch.equal(v_torch, v_flag)
    assert all(torch.equal(g_torch[name], g_flag[name]) for name in g_torch)


def test_cpu_route_never_imports_triton() -> None:
    script = """
    import json, sys, torch
    from tests.profiles import tiny_config
    from tests.test_diag_moments_triton_20261018 import _diagonal_route_config
    from vfe3.model.model import VFEModel
    from vfe3 import relational_scoring as rs
    model = VFEModel(_diagonal_route_config(relational_diag_moments_kernel="triton"))
    _, loss, _ = model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))
    loss.backward()
    print(json.dumps({"triton": "triton" in sys.modules,
                      "kernel_module": "vfe3.kernels._diagonal_moments_triton" in sys.modules,
                      "calls": rs.diagonal_moments_triton_calls(), "kernel": rs.diagonal_moments_kernel()}))
    """
    loaded = _run_script(script, _subprocess_env())
    assert loaded == {"triton": False, "kernel_module": False, "calls": 0, "kernel": "triton"}


# ------------------------------------------------------------------------------------- Triton interpreter

_INTERPRETER_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
from vfe3 import relational_scoring as rs
from vfe3.kernels import _diagonal_moments_triton as tk
from tests.test_diag_moments_triton_20261018 import _diagonal_route_config, _moments, _operands
from vfe3.model.model import VFEModel
assert tk.INTERPRETED
torch.set_num_threads(2)

# The interpreter runs the fused route on CPU tensors.
rs._diagonal_moments_triton_active = lambda *tensors: rs._DIAG_MOMENTS_KERNEL == "triton"

def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))

def case(B, N, H, M, d, r, roles, inverse_batch=0):
    out = {"shape": [B, N, H, M, d, r, roles, inverse_batch]}
    ops64 = _operands(B, N, H, M, d, r, roles, inverse_batch=inverse_batch)
    rs.reset_diagonal_moments_triton_calls()
    ref_v, ref_g = _moments(*ops64, H=H, kernel="torch")
    assert rs.diagonal_moments_triton_calls() == 0
    v, g = _moments(*ops64, H=H, kernel="triton")
    assert rs.diagonal_moments_triton_calls() == 1
    out["fp64"] = {"moments": rel(v, ref_v), **{n: rel(g[n], ref_g[n]) for n in ref_g}}
    ops32 = tuple(x.float() for x in ops64)
    for name, kernel in (("torch32", "torch"), ("triton32", "triton")):
        v32, g32 = _moments(*ops32, H=H, kernel=kernel)
        out[name] = {"moments": rel(v32, ref_v), **{n: rel(g32[n], ref_g[n]) for n in ref_g}}
    return out

cases = [
    case(2, 5, 1, 1, 8, 4, 1),                       # one head, one dense block
    case(1, 7, 2, 2, 8, 8, 2),                       # two roles
    case(2, 3, 4, 4, 8, 32, 1),                      # r = 32 over K = 32
    case(1, 19, 4, 4, 32, 32, 2),                    # the live block width, 19 tokens (not a tile multiple)
    case(2, 9, 2, 1, 16, 4, 2),                      # a single dense K x K frame block
    case(1, 6, 2, 4, 4, 8, 1),                       # M != H: four frame blocks under two heads
    case(2, 5, 4, 2, 8, 8, 2, inverse_batch=1),      # a frame batch broadcast over the belief batch
]

# A zero variance takes the zero root with zero gradient; a negative one reaches g as NaN on both routes.
mu, sigma, inverse, S = _operands(1, 4, 2, 2, 4, 4, 1)
sigma[0, 1, 3] = 0.0
_, gz_ref = _moments(mu, sigma, inverse, S, H=2, kernel="torch")
vz, gz = _moments(mu, sigma, inverse, S, H=2, kernel="triton")
zero = {"finite": bool(torch.isfinite(vz).all()), "zero_grad": float(gz["sigma"][0, 1, 3]),
        "rel": {n: rel(gz[n], gz_ref[n]) for n in gz_ref}}
sigma[0, 2, 1] = -0.5
neg = {}
for kernel in ("torch", "triton"):
    previous = rs.set_diagonal_moments_kernel(kernel)
    K = mu.shape[-1]
    _, g_neg = rs.projected_moments(mu.unsqueeze(-2).expand(1, 4, 2, K), sigma.unsqueeze(-2).expand(1, 4, 2, K),
                                    inverse, S, diagonal_only=True)
    rs.set_diagonal_moments_kernel(previous)
    neg[kernel] = torch.isnan(g_neg).nonzero().tolist()

# A tiny model on the diagonal comparison route: loss and every parameter gradient, float64.
model_out = {}
tokens, targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]), torch.tensor([[2, 3, 4, 5], [0, 1, 2, 3]])
results = {}
for kernel in ("torch", "triton"):
    torch.manual_seed(0)
    model = VFEModel(_diagonal_route_config(relational_diag_moments_kernel=kernel)).double()
    rs.reset_diagonal_moments_triton_calls()
    _, loss, _ = model(tokens, targets)
    loss.backward()
    results[kernel] = (float(loss), {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                     if p.grad is not None}, rs.diagonal_moments_triton_calls())
(l_t, g_t, c_t), (l_k, g_k, c_k) = results["torch"], results["triton"]
model_out = {"calls": [c_t, c_k], "loss": abs(l_k - l_t) / abs(l_t), "names": sorted(g_t) == sorted(g_k),
             "grads": {n: rel(g_k[n], g_t[n]) for n in g_t if g_t[n].norm() > 0}}
print(json.dumps({"cases": cases, "zero": zero, "neg": neg, "model": model_out}))
"""


def test_triton_interpreter_matches_the_torch_route(record_property: Any) -> None:
    results = _run_script(_INTERPRETER_SCRIPT, _subprocess_env(TRITON_INTERPRET="1", OMP_NUM_THREADS="2",
                                                               MKL_NUM_THREADS="2"))
    for case in results["cases"]:
        record_property(f"interp_{'_'.join(map(str, case['shape']))}", json.dumps(case))
        assert set(case["fp64"]) == {"moments", "mu", "sigma", "inverse", "S"}
        for name, err in case["fp64"].items():
            assert err <= 1e-12, (name, case)
        for name, err in case["triton32"].items():       # as close to float64 as the torch route itself
            assert err <= 4.0 * case["torch32"][name] + 1e-6, (name, case)
    zero = results["zero"]
    assert zero["finite"] and zero["zero_grad"] == 0.0
    assert all(err <= 1e-12 for err in zero["rel"].values()), zero
    assert results["neg"]["torch"] == results["neg"]["triton"] != []
    model = results["model"]
    record_property("interp_model", json.dumps(model))
    assert model["calls"][0] == 0 and model["calls"][1] > 0 and model["names"]
    assert model["loss"] <= 1e-12
    assert model["grads"] and all(err <= 1e-10 for err in model["grads"].values()), model["grads"]


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
from vfe3.kernels import _diagonal_moments_triton as tk
assert not tk.INTERPRETED
out = {}
for d in (4, 8, 16, 32, 64):
    for dt in ("fp32", "fp64"):
        cfg = tk.launch_config(d, dt == "fp64")
        for key, kernel, br in (("fwd", tk.diag_moments_fwd_kernel, cfg["BR_FWD"]),
                                ("tok", tk.diag_moments_bwd_tok_kernel, cfg["BR_TOK"]),
                                ("maps", tk.diag_moments_bwd_maps_kernel, cfg["BR_MAPS"])):
            consts = {"BR": br, "BD": cfg["BD"], "TB": cfg["TB"]}
            if "NEED_U" in kernel.arg_names:
                consts.update(NEED_U=True, NEED_ROOT=True)
            ints = ("T", "ROWS", "K", "M", "d", "GROUPS_PER_SPLIT")
            sig = {n: ("constexpr" if n in consts else "i32" if n in ints else "*" + dt) for n in kernel.arg_names}
            compiled = triton.compile(ASTSource(fn=kernel, signature=sig, constexprs=consts),
                                      target=GPUTarget("cuda", 120, 32),
                                      options={"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]})
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "k.ptx")
                with open(src, "w") as f:
                    f.write(compiled.asm["ptx"])
                log = subprocess.run([ptxas, "-v", "--gpu-name=sm_120a", src, "-o", src + ".o"],
                                     capture_output=True, text=True)
            text = log.stdout + log.stderr
            spill = re.search(r"(\\d+) bytes spill stores, (\\d+) bytes spill loads", text)
            out[f"{key}_{dt}_d{d}"] = {"registers": int(re.search(r"Used (\\d+) registers", text).group(1)),
                                       "spill_stores": int(spill.group(1)), "spill_loads": int(spill.group(2))}
print(json.dumps(out))
"""


def test_kernels_compile_for_sm120_without_register_spills(tmp_path: Path, record_property: Any) -> None:
    report = _run_script(_COMPILE_SCRIPT, _subprocess_env(TRITON_CACHE_DIR=str(tmp_path / "triton_cache")))
    assert len(report) == 5 * 2 * 3
    for key, r in report.items():
        record_property(f"sm120_{key}", json.dumps(r))
        assert r["spill_stores"] == 0 and r["spill_loads"] == 0, (key, r)
        assert r["registers"] <= 255


# ------------------------------------------------------------------------------------- CUDA

def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double().cpu() - b.double().cpu()).norm() / b.double().cpu().norm().clamp_min(1e-300))


@requires_cuda
def test_cuda_fused_moments_match_float64_like_the_torch_route(record_property: Any) -> None:
    r"""A live-shaped layer (H = 4, d = r = 32, both roles, 2 x 128 tokens): the fused float32 moments and gradients
    are as close to the float64 CPU torch route as the float32 CUDA torch route is, and bitwise reproducible."""
    pytest.importorskip("triton")
    H, M, d, r = 4, 4, 32, 32
    ops64 = _operands(2, 128, H, M, d, r, 2)
    ref_v, ref_g = _moments(*ops64, H=H, kernel="torch")
    ops32 = tuple(x.float().cuda() for x in ops64)
    rs.reset_diagonal_moments_triton_calls()
    v_t, g_t = _moments(*ops32, H=H, kernel="torch")
    assert rs.diagonal_moments_triton_calls() == 0
    v_k, g_k = _moments(*ops32, H=H, kernel="triton")
    v_k2, g_k2 = _moments(*ops32, H=H, kernel="triton")
    assert rs.diagonal_moments_triton_calls() == 2
    errs = {"moments": (_rel(v_t, ref_v), _rel(v_k, ref_v)),
            **{n: (_rel(g_t[n], ref_g[n]), _rel(g_k[n], ref_g[n])) for n in ref_g}}
    record_property("cuda_layer_rel_vs_fp64_torch_triton", json.dumps(errs))
    for name, (torch_err, triton_err) in errs.items():
        assert triton_err <= 4.0 * torch_err + 1e-6, (name, errs)
    assert torch.equal(v_k, v_k2) and all(torch.equal(g_k[n], g_k2[n]) for n in g_k)


def _grads(model: VFEModel) -> Dict[str, torch.Tensor]:
    return {name: p.grad.detach().double().cpu() for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


@requires_cuda
def test_cuda_live_route_with_fused_moments_matches_a_float64_cpu_copy(record_property: Any) -> None:
    r"""The tiny live profile on the diagonal comparison route with ``relational_diag_moments_kernel='triton'`` on
    CUDA against a float64 CPU torch copy: loss and every parameter gradient at 1e-4 relative, with the floor
    convention of ``tests/test_cuda_live_route_20261018.py`` (four times the CPU float32 error, or an absolute error
    below 1e-5 of the total gradient norm)."""
    pytest.importorskip("triton")
    rel_tol = 1e-4
    cfg = _diagonal_route_config(relational_diag_moments_kernel="triton")
    device = torch.device("cuda")
    torch.manual_seed(0)
    cuda_model = VFEModel(cfg).to(device)
    bank = cuda_model.prior_bank
    if bank.decode_unigram_prior:
        counts = torch.randint(0, 50, (bank.vocab_size,), generator=torch.Generator().manual_seed(6))
        bank.set_unigram_log_prior(counts.to(device))
    state = {key: value.detach().cpu() for key, value in cuda_model.state_dict().items()}
    cpu_model = VFEModel(cfg).double()
    cpu_model.load_state_dict(state)
    cpu32_model = VFEModel(cfg)
    cpu32_model.load_state_dict(state)

    g = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    tokens = torch.randint(0, cfg.vocab_size, shape, generator=g)
    targets = torch.randint(0, cfg.vocab_size, shape, generator=g)

    rs.reset_diagonal_moments_triton_calls()
    _, loss_cuda, _ = cuda_model(tokens.to(device), targets.to(device))
    loss_cuda.backward()
    assert rs.diagonal_moments_triton_calls() > 0
    calls = rs.diagonal_moments_triton_calls()
    _, loss_cpu, _ = cpu_model(tokens, targets)
    loss_cpu.backward()
    _, loss_cpu32, _ = cpu32_model(tokens, targets)
    loss_cpu32.backward()
    assert rs.diagonal_moments_triton_calls() == calls                     # CPU tensors ran the torch route

    loss_rel = abs(float(loss_cuda) - float(loss_cpu)) / abs(float(loss_cpu))
    g32, g64, g32_cpu = _grads(cuda_model), _grads(cpu_model), _grads(cpu32_model)
    assert set(g32) == set(g64) == set(g32_cpu) and g64

    def rel(grads: Dict[str, torch.Tensor], name: str) -> float:
        return float((grads[name] - g64[name]).norm() / g64[name].norm().clamp_min(1e-300))

    grad_rel = {name: rel(g32, name) for name in g64}
    floor = {name: max(rel_tol, 4.0 * rel(g32_cpu, name)) for name in g64}
    record_property("cuda_fused_live_loss_rel_vs_fp64", loss_rel)
    record_property("cuda_fused_live_grad_rel_vs_fp64", str(grad_rel))
    assert loss_rel <= rel_tol, loss_rel
    total = float(torch.sqrt(sum(g.double().norm() ** 2 for g in g64.values())))
    abs_err = {name: float((g32[name].double() - g64[name].double()).norm()) for name in g64}
    assert {name: (value, floor[name], abs_err[name]) for name, value in grad_rel.items()
            if value > floor[name] and float(g64[name].norm()) > 0.0 and abs_err[name] > 1e-5 * total} == {}


def _timed_fused_step(leaves: Tuple[torch.Tensor, ...], H: int, kernel: str, wa: torch.Tensor, wg: torch.Tensor,
                      repeats: int) -> float:
    r"""Mean CUDA milliseconds of one forward and backward of :func:`projected_moments` under ``kernel``."""
    mu, sigma, inverse, S = leaves
    K = mu.shape[-1]

    def step() -> None:
        mu_h = mu.unsqueeze(-2).expand(*mu.shape[:-1], H, K)
        sigma_h = sigma.unsqueeze(-2).expand(*sigma.shape[:-1], H, K)
        a, g = rs.projected_moments(mu_h, sigma_h, inverse, S, diagonal_only=True)
        torch.autograd.grad((a * wa).sum() + (g * wg).sum(), leaves)

    previous = rs.set_diagonal_moments_kernel(kernel)
    try:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            step()
        stop.record()
        torch.cuda.synchronize()
    finally:
        rs.set_diagonal_moments_kernel(previous)
    return start.elapsed_time(stop) / repeats


@requires_cuda
def test_cuda_fused_moments_parity_and_timing_at_the_reference_shape(record_property: Any) -> None:
    r"""The reference layer shape (K = 160 as M = H = 5 blocks of d = r = 32, both roles, 32 x 128 tokens; audit
    2026-10-20 item 15): the fused float32 moments and the hand-written backward's gradients with respect to the mean,
    the variance, the inverse frame and the comparison maps are as close to the float64 CPU torch route as the
    float32 CUDA torch route is, and the forward plus backward time of each kernel is printed and recorded in the
    JUnit report that ``check_gpu_tests.py`` keeps. No speed is asserted."""
    pytest.importorskip("triton")
    B, N, H, M, d, r = 32, 128, 5, 5, 32, 32
    ops64 = _operands(B, N, H, M, d, r, 2, seed=3)
    ref_v, ref_g = _moments(*ops64, H=H, kernel="torch")
    ops32 = tuple(x.float().cuda() for x in ops64)
    rs.reset_diagonal_moments_triton_calls()
    v_t, g_t = _moments(*ops32, H=H, kernel="torch")
    v_k, g_k = _moments(*ops32, H=H, kernel="triton")
    assert rs.diagonal_moments_triton_calls() == 1
    errs = {"moments": (_rel(v_t, ref_v), _rel(v_k, ref_v)),
            **{n: (_rel(g_t[n], ref_g[n]), _rel(g_k[n], ref_g[n])) for n in ref_g}}
    record_property("cuda_reference_shape_rel_vs_fp64_torch_triton", json.dumps(errs))
    for name, (torch_err, triton_err) in errs.items():
        assert triton_err <= 4.0 * torch_err + 1e-6, (name, errs)

    leaves = tuple(x.detach().clone().requires_grad_() for x in ops32)
    K = M * d
    with torch.no_grad():
        a, g = rs.projected_moments(leaves[0].unsqueeze(-2).expand(B, N, H, K),
                                    leaves[1].unsqueeze(-2).expand(B, N, H, K), leaves[2], leaves[3],
                                    diagonal_only=True)
    gen = torch.Generator(device="cuda").manual_seed(11)
    wa = torch.randn(a.shape, generator=gen, device="cuda")
    wg = torch.randn(g.shape, generator=gen, device="cuda")
    timing = {kernel: _timed_fused_step(leaves, H, kernel, wa, wg, repeats=20) for kernel in ("torch", "triton")}
    timing["speedup"] = timing["torch"] / timing["triton"]
    record_property("cuda_reference_shape_fwd_bwd_ms", json.dumps(timing))
    print(f"\ndiagonal moments forward+backward at B={B} N={N} H={H} K={K} r={r}: torch {timing['torch']:.3f} ms, "
          f"triton {timing['triton']:.3f} ms, speedup {timing['speedup']:.2f}x")
    assert all(value > 0.0 for value in timing.values())
