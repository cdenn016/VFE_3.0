r"""Remediation of the 2026-10-31 audit, package kernels-memory (docs/audit-results_2026-10-31.md).

A085 (memory-local-learner-1). At regression rank 2 to 4 the leave-one-out contraction
:math:`U_k^\top G^{(-k)}U_k` cancels where :math:`U_k` has entries of both signs and is nearly orthogonal to every
:math:`V_l`: its rounding error is of order :math:`\epsilon\lvert U_k\rvert^2\operatorname{tr}G^{(-k)}` whatever the
couplings, so the float32 variance of the rotated fixture ranged from the result noise alone to five times the true
value, and the clamp dropped :math:`\partial/\partial U_k`. Both routes now recompute the elements whose computed sum
lies below :math:`2^{12}\epsilon\lvert U_k\rvert^2\operatorname{tr}G^{(-k)}` directly as
:math:`\sum_{l\ne k}p_l(U_k\cdot V_l)^2` in float64.

A087 (memory-local-learner-3). The column backward reads the pairs kernel's packed clamp indicator instead of
recomputing it under a different scan layout.

A089 (memory-local-learner-7). No Python float reaches a kernel launch (compiled launches type it fp32); the
:math:`\tfrac K2\log 2\pi` of the log evidence is subtracted in torch in the route's dtype.

A200 (recent-changes-4). ``cr_fwd_weights_scan_kernel`` forms the exclusive Gram tile sums as an exclusive prefix plus
an exclusive suffix (O(NKT) per program), opt-in through ``launch_config()["fwd_weights"]["scan_tile_sums"]``.

A206 (triton-kernels-5). The expected-likelihood backward places its column sums at two levels (a 16-feature block,
merged once per block); the result is bitwise the previous kernels'.

A283, A315, A325, A341, A343: ``structured_head_scores`` band mass, the gathered-pair record's unique-row versions,
one exception class for a non-bool frame flag, BK at most 32 for the packed bits, and the plane budget's BK.

The Triton checks run under the interpreter (``TRITON_INTERPRET=1``) or as offline sm_120 compiles in subprocesses,
so the test process never imports triton. Tiny CPU shapes only; nothing runs on a GPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

import vfe3.kernels.cue_result as fused_kernels
import vfe3.model.cue_result_memory as crm
from tests.test_cue_result_triton_20261025 import _run_script, _subprocess_env
from vfe3.contracts import CanonicalFrameContext, _gathered_source, mark_gathered_frame_pair, trusted_identity_context
from vfe3.metrics import structured_head_scores
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior

_NAMES = ("xi", "c", "nu", "omega", "log_lambda", "log_result_noise", "log_pi")


@pytest.fixture(autouse=True)
def _one_thread() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------------------------------------ fixtures

def _rotated(theta: float, delta: float, scale: float) -> Dict[str, torch.Tensor]:
    r"""K = 2, rank 2, one component (memory-local-learner-1): :math:`F = I + UV^\top = [[0, -\delta], [0, 1]]` for
    every rotation :math:`\theta` of the factorization (:math:`UR`, :math:`VR`), zero means, :math:`c = \lambda` =
    ``scale`` and result noise :math:`e^{-25}`. Float32 values; the references read the same values in float64."""
    rot = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=torch.float64)
    U = (torch.tensor([[[1.0, -1.0], [0.0, 0.0]]], dtype=torch.float64) @ rot).float()
    V = (torch.tensor([[[-0.5, 0.5], [1.0, 1.0 + delta]]], dtype=torch.float64) @ rot).float()
    return {"xi": torch.zeros(1, 2), "c": torch.full((1, 2), scale), "nu": torch.zeros(1, 2),
            "omega": torch.zeros(1, 2), "log_lambda": torch.full((1, 2), math.log(scale)),
            "log_result_noise": torch.full((1, 2), -25.0), "log_pi": torch.zeros(1),
            "regression_left": U, "regression_right": V}


def _corner(dtype: torch.dtype, K: int = 40, k_star: int = 33, rows: int = 3, seed: int = 0) -> Dict[str, torch.Tensor]:
    r"""One component, rank 2, :math:`\lambda = 1` and record variances in [0.5, 1.5), zero means. Coordinate
    :math:`k^*` cancels its own term (:math:`V_{k^*} = -U_{k^*}/\lvert U_{k^*}\rvert^2`, :math:`a_{k^*} = -1`) and every
    other :math:`V_l = s_ld + \varepsilon_le` is nearly parallel to a direction :math:`d` orthogonal to
    :math:`U_{k^*} = e`, so the off-axis sum of :math:`k^*` is about :math:`\sum_lp_l\varepsilon_l^2` while
    :math:`G^{(-k^*)}` is dominated by :math:`(\sum_lp_ls_l^2)dd^\top`. :math:`\varepsilon_l` is of order 1e-4 in float32
    and :math:`2^{-30}` in float64, and the result noise lies far below the off-axis sum. With K = 40 at BK = 32,
    :math:`k^* = 33` sits in the second coordinate tile. The other coordinates are generic."""
    g = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    phi = 0.6
    d = torch.tensor([math.cos(phi), math.sin(phi)], dtype=f64)
    e = torch.tensor([-math.sin(phi), math.cos(phi)], dtype=f64)                 # entries of both signs
    small = 1e-4 if dtype == torch.float32 else 2.0 ** -30
    s = 0.5 + torch.rand(K, generator=g, dtype=f64)
    eps_l = small * torch.randn(K, generator=g, dtype=f64)
    V = s[:, None] * d + eps_l[:, None] * e                                      # (K, 2)
    U = 0.6 * torch.randn(K, 2, generator=g, dtype=f64)
    U[k_star] = e
    V[k_star] = -e
    log_q = torch.zeros(1, K, dtype=f64)
    log_q[0, k_star] = -25.0 if dtype == torch.float32 else -80.0
    ops = {"xi": torch.zeros(rows, K, dtype=f64), "c": 0.5 + torch.rand(rows, K, generator=g, dtype=f64),
           "nu": torch.zeros(1, K, dtype=f64), "omega": torch.zeros(1, K, dtype=f64),
           "log_lambda": torch.zeros(1, K, dtype=f64), "log_result_noise": log_q,
           "log_pi": torch.zeros(1, dtype=f64), "regression_left": U[None], "regression_right": V[None]}
    return {k: v.to(dtype) for k, v in ops.items()}


def _lowrank(ops: Dict[str, torch.Tensor], index: Tuple[int, int], fused: bool = False) -> Dict[str, Any]:
    r"""The low-rank route in the operands' dtype: the variances and :math:`\partial v_{ik}/\partial U` at ``index``."""
    leaves = {k: v.clone() for k, v in ops.items()}
    U = leaves["regression_left"].requires_grad_(True)
    out = cue_result_diagonal_posterior(*(leaves[k] for k in _NAMES), regression_left=U,
                                        regression_right=leaves["regression_right"], fused=fused)
    (dU,) = torch.autograd.grad(out.covariance[index], [U])
    return {"var": out.covariance.detach(), "dU": dU}


def _dense_reference(ops: Dict[str, torch.Tensor], index: Tuple[int, int]) -> Dict[str, Any]:
    r"""The float64 dense route on :math:`F = I + UV^\top` built from the same values: the variance is the sum of
    nonnegative terms :math:`\sum_lF_{kl}^2p_l + q_k`, and :math:`\partial v_{ik}/\partial U` flows through
    :math:`F`."""
    d = {k: v.detach().double() for k, v in ops.items()}
    U = d["regression_left"].requires_grad_(True)
    out = cue_result_diagonal_posterior(*(d[k] for k in _NAMES),
                                        regression_offset=U @ d["regression_right"].transpose(-1, -2))
    (dU,) = torch.autograd.grad(out.covariance[index], [U])
    return {"var": out.covariance.detach(), "dU": dU}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300))


_ROTATED_CASES = [(theta, scale) for theta in (0.3, 1.1) for scale in (1.0, 4096.0)]


# ------------------------------------------------------------------------------------------------ A085, torch route

@pytest.mark.parametrize("theta, scale", _ROTATED_CASES)
def test_a085_torch_rank2_variance_and_factor_gradient_survive_a_rotated_factorization(theta: float,
                                                                                        scale: float) -> None:
    ops = _rotated(theta, 1e-4, scale)
    got, ref = _lowrank(ops, (0, 0)), _dense_reference(ops, (0, 0))
    assert float(got["var"][0, 0]) == pytest.approx(float(ref["var"][0, 0]), rel=1e-5)
    assert _rel(got["dU"][0, 0], ref["dU"][0, 0]) < 1e-3


@pytest.mark.parametrize("dtype, var_tol, grad_tol", [(torch.float32, 1e-5, 1e-3), (torch.float64, 1e-6, 1e-6)])
def test_a085_torch_multi_tile_corner_matches_the_dense_reference(dtype: torch.dtype, var_tol: float,
                                                                  grad_tol: float) -> None:
    ops = _corner(dtype)
    got, ref = _lowrank(ops, (1, 33)), _dense_reference(ops, (1, 33))
    assert ((got["var"].double() - ref["var"]).abs() / ref["var"]).max().item() < var_tol
    assert _rel(got["dU"][0, 33], ref["dU"][0, 33]) < grad_tol


def test_a085_generic_operands_flag_nothing_and_keep_the_contraction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = crm._rescue_off_axis

    def spy(off: torch.Tensor, *args: Any) -> torch.Tensor:
        out = original(off, *args)
        calls.append(out is off)
        return out

    monkeypatch.setattr(crm, "_rescue_off_axis", spy)
    g = torch.Generator().manual_seed(2)
    R, M, K, rank = 3, 4, 9, 3
    ops = {"xi": torch.randn(R, K, generator=g), "c": 0.3 + torch.rand(R, K, generator=g),
           "nu": torch.randn(M, K, generator=g), "omega": torch.randn(M, K, generator=g),
           "log_lambda": 0.5 * torch.randn(M, K, generator=g), "log_result_noise": torch.randn(M, K, generator=g) - 1,
           "log_pi": torch.randn(M, generator=g), "regression_left": 0.7 * torch.randn(M, K, rank, generator=g),
           "regression_right": 0.7 * torch.randn(M, K, rank, generator=g)}
    _lowrank(ops, (0, 0))
    assert calls and all(calls)                                     # every tile returned its contraction unchanged


# ------------------------------------------------------------------------------------ fused route (interpreter)

_FUSED_A085 = """
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
from tests.test_audit_20261031_kernels_memory import _ROTATED_CASES, _corner, _lowrank, _rotated
assert tk.INTERPRETED
fused_kernels.fused_route_active = lambda xi: True
out = {}
for theta, scale in _ROTATED_CASES:
    got = _lowrank(_rotated(theta, 1e-4, scale), (0, 0), fused=True)
    out[f"rotated_{theta}_{scale}"] = {"var": got["var"].double().tolist(), "dU": got["dU"].double().tolist()}
for name, dtype in (("corner32", torch.float32), ("corner64", torch.float64)):
    got = _lowrank(_corner(dtype), (1, 33), fused=True)
    out[name] = {"var": got["var"].double().tolist(), "dU": got["dU"].double().tolist()}
print(json.dumps(out))
"""


def test_a085_fused_route_keeps_the_off_axis_variance_and_the_factor_gradient() -> None:
    report = _run_script(_FUSED_A085, _subprocess_env(TRITON_INTERPRET="1"))
    for theta, scale in _ROTATED_CASES:
        got = report[f"rotated_{theta}_{scale}"]
        ops = _rotated(theta, 1e-4, scale)
        ref = _dense_reference(ops, (0, 0))
        assert got["var"][0][0] == pytest.approx(float(ref["var"][0, 0]), rel=1e-5), (theta, scale)
        # the fused backward keeps the Gram form 2 G^(-k) U_k, accurate to about eps |G^(-k)| |U_k| / |G^(-k) U_k|
        # (measured 2.9e-4 here and 5.0e-3 on the float32 corner; the clamp had dropped it entirely, error 1.0)
        assert _rel(torch.tensor(got["dU"])[0, 0], ref["dU"][0, 0]) < 3e-3, (theta, scale)
    for name, dtype, var_tol, grad_tol in (("corner32", torch.float32, 1e-5, 2e-2),
                                           ("corner64", torch.float64, 1e-6, 1e-5)):
        ref = _dense_reference(_corner(dtype), (1, 33))
        var = torch.tensor(report[name]["var"], dtype=torch.float64)
        assert ((var - ref["var"]).abs() / ref["var"]).max().item() < var_tol, name
        assert _rel(torch.tensor(report[name]["dU"])[0, 33], ref["dU"][0, 33]) < grad_tol, name


_KERNEL_LAUNCHES = """
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
fused_kernels.fused_route_active = lambda xi: True
NAMES = ("cr_fwd_weights_kernel", "cr_fwd_weights_scan_kernel", "cr_fwd_moments_kernel", "cr_bwd_pairs_kernel",
         "cr_bwd_rows_kernel", "cr_bwd_cols_kernel")
ORIGINAL = {name: getattr(tk, name) for name in NAMES if hasattr(tk, name)}
launches = []


class Launch:
    r'''Records every launch of one kernel (positional argument types) and runs an optional hook after it.'''

    def __init__(self, name, hook=None):
        self.name, self.kernel, self.hook = name, ORIGINAL[name], hook
        self.arg_names = self.kernel.arg_names

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            launches.append((self.name, [type(a).__name__ for a in args]))
            result = launch(*args, **kwargs)
            if self.hook is not None:
                self.hook(dict(zip(self.arg_names, args)))
            return result
        return run


def install(hooks):
    for name in ORIGINAL:
        setattr(tk, name, Launch(name, hooks.get(name)))


g = torch.Generator().manual_seed(11)
R, M, K, rank = 3, 5, 40, 2
d = torch.float64
ops = {"xi": torch.randn(R, K, generator=g, dtype=d), "c": 0.3 + torch.rand(R, K, generator=g, dtype=d),
       "nu": torch.randn(M, K, generator=g, dtype=d), "omega": torch.randn(M, K, generator=g, dtype=d),
       "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=d),
       "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=d) - 1.0,
       "log_pi": torch.randn(R, M, generator=g, dtype=d),
       "regression_left": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d),
       "regression_right": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d)}
cot = [torch.randn(R, K, generator=g, dtype=d), torch.randn(R, K, generator=g, dtype=d),
       torch.randn(R, M, generator=g, dtype=d), torch.randn(R, generator=g, dtype=d)]
names = ("xi", "c", "nu", "omega", "log_lambda", "log_result_noise", "log_pi")


def run(fused):
    leaves = {k: v.clone().requires_grad_(True) for k, v in ops.items()}
    out = cue_result_diagonal_posterior(*(leaves[k] for k in names), regression_left=leaves["regression_left"],
                                        regression_right=leaves["regression_right"], fused=fused)
    outs = (out.mean, out.covariance, out.log_weights, out.log_evidence)
    loss = sum((o * c).sum() for o, c in zip(outs, cot))
    grads = torch.autograd.grad(loss, list(leaves.values()))
    return {**dict(zip(("mean", "var", "log_weights", "log_evidence"), outs)), **dict(zip(leaves, grads))}


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


reference = run(False)
install({})
clean = run(True)
float_args = sorted({name for name, types in launches if "float" in types})
launched = sorted({name for name, _ in launches})


def zero_bits(args):
    args["IND"].zero_()


install({"cr_bwd_pairs_kernel": zero_bits})
poisoned = run(True)
print(json.dumps({"vs_torch": {k: rel(clean[k], reference[k]) for k in reference},
                  "float_args": float_args, "launched": launched,
                  "cols_takes_bits": "IND" in ORIGINAL["cr_bwd_cols_kernel"].arg_names,
                  "weights_takes_constant": "HALF_K_LOG2PI" in ORIGINAL["cr_fwd_weights_kernel"].arg_names,
                  "poisoned_changes": {k: not torch.equal(poisoned[k], clean[k]) for k in clean}}))
"""


def test_a087_a089_kernel_launches_share_the_indicator_and_take_no_python_float() -> None:
    report = _run_script(_KERNEL_LAUNCHES, _subprocess_env(TRITON_INTERPRET="1"))
    for name, err in report["vs_torch"].items():                     # float64, the torch route's posterior
        assert err <= 1e-11, (name, report)
    assert set(report["launched"]) == {"cr_fwd_weights_kernel", "cr_fwd_moments_kernel", "cr_bwd_pairs_kernel",
                                       "cr_bwd_rows_kernel", "cr_bwd_cols_kernel"}, report["launched"]
    # A089: no Python float reaches a launch (a compiled launch types it fp32), and the log evidence still agrees
    assert report["float_args"] == [] and not report["weights_takes_constant"], report
    # A087: the column backward reads the pairs kernel's packed indicator. With the bits zeroed after the pairs
    # kernel, the rows kernel's dL/dc and the column kernel's dL/dU, dL/dV and dL/dlambda all change, while the
    # forward outputs do not.
    assert report["cols_takes_bits"], report
    changes = report["poisoned_changes"]
    assert all(changes[k] for k in ("c", "regression_left", "regression_right", "log_lambda")), changes
    assert not (changes["mean"] or changes["var"] or changes["log_weights"] or changes["log_evidence"]), changes


_SCAN_TILE_SUMS = """
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
fused_kernels.fused_route_active = lambda xi: True
d = torch.float64


def planes(kernel, x, c, nu, lam, V, lp, rank):
    R, K = x.shape
    M = nu.shape[0]
    RP, BK = fused_kernels._rank_pad(rank), fused_kernels._tile_k()
    NKT = -(-K // BK)
    f = tk.launch_config()["fwd_weights"]
    T = torch.full((RP, R, M), float("nan"), dtype=d)
    GP = torch.full((RP * RP * NKT, R, M), float("nan"), dtype=d)
    logw = torch.empty(R, M, dtype=d)
    kernel[(R, -(-M // f["BA"]))](x, c, nu, lam, V, lp, logw, T, GP, R, M, K, NKT, 0, BA=f["BA"], BK=BK, RANK=rank,
                                  RP=RP, ACC=False, WRITE_LOGW=True, NKTP=fused_kernels._rank_pad(NKT))
    return T[:rank], GP.reshape(RP, RP, NKT, R, M)[:rank, :rank], logw


def exclusive_reference(c, lam, V):
    r'''O^(j)_{ia,rs} = sum_{j' != j} sum_{k in tile j'} V_akr p_iak V_aks, in float64.'''
    R, K = c.shape
    BK = fused_kernels._tile_k()
    p = lam[None] * c[:, None] / (lam[None] + c[:, None])                       # (R, M, K)
    e = torch.einsum("akr,iak,aks->rsiak", V, p, V)                            # (rank, rank, R, M, K)
    tiles = torch.stack([e[..., j:j + BK].sum(-1) for j in range(0, K, BK)], 2)   # (rank, rank, NKT, R, M)
    return tiles.sum(2, keepdim=True) - tiles


out = {"planes": []}
g = torch.Generator().manual_seed(3)
for K, rank in ((96, 2), (288, 3), (1024, 2)):                                 # NKT 3, 9 (NKTP 16) and 32
    R, M = 2, 5
    x, c = torch.randn(R, K, generator=g, dtype=d), 0.3 + torch.rand(R, K, generator=g, dtype=d)
    nu, lam = torch.randn(M, K, generator=g, dtype=d), (0.5 * torch.randn(M, K, generator=g, dtype=d)).exp()
    V = 0.7 * torch.randn(M, K, rank, generator=g, dtype=d)
    lp = torch.log_softmax(torch.randn(M, generator=g, dtype=d), -1)
    seq = planes(tk.cr_fwd_weights_kernel, x, c, nu, lam, V, lp, rank)
    scan = planes(tk.cr_fwd_weights_scan_kernel, x, c, nu, lam, V, lp, rank)
    ref = exclusive_reference(c, lam, V)
    scale = float(ref.abs().max())
    out["planes"].append({"K": K, "rank": rank, "seq_vs_ref": float((seq[1] - ref).abs().max()) / scale,
                          "scan_vs_ref": float((scan[1] - ref).abs().max()) / scale,
                          "t_and_logw_equal": bool(torch.equal(seq[0], scan[0]) and torch.equal(seq[2], scan[2]))})

shipped = tk.launch_config
tk.launch_config = lambda: {k: dict(v, scan_tile_sums=True) if k == "fwd_weights" else v for k, v in shipped().items()}
route = {}
for K in (70, 300):
    R, M, rank = 3, 5, 2
    ops = {"xi": torch.randn(R, K, generator=g, dtype=d), "c": 0.3 + torch.rand(R, K, generator=g, dtype=d),
           "nu": torch.randn(M, K, generator=g, dtype=d), "omega": torch.randn(M, K, generator=g, dtype=d),
           "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=d),
           "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=d) - 1.0,
           "log_pi": torch.randn(M, generator=g, dtype=d),
           "regression_left": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d),
           "regression_right": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d)}
    results = []
    for fused in (False, True):
        leaves = {k: v.clone().requires_grad_(True) for k, v in ops.items()}
        o = cue_result_diagonal_posterior(*(leaves[k] for k in ("xi", "c", "nu", "omega", "log_lambda",
                                                                 "log_result_noise", "log_pi")),
                                          regression_left=leaves["regression_left"],
                                          regression_right=leaves["regression_right"], fused=fused)
        loss = o.mean.square().sum() + o.covariance.sum() + o.log_evidence.sum()
        results.append([o.mean, o.covariance, o.log_evidence, *torch.autograd.grad(loss, list(leaves.values()))])
    route[K] = max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-300)) for a, b in zip(results[1], results[0]))
out["route"] = route
print(json.dumps(out))
"""


def test_a200_scan_tile_sums_match_the_sequential_sums_and_the_torch_route() -> None:
    report = _run_script(_SCAN_TILE_SUMS, _subprocess_env(TRITON_INTERPRET="1"))
    for entry in report["planes"]:
        assert entry["seq_vs_ref"] < 1e-13 and entry["scan_vs_ref"] < 1e-13, entry
        assert entry["t_and_logw_equal"], entry
    for K, err in report["route"].items():
        assert err <= 1e-11, (K, report["route"])


# ------------------------------------------------------------------------------------------------ A206

# The expected-likelihood backward kernels as they were before the two-level placement (triton 3.5.1 source of
# vfe3/kernels/_expected_likelihood_triton.py at 5bb0a57), the bitwise reference.
_EL_REFERENCE_SOURCE = '''
import triton
import triton.language as tl

from vfe3.kernels._expected_likelihood_triton import _energy_tile, _pair_weights


@triton.jit
def el_bwd_rows_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, M, LOGL, SCALE, GMQ, GSQ, GE, S, V,
                       K: tl.constexpr, KP: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
                       HAS_U: tl.constexpr, FAST_LOG: tl.constexpr):
    pid = tl.program_id(0)
    offs_i = pid * BI + tl.arange(0, BI)
    mi = offs_i < S
    c = 0.5 / tl.load(TAU)
    scale = tl.load(SCALE)
    t = tl.load(TGT + offs_i, mask=mi, other=-1)
    m = tl.load(M + offs_i, mask=mi, other=0.0)
    log_l = tl.load(LOGL + offs_i, mask=mi, other=0.0)
    kk = tl.arange(0, KP)
    dt = MQ.dtype.element_ty
    acc_m = tl.zeros([BI, KP], dt)
    acc_s = tl.zeros([BI, KP], dt)
    acc_e = tl.zeros([BI], dt)
    for v0 in range(0, V, BV):
        offs_v = v0 + tl.arange(0, BV)
        mv_ok = offs_v < V
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        w = _pair_weights(E, c, U, offs_v, mv_ok, mi, t, m, log_l, scale, HAS_U)
        acc_e += tl.sum(w * E, 1)
        for k in range(K):
            a = tl.load(MQ + offs_i * K + k, mask=mi, other=0.0)
            b = tl.load(SQ + offs_i * K + k, mask=mi, other=1.0)
            mv = tl.load(MVT + k * V + offs_v, mask=mv_ok, other=0.0)
            sv = tl.load(SVT + k * V + offs_v, mask=mv_ok, other=1.0)
            inv = 1.0 / (b[:, None] + sv[None, :])
            r = (a[:, None] - mv[None, :]) * inv
            acc_m += tl.where(kk[None, :] == k, tl.sum(w * r, 1)[:, None], 0.0)
            acc_s += tl.where(kk[None, :] == k, tl.sum(w * (r * r - inv), 1)[:, None], 0.0)
    km = mi[:, None] & (kk[None, :] < K)
    tl.store(GMQ + offs_i[:, None] * K + kk[None, :], -2.0 * c * acc_m, mask=km)
    tl.store(GSQ + offs_i[:, None] * K + kk[None, :], c * acc_s, mask=km)
    tl.store(GE + offs_i, acc_e, mask=mi)


@triton.jit
def el_bwd_cols_kernel(MQ, SQ, MVT, SVT, U, TGT, TAU, M, LOGL, SCALE, GMV, GSV, GU, S, V,
                       K: tl.constexpr, KP: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
                       HAS_U: tl.constexpr, FAST_LOG: tl.constexpr):
    pid = tl.program_id(0)
    offs_v = pid * BV + tl.arange(0, BV)
    mv_ok = offs_v < V
    c = 0.5 / tl.load(TAU)
    scale = tl.load(SCALE)
    kk = tl.arange(0, KP)
    dt = MQ.dtype.element_ty
    acc_m = tl.zeros([KP, BV], dt)
    acc_s = tl.zeros([KP, BV], dt)
    acc_u = tl.zeros([BV], dt)
    for i0 in range(0, S, BI):
        offs_i = i0 + tl.arange(0, BI)
        mi = offs_i < S
        E = _energy_tile(MQ, SQ, MVT, SVT, offs_i, mi, offs_v, mv_ok, V, K, BI, BV, FAST_LOG)
        t = tl.load(TGT + offs_i, mask=mi, other=-1)
        m = tl.load(M + offs_i, mask=mi, other=0.0)
        log_l = tl.load(LOGL + offs_i, mask=mi, other=0.0)
        w = _pair_weights(E, c, U, offs_v, mv_ok, mi, t, m, log_l, scale, HAS_U)
        acc_u += tl.sum(w, 0)
        for k in range(K):
            a = tl.load(MQ + offs_i * K + k, mask=mi, other=0.0)
            b = tl.load(SQ + offs_i * K + k, mask=mi, other=1.0)
            mv = tl.load(MVT + k * V + offs_v, mask=mv_ok, other=0.0)
            sv = tl.load(SVT + k * V + offs_v, mask=mv_ok, other=1.0)
            inv = 1.0 / (b[:, None] + sv[None, :])
            r = (a[:, None] - mv[None, :]) * inv
            acc_m += tl.where(kk[:, None] == k, tl.sum(w * r, 0)[None, :], 0.0)
            acc_s += tl.where(kk[:, None] == k, tl.sum(w * (r * r - inv), 0)[None, :], 0.0)
    km = (kk[:, None] < K) & mv_ok[None, :]
    tl.store(GMV + offs_v[None, :] * K + kk[:, None], 2.0 * c * acc_m, mask=km)
    tl.store(GSV + offs_v[None, :] * K + kk[:, None], c * acc_s, mask=km)
    tl.store(GU + offs_v, acc_u, mask=mv_ok)
'''

_EL_BITWISE = """
import json, os, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
sys.path.insert(0, os.environ["VFE3_EL_REFERENCE_DIR"])
import el_reference_kernels as old
from vfe3.kernels import _expected_likelihood_triton as tk
from vfe3.kernels.expected_likelihood import _grid_launch
assert tk.INTERPRETED


def backward(rows_kernel, cols_kernel, mu_q, sigma_q, mu_v, sigma_v, u, tau, targets):
    S, K = mu_q.shape
    V = mu_v.shape[0]
    dtype = mu_q.dtype
    mu_v_t, sigma_v_t = mu_v.t().contiguous(), sigma_v.t().contiguous()
    u_arg = u if u is not None else mu_v_t
    cfgs = tk.launch_config(K)
    m, log_l, ce = (torch.empty(S, dtype=dtype) for _ in range(3))
    _grid_launch(tk.el_fwd_kernel, -(-S // cfgs["fwd"]["BI"]), cfgs["fwd"], mu_q, sigma_q, mu_v_t, sigma_v_t, u_arg,
                 targets, tau, m, log_l, ce, S, V, K=K, HAS_U=u is not None, FAST_LOG=False)
    scale = torch.full((1,), 0.37, dtype=dtype)
    common = (mu_q, sigma_q, mu_v_t, sigma_v_t, u_arg, targets, tau, m, log_l, scale)
    consts = dict(K=K, KP=cfgs["rows"]["KP"], HAS_U=u is not None, FAST_LOG=False)
    g_mu_q, g_sigma_q, g_e = torch.zeros(S, K, dtype=dtype), torch.zeros(S, K, dtype=dtype), torch.zeros(S, dtype=dtype)
    rows, cols = cfgs["rows"], cfgs["cols"]
    _grid_launch(rows_kernel, -(-S // rows["BI"]), rows, *common, g_mu_q, g_sigma_q, g_e, S, V, **consts)
    g_mu_v, g_sigma_v, g_u = torch.zeros(V, K, dtype=dtype), torch.zeros(V, K, dtype=dtype), torch.zeros(V, dtype=dtype)
    _grid_launch(cols_kernel, -(-V // cols["BV"]), cols, *common, g_mu_v, g_sigma_v, g_u, S, V, **consts)
    return [g_mu_q, g_sigma_q, g_e, g_mu_v, g_sigma_v, g_u]


out = []
g = torch.Generator().manual_seed(1)
for K, dtype, unigram in ((3, torch.float32, True), (20, torch.float32, False), (20, torch.float64, True),
                          (40, torch.float32, True), (70, torch.float64, False), (192, torch.float32, True)):
    S, V = 7, 70
    mu_q, sigma_q = torch.randn(S, K, generator=g, dtype=dtype), 0.2 + torch.rand(S, K, generator=g, dtype=dtype)
    mu_v, sigma_v = torch.randn(V, K, generator=g, dtype=dtype), 0.2 + torch.rand(V, K, generator=g, dtype=dtype)
    u = torch.randn(V, generator=g, dtype=dtype) if unigram else None
    tau = torch.full((1,), 0.9 * K, dtype=dtype)
    targets = torch.randint(0, V, (S,), generator=g)
    args = (mu_q, sigma_q, mu_v, sigma_v, u, tau, targets)
    new = backward(tk.el_bwd_rows_kernel, tk.el_bwd_cols_kernel, *args)
    ref = backward(old.el_bwd_rows_kernel, old.el_bwd_cols_kernel, *args)
    out.append({"K": K, "dtype": str(dtype), "bitwise": [bool(torch.equal(a, b)) for a, b in zip(new, ref)],
                "nonzero": [bool(t.abs().max() > 0) for t in new]})
print(json.dumps(out))
"""


def test_a206_two_level_placement_is_bitwise_the_previous_kernels(tmp_path: Path) -> None:
    (tmp_path / "el_reference_kernels.py").write_text(_EL_REFERENCE_SOURCE, encoding="utf-8")
    report = _run_script(_EL_BITWISE, _subprocess_env(TRITON_INTERPRET="1", VFE3_EL_REFERENCE_DIR=str(tmp_path)))
    for entry in report:
        assert all(entry["bitwise"]) and all(entry["nonzero"]), entry


# ------------------------------------------------------------------------------------ offline sm_120 compile

_COMPILE = """
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
from vfe3.kernels import _cue_result_triton as cr
from vfe3.kernels import _expected_likelihood_triton as el
assert not cr.INTERPRETED


def compile_one(kernel, consts, ints, pointer_types, num_warps, num_stages):
    sig = {n: ("constexpr" if n in consts else "i32" if n in ints else pointer_types.get(n, "*fp32"))
           for n in kernel.arg_names}
    compiled = triton.compile(ASTSource(fn=kernel, signature=sig, constexprs=consts), target=GPUTarget("cuda", 120, 32),
                              options={"num_warps": num_warps, "num_stages": num_stages})
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "w") as f:
            f.write(compiled.asm["ptx"])
        log = subprocess.run([ptxas, "-v", "--gpu-name=sm_120a", src, "-o", src + ".o"], capture_output=True, text=True)
    text = log.stdout + log.stderr
    spill = re.search(r"(\\d+) bytes spill stores, (\\d+) bytes spill loads", text)
    return {"registers": int(re.search(r"Used (\\d+) registers", text).group(1)),
            "spill_stores": int(spill.group(1)), "spill_loads": int(spill.group(2))}


out = {}
cfgs = cr.launch_config()
cr_ints = {"R", "M", "K", "NKT", "LP_STRIDE", "ROWS_PER_SPLIT"}
kernels = {"fwd_weights": cr.cr_fwd_weights_kernel, "fwd_weights_scan": cr.cr_fwd_weights_scan_kernel,
           "fwd_moments": cr.cr_fwd_moments_kernel, "bwd_pairs": cr.cr_bwd_pairs_kernel,
           "bwd_rows": cr.cr_bwd_rows_kernel, "bwd_cols": cr.cr_bwd_cols_kernel}
for rank in (1, 2, 4):
    for key, kernel in kernels.items():
        cfg = cfgs[key.replace("_scan", "")]
        consts = {"BA": cfg["BA"], "BK": cfg["BK"], "RANK": rank, "RP": 1 << (rank - 1).bit_length(), "ACC": True}
        if "WRITE_LOGW" in kernel.arg_names:
            consts["WRITE_LOGW"] = True
        nktps = (8, 32) if "NKTP" in kernel.arg_names else (None,)
        for nktp in nktps:
            if nktp is not None:
                consts["NKTP"] = nktp
            label = f"cr_{key}_rank{rank}" + ("" if nktp is None else f"_nktp{nktp}")
            out[label] = compile_one(kernel, consts, cr_ints, {"IND": "*i32"}, cfg["num_warps"], cfg["num_stages"])
for K in (20, 40, 70, 192):
    cfgs = el.launch_config(K)
    for key, kernel in (("rows", el.el_bwd_rows_kernel), ("cols", el.el_bwd_cols_kernel)):
        cfg = cfgs[key]
        consts = {"K": K, "KP": cfg["KP"], "BI": cfg["BI"], "BV": cfg["BV"], "HAS_U": True, "FAST_LOG": True}
        out[f"el_{key}_K{K}"] = compile_one(kernel, consts, {"S", "V"}, {"TGT": "*i64"}, cfg["num_warps"],
                                            cfg["num_stages"])
print(json.dumps(out))
"""


def test_changed_kernels_compile_for_sm120(tmp_path: Path, record_property: Any) -> None:
    report = _run_script(_COMPILE, _subprocess_env(TRITON_CACHE_DIR=str(tmp_path / "triton_cache")))
    for key, r in report.items():
        record_property(f"sm120_{key}", json.dumps(r))
    for key, r in report.items():
        assert r["registers"] <= 255, (key, r)
    for key in ("el_rows_K192", "el_cols_K192", "el_cols_K20"):    # spill-free before the change (K = 192: audit log)
        assert report[key]["spill_stores"] == 0 and report[key]["spill_loads"] == 0, (key, report[key])
    for rank in (1, 2):                    # the launcher's rank and rank 2 at K <= 256: no spills (NKTP 32 is recorded)
        for key, r in report.items():
            if key.startswith("cr_") and f"_rank{rank}" in key and not key.endswith("_nktp32"):
                assert r["spill_stores"] == 0 and r["spill_loads"] == 0, (key, r)


# ------------------------------------------------------------------------------------------------ A341, A343

def _fake_kernels(bk: int) -> SimpleNamespace:
    tile = dict(BA=32, BK=bk, num_warps=4, num_stages=2)
    return SimpleNamespace(launch_config=lambda: {"fwd_weights": dict(tile, scan_tile_sums=False),
                                                  "fwd_moments": dict(tile, BA=16), "bwd_pairs": dict(tile, BA=16),
                                                  "bwd_rows": dict(tile, BA=16), "bwd_cols": dict(tile, BA=8)})


def test_a341_tile_width_above_the_packed_bit_width_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fused_kernels, "_kernels", lambda: _fake_kernels(32))
    assert fused_kernels._tile_k() == 32
    monkeypatch.setattr(fused_kernels, "_kernels", lambda: _fake_kernels(64))
    with pytest.raises(RuntimeError, match="at most 32"):
        fused_kernels._tile_k()


def test_a343_plane_budget_reads_the_kernels_tile_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fused_kernels, "_kernels", lambda: _fake_kernels(16))
    monkeypatch.setattr(fused_kernels, "fused_route_active", lambda xi: True)
    seen: Dict[str, int] = {}

    def stub(x: torch.Tensor, c: torch.Tensor, nu: torch.Tensor, omega: torch.Tensor, lam: torch.Tensor,
             q: torch.Tensor, lp: torch.Tensor, U: torch.Tensor, V: torch.Tensor, *, row_chunk: int) -> tuple:
        seen["row_chunk"] = row_chunk
        R, M = x.shape[0], nu.shape[0]
        return x.clone(), c.clone(), torch.zeros(R, M, dtype=x.dtype), torch.zeros(R, dtype=x.dtype)

    monkeypatch.setattr(fused_kernels, "cue_result_diagonal_lowrank_fused", stub)
    R, M, K, rank, tile = 4, 3, 40, 2, 10 ** 6
    d = torch.float64
    cue_result_diagonal_posterior(torch.zeros(R, K, dtype=d), torch.ones(R, K, dtype=d), torch.zeros(M, K, dtype=d),
                                  torch.zeros(M, K, dtype=d), torch.zeros(M, K, dtype=d), torch.zeros(M, K, dtype=d),
                                  torch.zeros(M, dtype=d), regression_left=torch.zeros(M, K, rank, dtype=d),
                                  regression_right=torch.zeros(M, K, rank, dtype=d), tile_bytes=tile, fused=True)
    planes = fused_kernels.plane_count(rank, K, 16)
    assert planes != fused_kernels.plane_count(rank, K, 32)          # K = 40: three tiles of 16, two of 32
    assert seen["row_chunk"] == tile // (8 * planes * M)


# ------------------------------------------------------------------------------------------------ A315, A325

def _gathered(seed: int = 6) -> Tuple[torch.Tensor, ...]:
    table = 0.1 * torch.randn(3, 2, 2, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    unique_forward = torch.linalg.matrix_exp(table).float()                       # (U, H, d, d)
    unique_inverse = torch.linalg.inv(unique_forward.double()).float()
    index = torch.tensor([2, 0, 2, 1, 0, 1, 2, 0])
    forward = unique_forward.index_select(0, index).reshape(2, 4, 2, 2, 2)       # (B, N, H, d, d) occurrences
    inverse = unique_inverse.index_select(0, index).reshape(2, 4, 2, 2, 2)
    mark_gathered_frame_pair(forward, inverse, unique_forward, unique_inverse)
    return forward, inverse, unique_forward, unique_inverse


@pytest.mark.parametrize("side", ["forward", "inverse"])
def test_a315_in_place_write_to_a_unique_tensor_voids_the_gather_record(side: str) -> None:
    forward, inverse, unique_forward, unique_inverse = _gathered()
    assert _gathered_source(forward, inverse) is not None
    unique = unique_forward if side == "forward" else unique_inverse
    unique[1].mul_(1.01)                                             # the unique pair no longer matches its copies
    assert _gathered_source(forward, inverse) is None
    CanonicalFrameContext(forward, inverse, True)                    # the occurrences, still exact copies, validate


@pytest.mark.parametrize("flag", [1, 0, np.bool_(True), "yes"])
def test_a325_both_frame_constructors_raise_one_dual_type_error(flag: Any) -> None:
    eye = torch.eye(2).expand(3, 2, 2, 2)
    for build in (lambda: CanonicalFrameContext(eye, eye, flag), lambda: trusted_identity_context(eye, flag)):
        with pytest.raises(TypeError, match="block_diagonal must be a bool"):
            build()
        with pytest.raises(ValueError, match="block_diagonal must be a bool"):
            build()


# ------------------------------------------------------------------------------------------------ A283

def test_a283_diagonal_band_is_the_mean_band_mass_of_the_rows_with_a_band_entry() -> None:
    n = 6
    causal = torch.tril(torch.ones(n, n))
    beta = (causal / causal.sum(-1, keepdim=True)).expand(2, n, n)      # uniform causal attention, two heads
    wide = structured_head_scores(beta, band_width=2)
    # rows 1..5 put 1/2, 2/3, 2/4, 2/5, 2/6 on their two previous tokens: mean 0.48 (the per-pair mean was 0.2667)
    assert wide["diagonal_band"].shape == (2,)
    assert wide["diagonal_band"][0].item() == pytest.approx(0.48)
    narrow = structured_head_scores(beta, band_width=1)
    assert torch.allclose(narrow["diagonal_band"], narrow["prev_token"])
    assert structured_head_scores(torch.ones(1, 1, 1))["diagonal_band"].item() == 0.0   # no row has a band entry
