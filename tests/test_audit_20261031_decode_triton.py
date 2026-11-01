r"""Audit 2026-10-31 remediation, package ``decode``: the fused Triton decode routes under the Triton interpreter.

triton-kernels-2 (A103)    a second derivative through KLDecodeCE, ExpectedLikelihoodCE, DiagonalRootSquareSums or
                           CueResultDiagonalLowRank raises for a leaf upstream of the Function (a nonlinear map),
                           through ``torch.autograd.grad`` and through ``.backward()``, and the first-order gradients
                           of a ``create_graph=True`` call are the plain ones bitwise.
triton-kernels-3 (A164)    in keep mode the decode backward writes G over the kept cross term (no (S, Vc) scratch),
                           matches the recompute mode bitwise, and a retained graph differentiated twice recomputes A.
numerics-1 (A007)          ``decode_backward_ieee_fp32`` pins the fused route's two backward GEMMs.
prior-bank-decode-1 (A028) the fused route under "fp64" equals a float64 bank's torch route.
dead-code-6 (A329),        the default fp32 fused route (kept and recomputed cross terms) is bitwise the 5bb0a57 fused
  prior-bank-decode-2      branch: value and every gradient.
perf-live-path-3 (A154)    ``decode_bank_triton``: the fused bank preparation matches the torch route (CE and every
                           table gradient), passes a float64 gradcheck, keeps the clamp masks, replaces the
                           vocabulary-table-sized torch kernels of the preparation, and its two kernels compile
                           offline for the RTX 5090 target (sm_120a) without register spills.

Every check runs the Triton interpreter on CPU tensors in a subprocess (skipped where triton is missing), as the other
fused-kernel tests do, so this process never imports triton; the compile check runs its subprocess uninterpreted and
needs no GPU.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton (or ptxas) unavailable


def _run_script(script: str, *, interpret: bool = True, **extra_env: str) -> Any:
    env: Dict[str, str] = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), **extra_env)
    if interpret:
        env["TRITON_INTERPRET"] = "1"
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_PRELUDE = """
import json, math, sys, copy, warnings
warnings.simplefilter("ignore")
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
from tests.profiles import make_prior_bank
from vfe3.model.prior_bank import PriorBank

def forced(self, lhs, u_all):                       # the interpreter runs the fused route on CPU tensors
    self.realized_decode_ce_kernel = "triton" if self.decode_ce_triton else "torch"
    return self.decode_ce_triton
PriorBank._decode_ce_triton_active = forced

def outcome(fn):
    try:
        fn()
    except RuntimeError as exc:
        return "once_differentiable" if "once_differentiable" in str(exc) else "raised: " + str(exc)[:80]
    return "returned"

def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))
"""


_DOUBLE_BACKWARD_SCRIPT = _PRELUDE + """
from vfe3 import relational_scoring as rs
from vfe3.kernels import cue_result as fused_kernels
from vfe3.kernels.expected_likelihood import expected_likelihood_ce
from vfe3.kernels.kl_decode import kl_decode_ce_rows
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior

rs._diagonal_moments_triton_active = lambda *tensors: rs._DIAG_MOMENTS_KERNEL == "triton"
fused_kernels.fused_route_active = lambda xi: True
dt = torch.float64
g = torch.Generator().manual_seed(0)
out = {}

def check(name, loss_fn, x0):
    # first order: create_graph=True returns the plain gradient bitwise
    x = x0.clone().requires_grad_()
    plain, = torch.autograd.grad(loss_fn(x), x)
    x = x0.clone().requires_grad_()
    first, = torch.autograd.grad(loss_fn(x), x, create_graph=True)
    out[name + "_first_bitwise"] = bool(torch.equal(first.detach(), plain))
    out[name + "_grad"] = outcome(lambda: torch.autograd.grad(first.sum(), x))
    x = x0.clone().requires_grad_()
    first, = torch.autograd.grad(loss_fn(x), x, create_graph=True)
    out[name + "_backward"] = outcome(lambda: first.sum().backward())

S, V, K = 4, 11, 3
sq = 0.5 + torch.rand(S, K, generator=g, dtype=dt)
inv_v = 1.0 / (0.5 + torch.rand(V, K, generator=g, dtype=dt))
mc_v = torch.randn(V, K, generator=g, dtype=dt)
rhs = torch.cat([inv_v, mc_v * inv_v], -1)
bias = (mc_v ** 2 * inv_v).sum(-1) - torch.log(inv_v).sum(-1)
per_pos = K + torch.log(sq).sum(-1)
tau = torch.tensor([0.7], dtype=dt)
targets = torch.randint(0, V, (S,), generator=g)
for keep in (True, False):
    check(f"kl_keep{keep}_query", lambda x: kl_decode_ce_rows(torch.cat([sq + x ** 2, -2.0 * x], -1), per_pos, rhs,
                                                               bias, None, tau, targets, chunk=4, keep=keep).sum(),
          torch.randn(S, K, generator=g, dtype=dt))
    check(f"kl_keep{keep}_table", lambda y: kl_decode_ce_rows(torch.cat([sq + 0.3, -0.6 * torch.ones_like(sq)], -1),
                                                               per_pos, rhs * torch.exp(0.1 * y), bias, None, tau,
                                                               targets, chunk=4, keep=keep).sum(),
          torch.randn(V, 2 * K, generator=g, dtype=dt))

sigma_q = 0.5 + torch.rand(S, K, generator=g, dtype=dt)
mu_v = torch.randn(V, K, generator=g, dtype=dt)
sigma_v = 0.5 + torch.rand(V, K, generator=g, dtype=dt)
mu_q = torch.randn(S, K, generator=g, dtype=dt)
check("el_query", lambda x: expected_likelihood_ce(x + 0.1 * x ** 2, sigma_q, mu_v, sigma_v, None, tau, targets),
      torch.randn(S, K, generator=g, dtype=dt))
check("el_table_mean", lambda y: expected_likelihood_ce(mu_q, sigma_q, y + 0.1 * y ** 2, sigma_v, None, tau, targets),
      torch.randn(V, K, generator=g, dtype=dt))
check("el_table_variance", lambda y: expected_likelihood_ce(mu_q, sigma_q, mu_v, torch.exp(y), None, tau, targets),
      0.2 * torch.randn(V, K, generator=g, dtype=dt))

B, N, H, M, d, r = 1, 3, 2, 2, 4, 2
Kd = M * d
mu = torch.randn(B, N, Kd, generator=g, dtype=dt)
inverse = torch.eye(d, dtype=dt) + 0.3 * torch.randn(B, N, M, d, d, generator=g, dtype=dt)
Smaps = torch.randn(2, H, r, Kd, generator=g, dtype=dt) / Kd ** 0.5
wg = torch.randn(2, B, N, H, r, generator=g, dtype=dt)

def dm(x):
    sigma = torch.exp(x)
    previous = rs.set_diagonal_moments_kernel("triton")
    try:
        _, gd = rs.projected_moments(mu.unsqueeze(-2).expand(B, N, H, Kd), sigma.unsqueeze(-2).expand(B, N, H, Kd),
                                     inverse, Smaps, diagonal_only=True)
    finally:
        rs.set_diagonal_moments_kernel(previous)
    return (gd * wg).sum()
check("diagonal_moments", dm, 0.3 * torch.randn(B, N, Kd, generator=g, dtype=dt))

R, Mc, Kc, rank = 3, 4, 5, 2
c = 0.5 + torch.rand(R, Kc, generator=g, dtype=dt)
nu, omega = torch.randn(Mc, Kc, generator=g, dtype=dt), torch.randn(Mc, Kc, generator=g, dtype=dt)
log_lambda = 0.2 * torch.randn(Mc, Kc, generator=g, dtype=dt)
log_q = -1.0 + 0.2 * torch.randn(Mc, Kc, generator=g, dtype=dt)
log_pi = torch.randn(Mc, generator=g, dtype=dt)
U = 0.3 * torch.randn(Mc, Kc, rank, generator=g, dtype=dt)
Vr = 0.3 * torch.randn(Mc, Kc, rank, generator=g, dtype=dt)
w_mean, w_var = torch.randn(R, Kc, generator=g, dtype=dt), torch.randn(R, Kc, generator=g, dtype=dt)

def cr(x):
    post = cue_result_diagonal_posterior(x + 0.1 * x ** 2, c, nu, omega, log_lambda, log_q, log_pi,
                                         regression_left=U, regression_right=Vr, fused=True)
    return (post.mean * w_mean).sum() + (post.covariance * w_var).sum()
check("cue_result", cr, torch.randn(R, Kc, generator=g, dtype=dt))
print(json.dumps(out))
"""


def test_a_double_backward_through_each_fused_function_raises() -> None:
    results = _run_script(_DOUBLE_BACKWARD_SCRIPT)
    names = {key.rsplit("_", 1)[0] for key in results if key.endswith("_grad")}
    assert names == {"kl_keepTrue_query", "kl_keepTrue_table", "kl_keepFalse_query", "kl_keepFalse_table",
                     "el_query", "el_table_mean", "el_table_variance", "diagonal_moments", "cue_result"}, names
    for name in names:
        assert results[name + "_first_bitwise"] is True, (name, results)
        # Before the fix each second derivative came back 50 to 92 percent wrong with no error.
        assert results[name + "_grad"] == "once_differentiable", (name, results)
        assert results[name + "_backward"] == "once_differentiable", (name, results)


_KEEP_SCRIPT = _PRELUDE + """
from torch.utils._python_dispatch import TorchDispatchMode
from vfe3.kernels.kl_decode import kl_decode_ce_rows

class Allocations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.sizes = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.empty.memory_format, torch.ops.aten.empty_strided.default):
            self.sizes.append(int(out.numel()))
        return out

out = {}
g = torch.Generator().manual_seed(0)
for dt in (torch.float32, torch.float64):
    S, V, K, chunk = 7, 23, 3, 8
    lhs0 = 0.5 + torch.rand(S, 2 * K, generator=g, dtype=dt)
    rhs0 = 0.5 + torch.rand(V, 2 * K, generator=g, dtype=dt)
    bias0 = torch.rand(V, generator=g, dtype=dt)
    pp0 = torch.rand(S, generator=g, dtype=dt)
    u = torch.randn(V, generator=g, dtype=dt)
    targets = torch.randint(0, V, (S,), generator=g)
    w = torch.linspace(0.5, 1.5, S, dtype=dt)
    runs = {}
    for keep in (False, True):
        leaves = [t.clone().requires_grad_() for t in (lhs0, pp0, rhs0, bias0)]
        tau = torch.tensor([0.4], dtype=dt, requires_grad=True)
        ce = kl_decode_ce_rows(leaves[0], leaves[1], leaves[2], leaves[3], u, tau, targets, chunk=chunk, keep=keep)
        loss = (ce * w).sum()
        with Allocations() as spy:
            first = torch.autograd.grad(loss, leaves + [tau], retain_graph=True)
        second = torch.autograd.grad(loss, leaves + [tau])                  # the retained graph, again
        runs[keep] = (ce, first, second, spy.sizes)
    key = str(dt)
    out[key + "_ce_equal"] = bool(torch.equal(runs[False][0], runs[True][0]))
    out[key + "_keep_matches_recompute"] = all(torch.equal(a, b) for a, b in zip(runs[False][1], runs[True][1]))
    out[key + "_second_matches_first"] = all(torch.equal(a, b) for a, b in zip(runs[True][1], runs[True][2]))
    out[key + "_scratch_allocations"] = sum(1 for n in runs[True][3] if n == S * min(chunk, V))
print(json.dumps(out))
"""


def test_keep_mode_writes_g_over_the_kept_cross_term() -> None:
    results = _run_script(_KEEP_SCRIPT)
    for dt in ("torch.float32", "torch.float64"):
        assert results[dt + "_ce_equal"] is True, results
        assert results[dt + "_keep_matches_recompute"] is True, results
        assert results[dt + "_second_matches_first"] is True, results
        assert results[dt + "_scratch_allocations"] == 0, results


_PIN_SCRIPT = _PRELUDE + """
from torch.utils._python_dispatch import TorchDispatchMode
from vfe3.kernels.kl_decode import kl_decode_ce_rows

aten = torch.ops.aten
GEMMS = {aten.mm.default, aten.mm.out, aten.addmm.default, aten.addmm_.default, aten.addmm.out, aten.bmm.default}

class Policies(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in GEMMS:
            self.seen.append(torch.get_float32_matmul_precision())
        return func(*args, **(kwargs or {}))

out = {}
g = torch.Generator().manual_seed(0)
S, V, K = 5, 19, 3
torch.set_float32_matmul_precision("high")
for ieee in (False, True):
    lhs = (0.5 + torch.rand(S, 2 * K, generator=g)).requires_grad_()
    rhs = (0.5 + torch.rand(V, 2 * K, generator=g)).requires_grad_()
    bias = torch.rand(V, generator=g).requires_grad_()
    ce = kl_decode_ce_rows(lhs, torch.rand(S, generator=g), rhs, bias, None, torch.tensor([0.4]),
                           torch.randint(0, V, (S,), generator=g), chunk=8, keep=True, ieee_backward=ieee)
    with Policies() as spy:
        ce.sum().backward()
    out[f"rows_{ieee}"] = spy.seen
    pb = make_prior_bank(V, 4, 3, decode_mode="diagonal_chunked", decode_ce_triton=True, decode_chunk_size=8,
                         decode_backward_ieee_fp32=ieee)
    mu = torch.randn(1, 3, 4, generator=g, requires_grad=True)
    ce = pb.decode_ce_diagonal_chunked(mu, torch.ones(1, 3, 4), torch.tensor([[0, 9, 18]]))
    assert pb.realized_decode_ce_kernel == "triton"
    with Policies() as spy:
        ce.backward()
    out[f"bank_{ieee}"] = spy.seen
print(json.dumps(out))
"""


def test_the_fused_route_backward_gemms_follow_the_toggle() -> None:
    results = _run_script(_PIN_SCRIPT)
    for prefix in ("rows", "bank"):
        # three vocabulary chunks, two GEMMs each (dlhs = G rhs, drhs = G^T lhs)
        assert results[f"{prefix}_False"] == ["high"] * 6, results
        assert results[f"{prefix}_True"] == ["highest"] * 6, results


_FP64_SCRIPT = _PRELUDE + """
out = {}
for mode in ("diagonal_chunked", "full_chunked"):
    full = mode == "full_chunked"
    V, K = 37, 8
    torch.manual_seed(0)
    pb = make_prior_bank(V, K, 3, decode_mode=mode, family="gaussian_full" if full else "gaussian_diagonal",
                         diagonal_covariance=not full, decode_tau=0.125, decode_chunk_size=16,
                         decode_av_precision="fp64", decode_unigram_prior=True, untie_decode_bank=True,
                         decode_ce_triton=True, irrep_dims=[4, 4])
    pb.decode_log_scale.requires_grad_(False)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        pb.decode_mu_embed.copy_(torch.randn(V, K, generator=g))
        pb.decode_sigma_log_embed.copy_(math.log(1e-3) + 0.5 * torch.randn(V, K, generator=g))
    pb.set_unigram_log_prior(torch.randint(1, 40, (V,), generator=g).float())
    t = torch.randint(0, V, (2, 6), generator=g)
    mu0 = pb.decode_mu_embed.detach()[t] + 0.02 * torch.randn(2, 6, K, generator=g)
    s0 = 1e-3 * torch.exp(0.3 * torch.randn(2, 6, K, generator=g))
    sigma0 = torch.diag_embed(s0) if full else s0
    targets = torch.randint(0, V, (2, 6), generator=g)
    targets[0, 2] = -100
    reference = copy.deepcopy(pb).double()
    reference.decode_ce_triton = False

    def run(bank, dtype):
        mu = mu0.to(dtype).requires_grad_()
        sigma = sigma0.to(dtype).requires_grad_()
        fn = bank.decode_ce_full_chunked if full else bank.decode_ce_diagonal_chunked
        ce = fn(mu, sigma, targets)
        return ce, torch.autograd.grad(ce, [mu, sigma, bank.decode_mu_embed, bank.decode_sigma_log_embed])

    ce, grads = run(pb, torch.float32)
    assert pb.realized_decode_ce_kernel == "triton"
    ce_ref, grads_ref = run(reference, torch.float64)
    assert reference.realized_decode_ce_kernel == "torch"
    out[mode] = {"ce": abs(float(ce) - float(ce_ref)) / abs(float(ce_ref)), "dtype": str(ce.dtype),
                 "grads": [rel(a, b.to(a.dtype)) for a, b in zip(grads, grads_ref)]}
print(json.dumps(out))
"""


def test_the_fused_route_under_fp64_is_a_float64_banks_torch_route() -> None:
    results = _run_script(_FP64_SCRIPT)
    for mode, r in results.items():
        assert r["dtype"] == "torch.float64", (mode, r)
        assert r["ce"] <= 1e-12, (mode, r)
        assert all(err <= 2e-6 for err in r["grads"]), (mode, r)


_FUSED_HEAD_SCRIPT = _PRELUDE + """
from vfe3.kernels.kl_decode import kl_decode_ce_rows
from vfe3.model.prior_bank import _decode_av_lhs, _decode_ce_chunk_activation_bytes, _decode_ce_should_checkpoint
from vfe3.numerics import bounded_variance_from_log

def head_fused_diagonal(self, mu_q, sigma_q, targets, ignore_index=-100):
    # The fused branch of decode_ce_diagonal_chunked and the body of _decode_ce_triton_rows at 5bb0a57 (fp32 route).
    has_ignored = self._validate_fused_ce_targets(targets, ignore_index=ignore_index, expected_shape=mu_q.shape[:-1])
    original_valid = targets != ignore_index
    if has_ignored:
        mu_q = mu_q[original_valid].unsqueeze(0)
        sigma_q = sigma_q[original_valid].unsqueeze(0)
        targets = targets[original_valid].unsqueeze(0)
    tau_eff = self._tau_eff(None)
    chunk = self.decode_chunk_size
    V = self.vocab_size
    sigma_v_all = bounded_variance_from_log(self._decode_sigma_log_table(), eps=self.eps)
    mu_v_all = self._decode_mu_table()
    c = mu_v_all.mean(dim=0, keepdim=True)
    u_all = self._unigram_bias() if self.decode_unigram_prior else None
    mc_q = mu_q - c
    lhs = _decode_av_lhs(sigma_q, mc_q)
    per_pos = self.K + torch.log(sigma_q.clamp(min=self.eps)).sum(-1, keepdim=True)
    valid = targets != ignore_index
    work = torch.float64 if self.decode_av_precision == "fp64" else lhs.dtype
    lhs_w = lhs if lhs.dtype is work else _decode_av_lhs(sigma_q.to(work), mc_q.to(work))
    mc_v = (mu_v_all - c).to(work)
    inv_v = (1.0 / sigma_v_all).to(work)
    lsum = torch.log(sigma_v_all).sum(-1).to(work)
    rhs = torch.cat([inv_v, mc_v * inv_v], dim=-1)
    bias = (mc_v ** 2 * inv_v).sum(-1) + lsum
    grad_active = torch.is_grad_enabled() and any(
        value is not None and value.requires_grad for value in (lhs, per_pos, mu_v_all, sigma_v_all, tau_eff))
    aggregate = self.decode_ce_checkpoint == "auto_aggregate"
    activation_bytes = _decode_ce_chunk_activation_bytes(
        lhs, V if aggregate else min(chunk, V), inner=2,
        workspace_bytes_per_scalar=(8 if aggregate and self.decode_av_precision == "fp64" else None))
    keep = grad_active and not _decode_ce_should_checkpoint(self.decode_ce_checkpoint, grad_active, activation_bytes)
    ce_rows = kl_decode_ce_rows(
        lhs_w.reshape(-1, lhs_w.shape[-1]), per_pos.to(work).reshape(-1), rhs, bias,
        u_all.to(work) if u_all is not None else None, tau_eff.to(work), targets.reshape(-1),
        chunk=chunk, keep=keep).reshape(targets.shape)
    return (ce_rows * valid).sum() / valid.sum().clamp_min(1), keep

out = {}
V, K = 29, 4
for checkpoint in ("auto", "always"):
    runs = {}
    for impl in ("current", "head"):
        torch.manual_seed(0)
        pb = make_prior_bank(V, K, 3, decode_mode="diagonal_chunked", decode_tau=0.3, decode_chunk_size=8,
                             decode_ce_checkpoint=checkpoint, decode_unigram_prior=True, untie_decode_bank=True,
                             decode_ce_triton=True, mu_init_std=0.7, irrep_dims=[2, 2])
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            pb.decode_sigma_log_embed.copy_(0.6 * torch.randn(V, K, generator=g))
        pb.set_unigram_log_prior(torch.randint(1, 30, (V,), generator=g).float())
        mu = (0.8 * torch.randn(2, 5, K, generator=g)).requires_grad_()
        sigma = 0.3 + torch.rand(2, 5, K, generator=g)
        sigma[0, 0, 0] = 1e-6                                       # at eps: the clamp is the identity
        sigma.requires_grad_()
        targets = torch.randint(0, V, (2, 5), generator=g)
        targets[1, 3] = -100
        if impl == "current":
            ce, keep = pb.decode_ce_diagonal_chunked(mu, sigma, targets), None
            assert pb.realized_decode_ce_kernel == "triton"
        else:
            ce, keep = head_fused_diagonal(pb, mu, sigma, targets)
        leaves = [mu, sigma, pb.decode_mu_embed, pb.decode_sigma_log_embed, pb.decode_log_scale]
        runs[impl] = (ce, torch.autograd.grad(ce, leaves), keep)
    (ce, grads, _), (ce_head, grads_head, keep) = runs["current"], runs["head"]
    out[checkpoint] = {"keep": keep, "ce": bool(torch.equal(ce, ce_head)),
                       "grads": [bool(torch.equal(a, b)) for a, b in zip(grads, grads_head)]}
print(json.dumps(out))
"""


def test_the_fused_route_is_bitwise_the_previous_fused_route() -> None:
    r"""The fused Triton route of the default fp32 diagonal decode CE, kept cross terms and recomputed ones: value and
    every gradient ``torch.equal`` to the 5bb0a57 fused branch (whose KLDecodeCE is the current one; that G may
    overwrite a kept A is pinned by ``test_keep_mode_writes_g_over_the_kept_cross_term``)."""
    results = _run_script(_FUSED_HEAD_SCRIPT)
    assert results["auto"]["keep"] is True and results["always"]["keep"] is False, results
    for checkpoint, r in results.items():
        assert r["ce"] and all(r["grads"]), (checkpoint, r)


_BANK_SCRIPT = _PRELUDE + """
from torch.utils._python_dispatch import TorchDispatchMode
from vfe3.kernels.decode_bank import decode_bank_factors

class TableOps(TorchDispatchMode):
    r'''Count torch op outputs of a vocabulary-table shape (V, K) or (V, 2K): allocations, detached aliases and the
    interpreter's own storage rebinding (set_) excluded, so the count is the table-sized kernels a CUDA run launches.'''
    def __init__(self, V):
        super().__init__()
        self.V, self.count = V, 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        aten = torch.ops.aten
        skip = (aten.empty.memory_format, aten.empty_strided.default, aten.zeros.default, aten.zeros_like.default,
                aten.empty_like.default, aten.detach.default, aten.alias.default,
                aten.set_.source_Storage_storage_offset)
        if func not in skip:
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == self.V and t.shape[1] in (4, 8):
                    self.count += 1
        return out

out = {}
for dt in (torch.float32, torch.float64):
    for mode in ("diagonal_chunked", "full_chunked"):
        full = mode == "full_chunked"
        V, K = 41, 4
        g = torch.Generator().manual_seed(3)
        def bank(fused):
            torch.manual_seed(0)
            pb = make_prior_bank(V, K, 3, decode_mode=mode, family="gaussian_full" if full else "gaussian_diagonal",
                                 diagonal_covariance=not full, decode_tau=0.3, decode_chunk_size=16,
                                 decode_unigram_prior=True, untie_decode_bank=True, decode_ce_triton=True,
                                 decode_bank_triton=fused, mu_init_std=0.5, irrep_dims=[2, 2]).to(dt)
            pb.decode_log_scale.requires_grad_(False)
            with torch.no_grad():
                gg = torch.Generator().manual_seed(1)
                pb.decode_sigma_log_embed.copy_(0.5 * torch.randn(V, K, generator=gg, dtype=dt))
                pb.decode_sigma_log_embed[0, 0] = math.log(pb.eps) - 0.5          # below the eps floor
                pb.decode_sigma_log_embed[1, 1] = 80.5                            # above max_log
                pb.decode_sigma_log_embed[2, 2] = math.log(pb.eps) + 1e-3         # just above the floor
            pb.set_unigram_log_prior(torch.arange(1, V + 1, dtype=torch.float32))
            return pb
        mu0 = torch.randn(2, 5, K, generator=g, dtype=dt)
        if full:
            a = 0.3 * torch.randn(2, 5, K, K, generator=g, dtype=dt)
            sigma0 = a @ a.transpose(-1, -2) + torch.eye(K, dtype=dt)
        else:
            sigma0 = 0.5 + torch.rand(2, 5, K, generator=g, dtype=dt)
        targets = torch.randint(0, V, (2, 5), generator=g)
        targets[1, 1] = -100
        res = {}
        for fused in (False, True):
            pb = bank(fused)
            mu, sigma = mu0.clone().requires_grad_(), sigma0.clone().requires_grad_()
            fn = pb.decode_ce_full_chunked if full else pb.decode_ce_diagonal_chunked
            with TableOps(V) as spy:
                ce = fn(mu, sigma, targets)
                grads = torch.autograd.grad(ce, [mu, sigma, pb.decode_mu_embed, pb.decode_sigma_log_embed])
            res[fused] = (ce, grads, spy.count)
        key = f"{mode}_{dt}"
        out[key] = {"ce": abs(float(res[True][0]) - float(res[False][0])) / abs(float(res[False][0])),
                    "grads": [rel(a, b) for a, b in zip(res[True][1], res[False][1])],
                    "masked_rows_zero": bool(res[True][1][3][0, 0] == 0 and res[True][1][3][1, 1] == 0),
                    "zero_pattern_agrees": bool(torch.equal(res[True][1][3] == 0, res[False][1][3] == 0)),
                    "table_ops": [res[False][2], res[True][2]]}

# float64 gradcheck of the fused bank factors, with the broadcast term of c
g = torch.Generator().manual_seed(4)
V, K = 9, 3
mu_v = torch.randn(V, K, generator=g, dtype=torch.float64, requires_grad=True)
log_s = (0.5 * torch.randn(V, K, generator=g, dtype=torch.float64)).requires_grad_()
w_rhs = torch.randn(V, 2 * K, generator=g, dtype=torch.float64)
w_bias = torch.randn(V, generator=g, dtype=torch.float64)
w_c = torch.randn(1, K, generator=g, dtype=torch.float64)
def scalar(m, l):
    c = m.mean(dim=0, keepdim=True)
    rhs, bias = decode_bank_factors(m, l, c, eps=1e-6)
    return (rhs * w_rhs).sum() + (bias * w_bias).sum() + (c * w_c).sum()
out["gradcheck"] = bool(torch.autograd.gradcheck(scalar, (mu_v, log_s)))
print(json.dumps(out))
"""


def test_the_fused_bank_preparation_matches_the_torch_route() -> None:
    results = _run_script(_BANK_SCRIPT)
    assert results.pop("gradcheck") is True
    for key, r in results.items():
        tol = 1e-12 if key.endswith("float64") else 1e-6
        assert r["ce"] <= tol, (key, r)
        assert all(err <= tol for err in r["grads"]), (key, r)
        assert r["masked_rows_zero"] and r["zero_pattern_agrees"], (key, r)
        unfused, fused = r["table_ops"]
        assert fused <= 12 < unfused, (key, r)


_BANK_COMPILE_SCRIPT = """
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
from vfe3.kernels import _decode_bank_triton as tk
from vfe3.kernels.decode_bank import DECODE_BANK_MAX_LOG
assert not tk.INTERPRETED
cfg = tk.launch_config()
out = {}
for key, kernel in (("fwd", tk.bank_fwd_kernel), ("bwd", tk.bank_bwd_kernel)):
    for dt in ("fp32", "fp64"):
        # The specialization decode_bank_factors launches on CUDA: ACCURATE (libdevice), IEEE division in float32.
        consts = {"BV": cfg["BV"], "BK": cfg["BK"], "MAX_LOG": DECODE_BANK_MAX_LOG, "ACCURATE": True,
                  "PRECISE_DIV": dt == "fp32"}
        sig = {n: ("constexpr" if n in consts else "i32" if n in ("V", "K") else "*" + dt) for n in kernel.arg_names}
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
                               "div_rn": ptx.count("div.rn."), "returncode": log.returncode}
print(json.dumps(out))
"""


def test_the_fused_bank_kernels_compile_for_sm120_without_register_spills(tmp_path: Path) -> None:
    r"""Offline compile of both bank kernels for the RTX 5090 target (no GPU is used): ptxas accepts them with zero
    spill bytes, and the float32 build divides with IEEE ``div.rn`` as torch does. Nothing here is a timing."""
    report = _run_script(_BANK_COMPILE_SCRIPT, interpret=False, TRITON_CACHE_DIR=str(tmp_path / "triton_cache"))
    assert set(report) == {"fwd_fp32", "fwd_fp64", "bwd_fp32", "bwd_fp64"}, report
    for key, r in report.items():
        assert r["returncode"] == 0 and r["spill_stores"] == 0 and r["spill_loads"] == 0, (key, r)
        assert r["registers"] <= 255, (key, r)
    assert report["fwd_fp32"]["div_rn"] > 0 and report["bwd_fp32"]["div_rn"] > 0, report
