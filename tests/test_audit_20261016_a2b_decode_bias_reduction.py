r"""Audit 2026-10-16 F115: the fused KL-decode backward reduces d_bias in float32, not through a GEMM column.

``KLDecodeCE.backward`` formed ``[d_rhs, d_bias] = G^T [lhs, 1]`` in one GEMM, which runs under the caller's
float32 matmul policy; under ``'high'`` (the live setting) a CUDA GEMM rounds G to TF32, so d_bias carried
TF32 error, while the torch route's d_bias is the float32 reduction of the ``+ bias`` broadcast. The fused
backward now computes ``d_bias = sum_i G_iv`` as a reduction and ``d_rhs = G^T lhs`` (width 2K).

The CPU check runs the Triton interpreter in a subprocess with ``torch.mm`` replaced by a TF32 emulation
(inputs rounded to a 10-bit mantissa), so a d_bias that passes through a GEMM shows TF32 error. The CUDA check
(``VFE3_TEST_DEVICE=cuda``) compares the fused and torch routes against float64 under ``'high'``.
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

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton unavailable

requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")


def _run_script(script: str) -> Any:
    env: Dict[str, str] = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), TRITON_INTERPRET="1")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _operands(S: int, V: int, K: int, dtype: torch.dtype, device: str, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    r"""Packed fused-decode operands (lhs, per_pos, rhs, bias, tau, targets) at tau = 0.1."""
    g = torch.Generator().manual_seed(seed)
    lhs = 0.5 + torch.rand(S, 2 * K, generator=g, dtype=torch.float64)
    rhs = 0.5 + torch.rand(V, 2 * K, generator=g, dtype=torch.float64)
    bias = torch.rand(V, generator=g, dtype=torch.float64)
    per_pos = 2.0 * torch.rand(S, generator=g, dtype=torch.float64)
    tau = torch.tensor([0.1], dtype=torch.float64)
    targets = torch.randint(0, V, (S,), generator=g)
    floats = [x.float().to(dtype).to(device) for x in (lhs, per_pos, rhs, bias, tau)]
    return (*floats, targets.to(device))


def _reference_bias_grad(lhs: torch.Tensor, per_pos: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor,
                         tau: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""Float64 d(sum_i CE_i)/d b of the decode CE written out in torch (the fused route's formula)."""
    lhs, per_pos, rhs, tau = (x.detach().double().cpu() for x in (lhs, per_pos, rhs, tau))
    bias = bias.detach().double().cpu().requires_grad_()
    a = lhs @ rhs.t() + bias                                               # (S, V)
    logits = -torch.clamp(0.5 * (a - per_pos[:, None]), min=0.0) / tau    # (S, V)
    ce = torch.logsumexp(logits, -1) - logits.gather(1, targets.cpu()[:, None]).squeeze(1)
    return torch.autograd.grad(ce.sum(), bias)[0]


_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
from tests.test_audit_20261016_a2b_decode_bias_reduction import _operands, _reference_bias_grad
from vfe3.kernels.kl_decode import kl_decode_ce_rows

_mm = torch.mm

def _tf32(x):
    bits = x.contiguous().view(torch.int32)
    bits = (bits + 0x0FFF + ((bits >> 13) & 1)) & ~0x1FFF               # round to nearest even, 10-bit mantissa
    return bits.view(torch.float32)

def tf32_mm(a, b, *, out=None):
    return _mm(_tf32(a), _tf32(b), out=out) if a.dtype == torch.float32 else _mm(a, b, out=out)

torch.mm = tf32_mm
out = {}
for keep in (False, True):
    lhs, per_pos, rhs, bias, tau, targets = _operands(256, 40, 4, torch.float32, "cpu")
    bias.requires_grad_()
    rhs.requires_grad_()
    ce = kl_decode_ce_rows(lhs, per_pos, rhs, bias, None, tau, targets, chunk=16, keep=keep)
    (g_bias,) = torch.autograd.grad(ce.sum(), (bias,))
    ref = _reference_bias_grad(lhs, per_pos, rhs, bias, tau, targets)
    out[f"keep{keep}"] = float((g_bias.double() - ref).norm() / ref.norm())
print(json.dumps(out))
"""


def test_fused_bias_gradient_does_not_pass_through_a_reduced_precision_gemm() -> None:
    results = _run_script(_SCRIPT)
    assert set(results) == {"keepFalse", "keepTrue"}
    for keep, err in results.items():
        assert err <= 1e-6, (keep, err)              # a TF32 GEMM column gave about 1e-4


@requires_cuda
def test_cuda_fused_bias_gradient_matches_the_torch_route_under_high_precision() -> None:
    r"""Under ``'high'`` the fused d_bias is as accurate as the torch route's float32 reduction."""
    from vfe3.kernels.kl_decode import kl_decode_ce_rows

    saved = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        lhs, per_pos, rhs, bias, tau, targets = _operands(2048, 512, 16, torch.float32, "cuda")
        ref = _reference_bias_grad(lhs, per_pos, rhs, bias, tau, targets)
        bias_f = bias.clone().requires_grad_()
        ce = kl_decode_ce_rows(lhs, per_pos, rhs, bias_f, None, tau, targets, chunk=128, keep=False)
        (fused,) = torch.autograd.grad(ce.sum(), (bias_f,))
        bias_t = bias.clone().requires_grad_()
        logits = -torch.clamp(0.5 * (lhs @ rhs.t() + bias_t - per_pos[:, None]), min=0.0) / tau
        ce_t = torch.logsumexp(logits, -1) - logits.gather(1, targets[:, None]).squeeze(1)
        (torch_route,) = torch.autograd.grad(ce_t.sum(), (bias_t,))
    finally:
        torch.set_float32_matmul_precision(saved)
    fused_err = float((fused.double().cpu() - ref).norm() / ref.norm())
    torch_err = float((torch_route.double().cpu() - ref).norm() / ref.norm())
    assert fused_err <= 4.0 * torch_err + 1e-6, (fused_err, torch_err)
