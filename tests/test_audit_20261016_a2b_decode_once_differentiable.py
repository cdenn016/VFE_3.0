r"""Audit 2026-10-16 F49: the fused decode Functions refuse a double backward.

``KLDecodeCE.backward`` and ``ExpectedLikelihoodCE.backward`` write their pair weights with Triton, which
autograd does not track, so a ``create_graph=True`` first gradient carried a truncated graph (through the
saved operands only) and its second derivative came back finite and wrong without an error. Both are now
``once_differentiable``, so the second backward raises. The check runs the Triton interpreter on CPU tensors
in a subprocess (skipped where triton is missing), as the other fused-kernel tests do.
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
_NO_TRITON = 77                                       # subprocess exit code: triton unavailable


def _run_script(script: str) -> Any:
    env: Dict[str, str] = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), TRITON_INTERPRET="1")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
from vfe3.kernels.expected_likelihood import expected_likelihood_ce
from vfe3.kernels.kl_decode import kl_decode_ce_rows

def outcome(fn):
    try:
        fn()
    except RuntimeError as exc:
        return "once_differentiable" if "once_differentiable" in str(exc) else "raised"
    return "returned"                                  # a finite second derivative with no error

def probe(name, ce, leaves):
    # (a) a plain cotangent: the second derivative of each first gradient must not come back.
    firsts = torch.autograd.grad(ce.sum(), leaves, create_graph=True)
    for leaf_name, first, leaf in zip(("query", "tau"), firsts, leaves):
        out[f"{name}_{leaf_name}"] = outcome(lambda: torch.autograd.grad(first.sum(), leaf))
    # (b) a cotangent that requires grad, where once_differentiable attaches its error node.
    cotangent = torch.ones_like(ce, requires_grad=True)
    firsts = torch.autograd.grad(ce, leaves, cotangent, create_graph=True)
    out[f"{name}_backward"] = outcome(lambda: sum(f.sum() for f in firsts).backward())

g = torch.Generator().manual_seed(0)
S, V, K = 5, 9, 3
out = {}
for keep in (False, True):
    lhs = (0.5 + torch.rand(S, 2 * K, generator=g, dtype=torch.float64)).requires_grad_()
    rhs = 0.5 + torch.rand(V, 2 * K, generator=g, dtype=torch.float64)
    bias = torch.rand(V, generator=g, dtype=torch.float64)
    per_pos = torch.rand(S, generator=g, dtype=torch.float64)
    tau = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, V, (S,), generator=g)
    probe(f"kl_keep{keep}", kl_decode_ce_rows(lhs, per_pos, rhs, bias, None, tau, targets, chunk=4, keep=keep),
          (lhs, tau))
    # With the decode tables requiring grad, the first create_graph backward itself must succeed.
    rhs.requires_grad_()
    bias.requires_grad_()
    ce = kl_decode_ce_rows(lhs, per_pos, rhs, bias, None, tau, targets, chunk=4, keep=keep)
    out[f"kl_keep{keep}_tables_first"] = outcome(
        lambda: torch.autograd.grad(ce.sum(), (lhs, rhs, bias, tau), create_graph=True))

mu_q = torch.randn(S, K, generator=g, dtype=torch.float64).requires_grad_()
sigma_q = 0.5 + torch.rand(S, K, generator=g, dtype=torch.float64)
mu_v = torch.randn(V, K, generator=g, dtype=torch.float64)
sigma_v = 0.5 + torch.rand(V, K, generator=g, dtype=torch.float64)
tau = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
targets = torch.randint(0, V, (S,), generator=g)
probe("el", expected_likelihood_ce(mu_q, sigma_q, mu_v, sigma_v, None, tau, targets), (mu_q, tau))
print(json.dumps(out))
"""


def test_double_backward_through_the_fused_decode_raises() -> None:
    results = _run_script(_SCRIPT)
    # Before the fix the second derivatives of the KL route and the EL temperature "returned" finite wrong values
    # with no error, and the KL first create_graph backward with the tables requiring grad raised from an out= GEMM.
    expected = {}
    for name in ("kl_keepFalse", "kl_keepTrue", "el"):
        expected.update({f"{name}_query": "raised", f"{name}_tau": "raised", f"{name}_backward": "once_differentiable"})
    expected.update({"kl_keepFalse_tables_first": "returned", "kl_keepTrue_tables_first": "returned"})
    assert results == expected, results
