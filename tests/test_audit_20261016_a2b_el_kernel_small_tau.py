r"""Audit 2026-10-16 F48: the expected-likelihood Triton kernel keeps the running max and log l apart.

``el_fwd_kernel`` stored :math:`\mathrm{lse} = m + \log l`, which rounds to the ulp of the logits' maximum; at
a small temperature the logits are large and every softmax weight :math:`\exp(\ell - \mathrm{lse})` inherited
that error (the KL decode kernel already kept the two apart, docs/2026-10-12.md). The repository interpreter
test runs at tau = 0.9, where the effect is invisible. Here the float32 kernel's gradients are compared with a
float64 reference at tau = 0.1 and 0.01 and must be as accurate as the float32 torch route's within a factor 4.
Measured here (max over leaves, kernel / torch): before the fix 3.7e-6 / 1.1e-7 and 5.8e-6 / 8.1e-7 at tau = 0.01
for K = 8 and 20; after it 3.5e-7 / 1.1e-7 and 8.1e-7 / 8.1e-7. The check runs the Triton interpreter on CPU
tensors in a subprocess and is skipped where triton is missing.
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
from vfe3.kernels import _expected_likelihood_triton as tk
from vfe3.model.prior_bank import PriorBank
from vfe3.numerics import bounded_variance_from_log
assert tk.INTERPRETED

def bank(K, V, tau, seed):
    torch.manual_seed(seed)
    pb = PriorBank(V, K, 3, decode_mode="expected_likelihood_chunked", decode_chunk_size=16,
                   decode_unigram_prior=True, unigram_kappa=0.7, decode_tau=tau)
    with torch.no_grad():
        pb.sigma_log_embed.normal_(0.0, 0.5); pb.mu_embed.normal_(0.0, 1.0); pb.decode_log_scale.fill_(0.2)
    pb.set_unigram_log_prior(torch.randint(0, 50, (V,), generator=torch.Generator().manual_seed(seed)).float())
    return pb

def grads(pb, mu_q, sigma_q, targets, fused):
    mu_q = mu_q.detach().clone().requires_grad_()
    sigma_q = sigma_q.detach().clone().requires_grad_()
    names = ["mu_q", "sigma_q"] + [n for n, p in pb.named_parameters() if p.requires_grad]
    leaves = [mu_q, sigma_q] + [p for p in pb.parameters() if p.requires_grad]
    if fused:
        ce = expected_likelihood_ce(
            mu_q.reshape(-1, mu_q.shape[-1]), sigma_q.reshape(-1, sigma_q.shape[-1]), pb._decode_mu_table(),
            bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps), pb._unigram_bias(),
            pb._tau_eff(), targets.reshape(-1))
    else:
        ce = pb.decode_ce_expected_likelihood_chunked(mu_q, sigma_q, targets)
    return {n: g for n, g in zip(names, torch.autograd.grad(ce, leaves, allow_unused=True)) if g is not None}

def case(K, tau, seed=0):
    V, N = 70, 23
    g = torch.Generator().manual_seed(seed + 1)
    mu_q = torch.randn(1, N, K, generator=g)
    sigma_q = 0.2 + torch.rand(1, N, K, generator=g)
    targets = torch.randint(0, V, (1, N), generator=g)
    pb32 = bank(K, V, tau, seed)
    pb64 = bank(K, V, tau, seed).double()               # the float32 bank's values, in float64
    ref = grads(pb64, mu_q.double(), sigma_q.double(), targets, fused=False)
    out = {"K": K, "tau": tau}
    for name, fused in (("torch", False), ("kernel", True)):
        got = grads(pb32, mu_q, sigma_q, targets, fused)
        out[name] = {n: float((got[n].double() - ref[n]).norm() / ref[n].norm()) for n in ref if ref[n].norm() > 0}
    return out

print(json.dumps([case(8, 0.1), case(8, 0.01), case(20, 0.1), case(20, 0.01)]))
"""


def test_float32_kernel_gradients_match_the_torch_route_accuracy_at_small_tau(record_property: Any) -> None:
    results = _run_script(_SCRIPT)
    for r in results:
        record_property(f"el_small_tau_K{r['K']}_tau{r['tau']}", json.dumps({"torch": r["torch"], "kernel": r["kernel"]}))
        assert set(r["kernel"]) == set(r["torch"]) >= {"mu_q", "sigma_q", "mu_embed", "sigma_log_embed"}
        torch_err = max(r["torch"].values())
        kernel_err = max(r["kernel"].values())
        assert kernel_err <= 4.0 * torch_err + 1e-7, r
