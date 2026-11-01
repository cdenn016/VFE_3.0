r"""The fused KL-decode cross-entropy rows under the Triton interpreter against an independent formula (audit
2026-10-31 prior-bank-decode-7, fused part).

The repository's tests compared the fused rows with the torch loop and the chunked CE with the dense decode, so an
algebra error shared by the routes would pass. Here, in float64 at V = 13 with chunks of 5 (ragged: 5, 5, 3), the rows
of :func:`vfe3.kernels.kl_decode.kl_decode_ce_rows`, with the cross term kept and recomputed, must equal the
cross-entropy of the logits

.. math::
    \ell_{iv} = -\mathrm{KL}\big(\mathcal N(\mu_i, \mathrm{diag}\,\sigma_i)\,\|\,\mathcal N(\mu_v, \mathrm{diag}\,s_v)\big)
    /\tau + u_v,

written here from the moments; their gradients must pass ``torch.autograd.gradcheck`` with respect to the expanded
factors (lhs, per_pos, rhs, bias) and the temperature, and, composed with the expanded factors, they must give the
autograd gradient of that independent cross-entropy with respect to the moments. The interpreter runs in a subprocess,
so the test process never imports triton.
"""
import json
from typing import Any

from tests.test_cue_result_triton_20261025 import _run_script, _subprocess_env

_ROWS_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
from vfe3.kernels.kl_decode import kl_decode_ce_rows

V, K, S, TAU, CHUNK = 13, 4, 6, 0.1, 5
d = torch.float64
g = torch.Generator().manual_seed(1)
moments = {"mu_q": 0.8 * torch.randn(S, K, generator=g, dtype=d),
           "s_q": torch.exp(0.4 * torch.randn(S, K, generator=g, dtype=d)),
           "mu_v": 0.7 * torch.randn(V, K, generator=g, dtype=d),
           "s_v": torch.exp(0.5 * torch.randn(V, K, generator=g, dtype=d))}
u = torch.log(torch.rand(V, generator=g, dtype=d) + 0.1)            # (V,) unigram bias, a buffer
targets = torch.randint(0, V, (S,), generator=g)
targets[-1] = V - 1                                                   # a target in the ragged last chunk
tau = torch.tensor([TAU], dtype=d)

def factors(mu_q, s_q, mu_v, s_v):
    c = mu_v.mean(0, keepdim=True)                                    # (1, K) centering offset
    mc_q, mc_v = mu_q - c, mu_v - c
    lhs = torch.cat([s_q + mc_q ** 2, -2.0 * mc_q], dim=-1)          # (S, 2K)
    rhs = torch.cat([1.0 / s_v, mc_v / s_v], dim=-1)                  # (V, 2K)
    bias = (mc_v ** 2 / s_v).sum(-1) + torch.log(s_v).sum(-1)         # (V,)
    per_pos = K + torch.log(s_q).sum(-1)                              # (S,)
    return lhs, per_pos, rhs, bias

def independent_rows(mu_q, s_q, mu_v, s_v):
    diff = mu_q[:, None, :] - mu_v[None]                              # (S, V, K)
    kl = 0.5 * ((s_q[:, None] + diff ** 2) / s_v - 1.0 + torch.log(s_v) - torch.log(s_q[:, None])).sum(-1)
    logits = -kl / TAU + u                                            # (S, V)
    return torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None])[:, 0]

out = {}
reference = independent_rows(**moments)
for keep in (True, False):
    rows = kl_decode_ce_rows(*factors(**moments), u, tau, targets, chunk=CHUNK, keep=keep)
    out[f"keep_{keep}_rows"] = float((rows - reference).abs().max())

    def fused(lhs, per_pos, rhs, bias, tau_):
        return kl_decode_ce_rows(lhs, per_pos, rhs, bias, u, tau_, targets, chunk=CHUNK, keep=keep).sum()

    leaves = tuple(x.detach().clone().requires_grad_(True) for x in (*factors(**moments), tau))
    out[f"keep_{keep}_gradcheck"] = bool(torch.autograd.gradcheck(fused, leaves, eps=1e-6, atol=1e-6, rtol=1e-5))
    fused_leaves = {k: v.clone().requires_grad_(True) for k, v in moments.items()}
    kl_decode_ce_rows(*factors(**fused_leaves), u, tau, targets, chunk=CHUNK, keep=keep).sum().backward()
    oracle_leaves = {k: v.clone().requires_grad_(True) for k, v in moments.items()}
    independent_rows(**oracle_leaves).sum().backward()
    out[f"keep_{keep}_moment_grads"] = max(float((fused_leaves[k].grad - oracle_leaves[k].grad).abs().max())
                                           for k in moments)
out["scale"] = float(reference.abs().max())
print(json.dumps(out))
"""


def test_fused_kl_decode_rows_match_an_independent_cross_entropy_and_its_gradients(record_property: Any) -> None:
    report = _run_script(_ROWS_SCRIPT, _subprocess_env(TRITON_INTERPRET="1"))
    record_property("fused_rows", json.dumps(report))
    assert report["scale"] > 0.0
    for keep in (True, False):
        assert report[f"keep_{keep}_rows"] < 1e-12, report
        assert report[f"keep_{keep}_gradcheck"], report
        assert report[f"keep_{keep}_moment_grads"] < 1e-12, report
