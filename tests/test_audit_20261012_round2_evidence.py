r"""Round-2 audit of 2026-10-12, item 12 and M1: what the minimal agent's logged evidence values are.

* The evidence lower bound under ``spread_corrected`` is taken at the covariance the readout returns, checked against
  Gauss-Hermite quadrature of the declared joint.
* The ADF evidence is exact for uncoupled heads, where the exact evidence is reported, against the label-tuple
  enumeration; with coupled heads the exact evidence is NaN.
* The importance-sampled evidence converges to the label-tuple enumeration with coupled heads, stays above the bound
  in expectation, and leaves the global RNG untouched.

Tiny float64 CPU cases only.
"""
import itertools
import math

import pytest
import torch

from tests.test_t1_t2_minimal_agent_elbo_20261012 import _brute_force_elbo, _instance
from vfe3.model import minimal_agent as ma

F64 = torch.float64


def _enumerated_evidence(xi, C, S, Tm, log_noise, lp) -> torch.Tensor:
    r""":math:`\log\sum_J\prod_h\hat\pi^h_{iJ^h}N(b_J; S\xi_i, SC_iS^\top + \oplus_hB^h_{J^h})` over every label tuple
    :math:`J`, with :math:`S` the stacked query maps (every receiver hears every head)."""
    N, K = xi.shape
    H, d, _ = S.shape
    S_all = S.reshape(H * d, K)
    log_pi = torch.log_softmax(lp, dim=-1)                                    # (H, N, N)
    out = torch.empty(N, dtype=F64)
    for i in range(N):
        terms = []
        for J in itertools.product(range(N), repeat=H):
            prior = sum(float(log_pi[h, i, j]) for h, j in enumerate(J))
            if not math.isfinite(prior):
                continue
            b = torch.cat([Tm[h] @ xi[j] for h, j in enumerate(J)])
            B = torch.block_diag(*[Tm[h] @ C[j] @ Tm[h].T + log_noise[h].exp() * torch.eye(d, dtype=F64)
                                   for h, j in enumerate(J)])
            cov = S_all @ C[i] @ S_all.T + B
            terms.append(prior + float(torch.distributions.MultivariateNormal(S_all @ xi[i], cov).log_prob(b)))
        out[i] = torch.logsumexp(torch.tensor(terms, dtype=F64), dim=0)
    return out


def test_spread_corrected_elbo_is_the_bound_at_the_returned_covariance() -> None:
    r"""The returned covariance is :math:`P + \sum_hP S_h^\top R'^hS_hP = C - \sum_g[\mathsf K, G][M, -R'][\mathsf K, G]^\top`,
    so the quadrature oracle reads it through stacked gains; the bound stays below the enumerated evidence."""
    xi, C, S, Tm, log_noise, lp = _instance(40, N=4, K=3, H=2, d=1)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", spread_corrected=True, elbo=True)
    returned = res._replace(gains=torch.cat([res.gains, res.spread_gains], dim=-3),
                            M=torch.cat([res.M, -res.label_spread], dim=-3))
    torch.testing.assert_close(res.elbo, _brute_force_elbo(xi, C, S, Tm, log_noise, lp, returned),
                               rtol=1e-10, atol=1e-10)
    plain = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    assert float((res.elbo - plain.elbo).abs().max()) > 1e-6                 # the spread changes q, so the bound
    assert bool((res.elbo <= _enumerated_evidence(xi, C, S, Tm, log_noise, lp) + 1e-10).all())


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
def test_exact_evidence_only_where_the_heads_are_uncoupled(inference: str) -> None:
    xi, C, _, Tm, log_noise, lp = _instance(41, N=3, K=2, H=2, d=1)
    C = torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1))
    S = torch.eye(2, dtype=F64).reshape(2, 1, 2)                              # S_1 C S_2^T = 0
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, inference, elbo=True)
    want = _enumerated_evidence(xi, C, S, Tm, log_noise, lp)
    torch.testing.assert_close(res.evidence, want, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(res.adf_evidence, want, rtol=1e-10, atol=1e-10)
    xi, C, S, Tm, log_noise, lp = _instance(42, N=3, K=2, H=2, d=1)
    coupled = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, inference, elbo=True)
    assert bool(torch.isnan(coupled.evidence).all()) and bool(torch.isfinite(coupled.adf_evidence).all())


def test_importance_evidence_converges_to_the_enumeration_with_coupled_heads() -> None:
    r"""With coupled heads the ADF value is an approximation; the importance-sampled estimate converges to the
    enumerated evidence as the sample count grows (error about :math:`n^{-1/2}`)."""
    xi, C, S, Tm, log_noise, lp = _instance(43, N=4, K=3, H=2, d=2)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    want = _enumerated_evidence(xi, C, S, Tm, log_noise, lp)
    noise = log_noise.exp()
    b = torch.einsum("hsk,nk->hns", Tm, xi)
    B = torch.einsum("hsk,nkl,htl->hnst", Tm, C, Tm) + noise[:, None, None, None] * torch.eye(2, dtype=F64)
    C_q = ma._posterior_covariance(C, res.gains, res.M, None, None)
    no_record = torch.zeros(2, 4, dtype=torch.bool)
    errors = []
    for samples in (16, 256, 20000):
        estimate = ma.minimal_agent_importance_evidence(xi, C, res.xi, C_q, S, b, B, lp, no_record, blocks=False,
                                                        samples=samples)
        errors.append(float((estimate - want).abs().max()))
    torch.testing.assert_close(res.is_evidence, ma.minimal_agent_importance_evidence(
        xi, C, res.xi, C_q, S, b, B, lp, no_record, blocks=False), rtol=1e-13, atol=1e-13)   # 64 draws
    assert errors[-1] < 5e-3 and errors[0] > errors[1] > errors[2]
    assert float((res.adf_evidence - want).abs().max()) > 1e-4                # ADF is an approximation here
    assert bool((res.elbo <= want + 1e-10).all())
    print(f"\n[M1] max |IS - enumeration| (nats) at 16 / 256 / 20000 samples: {[round(e, 5) for e in errors]}; "
          f"ADF {float((res.adf_evidence - want).abs().max()):.5f}")


def test_importance_evidence_leaves_the_global_rng_untouched() -> None:
    xi, C, S, Tm, log_noise, lp = _instance(44, N=3, K=2, H=2, d=1)
    torch.manual_seed(123)
    state = torch.get_rng_state()
    first = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True).is_evidence
    assert torch.equal(torch.get_rng_state(), state)
    second = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True).is_evidence
    assert torch.equal(first, second)
