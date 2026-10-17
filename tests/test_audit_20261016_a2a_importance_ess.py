r"""Reliability of the minimal agent's importance-sampled evidence (audit 2026-10-16, finding F47).

The proposal is the readout's :math:`q(X_i) = N(\xi_i^+, C_i^+)`. Under the mean field :math:`C_i^+` omits the
between-source spread, so with heterogeneous source covariances :math:`q` is narrower than the target posterior
mixture and the importance weights are heavy-tailed: at the default 64 draws single receivers are off the label-tuple
enumeration by more than a nat. The estimator also returns Kong's effective sample size fraction
:math:`(\sum_sw_s)^2/(n\sum_sw_s^2)` per receiver, which the readout reports as ``is_ess`` and the relational capture
logs per layer as ``relational_minimal_agent_is_ess``, beside ``is_evidence``.

* ``return_ess`` leaves the estimate bit for bit unchanged, and the fraction lies in :math:`[1/n, 1]`;
* a receiver with one admissible source has a Gaussian posterior, which is the mean-field :math:`q`, so its weights
  are constant: fraction 1 and the estimate equal to the exact evidence;
* on a heterogeneous coupled-head instance the receivers whose 64-draw estimate is off by more than half a nat read a
  fraction below 0.1 and the receivers within 0.05 nats read above 0.7;
* the readout returns ``is_ess`` with ``elbo=True`` and the relational capture logs its receiver mean.

Tiny float64 CPU cases only (N = 4, K = 3, H = 2, d = 2).
"""
import math

import torch

from tests.test_audit_20261012_round2_evidence import _enumerated_evidence
from tests.test_t1_t2_minimal_agent_elbo_20261012 import _instance
from vfe3.model import minimal_agent as ma

torch.set_num_threads(1)
F64 = torch.float64


def _heterogeneous(seed: int, N: int = 4, H: int = 2, causal: bool = False):
    r"""``_instance`` with source covariances scaled by :math:`e^{2z_j}`, :math:`z_j \sim N(0, 1.5^2)`, and channel
    noise lowered by :math:`e^{-2}`, so the records are informative and the sources differ in spread."""
    xi, C, S, Tm, log_noise, lp = _instance(1000 + seed, N=N, K=3, H=H, d=2, causal=causal)
    z = 1.5 * torch.randn(N, generator=torch.Generator().manual_seed(5000 + seed), dtype=F64)
    return xi, C * torch.exp(2.0 * z)[:, None, None], S, Tm, log_noise - 2.0, lp


def _evidence_inputs(xi, C, Tm, log_noise):
    r"""Utterance moments :math:`b_j = T_h\xi_j`, :math:`B_j = T_hC_jT_h^\top + n_hI` of the declared record."""
    d = Tm.shape[1]
    b = torch.einsum("hsk,nk->hns", Tm, xi)
    B = torch.einsum("hsk,nkl,htl->hnst", Tm, C, Tm) + log_noise.exp()[:, None, None, None] * torch.eye(d, dtype=F64)
    return b, B


def test_return_ess_keeps_the_estimate_and_bounds_the_fraction() -> None:
    xi, C, S, Tm, log_noise, lp = _heterogeneous(2)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    b, B = _evidence_inputs(xi, C, Tm, log_noise)
    C_q = ma._posterior_covariance(C, res.gains, res.M, None, None)
    no_record = torch.zeros(2, 4, dtype=torch.bool)
    for samples in (16, 64):
        plain = ma.minimal_agent_importance_evidence(xi, C, res.xi, C_q, S, b, B, lp, no_record, blocks=False,
                                                     samples=samples)
        estimate, ess = ma.minimal_agent_importance_evidence(xi, C, res.xi, C_q, S, b, B, lp, no_record,
                                                             blocks=False, samples=samples, return_ess=True)
        assert torch.equal(estimate, plain)
        assert ess.shape == estimate.shape and ess.dtype == F64
        assert bool(((ess >= 1.0 / samples - 1e-12) & (ess <= 1.0 + 1e-12)).all())


def test_a_gaussian_posterior_has_constant_weights() -> None:
    r"""Receiver 0 of a causal instance hears only itself, so its posterior is Gaussian and is the mean-field
    :math:`q`; every weight equals :math:`p(o)`."""
    xi, C, S, Tm, log_noise, lp = _heterogeneous(3, H=1, causal=True)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    assert abs(float(res.is_ess[0]) - 1.0) < 1e-9
    assert abs(float(res.is_evidence[0] - res.evidence[0])) < 1e-9          # one head: the evidence is exact


def test_a_low_fraction_flags_the_heavy_tailed_receivers() -> None:
    xi, C, S, Tm, log_noise, lp = _heterogeneous(2)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    error = (res.is_evidence - _enumerated_evidence(xi, C, S, Tm, log_noise, lp)).abs()   # (N,) nats
    far, near = error > 0.5, error < 0.05
    assert bool(far.any()) and bool(near.any())
    assert float(error.max()) > 1.0                                          # a single receiver off by over a nat
    assert bool((res.is_ess[far] < 0.1).all()) and bool((res.is_ess[near] > 0.7).all())
    print(f"\n[F47] |IS - enumeration| (nats) {error.numpy().round(3).tolist()}, "
          f"ESS / n {res.is_ess.numpy().round(3).tolist()}")


def test_the_readout_reports_the_fraction_only_with_the_bound() -> None:
    xi, C, S, Tm, log_noise, lp = _heterogeneous(4)
    off = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field")
    assert off.is_ess is None and off.is_evidence is None
    on = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    b, B = _evidence_inputs(xi, C, Tm, log_noise)
    C_q = ma._posterior_covariance(C, on.gains, on.M, None, None)
    _, want = ma.minimal_agent_importance_evidence(xi, C, on.xi, C_q, S, b, B, lp, torch.zeros(2, 4, dtype=torch.bool),
                                                   blocks=False, return_ess=True)
    torch.testing.assert_close(on.is_ess, want, rtol=1e-13, atol=1e-13)
    assert ma.MINIMAL_AGENT_ELBO_KEYS[-1] == "relational_minimal_agent_is_ess"
    assert math.isfinite(float(on.is_ess.mean()))
