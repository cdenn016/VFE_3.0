r"""Stage 0 of the local-learner test (design 2026-10-21): tiny-dimension checks, depth-4 tree, d = 3."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vfe3.local_learner.core import TreeSensor, assimilate, forecast, init_params, sample_sequences, start_state
from vfe3.local_learner.learners import (Stats, SensorCollector, accumulate_transition, m_step, node_moments,
                                         train_local)

torch.set_default_dtype(torch.float64)
D, d = 4, 3


@pytest.fixture
def setup():
    tree = TreeSensor(D)
    prm = init_params(d, tree.V, seed=1)
    g = torch.Generator().manual_seed(2)
    prm.A = torch.randn(tree.V - 1, d, generator=g)
    prm.b = 0.5 * torch.randn(tree.V - 1, generator=g)
    prm.G = 0.7 * torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    prm.E = 0.5 * torch.randn(tree.V, d, generator=g)
    return tree, prm


def _gaussian(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(d, d, generator=g)
    return torch.randn(d, generator=g), X @ X.T + 0.3 * torch.eye(d)


def test_forecast_rows_normalized(setup):
    tree, prm = setup
    m, P = _gaussian(3)
    y = torch.arange(tree.V)
    _, _, logp = assimilate(m.expand(tree.V, d), P.expand(tree.V, d, d), y, prm, tree)
    assert abs(float(logp.exp().sum()) - 1.0) < 1e-12


def test_single_node_probability_is_exact(setup):
    tree, prm = setup
    m, P = _gaussian(4)
    a, b = prm.A[0], prm.b[0]
    mean, var = float(a @ m + b), float(a @ P @ a)
    x, w = np.polynomial.hermite_e.hermegauss(120)
    from scipy.stats import norm
    exact = float((w * norm.cdf(mean + math.sqrt(var) * x)).sum() / math.sqrt(2 * math.pi))
    y = torch.tensor([tree.V - 1])                                     # first branch is +1 for the top bit
    kappa = (mean) / math.sqrt(1 + var)
    assert abs(exact - norm.cdf(kappa)) < 1e-10
    n, s = tree.nodes[y, 0], tree.dirs[y, 0]
    assert int(n) == 0 and float(s) == 1.0


def test_truncated_node_moments_match_monte_carlo(setup):
    tree, prm = setup
    m, P = _gaussian(5)
    a, b, s = prm.A[2], prm.b[2], torch.tensor([-1.0])
    Pa = P @ a
    s2 = 1 + a @ Pa
    sig = s2.sqrt()
    mu_u = a @ m + b
    kappa = s * mu_u / sig
    lam = torch.exp(-0.5 * kappa ** 2 - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(kappa))
    Ez, Ezz, Eu, Euz = node_moments(s, m[None], P[None], Pa[None], s2[None], sig[None], mu_u[None], kappa, lam)
    g = torch.Generator().manual_seed(6)
    z = m + torch.randn(2_000_000, d, generator=g) @ torch.linalg.cholesky(P).T
    u = z @ a + b + torch.randn(2_000_000, generator=g)
    keep = (s * u) > 0
    z, u = z[keep], u[keep]
    assert torch.allclose(Ez[0], z.mean(0), atol=1e-2)
    assert torch.allclose(Ezz[0], z.T @ z / len(z), atol=2e-2)
    assert abs(float(Eu[0] - u.mean())) < 1e-2
    assert torch.allclose(Euz[0], (u[:, None] * z).mean(0), atol=2e-2)


def test_node_score_identity(setup):
    r"""d/da log Phi(kappa) equals the posterior expectation of the utility complete-data score."""
    tree, prm = setup
    m, P = _gaussian(7)
    a0, b0, s = prm.A[1].clone(), prm.b[1].clone(), torch.tensor([1.0])

    def logprob(a: torch.Tensor) -> float:
        return float(torch.special.log_ndtr(s * (a @ m + b0) / torch.sqrt(1 + a @ P @ a)))

    h = 1e-6
    fd = torch.tensor([(logprob(a0 + h * e) - logprob(a0 - h * e)) / (2 * h) for e in torch.eye(d)])
    Pa = P @ a0
    s2 = 1 + a0 @ Pa
    mu_u = a0 @ m + b0
    kappa = s * mu_u / s2.sqrt()
    lam = torch.exp(-0.5 * kappa ** 2 - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(kappa))
    Ez, Ezz, _, Euz = node_moments(s, m[None], P[None], Pa[None], s2[None], s2.sqrt()[None], mu_u[None], kappa, lam)
    score = Euz[0] - Ezz[0] @ a0 - b0 * Ez[0]
    assert torch.allclose(score, fd, atol=1e-7)


def _one_chunk_stats(tree, prm, streams, blind: bool) -> Stats:
    st = Stats.zeros(d, tree.V)
    hook = SensorCollector(st)
    with torch.no_grad():
        m, P = start_state(streams[:, 0], prm, tree)
        for t in range(1, streams.shape[1]):
            mm, Pm = forecast(m, P, streams[:, t - 1], prm)
            m_prev, P_prev = m, P
            m, P, _ = assimilate(mm, Pm, streams[:, t], prm, tree, collect=hook)
            post = (mm, Pm) if blind else (m, P)
            accumulate_transition(st, prm.G, m_prev, P_prev, [post[0]], [post[1]], [mm], [Pm], [streams[:, t - 1]])
    return st.scaled(1.0 / (streams.shape[0] * (streams.shape[1] - 1)))


def test_m_step_does_not_increase_expected_energy(setup):
    tree, prm = setup
    streams = sample_sequences(prm, tree, 64, 40, seed=8)
    start = init_params(d, tree.V, seed=9)
    st = _one_chunk_stats(tree, start, streams, blind=False)
    ridge = 1e-6
    new = m_step(st, start, ridge, 1e-4)

    def sensor_energy(p) -> float:
        ab = torch.cat([p.A, p.b[:, None]], 1)
        return float((0.5 * torch.einsum("ni,nij,nj->n", ab, st.node_zz, ab) - (ab * st.node_uz).sum(-1)
                      + 0.5 * ridge * (ab ** 2).sum(-1)).sum())

    def trans_energy(p) -> float:
        W = torch.cat([p.G, p.E.T], 1)
        Sxx = torch.zeros(d + tree.V, d + tree.V)
        Sxx[:d, :d], Sxx[:d, d:], Sxx[d:, :d], Sxx[d:, d:] = st.pp, st.yp.T, st.yp, torch.diag(st.cnt)
        Szx = torch.cat([st.zp, st.yz.T], 1)
        return float(0.5 * torch.trace(W @ Sxx @ W.T) - torch.trace(W @ Szx.T) + 0.5 * ridge * (W ** 2).sum())

    assert sensor_energy(new) <= sensor_energy(start) + 1e-12
    assert trans_energy(new) <= trans_energy(start) + 1e-12


def test_blind_statistics_leave_the_transition_fixed_and_differ_from_local(setup):
    r"""With the forecast in place of the assimilated state the regression reproduces (G, E, Q): the
    target-blind control has no token-to-state credit path. The token-conditioned statistics differ."""
    tree, prm = setup
    streams = sample_sequences(prm, tree, 64, 30, seed=10)
    start = init_params(d, tree.V, seed=11)
    blind = _one_chunk_stats(tree, start, streams, blind=True)
    local = _one_chunk_stats(tree, start, streams, blind=False)
    new = m_step(blind, start, 1e-12, 1e-12)
    assert torch.allclose(new.G, start.G, atol=1e-6)
    seen = blind.cnt > 0
    assert torch.allclose(new.E[seen], start.E[seen], atol=1e-6)
    assert torch.allclose(new.Q, start.Q, atol=1e-6)
    moved = m_step(local, start, 1e-12, 1e-12)
    assert float((moved.G - start.G).abs().max()) > 1e-2


def test_window_smoothing_runs_and_backprop_gradient_matches_finite_differences(setup):
    tree, prm = setup
    streams = sample_sequences(prm, tree, 16, 33, seed=12)
    out = train_local(init_params(d, tree.V, seed=13), streams, tree, chunk=8, window=4)
    assert all(torch.isfinite(getattr(out, f)).all() for f in ("G", "E", "Q", "A", "b"))

    def nll(b0: torch.Tensor) -> torch.Tensor:
        p = prm.clone()
        p.b = torch.cat([b0.reshape(1), prm.b[1:]])
        m, P = start_state(streams[:, 0], p, tree)
        total = torch.zeros(())
        for t in range(1, 10):
            mm, Pm = forecast(m, P, streams[:, t - 1], p)
            m, P, logp = assimilate(mm, Pm, streams[:, t], p, tree)
            total = total - logp.sum()
        return total

    b0 = prm.b[0].clone().requires_grad_(True)
    nll(b0).backward()
    h = 1e-6
    fd = (float(nll(prm.b[0] + h)) - float(nll(prm.b[0] - h))) / (2 * h)
    assert abs(float(b0.grad) - fd) < 1e-5 * max(1.0, abs(fd))


def test_cuda_filter_and_learners_match_cpu(setup):
    r"""The filter NLL, one local stepwise-EM pass, one backprop pass and the particle filter's finiteness on
    CUDA agree with the CPU (float64; atomic index_add order is the only expected difference)."""
    import os
    if os.environ.get("VFE3_TEST_DEVICE", "cpu") != "cuda" or not torch.cuda.is_available():
        pytest.skip("CUDA lane only")
    from vfe3.local_learner import evaluate_filter, particle_filter_nll, train_backprop
    tree, prm = setup
    streams = sample_sequences(prm, tree, 32, 33, seed=14)
    gpu_tree = TreeSensor(D, "cuda")
    start = init_params(d, tree.V, seed=15)
    for train in (lambda p, s, t: train_local(p, s, t, chunk=8, window=4),
                  lambda p, s, t: train_backprop(p, s, t, chunk=8, lr=1e-2)):
        cpu = train(start, streams, tree)
        gpu = train(start.to("cuda"), streams.to("cuda"), gpu_tree).to("cpu")
        for f in ("G", "E", "Q", "A", "b"):
            assert torch.allclose(getattr(cpu, f), getattr(gpu, f), atol=1e-8), f
    nll_cpu, _ = evaluate_filter(prm, streams, tree)
    nll_gpu, _ = evaluate_filter(prm.to("cuda"), streams.to("cuda"), gpu_tree)
    assert abs(nll_cpu - nll_gpu) < 1e-8 * abs(nll_cpu)
    pf, _ = particle_filter_nll(prm.to("cuda"), streams.to("cuda"), gpu_tree, 256, seed=0)
    assert math.isfinite(pf)
