r"""Stage 0 of the local-learner test (design 2026-10-21): tiny-dimension checks, depth-4 tree, d = 3."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from vfe3.local_learner.core import (Params, TreeSensor, assimilate, forecast, init_params, init_runs,
                                     sample_sequences, start_state)
from vfe3.local_learner.learners import (Stats, SensorCollector, accumulate_transition, m_step, node_moments,
                                         train_backprop, train_local)

torch.set_default_dtype(torch.float64)
D, d = 4, 3


@pytest.fixture
def setup():
    tree = TreeSensor(D)
    prm = init_params(d, tree.V, seed=1)
    g = torch.Generator().manual_seed(2)
    prm.A = torch.randn(1, tree.V - 1, d, generator=g)
    prm.b = 0.5 * torch.randn(1, tree.V - 1, generator=g)
    prm.G = 0.7 * torch.linalg.qr(torch.randn(d, d, generator=g))[0][None]
    prm.E = 0.5 * torch.randn(1, tree.V, d, generator=g)
    return tree, prm


def _gaussian(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(d, d, generator=g)
    return torch.randn(d, generator=g), X @ X.T + 0.3 * torch.eye(d)


def _node(m, P, a, b, s):
    Pa = P @ a
    s2 = 1 + a @ Pa
    mu_u = a @ m + b
    kappa = s * mu_u / s2.sqrt()
    lam = torch.exp(-0.5 * kappa ** 2 - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(kappa))
    return node_moments(s, m, P, Pa, s2, s2.sqrt(), mu_u, kappa, lam)


def test_forecast_rows_normalized(setup):
    tree, prm = setup
    m, P = _gaussian(3)
    y = torch.arange(tree.V)
    _, _, logp = assimilate(m.expand(1, tree.V, d), P.expand(1, tree.V, d, d), y, prm, tree)
    assert abs(float(logp.exp().sum()) - 1.0) < 1e-12


def test_single_node_probability_is_exact(setup):
    r"""E_{N(m,P)}[Phi(a^T z + b)] = Phi((a^T m + b)/sqrt(1 + a^T P a)), checked by Gauss-Hermite quadrature."""
    from scipy.stats import norm
    tree, prm = setup
    m, P = _gaussian(4)
    a, b = prm.A[0, 0], prm.b[0, 0]
    mean, var = float(a @ m + b), float(a @ P @ a)
    x, w = np.polynomial.hermite_e.hermegauss(120)
    exact = float((w * norm.cdf(mean + math.sqrt(var) * x)).sum() / math.sqrt(2 * math.pi))
    assert abs(exact - norm.cdf(mean / math.sqrt(1 + var))) < 1e-10
    assert int(tree.nodes[tree.V - 1, 0]) == 0 and float(tree.dirs[tree.V - 1, 0]) == 1.0


def test_truncated_node_moments_match_monte_carlo(setup):
    tree, prm = setup
    m, P = _gaussian(5)
    a, b, s = prm.A[0, 2], prm.b[0, 2], torch.tensor(-1.0)
    Ez, Ezz, Eu, Euz = _node(m, P, a, b, s)
    g = torch.Generator().manual_seed(6)
    z = m + torch.randn(2_000_000, d, generator=g) @ torch.linalg.cholesky(P).T
    u = z @ a + b + torch.randn(2_000_000, generator=g)
    keep = (s * u) > 0
    z, u = z[keep], u[keep]
    assert torch.allclose(Ez, z.mean(0), atol=1e-2)
    assert torch.allclose(Ezz, z.T @ z / len(z), atol=2e-2)
    assert abs(float(Eu - u.mean())) < 1e-2
    assert torch.allclose(Euz, (u[:, None] * z).mean(0), atol=2e-2)


def test_node_score_identity(setup):
    r"""d/da log Phi(kappa) equals the posterior expectation of the utility complete-data score."""
    tree, prm = setup
    m, P = _gaussian(7)
    a0, b0, s = prm.A[0, 1].clone(), prm.b[0, 1].clone(), torch.tensor(1.0)

    def logprob(a: torch.Tensor) -> float:
        return float(torch.special.log_ndtr(s * (a @ m + b0) / torch.sqrt(1 + a @ P @ a)))

    h = 1e-6
    fd = torch.tensor([(logprob(a0 + h * e) - logprob(a0 - h * e)) / (2 * h) for e in torch.eye(d)])
    Ez, Ezz, _, Euz = _node(m, P, a0, b0, s)
    assert torch.allclose(Euz - Ezz @ a0 - b0 * Ez, fd, atol=1e-7)


def _one_chunk_stats(tree, prm, streams, blind: bool) -> Stats:
    st = Stats.zeros(prm.R, d, tree.V)
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
        ab = torch.cat([p.A[0], p.b[0, :, None]], 1)
        return float((0.5 * torch.einsum("ni,nij,nj->n", ab, st.node_zz[0], ab) - (ab * st.node_uz[0]).sum(-1)
                      + 0.5 * ridge * (ab ** 2).sum(-1)).sum())

    def trans_energy(p) -> float:
        W = torch.cat([p.G[0], p.E[0].T], 1)
        Sxx = torch.zeros(d + tree.V, d + tree.V)
        Sxx[:d, :d], Sxx[:d, d:], Sxx[d:, :d], Sxx[d:, d:] = st.pp[0], st.yp[0].T, st.yp[0], torch.diag(st.cnt[0])
        Szx = torch.cat([st.zp[0], st.yz[0].T], 1)
        return float(0.5 * torch.trace(W @ Sxx @ W.T) - torch.trace(W @ Szx.T) + 0.5 * ridge * (W ** 2).sum())

    assert sensor_energy(new) <= sensor_energy(start) + 1e-12
    assert trans_energy(new) <= trans_energy(start) + 1e-12


def test_blind_statistics_leave_the_transition_fixed_and_differ_from_local(setup):
    r"""With the forecast in place of the assimilated state the regression reproduces (G, E, Q): the
    target-blind control has no token-to-state credit path. The token-conditioned statistics move it."""
    tree, prm = setup
    streams = sample_sequences(prm, tree, 64, 30, seed=10)
    start = init_params(d, tree.V, seed=11)
    blind = _one_chunk_stats(tree, start, streams, blind=True)
    local = _one_chunk_stats(tree, start, streams, blind=False)
    new = m_step(blind, start, 1e-12, 1e-12)
    assert torch.allclose(new.G, start.G, atol=1e-6)
    seen = blind.cnt[0] > 0
    assert torch.allclose(new.E[0, seen], start.E[0, seen], atol=1e-6)
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
        p.b = torch.cat([b0.reshape(1, 1), prm.b[:, 1:]], 1)
        m, P = start_state(streams[:, 0], p, tree)
        total = torch.zeros(())
        for t in range(1, 10):
            mm, Pm = forecast(m, P, streams[:, t - 1], p)
            m, P, logp = assimilate(mm, Pm, streams[:, t], p, tree)
            total = total - logp.sum()
        return total

    b0 = prm.b[0, 0].clone().requires_grad_(True)
    nll(b0).backward()
    h = 1e-6
    fd = (float(nll(prm.b[0, 0] + h)) - float(nll(prm.b[0, 0] - h))) / (2 * h)
    assert abs(float(b0.grad) - fd) < 1e-5 * max(1.0, abs(fd))


def test_stacked_runs_equal_separate_runs(setup):
    r"""Three seeds trained as one stacked run equal the three seeds trained alone (both learning rules and the
    blind control), and the stacked evaluation and particle filter return per-run values."""
    from vfe3.local_learner import evaluate_filter, particle_filter_nll
    tree, prm = setup
    streams = sample_sequences(prm, tree, 16, 25, seed=16)
    seeds = [3, 4, 5]
    for train in (lambda p: train_local(p, streams, tree, chunk=8, window=4),
                  lambda p: train_local(p, streams, tree, chunk=8, window=1, blind=True),
                  lambda p: train_backprop(p, streams, tree, chunk=8, lr=1e-2)):
        stacked = train(init_runs(d, tree.V, seeds))
        for r, s in enumerate(seeds):
            alone = train(init_params(d, tree.V, s))
            for f in ("G", "E", "Q", "A", "b"):
                assert torch.allclose(getattr(stacked.run(r), f), getattr(alone, f), atol=1e-9), (f, r)
    both = Params.stack([prm, init_params(d, tree.V, 7)])
    tot, _ = evaluate_filter(both, streams, tree)
    assert tot.shape == (2,) and abs(float(tot[0]) - float(evaluate_filter(prm, streams, tree)[0][0])) < 1e-9
    pf, _ = particle_filter_nll(both, streams, tree, 64, seed=0)
    assert pf.shape == (2,) and torch.isfinite(pf).all()


def test_cuda_filter_and_learners_match_cpu(setup):
    r"""The filter NLL, one local stepwise-EM pass and one backprop pass over stacked runs on CUDA agree with
    the CPU (float64; atomic index_add order is the only expected difference); the particle filter is finite."""
    if os.environ.get("VFE3_TEST_DEVICE", "cpu") != "cuda" or not torch.cuda.is_available():
        pytest.skip("CUDA lane only")
    from vfe3.local_learner import evaluate_filter, particle_filter_nll
    tree, prm = setup
    streams = sample_sequences(prm, tree, 32, 33, seed=14)
    gpu_tree = TreeSensor(D, "cuda")
    start = init_runs(d, tree.V, [15, 16])
    for train in (lambda p, s, t: train_local(p, s, t, chunk=8, window=4),
                  lambda p, s, t: train_backprop(p, s, t, chunk=8, lr=1e-2)):
        cpu = train(start, streams, tree)
        gpu = train(start.to("cuda"), streams.to("cuda"), gpu_tree).to("cpu")
        for f in ("G", "E", "Q", "A", "b"):
            assert torch.allclose(getattr(cpu, f), getattr(gpu, f), atol=1e-8), f
    nll_cpu, _ = evaluate_filter(prm, streams, tree)
    nll_gpu, _ = evaluate_filter(prm.to("cuda"), streams.to("cuda"), gpu_tree)
    assert torch.allclose(nll_cpu, nll_gpu, rtol=1e-10)
    pf, _ = particle_filter_nll(prm.to("cuda"), streams.to("cuda"), gpu_tree, 256, seed=0)
    assert torch.isfinite(pf).all()


def test_gamma_schedule_options(setup):
    r"""Restarting every update and a unit floor both reduce stepwise EM to per-chunk batch EM, and the
    explicit default schedule reproduces the default call."""
    tree, prm = setup
    streams = sample_sequences(prm, tree, 16, 33, seed=17)
    start = init_params(d, tree.V, seed=18)
    reset1 = train_local(start, streams, tree, chunk=8, reset_every=1)
    floor1 = train_local(start, streams, tree, chunk=8, gamma_floor=1.0)
    default = train_local(start, streams, tree, chunk=8)
    explicit = train_local(start, streams, tree, chunk=8, gamma_exponent=0.6, gamma_floor=0.0, reset_every=None)
    for f in ("G", "E", "Q", "A", "b"):
        assert torch.allclose(getattr(reset1, f), getattr(floor1, f), atol=1e-12)
        assert torch.equal(getattr(default, f), getattr(explicit, f))
    assert float((reset1.G - default.G).abs().max()) > 1e-6
    with pytest.raises(ValueError):
        train_local(start, streams, tree, chunk=8, gamma_exponent=0.0)


def test_cavity_statistics(setup):
    r"""The EP cavity of the last path node is its sequential state, so their node statistics agree there and
    differ at earlier nodes; every cavity equals removing that node's Gaussian site from the posterior in
    natural parameters and is positive definite; stacked cavity runs equal separate ones."""
    from vfe3.local_learner.learners import CavitySensorCollector
    tree, prm = setup
    m, P = _gaussian(19)
    y = torch.full((5,), 11)
    mm, Pm = m.expand(1, 5, d).clone(), P.expand(1, 5, d, d).clone()
    seq, cav = Stats.zeros(1, d, tree.V), Stats.zeros(1, d, tree.V)
    _ = assimilate(mm, Pm, y, prm, tree, collect=SensorCollector(seq))
    hook = CavitySensorCollector(cav)
    m_post, P_post, _ = assimilate(mm, Pm, y, prm, tree, collect=hook)
    sites = list(hook.sites)
    hook.finish(torch.zeros(1, 1, dtype=torch.long), m_post, P_post, prm)
    path = tree.nodes[11]
    last, first = int(path[-1]), int(path[0])
    assert torch.allclose(seq.node_zz[0, last], cav.node_zz[0, last], atol=1e-10)
    assert torch.allclose(seq.node_uz[0, last], cav.node_uz[0, last], atol=1e-10)
    assert float((seq.node_uz[0, first] - cav.node_uz[0, first]).abs().max()) > 1e-6
    Lam, eta = torch.linalg.inv(P_post[0, 0]), torch.linalg.solve(P_post[0, 0], m_post[0, 0])
    for n, s, tau, nu in sites:
        a, bb = prm.A[0, int(n[0])], prm.b[0, int(n[0])]
        Lam_c = Lam - tau[0, 0] * torch.outer(a, a)                    # site exp(-tau f^2/2 + nu f), f = a^T z + b
        eta_c = eta - (nu[0, 0] - tau[0, 0] * bb) * a
        P_c = torch.linalg.inv(Lam_c)
        assert torch.linalg.eigvalsh(P_c).min() > 0
        w = P_post[0, 0] @ a
        den = 1 - tau[0, 0] * (a @ w)
        assert torch.allclose(P_post[0, 0] + tau[0, 0] / den * torch.outer(w, w), P_c, atol=1e-9)
        m_c = m_post[0, 0] + w / den * (tau[0, 0] * (a @ m_post[0, 0] + bb) - nu[0, 0])
        assert torch.allclose(m_c, P_c @ eta_c, atol=1e-9)
    streams = sample_sequences(prm, tree, 16, 25, seed=20)
    stacked = train_local(init_runs(d, tree.V, [3, 4]), streams, tree, chunk=8, window=4, sensor_stats="cavity")
    for r, sd in enumerate([3, 4]):
        alone = train_local(init_params(d, tree.V, sd), streams, tree, chunk=8, window=4, sensor_stats="cavity")
        assert torch.allclose(stacked.run(r).A, alone.A, atol=1e-9)
    assert torch.isfinite(stacked.G).all()
