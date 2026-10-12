r"""Mean-field M-step for the minimal agent's channel noise, a logged diagnostic (audit 2026-10-12, T6).

:func:`vfe3.model.minimal_agent.channel_noise_mstep` minimizes, over :math:`n\ge0`,

.. math::
    L(n) = \sum_{i,j}\beta_{ij}\Bigl[\tfrac12\log\det(V_j + nI)
        + \tfrac12\operatorname{tr}\bigl((V_j + nI)^{-1}(A_i^+ + (a_i^+ - b_j)(a_i^+ - b_j)^\top)\bigr)\Bigr],

the :math:`n`-dependent part of the negative expected log-likelihood of the collapsed records under the mean-field
posterior. Pinned here against a brute-force evaluation of :math:`L` with dense matrices (no eigendecomposition), the
one-direction closed form :math:`n^* = (e - v)^+`, stationarity by autograd, and the readout's capture.
"""
import math

import pytest
import torch

import vfe3.model.minimal_agent as ma

F64 = torch.float64


def _spd(g: torch.Generator, *shape: int, d: int, shift: float = 0.2) -> torch.Tensor:
    X = torch.randn(*shape, d, d, generator=g, dtype=F64)
    return X @ X.transpose(-1, -2) / d + shift * torch.eye(d, dtype=F64)


def _instance(seed: int, B: int = 2, N: int = 5, H: int = 2, d: int = 3, spread: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    a = spread * torch.randn(B, N, H, d, generator=g, dtype=F64)                 # (B, N, H, d)
    A = _spd(g, B, N, H, d=d)                                                   # (B, N, H, d, d)
    b = spread * torch.randn(B, H, N, d, generator=g, dtype=F64)                 # (B, H, N, d)
    V = _spd(g, B, H, N, d=d, shift=0.05)                                       # (B, H, N, d, d)
    logits = torch.randn(B, H, N, N, generator=g, dtype=F64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    weights = logits.masked_fill(~causal, -math.inf).softmax(-1)                 # (B, H, N, N)
    heard = torch.ones(B, H, N, dtype=torch.bool)
    return a, A, b, V, weights, heard


def _loss(n: torch.Tensor, a, A, b, V, weights, heard, h: int) -> torch.Tensor:
    r"""Dense :math:`L(n)` of head ``h`` (every pair formed explicitly)."""
    d = b.shape[-1]
    W = V[:, h] + n * torch.eye(d, dtype=F64)                                   # (B, N_j, d, d)
    W_inv = torch.linalg.inv(W)
    delta = a[:, :, h].unsqueeze(2) - b[:, h].unsqueeze(1)                       # (B, N_i, N_j, d)
    second = A[:, :, h].unsqueeze(2) + delta.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, N_i, N_j, d, d)
    trace = torch.einsum("bjrs,bijsr->bij", W_inv, second)
    logdet = torch.logdet(W).unsqueeze(1)                                       # (B, 1, N_j)
    beta = weights[:, h] * heard[:, h].unsqueeze(-1)
    return 0.5 * (beta * (logdet + trace)).sum()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mstep_is_the_global_minimizer_of_the_dense_objective(seed: int) -> None:
    inputs = _instance(seed, spread=1.5)
    n_star, boundary = ma.channel_noise_mstep(*inputs)
    assert n_star.shape == (2,) and n_star.dtype == F64 and boundary.dtype == torch.bool
    assert torch.equal(torch.isnan(n_star), boundary)                         # n* is reported only when interior
    for h in range(2):
        grid = torch.logspace(-6, 3, 2000, dtype=F64)
        dense = torch.stack([_loss(n, *inputs, h) for n in grid])
        best = float(dense.min())
        n_eval = torch.zeros((), dtype=F64) if bool(boundary[h]) else n_star[h]
        at_star = float(_loss(n_eval, *inputs, h))
        assert at_star <= best + 1e-10 * abs(best)                             # no grid point beats n*
        if not bool(boundary[h]):
            n = n_star[h].clone().requires_grad_(True)
            gradient, = torch.autograd.grad(_loss(n, *inputs, h), n)
            assert abs(float(gradient)) < 1e-8 * max(1.0, abs(best))           # stationary


def test_one_direction_closed_form_and_the_boundary() -> None:
    def single(a: float, A: float, b: float, V: float) -> tuple:
        t = lambda x: torch.tensor(x, dtype=F64)
        n_star, boundary = ma.channel_noise_mstep(t(a).reshape(1, 1, 1, 1), t(A).reshape(1, 1, 1, 1, 1),
                                                  t(b).reshape(1, 1, 1, 1), t(V).reshape(1, 1, 1, 1, 1),
                                                  torch.ones(1, 1, 1, dtype=F64), torch.ones(1, 1, 1, dtype=torch.bool))
        return float(n_star[0]), bool(boundary[0])
    # e = A + (a - b)^2 = 0.5 + 4 = 4.5, v = 0.3: n* = 4.2, interior.
    n_star, boundary = single(0.0, 0.5, 2.0, 0.3)
    assert n_star == pytest.approx(4.2, rel=1e-12) and not boundary
    # e - v < 0: the records are less dispersed than the key law alone explains; the minimizer is the n -> 0 bracket,
    # flagged, with no value reported (audit 2026-10-12 round 2, M2).
    n_star, boundary = single(0.0, 0.1, 0.1, 0.5)
    assert math.isnan(n_star) and boundary


def test_unheard_receivers_and_masked_pairs_do_not_enter() -> None:
    a, A, b, V, weights, heard = _instance(5)
    reference = ma.channel_noise_mstep(a, A, b, V, weights, heard)[0]
    # Receiver 0 hears no record: its row is dropped, so changing its moments changes nothing.
    heard = heard.clone()
    heard[..., 0] = False
    dropped = ma.channel_noise_mstep(a, A, b, V, weights, heard)[0]
    a2 = a.clone()
    a2[:, 0] += 50.0
    torch.testing.assert_close(ma.channel_noise_mstep(a2, A, b, V, weights, heard)[0], dropped, rtol=1e-12, atol=0,
                               equal_nan=True)
    assert not torch.allclose(dropped, reference, equal_nan=True)


@pytest.mark.parametrize("inference", ["mean_field", "exact"])
def test_readout_reports_the_mstep_at_its_own_posterior(inference: str) -> None:
    g = torch.Generator().manual_seed(9)
    Bt, N, H, d = 1, 4, 2, 2
    K = H * d
    xi = torch.randn(Bt, N, K, generator=g, dtype=F64)
    C = _spd(g, Bt, N, d=K, shift=0.3)
    S = torch.eye(K, dtype=F64).reshape(H, d, K) + 0.2 * torch.randn(H, d, K, generator=g, dtype=F64)
    T = torch.eye(K, dtype=F64).reshape(H, d, K) + 0.2 * torch.randn(H, d, K, generator=g, dtype=F64)
    log_noise = torch.tensor([-1.0, 0.5], dtype=F64)
    log_prior = torch.zeros(N, N, dtype=F64).masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -math.inf)
    result = ma.minimal_agent_readout(xi, C, S, T, log_noise, log_prior, inference, diagnostics=True,
                                      cavi_steps=2)
    if inference == "exact":
        assert torch.isnan(result.channel_noise_mstep).all()
        return
    n_star = result.channel_noise_mstep
    assert n_star.shape == (H,) and torch.isfinite(n_star).all()
    a_post, A_post = ma._posterior_query_moments(result.xi, result.query_covariance, S, result.gains, result.M)
    b = torch.einsum("hsk,...nk->...hns", T, xi)
    V = result.key_covariance.movedim(-3, -4)
    heard = torch.ones(Bt, H, N, dtype=torch.bool)
    inputs = (a_post, A_post, b, V, result.weights, heard)
    torch.testing.assert_close(n_star, ma.channel_noise_mstep(*inputs)[0], rtol=1e-12, atol=0)
    for h in range(H):
        # The M-step does not raise the free energy relative to the learned noise at the same posterior.
        assert float(_loss(n_star[h], *inputs, h)) <= float(_loss(log_noise[h].exp(), *inputs, h)) + 1e-12
    # Not the uncollapsed closed form sum beta E||S X_i - T X_j||^2 / (d sum beta).
    for h in range(H):
        delta = a_post[:, :, h].unsqueeze(2) - b[:, h].unsqueeze(1)
        second = (torch.diagonal(A_post[:, :, h], dim1=-2, dim2=-1).sum(-1).unsqueeze(2)
                  + torch.diagonal(V[:, h], dim1=-2, dim2=-1).sum(-1).unsqueeze(1) + delta.square().sum(-1))
        closed = float((result.weights[:, h] * second).sum() / (d * result.weights[:, h].sum()))
        assert abs(closed - float(n_star[h])) > 1e-3


def test_capture_and_layer_row_carry_the_mstep() -> None:
    from tests.test_relational_integration import _cfg, _tokens
    from vfe3.model.model import VFEModel

    cfg = _cfg(attention_mode="projected_predictive", predictive_readout="minimal_agent",
               minimal_agent_inference="mean_field", beta_attention_prior="causal")
    torch.manual_seed(3)
    model = VFEModel(cfg)
    ids, _ = _tokens()
    cap: dict = {}
    model.forward_beliefs(ids, capture=cap)
    value = cap["relational_capture"]["relational_minimal_agent_channel_noise_mstep"]
    boundary = cap["relational_capture"]["relational_minimal_agent_channel_noise_mstep_boundary"]
    assert value.shape == boundary.shape == (len(model.group.irrep_dims),)
    assert torch.equal(torch.isnan(value), boundary == 1.0) and bool(((boundary == 0) | (boundary == 1)).all())
    row = model._minimal_agent_diagnostics(cap["relational_capture"])
    assert row["relational_minimal_agent_channel_noise_mstep"] == pytest.approx(
        float(value.double().nanmean()), nan_ok=True)
    assert row["relational_minimal_agent_channel_noise_mstep_boundary"] == pytest.approx(float(boundary.double().mean()))
