r"""Minimal-agent attention readout (docs/implementation/minimal-agent-plan-2026-10-06.md, Section 2.6).

The receiver conditions exactly on the noisy comparison record o_h = S_h X_i - T_h X_J + eps_h = 0 with the tied key
S_K = T (``predictive_readout='minimal_agent'``; vfe3/model/minimal_agent.py). Tiny float64 CPU cases only
(K <= 6, N <= 4, at most two heads), checked against independent references built here from explicit per-pair
Gaussian conditioning, full enumeration of the label tuples of the per-head-snapshot model, and the explicit
information-form power posterior. Covers the plan's items 1-11 (exact single head, factorization under the query
condition, the sequential ADF reference, the noise-free and full-rank limits, the mean-field variant and Proposition
ma:prop:mean_field (iii), passive gauge covariance, causality, blockwise against dense, finite-difference gradients
and tile boundaries, config/optimizer/resource/state-dict wiring, short training and the checkpointed readout) and
the consumer audit (diagnostics, posthoc refusals, the full-link refusal, the persisted metrics columns).
"""
from tests.profiles import make_relational_operators
import itertools
import logging
import math
from typing import Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows
from vfe3.gradients.relational import relational_predictive_readout, relational_score
from vfe3.model import minimal_agent as ma
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import (
    projected_noisy_query_cost, set_projected_condition_recovery, source_recovery_counter,
)
from vfe3.run_artifacts import _relational_artifact_fields
from vfe3.run_resources import parameter_breakdown, parameter_role

torch.set_num_threads(1)
F64 = torch.float64
INFERENCE = ("exact", "mean_field")
FLOOR = 1e-3


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)


# ----------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------

def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _rn(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=F64)


def _spd(g: torch.Generator, *shape: int, jitter: float = 0.5) -> torch.Tensor:       # (..., K) -> (..., K, K)
    X = _rn(g, *shape, shape[-1])
    return X @ X.transpose(-1, -2) + jitter * torch.eye(shape[-1], dtype=F64)


def _causal_prior(g: torch.Generator, N: int, H: int = 1, scale: float = 0.3) -> torch.Tensor:   # (H, N, N)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    return (scale * _rn(g, H, N, N)).masked_fill(~causal, -math.inf)


def _reference_moments(g: torch.Generator, N: int, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Reference moments from random local beliefs and random dense frames: xi = U^{-1} mu, C = U^{-1} Sigma U^{-T}."""
    U = torch.eye(K, dtype=F64) + 0.3 * _rn(g, N, K, K)
    mu = _rn(g, N, K) + 1.5                                           # nonzero means
    Sigma = _spd(g, N, K)
    U_inv = torch.linalg.inv(U)
    xi = (U_inv @ mu.unsqueeze(-1)).squeeze(-1)
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)
    return xi, 0.5 * (C + C.transpose(-1, -2))


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:              # (..., G, g, g) -> (..., K, K)
    G, d = blocks.shape[-3], blocks.shape[-1]
    out = blocks.new_zeros(*blocks.shape[:-3], G * d, G * d)
    for m in range(G):
        out[..., m * d:(m + 1) * d, m * d:(m + 1) * d] = blocks[..., m, :, :]
    return out


def _posterior_cov(C_dense: torch.Tensor, result: ma.MinimalAgentResult) -> torch.Tensor:   # (..., N, K, K)
    return C_dense - torch.einsum("...nhks,...nhst,...nhlt->...nkl", result.gains, result.M, result.gains)


def _close(got: torch.Tensor, want: torch.Tensor, tol: float) -> None:
    torch.testing.assert_close(got, want, rtol=tol, atol=tol * max(1.0, float(want.abs().max())))


def _ops(inference: str, H: int = 2, d: int = 2, seed: int = 0, **kwargs: object) -> RelationalOperators:
    r"""Float64 minimal-agent operators with generic S, T and channel noises."""
    K = H * d
    ops = make_relational_operators([d] * H, 11, predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
                              predictive_readout="minimal_agent", minimal_agent_inference=inference,
                              **kwargs).double()
    g = _gen(100 + seed)
    with torch.no_grad():
        ops.raw_score_q.copy_(torch.eye(K, dtype=F64).reshape(H, d, K) + 0.4 * _rn(g, H, d, K))
        ops.T.copy_(torch.eye(K, dtype=F64).reshape(H, d, K) + 0.4 * _rn(g, H, d, K))
        ops.channel_log_noise.copy_(torch.log(0.4 + torch.rand(H, generator=g, dtype=F64)))
    return ops


def _inputs(H: int = 2, d: int = 2, N: int = 4, seed: int = 0, batch: Tuple[int, ...] = (1,)) -> Dict:
    r"""Block frames (compact and dense), means with a nonzero token-0 mean, both families, causal per-head priors."""
    K = H * d
    g = _gen(seed)
    blocks = torch.eye(d, dtype=F64) + 0.3 * _rn(g, *batch, N, H, d, d)
    U = _block_dense(blocks)
    compact = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    dense = CanonicalFrameContext(U, torch.linalg.inv(U))
    mu = _rn(g, *batch, N, K) + 2.0
    sigma_diag = torch.rand(*batch, N, K, generator=g, dtype=F64) + 0.3
    sigma_full = _spd(g, *batch, N, K)
    return dict(U=U, compact=compact, dense=dense, mu=mu, sigma_diag=sigma_diag, sigma_full=sigma_full,
                log_prior=_causal_prior(g, N, H))


def _prior(like: torch.Tensor, full: bool, seed: int = 99) -> torch.Tensor:
    r"""A fixed local prior covariance Sigma_p in ``like``'s layout (``full``: SPD (..., N, K, K), else a positive
    diagonal (..., N, K)): the anchor of the prior-anchored floor f Sigma_p, distinct from the belief covariance."""
    g = _gen(seed)
    if not full:
        return torch.rand(like.shape, generator=g, dtype=F64) + 0.4
    return _spd(g, *like.shape[:-1])


# ----------------------------------------------------------------------------------------------
# Independent references
# ----------------------------------------------------------------------------------------------

def _logpdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    return torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).log_prob(x)


def _one_head_reference(
    xi_r: torch.Tensor, C_r: torch.Tensor,                          # receivers (N, K), (N, K, K)
    xi_s: torch.Tensor, C_s: torch.Tensor,                          # sources (J, K), (J, K, K)
    S: torch.Tensor, T_: torch.Tensor, n: float, lp: torch.Tensor,  # (d, K), (d, K), noise, (N, J)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Explicit per-pair conditioning: weights from densities, component moments, mixture moments."""
    N, J, d = xi_r.shape[0], xi_s.shape[0], S.shape[0]
    weights, means, covs = [], [], []
    for i in range(N):
        logw, comps = [], []
        for j in range(J):
            W = S @ C_r[i] @ S.T + T_ @ C_s[j] @ T_.T + n * torch.eye(d, dtype=F64)
            G = C_r[i] @ S.T @ torch.linalg.inv(W)
            m = xi_r[i] + G @ (T_ @ xi_s[j] - S @ xi_r[i])
            P = C_r[i] - G @ S @ C_r[i]
            logw.append(lp[i, j] + _logpdf(T_ @ xi_s[j], S @ xi_r[i], W))
            comps.append((m, P))
        w = torch.softmax(torch.stack(logw), dim=0)
        mean = sum(w[j] * comps[j][0] for j in range(J))
        cov = sum(w[j] * (comps[j][1] + torch.outer(comps[j][0] - mean, comps[j][0] - mean)) for j in range(J))
        weights.append(w)
        means.append(mean)
        covs.append(cov)
    return torch.stack(weights), torch.stack(means), torch.stack(covs)


def _enumerated_reference(
    xi: torch.Tensor, C: torch.Tensor, S: torch.Tensor, T_: torch.Tensor, noise: torch.Tensor, lp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exact posterior of the per-head-snapshot model by enumerating every label tuple, moment matched.

    For a tuple (j_1, ..., j_H) the stacked record has covariance S C_i S^T + diag_h(T_h C_{j_h} T_h^T + n_h I) (each
    head hears its own snapshot, so no coincidence term) and cross-covariance C_i S^T with X_i.
    """
    H, d, K = S.shape
    N = xi.shape[0]
    Ss = S.reshape(H * d, K)
    means, covs = [], []
    for i in range(N):
        logw, comps = [], []
        for labels in itertools.product(range(N), repeat=H):
            y = torch.cat([T_[h] @ xi[j] for h, j in enumerate(labels)])
            V = Ss @ C[i] @ Ss.T + torch.block_diag(*[T_[h] @ C[j] @ T_[h].T + noise[h] * torch.eye(d, dtype=F64)
                                                       for h, j in enumerate(labels)])
            G = C[i] @ Ss.T @ torch.linalg.inv(V)
            comps.append((xi[i] + G @ (y - Ss @ xi[i]), C[i] - G @ Ss @ C[i]))
            logw.append(sum(lp[h, i, j] for h, j in enumerate(labels)) + _logpdf(y, Ss @ xi[i], V))
        w = torch.softmax(torch.stack(logw), dim=0)
        mean = sum(w[k] * comps[k][0] for k in range(len(comps)))
        means.append(mean)
        covs.append(sum(w[k] * (comps[k][1] + torch.outer(comps[k][0] - mean, comps[k][0] - mean))
                        for k in range(len(comps))))
    return torch.stack(means), torch.stack(covs)


def _mean_field_reference(
    xi: torch.Tensor, C: torch.Tensor, S: torch.Tensor, T_: torch.Tensor, noise: torch.Tensor, lp: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""KL labels at the input moments and the explicit information-form power posterior."""
    H, d, K = S.shape
    N = xi.shape[0]
    eye = torch.eye(d, dtype=F64)
    labels = torch.zeros(H, N, N, dtype=F64)
    means, covs = [], []
    for i in range(N):
        precision, info = torch.linalg.inv(C[i]), torch.linalg.solve(C[i], xi[i])
        for h in range(H):
            a, A = S[h] @ xi[i], S[h] @ C[i] @ S[h].T
            costs, Bs, bs = [], [], []
            for j in range(N):
                B, b = T_[h] @ C[j] @ T_[h].T + noise[h] * eye, T_[h] @ xi[j]
                B_inv = torch.linalg.inv(B)
                costs.append(0.5 * (torch.trace(B_inv @ A) + (a - b) @ B_inv @ (a - b) - d
                                    + torch.logdet(B) - torch.logdet(A)))
                Bs.append(B_inv)
                bs.append(b)
            beta = torch.softmax(lp[h, i] - torch.stack(costs), dim=0)
            labels[h, i] = beta
            Lambda = sum(beta[j] * Bs[j] for j in range(N))
            eta = sum(beta[j] * Bs[j] @ bs[j] for j in range(N))
            precision = precision + S[h].T @ Lambda @ S[h]
            info = info + S[h].T @ eta
        P = torch.linalg.inv(precision)
        means.append(P @ info)
        covs.append(P)
    return labels, torch.stack(means), torch.stack(covs)


# ----------------------------------------------------------------------------------------------
# 1. One head against explicit per-pair Gaussian conditioning
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("K,d", [(5, 3), (6, 2), (4, 4)])
def test_single_head_matches_explicit_gaussian_conditioning(K: int, d: int) -> None:
    g = _gen(K + 10 * d)
    N = 4
    xi, C = _reference_moments(g, N, K)
    S, T_ = _rn(g, 1, d, K), _rn(g, 1, d, K)
    n = 0.7
    lp = _causal_prior(g, N)
    result = ma.minimal_agent_readout(xi, C, S, T_, torch.log(torch.tensor([n], dtype=F64)), lp, "exact",
                                      diagnostics=True)
    assert torch.equal(result.head_coupling, torch.zeros(N, 1, dtype=F64))   # one head: no coupling, ADF exact
    assert bool(((result.gain_fraction >= 0) & (result.gain_fraction < 1)).all())
    w_ref, mean_ref, cov_ref = _one_head_reference(xi, C, xi, C, S[0], T_[0], n, lp[0])
    _close(result.weights[0], w_ref, 1e-10)
    assert float(result.weights[0][~torch.isfinite(lp[0])].abs().max()) == 0.0         # masked pairs: exactly 0
    _close(result.xi, mean_ref, 1e-10)
    _close(_posterior_cov(C, result), cov_ref, 1e-10)
    # (u, Lambda_bar, R) against the explicit sums, and the cost against the noisy-query cost plus its row constant.
    sym = lambda X: 0.5 * (X + X.transpose(-1, -2))
    a, A = xi @ S[0].T, sym(S[0] @ C @ S[0].T)
    b, B = xi @ T_[0].T, sym(T_[0] @ C @ T_[0].T) + n * torch.eye(d, dtype=F64)
    post = ma.exact_record_posterior(a, A, b, B, lp[0])
    for i in range(N):
        W_inv = [torch.linalg.inv(A[i] + B[j]) for j in range(N)]
        r = [W_inv[j] @ (b[j] - a[i]) for j in range(N)]
        u = sum(w_ref[i, j] * r[j] for j in range(N))
        _close(post.u[i], u, 1e-10)
        _close(post.Lambda_bar[i], sum(w_ref[i, j] * W_inv[j] for j in range(N)), 1e-10)
        _close(post.R[i], sum(w_ref[i, j] * torch.outer(r[j], r[j]) for j in range(N)) - torch.outer(u, u), 1e-10)
    noisy = projected_noisy_query_cost(a[None, :, None], A[None, :, None], b[None, :, None], B[None, :, None])[0, 0]
    _close(post.cost, noisy + 0.5 * torch.logdet(A).unsqueeze(-1), 1e-10)


# ----------------------------------------------------------------------------------------------
# 2-3. Several heads: factorization under the query condition, the sequential reference otherwise
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["blocks", "dense"])
def test_two_orthogonal_heads_equal_the_enumerated_posterior(layout: str) -> None:
    r"""Block-local query maps on block-diagonal C: S_1 C S_2^T = 0, so ADF is the exact per-head-snapshot posterior."""
    g = _gen(21)
    N, H, d = 4, 2, 2
    K = H * d
    C_blocks = _spd(g, N, H, d)                                       # (N, 2, 2, 2) blocks C_{i,m}
    C = _block_dense(C_blocks)
    xi = _rn(g, N, K) + 1.0
    S = torch.zeros(H, d, K, dtype=F64)
    S[0, :, :d], S[1, :, d:] = _rn(g, d, d) + 2 * torch.eye(d, dtype=F64), _rn(g, d, d) + 2 * torch.eye(d, dtype=F64)
    T_ = _rn(g, H, d, K)                                              # dense sending maps: no sender condition
    noise = torch.tensor([0.6, 1.3], dtype=F64)
    lp = _causal_prior(g, N, H)
    result = ma.minimal_agent_readout(xi, C_blocks if layout == "blocks" else C, S, T_, noise.log(), lp, "exact",
                                      diagnostics=True)
    assert float(result.head_coupling.abs().max()) < 1e-12           # the query condition holds
    mean_ref, cov_ref = _enumerated_reference(xi, C, S, T_, noise, lp)
    _close(result.xi, mean_ref, 1e-10)
    _close(_posterior_cov(C, result), cov_ref, 1e-10)


def test_two_coupled_heads_equal_the_sequential_reference() -> None:
    r"""Without the query condition ADF is head 1 then head 2 with full conditioning and collapse (sources stay input)."""
    g = _gen(31)
    N, H, d, K = 4, 2, 2, 5
    xi, C = _reference_moments(g, N, K)
    S, T_ = _rn(g, H, d, K), _rn(g, H, d, K)
    noise = torch.tensor([0.8, 0.5], dtype=F64)
    lp = _causal_prior(g, N, H)
    result = ma.minimal_agent_readout(xi, C, S, T_, noise.log(), lp, "exact", diagnostics=True)
    w1, xi1, C1 = _one_head_reference(xi, C, xi, C, S[0], T_[0], float(noise[0]), lp[0])
    w2, xi2, C2 = _one_head_reference(xi1, C1, xi, C, S[1], T_[1], float(noise[1]), lp[1])
    _close(result.weights, torch.stack((w1, w2)), 1e-10)
    _close(result.xi, xi2, 1e-10)
    _close(_posterior_cov(C, result), C2, 1e-10)
    mean_ex, cov_ex = _enumerated_reference(xi, C, S, T_, noise, lp)
    gap_mean = float((result.xi - mean_ex).abs().max() / mean_ex.abs().max())
    gap_cov = float((_posterior_cov(C, result) - cov_ex).abs().max() / cov_ex.abs().max())
    print(f"\n[minimal agent] coupled heads: ADF vs enumerated exact posterior, relative max gap "
          f"mean {gap_mean:.3e}, covariance {gap_cov:.3e}; head coupling {float(result.head_coupling.max()):.3f}")
    assert math.isfinite(gap_mean) and gap_mean > 1e-8              # the approximation is visible here
    assert float(result.head_coupling.min()) > 0.0


# ----------------------------------------------------------------------------------------------
# 4. Limits: noise-free point-mass senders (item c) and full rank (item d)
# ----------------------------------------------------------------------------------------------

def test_noise_free_point_mass_limit_is_the_minimal_change_projection() -> None:
    g = _gen(41)
    N, K, d = 4, 5, 2
    xi, C = _reference_moments(g, N, K)
    S, T_ = _rn(g, d, K), _rn(g, d, K)
    lp = _causal_prior(g, N)[0]
    eps = 1e-10
    a, A, b = xi @ S.T, S @ C @ S.T, xi @ T_.T
    B = eps * (T_ @ T_.T + torch.eye(d, dtype=F64)).expand(N, d, d)  # C_j = eps I and n = eps
    post = ma.exact_record_posterior(a, A, b, B, lp)
    Kmat = C @ S.T                                                    # (N, K, d)
    xi_plus = xi + (Kmat @ post.u.unsqueeze(-1)).squeeze(-1)
    C_plus = C - Kmat @ (post.Lambda_bar - post.R) @ Kmat.transpose(-1, -2)
    eye = torch.eye(K, dtype=F64)
    for i in range(N):
        R_Q = C[i] @ S.T @ torch.linalg.inv(S @ C[i] @ S.T)          # C^{-1}-minimal right inverse of S
        Pi = R_Q @ S
        for j in range(N):                                           # every component gain G_ij -> R_Q
            _close(Kmat[i] @ torch.linalg.inv(A[i] + B[j]), R_Q, 1e-7)
        _close(Kmat[i] @ post.Lambda_bar[i], R_Q, 1e-7)             # so does the label-averaged gain
        w0 = torch.softmax(lp[i] + torch.stack([_logpdf(b[j], a[i], A[i]) for j in range(N)]), dim=0)
        z_bar = w0 @ b
        spread = sum(w0[j] * torch.outer(b[j] - z_bar, b[j] - z_bar) for j in range(N))
        _close(post.weights[i], w0, 1e-7)
        _close(xi_plus[i], (eye - Pi) @ xi[i] + R_Q @ z_bar, 1e-7)
        _close(S @ xi_plus[i], z_bar, 1e-7)                         # the query now reports the heard content
        _close((eye - Pi) @ xi_plus[i], (eye - Pi) @ xi[i], 1e-7)  # the unprobed component is kept
        _close(C_plus[i], (eye - Pi) @ C[i] + R_Q @ spread @ R_Q.T, 1e-6)


def test_full_rank_head_gives_the_linked_weights_and_means() -> None:
    g = _gen(43)
    N, K = 4, 4
    xi, C = _reference_moments(g, N, K)
    S = _rn(g, 1, K, K) + 2.0 * torch.eye(K, dtype=F64)
    T_ = _rn(g, 1, K, K) + 2.0 * torch.eye(K, dtype=F64)
    n = 0.6
    lp = _causal_prior(g, N)
    result = ma.minimal_agent_readout(xi, C, S, T_, torch.log(torch.tensor([n], dtype=F64)), lp, "exact")
    S_inv = torch.linalg.inv(S[0])
    Theta = S_inv @ T_[0]                                             # the comparison link of Lemma ma:lem:link (b)
    for i in range(N):
        tilde = [Theta @ C[j] @ Theta.T + n * S_inv @ S_inv.T for j in range(N)]
        w = torch.softmax(lp[0, i] + torch.stack([_logpdf(Theta @ xi[j], xi[i], C[i] + tilde[j]) for j in range(N)]),
                          dim=0)
        m = [xi[i] + C[i] @ torch.linalg.solve(C[i] + tilde[j], Theta @ xi[j] - xi[i]) for j in range(N)]
        _close(result.weights[0, i], w, 1e-10)
        _close(result.xi[i], sum(w[j] * m[j] for j in range(N)), 1e-10)


# ----------------------------------------------------------------------------------------------
# 5. Mean field: explicit power posterior; Proposition ma:prop:mean_field (iii) on the assembly
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["dense", "blocks"])
def test_mean_field_equals_the_explicit_power_posterior(layout: str) -> None:
    g = _gen(51)
    N, H, d = 4, 2, 2
    K = H * d
    if layout == "dense":
        xi, C = _reference_moments(g, N, K)
        C_in = C
    else:
        C_in = _spd(g, N, H, d)
        C = _block_dense(C_in)
        xi = _rn(g, N, K) + 1.0
    S, T_ = _rn(g, H, d, K), _rn(g, H, d, K)                          # coupled heads
    noise = torch.tensor([0.9, 0.4], dtype=F64)
    lp = _causal_prior(g, N, H)
    result = ma.minimal_agent_readout(xi, C_in, S, T_, noise.log(), lp, "mean_field")
    labels, mean_ref, cov_ref = _mean_field_reference(xi, C, S, T_, noise, lp)
    _close(result.weights, labels, 1e-10)
    _close(result.xi, mean_ref, 1e-10)
    _close(_posterior_cov(C, result), cov_ref, 1e-10)


def test_power_posterior_at_the_exact_weights_has_the_exact_mean_and_the_spread_free_covariance() -> None:
    r"""One head, equal B_j: the power posterior at beta = w (not the KL labels) keeps the exact mean and drops G Cov_w G^T."""
    g = _gen(53)
    N, K, d = 4, 5, 3
    C0 = _spd(g, K)
    C = C0.expand(N, K, K).clone()                                   # every source shares C_j, so B_j = B
    xi = _rn(g, N, K) + 1.0
    S, T_ = _rn(g, 1, d, K), _rn(g, 1, d, K)
    n = 0.8
    lp = _causal_prior(g, N)
    a, A = xi @ S[0].T, S[0] @ C @ S[0].T
    b, B = xi @ T_[0].T, T_[0] @ C @ T_[0].T + n * torch.eye(d, dtype=F64)
    exact = ma.exact_record_posterior(a, A, b, B, lp[0])
    X0 = ma.gain_basis(C, S, blocks=False)                            # (N, 1, K, d)
    Kmat = X0[:, 0]                                                   # (N, K, d)

    def power_posterior_at_exact_weights(h: int, a_h: torch.Tensor, A_h: torch.Tensor):
        Lambda, eta, _ = ma.label_statistics(exact.weights, b, B)
        u, M, _ = ma.information_step(a_h, A_h, Lambda, eta)
        return u, M

    xi_mf, gains, M, _ = ma.sequential_head_assembly(xi, X0, S, power_posterior_at_exact_weights)
    xi_exact = xi + (Kmat @ exact.u.unsqueeze(-1)).squeeze(-1)
    C_exact = C - Kmat @ (exact.Lambda_bar - exact.R) @ Kmat.transpose(-1, -2)
    spread = Kmat @ exact.R @ Kmat.transpose(-1, -2)                  # G Cov_w(T xi) G^T
    _close(xi_mf, xi_exact, 1e-10)
    _close(C - gains[:, 0] @ M[:, 0] @ gains[:, 0].transpose(-1, -2), C_exact - spread, 1e-10)
    assert float(spread.abs().max()) > 1e-3                          # the spread is not trivially zero here
    kl_labels = ma.mean_field_record_posterior(a, A, b, B, lp[0]).weights
    assert float((kl_labels - exact.weights).abs().max()) > 1e-3     # the KL labels are not the exact weights


def test_mean_field_gain_fraction_is_evaluated_at_the_input_moments() -> None:
    r"""Every head's mean-field gain fraction is tr(A0 M0)/d at the input moments, A0 = S_h C S_h^T and
    M0 = (A0 + Lambda^{-1})^{-1} with Lambda from the KL labels, so reversing the head order permutes the fractions
    (and leaves the fused posterior unchanged). The exact path's fraction reads the ADF moments and does move."""
    g = _gen(57)
    N, H, d = 4, 2, 2
    K = H * d
    xi, C = _reference_moments(g, N, K)
    S, T_ = _rn(g, H, d, K), _rn(g, H, d, K)                          # coupled heads
    noise = torch.tensor([0.9, 0.4], dtype=F64)
    lp = _causal_prior(g, N, H)
    result = ma.minimal_agent_readout(xi, C, S, T_, noise.log(), lp, "mean_field", diagnostics=True)
    labels, _, _ = _mean_field_reference(xi, C, S, T_, noise, lp)
    eye = torch.eye(d, dtype=F64)
    for h in range(H):
        A0 = S[h] @ C @ S[h].T                                        # (N, d, d)
        Lambda = sum(labels[h, :, j, None, None] * torch.linalg.inv(T_[h] @ C[j] @ T_[h].T + noise[h] * eye)
                     for j in range(N))                               # (N, d, d)
        M0 = torch.linalg.inv(A0 + torch.linalg.inv(Lambda))
        _close(result.gain_fraction[:, h], torch.diagonal(A0 @ M0, dim1=-2, dim2=-1).sum(-1) / d, 1e-10)
    order = [1, 0]
    swapped = ma.minimal_agent_readout(xi, C, S[order], T_[order], noise.log()[order], lp[order], "mean_field",
                                       diagnostics=True)
    _close(swapped.gain_fraction, result.gain_fraction[:, order], 1e-12)
    _close(swapped.xi, result.xi, 1e-10)
    _close(_posterior_cov(C, swapped), _posterior_cov(C, result), 1e-10)
    exact = ma.minimal_agent_readout(xi, C, S, T_, noise.log(), lp, "exact", diagnostics=True)
    exact_swapped = ma.minimal_agent_readout(xi, C, S[order], T_[order], noise.log()[order], lp[order], "exact",
                                             diagnostics=True)
    assert ma.minimal_agent_readout(xi, C, S, T_, noise.log(), lp, "mean_field").gain_fraction is None   # on request
    assert float((exact_swapped.gain_fraction - exact.gain_fraction[:, order]).abs().max()) > 1e-6


# ----------------------------------------------------------------------------------------------
# 6-8. Operator level: passive gauge covariance, causality, blockwise against dense
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("layout", ["dense", "compact"])
def test_passive_gauge_action_maps_outputs_covariantly(inference: str, layout: str) -> None:
    r"""mu -> g mu, Sigma -> g Sigma g^T, U -> g U and the prior Sigma_p -> g Sigma_p g^T give mu^+ -> g mu^+ and
    Sigma^+ -> g Sigma^+ g^T (the floor f Sigma_p moves with the prior)."""
    ops = _ops(inference, seed=61)
    data = _inputs(seed=62)
    N, K = data["mu"].shape[-2:]
    g = _gen(63)
    if layout == "dense":
        G = torch.eye(K, dtype=F64) + 0.4 * _rn(g, 1, N, K, K)
        frame = data["dense"]
        moved = CanonicalFrameContext(G @ data["U"], torch.linalg.inv(G @ data["U"]))
    else:
        g_blocks = torch.eye(2, dtype=F64) + 0.4 * _rn(g, 1, N, 2, 2, 2)
        G = _block_dense(g_blocks)
        frame = data["compact"]
        moved_blocks = g_blocks @ frame.forward
        moved = CanonicalFrameContext(moved_blocks, torch.linalg.inv(moved_blocks), block_diagonal=True)
    mu, sigma, lp = data["mu"], data["sigma_full"], data["log_prior"]
    prior = _prior(sigma, True)
    mean, cov = ops.minimal_agent_update(mu, sigma, frame, lp, return_covariance=True, floor_anchor=prior)
    mean_g, cov_g = ops.minimal_agent_update((G @ mu.unsqueeze(-1)).squeeze(-1), G @ sigma @ G.transpose(-1, -2),
                                             moved, lp, return_covariance=True,
                                             floor_anchor=G @ prior @ G.transpose(-1, -2))
    _close(mean_g, (G @ mean.unsqueeze(-1)).squeeze(-1), 1e-9)
    _close(cov_g, G @ cov @ G.transpose(-1, -2), 1e-9)


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_future_tokens_do_not_reach_earlier_receivers(inference: str, family: str) -> None:
    ops = _ops(inference, seed=71)
    data = _inputs(seed=72)
    N = data["mu"].shape[-2]
    sigma = data["sigma_diag"] if family == "diagonal" else data["sigma_full"]
    prior = _prior(sigma, family == "full")
    base = ops.minimal_agent_update(data["mu"], sigma, data["compact"], data["log_prior"], return_covariance=True,
                                    floor_anchor=prior)
    mu2, sigma2 = data["mu"].clone(), sigma.clone()
    mu2[0, N - 1] += 3.0
    sigma2[0, N - 1] *= 2.5
    moved = ops.minimal_agent_update(mu2, sigma2, data["compact"], data["log_prior"], return_covariance=True,
                                     floor_anchor=prior)
    for got, want in zip(moved, base):
        torch.testing.assert_close(got[0, :N - 1], want[0, :N - 1], rtol=1e-12, atol=1e-12)
        assert not torch.allclose(got[0, N - 1], want[0, N - 1])     # the last receiver does see its own change


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("batch", [(1,), (2,)])
def test_blockwise_equals_dense(inference: str, batch: Tuple[int, ...]) -> None:
    r"""Diagonal beliefs on block frames (C blocks) against the same beliefs as dense C (dense frame, or full family)."""
    ops = _ops(inference, H=2, d=3, seed=81)
    data = _inputs(H=2, d=3, seed=82, batch=batch)
    mu, sigma, lp = data["mu"], data["sigma_diag"], data["log_prior"]
    calls: List[int] = []
    original = ma.gain_basis

    def recording(C: torch.Tensor, S: torch.Tensor, blocks: bool) -> torch.Tensor:
        calls.append(int(blocks))
        return original(C, S, blocks)

    ma.gain_basis = recording
    try:
        prior = _prior(sigma, False)
        blockwise = ops.minimal_agent_update(mu, sigma, data["compact"], lp, return_covariance=True,
                                             floor_anchor=prior)
        dense_frame = ops.minimal_agent_update(mu, sigma, data["dense"], lp, return_covariance=True,
                                               floor_anchor=prior)
        full_family = ops.minimal_agent_update(mu, torch.diag_embed(sigma), data["compact"], lp,
                                               return_covariance=True, floor_anchor=torch.diag_embed(prior))
    finally:
        ma.gain_basis = original
    assert calls == [1, 1, 0, 0, 0, 0]                                # the block route ran first, dense twice
    for other in (dense_frame, full_family):
        _close(other[0], blockwise[0], 1e-10)
    _close(dense_frame[1], blockwise[1], 1e-10)
    _close(torch.diagonal(full_family[1], dim1=-2, dim2=-1), blockwise[1], 1e-10)


# ----------------------------------------------------------------------------------------------
# 9. Finite-difference gradients and tile boundaries
# ----------------------------------------------------------------------------------------------

class _Readout(torch.nn.Module):
    r"""A scalar of the minimal-agent mean and covariance."""

    def __init__(self, ops: RelationalOperators, weights: List[torch.Tensor]) -> None:
        super().__init__()
        self.ops = ops
        self.weights = weights

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, frame: CanonicalFrameContext,
                log_prior: torch.Tensor) -> torch.Tensor:
        mean, cov = self.ops.minimal_agent_update(mu, sigma, frame, log_prior, return_covariance=True,
                                                  floor_anchor=_prior(sigma, sigma.dim() == mu.dim() + 1))
        return (self.weights[0] * mean).sum() + (self.weights[1] * cov).sum()


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_finite_difference_gradients(inference: str, family: str) -> None:
    ops = _ops(inference, seed=91)
    data = _inputs(N=3, seed=92)
    sigma0 = data["sigma_diag"] if family == "diagonal" else data["sigma_full"]
    g = _gen(93)
    wrap = _Readout(ops, [_rn(g, *data["mu"].shape), _rn(g, *sigma0.shape)])
    names = ("ops.raw_score_q", "ops.T", "ops.channel_log_noise")
    masked = ~torch.isfinite(data["log_prior"])                       # the causal mask stays -inf

    def scalar(S: torch.Tensor, T_: torch.Tensor, log_noise: torch.Tensor, mu: torch.Tensor,
               sigma: torch.Tensor, prior_entries: torch.Tensor) -> torch.Tensor:
        sigma = sigma if family == "diagonal" else 0.5 * (sigma + sigma.transpose(-1, -2))
        log_prior = prior_entries.masked_fill(masked, -math.inf)     # a learnable prior: its finite entries vary
        return torch.func.functional_call(wrap, dict(zip(names, (S, T_, log_noise))),
                                          (mu, sigma, data["compact"], log_prior))

    params = dict(ops.named_parameters())
    inputs = tuple(x.detach().clone().requires_grad_(True) for x in (
        params["raw_score_q"], params["T"], params["channel_log_noise"], data["mu"], sigma0,
        data["log_prior"].masked_fill(masked, 0.0)))
    assert torch.autograd.gradcheck(scalar, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("inference", ["exact"])
def test_tile_boundaries_give_identical_values_and_gradients(inference: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""One pair per tile (checkpointed, online log-sum-exp over source tiles) against the single default tile."""
    ops = _ops(inference, seed=95)
    data = _inputs(N=4, seed=96)

    def run() -> List[torch.Tensor]:
        ops.zero_grad(set_to_none=True)
        mu = data["mu"].clone().requires_grad_(True)
        sigma = data["sigma_diag"].clone().requires_grad_(True)
        mean, cov = ops.minimal_agent_update(mu, sigma, data["compact"], data["log_prior"], return_covariance=True,
                                             floor_anchor=_prior(sigma, False))
        (mean.square().sum() + cov.sum()).backward()
        return [mean.detach(), cov.detach(), mu.grad, sigma.grad] + [p.grad for p in ops.parameters()]

    default = run()
    calls: List[int] = []
    real = ma.checkpoint
    monkeypatch.setattr(ma, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(ma, "PAIR_TILE_BYTES", 1)
    tiny = run()
    assert len(calls) == 2 * 10              # two heads, N(N+1)/2 = 10 causal one-pair tiles each (6 masked: skipped)
    for a, b in zip(tiny, default):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)


def test_fully_masked_tiles_are_skipped_and_change_nothing_on_the_support(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A tile the prior masks everywhere is not factored: the statistics equal the single-tile values, the costs agree
    on the prior's support and read 0 on the skipped pairs, and only tiles with a finite prior entry are factored."""
    g = _gen(97)
    N, d = 5, 2
    a, b = _rn(g, 2, N, d), _rn(g, 2, N, d)
    A, B = _spd(g, 2, N, d), _spd(g, 2, N, d)
    lp = _causal_prior(g, N)[0]
    default = ma.exact_record_posterior(a, A, b, B, lp)
    factored: List[bool] = []
    original = ma._exact_tile
    monkeypatch.setattr(ma, "_exact_tile", lambda *args: factored.append(bool(torch.isfinite(args[-1]).any()))
                        or original(*args))
    monkeypatch.setattr(ma, "PAIR_TILE_BYTES", 1)
    tiny = ma.exact_record_posterior(a, A, b, B, lp)
    assert len(factored) == N * (N + 1) // 2 and all(factored)                 # the 10 masked one-pair tiles: skipped
    masked = ~torch.isfinite(lp)                                                 # (N, N) causal upper triangle
    for name in ("weights", "u", "Lambda_bar", "R"):
        torch.testing.assert_close(getattr(tiny, name), getattr(default, name), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(tiny.cost[..., ~masked], default.cost[..., ~masked], rtol=1e-12, atol=1e-12)
    assert torch.equal(tiny.cost[..., masked], torch.zeros(2, N * (N - 1) // 2, dtype=F64))
    assert float(tiny.weights[..., masked].abs().max()) == 0.0 and not bool(tiny.failed)


def test_failed_factorization_raises_without_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A record covariance that is not positive definite raises after one host check; nothing is repaired."""
    g = _gen(99)
    N, d = 3, 2
    a, b = _rn(g, N, d), _rn(g, N, d)
    A = -torch.eye(d, dtype=F64).expand(N, d, d)                    # W_ij = A_i + B_j = -0.5 I
    B = 0.5 * torch.eye(d, dtype=F64).expand(N, d, d)
    with pytest.raises(ValueError, match="positive definite"):
        ma.exact_record_posterior(a, A, b, B)
    assert bool(ma.exact_record_posterior(a, A, b, B, check=False).failed)
    syncs: List[int] = []
    original = torch.Tensor.__bool__
    monkeypatch.setattr(torch.Tensor, "__bool__", lambda t: syncs.append(1) or original(t))
    ma.exact_record_posterior(a, A.abs(), b, B, _causal_prior(g, N)[0])   # healthy: exactly one host check
    assert len(syncs) == 1


# ----------------------------------------------------------------------------------------------
# 9b. Float32: the float64 recomputation of a failed layer call, the operating range, the certificate
# ----------------------------------------------------------------------------------------------

_N32, _H32, _D32 = 32, 4, 8                                          # K = 32: large enough for float32 cancellation


def _orthogonal(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.linalg.qr(_rn(g, *shape))[0]


def _k32_case(kind: str) -> Tuple[CanonicalFrameContext, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor]:
    r"""K = 32 (H = 4, d = 8), N = 32, block frames, diagonal beliefs, causal prior; float64 masters.

    ``"coupled"`` is the review's case (frames and maps I + 0.3 randn; max head coupling 0.91, max cond C_i 8.7e6):
    in float32 the second head's ADF query covariance loses positive definiteness. ``"benign"`` (frames of condition
    10, variances log-uniform over a factor 1e2, maps I + 0.1 randn; max cond C_i 3.2e3) and ``"ill_conditioned"``
    (frames of condition 100; max cond C_i 2.8e5) factor in float32; the latter fails the query certificate.
    """
    N, H, d = _N32, _H32, _D32
    K = H * d
    if kind == "coupled":
        g = _gen(7)
        blocks = torch.eye(d, dtype=F64) + 0.3 * _rn(g, 1, N, H, d, d)
        mu = _rn(g, 1, N, K) + 1.0
        sigma = torch.rand(1, N, K, generator=g, dtype=F64) + 0.3
        perturb = 0.3
    else:
        g = _gen(5)
        exponent = 1 if kind == "benign" else 2
        singular = torch.logspace(-exponent / 2, exponent / 2, d, dtype=F64)
        blocks = (_orthogonal(g, 1, N, H, d, d) @ torch.diag_embed(singular.expand(1, N, H, d))
                  @ _orthogonal(g, 1, N, H, d, d).transpose(-1, -2))
        mu = _rn(g, 1, N, K) + 1.0
        sigma = torch.pow(10.0, (torch.rand(1, N, K, generator=g, dtype=F64) - 0.5) * 2)
        perturb = 0.1
    S = torch.eye(K, dtype=F64).reshape(H, d, K) + perturb * _rn(g, H, d, K)
    T_ = torch.eye(K, dtype=F64).reshape(H, d, K) + perturb * _rn(g, H, d, K)
    frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    lp = torch.zeros(N, N, dtype=F64).masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -math.inf)
    return frame, mu, sigma, S, T_, lp


def _k32_operators(inference: str, S: torch.Tensor, T_: torch.Tensor, dtype: torch.dtype) -> RelationalOperators:
    ops = make_relational_operators([_D32] * _H32, 3, predictive_covariance="moment_matched", predictive_covariance_floor=1e-4,
                              predictive_readout="minimal_agent", minimal_agent_inference=inference).to(dtype)
    with torch.no_grad():
        ops.raw_score_q.copy_(S.to(dtype))
        ops.T.copy_(T_.to(dtype))
        ops.channel_log_noise.zero_()                                  # n_h = 1
    return ops


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.detach().double() - want).abs().max() / want.abs().max())


@pytest.mark.parametrize("inference", INFERENCE)
def test_float32_failure_is_recomputed_in_float64(inference: str) -> None:
    r"""A float32 layer call whose ADF query covariance loses positive definiteness is recomputed whole in float64.

    The result and every gradient equal, bitwise, the float64 readout of the float32 inputs and parameters cast back
    to float32. The recovery counter grows by every query and key entry (2 B N H), and the result lies within 1e-4
    of the float64 readout of the unrounded inputs (measured: 3e-6 to 8e-6, from the rounding of the inputs)."""
    frame, mu, sigma, S, T_, lp = _k32_case("coupled")
    ops32 = _k32_operators(inference, S, T_, torch.float32)
    frame32 = CanonicalFrameContext(frame.forward.float(), frame.inverse.float(), block_diagonal=True)
    maps32 = (ops32.raw_score_q.detach(), ops32.T.detach(), ops32.channel_log_noise.detach())
    with torch.no_grad():                                              # the float32 failure is real
        xi32 = ops32.canonical_mean(mu.float(), frame32)
        C32 = ops32._reference_covariance(sigma.float(), frame32, torch.float32, diagonal=True)
        assert bool(ma.minimal_agent_readout(xi32, C32, *maps32, lp.float(), inference, check=False).failed)
        with pytest.raises(ValueError, match="ADF query covariance.*float32"):
            ma.minimal_agent_readout(xi32, C32, *maps32, lp.float(), inference)
    mu32 = mu.float().requires_grad_(True)
    sigma32 = sigma.float().requires_grad_(True)
    before = int(source_recovery_counter(mu.device))
    capture: dict = {}
    mean32, cov32 = ops32.minimal_agent_update(mu32, sigma32, frame32, lp.float(), return_covariance=True,
                                               capture=capture, floor_anchor=sigma.float())
    assert int(source_recovery_counter(mu.device)) - before == 2 * _N32 * _H32
    assert mean32.dtype == cov32.dtype == capture["beta_pre"].dtype == torch.float32
    ops64 = _k32_operators(inference, maps32[0].double(), maps32[1].double(), F64)
    mu64 = mu32.detach().double().requires_grad_(True)
    sigma64 = sigma32.detach().double().requires_grad_(True)
    mean64, cov64 = ops64.minimal_agent_update(mu64, sigma64, frame32, lp, return_covariance=True,
                                               floor_anchor=sigma.float().double())
    assert torch.equal(mean32, mean64.float()) and torch.equal(cov32, cov64.float())
    g = _gen(8)
    w_mean, w_cov = _rn(g, *mean32.shape).float(), _rn(g, *cov32.shape).float()
    ((w_mean * mean32).sum() + (w_cov * cov32).sum()).backward()
    ((w_mean * mean64.float()).sum() + (w_cov * cov64.float()).sum()).backward()
    assert torch.equal(mu32.grad, mu64.grad.float()) and torch.equal(sigma32.grad, sigma64.grad.float())
    for p32, p64 in zip(ops32.parameters(), ops64.parameters()):
        assert torch.equal(p32.grad, p64.grad.float())
    with torch.no_grad():
        mean_ref, cov_ref = _k32_operators(inference, S, T_, F64).minimal_agent_update(
            mu, sigma, frame, lp, return_covariance=True, floor_anchor=sigma)
    gap_mean, gap_cov = _relative(mean32, mean_ref), _relative(cov32, cov_ref)
    print(f"\n[minimal agent] float32 failure recomputed in float64 ({inference}): relative gap to the unrounded "
          f"float64 readout, mean {gap_mean:.2e}, covariance {gap_cov:.2e}")
    assert gap_mean < 1e-4 and gap_cov < 1e-4


@pytest.mark.parametrize("inference", INFERENCE)
def test_float32_operating_range_and_the_query_certificate(inference: str) -> None:
    r"""Two K = 32 instances without a float32 failure. Certificate off, neither is recomputed and both lie within
    1e-4 of the float64 readout with positive variances (measured: about 1e-6 benign, 1e-5 ill-conditioned). Under
    ``projected_cov_condition_recovery`` the benign one still runs in float32. The ill-conditioned one fails the
    query certificate and is recomputed in float64, bitwise equal to the float64 readout of its float32 inputs."""
    for kind, trips in (("benign", False), ("ill_conditioned", True)):
        frame, mu, sigma, S, T_, lp = _k32_case(kind)
        ops32 = _k32_operators(inference, S, T_, torch.float32)
        frame32 = CanonicalFrameContext(frame.forward.float(), frame.inverse.float(), block_diagonal=True)
        with torch.no_grad():
            mean_ref, cov_ref = _k32_operators(inference, S, T_, F64).minimal_agent_update(
                mu, sigma, frame, lp, return_covariance=True, floor_anchor=sigma)
            rounded = _k32_operators(inference, ops32.raw_score_q.double(), ops32.T.double(), F64).minimal_agent_update(
                mu.float().double(), sigma.float().double(), frame32, lp, return_covariance=True,
                floor_anchor=sigma.float().double())
        for recovery in (False, True):
            previous = set_projected_condition_recovery(recovery)
            try:
                before = int(source_recovery_counter(mu.device))
                with torch.no_grad():
                    mean32, cov32 = ops32.minimal_agent_update(mu.float(), sigma.float(), frame32, lp.float(),
                                                               return_covariance=True, floor_anchor=sigma.float())
                recovered = int(source_recovery_counter(mu.device)) - before
            finally:
                set_projected_condition_recovery(previous)
            assert recovered == (2 * _N32 * _H32 if recovery and trips else 0), (kind, recovery)
            gap_mean, gap_cov = _relative(mean32, mean_ref), _relative(cov32, cov_ref)
            print(f"\n[minimal agent] float32 {kind} ({inference}, certificate {'on' if recovery else 'off'}, "
                  f"{'float64 recompute' if recovered else 'float32'}): relative gap mean {gap_mean:.2e}, "
                  f"covariance {gap_cov:.2e}")
            assert gap_mean < 1e-4 and gap_cov < 1e-4 and bool((cov32 > 0).all())
            if recovered:
                assert torch.equal(mean32, rounded[0].float()) and torch.equal(cov32, rounded[1].float())


@pytest.mark.parametrize("inference", INFERENCE)
def test_float64_failure_raises_and_names_the_adf_query_covariance(inference: str) -> None:
    r"""A float64 call is never recomputed: negative variances (C_i negative definite) raise, naming A^h."""
    ops = _ops(inference, seed=87)
    data = _inputs(seed=88)
    before = int(source_recovery_counter(data["mu"].device))
    with pytest.raises(ValueError, match="ADF query covariance") as info:
        ops.minimal_agent_update(data["mu"], -data["sigma_diag"], data["compact"], data["log_prior"])
    assert "float32" not in str(info.value)
    assert int(source_recovery_counter(data["mu"].device)) == before


# ----------------------------------------------------------------------------------------------
# 10. Config, operator refusals, optimizer, resources, state dict
# ----------------------------------------------------------------------------------------------

_AGENT = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent")


def test_config_validation_and_inactive_fields() -> None:
    kernel = test_config(attention_mode="projected_predictive")
    assert {"minimal_agent_inference", "minimal_agent_channel_noise"} <= set(kernel.inactive_fields())
    assert "predictive_readout" not in kernel.inactive_fields()
    assert "predictive_readout" in test_config().inactive_fields()   # projected_gradient reads no predictive readout
    for inference in INFERENCE:
        cfg = test_config(**_AGENT, minimal_agent_inference=inference, minimal_agent_channel_noise=0.25)
        inactive = cfg.inactive_fields()
        assert {"relational_routing", "content_link_chart", "attention_residual", "relational_output_init",
                "kappa_beta", "projected_factor_route"} <= set(inactive)
        assert not {"minimal_agent_inference", "minimal_agent_channel_noise", "predictive_readout"} & set(inactive)
        # Passthrough covariance: the floor is inert, the whole-readout checkpoint is not.
        assert "predictive_covariance_floor" in inactive and "predictive_covariance_checkpoint" not in inactive
    with pytest.raises(ValueError, match="projected_predictive"):
        test_config(predictive_readout="minimal_agent")             # test_config defaults to projected_gradient
    with pytest.raises(ValueError, match="predictive_readout"):
        test_config(attention_mode="projected_predictive", predictive_readout="dense")
    with pytest.raises(ValueError, match="minimal_agent_inference"):
        test_config(**_AGENT, minimal_agent_inference="laplace")
    for bad in (0.0, -1.0, math.inf, math.nan, True, "1.0"):
        with pytest.raises(ValueError, match="minimal_agent_channel_noise"):
            test_config(**_AGENT, minimal_agent_channel_noise=bad)
    with pytest.raises(ValueError, match="content_link_edges"):
        test_config(**_AGENT, content_link_chart="exponential", content_link_edges="lowrank", content_link_edge_rank=2)
    with pytest.raises(ValueError, match="rope_on_value"):
        test_config(**_AGENT, pos_rotation="rope", rope_on_value=False)
    test_config(**_AGENT, pos_rotation="rope", rope_on_value=True)  # one frame for key and content: accepted
    with pytest.raises(ValueError, match="equal heads"):
        test_config(**_AGENT, gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2), ("l2", 1)], embed_dim=11,
                    n_heads=3)


def test_operator_constructor_refusals_parameters_and_refusing_methods() -> None:
    kwargs = dict(predictive_readout="minimal_agent")
    with pytest.raises(ValueError, match="predictive_content"):
        make_relational_operators([2, 2], 1, readout_mode="full_link_update", **kwargs)
    with pytest.raises(ValueError, match="equal heads"):
        make_relational_operators([3, 3, 5], 1, **kwargs)
    with pytest.raises(ValueError, match="content_link_edges"):
        make_relational_operators([2, 2], 1, content_link_chart="exponential", content_link_edges="lowrank",
                            content_link_edge_rank=2, **kwargs)
    with pytest.raises(ValueError, match="attention_residual"):
        make_relational_operators([2, 2], 1, attention_residual=True, **kwargs)
    with pytest.raises(ValueError, match="predictive_readout"):
        make_relational_operators([2, 2], 1, predictive_readout="dense")
    with pytest.raises(ValueError, match="minimal_agent_inference"):
        make_relational_operators([2, 2], 1, minimal_agent_inference="laplace", **kwargs)
    with pytest.raises(ValueError, match="minimal_agent_channel_noise"):
        make_relational_operators([2, 2], 1, minimal_agent_channel_noise=0.0, **kwargs)
    ops = make_relational_operators([2, 2], 1, minimal_agent_channel_noise=0.5, **kwargs)
    assert set(ops.state_dict()) == {"raw_score_q", "T", "channel_log_noise"}
    assert ops.raw_score_k is None and ops.O is None
    assert torch.equal(ops.raw_score_q, ops.T)                        # S_Q = T's block projector at step 0
    assert torch.equal(ops.T, torch.eye(4).reshape(2, 2, 4))
    torch.testing.assert_close(ops.channel_log_noise.detach(), torch.full((2,), math.log(0.5)))
    (S_q, S_k), = ops.comparison_runs()
    assert S_q is ops.raw_score_q and S_k is ops.T                    # tied key
    assert ops.comparison_pair()[1] is ops.T
    assert ops.inference_semantics == "relational_minimal_agent_exact"
    mu, beta = torch.zeros(1, 3, 4), torch.full((1, 2, 3, 3), 1 / 3)
    frame = CanonicalFrameContext(torch.eye(2).expand(1, 3, 2, 2, 2), torch.eye(2).expand(1, 3, 2, 2, 2),
                                  block_diagonal=True)
    for call in (ops.content_factors, ops.content_factor_runs, lambda: ops._ambient_content_maps(torch.float64),
                 lambda: ops.predictive_content(mu, beta, frame),
                 lambda: ops.predictive_covariance_update(mu, torch.ones(1, 3, 4), beta, frame)):
        with pytest.raises(ValueError, match="minimal_agent"):
            call()
    kernel = make_relational_operators([2, 2], 1)
    assert set(kernel.state_dict()) == {"raw_score_q", "raw_score_k", "T", "O"}
    assert not hasattr(kernel, "channel_log_noise")
    with pytest.raises(ValueError, match="minimal_agent"):
        kernel.minimal_agent_update(mu, torch.ones(1, 3, 4), frame)


def _agent_model(inference: str = "exact", **overrides: object) -> VFEModel:
    cfg = tiny_config(**{**_AGENT, "minimal_agent_inference": inference, "max_seq_len": 4, "n_layers": 2,
                         **overrides})
    torch.manual_seed(0)
    return VFEModel(cfg)


def test_optimizer_grouping_resources_and_run_record() -> None:
    model = _agent_model()
    cfg = model.cfg
    optimizer = T.build_optimizer(model, cfg)                         # the ungrouped-parameter guard runs here
    noise = {name: p for name, p in model.named_parameters() if name.endswith("channel_log_noise")}
    assert len(noise) == 2
    for name, p in noise.items():
        groups = [group for group in optimizer.param_groups if any(q is p for q in group["params"])]
        assert len(groups) == 1
        assert groups[0].get("lr_aux_role") == "relational_score" and groups[0]["weight_decay"] == 0.0
        assert T._parameter_group(name) == "relational_score"
        assert parameter_role(name) == ("context_processing", "relational_score")
    H, d, K, L = 2, 2, 4, 2
    breakdown = parameter_breakdown(dict(model.named_parameters()), cfg)
    assert breakdown["context_processing"]["relational_score"] == L * (H * d * K + H)
    assert breakdown["context_processing"]["relational_content"] == L * H * d * K
    record = _relational_artifact_fields(cfg)["relational_operators"]
    assert record["per_layer_parameter_counts"] == {"score": K * K + H, "content": K * K}
    assert record["total_parameters"] == sum(p.numel() for ops in model.relational_operators_by_layer
                                             for p in ops.parameters())
    assert record["predictive_readout"] == "minimal_agent" and record["key_tied_to_content"] is True
    assert record["inference_semantics"] == model.conditional_semantics == "relational_minimal_agent_exact"
    # No source tuples: a float32 failure (or certificate) recomputes the layer call in float64; float64 raises.
    assert record["factorization_fallback"] == "fp32_failure_or_certificate_layer_fp64_recompute"
    assert record["routing_state"] == "pre_update_sources_adf_receiver"
    assert record["score_covariance_contract"].endswith("_plus_channel_noise")
    mean_field = _relational_artifact_fields(tiny_config(**_AGENT, minimal_agent_inference="mean_field"))
    assert mean_field["relational_operators"]["routing_state"] == "pre_update"      # labels at the input moments
    assert mean_field["relational_operators"]["factorization_fallback"] == (
        "fp32_failure_or_certificate_layer_fp64_recompute_kl_certificate_fp64_refactor")
    kernel = _relational_artifact_fields(tiny_config())["relational_operators"]
    assert "predictive_readout" not in kernel
    assert kernel["factorization_fallback"] == "failed_entries_source_square_root_qr_fp64"
    assert kernel["routing_state"] == "pre_update"
    assert kernel["score_covariance_contract"] == "exact_full_reduced_covariance_from_local_gaussian"


def test_state_dict_keys_and_round_trip() -> None:
    torch.manual_seed(0)
    default = VFEModel(tiny_config(max_seq_len=4, n_layers=2))
    keys = set(default.state_dict())
    assert not any("channel_log_noise" in key for key in keys)       # a default model has no new key
    model = _agent_model("mean_field")
    agent_keys = set(model.state_dict())
    prefixes = ("relational_operators", "relational_operators_extra.0")
    assert agent_keys - keys == {f"{p}.channel_log_noise" for p in prefixes}
    assert keys - agent_keys == {f"{p}.{name}" for p in prefixes for name in ("raw_score_k", "O")}
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.raw_score_q.add_(0.1 * torch.randn_like(ops.raw_score_q))
            ops.channel_log_noise.add_(0.3)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    fresh = _agent_model("mean_field")
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        want, _ = model.eval().forward_beliefs(ids)
        got, _ = fresh.eval().forward_beliefs(ids)
    assert torch.equal(got.mu, want.mu) and torch.equal(got.sigma, want.sigma)
    with pytest.raises(RuntimeError):
        default.load_state_dict(model.state_dict())                  # strict: the parameter sets differ


# ----------------------------------------------------------------------------------------------
# 11. Model smoke: short training, the checkpointed readout
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("covariance", ["passthrough", "moment_matched"])
def test_short_training_is_finite_and_trains_the_maps(inference: str, covariance: str) -> None:
    model = _agent_model(inference, predictive_covariance=covariance)
    optimizer = T.build_optimizer(model, model.cfg)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 6, 2, 1]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(ids, targets)
        assert math.isfinite(float(loss))
        loss.backward()
        for ops in model.relational_operators_by_layer:
            for p in (ops.raw_score_q, ops.T, ops.channel_log_noise):
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0.0
        optimizer.step()


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("covariance", ["passthrough", "moment_matched"])
def test_checkpointed_readout_is_bit_identical(inference: str, covariance: str) -> None:
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 6, 2, 1]])
    outputs = []
    for flag in (False, True):
        model = _agent_model(inference, predictive_covariance=covariance, predictive_covariance_checkpoint=flag)
        _, loss, _ = model(ids, targets)
        loss.backward()
        outputs.append([loss.detach()] + [p.grad for p in model.parameters()])
    assert sum(g is not None for g in outputs[0]) > 4
    for plain, recomputed in zip(*outputs):
        assert (plain is None and recomputed is None) or torch.equal(plain, recomputed)


# ----------------------------------------------------------------------------------------------
# Consumers: diagnostics, posthoc refusals, full-link refusal, value frame, persisted columns
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
def test_diagnostics_and_consumers(inference: str) -> None:
    from vfe3.gradients.full_link import build_full_link_context
    from vfe3.metrics import LINK_SPECTRUM_KEYS
    from vfe3.posthoc import interventions as iv
    from vfe3.posthoc.holonomy import layer_content_maps
    from vfe3.posthoc.maps import map_invariants
    from vfe3.posthoc.routing import not_applicable_reason

    model = _agent_model(inference).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    rows = model.diagnostics_per_layer(ids)
    for key in ma.MINIMAL_AGENT_DIAGNOSTIC_KEYS:
        defined = ma.MINIMAL_AGENT_MODE_ONLY_KEYS.get(key, inference) == inference   # else NaN by design
        assert all(math.isfinite(v) == defined for v in rows[key]), key
    assert all(0.0 < v < 1.0 for v in rows["relational_minimal_agent_gain_fraction"])
    assert rows["relational_minimal_agent_channel_noise"] == [1.0, 1.0]
    assert all(math.isnan(v) for v in rows["relational_content_max_singular_value"])
    assert all(math.isnan(v) for key in LINK_SPECTRUM_KEYS for v in rows[key])
    assert all(v == 0.0 for v in rows["relational_score_klmax_frac"])
    assert sorted("val_" + key for key in rows if key.startswith("relational_minimal_agent_")) == sorted(
        T._VAL_MINIMAL_AGENT_DIAG_KEYS)
    capture: dict = {"diagnostic": {}}
    model.forward_beliefs(ids, capture=capture)
    for cap in capture["diagnostic"]["layer_conditional_captures"]:
        torch.testing.assert_close(cap["beta_pre"].sum(-1), torch.ones(1, 2, 4))
        assert cap["relational_minimal_agent_gain_fraction"].shape == (2,)
    # Posthoc: map invariants read the tied key; holonomy and routing decomposition refuse.
    for row in map_invariants(model.state_dict(), model.cfg):
        assert row["key_tied_to_content"] == 1.0 and row["channel_noise"] == pytest.approx(1.0)
        T_h = model.relational_operators_by_layer[row["layer"]].T.detach()[row["head"]].double()
        assert row["S_K_fro"] == pytest.approx(float(T_h.norm()))   # the S_K columns describe the tied T
        assert math.isnan(row["O_fro"]) and math.isnan(row["M_sigma_max"])   # no content factor O
    with pytest.raises(NotImplementedError, match="minimal_agent"):
        layer_content_maps(model.state_dict(), model.cfg)
    assert "minimal_agent" in not_applicable_reason(model)
    with pytest.raises(iv.NotApplicable):
        with iv.residual_removed(model):
            pass
    with pytest.raises(iv.NotApplicable):                            # no routing softmax is intercepted
        with iv.routing_costs_removed(model):
            model.forward_beliefs(ids)
    ops = model.relational_operators
    frame = CanonicalFrameContext(torch.eye(4).expand(1, 4, 4, 4), torch.eye(4).expand(1, 4, 4, 4))
    with pytest.raises(ValueError, match="minimal_agent"):
        build_full_link_context(torch.zeros(1, 4, 4), torch.ones(1, 4, 4), torch.zeros(1, 4, 4),
                                torch.ones(1, 4, 4), frame, ops)
    score = relational_score(torch.zeros(1, 4, 4), torch.ones(1, 4, 4), torch.zeros(1, 4, 4), torch.ones(1, 4, 4),
                             frame, ops)                              # reads (S_Q, T), the tied comparison pair
    assert score.shape == (1, 2, 4, 4) and bool(torch.isfinite(score).all())


def test_checkpoint_holonomy_refuses_before_the_validation_pass(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from types import SimpleNamespace
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc.holonomy import checkpoint_holonomy

    model = _agent_model("exact")
    monkeypatch.setattr(ev, "load_model", lambda *args, **kwargs: SimpleNamespace(model=model, cfg=model.cfg))
    monkeypatch.setattr(ev, "nll_pass", lambda *args, **kwargs: pytest.fail("the validation pass ran first"))
    with pytest.raises(NotImplementedError, match="minimal_agent"):
        checkpoint_holonomy(tmp_path, "final")


def test_value_frame_must_equal_the_routing_frame() -> None:
    ops = _ops("exact", seed=97)
    data = _inputs(seed=98)
    mu, sigma, frame = data["mu"], data["sigma_diag"], data["compact"]
    want = ops.minimal_agent_update(mu, sigma, frame, data["log_prior"])[0]
    copy = CanonicalFrameContext(frame.forward.clone(), frame.inverse.clone(), block_diagonal=True)
    got = relational_predictive_readout(mu, sigma, frame, ops, log_prior=data["log_prior"], value_frame=copy)
    assert torch.equal(got, want)
    other = CanonicalFrameContext(frame.forward * 1.1, frame.inverse / 1.1, block_diagonal=True)
    with pytest.raises(ValueError, match="value frame"):
        relational_predictive_readout(mu, sigma, frame, ops, log_prior=data["log_prior"], value_frame=other)


def test_persisted_run_logs_the_minimal_agent_columns(tmp_path) -> None:
    import csv
    model = _agent_model("exact", decode_mode="linear", eval_stride=2)
    cfg = model.cfg
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    train = DataLoader(TokenWindows(torch.arange(41) % 7, 4), batch_size=2, shuffle=True)
    val = DataLoader(TokenWindows((torch.arange(23) * 3) % 7, 4, pad_final=True, stride=2), batch_size=2)
    torch.manual_seed(11)
    T.train(model, train, cfg, n_steps=4, grad_clip=1.0, log_interval=2, eval_interval=2, val_loader=val,
            device=torch.device("cpu"), logger=logging.getLogger("minimal-agent-test"), artifacts=artifacts,
            generate_samples=False)
    with (tmp_path / "run" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [2, 4]
    for row in rows:
        assert int(row["diagnostics_failures"]) == 0
        assert row["param_norm_group_other"] == ""                    # channel_log_noise is relational_score
        for key in T._VAL_MINIMAL_AGENT_DIAG_KEYS:
            if ma.MINIMAL_AGENT_MODE_ONLY_KEYS.get(key[len("val_"):], "exact") != "exact":
                assert row[key] == "", key                            # mean-field-only: blank (NaN) here
            else:
                assert math.isfinite(float(row[key])), key
