r"""Independent brute-force reference tests for the minimal-agent attention readout (verifier: Claude Fable 5.1).

Implementation under test: ``predictive_readout='minimal_agent'`` (author: Claude Opus 5.5; ``vfe3/model/minimal_agent.py``,
``RelationalOperators.minimal_agent_update``, ``relational_predictive_readout``). Theory: ``minimal_agent_model.tex``
Sections 2 to 4 (Definition ma:def:model, Proposition ma:prop:minimal (a) to (e), Proposition ma:prop:mean_field);
specification: ``docs/implementation/minimal-agent-plan-2026-10-06.md`` Sections 0, 1, 2, 4 and 5.

Every expected value here is derived from the model DEFINITION, not from the closed forms of the proposition:

* The record model is built literally. For a receiver :math:`i` and a label tuple :math:`(j_1, \dots, j_H)` the joint
  Gaussian vector :math:`Z = (X_i, X_{\text{slot}_1}, \dots, \varepsilon_1, \dots, \varepsilon_H)` is formed with
  block-diagonal covariance (one independent source snapshot per head, or one per distinct source for the note's
  shared-source model), the stacked record :math:`o = LZ` with :math:`o_h = S_hX_i - T_hX_{j_h} + \varepsilon_h` is a
  linear image of it, and :math:`X_i\mid o = 0` follows from generic Gaussian conditioning
  :math:`m = \mathbb EX + \operatorname{Cov}(X, o)\operatorname{Cov}(o)^{-1}(0 - \mathbb Eo)`,
  :math:`P = \operatorname{Cov}X - \operatorname{Cov}(X, o)\operatorname{Cov}(o)^{-1}\operatorname{Cov}(o, X)`, with tuple
  weight :math:`\pi\,N(0; \mathbb Eo, \operatorname{Cov}o)` (Bayes at :math:`o = 0`). No record-covariance, gain or
  residual formula of the proposition is used. The mixture over tuples is moment matched.
* The ADF reference conditions the receiver on head 1 by that brute force, collapses, and feeds the collapsed Gaussian
  (sources unchanged) to head 2.
* Power posteriors are read off the explicit log density :math:`\log N(x; \xi_i, C_i) + \sum_{h,j}\beta^h_{ij}\log N(T_h\xi_j;
  S_hx, B^h_j)` through its autograd Hessian and gradient (the density is quadratic in :math:`x`).
* KL labels use ``torch.distributions.kl_divergence``; weights use ``MultivariateNormal.log_prob``.

Tiny float64 CPU instances (:math:`K\le 6`, :math:`N\le 4`, :math:`H\le 2`, batch :math:`\le 2`), one thread. Tolerances
are :math:`10^{-10}` relative (scale-aware) unless a test explains a looser one.
"""
from tests.profiles import make_relational_operators
import itertools
import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

import vfe3.train as train_module
from tests.profiles import test_config, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model import minimal_agent as ma
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import _relational_artifact_fields
from vfe3.run_resources import parameter_breakdown, parameter_role

torch.set_num_threads(1)
F64 = torch.float64
INFERENCE = ("exact", "mean_field")
FLOOR = 2e-3
AGENT = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent")


# ----------------------------------------------------------------------------------------------
# Random instances
# ----------------------------------------------------------------------------------------------

def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=F64)


def _spd(g: torch.Generator, *shape: int, ridge: float = 0.4) -> torch.Tensor:          # (..., n) -> (..., n, n)
    X = _randn(g, *shape, shape[-1])
    return X @ X.transpose(-1, -2) + ridge * torch.eye(shape[-1], dtype=F64)


def _sym(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X + X.transpose(-1, -2))


def _causal_log_prior(g: torch.Generator, H: int, N: int, scale: float = 0.4) -> torch.Tensor:   # (H, N, N)
    r"""A causal label prior with random finite entries on and below the diagonal and :math:`-\infty` above it."""
    mask = torch.ones(N, N, dtype=torch.bool).triu(1)
    return (scale * _randn(g, H, N, N)).masked_fill(mask, -math.inf)


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:                                   # (..., G, g, g) -> (..., K, K)
    G, g = blocks.shape[-3], blocks.shape[-1]
    out = blocks.new_zeros(*blocks.shape[:-3], G * g, G * g)
    for m in range(G):
        out[..., m * g:(m + 1) * g, m * g:(m + 1) * g] = blocks[..., m, :, :]
    return out


def _close(got: torch.Tensor, want: torch.Tensor, tol: float, what: str = "") -> None:
    scale = max(1.0, float(want.abs().max()))
    torch.testing.assert_close(got, want, rtol=tol, atol=tol * scale, msg=lambda m: f"{what}\n{m}")


def _reference_moments(g: torch.Generator, N: int, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Reference moments :math:`\xi_i = U_i^{-1}\mu_i`, :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}` of random local beliefs."""
    U = torch.eye(K, dtype=F64) + 0.3 * _randn(g, N, K, K)
    mu = _randn(g, N, K) + 1.2
    Sigma = _spd(g, N, K)
    U_inv = torch.linalg.inv(U)
    return (U_inv @ mu.unsqueeze(-1)).squeeze(-1), _sym(U_inv @ Sigma @ U_inv.transpose(-1, -2))


class _Local(NamedTuple):
    U:          torch.Tensor                 # (B, N, K, K) dense frames (block diagonal)
    compact:    CanonicalFrameContext        # the same frames as (B, N, H, d, d) blocks
    dense:      CanonicalFrameContext
    mu:         torch.Tensor                 # (B, N, K)
    sigma_diag: torch.Tensor                 # (B, N, K)
    sigma_full: torch.Tensor                 # (B, N, K, K)
    log_prior:  torch.Tensor                 # (H, N, N) causal


def _local_inputs(seed: int, N: int, H: int, d: int, batch: int = 1) -> _Local:
    K = H * d
    g = _gen(seed)
    blocks = torch.eye(d, dtype=F64) + 0.3 * _randn(g, batch, N, H, d, d)
    U = _block_dense(blocks)
    return _Local(U, CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True),
                  CanonicalFrameContext(U, torch.linalg.inv(U)), _randn(g, batch, N, K) + 1.0,
                  torch.rand(batch, N, K, generator=g, dtype=F64) + 0.4, _spd(g, batch, N, K),
                  _causal_log_prior(g, H, N))


def _operators(inference: str, H: int, d: int, seed: int, *, checkpoint: bool = False,
               noise_init: float = 1.0) -> RelationalOperators:
    r"""Float64 minimal-agent operators with generic full-row-rank :math:`S_h`, :math:`T_h` and channel noises."""
    K = H * d
    ops = make_relational_operators([d] * H, 11, predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
                              predictive_covariance_checkpoint=checkpoint, predictive_readout="minimal_agent",
                              minimal_agent_inference=inference, minimal_agent_channel_noise=noise_init).double()
    g = _gen(1000 + seed)
    with torch.no_grad():
        ops.raw_score_q.copy_(torch.eye(K, dtype=F64).reshape(H, d, K) + 0.35 * _randn(g, H, d, K))
        ops.T.copy_(torch.eye(K, dtype=F64).reshape(H, d, K) + 0.35 * _randn(g, H, d, K))
        ops.channel_log_noise.copy_(torch.log(0.3 + torch.rand(H, generator=g, dtype=F64)))
    return ops


def _maps(ops: RelationalOperators) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ops.raw_score_q.detach(), ops.T.detach(), ops.channel_log_noise.detach().exp()


# ----------------------------------------------------------------------------------------------
# Brute-force references from the model definition
# ----------------------------------------------------------------------------------------------

def _mvn_logpdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    return MultivariateNormal(mean, covariance_matrix=_sym(cov)).log_prob(x)


class _Component(NamedTuple):
    labels:      Tuple[int, ...]
    log_weight:  torch.Tensor      # () log pi(tuple) + log N(0; E[o], Cov o); -inf when the prior masks the tuple
    mean:        torch.Tensor      # (K,) E[X_i | o = 0, tuple]
    cov:         torch.Tensor      # (K, K) Cov[X_i | o = 0, tuple]
    record_mean: torch.Tensor      # (H d,) E[o | tuple]
    record_cov:  torch.Tensor      # (H d, H d) Cov[o | tuple]


def _record_components(
    xi_i: torch.Tensor, C_i: torch.Tensor,                 # receiver (K,), (K, K)
    S: torch.Tensor, T: torch.Tensor, noise: torch.Tensor,  # (H, d, K), (H, d, K), (H,)
    xi_src: torch.Tensor, C_src: torch.Tensor,             # sources (J, K), (J, K, K) (positive semidefinite)
    log_prior_i: torch.Tensor,                             # (H, J) log pi^h_{ij}
    *,
    shared_sources: bool,
) -> List[_Component]:
    r"""Every label tuple of the record model for one receiver, by generic conditioning of the literal joint law.

    With ``shared_sources=False`` each head hears its own independent snapshot of the selected source (the
    specification's per-head snapshot model); with ``True`` a source selected by several heads enters all their
    records as one random vector (the note's Definition ma:def:model).
    """
    H, d, K = S.shape
    J = xi_src.shape[0]
    eye_d, eye_K = torch.eye(d, dtype=F64), torch.eye(K, dtype=F64)
    zero_o = torch.zeros(H * d, dtype=F64)
    out: List[_Component] = []
    for labels in itertools.product(range(J), repeat=H):
        log_pi = sum(float(log_prior_i[h, j]) for h, j in enumerate(labels))
        slots = list(dict.fromkeys(labels)) if shared_sources else list(labels)
        slot_of_head = [slots.index(j) if shared_sources else h for h, j in enumerate(labels)]
        n_slots = len(slots)
        dim = K + n_slots * K + H * d
        mean_Z = torch.cat([xi_i] + [xi_src[j] for j in slots] + [zero_o])
        cov_Z = torch.block_diag(C_i, *[C_src[j] for j in slots], *[noise[h] * eye_d for h in range(H)])
        L = torch.zeros(H * d, dim, dtype=F64)                                   # o = L Z
        for h in range(H):
            rows = slice(h * d, (h + 1) * d)
            L[rows, :K] = S[h]
            s = slot_of_head[h]
            L[rows, K + s * K:K + (s + 1) * K] = -T[h]
            L[rows, K + n_slots * K + h * d:K + n_slots * K + (h + 1) * d] = eye_d
        E = torch.zeros(K, dim, dtype=F64)                                       # X_i = E Z
        E[:, :K] = eye_K
        record_mean = L @ mean_Z
        record_cov = _sym(L @ cov_Z @ L.T)
        cross = E @ cov_Z @ L.T                                                  # Cov(X_i, o)
        gain = torch.linalg.solve(record_cov, cross.T).T                         # Cov(X_i, o) Cov(o)^{-1}
        mean = xi_i + gain @ (zero_o - record_mean)
        cov = _sym(C_i - gain @ cross.T)
        log_weight = (torch.tensor(log_pi, dtype=F64) + _mvn_logpdf(zero_o, record_mean, record_cov)
                      if math.isfinite(log_pi) else torch.tensor(-math.inf, dtype=F64))
        out.append(_Component(tuple(labels), log_weight, mean, cov, record_mean, record_cov))
    return out


def _mixture(components: Sequence[_Component]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Tuple posterior weights and the moment-matched mixture :math:`\sum_kw_k[P_k + (m_k - \bar m)(m_k - \bar m)^\top]`."""
    w = torch.softmax(torch.stack([c.log_weight for c in components]), dim=0)
    mean = sum(w[k] * c.mean for k, c in enumerate(components))
    cov = sum(w[k] * (c.cov + torch.outer(c.mean - mean, c.mean - mean)) for k, c in enumerate(components))
    return w, mean, _sym(cov)


def _marginal(w: torch.Tensor, components: Sequence[_Component], h: int, J: int) -> torch.Tensor:
    r"""Marginal posterior of head :math:`h`'s label."""
    out = torch.zeros(J, dtype=F64)
    for k, c in enumerate(components):
        out[c.labels[h]] += w[k]
    return out


class _Sequential(NamedTuple):
    xi:            torch.Tensor    # (N, K)
    C:             torch.Tensor    # (N, K, K)
    weights:       torch.Tensor    # (H, N, N) labels used by head h at its current receiver moments
    cost:          torch.Tensor    # (H, N, N) -log N(0; E o, Cov o) - (d/2) log 2 pi
    gain_fraction: torch.Tensor    # (N, H) sum_j w_ij tr(A_i W_ij^{-1}) / d at head h's current moments


def _sequential_reference(
    xi_r: torch.Tensor, C_r: torch.Tensor,                 # receivers (N, K), (N, K, K)
    xi_s: torch.Tensor, C_s: torch.Tensor,                 # sources (J, K), (J, K, K), fixed input snapshots
    S: torch.Tensor, T: torch.Tensor, noise: torch.Tensor, log_prior: torch.Tensor,   # (H, N, J)
) -> _Sequential:
    r"""Heads in index order: head :math:`h` conditions the Gaussian left by heads :math:`1..h-1` on its own record
    by the brute force of :func:`_record_components` and collapses the mixture to two moments."""
    H, d, _ = S.shape
    N, J = xi_r.shape[0], xi_s.shape[0]
    weights, cost = torch.zeros(H, N, J, dtype=F64), torch.zeros(H, N, J, dtype=F64)
    fraction = torch.zeros(N, H, dtype=F64)
    xi_cur, C_cur = xi_r.clone(), C_r.clone()
    for h in range(H):
        new_xi, new_C = [], []
        for i in range(N):
            comps = _record_components(xi_cur[i], C_cur[i], S[h:h + 1], T[h:h + 1], noise[h:h + 1], xi_s, C_s,
                                       log_prior[h:h + 1, i], shared_sources=False)
            w, mean, cov = _mixture(comps)
            A_i = S[h] @ C_cur[i] @ S[h].T
            for j, c in enumerate(comps):
                weights[h, i, j] = w[j]
                cost[h, i, j] = -_mvn_logpdf(torch.zeros(d, dtype=F64), c.record_mean, c.record_cov) \
                    - 0.5 * d * math.log(2 * math.pi)
                fraction[i, h] += w[j] * torch.trace(torch.linalg.solve(c.record_cov, A_i)) / d
            new_xi.append(mean)
            new_C.append(cov)
        xi_cur, C_cur = torch.stack(new_xi), torch.stack(new_C)
    return _Sequential(xi_cur, C_cur, weights, cost, fraction)


def _enumerated_reference(
    xi: torch.Tensor, C: torch.Tensor, S: torch.Tensor, T: torch.Tensor, noise: torch.Tensor, log_prior: torch.Tensor,
    *, shared_sources: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Exact posterior over all :math:`N^H` label tuples, moment matched; also the per-head marginals (H, N, N)."""
    H = S.shape[0]
    N = xi.shape[0]
    means, covs, marginals = [], [], torch.zeros(H, N, N, dtype=F64)
    for i in range(N):
        comps = _record_components(xi[i], C[i], S, T, noise, xi, C, log_prior[:, i], shared_sources=shared_sources)
        w, mean, cov = _mixture(comps)
        means.append(mean)
        covs.append(cov)
        for h in range(H):
            marginals[h, i] = _marginal(w, comps, h, N)
    return torch.stack(means), torch.stack(covs), marginals


def _quad_logpdf(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    r""":math:`\log N(x; m, \Sigma) = -\tfrac12(x - m)^\top\Sigma^{-1}(x - m) - \tfrac12\log\det\Sigma - \tfrac k2\log 2\pi`,
    written with differentiable primitives for the autograd Hessian."""
    delta = x - mean
    return (-0.5 * delta @ torch.linalg.solve(cov, delta) - 0.5 * torch.logdet(cov)
            - 0.5 * x.shape[-1] * math.log(2 * math.pi))


def _power_posterior(
    xi_i: torch.Tensor, C_i: torch.Tensor, S: torch.Tensor, T: torch.Tensor, noise: torch.Tensor,
    beta_i: torch.Tensor,                                  # (H, J) fixed labels
    xi_src: torch.Tensor, C_src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Mean and covariance of :math:`q(x)\propto N(x; \xi_i, C_i)\prod_{h,j}N(T_h\xi_j; S_hx, B^h_j)^{\beta^h_{ij}}`,
    :math:`B^h_j = T_hC_jT_h^\top + n_hI`, from the Hessian :math:`\mathcal H` and gradient :math:`g_0` of its log density
    at :math:`x_0 = 0`: the density is quadratic, so the precision is :math:`-\mathcal H` and the mean is
    :math:`x_0 - \mathcal H^{-1}g_0`."""
    H, d, K = S.shape
    eye_d = torch.eye(d, dtype=F64)
    B = [[_sym(T[h] @ C_src[j] @ T[h].T) + noise[h] * eye_d for j in range(xi_src.shape[0])] for h in range(H)]

    def log_density(x: torch.Tensor) -> torch.Tensor:
        value = _quad_logpdf(x, xi_i, C_i)
        for h in range(H):
            for j in range(xi_src.shape[0]):
                if float(beta_i[h, j]) > 0.0:
                    value = value + beta_i[h, j] * _quad_logpdf(T[h] @ xi_src[j], S[h] @ x, B[h][j])
        return value

    x0 = torch.zeros(K, dtype=F64)
    hessian = torch.autograd.functional.hessian(log_density, x0)
    leaf = x0.clone().requires_grad_(True)
    grad = torch.autograd.grad(log_density(leaf), leaf)[0]
    mean = x0 - torch.linalg.solve(hessian, grad)
    return mean, _sym(torch.linalg.inv(-hessian))


class _MeanField(NamedTuple):
    labels: torch.Tensor   # (H, N, N)
    kl:     torch.Tensor   # (H, N, N) KL(N(S_h xi_i, S_h C_i S_h^T) || N(T_h xi_j, T_h C_j T_h^T + n_h I))
    xi:     torch.Tensor   # (N, K)
    C:      torch.Tensor   # (N, K, K)


def _mean_field_reference(
    xi: torch.Tensor, C: torch.Tensor, S: torch.Tensor, T: torch.Tensor, noise: torch.Tensor, log_prior: torch.Tensor,
) -> _MeanField:
    r"""Label step at the input moments with ``torch.distributions`` KL, then the power posterior at those labels."""
    H, d, _ = S.shape
    N = xi.shape[0]
    eye_d = torch.eye(d, dtype=F64)
    kl = torch.zeros(H, N, N, dtype=F64)
    for h in range(H):
        for i in range(N):
            q = MultivariateNormal(S[h] @ xi[i], covariance_matrix=_sym(S[h] @ C[i] @ S[h].T))
            for j in range(N):
                p = MultivariateNormal(T[h] @ xi[j], covariance_matrix=_sym(T[h] @ C[j] @ T[h].T) + noise[h] * eye_d)
                kl[h, i, j] = kl_divergence(q, p)
    labels = torch.softmax(log_prior - kl, dim=-1)
    means, covs = zip(*[_power_posterior(xi[i], C[i], S, T, noise, labels[:, i], xi, C) for i in range(N)])
    return _MeanField(labels, kl, torch.stack(means), torch.stack(covs))


def _reference_posterior_covariance(C_dense: torch.Tensor, result: ma.MinimalAgentResult) -> torch.Tensor:
    r"""The implementation's factored covariance :math:`C_i - \sum_h\mathsf K_i^hM_i^h\mathsf K_i^{h\top}`, assembled."""
    return C_dense - torch.einsum("...nhks,...nhst,...nhlt->...nkl", result.gains, result.M, result.gains)


def _local_reference(
    mu: torch.Tensor, sigma: torch.Tensor, U: torch.Tensor,          # (N, K), (N, K) or (N, K, K), (N, K, K)
    S: torch.Tensor, T: torch.Tensor, noise: torch.Tensor, log_prior: torch.Tensor, inference: str, floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Local outputs :math:`\mu^+ = U\xi^+`, :math:`\Sigma^+ = UC^+U^\top + f\,\tfrac{\operatorname{tr}C}{K}UU^\top`
    (its diagonal for a diagonal input) from the brute-force reference posteriors, plus the labels and costs."""
    N, K = mu.shape
    diagonal = sigma.dim() == 2
    U_inv = torch.linalg.inv(U)
    xi = (U_inv @ mu.unsqueeze(-1)).squeeze(-1)
    Sigma = torch.diag_embed(sigma) if diagonal else sigma
    C = _sym(U_inv @ Sigma @ U_inv.transpose(-1, -2))
    if inference == "exact":
        seq = _sequential_reference(xi, C, xi, C, S, T, noise, log_prior)
        xi_plus, C_plus, labels, cost = seq.xi, seq.C, seq.weights, seq.cost
    else:
        mf = _mean_field_reference(xi, C, S, T, noise, log_prior)
        xi_plus, C_plus, labels, cost = mf.xi, mf.C, mf.labels, mf.kl
    mu_plus = (U @ xi_plus.unsqueeze(-1)).squeeze(-1)
    trace = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)                                  # (N,)
    Sigma_plus = U @ C_plus @ U.transpose(-1, -2) + (floor * trace / K)[:, None, None] * (U @ U.transpose(-1, -2))
    Sigma_plus = _sym(Sigma_plus)
    return mu_plus, (torch.diagonal(Sigma_plus, dim1=-2, dim2=-1) if diagonal else Sigma_plus), labels, cost


# ----------------------------------------------------------------------------------------------
# A. One head: the readout equals generic conditioning of the literal record model (item (a))
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("K,d", [(5, 2), (6, 3), (3, 3)])
def test_single_head_equals_generic_conditioning_of_the_record_model(K: int, d: int) -> None:
    g = _gen(7 * K + d)
    N = 4
    xi, C = _reference_moments(g, N, K)
    S, T = _randn(g, 1, d, K), _randn(g, 1, d, K)
    noise = torch.tensor([0.65], dtype=F64)
    lp = _causal_log_prior(g, 1, N)
    result = ma.minimal_agent_readout(xi, C, S, T, noise.log(), lp, "exact", diagnostics=True)
    ref = _sequential_reference(xi, C, xi, C, S, T, noise, lp)   # one head: the per-pair brute force itself
    _close(result.weights, ref.weights, 1e-10, "source posterior w_ij")
    _close(result.cost, ref.cost, 1e-10, "per-pair cost -log N(b_j; a_i, W_ij) - (d/2) log 2 pi")
    _close(result.xi, ref.xi, 1e-10, "posterior mean")
    _close(_reference_posterior_covariance(C, result), ref.C, 1e-10, "posterior covariance (mixture)")
    _close(result.gain_fraction, ref.gain_fraction, 1e-10, "gain fraction tr(A Lambda_bar)/d")
    assert bool(((result.gain_fraction > 0) & (result.gain_fraction < 1)).all())
    assert torch.equal(result.head_coupling, torch.zeros(N, 1, dtype=F64))
    # Masked pairs carry weight exactly 0 and the rows are normalized.
    masked = ~torch.isfinite(lp)
    assert float(result.weights[masked].abs().max()) == 0.0
    _close(result.weights.sum(-1), torch.ones(1, N, dtype=F64), 1e-12, "row sums")
    # The two covariances at the input moments.
    _close(result.query_covariance[:, 0], _sym(S[0] @ C @ S[0].T), 1e-12, "S C S^T")
    _close(result.key_covariance[:, 0], _sym(T[0] @ C @ T[0].T), 1e-12, "T C T^T")
    # Component form of item (a), from the brute-force components: m_ij, P_ij and the mixture identity.
    for i in range(N):
        comps = _record_components(xi[i], C[i], S, T, noise, xi, C, lp[:, i], shared_sources=False)
        w, mean, cov = _mixture(comps)
        _close(result.xi[i], mean, 1e-10)
        for j, c in enumerate(comps):
            # The record covariance of the literal model is W_ij = S C_i S^T + T C_j T^T + n I.
            _close(c.record_cov, _sym(S[0] @ C[i] @ S[0].T + T[0] @ C[j] @ T[0].T) + noise[0] * torch.eye(d, dtype=F64),
                   1e-12, "record covariance")
            # The residual of every component is precision weighted: I - G S = P_ij C_i^{-1}.
            G = c.record_cov.new_zeros(K, d)
            G = torch.linalg.solve(c.record_cov, (C[i] @ S[0].T).T).T                   # C_i S^T W_ij^{-1}
            _close(torch.eye(K, dtype=F64) - G @ S[0], c.cov @ torch.linalg.inv(C[i]), 1e-10, "I - G S = P C^{-1}")


def test_exact_record_posterior_statistics_match_their_definitions() -> None:
    r"""``(u, Lambda_bar, R)`` of :func:`exact_record_posterior` against sums over the brute-force components."""
    g = _gen(77)
    N, K, d = 4, 5, 2
    xi, C = _reference_moments(g, N, K)
    S, T = _randn(g, 1, d, K), _randn(g, 1, d, K)
    n = 0.9
    lp = _causal_log_prior(g, 1, N)
    a, A = xi @ S[0].T, _sym(S[0] @ C @ S[0].T)
    b, B = xi @ T[0].T, _sym(T[0] @ C @ T[0].T) + n * torch.eye(d, dtype=F64)
    post = ma.exact_record_posterior(a, A, b, B, lp[0])
    assert not bool(post.failed)
    for i in range(N):
        comps = _record_components(xi[i], C[i], S, T, torch.tensor([n], dtype=F64), xi, C, lp[:, i],
                                   shared_sources=False)
        w, _, _ = _mixture(comps)
        W_inv = [torch.linalg.inv(c.record_cov) for c in comps]
        r = [-(W_inv[j] @ c.record_mean) for j, c in enumerate(comps)]      # W^{-1}(b_j - a_i) = -W^{-1} E[o]
        u = sum(w[j] * r[j] for j in range(N))
        _close(post.weights[i], w, 1e-10)
        _close(post.u[i], u, 1e-10, "u_i")
        _close(post.Lambda_bar[i], sum(w[j] * W_inv[j] for j in range(N)), 1e-10, "Lambda_bar_i")
        _close(post.R[i], sum(w[j] * torch.outer(r[j], r[j]) for j in range(N)) - torch.outer(u, u), 1e-10, "R_i")
        # The assembled update equals the mixture moments.
        Kmat = C[i] @ S[0].T
        _, mean, cov = _mixture(comps)
        _close(xi[i] + Kmat @ post.u[i], mean, 1e-10)
        _close(C[i] - Kmat @ (post.Lambda_bar[i] - post.R[i]) @ Kmat.T, cov, 1e-10)
    # A record covariance that is not positive definite raises; check=False reports it instead.
    bad_A = -torch.eye(d, dtype=F64).expand(N, d, d)
    with pytest.raises(ValueError, match="positive definite"):
        ma.exact_record_posterior(a, bad_A, b, 0.5 * torch.eye(d, dtype=F64).expand(N, d, d))
    assert bool(ma.exact_record_posterior(a, bad_A, b, 0.5 * torch.eye(d, dtype=F64).expand(N, d, d),
                                          check=False).failed)


# ----------------------------------------------------------------------------------------------
# B. Two heads: enumeration of the N^2 label pairs under the query condition; the ADF reference otherwise
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["blocks", "dense"])
@pytest.mark.parametrize("sender_condition", [False, True])
def test_two_heads_under_the_query_condition_equal_the_enumerated_posterior(layout: str, sender_condition: bool) -> None:
    r"""Block-local :math:`S_h` on block-diagonal :math:`C` gives :math:`S_1CS_2^\top = 0`. With dense :math:`T` the
    reference is the per-head snapshot model (the specification's declaration); with block-local :math:`T` the sender
    condition :math:`T_1C_jT_2^\top = 0` also holds and the note's shared-source model (item (b)) must agree too."""
    g = _gen(200 + int(sender_condition))
    N, H, d = 4, 2, 2
    K = H * d
    C_blocks = _spd(g, N, H, d)                                              # (N, H, d, d)
    C = _block_dense(C_blocks)
    xi = _randn(g, N, K) + 0.8
    S = torch.zeros(H, d, K, dtype=F64)
    for h in range(H):
        S[h, :, h * d:(h + 1) * d] = _randn(g, d, d) + 1.5 * torch.eye(d, dtype=F64)
    if sender_condition:
        T = torch.zeros(H, d, K, dtype=F64)
        for h in range(H):
            T[h, :, h * d:(h + 1) * d] = _randn(g, d, d) + 1.5 * torch.eye(d, dtype=F64)
    else:
        T = _randn(g, H, d, K)
    noise = torch.tensor([0.7, 1.1], dtype=F64)
    lp = _causal_log_prior(g, H, N)
    result = ma.minimal_agent_readout(xi, C_blocks if layout == "blocks" else C, S, T, noise.log(), lp, "exact",
                                      diagnostics=True)
    assert float(result.head_coupling.abs().max()) < 1e-13                 # the query condition holds exactly
    mean_ref, cov_ref, marg_ref = _enumerated_reference(xi, C, S, T, noise, lp, shared_sources=False)
    _close(result.xi, mean_ref, 1e-10, "posterior mean against the N^2 enumeration")
    _close(_reference_posterior_covariance(C, result), cov_ref, 1e-10, "posterior covariance against the enumeration")
    _close(result.weights, marg_ref, 1e-10, "per-head labels equal the enumerated marginals")
    mean_shared, cov_shared, marg_shared = _enumerated_reference(xi, C, S, T, noise, lp, shared_sources=True)
    if sender_condition:
        _close(result.xi, mean_shared, 1e-10, "the note's model agrees under both orthogonality conditions")
        _close(_reference_posterior_covariance(C, result), cov_shared, 1e-10)
        _close(result.weights, marg_shared, 1e-10)
    else:
        gap = float((result.xi - mean_shared).abs().max() / mean_shared.abs().max())
        print(f"\n[reference] query condition only: per-head snapshot (implemented) vs shared-source model, "
              f"relative mean gap {gap:.3e}")


def test_two_coupled_heads_equal_the_sequential_brute_force_and_differ_from_the_exact_posterior() -> None:
    g = _gen(303)
    N, H, d, K = 4, 2, 2, 5
    xi, C = _reference_moments(g, N, K)
    S, T = _randn(g, H, d, K), _randn(g, H, d, K)
    noise = torch.tensor([0.5, 0.9], dtype=F64)
    lp = _causal_log_prior(g, H, N)
    result = ma.minimal_agent_readout(xi, C, S, T, noise.log(), lp, "exact", diagnostics=True)
    ref = _sequential_reference(xi, C, xi, C, S, T, noise, lp)
    _close(result.weights, ref.weights, 1e-10, "labels of head 1 at the input and of head 2 at the collapsed moments")
    _close(result.cost, ref.cost, 1e-10)
    _close(result.xi, ref.xi, 1e-10)
    _close(_reference_posterior_covariance(C, result), ref.C, 1e-10)
    _close(result.gain_fraction, ref.gain_fraction, 1e-10)
    # Head coupling at the input moments, from its definition.
    for i in range(N):
        cross = S[0] @ C[i] @ S[1].T
        own = [S[h] @ C[i] @ S[h].T for h in range(H)]
        want = cross.norm() / math.sqrt(float(own[0].norm() * own[1].norm()))
        _close(result.head_coupling[i], torch.full((H,), float(want), dtype=F64), 1e-10, "head coupling")
    assert float(result.head_coupling.min()) > 0.05
    mean_ex, cov_ex, _ = _enumerated_reference(xi, C, S, T, noise, lp, shared_sources=False)
    gap = float((result.xi - mean_ex).abs().max() / mean_ex.abs().max())
    print(f"\n[reference] coupled heads: ADF vs exact per-head-snapshot posterior, relative mean gap {gap:.3e}")
    assert gap > 1e-6                                                       # the order-dependent approximation is visible


def test_note_one_dimensional_example_grounds_the_brute_force_and_the_adf() -> None:
    r"""Section ma:sec:open: :math:`K = 1`, two heads with :math:`S = T = 1`, receiver at 0 with unit variance, unit
    channel noise, point senders at 0 and 2, uniform priors. The exact tuple masses are :math:`\propto(1, e^{-4/3},
    e^{-4/3}, e^{-4/3})`, the marginal of source 2 is 0.294, the exact mean 0.39252; item (a) per head gives 0.269 and
    the per-head mean 0.53788. The ADF of the implementation is a third number, computed here for the record."""
    xi_r, C_r = torch.zeros(1, 1, dtype=F64), torch.ones(1, 1, 1, dtype=F64)
    xi_s, C_s = torch.tensor([[0.0], [2.0]], dtype=F64), torch.zeros(2, 1, 1, dtype=F64)
    S = T = torch.ones(2, 1, 1, dtype=F64)
    noise = torch.ones(2, dtype=F64)
    lp = torch.zeros(2, 1, 2, dtype=F64)
    comps = _record_components(xi_r[0], C_r[0], S, T, noise, xi_s, C_s, lp[:, 0], shared_sources=True)
    w, mean, _ = _mixture(comps)
    masses = torch.tensor([1.0, math.exp(-4 / 3), math.exp(-4 / 3), math.exp(-4 / 3)], dtype=F64)
    _close(w, masses / masses.sum(), 1e-12, "tuple masses")
    assert abs(float(_marginal(w, comps, 0, 2)[1]) - 0.294) < 5e-4
    assert abs(float(mean[0]) - 0.39252) < 2e-5
    a, A = (S[0] @ xi_r[0]).unsqueeze(0), (S[0] @ C_r[0] @ S[0].T).unsqueeze(0)
    b = torch.stack([T[0] @ xi_s[j] for j in range(2)])
    B = torch.stack([T[0] @ C_s[j] @ T[0].T + noise[0] * torch.eye(1, dtype=F64) for j in range(2)])
    one_head = ma.exact_record_posterior(a, A, b, B)
    _close(one_head.weights[0], torch.tensor([1 / (1 + math.exp(-1)), math.exp(-1) / (1 + math.exp(-1))], dtype=F64),
           1e-12, "item (a) per-head weights")
    per_head_mean = 2 * float(one_head.weights[0, 1])                        # H G^(h) 2 w_2 with G^(h) = 1/2
    assert abs(per_head_mean - 0.53788) < 2e-5

    def head_update(h: int, a_h: torch.Tensor, A_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        post = ma.exact_record_posterior(a_h, A_h, b, B)
        return post.u, post.Lambda_bar - post.R

    X0 = ma.gain_basis(C_r, S, blocks=False)                                 # (1, 2, 1, 1)
    xi_adf, _, _, _ = ma.sequential_head_assembly(xi_r, X0, S, head_update)
    seq = _sequential_reference(xi_r, C_r, xi_s, C_s, S, T, noise, lp)
    _close(xi_adf, seq.xi, 1e-12, "ADF through the public assembly against the sequential brute force")
    adf = float(xi_adf[0, 0])
    print(f"\n[reference] one-dimensional example: exact {float(mean[0]):.5f}, per-head {per_head_mean:.5f}, "
          f"ADF {adf:.5f}")
    assert abs(adf - float(mean[0])) > 1e-3 and abs(adf - per_head_mean) > 1e-3


# ----------------------------------------------------------------------------------------------
# C. Limits: noise-free point-mass senders (item (c)) and full rank (item (d))
# ----------------------------------------------------------------------------------------------

def test_noise_free_point_mass_limit_converges_to_the_minimal_change_projection() -> None:
    r"""Sources with :math:`B_j = \epsilon I` (point senders, channel noise :math:`\epsilon`). As :math:`\epsilon\to 0`
    the weights tend to :math:`w^0_{ij}\propto\pi_{ij}N(b_j; a_i, A_i)`, the mean to the :math:`C^{-1}`-minimal change
    with :math:`S\xi^+ = \bar z`, which is checked by solving the KKT system of that constrained least-squares problem
    (no right-inverse formula), and the covariance to :math:`(I - \Pi)C + R_Q\operatorname{Cov}_w(b)R_Q^\top`. The gain is
    :math:`C S^\top(A + \epsilon I)^{-1}`, so the gap is first order in :math:`\epsilon`: the test asserts the gap at
    :math:`\epsilon = 10^{-8}` is below :math:`10^{-6}` and at least 10 times smaller than at :math:`10^{-6}`."""
    g = _gen(404)
    N, K, d = 4, 5, 2
    xi, C = _reference_moments(g, N, K)
    S, T = _randn(g, d, K), _randn(g, d, K)
    lp = _causal_log_prior(g, 1, N)[0]
    a, A, b = xi @ S.T, _sym(S @ C @ S.T), xi @ T.T
    eye_K, eye_d = torch.eye(K, dtype=F64), torch.eye(d, dtype=F64)
    gaps = []
    for eps in (1e-6, 1e-8):
        post = ma.exact_record_posterior(a, A, b, (eps * eye_d).expand(N, d, d), lp)
        Kmat = C @ S.T
        xi_plus = xi + (Kmat @ post.u.unsqueeze(-1)).squeeze(-1)
        C_plus = C - Kmat @ (post.Lambda_bar - post.R) @ Kmat.transpose(-1, -2)
        worst = 0.0
        for i in range(N):
            logits = lp[i] + torch.stack([_mvn_logpdf(b[j], a[i], A[i]) for j in range(N)])
            w0 = torch.softmax(logits, 0)
            z_bar = w0 @ b
            # KKT system of min delta^T C^{-1} delta subject to S delta = z_bar - a_i.
            kkt = torch.zeros(K + d, K + d, dtype=F64)
            kkt[:K, :K] = torch.linalg.inv(C[i])
            kkt[:K, K:] = S.T
            kkt[K:, :K] = S
            rhs = torch.cat([torch.zeros(K, dtype=F64), z_bar - a[i]])
            delta = torch.linalg.solve(kkt, rhs)[:K]
            R_Q = C[i] @ S.T @ torch.linalg.inv(A[i])
            Pi = R_Q @ S
            spread = sum(w0[j] * torch.outer(b[j] - z_bar, b[j] - z_bar) for j in range(N))
            want_mean, want_cov = xi[i] + delta, (eye_K - Pi) @ C[i] + R_Q @ spread @ R_Q.T
            worst = max(worst, float((post.weights[i] - w0).abs().max()),
                        float((xi_plus[i] - want_mean).abs().max() / want_mean.abs().max()),
                        float((C_plus[i] - want_cov).abs().max() / want_cov.abs().max()),
                        float((S @ xi_plus[i] - z_bar).abs().max()),
                        float(((eye_K - Pi) @ (xi_plus[i] - xi[i])).abs().max()))
            if eps == 1e-8:
                _close(delta, R_Q @ (z_bar - a[i]), 1e-9, "the KKT solution is R_Q applied to the innovation")
                assert float(torch.linalg.matrix_rank(Pi)) == d and float((Pi @ Pi - Pi).abs().max()) < 1e-10
        gaps.append(worst)
    print(f"\n[reference] noise-free limit gaps: eps=1e-6 -> {gaps[0]:.3e}, eps=1e-8 -> {gaps[1]:.3e}")
    assert gaps[1] < 1e-6 and gaps[1] * 10 < gaps[0]


def test_full_rank_head_gives_the_linked_weights_and_the_normalized_product_components() -> None:
    r"""Item (d) with :math:`d = K`: :math:`\Theta = S^{-1}T`, :math:`\tilde C_{ij} = \Theta C_j\Theta^\top + nS^{-1}S^{-\top}`,
    :math:`w_{ij}\propto\pi_{ij}N(\Theta\xi_j; \xi_i, C_i + \tilde C_{ij})`, and the component given :math:`j` is the
    normalized product of :math:`N(\xi_i, C_i)` and :math:`N(\Theta\xi_j, \tilde C_{ij})` (Appendix ma:app:item4),
    :math:`P_j = (C_i^{-1} + \tilde C_{ij}^{-1})^{-1}`, :math:`m_j = P_j(C_i^{-1}\xi_i + \tilde C_{ij}^{-1}\Theta\xi_j)`."""
    g = _gen(505)
    N, K = 4, 3
    xi, C = _reference_moments(g, N, K)
    S = _randn(g, 1, K, K) + 2.0 * torch.eye(K, dtype=F64)
    T = _randn(g, 1, K, K) + 2.0 * torch.eye(K, dtype=F64)
    n = 0.45
    lp = _causal_log_prior(g, 1, N)
    result = ma.minimal_agent_readout(xi, C, S, T, torch.tensor([math.log(n)], dtype=F64), lp, "exact")
    S_inv = torch.linalg.inv(S[0])
    Theta = S_inv @ T[0]
    C_post = _reference_posterior_covariance(C, result)
    for i in range(N):
        tilde = [_sym(Theta @ C[j] @ Theta.T + n * S_inv @ S_inv.T) for j in range(N)]
        logits = lp[0, i] + torch.stack([_mvn_logpdf(Theta @ xi[j], xi[i], C[i] + tilde[j]) for j in range(N)])
        w = torch.softmax(logits, 0)
        P = [torch.linalg.inv(torch.linalg.inv(C[i]) + torch.linalg.inv(tilde[j])) for j in range(N)]
        m = [P[j] @ (torch.linalg.solve(C[i], xi[i]) + torch.linalg.solve(tilde[j], Theta @ xi[j])) for j in range(N)]
        mean = sum(w[j] * m[j] for j in range(N))
        cov = sum(w[j] * (P[j] + torch.outer(m[j] - mean, m[j] - mean)) for j in range(N))
        _close(result.weights[0, i], w, 1e-10, "item (d) weights")
        _close(result.xi[i], mean, 1e-10, "item (d) mean from normalized-product components")
        _close(C_post[i], cov, 1e-10, "item (d) covariance")
    # Lemma ma:lem:link (b) on the mean-field cost: the KL in the comparison space equals the KL of the receiver's
    # full law from the sender's law carried through the link S^{-1} (with the noise carried too).
    mf = ma.minimal_agent_readout(xi, C, S, T, torch.tensor([math.log(n)], dtype=F64), lp, "mean_field")
    for i in range(N):
        q = MultivariateNormal(xi[i], covariance_matrix=C[i])
        for j in range(N):
            p = MultivariateNormal(Theta @ xi[j], covariance_matrix=_sym(Theta @ C[j] @ Theta.T + n * S_inv @ S_inv.T))
            _close(mf.cost[0, i, j], kl_divergence(q, p), 1e-10, "link identity of the Gibbs cost")


def test_full_rank_precise_senders_transport_the_source_means() -> None:
    r"""Item (d), :math:`\tilde C_{ij}\to 0`: the mean tends to :math:`\sum_jw^0_{ij}\Theta\xi_j` with
    :math:`w^0_{ij}\propto\pi_{ij}N(\Theta\xi_j; \xi_i, C_i)`; first order in :math:`\epsilon` as in the noise-free test."""
    g = _gen(606)
    N, K = 4, 3
    xi, C = _reference_moments(g, N, K)
    S = _randn(g, K, K) + 2.0 * torch.eye(K, dtype=F64)
    T = _randn(g, K, K) + 2.0 * torch.eye(K, dtype=F64)
    Theta = torch.linalg.solve(S, T)
    lp = _causal_log_prior(g, 1, N)[0]
    a, A, b = xi @ S.T, _sym(S @ C @ S.T), xi @ T.T
    gaps = []
    for eps in (1e-6, 1e-8):
        post = ma.exact_record_posterior(a, A, b, (eps * torch.eye(K, dtype=F64)).expand(N, K, K), lp)
        xi_plus = xi + ((C @ S.T) @ post.u.unsqueeze(-1)).squeeze(-1)
        worst = 0.0
        for i in range(N):
            logits = lp[i] + torch.stack([_mvn_logpdf(Theta @ xi[j], xi[i], C[i]) for j in range(N)])
            w0 = torch.softmax(logits, 0)
            want = sum(w0[j] * Theta @ xi[j] for j in range(N))
            worst = max(worst, float((xi_plus[i] - want).abs().max() / want.abs().max()),
                        float((post.weights[i] - w0).abs().max()))
        gaps.append(worst)
    assert gaps[1] < 1e-6 and gaps[1] * 10 < gaps[0]


# ----------------------------------------------------------------------------------------------
# D. Mean field: explicit power posterior; Proposition ma:prop:mean_field (iii)
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["dense", "blocks"])
def test_mean_field_equals_kl_labels_and_the_autograd_power_posterior(layout: str) -> None:
    g = _gen(707)
    N, H, d = 4, 2, 2
    K = H * d
    if layout == "dense":
        xi, C = _reference_moments(g, N, K)
        C_in = C
    else:
        C_in = _spd(g, N, H, d)
        C = _block_dense(C_in)
        xi = _randn(g, N, K) + 0.7
    S, T = _randn(g, H, d, K), _randn(g, H, d, K)                              # coupled heads
    noise = torch.tensor([0.8, 0.35], dtype=F64)
    lp = _causal_log_prior(g, H, N)
    result = ma.minimal_agent_readout(xi, C_in, S, T, noise.log(), lp, "mean_field", diagnostics=True)
    ref = _mean_field_reference(xi, C, S, T, noise, lp)
    _close(result.cost, ref.kl, 1e-10, "mean-field cost is the KL (torch.distributions)")
    _close(result.weights, ref.labels, 1e-10, "KL labels")
    _close(result.xi, ref.xi, 1e-10, "power posterior mean")
    _close(_reference_posterior_covariance(C, result), ref.C, 1e-10, "power posterior covariance")
    # The mean-field gain fraction tr(A^h M^h)/d with M^h = (A^h + Lambda^{h -1})^{-1} at the sequential moments; for
    # the FIRST head the sequential moments are the input moments, so it is checked from the labels alone.
    Lambda_1 = sum(ref.labels[0, :, j, None, None] * torch.linalg.inv(_sym(T[0] @ C[j] @ T[0].T)
                                                                      + noise[0] * torch.eye(d, dtype=F64))
                   for j in range(N))                                                       # (N, d, d)
    A_1 = _sym(S[0] @ C @ S[0].T)
    M_1 = torch.linalg.inv(A_1 + torch.linalg.inv(Lambda_1))
    _close(result.gain_fraction[:, 0], torch.diagonal(A_1 @ M_1, dim1=-2, dim2=-1).sum(-1) / d, 1e-10,
           "mean-field gain fraction of head 1")
    assert bool(((result.gain_fraction > 0) & (result.gain_fraction < 1)).all())


def test_power_posterior_at_the_exact_weights_keeps_the_exact_mean_and_drops_the_spread() -> None:
    r"""Proposition ma:prop:mean_field (iii): one head, every source with the same :math:`C_j` (so :math:`B_j = B`).
    The exact weights, component means and mixture moments come from the brute force; the power posterior at
    :math:`\beta = w` from the autograd Hessian. Its mean is the exact mean and its covariance the exact covariance
    minus the between-component spread :math:`\sum_jw_j(m_j - \bar m)(m_j - \bar m)^\top`. The implementation's assembly
    (``label_statistics`` + ``information_step`` through ``sequential_head_assembly``) must return the same."""
    g = _gen(808)
    N, K, d = 4, 5, 3
    C0 = _spd(g, K)
    C = C0.expand(N, K, K).clone()
    xi = _randn(g, N, K) + 0.9
    S, T = _randn(g, 1, d, K), _randn(g, 1, d, K)
    noise = torch.tensor([0.6], dtype=F64)
    lp = _causal_log_prior(g, 1, N)
    b = xi @ T[0].T
    B = (_sym(T[0] @ C0 @ T[0].T) + noise[0] * torch.eye(d, dtype=F64)).expand(N, d, d)
    exact_w, exact_mean, exact_cov, spreads = [], [], [], []
    for i in range(N):
        comps = _record_components(xi[i], C[i], S, T, noise, xi, C, lp[:, i], shared_sources=False)
        w, mean, cov = _mixture(comps)
        exact_w.append(w)
        exact_mean.append(mean)
        exact_cov.append(cov)
        spreads.append(sum(w[k] * torch.outer(c.mean - mean, c.mean - mean) for k, c in enumerate(comps)))
    exact_w = torch.stack(exact_w)
    for i in range(N):
        mean_pp, cov_pp = _power_posterior(xi[i], C[i], S, T, noise, exact_w[i:i + 1], xi, C)
        _close(mean_pp, exact_mean[i], 1e-10, "power posterior mean at beta = w equals the exact mean")
        _close(cov_pp, exact_cov[i] - spreads[i], 1e-10, "power posterior covariance = exact covariance - spread")
        if i > 0:                                                               # receiver 0 hears one source only
            assert float(spreads[i].abs().max()) > 1e-4                         # the spread is not trivially zero
    Lambda, eta, failed = ma.label_statistics(exact_w, b, B)
    assert not bool(failed)
    X0 = ma.gain_basis(C, S, blocks=False)

    def head_update(h: int, a_h: torch.Tensor, A_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u, M, _ = ma.information_step(a_h, A_h, Lambda, eta)
        return u, M

    xi_asm, gains, M, _ = ma.sequential_head_assembly(xi, X0, S, head_update)
    _close(xi_asm, torch.stack(exact_mean), 1e-10, "assembly mean at beta = w")
    _close(C - gains[:, 0] @ M[:, 0] @ gains[:, 0].transpose(-1, -2), torch.stack(exact_cov) - torch.stack(spreads),
           1e-10, "assembly covariance at beta = w")
    kl_labels = ma.mean_field_record_posterior(xi @ S[0].T, _sym(S[0] @ C @ S[0].T), b, B, lp[0]).weights
    assert float((kl_labels - exact_w).abs().max()) > 1e-3                      # KL labels differ from exact weights


# ----------------------------------------------------------------------------------------------
# E. Operator in local coordinates: brute force through frames, both layouts, the dispatch, the covariance contract
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("frame_kind,family", [("compact", "diagonal"), ("compact", "full"),
                                               ("dense", "diagonal"), ("dense", "full")])
def test_operator_local_update_matches_the_brute_force_through_the_frames(inference: str, frame_kind: str,
                                                                            family: str) -> None:
    ops = _operators(inference, 2, 2, seed=11)
    data = _local_inputs(12, N=4, H=2, d=2, batch=2)
    S, T, noise = _maps(ops)
    sigma = data.sigma_diag if family == "diagonal" else data.sigma_full
    frame = data.compact if frame_kind == "compact" else data.dense
    capture: Dict[str, torch.Tensor] = {}
    mean, cov = ops.minimal_agent_update(data.mu, sigma, frame, data.log_prior, return_covariance=True,
                                         capture=capture)
    assert mean.shape == data.mu.shape and cov.shape == sigma.shape
    for bidx in range(2):
        want_mean, want_cov, labels, cost = _local_reference(data.mu[bidx], sigma[bidx], data.U[bidx], S, T, noise,
                                                             data.log_prior, inference, FLOOR)
        _close(mean[bidx], want_mean, 1e-10, f"local mean, batch {bidx}")
        _close(cov[bidx], want_cov, 1e-10, f"local covariance plus floor, batch {bidx}")
        _close(capture["beta_pre"][bidx], labels, 1e-10, "captured labels")
        _close(capture["score_energy_pre"][bidx], cost, 1e-10, "captured costs")
    _close(capture["beta_pre"].sum(-1), torch.ones(2, 2, 4, dtype=F64), 1e-12)
    assert torch.equal(capture["relational_minimal_agent_channel_noise"], noise)
    assert capture["relational_minimal_agent_gain_fraction"].shape == (2,)
    assert float(capture["relational_mean_displacement_norm"]) == pytest.approx(float((mean - data.mu).norm()))
    # Without a requested covariance the mean is unchanged and no covariance is formed (passthrough is the caller's).
    mean_only, none = ops.minimal_agent_update(data.mu, sigma, frame, data.log_prior)
    assert none is None and torch.equal(mean_only, mean)
    # The dispatch returns the same tensors and ignores tau and kl_max.
    via_mean, via_cov = relational_predictive_readout(data.mu, sigma, frame, ops, log_prior=data.log_prior,
                                                      return_covariance=True, tau=0.37, kl_max=2.0)
    assert torch.equal(via_mean, mean) and torch.equal(via_cov, cov)
    assert torch.equal(relational_predictive_readout(data.mu, sigma, frame, ops, log_prior=data.log_prior), mean)


def test_blockwise_equals_dense_and_the_full_family_equals_its_diagonal_projection() -> None:
    ops = _operators("exact", 2, 3, seed=21)
    data = _local_inputs(22, N=4, H=2, d=3, batch=1)
    mu, sigma, lp = data.mu, data.sigma_diag, data.log_prior
    blockwise = ops.minimal_agent_update(mu, sigma, data.compact, lp, return_covariance=True)
    dense_frame = ops.minimal_agent_update(mu, sigma, data.dense, lp, return_covariance=True)
    full_family = ops.minimal_agent_update(mu, torch.diag_embed(sigma), data.compact, lp, return_covariance=True)
    _close(dense_frame[0], blockwise[0], 1e-10)
    _close(dense_frame[1], blockwise[1], 1e-10)
    _close(full_family[0], blockwise[0], 1e-10)
    _close(torch.diagonal(full_family[1], dim1=-2, dim2=-1), blockwise[1], 1e-10)


@pytest.mark.parametrize("inference", INFERENCE)
def test_value_frame_with_equal_matrices_is_accepted_and_a_different_one_refused(inference: str) -> None:
    ops = _operators(inference, 2, 2, seed=31)
    data = _local_inputs(32, N=3, H=2, d=2)
    frame = data.compact
    want = ops.minimal_agent_update(data.mu, data.sigma_diag, frame, data.log_prior)[0]
    same = CanonicalFrameContext(frame.forward.clone(), frame.inverse.clone(), block_diagonal=True)
    got = relational_predictive_readout(data.mu, data.sigma_diag, frame, ops, log_prior=data.log_prior,
                                        value_frame=same)
    assert torch.equal(got, want)
    other = CanonicalFrameContext(frame.forward * 1.05, frame.inverse / 1.05, block_diagonal=True)
    with pytest.raises(ValueError, match="value frame differs"):
        relational_predictive_readout(data.mu, data.sigma_diag, frame, ops, log_prior=data.log_prior,
                                      value_frame=other)


# ----------------------------------------------------------------------------------------------
# F. Passive gauge covariance and causality
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("frame_kind", ["dense", "compact"])
def test_passive_gauge_action_is_covariant(inference: str, frame_kind: str) -> None:
    r""":math:`(U_i, \mu_i, \Sigma_i)\mapsto(g_iU_i, g_i\mu_i, g_i\Sigma_ig_i^\top)` leaves the reference moments, hence
    the labels, unchanged and maps :math:`\mu_i^+\mapsto g_i\mu_i^+`, :math:`\Sigma_i^+\mapsto g_i\Sigma_i^+g_i^\top`
    (the floor :math:`f\,\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top` transforms the same way). Tolerance :math:`10^{-9}`:
    the random :math:`g_i` and frames have condition numbers of a few units, and two inversions enter."""
    ops = _operators(inference, 2, 2, seed=41)
    data = _local_inputs(42, N=4, H=2, d=2)
    N, K = 4, 4
    g = _gen(43)
    if frame_kind == "dense":
        G = torch.eye(K, dtype=F64) + 0.4 * _randn(g, 1, N, K, K)
        frame = data.dense
        moved = CanonicalFrameContext(G @ data.U, torch.linalg.inv(G @ data.U))
    else:
        g_blocks = torch.eye(2, dtype=F64) + 0.4 * _randn(g, 1, N, 2, 2, 2)
        G = _block_dense(g_blocks)
        frame = data.compact
        moved_blocks = g_blocks @ frame.forward
        moved = CanonicalFrameContext(moved_blocks, torch.linalg.inv(moved_blocks), block_diagonal=True)
    mu, sigma, lp = data.mu, data.sigma_full, data.log_prior
    cap, cap_g = {}, {}
    mean, cov = ops.minimal_agent_update(mu, sigma, frame, lp, return_covariance=True, capture=cap)
    mean_g, cov_g = ops.minimal_agent_update((G @ mu.unsqueeze(-1)).squeeze(-1), _sym(G @ sigma @ G.transpose(-1, -2)),
                                             moved, lp, return_covariance=True, capture=cap_g)
    _close(mean_g, (G @ mean.unsqueeze(-1)).squeeze(-1), 1e-9, "mean covariance under the passive action")
    _close(cov_g, G @ cov @ G.transpose(-1, -2), 1e-9, "covariance covariance under the passive action")
    _close(cap_g["beta_pre"], cap["beta_pre"], 1e-9, "labels are gauge invariant")
    _close(cap_g["score_energy_pre"], cap["score_energy_pre"], 1e-9, "costs are gauge invariant")
    _close(cap_g["projected_covariance_pre"], cap["projected_covariance_pre"], 1e-9)


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_causal_prior_makes_later_tokens_invisible_to_earlier_receivers(inference: str, family: str) -> None:
    r"""Changing :math:`(\mu_j, \Sigma_j)` for :math:`j > i` leaves receiver :math:`i` unchanged (masked pairs have
    weight exactly 0 and finite statistics, so the sums do not move); receiver :math:`j` itself changes."""
    ops = _operators(inference, 2, 2, seed=51)
    data = _local_inputs(52, N=4, H=2, d=2)
    sigma = data.sigma_diag if family == "diagonal" else data.sigma_full
    base = ops.minimal_agent_update(data.mu, sigma, data.compact, data.log_prior, return_covariance=True)
    mu2, sigma2 = data.mu.clone(), sigma.clone()
    mu2[0, 2:] += 2.0
    sigma2[0, 2:] *= 1.7
    moved = ops.minimal_agent_update(mu2, sigma2, data.compact, data.log_prior, return_covariance=True)
    for got, want in zip(moved, base):
        torch.testing.assert_close(got[0, :2], want[0, :2], rtol=1e-13, atol=1e-13)
        assert float((got[0, 2:] - want[0, 2:]).abs().max()) > 1e-3
    exact_bits = all(torch.equal(got[0, :2], want[0, :2]) for got, want in zip(moved, base))
    print(f"\n[reference] causality: earlier receivers bitwise identical = {exact_bits}")


# ----------------------------------------------------------------------------------------------
# G. Gradients: central differences, tile boundaries, the whole-readout checkpoint
# ----------------------------------------------------------------------------------------------

def _readout_scalar(ops: RelationalOperators, mu: torch.Tensor, sigma: torch.Tensor, frame: CanonicalFrameContext,
                    lp: torch.Tensor, weights: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, cov = ops.minimal_agent_update(mu, sigma, frame, lp, return_covariance=True)
    return (weights[0] * mean).sum() + (weights[1] * cov).sum()


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_central_difference_gradients(inference: str, family: str) -> None:
    r"""Directional derivatives of a scalar of (mean, covariance) with respect to :math:`S`, :math:`T`,
    :math:`\log n`, :math:`\mu` and :math:`\Sigma` by central differences with :math:`h = 10^{-5}` in float64 against
    autograd. Truncation is :math:`O(h^2)\approx 10^{-10}` and roundoff :math:`\approx\epsilon|f|/h\approx 10^{-11}|f|`,
    so the tolerance is :math:`10^{-7}\max(1, |\nabla f\cdot v|)`."""
    ops = _operators(inference, 2, 2, seed=61)
    data = _local_inputs(62, N=3, H=2, d=2)
    sigma0 = data.sigma_diag if family == "diagonal" else data.sigma_full
    g = _gen(63)
    weights = (_randn(g, *data.mu.shape), _randn(g, *sigma0.shape))
    params = [ops.raw_score_q, ops.T, ops.channel_log_noise]
    mu = data.mu.clone().requires_grad_(True)
    sigma = sigma0.clone().requires_grad_(True)
    ops.zero_grad(set_to_none=True)
    value = _readout_scalar(ops, mu, sigma, data.compact, data.log_prior, weights)
    value.backward()
    grads = [p.grad.detach().clone() for p in params] + [mu.grad.detach().clone(), sigma.grad.detach().clone()]
    leaves = [p.detach().clone() for p in params] + [data.mu.clone(), sigma0.clone()]
    h = 1e-5

    def evaluate(values: List[torch.Tensor]) -> float:
        with torch.no_grad():
            for p, v in zip(params, values[:3]):
                p.copy_(v)
            return float(_readout_scalar(ops, values[3], values[4], data.compact, data.log_prior, weights))

    checked = 0
    for index in range(5):
        for _ in range(2):
            v = _randn(g, *leaves[index].shape)
            if index == 4 and family == "full":
                v = _sym(v)                                                      # keep the covariance symmetric
            plus, minus = [x.clone() for x in leaves], [x.clone() for x in leaves]
            plus[index] = leaves[index] + h * v
            minus[index] = leaves[index] - h * v
            fd = (evaluate(plus) - evaluate(minus)) / (2 * h)
            analytic = float((grads[index] * v).sum())
            assert abs(fd - analytic) <= 1e-7 * max(1.0, abs(analytic)), (index, fd, analytic)
            checked += 1
    with torch.no_grad():
        for p, v in zip(params, leaves[:3]):
            p.copy_(v)
    assert checked == 10


def test_one_pair_tiles_reproduce_the_single_tile_values_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""``PAIR_TILE_BYTES = 1`` forces one pair per tile (checkpointed, online log-sum-exp across source tiles). The
    arithmetic order differs, so the comparison is at :math:`10^{-12}`, not bitwise."""
    ops = _operators("exact", 2, 2, seed=71)
    data = _local_inputs(72, N=4, H=2, d=2)

    def run() -> List[torch.Tensor]:
        ops.zero_grad(set_to_none=True)
        mu = data.mu.clone().requires_grad_(True)
        sigma = data.sigma_full.clone().requires_grad_(True)
        mean, cov = ops.minimal_agent_update(mu, sigma, data.compact, data.log_prior, return_covariance=True)
        (mean.square().sum() + cov.sum()).backward()
        return [mean.detach(), cov.detach(), mu.grad, sigma.grad] + [p.grad for p in ops.parameters()]

    default = run()
    seen: List[Tuple[int, int]] = []
    original = ma._exact_tile

    def recording(a, A, b, B, lp):
        seen.append((a.shape[-2], b.shape[-2], bool(torch.isfinite(lp).any())))
        return original(a, A, b, B, lp)

    monkeypatch.setattr(ma, "_exact_tile", recording)
    monkeypatch.setattr(ma, "PAIR_TILE_BYTES", 1)
    tiny = run()
    # Two heads with N(N+1)/2 = 10 causal one-pair tiles each (the 6 tiles the prior masks everywhere are skipped);
    # every factored tile is a non-reentrant checkpoint, so it runs once in the forward pass and once more when
    # backward recomputes it.
    assert len(seen) == 2 * (2 * 10) and all(entry == (1, 1, True) for entry in seen)
    for got, want in zip(tiny, default):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_checkpointed_readout_is_bitwise_identical(inference: str, family: str) -> None:
    data = _local_inputs(82, N=4, H=2, d=2)
    sigma0 = data.sigma_diag if family == "diagonal" else data.sigma_full
    outputs = []
    for flag in (False, True):
        ops = _operators(inference, 2, 2, seed=81, checkpoint=flag)
        mu = data.mu.clone().requires_grad_(True)
        sigma = sigma0.clone().requires_grad_(True)
        mean, cov = ops.minimal_agent_update(mu, sigma, data.compact, data.log_prior, return_covariance=True)
        (mean.square().sum() + cov.sum()).backward()
        outputs.append([mean.detach(), cov.detach(), mu.grad, sigma.grad] + [p.grad for p in ops.parameters()])
    for plain, recomputed in zip(*outputs):
        assert torch.equal(plain, recomputed)


# ----------------------------------------------------------------------------------------------
# H. Configuration, construction, optimizer roles, resource counts, run record
# ----------------------------------------------------------------------------------------------

def test_config_validation_and_inactive_fields() -> None:
    cfg = test_config(**AGENT)
    assert cfg.predictive_readout == "minimal_agent" and cfg.minimal_agent_inference == "exact"
    assert cfg.minimal_agent_channel_noise == 1.0
    kernel = test_config(attention_mode="projected_predictive")
    assert kernel.predictive_readout == "kernel"                                  # the default is unchanged
    assert {"minimal_agent_inference", "minimal_agent_channel_noise"} <= set(kernel.inactive_fields())
    assert "predictive_readout" not in kernel.inactive_fields()
    assert "predictive_readout" in test_config().inactive_fields()                 # projected_gradient
    for inference in INFERENCE:
        inactive = set(test_config(**AGENT, minimal_agent_inference=inference).inactive_fields())
        assert {"relational_routing", "content_link_chart", "attention_residual", "relational_output_init",
                "kappa_beta", "projected_factor_route"} <= inactive
        assert not inactive & {"predictive_readout", "minimal_agent_inference", "minimal_agent_channel_noise"}
        assert "predictive_covariance_floor" in inactive                           # passthrough: the floor is inert
        assert "predictive_covariance_checkpoint" not in inactive
    moment = set(test_config(**AGENT, predictive_covariance="moment_matched").inactive_fields())
    assert "predictive_covariance_floor" not in moment
    prefix = "predictive_readout='minimal_agent' is unsupported here: "
    for overrides, needle in ((dict(predictive_readout="minimal_agent"), "attention_mode"),
                              (dict(**AGENT, pos_rotation="rope", rope_on_value=False), "rope"),
                              (dict(**AGENT, content_link_chart="exponential", content_link_edges="lowrank",
                                    content_link_edge_rank=2), "content_link_edges")):
        with pytest.raises(ValueError) as info:
            test_config(**overrides)
        assert str(info.value).startswith(prefix) and needle in str(info.value), str(info.value)
    test_config(**AGENT, pos_rotation="rope", rope_on_value=True)                  # one frame: accepted
    with pytest.raises(ValueError, match="predictive_readout"):
        test_config(attention_mode="projected_predictive", predictive_readout="other")
    with pytest.raises(ValueError, match="minimal_agent_inference"):
        test_config(**AGENT, minimal_agent_inference="variational")
    for bad in (0.0, -0.5, math.inf, math.nan, True, "1.0"):
        with pytest.raises(ValueError, match="minimal_agent_channel_noise"):
            test_config(**AGENT, minimal_agent_channel_noise=bad)


def test_operator_construction_parameters_and_refusals() -> None:
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
    for bad in (0.0, -1.0, math.inf, True, "1.0"):
        with pytest.raises(ValueError, match="minimal_agent_channel_noise"):
            make_relational_operators([2, 2], 1, minimal_agent_channel_noise=bad, **kwargs)
    ops = make_relational_operators([2, 2], 1, minimal_agent_channel_noise=0.5, minimal_agent_inference="mean_field",
                              **kwargs)
    assert set(ops.state_dict()) == {"raw_score_q", "T", "channel_log_noise"}
    assert ops.raw_score_k is None and ops.O is None
    assert torch.equal(ops.T, torch.eye(4).reshape(2, 2, 4)) and torch.equal(ops.raw_score_q, ops.T)
    torch.testing.assert_close(ops.channel_log_noise.detach(), torch.full((2,), math.log(0.5)))
    assert ops.inference_semantics == "relational_minimal_agent_mean_field"
    assert ops.comparison_runs() == [(ops.raw_score_q, ops.T)] and ops.comparison_pair() == (ops.raw_score_q, ops.T)
    frame = CanonicalFrameContext(torch.eye(2).expand(1, 3, 2, 2, 2), torch.eye(2).expand(1, 3, 2, 2, 2),
                                  block_diagonal=True)
    mu, beta = torch.zeros(1, 3, 4), torch.full((1, 2, 3, 3), 1 / 3)
    for call in (ops.content_factors, ops.content_factor_runs, lambda: ops._ambient_content_maps(torch.float32),
                 lambda: ops.predictive_content(mu, beta, frame),
                 lambda: ops.predictive_covariance_update(mu, torch.ones(1, 3, 4), beta, frame)):
        with pytest.raises(ValueError, match="minimal_agent"):
            call()
    kernel = make_relational_operators([2, 2], 1)
    assert set(kernel.state_dict()) == {"raw_score_q", "raw_score_k", "T", "O"}
    with pytest.raises(ValueError, match="minimal_agent"):
        kernel.minimal_agent_update(mu, torch.ones(1, 3, 4), frame)


def _agent_model(inference: str = "exact", **overrides: object) -> VFEModel:
    cfg = tiny_config(**{**AGENT, "minimal_agent_inference": inference, "max_seq_len": 4, "n_layers": 2, **overrides})
    torch.manual_seed(0)
    return VFEModel(cfg)


def test_model_state_dict_optimizer_roles_counts_and_run_record() -> None:
    torch.manual_seed(0)
    default = VFEModel(tiny_config(max_seq_len=4, n_layers=2))
    default_keys = set(default.state_dict())
    assert not any("channel_log_noise" in key for key in default_keys)
    model = _agent_model("exact")
    keys = set(model.state_dict())
    layers = ("relational_operators", "relational_operators_extra.0")
    assert keys - default_keys == {f"{p}.channel_log_noise" for p in layers}
    assert default_keys - keys == {f"{p}.{name}" for p in layers for name in ("raw_score_k", "O")}
    H, d, K, L = 2, 2, 4, 2
    for ops in model.relational_operators_by_layer:
        assert torch.equal(ops.T.detach(), torch.eye(K).reshape(H, d, K))
        assert torch.equal(ops.raw_score_q.detach(), ops.T.detach())
        assert torch.equal(ops.channel_log_noise.detach(), torch.zeros(H))          # log 1.0
    optimizer = train_module.build_optimizer(model, model.cfg)
    noise_params = {name: p for name, p in model.named_parameters() if name.endswith("channel_log_noise")}
    assert len(noise_params) == L
    for name, p in noise_params.items():
        groups = [grp for grp in optimizer.param_groups if any(q is p for q in grp["params"])]
        assert len(groups) == 1 and groups[0].get("lr_aux_role") == "relational_score"
        assert groups[0]["weight_decay"] == 0.0
        assert train_module._parameter_group(name) == "relational_score"
        assert parameter_role(name) == ("context_processing", "relational_score")
    breakdown = parameter_breakdown(dict(model.named_parameters()), model.cfg)
    assert breakdown["context_processing"]["relational_score"] == L * (H * d * K + H)
    assert breakdown["context_processing"]["relational_content"] == L * H * d * K
    record = _relational_artifact_fields(model.cfg)["relational_operators"]
    assert record["per_layer_parameter_counts"] == {"score": K * K + H, "content": K * K}
    assert record["predictive_readout"] == "minimal_agent" and record["minimal_agent_inference"] == "exact"
    assert record["minimal_agent_channel_noise_init"] == 1.0 and record["key_tied_to_content"] is True
    assert record["routing"] == "minimal_agent_exact_source_posterior"
    assert record["mean_update"] == "minimal_agent_posterior_mean"
    assert record["score_initialization"] == "query_equal_to_sending_block_projector"
    assert record["score_role_parameter_counts"] == {"q": L * K * K, "k": 0, "channel_log_noise": L * H}
    assert record["content_link_chart"] is None
    assert record["temperature_convention"] == "unit_temperature_exact_posterior_no_clamp"
    assert record["score_covariance_contract"].endswith("_plus_channel_noise")
    assert record["factorization_fallback"] == "fp32_failure_or_certificate_layer_fp64_recompute"
    assert record["routing_state"] == "pre_update_sources_adf_receiver"
    mean_field = _relational_artifact_fields(_agent_model("mean_field").cfg)["relational_operators"]
    assert mean_field["factorization_fallback"] == (
        "fp32_failure_or_certificate_layer_fp64_recompute_kl_certificate_fp64_refactor")
    assert mean_field["routing_state"] == "pre_update"
    kernel = _relational_artifact_fields(default.cfg)["relational_operators"]
    assert "predictive_readout" not in kernel and kernel["per_layer_parameter_counts"] == {"score": 2 * K * K,
                                                                                            "content": 2 * K * K}


# ----------------------------------------------------------------------------------------------
# I. Model path: finite training step, diagnostics, captured labels, causality, checkpoint
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("covariance", ["passthrough", "moment_matched"])
def test_model_step_is_finite_with_gradients_on_the_new_parameters(inference: str, covariance: str) -> None:
    model = _agent_model(inference, predictive_covariance=covariance)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 6, 2, 1]])
    _, loss, _ = model(ids, targets)
    assert math.isfinite(float(loss))
    loss.backward()
    for ops in model.relational_operators_by_layer:
        for p in (ops.raw_score_q, ops.T, ops.channel_log_noise):
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0.0


@pytest.mark.parametrize("inference", INFERENCE)
def test_model_diagnostics_and_captures(inference: str) -> None:
    from vfe3.metrics import LINK_SPECTRUM_KEYS
    model = _agent_model(inference).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    rows = model.diagnostics_per_layer(ids)
    for key in ma.MINIMAL_AGENT_DIAGNOSTIC_KEYS:
        defined = ma.MINIMAL_AGENT_MODE_ONLY_KEYS.get(key, inference) == inference   # else NaN by design
        assert len(rows[key]) == 2 and all(math.isfinite(v) == defined for v in rows[key]), key
    assert all(0.0 < v < 1.0 for v in rows["relational_minimal_agent_gain_fraction"])
    assert rows["relational_minimal_agent_channel_noise"] == [1.0, 1.0]
    assert all(v >= 0.0 for v in rows["relational_minimal_agent_head_coupling"])
    for key in ("relational_content_max_singular_value", "relational_content_effective_rank", *LINK_SPECTRUM_KEYS):
        assert all(math.isnan(v) for v in rows[key]), key
    assert rows["relational_score_klmax_frac"] == [0.0, 0.0]
    assert sorted(train_module._VAL_MINIMAL_AGENT_DIAG_KEYS) == sorted(
        "val_" + key for key in ma.MINIMAL_AGENT_DIAGNOSTIC_KEYS)
    capture: dict = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(ids, capture=capture)
    captures = capture["diagnostic"]["layer_conditional_captures"]
    assert len(captures) == 2
    causal_mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
    for cap, ops in zip(captures, model.relational_operators_by_layer):
        beta = cap["beta_pre"]
        assert beta.shape == (1, 2, 4, 4) and cap["score_energy_pre"].shape == (1, 2, 4, 4)
        torch.testing.assert_close(beta.sum(-1), torch.ones(1, 2, 4), rtol=1e-6, atol=1e-6)
        assert float(beta[..., causal_mask].abs().max()) == 0.0                   # the causal prior masks exactly
        assert bool(torch.isfinite(cap["score_energy_pre"]).all())
        for key in ("projected_covariance_pre", "projected_key_covariance_pre"):
            cov = cap[key]
            assert cov.shape == (1, 4, 2, 2, 2)
            torch.testing.assert_close(cov, cov.transpose(-1, -2))
            assert float(torch.linalg.eigvalsh(cov.double()).min()) > 0.0
        for key in ma.MINIMAL_AGENT_DIAGNOSTIC_KEYS:
            assert cap[key].shape == (2,)
        torch.testing.assert_close(cap["relational_minimal_agent_channel_noise"],
                                   ops.channel_log_noise.detach().exp())


def test_model_beliefs_are_causal_and_the_checkpoint_is_bitwise() -> None:
    model = _agent_model("exact").eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 3, 6]])
    with torch.no_grad():
        base, _ = model.forward_beliefs(ids)
        moved, _ = model.forward_beliefs(changed)
    torch.testing.assert_close(moved.mu[:, :3], base.mu[:, :3], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(moved.sigma[:, :3], base.sigma[:, :3], rtol=1e-6, atol=1e-6)
    assert float((moved.mu[:, 3] - base.mu[:, 3]).abs().max()) > 1e-4
    targets = torch.tensor([[2, 3, 4, 5]])
    outputs = []
    for flag in (False, True):
        m = _agent_model("exact", predictive_covariance="moment_matched", predictive_covariance_checkpoint=flag)
        _, loss, _ = m(ids, targets)
        loss.backward()
        outputs.append([loss.detach()] + [p.grad for p in m.parameters()])
    for plain, recomputed in zip(*outputs):
        assert (plain is None and recomputed is None) or torch.equal(plain, recomputed)
