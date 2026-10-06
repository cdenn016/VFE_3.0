r"""Prototype-agent block MLP (``block_mlp_mode='prototype_agents'``), plan 2026-10-06 Section 3.4.

The block is the minimal agent model (``minimal_agent_model.tex``, Definition ma:def:model) with
:math:`S = T = I` and :math:`M = eK` template agents :math:`p_a = \mathcal N(\nu_a, \operatorname{diag}\lambda_a)`.
Float64 checks at small K and M compare the implementation with explicit Gaussian conditioning on each
template followed by mixture moment matching (exact path) and with the explicit power posterior at the
KL labels (mean-field path), for diagonal, block and dense working covariances. These small instances
validate the implementation against exact results; they carry no information about trained models.
Tiny CPU models only, one thread.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Tuple

import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from tests.profiles import test_config
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import prototype_agents as pa
from vfe3.model.block import _reference_coordinate_block
from vfe3.model.block_mlp import (
    BlockMLP,
    GaussianTemplateBlockMLP,
    PrototypeAgentBlockMLP,
    build_block_mlp,
    get_block_mlp_registration,
)
from vfe3.model.frame_actions import local_covariance, reference_covariance_compact
from vfe3.model.model import VFEModel

DT = torch.float64
LAYOUTS = ("diagonal", "blocks", "dense")
INFERENCE = ("mean_field", "exact")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# Explicit references (per token, dense float64 linear algebra)
# ----------------------------------------------------------------------------------------------

def _templates(K: int, M: int, seed: int, scale: float = 0.5) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    nu = torch.randn(M, K, generator=g, dtype=DT)
    log_lambda = scale * torch.randn(M, K, generator=g, dtype=DT)
    log_pi = 0.5 * torch.randn(M, generator=g, dtype=DT)
    return nu, log_lambda, log_pi


def _spd(shape: Tuple[int, ...], k: int, seed: int) -> torch.Tensor:
    A = torch.randn(*shape, k, k, generator=torch.Generator().manual_seed(seed), dtype=DT)
    return A @ A.transpose(-1, -2) + 0.3 * torch.eye(k, dtype=DT)


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:          # (..., G, b, b) -> (..., K, K)
    G, b = blocks.shape[-3], blocks.shape[-1]
    out = torch.zeros(*blocks.shape[:-3], G * b, G * b, dtype=blocks.dtype)
    for g in range(G):
        out[..., g * b:(g + 1) * b, g * b:(g + 1) * b] = blocks[..., g, :, :]
    return out


def _working(layout: str, R: int, K: int, G: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""A working covariance in ``layout`` and its dense equivalent, for R tokens."""
    if layout == "diagonal":
        c = torch.rand(R, K, generator=torch.Generator().manual_seed(seed), dtype=DT) + 0.5
        return c, torch.diag_embed(c)
    if layout == "blocks":
        blocks = _spd((R, G), K // G, seed)
        return blocks, _block_diag(blocks)
    dense = _spd((R,), K, seed)
    return dense, dense


def _project(cov: torch.Tensor, layout: str, G: int) -> torch.Tensor:
    r"""The member of the layout's family read from a dense (K, K) matrix: diagonal, diagonal blocks, or itself."""
    if layout == "diagonal":
        return torch.diagonal(cov, dim1=-2, dim2=-1)
    if layout == "blocks":
        b = cov.shape[-1] // G
        return torch.stack([cov[..., g * b:(g + 1) * b, g * b:(g + 1) * b] for g in range(G)], dim=-3)
    return cov


def _explicit_exact(xi: torch.Tensor, C: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor,
                    log_pi: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""Condition on each template, w_a from explicit densities, then mixture moments; also (u, Lambda_bar, R)."""
    log_prior = torch.log_softmax(log_pi, dim=0)
    logw, means, covs, rs, W_invs = [], [], [], [], []
    for a in range(nu.shape[0]):
        W = C + torch.diag(lam[a])
        W_inv = torch.linalg.inv(W)
        gain = C @ W_inv
        means.append(xi + gain @ (nu[a] - xi))                   # m_ia
        covs.append(C - gain @ C)                                 # P_ia
        rs.append(W_inv @ (nu[a] - xi))
        W_invs.append(W_inv)
        logw.append(log_prior[a] + MultivariateNormal(xi, covariance_matrix=W).log_prob(nu[a]))
    w = torch.softmax(torch.stack(logw), dim=0)
    mean = sum(w[a] * means[a] for a in range(len(means)))
    cov = sum(w[a] * (covs[a] + torch.outer(means[a] - mean, means[a] - mean)) for a in range(len(means)))
    u = sum(w[a] * rs[a] for a in range(len(rs)))
    Lambda_bar = sum(w[a] * W_invs[a] for a in range(len(rs)))
    R = sum(w[a] * torch.outer(rs[a], rs[a]) for a in range(len(rs))) - torch.outer(u, u)
    return dict(mean=mean, cov=cov, w=w, u=u, Lambda_bar=Lambda_bar, R=R)


def _explicit_mean_field(xi: torch.Tensor, C: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor,
                         log_pi: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""KL labels at the prior, then the power posterior P = (C^{-1} + diag D)^{-1}."""
    q = MultivariateNormal(xi, covariance_matrix=C)
    kl = torch.stack([kl_divergence(q, MultivariateNormal(nu[a], covariance_matrix=torch.diag(lam[a])))
                      for a in range(nu.shape[0])])
    r = torch.softmax(torch.log_softmax(log_pi, dim=0) - kl, dim=0)
    D = (r[:, None] / lam).sum(0)
    eta = (r[:, None] * nu / lam).sum(0)
    P = torch.linalg.inv(torch.linalg.inv(C) + torch.diag(D))
    return dict(mean=P @ (torch.linalg.solve(C, xi) + eta), cov=P, r=r, D=D, eta=eta)


# ----------------------------------------------------------------------------------------------
# 1. Exact path against explicit conditioning and mixture moments
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_exact_posterior_equals_explicit_conditioning_and_mixture_moments(layout: str) -> None:
    R, K, G, M = 4, 6, 2, 5
    nu, log_lambda, log_pi = _templates(K, M, seed=1)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=torch.Generator().manual_seed(2), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=3)
    out = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True, need_precision=True)
    assert out.C_plus.shape == C.shape and out.R.shape == C.shape and out.Lambda_bar.shape == C.shape
    for i in range(R):
        ref = _explicit_exact(xi[i], Cd[i], nu, lam, log_pi)
        assert (out.xi_plus[i] - ref["mean"]).abs().max() < 1e-10
        assert (out.weights[i] - ref["w"]).abs().max() < 1e-10
        assert (out.C_plus[i] - _project(ref["cov"], layout, G)).abs().max() < 1e-10
        # The (u, Lambda_bar, R) form equals the mixture form.
        assert (out.u[i] - ref["u"]).abs().max() < 1e-10
        assert (out.Lambda_bar[i] - _project(ref["Lambda_bar"], layout, G)).abs().max() < 1e-10
        assert (out.R[i] - _project(ref["R"], layout, G)).abs().max() < 1e-10
        assert (xi[i] + Cd[i] @ ref["u"] - ref["mean"]).abs().max() < 1e-10
        spec_form = Cd[i] - Cd[i] @ (ref["Lambda_bar"] - ref["R"]) @ Cd[i]      # C - C(Lambda_bar - R)C
        assert (spec_form - ref["cov"]).abs().max() < 1e-10
    assert (out.xi_plus - xi - out.delta_xi).abs().max() < 1e-14


@pytest.mark.parametrize("layout", LAYOUTS)
def test_exact_covariance_uses_the_cancellation_free_form(layout: str) -> None:
    r"""A sharp template (lambda / c = 1e-7) pins the belief; C Psi + C R C stays positive in float32."""
    R, K, G = 3, 4, 2
    nu = torch.zeros(2, K, dtype=DT)
    nu[1] += 3.0
    log_lambda = torch.full((2, K), math.log(1e-6), dtype=DT)
    log_pi = torch.tensor([8.0, 0.0], dtype=DT)
    xi = torch.full((R, K), 0.1, dtype=DT)
    C, _ = _working(layout, R, K, G, seed=4)
    C = 10.0 * C
    out32 = pa.prototype_exact_posterior(xi.float(), C.float(), nu.float(), log_lambda.float(), log_pi.float())
    out64 = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi)
    diag32 = pa.covariance_diagonal(out32.C_plus, layout)
    assert out32.C_plus.dtype == torch.float32 and bool((diag32 > 0).all())
    rel = (diag32.double() - pa.covariance_diagonal(out64.C_plus, layout)).abs() \
        / pa.covariance_diagonal(out64.C_plus, layout)
    assert rel.max() < 1e-3


# ----------------------------------------------------------------------------------------------
# 2. Mean-field path against the explicit power posterior; responsibilities = gaussian_template
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_mean_field_posterior_equals_explicit_power_posterior(layout: str) -> None:
    R, K, G, M = 4, 6, 3, 5
    nu, log_lambda, log_pi = _templates(K, M, seed=5)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=torch.Generator().manual_seed(6), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=7)
    out = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi)
    assert out.C_plus.shape == C.shape
    for i in range(R):
        ref = _explicit_mean_field(xi[i], Cd[i], nu, lam, log_pi)
        assert (out.xi_plus[i] - ref["mean"]).abs().max() < 1e-10
        assert (out.C_plus[i] - _project(ref["cov"], layout, G)).abs().max() < 1e-10
        assert (out.responsibilities[i] - ref["r"]).abs().max() < 1e-10
        assert (out.D[i] - ref["D"]).abs().max() < 1e-10 and (out.eta[i] - ref["eta"]).abs().max() < 1e-10


@pytest.mark.parametrize("full", [False, True])
def test_mean_field_responsibilities_equal_gaussian_template(full: bool) -> None:
    K, e = 3, 2
    torch.manual_seed(0)
    template = GaussianTemplateBlockMLP(K, e, 0.0).double()
    prototype = PrototypeAgentBlockMLP(K, e, 0.0, inference="mean_field").double()
    with torch.no_grad():
        template.nu.normal_(0.0, 1.0)
        template.log_lambda.normal_(0.0, 0.5)
        template.log_pi.normal_(0.0, 0.5)
        for name in ("nu", "log_lambda", "log_pi"):
            getattr(prototype, name).copy_(getattr(template, name))
    mu = torch.randn(2, 4, K, dtype=DT)
    sigma = _spd((2, 4), K, seed=8) if full else torch.rand(2, 4, K, dtype=DT) + 0.5
    expected = template.responsibilities(mu, sigma)
    got = prototype.posterior(mu, sigma).responsibilities
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=1e-15)


def test_template_kl_logits_ignore_bf16_autocast() -> None:
    r"""Called directly under bf16 autocast, the label logits still run their three GEMMs in float32."""
    K, M = 8, 16
    nu, log_lambda, log_pi = (t.float() for t in _templates(K, M, seed=9))
    g = torch.Generator().manual_seed(10)
    xi = torch.randn(5, K, generator=g)
    c = torch.rand(5, K, generator=g) + 0.5
    expected = pa.template_kl_logits(xi, c, nu, log_lambda, log_pi)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = pa.template_kl_logits(xi, c, nu, log_lambda, log_pi)
    assert got.dtype == torch.float32 and torch.equal(got, expected)


# ----------------------------------------------------------------------------------------------
# 3. One template: exact and mean-field agree (zero spread)
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_single_template_exact_equals_mean_field(layout: str) -> None:
    R, K, G = 3, 4, 2
    nu, log_lambda, log_pi = _templates(K, 1, seed=9)
    xi = torch.randn(R, K, generator=torch.Generator().manual_seed(10), dtype=DT)
    C, _ = _working(layout, R, K, G, seed=11)
    exact = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True)
    mean_field = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi)
    assert (exact.xi_plus - mean_field.xi_plus).abs().max() < 1e-12
    assert exact.R.abs().max() < 1e-13                                         # zero between-template spread
    assert (exact.C_plus - mean_field.C_plus).abs().max() < 1e-12
    assert torch.equal(exact.weights, torch.ones_like(exact.weights))


# ----------------------------------------------------------------------------------------------
# 4. Limits: broad templates give the identity; a sharp dominant template is retrieved
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("inference", INFERENCE)
def test_broad_and_sharp_template_limits(layout: str, inference: str) -> None:
    r"""Both limits are approached at first order: the deviation shrinks 100-fold when lambda moves 100-fold."""
    R, K, G, M = 3, 4, 2, 3
    posterior = pa.prototype_exact_posterior if inference == "exact" else pa.prototype_mean_field_posterior
    xi = torch.randn(R, K, generator=torch.Generator().manual_seed(12), dtype=DT)
    C, _ = _working(layout, R, K, G, seed=13)
    nu, _, _ = _templates(K, M, seed=14)

    def run(lam: float, templates: torch.Tensor, log_pi: torch.Tensor):
        return posterior(xi, C, templates, torch.full((M, K), math.log(lam), dtype=DT), log_pi)

    # lambda -> inf: the identity, xi^+ - xi = O(1/lambda) and C^+ - C = -C D C = O(1/lambda).
    broad = [run(lam, nu, torch.zeros(M, dtype=DT)) for lam in (1e8, 1e10)]
    deviation = [max(float((b.xi_plus - xi).abs().max()), float((b.C_plus - C).abs().max())) for b in broad]
    assert deviation[1] < 1e-7 and deviation[1] < 0.02 * deviation[0]
    # lambda -> 0 with one dominant template (nearest and favored a priori): xi^+ -> nu_0, C^+ -> 0.
    nu = nu.clone()
    nu[0] = xi.mean(0) + 0.05
    nu[1:] += 5.0
    log_pi = torch.tensor([40.0, 0.0, 0.0], dtype=DT)
    sharp = [run(lam, nu, log_pi) for lam in (1e-8, 1e-10)]
    error = [float((s.xi_plus - nu[0]).abs().max()) for s in sharp]
    assert error[1] < 1e-7 and error[1] < 0.02 * error[0]
    assert pa.covariance_diagonal(sharp[1].C_plus, layout).max() < 1e-9


# ----------------------------------------------------------------------------------------------
# 5. Covariance contract, dropout and module wiring
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("inference", INFERENCE)
def test_covariance_contract(layout: str, inference: str) -> None:
    K, e, G = 4, 1, 2
    torch.manual_seed(0)
    passthrough = PrototypeAgentBlockMLP(K, e, 0.0, inference=inference).double()
    posterior = PrototypeAgentBlockMLP(K, e, 0.0, inference=inference, covariance_contract="posterior").double()
    posterior.load_state_dict(passthrough.state_dict())
    mu = torch.randn(2, 3, K, dtype=DT)
    C, Cd = _working(layout, 6, K, G, seed=15)
    sigma = C.reshape(2, 3, *C.shape[1:])
    kept = passthrough.forward_moments(mu, sigma)
    assert kept.sigma is sigma and kept.jacobian is None
    new = posterior.forward_moments(mu, sigma)
    assert new.sigma.shape == sigma.shape and new.sigma.dtype == sigma.dtype
    if layout == "diagonal":
        assert bool((new.sigma > 0).all())
    else:
        assert torch.equal(new.sigma, new.sigma.transpose(-1, -2))
        assert torch.linalg.eigvalsh(new.sigma).min() > 0
    pure = (pa.prototype_exact_posterior if inference == "exact" else pa.prototype_mean_field_posterior)(
        mu, sigma, posterior.nu, posterior.log_lambda, posterior.log_pi)
    assert torch.equal(new.mu, kept.mu) and (new.mu - pure.xi_plus).abs().max() < 1e-14
    assert (new.sigma - pure.C_plus).abs().max() < 1e-14
    with pytest.raises(ValueError, match="hidden_bias does not apply.*template agents"):
        posterior.forward_moments(mu, sigma, hidden_bias=torch.zeros(2, 3, 1, dtype=DT))


def test_invalid_module_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="covariance_contract"):
        PrototypeAgentBlockMLP(4, 1, 0.0, covariance_contract="delta_full")
    with pytest.raises(ValueError, match="inference"):
        PrototypeAgentBlockMLP(4, 1, 0.0, inference="variational")
    for bad in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="template_init_variance"):
            PrototypeAgentBlockMLP(4, 1, 0.0, template_init_variance=bad)
    with pytest.raises(ValueError, match="working covariance"):
        pa.prototype_mean_field_posterior(torch.zeros(2, 4), torch.ones(2, 3), torch.zeros(1, 4),
                                          torch.zeros(1, 4), torch.zeros(1))


def test_dropout_acts_on_the_mean_increment_only() -> None:
    K = 4
    torch.manual_seed(0)
    mlp = PrototypeAgentBlockMLP(K, 2, 0.5, covariance_contract="posterior").double()
    mu = torch.randn(8, K, dtype=DT)
    sigma = torch.rand(8, K, dtype=DT) + 0.5
    pure = pa.prototype_mean_field_posterior(mu, sigma, mlp.nu, mlp.log_lambda, mlp.log_pi)
    mlp.eval()
    evaluated = mlp.forward_moments(mu, sigma)
    assert torch.equal(evaluated.mu, mu + pure.delta_xi)
    mlp.train()
    torch.manual_seed(1)
    trained = mlp.forward_moments(mu, sigma)
    increment = trained.mu - mu
    kept = (increment - 2.0 * pure.delta_xi).abs() < 1e-14
    dropped = increment.abs() < 1e-15
    assert bool((kept | dropped).all()) and bool(kept.any()) and bool(dropped.any())
    assert torch.equal(trained.sigma, evaluated.sigma)                         # covariance is not dropped


def test_statistics_at_initialization() -> None:
    K, e = 4, 2
    for inference in INFERENCE:
        torch.manual_seed(0)
        mlp = PrototypeAgentBlockMLP(K, e, 0.0, inference=inference).double()
        stats = mlp.statistics(torch.randn(5, K, dtype=DT), torch.ones(5, K, dtype=DT))
        # c = 1, lambda = lambda_0: c / (c + lambda_0) for every template and coordinate, in both modes;
        # lambda_0 is the float32 initialization log(10) cast to float64, 10 (1 +- 6e-8).
        lam = float(mlp.log_lambda.exp().mean())
        assert torch.equal(mlp.log_lambda, torch.full_like(mlp.log_lambda, math.log(lam)))
        assert abs(stats["prototype_gain_fraction"] - 1.0 / (1.0 + lam)) < 1e-12
        assert abs(stats["prototype_gain_fraction"] - 1.0 / 11.0) < 1e-8
        assert abs(stats["prototype_template_variance"] - 10.0) < 1e-6
        assert abs(stats["prototype_prior_entropy"] - math.log(K * e)) < 1e-12
        assert 1.0 <= stats["prototype_effective_templates"] <= K * e
        dense = mlp.statistics(torch.randn(5, K, dtype=DT), torch.eye(K, dtype=DT).expand(5, K, K))
        assert abs(dense["prototype_gain_fraction"] - 1.0 / (1.0 + lam)) < 1e-12


# ----------------------------------------------------------------------------------------------
# 6. Reference coordinates: hand-built conjugation and passive covariance
# ----------------------------------------------------------------------------------------------

B, N, G, D = 2, 3, 2, 2
K_REF = G * D


def _frame(kind: str, seed: int, scale: float = 0.4) -> Tuple[CanonicalFrameContext, torch.Tensor]:
    r"""A realized frame (compact blocks or dense) and its dense matrix U (B, N, K, K)."""
    g = torch.Generator().manual_seed(seed)
    if kind == "blocks":
        blocks = torch.linalg.matrix_exp(scale * torch.randn(B, N, G, D, D, generator=g, dtype=DT))
        return CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True), _block_diag(blocks)
    U = torch.linalg.matrix_exp(scale * torch.randn(B, N, K_REF, K_REF, generator=g, dtype=DT))
    return CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=False), U


def _reference_block(mlp: PrototypeAgentBlockMLP, frame: CanonicalFrameContext, mu: torch.Tensor,
                     sigma: torch.Tensor) -> BeliefState:
    belief = BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(B, N, 1, dtype=DT))
    return _reference_coordinate_block(belief, block_norm=None, block_mlp=mlp, frame=frame, capture=None,
                                       hidden_bias=None)


def _ref_mlp(inference: str, contract: str) -> PrototypeAgentBlockMLP:
    torch.manual_seed(3)
    mlp = PrototypeAgentBlockMLP(K_REF, 2, 0.0, inference=inference, covariance_contract=contract).double()
    with torch.no_grad():
        mlp.nu.normal_(0.0, 1.0)
        mlp.log_lambda.normal_(0.0, 0.5)
        mlp.log_pi.normal_(0.0, 0.5)
    return mlp


@pytest.mark.parametrize("frame_kind", ["blocks", "dense"])
@pytest.mark.parametrize("beliefs", ["diagonal", "full"])
@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("contract", ["passthrough", "posterior"])
def test_reference_block_equals_hand_built_conjugation(frame_kind: str, beliefs: str, inference: str,
                                                       contract: str) -> None:
    r""":math:`\mu^+ = U\xi^+`, :math:`\Sigma^+ = UC^+U^\top` from dense float64 :math:`\xi = U^{-1}\mu`, :math:`C = U^{-1}\Sigma U^{-\top}`."""
    mlp = _ref_mlp(inference, contract)
    frame, U = _frame(frame_kind, seed=20)
    mu = torch.randn(B, N, K_REF, generator=torch.Generator().manual_seed(21), dtype=DT)
    sigma = (_spd((B, N), K_REF, seed=22) if beliefs == "full"
             else torch.rand(B, N, K_REF, generator=torch.Generator().manual_seed(22), dtype=DT) + 0.5)
    Sigma = sigma if beliefs == "full" else torch.diag_embed(sigma)
    U_inv = torch.linalg.inv(U)
    xi = (U_inv @ mu.unsqueeze(-1)).squeeze(-1)
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)
    pure = (pa.prototype_exact_posterior if inference == "exact" else pa.prototype_mean_field_posterior)(
        xi, C, mlp.nu, mlp.log_lambda, mlp.log_pi)
    out = _reference_block(mlp, frame, mu, sigma)
    assert (out.mu - (U @ pure.xi_plus.unsqueeze(-1)).squeeze(-1)).abs().max() < 1e-10
    if contract == "passthrough":
        assert out.sigma is sigma
    else:
        Sigma_plus = U @ pure.C_plus @ U.transpose(-1, -2)
        expected = Sigma_plus if beliefs == "full" else torch.diagonal(Sigma_plus, dim1=-2, dim2=-1)
        assert out.sigma.shape == sigma.shape
        assert (out.sigma - expected).abs().max() < 1e-10


@pytest.mark.parametrize("frame_kind", ["blocks", "dense"])
@pytest.mark.parametrize("inference", INFERENCE)
def test_reference_block_passive_covariance(frame_kind: str, inference: str) -> None:
    r""":math:`(\mu, \Sigma, U) \to (g\mu, g\Sigma g^\top, gU)` maps :math:`(\mu^+, \Sigma^+) \to (g\mu^+, g\Sigma^+g^\top)`."""
    mlp = _ref_mlp(inference, "posterior")
    frame, U = _frame(frame_kind, seed=30)
    g_frame, g = _frame(frame_kind, seed=31, scale=0.6)                        # g in prod GL(d) or GL(K)
    mu = torch.randn(B, N, K_REF, generator=torch.Generator().manual_seed(32), dtype=DT)
    Sigma = _spd((B, N), K_REF, seed=33)
    moved_frame = (CanonicalFrameContext(g_frame.forward @ frame.forward, frame.inverse @ g_frame.inverse,
                                         block_diagonal=frame.block_diagonal))
    gmv = lambda x: (g @ x.unsqueeze(-1)).squeeze(-1)                          # noqa: E731
    base = _reference_block(mlp, frame, mu, Sigma)
    moved = _reference_block(mlp, moved_frame, gmv(mu), g @ Sigma @ g.transpose(-1, -2))
    assert (moved.mu - gmv(base.mu)).abs().max() < 1e-9
    assert (moved.sigma - g @ base.sigma @ g.transpose(-1, -2)).abs().max() < 1e-9
    local = PrototypeAgentBlockMLP(K_REF, 2, 0.0, inference=inference, covariance_contract="posterior").double()
    local.load_state_dict(mlp.state_dict())
    local_moved = local.forward_moments(gmv(mu), g @ Sigma @ g.transpose(-1, -2)).mu
    assert (local_moved - gmv(local.forward_moments(mu, Sigma).mu)).abs().max() > 1e-3   # negative control


@pytest.mark.parametrize("frame_kind", ["blocks", "dense"])
def test_frame_helpers_match_dense_formulas(frame_kind: str) -> None:
    frame, U = _frame(frame_kind, seed=40)
    U_inv = torch.linalg.inv(U)
    mu = torch.zeros(B, N, K_REF, dtype=DT)
    sigma = torch.rand(B, N, K_REF, dtype=DT) + 0.5
    C = U_inv @ torch.diag_embed(sigma) @ U_inv.transpose(-1, -2)
    compact = reference_covariance_compact(mu, sigma, frame)
    if frame_kind == "blocks":
        assert compact.shape == (B, N, G, D, D)
        assert (_block_diag(compact) - C).abs().max() < 1e-12
    else:
        assert (compact - C).abs().max() < 1e-12
    full = _spd((B, N), K_REF, seed=41)
    assert (reference_covariance_compact(mu, full, frame) - U_inv @ full @ U_inv.transpose(-1, -2)).abs().max() < 1e-12
    for C_ref in (compact, full):
        dense_C = _block_diag(C_ref) if C_ref.dim() == mu.dim() + 2 else C_ref
        Sigma = U @ dense_C @ U.transpose(-1, -2)
        assert (local_covariance(mu, C_ref, frame, diagonal=False) - Sigma).abs().max() < 1e-12
        assert (local_covariance(mu, C_ref, frame, diagonal=True)
                - torch.diagonal(Sigma, dim1=-2, dim2=-1)).abs().max() < 1e-12
    assert (local_covariance(mu, compact, frame, diagonal=True) - sigma).abs().max() < 1e-12   # round trip


# ----------------------------------------------------------------------------------------------
# 7. Finite-difference gradients and the chunk boundary
# ----------------------------------------------------------------------------------------------

def _gradcheck_inputs(layout: str, seed: int) -> Tuple[torch.Tensor, ...]:
    R, K, M = 2, 4, 3
    nu, log_lambda, log_pi = _templates(K, M, seed=seed, scale=0.3)
    xi = torch.randn(R, K, generator=torch.Generator().manual_seed(seed + 1), dtype=DT)
    if layout == "diagonal":
        cov_param = torch.rand(R, K, generator=torch.Generator().manual_seed(seed + 2), dtype=DT) + 0.5
    elif layout == "blocks":
        cov_param = 0.5 * torch.randn(R, 2, 2, 2, generator=torch.Generator().manual_seed(seed + 2), dtype=DT)
    else:
        cov_param = 0.5 * torch.randn(R, K, K, generator=torch.Generator().manual_seed(seed + 2), dtype=DT)
    return tuple(t.requires_grad_(True) for t in (nu, log_lambda, log_pi, xi, cov_param))


def _covariance_from(layout: str, cov_param: torch.Tensor) -> torch.Tensor:
    if layout == "diagonal":
        return cov_param
    return cov_param @ cov_param.transpose(-1, -2) + 0.5 * torch.eye(cov_param.shape[-1], dtype=DT)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("inference", INFERENCE)
def test_finite_difference_gradients(layout: str, inference: str) -> None:
    posterior = pa.prototype_exact_posterior if inference == "exact" else pa.prototype_mean_field_posterior

    def f(nu, log_lambda, log_pi, xi, cov_param):
        out = posterior(xi, _covariance_from(layout, cov_param), nu, log_lambda, log_pi)
        return out.xi_plus, out.C_plus

    assert torch.autograd.gradcheck(f, _gradcheck_inputs(layout, seed=50), eps=1e-6, atol=1e-6)


class _Moments(torch.nn.Module):
    r"""Expose both outputs of ``forward_moments`` to ``torch.func.functional_call``."""

    def __init__(self, inner: PrototypeAgentBlockMLP) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.inner.forward_moments(mu, sigma)
        return result.mu, result.sigma


@pytest.mark.parametrize("inference", INFERENCE)
def test_module_finite_difference_gradients_through_parameters(inference: str) -> None:
    torch.manual_seed(0)
    module = _Moments(PrototypeAgentBlockMLP(2, 2, 0.0, inference=inference,
                                             covariance_contract="posterior").double())
    names = [name for name, _ in module.named_parameters()]
    assert names == ["inner.nu", "inner.log_lambda", "inner.log_pi"]
    params = tuple(p.detach().clone().requires_grad_(True) for p in module.parameters())
    mu = torch.randn(3, 2, dtype=DT, requires_grad=True)
    sigma = (torch.rand(3, 2, dtype=DT) + 0.5).requires_grad_(True)

    def f(mu_in, sigma_in, *values):
        return torch.func.functional_call(module, dict(zip(names, values)), (mu_in, sigma_in))

    assert torch.autograd.gradcheck(f, (mu, sigma) + params, eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_chunk_boundary_gives_identical_values_and_gradients(layout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Tiny tiles (several template chunks and row tiles, checkpointed) equal the single-tile result.

    Equality is to rounding (1e-12 in float64), not bitwise: the online merge sums in a different order.
    """
    calls = {"n": 0}
    real_checkpoint = pa.checkpoint

    def counting_checkpoint(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(pa, "checkpoint", counting_checkpoint)
    R, K, G, M = 5, 4, 2, 7
    nu, log_lambda, log_pi = _templates(K, M, seed=60)
    xi = torch.randn(R, K, generator=torch.Generator().manual_seed(61), dtype=DT)
    C, _ = _working(layout, R, K, G, seed=62)
    inputs = [t.clone().requires_grad_(True) for t in (xi, C, nu, log_lambda, log_pi)]
    results = []
    for tile_bytes in (pa.PROTOTYPE_TILE_BYTES, 1):
        leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
        before = calls["n"]
        out = pa.prototype_exact_posterior(*leaves, need_covariance=True, need_precision=True,
                                           tile_bytes=tile_bytes)
        tiles = calls["n"] - before
        loss = (out.xi_plus.square().sum() + out.C_plus.sum() + (out.weights * torch.arange(M, dtype=DT)).sum()
                + out.Lambda_bar.sum() + out.R.sum())
        grads = torch.autograd.grad(loss, leaves)
        results.append((out, grads, tiles))
    (single, single_grads, single_tiles), (tiled, tiled_grads, tiled_tiles) = results
    assert single_tiles == 0 and tiled_tiles == R * M                         # one pair per checkpointed tile
    for field in ("xi_plus", "C_plus", "weights", "u", "Lambda_bar", "R"):
        assert (getattr(single, field) - getattr(tiled, field)).abs().max() < 1e-12, field
    for a, b in zip(single_grads, tiled_grads):
        assert (a - b).abs().max() < 1e-12


def test_factorization_failure_is_raised_not_repaired() -> None:
    xi = torch.zeros(1, 2, dtype=DT)
    bad = torch.tensor([[[1.0, 0.0], [0.0, -5.0]]], dtype=DT)                  # not positive semidefinite
    with pytest.raises(ValueError, match="not positive definite"):
        pa.prototype_exact_posterior(xi, bad, torch.zeros(1, 2, dtype=DT), torch.zeros(1, 2, dtype=DT),
                                     torch.zeros(1, dtype=DT))
    with pytest.raises(ValueError, match="positive and finite"):
        pa.prototype_exact_posterior(xi, torch.tensor([[1.0, -5.0]], dtype=DT), torch.zeros(1, 2, dtype=DT),
                                     torch.zeros(1, 2, dtype=DT), torch.zeros(1, dtype=DT))


# ----------------------------------------------------------------------------------------------
# 8. Configuration, registry, optimizer, resources, state dict; existing modes unchanged
# ----------------------------------------------------------------------------------------------

def _cfg(**over: Any):
    values: Dict[str, Any] = dict(attention_mode="projected_predictive", block_mlp_mode="prototype_agents",
                                  block_mlp_expansion=2, n_layers=2, max_seq_len=4, attention_residual=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def test_config_validation() -> None:
    _cfg()
    _cfg(block_mlp_covariance="posterior", block_mlp_inference="exact", block_mlp_template_init_variance=2.5)
    _cfg(block_coordinates="reference", block_mlp_covariance="posterior")
    with pytest.raises(ValueError, match="block_mlp_inference"):
        _cfg(block_mlp_inference="variational")
    for bad in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="block_mlp_template_init_variance"):
            _cfg(block_mlp_template_init_variance=bad)
    for mode in ("coordinate", "gaussian_template", "gauge_gate"):
        with pytest.raises(ValueError, match="block_mlp_covariance"):
            _cfg(block_mlp_mode=mode, block_mlp_covariance="posterior", family="gaussian_full",
                 decode_mode="full_chunked")
    with pytest.raises(ValueError, match="block_mlp_covariance"):
        _cfg(block_mlp_covariance="delta_full", family="gaussian_full", decode_mode="full_chunked")
    with pytest.raises(ValueError, match="per_type_hidden_bias"):
        _cfg(per_type_hidden_bias=True)
    with pytest.raises(ValueError, match="block_mlp_covariance"):                # reference still refuses delta_full
        _cfg(block_coordinates="reference", block_mlp_mode="coordinate", block_mlp_covariance="delta_full",
             family="gaussian_full", decode_mode="full_chunked")


def test_inactive_fields() -> None:
    knobs = {"block_mlp_inference", "block_mlp_template_init_variance"}
    assert knobs <= set(test_config().inactive_fields())                       # default coordinate mode
    assert knobs <= set(_cfg(block_mlp_mode="gaussian_template").inactive_fields())
    assert knobs <= set(_cfg(use_block_mlp=False).inactive_fields())
    active = _cfg().inactive_fields()
    assert not knobs & set(active)
    assert {"block_mlp_activation", "block_mlp_covariance_floor"} <= set(active)
    assert "block_mlp_expansion" not in active and "block_mlp_dropout" not in active
    assert "family" not in _cfg(block_mlp_covariance="posterior", family="gaussian_full",
                                decode_mode="full_chunked").inactive_fields()


def test_registry_parameter_count_flops_and_labels() -> None:
    registration = get_block_mlp_registration("prototype_agents")
    assert registration.covariance_kinds == frozenset({"passthrough", "posterior"})
    assert registration.gauge_class == "coordinate_nonintertwiner" and not registration.intertwiner_compatible
    for other in ("coordinate", "gauge_gate", "canonical_frame", "gaussian_template"):
        assert "posterior" not in get_block_mlp_registration(other).covariance_kinds
        assert get_block_mlp_registration(other).factory_options == frozenset()
    for contract, label in (("passthrough", "prototype_agents_mean_only_nonintertwiner"),
                            ("posterior", "prototype_agents_posterior_covariance_nonintertwiner")):
        cfg = _cfg(block_mlp_covariance=contract)
        model = _model(cfg)
        metadata = model.executable_build.block_mlp
        realized = sum(p.numel() for p in model.block_mlps.parameters())
        K, e = cfg.embed_dim, cfg.block_mlp_expansion
        assert metadata.parameter_count == realized == cfg.n_layers * (2 * e * K * K + e * K)
        assert metadata.flops_per_token == cfg.n_layers * (8.0 * e * K * K + 7.0 * K)    # diagonal family
        assert metadata.report_label == label and metadata.covariance == contract
    full = _model(_cfg(family="gaussian_full", decode_mode="full_chunked")).executable_build.block_mlp
    K, e = 4, 2
    assert full.flops_per_token == 2 * (8.0 * e * K * K + K ** 3 / 3.0 + 6.0 * K ** 2)
    # posterior on the full layout adds the K-right-hand-side Cholesky solve and scalings of P (2 K^3 + 4 K^2)
    full_posterior = _model(_cfg(family="gaussian_full", decode_mode="full_chunked",
                                 block_mlp_covariance="posterior")).executable_build.block_mlp
    assert full_posterior.flops_per_token == 2 * (8.0 * e * K * K + K ** 3 / 3.0 + 6.0 * K ** 2
                                                  + (2.0 * K ** 3 + 4.0 * K ** 2))


def test_module_construction_and_initialization() -> None:
    cfg = _cfg(block_mlp_inference="exact", block_mlp_template_init_variance=3.0, block_mlp_covariance="posterior")
    model = _model(cfg)
    for mlp in model.block_mlps:
        assert type(mlp) is PrototypeAgentBlockMLP
        assert mlp.inference == "exact" and mlp.covariance_contract == "posterior"
        M, K = cfg.block_mlp_expansion * cfg.embed_dim, cfg.embed_dim
        assert mlp.nu.shape == (M, K) and mlp.log_lambda.shape == (M, K) and mlp.log_pi.shape == (M,)
        assert torch.equal(mlp.log_lambda, torch.full((M, K), math.log(3.0)))
        assert torch.equal(mlp.log_pi, torch.zeros(M))
        assert mlp.nu.abs().max() <= K ** -0.5
    keys = sorted(k for k in model.state_dict() if k.startswith("block_mlps."))
    assert keys == sorted(f"block_mlps.{layer}.{name}" for layer in range(cfg.n_layers)
                          for name in ("nu", "log_lambda", "log_pi"))


def test_optimizer_grouping_and_resource_accounting() -> None:
    from vfe3 import train as T
    from vfe3.run_resources import parameter_breakdown

    cfg = _cfg()
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)                                  # coverage guard must stay green
    groups = [g for g in optimizer.param_groups if g.get("lr_aux_role") == "block_mlp"]
    assert len(groups) == 1
    # The template means keep the global decay; log(lambda) and log(pi) train at the same rate without decay.
    assert {id(p) for p in groups[0]["params"]} == {id(mlp.nu) for mlp in model.block_mlps}
    assert groups[0]["weight_decay"] == cfg.weight_decay > 0.0 and groups[0]["lr"] == cfg.m_p_mu_lr
    priors = [g for g in optimizer.param_groups
              if {id(p) for p in g["params"]} & {id(mlp.log_pi) for mlp in model.block_mlps}]
    assert len(priors) == 1 and priors[0].get("lr_aux_role") is None and priors[0]["role"] == "mu"
    assert {id(p) for p in priors[0]["params"]} == {id(p) for mlp in model.block_mlps for p in (mlp.log_lambda,
                                                                                               mlp.log_pi)}
    assert priors[0]["weight_decay"] == 0.0 and priors[0]["lr"] == cfg.m_p_mu_lr
    assert all(T._parameter_group(name) == "block_mlp"
               for name, _ in model.named_parameters() if name.startswith("block_mlps."))
    counts = parameter_breakdown(dict(model.named_parameters()), cfg)
    assert counts["context_processing"]["block_mlp"] == model.executable_build.block_mlp.parameter_count


def test_state_dict_round_trip() -> None:
    cfg = _cfg(block_mlp_covariance="posterior")
    source = _model(cfg, seed=0)
    with torch.no_grad():
        for mlp in source.block_mlps:
            mlp.log_lambda.add_(0.3 * torch.randn_like(mlp.log_lambda))
            mlp.log_pi.add_(0.3 * torch.randn_like(mlp.log_pi))
    target = _model(cfg, seed=1)
    target.load_state_dict(source.state_dict(), strict=True)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(5))
    source.eval(), target.eval()
    with torch.no_grad():
        a, b = source.forward_beliefs(tokens)[0], target.forward_beliefs(tokens)[0]
    assert torch.equal(a.mu, b.mu) and torch.equal(a.sigma, b.sigma)


def test_existing_modes_and_defaults_are_unchanged() -> None:
    default = _model(test_config())
    assert all(type(mlp) is BlockMLP for mlp in default.block_mlps)
    assert not any(k.rsplit(".", 1)[-1] in ("nu", "log_lambda", "log_pi")
                   for k in default.state_dict() if k.startswith("block_mlps."))
    shared = dict(embed_dim=3, irrep_dims=(3,), expansion=2, activation="gelu", dropout=0.0,
                  covariance_contract="passthrough", covariance_floor=1e-4)
    torch.manual_seed(0)
    built = build_block_mlp("gaussian_template", inference="exact", template_init_variance=3.0, **shared)
    torch.manual_seed(0)
    direct = GaussianTemplateBlockMLP(3, 2, 0.0)
    assert type(built) is GaussianTemplateBlockMLP
    assert all(torch.equal(built.state_dict()[k], v) for k, v in direct.state_dict().items())
    torch.manual_seed(0)
    coordinate = build_block_mlp("coordinate", inference="exact", **shared)
    torch.manual_seed(0)
    assert all(torch.equal(coordinate.state_dict()[k], v)
               for k, v in BlockMLP(3, 2, "gelu", 0.0).state_dict().items())


def test_consumers_are_aware_or_refuse() -> None:
    from vfe3.metrics import covariance_frozen_at_encode
    from vfe3.posthoc.interventions import NotApplicable, _mlp_plan

    assert covariance_frozen_at_encode(_cfg(predictive_covariance="passthrough"))
    assert not covariance_frozen_at_encode(_cfg(predictive_covariance="passthrough",
                                                block_mlp_covariance="posterior"))
    with pytest.raises(NotApplicable, match="prototype_agents"):
        _mlp_plan(PrototypeAgentBlockMLP(4, 1, 0.0))


# ----------------------------------------------------------------------------------------------
# 9. Model smoke: a few training steps in every coordinate, inference and covariance setting
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("coordinates", ["local", "reference"])
@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("contract", ["passthrough", "posterior"])
@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_model_smoke_training_steps(coordinates: str, inference: str, contract: str, family: str) -> None:
    from vfe3.train import build_optimizer

    extra = dict(decode_mode="full_chunked") if family == "gaussian_full" else {}
    cfg = _cfg(block_coordinates=coordinates, block_mlp_inference=inference, block_mlp_covariance=contract,
               family=family, **extra)
    model = _model(cfg).train()
    optimizer = build_optimizer(model, cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(7))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(tokens, tokens)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("block_mlps."):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().max() > 0, name
        optimizer.step()
    if contract == "posterior":                                                 # the block rewrote the covariance
        capture: Dict[str, Any] = {"diagnostic": {}}
        with torch.no_grad():
            model.forward_beliefs(tokens, capture=capture)
        first = capture["diagnostic"]["layer_block_mlp_inputs"][0].sigma
        assert not torch.equal(capture["diagnostic"]["layer_block_mlp_outputs"][0].sigma, first)
