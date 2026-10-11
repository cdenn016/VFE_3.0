r"""Multi-step CAVI and the prototype-template EM M-step term (audit 2026-10-10, items 23 and 24).

``minimal_agent_cavi_steps`` and ``block_mlp_cavi_steps`` run n coordinate-ascent sweeps of the structured mean field
(:func:`vfe3.model.minimal_agent.minimal_agent_readout`, :func:`vfe3.model.prototype_agents.prototype_mean_field_posterior`):
each later sweep recomputes the labels at the previous sweep's posterior and fuses again from the input moments.
``block_mlp_em_weight`` adds :math:`-w\sum_l\operatorname{mean}_iQ_{il}` to the training loss, with :math:`Q_{il}` the
expected complete-data log-likelihood of the prototype templates under the detached posterior the block ran
(:meth:`vfe3.model.block_mlp.PrototypeAgentBlockMLP.em_objective`; audit 2026-10-10 round 2, item 15). Tiny CPU cases
only: one sweep against the first-step reference, the label staleness over 20 sweeps, the EM gradient against central
differences of the evidence (exact) and of the final-state M-step (mean field), the w = 0 path, and one training step
with each option on.
"""
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model import minimal_agent as ma
from vfe3.model import prototype_agents as pa
from vfe3.model.block_mlp import PrototypeAgentBlockMLP
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

torch.set_num_threads(1)
F64 = torch.float64
_MINIMAL = dict(predictive_readout="minimal_agent", minimal_agent_inference="mean_field", attention_residual=False,
                max_seq_len=4, n_layers=2)
_PROTOTYPE = dict(use_block_mlp=True, block_mlp_mode="prototype_agents", max_seq_len=4, n_layers=2)


def _model(cfg: Any, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def _tokens(cfg: Any) -> torch.Tensor:                                    # (2, 4)
    return torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(7))


# ----------------------------------------------------------------------------------------------
# Minimal agent
# ----------------------------------------------------------------------------------------------

def _minimal_case(seed: int, *, rope: bool = False) -> Tuple[torch.Tensor, ...]:
    r"""Dense reference moments (1, N, K), generic maps (H, d, K), noise 0.3, a causal prior and an optional
    per-position rotation (N, d, d)."""
    g = torch.Generator().manual_seed(seed)
    H, d, N = 2, 2, 5
    K = H * d
    xi = 2.0 * torch.randn(1, N, K, generator=g, dtype=F64)
    X = torch.randn(1, N, K, K, generator=g, dtype=F64)
    C = X @ X.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=F64)
    S = torch.eye(K, dtype=F64).reshape(H, d, K) + 0.4 * torch.randn(H, d, K, generator=g, dtype=F64)
    T = torch.eye(K, dtype=F64).reshape(H, d, K) + 0.4 * torch.randn(H, d, K, generator=g, dtype=F64)
    log_noise = torch.full((H,), math.log(0.3), dtype=F64)
    lp = torch.zeros(H, N, N, dtype=F64).masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -math.inf)
    R = None
    if rope:
        angle = 0.9 * torch.arange(N, dtype=F64)[:, None]
        c, s = torch.cos(angle), torch.sin(angle)
        R = torch.stack([torch.cat([c, -s], -1), torch.cat([s, c], -1)], -2)    # (N, 2, 2)
    return xi, C, S, T, log_noise, lp, R


def _sources(xi: torch.Tensor, C: torch.Tensor, S: torch.Tensor, T: torch.Tensor, log_noise: torch.Tensor,
             R: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Utterance moments :math:`(R_jT_h\xi_j, R_j(T_hC_jT_h^\top + n_hI)R_j^\top)`, ``(1, H, N, d)`` and ``(1, H, N, d, d)``."""
    d = S.shape[1]
    b = torch.einsum("hsk,bnk->bhns", T, xi)
    B = torch.einsum("hsk,bnkl,htl->bhnst", T, C, T) + log_noise.exp()[:, None, None, None] * torch.eye(d, dtype=F64)
    if R is not None:
        b = torch.einsum("nst,bhnt->bhns", R, b)
        B = torch.einsum("nst,bhntu,nvu->bhnsv", R, B, R)
    return b, 0.5 * (B + B.transpose(-1, -2))


def _queries(xi: torch.Tensor, C: torch.Tensor, S: torch.Tensor,
             R: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Receiver query moments :math:`(R_iS_h\xi_i, R_iS_hC_iS_h^\top R_i^\top)`, ``(1, H, N, d)`` and ``(1, H, N, d, d)``."""
    a = torch.einsum("hsk,bnk->bhns", S, xi)
    A = torch.einsum("hsk,bnkl,htl->bhnst", S, C, S)
    if R is not None:
        a = torch.einsum("nst,bhnt->bhns", R, a)
        A = torch.einsum("nst,bhntu,nvu->bhnsv", R, A, R)
    return a, 0.5 * (A + A.transpose(-1, -2))


def test_minimal_agent_one_sweep_is_the_first_step() -> None:
    r"""n = 1 is the default call, and bitwise the first mean-field step: labels at the input, one information fusion."""
    xi, C, S, T, log_noise, lp, _ = _minimal_case(2)
    default = ma.minimal_agent_readout(xi, C, S, T, log_noise, lp, "mean_field", diagnostics=True)
    one = ma.minimal_agent_readout(xi, C, S, T, log_noise, lp, "mean_field", diagnostics=True, cavi_steps=1)
    for got, want in zip(one, default):
        if torch.is_tensor(want):                                         # spread_share is NaN in the mean field
            torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, equal_nan=True)
    # The first step, assembled from the module's own pieces exactly as the one-sweep readout runs them.
    X0 = ma.gain_basis(C, S, False)                                       # (1, N, H, K, d)
    V_Q = ma._symmetrize(torch.einsum("hsk,...nhkt->...nhst", S, X0))
    V_K = ma._symmetrize(torch.einsum("hsk,...nhkt->...nhst", T, ma.gain_basis(C, T, False)))
    b = torch.einsum("hsk,...nk->...hns", T, xi)
    B = V_K.movedim(-3, -4) + log_noise.exp()[:, None, None, None] * torch.eye(2, dtype=F64)
    labels = ma.mean_field_record_posterior(torch.einsum("hsk,...nk->...hns", S, xi), V_Q.movedim(-3, -4), b, B, lp)

    def head_update(h: int, a: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u, M, _ = ma.information_step(a, A, labels.Lambda[..., h, :, :, :], labels.eta[..., h, :, :])
        return u, M

    xi_plus, gains, M, _ = ma.sequential_head_assembly(xi, X0, S, head_update)
    assert torch.equal(one.xi, xi_plus) and torch.equal(one.gains, gains) and torch.equal(one.M, M)
    assert torch.equal(one.weights, labels.weights) and torch.equal(one.cost, labels.cost)
    with pytest.raises(ValueError, match="cavi_steps"):
        ma.minimal_agent_readout(xi, C, S, T, log_noise, lp, "mean_field", cavi_steps=0)


@pytest.mark.parametrize("rope", [False, True])
def test_minimal_agent_later_sweep_relabels_at_the_posterior(rope: bool) -> None:
    r"""Sweep 2 uses the KL labels at sweep 1's posterior query moments :math:`(S_h\xi^+, S_hC^+S_h^\top)` (rotated by
    :math:`R_i`), built here from the dense :math:`C^+ = C - \sum_h\mathsf K^hM^h\mathsf K^{h\top}`."""
    xi, C, S, T, log_noise, lp, R = _minimal_case(2, rope=rope)
    first = ma.minimal_agent_readout(xi, C, S, T, log_noise, lp, "mean_field", rope=R)
    second = ma.minimal_agent_readout(xi, C, S, T, log_noise, lp, "mean_field", rope=R, cavi_steps=2)
    C_plus = C - torch.einsum("bnhks,bnhst,bnhlt->bnkl", first.gains, first.M, first.gains)
    a, A = _queries(first.xi, C_plus, S, R)
    b, B = _sources(xi, C, S, T, log_noise, R)
    want = ma.mean_field_record_posterior(a, A, b, B, lp)
    torch.testing.assert_close(second.weights, want.weights, rtol=1e-11, atol=1e-12)
    assert float((second.weights - first.weights).abs().max()) > 1e-3     # the labels moved


@pytest.mark.parametrize("rope", [False, True])
def test_minimal_agent_sweeps_drive_the_staleness_to_zero(rope: bool) -> None:
    r"""The label staleness after the last sweep (TV between its labels and the labels at its posterior) falls from
    above 0.1 at one sweep to below 1e-5 at 20, contracting geometrically between sweeps 10 and 20."""
    xi, C, S, T, log_noise, lp, R = _minimal_case(2, rope=rope)
    staleness = {}
    results = {}
    for n in (1, 10, 20, 21):
        results[n] = ma.minimal_agent_readout(xi, C, S, T, log_noise, lp, "mean_field", diagnostics=True, rope=R,
                                              cavi_steps=n)
        staleness[n] = float(results[n].label_staleness.max())
    rate = (staleness[20] / staleness[10]) ** 0.1                           # per-sweep contraction factor
    assert staleness[1] > 0.1 and staleness[20] < 1e-5 and rate < 0.7, (staleness, rate)
    torch.testing.assert_close(results[21].xi, results[20].xi, rtol=1e-4, atol=1e-4)   # near the fixed point


def test_minimal_agent_cavi_steps_reach_the_model_readout() -> None:
    r"""``minimal_agent_cavi_steps`` reaches every layer's operators and the runtime readout: more sweeps lower the
    final layer's label staleness and change the loss."""
    one = _model(tiny_config(**_MINIMAL))
    many = _model(tiny_config(**_MINIMAL, minimal_agent_cavi_steps=6))
    many.load_state_dict(one.state_dict())
    assert all(ops.minimal_agent_cavi_steps == 6 for ops in many.relational_operators_by_layer)
    assert all(ops.minimal_agent_cavi_steps == 1 for ops in one.relational_operators_by_layer)
    tokens = _tokens(one.cfg)
    stale = []
    for model in (one, many):
        capture: Dict[str, Any] = {}
        with torch.no_grad():
            model.forward_beliefs(tokens, capture=capture)
        stale.append(float(capture["relational_capture"]["relational_minimal_agent_label_staleness"].max()))
    assert stale[1] < stale[0], stale
    with torch.no_grad():
        assert not torch.equal(one(tokens, tokens)[1], many(tokens, tokens)[1])


# ----------------------------------------------------------------------------------------------
# Prototype agents
# ----------------------------------------------------------------------------------------------

_LAYOUT_SEEDS = {"diagonal": 3, "dense": 0, "blocks": 0}


def _prototype_case(seed: int, layout: str) -> Tuple[torch.Tensor, ...]:
    r"""Working beliefs (6, K) with a diagonal, dense or (2, 2, 2)-block covariance and M = 8 templates of variance 0.5."""
    g = torch.Generator().manual_seed(seed)
    K, M = 4, 8
    xi = 1.5 * torch.randn(6, K, generator=g, dtype=F64)
    if layout == "diagonal":
        C = 0.5 + 2.0 * torch.rand(6, K, generator=g, dtype=F64)
    elif layout == "dense":
        X = torch.randn(6, K, K, generator=g, dtype=F64)
        C = 0.5 * X @ X.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=F64)
    else:
        X = torch.randn(6, 2, 2, 2, generator=g, dtype=F64)
        C = 0.5 * X @ X.transpose(-1, -2) + 0.5 * torch.eye(2, dtype=F64)
    nu = 1.5 * torch.randn(M, K, generator=g, dtype=F64)
    log_lambda = torch.full((M, K), math.log(0.5), dtype=F64)
    return xi, C, nu, log_lambda, torch.zeros(M, dtype=F64)


def _prototype_staleness(post: pa.MeanFieldPrototypePosterior, layout: str, nu: torch.Tensor,
                         log_lambda: torch.Tensor, log_pi: torch.Tensor) -> float:
    r"""Max TV between the last sweep's responsibilities and the label update at its posterior :math:`(\xi^+, C^+)`."""
    relabeled = torch.softmax(pa.template_kl_logits(post.xi_plus, pa.covariance_diagonal(post.C_plus, layout), nu,
                                                    log_lambda, log_pi), dim=-1)
    return float(0.5 * (relabeled - post.responsibilities).abs().sum(dim=-1).max())


def test_prototype_one_sweep_is_the_first_step() -> None:
    r"""n = 1 is bitwise the first step on the diagonal layout, and matches the explicit power posterior on the dense
    layout; ``need_covariance=False`` still returns no covariance."""
    xi, c, nu, ll, lpi = _prototype_case(3, "diagonal")
    post = pa.prototype_mean_field_posterior(xi, c, nu, ll, lpi, cavi_steps=1)
    r = torch.softmax(pa.template_kl_logits(xi, c, nu, ll, lpi), dim=-1)
    precision = torch.exp(-ll)
    D = r @ precision
    P = c / (1.0 + c * D)
    delta = P * (r @ (nu * precision) - D * xi)
    assert torch.equal(post.responsibilities, r) and torch.equal(post.delta_xi, delta)
    assert torch.equal(post.xi_plus, xi + delta) and torch.equal(post.C_plus, P)
    assert torch.equal(pa.prototype_mean_field_posterior(xi, c, nu, ll, lpi).xi_plus, post.xi_plus)
    xi, C, nu, ll, lpi = _prototype_case(0, "dense")
    for need in (True, False):
        post = pa.prototype_mean_field_posterior(xi, C, nu, ll, lpi, need_covariance=need, cavi_steps=1)
        r = torch.softmax(pa.template_kl_logits(xi, torch.diagonal(C, dim1=-2, dim2=-1), nu, ll, lpi), dim=-1)
        D = r @ torch.exp(-ll)
        P = torch.linalg.inv(torch.linalg.inv(C) + torch.diag_embed(D))
        want = (P @ (torch.linalg.solve(C, xi.unsqueeze(-1)).squeeze(-1) + r @ (nu * torch.exp(-ll))).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(post.xi_plus, want, rtol=1e-10, atol=1e-10)
        assert (post.C_plus is None) == (not need)
    with pytest.raises(ValueError, match="cavi_steps"):
        pa.prototype_mean_field_posterior(xi, C, nu, ll, lpi, cavi_steps=True)


@pytest.mark.parametrize("layout", ["diagonal", "dense", "blocks"])
def test_prototype_sweeps_relabel_and_drive_the_staleness_to_zero(layout: str) -> None:
    r"""Sweep 2's responsibilities are the label update at sweep 1's :math:`(\xi^+, \operatorname{diag}C^+)`, and the
    power posterior is always formed from the prior moments; 20 sweeps bring the staleness below 1e-5."""
    xi, C, nu, ll, lpi = _prototype_case(_LAYOUT_SEEDS[layout], layout)
    first = pa.prototype_mean_field_posterior(xi, C, nu, ll, lpi, need_covariance=True)
    second = pa.prototype_mean_field_posterior(xi, C, nu, ll, lpi, need_covariance=False, cavi_steps=2)
    want = torch.softmax(pa.template_kl_logits(first.xi_plus, pa.covariance_diagonal(first.C_plus, layout), nu, ll,
                                               lpi), dim=-1)
    torch.testing.assert_close(second.responsibilities, want, rtol=1e-12, atol=1e-13)
    assert second.C_plus is None
    # From the prior moments with fixed responsibilities r: D = r Lambda^{-1}, xi^+ = (C^-1 + D)^-1 (C^-1 xi + eta).
    if layout == "diagonal":
        dense = torch.diag_embed(C)                                         # (6, K, K)
    elif layout == "dense":
        dense = C
    else:
        dense = torch.stack([torch.block_diag(*C[i]) for i in range(C.shape[0])])
    P = torch.linalg.inv(torch.linalg.inv(dense) + torch.diag_embed(second.D))
    target = (P @ (torch.linalg.solve(dense, xi.unsqueeze(-1)).squeeze(-1) + second.eta).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(second.xi_plus, target, rtol=1e-10, atol=1e-10)
    staleness = {n: _prototype_staleness(pa.prototype_mean_field_posterior(xi, C, nu, ll, lpi, cavi_steps=n), layout,
                                         nu, ll, lpi) for n in (1, 10, 20)}
    rate = (staleness[20] / staleness[10]) ** 0.1 if staleness[10] > 0 else 0.0
    assert staleness[1] > 0.1 and staleness[20] < 1e-5 and rate < 0.6, (staleness, rate)


def test_prototype_cavi_steps_reach_the_block_mlp() -> None:
    r"""``block_mlp_cavi_steps`` reaches every prototype block and changes the loss; it is inactive under exact inference."""
    one = _model(tiny_config(**_PROTOTYPE))
    many = _model(tiny_config(**_PROTOTYPE, block_mlp_cavi_steps=4))
    many.load_state_dict(one.state_dict())
    assert all(mlp.cavi_steps == 4 for mlp in many.block_mlps) and all(mlp.cavi_steps == 1 for mlp in one.block_mlps)
    tokens = _tokens(one.cfg)
    with torch.no_grad():
        assert not torch.equal(one(tokens, tokens)[1], many(tokens, tokens)[1])
    assert "block_mlp_cavi_steps" in tiny_config(**_PROTOTYPE, block_mlp_inference="exact").inactive_fields()


# ----------------------------------------------------------------------------------------------
# EM M-step term
# ----------------------------------------------------------------------------------------------

def _dense_covariance(sigma: torch.Tensor, layout: str) -> torch.Tensor:   # (..., K, K)
    if layout == "diagonal":
        return torch.diag_embed(sigma)
    if layout == "dense":
        return sigma
    G, b = sigma.shape[-3], sigma.shape[-1]
    out = sigma.new_zeros(*sigma.shape[:-3], G * b, G * b)
    for h in range(G):
        out[..., h * b:(h + 1) * b, h * b:(h + 1) * b] = sigma[..., h, :, :]
    return out


def _template_weights(mlp: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        post = mlp.posterior(mu, sigma)
    return post.weights if mlp.inference == "exact" else post.responsibilities


def _q_reference(mlp: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:   # (...) Q_i
    r""":math:`Q_i = \sum_aq(a)[\log\hat\pi_a - \tfrac12\sum_k(\log 2\pi\lambda_{ak} + ((m_{iak} - \nu_{ak})^2 + p_{iak})/\lambda_{ak})]`
    from explicit float64 moments: the exact component posteriors :math:`m_{ia} = \xi_i + C_iW_{ia}^{-1}(\nu_a - \xi_i)`,
    :math:`P_{ia} = C_i - C_iW_{ia}^{-1}C_i` by dense inverses, or the last CAVI sweep's :math:`(r, \xi^+, \operatorname{diag}C^+)`."""
    with torch.no_grad():
        layout = pa.covariance_layout(mu, sigma)
        lam = mlp.log_lambda.detach().double().exp()                        # (M, K)
        nu = mlp.nu.detach().double()
        log_pi_hat = torch.log_softmax(mlp.log_pi.detach().double(), dim=-1)
        x = mu.double()
        if mlp.inference == "exact":
            q_a = mlp.posterior(mu, sigma).weights.double()                 # (..., M)
            C = _dense_covariance(sigma.double(), layout).unsqueeze(-3)     # (..., 1, K, K)
            W_inv = torch.linalg.inv(C + torch.diag_embed(lam))             # (..., M, K, K)
            m = x.unsqueeze(-2) + (C @ W_inv @ (nu - x.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)   # (..., M, K)
            p = torch.diagonal(C - C @ W_inv @ C, dim1=-2, dim2=-1)         # (..., M, K)
        else:
            post = mlp.posterior(mu, sigma, need_covariance=True)
            q_a = post.responsibilities.double()
            m = post.xi_plus.double().unsqueeze(-2)                          # (..., 1, K)
            p = pa.covariance_diagonal(post.C_plus, layout).double().unsqueeze(-2)
        log_density = -0.5 * (torch.log(2.0 * math.pi * lam) + ((m - nu).square() + p) / lam).sum(dim=-1)
        return (q_a * (log_pi_hat + log_density)).sum(dim=-1)


def _q_block_input(mlp: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""The pre-fix term (negative control): the posterior's weights with expectations under the block-input moments
    :math:`(\xi_i, \operatorname{diag}C_i)`."""
    c = pa.covariance_diagonal(sigma, pa.covariance_layout(mu, sigma))
    lam = mlp.log_lambda.exp()
    diff = mu.unsqueeze(-2) - mlp.nu                                        # (..., M, K)
    log_density = -0.5 * (torch.log(2.0 * math.pi * lam) + (diff.square() + c.unsqueeze(-2)) / lam).sum(dim=-1)
    return (_template_weights(mlp, mu, sigma) * (torch.log_softmax(mlp.log_pi, dim=-1) + log_density)).sum(dim=-1)


def _log_evidence(mlp: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r""":math:`\sum_i\log p(\xi_i \mid \theta) = \sum_i\log\sum_a\hat\pi_a\mathcal N(\nu_a;\xi_i, C_i + \Lambda_a)`, dense float64."""
    C = _dense_covariance(sigma, pa.covariance_layout(mu, sigma)).unsqueeze(-3)          # (..., 1, K, K)
    W = C + torch.diag_embed(mlp.log_lambda.exp())                          # (..., M, K, K)
    diff = mlp.nu - mu.unsqueeze(-2)                                        # (..., M, K)
    maha = (diff.unsqueeze(-2) @ torch.linalg.solve(W, diff.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    log_density = -0.5 * (maha + torch.logdet(W) + mu.shape[-1] * math.log(2.0 * math.pi))
    return torch.logsumexp(torch.log_softmax(mlp.log_pi, dim=-1) + log_density, dim=-1).sum()


def _final_state_m_step(mlp: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor,
                        post: pa.MeanFieldPrototypePosterior) -> torch.Tensor:
    r"""The mean-field ELBO's :math:`\theta`-dependent part at the frozen final CAVI state :math:`q(a_i) = r_i`,
    :math:`q(X_i) = \mathcal N(\xi_i^+, C_i^+)`: :math:`\sum_{ia}r_{ia}E_q[\log\hat\pi_a + \log\mathcal N(X_i;\nu_a,\Lambda_a)]`."""
    lam = mlp.log_lambda.exp()
    c_plus = pa.covariance_diagonal(post.C_plus, pa.covariance_layout(mu, sigma)).unsqueeze(-2)
    diff = post.xi_plus.unsqueeze(-2) - mlp.nu                              # (..., M, K)
    log_density = -0.5 * (torch.log(2.0 * math.pi * lam) + (diff.square() + c_plus) / lam).sum(dim=-1)
    return (post.responsibilities * (torch.log_softmax(mlp.log_pi, dim=-1) + log_density)).sum()


def _central_differences(fn: Any, params: Tuple[torch.Tensor, ...], h: float = 1e-6) -> List[torch.Tensor]:
    grads = []
    with torch.no_grad():
        for p in params:
            grad = torch.zeros_like(p)
            flat, gflat = p.view(-1), grad.view(-1)
            for j in range(flat.numel()):
                original = float(flat[j])
                flat[j] = original + h
                up = float(fn())
                flat[j] = original - h
                down = float(fn())
                flat[j] = original
                gflat[j] = (up - down) / (2.0 * h)
            grads.append(grad)
    return grads


def _em_case(inference: str, layout: str) -> Tuple[PrototypeAgentBlockMLP, torch.Tensor, torch.Tensor]:
    r"""A float64 block with generic templates (K = 4, M = 8) and a (2, 3) batch of working beliefs in ``layout``."""
    torch.manual_seed(0)
    K, e = 4, 2
    mlp = PrototypeAgentBlockMLP(K, e, 0.0, inference=inference, cavi_steps=3 if inference == "mean_field" else 1)
    mlp = mlp.double()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        mlp.nu.copy_(1.5 * torch.randn(mlp.nu.shape, generator=g, dtype=F64))
        mlp.log_lambda.copy_(0.3 * torch.randn(mlp.log_lambda.shape, generator=g, dtype=F64))
        mlp.log_pi.copy_(0.5 * torch.randn(mlp.log_pi.shape, generator=g, dtype=F64))
    mu = (1.5 * torch.randn(2, 3, K, generator=g, dtype=F64)).requires_grad_()
    if layout == "diagonal":
        sigma = 0.5 + torch.rand(2, 3, K, generator=g, dtype=F64)
    elif layout == "dense":
        X = torch.randn(2, 3, K, K, generator=g, dtype=F64)
        sigma = 0.3 * X @ X.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=F64)
    else:
        X = torch.randn(2, 3, 2, 2, 2, generator=g, dtype=F64)
        sigma = 0.3 * X @ X.transpose(-1, -2) + 0.5 * torch.eye(2, dtype=F64)
    return mlp, mu, sigma.requires_grad_()


@pytest.mark.parametrize("inference,layout", [("mean_field", "diagonal"), ("mean_field", "dense"),
                                              ("exact", "diagonal"), ("exact", "dense"), ("exact", "blocks")])
def test_em_gradient_is_the_gradient_of_the_intended_objective(inference: str, layout: str) -> None:
    r"""Audit 2026-10-10 round 2, item 15. The EM rows equal :math:`Q_i` under the posterior the block ran, and
    :math:`\sum_iQ_i` has, against float64 central differences, the gradient of :math:`\sum_i\log p(\xi_i \mid \theta)`
    (exact, Fisher's identity) or of the M-step at the final CAVI state (mean field, three sweeps); none reaches the
    block input. The pre-fix block-input term misses both."""
    mlp, mu, sigma = _em_case(inference, layout)
    rows: List[torch.Tensor] = []
    mlp.forward_moments(mu, sigma, em_rows=rows)
    assert len(rows) == 1 and rows[0].shape == (2, 3)
    torch.testing.assert_close(rows[0], _q_reference(mlp, mu.detach(), sigma.detach()), rtol=1e-12, atol=1e-12)
    params = (mlp.nu, mlp.log_lambda, mlp.log_pi)
    *got, g_mu, g_sigma = torch.autograd.grad(rows[0].sum(), (*params, mu, sigma), allow_unused=True)
    assert g_mu is None and g_sigma is None                                 # the posterior is detached
    x, S = mu.detach(), sigma.detach()
    if inference == "exact":
        objective = lambda: _log_evidence(mlp, x, S)                        # noqa: E731
    else:
        with torch.no_grad():
            post = mlp.posterior(x, S, need_covariance=True)                # the final sweep, frozen
        objective = lambda: _final_state_m_step(mlp, x, S, post)            # noqa: E731
    want = _central_differences(objective, params)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    stale = torch.autograd.grad(_q_block_input(mlp, x, S).sum(), params[:2])
    gap = max(float((a - b).norm() / b.norm()) for a, b in zip(stale, want[:2]))
    assert gap > 0.05, gap                                                  # the old pairing is a different gradient


@pytest.mark.parametrize("coordinates", ["local", "reference"])
@pytest.mark.parametrize("inference", ["mean_field", "exact"])
def test_em_term_enters_the_loss_with_the_m_step_gradient(coordinates: str, inference: str,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    r"""At :math:`w > 0` the loss is :math:`\mathrm{CE} - w\sum_l\operatorname{mean}_{i\in\mathrm{scored}}Q_{il}` under the
    posterior each block ran at its input, the reported CE is unchanged, :math:`\log\pi` of layer :math:`l` gains
    :math:`-w(\operatorname{mean}_i\bar r_{ia} - \hat\pi_a)` and no parameter outside the templates changes gradient."""
    w = 0.4
    base = dict(_PROTOTYPE, block_coordinates=coordinates, block_mlp_inference=inference)
    off = _model(tiny_config(**base))
    on = _model(tiny_config(**base, block_mlp_em_weight=w))
    on.load_state_dict(off.state_dict())
    tokens = _tokens(off.cfg)
    targets = tokens.clone()
    targets[0, 1] = -100                                                    # one unscored position
    _, loss_off, ce_off = off(tokens, targets)
    loss_off.backward()
    seen: List[Tuple[PrototypeAgentBlockMLP, torch.Tensor, torch.Tensor]] = []
    original = PrototypeAgentBlockMLP.forward_moments

    def spy(self: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> Any:
        assert "em_rows" in kwargs
        seen.append((self, mu.detach(), sigma.detach()))
        return original(self, mu, sigma, **kwargs)

    monkeypatch.setattr(PrototypeAgentBlockMLP, "forward_moments", spy)
    _, loss_on, ce_on = on(tokens, targets)
    monkeypatch.setattr(PrototypeAgentBlockMLP, "forward_moments", original)
    loss_on.backward()
    assert [entry[0] for entry in seen] == list(on.block_mlps)
    scored = (targets != -100).to(torch.float32)                            # (B, N)
    n_scored = scored.sum()
    em = sum((_q_reference(mlp, mu, sigma).float() * scored).sum() for mlp, mu, sigma in seen)
    torch.testing.assert_close(ce_on, ce_off, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loss_on.detach() - ce_on, -w * em / n_scored, rtol=1e-4, atol=1e-5)
    off_grads = dict(off.named_parameters())
    for layer, (mlp, mu, sigma) in enumerate(seen):
        r = _template_weights(mlp, mu, sigma)                               # (B, N, M)
        r_mean = (r * scored.unsqueeze(-1)).sum(dim=(0, 1)) / n_scored
        want = -w * (r_mean - torch.softmax(mlp.log_pi.detach(), dim=-1))
        got = mlp.log_pi.grad - off_grads[f"block_mlps.{layer}.log_pi"].grad
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6)
    for name, parameter in on.named_parameters():
        if parameter.grad is not None and not name.startswith("block_mlps."):
            torch.testing.assert_close(parameter.grad, off_grads[name].grad, rtol=1e-5, atol=1e-7)


def test_em_weight_zero_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""At w = 0 (and at w > 0 on a non-prototype block MLP, where the weight is inactive) no capture is allocated, no
    block receives an EM list and the loss is the CE tensor itself."""
    calls: List[Dict[str, Any]] = []
    captures: List[Any] = []
    forward_moments = PrototypeAgentBlockMLP.forward_moments
    forward_beliefs = VFEModel.forward_beliefs

    def spy_moments(self: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return forward_moments(self, mu, sigma, **kwargs)

    def spy_beliefs(self: VFEModel, *args: Any, **kwargs: Any) -> Any:
        captures.append(kwargs.get("capture"))
        return forward_beliefs(self, *args, **kwargs)

    monkeypatch.setattr(PrototypeAgentBlockMLP, "forward_moments", spy_moments)
    monkeypatch.setattr(VFEModel, "forward_beliefs", spy_beliefs)
    for cfg in (tiny_config(**_PROTOTYPE),
                tiny_config(use_block_mlp=True, block_mlp_mode="coordinate", block_mlp_em_weight=0.5)):
        model = _model(cfg)
        tokens = _tokens(cfg)
        _, loss, ce = model(tokens, tokens)
        assert torch.equal(loss.detach(), ce)
    assert captures == [None, None] and len(calls) == 2 and all("em_rows" not in kwargs for kwargs in calls)
    assert "block_mlp_em_weight" in tiny_config(use_block_mlp=True, block_mlp_mode="coordinate",
                                                block_mlp_em_weight=0.5).inactive_fields()


# ----------------------------------------------------------------------------------------------
# One finite training step with each option on
# ----------------------------------------------------------------------------------------------

_OPTIONS: Dict[str, Dict[str, Any]] = {
    "minimal_cavi": dict(_MINIMAL, minimal_agent_cavi_steps=3),
    "prototype_cavi_local": dict(_PROTOTYPE, block_mlp_cavi_steps=3, block_mlp_covariance="posterior"),
    "prototype_cavi_reference": dict(_PROTOTYPE, block_mlp_cavi_steps=3, block_coordinates="reference"),
    "prototype_em_mean_field": dict(_PROTOTYPE, block_mlp_em_weight=0.5, block_mlp_cavi_steps=2),
    "prototype_em_exact": dict(_PROTOTYPE, block_mlp_em_weight=0.5, block_mlp_inference="exact"),
}


@pytest.mark.parametrize("option", sorted(_OPTIONS))
def test_one_training_step_with_each_option(option: str) -> None:
    cfg = tiny_config(**_OPTIONS[option])
    model = _model(cfg).train()
    optimizer = build_optimizer(model, cfg)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    tokens = _tokens(cfg)
    optimizer.zero_grad(set_to_none=True)
    _, loss, ce = model(tokens, tokens)
    assert torch.isfinite(loss) and torch.isfinite(ce)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(grad).all() for grad in grads.values())
    if cfg.use_block_mlp:
        assert all(grads[f"block_mlps.{layer}.log_pi"].abs().max() > 0 for layer in range(cfg.n_layers))
    optimizer.step()
    assert any(not torch.equal(before[name], p.detach()) for name, p in model.named_parameters() if name in before)
    assert all(torch.isfinite(p).all() for p in model.parameters())
