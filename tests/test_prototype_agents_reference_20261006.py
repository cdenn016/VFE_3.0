r"""Independent reference tests for the prototype-agent block MLP (``block_mlp_mode='prototype_agents'``).

Verifier: Claude Fable 5.1. Author of the implementation under test: Claude Opus 5.5. Every expected
value is derived here from the theory (``Research/manuscripts/Attention/minimal_agent_model.tex``,
Sections 2 to 4: Definition ma:def:model, Proposition ma:prop:minimal items (a), (c), (d), Proposition
ma:prop:mean_field) and the plan (``docs/implementation/minimal-agent-plan-2026-10-06.md``, Sections 1
and 3), by explicit float64 constructions at tiny sizes and never from the implementation's formulas:

* the exact posterior from the stacked joint law of the receiver state :math:`X \sim \mathcal N(\xi, C)`
  and one template agent :math:`Y_a \sim \mathcal N(\nu_a, \Lambda_a)`, mapped linearly to the pair
  (state, record) with the record :math:`o = X - Y_a`, conditioned on :math:`o = 0` by generic Gaussian
  conditioning in information form (the joint precision), with the label posterior from the record
  density at zero and the mixture moments from the law of total covariance;
* a two-dimensional grid quadrature of the overlaps :math:`\int q\,p_a`, the normalized products, the
  divergences and the power posterior, which uses no conditioning formula at all;
* the mean-field step from the explicit divergence formula and the explicit power-posterior precision;
* the precise-template (noise-free, full-rank) and broad-template limits with their first-order
  coefficients;
* passive gauge transformations of frames and beliefs;
* central finite differences of scalar losses against autograd.

The prototype block is a one-label model (one template per token, :math:`S = T = I`, no heads and no
causal mask), so the several-head label enumeration, the per-head ADF reference and the causality
perturbations of the attention feature have no counterpart here; their analogs below are the
enumeration over the :math:`M` templates, two block rounds against sequential conditioning with a
Gaussian collapse, and the independence of tokens.

Tolerances: ``TOL = 1e-10`` relative to the scale of the compared quantity for float64 linear algebra at
:math:`K \le 6` (the working covariances have eigenvalues in :math:`[0.5, 1.5]`); ``TOL_QUAD = 1e-9``
for the grid quadrature (domain truncation below :math:`10^{-13}`, trapezoidal error negligible for
Gaussians at step 0.05, summation over about :math:`3\times10^5` cells); ``TOL_FD = 1e-7`` for central
finite differences, whose own truncation and cancellation error at step :math:`10^{-6}` is about
:math:`10^{-9}`. CPU only, one thread, tiny models.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Callable, Dict, List, Sequence, Tuple

import pytest
import torch

from tests.profiles import config_values, test_config
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import prototype_agents as pa
from vfe3.model.block import _reference_coordinate_block
from vfe3.model.block_mlp import (
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
TOL = 1e-10
TOL_QUAD = 1e-9
TOL_FD = 1e-7


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------

def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _close(got: torch.Tensor, want: torch.Tensor, tol: float, what: str = "") -> None:
    r"""Assert :math:`\max|g - w| \le \mathrm{tol}\,(1 + \max|w|)`."""
    got = torch.as_tensor(got).detach().to(DT)
    want = torch.as_tensor(want).detach().to(DT)
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    scale = 1.0 + (float(want.abs().max()) if want.numel() else 0.0)
    err = float((got - want).abs().max()) if want.numel() else 0.0
    assert err <= tol * scale, f"{what}: max abs error {err:.3e} > {tol:.1e} x scale {scale:.3e}"


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:                # (..., G, b, b) -> (..., K, K)
    G, b = blocks.shape[-3], blocks.shape[-1]
    out = torch.zeros(*blocks.shape[:-3], G * b, G * b, dtype=blocks.dtype)
    for g in range(G):
        out[..., g * b:(g + 1) * b, g * b:(g + 1) * b] = blocks[..., g, :, :]
    return out


def _diag_blocks(dense: torch.Tensor, G: int) -> torch.Tensor:        # (..., K, K) -> (..., G, b, b)
    b = dense.shape[-1] // G
    return torch.stack([dense[..., g * b:(g + 1) * b, g * b:(g + 1) * b] for g in range(G)], dim=-3)


def _project(dense: torch.Tensor, layout: str, G: int) -> torch.Tensor:
    r"""The member of a layout's family read from a dense matrix: its diagonal, its diagonal blocks, or itself."""
    if layout == "diagonal":
        return torch.diagonal(dense, dim1=-2, dim2=-1)
    if layout == "blocks":
        return _diag_blocks(dense, G)
    return dense


def _spd(shape: Tuple[int, ...], k: int, seed: int, lo: float = 0.5, hi: float = 1.5) -> torch.Tensor:
    r"""Symmetric positive definite matrices :math:`Q\operatorname{diag}(e)Q^\top` with eigenvalues :math:`e \in [lo, hi]`."""
    g = _gen(seed)
    Q, _ = torch.linalg.qr(torch.randn(*shape, k, k, generator=g, dtype=DT))
    e = lo + (hi - lo) * torch.rand(*shape, k, generator=g, dtype=DT)
    return (Q * e.unsqueeze(-2)) @ Q.transpose(-1, -2)


def _working(layout: str, R: int, K: int, G: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""A working covariance in ``layout`` for R tokens and its dense equivalent ``(R, K, K)``."""
    if layout == "diagonal":
        c = 0.5 + torch.rand(R, K, generator=_gen(seed), dtype=DT)
        return c, torch.diag_embed(c)
    if layout == "blocks":
        blocks = _spd((R, G), K // G, seed)
        return blocks, _block_diag(blocks)
    dense = _spd((R,), K, seed)
    return dense, dense


def _templates(K: int, M: int, seed: int, scale: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = _gen(seed)
    nu = torch.randn(M, K, generator=g, dtype=DT)
    log_lambda = scale * torch.randn(M, K, generator=g, dtype=DT)
    log_pi = 0.5 * torch.randn(M, generator=g, dtype=DT)
    return nu, log_lambda, log_pi


def _module(K: int, e: int, inference: str, contract: str, seed: int, dropout: float = 0.0) -> PrototypeAgentBlockMLP:
    r"""A float64 block with random templates, in eval mode (dropout off unless ``train()`` is called)."""
    torch.manual_seed(seed)
    mlp = PrototypeAgentBlockMLP(K, e, dropout, inference=inference, covariance_contract=contract).double()
    g = _gen(seed + 100)
    with torch.no_grad():
        mlp.nu.copy_(torch.randn(*mlp.nu.shape, generator=g, dtype=DT))
        mlp.log_lambda.copy_(0.5 * torch.randn(*mlp.log_lambda.shape, generator=g, dtype=DT))
        mlp.log_pi.copy_(0.5 * torch.randn(*mlp.log_pi.shape, generator=g, dtype=DT))
    return mlp.eval()


# ----------------------------------------------------------------------------------------------
# Reference constructions (dense float64, one token at a time)
# ----------------------------------------------------------------------------------------------

def _log_gaussian(x: torch.Tensor, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    r""":math:`\log\mathcal N(x; m, S) = -\tfrac12(x-m)^\top S^{-1}(x-m) - \tfrac12\log\det S - \tfrac K2\log 2\pi`."""
    d = (x - m).unsqueeze(-1)
    quad = (d * torch.linalg.solve(S, d)).sum()
    sign, logabsdet = torch.linalg.slogdet(S)
    assert float(sign) > 0.0
    return -0.5 * quad - 0.5 * logabsdet - 0.5 * x.shape[-1] * math.log(2.0 * math.pi)


def _condition_on_record(
    xi: torch.Tensor, C: torch.Tensor, nu_a: torch.Tensor, lam_a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Brute-force conditioning of the receiver on one template's record (Definition ma:def:model).

    The stacked joint law of :math:`(X, Y_a)`, :math:`X \sim \mathcal N(\xi, C)` independent of
    :math:`Y_a \sim \mathcal N(\nu_a, \Lambda_a)`, is mapped by :math:`(X, Y_a) \mapsto (X, o)`,
    :math:`o = X - Y_a` (:math:`S = T = I`, channel noise absorbed into :math:`\Lambda_a`). With the joint
    precision :math:`J` of :math:`(X, o)`, generic Gaussian conditioning at :math:`o = 0` gives

    .. math::
        X \mid o = 0 \sim \mathcal N\bigl(\xi - J_{XX}^{-1}J_{Xo}(0 - \mathbb E o),\; J_{XX}^{-1}\bigr),

    and the record density :math:`\mathcal N(0; \mathbb E o, \operatorname{Cov} o)` is the label likelihood.
    Returns (log density of the record at zero, conditional mean, conditional covariance).
    """
    K = xi.shape[0]
    eye = torch.eye(K, dtype=DT)
    mean_xy = torch.cat([xi, nu_a])                                              # (2K,)
    cov_xy = torch.zeros(2 * K, 2 * K, dtype=DT)
    cov_xy[:K, :K] = C
    cov_xy[K:, K:] = torch.diag(lam_a)
    A = torch.zeros(2 * K, 2 * K, dtype=DT)                                      # (X, Y) -> (X, o)
    A[:K, :K] = eye
    A[K:, :K] = eye
    A[K:, K:] = -eye
    mean_xo = A @ mean_xy
    cov_xo = A @ cov_xy @ A.T
    mu_o, S_oo = mean_xo[K:], cov_xo[K:, K:]
    log_density = _log_gaussian(torch.zeros(K, dtype=DT), mu_o, S_oo)
    J = torch.linalg.inv(cov_xo)                                                 # joint precision
    P = torch.linalg.inv(J[:K, :K])                                              # conditional covariance
    m = xi - P @ J[:K, K:] @ (torch.zeros(K, dtype=DT) - mu_o)                   # conditional mean at o = 0
    return log_density, m, 0.5 * (P + P.T)


def _exact_reference(
    xi: torch.Tensor, C: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor, log_pi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Proposition ma:prop:minimal (a): label posterior, component posteriors and mixture moments.

    :math:`w_a \propto \pi_a\,p(o = 0 \mid a)`, :math:`\xi^+ = \sum_a w_a m_a`,
    :math:`C^+ = \sum_a w_a[P_a + (m_a - \xi^+)(m_a - \xi^+)^\top]`. The plan's auxiliary quantities
    :math:`r_a = W_a^{-1}(\nu_a - \xi)`, :math:`u = \sum_a w_a r_a`, :math:`\bar\Lambda = \sum_a w_a W_a^{-1}`,
    :math:`R = \sum_a w_a r_ar_a^\top - uu^\top` with :math:`W_a = C + \Lambda_a` are evaluated from their
    definitions with explicit inverses.
    """
    M = nu.shape[0]
    log_prior = torch.log_softmax(log_pi, dim=0)
    logw, ms, Ps, rs, W_invs = [], [], [], [], []
    for a in range(M):
        log_density, m, P = _condition_on_record(xi, C, nu[a], lam[a])
        logw.append(log_prior[a] + log_density)
        ms.append(m)
        Ps.append(P)
        W_inv = torch.linalg.inv(C + torch.diag(lam[a]))
        W_invs.append(W_inv)
        rs.append(W_inv @ (nu[a] - xi))
    w = torch.softmax(torch.stack(logw), dim=0)
    mean = sum(w[a] * ms[a] for a in range(M))
    cov = sum(w[a] * (Ps[a] + torch.outer(ms[a] - mean, ms[a] - mean)) for a in range(M))
    u = sum(w[a] * rs[a] for a in range(M))
    Lambda_bar = sum(w[a] * W_invs[a] for a in range(M))
    R = sum(w[a] * torch.outer(rs[a], rs[a]) for a in range(M)) - torch.outer(u, u)
    return dict(mean=mean, cov=cov, weights=w, u=u, Lambda_bar=Lambda_bar, R=R,
                m=torch.stack(ms), P=torch.stack(Ps))


def _kl_gaussian(xi: torch.Tensor, C: torch.Tensor, nu_a: torch.Tensor, lam_a: torch.Tensor) -> torch.Tensor:
    r""":math:`D_{\mathrm{KL}}(\mathcal N(\xi, C)\|\mathcal N(\nu_a, \Lambda_a)) = \tfrac12[\operatorname{tr}\Lambda_a^{-1}C + (\nu_a-\xi)^\top\Lambda_a^{-1}(\nu_a-\xi) - K + \log\det\Lambda_a - \log\det C]`."""
    K = xi.shape[0]
    Lam_inv = torch.diag(1.0 / lam_a)
    d = nu_a - xi
    return 0.5 * (torch.trace(Lam_inv @ C) + d @ Lam_inv @ d - K + torch.log(lam_a).sum() - torch.logdet(C))


def _mean_field_reference(
    xi: torch.Tensor, C: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor, log_pi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Proposition ma:prop:mean_field: the divergence label step at the prior and the power posterior.

    :math:`r_a \propto \pi_a e^{-D_{\mathrm{KL}}(\mathcal N(\xi, C)\|p_a)}`; the power posterior
    :math:`q(x) \propto \mathcal N(x; \xi, C)\prod_a\mathcal N(\nu_a; x, \Lambda_a)^{r_a}` has the explicit
    precision :math:`C^{-1} + \sum_a r_a\Lambda_a^{-1}` and information vector
    :math:`C^{-1}\xi + \sum_a r_a\Lambda_a^{-1}\nu_a`, read off its quadratic log density.
    """
    M = nu.shape[0]
    kl = torch.stack([_kl_gaussian(xi, C, nu[a], lam[a]) for a in range(M)])
    r = torch.softmax(torch.log_softmax(log_pi, dim=0) - kl, dim=0)
    D = (r[:, None] / lam).sum(0)
    eta = (r[:, None] * nu / lam).sum(0)
    C_inv = torch.linalg.inv(C)
    P = torch.linalg.inv(C_inv + torch.diag(D))
    P = 0.5 * (P + P.T)
    mean = P @ (C_inv @ xi + eta)
    return dict(mean=mean, cov=P, weights=r, D=D, eta=eta, delta=P @ (eta - D * xi))


def _reference(
    inference: str, xi_rows: torch.Tensor, C_rows: torch.Tensor,
    nu: torch.Tensor, lam: torch.Tensor, log_pi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Per-token reference moments: means ``(R, K)``, dense covariances ``(R, K, K)``, weights ``(R, M)``."""
    fn = _exact_reference if inference == "exact" else _mean_field_reference
    outs = [fn(xi_rows[i], C_rows[i], nu, lam, log_pi) for i in range(xi_rows.shape[0])]
    return (torch.stack([o["mean"] for o in outs]), torch.stack([o["cov"] for o in outs]),
            torch.stack([o["weights"] for o in outs]))


# Two-dimensional grid quadrature (K = 2): no conditioning formula enters.

def _grid(L: float = 14.0, h: float = 0.05) -> Tuple[torch.Tensor, float]:
    axis = torch.arange(-L, L + 0.5 * h, h, dtype=DT)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1), h * h


def _log_density_grid(pts: torch.Tensor, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    d = pts - m
    quad = ((d @ torch.linalg.inv(S)) * d).sum(-1)
    return -0.5 * quad - 0.5 * torch.logdet(S) - math.log(2.0 * math.pi)


def _grid_moments(pts: torch.Tensor, log_w: torch.Tensor, cell: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Log mass, mean and covariance of an unnormalized log density tabulated on the grid."""
    shift = log_w.max()
    w = torch.exp(log_w - shift)
    total = w.sum()
    mean = (w.unsqueeze(-1) * pts).sum(0) / total
    d = pts - mean
    cov = (w.unsqueeze(-1).unsqueeze(-1) * d.unsqueeze(-1) * d.unsqueeze(-2)).sum(0) / total
    return shift + torch.log(total * cell), mean, cov


def _quadrature_exact(
    xi: torch.Tensor, C: torch.Tensor, nu: torch.Tensor, lam: torch.Tensor, log_pi: torch.Tensor,
    pts: torch.Tensor, cell: float,
) -> Dict[str, torch.Tensor]:
    r"""Item (d) with :math:`\Theta = I`: :math:`w_a \propto \pi_a\int q\,p_a`, components the normalized products :math:`q\,p_a`."""
    log_prior = torch.log_softmax(log_pi, dim=0)
    log_q = _log_density_grid(pts, xi, C)
    logw, ms, Ps = [], [], []
    for a in range(nu.shape[0]):
        log_mass, m, P = _grid_moments(pts, log_q + _log_density_grid(pts, nu[a], torch.diag(lam[a])), cell)
        logw.append(log_prior[a] + log_mass)
        ms.append(m)
        Ps.append(P)
    w = torch.softmax(torch.stack(logw), dim=0)
    mean = sum(w[a] * ms[a] for a in range(len(ms)))
    cov = sum(w[a] * (Ps[a] + torch.outer(ms[a] - mean, ms[a] - mean)) for a in range(len(ms)))
    return dict(mean=mean, cov=cov, weights=w, m=torch.stack(ms), P=torch.stack(Ps))


def _central_difference(loss: Callable[[], torch.Tensor], tensors: Sequence[torch.Tensor], eps: float = 1e-6) -> List[torch.Tensor]:
    r"""Central differences :math:`(L(\theta + \epsilon e_k) - L(\theta - \epsilon e_k))/2\epsilon`, in place, under ``no_grad``."""
    grads = []
    with torch.no_grad():
        for t in tensors:
            flat = t.view(-1)
            g = torch.zeros_like(flat)
            for idx in range(flat.numel()):
                original = float(flat[idx])
                flat[idx] = original + eps
                plus = float(loss())
                flat[idx] = original - eps
                minus = float(loss())
                flat[idx] = original
                g[idx] = (plus - minus) / (2.0 * eps)
            grads.append(g.view_as(t))
    return grads


# ----------------------------------------------------------------------------------------------
# 1. Layouts and constants
# ----------------------------------------------------------------------------------------------

def test_public_constants() -> None:
    assert pa.PROTOTYPE_TILE_BYTES == 32 * 2 ** 20
    assert pa.PROTOTYPE_INFERENCE_MODES == ("mean_field", "exact")
    assert pa.PROTOTYPE_COVARIANCE_CONTRACTS == ("passthrough", "posterior")
    assert set(pa.COVARIANCE_LAYOUTS) == {"diagonal", "dense", "blocks"}


def test_covariance_layout_detection_and_diagonal() -> None:
    xi = torch.zeros(2, 3, 6, dtype=DT)
    assert pa.covariance_layout(xi, torch.ones(2, 3, 6, dtype=DT)) == "diagonal"
    assert pa.covariance_layout(xi, torch.eye(6, dtype=DT).expand(2, 3, 6, 6)) == "dense"
    assert pa.covariance_layout(xi, torch.eye(2, dtype=DT).expand(2, 3, 3, 2, 2)) == "blocks"
    assert pa.covariance_layout(xi, torch.eye(3, dtype=DT).expand(2, 3, 2, 3, 3)) == "blocks"
    for bad in (torch.ones(2, 3, 5), torch.ones(2, 3, 6, 5), torch.ones(2, 3, 2, 2, 2),
                torch.ones(3, 3, 6), torch.ones(2, 3, 2, 3, 2), torch.ones(2, 3, 6, 6, 6)):
        with pytest.raises(ValueError, match="working covariance"):
            pa.covariance_layout(xi, bad.to(DT))
    blocks = _spd((2, 3, 2), 3, seed=1)
    assert torch.equal(pa.covariance_diagonal(blocks, "blocks"),
                       torch.diagonal(_block_diag(blocks), dim1=-2, dim2=-1))
    dense = _spd((2, 3), 6, seed=2)
    assert torch.equal(pa.covariance_diagonal(dense, "dense"), torch.diagonal(dense, dim1=-2, dim2=-1))
    c = torch.rand(2, 3, 6, dtype=DT)
    assert pa.covariance_diagonal(c, "diagonal") is c


# ----------------------------------------------------------------------------------------------
# 2. Exact path: joint-law conditioning, mixture moments, the plan's closed form, quadrature
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_exact_posterior_equals_joint_conditioning_and_mixture_moments(layout: str) -> None:
    R, K, G, M = 3, 6, 3, 4
    nu, log_lambda, log_pi = _templates(K, M, seed=11)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=_gen(12), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=13)
    out = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True, need_precision=True)
    assert out.C_plus.shape == C.shape == out.R.shape == out.Lambda_bar.shape
    for i in range(R):
        ref = _exact_reference(xi[i], Cd[i], nu, lam, log_pi)
        _close(out.weights[i], ref["weights"], TOL, "weights")
        _close(out.xi_plus[i], ref["mean"], TOL, "xi_plus")
        _close(out.u[i], ref["u"], TOL, "u")
        _close(out.delta_xi[i], Cd[i] @ ref["u"], TOL, "delta_xi = C u")
        _close(out.C_plus[i], _project(ref["cov"], layout, G), TOL, "C_plus")
        _close(out.Lambda_bar[i], _project(ref["Lambda_bar"], layout, G), TOL, "Lambda_bar")
        _close(out.R[i], _project(ref["R"], layout, G), TOL, "R")
    _close(out.xi_plus, xi + out.delta_xi, 1e-15, "xi_plus = xi + delta_xi")


def test_plan_closed_form_and_normalized_products_equal_the_mixture_moments() -> None:
    r"""Hand identities grounding the plan: :math:`\xi^+ = \xi + Cu`, :math:`C^+ = C - C(\bar\Lambda - R)C`, and item (d)."""
    K, M = 5, 4
    nu, log_lambda, log_pi = _templates(K, M, seed=14)
    lam = log_lambda.exp()
    xi = torch.randn(K, generator=_gen(15), dtype=DT)
    C = _spd((), K, seed=16)
    ref = _exact_reference(xi, C, nu, lam, log_pi)
    _close(xi + C @ ref["u"], ref["mean"], TOL, "xi + C u")
    _close(C - C @ (ref["Lambda_bar"] - ref["R"]) @ C, ref["cov"], TOL, "C - C (Lambda_bar - R) C")
    # Item (d): each component posterior is the normalized product of N(xi, C) and p_a.
    C_inv = torch.linalg.inv(C)
    for a in range(M):
        P_a = torch.linalg.inv(C_inv + torch.diag(1.0 / lam[a]))
        _close(ref["P"][a], P_a, TOL, "P_a")
        _close(ref["m"][a], P_a @ (C_inv @ xi + nu[a] / lam[a]), TOL, "m_a")
    # The implementation's own fields satisfy the same closed form on the dense layout.
    out = pa.prototype_exact_posterior(xi[None], C[None], nu, log_lambda, log_pi,
                                       need_covariance=True, need_precision=True)
    _close(C - C @ (out.Lambda_bar[0] - out.R[0]) @ C, out.C_plus[0], TOL, "closed form of the fields")
    _close(xi + C @ out.u[0], out.xi_plus[0], TOL, "xi + C u of the fields")


@pytest.mark.parametrize("layout", ("diagonal", "dense"))
def test_exact_posterior_equals_two_dimensional_quadrature(layout: str) -> None:
    r"""At :math:`K = 2` the overlaps, the normalized products and the mixture moments by grid integration."""
    K, M = 2, 3
    pts, cell = _grid()
    g = _gen(21)
    nu = 4.0 * (torch.rand(M, K, generator=g, dtype=DT) - 0.5)                   # in [-2, 2]
    lam = 0.5 + 1.5 * torch.rand(M, K, generator=g, dtype=DT)                    # in [0.5, 2]
    log_pi = torch.randn(M, generator=g, dtype=DT)
    xi = 2.0 * torch.rand(K, generator=g, dtype=DT) - 1.0                        # in [-1, 1]
    if layout == "diagonal":
        c = 0.5 + torch.rand(K, generator=g, dtype=DT)
        C_in, Cd = c, torch.diag(c)
    else:
        Cd = _spd((), K, seed=22)
        C_in = Cd
    quad = _quadrature_exact(xi, Cd, nu, lam, log_pi, pts, cell)
    out = pa.prototype_exact_posterior(xi[None], C_in[None], nu, lam.log(), log_pi, need_covariance=True)
    _close(out.weights[0], quad["weights"], TOL_QUAD, "weights by quadrature")
    _close(out.xi_plus[0], quad["mean"], TOL_QUAD, "mean by quadrature")
    _close(out.C_plus[0], _project(quad["cov"], layout, 1), TOL_QUAD, "covariance by quadrature")
    # The conditioning route and the quadrature route agree on the components as well.
    ref = _exact_reference(xi, Cd, nu, lam, log_pi)
    _close(quad["weights"], ref["weights"], TOL_QUAD, "weights: quadrature vs conditioning")
    _close(quad["m"], ref["m"], TOL_QUAD, "m_a: quadrature vs conditioning")
    _close(quad["P"], ref["P"], TOL_QUAD, "P_a: quadrature vs conditioning")


# ----------------------------------------------------------------------------------------------
# 3. Mean-field path: explicit divergence labels, explicit power posterior, quadrature
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_mean_field_posterior_equals_explicit_power_posterior(layout: str) -> None:
    R, K, G, M = 3, 6, 2, 4
    nu, log_lambda, log_pi = _templates(K, M, seed=31)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=_gen(32), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=33)
    out = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True)
    assert out.C_plus.shape == C.shape
    for i in range(R):
        ref = _mean_field_reference(xi[i], Cd[i], nu, lam, log_pi)
        _close(out.responsibilities[i], ref["weights"], TOL, "responsibilities")
        _close(out.D[i], ref["D"], TOL, "D")
        _close(out.eta[i], ref["eta"], TOL, "eta")
        _close(out.xi_plus[i], ref["mean"], TOL, "xi_plus")
        _close(out.delta_xi[i], ref["delta"], TOL, "delta_xi = P (eta - D xi)")
        _close(out.C_plus[i], _project(ref["cov"], layout, G), TOL, "C_plus = P")
    _close(out.xi_plus, xi + out.delta_xi, 1e-15, "xi_plus = xi + delta_xi")


def test_mean_field_step_equals_quadrature_divergence_and_power_posterior() -> None:
    r"""At :math:`K = 2`: divergences :math:`\int q\log(q/p_a)` and power-posterior moments by grid integration."""
    K, M = 2, 3
    pts, cell = _grid()
    g = _gen(34)
    nu = 4.0 * (torch.rand(M, K, generator=g, dtype=DT) - 0.5)
    lam = 0.5 + 1.5 * torch.rand(M, K, generator=g, dtype=DT)
    log_pi = torch.randn(M, generator=g, dtype=DT)
    xi = 2.0 * torch.rand(K, generator=g, dtype=DT) - 1.0
    Cd = _spd((), K, seed=35)
    log_q = _log_density_grid(pts, xi, Cd)
    q = torch.exp(log_q)
    log_p = [_log_density_grid(pts, nu[a], torch.diag(lam[a])) for a in range(M)]
    kl_quad = torch.stack([(q * (log_q - log_p[a])).sum() * cell for a in range(M)])
    r_quad = torch.softmax(torch.log_softmax(log_pi, dim=0) - kl_quad, dim=0)
    log_power = log_q + sum(r_quad[a] * log_p[a] for a in range(M))
    _, mean_quad, cov_quad = _grid_moments(pts, log_power, cell)
    out = pa.prototype_mean_field_posterior(xi[None], Cd[None], nu, lam.log(), log_pi)
    _close(out.responsibilities[0], r_quad, TOL_QUAD, "responsibilities by quadrature")
    _close(out.xi_plus[0], mean_quad, TOL_QUAD, "power-posterior mean by quadrature")
    _close(out.C_plus[0], cov_quad, TOL_QUAD, "power-posterior covariance by quadrature")


@pytest.mark.parametrize("full", (False, True))
def test_template_kl_logits_formula_and_gaussian_template_responsibilities(full: bool) -> None:
    R, K, e = 4, 4, 2
    M = e * K
    nu, log_lambda, log_pi = _templates(K, M, seed=36)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=_gen(37), dtype=DT)
    Cd = _spd((R,), K, seed=38) if full else torch.diag_embed(0.5 + torch.rand(R, K, generator=_gen(38), dtype=DT))
    c = torch.diagonal(Cd, dim1=-2, dim2=-1)
    expected = torch.stack([log_pi[a] - 0.5 * ((c + (xi - nu[a]).square()) / lam[a] + log_lambda[a]).sum(-1)
                            for a in range(M)], dim=-1)                                      # (R, M)
    got = pa.template_kl_logits(xi, c, nu, log_lambda, log_pi)
    _close(got, expected, 1e-12, "template_kl_logits")
    # Its softmax is the divergence label step: the row constant -(K + log det C)/2 cancels.
    for i in range(R):
        _close(torch.softmax(got[i], dim=0), _mean_field_reference(xi[i], Cd[i], nu, lam, log_pi)["weights"],
               TOL, "softmax of logits vs divergence labels")
    template = GaussianTemplateBlockMLP(K, e, 0.0).double()
    with torch.no_grad():
        template.nu.copy_(nu)
        template.log_lambda.copy_(log_lambda)
        template.log_pi.copy_(log_pi)
    sigma = Cd if full else c
    _close(template.responsibilities(xi, sigma), torch.softmax(got, dim=-1), 1e-13,
           "GaussianTemplateBlockMLP.responsibilities")
    prototype = PrototypeAgentBlockMLP(K, e, 0.0, inference="mean_field").double()
    prototype.load_state_dict({"nu": nu, "log_lambda": log_lambda, "log_pi": log_pi})
    _close(prototype.posterior(xi, sigma).responsibilities, template.responsibilities(xi, sigma), 1e-13,
           "module responsibilities")


# ----------------------------------------------------------------------------------------------
# 4. One template, and the precise-template and broad-template limits
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_single_template_exact_equals_mean_field(layout: str) -> None:
    r"""With :math:`M = 1` both modes are the normalized product :math:`\mathcal N(\xi, C)\,p_1`; the spread vanishes."""
    R, K, G = 3, 4, 2
    nu, log_lambda, log_pi = _templates(K, 1, seed=41)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=_gen(42), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=43)
    exact = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True)
    mf = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi)
    assert torch.equal(exact.weights, torch.ones_like(exact.weights))
    assert torch.equal(mf.responsibilities, torch.ones_like(mf.responsibilities))
    assert float(exact.R.abs().max()) <= 1e-14
    _close(exact.xi_plus, mf.xi_plus, 1e-11, "means")
    _close(exact.C_plus, mf.C_plus, 1e-11, "covariances")
    for i in range(R):
        C_inv = torch.linalg.inv(Cd[i])
        P = torch.linalg.inv(C_inv + torch.diag(1.0 / lam[0]))
        _close(exact.xi_plus[i], P @ (C_inv @ xi[i] + nu[0] / lam[0]), TOL, "normalized-product mean")
        _close(exact.C_plus[i], _project(P, layout, G), TOL, "normalized-product covariance")


@pytest.mark.parametrize("layout", ("diagonal", "dense"))
def test_precise_template_limit(layout: str) -> None:
    r"""Item (d), :math:`\Theta = I`, :math:`\widetilde C_{ia} = \Lambda_a = \epsilon I \to 0`.

    Exact: :math:`w \to w^0_a \propto \pi_a\mathcal N(\nu_a; \xi, C)`, :math:`\xi^+ \to \sum_a w^0_a\nu_a` and
    :math:`C^+ \to \sum_a w^0_a(\nu_a - \xi^{+0})(\nu_a - \xi^{+0})^\top`, at first order in :math:`\epsilon`.
    Mean field: the label cost :math:`|\nu_a - \xi|^2/2\epsilon` diverges, so :math:`r \to e_{a^*}` for the
    nearest template and :math:`\xi^+ \to \nu_{a^*}` with error :math:`\epsilon\,C^{-1}(\xi - \nu_{a^*})`; the two
    modes differ in this limit, as Section 4 of the note states.
    """
    R, K, M = 2, 3, 3
    g = _gen(44)
    xi = torch.randn(R, K, generator=g, dtype=DT)
    nu = xi.mean(0) + 0.6 * torch.randn(M, K, generator=g, dtype=DT)
    log_pi = 0.5 * torch.randn(M, generator=g, dtype=DT)
    C, Cd = _working(layout, R, K, 1, seed=45)
    log_prior = torch.log_softmax(log_pi, dim=0)
    w0 = torch.stack([torch.softmax(torch.stack([log_prior[a] + _log_gaussian(nu[a], xi[i], Cd[i])
                                                  for a in range(M)]), dim=0) for i in range(R)])     # (R, M)
    mean0 = w0 @ nu                                                                                  # (R, K)
    cov0 = torch.stack([sum(w0[i, a] * torch.outer(nu[a] - mean0[i], nu[a] - mean0[i]) for a in range(M))
                        for i in range(R)])
    nearest = (nu[None] - xi[:, None]).square().sum(-1).argmin(-1)                                   # (R,)
    gaps = (nu[None] - xi[:, None]).square().sum(-1).sort(-1).values
    assert float((gaps[:, 1] - gaps[:, 0]).min()) > 0.05                                             # a clear nearest template
    assert float(w0.max()) < 0.9                                                                     # the exact limit stays a mixture
    errors: Dict[str, List[float]] = {k: [] for k in ("w", "mean", "cov", "mf_mean")}
    for eps in (1e-4, 5e-5):
        log_lambda = torch.full((M, K), math.log(eps), dtype=DT)
        exact = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True)
        errors["w"].append(float((exact.weights - w0).abs().max()))
        errors["mean"].append(float((exact.xi_plus - mean0).abs().max()))
        errors["cov"].append(float((exact.C_plus - _project(cov0, layout, 1)).abs().max()))
        mf = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi)
        assert float(mf.responsibilities[torch.arange(R), nearest].min()) > 1.0 - 1e-9
        errors["mf_mean"].append(float((mf.xi_plus - nu[nearest]).abs().max()))
        # The mean-field error has the first-order form eps C^{-1}(xi - nu_a*).
        first_order = torch.stack([eps * torch.linalg.inv(Cd[i]) @ (xi[i] - nu[nearest[i]]) for i in range(R)])
        _close(mf.xi_plus - nu[nearest], first_order, 1e-6, "mean-field first-order error")
    for name, (e1, e2) in errors.items():
        assert e1 > 1e-8, f"{name}: no first-order deviation to measure"
        assert e2 < 1e-3 and abs(e2 / e1 - 0.5) < 0.02, f"{name}: errors {e1:.3e}, {e2:.3e} are not first order"


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("inference", INFERENCE)
def test_broad_template_limit_first_order_coefficients(layout: str, inference: str) -> None:
    r"""Common :math:`\lambda \to \infty`: :math:`\lambda(\xi^+ - \xi) \to C\sum_a\pi_a(\nu_a - \xi)`, :math:`\lambda(C - C^+) \to C^2`, weights :math:`\to \pi`.

    Exact: :math:`(C + \lambda I)^{-1} = \lambda^{-1}(I - C/\lambda + \dots)` and :math:`R = O(\lambda^{-2})`;
    mean field: :math:`P = C(I + C/\lambda)^{-1} = C - C^2/\lambda + O(\lambda^{-2})`. The residuals are
    :math:`O(1/\lambda)`, so they fall by ten when :math:`\lambda` grows by ten.
    """
    R, K, G, M = 2, 4, 2, 3
    fn = pa.prototype_exact_posterior if inference == "exact" else pa.prototype_mean_field_posterior
    nu, _, log_pi = _templates(K, M, seed=46)
    pi = torch.softmax(log_pi, dim=0)
    xi = torch.randn(R, K, generator=_gen(47), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=48)
    coef_mean = torch.stack([Cd[i] @ (pi @ nu - xi[i]) for i in range(R)])                            # (R, K)
    coef_cov = _project(Cd @ Cd, layout, G)
    residuals = []
    for lam in (1e5, 1e6):
        out = fn(xi, C, nu, torch.full((M, K), math.log(lam), dtype=DT), log_pi)
        weights = out.weights if inference == "exact" else out.responsibilities
        residuals.append((float((lam * out.delta_xi - coef_mean).abs().max()),
                          float((lam * (C - out.C_plus) - coef_cov).abs().max()),
                          float((weights - pi).abs().max())))
    for name, (r1, r2) in zip(("mean", "covariance", "weights"), zip(*residuals)):
        assert r1 > 1e-9, f"{name}: no first-order residual to measure"
        assert r2 < 1e-4 and 0.05 < r2 / r1 < 0.2, f"{name}: residuals {r1:.3e}, {r2:.3e} are not O(1/lambda)"


# ----------------------------------------------------------------------------------------------
# 5. Module contracts: covariance, dropout, dtype, errors, locality
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("inference", INFERENCE)
def test_forward_moments_contracts(layout: str, inference: str) -> None:
    K, e, G = 4, 2, 2
    passthrough = _module(K, e, inference, "passthrough", seed=51)
    posterior = _module(K, e, inference, "posterior", seed=51)
    assert all(torch.equal(passthrough.state_dict()[k], v) for k, v in posterior.state_dict().items())
    mu = torch.randn(2, 3, K, generator=_gen(52), dtype=DT)
    C, Cd = _working(layout, 6, K, G, seed=53)
    sigma = C.reshape(2, 3, *C.shape[1:])
    kept = passthrough.forward_moments(mu, sigma)
    assert kept.sigma is sigma and kept.jacobian is None and kept.mu.dtype == mu.dtype
    new = posterior.forward_moments(mu, sigma)
    assert new.sigma.shape == sigma.shape and new.sigma.dtype == sigma.dtype and new.jacobian is None
    assert torch.equal(new.mu, kept.mu)
    mean_ref, cov_ref, _ = _reference(inference, mu.reshape(6, K), Cd, posterior.nu, posterior.log_lambda.exp(),
                                      posterior.log_pi)
    _close(kept.mu.reshape(6, K), mean_ref, TOL, "mu_out")
    _close(new.sigma.reshape(6, *C.shape[1:]), _project(cov_ref, layout, G), TOL, "posterior covariance")
    if layout == "diagonal":
        assert bool((new.sigma > 0).all())
    else:
        assert torch.equal(new.sigma, new.sigma.transpose(-1, -2))
        assert float(torch.linalg.eigvalsh(new.sigma).min()) > 0.0
    assert torch.equal(posterior.forward(mu, sigma), new.mu)
    # posterior(): the covariance is computed only when the contract needs it, unless asked for.
    assert passthrough.posterior(mu, sigma).C_plus is None
    assert posterior.posterior(mu, sigma).C_plus is not None
    assert passthrough.posterior(mu, sigma, need_covariance=True).C_plus is not None
    if inference == "exact":
        asked = passthrough.posterior(mu, sigma, need_covariance=False, need_precision=True)
        assert asked.C_plus is None and asked.R is None and asked.Lambda_bar is not None
        assert posterior.posterior(mu, sigma).Lambda_bar is None
    with pytest.raises(ValueError):
        posterior.forward_moments(mu, sigma, hidden_bias=torch.zeros(2, 3, 1, dtype=DT))


def test_dropout_acts_on_the_mean_increment_only() -> None:
    K = 4
    mlp = _module(K, 2, "mean_field", "posterior", seed=54, dropout=0.5)
    mu = torch.randn(8, K, generator=_gen(55), dtype=DT)
    sigma = 0.5 + torch.rand(8, K, generator=_gen(56), dtype=DT)
    mean_ref, cov_ref, _ = _reference("mean_field", mu, torch.diag_embed(sigma), mlp.nu, mlp.log_lambda.exp(), mlp.log_pi)
    delta_ref = mean_ref - mu
    evaluated = mlp.forward_moments(mu, sigma)                                   # eval mode: no dropout
    _close(evaluated.mu, mu + delta_ref, TOL, "eval mean")
    _close(evaluated.sigma, torch.diagonal(cov_ref, dim1=-2, dim2=-1), TOL, "eval covariance")
    mlp.train()
    torch.manual_seed(1)
    trained = mlp.forward_moments(mu, sigma)
    increment = trained.mu - mu
    kept = (increment - 2.0 * delta_ref).abs() <= 1e-12 * (1.0 + delta_ref.abs())   # inverted dropout: 1/(1-p) = 2
    dropped = increment.abs() <= 1e-15
    assert bool((kept | dropped).all()) and bool(kept.any()) and bool(dropped.any())
    assert torch.equal(trained.sigma, evaluated.sigma)                           # the covariance is never dropped


def test_compute_dtype_policy() -> None:
    R, K, M = 2, 3, 2
    nu, log_lambda, log_pi = _templates(K, M, seed=57)
    xi = torch.randn(R, K, generator=_gen(58), dtype=DT)
    c = 0.5 + torch.rand(R, K, generator=_gen(59), dtype=DT)
    for fn in (pa.prototype_exact_posterior, pa.prototype_mean_field_posterior):
        out32 = fn(xi.float(), c.float(), nu.float(), log_lambda.float(), log_pi.float())
        assert all(t.dtype == torch.float32 for t in out32 if t is not None)
        mixed = fn(xi.float(), c.float(), nu, log_lambda.float(), log_pi.float())   # one float64 input
        assert all(t.dtype == torch.float64 for t in mixed if t is not None)
        out64 = fn(xi, c, nu, log_lambda, log_pi)
        _close(out32.xi_plus, out64.xi_plus, 1e-5, "float32 vs float64 mean")      # float32 rounding
        _close(mixed.xi_plus, out64.xi_plus, 1e-6, "mixed vs float64 mean")
    mlp = PrototypeAgentBlockMLP(K, 1, 0.0, covariance_contract="posterior")        # float32 parameters
    result = mlp.forward_moments(xi.float(), c.float())
    assert result.mu.dtype == torch.float32 and result.sigma.dtype == torch.float32


def test_invalid_inputs_raise_value_errors() -> None:
    xi = torch.zeros(1, 2, dtype=DT)
    nu, log_lambda, log_pi = torch.zeros(1, 2, dtype=DT), torch.zeros(1, 2, dtype=DT), torch.zeros(1, dtype=DT)
    with pytest.raises(ValueError, match="not positive definite"):                 # dense Cholesky, no repair
        pa.prototype_exact_posterior(xi, torch.tensor([[[1.0, 0.0], [0.0, -5.0]]], dtype=DT), nu, log_lambda, log_pi)
    with pytest.raises(ValueError, match="not positive definite"):                 # blocks
        pa.prototype_exact_posterior(xi, torch.tensor([[[[1.0]], [[-5.0]]]], dtype=DT), nu, log_lambda, log_pi)
    for bad in ([[1.0, -5.0]], [[1.0, float("nan")]]):
        with pytest.raises(ValueError, match="positive and finite"):
            pa.prototype_exact_posterior(xi, torch.tensor(bad, dtype=DT), nu, log_lambda, log_pi)
    with pytest.raises(ValueError, match="not positive definite"):
        pa.prototype_mean_field_posterior(xi, torch.tensor([[[1.0, 0.0], [0.0, -5.0]]], dtype=DT), nu, log_lambda, log_pi)
    for fn in (pa.prototype_exact_posterior, pa.prototype_mean_field_posterior):
        with pytest.raises(ValueError):
            fn(xi, torch.ones(1, 2, dtype=DT), torch.zeros(1, 3, dtype=DT), torch.zeros(1, 3, dtype=DT), log_pi)
        with pytest.raises(ValueError, match="working covariance"):
            fn(xi, torch.ones(1, 3, dtype=DT), nu, log_lambda, log_pi)
    with pytest.raises(ValueError):
        pa.prototype_exact_posterior(xi, torch.ones(1, 2, dtype=DT), nu, log_lambda, log_pi, tile_bytes=0)
    with pytest.raises(ValueError):
        pa.prototype_exact_posterior(torch.zeros(0, 2, dtype=DT), torch.ones(0, 2, dtype=DT), nu, log_lambda, log_pi)
    with pytest.raises(ValueError):
        PrototypeAgentBlockMLP(4, 1, 0.0, covariance_contract="delta_full")
    with pytest.raises(ValueError):
        PrototypeAgentBlockMLP(4, 1, 0.0, inference="variational")
    for bad_variance in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            PrototypeAgentBlockMLP(4, 1, 0.0, template_init_variance=bad_variance)


@pytest.mark.parametrize("inference", INFERENCE)
def test_tokens_are_independent_and_batch_shape_free(inference: str) -> None:
    r"""The block acts token by token: perturbing token 2 leaves the others unchanged; batch axes are immaterial."""
    R, K, M = 4, 4, 5
    fn = pa.prototype_exact_posterior if inference == "exact" else pa.prototype_mean_field_posterior
    nu, log_lambda, log_pi = _templates(K, M, seed=61)
    xi = torch.randn(R, K, generator=_gen(62), dtype=DT)
    C = _spd((R,), K, seed=63)
    base = fn(xi, C, nu, log_lambda, log_pi)
    xi2, C2 = xi.clone(), C.clone()
    xi2[2] += 1.0
    C2[2] = 0.5 * C2[2]
    perturbed = fn(xi2, C2, nu, log_lambda, log_pi)
    weights = (lambda o: o.weights) if inference == "exact" else (lambda o: o.responsibilities)
    for i in (0, 1, 3):
        _close(perturbed.xi_plus[i], base.xi_plus[i], 1e-13, f"token {i} mean")
        _close(perturbed.C_plus[i], base.C_plus[i], 1e-13, f"token {i} covariance")
        _close(weights(perturbed)[i], weights(base)[i], 1e-13, f"token {i} weights")
    assert float((perturbed.xi_plus[2] - base.xi_plus[2]).abs().max()) > 1e-3
    batched = fn(xi.reshape(2, 2, K), C.reshape(2, 2, K, K), nu, log_lambda, log_pi)
    _close(batched.xi_plus.reshape(R, K), base.xi_plus, 1e-13, "batched mean")
    _close(batched.C_plus.reshape(R, K, K), base.C_plus, 1e-13, "batched covariance")
    _close(weights(batched).reshape(R, M), weights(base), 1e-13, "batched weights")


# ----------------------------------------------------------------------------------------------
# 6. Gradients: central finite differences; tiled exact path
# ----------------------------------------------------------------------------------------------

def _fd_case(layout: str, seed: int) -> List[torch.Tensor]:
    R, K, M = 2, 4, 3
    nu, log_lambda, log_pi = _templates(K, M, seed, scale=0.3)
    xi = torch.randn(R, K, generator=_gen(seed + 1), dtype=DT)
    if layout == "diagonal":
        cov_param = 0.5 + torch.rand(R, K, generator=_gen(seed + 2), dtype=DT)
    elif layout == "blocks":
        cov_param = 0.5 * torch.randn(R, 2, 2, 2, generator=_gen(seed + 2), dtype=DT)
    else:
        cov_param = 0.5 * torch.randn(R, K, K, generator=_gen(seed + 2), dtype=DT)
    return [nu, log_lambda, log_pi, xi, cov_param]


def _covariance_from(layout: str, cov_param: torch.Tensor) -> torch.Tensor:
    r"""A positive definite working covariance from a free parameter: :math:`AA^\top + \tfrac12 I` for the matrix layouts."""
    if layout == "diagonal":
        return cov_param
    return cov_param @ cov_param.transpose(-1, -2) + 0.5 * torch.eye(cov_param.shape[-1], dtype=DT)


def _scalar_loss(inference: str, layout: str, probe: Sequence[torch.Tensor], seed: int,
                 tile_bytes: int = pa.PROTOTYPE_TILE_BYTES) -> Callable[..., torch.Tensor]:
    r"""A random linear functional of every output field (fixed coefficients drawn once from the probe's shapes)."""
    g = _gen(seed)

    def outputs(nu, log_lambda, log_pi, xi, cov_param):
        C = _covariance_from(layout, cov_param)
        if inference == "exact":
            out = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True,
                                               need_precision=True, tile_bytes=tile_bytes)
            return (out.xi_plus, out.C_plus, out.weights, out.Lambda_bar, out.R)
        out = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True)
        return (out.xi_plus, out.C_plus, out.responsibilities, out.D, out.eta)

    with torch.no_grad():
        coefficients = [torch.randn(*t.shape, generator=g, dtype=DT) for t in outputs(*probe)]

    def loss(*args):
        return sum((coef * term).sum() for coef, term in zip(coefficients, outputs(*args)))

    return loss


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("inference", INFERENCE)
def test_finite_difference_gradients_of_pure_functions(layout: str, inference: str) -> None:
    params = _fd_case(layout, seed=71)
    loss = _scalar_loss(inference, layout, params, seed=72)
    leaves = [p.clone().requires_grad_(True) for p in params]
    auto = torch.autograd.grad(loss(*leaves), leaves)
    fd_tensors = [p.clone() for p in params]
    fd = _central_difference(lambda: loss(*fd_tensors), fd_tensors)
    for a, f, name in zip(auto, fd, ("nu", "log_lambda", "log_pi", "xi", "covariance parameter")):
        assert float(f.abs().max()) > 1e-6, f"d/d{name}: the finite-difference gradient vanishes"
        _close(a, f, TOL_FD, f"d/d{name}")


@pytest.mark.parametrize("inference", INFERENCE)
def test_module_finite_difference_gradients_through_parameters_and_inputs(inference: str) -> None:
    K, e, R = 3, 2, 4
    mlp = _module(K, e, inference, "posterior", seed=73)
    assert [name for name, _ in mlp.named_parameters()] == ["nu", "log_lambda", "log_pi"]
    mu = torch.randn(R, K, generator=_gen(74), dtype=DT).requires_grad_(True)
    sigma = (0.5 + torch.rand(R, K, generator=_gen(75), dtype=DT)).requires_grad_(True)
    g = _gen(76)
    a, b = torch.randn(R, K, generator=g, dtype=DT), torch.randn(R, K, generator=g, dtype=DT)

    def loss() -> torch.Tensor:
        result = mlp.forward_moments(mu, sigma)
        return (a * result.mu).sum() + (b * result.sigma).sum()

    tensors = [mlp.nu, mlp.log_lambda, mlp.log_pi, mu, sigma]
    auto = torch.autograd.grad(loss(), tensors)
    fd = _central_difference(loss, tensors)
    for ag, fg, name in zip(auto, fd, ("nu", "log_lambda", "log_pi", "mu", "sigma")):
        assert float(fg.abs().max()) > 1e-6, f"d/d{name}: the finite-difference gradient vanishes"
        _close(ag, fg, TOL_FD, f"d/d{name}")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_tiled_exact_path_equals_single_tile_values_gradients_and_reference(layout: str) -> None:
    r"""``tile_bytes=1`` forces one (token, template) pair per checkpointed tile and an online merge of every chunk."""
    R, K, G, M = 5, 4, 2, 7
    nu, log_lambda, log_pi = _templates(K, M, seed=81)
    lam = log_lambda.exp()
    xi = torch.randn(R, K, generator=_gen(82), dtype=DT)
    C, Cd = _working(layout, R, K, G, seed=83)
    probe = [nu, log_lambda, log_pi, xi, C]
    g = _gen(84)
    with torch.no_grad():
        shapes = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True, need_precision=True)
        coefficients = [torch.randn(*t.shape, generator=g, dtype=DT)
                        for t in (shapes.xi_plus, shapes.C_plus, shapes.weights, shapes.Lambda_bar, shapes.R)]

    def run(tile_bytes: int):
        leaves = [t.clone().requires_grad_(True) for t in probe]
        out = pa.prototype_exact_posterior(leaves[3], leaves[4], leaves[0], leaves[1], leaves[2],
                                           need_covariance=True, need_precision=True, tile_bytes=tile_bytes)
        loss = sum((coef * term).sum() for coef, term in zip(
            coefficients, (out.xi_plus, out.C_plus, out.weights, out.Lambda_bar, out.R)))
        return out, torch.autograd.grad(loss, leaves)

    single, single_grads = run(pa.PROTOTYPE_TILE_BYTES)
    tiled, tiled_grads = run(1)
    for field in ("xi_plus", "delta_xi", "C_plus", "weights", "u", "Lambda_bar", "R"):
        _close(getattr(tiled, field), getattr(single, field), 1e-12, field)
    for a, b, name in zip(single_grads, tiled_grads, ("nu", "log_lambda", "log_pi", "xi", "C")):
        _close(b, a, 1e-11, f"tiled d/d{name}")
    for i in range(R):
        ref = _exact_reference(xi[i], Cd[i], nu, lam, log_pi)
        _close(tiled.xi_plus[i], ref["mean"], TOL, "tiled mean vs reference")
        _close(tiled.C_plus[i], _project(ref["cov"], layout, G), TOL, "tiled covariance vs reference")
        _close(tiled.weights[i], ref["weights"], TOL, "tiled weights vs reference")


# ----------------------------------------------------------------------------------------------
# 7. Reference coordinates: frame helpers, hand-built conjugation, passive gauge transformations
# ----------------------------------------------------------------------------------------------

B_, N_, G_, D_ = 2, 3, 2, 2
K_ = G_ * D_


def _frame(kind: str, seed: int, scale: float = 0.4) -> Tuple[CanonicalFrameContext, torch.Tensor]:
    r"""A realized frame :math:`U_i = \exp(\text{scale}\cdot X_i)` (compact blocks or dense) and its dense matrix."""
    g = _gen(seed)
    if kind == "blocks":
        blocks = torch.linalg.matrix_exp(scale * torch.randn(B_, N_, G_, D_, D_, generator=g, dtype=DT))
        return CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True), _block_diag(blocks)
    U = torch.linalg.matrix_exp(scale * torch.randn(B_, N_, K_, K_, generator=g, dtype=DT))
    return CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=False), U


def _run_reference_block(mlp: PrototypeAgentBlockMLP, frame: CanonicalFrameContext,
                         mu: torch.Tensor, sigma: torch.Tensor) -> BeliefState:
    belief = BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(B_, N_, 1, dtype=DT))
    return _reference_coordinate_block(belief, block_norm=None, block_mlp=mlp, frame=frame, capture=None,
                                       hidden_bias=None)


def _matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (A @ x.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("frame_kind", ("blocks", "dense"))
def test_frame_helpers_match_dense_formulas(frame_kind: str) -> None:
    r""":math:`C = U^{-1}\Sigma U^{-\top}` in its compact layout, and :math:`\Sigma = UCU^\top` back."""
    frame, U = _frame(frame_kind, seed=91)
    U_inv = torch.linalg.inv(U)
    mu = torch.randn(B_, N_, K_, generator=_gen(92), dtype=DT)
    sigma_d = 0.5 + torch.rand(B_, N_, K_, generator=_gen(93), dtype=DT)
    Sigma_full = _spd((B_, N_), K_, seed=94)
    C_diag = U_inv @ torch.diag_embed(sigma_d) @ U_inv.transpose(-1, -2)
    compact = reference_covariance_compact(mu, sigma_d, frame)
    if frame_kind == "blocks":
        assert compact.shape == (B_, N_, G_, D_, D_)
        _close(_block_diag(compact), C_diag, 1e-12, "block reference covariance")
        _close(C_diag - _block_diag(_diag_blocks(C_diag, G_)), torch.zeros_like(C_diag), 1e-12,
               "a block frame gives a block-diagonal C")
    else:
        assert compact.shape == (B_, N_, K_, K_)
        _close(compact, C_diag, 1e-12, "dense reference covariance of a diagonal belief")
    assert torch.equal(compact, compact.transpose(-1, -2))
    C_full = U_inv @ Sigma_full @ U_inv.transpose(-1, -2)
    compact_full = reference_covariance_compact(mu, Sigma_full, frame)
    assert compact_full.shape == (B_, N_, K_, K_)
    _close(compact_full, C_full, 1e-12, "dense reference covariance of a full belief")
    for C_ref, dense_C in ((compact, C_diag), (compact_full, C_full)):
        Sigma = U @ dense_C @ U.transpose(-1, -2)
        _close(local_covariance(mu, C_ref, frame, diagonal=False), Sigma, 1e-12, "local covariance")
        _close(local_covariance(mu, C_ref, frame, diagonal=True), torch.diagonal(Sigma, dim1=-2, dim2=-1), 1e-12,
               "local variances")
    _close(local_covariance(mu, compact, frame, diagonal=True), sigma_d, 1e-12, "round trip of diagonal beliefs")
    if frame_kind == "dense":
        with pytest.raises(ValueError):
            local_covariance(mu, torch.eye(D_, dtype=DT).expand(B_, N_, G_, D_, D_), frame, diagonal=True)


@pytest.mark.parametrize("frame_kind", ("blocks", "dense"))
@pytest.mark.parametrize("beliefs", ("diagonal", "full"))
@pytest.mark.parametrize("inference", INFERENCE)
@pytest.mark.parametrize("contract", ("passthrough", "posterior"))
def test_reference_block_equals_hand_built_conjugation(frame_kind: str, beliefs: str, inference: str,
                                                       contract: str) -> None:
    r""":math:`\mu^+ = U\xi^+`, :math:`\Sigma^+ = UC^+U^\top` from the dense reference moments and the reference posterior."""
    mlp = _module(K_, 2, inference, contract, seed=101)
    frame, U = _frame(frame_kind, seed=102)
    mu = torch.randn(B_, N_, K_, generator=_gen(103), dtype=DT)
    sigma = (_spd((B_, N_), K_, seed=104) if beliefs == "full"
             else 0.5 + torch.rand(B_, N_, K_, generator=_gen(104), dtype=DT))
    Sigma = sigma if beliefs == "full" else torch.diag_embed(sigma)
    U_inv = torch.linalg.inv(U)
    xi = _matvec(U_inv, mu)
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)
    rows = B_ * N_
    mean_ref, cov_ref, _ = _reference(inference, xi.reshape(rows, K_), C.reshape(rows, K_, K_),
                                      mlp.nu, mlp.log_lambda.exp(), mlp.log_pi)
    out = _run_reference_block(mlp, frame, mu, sigma)
    _close(out.mu, _matvec(U, mean_ref.reshape(B_, N_, K_)), TOL, "mu_plus = U xi_plus")
    if contract == "passthrough":
        assert out.sigma is sigma
    else:
        Sigma_plus = U @ cov_ref.reshape(B_, N_, K_, K_) @ U.transpose(-1, -2)
        expected = Sigma_plus if beliefs == "full" else torch.diagonal(Sigma_plus, dim1=-2, dim2=-1)
        assert out.sigma.shape == sigma.shape and out.sigma.dtype == sigma.dtype
        _close(out.sigma, expected, TOL, "Sigma_plus = U C_plus U^T")


@pytest.mark.parametrize("frame_kind", ("blocks", "dense"))
@pytest.mark.parametrize("inference", INFERENCE)
def test_reference_block_is_covariant_under_passive_gauge_transformations(frame_kind: str, inference: str) -> None:
    r"""Under :math:`(\mu, \Sigma, U) \mapsto (g\mu, g\Sigma g^\top, gU)` the pair :math:`(\xi, C)` is invariant and :math:`(\mu^+, \Sigma^+) \mapsto (g\mu^+, g\Sigma^+g^\top)`."""
    mlp = _module(K_, 2, inference, "posterior", seed=111)
    frame, U = _frame(frame_kind, seed=112)
    g_frame, g = _frame(frame_kind, seed=113, scale=0.7)                        # g in prod_g GL(d) or GL(K)
    mu = torch.randn(B_, N_, K_, generator=_gen(114), dtype=DT)
    Sigma = _spd((B_, N_), K_, seed=115)
    # Explicit invariance of the reference moments.
    U_inv, gU = torch.linalg.inv(U), g @ U
    gU_inv = torch.linalg.inv(gU)
    _close(_matvec(gU_inv, _matvec(g, mu)), _matvec(U_inv, mu), 1e-12, "xi invariant")
    _close(gU_inv @ (g @ Sigma @ g.transpose(-1, -2)) @ gU_inv.transpose(-1, -2),
           U_inv @ Sigma @ U_inv.transpose(-1, -2), 1e-12, "C invariant")
    moved_frame = CanonicalFrameContext(g_frame.forward @ frame.forward, frame.inverse @ g_frame.inverse,
                                        block_diagonal=frame.block_diagonal)
    base = _run_reference_block(mlp, frame, mu, Sigma)
    moved = _run_reference_block(mlp, moved_frame, _matvec(g, mu), g @ Sigma @ g.transpose(-1, -2))
    _close(moved.mu, _matvec(g, base.mu), TOL, "mu_plus covariant")
    _close(moved.sigma, g @ base.sigma @ g.transpose(-1, -2), TOL, "Sigma_plus covariant")
    # Diagonal beliefs: a diagonal g keeps the diagonal layout, Sigma -> diag(s^2 sigma).
    sigma_d = 0.5 + torch.rand(B_, N_, K_, generator=_gen(116), dtype=DT)
    s = torch.exp(0.5 * torch.randn(B_, N_, K_, generator=_gen(117), dtype=DT))
    if frame_kind == "blocks":
        s_blocks = s.reshape(B_, N_, G_, D_)
        scaled_frame = CanonicalFrameContext(s_blocks.unsqueeze(-1) * frame.forward,
                                             frame.inverse / s_blocks.unsqueeze(-2), block_diagonal=True)
    else:
        scaled_frame = CanonicalFrameContext(s.unsqueeze(-1) * frame.forward, frame.inverse / s.unsqueeze(-2),
                                             block_diagonal=False)
    base_d = _run_reference_block(mlp, frame, mu, sigma_d)
    moved_d = _run_reference_block(mlp, scaled_frame, s * mu, s * s * sigma_d)
    _close(moved_d.mu, s * base_d.mu, TOL, "diagonal beliefs: mu_plus covariant")
    _close(moved_d.sigma, s * s * base_d.sigma, TOL, "diagonal beliefs: variances covariant")
    # Negative control: the same block in local coordinates is not covariant, so the test has power.
    local_moved = mlp.forward_moments(_matvec(g, mu), g @ Sigma @ g.transpose(-1, -2)).mu
    assert float((local_moved - _matvec(g, mlp.forward_moments(mu, Sigma).mu)).abs().max()) > 1e-3


@pytest.mark.parametrize("layout", ("diagonal", "dense"))
@pytest.mark.parametrize("inference", INFERENCE)
def test_two_rounds_equal_sequential_conditioning_with_gaussian_collapse(layout: str, inference: str) -> None:
    r"""Two blocks with ``posterior`` covariance condition sequentially, collapsing to the layout's Gaussian between rounds."""
    R, K, e = 3, 4, 2
    mlp = _module(K, e, inference, "posterior", seed=121)
    lam = mlp.log_lambda.exp()
    mu = torch.randn(R, K, generator=_gen(122), dtype=DT)
    sigma, Sigma = _working(layout, R, K, 1, seed=123)
    first = mlp.forward_moments(mu, sigma)
    second = mlp.forward_moments(first.mu, first.sigma)
    mean1, cov1, _ = _reference(inference, mu, Sigma, mlp.nu, lam, mlp.log_pi)
    collapsed1 = torch.diag_embed(torch.diagonal(cov1, dim1=-2, dim2=-1)) if layout == "diagonal" else cov1
    mean2, cov2, _ = _reference(inference, mean1, collapsed1, mlp.nu, lam, mlp.log_pi)
    _close(second.mu, mean2, TOL, "second-round mean")
    _close(second.sigma, _project(cov2, layout, 1), TOL, "second-round covariance")


# ----------------------------------------------------------------------------------------------
# 8. Statistics, initialization, registry, configuration, configured model path
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("inference", INFERENCE)
def test_statistics_at_initialization_and_against_formulas(inference: str) -> None:
    K, e = 4, 2
    keys = {"prototype_weight_entropy", "prototype_effective_templates", "prototype_gain_fraction",
            "prototype_template_variance", "prototype_prior_entropy", "prototype_label_staleness"}
    torch.manual_seed(0)
    init = PrototypeAgentBlockMLP(K, e, 0.0, inference=inference).double()
    lam0 = float(init.log_lambda.exp()[0, 0])                                     # float32 log(10), about 10
    assert torch.equal(init.log_lambda, torch.full_like(init.log_lambda, float(init.log_lambda[0, 0])))
    mu = torch.randn(5, K, generator=_gen(131), dtype=DT)
    for sigma in (torch.ones(5, K, dtype=DT), torch.eye(K, dtype=DT).expand(5, K, K)):
        stats = init.statistics(mu, sigma)
        assert set(stats) == keys and all(isinstance(v, float) for v in stats.values())
        assert abs(stats["prototype_gain_fraction"] - 1.0 / (1.0 + lam0)) < 1e-12   # c = 1: c/(c + lambda_0)
        assert abs(stats["prototype_gain_fraction"] - 1.0 / 11.0) < 1e-7
        assert abs(stats["prototype_template_variance"] - lam0) < 1e-12
        assert abs(stats["prototype_prior_entropy"] - math.log(K * e)) < 1e-12
        assert 1.0 <= stats["prototype_effective_templates"] <= K * e
    mlp = _module(K, e, inference, "posterior", seed=132)
    c = 0.5 + torch.rand(5, K, generator=_gen(133), dtype=DT)
    stats = mlp.statistics(mu, c)
    lam = mlp.log_lambda.exp()
    fn = _exact_reference if inference == "exact" else _mean_field_reference
    entropies, gains = [], []
    for i in range(5):
        ref = fn(mu[i], torch.diag(c[i]), mlp.nu, lam, mlp.log_pi)
        w = ref["weights"]
        entropies.append(float(-(w * w.log()).sum()))
        if inference == "exact":
            gains.append(float((c[i] * torch.diagonal(ref["Lambda_bar"])).sum()) / K)      # tr(C Lambda_bar)/K
        else:
            gains.append(float((torch.diagonal(ref["cov"]) * ref["D"]).sum()) / K)          # tr(P diag D)/K
    assert abs(stats["prototype_weight_entropy"] - sum(entropies) / 5) < 1e-10
    assert abs(stats["prototype_effective_templates"] - sum(math.exp(h) for h in entropies) / 5) < 1e-10
    assert abs(stats["prototype_gain_fraction"] - sum(gains) / 5) < 1e-10
    assert abs(stats["prototype_template_variance"] - math.exp(float(mlp.log_lambda.mean()))) < 1e-10
    prior = torch.softmax(mlp.log_pi, dim=0)
    assert abs(stats["prototype_prior_entropy"] - float(-(prior * prior.log()).sum())) < 1e-10


def test_module_initialization_attributes_and_state_dict() -> None:
    K, e, variance = 5, 3, 2.5
    M = e * K
    torch.manual_seed(0)
    mlp = PrototypeAgentBlockMLP(K, e, 0.0, inference="exact", covariance_contract="posterior",
                                 template_init_variance=variance)
    assert mlp.nu.shape == (M, K) and mlp.log_lambda.shape == (M, K) and mlp.log_pi.shape == (M,)
    assert float(mlp.nu.abs().max()) <= K ** -0.5 and float(mlp.nu.std()) > 0.0
    assert torch.equal(mlp.log_lambda, torch.full((M, K), math.log(variance)))
    assert torch.equal(mlp.log_pi, torch.zeros(M))
    assert mlp.covariance_contract == "posterior" and mlp.inference == "exact"
    assert type(mlp.tile_bytes) is int and mlp.tile_bytes == pa.PROTOTYPE_TILE_BYTES
    assert set(mlp.state_dict()) == {"nu", "log_lambda", "log_pi"}
    assert set(mlp.state_dict().keys()).isdisjoint({"tile_bytes"})
    torch.manual_seed(7)
    prototype = PrototypeAgentBlockMLP(K, e, 0.0)
    torch.manual_seed(7)
    template = GaussianTemplateBlockMLP(K, e, 0.0)
    assert torch.equal(prototype.nu, template.nu)                                 # the same uniform draw
    assert prototype.covariance_contract == "passthrough" and prototype.inference == "mean_field"
    assert torch.equal(prototype.log_lambda, torch.full((M, K), math.log(10.0)))


def test_registry_metadata_and_factory_forwarding() -> None:
    registration = get_block_mlp_registration("prototype_agents")
    assert registration.covariance_kinds == frozenset({"passthrough", "posterior"})
    assert registration.gauge_class == "coordinate_nonintertwiner" and not registration.intertwiner_compatible
    assert not registration.requires_frame_context
    assert registration.factory_options == frozenset({"inference", "template_init_variance", "cavi_steps"})
    for K, e, L in ((4, 2, 1), (6, 3, 2)):
        assert registration.parameter_count(embed_dim=K, irrep_dims=(2, K - 2), expansion=e, n_layers=L) \
            == L * (2 * e * K * K + e * K)
        diagonal = registration.flops_per_token(embed_dim=K, irrep_dims=(2, K - 2), expansion=e,
                                                covariance_contract="passthrough", n_layers=L,
                                                covariance_layout="diagonal")
        assert diagonal == L * (8.0 * e * K * K + 7.0 * K)
        full = registration.flops_per_token(embed_dim=K, irrep_dims=(2, K - 2), expansion=e,
                                            covariance_contract="posterior", n_layers=L, covariance_layout="full")
        # Corrected 2026-10-06 (fix stage): 'posterior' on the full layout also executes P = D^{-1/2} B^{-1} D^{1/2} C,
        # a Cholesky solve with K right-hand sides (2 K^3) plus two scalings and the symmetrization (4 K^2).
        assert full == L * (8.0 * e * K * K + K ** 3 / 3.0 + 6.0 * K ** 2 + (2.0 * K ** 3 + 4.0 * K ** 2))
    assert registration.report_label("passthrough") == "prototype_agents_mean_only_nonintertwiner"
    assert registration.report_label("posterior") == "prototype_agents_posterior_covariance_nonintertwiner"
    module = PrototypeAgentBlockMLP(4, 2, 0.0)
    assert sum(p.numel() for p in module.parameters()) \
        == registration.parameter_count(embed_dim=4, irrep_dims=(2, 2), expansion=2)
    for other in ("coordinate", "gauge_gate", "canonical_frame", "gaussian_template"):
        assert "posterior" not in get_block_mlp_registration(other).covariance_kinds
        assert get_block_mlp_registration(other).factory_options == frozenset()
    shared = dict(embed_dim=3, irrep_dims=(3,), expansion=2, activation="gelu", dropout=0.0,
                  covariance_contract="passthrough", covariance_floor=1e-4)
    built = build_block_mlp("prototype_agents", inference="exact", template_init_variance=3.0, **shared)
    assert type(built) is PrototypeAgentBlockMLP and built.inference == "exact"
    assert torch.equal(built.log_lambda, torch.full((6, 3), math.log(3.0)))
    torch.manual_seed(0)
    template = build_block_mlp("gaussian_template", inference="exact", template_init_variance=3.0, **shared)
    torch.manual_seed(0)
    direct = GaussianTemplateBlockMLP(3, 2, 0.0)
    assert type(template) is GaussianTemplateBlockMLP
    assert all(torch.equal(template.state_dict()[k], v) for k, v in direct.state_dict().items())
    with pytest.raises(ValueError):
        build_block_mlp("prototype_agents", **{**shared, "covariance_contract": "delta_full"})
    with pytest.raises(ValueError):
        build_block_mlp("coordinate", **{**shared, "covariance_contract": "posterior"})


def _cfg(**overrides):
    values = dict(attention_mode="projected_predictive", block_mlp_mode="prototype_agents",
                  block_mlp_expansion=2, n_layers=2, max_seq_len=4, attention_residual=True)
    values.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def test_config_defaults_validation_and_inactive_fields() -> None:
    # The test baseline pins these values; the VFE3Config defaults are the user's to choose and only have to be
    # admitted (2026-10-21: a changed default must not fail a test).
    defaults = {f.name: f.default for f in dataclasses.fields(VFE3Config)}
    baseline = config_values()
    assert baseline["block_mlp_inference"] == "mean_field"
    assert baseline["block_mlp_template_init_variance"] == 10.0
    assert baseline["block_mlp_mode"] == "coordinate" and baseline["block_mlp_covariance"] == "passthrough"
    assert defaults["block_mlp_inference"] in ("mean_field", "exact")
    assert defaults["block_mlp_template_init_variance"] > 0.0
    assert _cfg().block_mlp_inference == "mean_field"
    cfg = _cfg(block_mlp_inference="exact", block_mlp_covariance="posterior", block_mlp_template_init_variance=2.5)
    assert cfg.block_mlp_inference == "exact" and cfg.block_mlp_template_init_variance == 2.5
    for contract in ("passthrough", "posterior"):
        assert _cfg(block_coordinates="reference", block_mlp_covariance=contract).block_coordinates == "reference"
    with pytest.raises(ValueError, match="block_mlp_inference"):
        _cfg(block_mlp_inference="variational")
    for bad in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="block_mlp_template_init_variance"):
            _cfg(block_mlp_template_init_variance=bad)
    for mode in ("coordinate", "gaussian_template"):
        with pytest.raises(ValueError, match="block_mlp_covariance"):
            _cfg(block_mlp_mode=mode, block_mlp_covariance="posterior")
    with pytest.raises(ValueError, match="block_mlp_covariance"):
        _cfg(block_mlp_covariance="delta_full")
    knobs = {"block_mlp_inference", "block_mlp_template_init_variance"}
    assert knobs <= set(test_config().inactive_fields())
    assert knobs <= set(_cfg(block_mlp_mode="gaussian_template").inactive_fields())
    assert knobs <= set(_cfg(use_block_mlp=False).inactive_fields())
    active = set(_cfg().inactive_fields())
    assert not (knobs & active)
    assert {"block_mlp_activation", "block_mlp_covariance_floor"} <= active
    assert "block_mlp_expansion" not in active and "block_mlp_covariance" not in active


@pytest.mark.parametrize("coordinates,inference,contract,family", [
    ("local", "mean_field", "passthrough", "gaussian_diagonal"),
    ("local", "exact", "posterior", "gaussian_diagonal"),
    ("reference", "exact", "posterior", "gaussian_diagonal"),
    ("reference", "mean_field", "posterior", "gaussian_full"),
])
def test_configured_model_path(coordinates: str, inference: str, contract: str, family: str) -> None:
    from vfe3.train import build_optimizer

    extra = {"decode_mode": "full_chunked"} if family == "gaussian_full" else {}
    cfg = _cfg(block_coordinates=coordinates, block_mlp_inference=inference, block_mlp_covariance=contract,
               family=family, **extra)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    assert len(model.block_mlps) == cfg.n_layers
    M, K = cfg.block_mlp_expansion * cfg.embed_dim, cfg.embed_dim
    for mlp in model.block_mlps:
        assert type(mlp) is PrototypeAgentBlockMLP
        assert mlp.inference == inference and mlp.covariance_contract == contract
        assert mlp.nu.shape == (M, K) and torch.equal(mlp.log_lambda, torch.full((M, K), math.log(10.0)))
        assert torch.equal(mlp.log_pi, torch.zeros(M))
    keys = sorted(k for k in model.state_dict() if k.startswith("block_mlps."))
    assert keys == sorted(f"block_mlps.{layer}.{name}" for layer in range(cfg.n_layers)
                          for name in ("nu", "log_lambda", "log_pi"))
    optimizer = build_optimizer(model, cfg)
    groups = [g for g in optimizer.param_groups if g.get("lr_aux_role") == "block_mlp"]
    assert len(groups) == 1
    assert {id(p) for p in groups[0]["params"]} == {id(mlp.nu) for mlp in model.block_mlps}
    assert groups[0]["weight_decay"] == cfg.weight_decay
    # The template log-variances and log-weights train undecayed (decay would pull lambda to 1, pi to uniform).
    exempt = {id(p) for mlp in model.block_mlps for p in (mlp.log_lambda, mlp.log_pi)}
    holders = [g for g in optimizer.param_groups if {id(p) for p in g["params"]} & exempt]
    assert len(holders) == 1 and {id(p) for p in holders[0]["params"]} == exempt
    assert holders[0]["weight_decay"] == 0.0 and holders[0]["lr"] == groups[0]["lr"]
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=_gen(141))
    model.train()
    _, loss, _ = model(tokens, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith("block_mlps."):
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
            assert float(parameter.grad.abs().max()) > 0.0, name
