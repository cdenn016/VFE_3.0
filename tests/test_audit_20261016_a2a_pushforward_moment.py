r"""Moment estimate of the LayerNorm covariance pushforward, ``block_norm_covariance='pushforward_moment'``
(audit 2026-10-16, F3 and F2).

The delta method of ``'pushforward'`` is valid while the input variance is small against :math:`s^2`. At the
launcher's initialization (:math:`\Sigma/s^2 \approx 5\cdot10^3`) its radial term dominates and the sphere-bound cap
keeps that shape: the capped diagonal is proportional to :math:`\gamma_k^2z_k^2` (uncorrelated with the Monte Carlo
covariance), and the capped full covariance is near rank one (condition number :math:`10^7`). ``'pushforward_moment'``
takes the delta method at the effective scale :math:`s_{\mathrm{eff}}^2 = s^2 + \operatorname{tr}(\Pi\Sigma\Pi)/K`,
the expected LayerNorm variance of the input, which is exact in the small-noise limit, gives the uniform-sphere
covariance :math:`\tfrac{K}{K-1}\Gamma\Pi\Gamma` in the large-noise limit, and tracks the Monte Carlo covariance in
between. ``'pushforward'`` is kept unchanged for saved runs. Tiny CPU cases in float64 (one float32 certificate).
"""
from __future__ import annotations

import math

import pytest
import torch

from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
from vfe3.model import block_mlp as bm

F64 = torch.float64
MODE = "pushforward_moment"
EPS = 1e-6
FLOOR = 1e-4


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _gamma(K: int, seed: int = 1) -> torch.Tensor:
    return 0.7 + 0.6 * torch.rand(K, generator=torch.Generator().manual_seed(seed), dtype=F64)   # non-uniform


def _affine(gamma: torch.Tensor) -> AffineLayerNorm:
    norm = AffineLayerNorm(gamma.shape[0], eps=EPS).double()
    with torch.no_grad():
        norm.weight.copy_(gamma)
    return norm


def _inputs(K: int, ratio: float, anisotropy: float, full: bool, seed: int = 2) -> tuple:
    r"""Pre-norm mean at the launcher's spread (std 0.03) and a covariance at ``ratio`` times :math:`s^2`."""
    g = torch.Generator().manual_seed(seed)
    mu = 0.03 * torch.randn(K, generator=g, dtype=F64)
    s2 = float((mu - mu.mean()).square().mean()) + EPS
    spectrum = torch.exp(anisotropy * torch.randn(K, generator=g, dtype=F64))
    spectrum = ratio * s2 * spectrum / spectrum.mean()                      # (K,) mean eigenvalue ratio * s^2
    if not full:
        return mu, spectrum
    Q, _ = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=F64))
    Sigma = (Q * spectrum) @ Q.T
    return mu, 0.5 * (Sigma + Sigma.T)


def _monte_carlo(mu: torch.Tensor, Sigma: torch.Tensor, gamma: torch.Tensor, n: int, seed: int = 3) -> torch.Tensor:
    r"""Sample covariance (K, K) of :math:`\gamma\odot\mathrm{LN}(u)`, :math:`u \sim \mathcal N(\mu, \Sigma)`."""
    K = mu.shape[-1]
    L = torch.linalg.cholesky(Sigma if Sigma.dim() == 2 else torch.diag(Sigma))
    u = mu + torch.randn(n, K, generator=torch.Generator().manual_seed(seed), dtype=F64) @ L.T
    return torch.cov((torch.nn.functional.layer_norm(u, (K,), eps=EPS) * gamma).T)


def _sphere_bound(gamma: torch.Tensor) -> float:
    return gamma.shape[0] * float(gamma.square().max())


@pytest.mark.parametrize("anisotropy", [0.0, 0.5])
@pytest.mark.parametrize("ratio", [1e3, 160.0, 30.0, 1.5, 0.1])
def test_diagonal_moment_pushforward_matches_monte_carlo_per_coordinate(ratio: float, anisotropy: float) -> None:
    r"""Per-coordinate agreement with the Monte Carlo variances (worst :math:`\lvert\log\rvert` ratio and correlation)
    across the capped, mid and valid regimes, with an anisotropic input and a non-uniform gamma. The capped
    ``'pushforward'`` fails the same check where its cap binds, and keeps only the sphere bound."""
    K = 32
    gamma = _gamma(K)
    norm = _affine(gamma)
    mu, sigma = _inputs(K, ratio, anisotropy, full=False)
    truth = torch.diagonal(_monte_carlo(mu, sigma, gamma, n=40000))         # (K,)
    with torch.no_grad():
        moment = bm.block_norm_covariance_pushforward(norm, mu[None], sigma[None], FLOOR, mode=MODE)[0]
        legacy = bm.block_norm_covariance_pushforward(norm, mu[None], sigma[None], FLOOR)[0]
    worst = float((moment / truth).log().abs().max())
    corr = float(torch.corrcoef(torch.stack([moment, truth]))[0, 1])
    assert worst < 0.2, worst
    assert corr > 0.97, corr
    assert float(legacy.sum()) <= _sphere_bound(gamma) * (1 + 1e-12)
    if ratio >= 30.0:                                                       # the cap binds: the legacy shape is wrong
        assert float((legacy / truth).log().abs().max()) > 0.9


@pytest.mark.parametrize("ratio", [1e3, 30.0, 1.5, 0.1])
def test_full_moment_pushforward_matches_monte_carlo(ratio: float) -> None:
    r"""The full layout against the Monte Carlo covariance for a rotated anisotropic :math:`\Sigma`: relative
    Frobenius error, and a spectrum floored at about :math:`f` of the pushed scale, where the capped delta method is
    near rank one (relative error above 1 and condition number above :math:`10^5` where the cap binds)."""
    K = 16
    gamma = _gamma(K)
    norm = _affine(gamma)
    mu, Sigma = _inputs(K, ratio, 0.5, full=True)
    truth = _monte_carlo(mu, Sigma, gamma, n=80000)
    with torch.no_grad():
        moment = bm.block_norm_covariance_pushforward(norm, mu[None], Sigma[None], FLOOR, mode=MODE)[0]
        legacy = bm.block_norm_covariance_pushforward(norm, mu[None], Sigma[None], FLOOR)[0]
    error = float((moment - truth).norm() / truth.norm())
    assert error < 0.12, error
    eig = torch.linalg.eigvalsh(moment)
    assert float(eig[0]) > 0.0
    assert float(eig[-1] / eig[0]) < 1e6
    if ratio >= 30.0:
        assert float((legacy - truth).norm() / truth.norm()) > 1.0
        legacy_eig = torch.linalg.eigvalsh(legacy)
        assert float(legacy_eig[-1] / legacy_eig[0]) > 1e5


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_moment_pushforward_is_the_delta_method_in_the_small_noise_limit(full: bool, affine: bool) -> None:
    r"""As :math:`\Sigma/s^2 \to 0` the effective scale is :math:`s` and the estimate is the delta method, up to its
    span floor :math:`f(T/K)DWW^\top D`, of order :math:`f` of the pushed scale."""
    K = 8
    gamma = _gamma(K) if affine else torch.ones(K, dtype=F64)
    norm = _affine(gamma) if affine else LayerNorm(K, eps=EPS)
    mu, sigma = _inputs(K, 1e-7, 0.5, full=full)
    with torch.no_grad():
        moment = bm.block_norm_covariance_pushforward(norm, mu[None], sigma[None], FLOOR, mode=MODE)
        legacy = bm.block_norm_covariance_pushforward(norm, mu[None], sigma[None], FLOOR)
    scale = float((legacy if not full else torch.diagonal(legacy, dim1=-2, dim2=-1)).mean())
    torch.testing.assert_close(moment, legacy, rtol=1e-5, atol=2.0 * FLOOR * scale)


@pytest.mark.parametrize("K", [4, 8])
def test_moment_pushforward_is_the_sphere_covariance_in_the_large_noise_limit(K: int) -> None:
    r"""For :math:`\Sigma = \sigma I` with :math:`\sigma/s^2 \to \infty` the LayerNorm output is uniform on the centered
    sphere of radius :math:`\sqrt K`, with covariance :math:`\tfrac{K}{K-1}\Gamma\Pi\Gamma`; the estimate is that plus
    its floors :math:`\tfrac{K}{K-1}f\Gamma^2` and :math:`f\Gamma\mathbf 1\mathbf 1^\top\Gamma/K` (the span floor on the
    direction the norm annihilates), exactly up to :math:`O(s^2/\sigma)`."""
    gamma = _gamma(K)
    norm = _affine(gamma)
    mu, sigma = _inputs(K, 1e10, 0.0, full=True)
    eye = torch.eye(K, dtype=F64)
    Pi = eye - 1.0 / K
    expected = gamma[:, None] * (K / (K - 1) * (Pi + FLOOR * eye) + FLOOR / K) * gamma[None, :]
    with torch.no_grad():
        full = bm.block_norm_covariance_pushforward(norm, mu[None], sigma[None], FLOOR, mode=MODE)[0]
        diagonal = bm.block_norm_covariance_pushforward(norm, mu[None], torch.diagonal(sigma)[None], FLOOR,
                                                        mode=MODE)[0]
    torch.testing.assert_close(full, expected, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(diagonal, torch.diagonal(expected), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("ratio", [1e-3, 3.0, 1e3])
def test_diagonal_layout_is_the_diagonal_of_the_full_layout(ratio: float) -> None:
    K = 8
    gamma = _gamma(K)
    mu, sigma = _inputs(K, ratio, 0.5, full=False)
    with torch.no_grad():
        diagonal = bm.layernorm_diagonal_pushforward(mu, sigma, eps=EPS, covariance_floor=FLOOR, weight=gamma,
                                                     effective_scale=True)
        full = bm.layernorm_full_pushforward(mu, torch.diag(sigma), eps=EPS, covariance_floor=FLOOR, weight=gamma,
                                             effective_scale=True)
    torch.testing.assert_close(diagonal, torch.diagonal(full), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("ratio", [3.0, 1e3])
def test_moment_pushforward_gradients_match_finite_differences(full: bool, ratio: float) -> None:
    r"""Float64 central differences in :math:`\mu`, :math:`\Sigma` (through a factor, so it stays SPD) and
    :math:`\gamma` at K = 4, in the mid and the large-noise regime."""
    K = 4
    gamma0 = _gamma(K)
    mu0, Sigma0 = _inputs(K, ratio, 0.5, full=True)
    factor0 = torch.linalg.cholesky(Sigma0)

    def pushed(mu: torch.Tensor, factor: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        Sigma = factor @ factor.T
        if full:
            return bm.layernorm_full_pushforward(mu, Sigma, eps=EPS, covariance_floor=FLOOR, weight=gamma,
                                                 effective_scale=True)
        return bm.layernorm_diagonal_pushforward(mu, torch.diagonal(Sigma), eps=EPS, covariance_floor=FLOOR,
                                                 weight=gamma, effective_scale=True)

    inputs = tuple(t.clone().requires_grad_(True) for t in (mu0, factor0, gamma0))
    assert torch.autograd.gradcheck(pushed, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("uniform_gamma", [False, True])
def test_moment_pushforward_keeps_the_sphere_bound_without_capping_at_the_launcher_initialization(
        uniform_gamma: bool) -> None:
    r"""At the launcher's initialization the moment estimate net of its floor terms is within the sphere bound on its
    own: the trace cap (kept as a guard, read net of the floor) does not bind, so its counter stays at zero, while it
    binds on every legacy token. With :math:`\gamma = 1` the floor itself can take the trace above :math:`B = K`
    (to :math:`K + fK^2/(K-1) + f` in the isotropic limit), which the guard allows."""
    K = 8
    gamma = torch.ones(K, dtype=F64) if uniform_gamma else _gamma(K)
    norm = _affine(gamma)
    g = torch.Generator().manual_seed(4)
    mu = 0.03 * torch.randn(6, K, generator=g, dtype=F64)
    sigma = torch.full((6, K), 5.0, dtype=F64)
    for layout in (sigma, torch.diag_embed(sigma)):
        bm.reset_layernorm_trace_cap_elements()
        with torch.no_grad():
            moment = bm.block_norm_covariance_pushforward(norm, mu, layout, FLOOR, mode=MODE)
            floor = bm._layernorm_moment_floor_trace(mu, layout, eps=EPS, covariance_floor=FLOOR, weight=gamma)
        assert bm.layernorm_trace_cap_elements() == 0
        trace = moment.sum(-1) if moment.dim() == 2 else torch.diagonal(moment, dim1=-2, dim2=-1).sum(-1)
        assert bool((trace - floor[..., 0] <= _sphere_bound(gamma)).all())
        bm.reset_layernorm_trace_cap_elements()
        with torch.no_grad():
            bm.block_norm_covariance_pushforward(norm, mu, layout, FLOOR)
        assert bm.layernorm_trace_cap_elements() == 6
    bm.reset_layernorm_trace_cap_elements()


@pytest.mark.parametrize("head_dims", [None, (4, 4)])
def test_float32_moment_pushforward_is_positive_definite_at_the_launcher_initialization(head_dims) -> None:
    r"""F2: the float32 full pushforward at the launcher's initialization. The capped delta method has condition
    number about :math:`10^7` there, so a float32 Cholesky downstream can fail; the moment estimate is floored at
    about :math:`f` of its scale, so every returned float32 matrix (or head block) factors."""
    K = 8
    gamma = _gamma(K).float()
    norm = AffineLayerNorm(K, eps=EPS)
    with torch.no_grad():
        norm.weight.copy_(gamma)
    g = torch.Generator().manual_seed(5)
    mu = 0.03 * torch.randn(64, K, generator=g)
    Sigma = torch.diag_embed(torch.full((64, K), 5.0))
    with torch.no_grad():
        moment = bm.block_norm_covariance_pushforward(norm, mu, Sigma, FLOOR, head_dims, mode=MODE)
        legacy = bm.block_norm_covariance_pushforward(norm, mu, Sigma, FLOOR, head_dims)
    assert moment.dtype == torch.float32
    assert bool((torch.linalg.cholesky_ex(moment).info == 0).all())
    eig = torch.linalg.eigvalsh(moment.double())
    assert float((eig[..., -1] / eig[..., 0]).max()) < 1e5
    if head_dims is None:                                                   # the whole near-rank-one matrix
        legacy_eig = torch.linalg.eigvalsh(legacy.double())
        assert float((legacy_eig[..., -1] / legacy_eig[..., 0]).median()) > 1e6


def test_unknown_pushforward_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="pushforward_moment"):
        bm.block_norm_covariance_pushforward(LayerNorm(4, eps=EPS), torch.zeros(1, 4), torch.ones(1, 4), FLOOR,
                                             mode="moment")
