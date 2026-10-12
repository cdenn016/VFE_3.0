r"""Sphere-bound cap of the LayerNorm covariance pushforward (audit 2026-10-12, item 1).

The exact LayerNorm output :math:`\gamma\odot z + b` has :math:`\lVert z\rVert^2 < K`, so its covariance trace is at
most :math:`B = K\max_k\gamma_k^2`. The delta method exceeds that by orders of magnitude once the input variance is
large against :math:`s^2` (at the launcher's initialization, :math:`\Sigma/s^2 \approx 4\cdot10^3`), and the model's
covariance reached NaN by the third layer. The pushforward is now scaled by :math:`\min(1, B/\operatorname{tr})`.
Tiny CPU cases only.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
from vfe3.model import block_mlp as bm
from vfe3.model.model import VFEModel

F64 = torch.float64


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _launcher_like(K: int = 20, n: int = 6, sigma: float = 15.0, full: bool = False) -> tuple:
    g = torch.Generator().manual_seed(0)
    mu = 0.065 * torch.randn(n, K, generator=g, dtype=F64)                 # the launcher's mean scale
    cov = torch.full((n, K), sigma, dtype=F64)
    return mu, (torch.diag_embed(cov) if full else cov)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_pushforward_trace_is_capped_at_the_sphere_bound(full: bool, affine: bool) -> None:
    K = 20
    norm = AffineLayerNorm(K, eps=1e-5).double() if affine else LayerNorm(K, eps=1e-5)
    if affine:
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, K, dtype=F64))
    mu, sigma = _launcher_like(K, full=full)
    uncapped = bm._uncapped_block_norm_pushforward(norm, mu, sigma, 1e-4)
    capped = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-4)
    trace = (lambda P: P.sum(-1)) if not full else (lambda P: torch.diagonal(P, dim1=-2, dim2=-1).sum(-1))
    bound = K * (1.5 ** 2 if affine else 1.0)
    assert (trace(uncapped) > 1e3 * bound).all()                          # the delta method far outside its regime
    torch.testing.assert_close(trace(capped), torch.full_like(trace(capped), bound), rtol=1e-12, atol=0.0)
    lam = bound / trace(uncapped)                                         # (n,) one scale per token
    expected = uncapped * (lam[:, None, None] if full else lam[:, None])
    torch.testing.assert_close(capped, expected, rtol=1e-12, atol=0.0)
    if full:
        assert (torch.linalg.eigvalsh(capped) > 0).all()


@pytest.mark.parametrize("full", [False, True])
def test_pushforward_unchanged_where_the_bound_does_not_bind(full: bool) -> None:
    K = 20
    norm = LayerNorm(K, eps=1e-5)
    mu, sigma = _launcher_like(K, sigma=1e-6, full=full)                  # Sigma ~ 2e-3 s^2: the valid regime
    uncapped = bm._uncapped_block_norm_pushforward(norm, mu, sigma, 1e-4)
    assert torch.equal(bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-4), uncapped)


def test_cap_counts_capped_tokens() -> None:
    bm.reset_layernorm_trace_cap_elements()
    mu, sigma = _launcher_like(20, n=5)
    bm.block_norm_covariance_pushforward(LayerNorm(20, eps=1e-5), mu, sigma, 1e-4)
    assert bm.layernorm_trace_cap_elements() == 5
    bm.reset_layernorm_trace_cap_elements()
    assert bm.layernorm_trace_cap_elements() == 0


@pytest.mark.parametrize("mlp", [dict(use_block_mlp=False),
                                 dict(use_block_mlp=True, block_mlp_mode="coordinate",
                                      block_mlp_covariance="delta_full")])
@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_model_with_pushforward_stays_finite_at_large_sigma_init(mlp: dict, family: str) -> None:
    r"""At a large initial variance against the mean spread the uncapped pushforward went NaN within a few layers;
    the capped one keeps the loss, every gradient and every layer covariance finite, and the cap binds."""
    cfg = tiny_config(max_seq_len=4, n_layers=3, norm_type_block="layernorm", block_norm_covariance="pushforward",
                      sigma_init=15.0, mu_init_std=0.01, family=family, predictive_covariance="moment_matched",
                      **mlp)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(1))
    bm.reset_layernorm_trace_cap_elements()
    _, loss, _ = model(tokens, tokens)
    assert bm.layernorm_trace_cap_elements() > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
