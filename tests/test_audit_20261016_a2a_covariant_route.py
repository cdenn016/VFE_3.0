r"""The fully_gauge_covariant switch on the live launcher route with the LayerNorm covariance pushforward
(audit 2026-10-16, F30 and F2).

:func:`vfe3.gauge_covariant.fully_gauge_covariant_overrides` puts the launcher's route on the gauge facet
(``gaussian_full``, reference block coordinates) and leaves ``block_norm_covariance`` as configured. No test built
that switched route, and the tests that ran ``gaussian_full`` + reference + ``'pushforward'`` asserted finiteness
only, which a covariance with a negative float64 eigenvalue and condition number :math:`10^7` satisfies. Here the
switched route is built at the launcher's initialization scale (``mu_init_std=0.03``, ``sigma_init=5``, where the
sphere-bound cap binds), stepped once, and its post-norm covariances are checked for positive definiteness and
conditioning: ``'pushforward_moment'`` passes; the capped ``'pushforward'``, kept for saved runs, is the documented
failure (strict xfail). Tiny CPU cases only.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from tests.profiles import live_values
from vfe3.config import VFE3Config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.geometry.norms import AffineLayerNorm
from vfe3.model import block_mlp as bm
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

F64 = torch.float64
MODE = "pushforward_moment"
CONDITION_BOUND = 1e6


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _switched_values(**overrides) -> dict:
    r"""The live route at tiny scale and the launcher's initialization, with the switch's overrides applied."""
    values = live_values(norm_type_block="layernorm", layernorm_affine=True, mu_init_std=0.03, sigma_init=5.0,
                         block_norm_covariance="pushforward", **overrides)
    return {**values, **fully_gauge_covariant_overrides(values)}


def _config(values: dict, mode: str) -> VFE3Config:
    r"""The config at ``block_norm_covariance=mode``; the validator admits ``'pushforward_moment'`` under the
    admissibility conditions of ``'pushforward'``."""
    return VFE3Config(**{**values, "block_norm_covariance": mode})


def _model(cfg: VFE3Config) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def _batch(cfg: VFE3Config) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    return (torch.randint(0, cfg.vocab_size, shape, generator=g),
            torch.randint(0, cfg.vocab_size, shape, generator=g))


def _post_norm_covariances(model: VFEModel, tokens: torch.Tensor) -> list:
    capture = {"diagnostic": {}}
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.forward_beliefs(tokens, capture=capture)
    return [belief.sigma for belief in capture["diagnostic"]["layer_block_mlp_inputs"]]


@pytest.mark.parametrize("mode", ["pushforward", MODE])
def test_switched_live_route_is_on_the_gauge_facet_and_steps(mode: str) -> None:
    cfg = _config(_switched_values(), mode)
    assert cfg.family in ("gaussian_full", "gaussian_block") and cfg.block_coordinates == "reference"
    assert gauge_covariance_failures(cfg) == []
    model = _model(cfg)
    tokens, targets = _batch(cfg)
    bm.reset_layernorm_trace_cap_elements()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss, _ = model(tokens, targets)
    capped = bm.layernorm_trace_cap_elements()
    if mode == "pushforward":
        assert capped > 0                                                   # the launcher's init is the capped regime
    else:
        assert capped == 0                                                  # the moment estimate is within the bound
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    optimizer = build_optimizer(model, cfg)
    scored = torch.tensor(targets.numel(), dtype=torch.int64)
    for name in ("phi_statistics_provider", "table_statistics_provider"):   # the Fisher M-steps, as train_step
        provider = getattr(optimizer, name, None)
        if provider is not None:
            provider.set_batch(tokens, scored)
    optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss_after, _ = model(tokens, targets)
    assert torch.isfinite(loss_after)


@pytest.mark.parametrize("n_layers", [2, 4])
@pytest.mark.parametrize("mode", [
    pytest.param("pushforward", marks=pytest.mark.xfail(
        strict=True, reason="audit 2026-10-16 F2: the capped delta method is near rank one at the launcher's "
                            "initialization (condition number about 1e7); kept unchanged for saved runs")),
    MODE])
def test_switched_live_route_post_norm_covariances_are_well_conditioned(mode: str, n_layers: int) -> None:
    r"""Every post-norm covariance of the switched route is positive definite in float64, factors in float32 and has
    condition number below :math:`10^6`, at every depth. It is in post-norm units (trace of order :math:`K`, against
    :math:`5K` passed through at the first layer), so the pushforward did act."""
    cfg = _config(_switched_values(n_layers=n_layers), mode)
    model = _model(cfg)
    tokens, _ = _batch(cfg)
    for layer, sigma in enumerate(_post_norm_covariances(model, tokens)):
        trace = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1)             # (B, N)
        assert float(trace.max()) < 2.0 * cfg.embed_dim, layer
        eig = torch.linalg.eigvalsh(sigma.double())                         # (B, N, K) ascending
        assert float(eig[..., 0].min()) > 0.0, layer
        assert float((eig[..., -1] / eig[..., 0]).max()) < CONDITION_BOUND, layer
        assert bool((torch.linalg.cholesky_ex(sigma).info == 0).all()), layer


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
@pytest.mark.parametrize("mode", ["pushforward", MODE])
def test_local_live_route_carries_the_selected_pushforward(mode: str, family: str) -> None:
    r"""The un-switched live route (local block coordinates): each layer's post-norm covariance is the selected
    pushforward of its converged pre-norm belief, so ``block_norm_covariance`` reaches the block as configured."""
    values = live_values(norm_type_block="layernorm", layernorm_affine=True, mu_init_std=0.03, sigma_init=5.0,
                         block_norm_covariance="pushforward", block_coordinates="local", family=family,
                         use_cg_coupling=False)
    cfg = _config(values, mode)
    model = _model(cfg)
    tokens, _ = _batch(cfg)
    capture = {"diagnostic": {}}
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.forward_beliefs(tokens, capture=capture)
        diagnostic = capture["diagnostic"]
        for converged, pushed in zip(diagnostic["layer_converged"], diagnostic["layer_block_mlp_inputs"]):
            expected = bm.block_norm_covariance_pushforward(model.block_norm, converged.mu, converged.sigma,
                                                            cfg.block_mlp_covariance_floor, mode=mode)
            torch.testing.assert_close(pushed.sigma, expected, rtol=0.0, atol=0.0)
    assert len(diagnostic["layer_converged"]) == cfg.n_layers


def test_moment_pushforward_conditioning_does_not_compound_through_a_chain_of_norms() -> None:
    r"""The LayerNorm output has no variance along :math:`\Gamma^{-1}\mathbf 1`, so a floor relative to the input
    covariance alone fills that direction with about :math:`f^\ell` of the scale after :math:`\ell` norms. Pushing a
    covariance through four LayerNorm seams in turn (fresh means, float64) keeps the condition number below
    :math:`10^6` at every seam."""
    K = 8
    norm = AffineLayerNorm(K, eps=1e-6).double()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.8, 1.2, K, dtype=F64))
    g = torch.Generator().manual_seed(6)
    Sigma = torch.diag_embed(torch.full((5, K), 5.0, dtype=F64))
    for seam in range(4):
        mu = (0.03 if seam == 0 else 1.0) * torch.randn(5, K, generator=g, dtype=F64)
        with torch.no_grad():
            Sigma = bm.block_norm_covariance_pushforward(norm, mu, Sigma, 1e-4, mode=MODE)
        eig = torch.linalg.eigvalsh(Sigma)
        assert float(eig[..., 0].min()) > 0.0, seam
        assert float((eig[..., -1] / eig[..., 0]).max()) < CONDITION_BOUND, seam
