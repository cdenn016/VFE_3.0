r"""``block_mlp_covariance='delta_full'`` composed with the block LayerNorm Jacobian (audit 2026-10-09, variational).

In local coordinates the block transform is :math:`T(u) = \mathrm{MLP}(\mathrm{LN}_{\gamma,b}(u))`, so the
delta-method pushforward is :math:`J\Sigma J^\top + f\Sigma` with :math:`J = J_{\mathrm{MLP}}J_{\mathrm{LN}}` and

.. math::
    J_{\mathrm{LN}} = \operatorname{diag}(\gamma)\,\frac{1}{s}\Bigl(I - \tfrac{1}{K}\mathbf 1\mathbf 1^\top
        - \tfrac{1}{K}zz^\top\Bigr),\qquad s = \sqrt{\operatorname{Var}(u) + \epsilon},\quad z = (u - \bar u)/s.

Checks, float64 and tiny K: the closed-form Jacobian against central finite differences of the live norm, the
composite pushforward against finite differences of the composite map (coordinate, canonical-frame and gauge-gate
MLPs), the executed block inside a model forward, and the relaxed configuration guard.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from tests.profiles import test_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.norms import AffineLayerNorm, LayerNorm, MahalanobisNorm
from vfe3.model.block_mlp import (
    BlockMLP, CanonicalFrameBlockMLP, GaugeGateBlockMLP, block_norm_jacobian, block_norm_linearization,
    block_norm_radial_term, layernorm_jacobian,
)
from vfe3.model.model import VFEModel

K = 4
EPS = 1e-6
FLOOR = 1e-3


@pytest.fixture(autouse=True)
def _double() -> None:
    torch.set_num_threads(1)


def _fd_jacobian(fn, u: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
    r"""Central finite-difference Jacobian of ``fn`` at the batch ``u`` (..., K) -> (..., K, K)."""
    columns = []
    for k in range(u.shape[-1]):
        e = torch.zeros_like(u)
        e[..., k] = h
        columns.append((fn(u + e) - fn(u - e)) / (2.0 * h))
    return torch.stack(columns, dim=-1)


def _spd(batch: tuple, generator: torch.Generator) -> torch.Tensor:
    A = torch.randn(*batch, K, K, generator=generator, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=torch.float64)


def _post_norm_floor(norm, u: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""The floor :math:`fD\Sigma D`, :math:`D = \operatorname{diag}(\gamma)/s` (audit 2026-10-09 round 2, #13), written out."""
    s = ((u - u.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + norm.eps).sqrt()   # (..., 1)
    d = (getattr(norm, "weight", None) if getattr(norm, "weight", None) is not None
         else torch.ones(u.shape[-1], dtype=u.dtype)) / s                                    # (..., K)
    return FLOOR * d.unsqueeze(-1) * sigma * d.unsqueeze(-2)


def _affine_norm(generator: torch.Generator) -> AffineLayerNorm:
    norm = AffineLayerNorm(K, eps=EPS).double()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.5 * torch.randn(K, generator=generator, dtype=torch.float64))
        norm.bias.copy_(0.3 * torch.randn(K, generator=generator, dtype=torch.float64))
    return norm


@pytest.mark.parametrize("affine", [False, True])
def test_layernorm_jacobian_matches_finite_differences(affine: bool) -> None:
    g = torch.Generator().manual_seed(0)
    u = 2.0 * torch.randn(3, 5, K, generator=g, dtype=torch.float64)
    norm = _affine_norm(g) if affine else LayerNorm(K, eps=EPS)
    sigma = torch.zeros(3, 5, K, K, dtype=torch.float64)
    J = block_norm_jacobian(norm, u)
    J_fd = _fd_jacobian(lambda x: norm(x, sigma), u)
    assert J.shape == (3, 5, K, K)
    assert (J - J_fd).abs().max() < 1e-8
    # The closed form, written out once more independently of the helper.
    centered = u - u.mean(-1, keepdim=True)
    s = (centered.square().mean(-1, keepdim=True) + EPS).sqrt()
    z = centered / s
    manual = (torch.eye(K, dtype=torch.float64) - 1.0 / K - z.unsqueeze(-1) * z.unsqueeze(-2) / K) / s.unsqueeze(-1)
    if affine:
        manual = norm.weight.unsqueeze(-1) * manual
    assert (layernorm_jacobian(u, eps=EPS, weight=norm.weight if affine else None) - manual).abs().max() < 1e-14


def test_block_norm_jacobian_refuses_the_mahalanobis_norm() -> None:
    with pytest.raises(ValueError, match="LayerNorm"):
        block_norm_jacobian(MahalanobisNorm(K, family="gaussian_full"), torch.zeros(2, K, dtype=torch.float64))


def _composite_case(mode: str, generator: torch.Generator):
    if mode == "coordinate":
        mlp = BlockMLP(K, 3, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=FLOOR)
        kwargs: dict = {}
    elif mode == "canonical_frame":
        mlp = CanonicalFrameBlockMLP(K, 3, "silu", 0.0, covariance_contract="delta_full", covariance_floor=FLOOR)
        U = torch.eye(K, dtype=torch.float64) + 0.3 * torch.randn(2, 3, K, K, generator=generator, dtype=torch.float64)
        kwargs = {"frame_context": CanonicalFrameContext(U, torch.linalg.inv(U))}
    else:
        mlp = GaugeGateBlockMLP([2, 2], 3, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=FLOOR)
        kwargs = {}
    mlp = mlp.double().eval()
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=generator, dtype=torch.float64))
    return mlp, kwargs


@pytest.mark.parametrize("mode", ["coordinate", "canonical_frame", "gauge_gate"])
def test_composite_pushforward_matches_finite_differences(mode: str) -> None:
    r""":math:`J_{\mathrm{MLP}}J_{\mathrm{LN}}\Sigma(\cdot)^\top + f\Sigma` against the FD Jacobian of MLP∘LN."""
    g = torch.Generator().manual_seed(1)
    norm = _affine_norm(g)
    mlp, kwargs = _composite_case(mode, g)
    u = 1.5 * torch.randn(2, 3, K, generator=g, dtype=torch.float64)
    sigma = _spd((2, 3), g)
    J_ln, scale = block_norm_linearization(norm, u)
    assert torch.equal(J_ln, block_norm_jacobian(norm, u))
    result = mlp.forward_moments(norm(u, sigma), sigma, input_jacobian=J_ln, input_scale=scale, **kwargs)

    def composite(x: torch.Tensor) -> torch.Tensor:
        return mlp.forward_moments(norm(x, sigma), sigma, **kwargs).mu

    J_fd = _fd_jacobian(composite, u)
    expected = J_fd @ sigma @ J_fd.transpose(-1, -2) + _post_norm_floor(norm, u, sigma)
    assert (result.jacobian - J_fd).abs().max() < 1e-7
    assert (result.sigma - expected).abs().max() / expected.abs().max() < 1e-8
    # Omitting the norm Jacobian (the pre-fix behavior) is a different covariance on this input.
    stale = mlp.forward_moments(norm(u, sigma), sigma, **kwargs).sigma
    assert (stale - expected).abs().max() / expected.abs().max() > 1e-2


def test_passthrough_contract_refuses_an_input_jacobian() -> None:
    mlp = BlockMLP(K, 2, "gelu", 0.0).double()
    u = torch.zeros(1, K, dtype=torch.float64)
    with pytest.raises(ValueError, match="delta_full"):
        mlp.forward_moments(u, torch.ones(1, K, dtype=torch.float64), input_jacobian=torch.eye(K, dtype=torch.float64),
                            input_scale=torch.ones(1, K, dtype=torch.float64))
    # The Jacobian and its scale come together: a norm Jacobian without the post-norm floor scale is refused.
    full = BlockMLP(K, 2, "gelu", 0.0, covariance_contract="delta_full").double()
    with pytest.raises(ValueError, match="input_scale"):
        full.forward_moments(u, torch.eye(K, dtype=torch.float64).expand(1, K, K),
                             input_jacobian=torch.eye(K, dtype=torch.float64).expand(1, K, K))


def _model_cfg(**over: object):
    values = dict(vocab_size=11, embed_dim=K, n_heads=2, max_seq_len=6, n_layers=1,
                  family="gaussian_full", decode_mode="full_chunked", attention_mode="projected_predictive",
                  predictive_covariance="moment_matched", attention_residual=True,
                  norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                  block_mlp_mode="coordinate", block_mlp_covariance="delta_full",
                  block_mlp_covariance_floor=FLOOR, block_mlp_expansion=2, skip_belief_sigma_update=False)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def test_executed_block_applies_the_composite_pushforward() -> None:
    r"""Inside the model, the block's output covariance is the composite delta method of its input."""
    cfg = _model_cfg()
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double().eval()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        model.block_norm.weight.copy_(1.0 + 0.4 * torch.randn(K, generator=g, dtype=torch.float64))
        model.block_norm.bias.copy_(0.2 * torch.randn(K, generator=g, dtype=torch.float64))
        model.prior_bank.phi_embed.normal_(0.0, 0.3, generator=g)
    tokens = torch.tensor([[1, 4, 2, 7, 3]])
    capture: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    converged = capture["converged"]                       # pre-norm belief after attention
    mu, sigma = converged.mu, converged.sigma              # (1, N, K), (1, N, K, K)
    mlp = model.block_mlps[0]

    def composite(x: torch.Tensor) -> torch.Tensor:
        return mlp(model.block_norm(x, sigma))

    J_fd = _fd_jacobian(composite, mu)
    expected = J_fd @ sigma @ J_fd.transpose(-1, -2) + _post_norm_floor(model.block_norm, mu, sigma)
    # The norm's radial second-order term rho (J_mlp g)(J_mlp g)^T (audit 2026-10-10 round 2, item 11).
    radial = block_norm_radial_term(model.block_norm, mu, sigma)
    h = (_fd_jacobian(mlp, model.block_norm(mu, sigma)) @ radial.direction.unsqueeze(-1)).squeeze(-1)   # (1, N, K)
    expected = expected + radial.rho.unsqueeze(-1) * h.unsqueeze(-1) * h.unsqueeze(-2)
    # Scaled by the norm's sphere-bound cap (audit 2026-10-12, item 1); the composite is linear in the norm stage.
    from vfe3.model.block_mlp import block_norm_trace_cap
    expected = expected * block_norm_trace_cap(model.block_norm, mu, sigma, cfg.block_mlp_covariance_floor).unsqueeze(-1)
    got = capture["block_mlp_output"].sigma
    assert (got - expected).abs().max() / expected.abs().max() < 1e-8


def test_config_admits_layernorm_and_refuses_mahalanobis() -> None:
    for affine in (False, True):
        cfg = _model_cfg(layernorm_affine=affine)
        assert cfg.block_mlp_covariance == "delta_full" and cfg.norm_type_block == "layernorm"
    with pytest.raises(ValueError, match="mahalanobis"):
        _model_cfg(norm_type_block="mahalanobis")
