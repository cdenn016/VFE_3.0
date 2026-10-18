r"""Audit 2026-10-17, remediation lane L2 (covariance path): items 1, 3/18, 7, 8, 12, 13, 19, 32, 33, 36, 41, 62, 63.

Tiny CPU cases only: isolated operator probes up to K = 32 and tiny models (K <= 8, two layers, two heads, vocab <= 9).
Each test names the audit item it pins.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Tuple

import pytest
import torch

from tests.profiles import make_relational_operators, tiny_config
from tests.test_blockwise_moment_matched_20261004 import _case
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry import retraction as rt
from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
from vfe3 import metrics
from vfe3.model import block_mlp as bm
from vfe3.model import covariance_ceiling as cc
from vfe3.model.model import VFEModel
from vfe3.numerics import safe_cholesky

F64 = torch.float64


@pytest.fixture(autouse=True)
def _two_threads() -> None:
    torch.set_num_threads(2)


def _quiet_config(**values: Any) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def _dense(blocks: torch.Tensor) -> torch.Tensor:   # (..., N, H, d, d) -> (..., N, K, K)
    H, d = blocks.shape[-3], blocks.shape[-1]
    eye = torch.eye(H, dtype=blocks.dtype)
    return torch.einsum("...hab,hg->...hagb", blocks, eye).reshape(*blocks.shape[:-3], H * d, H * d)


def _conditioned_frames(N: int, H: int, d: int, kappa: float, generator: torch.Generator,
                        batch: Tuple[int, ...] = (1,)) -> CanonicalFrameContext:
    r"""Block frames U = Q_1 diag(s) Q_2^T with singular values log-spaced in [1, kappa] (condition kappa)."""
    Q1, _ = torch.linalg.qr(torch.randn(*batch, N, H, d, d, generator=generator, dtype=F64))
    Q2, _ = torch.linalg.qr(torch.randn(*batch, N, H, d, d, generator=generator, dtype=F64))
    s = torch.exp(torch.linspace(0.0, math.log(kappa), d, dtype=F64))
    U = Q1 @ torch.diag_embed(s.expand(*batch, N, H, d)) @ Q2.mT
    return CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=True)


# Item 1: predictive_covariance_floor_mode = "max".

def test_max_floor_is_the_larger_floor_and_bounds_the_output_below() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 3, False, (2,), F64, seed=4)
    f = ops.predictive_covariance_floor
    anchor = torch.full_like(sigma, 0.2)
    anchor[..., ::2] = 50.0                                   # the prior floor wins on some coordinates only
    ops.predictive_covariance_floor_mode = "relative"
    relative = ops.floor_term(anchor, sigma, frame, diagonal=True)
    ops.predictive_covariance_floor_mode = "max"
    floor = ops.floor_term(anchor, sigma, frame, diagonal=True)
    prior = f * anchor
    assert bool((relative > prior).any()) and bool((relative < prior).any())
    torch.testing.assert_close(floor, torch.maximum(relative, prior), rtol=0, atol=0)
    out = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
    kernel = ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame, receive)
    torch.testing.assert_close(out, kernel + floor, rtol=1e-12, atol=1e-14)
    assert bool((out >= prior * (1 - 1e-12)).all())                             # Sigma^+ >= f Sigma_p
    larger = relative > prior
    torch.testing.assert_close((out - kernel)[larger], relative[larger], rtol=1e-9, atol=1e-12)


def test_max_floor_full_layout_dominates_both_floors() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 3, False, (), F64, seed=6)
    g = torch.Generator().manual_seed(7)
    A = torch.randn(*sigma.shape, sigma.shape[-1], generator=g, dtype=F64)
    full = A @ A.mT + torch.diag_embed(sigma)
    B = torch.randn(*sigma.shape, sigma.shape[-1], generator=g, dtype=F64)
    anchor = 30.0 * (B @ B.mT / sigma.shape[-1] + torch.eye(sigma.shape[-1], dtype=F64))
    ops.predictive_covariance_floor_mode = "relative"
    relative = ops.floor_term(anchor, full, frame, diagonal=False)
    ops.predictive_covariance_floor_mode = "max"
    floor = ops.floor_term(anchor, full, frame, diagonal=False)
    prior = ops.predictive_covariance_floor * anchor
    scale = floor.diagonal(dim1=-2, dim2=-1).abs().amax()
    for lower in (prior, relative):
        assert float(torch.linalg.eigvalsh(floor - lower).min()) >= -1e-12 * float(scale)
    # On diagonal inputs the whitened max is the elementwise max.
    from vfe3.model.relational_operators import _whitened_max
    p, r = torch.rand(5, 4, dtype=F64) + 0.1, torch.rand(5, 4, dtype=F64) + 0.1
    torch.testing.assert_close(_whitened_max(torch.diag_embed(p), torch.diag_embed(r)),
                               torch.diag_embed(torch.maximum(p, r)), rtol=1e-12, atol=1e-14)


def test_max_floor_config_and_wiring() -> None:
    cfg = _quiet_config(predictive_covariance_floor_mode="max")
    assert "predictive_covariance_floor_mode" not in cfg.inactive_fields()
    model = VFEModel(cfg)
    assert {m.predictive_covariance_floor_mode for m in model.relational_operators_by_layer} == {"max"}
    with pytest.raises(ValueError, match="predictive_covariance_floor_mode"):
        _quiet_config(predictive_covariance_floor_mode="absolute")


# Items 18 and 1: the relative floor's lower bound f sigma_min / K.

def test_relative_floor_lower_bound_is_f_sigma_min_over_K_and_is_approached() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 4, False, (3,), F64, seed=8)
    ops.predictive_covariance_floor_mode = "relative"
    f, K = ops.predictive_covariance_floor, sigma.shape[-1]
    g = torch.Generator().manual_seed(9)
    for kappa in (1.0, 20.0, 1e3):
        frames = _conditioned_frames(sigma.shape[-2], 2, 4, kappa, g, batch=(3,))
        floor = ops.floor_term(None, sigma, frames, diagonal=True)
        bound = f * sigma.amin(dim=-1, keepdim=True) / K
        assert bool((floor >= bound * (1 - 1e-12)).all())
    # U = diag(1, t, ..., t): coordinate 0 of the floor tends to f sigma_0 / K as t grows.
    t = 1e4
    blocks = torch.diag_embed(torch.tensor([1.0, t, t, t], dtype=F64)).expand(1, 5, 2, 4, 4).clone()
    blocks[..., 1, :, :] = t * torch.eye(4, dtype=F64)
    frames = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    s = torch.full((1, 5, 8), 7.0, dtype=F64)
    s[..., 0] = 0.3                                                            # sigma_min at coordinate 0
    floor = ops.floor_term(None, s, frames, diagonal=True)
    torch.testing.assert_close(floor[..., 0], torch.full_like(floor[..., 0], f * 0.3 / K), rtol=1e-5, atol=0)


# Item 36: the relative floor from frame norms equals the product formula.

@pytest.mark.parametrize("block", [True, False])
def test_relative_floor_frame_norms_match_the_product_formula(block: bool) -> None:
    g = torch.Generator().manual_seed(11)
    H, d, N = 2, 4, 6
    K = H * d
    frames = _conditioned_frames(N, H, d, 30.0, g, batch=(2,))
    if not block:
        frames = CanonicalFrameContext(_dense(frames.forward), _dense(frames.inverse), block_diagonal=False)
    ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                    predictive_covariance="moment_matched", predictive_covariance_floor_mode="relative")
    for dtype, tol in ((F64, 1e-12), (torch.float32, 1e-5)):
        sigma = (torch.rand(2, N, K, generator=g, dtype=F64) + 0.1).to(dtype)
        frame = CanonicalFrameContext(frames.forward.to(dtype), frames.inverse.to(dtype),
                                      block_diagonal=frames.block_diagonal)
        got = ops.relative_floor_term(sigma, frame, True)
        U, U_inv = (_dense(frames.forward) if block else frames.forward), (_dense(frames.inverse) if block
                                                                          else frames.inverse)
        C = U_inv @ torch.diag_embed(sigma.double()) @ U_inv.mT
        expected = (ops.predictive_covariance_floor * torch.diagonal(C, dim1=-2, dim2=-1).sum(-1, keepdim=True) / K
                    * torch.diagonal(U @ U.mT, dim1=-2, dim2=-1))
        torch.testing.assert_close(got.double(), expected, rtol=tol, atol=0)


# Items 3 and 18: depth recursion on ill-conditioned frames, and the projection-loss measurement.

def _depth_traces(kappa: float, layers: int = 6) -> Tuple[list, list]:
    r"""Max reference trace per layer, diagonal vs head-block family: identity content, uniform causal beta,
    no residual, prior floor, K = 8, H = 2, N = 8, float64."""
    g = torch.Generator().manual_seed(0)
    H, d, N = 2, 4, 8
    K = H * d
    frame = _conditioned_frames(N, H, d, kappa, g)
    mu = torch.randn(1, N, K, generator=g, dtype=F64)
    sigma0 = torch.rand(1, N, K, generator=g, dtype=F64) + 0.5
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = (causal / causal.sum(-1, keepdim=True)).to(F64).expand(1, H, N, N)
    eye = torch.eye(K, dtype=F64)
    receive = eye.reshape(K, H, d).permute(1, 0, 2).contiguous()               # (H, K, d) identity O'_h
    U_inv = _dense(frame.inverse)
    out = []
    for block in (False, True):
        ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                        predictive_covariance="moment_matched", predictive_covariance_floor=1e-3,
                                        block_covariance=block).double()
        with torch.no_grad():
            ops.T.copy_(eye.reshape(H, d, K))                                   # identity T_h
        sigma = torch.diag_embed(sigma0) if block else sigma0
        anchor, traces = sigma.clone(), []
        for _ in range(layers):
            with torch.no_grad():
                sigma = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
            local = sigma if block else torch.diag_embed(sigma)
            traces.append(float(torch.diagonal(U_inv @ local @ U_inv.mT, dim1=-2, dim2=-1).sum(-1).max()))
        out.append(traces)
    return out[0], out[1]


def test_diagonal_projection_inflates_the_reference_trace_with_depth_and_the_block_family_does_not() -> None:
    diagonal, block = _depth_traces(20.0)
    assert diagonal[-1] > 2.0 * block[-1] and diagonal[-1] > 1e3 * block[-1]
    assert all(b < 2.0 * block[0] for b in block)                               # the head-block family stays flat
    assert all(later > earlier for earlier, later in zip(diagonal, diagonal[1:]))   # geometric growth
    flat_diagonal, flat_block = _depth_traces(1.0)                              # kappa = 1: the families agree
    torch.testing.assert_close(torch.tensor(flat_diagonal), torch.tensor(flat_block), rtol=1e-10, atol=0)


def _direct_projection_kl(ops, mu, sigma, beta, frame, receive, anchor) -> Tuple[float, float]:
    r"""KL(N(m, blockdiag) || N(m, diag)) and the reference-trace ratio from the dense float64 update."""
    full = ops._predictive_covariance_update_dense(mu, torch.diag_embed(sigma), beta, frame, receive)   # U C U^T
    H, d = len(ops.irrep_dims), ops.irrep_dims[0]
    blocks = torch.diagonal(full.reshape(*full.shape[:-2], H, d, H, d), dim1=-4, dim2=-2).movedim(-1, -3)
    floor = (ops.predictive_covariance_floor * anchor).reshape(*anchor.shape[:-1], H, d)
    floored = blocks + torch.diag_embed(floor)
    kl = 0.5 * (torch.log(torch.diagonal(floored, dim1=-2, dim2=-1)).sum(-1) - torch.logdet(floored)).sum(-1)
    inv = frame.inverse
    proj = (torch.diagonal(blocks, dim1=-2, dim2=-1) * inv.square().sum(-2)).sum((-2, -1))
    ref = torch.diagonal(inv @ blocks @ inv.mT, dim1=-2, dim2=-1).sum((-2, -1))
    return float(kl.mean()), float((proj / ref).mean())


def test_diagonal_projection_statistics_match_a_dense_construction() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 3, True, (2,), F64, seed=12)
    anchor = torch.full_like(sigma, 2.0)
    ops.diagnose_diagonal_projection = True
    for squares in (False, True):
        ops.predictive_covariance_spread_squares = squares
        with torch.no_grad():
            ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
        stats = ops.diagonal_projection_stats
        with torch.no_grad():
            kl, ratio = _direct_projection_kl(ops, mu, sigma, beta, frame, receive, anchor)
        assert float(stats["kl_mean"]) == pytest.approx(kl, rel=1e-9) and kl > 0.0
        assert float(stats["trace_ratio_mean"]) == pytest.approx(ratio, rel=1e-9)
    ops.diagonal_projection_stats = None
    with torch.enable_grad():                                                  # never under autograd
        ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
    assert ops.diagonal_projection_stats is None


def test_model_reports_projection_loss_and_positional_minima_per_layer() -> None:
    torch.manual_seed(0)
    cfg = _quiet_config(embed_dim=8, n_heads=2, n_layers=2, vocab_size=9, max_seq_len=6, phi_scale=0.5)
    model = VFEModel(cfg).eval()
    tokens = torch.randint(0, 9, (2, 6), generator=torch.Generator().manual_seed(1))
    per_layer = model.diagnostics_per_layer(tokens)
    for key in ("relational_diagonal_projection_kl", "relational_diagonal_projection_trace_ratio"):
        assert len(per_layer[key]) == 2 and all(math.isfinite(v) for v in per_layer[key])
    assert all(v >= -1e-9 for v in per_layer["relational_diagonal_projection_kl"])
    d = model.diagnostics(tokens)
    for key in ("relational_diagonal_projection_kl_layer_max", "relational_diagonal_projection_kl_argmax_layer",
                "belief_lam_min_argmin_position", "belief_lam_min_from_position_1",
                "relational_projected_cov_min_eigenvalue_argmin_position",
                "relational_projected_cov_min_eigenvalue_from_position_1"):
        assert math.isfinite(d[key]), key
    assert d["belief_lam_min_from_position_1"] >= d["belief_lam_min"]
    assert all(not ops.diagnose_diagonal_projection for ops in model.relational_operators_by_layer)


# Item 13: positional minimum.

def test_positional_minimum_reports_the_argmin_and_the_minimum_beyond_position_zero() -> None:
    values = torch.tensor([[1e-9, 0.5, 0.2, 3.0], [2.0, float("nan"), 0.7, 0.1]])
    position, tail = metrics.positional_minimum(values)
    assert position == 0.0 and tail == pytest.approx(0.1)
    assert metrics.positional_minimum(torch.tensor([[float("nan")]])) == (pytest.approx(float("nan"), nan_ok=True),
                                                                          pytest.approx(float("nan"), nan_ok=True))


# Item 7: the uncapped pushforward is counted and warns once per run.

def _layernorm_cfg(**over: Any) -> VFE3Config:
    values = dict(embed_dim=8, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=2,
                  norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                  block_mlp_mode="coordinate", block_mlp_expansion=2, block_norm_covariance="pushforward")
    values.update(over)
    return _quiet_config(**values)


def test_cap_off_at_the_launcher_initialization_counts_and_warns_once() -> None:
    tokens = torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    model = VFEModel(_layernorm_cfg(block_norm_trace_cap=False, mu_init_std=0.03, sigma_init=5.0)).eval()
    bm.reset_layernorm_trace_cap_elements()
    with torch.no_grad():
        model.forward_beliefs(tokens)
    assert bm.layernorm_uncapped_trace_ratio_max() > bm.UNCAPPED_TRACE_RATIO_WARN
    with pytest.warns(RuntimeWarning, match="block_norm_trace_cap=False"):
        count = bm.layernorm_trace_cap_elements()
    assert count > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                         # once per run
        bm.layernorm_trace_cap_elements()
    bm.reset_layernorm_trace_cap_elements()
    assert bm.layernorm_trace_cap_elements() == 0 and bm.layernorm_uncapped_trace_ratio_max() == 0.0


@pytest.mark.parametrize("affine", [False, True])
def test_cap_off_pushforward_matches_finite_differences_in_the_small_noise_regime(affine: bool) -> None:
    K = 6
    g = torch.Generator().manual_seed(3)
    norm = AffineLayerNorm(K, eps=1e-5) if affine else LayerNorm(K, eps=1e-5)
    if affine:
        norm = norm.double()
        with torch.no_grad():
            norm.weight.copy_(0.5 + torch.rand(K, generator=g, dtype=F64))
    mu = torch.randn(2, K, generator=g, dtype=F64).requires_grad_(True)
    sigma = (1e-3 * (torch.rand(2, K, generator=g, dtype=F64) + 0.5)).requires_grad_(True)   # Sigma << s^2
    full = (lambda A: A @ A.mT * 1e-3 + 1e-4 * torch.eye(K, dtype=F64))(
        torch.randn(2, K, K, generator=g, dtype=F64)).requires_grad_(True)

    def diag(m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return bm.block_norm_covariance_pushforward(norm, m, s, 1e-3, trace_cap=False)

    assert torch.autograd.gradcheck(diag, (mu, sigma), eps=1e-7, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradcheck(diag, (mu, full), eps=1e-7, atol=1e-6, rtol=1e-5)


# Item 8: the moment fallback where the cap binds, against Monte Carlo on anisotropic input.

def test_capped_pushforward_shape_error_and_the_moment_fallback_on_anisotropic_input() -> None:
    K = 32
    g = torch.Generator().manual_seed(0)
    norm = LayerNorm(K, eps=1e-5)
    mu = 0.1 * torch.randn(1, K, generator=g, dtype=F64)
    sigma = torch.full((1, K), 25.0, dtype=F64)
    sigma[..., :4] = 1e4                                                       # four coordinates 400 times larger
    x = mu + sigma.sqrt() * torch.randn(100_000, K, generator=g, dtype=F64)
    c = x - x.mean(-1, keepdim=True)
    mc = (c / (c.square().mean(-1, keepdim=True) + 1e-5).sqrt()).var(0)      # (K,) Monte Carlo variances
    bm.reset_layernorm_trace_cap_elements()
    capped = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-4)[0]
    fallback = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-4, cap_moment_fallback=True)[0]
    moment = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-4, mode="pushforward_moment")[0]
    assert bm.layernorm_trace_cap_elements() >= 2                               # the cap binds on both capped calls
    ratio = capped / mc
    assert float(ratio.max() / ratio.min()) > 1e3                              # the documented shape error
    torch.testing.assert_close(fallback, moment, rtol=0, atol=0)
    assert 0.6 < float((fallback / mc).min()) and float((fallback / mc).max()) < 1.4
    # Where the cap does not bind the fallback leaves the capped delta method unchanged.
    small = torch.full((1, K), 1e-4, dtype=F64)
    torch.testing.assert_close(bm.block_norm_covariance_pushforward(norm, mu, small, 1e-4, cap_moment_fallback=True),
                               bm.block_norm_covariance_pushforward(norm, mu, small, 1e-4), rtol=0, atol=0)


def test_cap_moment_fallback_config_and_model_forward() -> None:
    assert _layernorm_cfg().block_norm_cap_moment_fallback is False
    assert "block_norm_cap_moment_fallback" not in _layernorm_cfg().inactive_fields()
    assert "block_norm_cap_moment_fallback" in _layernorm_cfg(block_norm_trace_cap=False).inactive_fields()
    assert "block_norm_cap_moment_fallback" in _layernorm_cfg(
        block_norm_covariance="pushforward_moment").inactive_fields()
    with pytest.raises(ValueError, match="block_norm_cap_moment_fallback"):
        _layernorm_cfg(block_norm_cap_moment_fallback=1)
    tokens = torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    model = VFEModel(_layernorm_cfg(block_norm_cap_moment_fallback=True, mu_init_std=0.03, sigma_init=5.0)).eval()
    with torch.no_grad():
        belief, _ = model.forward_beliefs(tokens)
    assert bool(torch.isfinite(belief.sigma).all()) and bool((belief.sigma > 0).all())


# Item 32: an overflowed trace no longer gives all zeros or NaN.

@pytest.mark.parametrize("scale", [1e14, 1e15, 1e16])
def test_trace_cap_survives_an_overflowing_trace(scale: float) -> None:
    K = 32
    g = torch.Generator().manual_seed(2)
    mu = (5.0 + 1e-3 * torch.randn(4, K, generator=g)).float()
    sigma = torch.full((4, K), scale, dtype=torch.float32)
    norm = LayerNorm(K, eps=1e-5)
    out = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-4)
    assert bool(torch.isfinite(out).all()) and bool((out > 0).all())
    torch.testing.assert_close(out.double().sum(-1), torch.full((4,), float(K), dtype=F64), rtol=1e-5, atol=0)


def test_trace_cap_scale_is_float64_and_unchanged_in_range() -> None:
    norm = AffineLayerNorm(4, eps=1e-5).double()
    pushed = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
    bm.reset_layernorm_trace_cap_elements()
    lam = bm.layernorm_trace_cap(pushed, None, True)
    assert lam.dtype == F64 and float(lam[1]) == 1.0
    assert float(lam[0]) == pytest.approx(4.0 / 10.0, rel=1e-15)
    assert bm.layernorm_trace_cap_elements() == 1


# Item 12: the routed spread as a sum of squares after the sandwich.

def _outlier_probe(squares: bool, H: int = 2, d: int = 4, N: int = 16, kappa: float = 13.0,
                   outlier: float = 1e4, seed: int = 0) -> float:
    r"""Max relative fp32 error of the blockwise diagonal update against float64: near-identity content maps,
    frames of condition kappa, one outlier mean coordinate on the later half of the tokens."""
    g = torch.Generator().manual_seed(seed)
    K = H * d
    ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=True,
                                    predictive_covariance="moment_matched").double()
    eye = torch.eye(K, dtype=F64)
    with torch.no_grad():
        ops.T.copy_(eye.reshape(H, d, K) + 0.05 * torch.randn(ops.T.shape, generator=g, dtype=F64))
        ops.O.copy_(eye.reshape(K, H, d).permute(1, 0, 2)
                    + 0.05 * torch.randn(ops.O.shape, generator=g, dtype=F64))
    frame = _conditioned_frames(N, H, d, kappa, g)
    mu = torch.randn(1, N, K, generator=g, dtype=F64)
    mu[..., N // 2:, 0] += outlier
    sigma = torch.rand(1, N, K, generator=g, dtype=F64) + 0.5
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(torch.randn(1, H, N, N, generator=g, dtype=F64).masked_fill(~causal, -torch.inf), -1)
    ops.predictive_covariance_spread_squares = squares
    with torch.no_grad():
        ref = ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame)
        U32 = frame.forward.float()
        frame32 = CanonicalFrameContext(U32, torch.linalg.inv(U32), block_diagonal=True)
        got = ops.float()._predictive_covariance_update_blockwise(mu.float(), sigma.float(), beta.float(), frame32)
    return float(((got.double() - ref).abs() / ref.abs()).max())


def test_spread_squares_cuts_the_fp32_error_on_the_outlier_probe() -> None:
    old, new = _outlier_probe(False), _outlier_probe(True)
    # the old path's fp32 error depends on the BLAS (about 2e-3 under one build, 3e-5 under another)
    assert new < 1e-4 and new < old / 4.0


@pytest.mark.parametrize("residual", [False, True])
def test_spread_squares_equals_the_pooled_spread_in_float64(residual: bool) -> None:
    ops, mu, sigma, beta, frame, receive = _case(3, 2, residual, (2,), F64, seed=5)
    g = torch.Generator().manual_seed(6)
    A = torch.randn(*sigma.shape, sigma.shape[-1], generator=g, dtype=F64)
    full = A @ A.mT + torch.diag_embed(sigma)
    blocks = ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame, receive)
    dense = ops._predictive_covariance_update_dense(mu, full, beta, frame, receive)
    ops.block_covariance = True
    head_blocks = ops._predictive_covariance_update_blockwise(mu, full, beta, frame, receive)
    ops.predictive_covariance_spread_squares = True
    torch.testing.assert_close(ops._predictive_covariance_update_blockwise(mu, full, beta, frame, receive),
                               head_blocks, rtol=1e-12, atol=1e-12)
    ops.block_covariance = False
    torch.testing.assert_close(ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame, receive), blocks,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ops._predictive_covariance_update_dense(mu, full, beta, frame, receive), dense,
                               rtol=1e-12, atol=1e-12)


# Item 19: head pairs h <= g only.

@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("H", [2, 3, 4])
def test_symmetric_pairs_equal_the_all_pair_update_in_float64(residual: bool, H: int) -> None:
    ops, mu, sigma, beta, frame, receive = _case(H, 2, residual, (2,), F64, seed=H)
    g = torch.Generator().manual_seed(13)
    A = torch.randn(*sigma.shape, sigma.shape[-1], generator=g, dtype=F64)
    full = A @ A.mT + torch.diag_embed(sigma)
    for block, s in ((False, sigma), (True, full)):
        ops.block_covariance = block
        for maps in (None, receive):
            ops.predictive_covariance_symmetric_pairs = False
            ref = ops._predictive_covariance_update_blockwise(mu, s, beta, frame, maps)
            ops.predictive_covariance_symmetric_pairs = True
            got = ops._predictive_covariance_update_blockwise(mu, s, beta, frame, maps)
            assert float((got - ref).abs().max() / ref.abs().max()) <= 1e-12


def test_symmetric_pairs_and_spread_squares_config_and_refusals() -> None:
    cfg = _quiet_config(predictive_covariance_symmetric_pairs=True, predictive_covariance_spread_squares=True)
    inactive = cfg.inactive_fields()
    assert "predictive_covariance_symmetric_pairs" not in inactive
    assert "predictive_covariance_spread_squares" not in inactive
    ops = VFEModel(cfg).relational_operators
    assert ops.predictive_covariance_symmetric_pairs and ops.predictive_covariance_spread_squares
    off = _quiet_config(predictive_covariance="passthrough")
    assert {"predictive_covariance_symmetric_pairs", "predictive_covariance_spread_squares"} <= set(
        off.inactive_fields())
    for name in ("predictive_covariance_symmetric_pairs", "predictive_covariance_spread_squares"):
        with pytest.raises(ValueError, match=name):
            make_relational_operators([2, 2], 1, readout_mode="predictive_content",
                                      predictive_covariance="moment_matched", content_link_chart="exponential",
                                      content_link_edges="lowrank", **{name: True})
        with pytest.raises(ValueError, match=name):
            _quiet_config(**{name: 1})


# Item 33: the self-coupling clamp.

def test_self_kl_max_config_and_serialization() -> None:
    assert _quiet_config().self_kl_max is None
    assert "self_kl_max" in _quiet_config().inactive_fields()
    assert "self_kl_max" not in _quiet_config(mstep_self_coupling_weight=1.0).inactive_fields()
    with pytest.raises(ValueError, match="self_kl_max"):
        _quiet_config(self_kl_max=0.0)
    cfg = _quiet_config(self_kl_max=math.inf, mstep_self_coupling_weight=1.0)
    payload = serialize_config(cfg)
    assert payload["self_kl_max"] == "inf"
    assert config_from_serialized(payload, source="test").self_kl_max == math.inf


def _self_term_model(**over: Any) -> VFEModel:
    torch.manual_seed(3)
    values = dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2, phi_scale=0.4, kl_max=1e-3)
    values.update(over)
    return VFEModel(_quiet_config(**values)).eval()


def test_mstep_self_term_reads_self_kl_max() -> None:
    tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    losses = {}
    for name, self_cap in (("none", None), ("same", 1e-3), ("inf", math.inf)):
        model = _self_term_model(mstep_self_coupling_weight=1.0, self_kl_max=self_cap)
        with torch.no_grad():
            losses[name] = float(model(tokens, targets)[1])
    assert losses["none"] == losses["same"]
    assert losses["inf"] != losses["none"]


def test_canonical_self_term_is_unclamped_and_reports_the_clamped_share() -> None:
    tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    model = _self_term_model(diagnose_canonical_free_energy=True)
    d = model.diagnostics(tokens)
    assert d["self_divergence"] > 1e-3                                          # unclamped: above kl_max
    assert d["canonical_self_clamped_frac"] == 1.0
    loose = _self_term_model(diagnose_canonical_free_energy=True, mstep_self_coupling_weight=1.0,
                             self_kl_max=math.inf).diagnostics(tokens)
    assert loose["canonical_self_clamped_frac"] == 0.0
    assert loose["self_divergence"] == pytest.approx(d["self_divergence"], rel=1e-12)


# Item 41: once_differentiable on the spectral autograd Functions.

def _second_derivatives(fn, x: torch.Tensor) -> Tuple[str, str]:
    r"""The a2b pattern (tests/test_audit_20261016_a2b_decode_once_differentiable.py): (a) a plain cotangent, whose
    first gradient must carry no second derivative; (b) a cotangent that requires grad, where once_differentiable
    attaches its error node, so the second backward raises."""
    def outcome(call) -> str:
        try:
            call()
        except RuntimeError as exc:
            return "once_differentiable" if "once_differentiable" in str(exc) else "raised"
        return "returned"

    def output() -> torch.Tensor:
        out = fn(x)
        return out if torch.is_tensor(out) else out[0] + out[1]

    out = output()
    (plain,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    first = outcome(lambda: torch.autograd.grad(plain.sum(), x))
    out = output()
    cotangent = torch.ones_like(out, requires_grad=True)
    (carried,) = torch.autograd.grad(out, x, cotangent, create_graph=True)
    second = outcome(lambda: carried.square().sum().backward())
    return first, second


def test_spectral_functions_refuse_double_backward() -> None:
    g = torch.Generator().manual_seed(1)
    A = torch.randn(3, 4, 4, generator=g, dtype=F64)
    spd = (A @ A.mT + torch.eye(4, dtype=F64)).requires_grad_(True)
    cases = {
        "spectral_map": lambda M: rt._symmetric_spectral_map(M, "sqrt_floor", lower=1e-6),
        "spectral_map_pair": lambda M: rt._symmetric_spectral_map_pair(M, "sqrt_floor", "inv_sqrt_floor", lower=1e-6),
        "clipped_rebuild": lambda M: rt._ClippedSpectralRebuild.apply(
            M, *[t.detach() for t in torch.linalg.eigh(0.5 * (M + M.mT))],
            torch.full((3, 1), 1.5, dtype=F64), None),
        "spectral_ceiling": lambda M: cc._SpectralCeiling.apply(M, 2.0),
    }
    for name, fn in cases.items():
        assert _second_derivatives(fn, spd) == ("raised", "once_differentiable"), name


# Item 62: the safe_cholesky ridge is relative at every scale.

def test_safe_cholesky_ridge_is_scale_covariant_below_unit_mean_diagonal() -> None:
    g = torch.Generator().manual_seed(4)
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g, dtype=F64))
    lam = torch.linspace(1.0, 2.0, 16, dtype=F64)
    lam[0] = -1e-4                                                             # marginally indefinite
    M = Q @ torch.diag(lam) @ Q.T
    reference, ok = safe_cholesky(M, eps=1e-6, rounds=4)
    assert bool(ok)
    for scale in (100.0, 1e-3, 1e-6):
        L, ok = safe_cholesky(scale * M, eps=1e-6, rounds=4)
        assert bool(ok)
        torch.testing.assert_close(L, math.sqrt(scale) * reference, rtol=1e-8, atol=0)


# Item 63: retract_spd_full skips the projection on rows certified inside the band.

def test_retract_spd_full_skip_equals_the_projected_reference() -> None:
    g = torch.Generator().manual_seed(5)
    K = 5
    A = torch.randn(12, K, K, generator=g, dtype=F64)
    sigma = A @ A.mT / K + 0.5 * torch.eye(K, dtype=F64)
    sigma[0] = 1e-6 * torch.eye(K, dtype=F64) + 1e-7 * (A[0] @ A[0].mT)      # near the floor
    sigma[1] = 9.5 * torch.eye(K, dtype=F64)                                   # near the ceiling
    delta = torch.randn(12, K, K, generator=g, dtype=F64)
    delta = 0.5 * (delta + delta.mT)
    sigma.requires_grad_(True)
    delta.requires_grad_(True)
    kwargs = dict(step_size=0.3, trust_region=2.0, eps=1e-6, sigma_max=10.0)
    skipped = rt._retract_spd_full(sigma, delta, skip_inside=True, **kwargs)
    reference = rt._retract_spd_full(sigma, delta, skip_inside=False, **kwargs)
    torch.testing.assert_close(skipped, reference, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(rt.retract_spd_full(sigma, delta, **kwargs), skipped, rtol=0, atol=0)
    weight = torch.randn(12, K, K, generator=g, dtype=F64)
    grads = [torch.autograd.grad((out * weight).sum(), (sigma, delta)) for out in (skipped, reference)]
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)
