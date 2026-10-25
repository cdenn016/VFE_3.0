r"""Tests for the diagonal cue/result carrier (``cue_result_memory`` on ``gaussian_frame_diagonal``).

The diagonal kernel is the exact posterior for a diagonal record covariance followed by the diagonal M-projection of
the result covariance. The reference is the dense exact kernel (itself checked against independent joint conditioning
in ``tests/test_cue_result_memory_20261024.py``) at :math:`C = \operatorname{diag}c`: the means, log weights and log
evidence must agree, and the variances must equal the diagonal of the dense covariance. The tests also cover tiling
with checkpointed gradients, float64 ``gradcheck``, the block's layout dispatch, the reference block on the
frame-diagonal family with its passive invariance, the configured model path (untied and tied) and the configuration
boundary. CPU only, one thread, :math:`K \le 4`.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import pytest
import torch

from tests.profiles import test_config
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.block import _reference_coordinate_block
from vfe3.model.block_mlp import CueResultMemoryBlockMLP
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior
from vfe3.model.model import VFEModel

torch.set_num_threads(1)
DT = torch.float64
TOL = 1e-11


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _close(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> None:
    a, b = a.detach(), b.detach()
    err = float((a - b).abs().max())
    scale = max(1.0, float(b.abs().max()))
    assert err <= tol * scale, f"{what}: max error {err:.3e} (scale {scale:.3e})"


def _bank(K: int, M: int, seed: int) -> Dict[str, torch.Tensor]:
    g = _gen(seed)
    return {"nu": 1.5 * torch.randn(M, K, generator=g, dtype=DT), "omega": torch.randn(M, K, generator=g, dtype=DT),
            "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=DT),
            "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=DT) - 1.0,
            "log_pi": torch.randn(M, generator=g, dtype=DT)}


def _regression(K: int, M: int, rank: Optional[int], seed: int) -> Dict[str, torch.Tensor]:
    g = _gen(seed)
    if rank is None:
        D = 0.6 * torch.randn(M, K, K, generator=g, dtype=DT)
        D[0] = -2.0 * torch.eye(K, dtype=DT)                                   # F = -I: a reversed response
        return {"regression_offset": D}
    return {"regression_left": 0.7 * torch.randn(M, K, rank, generator=g, dtype=DT),
            "regression_right": 0.7 * torch.randn(M, K, rank, generator=g, dtype=DT)}


@pytest.mark.parametrize("rank", (None, 1, 3))
def test_diagonal_kernel_matches_dense_kernel_on_diagonal_records(rank: Optional[int]) -> None:
    K, M, T = 4, 6, 5
    bank, reg = _bank(K, M, seed=1), _regression(K, M, rank, seed=2)
    xi = torch.randn(T, K, generator=_gen(3), dtype=DT)
    c = 0.3 + torch.rand(T, K, generator=_gen(4), dtype=DT)
    diag = cue_result_diagonal_posterior(xi, c, **bank, **reg)
    dense = cue_result_exact_posterior(xi, torch.diag_embed(c), **bank, **reg)
    assert diag.covariance.shape == (T, K)
    _close(diag.mean, dense.mean, TOL, "mean")
    _close(diag.log_weights, dense.log_weights, TOL, "log weights")
    _close(diag.log_evidence, dense.log_evidence, TOL, "log evidence")
    _close(diag.covariance, dense.covariance.diagonal(dim1=-2, dim2=-1), TOL, "variances = diag of dense covariance")
    # The projection discards real off-diagonal covariance (between components and through F), so the test has power.
    off = dense.covariance - torch.diag_embed(dense.covariance.diagonal(dim1=-2, dim2=-1))
    assert float(off.abs().max()) > 1e-3


@pytest.mark.parametrize("rank", (None, 2))
def test_tiling_and_checkpointed_gradients_do_not_change_results(rank: Optional[int]) -> None:
    K, M, T = 3, 7, 6
    bank, reg = _bank(K, M, seed=11), _regression(K, M, rank, seed=12)
    xi = torch.randn(T, K, generator=_gen(13), dtype=DT)
    c = 0.3 + torch.rand(T, K, generator=_gen(14), dtype=DT)
    params = {**bank, **reg}

    def run(tile: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        out = cue_result_diagonal_posterior(xi, c, **leaves, tile_bytes=tile)
        loss = (out.mean.square().sum() + out.covariance.sum() + out.log_evidence.sum())
        grads = torch.autograd.grad(loss, list(leaves.values()))
        return (out.mean, out.covariance, out.log_weights, out.log_evidence), grads

    whole, g_whole = run(10 ** 9)
    for tile in (1, 300, 2000):                                              # tiles of one, a few and many pairs
        part, g_part = run(tile)
        for a, b in zip(part, whole):
            _close(a, b, 1e-12, f"tile {tile}: value")
        for a, b in zip(g_part, g_whole):
            _close(a, b, 1e-11, f"tile {tile}: gradient")


@pytest.mark.parametrize("rank", (None, 1))
def test_gradcheck_float64(rank: Optional[int]) -> None:
    K, M, T = 2, 3, 2
    bank, reg = _bank(K, M, seed=21), _regression(K, M, rank, seed=22)
    xi = torch.randn(T, K, generator=_gen(23), dtype=DT)
    log_c = 0.3 * torch.randn(T, K, generator=_gen(24), dtype=DT)
    names = ("nu", "omega", "log_lambda", "log_result_noise", "log_pi")
    reg_names = tuple(reg)
    inputs = tuple(t.clone().requires_grad_(True) for t in (xi, log_c, *(bank[n] for n in names), *reg.values()))

    def f(xi_, log_c_, nu, omega, log_lambda, log_q, log_pi, *reg_):
        out = cue_result_diagonal_posterior(xi_, log_c_.exp(), nu, omega, log_lambda, log_q, log_pi,
                                            **dict(zip(reg_names, reg_)), tile_bytes=200)   # tiled and checkpointed
        return out.mean, out.covariance, out.log_evidence

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


def test_block_dispatches_on_the_covariance_layout() -> None:
    K = 3
    torch.manual_seed(0)
    mlp = CueResultMemoryBlockMLP(K, 2, 0.0, template_init_variance=2.0, regression_rank=2).double()
    with torch.no_grad():
        mlp.regression_left.normal_(0.0, 0.3, generator=_gen(31))
    xi = torch.randn(4, K, generator=_gen(32), dtype=DT)
    c = 0.3 + torch.rand(4, K, generator=_gen(33), dtype=DT)
    diag = mlp.forward_moments(xi, c)
    dense = mlp.forward_moments(xi, torch.diag_embed(c))
    assert diag.sigma.shape == (4, K) and dense.sigma.shape == (4, K, K)
    _close(diag.mu, dense.mu, TOL, "same mean on both layouts")
    _close(diag.sigma, dense.sigma.diagonal(dim1=-2, dim2=-1), TOL, "diagonal of the dense covariance")
    with pytest.raises(ValueError, match="reads a dense"):
        mlp.forward_moments(xi, torch.ones(4, K, K, K, dtype=DT))


def _frame(K: int, seed: int, scale: float = 0.4) -> CanonicalFrameContext:
    U = torch.linalg.matrix_exp(scale * torch.randn(2, 3, K, K, generator=_gen(seed), dtype=DT))
    return CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=False)


def _run(mlp: CueResultMemoryBlockMLP, frame: CanonicalFrameContext, mu: torch.Tensor, c: torch.Tensor) -> BeliefState:
    belief = BeliefState(mu=mu, sigma=c, phi=torch.zeros(2, 3, 1, dtype=DT))
    return _reference_coordinate_block(belief, block_norm=None, block_mlp=mlp, frame=frame, capture=None,
                                       hidden_bias=None, frame_diagonal=True)


def test_reference_block_on_frame_diagonal_reads_and_returns_reference_variances() -> None:
    K = 4
    torch.manual_seed(1)
    mlp = CueResultMemoryBlockMLP(K, 2, 0.0, template_init_variance=2.0, regression_rank=2).double()
    with torch.no_grad():
        mlp.regression_left.normal_(0.0, 0.3, generator=_gen(41))
        mlp.omega.add_(0.3 * torch.randn(mlp.omega.shape, generator=_gen(42), dtype=DT))
    frame = _frame(K, seed=43)
    mu = torch.randn(2, 3, K, generator=_gen(44), dtype=DT)
    c = 0.3 + torch.rand(2, 3, K, generator=_gen(45), dtype=DT)               # stored reference variances
    xi = (frame.inverse @ mu.unsqueeze(-1)).squeeze(-1)
    direct = mlp.posterior(xi, c)
    out = _run(mlp, frame, mu, c)
    _close(out.mu, (frame.forward @ direct.mean.unsqueeze(-1)).squeeze(-1), TOL, "mu_plus = U m_Y")
    _close(out.sigma, direct.covariance, TOL, "stored c_plus = reference result variances")
    # Passive action (mu, U) -> (g mu, g U) with c unchanged: mu_plus -> g mu_plus and c_plus unchanged.
    g = torch.linalg.matrix_exp(0.7 * torch.randn(2, 3, K, K, generator=_gen(46), dtype=DT))
    moved = CanonicalFrameContext(g @ frame.forward, frame.inverse @ torch.linalg.inv(g), block_diagonal=False)
    out_g = _run(mlp, moved, (g @ mu.unsqueeze(-1)).squeeze(-1), c)
    _close(out_g.mu, (g @ out.mu.unsqueeze(-1)).squeeze(-1), TOL, "mu_plus covariant")
    _close(out_g.sigma, out.sigma, TOL, "reference variances invariant")


_ROUTE = dict(attention_mode="projected_predictive", predictive_readout="kernel", family="gaussian_frame_diagonal",
              decode_mode="linear", norm_type_block="none", block_coordinates="reference", use_block_mlp=True,
              block_mlp_mode="cue_result_memory", block_mlp_inference="exact", block_mlp_covariance="posterior",
              block_mlp_dropout=0.0, block_mlp_expansion=2, n_layers=2, max_seq_len=4,
              block_memory_regression_rank=2)


def _cfg(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**{**_ROUTE, **overrides})


@pytest.mark.parametrize("over", (dict(), dict(block_mlp_tie_layers=True),
                                  dict(norm_type_block="layernorm", block_norm_covariance="pushforward"),
                                  dict(block_memory_regression_rank=None)))
def test_configured_frame_diagonal_model_path(over: dict) -> None:
    cfg = _cfg(**over)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=_gen(51))
        model.train()
        _, loss, _ = model(tokens, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    # Expected silent gradients: V_a while U_a = 0 at initialization, and, untied under the linear decoder, the last
    # layer's result-noise variances (the decoder reads only the mean of the final belief).
    last = cfg.n_layers - 1
    allowed = {f"block_mlps.{l}.regression_right" for l in range(cfg.n_layers)}
    if not cfg.block_mlp_tie_layers:
        allowed.add(f"block_mlps.{last}.log_result_noise")
    for name, p in model.named_parameters():
        if not name.startswith("block_mlps."):
            continue
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()), name
        if name not in allowed:
            assert p.grad is not None and float(p.grad.abs().max()) > 0.0, name


def test_configuration_boundary() -> None:
    assert _cfg().family == "gaussian_frame_diagonal"
    with pytest.raises(ValueError, match="cue_result_memory"):
        _cfg(family="gaussian_diagonal")                         # local diagonal storage is not covariant
    with pytest.raises(ValueError):
        _cfg(block_mlp_mode="gaussian_template", block_mlp_covariance="passthrough")   # still coordinate-only family
    with pytest.raises(ValueError, match="cue_result_memory"):
        _cfg(block_coordinates="local")
