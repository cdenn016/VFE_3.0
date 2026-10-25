r"""The cue/result memory's tile budget (``block_memory_tile_mib``): tiled and untiled results agree.

The exact posterior is tiled over (tokens x components) only to bound transient memory; the law of total covariance
merges the tiles, and with gradients on each tile is checkpointed. The budget must change the tile count and nothing
else: the dense and diagonal kernels, with a shared or a per-token (carried regime) prior, return the same moments,
log weights, log evidence and gradients at a budget of one pair per tile, a few pairs, and the whole problem in one
tile. The config field is validated, reaches the memory on the untied and tied paths, is inert off the memory, and the
whole tied model with its carried regime gives the same loss and gradients tiled and untiled. CPU only, one thread,
float64, :math:`K \le 4`.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import pytest
import torch

from tests.profiles import config_values, test_config
from vfe3.model.block_mlp import CueResultMemoryBlockMLP
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior
from vfe3.model.model import VFEModel

torch.set_num_threads(1)
DT = torch.float64
WHOLE = 10 ** 12                                                               # one tile: the untiled reference


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _close(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> None:
    err = float((a.detach() - b.detach()).abs().max())
    scale = max(1.0, float(b.detach().abs().max()))
    assert err <= tol * scale, f"{what}: max error {err:.3e} (scale {scale:.3e})"


def _params(K: int, M: int, T: int, rank: Optional[int], per_token: bool, seed: int) -> Dict[str, torch.Tensor]:
    g = _gen(seed)
    out = {"nu": 1.5 * torch.randn(M, K, generator=g, dtype=DT), "omega": torch.randn(M, K, generator=g, dtype=DT),
           "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=DT),
           "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=DT) - 1.0,
           "log_pi": torch.randn(*((T,) if per_token else ()), M, generator=g, dtype=DT)}   # (M,) or (T, M)
    if rank is None:
        out["regression_offset"] = 0.6 * torch.randn(M, K, K, generator=g, dtype=DT)
    else:
        out["regression_left"] = 0.7 * torch.randn(M, K, rank, generator=g, dtype=DT)
        out["regression_right"] = 0.7 * torch.randn(M, K, rank, generator=g, dtype=DT)
    return out


@pytest.mark.parametrize("layout", ("diagonal", "dense"))
@pytest.mark.parametrize("rank", (None, 2))
@pytest.mark.parametrize("per_token", (False, True))
def test_kernel_results_and_gradients_do_not_depend_on_the_tile_budget(layout: str, rank: Optional[int],
                                                                       per_token: bool) -> None:
    K, M, T = 3, 7, 6
    params = _params(K, M, T, rank, per_token, seed=1)
    xi = torch.randn(T, K, generator=_gen(2), dtype=DT)
    c = 0.3 + torch.rand(T, K, generator=_gen(3), dtype=DT)
    A = torch.randn(T, K, K, generator=_gen(4), dtype=DT)
    C = A @ A.transpose(-1, -2) / K + 0.2 * torch.eye(K, dtype=DT)          # (T, K, K) SPD record covariance

    def run(tile: int) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        x = xi.clone().requires_grad_(True)
        if layout == "diagonal":
            out = cue_result_diagonal_posterior(x, c, **leaves, tile_bytes=tile)
        else:
            out = cue_result_exact_posterior(x, C, **leaves, tile_bytes=tile)
        loss = out.mean.square().sum() + out.covariance.sum() + out.log_evidence.sum() + out.log_weights.exp().pow(2).sum()
        grads = torch.autograd.grad(loss, [x, *leaves.values()])
        return (out.mean, out.covariance, out.log_weights, out.log_evidence), grads

    whole, g_whole = run(WHOLE)
    for tile in (1, 500, 5000):                                              # one pair, a few pairs, many pairs
        part, g_part = run(tile)
        for name, a, b in zip(("mean", "covariance", "log weights", "log evidence"), part, whole):
            _close(a, b, 1e-12, f"tile {tile}: {name}")
        for a, b in zip(g_part, g_whole):
            _close(a, b, 1e-11, f"tile {tile}: gradient")


_ROUTE = dict(attention_mode="projected_predictive", predictive_readout="kernel", family="gaussian_frame_diagonal",
              decode_mode="linear", norm_type_block="none", block_coordinates="reference", use_block_mlp=True,
              block_mlp_mode="cue_result_memory", block_mlp_inference="exact", block_mlp_covariance="posterior",
              block_mlp_dropout=0.0, block_mlp_expansion=2, n_layers=2, max_seq_len=4,
              block_memory_regression_rank=2)


def _cfg(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**{**_ROUTE, **overrides})


def test_config_field_is_validated_and_inert_off_the_memory() -> None:
    assert config_values()["block_memory_tile_mib"] == 32
    for bad in (0, -1, 1.5, True, "32"):
        with pytest.raises(ValueError, match="block_memory_tile_mib"):
            _cfg(block_memory_tile_mib=bad)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = test_config(block_memory_tile_mib=64)
    assert "block_memory_tile_mib" in plain.inactive_fields()
    assert "block_memory_tile_mib" not in _cfg().inactive_fields()


@pytest.mark.parametrize("over", (dict(), dict(block_mlp_tie_layers=True, block_memory_regime_transition="sticky")))
def test_field_reaches_every_memory_module(over: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_cfg(block_memory_tile_mib=1024, **over))
    assert all(isinstance(m, CueResultMemoryBlockMLP) and m.tile_bytes == 1024 * 1024 ** 2 for m in model.block_mlps)


def test_tied_model_with_carried_regime_agrees_tiled_and_untiled() -> None:
    cfg = _cfg(block_mlp_tie_layers=True, block_memory_regime_transition="sticky", n_layers=3)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=_gen(9))

    def run(tile: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg).double()
            model.block_mlps[0].tile_bytes = tile                           # the shared bank serves every layer
            model.train()
            _, loss, _ = model(tokens, tokens)
        loss.backward()
        return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    loss_whole, g_whole = run(WHOLE)
    loss_tiled, g_tiled = run(1)                                             # one (token, component) pair per tile
    _close(loss_tiled, loss_whole, 1e-12, "loss")
    assert g_tiled.keys() == g_whole.keys()
    assert any("regime" in name for name in g_whole)
    for name in g_whole:
        _close(g_tiled[name], g_whole[name], 1e-10, f"gradient {name}")
