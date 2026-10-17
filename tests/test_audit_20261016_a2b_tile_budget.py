r"""Scorer tile budget of the 2026-10-16 audit (F26), tiny CPU shapes.

:func:`vfe3.relational_scoring.set_projection_tile_bytes` sets the byte budget of one scorer tile (projected moments,
source-root factor, GEMM-fallback direct pair tiles, mean-distance rows). A raised budget changes only the tile count:
a loop whose whole transient exceeds ``_CHECKPOINT_BYTES`` is still checkpointed when it fits one tile, so raising the
budget never makes autograd save the intermediates the default checkpoints. At the default budget the tiling and the
checkpointing are those before the knob.
"""
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

import vfe3.relational_scoring as scoring
from vfe3.relational_scoring import projected_gaussian_kl, projected_moments


def _inputs(seed: int = 0, B: int = 1, N: int = 4, H: int = 2, d: int = 4, r: int = 3) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(B, N, H, d, generator=g)
    sigma = torch.rand(B, N, H, d, generator=g) + .5
    inverse = torch.eye(d).expand(B, N, H, d, d) + .05 * torch.randn(B, N, H, d, d, generator=g)
    S = torch.randn(2, H, r, d, generator=g)                                       # role-stacked (R, H, r, d)
    return [x.clone().requires_grad_() for x in (mu, sigma, inverse, S)]


def _checkpoint_spy(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    calls: List[str] = []
    real = scoring.checkpoint

    def spy(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        calls.append(function.__name__)
        return real(function, *args, **kwargs)
    monkeypatch.setattr(scoring, "checkpoint", spy)
    return calls


def _readers(inputs: List[torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, ...]]:
    r"""Each tiled reader of the budget with its gradients with respect to the source moments."""
    mu, sigma, inverse, S = inputs
    out: Dict[str, Tuple[torch.Tensor, ...]] = {}
    a, G = projected_moments(mu, sigma, inverse, S)                                # (R, B, N, H, r[, r])
    out["moments"] = (a, G, *torch.autograd.grad(a.sum() + G.square().sum(), inputs))
    a, G = a.detach().requires_grad_(), G.detach().requires_grad_()
    C, _ = scoring._projected_root_factor((sigma, inverse, S[0]), G[0].shape[:-2], torch.float32)
    out["root_factor"] = (C, *torch.autograd.grad(C.square().sum(), (sigma, inverse)))
    D = projected_gaussian_kl(a[0], G[0], a[1], G[1], symmetric_inputs=True, gemm_expansion_tol=1e-30)
    out["direct_tiles"] = (D, *torch.autograd.grad(D.square().sum(), (a, G)))
    Dm = scoring.projected_mean_distance_cost(a[0], G[0], a[1], G[1])
    out["mean_distance"] = (Dm, *torch.autograd.grad(Dm.square().sum(), (a,)))
    return out


_TILES = {"moments": "_projected_moments_tile", "root_factor": "_projected_root_factor_tile",
          "direct_tiles": "_projected_kl_direct_tile", "mean_distance": "_mean_distance_tile"}


def test_projection_tile_bytes_setter_validates_and_returns_the_previous_budget() -> None:
    assert scoring.projection_tile_bytes() == 32 * 1024 ** 2
    previous = scoring.set_projection_tile_bytes(256 * 1024 ** 2)
    try:
        assert previous == 32 * 1024 ** 2
        assert scoring.projection_tile_bytes() == scoring._PROJECTION_TILE_BYTES == 256 * 1024 ** 2
        for bad in (0, -1, 1.5, True, "32"):
            with pytest.raises(ValueError, match="tile budget"):
                scoring.set_projection_tile_bytes(bad)                             # type: ignore[arg-type]
        assert scoring.projection_tile_bytes() == 256 * 1024 ** 2
    finally:
        assert scoring.set_projection_tile_bytes(previous) == 256 * 1024 ** 2
    assert scoring.projection_tile_bytes() == 32 * 1024 ** 2


def test_default_budget_leaves_tiny_scorer_loops_untiled_and_unsaved(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _checkpoint_spy(monkeypatch)
    _readers(_inputs())
    assert calls == []


def test_raised_budget_still_checkpoints_a_loop_above_the_checkpoint_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""One tile under a raised budget is still checkpointed when its transient exceeds the floor, bitwise unchanged."""
    reference = _readers(_inputs())                                                # untiled, nothing checkpointed
    monkeypatch.setattr(scoring, "_CHECKPOINT_BYTES", 64, raising=False)          # every tiny loop exceeds it
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1 << 30)              # every tiny loop fits one tile
    calls = _checkpoint_spy(monkeypatch)
    actual = _readers(_inputs())
    for name, tile in _TILES.items():
        assert calls.count(tile) == 1, (name, calls)
        for x, y in zip(actual[name], reference[name]):
            torch.testing.assert_close(x, y, rtol=0., atol=0.)


def test_budget_sets_the_tile_count_and_keeps_outputs_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Many small tiles against one: equal values, gradients to float32 reduction order (the shared maps sum tiles)."""
    monkeypatch.setattr(scoring, "_CHECKPOINT_BYTES", 64, raising=False)
    previous = scoring.set_projection_tile_bytes(1 << 30)
    try:
        single = _readers(_inputs(seed=3, N=8))
        scoring.set_projection_tile_bytes(1)                                       # one token, row or pair per tile
        calls = _checkpoint_spy(monkeypatch)
        tiled = _readers(_inputs(seed=3, N=8))
    finally:
        scoring.set_projection_tile_bytes(previous)
    assert calls.count("_projected_moments_tile") == 8 and calls.count("_mean_distance_tile") == 8
    for name in _TILES:
        for x, y in zip(tiled[name], single[name]):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
