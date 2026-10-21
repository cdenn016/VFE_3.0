r"""Audit 2026-10-20, items 16, 17 and 18 (and 6): tiling, layout and causal truncation of the moment-matched update.

Item 16: the routed-spread tiles are sized with every leading (batch) axis. Item 17: the blockwise head-pair chain and
the spread tiles emit their tensors in the order the next GEMM reads. Item 18: under a strictly causal routing prior
each spread tile contracts only its causal sources. Each change is pinned against an oracle, the pre-change code
copied below verbatim (module references qualified), in float64 to 1e-12 relative (values and every leaf gradient)
and in float32 to rounding, at tiny CPU shapes: odd N, B > 1, several heads, causal and causal ALiBi priors, both
spread forms, both pair forms, the update checkpoint on and off, and tile budgets that force one tile or many.
"""
from __future__ import annotations

from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.profiles import make_relational_operators
from vfe3.attention_prior import attention_log_prior
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients import relational as relational_gradients
from vfe3.model import relational_operators as ro

F64 = torch.float64


@pytest.fixture(autouse=True)
def _two_threads() -> None:
    torch.set_num_threads(2)


@pytest.fixture(autouse=True)
def _restore_tile_budget() -> Any:
    previous = ro.spread_tile_bytes()
    yield
    ro.set_spread_tile_bytes(previous)


# The oracle: the routed spreads and the blockwise body as of commit 5aac905 (origin/main, 2026-10-20).

def _old_centered_spread_tile(b: torch.Tensor, values: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    diff = values.unsqueeze(-4) - mean.unsqueeze(-3)                          # (..., I, N, H, d) v_j - m_i
    weighted = b.movedim(-3, -1).unsqueeze(-1) * diff                          # (..., I, N, H, d)
    spread = torch.einsum("...ijhs,...ijht->...ihst", weighted, diff)         # (..., I, H, d, d)
    residual = weighted.sum(dim=-3)                                            # (..., I, H, d) sum_j b (v_j - m_i)
    return spread - residual.unsqueeze(-1) * residual.unsqueeze(-2)


def _old_centered_spread(b: torch.Tensor, values: torch.Tensor, mean: torch.Tensor,
                         causal: bool = False) -> torch.Tensor:
    N = b.shape[-2]
    per_row = values[..., 0, :, :].numel() * values.element_size() * max(1, values.shape[-3])
    step = max(1, min(N, ro._SPREAD_TILE_BYTES // max(1, per_row)))
    tiles = []
    for i in range(0, N, step):
        args = (b[..., i:i + step, :], values, mean[..., i:i + step, :, :])
        if (torch.is_grad_enabled() and (step < N or N * per_row > ro._SPREAD_CHECKPOINT_BYTES)
                and any(x.requires_grad for x in args)):
            tiles.append(checkpoint(_old_centered_spread_tile, *args, use_reentrant=False, preserve_rng_state=False))
        else:
            tiles.append(_old_centered_spread_tile(*args))
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=-4)


def _old_spread_squares_tile(b: torch.Tensor, values: torch.Tensor, mean: torch.Tensor, lift: torch.Tensor,
                             layout: str, block: int) -> torch.Tensor:
    diff = values.unsqueeze(-4) - mean.unsqueeze(-3)                          # (..., I, N, H, d) v_j - m_i
    w = torch.einsum("...ihas,...ijhs->...ijha", lift, diff)                   # (..., I, N, H, A) W_ih (v_j - m_i)
    weight = b.movedim(-3, -1)                                                 # (..., I, N, H) beta_ij^h
    if layout == "diag":
        return torch.einsum("...ijh,...ijha->...ia", weight, w.square())      # (..., I, A)
    if layout == "blocks":
        w = w.reshape(*w.shape[:-1], w.shape[-1] // block, block)              # (..., I, N, H, A/block, block)
        return torch.einsum("...ijh,...ijhls,...ijhlt->...ilst", weight, w, w)
    return torch.einsum("...ijh,...ijha,...ijhc->...iac", weight, w, w)       # (..., I, A, A)


def _old_spread_squares(b: torch.Tensor, values: torch.Tensor, mean: torch.Tensor, lift: torch.Tensor, layout: str,
                        block: int = 1, causal: bool = False) -> torch.Tensor:
    N = b.shape[-2]
    A = lift.shape[-2]
    per_row = max(1, N * values.shape[-2] * A * values.element_size())
    step = max(1, min(N, ro._SPREAD_TILE_BYTES // per_row))
    lift = torch.broadcast_to(lift, (*mean.shape[:-2], *lift.shape[-3:]))     # (..., N, H, A, d)
    tiles = []
    for i in range(0, N, step):
        args = (b[..., i:i + step, :], values, mean[..., i:i + step, :, :], lift[..., i:i + step, :, :, :])
        if (torch.is_grad_enabled() and (step < N or N * per_row > ro._SPREAD_CHECKPOINT_BYTES)
                and any(x.requires_grad for x in args)):
            tiles.append(checkpoint(_old_spread_squares_tile, *args, layout, block, use_reentrant=False,
                                    preserve_rng_state=False))
        else:
            tiles.append(_old_spread_squares_tile(*args, layout, block))
    axis = -2 if layout == "diag" else (-4 if layout == "blocks" else -3)
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=axis)


def _old_blockwise(self: Any, mu: torch.Tensor, sigma: torch.Tensor, beta: torch.Tensor,
                   frame: CanonicalFrameContext, receive: Optional[torch.Tensor] = None, edges: Any = None,
                   **_: Any) -> torch.Tensor:
    H, d = len(self.irrep_dims), self.irrep_dims[0]
    block = sigma.dim() == mu.dim() + 1                                            # head-block family
    dtype = ro._compute_dtype(mu, sigma, self.T, self.O, frame.forward)
    if edges is None and self.content_link_edges != "off":
        edges = self.edge_context(mu, beta, frame, receive, sigma=sigma)
    if edges is not None:
        edges = ro.edge_links.cast_context(edges, dtype)
        receive = ro.edge_links.receive_heads(edges, H)
    with torch.autocast(device_type=mu.device.type, enabled=False):
        forward, inverse = frame.forward.to(dtype), frame.inverse.to(dtype)        # (..., N, H, d, d)
        if block:
            S = sigma.to(dtype).reshape(*sigma.shape[:-2], H, d, H, d)            # (..., N, H, d, H, d)
            sigma_blocks = torch.diagonal(S, dim1=-4, dim2=-2).movedim(-1, -3)    # (..., N, H, d, d) Sigma_{i,m}
            C = inverse @ sigma_blocks @ inverse.transpose(-1, -2)                 # (..., N, H, d, d) C_{i,m}
        else:
            sigma_h = sigma.to(dtype).reshape(*sigma.shape[:-1], H, d)             # (..., N, H, d)
            C = (inverse * sigma_h.unsqueeze(-2)) @ inverse.transpose(-1, -2)      # (..., N, H, d, d) C_{i,m}
        C = 0.5 * (C + C.transpose(-1, -2))
        mu_h = mu.to(dtype).reshape(*mu.shape[:-1], H, d)
        xi = (inverse @ mu_h.unsqueeze(-1)).squeeze(-1).flatten(-2)                # (..., N, K)
        T_hat, O_hat = self._ambient_content_maps(dtype, receive)                  # (H,d,K), (H,K,d)
        T_blocks = T_hat.reshape(H, d, H, d)                                       # [h, s, m, k] = T_h^{(m)}[s, k]
        O_blocks = O_hat.reshape(H, H, d, d)                                       # [h, l, k, s] = O_h^{(l)}[k, s]
        b = beta.to(dtype)                                                         # (..., H, N, N)
        eye_h = torch.eye(H, dtype=dtype, device=mu.device)

        xi = xi - xi[..., :1, :]                                                   # (..., N, K)
        values = torch.einsum("hsk,...nk->...nhs", T_hat, xi)                      # (..., N, H, d)
        nl = C.dim() - 4                                                           # leading batch dims
        C_cols = C.movedim(-3, 0).movedim(-2, 1).reshape(H, d, -1)                # (m, k, [..., n, t])
        T_rows = T_blocks.permute(2, 0, 1, 3).reshape(H, H * d, d)                # (m, [h, s], k)
        TC = ro._per_block_bmm(T_rows, C_cols).view(H, H, d, *C.shape[:-3], d)    # (m, h, s, ..., n, t)
        TC = TC.permute(*range(3, nl + 4), 1, 0, 2, nl + 4)                        # (..., N, H, H, d, d)
        mean = torch.einsum("...hij,...jhs->...ihs", b, values)                    # (..., N, H, d)
        squares = self.predictive_covariance_spread_squares
        spread = None if squares else _old_centered_spread(b, values, mean)        # (..., N, H, d, d)
        if self.predictive_covariance_symmetric_pairs:
            C_att = self._symmetric_pair_blocks(TC, T_blocks, O_blocks, b, spread)
            c = source = pooled = None
        else:
            source = torch.einsum("...nhmsk,gtmk->...nhgst", TC, T_blocks)        # (..., N, H, H, d, d)
            c = torch.einsum("...hij,...gij->...hgij", b, b)
            c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", b, eye_h)
            pooled = torch.einsum("...hgij,...jhgst->...ihgst", c, source)        # (..., N, H, H, d, d)
            if spread is not None:
                pooled.diagonal(dim1=-4, dim2=-3).add_(spread.movedim(-3, -1))    # P_hh += S^h, in place
            lifted = torch.einsum("hks,...nhgst->...ngkt", O_hat, pooled)         # (..., N, H, K, d)
            lifted = lifted.reshape(*lifted.shape[:-2], H, d, d)                   # (..., N, g, l, k, t)
            rows = lifted.permute(nl + 2, *range(nl + 1), nl + 3, nl + 1, nl + 4)  # (l, ..., n, k, g, t)
            O_cols = O_blocks.permute(1, 0, 3, 2).reshape(H, H * d, d)            # (l, [g, t], m)
            C_att = ro._per_block_bmm(rows.reshape(H, -1, H * d), O_cols)          # (l, [..., n, k], m)
            C_att = C_att.view(H, *C.shape[:-3], d, d).movedim(0, -3)              # (..., N, H, d, d)
        if edges is not None:
            A_hat, X_hat, Y = ro.edge_links.edge_covariance_factors(edges, b, c, source, values, mean)
            inner = C_att + ro.edge_links.edge_correction_blocks(A_hat, X_hat, Y, H)
            C_att = (ro.edge_links.outer_gauge_blocks(
                edges, inner, ro.edge_links.edge_inner_rows(edges, pooled, A_hat, X_hat, Y))
                if edges.mode == "pure_gauge" else inner)
        if self.attention_residual:
            self_weight = b.diagonal(dim1=-2, dim2=-1).transpose(-1, -2)          # (..., N, H) beta_ii^h
            weighted = TC * self_weight[..., None, None, None]                    # (..., N, H, H, d, d)
            weighted = weighted.permute(nl + 2, nl + 1, nl + 3, *range(nl + 1), nl + 4)   # (l, h, s, ..., n, m)
            O_rows = O_blocks.permute(1, 2, 0, 3).reshape(H, d, H * d)            # (l, k, [h, s])
            cross = ro._per_block_bmm(O_rows, weighted.reshape(H, H * d, -1))      # (l, k, [..., n, m])
            cross = cross.view(H, d, *C.shape[:-3], d).movedim((0, 1), (-3, -2))   # (..., N, H, d, d) X_{i,l}
            if edges is not None:
                cross = ro.edge_links.edge_residual_cross_blocks(edges, A_hat, self_weight, TC, cross)
            C_att = C_att + C + cross + cross.transpose(-1, -2)
        stash = self._projection_stash
        if squares:
            lift = torch.einsum("...lak,hlks->...hlas", forward, O_blocks).flatten(-3, -2)
            layout = "blocks" if (block or stash is not None) else "diag"
            spread_out = _old_spread_squares(b, values, mean, lift, layout, d)    # (..., N, H, d, d) or (..., N, K)
        if block:
            blocks = forward @ C_att @ forward.transpose(-1, -2)                   # (..., N, H, d, d)
            blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
            if squares:
                blocks = blocks + spread_out
            result = torch.einsum("...hab,hg->...hagb", blocks, eye_h).reshape(*sigma.shape)   # (..., N, K, K)
            return result.to(sigma.dtype)
        result = ((forward @ C_att) * forward).sum(dim=-1)                         # (..., N, H, d) diag(U C U^T)
        if squares:
            result = result + (torch.diagonal(spread_out, dim1=-2, dim2=-1) if spread_out.dim() == result.dim() + 1
                               else spread_out.reshape(result.shape))
    return result.flatten(-2).to(sigma.dtype)


# Inputs.

def _inputs(H: int, d: int, N: int, batch: Tuple[int, ...], prior: str, block: bool, seed: int
            ) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=g, dtype=F64)
    K = H * d
    blocks = torch.eye(d, dtype=F64) + 0.3 * randn(*batch, N, H, d, d)
    if block:
        A = randn(*batch, N, H, d, d)
        sigma = torch.einsum("...hab,hg->...hagb", A @ A.mT + 0.2 * torch.eye(d, dtype=F64),
                             torch.eye(H, dtype=F64)).reshape(*batch, N, K, K)
    else:
        sigma = torch.rand(*batch, N, K, generator=g, dtype=F64) + 0.2
    log_prior = attention_log_prior(prior, N, N, n_heads=H, alibi_slope=0.5, dtype=F64)
    anchor = torch.rand(*batch, N, K, generator=g, dtype=F64) + 0.5
    return dict(mu=randn(*batch, N, K), sigma=sigma, logits=randn(*batch, H, N, N), blocks=blocks,
                inverse=torch.linalg.inv(blocks), log_prior=log_prior, receive=randn(H, K, d),
                anchor=torch.diag_embed(anchor) if block else anchor,
                weights=randn(*sigma.shape), T=randn(H, d, K), O=randn(H, K, d))


def _operators(H: int, d: int, squares: bool, symmetric: bool, checkpointed: bool, block: bool,
               dtype: torch.dtype, maps: Dict[str, torch.Tensor]) -> Any:
    ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=True,
                                    predictive_covariance="moment_matched", predictive_covariance_floor=1e-3)
    ops = ops.to(dtype)
    with torch.no_grad():
        ops.T.copy_(maps["T"].reshape(ops.T.shape))
        ops.O.copy_(maps["O"].reshape(ops.O.shape))
    ops.predictive_covariance_spread_squares = squares
    ops.predictive_covariance_symmetric_pairs = symmetric
    ops.predictive_covariance_checkpoint = checkpointed
    ops.block_covariance = block
    return ops


def _update_and_grads(ops: Any, x: Dict[str, torch.Tensor], dtype: torch.dtype, causal: bool,
                      oracle: bool) -> List[torch.Tensor]:
    if oracle:
        ops._predictive_covariance_update_blockwise = MethodType(_old_blockwise, ops)
    leaves = {k: x[k].detach().to(dtype, copy=True).requires_grad_(True)
              for k in ("mu", "sigma", "logits", "blocks", "inverse", "receive")}       # fresh leaves per run
    beta = torch.softmax(leaves["logits"] + x["log_prior"].to(dtype), dim=-1)
    frame = CanonicalFrameContext(leaves["blocks"], leaves["inverse"], block_diagonal=True)
    for p in ops.parameters():
        p.grad = None
    kwargs = {"causal_sources": True} if causal else {}
    out = ops.predictive_covariance_update(leaves["mu"], leaves["sigma"], beta, frame, leaves["receive"],
                                           floor_anchor=x["anchor"].to(dtype), **kwargs)
    (out * x["weights"].to(dtype)).sum().backward()
    if oracle:
        del ops._predictive_covariance_update_blockwise
    grads = [leaves[k].grad for k in ("mu", "sigma", "logits", "blocks", "inverse", "receive")]
    return [out.detach()] + grads + [None if p.grad is None else p.grad.clone() for p in (ops.T, ops.O)]


def _relative_error(got: List[torch.Tensor], ref: List[torch.Tensor]) -> float:
    worst = 0.0
    for a, b in zip(got, ref):
        if a is None or b is None:
            assert a is None and b is None
            continue
        scale = float(b.abs().max())
        worst = max(worst, float((a.double() - b.double()).abs().max()) / max(scale, 1e-300))
    return worst


_BUDGETS = {"one_tile": 1 << 30, "many_tiles": 2048}


@pytest.mark.parametrize("prior", ["causal", "causal_alibi"])
@pytest.mark.parametrize("squares", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("budget", ["one_tile", "many_tiles"])
@pytest.mark.parametrize("block", [False, True])
def test_blockwise_update_matches_the_pre_change_body_in_float64(prior: str, squares: bool, symmetric: bool,
                                                                  checkpointed: bool, budget: str,
                                                                  block: bool) -> None:
    H, d, N = 3, 2, 7
    x = _inputs(H, d, N, (2,), prior, block, seed=3)
    ro.set_spread_tile_bytes(_BUDGETS[budget])
    results = {}
    for name, causal, oracle in (("oracle", False, True), ("full", False, False), ("causal", True, False)):
        ops = _operators(H, d, squares, symmetric, checkpointed, block, F64, x)
        results[name] = _update_and_grads(ops, x, F64, causal, oracle)
    for name in ("full", "causal"):
        assert _relative_error(results[name], results["oracle"]) <= 1e-12, name


@pytest.mark.parametrize("prior", ["causal", "causal_alibi"])
@pytest.mark.parametrize("squares", [False, True])
@pytest.mark.parametrize("budget", ["one_tile", "many_tiles"])
def test_blockwise_update_matches_the_pre_change_body_in_float32(prior: str, squares: bool, budget: str) -> None:
    H, d, N = 3, 2, 7
    x = _inputs(H, d, N, (2,), prior, False, seed=8)
    ro.set_spread_tile_bytes(_BUDGETS[budget])
    ops = _operators(H, d, squares, False, False, False, F64, x)
    reference = _update_and_grads(ops, x, F64, False, True)
    for causal in (False, True):
        ops = _operators(H, d, squares, False, False, False, torch.float32, x)
        got = _update_and_grads(ops, x, torch.float32, causal, False)
        ops = _operators(H, d, squares, False, False, False, torch.float32, x)
        old = _update_and_grads(ops, x, torch.float32, False, True)
        # The new float32 path is as close to float64 as the old one, up to a small factor of rounding.
        new_error, old_error = _relative_error(got, reference), _relative_error(old, reference)
        assert new_error <= max(4.0 * old_error, 1e-5), (new_error, old_error)
        assert _relative_error(got, old) <= 1e-4


def test_blockwise_update_matches_without_a_batch_axis_and_with_four_heads() -> None:
    H, d, N = 4, 3, 9
    x = _inputs(H, d, N, (), "causal_alibi", False, seed=11)
    ro.set_spread_tile_bytes(4096)
    for squares in (False, True):
        ref = _update_and_grads(_operators(H, d, squares, False, True, False, F64, x), x, F64, False, True)
        got = _update_and_grads(_operators(H, d, squares, False, True, False, F64, x), x, F64, True, False)
        assert _relative_error(got, ref) <= 1e-12


@pytest.mark.parametrize("squares", [False, True])
def test_dense_route_spreads_match_the_pre_change_helpers(monkeypatch: pytest.MonkeyPatch, squares: bool) -> None:
    H, d, N = 3, 2, 7
    x = _inputs(H, d, N, (2,), "causal", True, seed=5)
    ro.set_spread_tile_bytes(2048)

    def run(causal: bool, oracle: bool) -> List[torch.Tensor]:
        ops = _operators(H, d, squares, False, False, False, F64, x)
        leaves = {k: x[k].clone().requires_grad_(True) for k in ("mu", "sigma", "logits", "blocks", "inverse")}
        beta = torch.softmax(leaves["logits"] + x["log_prior"], dim=-1)
        frame = CanonicalFrameContext(leaves["blocks"], leaves["inverse"], block_diagonal=True)
        with monkeypatch.context() as patch:
            if oracle:
                patch.setattr(ro, "_centered_spread", _old_centered_spread)
                patch.setattr(ro, "_spread_squares", _old_spread_squares)
            kwargs = {"causal_sources": True} if causal else {}
            out = ops._predictive_covariance_update_dense(leaves["mu"], leaves["sigma"], beta, frame, None,
                                                          **kwargs)
        (out * x["weights"]).sum().backward()
        return [out.detach()] + [leaves[k].grad for k in leaves] + [ops.T.grad, ops.O.grad]

    reference = run(False, True)
    for causal in (False, True):
        assert _relative_error(run(causal, False), reference) <= 1e-12


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("budget", [1, 700, 1 << 30])
@pytest.mark.parametrize("layout", ["centered", "diag", "blocks", "full"])
def test_spread_helpers_match_the_pre_change_helpers(batch: Tuple[int, ...], budget: int, layout: str) -> None:
    g = torch.Generator().manual_seed(len(batch) + budget % 7)
    H, d, N, A, block = 2, 3, 7, 4, 2
    ro.set_spread_tile_bytes(budget)
    logits = torch.randn(*batch, H, N, N, generator=g, dtype=F64)
    log_prior = attention_log_prior("causal_alibi", N, N, n_heads=H, alibi_slope=0.5, dtype=F64)
    values = torch.randn(*batch, N, H, d, generator=g, dtype=F64)
    lift = torch.randn(*batch, N, H, A, d, generator=g, dtype=F64)

    def run(new: bool, causal: bool) -> List[torch.Tensor]:
        leaves = [t.clone().requires_grad_(True) for t in (logits, values, lift)]
        b = torch.softmax(leaves[0] + log_prior, dim=-1)
        mean = torch.einsum("...hij,...jhs->...ihs", b, leaves[1])
        if layout == "centered":
            out = (ro._centered_spread(b, leaves[1], mean, causal) if new
                   else _old_centered_spread(b, leaves[1], mean))
        else:
            out = (ro._spread_squares(b, leaves[1], mean, leaves[2], layout, block, causal) if new
                   else _old_spread_squares(b, leaves[1], mean, leaves[2], layout, block))
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=F64)
        (out * weights).sum().backward()
        return [out.detach()] + [leaf.grad for leaf in leaves]

    reference = run(False, False)
    for causal in (False, True):
        got = run(True, causal)
        if layout == "centered":
            got[-1], reference[-1] = None, None                    # the centered spread does not read the lift
        assert _relative_error(got, reference) <= 1e-12


# Item 16: the tile budget bounds the whole tile, batch included, and the checkpoint decision reads it too.

@pytest.mark.parametrize("squares", [False, True])
def test_spread_tiles_count_the_batch_axis(monkeypatch: pytest.MonkeyPatch, squares: bool) -> None:
    g = torch.Generator().manual_seed(2)
    B, H, d, N, A = 8, 2, 3, 9, 6
    values = torch.randn(B, N, H, d, generator=g, dtype=F64, requires_grad=True)
    b = torch.softmax(torch.randn(B, H, N, N, generator=g, dtype=F64), dim=-1)
    mean = torch.einsum("...hij,...jhs->...ihs", b, values)
    lift = torch.randn(B, N, H, A, d, generator=g, dtype=F64)
    width = A if squares else d
    per_row = B * N * H * width * 8                                    # bytes of the tile transient per query row
    ro.set_spread_tile_bytes(3 * per_row)
    rows: List[int] = []
    calls: List[int] = []
    real_tile = ro._spread_squares_tile if squares else ro._centered_spread_tile
    real_checkpoint = ro.checkpoint

    def tile(*args: Any) -> torch.Tensor:
        rows.append(args[2].shape[-3])                                 # the tile's query rows
        return real_tile(*args)

    def spy(fn: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        return real_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(ro, "_spread_squares_tile" if squares else "_centered_spread_tile", tile)
    monkeypatch.setattr(ro, "checkpoint", spy)
    if squares:
        ro._spread_squares(b, values, mean, lift, "diag")
    else:
        ro._centered_spread(b, values, mean)
    assert rows == [3, 3, 3]                                           # the old per-sequence sizing gave one tile
    assert max(rows) * per_row <= ro.spread_tile_bytes() and len(calls) == 3
    # One tile under the budget: checkpointed only when the whole transient, batch included, exceeds the floor.
    rows.clear(), calls.clear()
    ro.set_spread_tile_bytes(1 << 30)
    monkeypatch.setattr(ro, "_SPREAD_CHECKPOINT_BYTES", N * per_row - 1)
    if squares:
        ro._spread_squares(b, values, mean, lift, "diag")
    else:
        ro._centered_spread(b, values, mean)
    assert rows == [N] and calls == [1]
    rows.clear(), calls.clear()
    monkeypatch.setattr(ro, "_SPREAD_CHECKPOINT_BYTES", N * per_row)
    if squares:
        ro._spread_squares(b, values, mean, lift, "diag")
    else:
        ro._centered_spread(b, values, mean)
    assert rows == [N] and calls == []


# Item 18: the causal tiles read only their causal sources, and the readout derives the flag from its prior.

def test_causal_tiles_read_only_their_causal_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(4)
    H, d, N = 2, 3, 9
    values = torch.randn(1, N, H, d, generator=g, dtype=F64)
    log_prior = attention_log_prior("causal", N, N, dtype=F64)
    b = torch.softmax(torch.randn(1, H, N, N, generator=g, dtype=F64) + log_prior, dim=-1)
    mean = torch.einsum("...hij,...jhs->...ihs", b, values)
    seen: List[Tuple[int, int]] = []
    real = ro._centered_spread_tile

    def tile(b_t: torch.Tensor, values_t: torch.Tensor, mean_t: torch.Tensor) -> torch.Tensor:
        seen.append((b_t.shape[-2], values_t.shape[-2]))                # (rows, sources)
        return real(b_t, values_t, mean_t)

    monkeypatch.setattr(ro, "_centered_spread_tile", tile)
    ro.set_spread_tile_bytes(1 << 30)
    full = ro._centered_spread(b, values, mean)
    assert seen == [(N, N)]
    seen.clear()
    causal = ro._centered_spread(b, values, mean, causal=True)
    assert seen == [(3, 3), (3, 6), (3, 9)]                             # at least four tiles' worth of rows, ceil(9/4)
    assert float((causal - full).abs().max()) <= 1e-15 * float(full.abs().max())


def test_strictly_causal_routing_prior_reads_the_prior_tensor() -> None:
    N, H = 6, 3
    for name in ("causal", "causal_noself", "causal_alibi", "causal_alibi_noself"):
        assert ro.strictly_causal_routing_prior(attention_log_prior(name, N, N, n_heads=H, alibi_slope=0.5))
    assert ro.strictly_causal_routing_prior(attention_log_prior("causal_windowed", N, N, window=2))
    assert not ro.strictly_causal_routing_prior(None)
    assert not ro.strictly_causal_routing_prior(torch.zeros(N, N))
    assert not ro.strictly_causal_routing_prior(torch.zeros(N, N + 1))
    assert not ro.strictly_causal_routing_prior(torch.empty(N, N, device="meta"))
    prior = attention_log_prior("causal", N, N)
    assert ro.strictly_causal_routing_prior(prior)
    prior[1, 4] = 0.0                                                   # in place: the version counter moves
    assert not ro.strictly_causal_routing_prior(prior)
    with torch.inference_mode():
        frozen = attention_log_prior("causal", N, N)
    assert ro.strictly_causal_routing_prior(frozen)


@pytest.mark.parametrize("prior", [None, "causal", "causal_alibi"])
def test_readout_passes_the_causal_flag_from_its_prior(monkeypatch: pytest.MonkeyPatch, prior: Optional[str]) -> None:
    from tests.test_blockwise_moment_matched_20261004 import _case
    ops, mu, sigma, _, frame, _ = _case(2, 2, False, (2,), F64, seed=1)
    N = mu.shape[-2]
    log_prior = None if prior is None else attention_log_prior(prior, N, N, n_heads=2, alibi_slope=0.5, dtype=F64)
    seen: List[bool] = []
    real = ops.predictive_covariance_update

    def spy(*args: Any, causal_sources: bool = False, **kwargs: Any) -> torch.Tensor:
        seen.append(causal_sources)
        return real(*args, causal_sources=causal_sources, **kwargs)

    monkeypatch.setattr(ops, "predictive_covariance_update", spy)
    relational_gradients.relational_predictive_readout(mu, sigma, frame, ops, log_prior=log_prior,
                                                       return_covariance=True, floor_anchor=sigma)
    assert seen == [prior is not None]
