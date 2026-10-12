r"""Block MLP, LayerNorm pushforward, prototype agents and the Triton kernel (audit 2026-10-10 round 2, agent 4).

Items 11 (radial second-order variance of the LayerNorm pushforward), 29 (the block MLP's fp32 island under
``amp_dtype``), 50 (rank-two full-layout pushforward), 51 (covariance transport in reference coordinates), 54 (prototype
label staleness), 56 (two-level accumulation in the Triton energy, by emulation) and 57 (the no-sync float64 select).
The EM gradients of item 15 are checked against finite differences in ``tests/test_cavi_em_20261010.py``, together
with the evidence fixed point here. Float64 and tiny K unless stated; CPU only.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

import vfe3.train as T
from tests.profiles import tiny_config
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.families.gaussian import head_block_projection
from vfe3.geometry.norms import AffineLayerNorm
from vfe3.model import block_mlp as bm
from vfe3.model import prototype_agents as pa
from vfe3.model.block import _reference_coordinate_block
from vfe3.model.block_mlp import BlockMLP, PrototypeAgentBlockMLP
from vfe3.model.model import VFEModel

F64 = torch.float64
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _norm(k: int, eps: float, seed: int) -> AffineLayerNorm:
    g = torch.Generator().manual_seed(seed)
    norm = AffineLayerNorm(k, eps=eps).double()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.3 * torch.randn(k, generator=g, dtype=F64))
        norm.bias.copy_(0.2 * torch.randn(k, generator=g, dtype=F64))
    return norm


def _spd(shape: Tuple[int, ...], k: int, scale: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*shape, k, k, generator=g, dtype=F64)
    return scale * (A @ A.transpose(-1, -2) / k + 0.2 * torch.eye(k, dtype=F64))


def _outer(rho: torch.Tensor, h: torch.Tensor) -> torch.Tensor:          # rho (..., 1), h (..., K) -> (..., K, K)
    return rho.unsqueeze(-1) * h.unsqueeze(-1) * h.unsqueeze(-2)


def _first_order(norm: AffineLayerNorm, u: torch.Tensor, S: torch.Tensor, floor: float) -> torch.Tensor:
    r"""The pre-fix dense route :math:`J\Sigma J^\top + fD\Sigma D` (no radial term)."""
    lin = bm.block_norm_linearization(norm, u)
    return bm._delta_covariance(lin.jacobian, S, floor, lin.scale)


def _cfg(**over: Any):
    values: Dict[str, Any] = dict(embed_dim=4, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=1,
                                  attention_mode="projected_predictive", predictive_covariance="moment_matched",
                                  content_link_chart="exponential", norm_type_block="layernorm",
                                  layernorm_affine=True, use_block_mlp=True, block_mlp_mode="coordinate",
                                  block_mlp_expansion=2)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def _model(cfg, seed: int = 0, double: bool = True) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    model = model.double() if double else model
    g = torch.Generator().manual_seed(seed + 50)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad or name.endswith("phi_embed"):
                p.add_((0.3 if name.endswith("phi_embed") else 0.2) * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _tokens(seed: int = 3) -> torch.Tensor:
    return torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------------------------
# Item 11: the radial second-order variance
# ----------------------------------------------------------------------------------------------

def test_radial_variance_matches_monte_carlo() -> None:
    r"""The variance of the normalized output along :math:`\hat z` (the functional :math:`a = \gamma^{-1}\circ\hat z`
    on :math:`y = \mathrm{LN}(u)`, :math:`u \sim \mathcal N(\mu, \Sigma)`) against 400k float64 samples at
    :math:`\Sigma \approx 10^{-3}v`: the pushforward with the radial term agrees to 3%, the first-order one is low by
    more than 100x (the defect: :math:`J_{\mathrm{LN}}` has eigenvalue :math:`\epsilon/(v + \epsilon)` on :math:`z`)."""
    K, eps = 6, 1e-5
    norm = _norm(K, eps, 1)
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(K, generator=g, dtype=F64)
    S = _spd((), K, 1e-3, 3)
    y = norm(mu + torch.randn(400_000, K, generator=g, dtype=F64) @ torch.linalg.cholesky(S).T, None)
    c = mu - mu.mean()
    a = (c / c.norm()) / norm.weight.detach()                               # (K,) gamma^{-1} o zhat
    mc = float((y.detach() @ a).var())
    pushed = bm.block_norm_covariance_pushforward(norm, mu, S, 0.0).detach()
    first = _first_order(norm, mu, S, 0.0).detach()
    predicted, linear = float(a @ pushed @ a), float(a @ first @ a)
    print(f"radial variance: Monte Carlo {mc:.4e}, with the radial term {predicted:.4e}, first order {linear:.4e}")
    assert abs(predicted - mc) / mc < 0.03
    assert linear < 1e-2 * mc


def test_two_token_kl_no_longer_scales_as_the_inverse_floor() -> None:
    r"""Two tokens at K = 16 on ``gaussian_full``: with the radial term the pair KL converges as the floor
    :math:`f \to 0`; the first-order pushforward's KL grows as :math:`1/f` (the audit's 1.45e4, 1.42e6, 1.42e8)."""
    K = 16
    norm = _norm(K, 1e-6, 4)
    g = torch.Generator().manual_seed(5)
    u = torch.randn(2, K, generator=g, dtype=F64)
    S = _spd((2,), K, 1e-2, 6)
    y = norm(u, None).detach()

    def kl(P: torch.Tensor) -> float:
        d = y[1] - y[0]
        return float(0.5 * (torch.trace(torch.linalg.solve(P[1], P[0])) + d @ torch.linalg.solve(P[1], d) - K
                            + torch.logdet(P[1]) - torch.logdet(P[0])))

    pushed = {f: kl(bm.block_norm_covariance_pushforward(norm, u, S, f).detach()) for f in (1e-2, 1e-4, 1e-6)}
    first = {f: kl(_first_order(norm, u, S, f).detach()) for f in (1e-2, 1e-4, 1e-6)}
    print(f"pair KL with the radial term {pushed}, first order {first}")
    assert pushed[1e-6] / pushed[1e-4] < 1.05
    assert first[1e-6] / first[1e-4] > 50.0


def test_radial_term_is_applied_in_every_layout() -> None:
    r"""Diagonal closed form = diagonal of the full form; head blocks = projection of the full form; the delta_full
    composite carries :math:`\rho(J_{\mathrm{MLP}}g)(J_{\mathrm{MLP}}g)^\top` on both layouts."""
    K = 4
    norm = _norm(K, 1e-6, 7)
    g = torch.Generator().manual_seed(8)
    u = torch.randn(2, 3, K, generator=g, dtype=F64)
    sigma = 0.2 + torch.rand(2, 3, K, generator=g, dtype=F64)
    full = bm.block_norm_covariance_pushforward(norm, u, torch.diag_embed(sigma), 1e-3)
    diag = bm.block_norm_covariance_pushforward(norm, u, sigma, 1e-3)
    torch.testing.assert_close(diag, torch.diagonal(full, dim1=-2, dim2=-1), rtol=1e-12, atol=0)
    S = _spd((2, 3), K, 0.5, 9)
    blocks = bm.block_norm_covariance_pushforward(norm, u, S, 1e-3, head_dims=(2, 2))
    assert torch.equal(blocks, head_block_projection(bm.block_norm_covariance_pushforward(norm, u, S, 1e-3), [2, 2]))
    mlp = BlockMLP(K, 2, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=1e-3).double().eval()
    lin = bm.block_norm_linearization(norm, u)
    x = norm(u, None)
    for cov in (S, sigma):
        radial = bm.block_norm_radial_term(norm, u, cov)
        base = dict(input_jacobian=lin.jacobian, input_scale=lin.scale)
        with_term = mlp.forward_moments(x, cov, **base, input_radial=radial)
        without = mlp.forward_moments(x, cov, **base)
        _, J_mlp = mlp._coordinate_update_with_jacobian(x)
        h = (J_mlp @ radial.direction.unsqueeze(-1)).squeeze(-1)
        added = _outer(radial.rho, h)
        added = added if cov.dim() == x.dim() + 1 else torch.diagonal(added, dim1=-2, dim2=-1)
        torch.testing.assert_close(with_term.sigma - without.sigma, added, rtol=1e-9, atol=1e-15)
    with pytest.raises(ValueError, match="input_radial"):
        mlp.forward_moments(x, S, input_radial=bm.block_norm_radial_term(norm, u, S))


# ----------------------------------------------------------------------------------------------
# Item 50: rank-two full-layout pushforward; item 57: no-sync select
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("K", [6, 128])
def test_rank_two_pushforward_equals_the_dense_route(K: int) -> None:
    r""":math:`D(P\Sigma P + f\Sigma)D + \rho gg^\top` against the dense Gram route :math:`(JL)(JL)^\top + fD\Sigma D` plus the
    same radial term, float64, to rounding; the float32 result is certified SPD."""
    norm = _norm(K, 1e-6, 10)
    g = torch.Generator().manual_seed(11)
    u = torch.randn(3, K, generator=g, dtype=F64)
    S = _spd((3,), K, 0.5, 12)
    got = bm.block_norm_covariance_pushforward(norm, u, S, 1e-3).detach()
    radial = bm.block_norm_radial_term(norm, u, S)
    dense = (_first_order(norm, u, S, 1e-3) + _outer(radial.rho, radial.direction)).detach()
    error = float((got - dense).abs().max() / dense.abs().max())
    print(f"K={K}: rank-two against dense, relative {error:.1e}")
    assert error < 1e-13
    low = bm.block_norm_covariance_pushforward(norm.float(), u.float(), S.float(), 1e-3).detach()
    assert low.dtype == torch.float32 and bool((torch.linalg.cholesky_ex(low).info == 0).all())
    assert float((low.double() - got).abs().max() / got.abs().max()) < 1e-4


def test_head_block_certificate_factors_each_block() -> None:
    S = torch.eye(4, dtype=torch.float32).expand(2, 4, 4).clone()
    S[1, 2:, 2:] = torch.tensor([[1.0, 2.0], [2.0, 1.0]])                    # indefinite second block of token 1
    S[1, 0, 3] = S[1, 3, 0] = 5.0                                           # an off-block entry the blocks ignore
    assert bm._cholesky_failures(S, (2, 2)).tolist() == [False, True]
    S[1, 2:, 2:] = torch.eye(2)
    assert bm._cholesky_failures(S, (2, 2)).tolist() == [False, False]
    assert bm._cholesky_failures(S, None).tolist() == [False, True]          # the whole matrix is indefinite


def _old_diagonal_pushforward(u: torch.Tensor, sigma: torch.Tensor, eps: float, floor: float) -> torch.Tensor:
    r"""The pre-change dispatch: the float64 recompute only after a host read of the failure mask."""
    pushed = bm._layernorm_diagonal_pushforward_at(u, sigma, eps, floor, None)
    failed = pushed <= 0
    if bool(failed.any()):
        high = bm._layernorm_diagonal_pushforward_at(u.double(), sigma.double(), eps, floor, None)
        pushed = torch.where(failed, high.float(), pushed)
    return pushed


@pytest.mark.parametrize("K,floor", [(2, 1e-12), (8, 1e-4)])
def test_no_sync_select_equals_the_old_values(K: int, floor: float, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Saturated (K = 2, failures present) and healthy inputs: bitwise the old values, with no ``bool`` read of a
    tensor inside the call."""
    g = torch.Generator().manual_seed(13)
    u = torch.randn(4096, K, generator=g)
    sigma = 0.1 + torch.rand(4096, K, generator=g)
    want = _old_diagonal_pushforward(u, sigma, 1e-6, floor)
    if K == 2:
        assert bool((bm._layernorm_diagonal_pushforward_at(u, sigma, 1e-6, floor, None) <= 0).any())

    def no_host_read(self: torch.Tensor) -> bool:
        raise AssertionError("host read of a tensor inside layernorm_diagonal_pushforward")

    monkeypatch.setattr(torch.Tensor, "__bool__", no_host_read)
    got = bm.layernorm_diagonal_pushforward(u, sigma, eps=1e-6, covariance_floor=floor)
    monkeypatch.undo()
    assert torch.equal(got, want)


# ----------------------------------------------------------------------------------------------
# Item 51: covariance transport in reference coordinates (no norm runs before the decoder)
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("coordinates", ["local", "reference"])
def test_decoder_reads_the_stack_covariance_under_pushforward(coordinates: str,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    r"""With no norm between the stack and the decoder, the Gaussian decoder receives the stack output's
    covariance object under ``block_norm_covariance='pushforward'``."""
    model = _model(_cfg(family="gaussian_full", decode_mode="full_chunked", block_norm_covariance="pushforward",
                        block_coordinates=coordinates, decode_coordinates=coordinates))
    seen: List[torch.Tensor] = []
    decode = model._decode_belief_with_context

    def spy(mu: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.append(sigma)
        return decode(mu, sigma, **kwargs)

    monkeypatch.setattr(model, "_decode_belief_with_context", spy)
    capture: dict = {}
    with torch.no_grad():
        belief, _ = model.forward_beliefs(_tokens(), return_logits=True, capture=capture)
    assert len(seen) == 1 and seen[0] is belief.sigma is capture["out"].sigma


def _frame_blocks(B: int, N: int, G: int, d: int, scale: float, seed: int) -> torch.Tensor:   # (B, N, G, d, d)
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.matrix_exp(scale * torch.randn(B, N, G, d, d, generator=g, dtype=F64))


def _dense_blocks(blocks: torch.Tensor) -> torch.Tensor:                   # (B, N, K, K)
    G, d = blocks.shape[-3], blocks.shape[-1]
    out = torch.zeros(*blocks.shape[:-3], G * d, G * d, dtype=blocks.dtype)
    for h in range(G):
        out[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = blocks[..., h, :, :]
    return out


@pytest.mark.parametrize("contract", ["pushforward", "delta_full"])
def test_reference_block_covariance_transport_is_covariant(contract: str) -> None:
    r"""Under the passive action :math:`(\mu, \Sigma, U) \to (g\mu, g\Sigma g^\top, gU)`,
    :math:`g \in \prod_b GL(d_b)`, the reference block maps :math:`\Sigma' \to g\Sigma'g^\top` and :math:`\mu' \to g\mu'`;
    the covariance moved, and the local pushforward (a negative control) is not covariant."""
    B, N, G, d = 2, 3, 2, 2
    K = G * d
    norm = _norm(K, 1e-6, 14)
    torch.manual_seed(15)
    mlp = BlockMLP(K, 2, "gelu", 0.0, covariance_contract="delta_full" if contract == "delta_full" else "passthrough",
                   covariance_floor=1e-3).double().eval()
    blocks, g_blocks = _frame_blocks(B, N, G, d, 0.5, 16), _frame_blocks(B, N, G, d, 0.6, 17)
    gd = _dense_blocks(g_blocks)
    mu = torch.randn(B, N, K, generator=torch.Generator().manual_seed(18), dtype=F64)
    S = _spd((B, N), K, 0.5, 19)

    def run(frame_blocks: torch.Tensor, m: torch.Tensor, cov: torch.Tensor) -> BeliefState:
        frame = CanonicalFrameContext(frame_blocks, torch.linalg.inv(frame_blocks), block_diagonal=True)
        belief = BeliefState(mu=m, sigma=cov, phi=torch.zeros(B, N, 1, dtype=F64))
        return _reference_coordinate_block(belief, block_norm=norm, block_mlp=mlp, frame=frame, capture=None,
                                           hidden_bias=None, push_norm=contract == "pushforward",
                                           covariance_floor=1e-3)

    base = run(blocks, mu, S)
    moved = run(g_blocks @ blocks, (gd @ mu.unsqueeze(-1)).squeeze(-1), gd @ S @ gd.transpose(-1, -2))
    torch.testing.assert_close(moved.mu, (gd @ base.mu.unsqueeze(-1)).squeeze(-1), rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(moved.sigma, gd @ base.sigma @ gd.transpose(-1, -2), rtol=1e-10, atol=1e-11)
    assert float((base.sigma - S).abs().max()) > 1e-3                      # the covariance was transported
    local = bm.block_norm_covariance_pushforward(norm, mu, S, 1e-3)
    local_moved = bm.block_norm_covariance_pushforward(norm, (gd @ mu.unsqueeze(-1)).squeeze(-1),
                                                       gd @ S @ gd.transpose(-1, -2), 1e-3)
    assert float((local_moved - gd @ local @ gd.transpose(-1, -2)).abs().max()) > 1e-3


def test_reference_block_at_identity_frames_equals_the_local_pushforward() -> None:
    model = _model(_cfg(family="gaussian_full", decode_mode="full_chunked", block_norm_covariance="pushforward"))
    norm, mlp = model.block_norm, model.block_mlps[0]
    B, N, K = 2, 3, 4
    mu = torch.randn(B, N, K, generator=torch.Generator().manual_seed(20), dtype=F64)
    S = _spd((B, N), K, 0.5, 21)
    eye = torch.eye(K, dtype=F64).expand(B, N, K, K)
    belief = BeliefState(mu=mu, sigma=S, phi=torch.zeros(B, N, 1, dtype=F64))
    with torch.no_grad():
        ref = _reference_coordinate_block(belief, block_norm=norm, block_mlp=mlp,
                                          frame=CanonicalFrameContext(eye, eye), capture=None, hidden_bias=None,
                                          push_norm=True, covariance_floor=model.cfg.block_mlp_covariance_floor)
        local = bm.block_norm_covariance_pushforward(norm, mu, S, model.cfg.block_mlp_covariance_floor)
    torch.testing.assert_close(ref.sigma, local, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("over", [dict(block_norm_covariance="pushforward"),
                                  dict(block_mlp_covariance="delta_full")])
def test_reference_models_with_covariance_transport_run(over: dict) -> None:
    cfg = _cfg(family="gaussian_full", decode_mode="full_chunked", block_coordinates="reference", n_layers=2, **over)
    if "block_norm_covariance" in over:
        assert "block_norm_covariance" not in cfg.inactive_fields()
    model = _model(cfg)
    tokens = _tokens()
    _, loss, _ = model(tokens, _tokens(4))
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="block_norm_covariance='pushforward'"):
        _cfg(block_coordinates="reference", block_norm_covariance="pushforward")          # gaussian_diagonal
    with pytest.raises(ValueError, match="block_mlp_covariance"):
        _cfg(block_coordinates="reference", block_mlp_covariance="delta_full")            # gaussian_diagonal


def test_delta_full_marks_the_norm_pushforward_inert() -> None:
    base = dict(family="gaussian_full", decode_mode="full_chunked", block_mlp_covariance="delta_full",
                block_norm_covariance="pushforward")
    assert "block_norm_covariance" in _cfg(**base).inactive_fields()


# ----------------------------------------------------------------------------------------------
# Item 29: the block MLP's fp32 island under amp_dtype
# ----------------------------------------------------------------------------------------------

def _fc1_dtypes(model: VFEModel, tokens: torch.Tensor) -> List[torch.dtype]:
    dtypes: List[torch.dtype] = []
    handles = [mlp.fc1.register_forward_hook(lambda _m, _i, out: dtypes.append(out.dtype)) for mlp in model.block_mlps]
    with torch.no_grad():
        model(tokens, tokens)
    for handle in handles:
        handle.remove()
    return dtypes


@pytest.mark.parametrize("coordinates", ["local", "reference"])
def test_block_mlp_runs_in_fp32_under_amp(coordinates: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Under ``amp_dtype`` the block MLP's mean path runs with autocast disabled on a float32 input (hardcoded
    2026-10-12), in local and reference coordinates."""
    model = _model(_cfg(amp_dtype="bf16", block_coordinates=coordinates, n_layers=2), double=False)
    tokens = _tokens()
    assert set(_fc1_dtypes(model, tokens)) == {torch.float32}
    inputs: List[Tuple[torch.dtype, bool]] = []
    original = BlockMLP.forward_moments

    def spy(self: BlockMLP, mu: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> Any:
        inputs.append((mu.dtype, torch.is_autocast_enabled("cpu")))
        return original(self, mu, sigma, **kwargs)

    monkeypatch.setattr(BlockMLP, "forward_moments", spy)
    with torch.no_grad():
        model.forward_beliefs(tokens)
    assert inputs and all(dtype == torch.float32 and not enabled for dtype, enabled in inputs)


# ----------------------------------------------------------------------------------------------
# Item 15: the evidence fixed point; item 54: the prototype label staleness
# ----------------------------------------------------------------------------------------------

def test_exact_em_fixed_point_is_the_evidence_maximum() -> None:
    r"""One template (K = 1, M = 1) and working beliefs :math:`\mathcal N(\xi_i, c)`: ascending :math:`\sum_iQ_i` with the
    posterior refreshed each step reaches :math:`\lambda = V - c`, the evidence maximum (:math:`V` the variance of
    :math:`\xi`); the pre-fix pairing's fixed point was :math:`V + c`."""
    mlp = PrototypeAgentBlockMLP(1, 1, 0.0, inference="exact").double()
    g = torch.Generator().manual_seed(22)
    xi = 2.0 + 1.5 * torch.randn(64, 1, generator=g, dtype=F64)
    c = torch.full((64, 1), 0.4, dtype=F64)
    optimizer = torch.optim.SGD([mlp.nu, mlp.log_lambda], lr=0.5)
    for _ in range(400):
        rows: List[torch.Tensor] = []
        mlp.forward_moments(xi, c, em_rows=rows)
        optimizer.zero_grad()
        (-rows[0].mean()).backward()
        optimizer.step()
    V = float(xi.var(unbiased=False))
    lam, nu = float(mlp.log_lambda.exp()), float(mlp.nu)
    assert abs(nu - float(xi.mean())) < 1e-6 and abs(lam - (V - 0.4)) < 1e-6, (lam, V)


def _staleness_reference(mlp: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""Per-token TV of ``tests/test_cavi_em_20261010.py::_prototype_staleness`` (which reports its maximum)."""
    post = mlp.posterior(mu, sigma, need_covariance=True)
    layout = pa.covariance_layout(mu, sigma)
    relabeled = torch.softmax(pa.template_kl_logits(post.xi_plus, pa.covariance_diagonal(post.C_plus, layout), mlp.nu,
                                                    mlp.log_lambda, mlp.log_pi), dim=-1)
    return 0.5 * (relabeled - post.responsibilities).abs().sum(dim=-1)


@pytest.mark.parametrize("cavi_steps", [1, 4])
@pytest.mark.parametrize("layout", ["diagonal", "dense"])
def test_prototype_label_staleness_statistic(cavi_steps: int, layout: str) -> None:
    K = 4
    mlp = PrototypeAgentBlockMLP(K, 2, 0.0, inference="mean_field", cavi_steps=cavi_steps).double()
    g = torch.Generator().manual_seed(23)
    with torch.no_grad():
        mlp.nu.copy_(1.5 * torch.randn(mlp.nu.shape, generator=g, dtype=F64))
        mlp.log_lambda.fill_(math.log(0.5))
    mu = 1.5 * torch.randn(6, K, generator=g, dtype=F64)
    sigma = (0.5 + torch.rand(6, K, generator=g, dtype=F64) if layout == "diagonal" else _spd((6,), K, 1.0, 24))
    with torch.no_grad():
        want = _staleness_reference(mlp, mu, sigma)
        _, _, got = mlp.weights_gain_and_staleness(mu, sigma)
        stats = mlp.statistics(mu, sigma)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-14)
    assert stats["prototype_label_staleness"] == pytest.approx(float(want.mean()), rel=1e-12, abs=1e-15)
    assert float(want.max()) > 1e-3                                         # a nonzero staleness is measured
    exact = PrototypeAgentBlockMLP(K, 2, 0.0, inference="exact").double()
    assert math.isnan(exact.statistics(mu, sigma)["prototype_label_staleness"])


@pytest.mark.parametrize("inference", ["mean_field", "exact"])
def test_label_staleness_reaches_the_template_rows(inference: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(_cfg(block_mlp_mode="prototype_agents", block_mlp_inference=inference, n_layers=2,
                        attention_residual=False), double=False)
    seen: List[Tuple[PrototypeAgentBlockMLP, torch.Tensor, torch.Tensor]] = []
    original = PrototypeAgentBlockMLP.forward_moments

    def spy(self: PrototypeAgentBlockMLP, mu: torch.Tensor, sigma: torch.Tensor, **kwargs: Any) -> Any:
        seen.append((self, mu.detach(), sigma.detach()))
        return original(self, mu, sigma, **kwargs)

    monkeypatch.setattr(PrototypeAgentBlockMLP, "forward_moments", spy)
    windows = torch.randint(0, 9, (3, 5), generator=torch.Generator().manual_seed(25))
    features: dict = {}
    T._residual_and_attention_rows(model, windows, 5, torch.device("cpu"), features_out=features)
    _, rows = T._feature_rows(model, features, 5)
    assert len(rows) == model.cfg.n_layers and all(set(row) == set(T.PROTOTYPE_TEMPLATE_FIELDS) for row in rows)
    for row in rows:
        layer_inputs = [(mu, sigma) for mlp, mu, sigma in seen if mlp is model.block_mlps[row["layer"]]]
        values = torch.cat([mlp_stale.reshape(-1) for mlp_stale in (
            model.block_mlps[row["layer"]].weights_gain_and_staleness(mu, sigma)[2] for mu, sigma in layer_inputs)])
        if inference == "exact":
            assert math.isnan(row["label_staleness"])
        else:
            assert row["label_staleness"] == pytest.approx(float(values.mean()), rel=1e-9)


# ----------------------------------------------------------------------------------------------
# Item 56: two-level accumulation of the Triton energy (emulated: triton is not importable under the CPU interpreter)
# ----------------------------------------------------------------------------------------------

def _energy(mq: np.ndarray, sq: np.ndarray, mv: np.ndarray, sv: np.ndarray, block: int) -> np.ndarray:
    r"""The kernel's float32 order of operations for :math:`E_{iv}`: running sums over k of :math:`d^2/s` and
    :math:`\log s`, sequential (``block = 0``) or in partial sums of ``block`` features."""
    S, K = mq.shape
    Q = np.zeros((S, mv.shape[0]), np.float32)
    L = np.zeros_like(Q)
    starts = [0] if block == 0 else list(range(0, K, block))
    for k0 in starts:
        q = np.zeros_like(Q)
        lb = np.zeros_like(Q)
        for k in range(k0, K if block == 0 else min(k0 + block, K)):
            d = mq[:, k, None] - mv[None, :, k]
            s = sq[:, k, None] + sv[None, :, k]
            q += d * d / s
            lb += np.log(s)
        Q += q
        L += lb
    return Q + L


def _row_ce(E: np.ndarray, targets: np.ndarray, tau: float) -> np.ndarray:
    logits = -(0.5 / tau) * E.astype(np.float64)
    lse = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
    return lse - logits[np.arange(len(targets)), targets]


def test_two_level_energy_accumulation_cuts_the_row_error() -> None:
    r"""At K = 128 and 512 and the live decode temperature 0.1 (float32 emulation of the kernel's sums, fp64
    reference) the per-row cross-entropy error of blocks of 16 is at most half the sequential sum's (about a third
    measured; the statistical accumulation error falls by about :math:`\sqrt{16}`, and the per-term rounding stays);
    the kernel defines the same block size."""
    rng = np.random.default_rng(26)
    errors = {}
    for K in (128, 512):
        S, V, tau = 32, 192, 0.1
        mq = rng.normal(0.0, 1.0, (S, K)).astype(np.float32)
        sq = np.exp(rng.normal(0.0, 0.5, (S, K))).astype(np.float32)
        mv = rng.normal(0.0, 1.0, (V, K)).astype(np.float32)
        sv = np.exp(rng.normal(0.0, 0.5, (V, K))).astype(np.float32)
        targets = rng.integers(0, V, S)
        exact = ((mq[:, None, :].astype(np.float64) - mv[None].astype(np.float64)) ** 2
                 / (sq[:, None, :].astype(np.float64) + sv[None]) + np.log(sq[:, None, :].astype(np.float64) + sv[None])
                 ).sum(-1)
        ce = _row_ce(exact, targets, tau)
        errors[K] = {block: float(np.abs(_row_ce(_energy(mq, sq, mv, sv, block), targets, tau) - ce).max())
                     for block in (0, 16)}
    print(f"per-row CE error, sequential (0) against blocks of 16: {errors}")
    assert all(e[16] < 0.5 * e[0] for e in errors.values())
    source = (REPO / "vfe3" / "kernels" / "_expected_likelihood_triton.py").read_text(encoding="utf-8")
    assert re.search(r"_KB: tl\.constexpr = tl\.constexpr\(16\)", source)
    assert source.count("E = _energy_tile(MQ, SQ, MVT, SVT") == 3            # one energy for the three kernels
