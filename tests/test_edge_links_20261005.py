r"""Edge-dependent content links, Level 1 (docs/implementation/edge-and-latent-links-plan-2026-10-05.md, Section 2).

Every edge j -> i carries the reference link G_ij = g_i e^L E_ij, E_ij = I + A_i Phi_ij B_j^T
(``content_link_edges`` 'lowrank': E_ij = exp(A_i B_j^T); 'pure_gauge': G_ij = g_i e^L g_j^{-1}). Tiny
float64 CPU cases only (K <= 6, N <= 4, two heads). Covers the plan's Section 2.6: the W_B = 0 reduction to
the exponential chart (both covariance bodies, both families, both frame layouts, residual on and off),
a brute-force reference built from explicit matrix exponentials (unshifted spread, nonzero token-0 mean),
blockwise against dense, passive gauge covariance, triangle holonomies, causality under large factors,
finite-difference gradients, config/optimizer/resource/state-dict wiring and short model training.
"""
from tests.profiles import make_relational_operators
import itertools
import logging
import math
from typing import Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model import edge_links
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import _relational_artifact_fields
from vfe3.run_resources import parameter_breakdown, parameter_role

torch.set_num_threads(1)
F64 = torch.float64
MODES = ("lowrank", "pure_gauge")
FLOOR = 1e-3


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)


# ----------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------

def _ops(
    mode:     str,
    residual: bool,
    H:        int = 2,
    d:        int = 2,
    r:        int = 2,
    seed:     int = 0,
    sender:   float = 0.4,
    **kwargs: object,
) -> RelationalOperators:
    r"""Float64 edge operators with generic T, O, W_A (or A) and W_B (``sender`` scales W_B; 0 keeps B = 0)."""
    K = H * d
    ops = make_relational_operators([d] * H, 11, attention_residual=residual, predictive_covariance="moment_matched",
                              predictive_covariance_floor=FLOOR, content_link_chart="exponential",
                              content_link_edges=mode, content_link_edge_rank=r, **kwargs).double()
    g = torch.Generator().manual_seed(100 + seed)
    rn = lambda *shape: torch.randn(*shape, generator=g, dtype=F64)
    with torch.no_grad():
        ops.T.copy_(torch.eye(K, dtype=F64).reshape(H, d, K) + 0.3 * rn(H, d, K))
        ops.O.copy_(0.4 * rn(H, K, d))
        ops.edge_sender_map.copy_(sender * rn(*ops.edge_sender_map.shape))
        if mode == "lowrank":
            ops.edge_receiver_map.copy_(0.5 * rn(*ops.edge_receiver_map.shape))
    return ops


def _chart_twin(ops: RelationalOperators) -> RelationalOperators:
    r"""The same shared maps with the edges off (the exponential chart)."""
    twin = make_relational_operators(ops.irrep_dims, 11, attention_residual=ops.attention_residual,
                               predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
                               content_link_chart="exponential").double()
    with torch.no_grad():
        twin.T.copy_(ops.T)
        twin.O.copy_(ops.O)
    return twin


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:          # (..., H, d, d) -> (..., K, K)
    H, d = blocks.shape[-3], blocks.shape[-1]
    out = blocks.new_zeros(*blocks.shape[:-3], H * d, H * d)
    for h in range(H):
        out[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = blocks[..., h, :, :]
    return out


def _inputs(H: int = 2, d: int = 2, N: int = 4, seed: int = 0, batch: Tuple[int, ...] = (1,)) -> Dict:
    r"""Block frames (compact and dense), means with a nonzero token-0 mean, both families, causal routing."""
    K = H * d
    g = torch.Generator().manual_seed(seed)
    rn = lambda *shape: torch.randn(*shape, generator=g, dtype=F64)
    blocks = torch.eye(d, dtype=F64) + 0.3 * rn(*batch, N, H, d, d)
    U = _block_dense(blocks)
    compact = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    dense = CanonicalFrameContext(U, torch.linalg.inv(U))
    mu = rn(*batch, N, K) + 2.0                                    # token 0's reference mean is far from 0
    sigma_diag = torch.rand(*batch, N, K, generator=g, dtype=F64) + 0.3
    factor = rn(*batch, N, K, K)
    sigma_full = factor @ factor.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=F64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(rn(*batch, H, N, N).masked_fill(~causal, -torch.inf), dim=-1)
    return dict(U=U, compact=compact, dense=dense, mu=mu, sigma_diag=sigma_diag, sigma_full=sigma_full,
                beta=beta)


def _prior(like: torch.Tensor, full: bool, seed: int = 99) -> torch.Tensor:
    r"""A local prior covariance Sigma_p in ``like``'s layout (``full``: SPD (..., N, K, K), else a positive
    diagonal (..., N, K)), drawn from its own generator: the anchor of the floor f Sigma_p, distinct from sigma."""
    generator = torch.Generator().manual_seed(seed)
    if not full:
        return torch.rand(like.shape, generator=generator, dtype=F64) + 0.4
    n = like.shape[-1]
    factor = torch.randn(like.shape, generator=generator, dtype=F64)
    return factor @ factor.transpose(-1, -2) / n + 0.5 * torch.eye(n, dtype=F64)


# ----------------------------------------------------------------------------------------------
# Independent brute-force reference (explicit matrix exponentials and loops)
# ----------------------------------------------------------------------------------------------

def _shared(ops: RelationalOperators) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    r"""(F = e^{OT}, R = T^{-1}, [Pi_h]) from the stored maps."""
    H, d, K = len(ops.irrep_dims), ops.irrep_dims[0], ops.input_dim
    Ts = ops.T.detach().reshape(K, K)
    Os = ops.O.detach().permute(1, 0, 2).reshape(K, K)
    R = torch.linalg.inv(Ts)
    return torch.linalg.matrix_exp(Os @ Ts), R, [R[:, h * d:(h + 1) * d] @ Ts[h * d:(h + 1) * d] for h in range(H)]


def _links(ops: RelationalOperators, xi: torch.Tensor) -> List[List[torch.Tensor]]:
    r"""G[i][j] for one sequence of reference means xi (N, K): F exp(A_i B_j^T) or g_i F g_j^{-1}."""
    K, r = ops.input_dim, ops.content_link_edge_rank
    N = xi.shape[0]
    F, _, _ = _shared(ops)
    xh = xi / torch.sqrt(xi.square().mean(dim=-1, keepdim=True) + 1e-12)
    B = (xh @ ops.edge_sender_map.detach().T).reshape(N, K, r)
    if ops.content_link_edges == "lowrank":
        A = (xh @ ops.edge_receiver_map.detach().T).reshape(N, K, r)
        return [[F @ torch.linalg.matrix_exp(A[i] @ B[j].T) for j in range(N)] for i in range(N)]
    A = ops.edge_site_basis.detach()
    g = [torch.linalg.matrix_exp(A @ B[i].T) for i in range(N)]
    return [[g[i] @ F @ torch.linalg.inv(g[j]) for j in range(N)] for i in range(N)]


def _reference(
    ops:   RelationalOperators,
    mu:    torch.Tensor,                # (1, N, K)
    sigma: torch.Tensor,                # (1, N, K, K) local covariance
    beta:  torch.Tensor,                # (1, H, N, N)
    U:     torch.Tensor,                # (1, N, K, K)
    prior: torch.Tensor,                # (1, N, K, K) local prior covariance Sigma_p
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Local mean m_i (residual-free) and covariance U C^att U^T + f Sigma_{p,i} of the generic edge kernel, by
    loops; the prior-anchored floor is added in local coordinates (audit 2026-10-12 T5)."""
    H, K = len(ops.irrep_dims), ops.input_dim
    N = mu.shape[-2]
    _, _, Pi = _shared(ops)
    U_inv = torch.linalg.inv(U[0])
    xi = (U_inv @ mu[0].unsqueeze(-1)).squeeze(-1)                     # (N, K) unshifted
    C = U_inv @ sigma[0] @ U_inv.transpose(-1, -2)
    G = _links(ops, xi)
    M = [[[G[i][j] @ Pi[h] for h in range(H)] for j in range(N)] for i in range(N)]
    means, covs = [], []
    for i in range(N):
        m = sum(beta[0, h, i, j] * M[i][j][h] @ xi[j] for j in range(N) for h in range(H))
        cov = torch.zeros(K, K, dtype=F64)
        for j in range(N):
            for h, hp in itertools.product(range(H), repeat=2):
                c = beta[0, h, i, j] if h == hp else beta[0, h, i, j] * beta[0, hp, i, j]
                cov = cov + c * M[i][j][h] @ C[j] @ M[i][j][hp].T
        for h in range(H):
            n = [M[i][j][h] @ xi[j] for j in range(N)]                 # unshifted n_ij^h
            n_bar = sum(beta[0, h, i, j] * n[j] for j in range(N))
            cov = cov + sum(beta[0, h, i, j] * torch.outer(n[j], n[j]) for j in range(N)) - torch.outer(n_bar, n_bar)
        if ops.attention_residual:
            cross = sum(beta[0, h, i, i] * M[i][i][h] @ C[i] for h in range(H))
            cov = cov + C[i] + cross + cross.T
        means.append(U[0, i] @ m)
        covs.append(U[0, i] @ cov @ U[0, i].T + FLOOR * prior[0, i])
    return torch.stack(means)[None], torch.stack(covs)[None]


def _close(got: torch.Tensor, want: torch.Tensor, tol: float) -> None:
    torch.testing.assert_close(got, want, rtol=tol, atol=tol * float(want.abs().max()))


def _cases(data: Dict) -> List[Tuple[str, torch.Tensor, CanonicalFrameContext]]:
    return [("diag_compact", data["sigma_diag"], data["compact"]), ("diag_dense", data["sigma_diag"], data["dense"]),
            ("full_compact", data["sigma_full"], data["compact"]), ("full_dense", data["sigma_full"], data["dense"])]


# ----------------------------------------------------------------------------------------------
# 1. Reduction to the exponential chart at W_B = 0
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("residual", [False, True])
def test_zero_sender_map_reduces_to_the_exponential_chart(mode: str, residual: bool) -> None:
    ops = _ops(mode, residual, sender=0.0)
    twin = _chart_twin(ops)
    data = _inputs()
    mu, beta = data["mu"], data["beta"]
    for _, sigma, frame in _cases(data):
        _close(ops.predictive_content(mu, beta, frame), twin.predictive_content(mu, beta, frame), 1e-12)
        prior = _prior(sigma, sigma.dim() == 4)
        _close(ops.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=prior),
               twin.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=prior), 1e-12)
        edges = ops.edge_context(mu, beta, frame)
        for body in ("_predictive_covariance_update_dense", "_predictive_covariance_update_blockwise"):
            if body.endswith("blockwise") and not (frame.block_diagonal and sigma.dim() == mu.dim()):
                continue
            _close(getattr(ops, body)(mu, sigma, beta, frame, None, edges),
                   getattr(twin, body)(mu, sigma, beta, frame), 1e-12)


# ----------------------------------------------------------------------------------------------
# 2. Brute-force reference
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("H,d,r", [(2, 2, 1), (2, 2, 2), (2, 3, 1)])
def test_mean_and_covariance_match_the_brute_force_kernel(mode: str, residual: bool, H: int, d: int, r: int) -> None:
    ops = _ops(mode, residual, H=H, d=d, r=r, seed=r + d)
    data = _inputs(H, d, seed=3)
    mu, beta, U = data["mu"], data["beta"], data["U"]
    xi0 = torch.linalg.solve(U[0, 0], mu[0, 0])
    assert float(xi0.norm()) > 1.0                                    # the token-0 shift is not trivial
    for _, sigma, frame in _cases(data):
        dense_sigma = sigma if sigma.dim() == 4 else torch.diag_embed(sigma)
        prior = _prior(sigma, sigma.dim() == 4)
        dense_prior = prior if sigma.dim() == 4 else torch.diag_embed(prior)
        mean_ref, cov_ref = _reference(ops, mu, dense_sigma, beta, U, dense_prior)
        _close(ops.predictive_content(mu, beta, frame), mean_ref, 1e-10)
        want = cov_ref if sigma.dim() == 4 else torch.diagonal(cov_ref, dim1=-2, dim2=-1)
        _close(ops.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=prior), want, 1e-10)


@pytest.mark.parametrize("mode", MODES)
def test_shift_trap_is_detected(mode: str) -> None:
    r"""p_j^h from the token-0-shifted values (the violation the plan warns of) misses the reference by O(1)."""
    ops = _ops(mode, True, seed=3)
    data = _inputs(seed=3)
    mu, beta, U, frame, sigma = data["mu"], data["beta"], data["U"], data["compact"], data["sigma_diag"]
    ctx = ops.edge_context(mu, beta, frame)
    H, d, K = 2, 2, 4
    xi = torch.linalg.solve(U, mu.unsqueeze(-1)).squeeze(-1)
    values = torch.einsum("hsk,...nk->...nhs", ctx.T.reshape(H, d, K), xi - xi[..., :1, :])
    shifted = torch.einsum("...nhsa,...nhs->...nha", ctx.B_ref.reshape(*ctx.B_ref.shape[:-2], H, d, -1), values)
    prior = _prior(sigma, False)
    want = torch.diagonal(_reference(ops, mu, torch.diag_embed(sigma), beta, U, torch.diag_embed(prior))[1],
                          dim1=-2, dim2=-1)
    _close(ops.predictive_covariance_update(mu, sigma, beta, frame, edges=ctx, floor_anchor=prior), want, 1e-10)
    wrong = ops.predictive_covariance_update(mu, sigma, beta, frame, edges=ctx._replace(p=shifted),
                                             floor_anchor=prior)
    assert float((wrong - want).abs().max() / want.abs().max()) > 1e-2


@pytest.mark.parametrize("mode", MODES)
def test_explicit_link_matrices_match_matrix_exponentials(mode: str) -> None:
    ops = _ops(mode, False, seed=4)
    data = _inputs(seed=5)
    ctx = ops.edge_context(data["mu"], data["beta"], data["compact"])
    G = edge_links.edge_link_matrices(ctx)[0]                          # (N, N, K, K)
    xi = torch.linalg.solve(data["U"][0], data["mu"][0].unsqueeze(-1)).squeeze(-1)
    for i, row in enumerate(_links(ops, xi)):
        for j, G_ref in enumerate(row):
            torch.testing.assert_close(G[i, j], G_ref, rtol=1e-11, atol=1e-11)


# ----------------------------------------------------------------------------------------------
# 3. Blockwise body against the dense body with edges on
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("batch", [(1,), (2,)])
def test_blockwise_equals_dense_with_edges(mode: str, residual: bool, batch: Tuple[int, ...]) -> None:
    ops = _ops(mode, residual, H=3, d=2, r=2, seed=6)
    data = _inputs(3, 2, seed=7, batch=batch)
    mu, sigma, beta, frame = data["mu"], data["sigma_diag"], data["beta"], data["compact"]
    edges = ops.edge_context(mu, beta, frame)
    blockwise = ops._predictive_covariance_update_blockwise(mu, sigma, beta, frame, None, edges)
    dense = ops._predictive_covariance_update_dense(mu, sigma, beta, frame, None, edges)
    torch.testing.assert_close(blockwise, dense, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("mode", MODES)
def test_blockwise_gradients_equal_dense_with_edges(mode: str) -> None:
    ops = _ops(mode, True, seed=8)
    data = _inputs(seed=9)
    weights = torch.randn(data["sigma_diag"].shape, generator=torch.Generator().manual_seed(1), dtype=F64)
    grads = []
    for body in ("_predictive_covariance_update_blockwise", "_predictive_covariance_update_dense"):
        ops.zero_grad(set_to_none=True)
        mu = data["mu"].clone().requires_grad_(True)
        sigma = data["sigma_diag"].clone().requires_grad_(True)
        (getattr(ops, body)(mu, sigma, data["beta"], data["compact"]) * weights).sum().backward()
        grads.append([mu.grad, sigma.grad] + [p.grad for p in ops.edge_parameters()] + [ops.T.grad, ops.O.grad])
    for blockwise, dense in zip(*grads):
        torch.testing.assert_close(blockwise, dense, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------------------------------------
# 4. Passive gauge covariance
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("residual", [False, True])
def test_passive_gauge_action_maps_outputs_covariantly(mode: str, residual: bool) -> None:
    r"""mu -> g mu, Sigma -> g Sigma g^T, U -> g U, Sigma_p -> g Sigma_p g^T gives m -> g m, Sigma^+ -> g Sigma^+ g^T
    at fixed beta (the prior-anchored floor moves with the prior)."""
    ops = _ops(mode, residual, seed=10)
    data = _inputs(seed=11)
    N, K = data["mu"].shape[-2:]
    g = torch.eye(K, dtype=F64) + 0.4 * torch.randn(1, N, K, K, generator=torch.Generator().manual_seed(2),
                                                     dtype=F64)
    mu, sigma, beta, U = data["mu"], data["sigma_full"], data["beta"], data["U"]
    moved_U = g @ U
    moved = CanonicalFrameContext(moved_U, torch.linalg.inv(moved_U))
    mean = ops.predictive_content(mu, beta, data["dense"])
    prior = _prior(sigma, True)
    cov = ops.predictive_covariance_update(mu, sigma, beta, data["dense"], floor_anchor=prior)
    mean_g = ops.predictive_content((g @ mu.unsqueeze(-1)).squeeze(-1), beta, moved)
    cov_g = ops.predictive_covariance_update((g @ mu.unsqueeze(-1)).squeeze(-1), g @ sigma @ g.transpose(-1, -2),
                                             beta, moved, floor_anchor=g @ prior @ g.transpose(-1, -2))
    _close(mean_g, (g @ mean.unsqueeze(-1)).squeeze(-1), 1e-9)
    _close(cov_g, g @ cov @ g.transpose(-1, -2), 1e-9)


@pytest.mark.parametrize("mode", MODES)
def test_readout_is_gauge_covariant_through_routing(mode: str) -> None:
    r"""Block-diagonal g keeps the frame compact; routing, mean and covariance all transform covariantly."""
    ops = _ops(mode, True, seed=12)
    data = _inputs(seed=13)
    N, K = data["mu"].shape[-2:]
    g_blocks = torch.eye(2, dtype=F64) + 0.4 * torch.randn(1, N, 2, 2, 2, generator=torch.Generator().manual_seed(3),
                                                           dtype=F64)
    g = _block_dense(g_blocks)
    blocks = data["compact"].forward
    moved = CanonicalFrameContext(g_blocks @ blocks, torch.linalg.inv(g_blocks @ blocks), block_diagonal=True)
    mu, sigma = data["mu"], data["sigma_full"]
    prior = _prior(sigma, True)
    mean, cov = relational_predictive_readout(mu, sigma, data["compact"], ops, return_covariance=True,
                                              floor_anchor=prior)
    mean_g, cov_g = relational_predictive_readout((g @ mu.unsqueeze(-1)).squeeze(-1), g @ sigma @ g.transpose(-1, -2),
                                                  moved, ops, return_covariance=True,
                                                  floor_anchor=g @ prior @ g.transpose(-1, -2))
    _close(mean_g, (g @ mean.unsqueeze(-1)).squeeze(-1), 1e-9)
    _close(cov_g, g @ cov @ g.transpose(-1, -2), 1e-9)


# ----------------------------------------------------------------------------------------------
# 5. Triangle holonomies
# ----------------------------------------------------------------------------------------------

def _power_traces(X: torch.Tensor) -> torch.Tensor:                 # similarity invariants tr X^p, p = 1..K
    K = X.shape[-1]
    out, P = [], torch.eye(K, dtype=X.dtype)
    for _ in range(K):
        P = P @ X
        out.append(torch.diagonal(P, dim1=-2, dim2=-1).sum(-1))
    return torch.stack(out, dim=-1)


@pytest.mark.parametrize("mode", MODES)
def test_triangle_holonomies(mode: str) -> None:
    r"""Pure gauge: G_ik^{-1} G_ij G_jk = g_k F g_k^{-1} has the spectrum of F = e^L; low rank generically not."""
    ops = _ops(mode, False, seed=14, sender=0.6)
    data = _inputs(seed=15)
    ctx = ops.edge_context(data["mu"], data["beta"], data["compact"])
    G = edge_links.edge_link_matrices(ctx)[0]
    F_traces = _power_traces(ctx.F)
    gaps = []
    N = G.shape[0]
    for k, j, i in itertools.combinations(range(N), 3):              # k < j < i
        W = torch.linalg.solve(G[i, k], G[i, j] @ G[j, k])
        gaps.append(float((_power_traces(W) - F_traces).abs().max() / F_traces.abs().max()))
    if mode == "pure_gauge":
        assert max(gaps) < 1e-10
    else:
        assert max(gaps) > 1e-3


# ----------------------------------------------------------------------------------------------
# 6. Causality with large factors (masking before phi_1)
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_future_tokens_do_not_reach_earlier_receivers_even_when_phi1_overflows(mode: str) -> None:
    H, d, N, r = 2, 2, 4, 2
    K = H * d
    ops = _ops(mode, True, H=H, d=d, r=r, seed=16, sender=0.3)
    g = torch.Generator().manual_seed(17)
    u = torch.randn(K * r, generator=g, dtype=F64)
    e_K = torch.zeros(K, dtype=F64)
    e_K[-1] = 1.0
    with torch.no_grad():                                             # a huge sender map along e_K only
        ops.edge_sender_map.add_(1e3 * torch.outer(u / u.norm(), e_K))
    data = _inputs(H, d, N, seed=18)
    blocks, beta = data["compact"].forward, data["beta"]
    xi = torch.randn(1, N, K, generator=g, dtype=F64) + 1.0
    xi[..., -1] = 0.0                                                 # no token reads the huge direction
    local = lambda x: (_block_dense(blocks) @ x.unsqueeze(-1)).squeeze(-1)
    sigma = data["sigma_diag"]
    prior = _prior(sigma, False)
    base_mean = ops.predictive_content(local(xi), beta, data["compact"])
    base_cov = ops.predictive_covariance_update(local(xi), sigma, beta, data["compact"], floor_anchor=prior)
    assert bool(torch.isfinite(base_mean).all()) and bool(torch.isfinite(base_cov).all())
    overflowed = False
    for sign in (1.0, -1.0):
        moved = xi.clone()
        moved[0, N - 1] = 0.0
        moved[0, N - 1, -1] = 5.0 * sign                              # the last token reads it: B_{N-1} huge
        ctx = ops.edge_context(local(moved), beta, data["compact"])
        B_last = ctx.B[0, N - 1]
        if mode == "lowrank":
            unmasked = [edge_links.phi1(B_last.T @ ctx.A[0, i]) for i in range(N - 1)]
        else:
            unmasked = [-edge_links.phi1(-(B_last.T @ ctx.A))]
        overflowed = overflowed or any(not bool(torch.isfinite(x).all()) for x in unmasked)
        mean = ops.predictive_content(local(moved), beta, data["compact"])
        cov = ops.predictive_covariance_update(local(moved), sigma, beta, data["compact"], floor_anchor=prior)
        dense_cov = ops.predictive_covariance_update(local(moved), torch.diag_embed(sigma), beta, data["compact"],
                                                     floor_anchor=torch.diag_embed(prior))
        dense_base = ops.predictive_covariance_update(local(xi), torch.diag_embed(sigma), beta, data["compact"],
                                                      floor_anchor=torch.diag_embed(prior))
        for got, want in ((mean, base_mean), (cov, base_cov), (dense_cov, dense_base)):
            assert bool(torch.isfinite(got[0, :N - 1]).all())
            torch.testing.assert_close(got[0, :N - 1], want[0, :N - 1], rtol=1e-12, atol=1e-12)
    assert overflowed                                                 # the unmasked phi_1 would have been inf


# ----------------------------------------------------------------------------------------------
# 7. Finite-difference gradients
# ----------------------------------------------------------------------------------------------

class _Readout(torch.nn.Module):
    r"""A scalar of the edge mean and both covariance bodies (block frames, diagonal and full families)."""

    def __init__(self, ops: RelationalOperators, weights: List[torch.Tensor]) -> None:
        super().__init__()
        self.ops = ops
        self.weights = weights

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, beta: torch.Tensor,
                frame: CanonicalFrameContext) -> torch.Tensor:
        w_mean, w_diag, w_full = self.weights
        full = torch.diag_embed(sigma) + 0.1 * torch.ones(sigma.shape[-1], sigma.shape[-1], dtype=sigma.dtype)
        return ((w_mean * self.ops.predictive_content(mu, beta, frame)).sum()
                + (w_diag * self.ops.predictive_covariance_update(mu, sigma, beta, frame,
                                                                  floor_anchor=_prior(sigma, False))).sum()
                + (w_full * self.ops.predictive_covariance_update(mu, full, beta, frame,
                                                                  floor_anchor=_prior(full, True))).sum())


@pytest.mark.parametrize("mode", MODES)
def test_finite_difference_gradients(mode: str) -> None:
    ops = _ops(mode, True, H=2, d=2, r=2, seed=19)
    data = _inputs(N=3, seed=20)
    g = torch.Generator().manual_seed(4)
    weights = [torch.randn(data["mu"].shape, generator=g, dtype=F64),
               torch.randn(data["sigma_diag"].shape, generator=g, dtype=F64),
               torch.randn(data["sigma_full"].shape[:-3] + (3, 4, 4), generator=g, dtype=F64)]
    wrap = _Readout(ops, weights)
    names = ["ops.edge_sender_map", "ops.edge_receiver_map" if mode == "lowrank" else "ops.edge_site_basis"]

    def scalar(W_B: torch.Tensor, W_A: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(wrap, {names[0]: W_B, names[1]: W_A},
                                          (mu, sigma, data["beta"], data["compact"]))

    inputs = tuple(x.detach().clone().requires_grad_(True) for x in (
        dict(ops.named_parameters())[names[0][4:]], dict(ops.named_parameters())[names[1][4:]],
        data["mu"], data["sigma_diag"]))
    assert torch.autograd.gradcheck(scalar, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)


# ----------------------------------------------------------------------------------------------
# Checkpointed covariance with edges
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_checkpointed_covariance_is_bit_identical_with_edges(mode: str) -> None:
    outputs = []
    for flag in (False, True):
        ops = _ops(mode, True, seed=21, predictive_covariance_checkpoint=flag)
        data = _inputs(seed=22)
        mu = data["mu"].clone().requires_grad_(True)
        out = relational_predictive_readout(mu, data["sigma_diag"], data["compact"], ops, return_covariance=True,
                                            floor_anchor=_prior(data["sigma_diag"], False))
        (out[0].sum() + out[1].square().sum()).backward()
        outputs.append([out[0].detach(), out[1].detach(), mu.grad] + [p.grad for p in ops.parameters()])
    for plain, recomputed in zip(*outputs):
        assert torch.equal(plain, recomputed)


# ----------------------------------------------------------------------------------------------
# 8. Config, operator refusals, optimizer, resources, state dict, consumers
# ----------------------------------------------------------------------------------------------

_EDGE = dict(attention_mode="projected_predictive", content_link_chart="exponential")


def test_config_validation_and_inactive_fields() -> None:
    assert "content_link_edge_rank" in test_config(**_EDGE).inactive_fields()
    for mode in MODES:
        cfg = test_config(**_EDGE, content_link_edges=mode, content_link_edge_rank=2)
        assert "content_link_edge_rank" not in cfg.inactive_fields()
    with pytest.raises(ValueError, match="content_link_edges"):
        test_config(**_EDGE, content_link_edges="dense")
    with pytest.raises(ValueError, match="exponential"):
        test_config(attention_mode="projected_predictive", content_link_chart="affine", content_link_edges="lowrank")
    with pytest.raises(ValueError, match="projected_predictive"):
        test_config(attention_mode="projected_gradient", content_link_chart="exponential",
                    content_link_edges="lowrank")
    for rank in (0, 5):
        with pytest.raises(ValueError, match="content_link_edge_rank"):
            test_config(**_EDGE, content_link_edges="pure_gauge", content_link_edge_rank=rank)
    with pytest.raises(TypeError):
        test_config(**_EDGE, content_link_edges="lowrank", content_link_edge_rank=2.0)
    with pytest.raises(ValueError, match="equal heads"):
        test_config(**_EDGE, content_link_edges="lowrank", gauge_group="so_n", group_n=3,
                    irrep_spec=[("l1", 2), ("l2", 1)], embed_dim=11, n_heads=3)


def test_operator_constructor_refusals() -> None:
    kwargs = dict(content_link_chart="exponential", content_link_edges="lowrank", content_link_edge_rank=2)
    with pytest.raises(ValueError, match="exponential"):
        make_relational_operators([2, 2], 1, **{**kwargs, "content_link_chart": "first_order"})
    with pytest.raises(ValueError, match="predictive_content"):
        make_relational_operators([2, 2], 1, readout_mode="full_link_update", **kwargs)
    with pytest.raises(ValueError, match="equal heads"):
        make_relational_operators([3, 3, 5], 1, **kwargs)
    with pytest.raises(ValueError, match="content_link_edge_rank"):
        make_relational_operators([2, 2], 1, **{**kwargs, "content_link_edge_rank": 5})
    with pytest.raises(ValueError, match="content_link_edges"):
        make_relational_operators([2, 2], 1, **{**kwargs, "content_link_edges": "curved"})
    off = make_relational_operators([2, 2], 1, content_link_chart="exponential")
    assert off.edge_parameters() == [] and not any("edge" in k for k in off.state_dict())


def _edge_model(mode: str, **overrides: object) -> VFEModel:
    cfg = tiny_config(**{**_EDGE, "content_link_edges": mode, "content_link_edge_rank": 2, "max_seq_len": 4,
                         "n_layers": 2, **overrides})
    torch.manual_seed(0)
    return VFEModel(cfg)


def test_initialization_and_state_dict_keys() -> None:
    torch.manual_seed(0)
    default = VFEModel(tiny_config(**_EDGE, max_seq_len=4, n_layers=2))
    keys = set(default.state_dict())
    assert not any("edge" in key for key in keys)
    for mode in MODES:
        model = _edge_model(mode)
        new = set(model.state_dict()) - keys
        leaf = "edge_receiver_map" if mode == "lowrank" else "edge_site_basis"
        assert new == {f"{prefix}.{name}" for prefix in ("relational_operators", "relational_operators_extra.0")
                       for name in (leaf, "edge_sender_map")}
        for ops, ref in zip(model.relational_operators_by_layer, default.relational_operators_by_layer):
            assert torch.equal(ops.raw_score_q, ref.raw_score_q) and torch.equal(ops.O, ref.O)
            assert float(ops.edge_sender_map.abs().max()) == 0.0
            if mode == "pure_gauge":
                torch.testing.assert_close(ops.edge_site_basis.T @ ops.edge_site_basis, torch.eye(2),
                                           rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_optimizer_grouping_and_parameter_roles(mode: str) -> None:
    model = _edge_model(mode)
    cfg = model.cfg
    optimizer = T.build_optimizer(model, cfg)                         # the ungrouped-parameter guard runs here
    edge = {name: p for name, p in model.named_parameters() if ".edge_" in name}
    assert len(edge) == 4
    for name, p in edge.items():
        groups = [group for group in optimizer.param_groups if any(q is p for q in group["params"])]
        assert len(groups) == 1
        assert groups[0].get("lr_aux_role") == "relational_content" and groups[0]["weight_decay"] == 0.0
        assert T._parameter_group(name) == "relational_content"
        assert parameter_role(name) == ("context_processing", "relational_content")
    breakdown = parameter_breakdown(dict(model.named_parameters()), cfg)
    content = sum(p.numel() for name, p in model.named_parameters()
                  if name.rsplit(".", 1)[-1] in ("T", "O") or ".edge_" in name)
    assert breakdown["context_processing"]["relational_content"] == content
    record = _relational_artifact_fields(cfg)["relational_operators"]
    assert record["content_link_edges"] == mode
    assert record["total_parameters"] == sum(p.numel() for ops in model.relational_operators_by_layer
                                             for p in ops.parameters())
    assert "content_link_edges" not in _relational_artifact_fields(tiny_config(**_EDGE))["relational_operators"]


@pytest.mark.parametrize("mode", MODES)
def test_state_dict_round_trip(mode: str) -> None:
    model = _edge_model(mode)
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.edge_sender_map.normal_(std=0.3)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    fresh = _edge_model(mode)
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        want, _ = model.eval().forward_beliefs(ids)
        got, _ = fresh.eval().forward_beliefs(ids)
    assert torch.equal(got.mu, want.mu) and torch.equal(got.sigma, want.sigma)


@pytest.mark.parametrize("mode", MODES)
def test_diagnostics_and_consumers(mode: str) -> None:
    model = _edge_model(mode).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    base = VFEModel(tiny_config(**_EDGE, max_seq_len=4, n_layers=2)).eval()
    base.load_state_dict(model.state_dict(), strict=False)
    d0, b0 = model.diagnostics(ids), base.diagnostics(ids)
    for key in ("relational_link_selfloop_spectral_radius_max", "relational_link_selfloop_spectral_radius_mean"):
        assert math.isclose(d0[key], b0[key], rel_tol=1e-5)           # B = 0: the exponential chart's self loop
    assert d0["relational_edge_part_norm_max"] == 0.0
    assert d0["relational_edge_link_spectrum_shared_generator"] == 1.0   # marks the shared-generator columns
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.edge_sender_map.normal_(std=0.5)
    rows = model.diagnostics_per_layer(ids)
    edge_keys = [key for key in rows if key.startswith("relational_edge_")]
    assert sorted("val_" + key for key in edge_keys) == sorted(T._VAL_EDGE_DIAG_KEYS)
    assert all(math.isfinite(v) and v > 0 for key in edge_keys for v in rows[key])
    assert not any(key.startswith("relational_edge_") for key in b0)
    # Map invariants carry the edge maps; the holonomy analysis and the full-link builder refuse.
    from vfe3.posthoc.holonomy import layer_content_maps
    from vfe3.posthoc.maps import map_invariants
    invariant_rows = map_invariants(model.state_dict(), model.cfg)
    assert all(row["edge_sender_map_fro"] > 0 for row in invariant_rows)
    from vfe3.posthoc.maps import link_eigenvalue_rows
    assert all(row["link_shared_generator_only"] == 1.0
               for row in invariant_rows + link_eigenvalue_rows(model.state_dict(), model.cfg))
    assert not any("link_shared_generator_only" in row for row in map_invariants(base.state_dict(), base.cfg))
    with pytest.raises(NotImplementedError, match="content_link_edges"):
        layer_content_maps(model.state_dict(), model.cfg)
    from vfe3.gradients.full_link import build_full_link_context
    ops = model.relational_operators
    with pytest.raises(ValueError, match="content_link_edges"):
        build_full_link_context(torch.zeros(1, 4, 4), torch.ones(1, 4, 4), torch.zeros(1, 4, 4),
                                torch.ones(1, 4, 4), CanonicalFrameContext(torch.eye(4).expand(1, 4, 4, 4),
                                                                           torch.eye(4).expand(1, 4, 4, 4)), ops)


# ----------------------------------------------------------------------------------------------
# 9. Model smoke: short training
# ----------------------------------------------------------------------------------------------

_ROUTES = {
    "diagonal": dict(family="gaussian_diagonal"),
    "full": dict(family="gaussian_full"),
    "passthrough": dict(predictive_covariance="passthrough"),
    "checkpoint": dict(predictive_covariance_checkpoint=True),
    "tied_frames": dict(gauge_tying="shared"),
    "no_residual": dict(attention_residual=False),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_short_training_is_finite_and_trains_the_edge_maps(mode: str, route: str) -> None:
    model = _edge_model(mode, decode_mode="linear", **_ROUTES[route])
    cfg = model.cfg
    optimizer = T.build_optimizer(model, cfg)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 6, 2, 1]])
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(ids, targets)
        assert math.isfinite(float(loss))
        loss.backward()
        for ops in model.relational_operators_by_layer:
            assert float(ops.edge_sender_map.grad.abs().max()) > 0.0
            receiver = ops.edge_receiver_map if mode == "lowrank" else ops.edge_site_basis
            if step > 0:                                              # B != 0 after the first update
                assert float(receiver.grad.abs().max()) > 0.0
        optimizer.step()


def test_persisted_run_logs_the_edge_columns(tmp_path) -> None:
    import csv
    model = _edge_model("lowrank", decode_mode="linear", eval_stride=2)
    cfg = model.cfg
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    train = DataLoader(TokenWindows(torch.arange(41) % 7, 4), batch_size=2, shuffle=True)
    val = DataLoader(TokenWindows((torch.arange(23) * 3) % 7, 4, pad_final=True, stride=2), batch_size=2)
    torch.manual_seed(11)
    T.train(model, train, cfg, n_steps=4, grad_clip=1.0, log_interval=2, eval_interval=2, val_loader=val,
            device=torch.device("cpu"), logger=logging.getLogger("edge-test"), artifacts=artifacts,
            generate_samples=False)
    with (tmp_path / "run" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [2, 4]
    for row in rows:
        assert int(row["diagnostics_failures"]) == 0
        assert row["param_norm_group_other"] == ""                    # the edge maps are relational_content
        for key in T._VAL_EDGE_DIAG_KEYS:
            assert math.isfinite(float(row[key])), key


@pytest.mark.parametrize("mode", MODES)
def test_train_loop_runs_with_edges(mode: str) -> None:
    model = _edge_model(mode)
    cfg = model.cfg
    torch.manual_seed(11)
    losses = T.train(model, DataLoader(TokenWindows(torch.arange(41) % 7, 4), batch_size=2, shuffle=True), cfg,
                     n_steps=3, grad_clip=1.0, log_interval=0, eval_interval=0, device=torch.device("cpu"),
                     logger=logging.getLogger("edge-test"), artifacts=None, generate_samples=False)
    assert all(math.isfinite(float(x)) for x in losses)
    assert all(float(ops.edge_sender_map.abs().max()) > 0.0 for ops in model.relational_operators_by_layer)
