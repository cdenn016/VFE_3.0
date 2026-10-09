r"""Independent brute-force reference for the Level 1 edge links (plan Section 2, 2026-10-05).

Written without reading ``vfe3/model/edge_links.py`` or ``tests/test_edge_links_20261005.py``. Every
link :math:`G_{ij}` is built explicitly in float64 from the plan's definitions,

.. math::
    G_{ij} = e^{L}\exp(A_iB_j^\top)\ \text{(lowrank)},\qquad
    G_{ij} = g_i e^{L} g_j^{-1},\ g_i=\exp(AB_i^\top)\ \text{(pure gauge)},

with :math:`L = OT` (head stacks), :math:`A_i=\mathrm{reshape}(W_A\hat\xi_i,(K,r))`,
:math:`B_j=\mathrm{reshape}(W_B\hat\xi_j,(K,r))`, :math:`\hat\xi=\xi/\sqrt{\mathrm{mean}\,\xi^2+10^{-12}}`,
:math:`\xi_j=U_j^{-1}\mu_j`. The head edge map is :math:`M_h^{ij}=G_{ij}\Pi_h` with
:math:`\Pi_h=R_hT_h`, :math:`R=T^{-1}`, and in local coordinates :math:`L^h_{ij}=U_iM_h^{ij}U_j^{-1}`.

The moments are derived here from the kernel

.. math::
    X = r\,k_i + \sum_h L^h_{iJ^h}k_{J^h} + \epsilon,\qquad
    k_s\sim N(\mu_s,\Sigma_s)\ \text{independent},\quad J^h\sim\beta^h_i\ \text{independent},\quad
    \epsilon\sim N\bigl(0,\ \phi\,\tfrac{\mathrm{tr}C_i}{K}\,U_iU_i^\top\bigr),

by the law of total covariance, :math:`\mathrm{Cov}X=E_J[\mathrm{Cov}(X|J)]+\mathrm{Cov}_J[E(X|J)]`.
Conditioned on the labels, :math:`X=\sum_sA_s(J)k_s+\epsilon` with
:math:`A_s(J)=r\delta_{si}I+\sum_h1[J^h=s]L^h_{is}`, so with :math:`E\,1[J^h=s]=\beta^h_{is}` and
:math:`E\,1[J^h=s]1[J^{h'}=s]=c^{hh'}_{is}` (:math:`\beta^h_{is}` on the diagonal, the product off it),

.. math::
    E_J[\mathrm{Cov}(X|J)] = r\Sigma_i + r\sum_h\beta^h_{ii}\bigl(L^h_{ii}\Sigma_i+\Sigma_iL^{h\top}_{ii}\bigr)
      + \sum_j\sum_{h,h'}c^{hh'}_{ij}L^h_{ij}\Sigma_jL^{h'\top}_{ij},

(:math:`r^2=r` for :math:`r\in\{0,1\}`), and since the residual :math:`r\mu_i` does not depend on the
labels and the heads are independent,

.. math::
    \mathrm{Cov}_J[E(X|J)] = \sum_h\Bigl[\sum_j\beta^h_{ij}n^h_{ij}n^{h\top}_{ij}-\bar n^h_i\bar n^{h\top}_i\Bigr],
    \qquad n^h_{ij}=L^h_{ij}\mu_j\ \text{(unshifted)},\ \bar n^h_i=\sum_j\beta^h_{ij}n^h_{ij}.

The mean is :math:`E X = r\mu_i+\sum_h\sum_j\beta^h_{ij}n^h_{ij}`. A second, label-enumerating
computation of the same kernel checks this derivation against itself. Pairs with zero routing weight
in every head are never visited, so the reference stays finite where the implementation must mask.
"""
from tests.profiles import make_relational_operators
import itertools
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model import edge_links
from vfe3.model.relational_operators import RelationalOperators

torch.set_num_threads(1)
F64 = torch.float64
FLOOR = 1e-3
H = 2


def _randn(generator: torch.Generator):
    return lambda *shape: torch.randn(*shape, generator=generator, dtype=F64)


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:   # (N, H, d, d) -> (N, K, K)
    r"""Block-diagonal (K, K) matrices from per-head blocks."""
    N, n_heads, d, _ = blocks.shape
    out = blocks.new_zeros(N, n_heads * d, n_heads * d)
    for h in range(n_heads):
        out[:, h*d:(h+1)*d, h*d:(h+1)*d] = blocks[:, h]
    return out


def _make_ops(
    d:        int,
    mode:     str,
    rank:     int,
    residual: bool,
    seed:     int = 11,
) -> RelationalOperators:
    r"""An edge-enabled (or "off") exponential-chart operator in float64."""
    return make_relational_operators(
        [d] * H, seed, readout_mode="predictive_content", attention_residual=residual,
        predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
        content_link_chart="exponential", content_link_edges=mode, content_link_edge_rank=rank,
    ).double()


def _set_maps(ops: RelationalOperators, generator: torch.Generator, T_scale: float = 0.5,
              edge_scale: float = 1.0) -> None:
    r"""Random T, O (near identity), random nonzero W_A / A and W_B."""
    randn = _randn(generator)
    K = ops.input_dim
    with torch.no_grad():
        ops.T.copy_(ops.T + T_scale * randn(*ops.T.shape) / K ** 0.5)
        ops.O.copy_(ops.O + T_scale * randn(*ops.O.shape) / K ** 0.5)
        if ops.content_link_edges == "lowrank":
            ops.edge_receiver_map.copy_(edge_scale * randn(*ops.edge_receiver_map.shape) / K)
        elif ops.content_link_edges == "pure_gauge":
            ops.edge_site_basis.copy_(RelationalOperators._orthonormal_columns(randn(*ops.edge_site_basis.shape)))
        if ops.content_link_edges != "off":
            ops.edge_sender_map.copy_(edge_scale * randn(*ops.edge_sender_map.shape) / K)


def _inputs(
    N:        int,
    K:        int,
    d:        int,
    generator: torch.Generator,
    block:    bool,
    token0_mean: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CanonicalFrameContext]:
    r"""Random means (token 0 shifted by ``token0_mean``), diagonal and full covariances, causal
    routing from random logits, random frames (block-diagonal per head or dense)."""
    randn = _randn(generator)
    mu = randn(N, K)
    mu[0] = mu[0] + token0_mean                                     # nonzero token-0 mean (shift trap)
    sigma_diag = torch.rand(N, K, generator=generator, dtype=F64) + 0.3
    factor = randn(N, K, K)
    sigma_full = factor @ factor.transpose(-1, -2) / K + 0.5 * torch.eye(K, dtype=F64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(randn(H, N, N).masked_fill(~causal, -torch.inf), dim=-1)
    if block:
        blocks = torch.eye(d, dtype=F64) + 0.3 * randn(N, H, d, d)
        frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
        U = _block_dense(blocks)
    else:
        U = torch.eye(K, dtype=F64) + 0.3 * randn(N, K, K)
        frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    return mu, sigma_diag, sigma_full, beta, U, frame


# --------------------------------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------------------------------

def _chart(ops: RelationalOperators) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    r"""Float64 :math:`F=e^{OT}`, :math:`R=T^{-1}` and the head projectors :math:`\Pi_h=R_hT_h`."""
    T = ops.T.detach().to(F64)                                      # (H, d, K)
    O = ops.O.detach().to(F64)                                      # (H, K, d)
    n_heads, d, K = T.shape
    T_stack = T.reshape(K, K)                                       # rows by head
    O_stack = O.permute(1, 0, 2).reshape(K, K)                      # columns by head
    F = torch.linalg.matrix_exp(O_stack @ T_stack)
    R = torch.linalg.inv(T_stack)
    Pi = [R[:, h*d:(h+1)*d] @ T[h] for h in range(n_heads)]         # each (K, K)
    return F, R, Pi


def _reference_links(
    ops:  RelationalOperators,
    mu:   torch.Tensor,     # (N, K)
    beta: torch.Tensor,     # (H, N, N)
    U:    torch.Tensor,     # (N, K, K) dense frames
) -> Dict[Tuple[int, int], torch.Tensor]:
    r"""Explicit :math:`G_{ij}` (reference coordinates) on every routed pair (some head weight nonzero)."""
    N, K = mu.shape
    r = ops.content_link_edge_rank
    F, _, _ = _chart(ops)
    U_inv = torch.linalg.inv(U)
    xi = torch.stack([U_inv[j] @ mu[j] for j in range(N)])          # (N, K)
    xi_hat = xi / torch.sqrt((xi ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    W_B = ops.edge_sender_map.detach().to(F64)
    B = [(W_B @ xi_hat[j]).reshape(K, r) for j in range(N)]
    if ops.content_link_edges == "lowrank":
        W_A = ops.edge_receiver_map.detach().to(F64)
        A = [(W_A @ xi_hat[i]).reshape(K, r) for i in range(N)]
        link = lambda i, j: F @ torch.linalg.matrix_exp(A[i] @ B[j].T)
    else:
        A0 = ops.edge_site_basis.detach().to(F64)                   # (K, r)
        g = [torch.linalg.matrix_exp(A0 @ B[i].T) for i in range(N)]
        g_inv = [torch.linalg.matrix_exp(-A0 @ B[j].T) for j in range(N)]
        link = lambda i, j: g[i] @ F @ g_inv[j]
    routed = (beta != 0).any(dim=0)                                 # (N, N)
    return {(i, j): link(i, j) for i in range(N) for j in range(N) if bool(routed[i, j])}


def _reference_moments(
    ops:   RelationalOperators,
    mu:    torch.Tensor,    # (N, K)
    sigma: torch.Tensor,    # (N, K, K) full local covariance
    beta:  torch.Tensor,    # (H, N, N)
    U:     torch.Tensor,    # (N, K, K)
    residual: bool,
    floor: float = FLOOR,
    shift_trap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Mean (without the residual, as ``predictive_content`` returns it) and the moment-matched
    covariance of the kernel, by explicit loops over j, h, h' (module docstring).

    ``shift_trap`` deliberately forms the spread from the token-0-shifted reference means
    :math:`\xi_j-\xi_0` (the shortcut the edge-free code may take, invalid with edge-dependent links);
    it exists only to show the test can see that error."""
    N, K = mu.shape
    r = 1.0 if residual else 0.0
    _, _, Pi = _chart(ops)
    U_inv = torch.linalg.inv(U)
    G = _reference_links(ops, mu, beta, U)
    L = {(h, i, j): U[i] @ G[i, j] @ Pi[h] @ U_inv[j] for (i, j) in G for h in range(H)}
    xi0 = U_inv[0] @ mu[0]
    means, covs = [], []
    for i in range(N):
        senders = [j for j in range(N) if (i, j) in G]
        n = {(h, j): L[h, i, j] @ mu[j] for h in range(H) for j in senders}
        n_bar = [sum(beta[h, i, j] * n[h, j] for j in senders) for h in range(H)]
        mean = sum(n_bar)
        if shift_trap:
            n = {(h, j): L[h, i, j] @ (mu[j] - U[j] @ xi0) for h in range(H) for j in senders}
            n_bar = [sum(beta[h, i, j] * n[h, j] for j in senders) for h in range(H)]
        cov = r * sigma[i]
        if (i, i) in G:
            for h in range(H):
                cov = cov + r * beta[h, i, i] * (L[h, i, i] @ sigma[i] + sigma[i] @ L[h, i, i].T)
        for j in senders:
            for h in range(H):
                for hp in range(H):
                    c = beta[h, i, j] if h == hp else beta[h, i, j] * beta[hp, i, j]
                    cov = cov + c * L[h, i, j] @ sigma[j] @ L[hp, i, j].T
        for h in range(H):
            cov = cov + sum(beta[h, i, j] * torch.outer(n[h, j], n[h, j]) for j in senders)
            cov = cov - torch.outer(n_bar[h], n_bar[h])
        C_i = U_inv[i] @ sigma[i] @ U_inv[i].T
        cov = cov + floor * torch.trace(C_i) / K * (U[i] @ U[i].T)
        means.append(mean)
        covs.append(cov)
    return torch.stack(means), torch.stack(covs)


def _enumerated_moments(
    ops:   RelationalOperators,
    mu:    torch.Tensor,
    sigma: torch.Tensor,
    beta:  torch.Tensor,
    U:     torch.Tensor,
    residual: bool,
    floor: float = FLOOR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The same kernel by enumerating every head-label tuple (checks the loop derivation)."""
    N, K = mu.shape
    _, _, Pi = _chart(ops)
    U_inv = torch.linalg.inv(U)
    G = _reference_links(ops, mu, beta, U)
    means, covs = [], []
    for i in range(N):
        first = torch.zeros(K, dtype=F64)
        second = torch.zeros(K, K, dtype=F64)
        within = torch.zeros(K, K, dtype=F64)
        for labels in itertools.product(range(N), repeat=H):
            weight = torch.prod(torch.stack([beta[h, i, j] for h, j in enumerate(labels)]))
            if weight == 0:
                continue
            A_s = {s: torch.zeros(K, K, dtype=F64) for s in range(N)}
            if residual:
                A_s[i] = A_s[i] + torch.eye(K, dtype=F64)
            for h, j in enumerate(labels):
                A_s[j] = A_s[j] + U[i] @ G[i, j] @ Pi[h] @ U_inv[j]
            mean = sum(A_s[s] @ mu[s] for s in range(N))
            first = first + weight * mean
            second = second + weight * torch.outer(mean, mean)
            within = within + weight * sum(A_s[s] @ sigma[s] @ A_s[s].T for s in range(N))
        C_i = U_inv[i] @ sigma[i] @ U_inv[i].T
        means.append(first - (mu[i] if residual else 0.0))
        covs.append(within + second - torch.outer(first, first) + floor * torch.trace(C_i) / K * (U[i] @ U[i].T))
    return torch.stack(means), torch.stack(covs)


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------

CASES = [(mode, rank) for mode in ("lowrank", "pure_gauge") for rank in (1, 2)]


@pytest.mark.parametrize("mode,rank", CASES)
@pytest.mark.parametrize("residual", [False, True])
def test_reference_loop_derivation_matches_label_enumeration(mode: str, rank: int, residual: bool) -> None:
    r"""Self-check of the reference: the j, h, h' loop formula equals the enumerated kernel."""
    generator = torch.Generator().manual_seed(1)
    d, N = 3, 4
    ops = _make_ops(d, mode, rank, residual)
    _set_maps(ops, generator)
    mu, _, sigma_full, beta, U, _ = _inputs(N, H * d, d, generator, block=False)
    mean_a, cov_a = _reference_moments(ops, mu, sigma_full, beta, U, residual)
    mean_b, cov_b = _enumerated_moments(ops, mu, sigma_full, beta, U, residual)
    torch.testing.assert_close(mean_a, mean_b, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(cov_a, cov_b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("mode,rank", CASES)
@pytest.mark.parametrize("block", [True, False])
def test_link_matrices_match_explicit_links(mode: str, rank: int, block: bool) -> None:
    r"""The implementation's :math:`G_{ij}` equal the explicit matrix exponentials on routed pairs."""
    generator = torch.Generator().manual_seed(2)
    d, N = 3, 4
    ops = _make_ops(d, mode, rank, False)
    _set_maps(ops, generator)
    mu, _, _, beta, U, frame = _inputs(N, H * d, d, generator, block=block)
    G_ref = _reference_links(ops, mu, beta, U)
    with torch.no_grad():
        G_impl = edge_links.edge_link_matrices(ops.edge_context(mu, beta, frame))   # (N, N, K, K)
    assert G_impl.shape == (N, N, H * d, H * d)
    for (i, j), G in G_ref.items():
        torch.testing.assert_close(G_impl[i, j], G, atol=1e-10, rtol=1e-10, msg=f"G[{i},{j}] {mode} r={rank}")


@pytest.mark.parametrize("mode,rank", CASES)
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("block", [True, False])
@pytest.mark.parametrize("family", ["diag", "full"])
def test_mean_and_covariance_match_reference(mode: str, rank: int, residual: bool, block: bool, family: str) -> None:
    r"""Plan 2.6 item 2: implementation vs the brute-force kernel moments, both bodies and families."""
    generator = torch.Generator().manual_seed(3 + rank)
    d, N = 3, 4
    ops = _make_ops(d, mode, rank, residual)
    _set_maps(ops, generator)
    mu, sigma_diag, sigma_full, beta, U, frame = _inputs(N, H * d, d, generator, block=block)
    sigma = sigma_diag if family == "diag" else sigma_full
    dense_sigma = torch.diag_embed(sigma) if family == "diag" else sigma
    mean_ref, cov_ref = _reference_moments(ops, mu, dense_sigma, beta, U, residual)
    with torch.no_grad():
        mean = ops.predictive_content(mu, beta, frame)
        cov = ops.predictive_covariance_update(mu, sigma, beta, frame)
    expected = torch.diagonal(cov_ref, dim1=-2, dim2=-1) if family == "diag" else cov_ref
    assert mean.shape == mu.shape and cov.shape == sigma.shape
    torch.testing.assert_close(mean, mean_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cov, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode,rank", CASES)
def test_reference_has_teeth(mode: str, rank: int) -> None:
    r"""The configuration of the reference test can see two errors: ignoring the edges (the "off"
    exponential chart differs by more than the tolerance) and the shift trap (a spread formed from
    token-0-shifted means differs from the implementation by more than the tolerance)."""
    generator = torch.Generator().manual_seed(3 + rank)
    d, N = 3, 4
    ops = _make_ops(d, mode, rank, True)
    off = _make_ops(d, "off", rank, True)
    _set_maps(ops, generator)
    with torch.no_grad():
        off.T.copy_(ops.T)
        off.O.copy_(ops.O)
    mu, _, sigma_full, beta, U, frame = _inputs(N, H * d, d, generator, block=False)
    _, cov_ref = _reference_moments(ops, mu, sigma_full, beta, U, True)
    _, cov_trap = _reference_moments(ops, mu, sigma_full, beta, U, True, shift_trap=True)
    with torch.no_grad():
        mean, mean_off = ops.predictive_content(mu, beta, frame), off.predictive_content(mu, beta, frame)
        cov = ops.predictive_covariance_update(mu, sigma_full, beta, frame)
    scale = float(cov.abs().max())
    assert float((mean - mean_off).abs().max()) > 1e-3 * float(mean.abs().max())
    assert float((cov_trap - cov).abs().max()) > 1e-4 * scale
    assert float((cov_ref - cov).abs().max()) <= 1e-10 * scale


@pytest.mark.parametrize("mode,rank", CASES)
@pytest.mark.parametrize("residual", [False, True])
def test_batched_inputs_match_reference(mode: str, rank: int, residual: bool) -> None:
    r"""A leading batch axis gives each batch element its own reference moments."""
    generator = torch.Generator().manual_seed(5)
    d, N, B = 2, 3, 2
    K = H * d
    ops = _make_ops(d, mode, rank, residual)
    _set_maps(ops, generator)
    items = [_inputs(N, K, d, generator, block=True) for _ in range(B)]
    mu = torch.stack([it[0] for it in items])
    sigma = torch.stack([it[1] for it in items])
    beta = torch.stack([it[3] for it in items])
    blocks = torch.stack([it[5].forward for it in items])
    frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    with torch.no_grad():
        mean = ops.predictive_content(mu, beta, frame)
        cov = ops.predictive_covariance_update(mu, sigma, beta, frame)
    for b in range(B):
        mean_ref, cov_ref = _reference_moments(ops, mu[b], torch.diag_embed(sigma[b]), beta[b], items[b][4], residual)
        torch.testing.assert_close(mean[b], mean_ref, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(cov[b], torch.diagonal(cov_ref, dim1=-2, dim2=-1), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode,rank", CASES)
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("block", [True, False])
@pytest.mark.parametrize("family", ["diag", "full"])
def test_zero_sender_map_reduces_to_exponential_chart(mode: str, rank: int, residual: bool, block: bool, family: str) -> None:
    r"""Plan 2.6 item 1: with :math:`W_B=0` both modes equal the edge-free exponential chart."""
    generator = torch.Generator().manual_seed(7)
    d, N = 3, 4
    ops = _make_ops(d, mode, rank, residual)
    off = _make_ops(d, "off", rank, residual)
    _set_maps(ops, generator)
    with torch.no_grad():
        ops.edge_sender_map.zero_()
        off.T.copy_(ops.T)
        off.O.copy_(ops.O)
    mu, sigma_diag, sigma_full, beta, U, frame = _inputs(N, H * d, d, generator, block=block)
    sigma = sigma_diag if family == "diag" else sigma_full
    with torch.no_grad():
        mean_on, mean_off = ops.predictive_content(mu, beta, frame), off.predictive_content(mu, beta, frame)
        cov_on = ops.predictive_covariance_update(mu, sigma, beta, frame)
        cov_off = off.predictive_covariance_update(mu, sigma, beta, frame)
    torch.testing.assert_close(mean_on, mean_off, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov_on, cov_off, atol=1e-12, rtol=1e-12)


def _gauge(generator: torch.Generator, N: int, K: int, d: int, block: bool) -> torch.Tensor:   # (N, K, K)
    randn = _randn(generator)
    if block:
        return _block_dense(torch.eye(d, dtype=F64) + 0.3 * randn(N, H, d, d))
    return torch.eye(K, dtype=F64) + 0.3 * randn(N, K, K)


@pytest.mark.parametrize("mode,rank", CASES)
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("block", [True, False])
def test_passive_gauge_covariance(mode: str, rank: int, residual: bool, block: bool) -> None:
    r"""Plan 2.6 item 4: :math:`\mu\to g\mu`, :math:`\Sigma\to g\Sigma g^\top`, :math:`U\to gU` gives
    :math:`m\to gm` and :math:`\Sigma^+\to g\Sigma^+g^\top` (full family, so the output is a matrix)."""
    generator = torch.Generator().manual_seed(9)
    d, N = 3, 4
    K = H * d
    ops = _make_ops(d, mode, rank, residual)
    _set_maps(ops, generator)
    mu, _, sigma_full, beta, U, frame = _inputs(N, K, d, generator, block=block)
    g = _gauge(generator, N, K, d, block)
    mu_g = (g @ mu.unsqueeze(-1)).squeeze(-1)
    sigma_g = g @ sigma_full @ g.transpose(-1, -2)
    U_g = g @ U
    if block:
        blocks = torch.stack([U_g[:, h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=1)
        frame_g = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    else:
        frame_g = CanonicalFrameContext(U_g, torch.linalg.inv(U_g))
    with torch.no_grad():
        mean = ops.predictive_content(mu, beta, frame)
        cov = ops.predictive_covariance_update(mu, sigma_full, beta, frame)
        mean_g = ops.predictive_content(mu_g, beta, frame_g)
        cov_g = ops.predictive_covariance_update(mu_g, sigma_g, beta, frame_g)
    torch.testing.assert_close(mean_g, (g @ mean.unsqueeze(-1)).squeeze(-1), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(cov_g, g @ cov @ g.transpose(-1, -2), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("rank", [1, 2])
def test_pure_gauge_triangle_holonomies_are_conjugate_to_shared_link(rank: int) -> None:
    r"""Plan 2.6 item 5: :math:`G_{ik}^{-1}G_{ij}G_{jk}` has the eigenvalues of :math:`e^{L}` in pure
    gauge, both for the explicit reference links and the implementation's; low-rank links do not."""
    generator = torch.Generator().manual_seed(13)
    d, N = 3, 4
    K = H * d
    results = {}
    for mode in ("pure_gauge", "lowrank"):
        ops = _make_ops(d, mode, rank, False)
        _set_maps(ops, generator, edge_scale=2.0)
        mu, _, _, beta, U, frame = _inputs(N, K, d, generator, block=False)
        F, _, _ = _chart(ops)
        ref_eigs = torch.linalg.eigvals(F)                          # complex in general
        G_ref = _reference_links(ops, mu, beta, U)
        with torch.no_grad():
            G_impl = edge_links.edge_link_matrices(ops.edge_context(mu, beta, frame))
        deviations = []
        for k in range(N):
            for j in range(k, N):
                for i in range(j, N):                       # k <= j <= i: every pair routed
                    for G in (lambda a, b: G_ref[a, b], lambda a, b: G_impl[a, b]):
                        W = torch.linalg.solve(G(i, k), G(i, j) @ G(j, k))
                        eigs = torch.linalg.eigvals(W)
                        # Symmetric Hausdorff distance between the two spectra.
                        gap = (eigs.unsqueeze(1) - ref_eigs.unsqueeze(0)).abs()     # (K, K)
                        deviations.append(float(torch.maximum(gap.min(1)[0].max(), gap.min(0)[0].max())))
        results[mode] = max(deviations)
    assert results["pure_gauge"] < 1e-9, results
    assert results["lowrank"] > 1e-3, results


def _structured_inputs(
    ops:      RelationalOperators,
    generator: torch.Generator,
    N:        int,
    large:    float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, CanonicalFrameContext]:
    r"""Low-rank factors whose :math:`B_j^\top A_i` is about :math:`K\cdot\mathrm{large}` on every
    future pair :math:`j>i` and :math:`O(1)` on causal pairs: :math:`\hat\xi_i\approx\sqrt K e_i` and
    :math:`B_j^\top A_i\approx K\,M_{ij}` with ``large`` on the strictly upper triangle of :math:`M`
    (for the first column of the factors; further columns are small and random)."""
    randn = _randn(generator)
    K, r = ops.input_dim, ops.content_link_edge_rank
    M = 0.2 * randn(K, K)
    M[torch.ones(K, K, dtype=torch.bool).triu(1)] = large          # s_ij ~ K M[i, j], large for j > i
    W_A = torch.zeros(K * r, K, dtype=F64)
    W_B = torch.zeros(K * r, K, dtype=F64)
    for k in range(K):
        W_A[k * r] = torch.eye(K, dtype=F64)[k]
        W_B[k * r] = M[k]
        for c in range(1, r):
            W_A[k * r + c] = 0.1 * randn(K)
            W_B[k * r + c] = 0.1 * randn(K)
    with torch.no_grad():
        ops.edge_receiver_map.copy_(W_A)
        ops.edge_sender_map.copy_(W_B)
    U = torch.eye(K, dtype=F64) + 0.2 * randn(N, K, K)
    xi = torch.eye(K, dtype=F64)[:N] + 0.002 * randn(N, K)
    mu = (U @ xi.unsqueeze(-1)).squeeze(-1)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(randn(H, N, N).masked_fill(~causal, -torch.inf), dim=-1)
    return mu, beta, U, CanonicalFrameContext(U, torch.linalg.inv(U))


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("residual", [False, True])
def test_lowrank_causality_with_overflowing_future_pairs(rank: int, residual: bool) -> None:
    r"""Plan 2.6 item 6: future pairs have :math:`B_j^\top A_i\approx 2400`, so an unmasked
    :math:`\varphi_1` overflows float64; outputs must stay finite, equal the reference (which never
    visits a future pair), and be unchanged by a change of a future token."""
    generator = torch.Generator().manual_seed(17)
    d, N = 3, 4
    K = H * d
    ops = _make_ops(d, "lowrank", rank, residual)
    _set_maps(ops, generator)
    mu, beta, U, frame = _structured_inputs(ops, generator, N, large=400.0)
    # Confirm the regime: at least one future pair overflows a bare matrix exponential.
    W_A, W_B = ops.edge_receiver_map.detach(), ops.edge_sender_map.detach()
    xi = torch.linalg.solve(U, mu.unsqueeze(-1)).squeeze(-1)
    xi_hat = xi / torch.sqrt((xi ** 2).mean(-1, keepdim=True) + 1e-12)
    A3 = (W_A @ xi_hat[0]).reshape(K, rank)
    B3 = (W_B @ xi_hat[N - 1]).reshape(K, rank)
    assert not torch.isfinite(torch.linalg.matrix_exp(A3 @ B3.T)).all()
    sigma = torch.rand(N, K, generator=generator, dtype=F64) + 0.3
    sigma_full = torch.diag_embed(sigma)
    mean_ref, cov_ref = _reference_moments(ops, mu, sigma_full, beta, U, residual)
    with torch.no_grad():
        mean = ops.predictive_content(mu, beta, frame)
        cov = ops.predictive_covariance_update(mu, sigma_full, beta, frame)
    assert torch.isfinite(mean).all() and torch.isfinite(cov).all()
    torch.testing.assert_close(mean, mean_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(cov, cov_ref, atol=1e-9, rtol=1e-9)
    # Change the last token only (staying in the regime, so its own routed pairs stay finite);
    # every earlier receiver is unchanged.
    mu2 = mu.clone()
    mu2[N - 1] = U[N - 1] @ (torch.eye(K, dtype=F64)[N - 1] + 0.002 * torch.randn(K, generator=generator, dtype=F64))
    with torch.no_grad():
        mean2 = ops.predictive_content(mu2, beta, frame)
        cov2 = ops.predictive_covariance_update(mu2, sigma_full, beta, frame)
    torch.testing.assert_close(mean2[:N - 1], mean[:N - 1], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov2[:N - 1], cov[:N - 1], atol=1e-12, rtol=1e-12)
    assert (mean2[N - 1] - mean[N - 1]).abs().max() > 1e-6


@pytest.mark.parametrize("mode,rank", CASES)
def test_causality_with_large_but_finite_factors(mode: str, rank: int) -> None:
    r"""Plan 2.6 item 6 for both modes at large finite factor norms (pure-gauge coefficients are per
    sender, so overflow cannot be confined to future pairs there): a change of :math:`\mu_j` for
    :math:`j>i` leaves receiver :math:`i` unchanged, and the outputs equal the reference."""
    generator = torch.Generator().manual_seed(19)
    d, N = 3, 4
    K = H * d
    ops = _make_ops(d, mode, rank, True)
    _set_maps(ops, generator, edge_scale=6.0)
    mu, sigma_diag, _, beta, U, frame = _inputs(N, K, d, generator, block=True)
    mean_ref, cov_ref = _reference_moments(ops, mu, torch.diag_embed(sigma_diag), beta, U, True)
    with torch.no_grad():
        mean = ops.predictive_content(mu, beta, frame)
        cov = ops.predictive_covariance_update(mu, sigma_diag, beta, frame)
    assert torch.isfinite(mean).all() and torch.isfinite(cov).all()
    torch.testing.assert_close(mean, mean_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(cov, torch.diagonal(cov_ref, dim1=-2, dim2=-1), atol=1e-9, rtol=1e-9)
    mu2 = mu.clone()
    mu2[1:] = mu2[1:] + 0.7 * torch.randn(N - 1, K, generator=generator, dtype=F64)
    with torch.no_grad():
        mean2 = ops.predictive_content(mu2, beta, frame)
        cov2 = ops.predictive_covariance_update(mu2, sigma_diag, beta, frame)
    torch.testing.assert_close(mean2[0], mean[0], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov2[0], cov[0], atol=1e-12, rtol=1e-12)
