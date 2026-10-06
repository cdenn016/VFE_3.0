r"""Independent brute-force reference for the Level 2 latent receiver factors (plan Section 3, 2026-10-05).

Written from the plan's definitions and the public signatures of ``vfe3/model/edge_links.py`` only,
without reading ``tests/test_latent_links_20261005.py`` or the body of ``infer_latent_receivers``.
Everything here is float64 at :math:`K\le 6`, :math:`N\le 4`, :math:`H = 2`, :math:`r\in\{1,2\}`, and every
link is an explicit matrix.

Links. For receiver factors :math:`A\in\mathbb R^{N\times K\times r}` (any value, not only the amortized
:math:`A^0_i = \mathrm{reshape}(W_A\hat\xi_i)`), sender factors :math:`B_j = \mathrm{reshape}(W_B\hat\xi_j)`
and the chart :math:`F = e^{OT}`, :math:`R = T^{-1}`, :math:`\Pi_h = R_hT_h`,

.. math::
    G_{ij}(A_i) = F\exp(A_iB_j^\top),\qquad M^{ij}_h = G_{ij}\Pi_h .

Energy. The kernel :math:`X = r\,k_i + \sum_h M^{iJ^h}_h k_{J^h} + \epsilon` with independent labels
:math:`J^h\sim\beta^h_i` and sources :math:`k_j\sim N(\xi_j, C_j)`, :math:`C_j = U_j^{-1}\Sigma_jU_j^{-\top}`,
has (law of total covariance, as in the Level 1 reference) the reference covariance

.. math::
    C_i^{\mathrm{att}} = rC_i + r\sum_h\beta^h_{ii}(M^{ii}_hC_i + C_iM^{ii\top}_h)
      + \sum_j\sum_{h,h'}c^{hh'}_{ij}M^{ij}_hC_jM^{ij\top}_{h'}
      + \sum_h\Bigl[\sum_j\beta^h_{ij}n^h_{ij}n^{h\top}_{ij} - \bar n^h_i\bar n^{h\top}_i\Bigr],
    \qquad n^h_{ij} = M^{ij}_h\xi_j,

with no floor. The plan's per-receiver MAP energy is

.. math::
    \mathcal F_i(A_i) = \frac{\operatorname{tr}C^{\mathrm{att}}_i(A_i)}{2c_i}
      + \frac{\|A_i - A^0_i\|_F^2}{2\sigma_L^2}
      + \lambda_W\sum_{(j,k):\,i-w\le k<j\le i}\bar\pi_{ij}\bar\pi_{jk}\,\tfrac12\|W_{ijk} - I\|_F^2,
    \qquad c_i = \operatorname{tr}C_i/K,

with the loops built from explicit matrices of both layers,
:math:`W_{ijk} = (G^{l+1}_{ik}G^l_{kk})^{-1}G^{l+1}_{ij}G^l_{jk}`, :math:`G^l_{ab} = F_l\exp(a_ab_b^\top)`
(the previous layer's chart and factors), and :math:`\bar\pi` the head mean of ``softmax(log_prior)``
rows (causal uniform without a prior). The gradient is taken by central finite differences and, as a
second independent route, by autograd through these explicit formulas; the damped fixed point
:math:`A\leftarrow(1-\rho)A + \rho(A^0 - \sigma_L^2\nabla[E_i + \lambda_W\mathrm{Loop}_i])` is iterated
here on the reference energy. The implementation is compared on the energy, the gradient, the inferred
factors, the final mean and covariance (local coordinates, floor included) and the recorded statistics.
"""
import math
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model import edge_links
from vfe3.model.edge_links import LinkState, LoopInputs
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators

torch.set_num_threads(1)
F64 = torch.float64
FLOOR = 1e-3
H = 2
Loop = Tuple[int, int, int]
KEYS = edge_links.LATENT_DIAGNOSTIC_KEYS


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
    rank:     int,
    residual: bool,
    *,
    latent:      bool = True,
    sigma_L:     float = 0.05,
    steps:       int = 1,
    damping:     float = 1.0,
    loop_weight: float = 0.0,
    window:      Optional[int] = 1,
    seed:        int = 11,
) -> RelationalOperators:
    r"""A low-rank edge operator in float64, with or without latent inference."""
    return RelationalOperators(
        [d] * H, seed, readout_mode="predictive_content", attention_residual=residual,
        predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
        content_link_chart="exponential", content_link_edges="lowrank", content_link_edge_rank=rank,
        content_link_latent=latent, content_link_latent_steps=steps, content_link_latent_sigma=sigma_L,
        content_link_latent_damping=damping, content_link_loop_weight=loop_weight,
        content_link_loop_window=window,
    ).double()


def _set_maps(ops: RelationalOperators, generator: torch.Generator, T_scale: float = 0.3,
              edge_scale: float = 1.0) -> None:
    r"""Random T, O (near identity) and random nonzero W_A, W_B."""
    randn = _randn(generator)
    K = ops.input_dim
    with torch.no_grad():
        ops.T.copy_(ops.T + T_scale * randn(*ops.T.shape) / K ** 0.5)
        ops.O.copy_(ops.O + T_scale * randn(*ops.O.shape) / K ** 0.5)
        ops.edge_receiver_map.copy_(edge_scale * randn(*ops.edge_receiver_map.shape) / K)
        ops.edge_sender_map.copy_(edge_scale * randn(*ops.edge_sender_map.shape) / K)


def _twin(ops: RelationalOperators) -> RelationalOperators:
    r"""The same operator with latent inference off (Level 1), sharing every parameter value."""
    twin = _make_ops(ops.irrep_dims[0], ops.content_link_edge_rank, ops.attention_residual, latent=False)
    twin.load_state_dict(ops.state_dict())
    return twin


def _inputs(
    N:        int,
    K:        int,
    d:        int,
    generator: torch.Generator,
    block:    bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CanonicalFrameContext, torch.Tensor]:
    r"""Random means (token 0 shifted), diagonal and full covariances, causal routing, frames, a causal
    routing log-prior ``(H, N, N)``."""
    randn = _randn(generator)
    mu = randn(N, K)
    mu[0] = mu[0] + 3.0
    sigma_diag = torch.rand(N, K, generator=generator, dtype=F64) + 0.3
    factor = randn(N, K, K)
    sigma_full = factor @ factor.transpose(-1, -2) / K + 0.5 * torch.eye(K, dtype=F64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(randn(H, N, N).masked_fill(~causal, -torch.inf), dim=-1)
    log_prior = randn(H, N, N).masked_fill(~causal, -torch.inf)
    if block:
        blocks = torch.eye(d, dtype=F64) + 0.3 * randn(N, H, d, d)
        frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
        U = _block_dense(blocks)
    else:
        U = torch.eye(K, dtype=F64) + 0.3 * randn(N, K, K)
        frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    return mu, sigma_diag, sigma_full, beta, U, frame, log_prior


def _previous_state(generator: torch.Generator, N: int, K: int, r: int, scale: float = 0.7) -> LinkState:
    r"""A random previous layer: :math:`F_l = e^{L_l}`, its inverse, inferred factors ``a`` and ``b``."""
    randn = _randn(generator)
    L = 0.3 * randn(K, K) / K ** 0.5
    return LinkState(torch.linalg.matrix_exp(L), torch.linalg.matrix_exp(-L),
                     scale * randn(N, K, r) / K ** 0.5, scale * randn(N, K, r) / K ** 0.5)


# --------------------------------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------------------------------

def _chart(ops: RelationalOperators) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    r"""Float64 :math:`F = e^{OT}`, :math:`R = T^{-1}`, :math:`\Pi_h = R_hT_h` and :math:`L = OT`."""
    T = ops.T.detach().to(F64)                                      # (H, d, K)
    O = ops.O.detach().to(F64)                                      # (H, K, d)
    n_heads, d, K = T.shape
    T_stack = T.reshape(K, K)
    O_stack = O.permute(1, 0, 2).reshape(K, K)
    L = O_stack @ T_stack
    R = torch.linalg.inv(T_stack)
    Pi = [R[:, h*d:(h+1)*d] @ T[h] for h in range(n_heads)]
    return torch.linalg.matrix_exp(L), R, Pi, L


def _reference_factors(ops: RelationalOperators, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r""":math:`A^0_i = \mathrm{reshape}(W_A\hat\xi_i, (K, r))`, :math:`B_j = \mathrm{reshape}(W_B\hat\xi_j, (K, r))`."""
    N, K = xi.shape
    r = ops.content_link_edge_rank
    xi_hat = xi / torch.sqrt((xi ** 2).mean(dim=-1, keepdim=True) + 1e-12)
    W_A = ops.edge_receiver_map.detach().to(F64)
    W_B = ops.edge_sender_map.detach().to(F64)
    A0 = torch.stack([(W_A @ xi_hat[i]).reshape(K, r) for i in range(N)])
    B = torch.stack([(W_B @ xi_hat[j]).reshape(K, r) for j in range(N)])
    return A0, B


def _links(
    F:  torch.Tensor,          # (K, K)
    Pi: List[torch.Tensor],    # H x (K, K)
    A:  torch.Tensor,          # (N, K, r)
    B:  torch.Tensor,          # (N, K, r)
    routed: torch.Tensor,      # (N, N) bool
    U: Optional[torch.Tensor] = None,       # (N, K, K): local maps U_i M U_j^{-1} when given
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    r""":math:`M^{ij}_h = F\exp(A_iB_j^\top)\Pi_h` on routed pairs (local coordinates when ``U`` is given)."""
    N = A.shape[0]
    U_inv = None if U is None else torch.linalg.inv(U)
    M = {}
    for i in range(N):
        for j in range(N):
            if not bool(routed[i, j]):
                continue
            G = F @ torch.linalg.matrix_exp(A[i] @ B[j].T)
            for h, P in enumerate(Pi):
                M[h, i, j] = G @ P if U is None else U[i] @ G @ P @ U_inv[j]
    return M


def _kernel_moments(
    M:     Dict[Tuple[int, int, int], torch.Tensor],   # (h, i, j) -> (K, K)
    x:     torch.Tensor,     # (N, K) source means
    S:     torch.Tensor,     # (N, K, K) source covariances
    beta:  torch.Tensor,     # (H, N, N)
    residual: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Mean (without the residual) and covariance (without floor) of the kernel, explicit loops over j, h, h'."""
    N, K = x.shape
    r = 1.0 if residual else 0.0
    means, covs = [], []
    for i in range(N):
        senders = sorted({j for (h, ii, j) in M if ii == i})
        n = {(h, j): M[h, i, j] @ x[j] for h in range(H) for j in senders}
        n_bar = [sum(beta[h, i, j] * n[h, j] for j in senders) for h in range(H)]
        cov = r * S[i]
        if i in senders:
            for h in range(H):
                cov = cov + r * beta[h, i, i] * (M[h, i, i] @ S[i] + S[i] @ M[h, i, i].T)
        for j in senders:
            for h in range(H):
                for hp in range(H):
                    c = beta[h, i, j] if h == hp else beta[h, i, j] * beta[hp, i, j]
                    cov = cov + c * M[h, i, j] @ S[j] @ M[hp, i, j].T
        for h in range(H):
            cov = cov + sum(beta[h, i, j] * torch.outer(n[h, j], n[h, j]) for j in senders)
            cov = cov - torch.outer(n_bar[h], n_bar[h])
        means.append(sum(n_bar))
        covs.append(cov)
    return torch.stack(means), torch.stack(covs)


def _reference_transport_energy(
    ops:  RelationalOperators,
    A:    torch.Tensor,      # (N, K, r), may require grad
    B:    torch.Tensor,      # (N, K, r)
    xi:   torch.Tensor,      # (N, K)
    C:    torch.Tensor,      # (N, K, K)
    beta: torch.Tensor,      # (H, N, N)
    residual: bool,
) -> torch.Tensor:           # (N,) E_i
    r""":math:`E_i = \operatorname{tr}C_i^{\mathrm{att}}(A_i)/(2c_i)`, :math:`c_i = \operatorname{tr}C_i/K`, no floor."""
    F, _, Pi, _ = _chart(ops)
    M = _links(F, Pi, A, B, (beta != 0).any(dim=0))
    _, cov = _kernel_moments(M, xi, C, beta, residual)
    K = xi.shape[-1]
    c = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / K
    return torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1) / (2.0 * c)


def _reference_prior_weights(log_prior: Optional[torch.Tensor], N: int) -> torch.Tensor:   # (N, N)
    r"""Head mean of the ``softmax(log_prior)`` rows; causal uniform without a prior."""
    if log_prior is None:
        causal = torch.ones(N, N, dtype=F64).tril()
        return causal / causal.sum(-1, keepdim=True)
    return torch.softmax(log_prior.to(F64), dim=-1).reshape(-1, N, N).mean(0)


def _loop_set(N: int, window: Optional[int]) -> List[Loop]:
    r"""Every :math:`(i, j, k)` with :math:`i - w\le k<j\le i` (all :math:`k\ge 0` when ``window`` is None)."""
    return [(i, j, k) for i in range(N) for j in range(i + 1) for k in range(j)
            if window is None or i - window <= k]


def _reference_loop_matrices(
    F_next:   torch.Tensor,   # (K, K) this layer's e^{L}
    A:        torch.Tensor,   # (N, K, r) this layer's receiver factors
    B:        torch.Tensor,   # (N, K, r) this layer's sender factors
    previous: LinkState,
    loops:    List[Loop],
) -> Dict[Loop, torch.Tensor]:
    r""":math:`W_{ijk} = (G^{l+1}_{ik}G^l_{kk})^{-1}G^{l+1}_{ij}G^l_{jk}` from explicit links of both layers."""
    a, b, F_prev = previous.A.to(F64), previous.B.to(F64), previous.F.to(F64)
    G_next = lambda s, t: F_next @ torch.linalg.matrix_exp(A[s] @ B[t].T)
    G_prev = lambda s, t: F_prev @ torch.linalg.matrix_exp(a[s] @ b[t].T)
    return {(i, j, k): torch.linalg.solve(G_next(i, k) @ G_prev(k, k), G_next(i, j) @ G_prev(j, k))
            for (i, j, k) in loops}


def _reference_loop_energy(
    F_next:   torch.Tensor,
    A:        torch.Tensor,
    B:        torch.Tensor,
    previous: LinkState,
    weights:  torch.Tensor,   # (N, N) pibar
    window:   Optional[int],
) -> torch.Tensor:            # (N,) Loop_i
    r""":math:`\mathrm{Loop}_i = \sum_{(j,k)}\bar\pi_{ij}\bar\pi_{jk}\tfrac12\|W_{ijk} - I\|_F^2`."""
    N, K = A.shape[0], A.shape[1]
    eye = torch.eye(K, dtype=F64)
    W = _reference_loop_matrices(F_next, A, B, previous, _loop_set(N, window))
    out = [torch.zeros((), dtype=F64) for _ in range(N)]
    for (i, j, k), Wm in W.items():
        out[i] = out[i] + weights[i, j] * weights[j, k] * 0.5 * ((Wm - eye) ** 2).sum()
    return torch.stack(out)


class _Case(NamedTuple):
    ops:        RelationalOperators
    twin:       RelationalOperators
    mu:         torch.Tensor
    sigma:      torch.Tensor          # (N, K) or (N, K, K), what the operator receives
    sigma_dense: torch.Tensor         # (N, K, K)
    beta:       torch.Tensor
    U:          torch.Tensor
    frame:      CanonicalFrameContext
    log_prior:  torch.Tensor
    xi:         torch.Tensor
    C:          torch.Tensor
    A0:         torch.Tensor
    B:          torch.Tensor
    ctx0:       edge_links.EdgeContext
    sources:    edge_links.LatentSources
    previous:   LinkState
    weights:    torch.Tensor
    loop:       LoopInputs


def _build(
    d: int, rank: int, residual: bool, generator: torch.Generator, *,
    block: bool = False, family: str = "full", N: int = 4, edge_scale: float = 1.0, **ops_kw,
) -> _Case:
    r"""Operator, Level 1 twin, inputs, reference coordinates and the implementation's amortized context."""
    K = H * d
    ops = _make_ops(d, rank, residual, **ops_kw)
    _set_maps(ops, generator, edge_scale=edge_scale)
    twin = _twin(ops)
    mu, sigma_diag, sigma_full, beta, U, frame, log_prior = _inputs(N, K, d, generator, block)
    sigma = sigma_diag if family == "diag" else sigma_full
    sigma_dense = torch.diag_embed(sigma_diag) if family == "diag" else sigma_full
    U_inv = torch.linalg.inv(U)
    xi = (U_inv @ mu.unsqueeze(-1)).squeeze(-1)
    C = U_inv @ sigma_dense @ U_inv.transpose(-1, -2)
    A0, B = _reference_factors(ops, xi)
    previous = _previous_state(generator, N, K, rank)
    weights = _reference_prior_weights(log_prior, N)
    with torch.no_grad():
        ctx0 = twin.edge_context(mu, beta, frame)
        sources = edge_links.latent_sources(ctx0, beta, twin.canonical_mean(mu, frame), C, residual)
    loop = LoopInputs(previous, weights, ops.content_link_loop_window)
    return _Case(ops, twin, mu, sigma, sigma_dense, beta, U, frame, log_prior, xi, C, A0, B, ctx0,
                 sources, previous, weights, loop)


def _reference_link_energy(case: _Case, A: torch.Tensor, loop_weight: float, window: Optional[int]) -> torch.Tensor:
    r""":math:`E_i + \lambda_W\mathrm{Loop}_i` at ``A`` (reference)."""
    F, _, _, _ = _chart(case.ops)
    energy = _reference_transport_energy(case.ops, A, case.B, case.xi, case.C, case.beta, case.ops.attention_residual)
    if loop_weight > 0.0:
        energy = energy + loop_weight * _reference_loop_energy(F, A, case.B, case.previous, case.weights, window)
    return energy


def _reference_energy(case: _Case, A: torch.Tensor, sigma_L: float, loop_weight: float,
                      window: Optional[int]) -> torch.Tensor:
    r"""The full :math:`\mathcal F_i(A_i)`."""
    return (_reference_link_energy(case, A, loop_weight, window)
            + ((A - case.A0) ** 2).sum(dim=(-2, -1)) / (2.0 * sigma_L ** 2))


def _reference_inference(
    link_energy: Callable[[torch.Tensor], torch.Tensor],   # A -> (N,) E_i + lambda_W Loop_i
    A0:      torch.Tensor,
    sigma_L: float,
    steps:   int,
    damping: float,
) -> torch.Tensor:
    r""":math:`A^{(t)} = (1-\rho)A^{(t-1)} + \rho(A^0 - \sigma_L^2\nabla[E + \lambda_W\mathrm{Loop}](A^{(t-1)}))`."""
    A = A0.clone()
    for _ in range(steps):
        A_var = A.detach().clone().requires_grad_(True)
        grad = torch.autograd.grad(link_energy(A_var).sum(), A_var)[0]
        A = (1.0 - damping) * A + damping * (A0 - sigma_L ** 2 * grad)
    return A.detach()


def _fd_gradient(f: Callable[[torch.Tensor], torch.Tensor], A: torch.Tensor, h: float = 1e-5) -> torch.Tensor:
    r"""Central finite differences of the scalar ``f`` with respect to every entry of ``A``."""
    base = A.detach().clone()
    grad = torch.zeros_like(base)
    for idx in range(base.numel()):
        plus, minus = base.clone().view(-1), base.clone().view(-1)
        plus[idx] += h
        minus[idx] -= h
        grad.view(-1)[idx] = (f(plus.view_as(base)) - f(minus.view_as(base))) / (2.0 * h)
    return grad


def _reference_outputs(case: _Case, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Local mean (without the residual) and moment-matched covariance (floor included) at factors ``A``."""
    F, _, Pi, _ = _chart(case.ops)
    N, K = case.mu.shape
    M = _links(F, Pi, A, case.B, (case.beta != 0).any(dim=0), U=case.U)
    mean, cov = _kernel_moments(M, case.mu, case.sigma_dense, case.beta, case.ops.attention_residual)
    trace = torch.diagonal(case.C, dim1=-2, dim2=-1).sum(-1)                   # (N,)
    cov = cov + (FLOOR * trace / K)[:, None, None] * (case.U @ case.U.transpose(-1, -2))
    return mean, cov


def _impl_outputs(case: _Case, ctx: edge_links.EdgeContext) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mean = case.ops.predictive_content(case.mu, case.beta, case.frame, edges=ctx)
        cov = case.ops.predictive_covariance_update(case.mu, case.sigma, case.beta, case.frame, edges=ctx)
    return mean, cov


def _expected(case: _Case, cov: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(cov, dim1=-2, dim2=-1) if case.sigma.dim() == 2 else cov


# --------------------------------------------------------------------------------------------------
# Self-checks of the reference
# --------------------------------------------------------------------------------------------------

def test_reference_gradient_routes_agree() -> None:
    r"""Finite differences and autograd through the explicit reference agree (loops on)."""
    generator = torch.Generator().manual_seed(1)
    case = _build(3, 2, True, generator, loop_weight=0.7, window=None)
    A = case.A0 + 0.3 * torch.randn(*case.A0.shape, generator=generator, dtype=F64)
    f = lambda X: _reference_energy(case, X, 0.05, 0.7, None).sum()
    A_var = A.clone().requires_grad_(True)
    grad_auto = torch.autograd.grad(f(A_var), A_var)[0]
    grad_fd = _fd_gradient(f, A)
    torch.testing.assert_close(grad_auto, grad_fd, atol=1e-6, rtol=1e-6)


def test_reference_transport_energy_is_the_level_one_trace() -> None:
    r"""The reference energy equals the Level 1 reference covariance trace pulled back to reference
    coordinates, floor removed; and the amortized factors of the implementation equal the reference's."""
    generator = torch.Generator().manual_seed(2)
    case = _build(3, 2, True, generator)
    torch.testing.assert_close(case.ctx0.A, case.A0, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(case.ctx0.B, case.B, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(case.sources.scale, torch.diagonal(case.C, dim1=-2, dim2=-1).sum(-1) / case.mu.shape[-1],
                               atol=1e-12, rtol=1e-12)
    _, cov_local = _reference_outputs(case, case.A0)
    K = case.mu.shape[-1]
    U_inv = torch.linalg.inv(case.U)
    trace_C = torch.diagonal(case.C, dim1=-2, dim2=-1).sum(-1)
    cov_ref = U_inv @ cov_local @ U_inv.transpose(-1, -2) - (FLOOR * trace_C / K)[:, None, None] * torch.eye(K, dtype=F64)
    E_from_level_one = torch.diagonal(cov_ref, dim1=-2, dim2=-1).sum(-1) / (2.0 * trace_C / K)
    E = _reference_transport_energy(case.ops, case.A0, case.B, case.xi, case.C, case.beta, True)
    torch.testing.assert_close(E, E_from_level_one, atol=1e-10, rtol=1e-10)
    # Level 1 outputs of the twin equal the reference outputs at A^0 (the Level 1 reference, re-derived).
    mean_impl, cov_impl = _impl_outputs(case, case.ctx0)
    mean_ref, cov_ref_local = _reference_outputs(case, case.A0)
    torch.testing.assert_close(mean_impl, mean_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cov_impl, _expected(case, cov_ref_local), atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------------------------------------
# Energy, gradient, inference
# --------------------------------------------------------------------------------------------------

CASES = [(rank, residual) for rank in (1, 2) for residual in (False, True)]


@pytest.mark.parametrize("rank,residual", CASES)
@pytest.mark.parametrize("loop_weight", [0.0, 0.7])
def test_energy_matches_reference_at_random_factors(rank: int, residual: bool, loop_weight: float) -> None:
    r"""``latent_transport_energy``, ``loop_energy`` and ``latent_receiver_energy`` at random :math:`A\ne A^0`."""
    generator = torch.Generator().manual_seed(3 + rank)
    case = _build(3, rank, residual, generator, loop_weight=loop_weight, window=None)
    A = case.A0 + 0.3 * torch.randn(*case.A0.shape, generator=generator, dtype=F64)
    F, _, _, _ = _chart(case.ops)
    with torch.no_grad():
        E = edge_links.latent_transport_energy(A, case.ctx0, case.beta, case.sources)
        loop = edge_links.loop_energy(A, case.ctx0.B, case.previous, case.weights, None)
        total = edge_links.latent_receiver_energy(A, case.ctx0.A, case.ctx0, case.beta, case.sources, 0.05,
                                                  loop_weight, case.loop if loop_weight > 0 else None)
    E_ref = _reference_transport_energy(case.ops, A, case.B, case.xi, case.C, case.beta, residual)
    loop_ref = _reference_loop_energy(F, A, case.B, case.previous, case.weights, None)
    total_ref = _reference_energy(case, A, 0.05, loop_weight, None)
    assert E.shape == loop.shape == total.shape == (case.mu.shape[0],)
    torch.testing.assert_close(E, E_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(loop, loop_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(total, total_ref, atol=1e-10, rtol=1e-10)
    assert float(loop_ref.abs().sum()) > 1e-4                       # the loop term is not trivially zero here


@pytest.mark.parametrize("rank,residual", CASES)
@pytest.mark.parametrize("loop_weight", [0.0, 0.7])
def test_gradient_matches_finite_differences(rank: int, residual: bool, loop_weight: float) -> None:
    r"""``autograd.grad`` of the implementation's energy equals the central-difference gradient of the
    reference and the reference's own autograd; receivers decouple."""
    generator = torch.Generator().manual_seed(5 + rank)
    case = _build(3, rank, residual, generator, loop_weight=loop_weight, window=None)
    A = case.A0 + 0.3 * torch.randn(*case.A0.shape, generator=generator, dtype=F64)
    loop = case.loop if loop_weight > 0 else None
    A_var = A.clone().requires_grad_(True)
    energy = edge_links.latent_receiver_energy(A_var, case.ctx0.A, case.ctx0, case.beta, case.sources, 0.05,
                                               loop_weight, loop)
    grad_impl = torch.autograd.grad(energy.sum(), A_var, retain_graph=True)[0]
    f = lambda X: _reference_energy(case, X, 0.05, loop_weight, None).sum()
    grad_fd = _fd_gradient(f, A)
    A_ref = A.clone().requires_grad_(True)
    grad_ref = torch.autograd.grad(f(A_ref), A_ref)[0]
    torch.testing.assert_close(grad_impl, grad_fd, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad_impl, grad_ref, atol=1e-9, rtol=1e-9)
    # Receiver 1's energy depends on A_1 only.
    partial = torch.autograd.grad(energy[1], A_var)[0]
    assert float(partial[1].abs().max()) > 0.0
    assert float(torch.cat([partial[:1], partial[2:]]).abs().max()) == 0.0


@pytest.mark.parametrize("steps,damping", [(1, 1.0), (3, 1.0), (3, 0.5)])
@pytest.mark.parametrize("loop_weight", [0.0, 0.7])
def test_inference_matches_damped_fixed_point(steps: int, damping: float, loop_weight: float) -> None:
    r"""``infer_latent_receivers`` equals the reference iteration and reports the plan's statistics."""
    generator = torch.Generator().manual_seed(7 + steps)
    case = _build(3, 2, True, generator, loop_weight=loop_weight, window=None, steps=steps, damping=damping)
    loop = case.loop if loop_weight > 0 else None
    stats: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        A_impl = edge_links.infer_latent_receivers(case.ctx0, case.beta, case.sources, sigma=0.05, steps=steps,
                                                   damping=damping, loop_weight=loop_weight, loop=loop, stats=stats)
    A_ref = _reference_inference(lambda X: _reference_link_energy(case, X, loop_weight, None),
                                 case.A0, 0.05, steps, damping)
    torch.testing.assert_close(A_impl, A_ref, atol=1e-9, rtol=1e-9)
    assert float((A_ref - case.A0).abs().max()) > 1e-4                 # the inference moved the factors
    assert set(stats) == set(KEYS)
    F, _, _, _ = _chart(case.ops)
    energy_0 = _reference_energy(case, case.A0, 0.05, loop_weight, None)
    energy_T = _reference_energy(case, A_ref, 0.05, loop_weight, None)
    loop_T = _reference_loop_energy(F, A_ref, case.B, case.previous, case.weights, None)
    shift = (A_ref - case.A0).flatten(1).norm(dim=1) / case.A0.flatten(1).norm(dim=1)
    torch.testing.assert_close(stats["relational_latent_energy_initial"].to(F64), energy_0.mean(), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(stats["relational_latent_energy_final"].to(F64), energy_T.mean(), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(stats["relational_latent_receiver_shift"].to(F64), shift.mean(), atol=1e-9, rtol=1e-9)
    loop_stat = stats["relational_latent_loop_energy"].to(F64)
    if loop_weight > 0:
        torch.testing.assert_close(loop_stat, loop_T.mean(), atol=1e-9, rtol=1e-9)
    else:
        assert torch.isnan(loop_stat)


@pytest.mark.parametrize("rank,residual", CASES)
@pytest.mark.parametrize("block,family", [(True, "diag"), (True, "full"), (False, "diag"), (False, "full")])
def test_public_path_matches_reference_outputs(rank: int, residual: bool, block: bool, family: str) -> None:
    r"""``edge_context`` with latent inference and loops: inferred factors, mean, covariance, link state."""
    generator = torch.Generator().manual_seed(11 + rank)
    case = _build(3, rank, residual, generator, block=block, family=family, loop_weight=0.7, window=2, steps=2)
    link_state = {"previous": case.previous}
    stats: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ctx = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                    link_state=link_state, latent_out=stats)
    A_ref = _reference_inference(lambda X: _reference_link_energy(case, X, 0.7, 2), case.A0, 0.05, 2, 1.0)
    torch.testing.assert_close(ctx.A, A_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(ctx.B, case.B, atol=1e-12, rtol=1e-12)
    mean_impl, cov_impl = _impl_outputs(case, ctx)
    mean_ref, cov_ref = _reference_outputs(case, A_ref)
    torch.testing.assert_close(mean_impl, mean_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(cov_impl, _expected(case, cov_ref), atol=1e-9, rtol=1e-9)
    # The outputs differ from Level 1 (the inference matters at this sigma_L).
    mean_l1, cov_l1 = _impl_outputs(case, case.ctx0)
    assert float((mean_l1 - mean_impl).abs().max()) > 1e-6 * float(mean_impl.abs().max())
    assert float((cov_l1 - cov_impl).abs().max()) > 1e-6 * float(cov_impl.abs().max())
    # The written link state is this layer's (F, e^{-L}, A^{(T)}, B).
    F, _, _, L = _chart(case.ops)
    current = link_state["current"]
    torch.testing.assert_close(current.F.to(F64), F, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(current.F_inv.to(F64), torch.linalg.matrix_exp(-L), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(current.A, ctx.A, atol=0, rtol=0)
    torch.testing.assert_close(current.B, ctx.B, atol=0, rtol=0)
    assert link_state["previous"] is case.previous
    # Statistics from the public path.
    energy_T = _reference_energy(case, A_ref, 0.05, 0.7, 2)
    torch.testing.assert_close(stats["relational_latent_energy_final"].to(F64), energy_T.mean(), atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(stats["relational_latent_energy_initial"].to(F64),
                               _reference_energy(case, case.A0, 0.05, 0.7, 2).mean(), atol=1e-9, rtol=1e-9)


def test_prior_routing_weights_match_reference() -> None:
    generator = torch.Generator().manual_seed(13)
    N = 4
    randn = _randn(generator)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    for log_prior in (randn(H, N, N).masked_fill(~causal, -torch.inf), randn(N, N).masked_fill(~causal, -torch.inf), None):
        impl = edge_links.prior_routing_weights(log_prior, N, dtype=F64, device=torch.device("cpu"))
        torch.testing.assert_close(impl, _reference_prior_weights(log_prior, N), atol=1e-12, rtol=1e-12)


# --------------------------------------------------------------------------------------------------
# Limits and structure
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rank,residual", CASES)
@pytest.mark.parametrize("damping", [1.0, 0.3])
def test_static_limit_equals_level_one(rank: int, residual: bool, damping: float) -> None:
    r"""Plan 3.6 item 1: :math:`\sigma_L = 10^{-12}` reproduces the Level 1 twin to roundoff."""
    generator = torch.Generator().manual_seed(17 + rank)
    case = _build(3, rank, residual, generator, block=True, family="diag", sigma_L=1e-12, steps=3,
                  damping=damping, loop_weight=0.7, window=None)
    with torch.no_grad():
        ctx = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                    link_state={"previous": case.previous})
    torch.testing.assert_close(ctx.A, case.ctx0.A, atol=1e-12, rtol=1e-12)
    mean, cov = _impl_outputs(case, ctx)
    mean_l1, cov_l1 = _impl_outputs(case, case.ctx0)
    torch.testing.assert_close(mean, mean_l1, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov, cov_l1, atol=1e-12, rtol=1e-12)


def _overflowing_future(case: _Case, generator: torch.Generator, split: int = 2, big: float = 20.0,
                        small: float = 0.0125) -> _Case:
    r"""Factor maps and means on which every future pair of an early receiver overflows while every causal
    link stays :math:`O(1)`. With :math:`\xi_i = e_i` exactly (:math:`\hat\xi_i = \sqrt K e_i`), the receiver
    factors are :math:`A_i = s_i\sqrt K(e_i, 0.1\,\epsilon_i e_i)` (supported on coordinate i) and the sender
    factors :math:`B_j = t_j\sqrt K(m_j, 0.1\,\epsilon'_j)` with :math:`s_i = \mathrm{big}` for
    :math:`i < \mathrm{split}` and ``small`` after, :math:`t_j` the reverse, and :math:`m_j` a random vector
    whose entries on the early coordinates are :math:`\tfrac12` for late :math:`j`. Then
    :math:`(B_j^\top A_i)_{00} = K s_it_j(m_j)_i`: about :math:`1200` on the future pairs
    :math:`i < \mathrm{split}\le j` (a bare :math:`\varphi_1` overflows float64) and :math:`s_it_j\le 0.25`
    times :math:`O(1)` on causal pairs, where also :math:`\|A_i\|\|B_j\|\lesssim 1`, so the links and the
    iteration are well scaled."""
    randn = _randn(generator)
    N, K = case.mu.shape
    r = case.ops.content_link_edge_rank
    early = torch.arange(K) < split
    s = torch.where(early, torch.tensor(big, dtype=F64), torch.tensor(small, dtype=F64))     # (K,) receiver scales
    t = torch.where(early, torch.tensor(small, dtype=F64), torch.tensor(big, dtype=F64))     # (K,) sender scales
    M = 0.3 * randn(K, K)
    M[:split, split:] = 0.5                                             # (m_j)_i for early i, late j
    W_A = torch.zeros(K * r, K, dtype=F64)
    W_B = torch.zeros(K * r, K, dtype=F64)
    for k in range(K):
        W_A[k * r] = s[k] * torch.eye(K, dtype=F64)[k]
        W_B[k * r] = M[k] * t
        for c in range(1, r):
            W_A[k * r + c] = 0.1 * s[k] * randn(1) * torch.eye(K, dtype=F64)[k]
            W_B[k * r + c] = 0.1 * randn(K) * t
    with torch.no_grad():
        for ops in (case.ops, case.twin):
            ops.edge_receiver_map.copy_(W_A)
            ops.edge_sender_map.copy_(W_B)
    xi = torch.eye(K, dtype=F64)[:N]
    mu = (case.U @ xi.unsqueeze(-1)).squeeze(-1)
    xi = torch.linalg.solve(case.U, mu.unsqueeze(-1)).squeeze(-1)
    A0, B = _reference_factors(case.ops, xi)
    with torch.no_grad():
        ctx0 = case.twin.edge_context(mu, case.beta, case.frame)
        sources = edge_links.latent_sources(ctx0, case.beta, case.twin.canonical_mean(mu, case.frame), case.C,
                                            case.ops.attention_residual)
    return case._replace(mu=mu, xi=xi, A0=A0, B=B, ctx0=ctx0, sources=sources)


@pytest.mark.parametrize("regime", ["random", "overflow"])
def test_causality(regime: str) -> None:
    r"""Plan 3.6 item 4: changing tokens :math:`j\ge m` (means, covariances, routing rows, log-prior rows,
    the previous layer's factors) leaves :math:`A^{(T)}_i` and the outputs of :math:`i<m` unchanged.
    In the ``overflow`` regime the early receivers' future pairs have :math:`B_j^\top A_i\approx 1200`, so an unmasked
    :math:`\varphi_1` inside the iteration would give inf and :math:`0\cdot\infty` = NaN."""
    generator = torch.Generator().manual_seed(19)
    # The overflow regime has senders with ||B_j|| ~ 50 for the late receivers; the plan's unit step is
    # stable only for sigma_L^2 ||B||^2 < 1, so that regime runs at a smaller prior scale.
    sigma_L = 0.05 if regime == "random" else 0.005
    case = _build(3, 2, True, generator, loop_weight=0.7, window=None, steps=2, sigma_L=sigma_L)
    N, K = case.mu.shape
    m = 2
    randn = _randn(generator)
    if regime == "overflow":
        case = _overflowing_future(case, generator)
        A_last = case.A0[0]
        B_future = case.B[N - 1]
        assert not torch.isfinite(torch.linalg.matrix_exp(A_last @ B_future.T)).all()
    with torch.no_grad():
        ctx = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                    link_state={"previous": case.previous})
    mean, cov = _impl_outputs(case, ctx)
    assert torch.isfinite(ctx.A).all() and torch.isfinite(mean).all() and torch.isfinite(cov).all()
    A_ref = _reference_inference(lambda X: _reference_link_energy(case, X, 0.7, None), case.A0, sigma_L, 2, 1.0)
    torch.testing.assert_close(ctx.A, A_ref, atol=1e-9, rtol=1e-9)
    mean_ref, cov_ref = _reference_outputs(case, A_ref)
    torch.testing.assert_close(mean, mean_ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(cov, _expected(case, cov_ref), atol=1e-9, rtol=1e-9)
    mu2 = case.mu.clone()
    if regime == "overflow":                                          # stay in the regime: rescale plus tiny noise
        xi2 = 1.7 * case.xi[m:] + 1e-4 * randn(N - m, K)
        mu2[m:] = (case.U[m:] @ xi2.unsqueeze(-1)).squeeze(-1)
    else:
        mu2[m:] += 0.7 * randn(N - m, K)
    sigma2 = case.sigma.clone()
    factor = randn(N - m, K, K)
    sigma2[m:] = factor @ factor.transpose(-1, -2) / K + 0.5 * torch.eye(K, dtype=F64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta2 = case.beta.clone()
    beta2[:, m:] = torch.softmax(randn(H, N - m, N).masked_fill(~causal[m:], -torch.inf), dim=-1)
    log_prior2 = case.log_prior.clone()
    log_prior2[:, m:] = randn(H, N - m, N).masked_fill(~causal[m:], -torch.inf)
    prev2 = case.previous._replace(A=case.previous.A.clone(), B=case.previous.B.clone())
    prev2.A[m:] += randn(N - m, K, 2) / K ** 0.5
    prev2.B[m:] += randn(N - m, K, 2) / K ** 0.5
    with torch.no_grad():
        ctx2 = case.ops.edge_context(mu2, beta2, case.frame, sigma=sigma2, log_prior=log_prior2,
                                     link_state={"previous": prev2})
        mean2 = case.ops.predictive_content(mu2, beta2, case.frame, edges=ctx2)
        cov2 = case.ops.predictive_covariance_update(mu2, sigma2, beta2, case.frame, edges=ctx2)
    torch.testing.assert_close(ctx2.A[:m], ctx.A[:m], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mean2[:m], mean[:m], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov2[:m], cov[:m], atol=1e-12, rtol=1e-12)
    assert float((ctx2.A[m:] - ctx.A[m:]).abs().max()) > 1e-6
    # And the reference agrees on the perturbed problem (masking changed nothing but the future).
    case2 = case._replace(mu=mu2, sigma=sigma2, sigma_dense=sigma2, beta=beta2, log_prior=log_prior2, previous=prev2,
                          weights=_reference_prior_weights(log_prior2, N),
                          xi=torch.linalg.solve(case.U, mu2.unsqueeze(-1)).squeeze(-1),
                          C=torch.linalg.inv(case.U) @ sigma2 @ torch.linalg.inv(case.U).transpose(-1, -2))
    A0_2, B_2 = _reference_factors(case.ops, case2.xi)
    case2 = case2._replace(A0=A0_2, B=B_2)
    A_ref2 = _reference_inference(lambda X: _reference_link_energy(case2, X, 0.7, None), A0_2, sigma_L, 2, 1.0)
    torch.testing.assert_close(ctx2.A, A_ref2, atol=1e-9, rtol=1e-9)


def test_zero_loop_weight_equals_loop_free_model() -> None:
    r"""Plan 3.6 item 7: :math:`\lambda_W = 0` reproduces the loop-free latent model exactly."""
    generator = torch.Generator().manual_seed(23)
    case = _build(3, 2, True, generator, loop_weight=0.0, window=None, steps=2)
    with torch.no_grad():
        A_free = edge_links.infer_latent_receivers(case.ctx0, case.beta, case.sources, sigma=0.05, steps=2, damping=1.0)
        A_zero = edge_links.infer_latent_receivers(case.ctx0, case.beta, case.sources, sigma=0.05, steps=2, damping=1.0,
                                                   loop_weight=0.0, loop=case.loop)
        e_free = edge_links.latent_receiver_energy(A_free, case.ctx0.A, case.ctx0, case.beta, case.sources, 0.05)
        e_zero = edge_links.latent_receiver_energy(A_free, case.ctx0.A, case.ctx0, case.beta, case.sources, 0.05, 0.0, case.loop)
        ctx_a = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma)
        ctx_b = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                      link_state={"previous": case.previous})
    torch.testing.assert_close(A_zero, A_free, atol=0, rtol=0)
    torch.testing.assert_close(e_zero, e_free, atol=0, rtol=0)
    torch.testing.assert_close(ctx_a.A, A_free, atol=0, rtol=0)
    torch.testing.assert_close(ctx_b.A, A_free, atol=0, rtol=0)
    # The loop-free energy lacks the loop term the reference would add.
    F, _, _, _ = _chart(case.ops)
    assert float(_reference_loop_energy(F, A_free, case.B, case.previous, case.weights, None).sum()) > 1e-4


def test_loops_are_identity_when_edge_parts_vanish() -> None:
    r"""Plan 3.6 item 7: with :math:`B = 0` and :math:`b = 0` every :math:`W_{ijk} = I` (explicitly) and the
    loop energy is exactly zero whatever :math:`A`, :math:`a`, :math:`F_l`; a vanishing edge part at one
    layer only does not suffice."""
    generator = torch.Generator().manual_seed(29)
    N, K, r = 4, 6, 2
    randn = _randn(generator)
    previous = _previous_state(generator, N, K, r)
    F_next = torch.linalg.matrix_exp(0.3 * randn(K, K) / K ** 0.5)
    A = randn(N, K, r)
    zero = torch.zeros(N, K, r, dtype=F64)
    weights = _reference_prior_weights(None, N)
    prev0 = previous._replace(B=zero)
    W = _reference_loop_matrices(F_next, A, zero, prev0, _loop_set(N, None))
    for Wm in W.values():
        torch.testing.assert_close(Wm, torch.eye(K, dtype=F64), atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        loop = edge_links.loop_energy(A, zero, prev0, weights, None)
        half_a = edge_links.loop_energy(A, zero, previous, weights, None)
        half_b = edge_links.loop_energy(A, randn(N, K, r), prev0, weights, None)
    assert float(loop.abs().max()) == 0.0
    assert float(half_a[1:].min()) > 1e-6 and float(half_b[1:].min()) > 1e-6
    # Through the operator: W_B = 0 at this layer and b = 0 before it leave the inference loop-free.
    case = _build(3, r, True, generator, loop_weight=5.0, window=None, steps=2)
    with torch.no_grad():
        case.ops.edge_sender_map.zero_()
        case.twin.edge_sender_map.zero_()
        ctx0 = case.twin.edge_context(case.mu, case.beta, case.frame)
        sources = edge_links.latent_sources(ctx0, case.beta, case.twin.canonical_mean(case.mu, case.frame), case.C, True)
        A_free = edge_links.infer_latent_receivers(ctx0, case.beta, sources, sigma=0.05, steps=2, damping=1.0)
        stats: Dict[str, torch.Tensor] = {}
        ctx = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                    link_state={"previous": prev0}, latent_out=stats)
    torch.testing.assert_close(ctx.A, A_free, atol=1e-12, rtol=1e-12)
    assert float(stats["relational_latent_loop_energy"]) == 0.0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("window", [1, 2, 3, 10, None])
def test_loop_window_indices(n: int, window: Optional[int]) -> None:
    r"""``loop_indices`` enumerates exactly :math:`\{(i,j,k): i-w\le k<j\le i\}`, each loop once."""
    i, j, k = edge_links.loop_indices(n, window, torch.device("cpu"))
    got = sorted(zip(i.tolist(), j.tolist(), k.tolist()))
    assert got == sorted(_loop_set(n, window))
    assert len(got) == len(set(got))


@pytest.mark.parametrize("window", [1, 2, 3, None])
def test_loop_energy_window_semantics(window: Optional[int]) -> None:
    r"""The loop energy over a window equals the reference restricted to that window; :math:`w\ge N-1` is
    the full set; :math:`w = 1` is the time-like loops :math:`(i, i, i-1)` only."""
    generator = torch.Generator().manual_seed(31)
    N, K, r = 4, 6, 2
    randn = _randn(generator)
    previous = _previous_state(generator, N, K, r)
    F_next = torch.linalg.matrix_exp(0.3 * randn(K, K) / K ** 0.5)
    A, B = randn(N, K, r) / K ** 0.5, randn(N, K, r) / K ** 0.5
    weights = _reference_prior_weights(randn(H, N, N).masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -torch.inf), N)
    with torch.no_grad():
        impl = edge_links.loop_energy(A, B, previous, weights, window)
    ref = _reference_loop_energy(F_next, A, B, previous, weights, window)
    torch.testing.assert_close(impl, ref, atol=1e-10, rtol=1e-10)
    assert float(impl[0]) == 0.0                                      # receiver 0 has no loop
    if window == 1:
        W = _reference_loop_matrices(F_next, A, B, previous, [(i, i, i - 1) for i in range(1, N)])
        time_like = torch.stack([torch.zeros((), dtype=F64)] + [
            weights[i, i] * weights[i, i - 1] * 0.5 * ((W[i, i, i - 1] - torch.eye(K, dtype=F64)) ** 2).sum()
            for i in range(1, N)])
        torch.testing.assert_close(impl, time_like, atol=1e-10, rtol=1e-10)
    if window == 3:
        with torch.no_grad():
            full = edge_links.loop_energy(A, B, previous, weights, None)
        torch.testing.assert_close(impl, full, atol=0, rtol=0)
    if window == 2:
        with torch.no_grad():
            narrower = edge_links.loop_energy(A, B, previous, weights, 1)
        assert float((impl - narrower)[2:].min()) > 1e-8              # window 2 adds loops for i >= 2


def test_layer_zero_has_no_loops_and_threads_its_state() -> None:
    r"""Plan 3.6 item 7: an empty ``link_state`` (layer 0) gives the loop-free inference with a zero loop
    statistic and writes ``"current"``; feeding that state to the next layer activates the loops."""
    generator = torch.Generator().manual_seed(37)
    case = _build(3, 2, True, generator, loop_weight=0.7, window=None, steps=2)
    link_state: dict = {}
    stats: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ctx = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                    link_state=link_state, latent_out=stats)
        A_free = edge_links.infer_latent_receivers(case.ctx0, case.beta, case.sources, sigma=0.05, steps=2, damping=1.0)
    torch.testing.assert_close(ctx.A, A_free, atol=1e-12, rtol=1e-12)      # C symmetrized inside edge_context
    assert float(stats["relational_latent_loop_energy"]) == 0.0
    assert "previous" not in link_state and isinstance(link_state["current"], LinkState)
    # Next layer: a second operator reads layer 0's state through the stack's promotion.
    link_state["previous"] = link_state.pop("current")
    stats1: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ctx1 = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                     link_state=link_state, latent_out=stats1)
    assert float(stats1["relational_latent_loop_energy"]) > 0.0
    assert float((ctx1.A - ctx.A).abs().max()) > 1e-8
    case1 = case._replace(previous=link_state["previous"])
    A_ref1 = _reference_inference(lambda X: _reference_link_energy(case1, X, 0.7, None), case.A0, 0.05, 2, 1.0)
    torch.testing.assert_close(ctx1.A, A_ref1, atol=1e-9, rtol=1e-9)


def test_no_grad_matches_grad_mode_and_outer_gradients_flow() -> None:
    r"""Plan 3.6 item 6: the same inferred factors and outputs with gradients enabled and under
    ``torch.no_grad``; in grad mode the outputs reach :math:`W_A, W_B, T, O, \mu, \sigma`."""
    generator = torch.Generator().manual_seed(41)
    case = _build(3, 2, True, generator, block=True, family="diag", loop_weight=0.7, window=None, steps=2)
    mu = case.mu.clone().requires_grad_(True)
    sigma = case.sigma.clone().requires_grad_(True)
    link_state = {"previous": case.previous}
    ctx_g = case.ops.edge_context(mu, case.beta, case.frame, sigma=sigma, log_prior=case.log_prior,
                                  link_state=link_state)
    mean_g = case.ops.predictive_content(mu, case.beta, case.frame, edges=ctx_g)
    cov_g = case.ops.predictive_covariance_update(mu, sigma, case.beta, case.frame, edges=ctx_g)
    assert ctx_g.A.requires_grad
    with torch.no_grad():
        ctx_n = case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                      link_state={"previous": case.previous})
    mean_n, cov_n = _impl_outputs(case, ctx_n)
    assert not ctx_n.A.requires_grad
    torch.testing.assert_close(ctx_n.A, ctx_g.A.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mean_n, mean_g.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov_n, cov_g.detach(), atol=1e-12, rtol=1e-12)
    (mean_g.square().sum() + cov_g.sum()).backward()
    for name in ("edge_receiver_map", "edge_sender_map", "T", "O"):
        grad = getattr(case.ops, name).grad
        assert grad is not None and float(grad.abs().max()) > 0.0, name
    assert float(mu.grad.abs().max()) > 0.0 and float(sigma.grad.abs().max()) > 0.0
    # The readout call site: identical values under both modes, latent statistics captured.
    capture_g: dict = {}
    out_g, cov_rg = relational_predictive_readout(case.mu, case.sigma, case.frame, case.ops, log_prior=case.log_prior,
                                                  capture=capture_g, return_covariance=True,
                                                  link_state={"previous": case.previous})
    with torch.no_grad():
        capture_n: dict = {}
        out_n, cov_rn = relational_predictive_readout(case.mu, case.sigma, case.frame, case.ops, log_prior=case.log_prior,
                                                      capture=capture_n, return_covariance=True,
                                                      link_state={"previous": case.previous})
    torch.testing.assert_close(out_n, out_g.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov_rn, cov_rg.detach(), atol=1e-12, rtol=1e-12)
    for key in KEYS:
        assert key in capture_g and key in capture_n
        assert math.isfinite(float(capture_g[key])) and float(capture_g[key]) == float(capture_n[key]), key
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            case.ops.edge_context(case.mu, case.beta, case.frame, sigma=case.sigma, log_prior=case.log_prior,
                                  link_state={"previous": case.previous})


# --------------------------------------------------------------------------------------------------
# Tiny model
# --------------------------------------------------------------------------------------------------

_EDGE = dict(attention_mode="projected_predictive", content_link_chart="exponential", content_link_edges="lowrank",
             content_link_edge_rank=2, max_seq_len=4, n_layers=2)


def _model(**overrides: object) -> VFEModel:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(**{**_EDGE, **overrides}))
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.edge_sender_map.copy_(torch.randn(*ops.edge_sender_map.shape, generator=generator) / ops.input_dim)
    return model


def test_tiny_model_layer_zero_has_no_loops_and_eval_matches_training() -> None:
    r"""Through the stack: layer 0 reports a zero loop energy and layer 1 a positive one; the loss is the
    same in training mode with gradients, under ``no_grad`` and in eval mode (control: the same model with
    latent inference off)."""
    latent = dict(content_link_latent=True, content_link_latent_steps=2, content_link_latent_sigma=0.05,
                  content_link_loop_weight=1.0, content_link_loop_window=None)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 6, 2, 1]])
    model = _model(**latent)
    rows = model.diagnostics_per_layer(ids)
    for key in KEYS:
        assert key in rows and len(rows[key]) == 2, key
        assert all(math.isfinite(float(v)) for v in rows[key]), key
    assert float(rows["relational_latent_loop_energy"][0]) == 0.0
    assert float(rows["relational_latent_loop_energy"][1]) > 0.0
    assert all(float(v) > 0.0 for v in rows["relational_latent_receiver_shift"])
    losses = {}
    controls = {}
    control = _model()
    for name, m in (("latent", model), ("control", control)):
        store = losses if name == "latent" else controls
        m.train()
        _, loss, _ = m(ids, targets)
        loss.backward()
        store["train_grad"] = float(loss)
        with torch.no_grad():
            store["train_no_grad"] = float(m(ids, targets)[1])
        m.eval()
        with torch.no_grad():
            store["eval_no_grad"] = float(m(ids, targets)[1])
    assert math.isfinite(losses["train_grad"])
    assert losses["train_no_grad"] == pytest.approx(losses["train_grad"], rel=1e-6, abs=1e-6)
    if controls["eval_no_grad"] == pytest.approx(controls["train_grad"], rel=1e-6, abs=1e-6):
        assert losses["eval_no_grad"] == pytest.approx(losses["train_grad"], rel=1e-6, abs=1e-6)
    for ops in model.relational_operators_by_layer:
        for name in ("edge_receiver_map", "edge_sender_map", "T", "O", "raw_score_q"):
            grad = getattr(ops, name).grad
            assert grad is not None and float(grad.abs().max()) > 0.0, name
