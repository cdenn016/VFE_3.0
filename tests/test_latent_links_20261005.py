r"""Latent receiver factors, Level 2 (docs/implementation/edge-and-latent-links-plan-2026-10-05.md, Section 3).

With ``content_link_edges='lowrank'`` and ``content_link_latent=True`` the receiver factor A_i of every edge link
G_ij = e^L exp(A_i B_j^T) is the per-context MAP estimate of the energy

    F_i(A_i) = tr C_i^att(A_i)|_no floor / (2 tr(C_i)/K) + ||A_i - A_i^0||_F^2 / (2 sigma_L^2) + lambda_W Loop_i(A_i),

reached by damped fixed-point steps A <- (1 - rho) A + rho (A^0 - sigma_L^2 grad[E + lambda_W Loop]). Tiny float64
CPU cases (K = 4, N <= 5, two heads). Covers the plan's Section 3.6: the static limit, descent, the fixed point and a
finite-difference gradient of F_i, causality, gauge covariance, outer gradients through the unrolled steps and the
no_grad evaluation, the loop prior (brute force, identity, windows, layer 0, lambda_W = 0), config validation and
short training with latent factors and loops.
"""
from tests.profiles import make_relational_operators
import logging
import math
from typing import Dict, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config, tiny_config
from tests.test_edge_links_20261005 import FLOOR, _block_dense, _cases, _close, _inputs, _ops
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model import edge_links
from vfe3.model.edge_links import LinkState, LoopInputs
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import _relational_artifact_fields

torch.set_num_threads(1)
F64 = torch.float64
LATENT = dict(content_link_latent=True, content_link_latent_steps=2, content_link_latent_sigma=0.1)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)


# ----------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------

def _latent_ops(
    residual: bool = True,
    seed:     int = 8,
    sender:   float = 0.2,
    receiver: float = 0.2,
    **kwargs: object,
) -> RelationalOperators:
    r"""Float64 lowrank operators with latent inference; ``sender`` / ``receiver`` are the W_B / W_A scales.

    Moderate factor scales: the unit natural-gradient step overshoots when sigma_L^2 times the curvature of
    E_i is large (see the costs and limits in docs/2026-10-05.md).
    """
    ops = _ops("lowrank", residual, seed=seed, sender=sender, **{**LATENT, **kwargs})
    with torch.no_grad():
        ops.edge_receiver_map.mul_(receiver / 0.5)                     # _ops draws W_A with scale 0.5
    return ops


def _level_one(ops: RelationalOperators) -> RelationalOperators:
    r"""The same maps with the latent inference off (Level 1, amortized receiver factors)."""
    twin = make_relational_operators(ops.irrep_dims, 11, attention_residual=ops.attention_residual,
                               predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
                               content_link_chart="exponential", content_link_edges="lowrank",
                               content_link_edge_rank=ops.content_link_edge_rank).double()
    twin.load_state_dict(ops.state_dict())
    return twin


def _problem(
    ops:   RelationalOperators,
    mu:    torch.Tensor,
    sigma: torch.Tensor,
    beta:  torch.Tensor,
    frame: CanonicalFrameContext,
) -> Tuple[edge_links.EdgeContext, edge_links.LatentSources]:
    r"""The amortized lowrank context and the latent sources of one readout."""
    ctx = _level_one(ops).edge_context(mu, beta, frame)
    xi = ops.canonical_mean(mu, frame)
    C = ops._reference_covariance(sigma, frame, F64, diagonal=sigma.dim() == mu.dim())
    return ctx, edge_links.latent_sources(ctx, beta, xi, C, ops.attention_residual)


def _previous(N: int, K: int = 4, r: int = 2, seed: int = 0, batch: Tuple[int, ...] = (1,),
              scale: float = 0.4) -> LinkState:
    r"""A random previous-layer link state (F = e^L, e^{-L}, A^l, B^l)."""
    g = torch.Generator().manual_seed(1000 + seed)
    L = 0.3 * torch.randn(K, K, generator=g, dtype=F64)
    return LinkState(torch.linalg.matrix_exp(L), torch.linalg.matrix_exp(-L),
                     scale * torch.randn(*batch, N, K, r, generator=g, dtype=F64),
                     scale * torch.randn(*batch, N, K, r, generator=g, dtype=F64))


def _readout(ops: RelationalOperators, mu: torch.Tensor, sigma: torch.Tensor, frame: CanonicalFrameContext,
             **kwargs: object) -> Tuple[torch.Tensor, torch.Tensor]:
    return relational_predictive_readout(mu, sigma, frame, ops, return_covariance=True, **kwargs)


# ----------------------------------------------------------------------------------------------
# 1. Static limit sigma_L -> 0
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("damping", [1.0, 0.3])
@pytest.mark.parametrize("residual", [False, True])
def test_static_limit_reproduces_level_one(damping: float, residual: bool) -> None:
    ops = _latent_ops(residual, seed=1, sender=0.4, receiver=0.5, content_link_latent_sigma=1e-12,
                      content_link_latent_damping=damping, content_link_latent_steps=3)
    twin = _level_one(ops)
    assert set(ops.state_dict()) == set(twin.state_dict())           # the latent inference adds no parameter
    data = _inputs(seed=2)
    for _, sigma, frame in _cases(data):
        mean, cov = _readout(ops, data["mu"], sigma, frame)
        mean_1, cov_1 = _readout(twin, data["mu"], sigma, frame)
        _close(mean, mean_1, 1e-12)
        _close(cov, cov_1, 1e-12)


@pytest.mark.parametrize("residual", [False, True])
def test_transport_energy_is_the_trace_of_the_dense_covariance_body(residual: bool) -> None:
    r"""E_i(A) = tr C_i^att(A)|_no floor / (2 tr(C_i)/K) at a random A != A^0, against the Level 1 dense body."""
    ops = _latent_ops(residual, seed=3, sender=0.3, receiver=0.3)
    twin = _level_one(ops)
    twin.predictive_covariance_floor = 0.0
    data = _inputs(seed=4, batch=(2,))
    mu, beta = data["mu"], data["beta"]
    g = torch.Generator().manual_seed(7)
    K = mu.shape[-1]
    for _, sigma, frame in _cases(data):
        ctx, sources = _problem(ops, mu, sigma, beta, frame)
        A = ctx.A + 0.3 * torch.randn(ctx.A.shape, generator=g, dtype=F64)
        E = edge_links.latent_transport_energy(A, ctx, beta, sources)                  # (B, N)
        Sigma = torch.diag_embed(sigma) if sigma.dim() == mu.dim() else sigma
        local = twin._predictive_covariance_update_dense(
            mu, Sigma, beta, frame, None, edge_links.with_receiver_factors(ctx, A, beta))  # (B, N, K, K)
        U_inv = torch.linalg.inv(_block_dense(frame.forward) if frame.block_diagonal else frame.forward)
        trace = lambda X: torch.diagonal(U_inv @ X @ U_inv.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1)
        torch.testing.assert_close(E, trace(local) / (2.0 * trace(Sigma) / K), rtol=1e-12, atol=0.0)


# ----------------------------------------------------------------------------------------------
# 2. Descent and 3. fixed point
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("damping", [1.0, 0.5])
@pytest.mark.parametrize("seed", [8, 9])
def test_one_step_lowers_every_receiver_energy(damping: float, seed: int) -> None:
    sigma_L = 0.02
    ops = _latent_ops(True, seed=seed)
    data = _inputs(seed=seed + 1, batch=(2,))
    ctx, sources = _problem(ops, data["mu"], data["sigma_diag"], data["beta"], data["compact"])
    with torch.no_grad():
        A1 = edge_links.infer_latent_receivers(ctx, data["beta"], sources, sigma=sigma_L, steps=1, damping=damping)
        F0 = edge_links.latent_receiver_energy(ctx.A, ctx.A, ctx, data["beta"], sources, sigma_L)
        F1 = edge_links.latent_receiver_energy(A1, ctx.A, ctx, data["beta"], sources, sigma_L)
    assert float((A1 - ctx.A).abs().max()) > 0.0
    assert bool((F1 < F0).all()), (F0 - F1)


def test_fixed_point_satisfies_stationarity() -> None:
    sigma_L = 0.1
    ops = _latent_ops(True, seed=8)
    data = _inputs(seed=9, batch=(2,))
    beta = data["beta"]
    ctx, sources = _problem(ops, data["mu"], data["sigma_diag"], beta, data["compact"])
    A0 = ctx.A.detach()
    with torch.no_grad():
        A = edge_links.infer_latent_receivers(ctx, beta, sources, sigma=sigma_L, steps=100, damping=0.8)
    point = A.clone().requires_grad_(True)
    grad, = torch.autograd.grad(edge_links.latent_transport_energy(point, ctx, beta, sources).sum(), point)
    assert float((A - A0).norm() / A0.norm()) > 1e-2                 # the latent factor moved
    assert float((A - (A0 - sigma_L ** 2 * grad)).norm() / A0.norm()) < 1e-10
    # Stationarity of F_i: the gradient of the full MAP energy vanishes at the fixed point.
    point = A.clone().requires_grad_(True)
    full, = torch.autograd.grad(
        edge_links.latent_receiver_energy(point, A0, ctx, beta, sources, sigma_L).sum(), point)
    assert float(full.norm()) <= 1e-6 * float(grad.norm())


@pytest.mark.parametrize("residual", [False, True])
def test_map_energy_gradient_matches_finite_differences(residual: bool) -> None:
    r"""gradcheck of A -> F_i(A) (transport energy, prior and loop prior) in float64; the Jacobian is block diagonal."""
    ops = _latent_ops(residual, seed=3, sender=0.3, receiver=0.3)
    data = _inputs(seed=4)
    beta = data["beta"]
    ctx, sources = _problem(ops, data["mu"], data["sigma_full"], beta, data["compact"])
    A0 = ctx.A.detach()
    loop = LoopInputs(_previous(4, seed=1), edge_links.prior_routing_weights(None, 4, dtype=F64, device=A0.device), 2)
    A = (A0 + 0.05 * torch.randn(A0.shape, generator=torch.Generator().manual_seed(5), dtype=F64)).requires_grad_(True)
    energy = lambda X: edge_links.latent_receiver_energy(X, A0, ctx, beta, sources, 0.3, 0.7, loop)
    assert torch.autograd.gradcheck(energy, (A,), eps=1e-6, atol=1e-6, rtol=1e-5)


# ----------------------------------------------------------------------------------------------
# 4. Causality
# ----------------------------------------------------------------------------------------------

def test_future_perturbations_leave_earlier_receivers_unchanged() -> None:
    r"""Receiver i's inferred A_i, mean and covariance ignore every token j > i, with the loop prior on."""
    N = 4
    ops = _latent_ops(True, seed=6, content_link_loop_weight=0.5, content_link_loop_window=None,
                      content_link_latent_steps=3)
    data = _inputs(N=N, seed=7)
    beta, frame = data["beta"], data["compact"]
    previous = _previous(N, seed=2)
    g = torch.Generator().manual_seed(8)

    def run(mu: torch.Tensor, sigma: torch.Tensor, previous: LinkState) -> Tuple[torch.Tensor, ...]:
        ctx = ops.edge_context(mu, beta, frame, sigma=sigma, link_state={"previous": previous})
        return (ctx.A, ops.predictive_content(mu, beta, frame, edges=ctx),
                ops.predictive_covariance_update(mu, sigma, beta, frame, edges=ctx))

    base = run(data["mu"], data["sigma_diag"], previous)
    for last in range(1, N):
        mu, sigma = data["mu"].clone(), data["sigma_diag"].clone()
        mu[:, last:] += torch.randn(mu[:, last:].shape, generator=g, dtype=F64)
        sigma[:, last:] *= 1.5
        a, b = previous.A.clone(), previous.B.clone()                 # the previous layer's future links move too
        a[:, last:] += 0.3 * torch.randn(a[:, last:].shape, generator=g, dtype=F64)
        b[:, last:] += 0.3 * torch.randn(b[:, last:].shape, generator=g, dtype=F64)
        moved = run(mu, sigma, previous._replace(A=a, B=b))
        for got, want in zip(moved, base):
            torch.testing.assert_close(got[:, :last], want[:, :last], rtol=1e-12, atol=1e-12)
        assert float((moved[0][:, last:] - base[0][:, last:]).abs().max()) > 1e-6


def test_masking_before_phi1_survives_a_future_overflow() -> None:
    r"""A future token whose phi_1 would overflow leaves every earlier inferred factor and output finite and unchanged."""
    H, d, N = 2, 2, 4
    K = H * d
    ops = _latent_ops(True, seed=16, sender=0.2)
    g = torch.Generator().manual_seed(17)
    u = torch.randn(K * 2, generator=g, dtype=F64)
    e_K = torch.zeros(K, dtype=F64)
    e_K[-1] = 1.0
    with torch.no_grad():                                             # a huge sender map along e_K only
        ops.edge_sender_map.add_(1e4 * torch.outer(u / u.norm(), e_K))
    data = _inputs(H, d, N, seed=18)
    blocks, beta, frame, sigma = data["compact"].forward, data["beta"], data["compact"], data["sigma_diag"]
    local = lambda x: (_block_dense(blocks) @ x.unsqueeze(-1)).squeeze(-1)
    xi = torch.randn(1, N, K, generator=g, dtype=F64) + 1.0
    xi[..., -1] = 0.0                                                 # no token reads the huge direction

    def run(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx = ops.edge_context(local(x), beta, frame, sigma=sigma)
        return (ctx.A, ops.predictive_content(local(x), beta, frame, edges=ctx),
                ops.predictive_covariance_update(local(x), sigma, beta, frame, edges=ctx))

    base = run(xi)
    assert all(bool(torch.isfinite(x).all()) for x in base)
    moved_xi = xi.clone()
    moved_xi[0, N - 1] = 0.0
    moved_xi[0, N - 1, -1] = 5.0                                      # the last token reads it: B_{N-1} huge
    moved = run(moved_xi)
    for got, want in zip(moved, base):
        assert bool(torch.isfinite(got[0, :N - 1]).all())
        torch.testing.assert_close(got[0, :N - 1], want[0, :N - 1], rtol=1e-12, atol=1e-12)
    B_last = _level_one(ops).edge_context(local(moved_xi), beta, frame).B[0, N - 1]
    unmasked = [edge_links.phi1(B_last.T @ moved[0][0, i]) for i in range(N - 1)]
    assert any(not bool(torch.isfinite(x).all()) for x in unmasked)   # the unmasked phi_1 would have overflowed


# ----------------------------------------------------------------------------------------------
# 5. Gauge covariance
# ----------------------------------------------------------------------------------------------

def test_readout_is_gauge_covariant_through_routing_with_latent_factors() -> None:
    ops = _latent_ops(True, seed=12)
    data = _inputs(seed=13)
    N, K = data["mu"].shape[-2:]
    g_blocks = torch.eye(2, dtype=F64) + 0.4 * torch.randn(1, N, 2, 2, 2, generator=torch.Generator().manual_seed(3),
                                                           dtype=F64)
    g = _block_dense(g_blocks)
    blocks = data["compact"].forward
    moved = CanonicalFrameContext(g_blocks @ blocks, torch.linalg.inv(g_blocks @ blocks), block_diagonal=True)
    mu, sigma = data["mu"], data["sigma_full"]
    capture: Dict[str, torch.Tensor] = {}
    mean, cov = _readout(ops, mu, sigma, data["compact"], capture=capture)
    assert float(capture["relational_latent_receiver_shift"]) > 1e-3   # the inference moved the factors
    mean_g, cov_g = _readout(ops, (g @ mu.unsqueeze(-1)).squeeze(-1), g @ sigma @ g.transpose(-1, -2), moved)
    _close(mean_g, (g @ mean.unsqueeze(-1)).squeeze(-1), 1e-9)
    _close(cov_g, g @ cov @ g.transpose(-1, -2), 1e-9)


def test_passive_gauge_action_at_fixed_routing_with_loops() -> None:
    r"""Arbitrary dense g: the inferred factors are invariant and the outputs covariant (loop prior on)."""
    ops = _latent_ops(True, seed=10, content_link_loop_weight=1.0, content_link_loop_window=2)
    data = _inputs(seed=11)
    N, K = data["mu"].shape[-2:]
    g = torch.eye(K, dtype=F64) + 0.4 * torch.randn(1, N, K, K, generator=torch.Generator().manual_seed(2), dtype=F64)
    mu, sigma, beta, U = data["mu"], data["sigma_full"], data["beta"], data["U"]
    previous = _previous(N, seed=3)
    moved_U = g @ U
    moved = CanonicalFrameContext(moved_U, torch.linalg.inv(moved_U))
    mu_g, sigma_g = (g @ mu.unsqueeze(-1)).squeeze(-1), g @ sigma @ g.transpose(-1, -2)
    ctx = ops.edge_context(mu, beta, data["dense"], sigma=sigma, link_state={"previous": previous})
    ctx_g = ops.edge_context(mu_g, beta, moved, sigma=sigma_g, link_state={"previous": previous})
    _close(ctx_g.A, ctx.A, 1e-9)
    _close(ops.predictive_content(mu_g, beta, moved, edges=ctx_g),
           (g @ ops.predictive_content(mu, beta, data["dense"], edges=ctx).unsqueeze(-1)).squeeze(-1), 1e-9)
    cov = ops.predictive_covariance_update(mu, sigma, beta, data["dense"], edges=ctx)
    _close(ops.predictive_covariance_update(mu_g, sigma_g, beta, moved, edges=ctx_g), g @ cov @ g.transpose(-1, -2), 1e-9)


# ----------------------------------------------------------------------------------------------
# 6. Outer gradients through the unrolled steps; no_grad evaluation
# ----------------------------------------------------------------------------------------------

def test_training_backward_reaches_every_outer_parameter() -> None:
    ops = _latent_ops(True, seed=14, content_link_latent_damping=0.7)
    data = _inputs(seed=15)
    mu = data["mu"].clone().requires_grad_(True)
    sigma = data["sigma_diag"].clone().requires_grad_(True)
    mean, cov = _readout(ops, mu, sigma, data["compact"])
    (mean.square().sum() + cov.square().sum()).backward()
    for name in ("edge_receiver_map", "edge_sender_map", "T", "O", "raw_score_q", "raw_score_k"):
        grad = getattr(ops, name).grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0.0, name
    assert float(mu.grad.abs().max()) > 0.0 and float(sigma.grad.abs().max()) > 0.0


def test_no_grad_evaluation_matches_training_forward() -> None:
    ops = _latent_ops(True, seed=16, content_link_latent_steps=3, content_link_latent_damping=0.6)
    data = _inputs(seed=17)
    for _, sigma, frame in _cases(data):
        mean, cov = _readout(ops, data["mu"], sigma, frame)
        assert mean.requires_grad
        with torch.no_grad():
            mean_eval, cov_eval = _readout(ops, data["mu"], sigma, frame)
        assert not mean_eval.requires_grad
        torch.testing.assert_close(mean_eval, mean.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(cov_eval, cov.detach(), rtol=1e-12, atol=1e-12)


class _LatentReadout(torch.nn.Module):
    r"""A scalar of the latent readout's mean and covariance (block frames, diagonal family)."""

    def __init__(self, ops: RelationalOperators, weights: Tuple[torch.Tensor, torch.Tensor]) -> None:
        super().__init__()
        self.ops = ops
        self.weights = weights

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, frame: CanonicalFrameContext) -> torch.Tensor:
        mean, cov = relational_predictive_readout(mu, sigma, frame, self.ops, return_covariance=True)
        return (self.weights[0] * mean).sum() + (self.weights[1] * cov).sum()


def test_outer_gradients_match_finite_differences() -> None:
    r"""gradcheck through the unrolled create_graph steps (double backward of phi_1) for W_A, W_B, mu and sigma."""
    ops = _latent_ops(True, seed=19, content_link_latent_steps=2, content_link_latent_damping=0.7)
    data = _inputs(N=3, seed=20)
    g = torch.Generator().manual_seed(4)
    wrap = _LatentReadout(ops, (torch.randn(data["mu"].shape, generator=g, dtype=F64),
                                torch.randn(data["sigma_diag"].shape, generator=g, dtype=F64)))
    names = ("ops.edge_sender_map", "ops.edge_receiver_map")

    def scalar(W_B: torch.Tensor, W_A: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(wrap, {names[0]: W_B, names[1]: W_A}, (mu, sigma, data["compact"]))

    inputs = tuple(x.detach().clone().requires_grad_(True) for x in (
        ops.edge_sender_map, ops.edge_receiver_map, data["mu"], data["sigma_diag"]))
    assert torch.autograd.gradcheck(scalar, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)


def test_checkpointed_covariance_is_bit_identical_with_latent_factors() -> None:
    outputs = []
    for flag in (False, True):
        ops = _latent_ops(True, seed=21, predictive_covariance_checkpoint=flag)
        data = _inputs(seed=22)
        mu = data["mu"].clone().requires_grad_(True)
        out = _readout(ops, mu, data["sigma_diag"], data["compact"])
        (out[0].sum() + out[1].square().sum()).backward()
        outputs.append([out[0].detach(), out[1].detach(), mu.grad] + [p.grad for p in ops.parameters()])
    for plain, recomputed in zip(*outputs):
        assert torch.equal(plain, recomputed)


# ----------------------------------------------------------------------------------------------
# 7. Loop prior
# ----------------------------------------------------------------------------------------------

def _explicit_loop_energy(A: torch.Tensor, B: torch.Tensor, previous: LinkState, weights: torch.Tensor,
                          window) -> torch.Tensor:
    r"""Loop_i from explicit K x K matrices W_ijk = (E_kk^l)^{-1} F^{-1} (E_ik^{l+1})^{-1} E_ij^{l+1} F E_jk^l."""
    E = lambda X, Y: torch.linalg.matrix_exp(X @ Y.transpose(-1, -2))
    *batch, N, K, _ = A.shape
    out = torch.zeros(*batch, N, dtype=F64)
    a, b, F, F_inv = previous.A, previous.B, previous.F, previous.F_inv
    for s in range(batch[0]):
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if not k < j <= i or (window is not None and k < i - window):
                        continue
                    W = (torch.linalg.inv(E(a[s, k], b[s, k])) @ F_inv @ torch.linalg.inv(E(A[s, i], B[s, k]))
                         @ E(A[s, i], B[s, j]) @ F @ E(a[s, j], b[s, k]))
                    out[s, i] += weights[i, j] * weights[j, k] * 0.5 * (W - torch.eye(K, dtype=F64)).square().sum()
    return out


@pytest.mark.parametrize("window", [1, 2, None])
def test_loop_energy_matches_explicit_matrices(window) -> None:
    N, K, r = 5, 4, 2
    g = torch.Generator().manual_seed(30)
    A, B = (0.5 * torch.randn(2, N, K, r, generator=g, dtype=F64) for _ in range(2))
    previous = _previous(N, seed=4, batch=(2,), scale=0.5)
    weights = torch.rand(N, N, generator=g, dtype=F64).tril()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    got = edge_links.loop_energy(A, B, previous, weights, window)
    want = _explicit_loop_energy(A, B, previous, weights, window)
    assert float(want[..., 1:].min()) > 1e-3                          # every receiver i >= 1 closes a loop
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)
    assert float(got[..., 0].abs().max()) == 0.0                      # receiver 0 closes no loop


def test_loops_are_identity_when_every_edge_part_vanishes() -> None:
    r"""B = 0 at both layers: every E is the identity and W_ijk = e^{-L} e^{L} = I (the static shared-generator model)."""
    N, K, r = 4, 4, 2
    g = torch.Generator().manual_seed(31)
    A = torch.randn(1, N, K, r, generator=g, dtype=F64)
    previous = _previous(N, seed=5)._replace(B=torch.zeros(1, N, K, r, dtype=F64))
    weights = edge_links.prior_routing_weights(None, N, dtype=F64, device=A.device)
    zero = edge_links.loop_energy(A, torch.zeros_like(A), previous, weights, None)
    assert float(zero.abs().max()) < 1e-24
    assert float(edge_links.loop_energy(A, 0.3 * torch.randn(A.shape, generator=g, dtype=F64), previous,
                                        weights, None).min()) >= 0.0


def test_loop_window_semantics() -> None:
    cpu = torch.device("cpu")
    as_set = lambda idx: set(zip(*(t.tolist() for t in idx)))
    assert as_set(edge_links.loop_indices(5, 1, cpu)) == {(i, i, i - 1) for i in range(1, 5)}
    assert as_set(edge_links.loop_indices(5, 2, cpu)) == {
        (i, j, k) for i in range(5) for j in range(5) for k in range(5) if i - 2 <= k < j <= i}
    assert as_set(edge_links.loop_indices(5, None, cpu)) == {
        (i, j, k) for i in range(5) for j in range(5) for k in range(5) if k < j <= i}
    assert edge_links.loop_indices(1, None, cpu)[0].numel() == 0
    assert edge_links.loop_indices(6, 10, cpu)[0].numel() == edge_links.loop_indices(6, None, cpu)[0].numel()


def test_zero_weight_loops_are_skipped() -> None:
    r"""A hard-masked prior (causal band of width 1) gives zero weight to the loops through sender 0 at receiver 3;
    sender 0's phi_1 overflows there, which must not turn into 0 * inf = NaN when every loop is kept."""
    N, K, r = 5, 4, 2
    g = torch.Generator().manual_seed(0)
    A, B = (0.5 * torch.randn(1, N, K, r, generator=g, dtype=F64) for _ in range(2))
    B[0, 0] = -800.0 * A[0, 3] / A[0, 3].norm()                      # phi_1(-B_0^T A_3) overflows
    previous = _previous(N, seed=7, scale=0.3)
    lag = torch.arange(N)[:, None] - torch.arange(N)[None, :]
    log_prior = torch.zeros(N, N, dtype=F64).masked_fill(~((lag >= 0) & (lag <= 1)), -torch.inf)
    weights = edge_links.prior_routing_weights(log_prior, N, dtype=F64, device=A.device)
    band = edge_links.loop_energy(A, B, previous, weights, 2)
    every = edge_links.loop_energy(A, B, previous, weights, None)
    assert math.isfinite(float(every[0, 3]))
    assert float(every[0, 3]) > 0.0
    torch.testing.assert_close(every[0, 3:], band[0, 3:], rtol=1e-12, atol=0.0)


def test_prior_routing_weights() -> None:
    N = 4
    uniform = edge_links.prior_routing_weights(None, N, dtype=F64, device=torch.device("cpu"))
    torch.testing.assert_close(uniform, torch.ones(N, N, dtype=F64).tril() / torch.arange(1, N + 1, dtype=F64)[:, None])
    log_prior = torch.randn(2, N, N, generator=torch.Generator().manual_seed(6), dtype=F64)
    log_prior = log_prior.masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -torch.inf)
    want = torch.softmax(log_prior, dim=-1).mean(dim=0)
    torch.testing.assert_close(edge_links.prior_routing_weights(log_prior, N, dtype=F64, device=log_prior.device), want)


def test_zero_loop_weight_is_the_loop_free_inference_exactly() -> None:
    ops = _latent_ops(True, seed=23)
    data = _inputs(seed=24)
    beta = data["beta"]
    ctx, sources = _problem(ops, data["mu"], data["sigma_diag"], beta, data["compact"])
    loop = LoopInputs(_previous(4, seed=6), edge_links.prior_routing_weights(None, 4, dtype=F64, device=ctx.A.device), 1)
    common = dict(sigma=0.1, steps=3, damping=0.5)
    with torch.no_grad():
        free = edge_links.infer_latent_receivers(ctx, beta, sources, **common)
        zero = edge_links.infer_latent_receivers(ctx, beta, sources, loop_weight=0.0, loop=loop, **common)
        on = edge_links.infer_latent_receivers(ctx, beta, sources, loop_weight=1.0, loop=loop, **common)
    assert torch.equal(free, zero)
    assert float((on - free).abs().max()) > 1e-8


# ----------------------------------------------------------------------------------------------
# Model level: layer 0, threading, causality, static limit, witness
# ----------------------------------------------------------------------------------------------

_EDGE = dict(attention_mode="projected_predictive", content_link_chart="exponential")


def _latent_model(**overrides: object) -> VFEModel:
    cfg = tiny_config(**{**_EDGE, "content_link_edges": "lowrank", "content_link_edge_rank": 2, "max_seq_len": 4,
                         "n_layers": 3, "content_link_latent": True, "content_link_latent_steps": 2,
                         "content_link_latent_sigma": 0.2, **overrides})
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.edge_sender_map.normal_(std=0.3)
    return model


def _diagnostic_forward(model: VFEModel, ids: torch.Tensor) -> Dict:
    capture: Dict = {"diagnostic": {}}
    with torch.no_grad():
        model.eval().forward_beliefs(ids, capture=capture)
    return capture["diagnostic"]


def test_layer_zero_has_no_loops_and_loops_are_threaded_only_when_weighted(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.model.stack as stack_module
    ids = torch.tensor([[1, 2, 3, 4]])
    looped = _latent_model(content_link_loop_weight=0.5, content_link_loop_window=2)
    free = _latent_model()
    free.load_state_dict(looped.state_dict())
    seen = []
    original = stack_module.vfe_block

    def recording(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("link_state"))
        return original(*args, **kwargs)

    monkeypatch.setattr(stack_module, "vfe_block", recording)
    loop_diag, free_diag = _diagnostic_forward(looped, ids), _diagnostic_forward(free, ids)
    assert all(state is not None for state in seen[:3]) and all(state is None for state in seen[3:])
    caps = loop_diag["layer_conditional_captures"]
    assert float(caps[0]["relational_latent_loop_energy"]) == 0.0      # layer 0 closes no loop
    assert all(float(cap["relational_latent_loop_energy"]) > 0.0 for cap in caps[1:])
    assert all(math.isnan(float(cap["relational_latent_loop_energy"])) for cap in free_diag["layer_conditional_captures"])
    assert torch.equal(loop_diag["layer_outputs"][0].mu, free_diag["layer_outputs"][0].mu)
    assert float((loop_diag["layer_outputs"][-1].mu - free_diag["layer_outputs"][-1].mu).abs().max()) > 1e-7


@pytest.mark.parametrize("loops", [False, True])
def test_model_forward_is_causal_with_latent_factors(loops: bool) -> None:
    model = _latent_model(**({"content_link_loop_weight": 1.0, "content_link_loop_window": None} if loops else {}))
    ids = torch.tensor([[1, 2, 3, 4]])
    moved = torch.tensor([[1, 2, 3, 6]])
    for train in (False, True):
        model.train(train)
        with torch.set_grad_enabled(train):
            want, _ = model.forward_beliefs(ids)
            got, _ = model.forward_beliefs(moved)
        torch.testing.assert_close(got.mu[:, :3], want.mu[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(got.sigma[:, :3], want.sigma[:, :3], rtol=0, atol=0)


def test_model_static_limit_equals_level_one() -> None:
    latent = _latent_model(content_link_latent_sigma=1e-12, content_link_latent_damping=0.5)
    level_one = VFEModel(tiny_config(**{**_EDGE, "content_link_edges": "lowrank", "content_link_edge_rank": 2,
                                        "max_seq_len": 4, "n_layers": 3}))
    assert set(latent.state_dict()) == set(level_one.state_dict())
    level_one.load_state_dict(latent.state_dict())
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    with torch.no_grad():
        want, _ = level_one.eval().forward_beliefs(ids)
        got, _ = latent.eval().forward_beliefs(ids)
    torch.testing.assert_close(got.mu, want.mu, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got.sigma, want.sigma, rtol=1e-6, atol=1e-6)


def test_gauge_equivariance_witness_with_latent_loops() -> None:
    from vfe3.viz import extract
    model = _latent_model(content_link_loop_weight=1.0, content_link_loop_window=2)
    result = extract.projected_gauge_equivariance(model.eval(), torch.tensor([[1, 2, 3, 4]]), n_draws=2)
    assert result["n_layers"] == 3
    for key, values in result["residuals"].items():
        assert all(value < 1e-8 for value in values), (key, values)


# ----------------------------------------------------------------------------------------------
# 8. Config, refusals, run record, short training
# ----------------------------------------------------------------------------------------------

def test_config_validation_and_inactive_fields() -> None:
    latent_fields = ("content_link_latent_steps", "content_link_latent_sigma", "content_link_latent_damping",
                     "content_link_loop_weight")
    default = test_config(**_EDGE).inactive_fields()
    assert all(name in default for name in latent_fields + ("content_link_latent", "content_link_loop_window"))
    lowrank = dict(_EDGE, content_link_edges="lowrank", content_link_edge_rank=2)
    inactive = test_config(**lowrank).inactive_fields()
    assert "content_link_latent" not in inactive and all(name in inactive for name in latent_fields)
    on = test_config(**lowrank, content_link_latent=True).inactive_fields()
    assert not any(name in on for name in latent_fields) and "content_link_loop_window" in on
    looped = test_config(**lowrank, content_link_latent=True, content_link_loop_weight=0.5).inactive_fields()
    assert "content_link_loop_window" not in looped
    test_config(**lowrank, content_link_latent=True, content_link_loop_weight=0.5, content_link_loop_window=None)
    with pytest.raises(ValueError, match="lowrank"):
        test_config(**_EDGE, content_link_edges="pure_gauge", content_link_edge_rank=2, content_link_latent=True)
    with pytest.raises(ValueError, match="lowrank"):
        test_config(**_EDGE, content_link_latent=True)
    bad = [("content_link_latent_steps", 0), ("content_link_latent_sigma", 0.0),
           ("content_link_latent_sigma", math.inf), ("content_link_latent_damping", 0.0),
           ("content_link_latent_damping", 1.5), ("content_link_loop_weight", -0.1),
           ("content_link_loop_weight", math.nan), ("content_link_loop_window", 0),
           ("content_link_loop_window", 1.0), ("content_link_loop_window", True)]
    for name, value in bad:
        with pytest.raises(ValueError, match=name):
            test_config(**lowrank, content_link_latent=True, **{name: value})
    with pytest.raises(ValueError, match="content_link_latent=True"):
        test_config(**lowrank, content_link_loop_weight=0.5)
    with pytest.raises(TypeError):
        test_config(**lowrank, content_link_latent=True, content_link_latent_steps=2.0)


def test_operator_and_inference_refusals() -> None:
    kwargs = dict(content_link_chart="exponential", content_link_edges="lowrank", content_link_edge_rank=2)
    with pytest.raises(ValueError, match="lowrank"):
        make_relational_operators([2, 2], 1, **{**kwargs, "content_link_edges": "pure_gauge"}, content_link_latent=True)
    with pytest.raises(ValueError, match="content_link_latent=True"):
        make_relational_operators([2, 2], 1, **kwargs, content_link_loop_weight=1.0)
    with pytest.raises(ValueError, match="content_link_latent_damping"):
        make_relational_operators([2, 2], 1, **kwargs, content_link_latent=True, content_link_latent_damping=0.0)
    ops = _latent_ops(True, seed=25, content_link_loop_weight=1.0)
    data = _inputs(seed=26)
    mu, beta, frame, sigma = data["mu"], data["beta"], data["compact"], data["sigma_diag"]
    with pytest.raises(ValueError, match="sigma"):
        ops.predictive_content(mu, beta, frame)
    with pytest.raises(ValueError, match="link_state"):
        ops.edge_context(mu, beta, frame, sigma=sigma)
    with pytest.raises(ValueError, match="link_state"):
        _readout(ops, mu, sigma, frame)
    state: Dict = {}
    ops.edge_context(mu, beta, frame, sigma=sigma, link_state=state)  # layer 0: no previous, current written
    assert isinstance(state["current"], LinkState)
    torch.testing.assert_close(state["current"].F @ state["current"].F_inv, torch.eye(4, dtype=F64))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        ops.edge_context(mu, beta, frame, sigma=sigma, link_state={})


def test_state_dict_run_record_and_columns() -> None:
    model = _latent_model(content_link_loop_weight=0.5)
    edges_only = VFEModel(tiny_config(**{**_EDGE, "content_link_edges": "lowrank", "content_link_edge_rank": 2,
                                         "max_seq_len": 4, "n_layers": 3}))
    assert set(model.state_dict()) == set(edges_only.state_dict())     # no parameter or buffer added
    record = _relational_artifact_fields(model.cfg)["relational_operators"]
    assert record["content_link_latent"] is True and record["content_link_loop_weight"] == 0.5
    assert record["total_parameters"] == sum(p.numel() for ops in model.relational_operators_by_layer
                                             for p in ops.parameters())
    assert "content_link_latent" not in _relational_artifact_fields(edges_only.cfg)["relational_operators"]
    rows = model.diagnostics_per_layer(torch.tensor([[1, 2, 3, 4]]))
    latent_keys = sorted(key for key in rows if key.startswith("relational_latent_"))
    assert sorted("val_" + key for key in latent_keys) == sorted(T._VAL_LATENT_DIAG_KEYS)
    assert rows["relational_latent_loop_energy"][0] == 0.0
    assert all(math.isfinite(v) for key in latent_keys for v in rows[key])
    assert all(final <= initial + 1e-6 for initial, final in zip(rows["relational_latent_energy_initial"],
                                                                  rows["relational_latent_energy_final"]))
    assert not any(key.startswith("relational_latent_") for key in edges_only.diagnostics_per_layer(
        torch.tensor([[1, 2, 3, 4]])))


_ROUTES = {
    "diagonal": dict(family="gaussian_diagonal"),
    "full": dict(family="gaussian_full"),
    "passthrough": dict(predictive_covariance="passthrough"),
    "checkpoint": dict(predictive_covariance_checkpoint=True),
    "tied_frames": dict(gauge_tying="shared"),
    "every_loop": dict(content_link_loop_window=None, content_link_latent_damping=0.5),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_short_training_with_latent_factors_and_loops(route: str) -> None:
    model = _latent_model(decode_mode="linear", n_layers=2, content_link_loop_weight=0.5, **_ROUTES[route])
    cfg = model.cfg
    optimizer = T.build_optimizer(model, cfg)
    ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 2]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 6, 2, 1]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(ids, targets)
        assert math.isfinite(float(loss))
        loss.backward()
        for ops in model.relational_operators_by_layer:
            for name in ("edge_sender_map", "edge_receiver_map", "T", "O", "raw_score_q"):
                grad = getattr(ops, name).grad
                assert bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0.0, name
        optimizer.step()


def test_persisted_run_logs_the_latent_columns(tmp_path) -> None:
    import csv
    model = _latent_model(decode_mode="linear", eval_stride=2, n_layers=2, content_link_loop_weight=0.5)
    cfg = model.cfg
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    train = DataLoader(TokenWindows(torch.arange(41) % 7, 4), batch_size=2, shuffle=True)
    val = DataLoader(TokenWindows((torch.arange(23) * 3) % 7, 4, pad_final=True, stride=2), batch_size=2)
    torch.manual_seed(11)
    T.train(model, train, cfg, n_steps=4, grad_clip=1.0, log_interval=2, eval_interval=2, val_loader=val,
            device=torch.device("cpu"), logger=logging.getLogger("latent-test"), artifacts=artifacts,
            generate_samples=False)
    with (tmp_path / "run" / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [2, 4]
    for row in rows:
        assert int(row["diagnostics_failures"]) == 0
        for key in T._VAL_LATENT_DIAG_KEYS + T._VAL_EDGE_DIAG_KEYS:
            assert math.isfinite(float(row[key])), key
