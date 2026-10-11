r"""Dual receiving measurement of the channel_likelihood receiver (audit 2026-10-10, item 11).

``relational_receiver_measurement='inverse'`` measures :math:`\xi_i` on head h by the head-h rows of
:math:`O'^{-1}` instead of :math:`O_h'^\top`. Pins, on CPU at tiny size in float64: with equal channel
precisions the alpha -> 0 frozen-context minimizer is the kernel mean; the minimizer is invariant under
:math:`(U h, T h, S h, h^{-1}O')`; the default 'adjoint' context is unchanged; config -> model -> operator
wiring and one finite training step under both receiver updates; and the gradient of the context build.
"""
from typing import Any, Dict, List, Tuple

import pytest
import torch

from tests.profiles import make_relational_operators
from tests.test_projected_gradient import _cfg, _tokens
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients import full_link
from vfe3.gradients.full_link import build_full_link_context
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators

DTYPE = torch.float64
N = 3
CAUSAL = torch.tensor([[0., -torch.inf, -torch.inf], [0., 0., -torch.inf], [0., 0., 0.]], dtype=DTYPE)


@pytest.fixture(autouse=True)
def _single_cpu_thread() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _operators(dims: List[int], measurement: str, chart: str = "affine") -> RelationalOperators:
    r"""Float64 channel-law operators moved off the identity start, where O' is orthogonal."""
    ops = make_relational_operators(dims, 5, readout_mode="full_link_update", receiver_law="channel_likelihood",
                                    receiver_measurement=measurement, content_link_chart=chart).double()
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in ops.parameters():
            parameter.add_(0.4 * torch.randn(parameter.shape, generator=generator, dtype=DTYPE))
    return ops


def _block_frames(dims: List[int], seed: int, orthogonal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Head-block-diagonal (1, N, K, K) frames U and U^{-1}: rotations, or exp of random blocks."""
    generator = torch.Generator().manual_seed(seed)
    forward, inverse = [], []
    for _ in range(N):
        blocks = [torch.randn(d, d, generator=generator, dtype=DTYPE) for d in dims]
        blocks = [torch.linalg.qr(b).Q if orthogonal else torch.linalg.matrix_exp(.6 * b) for b in blocks]
        forward.append(torch.block_diag(*blocks))
        inverse.append(torch.block_diag(*(torch.linalg.inv(b) for b in blocks)))
    return torch.stack(forward)[None], torch.stack(inverse)[None]


def _minimizer(context: full_link.FullLinkContext) -> torch.Tensor:
    r"""Frozen-context mean minimizer mu* = Lambda^{-1} eta, (1, N, K)."""
    return torch.linalg.solve(context.Lambda, context.eta.unsqueeze(-1)).squeeze(-1)


def _state(K: int, seed: int) -> Tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(1, N, K, generator=generator, dtype=DTYPE)
    sigma = torch.rand(1, N, K, generator=generator, dtype=DTYPE) + .5
    mu_p = torch.randn(1, N, K, generator=generator, dtype=DTYPE)
    sigma_p = torch.rand(1, N, K, generator=generator, dtype=DTYPE) + .5
    return mu, sigma, mu_p, sigma_p


def _reference_change(ops: RelationalOperators, h: torch.Tensor) -> None:
    r"""(T, S_Q, S_K, O) -> (T h, S_Q h, S_K h, h^{-1} O), which maps O' -> h^{-1} O' in every chart."""
    h_inv = torch.linalg.inv(h)
    with torch.no_grad():
        for name in ("raw_score_q", "raw_score_k", "T"):
            for X in ops._stored_runs(name):
                X.copy_(X @ h)
        for O in ops._stored_runs("O"):
            O.copy_(h_inv @ O)


def kernel_mean_gaps(dims: List[int]) -> Dict[str, float]:
    r"""max |mu* - kernel mean| per measurement, equal channel precisions and alpha / sigma_p = 1e-24.

    Orthonormal sending rows (T_h T_h^T = I), rotation frames and a constant source variance s give
    R_hj = (s + nu) I for every (h, j), so W_ih = I / (s + nu) for scalar lambda_beta.
    """
    K = sum(dims)
    Q = torch.linalg.qr(torch.randn(K, K, generator=torch.Generator().manual_seed(2), dtype=DTYPE)).Q
    U, U_inv = _block_frames(dims, seed=3, orthogonal=True)
    frame = CanonicalFrameContext(U, U_inv)
    mu = torch.randn(1, N, K, generator=torch.Generator().manual_seed(4), dtype=DTYPE)
    sigma = torch.full((1, N, K), .6, dtype=DTYPE)
    gaps = {}
    for measurement in ("adjoint", "inverse"):
        ops = _operators(dims, measurement)
        with torch.no_grad():
            start = 0
            for T in ops._stored_runs("T"):
                n, d = T.shape[0], T.shape[1]
                T.copy_(Q[start:start + n * d].reshape(n, d, K))
                start += n * d
        context = build_full_link_context(mu, sigma, mu, sigma * 1e12, frame, ops, value=1e-12,
                                          receiver_noise=.7, log_prior=CAUSAL)
        kernel = ops.predictive_content(mu, context.beta, frame)                     # (1, N, K)
        gaps[measurement] = float((_minimizer(context) - kernel).abs().max())
    return gaps


def reference_change_moves(dims: List[int], chart: str, value: float) -> Dict[str, float]:
    r"""max |mu*(U h, T h, S h, h^{-1} O') - mu*(U, T, S, O')| per measurement, block h."""
    K = sum(dims)
    U, U_inv = _block_frames(dims, seed=11, orthogonal=False)
    h, h_inv = (x[0, 0] for x in _block_frames(dims, seed=12, orthogonal=False))
    mu, sigma, mu_p, sigma_p = _state(K, seed=13)
    moves = {}
    for measurement in ("adjoint", "inverse"):
        ops = _operators(dims, measurement, chart)
        build = lambda f: build_full_link_context(mu, sigma, mu_p, sigma_p, f, ops, value=value,
                                                  receiver_noise=.7, log_prior=CAUSAL)
        before = build(CanonicalFrameContext(U, U_inv))
        kernel_before = ops.predictive_content(mu, before.beta, CanonicalFrameContext(U, U_inv))
        _reference_change(ops, h)
        frame_h = CanonicalFrameContext(U @ h, h_inv @ U_inv)
        after = build(frame_h)
        # The construction leaves routing and the kernel mean invariant.
        torch.testing.assert_close(after.beta, before.beta, atol=1e-12, rtol=0)
        torch.testing.assert_close(ops.predictive_content(mu, after.beta, frame_h), kernel_before,
                                   atol=1e-11, rtol=0)
        moves[measurement] = float((_minimizer(after) - _minimizer(before)).abs().max())
    return moves


@pytest.mark.parametrize("dims", [[2, 2], [2, 3]])
def test_inverse_minimizer_is_the_kernel_mean_with_equal_channel_precisions(dims: List[int]) -> None:
    gaps = kernel_mean_gaps(dims)
    assert gaps["inverse"] < 1e-12, gaps
    assert gaps["adjoint"] > 1e-2, gaps


@pytest.mark.parametrize("chart", ["affine", "exponential"])
@pytest.mark.parametrize("dims", [[2, 2], [2, 3]])
@pytest.mark.parametrize("value", [1e-12, .4])
def test_inverse_minimizer_is_invariant_under_the_reference_change(
        dims: List[int], chart: str, value: float) -> None:
    moves = reference_change_moves(dims, chart, value)
    assert moves["inverse"] < 1e-10, moves
    assert moves["adjoint"] > 1e-3, moves


def test_default_measurement_is_the_unchanged_adjoint() -> None:
    assert make_relational_operators([2, 2], 5).receiver_measurement == "adjoint"
    with pytest.raises(ValueError, match="receiver_measurement"):
        make_relational_operators([2, 2], 5, receiver_measurement="other")
    mu, sigma, mu_p, sigma_p = _state(5, seed=21)
    U, U_inv = _block_frames([2, 3], seed=22, orthogonal=False)
    frame = CanonicalFrameContext(U, U_inv)
    ops = _operators([2, 3], "adjoint")
    build = lambda: build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, value=.4,
                                            receiver_noise=.7, log_prior=CAUSAL)
    default = build()
    del ops.receiver_measurement                                     # the pre-option operator
    legacy = build()
    for name in ("Lambda", "eta", "c", "beta", "score"):
        assert torch.equal(getattr(default, name), getattr(legacy, name)), name
    ops.receiver_measurement = "inverse"
    assert not torch.allclose(build().Lambda, default.Lambda)
    # ambient_alignment never reads the measurement.
    ops.receiver_law = "ambient_alignment"
    ambient = build()
    ops.receiver_measurement = "adjoint"
    assert torch.equal(build().Lambda, ambient.Lambda)


@pytest.mark.parametrize("update", ["closed_form", "euler"])
def test_tiny_model_trains_one_step_under_the_inverse_measurement(
        update: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    original = full_link._inverse_measurement_runs
    monkeypatch.setattr(full_link, "_inverse_measurement_runs",
                        lambda *args: calls.append(1) or original(*args))
    ids, targets = _tokens()
    losses = {}
    for measurement in ("adjoint", "inverse"):
        cfg = _cfg(relational_receiver_law="channel_likelihood", relational_receiver_update=update,
                   relational_receiver_measurement=measurement)
        torch.manual_seed(0)
        model = VFEModel(cfg)
        layers = model.relational_operators_by_layer
        assert all(ops.receiver_measurement == measurement for ops in layers)
        generator = torch.Generator().manual_seed(3)
        with torch.no_grad():                    # off the orthogonal identity start, where both agree
            for ops in layers:
                ops.O.add_(.3 * torch.randn(ops.O.shape, generator=generator))
        _, loss, _ = model(ids, targets)
        loss.backward()
        assert torch.isfinite(loss)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
        for ops in layers:
            assert float(ops.O.grad.abs().max()) > 0
        torch.optim.SGD(model.parameters(), lr=1e-2).step()
        _, after, _ = model(ids, targets)
        assert torch.isfinite(after)
        losses[measurement] = float(loss)
    assert calls, "the inverse measurement never ran"
    assert losses["inverse"] != losses["adjoint"]


@pytest.mark.parametrize("chart", ["affine", "exponential"])
def test_context_build_gradient_under_the_inverse_measurement(chart: str) -> None:
    dims = [2, 3]
    ops = _operators(dims, "inverse", chart)
    mu, sigma, mu_p, sigma_p = _state(5, seed=31)
    U, U_inv = _block_frames(dims, seed=32, orthogonal=False)
    frame = CanonicalFrameContext(U, U_inv)

    def build(m: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context = build_full_link_context(m, s, mu_p, sigma_p, frame, ops, value=.4,
                                          receiver_noise=.7, log_prior=CAUSAL)
        return context.Lambda, context.eta

    assert torch.autograd.gradcheck(build, (mu.clone().requires_grad_(), sigma.clone().requires_grad_()))
    # Stored maps (including O'^{-1}): central differences of a fixed linear functional along a random direction.
    generator = torch.Generator().manual_seed(33)
    A = torch.randn(1, N, 5, 5, generator=generator, dtype=DTYPE)
    b = torch.randn(1, N, 5, generator=generator, dtype=DTYPE)
    functional = lambda: (lambda L, e: (L * A).sum() + (e * b).sum())(*build(mu, sigma))
    parameters = list(ops.parameters())
    grads = torch.autograd.grad(functional(), parameters)
    directions = [torch.randn(p.shape, generator=generator, dtype=DTYPE) for p in parameters]
    analytic = sum(float((g * D).sum()) for g, D in zip(grads, directions))
    eps = 1e-6
    with torch.no_grad():
        for p, D in zip(parameters, directions):
            p.add_(eps * D)
        plus = float(functional())
        for p, D in zip(parameters, directions):
            p.sub_(2 * eps * D)
        minus = float(functional())
        for p, D in zip(parameters, directions):
            p.add_(eps * D)
    numeric = (plus - minus) / (2 * eps)
    assert abs(analytic - numeric) <= 1e-7 * max(1.0, abs(numeric)), (analytic, numeric)
