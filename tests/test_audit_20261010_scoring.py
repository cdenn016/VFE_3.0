"""Projected Gaussian scorer remediation of the October 10 audit (punch rows 27, 28, 33, 45).

#27: the healthy Gram path's key factor feeds only P_j = G_j^{-1} and log det G_j; ``_GramPrecision`` returns
them with the closed-form backward -P sym(g_P) P + g_ld P, its forward bitwise the unfused ops. #28: the
diagonal moments use the source root X = S U^{-1} diag(sigma)^{1/2}, G = X X^T, a = S U^{-1} mu. #33: a tied
covariance (G_q is G_k) is factored once. #45: the factor-route pair-KL tile runs at IEEE fp32. float64
references, tiny shapes only.
"""
import math
from collections import Counter
from typing import Any, Dict, Set, Tuple

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import vfe3.relational_scoring as scoring
from vfe3.relational_scoring import projected_gaussian_kl, projected_moments
from vfe3.runtime import ieee_fp32_matmul


class _OpCounts(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter = Counter()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.counts[str(func.overloadpacket.__name__)] += 1
        return func(*args, **(kwargs or {}))


class _Policies(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.seen: Dict[str, Set[str]] = {}

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.seen.setdefault(getattr(func, "__name__", ""), set()).add(torch.get_float32_matmul_precision())
        return func(*args, **(kwargs or {}))


def _solve_policies(recorder: _Policies) -> Set[str]:
    return set().union(*(seen for name, seen in recorder.seen.items() if "solve_triangular" in name))


def _spd(shape: Tuple[int, ...], r: int, cond: float, g: torch.Generator) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(*shape, r, r, generator=g, dtype=torch.float64))
    spectrum = torch.logspace(0, -math.log10(cond), r, dtype=torch.float64).expand(*shape, r)
    G = (Q * spectrum.unsqueeze(-2)) @ Q.transpose(-1, -2)
    return .5 * (G + G.transpose(-1, -2))


def _unfused_factor(G: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    r"""The pre-#27 key factor computation: cholesky_ex, failure mask, cholesky_inverse, 2 sum log C_aa."""
    with ieee_fp32_matmul():
        chol, info = torch.linalg.cholesky_ex(G, check_errors=False)
        failed = (info != 0) | ~torch.isfinite(chol).all(dim=(-2, -1))
        eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
        P = torch.cholesky_inverse(torch.where(failed[..., None, None], eye, chol))
    return chol, failed, P, 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def _scorer_inputs(device: torch.device, dtype: torch.dtype, B: int = 2, N: int = 5, H: int = 2, r: int = 3,
                   seed: int = 0) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    a_q = torch.randn(B, N, H, r, generator=g, dtype=torch.float64)
    a_k = torch.randn(B, N, H, r, generator=g, dtype=torch.float64)
    G_q, G_k = _spd((B, N, H), r, 30., g), _spd((B, N, H), r, 30., g)
    return tuple(x.to(device=device, dtype=dtype) for x in (a_q, G_q, a_k, G_k))


# ---------------------------------------------------------------------------------------------- #27

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gram_precision_forward_is_bitwise_the_unfused_ops(device: torch.device, dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(1)
    G = _spd((4, 3), 5, 1e3, g)
    G[1, 2] = -G[1, 2]                                                        # one failed entry
    G = G.to(device=device, dtype=dtype)
    for x, y in zip(scoring._GramPrecision.apply(G), _unfused_factor(G)):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)   # bitwise; the failed log det is NaN


def _symmetrized(outputs: Tuple[int, ...]):
    def fn(A: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        chol, _, P, logdet = scoring._GramPrecision.apply(.5 * (A + A.transpose(-1, -2)))
        return tuple((chol, P, logdet)[i] for i in outputs)
    return fn


@pytest.mark.parametrize("cond", [1., 1e3])
@pytest.mark.parametrize("outputs", [(1, 2), (0,), (0, 1, 2)])
def test_gram_precision_gradcheck(device: torch.device, cond: float, outputs: Tuple[int, ...]) -> None:
    r"""(P, log det) is the healthy path; the factor output is the recovery-path consumer. cond 1e3 is SPD
    but near-singular in float64 terms of the finite-difference step."""
    g = torch.Generator().manual_seed(2)
    A = _spd((3,), 3, cond, g).to(device).requires_grad_()
    assert torch.autograd.gradcheck(_symmetrized(outputs), (A,))
    assert torch.autograd.gradgradcheck(_symmetrized(outputs), (A,))


def test_gram_precision_backward_matches_the_autograd_chain(device: torch.device) -> None:
    g = torch.Generator().manual_seed(3)
    G0 = _spd((6,), 4, 1e2, g).to(device)
    cotangents = [torch.randn(6, 4, 4, generator=g, dtype=torch.float64).to(device) for _ in range(2)]
    w_ld = torch.randn(6, generator=g, dtype=torch.float64).to(device)
    grads = []
    for factor in (scoring._GramPrecision.apply, _unfused_factor):
        G = G0.clone().requires_grad_()
        chol, _, P, logdet = factor(G)
        loss = (chol * cotangents[0]).sum() + (P * cotangents[1]).sum() + (logdet * w_ld).sum()
        grads.append(torch.autograd.grad(loss, G)[0])
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-10, atol=1e-12)


def test_gram_precision_failed_entries_receive_zero_gradient(device: torch.device) -> None:
    g = torch.Generator().manual_seed(4)
    G = _spd((3,), 3, 10., g)
    G[1] = -G[1]
    G = G.to(device).requires_grad_()
    _, failed, P, logdet = scoring._GramPrecision.apply(G)
    assert failed.tolist() == [False, True, False]
    grad, = torch.autograd.grad(P.sum() + torch.where(failed, 0., logdet).sum(), G)
    assert torch.equal(grad[1], torch.zeros_like(grad[1])) and torch.isfinite(grad).all()


@pytest.mark.parametrize("row_constant_grad", [True, False])
@pytest.mark.parametrize("tied", [False, True])
def test_healthy_scorer_forward_is_the_unfused_gemm_bitwise(device: torch.device, tied: bool,
                                                            row_constant_grad: bool) -> None:
    a_q, G_q, a_k, G_k = _scorer_inputs(device, torch.float32)
    if tied:
        a_k, G_k = a_q, G_q
    D = projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True,
                              query_row_constant_grad=row_constant_grad)
    logdet_q, (_, _, P_k, logdet_k) = _unfused_factor(G_q)[3], _unfused_factor(G_k)
    reference = scoring._projected_kl_gemm(a_q.movedim(-2, -3), a_k.movedim(-2, -3), G_q.movedim(-3, -4),
                                           P_k.movedim(-3, -4), logdet_q.transpose(-1, -2),
                                           logdet_k.transpose(-1, -2))
    assert torch.equal(D, reference)


def _direct_kl64(a_q: torch.Tensor, G_q: torch.Tensor, a_k: torch.Tensor, G_k: torch.Tensor) -> torch.Tensor:
    P = torch.linalg.inv(G_k)
    delta = a_q.unsqueeze(-3) - a_k.unsqueeze(-4)                             # (B, Nq, Nk, H, r)
    trace = torch.einsum("bihrs,bjhsr->bhij", G_q, P)
    quadratic = torch.einsum("bijhr,bjhrs,bijhs->bhij", delta, P, delta)
    logdet_q = torch.linalg.slogdet(G_q)[1].transpose(-1, -2)
    logdet_k = torch.linalg.slogdet(G_k)[1].transpose(-1, -2)
    return .5 * (trace + quadratic - a_q.shape[-1] + logdet_k.unsqueeze(-2) - logdet_q.unsqueeze(-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_healthy_scorer_gradients_match_the_float64_direct_kl(device: torch.device, dtype: torch.dtype) -> None:
    inputs = [x.requires_grad_() for x in _scorer_inputs(device, dtype)]
    reference = [x.detach().double().requires_grad_() for x in inputs]
    W = torch.randn(2, 2, 5, 5, generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(device)
    D = projected_gaussian_kl(*inputs, symmetric_inputs=True)
    grads = torch.autograd.grad((D * W.to(dtype)).sum(), inputs)
    expected = torch.autograd.grad((_direct_kl64(*reference) * W).sum(), reference)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    for x, y in zip(grads, expected):
        torch.testing.assert_close(x.double(), y, rtol=tolerance, atol=tolerance)


def test_healthy_key_backward_runs_no_triangular_solve(device: torch.device) -> None:
    a_q, G_q, a_k, G_k = (x.requires_grad_() for x in _scorer_inputs(device, torch.float32))
    D = projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True, query_row_constant_grad=False)
    with _OpCounts() as ops:
        torch.autograd.grad(D.square().sum(), (a_q, G_q, a_k, G_k))
    assert ops.counts["linalg_solve_triangular"] == 0


# ---------------------------------------------------------------------------------------------- #33

@pytest.mark.parametrize("row_constant_grad", [True, False])
def test_tied_covariance_is_factored_once(device: torch.device, row_constant_grad: bool) -> None:
    a, G0 = (x.requires_grad_() for x in _scorer_inputs(device, torch.float64)[:2])
    results = {}
    for tied in (True, False):
        G_k = G0 if tied else G0 * 1.                                         # untied: another tensor, same values
        with _OpCounts() as forward:
            D = projected_gaussian_kl(a, G0, a, G_k, symmetric_inputs=True,
                                      query_row_constant_grad=row_constant_grad)
        with _OpCounts() as backward:
            grads = torch.autograd.grad(D.square().sum(), (a, G0))
        results[tied] = (D.detach(), grads, forward.counts["linalg_cholesky_ex"],
                         backward.counts["linalg_solve_triangular"])
    assert results[True][2] == 1 and results[False][2] == 2
    assert torch.equal(results[True][0], results[False][0])
    for x, y in zip(results[True][1], results[False][1]):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-11)
    # Tied, the query log-determinant also takes the closed-form backward.
    assert results[True][3] == 0


def test_tied_factor_route_builds_one_source_root(device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(6)
    B, N, H, d = 1, 4, 2, 2
    K = H * d
    mu = torch.randn(B, N, K, generator=g, dtype=torch.float64).to(device)
    sigma = (torch.rand(B, N, K, generator=g, dtype=torch.float64) + .5).to(device)
    inverse = (torch.eye(d, dtype=torch.float64)
               + .2 * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)).to(device)
    S = torch.randn(H, d, K, generator=g, dtype=torch.float64).to(device)
    # Two head expansions per role, as relational_score builds them: distinct views of one base.
    views = [(x.unsqueeze(-2).expand(B, N, H, K) for x in (mu, sigma)) for _ in range(2)]
    (mu_q, sigma_q), (mu_k, sigma_k) = (tuple(v) for v in views)
    a, G = projected_moments(mu_q, sigma_q, inverse, S)
    calls = Counter()
    real = scoring._projected_root_factor

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls["root"] += 1
        return real(*args, **kwargs)
    monkeypatch.setattr(scoring, "_projected_root_factor", counted)
    previous = scoring.set_projected_factor_route("cholesky_qr2")
    try:
        tied = projected_gaussian_kl(a, G, a, G, projection_q=(sigma_q, inverse, S, mu_q),
                                     projection_k=(sigma_k, inverse, S, mu_k), symmetric_inputs=True)
        assert calls["root"] == 1
        untied = projected_gaussian_kl(a, G, a, G * 1., projection_q=(sigma_q, inverse, S, mu_q),
                                       projection_k=(sigma_k, inverse, S, mu_k), symmetric_inputs=True)
        assert calls["root"] == 3
    finally:
        scoring.set_projected_factor_route(previous)
    assert torch.equal(tied, untied)
    assert scoring._same_source((sigma_q, inverse, S, mu_q), (sigma_k, inverse, S, mu_k))
    assert not scoring._same_source((sigma_q, inverse, S, mu_q), (sigma_k, inverse, S, mu_k.clone()))


# ---------------------------------------------------------------------------------------------- #28

def _reference_moments(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor,
                       S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The pre-#28 formula: L = S U^{-1}, a = L mu, G = (L Sigma) L^T, per role."""
    maps = [S] if S.ndim == 3 else list(S)
    out = []
    for S_role in maps:
        L = scoring._projected_linear_map(S_role, inverse)                    # (..., N, H, r, K)
        a = (L @ mu.unsqueeze(-1)).squeeze(-1)
        G = (L * sigma.unsqueeze(-2)) @ L.transpose(-1, -2)
        out.append((a, .5 * G + .5 * G.transpose(-1, -2)))
    if S.ndim == 3:
        return out[0]
    return torch.stack([x[0] for x in out]), torch.stack([x[1] for x in out])


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("frame", ["compact", "dense"])
@pytest.mark.parametrize("shared", [True, False])
def test_diagonal_moments_match_the_reference_formula(device: torch.device, monkeypatch: pytest.MonkeyPatch,
                                                      tiled: bool, stacked: bool, frame: str,
                                                      shared: bool) -> None:
    if tiled:
        monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    g = torch.Generator().manual_seed(7)
    B, N, H, d, r = 2, 3, 2, 3, 2
    K = H * d
    mu = torch.randn(B, N, K, generator=g, dtype=torch.float64).to(device).requires_grad_()
    sigma = (torch.rand(B, N, K, generator=g, dtype=torch.float64) + .2).to(device).requires_grad_()
    if frame == "compact":
        inverse = torch.eye(d, dtype=torch.float64) + .3 * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    else:
        inverse = torch.eye(K, dtype=torch.float64) + .3 * torch.randn(B, N, H, K, K, generator=g, dtype=torch.float64)
    inverse = inverse.to(device).requires_grad_()
    S = torch.randn(*((2,) if stacked else ()), H, r, K, generator=g, dtype=torch.float64).to(device).requires_grad_()
    # relational_score's head expansion (stride 0) or materialized per-head copies (posthoc's .double()).
    def expand(x: torch.Tensor) -> torch.Tensor:
        heads = x.unsqueeze(-2).expand(B, N, H, K)
        return heads if shared else heads.contiguous()
    leaves = (mu, sigma, inverse, S)
    actual = projected_moments(expand(mu), expand(sigma), inverse, S)
    expected = _reference_moments(expand(mu), expand(sigma), inverse, S)
    weights = [torch.randn(x.shape, generator=g, dtype=torch.float64).to(device) for x in expected]
    for x, y in zip(actual, expected):
        assert x.shape == y.shape
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
    grads = torch.autograd.grad(sum((x * w).sum() for x, w in zip(actual, weights)), leaves)
    references = torch.autograd.grad(sum((x * w).sum() for x, w in zip(expected, weights)), leaves)
    for x, y in zip(grads, references):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-11)


def test_diagonal_moments_float32_stay_within_rounding_of_float64(device: torch.device) -> None:
    g = torch.Generator().manual_seed(8)
    B, N, H, d = 2, 4, 2, 4
    K = H * d
    mu = torch.randn(B, N, K, generator=g, dtype=torch.float64)
    sigma = torch.exp(.5 * torch.randn(B, N, K, generator=g, dtype=torch.float64))
    inverse = torch.eye(d, dtype=torch.float64) + .2 * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    S = torch.randn(2, H, d, K, generator=g, dtype=torch.float64) / K ** .5
    expected = _reference_moments(mu.unsqueeze(-2).expand(B, N, H, K), sigma.unsqueeze(-2).expand(B, N, H, K),
                                  inverse, S)
    mu32, sigma32, inverse32, S32 = (x.float().to(device) for x in (mu, sigma, inverse, S))
    actual = projected_moments(mu32.unsqueeze(-2).expand(B, N, H, K), sigma32.unsqueeze(-2).expand(B, N, H, K),
                               inverse32, S32)
    for x, y in zip(actual, expected):
        assert float((x.double().cpu() - y).norm() / y.norm()) < 1e-6


# ---------------------------------------------------------------------------------------------- #45

def test_factor_route_tile_runs_ieee_fp32(device: torch.device) -> None:
    g = torch.Generator().manual_seed(9)
    r = 3
    a_q, a_k = (torch.randn(1, 2, 4, r, generator=g).to(device) for _ in range(2))
    L_q, L_k = (torch.linalg.cholesky(_spd((1, 2, 4), r, 10., g)).float().to(device) for _ in range(2))
    logdet = torch.zeros(1, 2, 4, device=device)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with _Policies() as policies:
            scoring._projected_kl_tile(a_q, a_k, L_q, L_k, logdet, logdet, True)
        assert _solve_policies(policies) == {"highest"}
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_recovered_pairs_reach_the_pinned_factor_tile(device: torch.device,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A float32 Gram that loses SPD takes the factor-route tile, which now sees IEEE fp32."""
    from tests.test_relational_source_factorization import _inputs

    mu, sigma, inverse, S = _inputs(device)
    a, G = projected_moments(mu, sigma, inverse, S)
    assert (torch.linalg.cholesky_ex(G)[1] != 0).any()                         # precondition: recovery runs
    projection = (sigma, inverse, S, mu)
    calls = []
    real = scoring._projected_kl_tile

    def spy(*args: Any) -> torch.Tensor:
        calls.append(args[-1])
        return real(*args)
    monkeypatch.setattr(scoring, "_projected_kl_tile", spy)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with _Policies() as policies:
            D = projected_gaussian_kl(a, G, a, G, projection_q=projection, projection_k=projection,
                                      symmetric_inputs=True)
    finally:
        torch.set_float32_matmul_precision(previous)
    assert True in calls and torch.isfinite(D).all()
    assert _solve_policies(policies) == {"highest"}
