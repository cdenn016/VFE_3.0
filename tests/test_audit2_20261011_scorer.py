r"""Scorer and numerics remediation of the October 10 round-2 audit (punch rows 2, 12, 14, 30, 32, 39, 40, 41, 48, 55).

#2: the pair-KL GEMM backward runs at IEEE fp32 (structural check of the matmul policy seen in backward) and equals
plain autograd bit for bit on the CPU. #30: the structurally diagonal rotation-invariant route equals the dense route
and a direct-difference float64 reference, values and every gradient, and is selected exactly when P(G) is diagonal.
#12: a failed factorization under rotation_invariant with rope_fraction < 1 is rebuilt as P(X X^T) from the source.
#14: the inverse measurement from chart factors against a 60-digit reference. #32, #48: configuration. #39: the
diagonal-family floor, since audit 2026-10-12 T5 the prior floor f diag Sigma_p in every frame. #40: finite source-root gradient at zero variance. #41: non-finite chart blocks reach
matrix_exp as NaN. #55: raise-only scorer verdicts are deferred. Tiny CPU shapes only.
"""
import math
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import mpmath
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vfe3.config as config_module
import vfe3.gradients.relational as relational
import vfe3.relational_scoring as scoring
from tests.profiles import make_relational_operators, tiny_config
from vfe3.contracts import CanonicalFrameContext, deferred_host_checks
from vfe3.geometry.rope import build_rope_rotation, rope_commutant_projection, rope_rotated_pairs
from vfe3.gradients.full_link import _inverse_measurement_runs
from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import _chart_receive

F64 = torch.float64


@pytest.fixture(autouse=True)
def _restore_scorer_policies() -> Iterator[None]:
    saved = (scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE, torch.get_float32_matmul_precision())
    yield
    scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE = saved[:2]
    torch.set_float32_matmul_precision(saved[2])


class _MatmulPolicies(TorchDispatchMode):
    r"""The fp32 matmul policy in force at every aten GEMM dispatched inside the mode."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = func.overloadpacket.__name__
        if "mm" in name or "matmul" in name:
            self.seen.append(torch.get_float32_matmul_precision())
        return func(*args, **(kwargs or {}))


class _LargestOutput(TorchDispatchMode):
    r"""The largest tensor any aten op returns inside the mode, and every op name."""

    def __init__(self) -> None:
        super().__init__()
        self.numel, self.names = 0, set()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        self.names.add(func.overloadpacket.__name__)
        for x in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(x, torch.Tensor):
                self.numel = max(self.numel, x.numel())
        return out


def _spd(shape: Tuple[int, ...], r: int, g: torch.Generator) -> torch.Tensor:
    A = torch.randn(*shape, r, r, generator=g, dtype=F64)
    return A @ A.transpose(-1, -2) + .5 * torch.eye(r, dtype=F64)


def _gemm_inputs(dtype: torch.dtype, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    r"""``_projected_kl_gemm`` inputs (..., H, N, r), small variance against a large mean spread."""
    g = torch.Generator().manual_seed(seed)
    B, H, N, r = 2, 2, 5, 3
    aq = 30. * torch.randn(B, H, N, r, generator=g, dtype=F64)
    ak = 30. * torch.randn(B, H, N, r, generator=g, dtype=F64)
    q_cov = 1e-2 * _spd((B, H, N), r, g)
    k_prec = torch.linalg.inv(1e-2 * _spd((B, H, N), r, g))
    logdet_q, logdet_k = torch.randn(B, H, N, generator=g, dtype=F64), torch.randn(B, H, N, generator=g, dtype=F64)
    return tuple(x.to(dtype).requires_grad_() for x in (aq, ak, q_cov, k_prec, logdet_q, logdet_k))


# ---------------------------------------------------------------------------------------------- #2

def test_pair_kl_gemm_backward_runs_at_ieee_fp32_under_a_tf32_policy() -> None:
    r"""The caller's "high" policy reaches the plain autograd backward of the expansion, never the wrapped one's."""
    torch.set_float32_matmul_precision("high")
    for fn, expected in ((scoring._projected_kl_gemm, {"highest"}), (scoring._projected_kl_expansion, {"high"})):
        inputs = _gemm_inputs(torch.float32)
        out = fn(*inputs)
        with _MatmulPolicies() as policies:
            out.square().sum().backward()
        assert policies.seen and set(policies.seen) == expected, fn.__name__
        assert torch.get_float32_matmul_precision() == "high"                    # the process policy is restored


def test_pair_kl_gemm_equals_plain_autograd_bitwise_on_cpu() -> None:
    inputs = _gemm_inputs(torch.float32, seed=1)
    reference = tuple(x.detach().clone().requires_grad_() for x in inputs)
    out, plain = scoring._projected_kl_gemm(*inputs), scoring._projected_kl_expansion(*reference)
    assert torch.equal(out, plain)
    weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(2))
    grads = torch.autograd.grad((out * weight).sum(), inputs)
    references = torch.autograd.grad((plain * weight).sum(), reference)
    for x, y in zip(grads, references):
        assert torch.equal(x, y)


def test_pair_kl_gemm_saves_only_its_inputs_and_supports_retained_graphs() -> None:
    inputs = _gemm_inputs(F64, seed=3)
    out = scoring._projected_kl_gemm(*inputs)
    assert type(out.grad_fn).__name__.startswith("_IEEEFP32Recompute")
    first = torch.autograd.grad(out.sum(), inputs, retain_graph=True)
    second = torch.autograd.grad(out.sum(), inputs)
    for x, y in zip(first, second):
        assert torch.equal(x, y)
    with torch.no_grad():
        assert scoring._projected_kl_gemm(*inputs).grad_fn is None


# ---------------------------------------------------------------------------------------------- #30

def _ops(*, fraction: float = 1.0, tie: bool = False, d: int = 4, heads: int = 2,
         covariance: str = "rotation_invariant", base: Optional[float] = 100.0, **extra: Any) -> torch.nn.Module:
    torch.manual_seed(0)
    return make_relational_operators([d] * heads, 5, comparison_rope_base=base, comparison_rope_fraction=fraction,
                                     tie_comparison_maps=tie, rope_comparison_covariance=covariance, **extra).double()


def _beliefs(d: int, heads: int, *, full: bool, dtype: torch.dtype = F64, seed: int = 3,
             n: int = 5) -> Tuple[torch.Tensor, torch.Tensor, CanonicalFrameContext]:
    g = torch.Generator().manual_seed(seed)
    K = heads * d
    mu = torch.randn(2, n, K, generator=g, dtype=F64)
    if full:
        A = torch.randn(2, n, K, K, generator=g, dtype=F64)
        sigma = A @ A.transpose(-1, -2) + .3 * torch.eye(K, dtype=F64)
    else:
        sigma = .3 + torch.rand(2, n, K, generator=g, dtype=F64)
    blocks = torch.eye(d, dtype=F64) + .2 * torch.randn(2, n, heads, d, d, generator=g, dtype=F64)
    inverse = torch.linalg.inv(blocks)
    mu, sigma, blocks, inverse = (x.to(dtype) for x in (mu, sigma, blocks, inverse))
    frame = CanonicalFrameContext(forward=blocks, inverse=inverse.requires_grad_(), block_diagonal=True)
    return mu.requires_grad_(), sigma.requires_grad_(), frame


def _score_and_grads(ops: torch.nn.Module, mu: torch.Tensor, sigma: torch.Tensor, frame: CanonicalFrameContext,
                     dense: bool, monkeypatch: pytest.MonkeyPatch) -> Tuple[torch.Tensor, list, Dict[str, Any]]:
    with monkeypatch.context() as patch:
        if dense:
            patch.setattr(relational, "_diagonal_comparison_route", lambda *args: False)
        capture: Dict[str, Any] = {}
        D = relational_score(mu, sigma, mu, sigma, frame, ops, capture)
    weight = torch.randn(D.shape, generator=torch.Generator().manual_seed(9), dtype=D.dtype)
    leaves = [mu, sigma, frame.inverse, *(p for p in ops.parameters() if p.requires_grad)]
    return D, list(torch.autograd.grad((D * weight).sum(), leaves, allow_unused=True)), capture


def _direct_reference(ops: torch.nn.Module, mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor,
                      d: int, heads: int, fraction: float) -> torch.Tensor:
    r"""float64 direct differences: a_i = R_i S U_i^{-1} mu_i, g_i = diag P(S C_i S^T), sum over coordinates."""
    n = mu.shape[1]
    U_inv = torch.zeros(*inverse.shape[:2], heads * d, heads * d, dtype=F64)
    for h in range(heads):
        U_inv[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = inverse[:, :, h].detach()
    xi = (U_inv @ mu.detach().unsqueeze(-1)).squeeze(-1)                       # (B, N, K)
    Sigma = sigma.detach() if sigma.ndim == 4 else torch.diag_embed(sigma.detach())
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)                                # (B, N, K, K)
    R = build_rope_rotation(torch.arange(n), [d], base=100.0, fraction=fraction, dtype=F64)   # (N, d, d)
    pairs = rope_rotated_pairs(d, fraction)
    roles = []
    for S in (ops.raw_score_q, ops.raw_score_q if ops.tie_comparison_maps else ops.raw_score_k):
        S = S.detach()
        a = torch.einsum("hrk,bnk->bnhr", S, xi)
        a = torch.einsum("nst,bnht->bnhs", R, a)
        G = torch.einsum("hrk,bnkl,hsl->bnhrs", S, C, S)
        roles.append((a, torch.diagonal(rope_commutant_projection(G, pairs), dim1=-2, dim2=-1)))
    (a_q, g_q), (a_k, g_k) = roles                                            # (B, N, H, r)
    delta = a_q.unsqueeze(2) - a_k.unsqueeze(1)                                # (B, Nq, Nk, H, r)
    gq, gk = g_q.unsqueeze(2), g_k.unsqueeze(1)
    D = .5 * (gq / gk + delta.square() / gk + gk.log() - gq.log() - 1).sum(-1)  # (B, Nq, Nk, H)
    return D.permute(0, 3, 1, 2)                                               # (B, H, Nq, Nk)


@pytest.mark.parametrize("full, tie, d", [(False, False, 4), (True, False, 4), (False, True, 4), (False, False, 5)])
def test_diagonal_route_matches_dense_route_and_direct_reference(monkeypatch: pytest.MonkeyPatch, full: bool,
                                                                 tie: bool, d: int) -> None:
    heads = 2
    ops = _ops(tie=tie, d=d, heads=heads)
    mu, sigma, frame = _beliefs(d, heads, full=full)
    D, grads, capture = _score_and_grads(ops, mu, sigma, frame, False, monkeypatch)
    D_dense, grads_dense, capture_dense = _score_and_grads(ops, mu, sigma, frame, True, monkeypatch)
    torch.testing.assert_close(D, D_dense, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(D, _direct_reference(ops, mu, sigma, frame.inverse, d, heads, 1.0),
                               rtol=1e-12, atol=1e-12)
    assert sum(x is not None for x in grads) >= 4
    for x, y in zip(grads, grads_dense):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10 * float(y.abs().max()))
    for key in ("projected_covariance_pre", "projected_key_covariance_pre"):
        assert capture[key].shape == capture_dense[key].shape == (2, 5, heads, d, d)
        torch.testing.assert_close(capture[key], capture_dense[key], rtol=1e-13, atol=1e-13)


def _route_record(monkeypatch: pytest.MonkeyPatch, ops: torch.nn.Module) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    original = scoring.ROUTING_COSTS["gaussian_kl"]

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        seen.update(diagonal=kwargs.get("diagonal_covariance", False), pairs=kwargs.get("commutant_pairs"),
                    sources=kwargs.get("projection_q") is not None)
        return original(*args, **kwargs)
    monkeypatch.setitem(scoring.ROUTING_COSTS, "gaussian_kl", spy)
    d = ops.irrep_dims[0]
    mu, sigma, frame = _beliefs(d, len(ops.irrep_dims), full=False)
    relational_score(mu, sigma, mu, sigma, frame, ops)
    return seen


@pytest.mark.parametrize("case, expected", [
    (dict(), (True, None, False)),                                      # every coordinate rotates
    (dict(d=5), (True, None, False)),                                   # one unrotated coordinate, 1 x 1 block
    (dict(fraction=.5), (False, 1, True)),                              # a 2 x 2 non-rotating block
    (dict(fraction=.5, d=6), (False, 1, True)),
    (dict(covariance="full"), (False, None, True)),
    (dict(base=None), (False, None, True)),                             # no comparison RoPE
    (dict(pair_kl_gemm_tol=1e-3), (False, 2, True)),                    # the GEMM fallback keeps the dense route
    (dict(recovery=True), (False, 2, True)),                            # condition recovery keeps the dense route
])
def test_diagonal_route_is_selected_exactly_when_P_is_structurally_diagonal(
        monkeypatch: pytest.MonkeyPatch, case: Dict[str, Any], expected: Tuple[bool, Optional[int], bool]) -> None:
    case = dict(case)
    if case.pop("recovery", False):
        scoring.set_projected_condition_recovery(True)
    seen = _route_record(monkeypatch, _ops(**case))
    assert (seen["diagonal"], seen["pairs"], seen["sources"]) == expected


def test_diagonal_route_forms_no_pair_difference_tensor_and_no_factorization() -> None:
    B, H, N, r = 2, 2, 8, 4
    g = torch.Generator().manual_seed(4)
    a_q, a_k = (torch.randn(B, N, H, r, generator=g).requires_grad_() for _ in range(2))
    g_q, g_k = ((.5 + torch.rand(B, N, H, r, generator=g)).requires_grad_() for _ in range(2))
    with _LargestOutput() as largest:
        D = scoring.projected_gaussian_kl(a_q, g_q, a_k, g_k, diagonal_covariance=True)
        D.square().sum().backward()
    assert D.shape == (B, H, N, N)
    assert largest.numel < B * H * N * N * r                                  # no (B, H, N, N, r) tensor
    assert not any("cholesky" in name for name in largest.names)


def test_diagonal_route_backward_runs_at_ieee_fp32() -> None:
    torch.set_float32_matmul_precision("high")
    g = torch.Generator().manual_seed(5)
    a_q, a_k = (30. * torch.randn(1, 4, 2, 3, generator=g) for _ in range(2))
    g_q, g_k = (1e-2 + torch.rand(1, 4, 2, 3, generator=g) for _ in range(2))
    leaves = [x.requires_grad_() for x in (a_q, g_q, a_k, g_k)]
    D = scoring.projected_gaussian_kl(*leaves, diagonal_covariance=True)
    with _MatmulPolicies() as policies:
        D.sum().backward()
    assert policies.seen and set(policies.seen) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


def test_diagonal_route_verdict_raises_at_once_or_at_scope_exit() -> None:
    a = torch.zeros(1, 3, 1, 2)
    g = torch.ones(1, 3, 1, 2)
    bad = g.clone()
    bad[0, 1, 0, 0] = 0.
    with pytest.raises(ValueError, match="not positive definite"):
        scoring.projected_gaussian_kl(a, g, a, bad, diagonal_covariance=True)
    reached = False
    with pytest.raises(ValueError, match="not positive definite"):
        with deferred_host_checks():
            scoring.projected_gaussian_kl(a, g, a, bad, diagonal_covariance=True)
            reached = True
    assert reached
    with pytest.raises(ValueError, match="no GEMM fallback"):
        scoring.projected_gaussian_kl(a, g, a, g, diagonal_covariance=True, gemm_expansion_tol=1e-3)


# ---------------------------------------------------------------------------------------------- #12

@pytest.mark.parametrize("n_pairs, r, d", [(1, 4, 6), (2, 6, 8), (2, 4, 7)])
@pytest.mark.parametrize("shift", [False, True])
def test_commutant_root_factor_factors_the_projected_gram(n_pairs: int, r: int, d: int, shift: bool) -> None:
    root = torch.randn(3, r, d, generator=torch.Generator().manual_seed(6), dtype=F64)
    C, failed = scoring._commutant_root_factor(root, n_pairs, shift)
    expected = rope_commutant_projection(root @ root.transpose(-1, -2), n_pairs)
    assert not bool(failed.any())
    assert torch.equal(C, C.tril())
    torch.testing.assert_close(C @ C.transpose(-1, -2), expected, rtol=1e-13, atol=1e-13)


def test_commutant_root_factor_refuses_a_degenerate_plane() -> None:
    root = torch.randn(2, 4, 5, generator=torch.Generator().manual_seed(7), dtype=F64)
    root[1, :2] = 0.                                                          # entry 1: its first plane vanishes
    with pytest.raises(ValueError, match="lost numerical row rank"):
        scoring._commutant_root_factor(root, 1, False)
    assert scoring._commutant_root_factor(root, 1, True)[1].tolist() == [False, True]


def _inject_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Flag entry (0, 0, 0) of both roles as a failed float32 factor in the scorer's own status calls."""
    real = scoring._cholesky_status

    def failing(G: torch.Tensor, symmetric: bool = False, factor: Optional[tuple] = None) -> Tuple[torch.Tensor, ...]:
        chol, failed, screen, status = real(G, symmetric, factor)
        if not symmetric:                                                    # the recovery's own refactor
            return chol, failed, screen, status
        failed, status = failed.clone(), status.clone()
        failed[0, 0, 0] = True
        status[2] = status[3] = True
        return chol, failed, screen, status
    monkeypatch.setattr(scoring, "_cholesky_status", failing)


def _failing_retry(real: Callable[..., Tuple[torch.Tensor, torch.Tensor]]) -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
    r"""``cholesky_qr2`` whose shifted (float32 retry) calls report failure, forcing the float64 fallback."""
    def failing(root: torch.Tensor, shift: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        factor, failed = real(root, shift=shift)
        return factor, (torch.ones_like(failed) if shift else failed)
    return failing


@pytest.mark.parametrize("fallback", [False, True])
def test_rotation_invariant_recovery_rebuilds_the_commutant_projection(monkeypatch: pytest.MonkeyPatch,
                                                                       fallback: bool) -> None:
    r"""rope_fraction < 1: the non-rotating block is dense, a failed float32 factor is rebuilt from the source
    as P(X X^T) instead of raising, and every pair matches the healthy call to float32 accuracy, by float32
    shifted CholeskyQR3 and (``fallback``: that retry reports failure) by the float64 QR fallback."""
    if fallback:
        monkeypatch.setattr(scoring, "cholesky_qr2", _failing_retry(scoring.cholesky_qr2))
    d, heads = 4, 2
    ops = _ops(fraction=.5, d=d, heads=heads).float()
    mu, sigma, frame = _beliefs(d, heads, full=False, dtype=torch.float32)
    healthy = relational_score(mu, sigma, mu, sigma, frame, ops).detach()
    before = scoring.source_recovery_elements()
    with monkeypatch.context() as patch:
        _inject_failure(patch)
        recovered = relational_score(mu, sigma, mu, sigma, frame, ops)
    assert scoring.source_recovery_elements() > before
    torch.testing.assert_close(recovered.detach(), healthy, rtol=2e-4, atol=2e-4 * float(healthy.abs().max()))
    recovered.sum().backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in ops.parameters() if p.grad is not None)
    # Without source tuples, the pre-remediation call, the same failure has nothing to rebuild from.
    a = torch.zeros(1, 2, 1, 4)
    G = torch.eye(4).expand(1, 2, 1, 4, 4).contiguous()
    with monkeypatch.context() as patch:
        _inject_failure(patch)
        with pytest.raises(ValueError, match="not positive definite"):
            scoring.projected_gaussian_kl(a, G, a, G, symmetric_inputs=True, commutant_pairs=1)


# ---------------------------------------------------------------------------------------------- #14

def _chart_case(cond: float, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""K = 4 stacks with cond(T) = ``cond`` and the per-run (n_r, d_r, K) / (n_r, K, d_r) maps of two heads."""
    K = 4
    g = torch.Generator().manual_seed(seed)
    Q1, Q2 = (torch.linalg.qr(torch.randn(K, K, generator=g, dtype=F64)).Q for _ in range(2))
    T_stack = (Q1 * torch.logspace(0, -math.log10(cond), K, dtype=F64)) @ Q2.T
    O_stack = .5 * torch.randn(K, K, generator=g, dtype=F64)
    return T_stack, O_stack, T_stack.reshape(2, 2, K), O_stack.reshape(K, 2, 2).permute(1, 0, 2)


def _mp_inverse_measurement(T_stack: torch.Tensor, O_stack: torch.Tensor, chart: str) -> torch.Tensor:
    with mpmath.workdps(60):
        T, O = mpmath.matrix(T_stack.tolist()), mpmath.matrix(O_stack.tolist())
        receive = (mpmath.expm(O * T) * mpmath.inverse(T) if chart == "exponential"
                   else mpmath.inverse(T) + O if chart == "first_order" else O)
        P = mpmath.inverse(receive)
        return torch.tensor([[float(P[i, j]) for j in range(P.cols)] for i in range(P.rows)], dtype=F64)


@pytest.mark.parametrize("chart", ["exponential", "first_order"])
def test_inverse_measurement_from_chart_factors_matches_a_60_digit_reference(chart: str) -> None:
    T_stack, O_stack, T, O = _chart_case(1e12)
    reference = _mp_inverse_measurement(T_stack, O_stack, chart)
    P = torch.cat([p.reshape(-1, 4) for p in _inverse_measurement_runs([T], [O], chart, F64)])
    old = torch.linalg.inv_ex(_chart_receive(T_stack, O_stack, chart)[1]).inverse   # inv_ex of the float64 O'
    error = lambda X: float((X - reference).norm() / reference.norm())
    assert error(P) < 1e-13
    assert error(old) > 1e-8


def test_affine_inverse_measurement_is_certified() -> None:
    T_stack, O_stack, T, O = _chart_case(10.)
    P = torch.cat([p.reshape(-1, 4) for p in _inverse_measurement_runs([T], [O], "affine", F64)])
    torch.testing.assert_close(P, _mp_inverse_measurement(T_stack, O_stack, "affine"), rtol=1e-13, atol=1e-13)
    O_bad = _chart_case(1e12)[0].reshape(4, 2, 2).permute(1, 0, 2)           # kappa_F(O) ~ 1e12 > 5.4e8
    with pytest.raises(ValueError, match="inverse receiver measurement refused"):
        _inverse_measurement_runs([T], [O_bad], "affine", F64)
    reached = False
    with pytest.raises(ValueError, match="inverse receiver measurement refused"):
        with deferred_host_checks():
            _inverse_measurement_runs([T], [O_bad], "affine", F64)
            reached = True
    assert reached


# ---------------------------------------------------------------------------------------------- #32, #48

def _invariant_config(**overrides: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(embed_dim=8, rope_mode="comparison_invariant", **overrides)


def test_rotation_invariant_with_every_coordinate_rotating_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_DIAGONAL_COMPARISON_WARNED", set())
    values = dict(embed_dim=128, n_heads=4, max_seq_len=128, rope_base=10000.0, rope_mode="comparison_invariant")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiny_config(**values)
        tiny_config(**values)
        tiny_config(**{**values, "rope_fraction": .5})                       # a dense non-rotating block
    messages = [str(w.message) for w in caught if "P(G) is diagonal" in str(w.message)]
    assert len(messages) == 1
    assert "6 of 16 planes for heads of size 32" in messages[0] and "0.5 rad" in messages[0]
    # Plane k turns by 127 * 1e4^(-2k/32); planes 10..15 turn by less than 0.5 rad.
    assert sum(127 * 1e4 ** (-2 * k / 32) < .5 for k in range(16)) == 6


# ---------------------------------------------------------------------------------------------- #39

def test_diagonal_family_floor_bound() -> None:
    r"""The diagonal family's floor is f diag Sigma_p in local coordinates, whatever the frames (audit 2026-10-12 T5).

    The former reference floor f tau I survived the diagonal projection only as diag(U (f tau I) U^T), whose
    guarantee was up to d times weaker than the full family's. With zero content maps and no residual the kernel
    covariance vanishes, so the readout is exactly f Sigma_p for frames of growing non-orthogonality, and with the
    content restored every variance exceeds f sigma_p (the bound is the same in both families)."""
    g = torch.Generator().manual_seed(8)
    H, d, N, f = 2, 2, 3, .3
    K = H * d
    ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                    predictive_covariance="moment_matched", predictive_covariance_floor=f).double()
    mu = torch.randn(1, N, K, generator=g, dtype=F64)
    sigma = torch.rand(1, N, K, generator=g, dtype=F64) + .3
    prior = torch.rand(1, N, K, generator=g, dtype=F64) + .4
    beta = torch.softmax(torch.randn(1, H, N, N, generator=g, dtype=F64), dim=-1)
    O = ops.O.detach().clone()
    for scale in (.1, .5, 1.):
        blocks = torch.linalg.matrix_exp(scale * torch.randn(1, N, H, d, d, generator=g, dtype=F64))
        frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
        with torch.no_grad():
            ops.O.zero_()
        torch.testing.assert_close(ops.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=prior),
                                   f * prior, rtol=1e-12, atol=0)
        with torch.no_grad():
            ops.O.copy_(O)
        content = ops.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=prior)
        assert bool((content - f * prior > 0).all())


# ---------------------------------------------------------------------------------------------- #40

def _plain_root_moments(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor, S: torch.Tensor) -> tuple:
    r"""The pre-remediation compact-block source root, ``root = sigma.sqrt()``, for a per-head mean and variance."""
    R, H, r, K = S.shape
    M, d = inverse.shape[-3], inverse.shape[-1]
    blocks = inverse.unsqueeze(-4)
    xi = (blocks @ mu.unflatten(-1, (M, d)).unsqueeze(-1)).flatten(-3)
    W = blocks * sigma.sqrt().unflatten(-1, (M, d)).unsqueeze(-2)
    X = torch.einsum("Rhrmd,...hmde->...Rhrme", S.unflatten(-1, (M, d)), W).flatten(-2)
    return torch.einsum("Rhrk,...hk->...Rhr", S, xi), X


def test_source_root_has_a_finite_gradient_at_zero_variance() -> None:
    g = torch.Generator().manual_seed(10)
    H, d = 2, 3
    K = H * d
    mu = torch.randn(2, 4, H, K, generator=g, dtype=F64)
    sigma = torch.rand(2, 4, H, K, generator=g, dtype=F64) + .1
    sigma[0, 1, 0, 2] = 0.
    sigma[1, 2, 1, 4] = 1e-310                                               # a subnormal positive variance
    inverse = torch.eye(d, dtype=F64) + .3 * torch.randn(2, 4, H, d, d, generator=g, dtype=F64)
    S = torch.randn(1, H, 2, K, generator=g, dtype=F64)
    positive = sigma.clone()
    positive[0, 1, 0, 2] = .7
    for x, y in zip(scoring._diagonal_root_moments(mu, positive, inverse, S),
                    _plain_root_moments(mu, positive, inverse, S)):
        assert torch.equal(x, y)                                             # sigma > 0: bit for bit
    sigma.requires_grad_()
    a, G = scoring.projected_moments(mu, sigma, inverse, S[0])
    grad, = torch.autograd.grad(G.sum(), sigma)
    assert bool(torch.isfinite(grad).all()) and float(grad[0, 1, 0, 2]) == 0.
    plain = sigma.detach().clone().requires_grad_()
    _, X = _plain_root_moments(mu, plain, inverse, S)
    assert not bool(torch.isfinite(torch.autograd.grad((X @ X.transpose(-1, -2)).sum(), plain)[0]).all())


# ---------------------------------------------------------------------------------------------- #41

def test_nonfinite_chart_blocks_reach_matrix_exp_as_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.geometry.transport as transport
    seen: List[Tuple[bool, bool]] = []
    real = torch.linalg.matrix_exp

    def recording(x: torch.Tensor) -> torch.Tensor:
        seen.append((bool(torch.isinf(x).any()), bool(torch.isnan(x).any())))
        return real(x)
    monkeypatch.setattr(torch.linalg, "matrix_exp", recording)
    blocks = .1 * torch.randn(2, 3, 2, 2, generator=torch.Generator().manual_seed(11))
    exp_pos, _ = transport._stable_compact_glk_exp_pair(blocks, validity_max_norm=10.0)
    assert torch.equal(exp_pos, real(blocks))                                # finite blocks: unchanged
    seen.clear()
    bad = blocks.clone()
    bad[1, 2, 0, 1] = float("inf")
    with pytest.raises(ValueError, match="nonfinite matrix norm"):
        with deferred_host_checks():
            transport._stable_compact_glk_exp_pair(bad, validity_max_norm=10.0)
    assert seen and seen[0] == (False, True)


# ---------------------------------------------------------------------------------------------- #55

def _counted_host_reads(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    calls: List[int] = []
    real = scoring.host_read
    monkeypatch.setattr(scoring, "host_read", lambda values: calls.append(1) or real(values))
    return calls


def _kl_inputs() -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(12)
    a_q, a_k = torch.randn(1, 4, 2, 3, generator=g), torch.randn(1, 4, 2, 3, generator=g)
    G_q, G_k = (_spd((1, 4, 2), 3, g).float() for _ in range(2))
    return a_q, G_q, a_k, G_k


def test_raise_only_scorer_verdict_is_deferred_and_takes_the_healthy_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    a_q, G_q, a_k, G_k = _kl_inputs()
    outside = scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True)
    calls = _counted_host_reads(monkeypatch)
    with deferred_host_checks():
        inside = scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True)
        assert not calls                                                     # no per-call host read
    assert torch.equal(inside, outside)
    # A recovery that could act (a source tuple, or condition recovery) keeps the decision read.
    source = (torch.ones(1, 4, 2, 3), torch.eye(3).expand(1, 4, 2, 3, 3), torch.eye(3).expand(2, 3, 3))
    with deferred_host_checks():
        scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True, projection_q=source)
    assert len(calls) == 1
    scoring.set_projected_condition_recovery(True)
    with deferred_host_checks():
        scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True)
    assert len(calls) == 2


def test_deferred_scorer_verdict_still_raises_in_order() -> None:
    a_q, G_q, a_k, G_k = _kl_inputs()
    G_k = G_k.clone()
    G_k[0, 2, 1] = -G_k[0, 2, 1]                                              # one key entry is not SPD
    with pytest.raises(ValueError, match="not positive definite"):
        scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True)
    reached = False
    with pytest.raises(ValueError, match="not positive definite"):
        with deferred_host_checks():
            scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, symmetric_inputs=True)
            reached = True
    assert reached
    a_bad = a_q.clone()
    a_bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="projected means must contain only finite values"):
        with deferred_host_checks():
            scoring.projected_gaussian_kl(a_bad, G_q, a_k, G_k, symmetric_inputs=True)
