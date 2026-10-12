"""Square-root projected factors (optimization_plan.md WS1) and their certificate (WS2), 2026-09-24.

CholeskyQR2 returns the lower Cholesky factor C of G = X X^T without forming the float32 Gram as the
factorized matrix, so its error scales with kappa(X) instead of kappa(X)^2. The triangular
condition estimate is the Hager-Higham 1-norm lower bound used by the factor certificate.
"""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vfe3.numerics import cholesky_qr2, triangular_condition_estimate


def _conditioned_root(kappa: float, rows: int = 4, cols: int = 8, seed: int = 0) -> torch.Tensor:
    r"""float64 X (rows, cols) with singular values geometric from 1 to 1/kappa."""
    generator = torch.Generator().manual_seed(seed)
    left = torch.linalg.qr(torch.randn(rows, rows, generator=generator, dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(cols, rows, generator=generator, dtype=torch.float64))[0]
    singular = torch.logspace(0., -torch.log10(torch.tensor(kappa)).item(), rows, dtype=torch.float64)
    return (left * singular) @ right.T


def _whitened_error(C: torch.Tensor, X: torch.Tensor) -> float:
    r"""||C_ref^{-1} C - I||_F: the computed Gaussian's error in the exact one's own frame.

    This is the quantity the KL consumes (normwise ||C - C_ref|| / ||C_ref|| hides the error in the
    weak directions). ``C_ref`` is the float64 factor of the Gram of the same float32 input.
    """
    reference = torch.linalg.cholesky(X.double() @ X.double().transpose(-1, -2))
    relative = torch.linalg.solve_triangular(reference, C.double(), upper=False)
    return (relative - torch.eye(C.shape[-1], dtype=torch.float64, device=C.device)).norm().item()


@pytest.mark.parametrize("kappa", [10., 1e3, 3e3])
def test_cholesky_qr2_factor_error_scales_with_kappa_not_kappa_squared(device: torch.device,
                                                                     kappa: float) -> None:
    X64 = _conditioned_root(kappa).to(device)
    X = X64.float()
    C, failed = cholesky_qr2(X)
    assert C.dtype == torch.float32 and not bool(failed)
    assert _whitened_error(C, X) <= 4 * torch.finfo(torch.float32).eps * kappa
    assert torch.equal(C, torch.tril(C)) and bool((C.diagonal() > 0).all())


def test_gram_factor_would_miss_the_kappa_bound(device: torch.device) -> None:
    r"""Guard the test above: the explicit float32 Gram route is kappa^2-limited at kappa = 3e3."""
    X = _conditioned_root(3e3).to(device).float()
    assert _whitened_error(torch.linalg.cholesky(X @ X.T), X) > 20 * torch.finfo(torch.float32).eps * 3e3


def test_cholesky_qr2_batched_and_reports_step_one_failure(device: torch.device) -> None:
    X = torch.stack([_conditioned_root(10., seed=seed) for seed in range(3)]).to(device).float()
    X[1] = 0.                                               # rank zero: step-one Cholesky fails
    C, failed = cholesky_qr2(X)
    assert failed.tolist() == [False, True, False]
    assert torch.isfinite(C[0]).all() and torch.isfinite(C[2]).all()


def test_shifted_cholesky_qr3_recovers_beyond_the_cholesky_qr2_range(device: torch.device) -> None:
    r"""kappa = 1e5 exceeds CholeskyQR2's guaranteed range (kappa^2 eps >> 1).

    Whether unshifted step one then fails is platform-dependent (CPU LAPACK rejects the Gram, while
    cuSOLVER accepted it in the 2026-09-24 CUDA run), so only the shifted guarantee is asserted.
    """
    X = _conditioned_root(1e5).to(device).float()
    C, failed = cholesky_qr2(X, shift=True)
    assert not bool(failed)
    assert _whitened_error(C, X) <= 4 * torch.finfo(torch.float32).eps * 1e5


def test_cholesky_qr2_gradient_matches_finite_differences() -> None:
    X = _conditioned_root(5., rows=3, cols=5).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: cholesky_qr2(x)[0], (X,))


@pytest.mark.parametrize("kappa", [3., 1e2, 1e4])
def test_triangular_condition_estimate_brackets_the_true_1_norm_condition(device: torch.device,
                                                                          kappa: float) -> None:
    X = torch.stack([_conditioned_root(kappa, seed=seed) for seed in range(8)]).to(device)
    C = torch.linalg.cholesky(X @ X.transpose(-1, -2))
    true = torch.linalg.matrix_norm(C, ord=1) * torch.linalg.matrix_norm(torch.linalg.inv(C), ord=1)
    estimate = triangular_condition_estimate(C)
    assert estimate.shape == true.shape
    assert bool((estimate <= true * (1 + 1e-10)).all())       # a lower bound
    assert bool((estimate >= true / 3).all())                 # and a tight one here


@pytest.fixture
def factor_route():
    from vfe3.relational_scoring import projected_factor_route, set_projected_factor_route

    previous = projected_factor_route()
    yield set_projected_factor_route
    set_projected_factor_route(previous)


def test_factor_route_config_validated_and_active_on_supported_routes() -> None:
    from tests.profiles import tiny_config
    from tests.test_relational_integration import _cfg
    assert _cfg(projected_factorization="gram").projected_factorization == "gram"
    with pytest.raises(ValueError, match="projected_factorization"):
        _cfg(projected_factorization="qr")
    assert "projected_factorization" not in _cfg().inactive_fields()
    assert "projected_factorization" in _cfg(attention_mode="projected_predictive",

                                               relational_routing="mean_distance").inactive_fields()


@pytest.mark.parametrize("route", ["gram", "cholesky_qr2"])
def test_model_installs_the_factor_route(factor_route, route: str) -> None:
    from tests.test_relational_integration import _cfg
    from vfe3.model.model import VFEModel
    from vfe3.relational_scoring import projected_factor_route

    factor_route("cholesky_qr2" if route == "gram" else "gram")
    VFEModel(_cfg(projected_factorization="gram" if route == "gram" else "cholesky_qr2_certified"))
    assert projected_factor_route() == route


def test_set_factor_route_rejects_unknown_routes(factor_route) -> None:
    with pytest.raises(ValueError, match="projected factor route"):
        factor_route("householder")


def _projected_case(device: torch.device, dtype: torch.dtype = torch.float32, kappa: float = 300.,
                    tokens: int = 5) -> tuple:
    r"""Diagonal-covariance projected inputs whose query roots have kappa(X) ~ ``kappa``.

    Two heads of width 3 read the full 6-coordinate belief (full input scope), rank r = 3.
    """
    generator = torch.Generator().manual_seed(11)
    H, d, K = 2, 3, 6
    mu = torch.randn(1, tokens, K, generator=generator, dtype=torch.float64)
    sigma = torch.exp(torch.randn(1, tokens, K, generator=generator, dtype=torch.float64) * .3)
    blocks = torch.linalg.matrix_exp(.3 * torch.randn(1, tokens, H, d, d, generator=generator,
                                                      dtype=torch.float64))
    S_k = torch.randn(H, d, K, generator=generator, dtype=torch.float64)
    # Query map with singular values geometric from 1 to 1/kappa in each head.
    S_q = torch.stack([_conditioned_root(kappa, d, K, seed=seed) for seed in range(H)])
    mu_h, sigma_h = (x.unsqueeze(-2).expand(1, tokens, H, K) for x in (mu, sigma))
    cast = [x.to(device=device, dtype=dtype) for x in (mu_h, sigma_h, blocks, S_q, S_k)]
    return tuple(cast)


def _score(mu_h, sigma_h, blocks, S_q, S_k) -> torch.Tensor:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    aq, Gq = projected_moments(mu_h, sigma_h, blocks, S_q)
    ak, Gk = projected_moments(mu_h, sigma_h, blocks, S_k)
    return projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=(sigma_h, blocks, S_q, mu_h),
                                 projection_k=(sigma_h, blocks, S_k, mu_h))


def test_factor_route_error_scales_with_kappa_and_beats_the_gram_route(factor_route,
                                                                      device: torch.device) -> None:
    case = _projected_case(device, kappa=1e3)
    reference = _score(*(x.double() for x in case))
    factor_route("gram")
    gram = (_score(*case).double() - reference).abs().max().item()
    factor_route("cholesky_qr2")
    factored = _score(*case)
    assert factored.dtype == torch.float32
    factor_error = (factored.double() - reference).abs().max().item()
    assert factor_error < gram / 10
    # First-order model: |dKL| <= eps (kappa_q + kappa_k) (tr + quad + r) with kappa_q ~ 1e3 here.
    assert factor_error <= torch.finfo(torch.float32).eps * 2e3 * reference.abs().max().item()


def test_factor_route_gradients_match_finite_differences(factor_route) -> None:
    factor_route("cholesky_qr2")
    case = [x.detach().clone().requires_grad_(True)
            for x in _projected_case(torch.device("cpu"), torch.float64, kappa=5., tokens=3)]
    mu_h, sigma_h, blocks, S_q, S_k = case

    def score(mu, sigma, U, Sq, Sk):
        return _score(mu, sigma, U, Sq, Sk)

    assert torch.autograd.gradcheck(score, (mu_h, sigma_h, blocks, S_q, S_k))


def test_factor_route_float32_gradients_agree_with_the_float64_explicit_route(
        factor_route, device: torch.device) -> None:
    case = _projected_case(device, kappa=30.)
    factor_route("gram")
    reference = [x.double().detach().requires_grad_(True) for x in case]
    _score(*reference).sum().backward()
    factor_route("cholesky_qr2")
    working = [x.detach().requires_grad_(True) for x in case]
    _score(*working).sum().backward()
    for got, want in zip(working, reference):
        torch.testing.assert_close(got.grad.double(), want.grad, rtol=2e-3, atol=2e-3)


class _OpNames(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.names: set = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def _nearly_dependent_case(device: torch.device) -> list:
    r"""Head 0's third query row is its first row plus 1e-5 v: kappa(X) ~ 1e5, beyond CholeskyQR2."""
    case = [x.detach().clone() for x in _projected_case(device, torch.float64, kappa=3.)]
    generator = torch.Generator().manual_seed(3)
    direction = torch.randn(case[3].shape[-1], generator=generator, dtype=torch.float64).to(device)
    case[3][0, 2] = case[3][0, 0] + 1e-5 * direction / direction.norm()
    return [x.float() for x in case]


@pytest.mark.parametrize("recovery", [False, True])
def test_factor_route_recovers_a_failed_root_with_finite_gradients(
        factor_route, device: torch.device, recovery: bool) -> None:
    from vfe3.relational_scoring import (
        projected_condition_recovery, reset_source_recovery_elements, set_projected_condition_recovery,
        source_recovery_elements,
    )

    factor_route("cholesky_qr2")
    case = _nearly_dependent_case(device)
    previous = set_projected_condition_recovery(recovery)
    try:
        working = [x.clone().requires_grad_(True) for x in case]
        reset_source_recovery_elements()
        ops = _OpNames()
        with ops:
            score = _score(*working)
        # Off: the outright CholeskyQR2 failure is rebuilt by float32 shifted CholeskyQR3 only.
        # On: the certificate also flags it, and it takes the float64 QR.
        assert ("linalg_qr" in ops.names) is recovery
        assert source_recovery_elements() > 0
        assert torch.isfinite(score).all()
        score.sum().backward()
        assert all(torch.isfinite(x.grad).all() for x in working)
        reference = _score(*(x.double() for x in case))
        tolerance = 1e-4 if recovery else 5e-2
        torch.testing.assert_close(score.detach().double(), reference, rtol=tolerance, atol=tolerance)
    finally:
        set_projected_condition_recovery(previous)


def test_factor_certificate_runs_only_under_condition_recovery(factor_route, device: torch.device) -> None:
    from vfe3.relational_scoring import (
        projected_condition_recovery, reset_source_recovery_elements, set_projected_condition_recovery,
        source_recovery_elements,
    )

    factor_route("cholesky_qr2")
    previous = projected_condition_recovery()
    try:
        # kappa(X) = 3e3 factors by CholeskyQR2 but exceeds the certificate eps kappa_1 > 3e-4.
        for kappa, enabled, fires in ((30., True, False), (3e3, False, False), (3e3, True, True)):
            set_projected_condition_recovery(enabled)
            reset_source_recovery_elements()
            _score(*_projected_case(device, kappa=kappa))
            assert (source_recovery_elements() > 0) is fires, (kappa, enabled)
    finally:
        set_projected_condition_recovery(previous)
