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
        _source_root, projected_moments, reset_source_recovery_elements, set_projected_condition_recovery,
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
        # The outright CholeskyQR2 failure is rebuilt by float32 shifted CholeskyQR3, with or without the
        # certificate (fp32 recovery); no float64 QR runs.
        assert "linalg_qr" not in ops.names
        assert source_recovery_elements() > 0
        assert torch.isfinite(score).all()
        score.sum().backward()
        assert all(torch.isfinite(x.grad).all() for x in working)
        reference = _score(*(x.double() for x in case))
        # Per pair, the first-order model of _factor_status: |dKL_ij| <= eps (kappa_q,i + kappa_k,j)
        # (tr P_k G_q + d^T P_k d + r), with kappa the float64 condition numbers of the source roots (head 0's
        # query roots reach 1.3e6, eps kappa = 0.15). Measured: at most 0.21 of the bound (8.3e-3 on head 0).
        mu_h, sigma_h, blocks, S_q, S_k = (x.double() for x in case)
        moments = [projected_moments(mu_h, sigma_h, blocks, S) for S in (S_q, S_k)]
        kappas = []
        for (_, G), S in zip(moments, (S_q, S_k)):
            root = _source_root(G, (sigma_h, blocks, S), torch.ones(G.shape[:-2], dtype=torch.bool))
            singular = torch.linalg.svdvals(root)
            kappas.append((singular[:, 0] / singular[:, -1]).reshape(G.shape[:-2]).transpose(-1, -2))  # (1, H, N)
        (aq, Gq), (ak, Gk) = moments
        Pk = torch.linalg.inv(Gk)
        trace = torch.einsum("znhxy,zmhyx->zhnm", Gq, Pk)
        delta = aq.unsqueeze(2) - ak.unsqueeze(1)                                 # (1, Nq, Nk, H, r)
        quadratic = torch.einsum("znmhx,zmhxy,znmhy->zhnm", delta, Pk, delta)
        bound = (torch.finfo(torch.float32).eps * (kappas[0].unsqueeze(-1) + kappas[1].unsqueeze(-2))
                 * (trace + quadratic + Gq.shape[-1]))
        error = (score.detach().double() - reference).abs()
        assert bool((error <= bound).all()), float((error / bound).max())
    finally:
        set_projected_condition_recovery(previous)


def test_factor_certificate_runs_only_under_condition_recovery(factor_route, device: torch.device) -> None:
    from vfe3.relational_scoring import (
        _factor_status, projected_condition_recovery, projected_moments, reset_source_recovery_elements,
        set_projected_condition_recovery, source_recovery_elements,
    )

    factor_route("cholesky_qr2")
    previous = projected_condition_recovery()
    try:
        # kappa(X) = 3e3 factors by CholeskyQR2 but exceeds the certificate eps kappa_1 > 3e-4.
        for kappa, enabled, fires in ((30., True, False), (3e3, False, False), (3e3, True, True)):
            set_projected_condition_recovery(enabled)
            mu_h, sigma_h, blocks, S_q, _ = _projected_case(device, kappa=kappa)
            _, Gq = projected_moments(mu_h, sigma_h, blocks, S_q)
            flagged = _factor_status(Gq, (sigma_h, blocks, S_q, mu_h))[2]
            assert bool(flagged.any()) is fires, (kappa, enabled)
            # Every certificate-flagged entry is rebuilt and counted; none fails outright here.
            reset_source_recovery_elements()
            _score(*_projected_case(device, kappa=kappa))
            assert source_recovery_elements() == int(flagged.sum()), (kappa, enabled)
    finally:
        set_projected_condition_recovery(previous)


def _mixed_case(device: torch.device, dtype: torch.dtype = torch.float32, kappa: float = 3e3, seed: int = 0,
                tokens: int = 5) -> list:
    r"""Diagonal-covariance projected inputs: head 0's query roots have kappa(X) ~ ``kappa``, head 1's ~ 3.

    Two heads of width 3 read the full 6-coordinate belief, rank r = 3; the key maps are Gaussian.
    """
    generator = torch.Generator().manual_seed(seed)
    H, d, K = 2, 3, 6
    mu = torch.randn(1, tokens, K, generator=generator, dtype=torch.float64)
    sigma = torch.exp(torch.randn(1, tokens, K, generator=generator, dtype=torch.float64) * .3)
    blocks = torch.linalg.matrix_exp(.3 * torch.randn(1, tokens, H, d, d, generator=generator,
                                                      dtype=torch.float64))
    S_k = torch.randn(H, d, K, generator=generator, dtype=torch.float64)
    S_q = torch.stack([_conditioned_root(kappa, d, K, seed=seed), _conditioned_root(3., d, K, seed=seed + 1)])
    mu_h, sigma_h = (x.unsqueeze(-2).expand(1, tokens, H, K) for x in (mu, sigma))
    return [x.to(device=device, dtype=dtype) for x in (mu_h, sigma_h, blocks, S_q, S_k)]


def _rebuilt_entries(case: list) -> tuple:
    r"""(query, key) entries (1, N, H) the certified factor route rebuilds: failed or certificate-flagged."""
    from vfe3.relational_scoring import _factor_status, projected_moments, set_projected_condition_recovery

    mu_h, sigma_h, blocks, S_q, S_k = case
    masks = []
    previous = set_projected_condition_recovery(True)
    try:
        for S in (S_q, S_k):
            _, G = projected_moments(mu_h, sigma_h, blocks, S)
            _, failed, screen, _ = _factor_status(G, (sigma_h, blocks, S, mu_h))
            masks.append(failed | screen)
    finally:
        set_projected_condition_recovery(previous)
    return tuple(masks)


def _root_kappas(case: list) -> list:
    r"""(query, key) float64 condition numbers (1, H, N) of the source roots X = S U^{-1} Sigma^{1/2}."""
    from vfe3.relational_scoring import _source_root, projected_moments

    mu_h, sigma_h, blocks, S_q, S_k = (x.detach().cpu().double() for x in case)
    kappas = []
    for S in (S_q, S_k):
        _, G = projected_moments(mu_h, sigma_h, blocks, S)
        root = _source_root(G, (sigma_h, blocks, S), torch.ones(G.shape[:-2], dtype=torch.bool))
        singular = torch.linalg.svdvals(root)
        kappas.append((singular[:, 0] / singular[:, -1]).reshape(G.shape[:-2]).transpose(-1, -2))
    return kappas


def _pair_bound(case: list) -> torch.Tensor:
    r"""(1, H, Nq, Nk) first-order fp32 bound eps (kappa_q,i + kappa_k,j)(tr P_k G_q + d^T P_k d + r).

    The model of :func:`vfe3.relational_scoring._factor_status`, with kappa the float64 condition numbers of the
    source roots X = S U^{-1} Sigma^{1/2}.
    """
    from vfe3.relational_scoring import projected_moments

    mu_h, sigma_h, blocks, S_q, S_k = (x.detach().cpu().double() for x in case)
    moments = [projected_moments(mu_h, sigma_h, blocks, S) for S in (S_q, S_k)]
    kappas = _root_kappas(case)
    (aq, Gq), (ak, Gk) = moments
    Pk = torch.linalg.inv(Gk)
    trace = torch.einsum("znhxy,zmhyx->zhnm", Gq, Pk)
    delta = aq.unsqueeze(2) - ak.unsqueeze(1)                                 # (1, Nq, Nk, H, r)
    quadratic = torch.einsum("znmhx,zmhxy,znmhy->zhnm", delta, Pk, delta)
    return (torch.finfo(torch.float32).eps * (kappas[0].unsqueeze(-1) + kappas[1].unsqueeze(-2))
            * (trace + quadratic + Gq.shape[-1]))


def _score_and_count(case: list, recovery: bool) -> tuple:
    from vfe3.relational_scoring import (
        reset_source_recovery_elements, set_projected_condition_recovery, source_recovery_elements,
    )

    previous = set_projected_condition_recovery(recovery)
    try:
        reset_source_recovery_elements()
        score = _score(*case)
        return score, source_recovery_elements()
    finally:
        set_projected_condition_recovery(previous)


def test_certified_factor_route_rebuilds_its_flagged_entries(factor_route, device: torch.device) -> None:
    r"""``cholesky_qr2_certified`` rebuilds every certificate-flagged CholeskyQR2 factor from its source root.

    Head 0's query roots have kappa(X) = 3e3: they factor by CholeskyQR2 but fail eps kappa_1(C) <= 3e-4. Each
    flagged entry is counted, the pairs touching it change, every pair stays inside the first-order fp32 bound of
    :func:`_pair_bound`, and the pairs touching no flagged entry are bitwise those of the uncertified route.

    The rebuilt factor (float32 shifted CholeskyQR3 of a float64-formed root) has the same O(eps kappa) whitened
    error as the CholeskyQR2 factor it replaces, so the error falls on aggregate rather than on every case:
    measured over these six seeds the flagged pairs' mean error fell in five and rose in one (1.3e-5 to 2.5e-5),
    with the summed means 1.50e-4 kept against 1.14e-4 rebuilt. The rebuilt pairs reach at most 0.025 of the bound;
    the largest ratio over all pairs, 0.44, is on a healthy well-conditioned pair whose bound is tight.
    """
    factor_route("cholesky_qr2")
    kept_means, rebuilt_means = [], []
    for seed in range(6):
        case = _mixed_case(device, seed=seed)
        reference = _score(*(x.double() for x in case)).cpu()
        rebuilt_q, rebuilt_k = (mask.cpu() for mask in _rebuilt_entries(case))
        touched = (rebuilt_q[0].T.unsqueeze(-1) | rebuilt_k[0].T.unsqueeze(-2)).unsqueeze(0)  # (1, H, Nq, Nk)
        assert bool(rebuilt_q[0, :, 0].any()) and not bool(rebuilt_q[0, :, 1].any()), seed
        kept, kept_count = _score_and_count(case, False)
        rebuilt, rebuilt_count = _score_and_count(case, True)
        assert kept_count == 0 and rebuilt_count == int(rebuilt_q.sum() + rebuilt_k.sum()), seed
        kept, rebuilt = kept.cpu(), rebuilt.cpu()
        assert rebuilt.dtype == torch.float32 and torch.isfinite(rebuilt).all()
        assert torch.equal(rebuilt[~touched], kept[~touched]), seed
        assert not torch.equal(rebuilt[touched], kept[touched]), seed
        error = (rebuilt.double() - reference).abs()
        bound = _pair_bound(case)
        assert bool((error <= bound).all()), (seed, float((error / bound).max()))
        kept_means.append(float((kept.double() - reference).abs()[touched].mean()))
        rebuilt_means.append(float(error[touched].mean()))
    assert sum(rebuilt_means) < sum(kept_means), (kept_means, rebuilt_means)


def test_certified_factor_route_is_inert_at_float64(factor_route, device: torch.device) -> None:
    r"""The certificate is evaluated only on float32 factors, so a float64 call rebuilds nothing and is bitwise
    the uncertified route."""
    factor_route("cholesky_qr2")
    case = _mixed_case(device, torch.float64)
    kept, kept_count = _score_and_count(case, False)
    rebuilt, rebuilt_count = _score_and_count(case, True)
    assert kept_count == rebuilt_count == 0
    assert rebuilt.dtype == torch.float64 and torch.equal(rebuilt, kept)


def test_certified_factor_route_gradients_match_float64_central_differences(factor_route) -> None:
    r"""Gradients through the rebuilt factors are finite and match float64 central differences.

    Query token 0 of head 0 is certificate-flagged (kappa(X) = 3.1e3). The fp32 gradient bound is that of
    tests/test_relational_source_factorization.py: dKL/dG is at most quadratic in P = G^{-1}, whose relative
    error under an O(eps kappa) factor is <= 2 eps kappa, and the float32 means add <= 2 eps kappa, so normwise
    |dg| <= 6 eps kappa |g| per input (measured at most 0.004 of the bound). The reference is the float64 score's
    central difference with step 1e-6 (truncation O(1e-12), rounding O(1e-10) relative), not its autograd.
    """
    from vfe3.relational_scoring import set_projected_condition_recovery

    factor_route("cholesky_qr2")
    case = _mixed_case(torch.device("cpu"), seed=0, tokens=2)
    rebuilt_q, rebuilt_k = _rebuilt_entries(case)
    assert bool(rebuilt_q.any())
    previous = set_projected_condition_recovery(True)
    try:
        working = [x.detach().clone().requires_grad_(True) for x in case]
        _score(*working).sum().backward()
        reference = [x.double() for x in case]
        step = 1e-6
        kappa = max(float(k.max()) for k in _root_kappas(case))
        bound = 6 * torch.finfo(torch.float32).eps * kappa
        for index, x in enumerate(working):
            assert torch.isfinite(x.grad).all(), index
            central = torch.zeros_like(reference[index])
            for flat in range(central.numel()):
                shifted = [y.clone() for y in reference]
                shifted[index].view(-1)[flat] += step
                plus = float(_score(*shifted).sum())
                shifted[index].view(-1)[flat] -= 2 * step
                minus = float(_score(*shifted).sum())
                central.view(-1)[flat] = (plus - minus) / (2 * step)
            error = float((x.grad.double() - central).norm())
            assert error <= bound * float(central.norm()), (index, error / float(central.norm()), bound)
    finally:
        set_projected_condition_recovery(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gram_routes_are_unchanged_by_the_factor_route_rebuild(factor_route, device: torch.device,
                                                               dtype: torch.dtype) -> None:
    r"""``gram`` rebuilds only failures (none here); ``gram_certified`` rebuilds the entries its Gram certificate
    flags at float32 and nothing at float64, with untouched pairs bitwise those of ``gram``."""
    from vfe3.relational_scoring import _cholesky_status, projected_moments, set_projected_condition_recovery

    factor_route("gram")
    case = _mixed_case(device, dtype)
    mu_h, sigma_h, blocks, S_q, S_k = case
    previous = set_projected_condition_recovery(True)
    try:
        flags = [_cholesky_status(projected_moments(mu_h, sigma_h, blocks, S)[1])
                 for S in (S_q, S_k)]
    finally:
        set_projected_condition_recovery(previous)
    rebuilt_q, rebuilt_k = ((status[1] | status[2]).cpu() for status in flags)
    gram, gram_count = _score_and_count(case, False)
    certified, certified_count = _score_and_count(case, True)
    assert gram_count == 0 and certified_count == int(rebuilt_q.sum() + rebuilt_k.sum())
    assert bool(rebuilt_q.any()) is (dtype == torch.float32)
    touched = (rebuilt_q[0].T.unsqueeze(-1) | rebuilt_k[0].T.unsqueeze(-2)).unsqueeze(0)
    gram, certified = gram.cpu(), certified.cpu()
    assert torch.equal(certified[~touched], gram[~touched])
    if dtype == torch.float32:
        reference = _score(*(x.double() for x in case)).cpu()
        # The Gram factor's O(eps kappa^2) error is what the rebuild removes.
        kept_error = float((gram.double() - reference).abs()[touched].max())
        rebuilt_error = (certified.double() - reference).abs()
        assert float(rebuilt_error[touched].max()) < kept_error / 10
        assert bool((rebuilt_error <= _pair_bound(case)).all())
