import pytest
import torch

from vfe3.numerics import (
    bounded_variance_from_log,
    check_finite,
    condition_number,
    floor_eigenvalues,
    nan_inf_fraction,
    run_monitors,
    safe_cholesky,
    safe_spd_inverse,
)


@pytest.mark.parametrize(("diagonal_shift", "seed", "ridge"), [
    pytest.param(0.1, 1, None, id="default"),
    pytest.param(1.0, 0, 0.0,  id="zero"),
    pytest.param(1.0, 0, 1e-7, id="regularized"),
])
def test_safe_spd_inverse_matches_linalg_inverse(
    diagonal_shift: float,
    seed:           int,
    ridge:          float | None,
):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 4, 4, generator=g)
    eye = torch.eye(4)
    M = A @ A.transpose(-1, -2) + diagonal_shift * eye      # SPD, well-conditioned
    effective_ridge = 1e-6 if ridge is None else ridge
    out = safe_spd_inverse(M) if ridge is None else safe_spd_inverse(M, eps=ridge)
    expected = torch.linalg.inv(M + effective_ridge * eye)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)


def test_safe_spd_inverse_is_finite_on_singular():
    M = torch.zeros(4, 4)                                   # singular; pure Cholesky fails
    out = safe_spd_inverse(M)
    assert torch.isfinite(out).all()                       # jitter/pinv fallback keeps it finite


def test_floor_eigenvalues_clamps_spectrum():
    M = torch.diag(torch.tensor([5.0, 1e-9, -0.3]))        # one tiny, one negative
    out = floor_eigenvalues(M, floor=1e-3)
    evals = torch.linalg.eigvalsh(out)
    assert (evals >= 1e-3 - 1e-6).all()


def test_condition_number_known_values():
    assert torch.allclose(condition_number(torch.eye(4)), torch.tensor(1.0), atol=1e-5)
    M = torch.diag(torch.tensor([1.0, 100.0]))
    assert torch.allclose(condition_number(M), torch.tensor(100.0), atol=1e-3)


def test_condition_number_non_pd_returns_inf():
    # A symmetric matrix with a negative eigenvalue has no condition number: the monitor must surface
    # +inf, not a large positive value from clamping lambda_min up to eps (which would read as a
    # merely ill-conditioned SPD matrix). (audit 2026-06-17 id 39)
    torch.manual_seed(0)
    K = 3
    Q, _ = torch.linalg.qr(torch.randn(K, K))
    M = Q @ torch.diag(torch.tensor([-1.0, 1.0, 3.0])) @ Q.transpose(-1, -2)   # spectrum {-1,1,3}
    cond = condition_number(M)
    assert torch.isinf(cond) and cond > 0


def test_condition_number_diagonal_kind_disambiguates_square_table():
    table = torch.tensor([
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    diagonal = condition_number(table, kind="diagonal")
    full = condition_number(table, kind="full")

    assert torch.allclose(diagonal, torch.tensor([2.0, 2.0]))
    assert torch.allclose(full, torch.tensor(3.0), atol=1e-6)
    assert torch.equal(condition_number(table), full)                    # auto preserves square -> full


def test_condition_number_full_kind_requires_square_matrix():
    with pytest.raises(ValueError, match="square trailing dimensions"):
        condition_number(torch.ones(2, 3), kind="full")


def test_condition_number_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        condition_number(torch.ones(3), kind="spectrum")


def test_nan_inf_fraction_counts_nonfinite():
    t = torch.tensor([1.0, float("nan"), float("inf"), 2.0])
    assert abs(nan_inf_fraction(t) - 0.5) < 1e-6
    assert nan_inf_fraction(torch.ones(10)) == 0.0


def test_check_finite_warns_and_can_raise():
    bad = torch.tensor([1.0, float("nan")])
    assert check_finite(torch.ones(3)) is True
    with pytest.warns(RuntimeWarning):
        assert check_finite(bad) is False
    with pytest.raises(FloatingPointError):
        check_finite(bad, raise_on_nonfinite=True)


def test_safe_cholesky_factors_spd_byte_identical():
    r"""On SPD inputs safe_cholesky's round-0 (zero added jitter) factor is byte-identical
    to torch.linalg.cholesky, and the ok-mask is all-True."""
    g = torch.Generator().manual_seed(5)
    A = torch.randn(3, 4, 4, generator=g)
    M = A @ A.transpose(-1, -2) + torch.eye(4)              # SPD, well-conditioned
    L, ok = safe_cholesky(M)
    assert ok.all()
    assert torch.equal(L, torch.linalg.cholesky(M))         # byte-identical, not approx


def test_safe_cholesky_indefinite_marks_failed_no_raise():
    r"""An indefinite matrix yields ok=False for that element WITHOUT raising."""
    bad = torch.diag(torch.tensor([1.0, -1.0]))             # indefinite
    L, ok = safe_cholesky(bad)                              # must not raise
    assert ok.item() is False


def test_safe_cholesky_mixed_batch_isolates_failure():
    r"""In a mixed batch the failed (indefinite) element is masked while the good
    element keeps the exact factor it would get alone."""
    good = torch.eye(3)
    bad = torch.diag(torch.tensor([1.0, -2.0, 1.0]))
    M = torch.stack([good, bad])
    L, ok = safe_cholesky(M)                               # must not raise
    assert ok[0].item() is True
    assert ok[1].item() is False
    assert torch.equal(L[0], torch.linalg.cholesky(good))  # good element unperturbed


def test_run_monitors_record():
    rec = run_monitors(torch.tensor([1.0, 2.0, float("nan")]))
    assert set(rec) == {"nan_fraction", "abs_max"}
    assert abs(rec["nan_fraction"] - 1.0 / 3.0) < 1e-6
    assert rec["abs_max"] == 2.0
    # matrix probe on request
    M = torch.diag(torch.tensor([1.0, 9.0]))
    rec2 = run_monitors(M, ["condition_number"])
    assert abs(rec2["condition_number"] - 9.0) < 1e-3


def test_bounded_variance_overflow_check_is_version_cached(monkeypatch):
    """Audit 2026-07-12 N13: bounded_variance_from_log forced a device->host sync per call via
    bool((log_sigma > max_log).any()) -- its hot-path callers re-read the SAME parameter tables
    several times per forward. The check result is now cached on (identity, _version) with a
    weakref liveness guard (the lie_ops/killing-cache pattern): one sync per table mutation, not
    per call. Warning and value semantics are byte-identical -- a table above max_log still warns
    on EVERY call (from the cached host bool) and still exponentiates the clamped values."""
    import warnings as warnings_module

    from vfe3 import numerics as numerics_module

    calls = {"n": 0}
    real_check = numerics_module._max_log_exceeded

    def _counting_check(log_sigma, max_log):
        calls["n"] += 1
        return real_check(log_sigma, max_log)

    monkeypatch.setattr(numerics_module, "_max_log_exceeded", _counting_check)

    table = torch.nn.Parameter(torch.zeros(4, 3))
    for _ in range(5):
        numerics_module.bounded_variance_from_log(table)
    assert calls["n"] == 1, f"unchanged tensor re-synced {calls['n']} times"

    with torch.no_grad():
        table[0, 0] = 99.0                       # in-place write bumps _version -> one recheck
    with pytest.warns(RuntimeWarning, match="max_log"):
        out = numerics_module.bounded_variance_from_log(table)
    assert calls["n"] == 2
    torch.testing.assert_close(out[0, 0], torch.exp(torch.tensor(80.0)))   # clamped exp unchanged

    # unchanged again -> no new sync, but the warning still fires every call (cached host bool)
    with warnings_module.catch_warnings(record=True) as records:
        warnings_module.simplefilter("always")
        numerics_module.bounded_variance_from_log(table)
    assert calls["n"] == 2
    assert any(issubclass(r.category, RuntimeWarning) for r in records)


def test_bounded_variance_from_log_works_under_inference_mode():
    """Regression guard (2026-07-12 review of the N13 cache): inference tensors track NO
    _version counter, so the version-keyed cache must BYPASS (direct uncached check, the
    pre-cache behavior) rather than crash reading log_sigma._version."""
    with torch.inference_mode():
        out = bounded_variance_from_log(torch.zeros(3))
    torch.testing.assert_close(out, torch.ones(3))
