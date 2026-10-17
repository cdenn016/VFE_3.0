r"""Audit 2026-10-16 F42: the shift of shifted CholeskyQR3 (``cholesky_qr2(shift=True)``).

Fukaya et al. (SIAM J. Sci. Comput. 42 (2020) A477, eq. (67)) shift the first Gram by
:math:`s = 11(mn + n(n+1)) u \lVert X \rVert_2^2` with u the unit roundoff. The repository used
:math:`\varepsilon = 2u` and :math:`\lVert X \rVert_F^2`, up to 2r times larger. A larger shift leaves
:math:`\kappa(C_1^{-1} X)` above the inner CholeskyQR2's :math:`u^{-1/2}` limit sooner, so more float32
recoveries failed over to float64 QR (measured at r = 32, m = 160: 0.48 of flat spectra failed at
:math:`\kappa = 3 \times 10^4`, against none with the paper's shift). These tests pin the failure rates, the
accuracy of the successful factors, and the gradient.
"""
import torch

from vfe3.numerics import cholesky_qr2


def _roots(n: int, r: int, m: int, kappa: float, kind: str, seed: int) -> torch.Tensor:
    r"""(n, r, m) float64 roots :math:`X = U \mathrm{diag}(\sigma) V^\top` with :math:`\kappa_2(X) = \kappa`."""
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n, r, r, dtype=torch.float64, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, m, r, dtype=torch.float64, generator=g))
    if kind == "log":
        sv = torch.logspace(0.0, -torch.log10(torch.tensor(kappa)).item(), r, dtype=torch.float64)
    else:                                                          # flat, one small singular value
        sv = torch.ones(r, dtype=torch.float64)
        sv[-1] = 1.0 / kappa
    return (U * sv) @ V.transpose(-1, -2)


def _reference_factor(X: torch.Tensor) -> torch.Tensor:
    r"""Float64 lower Cholesky factor of :math:`X X^\top` from the QR of :math:`X^\top` (positive diagonal)."""
    C = torch.linalg.qr(X.transpose(-1, -2))[1].transpose(-1, -2)
    return C * torch.sign(torch.diagonal(C, dim1=-2, dim2=-1)).unsqueeze(-2)


def _failure_rate_and_error(kind: str, kappa: float) -> tuple:
    X64 = _roots(64, 32, 160, kappa, kind, seed=0)
    C, failed = cholesky_qr2(X64.float(), shift=True)
    C_ref = _reference_factor(X64)
    err = (C.double() - C_ref).norm(dim=(-2, -1)) / C_ref.norm(dim=(-2, -1))
    return float(failed.float().mean()), float(err[~failed].max()) if bool((~failed).any()) else 0.0


def test_float32_recovery_succeeds_on_a_flat_spectrum_inside_the_paper_range() -> None:
    rate, err = _failure_rate_and_error("flat", 3e4)
    assert rate <= 0.05, rate                         # was 0.48 with eps and ||X||_F^2
    assert err <= 1e-5, err


def test_float32_recovery_failure_rate_on_a_log_spectrum_at_kappa_1e5() -> None:
    rate, err = _failure_rate_and_error("log", 1e5)
    assert rate <= 0.25, rate                         # was 0.76 with eps and ||X||_F^2
    assert err <= 1e-5, err


def test_failures_are_reported_beyond_the_range() -> None:
    rate, _ = _failure_rate_and_error("log", 1e7)
    assert rate == 1.0


def test_shifted_factor_gradient_matches_finite_differences() -> None:
    X = _roots(2, 3, 5, 50.0, "log", seed=3).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: cholesky_qr2(x, shift=True)[0], (X,))
    C, failed = cholesky_qr2(X, shift=True)
    assert not bool(failed.any())
    assert torch.allclose(C, _reference_factor(X.detach()), rtol=1e-12, atol=1e-13)
