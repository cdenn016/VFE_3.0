r"""Audit 2026-10-16 F44: cancellation-free :math:`x - \log(1 + x)` in the KL kernels.

The full-Gaussian per-coordinate KL formed ``delta - log1p(delta)`` directly, which loses about
:math:`\varepsilon / \delta` relative accuracy near equality, and :func:`x_minus_log1p` switched to that
same cancelling form at :math:`|x| = 10^{-3}` whatever the dtype, where float32 still lost about
:math:`10^{-4}` relative accuracy in the value and the gradient. These tests pin the float32 and float64
accuracy of the helper (value and autograd gradient) and of the full-Gaussian coordinate terms.
"""
from typing import Tuple

import pytest
import torch

from vfe3.families.base import x_minus_log1p
from vfe3.families.gaussian import _full_gaussian_kl_coordinate_terms


def _reference(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Float64 value and derivative of :math:`x - \log(1 + x)` for :math:`|x| \le 0.3`.

    The value is the Taylor series :math:`\sum_{n \ge 2} (-1)^n x^n / n` summed to 60 terms (truncation
    below :math:`0.3^{59}`), so it carries no cancellation; the derivative is :math:`x / (1 + x)`.
    """
    x = x.double()
    value = torch.zeros_like(x)
    for n in range(61, 1, -1):                         # Horner from the highest term down
        value = value * x + (-1.0) ** n / n
    return value * x * x, x / (1.0 + x)


def _signed_magnitudes(lo: float, hi: float, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mag = torch.logspace(torch.log10(torch.tensor(lo)).item(), torch.log10(torch.tensor(hi)).item(), n,
                         dtype=torch.float64)
    sign = torch.where(torch.rand(n, generator=g, dtype=torch.float64) < 0.5, -1.0, 1.0)
    return mag * sign


@pytest.mark.parametrize("dtype, value_tol, grad_tol", [
    (torch.float32, 3e-6, 3e-6),      # was about 1.1e-4 (value) and 6e-5 (gradient) just above |x| = 1e-3
    (torch.float64, 1.5e-13, 1.5e-13),  # was about 2.2e-13 just above |x| = 1e-3
])
def test_x_minus_log1p_value_and_gradient_keep_relative_accuracy(
    dtype: torch.dtype, value_tol: float, grad_tol: float) -> None:
    x = _signed_magnitudes(1e-6, 0.3, 4001, seed=0).to(dtype)
    ref_value, ref_grad = _reference(x)                # reference at the dtype-rounded inputs
    x = x.clone().requires_grad_()
    value = x_minus_log1p(x)
    (grad,) = torch.autograd.grad(value.sum(), x)
    value_err = ((value.detach().double() - ref_value).abs() / ref_value).max()
    grad_err = ((grad.double() - ref_grad).abs() / ref_grad.abs()).max()
    assert float(value_err) < value_tol, float(value_err)
    assert float(grad_err) < grad_tol, float(grad_err)


def test_x_minus_log1p_stays_finite_inside_the_domain() -> None:
    assert torch.isinf(x_minus_log1p(torch.tensor([-1.0]))).all()
    x = torch.tensor([-0.999999, -0.5, 0.0, 5.0, 1e6], requires_grad=True)
    value = x_minus_log1p(x)
    assert torch.isfinite(value).all() and float(value.detach()[2]) == 0.0
    (grad,) = torch.autograd.grad(value.sum(), x)
    assert torch.isfinite(grad).all() and float(grad[2]) == 0.0


def test_full_gaussian_coordinate_terms_keep_relative_accuracy_near_equality() -> None:
    r"""Float32 :math:`D_k` of :math:`q = N(\mu, \mathrm{diag}(t^2))` against :math:`p = N(\mu, I)`.

    Here :math:`L_p = I` and :math:`L_q = \mathrm{diag}(t)` are exact in float32 (round-0 Cholesky adds no
    jitter, and :math:`\mathrm{fl}(\sqrt{\mathrm{fl}(t^2)}) = t`), so :math:`T_{kk} = t_k` and
    :math:`D_k = (\delta_k^2 + 2(\delta_k - \log(1 + \delta_k)))/2` with :math:`\delta_k = t_k - 1` exact.
    The direct form gave about 9e-3 relative error at :math:`|\delta| \sim 10^{-5}`.
    """
    n, K = 300, 6
    delta = _signed_magnitudes(1e-6, 1e-1, n * K, seed=1).reshape(n, K)
    t = (1.0 + delta).float()                                          # (n, K) float32 pivots
    sigma_q = torch.diag_embed(t * t)                                  # (n, K, K)
    sigma_p = torch.eye(K).expand(n, K, K).contiguous()                # (n, K, K)
    mu = torch.randn(n, K)
    assert torch.equal(torch.linalg.cholesky(sigma_q).diagonal(dim1=-2, dim2=-1), t)
    terms, ok, _ = _full_gaussian_kl_coordinate_terms(mu, sigma_q, mu, sigma_p, 1e-6)
    assert terms.dtype == torch.float32 and bool(ok.all())
    d = t.double() - 1.0                                               # exact in float64
    ref = 0.5 * (d.square() + 2.0 * _reference(d)[0])                  # (n, K)
    rel = ((terms.double() - ref).abs() / ref).max()
    assert float(rel) < 1e-5, float(rel)
