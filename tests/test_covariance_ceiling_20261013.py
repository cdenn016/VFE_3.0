r"""The prior-anchored Loewner ceiling of the moment-matched readouts (``predictive_covariance_ceiling``).

Pinned on tiny float64 CPU cases, for both readouts and the diagonal and full families:

* off by default: ``None`` returns the readout bit for bit;
* the band :math:`f\Sigma_{p,i} \preceq \Sigma_i^+ \preceq c\Sigma_{p,i}` with a ceiling that binds on some tokens and
  not on others, and entries below the ceiling unchanged;
* idempotence of the clamp;
* exact invariance under a common non-orthogonal reference change (the ceiling involves no frame);
* the Daleckii--Krein backward against central finite differences, including on the degenerate eigenvalues the
  clamp creates;
* config validation (``c`` must exceed the floor).
"""
import warnings

import pytest
import torch

from tests.test_audit_20261012_floor import (
    FAMILIES, FLOOR, READOUTS, _change_reference, _inputs, _operators, _readout, _reference_change, _relative)
from vfe3.config import VFE3Config
from vfe3.model import covariance_ceiling as cc

F64 = torch.float64


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _whitened_max(cov: torch.Tensor, prior: torch.Tensor, diagonal: bool) -> torch.Tensor:   # (1, N)
    r""":math:`\lambda_{\max}(\Sigma_p^{-1/2}\Sigma\Sigma_p^{-1/2})` per token."""
    if diagonal:
        return (cov / prior).amax(-1)
    L = torch.linalg.cholesky(prior)
    W = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, cov, upper=False).mT, upper=False)
    return torch.linalg.eigvalsh(W)[..., -1]


def _binding_ceiling(readout: str, family: str) -> float:
    r"""A ceiling between the smallest and largest per-token whitened maxima, so it binds on some tokens only."""
    U, mu, sigma, prior, log_prior = _inputs(family)
    _, cov = _readout(_operators(readout), U, mu, sigma, prior, log_prior)
    top = _whitened_max(cov, prior, family == "diagonal").flatten().sort().values
    return float(0.5 * (top[0] + top[-1]))


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_off_by_default_is_the_identity(readout: str, family: str) -> None:
    U, mu, sigma, prior, log_prior = _inputs(family)
    ops = _operators(readout)
    assert ops.predictive_covariance_ceiling is None
    cov = _readout(ops, U, mu, sigma, prior, log_prior)[1]
    assert torch.equal(ops.ceiling_term(cov, prior, diagonal=family == "diagonal"), cov)


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_band_holds_and_unclamped_entries_are_unchanged(readout: str, family: str) -> None:
    diagonal = family == "diagonal"
    U, mu, sigma, prior, log_prior = _inputs(family)
    ops = _operators(readout)
    _, free = _readout(ops, U, mu, sigma, prior, log_prior)
    c = _binding_ceiling(readout, family)
    ops.predictive_covariance_ceiling = c
    cc.reset_covariance_ceiling_elements()
    _, cov = _readout(ops, U, mu, sigma, prior, log_prior)
    top_free = _whitened_max(free, prior, diagonal)                              # (1, N)
    binds = top_free > c
    assert bool(binds.any()) and bool((~binds).any())
    assert cc.covariance_ceiling_elements() == int(binds.sum())
    top = _whitened_max(cov, prior, diagonal)
    assert float(top.max()) <= c * (1 + 1e-10)                                  # Sigma^+ <= c Sigma_p
    assert float(top[binds].min()) >= c * (1 - 1e-10)                           # clamped tokens touch the ceiling
    if diagonal:
        assert bool((cov >= FLOOR * prior * (1 - 1e-12)).all())
    else:
        excess = torch.linalg.eigvalsh(cov - FLOOR * prior)
        assert float(excess.min()) >= -1e-10 * float(excess.max())               # the floor survives the clamp
    assert torch.equal(cov[~binds], free[~binds])                                # below the ceiling: untouched
    again = ops.ceiling_term(cov, prior, diagonal=diagonal)
    assert _relative(again, cov) < 1e-12                                         # idempotent


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_common_reference_change_is_exact_with_the_ceiling(readout: str, family: str) -> None:
    U, mu, sigma, prior, log_prior = _inputs(family)
    ops = _operators(readout)
    ops.predictive_covariance_ceiling = _binding_ceiling(readout, family)
    _, cov = _readout(ops, U, mu, sigma, prior, log_prior)
    a = _reference_change()
    _change_reference(ops, a)
    _, cov_a = _readout(ops, U @ a, mu, sigma, prior, log_prior)
    assert _relative(cov_a, cov) < 1e-11


def test_spectral_ceiling_backward_matches_finite_differences() -> None:
    r"""Inputs with eigenvalues on both sides of ``c`` and a pair exactly tied at the ceiling after the clamp."""
    g = torch.Generator().manual_seed(4)
    r, c = 4, 2.0
    Q = torch.linalg.qr(torch.randn(3, r, r, generator=g, dtype=F64))[0]
    lam = torch.tensor([[0.5, 1.2, 3.0, 4.5], [0.7, 1.9, 2.1, 6.0], [0.3, 0.9, 1.5, 1.8]], dtype=F64)
    W = (Q * lam.unsqueeze(-2)) @ Q.mT
    W.requires_grad_(True)
    G = torch.randn(3, r, r, generator=g, dtype=F64)
    (cc._SpectralCeiling.apply(W, c) * G).sum().backward()
    analytic = W.grad
    h = 1e-6
    numeric = torch.zeros_like(W)
    for idx in torch.cartesian_prod(*(torch.arange(n) for n in W.shape)):
        E = torch.zeros_like(W)
        E[tuple(idx)] = 1.0
        E = 0.5 * (E + E.mT)                                                     # symmetric perturbation
        with torch.no_grad():
            up = (cc._SpectralCeiling.apply(W + h * E, c) * G).sum()
            dn = (cc._SpectralCeiling.apply(W - h * E, c) * G).sum()
        numeric[tuple(idx)] = (up - dn) / (2 * h)
    sym = lambda X: 0.5 * (X + X.mT)                                             # noqa: E731
    assert torch.isfinite(analytic).all()
    assert float((sym(analytic) - numeric).abs().max()) < 1e-7


def test_config_rejects_a_ceiling_at_or_below_the_floor() -> None:
    with pytest.raises(ValueError, match="predictive_covariance_ceiling"):
        VFE3Config(predictive_covariance_floor=1e-2, predictive_covariance_ceiling=1e-2)
    assert VFE3Config(predictive_covariance_ceiling=100.0).predictive_covariance_ceiling == 100.0


@pytest.mark.parametrize("diagonal", [True, False])
def test_the_anchor_is_a_constant_for_gradients(diagonal: bool) -> None:
    r"""A binding ceiling does not train Sigma_p (audit 2026-10-15, finding 1): the clamped entries' gradient no
    longer flows into the anchor, so it cannot pile onto the token's own prior-variance row."""
    torch.manual_seed(0)
    if diagonal:
        sigma = (torch.rand(3, 4, dtype=F64) * 10).requires_grad_(True)
        anchor = torch.ones(3, 4, dtype=F64, requires_grad=True)
    else:
        A = torch.randn(3, 4, 4, dtype=F64)
        sigma = (5.0 * (A @ A.mT) + torch.eye(4, dtype=F64)).requires_grad_(True)
        anchor = torch.eye(4, dtype=F64).expand(3, 4, 4).clone().requires_grad_(True)
    out = cc.covariance_ceiling(sigma, anchor, 2.0, diagonal=diagonal)
    assert not torch.allclose(out, sigma)                              # the ceiling binds
    out.sum().backward()
    assert anchor.grad is None
    assert sigma.grad is not None and bool(torch.isfinite(sigma.grad).all())
