r"""Numerical conventions of the prior-anchored Loewner ceiling (audit 2026-10-16, finding F43).

* The spectral clamp eigendecomposes the whitened covariances through :func:`vfe3.numerics.safe_eigh`, so a call that
  clamps more matrices of size :math:`\le 32` than the cuSOLVER batched eigensolver accepts in one launch completes.
  The ceiling is emulated on CPU: ``torch.linalg.eigh`` is wrapped to reject a batch above
  :data:`vfe3.numerics._EIG_MAX_BATCH`, and the chunking predicate is made device-blind, so the bare call fails and
  the chunked call returns the unchunked values.
* The anchor :math:`\Sigma_p` is factored by ``cholesky_ex`` with a raise-only verdict that names the contract (an SPD
  prior covariance) instead of the bare raising ``torch.linalg.cholesky``; inside
  :func:`vfe3.contracts.deferred_host_checks` the verdict waits for the scope's host read.

Tiny CPU cases only (2 x 2 blocks; the batch test holds 16 400 of them).
"""
import pytest
import torch

from vfe3 import numerics
from vfe3.contracts import deferred_host_checks
from vfe3.model import covariance_ceiling as cc

torch.set_num_threads(1)
F64 = torch.float64


def _spd_batch(n: int, r: int, seed: int, scale: float, ridge: float = 1.0) -> torch.Tensor:   # (n, r, r)
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, r, r, generator=g, dtype=F64)
    return scale * (A @ A.mT) + ridge * torch.eye(r, dtype=F64)


def test_the_spectral_clamp_chunks_past_the_batched_eigensolver_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    n, r, c = numerics._EIG_MAX_BATCH + 16, 2, 2.0
    sigma = _spd_batch(n, r, seed=0, scale=5.0, ridge=3.0)                   # every entry above c * I
    anchor = torch.eye(r, dtype=F64).expand(n, r, r)
    want = cc.covariance_ceiling(sigma, anchor, c, diagonal=False)          # unpatched reference
    eigh = torch.linalg.eigh

    def capped_eigh(matrix: torch.Tensor, *args, **kwargs):
        batch = matrix.numel() // (matrix.shape[-1] * matrix.shape[-2])
        if matrix.dim() >= 3 and matrix.shape[-1] <= 32 and batch > numerics._EIG_MAX_BATCH:
            raise RuntimeError("emulated CUSOLVER_STATUS_INVALID_VALUE: batched syevj past the batch ceiling")
        return eigh(matrix, *args, **kwargs)

    def device_blind(matrix: torch.Tensor, max_batch: int) -> bool:
        return (matrix.dim() >= 3 and matrix.shape[-1] <= 32
                and matrix.numel() // (matrix.shape[-1] * matrix.shape[-2]) > max_batch)

    monkeypatch.setattr(torch.linalg, "eigh", capped_eigh)
    monkeypatch.setattr(numerics, "_eig_needs_chunking", device_blind)
    cc.reset_covariance_ceiling_elements()
    out = cc.covariance_ceiling(sigma, anchor, c, diagonal=False)
    assert cc.covariance_ceiling_elements() == n                            # the clamp binds past one launch
    torch.testing.assert_close(out, want, rtol=0.0, atol=1e-12)
    assert float(torch.linalg.eigvalsh(c * anchor - out).min()) >= -1e-10   # Sigma^+ <= c Sigma_p


def test_a_non_positive_definite_anchor_raises_a_named_verdict() -> None:
    sigma = _spd_batch(3, 2, seed=1, scale=5.0)
    anchor = torch.diag(torch.tensor([1.0, -1.0], dtype=F64)).expand(3, 2, 2)   # not SPD: the bound has no whitening
    with pytest.raises(ValueError, match="anchor Sigma_p"):
        cc.covariance_ceiling(sigma, anchor, 2.0, diagonal=False)


def test_the_anchor_verdict_waits_for_the_deferred_host_read() -> None:
    sigma = _spd_batch(3, 2, seed=2, scale=5.0)
    anchor = torch.diag(torch.tensor([1.0, -1.0], dtype=F64)).expand(3, 2, 2)
    reached = False
    with pytest.raises(ValueError, match="anchor Sigma_p"):
        with deferred_host_checks():
            out = cc.covariance_ceiling(sigma, anchor, 2.0, diagonal=False)
            reached = True                                                  # queued, not raised in the call
            assert bool(torch.isfinite(out).all())                          # failed anchors factor the identity
    assert reached


def test_a_healthy_anchor_is_unchanged_by_the_guarded_factor() -> None:
    r"""On SPD anchors the ``cholesky_ex`` factor is the ``cholesky`` factor, so the clamp is the whitened
    spectral clamp :math:`L V\min(\Lambda, c)V^\top L^\top` computed directly."""
    n, r, c = 5, 3, 2.0
    sigma = _spd_batch(n, r, seed=3, scale=4.0)
    anchor = _spd_batch(n, r, seed=4, scale=0.5)
    out = cc.covariance_ceiling(sigma, anchor, c, diagonal=False)
    L = torch.linalg.cholesky(anchor)
    W = torch.linalg.solve_triangular(L, torch.linalg.solve_triangular(L, sigma, upper=False).mT, upper=False)
    lam, V = torch.linalg.eigh(0.5 * (W + W.mT))
    want = L @ ((V * lam.clamp(max=c).unsqueeze(-2)) @ V.mT) @ L.mT
    clamped = lam.amax(-1) > c
    torch.testing.assert_close(out[clamped], 0.5 * (want + want.mT)[clamped], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(out[~clamped], sigma[~clamped], rtol=0.0, atol=0.0)
