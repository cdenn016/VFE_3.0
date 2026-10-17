r"""``safe_eigvalsh`` / ``safe_eigh`` must not hit cuSOLVER's batched-eigensolver ceiling.

``torch.linalg.eigvalsh`` dispatches to cuSOLVER's ``syevBatched`` whenever ``n <= 32``, and that
routine rejects the CALL -- ``CUSOLVER_STATUS_INVALID_VALUE`` out of
``cusolverDnXsyevBatched_bufferSize`` -- once the flattened batch exceeds roughly 2.6e4-3.2e4
matrices. Measured last-OK / first-FAIL, and IDENTICAL in float32 and float64 at each K, which is
what rules out a workspace-byte or int32 story and pins it as a shape-only parameter rejection
(PyTorch 2.8.0 regression ``syevjBatched_bufferSize`` -> ``xsyevBatched_bufferSize``, PR #155695,
tracked as pytorch/pytorch#166004)::

    K=2  32016/32017     K=8  29915/29916     K=20 26305/26306     K=32 23325/23326
    K=33 -> non-batched path, no ceiling at all

The acute case was the alpha > 1 blend-PD gate of ``FullGaussian.renyi_closed_form``, which was
replaced by ``cholesky_ex`` and then removed with the alpha != 1 kernels (audit 2026-10-16 F93).
The wrappers pinned here keep every other spectral site below the ceiling.
"""

import pytest
import torch

from vfe3.numerics import safe_eigh, safe_eigvalsh


# Above every measured first-FAIL (max 32017 at K=2), so the raw eigvalsh raises regardless of K.
_ABOVE_CEILING = 40_000

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the ceiling is a cuSOLVER batched-eigensolver limit"
)


def _spd_batch(batch, K, *, device="cpu", dtype=torch.float32, seed=0, jitter=1e-2):
    r"""``(batch, K, K)`` SPD covariances via ``A A^T + jitter I``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(batch, K, K, generator=g, dtype=torch.float64)
    M = A @ A.transpose(-1, -2) + jitter * torch.eye(K, dtype=torch.float64)
    return M.to(device=device, dtype=dtype)


# safe_eigvalsh / safe_eigh: the same ceiling on every OTHER spectral site ------------------------
#
# The ceiling is a property of the routine, not of one call site.
# `BeliefParams.diagnostic_statistics` (base.py) and `covariance_spectrum` (metrics.py) run on the
# per-token belief covariance, flattened batch B*N -- under the ceiling at the live config, but over
# it from batch_size >= 206 at N=128, K=20. These wrappers make that unreachable.

def test_safe_eigvalsh_is_value_identical_below_the_cap():
    m = _spd_batch(64, 10).reshape(4, 16, 10, 10)
    assert torch.equal(safe_eigvalsh(m), torch.linalg.eigvalsh(m))


def test_safe_eigvalsh_preserves_shape_when_it_chunks():
    r"""Forced chunking via a tiny cap, on CPU, so the reshape/cat round-trip is checked anywhere."""
    m = _spd_batch(120, 6).reshape(2, 3, 20, 6, 6)
    got = safe_eigvalsh(m, max_batch=7)
    assert got.shape == (2, 3, 20, 6)
    assert torch.allclose(got, torch.linalg.eigvalsh(m))


def test_safe_eigh_preserves_shape_and_values_when_it_chunks():
    m = _spd_batch(120, 6).reshape(2, 3, 20, 6, 6)
    evals, evecs = safe_eigh(m, max_batch=7)
    ref_vals, _ = torch.linalg.eigh(m)
    assert evals.shape == (2, 3, 20, 6) and evecs.shape == (2, 3, 20, 6, 6)
    assert torch.allclose(evals, ref_vals)
    # Reconstruct rather than compare eigenvectors directly (sign/degeneracy are not canonical).
    rebuilt = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    assert torch.allclose(rebuilt, m, atol=1e-4)


def test_safe_eigvalsh_is_differentiable_across_the_chunk_seam():
    m = _spd_batch(40, 5, dtype=torch.float64).requires_grad_(True)
    safe_eigvalsh(m, max_batch=7).sum().backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()


@requires_cuda
def test_safe_eigvalsh_survives_a_batch_above_the_ceiling():
    m = _spd_batch(_ABOVE_CEILING, 20, device="cuda")
    with pytest.raises(torch._C._LinAlgError):
        torch.linalg.eigvalsh(m)                       # the raw op still fails
    got = safe_eigvalsh(m)                             # the wrapper does not
    assert got.shape == (_ABOVE_CEILING, 20)
    assert torch.isfinite(got).all() and (got > 0).all()
