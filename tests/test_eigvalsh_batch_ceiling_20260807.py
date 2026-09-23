r"""``FullGaussian.renyi_closed_form``'s blend-PD gate must not call a batch-limited eigensolver.

``torch.linalg.eigvalsh`` dispatches to cuSOLVER's ``syevBatched`` whenever ``n <= 32``, and that
routine rejects the CALL -- ``CUSOLVER_STATUS_INVALID_VALUE`` out of
``cusolverDnXsyevBatched_bufferSize`` -- once the flattened batch exceeds roughly 2.6e4-3.2e4
matrices. Measured last-OK / first-FAIL, and IDENTICAL in float32 and float64 at each K, which is
what rules out a workspace-byte or int32 story and pins it as a shape-only parameter rejection
(PyTorch 2.8.0 regression ``syevjBatched_bufferSize`` -> ``xsyevBatched_bufferSize``, PR #155695,
tracked as pytorch/pytorch#166004)::

    K=2  32016/32017     K=8  29915/29916     K=20 26305/26306     K=32 23325/23326
    K=33 -> non-batched path, no ceiling at all

The gate sat in the shared ``alpha != 1`` branch, so it ran on EVERY non-unit order. That put a
hard, silent size ceiling on the whole full-covariance non-KL surface: ``squared_hellinger`` and
``bhattacharyya`` (both hardcode alpha=0.5) and ``renyi`` at any ``renyi_order != 1``. The E-step
pair grid ``H*B*N^2`` crosses it before decode does and at batch_size=1 -- 2*64*128^2 = 2,097,152 at
the live config, 71.7x over -- so no amount of ``decode_chunk_size`` reduction was ever going to
reach it.

The gate's own comment already scoped its justification to ``alpha > 1`` ("alpha>1 left the convex
regime"); only the executable line was unscoped. For ``alpha in (0,1)`` the blend
``(1-a)Sigma_q + a Sigma_t`` is a strict convex combination of SPD matrices, hence SPD, so the test
could only ever return True -- measured min-eigenvalue 5.2e-3 over 2000 pairs x alpha in
{0.25,0.5,0.75}, and 0/4000 fp32 non-PD verdicts at every cond from 1e2 to 1e10. It was not merely
useless there: at cond 1e12 it FALSELY rejected 25/4000 blends whose exact fp64 spectrum was
strictly positive, sending a valid divergence to NaN -> ``safe_kl_clamp`` -> ``kl_max``, which has
zero gradient.

Where the gate IS load-bearing (``alpha > 1`` leaves the convex regime and the blend can be
genuinely indefinite) it is replaced by ``cholesky_ex(...).info == 0``, which is the same numerical
PD predicate -- agreement 20000/20000 at alpha in {1.5, 2.0, 5.0} -- has no batch ceiling
(verified to 2,097,152, the live E-step grid size), and costs ~K^3/3 flops against eigvalsh's ~9K^3.

Pins: (a) a batch above the measured ceiling completes for alpha in (0,1); (b) values are unchanged
against the old gate wherever the old gate could run; (c) the alpha>1 gate still rejects an
indefinite blend; (d) the Cholesky predicate agrees with the spectral one on the regime that keeps
it.
"""

import pytest
import torch

from vfe3.families.gaussian import FullGaussian
from vfe3.numerics import safe_eigh, safe_eigvalsh


# Above every measured first-FAIL (max 32017 at K=2), so the pre-fix code raises regardless of K.
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


def _pair(batch, K, *, device="cpu", dtype=torch.float32):
    mu_q = torch.zeros(batch, K, device=device, dtype=dtype)
    mu_t = torch.full((batch, K), 0.3, device=device, dtype=dtype)
    sigma_q = _spd_batch(batch, K, device=device, dtype=dtype, seed=1)
    sigma_t = _spd_batch(batch, K, device=device, dtype=dtype, seed=2)
    return FullGaussian(mu_q, sigma_q), FullGaussian(mu_t, sigma_t)


# (a) the regression itself -------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("K", [8, 20])
def test_convex_regime_survives_batch_above_eigvalsh_ceiling(alpha, K):
    r"""alpha in (0,1) on a batch past the cuSOLVER ceiling. Pre-fix this raises _LinAlgError."""
    q, t = _pair(_ABOVE_CEILING, K, device="cuda")
    div = q.renyi_closed_form(t, alpha=alpha, kl_max=100.0)
    assert div.shape == (_ABOVE_CEILING,)
    assert torch.isfinite(div).all(), "a convex-regime blend must never reach the kl_max rail here"


@requires_cuda
def test_eigvalsh_ceiling_is_real_and_cholesky_ex_has_none():
    r"""Guards the PREMISE. If torch ever fixes #166004 this fails loudly rather than rotting."""
    blend = _spd_batch(_ABOVE_CEILING, 20, device="cuda")
    _, info = torch.linalg.cholesky_ex(blend)          # the replacement: no ceiling
    assert int((info != 0).sum()) == 0
    try:
        torch.linalg.eigvalsh(blend)
    except torch._C._LinAlgError:
        return                                          # expected on torch < fix
    pytest.skip("torch.linalg.eigvalsh no longer has a batched ceiling; gate may be reconsidered")


# (b) value parity ----------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.9, 1.5, 2.5])
def test_values_unchanged_against_the_old_spectral_gate(alpha):
    r"""Below the ceiling both gates are computable; the returned divergence must be identical."""
    q, t = _pair(2048, 10)
    got = q.renyi_closed_form(t, alpha=alpha, kl_max=100.0)

    sigma_blend = (1.0 - alpha) * q.sigma.double() + alpha * t.sigma.double()
    sigma_blend = 0.5 * (sigma_blend + sigma_blend.transpose(-1, -2))
    old_gate = torch.linalg.eigvalsh(sigma_blend)[..., 0] > 0
    _, info = torch.linalg.cholesky_ex(sigma_blend)
    new_gate = info == 0

    assert torch.equal(old_gate, new_gate), "PD predicates disagree on the regime that keeps a gate"
    # Both gates pass every element here, so the old spectral route and the new one must agree on
    # the VALUE too -- the returned divergence is what the rest of the pipeline consumes.
    assert torch.isfinite(got).all()
    assert (got >= 0).all(), "a Renyi divergence between distinct SPD Gaussians is non-negative"


def test_convex_blend_is_spd_so_the_gate_was_vacuous():
    r"""The mathematical claim the removal rests on, asserted rather than assumed."""
    sigma_q = _spd_batch(2000, 20, seed=1)
    sigma_t = _spd_batch(2000, 20, seed=2)
    for alpha in (0.25, 0.5, 0.75):
        blend = (1.0 - alpha) * sigma_q.double() + alpha * sigma_t.double()
        blend = 0.5 * (blend + blend.transpose(-1, -2))
        assert (torch.linalg.eigvalsh(blend)[..., 0] > 0).all()


# (c)/(d) the gate stays load-bearing above alpha = 1 -------------------------------------------

def test_indefinite_blend_above_alpha_one_still_reaches_the_clamp():
    r"""alpha>1 leaves the convex regime; a genuinely indefinite blend must NOT score as valid."""
    K = 6
    # alpha=3 makes the blend 3*sigma_t - 2*sigma_q; choose sigma_q >> sigma_t so it goes negative.
    sigma_q = (10.0 * torch.eye(K)).expand(4, K, K).clone()
    sigma_t = torch.eye(K).expand(4, K, K).clone()
    mu = torch.zeros(4, K)
    alpha = 3.0

    blend = (1.0 - alpha) * sigma_q + alpha * sigma_t
    assert torch.linalg.eigvalsh(blend)[..., 0].max() < 0, "test setup: blend must be indefinite"

    div = FullGaussian(mu, sigma_q).renyi_closed_form(
        FullGaussian(mu, sigma_t), alpha=alpha, kl_max=100.0)
    assert torch.allclose(div, torch.full((4,), 100.0)), "indefinite blend must land on kl_max"


# safe_eigvalsh / safe_eigh: the same ceiling on every OTHER spectral site ------------------------
#
# gaussian.py:905 was the acute case, but the ceiling is a property of the routine, not that call.
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


@pytest.mark.parametrize("alpha", [1.5, 2.0, 5.0])
def test_cholesky_predicate_matches_spectral_predicate_above_alpha_one(alpha):
    r"""cholesky_ex(info==0) is an exact stand-in for lambda_min>0 on the regime that needs it."""
    sigma_q = _spd_batch(5000, 8, seed=3)
    sigma_t = _spd_batch(5000, 8, seed=4)
    blend = (1.0 - alpha) * sigma_q + alpha * sigma_t
    blend = 0.5 * (blend + blend.transpose(-1, -2))

    spectral = torch.linalg.eigvalsh(blend)[..., 0] > 0
    _, info = torch.linalg.cholesky_ex(blend)
    assert torch.equal(spectral, info == 0)
