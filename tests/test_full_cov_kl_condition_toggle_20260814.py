r"""Full-covariance KL escalation gate, hardcoded 2026-10-12 to the former ``fp32_escalate_cond``.

A float32 pair KL runs both zero-jitter certificates and the pivot-ratio trigger; a pair that fails
either is recomputed and returned in float64, and a well-conditioned pair stays float32.
"""

import torch

import vfe3.families.gaussian as gaussian_mod
from vfe3.families.gaussian import FullGaussian


def _pair(*, ill_conditioned: bool = False):
    if ill_conditioned:
        diagonal = torch.tensor([1.0, 1.0e-3, 1.0e-5, 1.0e-6])
        sigma = torch.diag(diagonal).unsqueeze(0)
    else:
        sigma = torch.eye(4).unsqueeze(0)
    q = FullGaussian(torch.zeros(1, 4), sigma)
    t = FullGaussian(torch.full((1, 4), 0.1), sigma)
    return q, t


def test_float32_kl_runs_both_spectral_certificates(monkeypatch):
    calls = []
    original = gaussian_mod.validated_cholesky_solve

    def recording(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(gaussian_mod, "validated_cholesky_solve", recording)
    q, t = _pair()
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert torch.isfinite(result).all()
    assert calls == [(1, 4, 4), (1, 4, 4)]


def test_ill_conditioned_valid_pair_is_promoted_to_float64():
    q, t = _pair(ill_conditioned=True)
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert result.dtype is torch.float64


def test_well_conditioned_pair_stays_float32_and_matches_float64():
    q, t = _pair()
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert result.dtype is torch.float32
    reference = FullGaussian(q.mu.double(), q.sigma.double()).renyi_closed_form(
        FullGaussian(t.mu.double(), t.sigma.double()), alpha=1.0, kl_max=1.0e9)
    torch.testing.assert_close(result.double(), reference, atol=1e-7, rtol=1e-6)


def test_float64_family_skips_the_certificates(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a float64 family called validated_cholesky_solve")

    monkeypatch.setattr(gaussian_mod, "validated_cholesky_solve", forbidden)
    q, t = _pair()
    result = FullGaussian(q.mu.double(), q.sigma.double()).renyi_closed_form(
        FullGaussian(t.mu.double(), t.sigma.double()), alpha=1.0, kl_max=1.0e9)
    assert result.dtype is torch.float64
