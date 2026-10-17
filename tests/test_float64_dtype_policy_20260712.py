r"""Audit 2026-07-12 N11/N12/N4: one float64 dtype policy across the SPD/divergence kernels.

Policy (the Jul-11 F12 precedent, already carried by ``retract_spd_full``): float64 inputs COMPUTE
in float64 (preserving an fp64 island end to end); half-precision inputs still promote to float32.
Previously the diagonal SPD retraction, ``natural_gradient``, ``safe_spd_inverse``, and the full-covariance
Renyi/KL closed form all unconditionally downcast to float32 -- the retraction/nat-grad/inverse
kernels cast the fp32 result back to float64 while the divergence returned raw fp32; either way
only fp32 precision survived, silently collapsing any fp64 island built on top of them.

Every test here uses values whose fp64 result is unrepresentable at fp32 precision, so the
assertions fail against fp32 compute and pass only when the kernel genuinely computes in fp64.
"""

import math

import torch

from vfe3.families.base import get_family
from vfe3.geometry.retraction import natural_gradient, retract_spd_diagonal


def test_diagonal_spd_retraction_preserves_float64():
    """sigma * exp(delta/sigma) with a 1e-8-scale tangent: fp32 rounds exp(~1e-8) to 1.0 exactly
    (losing the whole step); fp64 keeps it."""
    sigma = torch.tensor([1.25, 2.5, 4.0], dtype=torch.float64)
    delta = torch.tensor([1e-8, -1e-8, 2e-8], dtype=torch.float64)
    out = retract_spd_diagonal(sigma, delta, trust_region=0.0, sigma_max=None)
    expected = sigma * torch.exp(delta / sigma)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-12)




def test_natural_gradient_preserves_float64_diagonal_and_full():
    """nat_mu = Sigma grad_mu, nat_sigma = 2 Sigma grad_sigma Sigma: with 1/3-style entries the
    fp32 products differ from fp64 at ~1e-8; the fp64 path must agree to 1e-14."""
    third = torch.tensor([1.0 / 3.0, 1.0 / 7.0], dtype=torch.float64)
    grad_mu = torch.tensor([1.0 / 9.0, 1.0 / 11.0], dtype=torch.float64)
    grad_sigma = torch.tensor([1.0 / 13.0, 1.0 / 17.0], dtype=torch.float64)

    nat_mu, nat_sigma = natural_gradient(grad_mu, grad_sigma, third)
    assert nat_mu.dtype == torch.float64 and nat_sigma.dtype == torch.float64
    torch.testing.assert_close(nat_mu, third * grad_mu, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(nat_sigma, 2.0 * third * third * grad_sigma,
                               rtol=0.0, atol=1e-14)

    sigma_full = torch.diag(third)
    grad_sigma_full = torch.diag(grad_sigma)
    nat_mu_f, nat_sigma_f = natural_gradient(grad_mu, grad_sigma_full, sigma_full)
    assert nat_mu_f.dtype == torch.float64 and nat_sigma_f.dtype == torch.float64
    torch.testing.assert_close(nat_mu_f, sigma_full @ grad_mu, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(nat_sigma_f, 2.0 * sigma_full @ grad_sigma_full @ sigma_full,
                               rtol=0.0, atol=1e-14)


def _diag_full_gaussians(dtype):
    """Two full-covariance Gaussians with DIAGONAL covariances, so the exact fp64 reference is the
    elementwise diagonal closed form computed in-test."""
    fam = get_family("gaussian_full")
    mu_q = torch.tensor([0.1, -0.2], dtype=dtype)
    mu_t = torch.tensor([0.25, 0.05], dtype=dtype)
    sq = torch.tensor([1.0 / 3.0, 1.0 / 7.0], dtype=dtype)
    st = torch.tensor([1.0 / 9.0, 3.0], dtype=dtype)
    q = fam(mu_q, torch.diag(sq))
    t = fam(mu_t, torch.diag(st))
    return q, t, mu_q, mu_t, sq, st


def test_full_cov_kl_closed_form_preserves_float64():
    """alpha=1 branch (Cholesky KL): fp32 compute carries ~1e-7 error against the analytic
    diagonal KL; the fp64 path must agree to 1e-12."""
    q, t, mu_q, mu_t, sq, st = _diag_full_gaussians(torch.float64)
    out = q.renyi_closed_form(t, alpha=1.0, kl_max=1e12)
    dmu = mu_t - mu_q
    expected = 0.5 * (sq / st + dmu * dmu / st - 1.0 + torch.log(st) - torch.log(sq)).sum()
    assert out.dtype == torch.float64
    torch.testing.assert_close(out.reshape(()), expected, rtol=0.0, atol=1e-12)


def test_full_cov_divergence_fp64_keyed_on_any_operand():
    """Review follow-up: the fp64 key must consider the COVARIANCES too (the sibling policies
    key on the metric/covariance tensor) -- an fp64 sigma with fp32 means still selects fp64."""
    fam = get_family("gaussian_full")
    q = fam(torch.tensor([0.1, -0.2], dtype=torch.float32),
            torch.diag(torch.tensor([1.0 / 3.0, 1.0 / 7.0], dtype=torch.float64)))
    t = fam(torch.tensor([0.25, 0.05], dtype=torch.float32),
            torch.diag(torch.tensor([1.0 / 9.0, 3.0], dtype=torch.float64)))
    out = q.renyi_closed_form(t, alpha=1.0, kl_max=1e12)
    assert out.dtype == torch.float64


def test_half_precision_still_promotes_to_float32():
    """The other half of the policy: fp16/bf16 inputs keep the existing fp32 promotion (the
    retraction casts back to the input dtype; the divergence returns its fp32 compute dtype)."""
    sigma16 = torch.tensor([1.25, 2.5], dtype=torch.bfloat16)
    delta16 = torch.tensor([0.1, -0.1], dtype=torch.bfloat16)
    out = retract_spd_diagonal(sigma16, delta16, trust_region=0.0, sigma_max=None)
    assert out.dtype == torch.bfloat16

    q, t, *_ = _diag_full_gaussians(torch.float32)
    out32 = q.renyi_closed_form(t, alpha=1.0, kl_max=1e12)
    assert out32.dtype == torch.float32
