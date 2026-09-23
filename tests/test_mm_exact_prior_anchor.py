r"""mm_exact prior-anchor regression tests (2026-07-10 fix).

The model enters the E-step with the belief anchored to the prior (q0 == p EXACTLY:
``forward_beliefs`` passes ``vfe_stack(beliefs, beliefs.mu, beliefs.sigma)``), so the raw
self-divergence KL(q||p) == 0.0 for every token on the first inner iteration. The mm fusion's
self mask must therefore gate only the UPPER (kl_max) saturation: reusing the gradient
kernel's lower gate (raw_self > 0) as the prior PRECISION WEIGHT zeroed the prior anchor at
exactly the state the model always initializes in, snapping mu* to the unanchored
(self-excluded under *_noself priors) neighbor consensus, and severing the live mu_p value
path. The gradient kernel keeps BOTH gates (exact there: dD/dtheta = 0 at q == p, so the
lower gate never changes a gradient VALUE). See docs/2026-07-10-mm-exact-prior-anchor-fix.md.
"""

import dataclasses
import inspect

import torch

from vfe3.families import get_family
from vfe3.free_energy import attention_weights, pairwise_energy
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (compute_transport_operators, transport_covariance,
                                     transport_mean)
from vfe3.gradients import kernels as kernels_mod
from vfe3.gradients.kernels import belief_gradients, mm_exact_update


def _setup(N=5, K=3, seed=0, device=torch.device("cpu")):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    grp = dataclasses.replace(grp, generators=grp.generators.to(device))
    phi = (0.1 * torch.randn(1, N, grp.generators.shape[0], generator=g)).to(device)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu_p = (0.5 * torch.randn(N, K, generator=g)).to(device)
    sigma_p = torch.full((N, K), 3.0, device=device)
    log_prior = torch.full((N, N), float("-inf"), device=device)   # causal noself prior:
    for i in range(1, N):                                          # row i attends j < i,
        log_prior[i, :i] = 0.0                                     # row 0 keeps only (0, 0)
    log_prior[0, 0] = 0.0
    return mu_p, sigma_p, omega, log_prior


def _fusion_pieces(mu, sigma, omega, log_prior, tau=1.0, eps=1e-6, kl_max=100.0):
    """The frozen intermediates the fusion consumes: masked beta, transported moments."""
    fam = get_family("gaussian_diagonal")
    mu_t = transport_mean(omega, mu)
    sigma_t = transport_covariance(omega, sigma, diagonal_out=True)
    energy = pairwise_energy(fam(mu, sigma), fam(mu_t, sigma_t), alpha=1.0,
                             kl_max=kl_max, eps=eps)
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)
    w = beta * ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    st = sigma_t.clamp(min=eps)
    pair_prec = torch.einsum("ij,ijk->ik", w, 1.0 / st)
    pair_mean = torch.einsum("ij,ijk->ik", w, mu_t / st)
    pair_mass = w.sum(-1).unsqueeze(-1).expand_as(pair_prec)
    return pair_prec, pair_mean, pair_mass


def test_prior_anchored_at_exact_init(device):
    # q0 == p -> D == 0 -> the fusion must use the envelope weight a = c0/(b0+0) = 1, NOT 0:
    # mu* = (a mu_p/sp + sum_j w mu_t/st) / (a/sp + sum_j w/st), jointly with the prior.
    mu_p, sigma_p, omega, log_prior = _setup(device=device)
    mu_star, sigma_star = mm_exact_update(
        mu_p.clone(), sigma_p.clone(), mu_p, sigma_p, omega,
        tau=1.0, lambda_beta=1.0, lambda_alpha_mode="state_dependent", log_prior=log_prior)
    pair_prec, pair_mean, pair_mass = _fusion_pieces(mu_p, sigma_p, omega, log_prior)
    a = 1.0
    prec = a / sigma_p + pair_prec
    expect_mu = (a * mu_p / sigma_p + pair_mean) / prec
    expect_sigma = (a + pair_mass) / prec
    assert torch.allclose(mu_star, expect_mu, atol=1e-5)
    assert torch.allclose(sigma_star, expect_sigma, atol=1e-5)
    # regression pin: the buggy lower gate (a == 0) returned the UNANCHORED pure consensus
    consensus = pair_mean / pair_prec.clamp(min=1e-6)
    assert not torch.allclose(mu_star[1:], consensus[1:], atol=1e-3)


def test_continuous_at_exact_init(device):
    # the buggy mask made the update DISCONTINUOUS at q0 == p (mask flips 0 -> 1 under an
    # infinitesimal perturbation); the fixed update must be continuous there.
    mu_p, sigma_p, omega, log_prior = _setup(device=device)
    mu_p, sigma_p = mu_p.double(), sigma_p.double()
    omega, log_prior = omega.double(), log_prior.double()
    kw = dict(tau=1.0, lambda_beta=1.0, lambda_alpha_mode="state_dependent",
              log_prior=log_prior)
    mu_a, sigma_a = mm_exact_update(mu_p.clone(), sigma_p.clone(), mu_p, sigma_p, omega, **kw)
    g = torch.Generator().manual_seed(1)
    d = (1e-7 * torch.randn(mu_p.shape, generator=g)).to(mu_p)
    mu_b, sigma_b = mm_exact_update(mu_p + d, sigma_p.clone(), mu_p, sigma_p, omega, **kw)
    assert torch.allclose(mu_a, mu_b, atol=1e-5)
    assert torch.allclose(sigma_a, sigma_b, atol=1e-5)


def test_upper_gate_still_zeroes_saturated_self(device):
    # the kl_max clamp flattens the objective in a neighborhood of a saturated self term, so
    # the upper gate must STILL zero the prior weight there (a == 0 -> pure pair fusion).
    mu_p, sigma_p, omega, log_prior = _setup(device=device)
    kl_max = 30.0
    mu_q = mu_p + 10.0            # self KL ~ K * 0.5 * 100/3 >> kl_max; pair energies stay small
    mu_star, _ = mm_exact_update(
        mu_q, sigma_p.clone(), mu_p, sigma_p, omega,
        tau=1.0, lambda_beta=1.0, kl_max=kl_max, lambda_alpha_mode="state_dependent",
        log_prior=log_prior)
    pair_prec, pair_mean, _ = _fusion_pieces(mu_q, sigma_p, omega, log_prior, kl_max=kl_max)
    consensus = pair_mean / pair_prec.clamp(min=1e-6)
    assert torch.allclose(mu_star[1:], consensus[1:], atol=1e-5)   # rows with live pair mass


def test_lambda_beta_zero_returns_prior_on_and_off_init(device):
    # lambda_beta = 0: the exact coordinate minimizer of the pure self term is q* = p. Value-
    # identical to the pre-fix code both at init (the old degenerate guard kept q0 == p, the
    # same numbers) and off init (the old mask was 1 there); pins that the fix changed nothing
    # on this axis.
    mu_p, sigma_p, omega, log_prior = _setup(device=device)
    for mu_q in (mu_p.clone(),
                 mu_p + 0.3 * torch.randn(mu_p.shape,
                                          generator=torch.Generator().manual_seed(2)).to(mu_p)):
        mu_star, sigma_star = mm_exact_update(
            mu_q, sigma_p.clone(), mu_p, sigma_p, omega,
            tau=1.0, lambda_beta=0.0, lambda_alpha_mode="state_dependent",
            log_prior=log_prior)
        assert torch.allclose(mu_star, mu_p, atol=1e-5)
        assert torch.allclose(sigma_star, sigma_p, atol=1e-5)


def test_gradient_kernel_keeps_two_sided_mask():
    # the GRADIENT kernel's lower gate is exact (it mirrors safe_kl_clamp's zero boundary
    # derivative, and dD = 0 at q == p regardless); only the mm fusion drops it. Guard against
    # a well-meaning "same fix" being applied there.
    src_kernel = inspect.getsource(kernels_mod._diag_kl_filtering_kernel)
    assert "(raw_self > 0.0) & (raw_self < kl_max)" in src_kernel
    src_mm = inspect.getsource(kernels_mod.mm_exact_update)
    assert "(raw_self > 0.0)" not in src_mm


def test_gradient_kernel_zero_self_grad_when_saturated(device):
    # upper-saturation behavior of the gradient kernel unchanged by the fix: a self term past
    # kl_max contributes zero gradient (with lambda_beta = 0 the whole gradient is zero).
    mu_p, sigma_p, omega, log_prior = _setup(device=device)
    g_mu, g_sigma = belief_gradients(
        mu_p + 10.0, sigma_p.clone(), mu_p, sigma_p, omega,
        tau=1.0, lambda_beta=0.0, kl_max=30.0, lambda_alpha_mode="state_dependent",
        log_prior=log_prior)
    assert torch.allclose(g_mu, torch.zeros_like(g_mu))
    assert torch.allclose(g_sigma, torch.zeros_like(g_sigma))
