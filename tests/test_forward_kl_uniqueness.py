r"""Forward-KL uniqueness theorem as an executable property test.

GL(K)_supplementary.tex Sec "Conditional Uniqueness of the Forward KL Divergence via Variational
Duality" (theorem at :1203, eq:geometric_mean_target_supp :1214) proves that the stationary belief of
the inter-agent free energy

    F_i = D_KL(q_i || p_i) + sum_j beta_ij D(q_i || Omega_ij q_j)

assumes the geometric-mean Boltzmann (product-of-experts) form

    q_i* propto p_i^{1/2} prod_j (Omega_ij q_j)^{beta_ij/2}          (eq:geometric_mean_target_supp)

for all priors and neighbor beliefs IF AND ONLY IF D is the forward KL divergence (:1218). The
companion corollary (GL(K)_attention.tex:1096) notes this geometric-mean target is specific to the
alpha=1 / KL case and becomes transcendental for other divergences.

For a diagonal Gaussian the geometric-mean target is the precision-weighted product of experts:

    P_i*   = (1/2) ( 1/sigma_p + sum_j beta_ij / sigma_t,ij ),        P = 1/sigma  (precision)
    mu_i*  = ( mu_p/sigma_p + sum_j beta_ij mu_t,ij/sigma_t,ij )
             / ( 1/sigma_p   + sum_j beta_ij / sigma_t,ij ),

with (mu_t,ij, sigma_t,ij) = Omega_ij q_j the transported neighbor (the self-pair j=i contributes
zero to the gradient since Omega_ii = I, and is reproduced automatically by the full j-sum at the
fixed point). This is exactly the zero-gradient fixed point of the diagonal-KL belief kernel, so
running the E-step to convergence at renyi_order=1 must land on this closed form; at renyi_order != 1
the (Renyi-coupled) fixed point does not. The test pins both directions of the theorem in code.
"""

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.free_energy import attention_weights, pairwise_energy
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)
from vfe3.inference.e_step import e_step

_TAU = 1.0
_N_ITER = 4000
_LR = 0.3


def _setup(seed=0, N=3, K=2):
    # Distinct prior means/variances and a non-trivial (but fixed, e_phi_lr=0) gauge so the
    # transport genuinely mixes the agents and the geometric mean is non-degenerate.
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    n_gen = grp.generators.shape[0]
    belief = BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.25 * torch.randn(N, n_gen, generator=g),
    )
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return belief, mu_p, sigma_p, grp


def _geometric_mean_target(belief, mu_p, sigma_p, grp, tau, renyi_order):
    # The eq:geometric_mean_target_supp closed form for a diagonal Gaussian, evaluated at the
    # converged belief (the j=i self term uses Omega_ii q_i = q_i, reproduced by the full j-sum).
    omega = compute_transport_operators(belief.phi.unsqueeze(0), grp)["Omega"][0]   # (N, N, K, K)
    mu_t = transport_mean(omega, belief.mu)                                          # (N, N, K)
    sigma_t = transport_covariance(omega, belief.sigma)                             # (N, N, K)
    energy = pairwise_energy(DiagonalGaussian(belief.mu, belief.sigma),
                             DiagonalGaussian(mu_t, sigma_t), alpha=renyi_order)     # (N, N)
    beta = attention_weights(energy, tau=tau)                                        # (N, N)
    Pp = 1.0 / sigma_p                                                               # (N, K)
    Pt = 1.0 / sigma_t                                                               # (N, N, K)
    sum_beta_Pt    = torch.einsum("ij,ijk->ik", beta, Pt)                            # (N, K)
    sum_beta_Pt_mu = torch.einsum("ij,ijk->ik", beta, Pt * mu_t)                     # (N, K)
    sigma_star = 1.0 / (0.5 * (Pp + sum_beta_Pt))
    mu_star = (Pp * mu_p + sum_beta_Pt_mu) / (Pp + sum_beta_Pt)
    return mu_star, sigma_star


def test_forward_kl_converges_to_geometric_mean_fixed_point():
    # renyi_order=1 (forward KL): the converged belief equals the geometric-mean Boltzmann target
    # of eq:geometric_mean_target_supp to convergence tolerance.
    belief, mu_p, sigma_p, grp = _setup()
    out = e_step(belief, mu_p, sigma_p, grp, tau=_TAU, n_iter=_N_ITER,
                 e_q_mu_lr=_LR, e_q_sigma_lr=_LR, e_phi_lr=0.0, renyi_order=1.0)
    mu_star, sigma_star = _geometric_mean_target(out, mu_p, sigma_p, grp, _TAU, 1.0)
    assert torch.allclose(out.mu, mu_star, atol=1e-4)
    assert torch.allclose(out.sigma, sigma_star, atol=1e-4)


def test_non_kl_divergence_breaks_the_geometric_mean_form():
    # renyi_order=0.5 (a non-KL convex f-divergence): the converged belief does NOT take the
    # geometric-mean form -- the closed form is forward-KL-specific (the theorem's "only if").
    belief, mu_p, sigma_p, grp = _setup()
    out = e_step(belief, mu_p, sigma_p, grp, tau=_TAU, n_iter=_N_ITER,
                 e_q_mu_lr=_LR, e_q_sigma_lr=_LR, e_phi_lr=0.0, renyi_order=0.5)
    mu_star, sigma_star = _geometric_mean_target(out, mu_p, sigma_p, grp, _TAU, 0.5)
    dev = max((out.mu - mu_star).abs().max().item(), (out.sigma - sigma_star).abs().max().item())
    assert dev > 1e-2          # converged Renyi fixed point departs the geometric-mean target


def test_geometric_mean_gap_is_kl_specific():
    # The two directions side by side: the KL gap is orders of magnitude below the non-KL gap.
    belief, mu_p, sigma_p, grp = _setup(seed=1)
    kw = dict(tau=_TAU, n_iter=_N_ITER, e_q_mu_lr=_LR, e_q_sigma_lr=_LR, e_phi_lr=0.0)
    out_kl = e_step(belief, mu_p, sigma_p, grp, renyi_order=1.0, **kw)
    out_r = e_step(belief, mu_p, sigma_p, grp, renyi_order=0.5, **kw)
    m1, s1 = _geometric_mean_target(out_kl, mu_p, sigma_p, grp, _TAU, 1.0)
    m2, s2 = _geometric_mean_target(out_r, mu_p, sigma_p, grp, _TAU, 0.5)
    gap_kl = max((out_kl.mu - m1).abs().max().item(), (out_kl.sigma - s1).abs().max().item())
    gap_r  = max((out_r.mu - m2).abs().max().item(),  (out_r.sigma - s2).abs().max().item())
    assert gap_r > 100.0 * gap_kl
