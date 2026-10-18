import torch

from vfe3.attention_prior import attention_log_prior
from vfe3.free_energy import (
    attention_weights,
    free_energy,
    reduced_free_energy,
)

# A concrete non-uniform setup reused across tests.
_E   = torch.tensor([1.0, 2.0, 0.5])               # distinct per-key energies
_PI  = torch.tensor([0.5, 0.3, 0.2])               # normalized non-uniform prior
_B   = torch.log(_PI)                              # log-prior bias
_TAU = 2.0


def test_attention_tau_keys_off_energy_dimension():
    # The softmax temperature must match the dimension the energy accumulates over: the irrep BLOCK.
    # Single-block groups (glk/so_k/sp, irrep_dims=[K]) accumulate over the full K -> sqrt(K); per-head
    # multi-block (block_glk, irrep_dims=[d_head]*H) accumulates per head -> sqrt(d_head).
    from vfe3.free_energy import attention_tau
    assert abs(attention_tau(1.0, [64]) - 8.0) < 1e-6           # single block: sqrt(64)
    assert abs(attention_tau(2.0, [64]) - 16.0) < 1e-6          # kappa scales it
    assert abs(attention_tau(1.0, [8] * 8) - 8.0 ** 0.5) < 1e-6  # per-head: sqrt(d_head=8)


def test_beta_is_softmax_logprior_minus_energy_over_tau():
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    logits = _B - _E / _TAU
    expect = torch.softmax(logits, dim=-1)
    assert torch.allclose(beta, expect, atol=1e-6)
    assert torch.allclose(beta.sum(-1), torch.tensor(1.0), atol=1e-6)


def test_envelope_identity_canonical_block_equals_neg_tau_logZ():
    # Sum_j beta* E + tau Sum_j beta* log(beta*/pi) == -tau log Z, with non-uniform pi.
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    canon_block = (beta * _E).sum(-1) + _TAU * (beta * (torch.log(beta) - torch.log(pi))).sum(-1)
    fred = reduced_free_energy(_E, log_prior=_B, tau=_TAU)        # -tau log Z
    assert torch.allclose(canon_block, fred, atol=1e-5)
    # hand-computed literal backstop (catches a tau*log N offset):
    assert torch.allclose(fred, torch.tensor(1.1264), atol=1e-3)


def test_envelope_holds_for_raw_registry_priors_zero_causal_alibi():
    # The envelope must hold for the RAW seam output B (logsumexp(B) != 0), not only
    # a hand-normalized log(pi). A non-uniform-but-NORMALIZED prior cannot catch the
    # +tau*logsumexp(B) per-row offset (zero B -> offset tau*log N; causal_alibi -> tau*lse(B)).
    N, tau = 3, 2.0
    energy = torch.tensor([[1.0, 2.0, 0.5],
                           [0.7, 0.3, 1.1],
                           [1.2, 0.9, 0.4]])
    for name, B in [("zero", torch.zeros(N, N)), ("causal", attention_log_prior("causal", N, N)),
                    ("causal_alibi", attention_log_prior("causal_alibi", N, N, n_heads=1)[0])]:
        # B: (N, N) un-normalized log-bias
        beta = attention_weights(energy, log_prior=B, tau=tau)
        pi = torch.softmax(B, dim=-1)
        canon_block = (beta * energy).sum(-1) + tau * (
            beta * (torch.log(beta.clamp(min=1e-12)) - torch.log(pi.clamp(min=1e-12)))
        ).sum(-1)
        fred = reduced_free_energy(energy, log_prior=B, tau=tau)   # -tau log Z
        assert torch.allclose(canon_block, fred, atol=1e-5), name

    # None prior -> uniform 1/N; must match the zero-B result, not drift by tau*log N.
    B0 = torch.zeros(N, N)
    assert torch.allclose(reduced_free_energy(energy, log_prior=None, tau=tau),
                          reduced_free_energy(energy, log_prior=B0, tau=tau), atol=1e-5)


def test_stationarity_residual_constant_across_keys():
    # At beta*, E_j + tau log(beta*_j/pi_j) is the SAME for every key j (= -tau log Z).
    beta = attention_weights(_E, log_prior=_B, tau=_TAU)
    pi = torch.softmax(_B, dim=-1)
    residual = _E + _TAU * (torch.log(beta) - torch.log(pi))
    assert (residual.max() - residual.min()).abs() < 1e-5
    assert torch.allclose(residual.mean(), reduced_free_energy(_E, log_prior=_B, tau=_TAU), atol=1e-5)


def test_free_energy_entropy_exact_for_deep_finite_prior(
    device: torch.device,
) -> None:
    # m8: a finite prior entry below ~-27.6 nats (the old log-clamp floor) whose beta is non-negligible
    # made free_energy's entropy term deviate from the exact -tau logZ. With log_softmax the envelope
    # identity F == sum_i reduced_free_energy_i holds (self_div=0), and a -inf mask stays finite.
    from vfe3.free_energy import free_energy
    tau       = 1.0
    energy    = torch.tensor([[1.0, 1.0, -35.0]] * 3).to(device)     # entry 2 attractive enough that beta_2 ~ 1/3
    log_prior = torch.tensor([[0.0, 0.0, -35.0]] * 3).to(device)     # entry 2 finite but far below the -27.6 floor
    self_div  = torch.zeros(3).to(device)
    alpha     = torch.zeros(3).to(device)
    F = free_energy(self_div, energy, alpha, tau=tau, lambda_beta=1.0,
                    log_prior=log_prior)
    fred = reduced_free_energy(energy, tau=tau, log_prior=log_prior).sum()
    assert torch.isfinite(F)
    assert torch.allclose(F, fred, atol=1e-4)             # RED pre-fix: the clamp distorts the deep entry
    lp_masked = torch.tensor([[0.0, 0.0, float("-inf")]] * 3).to(device)   # -inf hard mask must not NaN (beta==0 there)
    Fm = free_energy(self_div, energy, alpha, tau=tau, lambda_beta=1.0,
                     log_prior=lp_masked)
    assert torch.isfinite(Fm)


from vfe3.free_energy import free_energy


def test_canonical_minus_surrogate_is_tau_times_entropy():
    # Canonical F - surrogate F = tau * Sum_i Sum_j beta* log(beta*/pi)  (the entropy block).
    N = 3
    self_div = torch.zeros(N)                            # alpha term zero (isolate beta block)
    energy = torch.tensor([[1.0, 2.0, 0.5],
                           [0.7, 0.3, 1.1],
                           [1.2, 0.9, 0.4]])
    B = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    log_prior = B.expand(N, N)
    alpha = torch.zeros(N)
    fe_canon = free_energy(self_div, energy, alpha, log_prior=log_prior, tau=2.0,
                           include_attention_entropy=True)
    fe_surr  = free_energy(self_div, energy, alpha, log_prior=log_prior, tau=2.0,
                           include_attention_entropy=False)
    beta = attention_weights(energy, log_prior=log_prior, tau=2.0)
    pi = torch.softmax(log_prior, dim=-1)
    entropy_block = 2.0 * (beta * (torch.log(beta) - torch.log(pi))).sum()
    assert torch.allclose(fe_canon - fe_surr, entropy_block, atol=1e-5)


def test_known_value_F_self_coupling_only():
    # q == p -> self_div == 0; energy all-equal + uniform prior -> beta uniform.
    # With alpha=2, self_div=[0.5,1.0], no entropy (surrogate), energy uniform=c:
    # F = sum_i alpha_i*self_div_i + sum_ij beta_ij*c. beta uniform=1/N so sum_j beta*c=c.
    self_div = torch.tensor([0.5, 1.0])
    energy = torch.full((2, 2), 0.3)
    alpha = torch.full((2,), 2.0)
    fe = free_energy(self_div, energy, alpha, log_prior=None, tau=1.0,
                     include_attention_entropy=False)
    expect = (2.0 * 0.5 + 2.0 * 1.0) + (0.3 + 0.3)
    assert torch.allclose(fe, torch.tensor(expect), atol=1e-5)


def test_autograd_F_matches_finite_difference():
    torch.manual_seed(0)
    N, K = 3, 4
    mu_q = torch.randn(N, K, requires_grad=True)
    base = {"sigma_q": torch.rand(N, K) + 0.5, "mu_p": torch.randn(N, K),
            "sigma_p": torch.rand(N, K) + 0.5}
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import self_divergence

    def scalar(mu):
        sd = self_divergence(DiagonalGaussian(mu, base["sigma_q"]),
                             DiagonalGaussian(base["mu_p"], base["sigma_p"]))
        energy = torch.cdist(mu, mu) ** 2 + 0.1           # a smooth differentiable (N,N) energy
        alpha = torch.ones(N)
        return free_energy(sd, energy, alpha, log_prior=None, tau=1.5,
                           include_attention_entropy=True)

    F = scalar(mu_q); F.backward()
    g_auto = mu_q.grad.clone()
    # F is fp32 throughout. The central-difference error is truncation O(h^2|F'''|)
    # plus roundoff O(eps_mach |F| / h); the sum is minimized near h ~ eps_mach^(1/3)
    # ~ 5e-3 for fp32. h=1e-3 is roundoff-dominated (eps_mach|F|/h ~ 1.2e-3, i.e. AT
    # the 1e-3 tolerance), so the step -- not the tolerance -- is the right knob.
    eps = 5e-3
    g_fd = torch.zeros_like(mu_q)
    with torch.no_grad():
        for a in range(N):
            for b in range(K):
                d = torch.zeros(N, K); d[a, b] = eps
                g_fd[a, b] = (scalar(mu_q + d) - scalar(mu_q - d)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, atol=1e-3, rtol=1e-3)


def test_gradient_gap_canonical_minus_surrogate_is_neg_cov_over_tau():
    # The envelope theorem: with beta* a LIVE function of x, autograd of the canonical
    # beta-block collapses to the envelope Sum_j beta* dE; the surrogate keeps the
    # dbeta term, and (surrogate - canonical) gradients = -tau^{-1} Cov_beta*(E, dE).
    torch.manual_seed(1)
    tau = 1.5
    x = torch.randn(4, requires_grad=True)               # a differentiable parameter
    A = torch.randn(3, 4)
    log_prior = torch.log(torch.tensor([0.5, 0.3, 0.2]))  # non-uniform

    def energy_of(x_):                                   # (3,) energies, differentiable in x
        return (A @ x_) ** 2 + 0.2

    def canonical_block(x_):
        E = energy_of(x_)
        beta = torch.softmax(log_prior - E / tau, dim=-1)
        pi = torch.softmax(log_prior, dim=-1)
        return (beta * E).sum() + tau * (beta * (torch.log(beta) - torch.log(pi))).sum()

    def surrogate_block(x_):
        E = energy_of(x_)
        beta = torch.softmax(log_prior - E / tau, dim=-1)
        return (beta * E).sum()

    gc = torch.autograd.grad(canonical_block(x), x)[0]
    gs = torch.autograd.grad(surrogate_block(x), x)[0]

    # envelope: canonical grad == Sum_j beta* dE_j  (beta* detached here on purpose)
    E = energy_of(x)
    beta = torch.softmax(log_prior - E / tau, dim=-1).detach()
    JE = torch.autograd.functional.jacobian(energy_of, x)     # (3,4) dE_j/dx
    env = (beta.unsqueeze(-1) * JE).sum(0)
    assert torch.allclose(gc, env, atol=1e-4)

    # gap == -tau^{-1} Cov_beta*(E, dE)
    Edet = E.detach()
    mean_E  = (beta * Edet).sum()
    mean_J  = (beta.unsqueeze(-1) * JE).sum(0)                 # (4,)
    cross   = (beta.unsqueeze(-1) * Edet.unsqueeze(-1) * JE).sum(0)   # (4,)
    cov = cross - mean_E * mean_J
    assert torch.allclose(gs - gc, -cov / tau, atol=1e-4)


def test_alpha_envelope_grad_q_F_equals_alpha_star_times_grad_q_D():
    # State-dependent alpha*: at alpha*, dF/dalpha = 0, so d/dq [alpha*(D)*D + R(alpha*(D))]
    # == alpha* * dD/dq (the explicit alpha-path vanishes). De-risks Phase 4.
    from vfe3.alpha_i import self_coupling_alpha
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import self_divergence

    b0, c0 = 0.5, 2.0
    mu_q = torch.randn(2, 3, requires_grad=True)
    sigma_q = torch.rand(2, 3) + 0.5
    mu_p = torch.randn(2, 3); sigma_p = torch.rand(2, 3) + 0.5

    def adaptive_self(mu):
        D = self_divergence(DiagonalGaussian(mu, sigma_q), DiagonalGaussian(mu_p, sigma_p))   # (2,)
        a, r = self_coupling_alpha(D, mode="state_dependent", b0=b0, c0=c0)
        return (a * D + r).sum()

    g_full = torch.autograd.grad(adaptive_self(mu_q), mu_q)[0]
    # envelope RHS: alpha*(D) detached, times dD/dq
    D = self_divergence(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_p, sigma_p))
    a_star = (c0 / (b0 + D)).detach()
    g_env = torch.autograd.grad((a_star * D).sum(), mu_q)[0]
    assert torch.allclose(g_full, g_env, atol=1e-5)


def test_self_divergence_for_alpha_routes_by_declared_reduction():
    # The routing seam returns a per-POSITION (N,) divergence for a per-position alpha form
    # and a per-COORDINATE (N,K) divergence for a per-coordinate one, driven only by the
    # form's declared per_coord flag. Below kl_max the per-coordinate divergence sums back
    # to the per-position one, and the per-coordinate alpha genuinely varies across k.
    from vfe3.alpha_i import self_coupling_alpha
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import self_divergence_for_alpha

    N, K = 3, 4
    torch.manual_seed(0)
    mu = torch.randn(N, K); sigma = torch.rand(N, K) + 0.5
    mu_p = torch.randn(N, K); sigma_p = torch.rand(N, K) + 0.5
    q, p = DiagonalGaussian(mu, sigma), DiagonalGaussian(mu_p, sigma_p)

    summed = self_divergence_for_alpha(q, p, lambda_alpha_mode="state_dependent")
    per = self_divergence_for_alpha(q, p, lambda_alpha_mode="state_dependent_per_coord")
    assert summed.shape == (N,)
    assert per.shape == (N, K)
    assert torch.allclose(per.sum(dim=-1), summed, atol=1e-5)    # unsaturated -> sum recovers it

    a, _ = self_coupling_alpha(per, mode="state_dependent_per_coord", b0=1.0, c0=1.0)
    assert a.shape == (N, K)
    assert (a.std(dim=-1) > 1e-6).all()                          # alpha actually varies across coords


def test_self_divergence_per_coord_requires_kl_for_full_and_a_registered_functional():
    # Full covariance is supported for KL only: ordered conditional KL contributions that sum to
    # the full KL before clipping (docs/2026-09-22.md); a non-KL Renyi order raises.
    import pytest
    from vfe3.families.gaussian import FullGaussian
    from vfe3.free_energy import self_divergence_per_coord

    N, K = 2, 3
    mu = torch.randn(N, K)
    mu_p = torch.randn(N, K)
    cov = torch.eye(K).expand(N, K, K).contiguous()
    q_full, p_full = FullGaussian(mu, cov), FullGaussian(mu_p, cov.clone())
    per_coord = self_divergence_per_coord(q_full, p_full)
    assert per_coord.shape == (N, K)
    assert torch.allclose(per_coord.sum(-1), q_full.renyi_closed_form(p_full), atol=1e-5)
    with pytest.raises(ValueError, match="KL"):
        self_divergence_per_coord(q_full, p_full, alpha=0.5)


# --- value-gauge decoupling: free_energy coupling_energy --------------------

def test_free_energy_coupling_energy_none_equals_passing_same_energy():
    # coupling_energy defaults to `energy`; passing it explicitly is byte-identical.
    torch.manual_seed(0)
    N = 4
    sd = torch.rand(N)
    energy = torch.rand(N, N)
    alpha = torch.ones(N)
    f_default = free_energy(sd, energy, alpha, tau=1.3)
    f_explicit = free_energy(sd, energy, alpha, tau=1.3, coupling_energy=energy)
    assert torch.allclose(f_default, f_explicit)


def test_free_energy_coupling_energy_decouples_beta_from_coupling_sum():
    # beta = softmax(-energy/tau) from the SCORE energy; the coupling sum uses coupling_energy.
    torch.manual_seed(1)
    N = 4
    sd = torch.zeros(N)
    alpha = torch.zeros(N)                      # kill the self term to isolate the coupling block
    energy = torch.rand(N, N)                   # drives beta
    coupling_energy = torch.rand(N, N)          # drives the coupling sum
    tau = 1.0
    f = free_energy(sd, energy, alpha, tau=tau, coupling_energy=coupling_energy,
                    include_attention_entropy=False)
    beta = torch.softmax(-energy / tau, dim=-1)
    expected = (beta * coupling_energy).sum()
    assert torch.allclose(f, expected)
    # And it genuinely differs from the coherent (coupling_energy=None) free energy.
    f_coherent = free_energy(sd, energy, alpha, tau=tau, include_attention_entropy=False)
    assert not torch.allclose(f, f_coherent)
