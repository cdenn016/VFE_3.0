import torch

from vfe3.attention_prior import attention_log_prior
from vfe3.free_energy import (
    attention_weights,
    free_energy,
    log_partition,
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


def test_envelope_holds_for_raw_registry_priors_uniform_causal_alibi():
    # The envelope must hold for the RAW seam output B (logsumexp(B) != 0), not only
    # a hand-normalized log(pi). A non-uniform-but-NORMALIZED prior cannot catch the
    # +tau*logsumexp(B) per-row offset (uniform B=0 -> offset tau*log N; alibi -> tau*lse(B)).
    N, tau = 3, 2.0
    energy = torch.tensor([[1.0, 2.0, 0.5],
                           [0.7, 0.3, 1.1],
                           [1.2, 0.9, 0.4]])
    for name, kw in [("uniform", {}), ("causal", {}), ("alibi", {"slope": 0.5})]:
        B = attention_log_prior(name, N, N, **kw)            # un-normalized log-bias
        beta = attention_weights(energy, log_prior=B, tau=tau)
        pi = torch.softmax(B, dim=-1)
        canon_block = (beta * energy).sum(-1) + tau * (
            beta * (torch.log(beta.clamp(min=1e-12)) - torch.log(pi.clamp(min=1e-12)))
        ).sum(-1)
        fred = reduced_free_energy(energy, log_prior=B, tau=tau)   # -tau log Z
        assert torch.allclose(canon_block, fred, atol=1e-5), name

    # None prior -> uniform 1/N; must match the uniform-B result, not drift by tau*log N.
    B0 = attention_log_prior("uniform", N, N)
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
                    include_attention_entropy=True, log_prior=log_prior)
    fred = reduced_free_energy(energy, tau=tau, log_prior=log_prior).sum()
    assert torch.isfinite(F)
    assert torch.allclose(F, fred, atol=1e-4)             # RED pre-fix: the clamp distorts the deep entry
    lp_masked = torch.tensor([[0.0, 0.0, float("-inf")]] * 3).to(device)   # -inf hard mask must not NaN (beta==0 there)
    Fm = free_energy(self_div, energy, alpha, tau=tau, lambda_beta=1.0,
                     include_attention_entropy=True, log_prior=lp_masked)
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


def test_pairwise_energy_diagonal_and_full_match_hand_loop():
    # E_ij = D(q_i || Omega_ij q_j) for every (query i, key j) pair. The key axis is
    # inserted from the family's covariance structure, so it stays correct even when
    # sigma_q carries a leading batch dim that mu_q does not.
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import DiagonalGaussian, FullGaussian
    from vfe3.free_energy import pairwise_energy

    torch.manual_seed(7)
    N, K = 3, 4
    mu_q = torch.randn(N, K)
    mu_t = torch.randn(N, N, K)                              # Omega_ij mu_j, per (i,j)

    # diagonal family
    sigma_q = torch.rand(N, K) + 0.5
    sigma_t = torch.rand(N, N, K) + 0.5
    E = pairwise_energy(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_t, sigma_t))
    E_ref = torch.stack([torch.stack([
        renyi(DiagonalGaussian(mu_q[i], sigma_q[i]), DiagonalGaussian(mu_t[i, j], sigma_t[i, j]))
        for j in range(N)]) for i in range(N)])
    assert torch.allclose(E, E_ref, atol=1e-5)

    # diagonal sigma_q carrying a leading batch dim mu_q lacks (the misclassified case):
    sigma_q_b = sigma_q.unsqueeze(0)                         # (1, N, K), rank mu_q.dim()+1
    sigma_t_b = sigma_t.unsqueeze(0)                         # (1, N, N, K)
    E_b = pairwise_energy(DiagonalGaussian(mu_q, sigma_q_b), DiagonalGaussian(mu_t, sigma_t_b))
    assert torch.allclose(E_b[0], E_ref, atol=1e-5)

    # full-covariance family
    A = torch.randn(N, K, K)
    sig_q_full = A @ A.transpose(-1, -2) + K * torch.eye(K)
    Bf = torch.randn(N, N, K, K)
    sig_t_full = Bf @ Bf.transpose(-1, -2) + K * torch.eye(K)
    Ef = pairwise_energy(FullGaussian(mu_q, sig_q_full), FullGaussian(mu_t, sig_t_full))
    Ef_ref = torch.stack([torch.stack([
        renyi(FullGaussian(mu_q[i], sig_q_full[i]), FullGaussian(mu_t[i, j], sig_t_full[i, j]))
        for j in range(N)]) for i in range(N)])
    assert torch.allclose(Ef, Ef_ref, atol=1e-4)


def test_pairwise_energy_per_head_splits_by_irrep_block():
    # GL(K) finding #1: with irrep_dims the energy carries a per-head (per-irrep-block) axis
    # (...,H,N,N); head h is the divergence over block h's coordinates, and (diagonal KL being
    # additive over independent blocks) the heads sum to the full-K energy.
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import pairwise_energy

    torch.manual_seed(3)
    N, K = 3, 4
    mu_q = torch.randn(N, K); mu_t = torch.randn(N, N, K)
    sigma_q = torch.rand(N, K) + 0.5; sigma_t = torch.rand(N, N, K) + 0.5
    q = DiagonalGaussian(mu_q, sigma_q); key = DiagonalGaussian(mu_t, sigma_t)

    E = pairwise_energy(q, key, irrep_dims=[2, 2])
    assert E.shape == (2, N, N)
    assert not torch.allclose(E[0], E[1], atol=1e-3)    # heads are genuinely distinct, not a broadcast
    for h, (s, e) in enumerate([(0, 2), (2, 4)]):
        Eh_ref = torch.stack([torch.stack([
            renyi(DiagonalGaussian(mu_q[i, s:e], sigma_q[i, s:e]),
                  DiagonalGaussian(mu_t[i, j, s:e], sigma_t[i, j, s:e]))
            for j in range(N)]) for i in range(N)])
        assert torch.allclose(E[h], Eh_ref, atol=1e-5)

    E_full = pairwise_energy(q, key, irrep_dims=None)
    assert E_full.shape == (N, N)                       # None -> single full-K energy (backward compat)
    assert torch.allclose(E.sum(0), E_full, atol=1e-5)  # diagonal KL is additive over blocks
    # single-block irrep_dims reduces to the full-K energy (bit-identical to None)
    E_one = pairwise_energy(q, key, irrep_dims=[4])
    assert torch.allclose(E_one, E_full, atol=1e-6)


def _pairwise_energy_loop_reference(q, key, irrep_dims):
    # An EXPLICIT per-block-loop reference, recomputed in the test so the batched pairwise_energy
    # is gated against the loop arithmetic directly (not against itself).
    from vfe3.divergence import renyi
    q_b = q.broadcast_over_keys()
    energies, start = [], 0
    for d in irrep_dims:
        end = start + d
        energies.append(renyi(q_b.block(start, end), key.block(start, end)))
        start = end
    return torch.stack(energies, dim=-3)


def test_pairwise_energy_equal_blocks_batched_is_bit_identical_to_loop_diagonal():
    # Equal-size, more-than-one irrep blocks (the default block_glk case) take the batched path;
    # it is the SAME arithmetic in a different layout, so it equals the explicit per-block loop
    # EXACTLY (torch.equal, atol=0), not merely allclose.
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import _stackable_for_batching, pairwise_energy

    torch.manual_seed(7)
    N, K, irrep_dims = 4, 8, [2, 2, 2, 2]
    q = DiagonalGaussian(torch.randn(N, K), torch.rand(N, K) + 0.5)
    key = DiagonalGaussian(torch.randn(N, N, K), torch.rand(N, N, K) + 0.5)

    assert _stackable_for_batching(q.broadcast_over_keys(), key) is True   # the batched branch fires
    E = pairwise_energy(q, key, irrep_dims=irrep_dims)
    E_ref = _pairwise_energy_loop_reference(q, key, irrep_dims)
    assert E.shape == (len(irrep_dims), N, N)
    assert torch.equal(E, E_ref)


def test_pairwise_energy_equal_blocks_batched_is_bit_identical_to_loop_full():
    # Same bit-identity gate for the full-covariance family (matrix sigma sub-blocks).
    from vfe3.families.gaussian import FullGaussian
    from vfe3.free_energy import _stackable_for_batching, pairwise_energy

    torch.manual_seed(8)
    N, K, irrep_dims = 3, 6, [2, 2, 2]
    A = torch.randn(N, K, K); sig_q = A @ A.transpose(-1, -2) + K * torch.eye(K)
    Bf = torch.randn(N, N, K, K); sig_t = Bf @ Bf.transpose(-1, -2) + K * torch.eye(K)
    q = FullGaussian(torch.randn(N, K), sig_q)
    key = FullGaussian(torch.randn(N, N, K), sig_t)

    assert _stackable_for_batching(q.broadcast_over_keys(), key) is True   # the batched branch fires
    E = pairwise_energy(q, key, irrep_dims=irrep_dims)
    E_ref = _pairwise_energy_loop_reference(q, key, irrep_dims)
    assert E.shape == (len(irrep_dims), N, N)
    assert torch.equal(E, E_ref)


def test_pairwise_energy_equal_blocks_batched_with_leading_batch_dim():
    # The real training layout: mu (B,N,K), key (B,N,N,K). The loop stacks at dim=-3 -> (B,H,N,N);
    # the batched path must reproduce that exact layout (movedim of the leading head axis).
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import pairwise_energy

    torch.manual_seed(9)
    B, N, K, irrep_dims = 2, 3, 8, [2, 2, 2, 2]
    q = DiagonalGaussian(torch.randn(B, N, K), torch.rand(B, N, K) + 0.5)
    key = DiagonalGaussian(torch.randn(B, N, N, K), torch.rand(B, N, N, K) + 0.5)

    E = pairwise_energy(q, key, irrep_dims=irrep_dims)
    E_ref = _pairwise_energy_loop_reference(q, key, irrep_dims)
    assert E.shape == (B, len(irrep_dims), N, N)
    assert torch.equal(E, E_ref)


def test_pairwise_energy_unequal_blocks_fall_back_to_loop():
    # Unequal block sizes cannot be stacked into one functional call; the existing per-block loop
    # is the path, and the result is unchanged from it.
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import pairwise_energy

    torch.manual_seed(10)
    N, K, irrep_dims = 3, 5, [2, 3]
    q = DiagonalGaussian(torch.randn(N, K), torch.rand(N, K) + 0.5)
    key = DiagonalGaussian(torch.randn(N, N, K), torch.rand(N, N, K) + 0.5)

    E = pairwise_energy(q, key, irrep_dims=irrep_dims)
    E_ref = _pairwise_energy_loop_reference(q, key, irrep_dims)
    assert E.shape == (len(irrep_dims), N, N)
    assert torch.equal(E, E_ref)


def test_pairwise_energy_single_block_and_none_unchanged():
    # irrep_dims None or a single block returns the full-K energy (..., N, N), bit-identical to the
    # direct functional call -- the batched branch must not touch this path.
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import pairwise_energy

    torch.manual_seed(11)
    N, K = 3, 4
    q = DiagonalGaussian(torch.randn(N, K), torch.rand(N, K) + 0.5)
    key = DiagonalGaussian(torch.randn(N, N, K), torch.rand(N, N, K) + 0.5)

    E_direct = renyi(q.broadcast_over_keys(), key)
    assert torch.equal(pairwise_energy(q, key, irrep_dims=None), E_direct)
    assert torch.equal(pairwise_energy(q, key, irrep_dims=[K]), E_direct)


def test_pairwise_energy_equal_blocks_mismatched_sigma_rank_falls_back_to_loop():
    # Guard the BLOCKED concern: when sigma_q carries a leading batch dim mu_q lacks, stacking both
    # tensors along a new leading axis would right-align H against sigma's first batch dim and
    # broadcast spuriously. The batched branch must DECLINE this case and fall back to the loop, so
    # the result still equals the explicit per-block loop reference.
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import pairwise_energy

    torch.manual_seed(12)
    N, K, irrep_dims = 3, 8, [2, 2, 2, 2]
    mu_q = torch.randn(N, K)
    sigma_q = (torch.rand(N, K) + 0.5).unsqueeze(0)          # (1, N, K), rank mu_q.dim()+1
    mu_t = torch.randn(N, N, K)
    sigma_t = (torch.rand(N, N, K) + 0.5).unsqueeze(0)       # (1, N, N, K)
    q = DiagonalGaussian(mu_q, sigma_q)
    key = DiagonalGaussian(mu_t, sigma_t)

    E = pairwise_energy(q, key, irrep_dims=irrep_dims)
    E_ref = _pairwise_energy_loop_reference(q, key, irrep_dims)
    assert E.shape == (1, len(irrep_dims), N, N)
    assert torch.equal(E, E_ref)


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


def test_self_divergence_per_coord_requires_diagonal_renyi():
    # Per-coordinate self-divergence exists only for the diagonal family (full-cov KL does
    # not decompose coordinate-wise) and the Renyi functional (the only registered one).
    # Both restrictions raise rather than silently summing the wrong thing.
    import pytest
    from vfe3.families.gaussian import DiagonalGaussian, FullGaussian
    from vfe3.free_energy import self_divergence_per_coord

    N, K = 2, 3
    mu = torch.randn(N, K); sigma = torch.rand(N, K) + 0.5
    mu_p = torch.randn(N, K); sigma_p = torch.rand(N, K) + 0.5
    cov = torch.eye(K).expand(N, K, K).contiguous()
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        self_divergence_per_coord(FullGaussian(mu, cov), FullGaussian(mu_p, cov.clone()))
    with pytest.raises((ValueError, NotImplementedError, KeyError)):
        self_divergence_per_coord(DiagonalGaussian(mu, sigma), DiagonalGaussian(mu_p, sigma_p),
                                  divergence_family="not_a_functional")


def test_pairwise_energy_dispatches_on_declared_cov_kind_not_name():
    """A diagonal-structured family whose NAME lacks the 'diagonal' substring must still take the
    diagonal energy path. The retired `"diagonal" in family` name-sniff would route it to the
    full-covariance branch (wrong broadcast axis); the registered cov_kind (carried by the
    BeliefParams subclass via broadcast_over_keys) drives the dispatch."""
    import torch

    from vfe3.free_energy import pairwise_energy
    from vfe3.families.base import get_family, register_family, _FAMILIES
    from vfe3.families.gaussian import DiagonalGaussian

    name = "elliptical_scale_test"                 # no "diagonal" substring, declared diagonal

    @register_family(name)
    class _Elliptical(DiagonalGaussian):
        cov_kind = "diagonal"

    try:
        torch.manual_seed(11)
        N, K = 3, 4
        mu_q = torch.randn(N, K)
        mu_t = torch.randn(N, N, K)
        sigma_q = torch.rand(N, K) + 0.5
        sigma_t = torch.rand(N, N, K) + 0.5
        new = get_family(name)
        E_new = pairwise_energy(new(mu_q, sigma_q), new(mu_t, sigma_t))
        E_ref = pairwise_energy(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_t, sigma_t))
        assert torch.allclose(E_new, E_ref, atol=1e-6)
    finally:
        _FAMILIES.pop(name, None)


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
