import math

import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.gradients.oracle import belief_gradients_autograd


def _rope_rotation(N, K, base_theta=0.3):
    # Block-diagonal SO(2) gauge-RoPE rotation R(theta_i), theta_i = i * base_theta (K=2 single block).
    rot = torch.zeros(N, K, K)
    for i in range(N):
        th = i * base_theta
        c, s = math.cos(th), math.sin(th)
        rot[i, 0, 0] = c; rot[i, 0, 1] = -s; rot[i, 1, 0] = s; rot[i, 1, 1] = c
    return rot


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]          # (N, N, K, K)
    mu = torch.randn(N, K, generator=g)
    sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def _F_filtering(mu_q, sigma_q, mu_p, sigma_p, mu_t, sigma_t, tau):
    # F as a function of the QUERY role only. The transported keys (mu_t, sigma_t)
    # are FROZEN: built once from the unperturbed belief and passed in, so a
    # finite difference of this F holds the key role fixed (the filtering F).
    # (Re-deriving mu_t = mu_q.detach() inside would NOT freeze the keys under FD,
    # since .detach() blocks autograd, not numeric perturbation -- that would
    # measure the full/smoothing gradient instead.)
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
    sd = self_divergence(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_p, sigma_p))
    energy = pairwise_energy(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_t, sigma_t))
    alpha = torch.ones(mu_q.shape[0])
    return free_energy(sd, energy, alpha, tau=tau, include_attention_entropy=True)


def test_filtering_oracle_matches_finite_difference_of_F_filt():
    from vfe3.geometry.transport import transport_covariance, transport_mean
    mu, sigma, mu_p, sigma_p, omega = _setup()
    tau = 1.5
    gmu, gsig = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                          tau=tau, gradient_mode="filtering")
    # frozen keys from the unperturbed belief (the filtering split)
    mu_t = transport_mean(omega.unsqueeze(0), mu.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma.unsqueeze(0))[0]
    eps = 5e-3
    gmu_fd = torch.zeros_like(mu)
    for a in range(mu.shape[0]):
        for b in range(mu.shape[1]):
            d = torch.zeros_like(mu); d[a, b] = eps
            fp = _F_filtering(mu + d, sigma, mu_p, sigma_p, mu_t, sigma_t, tau)
            fm = _F_filtering(mu - d, sigma, mu_p, sigma_p, mu_t, sigma_t, tau)
            gmu_fd[a, b] = (fp - fm) / (2 * eps)
    assert torch.allclose(gmu, gmu_fd, atol=1e-3, rtol=1e-3)


def test_smoothing_differs_from_filtering_by_keyside():
    mu, sigma, mu_p, sigma_p, omega = _setup()
    gf_mu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                         tau=1.5, gradient_mode="filtering")
    gs_mu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                         tau=1.5, gradient_mode="smoothing")
    # the key-side (column) term is non-zero -> the two modes differ
    assert not torch.allclose(gf_mu, gs_mu, atol=1e-4)


def test_oracle_runs_under_no_grad():
    # The belief oracle is part of the model FORWARD (iterative belief minimization), so it
    # must compute the belief gradient even when the caller wraps the forward in no_grad --
    # the eval() / diagnostics() / generate() regime (evaluate is @torch.no_grad). The
    # autograd oracle therefore needs its own enable_grad island (the phi step already has
    # one); without it torch.autograd.grad(F, ...) raises because F has no grad_fn under
    # no_grad. The closed-form kernel path is unaffected, so this gap silently breaks
    # evaluation of every NON-kernel config (surrogate / smoothing / full-cov / renyi a!=1).
    mu, sigma, mu_p, sigma_p, omega = _setup()
    g_on = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, tau=1.5, gradient_mode="filtering")
    with torch.no_grad():
        g_off = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, tau=1.5, gradient_mode="filtering")
    # value-identical regardless of the caller's ambient grad state, and detached (safe to use
    # as a constant tangent inside a no_grad forward)
    assert torch.allclose(g_off[0], g_on[0], atol=1e-6)
    assert torch.allclose(g_off[1], g_on[1], atol=1e-6)
    assert not g_off[0].requires_grad and not g_off[1].requires_grad


def _F_full(mu_q, sigma_q, mu_p, sigma_p, omega, tau):
    # F as a function of a SINGLE shared leaf: the transported keys are rebuilt
    # from the SAME live (mu_q, sigma_q) on every FD perturbation, so the column
    # (key-side) role moves too -> a finite difference of this F measures the FULL
    # d F_red (the smoothing gradient), the Omega^T pullback included.
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
    from vfe3.geometry.transport import transport_covariance, transport_mean
    mu_t = transport_mean(omega.unsqueeze(0), mu_q.unsqueeze(0))[0]
    sigma_t = transport_covariance(omega.unsqueeze(0), sigma_q.unsqueeze(0))[0]
    sd = self_divergence(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_p, sigma_p))
    energy = pairwise_energy(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_t, sigma_t))
    alpha = torch.ones(mu_q.shape[0])
    return free_energy(sd, energy, alpha, tau=tau, include_attention_entropy=True)


def test_smoothing_oracle_matches_finite_difference_of_F_full():
    # POSITIVE anchor for the pure path: smoothing == FD(full F). Pins the key-side
    # Omega^T pullback's sign and scale (a flipped/mis-scaled/missing key-side would
    # fail here, where `smoothing != filtering` alone would still pass).
    mu, sigma, mu_p, sigma_p, omega = _setup()
    tau = 1.5
    gmu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega,
                                       tau=tau, gradient_mode="smoothing")
    eps = 1e-3
    gmu_fd = torch.zeros_like(mu)
    for a in range(mu.shape[0]):
        for b in range(mu.shape[1]):
            d = torch.zeros_like(mu); d[a, b] = eps
            fp = _F_full(mu + d, sigma, mu_p, sigma_p, omega, tau)
            fm = _F_full(mu - d, sigma, mu_p, sigma_p, omega, tau)
            gmu_fd[a, b] = (fp - fm) / (2 * eps)
    assert torch.allclose(gmu, gmu_fd, atol=2e-3, rtol=2e-3)


# --- value-gauge decoupling (RopeTransport.on_value=False) ------------------

def test_value_gauge_decoupling_excluded_from_kernel_route():
    from vfe3.gradients.kernels import uses_kernel_route
    common = dict(renyi_order=1.0, gradient_mode="filtering", family="gaussian_diagonal",
                  divergence_family="renyi", include_attention_entropy=True)
    assert uses_kernel_route(**common, decoupled_value_gauge=False) is True
    assert uses_kernel_route(**common, decoupled_value_gauge=True) is False


def test_value_gauge_decoupling_changes_oracle_gradient():
    from vfe3.geometry.transport import RopeTransport
    mu, sigma, mu_p, sigma_p, omega = _setup(N=4, K=2)
    rope = _rope_rotation(4, 2)
    coh = RopeTransport(base=omega, rope=rope, on_cov=False, on_value=True, insertion="left")
    dec = RopeTransport(base=omega, rope=rope, on_cov=False, on_value=False, insertion="left")
    g_coh, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, coh, tau=1.5, gradient_mode="filtering")
    g_dec, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, dec, tau=1.5, gradient_mode="filtering")
    # Decoupling the value gauge from the score gauge genuinely changes the belief-coupling pull.
    assert not torch.allclose(g_coh, g_dec, atol=1e-4)


def _F_decoupled_filtering(mu_q, sigma_q, mu_p, sigma_p, mu_ts, sigma_ts, mu_tv, sigma_tv, tau):
    # Decoupled filtering F: beta from the rotated SCORE energy, coupling sum from the base VALUE
    # energy; both transported-key sets frozen from the unperturbed belief (filtering split).
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.free_energy import free_energy, pairwise_energy, self_divergence
    sd = self_divergence(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_p, sigma_p))
    energy = pairwise_energy(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_ts, sigma_ts))
    coupling_energy = pairwise_energy(DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_tv, sigma_tv))
    alpha = torch.ones(mu_q.shape[0])
    return free_energy(sd, energy, alpha, tau=tau, coupling_energy=coupling_energy,
                       include_attention_entropy=True)


def test_value_gauge_decoupled_oracle_matches_fd_of_decoupled_F():
    from vfe3.geometry.transport import RopeTransport, transport_covariance, transport_mean
    mu, sigma, mu_p, sigma_p, omega = _setup(N=4, K=2)
    rope = _rope_rotation(4, 2)
    dec = RopeTransport(base=omega, rope=rope, on_cov=False, on_value=False, insertion="left")
    tau = 1.5
    gmu, _ = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, dec, tau=tau, gradient_mode="filtering")
    # Frozen keys (filtering): score keys via the rotated transport, value keys via the base transport.
    mu_ts = transport_mean(dec, mu)
    sigma_ts = transport_covariance(dec, sigma)
    mu_tv = transport_mean(omega.unsqueeze(0), mu.unsqueeze(0))[0]
    sigma_tv = transport_covariance(omega.unsqueeze(0), sigma.unsqueeze(0))[0]
    eps = 5e-3
    gmu_fd = torch.zeros_like(mu)
    for a in range(mu.shape[0]):
        for b in range(mu.shape[1]):
            d = torch.zeros_like(mu); d[a, b] = eps
            fp = _F_decoupled_filtering(mu + d, sigma, mu_p, sigma_p, mu_ts, sigma_ts, mu_tv, sigma_tv, tau)
            fm = _F_decoupled_filtering(mu - d, sigma, mu_p, sigma_p, mu_ts, sigma_ts, mu_tv, sigma_tv, tau)
            gmu_fd[a, b] = (fp - fm) / (2 * eps)
    assert torch.allclose(gmu, gmu_fd, atol=1e-3, rtol=1e-3)
