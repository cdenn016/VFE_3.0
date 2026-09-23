import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import free_energy_value


def _belief(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    n_gen = grp.generators.shape[0]
    b = BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return b, mu_p, sigma_p, grp


def test_belief_state_fields():
    b, *_ = _belief()
    assert b.mu.shape == (3, 2) and b.sigma.shape == (3, 2)


def test_free_energy_value_is_finite_scalar():
    b, mu_p, sigma_p, grp = _belief()
    F = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)
    assert F.shape == () and torch.isfinite(F)


def test_free_energy_filtering_equals_global_at_a_point():
    # F_filt and global F are the SAME NUMBER at a fixed belief (detach changes
    # gradients, not the value); they differ only as functions under a step.
    b, mu_p, sigma_p, grp = _belief()
    Fg = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=None)
    Ff = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=b)   # keys frozen at b
    assert torch.allclose(Fg, Ff, atol=1e-6)


def test_filtered_free_energy_tracks_the_current_query_frame():
    # GL(K) finding #6: with frozen keys, the filtered objective must build Omega_ij from the
    # CURRENT query frame phi_i (belief) and the FROZEN key frame phi_j (keys) -- not freeze
    # both at keys. So changing only the query phi (keys fixed) must change the filtered F.
    b, mu_p, sigma_p, grp = _belief()
    b2 = BeliefState(mu=b.mu, sigma=b.sigma, phi=b.phi + 0.5)        # same beliefs, shifted query frame
    F1 = free_energy_value(b,  mu_p, sigma_p, grp, tau=1.5, keys=b)  # keys frozen at b
    F2 = free_energy_value(b2, mu_p, sigma_p, grp, tau=1.5, keys=b)  # keys still frozen at b
    assert not torch.allclose(F1, F2, atol=1e-5)


# --- Task 2: one inner iteration -------------------------------------------
from vfe3.inference.e_step import e_step_iteration


def test_iteration_keeps_sigma_positive_and_shapes():
    b, mu_p, sigma_p, grp = _belief()
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5,
                           e_q_mu_lr=0.05, e_q_sigma_lr=0.05, e_phi_lr=0.05)
    assert (out.sigma > 0).all()
    assert out.mu.shape == b.mu.shape and out.phi.shape == b.phi.shape


def test_decoupled_learning_rates_freeze_components():
    b, mu_p, sigma_p, grp = _belief()
    o1 = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.05, e_q_sigma_lr=0.05, e_phi_lr=0.0)
    assert torch.allclose(o1.phi, b.phi, atol=1e-7)
    o2 = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.0, e_q_sigma_lr=0.05, e_phi_lr=0.0)
    assert torch.allclose(o2.mu, b.mu, atol=1e-7)


def test_e_step_iteration_spd_affine_default_is_byte_identical():
    """The registry-routed default spd_affine retraction reproduces the legacy
    natural_gradient + retract_spd_diagonal sigma update bit-for-bit (atol=0)."""
    from vfe3.geometry.retraction import natural_gradient, retract_spd_diagonal
    from vfe3.gradients.kernels import belief_gradients
    from vfe3.inference.e_step import _transport

    b, mu_p, sigma_p, grp = _belief()
    e_q_sigma_lr, e_sigma_q_trust, eps, sigma_max = 0.05, 5.0, 1e-6, 5.0

    # Hand-compose the legacy sigma update exactly as e_step_iteration did pre-refactor.
    omega = _transport(b.phi, grp)
    grad_mu, grad_sigma = belief_gradients(
        b.mu, b.sigma, mu_p, sigma_p, omega, tau=1.5, irrep_dims=grp.irrep_dims,
    )
    _, nat_sigma = natural_gradient(grad_mu, grad_sigma, b.sigma, eps=eps)
    legacy_sigma = retract_spd_diagonal(
        b.sigma, -e_q_sigma_lr * nat_sigma, trust_region=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
    )

    out = e_step_iteration(
        b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.0, e_q_sigma_lr=e_q_sigma_lr, e_phi_lr=0.0,
        e_sigma_q_trust=e_sigma_q_trust, eps=eps, sigma_max=sigma_max,
    )
    assert torch.equal(out.sigma, legacy_sigma)


def test_transport_flat_kwarg_is_byte_identical_to_default():
    """``_transport`` with the default transport_mode='flat' is bit-identical (torch.equal) to
    the registry-routed flat builder, on both the 2-D (diagnostics) and 3-D (batched) paths."""
    from vfe3.inference.e_step import _transport

    b, mu_p, sigma_p, grp = _belief()
    # 2-D (N, n_gen) diagnostics path
    o2_default = _transport(b.phi, grp)
    o2_flat    = _transport(b.phi, grp, transport_mode="flat")
    assert torch.equal(o2_flat, o2_default)
    # 3-D (B, N, n_gen) batched path
    phi3 = b.phi.unsqueeze(0).expand(2, *b.phi.shape).contiguous()
    o3_default = _transport(phi3, grp)
    o3_flat    = _transport(phi3, grp, transport_mode="flat")
    assert torch.equal(o3_flat, o3_default)


def test_e_step_iteration_transport_flat_default_is_byte_identical():
    """The registry-routed default transport_mode='flat' reproduces the E-step iteration
    bit-for-bit (atol=0) against the run with no transport_mode passed."""
    b, mu_p, sigma_p, grp = _belief()
    base = e_step_iteration(
        b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.05, e_q_sigma_lr=0.05, e_phi_lr=0.05,
    )
    routed = e_step_iteration(
        b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.05, e_q_sigma_lr=0.05, e_phi_lr=0.05,
        transport_mode="flat",
    )
    assert torch.equal(routed.mu, base.mu)
    assert torch.equal(routed.sigma, base.sigma)
    assert torch.equal(routed.phi, base.phi)


# --- Task 3: descent directions (the right objective per mode) -------------
def test_filtering_step_descends_F_filt():
    # filtering (query-side) gradient descends F with KEYS FROZEN at the pre-step belief.
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5, keys=b)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=1e-3, e_q_sigma_lr=1e-3,
                           e_phi_lr=0.0, gradient_mode="filtering", e_sigma_q_trust=0.0)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5, keys=b)   # SAME frozen keys b
    assert F_after < F_before


def test_smoothing_step_descends_global_F():
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)            # global (keys=belief)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=1e-3, e_q_sigma_lr=1e-3,
                           e_phi_lr=0.0, gradient_mode="smoothing", e_sigma_q_trust=0.0)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5)
    assert F_after < F_before


def test_phi_step_descends_global_F_with_beliefs_frozen():
    b, mu_p, sigma_p, grp = _belief()
    F_before = free_energy_value(b, mu_p, sigma_p, grp, tau=1.5)
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=0.0, e_q_sigma_lr=0.0, e_phi_lr=1e-3)
    F_after = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5)
    assert F_after < F_before


def test_keys_freezing_bites_on_stepped_belief():
    # Pin that `keys` actually freezes the transported second KL argument: after a
    # (phi-frozen) filtering step the query role has moved off the pre-step keys, so
    # F_filt (keys=b, transported from the frozen pre-step belief) MUST differ from
    # global F (keys=None, re-transported from the moved belief). Without this the
    # F_filt machinery is unobserved -- both descent tests stay green if `keys` were dead.
    b, mu_p, sigma_p, grp = _belief()
    out = e_step_iteration(b, mu_p, sigma_p, grp, tau=1.5, e_q_mu_lr=5e-2, e_q_sigma_lr=5e-2,
                           e_phi_lr=0.0, gradient_mode="filtering")
    F_filt = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5, keys=b)      # frozen pre-step keys
    F_glob = free_energy_value(out, mu_p, sigma_p, grp, tau=1.5, keys=None)   # re-transported
    assert not torch.allclose(F_filt, F_glob, atol=1e-5)


# --- Task 4: e_step loop + trajectory + fixed-seed regression --------------
from vfe3.inference.e_step import e_step

# Re-frozen 2026-09-23 (audit #6): the filtering phi step now differentiates only each frame's
# own receiver row (sender frames detached); with that detach disabled the run still gives 6.6499.
EXPECTED_CHECKSUM = 6.7064   # seed=7, n_iter=3


def test_e_step_runs_n_iter_and_returns_trajectory():
    b, mu_p, sigma_p, grp = _belief()
    out, traj = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=5,
                       e_q_mu_lr=1e-2, e_q_sigma_lr=1e-2, e_phi_lr=1e-2, return_trajectory=True)
    assert len(traj) == 6
    assert (out.sigma > 0).all()


def test_smoothing_loop_decreases_F_overall():
    b, mu_p, sigma_p, grp = _belief()
    out, traj = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=10,
                       e_q_mu_lr=2e-3, e_q_sigma_lr=2e-3, e_phi_lr=2e-3,
                       gradient_mode="smoothing", e_sigma_q_trust=0.0, return_trajectory=True)
    assert traj[-1] < traj[0]


def test_fixed_seed_regression():
    b, mu_p, sigma_p, grp = _belief(seed=7)
    out = e_step(b, mu_p, sigma_p, grp, tau=1.5, n_iter=3,
                 e_q_mu_lr=1e-2, e_q_sigma_lr=1e-2, e_phi_lr=1e-2)
    assert torch.isfinite(out.mu).all() and torch.isfinite(out.sigma).all() and torch.isfinite(out.phi).all()
    checksum = float(out.mu.sum() + out.sigma.sum() + out.phi.sum())
    assert abs(checksum - EXPECTED_CHECKSUM) < 1e-3


# --- audit Group 2 (1c/5a): the **kwargs sink became explicit accept-and-ignore knobs ---
def test_free_energy_value_rejects_misspelled_kwarg():
    """A misspelled real parameter now raises TypeError instead of being silently swallowed,
    while genuine iteration-only knobs are still accepted and ignored."""
    b, mu_p, sigma_p, grp = _belief()
    # iteration-only knobs accepted and ignored (one call site forwards them to both
    # free_energy_value and e_step_iteration)
    F = free_energy_value(
        b, mu_p, sigma_p, grp, tau=1.0,
        gradient_mode="filtering", phi_precond_mode="none", phi_retract_mode="euclidean",
        sigma_max=5.0, e_sigma_q_trust=5.0,
    )
    assert torch.isfinite(F)
    with pytest.raises(TypeError):
        free_energy_value(b, mu_p, sigma_p, grp, tau=1.0, familly="gaussian_diagonal")


def test_oracle_create_graph_keeps_belief_gradient_connected():
    # Non-kernel configs (renyi_order != 1, gaussian_full, smoothing) fall back to the autograd oracle.
    # Under the unrolled E-step (create_graph=True) the oracle must return a belief gradient that is
    # still DIFFERENTIABLE, so the unrolled-through-inference signal reaches the prior tables; the
    # default (create_graph=False) path returns a detached constant tangent (byte-compatible with old).
    from vfe3.geometry.transport import compute_transport_operators
    from vfe3.gradients.kernels import belief_gradients

    grp = get_group("glk")(4)
    N, K = 3, 4
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(N, K, generator=g, requires_grad=True)
    sigma = (torch.rand(N, K, generator=g) + 0.5).requires_grad_(True)
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    omega = compute_transport_operators(
        torch.zeros(1, N, grp.generators.shape[0]), grp)["Omega"][0]      # (N,N,K,K), Omega=I
    kw = dict(family="gaussian_diagonal", divergence_family="renyi", renyi_order=2.0)  # alpha!=1 -> oracle

    g_mu, _ = belief_gradients(mu, sigma, mu_p, sigma_p, omega, create_graph=True, **kw)
    assert g_mu.requires_grad and g_mu.grad_fn is not None     # connected -> unrolled signal flows
    g_mu_d, _ = belief_gradients(mu, sigma, mu_p, sigma_p, omega, **kw)
    assert not g_mu_d.requires_grad                            # default detached (old behavior)
    assert torch.allclose(g_mu.detach(), g_mu_d)               # same VALUES; only connectivity differs


def test_oracle_unroll_grad_toggle_changes_prior_gradient():
    # End-to-end: for a DIAGONAL non-kernel family (renyi_order != 1 -> autograd oracle), the opt-in
    # oracle_unroll_grad adds the unrolled-through-inference term to the prior gradient, so mu_embed's
    # gradient DIFFERS from the default detached oracle (both finite). gaussian_diagonal keeps the
    # double-backward stable; gaussian_full is intentionally excluded (its eigh double-backward NaNs).
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    def grad_of(toggle: bool) -> torch.Tensor:
        cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=1, max_seq_len=5, n_layers=1,
                         n_e_steps=3, e_q_mu_lr=0.2, e_q_sigma_lr=0.1, e_phi_lr=0.0,
                         renyi_order=2.0, pos_phi="none", oracle_unroll_grad=toggle, seed=0)
        torch.manual_seed(0)
        m = VFEModel(cfg)
        tok = torch.randint(0, 12, (2, 5), generator=torch.Generator().manual_seed(1))
        tgt = torch.randint(0, 12, (2, 5), generator=torch.Generator().manual_seed(2))
        _, loss, _ = m(tok, tgt)
        loss.backward()
        return m.prior_bank.mu_embed.grad.clone()

    g_off, g_on = grad_of(False), grad_of(True)
    assert torch.isfinite(g_off).all() and torch.isfinite(g_on).all()
    assert not torch.allclose(g_off, g_on)     # the unrolled-through-inference signal reaches the prior


def test_full_mm_failed_sigma_star_retains_the_old_state_and_counts_the_fallback():
    """A non-PD candidate cannot move a full-Gaussian belief row."""
    from vfe3.gradients.kernels import mm_damped_precision_blend_full
    from vfe3.numerics import mm_cholesky_fallback_elements, reset_mm_cholesky_fallback_elements

    mu_old = torch.tensor([[1.5, -2.0]])
    sigma_old = torch.eye(2).unsqueeze(0)
    mu_star = torch.tensor([[9.0, 7.0]])
    sigma_star = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]])
    reset_mm_cholesky_fallback_elements()

    mu, sigma = mm_damped_precision_blend_full(
        mu_old, sigma_old, mu_star, sigma_star, 0.75, eps=1e-6, sigma_max=None)

    assert torch.equal(mu, mu_old)
    assert torch.equal(sigma, sigma_old)
    assert mm_cholesky_fallback_elements() == 1


def test_full_mm_mixed_batch_updates_valid_row_and_retains_invalid_neighbor():
    """One failed candidate cannot freeze or contaminate its valid batch neighbor."""
    from vfe3.gradients.kernels import mm_damped_precision_blend_full
    from vfe3.numerics import mm_cholesky_fallback_elements, reset_mm_cholesky_fallback_elements

    mu_old = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    sigma_old = torch.eye(2).repeat(2, 1, 1)
    mu_star = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    sigma_star = torch.stack((2.0 * torch.eye(2), torch.diag(torch.tensor([1.0, -1.0]))))
    reset_mm_cholesky_fallback_elements()

    mu, sigma = mm_damped_precision_blend_full(
        mu_old, sigma_old, mu_star, sigma_star, 1.0, eps=1e-6, sigma_max=None)

    torch.testing.assert_close(mu[0], mu_star[0])
    torch.testing.assert_close(sigma[0], sigma_star[0])
    assert torch.equal(mu[1], mu_old[1])
    assert torch.equal(sigma[1], sigma_old[1])
    assert mm_cholesky_fallback_elements() == 1


def test_full_mm_rejects_a_monkeypatched_partial_cholesky_factor(monkeypatch):
    """A finite partial factor is still a failed factor and must select the old state."""
    from vfe3.gradients import kernels as kernels_module

    mu_old = torch.tensor([[1.0, -2.0]])
    sigma_old = torch.eye(2).unsqueeze(0)
    mu_star = torch.tensor([[8.0, 9.0]])
    sigma_star = 2.0 * torch.eye(2).unsqueeze(0)
    real_safe_cholesky = kernels_module.safe_cholesky
    calls = {"count": 0}

    def partial_factor(matrix, *, eps, rounds):
        calls["count"] += 1
        factor, ok = real_safe_cholesky(matrix, eps=eps, rounds=rounds)
        if calls["count"] == 2:  # sigma_star: supply a finite factor with an explicit failure mask
            return torch.eye(matrix.shape[-1], dtype=matrix.dtype).expand_as(factor), torch.zeros_like(ok)
        return factor, ok

    monkeypatch.setattr(kernels_module, "safe_cholesky", partial_factor)
    mu, sigma = kernels_module.mm_damped_precision_blend_full(
        mu_old, sigma_old, mu_star, sigma_star, 0.5, eps=1e-6, sigma_max=None)

    assert calls["count"] >= 2
    assert torch.equal(mu, mu_old)
    assert torch.equal(sigma, sigma_old)


def test_full_mm_multiblock_failure_retains_the_entire_old_row(monkeypatch):
    """A failed head factor invalidates the full covariance row, not only that block."""
    from vfe3.gradients import kernels as kernels_module
    from vfe3.numerics import mm_cholesky_fallback_elements, reset_mm_cholesky_fallback_elements

    mu_old = torch.tensor([[1.0, -2.0]])
    sigma_old = torch.eye(2).unsqueeze(0)
    mu_p = torch.tensor([[4.0, 5.0]])
    sigma_p = torch.eye(2).unsqueeze(0)
    mu_t = torch.zeros(1, 1, 2)
    sigma_t = torch.eye(2).reshape(1, 1, 2, 2)
    a = torch.ones(1, 1)
    w = torch.ones(2, 1, 1)
    real_safe_cholesky = kernels_module.safe_cholesky
    calls = {"count": 0}

    def first_head_fails(matrix, *, eps, rounds):
        calls["count"] += 1
        factor, ok = real_safe_cholesky(matrix, eps=eps, rounds=rounds)
        if calls["count"] == 1:
            return torch.eye(matrix.shape[-1], dtype=matrix.dtype).expand_as(factor), torch.zeros_like(ok)
        return factor, ok

    monkeypatch.setattr(kernels_module, "safe_cholesky", first_head_fails)
    reset_mm_cholesky_fallback_elements()
    mu, sigma = kernels_module._mm_exact_full_covariance(
        mu=mu_old, sigma=sigma_old, mu_p=mu_p, sigma_p=sigma_p,
        mu_t=mu_t, sigma_t=sigma_t, a=a, w=w, eps=1e-6, irrep_dims=[1, 1],
        need_sigma_update=True,
    )

    assert calls["count"] == 2
    assert torch.equal(mu, mu_old)
    assert torch.equal(sigma, sigma_old)
    assert mm_cholesky_fallback_elements() == 1
