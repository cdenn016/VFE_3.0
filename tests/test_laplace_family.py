r"""DiagonalLaplace -- the first non-Gaussian belief family (vfe3/families/laplace.py).

Pins the closed-form KL/Renyi against deterministic float64 trapezoidal integration, the divergence
axioms (non-negativity, self-divergence=0), the non-EF contract (natural/log_partition_at raise so
the generic Bregman path is bypassed by renyi_closed_form), gauge-transport honesty (permutation
preserves the diagonal Laplace exactly; a non-permutation rotation projects the marginal scale), and
end-to-end reachability through the autograd-of-F oracle and a full model.
"""

from tests.profiles import test_config

import math

import pytest
import torch

from vfe3.divergence import (
    bhattacharyya,
    divergence_families,
    get_family,
    jeffreys,
    kl,
    renyi,
    squared_hellinger,
)
from vfe3.families.laplace import DiagonalLaplace


# ---- deterministic 1-D references (float64 trapezoid) -----------------------

def _grid(mu_q, b_q, mu_p, b_p):
    lo = min(mu_q, mu_p) - 60.0 * max(b_q, b_p)
    hi = max(mu_q, mu_p) + 60.0 * max(b_q, b_p)
    return torch.linspace(lo, hi, 2_000_001, dtype=torch.float64)


def _logpdf(x, mu, b):
    return -math.log(2.0 * b) - (x - mu).abs() / b


def _kl_ref(mu_q, b_q, mu_p, b_p):
    x = _grid(mu_q, b_q, mu_p, b_p)
    q = torch.exp(_logpdf(x, mu_q, b_q))
    return torch.trapz(q * (_logpdf(x, mu_q, b_q) - _logpdf(x, mu_p, b_p)), x).item()


def _renyi_ref(mu_q, b_q, mu_p, b_p, alpha):
    x = _grid(mu_q, b_q, mu_p, b_p)
    aff = torch.trapz(torch.exp(alpha * _logpdf(x, mu_q, b_q) + (1.0 - alpha) * _logpdf(x, mu_p, b_p)), x)
    return (torch.log(aff) / (alpha - 1.0)).item()


def _entropy_ref(b):
    x = torch.linspace(-80.0 * b, 80.0 * b, 2_000_001, dtype=torch.float64)
    q = torch.exp(_logpdf(x, 0.0, b))
    return torch.trapz(-q * _logpdf(x, 0.0, b), x).item()


def _lap(mu, b):
    return DiagonalLaplace(torch.tensor(mu), torch.tensor(b))


# ---- registration / config ------------------------------------------------

def test_laplace_registered_and_config_selectable():
    assert "laplace_diagonal" in divergence_families()
    assert get_family("laplace_diagonal") is DiagonalLaplace
    assert DiagonalLaplace.cov_kind == "diagonal"
    with pytest.raises(ValueError, match="family=.*laplace_diagonal"):
        test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, family="laplace_diagonal")


# ---- non-EF contract: generic path is bypassed ----------------------------

def test_natural_and_log_partition_raise():
    q = _lap([0.0, 1.0], [1.0, 2.0])
    for fn in (q.natural, lambda: type(q).log_partition_at((q.mu, q.sigma))):
        try:
            fn()
            assert False, "expected NotImplementedError (Laplace is not an EF when location varies)"
        except NotImplementedError:
            pass


def test_all_registered_divergences_use_closed_form():
    # natural()/log_partition_at() raise, so a finite result PROVES renyi_closed_form was used
    # (the generic Bregman/Renyi-from-A path would have raised).
    q = _lap([0.0, 0.5], [1.0, 1.5])
    p = _lap([0.3, -0.2], [1.2, 0.8])
    for d in (kl(q, p), renyi(q, p, alpha=0.5), squared_hellinger(q, p), bhattacharyya(q, p), jeffreys(q, p)):
        assert torch.isfinite(d).all()


# ---- axioms ---------------------------------------------------------------

def test_self_divergence_is_zero():
    g = torch.Generator().manual_seed(0)
    q = DiagonalLaplace(torch.randn(4, 3, generator=g), torch.rand(4, 3, generator=g) + 0.3)
    for alpha in (0.1, 0.5, 0.9, 1.0):
        d = q.renyi_closed_form(q, alpha=alpha)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


def test_non_negativity():
    g = torch.Generator().manual_seed(1)
    q = DiagonalLaplace(torch.randn(6, 3, generator=g), torch.rand(6, 3, generator=g) + 0.3)
    p = DiagonalLaplace(torch.randn(6, 3, generator=g), torch.rand(6, 3, generator=g) + 0.3)
    for alpha in (0.1, 0.25, 0.5, 0.9, 1.0):
        assert (q.renyi_closed_form(p, alpha=alpha) >= 0.0).all()


def test_per_coord_sums_to_closed_form():
    g = torch.Generator().manual_seed(2)
    q = DiagonalLaplace(torch.randn(5, 4, generator=g), torch.rand(5, 4, generator=g) + 0.5)
    p = DiagonalLaplace(torch.randn(5, 4, generator=g), torch.rand(5, 4, generator=g) + 0.5)
    for alpha in (0.3, 0.5, 1.0):
        summed = q.renyi_per_coord(p, alpha=alpha).sum(dim=-1)
        assert torch.allclose(summed, q.renyi_closed_form(p, alpha=alpha), atol=1e-4)


# ---- golden vs deterministic integration ----------------------------------

_PAIRS = [(0.3, 1.0, -0.7, 2.0), (0.0, 0.5, 1.5, 0.5), (-2.0, 3.0, 2.0, 0.8)]


def test_kl_matches_reference():
    for mu_q, b_q, mu_p, b_p in _PAIRS:
        got = _lap([mu_q], [b_q]).renyi_closed_form(_lap([mu_p], [b_p]), alpha=1.0).item()
        assert abs(got - _kl_ref(mu_q, b_q, mu_p, b_p)) < 1e-4


def test_renyi_matches_reference():
    for alpha in (0.1, 0.25, 0.5, 0.9):
        for mu_q, b_q, mu_p, b_p in _PAIRS:
            got = _lap([mu_q], [b_q]).renyi_closed_form(_lap([mu_p], [b_p]), alpha=alpha).item()
            assert abs(got - _renyi_ref(mu_q, b_q, mu_p, b_p, alpha)) < 1e-4


def test_renyi_singularity_branch_is_continuous():
    # alpha* = b_q/(b_p+b_q): with b_q=1,b_p=2 the singular order is 1/3; the float64 limit branch
    # must stay continuous and match the (smooth) reference integral straddling and AT alpha*.
    b_q, b_p, mu_q, mu_p = 1.0, 2.0, 0.0, 1.0
    astar = b_q / (b_p + b_q)
    for alpha in (astar - 1e-3, astar, astar + 1e-3):
        got = _lap([mu_q], [b_q]).renyi_closed_form(_lap([mu_p], [b_p]), alpha=alpha).item()
        assert math.isfinite(got)
        assert abs(got - _renyi_ref(mu_q, b_q, mu_p, b_p, alpha)) < 1e-3


def test_entropy_matches_reference():
    for b in (0.5, 1.0, 2.5):
        got = _lap([0.0], [b]).entropy().item()
        assert abs(got - _entropy_ref(b)) < 1e-4
        assert abs(got - math.log(2.0 * b * math.e)) < 1e-6


# ---- gauge-transport honesty (the corrected boundary) ---------------------

def test_transport_permutation_exact_rotation_projects():
    K = 2
    b = torch.tensor([[1.0, 4.0]])                                  # (1, K) diagonal scale
    perm = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).reshape(1, 1, K, K)   # coord swap (permutation)
    family = get_family("laplace_diagonal")
    bt_perm = family.transport_dispersion(b, perm)[0, 0]            # exact |P| b = permuted b
    assert torch.allclose(bt_perm, torch.tensor([4.0, 1.0]), atol=1e-6)
    theta = 0.5
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]]).reshape(1, 1, K, K)
    bt_rot = torch.sort(family.transport_dispersion(b, R)[0, 0]).values   # variance-matched scales mix
    assert not torch.allclose(bt_rot, torch.sort(b[0]).values, atol=1e-3)   # not any permutation of b


def test_transport_zero_scale_uses_finite_zero_norm_subgradient():
    scale = torch.tensor([[0.0, 2.0]], requires_grad=True)
    omega = torch.eye(2).reshape(1, 1, 2, 2).requires_grad_()

    transported = DiagonalLaplace.transport_dispersion(scale, omega)
    transported.sum().backward()

    assert torch.equal(transported, torch.tensor([[[0.0, 2.0]]]))
    assert torch.equal(scale.grad, torch.tensor([[0.0, 1.0]]))
    assert torch.equal(
        omega.grad,
        torch.tensor([[[[0.0, 0.0], [0.0, 2.0]]]]),
    )


# ---- end-to-end reachability ----------------------------------------------

def test_oracle_gradients_finite_for_laplace():
    from vfe3.geometry.groups import get_group
    from vfe3.gradients.oracle import belief_gradients_autograd
    from vfe3.inference.e_step import build_belief_transport
    torch.manual_seed(0)
    g = get_group("block_glk")(4, 2)
    N, K, n_gen = 5, 4, g.generators.shape[0]
    phi = torch.randn(1, N, n_gen) * 0.1
    omega = build_belief_transport(phi, g, transport_mode="flat")
    mu = torch.randn(1, N, K); sigma = torch.rand(1, N, K) + 0.5
    mu_p = torch.randn(1, N, K); sigma_p = torch.rand(1, N, K) + 0.5
    kw = dict(tau=1.0, renyi_order=1.0, kl_max=100.0, eps=1e-6, b0=1.0, c0=1.0, value=1.0,
              include_attention_entropy=True, gradient_mode="filtering", family="laplace_diagonal",
              divergence_family="renyi", lambda_alpha_mode="constant", irrep_dims=g.irrep_dims, log_prior=None)
    g_mu, g_sigma = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, **kw)
    assert torch.isfinite(g_mu).all() and torch.isfinite(g_sigma).all()




# ---- gradient correctness: finite difference of F vs the autograd oracle ----
# The family's tests above pin divergence VALUES; these pin that the oracle's belief GRADIENTS are
# correct -- the |x-mu| cusp and the float64 singularity island must differentiate to match a central
# finite difference of the exact same F (CLAUDE.md: FD checks against the autograd-of-F oracle).

def _oracle_setup(N=3, K=2, seed=3):
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import compute_transport_operators
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]            # (N, N, K, K)
    mu = torch.randn(N, K, generator=g)
    sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def _F_laplace_filtering(mu_q, sigma_q, mu_p, sigma_p, mu_t, sigma_t, tau, order):
    # Mirrors belief_gradients_autograd's F construction EXACTLY for family='laplace_diagonal',
    # constant alpha, no rope / no omega_builder. Keys (mu_t, sigma_t) frozen -> the filtering F.
    from vfe3.alpha_i import self_coupling_alpha
    from vfe3.free_energy import free_energy, pairwise_energy, self_divergence_for_alpha
    sd = self_divergence_for_alpha(DiagonalLaplace(mu_q, sigma_q), DiagonalLaplace(mu_p, sigma_p),
                                   alpha=order, kl_max=100.0, eps=1e-6,
                                   divergence_family="renyi", lambda_alpha_mode="constant")
    alpha, _reg = self_coupling_alpha(sd, mode="constant", value=1.0, b0=1.0, c0=1.0)
    energy = pairwise_energy(DiagonalLaplace(mu_q, sigma_q), DiagonalLaplace(mu_t, sigma_t),
                             alpha=order, kl_max=100.0, eps=1e-6, divergence_family="renyi", irrep_dims=None)
    return free_energy(sd, energy, alpha, tau=tau, lambda_beta=1.0,
                       include_attention_entropy=True, log_prior=None, alpha_reg=None, coupling_energy=None)


def _check_oracle_fd(order, atol):
    from vfe3.geometry.transport import transport_mean
    from vfe3.gradients.oracle import belief_gradients_autograd
    mu, sigma, mu_p, sigma_p, omega = _oracle_setup()
    tau = 1.5
    gmu, gsig = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, tau=tau,
                                          renyi_order=order, gradient_mode="filtering",
                                          family="laplace_diagonal")
    mu_t = transport_mean(omega.unsqueeze(0), mu.unsqueeze(0))[0]          # frozen keys (filtering)
    family = get_family("laplace_diagonal")
    sigma_t = family.transport_dispersion(sigma.unsqueeze(0), omega.unsqueeze(0))[0]

    # 4th-order central difference: f'(x) = [f(x-2h) - 8 f(x-h) + 8 f(x+h) - f(x+2h)] / (12 h).
    # O(h^4) truncation keeps the FD tight even where the Renyi curvature is high near a coordinate's
    # singular order alpha*_k (a 2-point stencil there leaves ~5e-3 truncation; this stays ~1e-4).
    h = 1e-3
    gmu_fd = torch.zeros_like(mu); gsig_fd = torch.zeros_like(sigma)
    for a in range(mu.shape[0]):
        for b in range(mu.shape[1]):
            def fm(t, a=a, b=b):
                d = torch.zeros_like(mu); d[a, b] = t
                return _F_laplace_filtering(mu + d, sigma, mu_p, sigma_p, mu_t, sigma_t, tau, order)
            def fs(t, a=a, b=b):
                d = torch.zeros_like(sigma); d[a, b] = t
                return _F_laplace_filtering(mu, sigma + d, mu_p, sigma_p, mu_t, sigma_t, tau, order)
            gmu_fd[a, b] = (fm(-2 * h) - 8 * fm(-h) + 8 * fm(h) - fm(2 * h)) / (12 * h)
            gsig_fd[a, b] = (fs(-2 * h) - 8 * fs(-h) + 8 * fs(h) - fs(2 * h)) / (12 * h)
    assert torch.allclose(gmu, gmu_fd, atol=atol, rtol=atol)
    assert torch.allclose(gsig, gsig_fd, atol=atol, rtol=atol)


def test_laplace_oracle_gradient_matches_fd_kl():
    _check_oracle_fd(order=1.0, atol=2e-3)


def test_laplace_oracle_gradient_matches_fd_renyi():
    # alpha=0.5 drives the float64 singularity-island branch; autograd through the .double()/.to()
    # casts and the where-limit must still match a (4th-order) finite difference of F.
    _check_oracle_fd(order=0.5, atol=2e-3)


# ---- alpha>1 (non-convex regime), saturation, cusp, eps floor --------------

def test_laplace_alpha_gt_one_divergent_saturates_to_kl_max():
    # csum = alpha/b_q + (1-alpha)/b_p <= 0 -> the affinity integral diverges -> NaN -> kl_max
    # (the Gaussian non-PD-blend policy). c_q=0.15, c_p=-0.5 -> csum<0.
    d = _lap([0.0], [10.0]).renyi_closed_form(_lap([1.0], [1.0]), alpha=1.5, kl_max=100.0)
    assert torch.allclose(d, torch.tensor(100.0), atol=1e-4)


def test_laplace_alpha_gt_one_convergent_matches_reference():
    # csum > 0 (equal scales): the closed form holds beyond the convex regime; finite, >=0, == ref.
    got = _lap([0.0], [1.0]).renyi_closed_form(_lap([1.2], [1.0]), alpha=1.5, kl_max=100.0).item()
    assert math.isfinite(got) and got >= 0.0
    assert abs(got - _renyi_ref(0.0, 1.0, 1.2, 1.0, 1.5)) < 1e-3


def test_laplace_kl_max_saturation():
    # KL ~ 199 for a far-separated pair -> clamped to kl_max.
    d = _lap([0.0], [1.0]).renyi_closed_form(_lap([200.0], [1.0]), alpha=1.0, kl_max=100.0)
    assert torch.allclose(d, torch.tensor(100.0), atol=1e-4)


def test_laplace_cusp_and_eps_floor_finite():
    # s=0 cusp (equal means, unequal scales): finite and matches the reference at both KL and Renyi.
    for alpha in (0.5, 1.0):
        got = _lap([0.5], [1.0]).renyi_closed_form(_lap([0.5], [2.0]), alpha=alpha).item()
        ref = _kl_ref(0.5, 1.0, 0.5, 2.0) if alpha == 1.0 else _renyi_ref(0.5, 1.0, 0.5, 2.0, alpha)
        assert math.isfinite(got) and abs(got - ref) < 1e-3
    # scale below the eps floor is clamped, not NaN/inf (defense-in-depth)
    tiny, other = _lap([0.0], [1e-9]), _lap([0.3], [1.0])
    assert torch.isfinite(tiny.renyi_closed_form(other, alpha=1.0)).all()
    assert torch.isfinite(other.renyi_closed_form(tiny, alpha=1.0)).all()


# ---- device agreement (runs on the user's CUDA box; skipped on a CPU-only build) ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_laplace_cuda_matches_cpu():
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    mu_q = torch.randn(5, 3, generator=g); b_q = torch.rand(5, 3, generator=g) + 0.3
    mu_p = torch.randn(5, 3, generator=g); b_p = torch.rand(5, 3, generator=g) + 0.3
    for alpha in (0.5, 1.0):
        cpu = DiagonalLaplace(mu_q, b_q).renyi_closed_form(DiagonalLaplace(mu_p, b_p), alpha=alpha)
        cu = DiagonalLaplace(mu_q.to(dev), b_q.to(dev)).renyi_closed_form(
            DiagonalLaplace(mu_p.to(dev), b_p.to(dev)), alpha=alpha)
        assert torch.allclose(cpu, cu.cpu(), atol=1e-5)
    assert torch.allclose(DiagonalLaplace(mu_q, b_q).entropy(),
                          DiagonalLaplace(mu_q.to(dev), b_q.to(dev)).entropy().cpu(), atol=1e-5)
    # Model half retired with transported attention (PR #32): both projected routes reject Laplace.
    with pytest.raises(ValueError, match="laplace_diagonal"):
        test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                    n_e_steps=1, e_phi_lr=0.0, m_phi_lr=0.0, family="laplace_diagonal")


# ===========================================================================
# PB-14 (Task 5, 2026-07-12): the family-consistent decode reads a Laplace belief out under
# the Laplace divergence (not a hardcoded Gaussian KL). decode_mode='family' scores
# logits = -D_configured(q||p_v)/tau_eff through the configured Laplace family + functional,
# so a non-Gaussian belief has a geometry-matched prior-bank readout.
# ===========================================================================

from vfe3.divergence import get_functional as _get_functional          # noqa: E402
from vfe3.model.prior_bank import PriorBank, get_decode                 # noqa: E402
from vfe3.numerics import bounded_variance_from_log                     # noqa: E402


def _laplace_reference_logits(pb, mu_q, sigma_q, tau_eff):
    family_cls = get_family(pb.family)
    q = family_cls(mu_q.unsqueeze(-2), sigma_q.unsqueeze(-2))
    p_sigma = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)
    p = family_cls(pb._decode_mu_table(), p_sigma)
    functional = _get_functional(pb.divergence_family)
    energy = functional(q, p, alpha=pb.renyi_order, kl_max=float("inf"), eps=pb.eps)
    return -energy / tau_eff


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_family_decode_matches_direct_functional_laplace(alpha):
    torch.manual_seed(0)
    V, K, n_gen = 7, 3, 4
    pb = PriorBank(V, K, n_gen, decode_tau=1.4, family="laplace_diagonal",
                   divergence_family="renyi", renyi_order=alpha, decode_mode="family")
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.6)
        pb.sigma_log_embed.normal_(0.0, 0.3)
    mu_q = torch.randn(2, 4, K); sigma_q = torch.rand(2, 4, K) + 0.3
    tau_eff = pb._tau_eff()
    got = get_decode("family")(pb, mu_q, sigma_q, tau_eff)
    exp = _laplace_reference_logits(pb, mu_q, sigma_q, tau_eff)
    assert got.shape == (2, 4, V)
    assert torch.allclose(got, exp, atol=1e-4, rtol=0.0)


def test_laplace_family_decode_ranking_differs_from_gaussian_kl():
    # The same (mu, sigma) tables read out under the Laplace KL rank the vocabulary differently
    # from the Gaussian KL (the L1 vs L2 penalty on |mu_q - mu_v| and the scale term differ).
    torch.manual_seed(3)
    V, K, n_gen = 7, 3, 4
    lap = PriorBank(V, K, n_gen, family="laplace_diagonal", renyi_order=1.0, decode_mode="family")
    gauss = PriorBank(V, K, n_gen, family="gaussian_diagonal", renyi_order=1.0, decode_mode="family")
    with torch.no_grad():
        mu = torch.randn(V, K); sig_log = (torch.randn(V, K) * 0.5)
        for pb in (lap, gauss):
            pb.mu_embed.copy_(mu); pb.sigma_log_embed.copy_(sig_log)
    mu_q = torch.randn(1, 6, K); sigma_q = torch.rand(1, 6, K) + 0.3
    tau_eff = lap._tau_eff()
    lap_logits = get_decode("family")(lap, mu_q, sigma_q, tau_eff)
    gauss_logits = get_decode("family")(gauss, mu_q, sigma_q, tau_eff)
    assert not torch.equal(lap_logits.argsort(-1), gauss_logits.argsort(-1))   # ranking genuinely differs


def test_laplace_prior_bank_requires_family_consistent_decode() -> None:
    from vfe3.model.prior_bank import get_decode_registration

    # Registration metadata names the decoder that preserves the selected family.
    assert get_decode_registration("family_chunked").family_consistent
    assert not get_decode_registration("diagonal_chunked").family_consistent
    bank = PriorBank(6, 4, 8, family="laplace_diagonal", decode_mode="family_chunked")
    mu = torch.randn(2, 3, 4, requires_grad=True)
    sigma = torch.ones_like(mu)
    targets = torch.randint(0, 6, (2, 3))
    loss = bank.decode_ce_family_chunked(mu, sigma, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
