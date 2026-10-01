r"""``e_step_update="mm_exact"`` unblocked for ``family="gaussian_full"`` (2026-08-07 build).

Covers the required changes of the WAVE4 mm_exact-full-covariance task:

  A. capability-driven ``uses_kernel_route`` (kernels.py), replacing the ``family ==
     'gaussian_diagonal'`` literal;
  B. ``mm_exact_update``'s rank-generic dispatch, with a NEW dense K x K precision-fusion branch
     for a full-covariance family (``_mm_exact_full_covariance``), single gauge-irrep block only;
  C. the relaxed ``config.py`` ``e_step_update='mm_exact'`` guard (mm_exact CAPABILITY, decoupled
     from the analytic-gradient-kernel capability, plus a multi-head fail-fast);
  D. the ``gaussian_full`` ``_KERNELS`` registration (mm_exact-only capability; no analytic
     gradient kernel -- ``e_step_update='gradient'`` must keep routing to the autograd oracle).

Test 1 is the highest-priority requirement: the pre-existing ``family='gaussian_diagonal'`` path
must be BYTE-IDENTICAL to the pre-change code. It is checked against the actual pre-change source
(``git show cfc7e33:vfe3/gradients/kernels.py``, the commit this branch built on), executed as an
isolated module, rather than a hand-rederived reference -- the strongest form of "capture golden
values before editing" available after the fact.
"""

from tests.profiles import test_config

import dataclasses
import importlib.util
import pathlib
import subprocess

import pytest
import torch

from vfe3.alpha_i import alpha_gradient_coefficient, alpha_is_per_coord
from vfe3.belief import BeliefState
from vfe3.families.base import get_family
from vfe3.free_energy import attention_weights, pairwise_energy, self_divergence_for_alpha
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    compute_transport_operators,
    transport_covariance,
    transport_mean,
)
from vfe3.gradients import kernels as kernels_mod
from vfe3.gradients.kernels import (
    belief_gradients,
    get_kernel_registration,
    has_kernel,
    mm_damped_precision_blend_full,
    mm_exact_families,
    mm_exact_update,
    uses_kernel_route,
)
from vfe3.inference.e_step import BeliefState as _E_BeliefState  # noqa: F401  (sanity: same type)
from vfe3.inference.e_step import e_step_iteration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PRE_CHANGE_COMMIT = "cfc7e33"   # "Record the WAVE4 pure-path deep audit" -- tip before this task


# --------------------------------------------------------------------------------------------
# Golden (pre-change) module loader, for the byte-identical diagonal-path regression guard.
# --------------------------------------------------------------------------------------------

def _load_golden_kernels_module():
    r"""Exec the PRE-CHANGE ``vfe3/gradients/kernels.py`` (git blob) as an isolated module.

    Every OTHER module it imports from (``vfe3.families.gaussian``, ``vfe3.free_energy``,
    ``vfe3.geometry.transport``, ``vfe3.alpha_i``, ``vfe3.gradients.oracle``,
    ``vfe3.gradients.pairwise_stats``) is untouched by this task's diff, so the golden module's
    absolute imports resolve against the SAME live dependency code the current module uses -- the
    only thing this isolates is ``kernels.py`` itself (its own ``_KERNELS``/``_COMPILED_KERNELS``
    live in the golden module's own fresh namespace, so registering "gaussian_diagonal" there
    cannot collide with the live registry).
    """
    src = subprocess.check_output(
        ["git", "show", f"{_PRE_CHANGE_COMMIT}:vfe3/gradients/kernels.py"],
        cwd=str(REPO_ROOT), text=True,
    )
    # The historical kernel only uses this removed class for a type membership
    # check. These dense-tensor oracles never instantiate it; keep the shim local.
    src = src.replace("    DirectLinkTransport,\n", "")
    spec = importlib.util.spec_from_loader("_golden_kernels_20260807", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.DirectLinkTransport = type("HistoricalDirectLinkTransport", (), {})
    exec(compile(src, f"<golden:{_PRE_CHANGE_COMMIT}:vfe3/gradients/kernels.py>", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def golden_kernels():
    # An independent clone may not carry the pre-change history (VFE_3.0's starts at 57d9ea0). The
    # byte-identity guard cannot run without the historical blob, so skip with that reason: the
    # nodes are NOT RUN, never passed. Any other git failure still errors in the loader.
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{_PRE_CHANGE_COMMIT}^{{commit}}"],
        cwd=str(REPO_ROOT), capture_output=True,
    ).returncode == 0
    if not present:
        pytest.skip(f"golden commit {_PRE_CHANGE_COMMIT} is not in this clone's git history; the "
                    "pre-change kernels.py oracle is unavailable (byte-identity guard not run)")
    return _load_golden_kernels_module()


def _diag_setup(N=6, K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    phi = 0.1 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g)
    sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    log_prior = torch.zeros(N, N)
    return mu, sigma, mu_p, sigma_p, omega, log_prior, grp


class Test1_DiagonalPathByteIdentical:
    r"""Highest priority: family='gaussian_diagonal' must be UNCHANGED by this task's diff."""

    def test_mm_exact_update_byte_identical(self, golden_kernels):
        for seed in range(5):
            mu, sigma, mu_p, sigma_p, omega, log_prior, _ = _diag_setup(seed=seed)
            g_mu, g_sigma = golden_kernels.mm_exact_update(
                mu, sigma, mu_p, sigma_p, omega, tau=1.3, lambda_beta=0.8,
                lambda_alpha_mode="state_dependent", log_prior=log_prior,
            )
            c_mu, c_sigma = mm_exact_update(
                mu, sigma, mu_p, sigma_p, omega, tau=1.3, lambda_beta=0.8,
                lambda_alpha_mode="state_dependent", log_prior=log_prior,
            )
            assert torch.equal(g_mu, c_mu), f"seed={seed}: mu differs"
            assert torch.equal(g_sigma, c_sigma), f"seed={seed}: sigma differs"

    def test_belief_gradients_byte_identical(self, golden_kernels):
        for seed in range(5):
            mu, sigma, mu_p, sigma_p, omega, log_prior, _ = _diag_setup(seed=seed)
            g_gmu, g_gsig = golden_kernels.belief_gradients(
                mu, sigma, mu_p, sigma_p, omega, tau=1.3, log_prior=log_prior,
            )
            c_gmu, c_gsig = belief_gradients(
                mu, sigma, mu_p, sigma_p, omega, tau=1.3, log_prior=log_prior,
            )
            assert torch.equal(g_gmu, c_gmu), f"seed={seed}: grad_mu differs"
            assert torch.equal(g_gsig, c_gsig), f"seed={seed}: grad_sigma differs"

    def test_uses_kernel_route_agrees_with_golden_on_every_structural_combo(self, golden_kernels):
        # golden uses_kernel_route(family=...) has no route= kwarg (route="gradient" is the only
        # meaning it had); the current predicate must agree with it EXACTLY on every combo the old
        # signature could express.
        for family in ("gaussian_diagonal", "gaussian_full", "laplace_diagonal"):
            for renyi_order in (1.0, 0.5, 2.0):
                for divergence_family in ("renyi", "jeffreys"):
                    for entropy in (True, False):
                        kwargs = dict(
                            renyi_order=renyi_order, gradient_mode="filtering", family=family,
                            divergence_family=divergence_family, include_attention_entropy=entropy,
                        )
                        golden = golden_kernels.uses_kernel_route(**kwargs)
                        current = uses_kernel_route(**kwargs, route="gradient")
                        assert golden == current, kwargs


# --------------------------------------------------------------------------------------------
# Full-covariance setup helpers
# --------------------------------------------------------------------------------------------

def _spd(N, K, *, generator, dtype, scale=1.0, floor=1.0):
    A = scale * torch.randn(N, K, K, generator=generator, dtype=dtype)
    return A @ A.transpose(-1, -2) + floor * torch.eye(K, dtype=dtype)


def _full_setup(N=8, K=4, seed=0, dtype=torch.float64, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
    grp = dataclasses.replace(grp, generators=grp.generators.to(dtype))
    assert grp.irrep_dims == [K]                       # single block, the shape this fusion covers
    phi = (0.1 * torch.randn(1, N, grp.generators.shape[0], generator=g)).to(dtype)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g, dtype=dtype)
    sigma = _spd(N, K, generator=g, dtype=dtype)
    mu_p = torch.randn(N, K, generator=g, dtype=dtype)
    sigma_p = _spd(N, K, generator=g, dtype=dtype)
    if requires_grad:
        mu = mu.detach().clone().requires_grad_(True)
        sigma = sigma.detach().clone().requires_grad_(True)
    return mu, sigma, mu_p, sigma_p, omega, grp


# --------------------------------------------------------------------------------------------
# Test 2: config construction
# --------------------------------------------------------------------------------------------

class Test2_ConfigConstruction:


    def test_gaussian_full_gradient_mode_still_routes_to_oracle(self):
        # Regression: registering gaussian_full's mm_exact-only capability must NOT flip
        # e_step_update='gradient' onto a (nonexistent) analytic kernel.
        assert has_kernel("gaussian_full")
        registration = get_kernel_registration("gaussian_full")
        assert registration.provides_mm_exact is True
        assert registration.provides_gradient is False
        assert uses_kernel_route(
            renyi_order=1.0, gradient_mode="filtering", family="gaussian_full",
            divergence_family="renyi", include_attention_entropy=True, route="gradient",
        ) is False
        assert uses_kernel_route(
            renyi_order=1.0, gradient_mode="filtering", family="gaussian_full",
            divergence_family="renyi", include_attention_entropy=True, route="mm_exact",
        ) is True
        assert "gaussian_full" in mm_exact_families()

    def test_diagonal_still_provides_both_capabilities(self):
        registration = get_kernel_registration("gaussian_diagonal")
        assert registration.provides_gradient is True
        assert registration.provides_mm_exact is True


# --------------------------------------------------------------------------------------------
# Test 3: forward + backward through the real model
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Shared frozen-beta surrogate (F_hat) scoring, mirroring mm_exact_update's own preamble.
# --------------------------------------------------------------------------------------------

def _frozen_pieces(mu, sigma, mu_p, sigma_p, omega, *, tau=1.0, kl_max=100.0, eps=1e-6,
                    lambda_beta=1.0, b0=1.0, c0=1.0, lambda_alpha_mode="constant"):
    fam = get_family("gaussian_full")
    mu_t = transport_mean(omega, mu)
    sigma_t = transport_covariance(omega, sigma, diagonal_out=False)
    sd = self_divergence_for_alpha(
        fam(mu, sigma), fam(mu_p, sigma_p), alpha=1.0, kl_max=kl_max, eps=eps,
        divergence_family="renyi", lambda_alpha_mode=lambda_alpha_mode,
    )
    energy = pairwise_energy(
        fam(mu, sigma), fam.from_transported(mu_t, sigma_t, sigma),
        alpha=1.0, kl_max=kl_max, eps=eps, divergence_family="renyi",
    )
    beta = attention_weights(energy, tau=tau)
    pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    coef = alpha_gradient_coefficient(sd, value=1.0, b0=b0, c0=c0, mode=lambda_alpha_mode)
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)
    self_mask = (sd < kl_max).to(mu.dtype).unsqueeze(-1)
    a = self_mask * coef
    w = lambda_beta * (beta * pair_mask)
    return a, w, mu_t, sigma_t


def _f_hat(mu_q, sigma_q, mu_p, sigma_p, mu_t, sigma_t, a, w, *, eps=1e-6, kl_max=1.0e6):
    r"""The frozen-(a, w) surrogate mm_exact_update's full-covariance branch is the exact
    minimizer of: F_hat = sum_i a_i KL(q_i||p_i) + sum_ij w_ij KL(q_i||N(mu_t_ij, sigma_t_ij))."""
    fam = get_family("gaussian_full")
    self_div = fam(mu_q, sigma_q).renyi_closed_form(fam(mu_p, sigma_p), alpha=1.0, kl_max=kl_max, eps=eps)
    pair_div = pairwise_energy(
        fam(mu_q, sigma_q), fam.from_transported(mu_t, sigma_t, sigma_q),
        alpha=1.0, kl_max=kl_max, eps=eps, divergence_family="renyi",
    )
    return (a.squeeze(-1) * self_div).sum() + (w * pair_div).sum()


# --------------------------------------------------------------------------------------------
# Test 4: MM descent property
# --------------------------------------------------------------------------------------------

class Test4_MMDescent:
    @pytest.mark.parametrize("seed", range(6))
    def test_descent_at_eta_one(self, seed):
        mu, sigma, mu_p, sigma_p, omega, grp = _full_setup(seed=seed, dtype=torch.float64)
        a, w, mu_t, sigma_t = _frozen_pieces(mu, sigma, mu_p, sigma_p, omega)
        f0 = _f_hat(mu, sigma, mu_p, sigma_p, mu_t, sigma_t, a, w)
        mu_star, sigma_star = mm_exact_update(
            mu, sigma, mu_p, sigma_p, omega, family="gaussian_full", irrep_dims=grp.irrep_dims,
        )
        f1 = _f_hat(mu_star, sigma_star, mu_p, sigma_p, mu_t, sigma_t, a, w)
        assert torch.isfinite(f0) and torch.isfinite(f1)
        assert f1 <= f0 + 1e-9, f"seed={seed}: F_hat increased ({f1.item()} > {f0.item()})"

    @pytest.mark.parametrize("eta", [0.1, 0.3, 0.7, 1.0])
    def test_descent_under_damping(self, eta):
        # Convexity of F_hat in the natural parameters means ANY damped step toward the exact
        # minimizer is non-increasing (see the test module's derivation note in the PR report);
        # this checks it holds in practice, not just at eta=1.
        mu, sigma, mu_p, sigma_p, omega, grp = _full_setup(seed=3, dtype=torch.float64)
        a, w, mu_t, sigma_t = _frozen_pieces(mu, sigma, mu_p, sigma_p, omega)
        f0 = _f_hat(mu, sigma, mu_p, sigma_p, mu_t, sigma_t, a, w)
        mu_star, sigma_star = mm_exact_update(
            mu, sigma, mu_p, sigma_p, omega, family="gaussian_full", irrep_dims=grp.irrep_dims,
        )
        mu_eta, sigma_eta = mm_damped_precision_blend_full(
            mu, sigma, mu_star, sigma_star, eta, eps=1e-6, sigma_max=None,
        )
        f_eta = _f_hat(mu_eta, sigma_eta, mu_p, sigma_p, mu_t, sigma_t, a, w)
        assert f_eta <= f0 + 1e-8, f"eta={eta}: F_hat increased"


# --------------------------------------------------------------------------------------------
# Test 5: fixed-point agreement with the gradient path
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Test 6: diagonal consistency
# --------------------------------------------------------------------------------------------

class Test6_DiagonalConsistency:
    def test_full_covariance_kernel_matches_diagonal_kernel_on_a_diagonal_state(self):
        N, K = 7, 5
        dtype = torch.float64
        g = torch.Generator().manual_seed(11)
        # A DIAGONAL Omega_ij = diag(exp(phi_i - phi_j)): the special case where the (..., K, K)
        # dense sandwich the full-covariance route builds is EXACTLY diagonal (no truncation), so
        # the full-family route and the diagonal-family route score the identical KL and can be
        # compared bit-for-bit up to fp64 tolerance.
        phi = 0.3 * torch.randn(N, K, generator=g, dtype=dtype)
        omega = torch.zeros(N, N, K, K, dtype=dtype)
        for i in range(N):
            for j in range(N):
                omega[i, j] = torch.diag(torch.exp(phi[i] - phi[j]))

        mu = torch.randn(N, K, generator=g, dtype=dtype)
        sigma_diag = torch.rand(N, K, generator=g, dtype=dtype) + 0.5
        mu_p = torch.randn(N, K, generator=g, dtype=dtype)
        sigma_p_diag = torch.rand(N, K, generator=g, dtype=dtype) + 0.5
        sigma_full = torch.diag_embed(sigma_diag)
        sigma_p_full = torch.diag_embed(sigma_p_diag)

        mu_star_diag, sigma_star_diag = mm_exact_update(
            mu, sigma_diag, mu_p, sigma_p_diag, omega, family="gaussian_diagonal", irrep_dims=[K],
        )
        mu_star_full, sigma_star_full = mm_exact_update(
            mu, sigma_full, mu_p, sigma_p_full, omega, family="gaussian_full", irrep_dims=[K],
        )

        torch.testing.assert_close(mu_star_full, mu_star_diag, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(
            torch.diagonal(sigma_star_full, dim1=-2, dim2=-1), sigma_star_diag, atol=1e-8, rtol=1e-6,
        )
        off_diag = sigma_star_full - torch.diag_embed(torch.diagonal(sigma_star_full, dim1=-2, dim2=-1))
        assert off_diag.abs().max() < 1e-8, "fusion introduced spurious off-diagonal covariance"


# --------------------------------------------------------------------------------------------
# Test 7: mm_damping semantics
# --------------------------------------------------------------------------------------------

class Test7_MMDampingSemantics:
    def test_eta_one_is_the_exact_minimizer(self):
        mu, sigma, mu_p, sigma_p, omega, grp = _full_setup(seed=2, dtype=torch.float64)
        mu_star, sigma_star = mm_exact_update(
            mu, sigma, mu_p, sigma_p, omega, family="gaussian_full", irrep_dims=grp.irrep_dims,
        )
        mu_eta1, sigma_eta1 = mm_damped_precision_blend_full(
            mu, sigma, mu_star, sigma_star, 1.0, eps=1e-6, sigma_max=None,
        )
        torch.testing.assert_close(mu_eta1, mu_star, atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(sigma_eta1, sigma_star, atol=1e-9, rtol=1e-7)

    def test_eta_zero_is_no_movement(self):
        # Convention pinned by reading the diagonal path (e_step.py's mm_exact branch): eta=0 ->
        # lam_new = lam_old exactly -> mu, sigma pass through unchanged. mm_damped_precision_blend_full
        # is not itself range-restricted to (0, 1] (only the config/e_step_iteration callers are), so
        # eta=0.0 can be exercised directly here.
        mu, sigma, mu_p, sigma_p, omega, grp = _full_setup(seed=4, dtype=torch.float64)
        mu_star, sigma_star = mm_exact_update(
            mu, sigma, mu_p, sigma_p, omega, family="gaussian_full", irrep_dims=grp.irrep_dims,
        )
        mu_eta0, sigma_eta0 = mm_damped_precision_blend_full(
            mu, sigma, mu_star, sigma_star, 0.0, eps=1e-6, sigma_max=None,
        )
        torch.testing.assert_close(mu_eta0, mu, atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(sigma_eta0, sigma, atol=1e-9, rtol=1e-7)


# --------------------------------------------------------------------------------------------
# Multi-head prior-structure guard. The per-head fusion is exact precisely BECAUSE the self/prior
# term's Sigma_p^{-1} is block-diagonal; a genuinely dense prior covariance (one carrying
# off-diagonal structure within a block) breaks the decomposition, so the kernel must reject it on
# an EXECUTED check rather than on the group name.
# --------------------------------------------------------------------------------------------

def _multi_head_fixture(dtype=torch.float32):
    N, K, H = 4, 4, 2
    grp = get_group("block_glk")(K, n_heads=H)
    assert len(grp.irrep_dims) == H
    g = torch.Generator().manual_seed(0)
    phi = 0.1 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g)
    sigma = _spd(N, K, generator=g, dtype=dtype)
    mu_p = torch.randn(N, K, generator=g)
    return grp, omega, mu, sigma, mu_p, g, N, K


def test_multi_head_dense_prior_raises_at_the_kernel_call_site():
    """A dense (non-block-diagonal) sigma_p must be refused, loudly, at the kernel call site."""
    grp, omega, mu, sigma, mu_p, g, N, K = _multi_head_fixture()
    sigma_p = _spd(N, K, generator=g, dtype=torch.float32)   # dense: off-block entries nonzero
    with pytest.raises(ValueError, match="BLOCK-DIAGONAL"):
        mm_exact_update(mu, sigma, mu_p, sigma_p, omega,
                        family="gaussian_full", irrep_dims=grp.irrep_dims)


def test_multi_head_block_diagonal_prior_is_accepted():
    """The live encode prior is torch.diag_embed(...) -- diagonal, hence block-diagonal. It must run.

    This is the 2026-08-07 gauge-audit reversal: cross-head Sigma entries were measured at exactly
    0.0 through 8 E-steps and 4 layers, so `KL = sum_h KL_h` holds and the fusion decomposes.
    """
    grp, omega, mu, sigma, mu_p, g, N, K = _multi_head_fixture()
    sigma_p = torch.diag_embed(torch.rand(N, K, generator=g) + 0.5)   # exactly how the live prior is built
    mu_new, sigma_new = mm_exact_update(mu, sigma, mu_p, sigma_p, omega,
                                        family="gaussian_full", irrep_dims=grp.irrep_dims)
    assert mu_new.shape == mu.shape and sigma_new.shape == sigma.shape
    assert torch.isfinite(mu_new).all() and torch.isfinite(sigma_new).all()
    # SPD, and block-diagonal structure preserved on the output.
    evals = torch.linalg.eigvalsh(0.5 * (sigma_new + sigma_new.transpose(-1, -2)))
    assert (evals > 0).all(), f"mm_exact returned a non-SPD covariance: min eig {evals.min()}"
    d = K // len(grp.irrep_dims)
    off = sigma_new[..., :d, d:].abs().max()
    assert off <= 1e-5 * sigma_new.abs().max(), f"cross-block leakage {off}"
