import pytest
import torch
import vfe3.geometry.retraction as retraction_module

from vfe3.geometry.retraction import (
    get_retraction,
    natural_gradient,
    register_retraction,
    retract_spd_diagonal,
    retract_spd_full,
)


def test_sigma_max_caps_variance_consistently_across_diag_and_full():
    # sigma_max must denote ONE physical variance ceiling on both arms: the eigenvalues of a full
    # covariance ARE variances, so the full retraction must cap them at sigma_max -- NOT sigma_max**2,
    # which let the full family hold variances a factor sigma_max larger than the diagonal arm under
    # the same knob (the cov_kind-seam mismatch).
    sigma_max, K = 3.0, 4
    out_d = retract_spd_diagonal(torch.ones(2, K), torch.full((2, K), 1e3), sigma_max=sigma_max)
    assert out_d.max().item() <= sigma_max + 1e-5
    sig_f = torch.eye(K).expand(2, K, K).contiguous()
    huge  = torch.eye(K).expand(2, K, K).contiguous() * 1e3
    eig_affine = torch.linalg.eigvalsh(
        retract_spd_full(sig_f, huge, trust_region=0.0, sigma_max=sigma_max))
    assert eig_affine.max().item() <= sigma_max + 1e-4
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_public_spd_certificate_repairs_reconstruction_boundary(dtype):
    from vfe3.geometry import retraction as retraction_module

    boundary = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=dtype, requires_grad=True)
    repaired = retraction_module._certify_public_spd(
        boundary,
        eps=1e-6,
        sigma_max=1.0,
    )

    assert repaired.dtype == dtype
    assert torch.linalg.cholesky_ex(repaired.float(), check_errors=False).info.item() == 0
    assert torch.linalg.eigvalsh(repaired.float()).max().item() <= 1.0
    repaired.float().sum().backward()
    assert boundary.grad is not None
    assert torch.isfinite(boundary.grad).all()


def test_public_spd_repair_margin_remains_in_autograd_graph():
    from vfe3.geometry import retraction as retraction_module

    boundary = torch.diag(torch.tensor([-2.0, 0.0])).requires_grad_(True)
    repaired = retraction_module._certify_public_spd(
        boundary,
        eps=1.0,
        sigma_max=100.0,
    )
    repaired.trace().backward()

    assert boundary.grad is not None
    assert boundary.grad[0, 0].abs().item() > 0.0


def test_public_spd_certificate_rejects_interval_without_representable_interior():
    from vfe3.geometry import retraction as retraction_module

    boundary = torch.ones(1, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="no representable strict interior"):
        retraction_module._certify_public_spd(
            boundary,
            eps=1.001,
            sigma_max=1.002,
        )


def test_public_spd_fallback_bound_is_certifiable_after_endpoint_rounding():
    from vfe3.geometry import retraction as retraction_module

    eps = 1.00000001
    lower, upper = retraction_module._public_spd_bounds(torch.float32, eps, 2.00000001)

    assert torch.tensor(lower, dtype=torch.float32) - torch.tensor(
        eps, dtype=torch.float32,
    ) > 0
    assert torch.tensor(2.00000001, dtype=torch.float32) - torch.tensor(
        upper, dtype=torch.float32,
    ) > 0


def test_diagonal_retraction_positive_and_bounded():
    g = torch.Generator().manual_seed(0)
    sigma = torch.rand(4, 6, generator=g) + 0.1
    delta = 5.0 * torch.randn(4, 6, generator=g)
    out = retract_spd_diagonal(sigma, delta, sigma_max=5.0)
    assert (out >= 1e-6).all()
    assert (out <= 5.0 + 1e-6).all()


def test_full_retraction_stays_spd(device: torch.device) -> None:
    g = torch.Generator().manual_seed(1)
    A = torch.randn(3, 4, 4, generator=g)
    identity = torch.eye(4)
    sigma = (A @ A.transpose(-1, -2) + identity).to(device)
    D = torch.randn(3, 4, 4, generator=g)
    delta = (0.5 * (D + D.transpose(-1, -2))).to(device)
    out = retract_spd_full(sigma, delta)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-4)
    assert (torch.linalg.eigvalsh(out) > 0).all()


def test_full_retraction_identity_tangent_is_identity():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 4, 4, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    zero = torch.zeros(3, 4, 4)
    # sigma_max=1e9 keeps the (variance) safety ceiling from binding: this random sigma has
    # eigenvalues > the default sigma_max, and the centering axiom R(Sigma,0)=Sigma is the geodesic
    # exp-map property, tested WITHIN the variance box (the output clamp correctly caps an
    # out-of-box base point at sigma_max on BOTH the diagonal and full arms).
    out = retract_spd_full(sigma, zero, sigma_max=1e9)
    assert torch.allclose(out, sigma, atol=1e-3)


def test_natural_gradient_diagonal_formula():
    g = torch.Generator().manual_seed(3)
    sigma = torch.rand(4, 5, generator=g) + 0.1
    gmu = torch.randn(4, 5, generator=g)
    gsig = torch.randn(4, 5, generator=g)
    nmu, nsig = natural_gradient(gmu, gsig, sigma)
    assert torch.allclose(nmu, sigma * gmu, atol=1e-6)
    assert torch.allclose(nsig, 2.0 * sigma * sigma * gsig, atol=1e-6)


def test_full_retraction_K1_matches_diagonal_formula():
    # For K=1 the affine-invariant SPD exp map reduces to the diagonal rule
    # sigma_new = sigma * exp(tau * delta/sigma).
    sigma = torch.tensor([[[2.0]]])          # (1,1,1) as (B,K,K) with K=1
    delta = torch.tensor([[[0.6]]])
    out = retract_spd_full(sigma, delta, trust_region=0.0)   # disable TR for exact check
    expected = 2.0 * torch.exp(torch.tensor(0.6 / 2.0))
    assert torch.allclose(out.reshape(()), expected, atol=1e-4)


def test_full_retraction_float64_tiny_diagonal_tangent_matches_affine_exponential():
    sigma_diag = torch.tensor([1.25, 2.5, 4.0], dtype=torch.float64)
    delta_diag = torch.tensor([1e-8, -1e-8, 2e-8], dtype=torch.float64)
    sigma = torch.diag(sigma_diag).unsqueeze(0)
    delta = torch.diag(delta_diag).unsqueeze(0)

    out = retract_spd_full(sigma, delta, trust_region=0.0, sigma_max=None)
    expected = torch.diag(sigma_diag * torch.exp(delta_diag / sigma_diag)).unsqueeze(0)

    assert out.dtype == torch.float64
    torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("trust_region", [5.0, 0.0])
def test_full_retraction_uncapped_path_retains_spd_floor(trust_region: float) -> None:
    eps = 1e-6
    sigma = eps * torch.eye(2).unsqueeze(0)
    delta = -torch.eye(2).unsqueeze(0)

    out = retract_spd_full(
        sigma,
        delta,
        trust_region=trust_region,
        eps=eps,
        sigma_max=None,
    )

    assert (torch.linalg.eigvalsh(out) >= eps).all()


# --- register_retraction / get_retraction seam (roadmap item 4) ------------
def test_retraction_registry_round_trip():
    """register_retraction/get_retraction round-trip and the unknown-name KeyError."""
    sentinel = object()

    @register_retraction("_test_dummy_retraction")
    def _dummy(*args, **kwargs):
        return sentinel

    try:
        assert get_retraction("_test_dummy_retraction")() is sentinel
        with pytest.raises(KeyError):
            get_retraction("nope")
    finally:
        from vfe3.geometry.retraction import _RETRACTIONS
        _RETRACTIONS.pop("_test_dummy_retraction", None)


def test_spd_affine_is_registered():
    """The default affine-invariant SPD retraction is registered under 'spd_affine'."""
    assert callable(get_retraction("spd_affine"))


def test_log_euclidean_helpers_are_retired() -> None:
    assert not hasattr(retraction_module, "retract_log_euclidean")
    assert not hasattr(retraction_module, "retract_logeuclidean_full")


def test_spd_affine_bit_identical_diagonal():
    """spd_affine reproduces the bare retract_spd_diagonal call bit-for-bit (atol=0)."""
    g = torch.Generator().manual_seed(11)
    sigma = torch.rand(3, 4, 5, generator=g) + 0.1      # (B, N, K) diagonal
    nat_sigma = torch.randn(3, 4, 5, generator=g)
    step = 0.015
    legacy = retract_spd_diagonal(
        sigma, -step * nat_sigma, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    seam = get_retraction("spd_affine")(
        sigma, -step * nat_sigma, mean_ndim=3, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    assert torch.equal(seam, legacy)


def test_spd_affine_bit_identical_full():
    """spd_affine reproduces the bare retract_spd_full call bit-for-bit (atol=0)."""
    g = torch.Generator().manual_seed(12)
    A = torch.randn(3, 4, 5, 5, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(5)       # (B, N, K, K) full SPD
    D = torch.randn(3, 4, 5, 5, generator=g)
    nat_sigma = 0.5 * (D + D.transpose(-1, -2))
    step = 0.015
    legacy = retract_spd_full(
        sigma, -step * nat_sigma, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    seam = get_retraction("spd_affine")(
        sigma, -step * nat_sigma, mean_ndim=3, trust_region=5.0, eps=1e-6, sigma_max=5.0,
    )
    assert torch.equal(seam, legacy)


# --- log_euclidean SPD retraction variant (ambient tangent -> log chart) ---














def _f_uses_eigvecs(eighfn, A):
    """A scalar whose gradient genuinely depends on the EIGENVECTORS, so the off-diagonal
    F = 1/(lambda_i - lambda_j) gap term (and its SIGN) is actually exercised. Contracting sqrtA with
    a FIXED ASYMMETRIC weight G picks up the off-diagonal entries of sqrtA = V diag(sqrt w) V^T, which
    depend on V. (The earlier ``(sqrtA*sqrtA).sum()`` was secretly eigenvalue-only: it equals
    ||sqrtA||_F^2 = tr(A) = sum(w), so its gradient lives entirely in the eigenvalue path and the test
    passed for EITHER sign of F -- it could not catch a wrong-sign adjoint.)"""
    w, V = eighfn(A)
    sqrtA = (V * w.clamp(min=1e-6).sqrt().unsqueeze(-2)) @ V.transpose(-1, -2)   # V diag(sqrt w) V^T
    n = A.shape[-1]
    idx = torch.arange(n, dtype=A.dtype, device=A.device)
    G = (1.0 + idx).unsqueeze(-1) * (1.0 + 2.0 * idx).unsqueeze(-2)              # fixed, asymmetric
    return (sqrtA * G).sum() + (w * w).sum()


def test_eigh_damped_matches_stock_eigh_backward_on_separated_spectrum():
    """De-risking check against an INDEPENDENT oracle: with a tiny gap_eps and a well-separated
    spectrum (all gaps >> sqrt(gap_eps)), the damped-eigh backward must agree with stock
    torch.linalg.eigh's backward to tolerance -- this validates the adjoint formula itself
    (sign, symmetrization, the F-circ-(V^T gV) term), not just that it stopped NaN-ing."""
    from vfe3.geometry.retraction import _eigh_damped
    torch.manual_seed(0)
    A = torch.randn(3, 5, 5)
    A = A @ A.transpose(-1, -2) + torch.diag_embed(torch.arange(1.0, 6.0).expand(3, 5))
    A = 0.5 * (A + A.transpose(-1, -2))                     # well-separated SPD
    a1 = A.clone().requires_grad_(True)
    a2 = A.clone().requires_grad_(True)
    _f_uses_eigvecs(lambda X: torch.linalg.eigh(X), a1).backward()
    _f_uses_eigvecs(lambda X: _eigh_damped(X, 1e-12), a2).backward()
    assert torch.allclose(a1.grad, a2.grad, atol=1e-4, rtol=1e-3)


def test_eigh_damped_fd_gradient_on_separated_spectrum():
    """Finite-difference check on a well-separated spectrum with gap_eps small enough that the
    damping is negligible there (so FD does not pass trivially while masking a formula error)."""
    from vfe3.geometry.retraction import _eigh_damped
    torch.manual_seed(1)
    A = torch.randn(4, 4, dtype=torch.float64)
    A = A @ A.t() + torch.diag(torch.arange(1.0, 5.0, dtype=torch.float64))
    A = (0.5 * (A + A.t())).requires_grad_(True)
    torch.autograd.gradcheck(
        lambda X: _f_uses_eigvecs(lambda Y: _eigh_damped(Y, 1e-14), 0.5 * (X + X.transpose(-1, -2))),
        (A,), eps=1e-6, atol=1e-4, rtol=1e-3,
    )


def test_eigh_damped_finite_backward_at_isotropic():
    """At the fully-degenerate spectrum (Sigma = I) the stock eigh backward is all-NaN; the damped
    version stays finite (the gap term is Lorentzian-damped to 0, not 1/0)."""
    from vfe3.geometry.retraction import _eigh_damped
    A = torch.eye(4, requires_grad=True)
    _f_uses_eigvecs(lambda X: _eigh_damped(X, 1e-8), A).backward()
    assert torch.isfinite(A.grad).all()


def test_retract_spd_full_finite_backward_at_isotropic_init():
    """The default gaussian_full prior init is Sigma = I (fully degenerate spectrum); the unrolled
    E-step backward through retract_spd_full must NOT be NaN there (was 100% NaN before the fix)."""
    sigma = torch.eye(4).reshape(1, 4, 4).clone().requires_grad_(True)
    out = retract_spd_full(sigma, torch.zeros(1, 4, 4))
    out.sum().backward()
    assert torch.isfinite(sigma.grad).all()




def test_near_degenerate_exp_preserves_active_off_diagonal_derivative():
    """An fp32 output roundoff must not erase an active exponential Fréchet derivative."""
    from vfe3.geometry import retraction as retraction_module

    active = torch.diag(torch.tensor([0.0, 1.0e-8], dtype=torch.float32)).requires_grad_(True)
    cotangent = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    active_out = retraction_module._symmetric_spectral_map(
        active,
        "exp_bounded",
        lower=-50.0,
        upper=50.0,
    )
    active_grad = torch.autograd.grad((active_out * cotangent).sum(), active)[0]
    expected = torch.expm1(torch.tensor(1.0e-8, dtype=torch.float32)) / 1.0e-8

    assert torch.allclose(active_grad[0, 1], expected, rtol=1e-6, atol=1e-6)

    flat = torch.diag(torch.tensor([51.0, 52.0], dtype=torch.float32)).requires_grad_(True)
    flat_out = retraction_module._symmetric_spectral_map(
        flat,
        "exp_bounded",
        lower=-50.0,
        upper=50.0,
    )
    flat_grad = torch.autograd.grad((flat_out * cotangent).sum(), flat)[0]

    assert flat_grad[0, 1].item() == 0.0


def test_full_cov_e_step_isotropic_init_finite_backward():
    """End-to-end reachability: a full-cov E-step at the ISOTROPIC Sigma = I init (the real default
    prior init) gives finite grads on the DEFAULT 'unroll' estimator via spd_affine -> retract_spd_full."""
    from vfe3.belief import BeliefState
    from vfe3.geometry.groups import get_group
    from vfe3.inference.e_step import e_step_iteration
    torch.manual_seed(2)
    N, K = 4, 4
    group = get_group("glk")(K)
    mu = torch.randn(1, N, K, requires_grad=True)
    sigma = torch.eye(K).expand(1, N, K, K).clone().requires_grad_(True)   # isotropic == default init
    phi = torch.zeros(1, N, group.generators.shape[0])
    belief = BeliefState(mu=mu, sigma=sigma, phi=phi)
    mu_p = torch.randn(N, K)
    sigma_p = torch.eye(K).expand(N, K, K).clone()
    out = e_step_iteration(belief, mu_p, sigma_p, group,
                           e_q_mu_lr=0.1, e_q_sigma_lr=0.05, e_phi_lr=0.0,
                           family="gaussian_full")
    (out.mu.pow(2).sum() + out.sigma.pow(2).sum()).backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(sigma.grad).all()


def test_full_cov_model_first_backward_finite_at_default_init():
    """Model-level reachability of the eigh-NaN fix: a full-covariance VFEModel at its DEFAULT init
    (prior sigma = diag_embed(exp(0)) = I, the degenerate spectrum) must produce a finite FIRST
    backward on the default 'unroll' estimator. Pre-fix this NaN-poisoned prior_bank.sigma_log_embed
    on the very first step (the defender's end-to-end repro)."""
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, n_layers=1,
                     family="gaussian_full", decode_mode="full_chunked")
    model = VFEModel(cfg)
    x = torch.randint(0, 8, (2, 6))
    y = torch.randint(0, 8, (2, 6))
    _, loss, _ = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient on the full-cov first step"


# --- F2 (audit 2026-07-01): invalid sigma_max must raise, not corrupt Sigma ---
@pytest.mark.parametrize("bad_sigma_max", [1e-9, 1e-6, -1.0, float("nan")])
@pytest.mark.parametrize("arm", ["diagonal", "full"])
def test_retract_rejects_invalid_sigma_max(arm, bad_sigma_max):
    """A sub-eps / negative / NaN sigma_max used to flow straight into torch.clamp and silently
    yield sub-eps, negative, or NaN variances; the _check_sigma_max guard now raises ValueError."""
    K = 3
    with pytest.raises(ValueError):
        if arm == "diagonal":
            retract_spd_diagonal(torch.ones(2, K), torch.zeros(2, K), sigma_max=bad_sigma_max)
        else:
            retract_spd_full(torch.eye(K).expand(2, K, K).contiguous(),
                             torch.zeros(2, K, K), sigma_max=bad_sigma_max)


@pytest.mark.parametrize("sigma_max", [None, 10.0])
def test_retract_valid_sigma_max_unaffected(sigma_max):
    """sigma_max=None (pure path: eps floor only) and the default 10.0 stay accepted and return
    finite PD output -- the guard does not regress the pure/default path."""
    K = 3
    out_d = retract_spd_diagonal(torch.ones(2, K), 0.1 * torch.ones(2, K), sigma_max=sigma_max)
    assert torch.isfinite(out_d).all() and (out_d > 0).all()
    out_f = retract_spd_full(torch.eye(K).expand(2, K, K).contiguous(),
                             0.1 * torch.eye(K).expand(2, K, K).contiguous(), sigma_max=sigma_max)
    assert torch.isfinite(out_f).all()
    assert (torch.linalg.eigvalsh(out_f) > 0).all()




