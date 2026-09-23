import pytest
import torch

from vfe3.divergence import safe_kl_clamp


def test_safe_kl_clamp_bounds_and_nan(device: torch.device) -> None:
    x = torch.tensor(
        [-1.0, 0.5, 1e9, float("nan"), float("inf"), float("-inf")]
    ).to(device)
    out = safe_kl_clamp(x, kl_max=100.0)
    assert torch.equal(
        out, torch.tensor([0.0, 0.5, 100.0, 100.0, 100.0, 0.0]).to(device)
    )


def test_divergence_delegates_to_families():
    """renyi(...) must route through the families layer (DiagonalGaussian closed form)."""
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import renyi as fam_renyi

    g = torch.Generator().manual_seed(21)
    mu_q = torch.randn(3, 2, generator=g)
    mu_p = torch.randn(3, 2, generator=g)
    s_q = torch.rand(3, 2, generator=g) + 0.5
    s_p = torch.rand(3, 2, generator=g) + 0.5
    got = renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=0.5)
    want = fam_renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=0.5)
    assert torch.allclose(got, want, atol=0.0)
    assert renyi is fam_renyi                       # divergence.renyi IS the param functional


def test_kl_equals_renyi_at_alpha_one():
    from vfe3.divergence import kl, renyi
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(7)
    mu_q = torch.randn(3, 5, generator=g)
    mu_t = torch.randn(3, 5, generator=g)
    sigma_q = torch.rand(3, 5, generator=g) + 0.1
    sigma_t = torch.rand(3, 5, generator=g) + 0.1
    q, p = DiagonalGaussian(mu_q, sigma_q), DiagonalGaussian(mu_t, sigma_t)
    a = kl(q, p)
    b = renyi(q, p, alpha=1.0)
    assert torch.allclose(a, b)


def test_renyi_dispatches_on_family():
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import FullGaussian
    g = torch.Generator().manual_seed(8)
    mu = torch.randn(2, 4, generator=g)
    A = torch.randn(2, 4, 4, generator=g)
    sigma_full = A @ A.transpose(-1, -2) + torch.eye(4)
    out = renyi(FullGaussian(mu, sigma_full), FullGaussian(mu, sigma_full), alpha=1.0)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros(2), atol=1e-4)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_self_divergence_is_zero(family):
    from vfe3.divergence import get_family, kl
    g = torch.Generator().manual_seed(11)
    mu = torch.randn(4, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(4, 6, generator=g) + 0.1
    else:
        A = torch.randn(4, 6, 6, generator=g)
        sigma = A @ A.transpose(-1, -2) + torch.eye(6)
    fam = get_family(family)
    out = kl(fam(mu, sigma), fam(mu, sigma))
    assert torch.allclose(out, torch.zeros(4), atol=1e-4)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_divergence_nonnegative(family):
    from vfe3.divergence import get_family, kl
    g = torch.Generator().manual_seed(12)
    mu_q = torch.randn(8, 6, generator=g)
    mu_t = torch.randn(8, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma_q = torch.rand(8, 6, generator=g) + 0.1
        sigma_t = torch.rand(8, 6, generator=g) + 0.1
    else:
        Aq = torch.randn(8, 6, 6, generator=g)
        At = torch.randn(8, 6, 6, generator=g)
        sigma_q = Aq @ Aq.transpose(-1, -2) + torch.eye(6)
        sigma_t = At @ At.transpose(-1, -2) + torch.eye(6)
    fam = get_family(family)
    out = kl(fam(mu_q, sigma_q), fam(mu_t, sigma_t))
    assert (out >= 0.0).all()


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_alpha_gt_one_warns(family):
    # Equal q/t covariance makes the blend (1-a)S + aS == S, guaranteed SPD
    # for any alpha, so the warning fires without a Cholesky failure.
    from vfe3.divergence import get_family, renyi
    g = torch.Generator().manual_seed(13)
    mu_q = torch.randn(3, 4, generator=g)
    mu_t = torch.randn(3, 4, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(3, 4, generator=g) + 0.5
    else:
        A = torch.randn(3, 4, 4, generator=g)
        sigma = A @ A.transpose(-1, -2) + torch.eye(4)
    fam = get_family(family)
    with pytest.warns(RuntimeWarning, match=r"alpha=1.5 > 1"):
        renyi(fam(mu_q, sigma), fam(mu_t, sigma), alpha=1.5)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_alpha_le_one_does_not_warn(alpha, family):
    import warnings

    from vfe3.divergence import get_family, renyi
    g = torch.Generator().manual_seed(14)
    mu_q = torch.randn(3, 4, generator=g)
    mu_t = torch.randn(3, 4, generator=g)
    if family == "gaussian_diagonal":
        sigma_q = torch.rand(3, 4, generator=g) + 0.1
        sigma_t = torch.rand(3, 4, generator=g) + 0.1
    else:
        Aq = torch.randn(3, 4, 4, generator=g)
        At = torch.randn(3, 4, 4, generator=g)
        sigma_q = Aq @ Aq.transpose(-1, -2) + torch.eye(4)
        sigma_t = At @ At.transpose(-1, -2) + torch.eye(4)
    fam = get_family(family)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        renyi(fam(mu_q, sigma_q), fam(mu_t, sigma_t), alpha=alpha)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


@pytest.mark.parametrize("bad_alpha", [0.0, -1.0])
def test_alpha_nonpositive_raises(bad_alpha):
    from vfe3.divergence import renyi
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(15)
    mu = torch.randn(2, 4, generator=g)
    sigma = torch.rand(2, 4, generator=g) + 0.1
    with pytest.raises(ValueError):
        renyi(DiagonalGaussian(mu, sigma), DiagonalGaussian(mu, sigma), alpha=bad_alpha)


# ---------------------------------------------------------------------------
# Full-covariance Renyi at alpha > 1 must be NON-RAISING and per-element robust.
# At alpha > 1 the blend (1-alpha)Sigma_q + alpha*Sigma_t is NOT convex and can be
# indefinite for some (i,j) pairs; a single non-PD blend used to make the WHOLE
# batched torch.linalg.cholesky RAISE (LinAlgError). The hardened path uses
# cholesky_ex + an info-driven mask so a bad pair maps to kl_max (via safe_kl_clamp)
# while good pairs in the SAME batch keep their finite divergence.
# ---------------------------------------------------------------------------


def test_full_renyi_alpha_gt_one_mixed_batch_no_raise():
    r"""Mixed batch at alpha=1.5: one indefinite-blend pair, one fine pair.

    Desired (pinned) behavior: NO exception; the bad pair == kl_max; the good pair
    finite and equal to its value computed in isolation. (RED against the old code,
    which raised LinAlgError for the whole batched call.)"""
    import warnings

    from vfe3.divergence import renyi
    from vfe3.families.gaussian import FullGaussian

    K = 3
    eye = torch.eye(K)
    # pair 0 (GOOD): well-separated SPD covariances; blend stays PD at alpha=1.5.
    sig_q_good = eye.clone()
    sig_t_good = 2.0 * eye
    # pair 1 (BAD): Sigma_q huge, Sigma_t tiny -> (1-1.5)Sigma_q + 1.5 Sigma_t indefinite.
    sig_q_bad = 100.0 * eye
    sig_t_bad = 0.01 * eye
    sigma_q = torch.stack([sig_q_good, sig_q_bad])
    sigma_t = torch.stack([sig_t_good, sig_t_bad])
    mu = torch.zeros(2, K)
    q = FullGaussian(mu, sigma_q)
    t = FullGaussian(mu, sigma_t)

    kl_max = 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # alpha>1 warning is expected
        out = renyi(q, t, alpha=1.5, kl_max=kl_max)         # must NOT raise
        # good pair computed in ISOLATION (single-element batch) must match in-batch value
        good_alone = renyi(
            FullGaussian(mu[:1], sig_q_good.unsqueeze(0)),
            FullGaussian(mu[:1], sig_t_good.unsqueeze(0)),
            alpha=1.5,
            kl_max=kl_max,
        )

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], good_alone[0], atol=1e-6)  # good survives, unperturbed
    assert out[0] < kl_max                                   # good is a genuine finite divergence
    assert out[1] == kl_max                                  # bad masked to kl_max


def test_full_renyi_alpha_gt_one_mixed_batch_custom_kl_max():
    r"""Same mixed batch, but a non-default kl_max: bad pair maps to the PASSED kl_max."""
    import warnings

    from vfe3.divergence import renyi
    from vfe3.families.gaussian import FullGaussian

    K = 3
    eye = torch.eye(K)
    sigma_q = torch.stack([eye.clone(), 100.0 * eye])
    sigma_t = torch.stack([2.0 * eye, 0.01 * eye])
    mu = torch.zeros(2, K)
    q = FullGaussian(mu, sigma_q)
    t = FullGaussian(mu, sigma_t)

    kl_max = 37.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = renyi(q, t, alpha=1.5, kl_max=kl_max)

    assert torch.isfinite(out).all()
    assert out[1] == kl_max
    assert out[0] < kl_max


def test_full_renyi_alpha_gt_one_all_good_batch_finite():
    r"""All-good batch at alpha>1 (equal Sigma_q==Sigma_t so blend==Sigma, PD for any
    alpha): every pair is finite and below kl_max -- no spurious masking."""
    import warnings

    from vfe3.divergence import renyi
    from vfe3.families.gaussian import FullGaussian

    g = torch.Generator().manual_seed(31)
    K = 4
    mu_q = torch.randn(5, K, generator=g)
    mu_t = torch.randn(5, K, generator=g)
    A = torch.randn(5, K, K, generator=g)
    sigma = A @ A.transpose(-1, -2) + torch.eye(K)
    q = FullGaussian(mu_q, sigma)
    t = FullGaussian(mu_t, sigma)                            # same Sigma -> blend == Sigma (PD)

    kl_max = 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = renyi(q, t, alpha=1.5, kl_max=kl_max)

    assert torch.isfinite(out).all()
    assert (out < kl_max).all()                             # no spurious masking
    assert (out >= 0.0).all()


# ---------------------------------------------------------------------------
# DIAGONAL Renyi at alpha > 1 must mirror the full-cov mask: a coordinate whose
# blend (1-alpha)s_q + alpha*s_t goes NON-POSITIVE makes the divergence undefined
# and must map to kl_max (via NaN -> safe_kl_clamp), NOT be silently clamp(min=eps)'d
# to a wrong finite value. The convex alpha in (0,1) blend is always > 0, so this is
# inert there (byte-identical). RED against the clamp(min=eps) code.
# ---------------------------------------------------------------------------


def test_diagonal_renyi_alpha_gt_one_negative_blend_masks_to_kl_max():
    r"""Summed diagonal Renyi at alpha=1.5: a pair with one negative-blend coordinate
    maps to kl_max; a good pair stays finite and equal to its isolated value."""
    import warnings

    from vfe3.families.gaussian import DiagonalGaussian

    K = 3
    mu = torch.zeros(2, K)
    # pair 0 (GOOD): blend = -0.5*s_q + 1.5*s_t stays positive on every coordinate.
    sig_q_good = torch.tensor([1.0, 1.0, 1.0])
    sig_t_good = torch.tensor([2.0, 2.0, 2.0])
    # pair 1 (BAD): coord 1 has s_q huge, s_t tiny -> blend = -50 + 0.015 < 0 at alpha=1.5.
    sig_q_bad = torch.tensor([1.0, 100.0, 1.0])
    sig_t_bad = torch.tensor([2.0, 0.01, 1.0])
    q = DiagonalGaussian(mu, torch.stack([sig_q_good, sig_q_bad]))
    t = DiagonalGaussian(mu, torch.stack([sig_t_good, sig_t_bad]))

    kl_max = 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # alpha>1 warning not raised here (closed form)
        out = q.renyi_closed_form(t, alpha=1.5, kl_max=kl_max)
        good_alone = DiagonalGaussian(mu[:1], sig_q_good.unsqueeze(0)).renyi_closed_form(
            DiagonalGaussian(mu[:1], sig_t_good.unsqueeze(0)), alpha=1.5, kl_max=kl_max)

    assert torch.isfinite(out).all()
    assert out[1] == kl_max                                  # bad pair masked
    assert out[0] < kl_max                                   # good pair finite
    assert torch.allclose(out[0], good_alone[0], atol=1e-6)  # good unperturbed by the bad pair


def test_diagonal_renyi_per_coord_alpha_gt_one_masks_only_bad_coord():
    r"""Per-coordinate diagonal Renyi at alpha=1.5: only the coordinate whose blend goes
    non-positive maps to kl_max; the other coordinates keep their finite divergence."""
    import warnings

    from vfe3.families.gaussian import DiagonalGaussian

    mu = torch.zeros(1, 3)
    sigma_q = torch.tensor([[1.0, 100.0, 2.0]])             # coord 1 blend goes negative
    sigma_t = torch.tensor([[2.0, 0.01, 1.0]])
    q = DiagonalGaussian(mu, sigma_q)
    t = DiagonalGaussian(mu, sigma_t)

    kl_max = 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = q.renyi_per_coord(t, alpha=1.5, kl_max=kl_max)

    assert torch.isfinite(out).all()
    assert out[0, 1] == kl_max                               # coord 1: blend < 0 -> masked
    assert out[0, 0] < kl_max                                # coord 0: blend = 2.5 > 0 -> finite
    assert out[0, 2] < kl_max                                # coord 2: blend = 0.5 > 0 -> finite


def test_diagonal_kl_matches_torch_distributions():
    # Independent reference: PyTorch's own Normal KL (summed over dims).
    from torch.distributions import Normal, kl_divergence
    from vfe3.divergence import kl
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(20)
    mu_q = torch.randn(4, 5, generator=g)
    mu_t = torch.randn(4, 5, generator=g)
    var_q = torch.rand(4, 5, generator=g) + 0.2
    var_t = torch.rand(4, 5, generator=g) + 0.2
    ref = kl_divergence(Normal(mu_q, var_q.sqrt()), Normal(mu_t, var_t.sqrt())).sum(-1)
    got = kl(DiagonalGaussian(mu_q, var_q), DiagonalGaussian(mu_t, var_t))
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_full_kl_matches_torch_distributions():
    # Independent reference: PyTorch's own MultivariateNormal KL.
    from torch.distributions import MultivariateNormal, kl_divergence
    from vfe3.divergence import kl
    from vfe3.families.gaussian import FullGaussian
    g = torch.Generator().manual_seed(21)
    K = 4
    mu_q = torch.randn(3, K, generator=g)
    mu_t = torch.randn(3, K, generator=g)
    Aq = torch.randn(3, K, K, generator=g)
    At = torch.randn(3, K, K, generator=g)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(K)
    S_t = At @ At.transpose(-1, -2) + torch.eye(K)
    ref = kl_divergence(MultivariateNormal(mu_q, S_q), MultivariateNormal(mu_t, S_t))
    got = kl(FullGaussian(mu_q, S_q), FullGaussian(mu_t, S_t), eps=0.0)
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)


def test_diagonal_kl_closed_form_1d():
    # 1-D Gaussian KL: 0.5 (v1/v2 + (m2-m1)^2/v2 - 1 + ln(v2/v1)).
    from vfe3.divergence import kl
    from vfe3.families.gaussian import DiagonalGaussian
    mu_q = torch.tensor([[0.5]])
    mu_t = torch.tensor([[-1.0]])
    v_q = torch.tensor([[2.0]])
    v_t = torch.tensor([[0.5]])
    expected = 0.5 * (2.0 / 0.5 + (1.5 ** 2) / 0.5 - 1.0 + torch.log(torch.tensor(0.5 / 2.0)))
    got = kl(DiagonalGaussian(mu_q, v_q), DiagonalGaussian(mu_t, v_t))
    assert torch.allclose(got, expected.reshape(1), atol=1e-5)


def test_divergence_reexports_family_cov_kind():
    """A family declares its covariance structure at registration; consumers read the declared
    kind via family_cov_kind (re-exported from vfe3.divergence) rather than sniffing the name."""
    from vfe3.divergence import family_cov_kind

    assert family_cov_kind("gaussian_diagonal") == "diagonal"
    assert family_cov_kind("gaussian_full") == "full"


def test_family_cov_kind_unregistered_raises():
    from vfe3.divergence import family_cov_kind

    with pytest.raises(KeyError):
        family_cov_kind("no_such_family")


# ---------------------------------------------------------------------------
# squared-Hellinger f-divergence (second functional-registry member)
#
# Math: H^2(q,p) = 1 - BC(q,p), BC = exp(-D_{1/2}(q||p)/2), where D_{1/2} is the
# Renyi-1/2 divergence the code already computes. The independent oracles below
# (analytic diagonal/full Bhattacharyya, symmetry, self-zero, bounds) do NOT
# re-assert that definition -- they compute H^2 by a different route.
# ---------------------------------------------------------------------------


def _diag_hellinger_analytic(mu_q, sigma_q, mu_p, sigma_p):
    r"""Independent analytic diagonal-Gaussian H^2 in float64.

    BC = prod_k sqrt( 2 sqrt(s_q^k s_p^k) / (s_q^k + s_p^k) )
              * exp( -(mu_q^k - mu_p^k)^2 / (4 (s_q^k + s_p^k)) );  H^2 = 1 - BC.
    Bhattacharyya factorizes over coordinates, so BC is a PRODUCT over k and H^2
    is 1 - that product (NOT a per-coordinate H^2 summed)."""
    mu_q, sigma_q = mu_q.double(), sigma_q.double()
    mu_p, sigma_p = mu_p.double(), sigma_p.double()
    s_sum = sigma_q + sigma_p
    bc_k = torch.sqrt(2.0 * torch.sqrt(sigma_q * sigma_p) / s_sum) \
        * torch.exp(-((mu_q - mu_p) ** 2) / (4.0 * s_sum))
    return 1.0 - bc_k.prod(dim=-1)


def _full_hellinger_analytic(mu_q, sigma_q, mu_p, sigma_p):
    r"""Independent analytic full-covariance H^2 in float64 via the Bhattacharyya
    distance D_B = 1/8 dmu^T Sbar^{-1} dmu + 1/2 ( ln|Sbar| - 1/2 ln|S_q| - 1/2 ln|S_p| ),
    Sbar = (S_q + S_p)/2; BC = exp(-D_B), H^2 = 1 - BC. Uses slogdet/solve -- a
    different numerical path than the kernel's Cholesky-of-blend."""
    mu_q, sigma_q = mu_q.double(), sigma_q.double()
    mu_p, sigma_p = mu_p.double(), sigma_p.double()
    s_bar = 0.5 * (sigma_q + sigma_p)
    dmu = (mu_p - mu_q).unsqueeze(-1)
    quad = (dmu.transpose(-1, -2) @ torch.linalg.solve(s_bar, dmu)).squeeze(-1).squeeze(-1)
    ld_bar = torch.linalg.slogdet(s_bar)[1]
    ld_q = torch.linalg.slogdet(sigma_q)[1]
    ld_p = torch.linalg.slogdet(sigma_p)[1]
    d_b = 0.125 * quad + 0.5 * (ld_bar - 0.5 * (ld_q + ld_p))
    return 1.0 - torch.exp(-d_b)


def test_squared_hellinger_diagonal_matches_analytic():
    from vfe3.divergence import get_functional
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(101)
    mu_q = torch.randn(7, 5, generator=g)
    mu_p = torch.randn(7, 5, generator=g)
    s_q = torch.rand(7, 5, generator=g) + 0.5
    s_p = torch.rand(7, 5, generator=g) + 0.5
    h2 = get_functional("squared_hellinger")(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p))
    ref = _diag_hellinger_analytic(mu_q, s_q, mu_p, s_p)
    assert torch.allclose(h2.double(), ref, atol=1e-5), (h2.double() - ref).abs().max()


def test_squared_hellinger_full_matches_analytic():
    from vfe3.divergence import get_functional
    from vfe3.families.gaussian import FullGaussian
    g = torch.Generator().manual_seed(102)
    K = 4
    mu_q = torch.randn(3, K, generator=g)
    mu_p = torch.randn(3, K, generator=g)
    Aq = torch.randn(3, K, K, generator=g)
    Ap = torch.randn(3, K, K, generator=g)
    S_q = Aq @ Aq.transpose(-1, -2) + K * torch.eye(K)
    S_p = Ap @ Ap.transpose(-1, -2) + K * torch.eye(K)
    h2 = get_functional("squared_hellinger")(FullGaussian(mu_q, S_q), FullGaussian(mu_p, S_p), eps=0.0)
    ref = _full_hellinger_analytic(mu_q, S_q, mu_p, S_p)
    assert torch.allclose(h2.double(), ref, atol=1e-5), (h2.double() - ref).abs().max()


def test_squared_hellinger_equals_definitional_identity():
    """H^2 = 1 - exp(-D_{1/2}/2) (the spec sympy-verified Gaussian identity)."""
    from vfe3.divergence import get_functional, renyi
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(103)
    mu_q = torch.randn(6, 4, generator=g)
    mu_p = torch.randn(6, 4, generator=g)
    s_q = torch.rand(6, 4, generator=g) + 0.5
    s_p = torch.rand(6, 4, generator=g) + 0.5
    q, p = DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p)
    h2 = get_functional("squared_hellinger")(q, p)
    d_half = renyi(q, p, alpha=0.5)
    assert torch.allclose(h2, 1.0 - torch.exp(-0.5 * d_half), atol=1e-6)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_squared_hellinger_is_symmetric(family):
    """Strong INDEPENDENT check: Hellinger is symmetric, unlike KL / Renyi at alpha != 1/2."""
    from vfe3.divergence import get_family, get_functional
    g = torch.Generator().manual_seed(104)
    mu_q = torch.randn(5, 4, generator=g)
    mu_p = torch.randn(5, 4, generator=g)
    if family == "gaussian_diagonal":
        s_q = torch.rand(5, 4, generator=g) + 0.5
        s_p = torch.rand(5, 4, generator=g) + 0.5
    else:
        Aq = torch.randn(5, 4, 4, generator=g)
        Ap = torch.randn(5, 4, 4, generator=g)
        s_q = Aq @ Aq.transpose(-1, -2) + 4 * torch.eye(4)
        s_p = Ap @ Ap.transpose(-1, -2) + 4 * torch.eye(4)
    fam = get_family(family)
    fn = get_functional("squared_hellinger")
    qp = fn(fam(mu_q, s_q), fam(mu_p, s_p))
    pq = fn(fam(mu_p, s_p), fam(mu_q, s_q))
    assert torch.allclose(qp, pq, atol=1e-5), (qp - pq).abs().max()


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_squared_hellinger_self_is_zero(family):
    from vfe3.divergence import get_family, get_functional
    g = torch.Generator().manual_seed(105)
    mu = torch.randn(4, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(4, 6, generator=g) + 0.1
    else:
        A = torch.randn(4, 6, 6, generator=g)
        sigma = A @ A.transpose(-1, -2) + 6 * torch.eye(6)
    fam = get_family(family)
    out = get_functional("squared_hellinger")(fam(mu, sigma), fam(mu, sigma))
    assert torch.allclose(out, torch.zeros(4), atol=1e-5)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_squared_hellinger_bounded(family):
    from vfe3.divergence import get_family, get_functional
    g = torch.Generator().manual_seed(106)
    mu_q = torch.randn(8, 6, generator=g)
    mu_t = torch.randn(8, 6, generator=g)
    if family == "gaussian_diagonal":
        s_q = torch.rand(8, 6, generator=g) + 0.1
        s_t = torch.rand(8, 6, generator=g) + 0.1
    else:
        Aq = torch.randn(8, 6, 6, generator=g)
        At = torch.randn(8, 6, 6, generator=g)
        s_q = Aq @ Aq.transpose(-1, -2) + torch.eye(6)
        s_t = At @ At.transpose(-1, -2) + torch.eye(6)
    fam = get_family(family)
    out = get_functional("squared_hellinger")(fam(mu_q, s_q), fam(mu_t, s_t))
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_squared_hellinger_ignores_alpha_and_does_not_warn():
    """H^2 forwards alpha=0.5 internally regardless of any `alpha` the call sites pass;
    a passed alpha is absorbed by **kwargs and never reaches renyi (so the alpha>1 blend
    warning never fires). The two strongest checks that alpha is truly ignored."""
    import warnings
    from vfe3.divergence import get_functional
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(107)
    mu_q = torch.randn(4, 3, generator=g)
    mu_p = torch.randn(4, 3, generator=g)
    s_q = torch.rand(4, 3, generator=g) + 0.5
    s_p = torch.rand(4, 3, generator=g) + 0.5
    q, p = DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p)
    fn = get_functional("squared_hellinger")
    base = fn(q, p)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        perturbed = fn(q, p, alpha=2.0)
    assert torch.allclose(base, perturbed, atol=0.0)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_divergence_functionals_registry_derived():
    """The functional registry exposes both members and the config helper reads from it."""
    from vfe3.divergence import divergence_functionals
    names = divergence_functionals()
    assert "renyi" in names and "squared_hellinger" in names
    assert names == tuple(sorted(names))


def test_config_accepts_squared_hellinger_and_rejects_unknown():
    from vfe3.config import VFE3Config
    VFE3Config(divergence_family="squared_hellinger")     # accepted
    VFE3Config(divergence_family="renyi")                 # still accepted
    with pytest.raises(ValueError):
        VFE3Config(divergence_family="no_such_functional")


def test_model_forward_under_squared_hellinger():
    """End-to-end: a VFEModel forward + finite loss with the new functional flowing through
    pairwise_energy / self_divergence."""
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0,
                     divergence_family="squared_hellinger")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    logits = model(tokens)
    assert logits.shape == (3, 5, 20)
    _, loss, _ = model(tokens, targets)
    assert loss.shape == () and torch.isfinite(loss)


# --- bhattacharyya and jeffreys (added 2026-06-13: extend the f-divergence registry) -----------
def test_bhattacharyya_equals_half_d_half():
    """D_B(q||p) = D_{1/2}(q||p)/2 = renyi(q,p,alpha=0.5)/2 (the Bhattacharyya-distance identity)."""
    from vfe3.divergence import get_functional, renyi
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(201)
    q = DiagonalGaussian(torch.randn(6, 4, generator=g), torch.rand(6, 4, generator=g) + 0.5)
    p = DiagonalGaussian(torch.randn(6, 4, generator=g), torch.rand(6, 4, generator=g) + 0.5)
    d_b = get_functional("bhattacharyya")(q, p)
    assert torch.allclose(d_b, 0.5 * renyi(q, p, alpha=0.5), atol=1e-6)


def test_jeffreys_equals_symmetrized_kl():
    """J(q||p) = KL(q||p) + KL(p||q) = renyi(q,p,1) + renyi(p,q,1)."""
    from vfe3.divergence import get_functional, renyi
    from vfe3.families.gaussian import DiagonalGaussian
    g = torch.Generator().manual_seed(202)
    q = DiagonalGaussian(torch.randn(6, 4, generator=g), torch.rand(6, 4, generator=g) + 0.5)
    p = DiagonalGaussian(torch.randn(6, 4, generator=g), torch.rand(6, 4, generator=g) + 0.5)
    j = get_functional("jeffreys")(q, p)
    assert torch.allclose(j, renyi(q, p, alpha=1.0) + renyi(p, q, alpha=1.0), atol=1e-6)


@pytest.mark.parametrize("name", ["bhattacharyya", "jeffreys"])
@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_new_divergence_symmetric_and_nonneg(name, family):
    """Both are SYMMETRIC (unlike KL) and non-negative across both Gaussian families."""
    from vfe3.divergence import get_family, get_functional
    g = torch.Generator().manual_seed(203)
    mu_q = torch.randn(5, 4, generator=g)
    mu_p = torch.randn(5, 4, generator=g)
    if family == "gaussian_diagonal":
        s_q = torch.rand(5, 4, generator=g) + 0.5
        s_p = torch.rand(5, 4, generator=g) + 0.5
    else:
        Aq = torch.randn(5, 4, 4, generator=g)
        Ap = torch.randn(5, 4, 4, generator=g)
        s_q = Aq @ Aq.transpose(-1, -2) + 4 * torch.eye(4)
        s_p = Ap @ Ap.transpose(-1, -2) + 4 * torch.eye(4)
    fam, fn = get_family(family), get_functional(name)
    qp = fn(fam(mu_q, s_q), fam(mu_p, s_p))
    pq = fn(fam(mu_p, s_p), fam(mu_q, s_q))
    assert torch.allclose(qp, pq, atol=1e-5), (qp - pq).abs().max()
    assert bool((qp >= 0).all())


@pytest.mark.parametrize("name", ["bhattacharyya", "jeffreys"])
@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_new_divergence_self_is_zero(name, family):
    from vfe3.divergence import get_family, get_functional
    g = torch.Generator().manual_seed(204)
    mu = torch.randn(4, 6, generator=g)
    if family == "gaussian_diagonal":
        sigma = torch.rand(4, 6, generator=g) + 0.1
    else:
        A = torch.randn(4, 6, 6, generator=g)
        sigma = A @ A.transpose(-1, -2) + 6 * torch.eye(6)
    fam = get_family(family)
    out = get_functional(name)(fam(mu, sigma), fam(mu, sigma))
    assert torch.allclose(out, torch.zeros(4), atol=1e-5)


@pytest.mark.parametrize("name", ["bhattacharyya", "jeffreys"])
def test_model_forward_under_new_divergence(name):
    """End-to-end: a VFEModel forward + finite loss with the new functional flowing through
    pairwise_energy / self_divergence (the autograd-oracle route, as for squared_hellinger)."""
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, divergence_family=name)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    assert model(tokens).shape == (3, 5, 20)
    _, loss, _ = model(tokens, targets)
    assert loss.shape == () and torch.isfinite(loss)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None   # belief/prior table must get gradient (t7: was a weak OR)
