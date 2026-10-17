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
    got = renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=1.0)
    want = fam_renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=1.0)
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
def test_kl_does_not_warn(family):
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
        renyi(fam(mu_q, sigma_q), fam(mu_t, sigma_t), alpha=1.0)
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


def test_divergence_functionals_registry_derived():
    """The functional registry exposes the KL member and the config helper reads from it."""
    from vfe3.divergence import divergence_functionals
    names = divergence_functionals()
    assert "renyi" in names
    assert names == tuple(sorted(names))
