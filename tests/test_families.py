import math

import pytest
import torch

from vfe3.families.base import BeliefParams
from vfe3.families.gaussian import FullGaussian


class _ExpFamily(BeliefParams):
    r"""Univariate exponential family, parameter lam > 0.

    p(x; lam) = lam exp(-lam x), natural eta = -lam, A(eta) = -log(-eta),
    E[T] = 1/lam. Defines ONLY natural/log_partition_at/expected_statistic (no
    moment closed form), so it drives the generic Bregman/Renyi-from-A path.
    Per the families convention, ``natural``/``expected_statistic`` carry a
    trailing coordinate axis of size 1 (so the inner-product .sum(dim=-1) is
    well-defined), while ``log_partition_at`` sums that coordinate axis.
    """

    cov_kind = "diagonal"

    def __init__(self, lam): self.lam = lam                          # (...,)
    def coordinate_dim(self): return 1
    def block(self, start, end): return _ExpFamily(self.lam)
    def broadcast_over_keys(self): return _ExpFamily(self.lam.unsqueeze(-1))
    def natural(self): return (-self.lam.unsqueeze(-1),)             # (..., 1)
    @classmethod
    def log_partition_at(cls, theta): return (-torch.log(-theta[0])).sum(dim=-1)
    def expected_statistic(self): return (1.0 / self.lam.unsqueeze(-1),)  # (..., 1)
    def entropy(self): return 1.0 - torch.log(self.lam)


def test_family_registry_register_get_and_cov_kind():
    from vfe3.families.base import (
        BeliefParams, register_family, get_family, family_cov_kind, divergence_families,
    )

    class _ToyParams(BeliefParams):
        cov_kind = "diagonal"
        def __init__(self, x): self.x = x
        def coordinate_dim(self): return self.x.shape[-1]
        def block(self, start, end): return _ToyParams(self.x[..., start:end])
        def broadcast_over_keys(self): return _ToyParams(self.x.unsqueeze(-2))
        def natural(self): return (self.x,)
        @classmethod
        def log_partition_at(cls, theta): return theta[0].sum(dim=-1)
        def entropy(self): return self.x.sum(dim=-1) * 0.0

    register_family("toy_reg_test")(_ToyParams)
    try:
        assert get_family("toy_reg_test") is _ToyParams
        assert family_cov_kind("toy_reg_test") == "diagonal"
        assert "toy_reg_test" in divergence_families()
    finally:
        from vfe3.families.base import _FAMILIES
        _FAMILIES.pop("toy_reg_test", None)


def test_family_cov_kind_unregistered_raises():
    from vfe3.families.base import family_cov_kind
    with pytest.raises(KeyError):
        family_cov_kind("no_such_family")


def test_generic_kl_from_A_matches_exponential_closed_form():
    from vfe3.families.base import kl
    l1 = torch.tensor([2.0, 0.5, 1.0])
    l2 = torch.tensor([1.0, 1.5, 1.0])
    got = kl(_ExpFamily(l1), _ExpFamily(l2))
    want = torch.log(l1 / l2) + l2 / l1 - 1.0
    assert torch.allclose(got, want, atol=1e-5), (got, want)


def test_generic_renyi_from_A_matches_exponential_closed_form():
    from vfe3.families.base import renyi
    l1 = torch.tensor([2.0, 0.5])
    l2 = torch.tensor([1.0, 1.5])
    for a in (0.3, 0.7):
        got = renyi(_ExpFamily(l1), _ExpFamily(l2), alpha=a)
        want = (-torch.log(a * l1 + (1.0 - a) * l2) + a * torch.log(l1)
                + (1.0 - a) * torch.log(l2)) / (a - 1.0)
        assert torch.allclose(got, want, atol=1e-5), (a, got, want)


def test_diagonal_gaussian_closed_form_matches_legacy_divergence():
    from vfe3.divergence import renyi as div_renyi             # now the param functional
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import renyi as fam_renyi
    assert div_renyi is fam_renyi                              # divergence re-exports the param renyi
    torch.manual_seed(3)
    mu_q, mu_p = torch.randn(5, 4), torch.randn(5, 4)
    s_q, s_p = torch.rand(5, 4) + 0.5, torch.rand(5, 4) + 0.5
    for a in (0.5, 1.0):
        want = DiagonalGaussian(mu_q, s_q).renyi_closed_form(DiagonalGaussian(mu_p, s_p), alpha=a)
        got = fam_renyi(DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p), alpha=a)
        assert torch.allclose(got, want, atol=1e-6), (a, (got - want).abs().max())


def test_diagonal_gaussian_generic_from_A_equals_closed_form():
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.families.base import _renyi_from_log_partition
    torch.manual_seed(4)
    mu_q, mu_p = torch.randn(6, 3), torch.randn(6, 3)
    s_q, s_p = torch.rand(6, 3) + 0.5, torch.rand(6, 3) + 0.5
    q, p = DiagonalGaussian(mu_q, s_q), DiagonalGaussian(mu_p, s_p)
    for a in (0.5, 1.0):
        closed = q.renyi_closed_form(p, alpha=a, kl_max=float("inf"), eps=1e-6)
        generic = _renyi_from_log_partition(q, p, alpha=a, kl_max=float("inf"), eps=1e-6)
        assert torch.allclose(closed, generic, atol=1e-4), (a, (closed - generic).abs().max())


def test_diagonal_block_and_broadcast():
    from vfe3.families.gaussian import DiagonalGaussian
    mu, s = torch.randn(2, 6), torch.rand(2, 6) + 0.5
    q = DiagonalGaussian(mu, s)
    qb = q.block(2, 4)
    assert torch.equal(qb.mu, mu[..., 2:4]) and torch.equal(qb.sigma, s[..., 2:4])
    qk = q.broadcast_over_keys()
    assert qk.mu.shape == (2, 1, 6) and qk.sigma.shape == (2, 1, 6)


def test_diagonal_stack_round_trips():
    # `stack` is the family-agnostic primitive pairwise_energy batches the per-head loop on:
    # it stacks the underlying tensors along a new axis and the result's block/mu/sigma stay
    # consistent with the part it was built from.
    from vfe3.families.gaussian import DiagonalGaussian
    torch.manual_seed(0)
    N, d = 3, 2
    parts = [DiagonalGaussian(torch.randn(N, 1, d), torch.rand(N, 1, d) + 0.5) for _ in range(4)]
    stacked = DiagonalGaussian.stack(parts, dim=0)
    assert stacked.mu.shape == (4, N, 1, d) and stacked.sigma.shape == (4, N, 1, d)
    for h, part in enumerate(parts):
        assert torch.equal(stacked.mu[h], part.mu) and torch.equal(stacked.sigma[h], part.sigma)
    # the stacked object is a DiagonalGaussian whose block slices the trailing coordinate axis
    blk = stacked.block(0, 1)
    assert torch.equal(blk.mu, stacked.mu[..., 0:1])


def test_full_gaussian_closed_form_matches_legacy_and_block():
    from vfe3.families.gaussian import FullGaussian
    from vfe3.families.base import renyi as fam_renyi
    torch.manual_seed(5)
    N, K = 4, 3
    mu_q, mu_p = torch.randn(N, K), torch.randn(N, K)
    Aq = torch.randn(N, K, K); s_q = Aq @ Aq.transpose(-1, -2) + K * torch.eye(K)
    Ap = torch.randn(N, K, K); s_p = Ap @ Ap.transpose(-1, -2) + K * torch.eye(K)
    for a in (0.5, 1.0):
        want = FullGaussian(mu_q, s_q).renyi_closed_form(FullGaussian(mu_p, s_p), alpha=a)
        got = fam_renyi(FullGaussian(mu_q, s_q), FullGaussian(mu_p, s_p), alpha=a)
        assert torch.allclose(got, want, atol=1e-4), (a, (got - want).abs().max())
    qb = FullGaussian(mu_q, s_q).block(1, 3)
    assert torch.equal(qb.mu, mu_q[..., 1:3]) and torch.equal(qb.sigma, s_q[..., 1:3, 1:3])


def test_full_stack_round_trips():
    # FullGaussian carries a matrix sigma (..., d, d); `stack` adds the new leading axis to both
    # mu and the matrix sigma and the parts read back unchanged.
    from vfe3.families.gaussian import FullGaussian
    torch.manual_seed(1)
    N, d = 3, 2
    parts = []
    for _ in range(4):
        A = torch.randn(N, 1, d, d)
        parts.append(FullGaussian(torch.randn(N, 1, d), A @ A.transpose(-1, -2) + d * torch.eye(d)))
    stacked = FullGaussian.stack(parts, dim=0)
    assert stacked.mu.shape == (4, N, 1, d) and stacked.sigma.shape == (4, N, 1, d, d)
    for h, part in enumerate(parts):
        assert torch.equal(stacked.mu[h], part.mu) and torch.equal(stacked.sigma[h], part.sigma)


def test_full_gaussian_policy_propagates_and_conflicts_are_explicit():
    from vfe3.families.gaussian import FullGaussian

    mu = torch.zeros(2, 2)
    sigma = torch.eye(2).expand(2, 2, 2)
    q64 = FullGaussian(mu, sigma, _precision_policy="fp64")
    q32 = FullGaussian(mu, sigma, _precision_policy="fp32_escalate")

    assert q64.block(0, 1)._precision_policy == "fp64"
    assert q64.broadcast_over_keys()._precision_policy == "fp64"
    assert FullGaussian.from_transported(mu, sigma, sigma, _precision_policy="fp64")._precision_policy == "fp64"
    assert FullGaussian.stack([q64, q64])._precision_policy == "fp64"
    with pytest.raises(ValueError, match="conflicting explicit precision policies"):
        FullGaussian.stack([q64, q32])


def test_full_gaussian_conflicting_operand_policies_require_override():
    from vfe3.families.gaussian import FullGaussian

    mu = torch.zeros(1, 2)
    sigma = torch.eye(2).unsqueeze(0)
    q64 = FullGaussian(mu, sigma, _precision_policy="fp64")
    q32 = FullGaussian(mu + 0.1, sigma, _precision_policy="fp32_escalate")

    with pytest.raises(ValueError, match="conflicting explicit precision policies"):
        q64.renyi_closed_form(q32)
    assert torch.isfinite(q64.renyi_closed_form(q32, precision_policy="fp32_escalate")).all()


def test_legacy_full_gaussian_subclass_coupling_keeps_legacy_constructor_api():
    from vfe3.families.gaussian import FullGaussian

    class LegacyFullGaussian(FullGaussian):
        def __init__(self, mu, sigma):
            super().__init__(mu, sigma)

        @classmethod
        def from_transported(cls, mu, dispersion, source_dispersion):
            return cls(mu, dispersion)

        @classmethod
        def transport_dispersion(cls, dispersion, omega, *, diagonal_out=None, marginal_blocks=None):
            return super().transport_dispersion(
                dispersion, omega, diagonal_out=diagonal_out, marginal_blocks=marginal_blocks)

    mu = torch.zeros(1, 2, 2)
    sigma = torch.eye(2).expand(1, 2, 2, 2)
    omega = torch.eye(2).expand(1, 2, 2, 2, 2)
    out = LegacyFullGaussian.coupling_energy(mu, sigma, mu, sigma, omega)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("mu_dtype", "sigma_dtype"),
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
)
def test_full_gaussian_natural_promotes_mixed_public_dtypes_before_lu_solve(mu_dtype, sigma_dtype):
    """Mixed public Gaussian moments must use the family's promoted public dtype for the LU solve."""
    mu = torch.tensor([[1.0, -2.0]], dtype=mu_dtype)
    sigma = torch.tensor([[[2.0, 0.25], [0.25, 3.0]]], dtype=sigma_dtype)
    public_dtype = torch.float64
    mu_public = mu.to(public_dtype)
    sigma_public = sigma.to(public_dtype)
    eye = torch.eye(2, dtype=public_dtype)
    precision = torch.linalg.solve(sigma_public + 1e-6 * eye, eye.expand_as(sigma_public))
    expected = (
        (precision @ mu_public.unsqueeze(-1)).squeeze(-1),
        -0.5 * precision,
    )

    got = FullGaussian(mu, sigma).natural()

    assert got[0].dtype is public_dtype and got[1].dtype is public_dtype
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


def test_full_gaussian_per_coord_raises():
    from vfe3.families.gaussian import FullGaussian
    q = FullGaussian(torch.zeros(2, 2), torch.eye(2).expand(2, 2, 2))
    with pytest.raises((AttributeError, NotImplementedError)):
        q.renyi_per_coord(q, alpha=1.0)


def test_families_package_reexports_api_and_registers_builtins():
    """`import vfe3.families` exposes the public API and (via __init__ importing the
    concrete-family module) registers the built-in families, so a config-driven get_family
    works at the package level without importing vfe3.families.gaussian by hand."""
    import vfe3.families as families
    for name in ("BeliefParams", "renyi", "kl", "get_family", "register_family",
                 "family_cov_kind", "divergence_families"):
        assert hasattr(families, name), name
    assert families.get_family("gaussian_diagonal").cov_kind == "diagonal"
    assert families.get_family("gaussian_full").cov_kind == "full"
    assert set(families.divergence_families()) >= {"gaussian_diagonal", "gaussian_full"}


def test_generic_kl_without_expected_statistic_raises_clearly():
    """A family with no renyi_closed_form and no expected_statistic must fail the generic
    KL path with a clear NotImplementedError, not a cryptic AttributeError."""
    from vfe3.families.base import BeliefParams, kl

    class _NoStat(BeliefParams):
        cov_kind = "diagonal"
        def __init__(self, t): self.t = t
        def coordinate_dim(self): return 1
        def block(self, start, end): return _NoStat(self.t)
        def broadcast_over_keys(self): return _NoStat(self.t.unsqueeze(-2))
        def natural(self): return (self.t,)
        @classmethod
        def log_partition_at(cls, theta): return theta[0].sum(dim=-1)
        def entropy(self): return self.t.sum(dim=-1) * 0.0

    with pytest.raises(NotImplementedError):
        kl(_NoStat(torch.ones(3, 1)), _NoStat(torch.ones(3, 1) * 2.0))


def test_generic_kl_from_A_works_for_matrix_sufficient_statistic():
    """The generic Bregman KL (alpha=1) must contract a MATRIX sufficient statistic over ALL
    its parameter axes, not just the last one. FullGaussian's t2 / E[xxᵀ] is (..., K, K), so a
    last-axis-only reducer returns the wrong shape/value; this pins the full contraction. The
    fix is what lets a future matrix-parameter family use the generic path (the M2 promise)."""
    from vfe3.families.gaussian import FullGaussian
    from vfe3.families.base import _renyi_from_log_partition

    torch.manual_seed(6)
    N, K = 4, 3
    mu_q, mu_p = torch.randn(N, K), torch.randn(N, K)
    Aq = torch.randn(N, K, K); s_q = Aq @ Aq.transpose(-1, -2) + K * torch.eye(K)
    Ap = torch.randn(N, K, K); s_p = Ap @ Ap.transpose(-1, -2) + K * torch.eye(K)
    q, p = FullGaussian(mu_q, s_q), FullGaussian(mu_p, s_p)
    closed = q.renyi_closed_form(p, alpha=1.0, kl_max=float("inf"), eps=1e-6)
    generic = _renyi_from_log_partition(q, p, alpha=1.0, kl_max=float("inf"), eps=1e-6)
    assert generic.shape == closed.shape, (generic.shape, closed.shape)
    assert torch.allclose(generic, closed, atol=1e-3), (generic - closed).abs().max()
