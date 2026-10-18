from tests.profiles import test_config
import pytest
import torch

from vfe3.geometry.groups import get_group


def test_single_head_block_glk_group_full_basis():
    grp = get_group("block_glk")(n_heads=1, K=5)
    assert grp.generators.shape == (25, 5, 5)
    assert grp.irrep_dims == [5]
    assert grp.skew_symmetric is False


def test_block_glk_group_is_block_diagonal():
    grp = get_group("block_glk")(K=6, n_heads=3)
    assert grp.irrep_dims == [2, 2, 2]
    d = 2
    mask = torch.ones(6, 6)
    for h in range(3):
        mask[h * d:(h + 1) * d, h * d:(h + 1) * d] = 0.0
    for g in grp.generators:
        assert torch.count_nonzero(g * mask) == 0


def test_so_n_defining_rep_group_is_skew():
    grp = get_group("so_n")(4, group_n=4, irrep_spec=[("l1", 1)])
    assert grp.skew_symmetric is True
    assert torch.allclose(
        grp.generators + grp.generators.transpose(-1, -2),
        torch.zeros_like(grp.generators),
        atol=1e-6,
    )


def test_build_group_default_none_is_byte_identical():
    # build_group produces a group whose generators are bit-identical to the direct builder call.
    from vfe3.model.model import build_group
    cfg = test_config(embed_dim=4, n_heads=2, gauge_group="block_glk")
    wired = build_group(cfg)
    direct = get_group("block_glk")(4, 2)
    assert torch.equal(wired.generators, direct.generators)
    assert wired.irrep_dims == direct.irrep_dims


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        get_group("not_a_group")


def test_gaussian_admissibility_is_declared():
    grp = get_group("block_glk")(n_heads=1, K=4)
    assert grp.invariant_for("gaussian_full") is True


@pytest.mark.parametrize("spec", [
    ("block_glk", {"K": 4, "n_heads": 1}),
    ("block_glk", {"K": 6, "n_heads": 3}),
    ("so_n",      {"K": 4, "group_n": 4, "irrep_spec": [("l1", 1)]}),
    ("so_n",      {"K": 4, "group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
])
def test_full_kl_invariant_under_group_pushforward(spec):
    # For a random group element g = exp(sum_a c_a G_a), the Gaussian KL is
    # invariant under common pushforward mu->g mu, Sigma->g Sigma g^T.
    from vfe3.divergence import kl
    from vfe3.families.gaussian import FullGaussian
    name, kwargs = spec
    grp = get_group(name)(**kwargs)
    K = sum(grp.irrep_dims)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.2 * torch.randn(grp.generators.shape[0], generator=gen)
    M = torch.einsum("a,aij->ij", coeff, grp.generators)
    g = torch.linalg.matrix_exp(M)                              # (K, K) in G

    mu_q = torch.randn(5, K, generator=gen)
    mu_p = torch.randn(5, K, generator=gen)
    Aq = torch.randn(5, K, K, generator=gen)
    Ap = torch.randn(5, K, K, generator=gen)
    S_q = Aq @ Aq.transpose(-1, -2) + torch.eye(K)
    S_p = Ap @ Ap.transpose(-1, -2) + torch.eye(K)

    base = kl(FullGaussian(mu_q, S_q), FullGaussian(mu_p, S_p))
    mu_q2 = torch.einsum("kl,nl->nk", g, mu_q)
    mu_p2 = torch.einsum("kl,nl->nk", g, mu_p)
    S_q2 = g @ S_q @ g.transpose(-1, -2)
    S_p2 = g @ S_p @ g.transpose(-1, -2)
    moved = kl(FullGaussian(mu_q2, S_q2), FullGaussian(mu_p2, S_p2))

    assert grp.invariant_for("gaussian_full")
    assert torch.allclose(base, moved, atol=1e-3, rtol=1e-3)


def test_single_head_block_glk_generators_exact_entries():
    grp = get_group("block_glk")(n_heads=1, K=2)
    expected = torch.tensor([
        [[1., 0.], [0., 0.]],   # E_00
        [[0., 1.], [0., 0.]],   # E_01
        [[0., 0.], [1., 0.]],   # E_10
        [[0., 0.], [0., 1.]],   # E_11
    ])
    assert torch.equal(grp.generators, expected)


def test_son_generators_exact_entries():
    from vfe3.geometry.generators import generate_son
    expected = torch.tensor([[[0., 1.], [-1., 0.]]])   # L_01 = E_01 - E_10
    assert torch.equal(generate_son(2), expected)


