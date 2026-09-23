import pytest
import torch

from vfe3.geometry.groups import GaugeGroup, get_group, register_group


def test_glk_group_full_basis():
    grp = get_group("glk")(K=5)
    assert grp.generators.shape == (25, 5, 5)
    assert grp.irrep_dims == [5]
    assert grp.skew_symmetric is False


def test_tied_block_glk_is_tied_across_heads():
    # tied_block_glk uses kron(I_n, gl(d)): one shared gl(d) for all heads (n_gen = d^2, vs
    # n_heads*d^2 for the untied block_glk), so exp(phi.G) is the SAME GL(d) frame in every block.
    grp = get_group("tied_block_glk")(K=6, n_heads=3)
    d = 2
    assert grp.generators.shape == (d * d, 6, 6)            # d^2 generators (shared), not n_heads*d^2
    assert grp.irrep_dims == [d, d, d]
    assert grp.skew_symmetric is False
    phi = torch.randn(d * d)
    M = torch.einsum("a,aij->ij", phi, grp.generators)
    expM = torch.matrix_exp(M)
    for h in range(1, 3):                                   # every diagonal block equals block 0 (tied)
        assert torch.allclose(expM[:d, :d], expM[h * d:(h + 1) * d, h * d:(h + 1) * d], atol=1e-6)
    # and it is block-diagonal (zero off-block)
    assert torch.allclose(expM[:d, d:2 * d], torch.zeros(d, d), atol=1e-6)


def test_block_glk_group_is_block_diagonal():
    grp = get_group("block_glk")(K=6, n_heads=3)
    assert grp.irrep_dims == [2, 2, 2]
    d = 2
    mask = torch.ones(6, 6)
    for h in range(3):
        mask[h * d:(h + 1) * d, h * d:(h + 1) * d] = 0.0
    for g in grp.generators:
        assert torch.count_nonzero(g * mask) == 0


def test_so_k_group_is_skew():
    grp = get_group("so_k")(K=4)
    assert grp.skew_symmetric is True
    assert torch.allclose(
        grp.generators + grp.generators.transpose(-1, -2),
        torch.zeros_like(grp.generators),
        atol=1e-6,
    )


def _symplectic_form(K: int) -> torch.Tensor:
    # J = [[0, I_m], [-I_m, 0]], the standard symplectic form on R^{2m}.
    m = K // 2
    J = torch.zeros(K, K)
    J[:m, m:] = torch.eye(m)
    J[m:, :m] = -torch.eye(m)
    return J


@pytest.mark.parametrize("K", [4, 6])
def test_sp_generators_are_sp_algebra(K):
    # Every generator G of sp(2m,R) satisfies the algebra membership identity J G + G^T J = 0.
    grp = get_group("sp")(K=K)
    J = _symplectic_form(K)
    lhs = torch.einsum("ij,gjk->gik", J, grp.generators) \
        + torch.einsum("gji,jk->gik", grp.generators, J)
    assert torch.allclose(lhs, torch.zeros_like(lhs), atol=1e-6)
    assert grp.skew_symmetric is False
    assert grp.irrep_dims == [K]


@pytest.mark.parametrize("K", [4, 6])
def test_sp_basis_count_and_independent(K):
    # dim sp(2m,R) = m(2m+1); the basis is linearly independent (flattened rank == n_gen).
    m = K // 2
    grp = get_group("sp")(K=K)
    n_gen = grp.generators.shape[0]
    assert n_gen == m * (2 * m + 1)
    flat = grp.generators.reshape(n_gen, -1)
    assert torch.linalg.matrix_rank(flat) == n_gen


def test_sp_odd_K_raises():
    from vfe3.geometry.generators import generate_sp
    with pytest.raises(ValueError):
        generate_sp(5)


@pytest.mark.parametrize("K", [4, 6])
def test_sp_exp_preserves_symplectic_form(K):
    # exp(t G) is a one-parameter symplectic subgroup: g^T J g = J for each generator.
    grp = get_group("sp")(K=K)
    J = _symplectic_form(K)
    for G in grp.generators:
        g = torch.linalg.matrix_exp(0.3 * G)
        assert torch.allclose(g.transpose(-1, -2) @ J @ g, J, atol=1e-5)


def test_block_glk_with_cross_coupling_grows_basis():
    base = get_group("block_glk")(K=6, n_heads=3)
    coupled = get_group("block_glk")(K=6, n_heads=3, cross_couplings=[(0, 1)])
    assert coupled.generators.shape[0] == base.generators.shape[0] + 4
    # A cross-coupled group is not block-diagonal with d_head blocks; its
    # irrep structure is reported as the single block [K] (super-block
    # decomposition is deferred to Phase 2b transport).
    assert base.irrep_dims == [2, 2, 2]
    assert coupled.irrep_dims == [6]


def test_build_group_default_none_is_byte_identical():
    # Wiring cross_couplings into build_group must add NOTHING when unset: the default-None
    # path produces a group whose generators are bit-identical to the direct builder call.
    from vfe3.config import VFE3Config
    from vfe3.model.model import build_group
    cfg = VFE3Config(embed_dim=4, n_heads=2, gauge_group="block_glk")
    assert cfg.cross_couplings is None
    wired = build_group(cfg)
    direct = get_group("block_glk")(4, 2)
    assert torch.equal(wired.generators, direct.generators)
    assert wired.irrep_dims == direct.irrep_dims


def test_build_group_forwards_cross_couplings():
    # With cross_couplings set, build_group must forward the kwarg, yielding the extended
    # (strictly larger) cross-head basis and the [K] super-block irrep structure. d_head = 2,
    # base = n_heads * d_head^2 = 2*4 = 8; one coupling adds d_head^2 = 4 -> 12.
    from vfe3.config import VFE3Config
    from vfe3.model.model import build_group
    base = build_group(VFE3Config(embed_dim=4, n_heads=2, gauge_group="block_glk"))
    # Headless positional prior: the default per-head causal_alibi bias is fail-closed rejected
    # once cross-coupling collapses the irreps to a single [K] block.
    coupled = build_group(VFE3Config(embed_dim=4, n_heads=2, gauge_group="block_glk",
                                     cross_couplings=[(0, 1)], beta_attention_prior="causal"))
    assert coupled.generators.shape[0] == base.generators.shape[0] + 4
    assert coupled.generators.shape[0] > base.generators.shape[0]   # strictly larger
    assert base.irrep_dims == [2, 2]
    assert coupled.irrep_dims == [4]                                # single super-block [K]


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        get_group("not_a_group")


def test_gaussian_admissibility_is_declared():
    grp = get_group("glk")(K=4)
    assert grp.invariant_for("gaussian_full") is True


@pytest.mark.parametrize("spec", [
    ("glk",       {"K": 4}),
    ("block_glk", {"K": 6, "n_heads": 3}),
    ("so_k",      {"K": 4}),
    ("sp",        {"K": 4}),
    ("so_n",      {"K": 4, "group_n": 3, "irrep_spec": [("l0", 1), ("l1", 1)]}),
    ("sp_n",      {"K": 5, "group_n": 4, "irrep_spec": [("sym0", 1), ("sym1", 1)]}),
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


def test_full_model_logits_invariant_under_global_gauge():
    # t8 (audit 2026-07-06): the component tests pin gauge invariance of the divergence / transport /
    # mixer / metric, but the COMPOSITE claim -- a full VFEModel's decode logits are invariant when the
    # (tied) prior tables are transformed by ONE global structure-group element g -- is never pinned
    # end-to-end. Pure path chosen so the diagonal (V,K) sigma table can represent the gauge action:
    # so_k => g is ORTHOGONAL (g Sigma g^T = Sigma stays diagonal), Sigma = I, frames = identity (Omega = I),
    # family=gaussian_full (diagonal KL is NOT congruence-invariant), flat transport, pos_rotation='none',
    # no head mixer, and a prior-bank decoder that co-transforms. Transforming every (V,K) mean table
    # (mu_embed feeds the belief, and the decode vocab prior on a tied bank; the untied default adds
    # decode_mu_embed) realizes the global gauge; the E-step still evolves non-trivial full
    # covariances that transform by congruence. The fixed-basis seams now on by default are pinned
    # off, since each breaks exact global-gauge invariance by construction: the block LayerNorm
    # (non-equivariant registry entry), the coordinate block MLP, and the frozen-sigma E-step (which
    # would leave the covariances untransformed rather than evolving them).
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    def _cfg(**over):
        base = dict(vocab_size=6, embed_dim=4, n_heads=1, max_seq_len=4, n_layers=1, n_e_steps=3,
                    gauge_group="so_k", family="gaussian_full", transport_mode="flat",
                    pos_rotation="none", use_head_mixer=False, decode_mode="full_chunked",
                    norm_type_block="none", use_block_mlp=False, skip_belief_sigma_update=False)
        base.update(over)
        return VFE3Config(**base)

    def _delta_logits(dbl, **over):
        torch.manual_seed(0)
        m = VFEModel(_cfg(**over))
        with torch.no_grad():
            m.prior_bank.phi_embed.zero_()                 # frames -> identity (Omega = I)
            if hasattr(m, "pos_phi_free"):
                m.pos_phi_free.zero_()                     # no positional gauge
            m.prior_bank.sigma_log_embed.zero_()           # Sigma = I (invariant under orthogonal g)
        if dbl:
            m = m.double()
        m.eval()
        gen = m.group.generators.to(torch.float64 if dbl else torch.float32)
        c = 0.3 * torch.randn(gen.shape[0], generator=torch.Generator().manual_seed(1)).to(gen.dtype)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, gen))
        eye = torch.eye(gen.shape[-1], dtype=gen.dtype)
        assert torch.allclose(g @ g.transpose(-1, -2), eye, atol=1e-6)   # so_k => g orthogonal
        tok = torch.randint(0, 6, (1, 4), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            logits0 = m(tok)[0].clone()
            for table in (m.prior_bank.mu_embed, getattr(m.prior_bank, "decode_mu_embed", None)):
                if table is not None:                      # every (V,K) mean table the model holds
                    table.copy_(torch.einsum("kl,vl->vk", g, table))
            logits1 = m(tok)[0].clone()
        return float((logits0 - logits1).abs().max())

    # Pure path: the full-model decode logits are invariant to the global gauge transform (fp64).
    assert _delta_logits(dbl=True) < 1e-6
    # Teeth (non-vacuity): the linear mu@W^T decode does not co-transform, so the
    # SAME transform changes the logits -- proving the invariance assertion above actually has bite.
    assert _delta_logits(dbl=False, decode_mode="linear") > 1e-4


def test_glk_generators_exact_entries():
    grp = get_group("glk")(K=2)
    expected = torch.tensor([
        [[1., 0.], [0., 0.]],   # E_00
        [[0., 1.], [0., 0.]],   # E_01
        [[0., 0.], [1., 0.]],   # E_10
        [[0., 0.], [0., 1.]],   # E_11
    ])
    assert torch.equal(grp.generators, expected)


def test_son_generators_exact_entries():
    grp = get_group("so_k")(K=2)
    expected = torch.tensor([[[0., 1.], [-1., 0.]]])   # L_01 = E_01 - E_10
    assert torch.equal(grp.generators, expected)


def test_closure_of_e01_e10_is_sl2():
    # {E01, E10} closes to a 3-dim algebra (adds E00 - E11 direction).
    from vfe3.geometry.closure import close_under_brackets
    E01 = torch.tensor([[0., 1.], [0., 0.]])
    E10 = torch.tensor([[0., 0.], [1., 0.]])
    gens = torch.stack([E01, E10], dim=0)
    closed, info = close_under_brackets(gens)
    assert info["final_dim"] == 3 and info["converged"]
    # Re-closing an already-closed basis adds nothing.
    closed2, info2 = close_under_brackets(closed)
    assert info2["n_added"] == 0
