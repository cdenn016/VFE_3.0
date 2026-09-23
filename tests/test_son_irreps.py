r"""SO(N)/Sp(2m) irrep-tower gauge groups ('so_n' / 'sp_n'): irrep construction
(symmetric-traceless tensor powers / Sym^p), representation property, group builders,
per-head tau for unequal blocks, config validation, and the end-to-end model smoke.
"""

import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.free_energy import attention_tau
from vfe3.geometry.generators import generate_son, generate_sp
from vfe3.geometry.groups import check_admissible, get_group
from vfe3.geometry.irreps import direct_sum_generators, irrep_dim, structure_constants
from vfe3.model.model import VFEModel


SO3_TOWER = [("l0", 1), ("l1", 1), ("l2", 1), ("l3", 1)]      # dims 1+3+5+7 = 16


def _max_bracket_residual(G: torch.Tensor, f: torch.Tensor) -> float:
    """max_ab || [G_a, G_b] - f_ab^c G_c ||_inf for an assembled generator set."""
    n = G.shape[0]
    res = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            lhs = G[a] @ G[b] - G[b] @ G[a]
            rhs = torch.einsum("c,cij->ij", f[a, b], G)
            res = max(res, (lhs - rhs).abs().max().item())
    return res


# ---------------------------------------------------------------- irrep construction

def test_irrep_dim_closed_forms():
    # SO(3): the spin tower, dim 2p + 1
    assert [irrep_dim(3, algebra="so", label=f"l{p}") for p in range(5)] == [1, 3, 5, 7, 9]
    # general N: C(N+p-1, p) - C(N+p-3, p-2)
    assert irrep_dim(4, algebra="so", label="l2") == 9
    assert irrep_dim(5, algebra="so", label="l2") == 14
    # Sp(2m): Sym^p of the defining rep, dim C(2m+p-1, p)
    assert irrep_dim(4, algebra="sp", label="sym0") == 1
    assert irrep_dim(4, algebra="sp", label="sym1") == 4
    assert irrep_dim(4, algebra="sp", label="sym2") == 10


def test_unknown_irrep_label_raises():
    with pytest.raises(ValueError, match="unknown irrep label"):
        irrep_dim(3, algebra="so", label="wedge2")
    with pytest.raises(ValueError, match="unknown irrep label"):
        irrep_dim(3, algebra="so", label="l")              # no rank digits


def test_so3_tower_skew_homomorphic_block_diagonal():
    G_def = generate_son(3, dtype=torch.float64)
    G, dims = direct_sum_generators(G_def, algebra="so", irrep_spec=SO3_TOWER)
    assert dims == [1, 3, 5, 7]
    assert G.shape == (3, 16, 16)
    # exactly skew in the real orthonormal irrep bases (the skew_symmetric=True contract)
    assert (G + G.transpose(-1, -2)).abs().max().item() < 1e-12
    # the assembled direct sum is a genuine so(3) representation
    f = structure_constants(G_def)
    assert _max_bracket_residual(G, f) < 1e-10
    # zero off-block entries (block-diagonality the transport layer assumes)
    mask = torch.ones(16, 16, dtype=torch.bool)
    start = 0
    for d in dims:
        mask[start:start + d, start:start + d] = False
        start += d
    assert G[:, mask].abs().max().item() == 0.0
    # l0 (trivial) block carries zero generators: those coordinates are gauge-invariant
    assert G[:, 0, :].abs().max().item() == 0.0


def test_sp_sym_tower_homomorphic():
    G_def = generate_sp(4, dtype=torch.float64)               # sp(4), m=2, n_gen = 10
    G, dims = direct_sum_generators(G_def, algebra="sp",
                                    irrep_spec=[("sym0", 1), ("sym1", 1)])
    assert dims == [1, 4]
    assert G.shape == (10, 5, 5)
    f = structure_constants(G_def)
    assert _max_bracket_residual(G, f) < 1e-10


# ---------------------------------------------------------------- group builders

def test_so_n_group_builder_fields():
    grp = get_group("so_n")(16, group_n=3, irrep_spec=SO3_TOWER)
    assert grp.name == "so_n"
    assert grp.generators.shape == (3, 16, 16)                # n_gen = dim so(3) = 3
    assert grp.irrep_dims == [1, 3, 5, 7]
    assert grp.skew_symmetric is True
    assert grp.generators.dtype == torch.float32
    # exp(phi . G) is block-diagonal orthogonal (one SO(3) element through every irrep)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.3 * torch.randn(3, generator=gen)
    g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, grp.generators))
    assert torch.allclose(g @ g.transpose(-1, -2), torch.eye(16), atol=1e-5)
    mask = torch.ones(16, 16, dtype=torch.bool)
    start = 0
    for d in grp.irrep_dims:
        mask[start:start + d, start:start + d] = False
        start += d
    assert g[mask].abs().max().item() < 1e-6


def test_so_n_multiplicity_ties_identical_blocks():
    grp = get_group("so_n")(6, group_n=3, irrep_spec=[("l1", 2)])
    assert grp.irrep_dims == [3, 3]
    # the SAME irrep image in both blocks: one phi drives identical 3x3 factors (tied gauge)
    assert torch.equal(grp.generators[:, :3, :3], grp.generators[:, 3:, 3:])


def test_sp_n_group_builder_fields():
    grp = get_group("sp_n")(5, group_n=4, irrep_spec=[("sym0", 1), ("sym1", 1)])
    assert grp.name == "sp_n"
    assert grp.generators.shape == (10, 5, 5)                 # n_gen = m(2m+1) = 10 for m=2
    assert grp.irrep_dims == [1, 4]
    assert grp.skew_symmetric is False


def test_so_n_builder_errors():
    with pytest.raises(ValueError, match="requires group_n"):
        get_group("so_n")(16)
    with pytest.raises(ValueError, match="sum to"):
        get_group("so_n")(8, group_n=3, irrep_spec=SO3_TOWER)     # 16 != 8
    with pytest.raises(ValueError, match="unknown irrep label"):
        get_group("so_n")(4, group_n=3, irrep_spec=[("sym1", 1), ("l1", 1)])


def test_so_n_full_gaussian_admissible():
    grp = get_group("so_n")(9, group_n=3, irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)])
    assert check_admissible(grp, "gaussian_full", n_samples=4) is True


# ---------------------------------------------------------------- per-head tau

def test_attention_tau_unequal_blocks_per_head_sqrt():
    tau = attention_tau(2.0, [1, 3, 5, 7])
    assert isinstance(tau, torch.Tensor) and tau.shape == (4,)
    assert torch.allclose(tau, 2.0 * torch.tensor([1.0, 3.0, 5.0, 7.0]).sqrt())


def test_attention_tau_unequal_blocks_per_head_kappa_elementwise():
    tau = attention_tau(torch.tensor([1.0, 2.0]), [1, 3])
    assert torch.allclose(tau, torch.tensor([1.0, 2.0 * math.sqrt(3.0)]))


def test_attention_tau_equal_blocks_unchanged_scalar():
    tau = attention_tau(1.0, [4, 4])
    assert isinstance(tau, float) and abs(tau - 2.0) < 1e-12


# ---------------------------------------------------------------- config validation

def _so_n_cfg(**overrides) -> VFE3Config:
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                gauge_group="so_n", group_n=3, irrep_spec=[("l0", 1), ("l1", 1)],
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0)
    base.update(overrides)
    return VFE3Config(**base)


def test_config_so_n_accepts_and_coerces_json_pairs():
    cfg = _so_n_cfg(irrep_spec=[["l0", 1], ["l1", 1]])        # JSON round-trip: lists -> tuples
    assert cfg.irrep_spec == [("l0", 1), ("l1", 1)]


def test_config_so_n_dim_sum_mismatch_rejected():
    with pytest.raises(ValueError, match="sum to"):
        _so_n_cfg(irrep_spec=[("l1", 1)])                     # 3 != embed_dim 4


def test_config_so_n_requires_both_fields():
    with pytest.raises(ValueError, match="requires both"):
        _so_n_cfg(group_n=None)


def test_config_irrep_fields_rejected_for_other_groups():
    with pytest.raises(ValueError, match="consumed only by"):
        VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                   gauge_group="block_glk", irrep_spec=[("l1", 1)])


def test_config_so_n_rejects_per_block_phi_preconditioners():
    for mode in ("killing_per_block", "pullback_per_block"):
        with pytest.raises(ValueError, match="per-block metric is undefined"):
            _so_n_cfg(phi_precond_mode=mode)


def test_config_so_n_per_head_kappa_keyed_to_block_count():
    cfg = _so_n_cfg(kappa_beta=[1.0, 2.0])                    # 2 irrep blocks -> ok
    assert list(cfg.kappa_beta) == [1.0, 2.0]
    with pytest.raises(ValueError, match="one entry per irrep block"):
        _so_n_cfg(kappa_beta=[1.0, 2.0, 3.0])


def test_config_so_n_alibi_requires_head_count_match():
    with pytest.raises(ValueError, match="irrep blocks"):
        _so_n_cfg(beta_attention_prior="causal_alibi", n_heads=4)
    cfg = _so_n_cfg(beta_attention_prior="causal_alibi", n_heads=2)    # 2 blocks == n_heads
    assert cfg.beta_attention_prior == "causal_alibi"


# ---------------------------------------------------------------- end-to-end model

def test_model_runs_under_so_n_irrep_tower():
    # The SO(3) spin tower l0 + l1 (UNEQUAL block dims [1, 3]) must run end-to-end through the
    # same model/E-step machinery: forward + loss.backward() with targets yields a finite loss
    # and finite grad on the prior tables. Mirrors test_model_runs_under_sp_gauge_group.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05, m_phi_lr=0.01,
                     gauge_group="so_n", group_n=3, irrep_spec=[("l0", 1), ("l1", 1)],
                     phi_precond_mode="none")
    model = VFEModel(cfg)
    assert model.group.irrep_dims == [1, 3]
    assert model.group.generators.shape[0] == 3               # n_gen = dim so(3)
    assert model.prior_bank.phi_embed.shape == (20, 3)        # phi table sized by n_gen
    tok = torch.randint(0, 20, (3, 5)); tgt = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
    assert torch.isfinite(model.prior_bank.phi_embed.grad).all()
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
    # per-head attention over UNEQUAL irrep blocks: H = number of blocks
    maps = model.attention_maps(torch.randint(0, 20, (2, 5)))
    assert maps.shape == (1, 2, 5, 5)


# ---------------------------------------------------------------- irrep_labels field

def test_groups_expose_irrep_labels():
    grp = get_group("so_n")(14, group_n=3,
                            irrep_spec=[("l0", 1), ("l1", 2), ("l3", 1)])
    assert grp.irrep_dims == [1, 3, 3, 7]
    assert grp.irrep_labels == ["l0", "l1", "l1", "l3"]
    grp2 = get_group("sp_n")(5, group_n=4, irrep_spec=[("sym0", 1), ("sym1", 1)])
    assert grp2.irrep_labels == ["sym0", "sym1"]
    # legacy groups carry no labels
    assert get_group("glk")(4).irrep_labels is None
    assert get_group("block_glk")(6, 3).irrep_labels is None


def test_irrep_labels_length_validated():
    from vfe3.geometry.groups import GaugeGroup
    with pytest.raises(ValueError, match="irrep_labels"):
        GaugeGroup(name="x", generators=torch.zeros(1, 4, 4), irrep_dims=[2, 2],
                   skew_symmetric=True, irrep_labels=["a"])
