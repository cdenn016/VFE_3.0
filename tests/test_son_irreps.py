r"""SO(N)/Sp(2m) irrep-tower gauge groups ('so_n' / 'sp_n'): irrep construction
(symmetric-traceless tensor powers / Sym^p), representation property, group builders,
per-head tau for unequal blocks, config validation, and the end-to-end model smoke.
"""
from tests.profiles import test_config

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









def test_config_irrep_fields_rejected_for_other_groups():
    with pytest.raises(ValueError, match="consumed only by"):
        test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                   gauge_group="block_glk", irrep_spec=[("l1", 1)])








# ---------------------------------------------------------------- end-to-end model



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


@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_projected_model_rejects_so_n_group(mode: str) -> None:
    with pytest.raises(ValueError, match="gauge_group=.*so_n"):
        test_config(attention_mode=mode, gauge_group="so_n", group_n=3,
                    irrep_spec=[("l0", 1), ("l1", 1)])
