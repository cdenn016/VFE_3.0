r"""Step 1 of the gauge-group buildout plan (gauge-group-buildouts-2026-10-05.md): shared seams.

Pins, on CPU at tiny size: the registration traits (``GroupTraits``) that replace the config's
name checks on ``so_n``; the Frobenius-orthonormal local bases so, co, sl and b+ and their per-head
assembly; the block-local builder; and, through a throwaway block-local CO(d) group registered only
inside the test, the config seam (tyings, minimum head size, RoPE) and a projected train step with
the gauge-equivariance residual on a group the config knows only by its traits.
"""
import math
import warnings

import pytest
import torch

import vfe3.geometry.groups as groups
from tests.profiles import test_config, tiny_config
from vfe3.config import ConfigNotice
from vfe3.geometry.generators import (
    generate_block_local, local_basis_borel, local_basis_co, local_basis_sl, local_basis_so,
)
from vfe3.geometry.groups import GroupTraits, _build_block_local, group_traits, register_group

BASES = {"so": local_basis_so, "co": local_basis_co, "sl": local_basis_sl, "borel": local_basis_borel}
DIMS = {"so": lambda d: d * (d - 1) // 2, "co": lambda d: d * (d - 1) // 2 + 1,
        "sl": lambda d: d * d - 1, "borel": lambda d: d * (d + 1) // 2}


# ---------------------------------------------------------------- traits

def test_registered_traits_replace_the_so_n_name_checks() -> None:
    son, glk = group_traits("so_n"), group_traits("block_glk")
    assert son.algebra == "so" and son.native_tying == "shared" and not son.contains_pair_rotations
    assert glk.algebra is None and glk.native_tying == "per_head" and glk.contains_pair_rotations
    assert set(son.tyings) == set(glk.tyings) == {"per_head", "shared"}
    with pytest.raises(ValueError, match="native_tying"):
        register_group("test_inconsistent", traits=GroupTraits(tyings=("per_head",), native_tying="shared"))(
            lambda K: None)


# ---------------------------------------------------------------- local bases

@pytest.mark.parametrize("kind", sorted(BASES))
@pytest.mark.parametrize("d", [2, 3, 4])
def test_local_bases_are_orthonormal_closed_and_shaped(kind: str, d: int) -> None:
    B = BASES[kind](d)
    m = DIMS[kind](d)
    assert B.shape == (m, d, d) and B.dtype == torch.float64
    torch.testing.assert_close(torch.einsum("aij,bij->ab", B, B), torch.eye(m, dtype=torch.float64))
    # Bracket closure: every [A, B] lies in the span (orthogonal projection residual is zero).
    flat = B.reshape(m, -1)
    for a in range(m):
        for b in range(m):
            c = (B[a] @ B[b] - B[b] @ B[a]).reshape(-1)
            assert float((c - flat.T @ (flat @ c)).abs().max()) < 1e-12, (kind, a, b)
    if kind == "so":
        assert float((B + B.transpose(-1, -2)).abs().max()) == 0.0
    if kind == "sl":
        assert float(torch.diagonal(B, dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-15
    if kind == "borel":
        assert float(torch.tril(B, diagonal=-1).abs().max()) == 0.0
    if kind == "co":
        torch.testing.assert_close(B[-1], torch.eye(d, dtype=torch.float64) / math.sqrt(d))


def test_generate_block_local_copies_the_basis_head_major() -> None:
    local = local_basis_so(3)
    G = generate_block_local(6, 2, local, dtype=torch.float64)
    assert G.shape == (6, 6, 6)
    torch.testing.assert_close(G[:3, :3, :3], local)
    torch.testing.assert_close(G[3:, 3:, 3:], local)
    assert float(G[:3, 3:, :].abs().max()) == 0.0 and float(G[:3, :, 3:].abs().max()) == 0.0
    assert float(G[3:, :3, :].abs().max()) == 0.0 and float(G[3:, :, :3].abs().max()) == 0.0
    with pytest.raises(ValueError, match="local basis must have shape"):
        generate_block_local(6, 3, local)


# ---------------------------------------------------------------- builder

def test_block_local_builder_metadata_and_group_membership() -> None:
    group = _build_block_local("test_co", 6, 2, local_basis_co(3), skew=False, dtype=torch.float64)
    assert group.generator_shape == (8, 6, 6) and group.irrep_dims == [3, 3] and group.tying == "per_head"
    assert group.frobenius_gram_uniform == 1.0 and group.phi_coordinate_layout is None
    torch.testing.assert_close(group.block_local_basis, local_basis_co(3))
    torch.testing.assert_close(torch.einsum("aij,bij->ab", group.generators, group.generators),
                               torch.eye(8, dtype=torch.float64))
    phi = torch.randn(5, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    U = torch.linalg.matrix_exp(torch.einsum("na,aij->nij", phi, group.generators))
    for h in range(2):
        block = U[:, 3 * h:3 * h + 3, 3 * h:3 * h + 3]
        gram = block.transpose(-1, -2) @ block                              # lambda^2 I for CO(3)
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1)
        torch.testing.assert_close(gram, scale[:, None, None] * torch.eye(3, dtype=torch.float64))
        assert bool((torch.linalg.det(block) > 0).all())
    so = _build_block_local("test_so", 6, 2, local_basis_so(3), skew=True, dtype=torch.float64)
    V = torch.linalg.matrix_exp(torch.einsum("na,aij->nij", phi[:, :6], so.generators))
    torch.testing.assert_close(V.transpose(-1, -2) @ V, torch.eye(6, dtype=torch.float64).expand(5, 6, 6))
    with pytest.raises(ValueError, match="orthonormal"):
        _build_block_local("bad", 6, 2, 2.0 * local_basis_so(3), skew=True)
    with pytest.raises(ValueError, match="skew"):
        _build_block_local("bad", 6, 2, local_basis_co(3), skew=True)


# ---------------------------------------------------------------- a group the config knows by its traits

@pytest.fixture
def block_co(monkeypatch: pytest.MonkeyPatch) -> str:
    name = "test_block_co"
    monkeypatch.setattr(groups, "_GROUPS", dict(groups._GROUPS))           # restored after the test

    @register_group(name, invariant_families=("gaussian_full",), traits=GroupTraits(min_head_dim=2))
    def _build(K: int, n_heads: int, *, dtype: torch.dtype = torch.float32,
               device: 'torch.device | str | None' = None) -> groups.GaugeGroup:
        return _build_block_local(name, K, n_heads, local_basis_co(K // n_heads), skew=False,
                                  dtype=dtype, device=device)
    return name


def test_config_seam_reads_the_traits_of_a_new_group(block_co: str) -> None:
    cfg = test_config(gauge_group=block_co, embed_dim=6, n_heads=2)
    assert cfg.gauge_tying == "per_head" and cfg.head_dims == [3, 3] and cfg.d_head == 3
    with pytest.raises(ValueError, match="gauge_tying"):
        test_config(gauge_group=block_co, embed_dim=6, n_heads=2, gauge_tying="shared")
    with pytest.raises(ValueError, match="head blocks of size >= 2"):
        test_config(gauge_group=block_co, embed_dim=4, n_heads=4)
    with pytest.raises(ValueError, match="irrep-tower"):
        test_config(gauge_group=block_co, embed_dim=6, n_heads=2, group_n=3)
    with pytest.raises(ValueError, match="irrep-labeled tower"):
        test_config(gauge_group=block_co, embed_dim=6, n_heads=2, use_cg_coupling=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # no structure-group warning
        warnings.simplefilter("ignore", ConfigNotice)              # inert-setting advisories are not the subject
        test_config(gauge_group=block_co, embed_dim=6, n_heads=2, pos_rotation="rope")
    with pytest.warns(UserWarning, match="leaves the structure"):
        test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2,
                    pos_rotation="rope")


@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_a_traits_only_group_trains_and_is_equivariant(block_co: str, route: str) -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    extra = (dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                  attention_residual=False) if route == "gradient" else {})
    torch.manual_seed(0)
    model = VFEModel(tiny_config(gauge_group=block_co, embed_dim=6, n_heads=2, **extra))
    assert model.group.name == block_co and model.prior_bank.phi_embed.shape[-1] == 8
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    residuals = extract.projected_gauge_equivariance(
        model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-10 for v in residuals.values()), residuals
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))
