r"""Bracket-closure certificates measured on the bilinear scale (audit 2026-10-16, F6).

``_strict_structure_constants`` (the strict pullback_group preparation) and ``require_basis_closed`` (the BCH
guard) used to divide each bracket's out-of-span residual by that bracket's own norm. In so(N) with N >= 4,
generators on disjoint index pairs commute, and in the Gram-Schmidt basis of a tensor-power irrep ('l<p>',
p >= 2) those brackets are about 1e-16 of rounding with no component in the span, so the ratio was O(1) and a
closed tower was refused: pullback_group raised at the first M-step. The residual is now divided by the bound
:math:`2\lVert G_a\rVert_F\lVert G_b\rVert_F` on the bracket norm. Non-closed bases still score O(1).

Tiny float64 checks plus one tiny pullback_group train step per gauge tying.
"""
from tests.profiles import test_config

import math
import warnings

import pytest
import torch

F64 = torch.float64


def _cfg(**values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _son_basis(N: int, label: str, K: int) -> torch.Tensor:
    from vfe3.geometry.groups import get_group

    return get_group("so_n")(K, group_n=N, irrep_spec=[(label, 1)], dtype=F64).generators


def _elementary(K: int, pairs: list[tuple[int, int]], scales: list[float]) -> torch.Tensor:
    G = torch.zeros(len(pairs), K, K, dtype=F64)
    for a, ((i, j), s) in enumerate(zip(pairs, scales)):
        G[a, i, j] = s
    return G


# Closed so(N) towers with commuting generator pairs ---------------------------------------------------

@pytest.mark.parametrize("N,label,K", [(4, "l2", 9), (5, "l2", 14), (4, "l3", 16)])
@pytest.mark.parametrize("float32_rounded", [False, True])
def test_strict_structure_constants_accept_closed_tensor_power_towers(
        N: int, label: str, K: int, float32_rounded: bool) -> None:
    from vfe3.geometry.phi_preconditioner import _strict_structure_constants

    G = _son_basis(N, label, K)
    if float32_rounded:                       # the live path promotes the float32 runtime generators
        G = G.to(torch.float32).to(F64)
    coordinates, gram = _strict_structure_constants(G)
    bracket = torch.einsum("aij,bjk->abik", G, G) - torch.einsum("bij,ajk->abik", G, G)
    reconstructed = torch.einsum("abc,cij->abij", coordinates, G)
    tol = 1e-12 if not float32_rounded else 1e-5
    assert float((reconstructed - bracket).abs().max()) < tol * float(G.abs().max()) ** 2
    assert gram.shape == (G.shape[0], G.shape[0])


@pytest.mark.parametrize("N,label,K", [(4, "l2", 9), (5, "l2", 14)])
def test_require_basis_closed_accepts_closed_tensor_power_towers(N: int, label: str, K: int) -> None:
    from vfe3.geometry.lie_ops import require_basis_closed

    require_basis_closed(_son_basis(N, label, K), where="test")


# Non-closed bases are still refused -------------------------------------------------------------------

def _non_closed_bases() -> dict[str, torch.Tensor]:
    return {
        "so4_l2_without_L01": _son_basis(4, "l2", 9)[1:],             # [L_02, L_12] ~ L_01 leaves the span
        "E12_E23": _elementary(3, [(0, 1), (1, 2)], [1.0, 1.0]),       # [E12, E23] = E13
        "scaled_E12_E23": _elementary(3, [(0, 1), (1, 2)], [1e-3, 1.0]),
    }


@pytest.mark.parametrize("name", ["so4_l2_without_L01", "E12_E23", "scaled_E12_E23"])
def test_strict_structure_constants_still_refuse_non_closed_bases(name: str) -> None:
    from vfe3.geometry.phi_preconditioner import _strict_structure_constants

    with pytest.raises(ValueError, match="bracket-closed generator span"):
        _strict_structure_constants(_non_closed_bases()[name])


@pytest.mark.parametrize("name", ["so4_l2_without_L01", "E12_E23", "scaled_E12_E23"])
def test_require_basis_closed_still_refuses_non_closed_bases(name: str) -> None:
    from vfe3.geometry.lie_ops import require_basis_closed

    with pytest.raises(ValueError, match="not closed under the Lie bracket"):
        require_basis_closed(_non_closed_bases()[name], where="test")


# The pullback_group M-step on a so(4) tower with an 'l2' block ----------------------------------------

@pytest.mark.parametrize("tying", ["shared", "per_head"])
def test_so4_l1_l2_tower_takes_a_pullback_group_train_step(tying: str) -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = _cfg(vocab_size=12, embed_dim=13, n_heads=2, max_seq_len=8, n_layers=1, pos_phi="none",
               gauge_group="so_n", group_n=4, irrep_spec=[("l1", 1), ("l2", 1)], gauge_tying=tying,
               m_phi_update_mode="pullback_group", m_phi_group_retraction="chart",
               transport_chart_max_norm=6.0, m_phi_lr=0.1, rope_mode="none")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.randint(0, 12, (4, 8)),
                      torch.randint(0, 12, (4, 8)), grad_clip=1.0)
    assert math.isfinite(float(loss))
    assert not torch.equal(model.prior_bank.phi_embed.detach(), before)
