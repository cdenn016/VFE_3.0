"""P1 compact phi block transport: routing, equivalence, and model wiring."""

from tests.profiles import test_config

import importlib

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
)
from vfe3.inference.e_step import build_belief_transport
from vfe3.model.model import VFEModel


e_step_module = importlib.import_module("vfe3.inference.e_step")
lie_ops_module = importlib.import_module("vfe3.geometry.lie_ops")
transport_module = importlib.import_module("vfe3.geometry.transport")


def _equal_diag_blocks(
    matrix:    torch.Tensor,              # (..., K, K) block-diagonal, H equal blocks
    n_blocks:  int,
    block_dim: int,
) -> torch.Tensor:                        # (..., H, d, d) the diagonal blocks
    r"""Gather the H equal diagonal blocks of a block-diagonal matrix."""
    m5 = matrix.reshape(*matrix.shape[:-2], n_blocks, block_dim, n_blocks, block_dim)
    return torch.diagonal(m5, dim1=-4, dim2=-2).movedim(-1, -3).contiguous()


def test_compact_phi_block_transport_changes_only_opted_in_block_glk_layout(
    device:      torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = 0.1 * torch.randn(2, 3, group.generators.shape[0], device=device)
    compact_calls: list[torch.Size] = []
    original = transport_module._stable_compact_glk_exp_pair

    def _spy(blocks: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        compact_calls.append(blocks.shape)
        return original(blocks, **kwargs)

    monkeypatch.setattr(transport_module, "_stable_compact_glk_exp_pair", _spy)

    legacy = build_belief_transport(phi, group)
    explicit_false = build_belief_transport(
        phi,
        group,
        compact_phi_block_transport=False,
    )
    assert compact_calls == []
    compact = build_belief_transport(
        phi,
        group,
        compact_phi_block_transport=True,
    )

    assert isinstance(legacy, FactoredTransport)
    assert isinstance(explicit_false, FactoredTransport)
    assert torch.equal(legacy.exp_phi, explicit_false.exp_phi)
    assert torch.equal(legacy.exp_neg_phi, explicit_false.exp_neg_phi)
    assert isinstance(compact, CompactFactoredTransport)
    assert compact_calls == [torch.Size((2, 3, 2, 2, 2))]
    assert compact.exp_blocks.shape == (2, 3, 2, 2, 2)
    assert compact.inv_blocks.shape == (2, 3, 2, 2, 2)


def test_compact_phi_block_transport_shares_the_global_chart_and_dtype_rule(
    device:         torch.device,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = torch.zeros(1, 1, group.generators.shape[0], device=device)
    phi[..., 0] = 15.0
    phi[..., 4] = 15.0                                   # each block 15, whole frame 21.2 > 20
    radius = 20.0                                        # pinned: the default radius is a config choice
    for compact_route in (False, True):
        with pytest.raises(ValueError, match="validity bound exceeded"):
            build_belief_transport(phi, group, compact_phi_block_transport=compact_route,
                                   validity_max_norm=radius)
    phi[..., 4] = 10.0                                   # whole frame 18.0, inside the chart

    legacy = build_belief_transport(phi, group, validity_max_norm=radius)
    compact = build_belief_transport(phi, group, compact_phi_block_transport=True,
                                     validity_max_norm=radius)

    legacy_pos = _equal_diag_blocks(legacy.exp_phi, 2, 2)
    legacy_neg = _equal_diag_blocks(legacy.exp_neg_phi, 2, 2)
    assert torch.equal(compact.exp_blocks, legacy_pos)
    assert torch.equal(compact.inv_blocks, legacy_neg)


def test_self_link_certificates_reject_non_boolean_values() -> None:
    dense = torch.eye(4).expand(1, 3, 4, 4).clone()
    blocks = torch.eye(2).expand(1, 3, 2, 2, 2).clone()
    with pytest.raises(ValueError, match="same_frame_flat_cocycle must be a bool"):
        FactoredTransport(dense, dense, [2, 2], same_frame_flat_cocycle=1)
    with pytest.raises(ValueError, match="same_frame_flat_cocycle must be a bool"):
        CompactFactoredTransport(blocks, blocks, 4, same_frame_flat_cocycle=1)
    base = FactoredTransport(dense, dense, [2, 2], same_frame_flat_cocycle=True)
    with pytest.raises(ValueError, match="same_frame_flat_cocycle must be a bool"):
        RopeTransport(
            insertion="left",   # legacy composition under test
            base=base,
            rope=dense,
            same_frame_flat_cocycle=1,
        )


def test_compact_phi_block_transport_keeps_float64_under_autocast() -> None:
    group = get_group("block_glk")(4, 2, dtype=torch.float64)
    phi = 0.03 * torch.randn(1, 3, group.generators.shape[0], dtype=torch.float64)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        legacy = build_belief_transport(phi, group)
        compact = build_belief_transport(
            phi,
            group,
            compact_phi_block_transport=True,
        )

    assert legacy.exp_phi.dtype == torch.float64
    assert compact.exp_blocks.dtype == torch.float64
    assert torch.equal(compact.exp_blocks, _equal_diag_blocks(legacy.exp_phi, 2, 2))


def test_compact_phi_block_transport_embeds_an_unmarked_same_shape_basis_through_its_blocks() -> None:
    r"""The packed-coordinate reshape needs the certified row-major basis; any other basis supported on the head
    blocks takes the compact route through its blocks (audit 2026-10-10, item 26), with the dense values."""
    canonical = get_group("block_glk")(4, 2)
    custom = GaugeGroup(
        name="block_glk",
        generators=canonical.generators.roll(1, dims=0),
        irrep_dims=list(canonical.irrep_dims),
        skew_symmetric=False,
    )
    phi = 0.03 * torch.randn(1, 3, custom.generators.shape[0])

    transport = build_belief_transport(
        phi,
        custom,
        compact_phi_block_transport=True,
    )
    dense = build_belief_transport(phi, custom, compact_phi_block_transport=False)

    assert isinstance(transport, CompactFactoredTransport) and isinstance(dense, FactoredTransport)
    torch.testing.assert_close(CompactBlockElement(transport.exp_blocks, 4).to_dense(), dense.exp_phi,
                               rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("route", ["single_head", "block_glk"])
def test_compact_phi_block_transport_routes_by_group_layout(route: str) -> None:
    if route == "single_head":
        group = get_group("block_glk")(4, 1)
    else:
        group = get_group("block_glk")(4, 2)
    phi = 0.05 * torch.randn(1, 3, group.generators.shape[0])

    transport = build_belief_transport(
        phi,
        group,
        compact_phi_block_transport=True,
    )

    assert isinstance(transport, CompactFactoredTransport) is (route == "block_glk")


def _tiny_config() -> VFE3Config:
    return test_config(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        decode_mode="diagonal_chunked",
    )


def test_compact_phi_block_transport_reaches_shared_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    dense_model = VFEModel(_tiny_config()).eval()
    compact_model = VFEModel(_tiny_config()).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    monkeypatch.setattr(dense_model, "_compact_phi_blocks_enabled", lambda: False)
    token_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # logits is None on the fused-CE branch by design and the default decode_mode is chunked
    # (audit 2026-08-06), so the logit parity is checked on the unfused no-targets path.
    dense_logits = dense_model(token_ids)
    _, dense_loss, _ = dense_model(token_ids, targets)
    dense_grad, = torch.autograd.grad(dense_loss, dense_model.prior_bank.phi_embed)

    seen: list[object] = []
    original = e_step_module.build_belief_transport

    def _spy(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(e_step_module, "build_belief_transport", _spy)
    compact_logits = compact_model(token_ids)
    _, compact_loss, _ = compact_model(token_ids, targets)
    compact_grad, = torch.autograd.grad(compact_loss, compact_model.prior_bank.phi_embed)

    assert sum(isinstance(item, CompactFactoredTransport) for item in seen) >= 2
    assert torch.allclose(compact_logits, dense_logits, atol=2e-5, rtol=1e-5)
    assert torch.allclose(compact_loss, dense_loss, atol=2e-5, rtol=1e-5)
    assert torch.allclose(compact_grad, dense_grad, atol=5e-5, rtol=1e-5)
