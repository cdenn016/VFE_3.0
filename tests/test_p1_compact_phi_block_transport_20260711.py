"""P1 compact phi block transport: routing, equivalence, and model wiring."""

from tests.profiles import test_config

import importlib

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import (
    _COMPOSE,
    _equal_diag_blocks,
    compose_bch,
    compose_phi,
    register_compose,
)
from vfe3.geometry.retraction import retract_phi
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


def test_dense_and_compact_bch_keep_float32_under_cpu_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(37)
    group = get_group("block_glk")(4, 2)
    phi1 = 0.03 * torch.randn(2, 3, group.generators.shape[0])
    phi2 = 0.03 * torch.randn(3, group.generators.shape[0])
    kwargs = {
        "order": 4,
        "gram_pinv_": group.gram_pinv(),
        "block_dims": group.irrep_dims,
    }
    dense_ref = compose_bch(phi1, phi2, group.generators, **kwargs)
    compact_ref = compose_bch(
        phi1,
        phi2,
        group.generators,
        compact_blocks=True,
        **kwargs,
    )

    original_autocast = torch.amp.autocast
    autocast_calls: list[tuple[str, bool]] = []

    def _autocast_spy(device_type: str, *args: object, **kwargs: object) -> object:
        autocast_calls.append((device_type, bool(kwargs.get("enabled", True))))
        return original_autocast(device_type, *args, **kwargs)

    monkeypatch.setattr(torch.amp, "autocast", _autocast_spy)
    dense_plain = compose_bch(phi1, phi2, group.generators, **kwargs)
    compact_plain = compose_bch(
        phi1,
        phi2,
        group.generators,
        compact_blocks=True,
        **kwargs,
    )
    assert autocast_calls == []
    assert torch.equal(dense_plain, dense_ref)
    assert torch.equal(compact_plain, compact_ref)

    with original_autocast("cpu", dtype=torch.bfloat16):
        dense = compose_bch(phi1, phi2, group.generators, **kwargs)
        compact = compose_bch(
            phi1,
            phi2,
            group.generators,
            compact_blocks=True,
            **kwargs,
        )
    assert autocast_calls == [("cpu", False), ("cpu", False)]
    assert dense.dtype == torch.float32
    assert compact.dtype == torch.float32
    assert torch.equal(dense, dense_ref)
    assert torch.equal(compact, compact_ref)

    phi1_double = phi1.double()
    phi2_double = phi2.double()
    generators_double = group.generators.double()
    gram_double = group.gram_pinv().double()
    with original_autocast("cpu", dtype=torch.bfloat16):
        dense_double = compose_bch(
            phi1_double,
            phi2_double,
            generators_double,
            order=4,
            gram_pinv_=gram_double,
            block_dims=group.irrep_dims,
        )
        compact_double = compose_bch(
            phi1_double,
            phi2_double,
            generators_double,
            order=4,
            compact_blocks=True,
            gram_pinv_=gram_double,
            block_dims=group.irrep_dims,
        )
    assert autocast_calls == [("cpu", False), ("cpu", False)]
    assert dense_double.dtype == torch.float64
    assert compact_double.dtype == torch.float64


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


@pytest.mark.parametrize(
    "threshold",
    [20.0, 0.0],
)
def test_compact_phi_block_transport_preserves_global_full_frame_clamp_and_dtype_rule(
    device:         torch.device,
    threshold:     float,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = torch.zeros(1, 1, group.generators.shape[0], device=device)
    phi[..., 0] = 15.0
    phi[..., 4] = 15.0

    legacy = build_belief_transport(
        phi,
        group,
        exp_fp64_norm_threshold=threshold,
    )
    compact = build_belief_transport(
        phi,
        group,
        exp_fp64_norm_threshold=threshold,
        compact_phi_block_transport=True,
    )

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


def test_compact_phi_block_transport_rejects_unmarked_same_shape_basis() -> None:
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

    assert isinstance(transport, FactoredTransport)


def test_compose_registry_default_call_does_not_require_compact_keyword() -> None:
    name = "_p1_legacy_compose_signature"

    @register_compose(name)
    def _legacy_compose(
        phi1:       torch.Tensor,
        phi2:       torch.Tensor,
        generators: torch.Tensor,

        *,
        order:      int,
        gram_pinv_: torch.Tensor,
        block_dims: list[int],
    ) -> torch.Tensor:
        del generators, order, gram_pinv_, block_dims
        return phi1 + phi2

    try:
        generators = get_group("block_glk")(4, 2).generators
        phi1 = torch.zeros(2, generators.shape[0])
        phi2 = torch.ones_like(phi1)
        out = compose_phi(
            phi1,
            phi2,
            generators,
            mode=name,
            gram_pinv_=torch.eye(generators.shape[0]),
            block_dims=[2, 2],
        )
        assert torch.equal(out, phi2)
    finally:
        _COMPOSE.pop(name, None)


def test_compact_phi_block_transport_keeps_bch_retraction_packed_and_matches_vjp(
    device:      torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = (0.02 * torch.randn(1, 3, group.generators.shape[0], device=device)).requires_grad_()
    delta = (0.01 * torch.randn_like(phi)).requires_grad_()
    cotangent = torch.randn_like(phi)

    legacy = retract_phi(
        phi,
        delta,
        group,
        mode="bch",
        compact_blocks=False,
    )
    legacy_grad = torch.autograd.grad(legacy, (phi, delta), cotangent)

    def _dense_embed_forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("canonical packed BCH retraction must not embed a dense KxK matrix")

    monkeypatch.setattr(lie_ops_module, "embed_phi", _dense_embed_forbidden)
    phi_compact = phi.detach().clone().requires_grad_()
    delta_compact = delta.detach().clone().requires_grad_()
    compact = retract_phi(
        phi_compact,
        delta_compact,
        group,
        mode="bch",
        compact_blocks=True,
    )
    compact_grad = torch.autograd.grad(compact, (phi_compact, delta_compact), cotangent)

    assert torch.allclose(compact, legacy, atol=2e-6, rtol=1e-5)
    assert torch.allclose(compact_grad[0], legacy_grad[0], atol=2e-6, rtol=1e-5)
    assert torch.allclose(compact_grad[1], legacy_grad[1], atol=2e-6, rtol=1e-5)


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
