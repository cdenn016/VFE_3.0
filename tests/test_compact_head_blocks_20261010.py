r"""Compact head-block frames for every equal head-block group (audit 2026-10-10, item 26).

block_so, block_co, block_glmc, shared-tied block_glk and equal-block so_n towers have every generator supported
on the equal diagonal head blocks, so their transport keeps (..., N, H, d, d) vertex factors instead of dense
K x K ones. Pins, on tiny CPU shapes: the head-block certificate (present for those groups, absent for the
canonical per-head block_glk basis, a basis with off-block support, an unequal so_n tower, and after an
in-place mutation that breaks the support); the compact factors equal the dense ones, with and without a right
factor; whole-model float64 losses and gradients agree between the compact and dense routes; and the
gaussian_block family is admitted and trains on each of these groups while an unequal tower stays refused.
"""

import warnings
from typing import Dict

import pytest
import torch

from tests.profiles import test_config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.geometry.transport import CompactFactoredTransport, FactoredTransport, build_factored_transport
from vfe3.model.model import VFEModel

_GROUPS: Dict[str, tuple] = {
    "block_so": ("block_so", (6, 2), {}),
    "block_co": ("block_co", (6, 2), {}),
    "block_glmc": ("block_glmc", (8, 2), {}),
    "block_glk_shared": ("block_glk", (6, 2), {"tying": "shared"}),
    "so_n_tied": ("so_n", (6,), {"group_n": 3, "irrep_spec": [("l1", 2)]}),
    "so_n_untied": ("so_n", (6,), {"group_n": 3, "irrep_spec": [("l1", 2)], "tying": "per_head"}),
}
_CONFIGS: Dict[str, dict] = {
    "block_so": dict(gauge_group="block_so", embed_dim=6, n_heads=2),
    "block_co": dict(gauge_group="block_co", embed_dim=6, n_heads=2),
    "block_glmc": dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
    "block_glk_shared": dict(gauge_group="block_glk", gauge_tying="shared", embed_dim=6, n_heads=2),
    "so_n_tied": dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2),
    "so_n_untied": dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2,
                        gauge_tying="per_head"),
}
_TOKENS, _TARGETS = torch.tensor([[1, 4, 2, 5]]), torch.tensor([[4, 2, 5, 3]])


def _group(name: str, dtype: torch.dtype = torch.float32) -> GaugeGroup:
    builder, args, kwargs = _GROUPS[name]
    return get_group(builder)(*args, dtype=dtype, **kwargs)


@pytest.mark.parametrize("name", sorted(_GROUPS))
def test_head_block_certificate_holds_for_every_equal_block_group(name: str) -> None:
    group = _group(name)
    blocks = group.head_block_generators()
    H, d = len(group.irrep_dims), group.irrep_dims[0]
    assert blocks is not None and tuple(blocks.shape) == (group.generator_shape[0], H, d, d)
    dense = torch.stack([torch.block_diag(*blocks[a]) for a in range(blocks.shape[0])])
    assert torch.equal(dense, group.generators)


def test_head_block_certificate_is_absent_where_the_basis_is_not_supported_on_equal_blocks() -> None:
    assert get_group("block_glk")(6, 2).head_block_generators() is None          # packed coordinates instead
    unequal = get_group("so_n")(8, group_n=3, irrep_spec=[("l1", 1), ("l2", 1)])
    assert unequal.head_block_generators() is None
    G = torch.zeros(1, 4, 4)
    G[0, 0, 3] = 1.0                                                              # couples the two heads
    assert GaugeGroup(name="coupled", generators=G, irrep_dims=[2, 2], skew_symmetric=False
                      ).head_block_generators() is None
    group = _group("block_so")
    assert group.head_block_generators() is not None
    with torch.no_grad():
        group.generators[0, 0, 5] = 1.0                                           # in-place: breaks the support
    assert group.head_block_generators() is None


@pytest.mark.parametrize("name", sorted(_GROUPS))
@pytest.mark.parametrize("with_right", [False, True])
def test_compact_factors_equal_the_dense_factors(name: str, with_right: bool) -> None:
    group = _group(name)
    g = torch.Generator().manual_seed(0)
    phi = 0.3 * torch.randn(2, 3, group.generator_shape[0], generator=g)
    right = 0.2 * torch.randn(2, 3, group.generator_shape[0], generator=g) if with_right else None
    compact = build_factored_transport(phi, group, compact_blocks=True, right_phi=right)
    dense = build_factored_transport(phi, group, compact_blocks=False, right_phi=right)
    assert isinstance(compact, CompactFactoredTransport) and isinstance(dense, FactoredTransport)
    K = group.generator_shape[-1]
    torch.testing.assert_close(CompactBlockElement(compact.exp_blocks, K).to_dense(), dense.exp_phi,
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(CompactBlockElement(compact.inv_blocks, K).to_dense(), dense.exp_neg_phi,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_float64_model_loss_and_gradients_match_the_dense_route(name: str) -> None:
    cfg = test_config(vocab_size=7, max_seq_len=4, n_layers=2, **_CONFIGS[name])
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double()
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(1))
    assert model._compact_phi_blocks_enabled()
    _, loss_compact, _ = model(_TOKENS, _TARGETS)
    loss_compact.backward()
    compact = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model._compact_phi_blocks_enabled = lambda: False                             # the former dense route
    _, loss_dense, _ = model(_TOKENS, _TARGETS)
    loss_dense.backward()
    torch.testing.assert_close(loss_compact, loss_dense, rtol=1e-13, atol=0.0)
    for n, p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(compact[n], p.grad, rtol=1e-11, atol=1e-13, msg=n)


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_gaussian_block_is_admitted_and_trains_on_compact_head_block_groups(name: str) -> None:
    cfg = test_config(vocab_size=7, max_seq_len=4, n_layers=2, family="gaussian_block", decode_mode="full_chunked",
                      attention_mode="projected_predictive", predictive_covariance="moment_matched", **_CONFIGS[name])
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    _, loss, _ = model(_TOKENS, _TARGETS)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gaussian_block_still_refuses_an_unequal_irrep_tower() -> None:
    with pytest.raises(ValueError, match="equal head blocks"):
        test_config(vocab_size=7, max_seq_len=4, n_layers=1, family="gaussian_block", decode_mode="full_chunked",
                    attention_mode="projected_predictive", predictive_covariance="moment_matched",
                    gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8, n_heads=2)
