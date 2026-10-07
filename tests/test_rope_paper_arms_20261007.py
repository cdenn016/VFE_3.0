r"""RoPE for the paper arms (October 7, 2026).

Comparison-space RoPE rotates each head's projected routing moments and never the frame, so the frame-only
controls (identity frames, the additive encoder, reference coordinates) accept it; frame-space RoPE stays
rejected there. The conventional baseline gains the same per-head rotary query/key map, copied from a RoPE
base by ``matched_to``, and the paper arms check and swap it.
"""

from tests.profiles import test_config
import warnings
from typing import Any, Dict

import pytest
import torch

import paper_arms as pa
from vfe3.baseline import BaselineConfig, BaselineTransformer
from vfe3.config import VFE3Config
from vfe3.geometry.rope import build_rope_rotation

V, K, H, L, E, N, B = 11, 8, 2, 2, 2, 6, 2


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _vcfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, eval_stride=3,
        n_layers=L, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, generate_embedding_figures=False, use_block_mlp=True,
        block_mlp_expansion=E, block_mlp_mode="coordinate", beta_attention_prior="causal",
        family="gaussian_diagonal", gauge_group="block_glk", attention_mode="projected_predictive",
        predictive_readout="kernel", attention_residual=True, decode_mode="diagonal_chunked",
        pos_phi="none", pos_rotation="rope", rope_space="comparison", rope_on_value=False, rope_base=100.0,
        norm_type_block="layernorm", layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _bcfg(**over: Any) -> BaselineConfig:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, n_layers=L, mlp_expansion=E, max_seq_len=N,
        batch_size=B, eval_stride=3, max_steps=8, warmup_steps=1, lr=3e-3, log_interval=0,
        eval_interval=0, checkpoint_interval=0, seed=5, beta_attention_prior="causal")
    values.update(over)
    return BaselineConfig(**values)


@pytest.mark.parametrize("over", [
    {"gauge_transport": "off"},
    {"encode_mode": "per_token_additive"},
    {"block_coordinates": "reference"},
    {"decode_coordinates": "reference"},
])
def test_frame_controls_accept_comparison_rope_and_reject_frame_rope(over: Dict[str, Any]) -> None:
    cfg = _vcfg(**over)
    assert cfg.pos_rotation == "rope" and cfg.rope_space == "comparison"
    with pytest.raises(ValueError, match="pos_rotation"):
        _vcfg(rope_space="frame", **over)


def test_identity_frames_keep_comparison_rope_routing() -> None:
    from vfe3.model.model import VFEModel

    cfg = _vcfg(gauge_transport="off")
    assert cfg.phi_scale == 0.0 and cfg.pos_phi == "none"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    assert model.relational_operators.comparison_rope_base == cfg.rope_base
    assert model._rope_rotation(N, torch.device("cpu")) is None              # no rotation in the frame


def test_baseline_rope_scores_depend_only_on_relative_position() -> None:
    r"""Identical tokens: :math:`\log\beta_{i,i-m} - \log\beta_{ii}` is the same for every row :math:`i`."""
    torch.manual_seed(0)
    model = BaselineTransformer(_bcfg(pos_rotation="rope"))
    tokens = torch.full((1, N), 3, dtype=torch.long)
    weights: list = []
    with torch.no_grad():
        model.blocks[0].attn.w_q.mul_(40.0)                                  # O(1) scores at a tiny init
        model.blocks[0].attn.w_k.mul_(40.0)
        model.hidden(tokens, weights_out=weights)
    log_beta = weights[0][0].double().log()                                   # (H, N, N), layer 0
    for m in range(1, N):
        gaps = torch.stack([log_beta[:, i, i - m] - log_beta[:, i, i] for i in range(m, N)], dim=-1)
        assert torch.allclose(gaps, gaps[..., :1].expand_as(gaps), atol=1e-6), m
    torch.manual_seed(0)
    plain = BaselineTransformer(_bcfg())
    flat: list = []
    with torch.no_grad():
        plain.blocks[0].attn.w_q.mul_(40.0)
        plain.blocks[0].attn.w_k.mul_(40.0)
        plain.hidden(tokens, weights_out=flat)
    assert torch.allclose(flat[0][0, :, -1, :], torch.full((H, N), 1.0 / N), atol=1e-6)   # NoPE: uniform
    assert not torch.allclose(weights[0][0, :, -1, :], flat[0][0, :, -1, :], atol=1e-3)   # RoPE acts


def test_baseline_rope_is_the_vfe_comparison_rotation_per_head() -> None:
    torch.manual_seed(0)
    model = BaselineTransformer(_bcfg(pos_rotation="rope", rope_base=37.0))
    R = model._rope_rotation(N, torch.device("cpu"))                          # (N, d, d)
    expected = build_rope_rotation(torch.arange(N), [K // H], base=37.0)
    assert torch.equal(R, expected)
    assert model._rope_rotation(N, torch.device("cpu")) is R                  # cached


def test_matched_to_copies_rope_and_the_arms_check_it() -> None:
    vcfg = _vcfg()
    bcfg = BaselineConfig.matched_to(vcfg)
    assert (bcfg.pos_rotation, bcfg.rope_base) == ("rope", vcfg.rope_base)
    assert BaselineConfig.from_serialized(bcfg.serialize()) == bcfg
    assert pa.check_baseline_arm(vcfg, bcfg) == {}
    assert BaselineConfig.matched_to(_vcfg(pos_rotation="none", pos_phi="learned")).pos_rotation == "none"
    swapped = BaselineConfig.matched_to(vcfg, **pa._ALIBI)
    with pytest.raises(pa.ArmDefinitionError, match="pos_rotation"):
        pa.check_baseline_arm(vcfg, swapped)
    diff = pa.check_baseline_arm(vcfg, swapped, declared=tuple(pa._ALIBI))
    assert diff["pos_rotation"] == ("rope", "none")
    assert diff["beta_attention_prior"] == ("causal", "causal_alibi")
