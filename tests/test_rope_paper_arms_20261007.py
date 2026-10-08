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


@pytest.mark.parametrize("fraction, rotated", [(1.0, 5), (0.75, 3), (0.5, 2), (0.2, 1)])
def test_p_rope_rotates_only_the_fastest_pairs_at_full_rope_frequencies(fraction: float, rotated: int) -> None:
    r"""p-RoPE (Barbero et al. 2025): pairs k < floor(p d / 2) rotate by n base^{-2k/d}; the rest stay fixed."""
    d, n, base = 10, 7, 100.0
    R = build_rope_rotation(torch.arange(n), [d], base=base, fraction=fraction, dtype=torch.float64)
    full = build_rope_rotation(torch.arange(n), [d], base=base, dtype=torch.float64)
    for k in range(d // 2):
        block = R[:, 2 * k:2 * k + 2, 2 * k:2 * k + 2]
        if k < rotated:
            assert torch.equal(block, full[:, 2 * k:2 * k + 2, 2 * k:2 * k + 2]), k
        else:
            assert torch.equal(block, torch.eye(2, dtype=torch.float64).expand(n, 2, 2)), k
    with pytest.raises(ValueError, match="rotates no pair"):
        build_rope_rotation(torch.arange(n), [d], fraction=0.1)


def test_rope_fraction_reaches_every_rope_path() -> None:
    from vfe3.model.model import VFEModel

    with pytest.raises(ValueError, match="rope_fraction"):
        _vcfg(rope_fraction=0.1)                                              # d = 4: no pair rotated
    with pytest.raises(ValueError, match="rope_fraction"):
        _vcfg(rope_fraction=0.0)
    cfg = _vcfg(rope_fraction=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    assert model.relational_operators.comparison_rope_fraction == 0.5
    frame = _vcfg(rope_space="frame", rope_fraction=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R = VFEModel(frame)._rope_rotation(N, torch.device("cpu"))           # (N, K, K)
    d = K // H
    expected = build_rope_rotation(torch.arange(N), [d] * H, base=frame.rope_base, fraction=0.5, dtype=R.dtype)
    assert torch.equal(R, expected) and torch.equal(R[:, 2:4, 2:4], torch.eye(2).expand(N, 2, 2))
    bcfg = BaselineConfig.matched_to(cfg)
    assert bcfg.rope_fraction == 0.5
    torch.manual_seed(0)
    Rb = BaselineTransformer(bcfg)._rope_rotation(N, torch.device("cpu"))
    assert torch.equal(Rb, build_rope_rotation(torch.arange(N), [d], base=cfg.rope_base, fraction=0.5))
    assert "rope_fraction" in pa.check_baseline_arm(cfg, BaselineConfig.matched_to(cfg, rope_fraction=1.0),
                                                    declared=("rope_fraction",))


def test_comparison_p_rope_still_routes_on_relative_position() -> None:
    r"""With p < 1 the comparison-space cost of a constant sequence is still a function of j - i only."""
    import vfe3.gradients.relational as rel
    from vfe3.contracts import CanonicalFrameContext
    from vfe3.model.model import VFEModel

    cfg = _vcfg(rope_fraction=0.5)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double()
    ops = model.relational_operators
    n, d = 6, K // H
    mu = torch.randn(1, 1, K, dtype=torch.float64).expand(1, n, K)
    sigma = (0.5 + torch.rand(1, 1, K, dtype=torch.float64)).expand(1, n, K)
    eye = torch.eye(d, dtype=torch.float64).expand(1, n, H, d, d).clone()
    frame = CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=True)
    cost = rel.relational_score(mu, sigma, mu, sigma, frame, ops)              # (1, H, n, n)
    for offset in range(-n + 1, n):
        diag = torch.diagonal(cost[0], offset=offset, dim1=-2, dim2=-1)
        assert torch.allclose(diag, diag[..., :1].expand_as(diag), rtol=1e-9, atol=1e-10), offset
