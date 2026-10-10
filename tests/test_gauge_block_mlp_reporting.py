from tests.profiles import test_config
from dataclasses import asdict, replace

import pytest
import torch

import scaling
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _cost_model_fields, _pure_path_report


def _pure_cfg(mode, covariance="passthrough"):
    return test_config(
        vocab_size=7,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=2,
        n_e_steps=1,
        family="gaussian_full",
        decode_mode="full_chunked",
        use_block_mlp=True,
        block_mlp_mode=mode,
        block_mlp_covariance=covariance,
        gauge_group="block_glk",
        pos_phi="none",
        pos_rotation="none",
        norm_type_block="none",          # the default "layernorm" is not gauge-equivariant
        sigma_max=None,
        transport_chart_max_norm=12.0,
    )


def test_pure_path_report_distinguishes_strict_gate_from_frame_fixed_mode():
    reference_kwargs = dict(
        attention_mode="projected_predictive", decode_coordinates="reference",

    )
    gate = _pure_path_report(
        replace(_pure_cfg("gauge_gate", "delta_full"), **reference_kwargs), [])
    canonical = _pure_path_report(
        replace(_pure_cfg("canonical_frame"), **reference_kwargs), [])

    assert gate["config_toggles"]["block_mlp_mode"] == "gauge_gate"
    assert gate["config_toggles"]["block_mlp_structural_mode"] == "invariant_scalar_gate"
    assert gate["config_toggles"]["block_mlp_covariance_contract"] == "delta_full_plus_covariant_floor"
    assert gate["config_toggles"]["block_mlp_covariance_floor"] == 1e-4
    assert gate["gauge_flags"]["block_mlp_intertwiner_compatible"] is True
    assert gate["on_gauge_pure_path"] is True
    assert gate["pure_flags"]["no_block_mlp"] is False
    assert gate["on_pure_path"] is False

    assert canonical["config_toggles"]["block_mlp_mode"] == "canonical_frame"
    assert (
        canonical["config_toggles"]["block_mlp_structural_mode"]
        == "canonical_frame_left_equivariant_right_fixed"
    )
    assert canonical["gauge_flags"]["block_mlp_intertwiner_compatible"] is False
    assert canonical["on_gauge_pure_path"] is False


def test_gauge_gate_parameter_accounting_uses_invariant_block_width():
    cfg = _pure_cfg("gauge_gate")
    model = VFEModel(cfg)
    n_blocks = len(model.group.irrep_dims)
    expansion = cfg.block_mlp_expansion
    per_layer = (
        2 * expansion * n_blocks * n_blocks
        + (expansion + 1) * n_blocks
    )
    predicted, _ = scaling.predict_n_params(cfg)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert predicted == actual

    cost = _cost_model_fields(model, cfg, n_params=actual, tokens_seen=13)
    assert cost["block_mlp_params"] == cfg.n_layers * per_layer


@pytest.mark.parametrize("mode", ("coordinate", "gauge_gate", "canonical_frame"))
@pytest.mark.parametrize("covariance", ("passthrough", "delta_full"))
def test_registered_block_mlp_accounting_and_cost_fields_preserve_legacy_parity(
    mode,
    covariance,
):
    cfg = _pure_cfg(mode, covariance)
    model = VFEModel(cfg)
    width = len(model.group.irrep_dims) if mode == "gauge_gate" else cfg.embed_dim
    per_layer_params = (
        2 * cfg.block_mlp_expansion * width * width
        + (cfg.block_mlp_expansion + 1) * width
    )
    expected_params = cfg.n_layers * per_layer_params
    expected_flops = (
        4.0 * cfg.n_layers * cfg.block_mlp_expansion * width * width
    )
    if mode == "gauge_gate":
        expected_flops += cfg.n_layers * sum(
            2.0 * dim ** 3 + 2.0 * dim ** 2
            for dim in model.group.irrep_dims
        )
    elif mode == "canonical_frame":
        expected_flops += 4.0 * cfg.n_layers * cfg.embed_dim ** 2
    if covariance == "delta_full":
        expected_flops += (4.0 + 2.0 / 3.0) * cfg.n_layers * cfg.embed_dim ** 3   # two Cholesky, J L, Gram

    realized_params = sum(
        parameter.numel() for parameter in model.block_mlps.parameters()
    )
    cost = _cost_model_fields(
        model,
        cfg,
        n_params=sum(parameter.numel() for parameter in model.parameters()),
        tokens_seen=13,
    )
    assert realized_params == expected_params
    assert model.executable_build.block_mlp.parameter_count == expected_params
    assert cost["block_mlp_params"] == expected_params
    assert cost["flops_per_token_block_mlp"] == expected_flops


@torch.no_grad()
def test_selected_modes_execute_complete_model_forward():
    tokens = torch.tensor([[1, 2, 3, 4]])
    for mode, covariance in (
        ("coordinate", "delta_full"),
        ("gauge_gate", "passthrough"),
        ("gauge_gate", "delta_full"),
        ("canonical_frame", "passthrough"),
        ("canonical_frame", "delta_full"),
    ):
        model = VFEModel(_pure_cfg(mode, covariance))
        logits = model(tokens)
        assert logits.shape == (1, 4, 7)
        assert torch.isfinite(logits).all()


@torch.no_grad()
def test_canonical_frame_executes_with_right_full_gauge_rope():
    cfg = _pure_cfg("canonical_frame")
    cfg = test_config(**{
        **asdict(cfg),
        "pos_rotation": "rope",
        "rope_on_value": True,
    })
    logits = VFEModel(cfg)(torch.tensor([[1, 2, 3, 4]]))
    assert torch.isfinite(logits).all()
