from tests.profiles import test_config
import pytest
import torch

from vfe3.model.block_mlp import GaugeGateBlockMLP
from vfe3.model.model import VFEModel


def _training_cfg(mode, covariance):
    return test_config(
        vocab_size=17,
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
        # delta_full omits the block-norm Jacobian, so the config admits it only without a block norm.
        norm_type_block="none" if covariance == "delta_full" else "layernorm",
        gauge_group="block_glk",
        pos_phi="none",
        pos_rotation="none",
    )


@pytest.mark.parametrize("mode", ["gauge_gate", "canonical_frame"])
@pytest.mark.parametrize("covariance", ["passthrough", "delta_full"])
def test_selected_gauge_mlp_modes_receive_finite_training_gradients(mode, covariance):
    model = VFEModel(_training_cfg(mode, covariance)).train()
    tokens = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])

    _, loss, _ = model(tokens, targets)
    loss.backward()

    for layer, mlp in enumerate(model.block_mlps):
        for name, parameter in mlp.named_parameters():
            assert parameter.grad is not None, f"missing gradient for layer={layer} parameter={name}"
            assert torch.isfinite(parameter.grad).all(), (
                f"non-finite gradient for layer={layer} parameter={name}"
            )


def _diagonal_cfg(mode, **overrides):
    r"""Tiny gaussian_diagonal projected_predictive route; the fields the WIP defaults flip are pinned."""
    values = dict(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=6,
        n_layers=2,
        n_e_steps=1,
        family="gaussian_diagonal",
        attention_mode="projected_predictive",
        decode_mode="linear",
        gauge_group="block_glk",
        pos_phi="learned",
        pos_rotation="none",
        use_block_mlp=True,
        block_mlp_mode=mode,
        norm_type_block="none",
        layernorm_affine=False,
        skip_belief_sigma_update=False,
    )
    values.update(overrides)
    return test_config(**values)


def _assert_diagonal_gate_training_gradients(model):
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    targets = torch.tensor([[2, 3, 4, 5, 6, 7]])
    _, loss, _ = model(tokens, targets)
    loss.backward()

    assert torch.isfinite(loss)
    for layer, mlp in enumerate(model.block_mlps):
        for name, parameter in mlp.named_parameters():
            assert parameter.grad is not None, f"missing gradient for layer={layer} parameter={name}"
            assert torch.isfinite(parameter.grad).all(), f"layer={layer} parameter={name}"
    sigma_table = model.prior_bank.sigma_log_embed
    assert sigma_table.grad is not None and torch.isfinite(sigma_table.grad).all()
    assert sigma_table.grad.abs().sum() > 0


def _activate_gates(model):
    torch.manual_seed(71)
    with torch.no_grad():
        for mlp in model.block_mlps:
            mlp.fc2.weight.normal_(0.0, 0.1)            # zero-initialized fc2 makes the gate inert


def test_diagonal_gate_in_a_diagonal_model_receives_finite_gradients_including_sigma():
    r"""Module-level wiring, independent of config validation: a coordinate-built model with gates swapped in."""
    model = VFEModel(_diagonal_cfg("coordinate"))
    model.block_mlps = torch.nn.ModuleList([
        GaugeGateBlockMLP(model.group.irrep_dims, 2, "gelu", 0.0) for _ in range(2)])
    _activate_gates(model)

    _assert_diagonal_gate_training_gradients(model.train())


@pytest.mark.parametrize("norm", ["none", "mahalanobis"])
def test_configured_diagonal_gauge_gate_trains_with_finite_gradients(norm):
    r"""Needs the model_core relaxation admitting block_mlp_mode='gauge_gate' for gaussian_diagonal."""
    model = VFEModel(_diagonal_cfg("gauge_gate", norm_type_block=norm))
    assert all(type(mlp) is GaugeGateBlockMLP for mlp in model.block_mlps)
    assert all(mlp.irrep_dims == (2, 2) for mlp in model.block_mlps)
    _activate_gates(model)

    _assert_diagonal_gate_training_gradients(model.train())
