import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _training_cfg(mode, covariance):
    return VFE3Config(
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
        gauge_group="block_glk",
        transport_mode="flat",
        pos_phi="none",
        pos_rotation="none",
        e_phi_lr=0.0,
        oracle_unroll_grad=True,
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
