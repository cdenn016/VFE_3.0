from tests.profiles import test_config
import torch

from vfe3.model.model import VFEModel
from vfe3.viz.extract import across_layer_belief_trace, converged_state


def _canonical_cfg():
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
        block_mlp_mode="canonical_frame",
        block_mlp_covariance="passthrough",
        gauge_group="block_glk",
        transport_mode="flat",
        pos_phi="none",
        pos_rotation="none",
        e_phi_lr=0.0,
    )


@torch.no_grad()
def test_canonical_frame_mode_replays_all_model_diagnostics():
    model = VFEModel(_canonical_cfg()).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])

    snapshot = model.build_diagnostic_snapshot(tokens)
    diagnostics = model.diagnostics(tokens)
    attention = model.attention_maps(tokens)
    per_layer = model.diagnostics_per_layer(tokens)
    converged = converged_state(model, tokens)
    layer_trace = across_layer_belief_trace(model, tokens)

    assert snapshot.stack_output.mu.shape == (1, 4, 4)
    assert diagnostics
    assert attention.shape[:2] == (2, 2)
    assert len(per_layer["total"]) == 2
    assert converged["mu"].shape == (4, 4)
    assert layer_trace["mu"].shape == (2, 4, 4)
