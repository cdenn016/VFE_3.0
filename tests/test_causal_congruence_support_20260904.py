"""Causal support must exclude Gaussian pairs before numerical certification."""

from tests.profiles import test_config
import os

import pytest
import torch



@pytest.fixture
def device() -> torch.device:
    return torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.mark.parametrize("n_heads", (2,))
def test_tiny_full_gaussian_forward_backward_and_logger(
    device: torch.device, n_heads: int,
) -> None:
    import math

    from vfe3.model.model import VFEModel

    torch.manual_seed(23)
    cfg = test_config(
        vocab_size=7, embed_dim=4, n_heads=n_heads, max_seq_len=3, batch_size=1,
        n_layers=1, n_e_steps=1, family="gaussian_full", gauge_group="block_glk",
        encode_mode="per_token", decode_mode="full_chunked",
        beta_attention_prior="causal_alibi_noself",
        pos_phi="learned", rope_mode="none",
        full_cov_kl_precision="fp32_escalate",
    )
    model = VFEModel(cfg).to(device)
    tokens = torch.tensor([[0, 1, 2]], device=device)
    _, loss, _ = model(tokens, targets=tokens)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    diagnostics = model.diagnostics(tokens)
    assert diagnostics["relational_legacy_global_f_available"] == 0.0
    assert math.isfinite(diagnostics["relational_receiver_cost_after"])
    assert math.isfinite(diagnostics["effective_rank"])
    assert math.isfinite(diagnostics["belief_cond_median"])
