"""Relational perplexity must equal independent token-weighted decoder CE."""
from tests.profiles import test_config
import math

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.train import evaluate


@pytest.mark.parametrize("identity_content", [False])
def test_relational_strided_evaluation_matches_logits(identity_content: bool) -> None:
    torch.manual_seed(79)
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
        n_layers=1, n_e_steps=1, gauge_group="block_glk", family="gaussian_diagonal",
        skip_belief_sigma_update=True,
        attention_mode="projected_gradient",
        )
    model = VFEModel(cfg)
    stream = torch.tensor([0, 1, 2, 3, 4, 1, 3, 2, 0, 4])
    dataset = TokenWindows(stream, 4, stride=2, pad_final=True)
    loader = DataLoader(dataset, batch_size=2)
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for tokens, targets in dataset:
            _, logits = model.forward_beliefs(tokens.unsqueeze(0), return_logits=True)
            total += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, 7), targets, ignore_index=-100, reduction="sum"))
            count += int((targets != -100).sum())
    model.train()
    actual = evaluate(model, loader, tokens_per_char=0.5, device=torch.device("cpu"))
    assert model.training
    assert count == len(stream) - 1 == actual["scored_targets"]
    assert actual["expected_targets"] == actual["scored_targets"] + actual["excluded_targets"]
    assert actual["excluded_targets"] > 0
    assert actual["ce"] == pytest.approx(total / count, rel=1e-6)
    assert actual["ppl"] == pytest.approx(math.exp(total / count), rel=1e-6)
    assert actual["bits_per_token"] == pytest.approx(actual["ce"] / math.log(2.0))
    assert actual["bpc"] == pytest.approx(0.5 * actual["bits_per_token"])
