r"""Causal windowed (local-attention) attention prior (PL13-priors).

The prior extends the attention-prior registry beyond causal/causal_alibi. It is a pure builder
returning a log-prior bias (Nq, Nk), config-selectable through the live _PRIORS registry (config.py
validates beta_attention_prior against tuple(sorted(_PRIORS))). The T5 relative-position bias that
shared this file was removed on 2026-10-12 (round-2 audit).
"""

from tests.profiles import test_config

import torch

from vfe3.attention_prior import _PRIORS, attention_log_prior
from vfe3.model.model import VFEModel


def test_windowed_prior_is_registered_and_config_selectable() -> None:
    assert "causal_windowed" in _PRIORS
    assert "t5_relative_bias" not in _PRIORS
    test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, beta_attention_prior="causal_windowed")


def test_causal_windowed_is_causal_and_banded() -> None:
    B = attention_log_prior("causal_windowed", 6, 6, window=2)
    inf = float("-inf")
    for i in range(6):
        for j in range(6):
            if 0 <= i - j <= 2:                     # past, within band
                assert B[i, j] == 0.0
            else:                                   # future OR too far back
                assert B[i, j] == inf


def test_model_forward_under_windowed_prior() -> None:
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                      n_e_steps=1, m_phi_lr=0.0, beta_attention_prior="causal_windowed")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 8))
    targets = torch.randint(0, 12, (2, 8))
    logits = model(tokens)
    _, _, ce = model(tokens, targets)
    assert torch.isfinite(logits).all() and torch.isfinite(ce)
