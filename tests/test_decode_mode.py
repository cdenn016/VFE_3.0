r"""Behavioral boundary between linear and prior-bank decoder modes.

``decode_mode='linear'`` uses a plain output projection ``mu -> logits`` and discards
``sigma``. Retained prior-bank modes score beliefs against vocabulary priors. Encoding and
free-energy self-coupling stay on ``PriorBank`` for every decoder mode.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.train import build_optimizer


def test_linear_decode_is_plain_projection_of_mu():
    # Linear decode is exactly logits = mu_q @ W^T, W = output_proj_weight (V, K).
    V, K, n_gen = 12, 4, 16
    pb = PriorBank(V, K, n_gen, decode_mode="linear")
    assert pb.output_proj_weight is not None
    assert pb.output_proj_weight.shape == (V, K)
    mu_q = torch.randn(2, 3, K)
    sigma_q = torch.rand(2, 3, K) + 0.5
    logits = pb.decode(mu_q, sigma_q)
    expected = mu_q @ pb.output_proj_weight.t()
    assert logits.shape == (2, 3, V)
    assert torch.allclose(logits, expected, atol=1e-5)


def test_linear_decode_discards_sigma():
    # The linear projection reads only mu: changing sigma must not move the logits.
    V, K, n_gen = 8, 4, 16
    pb = PriorBank(V, K, n_gen, decode_mode="linear")
    mu_q = torch.randn(1, 2, K)
    l1 = pb.decode(mu_q, torch.rand(1, 2, K) + 0.5)
    l2 = pb.decode(mu_q, torch.rand(1, 2, K) + 5.0)
    assert torch.allclose(l1, l2)


def test_prior_bank_mode_has_no_output_projection():
    # The pure path carries no extra projection weight.
    pb = PriorBank(6, 3, 9, decode_mode="diagonal_chunked")
    assert pb.output_proj_weight is None


def test_encode_is_unchanged_by_decoder_mode():
    # Decoder selection does not change the per-token prior-bank lookup.
    V, K, n_gen = 6, 3, 9
    torch.manual_seed(0); pb_prior = PriorBank(V, K, n_gen, decode_mode="diagonal_chunked")
    torch.manual_seed(0); pb_linear = PriorBank(V, K, n_gen, decode_mode="linear")
    tok = torch.tensor([[0, 1, 2]])
    prior, linear = pb_prior.encode(tok), pb_linear.encode(tok)
    assert torch.allclose(prior.mu, linear.mu)
    assert torch.allclose(prior.sigma, linear.sigma)
    assert torch.allclose(prior.phi, linear.phi)


def test_linear_mode_uses_no_nn_linear_module():
    # The exception is a single learned weight matrix (nn.Parameter), NOT an nn.Linear/MLP.
    import torch.nn as nn
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3,
                     decode_mode="linear")
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d))


def test_model_forward_and_backward_under_linear_decode():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, decode_mode="linear")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    logits = model(tokens)
    train_logits, loss, _ = model(tokens, targets)
    assert logits.shape == (3, 5, 20)
    assert train_logits is None
    assert torch.isfinite(loss)
    loss.backward()
    g = model.prior_bank.output_proj_weight.grad
    assert g is not None and g.abs().sum() > 0          # the projection actually trains
    # encode-side mean prior still reached (encode stays on the prior bank)
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0


def test_optimizer_covers_all_params_under_linear_decode():
    # build_optimizer must group the output projection too: every model parameter is covered
    # by exactly one group (a missing group would silently freeze that weight).
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, decode_mode="linear")
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    grouped = {p for grp in opt.param_groups for p in grp["params"]}
    assert grouped == set(model.parameters())
    assert model.prior_bank.output_proj_weight in grouped


def test_optimizer_still_exactly_covers_params_under_prior_bank_decode():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2,
                     decode_mode="diagonal_chunked")
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    grouped = {p for grp in opt.param_groups for p in grp["params"]}
    assert grouped == set(model.parameters())


