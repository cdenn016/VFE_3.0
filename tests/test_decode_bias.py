r"""decode_bias: a learned per-vocab log-unigram bias on the linear decode path.

Zero-initialised so the decode is bit-identical to no-bias at construction; routed to a
weight-decay-free optimizer group (decaying a unigram prior toward zero biases it to a flat
distribution). Inert under prior-bank decoder modes, whose per-vocab priors
already carry the unigram role), where it warns and creates no parameter.
"""

import pytest
import torch
import torch.nn.functional as F

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
from vfe3.train import build_optimizer


def test_bias_param_created_only_on_linear_path_when_enabled():
    V, K, n_gen = 20, 4, 16
    pb = PriorBank(V, K, n_gen, decode_mode="linear", decode_bias=True)
    assert isinstance(pb.output_proj_bias, torch.nn.Parameter)
    assert pb.output_proj_bias.shape == (V,)
    assert torch.equal(pb.output_proj_bias, torch.zeros(V))         # zero-init

    pb_nob = PriorBank(V, K, n_gen, decode_mode="linear", decode_bias=False)
    assert pb_nob.output_proj_bias is None                          # off -> no param

    pb_kl = PriorBank(V, K, n_gen, decode_mode="diagonal_chunked", decode_bias=True)
    assert pb_kl.output_proj_bias is None                           # inert on the KL path


def test_zero_init_bias_is_bit_identical_to_no_bias():
    V, K, n_gen = 12, 4, 16
    torch.manual_seed(0); pb_b  = PriorBank(V, K, n_gen, decode_mode="linear", decode_bias=True)
    torch.manual_seed(0); pb_nb = PriorBank(V, K, n_gen, decode_mode="linear", decode_bias=False)
    rng = torch.Generator().manual_seed(1)
    mu_q = torch.randn(2, 3, K, generator=rng)
    sigma_q = torch.rand(2, 3, K, generator=rng) + 0.5
    assert torch.equal(pb_b.decode(mu_q, sigma_q), pb_nb.decode(mu_q, sigma_q))


def test_bias_adds_per_vocab_shift_to_linear_logits():
    V, K, n_gen = 10, 3, 9
    pb = PriorBank(V, K, n_gen, decode_mode="linear", decode_bias=True)
    rng = torch.Generator().manual_seed(2)
    mu_q = torch.randn(2, 4, K, generator=rng)
    sigma_q = torch.rand(2, 4, K, generator=rng) + 0.5
    base = pb.decode(mu_q, sigma_q)                                 # bias still zero
    b = torch.randn(V, generator=rng)
    with torch.no_grad():
        pb.output_proj_bias.copy_(b)
    assert torch.allclose(pb.decode(mu_q, sigma_q), base + b, atol=1e-5)


def test_bias_gradient_descends_toward_log_unigram():
    # Weight frozen at zero -> the only logit signal is the bias; one CE backward at bias=0
    # (uniform softmax) gives grad_v = 1/V - p_emp(v), so gradient DESCENT raises the most
    # frequent token's bias the most: argmin(grad) == argmax(freq), grad monotone in freq.
    V, K, n_gen = 8, 3, 9
    pb = PriorBank(V, K, n_gen, decode_mode="linear", decode_bias=True)
    with torch.no_grad():
        pb.output_proj_weight.zero_()
    mu_q = torch.zeros(1, 50, K)
    sigma_q = torch.ones(1, 50, K)
    targets = torch.tensor([3] * 30 + [1] * 12 + [0] * 8).reshape(1, 50)   # freq: 3 > 1 > 0 > (2=0)
    logits = pb.decode(mu_q, sigma_q)
    F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)).backward()
    g = pb.output_proj_bias.grad
    assert g.argmin().item() == 3                                   # most frequent token rises most
    assert g[3] < g[1] < g[0] < g[2]                               # monotone decreasing in frequency


def test_bias_in_weight_decay_free_optimizer_group():
    cfg = VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     decode_mode="linear", decode_bias=True, weight_decay=0.05)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    bias = model.prior_bank.output_proj_bias
    groups = [g for g in opt.param_groups if any(p is bias for p in g["params"])]
    assert len(groups) == 1                                         # grouped exactly once
    assert groups[0]["weight_decay"] == 0.0                         # unigram prior is not decayed


def test_decode_bias_inert_under_prior_bank_warns():
    with pytest.warns(UserWarning, match="decode_bias"):
        VFE3Config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4,
                   decode_mode="diagonal_chunked", decode_bias=True)
