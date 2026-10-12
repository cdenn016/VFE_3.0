r"""Equivalence gate for the fused chunked-vocab decode+CE (decode_mode='diagonal_chunked').

The chunked CE must equal the full-decode CE (decode -> F.cross_entropy) to atol-1e-3 (the
cancellation-sensitive decode's golden tolerance, per docs/perf/2026-05-31-speedup-opportunities.md)
for several chunk sizes, honor ignore_index identically, and give prior-table grads allclose to the
full path. The chunking is a memory reassociation of the SAME CE, never a formula change.
"""

from tests.profiles import test_config
import torch
import torch.nn.functional as F

from vfe3.model.model import VFEModel


def _full_ce(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reference CE on the dense diagonal logits: decode -> F.cross_entropy."""
    logits = model.prior_bank.decode(*_converged(model, tokens))      # (B, N, V)
    flat_logits = logits.reshape(-1, model.cfg.vocab_size)
    flat_targets = targets.reshape(-1)
    return F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)


def _dense_forward_ce(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reference CE through model.forward: the targets=None logits (the dense diagonal oracle)
    scored by F.cross_entropy, so positional-phi composition is applied exactly as in training."""
    logits = model(tokens)                                              # (B, N, V)
    return F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size).float(), targets.reshape(-1),
                           ignore_index=-100)


def _converged(model: VFEModel, tokens: torch.Tensor):
    """Return the same-forward full-width belief state consumed by the decoder."""
    out, _ = model.forward_beliefs(tokens)
    return out.mu.float(), out.sigma.float()


def _model(vocab_size=64, **kw):
    kw.setdefault("decode_mode", "diagonal_chunked")
    cfg = test_config(vocab_size=vocab_size, embed_dim=4, n_heads=2, max_seq_len=6,
                     n_layers=1, n_e_steps=1, e_q_mu_lr=0.05, **kw)
    return VFEModel(cfg)


def test_chunked_ce_matches_full_ce_multiple_chunk_sizes():
    torch.manual_seed(0)
    V = 64
    tokens = torch.randint(0, V, (3, 6))
    targets = torch.randint(0, V, (3, 6))
    full = _model(vocab_size=V)
    full_loss = _full_ce(full, tokens, targets)
    # chunk sizes: divides V evenly, does NOT divide V, and >= V (single chunk).
    for chunk in (16, 8192, V, 7, 100):
        ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=chunk)
        ch.load_state_dict(full.state_dict())
        mu, sigma = _converged(ch, tokens)
        ce = ch.prior_bank.decode_ce_diagonal_chunked(mu, sigma, targets)
        assert torch.allclose(ce, full_loss, atol=1e-3), (
            f"chunk={chunk}: chunked CE {ce.item()} != full CE {full_loss.item()}"
        )


def test_chunked_ce_honors_ignore_index():
    torch.manual_seed(1)
    V = 50
    tokens = torch.randint(0, V, (2, 6))
    targets = torch.randint(0, V, (2, 6))
    targets[0, 0] = -100
    targets[1, 3] = -100
    targets[1, 5] = -100
    full = _model(vocab_size=V)
    full_loss = _full_ce(full, tokens, targets)
    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=13)
    ch.load_state_dict(full.state_dict())
    mu, sigma = _converged(ch, tokens)
    ce = ch.prior_bank.decode_ce_diagonal_chunked(mu, sigma, targets)
    assert torch.allclose(ce.double(), full_loss.double(), atol=1e-3)


def test_chunked_ce_all_ignore_is_finite_zero():
    # Every target == -100: full path emits a finite grad-connected zero; chunked must too.
    torch.manual_seed(2)
    V = 32
    tokens = torch.randint(0, V, (2, 4))
    targets = torch.full((2, 4), -100)
    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=10)
    mu, sigma = _converged(ch, tokens)
    mu = mu.detach().requires_grad_(True)
    ce = ch.prior_bank.decode_ce_diagonal_chunked(mu, sigma, targets)
    assert torch.isfinite(ce) and ce.item() == 0.0
    ce.backward()                                          # grad-connected (no autograd error)


def test_chunked_ce_grad_matches_full():
    # The reference ('diagonal') and the chunked path must BOTH run through model.forward() so
    # each applies the positional-phi gauge composition identically; the only difference is the
    # decode+CE reassociation. A reference built from a bare vfe_stack (_full_ce -> _converged)
    # skips _apply_pos_phi, so with a nonzero learned pos_phi the two graphs diverge and the grads
    # differ by the pos_phi Jacobian -- not a kernel bug (the same-mu kernel-grad equivalence is
    # gated to atol-1e-5 by test_linear_chunked_ce_matches_dense_value_and_grads). This is the
    # faithful "chunked forward grad == diagonal forward grad" gate.
    torch.manual_seed(3)
    V = 48
    tokens = torch.randint(0, V, (2, 5))
    targets = torch.randint(0, V, (2, 5))

    full = _model(vocab_size=V)
    full_loss = _dense_forward_ce(full, tokens, targets)   # through forward(): applies pos_phi
    full_loss.backward()
    g_mu_full = full.prior_bank.mu_embed.grad.clone()
    g_sig_full = full.prior_bank.sigma_log_embed.grad.clone()

    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=11)
    ch.load_state_dict(full.state_dict())
    _, loss, _ = ch(tokens, targets)                       # same forward(), chunked decode+CE
    loss.backward()
    g_mu_ch = ch.prior_bank.mu_embed.grad
    g_sig_ch = ch.prior_bank.sigma_log_embed.grad

    assert torch.allclose(g_mu_ch, g_mu_full, atol=1e-3)
    assert torch.allclose(g_sig_ch, g_sig_full, atol=1e-3)


def test_chunked_forward_loss_matches_diagonal_forward_loss():
    # End-to-end: model.forward under diagonal_chunked returns the same loss/ce as diagonal.
    torch.manual_seed(4)
    V = 40
    tokens = torch.randint(0, V, (3, 6))
    targets = torch.randint(0, V, (3, 6))
    base = _model(vocab_size=V)
    loss_full = ce_full = _dense_forward_ce(base, tokens, targets)

    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    out = ch(tokens, targets)
    logits_ch, loss_ch, ce_ch = out
    assert logits_ch is None                               # fused path forms no (B,N,V) logits
    # Under the default decode_av_precision='fp64' the fused chunked CE keeps float64 while
    # model.forward's dense branch casts to float32; compare values across that dtype boundary.
    assert torch.allclose(loss_ch.to(loss_full.dtype), loss_full, atol=1e-3)
    assert torch.allclose(ce_ch.to(ce_full.dtype), ce_full, atol=1e-3)


def test_chunked_inference_targets_none_returns_full_logits():
    # targets=None (inference/generation) falls back to materialized logits equal to diagonal.
    torch.manual_seed(5)
    V = 40
    tokens = torch.randint(0, V, (2, 5))
    base = _model(vocab_size=V)
    logits_full = base(tokens)

    ch = _model(vocab_size=V, decode_mode="diagonal_chunked", decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    logits_ch = ch(tokens)
    assert logits_ch.shape == (2, 5, V)
    assert torch.allclose(logits_ch, logits_full, atol=1e-3)


# --- vram audit 2026-06-10: fused chunked CE for the linear decode ---
def _linear_model(vocab_size=64, **kw):
    cfg = test_config(vocab_size=vocab_size, embed_dim=4, n_heads=2, max_seq_len=6,
                     n_layers=1, n_e_steps=1, e_q_mu_lr=0.05,
                     decode_mode="linear", **kw)
    return VFEModel(cfg)


def test_linear_chunked_ce_matches_dense_value_and_grads():
    # decode_ce_linear_chunked must equal F.cross_entropy over the dense logits = mu @ W^T
    # in value AND in the gradients to mu and W, for chunk sizes that divide / don't divide /
    # exceed V. The linear map has no cancellation, so the tolerance is tighter than the
    # diagonal kernel's 1e-3.
    torch.manual_seed(6)
    V = 64
    tokens = torch.randint(0, V, (3, 6))
    targets = torch.randint(0, V, (3, 6))
    targets[0, 1] = -100
    m = _linear_model(vocab_size=V, decode_chunk_size=13)
    mu, _ = _converged(m, tokens)
    W = m.prior_bank.output_proj_weight

    mu_a = mu.detach().clone().requires_grad_(True)
    dense = F.cross_entropy(
        (mu_a @ W.transpose(-1, -2)).reshape(-1, V), targets.reshape(-1), ignore_index=-100,
    )
    dense.backward()
    g_mu_dense, g_w_dense = mu_a.grad.clone(), W.grad.clone()

    for chunk in (13, 16, V, 100):
        W.grad = None
        mu_b = mu.detach().clone().requires_grad_(True)
        ce = m.prior_bank.decode_ce_linear_chunked(mu_b, targets, chunk_size=chunk)
        assert torch.allclose(ce, dense, atol=1e-5), f"chunk={chunk}"
        ce.backward()
        assert torch.allclose(mu_b.grad, g_mu_dense, atol=1e-5), f"chunk={chunk} mu grad"
        assert torch.allclose(W.grad, g_w_dense, atol=1e-5), f"chunk={chunk} W grad"


def test_linear_chunked_ce_with_decode_bias():
    # The learned per-vocab log-unigram bias (always on the linear decoder) must enter every chunk's logits
    # and receive the same gradient as on the dense path.
    torch.manual_seed(7)
    V = 50
    tokens = torch.randint(0, V, (2, 6))
    targets = torch.randint(0, V, (2, 6))
    m = _linear_model(vocab_size=V, decode_chunk_size=11)
    with torch.no_grad():
        m.prior_bank.output_proj_bias.normal_(std=0.3)     # nonzero so the bias is load-bearing
    mu, _ = _converged(m, tokens)
    W, b = m.prior_bank.output_proj_weight, m.prior_bank.output_proj_bias

    mu_a = mu.detach().clone().requires_grad_(True)
    dense = F.cross_entropy(
        (mu_a @ W.transpose(-1, -2) + b).reshape(-1, V), targets.reshape(-1), ignore_index=-100,
    )
    dense.backward()
    g_b_dense = b.grad.clone()

    b.grad = None
    mu_b = mu.detach().clone().requires_grad_(True)
    ce = m.prior_bank.decode_ce_linear_chunked(mu_b, targets)
    assert torch.allclose(ce, dense, atol=1e-5)
    ce.backward()
    assert torch.allclose(b.grad, g_b_dense, atol=1e-5)


def test_linear_chunked_ce_all_ignore_is_finite_zero():
    torch.manual_seed(8)
    V = 32
    tokens = torch.randint(0, V, (2, 4))
    targets = torch.full((2, 4), -100)
    m = _linear_model(vocab_size=V, decode_chunk_size=10)
    mu, _ = _converged(m, tokens)
    mu = mu.detach().requires_grad_(True)
    ce = m.prior_bank.decode_ce_linear_chunked(mu, targets)
    assert torch.isfinite(ce) and ce.item() == 0.0
    ce.backward()                                          # grad-connected (no autograd error)


def test_linear_chunked_forward_loss_matches_dense_forward_loss():
    # End-to-end: the linear fused path returns None logits
    # and the same loss/ce as the dense linear decode -> F.cross_entropy path.
    torch.manual_seed(9)
    V = 40
    tokens = torch.randint(0, V, (3, 6))
    targets = torch.randint(0, V, (3, 6))
    base = _linear_model(vocab_size=V)
    _, loss_full, ce_full = base(tokens, targets)

    ch = _linear_model(vocab_size=V, decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    logits_ch, loss_ch, ce_ch = ch(tokens, targets)
    assert logits_ch is None                               # fused path forms no (B,N,V) logits
    assert torch.allclose(loss_ch, loss_full, atol=1e-5)
    assert torch.allclose(ce_ch, ce_full, atol=1e-5)
