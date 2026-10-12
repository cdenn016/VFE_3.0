r"""Second-order (delta-method) expected log-loss for the linear decode (audit 2026-10-12, T3).

With :math:`q = \mathcal{N}(\mu, \mathrm{diag}\,\sigma)` and :math:`p = \mathrm{softmax}(W\mu + c)`,

.. math::
    \mathbb{E}_q[-\log p_y] \approx \mathrm{CE}(\mu)
    + \tfrac12 \Big[\sum_v p_v ((W \circ W)\sigma)_v - \sum_k \sigma_k (W^\top p)_k^2\Big].

The fused chunked kernel must match a dense float64 reference in value and gradients, pass
``gradcheck`` in float64, reduce to the plug-in CE as :math:`\sigma \to 0`, leave the default
(toggle-off) loss bitwise unchanged, and reach the model loss only in training mode.
"""
import pytest
import torch
import torch.nn.functional as F

from tests.profiles import test_config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank

V, K, N_GEN = 23, 4, 4


def _bank(dtype: torch.dtype = torch.float64) -> PriorBank:
    r"""A tiny linear-decode bank with a nonzero bias and a set unigram prior."""
    torch.manual_seed(0)
    pb = PriorBank(V, K, N_GEN, decode_mode="linear", decode_bias=True, decode_unigram_prior=True,
                   unigram_kappa=0.7).to(dtype)
    with torch.no_grad():
        pb.output_proj_weight.normal_(std=0.8)
        pb.output_proj_bias.normal_(std=0.3)
    pb.set_unigram_log_prior(torch.randint(0, 50, (V,)))
    return pb


def _dense_reference(
    pb:      PriorBank,
    mu:      torch.Tensor,               # (B, N, K)
    sigma:   torch.Tensor,               # (B, N, K) diagonal variances
    targets: torch.Tensor,               # (B, N) with -100 ignored
    W:       torch.Tensor,               # (V, K)
) -> torch.Tensor:
    r"""Dense :math:`\mathrm{CE}(\mu) + \tfrac12[p^\top (W\circ W)\sigma - \sigma^\top (W^\top p)^2]`."""
    logits = mu @ W.T + pb.output_proj_bias + pb._unigram_bias()        # (B, N, V)
    p = torch.softmax(logits, dim=-1)                                   # (B, N, V)
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100,
                         reduction="none").reshape(targets.shape)       # (B, N)
    corr = 0.5 * ((p @ (W * W)) * sigma).sum(-1) - 0.5 * (sigma * (p @ W) ** 2).sum(-1)  # (B, N)
    valid = targets != -100
    return ((ce + corr) * valid).sum() / valid.sum().clamp_min(1)


def _inputs(dtype: torch.dtype = torch.float64):
    torch.manual_seed(1)
    mu = (0.7 * torch.randn(2, 5, K, dtype=dtype)).requires_grad_(True)
    sigma = (0.05 + 0.3 * torch.rand(2, 5, K, dtype=dtype)).requires_grad_(True)
    targets = torch.randint(0, V, (2, 5))
    targets[0, 1] = -100
    targets[1, 4] = -100
    return mu, sigma, targets


@pytest.mark.parametrize("checkpoint", ["never", "always"])
@pytest.mark.parametrize("chunk", [5, 7, V, 100])
def test_matches_dense_float64_reference_in_value_and_gradients(chunk: int, checkpoint: str) -> None:
    pb = _bank()
    pb.decode_ce_checkpoint = checkpoint
    mu, sigma, targets = _inputs()
    W, b = pb.output_proj_weight, pb.output_proj_bias

    ref = _dense_reference(pb, mu, sigma, targets, W)
    grads_ref = torch.autograd.grad(ref, (mu, sigma, W, b))
    ce = pb.decode_ce_linear_chunked(mu, targets, chunk_size=chunk, sigma_q=sigma)
    grads = torch.autograd.grad(ce, (mu, sigma, W, b))

    assert torch.allclose(ce, ref, atol=1e-12, rtol=0)
    for name, g, g_ref in zip(("mu", "sigma", "W", "b"), grads, grads_ref):
        assert torch.allclose(g, g_ref, atol=1e-12, rtol=0), name


def test_gradcheck_float64_mu_sigma_and_W() -> None:
    pb = _bank()
    mu, sigma, targets = _inputs()
    W0 = pb.output_proj_weight.detach().clone().requires_grad_(True)
    del pb.output_proj_weight                                            # read W from the closure

    def loss(mu_: torch.Tensor, sigma_: torch.Tensor, W_: torch.Tensor) -> torch.Tensor:
        pb.output_proj_weight = W_
        return pb.decode_ce_linear_chunked(mu_, targets, chunk_size=6, sigma_q=sigma_)

    assert torch.autograd.gradcheck(loss, (mu.detach().requires_grad_(True),
                                           sigma.detach().requires_grad_(True), W0))


def test_reduces_to_plug_in_as_sigma_vanishes() -> None:
    pb = _bank()
    mu, sigma, targets = _inputs()
    plug_in = pb.decode_ce_linear_chunked(mu, targets, chunk_size=7)
    zero = pb.decode_ce_linear_chunked(mu, targets, chunk_size=7, sigma_q=torch.zeros_like(sigma))
    assert torch.allclose(zero, plug_in, atol=1e-14, rtol=0)
    gaps = [(pb.decode_ce_linear_chunked(mu, targets, chunk_size=7, sigma_q=t * sigma) - plug_in).item()
            for t in (1e-2, 1e-4)]
    assert gaps[0] > 0.0 and gaps[1] > 0.0                               # LSE convexity: term >= 0
    assert gaps[1] == pytest.approx(gaps[0] * 1e-2, rel=1e-9)           # exactly linear in sigma


def test_close_to_monte_carlo_expected_log_loss_at_small_variance() -> None:
    pb = _bank()
    mu, sigma, targets = _inputs()
    sigma = 0.02 * torch.ones_like(sigma)
    delta = pb.decode_ce_linear_chunked(mu, targets, chunk_size=7, sigma_q=sigma).item()
    plug_in = pb.decode_ce_linear_chunked(mu, targets, chunk_size=7).item()
    gen = torch.Generator().manual_seed(2)
    draws = mu.detach() + sigma.sqrt() * torch.randn((20000, *mu.shape), generator=gen, dtype=mu.dtype)
    W = pb.output_proj_weight.detach()
    logits = draws @ W.T + pb.output_proj_bias.detach() + pb._unigram_bias()   # (S, B, N, V)
    nll = torch.logsumexp(logits, -1) - logits.gather(
        -1, targets.clamp_min(0).expand(20000, -1, -1).unsqueeze(-1)).squeeze(-1)  # (S, B, N)
    valid = targets != -100
    mc = ((nll.mean(0) * valid).sum() / valid.sum()).item()
    assert abs(delta - mc) < 0.1 * abs(mc - plug_in)


def test_toggle_off_loss_is_bitwise_unchanged_and_refuses_reference_coordinates() -> None:
    pb = _bank(torch.float32)
    mu, sigma, targets = _inputs(torch.float32)
    a = pb.decode_ce_linear_chunked(mu, targets, chunk_size=7)
    b = pb.decode_ce_linear_chunked(mu, targets, chunk_size=7, sigma_q=None)
    assert torch.equal(a, b)
    pb.decode_coordinates = "reference"
    with pytest.raises(ValueError, match="local coordinates"):
        pb.decode_ce_linear_chunked(mu, targets, sigma_q=sigma)


def _model(**kw) -> VFEModel:
    cfg = test_config(vocab_size=19, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1, n_e_steps=1,
                      decode_mode="linear", **kw)
    return VFEModel(cfg)


def test_model_loss_off_bitwise_and_on_only_in_training_mode() -> None:
    torch.manual_seed(3)
    tokens = torch.randint(0, 19, (2, 5))
    targets = torch.randint(0, 19, (2, 5))
    off = _model()
    on = _model(decode_second_order=True)
    on.load_state_dict(off.state_dict())
    _, loss_off, _ = off(tokens, targets)
    _, loss_on, _ = on(tokens, targets)
    assert torch.isfinite(loss_on) and loss_on.item() > loss_off.item()
    loss_on.backward()
    assert on.prior_bank.output_proj_weight.grad is not None
    off.eval()
    on.eval()
    with torch.no_grad():
        _, eval_off, _ = off(tokens, targets)
        _, eval_on, _ = on(tokens, targets)
    assert torch.equal(eval_off, eval_on)


def test_config_validation_and_inactive_marking() -> None:
    assert "decode_second_order" in test_config(decode_mode="diagonal_chunked").inactive_fields()
    assert "decode_second_order" not in test_config(decode_mode="linear").inactive_fields()
    with pytest.raises(ValueError, match="must be a plain bool"):
        test_config(decode_second_order=1)
    with pytest.raises(ValueError, match="gaussian_diagonal"):
        test_config(decode_mode="linear", decode_second_order=True, family="gaussian_full",
                    attention_mode="projected_predictive", predictive_readout="kernel")
