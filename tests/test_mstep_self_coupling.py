r"""Opt-in M-step self-coupling regularizer alpha_hat * sum_i KL(q_i*||p_i).

Manuscript Algorithm 1 (GL(K)_attention.tex:2083): the M-step loss carries a self-
coupling term alpha_hat * sum_i KL(q_i*||p_i) at the converged belief against the
per-block prior. It is wired as an OPT-IN, DEFAULT-OFF scalar coefficient
``cfg.mstep_self_coupling_weight`` (alpha_hat). These tests pin the two contracts:
(1) at weight 0 the term is absent (loss == ce), and (2) at weight w the loss is
exactly ce + w * mean KL(q*||p), with the self-divergence ``sc`` recomputed independently by the same recipe ``forward`` uses.
"""

from tests.profiles import test_config

import pytest
import torch

from vfe3.families import get_family
from vfe3.free_energy import self_divergence_for_alpha
from vfe3.model.model import VFEModel
from vfe3.contracts import MStepCapture


def _converged_self_divergence(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    r"""Independently evaluate mean KL(q*||p) from the live final-block capture."""
    cfg = model.cfg
    capture: MStepCapture = {}
    model.forward_beliefs(tokens, capture=capture)
    out = capture["converged"]
    mu_p, sigma_p = capture["final_block_prior"]
    fam = get_family(cfg.family)
    return self_divergence_for_alpha(
        fam(out.mu, out.sigma), fam(mu_p, sigma_p),
        kl_max=cfg.kl_max, eps=cfg.eps, lambda_alpha_mode=cfg.lambda_alpha_mode,
    ).mean()


def test_noop_at_weight_zero():
    # The key oracle: at weight 0 the new term is absent, so loss == ce.
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5,
                     mstep_self_coupling_weight=0.0, seed=0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, ce = model(tokens, targets)          # ce returned is ce.detach()
    assert torch.allclose(loss, ce)               # weight 0 changes nothing


def test_linear_in_weight():
    # Pins the term: loss == ce + w * mean KL(q*||p), with sc recomputed independently.
    w = 0.5
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5,
                     mstep_self_coupling_weight=w, seed=0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, ce = model(tokens, targets)
    sc = _converged_self_divergence(model, tokens)
    assert sc > 1e-6                               # non-vacuous: the belief moved off the prior
    assert torch.allclose(loss, ce + w * sc, atol=1e-6)


def _converged_self_coupling_per_coord(
    model: VFEModel, tokens: torch.Tensor, *, b0: float, c0: float,
) -> torch.Tensor:
    r"""Evaluate sum_k alpha_i^(k) D_i^(k), using alpha_i^(k)=c0/(b0+D_i^(k))."""
    cfg = model.cfg
    capture: MStepCapture = {}
    model.forward_beliefs(tokens, capture=capture)
    out = capture["converged"]
    mu_p, sigma_p = capture["final_block_prior"]
    fam = get_family(cfg.family)
    D = self_divergence_for_alpha(
        fam(out.mu, out.sigma), fam(mu_p, sigma_p),
        kl_max=cfg.kl_max, eps=cfg.eps, lambda_alpha_mode=cfg.lambda_alpha_mode,
    )
    a = (c0 / (b0 + D)).detach()
    return (a * D).sum(dim=-1).mean()


def test_per_coord_alpha_weighting():
    # Under state_dependent_per_coord the M-step self-term must carry the SAME per-token,
    # per-coordinate alpha^(k)* = c0/(b0+D^(k)) (detached) as the E-step -- NOT a flat scalar:
    #   loss == ce + w * mean_i sum_k alpha_i^(k)* D_i^(k).
    w, b0, c0 = 0.5, 0.5, 2.0
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.5,
                     attention_mode="projected_predictive",
                     lambda_alpha_mode="state_dependent_per_coord", b0=b0, c0=c0,
                     mstep_self_coupling_weight=w, seed=0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    targets = torch.randint(0, 20, (3, 5))
    _, loss, ce = model(tokens, targets)
    sc = _converged_self_coupling_per_coord(model, tokens, b0=b0, c0=c0)
    assert sc > 1e-6                               # non-vacuous: the belief moved off the prior
    assert torch.allclose(loss, ce + w * sc, atol=1e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        test_config(mstep_self_coupling_weight=-1.0)
    test_config(mstep_self_coupling_weight=0.0)     # accepted
    test_config(mstep_self_coupling_weight=0.5)     # accepted


def test_backward_finite_grads_on_prior_tables():
    # The term is grad-connected (no detach): backprop with weight>0 reaches the prior tables.
    cfg = test_config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_q_mu_lr=0.05,
                     mstep_self_coupling_weight=0.5, seed=0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4))
    targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
