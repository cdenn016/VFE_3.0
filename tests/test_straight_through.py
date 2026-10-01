r"""Straight-through E-step gradient mode (manuscript Algorithm 1, GL(K)_attention.tex:2050).

The estimator: each inner E-step update computes its tangent (delta) DETACHED, but rebuilds
the belief grad-connected to the PREVIOUS belief (mu_next = mu_prev + delta.detach(),
sigma_next = retract(sigma_prev, delta.detach())), so d belief_next/d belief_prev = I flows
WITHOUT the second-order d delta/d belief_prev term that full unroll keeps. The phi step is
already straight-through; this brings the mu/sigma updates to match.

Oracle: straight_through changes only the BACKWARD, never the forward, so the converged
belief / logits / loss VALUE is byte-identical to unroll (torch.equal). The gradient flows,
but DIFFERS from unroll (the dropped second-order term).
"""
from tests.profiles import test_config

import pytest
import torch

from vfe3.model.model import VFEModel


def _model(e_step_gradient="unroll", **over):
    cfg = dict(
        vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=2, e_q_mu_lr=0.1, e_q_sigma_lr=0.02, e_phi_lr=0.0,
        gradient_mode="filtering", e_step_gradient=e_step_gradient,
        decode_mode="diagonal_chunked",  # observe encode tables via KL-to-prior decode
    )
    cfg.update(over)
    return VFEModel(test_config(**cfg))


def _data(seed=0, V=15, B=2, N=4):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, V, (B, N), generator=g)
    targets = torch.randint(0, V, (B, N), generator=g)
    return tokens, targets


# --- config validation -----------------------------------------------------
def test_config_accepts_the_three_modes():
    # The block MLP (on by default) runs inside the no-grad detach stack and would freeze, so
    # detach is admitted only with it off; the fail-closed gate itself is pinned below.
    for mode in ("unroll", "straight_through", "detach"):
        cfg = test_config(e_step_gradient=mode, use_block_mlp=False)
        assert cfg.e_step_gradient == mode
    with pytest.raises(ValueError, match="gradient-carrying E-step"):
        test_config(e_step_gradient="detach", use_block_mlp=True)


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError):
        test_config(e_step_gradient="not_a_mode")


def test_config_explicit_unroll_is_preserved():
    assert test_config(e_step_gradient="unroll").e_step_gradient == "unroll"






def test_config_mode_passthrough_is_canonical():
    assert test_config(e_step_gradient="unroll").e_step_gradient == "unroll"
    assert test_config(e_step_gradient="straight_through").e_step_gradient == "straight_through"
    assert test_config(e_step_gradient="detach", use_block_mlp=False).e_step_gradient == "detach"


# --- forward identity (the strong oracle) -----------------------------------
def test_straight_through_forward_byte_identical_to_unroll():
    # straight_through only changes the BACKWARD: mu_prev + delta is the same NUMBER whether
    # or not delta's graph is tracked, so converged logits/loss are byte-identical (torch.equal).
    tokens, targets = _data()
    m_un = _model("unroll")
    m_st = _model("straight_through")
    # same parameters (same seed in __init__ -> PriorBank init is seeded by run_training, not
    # here, so copy state to force identical weights).
    m_st.load_state_dict(m_un.state_dict())

    logits_un = m_un(tokens)[0]
    logits_st = m_st(tokens)[0]
    _, loss_un, ce_un = m_un(tokens, targets)
    _, loss_st, ce_st = m_st(tokens, targets)
    assert m_un.cfg.e_step_gradient == "unroll"
    assert m_st.cfg.e_step_gradient == "straight_through"
    assert torch.equal(logits_un, logits_st)
    assert torch.equal(loss_un, loss_st)
    assert torch.equal(ce_un, ce_st)


def test_straight_through_grad_flows_to_encode_tables():
    tokens, targets = _data()
    model = _model("straight_through")
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
    assert model.prior_bank.sigma_log_embed.grad is not None
    assert torch.isfinite(model.prior_bank.sigma_log_embed.grad).all()


# --- gradients DIFFER from unroll (distinct estimators) ---------------------
def test_straight_through_grad_differs_from_unroll():
    # unroll keeps the second-order d delta/d belief term; straight_through drops it. With a
    # config where the delta depends on the belief (e_q_mu_lr>0, e_q_sigma_lr>0, n_e_steps>=2 to
    # compound the second-order term), the mu_embed.grad must DIFFER between the two estimators.
    tokens, targets = _data()
    m_un = _model("unroll")
    m_st = _model("straight_through")
    m_st.load_state_dict(m_un.state_dict())

    _, loss_un, _ = m_un(tokens, targets)
    loss_un.backward()
    g_un = m_un.prior_bank.mu_embed.grad.clone()

    _, loss_st, _ = m_st(tokens, targets)
    loss_st.backward()
    g_st = m_st.prior_bank.mu_embed.grad.clone()

    assert not torch.allclose(g_st, g_un)


