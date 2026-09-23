r"""phi_weight_decay: a SEPARATE AdamW weight decay for the gauge-frame coordinate tables
(phi_embed and the learned pos_phi_free), default 0.065.

Decoupled AdamW decay on phi sets an LR-invariant ceiling on the frame norm
(|phi*| ~ E[normalized-grad]/wd), pulling the gauge transport exp(phi.G) toward the identity
independently of the M-step LRs. The gauge frames are protected at weight_decay=0; this field
makes that protection a first-class, sweepable knob (set phi_weight_decay=0 for full protection)
without changing the generic weight_decay on the belief tables. Under pullback-group descent phi is
stepped by a local group retraction, so its AdamW decay stays 0 regardless of the field.
"""

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)


def _group_of(opt, param):
    gs = [g for g in opt.param_groups if any(p is param for p in g["params"])]
    assert len(gs) == 1, f"expected exactly one group for the param, got {len(gs)}"
    return gs[0]


def _wd(group, opt):
    return group.get("weight_decay", opt.defaults["weight_decay"])


def test_phi_weight_decay_default_is_0p065():
    assert VFE3Config(**BASE).phi_weight_decay == 0.065


def test_phi_decay_distinct_from_generic_by_default():
    cfg = VFE3Config(**BASE)                                     # weight_decay 0.05, phi_weight_decay 0.065
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert _wd(_group_of(opt, model.prior_bank.phi_embed), opt) == 0.065   # phi: own decay
    assert _wd(_group_of(opt, model.prior_bank.mu_embed), opt) == 0.05     # belief tables: generic
    model.prior_bank.phi_embed.grad = torch.ones_like(model.prior_bank.phi_embed)
    opt.step()
    assert set(opt.state[model.prior_bank.phi_embed]) == {"step", "exp_avg", "exp_avg_sq"}


def test_phi_weight_decay_override_protects_phi_only():
    cfg = VFE3Config(**BASE, weight_decay=0.05, phi_weight_decay=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert _wd(_group_of(opt, model.prior_bank.phi_embed), opt) == 0.0     # phi protected (weight_decay=0)
    assert _wd(_group_of(opt, model.prior_bank.mu_embed), opt) == 0.05     # mu untouched


def test_pos_phi_free_uses_phi_weight_decay():
    cfg = VFE3Config(**BASE, pos_phi="learned", phi_weight_decay=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert _wd(_group_of(opt, model.pos_phi_free), opt) == 0.0             # pos frame is a gauge frame too


def test_pullback_group_forces_phi_decay_zero_regardless_of_field():
    cfg = VFE3Config(**BASE, gauge_group="block_glk", m_phi_update_mode="pullback_group",
                     phi_precond_mode="pullback_per_block", transport_chart_max_norm=6.0,
                     phi_weight_decay=0.065)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    group = _group_of(opt, model.prior_bank.phi_embed)
    assert _wd(group, opt) == 0.0
    assert group["pullback_group"] is True
    model.prior_bank.phi_embed.grad = torch.zeros_like(model.prior_bank.phi_embed)
    model.prior_bank.phi_embed.grad[0, 0] = 1.0
    opt.step()
    assert model.prior_bank.phi_embed not in opt.state
