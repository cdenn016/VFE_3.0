r"""mu_weight_decay: a SEPARATE AdamW weight decay for the MEAN-role capacity tables, the exact
mirror of sigma_weight_decay (default None = inherit the global weight_decay).

A dead TENSOR escapes AdamW decay, because zero_grad(set_to_none=True) leaves p.grad = None and
AdamW skips such parameters. A rarely-visited ROW inside a LIVE tensor does not: the embedding
gather produces a dense gradient, so decoupled decay reaches all V rows on every step whether or
not the token appeared. On the K=300 wikitext-103 run that drove the zero-count rows of the mean
tables to norm 0.000 and the count-1..15 rows below their random initialization. mu_weight_decay=0.0
exempts the mean sector; the frequent rows keep their dense likelihood gradient either way.

The field must leave the shipped default byte-identical (None = inherit), and must reach EVERY
mean-role capacity table.
"""

from tests.profiles import test_config

import torch

from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

BASE = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
            weight_decay=0.05, sigma_weight_decay=None)


def _group_of(opt, param):
    gs = [g for g in opt.param_groups if any(p is param for p in g["params"])]
    assert len(gs) == 1, f"expected exactly one group for the param, got {len(gs)}"
    return gs[0]


def _wd(group, opt):
    return group.get("weight_decay", opt.defaults["weight_decay"])


def test_mu_weight_decay_none_inherits():
    cfg = test_config(**BASE, mu_weight_decay=None)
    assert cfg.mu_weight_decay is None
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert _wd(_group_of(opt, model.prior_bank.mu_embed), opt) == cfg.weight_decay


def test_mu_weight_decay_zero_exempts_the_mean_sector():
    cfg = test_config(**BASE, mu_weight_decay=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert _wd(_group_of(opt, model.prior_bank.mu_embed), opt) == 0.0
    # the sigma sector is untouched by the mean-sector knob
    assert _wd(_group_of(opt, model.prior_bank.sigma_log_embed), opt) == cfg.weight_decay


def test_zero_decay_leaves_an_unvisited_row_untouched():
    """The mechanism, end to end: a row with no gradient signal must not shrink under
    mu_weight_decay=0.0, while it does shrink when the global decay is inherited."""
    for wd, expect_shrink in ((None, True), (0.0, False)):
        torch.manual_seed(0)
        kwargs = dict(BASE, mu_weight_decay=wd)
        cfg = test_config(**kwargs)
        model = VFEModel(cfg)
        opt = build_optimizer(model, cfg)
        table = model.prior_bank.mu_embed
        before = table[7].norm().item()
        # a dense gradient that is exactly zero on row 7: the "token never appeared" case
        grad = torch.ones_like(table)
        grad[7] = 0.0
        table.grad = grad
        for _ in range(5):
            opt.step()
        after = table[7].norm().item()
        if expect_shrink:
            assert after < before, "inherited decay must reach the unvisited row"
        else:
            assert after == before, f"mu_weight_decay=0.0 must leave it untouched, {before} -> {after}"
