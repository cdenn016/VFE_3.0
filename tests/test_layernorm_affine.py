r"""Learned per-feature affine LayerNorm (layernorm_affine).

layernorm_affine=True adds a learned gamma/beta to the "layernorm" block norm via the
AffineLayerNorm nn.Module (mu_norm = gamma*LN(mu) + beta). Default-ON member of the
learned-parameter exception family: gamma inits to 1 and beta to 0, so step 0 is
byte-identical to the parameter-free "layernorm". ``_cfg`` pins the parameter-free variant
(layernorm_affine=False) and the block MLP off, so each test opts into exactly the seam it checks. Non-gauge-equivariant (the same break
"layernorm" itself carries). The block norm trains through the supported unrolled E-step.

All models here are tiny (K = 4, single-digit dims), CPU-bound per the project testing rules.
"""

from tests.profiles import test_config

import pytest
import torch

from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
from vfe3.model.model import VFEModel


def _cfg(**kw):
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=2,
                n_e_steps=1, m_phi_lr=0.0,
                layernorm_affine=False, use_block_mlp=False)
    base.update(kw)
    return test_config(**base)


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 12, (2, 8), generator=g),
            torch.randint(0, 12, (2, 8), generator=g))


# --- defaults / pure path ----------------------------------------------------

def test_layernorm_affine_can_be_selected():
    assert test_config(layernorm_affine=True).layernorm_affine is True


def test_param_free_when_affine_off():
    # norm_type=layernorm but affine off -> plain (param-free) LayerNorm, NOT an nn.Module, no params.
    m = VFEModel(_cfg(norm_type_block="layernorm"))
    assert isinstance(m.block_norm, LayerNorm) and not isinstance(m.block_norm, torch.nn.Module)
    assert not any(n.startswith("block_norm") for n in dict(m.named_parameters()))


# --- param creation ----------------------------------------------------------

def test_affine_params_created_and_registered():
    m = VFEModel(_cfg(norm_type_block="layernorm", layernorm_affine=True))
    assert isinstance(m.block_norm, AffineLayerNorm)
    names = dict(m.named_parameters())
    for key in ("block_norm.weight", "block_norm.bias"):
        assert key in names and names[key].shape == (4,)              # per-feature (K = embed_dim = 4)
    assert torch.equal(m.block_norm.weight, torch.ones(4))            # gamma init 1
    assert torch.equal(m.block_norm.bias, torch.zeros(4))            # beta init 0


# --- step-0 byte-identity ----------------------------------------------------

def test_step0_byte_identity():
    # gamma=1, beta=0 -> the affine model's forward/loss is byte-identical to the parameter-free
    # "layernorm" at construction (affine param creation draws zero RNG: torch.ones/zeros).
    x, y = _batch()
    torch.manual_seed(0)
    m_on = VFEModel(_cfg(norm_type_block="layernorm", layernorm_affine=True))
    torch.manual_seed(0)
    m_off = VFEModel(_cfg(norm_type_block="layernorm"))
    assert torch.equal(m_on(x), m_off(x))
    _, loss_on, _ = m_on(x, y)
    _, loss_off, _ = m_off(x, y)
    assert torch.equal(loss_on, loss_off)


def test_perturbed_affine_changes_loss():
    # The affine is live: shifting beta changes the decoded logits and therefore the loss.
    x, y = _batch()
    torch.manual_seed(0)
    m = VFEModel(_cfg(norm_type_block="layernorm", layernorm_affine=True))
    _, loss1, _ = m(x, y)
    with torch.no_grad():
        m.block_norm.bias.add_(1.0)
    _, loss2, _ = m(x, y)
    assert not torch.equal(loss1, loss2)


# --- gradient flow / freeze --------------------------------------------------

def test_block_affine_trains_under_unroll():
    # Canonical 'unroll' E-step (default): the block norm's gamma/beta receive a real gradient.
    x, y = _batch()
    torch.manual_seed(0)
    m = VFEModel(_cfg(norm_type_block="layernorm", layernorm_affine=True))
    _, loss, _ = m(x, y)
    loss.backward()
    assert m.block_norm.weight.grad is not None and m.block_norm.weight.grad.abs().sum() > 0
    assert m.block_norm.bias.grad is not None


def test_inert_warning_when_no_layernorm_seam():
    # layernorm_affine=True but no "layernorm" block norm -> no affine params created, toggle inert (warn).
    with pytest.warns(UserWarning, match="inert"):
        VFEModel(_cfg(norm_type_block="mahalanobis", layernorm_affine=True))


# --- optimizer wiring --------------------------------------------------------

def test_optimizer_grouping():
    # Both affine params land in exactly one group: role='mu', weight_decay=0.0, no gauge flag;
    # the exact-coverage guard in build_optimizer passes.
    from vfe3.train import build_optimizer
    torch.manual_seed(0)
    m = VFEModel(_cfg(norm_type_block="layernorm", layernorm_affine=True))
    opt = build_optimizer(m, m.cfg)                                   # coverage guard raises if ungrouped
    for p in (m.block_norm.weight, m.block_norm.bias):
        gs = [g for g in opt.param_groups if any(q is p for q in g["params"])]
        assert len(gs) == 1
        g = gs[0]
        assert g["lr"] == m.cfg.m_p_mu_lr
        assert g["weight_decay"] == 0.0
        assert g["role"] == "mu"
        assert not g.get("gauge", False)


def test_training_updates_affine():
    # A few AdamW steps move the block-norm gamma/beta (end-to-end optimizer path).
    from vfe3.train import build_optimizer
    torch.manual_seed(0)
    m = VFEModel(_cfg(norm_type_block="layernorm", layernorm_affine=True))
    opt = build_optimizer(m, m.cfg)
    w0 = m.block_norm.weight.detach().clone()
    b0 = m.block_norm.bias.detach().clone()
    x, y = _batch()
    for _ in range(3):
        opt.zero_grad()
        _, loss, _ = m(x, y)
        loss.backward()
        opt.step()
    assert (not torch.equal(m.block_norm.weight.detach(), w0)
            or not torch.equal(m.block_norm.bias.detach(), b0))
