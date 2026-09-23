"""gauge_transport='frozen'/'off' sever the gauge frame from autograd.

``config.__post_init__`` already coerces ``e_phi_lr=m_phi_lr=0.0`` on both ablation arms, which
skips the E-step phi substep (gated on ``e_phi_lr > 0``) and makes the M-step update a literal zero
step. What it did NOT do is stop the backward pass from building and traversing the Omega
exp/einsum adjoint for ``d/dphi`` -- that gradient was computed in full and then multiplied by
``lr=0``. ``VFEModel.__init__`` now sets ``requires_grad=False`` on the frame tables for those two
arms, so the transport build records no graph edge from the frame at all.

These contracts pin four things: the frame is severed and receives no gradient on either arm; the
default ``'on'`` arm is untouched and still receives one; the frame stays in its optimizer group
(the exact-coverage guard exempts ``requires_grad=False`` as frozen-by-design, and the phi-role
weight norm stays reportable); and the freeze is numerically inert -- a real optimizer step moves
the frame exactly as far as it moved before, that is, not at all.
"""

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

from tests.test_relational_integration import _cfg, _tokens

# The mixer and block-MLP seams are off here so the belief path is the plain relational one; this is
# the same minimal projected_filtering config the retirement suite anchors its live-frame contract
# on, which keeps the 'on' control below a known-good reference rather than a fresh guess.
def _model(**overrides: Any) -> tuple[VFEModel, Any]:
    cfg = _cfg(relational_input_scope="full", use_head_mixer=False, use_block_mlp=False, **overrides)
    torch.manual_seed(11)
    return VFEModel(cfg), cfg


def _ce_loss(model: VFEModel, vocab_size: int) -> torch.Tensor:
    ids, targets = _tokens()
    _, logits = model.forward_beliefs(ids, return_logits=True)
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.flatten())


@pytest.mark.parametrize("mode", ["frozen", "off"])
def test_frozen_and_off_sever_the_frame_from_autograd(mode: str) -> None:
    """Both ablation arms leave the frame with requires_grad=False and no gradient after backward."""
    with pytest.warns(UserWarning, match=f"gauge_transport='{mode}'"):
        model, cfg = _model(gauge_transport=mode)
    assert model.prior_bank.phi_embed.requires_grad is False
    _ce_loss(model, cfg.vocab_size).backward()
    assert model.prior_bank.phi_embed.grad is None


def test_default_on_frame_still_receives_a_gradient() -> None:
    """The control: the pure learned-frame path is untouched by the freeze, frame gradient live."""
    model, cfg = _model()                                   # gauge_transport='on' is the default
    assert cfg.gauge_transport == "on"
    assert model.prior_bank.phi_embed.requires_grad is True
    _ce_loss(model, cfg.vocab_size).backward()
    grad = model.prior_bank.phi_embed.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_frozen_frame_stays_in_the_optimizer_group() -> None:
    """The freeze is confined to autograd. build_optimizer still succeeds -- its exact-coverage
    guard exempts requires_grad=False parameters -- and the frame is still grouped, so the
    phi-role weight norm it feeds stays reportable; AdamW skips it on a None gradient."""
    with pytest.warns(UserWarning, match="gauge_transport='frozen'"):
        model, cfg = _model(gauge_transport="frozen")
    optimizer = build_optimizer(model, cfg)                 # raises if a trainable param is ungrouped
    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.prior_bank.phi_embed) in grouped
    assert any(group.get("role") == "phi" for group in optimizer.param_groups)


def test_frozen_mode_does_not_move_the_frame() -> None:
    """Numerical inertness: a real optimizer step leaves the table bit-identical, so the freeze
    cannot have changed the ablation's trajectory."""
    with pytest.warns(UserWarning, match="gauge_transport='frozen'"):
        model, cfg = _model(gauge_transport="frozen")
    before = model.prior_bank.phi_embed.detach().clone()
    assert before.abs().sum() > 0                        # 'frozen' keeps a RANDOM frame, not identity
    optimizer = build_optimizer(model, cfg)
    _ce_loss(model, cfg.vocab_size).backward()
    optimizer.step()
    assert torch.equal(model.prior_bank.phi_embed.detach(), before)


def test_off_drops_the_positional_frame() -> None:
    """'off' coerces pos_phi='none', so the positional table is never created on that arm -- there
    is no second frame for the freeze to reach."""
    with pytest.warns(UserWarning, match="gauge_transport='off'"):
        model, cfg = _model(gauge_transport="off")
    assert cfg.pos_phi == "none"
    assert getattr(model, "pos_phi_free", None) is None


def test_frozen_severs_the_positional_frame_too() -> None:
    """'frozen' keeps pos_phi='learned', so the positional table exists -- and its LR is zeroed by
    the same coercion, so it is severed on the same grounds as the token frame."""
    with pytest.warns(UserWarning, match="gauge_transport='frozen'"):
        model, cfg = _model(gauge_transport="frozen")
    assert cfg.pos_phi == "learned"
    pos = getattr(model, "pos_phi_free", None)
    assert pos is not None and pos.requires_grad is False
