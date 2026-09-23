"""Decisive minimal run: inference + train loss for none / rope-unwired / rope-wired.
Fresh process, no frame monkeypatch, order-controlled.
"""
import inspect
import textwrap
import sys

import torch

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\V3_Transformer")
import train_vfe3 as T
from vfe3.config import VFE3Config

LIVE = dict(T.config)
TINY = dict(vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8,
            batch_size=1, n_e_steps=1, eval_stride=None)
TINY["kl_max"] = 8 * TINY["embed_dim"]
BASE = {**LIVE, **TINY}

_src = textwrap.dedent(inspect.getsource(VFE3Config.__post_init__))
_src = _src.replace('"pos_rotation": "none",', '"pos_rotation": self.pos_rotation,')
_ns = dict(vars(sys.modules["vfe3.config"]))
exec(compile(_src, "<patched>", "exec"), _ns)
VFE3Config.__post_init__ = _ns["__post_init__"]

from vfe3.model.model import VFEModel          # noqa: E402

_orig = VFEModel._build_shared_flat_transport
state = {"wire": False}
seen = []


def hooked(self, beliefs, rope, **kw):
    if rope is None and state["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
        seen.append("substituted")
    else:
        seen.append(f"none={rope is None}")
    return _orig(self, beliefs, rope, **kw)


VFEModel._build_shared_flat_transport = hooked

tokens = torch.randint(0, 7, (1, 8))


def build(pos):
    torch.manual_seed(0)
    return VFEModel(VFE3Config(**{**BASE, "pos_rotation": pos}))


for pos, wire, label in (("none", False, "none"), ("rope", False, "rope unwired"),
                         ("rope", True, "rope WIRED")):
    state["wire"] = wire
    model = build(pos)
    seen.clear()
    model.zero_grad(set_to_none=True)
    b, lg = model.forward_beliefs(tokens, return_logits=True)
    inf = (float(b.mu.abs().max()), float(lg.abs().max()))
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(tokens, tokens)
    loss.backward()
    rel = sorted((n, round(float(p.grad.norm()), 6)) for n, p in model.named_parameters()
                 if p.grad is not None and ("relational" in n or "score" in n))[:4]
    print(f"{label:12s} build_calls={seen} inf(mu_max,logit_max)={inf} "
          f"train_loss={float(loss):.8f} relgrads={rel}")
