"""Final checks: (1) 5 AdamW steps clean, fresh models, order-controlled;
(2) live head geometry embed_dim=160/n_heads=5 with the pin removed and wiring on/off."""
import inspect
import textwrap
import sys

import torch

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\V3_Transformer")
import train_vfe3 as T
from vfe3.config import VFE3Config

LIVE = dict(T.config)

_src = textwrap.dedent(inspect.getsource(VFE3Config.__post_init__))
_src = _src.replace('"pos_rotation": "none",', '"pos_rotation": self.pos_rotation,')
_ns = dict(vars(sys.modules["vfe3.config"]))
exec(compile(_src, "<patched>", "exec"), _ns)
VFE3Config.__post_init__ = _ns["__post_init__"]

from vfe3.model.model import VFEModel          # noqa: E402

_orig = VFEModel._build_shared_flat_transport
state = {"wire": False}


def hooked(self, beliefs, rope, **kw):
    if rope is None and state["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    return _orig(self, beliefs, rope, **kw)


VFEModel._build_shared_flat_transport = hooked


def maker(**over):
    cfg = {**LIVE, **over}
    return cfg


TINY = maker(vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8,
             batch_size=1, n_e_steps=1, eval_stride=None, kl_max=64)


def five_steps(cfgd, wire, label):
    state["wire"] = wire
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**cfgd))
    tok = torch.randint(0, cfgd["vocab_size"], (1, 8))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ls = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = m(tok, tok)
        loss.backward()
        opt.step()
        ls.append(round(float(loss.detach()), 6))
    print(f"  [{label}] {ls}")
    return ls


print("(1) 5 AdamW steps, tiny dims, fresh model each arm")
a = five_steps(TINY, False, "none        ")
b = five_steps({**TINY, "pos_rotation": "rope"}, False, "rope unwired")
c = five_steps({**TINY, "pos_rotation": "rope"}, True, "rope WIRED  ")
print("   none == unwired:", a == b, "  none == wired:", a == c)

print()
print("(2) live head geometry (embed_dim=160, n_heads=5), pin removed")
LIVE_SMALL = maker(embed_dim=160, n_heads=5, n_layers=1, max_seq_len=8, batch_size=1,
                   n_e_steps=1, eval_stride=None, kl_max=8 * 160)
for wire, lab in ((False, "rope unwired"), (True, "rope WIRED")):
    state["wire"] = wire
    torch.manual_seed(0)
    try:
        m = VFEModel(VFE3Config(**{**LIVE_SMALL, "pos_rotation": "rope"}))
        n = sum(p.numel() for p in m.parameters())
        tok = torch.randint(0, LIVE_SMALL["vocab_size"], (1, 8))
        _, loss, _ = m(tok, tok)
        loss.backward()
        fin = sum(1 for p in m.parameters()
                  if p.grad is not None and torch.isfinite(p.grad).all())
        tot = sum(1 for p in m.parameters() if p.requires_grad)
        print(f"  [{lab}] OK params={n} loss={float(loss):.8f} finite_grads={fin}/{tot}")
    except Exception as exc:                             # noqa: BLE001
        print(f"  [{lab}] EXCEPTION {type(exc).__name__}: {exc}")

print()
print("(3) frame shape / block_diagonal on the tiny model")
_orig_frame = VFEModel._canonical_frame_context
VFEModel._canonical_frame_context = staticmethod(
    lambda t, **kw: (lambda ctx: (print("    frame.forward", tuple(ctx.forward.shape),
                                        "block_diagonal", ctx.block_diagonal), ctx)[1])(
        _orig_frame(t, **kw)))
state["wire"] = True
torch.manual_seed(0)
m = VFEModel(VFE3Config(**{**TINY, "pos_rotation": "rope"}))
tok = torch.randint(0, 7, (1, 8))
m.forward_beliefs(tok, return_logits=True)
