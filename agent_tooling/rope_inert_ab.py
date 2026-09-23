"""Adversarial check of the inertness claim + the wiring-fix hypothesis.

Everything is monkeypatched in memory; no tracked file is touched.
"""
import inspect
import textwrap
import sys

import torch

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\V3_Transformer")

import train_vfe3 as T
from vfe3.config import VFE3Config
from vfe3.geometry.transport import RopeTransport

LIVE = dict(T.config)
TINY = dict(vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8,
            batch_size=1, n_e_steps=1, eval_stride=None)
TINY["kl_max"] = 8 * TINY["embed_dim"]
BASE = {**LIVE, **TINY}

# ---- remove the pin in memory (source rewrite) --------------------------------
_src = textwrap.dedent(inspect.getsource(VFE3Config.__post_init__))
_src = _src.replace('"pos_rotation": "none",', '"pos_rotation": self.pos_rotation,')
_ns = dict(vars(sys.modules["vfe3.config"]))
exec(compile(_src, "<patched>", "exec"), _ns)
VFE3Config.__post_init__ = _ns["__post_init__"]

from vfe3.model.model import VFEModel          # noqa: E402


def make(pos_rotation, seed=0):
    cfg = VFE3Config(**{**BASE, "pos_rotation": pos_rotation})
    torch.manual_seed(seed)
    return VFEModel(cfg)


tokens = torch.randint(0, 7, (1, 8))

# ---- A) parameter identity ---------------------------------------------------
m_rope = make("rope")
m_none = make("none")
same_params = all(
    torch.equal(a, b) for a, b in zip(m_rope.parameters(), m_none.parameters()))
print("A) identical params at build (same seed):", same_params)

# ---- B) instrument the transport build site ----------------------------------
calls = []
_orig_build = VFEModel._build_shared_flat_transport


def spy_build(self, beliefs, rope, **kw):
    out = _orig_build(self, beliefs, rope, **kw)
    calls.append((rope is None, type(out).__name__))
    return out


VFEModel._build_shared_flat_transport = spy_build


def run(model, label, wire_rope=False):
    calls.clear()
    if wire_rope:
        def wired(self, beliefs, rope, **kw):
            if rope is None:
                n = beliefs.mu.shape[-2]
                rope = self._rope_rotation(n, beliefs.mu.device)
            return _orig_build(self, beliefs, rope, **kw)
        VFEModel._build_shared_flat_transport = wired
    else:
        VFEModel._build_shared_flat_transport = spy_build

    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    beliefs, logits = model.forward_beliefs(tokens, return_logits=True)
    print(f"  [{label}] build_shared_flat_transport calls (rope is None, type): {list(calls)}")
    return beliefs, logits


b_rope, l_rope = run(m_rope, "rope as-shipped")
b_none, l_none = run(m_none, "none")

print("B) AS-SHIPPED rope vs none")
print("   max|mu_rope - mu_none|     =", float((b_rope.mu - b_none.mu).abs().max()))
print("   max|logits_rope - logits_none| =", float((l_rope - l_none).abs().max()))
print("   torch.equal(mu):", torch.equal(b_rope.mu, b_none.mu),
      " torch.equal(logits):", torch.equal(l_rope, l_none))

# ---- C) wiring fix: substitute the model's cached rope at the build site -----
w_rope, wl_rope = run(m_rope, "rope wired", wire_rope=True)
print("C) wired rope vs none")
print("   build site now sees:", list(calls))
print("   max|mu_wired - mu_none|       =", float((w_rope.mu - b_none.mu).abs().max()))
print("   max|logits_wired - logits_none| =", float((wl_rope - l_none).abs().max()))
print("   torch.equal(mu):", torch.equal(w_rope.mu, b_none.mu),
      " torch.equal(logits):", torch.equal(wl_rope, l_none))

# ---- D) trained loss + grads, all three arms ---------------------------------
def train_loss_and_grads(model, label, wire_rope=False):
    if wire_rope:
        def wired(self, beliefs, rope, **kw):
            if rope is None:
                n = beliefs.mu.shape[-2]
                rope = self._rope_rotation(n, beliefs.mu.device)
            return _orig_build(self, beliefs, rope, **kw)
        VFEModel._build_shared_flat_transport = wired
    else:
        VFEModel._build_shared_flat_transport = spy_build
    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    _, loss, ce = model(tokens, tokens)
    loss.backward()
    rel_grads = [
        float(p.grad.norm()) for name, p in model.named_parameters()
        if ("relational" in name or "score" in name) and p.grad is not None]
    print(f"  [{label}] loss = {float(loss):.8f}  rel/scores grads = "
          f"{[round(g, 6) for g in rel_grads[:6]]}")
    return float(loss), rel_grads


print("D) training loss + gradients (train forward)")
l_none_v, g_none = train_loss_and_grads(m_none, "none")
l_rope_v, g_rope = train_loss_and_grads(m_rope, "rope as-shipped")
l_wire_v, g_wire = train_loss_and_grads(m_rope, "rope wired")
print("   loss none vs as-shipped equal:", l_none_v == l_rope_v)
print("   loss none vs wired equal:", l_none_v == l_wire_v)

# ---- E) optimizer steps ------------------------------------------------------
VFEModel._build_shared_flat_transport = spy_build


def steps(model, label, wire_rope=False):
    if wire_rope:
        def wired(self, beliefs, rope, **kw):
            if rope is None:
                n = beliefs.mu.shape[-2]
                rope = self._rope_rotation(n, beliefs.mu.device)
            return _orig_build(self, beliefs, rope, **kw)
        VFEModel._build_shared_flat_transport = wired
    else:
        VFEModel._build_shared_flat_transport = spy_build
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    torch.manual_seed(3)
    ls = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        opt.step()
        ls.append(round(float(loss.detach()), 6))
    print(f"  [{label}] 5 AdamW step losses = {ls}")


print("E) multi-step")
steps(make("none"), "none")
steps(make("rope"), "rope as-shipped")
steps(make("rope"), "rope wired", wire_rope=True)
