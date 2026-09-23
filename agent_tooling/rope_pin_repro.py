"""Adversarial reproduction: pos_rotation='rope' under the projected_filtering envelope.

Scratch only. Does not modify tracked files. Monkeypatches VFE3Config.__post_init__
IN MEMORY (source-rewrite of the local `requirements` pin) so the pin can be removed
without touching disk.
"""
import inspect
import textwrap
import sys
import traceback

import torch

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\V3_Transformer")

import train_vfe3 as T                                  # real live config dict
from vfe3.config import VFE3Config

LIVE = dict(T.config)

TINY = dict(
    vocab_size=7,
    embed_dim=8,
    n_heads=2,
    n_layers=1,
    max_seq_len=8,
    batch_size=1,
    n_e_steps=1,
    eval_stride=None,
)
TINY["kl_max"] = 8 * TINY["embed_dim"]

BASE = {**LIVE, **TINY}


def build(pos_rotation):
    cfg = dict(BASE)
    cfg["pos_rotation"] = pos_rotation
    return VFE3Config(**cfg)


print("=" * 78)
print("PASS 1: real train_vfe3.py config + tiny dims, pos_rotation='rope'")
print("=" * 78)
try:
    c = build("rope")
    print("NO EXCEPTION -- constructed ok, pos_rotation =", c.pos_rotation)
except Exception as exc:                                 # noqa: BLE001
    print("EXCEPTION TYPE:", type(exc).__name__)
    print("MESSAGE:", str(exc))

print()
print("=" * 78)
print("PASS 2: same config, pos_rotation='none' (baseline)")
print("=" * 78)
try:
    c = build("none")
    print("CONSTRUCTED OK, pos_rotation =", c.pos_rotation)
    print("  family            =", c.family)
    print("  transport_mode    =", c.transport_mode)
    print("  e_step_gradient   =", c.e_step_gradient)
    print("  oracle_unroll_grad=", c.oracle_unroll_grad)
    print("  include_attn_ent  =", c.include_attention_entropy)
    print("  gradient_mode     =", c.gradient_mode)
    print("  relational_mode   =", c.relational_mode)
    print("  readout/output    =", c.relational_readout_mode, c.relational_output_scope)
    print("  scope             =", c.relational_input_scope)
    print("  identity_content  =", c.relational_identity_content)
    print("  n_heads/embed_dim =", c.n_heads, c.embed_dim)
    print("  block_mlp_mode    =", c.block_mlp_mode, "use_block_mlp =", c.use_block_mlp)
    print("  rope_insertion    =", c.rope_insertion, "rope_full_gauge =", c.rope_full_gauge)
except Exception as exc:                                 # noqa: BLE001
    print("UNEXPECTED EXCEPTION:", type(exc).__name__, str(exc))
    traceback.print_exc()

print()
print("=" * 78)
print("PASS 3: remove the pos_rotation pin IN MEMORY (source rewrite), rope kept")
print("=" * 78)

_ORIG = VFE3Config.__post_init__
_src = textwrap.dedent(inspect.getsource(_ORIG))
_needle = '"pos_rotation": "none",'
if _needle not in _src:
    raise SystemExit("needle not found -- pin already absent?")
_patched = _src.replace(_needle, '"pos_rotation": self.pos_rotation,')
_ns = dict(vars(sys.modules["vfe3.config"]))
exec(compile(_patched, "<patched_post_init>", "exec"), _ns)
VFE3Config.__post_init__ = _ns["__post_init__"]
print("patched __post_init__ (in memory only)")

try:
    cfg_rope = build("rope")
    print("CONSTRUCTED OK with pin removed, pos_rotation =", cfg_rope.pos_rotation)
    cfg_none = build("none")
    print("CONSTRUCTED OK pos_rotation='none' (control)")
except Exception as exc:                                 # noqa: BLE001
    print("SECOND EXCEPTION (pin was NOT the only blocker):", type(exc).__name__)
    print("MESSAGE:", str(exc))
    traceback.print_exc()
    raise SystemExit(1)

import warnings
warnings.simplefilter("always")

from vfe3.model.model import VFEModel                            # noqa: E402

torch.manual_seed(0)
model = VFEModel(cfg_rope)
n = sum(p.numel() for p in model.parameters())
print(f"model built: {n} params")

tokens = torch.randint(0, TINY["vocab_size"], (1, 8))

print()
print("PASS 3b: forward+backward with pos_rotation='rope', pin removed")
try:
    torch.manual_seed(1)
    logits, loss, ce = model(tokens, tokens)
    print("forward+backward OK. loss =", float(loss), " ce =", float(ce))
    print("logits shape:", None if logits is None else tuple(logits.shape))
    loss.backward()
    nz = sum(1 for p in model.parameters()
             if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    fin = sum(1 for p in model.parameters()
              if p.grad is not None and torch.isfinite(p.grad).all())
    tot = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"params with finite grad: {fin} / {tot}; nonzero: {nz}")
except Exception as exc:                                 # noqa: BLE001
    print("SECOND EXCEPTION on forward/backward:", type(exc).__name__)
    print("MESSAGE:", str(exc))
    traceback.print_exc()
