"""Probe: does the train-path forward ever see the wired rope, and does its
relational_frame differ between wired/unwired?"""
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

cfg = VFE3Config(**{**BASE, "pos_rotation": "rope"})
torch.manual_seed(0)
model = VFEModel(cfg)
tokens = torch.randint(0, 7, (1, 8))

_orig_build = VFEModel._build_shared_flat_transport
_orig_frame = VFEModel._canonical_frame_context

log = []
mode = {"wire": False}


def build(self, beliefs, rope, **kw):
    if rope is None and mode["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
        log.append("build: rope SUBSTITUTED")
    else:
        log.append(f"build: rope_arg_is_none={rope is None}")
    return _orig_build(self, beliefs, rope, **kw)


def frame(transport, **kw):
    ctx = _orig_frame(transport, **kw)
    log.append(f"frame: transport={type(transport).__name__} fold_rope={kw.get('fold_rope')} "
               f"block_diag={getattr(ctx, 'block_diagonal', None)}")
    frame.last = ctx
    return ctx


VFEModel._build_shared_flat_transport = build
VFEModel._canonical_frame_context = staticmethod(frame)


def run(label, train_path, wire):
    mode["wire"] = wire
    log.clear()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    if train_path:
        _, loss, _ = model(tokens, tokens)
        out = ("loss", float(loss))
    else:
        b, lg = model.forward_beliefs(tokens, return_logits=True)
        out = ("mu_max", float(b.mu.abs().max()), "logits_max", float(lg.abs().max()))
    fwd = None
    if getattr(frame, "last", None) is not None:
        fwd = frame.last.forward.detach().clone()
    print(f"--- {label} (train_path={train_path}, wire={wire})")
    for line in log:
        print("   ", line)
    print("    frame.forward[:1,:2,0] =",
          None if fwd is None else fwd.reshape(-1, fwd.shape[-1])[:2].tolist())
    print("    out:", out)
    return fwd


f_inf_none = run("inference unwired", False, False)
f_inf_wire = run("inference wired", False, True)
print("inference: frame equal unwired vs wired:", torch.equal(f_inf_none, f_inf_wire))
print()
f_tr_none = run("train unwired", True, False)
f_tr_wire = run("train wired", True, True)
print("train: frame equal unwired vs wired:",
      None if (f_tr_none is None or f_tr_wire is None) else torch.equal(f_tr_none, f_tr_wire))
