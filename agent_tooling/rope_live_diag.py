"""Diagnose the live-geometry fold failure: is the rope block structure aligned with
the transport's head blocks?"""
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
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import fold_rope_into_frame   # noqa: E402

for K, H in ((8, 2), (16, 2), (32, 2), (64, 4), (160, 5)):
    g = get_group("block_glk")(K=K, n_heads=H)
    print(f"K={K:4d} H={H}  irrep_dims={g.irrep_dims}  phi_layout={g.phi_coordinate_layout}")

print()
K, H = 160, 5
cfg = {**LIVE, "embed_dim": K, "n_heads": H, "n_layers": 1, "max_seq_len": 8,
       "batch_size": 1, "n_e_steps": 1, "eval_stride": None, "kl_max": 8 * K,
       "pos_rotation": "rope"}
torch.manual_seed(0)
m = VFEModel(VFE3Config(**cfg))

_orig = VFEModel._build_shared_flat_transport


def hooked(self, beliefs, rope, **kw):
    if rope is None:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    out = _orig(self, beliefs, rope, **kw)
    hooked.last = out
    return out


VFEModel._build_shared_flat_transport = hooked
tok = torch.randint(0, 100, (1, 8))
try:
    m.forward_beliefs(tok, return_logits=True)
except Exception as exc:                                 # noqa: BLE001
    print("forward raised:", type(exc).__name__, str(exc)[:160])

t = hooked.last
print("transport type:", type(t).__name__)
print("  rope shape:", tuple(t.rope.shape))
print("  base:", type(t.base).__name__, "exp_blocks", tuple(t.base.exp_blocks.shape),
      "inv_blocks", tuple(t.base.inv_blocks.shape))
base = t.score_operator()
print("  folded:", type(base).__name__, tuple(base.exp_blocks.shape))
d = base.exp_blocks.shape[-1]
R = t.rope
nq = base.exp_blocks.shape[-4]
print("  n_query(token axis) =", nq, " rope N =", R.shape[-3], " d =", d, " H =",
      base.exp_blocks.shape[-3])

# Block-alignment test: is R^T R block-diagonal at multiples of d only?
Rt = R.transpose(-1, -2)
prod = Rt @ R
off = prod - torch.eye(K, dtype=prod.dtype)
# residual of forward @ inverse for the folded transport
fi = base.exp_blocks @ base.inv_blocks
res = (fi - torch.eye(d, dtype=fi.dtype)).abs().amax(dim=(-2, -1))
print("  max |R^T R - I|          =", float(off.abs().max()))
print("  max |U R^T R U^-1 - I|   =", float(res.max()))
print("  tolerance (dim=K, fp32)  =", 64.0 * d * base.exp_blocks.shape[-3]
      * torch.finfo(torch.float32).eps)

# how much of the residual is inside a single head block vs across head blocks?
per_head = (fi - torch.eye(d, dtype=fi.dtype)).abs().amax(dim=(-2, -1))
print("  residual per (batch,token,head):", [round(float(x), 6) for x in per_head.reshape(-1)[:10]])

# Compare with a direct, alignment-free fold using the true group blocks
print()
print("  direct check: fold with per-head slices of R at offsets h*d")
for h in range(base.exp_blocks.shape[-3]):
    blk = Rt[..., h * d:(h + 1) * d, h * d:(h + 1) * d]
    orth = (blk.transpose(-1, -2) @ blk - torch.eye(d, dtype=blk.dtype)).abs().max()
    print(f"    head {h}: slice offset {h*d:3d}  max|B^T B - I| = {float(orth):.3e}")

# Is R actually block-diagonal on the GROUP's irrep_dims (32) or on something else?
g = get_group("block_glk")(K=K, n_heads=H)
print("  group irrep_dims:", g.irrep_dims)
