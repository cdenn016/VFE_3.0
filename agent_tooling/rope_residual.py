"""Measure the fp64 inverse residuals the CanonicalFrameContext check computes,
for the unwired vs wired frame, at tiny and live head geometry."""
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
from vfe3.geometry.transport import RopeTransport   # noqa: E402

_orig = VFEModel._build_shared_flat_transport
state = {"wire": False}
store = {}


def hooked(self, beliefs, rope, **kw):
    if rope is None and state["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    out = _orig(self, beliefs, rope, **kw)
    store["t"] = out
    store["phi_max"] = float(beliefs.phi.abs().max())
    return out


VFEModel._build_shared_flat_transport = hooked


def report(K, H, tag):
    cfg = {**LIVE, "embed_dim": K, "n_heads": H, "n_layers": 1, "max_seq_len": 8,
           "batch_size": 1, "n_e_steps": 1, "eval_stride": None, "kl_max": 8 * K,
           "pos_rotation": "rope"}
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**cfg))
    tok = torch.randint(0, 100, (1, 8))
    for wire in (False, True):
        state["wire"] = wire
        store.clear()
        try:
            m.forward_beliefs(tok, return_logits=True)
            err = "OK"
        except Exception as exc:                          # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:120]}"
        t = store.get("t")
        if t is None:
            print(f"  {tag} wire={wire}: no transport captured ({err})")
            continue
        base = t.score_operator() if isinstance(t, RopeTransport) else t
        U = base.exp_blocks.detach().to(torch.float64)
        V = base.inv_blocks.detach().to(torch.float64)
        d = U.shape[-1]
        I = torch.eye(d, dtype=torch.float64)
        prod = U @ V
        res = (prod - I).abs().amax()
        # conditioning proxy: worst row-sum scales
        fsc = U.abs().sum(-1).amax()
        isc = V.abs().sum(-1).amax()
        cond = float(fsc * isc)
        tol = min(64.0 * d * base.exp_blocks.shape[-3] * torch.finfo(base.exp_blocks.dtype).eps
                  * max(cond, 1.0), 1e-2)
        print(f"  {tag} wire={wire}: phi_max={store['phi_max']:.4f} d={d} H={base.exp_blocks.shape[-3]} "
              f"res={float(res):.3e} cond={cond:.3e} tol={tol:.3e} -> {err[:70]}")


print("tiny K=8 H=2")
report(8, 2, "tiny ")
print("live K=160 H=5")
report(160, 5, "live ")
print("intermediate K=32 H=2")
report(32, 2, "mid32")
print("intermediate K=64 H=4")
report(64, 4, "mid64")
