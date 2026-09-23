"""Is the live-geometry wired failure caused by the bf16 autocast around the fold?"""
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

import vfe3.geometry.transport as TR               # noqa: E402
from vfe3.model.model import VFEModel               # noqa: E402

_orig_build = VFEModel._build_shared_flat_transport
state = {"wire": False, "fold_fp32": False}
_orig_fold = TR.fold_rope_into_frame


def fold_fp32(base, rope):
    if state["fold_fp32"]:
        dtype = base.exp_blocks.dtype if hasattr(base, "exp_blocks") else base.exp_phi.dtype
        return _orig_fold(
            type(base)(**{f: (getattr(base, f).float() if torch.is_tensor(getattr(base, f))
                               and getattr(base, f).is_floating_point() else getattr(base, f))
                          for f in base.__dataclass_fields__}),
            rope.float())
    return _orig_fold(base, rope)


TR.fold_rope_into_frame = fold_fp32
import vfe3.geometry.transport as _t2  # noqa: E402
VFE3Config.__post_init__ = _ns["__post_init__"]


def hooked(self, beliefs, rope, **kw):
    if rope is None and state["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    return _orig_build(self, beliefs, rope, **kw)


VFEModel._build_shared_flat_transport = hooked


def trial(K, H, amp, fold32, label):
    state["wire"], state["fold_fp32"] = True, fold32
    cfg = {**LIVE, "embed_dim": K, "n_heads": H, "n_layers": 1, "max_seq_len": 8,
           "batch_size": 1, "n_e_steps": 1, "eval_stride": None, "kl_max": 8 * K,
           "pos_rotation": "rope", "amp_dtype": amp}
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**cfg))
    tok = torch.randint(0, 100, (1, 8))
    try:
        _, loss, _ = m(tok, tok)
        loss.backward()
        fin = all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        print(f"  K={K:3d} H={H} amp={str(amp):5s} fold_fp32={fold32} -> OK loss={float(loss):.6f} "
              f"finite_grads={fin}")
    except Exception as exc:                          # noqa: BLE001
        print(f"  K={K:3d} H={H} amp={str(amp):5s} fold_fp32={fold32} -> "
              f"{type(exc).__name__}: {str(exc)[:95]}")


print("wired fix, live head geometry, isolating the AMP/fold interaction")
trial(160, 5, "bf16", False, "shipped")
trial(160, 5, None,   False, "fp32 amp off")
trial(160, 5, "bf16", True,  "bf16 amp + fp32 fold")
trial(64, 4, "bf16", False, "shipped")
trial(64, 4, "bf16", True,  "bf16 amp + fp32 fold")
trial(8, 2, "bf16", False,  "shipped (tiny)")
