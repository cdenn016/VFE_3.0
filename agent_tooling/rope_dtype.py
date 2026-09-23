import inspect, textwrap, sys, torch
sys.path.insert(0, r"C:\Users\chris and christine\Desktop\V3_Transformer")
import train_vfe3 as T
from vfe3.config import VFE3Config
LIVE = dict(T.config)
_src = textwrap.dedent(inspect.getsource(VFE3Config.__post_init__)).replace(
    '"pos_rotation": "none",', '"pos_rotation": self.pos_rotation,')
_ns = dict(vars(sys.modules["vfe3.config"]))
exec(compile(_src, "<patched>", "exec"), _ns)
VFE3Config.__post_init__ = _ns["__post_init__"]
from vfe3.model.model import VFEModel
from vfe3.geometry.transport import RopeTransport
_orig = VFEModel._build_shared_flat_transport
st = {"wire": False}
store = {}
def hooked(self, beliefs, rope, **kw):
    if rope is None and st["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    out = _orig(self, beliefs, rope, **kw)
    store["t"] = out
    return out
VFEModel._build_shared_flat_transport = hooked

for amp in ("bf16", None):
    st["wire"] = True
    cfg = {**LIVE, "embed_dim": 160, "n_heads": 5, "n_layers": 1, "max_seq_len": 8,
           "batch_size": 1, "n_e_steps": 1, "eval_stride": None, "kl_max": 1280,
           "pos_rotation": "rope", "amp_dtype": amp}
    torch.manual_seed(0)
    m = VFEModel(VFE3Config(**cfg))
    tok = torch.randint(0, 100, (1, 8))
    try:
        m.forward_beliefs(tok, return_logits=True); err = "OK"
    except Exception as e:
        err = f"{type(e).__name__}"
    t = store["t"]
    base = t.base
    folded = t.score_operator()
    print(f"amp={str(amp):5s} {err:12s} unfolded(U) dtype={base.exp_blocks.dtype} "
          f"folded dtype={folded.exp_blocks.dtype} rope dtype={t.rope.dtype} "
          f"resid_folded={float((folded.exp_blocks.float()@folded.inv_blocks.float() - torch.eye(32)).abs().max()):.3e}")
    # autocast-disabled fold check
    with torch.autocast(device_type="cpu", enabled=False):
        f2 = TR = __import__("vfe3.geometry.transport", fromlist=["fold_rope_into_frame"]).fold_rope_into_frame
        folded2 = f2(base, t.rope)
    print(f"        fold under autocast(enabled=False): dtype={folded2.exp_blocks.dtype} "
          f"resid={float((folded2.exp_blocks.float()@folded2.inv_blocks.float() - torch.eye(32)).abs().max()):.3e}")
