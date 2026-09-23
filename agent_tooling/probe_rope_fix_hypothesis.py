"""Fix-hypothesis probe: hand the model's rope tensor to the relational frame build.

Monkeypatches VFEModel._build_shared_flat_transport so that a None rope argument is
replaced by the model's own cached rope tensor. Nothing on disk is touched.
"""
import sys, warnings, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import vfe3.config as cfgmod

spec = importlib.util.spec_from_file_location("train_vfe3_live", ROOT / "train_vfe3.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
LIVE = dict(mm.config)

SRC = Path(cfgmod.__file__).read_text(encoding="utf-8")
NEEDLE = '                "pos_rotation": "none",\n'
exec(compile(SRC.replace(NEEDLE, ""), cfgmod.__file__, "exec"), cfgmod.__dict__)
VFE3Config = cfgmod.VFE3Config

from vfe3.model.model import VFEModel

USER = dict(LIVE)
USER.update(vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8,
            eval_stride=4, batch_size=2, max_steps=2)

SEEN = []


def run(pos_rotation, wire):
    cfg = VFE3Config(**{**USER, "pos_rotation": pos_rotation})
    orig = VFEModel._build_shared_flat_transport

    def spy(self, beliefs, rope, **kw):
        if wire and rope is None:
            rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.phi.device)
        SEEN.append((None if rope is None else tuple(rope.shape), kw.get("require_vertex_factors")))
        return orig(self, beliefs, rope, **kw)

    VFEModel._build_shared_flat_transport = spy
    try:
        torch.manual_seed(17)
        model = VFEModel(cfg)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
        tgt = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
        cap = {}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b, lg = model.forward_beliefs(ids, return_logits=True, capture=cap)
            loss = torch.nn.functional.cross_entropy(lg.reshape(-1, 7), tgt.reshape(-1))
            loss.backward()
        err = None
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        model, b, lg, loss = None, None, None, None
    finally:
        VFEModel._build_shared_flat_transport = orig
    tag = f"rope={pos_rotation} wire={wire}"
    print(f"\n--- {tag} ---")
    if err:
        print("  EXCEPTION:", err)
        return None, None
    print("  loss =", round(float(loss.detach()), 8), " finite logits:",
          bool(torch.isfinite(lg).all()))
    print("  build calls (rope arg shape, require_vertex_factors):", SEEN)
    return lg.detach().clone(), b.mu.detach().clone()


print("=== 1. current behavior: rope='rope', NOT wired ===")
SEEN.clear(); l1, m1 = run("rope", wire=False)
print("=== 2. rope='none' baseline ===")
SEEN.clear(); l0, m0 = run("none", wire=False)
print("=== 3. hypothesis fix: rope='rope' WIRED into the relational frame ===")
SEEN.clear(); l2, m2 = run("rope", wire=True)

if l1 is not None and l0 is not None:
    print("\n  max|rope_unwired - none| =", float((l1 - l0).abs().max()),
          " equal:", bool(torch.equal(l1, l0)))
if l2 is not None and l0 is not None:
    print("  max|rope_wired   - none| =", float((l2 - l0).abs().max()),
          " equal:", bool(torch.equal(l2, l0)))
    print("  max|mu_wired     - mu_none| =", float((m2 - m0).abs().max()))
