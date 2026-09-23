"""Independent check of the proposed fix for pos_rotation='rope' under projected_filtering.

Tests the REAL proposed mechanism (autocast-disabled island inside fold_rope_into_frame),
not an input upcast, which autocast re-downcasts anyway.

Three arms, all with the validation pin bypassed in memory and rope wired at
model.py:1165 (the proposed one-argument change), at the user's live head geometry:

  A. shipped      : no island            -> expect the CanonicalFrameContext ValueError
  B. island       : autocast disabled    -> the proposed fix
  C. no-rope      : rope never wired     -> the inert baseline (bit-identity control)

Prints the traceback for arm A so the raise site is identified rather than assumed,
then checks that rope is LIVE under arm B (output differs from arm C) and that the
backward pass is finite.
"""
import inspect
import sys
import textwrap
import traceback

import torch

ROOT = r"C:\Users\chris and christine\Desktop\V3_Transformer"
sys.path.insert(0, ROOT)

import train_vfe3 as T                                       # noqa: E402
from vfe3.config import VFE3Config                           # noqa: E402

LIVE = dict(T.config)

# -- bypass the pin in memory (never touches a tracked file) --------------------
_src = textwrap.dedent(inspect.getsource(VFE3Config.__post_init__))
_src = _src.replace('"pos_rotation": "none",', '"pos_rotation": self.pos_rotation,')
_ns = dict(vars(sys.modules["vfe3.config"]))
exec(compile(_src, "<patched>", "exec"), _ns)
VFE3Config.__post_init__ = _ns["__post_init__"]

import vfe3.geometry.transport as TR                         # noqa: E402
from vfe3.model.model import VFEModel                        # noqa: E402

_orig_fold = TR.fold_rope_into_frame
_orig_build = VFEModel._build_shared_flat_transport
STATE = {"wire": True, "island": False}


def _fold_with_island(base, rope):
    if not STATE["island"]:
        return _orig_fold(base, rope)
    with torch.autocast(device_type=rope.device.type, enabled=False):
        return _orig_fold(base, rope)


TR.fold_rope_into_frame = _fold_with_island


def _hooked_build(self, beliefs, rope, **kw):
    if rope is None and STATE["wire"]:
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    return _orig_build(self, beliefs, rope, **kw)


VFEModel._build_shared_flat_transport = _hooked_build


def _cfg(K, H, amp):
    return VFE3Config(**{
        **LIVE, "embed_dim": K, "n_heads": H, "n_layers": 1, "max_seq_len": 8,
        "batch_size": 1, "n_e_steps": 1, "eval_stride": None, "kl_max": 8 * K,
        "pos_rotation": "rope", "amp_dtype": amp,
    })


def run(K, H, amp, wire, island, label, show_tb=False):
    STATE["wire"], STATE["island"] = wire, island
    torch.manual_seed(0)
    model = VFEModel(_cfg(K, H, amp))
    tok = torch.randint(0, 100, (1, 8))
    try:
        with torch.no_grad():
            belief, logits = model.forward_beliefs(tok, return_logits=True)
        out = {"mu": belief.mu.detach().clone(), "logits": logits.detach().clone()}
        print(f"  {label:34s} K={K:3d} H={H} amp={str(amp):5s} -> OK "
              f"mu_norm={float(out['mu'].norm()):.6f} logits_norm={float(out['logits'].norm()):.6f}")
        return out
    except Exception as exc:                                  # noqa: BLE001
        print(f"  {label:34s} K={K:3d} H={H} amp={str(amp):5s} -> "
              f"{type(exc).__name__}: {str(exc)[:130]}")
        if show_tb:
            tb = traceback.format_exc().splitlines()
            keep = [ln for ln in tb if "site-packages" not in ln]
            print("\n".join("      " + ln for ln in keep[-28:]))
        return None


print("=== K=160 H=5 (user geometry), amp=bf16 ===")
none_arm = run(160, 5, "bf16", wire=False, island=False, label="C. no-rope baseline")
ship_arm = run(160, 5, "bf16", wire=True, island=False, label="A. wired, no island", show_tb=True)
isl_arm = run(160, 5, "bf16", wire=True, island=True, label="B. wired + fp32 island")

print("\n=== K=64 H=4 (d=16), amp=bf16 ===")
run(64, 4, "bf16", True, False, "A. wired, no island")
run(64, 4, "bf16", True, True, "B. wired + fp32 island")

print("\n=== K=160 H=5, amp=None (fp32 control) ===")
run(160, 5, None, True, False, "A. wired, no island (fp32)")

if none_arm and isl_arm:
    dmu = float((isl_arm["mu"] - none_arm["mu"]).abs().max())
    dlog = float((isl_arm["logits"] - none_arm["logits"]).abs().max())
    print(f"\nrope LIVE under the island fix? max|dmu|={dmu:.6g} max|dlogits|={dlog:.6g} "
          f"-> {'LIVE' if dlog > 0 else 'INERT'}")
    print(f"fwd-only check: use the training path below for grads")

# backward finiteness under the proposed fix at live geometry
STATE["wire"], STATE["island"] = True, True
torch.manual_seed(0)
model = VFEModel(_cfg(160, 5, "bf16"))
tok = torch.randint(0, 100, (1, 8))
try:
    _, loss, _ = model(tok, tok)
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    n_with = sum(p.grad is not None for p in model.parameters())
    print(f"backward OK: loss={float(loss.detach()):.6f} params_with_grad={n_with} "
          f"nonfinite={bad if bad else 'none'}")
    rel = [n for n, p in model.named_parameters() if "relational" in n or "raw_score" in n]
    print(f"relational params present: {len(rel)}; sample={rel[:3]}")
except Exception as exc:                                      # noqa: BLE001
    print(f"backward FAILED: {type(exc).__name__}: {str(exc)[:200]}")
