"""PASS 3: remove the envelope pin in-memory (no tracked file edited) and run.

The pin lives in a dict literal local to __post_init__, so the only faithful bypass is to
re-exec vfe3.config with the literal replaced BEFORE any downstream import binds the class.
"""
import sys, importlib.util, warnings, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import vfe3.config as cfgmod

SRC = Path(cfgmod.__file__).read_text(encoding="utf-8")
NEEDLE = '                "pos_rotation": "none",\n'
assert SRC.count(NEEDLE) == 1, SRC.count(NEEDLE)
PATCHED = SRC.replace(NEEDLE, '')   # drop the pin from the envelope entirely
exec(compile(PATCHED, cfgmod.__file__, "exec"), cfgmod.__dict__)
VFE3Config = cfgmod.VFE3Config
print("patched pin -> ", [l for l in PATCHED.splitlines() if '"pos_rotation"' in l])

# downstream imports only now, so they bind the patched class
from vfe3.model.model import VFEModel

BASE = dict(
    vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8, n_e_steps=1,
    batch_size=2, max_steps=2, log_interval=0, eval_interval=0, warmup_steps=0,
    family="gaussian_diagonal", divergence_family="renyi", renyi_order=1.0,
    include_attention_entropy=True, gradient_mode="filtering", e_step_update="gradient",
    e_step_gradient="unroll", oracle_unroll_grad=True, skip_belief_sigma_update=True,
    e_phi_lr=0.0, transport_mode="flat", gauge_group="block_glk", gauge_transport="on",
    encode_mode="per_token", decode_mode="linear",
    pos_phi="none", use_block_mlp=True, block_mlp_expansion=1, use_head_mixer=False,
    relational_mode="projected_filtering", relational_input_scope="full",
    relational_readout_mode="predictive_content", relational_output_scope="full",
    relational_identity_content=False, attention_residual=True,
    rope_on_value=False, rope_full_gauge=False, rope_base=100.0,
)

print("\n=== PASS 3a: construct with pin removed, pos_rotation='rope' ===")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cfg_rope = VFE3Config(**{**BASE, "pos_rotation": "rope"})
    print("  CONSTRUCTED OK. pos_rotation =", cfg_rope.pos_rotation)
    print("  decoupled_value_gauge =", getattr(cfg_rope, "decoupled_value_gauge", "<absent>"))
    print("  rope_insertion =", getattr(cfg_rope, "rope_insertion", "<absent>"))
    print("  warnings raised:")
    for x in w:
        print(f"    [{x.category.__name__}] {str(x.message)[:170]}")

print("\n=== PASS 3b: build VFEModel + forward + backward (rope) ===")
torch.manual_seed(17)
try:
    model_rope = VFEModel(cfg_rope)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
    targets = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
    cap = {}
    belief, logits = model_rope.forward_beliefs(ids, return_logits=True, capture=cap)
    print("  forward OK. logits", tuple(logits.shape), "mu", tuple(belief.mu.shape))
    print("  logits finite:", bool(torch.isfinite(logits).all()))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    print("  loss =", float(loss))
    grads = [p.grad for p in model_rope.parameters() if p.grad is not None]
    print("  params with grad:", len(grads), "all finite:",
          all(bool(torch.isfinite(g).all()) for g in grads))
    rop = list(model_rope.relational_operators.parameters())
    print("  relational op grads nonzero:",
          all(float(g.grad.abs().sum()) > 0 for g in rop if g.grad is not None))
    ROPE_OK = True
    ROPE_LOGITS = logits.detach().clone()
    ROPE_MU = belief.mu.detach().clone()
except Exception as exc:
    print(f"  EXCEPTION TYPE: {type(exc).__name__}")
    print("  FULL MESSAGE:")
    print("  " + str(exc).replace("\n", "\n  "))
    traceback.print_exc()
    ROPE_OK = False

print("\n=== PASS 3c: same model with pos_rotation='none' (numerical A/B) ===")
cfg_none = VFE3Config(**{**BASE, "pos_rotation": "none"})
torch.manual_seed(17)
model_none = VFEModel(cfg_none)
cap2 = {}
belief_n, logits_n = model_none.forward_beliefs(ids, return_logits=True, capture=cap2)
print("  none logits finite:", bool(torch.isfinite(logits_n).all()))

if ROPE_OK:
    d_logits = (logits_n - ROPE_LOGITS).abs().max().item()
    d_mu = (belief_n.mu - ROPE_MU).abs().max().item()
    print(f"  max |logits_rope - logits_none| = {d_logits:.6e}")
    print(f"  max |mu_rope     - mu_none|     = {d_mu:.6e}")
    print("  bitwise identical:", bool(torch.equal(logits_n, ROPE_LOGITS)))
    r = torch.manual_seed(0)  # (no-op, keep rng tidy)
    # direct check that RopeTransport is actually wired into the frame
    print("\n=== PASS 3d: is the rotation actually applied? ===")
    try:
        from vfe3.geometry.rope import build_rope_rotation, get_pos_rotation
        group = cfgmod.__dict__.get("get_group")
        from vfe3.geometry.groups import get_group
        dims = get_group("block_glk")(K=8, n_heads=2).irrep_dims
        R = build_rope_rotation(torch.arange(8), dims, base=100.0)
        print("  irrep_dims =", dims, " R shape =", tuple(R.shape))
        print("  R[1] differs from I:", bool((R[1] - torch.eye(8)).abs().max() > 1e-6),
              " max dev =", float((R[1] - torch.eye(8)).abs().max()))
        print("  R orthogonal (R R^T - I):",
              float((R[1] @ R[1].T - torch.eye(8)).abs().max()))
    except Exception as exc:
        print("  rope probe failed:", type(exc).__name__, exc)
