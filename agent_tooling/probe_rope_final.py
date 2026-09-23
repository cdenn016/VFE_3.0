"""Final probe (agent_tooling/probe_rope_final.py).

Uses the REAL train_vfe3.py config dict, shrunk to tiny dims, with pos_rotation='rope'.
Confirms: (1) the pin is the only blocker, (2) whether the rope tensor reaches the numerics.
"""
import sys, warnings, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import vfe3.config as cfgmod

# --- 1. real live config -------------------------------------------------------------
spec = importlib.util.spec_from_file_location("train_vfe3_live", ROOT / "train_vfe3.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
LIVE = dict(m.config)

RUN_KEYS = ("pos_phi", "rope_base", "rope_full_gauge", "rope_on_value",
            "relational_mode", "relational_input_scope", "relational_readout_mode",
            "relational_output_scope", "attention_residual", "gauge_group",
            "gauge_transport", "transport_mode", "family", "divergence_family",
            "renyi_order", "gradient_mode", "e_step_update", "n_e_steps",
            "skip_belief_sigma_update", "e_phi_lr", "encode_mode", "decode_mode",
            "m_phi_lr", "m_pos_phi_lr", "phi_scale", "use_head_mixer", "use_block_mlp",
            "block_mlp_mode", "lambda_beta", "beta_attention_prior")
TINY = {k: LIVE.get(k, "<ABSENT>") for k in RUN_KEYS}
print("=== live-config keys used (unchanged from train_vfe3.py) ===")
for k, v in TINY.items():
    print(f"  {k:28s} = {v!r}")

# --- 2. patch the pin out in memory, before downstream imports ------------------------
SRC = Path(cfgmod.__file__).read_text(encoding="utf-8")
NEEDLE = '                "pos_rotation": "none",\n'
assert SRC.count(NEEDLE) == 1
exec(compile(SRC.replace(NEEDLE, ""), cfgmod.__file__, "exec"), cfgmod.__dict__)
VFE3Config = cfgmod.VFE3Config

from vfe3.model.model import VFEModel
import vfe3.inference.e_step as es
from vfe3.geometry.transport import RopeTransport

CALLS = {"predictive_readout": 0, "frames": {}, "omega_rope": []}
_orig_readout = es.relational_predictive_readout
def readout(mu, sigma, relational_frame, ops, **kw):
    CALLS["predictive_readout"] += 1
    CALLS["frames"][kw.get("_tag", "f")] = relational_frame.forward.detach().clone()
    return _orig_readout(mu, sigma, relational_frame, ops, **kw)
es.relational_predictive_readout = readout

_orig_build = VFEModel._build_shared_flat_transport
def spy_build(self, beliefs, rope, **kw):
    out = _orig_build(self, beliefs, rope, **kw)
    CALLS["omega_rope"].append(("rope=None" if rope is None else f"rope{tuple(rope.shape)}",
                                type(out).__name__))
    return out
VFEModel._build_shared_flat_transport = spy_build

# the user's config, tiny dims
USER = dict(LIVE)
USER.update(vocab_size=7, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8,
            eval_stride=4, batch_size=2, max_steps=2, pos_rotation="rope")

def build_and_run(cfg, tag):
    torch.manual_seed(17)
    model = VFEModel(cfg)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0]])
    tgt = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 1]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cap = {}
        belief, logits = model.forward_beliefs(ids, return_logits=True, capture=cap)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), tgt.reshape(-1))
        loss.backward()
    print(f"\n--- tag={tag} pos_rotation={cfg.pos_rotation!r} pos_phi={cfg.pos_phi!r} ---")
    print("  forward+backward OK. loss =", round(float(loss.detach()), 8),
          " logits", tuple(logits.shape),
          " relational_capture present:", "relational_capture" in cap)
    print("  params with finite grad:",
          sum(1 for p in model.parameters() if p.grad is not None),
          "/", sum(1 for _ in model.parameters()),
          " all finite:", all(bool(torch.isfinite(p.grad).all())
                              for p in model.parameters() if p.grad is not None))
    rp = [p for p in model.relational_operators.parameters()]
    print("  relational op grads nonzero:", all(float(p.grad.abs().sum()) > 0 for p in rp),
          [round(float(p.grad.abs().sum()), 8) for p in rp])
    print("  ConfigNotices/warnings:", sorted({x.category.__name__ for x in w}))
    for x in w:
        if x.category.__name__ != "UserWarning":
            print("    ", str(x.message)[:150])
    return logits.detach().clone(), belief.mu.detach().clone(), model

print("\n=== A: pin REMOVED, pos_rotation='rope' ===")
lg_r, mu_r, mdl_r = build_and_run(VFE3Config(**USER), "rope")
print("  _rope_rotation is not None:",
      mdl_r._rope_rotation(8, torch.device("cpu")) is not None)
print("  build_shared_flat_transport calls (rope arg, type):", CALLS["omega_rope"])
print("  relational_predictive_readout calls:", CALLS["predictive_readout"])

print("\n=== B: pos_rotation='none' (same everything else) ===")
CALLS["omega_rope"].clear(); CALLS["predictive_readout"] = 0
USER_N = dict(USER); USER_N["pos_rotation"] = "none"
lg_n, mu_n, mdl_n = build_and_run(VFE3Config(**USER_N), "none")
print("  build_shared_flat_transport calls (rope arg, type):", CALLS["omega_rope"])
print("  relational_predictive_readout calls:", CALLS["predictive_readout"])

print("\n=== C: A/B ===")
print("  max |logits_rope - logits_none| =", float((lg_r - lg_n).abs().max()))
print("  max |mu_rope     - mu_none|     =", float((mu_r - mu_n).abs().max()))
print("  torch.equal(logits):", bool(torch.equal(lg_r, lg_n)),
      "  torch.equal(mu):", bool(torch.equal(mu_r, mu_n)))
