"""Empirical repro: pos_rotation='rope' under relational_mode='projected_filtering'.

Reads the REAL active config dict out of train_vfe3.py, shrinks it to tiny dims, and
reproduces the config-construction failure, then bypasses the envelope pin and runs.
"""
import sys, traceback, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

# ---- pull the real active config out of train_vfe3.py -------------------------------
import importlib.util
spec = importlib.util.spec_from_file_location("train_vfe3_live", ROOT / "train_vfe3.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
LIVE = dict(mod.config)

print("=== LIVE config (relational / pos / gauge keys) ===")
for k in ("relational_mode", "relational_input_scope", "relational_readout_mode",
          "relational_output_scope", "relational_identity_content", "attention_residual",
          "family", "divergence_family", "renyi_order", "include_attention_entropy",
          "gradient_mode", "e_step_update", "e_step_gradient", "oracle_unroll_grad",
          "n_e_steps", "skip_belief_sigma_update", "e_phi_lr", "transport_mode",
          "gauge_group", "gauge_transport", "pos_phi", "pos_rotation", "rope_base",
          "rope_full_gauge", "rope_on_value", "encode_mode", "decode_mode",
          "embed_dim", "n_heads", "n_layers", "use_block_mlp", "block_mlp_mode"):
    print(f"  {k:32s} = {LIVE.get(k, '<ABSENT>')!r}")

# ---- shrink to tiny dims ------------------------------------------------------------
TINY = dict(LIVE)
TINY.update(
    vocab_size=64,
    embed_dim=8,
    n_heads=2,
    n_layers=1,
    max_seq_len=8,
    eval_stride=4,
    batch_size=2,
    max_steps=2,
    pos_rotation="rope",
    pos_phi="none",          # isolate the rope variable; pos_phi='learned' is orthogonal
)

from vfe3.config import VFE3Config

warnings.simplefilter("always")

print("\n=== PASS 1: pos_rotation='rope' (user's request), VFE3Config(**TINY) ===")
try:
    cfg = VFE3Config(**TINY)
    print("  CONSTRUCTED OK: pos_rotation =", cfg.pos_rotation)
except Exception as exc:
    print(f"  EXCEPTION TYPE: {type(exc).__name__}")
    print(f"  EXCEPTION MODULE: {type(exc).__module__}")
    print("  FULL MESSAGE:")
    print("  " + str(exc).replace("\n", "\n  "))

print("\n=== PASS 2: pos_rotation='none' (baseline; envelope must accept) ===")
TINY_NONE = dict(TINY)
TINY_NONE["pos_rotation"] = "none"
try:
    cfg_none = VFE3Config(**TINY_NONE)
    print("  CONSTRUCTED OK: pos_rotation =", cfg_none.pos_rotation)
    print("  serialized pos_rotation =", cfg_none.pos_rotation)
except Exception as exc:
    print(f"  EXCEPTION TYPE: {type(exc).__name__}")
    print("  " + str(exc).replace("\n", "\n  "))
