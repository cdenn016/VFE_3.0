"""Probe: is pos_rotation='rope' a silent no-op on the projected_filtering path?

Constructs the same tiny predictive_content config twice, mutating cfg.pos_rotation
AFTER construction (which sidesteps the requirements-dict pin), then compares the
beliefs/logits. Identity would mean the rope never reaches the relational frame.
"""
from typing import Any
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**overrides: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                  n_layers=1, n_e_steps=1, family="gaussian_diagonal",
                  gauge_group="block_glk", skip_belief_sigma_update=True,
                  e_phi_lr=0.0, oracle_unroll_grad=True, warmup_steps=0,
                  max_steps=2, log_interval=0, eval_interval=0,
                  pos_phi="learned", use_block_mlp=True,
                  block_mlp_expansion=1, relational_mode="projected_filtering",
                  relational_input_scope="full",
                  relational_readout_mode="predictive_content",
                  relational_output_scope="full")
    values.update(overrides)
    return VFE3Config(**values)


TOK = torch.tensor([[0, 1, 2, 3]])


def run(rotation: str, **overrides):
    torch.manual_seed(17)
    cfg = _cfg(**overrides)
    assert cfg.pos_rotation == "none", cfg.pos_rotation
    if rotation != "none":
        cfg.pos_rotation = rotation          # post-construction: skips the pin
    model = VFEModel(cfg)
    model.eval()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(TOK, return_logits=True)
    return belief, logits


for readout, scope in (("predictive_content", "full"), ("conditional_update", "head")):
    print(f"== relational_readout_mode={readout!r}, input_scope={scope!r}")
    b_none, l_none = run("none", relational_readout_mode=readout,
                         relational_input_scope=scope,
                         relational_output_scope="full" if readout == "predictive_content" else "head")
    b_rope, l_rope = run("rope", relational_readout_mode=readout,
                         relational_input_scope=scope,
                         relational_output_scope="full" if readout == "predictive_content" else "head")
    print("  mu identical     :", torch.equal(b_none.mu, b_rope.mu))
    print("  logits identical :", torch.equal(l_none, l_rope))
    print("  mu max |diff|    :", (b_none.mu - b_rope.mu).abs().max().item())
    print("  logits max |diff|:", (l_none - l_rope).abs().max().item())

# Second probe: does the conditional path's transport ever carry the rope?
from vfe3.model.model import VFEModel as M
import inspect
src = inspect.getsource(M.forward_beliefs)
for i, line in enumerate(src.splitlines()):
    if "conditional_filtering_enabled" in line or "fold_rope" in line or "require_vertex_factors=True" in line:
        print(f"  model.py forward_beliefs +{i}: {line.strip()}")
