"""Probe 2: does the relational frame machinery ACCEPT a rope-wrapped transport?

Monkeypatches VFEModel._build_shared_flat_transport so the conditional path actually
receives the model's rope rotation. If the forward then runs and differs from
pos_rotation='none', the pin is a wiring gap rather than a mathematical
incompatibility.
"""
from typing import Any
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

_orig = VFEModel._build_shared_flat_transport


def _patched(self, beliefs, rope, **kw):
    if rope is None and self.cfg.pos_rotation != "none":
        n = beliefs.phi.shape[-2]
        rope = self._rope_rotation(n, beliefs.phi.device)
    return _orig(self, beliefs, rope, **kw)


VFEModel._build_shared_flat_transport = _patched


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


def run(rotation: str):
    torch.manual_seed(17)
    cfg = _cfg()
    if rotation != "none":
        cfg.pos_rotation = rotation
    model = VFEModel(cfg)
    model.eval()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(TOK, return_logits=True)
    return belief, logits


b_none, l_none = run("none")
b_rope, l_rope = run("rope")
print("patched: rope IS handed to the conditional transport builder")
print("  mu identical     :", torch.equal(b_none.mu, b_rope.mu))
print("  logits identical :", torch.equal(l_none, l_rope))
print("  mu max |diff|    :", (b_none.mu - b_rope.mu).abs().max().item())
print("  logits max |diff|:", (l_none - l_rope).abs().max().item())
