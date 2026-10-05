"""Skeptic check: if the rope WERE handed to the conditional frame build, does it work (no exception)?"""
from typing import Any, Optional
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

_ORIG = VFEModel._build_shared_flat_transport


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


def patched(self, beliefs, rope, **kw):
    if rope is None and self.cfg.pos_rotation != "none":
        rope = self._rope_rotation(beliefs.mu.shape[-2], beliefs.mu.device)
    return _ORIG(self, beliefs, rope, **kw)


def run(rotation: str, monkey: bool):
    torch.manual_seed(29)
    cfg = _cfg()
    if rotation != "none":
        cfg.pos_rotation = rotation
    model = VFEModel(cfg)
    model.eval()
    if monkey:
        VFEModel._build_shared_flat_transport = patched
    try:
        with torch.no_grad():
            b, l = model.forward_beliefs(TOK, return_logits=True)
    finally:
        VFEModel._build_shared_flat_transport = _ORIG
    return b, l


b_n, l_n = run("none", False)
b_r, l_r = run("rope", False)
b_m, l_m = run("rope", True)
print("unpatched rope vs none: mu max|diff|", (b_n.mu - b_r.mu).abs().max().item(),
      " logits max|diff|", (l_n - l_r).abs().max().item())
print("patched   rope vs none: mu max|diff|", (b_n.mu - b_m.mu).abs().max().item(),
      " logits max|diff|", (l_n - l_m).abs().max().item())
