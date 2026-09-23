"""Skeptic control: does the rope tensor change output at all on a path where it is honored?"""
from typing import Any
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _cfg(**overrides: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                  n_layers=1, n_e_steps=1, family="gaussian_diagonal",
                  gauge_group="block_glk", skip_belief_sigma_update=True,
                  e_phi_lr=0.0, oracle_unroll_grad=True, warmup_steps=0,
                  max_steps=2, log_interval=0, eval_interval=0, pos_phi="learned")
    values.update(overrides)
    return VFE3Config(**values)


TOK = torch.tensor([[0, 1, 2, 3]])


def run(rotation: str, **overrides):
    torch.manual_seed(17)
    cfg = _cfg(**overrides)
    cfg.pos_rotation = rotation
    model = VFEModel(cfg)
    model.eval()
    with torch.no_grad():
        b, l = model.forward_beliefs(TOK, return_logits=True)
    return b, l


for label, ov in (("relational off (legacy path)", dict()),
                  ("relational projected_filtering", dict(relational_mode="projected_filtering",
                                                          relational_input_scope="full",
                                                          relational_readout_mode="predictive_content",
                                                          relational_output_scope="full"))):
    print(f"== {label}")
    try:
        b_n, l_n = run("none", **ov)
        b_r, l_r = run("rope", **ov)
    except Exception as e:
        print("  raised:", type(e).__name__, str(e)[:160])
        continue
    print("  mu max |diff|    :", (b_n.mu - b_r.mu).abs().max().item())
    print("  logits max |diff|:", (l_n - l_r).abs().max().item())
