"""Skeptic check: is pos_rotation='rope' a no-op on the BACKWARD too, not just the forward?"""
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
                  pos_phi="learned", use_head_mixer=True, use_block_mlp=True,
                  block_mlp_expansion=1, relational_mode="projected_filtering",
                  relational_input_scope="full",
                  relational_readout_mode="predictive_content",
                  relational_output_scope="full")
    values.update(overrides)
    return VFE3Config(**values)


TOK = torch.tensor([[0, 1]])
TGT = torch.tensor([[1, 2]])


def run(rotation: str):
    torch.manual_seed(23)
    cfg = _cfg()
    if rotation != "none":
        cfg.pos_rotation = rotation
    model = VFEModel(cfg)
    model.train()
    model.zero_grad(set_to_none=True)
    _, logits = model.forward_beliefs(TOK, return_logits=True)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), TGT.reshape(-1))
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in model.named_parameters()}
    return float(loss), grads


l_n, g_n = run("none")
l_r, g_r = run("rope")
print("loss none/rope :", l_n, l_r)
common = sorted(set(g_n) & set(g_r))
missing_n = [n for n in g_n if g_n[n] is None]
print("params:", len(common), "None-grad (none):", len(missing_n))
worst = 0.0
worst_name = None
n_diff = 0
for n in common:
    a, b = g_n[n], g_r[n]
    if a is None or b is None:
        if (a is None) != (b is None):
            print("  grad presence differs:", n)
        continue
    d = (a - b).abs().max().item()
    if d > 0:
        n_diff += 1
    if d > worst:
        worst, worst_name = d, n
print("params with differing grad:", n_diff, "/", len(common))
print("worst |grad diff|:", worst, "at", worst_name)
