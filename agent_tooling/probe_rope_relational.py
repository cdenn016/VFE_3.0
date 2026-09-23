"""Probe: does pos_rotation='rope' affect the relational projected_filtering forward?"""
import contextlib, io, sys
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
import vfe3.geometry.transport as T


def cfg(**ov):
    values = dict(vocab_size=7, embed_dim=8, n_heads=2, max_seq_len=4, n_layers=1,
                  n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
                  skip_belief_sigma_update=True, e_phi_lr=0.0, oracle_unroll_grad=True,
                  relational_mode="projected_filtering",
                  relational_input_scope="full",
                  relational_readout_mode="predictive_content",
                  relational_output_scope="full",
                  pos_rotation="none")
    values.update(ov)
    return VFE3Config(**values)


count = {"n": 0}
orig_post = T.RopeTransport.__post_init__


def patched_post(self):
    count["n"] += 1
    return orig_post(self)


T.RopeTransport.__post_init__ = patched_post

torch.manual_seed(17)
c = cfg()
print("validation OK; pos_rotation =", c.pos_rotation)
model = VFEModel(c)
ids = torch.tensor([[0, 1, 2, 3]])
targets = torch.tensor([[1, 2, 3, 4]])

captures = {}
with torch.no_grad():
    count["n"] = 0
    b_none, logits_none = model.forward_beliefs(ids, return_logits=True, capture=captures.setdefault("none", {}))
    n_none = count["n"]

    # bypass validation: flip pos_rotation in place (model reads cfg at call time)
    object.__setattr__(c, "pos_rotation", "rope")
    print("after flip: cfg.pos_rotation =", c.pos_rotation)
    rope = model._rope_rotation(4, ids.device)
    print("rope tensor built:", None if rope is None else tuple(rope.shape))
    count["n"] = 0
    b_rope, logits_rope = model.forward_beliefs(ids, return_logits=True, capture=captures.setdefault("rope", {}))
    n_rope = count["n"]

print("RopeTransport constructions: none-", n_none, " rope-", n_rope)
print("mu equal      :", torch.equal(b_none.mu, b_rope.mu))
print("sigma equal   :", torch.equal(b_none.sigma, b_rope.sigma))
print("phi equal     :", torch.equal(b_none.phi, b_rope.phi))
print("logits equal  :", torch.equal(logits_none, logits_rope))
print("max |dmu|     :", float((b_none.mu - b_rope.mu).abs().max()))
print("max |dlogits| :", float((logits_none - logits_rope).abs().max()))
print("same relational_frame object identity:", captures["none"].get("canonical_frame") is captures["rope"].get("canonical_frame"))
for k, v in captures["rope"].items():
    print("  rope capture key:", k, type(v))
