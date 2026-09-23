"""Probe: conditional_update route under relational projected_filtering with rope flipped on."""
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
import vfe3.geometry.transport as T

count = {"n": 0}
orig_post = T.RopeTransport.__post_init__
T.RopeTransport.__post_init__ = lambda self: (count.__setitem__("n", count["n"] + 1), orig_post(self))[1]


def cfg(**ov):
    values = dict(vocab_size=7, embed_dim=8, n_heads=2, max_seq_len=4, n_layers=1,
                  n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
                  skip_belief_sigma_update=True, e_phi_lr=0.0, oracle_unroll_grad=True,
                  relational_mode="projected_filtering",
                  relational_readout_mode="conditional_update", pos_rotation="none")
    values.update(ov)
    return VFE3Config(**values)


torch.manual_seed(17)
c = cfg()
model = VFEModel(c)
ids = torch.tensor([[0, 1, 2, 3]])
with torch.no_grad():
    count["n"] = 0
    b0, l0 = model.forward_beliefs(ids, return_logits=True)
    n0 = count["n"]
    object.__setattr__(c, "pos_rotation", "rope")
    count["n"] = 0
    b1, l1 = model.forward_beliefs(ids, return_logits=True)
    n1 = count["n"]
print("conditional_update: RopeTransport built:", n0, "->", n1)
print("  mu equal:", torch.equal(b0.mu, b1.mu), "sigma equal:", torch.equal(b0.sigma, b1.sigma),
      "logits equal:", torch.equal(l0, l1))
