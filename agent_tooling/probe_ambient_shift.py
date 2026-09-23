"""Ambient (non-relational) rope path: is it shift-invariant?"""
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.geometry.rope import get_pos_rotation

c = VFE3Config(vocab_size=7, embed_dim=8, n_heads=2, max_seq_len=4, n_layers=1,
               n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
               skip_belief_sigma_update=True, e_phi_lr=0.0, oracle_unroll_grad=True,
               relational_mode="off", pos_rotation="none")
object.__setattr__(c, "pos_rotation", "rope")
torch.manual_seed(17)
m = VFEModel(c)

shift = {"v": 0.0}
orig = VFEModel._rope_rotation


def patched(self, n, device):
    if shift["v"] == 0.0:
        return orig(self, n, device)
    pos = torch.arange(n, device=device, dtype=torch.float64) + shift["v"]
    return get_pos_rotation("rope")(pos, self.group.irrep_dims, base=self.cfg.rope_base,
                                    device=device, dtype=self.group.generator_dtype)


VFEModel._rope_rotation = patched
ids = torch.tensor([[0, 1, 2, 3]])
with torch.no_grad():
    out = {}
    for s in (0.0, 0.5, 4.0, 40.0):
        shift["v"] = s
        b, l = m.forward_beliefs(ids, return_logits=True)
        out[s] = (b.mu.clone(), l.clone())
print("AMBIENT (standard attention, pos_rotation='rope'):")
for s in (0.5, 4.0, 40.0):
    print(f"  shift {s:5.1f}: max|dmu| = {float((out[0.0][0]-out[s][0]).abs().max()):.6e}"
          f"  max|dlogits| = {float((out[0.0][1]-out[s][1]).abs().max()):.6e}")
