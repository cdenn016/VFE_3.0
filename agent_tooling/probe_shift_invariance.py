"""Shift-invariance contrast: ambient rope path vs relational-folded path."""
import torch
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.geometry.rope import get_pos_rotation

base = dict(vocab_size=7, embed_dim=8, n_heads=2, max_seq_len=4, n_layers=1,
            n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
            skip_belief_sigma_update=True, e_phi_lr=0.0, oracle_unroll_grad=True,
            relational_mode="projected_filtering", relational_input_scope="full",
            relational_readout_mode="predictive_content", relational_output_scope="full",
            pos_rotation="none")

shift = {"v": 0.0}
orig_build = VFEModel._build_shared_flat_transport


def make_rope(model, n, device):
    pos = torch.arange(n, device=device, dtype=torch.float64) + shift["v"]
    return get_pos_rotation("rope")(pos, model.group.irrep_dims, base=model.cfg.rope_base,
                                    device=device, dtype=model.group.generator_dtype)


def patched(self, beliefs, rope, **kw):
    if rope is None:
        rope = make_rope(self, beliefs.mu.shape[-2], beliefs.mu.device)
    return orig_build(self, beliefs, rope, **kw)


ids = torch.tensor([[0, 1, 2, 3]])

# --- relational path, naive fold of rope into the frame ---
torch.manual_seed(17)
c = VFE3Config(**{**base, "pos_rotation": "none"})
object.__setattr__(c, "pos_rotation", "rope")
m = VFEModel(c)
VFEModel._build_shared_flat_transport = patched
with torch.no_grad():
    res = {}
    for s in (0.0, 0.5, 4.0, 40.0):
        shift["v"] = s
        b, l = m.forward_beliefs(ids, return_logits=True)
        res[s] = (b.mu.clone(), l.clone())
print("RELATIONAL (rope folded into the shared relational frame):")
for s in (0.5, 4.0, 40.0):
    print(f"  shift {s:5.1f}: max|dmu| = {float((res[0.0][0]-res[s][0]).abs().max()):.6e}"
          f"  max|dlogits| = {float((res[0.0][1]-res[s][1]).abs().max()):.6e}")

# --- ambient (standard) path: same rope, same shift ---
VFEModel._build_shared_flat_transport = orig_build
torch.manual_seed(17)
c2 = VFE3Config(**{**base, "relational_mode": "off"})
object.__setattr__(c2, "pos_rotation", "rope")
m2 = VFEModel(c2)
with torch.no_grad():
    res2 = {}
    for s in (0.0, 0.5, 4.0, 40.0):
        shift["v"] = s
        b, l = m2.forward_beliefs(ids, return_logits=True)
        res2[s] = (b.mu.clone(), l.clone())
print("AMBIENT (standard attention, pos_rotation='rope'):")
for s in (0.5, 4.0, 40.0):
    print(f"  shift {s:5.1f}: max|dmu| = {float((res2[0.0][0]-res2[s][0]).abs().max()):.6e}"
          f"  max|dlogits| = {float((res2[0.0][1]-res2[s][1]).abs().max()):.6e}")
