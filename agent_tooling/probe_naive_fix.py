"""Probe: what happens if the transport builder is naively given rope on the relational path?"""
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


c = cfg()
print("rope_insertion =", c.rope_insertion, "| rope_on_value =", c.rope_on_value)
object.__setattr__(c, "pos_rotation", "rope")
torch.manual_seed(17)
model = VFEModel(c)
ids = torch.tensor([[0, 1, 2, 3]])

# cache the rope tensor for positions arange(n) and for arange(n)+shift
orig_build = VFEModel._build_shared_flat_transport
shift = {"v": 0.0}


def make_rope(model, n, device):
    from vfe3.geometry.rope import get_pos_rotation
    pos = torch.arange(n, device=device, dtype=torch.float64) + shift["v"]
    return get_pos_rotation(model.cfg.pos_rotation)(
        pos, model.group.irrep_dims, base=model.cfg.rope_base, device=device,
        dtype=model.group.generator_dtype)


def patched(self, beliefs, rope, **kw):
    if rope is None:
        rope = make_rope(self, beliefs.mu.shape[-2], beliefs.mu.device)
    return orig_build(self, beliefs, rope, **kw)


VFEModel._build_shared_flat_transport = patched

try:
    with torch.no_grad():
        shift["v"] = 0.0
        b0, l0 = model.forward_beliefs(ids, return_logits=True)
        shift["v"] = 0.5
        b1, l1 = model.forward_beliefs(ids, return_logits=True)
    print("naive fix ran. position-shift 0.0 vs 0.5 (identical input tokens):")
    print("  max|dmu|     =", float((b0.mu - b1.mu).abs().max()),
          " max|dlogits| =", float((l0 - l1).abs().max()))
    print("  mu equal:", torch.equal(b0.mu, b1.mu), " logits equal:", torch.equal(l0, l1))
except Exception as e:
    print("naive fix RAISED:", type(e).__name__, e)
