"""Does rope reach the relational path through the block-MLP canonical frame channel?"""
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
                  relational_mode="projected_filtering", relational_input_scope="full",
                  relational_readout_mode="predictive_content", relational_output_scope="full",
                  pos_rotation="none", use_block_mlp=True)
    values.update(ov)
    return VFE3Config(**values)


ids = torch.tensor([[0, 1, 2, 3]])
for mode in ("coordinate", "gauge_gate", "canonical_frame"):
    try:
        torch.manual_seed(17)
        c = cfg(block_mlp_mode=mode)
    except Exception as e:
        print(f"{mode:16s}: CONFIG REJECTED: {type(e).__name__}: {e}")
        continue
    object.__setattr__(c, "pos_rotation", "rope")
    torch.manual_seed(17)
    m = VFEModel(c)
    print(f"{mode:16s}: requires_frame_context =",
          m._block_mlp_registration.requires_frame_context)
    with torch.no_grad():
        c0 = c
        object.__setattr__(c, "pos_rotation", "none")
        b0, l0 = m.forward_beliefs(ids, return_logits=True)
        n0 = count["n"]
        count["n"] = 0
        object.__setattr__(c, "pos_rotation", "rope")
        b1, l1 = m.forward_beliefs(ids, return_logits=True)
        n1 = count["n"]
    print(f"                  RopeTransport {n0} -> {n1}; mu equal = "
          f"{torch.equal(b0.mu, b1.mu)}, logits equal = {torch.equal(l0, l1)}, "
          f"max|dmu| = {float((b0.mu - b1.mu).abs().max()):.3e}")
