"""Skeptic probe: independently check the 'rope is a silent no-op under projected_filtering' claim."""
from typing import Any
import torch

from vfe3.config import VFE3Config
from vfe3.model import model as model_mod
from vfe3.model.model import VFEModel
from vfe3.geometry.transport import RopeTransport


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


TOK = torch.tensor([[0, 1, 2, 3]])

LOG = []


def run(rotation: str, **overrides):
    torch.manual_seed(17)
    cfg = _cfg(**overrides)
    if rotation != "none":
        cfg.pos_rotation = rotation
    model = VFEModel(cfg)
    model.eval()
    # is the rope tensor actually materialized under the post-construction mutation?
    rope = model._rope_rotation(4, torch.device("cpu"))
    print(f"  cfg.pos_rotation={cfg.pos_rotation!r}  rope is None: {rope is None}")
    # intercept the transport handed to vfe_stack
    seen = {}
    real_vfe_stack = model_mod.vfe_stack

    def spy(*args, **kwargs):
        seen["prebuilt"] = kwargs.get("prebuilt_transport")
        seen["relational_frame"] = args[13] if len(args) > 13 else kwargs.get("relational_frame")
        seen["rope"] = kwargs.get("rope")
        return real_vfe_stack(*args, **kwargs)

    model_mod.vfe_stack = spy
    try:
        with torch.no_grad():
            belief, logits = model.forward_beliefs(TOK, return_logits=True)
    finally:
        model_mod.vfe_stack = real_vfe_stack
    pb = seen.get("prebuilt")
    rf = seen.get("relational_frame")
    print(f"  prebuilt_transport type: {type(pb).__name__}  is_RopeTransport: {isinstance(pb, RopeTransport)}")
    print(f"  relational_frame is None: {rf is None}  rf type: {type(rf).__name__}")
    print(f"  vfe_stack rope is None: {seen.get('rope') is None}")
    return belief, logits


for readout, scope in (("predictive_content", "full"), ("conditional_update", "head")):
    print(f"== readout={readout!r} scope={scope!r}")
    b_none, l_none = run("none", relational_readout_mode=readout, relational_input_scope=scope,
                         relational_output_scope="full" if readout == "predictive_content" else "head")
    b_rope, l_rope = run("rope", relational_readout_mode=readout, relational_input_scope=scope,
                         relational_output_scope="full" if readout == "predictive_content" else "head")
    print("  mu identical     :", torch.equal(b_none.mu, b_rope.mu))
    print("  logits identical :", torch.equal(l_none, l_rope))
    print("  mu max |diff|    :", (b_none.mu - b_rope.mu).abs().max().item())
    print("  logits max |diff|:", (l_none - l_rope).abs().max().item())
