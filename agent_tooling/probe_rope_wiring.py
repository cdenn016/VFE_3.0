"""Probe 2: what happens if the rope tensor IS handed to the relational frame build.

`fake` mode: monkeypatch VFEModel._build_shared_flat_transport so that a None rope
argument is replaced by the model's own cached rope tensor when cfg.pos_rotation=='rope'.
That is exactly the one-line wiring change (model.py:1165-1166 / 1139-1141 pass None).
Nothing on disk is touched.
"""
import importlib

import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
import vfe3.inference.e_step as es
import vfe3.geometry.transport as gtr

BUILT = []


class _CountingRope(gtr.RopeTransport):
    def __init__(self, *a, **kw):
        BUILT.append(type(kw.get("base", a[0] if a else None)).__name__)
        super().__init__(*a, **kw)


for _name in ["vfe3.inference.e_step", "vfe3.model.model", "vfe3.gradients.kernels",
              "vfe3.gradients.oracle", "vfe3.families.exact_congruence"]:
    _m = importlib.import_module(_name)
    if getattr(_m, "RopeTransport", None) is gtr.RopeTransport:
        _m.RopeTransport = _CountingRope


def _cfg(**over):
    vals = dict(vocab_size=11, embed_dim=8, n_heads=2, max_seq_len=6,
                n_layers=1, n_e_steps=1, family="gaussian_diagonal",
                gauge_group="block_glk", skip_belief_sigma_update=True,
                e_phi_lr=0.0, oracle_unroll_grad=True, warmup_steps=0,
                max_steps=2, log_interval=0, eval_interval=0,
                pos_phi="learned", use_block_mlp=False,
                relational_mode="projected_filtering",
                relational_input_scope="full",
                relational_readout_mode="predictive_content",
                relational_output_scope="full",
                attention_residual=True,
                pos_rotation="none")
    vals.update(over)
    return VFE3Config(**vals)


def run(cfg, ids, wire):
    seen = {}
    orig_build = VFEModel._build_shared_flat_transport
    orig_frame = VFEModel._canonical_frame_context

    def spy_build(self, beliefs, rope, **kw):
        if wire and rope is None:
            n = beliefs.mu.shape[-2]
            rope = self._rope_rotation(n, beliefs.phi.device)
        seen["rope_arg"] = None if rope is None else tuple(rope.shape)
        out = orig_build(self, beliefs, rope, **kw)
        seen["transport_type"] = type(out).__name__
        return out

    def spy_frame(transport, **kw):
        seen["frame_src_type"] = type(transport).__name__
        out = orig_frame(transport, **kw)
        seen["frame_forward_shape"] = tuple(out.forward.shape)
        return out

    VFEModel._build_shared_flat_transport = spy_build
    VFEModel._canonical_frame_context = staticmethod(spy_frame)
    torch.manual_seed(3)
    model = VFEModel(cfg)
    try:
        with torch.no_grad():
            belief, _ = model.forward_beliefs(ids, return_logits=False)
    finally:
        VFEModel._build_shared_flat_transport = orig_build
        VFEModel._canonical_frame_context = orig_frame
    print(f"[wire={wire}] transport rope arg={seen.get('rope_arg')} "
          f"type={seen.get('transport_type')} frame_src={seen.get('frame_src_type')} "
          f"frame_shape={seen.get('frame_forward_shape')}")
    return belief


if __name__ == "__main__":
    ids = torch.tensor([[0, 1, 2, 3, 4]])

    cfg = _cfg()
    base = run(cfg, ids, wire=False)

    cfg_rope = _cfg()
    cfg_rope.pos_rotation = "rope"
    BUILT.clear()
    wired = run(cfg_rope, ids, wire=True)
    print("RopeTransport instances when wired:", len(BUILT), BUILT)
    print("mu identical (no-rope vs wired-rope):", torch.equal(base.mu, wired.mu))
    print("max |dmu| wired:", (base.mu - wired.mu).abs().max().item())
