from tests.profiles import test_config
import torch

from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.relational_operators import RelationalOperators
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.stack import vfe_stack


def _belief_and_group(n=6, K=4, n_heads=2):
    group = get_group("block_glk")(K, n_heads)
    torch.manual_seed(0)
    n_gen = group.generators.shape[0]
    belief = BeliefState(mu=torch.randn(n, K), sigma=torch.rand(n, K) + 0.5,
                         phi=torch.zeros(n, n_gen))
    return belief, group


def _relational_stack_kwargs(belief: BeliefState, group: GaugeGroup) -> dict:
    H, d = len(group.irrep_dims), group.irrep_dims[0]
    frame = torch.eye(d, dtype=belief.mu.dtype).expand(belief.mu.shape[0], H, d, d)
    return {
        "relational_operators": RelationalOperators(list(group.irrep_dims), 11,
                                                   readout_mode="full_link_update"),
        "relational_frame": CanonicalFrameContext(frame, frame, block_diagonal=True),
    }


def _nonidentity_mixer(group):
    mixer = HeadMixer(group.irrep_dims)
    with torch.no_grad():
        mixer.mixer_delta.copy_(0.1 * torch.randn_like(mixer.mixer_delta))
    return mixer


def test_head_mixer_per_block_differs_from_post_stack_at_two_layers():
    # With L=2 and full handoff, mixing INSIDE each block (so the mixed belief feeds the next
    # block's prior) is NOT the same as mixing once after a no-mixer 2-block stack. This is the
    # only configuration with test signal -- it pins the per-block mixing semantics.
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=2,
                     gauge_group="block_glk", prior_handoff_rho=1.0)
    belief, group = _belief_and_group()
    mixer = _nonidentity_mixer(group)
    kwargs = _relational_stack_kwargs(belief, group)
    out_pb = vfe_stack(belief, belief.mu, belief.sigma, group, cfg, head_mixer=mixer, **kwargs)
    out_no = vfe_stack(belief, belief.mu, belief.sigma, group, cfg, head_mixer=None, **kwargs)
    mu_once, _ = mixer(out_no.mu, out_no.sigma)
    assert not torch.allclose(out_pb.mu, mu_once, atol=1e-4)


def test_head_mixer_per_block_equals_post_stack_at_one_layer():
    # At the shipped n_layers=1 the handoff loop runs once, so per-block mixing == mixing once after
    # the stack (block_norm off): the change is behavior-preserving on the default config.
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     gauge_group="block_glk")
    belief, group = _belief_and_group()
    mixer = _nonidentity_mixer(group)
    kwargs = _relational_stack_kwargs(belief, group)
    out_pb = vfe_stack(belief, belief.mu, belief.sigma, group, cfg, head_mixer=mixer, **kwargs)
    out_no = vfe_stack(belief, belief.mu, belief.sigma, group, cfg, head_mixer=None, **kwargs)
    mu_once, sig_once = mixer(out_no.mu, out_no.sigma)
    assert torch.allclose(out_pb.mu, mu_once, atol=1e-5)
    assert torch.allclose(out_pb.sigma, sig_once, atol=1e-5)
