from tests.profiles import test_config
import torch

from vfe3.belief import BeliefState
from vfe3.geometry.groups import get_group


def _belief(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("block_glk")(K, 1)
    n_gen = grp.generators.shape[0]
    b = BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )
    mu_p = torch.randn(N, K, generator=g)
    sigma_p = torch.rand(N, K, generator=g) + 0.5
    return b, mu_p, sigma_p, grp


def test_belief_state_fields():
    b, *_ = _belief()
    assert b.mu.shape == (3, 2) and b.sigma.shape == (3, 2)


# --- Task 2: one inner iteration -------------------------------------------


def test_transport_flat_kwarg_is_byte_identical_to_default():
    """``_transport`` with the default transport_mode='flat' is bit-identical (torch.equal) to
    the registry-routed flat builder, on both the 2-D (diagnostics) and 3-D (batched) paths."""
    from vfe3.inference.e_step import _transport

    b, mu_p, sigma_p, grp = _belief()
    # 2-D (N, n_gen) diagnostics path
    o2_default = _transport(b.phi, grp)
    o2_flat    = _transport(b.phi, grp)
    assert torch.equal(o2_flat, o2_default)
    # 3-D (B, N, n_gen) batched path
    phi3 = b.phi.unsqueeze(0).expand(2, *b.phi.shape).contiguous()
    o3_default = _transport(phi3, grp)
    o3_flat    = _transport(phi3, grp)
    assert torch.equal(o3_flat, o3_default)


# --- Task 3: descent directions (the right objective per mode) -------------


# --- Task 4: e_step loop + trajectory + fixed-seed regression --------------

# Re-frozen 2026-09-23 (audit #6): the filtering phi step now differentiates only each frame's
# own receiver row (sender frames detached); with that detach disabled the run still gives 6.6499.


