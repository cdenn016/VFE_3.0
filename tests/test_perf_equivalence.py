"""Audit Group 4: golden equivalence gates for the performance rewrite.

These tests freeze representative numerics as literal checksums so performance changes
(factored transport, per-block matrix_exp, batch vectorization, cached invariants) preserve
the result rather than merely run. The K=4 model-forward checksums were captured from the
unmodified `origin/main` commit 1b58d4f before the 2026-07-13 audit remediation.

Perf note: equivalence is all that is asserted here. The wins (avoiding the dense
(B,N,N,K,K) Omega; per-block float32 exp instead of a float64 full-K exp; a vectorized
E-step) are GPU memory/throughput gains that CANNOT be measured on CPU at B=1; they are
analyzed by construction in the post-edit doc, not timed here.
"""

from tests.profiles import test_config

import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators
from vfe3.model.model import VFEModel


def _fwd_cfg(**overrides: object) -> VFE3Config:
    # pos_phi="none" pins the canonical no-positional-composition forward these checksums were
    # captured on; the perf refactor this oracle guards (factored transport / per-block exp / batch
    # vectorization) is independent of the pos_phi gauge composition.
    # These checksums were captured on the KL-to-prior decoder that the oracle guards.
    return test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=2,
                      n_e_steps=2, pos_phi="none",
                      decode_mode="diagonal_chunked", **overrides)


def test_batched_forward_equals_per_sample():
    # The property batch-vectorization (4c) must preserve: the B=2 batched forward equals
    # the two samples run independently as B=1 and stacked. Distinct sequences so cross-batch
    # leakage (a vectorization bug) would be caught, not hidden by identical rows.
    cfg = _fwd_cfg()
    tok = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 0]])

    torch.manual_seed(0)
    m = VFEModel(cfg)
    with torch.no_grad():
        batched = m(tok)                                   # (2, 6, V)
        s0 = m(tok[0:1])
        s1 = m(tok[1:2])
    assert torch.allclose(batched[0].double(), s0[0].double(), atol=1e-5)
    assert torch.allclose(batched[1].double(), s1[0].double(), atol=1e-5)


def _transport_inputs():
    torch.manual_seed(0)
    grp = get_group("block_glk")(8, 2)
    n_gen = grp.generators.shape[0]
    phi = 0.2 * torch.randn(2, 5, n_gen)
    mu = torch.randn(2, 5, 8)
    sig = torch.rand(2, 5, 8) + 0.5
    return grp, phi, mu, sig


def test_block_glk_phi_matrix_is_block_diagonal():
    # Precondition for per-block matrix_exp (4b): block_glk without cross-couplings has a
    # genuinely block-diagonal generator embedding (irrep_dims [4,4]), so exp factors per block.
    grp, phi, _, _ = _transport_inputs()
    assert grp.irrep_dims == [4, 4]
    pm = torch.einsum("bna,aij->bnij", phi, grp.generators)   # (B, N, 8, 8)
    off = pm[:, :, :4, 4:].abs().max() + pm[:, :, 4:, :4].abs().max()
    assert float(off) == 0.0


def test_per_block_exp_is_bit_equivalent_to_full_exp():
    # Audit 4b: the per-block matrix_exp path must be BIT-equivalent to the full-matrix exp for a
    # block-diagonal generator embedding -- not merely "close". Tripwire at 1e-12: a loose pass
    # (~1e-3) would mean a small block silently dropped to float32; that is a failure, not a pass.
    from vfe3.geometry.transport import _blockwise_matrix_exp, stable_matrix_exp_pair
    grp, phi, _, _ = _transport_inputs()
    pm = torch.einsum("bna,aij->bnij", phi, grp.generators)   # (B, N, 8, 8) block-diagonal

    # blockwise vs full, both float64, no clamp (norms are tiny here)
    pm64 = pm.double()
    blk = _blockwise_matrix_exp(pm64, grp.irrep_dims)
    full = torch.linalg.matrix_exp(pm64)
    assert float((blk - full).abs().max()) < 1e-12

    # and the public pair (as called by compute_transport_operators) matches the full path
    ep_blk, en_blk = stable_matrix_exp_pair(pm, block_dims=grp.irrep_dims)
    ep_full, en_full = stable_matrix_exp_pair(pm, block_dims=None)
    assert float((ep_blk - ep_full).abs().max()) < 1e-6
    assert float((en_blk - en_full).abs().max()) < 1e-6


# --- P0 #2: the E-step fused-transport selector (factored on flat+equal-block, dense otherwise) ---
def test_estep_fuses_flat_block_glk_to_factored_container():
    # block_glk (irrep_dims [4,4], multi equal block, flat) -> the E-step builds the FACTORED
    # container (no dense (B,N,N,K,K) Omega).
    from vfe3.geometry.transport import FactoredTransport
    from vfe3.inference.e_step import build_belief_transport
    grp, phi, mu, sig = _transport_inputs()
    out = build_belief_transport(phi, grp)
    assert isinstance(out, FactoredTransport)


def test_flat_builders_forward_norm_threshold_and_block_layout(monkeypatch):
    # Both flat builders preserve the caller's norm threshold and gauge block layout.
    import vfe3.geometry.transport as T
    calls = []
    orig = T.stable_matrix_exp_pair
    def spy(matrix, **kw):
        calls.append((kw.get("exp_fp64_norm_threshold"), kw.get("block_dims")))
        return orig(matrix, **kw)
    monkeypatch.setattr(T, "stable_matrix_exp_pair", spy)
    grp, phi, _, _ = _transport_inputs()
    T.build_factored_transport(phi, grp, exp_fp64_norm_threshold=3.0)
    compute_transport_operators(phi, grp, exp_fp64_norm_threshold=3.0)
    assert calls == [(3.0, [4, 4]), (3.0, [4, 4])]


def test_compose_bch_blocked_matches_dense():
    # The blocked Dynkin cascade (brackets on the (H, d, d) diagonal-block stacks) must equal
    # the dense (K, K) cascade for a block-diagonal embedding, at every truncation order and
    # under the broadcast (unbatched positional element) pattern, in value AND gradient.
    from vfe3.geometry.lie_ops import compose_bch
    grp, phi, _, _ = _transport_inputs()                   # block_glk K=8, irrep_dims [4,4]
    n_gen = grp.generators.shape[0]
    torch.manual_seed(1)
    coords = 0.1 * torch.randn(5, n_gen)                   # unbatched positional element
    for order in (1, 2, 3, 4):
        p1 = phi.clone().requires_grad_(True)
        p2 = phi.clone().requires_grad_(True)
        dense = compose_bch(p1, coords, grp.generators, order=order, block_dims=None)
        blocked = compose_bch(p2, coords, grp.generators, order=order,
                              block_dims=list(grp.irrep_dims))
        assert torch.allclose(dense, blocked, atol=1e-6), f"order={order}"
        g_dense, = torch.autograd.grad(dense.sum(), p1)
        g_blocked, = torch.autograd.grad(blocked.sum(), p2)
        assert torch.allclose(g_dense, g_blocked, atol=1e-6), f"order={order} grad"


def test_estep_single_block_groups_stay_dense():
    # Single-block groups (block_glk with one head, so_n defining rep) report irrep_dims [K] -> the
    # fused route is NOT taken; the E-step keeps the dense (N,N,K,K)/(B,N,N,K,K) Omega tensor.
    from vfe3.geometry.transport import FactoredTransport
    from vfe3.inference.e_step import build_belief_transport
    torch.manual_seed(0)
    for name, ctor in (("block_glk_1", lambda: get_group("block_glk")(8, 1)),
                       ("so_n_l1", lambda: get_group("so_n")(8, group_n=8, irrep_spec=[("l1", 1)]))):
        grp = ctor()
        n_gen = grp.generators.shape[0]
        phi = 0.2 * torch.randn(2, 5, n_gen)
        out = build_belief_transport(phi, grp)
        assert not isinstance(out, FactoredTransport), name
        assert torch.is_tensor(out), name


