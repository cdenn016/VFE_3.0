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

import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators, transport_covariance, transport_mean
from vfe3.model.model import VFEModel


# --- frozen model-forward oracle (B=2, distinct sequences, seed 0) ---
_FWD_LOSS       = 2.4850413799
_FWD_LOGITS_SUM = -0.1507924795


def _fwd_cfg() -> VFE3Config:
    # pos_phi="none" pins the canonical no-positional-composition forward these checksums were
    # captured on; the perf refactor this oracle guards (factored transport / per-block exp / batch
    # vectorization) is independent of the pos_phi gauge composition.
    # These checksums were captured on the KL-to-prior decoder that the oracle guards.
    return VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=2,
                      n_e_steps=2, e_phi_lr=0.1, pos_phi="none",
                      decode_mode="diagonal_chunked")


def test_beta_to_coordinate_equal_blocks_matches_repeat_interleave():
    # The equal-block expand/reshape fast path (avoids the repeat_interleave gather) must be
    # BIT-identical to the repeat_interleave reference: coordinate k carries head h(k)'s beta_ij.
    from vfe3.gradients.kernels import _beta_to_coordinate
    torch.manual_seed(0)
    H, N, d = 3, 5, 4
    beta = torch.softmax(torch.randn(H, N, N), dim=-1)
    irrep = [d, d, d]
    out = _beta_to_coordinate(beta, irrep, H * d)
    ref = torch.repeat_interleave(beta.movedim(-3, -1), torch.tensor(irrep), dim=-1)
    assert torch.equal(out, ref)
    for h in range(H):
        assert torch.equal(out[..., h * d:(h + 1) * d], beta[h].unsqueeze(-1).expand(N, N, d))


def test_model_forward_matches_frozen_oracle():
    torch.manual_seed(0)
    m = VFEModel(_fwd_cfg())
    tok = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 0]])
    tgt = torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1]])
    with torch.no_grad():
        _, loss, ce = m(tok, tgt)
        # logits is None on the fused-CE branch by design (audit 2026-08-06); the default
        # decode_mode is chunked now, so take them from the unfused no-targets path.
        logits = m(tok)
    assert logits.shape == (2, 6, 12)
    assert abs(float(loss) - _FWD_LOSS) < 1e-5
    assert abs(float(logits.sum()) - _FWD_LOGITS_SUM) < 1e-5


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


# --- frozen transport oracle (block_glk K=8 n_heads=2 -> irrep_dims [4,4], seed 0) ---
_OMEGA_SUM  = 401.9380187988
_MT_ABS_SUM = 393.5396728516
_ST_SUM     = 532.2587280273


def _transport_inputs():
    torch.manual_seed(0)
    grp = get_group("block_glk")(8, 2)
    n_gen = grp.generators.shape[0]
    phi = 0.2 * torch.randn(2, 5, n_gen)
    mu = torch.randn(2, 5, 8)
    sig = torch.rand(2, 5, 8) + 0.5
    return grp, phi, mu, sig


def test_transport_matches_frozen_oracle():
    grp, phi, mu, sig = _transport_inputs()
    td = compute_transport_operators(phi, grp)
    omega = td["Omega"]                                    # (B, N, N, K, K)
    mt = transport_mean(omega, mu)                         # (B, N, N, K)
    st = transport_covariance(omega, sig)                  # (B, N, N, K) diagonal sandwich
    assert abs(float(omega.sum()) - _OMEGA_SUM) < 1e-3
    assert abs(float(mt.abs().sum()) - _MT_ABS_SUM) < 1e-3
    assert abs(float(st.sum()) - _ST_SUM) < 1e-3


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
    # container (no dense (B,N,N,K,K) Omega), which transport_mean/covariance consume on a fast path.
    from vfe3.geometry.transport import FactoredTransport
    from vfe3.inference.e_step import build_belief_transport
    grp, phi, mu, sig = _transport_inputs()
    out = build_belief_transport(phi, grp, transport_mode="flat")
    assert isinstance(out, FactoredTransport)


def test_estep_fuses_tied_block_glk_to_factored_container():
    # tied_block_glk is also block-diagonal with equal blocks -> qualifies for the fused route.
    from vfe3.geometry.transport import FactoredTransport
    from vfe3.inference.e_step import build_belief_transport
    grp = get_group("tied_block_glk")(8, 2)
    n_gen = grp.generators.shape[0]
    torch.manual_seed(0)
    phi = 0.2 * torch.randn(2, 5, n_gen)
    out = build_belief_transport(phi, grp, transport_mode="flat")
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


def test_pair_contract_matches_beta_coord_reference():
    # _pair_contract (beta in compact per-head form against a head-shaped view of the pair
    # difference) must reproduce the legacy beta_coord-broadcast einsum on every branch:
    # per-head equal blocks, single block, and the unequal-tower fallback.
    from vfe3.gradients.kernels import _beta_to_coordinate, _pair_contract
    torch.manual_seed(2)
    B, N, H, d = 2, 5, 3, 4
    K = H * d
    diff = torch.randn(B, N, N, K)
    beta_h = torch.softmax(torch.randn(B, H, N, N), dim=-1)
    ref = torch.einsum("...ijk,...ijk->...ik",
                       _beta_to_coordinate(beta_h, [d] * H, K), diff)
    assert torch.allclose(_pair_contract(beta_h, diff, [d] * H), ref, atol=1e-6)

    beta_1 = torch.softmax(torch.randn(B, N, N), dim=-1)   # single block
    ref_1 = torch.einsum("...ijk,...ijk->...ik",
                         beta_1.unsqueeze(-1).expand(B, N, N, K), diff)
    assert torch.allclose(_pair_contract(beta_1, diff, None), ref_1, atol=1e-6)
    assert torch.allclose(_pair_contract(beta_1, diff, [K]), ref_1, atol=1e-6)

    irrep_uneq = [3, 9]                                    # unequal blocks -> gather fallback
    beta_u = torch.softmax(torch.randn(B, 2, N, N), dim=-1)
    ref_u = torch.einsum("...ijk,...ijk->...ik",
                         _beta_to_coordinate(beta_u, irrep_uneq, K), diff)
    assert torch.allclose(_pair_contract(beta_u, diff, irrep_uneq), ref_u, atol=1e-6)


def test_estep_single_block_groups_stay_dense():
    # glk and so_k report irrep_dims [K] (single block) -> the fused route is NOT taken; the
    # E-step keeps the dense (N,N,K,K)/(B,N,N,K,K) Omega tensor exactly as today.
    from vfe3.geometry.transport import FactoredTransport
    from vfe3.inference.e_step import build_belief_transport
    torch.manual_seed(0)
    for name, ctor in (("glk", lambda: get_group("glk")(8)),
                       ("so_k", lambda: get_group("so_k")(8))):
        grp = ctor()
        n_gen = grp.generators.shape[0]
        phi = 0.2 * torch.randn(2, 5, n_gen)
        out = build_belief_transport(phi, grp, transport_mode="flat")
        assert not isinstance(out, FactoredTransport), name
        assert torch.is_tensor(out), name


def test_estep_cross_coupled_block_glk_stays_dense():
    # A cross-coupled block_glk reports irrep_dims [K] (its group elements have off-block entries),
    # so it is NOT block-diagonal with equal blocks -> the fused route is excluded, dense kept.
    from vfe3.geometry.transport import FactoredTransport
    from vfe3.inference.e_step import build_belief_transport
    grp = get_group("block_glk")(8, 2, cross_couplings=[(0, 1)])
    assert grp.irrep_dims == [8]
    n_gen = grp.generators.shape[0]
    torch.manual_seed(0)
    phi = 0.2 * torch.randn(2, 5, n_gen)
    out = build_belief_transport(phi, grp, transport_mode="flat")
    assert not isinstance(out, FactoredTransport)
    assert torch.is_tensor(out)
