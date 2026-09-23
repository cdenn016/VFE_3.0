import math

import torch

from vfe3.geometry.generators import generate_glk, generate_son
from vfe3.geometry.lie_ops import embed_phi, retract_glk, retract_son


def test_retract_glk_trust_region_clamps_step():
    # Small trust_region caps the per-step coordinate-norm: ||update|| <= trust_region,
    # so euclidean compose keeps ||phi_new|| <= ||phi|| + trust_region (no amplification).
    G = generate_glk(3)                                   # (9,3,3)
    phi = 0.1 * torch.randn(5, 9)
    delta = 50.0 * torch.randn(5, 9)                      # large step -> trust region binds
    out = retract_glk(phi, delta, G, step_size=1.0, trust_region=0.1, max_norm=5.0)
    assert (out.norm(dim=-1) <= phi.norm(dim=-1) + 0.1 + 1e-5).all()


def test_retract_glk_max_norm_clamps_frame_to_ceiling():
    # Isolate the max-norm ceiling: a loose trust_region lets the huge aligned step
    # through, so compose drives ||phi_new|| past max_norm and the clamp pins it to
    # exactly max_norm (up to the eps=1e-6 bias in max_norm/(n_norm+eps)).
    G = generate_glk(3)                                   # (9,3,3)
    phi = torch.zeros(3, 9)
    delta = 1e3 * torch.ones(3, 9)                        # huge, aligned
    out = retract_glk(phi, delta, G, step_size=1.0, trust_region=100.0, max_norm=5.0)
    assert torch.allclose(out.norm(dim=-1), torch.full((3,), 5.0), atol=1e-3)


def test_retract_glk_euclidean_unclamped_is_plain_step():
    # Pin the UPDATE LOGIC directly: with both clamps disabled (trust_region=0,
    # max_norm=0) the euclidean retraction is exactly phi + step_size * delta.
    # The det>0 / orthogonality tests below are properties of exp(embed(.)) that
    # hold for ANY output, so this is the test that actually pins the formula.
    G = generate_glk(3)
    phi = 0.05 * torch.randn(4, 9)
    delta = 0.01 * torch.randn(4, 9)
    out = retract_glk(phi, delta, G, step_size=0.5, trust_region=0.0, max_norm=0.0, mode="euclidean")
    assert torch.allclose(out, phi + 0.5 * delta, atol=1e-6)


def test_retract_glk_keeps_det_positive():
    # det(exp(embed phi)) = exp(tr) > 0 always: the GL+(K) identity-component property.
    G = generate_glk(3)
    phi = 0.3 * torch.randn(8, 9)
    delta = torch.randn(8, 9)
    out = retract_glk(phi, delta, G)
    dets = torch.linalg.det(torch.linalg.matrix_exp(embed_phi(out, G)))
    assert (dets > 0).all()


def test_retract_son_stays_orthogonal():
    # SO(N): embed(phi) is skew -> exp is orthogonal with det +1 (group membership).
    G = generate_son(4)                                   # (6,4,4)
    phi = 0.2 * torch.randn(7, 6)
    delta = torch.randn(7, 6)
    out = retract_son(phi, delta, G, max_norm=math.pi)
    A = embed_phi(out, G)
    assert torch.allclose(A, -A.transpose(-1, -2), atol=1e-5)          # skew
    R = torch.linalg.matrix_exp(A)
    eye = torch.eye(4).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)     # orthogonal
    assert torch.allclose(torch.linalg.det(R), torch.ones(7), atol=1e-4)


from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.lie_ops import clamp_phi_trace, project_phi_to_slk


def _block_traces(phi, G, irrep_dims):
    A = embed_phi(phi, G)
    outs, start = [], 0
    for d in irrep_dims:
        end = start + d
        outs.append(A[..., start:end, start:end].diagonal(dim1=-2, dim2=-1).sum(-1))
        start = end
    return torch.stack(outs, dim=-1)                      # (..., n_blocks)


def test_project_slk_zeros_block_trace_and_unit_det():
    G = generate_glk_multihead(6, 2)                      # 2 blocks of gl(3)
    irrep = [3, 3]
    phi = 0.5 * torch.randn(5, G.shape[0])
    out = project_phi_to_slk(phi, G, irrep)
    assert torch.allclose(_block_traces(out, G, irrep), torch.zeros(5, 2), atol=1e-5)
    # det of each block's group element == 1
    A = embed_phi(out, G)
    for s, d in [(0, 3), (3, 3)]:
        blk = A[..., s:s + d, s:s + d]
        det = torch.linalg.det(torch.linalg.matrix_exp(blk))
        assert torch.allclose(det, torch.ones(5), atol=1e-4)


def test_clamp_phi_trace_bounds_block_trace():
    G = generate_glk_multihead(6, 2)
    irrep = [3, 3]
    phi = 2.0 * torch.randn(5, G.shape[0])                # large traces
    T = 0.5
    out = clamp_phi_trace(phi, G, irrep, trace_max=T)
    assert (_block_traces(out, G, irrep).abs() <= T + 1e-4).all()


from vfe3.geometry.groups import get_group


def test_project_slk_zeros_block_trace_under_tied_gauge():
    # tied_block_glk's generators kron(I_n, E_ij) make every block's trace functional IDENTICAL,
    # so the per-block-INDEPENDENT projection (coeffs = s / ||V_h||^2) over-subtracts by n_heads
    # (sign flip + factor-n_heads). The joint Gram solve coeffs = s @ pinv(V V^T) must still drive
    # each block's trace to 0 (det Omega_h = 1).
    grp = get_group("tied_block_glk")(6, 2)               # 2 tied blocks of gl(3)
    G, irrep = grp.generators, grp.irrep_dims
    torch.manual_seed(0)
    phi = 0.5 * torch.randn(5, G.shape[0])
    out = project_phi_to_slk(phi, G, irrep)
    assert torch.allclose(_block_traces(out, G, irrep), torch.zeros(5, len(irrep)), atol=1e-5)


def test_clamp_phi_trace_bounds_block_trace_under_tied_gauge():
    grp = get_group("tied_block_glk")(6, 2)
    G, irrep = grp.generators, grp.irrep_dims
    torch.manual_seed(0)
    phi = 2.0 * torch.randn(5, G.shape[0])
    T = 0.5
    out = clamp_phi_trace(phi, G, irrep, trace_max=T)
    assert (_block_traces(out, G, irrep).abs() <= T + 1e-4).all()


def test_trace_projection_survives_bf16_autocast():
    # Under AMP the (V V^T) matmul is downcast and linalg.pinv has no bf16 kernel: both trace
    # solves must run in their own fp32 island (amp_dtype='bf16' + pos_phi_project_slk=True).
    G = generate_glk_multihead(6, 2)
    irrep = [3, 3]
    torch.manual_seed(0)
    phi = 0.5 * torch.randn(5, G.shape[0])
    T = 0.1
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        projected = project_phi_to_slk(phi, G, irrep)
        clamped = clamp_phi_trace(phi, G, irrep, trace_max=T)
    assert projected.dtype == phi.dtype and clamped.dtype == phi.dtype
    assert torch.allclose(_block_traces(projected, G, irrep), torch.zeros(5, 2), atol=1e-5)
    assert (_block_traces(clamped, G, irrep).abs() <= T + 1e-4).all()
    # fp32-exact: the island must not have let bf16 leak into the trace solve
    assert torch.allclose(projected, project_phi_to_slk(phi, G, irrep))
    assert torch.allclose(clamped, clamp_phi_trace(phi, G, irrep, trace_max=T))


from vfe3.geometry.groups import get_group
from vfe3.geometry.retraction import retract_phi


def test_retract_phi_glk_with_slk_projection():
    grp = get_group("block_glk")(6, 2)                    # block GL(3)^2, irrep [3,3]
    phi = 0.5 * torch.randn(4, grp.generators.shape[0])
    delta = torch.randn_like(phi)
    out = retract_phi(phi, delta, grp, project_slk=True)
    assert torch.allclose(_block_traces(out, grp.generators, grp.irrep_dims),
                          torch.zeros(4, 2), atol=1e-4)


def test_retract_phi_son_path_orthogonal_no_det_control():
    grp = get_group("so_k")(4)
    phi = 0.2 * torch.randn(4, grp.generators.shape[0])
    delta = torch.randn_like(phi)
    out = retract_phi(phi, delta, grp)                    # skew -> SO path, det control ignored
    R = torch.linalg.matrix_exp(embed_phi(out, grp.generators))
    eye = torch.eye(4).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-4)


def test_retract_phi_defaults_pick_group_constants():
    # The dispatcher selects max_norm from the group's compactness: non-compact
    # GL(K) -> 5.0 (bounds log-singular-values), compact SO(N) -> pi (bounds
    # principal angles). Override only trust_region (large) so the huge aligned
    # delta reaches the max-norm ceiling; the defaulted max_norm is what binds.
    # The frame saturates to exactly that group constant, discriminating 5.0 vs pi.
    grp_gl = get_group("glk")(3)
    phi_gl = torch.zeros(2, 9)
    delta_gl = 1e3 * torch.ones(2, 9)
    out_gl = retract_phi(phi_gl, delta_gl, grp_gl, trust_region=100.0)
    assert torch.allclose(out_gl.norm(dim=-1), torch.full((2,), 5.0), atol=1e-3)

    grp_so = get_group("so_k")(4)
    n_gen = grp_so.generators.shape[0]
    phi_so = torch.zeros(2, n_gen)
    delta_so = 1e3 * torch.ones(2, n_gen)
    out_so = retract_phi(phi_so, delta_so, grp_so, trust_region=100.0)
    assert torch.allclose(out_so.norm(dim=-1), torch.full((2,), math.pi), atol=1e-3)
