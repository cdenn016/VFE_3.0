import torch

from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.lie_ops import embed_phi, project_phi_to_slk


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


def test_trace_projection_survives_bf16_autocast():
    # Under AMP the (V V^T) matmul is downcast and linalg.pinv has no bf16 kernel: the trace
    # solve must run in its own fp32 island (amp_dtype='bf16' + pos_phi_project_slk=True).
    G = generate_glk_multihead(6, 2)
    irrep = [3, 3]
    torch.manual_seed(0)
    phi = 0.5 * torch.randn(5, G.shape[0])
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        projected = project_phi_to_slk(phi, G, irrep)
    assert projected.dtype == phi.dtype
    assert torch.allclose(_block_traces(projected, G, irrep), torch.zeros(5, 2), atol=1e-5)
    # fp32-exact: the island must not have let bf16 leak into the trace solve
    assert torch.allclose(projected, project_phi_to_slk(phi, G, irrep))
