import math

import torch

from vfe3.geometry.generators import generate_glk_multihead, generate_son
from vfe3.geometry.lie_ops import (
    embed_phi,
    extract_phi,
    lie_bracket_coords,
)


def test_embed_extract_roundtrip_independent_basis():
    # gl(2) elementary basis is orthonormal under Frobenius -> extract(embed(c)) == c.
    G = generate_glk_multihead(2, 1)                                   # (4, 2, 2)
    c = torch.randn(3, 4)
    out = extract_phi(embed_phi(c, G), G)
    assert torch.allclose(out, c, atol=1e-6)


def test_embed_extract_projection_overcomplete():
    # The trace-projected gl(K) spanning set for sl(K) is OVERCOMPLETE (rank K^2-1):
    # extract(embed(c)) need NOT equal c, but embed o extract o embed == embed.
    G = generate_glk_multihead(3, 1)                      # gl(3) elementary basis (9, 3, 3)
    trace_dir = torch.eye(3) / math.sqrt(3)
    G = G - torch.einsum("aij,ij->a", G, trace_dir)[:, None, None] * trace_dir   # sl(3) set, rank 8
    c = torch.randn(2, G.shape[0])
    M = embed_phi(c, G)
    M2 = embed_phi(extract_phi(M, G), G)
    assert torch.allclose(M2, M, atol=1e-5)


def test_bracket_so3_structure_constants():
    # generate_son(3) basis: G0=E01-E10, G1=E02-E20, G2=E12-E21.
    # Hand-derived: [G0,G1]=-G2, [G0,G2]=+G1, [G1,G2]=-G0.
    G = generate_son(3)                                   # (3, 3, 3)
    e = torch.eye(3)
    c01 = lie_bracket_coords(e[0], e[1], G)
    c02 = lie_bracket_coords(e[0], e[2], G)
    c12 = lie_bracket_coords(e[1], e[2], G)
    assert torch.allclose(c01, torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert torch.allclose(c02, torch.tensor([0.0, 1.0,  0.0]), atol=1e-6)
    assert torch.allclose(c12, torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)

