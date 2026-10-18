"""Shared tiny full-state Gaussian and canonical-frame fixture."""
from tests.profiles import make_relational_operators
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.transport import FactoredTransport


def _fixture() -> tuple:
    mu = torch.tensor([[[0.2, -0.4, 0.1, 0.7], [1.1, 0.5, -0.3, 0.8],
                        [0.6, 1.2, 0.9, -0.2]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([[[0.8, 1.2, 0.9, 1.4], [1.1, 0.6, 1.5, 0.7],
                           [0.7, 1.3, 0.8, 1.1]]], dtype=mu.dtype)
    U = torch.eye(4, dtype=mu.dtype).expand(1, 3, 4, 4).clone()
    U[..., 0, 1] = 0.3
    U[..., 2, 3] = -0.2
    V = torch.linalg.inv(U)
    frame = CanonicalFrameContext(U, V)
    omega = FactoredTransport(U, V, [2, 2], same_frame_flat_cocycle=True)
    ops = make_relational_operators([2, 2], 11).double()
    prior = torch.tensor([[0., -torch.inf, -torch.inf], [0., 0., -torch.inf],
                          [0., 0., 0.]], dtype=mu.dtype)
    return mu, sigma, omega, frame, ops, prior
