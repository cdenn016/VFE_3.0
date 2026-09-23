"""Common-frame scoring must respect the CUDA eigensolver batch ceiling."""

import os

import pytest
import torch

from vfe3.families.full_gaussian_score import common_frame_kl_energy
from vfe3.geometry.transport import CompactFactoredTransport


@pytest.mark.parametrize("precision_policy", ("fp32_escalate", "fp64"))
def test_common_frame_scoring_above_cuda_eigensolver_batch_ceiling(precision_policy: str) -> None:
    device = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))
    # CUDA: 8192 batches * 2 tokens * 2 heads = 32768 independent 2x2 matrices.
    # CPU keeps the same analytical value/gradient check tiny. No VFEModel is built.
    batch = 8192 if device.type == "cuda" else 2
    mu_seed = torch.tensor([[[0., 0., 0., 0.], [.5, 0., .5, 0.]]],
                           device=device, requires_grad=True)
    mu = mu_seed.expand(batch, 2, 4)
    Sigma = torch.eye(4, device=device).expand(batch, 2, 4, 4)
    frames = torch.eye(2, device=device).expand(batch, 2, 2, 2, 2)
    omega = CompactFactoredTransport(frames, frames, 4, same_frame_flat_cocycle=True)
    actual = common_frame_kl_energy(
        mu, Sigma, mu, Sigma, omega, kl_max=10., precision_policy=precision_policy,
        irrep_dims=[2, 2])
    assert actual is not None
    assert actual.shape == (batch, 2, 2, 2)
    difference = (mu_seed[:, :, None, :] - mu_seed[:, None, :, :]).reshape(1, 2, 2, 2, 2)
    expected = (.5 * difference.square().sum(-1)).movedim(-1, -3)
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-6, atol=1e-7)
    gradient, = torch.autograd.grad(actual.sum() / batch, (mu_seed,))
    reference_gradient, = torch.autograd.grad(expected.sum(), (mu_seed,))
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-6, atol=1e-7)
