"""Full-Gaussian covariance transport keeps the SPD sandwich in float64."""

from __future__ import annotations


import torch

from vfe3.families.gaussian import FullGaussian


def test_genuine_float64_full_gaussian_still_returns_float64() -> None:
    mu = torch.zeros(1, 4, dtype=torch.float64)
    sigma = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4)

    divergence = FullGaussian(mu, sigma).renyi_closed_form(
        FullGaussian(mu, sigma),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )

    assert divergence.dtype == torch.float64


def test_genuine_float64_covariance_with_float32_mean_returns_float64() -> None:
    mu = torch.zeros(1, 4, dtype=torch.float32)
    sigma = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4)

    divergence = FullGaussian(mu, sigma).renyi_closed_form(
        FullGaussian(mu, sigma),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )

    assert divergence.dtype == torch.float64


