r"""Tests for lambda_beta: the belief-coupling weight.

lambda_beta scales the WHOLE belief-coupling block of F -- sum_ij [ beta_ij E_ij +
tau beta_ij log(beta_ij/pi_ij) ] -- relative to the alpha self-term. The correctness
invariant: it scales coupling AND entropy by the same factor and leaves beta = softmax(-E/tau)
alone, so (a) the gradient is AFFINE in lambda_beta (a lambda-into-softmax leak would make it
nonlinear) and (b) the analytic kernel (which scales only its pair term) agrees with the autograd
oracle (which differentiates lambda_beta*F_red).
"""

from tests.profiles import test_config

import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators


def _setup(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("block_glk")(K, 1)
    phi = 0.15 * torch.randn(1, N, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    mu = torch.randn(N, K, generator=g); sigma = torch.rand(N, K, generator=g) + 0.5
    mu_p = torch.randn(N, K, generator=g); sigma_p = torch.rand(N, K, generator=g) + 0.5
    return mu, sigma, mu_p, sigma_p, omega


def test_config_rejects_negative_lambda_beta():
    with pytest.raises(ValueError, match="lambda_beta must be >= 0"):
        test_config(vocab_size=20, embed_dim=4, n_heads=2, lambda_beta=-0.5)
