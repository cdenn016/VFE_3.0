r"""E-step mu trust region: the Mahalanobis ball.

Every E-step mean update passes through `apply_mu_trust_region`, which scales the whole whitened
vector by min(trust/||whitened||_2, 1) (direction-preserving). `e_mu_q_trust=None` reproduces the
unbounded update bit-for-bit. The per-coordinate box was retired on October 2, 2026.
"""

from tests.profiles import test_config

import torch

from vfe3.belief import BeliefState
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import e_step_iteration
from vfe3.numerics import apply_mu_trust_region


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


# --- helper unit tests ---

def test_ball_uses_sigma_whitening():
    delta = torch.tensor([[12.0]]); sigma = torch.tensor([[4.0]])   # scale 2 -> bound 10
    out = apply_mu_trust_region(delta, sigma, trust=5.0, is_diagonal=True)
    assert torch.allclose(out, torch.tensor([[10.0]]))


def test_ball_scales_direction_preserving():
    delta = torch.tensor([[6.0, 8.0]]); sigma = torch.tensor([[1.0, 1.0]])   # ||.||=10>5 -> x0.5
    out = apply_mu_trust_region(delta, sigma, trust=5.0, is_diagonal=True)
    assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))


def test_full_cov_ball_is_the_mahalanobis_norm():
    delta = torch.tensor([[6.0, 8.0]])
    sigma_full = torch.tensor([[[4.0, 0.0], [0.0, 1.0]]])            # whitened [3, 8], norm sqrt(73)
    out = apply_mu_trust_region(delta, sigma_full, trust=5.0, is_diagonal=False)
    assert torch.allclose(out, delta * (5.0 / 73.0 ** 0.5))
    whitened = torch.linalg.solve(torch.linalg.cholesky(sigma_full), out.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(whitened.norm(dim=-1), torch.tensor([5.0]))


def test_no_op_within_trust():
    delta = torch.tensor([[1.0, -2.0]]); sigma = torch.tensor([[1.0, 1.0]])
    out = apply_mu_trust_region(delta, sigma, trust=5.0, is_diagonal=True)
    assert torch.allclose(out, delta)


# --- config + e_step integration ---

def test_config_can_explicitly_disable_mean_trust():
    cfg = test_config(e_mu_q_trust=None)
    assert cfg.e_mu_q_trust is None                                 # explicit unbounded profile
    assert not hasattr(cfg, "mu_trust_mode")
