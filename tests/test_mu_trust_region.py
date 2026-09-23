r"""E-step mu trust-region (default OFF).

This passes every E-step mean update through a per-coordinate whitened box (or Mahalanobis ball)
trust region (`apply_mu_trust_region`, winning run `e_mu_q_trust=5.0`, `mu_trust_mode='box'`),
bounding an otherwise-unbounded update. The opt-in knob's default
(`e_mu_q_trust=None`) reproduces the unbounded behavior bit-for-bit.

box  : whitened = delta_mu / sqrt(sigma); clamp to +/-trust; un-whiten -> per-coord +/-trust*sigma^.5
ball : scale the whole whitened vector by min(trust/||whitened||_2, 1) (direction-preserving)
"""

import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import e_step_iteration
from vfe3.numerics import apply_mu_trust_region


def _belief(N=3, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    grp = get_group("glk")(K)
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

def test_box_clamps_whitened_per_coordinate():
    delta = torch.tensor([[10.0, 3.0]]); sigma = torch.tensor([[1.0, 1.0]])
    out = apply_mu_trust_region(delta, sigma, trust=5.0, mode="box", is_diagonal=True)
    assert torch.allclose(out, torch.tensor([[5.0, 3.0]]))          # 10->5 clamped; 3 kept


def test_box_uses_sigma_whitening():
    delta = torch.tensor([[12.0]]); sigma = torch.tensor([[4.0]])   # scale 2 -> bound 10
    out = apply_mu_trust_region(delta, sigma, trust=5.0, mode="box", is_diagonal=True)
    assert torch.allclose(out, torch.tensor([[10.0]]))


def test_ball_scales_direction_preserving():
    delta = torch.tensor([[6.0, 8.0]]); sigma = torch.tensor([[1.0, 1.0]])   # ||.||=10>5 -> x0.5
    out = apply_mu_trust_region(delta, sigma, trust=5.0, mode="ball", is_diagonal=True)
    assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))


def test_full_cov_uses_sigma_diagonal():
    delta = torch.tensor([[10.0, 3.0]]); sigma_full = torch.eye(2).unsqueeze(0)   # diag [1,1]
    out = apply_mu_trust_region(delta, sigma_full, trust=5.0, mode="box", is_diagonal=False)
    assert torch.allclose(out, torch.tensor([[5.0, 3.0]]))


def test_no_op_within_trust():
    delta = torch.tensor([[1.0, -2.0]]); sigma = torch.tensor([[1.0, 1.0]])
    for mode in ("box", "ball"):
        out = apply_mu_trust_region(delta, sigma, trust=5.0, mode=mode, is_diagonal=True)
        assert torch.allclose(out, delta)


# --- config + e_step integration ---

def test_config_can_explicitly_disable_mean_trust():
    cfg = VFE3Config(e_mu_q_trust=None, mu_trust_mode="ball")
    assert cfg.e_mu_q_trust is None                                 # explicit unbounded profile
    assert cfg.mu_trust_mode == "ball"


def test_e_step_trust_none_is_current_behavior():
    b, mu_p, sigma_p, grp = _belief()
    off  = e_step_iteration(b, mu_p, sigma_p, grp, e_q_mu_lr=0.5, e_phi_lr=0.0)
    none = e_step_iteration(b, mu_p, sigma_p, grp, e_q_mu_lr=0.5, e_phi_lr=0.0, e_mu_q_trust=None)
    assert torch.equal(off.mu, none.mu)                            # default == explicit None == unbounded


def test_e_step_trust_binds_changes_mu():
    b, mu_p, sigma_p, grp = _belief()
    off   = e_step_iteration(b, mu_p, sigma_p, grp, e_q_mu_lr=0.5, e_phi_lr=0.0)
    clamp = e_step_iteration(b, mu_p, sigma_p, grp, e_q_mu_lr=0.5, e_phi_lr=0.0,
                             e_mu_q_trust=1e-4, mu_trust_mode="box")
    assert not torch.allclose(off.mu, clamp.mu)                    # a tight box bounds the step
    assert (clamp.mu - b.mu).abs().max() < (off.mu - b.mu).abs().max()
