r"""Config-wired belief-table init scales: mu_init_std, sigma_init, phi_scale.

PriorBank already accepted these kwargs; VFE3Config now exposes them and VFEModel threads
them through, so the initial belief tables mu_embed ~ N(0, mu_init_std^2), every coordinate
variance = sigma_init (stored as log), and phi_embed ~ N(0, phi_scale^2) are adjustable from
config (and sweepable in ablation.py). sigma_init must be > 0 (log is taken); mu_init_std and
phi_scale may be 0 (deterministic zero table).
"""

import math

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

BASE = dict(vocab_size=512, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)


def test_defaults_match_prior_bank_defaults():
    cfg = VFE3Config(**BASE)
    assert cfg.mu_init_std == 0.02
    assert cfg.sigma_init == 1.0
    assert cfg.phi_scale == 0.01


def test_scales_thread_through_to_the_belief_tables():
    cfg = VFE3Config(**BASE, mu_init_std=0.5, sigma_init=4.0, phi_scale=0.3)
    pb = VFEModel(cfg).prior_bank
    # std of a large random table tracks the configured scale (loose tolerance for sample noise)
    assert pb.mu_embed.std().item() == pytest.approx(0.5, rel=0.1)
    assert pb.phi_embed.std().item() == pytest.approx(0.3, rel=0.1)
    # sigma is a constant table at log(sigma_init), not random spread
    assert torch.allclose(pb.sigma_log_embed, torch.full_like(pb.sigma_log_embed, math.log(4.0)))


def test_zero_scales_give_deterministic_zero_tables():
    cfg = VFE3Config(**BASE, mu_init_std=0.0, phi_scale=0.0)
    pb = VFEModel(cfg).prior_bank
    assert torch.count_nonzero(pb.mu_embed) == 0
    assert torch.count_nonzero(pb.phi_embed) == 0


def test_nonpositive_sigma_init_is_rejected():
    with pytest.raises(ValueError, match="sigma_init must be positive"):
        VFE3Config(**BASE, sigma_init=0.0)


def test_negative_mu_init_std_is_rejected():
    with pytest.raises(ValueError, match="mu_init_std must be >= 0"):
        VFE3Config(**BASE, mu_init_std=-0.1)
