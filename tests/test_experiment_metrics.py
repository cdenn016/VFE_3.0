r"""Tests for the EXP-7 (rank collapse) and EXP-3 (Sigma_q calibration) foundational metrics
added in the 2026-06-21 experiment buildout: rank_one_residual, depth_decay_rate, sigma_trace,
spearman_rho, cv. Pure tensor functions; device-agnostic (CPU)."""
import pytest
import torch

from vfe3.metrics import cv, depth_decay_rate, rank_one_residual, sigma_trace, spearman_rho


def test_sigma_trace_diagonal_and_full_agree():
    torch.manual_seed(0)
    s = torch.rand(4, 5) + 0.1
    assert torch.allclose(sigma_trace(s), s.sum(-1))
    assert torch.allclose(sigma_trace(torch.diag_embed(s), diagonal=False), s.sum(-1), atol=1e-6)


def test_rank_one_residual_zero_when_all_tokens_identical():
    x = torch.randn(1, 7).expand(6, 7).contiguous()             # every row equal -> rank one
    assert rank_one_residual(x).item() < 1e-6


def test_rank_one_residual_positive_and_shrinks_under_common_shift():
    torch.manual_seed(1)
    x = torch.randn(20, 8)
    r = rank_one_residual(x).item()
    assert 0.0 < r <= 1.0 + 1e-6
    # a common offset moves xbar with the cloud (numerator fixed) but grows ||X||_F -> smaller r
    assert rank_one_residual(x + 50.0).item() < r


def test_depth_decay_rate_recovers_known_slope():
    b, n = -0.4, 6
    curve = torch.exp(b * torch.arange(n, dtype=torch.float32))
    assert abs(depth_decay_rate(curve) - b) < 1e-5


def test_spearman_is_rank_invariant():
    x = torch.arange(10, dtype=torch.float32)
    assert abs(spearman_rho(x, 2 * x + 1) - 1.0) < 1e-6
    assert abs(spearman_rho(x, torch.exp(x)) - 1.0) < 1e-6      # any increasing transform -> +1
    assert abs(spearman_rho(x, -x) + 1.0) < 1e-6
    assert abs(spearman_rho(x, x.flip(0)) + 1.0) < 1e-6


def test_cv_constant_zero_and_monotone_in_spread():
    assert cv(torch.full((5,), 3.0)) < 1e-9
    lo = cv(torch.tensor([1.0, 1.1, 0.9, 1.05, 0.95]))
    hi = cv(torch.tensor([1.0, 5.0, 0.2, 3.0, 0.1]))
    assert hi > lo > 0.0


def test_metric_input_guards():
    with pytest.raises(ValueError):
        depth_decay_rate(torch.tensor([1.0]))
    with pytest.raises(ValueError):
        spearman_rho(torch.arange(3.0), torch.arange(4.0))
    with pytest.raises(ValueError):
        cv(torch.tensor([1.0]))
