r"""Tests for the principled exact route: the minimal agent on ``gaussian_frame_diagonal`` and the one-switch toggle.

The minimal-agent readout on the frame-diagonal family must be the full-covariance readout at
:math:`\Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top`: the same posterior means, and stored variances
:math:`c_i^+ = \operatorname{diag}(U_i^{-1}\Sigma_i^+U_i^{-\top})` (the family's reference M-projection), with the floor anchored at
:math:`c_p` against :math:`U\operatorname{diag}(c_p)U^\top`. The toggle tests resolve the test baseline and the launcher
through :func:`vfe3.exact_route.principled_exact_route_overrides`, check that every requirement holds, that the switch
is idempotent and keeps a learned regime transition, that unmet requirements are reported, and that the resolved tiny
model trains a step. CPU only, one thread, tiny models.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest
import torch

from tests.profiles import config_values
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.exact_route import (EXACT_ROUTE_SETTINGS, describe_principled_exact_route_overrides,
                              principled_exact_route_failures, principled_exact_route_overrides)
from vfe3.model.model import VFEModel

torch.set_num_threads(1)
DT = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _resolved(**overrides) -> VFE3Config:
    values = config_values(**overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**{**values, **principled_exact_route_overrides(values)})


def test_minimal_agent_on_frame_diagonal_equals_full_covariance_readout() -> None:
    cfg = _resolved(embed_dim=4, n_heads=2, max_seq_len=5, block_memory_regression_rank=2)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double()
    op = model.relational_operators
    assert op.frame_diagonal and op.predictive_readout == "minimal_agent"
    B, N, K = 2, 5, cfg.embed_dim
    U = torch.linalg.matrix_exp(0.4 * torch.randn(B, N, K, K, generator=_gen(1), dtype=DT))
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=False)
    mu = torch.randn(B, N, K, generator=_gen(2), dtype=DT)
    c = 0.5 + torch.rand(B, N, K, generator=_gen(3), dtype=DT)
    c_p = 0.5 + torch.rand(B, N, K, generator=_gen(4), dtype=DT)
    log_prior = torch.zeros(N, N, dtype=DT).masked_fill(torch.ones(N, N).triu(1).bool(), float("-inf"))
    mu_d, c_plus = op.minimal_agent_update(mu, c, frame, log_prior, return_covariance=True, floor_anchor=c_p)
    local = lambda v: U @ torch.diag_embed(v) @ U.transpose(-1, -2)          # noqa: E731
    op.frame_diagonal = False                                                # the same operator, full covariance
    try:
        mu_f, Sigma_plus = op.minimal_agent_update(mu, local(c), frame, log_prior, return_covariance=True,
                                                   floor_anchor=local(c_p))
    finally:
        op.frame_diagonal = True
    U_inv = torch.linalg.inv(U)
    C_plus = U_inv @ Sigma_plus @ U_inv.transpose(-1, -2)
    assert c_plus.shape == (B, N, K)
    assert torch.allclose(mu_d, mu_f, rtol=1e-10, atol=1e-11)
    assert torch.allclose(c_plus, C_plus.diagonal(dim1=-2, dim2=-1), rtol=1e-10, atol=1e-11)
    # The projection discards real off-diagonal reference covariance, so the comparison has power.
    off = C_plus - torch.diag_embed(C_plus.diagonal(dim1=-2, dim2=-1))
    assert float(off.abs().max()) > 1e-4
    # Passive action (mu, U) -> (g mu, g U), c fixed: mu^+ -> g mu^+ and c^+ unchanged.
    g = torch.linalg.matrix_exp(0.6 * torch.randn(B, N, K, K, generator=_gen(5), dtype=DT))
    moved = CanonicalFrameContext(g @ U, torch.linalg.inv(g @ U), block_diagonal=False)
    mu_g, c_g = op.minimal_agent_update((g @ mu.unsqueeze(-1)).squeeze(-1), c, moved, log_prior,
                                        return_covariance=True, floor_anchor=c_p)
    assert torch.allclose(mu_g, (g @ mu_d.unsqueeze(-1)).squeeze(-1), rtol=1e-9, atol=1e-10)
    assert torch.allclose(c_g, c_plus, rtol=1e-9, atol=1e-10)


def test_toggle_resolves_baseline_and_is_idempotent() -> None:
    values = config_values()
    overrides = principled_exact_route_overrides(values)
    assert overrides and describe_principled_exact_route_overrides(values, overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(**{**values, **overrides})
    assert principled_exact_route_failures(cfg) == []
    for key, value in EXACT_ROUTE_SETTINGS.items():
        assert getattr(cfg, key) == value, key
    assert cfg.block_memory_regime_transition == "sticky"
    assert principled_exact_route_overrides({**values, **overrides}) == {}
    kept = principled_exact_route_overrides({**values, "block_memory_regime_transition": "learned"})
    assert "block_memory_regime_transition" not in kept


def test_failures_name_unmet_requirements() -> None:
    values = config_values()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(**{**values, **principled_exact_route_overrides(values), "decode_tau": 0.1})
        plain = VFE3Config(**values)
    assert any(f.startswith("decode_tau=") for f in principled_exact_route_failures(cfg))
    names = principled_exact_route_failures(plain)
    assert any(f.startswith("predictive_readout=") for f in names)
    assert any(f.startswith("block_memory_regime_transition=") for f in names)


def test_toggled_tiny_model_trains_a_step() -> None:
    cfg = _resolved(n_layers=2, max_seq_len=4, block_mlp_expansion=2, block_memory_regression_rank=2)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=_gen(9))
        model.train()
        _, loss, _ = model(tokens, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.block_mlps[0].regime_stay_logit.grad is not None


def test_launcher_toggle_defaults_off_and_resolves() -> None:
    path = Path(__file__).resolve().parents[1] / "train_vfe3.py"
    spec = importlib.util.spec_from_file_location("_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)                                      # module level only: no training
    assert module.principled_exact_route is False
    config = {k: v for k, v in module.config.items() if k in VFE3Config.__dataclass_fields__}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(**{**config, **principled_exact_route_overrides(config)})
    assert principled_exact_route_failures(cfg) == []
