r"""pullback_group under group_log without structure constants (audit 2026-10-16, F40).

Under ``m_phi_group_retraction='group_log'`` the Frobenius-pullback body direction is the closed form
:math:`\xi = U^{-1}\bar G` on rows inside the principal strip with :math:`\max_h\kappa_2(U_h) \le 10^3`; every
other row fell back to the certified series direction, which builds the dense gl(d) structure constants. Their
:math:`3\cdot 8 n_{\rm gen}^2 d^2` bytes refuse ``block_glk`` from d_head = 17 on (24 GiB at d_head = 32), so the
config refused pullback_group at the active head width even though off-strip blocks can be solved by inverting
:math:`\Psi_L(\mathrm{ad}_{X^\top})` in the eigenbasis of the chart block (the fisher_group route already does).
``m_phi_group_log_fallback='eigenbasis'`` solves those rows that way, rejects the rows the closed form leaves
uncertified for conditioning (:math:`\max_h\kappa_2(U_h) > 10^3`) or whose eigenbasis solve fails its gate, and
needs no structure constants, so the budget no longer applies under group_log. The default ``'series'`` keeps
the historical fallback.

Tiny float64 checks (d = 3) and config construction only; the cfg-to-optimizer line in
``vfe3/train.py build_optimizer`` is lane C1's.
"""
from __future__ import annotations

import math
import warnings

import pytest
import torch
from torch import nn

from tests.profiles import config_values, test_config
from vfe3.config import VFE3Config

F64 = torch.float64


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _wide(**overrides) -> VFE3Config:
    values = dict(gauge_group="block_glk", gauge_tying="per_head", embed_dim=64, n_heads=2, pos_phi="none",
                  m_phi_update_mode="pullback_group", m_phi_group_retraction="group_log")
    values.update(overrides)
    return test_config(**values)


# Config ------------------------------------------------------------------------------------------------

def test_default_is_the_series_fallback() -> None:
    # The test baseline pins the series fallback; the VFE3Config default only has to be admitted.
    assert config_values()["m_phi_group_log_fallback"] == "series"
    default = VFE3Config.__dataclass_fields__["m_phi_group_log_fallback"].default
    assert test_config(m_phi_group_log_fallback=default).m_phi_group_log_fallback == default
    with pytest.raises(ValueError, match="m_phi_group_log_fallback"):
        test_config(m_phi_group_log_fallback="exact")


def test_eigenbasis_fallback_admits_pullback_group_at_head_width_32() -> None:
    assert _wide(m_phi_group_log_fallback="eigenbasis").m_phi_group_log_fallback == "eigenbasis"
    with pytest.raises(ValueError, match="structure-constant workspace"):
        _wide()                                                           # series fallback: over budget
    with pytest.raises(ValueError, match="structure-constant workspace"):
        _wide(m_phi_group_retraction="chart", m_phi_group_log_fallback="eigenbasis")   # chart needs the series


@pytest.mark.parametrize("mode,retraction,inert", [
    ("adamw", "chart", True), ("pullback_group", "chart", True), ("pullback_group", "group_log", False),
])
def test_fallback_is_marked_inert_off_pullback_group_log(mode: str, retraction: str, inert: bool) -> None:
    values = dict(m_phi_update_mode=mode)
    if retraction == "group_log":
        values.update(gauge_group="block_glk", gauge_tying="per_head", m_phi_group_retraction="group_log")
    cfg = test_config(**values)
    assert ("m_phi_group_log_fallback" in cfg.inactive_fields()) is inert


def test_differential_mode_is_inert_under_the_eigenbasis_fallback() -> None:
    values = dict(gauge_group="block_glk", gauge_tying="per_head", m_phi_update_mode="pullback_group",
                  m_phi_group_retraction="group_log")
    assert "m_phi_group_differential_mode" not in test_config(**values).inactive_fields()
    inactive = test_config(m_phi_group_log_fallback="eigenbasis", **values).inactive_fields()
    assert "m_phi_group_differential_mode" in inactive


# Direction -----------------------------------------------------------------------------------------------

def _rotation(d: int, angle: float) -> torch.Tensor:
    X = torch.zeros(d, d, dtype=F64)
    X[0, 1], X[1, 0] = -angle, angle
    return X


def _charts() -> torch.Tensor:
    r"""(3, H=2, 3, 3): an on-strip row, a row with one block rotated 3.6 rad (off the strip), and a row with
    :math:`\kappa_2(U) = e^{8} > 10^3`."""
    gen = torch.Generator().manual_seed(3)
    X = 0.05 * torch.randn(3, 2, 3, 3, generator=gen, dtype=F64)
    X[1, 0] += _rotation(3, 3.6)
    X[2, 1] += torch.diag(torch.tensor([4.0, -4.0, 0.0], dtype=F64))
    return X


def _linear_loss_covector(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    r"""Chart covector :math:`M = \nabla_X \sum_h \langle C_h, e^{X_h}\rangle_F`, whose ambient gradient is
    :math:`\bar G = C` and whose exact body direction is :math:`\xi = U^{-1} C`."""
    X = X.clone().requires_grad_(True)
    (torch.linalg.matrix_exp(X) * C).sum().backward()
    return X.grad.detach()


def _forbid_structure_constants(monkeypatch) -> None:
    import vfe3.gauge_optim as gauge_optim
    from vfe3.geometry import phi_preconditioner as pp

    def refuse(*args, **kwargs):
        raise AssertionError("the eigenbasis fallback must not call the series direction or its structure constants")

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", refuse)
    monkeypatch.setattr(pp, "_strict_structure_constants", refuse)


def _count_series_calls(monkeypatch) -> list[int]:
    r"""Count calls of the series direction (its structure constants may come from the preparation cache)."""
    import vfe3.gauge_optim as gauge_optim

    calls: list[int] = []
    original = gauge_optim.pullback_group_direction

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", counting)
    return calls


def test_eigenbasis_fallback_is_exact_off_the_strip_and_rejects_ill_conditioned_frames(monkeypatch) -> None:
    from vfe3.gauge_optim import pullback_group_log_direction
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(6, 2)
    X = _charts()
    C = torch.randn(X.shape, generator=torch.Generator().manual_seed(4), dtype=F64)
    M = _linear_loss_covector(X, C)
    U = torch.linalg.matrix_exp(X)
    _forbid_structure_constants(monkeypatch)
    result = pullback_group_log_direction(M.reshape(3, -1), X.reshape(3, -1), U, collect=True, group=group,
                                          fallback="eigenbasis")
    assert result.accepted.tolist() == [True, True, False]
    assert result.fast_rows.tolist() == [True, False, False]
    exact = torch.linalg.solve(U, C)                                      # (3, 2, 3, 3) U^{-1} C
    for row in (0, 1):
        assert torch.allclose(result.xi[row], exact[row], rtol=0.0, atol=1e-9 * float(exact[row].abs().max()))
        assert torch.allclose(result.covector[row], C[row], rtol=0.0, atol=1e-9 * float(C[row].abs().max()))
    assert float(result.xi[2].abs().max()) == 0.0
    # The chart velocity of the off-strip row solves D_X(v) = U xi: v = Psi_R(ad_X)^{-1}(U xi U^{-1}).
    v = result.v_phi[1].reshape(2, 3, 3)
    t = 1e-6
    finite_difference = (torch.linalg.matrix_exp(X[1] + t * v) - torch.linalg.matrix_exp(X[1] - t * v)) / (2 * t)
    assert torch.allclose(finite_difference, U[1] @ exact[1], rtol=0.0, atol=1e-6)


def test_series_fallback_is_unchanged_and_calls_the_series_direction(monkeypatch) -> None:
    from vfe3.gauge_optim import pullback_group_log_direction
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(6, 2)
    X = _charts()
    M = _linear_loss_covector(X, torch.randn(X.shape, generator=torch.Generator().manual_seed(4), dtype=F64))
    U = torch.linalg.matrix_exp(X)
    builds = _count_series_calls(monkeypatch)
    default = pullback_group_log_direction(M.reshape(3, -1), X.reshape(3, -1), U, collect=False, group=group)
    named = pullback_group_log_direction(M.reshape(3, -1), X.reshape(3, -1), U, collect=False, group=group,
                                         fallback="series")
    assert builds
    assert torch.equal(default.xi, named.xi) and torch.equal(default.accepted, named.accepted)
    with pytest.raises(ValueError, match="fallback"):
        pullback_group_log_direction(M.reshape(3, -1), X.reshape(3, -1), U, collect=False, group=group,
                                     fallback="exact")


# Optimizer -----------------------------------------------------------------------------------------------

def _optimizer(**kwargs):
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(6, 2)
    X = _charts()
    phi = nn.Parameter(X.reshape(3, -1).clone())
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group, phi_group_trust_radius=0.1, phi_chart_max_norm=8.0, phi_group_retraction="group_log",
        weight_decay=0.0, **kwargs)
    phi.grad = _linear_loss_covector(
        X, torch.randn(X.shape, generator=torch.Generator().manual_seed(5), dtype=F64)).reshape(3, -1)
    return optimizer, phi


def test_optimizer_steps_off_strip_rows_without_structure_constants(monkeypatch) -> None:
    _forbid_structure_constants(monkeypatch)
    optimizer, phi = _optimizer(phi_group_log_fallback="eigenbasis")
    before = phi.detach().clone()
    optimizer.step()
    after = phi.detach()
    assert torch.isfinite(after).all()
    assert not torch.equal(after[0], before[0]) and not torch.equal(after[1], before[1])
    assert torch.equal(after[2], before[2])                               # kappa_2(U) > 1e3: rejected
    # The off-strip row moved along exp(phi) -> exp(phi) exp(-eta s xi): the frame, not the chart, is continuous.
    moved = torch.linalg.matrix_exp(after[1].reshape(2, 3, 3)) - torch.linalg.matrix_exp(before[1].reshape(2, 3, 3))
    assert 0.0 < float(moved.norm()) <= 0.2 * float(torch.linalg.matrix_exp(before[1].reshape(2, 3, 3)).norm())


def test_optimizer_refuses_an_unknown_fallback() -> None:
    with pytest.raises(ValueError, match="phi_group_log_fallback"):
        _optimizer(phi_group_log_fallback="exact")


def test_optimizer_default_fallback_is_the_series(monkeypatch) -> None:
    builds = _count_series_calls(monkeypatch)
    optimizer, phi = _optimizer()
    optimizer.step()
    assert optimizer._phi_group_log_fallback == "series"
    assert builds and torch.isfinite(phi.detach()).all()
    assert math.isfinite(float(phi.detach().norm()))
