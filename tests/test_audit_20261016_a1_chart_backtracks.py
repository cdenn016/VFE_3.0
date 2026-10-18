r"""Chart-route halving count of the manifold phi step (audit 2026-10-16, F14).

On ``m_phi_group_retraction='chart'`` the fixed :math:`10^{-6}` product-residual certificate is quadratic in
the step and, away from the identity frame, halves a trust-saturated factor scale before
``m_phi_group_trust_radius`` binds (docs/derivations/2026-09-28-phi-natgrad.tex, Table tab:cap). The halving
count was collected only under ``group_log`` (``phi_group_log_backtracks_max``), so no column separated
residual halving from trust clipping on the chart route. ``phi_group_backtracks_mean`` /
``phi_group_backtracks_max`` now report it on both retractions; the group_log key keeps its schema.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from vfe3.gauge_optim import (
    GROUP_LOG_DIAGNOSTIC_KEYS,
    GaugeManifoldAdamW,
    pullback_group_log_direction,
    stage_group_log_candidate,
    stage_pullback_group_candidate,
)
from vfe3.geometry.groups import get_group

F64 = torch.float64
_OPTIONS = dict(learning_rate=10.0, trust_radius=0.1, chart_max_norm=20.0)


def _rows() -> tuple:
    generator = torch.Generator().manual_seed(8)
    phi = torch.randn(4, 18, generator=generator, dtype=F64) / 3.0           # ||X_h||_F ~ 1
    grad = torch.randn(4, 18, generator=generator, dtype=F64)
    return phi, grad


def _step(retraction: str):
    group = get_group("block_glk")(K=6, n_heads=2, dtype=F64)
    phi0, grad = _rows()
    phi = nn.Parameter(phi0.clone())
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": _OPTIONS["learning_rate"], "pullback_group": True, "weight_decay": 0.0}],
        group,
        phi_group_trust_radius=_OPTIONS["trust_radius"],
        phi_chart_max_norm=_OPTIONS["chart_max_norm"],
        phi_group_retraction=retraction,
        weight_decay=0.0,
    )
    phi.grad = grad.clone()
    optimizer._collect_gauge_diag = True
    optimizer.step()
    return group, phi0, grad, optimizer._gauge_diag


def test_chart_route_reports_its_halving_count() -> None:
    group, phi0, grad, diagnostics = _step("chart")
    expected = stage_pullback_group_candidate(grad, phi0, group, **_OPTIONS).backtracking_reductions
    assert int(expected.min()) > 0                                             # the certificate binds here
    assert diagnostics["phi_group_backtracks_max"] == float(expected.max())
    assert math.isclose(diagnostics["phi_group_backtracks_mean"], float(expected.double().mean()))
    assert not set(GROUP_LOG_DIAGNOSTIC_KEYS) & set(diagnostics)              # chart columns only


def test_group_log_route_reports_the_same_columns() -> None:
    group, phi0, grad, diagnostics = _step("group_log")
    expected = stage_group_log_candidate(
        grad, phi0, group, direction=lambda *a, **k: pullback_group_log_direction(*a, group=group, **k),
        **_OPTIONS).backtracking_reductions
    assert diagnostics["phi_group_backtracks_max"] == float(expected.max()) == 0.0
    assert diagnostics["phi_group_backtracks_mean"] == 0.0
    assert diagnostics["phi_group_log_backtracks_max"] == diagnostics["phi_group_backtracks_max"]
    assert set(GROUP_LOG_DIAGNOSTIC_KEYS) <= set(diagnostics)
