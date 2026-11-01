r"""Audit 2026-10-31 remediation, package viz-figures: the offset-law exponent interval at a box bound.

analysis-statistics-2 (A041): when the sizes carry no curvature the offset law :math:`E + A N^{-\alpha}` is not
identified. ``curve_fit`` pins :math:`E` on its lower bound in the point fit or in many bootstrap refits, and the
percentile interval of :math:`\alpha` then undercovers. ``_fit_power_law`` records its active bounds, the analyzer
withholds the offset interval with status ``unassessed`` and reports the two-parameter law beside it, and an
identified offset law carries a bootstrap interval for :math:`E`. Every fixture is a synthetic point list or a tiny
JSON run directory; no model is built.
"""

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pytest

import scaling_analysis as sa
from tests.test_audit_20261008_stats import _write_scaling_run
from vfe3.viz.figures import _fit_power_law

pytest.importorskip("scipy")

_V = 50257
_WIDTHS = (20, 40, 80, 120, 160)                                  # the shipped scaling widths
_N = [int(_V * (k * k // 5 + 3 * k)) for k in _WIDTHS]
_LOG_JITTER = ((-0.002, 0.0005, 0.0015), (0.001, -0.0025, 0.0015), (0.0, 0.002, -0.002),
               (0.0025, -0.001, -0.0015), (-0.0015, 0.0, 0.0015))


def _ce_no_curvature(n: float) -> float:
    r"""A law with a slightly negative offset: the constrained fit puts :math:`E` on its lower bound 0."""
    return 5.6 * (n / _N[0]) ** -0.08 - 0.1


def _ce_interior(n: float) -> float:
    r"""An identified offset law, :math:`E = 3`, :math:`\alpha = 0.3`."""
    return 3.0 + 2.5 * (n / _N[0]) ** -0.3


def _points(law: Callable[[float], float]) -> List[Dict[str, Any]]:
    out = []
    for n, jitter in zip(_N, _LOG_JITTER):
        seeds = [float(law(n) * math.exp(j)) for j in jitter]
        out.append({"n_params": float(n), "ce_seeds": seeds, "ce_mean": float(np.mean(seeds))})
    return out


def _fit(points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    xs = np.array([p["n_params"] for p in points], dtype=float)
    ys = np.array([p["ce_mean"] for p in points], dtype=float)
    return _fit_power_law(xs, ys, with_offset=True)


def test_offset_fit_records_its_active_box_bounds() -> None:
    pinned = _fit(_points(_ce_no_curvature))
    assert pinned["form"] == "offset_power_law"
    assert "E_lower" in pinned["at_bound"]
    interior = _fit(_points(_ce_interior))
    assert interior["form"] == "offset_power_law" and interior["at_bound"] == []
    plain = _fit_power_law(np.array(_N, dtype=float), np.array([_ce_interior(n) for n in _N]))
    assert plain["form"] == "power_law" and plain["at_bound"] == []


def test_identification_withholds_a_bound_pinned_offset_law() -> None:
    points = _points(_ce_no_curvature)
    out = sa.offset_law_identification(points, _fit(points), n_boot=40)
    assert out["identified"] is False
    assert "bound" in out["reason"] and "E_lower" in out["at_bound"]
    assert out["E_ci"] is None


def test_identified_offset_law_carries_an_E_interval_from_the_ci_replicates() -> None:
    points = _points(_ce_interior)
    fit = _fit(points)
    out = sa.offset_law_identification(points, fit, n_boot=60)
    assert out["identified"] is True and out["bound_fraction"] == 0.0
    lo, hi = out["E_ci"]
    assert lo < fit["E"] < hi
    # The replicates are those of bootstrap_exponent_ci: their exponent percentiles are its interval.
    _alpha, ci_lo, ci_hi = sa.bootstrap_exponent_ci(points, n_boot=60, with_offset=True)
    assert out["alpha_ci"] == [ci_lo, ci_hi]


def test_identification_withholds_when_bootstrap_refits_reach_a_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    points = _points(_ce_interior)
    fit = _fit(points)
    calls = {"n": 0}

    def pinned_every_other(N: object, L: object, **kwargs: Any) -> Dict[str, Any]:
        out = dict(_fit_power_law(N, L, **kwargs))
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            out["at_bound"] = ["E_lower"]
        return out

    monkeypatch.setattr("vfe3.viz.figures._fit_power_law", pinned_every_other)
    out = sa.offset_law_identification(points, fit, n_boot=20)
    assert out["identified"] is False and out["bound_fraction"] == pytest.approx(0.5)
    assert out["E_ci"] is None


def _sweep(root: Path, law: Callable[[float], float]) -> None:
    for (k, n), jitter in zip(zip(_WIDTHS, _N), _LOG_JITTER):
        for seed, j in zip((1, 2, 3), jitter):
            _write_scaling_run(root, route="width", label=f"K{k}", seed=seed, n_params=n,
                               test_ce=float(law(n) * math.exp(j)), embed_dim=4 + 4 * _WIDTHS.index(k))


def _analyze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, law: Callable[[float], float]) -> Dict[str, Any]:
    _sweep(tmp_path, law)
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(sa.CONFIG, "with_offset", True)
    monkeypatch.setitem(sa.CONFIG, "n_bootstrap", 40)
    monkeypatch.setattr(sa, "_make_figures", lambda *a, **k: None)
    sa.analyze()
    return json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))


def test_analyze_withholds_the_offset_interval_and_reports_the_two_parameter_law(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _analyze(tmp_path, monkeypatch, _ce_no_curvature)
    pf = summary["pooled_fit"]
    assert pf["form"] == "offset_power_law" and "E_lower" in pf["at_bound"]
    assert pf["alpha_ci"] is None and pf["alpha_ci_status"] == "unassessed"
    assert pf["E_ci"] is None and pf["offset_identification"]["identified"] is False
    two = pf["two_parameter"]
    assert two["form"] == "power_law" and math.isfinite(two["alpha"])
    assert all(math.isfinite(v) for v in two["alpha_ci"]) and two["alpha_ci"][0] < two["alpha_ci"][1]
    report = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "offset law not identified" in report and "two-parameter law" in report


def test_analyze_keeps_the_interval_of_an_identified_offset_law(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _analyze(tmp_path, monkeypatch, _ce_interior)
    pf = summary["pooled_fit"]
    assert pf["form"] == "offset_power_law" and pf["at_bound"] == []
    assert pf["alpha_ci_status"] == "assessed" and all(math.isfinite(v) for v in pf["alpha_ci"])
    lo, hi = pf["E_ci"]
    assert lo < pf["E"] < hi
    assert pf["two_parameter"] is None
    report = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "E = " in report and "95% CI" in report
