r"""Audit 2026-10-16 lane B: scaling_analysis statistics (F55, F57).

F55: the frontier-collapse F-test counts each training run once when the nested standard routes share
cells (``width`` K{k} is ``width_depth`` K{k}_L6 and ``depth`` L{d} is ``width_depth`` K160_L{d}).
F57: the scaling summary records whether the seed replicates shared one training data order, and states
the lower-bound caveat beside the exponent CI and the lack-of-fit test. Every fixture is a tiny synthetic
point list or JSON run directory; no model is built.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import scaling_analysis as sa
from tests.test_audit_20261008_stats import _write_scaling_run


# =============================================================================
# F55: identical runs under two routes enter the frontier test once
# =============================================================================

_WIDTHS = (20, 40, 80, 120, 160)
_DEPTHS = (1, 2, 4, 6)
_BASE_K, _BASE_L = 160, 6


def _n_params(k: int, depth: int) -> float:
    return float(1000 * k + 40 * k * k * depth)


def _point(route: str, label: str, k: int, depth: int, rng: np.random.Generator) -> Dict[str, Any]:
    n = _n_params(k, depth)
    ce = float(np.exp(np.log(3.0 + 60.0 * n ** -0.3) + rng.normal(0.0, 0.01)))
    return {"route": route, "label": label, "n_params": n, "ce_mean": ce, "ce_sem": 0.0,
            "ce_seeds": [ce], "effective_run_keys": (f"run_K{k}_L{depth}",)}


def _nested_standard_routes() -> List[Dict[str, Any]]:
    rng = np.random.default_rng(7)
    points = [_point("width", f"K{k}", k, _BASE_L, rng) for k in _WIDTHS]
    points += [_point("depth", f"L{d}", _BASE_K, d, rng) for d in _DEPTHS]
    points += [_point("width_depth", f"K{k}_L{d}", k, d, rng) for k in _WIDTHS for d in _DEPTHS]
    return sorted(points, key=lambda p: (p["route"], p["label"]))


def test_frontier_test_counts_each_training_run_once():
    points = _nested_standard_routes()
    assert len(points) == 29 and len({p["effective_run_keys"] for p in points}) == 20
    frontier = sa.ancova_frontier_collapse(points)
    assert frontier["testable"]
    # 20 distinct runs: depth L1..L6 (4), width K20..K120 (4; K160 is depth L6), width_depth 12.
    assert frontier["routes"] == {"depth": 4, "width": 4, "width_depth": 12}
    assert (frontier["df1"], frontier["df2"]) == (4, 14)
    assert frontier["n_identical_points_dropped"] == 9


def test_frontier_test_is_unchanged_for_disjoint_routes():
    rng = np.random.default_rng(3)
    points = [_point("width", f"K{k}", k, _BASE_L, rng) for k in _WIDTHS]
    points += [_point("depth", f"L{d}", 30, d, rng) for d in _DEPTHS]
    frontier = sa.ancova_frontier_collapse(points)
    assert frontier["routes"] == {"width": 5, "depth": 4}
    assert (frontier["df1"], frontier["df2"]) == (2, 5)
    assert frontier["n_identical_points_dropped"] == 0
