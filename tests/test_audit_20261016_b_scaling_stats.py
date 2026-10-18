r"""Audit 2026-10-16 lane B: scaling_analysis statistics (F55, F57).

F55: the frontier-collapse F-test counts each training run once when the nested standard routes share
cells (``width`` K{k} is ``width_depth`` K{k}_L6 and ``depth`` L{d} is ``width_depth`` K160_L{d}).
F57: the scaling summary records whether the seed replicates shared one training data order, and states
the lower-bound caveat beside the exponent CI and the lack-of-fit test. Every fixture is a tiny synthetic
point list or JSON run directory; no model is built.
"""

import json
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


# =============================================================================
# F57: the summary states whether seed replicates shared one data order
# =============================================================================

_UNRECORDED = object()


def _rows(data_seeds: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, seeds in data_seeds.items():
        for i, ds in enumerate(seeds):
            row: Dict[str, Any] = {"route": "width", "label": label, "seed": i}
            if ds is not _UNRECORDED:
                row["data_seed"] = ds
            rows.append(row)
    return rows


def test_data_order_status_classifies_the_replicate_cells():
    shared = sa._data_order_status(_rows({"K8": [3, 3, 3], "K12": [3, 3, 3]}))
    assert shared["status"] == "shared_explicit" and shared["data_seeds"] == [3]
    assert "lower bound" in shared["caveat"]
    distinct = sa._data_order_status(_rows({"K8": [6, 64], "K12": [6, 64]}))
    assert distinct["status"] == "distinct" and "lower bound" not in distinct["caveat"]
    assert sa._data_order_status(_rows({"K8": [3, 3], "K12": [6, 64]}))["status"] == "partially_shared"
    assert sa._data_order_status(_rows({"K8": [6, 6, 64]}))["status"] == "partially_shared"
    # data_seed=None: the order follows the global RNG seeded by each model seed.
    assert sa._data_order_status(_rows({"K8": [None, None]}))["status"] == "nonshared_unspecified"
    assert sa._data_order_status(_rows({"K8": [3, None]}))["status"] == "distinct"
    unknown = sa._data_order_status(_rows({"K8": [3, _UNRECORDED]}))
    assert unknown["status"] == "unknown" and "lower bound" in unknown["caveat"]
    # A single-seed cell has no seed spread and does not enter the classification.
    assert sa._data_order_status(_rows({"K8": [3], "K12": [3]}))["status"] == "no_replicates"
    assert sa._data_order_status(_rows({"K8": [3, 3], "K12": [6]}))["status"] == "shared_explicit"


def test_analyze_records_the_shared_order_beside_the_ci_and_lack_of_fit(tmp_path, monkeypatch):
    law = {8: 4.00, 12: 3.80, 16: 3.70, 20: 3.64}
    for seed, jitter in ((1, 0.01), (2, -0.012), (3, 0.002)):
        for k, ce in law.items():
            _write_scaling_run(tmp_path, route="width", label=f"K{k}", seed=seed, n_params=10 * k * k,
                               test_ce=ce + jitter + 0.001 * k * (seed - 2), embed_dim=k)
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(sa.CONFIG, "with_offset", False)
    monkeypatch.setitem(sa.CONFIG, "n_bootstrap", 20)
    monkeypatch.setattr(sa, "_make_figures", lambda *a, **k: None)
    sa.analyze()
    rows = list((tmp_path / "scaling_points.csv").read_text(encoding="utf-8").splitlines())
    assert "data_seed" in rows[0].split(",")
    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["data_order"]["status"] == "shared_explicit"
    assert summary["data_order"]["data_seeds"] == [3]
    assert summary["pooled_fit_lack_of_fit"]["testable"] is True
    report = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "shared_explicit" in report and "lower bound" in report


def test_module_docstring_states_the_non_embedding_share():
    doc = " ".join(sa.__doc__.split())
    assert "~1% of ``n_params``" in doc
