r"""Validation-metric adapters for the registered scaling figures (audit finding PB-07).

``scaling_analysis.aggregate_points`` (and the ``test_ce`` / ``test_ppl`` / ``test_bpc`` /
``bpc_mean`` fields it carries downstream) is the character-corrected TEST-set path: the L(N)
power-law fit, the frontier-collapse F-test, and every figure built from it read the held-out test
split, and none of that is touched here or ever relabeled as a validation quantity.
``plot_capacity_scaling`` and ``plot_pareto_frontier`` (``vfe3/viz/figures.py``) are about the
VALIDATION split instead, and previously rendered only from hand-built dicts in tests -- no
production driver fed them. This module is that seam: ``aggregate_validation_points`` collapses
harvested rows into one point per ``(route, label)`` keyed on the persisted ``best_val_ppl`` (never
on a test metric), and ``capacity_scaling_kwargs`` / ``pareto_frontier_kwargs`` turn those points
into the two plotters' exact kwargs.

The quantity plotted is ``log2(best_val_ppl)`` -- validation BITS PER TOKEN, not BPC: the persisted
scaling headline carries no tokens-per-character factor for the validation split, so labeling it
"bits per character" would be wrong.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.data.datasets import _sha256_file
from vfe3.metrics import bootstrap_token_ce_band

logger = logging.getLogger(__name__)

_MIN_AXIS_POINTS   = 2                                        # a capacity-scaling axis needs >= 2 sizes
_MIN_PARETO_POINTS = 2                                        # a Pareto frontier needs >= 2 points
_INFERENCE_ROUTE   = "inference"                              # flat-N runs are not a parameter frontier


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_validation_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""Collapse seeds into one point per ``(route, label)``, keyed on persisted ``best_val_ppl``.

    A group is retained iff at least one exact seed carries BOTH a finite, positive
    ``best_val_ppl`` and a finite ``wall_time_s``. Test metrics (``test_ce`` / ``test_ppl`` /
    ``test_bpc``) are never consulted, so a validation-only row (every test metric null) still
    survives. ``val_bits_per_token_mean`` and ``wall_time_mean`` are means over the same paired seed
    intersection; paired and missing cohorts remain explicit in the returned point.

    A row with an EXPLICIT ``best_val_ppl: None`` (the key is present, just null -- not merely
    absent) alongside a sibling seed carrying a finite value in the SAME group is a data-integrity
    fault, not an absent metric: this raises ``ValueError`` rather than silently averaging over the
    surviving seeds or falling back to a test metric.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["route"], r["label"]), []).append(r)

    points: List[Dict[str, Any]] = []
    for (route, label), grp in sorted(groups.items()):
        by_seed: Dict[int, Dict[str, Any]] = {}
        for row in grp:
            seed = row.get("seed")
            if type(seed) is not int or seed < 0:
                raise ValueError(
                    f"{route}/{label}: expected an exact nonnegative integer seed, got {seed!r}"
                )
            if seed in by_seed:
                raise ValueError(f"{route}/{label}: duplicate seed {seed}")
            by_seed[seed] = row

        observed_seeds = sorted(by_seed)
        has_explicit_null = any("best_val_ppl" in g and g["best_val_ppl"] is None for g in grp)
        val_by_seed = {
            seed: value
            for seed, row in by_seed.items()
            if np.isfinite(value := _as_float(row.get("best_val_ppl"))) and value > 0.0
        }
        if has_explicit_null and val_by_seed:
            raise ValueError(
                f"{route}/{label}: explicit-null best_val_ppl alongside {len(val_by_seed)} finite "
                "seed(s) -- refusing to silently drop it or substitute a test metric"
            )
        wall_by_seed = {
            seed: value
            for seed, row in by_seed.items()
            if np.isfinite(value := _as_float(row.get("wall_time_s")))
        }
        paired_seeds = sorted(set(val_by_seed).intersection(wall_by_seed))
        if not paired_seeds:
            continue
        bits_seeds = [float(np.log2(val_by_seed[seed])) for seed in paired_seeds]
        wall_seeds = [wall_by_seed[seed] for seed in paired_seeds]
        missing_val_seeds = sorted(set(observed_seeds).difference(val_by_seed))
        missing_wall_time_seeds = sorted(set(observed_seeds).difference(wall_by_seed))
        npars      = [w for w in (_as_float(g.get("n_params")) for g in grp) if np.isfinite(w)]
        points.append({
            "route": route, "label": label, "scale_knob": grp[0]["scale_knob"],
            "n_params":   float(np.mean(npars)) if npars else float("nan"),
            "embed_dim":  _as_float(grp[0].get("embed_dim")),
            "n_heads":    _as_float(grp[0].get("n_heads")),
            "n_layers":   _as_float(grp[0].get("n_layers")),
            "n_e_steps":  _as_float(grp[0].get("n_e_steps")),
            "observed_seeds": observed_seeds,
            "paired_seeds": paired_seeds,
            "n_paired_seeds": len(paired_seeds),
            "missing_val_seeds": missing_val_seeds,
            "n_missing_val_seeds": len(missing_val_seeds),
            "missing_wall_time_seeds": missing_wall_time_seeds,
            "n_missing_wall_time_seeds": len(missing_wall_time_seeds),
            "val_bits_per_token_seeds": bits_seeds,
            "val_bits_per_token_mean":  float(np.mean(bits_seeds)),
            "n_val_seeds": len(bits_seeds),
            "wall_time_seeds": wall_seeds,
            "wall_time_mean": float(np.mean(wall_seeds)),
        })
    return points


def capacity_scaling_kwargs(
    points:      List[Dict[str, Any]],
    axis_routes: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    r"""Build ``plot_capacity_scaling``'s ``scaling`` kwarg from validation points, one axis at a
    time. A point enters ``axis`` only when BOTH ``point["route"] == axis_routes[axis]`` AND
    ``point["scale_knob"] == axis`` -- route equality alone is not enough because the ``inference``
    route carries both ``n_e_steps`` and ``n_layers`` cells, and only the latter belongs on the
    ``n_layers`` panel.

    An axis whose route never appears in ``points`` is simply omitted (that sweep was not run). An
    axis whose route DOES appear but yields fewer than ``_MIN_AXIS_POINTS`` finite points is a gap
    in a sweep this figure otherwise relies on: the whole figure is withheld (returns ``None``)
    rather than silently rendering a partial capacity-scaling figure, and the gap is logged.
    """
    reasons: List[str] = []
    scaling: Dict[str, Dict[str, np.ndarray]] = {}
    for axis, route in axis_routes.items():
        if not any(p["route"] == route for p in points):
            continue                                          # route never ran: omit, not an error
        axis_points = [
            p for p in points
            if p["route"] == route and p["scale_knob"] == axis
            and np.isfinite(p.get("val_bits_per_token_mean", float("nan")))
        ]
        if len(axis_points) < _MIN_AXIS_POINTS:
            reasons.append(f"{axis} (route={route!r}): only {len(axis_points)} finite point(s), "
                            f"need >= {_MIN_AXIS_POINTS}")
            continue
        ordered = sorted(axis_points, key=lambda p: p[axis])
        scaling[axis] = {
            "x":              np.array([p[axis] for p in ordered], dtype=float),
            "bits_per_token": np.array([p["val_bits_per_token_mean"] for p in ordered], dtype=float),
            "wall_time":      np.array([p["wall_time_mean"] for p in ordered], dtype=float),
        }
    if reasons:
        logger.warning("capacity_scaling withheld: %s", "; ".join(reasons))
        return None
    if not scaling:
        return None
    return {"scaling": scaling}


def pareto_frontier_kwargs(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    r"""Build ``plot_pareto_frontier``'s ``points`` kwarg from aggregated PARAMETER points -- the
    flat-N ``inference`` route is excluded, mirroring ``scaling_analysis``'s own
    ``param_points`` / ``INFERENCE_ROUTE`` split: a Pareto frontier over parameter count is not
    meaningful for cells whose parameter count never varies. Needs >= ``_MIN_PARETO_POINTS`` points
    carrying both a finite ``val_bits_per_token_mean`` and a finite ``n_params``; otherwise ``None``.
    """
    eligible = [
        p for p in points
        if p["route"] != _INFERENCE_ROUTE
        and np.isfinite(p.get("val_bits_per_token_mean", float("nan")))
        and np.isfinite(p.get("n_params", float("nan")))
    ]
    if len(eligible) < _MIN_PARETO_POINTS:
        return None
    ordered = sorted(eligible, key=lambda p: p["n_params"])
    plot_points = {
        "bits_per_token": np.array([p["val_bits_per_token_mean"] for p in ordered], dtype=float),
        "n_params":       np.array([p["n_params"] for p in ordered], dtype=float),
        "label":          [p["label"] for p in ordered],
    }
    wall_time = np.array([p.get("wall_time_mean", float("nan")) for p in ordered], dtype=float)
    if np.all(np.isfinite(wall_time)):
        plot_points["wall_time"] = wall_time
    return {"points": plot_points}


# =============================================================================
# PB-07: ablation-report adapters (component forest, joint-LR grid).
#
# The forest reads per-cell paired-token nats persisted as ``val_token_nats.pt`` and re-verifies the
# marker's byte/tensor identity (sha256 + size + numel + dtype) before trusting a file, so a
# same-shape finite overwrite (a re-run that changed the bytes without updating the marker) is
# rejected rather than silently plotted. The grid reads the accumulated sweep rows and requires the
# exact Cartesian product declared by persisted sweep metadata.
# =============================================================================


def _successful_markers_by_label(sweep_dir: Path) -> Dict[str, Dict[str, Any]]:
    r"""Every successful cell's ``ablation_result.json`` under ``sweep_dir``, keyed by its label.

    Mirrors the sweep runner's success criteria (``status == "success"``, no ``error_kind``, finite
    terminal PPL); each returned marker is augmented with its resolved cell directory under the
    private ``_cell_dir`` key so ``_load_token_vector`` can find the sibling token file. Unreadable,
    non-object, failed, errored, and nonfinite-terminal markers are skipped.
    """
    markers: Dict[str, Dict[str, Any]] = {}
    for marker_path in sorted(sweep_dir.glob("*/ablation_result.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:                                        # unreadable marker -> skip
            continue
        if not isinstance(marker, Mapping):
            continue
        if marker.get("status") != "success" or marker.get("error_kind") is not None:
            continue
        try:
            terminal_ppl = float(marker["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(terminal_ppl):
            continue
        label = marker.get("label")
        if not isinstance(label, str):
            continue
        augmented = dict(marker)
        augmented["_cell_dir"] = str(marker_path.parent)
        markers[label] = augmented
    return markers


def _load_token_vector(sweep_dir: Path, marker: Mapping[str, Any]) -> Optional[torch.Tensor]:
    r"""Safe-load the marker's paired-token nats vector, or ``None`` if its identity does not verify.

    Resolves ONLY the marker's exact child filename (``val_token_nats.pt``) under the cell directory
    -- any other value is refused, so no path outside the cell can be read -- then re-verifies the
    same byte/tensor identity the resume validator binds: a true ``paired_token_bootstrap`` flag, an
    existing file whose recomputed streamed SHA-256 and byte size equal the marker, and a
    ``weights_only`` safe load that yields a finite, nonempty, one-dimensional tensor whose numel and
    string dtype equal the marker. Any mismatch fails closed (returns ``None``).
    """
    if not marker.get("paired_token_bootstrap"):
        return None
    if marker.get("val_token_nats_path") != "val_token_nats.pt":
        return None
    cell_dir = marker.get("_cell_dir")
    if not cell_dir:
        return None
    expected_sha    = marker.get("val_token_nats_sha256")
    expected_size   = marker.get("val_token_nats_size_bytes")
    expected_numel  = marker.get("val_token_nats_numel")
    expected_dtype  = marker.get("val_token_nats_dtype")
    if not (isinstance(expected_sha, str) and expected_sha):
        return None
    path = Path(cell_dir) / "val_token_nats.pt"
    if not path.is_file():
        return None
    if path.stat().st_size != expected_size:
        return None
    if _sha256_file(path) != expected_sha:
        return None
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim != 1 or tensor.numel() == 0:
        return None
    if int(tensor.numel()) != expected_numel or str(tensor.dtype) != expected_dtype:
        return None
    if not bool(torch.isfinite(tensor).all()):
        return None
    return tensor


def ablation_forest_kwargs(
    sweep_dir:      Path,
    baseline_label: str,

    *,
    admitted_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    r"""Build ``plot_ablation_forest``'s ``rows`` kwarg from persisted, aligned paired-token vectors.

    The named baseline arm must exist and load; every arm must be aligned (same token length) with
    it; each arm's delta / lo / hi is the paired bootstrap-over-tokens band (``seed=0``) converted
    from nats to BITS per token (divided by ``ln 2``). A missing baseline, a shape mismatch, or a
    token file whose identity no longer verifies withholds the whole figure (returns ``None``).
    """
    if admitted_rows is None:
        markers = _successful_markers_by_label(sweep_dir)
    else:
        markers = {}
        for row in admitted_rows:
            label = row.get("label")
            cell_dir = row.get("_cell_dir")
            if not isinstance(label, str) or not isinstance(cell_dir, str):
                logger.warning(
                    "ablation_forest withheld: an admitted row lacks its label or cell directory"
                )
                return None
            markers[label] = dict(row)
    baseline_marker = markers.get(baseline_label)
    if baseline_marker is None:
        logger.warning("ablation_forest withheld: no successful %r baseline cell under %s",
                       baseline_label, sweep_dir)
        return None
    baseline = _load_token_vector(sweep_dir, baseline_marker)
    if baseline is None:
        logger.warning("ablation_forest withheld: baseline token vector failed identity verification")
        return None
    rows: List[Dict[str, Any]] = []
    for label in sorted(markers):
        arm = _load_token_vector(sweep_dir, markers[label])
        if arm is None or arm.shape != baseline.shape:
            logger.warning("ablation_forest withheld: arm %r has a missing or misaligned token vector",
                           label)
            return None
        band = bootstrap_token_ce_band(arm, baseline, seed=0)
        rows.append({"label": label, **{key: band[key] / math.log(2.0)
                                        for key in ("delta", "lo", "hi")}})
    return {"rows": rows}


def lr_grid_heatmap_kwargs(
    rows:     List[Dict[str, Any]],
    x_key:    str,
    y_key:    str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    baseline: Tuple[float, float],
) -> Optional[Dict[str, Any]]:
    r"""Build ``plot_lr_grid_heatmap``'s ``grid`` kwarg from one completed two-dimensional sweep.

    Requires the EXACT Cartesian product declared by the persisted grid metadata: every ``(x, y)``
    pair from ``x_values`` x ``y_values`` present exactly once, each carrying a finite
    ``primary_val_ppl``. A duplicate cell, a missing cell, an off-grid pair, a nonfinite value, or a
    row whose ``overrides`` lacks either learning rate withholds the whole figure (returns ``None``).
    """
    xs = tuple(float(value) for value in x_values)
    ys = tuple(float(value) for value in y_values)
    expected = {(x, y) for y in ys for x in xs}
    cells: Dict[Tuple[float, float], float] = {}
    for row in rows:
        overrides = row.get("overrides")
        if not isinstance(overrides, Mapping):
            logger.warning("lr_grid_heatmap withheld: a row carries no overrides mapping")
            return None
        try:
            pair = (float(overrides[x_key]), float(overrides[y_key]))
            value = float(row["primary_val_ppl"])
        except (KeyError, TypeError, ValueError):
            logger.warning("lr_grid_heatmap withheld: a row lacks %r/%r or a finite primary_val_ppl",
                           x_key, y_key)
            return None
        if pair not in expected or pair in cells or not math.isfinite(value):
            logger.warning("lr_grid_heatmap withheld: off-grid, duplicate, or nonfinite cell %s", pair)
            return None
        cells[pair] = value
    if set(cells) != expected:
        logger.warning("lr_grid_heatmap withheld: incomplete grid (%d of %d cells present)",
                       len(cells), len(expected))
        return None
    z = np.asarray([[cells[(x, y)] for x in xs] for y in ys], dtype=float)
    return {"grid": {"x": np.asarray(xs), "y": np.asarray(ys), "z": z,
                     "xlabel": x_key, "ylabel": y_key, "baseline": baseline}}
