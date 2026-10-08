r"""Click-to-run analysis + figures for the VFE_3.0 PARAMETER-scaling runs written by ``scaling.py``.

Harvests every run directory under ``input_dir`` (each carries ``summary.json`` with the enriched
``scaling_point`` block, plus ``config.json`` / ``provenance.json`` / ``scaling_cell.json``), writes a
tidy ``scaling_points.csv`` (one row per run), aggregates seeds per (route, size) point, fits the loss
power law ``test_ce = A * N^{-alpha}`` (with bootstrap confidence on the exponent), runs the multi-route
frontier-collapse F-test (do different ways of growing N share one ``L(N)`` curve?), prints the tables,
and writes the scaling figures via ``vfe3.viz.figures``. There is no model re-run and no CLI parsing:
edit ``CONFIG`` and run ``python scaling_analysis.py``.

The y-axis is ``test_ce`` (nats/token), the canonical additive scaling quantity; perplexity is a
nonlinear re-read. The x-axis is the recorded ``n_params`` (never a derived ``d^2`` proxy); when the runs
record ``n_non_embedding_params`` (total minus per-token / positional lookup tables, ~99% of ``n_params``
at the scaling widths) the law is also fit against that axis. R^2, the exponent CI, and the lack-of-fit
test are reported only from four distinct sizes; fewer leave the fit ``not_testable``. Runs whose
``test_ce`` is null (no test split) or non-positive are dropped before the log fit; the analysis warns
if the harvested points span more than one ``data_sha256`` (mixed corpus) or ``git_sha`` (code drift),
either of which confounds a frontier.

Two further figures, ``capacity_scaling.png`` and ``pareto_frontier.png`` (audit finding PB-07), are
dispatched separately from persisted validation metrics: ``vfe3.viz.sweep_adapters.aggregate_validation_points``
collapses seeds keyed on the persisted ``best_val_ppl`` (never a test metric) into ``validation bits/token``
(``log2(best_val_ppl)``), which is NOT the character-corrected test BPC that ``aggregate_points`` /
``bpc_mean`` carry above.
"""

import csv
import json
import logging
import math
import os
import typing
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


from vfe3.process_utils import run_isolated_python


_CHILD_TIMEOUT_SECONDS = 3600      # bound the isolated worker; see run_process_tree
_CHILD_SENTINEL = "_VFE3_SCALING_ANALYSIS_CHILD"


def _run_isolated_child() -> int:
    """Run this driver without loading scientific libraries in the parent."""
    return run_isolated_python(__file__, _CHILD_SENTINEL, _CHILD_TIMEOUT_SECONDS, "scaling-analysis")


if __name__ == "__main__" and os.environ.get(_CHILD_SENTINEL) != "1":
    raise SystemExit(_run_isolated_child())

if os.environ.get(_CHILD_SENTINEL) == "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from vfe3.artifact_utils import canonical_json_sha256 as _canonical_json_sha256
from vfe3.config import config_from_serialized, serialize_config
from vfe3.metric_contracts import (  # noqa: E402
    SCALING_MIN_TESTABLE_SIZES,
    perplexity_matches_ce,
    target_blind_objective_interpretation,
)
from vfe3.path_utils import portable_path_component_key
from vfe3.viz.sweep_adapters import aggregate_validation_points, capacity_scaling_kwargs, pareto_frontier_kwargs
from vfe3.run_artifacts import _write_json_atomic


F_CE_RELATIONSHIP_FILENAME = "f_ce_relationship.png"

logger = logging.getLogger("scaling_analysis")

INFERENCE_ROUTE = "inference"                                # flat-N routes, plotted separately from L(N)

# PB-07: validation-metric figures (capacity_scaling.png, pareto_frontier.png) read validation_points
# (aggregate_validation_points, keyed on the persisted best_val_ppl) -- never the test-metric
# points/param_points/infer_points used above. The route each axis reads is defined once here; the
# FigureSpec pair itself is built lazily inside _make_figures (behind the same guarded viz import the
# figure pass already uses) so `import scaling_analysis` never requires the matplotlib stack.
# Each capacity axis is the scale knob a route grows; the route is derived from the harvested
# validation points (``_axis_routes``), preferring the standard routes when several share a knob.
AXIS_ROUTE_PREFERENCE: Dict[str, Tuple[str, ...]] = {
    "embed_dim": ("width", "grow_K"),
    "n_heads":   ("blocksize",),
    "n_layers":  ("depth",),
}
# Scale knobs that change the training budget rather than the parameter count. Their cells are
# fixed-N points and never enter the L(N) fit; they feed only the tokens-axis figure.
NON_PARAMETER_KNOBS = frozenset({"max_steps"})

CONFIG: Dict[str, Any] = {
    "input_dir":                       "vfe3_scaling_results",  # scaling.py output root and design
    "with_offset":                     True,                    # False -> A*N^-alpha; True -> E + A*N^-alpha
    "n_bootstrap":                     2000,                    # nested points-by-seeds bootstrap
    "min_points":                      2,                       # sizes required for a per-route fit (R^2/CI need 4)
    "force_accept_code_identity_drift": False,                  # analysis-only; all other checks stay strict
}

_CSV_COLUMNS = [
    "route", "scale_knob", "label", "seed", "n_params", "n_learnable_params",
    "n_non_embedding_params", "embed_dim", "n_heads",
    "n_gen", "gauge_group", "n_layers", "n_e_steps", "family",
    "full_cov_kl_precision",
    "use_block_mlp", "block_mlp_mode", "block_mlp_covariance", "block_mlp_expansion",
    "block_mlp_activation", "block_mlp_dropout", "block_mlp_covariance_floor",
    "tokens_seen", "est_flops_6ND",
    "est_flops_analytic", "active_params_per_token", "test_ce", "test_ppl",
    "test_bits_per_token", "test_bpc",
    "estep_final_f_per_token", "best_val_ppl",
    "wall_time_s", "data_sha256", "train_data_sha256", "val_data_sha256", "test_data_sha256",
    "git_sha", "code_identity_forced",
    "cell_git_dirty", "cell_git_dirty_fingerprint",
    "provenance_git_dirty", "provenance_git_dirty_fingerprint",
]

_ROUTE_NOTES = {
    "blocks_K48_tied_2x": "tied structural ablation; not a strict full-covariance pure control",
}

_SCALING_CELL_SCHEMA_VERSION = 2
_SCALING_REUSE_DIGEST_FIELD = "reuse_contract_sha256"
_SCALING_SUMMARY_DIGEST_FIELD = "scaling_reuse_contract_sha256"
_SCALING_SOURCE_SPLITS = ("train", "validation", "test")
_SCALING_STRUCTURAL_FIELDS = (
    "n_params", "scale_knob", "embed_dim", "n_heads", "n_layers", "n_e_steps",
    "n_gen", "gauge_group", "family", "tokens_seen",
    "full_cov_kl_precision",
    "use_block_mlp", "block_mlp_mode", "block_mlp_covariance",
    "block_mlp_expansion", "block_mlp_activation", "block_mlp_dropout",
    "block_mlp_covariance_floor",
    "git_sha", "git_dirty", "git_dirty_fingerprint",
    "train_source_identity", "validation_source_identity", "test_source_identity",
)


# =============================================================================
# HARVEST
# =============================================================================

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_exact_nonnegative_seed(value: Any) -> bool:
    """Whether ``value`` is a plain Python ``int`` seed in the supported domain."""
    return type(value) is int and value >= 0


def _is_sha256(value: object) -> bool:
    """Whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_scaling_metrics(summary: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """Return the complete internally consistent metric payload bound by a scaling summary."""
    scaling_point = summary.get("scaling_point")
    if not isinstance(scaling_point, Mapping):
        return None
    required = ("n_params", "test_ce", "test_ppl", "test_bits_per_token", "test_bpc")
    if any(key not in summary or key not in scaling_point for key in required):
        return None
    if any(summary[key] != scaling_point[key] for key in required):
        return None
    metrics = {key: scaling_point[key] for key in required}
    n_params = metrics["n_params"]
    test_ce = metrics["test_ce"]
    test_ppl = metrics["test_ppl"]
    test_bits = metrics["test_bits_per_token"]
    test_bpc = metrics["test_bpc"]

    def _finite_positive(value: object) -> bool:
        return (not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > 0.0)

    if type(n_params) is not int or n_params <= 0:
        return None
    if not all(_finite_positive(value) for value in (test_ce, test_bits)):
        return None
    if not (isinstance(test_ppl, (int, float)) and not isinstance(test_ppl, bool)
            and float(test_ppl) > 0.0 and not math.isnan(float(test_ppl))):
        return None
    if test_bpc is not None and not _finite_positive(test_bpc):
        return None
    if (not perplexity_matches_ce(test_ce, test_ppl)
            or not math.isclose(
                float(test_bits), float(test_ce) / math.log(2.0), rel_tol=1e-9, abs_tol=1e-12)):
        return None
    return metrics


def _validated_bound_scaling_run(
    run:        Path,
    summary:    object,
    config_json: object,
    provenance: object,
    cell:       object,

    *,
    force_accept_code_identity_drift: bool = False,
) -> Optional[Dict[str, object]]:
    """Validate the schema-2 generation contract that binds one scaling cell to its summary."""
    if (run / "scaling_failure.json").exists():
        return None
    if not all(isinstance(value, Mapping) for value in (
            summary, config_json, provenance, cell)):
        return None
    config = config_json.get("config")
    if not isinstance(config, Mapping):
        return None
    if cell.get("schema_version") != _SCALING_CELL_SCHEMA_VERSION:
        return None
    dataset = cell.get("dataset")
    if not isinstance(dataset, str) or not dataset or config_json.get("dataset") != dataset:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            migrated = config_from_serialized(
                config,
                source=str(run / "config.json"),
            )
        effective_config = asdict(migrated)
        # Digest the strict-JSON serialized form exactly as scaling.run_cell and
        # _cell_resume_status do; raw asdict carries kl_max=inf, which strict JSON refuses.
        expected_config_sha256 = _canonical_json_sha256(
            json.loads(json.dumps(serialize_config(migrated), default=str)))
        saved_digest = cell.get(_SCALING_REUSE_DIGEST_FIELD)
    except (TypeError, ValueError):
        return None
    if (not isinstance(saved_digest, str)
            or len(saved_digest) != 64
            or summary.get(_SCALING_SUMMARY_DIGEST_FIELD) != saved_digest
            or cell.get("config_sha256") != expected_config_sha256):
        return None
    scaling_point = summary.get("scaling_point")
    if not isinstance(scaling_point, Mapping):
        return None
    for field in ("embed_dim", "n_heads", "gauge_group", "n_layers", "n_e_steps"):
        if field in scaling_point:
            observed = scaling_point[field]
            expected = effective_config[field]
            if type(observed) is not type(expected) or observed != expected:
                return None

    code_identity = cell.get("code_identity")
    if not isinstance(code_identity, Mapping):
        return None
    git_sha = code_identity.get("git_sha")
    git_dirty = code_identity.get("git_dirty")
    dirty_fingerprint = code_identity.get("git_dirty_fingerprint")
    provenance_git_sha = provenance.get("git_sha")
    provenance_git_dirty = provenance.get("git_dirty")
    provenance_dirty_fingerprint = provenance.get("git_dirty_fingerprint")
    cell_identity_valid = (
        isinstance(git_sha, str)
        and bool(git_sha)
        and type(git_dirty) is bool
        and (
            (git_dirty and isinstance(dirty_fingerprint, str) and bool(dirty_fingerprint))
            or (not git_dirty and dirty_fingerprint is None)
        )
    )
    provenance_identity_valid = (
        isinstance(provenance_git_sha, str)
        and bool(provenance_git_sha)
        and type(provenance_git_dirty) is bool
        and (
            (provenance_git_dirty
             and isinstance(provenance_dirty_fingerprint, str)
             and bool(provenance_dirty_fingerprint))
            or (not provenance_git_dirty and provenance_dirty_fingerprint is None)
        )
    )
    if (not cell_identity_valid
            or not provenance_identity_valid
            or provenance_git_sha != git_sha):
        return None
    code_identity_forced = (
        provenance_git_dirty != git_dirty
        or provenance_dirty_fingerprint != dirty_fingerprint
    )
    if code_identity_forced and not force_accept_code_identity_drift:
        return None
    if (code_identity_forced
            and not all(_is_sha256(provenance.get(field)) for field in (
                "train_data_sha256", "val_data_sha256", "test_data_sha256"))):
        return None

    sources = cell.get("data_sources")
    if not isinstance(sources, Mapping) or set(sources) != set(_SCALING_SOURCE_SPLITS):
        return None
    source_identities: Dict[str, str] = {}
    for split in _SCALING_SOURCE_SPLITS:
        source = sources[split]
        if (not isinstance(source, Mapping)
                or not isinstance(source.get("format"), str)
                or not source["format"]
                or type(source.get("size_bytes")) is not int
                or source["size_bytes"] < 0
                or not isinstance(source.get("sha256"), str)
                or not source["sha256"]):
            return None
        source_identities[split] = json.dumps(
            dict(source),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    seeds = (cell.get("seed"), effective_config.get("seed"), provenance.get("seed"))
    if (not all(_is_exact_nonnegative_seed(seed) for seed in seeds)
            or len(set(seeds)) != 1):
        return None
    for field in ("route", "label", "scale_knob"):
        if type(cell.get(field)) is not str or not cell[field]:
            return None
    metrics = _validated_scaling_metrics(summary)
    if metrics is None:
        return None
    effective_run_key = _canonical_json_sha256({
        "config": _float_fields_as_float(
            json.loads(json.dumps(serialize_config(migrated), default=str)), migrated),
        "dataset": dataset,
        "data_seed": cell.get("data_seed"),
        "max_tokens": cell.get("max_tokens"),
        "sources": source_identities,
        "code_identity": [git_sha, git_dirty, dirty_fingerprint],
    })
    return {
        **metrics,
        "effective_config": effective_config,
        "effective_run_key": effective_run_key,
        "seed": seeds[0],
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_dirty_fingerprint": dirty_fingerprint,
        "code_identity_forced": code_identity_forced,
        "cell_git_dirty": git_dirty,
        "cell_git_dirty_fingerprint": dirty_fingerprint,
        "provenance_git_dirty": provenance_git_dirty,
        "provenance_git_dirty_fingerprint": provenance_dirty_fingerprint,
        **{
            f"{split}_source_identity": source_identities[split]
            for split in _SCALING_SOURCE_SPLITS
        },
    }


def _float_fields_as_float(payload: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """Coerce exact-``int`` values of ``config``'s float-annotated fields to ``float`` in ``payload``.

    ``VFE3Config`` keeps an ``int`` passed to a float field (``kl_max=160`` versus the width route's
    ``160 * k / k = 160.0``), so two identical runs would otherwise get different identities.
    """
    out = dict(payload)
    hints = typing.get_type_hints(type(config))
    for name, value in out.items():
        hint = hints.get(name)
        if type(value) is int and (
                hint is float
                or (typing.get_origin(hint) is typing.Union
                    and float in typing.get_args(hint) and int not in typing.get_args(hint))):
            out[name] = float(value)
    return out


_BLOCK_MLP_STRUCTURAL_METADATA_FIELDS = {
    "use_block_mlp": "enabled",
    "block_mlp_mode": "mode",
    "block_mlp_covariance": "covariance",
    "block_mlp_expansion": "expansion",
    "block_mlp_activation": "activation",
    "block_mlp_dropout": "dropout",
    "block_mlp_covariance_floor": "covariance_floor",
}


def _block_mlp_structural_values(
    config_json: Mapping[str, Any],
    serialized_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prefer what the executable built; retain serialized-config fallback for old runs."""
    executable = config_json.get("executable_build")
    block_mlp = executable.get("block_mlp") if isinstance(executable, Mapping) else None
    if not isinstance(block_mlp, Mapping):
        return {
            field: serialized_config.get(field)
            for field in _BLOCK_MLP_STRUCTURAL_METADATA_FIELDS
        }
    return {
        field: block_mlp.get(metadata_field, serialized_config.get(field))
        for field, metadata_field in _BLOCK_MLP_STRUCTURAL_METADATA_FIELDS.items()
    }


def _structural_signature(row: Mapping[str, Any]) -> tuple:
    """Exact typed identity of fields that must not vary among seeds of one scaling cell."""
    return tuple(
        (type(row.get(field)).__module__, type(row.get(field)).__qualname__, repr(row.get(field)))
        for field in _SCALING_STRUCTURAL_FIELDS
    )


def _safe_manifest_component(value: object) -> bool:
    """Whether a manifest route or label is one regular Windows path component."""
    try:
        portable_path_component_key(value, field="manifest path component")
    except (TypeError, ValueError):
        return False
    return True


def _exact_join_key(row: Mapping[str, Any]) -> Optional[Tuple[str, str, int]]:
    """Return a scaling identity only when no route, label, or seed coercion is required."""
    if not isinstance(row, Mapping):
        return None
    route = row.get("route")
    label = row.get("label")
    seed = row.get("seed")
    if (
        type(route) is not str
        or not route
        or type(label) is not str
        or not label
        or not _is_exact_nonnegative_seed(seed)
    ):
        return None
    return route, label, seed


def _observed_join_key(row: Mapping[str, Any]) -> Optional[Tuple[str, str, int]]:
    """Return an exact join key only for a generation-bound harvested artifact."""
    if not isinstance(row, Mapping) or row.get("artifact_verified") is not True:
        return None
    return _exact_join_key(row)


def _forced_row_matches_strict_source_anchor(
    row:     Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
) -> bool:
    """Require a forced row's source records to match a strictly accepted peer cohort."""
    source_fields = (
        "train_source_identity",
        "validation_source_identity",
        "test_source_identity",
        "train_data_sha256",
        "val_data_sha256",
        "test_data_sha256",
    )
    return any(
        all(anchor.get(field) == row.get(field) for field in source_fields)
        for anchor in anchors
    )


def harvest(
    input_dir: Path,

    *,
    force_accept_code_identity_drift: bool = False,
) -> List[Dict[str, Any]]:
    r"""One row per run directory (a dir containing ``summary.json``). Pulls the scaling point, the
    held-out test numbers, the structural config, the cell's route/scale_knob, and provenance."""
    rows: List[Dict[str, Any]] = []
    for summ_path in sorted(input_dir.glob("**/summary.json")):
        if ".attempts" in summ_path.parts:
            continue
        run = summ_path.parent
        summ = _read_json(summ_path)
        sp = summ.get("scaling_point", {})
        cfgj = _read_json(run / "config.json")
        cfg = cfgj.get("config", {}) if isinstance(cfgj, Mapping) else {}
        prov = _read_json(run / "provenance.json")
        cell = _read_json(run / "scaling_cell.json")
        verified = _validated_bound_scaling_run(
            run,
            summ,
            cfgj,
            prov,
            cell,
            force_accept_code_identity_drift=force_accept_code_identity_drift,
        )
        if verified is None or not isinstance(sp, Mapping) or not isinstance(cfg, Mapping):
            logger.warning(
                "scaling_analysis: run %s failed generation-contract validation and is not harvested",
                run,
            )
            continue
        effective_cfg = verified.get("effective_config")
        if not isinstance(effective_cfg, Mapping):
            continue
        block_mlp_structure = _block_mlp_structural_values(cfgj, effective_cfg)
        rows.append({
            "run_dir":     str(run.resolve()),
            "effective_run_key": verified["effective_run_key"],
            "route":       cell["route"],
            "scale_knob":  cell["scale_knob"],
            "label":       cell["label"],
            "seed":        verified["seed"],
            "artifact_verified": True,
            "n_params":    verified["n_params"],
            "n_learnable_params":      sp.get("n_learnable_params"),
            # total minus per-token / positional lookup tables (absent on older runs)
            "n_non_embedding_params":  sp.get("n_non_embedding_params"),
            "embed_dim":   effective_cfg.get("embed_dim"),
            "n_heads":     effective_cfg.get("n_heads"),
            "n_gen":       sp.get("n_gen", cell.get("n_gen")),
            "gauge_group": effective_cfg.get("gauge_group"),
            "n_layers":    effective_cfg.get("n_layers"),
            "n_e_steps":   effective_cfg.get("n_e_steps"),
            "family":      effective_cfg.get("family"),
            "full_cov_kl_precision": effective_cfg.get("full_cov_kl_precision"),
            **block_mlp_structure,
            "tokens_seen": sp.get("tokens_seen"),
            "est_flops_6ND":           sp.get("est_flops_6ND"),
            "est_flops_analytic":      sp.get("est_flops_analytic"),
            "active_params_per_token": sp.get("active_params_per_token"),
            "test_ce":     verified["test_ce"],
            "test_ppl":    verified["test_ppl"],
            "test_bits_per_token": verified["test_bits_per_token"],
            "test_bpc": verified["test_bpc"],
            "estep_final_f_per_token": summ.get("estep_final_f_per_token"),
            "best_val_ppl": summ.get("best_val_ppl"),
            "wall_time_s": summ.get("wall_time_s", sp.get("wall_time_s")),
            "data_sha256": prov.get("data_sha256") or prov.get("test_data_sha256"),
            "train_data_sha256": prov.get("train_data_sha256"),
            "val_data_sha256": prov.get("val_data_sha256"),
            "test_data_sha256": prov.get("test_data_sha256") or prov.get("data_sha256"),
            "git_sha":     verified["git_sha"],
            "git_dirty":   verified["git_dirty"],
            "git_dirty_fingerprint": verified["git_dirty_fingerprint"],
            "code_identity_forced": verified["code_identity_forced"],
            "cell_git_dirty": verified["cell_git_dirty"],
            "cell_git_dirty_fingerprint": verified["cell_git_dirty_fingerprint"],
            "provenance_git_dirty": verified["provenance_git_dirty"],
            "provenance_git_dirty_fingerprint": verified["provenance_git_dirty_fingerprint"],
            "train_source_identity": verified["train_source_identity"],
            "validation_source_identity": verified["validation_source_identity"],
            "test_source_identity": verified["test_source_identity"],
        })
    if force_accept_code_identity_drift:
        strict_anchors = [row for row in rows if row["code_identity_forced"] is False]
        rows = [
            row for row in rows
            if row["code_identity_forced"] is False
            or _forced_row_matches_strict_source_anchor(row, strict_anchors)
        ]
    return rows


def _requested_design(
    input_dir: Path,
    rows:      List[Dict[str, Any]],

    *,
    force_accept_code_identity_drift: bool = False,
) -> Dict[str, Any]:
    """Left-join the declared scaling cells to harvested results and classify every request."""
    path = input_dir / "scaling_design.json"
    selected_route: Optional[str] = None
    if not path.is_file():
        parent_path = input_dir.parent / "scaling_design.json"
        if parent_path.is_file():
            path = parent_path
            selected_route = input_dir.name
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        return {
            "available": False,
            "complete": False,
            "status": "unverifiable_design",
            "cells": [],
            "counts": {},
        }
    requested = raw.get("cells")
    declared_routes = raw.get("routes")
    routes = declared_routes
    route_subset_valid = selected_route is None
    if selected_route is not None and isinstance(declared_routes, list):
        route_subset_valid = selected_route in declared_routes
        routes = [selected_route]
        if isinstance(requested, list):
            requested = [
                item for item in requested
                if (
                    not isinstance(item, Mapping)
                    or item.get("route") == selected_route
                    or type(item.get("route")) is not str
                    or item.get("route") not in declared_routes
                )
            ]
    seeds = raw.get("seeds")
    top_status = raw.get("status")
    known_statuses = {
        "pending", "running", "complete", "success", "incomplete", "failed",
        "missing", "duplicate", "nonfinite", "unreadable", "inconsistent",
    }
    valid_routes = (
        isinstance(declared_routes, list)
        and bool(declared_routes)
        and all(type(route) is str and bool(route) for route in declared_routes)
        and len(set(declared_routes)) == len(declared_routes)
        and route_subset_valid
    )
    valid_seeds = (
        isinstance(seeds, list)
        and bool(seeds)
        and all(_is_exact_nonnegative_seed(seed) for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    valid_header = (
        path.is_file()
        and raw.get("schema_version") == 1
        and valid_routes
        and valid_seeds
        and isinstance(top_status, str)
        and top_status in known_statuses
        and isinstance(requested, list)
        and bool(requested)
    )
    if not valid_header:
        return {
            "available": False,
            "complete": False,
            "status": "unverifiable_design",
            "cells": [],
            "counts": {},
        }

    observed: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = _observed_join_key(row)
        if key is None:
            continue
        observed.setdefault(key, []).append(row)

    cells: List[Dict[str, Any]] = []
    requested_keys: set[Tuple[str, str, int]] = set()
    route_labels: set[Tuple[str, str]] = set()
    matched_rows: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    schema_valid = True
    for item in requested:
        if not isinstance(item, Mapping):
            cell = {"status": "unreadable", "error": "design cell is not a mapping"}
            schema_valid = False
        else:
            cell = dict(item)
            route = cell.get("route")
            label = cell.get("label")
            seed = cell.get("seed")
            scale_knob = cell.get("scale_knob")
            declared_run_dir = cell.get("run_dir")
            declared = cell.get("status")
            expected_run_dir = (
                f"{route}/{label}/s{seed}"
                if (_safe_manifest_component(route)
                    and _safe_manifest_component(label)
                    and _is_exact_nonnegative_seed(seed))
                else None
            )
            if (
                not _safe_manifest_component(route)
                or route not in routes
                or not _safe_manifest_component(label)
                or not _is_exact_nonnegative_seed(seed)
                or seed not in seeds
                or type(scale_knob) is not str
                or not scale_knob
                or type(declared_run_dir) is not str
                or declared_run_dir != expected_run_dir
                or not isinstance(declared, str)
                or declared not in known_statuses
            ):
                cell["status"] = "unreadable"
                cell["error"] = (
                    "design cell lacks a valid route, label, seed, scale_knob, canonical run_dir, "
                    "or status"
                )
                schema_valid = False
            else:
                key = (route, label, seed)
                route_labels.add((route, label))
                duplicate_request = key in requested_keys
                if duplicate_request:
                    cell["status"] = "duplicate"
                    cell["error"] = "design declares the same route, label, and seed more than once"
                    schema_valid = False
                else:
                    requested_keys.add(key)
                expected_path = (path.parent / declared_run_dir).resolve()
                matches = [
                    row for row in observed.get(key, [])
                    if row.get("scale_knob") == scale_knob
                    and isinstance(row.get("run_dir"), str)
                    and Path(row["run_dir"]).resolve() == expected_path
                ]
                if duplicate_request:
                    pass
                elif declared not in {"complete", "success"}:
                    cell["status"] = declared
                elif len(matches) > 1:
                    cell["status"] = "duplicate"
                elif not matches:
                    cell["status"] = "missing"
                else:
                    value = _as_float(matches[0].get("test_ce"))
                    cell["status"] = (
                        "complete" if np.isfinite(value) and value > 0.0 else "nonfinite"
                    )
                    if cell["status"] == "complete":
                        matched_rows[key] = matches[0]
        cells.append(cell)

    structural_groups: Dict[Tuple[str, str], List[Tuple[Tuple[str, str, int], Dict[str, Any]]]] = {}
    for key, row in matched_rows.items():
        structural_groups.setdefault((key[0], key[1]), []).append((key, row))
    inconsistent_groups = {
        route_label
        for route_label, entries in structural_groups.items()
        if len({_structural_signature(row) for _key, row in entries}) != 1
    }
    if inconsistent_groups:
        for cell in cells:
            if ((cell.get("route"), cell.get("label")) in inconsistent_groups
                    and cell.get("status") == "complete"):
                cell["status"] = "inconsistent"
                cell["error"] = "seed runs disagree on structural scaling identity"
    counts: Dict[str, int] = {}
    for cell in cells:
        status = str(cell.get("status", "unreadable"))
        counts[status] = counts.get(status, 0) + 1

    expected_keys = {
        (route, label, seed)
        for route, label in route_labels
        for seed in seeds
    }
    schema_valid = (
        schema_valid
        and {route for route, _label in route_labels} == set(routes)
        and requested_keys == expected_keys
    )
    all_cells_complete = (
        schema_valid
        and bool(cells)
        and counts.get("complete", 0) == len(cells)
    )
    strict_complete = all_cells_complete and top_status in {"complete", "success"}
    forced_rows = [
        {
            "route": row["route"],
            "label": row["label"],
            "seed": row["seed"],
            "cell_git_dirty": row.get("cell_git_dirty"),
            "cell_git_dirty_fingerprint": row.get("cell_git_dirty_fingerprint"),
            "provenance_git_dirty": row.get("provenance_git_dirty"),
            "provenance_git_dirty_fingerprint": row.get(
                "provenance_git_dirty_fingerprint"
            ),
        }
        for row in matched_rows.values()
        if row.get("code_identity_forced") is True
    ]
    manifest_error = raw.get("error")
    forced_complete = (
        force_accept_code_identity_drift
        and all_cells_complete
        and top_status == "incomplete"
        and manifest_error == "code identity drifted during the scaling invocation"
        and bool(forced_rows)
    )
    complete = strict_complete or forced_complete
    return {
        "available": schema_valid,
        "complete": complete,
        "status": "complete" if complete else "incomplete" if schema_valid else "unverifiable_design",
        "manifest_status": top_status,
        "manifest_error": manifest_error,
        "forced_code_identity_acceptance": bool(forced_rows),
        "forced_manifest_completion": forced_complete,
        "forced_row_count": len(forced_rows),
        "forced_rows": forced_rows,
        "cells": cells,
        "counts": counts,
    }


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    r"""Tidy ``scaling_points.csv`` (fixed columns; missing keys blank). Python file I/O, never
    PowerShell ``>`` redirection (which writes UTF-16LE+BOM that numpy/csv misread)."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})


# =============================================================================
# AGGREGATION  (seeds -> one point per (route, label))
# =============================================================================

def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Pearson correlation without NumPy's unstable native corrcoef path."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if (len(left_values) != len(right_values) or len(left_values) < 2
            or any(not math.isfinite(value) for value in (*left_values, *right_values))):
        return float("nan")
    left_mean = math.fsum(left_values) / len(left_values)
    right_mean = math.fsum(right_values) / len(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered))
    if denominator == 0.0:
        return float("nan")
    return math.fsum(a * b for a, b in zip(left_centered, right_centered)) / denominator



def _sorted_strings(rows: List[Dict[str, Any]], key: str) -> List[str]:
    return sorted({str(r[key]) for r in rows if r.get(key)})


def _analysis_provenance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Summarize the code, data, and token-budget identities of the fitted parameter rows."""
    sha_keys = ("data_sha256", "train_data_sha256", "val_data_sha256", "test_data_sha256")
    sha_sets = {key: _sorted_strings(rows, key) for key in sha_keys}
    git_shas = _sorted_strings(rows, "git_sha")
    token_values = sorted({
        value for value in (_as_float(r.get("tokens_seen")) for r in rows)
        if np.isfinite(value)
    })
    token_budgets: List[int | float] = [int(v) if float(v).is_integer() else float(v) for v in token_values]
    missing = {
        "git_sha": sum(not r.get("git_sha") for r in rows),
        "train_data_sha256": sum(not r.get("train_data_sha256") for r in rows),
        "val_data_sha256": sum(not r.get("val_data_sha256") for r in rows),
        "test_data_sha256": sum(not (r.get("test_data_sha256") or r.get("data_sha256")) for r in rows),
    }
    mixed_corpus = any(len(sha_sets[key]) > 1 for key in sha_keys)
    forced_rows = [row for row in rows if row.get("code_identity_forced") is True]
    return {
        **sha_sets,
        "git_sha": git_shas,
        "n_distinct_data_sha256": len(sha_sets["data_sha256"]),
        "n_distinct_train_data_sha256": len(sha_sets["train_data_sha256"]),
        "n_distinct_val_data_sha256": len(sha_sets["val_data_sha256"]),
        "n_distinct_test_data_sha256": len(sha_sets["test_data_sha256"]),
        "n_distinct_git_sha": len(git_shas),
        "mixed_corpus": mixed_corpus,
        "code_drift": len(git_shas) > 1,
        "code_identity_forced": bool(forced_rows),
        "forced_code_identity_rows": len(forced_rows),
        "cell_dirty_fingerprints": _sorted_strings(rows, "cell_git_dirty_fingerprint"),
        "provenance_dirty_fingerprints": _sorted_strings(
            rows, "provenance_git_dirty_fingerprint"
        ),
        "token_budgets": token_budgets,
        "n_distinct_token_budgets": len(token_budgets),
        "token_budget_varies": len(token_budgets) > 1,
        "n_missing_token_budgets": sum(not np.isfinite(_as_float(r.get("tokens_seen"))) for r in rows),
        "missing": missing,
    }


def _pooled_status(
    provenance: Dict[str, Any],
    frontier:   Dict[str, Any],

    *,
    has_fit:          bool,
    n_routes:         int,
    n_distinct_sizes: int,
    lack_of_fit:      Optional[Mapping[str, Any]],
) -> Tuple[str, List[str]]:
    r"""Classify the pooled L(N) fit: provenance confounds first, then testability, then form.

    ``not_testable`` below ``SCALING_MIN_TESTABLE_SIZES`` distinct sizes (R^2 and the CI are
    withheld there); ``lack_of_fit`` when the pure-error F-test rejects the fitted law; ``clean``
    only when the design is testable, the lack-of-fit test ran and did not reject, and the frontier
    question is settled or moot."""
    if not has_fit:
        return "not_fitted", []
    reasons: List[str] = []
    if provenance.get("code_drift"):
        reasons.append("code_drift")
    if provenance.get("code_identity_forced"):
        reasons.append("code_identity_forced")
    if provenance.get("mixed_corpus"):
        reasons.append("mixed_corpus")
    if provenance.get("token_budget_varies"):
        reasons.append("token_budget_varies")
    if any(int(v) > 0 for v in (provenance.get("missing") or {}).values()):
        reasons.append("incomplete_provenance")
    if int(provenance.get("n_missing_token_budgets", 0)) > 0:
        reasons.append("incomplete_token_budget")
    if frontier.get("testable") and frontier.get("collapses") is False:
        reasons.append("routes_diverge")
    if reasons:
        return "confounded", reasons
    if n_distinct_sizes < SCALING_MIN_TESTABLE_SIZES:
        return "not_testable", []
    if lack_of_fit is not None and lack_of_fit.get("rejected") is True:
        return "lack_of_fit", []
    if frontier.get("testable") and frontier.get("collapses") is None:
        return "unassessed", []
    if n_routes > 1 and not frontier.get("testable"):
        return "unassessed", []
    if lack_of_fit is None or lack_of_fit.get("testable") is not True:
        return "unassessed", []
    return "clean", []


def _lack_of_fit_test(
    points: List[Dict[str, Any]],
    fit:    Mapping[str, Any],

    *,
    x_key:  str = "n_params",
) -> Dict[str, Any]:
    r"""Pure-error lack-of-fit F-test of a fitted L(N) law, in log-CE space.

    Seeds at one distinct size are replicates. With :math:`y_{ij} = \log \mathrm{CE}` of seed j at
    size i (:math:`n_i` seeds, m sizes, n observations), the fitted curve
    :math:`\hat y_i = \log(E + A N_i^{-\alpha})` and p fitted parameters (2, or 3 with the offset),

    .. math::
        SS_{PE} = \sum_{ij} (y_{ij} - \bar y_i)^2, \quad
        SS_{LOF} = \sum_i n_i (\bar y_i - \hat y_i)^2, \quad
        F = \frac{SS_{LOF} / (m - p)}{SS_{PE} / (n - m)} \sim F_{m-p,\, n-m}

    under the null that the law's form is right (Draper and Smith 1998, Sec. 2.1). The headline fit is
    SEM-weighted, so the test is approximate; it is untestable without a residual degree of freedom
    (m <= p) or without replicate seeds (n = m or zero pure error)."""
    groups: Dict[float, List[float]] = {}
    for p in points:
        size = _as_float(p.get(x_key))
        if not (np.isfinite(size) and size > 0.0):
            continue
        groups.setdefault(size, []).extend(
            math.log(c) for c in (_as_float(v) for v in p.get("ce_seeds", ()))
            if np.isfinite(c) and c > 0.0)
    groups = {size: ys for size, ys in groups.items() if ys}
    m = len(groups)
    n = sum(len(ys) for ys in groups.values())
    n_fit_parameters = 3 if fit.get("form") == "offset_power_law" else 2
    df_lof, df_pe = m - n_fit_parameters, n - m
    out: Dict[str, Any] = {"testable": False, "rejected": None, "df_lof": df_lof, "df_pe": df_pe,
                           "n_fit_parameters": n_fit_parameters}
    if df_lof < 1:
        out["reason"] = "no residual degree of freedom beyond the fitted parameters"
        return out
    if df_pe < 1:
        out["reason"] = "no replicate seeds (no pure-error estimate)"
        return out
    alpha, A, E = (_as_float(fit.get(key)) for key in ("alpha", "A", "E"))
    ss_pe = ss_lof = 0.0
    for size, ys in groups.items():
        ybar = sum(ys) / len(ys)
        ss_pe += sum((y - ybar) ** 2 for y in ys)
        predicted = E + A * size ** (-alpha)
        if not (np.isfinite(predicted) and predicted > 0.0):
            out["reason"] = "fitted curve is not positive at a fitted size"
            return out
        ss_lof += len(ys) * (ybar - math.log(predicted)) ** 2
    if ss_pe <= 0.0:
        out["reason"] = "zero pure error across replicate seeds"
        return out
    F = (ss_lof / df_lof) / (ss_pe / df_pe)
    try:
        from scipy.stats import f as f_dist
        p_value = float(f_dist.sf(F, df_lof, df_pe))
    except Exception:
        p_value = float("nan")
    out.update({"F": float(F), "ss_lof": ss_lof, "ss_pe": ss_pe})
    if not np.isfinite(p_value):
        out["reason"] = "F distribution unavailable"
        return out
    out.update({"testable": True, "p_value": p_value, "rejected": p_value <= 0.05})
    return out


def _frontier_verdict(frontier: Dict[str, Any]) -> str:
    collapses = frontier.get("collapses")
    if collapses is True or frontier.get("null_rejected") is False:
        return "shared-frontier null not rejected (equivalence unestablished)"
    if collapses is False:
        return "routes diverge (route-specific slope/intercept)"
    return "indeterminate (route collapse could not be decided)"


def aggregate_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""Collapse seeds: one point per (route, label) carrying ``ce_seeds`` (the per-seed test CE) and
    the exact shared structural fields. A structural disagreement withholds the cell instead of
    synthesizing a mean-capacity model that was never run. Points with no finite positive CE are dropped."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["route"], r["label"]), []).append(r)
    points: List[Dict[str, Any]] = []
    for (route, label), grp in sorted(groups.items()):
        signatures = {_structural_signature(row) for row in grp}
        if len(signatures) != 1:
            logger.warning(
                "structural identity varies across seeds for %s/%s; withholding the scaling cell",
                route,
                label,
            )
            continue
        ce = [_as_float(g["test_ce"]) for g in grp]
        ce = [c for c in ce if np.isfinite(c) and c > 0]
        if not ce:
            continue
        npars = [_as_float(g["n_params"]) for g in grp if np.isfinite(_as_float(g["n_params"]))]
        non_embedding = {_as_float(g.get("n_non_embedding_params")) for g in grp}
        # C2/EXP-5: per-seed converged final E-step F/token and named bit metrics.
        f_seeds = [c for c in (_as_float(g.get("estep_final_f_per_token")) for g in grp) if np.isfinite(c)]
        bits_per_token = [
            value for value in (_as_float(g.get("test_bits_per_token")) for g in grp)
            if np.isfinite(value)
        ]
        bpc = [c for c in (_as_float(g.get("test_bpc")) for g in grp) if np.isfinite(c)]
        # F1/EXP-6: per-seed test PPL for the muP K-stability figure (CE is the L(N) fit quantity above).
        ppl = [c for c in (_as_float(g.get("test_ppl")) for g in grp) if np.isfinite(c) and c > 0]
        points.append({
            "route": route, "label": label, "scale_knob": grp[0]["scale_knob"],
            "n_params": npars[0] if npars else float("nan"),
            # one finite value shared by every seed, else NaN (the non-embedding fit is withheld)
            "n_non_embedding_params": (
                next(iter(non_embedding))
                if len(non_embedding) == 1 and np.isfinite(next(iter(non_embedding)))
                else float("nan")),
            "embed_dim": _as_float(grp[0].get("embed_dim")),         # F1/EXP-6 K axis
            "n_gen": _as_float(grp[0].get("n_gen")),
            "tokens_seen": _as_float(grp[0].get("tokens_seen")),
            "est_flops_6ND": _as_float(grp[0].get("est_flops_6ND")),
            "est_flops_analytic": _as_float(grp[0].get("est_flops_analytic")),
            "n_e_steps": _as_float(grp[0].get("n_e_steps")),
            "n_layers": _as_float(grp[0].get("n_layers")),
            "ce_seeds": ce, "n_seeds": len(ce),
            # one entry per seed run; identical tuples mark the same training runs under two routes
            "effective_run_keys": tuple(sorted(
                str(g["effective_run_key"]) for g in grp if g.get("effective_run_key"))),
            "ce_mean": float(np.mean(ce)),
            "ce_sem": float(np.std(ce, ddof=1) / np.sqrt(len(ce))) if len(ce) > 1 else 0.0,
            "f_seeds": f_seeds,
            "f_mean": float(np.mean(f_seeds)) if f_seeds else float("nan"),
            "bits_per_token_seeds": bits_per_token,
            "bits_per_token_mean": (
                float(np.mean(bits_per_token)) if bits_per_token else float("nan")),
            "bpc_seeds": bpc,
            "bpc_mean": float(np.mean(bpc)) if bpc else float("nan"),
            "ppl_mean": float(np.mean(ppl)) if ppl else float("nan"),
            "ppl_sem": float(np.std(ppl, ddof=1) / np.sqrt(len(ppl))) if len(ppl) > 1 else 0.0,
        })
    return points


def _collapse_identical_points(
    points: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    r"""Keep one point per set of identical training runs before pooling.

    The standard routes share cells: ``width`` K{k} is ``width_depth`` K{k}_L6 and ``depth`` L{d}
    is ``width_depth`` K160_L{d}, trained with the same seeds, data order and deterministic kernels.
    Two points whose per-seed effective-run identities (config with float fields normalized,
    dataset, data seed, token cap, source and code identity) coincide are one set of observations,
    so the pooled fit, the lack-of-fit test and the bootstrap must count them once. Returns the
    retained points (first in (route, label) order) and a report of every collapsed group.
    Points without a recorded identity are kept as they are, and points must also record the same
    parameter count to collapse.
    """
    kept: List[Dict[str, Any]] = []
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for p in points:
        keys = tuple(p.get("effective_run_keys") or ())
        if not keys:
            kept.append(p)
            continue
        group_key = (keys, _as_float(p.get("n_params")))
        if group_key not in groups:
            kept.append(p)
        groups.setdefault(group_key, []).append(p)
    collapsed = [
        {"kept": f"{grp[0]['route']}/{grp[0]['label']}",
         "dropped_as_identical": [f"{q['route']}/{q['label']}" for q in grp[1:]],
         "n_seed_runs": len(grp[0].get("effective_run_keys") or ())}
        for grp in groups.values() if len(grp) > 1
    ]
    return kept, collapsed


def _axis_routes(points: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    r"""Map each capacity axis to the route that grows it, read from the points' ``scale_knob``.

    A route is a candidate for axis ``a`` when one of its points has ``scale_knob == a``. With
    several candidates the first in ``AXIS_ROUTE_PREFERENCE[a]`` wins (else the first by name), and
    the choice is logged; an axis no route grows is omitted.
    """
    out: Dict[str, str] = {}
    for axis, preference in AXIS_ROUTE_PREFERENCE.items():
        candidates = sorted({str(p["route"]) for p in points if p.get("scale_knob") == axis})
        if not candidates:
            continue
        preferred = [route for route in preference if route in candidates]
        out[axis] = preferred[0] if preferred else candidates[0]
        if len(candidates) > 1:
            logger.info("capacity axis %s drawn from route %s (candidates: %s)",
                        axis, out[axis], ", ".join(candidates))
    return out


def _kmup_series(param_points: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    r"""Group the grow_K / grow_K_mup points into the F1/EXP-6 ``kmup_stability`` series.

    SPLITS the muP route's matched ``K{k}_fixed`` / ``K{k}_mup`` arms -- they share
    ``route='grow_K_mup'`` (so a route-only grouping would collapse both into one duplicated-K curve
    and destroy the |b_fixed - b_muP| width-stability contrast). Keys: ``grow_K`` (the standalone
    width-fixed control), ``grow_K_mup/fixed``, ``grow_K_mup/mup`` -- one point per K each."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in param_points:
        if p["route"] not in ("grow_K", "grow_K_mup"):
            continue
        if not (np.isfinite(_as_float(p.get("embed_dim"))) and np.isfinite(_as_float(p.get("ppl_mean")))):
            continue
        key = p["route"]
        if p["route"] == "grow_K_mup":
            lab = str(p.get("label", ""))
            key = ("grow_K_mup/fixed" if lab.endswith("_fixed")
                   else "grow_K_mup/mup" if lab.endswith("_mup") else "grow_K_mup")
        out.setdefault(key, []).append({
            "embed_dim": _as_float(p["embed_dim"]), "ppl_mean": _as_float(p["ppl_mean"]),
            "ppl_sem": _as_float(p.get("ppl_sem", 0.0)), "n": int(p.get("n_seeds", 1)),
        })
    return out


# =============================================================================
# FITTING  (power law + bootstrap exponent CI + multi-route ANCOVA)
# =============================================================================

def bootstrap_exponent_ci(
    points:      List[Dict[str, Any]],

    *,
    weights:     Optional[np.ndarray] = None,
    n_boot:      int = 2000,
    with_offset: bool = False,
) -> Tuple[float, float, float]:
    r"""Bootstrap the power-law exponent with the same estimator form as the point estimate.

    Ordinary power-law replicates resample size clusters and their seed CEs. A successful offset-law
    fit keeps the distinct-size design fixed (cluster resampling usually makes that three-parameter
    fit underdetermined) and draws the size means from the pooled pure-error model in log-CE space,
    the same model the lack-of-fit test uses. With :math:`y_{ij} = \log \mathrm{CE}_{ij}` (seed j of
    size i, :math:`n_i` seeds, m sizes, n runs) and
    :math:`s^2 = \sum_{ij} (y_{ij} - \bar y_i)^2 / (n - m)`, each replicate draws

    .. math::
        \sigma^{*2} = s^2 (n - m) / \chi^2_{n-m}, \quad
        \bar y_i^* = \bar y_i + (\sigma^* / \sqrt{n_i}) z_i, \quad z_i \sim N(0, 1),

    the posterior of the size means under the noninformative prior for a normal model with a
    common variance (Gelman et al. 2013, *Bayesian Data Analysis*, 3rd ed., Sec. 3.2). For
    any linear functional of the means this reproduces the Student-t interval on n - m degrees of
    freedom exactly, so the interval carries the uncertainty of the estimated seed variance. The
    previous seed-only resampling had variance :math:`(n_i - 1)/n_i` of :math:`s_i^2/n_i` and no
    small-df correction; at 4 sizes x 3 seeds its nominal 95% interval covered the true exponent
    in 0.73 to 0.78 of replicates in two 200-replicate checks (audit 2026-10-08 gap-stats#1).
    Replicates whose realized fit form differs from the point estimate are rejected. Returns (alpha_hat, lo2.5, hi97.5). Needs >= 2 points; degenerate or no
    same-form replicates -> NaNs.

    ``weights`` (per-point, same order as ``points``) MUST match the headline fit's weighting so the
    point estimate and the CI come from ONE estimator -- otherwise a weighted headline alpha can sit
    outside its own unweighted CI (audit r2 id10). Resampled per replicate as ``weights[idx]``."""
    from vfe3.viz.figures import _fit_power_law
    xs = np.array([p["n_params"] for p in points], dtype=float)
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan")
    seed_arrays = [np.asarray(p["ce_seeds"], dtype=float) for p in points]
    means = np.array([a.mean() for a in seed_arrays])
    point_fit = _fit_power_law(xs, means, weights=weights, with_offset=with_offset)
    alpha_hat = point_fit["alpha"]
    point_form = point_fit.get("form")
    if point_form == "offset_power_law" and any(a.size < 2 for a in seed_arrays):
        return alpha_hat, float("nan"), float("nan")
    rng = np.random.default_rng(0)                           # fixed -> reproducible CI
    boot: List[float] = []
    n = len(points)
    if point_form == "offset_power_law":
        if np.any(np.concatenate(seed_arrays) <= 0.0):
            return alpha_hat, float("nan"), float("nan")
        log_seeds = [np.log(a) for a in seed_arrays]
        n_i = np.array([a.size for a in log_seeds], dtype=float)
        df_pe = int(n_i.sum()) - n
        ss_pe = float(sum(np.sum((a - a.mean()) ** 2) for a in log_seeds))
        if df_pe < 1 or not ss_pe > 0.0:
            return alpha_hat, float("nan"), float("nan")
        log_means = np.log(means)
        s2_pe = ss_pe / df_pe
    for _ in range(n_boot):
        if point_form == "offset_power_law":
            # Fixed design; size means from the pooled pure-error posterior (see the docstring).
            sigma2 = s2_pe * df_pe / rng.chisquare(df_pe)
            bx = xs
            by = np.exp(log_means + np.sqrt(sigma2 / n_i) * rng.standard_normal(n))
            bw = weights
        else:
            idx = rng.integers(0, n, n)
            bx = xs[idx]
            by = np.array([float(rng.choice(seed_arrays[i], size=seed_arrays[i].size).mean())
                           for i in idx])
            bw = weights[idx] if weights is not None else None
        fit = _fit_power_law(bx, by, weights=bw, with_offset=with_offset)
        a = fit["alpha"]
        if np.isfinite(a) and fit.get("form") == point_form:
            boot.append(a)
    if not boot:
        return alpha_hat, float("nan"), float("nan")
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    if np.isclose(lo, hi, rtol=8. * np.finfo(float).eps, atol=0.):
        return alpha_hat, float("nan"), float("nan")
    return alpha_hat, lo, hi


def _loglog_rss(x: np.ndarray, y: np.ndarray) -> float:
    r"""Residual sum of squares of a 1-D log-log least-squares line (NaN if < 2 points)."""
    if x.size < 2:
        return float("nan")
    b, a = np.polyfit(x, y, 1)
    return float(np.sum((y - (b * x + a)) ** 2))


def _offset_rss(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, str]:
    r"""Weighted RSS of the offset law :math:`E + A N^{-\alpha}` and the realized fit form.

    The fit is the headline estimator (``_fit_power_law(with_offset=True)`` with the SEM weights
    ``w``, i.e. ``curve_fit`` with :math:`\sigma_i = y_i / \sqrt{w_i}`), and the RSS is that fit's
    own objective :math:`\sum_i ((y_i - \hat y_i) / \sigma_i)^2`. A solver fallback to the log-log
    form is reported through the returned form string."""
    from vfe3.viz.figures import _fit_power_law
    fit = _fit_power_law(x, y, weights=w, with_offset=True)
    if fit.get("form") != "offset_power_law":
        return float("nan"), str(fit.get("form"))
    yhat = float(fit["E"]) + float(fit["A"]) * np.power(x, -float(fit["alpha"]))
    return float(np.sum(((y - yhat) * np.sqrt(w) / y) ** 2)), "offset_power_law"


def ancova_frontier_collapse(
    points:      List[Dict[str, Any]],
    min_points:  int = 2,

    *,
    with_offset: bool = False,
) -> Dict[str, Any]:
    r"""F-test for whether the routes share ONE L(N) curve of the headline form.

    ``with_offset=False``: pooled log-log line (2 params) vs a separate slope+intercept per route
    (2R params). ``with_offset=True``: the extra-sum-of-squares F test for the offset law
    :math:`L = E + A N^{-\alpha}`, pooled (3 params) vs one offset law per route (3R params),

    .. math::
        F = \frac{(RSS_{pooled} - RSS_{full}) / (3R - 3)}{RSS_{full} / (n - 3R)}
        \sim F_{3R-3, n-3R},

    on the residuals of the headline SEM-weighted fit (Bates and Watts 1988, *Nonlinear Regression Analysis
    and Its Applications*; Seber and Wild 1989, *Nonlinear Regression*; the F
    law is asymptotic for a nonlinear model). A log-log line is not the headline curve: the local
    log-log slope of :math:`E + A N^{-\alpha}` changes with N, so routes sampling one offset law
    over different N ranges would otherwise be reported as diverging. The offset test needs four
    distinct sizes per route and a converged offset fit for every route and for the pool;
    otherwise it is not testable. A significant F means the routes do NOT collapse. Non-rejection
    does not establish equivalence. Returns the F statistic, dof, p-value, rejection status, and
    per-route point counts."""
    by_route: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        n_params = _as_float(p.get("n_params"))
        ce_mean = _as_float(p.get("ce_mean"))
        if np.isfinite(n_params) and n_params > 0.0 and np.isfinite(ce_mean) and ce_mean > 0.0:
            by_route.setdefault(p["route"], []).append(p)
    route_sizes = {r: {float(p["n_params"]) for p in ps} for r, ps in by_route.items()}
    routes = {r: ps for r, ps in by_route.items() if len(route_sizes[r]) >= min_points}
    out: Dict[str, Any] = {
        "routes": {r: len(ps) for r, ps in routes.items()},
        "distinct_sizes": {r: len(route_sizes[r]) for r in routes},
        "testable": False,
    }
    if len(routes) < 2:
        out["reason"] = "need >= 2 routes with enough sizes"
        return out
    out["form"] = "offset_power_law" if with_offset else "loglog_line"
    if with_offset:
        return _offset_frontier_collapse(routes, out)
    x_all, y_all = [], []
    rss_full, n_total, R = 0.0, 0, len(routes)
    for r, ps in routes.items():
        x = np.log(np.array([p["n_params"] for p in ps], dtype=float))
        y = np.log(np.array([p["ce_mean"] for p in ps], dtype=float))
        x_all.append(x); y_all.append(y)
        rss_full += _loglog_rss(x, y)
        n_total += x.size
    rss_pooled = _loglog_rss(np.concatenate(x_all), np.concatenate(y_all))
    df1, df2 = 2 * R - 2, n_total - 2 * R
    if df2 <= 0 or not np.isfinite(rss_full) or rss_full <= 0:
        out["reason"] = "insufficient dof for the F-test"
        return out
    F = ((rss_pooled - rss_full) / df1) / (rss_full / df2)
    try:
        from scipy.stats import f as f_dist
        p_value = float(f_dist.sf(F, df1, df2))
    except Exception:
        p_value = float("nan")
    out.update({"testable": True, "F": float(F), "df1": int(df1), "df2": int(df2),
                "p_value": p_value, "rss_pooled": rss_pooled, "rss_full": rss_full,
                "null_rejected": (p_value <= 0.05) if np.isfinite(p_value) else None,
                "collapses": False if np.isfinite(p_value) and p_value <= 0.05 else None})
    return out


def _offset_frontier_collapse(
    routes: Mapping[str, List[Dict[str, Any]]],
    out:    Dict[str, Any],
) -> Dict[str, Any]:
    r"""Extra-sum-of-squares F test of one pooled offset law against one offset law per route."""
    n_law_params = 3
    short = sorted(r for r, ps in routes.items() if len({float(p["n_params"]) for p in ps}) < 4)
    if short:
        out["reason"] = ("offset-law collapse test needs >= 4 distinct sizes per route; "
                         f"{', '.join(short)} has fewer")
        return out
    from vfe3.viz.figures import _scaling_sem_weights
    x_all, y_all, w_all = [], [], []
    rss_full, n_total, R = 0.0, 0, len(routes)
    for r, ps in routes.items():
        x = np.array([p["n_params"] for p in ps], dtype=float)
        y = np.array([p["ce_mean"] for p in ps], dtype=float)
        w = _scaling_sem_weights(y, np.array([_as_float(p.get("ce_sem")) for p in ps], dtype=float))
        rss, form = _offset_rss(x, y, w)
        if not np.isfinite(rss):
            out["reason"] = f"route {r}: offset fit did not converge ({form})"
            return out
        x_all.append(x); y_all.append(y); w_all.append(w)
        rss_full += rss
        n_total += x.size
    rss_pooled, form = _offset_rss(np.concatenate(x_all), np.concatenate(y_all),
                                   np.concatenate(w_all))
    if not np.isfinite(rss_pooled):
        out["reason"] = f"pooled offset fit did not converge ({form})"
        return out
    df1, df2 = n_law_params * R - n_law_params, n_total - n_law_params * R
    if df2 <= 0 or not np.isfinite(rss_full) or rss_full <= 0:
        out["reason"] = "insufficient dof for the F-test"
        return out
    # The pooled law is a constrained per-route law, so RSS_pooled >= RSS_full up to solver tolerance.
    F = (max(rss_pooled - rss_full, 0.0) / df1) / (rss_full / df2)
    try:
        from scipy.stats import f as f_dist
        p_value = float(f_dist.sf(F, df1, df2))
    except Exception:
        p_value = float("nan")
    out.update({"testable": True, "F": float(F), "df1": int(df1), "df2": int(df2),
                "p_value": p_value, "rss_pooled": rss_pooled, "rss_full": rss_full,
                "null_rejected": (p_value <= 0.05) if np.isfinite(p_value) else None,
                "collapses": False if np.isfinite(p_value) and p_value <= 0.05 else None})
    return out


# =============================================================================
# REPORT + FIGURES
# =============================================================================

def _print_points_table(points: List[Dict[str, Any]]) -> None:
    print(f"\n{'route':<14}{'label':<18}{'N params':>14}{'n_gen':>8}{'seeds':>7}"
          f"{'test_ce':>11}{'+/-SEM':>10}")
    print("-" * 82)
    for p in sorted(points, key=lambda q: (q["route"], q["n_params"])):
        print(f"{p['route']:<14}{p['label']:<18}{int(p['n_params']):>14,}{int(p['n_gen']):>8}"
              f"{p['n_seeds']:>7}{p['ce_mean']:>11.4f}{p['ce_sem']:>10.4f}")


def analyze() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    input_dir = Path(CONFIG["input_dir"])
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir {input_dir} does not exist; run scaling.py first")

    force_code_drift = CONFIG.get("force_accept_code_identity_drift") is True
    rows = harvest(
        input_dir,
        force_accept_code_identity_drift=force_code_drift,
    )
    design = _requested_design(
        input_dir,
        rows,
        force_accept_code_identity_drift=force_code_drift,
    )
    if not rows and not design["available"]:
        print(f"No runs (with summary.json) under {input_dir}; run scaling.py first.")
        return
    fig_dir = input_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    write_csv(rows, input_dir / "scaling_points.csv")
    print(f"\nVFE_3.0 scaling analysis\n  input:   {input_dir}\n  runs:    {len(rows)}"
          f"\n  csv:     {input_dir / 'scaling_points.csv'}")
    if design.get("forced_code_identity_acceptance") is True:
        print(
            "\nWARNING: FORCED CODE-IDENTITY ACCEPTANCE is active for "
            f"{design['forced_row_count']} run(s); the fit spans dirty-worktree code cohorts."
        )

    allow_parameter_fit = design.get("complete") is True
    requested_keys = set()
    for cell in design.get("cells", []):
        key = _exact_join_key(cell)
        if cell.get("status") == "complete" and key is not None:
            requested_keys.add(key)
    eligible_rows: List[Dict[str, Any]] = []
    if allow_parameter_fit:
        for row in rows:
            row_key = _observed_join_key(row)
            if row_key is None:
                continue
            if row_key in requested_keys:
                eligible_rows.append(row)
    points = aggregate_points(eligible_rows if allow_parameter_fit else rows)
    # PB-07: best_val_ppl-keyed validation points for the two registered supplementary figures --
    # best-effort: aggregate_validation_points keeps its fail-loud ValueError (an explicit-null
    # best_val_ppl beside finite sibling seeds is a data-integrity fault, never silently dropped or
    # substituted with a test metric), but that fault withholds ONLY the validation figures; it must
    # never abort the legacy test-metric analysis below.
    if allow_parameter_fit:
        try:
            validation_points: Optional[List[Dict[str, Any]]] = aggregate_validation_points(eligible_rows)
        except Exception as exc:
            logger.warning("validation-point aggregation failed (%s); capacity_scaling/pareto_frontier withheld", exc)
            validation_points = None
    else:
        validation_points = None
    param_points = [p for p in points
                    if p["route"] != INFERENCE_ROUTE and p["scale_knob"] not in NON_PARAMETER_KNOBS]
    infer_points = [p for p in points if p["route"] == INFERENCE_ROUTE]
    # Fixed-N training-budget cells (scale_knob max_steps): tokens-axis figure only, never L(N).
    budget_points = [p for p in points
                     if p["route"] != INFERENCE_ROUTE and p["scale_knob"] in NON_PARAMETER_KNOBS]
    fit_infer_points = infer_points if allow_parameter_fit else []
    fit_param_mask = np.array([
        allow_parameter_fit
        and np.isfinite(_as_float(p.get("n_params"))) and _as_float(p.get("n_params")) > 0.0
        for p in param_points
    ], dtype=bool)
    fit_param_points = [p for p, keep in zip(param_points, fit_param_mask) if keep]
    retained_param_cells = {
        (p["route"], p["label"])
        for p in fit_param_points
    }
    param_rows = [
        r for r in eligible_rows
        if (r["route"], r["label"]) in retained_param_cells
        and np.isfinite(_as_float(r.get("test_ce"))) and _as_float(r.get("test_ce")) > 0.0
    ]
    _print_points_table(points)

    # Persist every identity used to judge whether the parameter frontier is comparable.  Scope the
    # sets to the rows that actually enter L(N), not flat-N inference arms that are plotted separately.
    provenance = _analysis_provenance(param_rows)
    if provenance["mixed_corpus"]:
        logger.warning("harvested parameter runs span multiple data SHA-256 identities -- mixed corpus confounds the fit")
    if provenance["code_drift"]:
        logger.warning("harvested parameter runs span %d distinct git_sha -- code drift across the sweep",
                       provenance["n_distinct_git_sha"])
    if provenance["token_budget_varies"]:
        logger.warning("harvested parameter runs span multiple token budgets -- data budget confounds the fit")

    routes = sorted({p["route"] for p in fit_param_points})
    # Identical training runs reached through two routes enter the pooled estimators once.
    pooled_param_points, collapsed_cells = _collapse_identical_points(fit_param_points)
    pooled_ids = {id(p) for p in pooled_param_points}
    pooled_mask = np.array([id(p) in pooled_ids for p in fit_param_points], dtype=bool)
    for group in collapsed_cells:
        logger.warning("identical training runs %s counted once in the pooled fit (kept %s)",
                       ", ".join(group["dropped_as_identical"]), group["kept"])
    frontier = (
        ancova_frontier_collapse(fit_param_points, min_points=CONFIG["min_points"],
                                 with_offset=CONFIG["with_offset"])
        if allow_parameter_fit
        else {"testable": False, "collapses": None, "reason": "incomplete_design"}
    )
    route_notes = {route: _ROUTE_NOTES[route] for route in routes if route in _ROUTE_NOTES}

    # Accumulate the printed fits / tests into a persisted summary (scaling_summary.json plus the
    # SCALING_ANALYSIS.md report): the pooled fit, bootstrap CI, per-route exponents, the
    # frontier-collapse F-test, and the E-step structural-EM correlations were console-only.
    summary: Dict[str, Any] = {
        "input_dir": str(input_dir), "n_runs": len(rows), "with_offset": CONFIG["with_offset"],
        "n_param_points": len(param_points), "n_inference_points": len(infer_points),
        "n_fitted_param_points": len(fit_param_points),
        "n_pooled_param_points": len(pooled_param_points),
        "collapsed_identical_cells": collapsed_cells,
        "n_budget_points": len(budget_points),
        "budget_points_excluded_from_fit": sorted(f"{p['route']}/{p['label']}" for p in budget_points),
        "design": design,
        "provenance": provenance, "pooled_fit": None, "pooled_fit_status": "not_fitted",
        "pooled_fit_confounds": [], "pooled_fit_lack_of_fit": None,
        "pooled_fit_non_embedding": None, "per_route": {}, "frontier_collapse": frontier,
        "route_notes": route_notes, "estep_structural": None,
    }

    weights_by_route: Dict[str, np.ndarray] = {}
    if allow_parameter_fit and param_points:
        from vfe3.viz.figures import _scaling_sem_weights
        all_means = np.array([p["ce_mean"] for p in param_points], dtype=float)
        all_sem = np.array([p["ce_sem"] for p in param_points], dtype=float)
        all_weights = _scaling_sem_weights(all_means, all_sem)
        for route in sorted({p["route"] for p in param_points}):
            route_mask = np.array([p["route"] == route for p in param_points], dtype=bool)
            positive_x = np.array([
                np.isfinite(_as_float(p.get("n_params"))) and _as_float(p.get("n_params")) > 0.0
                for p in param_points
            ], dtype=bool)
            weights_by_route[route] = all_weights[route_mask & positive_x]
        fit_weights = all_weights[fit_param_mask]
    else:
        all_weights = np.array([], dtype=float)
        fit_weights = np.array([], dtype=float)
    pooled_weights = fit_weights[pooled_mask] if fit_weights.size else fit_weights

    # ---- L(N) power law + bootstrap exponent CI (pooled over the parameter routes) ----
    n_harvested_param_sizes = len({float(p["n_params"]) for p in param_points})
    n_fit_param_sizes = len({float(p["n_params"]) for p in fit_param_points})
    summary["n_harvested_param_sizes"] = n_harvested_param_sizes
    summary["n_fit_param_sizes"] = n_fit_param_sizes
    summary["n_distinct_param_sizes"] = n_fit_param_sizes
    # Below SCALING_MIN_TESTABLE_SIZES the law is still fit (alpha is a descriptive slope), but R^2
    # and the CI are withheld and the status reads 'not_testable' (two sizes give R^2 = 1 exactly).
    fit_testable = n_fit_param_sizes >= SCALING_MIN_TESTABLE_SIZES
    lack_of_fit: Optional[Dict[str, Any]] = None
    if n_fit_param_sizes >= 2:
        from vfe3.viz.figures import _fit_power_law
        xs = np.array([p["n_params"] for p in pooled_param_points], dtype=float)
        ys = np.array([p["ce_mean"] for p in pooled_param_points], dtype=float)
        fit = _fit_power_law(xs, ys, weights=pooled_weights, with_offset=CONFIG["with_offset"])
        lo = hi = None
        if fit_testable:
            _a_hat, lo, hi = bootstrap_exponent_ci(
                pooled_param_points,
                weights=pooled_weights,
                n_boot=CONFIG["n_bootstrap"],
                with_offset=CONFIG["with_offset"],
            )
        if np.isfinite(_as_float(fit.get("alpha"))):
            n_fit_parameters = 3 if fit["form"] == "offset_power_law" else 2
            if fit_testable:
                lack_of_fit = _lack_of_fit_test(pooled_param_points, fit)
                print(f"\nPOOLED L(N) fit ({fit['form']}):  alpha = {fit['alpha']:.4f}  "
                      f"[95% CI {lo:.4f}, {hi:.4f}]   A = {fit['A']:.4g}   "
                      + (f"E = {fit['E']:.4f}   " if fit["form"] == "offset_power_law" else "")
                      + f"R^2 = {fit['r2']:.4f}   over {fit['n_distinct_sizes']} distinct sizes")
                if lack_of_fit.get("testable"):
                    print(f"  lack-of-fit F({lack_of_fit['df_lof']},{lack_of_fit['df_pe']}) = "
                          f"{lack_of_fit['F']:.3f}  p = {lack_of_fit['p_value']:.4g}")
                else:
                    print(f"  lack-of-fit test not testable ({lack_of_fit.get('reason')})")
            else:
                print(f"\nPOOLED L(N) fit ({fit['form']}):  alpha = {fit['alpha']:.4f}   "
                      f"A = {fit['A']:.4g}   over {fit['n_distinct_sizes']} distinct sizes -- "
                      f"NOT TESTABLE (< {SCALING_MIN_TESTABLE_SIZES} sizes): R^2 and CI withheld")
            summary["pooled_fit"] = {
                "form": fit["form"], "alpha": fit["alpha"],
                "alpha_ci": [lo, hi] if fit_testable else None,
                "A": fit["A"], "E": fit.get("E"), "r2": fit["r2"] if fit_testable else None,
                "n_points": fit["n_points"], "n_distinct_sizes": fit["n_distinct_sizes"],
                "n_fit_parameters": n_fit_parameters,
                "residual_dof": int(fit["n_distinct_sizes"]) - n_fit_parameters,
                "testable": fit_testable,
            }
            summary["pooled_fit_lack_of_fit"] = lack_of_fit
        else:
            print("\n(pooled L(N) estimator returned no finite exponent; no estimate persisted)")
        # ---- the same law against NON-EMBEDDING parameters (total minus per-token / positional
        # lookup tables, which are ~99% of N at the scaling widths; Kaplan et al. 2020) ----
        non_embedding = np.array([_as_float(p.get("n_non_embedding_params"))
                                  for p in pooled_param_points], dtype=float)
        if (non_embedding.size and np.all(np.isfinite(non_embedding)) and np.all(non_embedding > 0.0)
                and np.unique(non_embedding).size >= 2):
            ne_fit = _fit_power_law(non_embedding, ys, weights=pooled_weights,
                                    with_offset=CONFIG["with_offset"])
            ne_testable = int(ne_fit["n_distinct_sizes"]) >= SCALING_MIN_TESTABLE_SIZES
            if np.isfinite(_as_float(ne_fit.get("alpha"))):
                print(f"NON-EMBEDDING L(N) fit ({ne_fit['form']}):  alpha = {ne_fit['alpha']:.4f}   "
                      + (f"R^2 = {ne_fit['r2']:.4f}   " if ne_testable
                         else f"R^2 withheld (< {SCALING_MIN_TESTABLE_SIZES} sizes)   ")
                      + f"over {ne_fit['n_distinct_sizes']} distinct sizes")
                summary["pooled_fit_non_embedding"] = {
                    "x_axis": "n_non_embedding_params", "form": ne_fit["form"],
                    "alpha": ne_fit["alpha"], "A": ne_fit["A"], "E": ne_fit.get("E"),
                    "r2": ne_fit["r2"] if ne_testable else None,
                    "n_points": ne_fit["n_points"], "n_distinct_sizes": ne_fit["n_distinct_sizes"],
                    "testable": ne_testable,
                }
        # ---- per-route fits use exact slices of the one pooled SEM-weight vector ----
        if routes:
            print("\nper-route exponents:")
            for r in routes:
                pr = [p for p in fit_param_points if p["route"] == r]
                route_sizes = {float(p["n_params"]) for p in pr}
                if len(route_sizes) >= 2:
                    route_mask = np.array([p["route"] == r for p in fit_param_points], dtype=bool)
                    route_weights = fit_weights[route_mask]
                    fr = _fit_power_law(np.array([p["n_params"] for p in pr], dtype=float),
                                        np.array([p["ce_mean"] for p in pr], dtype=float),
                                        weights=route_weights, with_offset=CONFIG["with_offset"])
                    if not np.isfinite(_as_float(fr.get("alpha"))):
                        print(f"  {r:<14} (estimator returned no finite exponent; no estimate persisted)")
                        continue
                    display = f"{r} [structural ablation]" if r in route_notes else r
                    route_testable = int(fr["n_distinct_sizes"]) >= SCALING_MIN_TESTABLE_SIZES
                    r2_text = (f"R^2={fr['r2']:.4f}" if route_testable
                               else f"R^2 not testable (< {SCALING_MIN_TESTABLE_SIZES} sizes)")
                    print(f"  {display:<36} alpha={fr['alpha']:.4f}  {r2_text}  "
                          f"({fr['n_distinct_sizes']} distinct sizes; {fr['form']})")
                    summary["per_route"][r] = {
                        "form": fr["form"], "alpha": fr["alpha"], "A": fr["A"], "E": fr.get("E"),
                        "r2": fr["r2"] if route_testable else None, "n_points": fr["n_points"],
                        "n_distinct_sizes": fr["n_distinct_sizes"], "n_sizes": fr["n_distinct_sizes"],
                        "testable": route_testable,
                    }
                else:
                    print(f"  {r:<14} (only {len(route_sizes)} distinct size; need >= 2 to fit)")
    elif not allow_parameter_fit:
        print(
            f"\n({n_harvested_param_sizes} harvested parameter sizes; "
            "fitting withheld (incomplete_design))"
        )
    else:
        print(
            f"\n(only {n_fit_param_sizes} distinct parameter size present; "
            "add more sizes to fit L(N))"
        )

    if frontier.get("testable"):
        verdict = _frontier_verdict(frontier)
        print(f"\nfrontier-collapse F-test:  F({frontier['df1']},{frontier['df2']}) = {frontier['F']:.3f}  "
              f"p = {frontier['p_value']:.4g}  ->  {verdict}")
    else:
        print(f"\nfrontier-collapse F-test: not testable ({frontier.get('reason')})")

    if allow_parameter_fit:
        status, confounds = _pooled_status(
            provenance,
            frontier,
            has_fit=summary["pooled_fit"] is not None,
            n_routes=len(routes),
            n_distinct_sizes=n_fit_param_sizes,
            lack_of_fit=lack_of_fit,
        )
    else:
        status, confounds = "incomplete_design", ["incomplete_design"]
    summary["pooled_fit_status"] = status
    summary["pooled_fit_confounds"] = confounds
    if summary["pooled_fit"] is not None:
        detail = f" ({', '.join(confounds)})" if confounds else ""
        print(f"pooled fit status: {status.upper()}{detail}")

    # ---- C2/EXP-5: structural non-Neal-Hinton EM -- F-vs-CE decorrelation across the n_e_steps arms ----
    estep_pts = sorted([p for p in fit_infer_points if p["scale_knob"] == "n_e_steps"
                        and np.isfinite(p.get("f_mean", float("nan")))],
                       key=lambda q: q["n_e_steps"])
    if len(estep_pts) >= 2:
        ne = np.array([p["n_e_steps"] for p in estep_pts])
        ff = np.array([p["f_mean"] for p in estep_pts])
        ce = np.array([p["ce_mean"] for p in estep_pts])

        pearson_ne_f = _pearson_correlation(ne, ff)
        pearson_f_ce = _pearson_correlation(ff, ce)
        interpretation = target_blind_objective_interpretation()
        print(f"\nE-step structural-EM check ({len(estep_pts)} n_e_steps arms):"
              f"\n  Pearson(n_e_steps, final F/token) = {pearson_ne_f:+.4f}"
              f"\n  Pearson(final F/token, test CE)   = {pearson_f_ce:+.4f}"
              f"\n  Interpretation: {interpretation}")
        summary["estep_structural"] = {
            "n_arms": len(estep_pts), "pearson_ne_final_f": pearson_ne_f,
            "pearson_final_f_test_ce": pearson_f_ce,
            "interpretation": interpretation,
        }

    # ---- persist the analysis summary (json + a human-readable markdown report); best-effort so a
    # serialization/render error never suppresses the figure pass below ----
    try:
        _write_json_atomic(input_dir / "scaling_summary.json", summary)
        _write_scaling_md(input_dir / "SCALING_ANALYSIS.md", summary)
        print(f"\nsummary -> {input_dir / 'scaling_summary.json'}"
              f"\n           {input_dir / 'SCALING_ANALYSIS.md'}")
    except Exception as exc:
        logger.warning("scaling summary write failed (%s); skipped", exc)

    # ---- figures (best-effort, never fatal) ----
    figure_param_points = param_points if allow_parameter_fit else []
    _make_figures(figure_param_points, fit_infer_points, fig_dir, weights_by_route=weights_by_route,
                  validation_points=validation_points,
                  axis_routes=_axis_routes(validation_points or []),
                  budget_points=budget_points if allow_parameter_fit else [])
    print(f"\nfigures -> {fig_dir}")


def _write_scaling_md(path: Path, summary: Dict[str, Any]) -> None:
    r"""Render ``summary`` as a readable SCALING_ANALYSIS.md (the pooled fit, per-route exponents,
    persisted provenance/confounds, frontier test, and E-step structural-EM correlations)."""
    L = ["# VFE_3.0 scaling analysis", "",
         f"- input: `{summary.get('input_dir')}`",
         f"- runs: {summary.get('n_runs')}  (parameter points: {summary.get('n_param_points')}, "
         f"inference points: {summary.get('n_inference_points')})",
         f"- requested fit: {'E + A N^-alpha (offset)' if summary.get('with_offset') else 'A N^-alpha'}", ""]

    design = summary.get("design") or {}
    if design.get("forced_code_identity_acceptance") is True:
        L += ["## Forced code-identity acceptance", "",
              f"- forced rows: {int(design.get('forced_row_count', 0))}",
              "- scope: dirty-worktree code identity only; all other artifact checks remained strict",
              "- interpretation: the fit spans code cohorts and is not provenance-clean", ""]
        forced_rows = design.get("forced_rows") or []
        if forced_rows:
            L += [
                "| route | label | seed | cell dirty | cell fingerprint | provenance dirty | provenance fingerprint |",
                "|---|---|---:|---:|---|---:|---|",
            ]
            for row in sorted(
                    forced_rows,
                    key=lambda item: (str(item["route"]), str(item["label"]), int(item["seed"])),
            ):
                cell_fingerprint = row.get("cell_git_dirty_fingerprint") or "(clean)"
                provenance_fingerprint = (
                    row.get("provenance_git_dirty_fingerprint") or "(clean)"
                )
                L.append(
                    f"| {row['route']} | {row['label']} | {row['seed']} | "
                    f"{row.get('cell_git_dirty')} | {cell_fingerprint} | "
                    f"{row.get('provenance_git_dirty')} | {provenance_fingerprint} |"
                )
            L.append("")

    provenance = summary.get("provenance") or {}
    if provenance:
        def _values(key: str) -> str:
            values = provenance.get(key) or []
            return ", ".join(f"`{value}`" for value in values) if values else "(missing)"

        L += ["## Provenance and confounds", "",
              f"- Git SHA set: {_values('git_sha')}",
              f"- train-data SHA-256 set: {_values('train_data_sha256')}",
              f"- validation-data SHA-256 set: {_values('val_data_sha256')}",
              f"- test-data SHA-256 set: {_values('test_data_sha256')}",
              f"- legacy held-out data SHA-256 set: {_values('data_sha256')}",
              f"- token budgets: {_values('token_budgets')}",
              f"- code drift: {bool(provenance.get('code_drift'))}",
              f"- forced code identity: {bool(provenance.get('code_identity_forced'))}",
              f"- forced code-identity rows: {int(provenance.get('forced_code_identity_rows', 0))}",
              f"- cell dirty fingerprints: {_values('cell_dirty_fingerprints')}",
              f"- provenance dirty fingerprints: {_values('provenance_dirty_fingerprints')}",
              f"- mixed corpus: {bool(provenance.get('mixed_corpus'))}",
              f"- token-budget variation: {bool(provenance.get('token_budget_varies'))}",
              f"- missing identities: `{json.dumps(provenance.get('missing') or {}, sort_keys=True)}`",
              f"- missing token budgets: {int(provenance.get('n_missing_token_budgets', 0))}", ""]

    pf = summary.get("pooled_fit")
    status = str(summary.get("pooled_fit_status", "unassessed"))
    confounds = summary.get("pooled_fit_confounds") or []
    status_line = f"- status: **{status}**"
    if confounds:
        status_line += f" ({', '.join(str(reason) for reason in confounds)})"
    if pf:
        n_sizes = pf.get("n_distinct_sizes", pf.get("n_points"))
        alpha_ci = pf.get("alpha_ci")
        ci_text = (f"95% CI [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]"
                   if alpha_ci is not None and None not in alpha_ci
                   else f"CI withheld: not testable below {SCALING_MIN_TESTABLE_SIZES} distinct sizes")
        r2_text = (f"R^2 = {pf['r2']:.4f}" if pf.get("r2") is not None
                   else f"R^2 withheld (not testable below {SCALING_MIN_TESTABLE_SIZES} distinct sizes)")
        L += ["## Pooled L(N) power law", "",
              status_line,
              f"- realized fit form: `{pf.get('form', 'unknown')}`",
              f"- exponent alpha = {pf['alpha']:.4f}  ({ci_text})",
              f"- A = {pf['A']:.4g}" + (f", E = {pf['E']:.4f}"
                                        if pf.get("form") == "offset_power_law" and pf.get("E") is not None else ""),
              f"- {r2_text} over {n_sizes} distinct sizes ({pf['n_points']} points)"]
        if pf.get("residual_dof") is not None:
            L.append(f"- residual degrees of freedom: {pf['residual_dof']} "
                     f"({pf.get('n_fit_parameters')} fitted parameters)")
        lof = summary.get("pooled_fit_lack_of_fit")
        if lof:
            L.append(f"- lack-of-fit F({lof['df_lof']},{lof['df_pe']}) = {lof['F']:.3f}, "
                     f"p = {lof['p_value']:.4g}" if lof.get("testable")
                     else f"- lack-of-fit test not testable ({lof.get('reason')})")
        ne = summary.get("pooled_fit_non_embedding")
        if ne:
            L.append(f"- non-embedding parameters: alpha = {ne['alpha']:.4f}, "
                     + (f"R^2 = {ne['r2']:.4f}" if ne.get("r2") is not None else "R^2 withheld")
                     + f" over {ne['n_distinct_sizes']} distinct sizes")
        L.append("")
    else:
        L += ["## Pooled L(N) power law", "", status_line,
              "- no pooled estimate is available", ""]
    pr = summary.get("per_route") or {}
    if pr:
        L += ["## Per-route exponents", "",
              "| route | realized form | alpha | R^2 | distinct sizes |",
              "|---|---|---|---|---|"]
        L += [
            f"| {r} | {d.get('form', 'unknown')} | {d['alpha']:.4f} | "
            f"{(format(d['r2'], '.4f') if d.get('r2') is not None else 'not testable')} | "
            f"{d.get('n_distinct_sizes', d.get('n_sizes'))} |"
            for r, d in pr.items()
        ]
        L.append("")
    route_notes = summary.get("route_notes") or {}
    if route_notes:
        L += ["## Route interpretation", ""]
        L += [f"- `{route}`: {note}." for route, note in route_notes.items()]
        L.append("")
    anc = summary.get("frontier_collapse")
    if anc and anc.get("testable"):
        verdict = _frontier_verdict(anc)
        L += ["## Frontier-collapse F-test", "",
              f"- F({anc['df1']},{anc['df2']}) = {anc['F']:.3f}, p = {anc['p_value']:.4g} "
              f"(per-route curve: {anc.get('form', 'loglog_line')})",
              f"- verdict: {verdict}", ""]
    elif anc:
        L += ["## Frontier-collapse F-test", "", f"- not testable ({anc.get('reason')})", ""]
    collapsed = summary.get("collapsed_identical_cells") or []
    budget = summary.get("budget_points_excluded_from_fit") or []
    if collapsed or budget:
        L += ["## Points not counted separately in the pooled fit", ""]
        L += [f"- identical runs: kept `{group['kept']}`, counted once with "
              f"{', '.join(f'`{label}`' for label in group['dropped_as_identical'])}"
              for group in collapsed]
        L += [f"- training-budget cell (fixed N; tokens axis only): `{label}`" for label in budget]
        L.append("")
    es = summary.get("estep_structural")
    if es:
        L += ["## E-step structural-EM check", "",
              f"- arms: {es['n_arms']}",
              f"- Pearson(n_e_steps, final F/token) = {es['pearson_ne_final_f']:+.4f}",
              f"- Pearson(final F/token, test CE) = {es['pearson_final_f_test_ce']:+.4f}",
              f"- interpretation: {es.get('interpretation', target_blind_objective_interpretation())}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _make_figures(
    param_points: List[Dict[str, Any]],
    infer_points: List[Dict[str, Any]],
    fig_dir:      Path,

    *,
    weights_by_route:  Optional[Mapping[str, np.ndarray]]       = None,
    validation_points: Optional[List[Dict[str, Any]]]           = None,
    axis_routes:       Optional[Mapping[str, str]]              = None,
    budget_points:     Optional[List[Dict[str, Any]]]           = None,
) -> None:
    try:
        from vfe3.viz import figures as figs
        figs.set_publication_style()
    except Exception as exc:
        logger.warning("figures unavailable (%s); skipping the figure pass", exc)
        return

    if validation_points is not None and axis_routes is not None:
        # PB-07: dispatch the two registered validation-metric figures through the declarative
        # FigureSpec seam. vfe3.viz.specs is imported lazily behind this try (the same graceful
        # degradation the figs import above gets), and the whole dispatch is best-effort: a failure
        # here skips only these two figures, never the legacy figure pass below.
        try:
            from vfe3.viz.specs import FigureSpec, emit_registered_figures
            SCALING_FIGURE_SPECS = (
                FigureSpec("capacity_scaling", "capacity_scaling.png",
                           lambda ctx: capacity_scaling_kwargs(ctx["validation_points"], ctx["axis_routes"])),
                FigureSpec("pareto_frontier", "pareto_frontier.png",
                           lambda ctx: pareto_frontier_kwargs(ctx["validation_points"])),
            )
            written = emit_registered_figures(
                SCALING_FIGURE_SPECS,
                {"validation_points": validation_points, "axis_routes": axis_routes},
                fig_dir,
            )
            for out_path in written:
                print(f"  figure -> {out_path}")
        except Exception as exc:
            logger.warning("registered validation figures skipped (%s)", exc)

    def _try(name: str, fn) -> None:
        try:
            fig = fn()
            figs.plt.close(fig)
            print(f"  figure -> {fig_dir / name}")
        except Exception as exc:
            logger.warning("figure %s failed (%s); skipped", name, exc)

    if len(param_points) >= 2:
        _try("scaling_ce_vs_params.png", lambda: figs.plot_scaling_law(
            param_points, x_key="n_params", xlabel="parameters N",
            with_offset=CONFIG["with_offset"], path=str(fig_dir / "scaling_ce_vs_params.png")))
        # Same law against non-embedding parameters when every point records them.
        if all(np.isfinite(_as_float(p.get("n_non_embedding_params"))) for p in param_points):
            _try("scaling_ce_vs_non_embedding_params.png", lambda: figs.plot_scaling_law(
                param_points, x_key="n_non_embedding_params",
                xlabel="non-embedding parameters (N minus lookup tables)",
                with_offset=CONFIG["with_offset"],
                title="Scaling law (test CE vs non-embedding parameters)",
                path=str(fig_dir / "scaling_ce_vs_non_embedding_params.png")))
        if len({p["route"] for p in param_points}) > 1:
            _try("scaling_routes_overlay.png", lambda: figs.plot_scaling_routes(
                param_points, x_key="n_params", xlabel="parameters N",
                with_offset=CONFIG["with_offset"], weights_by_route=weights_by_route,
                path=str(fig_dir / "scaling_routes_overlay.png")))
        # compute axis (a proxy): CE vs estimated FLOPs across the size grid (tokens fixed -> compute
        # grows with N). Labeled a proxy; uses the 6ND column.
        if any(np.isfinite(p["est_flops_6ND"]) for p in param_points):
            _try("scaling_ce_vs_flops.png", lambda: figs.plot_scaling_law(
                param_points, x_key="est_flops_6ND", xlabel="est FLOPs (6ND proxy)",
                with_offset=CONFIG["with_offset"], title="Scaling vs compute (6ND proxy)",
                path=str(fig_dir / "scaling_ce_vs_flops.png")))
        # data axis: only meaningful if tokens_seen actually varies (the fixed-N budget cells
        # of a max_steps route, when present, are the data axis; they never enter L(N)).
        data_points = budget_points if budget_points else param_points
        toks = {p["tokens_seen"] for p in data_points if np.isfinite(p["tokens_seen"])}
        if len(toks) > 1:
            _try("scaling_ce_vs_tokens.png", lambda: figs.plot_scaling_law(
                data_points, x_key="tokens_seen", xlabel="tokens seen (data)",
                with_offset=CONFIG["with_offset"], title="Scaling vs data",
                path=str(fig_dir / "scaling_ce_vs_tokens.png")))
        # Pooled PPL offset law vs width (the June-27 headline result): PPL = E + A K^{-b} over ALL
        # parameter points, distinct from the per-arm kmup_stability split below. Uses the per-point
        # ppl_mean from aggregate_points; gated on >= 2 distinct widths.
        if len({_as_float(p.get("embed_dim")) for p in param_points
                if np.isfinite(_as_float(p.get("embed_dim")))}) > 1:
            _try("ppl_vs_embed_dim_offset.png", lambda: figs.plot_ppl_offset(
                param_points, path=str(fig_dir / "ppl_vs_embed_dim_offset.png")))

    # F1/EXP-6: muP width-stability -- grow_K (width-fixed) vs grow_K_mup's matched _fixed/_mup arms on
    # the shared K=embed_dim axis (test PPL per K + offset power-law fit, b annotated per arm). The
    # split is in _kmup_series so the |b_fixed - b_muP| contrast survives. Auto-emits when any arm has
    # >= 2 K cells.
    kmup = _kmup_series(param_points)
    if any(len(v) >= 2 for v in kmup.values()):
        _try("kmup_stability.png", lambda: figs.plot_kmup_stability(
            kmup, path=str(fig_dir / "kmup_stability.png")))

    if infer_points:
        # group the flat-N inference points by the knob they swept (n_e_steps / n_layers).
        series: Dict[str, List[Dict[str, Any]]] = {}
        for p in infer_points:
            knob = p["scale_knob"]
            x = p.get(knob, float("nan"))
            series.setdefault(knob, []).append({"x": _as_float(x), "ce_seeds": p["ce_seeds"]})
        n_flat = int(infer_points[0]["n_params"]) if np.isfinite(infer_points[0]["n_params"]) else None
        _try("inference_capacity.png", lambda: figs.plot_inference_capacity(
            series, n_params=n_flat, path=str(fig_dir / "inference_capacity.png")))

        # C2/EXP-5: the n_e_steps arms -> the final-iterate objective/CE relationship
        # (needs f_mean + ce_mean) and E-step-as-capacity (BPC + final-iterate F vs T; additionally
        # needs bpc_mean, gated separately, so a heterogeneous run set missing test_bpc still gets
        # the relationship figure rather than a NaN BPC point).
        estep = sorted([p for p in infer_points if p["scale_knob"] == "n_e_steps"
                        and np.isfinite(p.get("f_mean", float("nan")))],
                       key=lambda q: q["n_e_steps"])
        if len(estep) >= 2:
            arms = [{"n_e_steps": p["n_e_steps"], "final_f": p["f_mean"], "ce": p["ce_mean"]}
                    for p in estep]
            # The projected routes fix the divergence to KL, the registered "renyi" functional at
            # order 1; the retired divergence_family field is no longer serialized (audit 2026-10-07 F16).
            divergence_family = "renyi"
            _try(F_CE_RELATIONSHIP_FILENAME, lambda: figs.plot_f_ce_relationship(
                arms, divergence_family=divergence_family,
                path=str(fig_dir / F_CE_RELATIONSHIP_FILENAME)))
            cap = [p for p in estep if np.isfinite(p.get("bpc_mean", float("nan")))]
            if len(cap) >= 2:
                _try("estep_capacity.png", lambda: figs.plot_estep_capacity(
                    [p["n_e_steps"] for p in cap], [p["bpc_mean"] for p in cap],
                    [p["f_mean"] for p in cap], n_params=n_flat,
                    divergence_family=divergence_family,
                    path=str(fig_dir / "estep_capacity.png")))


def main() -> int:
    analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
