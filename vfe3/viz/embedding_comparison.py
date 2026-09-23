r"""Controlled diagnostics for cross-run belief-geometry comparisons.

The two-dimensional UMAP is a display only. Controlled cluster discovery uses the native
channel chart or deterministic PCA, and the persisted record contains metrics rather than
projection coordinates so independently fitted runs are never presented on shared axes.
"""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION              = 2
CONTROLLED_MAX_TOKENS       = 16_384
CONTROLLED_N_NEIGHBORS      = 32
CONTROLLED_MIN_DIST         = 0.1
CONTROLLED_SEEDS            = (0, 1, 2, 3, 4)
CONTROLLED_MIN_CLUSTER_SIZE = 128
CONTROLLED_MIN_SAMPLES      = 10
CONTROLLED_CLUSTER_EPSILON  = 0.0
DIAGNOSTIC_SAMPLE_SIZE      = 2_000
DIAGNOSTIC_N_NEIGHBORS      = 15


def _as_numpy(values: object) -> np.ndarray:
    """Detach a tensor or coerce an array without retaining a device view."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _null_metric(reason: str) -> dict:
    """Return the explicit JSON shape for an unavailable scalar metric."""
    return {"value": None, "reason": reason}


def _finite_metric(value: float) -> dict:
    """Return one finite scalar in the sidecar metric shape."""
    if not np.isfinite(value):
        return _null_metric("metric was nonfinite")
    return {"value": float(value), "reason": None}


def token_fingerprint(token_ids: object) -> str:
    """SHA-256 of ordered token IDs encoded as canonical little-endian signed int64."""
    ids = np.ascontiguousarray(_as_numpy(token_ids), dtype="<i8")
    return hashlib.sha256(ids.tobytes(order="C")).hexdigest()


def cluster_coordinates(
    features:       np.ndarray,
    max_components: int = 10,
) -> 'tuple[np.ndarray, str]':
    """Return native coordinates through 10-D, otherwise deterministic full-SVD PCA."""
    from sklearn.decomposition import PCA

    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"features must be a 2-D matrix, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("features must contain at least one row and one column")
    if not np.isfinite(values).all():
        raise ValueError("features contain nonfinite values")
    dimension = min(max_components, values.shape[1], max(1, values.shape[0] - 1))
    if dimension >= values.shape[1]:
        return values.copy(), f"native {values.shape[1]}-D"
    coordinates = PCA(n_components=dimension, svd_solver="full").fit_transform(values)
    return coordinates, f"PCA {dimension}-D"


def relative_position_quartiles(
    pos_idx: object,

    *,
    seq_len: int,
) -> np.ndarray:
    """Map absolute within-window positions to comparable sequence-relative quartiles."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    positions = _as_numpy(pos_idx).astype(np.int64, copy=False)
    if np.any(positions < 0) or np.any(positions >= seq_len):
        raise ValueError("pos_idx must lie in [0, seq_len)")
    return np.minimum((4 * positions) // seq_len, 3).astype(np.int64, copy=False)


def controlled_contract(
    *,
    kind:             str,
    channel:          str,
    feature_dim:      int,
    feature_chart:    str,
    clustering_space: str,
    seeds:            Sequence[int] = CONTROLLED_SEEDS,
) -> dict:
    """Build the fixed comparison contract persisted beside one controlled UMAP."""
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("controlled UMAP needs at least one seed")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("controlled UMAP seeds must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "controlled",
        "coordinate_status": (
            "Gauge-fixed coordinate diagnostic; values are not gauge-invariant and raw coordinates "
            "must not be pooled across independently trained models."
        ),
        "channel": {
            "kind": str(kind),
            "name": str(channel),
            "feature_dim": int(feature_dim),
            "feature_chart": str(feature_chart),
        },
        "display": {
            "method": "UMAP",
            "n_neighbors": CONTROLLED_N_NEIGHBORS,
            "min_dist": CONTROLLED_MIN_DIST,
            "n_components": 2,
            "init": "pca",
            "seeds": normalized_seeds,
            "display_seed": normalized_seeds[0],
        },
        "clustering": {
            "algorithm": "HDBSCAN",
            "space": str(clustering_space),
            "min_cluster_size": CONTROLLED_MIN_CLUSTER_SIZE,
            "min_samples": CONTROLLED_MIN_SAMPLES,
            "cluster_selection_method": "leaf",
            "cluster_selection_epsilon": CONTROLLED_CLUSTER_EPSILON,
        },
    }


def _diagnostic_indices(n_rows: int) -> np.ndarray:
    """Select one deterministic subset shared by every projection diagnostic."""
    if n_rows <= DIAGNOSTIC_SAMPLE_SIZE:
        return np.arange(n_rows)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n_rows, size=DIAGNOSTIC_SAMPLE_SIZE, replace=False))


def _native_silhouette(
    features: np.ndarray,
    labels:   Optional[object],
) -> dict:
    """Native-chart silhouette with explicit reasons for unavailable or degenerate labels."""
    if labels is None:
        return _null_metric("labels unavailable")
    from sklearn.metrics import silhouette_score

    target = _as_numpy(labels)
    if target.shape[0] != features.shape[0]:
        raise ValueError("silhouette labels are not aligned with features")
    unique = np.unique(target)
    if unique.size < 2:
        return _null_metric("labels are constant")
    if unique.size >= target.shape[0]:
        return _null_metric("every sample has a distinct label")
    sample_size = min(DIAGNOSTIC_SAMPLE_SIZE, features.shape[0])
    value = silhouette_score(
        features,
        target,
        sample_size=(sample_size if sample_size < features.shape[0] else None),
        random_state=0,
    )
    return _finite_metric(float(value))


def _trustworthiness_summary(
    features:       np.ndarray,
    coords_by_seed: Mapping[int, np.ndarray],
) -> dict:
    """Trustworthiness for each display seed on one shared deterministic subset."""
    from sklearn.manifold import trustworthiness

    indices = _diagnostic_indices(features.shape[0])
    if indices.size < 4:
        return {"per_seed": [], "mean": None, "std": None, "count": 0,
                "reason": "fewer than four samples"}
    k = min(DIAGNOSTIC_N_NEIGHBORS, max(1, (indices.size - 1) // 2))
    values = []
    for seed, coordinates in coords_by_seed.items():
        coords = np.asarray(coordinates, dtype=float)
        if coords.shape[0] != features.shape[0]:
            raise ValueError(f"seed {seed} coordinates are not aligned with features")
        value = float(trustworthiness(features[indices], coords[indices], n_neighbors=k))
        if not np.isfinite(value):
            raise ValueError(f"seed {seed} trustworthiness was nonfinite")
        values.append({"seed": int(seed), "value": value})
    scalars = np.asarray([item["value"] for item in values], dtype=float)
    return {
        "per_seed": values,
        "mean": float(scalars.mean()),
        "std": float(scalars.std(ddof=0)),
        "count": int(scalars.size),
        "n_neighbors": int(k),
        "sample_size": int(indices.size),
        "reason": None,
    }


def _neighbor_overlap_summary(coords_by_seed: Mapping[int, np.ndarray]) -> dict:
    """Mean k-neighbor overlap between the display-seed embedding and every other seed."""
    from sklearn.neighbors import NearestNeighbors

    seeds = list(coords_by_seed)
    if len(seeds) < 2:
        return {"reference_seed": (int(seeds[0]) if seeds else None), "per_seed": [],
                "mean": None, "std": None, "count": 0,
                "reason": "fewer than two projection seeds"}
    first = np.asarray(coords_by_seed[seeds[0]], dtype=float)
    indices = _diagnostic_indices(first.shape[0])
    if indices.size < 2:
        return {"reference_seed": int(seeds[0]), "per_seed": [], "mean": None,
                "std": None, "count": 0, "reason": "fewer than two samples"}
    k = min(DIAGNOSTIC_N_NEIGHBORS, indices.size - 1)

    def neighbor_indices(coordinates: np.ndarray) -> np.ndarray:
        subset = np.asarray(coordinates, dtype=float)[indices]
        return NearestNeighbors(n_neighbors=k + 1).fit(subset).kneighbors(
            subset,
            return_distance=False,
        )[:, 1:]

    reference = neighbor_indices(first)
    values = []
    for seed in seeds[1:]:
        current = neighbor_indices(np.asarray(coords_by_seed[seed], dtype=float))
        overlap = np.mean([
            len(set(left.tolist()).intersection(right.tolist())) / k
            for left, right in zip(reference, current)
        ])
        values.append({"seed": int(seed), "value": float(overlap)})
    scalars = np.asarray([item["value"] for item in values], dtype=float)
    return {
        "reference_seed": int(seeds[0]),
        "per_seed": values,
        "mean": float(scalars.mean()),
        "std": float(scalars.std(ddof=0)),
        "count": int(scalars.size),
        "n_neighbors": int(k),
        "sample_size": int(indices.size),
        "reason": None,
    }


def _adjusted_mutual_information(
    cluster_labels: np.ndarray,
    target_labels:  Optional[object],
) -> dict:
    """AMI after excluding HDBSCAN noise, with degenerate cases made explicit."""
    if target_labels is None:
        return _null_metric("labels unavailable")
    from sklearn.metrics import adjusted_mutual_info_score

    target = _as_numpy(target_labels)
    if target.shape[0] != cluster_labels.shape[0]:
        raise ValueError("AMI labels are not aligned with cluster labels")
    keep = cluster_labels != -1
    if int(keep.sum()) < 2:
        return _null_metric("fewer than two non-noise samples")
    clusters = cluster_labels[keep]
    target = target[keep]
    if np.unique(clusters).size < 2:
        return _null_metric("non-noise cluster labels are constant")
    if np.unique(target).size < 2:
        return _null_metric("target labels are constant after excluding noise")
    return _finite_metric(float(adjusted_mutual_info_score(clusters, target)))


def controlled_embedding_record(
    features:                    np.ndarray,
    coords_by_seed:              Mapping[int, np.ndarray],
    cluster_labels:              object,
    token_ids:                   object,
    bpe_labels:                  Optional[object],
    function_content_labels:     Optional[object],
    seq_idx:                     object,
    pos_idx:                     object,
    seq_len:                     int,
    contract:                    Mapping[str, object],
    taxonomy_unavailable_reason: Optional[str] = None,
    semantic_probes:             Optional[Mapping[str, object]] = None,
) -> dict:
    """Compute the machine-readable controlled diagnostics for one channel and population."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("controlled features must be a finite 2-D matrix")
    token_array = _as_numpy(token_ids)
    sequence_array = _as_numpy(seq_idx)
    position_array = _as_numpy(pos_idx)
    clusters = _as_numpy(cluster_labels).astype(np.int64, copy=False)
    for name, array in (("token_ids", token_array), ("seq_idx", sequence_array),
                        ("pos_idx", position_array), ("cluster_labels", clusters)):
        if array.shape[0] != values.shape[0]:
            raise ValueError(f"{name} is not aligned with features")
    if not coords_by_seed:
        raise ValueError("coords_by_seed must contain at least one projection")

    record = copy.deepcopy(dict(contract))
    expected_seeds = [int(seed) for seed in record["display"]["seeds"]]
    actual_seeds = [int(seed) for seed in coords_by_seed]
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"projection seeds {actual_seeds} do not match controlled contract {expected_seeds}"
        )
    record["sample"] = {
        "token_count": int(values.shape[0]),
        "sequence_count": int(np.unique(sequence_array).size),
        "sequence_length": int(seq_len),
        "token_sha256": token_fingerprint(token_array),
    }
    bpe_silhouette = (
        _null_metric(taxonomy_unavailable_reason)
        if bpe_labels is None and taxonomy_unavailable_reason is not None
        else _native_silhouette(values, bpe_labels)
    )
    function_content_silhouette = (
        _null_metric(taxonomy_unavailable_reason)
        if function_content_labels is None and taxonomy_unavailable_reason is not None
        else _native_silhouette(values, function_content_labels)
    )
    record["native_space"] = {
        "silhouette": {
            "bpe": bpe_silhouette,
            "function_content": function_content_silhouette,
        }
    }
    record["projection"] = {
        "trustworthiness": _trustworthiness_summary(values, coords_by_seed),
        "neighbor_overlap": _neighbor_overlap_summary(coords_by_seed),
    }
    position_quartiles = relative_position_quartiles(position_array, seq_len=seq_len)
    non_noise = clusters[clusters != -1]
    bpe_ami = (
        _null_metric(taxonomy_unavailable_reason)
        if bpe_labels is None and taxonomy_unavailable_reason is not None
        else _adjusted_mutual_information(clusters, bpe_labels)
    )
    function_content_ami = (
        _null_metric(taxonomy_unavailable_reason)
        if function_content_labels is None and taxonomy_unavailable_reason is not None
        else _adjusted_mutual_information(clusters, function_content_labels)
    )
    record["clusters"] = {
        "count": int(np.unique(non_noise).size),
        "noise_fraction": float(np.mean(clusters == -1)),
        "adjusted_mutual_information": {
            "bpe": bpe_ami,
            "function_content": function_content_ami,
            "position_quartile": _adjusted_mutual_information(clusters, position_quartiles),
            "sequence_identity": _adjusted_mutual_information(clusters, sequence_array),
        },
    }
    if semantic_probes is None:
        from vfe3.viz import semantic_probes as semantic_probe_metrics

        record["semantic_probes"] = semantic_probe_metrics.unavailable_record(
            "semantic probes not provided"
        )
    else:
        record["semantic_probes"] = copy.deepcopy(dict(semantic_probes))
    return record


_COMPARISON_FIELDS = (
    "schema_version",
    "mode",
    "sample.token_count",
    "sample.sequence_count",
    "sample.sequence_length",
    "sample.token_sha256",
    "channel.kind",
    "channel.name",
    "channel.feature_dim",
    "channel.feature_chart",
    "display.method",
    "display.n_neighbors",
    "display.min_dist",
    "display.n_components",
    "display.init",
    "display.seeds",
    "display.display_seed",
    "clustering.algorithm",
    "clustering.space",
    "clustering.min_cluster_size",
    "clustering.min_samples",
    "clustering.cluster_selection_method",
    "clustering.cluster_selection_epsilon",
    "semantic_probes.manifest.name",
    "semantic_probes.manifest.schema_version",
)


def _dotted_value(record: Mapping[str, object], field: str) -> object:
    """Read a required dotted field or raise a comparison-contract error naming it."""
    value: object = record
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"controlled comparison record is missing {field}")
        value = value[part]
    return value


def validate_comparison_records(records: Sequence[Mapping[str, object]]) -> None:
    """Fail closed unless every sidecar describes the same controlled comparison contract."""
    if len(records) < 2:
        raise ValueError("controlled comparison needs at least two sidecar records")
    reference = records[0]
    mismatches = []
    for index, record in enumerate(records[1:], start=1):
        for field in _COMPARISON_FIELDS:
            expected = _dotted_value(reference, field)
            actual = _dotted_value(record, field)
            if actual != expected:
                mismatches.append(
                    f"record[{index}].{field} ({actual!r} != {expected!r})"
                )
    if mismatches:
        raise ValueError("controlled comparison contract mismatch: " + "; ".join(mismatches))


def _metric_scalar(record: Mapping[str, object], field: str) -> Optional[float]:
    """Extract a scalar or metric-record value for the compact cross-run artifact."""
    value = _dotted_value(record, field)
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return None
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def comparison_summary(
    records: Sequence[Mapping[str, object]],
    labels:  Sequence[str],
) -> dict:
    """Build the metric-only cross-run summary after validating all controlled contracts."""
    validate_comparison_records(records)
    if len(labels) != len(records):
        raise ValueError("labels must contain one entry per sidecar record")
    if len(set(labels)) != len(labels):
        raise ValueError("comparison labels must be unique")
    arms = []
    for label, record in zip(labels, records):
        arms.append({
            "label": str(label),
            "sample": copy.deepcopy(record["sample"]),
            "metrics": {
                "native_silhouette_bpe": _metric_scalar(
                    record, "native_space.silhouette.bpe"
                ),
                "native_silhouette_function_content": _metric_scalar(
                    record, "native_space.silhouette.function_content"
                ),
                "trustworthiness": _metric_scalar(
                    record, "projection.trustworthiness.mean"
                ),
                "neighbor_overlap": _metric_scalar(
                    record, "projection.neighbor_overlap.mean"
                ),
                "cluster_count": _metric_scalar(record, "clusters.count"),
                "noise_fraction": _metric_scalar(record, "clusters.noise_fraction"),
                "ami_bpe": _metric_scalar(
                    record, "clusters.adjusted_mutual_information.bpe"
                ),
                "ami_function_content": _metric_scalar(
                    record, "clusters.adjusted_mutual_information.function_content"
                ),
                "ami_position_quartile": _metric_scalar(
                    record, "clusters.adjusted_mutual_information.position_quartile"
                ),
                "ami_sequence_identity": _metric_scalar(
                    record, "clusters.adjusted_mutual_information.sequence_identity"
                ),
                "semantic_field_silhouette": _metric_scalar(
                    record, "semantic_probes.native_space.semantic_field_silhouette"
                ),
                "semantic_field_ami": _metric_scalar(
                    record,
                    "semantic_probes.clusters.semantic_field_adjusted_mutual_information",
                ),
                "close_pair_mean_distance_percentile": _metric_scalar(
                    record,
                    "semantic_probes.aggregate.expectations.close.mean_distance_percentile",
                ),
                "close_pair_mean_reciprocal_rank": _metric_scalar(
                    record,
                    "semantic_probes.aggregate.expectations.close.mean_reciprocal_rank",
                ),
                "close_pair_hit_at_5_rate": _metric_scalar(
                    record,
                    "semantic_probes.aggregate.expectations.close.hit_at_5_rate",
                ),
                "close_pair_mean_hdbscan_co_membership": _metric_scalar(
                    record,
                    "semantic_probes.aggregate.expectations.close.mean_hdbscan_co_membership",
                ),
                "control_to_close_distance_ratio": _metric_scalar(
                    record,
                    "semantic_probes.aggregate.control_to_close_distance_ratio",
                ),
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "controlled_belief_geometry_comparison",
        "comparison_basis": (
            "Within-run scalar diagnostics only; independently fitted UMAP axes are not shared."
        ),
        "coordinate_status": records[0]["coordinate_status"],
        "validated_contract": {
            field: copy.deepcopy(_dotted_value(records[0], field))
            for field in _COMPARISON_FIELDS
        },
        "arms": arms,
    }


def write_json_atomic(
    record: Mapping[str, object],
    path:   'str | Path',
) -> Path:
    """Serialize one strict JSON record through a same-directory atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return destination
