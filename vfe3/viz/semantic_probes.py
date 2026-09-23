"""Preregistered semantic probes for controlled native-chart diagnostics.

The evaluator never consumes UMAP coordinates. Concept centroids and distances
are computed in the feature chart supplied by the caller; HDBSCAN labels add a
separate, noise-aware co-membership diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Optional

import numpy as np


SEMANTIC_RECORD_SCHEMA_VERSION = 1
_EXPECTATIONS = ("close", "control", "descriptive")


DEFAULT_SEMANTIC_MANIFEST: dict[str, object] = {
    "name": "english_gpt2_semantic_probes",
    "schema_version": 1,
    "concepts": [
        {"name": "king", "forms": [" king", "king"]},
        {"name": "queen", "forms": [" queen"]},
        {"name": "prince", "forms": [" prince"]},
        {"name": "princess", "forms": [" princess"]},
        {"name": "father", "forms": [" father", "father"]},
        {"name": "mother", "forms": [" mother", "mother"]},
        {"name": "son", "forms": [" son", "son"]},
        {"name": "daughter", "forms": [" daughter", "daughter"]},
        {"name": "dog", "forms": [" dog", "dog"]},
        {"name": "cat", "forms": [" cat", "cat"]},
        {"name": "horse", "forms": [" horse", "horse"]},
        {"name": "bird", "forms": [" bird", "bird"]},
        {"name": "red", "forms": [" red", "red"]},
        {"name": "blue", "forms": [" blue", "blue"]},
        {"name": "green", "forms": [" green", "green"]},
        {"name": "yellow", "forms": [" yellow", "yellow"]},
        {"name": "above", "forms": [" above", "above"]},
        {"name": "below", "forms": [" below", "below"]},
        {"name": "inside", "forms": [" inside", "inside"]},
        {"name": "outside", "forms": [" outside", "outside"]},
        {"name": "happy", "forms": [" happy", "happy"]},
        {"name": "sad", "forms": [" sad"]},
        {"name": "angry", "forms": [" angry"]},
        {"name": "afraid", "forms": [" afraid"]},
        {"name": "run", "forms": [" run", "run"]},
        {"name": "walk", "forms": [" walk", "walk"]},
        {"name": "jump", "forms": [" jump", "jump"]},
        {"name": "fly", "forms": [" fly", "fly"]},
        {"name": "think", "forms": [" think", "think"]},
        {"name": "know", "forms": [" know", "know"]},
        {"name": "believe", "forms": [" believe"]},
        {"name": "remember", "forms": [" remember", "remember"]},
    ],
    "fields": {
        "royalty": ["king", "queen", "prince", "princess"],
        "kinship": ["father", "mother", "son", "daughter"],
        "animals": ["dog", "cat", "horse", "bird"],
        "colors": ["red", "blue", "green", "yellow"],
        "spatial": ["above", "below", "inside", "outside"],
        "emotions": ["happy", "sad", "angry", "afraid"],
        "motion_verbs": ["run", "walk", "jump", "fly"],
        "cognition_verbs": ["think", "know", "believe", "remember"],
    },
    "pairs": [
        {"name": "king_queen", "concept_a": "king", "concept_b": "queen",
         "expectation": "close", "relation_type": "royalty association"},
        {"name": "prince_princess", "concept_a": "prince", "concept_b": "princess",
         "expectation": "close", "relation_type": "royalty association"},
        {"name": "father_mother", "concept_a": "father", "concept_b": "mother",
         "expectation": "close", "relation_type": "kinship association"},
        {"name": "son_daughter", "concept_a": "son", "concept_b": "daughter",
         "expectation": "close", "relation_type": "kinship association"},
        {"name": "dog_cat", "concept_a": "dog", "concept_b": "cat",
         "expectation": "close", "relation_type": "category association"},
        {"name": "horse_bird", "concept_a": "horse", "concept_b": "bird",
         "expectation": "close", "relation_type": "category association"},
        {"name": "red_blue", "concept_a": "red", "concept_b": "blue",
         "expectation": "close", "relation_type": "category association"},
        {"name": "green_yellow", "concept_a": "green", "concept_b": "yellow",
         "expectation": "close", "relation_type": "category association"},
        {"name": "above_below", "concept_a": "above", "concept_b": "below",
         "expectation": "close", "relation_type": "spatial antonymy"},
        {"name": "inside_outside", "concept_a": "inside", "concept_b": "outside",
         "expectation": "close", "relation_type": "spatial antonymy"},
        {"name": "happy_sad", "concept_a": "happy", "concept_b": "sad",
         "expectation": "close", "relation_type": "emotion antonymy"},
        {"name": "angry_afraid", "concept_a": "angry", "concept_b": "afraid",
         "expectation": "close", "relation_type": "emotion association"},
        {"name": "run_walk", "concept_a": "run", "concept_b": "walk",
         "expectation": "close", "relation_type": "motion association"},
        {"name": "jump_fly", "concept_a": "jump", "concept_b": "fly",
         "expectation": "close", "relation_type": "motion association"},
        {"name": "think_know", "concept_a": "think", "concept_b": "know",
         "expectation": "close", "relation_type": "cognition association"},
        {"name": "believe_remember", "concept_a": "believe", "concept_b": "remember",
         "expectation": "close", "relation_type": "cognition association"},
        {"name": "king_jump", "concept_a": "king", "concept_b": "jump",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "father_fly", "concept_a": "father", "concept_b": "fly",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "dog_believe", "concept_a": "dog", "concept_b": "believe",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "red_mother", "concept_a": "red", "concept_b": "mother",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "above_happy", "concept_a": "above", "concept_b": "happy",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "sad_horse", "concept_a": "sad", "concept_b": "horse",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "run_princess", "concept_a": "run", "concept_b": "princess",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "think_yellow", "concept_a": "think", "concept_b": "yellow",
         "expectation": "control", "relation_type": "cross-field control"},
        {"name": "king_father", "concept_a": "king", "concept_b": "father",
         "expectation": "descriptive", "relation_type": "male social-role association"},
        {"name": "queen_mother", "concept_a": "queen", "concept_b": "mother",
         "expectation": "descriptive", "relation_type": "female social-role association"},
        {"name": "prince_son", "concept_a": "prince", "concept_b": "son",
         "expectation": "descriptive", "relation_type": "male descendant association"},
        {"name": "princess_daughter", "concept_a": "princess", "concept_b": "daughter",
         "expectation": "descriptive", "relation_type": "female descendant association"},
    ],
}


def _as_numpy(values: object) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _metric(
    value:  Optional[float],
    reason: Optional[str] = None,
) -> dict[str, object]:
    if value is None:
        return {"value": None, "reason": reason or "metric unavailable"}
    scalar = float(value)
    if not np.isfinite(scalar):
        return {"value": None, "reason": reason or "metric is not finite"}
    return {"value": scalar, "reason": None}


def _manifest_parts(
    manifest: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], Mapping[str, Sequence[str]], list[Mapping[str, object]]]:
    concepts = manifest.get("concepts")
    fields = manifest.get("fields")
    pairs = manifest.get("pairs")
    if not isinstance(concepts, Sequence) or isinstance(concepts, (str, bytes)):
        raise ValueError("manifest concepts must be a sequence")
    if not isinstance(fields, Mapping):
        raise ValueError("manifest fields must be a mapping")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ValueError("manifest pairs must be a sequence")
    if not all(isinstance(item, Mapping) for item in concepts):
        raise ValueError("every manifest concept must be a mapping")
    if not all(isinstance(item, Mapping) for item in pairs):
        raise ValueError("every manifest pair must be a mapping")
    return list(concepts), fields, list(pairs)


def validate_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the semantic manifest before token resolution or scoring."""
    name = manifest.get("name")
    schema_version = manifest.get("schema_version")
    if not isinstance(name, str) or not name:
        raise ValueError("manifest name must be a non-empty string")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("manifest schema_version must be a positive integer")

    concepts, fields, pairs = _manifest_parts(manifest)
    concept_names: list[str] = []
    accepted_forms: set[str] = set()
    for concept in concepts:
        concept_name = concept.get("name")
        forms = concept.get("forms")
        if not isinstance(concept_name, str) or not concept_name:
            raise ValueError("concept name must be a non-empty string")
        if concept_name in concept_names:
            raise ValueError(f"duplicate concept name: {concept_name}")
        if not isinstance(forms, Sequence) or isinstance(forms, (str, bytes)) or not forms:
            raise ValueError(f"concept {concept_name} forms must be a non-empty sequence")
        if not all(isinstance(form, str) and form for form in forms):
            raise ValueError(f"concept {concept_name} forms must be non-empty strings")
        if len(set(forms)) != len(forms):
            raise ValueError(f"concept {concept_name} has duplicate accepted forms")
        overlap = accepted_forms.intersection(forms)
        if overlap:
            duplicate = sorted(overlap)[0]
            raise ValueError(f"accepted token form belongs to multiple concepts: {duplicate!r}")
        accepted_forms.update(forms)
        concept_names.append(concept_name)

    known = set(concept_names)
    membership: dict[str, str] = {}
    for field_name, members in fields.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("field name must be a non-empty string")
        if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
            raise ValueError(f"field {field_name} members must be a sequence")
        for concept_name in members:
            if concept_name not in known:
                raise ValueError(f"field {field_name} references unknown concept: {concept_name}")
            if concept_name in membership:
                raise ValueError(
                    f"concept {concept_name} appears in multiple fields: "
                    f"{membership[concept_name]} and {field_name}"
                )
            membership[concept_name] = field_name
    unassigned = sorted(known.difference(membership))
    if unassigned:
        raise ValueError(f"concept has no field membership: {unassigned[0]}")

    pair_names: set[str] = set()
    for pair in pairs:
        pair_name = pair.get("name")
        concept_a = pair.get("concept_a")
        concept_b = pair.get("concept_b")
        expectation = pair.get("expectation")
        if not isinstance(pair_name, str) or not pair_name:
            raise ValueError("pair name must be a non-empty string")
        if pair_name in pair_names:
            raise ValueError(f"duplicate pair name: {pair_name}")
        pair_names.add(pair_name)
        if concept_a not in known:
            raise ValueError(f"pair {pair_name} references unknown concept: {concept_a}")
        if concept_b not in known:
            raise ValueError(f"pair {pair_name} references unknown concept: {concept_b}")
        if concept_a == concept_b:
            raise ValueError(f"pair {pair_name} must reference distinct concepts")
        if expectation not in _EXPECTATIONS:
            raise ValueError(f"pair {pair_name} has invalid expectation: {expectation}")


def _manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": str(manifest["name"]),
        "schema_version": int(manifest["schema_version"]),
    }


def _null_expectation_summary(reason: str) -> dict[str, object]:
    return {
        "declared_pair_count": 0,
        "resolved_pair_count": 0,
        "mean_centroid_distance": _metric(None, reason),
        "mean_distance_percentile": _metric(None, reason),
        "mean_reciprocal_rank": _metric(None, reason),
        "hit_at_5_rate": _metric(None, reason),
        "mean_hdbscan_co_membership": _metric(None, reason),
    }


def unavailable_record(
    reason: str,

    *,
    manifest: Mapping[str, object] = DEFAULT_SEMANTIC_MANIFEST,
) -> dict[str, object]:
    """Return the stable semantic schema when evaluation is not applicable."""
    validate_manifest(manifest)
    return {
        "schema_version": SEMANTIC_RECORD_SCHEMA_VERSION,
        "manifest": _manifest_identity(manifest),
        "available": False,
        "reason": reason,
        "resolution": {
            "resolved_concept_count": 0,
            "total_concept_count": len(manifest["concepts"]),
            "concepts": {},
        },
        "native_space": {
            "metric": "Euclidean distance in the supplied feature chart",
            "concept_centroids": {},
            "fields": {},
            "semantic_field_silhouette": _metric(None, reason),
        },
        "clusters": {
            "noise_label": -1,
            "semantic_field_adjusted_mutual_information": _metric(None, reason),
        },
        "pairs": {},
        "aggregate": {
            "expectations": {
                expectation: _null_expectation_summary(reason)
                for expectation in _EXPECTATIONS
            },
            "control_to_close_distance_ratio": _metric(None, reason),
        },
    }


def resolve_concepts(
    token_ids: object,
    decode:    Callable[[list[int]], str],

    *,
    manifest: Mapping[str, object] = DEFAULT_SEMANTIC_MANIFEST,
) -> dict[str, object]:
    """Resolve exact decoded single-token forms against one contextual bank."""
    validate_manifest(manifest)
    tokens = _as_numpy(token_ids).reshape(-1)
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("semantic probe token_ids must be integers")
    tokens = tokens.astype(np.int64, copy=False)

    decoded_by_id = {
        int(token_id): str(decode([int(token_id)]))
        for token_id in sorted(np.unique(tokens).tolist())
    }
    concepts, _, _ = _manifest_parts(manifest)
    records: dict[str, object] = {}
    resolved_count = 0
    for concept in concepts:
        concept_name = str(concept["name"])
        forms = [str(form) for form in concept["forms"]]
        matches = [
            token_id for token_id, decoded in decoded_by_id.items()
            if decoded in forms
        ]
        indices = np.flatnonzero(np.isin(tokens, matches)).astype(int).tolist()
        decoded_forms = [decoded_by_id[token_id] for token_id in matches]
        resolved = bool(indices)
        if resolved:
            resolved_count += 1
        records[concept_name] = {
            "accepted_forms": forms,
            "resolved": resolved,
            "reason": None if resolved else "no accepted token form occurs in the bank",
            "token_ids": matches,
            "decoded_forms": decoded_forms,
            "occurrence_indices": indices,
            "occurrence_count": len(indices),
        }
    return {
        "resolved_concept_count": resolved_count,
        "total_concept_count": len(concepts),
        "concepts": records,
    }


def _mean_metric(values: Sequence[float], reason: str) -> dict[str, object]:
    if not values:
        return _metric(None, reason)
    return _metric(float(np.mean(np.asarray(values, dtype=float))))


def _pair_distance(
    centroids: Mapping[str, np.ndarray],
    concept_a: str,
    concept_b: str,
) -> float:
    return float(np.linalg.norm(centroids[concept_a] - centroids[concept_b]))


def _partner_rank(
    centroids: Mapping[str, np.ndarray],
    source:    str,
    target:    str,
) -> float:
    target_distance = _pair_distance(centroids, source, target)
    competitor_distances = [
        _pair_distance(centroids, source, other)
        for other in centroids
        if other != source
    ]
    strictly_closer = sum(
        distance < target_distance and not np.isclose(distance, target_distance)
        for distance in competitor_distances
    )
    return float(1 + strictly_closer)


def _distance_percentile(
    distance:      float,
    all_distances: Sequence[float],
) -> float:
    values = np.asarray(all_distances, dtype=float)
    below_or_tied = np.logical_or(values < distance, np.isclose(values, distance))
    return float(100.0 * below_or_tied.mean())


def _co_membership(
    cluster_labels: np.ndarray,
    indices_a:     Sequence[int],
    indices_b:     Sequence[int],
) -> float:
    labels_a = cluster_labels[np.asarray(indices_a, dtype=int)]
    labels_b = cluster_labels[np.asarray(indices_b, dtype=int)]
    total_a = float(labels_a.size)
    total_b = float(labels_b.size)
    clusters = set(labels_a.tolist()).union(labels_b.tolist()).difference({-1})
    return float(sum(
        float(np.count_nonzero(labels_a == cluster)) / total_a
        * float(np.count_nonzero(labels_b == cluster)) / total_b
        for cluster in clusters
    ))


def _field_records(
    manifest:   Mapping[str, object],
    resolution: Mapping[str, object],
    centroids:  Mapping[str, np.ndarray],
) -> dict[str, object]:
    _, fields, _ = _manifest_parts(manifest)
    concept_records = resolution["concepts"]
    output: dict[str, object] = {}
    for field_name, members in fields.items():
        resolved = [str(name) for name in members if str(name) in centroids]
        within = [
            _pair_distance(centroids, left, right)
            for left, right in combinations(resolved, 2)
        ]
        other = [name for name in centroids if name not in set(resolved)]
        between = [
            _pair_distance(centroids, member, outside)
            for member in resolved
            for outside in other
        ]
        within_metric = _mean_metric(within, "fewer than two resolved concepts in field")
        between_metric = _mean_metric(between, "no resolved concepts outside field")
        within_value = within_metric["value"]
        between_value = between_metric["value"]
        if within_value is None or between_value is None:
            ratio = _metric(None, "within-field or between-field distance unavailable")
        elif float(within_value) <= 0.0:
            ratio = _metric(None, "mean within-field distance is zero")
        else:
            ratio = _metric(float(between_value) / float(within_value))
        output[str(field_name)] = {
            "resolved_concept_count": len(resolved),
            "occurrence_count": sum(
                int(concept_records[name]["occurrence_count"])
                for name in resolved
            ),
            "resolved_concepts": resolved,
            "mean_within_distance": within_metric,
            "mean_between_distance": between_metric,
            "between_to_within_ratio": ratio,
        }
    return output


def _semantic_silhouette(
    manifest:  Mapping[str, object],
    centroids: Mapping[str, np.ndarray],
) -> dict[str, object]:
    _, fields, _ = _manifest_parts(manifest)
    field_by_concept = {
        str(concept): str(field_name)
        for field_name, members in fields.items()
        for concept in members
    }
    names = list(centroids)
    labels = np.asarray([field_by_concept[name] for name in names], dtype=object)
    if len(names) < 3:
        return _metric(None, "fewer than three resolved concepts")
    n_fields = np.unique(labels).size
    if n_fields < 2:
        return _metric(None, "fewer than two resolved semantic fields")
    if n_fields >= len(names):
        return _metric(None, "every resolved concept has a distinct semantic field")
    from sklearn.metrics import silhouette_score

    matrix = np.stack([centroids[name] for name in names], axis=0)
    return _metric(float(silhouette_score(matrix, labels, metric="euclidean")))


def _semantic_ami(
    cluster_labels: np.ndarray,
    manifest:       Mapping[str, object],
    resolution:     Mapping[str, object],
) -> dict[str, object]:
    _, fields, _ = _manifest_parts(manifest)
    concept_records = resolution["concepts"]
    occurrence_rows: list[int] = []
    semantic_labels: list[str] = []
    for field_name, members in fields.items():
        for concept_name in members:
            rows = concept_records[str(concept_name)]["occurrence_indices"]
            occurrence_rows.extend(int(row) for row in rows)
            semantic_labels.extend([str(field_name)] * len(rows))
    if len(occurrence_rows) < 2:
        return _metric(None, "fewer than two resolved semantic occurrences")
    rows = np.asarray(occurrence_rows, dtype=int)
    clusters = cluster_labels[rows]
    semantic = np.asarray(semantic_labels, dtype=object)
    keep = clusters != -1
    if int(keep.sum()) < 2:
        return _metric(None, "fewer than two non-noise semantic occurrences")
    clusters = clusters[keep]
    semantic = semantic[keep]
    if np.unique(clusters).size < 2:
        return _metric(None, "non-noise cluster labels are constant")
    if np.unique(semantic).size < 2:
        return _metric(None, "semantic field labels are constant after excluding noise")
    from sklearn.metrics import adjusted_mutual_info_score

    return _metric(float(adjusted_mutual_info_score(semantic, clusters)))


def _expectation_summary(
    pair_records: Mapping[str, Mapping[str, object]],
    expectation: str,
) -> dict[str, object]:
    selected = [
        record for record in pair_records.values()
        if record["expectation"] == expectation
    ]
    resolved = [record for record in selected if record["resolved"]]

    def values(key: str) -> list[float]:
        return [
            float(record[key]["value"])
            for record in resolved
            if record[key]["value"] is not None
        ]

    reason = f"no resolved {expectation} pairs"
    return {
        "declared_pair_count": len(selected),
        "resolved_pair_count": len(resolved),
        "mean_centroid_distance": _mean_metric(values("centroid_distance"), reason),
        "mean_distance_percentile": _mean_metric(values("distance_percentile"), reason),
        "mean_reciprocal_rank": _mean_metric(values("mean_reciprocal_rank"), reason),
        "hit_at_5_rate": _mean_metric(values("hit_at_5"), reason),
        "mean_hdbscan_co_membership": _mean_metric(values("hdbscan_co_membership"), reason),
    }


def evaluate_semantic_probes(
    features:       object,
    token_ids:      object,
    cluster_labels: object,
    decode:         Callable[[list[int]], str],

    *,
    manifest: Mapping[str, object] = DEFAULT_SEMANTIC_MANIFEST,
) -> dict[str, object]:
    """Evaluate preregistered concepts in a native feature chart.

    For concept occurrence set I_c and native vectors z_i, the concept centroid is
    m_c = |I_c|^{-1} sum_{i in I_c} z_i. Pair distance is ||m_a - m_b||_2.
    HDBSCAN co-membership is sum_{g != -1} p_a(g) p_b(g), where each p_c uses
    all occurrences in its denominator so noise reduces, but never creates,
    shared-cluster probability.
    """
    validate_manifest(manifest)
    matrix = _as_numpy(features).astype(float, copy=False)
    tokens = _as_numpy(token_ids).reshape(-1)
    labels = _as_numpy(cluster_labels).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("semantic probe features must be a two-dimensional matrix")
    if matrix.shape[0] != tokens.size or matrix.shape[0] != labels.size:
        raise ValueError("semantic probe features, token_ids, and cluster_labels must align")
    if not np.isfinite(matrix).all():
        raise ValueError("semantic probe features must be finite")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("semantic probe cluster_labels must be integers")
    labels = labels.astype(int, copy=False)

    resolution = resolve_concepts(tokens, decode, manifest=manifest)
    concept_records = resolution["concepts"]
    centroids = {
        concept_name: matrix[np.asarray(record["occurrence_indices"], dtype=int)].mean(axis=0)
        for concept_name, record in concept_records.items()
        if record["resolved"]
    }
    all_distances = [
        _pair_distance(centroids, left, right)
        for left, right in combinations(centroids, 2)
    ]

    _, _, pairs = _manifest_parts(manifest)
    pair_records: dict[str, dict[str, object]] = {}
    for pair in pairs:
        pair_name = str(pair["name"])
        concept_a = str(pair["concept_a"])
        concept_b = str(pair["concept_b"])
        missing = [name for name in (concept_a, concept_b) if name not in centroids]
        base: dict[str, object] = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "expectation": str(pair["expectation"]),
            "relation_type": str(pair.get("relation_type", "unspecified")),
            "resolved": not missing,
        }
        if missing:
            reason = f"unresolved concept: {missing[0]}"
            base.update({
                "centroid_distance": _metric(None, reason),
                "distance_percentile": _metric(None, reason),
                "rank_a_to_b": _metric(None, reason),
                "rank_b_to_a": _metric(None, reason),
                "mean_reciprocal_rank": _metric(None, reason),
                "hit_at_5": _metric(None, reason),
                "hdbscan_co_membership": _metric(None, reason),
            })
        else:
            distance = _pair_distance(centroids, concept_a, concept_b)
            rank_a = _partner_rank(centroids, concept_a, concept_b)
            rank_b = _partner_rank(centroids, concept_b, concept_a)
            indices_a = concept_records[concept_a]["occurrence_indices"]
            indices_b = concept_records[concept_b]["occurrence_indices"]
            base.update({
                "centroid_distance": _metric(distance),
                "distance_percentile": _metric(_distance_percentile(distance, all_distances)),
                "rank_a_to_b": _metric(rank_a),
                "rank_b_to_a": _metric(rank_b),
                "mean_reciprocal_rank": _metric(0.5 * (1.0 / rank_a + 1.0 / rank_b)),
                "hit_at_5": _metric(0.5 * (float(rank_a <= 5) + float(rank_b <= 5))),
                "hdbscan_co_membership": _metric(
                    _co_membership(labels, indices_a, indices_b)
                ),
            })
        pair_records[pair_name] = base

    expectation_summaries = {
        expectation: _expectation_summary(pair_records, expectation)
        for expectation in _EXPECTATIONS
    }
    close_distance = expectation_summaries["close"]["mean_centroid_distance"]["value"]
    control_distance = expectation_summaries["control"]["mean_centroid_distance"]["value"]
    if close_distance is None or control_distance is None:
        separation = _metric(None, "close or control mean distance unavailable")
    elif float(close_distance) <= 0.0:
        separation = _metric(None, "close-pair mean distance is zero")
    else:
        separation = _metric(float(control_distance) / float(close_distance))

    return {
        "schema_version": SEMANTIC_RECORD_SCHEMA_VERSION,
        "manifest": _manifest_identity(manifest),
        "available": True,
        "reason": None,
        "resolution": resolution,
        "native_space": {
            "metric": "Euclidean distance in the supplied feature chart",
            "concept_centroids": {
                name: centroid.astype(float).tolist()
                for name, centroid in centroids.items()
            },
            "fields": _field_records(manifest, resolution, centroids),
            "semantic_field_silhouette": _semantic_silhouette(manifest, centroids),
        },
        "clusters": {
            "noise_label": -1,
            "semantic_field_adjusted_mutual_information": _semantic_ami(
                labels,
                manifest,
                resolution,
            ),
        },
        "pairs": pair_records,
        "aggregate": {
            "expectations": expectation_summaries,
            "control_to_close_distance_ratio": separation,
        },
    }
