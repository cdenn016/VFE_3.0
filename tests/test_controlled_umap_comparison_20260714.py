"""Controlled belief-geometry comparison regressions (2026-07-14)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz import extract, report


def _tiny_model() -> VFEModel:
    cfg = VFE3Config(
        vocab_size=20,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_phi_lr=0.0,
    )
    torch.manual_seed(0)
    return VFEModel(cfg)


def _token_batches() -> list[torch.Tensor]:
    return [
        torch.arange(0, 8).reshape(2, 4),
        torch.arange(8, 16).reshape(2, 4),
    ]


def test_belief_bank_max_tokens_slices_every_aligned_field():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_tokens=11)

    aligned = ("mu", "sigma", "phi", "token_ids", "seq_idx", "pos_idx")
    assert {bank[key].shape[0] for key in aligned} == {11}
    assert bank["token_ids"].tolist() == list(range(11))
    assert bank["seq_idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]


def test_belief_bank_max_sequences_is_exact_and_position_aligned():
    bank = extract.belief_bank(_tiny_model(), _token_batches(), max_sequences=3)

    assert bank["mu"].shape[0] == 12
    assert bank["seq_idx"].tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert bank["pos_idx"].tolist() == [0, 1, 2, 3] * 3


def test_belief_bank_rejects_ambiguous_population_caps():
    model = _tiny_model()

    with pytest.raises(ValueError, match="max_tokens.*max_sequences"):
        extract.belief_bank(model, _token_batches(), max_tokens=8, max_sequences=2)


@pytest.mark.parametrize("cap_name", ["max_tokens", "max_sequences"])
@pytest.mark.parametrize("cap_value", [0, -1])
def test_belief_bank_rejects_nonpositive_population_caps(cap_name, cap_value):
    with pytest.raises(ValueError, match=cap_name):
        extract.belief_bank(_tiny_model(), _token_batches(), **{cap_name: cap_value})


@pytest.mark.parametrize(
    ("seq_len", "batch_size", "expected_batches"),
    [(128, 64, 2), (256, 32, 2), (512, 16, 2)],
)
def test_report_default_requests_same_controlled_token_population(
    seq_len,
    batch_size,
    expected_batches,
):
    max_tokens, max_sequences = report._resolve_bank_budget(
        max_tokens=None,
        max_sequences=None,
    )
    batches = [torch.zeros((batch_size, seq_len), dtype=torch.long) for _ in range(3)]
    collected = report._collect_batches(
        batches,
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )

    assert max_tokens == 16_384
    assert max_sequences is None
    assert len(collected) == expected_batches


def test_cluster_coordinates_use_pca_not_umap_above_ten_dimensions():
    from vfe3.viz import embedding_comparison

    features = np.random.default_rng(0).normal(size=(80, 20))
    coordinates, description = embedding_comparison.cluster_coordinates(features)
    repeated, repeated_description = embedding_comparison.cluster_coordinates(features)

    assert coordinates.shape == (80, 10)
    assert description == "PCA 10-D"
    assert repeated_description == description
    assert np.array_equal(coordinates, repeated)


def test_cluster_coordinates_preserve_native_features_at_ten_dimensions():
    from vfe3.viz import embedding_comparison

    features = np.random.default_rng(1).normal(size=(40, 10))
    coordinates, description = embedding_comparison.cluster_coordinates(features)

    assert description == "native 10-D"
    assert np.array_equal(coordinates, features)
    assert coordinates is not features


def test_token_fingerprint_is_stable_and_order_sensitive():
    from vfe3.viz import embedding_comparison

    token_ids = np.array([2, 7, 11, 2], dtype=np.int64)

    assert embedding_comparison.token_fingerprint(token_ids) == embedding_comparison.token_fingerprint(
        torch.tensor(token_ids)
    )
    assert embedding_comparison.token_fingerprint(token_ids) != embedding_comparison.token_fingerprint(
        token_ids[::-1]
    )


def test_relative_position_quartiles_are_sequence_length_normalized():
    from vfe3.viz import embedding_comparison

    positions = np.arange(8)

    assert embedding_comparison.relative_position_quartiles(positions, seq_len=8).tolist() == [
        0, 0, 1, 1, 2, 2, 3, 3,
    ]


def _controlled_record_fixture():
    from vfe3.viz import embedding_comparison

    rng = np.random.default_rng(4)
    features = np.vstack([
        rng.normal(-1.5, 0.25, size=(50, 12)),
        rng.normal(+1.5, 0.25, size=(50, 12)),
    ])
    base = features[:, :2]
    coordinates = {
        seed: base + rng.normal(0.0, 0.002 * (seed + 1), size=base.shape)
        for seed in embedding_comparison.CONTROLLED_SEEDS
    }
    cluster_labels = np.repeat([0, 1], 50)
    token_ids = np.tile([3, 8], 50)
    bpe_labels = token_ids % 2
    function_content_labels = cluster_labels.copy()
    seq_idx = np.repeat(np.arange(10), 10)
    pos_idx = np.tile(np.arange(10), 10)
    contract = embedding_comparison.controlled_contract(
        kind="Belief",
        channel="mu",
        feature_dim=features.shape[1],
        feature_chart="Euclidean means",
        clustering_space="PCA 10-D",
    )
    return {
        "features": features,
        "coords_by_seed": coordinates,
        "cluster_labels": cluster_labels,
        "token_ids": token_ids,
        "bpe_labels": bpe_labels,
        "function_content_labels": function_content_labels,
        "seq_idx": seq_idx,
        "pos_idx": pos_idx,
        "seq_len": 10,
        "contract": contract,
    }


def test_controlled_record_reports_projection_and_confound_metrics(tmp_path):
    from vfe3.viz import embedding_comparison

    record = embedding_comparison.controlled_embedding_record(**_controlled_record_fixture())
    sidecar = embedding_comparison.write_json_atomic(record, tmp_path / "belief_umap_mu.json")
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))

    assert loaded["projection"]["trustworthiness"]["count"] == 5
    assert np.isfinite(loaded["projection"]["neighbor_overlap"]["mean"])
    ami = loaded["clusters"]["adjusted_mutual_information"]
    assert set(ami) == {"bpe", "function_content", "position_quartile", "sequence_identity"}
    assert loaded["sample"]["token_count"] == 100
    assert len(loaded["sample"]["token_sha256"]) == 64
    assert not list(tmp_path.glob(".belief_umap_mu.*.tmp"))


def test_controlled_record_round_trips_semantic_probe_section(tmp_path):
    from vfe3.viz import embedding_comparison, semantic_probes

    semantic = semantic_probes.unavailable_record("fixture semantic record")
    record = embedding_comparison.controlled_embedding_record(
        **_controlled_record_fixture(),
        semantic_probes=semantic,
    )
    sidecar = embedding_comparison.write_json_atomic(record, tmp_path / "semantic.json")
    loaded = json.loads(sidecar.read_text(encoding="utf-8"))

    assert loaded["semantic_probes"] == semantic


def test_controlled_record_defaults_to_explicit_unavailable_semantics():
    from vfe3.viz import embedding_comparison

    record = embedding_comparison.controlled_embedding_record(**_controlled_record_fixture())

    assert record["semantic_probes"]["available"] is False
    assert record["semantic_probes"]["reason"] == "semantic probes not provided"


def test_controlled_record_uses_null_reason_for_unavailable_taxonomy():
    from vfe3.viz import embedding_comparison

    fixture = _controlled_record_fixture()
    fixture["bpe_labels"] = None
    record = embedding_comparison.controlled_embedding_record(**fixture)

    silhouette = record["native_space"]["silhouette"]["bpe"]
    ami = record["clusters"]["adjusted_mutual_information"]["bpe"]
    assert silhouette["value"] is None and silhouette["reason"] == "labels unavailable"
    assert ami["value"] is None and ami["reason"] == "labels unavailable"


def test_controlled_record_names_non_english_taxonomy_unavailability():
    from vfe3.viz import embedding_comparison

    fixture = _controlled_record_fixture()
    fixture["bpe_labels"] = None
    fixture["function_content_labels"] = None
    reason = "English-only linguistic taxonomies disabled for wiki-ar"
    record = embedding_comparison.controlled_embedding_record(
        **fixture,
        taxonomy_unavailable_reason=reason,
    )

    for taxonomy in ("bpe", "function_content"):
        silhouette = record["native_space"]["silhouette"][taxonomy]
        ami = record["clusters"]["adjusted_mutual_information"][taxonomy]
        assert silhouette == {"value": None, "reason": reason}
        assert ami == {"value": None, "reason": reason}
    assert record["clusters"]["adjusted_mutual_information"]["position_quartile"]["value"] is not None
    assert record["clusters"]["adjusted_mutual_information"]["sequence_identity"]["value"] is not None


def test_controlled_plot_uses_fixed_display_and_pca_clustering(tmp_path):
    from vfe3.viz import embedding_comparison, figures

    rng = np.random.default_rng(9)
    features = np.vstack([
        rng.normal(-1.0, 0.2, size=(80, 12)),
        rng.normal(+1.0, 0.2, size=(80, 12)),
    ])
    bank = {
        "mu": torch.tensor(features, dtype=torch.float32),
        "sigma": torch.ones((160, 12), dtype=torch.float32),
        "phi": torch.zeros((160, 2), dtype=torch.float32),
        "token_ids": torch.arange(160) % 8,
        "seq_idx": torch.arange(160) // 8,
        "pos_idx": torch.arange(160) % 8,
    }

    class RecordingWorker:
        def __init__(self):
            self.calls = []

        def embed(self, values, *, n_neighbors, min_dist, n_components, seed):
            self.calls.append((n_neighbors, min_dist, n_components, seed))
            array = np.asarray(values, dtype=float)
            return array[:, :n_components] + seed * 1e-5

        def embed_many(self, values, *, seeds, n_neighbors, min_dist, n_components):
            # The controlled path fits the seeds concurrently (2026-07-26); one `embed` per seed
            # keeps this double's recorded per-seed parameters exactly as they were.
            return {int(seed): self.embed(values, n_neighbors=n_neighbors, min_dist=min_dist,
                                          n_components=n_components, seed=int(seed))
                    for seed in seeds}

    worker = RecordingWorker()
    image_path = tmp_path / "belief_umap_mu.png"
    sidecar_path = tmp_path / "belief_umap_mu.json"
    figure = figures.plot_belief_umap(
        bank,
        "mu",
        controlled=True,
        english_linguistic_diagnostics=False,
        decode=lambda ids: f" token{int(ids[0])}",
        umap_worker=worker,
        path=str(image_path),
        sidecar_path=str(sidecar_path),
    )
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert figure.axes[0].get_title().startswith("Belief mu - controlled clusters")
    assert worker.calls == [
        (embedding_comparison.CONTROLLED_N_NEIGHBORS,
         embedding_comparison.CONTROLLED_MIN_DIST, 2, seed)
        for seed in embedding_comparison.CONTROLLED_SEEDS
    ]
    assert record["display"]["seeds"] == list(embedding_comparison.CONTROLLED_SEEDS)
    assert record["clustering"]["space"] == "PCA 10-D"
    assert record["channel"]["feature_chart"] == "Euclidean means"
    assert record["native_space"]["silhouette"]["bpe"] == {
        "value": None,
        "reason": "English-only linguistic taxonomies disabled for this dataset",
    }
    assert record["semantic_probes"]["available"] is False
    assert record["semantic_probes"]["reason"] == (
        "English semantic probes disabled for this dataset"
    )
    assert any("English linguistic taxonomies disabled" in text.get_text() for text in figure.texts)
    assert image_path.is_file() and sidecar_path.is_file()
    figures.plt.close(figure)


def test_controlled_plot_evaluates_semantics_from_native_features(tmp_path, monkeypatch):
    from vfe3.viz import figures

    features = np.asarray([
        [0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0], [4.2, 0.0, 0.0, 0.0],
        [4.3, 0.0, 0.0, 0.0], [4.5, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0], [0.2, 5.0, 0.0, 0.0],
        [4.0, 5.0, 0.0, 0.0], [4.2, 5.0, 0.0, 0.0],
    ])
    token_ids = np.repeat(np.arange(6), 2)
    bank = {
        "mu": torch.tensor(features, dtype=torch.float32),
        "sigma": torch.ones((12, 4), dtype=torch.float32),
        "phi": torch.zeros((12, 2), dtype=torch.float32),
        "token_ids": torch.tensor(token_ids),
        "seq_idx": torch.arange(12) // 4,
        "pos_idx": torch.arange(12) % 4,
    }
    decoded = {
        0: " king",
        1: " queen",
        2: " father",
        3: " mother",
        4: " dog",
        5: " cat",
    }

    class RecordingWorker:
        def embed(self, values, *, n_neighbors, min_dist, n_components, seed):
            return np.asarray(values, dtype=float)[:, :n_components] + seed * 1e-5

        def embed_many(self, values, *, seeds, n_neighbors, min_dist, n_components):
            return {int(seed): self.embed(values, n_neighbors=n_neighbors, min_dist=min_dist,
                                          n_components=n_components, seed=int(seed))
                    for seed in seeds}

    monkeypatch.setattr(
        figures,
        "_cluster_embedding",
        lambda values, **kwargs: (np.repeat([0, 1, 2], 4), "HDBSCAN test labels"),
    )
    sidecar_path = tmp_path / "semantic_native.json"
    figure = figures.plot_belief_umap(
        bank,
        "mu",
        controlled=True,
        decode=lambda ids: decoded[int(ids[0])],
        umap_worker=RecordingWorker(),
        path=str(tmp_path / "semantic_native.png"),
        sidecar_path=str(sidecar_path),
    )
    record = json.loads(sidecar_path.read_text(encoding="utf-8"))

    semantic = record["semantic_probes"]
    assert semantic["available"] is True
    assert semantic["resolution"]["resolved_concept_count"] == 6
    assert semantic["native_space"]["concept_centroids"]["king"] == pytest.approx([0.1, 0, 0, 0])
    assert semantic["pairs"]["king_queen"]["centroid_distance"]["value"] == pytest.approx(0.3)
    assert "coordinates" not in semantic
    figures.plt.close(figure)


def test_exploratory_plot_retains_adaptive_single_embedding(tmp_path):
    from vfe3.viz import figures

    rng = np.random.default_rng(2)
    bank = {
        "mu": torch.tensor(rng.normal(size=(40, 4)), dtype=torch.float32),
        "sigma": torch.ones((40, 4), dtype=torch.float32),
        "phi": torch.zeros((40, 2), dtype=torch.float32),
        "token_ids": torch.arange(40) % 4,
        "seq_idx": torch.arange(40) // 4,
        "pos_idx": torch.arange(40) % 4,
    }

    class RecordingWorker:
        def __init__(self):
            self.calls = []

        def embed(self, values, *, n_neighbors, min_dist, n_components, seed):
            self.calls.append((n_neighbors, min_dist, n_components, seed))
            return np.asarray(values, dtype=float)[:, :n_components]

    worker = RecordingWorker()
    figure = figures.plot_belief_umap(
        bank,
        "mu",
        umap_worker=worker,
        path=str(tmp_path / "exploratory.png"),
    )

    assert figure.axes[0].get_title().startswith("Belief mu - exploratory clusters")
    assert len(worker.calls) == 1
    assert worker.calls[0][0] == figures._UMAP_N_NEIGHBORS
    figures.plt.close(figure)


def _matching_controlled_records():
    from vfe3.viz import embedding_comparison, semantic_probes

    semantic = semantic_probes.unavailable_record("comparison fixture")
    semantic["available"] = True
    semantic["reason"] = None
    semantic["native_space"]["semantic_field_silhouette"] = {"value": 0.4, "reason": None}
    semantic["clusters"]["semantic_field_adjusted_mutual_information"] = {
        "value": 0.3,
        "reason": None,
    }
    close = semantic["aggregate"]["expectations"]["close"]
    close["mean_distance_percentile"] = {"value": 12.5, "reason": None}
    close["mean_reciprocal_rank"] = {"value": 0.75, "reason": None}
    close["hit_at_5_rate"] = {"value": 0.8, "reason": None}
    close["mean_hdbscan_co_membership"] = {"value": 0.6, "reason": None}
    semantic["aggregate"]["control_to_close_distance_ratio"] = {
        "value": 2.0,
        "reason": None,
    }
    left = embedding_comparison.controlled_embedding_record(
        **_controlled_record_fixture(),
        semantic_probes=semantic,
    )
    return left, copy.deepcopy(left)


def test_comparator_accepts_matching_controlled_contracts():
    from vfe3.viz import embedding_comparison

    left, right = _matching_controlled_records()

    embedding_comparison.validate_comparison_records([left, right])


def test_comparator_rejects_token_population_mismatch():
    from vfe3.viz import embedding_comparison

    left, right = _matching_controlled_records()
    right["sample"]["token_sha256"] = "different"

    with pytest.raises(ValueError, match="sample.token_sha256"):
        embedding_comparison.validate_comparison_records([left, right])


def test_comparator_reports_every_mismatched_contract_field():
    from vfe3.viz import embedding_comparison

    left, right = _matching_controlled_records()
    right["display"]["n_neighbors"] = 99
    right["clustering"]["min_samples"] = 3

    with pytest.raises(ValueError) as exc:
        embedding_comparison.validate_comparison_records([left, right])

    assert "display.n_neighbors" in str(exc.value)
    assert "clustering.min_samples" in str(exc.value)


def test_comparator_rejects_semantic_manifest_drift():
    from vfe3.viz import embedding_comparison

    left, right = _matching_controlled_records()
    right["semantic_probes"]["manifest"]["schema_version"] = 2

    with pytest.raises(ValueError, match="semantic_probes.manifest.schema_version"):
        embedding_comparison.validate_comparison_records([left, right])


def test_controlled_comparison_figure_has_six_metric_panels():
    from vfe3.viz import embedding_comparison, figures

    left, right = _matching_controlled_records()
    summary = embedding_comparison.comparison_summary([left, right], ["N=256", "N=512"])

    figure = figures.plot_controlled_embedding_comparison(summary)

    assert len(figure.axes) == 6
    assert all("UMAP" not in axis.get_title() for axis in figure.axes)
    figures.plt.close(figure)


def test_cross_run_figure_contains_metrics_not_independent_coordinates(tmp_path):
    from vfe3.viz import embedding_comparison

    left, right = _matching_controlled_records()
    left_path = embedding_comparison.write_json_atomic(left, tmp_path / "n256.json")
    right_path = embedding_comparison.write_json_atomic(right, tmp_path / "n512.json")

    json_path, figure_path = report.compare_belief_umap_sidecars(
        sidecars=[left_path, right_path],
        labels=["N=256", "N=512"],
        json_path=tmp_path / "comparison.json",
        figure_path=tmp_path / "comparison.png",
    )
    comparison = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.is_file() and figure_path.is_file()
    assert comparison["artifact"] == "controlled_belief_geometry_comparison"
    assert [arm["label"] for arm in comparison["arms"]] == ["N=256", "N=512"]
    assert "coordinates" not in comparison
    assert "umap_coordinates" not in json.dumps(comparison)
    assert set(comparison["arms"][0]["metrics"]) == {
        "native_silhouette_bpe",
        "native_silhouette_function_content",
        "trustworthiness",
        "neighbor_overlap",
        "cluster_count",
        "noise_fraction",
        "ami_bpe",
        "ami_function_content",
        "ami_position_quartile",
        "ami_sequence_identity",
        "semantic_field_silhouette",
        "semantic_field_ami",
        "close_pair_mean_distance_percentile",
        "close_pair_mean_reciprocal_rank",
        "close_pair_hit_at_5_rate",
        "close_pair_mean_hdbscan_co_membership",
        "control_to_close_distance_ratio",
    }
    assert comparison["arms"][0]["metrics"]["semantic_field_silhouette"] == 0.4
    assert comparison["arms"][0]["metrics"]["close_pair_mean_reciprocal_rank"] == 0.75


def test_generate_figures_default_wires_controlled_sidecars(tmp_path, monkeypatch):
    from vfe3.viz import figures

    model = _tiny_model()
    loader = [torch.arange(0, 8).reshape(2, 4)]
    calls = []

    class NoopWorker:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    def record_umap(*args, **kwargs):
        calls.append(kwargs)
        return figures.plt.figure()

    monkeypatch.setattr(report.figs, "UMAPWorker", NoopWorker)
    monkeypatch.setattr(report.figs, "plot_belief_umap", record_umap)

    report.generate_figures(
        tmp_path / "run",
        model=model,
        loader=loader,
        max_tokens=8,
        generate_embedding_figures=True,
    )

    assert len(calls) == 3
    assert all(call["controlled"] is True for call in calls)
    assert all(call["sidecar_path"].endswith(".json") for call in calls)
    assert {Path(call["sidecar_path"]).name for call in calls} == {
        "belief_umap_mu.json",
        "belief_umap_sigma.json",
        "belief_umap_phi.json",
    }
