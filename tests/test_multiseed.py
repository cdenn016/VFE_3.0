r"""Tests for the across-seed aggregator (multiseed_analysis).

Original EXP-1 scalar aggregator added 2026-06-21; full cross-seed digest (curves, per-layer,
research scalars, figure set) added 2026-06-22.
"""

from tests.profiles import test_config
import json
import math
from dataclasses import asdict
from enum import IntEnum

import numpy as np
import pytest

import multiseed_analysis as ms
from vfe3.config import serialize_config


def _write_run(d, ppl, seed):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"test_ppl": ppl}))
    (d / "config.json").write_text(json.dumps(serialize_config(test_config(seed=seed))))


def test_aggregate_scalar_mean_sd_and_seeds(tmp_path):
    vals = [10.0, 12.0, 11.0, 13.0, 9.0]                     # mean 11, ddof=1 var 2.5
    for i, v in enumerate(vals):
        _write_run(tmp_path / f"seed{i}", v, seed=i)
    out = ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))
    assert out["n"] == 5
    assert abs(out["mean"] - 11.0) < 1e-9
    assert abs(out["sd"] - math.sqrt(2.5)) < 1e-9
    assert abs(out["noise_floor"] - 2.7764451051977987 * math.sqrt(2.0) * math.sqrt(2.5)) < 1e-9   # t_{4, 0.975}
    assert abs(out["cv"] - math.sqrt(2.5) / 11.0) < 1e-9
    assert sorted(out["seeds"]) == [0, 1, 2, 3, 4]


def test_aggregate_skips_nonfinite_and_unreadable(tmp_path):
    _write_run(tmp_path / "a", 10.0, seed=1)
    _write_run(tmp_path / "b", float("inf"), seed=2)         # inf serializes to "Infinity" -> skipped
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "summary.json").write_text("{ not json")
    (tmp_path / "c" / "config.json").write_text(json.dumps(
        serialize_config(test_config(seed=3))))
    out = ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))
    assert out["n"] == 1 and abs(out["mean"] - 10.0) < 1e-9
    assert math.isnan(out["sd"])                            # n<2 -> SD undefined


def test_aggregate_empty_root(tmp_path):
    out = ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))
    assert out["n"] == 0 and math.isnan(out["mean"])


# =============================================================================
# Full cross-seed digest (2026-06-22): run-root resolution, seed source, curves,
# per-layer, research scalars, and the new figure functions.
# =============================================================================
def _write_full_run(d, *, ppl, seed, metrics=None, per_layer=None, research=None, config=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"test_ppl": ppl}))
    config = serialize_config(test_config(seed=seed)) if config is None else config
    (d / "config.json").write_text(json.dumps(config))                    # real seed lives in provenance
    (d / "provenance.json").write_text(json.dumps({
        "seed": seed,
        "git_sha": "a" * 40,
        "git_dirty": False,
        "git_dirty_fingerprint": None,
        "train_data_sha256": "b" * 64,
        "train_data_n_tokens": 100,
        "val_data_sha256": "c" * 64,
        "val_data_n_tokens": 20,
        "test_data_sha256": "d" * 64,
        "test_data_n_tokens": 20,
        "data_seed": 3,
        "max_tokens": None,
        "tokenizer_tag": "synthetic",
    }))
    if metrics is not None:
        (d / "metrics.csv").write_text(metrics)
    if per_layer is not None:
        (d / "metrics_per_layer.csv").write_text(per_layer)
    if research is not None:
        (d / "research.json").write_text(json.dumps(research))


def _write_request(root, seeds):
    (root / "multiseed_request.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "seeds": seeds,
        "cells": [{"seed": seed, "status": "complete"} for seed in seeds],
    }))


def test_resolve_run_root_bare_name_under_vfe3_runs(tmp_path, monkeypatch):
    (tmp_path / "vfe3_runs" / "K=20_GL(10)").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    resolved = ms._resolve_run_root("K=20_GL(10)")
    assert resolved.resolve() == (tmp_path / "vfe3_runs" / "K=20_GL(10)").resolve()


def test_resolve_run_root_prefers_existing_literal(tmp_path, monkeypatch):
    (tmp_path / "explicit").mkdir()
    monkeypatch.chdir(tmp_path)
    assert ms._resolve_run_root("explicit").resolve() == (tmp_path / "explicit").resolve()


def test_seed_for_accepts_agreeing_provenance_and_directory_identity(tmp_path):
    d = tmp_path / "run_s99"
    d.mkdir()
    (d / "provenance.json").write_text(json.dumps({"seed": 99}))
    (d / "config.json").write_text(json.dumps({"seed": None}))
    assert ms._seed_for(d) == 99


def test_seed_for_falls_back_to_config_then_dirname(tmp_path):
    cfg = tmp_path / "run_a"
    cfg.mkdir()
    (cfg / "config.json").write_text(json.dumps({"seed": 5}))
    assert ms._seed_for(cfg) == 5
    name = tmp_path / "wikitext_K20_s42"
    name.mkdir()
    assert ms._seed_for(name) == 42


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1", -1])
def test_seed_for_rejects_malformed_high_priority_seed_without_fallback(tmp_path, bad_seed):
    run = tmp_path / "run_s42"
    run.mkdir()
    (run / "provenance.json").write_text(json.dumps({"seed": bad_seed}), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"seed": 5}), encoding="utf-8")

    assert ms._seed_for(run) is None


def test_seed_for_rejects_malformed_high_priority_json_without_fallback(tmp_path):
    run = tmp_path / "run_s42"
    run.mkdir()
    (run / "provenance.json").write_text("{ not json", encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"seed": 5}), encoding="utf-8")

    assert ms._seed_for(run) is None


def test_seed_for_explicit_none_or_absent_key_allows_agreeing_fallback(tmp_path):
    explicit_none = tmp_path / "none_s5"
    explicit_none.mkdir()
    (explicit_none / "provenance.json").write_text(json.dumps({"seed": None}), encoding="utf-8")
    (explicit_none / "config.json").write_text(json.dumps({"seed": 5}), encoding="utf-8")
    absent_key = tmp_path / "absent_s42"
    absent_key.mkdir()
    (absent_key / "provenance.json").write_text(json.dumps({"git_sha": "abc"}), encoding="utf-8")

    assert ms._seed_for(explicit_none) == 5
    assert ms._seed_for(absent_key) == 42


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1", -1])
def test_multiseed_manifest_rejects_nonexact_or_negative_requested_seed(
    tmp_path, monkeypatch, bad_seed,
):
    request_path = tmp_path / "multiseed_request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ms, "_read_json", lambda path: {
        "schema_version": 1,
        "status": "complete",
        "seeds": [bad_seed],
        "cells": [{"seed": bad_seed, "status": "complete"}],
    })

    manifest = ms._request_manifest(tmp_path, [7])

    assert manifest["request_verified"] is False
    assert manifest["requested_seeds"] == [7]


def test_multiseed_manifest_rejects_integer_enum_requested_seed(tmp_path, monkeypatch):
    class Seed(IntEnum):
        ONE = 1

    request_path = tmp_path / "multiseed_request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ms, "_read_json", lambda path: {
        "schema_version": 1,
        "status": "complete",
        "seeds": [Seed.ONE],
        "cells": [{"seed": Seed.ONE, "status": "complete"}],
    })

    assert ms._request_manifest(tmp_path, [7])["request_verified"] is False


def test_multiseed_manifest_rejects_integer_enum_observed_seed(tmp_path):
    class Seed(IntEnum):
        ONE = 1

    manifest = ms._request_manifest(tmp_path, [Seed.ONE])

    assert manifest["request_verified"] is False
    assert manifest["requested_seeds"] == []


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1", -1])
def test_multiseed_manifest_rejects_invalid_observed_seed_without_request(tmp_path, bad_seed):
    manifest = ms._request_manifest(tmp_path, [bad_seed])

    assert manifest["request_verified"] is False
    assert manifest["requested_seeds"] == []


def test_aggregate_scalar_uses_provenance_seed(tmp_path):
    for v, s in [(10.0, 23), (12.0, 54)]:
        _write_full_run(tmp_path / f"run_s{s}", ppl=v, seed=s)
    out = ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))
    assert out["n"] == 2 and sorted(out["seeds"]) == [23, 54]


def test_seed_dirs_reject_mixed_semantic_configs(tmp_path):
    run_a = tmp_path / "run_s1"
    run_b = tmp_path / "run_s2"
    _write_full_run(
        run_a,
        ppl=10.0,
        seed=1,
        config={"config": serialize_config(test_config(
            seed=1, embed_dim=20, n_heads=4)), "timestamp": "first"},
    )
    _write_full_run(
        run_b,
        ppl=11.0,
        seed=2,
        config={"config": serialize_config(test_config(
            seed=2, embed_dim=40, n_heads=4)), "timestamp": "second"},
    )

    with pytest.raises(ValueError, match="mixed semantic config fingerprints") as exc:
        ms._seed_dirs(tmp_path)

    message = str(exc.value)
    assert ms._config_fingerprint(run_a / "config.json") in message
    assert ms._config_fingerprint(run_b / "config.json") in message
    assert str(run_a / "config.json") in message
    assert str(run_b / "config.json") in message
    with pytest.raises(ValueError, match="mixed semantic config fingerprints"):
        ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))


def test_seed_dirs_accept_homogeneous_semantic_configs(tmp_path):
    nested = tmp_path / "run_s1"
    flat = tmp_path / "run_s2"
    cfg_nested = test_config(seed=1, vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4)
    cfg_flat = test_config(seed=2, vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4)
    _write_full_run(
        nested,
        ppl=10.0,
        seed=1,
        config={
            "config": serialize_config(cfg_nested),
            "timestamp": "first",
        },
    )
    _write_full_run(
        flat,
        ppl=12.0,
        seed=2,
        config=serialize_config(cfg_flat),
    )

    assert (
        ms._config_fingerprint(nested / "config.json")
        == ms._config_fingerprint(flat / "config.json")
    )
    assert ms._seed_dirs(tmp_path) == [nested, flat]
    out = ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(11.0)


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("retired_surface", ["active_phi", "ambiguous_policy"])
def test_semantic_config_keeps_dropped_legacy_fields_in_the_fingerprint(
    tmp_path,
    nested,
    retired_surface,
):
    payload = asdict(test_config(seed=1, vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4))
    payload["pos_phi"] = "none"
    clean_path = tmp_path / f"clean-{nested}.json"
    clean_path.write_text(json.dumps({"config": payload} if nested else payload), encoding="utf-8")
    if retired_surface == "active_phi":
        payload.pop("m_phi_update_mode")
        retired = {
            "m_phi_natural_grad": True,
            "m_gauge_momentum": 0.9,
            "m_gauge_update_rule": "heavy_ball",
        }
    else:
        retired = {"policy_mode": "future_policy"}
    payload.update(retired)
    artifact = {"config": payload, "timestamp": "legacy"} if nested else payload
    config_path = tmp_path / f"{retired_surface}-{nested}.json"
    config_path.write_text(json.dumps(artifact), encoding="utf-8")

    # The lenient loader of vfe3.paper drops fields this build no longer declares; they stay in the
    # semantic config, so such a run never shares a fingerprint with one that lacks them.
    semantic = ms._semantic_config(config_path)
    assert {key: semantic[f"retired:{key}"] for key in retired} == retired
    assert ms._config_fingerprint(config_path) != ms._config_fingerprint(clean_path)


@pytest.mark.parametrize("config_text", [None, "{ not json"])
def test_seed_dirs_reject_missing_or_unreadable_config(tmp_path, config_text):
    run_dir = tmp_path / "run_s1"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"test_ppl": 10.0}))
    if config_text is not None:
        (run_dir / "config.json").write_text(config_text)

    with pytest.raises(ValueError, match="readable config.json"):
        ms._seed_dirs(tmp_path)


def test_requested_panel_reports_unexpected_duplicate_runs_and_withholds_publication(
    tmp_path, monkeypatch,
):
    _write_request(tmp_path, [8])
    for dirname, seed, ppl in (
        ("requested_s8", 8, 10.0),
        ("unexpected_s6", 6, 11.0),
        ("unexpected_retry_s6_2", 6, 12.0),
    ):
        _write_full_run(
            tmp_path / dirname,
            ppl=ppl,
            seed=seed,
            metrics=f"step,train_ce\n1,{ppl}\n",
            config={"config": serialize_config(test_config(
                seed=seed, embed_dim=20, n_heads=4,
                generate_embedding_figures=False,
            ))},
        )

    design = ms._requested_seed_design(tmp_path)
    assert design["requested_seeds"] == [8]
    assert design["accepted_seeds"] == [8]
    assert design["observed_seeds"] == [6, 6, 8]
    assert design["unexpected_seeds"] == [6]
    assert design["duplicate_seeds"] == [6]
    assert design["unidentified_runs"] == []
    assert design["n_accepted_seeds"] == 1
    assert design["n_observed_runs"] == 3
    assert design["complete"] is False

    aggregate = ms.aggregate_scalar(tmp_path, "test_ppl", sources=("summary.json",))
    assert aggregate["n"] == 1
    assert aggregate["seeds"] == [8]
    assert aggregate["accepted_seeds"] == [8]
    assert aggregate["observed_seeds"] == [6, 6, 8]
    assert aggregate["complete"] is False
    assert ms.aggregate_seed_curves(tmp_path, columns=["train_ce"]) == {}

    emitted = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))

    assert ms.main() != 0
    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["n_seeds"] == 1
    assert summary["seeds"] == [8]
    assert summary["n_observed_runs"] == 3
    assert summary["observed_seeds"] == [6, 6, 8]
    assert summary["design"]["accepted_seeds"] == [8]
    assert summary["design"]["unexpected_seeds"] == [6]
    assert summary["design"]["duplicate_seeds"] == [6]
    assert summary["design"]["complete"] is False
    assert summary["scalars"] == {}
    assert summary["curves_final_step"] == {}
    assert summary["withheld"] == {
        "scalars": True,
        "curves": True,
        "per_layer": True,
        "figures": True,
    }
    assert emitted == []


def test_requested_panel_classifies_unreadable_extra_before_config_homogeneity(tmp_path):
    _write_request(tmp_path, [8])
    _write_full_run(
        tmp_path / "requested_s8",
        ppl=10.0,
        seed=8,
        config={"config": serialize_config(test_config(
            seed=8, embed_dim=20, n_heads=4))},
    )
    unidentified = tmp_path / "unidentified"
    unidentified.mkdir()
    (unidentified / "summary.json").write_text(json.dumps({"test_ppl": 99.0}))
    (unidentified / "config.json").write_text("{ not json")

    design = ms._requested_seed_design(tmp_path)

    assert design["accepted_seeds"] == [8]
    assert design["observed_seeds"] == [8]
    assert design["unidentified_runs"] == [str(unidentified)]
    assert design["n_observed_runs"] == 2
    assert design["complete"] is False


def test_manifest_failed_cell_is_excluded_from_accepted_config_homogeneity(
    tmp_path, monkeypatch,
):
    (tmp_path / "multiseed_request.json").write_text(json.dumps({
        "schema_version": 1,
        "status": "incomplete",
        "seeds": [8, 23],
        "cells": [
            {"seed": 8, "status": "complete"},
            {"seed": 23, "status": "failed"},
        ],
    }))
    _write_full_run(
        tmp_path / "requested_s8",
        ppl=10.0,
        seed=8,
        config={"config": serialize_config(test_config(
            seed=8, embed_dim=20, n_heads=4,
            generate_embedding_figures=False,
        ))},
    )
    _write_full_run(
        tmp_path / "failed_s23",
        ppl=99.0,
        seed=23,
        config={"config": serialize_config(test_config(
            seed=23, embed_dim=40, n_heads=4,
            generate_embedding_figures=False,
        ))},
    )

    design = ms._requested_seed_design(tmp_path)

    assert design["accepted_seeds"] == [8]
    assert design["observed_seeds"] == [8, 23]
    assert design["cells"] == [
        {"seed": 8, "status": "complete", "run_dir": str(tmp_path / "requested_s8")},
        {"seed": 23, "status": "failed", "run_dir": None},
    ]
    assert design["complete"] is False

    emitted = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))

    assert ms.main() != 0
    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["n_seeds"] == 1
    assert summary["seeds"] == [8]
    assert summary["n_observed_runs"] == 2
    assert summary["observed_seeds"] == [8, 23]
    assert summary["design"]["complete"] is False
    assert summary["scalars"] == {}
    assert summary["curves_final_step"] == {}
    assert summary["withheld"] == {
        "scalars": True,
        "curves": True,
        "per_layer": True,
        "figures": True,
    }
    assert emitted == []


def test_mixed_code_identity_is_rejected_by_default_and_accepted_on_opt_in(tmp_path, monkeypatch):
    for seed in (1, 2):
        _write_full_run(tmp_path / f"run_s{seed}", ppl=10.0 + seed, seed=seed)
    prov_path = tmp_path / "run_s2" / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov.update(git_sha="e" * 40, git_dirty=True, git_dirty_fingerprint="f" * 64)
    prov_path.write_text(json.dumps(prov))
    _write_request(tmp_path, [1, 2])

    # Pin the default: the launcher's CONFIG is a user-edited click-to-run dict.
    monkeypatch.setitem(ms.CONFIG, "allow_mixed_code_identity", False)
    design = ms._requested_seed_design(tmp_path)
    assert design["complete"] is False
    assert design["accepted_seeds"] == []
    assert design["code_identity"]["mixed"] is True

    monkeypatch.setitem(ms.CONFIG, "allow_mixed_code_identity", True)
    design = ms._requested_seed_design(tmp_path)
    assert design["complete"] is True
    assert design["accepted_seeds"] == [1, 2]
    assert design["code_identity"]["mixed_allowed"] is True
    assert design["code_identity"]["n_distinct"] == 2
    assert design["code_identity"]["per_seed"]["2"]["git_sha"] == "e" * 40

    # Data identity remains enforced under the opt-in.
    prov.update(test_data_sha256="0" * 64)
    prov_path.write_text(json.dumps(prov))
    assert ms._requested_seed_design(tmp_path)["complete"] is False


def test_duplicate_observation_for_requested_seed_is_runtime_cohort_invalid(tmp_path):
    _write_request(tmp_path, [8])
    for dirname, ppl in (("requested_s8", 10.0), ("requested_retry_s8_2", 11.0)):
        _write_full_run(
            tmp_path / dirname,
            ppl=ppl,
            seed=8,
            config={"config": {"seed": 8, "embed_dim": 20, "n_heads": 4}},
        )

    design = ms._requested_seed_design(tmp_path)

    assert design["request_verified"] is True
    assert design["accepted_seeds"] == []
    assert design["observed_seeds"] == [8, 8]
    assert design["duplicate_seeds"] == [8]
    assert design["cells"] == [{"seed": 8, "status": "duplicate", "run_dir": None}]
    assert design["complete"] is False


def test_aggregate_scalar_reads_dotted_research_key(tmp_path):
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1, research={"corpus_freq_strata_ce": {"rare": 7.0}})
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, research={"corpus_freq_strata_ce": {"rare": 9.0}})
    out = ms.aggregate_scalar(tmp_path, "corpus_freq_strata_ce.rare")
    assert out["n"] == 2 and abs(out["mean"] - 8.0) < 1e-9


@pytest.mark.parametrize(
    ("first_source", "expected_status"),
    [
        ("{ not json", "unreadable"),
        (json.dumps({"test_ppl": float("inf")}), "nonfinite"),
    ],
)
def test_aggregate_scalar_does_not_fallback_past_corrupt_or_nonfinite_source(
    tmp_path, first_source, expected_status,
):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=10.0, seed=1)
    _write_full_run(tmp_path / "b", ppl=11.0, seed=2)
    (tmp_path / "b" / "summary.json").write_text(first_source)
    (tmp_path / "b" / "test_results.json").write_text(json.dumps({"test_ppl": 99.0}))

    out = ms.aggregate_scalar(tmp_path, "test_ppl")

    assert out["n"] == 1 and out["values"] == [10.0]
    assert out["complete"] is False
    assert {cell["seed"]: cell["status"] for cell in out["cells"]} == {
        1: "complete",
        2: expected_status,
    }


def test_aggregate_scalar_falls_back_when_earlier_source_is_absent(tmp_path):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=10.0, seed=1)
    _write_full_run(tmp_path / "b", ppl=11.0, seed=2)
    (tmp_path / "b" / "summary.json").unlink()
    (tmp_path / "b" / "test_results.json").write_text(json.dumps({"test_ppl": 12.0}))

    out = ms.aggregate_scalar(tmp_path, "test_ppl")

    assert out["complete"] is True
    assert out["values"] == [10.0, 12.0]


def test_aggregate_seed_curves_aligns_every_requested_seed(tmp_path):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1, metrics="step,x,y\n100,1,5\n200,3,9\n")
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, metrics="step,x,y\n100,3,7\n200,5,11\n")
    curves = ms.aggregate_seed_curves(tmp_path, columns=["x", "y"])
    np.testing.assert_allclose(curves["x"]["steps"], [100, 200])
    np.testing.assert_allclose(curves["x"]["mean"], [2.0, 4.0])
    np.testing.assert_allclose(curves["x"]["sd"][0], math.sqrt(2.0))      # ddof=1 over {1,3}
    np.testing.assert_allclose(curves["x"]["n"], [2, 2])
    np.testing.assert_allclose(curves["y"]["mean"], [6.0, 10.0])
    np.testing.assert_allclose(curves["y"]["n"], [2, 2])


def test_aggregate_seed_curves_respects_metric_specific_cadence(tmp_path, monkeypatch):
    _write_request(tmp_path, [1, 2])
    per_layer = "layer,self_coupling\n0,1\n"
    _write_full_run(
        tmp_path / "a",
        ppl=1.0,
        seed=1,
        metrics=(
            "step,train_ce,val_ppl,unsupported\n"
            "100,5,,\n"
            "200,4,10,\n"
            "300,3,,\n"
        ),
        per_layer=per_layer,
    )
    _write_full_run(
        tmp_path / "b",
        ppl=1.0,
        seed=2,
        metrics=(
            "step,train_ce,val_ppl,unsupported\n"
            "100,7,,\n"
            "200,6,14,\n"
            "300,5,,\n"
        ),
        per_layer=per_layer,
    )

    curves = ms.aggregate_seed_curves(tmp_path)

    assert set(curves) == {"train_ce", "val_ppl"}
    np.testing.assert_allclose(curves["train_ce"]["steps"], [100, 200, 300])
    np.testing.assert_allclose(curves["train_ce"]["mean"], [6.0, 5.0, 4.0])
    np.testing.assert_allclose(curves["train_ce"]["n"], [2, 2, 2])
    np.testing.assert_allclose(curves["val_ppl"]["steps"], [200])
    np.testing.assert_allclose(curves["val_ppl"]["mean"], [12.0])
    np.testing.assert_allclose(curves["val_ppl"]["n"], [2])

    emitted = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))

    ms.main()

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is True
    assert summary["diagnostics"]["curves_complete"] is True
    assert summary["withheld"] == {
        "scalars": False,
        "curves": False,
        "per_layer": False,
        "figures": False,
    }
    assert set(summary["curves_final_step"]) == {"train_ce", "val_ppl"}
    assert len(emitted) == 1


def test_main_publishes_nonlayer_channels_when_embeddings_are_disabled(
    tmp_path, monkeypatch,
):
    _write_request(tmp_path, [1, 2])
    for seed, ppl, train_ce in ((1, 10.0, 5.0), (2, 12.0, 7.0)):
        _write_full_run(
            tmp_path / f"run_s{seed}",
            ppl=ppl,
            seed=seed,
            metrics=f"step,train_ce\n100,{train_ce}\n",
            config={"config": {
                "attention_mode": "projected_predictive",
                "seed": seed,
                "pos_phi": "none",
                "generate_embedding_figures": False,
            }},
        )

    emitted = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))

    ms.main()

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is True
    assert summary["scalars"]["test_ppl"]["mean"] == pytest.approx(11.0)
    assert summary["curves_final_step"]["train_ce"]["mean"] == pytest.approx(6.0)
    assert summary["per_layer"] == {}
    assert summary["withheld"]["scalars"] is False
    assert summary["withheld"]["curves"] is False
    assert summary["withheld"]["figures"] is False
    assert summary["diagnostics"]["per_layer_complete"] is False
    assert len(emitted) == 1
    assert emitted[0][3] == {}


def test_main_requires_the_per_layer_channel_even_with_embeddings_disabled(
    tmp_path, monkeypatch,
):
    _write_request(tmp_path, [1, 2])
    for seed, ppl, train_ce in ((1, 10.0, 5.0), (2, 12.0, 7.0)):
        cfg = test_config(seed=seed, generate_embedding_figures=False)
        _write_full_run(
            tmp_path / f"run_s{seed}",
            ppl=ppl,
            seed=seed,
            metrics=f"step,train_ce\n100,{train_ce}\n",
            config={"config": serialize_config(cfg)},
        )

    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: None)

    assert ms.main() == 0

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics"]["per_layer_requested"] is True
    assert summary["diagnostics"]["per_layer_complete"] is False
    assert summary["withheld"]["per_layer"] is True


def test_main_withholds_only_partial_optional_per_layer_channel(tmp_path, monkeypatch):
    _write_request(tmp_path, [1, 2])
    for seed, ppl, train_ce in ((1, 10.0, 5.0), (2, 12.0, 7.0)):
        _write_full_run(
            tmp_path / f"run_s{seed}",
            ppl=ppl,
            seed=seed,
            metrics=f"step,train_ce\n100,{train_ce}\n",
            per_layer="layer,self_coupling\n0,1\n" if seed == 1 else None,
            config={"config": {
                "attention_mode": "projected_predictive",
                "seed": seed,
                "pos_phi": "none",
                "generate_embedding_figures": False,
            }},
        )

    emitted = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))

    ms.main()

    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is True
    assert summary["scalars"]["test_ppl"]["mean"] == pytest.approx(11.0)
    assert summary["curves_final_step"]["train_ce"]["mean"] == pytest.approx(6.0)
    assert summary["per_layer"] == {}
    assert summary["withheld"] == {
        "scalars": False,
        "curves": False,
        "per_layer": True,
        "figures": False,
    }
    assert summary["diagnostics"]["per_layer_requested"] is True
    assert summary["diagnostics"]["per_layer_complete"] is False
    assert len(emitted) == 1
    assert emitted[0][3] == {}


@pytest.mark.parametrize(
    "seed_two_metrics",
    [
        "step,x\n100,3\n",
        "step,x\n100,3\n200,inf\n",
    ],
    ids=["missing-step", "nonfinite-point"],
)
def test_aggregate_seed_curves_withholds_partial_seed_points_and_main_figures(
    tmp_path, monkeypatch, seed_two_metrics,
):
    _write_request(tmp_path, [1, 2])
    layer = "layer,self_coupling\n0,1.0\n"
    _write_full_run(
        tmp_path / "a",
        ppl=10.0,
        seed=1,
        metrics="step,x\n100,1\n200,2\n",
        per_layer=layer,
    )
    _write_full_run(
        tmp_path / "b",
        ppl=11.0,
        seed=2,
        metrics=seed_two_metrics,
        per_layer=layer,
    )

    assert ms.aggregate_seed_curves(tmp_path, columns=["x"]) == {}

    emitted = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))

    ms.main()

    # Audit 2026-10-08 general-C#5 / gap-stats#6: the partial column is withheld on its own with a
    # reason; the complete design still publishes its scalars and figures.
    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["complete"] is True
    assert summary["diagnostics"]["curves_complete"] is False
    assert "x" in summary["diagnostics"]["withheld_curves"]
    assert "x" not in summary["curves_final_step"]
    assert summary["scalars"]["test_ppl"]["n"] == 2
    assert summary["withheld"]["figures"] is False
    assert len(emitted) == 1


def test_aggregate_per_layer(tmp_path):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1, per_layer="layer,self_coupling\n0,10\n")
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, per_layer="layer,self_coupling\n0,20\n")
    out = ms.aggregate_per_layer(tmp_path)
    assert abs(out[0]["self_coupling"]["mean"] - 15.0) < 1e-9
    assert abs(out[0]["self_coupling"]["sd"] - math.sqrt(50.0)) < 1e-9   # ddof=1 over {10,20}


# --- new figure functions (smoke tests, test_viz style) ---
def _saved_nonempty(path):
    return path.exists() and path.stat().st_size > 0


def test_plot_curve_band_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    steps = np.array([1.0, 2.0, 3.0])
    fig = figs.plot_curve_band(steps, np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]),
                               label="x", ylabel="v", path=str(tmp_path / "band.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "band.png")


def test_plot_curve_band_grid_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    curves = [
        {"steps": np.array([1.0, 2.0]), "mean": np.array([1.0, 2.0]), "sd": np.array([0.1, 0.1]),
         "title": "a", "logy": False},
        {"steps": np.array([1.0, 2.0]), "mean": np.array([3.0, 4.0]), "sd": np.array([0.2, 0.2]),
         "title": "b", "logy": True},
    ]
    fig = figs.plot_curve_band_grid(curves, path=str(tmp_path / "grid.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "grid.png")


def test_plot_scalar_cv_summary_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    aggs = {
        "test_ppl": {"mean": 137.0, "sd": 0.9, "cv": 0.0065, "values": [136.0, 137.0, 138.0]},
        "test_ce":  {"mean": 4.9, "sd": 0.007, "cv": 0.0014, "values": [4.89, 4.9, 4.91]},
    }
    fig = figs.plot_scalar_cv_summary(aggs, path=str(tmp_path / "cv.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "cv.png")


def test_plot_per_layer_band_writes(tmp_path):
    from vfe3.viz import figures as figs
    figs.set_publication_style()
    per_layer = {0: {"self_coupling": {"mean": 15.0, "sd": 5.0, "n": 2, "values": [10.0, 20.0]}},
                 1: {"self_coupling": {"mean": 30.0, "sd": 4.0, "n": 2, "values": [28.0, 32.0]}}}
    fig = figs.plot_per_layer_band(per_layer, "self_coupling", path=str(tmp_path / "layer.png"))
    figs.plt.close(fig)
    assert _saved_nonempty(tmp_path / "layer.png")
