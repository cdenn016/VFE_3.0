r"""Regression tests for the 2026-10-08 audit statistics items (scaling_analysis, multiseed_analysis).

Covers gap-stats#1-#6, #8-#10 and general-C-analysis-tests#4, #5, #7. Every fixture is a tiny
synthetic JSON/CSV run directory; no model is built.
"""

from tests.profiles import test_config
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

import multiseed_analysis as ms
import scaling_analysis as sa
from tests.test_multiseed import _write_full_run, _write_request
from vfe3.artifact_utils import canonical_json_sha256
from vfe3.config import serialize_config


# =============================================================================
# scaling_analysis fixtures
# =============================================================================

def _scaling_cfg(seed: int, embed_dim: int, **overrides: Any):
    return test_config(seed=seed, vocab_size=16, embed_dim=embed_dim, n_heads=2, max_seq_len=8,
                       n_layers=1, n_e_steps=1, family="gaussian_diagonal",
                       gauge_group="block_glk", **overrides)


def _write_scaling_run(
    root:       Path,
    *,
    route:      str,
    label:      str,
    seed:       int,
    n_params:   int,
    test_ce:    float,
    embed_dim:  int = 8,
    scale_knob: str = "embed_dim",
    tokens_seen: int = 1000,
    **cfg_overrides: Any,
) -> None:
    r"""One schema-2 scaling cell whose config digest uses the producer's serialized form."""
    run = root / route / label / f"s{seed}"
    run.mkdir(parents=True)
    cfg = _scaling_cfg(seed, embed_dim, **cfg_overrides)
    config = json.loads(json.dumps(serialize_config(cfg), default=str))
    code_identity = {"git_sha": "git-a", "git_dirty": False, "git_dirty_fingerprint": None}
    cell = {
        "schema_version": 2, "route": route, "scale_knob": scale_knob, "label": label,
        "seed": seed, "dataset": "synthetic", "data_seed": 3, "max_tokens": None,
        "config_sha256": canonical_json_sha256(config),
        "code_identity": code_identity,
        "data_sources": {split: {"format": "pt", "size_bytes": len(split), "sha256": split[0] * 64}
                         for split in ("train", "validation", "test")},
    }
    digest = canonical_json_sha256(cell)
    cell["reuse_contract_sha256"] = digest
    metrics = {"n_params": n_params, "test_ce": test_ce, "test_ppl": math.exp(test_ce),
               "test_bits_per_token": test_ce / math.log(2.0), "test_bpc": None}
    (run / "summary.json").write_text(json.dumps({
        **metrics, "best_val_ppl": math.exp(test_ce), "wall_time_s": 1.0,
        "scaling_reuse_contract_sha256": digest,
        "scaling_point": {**metrics, "embed_dim": embed_dim, "n_heads": 2, "n_gen": embed_dim,
                          "gauge_group": "block_glk", "n_layers": 1, "n_e_steps": 1,
                          "tokens_seen": tokens_seen},
    }), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"dataset": "synthetic", "config": config}),
                                     encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed, **code_identity, "train_data_sha256": "train-a", "val_data_sha256": "val-a",
        "test_data_sha256": "test-a", "data_sha256": "test-a",
    }), encoding="utf-8")
    path = root / "scaling_design.json"
    design = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": 1, "routes": [], "seeds": [], "status": "complete", "cells": []}
    design["routes"] = sorted({*design["routes"], route})
    design["seeds"] = sorted({*design["seeds"], seed})
    design["cells"].append({"route": route, "label": label, "seed": seed, "scale_knob": scale_knob,
                            "run_dir": f"{route}/{label}/s{seed}", "status": "complete"})
    path.write_text(json.dumps(design), encoding="utf-8")


# =============================================================================
# general-C-analysis-tests#4: kl_max = inf cells validate and are harvested
# =============================================================================

def test_kl_max_inf_cell_is_harvested_and_complete(tmp_path):
    _write_scaling_run(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0,
                       kl_max=math.inf)
    rows = sa.harvest(tmp_path)
    assert len(rows) == 1
    design = sa._requested_design(tmp_path, rows)
    assert [cell["status"] for cell in design["cells"]] == ["complete"]


def test_failed_validation_is_logged_not_silent(tmp_path, caplog):
    _write_scaling_run(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0)
    cell_path = tmp_path / "width" / "K8" / "s1" / "scaling_cell.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["config_sha256"] = "0" * 64
    cell_path.write_text(json.dumps(cell), encoding="utf-8")
    with caplog.at_level("WARNING", logger="scaling_analysis"):
        assert sa.harvest(tmp_path) == []
    assert "failed generation-contract validation" in caplog.text


# =============================================================================
# gap-stats#3: identical cells under two routes are one set of observations
# =============================================================================

def test_identical_runs_share_an_effective_run_key_across_int_and_float(tmp_path):
    _write_scaling_run(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0,
                       kl_max=160)
    _write_scaling_run(tmp_path, route="depth", label="L1", seed=1, n_params=100, test_ce=4.0,
                       kl_max=160.0, scale_knob="n_layers")
    _write_scaling_run(tmp_path, route="width", label="K12", seed=1, n_params=150, test_ce=3.9,
                       embed_dim=12, kl_max=160.0)
    rows = {(r["route"], r["label"]): r for r in sa.harvest(tmp_path)}
    assert rows[("width", "K8")]["effective_run_key"] == rows[("depth", "L1")]["effective_run_key"]
    assert rows[("width", "K8")]["effective_run_key"] != rows[("width", "K12")]["effective_run_key"]


def test_collapse_identical_points_keeps_one_and_reports_the_rest():
    points = [
        {"route": "depth", "label": "L6", "n_params": 900, "effective_run_keys": ("a", "b")},
        {"route": "width", "label": "K160", "n_params": 900, "effective_run_keys": ("a", "b")},
        {"route": "width", "label": "K20", "n_params": 100, "effective_run_keys": ("c", "d")},
        {"route": "legacy", "label": "X", "n_params": 50, "effective_run_keys": ()},
    ]
    kept, collapsed = sa._collapse_identical_points(points)
    assert [(p["route"], p["label"]) for p in kept] == [
        ("depth", "L6"), ("width", "K20"), ("legacy", "X")]
    assert collapsed == [{"kept": "depth/L6", "dropped_as_identical": ["width/K160"],
                          "n_seed_runs": 2}]


# =============================================================================
# gap-stats#5: max_steps cells never enter L(N); gap-stats#3 in analyze()
# =============================================================================

def test_analyze_excludes_budget_route_and_counts_identical_cells_once(tmp_path, monkeypatch):
    law = {8: 4.00, 12: 3.80, 16: 3.70, 20: 3.64}
    for seed, jitter in ((1, 0.01), (2, -0.01)):
        for k, ce in law.items():
            _write_scaling_run(tmp_path, route="width", label=f"K{k}", seed=seed, n_params=10 * k * k,
                               test_ce=ce + jitter, embed_dim=k)
        # The same K20 run reached through a second route (identical config, seed, data, code).
        _write_scaling_run(tmp_path, route="width_depth", label="K20_L1", seed=seed,
                           n_params=4000, test_ce=law[20] + jitter, embed_dim=20,
                           scale_knob="embed_dim+n_layers")
        for steps, tokens in ((100, 500), (200, 2000)):
            _write_scaling_run(tmp_path, route="train_steps", label=f"steps{steps}", seed=seed,
                               n_params=640, test_ce=4.2 - steps / 1000 + jitter, embed_dim=8,
                               scale_knob="max_steps", tokens_seen=tokens,
                               max_steps=steps)
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(tmp_path))
    monkeypatch.setitem(sa.CONFIG, "with_offset", False)
    monkeypatch.setitem(sa.CONFIG, "n_bootstrap", 20)
    figure_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(sa, "_make_figures", lambda *a, **k: figure_calls.append(k))
    sa.analyze()
    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["n_budget_points"] == 2
    assert summary["budget_points_excluded_from_fit"] == ["train_steps/steps100",
                                                         "train_steps/steps200"]
    assert summary["provenance"]["token_budget_varies"] is False
    assert "token_budget_varies" not in summary["pooled_fit_confounds"]
    assert summary["collapsed_identical_cells"] == [
        {"kept": "width/K20", "dropped_as_identical": ["width_depth/K20_L1"], "n_seed_runs": 2}]
    assert summary["pooled_fit"]["n_points"] == 4
    assert summary["pooled_fit_lack_of_fit"]["df_pe"] == 4          # 8 runs at 4 sizes, not 10 at 4
    assert [p["route"] for p in figure_calls[0]["budget_points"]] == ["train_steps", "train_steps"]


# =============================================================================
# gap-stats#9: capacity axes come from scale_knob
# =============================================================================

def test_axis_routes_follow_scale_knob_and_prefer_standard_routes():
    points = [
        {"route": "width", "scale_knob": "embed_dim"},
        {"route": "grow_K", "scale_knob": "embed_dim"},
        {"route": "depth", "scale_knob": "n_layers"},
        {"route": "inference", "scale_knob": "n_e_steps"},
    ]
    assert sa._axis_routes(points) == {"embed_dim": "width", "n_layers": "depth"}
    assert sa._axis_routes([{"route": "grow_K", "scale_knob": "embed_dim"}]) == {
        "embed_dim": "grow_K"}


# =============================================================================
# gap-stats#2: the frontier test uses the headline offset law
# =============================================================================

def _offset_points(route: str, sizes: np.ndarray, E: float, noise: np.ndarray) -> List[Dict[str, Any]]:
    return [{"route": route, "label": f"{route}{i}", "n_params": float(n),
             "ce_mean": float(E + 60.0 * n ** -0.3 + eps)}
            for i, (n, eps) in enumerate(zip(sizes, noise))]


_NOISE = np.array([0.004, -0.003, 0.002, -0.004])


def test_offset_frontier_does_not_reject_one_law_over_disjoint_ranges():
    points = (_offset_points("width", np.array([1e5, 2e5, 4e5, 8e5]), 3.0, _NOISE)
              + _offset_points("depth", np.array([1.6e7, 3.2e7, 6.4e7, 1.28e8]), 3.0, -_NOISE))
    loglog = sa.ancova_frontier_collapse(points)
    offset = sa.ancova_frontier_collapse(points, with_offset=True)
    assert loglog["testable"] and loglog["collapses"] is False       # the documented defect
    assert offset["testable"] and offset["form"] == "offset_power_law"
    assert (offset["df1"], offset["df2"]) == (3, 2)
    assert offset["collapses"] is None and offset["p_value"] > 0.05


def test_offset_frontier_rejects_routes_on_different_laws():
    sizes = np.array([1e5, 2e5, 4e5, 8e5, 1.6e6])
    noise = np.array([0.002, -0.001, 0.001, -0.002, 0.001])
    points = (_offset_points("width", sizes, 3.0, noise)
              + _offset_points("depth", sizes * 1.5, 3.4, -noise))
    offset = sa.ancova_frontier_collapse(points, with_offset=True)
    assert offset["testable"] and offset["collapses"] is False


def test_offset_frontier_needs_four_sizes_per_route():
    points = (_offset_points("width", np.array([1e5, 2e5, 4e5, 8e5]), 3.0, _NOISE)
              + _offset_points("depth", np.array([1e6, 2e6, 4e6]), 3.0, _NOISE[:3]))
    offset = sa.ancova_frontier_collapse(points, with_offset=True)
    assert offset["testable"] is False and "depth" in offset["reason"]


# =============================================================================
# gap-stats#1: offset-law CI from the pooled pure-error posterior
# =============================================================================

def _ci_points(seeds: List[np.ndarray]) -> List[Dict[str, Any]]:
    sizes = [2e6, 4e6, 8e6, 1.6e7]
    return [{"n_params": n, "ce_seeds": list(3.0 + 60.0 * n ** -0.3 + s)} for n, s in zip(sizes, seeds)]


def test_offset_ci_is_finite_and_brackets_the_estimate():
    rng = np.random.default_rng(5)
    points = _ci_points([rng.normal(0.0, 0.01, 3) for _ in range(4)])
    alpha, lo, hi = sa.bootstrap_exponent_ci(points, n_boot=60, with_offset=True)
    assert np.isfinite(alpha) and np.isfinite(lo) and np.isfinite(hi)
    assert lo < alpha < hi


def test_offset_ci_widens_with_the_pooled_seed_variance():
    base = [np.array([0.01, -0.01, 0.0]) * scale for scale in (1.0, 0.5, 1.5, 1.0)]
    _a1, lo1, hi1 = sa.bootstrap_exponent_ci(_ci_points(base), n_boot=60, with_offset=True)
    _a2, lo2, hi2 = sa.bootstrap_exponent_ci(_ci_points([2.0 * s for s in base]), n_boot=60,
                                             with_offset=True)
    assert (hi2 - lo2) > (hi1 - lo1)


def test_offset_ci_withheld_without_replicate_seeds():
    points = _ci_points([np.array([0.0])] * 4)
    alpha, lo, hi = sa.bootstrap_exponent_ci(points, n_boot=10, with_offset=True)
    assert np.isfinite(alpha) and math.isnan(lo) and math.isnan(hi)


# =============================================================================
# multiseed_analysis
# =============================================================================

def test_fingerprint_ignores_int_versus_float_in_float_fields(tmp_path):
    for name, value in (("a", 1), ("b", 1.0)):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(
            {"config": serialize_config(test_config(seed=1, rope_fraction=value))}))
    raw = json.loads((tmp_path / "a" / "config.json").read_text())["config"]["rope_fraction"]
    assert type(raw) is int                                  # the int survives serialization
    assert (ms._config_fingerprint(tmp_path / "a" / "config.json")
            == ms._config_fingerprint(tmp_path / "b" / "config.json"))


def test_per_seed_data_order_panel_aggregates_with_nonshared_status(tmp_path):
    _write_request(tmp_path, [6, 54, 271])
    for seed, ppl in ((6, 10.0), (54, 11.0), (271, 12.0)):
        d = tmp_path / f"run_s{seed}"
        _write_full_run(d, ppl=ppl, seed=seed)
        prov = json.loads((d / "provenance.json").read_text())
        prov["data_seed"] = seed                              # DATA_SEED=None -> data_seed == seed
        (d / "provenance.json").write_text(json.dumps(prov))
    design = ms._requested_seed_design(tmp_path)
    assert design["complete"] is True
    assert design["data_order"]["status"] == "nonshared_or_mixed"
    assert ms.aggregate_scalar(tmp_path, "test_ppl")["n"] == 3


def _main(tmp_path, monkeypatch, scalar_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    emitted: List[Any] = []
    monkeypatch.setitem(ms.CONFIG, "run_root", str(tmp_path))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setattr(ms, "SCALAR_KEYS", scalar_keys or ["test_ppl"])
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: emitted.append(args))
    code = ms.main()
    summary = json.loads((tmp_path / "multiseed_summary.json").read_text(encoding="utf-8"))
    summary["_exit"] = code
    summary["_emitted"] = len(emitted)
    return summary


def test_column_in_one_seed_only_does_not_withhold_the_digest(tmp_path, monkeypatch):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=10.0, seed=1,
                    metrics="step,train_ce,log_row_ms\n100,5.0,3.0\n200,4.0,3.1\n")
    _write_full_run(tmp_path / "b", ppl=12.0, seed=2,
                    metrics="step,train_ce\n100,6.0\n200,5.0\n")
    summary = _main(tmp_path, monkeypatch)
    assert summary["_exit"] == 0 and summary["_emitted"] == 1
    assert summary["design"]["complete"] is True
    assert summary["scalars"]["test_ppl"]["mean"] == pytest.approx(11.0)
    assert summary["curves_final_step"]["train_ce"]["mean"] == pytest.approx(4.5)
    assert "log_row_ms" in summary["diagnostics"]["withheld_curves"]


def test_one_nonfinite_diagnostic_cell_withholds_only_that_column(tmp_path, monkeypatch):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=10.0, seed=1,
                    metrics="step,train_ce,attn_entropy\n100,5.0,1.0\n200,4.0,1.1\n")
    _write_full_run(tmp_path / "b", ppl=12.0, seed=2,
                    metrics="step,train_ce,attn_entropy\n100,6.0,nan\n200,5.0,1.2\n")
    summary = _main(tmp_path, monkeypatch)
    assert summary["design"]["complete"] is True and summary["_exit"] == 0
    assert "attn_entropy" not in summary["curves_final_step"]
    assert "non-finite" in summary["diagnostics"]["withheld_curves"]["attn_entropy"]
    assert summary["scalars"]["test_ppl"]["n"] == 2


def test_partial_scalar_is_withheld_alone(tmp_path, monkeypatch):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=10.0, seed=1, metrics="step,train_ce\n100,5.0\n",
                    research={"ece": 0.1})
    _write_full_run(tmp_path / "b", ppl=12.0, seed=2, metrics="step,train_ce\n100,6.0\n")
    summary = _main(tmp_path, monkeypatch, scalar_keys=["test_ppl", "ece"])
    assert summary["design"]["complete"] is True
    assert set(summary["scalars"]) == {"test_ppl"}
    assert summary["diagnostics"]["partial_scalars"] == ["ece"]


def test_cv_is_kept_only_for_positive_ratio_scale_metrics():
    signed = ms._ratio_scale_cv("sigma_ce_spearman",
                                ms._summarize([0.01, -0.02, 0.015]))
    ratio = ms._ratio_scale_cv("test_ppl", ms._summarize([10.0, 11.0, 12.0]))
    nonpositive = ms._ratio_scale_cv("test_ce", ms._summarize([1.0, -1.0, 2.0]))
    assert math.isnan(signed["cv"]) and "ratio-scale" in signed["cv_withheld"]
    assert math.isfinite(signed["sd"])
    assert ratio["cv"] == pytest.approx(1.0 / 11.0)
    assert math.isnan(nonpositive["cv"])


