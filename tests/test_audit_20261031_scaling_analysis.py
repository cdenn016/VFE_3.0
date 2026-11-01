r"""Audit 2026-10-31, package scaling-analysis: the scaling-law statistics of ``scaling_analysis``.

analysis-statistics-1 (A039): every fitted point is weighted by :math:`n_i / s^2` with the pooled pure-error
variance of log CE, the variance model of the lack-of-fit test, the offset bootstrap and the frontier test, so
both F-tests reject a true law at their nominal rate. analysis-statistics-4 (A072): the plain-law exponent
interval keeps the size design fixed and draws the size means from the pooled pure-error posterior, as the
offset law does. analysis-statistics-5 (A110): the headline figures fit the deduplicated points with the
summary's weights. analysis-statistics-8 (A350): the analysis also fits the law to the test CE at the
validation-selected weights and flags cells whose final weights sit further from their best validation point
than their seed spread. analysis-statistics-9 (A351): the compute figure plots the measured training FLOPs,
falling back to the analytic proxy and then to 6ND with labels that say so.

Every fixture is a synthetic point list or a tiny JSON run directory; no model is built.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

import scaling_analysis as sa
from tests.profiles import test_config
from vfe3.artifact_utils import canonical_json_sha256
from vfe3.config import serialize_config
from vfe3.viz import figures as figs

pytest.importorskip("scipy")


# =============================================================================
# fixtures
# =============================================================================

_V = 50257
_K = np.array([20, 40, 80, 120, 160], dtype=float)
_N = _V * (_K ** 2 / 5.0 + 3.0 * _K)                   # the five shipped width sizes of scaling.py


def _points(
    rng:       np.random.Generator,
    ce_true:   np.ndarray,
    sigma_log: float,

    *,
    route:     str = "width",
    sizes:     np.ndarray = _N,
    seeds:     int = 3,
) -> List[Dict[str, Any]]:
    r"""Seed replicates of an exact law with a common log-space standard deviation ``sigma_log``."""
    points = []
    for i, (n, c) in enumerate(zip(sizes, ce_true)):
        ce = np.exp(np.log(c) + sigma_log * rng.standard_normal(seeds))
        points.append({"route": route, "label": f"{route}{i}", "n_params": float(n),
                       "ce_seeds": [float(v) for v in ce], "ce_mean": float(ce.mean()),
                       "ce_sem": float(ce.std(ddof=1) / math.sqrt(seeds)) if seeds > 1 else 0.0})
    return points


def _analyzer_weights(points: Sequence[Dict[str, Any]]) -> np.ndarray:
    r"""The weights ``analyze`` gives its pooled points (its own call of ``_scaling_sem_weights``)."""
    means = np.array([p["ce_mean"] for p in points], dtype=float)
    sems = np.array([p["ce_sem"] for p in points], dtype=float)
    pooled = figs._scaling_pooled_log_variance([p["ce_seeds"] for p in points])
    return figs._scaling_sem_weights(means, sems, n_seeds=[len(p["ce_seeds"]) for p in points],
                                     pooled_log_variance=pooled)


def _pooled_variance_weights(seed_groups: Sequence[Sequence[float]]) -> np.ndarray:
    r"""Independent reference: :math:`n_i / s^2` with :math:`s^2 = \sum_{ij} (y_{ij} - \bar y_i)^2 / (n - m)`."""
    logs = [np.log(np.asarray(g, dtype=float)) for g in seed_groups]
    ss = sum(float(((y - y.mean()) ** 2).sum()) for y in logs)
    s2 = ss / (sum(y.size for y in logs) - len(logs))
    return np.array([y.size / s2 for y in logs], dtype=float)


def _write_cell(
    root:          Path,

    *,
    route:         str,
    label:         str,
    seed:          int,
    n_params:      int,
    test_ce:       float,
    embed_dim:     int = 8,
    n_layers:      int = 1,
    scale_knob:    str = "embed_dim",
    tokens_seen:   int = 1000,
    summary_extra: Optional[Dict[str, Any]] = None,
    point_extra:   Optional[Dict[str, Any]] = None,
) -> None:
    r"""One schema-2 scaling cell laid out as ``scaling.py`` writes it, plus its design-manifest entry."""
    run = root / route / label / f"s{seed}"
    run.mkdir(parents=True)
    cfg = test_config(seed=seed, vocab_size=16, embed_dim=embed_dim, n_heads=2, max_seq_len=8,
                      n_layers=n_layers, n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk")
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
    summary = {
        **metrics, "best_val_ppl": math.exp(test_ce), "wall_time_s": 1.0,
        "scaling_reuse_contract_sha256": digest,
        "scaling_point": {**metrics, "embed_dim": embed_dim, "n_heads": 2, "n_gen": embed_dim,
                          "gauge_group": "block_glk", "n_layers": n_layers, "n_e_steps": 1,
                          "tokens_seen": tokens_seen, **(point_extra or {})},
        **(summary_extra or {}),
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"dataset": "synthetic", "config": config}), encoding="utf-8")
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


def _analyze(root: Path, monkeypatch: pytest.MonkeyPatch, *, with_offset: bool,
             figures: bool = False) -> Dict[str, Any]:
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(root))
    monkeypatch.setitem(sa.CONFIG, "with_offset", with_offset)
    monkeypatch.setitem(sa.CONFIG, "n_bootstrap", 20)
    if not figures:
        monkeypatch.setattr(sa, "_make_figures", lambda *args, **kwargs: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa.analyze()
    return json.loads((root / "scaling_summary.json").read_text(encoding="utf-8"))


_WIDTHS = (8, 12, 16, 20, 24)
# Log-space seed offsets: the K12 seeds nearly agree, which gives K12 the largest own-SEM weight.
_OFFSETS = {8: (0.010, -0.012, 0.002), 12: (0.0005, -0.0004, -0.0001), 16: (-0.008, 0.011, -0.003),
            20: (0.006, -0.009, 0.004), 24: (-0.010, 0.007, 0.003)}


def _width_law(k: int) -> float:
    r"""A plain law with a bump at K12 (lack of fit), so the weighting changes the fitted exponent."""
    return 6.0 * (10.0 * k * k) ** -0.08 * (1.006 if k == 12 else 1.0)


# =============================================================================
# analysis-statistics-1: pooled pure-error weights calibrate both F-tests
# =============================================================================

@pytest.mark.parametrize("with_offset, reps", [(True, 300), (False, 1000)])
def test_lack_of_fit_rejects_a_true_law_at_the_nominal_rate(with_offset: bool, reps: int) -> None:
    rng = np.random.default_rng(1)
    if with_offset:
        ce_true = 3.0 + 2.5 * _N[0] ** 0.3 * _N ** -0.3          # CE 5.5 to 3.83 at the five sizes
    else:
        ce_true = 5.5 * _N[0] ** 0.08 * _N ** -0.08
    rejected = tested = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(reps):
            points = _points(rng, ce_true, 0.001)
            fit = figs._fit_power_law(_N, [p["ce_mean"] for p in points], weights=_analyzer_weights(points),
                                      with_offset=with_offset)
            lof = sa._lack_of_fit_test(points, fit)
            tested += int(lof["testable"])
            rejected += int(lof["rejected"] is True)
    assert tested == reps
    assert rejected / tested < 0.08                             # the per-point SEM weights gave 0.11 to 0.13


def test_frontier_collapse_rejects_one_shared_law_at_the_nominal_rate() -> None:
    rng = np.random.default_rng(1)
    A = 2.5 * _N[0] ** 0.3
    rejected = tested = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(250):
            points: List[Dict[str, Any]] = []
            for route, sizes in (("width", _N), ("blocks", 1.45 * _N)):
                points += _points(rng, 3.0 + A * sizes ** -0.3, 0.003, route=route, sizes=sizes)
            out = sa.ancova_frontier_collapse(points, min_points=2, with_offset=True)
            tested += int(out.get("testable") is True)
            rejected += int(out.get("null_rejected") is True)
    assert tested == 250
    assert rejected / tested < 0.08                             # the per-point SEM weights gave 0.118


def test_sem_weights_give_every_point_its_pooled_variance() -> None:
    points = [{"ce_seeds": [4.00, 4.04, 3.97]}, {"ce_seeds": [3.80, 3.8001, 3.7999]}, {"ce_seeds": [3.45]}]
    means = np.array([np.mean(p["ce_seeds"]) for p in points])
    sems = np.array([np.std(p["ce_seeds"], ddof=1) / np.sqrt(len(p["ce_seeds"])) if len(p["ce_seeds"]) > 1
                     else 0.0 for p in points])
    s2 = figs._scaling_pooled_log_variance([p["ce_seeds"] for p in points])
    w = figs._scaling_sem_weights(means, sems, n_seeds=[3, 3, 1], pooled_log_variance=s2)
    np.testing.assert_allclose(w, np.array([3.0, 3.0, 1.0]) / s2, rtol=1e-12)
    # Without the pooled variance the delta-method weight of each point's own SEM is unchanged.
    np.testing.assert_allclose(figs._scaling_sem_weights(means, sems)[:2], (means[:2] / sems[:2]) ** 2,
                               rtol=1e-12)


def test_analyze_weights_every_point_by_the_pooled_pure_error(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: Dict[int, List[float]] = {}
    for k in _WIDTHS:
        seeds[k] = [_width_law(k) * math.exp(offset) for offset in _OFFSETS[k]]
        for seed, ce in zip((1, 2, 3), seeds[k]):
            _write_cell(tmp_path, route="width", label=f"K{k}", seed=seed, n_params=10 * k * k,
                        test_ce=ce, embed_dim=k)
    summary = _analyze(tmp_path, monkeypatch, with_offset=False)
    N = np.array([10.0 * k * k for k in _WIDTHS])
    means = np.array([float(np.mean(seeds[k])) for k in _WIDTHS])
    pooled = figs._fit_power_law(N, means, weights=_pooled_variance_weights([seeds[k] for k in _WIDTHS]))
    sems = np.array([float(np.std(seeds[k], ddof=1)) / math.sqrt(3) for k in _WIDTHS])
    own_sem = figs._fit_power_law(N, means, weights=(means / sems) ** 2)
    assert abs(pooled["alpha"] - own_sem["alpha"]) > 1e-4         # the fixture separates the two weightings
    assert summary["pooled_fit"]["alpha"] == pytest.approx(pooled["alpha"], rel=1e-12, abs=0.0)


# =============================================================================
# analysis-statistics-4: the plain-law interval keeps the design fixed
# =============================================================================

def _plain_points(offsets: Dict[int, Sequence[float]]) -> List[Dict[str, Any]]:
    out = []
    for i, n in enumerate(_N):
        ce = 5.5 * _N[0] ** 0.08 * n ** -0.08 * np.exp(np.asarray(offsets[i], dtype=float))
        out.append({"route": "width", "label": f"w{i}", "n_params": float(n), "ce_seeds": [float(v) for v in ce],
                    "ce_mean": float(ce.mean()),
                    "ce_sem": float(ce.std(ddof=1) / math.sqrt(ce.size)) if ce.size > 1 else 0.0})
    return out


_PLAIN_OFFSETS = {0: (0.003, -0.002, 0.001), 1: (-0.004, 0.002, 0.003), 2: (0.001, 0.002, -0.004),
                  3: (0.002, -0.003, -0.001), 4: (-0.001, 0.004, -0.002)}


def test_plain_law_interval_is_the_pure_error_t_interval() -> None:
    from scipy.stats import t as t_dist

    points = _plain_points(_PLAIN_OFFSETS)
    w = _analyzer_weights(points)
    alpha, lo, hi = sa.bootstrap_exponent_ci(points, weights=w, n_boot=4000, with_offset=False)
    # With weights n_i / s^2 the exponent is a linear functional of the log means, so the posterior draw
    # reproduces the Student-t interval on n - m pure-error degrees of freedom.
    x = np.log(_N)
    n_i = np.array([len(p["ce_seeds"]) for p in points], dtype=float)
    s2 = figs._scaling_pooled_log_variance([p["ce_seeds"] for p in points])
    x_bar = float((n_i * x).sum() / n_i.sum())
    se = math.sqrt(s2 / float((n_i * (x - x_bar) ** 2).sum()))
    half = float(t_dist.ppf(0.975, int(n_i.sum()) - len(points))) * se
    assert alpha - lo == pytest.approx(half, rel=0.1)
    assert hi - alpha == pytest.approx(half, rel=0.1)


def test_plain_law_interval_covers_at_the_nominal_rate() -> None:
    r"""Coverage of the shipped call (2000 draws) on the five-size, three-seed design.

    With weights :math:`n_i / s^2` every replicate slope is :math:`\hat\alpha + \mathrm{SE}\, T^*` with
    :math:`\mathrm{SE} = s / \sqrt{\sum_i n_i (x_i - \bar x)^2}` and :math:`T^*` built from the standard normal
    and chi-square draws alone, and the generator is seeded with 0, so every dataset of one design gets the
    interval :math:`[\hat\alpha - a\,\mathrm{SE}, \hat\alpha + b\,\mathrm{SE}]` with the same (a, b). Since
    :math:`(\hat\alpha - \alpha) / \mathrm{SE} \sim t_{n-m}`, the coverage is
    :math:`F_{n-m}(a) - F_{n-m}(-b)` exactly, without a Monte Carlo loop over datasets."""
    from scipy.stats import t as t_dist

    x = np.log(_N)
    sxx = float((3.0 * (x - x.mean()) ** 2).sum())
    ce_true = 5.5 * _N[0] ** 0.08 * _N ** -0.08
    standardized = []
    for data_seed in (21, 22, 23):
        points = _points(np.random.default_rng(data_seed), ce_true, 0.003)
        se = math.sqrt(figs._scaling_pooled_log_variance([p["ce_seeds"] for p in points]) / sxx)
        alpha, lo, hi = sa.bootstrap_exponent_ci(points, weights=_analyzer_weights(points), n_boot=2000,
                                                 with_offset=False)
        standardized.append(((alpha - lo) / se, (hi - alpha) / se))
    a, b = standardized[0]
    for other in standardized[1:]:
        assert other == pytest.approx((a, b), rel=1e-6)          # one (a, b) for every dataset of the design
    coverage = float(t_dist.cdf(a, 15 - 5) - t_dist.cdf(-b, 15 - 5))
    # The pairs bootstrap's half-widths were about 1.45 times the t quantile and covered 0.995 to 0.998. At 200
    # to 300 draws the same seed-0 stream gives a and b near 1.6 and 2.1 and a coverage near 0.90.
    assert 0.94 <= coverage <= 0.96


def test_solver_fallback_replicates_refit_the_plain_law(monkeypatch: pytest.MonkeyPatch) -> None:
    points = _plain_points(_PLAIN_OFFSETS)
    w = _analyzer_weights(points)
    plain = sa.bootstrap_exponent_ci(points, weights=w, n_boot=200, with_offset=False)
    real_fit = figs._fit_power_law
    calls = {"n": 0}

    def point_fit_falls_back(N: Any, L: Any, *, weights: Any = None, with_offset: bool = False) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1 and with_offset:                      # only the point estimate's solve fails
            fit = real_fit(N, L, weights=weights, with_offset=False)
            fit["form"] = "power_law_fallback_solver"
            return fit
        return real_fit(N, L, weights=weights, with_offset=with_offset)

    monkeypatch.setattr(figs, "_fit_power_law", point_fit_falls_back)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fallback = sa.bootstrap_exponent_ci(points, weights=w, n_boot=200, with_offset=True)
    # Every replicate refits the reported plain-law estimator; none is selected by its own offset solve.
    assert fallback == plain and all(np.isfinite(value) for value in fallback)


def test_plain_law_interval_without_replicates_is_the_residual_t_interval() -> None:
    from scipy.stats import t as t_dist

    points = _plain_points({i: (offsets[0],) for i, offsets in _PLAIN_OFFSETS.items()})
    alpha, lo, hi = sa.bootstrap_exponent_ci(points, n_boot=50, with_offset=False)
    x, y = np.log(_N), np.log([p["ce_mean"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    se = math.sqrt(float(residual @ residual) / (x.size - 2) / float(((x - x.mean()) ** 2).sum()))
    half = float(t_dist.ppf(0.975, x.size - 2)) * se
    assert alpha == pytest.approx(-slope, rel=1e-12)
    assert (lo, hi) == pytest.approx((alpha - half, alpha + half), rel=1e-9)
    # The offset law still withholds its interval without replicate seeds.
    assert all(math.isnan(v) for v in sa.bootstrap_exponent_ci(points, n_boot=10, with_offset=True)[1:])


# =============================================================================
# analysis-statistics-5: the headline figures fit the summary's estimator
# =============================================================================

def _shared_route_sweep(root: Path) -> None:
    r"""``width`` K{k} is ``width_depth`` K{k}_L2: one set of training runs reached through two routes."""
    for k in _WIDTHS:
        for depth in (1, 2):
            n = 100 * k * k + 1000 * k * depth
            for seed in (1, 2, 3):
                ce = (2.0 + 30.0 * n ** -0.3) * math.exp(0.004 * (((7 * seed + 3 * k + depth) % 5) - 2))
                _write_cell(root, route="width_depth", label=f"K{k}_L{depth}", seed=seed, n_params=n,
                            test_ce=ce, embed_dim=k, n_layers=depth, scale_knob="embed_dim+n_layers")
                if depth == 2:
                    _write_cell(root, route="width", label=f"K{k}", seed=seed, n_params=n, test_ce=ce,
                                embed_dim=k, n_layers=depth)


@pytest.mark.parametrize("with_offset", [False, True])
def test_headline_figures_fit_the_summary_estimator_when_routes_share_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_offset: bool) -> None:
    _shared_route_sweep(tmp_path)
    real_fit, real_law, real_routes = figs._fit_power_law, figs.plot_scaling_law, figs.plot_scaling_routes
    state: Dict[str, Optional[str]] = {"figure": None}
    fits: Dict[str, List[Dict[str, Any]]] = {}

    def recording_fit(N: Any, L: Any, *, weights: Any = None, with_offset: bool = False) -> Dict[str, Any]:
        out = real_fit(N, L, weights=weights, with_offset=with_offset)
        if state["figure"] is not None:
            fits.setdefault(state["figure"], []).append({"n": int(np.asarray(N).size), **out})
        return out

    def tagged(builder: Any) -> Any:
        def run(*args: Any, **kwargs: Any) -> Any:
            state["figure"] = Path(kwargs["path"]).name
            try:
                return builder(*args, **kwargs)
            finally:
                state["figure"] = None
        return run

    monkeypatch.setattr(figs, "_fit_power_law", recording_fit)
    monkeypatch.setattr(figs, "plot_scaling_law", tagged(real_law))
    monkeypatch.setattr(figs, "plot_scaling_routes", tagged(real_routes))
    monkeypatch.setattr(figs, "_save", lambda fig, path=None: fig)
    summary = _analyze(tmp_path, monkeypatch, with_offset=with_offset, figures=True)

    assert summary["n_pooled_param_points"] == 10 and summary["n_fitted_param_points"] == 15
    headline = fits["scaling_ce_vs_params.png"]
    assert len(headline) == 1 and headline[0]["n"] == 10
    assert headline[0]["alpha"] == pytest.approx(summary["pooled_fit"]["alpha"], rel=1e-12, abs=0.0)
    overlay = fits["scaling_routes_overlay.png"]
    assert overlay[-1]["n"] == 10                                # the dashed pooled line
    assert overlay[-1]["alpha"] == pytest.approx(summary["pooled_fit"]["alpha"], rel=1e-12, abs=0.0)
    per_route = sorted(fit["alpha"] for fit in overlay[:-1])
    assert per_route == pytest.approx(sorted(d["alpha"] for d in summary["per_route"].values()), rel=1e-12)


# =============================================================================
# analysis-statistics-8: the validation-selected weights beside the final weights
# =============================================================================

def _selection_sweep(root: Path, *, drop_best_ce_for: Optional[tuple] = None) -> Dict[int, List[float]]:
    r"""Five widths x three seeds. K24 passed its best validation point 0.05 nats before the end; every other
    cell's best step is its final step (K20 also records its final validation PPL)."""
    best_ce: Dict[int, List[float]] = {}
    for k in _WIDTHS:
        best_ce[k] = []
        for seed, offset in zip((1, 2, 3), _OFFSETS[k]):
            ce = _width_law(k) * math.exp(offset)
            final_val_ce = _width_law(k) + 0.01 * (seed - 2)       # seed SD of the final validation CE: 0.01
            extra: Dict[str, Any] = {"evaluated_step": 1000}
            if k == 24:
                extra.update(best_val_ppl=math.exp(final_val_ce - 0.05), final_val_ppl=math.exp(final_val_ce),
                             best_step=800, test_best_ce=ce - 0.03)
            else:
                extra.update(best_val_ppl=math.exp(final_val_ce), best_step=1000, test_best_ce=ce)
                if k == 20:
                    extra["final_val_ppl"] = math.exp(final_val_ce)
            if drop_best_ce_for == (k, seed):
                extra["test_best_ce"] = None
            if extra["test_best_ce"] is not None:
                best_ce[k].append(extra["test_best_ce"])
            _write_cell(root, route="width", label=f"K{k}", seed=seed, n_params=10 * k * k, test_ce=ce,
                        embed_dim=k, summary_extra=extra)
    return best_ce


def test_best_weight_fit_and_selection_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    best_ce = _selection_sweep(tmp_path)
    summary = _analyze(tmp_path, monkeypatch, with_offset=False)
    N = np.array([10.0 * k * k for k in _WIDTHS])
    means = np.array([float(np.mean(best_ce[k])) for k in _WIDTHS])
    expected = figs._fit_power_law(N, means, weights=_pooled_variance_weights([best_ce[k] for k in _WIDTHS]))
    best = summary["pooled_fit_best_weights"]
    assert best["status"] == "fitted" and best["n_points"] == 5
    assert best["alpha"] == pytest.approx(expected["alpha"], rel=1e-12, abs=0.0)
    assert best["alpha"] != pytest.approx(summary["pooled_fit"]["alpha"], rel=1e-6)

    selection = summary["final_vs_best_validation"]
    assert selection["flagged"] == ["width/K24"]
    assert selection["n_cells"] == 5 and selection["n_assessed"] == 5
    cells = {cell["cell"]: cell for cell in selection["cells"]}
    assert cells["width/K24"]["mean_gap"] == pytest.approx(0.05, rel=1e-9)
    assert cells["width/K24"]["seed_sd"] == pytest.approx(0.01, rel=1e-9)
    assert cells["width/K8"]["mean_gap"] == 0.0 and cells["width/K8"]["flagged"] is False

    header = (tmp_path / "scaling_points.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"test_best_ce", "best_step", "final_val_ppl", "evaluated_step"} <= set(header)
    report = (tmp_path / "SCALING_ANALYSIS.md").read_text(encoding="utf-8")
    assert "validation-selected weights" in report and "width/K24" in report


def test_best_weight_fit_is_withheld_when_a_seed_lacks_its_test_at_best(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _selection_sweep(tmp_path, drop_best_ce_for=(16, 2))
    summary = _analyze(tmp_path, monkeypatch, with_offset=False)
    best = summary["pooled_fit_best_weights"]
    assert best["status"] == "withheld" and "width/K16" in best["reason"]
    assert summary["pooled_fit"]["alpha"] is not None


def test_selection_check_needs_the_final_validation_of_a_cell() -> None:
    rows = [{"best_val_ppl": math.exp(3.0), "best_step": 800, "evaluated_step": 1000},
            {"best_val_ppl": math.exp(3.1), "final_val_ppl": math.exp(3.2), "best_step": 800,
             "evaluated_step": 1000},
            {"best_val_ppl": math.exp(3.2), "best_step": 1000, "evaluated_step": 1000}]
    assert math.isnan(sa._final_validation_gap(rows[0]))      # final weights never validated
    assert sa._final_validation_gap(rows[1]) == pytest.approx(0.1, rel=1e-9)
    assert sa._final_validation_gap(rows[2]) == 0.0           # the best step is the final step


# =============================================================================
# analysis-statistics-9: the compute axis is the measured training FLOPs
# =============================================================================

def test_compute_axis_prefers_measured_then_analytic_then_6nd() -> None:
    full = {"est_train_flops_measured_3x": 3e12, "est_flops_analytic": 2e12, "est_flops_6ND": 6e14}
    assert sa._compute_axis([full, full])[0] == "est_train_flops_measured_3x"
    no_measured = {**full, "est_train_flops_measured_3x": float("nan")}
    assert sa._compute_axis([full, no_measured])[0] == "est_flops_analytic"
    only_6nd = {**no_measured, "est_flops_analytic": None}
    axis = sa._compute_axis([full, only_6nd])
    assert axis[0] == "est_flops_6ND" and "6ND" in axis[1]
    assert sa._compute_axis([{"est_flops_6ND": float("nan")}]) is None


@pytest.mark.parametrize("measured_everywhere", [True, False])
def test_compute_figure_plots_the_measured_flops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 measured_everywhere: bool) -> None:
    for k in _WIDTHS:
        for seed, offset in zip((1, 2, 3), _OFFSETS[k]):
            n = 10 * k * k
            extra = {"est_flops_6ND": 6 * n * 1000}
            if measured_everywhere or k != 16:
                extra["est_train_flops_measured_3x"] = 3.0 * (4.0 * 16 * k + 50.0 * k) * 1000
            _write_cell(tmp_path, route="width", label=f"K{k}", seed=seed, n_params=n,
                        test_ce=_width_law(k) * math.exp(offset), embed_dim=k, point_extra=extra)
    calls: Dict[str, Dict[str, Any]] = {}

    def record(points: List[Dict[str, Any]], **kwargs: Any) -> Any:
        calls[Path(kwargs["path"]).name] = {"points": points, **kwargs}
        return figs.plt.figure()

    monkeypatch.setattr(figs, "plot_scaling_law", record)
    monkeypatch.setattr(figs, "_save", lambda fig, path=None: fig)
    _analyze(tmp_path, monkeypatch, with_offset=False, figures=True)
    flops = calls["scaling_ce_vs_flops.png"]
    if measured_everywhere:
        assert flops["x_key"] == "est_train_flops_measured_3x" and "measured" in flops["xlabel"]
    else:
        assert flops["x_key"] == "est_flops_6ND" and "6ND" in flops["xlabel"]
    values = [p[flops["x_key"]] for p in flops["points"]]
    assert all(np.isfinite(values)) and len(values) == 5
    rows = sa.harvest(tmp_path)
    assert {"est_train_flops_measured_3x"} <= set(rows[0])
