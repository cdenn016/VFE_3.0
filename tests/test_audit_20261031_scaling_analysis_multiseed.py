r"""Audit 2026-10-31, package scaling-analysis: the multi-seed digest of ``multiseed_analysis``.

analysis-statistics-10 (A209): ``main`` builds the request design once and hands it to every aggregator, so a
three-seed group no longer rereads and revalidates every run once per scalar key (25 design builds and 77
config builds before). analysis-statistics-11 (A277): the seed-stability ranking compares only metrics on the
cross-entropy scale, since a perplexity is :math:`e^{CE}` and its CV is about CE times the CV of its CE, and
the docstrings state that seeds share a data order only under one explicit common ``DATA_SEED`` and no longer
give the ``grow_K`` route a stale size count.

Every fixture is a tiny JSON run directory; no model is built.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import multiseed_analysis as ms
from tests.test_multiseed import _write_full_run, _write_request
from vfe3.viz import figures as figs


def _group(root: Path, summaries: Dict[int, Dict[str, Any]]) -> None:
    _write_request(root, sorted(summaries))
    for seed, extra in sorted(summaries.items()):
        run = root / f"run_s{seed}"
        _write_full_run(run, ppl=float(extra.get("test_ppl", 10.0)), seed=seed,
                        metrics="step,train_ce\n100,5.0\n200,4.0\n",
                        per_layer="layer,self_coupling\n0,1.0\n1,1.5\n")
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        (run / "summary.json").write_text(json.dumps({**summary, **extra}), encoding="utf-8")


def _main(root: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    monkeypatch.setitem(ms.CONFIG, "run_root", str(root))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setitem(ms.CONFIG, "allow_mixed_code_identity", False)
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: None)
    assert ms.main() == 0
    return json.loads((root / "multiseed_summary.json").read_text(encoding="utf-8"))


_THREE_SEEDS = {6: {"test_ppl": 10.0}, 54: {"test_ppl": 11.0}, 271: {"test_ppl": 12.0}}


# =============================================================================
# analysis-statistics-10: one request design per digest
# =============================================================================

def test_main_builds_the_request_design_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _group(tmp_path, _THREE_SEEDS)
    counts = {"design": 0, "load_config": 0}
    real_design, real_load = ms._requested_seed_design, ms.load_config

    def counted_design(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        counts["design"] += 1
        return real_design(*args, **kwargs)

    def counted_load(*args: Any, **kwargs: Any) -> Any:
        counts["load_config"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(ms, "_requested_seed_design", counted_design)
    monkeypatch.setattr(ms, "load_config", counted_load)
    summary = _main(tmp_path, monkeypatch)                     # the shipped SCALAR_KEYS
    assert counts["design"] == 1
    assert counts["load_config"] <= 3 + 2                     # one per accepted seed, per-layer policy, fingerprint
    assert summary["scalars"]["test_ppl"]["mean"] == pytest.approx(11.0)
    assert summary["curves_final_step"]["train_ce"]["mean"] == pytest.approx(4.0)
    assert summary["per_layer"]["1"]["self_coupling"]["mean"] == pytest.approx(1.5)


def test_aggregators_given_the_design_match_their_own_build(tmp_path: Path) -> None:
    _group(tmp_path, _THREE_SEEDS)
    design = ms._requested_seed_design(tmp_path)
    for key in ("test_ppl", "not_recorded"):
        given = ms.aggregate_scalar(tmp_path, key, design=design)
        own = ms.aggregate_scalar(tmp_path, key)
        assert json.dumps(given, sort_keys=True, default=str) == json.dumps(own, sort_keys=True, default=str)
    np.testing.assert_equal(ms.aggregate_seed_curves(tmp_path, design=design), ms.aggregate_seed_curves(tmp_path))
    assert ms.aggregate_per_layer(tmp_path, design=design) == ms.aggregate_per_layer(tmp_path)


# =============================================================================
# analysis-statistics-11: seed stability is ranked on one scale
# =============================================================================

def test_seed_stability_ranking_compares_only_cross_entropy_metrics(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    ce = {6: 4.40, 54: 4.47, 271: 4.55}
    best = {6: 4.30, 54: 4.47, 271: 4.65}                      # a wider spread than test_ce
    _group(tmp_path, {
        seed: {"test_ppl": math.exp(value), "test_ce": value, "test_bits_per_token": value / math.log(2.0),
               "test_best_ce": best[seed], "wall_time_s": 100.0 + 0.001 * i}
        for i, (seed, value) in enumerate(ce.items())})
    _main(tmp_path, monkeypatch)
    report = (tmp_path / "MULTISEED_ANALYSIS.md").read_text(encoding="utf-8")
    lines = [line for line in report.splitlines() if "seed-stable" in line]
    assert len(lines) == 2
    most, least = lines
    assert "`test_ce`" in most or "`test_bits_per_token`" in most
    assert "`test_best_ce`" in least
    # A perplexity (exp of its CE) and the wall time are not ranked against the CE-scale metrics.
    assert not any("`test_ppl`" in line or "`wall_time_s`" in line for line in lines)
    assert "exp(CE)" in report


def test_docstrings_state_when_seeds_share_a_data_order() -> None:
    doc = " ".join(ms.__doc__.split())
    assert "currently shares the data-shuffle order" not in doc and "DATA_SEED" in doc
    band = " ".join(figs.plot_curve_band.__doc__.split())
    assert "the per-run reseed shares the data order" not in band and "DATA_SEED" in band
    # The kmup docstring states the four-width rule instead of a route size count that goes stale when the
    # scaling.py grids change (it named a three-size grow_K route; scaling.py grows grow_K over six widths).
    kmup = " ".join(figs.plot_kmup_stability.__doc__.split())
    assert "three-size" not in kmup and "fewer than four widths" in kmup
