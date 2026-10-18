r"""Audit 2026-10-16 F118: a seeded ablation sweep reports seed-paired differences against its reference arm.

Every cell of one seed shares the model-init seed and the data order (``_effective_data_seed``), so the difference
:math:`d_s = \mathrm{PPL}^{\rm arm}_s - \mathrm{PPL}^{\rm ref}_s` on the shared seeds has a smaller spread than the
two arms' seed SDs combined. ``analyze_sweep`` printed only per-arm mean / SD / CV; it now also prints, per arm, the
paired mean difference, its standard error and the 95% Student-t interval on :math:`n - 1` degrees of freedom. The
reference is the sweep's ``forest_baseline_label``, else the one arm that leaves the baseline unchanged. Checked on
synthetic sweep rows; no model is trained.
"""
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

import ablation


T_975_DF2 = 4.302652729749462                                     # scipy.stats.t.ppf(0.975, 2)


def _rows(ppl_by_arm: Dict[str, Dict[int, float]]) -> List[Dict[str, Any]]:
    return [{"label": f"{arm}__s{seed}", "seed": seed, "primary_val_ppl": ppl, "error": ""}
            for arm, by_seed in ppl_by_arm.items() for seed, ppl in by_seed.items()]


def test_paired_differences_on_shared_seeds() -> None:
    rows = _rows({
        "ref":  {6: 100.0, 23: 102.0, 54: 98.0},
        "armB": {6: 101.0, 23: 104.0, 54: 99.0},                  # d = 1, 2, 1
        "armC": {6: 97.0, 23: 99.0, 99: 50.0},                    # seed 99 has no reference cell
    })
    paired = {p["label"]: p for p in ablation._paired_seed_differences(rows, "ref")}
    assert set(paired) == {"armB", "armC"}
    b = paired["armB"]
    assert b["n"] == 3 and b["seeds"] == [6, 23, 54]
    assert b["mean"] == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert b["sd"] == pytest.approx(math.sqrt(1.0 / 3.0), abs=1e-12)
    assert b["se"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    half = T_975_DF2 * (1.0 / 3.0)
    assert b["ci95"] == pytest.approx([4.0 / 3.0 - half, 4.0 / 3.0 + half], abs=1e-9)
    c = paired["armC"]
    assert c["n"] == 2 and c["seeds"] == [6, 23] and c["mean"] == pytest.approx(-3.0, abs=1e-12)


def test_one_shared_seed_has_no_interval_and_a_missing_reference_pairs_nothing() -> None:
    rows = _rows({"ref": {6: 100.0}, "armB": {6: 101.0}})
    (b,) = ablation._paired_seed_differences(rows, "ref")
    assert b["n"] == 1 and b["mean"] == pytest.approx(1.0) and b["se"] is None and b["ci95"] is None
    assert ablation._paired_seed_differences(rows, "absent") == []


def test_reference_is_the_declared_or_the_unchanged_arm() -> None:
    assert ablation._paired_reference_label("component_ablation_forest") == "baseline"
    assert ablation._paired_reference_label("relational_features") == "relational_reference"
    assert ablation._paired_reference_label("new_attention_features") == "relational_reference"
    assert ablation._paired_reference_label("no_such_sweep") is None
    assert ablation._paired_reference_label(None) is None


def test_reference_of_a_value_sweep_is_the_cell_at_the_baseline_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ablation.SWEEPS, "_paired_probe", {"param": "m_phi_lr", "values": [0.001, 0.003, 0.009]})
    monkeypatch.setitem(ablation.BASELINE_CONFIG, "m_phi_lr", 0.003)
    assert ablation._paired_reference_label("_paired_probe") == "m_phi_lr=0.003"
    monkeypatch.setitem(ablation.BASELINE_CONFIG, "m_phi_lr", 0.004)    # no cell at the baseline value
    # Audit 2026-10-17 item 59: such a sweep gets the shared baseline cell, which is then its reference.
    assert ablation._paired_reference_label("_paired_probe") == "m_phi_lr=0.004"
    monkeypatch.setitem(ablation.SWEEPS, "_paired_probe", {"param": "m_phi_lr", "values": [0.001, 0.009],
                                                           "include_baseline": False})
    assert ablation._paired_reference_label("_paired_probe") is None


def test_analyze_sweep_prints_the_paired_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rows = _rows({
        "relational_reference":    {6: 100.0, 23: 102.0, 54: 98.0},
        "relational_frozen_score": {6: 101.0, 23: 104.0, 54: 99.0},
    })
    with open(tmp_path / "sweep_results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ablation._CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ablation._CSV_COLUMNS})
    (tmp_path / "sweep_meta.json").write_text(json.dumps({"sweep_name": "relational_features"}), encoding="utf-8")
    ablation.analyze_sweep(tmp_path)
    out = capsys.readouterr().out
    assert "paired across-seed differences vs relational_reference" in out
    line = next(line for line in out.splitlines() if line.startswith("relational_frozen_score") and "+1.333" in line)
    assert f"{4.0 / 3.0 - T_975_DF2 / 3.0:+.3f}" in line and f"{4.0 / 3.0 + T_975_DF2 / 3.0:+.3f}" in line
