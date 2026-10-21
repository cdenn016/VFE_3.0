r"""The evaluation-only table shrinkage probe (:mod:`vfe3.table_shrinkage`, 2026-10-21).

On the tiny fake run of ``tests/test_posthoc_20260928.py`` (vocabulary 24, K = 4, two heads, two layers, 8-token
windows at stride 4): the row weights follow the reset and shrink rules; the frame target is the identity and the
other targets are occurrence-weighted mean rows; an arm moves exactly the rows below the threshold and restores them
bitwise; the baseline equals the plain evaluation of the checkpoint; and the report is a new JSON file that leaves
every run artifact unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
import torch

from tests.test_posthoc_20260928 import DEVICE, _fake_run
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import streams as st
from vfe3.table_shrinkage import (TableShrinkageOptions, _MovedRows, row_weights, run_table_shrinkage,
                                  summary_table, table_target)


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _snapshot(run_dir: Path) -> Dict[str, bytes]:
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(run_dir.rglob("*")) if p.is_file()}


def test_row_weights_follow_the_reset_and_shrink_rules() -> None:
    counts = torch.tensor([0, 5, 10, 200])
    assert row_weights(counts, 10, "reset").tolist() == [0.0, 0.0, 1.0, 1.0]
    assert torch.allclose(row_weights(counts, 10, "shrink"),
                          torch.tensor([0.0, 5 / 15, 10 / 20, 200 / 210], dtype=torch.float64))
    with pytest.raises(ValueError):
        row_weights(counts, 10, "drop")


def test_targets_and_moved_rows_restore_exactly() -> None:
    table = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    counts = torch.tensor([1, 0, 3, 0])
    assert torch.equal(table_target(table, "phi_embed", counts), torch.zeros(3, dtype=torch.float64))
    mean = table_target(table, "mu_embed", counts)
    assert torch.allclose(mean, (1 * table[0].double() + 3 * table[2].double()) / 4)
    original = table.detach().clone()
    keep = row_weights(counts, 2, "reset")                        # rows 0, 1, 3 move; row 2 stays
    with torch.no_grad():
        with _MovedRows(table, mean, keep) as n_rows:
            assert n_rows == 3
            assert torch.equal(table[2], original[2])
            for row in (0, 1, 3):
                assert torch.allclose(table[row].double(), mean)
    assert torch.equal(table.detach(), original)


def test_probe_on_the_fake_run(tmp_path: Path) -> None:
    run_dir, cache_dir = _fake_run(tmp_path)
    before = _snapshot(run_dir)
    options = TableShrinkageOptions(device="cpu", precision="fp32", cache_dir=str(cache_dir), thresholds=(2, 1000),
                                    splits=("train_sample", "validation"), max_windows=None)
    report = run_table_shrinkage(run_dir, options=options, log=lambda _: None)

    loaded = ev.load_model(run_dir, "final_model.pt", device=DEVICE, precision="fp32")
    info = st.load_run_info(run_dir, "final_model.pt")
    tokens, _ = st.resolve_stream(info, "validation", cache_dir=cache_dir)
    dataset = st.eval_windows(tokens, info.seq_len, info.eval_stride)
    expected = float(ev.protocol_evaluate(loaded.model, dataset, range(len(dataset)), batch_size=info.cfg.batch_size,
                                          device=DEVICE)["ce"])
    assert report["baseline"]["validation"] == expected
    assert report["baseline_restored_matches"] is True

    tables = report["tables"]
    assert "phi_embed" in tables and "mu_embed" in tables
    labels = {(arm["table"], arm["threshold"]) for arm in report["arms"]}
    assert labels == {(name, c) for name in tables + ["all"] for c in (2, 1000)}
    for arm in report["arms"]:
        delta = arm["delta_ce"]
        assert arm["memorization_score"] == pytest.approx(delta["train_sample"] - delta["validation"])
        if arm["threshold"] == 1000:                              # every type of the tiny stream is rarer than 1000
            assert arm["rows_moved"] == info.cfg.vocab_size
            assert arm["train_token_mass_moved"] == pytest.approx(1.0)
    assert any(abs(arm["delta_ce"]["validation"]) > 0 for arm in report["arms"])
    assert "memorization" not in summary_table(report).splitlines()[0] and len(summary_table(report).splitlines()) > 1

    after = _snapshot(run_dir)
    assert set(after) - set(before) == {"table_shrinkage_probe.json"}
    assert all(after[name] == data for name, data in before.items())
    written = json.loads((run_dir / "table_shrinkage_probe.json").read_text(encoding="utf-8"))
    assert written["mode"] == "reset" and written["thresholds"] == [2, 1000]


def test_protected_output_and_bad_options_are_refused(tmp_path: Path) -> None:
    run_dir, cache_dir = _fake_run(tmp_path)
    for bad in (dict(out_name="summary.json"), dict(tables=("not_a_table",)), dict(thresholds=(0,)),
                dict(splits=("validation",))):
        with pytest.raises(ValueError):
            run_table_shrinkage(run_dir, options=TableShrinkageOptions(device="cpu", cache_dir=str(cache_dir), **bad),
                                log=lambda _: None)
