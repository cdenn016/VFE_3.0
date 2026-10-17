r"""Audit 2026-10-16 lane B: what the multi-seed digest states about its seeds (F62, F119).

F62: ``MULTISEED_ANALYSIS.md`` never said whether the seeds shared one training data order, and the JSON
status lumped per-seed orders together with groups in which only some seeds share an order. The digest
now writes the status and its sentence (under a shared order the SD is a lower bound on the variance
over data orders) beside the headline, and ``distinct`` and ``partially_shared`` are separate statuses.
F119: the realized-update counts (``skipped_steps``, ``accepted_update_frac``) and the final validation
perplexity enter the across-seed scalars, and the digest names every seed that skipped an update.

Every fixture is a tiny JSON run directory; no model is built.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import multiseed_analysis as ms
from tests.test_multiseed import _write_full_run, _write_request


def _group(root: Path, data_seeds: Dict[int, Optional[int]],
           summaries: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
    _write_request(root, sorted(data_seeds))
    for i, (seed, data_seed) in enumerate(sorted(data_seeds.items())):
        run = root / f"run_s{seed}"
        _write_full_run(run, ppl=10.0 + i, seed=seed)
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        provenance["data_seed"] = data_seed
        (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        if summaries and seed in summaries:
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            (run / "summary.json").write_text(json.dumps({**summary, **summaries[seed]}), encoding="utf-8")


def _main(root: Path, monkeypatch: pytest.MonkeyPatch, scalar_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    monkeypatch.setitem(ms.CONFIG, "run_root", str(root))
    monkeypatch.setitem(ms.CONFIG, "key", "test_ppl")
    monkeypatch.setitem(ms.CONFIG, "allow_mixed_code_identity", False)
    if scalar_keys is not None:
        monkeypatch.setattr(ms, "SCALAR_KEYS", scalar_keys)
    monkeypatch.setattr(ms, "_emit_figures", lambda *args: None)
    assert ms.main() == 0
    return json.loads((root / "multiseed_summary.json").read_text(encoding="utf-8"))


# =============================================================================
# F62: the data-order status, split and stated beside the headline
# =============================================================================

def test_data_order_status_separates_distinct_from_partially_shared():
    shared = ms._data_order_status([3, 3, 3])
    assert shared["status"] == "shared_explicit" and shared["shared"] is True and shared["data_seed"] == 3
    assert "lower bound" in shared["caveat"]
    distinct = ms._data_order_status([6, 54, 271])
    assert distinct["status"] == "distinct" and distinct["shared"] is False
    assert "lower bound" not in distinct["caveat"]
    assert ms._data_order_status([3, 3, 5])["status"] == "partially_shared"
    assert ms._data_order_status([3, 3, None])["status"] == "partially_shared"
    assert ms._data_order_status([None, None])["status"] == "nonshared_unspecified"
    assert ms._data_order_status([3, None])["status"] == "distinct"
    assert ms._data_order_status([])["status"] == "no_seeds"
    per_seed = ms._data_order_status([3, 3], seeds=[6, 54])
    assert per_seed["per_seed"] == {"6": 3, "54": 3}


def test_digest_states_the_shared_order_beside_the_headline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _group(tmp_path, {6: 3, 54: 3, 271: 3})
    summary = _main(tmp_path, monkeypatch, scalar_keys=["test_ppl"])
    order = summary["design"]["data_order"]
    assert order["status"] == "shared_explicit" and "lower bound" in order["caveat"]
    report = (tmp_path / "MULTISEED_ANALYSIS.md").read_text(encoding="utf-8")
    headline = report.index("## Headline scalars")
    assert "shared_explicit" in report[:headline] and "lower bound" in report[:headline]


def test_per_seed_orders_are_distinct_in_the_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _group(tmp_path, {6: 6, 54: 54, 271: 271})
    summary = _main(tmp_path, monkeypatch, scalar_keys=["test_ppl"])
    assert summary["design"]["data_order"]["status"] == "distinct"
    report = (tmp_path / "MULTISEED_ANALYSIS.md").read_text(encoding="utf-8")
    assert "`distinct`" in report and "includes data-order variance" in report
