r"""Analysis-driver fixes from the 2026-10-09 audit (general sweep: analysis, paper and visualization).

``make_figures`` validates run files without building the model and checks the final weights the
figure worker replays; ``aggregate_seed_curves`` keeps the last finite value of a repeated step;
the scaling harvest logs the specific failed check; incomplete optimization candidates do not enter
the interaction projection.
"""
from tests.profiles import test_config

import json
import math
from pathlib import Path

import numpy as np
import pytest

import make_figures as mf
import multiseed_analysis as ms
import parameter_optimization as po
import scaling_analysis as sa
from tests.test_audit_20261008_stats import _write_scaling_run
from tests.test_multiseed import _write_full_run, _write_request


def _figure_run(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"config": {"embed_dim": 4}}), encoding="utf-8")
    (path / "summary.json").write_text(
        json.dumps({"n_steps": 10, "n_params": 100, "test_ce": 4.0}), encoding="utf-8")
    (path / "best_model.pt").write_bytes(b"not a model; selection reads files only")
    return path


def test_make_figures_selection_checks_files_without_building_the_model(tmp_path):
    run = _figure_run(tmp_path / "run")
    assert mf._validated_run_dir(run) == run                  # an unloadable payload is not opened here
    (run / "final_model.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="empty final_model.pt"):
        mf._validated_run_dir(run)
    (run / "final_model.pt").write_bytes(b"x")
    assert mf._validated_run_dir(run) == run
    (run / "best_model.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="empty best_model.pt"):
        mf._validated_run_dir(run)


def test_aggregate_seed_curves_keeps_the_last_finite_value_of_a_repeated_step(tmp_path, caplog):
    _write_request(tmp_path, [1, 2])
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1,
                    metrics="step,x\n100,1\n200,2\n200,7\n200,\n300,4\n")
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2, metrics="step,x\n100,3\n200,9\n300,6\n")
    with caplog.at_level("WARNING", logger="multiseed_analysis"):
        curves = ms.aggregate_seed_curves(tmp_path, columns=["x"])
    np.testing.assert_allclose(curves["x"]["steps"], [100, 200, 300])
    np.testing.assert_allclose(curves["x"]["mean"], [2.0, 8.0, 5.0])     # step 200 uses 7, not 2
    assert "2 repeated step value(s)" in caplog.text


def test_scaling_harvest_logs_the_failed_check(tmp_path, caplog):
    _write_scaling_run(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0)
    cell_path = tmp_path / "width" / "K8" / "s1" / "scaling_cell.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["config_sha256"] = "0" * 64
    cell_path.write_text(json.dumps(cell), encoding="utf-8")
    with caplog.at_level("WARNING", logger="scaling_analysis"):
        assert sa.harvest(tmp_path) == []
    assert "(reuse digest or config_sha256 mismatch)" in caplog.text
    reasons: list = []
    assert sa._validated_bound_scaling_run(tmp_path, [], {}, {}, {}, rejection=reasons) is None
    assert reasons == ["summary, config.json, provenance or scaling_cell.json is not a mapping"]


def test_interaction_projection_excludes_incomplete_candidates():
    search_config = {"parameters": {
        "a": {"enabled": True, "values": [0, 1]},
        "b": {"enabled": True, "values": [0, 1]},
    }}
    candidates = [
        {"label": f"c{a}{b}", "overrides": {"a": a, "b": b}} for a in (0, 1) for b in (0, 1)
    ]
    ppl = {"c00": 10.0, "c01": 11.0, "c10": 12.0, "c11": 13.0}         # exactly additive
    rows = [{"label": label, "status": "complete", "mean_primary_val_ppl": value}
            for label, value in ppl.items()]
    (additive,) = po._interaction_rows(rows, candidates, search_config, n_bins=2)
    assert additive["interaction_rms_ppl"] == pytest.approx(0.0)
    rows[-1] = {"label": "c11", "status": "incomplete", "mean_primary_val_ppl": 50.0}
    (projected,) = po._interaction_rows(rows, candidates, search_config, n_bins=2)
    assert projected["_cell_medians"][1][1] is None                     # the 50.0 never enters
    assert math.isfinite(projected["interaction_rms_ppl"])
