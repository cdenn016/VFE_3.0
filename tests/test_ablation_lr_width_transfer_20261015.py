r"""Learning-rate width-transfer sweep of ablation.py (``LR_WIDTH_TRANSFER``, 2026-10-15).

The rates of the operating point were tuned at K=20 with two heads and did not carry to K=160. The sweep crosses
each width with each listed rate times each multiplier, and the fit reports the best multiplier per width and the
exponent b of best lr ~ K^b over the widths whose best multiplier is interior. The fit is checked on synthetic
sweep rows whose optimum scales exactly as K^-1; no model is trained.
"""
import math
from typing import Any, Dict, List

import pytest

import ablation


def _rows(widths, field: str, base: float, exponent: float, multipliers, diverged=()) -> List[Dict[str, Any]]:
    r"""Rows whose PPL is a parabola in log(lr) with its minimum at base * (K/20)^exponent."""
    rows = []
    for embed_dim, n_heads in widths:
        optimum = math.log(base * (embed_dim / 20) ** exponent)
        for m in multipliers:
            label = ablation._lr_width_transfer_label(embed_dim, n_heads, field, m)
            failed = (embed_dim, m) in diverged
            ppl = "" if failed else 50 + (math.log(base * m) - optimum) ** 2
            rows.append({"label": label, "primary_val_ppl": ppl, "final_val_ppl": ppl,
                         "error": "diverged" if failed else ""})
    return rows


def test_arms_cross_widths_rates_and_multipliers() -> None:
    settings = {"widths": [(20, 2), (40, 4)], "lr_fields": ["m_phi_lr", "m_block_mlp_lr"],
                "multipliers": [0.5, 2.0], "max_steps": 10, "enabled": True}
    baseline = {"m_phi_lr": 0.004, "m_block_mlp_lr": 0.0025}
    arms = ablation._lr_width_transfer_arms(settings, baseline)
    assert len(arms) == 2 * 2 * 2
    arm = next(a for a in arms if a["label"] == "K=40,H=4,m_phi_lr=x2")
    assert arm == {"label": "K=40,H=4,m_phi_lr=x2", "embed_dim": 40, "n_heads": 4, "m_phi_lr": 0.008}
    with pytest.raises(ValueError, match="n_heads must divide"):
        ablation._lr_width_transfer_arms({**settings, "widths": [(20, 3)]}, baseline)
    with pytest.raises(ValueError, match="positive BASELINE_CONFIG value"):
        ablation._lr_width_transfer_arms(settings, {"m_phi_lr": None, "m_block_mlp_lr": 0.0025})
    lenient = ablation._lr_width_transfer_arms(settings, {"m_phi_lr": None, "m_block_mlp_lr": 0.0025}, strict=False)
    assert {next(k for k in arm if k.endswith("_lr")) for arm in lenient} == {"m_block_mlp_lr"}


def test_fit_recovers_the_width_exponent() -> None:
    widths = [(20, 2), (40, 4), (80, 8)]
    multipliers = [2.0 ** k for k in range(-4, 3)]                  # 1/16 ... 4
    rows = _rows(widths, "m_phi_lr", 0.004, -1.0, multipliers)
    fit = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]
    assert [w["best_multiplier"] for w in fit["widths"]] == [1.0, 0.5, 0.25]
    assert fit["interior_widths"] == 3
    assert fit["exponent"] == pytest.approx(-1.0, abs=1e-12)
    assert fit["eta_at"]["160"] == pytest.approx(0.004 / 8, rel=1e-12)


def test_edge_optima_and_diverged_cells_are_reported_not_fitted() -> None:
    widths = [(20, 2), (40, 4), (80, 8)]
    multipliers = [0.5, 1.0, 2.0]
    rows = _rows(widths, "m_phi_lr", 0.004, -2.0, multipliers, diverged={(20, 2.0)})
    fit = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]
    k20, k40, k80 = fit["widths"]
    assert k20["diverged_multipliers"] == [2.0] and k20["ppl_by_multiplier"]["2"] is None
    assert k40["best_multiplier"] == 0.5 and k40["at_grid_edge"]       # the optimum (x0.25) lies off the grid
    assert k80["at_grid_edge"]
    assert "exponent" not in fit                                        # one interior width cannot fix a slope


def test_the_toggle_runs_only_the_transfer_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}

    def stop(names, **_: Any) -> None:
        seen["names"] = list(names)
        raise RuntimeError("stop before training")

    monkeypatch.setitem(ablation.LR_WIDTH_TRANSFER, "enabled", True)
    monkeypatch.setitem(ablation.CONFIG, "list_only", False)
    monkeypatch.setattr(ablation, "validate_sweeps", stop)
    with pytest.raises(RuntimeError, match="stop before training"):
        ablation.main()
    assert seen["names"] == ["lr_width_transfer"]


def test_the_sweep_is_registered_and_validates() -> None:
    assert ablation.LR_WIDTH_TRANSFER["enabled"] in (True, False)
    ablation.validate_sweeps(["lr_width_transfer"], require_construction=False)


def test_a_requested_cell_without_a_row_counts_as_diverged() -> None:
    widths = [(20, 2), (40, 4), (80, 8)]
    multipliers = [0.5, 1.0, 2.0, 4.0]
    rows = _rows(widths, "m_phi_lr", 0.004, 0.0, multipliers)
    missing = ablation._lr_width_transfer_label(80, 8, "m_phi_lr", 4.0)
    rows = [row for row in rows if row["label"] != missing]
    requested = [ablation._lr_width_transfer_label(k, h, "m_phi_lr", m) for k, h in widths for m in multipliers]
    fit = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004}, requested)["fields"]["m_phi_lr"]
    k80 = fit["widths"][2]
    assert k80["diverged_multipliers"] == [4.0] and k80["ppl_by_multiplier"]["4"] is None
    assert k80["best_multiplier"] == 1.0 and not k80["at_grid_edge"]     # the grid edge is x4, not the last row


def test_an_incomplete_transfer_sweep_is_still_fitted(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    calls: List[Any] = []

    def fake_run_sweep(name, output_dir, **_: Any) -> list:
        directory = output_dir / ablation._sweep_output_scope(name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sweep_meta.json").write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
        (directory / "sweep_results.csv").write_text("label,primary_val_ppl\n", encoding="utf-8")
        return []

    monkeypatch.setitem(ablation.LR_WIDTH_TRANSFER, "enabled", True)
    monkeypatch.setitem(ablation.CONFIG, "list_only", False)
    monkeypatch.setitem(ablation.CONFIG, "output_dir", str(tmp_path))
    monkeypatch.setitem(ablation.CONFIG, "device", "cpu")
    monkeypatch.setattr(ablation, "validate_sweeps", lambda *a, **k: None)
    monkeypatch.setattr(ablation, "_run_ablation_figures_isolated", lambda *a, **k: True)
    monkeypatch.setattr(ablation, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(ablation, "analyze_lr_width_transfer", lambda d: calls.append(d))
    assert ablation.main() == 1                                          # still reported incomplete
    assert [d.name for d in calls] == [ablation._sweep_output_scope("lr_width_transfer")]
