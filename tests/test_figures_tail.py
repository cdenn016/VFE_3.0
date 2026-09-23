r"""Tests for the 2026-06-22 figures-tail build-out (EXP-1/2/6/9/10/11 auto-dispatched figures):

  * the six new registered figures render from synthetic inputs;
  * the EXP-11 kappa_beta_per_head sweep carries the geo-mean-tau confound-control arms;
  * scaling_analysis.aggregate_points carries embed_dim + test PPL (the EXP-6 muP K axis);
  * train() logs the EXP-9 builder-break residual per eval (val_builder_resid) under a head mixer;
  * the EXP-2 / EXP-11 ablation drivers read per-cell JSON and emit a PNG (no-op safe).

Device-agnostic (CPU). Figures use the Agg backend.
"""
import csv as _csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

import ablation
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.metrics import bootstrap_token_ce_band
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import train
from vfe3.viz import figures as figs
from vfe3.viz.specs import FigureSpec, emit_registered_figures
from vfe3.viz.sweep_adapters import (
    ablation_forest_kwargs,
    aggregate_validation_points,
    capacity_scaling_kwargs,
    lr_grid_heatmap_kwargs,
    pareto_frontier_kwargs,
)

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


# --------------------------------------------------------------------------- figure renders

def test_gauge_transport_bars_renders():
    cells = [{"mode": m, "depth": d, "ppl": 30.0 + i,
              "omega_dev": (1e-7 if m == "off" else float("nan"))}
             for i, (m, d) in enumerate([(m, d) for d in ("L1", "L2") for m in ("on", "frozen", "off")])]
    fig = figs.plot_gauge_transport_bars(cells); assert fig is not None; plt.close(fig)


def test_gauge_residual_drift_renders():
    arms = [{"label": "tied", "step": [10, 20, 30], "resid": [1e-7, 1e-7, 1e-7]},
            {"label": "untied", "step": [10, 20, 30], "resid": [1e-7, 1e-3, 1e-1]}]
    fig = figs.plot_gauge_residual_drift(arms); assert fig is not None; plt.close(fig)


def test_ppl_equivariance_bars_renders():
    cells = [{"label": "cg_off", "ppl": 31.0, "resid": 1e-7},
             {"label": "cg_on", "ppl": 29.5, "resid": 1e-7}]
    fig = figs.plot_ppl_equivariance_bars(cells); assert fig is not None; plt.close(fig)


def test_kappa_dispersion_renders():
    cells = [{"label": "uniform", "dispersion": 0.0, "ppl": 30.0},
             {"label": "split", "dispersion": 0.2, "ppl": 30.5}]
    fig = figs.plot_kappa_dispersion(cells); assert fig is not None; plt.close(fig)


def test_kmup_stability_renders():
    routes = {"grow_K": [{"embed_dim": k, "ppl_mean": 100.0 / k + 5, "ppl_sem": 0.5}
                         for k in (20, 40, 80, 120)],
              "grow_K_mup": [{"embed_dim": k, "ppl_mean": 90.0 / k + 5, "ppl_sem": 0.4}
                            for k in (20, 40, 80, 120)]}
    fig = figs.plot_kmup_stability(routes); assert fig is not None; plt.close(fig)


def test_ppl_noise_band_renders():
    agg = {"values": [20.1, 20.4, 19.8, 20.2, 20.0], "seeds": [6, 23, 54, 66, 122],
           "mean": 20.1, "sd": 0.22}
    fig = figs.plot_ppl_noise_band(agg, grid={"lr=0.01": 20.3, "lr=0.02": 19.9})
    assert fig is not None; plt.close(fig)


# --------------------------------------------------------------------------- EXP-6 aggregation

def test_aggregate_points_carries_embed_dim_and_ppl():
    rows = [{"route": "grow_K", "scale_knob": "embed_dim", "label": "K20", "seed": 0,
             "n_params": 1000, "embed_dim": 20, "n_gen": 6, "tokens_seen": 1000,
             "est_flops_6ND": 1.0, "est_flops_analytic": 1.0, "n_e_steps": 1, "n_layers": 1,
             "test_ce": 3.0, "test_ppl": 20.0}]
    p = scaling_analysis.aggregate_points(rows)[0]
    assert p["embed_dim"] == 20.0 and abs(p["ppl_mean"] - 20.0) < 1e-9


# --------------------------------------------------------------------------- EXP-9 per-eval residual

def test_val_builder_resid_logged_with_head_mixer(tmp_path):
    torch.manual_seed(0)
    cfg = VFE3Config(vocab_size=32, embed_dim=4, n_heads=2, max_seq_len=8, max_steps=4,
                     log_interval=2, eval_interval=2, batch_size=2,
                     gauge_group="block_glk", use_head_mixer=True)
    model = VFEModel(cfg).to(DEVICE)
    batches = [(torch.randint(0, 32, (2, 8), device=DEVICE),
                torch.randint(0, 32, (2, 8), device=DEVICE)) for _ in range(4)]
    art = RunArtifacts(str(tmp_path), cfg, model, dataset="synthetic-period3", device=str(DEVICE))
    train(model, batches, cfg, n_steps=4, log_interval=2, eval_interval=2,
          val_loader=batches, artifacts=art, device=DEVICE, generate_samples=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows and "val_builder_resid" in rows[0]
    finite = [r for r in rows if r["val_builder_resid"] not in ("", "nan", None)]
    assert finite, "an eval row must carry a finite builder residual under a head mixer"


# --------------------------------------------------------------------------- driver smokes

def _write_cell(sweep_dir, label, result):
    d = sweep_dir / label
    d.mkdir(parents=True)
    marker = {"label": label, **result}
    marker.setdefault("final_val_ppl", marker.get("primary_val_ppl"))
    marker.setdefault("status", "success")
    marker.setdefault("error_kind", None)
    (d / "ablation_result.json").write_text(json.dumps(marker))


def _raw_fixture_rows(sweep_dir):
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(sweep_dir.glob("*/ablation_result.json"))
    ]


def test_plot_gauge_transport_driver(tmp_path, monkeypatch):
    sweep = tmp_path / "gauge_transport"; figdir = tmp_path / "figures"
    for lab, ppl, dev in [("on_L1", 30.0, 0.4), ("off_L1", 33.0, 1e-7), ("frozen_L1", 31.0, 0.4),
                          ("on_L2", 28.0, 0.4), ("off_L2", 32.0, 1e-7), ("frozen_L2", 30.0, 0.4)]:
        _write_cell(sweep, lab, {"primary_val_ppl": ppl, "omega_identity_dev": dev})
    monkeypatch.setattr(ablation, "_collect_sweep_results", _raw_fixture_rows)
    ablation._plot_gauge_transport(sweep, figdir)
    assert (figdir / "gauge_transport_gauge_bars.png").exists()


def test_plot_kappa_dispersion_driver(tmp_path, monkeypatch):
    sweep = tmp_path / "kappa_beta_per_head"; figdir = tmp_path / "figures"
    for lab, kb, ppl in [("uniform_1.0", [1.0, 1.0], 30.0), ("split_0.6_1.4", [0.6, 1.4], 30.8),
                         ("geomean_0.6_1.4", [0.91652, 0.91652], 30.2)]:
        _write_cell(sweep, lab, {"primary_val_ppl": ppl, "overrides": {"kappa_beta": kb}})
    monkeypatch.setattr(ablation, "_collect_sweep_results", _raw_fixture_rows)
    ablation._plot_kappa_dispersion(sweep, figdir)
    assert (figdir / "kappa_beta_per_head_kappa_dispersion.png").exists()


def test_plot_cg_coupling_driver(tmp_path, monkeypatch):
    sweep = tmp_path / "cg_coupling"; figdir = tmp_path / "figures"
    for lab, ppl in [("cg_off", 31.0), ("cg_on", 29.5)]:
        _write_cell(sweep, lab, {"primary_val_ppl": ppl, "gauge_resid_in": 1e-7,
                                 "overrides": {"use_cg_coupling": lab == "cg_on"}})
    monkeypatch.setattr(ablation, "_collect_sweep_results", _raw_fixture_rows)
    ablation._plot_cg_coupling(sweep, figdir)
    assert (figdir / "cg_coupling_ppl_equiv.png").exists()


def test_plot_gauge_residual_drift_driver(tmp_path, monkeypatch):
    sweep = tmp_path / "gauge_equivariance"; figdir = tmp_path / "figures"
    for lab, climb in [("tied_block_glk", 1e-7), ("untied_block_glk", 1e-2)]:
        d = sweep / lab; d.mkdir(parents=True)
        with open(d / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=["step", "val_builder_resid"])
            w.writeheader()
            for st, r in [(10, 1e-7), (20, climb)]:
                w.writerow({"step": st, "val_builder_resid": r})
    monkeypatch.setattr(
        ablation,
        "_compatible_cell_dirs",
        lambda _sweep_dir: sorted(path for path in sweep.iterdir() if path.is_dir()),
    )
    ablation._plot_gauge_residual_drift(sweep, figdir)
    assert (figdir / "gauge_equivariance_residual_drift.png").exists()


def test_kmup_series_splits_fixed_and_mup_arms():
    r"""EXP-6 fix: route_grow_k_mup emits K{k}_fixed and K{k}_mup BOTH under route='grow_K_mup'; the
    series builder must split them so the figure shows the fixed-vs-muP contrast, not one duplicated-K
    curve."""
    pts = []
    for k in (20, 40, 80):
        for suf in ("fixed", "mup"):
            pts.append({"route": "grow_K_mup", "label": f"K{k}_{suf}", "embed_dim": float(k),
                        "ppl_mean": 100.0 / k + (1.0 if suf == "mup" else 0.0),
                        "ppl_sem": 0.1, "n_seeds": 2})
    series = scaling_analysis._kmup_series(pts)
    assert set(series) == {"grow_K_mup/fixed", "grow_K_mup/mup"}
    assert all(len(v) == 3 for v in series.values())               # one point per K, no conflation
    assert [p["embed_dim"] for p in series["grow_K_mup/fixed"]] == [20.0, 40.0, 80.0]


# --------------------------------------------------------------------------- PB-07: validation
# adapters for capacity_scaling / pareto_frontier (vfe3.viz.sweep_adapters)
#
# aggregate_validation_points is keyed SOLELY on the persisted best_val_ppl -- every fixture below
# sets test_ce/test_ppl/test_bpc to a deliberately different sentinel (999.0) so a mis-wired adapter
# that silently read a test metric instead would fail these assertions rather than pass by accident.

_AXIS_ROUTES = {"embed_dim": "grow_K", "n_heads": "blocksize", "n_layers": "inference"}


def _val_row(route, scale_knob, label, seed, *, best_val_ppl, n_params=1000.0, embed_dim=None,
             n_heads=None, n_layers=None, n_e_steps=None, wall_time_s=10.0,
             test_ce=999.0, test_ppl=999.0, test_bpc=999.0):
    return {"route": route, "scale_knob": scale_knob, "label": label, "seed": seed,
            "n_params": n_params, "embed_dim": embed_dim, "n_heads": n_heads,
            "n_layers": n_layers, "n_e_steps": n_e_steps,
            "best_val_ppl": best_val_ppl, "wall_time_s": wall_time_s,
            "test_ce": test_ce, "test_ppl": test_ppl, "test_bpc": test_bpc}


def _grow_k_capacity_scaling_rows():
    r"""Four grow_K/embed_dim points (K20..K100); every seed pair is a power of two so
    log2(best_val_ppl) pins to a clean integer. K100 is validation-only (no test metric survives)."""
    rows = []
    for label, embed_dim, n_params, seeds, wall_times in (
        ("K20",  20,  1000.0, (16.0, 64.0), (10.0, 12.0)),   # log2 -> 4, 6 -> mean 5.0
        ("K40",  40,  2000.0, (8.0, 32.0),  (20.0, 24.0)),   # log2 -> 3, 5 -> mean 4.0
        ("K80",  80,  4000.0, (4.0, 4.0),   (30.0, 34.0)),   # log2 -> 2, 2 -> mean 2.0
    ):
        for seed, (ppl, wt) in enumerate(zip(seeds, wall_times)):
            rows.append(_val_row("grow_K", "embed_dim", label, seed, best_val_ppl=ppl,
                                  n_params=n_params, embed_dim=embed_dim, n_heads=4, n_layers=2,
                                  wall_time_s=wt))
    # K100: validation-only fixture -- every test metric is null, must still survive aggregation.
    for seed, (ppl, wt) in enumerate(zip((2.0, 2.0), (40.0, 44.0))):    # log2 -> 1, 1 -> mean 1.0
        rows.append(_val_row("grow_K", "embed_dim", "K100", seed, best_val_ppl=ppl,
                              n_params=8000.0, embed_dim=100, n_heads=4, n_layers=2, wall_time_s=wt,
                              test_ce=None, test_ppl=None, test_bpc=None))
    return rows


def _inference_capacity_scaling_rows():
    r"""infer_L (route=inference, scale_knob=n_layers) and infer_T (scale_knob=n_e_steps) rows
    sharing route='inference': only the n_layers cells belong on the n_layers panel."""
    rows = []
    for n_layers, ppl, wt in ((1, 8.0, 5.0), (2, 16.0, 10.0), (3, 32.0, 15.0)):   # log2 -> 3, 4, 5
        for seed in (0, 1):
            rows.append(_val_row("inference", "n_layers", f"L{n_layers}", seed, best_val_ppl=ppl,
                                  n_params=1000.0, embed_dim=20, n_heads=4, n_layers=n_layers,
                                  wall_time_s=wt))
    for n_e_steps in (1, 2):
        rows.append(_val_row("inference", "n_e_steps", f"T{n_e_steps}", 0, best_val_ppl=100.0,
                              n_params=1000.0, embed_dim=20, n_heads=4, n_layers=1,
                              n_e_steps=n_e_steps, wall_time_s=1.0))
    return rows


def test_aggregate_validation_points_val_bits_per_token_mean_from_best_val_ppl_feeds_capacity_scaling():
    points = aggregate_validation_points(_grow_k_capacity_scaling_rows())
    by_label = {p["label"]: p for p in points}
    assert by_label["K20"]["val_bits_per_token_mean"] == pytest.approx(
        (math.log2(16.0) + math.log2(64.0)) / 2)
    assert by_label["K20"]["val_bits_per_token_mean"] == pytest.approx(5.0)
    assert by_label["K40"]["val_bits_per_token_mean"] == pytest.approx(4.0)
    assert by_label["K80"]["val_bits_per_token_mean"] == pytest.approx(2.0)
    # every row's test_ce/test_ppl/test_bpc sentinel (999.0) is nowhere near these values -- proof
    # the adapter never read a test metric.
    for label in ("K20", "K40", "K80"):
        assert by_label[label]["val_bits_per_token_mean"] < 10.0


def test_aggregate_validation_points_survives_validation_only_row_and_reaches_capacity_scaling_and_pareto_frontier():
    rows = _grow_k_capacity_scaling_rows()
    points = aggregate_validation_points(rows)
    labels = {p["label"] for p in points}
    assert "K100" in labels                                     # test_ce/test_ppl/test_bpc all None
    k100 = next(p for p in points if p["label"] == "K100")
    assert k100["val_bits_per_token_mean"] == pytest.approx(1.0)

    cap = capacity_scaling_kwargs(points, _AXIS_ROUTES)
    assert cap is not None
    assert 100.0 in cap["scaling"]["embed_dim"]["x"].tolist()

    pareto = pareto_frontier_kwargs(points)
    assert pareto is not None
    assert "K100" in pareto["points"]["label"]


def test_capacity_scaling_kwargs_pins_sorted_axes_and_omits_unavailable_route():
    rows = _grow_k_capacity_scaling_rows() + _inference_capacity_scaling_rows()
    points = aggregate_validation_points(rows)
    result = capacity_scaling_kwargs(points, _AXIS_ROUTES)
    assert result is not None
    # "blocksize" (n_heads) never ran in this fixture -> omitted, not an error.
    assert set(result["scaling"]) == {"embed_dim", "n_layers"}

    embed = result["scaling"]["embed_dim"]
    assert embed["x"].tolist() == [20.0, 40.0, 80.0, 100.0]
    assert embed["bits_per_token"].tolist() == pytest.approx([5.0, 4.0, 2.0, 1.0])
    assert embed["wall_time"].tolist() == pytest.approx([11.0, 22.0, 32.0, 42.0])

    # only the infer_L (scale_knob=n_layers) cells enter this panel; the infer_T (n_e_steps) cells
    # sharing route='inference' must not, even though route equality alone would admit them.
    nlayers = result["scaling"]["n_layers"]
    assert nlayers["x"].tolist() == [1.0, 2.0, 3.0]
    assert nlayers["bits_per_token"].tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert nlayers["wall_time"].tolist() == pytest.approx([5.0, 10.0, 15.0])


def test_capacity_scaling_kwargs_none_when_selected_route_has_fewer_than_two_points():
    rows = (_grow_k_capacity_scaling_rows() + _inference_capacity_scaling_rows()
            + [_val_row("blocksize", "n_heads", "H4", 0, best_val_ppl=10.0, n_params=1000.0,
                        embed_dim=20, n_heads=4, n_layers=2, wall_time_s=1.0)])
    points = aggregate_validation_points(rows)
    # "blocksize" (n_heads) is SELECTED (it ran) but has only 1 point -- withhold the whole figure.
    assert capacity_scaling_kwargs(points, _AXIS_ROUTES) is None


def test_aggregate_validation_points_raises_on_explicit_null_best_val_ppl_in_capacity_scaling_route():
    rows = [
        _val_row("grow_K", "embed_dim", "K20", 0, best_val_ppl=16.0, embed_dim=20),
        _val_row("grow_K", "embed_dim", "K20", 1, best_val_ppl=64.0, embed_dim=20),
        _val_row("grow_K", "embed_dim", "K40", 0, best_val_ppl=8.0, embed_dim=40),
        _val_row("grow_K", "embed_dim", "K40", 1, best_val_ppl=None, embed_dim=40),   # explicit null
        _val_row("grow_K", "embed_dim", "K80", 0, best_val_ppl=4.0, embed_dim=80),
        _val_row("grow_K", "embed_dim", "K80", 1, best_val_ppl=4.0, embed_dim=80),
    ]
    with pytest.raises(ValueError, match="explicit-null"):
        aggregate_validation_points(rows)


def test_aggregate_validation_points_pairs_quality_and_time_by_exact_seed():
    rows = [
        _val_row("grow_K", "embed_dim", "K20", 0, best_val_ppl=16.0,
                 embed_dim=20, wall_time_s=None),
        _val_row("grow_K", "embed_dim", "K20", 1, best_val_ppl=64.0,
                 embed_dim=20, wall_time_s=100.0),
        _val_row("grow_K", "embed_dim", "K20", 2, best_val_ppl=float("nan"),
                 embed_dim=20, wall_time_s=10.0),
    ]

    point = aggregate_validation_points(rows)[0]

    assert point["observed_seeds"] == [0, 1, 2]
    assert point["paired_seeds"] == [1]
    assert point["n_paired_seeds"] == 1
    assert point["missing_val_seeds"] == [2]
    assert point["n_missing_val_seeds"] == 1
    assert point["missing_wall_time_seeds"] == [0]
    assert point["n_missing_wall_time_seeds"] == 1
    assert point["val_bits_per_token_seeds"] == pytest.approx([6.0])
    assert point["wall_time_seeds"] == pytest.approx([100.0])
    assert point["n_val_seeds"] == 1
    assert point["val_bits_per_token_mean"] == pytest.approx(6.0)
    assert point["wall_time_mean"] == pytest.approx(100.0)


def test_aggregate_validation_points_rejects_duplicate_seed_identity():
    rows = [
        _val_row("grow_K", "embed_dim", "K20", 1, best_val_ppl=16.0,
                 embed_dim=20, wall_time_s=10.0),
        _val_row("grow_K", "embed_dim", "K20", 1, best_val_ppl=64.0,
                 embed_dim=20, wall_time_s=100.0),
    ]

    with pytest.raises(ValueError, match="duplicate seed 1"):
        aggregate_validation_points(rows)


@pytest.mark.parametrize("bad_seed", [True, -1, 1.5, "1"])
def test_aggregate_validation_points_rejects_nonexact_seed_identity(bad_seed):
    row = _val_row("grow_K", "embed_dim", "K20", bad_seed, best_val_ppl=16.0,
                   embed_dim=20, wall_time_s=10.0)

    with pytest.raises(ValueError, match="exact nonnegative integer seed"):
        aggregate_validation_points([row])


def test_aggregate_validation_points_omits_group_without_paired_seed():
    rows = [
        _val_row("grow_K", "embed_dim", "K20", 0, best_val_ppl=16.0,
                 embed_dim=20, wall_time_s=None),
        _val_row("grow_K", "embed_dim", "K20", 1, best_val_ppl=float("nan"),
                 embed_dim=20, wall_time_s=10.0),
    ]

    assert aggregate_validation_points(rows) == []


def test_pareto_frontier_kwargs_pins_sorted_points_and_excludes_inference_route():
    rows = _grow_k_capacity_scaling_rows() + _inference_capacity_scaling_rows()
    points = aggregate_validation_points(rows)
    result = pareto_frontier_kwargs(points)
    assert result is not None
    assert result["points"]["n_params"].tolist() == [1000.0, 2000.0, 4000.0, 8000.0]
    assert result["points"]["bits_per_token"].tolist() == pytest.approx([5.0, 4.0, 2.0, 1.0])
    assert result["points"]["wall_time"].tolist() == pytest.approx([11.0, 22.0, 32.0, 42.0])
    assert result["points"]["label"] == ["K20", "K40", "K80", "K100"]   # the 5 inference points excluded


# --------------------------------------------------------------------------- PB-07: persisted-artifact
# adapters for the ablation forest and joint-LR grid figures (vfe3.viz.sweep_adapters).
#
# ablation_forest_kwargs reads persisted, aligned val_token_nats.pt tensors bound by a per-marker
# byte/tensor identity (sha256 + size + numel + dtype); lr_grid_heatmap_kwargs reads the accumulated
# sweep rows and the Cartesian metadata. Both are pure fixtures -- no model is built.


def _write_forest_cell(sweep_dir: Path, label: str, tensor: torch.Tensor, *, paired: bool = True):
    r"""One successful forest cell: an aligned per-token nats tensor plus a marker that records the
    exact byte/tensor identity the adapter re-verifies before trusting the file."""
    cell = sweep_dir / label.replace("=", "_").replace(",", "_")
    cell.mkdir(parents=True, exist_ok=True)
    tpath = cell / "val_token_nats.pt"
    torch.save(tensor, tpath)
    sha = hashlib.sha256(tpath.read_bytes()).hexdigest()
    marker = {
        "sweep": sweep_dir.name, "label": label, "status": "success", "error_kind": None,
        "primary_val_ppl": 10.0, "final_val_ppl": 10.0, "n_params": 100, "seed": 6,
        "paired_token_bootstrap": bool(paired),
        "val_token_nats_path": "val_token_nats.pt" if paired else None,
        "val_token_nats_sha256": sha if paired else None,
        "val_token_nats_size_bytes": tpath.stat().st_size if paired else None,
        "val_token_nats_numel": int(tensor.numel()) if paired else None,
        "val_token_nats_dtype": str(tensor.dtype) if paired else None,
    }
    (cell / "ablation_result.json").write_text(json.dumps(marker), encoding="utf-8")
    return cell, tpath


def test_ablation_forest_kwargs_rows_match_paired_bootstrap_oracle(tmp_path):
    torch.manual_seed(0)
    sweep = tmp_path / "component_ablation_forest"
    baseline = torch.rand(64, dtype=torch.float32)
    arms = {"baseline": baseline,
            "head_mixer_off": baseline + 0.1 * torch.rand(64),
            "precision_attention_off": baseline + 0.2 * torch.rand(64)}
    for label, arm in arms.items():
        _write_forest_cell(sweep, label, arm)

    out = ablation_forest_kwargs(sweep, "baseline")
    assert out is not None
    rows = {r["label"]: r for r in out["rows"]}
    assert set(rows) == set(arms)
    for label, arm in arms.items():
        band = bootstrap_token_ce_band(arm, baseline, seed=0)      # oracle: same seed as the adapter
        assert rows[label]["delta"] == pytest.approx(band["delta"] / math.log(2.0))
        assert rows[label]["lo"] == pytest.approx(band["lo"] / math.log(2.0))
        assert rows[label]["hi"] == pytest.approx(band["hi"] / math.log(2.0))


def test_ablation_forest_kwargs_none_without_baseline_marker(tmp_path):
    sweep = tmp_path / "component_ablation_forest"
    _write_forest_cell(sweep, "head_mixer_off", torch.rand(32, dtype=torch.float32))
    assert ablation_forest_kwargs(sweep, "baseline") is None


def test_ablation_forest_kwargs_none_on_shape_mismatch(tmp_path):
    sweep = tmp_path / "component_ablation_forest"
    _write_forest_cell(sweep, "baseline", torch.rand(64, dtype=torch.float32))
    _write_forest_cell(sweep, "head_mixer_off", torch.rand(32, dtype=torch.float32))  # length differs
    assert ablation_forest_kwargs(sweep, "baseline") is None


def test_ablation_forest_kwargs_rejects_same_shape_finite_replacement(tmp_path):
    r"""Overwriting a valid tensor with DIFFERENT finite values of the same shape/dtype leaves the
    marker's recorded digest stale, so the adapter must reject the whole figure (digest changed)."""
    sweep = tmp_path / "component_ablation_forest"
    _write_forest_cell(sweep, "baseline", torch.ones(64, dtype=torch.float32))
    _cell, tpath = _write_forest_cell(sweep, "head_mixer_off", torch.ones(64, dtype=torch.float32))
    torch.save(torch.full((64,), 2.0, dtype=torch.float32), tpath)   # same size/dtype, new bytes
    assert ablation_forest_kwargs(sweep, "baseline") is None


def _grid_row(mu, sigma, ppl):
    return {"label": f"mu={mu:g},sigma={sigma:g}", "primary_val_ppl": ppl,
            "overrides": {"e_q_mu_lr": mu, "e_q_sigma_lr": sigma}}


def _complete_grid_rows(xs, ys):
    return [_grid_row(mu, sigma, 100.0 + 10 * i + j)
            for i, sigma in enumerate(ys) for j, mu in enumerate(xs)]


def test_lr_grid_heatmap_kwargs_cartesian_complete_and_sorted(tmp_path):
    xs, ys = [0.5, 0.9], [0.0, 0.001]
    out = lr_grid_heatmap_kwargs(_complete_grid_rows(xs, ys),
                                 "e_q_mu_lr", "e_q_sigma_lr", xs, ys, (0.9, 0.001))
    assert out is not None
    grid = out["grid"]
    assert grid["x"].tolist() == [0.5, 0.9]
    assert grid["y"].tolist() == [0.0, 0.001]
    assert grid["z"].tolist() == [[100.0, 101.0], [110.0, 111.0]]     # z[iy][ix] from primary_val_ppl
    assert grid["xlabel"] == "e_q_mu_lr" and grid["ylabel"] == "e_q_sigma_lr"
    assert grid["baseline"] == (0.9, 0.001)


def test_lr_grid_heatmap_kwargs_none_on_duplicate_cell(tmp_path):
    xs, ys = [0.5, 0.9], [0.0, 0.001]
    rows = _complete_grid_rows(xs, ys) + [_grid_row(0.5, 0.0, 200.0)]  # duplicate (0.5, 0.0)
    assert lr_grid_heatmap_kwargs(rows, "e_q_mu_lr", "e_q_sigma_lr", xs, ys, (0.9, 0.001)) is None


def test_lr_grid_heatmap_kwargs_none_on_missing_cell(tmp_path):
    xs, ys = [0.5, 0.9], [0.0, 0.001]
    rows = _complete_grid_rows(xs, ys)[:-1]                            # 3 of 4 cells present
    assert lr_grid_heatmap_kwargs(rows, "e_q_mu_lr", "e_q_sigma_lr", xs, ys, (0.9, 0.001)) is None


def test_lr_grid_heatmap_kwargs_none_on_nonfinite_value(tmp_path):
    xs, ys = [0.5, 0.9], [0.0, 0.001]
    rows = _complete_grid_rows(xs, ys)
    rows[0] = _grid_row(0.5, 0.0, float("inf"))
    assert lr_grid_heatmap_kwargs(rows, "e_q_mu_lr", "e_q_sigma_lr", xs, ys, (0.9, 0.001)) is None


def test_emit_registered_ablation_forest_and_lr_grid_figures(tmp_path):
    torch.manual_seed(0)
    fig_dir = tmp_path / "figures"
    sweep = tmp_path / "component_ablation_forest"
    baseline = torch.rand(64, dtype=torch.float32)
    _write_forest_cell(sweep, "baseline", baseline)
    _write_forest_cell(sweep, "head_mixer_off", baseline + 0.1 * torch.rand(64))
    xs, ys = [0.5, 0.9], [0.0, 0.001]
    ctx = {"sweep_dir": sweep, "rows": _complete_grid_rows(xs, ys), "baseline_label": "baseline",
           "grid_x": "e_q_mu_lr", "grid_y": "e_q_sigma_lr",
           "grid_x_values": xs, "grid_y_values": ys, "baseline": (0.9, 0.001)}
    specs = [
        FigureSpec("ablation_forest", "ablation_forest.png",
                   lambda c: ablation_forest_kwargs(c["sweep_dir"], c["baseline_label"])),
        FigureSpec("lr_grid_heatmap", "lr_grid_heatmap.png",
                   lambda c: lr_grid_heatmap_kwargs(c["rows"], c["grid_x"], c["grid_y"],
                                                    c["grid_x_values"], c["grid_y_values"], c["baseline"])),
    ]
    emit_registered_figures(specs, ctx, fig_dir)
    assert (fig_dir / "ablation_forest.png").stat().st_size > 0
    assert (fig_dir / "lr_grid_heatmap.png").stat().st_size > 0


def test_emit_ablation_forest_skipped_when_baseline_absent(tmp_path):
    fig_dir = tmp_path / "figures"
    sweep = tmp_path / "component_ablation_forest"
    _write_forest_cell(sweep, "head_mixer_off", torch.rand(16, dtype=torch.float32))  # no baseline arm
    specs = [FigureSpec("ablation_forest", "ablation_forest.png",
                        lambda c: ablation_forest_kwargs(c["sweep_dir"], c["baseline_label"]))]
    emit_registered_figures(specs, {"sweep_dir": sweep, "baseline_label": "baseline"}, fig_dir)
    assert not (fig_dir / "ablation_forest.png").exists()
