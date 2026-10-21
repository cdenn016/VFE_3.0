r"""Tests for the 2026-06-21 experiment-readiness harness build (S1/S2/S3).

Covers:
  * locally selected registered sweeps validate and build a real VFEModel for every cell (S1).
  * the new per-cell diagnostic columns are registered in the sweep CSV schema (S3).

Device-agnostic (CPU by default; honors VFE3_TEST_DEVICE like the rest of the suite).
"""
from tests.profiles import config_values, test_config
import os

import pytest
import torch

import ablation
from vfe3.model.model import VFEModel

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.mark.parametrize("sweep_name", [
    "block_mlp_expansion", "block_mlp_activation", "block_mlp_dropout",
])
def test_experiment_arms_validate_and_build(
    sweep_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Registered MLP sweep arms build on a local tiny enabling profile."""
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(use_block_mlp=True))
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)               # the driver switch is the user's; off here
    ablation.validate_sweeps([sweep_name])                          # guard #1: real VFE3Config fields
    runs = ablation.make_run_overrides(sweep_name)
    assert len(runs) >= 2
    for _label, overrides in runs:
        cfg_dict = ablation._cell_cfg_dict(
            {**overrides, "vocab_size": 7, "max_seq_len": 3, "warmup_steps": 1}, seed=0, max_steps=1)
        model = VFEModel(test_config(**cfg_dict))                    # config + build validation
        assert model is not None


def test_gauge_transport_off_yields_identity_frame():
    r"""gauge_transport='off' must coerce the frame to Omega=I (phi_scale forced to 0).

    Asserted on the config directly: the ``gauge_transport`` sweep was retired in the 2026-09-18
    sweep cleanup, but the coercion it exercised is a live config contract."""
    cfg = test_config(vocab_size=64, embed_dim=4, n_heads=2, max_seq_len=16,
                     gauge_transport="off", phi_scale=1.0)
    assert cfg.gauge_transport == "off"
    assert cfg.phi_scale == 0.0


def test_diagnostic_csv_columns_registered():
    r"""S3: the per-cell diagnostic scalars are columns of sweep_results.csv."""
    needed = {"attn_entropy", "omega_identity_dev", "gauge_resid_in"}
    assert needed.issubset(set(ablation._CSV_COLUMNS))


def test_across_layer_trace_includes_rank_one_residual():
    r"""EXP-7: the per-layer belief trace surfaces the Dong rank-one residual r(X) by depth."""
    from vfe3.viz.extract import across_layer_belief_trace
    cfg = test_config(vocab_size=64, embed_dim=4, n_heads=2, max_seq_len=12, n_layers=3,
                     gauge_group="block_glk")
    model = VFEModel(cfg).to(DEVICE)
    tr = across_layer_belief_trace(model, torch.randint(0, 64, (1, 12), device=DEVICE))
    r = tr["rank_one_residual"]
    assert r.shape == (3,)
    assert bool((r >= 0).all()) and bool((r <= 1.0 + 1e-5).all())
