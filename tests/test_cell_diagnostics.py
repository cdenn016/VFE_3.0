r"""Tests for the 2026-06-21 experiment-readiness harness build (S1/S2/S3).

Covers:
  * ``head_mixer_gauge_residual`` (B9-a) -- the builder-break gauge certificate that
    DISTINGUISHES the exact-equivariant tied gauge from the strictly-broken untied one
    (the instrument A2/EXP-9 needs; the existing ``gauge_equivariance_residual`` is blind
    to this because it co-transforms a supplied Omega rather than rebuilding the operator).
  * locally selected registered sweeps validate and build a real VFEModel for every cell (S1).
  * the new per-cell diagnostic columns are registered in the sweep CSV schema (S3).

Device-agnostic (CPU by default; honors VFE3_TEST_DEVICE like the rest of the suite).
"""
from tests.profiles import config_values, test_config
import os
from types import SimpleNamespace

import pytest
import torch

import ablation
from vfe3.metrics import head_mixer_gauge_residual
from vfe3.model.model import VFEModel
from vfe3.model.head_mixer import HeadMixer
from vfe3.geometry.groups import get_group

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _drifted_mixer_model(group) -> SimpleNamespace:
    """Standalone gauge/mixer fixture, independent of attention-mode admission."""
    mixer = HeadMixer(group.irrep_dims, group.irrep_labels, family="gaussian_full").to(DEVICE)
    with torch.no_grad():
        for delta in mixer.mixer_deltas:
            delta.normal_(0.0, 0.3)
    return SimpleNamespace(group=group, head_mixer=mixer)


def test_builder_residual_distinguishes_tied_from_untied_gauge():
    r"""mix(g.mu, g.Sigma.g^T) == g.mix(.).g^T for IN-group g: holds (eps) under the TIED so_n gauge
    (two copies of the SO(3) defining irrep, isotypic mixer), breaks under the UNTIED per-head gauge
    of block_glk once the mixer has drifted from I."""
    torch.manual_seed(0)
    n, k = 5, 6
    mu = torch.randn(n, k, device=DEVICE)
    base = torch.randn(n, k, k, device=DEVICE)
    sigma = base @ base.transpose(-1, -2) + k * torch.eye(k, device=DEVICE)      # SPD full cov

    tied = _drifted_mixer_model(get_group("so_n")(6, group_n=3, irrep_spec=[("l1", 2)]))
    untied = _drifted_mixer_model(get_group("block_glk")(6, 2))

    r_tied = head_mixer_gauge_residual(mu, sigma, tied.head_mixer, tied.group, seed=0)
    r_untied = head_mixer_gauge_residual(mu, sigma, untied.head_mixer, untied.group, seed=0)

    tied_max = torch.cat([r_tied["mu_residual"], r_tied["sigma_residual"]]).max().item()
    untied_med = torch.cat([r_untied["mu_residual"], r_untied["sigma_residual"]]).median().item()

    assert tied_max < 1e-4, f"tied gauge must stay equivariant (mixer commutes), got max {tied_max}"
    assert untied_med > 1e-2, f"untied gauge must break as the mixer drifts, got median {untied_med}"


def test_builder_residual_zero_for_identity_mixer():
    r"""A mixer still at its identity init is trivially equivariant under ANY gauge (no drift,
    no break) -- the residual is exactly zero on both arms."""
    torch.manual_seed(1)
    n, k = 4, 4
    mu = torch.randn(n, k, device=DEVICE)
    base = torch.randn(n, k, k, device=DEVICE)
    sigma = base @ base.transpose(-1, -2) + k * torch.eye(k, device=DEVICE)
    cfg = test_config(vocab_size=64, embed_dim=4, n_heads=2, max_seq_len=8,
                     gauge_group="block_glk", use_head_mixer=True, family="gaussian_full")
    model = VFEModel(cfg).to(DEVICE)                                 # deltas at zero init
    r = head_mixer_gauge_residual(mu, sigma, model.head_mixer, model.group, seed=0)
    assert torch.cat([r["mu_residual"], r["sigma_residual"]]).max().item() < 1e-5


@pytest.mark.parametrize("sweep_name", [
    "block_mlp_expansion", "block_mlp_activation", "block_mlp_dropout",
])
def test_experiment_arms_validate_and_build(
    sweep_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Registered MLP sweep arms build on a local tiny enabling profile."""
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(use_block_mlp=True))
    ablation.validate_sweeps([sweep_name])                          # guard #1: real VFE3Config fields
    runs = ablation.make_run_overrides(sweep_name)
    assert len(runs) >= 2
    for _label, overrides in runs:
        cfg_dict = ablation._cell_cfg_dict(
            {**overrides, "vocab_size": 7, "max_seq_len": 3}, seed=0, max_steps=1)
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
    needed = {"attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in"}
    assert needed.issubset(set(ablation._CSV_COLUMNS))


def test_across_layer_trace_includes_rank_one_residual():
    r"""EXP-7: the per-layer belief trace surfaces the Dong rank-one residual r(X) by depth."""
    from vfe3.viz.extract import across_layer_belief_trace
    cfg = test_config(vocab_size=64, embed_dim=4, n_heads=2, max_seq_len=12, n_layers=3,
                     gauge_group="block_glk", use_head_mixer=True)
    model = VFEModel(cfg).to(DEVICE)
    tr = across_layer_belief_trace(model, torch.randint(0, 64, (1, 12), device=DEVICE))
    r = tr["rank_one_residual"]
    assert r.shape == (3,)
    assert bool((r >= 0).all()) and bool((r <= 1.0 + 1e-5).all())
