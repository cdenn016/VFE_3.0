"""Reporting regressions for the executed predictive readout, using tiny CPU models."""

from tests.profiles import test_config
import math
import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    collect_beta_channel_decomposition, collect_context_sensitivity, collect_estep_character,
)
from vfe3.viz import extract, figures, report


def _model() -> VFEModel:
    torch.manual_seed(71)
    return VFEModel(test_config(
        vocab_size=7, embed_dim=4, n_heads=2, batch_size=1, max_seq_len=4,
        n_layers=1, n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
        skip_belief_sigma_update=True, e_phi_lr=0.0, oracle_unroll_grad=True,
        pos_phi="learned", use_head_mixer=True, use_block_mlp=True, block_mlp_expansion=1,
        attention_mode="projected_predictive",

    )).eval()


def test_predictive_trace_uses_real_states_without_conditional_costs() -> None:
    model = _model()
    tokens = torch.tensor([[0, 1, 2, 3]])
    snapshot = model.build_diagnostic_snapshot(tokens)
    trace = extract.e_step_belief_trace(model, tokens, snapshot=snapshot)
    assert trace["free_energy_available"] is False
    assert trace["conditional_cost_available"] is False
    torch.testing.assert_close(trace["mu"][-1], snapshot.layer_converged[0].mu[0])
    state = extract.readout_state(model, tokens, snapshot=snapshot)
    torch.testing.assert_close(state["sigma"], snapshot.layer_converged[0].sigma[0])
    torch.testing.assert_close(state["energy"], snapshot.conditional_capture["score_energy_pre"])
    assert "selfdiv_klmax_frac" not in state["guard"]
    fig = figures.plot_iterate_trajectory(trace)
    assert any("unavailable" in t.get_text().lower() for ax in fig.axes for t in ax.texts)
    assert fig.axes[-1].lines
    figures.plt.close(fig)


def test_predictive_mechanism_probes_use_configured_forward() -> None:
    model = _model().train()
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    character = collect_estep_character(model, tokens, depths=[1, 2, 4])
    assert character["available"] is True
    assert [p["belief_depth"] for p in character["points"]] == [1]
    assert character["points"][0]["pair_precision_share"] is None
    context = collect_context_sensitivity(model, tokens, n_replicates=2)
    assert context["available"] is True
    assert context["stages"]["raw_prior"] == 0.0
    assert context["stages"]["predictive_content"] > 0.0
    channel = collect_beta_channel_decomposition(model, tokens, targets)
    assert channel["available"] is True
    assert all(math.isfinite(v) for v in channel["ce"].values())
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.training and model.cfg.n_e_steps == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        expected = F.cross_entropy(model(tokens).flatten(0, 1), targets.flatten())
    assert channel["ce"]["full"] == pytest.approx(float(expected), abs=1e-6)


def test_predictive_report_emits_state_figures_and_finite_readout_panels(tmp_path: Path) -> None:
    model = _model()
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    written = report.generate_figures(
        tmp_path, model=model, loader=[(tokens, targets)], device=torch.device("cpu"),
        max_sequences=1,
    )
    names = {p.name for p in written}
    assert {"numerical_trust.png", "per_layer_diagnostics.png", "head_block_norms.png"} <= names
    # Route gates (audit k160 Section 4): one readout and no E-step iterates, covariance frozen at
    # encode, flat transport; the retired gauge_head_specialization is replaced by head_block_norms.
    assert names.isdisjoint({"iterate_trajectory.png", "belief_trajectories.png",
                             "belief_spectrum.png", "spd_ellipses.png",
                             "gauge_head_specialization.png", "holonomy_curvature.png"})
    assert "gauge_equivariance.png" not in names
    fig = figures.plot_per_layer_diagnostics(model.diagnostics_per_layer(tokens))
    assert all(any(torch.isfinite(torch.as_tensor(line.get_ydata())).any() for line in ax.lines)
               for ax in fig.axes)
    assert not any("Free-energy budget" in ax.get_title() for ax in fig.axes)
    figures.plt.close(fig)


@pytest.mark.parametrize("route, ceiling_binds", [
    (dict(attention_mode="projected_predictive", predictive_covariance="moment_matched",
          n_e_steps=1), False),
    (dict(attention_mode="projected_gradient", skip_belief_sigma_update=False, n_e_steps=2,
          e_q_sigma_lr=0.5), True),
])
def test_report_draws_sigma_max_only_where_it_bounds_the_belief(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: dict, ceiling_binds: bool) -> None:
    # sigma_max caps only the sigma E-step retraction. The moment-matched predictive covariance is
    # an unclamped readout, so with a token table above the cap every plotted variance exceeds it.
    # N == K == 4 also pins that the guard reads the (N, K) diagonal as variances, not a K x K matrix.
    torch.manual_seed(71)
    model = VFEModel(test_config(
        vocab_size=7, embed_dim=4, n_heads=2, batch_size=1, max_seq_len=4,
        n_layers=2, family="gaussian_diagonal", gauge_group="block_glk",
        e_phi_lr=0.0, pos_phi="learned", use_block_mlp=True, block_mlp_expansion=1,
        sigma_init=15.0, sigma_max=2.0, **route,
    )).eval()
    assert ("sigma_max" not in model.cfg.inactive_fields()) is ceiling_binds
    seen = {}
    spectrum, trust = figures.plot_belief_spectrum, figures.plot_numerical_trust
    monkeypatch.setattr(figures, "plot_belief_spectrum",
                        lambda sigma, **kw: seen.update(sigma=sigma, spectrum=kw) or spectrum(sigma, **kw))
    monkeypatch.setattr(figures, "plot_numerical_trust",
                        lambda guard, *a, **kw: seen.update(guard=guard) or trust(guard, *a, **kw))
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    written = report.generate_figures(tmp_path, model=model, loader=[(tokens, targets)],
                                      device=torch.device("cpu"), max_sequences=1)
    assert {"belief_spectrum.png", "numerical_trust.png"} <= {p.name for p in written}
    if ceiling_binds:
        assert seen["spectrum"]["sigma_max"] == 2.0
        assert float(seen["sigma"].max()) <= 2.0
        # Receiver descent can move some coordinates below the cap; the guard must
        # report the actual spectrum, rather than assume every variance is saturated.
        expected_fraction = float(((seen["sigma"] - 2.0).abs() < 2.0e-3).float().mean())
        assert seen["guard"]["sigma_ceil_frac"] == expected_fraction
    else:
        assert float(seen["sigma"].min()) > 2.0
        assert seen["spectrum"]["sigma_max"] is None
        assert "sigma_ceil_frac" not in seen["guard"]


def test_existing_unavailable_probes_are_refreshed_without_changing_valid_records(tmp_path: Path) -> None:
    model = _model()
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    for name in ("estep_depth_sensitivity", "phi_numerics", "estep_character",
                 "beta_channel_decomposition", "context_sensitivity"):
        (tmp_path / f"{name}.json").write_text('{"available": false}', encoding="utf-8")
    report._prepare_missing_saved_probes(
        model, [(tokens, targets)], tmp_path, torch.device("cpu"), logging.getLogger(__name__),
        split="test",
    )
    for name in ("estep_character", "beta_channel_decomposition", "context_sensitivity"):
        record = json.loads((tmp_path / f"{name}.json").read_text())
        assert record["available"] is True
        assert record["report_replay"]["split"] == "test"
        if name == "estep_character":                 # single readout: record kept, figure skipped
            assert not (tmp_path / f"{name}.png").exists()
        else:
            assert (tmp_path / f"{name}.png").stat().st_size > 0
    # A subsequent call has no work; even the placeholder depth and phi records are preserved.
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report._prepare_missing_saved_probes(
        model, [], tmp_path, torch.device("cpu"), logging.getLogger(__name__))
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
