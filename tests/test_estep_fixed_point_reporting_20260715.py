"""Regression tests for E-step naming and fixed-point reporting buildout."""

from tests.profiles import test_config
import copy

import pytest
import torch

from vfe3.inference.e_step import canonical_e_step_update
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import collect_estep_depth_sensitivity
from vfe3.viz.extract import e_step_belief_trace, e_step_fixed_point_diagnostics


def _mm_model(update: str) -> VFEModel:
    cfg = test_config(
        vocab_size=12,
        embed_dim=4,
        n_heads=1,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_step_update=update,
        mm_damping=0.75,
        e_phi_lr=0.0,
        pos_phi="none",
        seed=0,
    )
    return VFEModel(cfg)




def test_group_product_accepts_the_flat_tied_phi_route() -> None:
    assert test_config(pos_phi="learned").pos_phi == "learned"
    assert test_config(pos_phi="learned").transport_mode == "flat"
    assert test_config(pos_phi="none").pos_phi == "none"


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_phi_mstep_max_matrix_norm_must_be_positive_or_none(value: float) -> None:
    assert test_config(phi_mstep_max_matrix_norm=None).phi_mstep_max_matrix_norm is None
    with pytest.raises(ValueError, match="phi_mstep_max_matrix_norm"):
        test_config(phi_mstep_max_matrix_norm=value)


def test_one_step_ahead_residual_is_distinct_from_configured_last_step() -> None:
    from vfe3 import metrics

    model = _mm_model("mm_exact")
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    trace = e_step_belief_trace(model, tokens, n_iter=2)
    diag = e_step_fixed_point_diagnostics(model, tokens)

    configured = metrics.estep_residuals(
        trace["mu"][:2],
        trace["sigma"][:2],
        trace["phi"][:2],
        diagonal=model.cfg.diagonal_covariance,
    )
    fixed_point_mu = (trace["mu"][2] - trace["mu"][1]).square().mean().sqrt()
    assert diag["estep_r_mu_last"] == pytest.approx(float(configured["r_mu"][-1].mean()))
    assert diag["estep_fp_mu_rms"] == pytest.approx(float(fixed_point_mu))
    assert diag["estep_fp_kl"] >= 0.0
    assert diag["estep_target_gap"] >= 0.0


def test_depth_sensitivity_marks_trained_depth_and_restores_state() -> None:
    model = _mm_model("mm_exact")
    model.train()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()

    record = collect_estep_depth_sensitivity(model, tokens, targets, depths=[0, 1, 2])

    assert record["trained_depth"] == 1
    assert [point["depth"] for point in record["points"]] == [0, 1, 2]
    assert all("ce" in point and "free_energy_per_token" in point for point in record["points"])
    assert model.training
    assert model.cfg.n_e_steps == 1
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0.0, atol=0.0)
