r"""Gauge equivariance of the executed projected routing and readout (report figure input).

Under the passive frame action :math:`(U_i,\mu_i,\Sigma_i)\mapsto(g_iU_i,g_i\mu_i,g_i\Sigma_ig_i^\top)`
the reference moments are fixed, so the score, the weights and the pulled-back readout must agree
to float64 roundoff. The negative control moves the beliefs but not the frames and must fail.
"""
from tests.profiles import tiny_config

import pytest
import torch

from vfe3.model.model import VFEModel
from vfe3.viz import extract

_ROUTES = {
    "predictive_diagonal": dict(),
    "predictive_full_rope": dict(family="gaussian_full", decode_mode="full_chunked",
                                 rope_mode="frame", n_layers=2),
    "gradient_euler": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                           attention_residual=False),
    "gradient_closed_form": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                                 attention_residual=False, family="gaussian_full",
                                 decode_mode="full_chunked", relational_receiver_update="closed_form"),
}


def _model(**overrides: object) -> VFEModel:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(**overrides)).eval()
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)       # nontrivial realized frames
    return model


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_projected_score_and_readout_are_gauge_equivariant(route: str) -> None:
    model = _model(**_ROUTES[route])
    tokens = torch.tensor([[1, 4, 2]])
    result = extract.projected_gauge_equivariance(model, tokens, n_draws=2)
    residuals = result["residuals"]
    expected = ({"routing_score", "routing_weights", "readout_mean", "readout_covariance"}
                if model.cfg.attention_mode == "projected_predictive" else
                {"routing_score", "routing_weights", "receiver_force", "receiver_update_mean",
                 "receiver_update_covariance", "receiver_energy"})
    assert set(residuals) == expected
    for key, values in residuals.items():
        assert len(values) == model.cfg.n_layers
        assert max(values) < 1e-10, (key, values)


def test_moving_beliefs_without_frames_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    monkeypatch.setattr(extract, "_act_on_frame", lambda g, frame: frame)
    residuals = extract.projected_gauge_equivariance(
        model, torch.tensor([[1, 4, 2]]), n_draws=1)["residuals"]
    assert residuals["routing_score"][0] > 1e-3
    assert residuals["readout_mean"][0] > 1e-3
