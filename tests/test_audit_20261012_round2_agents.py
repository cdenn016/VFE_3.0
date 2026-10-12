r"""Round-2 audit of 2026-10-12: model, agent and routing items 6, 8 and 9. Tiny CPU cases only."""
from __future__ import annotations

import math
import warnings

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model import minimal_agent
from vfe3.model.model import VFEModel


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _model(**overrides: object) -> VFEModel:
    cfg = tiny_config(**{"vocab_size": 7, "embed_dim": 4, "n_heads": 2, "max_seq_len": 4, **overrides})
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def _tokens() -> torch.Tensor:
    return torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("readout", ["kernel", "minimal_agent"])
def test_captured_and_uncaptured_forwards_agree_with_a_variance_handoff(readout: str) -> None:
    r"""The prior-anchored floor :math:`f\Sigma_p` reads the layer prior, so with ``prior_handoff_sigma > 0`` a plain
    forward must hand the variance off exactly as a captured one does (item 8)."""
    model = _model(n_layers=3, predictive_readout=readout, prior_handoff_sigma=0.5,
                   predictive_covariance_floor=0.3, sigma_init=2.0)
    assert "prior_handoff_sigma" not in model.cfg.inactive_fields()
    model.eval()
    ids = _tokens()
    with torch.no_grad():
        plain, plain_logits = model.forward_beliefs(ids, return_logits=True)
        captured, captured_logits = model.forward_beliefs(ids, return_logits=True, capture={"diagnostic": {}})
    assert torch.equal(plain.sigma, captured.sigma)
    assert torch.equal(plain.mu, captured.mu)
    assert torch.equal(plain_logits, captured_logits)


def test_variance_handoff_stays_inert_without_a_floored_readout() -> None:
    cfg = tiny_config(predictive_covariance="passthrough", prior_handoff_sigma=0.5)
    assert {"prior_handoff_rho", "prior_handoff_sigma"} <= set(cfg.inactive_fields())
    assert "prior_handoff_rho" in tiny_config(prior_handoff_sigma=0.5).inactive_fields()


@pytest.mark.parametrize("readout", ["kernel", "minimal_agent"])
def test_mstep_capture_runs_no_relational_diagnostics(readout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The M-step capture that ``block_mlp_em_weight > 0`` creates no longer switches on the readout diagnostics
    (item 6); a diagnostic snapshot still does, and the loss is unchanged."""
    model = _model(n_layers=2, predictive_readout=readout, use_block_mlp=True, block_mlp_mode="prototype_agents",
                   block_mlp_inference="mean_field", block_mlp_em_weight=0.1, attention_residual=readout == "kernel",
                   **({"minimal_agent_inference": "mean_field"} if readout == "minimal_agent" else {}))
    flags: list = []
    original = minimal_agent.minimal_agent_readout

    def spy(*args: object, **kwargs: object) -> object:
        flags.append(kwargs.get("diagnostics"))
        return original(*args, **kwargs)

    monkeypatch.setattr(minimal_agent, "minimal_agent_readout", spy)
    ids = _tokens()
    _, loss, _ = model(ids, ids)
    assert torch.isfinite(loss)
    if readout == "minimal_agent":
        assert flags == [False, False]
    snapshot = model.build_diagnostic_snapshot(ids, return_logits=False)
    assert "beta_pre" in snapshot.relational_capture
    if readout == "minimal_agent":
        assert flags[2:] == [True, True]


def _templates(M: int = 3, K: int = 4) -> tuple:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(M, K, generator=g, dtype=torch.float64), torch.zeros(M, K, dtype=torch.float64),
            torch.zeros(M, dtype=torch.float64))


@pytest.mark.parametrize("entry", ["inf_variance", "nan_mean", "negative_variance"])
def test_prototype_exact_diagonal_flags_non_finite_and_negative_inputs(entry: str) -> None:
    r"""The exact diagonal tile tested only :math:`W > 0`: an infinite variance or a NaN mean gave a non-finite
    posterior, and a slightly negative variance with :math:`c + \lambda > 0` gave :math:`C^+ < 0`, all unflagged
    (item 9)."""
    from vfe3.model import prototype_agents as pa
    nu, log_lambda, log_pi = _templates()
    xi = torch.randn(2, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    C = torch.ones(2, 4, dtype=torch.float64)
    if entry == "inf_variance":
        C[1, 2] = math.inf
    elif entry == "nan_mean":
        xi[0, 1] = math.nan
    else:
        C[1, 2] = -0.5                                                   # c + lambda = 0.5 > 0
    with pytest.raises(ValueError, match="positive and finite"):
        pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi)


def test_prototype_mean_field_flags_a_negative_variance_with_a_positive_power_matrix() -> None:
    r""":math:`c = -0.1` with :math:`1 + cD > 0` passed the power-matrix check and returned :math:`C^+ < 0` (item 9)."""
    from vfe3.model import prototype_agents as pa
    nu, log_lambda, log_pi = _templates()
    xi = torch.randn(2, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    C = torch.ones(2, 4, dtype=torch.float64)
    C[1, 2] = -0.1                                                       # D = 1 (lambda = 1): 1 + cD = 0.9
    with pytest.raises(ValueError, match="not positive semidefinite or not finite"):
        pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi)


@pytest.mark.parametrize("mean_field", [False, True])
def test_prototype_block_posterior_flags_a_non_finite_mean(mean_field: bool) -> None:
    from vfe3.model import prototype_agents as pa
    nu, log_lambda, log_pi = _templates()
    xi = torch.randn(2, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    xi[1, 0] = math.inf
    C = torch.eye(4, dtype=torch.float64).expand(2, 4, 4).clone()
    posterior = pa.prototype_mean_field_posterior if mean_field else pa.prototype_exact_posterior
    with pytest.raises(ValueError, match="not finite"):
        posterior(xi, C, nu, log_lambda, log_pi)


def test_prototype_flags_are_deferred_inside_the_scope() -> None:
    r"""The new flags ride the existing deferred host check and raise when the scope closes."""
    from vfe3.contracts import deferred_host_checks
    from vfe3.model import prototype_agents as pa
    nu, log_lambda, log_pi = _templates()
    xi = torch.randn(2, 4, dtype=torch.float64)
    C = torch.ones(2, 4, dtype=torch.float64)
    C[0, 0] = -0.5
    with pytest.raises(ValueError, match="positive and finite"):
        with deferred_host_checks():
            pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi)
