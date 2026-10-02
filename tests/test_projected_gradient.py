"""Tiny CPU contracts for frozen-router full-link receiver inference."""

from dataclasses import asdict
from importlib import import_module
from typing import Any

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.metrics import covariance_frozen_at_encode
from vfe3.run_artifacts import _pure_path_report, _relational_artifact_fields


@pytest.fixture(autouse=True)
def _single_cpu_thread() -> Any:
    """Keep all model witnesses small and independent of host thread count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _cfg(**overrides: Any) -> VFE3Config:
    values = dict(
        attention_mode="projected_gradient",


        attention_residual=False,
        predictive_covariance="passthrough",
        predictive_covariance_centering=False,
        n_e_steps=3,
        mu_init_std=0.3,
        sigma_init=0.8,
        phi_scale=0.02,
        pos_phi_scale=0.01,
        e_q_mu_lr=0.2,
        e_q_sigma_lr=0.05,
        e_sigma_q_trust=5.0,
        lambda_alpha_mode="constant",
        skip_belief_sigma_update=False,
    )
    values.update(overrides)
    return tiny_config(**values)


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[0, 1, 2]], device="cpu"), torch.tensor([[1, 2, 3]], device="cpu")


def _last_ce(model: VFEModel) -> torch.Tensor:
    ids, targets = _tokens()
    _, logits = model.forward_beliefs(ids, return_logits=True)
    return torch.nn.functional.cross_entropy(logits[:, -1], targets[:, -1])


@pytest.mark.parametrize(
    ("family", "n_layers", "n_e_steps"),
    [("gaussian_diagonal", 1, 3), ("gaussian_full", 2, 2)],
)
def test_outer_ce_reaches_every_layer_map_and_earlier_source(
    family: str,
    n_layers: int,
    n_e_steps: int,
) -> None:
    model = VFEModel(_cfg(family=family, n_layers=n_layers, n_e_steps=n_e_steps))
    _last_ce(model).backward()
    assert len(model.relational_operators_by_layer) == n_layers
    for operators in model.relational_operators_by_layer:
        assert operators.input_scope == operators.output_scope == "full"
        assert operators.readout_mode == "full_link_update"
        for name in ("raw_score_q", "raw_score_k", "T", "O"):
            derivative = getattr(operators, name).grad
            assert derivative is not None and torch.isfinite(derivative).all()
            assert derivative.abs().sum() > 0, name
        assert operators.T.grad[0, :, 2:].abs().sum() > 0
        assert operators.O.grad[0, 2:, :].abs().sum() > 0
    # Linear decoding has a separate weight table. Position zero can influence this
    # last-position-only CE only through prefix source/router credit.
    source_gradient = model.prior_bank.mu_embed.grad[0]
    assert torch.isfinite(source_gradient).all() and source_gradient.abs().sum() > 0


def test_frozen_context_is_built_once_per_layer_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("vfe3.inference.e_step")
    build = module.build_full_link_context
    gradients = module.full_link_gradients
    contexts = []
    gradient_contexts = []
    receiver_means = []

    def tracked_build(*args: Any, **kwargs: Any) -> Any:
        context = build(*args, **kwargs)
        contexts.append(context)
        return context

    def tracked_gradients(mu: torch.Tensor, sigma: torch.Tensor, context: Any) -> Any:
        gradient_contexts.append(context)
        receiver_means.append(mu.detach().clone())
        return gradients(mu, sigma, context)

    monkeypatch.setattr(module, "build_full_link_context", tracked_build)
    monkeypatch.setattr(module, "full_link_gradients", tracked_gradients)
    model = VFEModel(_cfg(n_layers=2, n_e_steps=3))
    ids, _ = _tokens()
    belief, _ = model.forward_beliefs(ids, return_logits=True)
    assert len(contexts) == 2
    assert contexts[0] is not contexts[1]
    assert len(gradient_contexts) == 6
    for layer in range(2):
        assert all(context is contexts[layer]
                   for context in gradient_contexts[3 * layer:3 * (layer + 1)])
        assert not torch.equal(receiver_means[3 * layer], receiver_means[3 * layer + 1])
    assert torch.isfinite(belief.mu).all() and torch.isfinite(belief.sigma).all()


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_mean_and_covariance_steps_are_live_and_sigma_skip_freezes(
    family: str,
) -> None:
    cfg = _cfg(family=family)
    model = VFEModel(cfg)
    ids, _ = _tokens()
    initial = model.prior_bank.encode(ids)
    changed, _ = model.forward_beliefs(ids)
    assert not torch.equal(changed.mu, initial.mu)
    assert not torch.equal(changed.sigma, initial.sigma)
    frozen = VFEModel(_cfg(family=family, skip_belief_sigma_update=True))
    frozen.load_state_dict(model.state_dict())
    skipped, _ = frozen.forward_beliefs(ids)
    assert not torch.equal(skipped.mu, initial.mu)
    torch.testing.assert_close(skipped.sigma, initial.sigma, rtol=0, atol=0)
    if family == "gaussian_full":
        assert torch.linalg.eigvalsh(changed.sigma).min() > 0
    else:
        assert changed.sigma.min() > 0


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
def test_multiple_episodes_preserve_prefix_causality(family: str) -> None:
    model = VFEModel(_cfg(family=family, n_layers=2, n_e_steps=3))
    ids, _ = _tokens()
    future_changed = ids.clone()
    future_changed[0, -1] = 5
    with torch.no_grad():
        before, _ = model.forward_beliefs(ids)
        after, _ = model.forward_beliefs(future_changed)
    torch.testing.assert_close(before.mu[:, :-1], after.mu[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(before.sigma[:, :-1], after.sigma[:, :-1], rtol=0, atol=0)
    assert not torch.equal(before.mu[:, -1], after.mu[:, -1])


@pytest.mark.parametrize("name", ["T", "O"])
def test_outer_map_derivative_matches_end_to_end_finite_difference(name: str) -> None:
    model = VFEModel(_cfg(n_e_steps=2))
    parameter = getattr(model.relational_operators, name)
    derivative = torch.autograd.grad(_last_ce(model), parameter)[0]
    index = tuple(int(component) for component in
                  torch.unravel_index(derivative.abs().argmax(), derivative.shape))
    assert derivative[index].abs() > 1e-6
    original = parameter.detach().clone()
    delta = 2e-3
    try:
        with torch.no_grad():
            parameter[index] += delta
            plus = _last_ce(model)
            parameter.copy_(original)
            parameter[index] -= delta
            minus = _last_ce(model)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    torch.testing.assert_close(
        derivative[index], (plus - minus) / (2 * delta), rtol=0.03, atol=2e-5)


@pytest.mark.parametrize("autocast", [False, True])
def test_eval_without_grad_is_finite_and_captures_frozen_router(autocast: bool) -> None:
    model = VFEModel(_cfg(family="gaussian_full", n_layers=2, n_e_steps=2)).eval()
    ids, _ = _tokens()
    capture = {}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        belief, logits = model.forward_beliefs(ids, return_logits=True, capture=capture)
    assert not belief.mu.requires_grad and not belief.sigma.requires_grad
    assert torch.isfinite(belief.mu).all() and torch.isfinite(belief.sigma).all()
    assert torch.isfinite(logits).all()
    beta = capture["relational_capture"]["beta_pre"]
    torch.testing.assert_close(beta.sum(-1), torch.ones_like(beta.sum(-1)))
    assert torch.count_nonzero(beta.triu(1)) == 0


def test_config_state_and_control_activity_round_trip() -> None:
    cfg = _cfg(n_layers=2)
    restored_cfg = config_from_serialized(serialize_config(cfg), source="projected-gradient-test")
    assert asdict(restored_cfg) == asdict(cfg)
    inactive = cfg.inactive_fields()
    assert "skip_belief_sigma_update" not in inactive
    assert "e_q_mu_lr" not in inactive
    assert "e_q_sigma_lr" not in inactive
    assert "relational_receiver_noise" not in inactive
    for field in ("predictive_covariance", "predictive_covariance_centering"):
        assert field in inactive
    model = VFEModel(cfg)
    restored = VFEModel(restored_cfg)
    restored.load_state_dict(model.state_dict(), strict=True)
    ids, _ = _tokens()
    with torch.no_grad():
        one, _ = model.forward_beliefs(ids)
        two, _ = restored.forward_beliefs(ids)
    torch.testing.assert_close(one.mu, two.mu, rtol=0, atol=0)
    torch.testing.assert_close(one.sigma, two.sigma, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"family": "gaussian_diagonal_exact"},
        {"family": "gaussian_frame_diagonal"},
        {"lambda_alpha_mode": "state_dependent"},
    ],
)
def test_unsupported_gradient_envelopes_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _cfg(**overrides)


@pytest.mark.parametrize("noise", [0.0, -1.0, float("inf"), float("nan"), True])
def test_receiving_noise_requires_finite_positive_model_variance(noise: Any) -> None:
    with pytest.raises(ValueError):
        _cfg(relational_receiver_noise=noise)


@pytest.mark.parametrize("skip_sigma", [False, True])
def test_covariance_and_artifact_reports_follow_executed_receiver_branch(
    skip_sigma: bool,
) -> None:
    cfg = _cfg(skip_belief_sigma_update=skip_sigma)
    assert covariance_frozen_at_encode(cfg) is skip_sigma
    artifact = _relational_artifact_fields(cfg)["relational_operators"]
    assert artifact["source_beliefs"] == "live_episode_initial"
    assert artifact["mean_update"] == "receiver_fisher_step"
    assert artifact["covariance_update"] == ("passthrough" if skip_sigma else "fisher_step")
    assert artifact["input_dimension"] == artifact["head_output_dimension"] == 4
    report = _pure_path_report(cfg, [])
    assert report["pure_flags"]["gradient_e_step_update"] is True
    assert report["pure_flags"]["full_sigma_update"] is (not skip_sigma)
    assert report["config_toggles"]["fixed_covariance_surrogate"] is skip_sigma
    retraction = report["config_toggles"]["spd_retraction_route"]
    assert (retraction == "not_applicable") is skip_sigma


def test_diagnostics_report_the_captured_frozen_receiver_cost() -> None:
    from vfe3.gradients.full_link import full_link_cost

    model = VFEModel(_cfg(family="gaussian_full"))
    ids, _ = _tokens()
    snapshot = model.build_diagnostic_snapshot(ids)
    diagnostics = model.diagnostics(ids, snapshot=snapshot)
    capture = snapshot.relational_capture
    assert capture is not None
    context = capture["full_link_context"]
    final = snapshot.layer_converged[0]
    actual_cost = full_link_cost(final.mu, final.sigma, context)
    torch.testing.assert_close(
        capture["relational_receiver_cost_after"], actual_cost, rtol=1e-5, atol=1e-6)
    before = float(capture["relational_receiver_cost_before"])
    after = float(capture["relational_receiver_cost_after"])
    assert diagnostics["relational_receiver_cost_before"] == before
    assert diagnostics["relational_receiver_cost_after"] == after
    assert diagnostics["relational_receiver_cost_delta"] == after - before
    assert diagnostics["relational_receiver_noise"] == model.cfg.relational_receiver_noise
    assert diagnostics["relational_legacy_global_f_available"] == 0.0
    assert snapshot.trace_free_energy.numel() == 0
