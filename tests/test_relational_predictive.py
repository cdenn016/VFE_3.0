"""Tiny predictive mixture references and live outer-gradient contracts."""
from tests.profiles import make_relational_operators
from dataclasses import asdict

import pytest
import torch

from vfe3.config import config_from_serialized, serialize_config
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _relational_artifact_fields
from tests.test_relational_filtering import _fixture
from tests.test_relational_integration import _cfg, _tokens


@pytest.mark.parametrize("output_scope", ["full"])
def test_predictive_dense_reference_and_source_gradients(output_scope: str) -> None:
    mu, sigma, _, frame, _, prior = _fixture()
    ops = make_relational_operators([2, 2], 11,
                              readout_mode="predictive_content").double()
    with torch.no_grad():
        ops.T.copy_(torch.arange(16, dtype=mu.dtype).reshape(2, 2, 4) / 9 + .1)
        ops.O.copy_(torch.arange(ops.O.numel(), dtype=mu.dtype).reshape_as(ops.O) / 8 + .1)
    capture = {}
    actual = relational_predictive_readout(mu, sigma, frame, ops, tau=.9, log_prior=prior, capture=capture)
    beta = capture["beta_pre"]
    xi = (frame.inverse @ mu.unsqueeze(-1)).squeeze(-1)
    expected = torch.zeros_like(mu)
    for i in range(3):
        body = torch.zeros(4, dtype=mu.dtype)
        for h in range(2):
            content = sum(beta[0, h, i, j] * (ops.T[h] @ xi[0, j]) for j in range(3))
            if output_scope == "full":
                body = body + ops.O[h] @ content
            else:
                body[h*2:(h+1)*2] = ops.O[h] @ content
        expected[0, i] = frame.forward[0, i] @ body
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert torch.isfinite(torch.autograd.grad(actual.square().sum(), mu)[0]).all()
    assert "conditional_context" not in capture


def test_predictive_readout_uses_a_separate_value_frame() -> None:
    mu, sigma, _, score_frame, _, prior = _fixture()
    ops = make_relational_operators([2, 2], 11,
                              readout_mode="predictive_content").double()
    with torch.no_grad():
        ops.T.copy_(torch.arange(16, dtype=mu.dtype).reshape(2, 2, 4) / 9 + .1)
        ops.O.copy_(torch.arange(ops.O.numel(), dtype=mu.dtype).reshape_as(ops.O) / 8 + .1)
    value_frame = type(score_frame)(score_frame.inverse, score_frame.forward)
    capture = {}
    actual = relational_predictive_readout(
        mu, sigma, score_frame, ops, tau=.9, log_prior=prior,
        value_frame=value_frame, capture=capture)
    expected = ops.predictive_content(mu, capture["beta_pre"], value_frame)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert not torch.allclose(actual, ops.predictive_content(mu, capture["beta_pre"], score_frame))

@pytest.mark.parametrize("name", ["raw_score_q", "raw_score_k", "T", "O"])
def test_predictive_finite_difference(name: str) -> None:
    mu, sigma, _, frame, _, prior = _fixture()
    ops = make_relational_operators([2, 2], 11,
                              readout_mode="predictive_content").double()
    parameter = getattr(ops, name)
    def objective() -> torch.Tensor:
        return relational_predictive_readout(mu, sigma, frame, ops, tau=.9, log_prior=prior)[0, 2].square().sum()
    derivative = torch.autograd.grad(objective(), parameter)[0]
    index = (0,) * parameter.ndim
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter[index] += 1e-6
    plus = objective().detach()
    with torch.no_grad():
        parameter.copy_(original)
        parameter[index] -= 1e-6
    minus = objective().detach()
    with torch.no_grad():
        parameter.copy_(original)
    torch.testing.assert_close(derivative[index], (plus-minus)/2e-6, rtol=2e-5, atol=2e-8)
    assert derivative.abs().sum() > 1e-8
def test_predictive_live_model_config_optimizer_and_diagnostics() -> None:
    cfg = _cfg(attention_mode="projected_predictive")
    model = VFEModel(cfg)
    ids, targets = _tokens()
    cap = {}
    belief, logits = model.forward_beliefs(ids, return_logits=True, capture=cap)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    for parameter in model.relational_operators.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.relational_operators.O.grad[0, 2:].abs().sum() > 0
    assert model.conditional_semantics == "relational_predictive_content"
    assert _relational_artifact_fields(cfg)["relational_operators"]["total_parameters"] == 64
    assert asdict(config_from_serialized(serialize_config(cfg), source="predictive")) == asdict(cfg)
    assert "conditional_context" not in cap["relational_capture"]


@pytest.mark.parametrize("rope_mode", ["frame", "frame_value"])
def test_predictive_full_scope_accepts_rope_and_keeps_outer_gradients(
    rope_mode: str,
) -> None:
    cfg = _cfg(

        attention_mode="projected_predictive",

        rope_mode=rope_mode,
    )
    model = VFEModel(cfg)
    ids, targets = _tokens()
    cap = {}
    _, logits = model.forward_beliefs(ids, return_logits=True, capture=cap)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    assert torch.isfinite(logits).all()
    for parameter in model.relational_operators.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.isfinite(cap["relational_capture"]["beta_pre"]).all()

def test_retired_identity_content_option_is_rejected() -> None:
    with pytest.raises(TypeError, match="relational_identity_content"):
        _cfg(attention_mode="projected_predictive", relational_identity_content=True)


def test_predictive_bypasses_fisher_step_and_uses_source_content_graph() -> None:
    settings = dict(attention_mode="projected_predictive",
                    use_block_mlp=False)
    first = VFEModel(_cfg(**settings, e_q_mu_lr=0.0))
    second = VFEModel(_cfg(**settings, e_q_mu_lr=9.0))
    second.load_state_dict(first.state_dict())
    ids, _ = _tokens()
    one = first.forward_beliefs(ids)[0]
    two = second.forward_beliefs(ids)[0]
    torch.testing.assert_close(one.mu, two.mu, rtol=0, atol=0)
    torch.testing.assert_close(one.sigma, two.sigma, rtol=0, atol=0)
    mu, _, _, frame, _, _ = _fixture()
    ops = make_relational_operators([2, 2], 11,
                              readout_mode="predictive_content").double()
    beta = torch.zeros(1, 2, 3, 3, dtype=mu.dtype)
    beta[..., 2, 0] = 1.
    out = ops.predictive_content(mu, beta, frame)
    derivative = torch.autograd.grad(out[0, 2].sum(), mu)[0]
    assert derivative[0, 0].abs().sum() > 0
    assert derivative[0, 2].abs().sum() == 0
