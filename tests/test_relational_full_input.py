"""Whole-belief readouts with independently realized compact gauge blocks."""
from dataclasses import asdict

import pytest
import torch

from vfe3.config import serialize_config, config_from_serialized
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import RelationalOperators
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _relational_artifact_fields
from tests.test_relational_filtering import _fixture
from tests.test_relational_integration import _cfg, _tokens
from vfe3.relational_scoring import projected_moments, projected_gaussian_kl
from tests.test_relational_source_factorization import _inputs


def test_full_input_matches_dense_distribution_oracle() -> None:
    mu, sigma, _, frame, _, _ = _fixture()
    ops = RelationalOperators([2, 2], 11, input_scope="full").double()
    compact = CanonicalFrameContext(
        torch.stack([frame.forward[..., :2, :2], frame.forward[..., 2:, 2:]], -3),
        torch.stack([frame.inverse[..., :2, :2], frame.inverse[..., 2:, 2:]], -3),
        block_diagonal=True)
    actual = relational_score(mu, sigma, mu.detach(), sigma, compact, ops)
    expected = torch.empty_like(actual)
    S_q, S_k = ops.comparison_pair()
    for h in range(2):
        for i in range(3):
            for j in range(3):
                distributions = []
                for token, S in ((i, S_q[h]), (j, S_k[h])):
                    L = S @ frame.inverse[0, token]
                    distributions.append(torch.distributions.MultivariateNormal(
                        L @ mu[0, token], covariance_matrix=L @ torch.diag(sigma[0, token]) @ L.T))
                expected[0, h, i, j] = torch.distributions.kl_divergence(*distributions)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(ops.apply_content(mu, compact), mu)
    changed = mu.detach().clone()
    changed[..., 2:] += 2.
    assert not torch.allclose(actual[..., 0, :, :], relational_score(changed, sigma, mu.detach(), sigma, compact, ops)[..., 0, :, :])
    with torch.no_grad():
        ops.T[0, 0, 2] = 1.
    assert not torch.allclose(ops.apply_content(changed, compact)[..., :2], ops.apply_content(mu, compact)[..., :2])
def test_full_input_live_model_cross_block_gradients_and_metadata() -> None:
    cfg = _cfg(relational_input_scope="full")
    model = VFEModel(cfg)
    ids, targets = _tokens()
    _, logits = model.forward_beliefs(ids, return_logits=True)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    ops = model.relational_operators
    for p in (ops.raw_score_q, ops.raw_score_k, ops.T):
        assert torch.isfinite(p.grad).all()
        assert p.grad[0, :, 2:].abs().sum() > 0
    meta = _relational_artifact_fields(cfg)["relational_operators"]
    assert meta["total_parameters"] == sum(p.numel() for p in ops.parameters()) == 56
    assert meta["score_shape"] == [2, 2, 4]
    assert meta["content_shapes"] == [[2, 2, 4], [2, 2, 2]]
    restored = config_from_serialized(serialize_config(cfg), source="full input fixture")
    assert asdict(restored) == asdict(cfg)


def test_compact_full_input_source_factorization() -> None:
    mu, sigma, inverse, S = _inputs()
    mu = mu.flatten(-2).repeat(1, 1, 2).unsqueeze(-2).expand(1, 2, 2, 6)
    sigma = torch.ones_like(mu)
    inverse = inverse.expand(1, 2, 2, 3, 3).clone().requires_grad_()
    S = torch.cat([S, torch.zeros_like(S)], -1).expand(2, 3, 6).clone().requires_grad_()
    a, G = projected_moments(mu, sigma, inverse, S)
    identity = torch.eye(6)[:3].expand(2, 3, 6)
    ak, Gk = projected_moments(mu, sigma, inverse, identity)
    assert (torch.linalg.cholesky_ex(G)[1] != 0).any()
    score = projected_gaussian_kl(a, G, ak, Gk, projection_q=(sigma, inverse, S))
    a64, G64 = projected_moments(mu.double(), sigma.double(), inverse.double(), S.double())
    ak64, Gk64 = projected_moments(mu.double(), sigma.double(), inverse.double(), identity.double())
    ref = projected_gaussian_kl(a64, G64, ak64, Gk64)
    torch.testing.assert_close(score.double(), ref, rtol=2e-5, atol=2e-5)
    actual_grads = torch.autograd.grad(score.sum(), (S, inverse), retain_graph=True)
    reference_grads = torch.autograd.grad(ref.sum(), (S, inverse))
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-3)


def test_full_input_contains_head_local_scores_and_values() -> None:
    mu, sigma, _, frame, _, _ = _fixture()
    local = RelationalOperators([2, 2], 17).double()
    full = RelationalOperators([2, 2], 17, input_scope="full").double()
    with torch.no_grad():
        for source, target in ((local.raw_score_q, full.raw_score_q),
                               (local.raw_score_k, full.raw_score_k), (local.T, full.T)):
            target.zero_()
            for h in range(2):
                target[h, :, 2*h:2*h+2] = source[h]
        full.O.copy_(local.O)
    torch.testing.assert_close(relational_score(mu, sigma, mu.detach(), sigma, frame, full),
                               relational_score(mu, sigma, mu.detach(), sigma, frame, local))
    torch.testing.assert_close(full.apply_content(mu, frame), local.apply_content(mu, frame))


@pytest.mark.parametrize("name", ["raw_score_q", "raw_score_k", "T", "O"])
def test_full_input_conditional_outer_gradient_finite_difference(name: str) -> None:
    from vfe3.gradients.relational import compute_relational_filtering_gradients
    mu, sigma, omega, frame, _, prior = _fixture()
    ops = RelationalOperators([2, 2], 19, input_scope="full").double()
    # Stay off the strict zero-KL endpoint mask the identity content start sits on.
    with torch.no_grad():
        for _p in (ops.raw_score_q, ops.raw_score_k, ops.T, ops.O):
            _p.add_(.02 * torch.randn_like(_p))
    parameter = getattr(ops, name)
    index = (0, 0, 2) if name != "O" else (0, 1, 0)

    def objective() -> torch.Tensor:
        gradient, _ = compute_relational_filtering_gradients(
            mu, sigma, mu.detach() + .2, sigma, omega, ops, frame,
            log_prior=prior, tau=.9, create_graph=True)
        return gradient.square().sum()

    expected = torch.autograd.grad(objective(), parameter)[0][index]
    original = parameter[index].item()
    values = []
    for shift in (1e-5, -1e-5):
        with torch.no_grad():
            parameter[index] = original + shift
        values.append(objective().detach())
    with torch.no_grad():
        parameter[index] = original
    finite_difference = (values[0] - values[1]) / 2e-5
    torch.testing.assert_close(expected, finite_difference, rtol=2e-4, atol=2e-7)


@pytest.mark.parametrize("value", [True, "global"])
def test_invalid_input_scope_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="relational_input_scope"):
        _cfg(relational_input_scope=value)


@pytest.mark.parametrize("mode", ["projected_conditional", "projected_predictive", "projected_gradient"])
def test_unspecified_scopes_score_like_the_explicit_mode_defaults(mode: str) -> None:
    """None resolves the same read/write maps as the mode's explicit valid scopes."""
    scope = "head" if mode == "projected_conditional" else "full"
    automatic = _cfg(attention_mode=mode, relational_input_scope=None, relational_output_scope=None)
    explicit = _cfg(attention_mode=mode, relational_input_scope=scope, relational_output_scope=scope)
    mu, sigma, _, frame, _, _ = _fixture()
    operators = [
        RelationalOperators([2, 2], 17, readout_mode=cfg.relational_readout_mode,
                            input_scope=cfg.relational_input_scope, output_scope=cfg.relational_output_scope).double()
        for cfg in (automatic, explicit)
    ]
    torch.testing.assert_close(
        relational_score(mu, sigma, mu, sigma, frame, operators[0]),
        relational_score(mu, sigma, mu, sigma, frame, operators[1]), rtol=0, atol=0)
