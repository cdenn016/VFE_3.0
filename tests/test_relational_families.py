"""Tiny family-aware projected routing and conditional filtering contracts."""
import pytest
import torch

from tests.test_relational_filtering import _fixture
from tests.test_relational_integration import _cfg, _tokens
from vfe3.gradients.relational import relational_score
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators


@pytest.mark.parametrize("family", ["gaussian_full", "gaussian_diagonal_exact", "gaussian_frame_diagonal"])
@pytest.mark.parametrize("readout", ["conditional_update", "predictive_content"])
def test_gaussian_family_live_forward_backward(family: str, readout: str) -> None:
    torch.manual_seed(17)
    model = VFEModel(_cfg(family=family, relational_input_scope="full",
                         attention_mode=("projected_predictive" if readout == "predictive_content" else "projected_conditional"),
                         relational_output_scope="full" if readout == "predictive_content" else "head"))
    ids, targets = _tokens()
    capture = {}
    belief, logits = model.forward_beliefs(ids, return_logits=True, capture=capture)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    assert torch.isfinite(logits).all()
    if readout == "conditional_update":
        # Only head marginals enter the conditional cost, so no K-by-K pair grid.
        assert capture["relational_capture"]["conditional_context"].sigma_t.shape == (1, 4, 4, 2, 2, 2)
    assert belief.sigma.shape == ((1, 4, 4, 4) if family == "gaussian_full" else (1, 4, 4))
    for parameter in model.relational_operators.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    snapshot = model.build_diagnostic_snapshot(ids)
    assert torch.isfinite(snapshot.relational_capture["score_energy_pre"]).all()


@pytest.mark.parametrize("scope", ["head", "full"])
def test_full_covariance_score_matches_dense_marginals(scope: str) -> None:
    mu, _, _, frame, _, _ = _fixture()
    root = torch.tensor([[1., .2, .3, 0.], [0., .9, .2, .1],
                         [.1, 0., 1.1, .2], [0., .1, 0., .8]], dtype=mu.dtype)
    Sigma = (root @ root.T).expand(1, 3, 4, 4).clone().requires_grad_()
    ops = RelationalOperators([2, 2], 11, input_scope=scope).double()
    with torch.no_grad():
        ops.raw_score_q.add_(.08)
        ops.raw_score_k.sub_(.03)
    actual = relational_score(mu, Sigma, mu, Sigma, frame, ops)
    S_q, S_k = ops.comparison_pair()
    expected = torch.empty_like(actual)
    for h in range(2):
        sl = slice(0, 4) if scope == "full" else slice(h*2, (h+1)*2)
        L_q = S_q[h] @ frame.inverse[0, :, sl, sl]
        L_k = S_k[h] @ frame.inverse[0, :, sl, sl]
        for i in range(3):
            q = torch.distributions.MultivariateNormal(L_q[i] @ mu[0, i, sl],
                 covariance_matrix=L_q[i] @ Sigma[0, i, sl, sl] @ L_q[i].T)
            for j in range(3):
                k = torch.distributions.MultivariateNormal(L_k[j] @ mu[0, j, sl],
                     covariance_matrix=L_k[j] @ Sigma[0, j, sl, sl] @ L_k[j].T)
                expected[0, h, i, j] = torch.distributions.kl_divergence(q, k)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(torch.autograd.grad(actual.sum(), Sigma)[0]).all()


@pytest.mark.parametrize("family", ["gaussian_full", "gaussian_diagonal_exact", "gaussian_frame_diagonal"])
@pytest.mark.parametrize("readout", ["conditional_update", "predictive_content"])
@pytest.mark.parametrize("rope_on_value", [False, True])
def test_gaussian_rope_autocast_snapshot(family: str, readout: str, rope_on_value: bool) -> None:
    torch.manual_seed(9)
    model = VFEModel(_cfg(family=family, relational_input_scope="full",
                         attention_mode=("projected_predictive" if readout == "predictive_content" else "projected_conditional"),
                         relational_output_scope="full" if readout == "predictive_content" else "head",
                         pos_rotation="rope", rope_full_gauge=family == "gaussian_full",
                         rope_on_value=rope_on_value))
    ids, targets = _tokens()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, logits = model.forward_beliefs(ids, return_logits=True)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    for parameter in model.relational_operators.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        snapshot = model.build_diagnostic_snapshot(ids)
    if readout == "conditional_update" and family == "gaussian_frame_diagonal":
        context = snapshot.relational_capture["conditional_context"]
        expected = torch.diag_embed(context.sigma.reshape(*context.sigma.shape[:-1], 2, 2))
        expected = expected.unsqueeze(-5).expand_as(context.sigma_t)
        torch.testing.assert_close(context.sigma_t, expected, rtol=0, atol=0)
    diagnostics = model.diagnostics(ids, snapshot=snapshot)
    assert diagnostics["relational_projected_cov_min_eigenvalue"] > 0
    assert 0 < diagnostics["effective_rank"] <= 4.0001


@pytest.mark.parametrize("readout", ["conditional_update", "predictive_content"])
def test_intrinsic_family_does_not_pull_back_stored_coordinates_twice(readout: str) -> None:
    torch.manual_seed(3)
    model = VFEModel(_cfg(family="gaussian_frame_diagonal", relational_input_scope="full",
                         attention_mode=("projected_predictive" if readout == "predictive_content" else "projected_conditional"), use_head_mixer=False, use_block_mlp=False))
    ids, _ = _tokens()
    first = model.forward_beliefs(ids)[0].mu
    with torch.no_grad():
        model.prior_bank.phi_embed.add_(.1)
    second = model.forward_beliefs(ids)[0].mu
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["gaussian_full", "gaussian_diagonal_exact"])
@pytest.mark.parametrize("name", ["raw_score_q", "raw_score_k", "T", "O"])
def test_gaussian_conditional_outer_finite_difference(family: str, name: str) -> None:
    from vfe3.gradients.relational import compute_relational_filtering_gradients
    mu, sigma, omega, frame, _, prior = _fixture()
    if family == "gaussian_full":
        sigma = torch.diag_embed(sigma)
        sigma[..., 0, 1] = sigma[..., 1, 0] = .12
        sigma[..., 1, 2] = sigma[..., 2, 1] = .08
    ops = RelationalOperators([2, 2], 11, input_scope="full").double()
    with torch.no_grad():
        ops.T.add_(.04)
        ops.O.add_(.02)
    parameter = getattr(ops, name)
    def objective() -> torch.Tensor:
        grad, _ = compute_relational_filtering_gradients(
            mu, sigma, mu.detach() + .2, sigma, omega, ops, frame,
            family=family, log_prior=prior, tau=.9, create_graph=True)
        return (mu - .1 * grad)[0, 2].square().sum()
    analytic = torch.autograd.grad(objective(), parameter)[0]
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
    torch.testing.assert_close(analytic[index], (plus - minus) / 2e-6, atol=2e-8, rtol=2e-5)
    assert analytic.abs().sum() > 1e-8


def test_laplace_remains_outside_gaussian_projection_envelope() -> None:
    with pytest.raises(ValueError, match="Gaussian family"):
        _cfg(family="laplace_diagonal")



@pytest.mark.parametrize("readout", ["conditional_update", "predictive_content", "ordinary"])
def test_full_gaussian_active_per_coordinate_alpha(readout: str) -> None:
    model = VFEModel(_cfg(family="gaussian_full", relational_input_scope="full",
                         attention_mode=("transported" if readout == "ordinary" else
                                         "projected_predictive" if readout == "predictive_content"
                                         else "projected_conditional"),
                         lambda_alpha_mode="state_dependent_per_coord",
                         mstep_self_coupling_weight=.2, full_cov_kl_precision="fp64"))
    ids, targets = _tokens()
    _, loss, _ = model(ids, targets)
    loss.backward()
    assert torch.isfinite(loss)
    if readout != "ordinary":
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.relational_operators.parameters())
        snapshot = model.build_diagnostic_snapshot(ids)
        assert torch.isfinite(snapshot.relational_capture["score_energy_pre"]).all()
    else:
        assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
