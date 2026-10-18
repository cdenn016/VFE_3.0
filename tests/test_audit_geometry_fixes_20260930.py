r"""Tiny CPU regressions for predictive spread."""

from tests.profiles import make_relational_operators
import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_score


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("compact_frame", [False, True])
def test_predictive_spread_survives_common_translation(
    diagonal:      bool,
    compact_frame: bool,
) -> None:
    mu = torch.tensor([11586., 11587., 11588.]).reshape(1, 3, 1).expand(1, 3, 4).clone()
    sigma = torch.full_like(mu, 15.)
    if not diagonal:
        sigma = torch.diag_embed(sigma)
    if compact_frame:
        eye = torch.eye(2).expand(1, 3, 2, 2, 2).clone()
    else:
        eye = torch.eye(4).expand(1, 3, 4, 4).clone()
    frame = CanonicalFrameContext(eye, eye, block_diagonal=compact_frame)
    beta = torch.tensor([[1., 0., 0.], [.5, .5, 0.], [.25, .25, .5]])
    beta = beta.expand(1, 2, 3, 3).clone()
    ops = make_relational_operators([2, 2], 6,
                              readout_mode="predictive_content",
                              predictive_covariance="moment_matched")
    # The prior equals the belief covariance here, so the floor f Sigma_p adds 1e-4 * 15 = 0.0015 to every variance.
    actual = ops.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=sigma)
    expected = torch.tensor([15.0015, 15.2515, 15.689]).reshape(1, 3, 1).expand(1, 3, 4)
    actual_diagonal = actual if diagonal else actual.diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(actual_diagonal, expected, atol=2e-6, rtol=1e-7)
    if not diagonal:
        assert torch.linalg.eigvalsh(actual).min() > 0.
    assert torch.isfinite(relational_score(mu, actual, mu, actual, frame, ops)).all()


def _centered_predictive_reference(
    mu:    torch.Tensor,
    sigma: torch.Tensor,
    beta:  torch.Tensor,
    T:     torch.Tensor,
    O:     torch.Tensor,
    frame: CanonicalFrameContext,
    floor: float,
    residual: bool,
    prior: torch.Tensor,
) -> torch.Tensor:
    r"""Direct centered covariance, retaining shared sources and residual cross terms, plus the prior-anchored floor
    ``floor * prior`` in local coordinates."""
    C = frame.inverse @ sigma @ frame.inverse.transpose(-1, -2)
    C = 0.5 * (C + C.transpose(-1, -2))
    xi = (frame.inverse @ mu.unsqueeze(-1)).squeeze(-1)
    xi = xi - xi[..., :1, :]
    M = O @ T
    values = torch.einsum("hkl,bjl->bjhk", M, xi)
    rows = []
    for i in range(mu.shape[-2]):
        cov = torch.zeros(4, 4, dtype=mu.dtype)
        for h in range(2):
            mean = (beta[0, h, i, :, None] * values[0, :, h]).sum(dim=0)
            centered = values[0, :, h] - mean
            cov = cov + torch.einsum("j,jk,jl->kl", beta[0, h, i], centered, centered)
            if residual:
                cross = beta[0, h, i, i] * M[h] @ C[0, i]
                cov = cov + cross + cross.T
            for g in range(2):
                weights = beta[0, h, i] if h == g else beta[0, h, i] * beta[0, g, i]
                for j in range(mu.shape[-2]):
                    cov = cov + weights[j] * M[h] @ C[0, j] @ M[g].T
        if residual:
            cov = cov + C[0, i]
        local = frame.forward[0, i] @ cov @ frame.forward[0, i].T
        rows.append(0.5 * (local + local.T) + floor * prior[0, i])
    return torch.stack(rows).unsqueeze(0)


@pytest.mark.parametrize("residual", [False, True])
def test_centered_predictive_covariance_retains_correlations_and_gradients(residual: bool) -> None:
    forward = torch.tensor([[2., .5, 0., 0.], [0., 1., 0., 0.],
                            [0., 0., .5, .25], [0., 0., 0., 1.]]).expand(1, 3, 4, 4).clone()
    frame = CanonicalFrameContext(forward, torch.linalg.inv(forward))
    xi = 12000. + torch.tensor([[0., .5, -.5, 1.], [1., -.5, .5, 0.], [2., 1., 0., -.5]])
    mu = (forward @ xi.unsqueeze(-1)).squeeze(-1).requires_grad_()
    factor = torch.tensor([[2., 0., 0., 0.], [.5, 1., 0., 0.],
                           [.25, -.5, 1.5, 0.], [0., .25, .5, 1.]])
    C = (factor @ factor.T).expand(1, 3, 4, 4).clone()
    sigma = (forward @ C @ forward.transpose(-1, -2)).requires_grad_()
    logits = torch.tensor([[[[.1, -.2, .3], [.4, .2, -.5], [-.3, .5, .1]],
                             [[-.2, .4, .1], [.5, -.1, .2], [.2, -.4, .5]]]], requires_grad=True)
    ops = make_relational_operators([2, 2], 6,
                              readout_mode="predictive_content", attention_residual=residual,
                              predictive_covariance="moment_matched")
    with torch.no_grad():
        ops.T.copy_(torch.tensor([[[1., .25, -.5, 0.], [0., .5, .25, 1.]],
                                 [[.5, 0., 1., .25], [-.25, 1., 0., .5]]]))
        ops.O.copy_(torch.tensor([[[1., .25], [.5, 1.], [.25, -.5], [0., .5]],
                                 [[.25, .5], [1., -.25], [.5, 1.], [1., .25]]]))
    actual = ops.predictive_covariance_update(mu, sigma, logits.softmax(-1), frame, floor_anchor=sigma.detach())
    inputs = (mu, sigma, logits, ops.T, ops.O)
    reference_inputs = tuple(t.detach().double().requires_grad_() for t in inputs)
    mu64, sigma64, logits64, T64, O64 = reference_inputs
    frame64 = CanonicalFrameContext(frame.forward.double(), frame.inverse.double())
    expected = _centered_predictive_reference(mu64, sigma64, logits64.softmax(-1), T64, O64,
                                              frame64, ops.predictive_covariance_floor, residual, sigma64.detach())
    torch.testing.assert_close(actual.double(), expected, atol=8e-6, rtol=3e-6)
    weight = torch.arange(1., 49.).reshape_as(actual) / 48.
    gradients = torch.autograd.grad((actual * weight).sum(), inputs)
    reference_gradients = torch.autograd.grad((expected * weight.double()).sum(), reference_inputs)
    for actual_gradient, expected_gradient in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual_gradient.double(), expected_gradient, atol=3e-5, rtol=2e-5)


