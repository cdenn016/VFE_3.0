"""Source-based factorization of valid but ill-conditioned projected Gaussians."""
import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_score
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import projected_gaussian_kl, projected_moments


def _inputs(device: str | torch.device = "cpu") -> tuple:
    generator = torch.Generator().manual_seed(4)
    Q = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    R = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    S = (Q @ torch.diag(torch.tensor([1., .5, 1.e-4])) @ R.T)[None].to(device)
    mu = torch.tensor([[[[.2, .3, .7]], [[.1, -.2, .4]]]], device=device)
    sigma = torch.ones_like(mu)
    inverse = torch.eye(3, device=device).expand(1, 2, 1, 3, 3)
    return mu, sigma, inverse, S


@pytest.mark.parametrize("role", ["q", "k"])
@pytest.mark.parametrize("full", [False, True])
def test_full_rank_source_recovers_lost_gram_precision(device: torch.device, role: str, full: bool) -> None:
    mu, sigma, inverse, S = _inputs(device)
    if full:
        sigma = torch.diag_embed(sigma)
    S = S.requires_grad_(True)
    sigma.requires_grad_()
    inverse.requires_grad_()
    identity = torch.eye(3, device=device)[None]
    S_q, S_k = (S, identity) if role == "q" else (identity, S)
    aq, Gq = projected_moments(mu, sigma, inverse, S_q)
    ak, Gk = projected_moments(mu, sigma, inverse, S_k)
    with pytest.raises(ValueError, match="positive definite"):
        projected_gaussian_kl(aq, Gq, ak, Gk)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        score = projected_gaussian_kl(
            aq, Gq, ak, Gk,
            projection_q=(sigma, inverse, S_q), projection_k=(sigma, inverse, S_k))
    aq64, Gq64 = projected_moments(mu.double(), sigma.double(), inverse.double(), S_q.double())
    ak64, Gk64 = projected_moments(mu.double(), sigma.double(), inverse.double(), S_k.double())
    reference = projected_gaussian_kl(aq64, Gq64, ak64, Gk64)
    torch.testing.assert_close(score.double(), reference, rtol=2e-5, atol=2e-5)
    actual_grads = torch.autograd.grad(score.sum(), (S, sigma, inverse), retain_graph=True)
    reference_grads = torch.autograd.grad(reference.sum(), (S, sigma, inverse))
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, reference_grad, rtol=5e-4, atol=5e-3)


def test_actual_relational_router_uses_source_factorization(device: torch.device) -> None:
    mu, sigma, inverse, S = _inputs(device)
    ops = RelationalOperators([3], 4, identity_content=True).to(device)
    with torch.no_grad():
        ops.raw_score_q.copy_(S)
        ops.raw_score_k.copy_(torch.eye(3, device=device)[None])
    frame = CanonicalFrameContext(inverse, inverse, block_diagonal=True)
    score = relational_score(mu.flatten(-2), sigma.flatten(-2),
                             mu.flatten(-2).detach(), sigma.flatten(-2), frame, ops)
    assert torch.isfinite(score).all()
    gradients = torch.autograd.grad(score.sum(), (ops.raw_score_q, ops.raw_score_k))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_rank_deficient_source_still_fails_without_repair(device: torch.device) -> None:
    mu, sigma, inverse, _ = _inputs(device)
    S = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]]], device=device)
    a, G = projected_moments(mu, sigma, inverse, S)
    with pytest.raises(ValueError, match="rank"):
        projected_gaussian_kl(a, G, a, G,
                             projection_q=(sigma, inverse, S), projection_k=(sigma, inverse, S))


def test_mixed_success_and_fallback_batch_has_finite_gradients(device: torch.device) -> None:
    mu, sigma, inverse, S = _inputs(device)
    inverse = inverse.clone()
    inverse[:, 1] = torch.linalg.inv(S)
    inverse.requires_grad_()
    S.requires_grad_()
    a, G = projected_moments(mu, sigma, inverse, S)
    assert torch.linalg.cholesky_ex(G)[1].ne(0).sum() == 1
    identity = torch.eye(3, device=device)[None]
    ak, Gk = projected_moments(mu, sigma, inverse, identity)
    score = projected_gaussian_kl(a, G, ak, Gk,
                                 projection_q=(sigma, inverse, S),
                                 projection_k=(sigma, inverse, identity))
    gradients = torch.autograd.grad(score.sum(), (S, inverse))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
