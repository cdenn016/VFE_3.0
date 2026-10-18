"""Source-based factorization of valid but ill-conditioned projected Gaussians."""
from tests.profiles import make_relational_operators
import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_score
from vfe3.relational_scoring import _strict_cholesky, projected_gaussian_kl, projected_moments


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
    # The relative KL error is bounded by the whitened error of the recovered float32 factor, O(eps kappa(X)) with
    # kappa(X) = 1e4 (4.8e-4 for this fixture). Measured: 1.0e-4 on the key role under the shift of Fukaya et al.
    # eq. (67) (audit 2026-10-16 F42; 6.2e-6 under the former larger shift). Over 300 random orientations of these
    # singular values the two shifts give the same error distribution (median 7e-5, p90 2.5e-4).
    torch.testing.assert_close(score.double(), reference, rtol=3e-4, atol=2e-5)
    actual_grads = torch.autograd.grad(score.sum(), (S, sigma, inverse), retain_graph=True)
    reference_grads = torch.autograd.grad(reference.sum(), (S, sigma, inverse))
    # fp32 recovery bound on the gradients. The source root X = S Sigma^{1/2} has kappa(X) = 1e4 and its float32
    # shifted CholeskyQR3 factor a whitened error <= eps kappa (1.2e-3). dKL/dG = (P - P (G_q + d d^T) P)/2 with
    # P = (X X^T)^{-1} is at most quadratic in P, whose relative error is <= 2 eps kappa, so it moves by
    # <= 4 eps kappa relative; the float32 means add <= 2 eps kappa through d. The chain to (S, Sigma, U^{-1}) is
    # the same linear map for both, so to first order |dg| <= 6 eps kappa |g| = 7.2e-3 |g| normwise (measured
    # <= 2.8e-4; elementwise the error reaches 2.6e-2 of entries that are small against |g|, since the gradient
    # spans 2.9e12 down to O(1) on the key role).
    singular = torch.linalg.svdvals(S.detach().double()[0])
    bound = 6 * torch.finfo(torch.float32).eps * float(singular[0] / singular[-1])
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.isfinite(actual_grad).all()
        assert float((actual_grad.double() - reference_grad).norm()) <= bound * float(reference_grad.norm())


def test_actual_relational_router_uses_source_factorization(device: torch.device) -> None:
    mu, sigma, inverse, S = _inputs(device)
    ops = make_relational_operators([3], 4).to(device)
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


@pytest.mark.usefixtures("condition_recovery_on")
def test_recovery_keeps_healthy_pair_tiles_in_working_dtype(device: torch.device) -> None:
    r"""Healthy-healthy pair tiles beside a recovered entry stay in the working dtype.

    Thirty-three query tokens span two 32-row pair tiles and only query token 0 is
    ill-conditioned, so the second row tile never touches a recovered factor and
    must pair working-dtype means with the working-dtype key precision.
    """
    generator = torch.Generator().manual_seed(5)
    mu_q, mu_k = (torch.randn(1, 33, 1, 2, generator=generator).to(device) for _ in range(2))
    sigma_q, sigma_k = (torch.ones(1, 33, 1, 2, device=device) for _ in range(2))
    sigma_q[0, 0, 0, 1] = 1e-4  # cond(G) = 1e4 exceeds the float32 certificate
    variables = tuple(x.requires_grad_() for x in (mu_q, mu_k, sigma_q, sigma_k))
    inverse = torch.eye(2, device=device).expand(1, 33, 1, 2, 2)
    S = torch.eye(2, device=device)[None]
    aq, Gq = projected_moments(mu_q, sigma_q, inverse, S)
    ak, Gk = projected_moments(mu_k, sigma_k, inverse, S)
    source_q, source_k = (sigma_q, inverse, S, mu_q), (sigma_k, inverse, S, mu_k)
    assert _strict_cholesky(Gq.detach(), source_q)[1].nonzero().tolist() == [[0, 0, 0]]
    assert _strict_cholesky(Gk.detach(), source_k)[1] is None
    score = projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=source_q, projection_k=source_k)
    assert score.dtype == torch.float32
    aq64, Gq64 = projected_moments(mu_q.double(), sigma_q.double(), inverse.double(), S.double())
    ak64, Gk64 = projected_moments(mu_k.double(), sigma_k.double(), inverse.double(), S.double())
    reference = projected_gaussian_kl(aq64, Gq64, ak64, Gk64)
    torch.testing.assert_close(score.double(), reference, rtol=2e-5, atol=2e-5)
    for actual_grad, reference_grad in zip(
            torch.autograd.grad(score.sum(), variables, retain_graph=True),
            torch.autograd.grad(reference.sum(), variables)):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=5e-4, atol=5e-3)
