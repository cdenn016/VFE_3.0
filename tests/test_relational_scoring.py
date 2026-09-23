"""Tiny exact Gaussian and gradient oracles for relational S/T/O operators."""

import importlib.util

import pytest
import torch


def test_exact_kernel_is_available() -> None:
    assert importlib.util.find_spec("vfe3.relational_scoring") is not None


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    mu = torch.randn(1, 3, 2, 3, generator=generator, dtype=torch.float64)
    sigma = torch.rand(mu.shape, generator=generator, dtype=mu.dtype) + 0.7
    U = torch.eye(3, dtype=mu.dtype).expand(1, 3, 2, 3, 3).clone()
    U[..., 0, 1] = 0.4
    U[:, 1, :, 1, 2] = -0.3
    return mu, sigma, U


@pytest.mark.parametrize("full", [False, True])
def test_projection_and_pairwise_kl_match_distribution_oracle(full: bool) -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    mu, sigma, U = _fixture()
    S = torch.tensor([[[1., .2, -.1], [0., 1., .3]]] * 2, dtype=mu.dtype)
    covariance = torch.diag_embed(sigma)
    if full:
        covariance = U @ covariance @ U.transpose(-1, -2)
    a, G = projected_moments(mu, covariance if full else sigma, torch.linalg.inv(U), S)
    L = S @ torch.linalg.inv(U)
    torch.testing.assert_close(a, (L @ mu.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(G, L @ covariance @ L.transpose(-1, -2))
    actual = projected_gaussian_kl(a[:, :2], G[:, :2], a, G)
    assert actual.shape == (1, 2, 2, 3)
    for h in range(2):
        for i in range(2):
            for j in range(3):
                expected = torch.distributions.kl_divergence(
                    torch.distributions.MultivariateNormal(a[0, i, h], G[0, i, h]),
                    torch.distributions.MultivariateNormal(a[0, j, h], G[0, j, h]),
                )
                torch.testing.assert_close(actual[0, h, i, j], expected, rtol=1e-9, atol=1e-10)


def test_rank_bottleneck_retains_independent_content() -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments, relational_mixture_mean

    mu = torch.tensor([[[[1., 1.]], [[1., -1.]]]], dtype=torch.float64)
    U = torch.eye(2, dtype=mu.dtype).expand(1, 2, 1, 2, 2)
    S = torch.tensor([[[1., 0.]]], dtype=mu.dtype)
    a, G = projected_moments(mu, torch.ones_like(mu), U, S)
    reduced = projected_gaussian_kl(a, G, a, G)
    torch.testing.assert_close(reduced, torch.zeros_like(reduced))
    a_full, G_full = projected_moments(mu, torch.ones_like(mu), U, torch.eye(2, dtype=mu.dtype)[None])
    full = projected_gaussian_kl(a_full, G_full, a_full, G_full)
    torch.testing.assert_close(full[0, 0, 0], torch.tensor([0., 2.], dtype=mu.dtype))
    T = torch.tensor([[[0., 1.]]], dtype=mu.dtype)
    O = T.transpose(-1, -2)
    result = relational_mixture_mean(mu, U, reduced.softmax(-1), T, O)
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_full_covariance_passive_frame_changes_preserve_scores_and_covary_content() -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments, relational_mixture_mean

    mu, sigma, U = _fixture()
    Sigma = torch.diag_embed(sigma)
    g = torch.eye(3, dtype=mu.dtype).expand_as(U).clone()
    g[..., 0, 2] = .6
    g[:, 1, :, 1, 0] = -.2
    S = torch.tensor([[[1., .2, -.1], [0., 1., .3]]] * 2, dtype=mu.dtype)
    a, G = projected_moments(mu, Sigma, torch.linalg.inv(U), S)
    changed_mu = (g @ mu.unsqueeze(-1)).squeeze(-1)
    changed_U = g @ U
    changed_Sigma = g @ Sigma @ g.transpose(-1, -2)
    a2, G2 = projected_moments(changed_mu, changed_Sigma, torch.linalg.inv(changed_U), S)
    beta = (-projected_gaussian_kl(a, G, a, G)).softmax(-1)
    beta2 = (-projected_gaussian_kl(a2, G2, a2, G2)).softmax(-1)
    torch.testing.assert_close(beta, beta2)
    eye = torch.eye(3, dtype=mu.dtype).expand(2, 3, 3)
    xi = (torch.linalg.inv(U) @ mu.unsqueeze(-1)).squeeze(-1)
    xi2 = (torch.linalg.inv(changed_U) @ changed_mu.unsqueeze(-1)).squeeze(-1)
    result = relational_mixture_mean(xi, U, beta, eye, eye)
    result2 = relational_mixture_mean(xi2, changed_U, beta2, eye, eye)
    torch.testing.assert_close(result2, (g @ result.unsqueeze(-1)).squeeze(-1))
    direct = torch.einsum("bhij,bihde,bjhe->bihd", beta, U, xi)
    torch.testing.assert_close(result, direct)


def test_full_rank_common_map_is_invariant_and_has_zero_score_derivative() -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    mu, sigma, U = _fixture()
    eye = torch.eye(3, dtype=mu.dtype).expand(2, 3, 3)
    S = (eye + .13 * torch.ones_like(eye)).clone().requires_grad_()
    a, G = projected_moments(mu, sigma, torch.linalg.inv(U), S)
    a0, G0 = projected_moments(mu, sigma, torch.linalg.inv(U), eye)
    D = projected_gaussian_kl(a, G, a, G)
    torch.testing.assert_close(D, projected_gaussian_kl(a0, G0, a0, G0))
    gradient, = torch.autograd.grad(D.square().sum(), S)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-9, rtol=0.)


def test_shear_projection_keeps_correlations_discarded_by_diagonalization() -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    mu, sigma, U = _fixture()
    S = torch.eye(3, dtype=mu.dtype).expand(2, 3, 3)
    a, G = projected_moments(mu, sigma, torch.linalg.inv(U), S)
    exact = projected_gaussian_kl(a, G, a, G)
    truncated = torch.diag_embed(G.diagonal(dim1=-2, dim2=-1))
    approximate = projected_gaussian_kl(a, truncated, a, truncated)
    assert (exact - approximate).abs().max() > .01


def test_reduced_score_and_content_have_correct_live_derivatives() -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments, relational_mixture_mean

    mu, sigma, U = _fixture()
    mu, sigma, U = mu[:, :2, :1, :2], sigma[:, :2, :1, :2], U[:, :2, :1, :2, :2]
    inverse = torch.linalg.inv(U)
    xi = (inverse @ mu.unsqueeze(-1)).squeeze(-1)
    S = torch.tensor([[[1., .2]]], dtype=mu.dtype, requires_grad=True)
    T = torch.tensor([[[.1, .8]]], dtype=mu.dtype, requires_grad=True)
    O = torch.tensor([[[.7], [-.3]]], dtype=mu.dtype, requires_grad=True)

    def reference(S: torch.Tensor, T: torch.Tensor, O: torch.Tensor) -> torch.Tensor:
        a, G = projected_moments(mu, sigma, inverse, S)
        beta = (-projected_gaussian_kl(a, G, a, G)).softmax(-1)
        return relational_mixture_mean(xi, U, beta, T, O)

    assert torch.autograd.gradcheck(reference, (S, T, O), eps=1e-6, atol=1e-5, rtol=1e-4)
    gradients = torch.autograd.grad(reference(S, T, O).square().sum(), (S, T, O))
    assert all(gradient.abs().max() > 1e-6 for gradient in gradients)


def test_operator_initialization_identity_content_and_compact_frames() -> None:
    from vfe3.contracts import CanonicalFrameContext
    from vfe3.model.relational_operators import RelationalOperators

    rng_before = torch.random.get_rng_state().clone()
    identity_content = RelationalOperators([3, 3], seed=7, identity_content=True).double()
    content = RelationalOperators([3, 3], seed=7).double()
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert identity_content.content_factors() is None
    identity = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    # Score rows are orthonormal at construction; they are stored in the parameter dtype,
    # so the residual is float32 rounding, not a projection error.
    S_q, S_k = content.comparison_pair()
    torch.testing.assert_close(S_q @ S_q.transpose(-1, -2), identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(S_k @ S_k.transpose(-1, -2), identity, atol=1e-6, rtol=1e-6)
    T, O = content.content_factors()
    B = O @ T
    torch.testing.assert_close(B, identity, atol=1e-6, rtol=1e-6)
    mu, _, U = _fixture()
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=True)
    flat_mu = mu.flatten(-2)
    xi = content.canonical_mean(flat_mu, frame)
    expected = (U @ (B @ xi.unsqueeze(-1))).squeeze(-1).flatten(-2)
    torch.testing.assert_close(content.apply_content(flat_mu, frame), expected)
    torch.testing.assert_close(identity_content.apply_content(flat_mu, frame), flat_mu)
    dense_U = torch.stack([torch.block_diag(*entry) for entry in U[0]])[None]
    dense_frame = CanonicalFrameContext(dense_U, torch.linalg.inv(dense_U))
    torch.testing.assert_close(content.apply_content(flat_mu, dense_frame), expected)


def test_operator_rejects_rank_loss_and_invalid_heads() -> None:
    from vfe3.model.relational_operators import RelationalOperators

    with pytest.raises(ValueError, match="equal-sized"):
        RelationalOperators([3, 2], seed=1)
    with pytest.raises(ValueError, match="identity_content"):
        RelationalOperators([3], seed=1, identity_content=1)
    degenerate = torch.ones(1, 3, 2)
    degenerate[..., 1] = degenerate[..., 0]
    with pytest.raises(ValueError, match="rank"):
        RelationalOperators._orthonormal_columns(degenerate)


def test_strict_factorization_rejects_singular_covariance_without_jitter() -> None:
    from vfe3.relational_scoring import projected_gaussian_kl

    a = torch.zeros(1, 2, 1, 2)
    G = torch.zeros(1, 2, 1, 2, 2)
    with pytest.raises(ValueError, match="positive definite|Cholesky"):
        projected_gaussian_kl(a, G, a, G)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_autocast_keeps_factorization_precision_and_leading_batch_dims(dtype: torch.dtype) -> None:
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    mu, sigma, U = (value.to(dtype).unsqueeze(0) for value in _fixture())
    S = torch.eye(3, dtype=dtype).expand(2, 3, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a, G = projected_moments(mu, sigma, torch.linalg.inv(U), S)
        D = projected_gaussian_kl(a, G, a, G)
    assert a.dtype == G.dtype == D.dtype == dtype
    assert D.shape == (1, 1, 2, 3, 3)
    assert torch.isfinite(D).all()


@pytest.mark.parametrize('smallest', [1e-4, 1e-5, 1e-6])
def test_ill_conditioned_source_maps_preserve_projected_kl(smallest: float) -> None:
    from vfe3.contracts import CanonicalFrameContext
    from vfe3.gradients.relational import relational_score
    from vfe3.model.relational_operators import RelationalOperators
    generator = torch.Generator().manual_seed(0)
    Q = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    R = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    S = Q @ torch.diag(torch.tensor([1., .5, smallest])) @ R.T
    mu = torch.tensor([[[.2, .3, .7, .2, .3, .7], [.1, -.2, .4, .1, -.2, .4]]])
    inverse = torch.eye(3).expand(1, 2, 2, 3, 3)
    frame = CanonicalFrameContext(inverse, inverse, block_diagonal=True)
    ops = RelationalOperators([3, 3], 0, input_scope='full', readout_mode='predictive_content', output_scope='full')
    with torch.no_grad():
        maps = torch.zeros(2, 3, 6)
        maps[0, :, :3] = S; maps[1, :, 3:] = S
        ops.raw_score_q.copy_(maps); ops.raw_score_k.copy_(maps)
    score = relational_score(mu, torch.ones_like(mu), mu, torch.ones_like(mu), frame, ops)
    expected = torch.tensor([[[[0., .175], [.175, 0.]], [[0., .175], [.175, 0.]]]])
    torch.testing.assert_close(score, expected, rtol=1e-4, atol=1e-5)
    gradients = torch.autograd.grad(score.sum(), (ops.raw_score_q, ops.raw_score_k))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g.norm() > 0 for g in gradients)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_pair_tiles_match_mean_distance_and_gradient_without_full_vector_grid(dtype: torch.dtype) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode
    from vfe3.relational_scoring import projected_gaussian_kl
    mu = torch.randn(1, 35, 1, 2, generator=torch.Generator().manual_seed(5), dtype=dtype, requires_grad=True)
    G = torch.eye(2, dtype=dtype).expand(1, 35, 1, 2, 2)
    vector_grids = []
    class RecordDifferences(TorchDispatchMode):
        def __torch_dispatch__(self, func: object, types: tuple, args: tuple = (), kwargs: dict | None = None) -> object:
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.sub.Tensor and isinstance(result, torch.Tensor) and result.ndim == 5:
                vector_grids.append(result.shape[-3:])
            return result
    with RecordDifferences():
        actual = projected_gaussian_kl(mu, G, mu, G)
        actual_grad, = torch.autograd.grad(actual.square().sum(), mu)
    expected = .5 * (mu[:, :, 0, None, :] - mu[:, None, :, 0, :]).square().sum(-1)[:, None]
    expected_grad, = torch.autograd.grad(expected.square().sum(), mu)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)
    assert vector_grids and (35, 35, 2) not in vector_grids


def test_healthy_projected_kl_uses_only_the_key_precision_inverse() -> None:
    """Healthy factors must not pay an inverse-based conditioning probe."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from vfe3.relational_scoring import projected_gaussian_kl

    class CountInverses(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func: object, types: tuple,
                                 args: tuple = (), kwargs: dict | None = None) -> object:
            if func == torch.ops.aten.cholesky_inverse.default:
                self.count += 1
            return func(*args, **(kwargs or {}))

    mu = torch.randn(1, 4, 1, 2, generator=torch.Generator().manual_seed(11))
    covariance = torch.eye(2).expand(1, 4, 1, 2, 2)
    counter = CountInverses()
    with counter:
        score = projected_gaussian_kl(mu, covariance, mu, covariance)
    assert torch.isfinite(score).all()
    assert counter.count == 1


def test_shear_source_recovery_matches_unprojected_kl() -> None:
    """Equal Cholesky pivots do not certify a well-conditioned covariance."""
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    mu = torch.tensor([[[[0., 0.]], [[.001, 0.]]]])
    sigma = torch.ones_like(mu)
    inverse = torch.eye(2).expand(1, 2, 1, 2, 2)
    S = torch.tensor([[[1., 0.], [3000., 1.]]], requires_grad=True)
    a, G = projected_moments(mu, sigma, inverse, S)
    source = (sigma, inverse, S, mu)
    actual = projected_gaussian_kl(a, G, a, G, projection_q=source, projection_k=source)
    expected = torch.tensor([[[[0., 5e-7], [5e-7, 0.]]]])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-5)
    gradient, = torch.autograd.grad(actual.sum(), S)
    assert torch.isfinite(gradient).all()


def test_healthy_projected_kl_batches_host_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.relational_scoring import projected_gaussian_kl

    calls = []
    original_list, original_bool = torch.Tensor.tolist, torch.Tensor.__bool__

    def tolist(value: torch.Tensor) -> list:
        calls.append("tolist")
        return original_list(value)

    def tobool(value: torch.Tensor) -> bool:
        calls.append("bool")
        return original_bool(value)

    mu = torch.zeros(1, 3, 1, 2)
    covariance = torch.eye(2).expand(1, 3, 1, 2, 2)
    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "tolist", tolist)
        scoped.setattr(torch.Tensor, "__bool__", tobool)
        projected_gaussian_kl(mu, covariance, mu, covariance)
    assert calls == ["tolist"]
