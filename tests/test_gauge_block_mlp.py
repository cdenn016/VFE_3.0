from tests.profiles import test_config
import torch
import pytest

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.block_mlp import (
    BlockMLP,
    CanonicalFrameBlockMLP,
    GaugeGateBlockMLP,
)


def _spd(batch_shape, dim):
    raw = torch.randn(*batch_shape, dim, dim, dtype=torch.float64)
    eye = torch.eye(dim, dtype=torch.float64)
    return raw @ raw.transpose(-1, -2) + 0.7 * eye


def _block_gauge(batch_shape, block_dim):
    blocks = []
    for _ in range(2):
        raw = torch.randn(*batch_shape, block_dim, block_dim, dtype=torch.float64)
        blocks.append(raw + 1.5 * torch.eye(block_dim, dtype=torch.float64))
    out = torch.zeros(*batch_shape, 2 * block_dim, 2 * block_dim, dtype=torch.float64)
    out[..., :block_dim, :block_dim] = blocks[0]
    out[..., block_dim:, block_dim:] = blocks[1]
    return out


def _congruence(g, sigma):
    return g @ sigma @ g.transpose(-1, -2)

def test_coordinate_delta_full_jacobian_and_covariance_match_autograd_reference():
    torch.manual_seed(8)
    mlp = BlockMLP(
        embed_dim=4, expansion=2, activation="gelu", dropout=0.0,
        covariance_contract="delta_full", covariance_floor=1e-4,
    ).double()
    mu = torch.randn(4, dtype=torch.float64, requires_grad=True)
    sigma = _spd((), 4)

    result = mlp.forward_moments(mu, sigma)
    reference_jacobian = torch.autograd.functional.jacobian(
        mlp, mu, create_graph=True,
    )
    reference_sigma = (
        reference_jacobian @ sigma @ reference_jacobian.transpose(-1, -2)
        + 1e-4 * sigma
    )

    assert torch.allclose(result.jacobian, reference_jacobian, atol=2e-10, rtol=2e-10)
    assert torch.allclose(result.sigma, reference_sigma, atol=2e-10, rtol=2e-10)
    (result.mu.square().sum() + result.sigma.square().sum()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in mlp.parameters())


def test_covariant_floor_keeps_rank_zero_gate_covariance_positive_definite():
    torch.manual_seed(9)
    first = GaugeGateBlockMLP(
        irrep_dims=[2, 2], expansion=1, activation="silu", dropout=0.0,
        covariance_contract="delta_full", covariance_floor=1e-4,
    ).double()
    second = GaugeGateBlockMLP(
        irrep_dims=[2, 2], expansion=1, activation="silu", dropout=0.0,
        covariance_contract="delta_full", covariance_floor=1e-4,
    ).double()
    with torch.no_grad():
        first.fc2.weight.zero_()
        first.fc2.bias.fill_(-1.0)  # realized mean map is zero, so J is exactly rank zero

    mu = torch.randn(3, 4, dtype=torch.float64)
    sigma = _spd((3,), 4)
    collapsed = first.forward_moments(mu, sigma)
    continued = second.forward_moments(collapsed.mu, collapsed.sigma)

    assert torch.all(torch.linalg.eigvalsh(collapsed.sigma) > 0)
    assert torch.allclose(collapsed.sigma, 1e-4 * sigma, atol=2e-12, rtol=2e-12)
    assert torch.isfinite(continued.mu).all()
    assert torch.isfinite(continued.sigma).all()



def test_gauge_gate_passthrough_commutes_with_block_glk_action():
    torch.manual_seed(12)
    mlp = GaugeGateBlockMLP(
        irrep_dims=[2, 2], expansion=2, activation="silu", dropout=0.0,
        covariance_contract="passthrough",
    ).double()
    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.tensor([
            [0.2, -0.1], [0.3, 0.4], [-0.5, 0.2], [0.1, 0.6]], dtype=torch.float64))
        mlp.fc1.bias.copy_(torch.tensor([0.1, -0.2, 0.05, 0.3], dtype=torch.float64))
        mlp.fc2.weight.copy_(torch.tensor([
            [0.4, -0.2, 0.3, 0.1], [-0.1, 0.5, 0.2, -0.3]], dtype=torch.float64))
        mlp.fc2.bias.copy_(torch.tensor([0.05, -0.07], dtype=torch.float64))

    mu = torch.randn(2, 3, 4, dtype=torch.float64)
    sigma = _spd((2, 3), 4)
    gauge = _block_gauge((2, 3), 2)

    base = mlp.forward_moments(mu, sigma)
    transformed = mlp.forward_moments(
        (gauge @ mu.unsqueeze(-1)).squeeze(-1),
        _congruence(gauge, sigma),
    )

    expected_mu = (gauge @ base.mu.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(transformed.mu, expected_mu, atol=2e-10, rtol=2e-10)
    assert torch.equal(base.sigma, sigma)
    assert torch.equal(transformed.sigma, _congruence(gauge, sigma))
def test_gauge_gate_dropout_is_equivariant_conditioned_on_the_same_scalar_mask():
    torch.manual_seed(13)
    mlp = GaugeGateBlockMLP(
        irrep_dims=[2, 2], expansion=2, activation="silu", dropout=0.5,
        covariance_contract="passthrough",
    ).double().train()
    with torch.no_grad():
        mlp.fc2.weight.fill_(0.2)
        mlp.fc2.bias.copy_(torch.tensor([0.1, -0.1], dtype=torch.float64))

    mu = torch.randn(2, 4, dtype=torch.float64)
    sigma = _spd((2,), 4)
    gauge = _block_gauge((2,), 2)
    rng_state = torch.random.get_rng_state()
    base = mlp.forward_moments(mu, sigma)
    torch.random.set_rng_state(rng_state)
    transformed = mlp.forward_moments(
        (gauge @ mu.unsqueeze(-1)).squeeze(-1), _congruence(gauge, sigma))

    assert torch.allclose(
        transformed.mu, (gauge @ base.mu.unsqueeze(-1)).squeeze(-1),
        atol=2e-10, rtol=2e-10,
    )


def test_gauge_gate_rejects_irrep_partition_that_does_not_cover_the_mean():
    mlp = GaugeGateBlockMLP(
        irrep_dims=[2, 1], expansion=1, activation="relu", dropout=0.0,
        covariance_contract="passthrough",
    ).double()
    mu = torch.randn(4, dtype=torch.float64)
    sigma = _spd((), 4)

    with pytest.raises(ValueError, match="irrep_dims.*mean dimension"):
        mlp.forward_moments(mu, sigma)




def test_gauge_gate_delta_full_jacobian_and_covariance_match_autograd_reference():
    torch.manual_seed(21)
    mlp = GaugeGateBlockMLP(
        irrep_dims=[2, 2], expansion=1, activation="gelu", dropout=0.0,
        covariance_contract="delta_full",
    ).double()
    with torch.no_grad():
        mlp.fc2.weight.fill_(0.17)
        mlp.fc2.bias.copy_(torch.tensor([0.03, -0.02], dtype=torch.float64))

    mu = torch.randn(4, dtype=torch.float64, requires_grad=True)
    sigma = _spd((), 4)
    result = mlp.forward_moments(mu, sigma)
    reference_jacobian = torch.autograd.functional.jacobian(
        lambda value: mlp.mean_update(value, sigma), mu, create_graph=True,
    )
    reference_sigma = (
        reference_jacobian @ sigma @ reference_jacobian.transpose(-1, -2)
        + mlp.covariance_floor * sigma)

    assert torch.allclose(result.jacobian, reference_jacobian, atol=2e-10, rtol=2e-10)
    assert torch.allclose(result.sigma, reference_sigma, atol=2e-10, rtol=2e-10)
    (result.mu.square().sum() + result.sigma.square().sum()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in mlp.parameters())


def test_gauge_gate_delta_full_covariance_commutes_with_block_glk_action():
    torch.manual_seed(31)
    mlp = GaugeGateBlockMLP(
        irrep_dims=[2, 2], expansion=2, activation="relu", dropout=0.0,
        covariance_contract="delta_full",
    ).double()
    with torch.no_grad():
        mlp.fc1.bias.fill_(0.5)
        mlp.fc2.weight.fill_(0.09)

    mu = torch.randn(2, 4, dtype=torch.float64)
    sigma = _spd((2,), 4)
    gauge = _block_gauge((2,), 2)
    base = mlp.forward_moments(mu, sigma)
    transformed = mlp.forward_moments(
        (gauge @ mu.unsqueeze(-1)).squeeze(-1), _congruence(gauge, sigma))

    expected_mu = (gauge @ base.mu.unsqueeze(-1)).squeeze(-1)
    expected_sigma = _congruence(gauge, base.sigma)
    assert torch.allclose(transformed.mu, expected_mu, atol=2e-9, rtol=2e-9)
    assert torch.allclose(transformed.sigma, expected_sigma, atol=2e-8, rtol=2e-8)


def test_canonical_frame_passthrough_is_left_equivariant():
    torch.manual_seed(41)
    mlp = CanonicalFrameBlockMLP(
        embed_dim=4, expansion=2, activation="silu", dropout=0.0,
        covariance_contract="passthrough",
    ).double()
    mu = torch.randn(2, 3, 4, dtype=torch.float64)
    sigma = _spd((2, 3), 4)
    frame = _block_gauge((2, 3), 2)
    frame_inv = torch.linalg.inv(frame)
    gauge = _block_gauge((2, 3), 2)
    gauge_inv = torch.linalg.inv(gauge)

    base = mlp.forward_moments(
        mu, sigma, frame_context=CanonicalFrameContext(frame, frame_inv))
    transformed = mlp.forward_moments(
        (gauge @ mu.unsqueeze(-1)).squeeze(-1),
        _congruence(gauge, sigma),
        frame_context=CanonicalFrameContext(gauge @ frame, frame_inv @ gauge_inv),
    )

    assert torch.allclose(
        transformed.mu, (gauge @ base.mu.unsqueeze(-1)).squeeze(-1),
        atol=2e-10, rtol=2e-10,
    )
    assert torch.equal(base.sigma, sigma)
    assert torch.equal(transformed.sigma, _congruence(gauge, sigma))


def test_canonical_frame_delta_full_matches_realized_coordinate_jacobian():
    torch.manual_seed(51)
    mlp = CanonicalFrameBlockMLP(
        embed_dim=4, expansion=1, activation="gelu", dropout=0.0,
        covariance_contract="delta_full",
    ).double()
    mu = torch.randn(4, dtype=torch.float64, requires_grad=True)
    sigma = _spd((), 4)
    frame = _block_gauge((), 2)
    frame_context = CanonicalFrameContext(frame, torch.linalg.inv(frame))
    result = mlp.forward_moments(mu, sigma, frame_context=frame_context)
    reference_jacobian = torch.autograd.functional.jacobian(
        lambda value: mlp.mean_update(value, frame_context), mu, create_graph=True,
    )
    reference_sigma = (
        reference_jacobian @ sigma @ reference_jacobian.transpose(-1, -2)
        + mlp.covariance_floor * sigma)

    assert torch.allclose(result.jacobian, reference_jacobian, atol=2e-10, rtol=2e-10)
    assert torch.allclose(result.sigma, reference_sigma, atol=2e-10, rtol=2e-10)

def test_canonical_frame_is_not_invariant_to_a_generic_right_frame_change():
    torch.manual_seed(42)
    mlp = CanonicalFrameBlockMLP(
        embed_dim=4, expansion=2, activation="silu", dropout=0.0,
        covariance_contract="passthrough",
    ).double()
    mu = torch.randn(4, dtype=torch.float64)
    sigma = _spd((), 4)
    frame = _block_gauge((), 2)
    right = _block_gauge((), 2)
    base_context = CanonicalFrameContext(frame, torch.linalg.inv(frame))
    changed_context = CanonicalFrameContext(
        frame @ right, torch.linalg.inv(right) @ torch.linalg.inv(frame))

    base = mlp.forward_moments(mu, sigma, frame_context=base_context)
    changed = mlp.forward_moments(mu, sigma, frame_context=changed_context)

    assert not torch.allclose(changed.mu, base.mu, atol=1e-9, rtol=1e-9)


# Diagonal covariance layout of the gauge gate (M2 intertwiner arm, 2026-09-28). The gate reads
# sigma (..., K) = diag Sigma and s_b = sum_{k in b} mu_k^2 / sigma_k.


def _diagonal_gate(irrep_dims, *, seed, covariance_contract="passthrough"):
    r"""A float64 gate with a nonzero, nonlinear response: fc1 columns scaled by 1/d_b since s_b ~ d_b."""
    torch.manual_seed(seed)
    mlp = GaugeGateBlockMLP(
        irrep_dims=irrep_dims, expansion=2, activation="gelu", dropout=0.0,
        covariance_contract=covariance_contract,
    ).double()
    with torch.no_grad():
        mlp.fc1.weight.normal_(0.0, 1.0).div_(torch.tensor(irrep_dims, dtype=torch.float64))
        mlp.fc2.weight.normal_(0.0, 0.3)
        mlp.fc2.bias.normal_(0.0, 0.1)
    return mlp


def _diagonal_state(batch_shape, dim):
    mu = torch.randn(*batch_shape, dim, dtype=torch.float64)
    sigma = torch.empty(*batch_shape, dim, dtype=torch.float64).uniform_(0.5, 2.0)
    return mu, sigma


def _orthogonal_times_scale(batch_shape, dim):
    r"""Q diag(exp u) with Q orthogonal and u ~ U(-0.7, 0.7), so cond <= e^{1.4}."""
    q, _ = torch.linalg.qr(torch.randn(*batch_shape, dim, dim, dtype=torch.float64))
    scale = torch.empty(*batch_shape, dim, dtype=torch.float64).uniform_(-0.7, 0.7).exp()
    return q * scale.unsqueeze(-2)


def _block_frame(batch_shape, irrep_dims):
    r"""g = (+)_b g_b in prod_b GL(d_b), each g_b = Q_b diag(exp u_b)."""
    dim = sum(irrep_dims)
    g = torch.zeros(*batch_shape, dim, dim, dtype=torch.float64)
    start = 0
    for block_dim in irrep_dims:
        stop = start + block_dim
        g[..., start:stop, start:stop] = _orthogonal_times_scale(batch_shape, block_dim)
        start = stop
    return g


def _matvec(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _block_sums(values, irrep_dims):
    return torch.stack([chunk.sum(dim=-1) for chunk in values.split(list(irrep_dims), dim=-1)], dim=-1)


@pytest.mark.parametrize("irrep_dims", [(2, 2), (3, 1, 2)])
def test_diagonal_gate_equals_the_full_gate_on_diag_embed(irrep_dims):
    mlp = _diagonal_gate(irrep_dims, seed=61)
    mu, sigma = _diagonal_state((2, 3), sum(irrep_dims))

    s_diagonal = mlp._invariants(mu, sigma)[0]
    s_full = mlp._invariants(mu, torch.diag_embed(sigma))[0]
    diagonal = mlp.forward_moments(mu, sigma)
    full = mlp.forward_moments(mu, torch.diag_embed(sigma))

    torch.testing.assert_close(
        s_diagonal, _block_sums(mu.square() / sigma, irrep_dims), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(s_diagonal, s_full, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(diagonal.mu, full.mu, rtol=1e-13, atol=1e-13)
    assert diagonal.sigma is sigma and diagonal.jacobian is None
    assert not torch.allclose(diagonal.mu, mu)          # the gate is not trivially the identity here


@pytest.mark.parametrize("irrep_dims", [(2, 2), (3, 1, 2), (32, 32)])
def test_diagonal_gate_is_covariant_under_the_block_frame_group(irrep_dims):
    r"""gate_full(g mu, g diag(sigma) g^T) = g gate_diag(mu, sigma) for g in prod_b GL(d_b)."""
    mlp = _diagonal_gate(irrep_dims, seed=62)
    mu, sigma = _diagonal_state((3,), sum(irrep_dims))
    g = _block_frame((3,), irrep_dims)

    base = mlp.forward_moments(mu, sigma)
    moved = mlp.forward_moments(_matvec(g, mu), _congruence(g, torch.diag_embed(sigma)))

    torch.testing.assert_close(moved.mu, _matvec(g, base.mu), rtol=1e-10, atol=1e-10)


def test_diagonal_gate_is_not_covariant_under_a_dense_frame():
    r"""Negative control: a dense GL(K) change mixes the blocks, so s_b is not invariant."""
    irrep_dims = (2, 2)
    mlp = _diagonal_gate(irrep_dims, seed=62)
    mu, sigma = _diagonal_state((3,), sum(irrep_dims))
    g = _orthogonal_times_scale((3,), sum(irrep_dims))

    base = mlp.forward_moments(mu, sigma)
    moved = mlp.forward_moments(_matvec(g, mu), _congruence(g, torch.diag_embed(sigma)))

    assert (moved.mu - _matvec(g, base.mu)).abs().max() > 1e-3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fresh_diagonal_gate_is_exactly_the_identity(dtype):
    torch.manual_seed(63)
    mlp = GaugeGateBlockMLP(
        irrep_dims=(2, 2), expansion=2, activation="gelu", dropout=0.0,
        covariance_contract="passthrough",
    ).to(dtype)
    mu = torch.randn(2, 3, 4, dtype=dtype)
    sigma = torch.rand(2, 3, 4, dtype=dtype) + 0.5

    out = mlp.forward_moments(mu, sigma)

    assert torch.equal(out.mu, mu)
    assert out.sigma is sigma


def test_reference_coordinate_invariant_equals_the_local_diagonal_invariant():
    r"""xi_b^T (C_bb)^{-1} xi_b = mu_b^T Sigma_bb^{-1} mu_b for xi = U^{-1} mu, C = U^{-1} Sigma U^{-T}."""
    irrep_dims = (3, 1, 2)
    mlp = _diagonal_gate(irrep_dims, seed=64)
    mu, sigma = _diagonal_state((2, 3), sum(irrep_dims))
    frame = _block_frame((2, 3), irrep_dims)
    frame_inverse = torch.linalg.inv(frame)
    xi = _matvec(frame_inverse, mu)
    C = _congruence(frame_inverse, torch.diag_embed(sigma))

    torch.testing.assert_close(
        mlp._invariants(xi, C)[0], mlp._invariants(mu, sigma)[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        _matvec(frame, mlp.forward_moments(xi, C).mu),
        mlp.forward_moments(mu, sigma).mu, rtol=1e-10, atol=1e-10)


def test_diagonal_layout_rejects_delta_full_in_the_module_and_the_config():
    mlp = _diagonal_gate((2, 2), seed=65, covariance_contract="delta_full")
    mu, sigma = _diagonal_state((3,), 4)

    with pytest.raises(ValueError, match="delta_full"):
        mlp.forward_moments(mu, sigma)
    assert mlp.forward_moments(mu, torch.diag_embed(sigma)).jacobian is not None


    with pytest.raises(ValueError, match="delta_full"):
        test_config(
            vocab_size=17, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1, n_e_steps=1,
            e_phi_lr=0.0, family="gaussian_diagonal", decode_mode="linear",
            use_block_mlp=True, block_mlp_mode="gauge_gate", block_mlp_covariance="delta_full",
            norm_type_block="none", layernorm_affine=False, skip_belief_sigma_update=False,
        )


def test_diagonal_invariants_are_evaluated_in_at_least_float32():
    mlp = GaugeGateBlockMLP(
        irrep_dims=(2, 2), expansion=1, activation="gelu", dropout=0.0,
        covariance_contract="passthrough",
    )
    torch.manual_seed(66)
    mu = torch.randn(5, 4).bfloat16()
    sigma = (torch.rand(5, 4) + 0.5).bfloat16()

    s = mlp._invariants(mu, sigma)[0]

    mu32, sigma32 = mu.float(), sigma.float()
    assert s.dtype == torch.float32
    torch.testing.assert_close(s, _block_sums(mu32 * (mu32 / sigma32), (2, 2)))


def test_diagonal_gate_gradients_match_finite_differences_including_sigma():
    r"""gradcheck in (mu, sigma); the sigma channel ds_b/dsigma_k = -mu_k^2 / sigma_k^2 is new."""
    irrep_dims = (2, 1)
    mlp = _diagonal_gate(irrep_dims, seed=67)
    mu, sigma = _diagonal_state((2,), sum(irrep_dims))
    mu.requires_grad_(True)
    sigma.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda mu_in, sigma_in: mlp.forward_moments(mu_in, sigma_in).mu, (mu, sigma))
    mlp.forward_moments(mu, sigma).mu.square().sum().backward()
    assert sigma.grad.abs().sum() > 0
    for name, parameter in mlp.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
