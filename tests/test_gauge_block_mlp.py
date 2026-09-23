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
