r"""Block-MLP pieces of the M2 main-results arms (2026-09-28).

The per-type hidden-bias table enters the block MLP as ``forward_moments(..., hidden_bias=E)``:
``E`` (B, N, w) is added to the first ``w <= H`` units of the fc1 pre-activation,

.. math::
    \mu \mapsto \mu + W_2\,\mathrm{act}(W_1 \mu + b_1 + [E, 0]) + b_2 .

``hidden_bias=None`` must keep every existing path bitwise unchanged. Modes without a coordinate fc1
(gauge_gate, whose fc1 reads the block invariants, and gaussian_template) reject the keyword. The
gauge-gate FLOP metadata is layout-aware (full Cholesky solve versus the diagonal quotient).
"""
import pytest
import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.model.block_mlp import (
    BlockMLP,
    CanonicalFrameBlockMLP,
    GaugeGateBlockMLP,
    GaussianTemplateBlockMLP,
    block_mlp_build_metadata,
    get_block_mlp_registration,
)


torch.set_num_threads(1)

EMBED_DIM, EXPANSION = 4, 2
HIDDEN = EMBED_DIM * EXPANSION


def _coordinate(covariance_contract: str = "passthrough", *, dtype=torch.float64, seed: int = 91):
    torch.manual_seed(seed)
    return BlockMLP(
        embed_dim=EMBED_DIM, expansion=EXPANSION, activation="gelu", dropout=0.0,
        covariance_contract=covariance_contract,
    ).to(dtype)


def _canonical(seed: int = 92) -> CanonicalFrameBlockMLP:
    torch.manual_seed(seed)
    return CanonicalFrameBlockMLP(
        embed_dim=EMBED_DIM, expansion=EXPANSION, activation="silu", dropout=0.0,
        covariance_contract="passthrough",
    ).double()


def _state(batch_shape=(2, 3), dtype=torch.float64):
    mu = torch.randn(*batch_shape, EMBED_DIM, dtype=dtype)
    sigma = torch.rand(*batch_shape, EMBED_DIM, dtype=dtype) + 0.5
    return mu, sigma


def _frame(batch_shape=(2, 3)) -> CanonicalFrameContext:
    raw = torch.randn(*batch_shape, EMBED_DIM, EMBED_DIM, dtype=torch.float64)
    forward = raw + 2.0 * torch.eye(EMBED_DIM, dtype=torch.float64)
    return CanonicalFrameContext(forward, torch.linalg.inv(forward))


def _matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _biased_reference(mlp: BlockMLP, mu: torch.Tensor, hidden_bias: torch.Tensor) -> torch.Tensor:
    r"""mu + fc2(act(fc1(mu) + [E, 0])) written independently of the module's helper."""
    hidden = mlp.fc1(mu).clone()                                  # (..., H)
    hidden[..., :hidden_bias.shape[-1]] += hidden_bias
    return mu + mlp.fc2(mlp.activation(hidden))


# ---------- hidden_bias=None is the unchanged path ----------


def test_coordinate_path_without_hidden_bias_is_bitwise_the_original_expression():
    mlp = _coordinate(dtype=torch.float32)
    mu, sigma = _state(dtype=torch.float32)

    original = mu + mlp.dropout(mlp.fc2(mlp.activation(mlp.fc1(mu))))
    result = mlp.forward_moments(mu, sigma)

    assert torch.equal(result.mu, original)
    assert torch.equal(mlp(mu), original)
    assert result.sigma is sigma and result.jacobian is None


def test_canonical_frame_path_without_hidden_bias_is_bitwise_the_original_expression():
    mlp = _canonical()
    mu, sigma = _state()
    frame = _frame()

    xi = torch.einsum("...ij,...j->...i", frame.inverse, mu)
    xi_out = xi + mlp.fc2(mlp.activation(mlp.fc1(xi)))
    original = torch.einsum("...ij,...j->...i", frame.forward, xi_out)

    assert torch.equal(mlp.forward_moments(mu, sigma, frame_context=frame).mu, original)


@pytest.mark.parametrize("width", [0, 3, HIDDEN])
def test_zero_hidden_bias_leaves_the_output_bit_identical(width):
    mlp = _coordinate(dtype=torch.float32)
    mu, sigma = _state(dtype=torch.float32)
    zero = torch.zeros(*mu.shape[:-1], width)

    assert torch.equal(
        mlp.forward_moments(mu, sigma, hidden_bias=zero).mu, mlp.forward_moments(mu, sigma).mu)


# ---------- the bias enters the first w hidden units ----------


@pytest.mark.parametrize("width", [0, 1, 3, HIDDEN])
def test_hidden_bias_is_added_to_the_first_w_fc1_units(width):
    mlp = _coordinate()
    mu, sigma = _state()
    hidden_bias = torch.randn(*mu.shape[:-1], width, dtype=torch.float64)

    result = mlp.forward_moments(mu, sigma, hidden_bias=hidden_bias)

    torch.testing.assert_close(result.mu, _biased_reference(mlp, mu, hidden_bias))
    assert result.sigma is sigma
    if width:
        assert not torch.allclose(result.mu, mlp.forward_moments(mu, sigma).mu)


def test_hidden_bias_delta_full_jacobian_matches_autograd():
    mlp = _coordinate("delta_full")
    mu = torch.randn(EMBED_DIM, dtype=torch.float64)
    sigma = torch.diag(torch.rand(EMBED_DIM, dtype=torch.float64) + 0.5)
    hidden_bias = torch.randn(5, dtype=torch.float64)

    result = mlp.forward_moments(mu, sigma, hidden_bias=hidden_bias)
    reference_jacobian = torch.autograd.functional.jacobian(
        lambda value: mlp(value, hidden_bias=hidden_bias), mu)

    torch.testing.assert_close(result.mu, _biased_reference(mlp, mu, hidden_bias))
    torch.testing.assert_close(result.jacobian, reference_jacobian, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        result.sigma,
        reference_jacobian @ sigma @ reference_jacobian.T + mlp.covariance_floor * sigma,
        rtol=1e-10, atol=1e-10)


def test_canonical_frame_hidden_bias_acts_in_canonical_coordinates():
    r"""mu -> U (xi + fc2(act(fc1(xi) + [E, 0]))), xi = U^{-1} mu."""
    mlp = _canonical()
    mu, sigma = _state()
    frame = _frame()
    hidden_bias = torch.randn(2, 3, 5, dtype=torch.float64)

    result = mlp.forward_moments(mu, sigma, frame_context=frame, hidden_bias=hidden_bias)
    xi = _matvec(frame.inverse, mu)

    torch.testing.assert_close(
        result.mu, _matvec(frame.forward, _biased_reference(mlp, xi, hidden_bias)))
    torch.testing.assert_close(
        mlp.mean_update(mu, frame, hidden_bias=hidden_bias), result.mu)


def test_hidden_bias_gradients_match_finite_differences():
    mlp = _coordinate()
    mu = torch.randn(2, 3, EMBED_DIM, dtype=torch.float64, requires_grad=True)
    sigma = torch.rand(2, 3, EMBED_DIM, dtype=torch.float64) + 0.5
    hidden_bias = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda mu_in, bias_in: mlp.forward_moments(mu_in, sigma, hidden_bias=bias_in).mu,
        (mu, hidden_bias))


def test_hidden_bias_keeps_the_autocast_dtype_of_the_unbiased_path():
    r"""Under bf16 autocast the float32 table is cast like fc1's own bias; its gradient stays float32."""
    mlp = _coordinate(dtype=torch.float32)
    mu, sigma = _state(dtype=torch.float32)
    table = torch.randn(*mu.shape[:-1], 5, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        plain = mlp.forward_moments(mu, sigma).mu
        biased = mlp.forward_moments(mu, sigma, hidden_bias=table).mu
    biased.float().sum().backward()

    assert biased.dtype == plain.dtype
    assert table.grad is not None and table.grad.dtype == torch.float32
    assert torch.isfinite(table.grad).all() and table.grad.abs().sum() > 0


# ---------- validation ----------


def test_hidden_bias_wider_than_the_hidden_layer_is_rejected():
    mlp = _coordinate()
    mu, sigma = _state()
    with pytest.raises(ValueError, match="exceeds the fc1 hidden width"):
        mlp.forward_moments(mu, sigma, hidden_bias=torch.zeros(2, 3, HIDDEN + 1, dtype=torch.float64))


@pytest.mark.parametrize("shape", [(3, 5), (1, 3, 5), (2, 4, 5), (2, 3, 1, 5)])
def test_hidden_bias_with_other_leading_axes_is_rejected(shape):
    mlp = _coordinate()
    mu, sigma = _state()
    with pytest.raises(ValueError, match="leading axes"):
        mlp.forward_moments(mu, sigma, hidden_bias=torch.zeros(*shape, dtype=torch.float64))


def test_modes_without_a_coordinate_hidden_layer_reject_hidden_bias():
    mu, sigma = _state()
    hidden_bias = torch.zeros(2, 3, 1, dtype=torch.float64)
    gate = GaugeGateBlockMLP(
        irrep_dims=(2, 2), expansion=2, activation="gelu", dropout=0.0).double()
    template = GaussianTemplateBlockMLP(embed_dim=EMBED_DIM, expansion=EXPANSION, dropout=0.0).double()

    with pytest.raises(ValueError, match="hidden_bias does not apply.*invariants"):
        gate.forward_moments(mu, sigma, hidden_bias=hidden_bias)
    with pytest.raises(ValueError, match="hidden_bias does not apply.*template"):
        template.forward_moments(mu, sigma, hidden_bias=hidden_bias)
    # hidden_bias=None is accepted by every mode.
    assert torch.equal(gate.forward_moments(mu, sigma, hidden_bias=None).mu, mu)
    template.forward_moments(mu, sigma, hidden_bias=None)


# ---------- layout-aware FLOP metadata ----------


def _metadata(mode: str, irrep_dims, **layout):
    return block_mlp_build_metadata(
        enabled=True, mode=mode, covariance="passthrough", expansion=3, activation="gelu",
        dropout=0.0, covariance_floor=1e-4, embed_dim=sum(irrep_dims), irrep_dims=irrep_dims,
        n_layers=2, **layout)


def test_gauge_gate_flops_follow_the_covariance_layout():
    irrep_dims = (3, 1, 2)
    n_blocks, K, expansion, n_layers = len(irrep_dims), sum(irrep_dims), 3, 2
    gate_mlp = 4.0 * expansion * n_blocks ** 2
    cholesky = sum(2.0 * d ** 3 + 2.0 * d ** 2 for d in irrep_dims)

    full = _metadata("gauge_gate", irrep_dims, covariance_layout="full")
    diagonal = _metadata("gauge_gate", irrep_dims, covariance_layout="diagonal")

    assert _metadata("gauge_gate", irrep_dims).flops_per_token == full.flops_per_token
    assert full.flops_per_token == n_layers * (gate_mlp + cholesky)
    assert diagonal.flops_per_token == n_layers * (gate_mlp + 3.0 * K)
    assert diagonal.parameter_count == full.parameter_count
    assert diagonal.report_label == full.report_label == "invariant_scalar_gate"


@pytest.mark.parametrize("mode", ["coordinate", "canonical_frame", "gaussian_template"])
def test_layout_does_not_change_other_modes_flops(mode):
    irrep_dims = (2, 2)
    assert get_block_mlp_registration(mode).diagonal_extra_flops_per_layer is None
    assert (_metadata(mode, irrep_dims, covariance_layout="diagonal").flops_per_token
            == _metadata(mode, irrep_dims).flops_per_token)


def test_unknown_covariance_layout_is_rejected():
    with pytest.raises(ValueError, match="covariance_layout"):
        _metadata("gauge_gate", (2, 2), covariance_layout="banded")
