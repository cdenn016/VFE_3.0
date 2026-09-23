"""Causal support must exclude Gaussian pairs before numerical certification."""

import os
from collections.abc import Iterator

import pytest
import torch

from vfe3.families.gaussian import FullGaussian
from vfe3.free_energy import free_energy, self_divergence_for_alpha
from vfe3.geometry import transport as T
from vfe3.gradients.oracle import belief_gradients_autograd


@pytest.fixture(autouse=True)
def _precision_policy() -> Iterator[None]:
    precision = T.set_full_cov_congruence_precision("fp32_escalate")
    certification = T.set_full_cov_congruence_certification(True)
    yield
    T.set_full_cov_congruence_precision(precision)
    T.set_full_cov_congruence_certification(certification)


@pytest.fixture
def device() -> torch.device:
    return torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _future_pair_fixture(device: torch.device) -> tuple:
    """Only the future pair has an unrepresentable covariance eigenvalue in float64."""
    rotation = torch.tensor([[1., -1.], [1., 1.]], device=device) / 2**0.5
    exp = torch.diag(torch.tensor([1e6, 1.], device=device))
    inv = torch.diag(torch.tensor([1e-6, 1.], device=device))
    eye = torch.eye(2, device=device)
    factors = T.CompactFactoredTransport(
        torch.stack((torch.stack((rotation, eye)), torch.stack((exp, eye)))).unsqueeze(0).requires_grad_(),
        torch.stack((torch.stack((rotation.T, eye)), torch.stack((inv, eye)))).unsqueeze(0).requires_grad_(),
        4, same_frame_flat_cocycle=True,
    )
    sigma = torch.diag_embed(torch.tensor(
        [[[1., 1., 1., 1.], [4e-6, 1., 1., 1.]]], device=device)).requires_grad_()
    mu = torch.zeros(1, 2, 4, device=device, requires_grad=True)
    prior = torch.tensor([[0., float("-inf")], [0., float("-inf")]], device=device).expand(2, 2, 2)
    return mu, sigma, factors, prior


def test_oracle_excludes_future_pair_before_spd_certificate(device: torch.device) -> None:
    mu, sigma, factors, prior = _future_pair_fixture(device)
    gradients = belief_gradients_autograd(
        mu, sigma, mu.detach(), sigma.detach(), factors, family="gaussian_full",
        irrep_dims=[2, 2], log_prior=prior, create_graph=True,
        full_cov_kl_precision="fp32_escalate",
    )
    assert all(torch.isfinite(value).all() for value in gradients)
    outer = torch.autograd.grad(
        sum(value.square().sum() for value in gradients),
        (mu, sigma, factors.exp_blocks, factors.inv_blocks),
    )
    assert all(torch.isfinite(value).all() for value in outer)


def test_included_bad_pair_still_fails_certificate(device: torch.device) -> None:
    mu, sigma, factors, prior = _future_pair_fixture(device)
    with pytest.raises(FloatingPointError, match="conditioning bounds"):
        belief_gradients_autograd(
            mu, sigma, mu.detach(), sigma.detach(), factors, family="gaussian_full",
            irrep_dims=[2, 2], log_prior=torch.zeros_like(prior),
            full_cov_kl_precision="fp32_escalate",
        )


def _benign_fixture(device: torch.device) -> tuple:
    generator = torch.Generator().manual_seed(41)
    coordinates = (0.1 * torch.randn(1, 3, 2, 2, 2, generator=generator)).to(device)
    factors = T.CompactFactoredTransport(
        torch.matrix_exp(coordinates).requires_grad_(),
        torch.matrix_exp(-coordinates).requires_grad_(), 4,
    )
    mu = torch.randn(1, 3, 4, generator=generator).to(device).requires_grad_()
    sigma = torch.diag_embed(torch.linspace(.7, 1.2, 12, device=device).reshape(1, 3, 4)).requires_grad_()
    support = torch.tensor([
        [[True, False, False], [True, False, False], [False, True, False]],
        [[True, False, False], [True, False, False], [True, True, False]],
    ], device=device)
    prior = torch.zeros(2, 3, 3, device=device).masked_fill(~support, float("-inf"))
    return mu, sigma, factors, prior


def test_supported_energy_preserves_weighted_value_and_all_gradients(device: torch.device) -> None:
    mu, sigma, factors, prior = _benign_fixture(device)
    kwargs = dict(irrep_dims=[2, 2], diagonal_out=False, precision_policy="fp32_escalate")
    reference = FullGaussian.coupling_energy(mu, sigma, mu, sigma, factors, **kwargs)
    actual = FullGaussian.coupling_energy(
        mu, sigma, mu, sigma, factors, pair_support=~torch.isneginf(prior), **kwargs)
    support = (~torch.isneginf(prior)).expand_as(actual)
    assert torch.count_nonzero(actual[~support]) == 0
    torch.testing.assert_close(actual[support], reference[support], rtol=2e-6, atol=2e-6)
    zero_self = mu.new_zeros(1, 3)
    f_actual = free_energy(zero_self, actual, 1., log_prior=prior)
    f_reference = free_energy(zero_self, reference, 1., log_prior=prior)
    torch.testing.assert_close(f_actual, f_reference)
    inputs = (mu, sigma, factors.exp_blocks, factors.inv_blocks)
    for a, b in zip(torch.autograd.grad(f_actual, inputs), torch.autograd.grad(f_reference, inputs)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


def test_oracle_preserves_the_direct_objective(device: torch.device) -> None:
    mu, sigma, factors, prior = _benign_fixture(device)
    mu_p, sigma_p = mu.detach() + .1, sigma.detach() * 1.1
    actual = belief_gradients_autograd(
        mu, sigma, mu_p, sigma_p, factors, family="gaussian_full", irrep_dims=[2, 2],
        log_prior=prior, full_cov_kl_precision="fp32_escalate",
    )
    energy = FullGaussian.coupling_energy(
        mu, sigma, mu.detach(), sigma.detach(), factors,
        irrep_dims=[2, 2], diagonal_out=False, precision_policy="fp32_escalate")
    self_div = self_divergence_for_alpha(
        FullGaussian(mu, sigma, _precision_policy="fp32_escalate"),
        FullGaussian(mu_p, sigma_p, _precision_policy="fp32_escalate"))
    objective = free_energy(self_div, energy, 1., log_prior=prior)
    expected = torch.autograd.grad(objective, (mu, sigma))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)


def test_supported_objective_gradient_matches_finite_difference(device: torch.device) -> None:
    mu, sigma, factors, prior = _benign_fixture(device)
    mu, sigma, prior = mu.double(), sigma.double(), prior.double()
    factors = T.CompactFactoredTransport(
        factors.exp_blocks.double(), factors.inv_blocks.double(), 4)
    scale = torch.tensor(1., device=device, dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        energy = FullGaussian.coupling_energy(
            value * mu, sigma, mu, sigma, factors, irrep_dims=[2, 2],
            diagonal_out=False, precision_policy="fp64",
            pair_support=~torch.isneginf(prior))
        return free_energy(mu.new_zeros(1, 3), energy, 1., log_prior=prior)

    derivative, = torch.autograd.grad(objective(scale), (scale,))
    step = 1e-5
    finite_difference = (objective(scale.detach() + step) - objective(scale.detach() - step)) / (2 * step)
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("n_heads", (2, 1))
def test_tiny_full_gaussian_forward_backward_and_logger(
    device: torch.device, n_heads: int,
) -> None:
    import math

    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    torch.manual_seed(23)
    cfg = VFE3Config(
        vocab_size=7, embed_dim=4, n_heads=n_heads, max_seq_len=3, batch_size=1,
        n_layers=1, n_e_steps=1, family="gaussian_full", gauge_group="block_glk",
        encode_mode="per_token", decode_mode="full_chunked",
        gradient_mode="filtering", e_step_update="gradient", e_step_gradient="unroll",
        beta_attention_prior="causal_alibi_noself",
        e_phi_lr=0., pos_phi="learned", pos_rotation="none",
        full_cov_congruence_precision="fp32_escalate", full_cov_congruence_certification=True,
        full_cov_kl_precision="fp32_escalate",
    )
    model = VFEModel(cfg).to(device)
    tokens = torch.tensor([[0, 1, 2]], device=device)
    _, loss, _ = model(tokens, targets=tokens)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    diagnostics = model.diagnostics(tokens)
    assert math.isfinite(diagnostics["total"])
    assert math.isfinite(diagnostics["sandwich_absmax"])
    pruned = n_heads > 1
    assert diagnostics["energy_evaluated_pair_frac"] == pytest.approx(4 / 9 if pruned else 1.)
    assert math.isnan(diagnostics["energy_abs_asymmetry"]) == pruned
    assert math.isnan(diagnostics["energy_rel_asymmetry"]) == pruned
