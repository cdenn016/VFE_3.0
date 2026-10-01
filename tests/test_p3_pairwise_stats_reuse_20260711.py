r"""Standalone construction and numerical consumer tests for diagonal-KL statistics reuse."""


import dataclasses
import importlib
from typing import Callable

import pytest
import torch

from vfe3.alpha_i import alpha_gradient_coefficient, alpha_is_per_coord
from vfe3.attention_prior import attention_log_prior
from vfe3.families.base import get_family
from vfe3.free_energy import (attention_weights, pairwise_energy, self_divergence_for_alpha)
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    transport_covariance,
    transport_mean,
)
from vfe3.gradients import kernels as kernels_module
from vfe3.gradients.pairwise_stats import diagonal_kl_pair_stats
from vfe3.gradients.kernels import belief_gradients, mm_exact_update


e_step_module = importlib.import_module("vfe3.inference.e_step")








@pytest.mark.parametrize(
    "irrep_dims",
    [None, [4], [2, 2], [1, 3]],
    ids=("no_blocks", "single_block", "equal_blocks", "unequal_blocks"),
)
def test_diagonal_kl_pair_stats_match_generic_energy_and_direct_statistics(
    irrep_dims: list[int] | None,
) -> None:
    eps = 1e-6
    kl_max = 12.0
    mu_q = torch.tensor(
        [
            [0.0, 0.5, -1.0, 2.0],
            [1.0, -0.5, 0.25, -1.5],
            [-0.75, 1.25, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    sigma_q = torch.tensor(
        [
            [0.5, 1.0, 2.0, 0.75],
            [1.5, 0.25, 0.8, 1.25],
            [0.4, 1.75, 0.6, 2.5],
        ],
        dtype=torch.float32,
    )
    mu_t = mu_q.unsqueeze(-2).expand(3, 3, 4).clone()
    sigma_t = sigma_q.unsqueeze(-2).expand(3, 3, 4).clone()
    mu_t[0, 1] += torch.tensor([0.25, -0.5, 0.75, -0.25])
    mu_t[1, 2] += 20.0
    sigma_t[2, 0] = torch.tensor([0.0, 0.5e-6, 1.5, 3.0])

    stats = diagonal_kl_pair_stats(
        mu_q,
        sigma_q,
        mu_t,
        sigma_t,
        kl_max=kl_max,
        eps=eps,
        irrep_dims=irrep_dims,
    )
    family = get_family("gaussian_diagonal")
    reference_energy = pairwise_energy(
        family(mu_q, sigma_q),
        family(mu_t, sigma_t),
        alpha=1.0,
        kl_max=kl_max,
        eps=eps,
        divergence_family="renyi",
        irrep_dims=irrep_dims,
    )
    reference_mask = (
        (reference_energy > 0.0) & (reference_energy < kl_max)
    ).to(reference_energy.dtype)

    torch.testing.assert_close(stats.energy, reference_energy, atol=1e-5, rtol=1e-6)
    assert torch.equal(stats.pair_mask, reference_mask)
    torch.testing.assert_close(stats.inv_sigma_t, 1.0 / sigma_t.clamp(min=eps))
    torch.testing.assert_close(stats.delta_tq, mu_t - mu_q.unsqueeze(-2))


def test_diagonal_kl_pair_stats_preserve_exact_kl_max_mask_boundary() -> None:
    mu_q = torch.tensor([[0.0]], dtype=torch.float32)
    sigma_q = torch.tensor([[3461.013427734375]], dtype=torch.float32)
    mu_t = torch.tensor([[[831.9871826171875]]], dtype=torch.float32)
    sigma_t = torch.tensor([[[3461.013427734375]]], dtype=torch.float32)
    family = get_family("gaussian_diagonal")
    reference_energy = pairwise_energy(
        family(mu_q, sigma_q),
        family(mu_t, sigma_t),
        alpha=1.0,
        kl_max=100.0,
        eps=1e-6,
        divergence_family="renyi",
    )
    reference_mask = (
        (reference_energy > 0.0) & (reference_energy < 100.0)
    ).to(reference_energy.dtype)
    stats = diagonal_kl_pair_stats(
        mu_q,
        sigma_q,
        mu_t,
        sigma_t,
        kl_max=100.0,
        eps=1e-6,
        irrep_dims=None,
    )

    assert torch.equal(reference_energy, torch.full_like(reference_energy, 100.0))
    assert torch.equal(stats.energy, reference_energy)
    assert torch.equal(stats.pair_mask, reference_mask)


def test_diagonal_kl_pair_stats_preserve_exact_zero_energy_boundary() -> None:
    mu_q = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    sigma_q = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    mu_t = mu_q.unsqueeze(-2).clone()
    sigma_t = sigma_q.unsqueeze(-2).clone()

    stats = diagonal_kl_pair_stats(
        mu_q,
        sigma_q,
        mu_t,
        sigma_t,
        kl_max=100.0,
        eps=1e-6,
        irrep_dims=None,
    )

    assert torch.equal(stats.energy, torch.zeros_like(stats.energy))
    assert torch.equal(stats.pair_mask, torch.zeros_like(stats.pair_mask))
    assert torch.equal(stats.delta_tq, torch.zeros_like(stats.delta_tq))


def test_diagonal_kl_pair_stats_match_certified_compact_self_link_mask() -> None:
    torch.manual_seed(0)
    group = get_group("block_glk")(4, 2)
    phi = 0.05 * torch.randn(1, 4, group.generators.shape[0], dtype=torch.float32)
    mu_q = torch.randn(1, 4, 4, dtype=torch.float32)
    sigma_q = torch.rand(1, 4, 4, dtype=torch.float32) + 0.5
    transport = e_step_module.build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        transport_mean_per_head=True,
        compact_phi_block_transport=True,
    )
    mu_t = transport_mean(transport, mu_q)
    sigma_t = transport_covariance(transport, sigma_q, diagonal_out=True)

    stats = diagonal_kl_pair_stats(
        mu_q,
        sigma_q,
        mu_t,
        sigma_t,
        kl_max=100.0,
        eps=1e-6,
        irrep_dims=[2, 2],
    )
    family = get_family("gaussian_diagonal")
    reference_energy = pairwise_energy(
        family(mu_q, sigma_q),
        family(mu_t, sigma_t),
        alpha=1.0,
        kl_max=100.0,
        eps=1e-6,
        divergence_family="renyi",
        irrep_dims=[2, 2],
    )
    reference_mask = (
        (reference_energy > 0.0) & (reference_energy < 100.0)
    ).to(reference_energy.dtype)
    self_links = torch.arange(mu_q.shape[-2])
    off_diagonal = ~torch.eye(mu_q.shape[-2], dtype=torch.bool)

    assert transport.same_frame_flat_cocycle
    torch.testing.assert_close(
        stats.energy[..., off_diagonal],
        reference_energy[..., off_diagonal],
        atol=1e-5,
        rtol=1e-6,
    )
    assert torch.equal(
        stats.energy[..., self_links, self_links],
        torch.zeros_like(stats.energy[..., self_links, self_links]),
    )
    assert torch.equal(
        stats.pair_mask[..., self_links, self_links],
        torch.zeros_like(stats.pair_mask[..., self_links, self_links]),
    )
    generic_active = reference_mask.bool()
    assert torch.equal(stats.pair_mask[generic_active], reference_mask[generic_active])


def test_p3_exact_kl_max_self_energy_gates_saturated_row_to_pass_through() -> None:
    mu = torch.tensor([[0.0]], dtype=torch.float32)
    sigma = torch.tensor([[3461.013427734375]], dtype=torch.float32)
    mu_p = torch.tensor([[831.9871826171875]], dtype=torch.float32)
    sigma_p = sigma.clone()
    omega = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    family = get_family("gaussian_diagonal")
    self_energy = self_divergence_for_alpha(
        family(mu, sigma),
        family(mu_p, sigma_p),
        alpha=1.0,
        kl_max=100.0,
        eps=1e-6,
        divergence_family="renyi",
        lambda_alpha_mode="constant",
    )

    assert torch.equal(self_energy, torch.full_like(self_energy, 100.0))
    grad_mu, grad_sigma = belief_gradients(
        mu,
        sigma,
        mu_p,
        sigma_p,
        omega,
        reuse_pairwise_kl_stats=True,
    )
    mu_star, sigma_star = mm_exact_update(
        mu,
        sigma,
        mu_p,
        sigma_p,
        omega,
        reuse_pairwise_kl_stats=True,
    )

    assert grad_sigma is not None
    assert torch.equal(grad_mu, torch.zeros_like(grad_mu))
    assert torch.equal(grad_sigma, torch.zeros_like(grad_sigma))
    assert torch.equal(mu_star, mu)
    assert torch.equal(sigma_star, sigma)


def _consumer_inputs(
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.tensor(
        [
            [-0.75, 0.20, 0.90, -0.40],
            [0.35, -1.10, 0.15, 0.80],
            [1.00, 0.45, -0.65, -0.25],
        ],
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    sigma = torch.tensor(
        [
            [0.70, 1.10, 0.85, 1.30],
            [1.20, 0.65, 1.40, 0.95],
            [0.80, 1.35, 0.75, 1.15],
        ],
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    mu_p = torch.tensor(
        [
            [-0.40, 0.10, 0.55, -0.10],
            [0.15, -0.75, 0.30, 0.45],
            [0.65, 0.20, -0.35, -0.05],
        ],
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    sigma_p = torch.tensor(
        [
            [0.95, 0.80, 1.05, 1.20],
            [0.90, 1.15, 0.75, 1.30],
            [1.25, 0.85, 1.10, 0.70],
        ],
        dtype=torch.float32,
    ).requires_grad_(requires_grad)
    K = mu.shape[-1]
    N = mu.shape[-2]
    omega = torch.eye(K, dtype=mu.dtype).expand(N, N, K, K).clone()
    return mu, sigma, mu_p, sigma_p, omega


def _dense_single_block_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """A flat GL(K) fixture whose generic self energies are exactly zero in float32."""
    torch.manual_seed(2)
    group = get_group("glk")(4)
    phi = 0.05 * torch.randn(1, 4, group.generators.shape[0], dtype=torch.float32)
    mu = torch.randn(1, 4, 4, dtype=torch.float32)
    sigma = torch.rand(1, 4, 4, dtype=torch.float32) + 0.5
    mu_p = torch.randn_like(mu)
    sigma_p = torch.rand_like(sigma) + 0.5
    omega = e_step_module.build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        transport_mean_per_head=True,
        compact_phi_block_transport=True,
    )
    assert isinstance(omega, torch.Tensor)
    return mu, sigma, mu_p, sigma_p, omega


def _certified_consumer_inputs(
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FactoredTransport]:
    mu, sigma, mu_p, sigma_p, _ = _consumer_inputs(requires_grad=requires_grad)
    N, K = mu.shape[-2:]
    eye = torch.eye(K, dtype=mu.dtype).expand(N, K, K).clone()
    omega = FactoredTransport(
        exp_phi=eye,
        exp_neg_phi=eye,
        irrep_dims=[2, 2],
        same_frame_flat_cocycle=True,
    )
    return mu, sigma, mu_p, sigma_p, omega


def _defensive_nested_rope_chain(terminal: object) -> RopeTransport:
    """Assemble an otherwise constructor-forbidden nested wrapper for predicate defense tests."""
    N, K = 3, 4
    dense = torch.eye(K, dtype=torch.float32).expand(N, N, K, K).clone()
    rope = torch.eye(K, dtype=torch.float32).expand(N, K, K).clone()
    inner = RopeTransport(base=dense, rope=rope, insertion="left")
    outer = RopeTransport(base=dense, rope=rope, insertion="left")
    inner.base = terminal
    outer.base = inner
    return outer


@pytest.mark.parametrize(
    ("terminal_name", "expected"),
    [
        pytest.param("dense", False, id="nested-dense"),
        pytest.param("factored", True, id="nested-factored"),
        pytest.param("compact", True, id="nested-compact"),
        pytest.param("unknown", False, id="nested-unknown"),
    ],
)
def test_pairwise_stats_reuse_predicate_fails_closed_after_nested_rope_unwrap(
    terminal_name: str,
    expected:      bool,
) -> None:
    _, _, _, _, factored = _certified_consumer_inputs()
    N, K = factored.exp_phi.shape[-3], factored.exp_phi.shape[-1]
    dense = torch.eye(K, dtype=torch.float32).expand(N, N, K, K).clone()
    compact_group = get_group("block_glk")(K, 2)
    compact = e_step_module.build_belief_transport(
        torch.zeros(N, compact_group.generators.shape[0]),
        compact_group,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )
    assert isinstance(compact, CompactFactoredTransport)
    terminals: dict[str, object] = {
        "dense": dense,
        "factored": factored,
        "compact": compact,
        "unknown": object(),
    }

    actual = kernels_module._pairwise_stats_reuse_is_sound(
        _defensive_nested_rope_chain(terminals[terminal_name]))

    assert actual is expected


def test_pairwise_stats_reuse_predicate_rejects_one_node_rope_cycle() -> None:
    wrapper = _defensive_nested_rope_chain(object())
    wrapper.base = wrapper

    assert kernels_module._pairwise_stats_reuse_is_sound(wrapper) is False


def test_pairwise_stats_reuse_predicate_rejects_two_node_rope_cycle() -> None:
    first = _defensive_nested_rope_chain(object())
    second = _defensive_nested_rope_chain(object())
    first.base = second
    second.base = first

    assert kernels_module._pairwise_stats_reuse_is_sound(first) is False


@pytest.mark.parametrize("consumer", [belief_gradients, mm_exact_update])
def test_dense_single_block_reuse_fails_closed_to_generic_pair_energy(
    monkeypatch: pytest.MonkeyPatch,
    consumer:    Callable,
) -> None:
    inputs = _dense_single_block_inputs()
    generic_energies: list[torch.Tensor] = []
    original_pairwise_energy = kernels_module.pairwise_energy

    def _capture_pairwise_energy(*args: object, **kwargs: object) -> torch.Tensor:
        energy = original_pairwise_energy(*args, **kwargs)
        generic_energies.append(energy.detach().clone())
        return energy

    expected = consumer(
        *inputs,
        irrep_dims=[4],
        reuse_pairwise_kl_stats=False,
    )
    generic_energies.clear()
    monkeypatch.setattr(kernels_module, "pairwise_energy", _capture_pairwise_energy)

    def _forbidden_dense_stats(*args: object, **kwargs: object) -> object:
        raise AssertionError("uncertified dense transport must recompute generic pair energy")

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _forbidden_dense_stats)
    actual = consumer(
        *inputs,
        irrep_dims=[4],
        reuse_pairwise_kl_stats=True,
    )

    assert len(generic_energies) == 1
    self_energy = generic_energies[0].diagonal(dim1=-2, dim2=-1)
    self_mask = ((self_energy > 0.0) & (self_energy < 100.0)).to(self_energy.dtype)
    assert torch.equal(self_energy, torch.zeros_like(self_energy))
    assert torch.equal(self_mask, torch.zeros_like(self_mask))
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.parametrize("consumer", [belief_gradients, mm_exact_update])
def test_uncertified_mixed_frame_factors_retain_pair_statistics_reuse(
    monkeypatch: pytest.MonkeyPatch,
    consumer:    Callable,
) -> None:
    mu, sigma, mu_p, sigma_p, omega = _certified_consumer_inputs()
    mixed_frame = dataclasses.replace(
        omega,
        exp_neg_phi=1.1 * omega.exp_neg_phi,
        same_frame_flat_cocycle=False,
    )
    expected = consumer(
        mu,
        sigma,
        mu_p,
        sigma_p,
        mixed_frame,
        irrep_dims=[2, 2],
        reuse_pairwise_kl_stats=False,
    )
    calls = 0
    real_helper = diagonal_kl_pair_stats

    def _counting_helper(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_helper(*args, **kwargs)

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _counting_helper)
    outputs = consumer(
        mu,
        sigma,
        mu_p,
        sigma_p,
        mixed_frame,
        irrep_dims=[2, 2],
        reuse_pairwise_kl_stats=True,
    )

    assert calls == 1
    assert all(torch.isfinite(output).all() for output in outputs)
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-6)


def _filtering_call(
    inputs:            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],

    *,
    lambda_alpha_mode: str,
    reuse:             bool,

    irrep_dims: list[int] | None = None,
    log_prior:  torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_mu, grad_sigma = belief_gradients(
        *inputs,
        tau=1.3,
        b0=0.7,
        c0=1.4,
        lambda_beta=0.9,
        value=0.8,
        lambda_alpha_mode=lambda_alpha_mode,
        irrep_dims=irrep_dims,
        log_prior=log_prior,
        reuse_pairwise_kl_stats=reuse,
    )
    assert grad_sigma is not None
    return grad_mu, grad_sigma


def _mm_call(
    inputs:            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],

    *,
    lambda_alpha_mode: str,
    need_sigma_update: bool,
    reuse:             bool,

    irrep_dims: list[int] | None,
    log_prior:  torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mm_exact_update(
        *inputs,
        tau=1.3,
        b0=0.7,
        c0=1.4,
        lambda_beta=0.9,
        value=0.8,
        lambda_alpha_mode=lambda_alpha_mode,
        need_sigma_update=need_sigma_update,
        irrep_dims=irrep_dims,
        log_prior=log_prior,
        reuse_pairwise_kl_stats=reuse,
    )


def test_p3_causal_noself_prior_preserves_exact_hard_masks() -> None:
    inputs = _certified_consumer_inputs()
    mu, sigma, _, _, omega = inputs
    log_prior = attention_log_prior(
        "causal_noself",
        mu.shape[-2],
        mu.shape[-2],
        device=mu.device,
        dtype=mu.dtype,
    )
    mu_t = transport_mean(omega, mu.detach())
    sigma_t = transport_covariance(omega, sigma.detach(), diagonal_out=True)
    stats = diagonal_kl_pair_stats(mu, sigma, mu_t, sigma_t, irrep_dims=[2, 2])
    beta = attention_weights(stats.energy, tau=1.3, log_prior=log_prior)
    forbidden = torch.triu(torch.ones_like(log_prior, dtype=torch.bool), diagonal=0)
    forbidden[0, 0] = False

    assert torch.equal(beta[..., forbidden], torch.zeros_like(beta[..., forbidden]))
    assert torch.equal(beta[..., 0, 0], torch.ones_like(beta[..., 0, 0]))

    filtering_off = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        reuse=False,
        irrep_dims=[2, 2],
        log_prior=log_prior,
    )
    filtering_on = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        reuse=True,
        irrep_dims=[2, 2],
        log_prior=log_prior,
    )
    mm_off = _mm_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        need_sigma_update=True,
        reuse=False,
        irrep_dims=[2, 2],
        log_prior=log_prior,
    )
    mm_on = _mm_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        need_sigma_update=True,
        reuse=True,
        irrep_dims=[2, 2],
        log_prior=log_prior,
    )

    for actual, expected in zip(filtering_on + mm_on, filtering_off + mm_off):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)


def test_p3_mm_frozen_sigma_is_exactly_equal_with_reuse() -> None:
    inputs = _certified_consumer_inputs()
    _, sigma_star = _mm_call(
        inputs,
        lambda_alpha_mode="state_dependent_per_coord",
        need_sigma_update=False,
        reuse=True,
        irrep_dims=[2, 2],
    )

    assert torch.equal(sigma_star, inputs[1])


def _legacy_pair_context(
    inputs:            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],

    *,
    lambda_alpha_mode: str,

    irrep_dims: list[int] | None,
) -> tuple[torch.Tensor, ...]:
    """Frozen pre-P3 pair setup shared by the filtering and MM references."""
    mu, sigma, mu_p, sigma_p, omega = inputs
    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega, mu_k)
    sigma_t = transport_covariance(
        omega,
        sigma_k,
        diagonal_out=(sigma_k.dim() == mu_k.dim()),
    )
    family = get_family("gaussian_diagonal")
    sd = self_divergence_for_alpha(
        family(mu, sigma),
        family(mu_p, sigma_p),
        alpha=1.0,
        kl_max=100.0,
        eps=1e-6,
        divergence_family="renyi",
        lambda_alpha_mode=lambda_alpha_mode,
    )
    energy = pairwise_energy(
        family(mu, sigma),
        family(mu_t, sigma_t),
        alpha=1.0,
        kl_max=100.0,
        eps=1e-6,
        divergence_family="renyi",
        irrep_dims=irrep_dims,
    )
    beta = attention_weights(energy, tau=1.3)
    pair_mask = ((energy > 0.0) & (energy < 100.0)).to(beta.dtype)
    coef = alpha_gradient_coefficient(
        sd,
        value=0.8,
        b0=0.7,
        c0=1.4,
        mode=lambda_alpha_mode,
    )
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)
    return mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, pair_mask, coef


def _legacy_filtering_reference(
    inputs:            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],

    *,
    lambda_alpha_mode: str,

    irrep_dims: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen pre-P3 filtering arithmetic for forward and VJP characterization."""
    mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, pair_mask, coef = _legacy_pair_context(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        irrep_dims=irrep_dims,
    )
    sp = sigma_p.clamp(min=1e-6)
    sq = sigma.clamp(min=1e-6)
    st = sigma_t.clamp(min=1e-6)
    if coef.shape[-1] == 1:
        raw_self = kernels_module._raw_diag_kl(mu, sigma, mu_p, sigma_p, eps=1e-6)
        self_mask = ((raw_self > 0.0) & (raw_self < 100.0)).to(mu.dtype).unsqueeze(-1)
    else:
        raw_self = kernels_module._raw_diag_kl_per_coord(mu, sigma, mu_p, sigma_p, eps=1e-6)
        self_mask = ((raw_self > 0.0) & (raw_self < 100.0)).to(mu.dtype)

    beta_pair = beta * pair_mask

    diff_mu = (mu.unsqueeze(-2) - mu_t) / st
    self_mu = self_mask * coef * (mu - mu_p) / sp
    pair_mu = kernels_module._pair_contract(beta_pair, diff_mu, irrep_dims)
    grad_mu = self_mu + 0.9 * pair_mu

    diff_sigma = 0.5 * (1.0 / st - 1.0 / sq.unsqueeze(-2))
    self_sigma = self_mask * coef * 0.5 * (1.0 / sp - 1.0 / sq)
    pair_sigma = kernels_module._pair_contract(beta_pair, diff_sigma, irrep_dims)
    grad_sigma = self_sigma + 0.9 * pair_sigma
    return grad_mu, grad_sigma


def _legacy_mm_reference(
    inputs:            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],

    *,
    lambda_alpha_mode: str,
    need_sigma_update: bool,

    irrep_dims: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen pre-P3 MM arithmetic for forward and VJP characterization."""
    mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, pair_mask, coef = _legacy_pair_context(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        irrep_dims=irrep_dims,
    )
    if coef.shape[-1] == 1:
        raw_self = kernels_module._raw_diag_kl(mu, sigma, mu_p, sigma_p, eps=1e-6)
        self_mask = (raw_self < 100.0).to(mu.dtype).unsqueeze(-1)
    else:
        raw_self = kernels_module._raw_diag_kl_per_coord(mu, sigma, mu_p, sigma_p, eps=1e-6)
        self_mask = (raw_self < 100.0).to(mu.dtype)
    a = self_mask * coef

    w = 0.9 * (beta * pair_mask)

    sp = sigma_p.clamp(min=1e-6)
    st = sigma_t.clamp(min=1e-6)
    prior_precision = a / sp
    pair_precision = kernels_module._pair_contract(w, 1.0 / st, irrep_dims)
    pair_mean = kernels_module._pair_contract(w, mu_t / st, irrep_dims)
    precision = prior_precision + pair_precision
    safe_precision = precision.clamp(min=1e-6)
    mu_star = (a * mu_p / sp + pair_mean) / safe_precision
    degenerate = precision <= 1e-6
    mu_star = torch.where(degenerate, mu, mu_star)
    if not need_sigma_update:
        return mu_star, sigma

    pair_mass = kernels_module._pair_mass(w, irrep_dims, mu.shape[-1])
    sigma_star = ((a + pair_mass) / safe_precision).clamp(min=1e-6)
    sigma_star = torch.where(degenerate, sigma, sigma_star)
    return mu_star, sigma_star


def test_p3_filtering_consumes_poisoned_statistics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _certified_consumer_inputs()
    reference = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        reuse=False,
        irrep_dims=[2, 2],
    )
    real_helper = diagonal_kl_pair_stats
    calls = 0

    def _poisoned_helper(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        stats = real_helper(*args, **kwargs)
        return dataclasses.replace(
            stats,
            inv_sigma_t=1.75 * stats.inv_sigma_t,
            delta_tq=stats.delta_tq + 0.40,
        )

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _poisoned_helper, raising=False)
    poisoned = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        reuse=True,
        irrep_dims=[2, 2],
    )

    assert calls == 1
    assert not torch.allclose(poisoned[0], reference[0], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(poisoned[1], reference[1], atol=1e-4, rtol=1e-4)


def test_p3_mm_consumes_poisoned_statistics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _certified_consumer_inputs()
    reference = _mm_call(
        inputs,
        lambda_alpha_mode="state_dependent_per_coord",
        need_sigma_update=True,
        reuse=False,
        irrep_dims=[2, 2],
    )
    real_helper = diagonal_kl_pair_stats
    calls = 0

    def _poisoned_helper(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        stats = real_helper(*args, **kwargs)
        return dataclasses.replace(
            stats,
            inv_sigma_t=1.75 * stats.inv_sigma_t,
            delta_tq=stats.delta_tq + 0.40,
        )

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _poisoned_helper, raising=False)
    poisoned = _mm_call(
        inputs,
        lambda_alpha_mode="state_dependent_per_coord",
        need_sigma_update=True,
        reuse=True,
        irrep_dims=[2, 2],
    )

    assert calls == 1
    assert not torch.allclose(poisoned[0], reference[0], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(poisoned[1], reference[1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("consumer", ["filtering", "mm_exact"])
def test_p3_false_forbids_statistics_helper(
    monkeypatch: pytest.MonkeyPatch,
    consumer:    str,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the pair-statistics helper must not run while reuse is false")

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _forbidden, raising=False)
    inputs = _consumer_inputs()
    if consumer == "filtering":
        _filtering_call(
            inputs,
            lambda_alpha_mode="state_dependent",
            reuse=False,
            irrep_dims=[2, 2],
        )
    else:
        _mm_call(
            inputs,
            lambda_alpha_mode="state_dependent",
            need_sigma_update=True,
            reuse=False,
            irrep_dims=[2, 2],
        )


@pytest.mark.parametrize(
    "route",
    [
        "smoothing",
        "renyi_half",
        "gaussian_full",
        "entropy_suppressed",
    ],
)
def test_p3_oracle_routes_forbid_statistics_helper(
    monkeypatch: pytest.MonkeyPatch,
    route:       str,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"the pair-statistics helper must not run on the {route} route")

    oracle_calls = 0

    def _oracle(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal oracle_calls
        oracle_calls += 1
        assert isinstance(args[0], torch.Tensor)
        assert isinstance(args[1], torch.Tensor)
        return torch.zeros_like(args[0]), torch.zeros_like(args[1])

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _forbidden, raising=False)
    monkeypatch.setattr(kernels_module, "belief_gradients_autograd", _oracle)
    inputs = _consumer_inputs()
    route_kwargs: dict[str, object] = {}
    if route == "smoothing":
        route_kwargs["gradient_mode"] = "smoothing"
    elif route == "renyi_half":
        route_kwargs["renyi_order"] = 0.5
    elif route == "gaussian_full":
        mu, sigma, mu_p, sigma_p, omega = inputs
        inputs = (mu, torch.diag_embed(sigma), mu_p, torch.diag_embed(sigma_p), omega)
        route_kwargs["family"] = "gaussian_full"
    elif route == "entropy_suppressed":
        route_kwargs["include_attention_entropy"] = False
    else:
        raise AssertionError(f"unhandled route {route!r}")

    outputs = belief_gradients(
        *inputs,
        reuse_pairwise_kl_stats=True,
        **route_kwargs,
    )

    assert oracle_calls == 1
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize("consumer", ["filtering", "mm_exact"])
def test_p3_float64_inputs_fall_back_without_statistics_helper(
    monkeypatch: pytest.MonkeyPatch,
    consumer:    str,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("the float32-only pair-statistics helper must not run")

    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _forbidden, raising=False)
    inputs = tuple(tensor.to(torch.float64) for tensor in _consumer_inputs())
    if consumer == "filtering":
        outputs = _filtering_call(
            inputs,
            lambda_alpha_mode="state_dependent",
            reuse=True,
            irrep_dims=None,
        )
        reference = _filtering_call(
            inputs,
            lambda_alpha_mode="state_dependent",
            reuse=False,
            irrep_dims=None,
        )
    else:
        outputs = _mm_call(
            inputs,
            lambda_alpha_mode="state_dependent",
            need_sigma_update=True,
            reuse=True,
            irrep_dims=None,
        )
        reference = _mm_call(
            inputs,
            lambda_alpha_mode="state_dependent",
            need_sigma_update=True,
            reuse=False,
            irrep_dims=None,
        )

    assert all(output.dtype == torch.float64 for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    assert all(torch.equal(output, expected) for output, expected in zip(outputs, reference))


@pytest.mark.parametrize(
    ("supply_inv", "supply_delta"),
    [(True, False), (False, True)],
    ids=("inverse_only", "delta_only"),
)
def test_p3_filtering_kernel_rejects_asymmetric_optional_statistics(
    supply_inv:   bool,
    supply_delta: bool,
) -> None:
    inputs = _consumer_inputs()
    mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, pair_mask, coef = _legacy_pair_context(
        inputs,
        lambda_alpha_mode="constant",
        irrep_dims=[2, 2],
    )
    stats = diagonal_kl_pair_stats(
        mu,
        sigma,
        mu_t,
        sigma_t,
        kl_max=100.0,
        eps=1e-6,
        irrep_dims=[2, 2],
    )
    optional_stats: dict[str, torch.Tensor] = {}
    if supply_inv:
        optional_stats["pair_inv_sigma_t"] = stats.inv_sigma_t
    if supply_delta:
        optional_stats["pair_delta_tq"] = stats.delta_tq

    with pytest.raises(
        ValueError,
        match="pair_inv_sigma_t and pair_delta_tq must be provided together",
    ):
        kernels_module._diag_kl_filtering_kernel(
            mu,
            sigma,
            mu_p,
            sigma_p,
            mu_t,
            sigma_t,
            beta,
            coef,
            irrep_dims=[2, 2],
            pair_mask=pair_mask,
            **optional_stats,
        )


def test_p3_filtering_receives_statistics_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The P3 reuse hoist must actually reach the kernel, not silently fall back to the legacy
    pair arithmetic (which is value-identical but slower, so an output comparison cannot tell)."""
    inputs = _certified_consumer_inputs()
    expected = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        reuse=True,
        irrep_dims=[2, 2],
    )
    captured: dict[str, object] = {}

    def _capturing_get_kernel(name: str) -> Callable:
        eager_kernel = kernels_module._KERNELS[name]

        def _capture(*args: object, **kwargs: object) -> object:
            captured["pair_inv_sigma_t"] = kwargs.get("pair_inv_sigma_t")
            captured["pair_delta_tq"] = kwargs.get("pair_delta_tq")
            captured["pair_mask_dtype"] = kwargs["pair_mask"].dtype
            captured["beta_dtype"] = args[6].dtype
            return eager_kernel(*args, **kwargs)

        return _capture

    monkeypatch.setattr(kernels_module, "get_kernel", _capturing_get_kernel)
    actual = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent",
        reuse=True,
        irrep_dims=[2, 2],
    )

    assert isinstance(captured["pair_inv_sigma_t"], torch.Tensor)
    assert isinstance(captured["pair_delta_tq"], torch.Tensor)
    assert captured["pair_mask_dtype"] == captured["beta_dtype"]
    for captured_value, eager_value in zip(actual, expected):
        torch.testing.assert_close(captured_value, eager_value, atol=1e-5, rtol=1e-6)


def _output_vjp(
    outputs: tuple[torch.Tensor, torch.Tensor],
    inputs:  tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    probe_mu = torch.linspace(
        0.2,
        1.1,
        outputs[0].numel(),
        device=outputs[0].device,
        dtype=outputs[0].dtype,
    ).reshape_as(outputs[0])
    probe_sigma = torch.linspace(
        -0.7,
        0.4,
        outputs[1].numel(),
        device=outputs[1].device,
        dtype=outputs[1].dtype,
    ).reshape_as(outputs[1])
    objective = (outputs[0] * probe_mu).sum() + (outputs[1] * probe_sigma).sum()
    return torch.autograd.grad(objective, inputs[:4], retain_graph=True)


@pytest.mark.parametrize(
    "lambda_alpha_mode",
    ["constant", "state_dependent", "state_dependent_per_coord"],
)
@pytest.mark.parametrize("irrep_dims", [None, [2, 2]], ids=("single", "two_head"))
def test_p3_filtering_on_matches_off_forward_and_vjp(
    lambda_alpha_mode: str,
    irrep_dims:        list[int] | None,
) -> None:
    inputs = _certified_consumer_inputs(requires_grad=True)
    legacy = _legacy_filtering_reference(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        irrep_dims=irrep_dims,
    )
    disabled = _filtering_call(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        reuse=False,
        irrep_dims=irrep_dims,
    )
    enabled = _filtering_call(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        reuse=True,
        irrep_dims=irrep_dims,
    )

    for actual, expected in zip(disabled, legacy):
        assert torch.equal(actual, expected)
    for actual, expected in zip(enabled, legacy):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)
    legacy_vjp = _output_vjp(legacy, inputs)
    disabled_vjp = _output_vjp(disabled, inputs)
    enabled_vjp = _output_vjp(enabled, inputs)
    for actual, expected in zip(disabled_vjp, legacy_vjp):
        assert torch.equal(actual, expected)
    for actual, expected in zip(enabled_vjp, legacy_vjp):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "lambda_alpha_mode",
    ["constant", "state_dependent", "state_dependent_per_coord"],
)
@pytest.mark.parametrize("need_sigma_update", [False, True], ids=("frozen_sigma", "update_sigma"))
@pytest.mark.parametrize("irrep_dims", [None, [2, 2]], ids=("single", "two_head"))
def test_p3_mm_on_matches_off_forward_and_vjp(
    lambda_alpha_mode: str,
    need_sigma_update: bool,
    irrep_dims:        list[int] | None,
) -> None:
    inputs = _certified_consumer_inputs(requires_grad=True)
    legacy = _legacy_mm_reference(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        need_sigma_update=need_sigma_update,
        irrep_dims=irrep_dims,
    )
    disabled = _mm_call(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        need_sigma_update=need_sigma_update,
        reuse=False,
        irrep_dims=irrep_dims,
    )
    enabled = _mm_call(
        inputs,
        lambda_alpha_mode=lambda_alpha_mode,
        need_sigma_update=need_sigma_update,
        reuse=True,
        irrep_dims=irrep_dims,
    )

    for actual, expected in zip(disabled, legacy):
        assert torch.equal(actual, expected)
    for actual, expected in zip(enabled, legacy):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)
    legacy_vjp = _output_vjp(legacy, inputs)
    disabled_vjp = _output_vjp(disabled, inputs)
    enabled_vjp = _output_vjp(enabled, inputs)
    for actual, expected in zip(disabled_vjp, legacy_vjp):
        assert torch.equal(actual, expected)
    for actual, expected in zip(enabled_vjp, legacy_vjp):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)


def test_diagonal_kl_pair_stats_preserve_batched_head_layout_and_live_vjps() -> None:
    torch.manual_seed(29)
    B, N, K = 2, 3, 4
    mu_q = torch.randn(B, N, K, dtype=torch.float32, requires_grad=True)
    sigma_q = (torch.rand(B, N, K, dtype=torch.float32) + 0.5).requires_grad_()
    mu_t = torch.randn(B, N, N, K, dtype=torch.float32, requires_grad=True)
    sigma_t = (torch.rand(B, N, N, K, dtype=torch.float32) + 0.5).requires_grad_()

    stats = diagonal_kl_pair_stats(
        mu_q,
        sigma_q,
        mu_t,
        sigma_t,
        kl_max=50.0,
        eps=1e-6,
        irrep_dims=[2, 2],
    )

    assert stats.energy.shape == (B, 2, N, N)
    assert stats.pair_mask.shape == (B, 2, N, N)
    assert stats.inv_sigma_t.shape == (B, N, N, K)
    assert stats.delta_tq.shape == (B, N, N, K)
    assert stats.energy.dtype == torch.float32
    assert stats.pair_mask.dtype == torch.float32

    objective = (
        stats.energy.sum()
        + 1e-3 * stats.inv_sigma_t.sum()
        + 1e-3 * stats.delta_tq.sum()
    )
    vjps = torch.autograd.grad(objective, (mu_q, sigma_q, mu_t, sigma_t))
    assert all(torch.isfinite(vjp).all() for vjp in vjps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_p3_cuda_filtering_and_mm_reuse_smoke() -> None:
    device = torch.device("cuda")
    mu, sigma, mu_p, sigma_p, omega = _certified_consumer_inputs()
    inputs = (
        mu.to(device).detach().requires_grad_(True),
        sigma.to(device).detach().requires_grad_(True),
        mu_p.to(device).detach().requires_grad_(True),
        sigma_p.to(device).detach().requires_grad_(True),
        dataclasses.replace(
            omega,
            exp_phi=omega.exp_phi.to(device),
            exp_neg_phi=omega.exp_neg_phi.to(device),
        ),
    )
    filtering = _filtering_call(
        inputs,
        lambda_alpha_mode="state_dependent_per_coord",
        reuse=True,
        irrep_dims=[2, 2],
    )
    mm_update = _mm_call(
        inputs,
        lambda_alpha_mode="state_dependent_per_coord",
        need_sigma_update=True,
        reuse=True,
        irrep_dims=[2, 2],
    )

    assert all(torch.isfinite(output).all() for output in filtering + mm_update)
    assert all(torch.isfinite(vjp).all() for vjp in _output_vjp(filtering, inputs))
    assert all(torch.isfinite(vjp).all() for vjp in _output_vjp(mm_update, inputs))

    mu_t = transport_mean(inputs[-1], inputs[0].detach())
    sigma_t = transport_covariance(inputs[-1], inputs[1].detach(), diagonal_out=True)
    stats = diagonal_kl_pair_stats(
        inputs[0],
        inputs[1],
        mu_t,
        sigma_t,
        kl_max=100.0,
        eps=1e-6,
        irrep_dims=[2, 2],
    )
    family = get_family("gaussian_diagonal")
    reference_energy = pairwise_energy(
        family(inputs[0], inputs[1]),
        family(mu_t, sigma_t),
        alpha=1.0,
        kl_max=100.0,
        eps=1e-6,
        divergence_family="renyi",
        irrep_dims=[2, 2],
    )
    reference_mask = (
        (reference_energy > 0.0) & (reference_energy < 100.0)
    ).to(reference_energy.dtype)

    assert torch.equal(stats.pair_mask, reference_mask)


@pytest.mark.parametrize(
    ("irrep_dims", "message"),
    [
        pytest.param([], "nonempty", id="empty"),
        pytest.param([0, 4], "positive", id="nonpositive"),
        pytest.param([1, 2], "sum to K=4", id="wrong_sum"),
    ],
)
def test_diagonal_kl_pair_stats_reject_invalid_irrep_partitions(
    irrep_dims: list[int],
    message:    str,
) -> None:
    mu_q = torch.zeros(2, 4, dtype=torch.float32)
    sigma_q = torch.ones_like(mu_q)
    mu_t = torch.zeros(2, 2, 4, dtype=torch.float32)
    sigma_t = torch.ones_like(mu_t)

    with pytest.raises(ValueError, match=message):
        diagonal_kl_pair_stats(
            mu_q,
            sigma_q,
            mu_t,
            sigma_t,
            irrep_dims=irrep_dims,
        )
