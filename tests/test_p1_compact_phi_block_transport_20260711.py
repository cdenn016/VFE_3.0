"""P1 compact phi block transport: routing, equivalence, and model wiring."""

from tests.profiles import test_config

import dataclasses
import importlib

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import (
    _COMPOSE,
    _equal_diag_blocks,
    compose_bch,
    compose_phi,
    register_compose,
)
from vfe3.geometry.retraction import retract_phi
from vfe3.geometry.rope import build_rope_rotation
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    transport_covariance,
    transport_mean,
    transport_scale,
)
from vfe3.gradients import oracle as oracle_module
from vfe3.gradients.pairwise_stats import diagonal_kl_pair_stats
from vfe3.inference.e_step import build_belief_transport
from vfe3.inference.e_step import phi_alignment_loss
from vfe3.model.model import VFEModel


e_step_module = importlib.import_module("vfe3.inference.e_step")
lie_ops_module = importlib.import_module("vfe3.geometry.lie_ops")
transport_module = importlib.import_module("vfe3.geometry.transport")


def test_dense_and_compact_bch_keep_float32_under_cpu_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(37)
    group = get_group("block_glk")(4, 2)
    phi1 = 0.03 * torch.randn(2, 3, group.generators.shape[0])
    phi2 = 0.03 * torch.randn(3, group.generators.shape[0])
    kwargs = {
        "order": 4,
        "gram_pinv_": group.gram_pinv(),
        "block_dims": group.irrep_dims,
    }
    dense_ref = compose_bch(phi1, phi2, group.generators, **kwargs)
    compact_ref = compose_bch(
        phi1,
        phi2,
        group.generators,
        compact_blocks=True,
        **kwargs,
    )

    original_autocast = torch.amp.autocast
    autocast_calls: list[tuple[str, bool]] = []

    def _autocast_spy(device_type: str, *args: object, **kwargs: object) -> object:
        autocast_calls.append((device_type, bool(kwargs.get("enabled", True))))
        return original_autocast(device_type, *args, **kwargs)

    monkeypatch.setattr(torch.amp, "autocast", _autocast_spy)
    dense_plain = compose_bch(phi1, phi2, group.generators, **kwargs)
    compact_plain = compose_bch(
        phi1,
        phi2,
        group.generators,
        compact_blocks=True,
        **kwargs,
    )
    assert autocast_calls == []
    assert torch.equal(dense_plain, dense_ref)
    assert torch.equal(compact_plain, compact_ref)

    with original_autocast("cpu", dtype=torch.bfloat16):
        dense = compose_bch(phi1, phi2, group.generators, **kwargs)
        compact = compose_bch(
            phi1,
            phi2,
            group.generators,
            compact_blocks=True,
            **kwargs,
        )
    assert autocast_calls == [("cpu", False), ("cpu", False)]
    assert dense.dtype == torch.float32
    assert compact.dtype == torch.float32
    assert torch.equal(dense, dense_ref)
    assert torch.equal(compact, compact_ref)

    phi1_double = phi1.double()
    phi2_double = phi2.double()
    generators_double = group.generators.double()
    gram_double = group.gram_pinv().double()
    with original_autocast("cpu", dtype=torch.bfloat16):
        dense_double = compose_bch(
            phi1_double,
            phi2_double,
            generators_double,
            order=4,
            gram_pinv_=gram_double,
            block_dims=group.irrep_dims,
        )
        compact_double = compose_bch(
            phi1_double,
            phi2_double,
            generators_double,
            order=4,
            compact_blocks=True,
            gram_pinv_=gram_double,
            block_dims=group.irrep_dims,
        )
    assert autocast_calls == [("cpu", False), ("cpu", False)]
    assert dense_double.dtype == torch.float64
    assert compact_double.dtype == torch.float64


def test_compact_phi_block_transport_changes_only_opted_in_block_glk_layout(
    device:      torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = 0.1 * torch.randn(2, 3, group.generators.shape[0], device=device)
    compact_calls: list[torch.Size] = []
    original = transport_module._stable_compact_glk_exp_pair

    def _spy(blocks: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        compact_calls.append(blocks.shape)
        return original(blocks, **kwargs)

    monkeypatch.setattr(transport_module, "_stable_compact_glk_exp_pair", _spy)

    legacy = build_belief_transport(phi, group, transport_mode="flat")
    explicit_false = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=False,
    )
    assert compact_calls == []
    compact = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )

    assert isinstance(legacy, FactoredTransport)
    assert isinstance(explicit_false, FactoredTransport)
    assert torch.equal(legacy.exp_phi, explicit_false.exp_phi)
    assert torch.equal(legacy.exp_neg_phi, explicit_false.exp_neg_phi)
    assert isinstance(compact, CompactFactoredTransport)
    assert compact_calls == [torch.Size((2, 3, 2, 2, 2))]
    assert compact.exp_blocks.shape == (2, 3, 2, 2, 2)
    assert compact.inv_blocks.shape == (2, 3, 2, 2, 2)


@pytest.mark.parametrize(
    "threshold",
    [20.0, 0.0],
)
def test_compact_phi_block_transport_preserves_global_full_frame_clamp_and_dtype_rule(
    device:         torch.device,
    threshold:     float,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = torch.zeros(1, 1, group.generators.shape[0], device=device)
    phi[..., 0] = 15.0
    phi[..., 4] = 15.0

    legacy = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        exp_fp64_norm_threshold=threshold,
    )
    compact = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        exp_fp64_norm_threshold=threshold,
        compact_phi_block_transport=True,
    )

    legacy_pos = _equal_diag_blocks(legacy.exp_phi, 2, 2)
    legacy_neg = _equal_diag_blocks(legacy.exp_neg_phi, 2, 2)
    assert torch.equal(compact.exp_blocks, legacy_pos)
    assert torch.equal(compact.inv_blocks, legacy_neg)


@pytest.mark.parametrize("compact_enabled", [False, True], ids=("dense_vertex", "compact"))
def test_certified_flat_cocycle_has_exact_self_links_and_off_diagonal_parity(
    monkeypatch:      pytest.MonkeyPatch,
    compact_enabled: bool,
) -> None:
    torch.manual_seed(41)
    group = get_group("block_glk")(4, 2)
    phi = (0.05 * torch.randn(1, 4, group.generators.shape[0])).requires_grad_()
    mu = torch.randn(1, 4, 4, requires_grad=True)
    sigma_diag = (torch.rand(1, 4, 4) + 0.5).requires_grad_()
    scale = (torch.rand(1, 4, 4) + 0.5).requires_grad_()
    raw = torch.randn(1, 4, 4, 4)
    sigma_full = (raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(4)).requires_grad_()
    transport = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        transport_mean_per_head=True,
        compact_phi_block_transport=compact_enabled,
    )

    expected_type = CompactFactoredTransport if compact_enabled else FactoredTransport
    assert isinstance(transport, expected_type)
    assert transport.same_frame_flat_cocycle
    uncertified = dataclasses.replace(transport, same_frame_flat_cocycle=False)

    def _dense_forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("certified self-link correction materialized a dense pair operator")

    monkeypatch.setattr(expected_type, "to_dense_omega", _dense_forbidden)
    mean_kernel_name = "_compact_factored_mean" if compact_enabled else "_factored_per_head_mean"
    original_mean_kernel = getattr(transport_module, mean_kernel_name)
    allocated_pointers: list[int] = []

    def _capture_mean_output(*args: object, **kwargs: object) -> torch.Tensor:
        output = original_mean_kernel(*args, **kwargs)
        allocated_pointers.append(output.data_ptr())
        return output

    monkeypatch.setattr(transport_module, mean_kernel_name, _capture_mean_output)
    mean = transport_mean(transport, mu)
    assert mean.data_ptr() == allocated_pointers[0]
    covariance_diag = transport_covariance(transport, sigma_diag, diagonal_out=True)
    covariance_full = transport_covariance(transport, sigma_full, diagonal_out=False)
    projected_scale = transport_scale(scale, transport)
    raw_mean = transport_mean(uncertified, mu)
    raw_covariance_diag = transport_covariance(uncertified, sigma_diag, diagonal_out=True)
    raw_covariance_full = transport_covariance(uncertified, sigma_full, diagonal_out=False)
    raw_projected_scale = transport_scale(scale, uncertified)

    self_links = torch.arange(mu.shape[-2])
    off_diagonal = ~torch.eye(mu.shape[-2], dtype=torch.bool)
    assert torch.equal(mean[:, self_links, self_links], mu)
    assert torch.equal(covariance_diag[:, self_links, self_links], sigma_diag)
    assert torch.equal(covariance_full[:, self_links, self_links], sigma_full)
    assert torch.equal(projected_scale[:, self_links, self_links], scale)
    assert torch.equal(mean[:, off_diagonal], raw_mean[:, off_diagonal])
    assert torch.equal(covariance_diag[:, off_diagonal], raw_covariance_diag[:, off_diagonal])
    assert torch.equal(covariance_full[:, off_diagonal], raw_covariance_full[:, off_diagonal])
    assert torch.equal(projected_scale[:, off_diagonal], raw_projected_scale[:, off_diagonal])

    stats = diagonal_kl_pair_stats(
        mu,
        sigma_diag,
        mean,
        covariance_diag,
        irrep_dims=[2, 2],
    )
    assert torch.equal(
        stats.energy[..., self_links, self_links],
        torch.zeros_like(stats.energy[..., self_links, self_links]),
    )
    assert torch.equal(
        stats.pair_mask[..., self_links, self_links],
        torch.zeros_like(stats.pair_mask[..., self_links, self_links]),
    )

    self_objective = (
        mean[:, self_links, self_links].sum()
        + covariance_diag[:, self_links, self_links].sum()
        + covariance_full[:, self_links, self_links].sum()
        + projected_scale[:, self_links, self_links].sum()
    )
    self_gradients = torch.autograd.grad(
        self_objective,
        (phi, mu, sigma_diag, sigma_full, scale),
        retain_graph=True,
    )
    assert torch.equal(self_gradients[0], torch.zeros_like(phi))
    assert torch.equal(self_gradients[1], torch.ones_like(mu))
    assert torch.equal(self_gradients[2], torch.ones_like(sigma_diag))
    assert torch.equal(self_gradients[3], torch.ones_like(sigma_full))
    torch.testing.assert_close(
        self_gradients[4],
        torch.ones_like(scale),
        atol=1e-7,
        rtol=0.0,
    )

    mean_weight = torch.randn_like(mean)
    diagonal_weight = torch.randn_like(covariance_diag)
    full_weight = torch.randn_like(covariance_full)
    scale_weight = torch.randn_like(projected_scale)
    mean_weight[:, self_links, self_links] = 0.0
    diagonal_weight[:, self_links, self_links] = 0.0
    full_weight[:, self_links, self_links] = 0.0
    scale_weight[:, self_links, self_links] = 0.0
    corrected_objective = (
        (mean * mean_weight).sum()
        + (covariance_diag * diagonal_weight).sum()
        + (covariance_full * full_weight).sum()
        + (projected_scale * scale_weight).sum()
    )
    raw_objective = (
        (raw_mean * mean_weight).sum()
        + (raw_covariance_diag * diagonal_weight).sum()
        + (raw_covariance_full * full_weight).sum()
        + (raw_projected_scale * scale_weight).sum()
    )
    corrected_gradients = torch.autograd.grad(
        corrected_objective,
        (phi, mu, sigma_diag, sigma_full, scale),
        retain_graph=True,
    )
    raw_gradients = torch.autograd.grad(
        raw_objective,
        (phi, mu, sigma_diag, sigma_full, scale),
    )
    for corrected, reference in zip(corrected_gradients, raw_gradients):
        assert torch.equal(corrected, reference)


def test_same_frame_certificate_survives_supported_compositions() -> None:
    torch.manual_seed(43)
    group = get_group("block_glk")(4, 2)
    phi = 0.03 * torch.randn(1, 3, group.generators.shape[0])
    dense_vertex = build_belief_transport(phi, group, transport_mode="flat")
    compact = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )
    rope_matrix = build_rope_rotation(
        torch.arange(3), group.irrep_dims,
        base=100.0, device=phi.device, dtype=phi.dtype,
    )
    wrapped = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        rope=rope_matrix,
        rope_on_cov=True,
        compact_phi_block_transport=True, rope_insertion="left")

    assert isinstance(dense_vertex, FactoredTransport)
    assert isinstance(compact, CompactFactoredTransport)
    assert isinstance(wrapped, RopeTransport)
    assert dense_vertex.same_frame_flat_cocycle
    assert compact.same_frame_flat_cocycle
    assert compact.unsqueeze(0).same_frame_flat_cocycle
    assert dataclasses.replace(dense_vertex, mean_per_head=True).same_frame_flat_cocycle
    assert oracle_module._transport_to_float(compact).same_frame_flat_cocycle
    assert oracle_module._transport_to_float(dense_vertex).same_frame_flat_cocycle
    assert wrapped.base.same_frame_flat_cocycle
    assert wrapped.same_frame_flat_cocycle
    assert oracle_module._transport_to_float(wrapped).same_frame_flat_cocycle

    mu = torch.randn(1, 3, 4)
    sigma = torch.rand(1, 3, 4) + 0.5
    self_links = torch.arange(mu.shape[-2])
    assert torch.equal(transport_mean(wrapped, mu)[:, self_links, self_links], mu)
    assert torch.equal(
        transport_covariance(wrapped, sigma, diagonal_out=True)[:, self_links, self_links],
        sigma,
    )


@pytest.mark.parametrize("rope_kind", ["nonorthogonal", "non_block"])
def test_direct_rope_wrapper_fails_closed_for_untrusted_rotation(rope_kind: str) -> None:
    group = get_group("block_glk")(4, 2)
    phi = torch.zeros(1, 3, group.generators.shape[0])
    base = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )
    assert isinstance(base, CompactFactoredTransport)
    assert base.same_frame_flat_cocycle

    rope = torch.eye(4).expand(1, 3, 4, 4).clone()
    if rope_kind == "nonorthogonal":
        rope[..., 0, 0] = 2.0
    else:
        rope[..., 0, 2] = 0.5
    wrapped = RopeTransport(base=base, rope=rope, on_cov=True, insertion="left")
    mu = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + 1.0
    pre_rotated = torch.einsum("...jlk,...jl->...jk", rope, mu)
    expected_mean = torch.einsum(
        "...ikl,...ijl->...ijk", rope, transport_mean(base, pre_rotated),
    )
    self_links = torch.arange(mu.shape[-2])

    assert not wrapped.same_frame_flat_cocycle
    assert torch.equal(transport_mean(wrapped, mu), expected_mean)
    assert not torch.equal(expected_mean[:, self_links, self_links], mu)

    if rope_kind == "nonorthogonal":
        sigma = torch.eye(4).expand(1, 3, 4, 4).clone()
        self_operator = rope @ rope.transpose(-1, -2)
        expected_self = self_operator @ sigma @ self_operator.transpose(-1, -2)
        covariance = transport_covariance(wrapped, sigma, diagonal_out=False)
        assert torch.equal(covariance[:, self_links, self_links], expected_self)


def test_certified_rope_over_uncertified_compact_base_keeps_raw_full_self_transport() -> None:
    N, H, d, K = 3, 2, 2, 4
    exp_blocks = 2.0 * torch.eye(d).expand(1, N, H, d, d).clone()
    inv_blocks = 3.0 * torch.eye(d).expand(1, N, H, d, d).clone()
    base = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    rope = torch.eye(K).expand(1, N, K, K).clone()
    wrapped = RopeTransport(
        insertion="left",   # legacy composition under test
        base=base,
        rope=rope,
        on_cov=True,
        same_frame_flat_cocycle=True,
    )
    mu = torch.arange(N * K, dtype=torch.float32).reshape(1, N, K) + 1.0
    sigma = torch.eye(K).expand(1, N, K, K).clone()
    self_links = torch.arange(N)

    assert not base.same_frame_flat_cocycle
    assert wrapped.same_frame_flat_cocycle
    assert torch.equal(
        transport_mean(wrapped, mu)[:, self_links, self_links],
        6.0 * mu,
    )
    assert torch.equal(
        transport_covariance(wrapped, sigma, diagonal_out=False)[:, self_links, self_links],
        36.0 * sigma,
    )


def test_self_link_certificates_reject_non_boolean_values() -> None:
    dense = torch.eye(4).expand(1, 3, 4, 4).clone()
    blocks = torch.eye(2).expand(1, 3, 2, 2, 2).clone()
    with pytest.raises(ValueError, match="same_frame_flat_cocycle must be a bool"):
        FactoredTransport(dense, dense, [2, 2], same_frame_flat_cocycle=1)
    with pytest.raises(ValueError, match="same_frame_flat_cocycle must be a bool"):
        CompactFactoredTransport(blocks, blocks, 4, same_frame_flat_cocycle=1)
    base = FactoredTransport(dense, dense, [2, 2], same_frame_flat_cocycle=True)
    with pytest.raises(ValueError, match="same_frame_flat_cocycle must be a bool"):
        RopeTransport(
            insertion="left",   # legacy composition under test
            base=base,
            rope=dense,
            same_frame_flat_cocycle=1,
        )


@pytest.mark.parametrize("mean_per_head", [False, True])
def test_compact_phi_block_transport_matches_mean_covariance_and_vjp(
    device:        torch.device,
    mean_per_head: bool,
) -> None:
    torch.manual_seed(7)
    group = get_group("block_glk")(4, 2, device=device)
    phi0 = 0.05 * torch.randn(1, 3, group.generators.shape[0], device=device)
    mu0 = torch.randn(1, 3, 4, device=device)
    sigma0 = torch.rand(1, 3, 4, device=device) + 0.5
    mean_weight = torch.randn(1, 3, 3, 4, device=device)
    cov_weight = torch.randn(1, 3, 3, 4, device=device)

    outputs = []
    gradients = []
    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        sigma = sigma0.clone().requires_grad_(True)
        transport = build_belief_transport(
            phi,
            group,
            transport_mode="flat",
            transport_mean_per_head=mean_per_head,
            compact_phi_block_transport=compact_enabled,
        )
        mean = transport_mean(transport, mu)
        covariance = transport_covariance(transport, sigma, diagonal_out=True)
        loss = (mean * mean_weight).sum() + 0.1 * (covariance * cov_weight).sum()
        outputs.append((mean, covariance))
        gradients.append(torch.autograd.grad(loss, (phi, mu, sigma)))

    for legacy, compact in zip(outputs[0], outputs[1]):
        assert torch.allclose(compact, legacy, atol=2e-6, rtol=1e-5)
    for legacy, compact in zip(gradients[0], gradients[1]):
        assert torch.allclose(compact, legacy, atol=5e-5, rtol=1e-5)


def test_compact_phi_block_transport_matches_full_covariance_and_vjp(
    device: torch.device,
) -> None:
    torch.manual_seed(17)
    group = get_group("block_glk")(4, 2, device=device)
    phi0 = 0.04 * torch.randn(1, 2, group.generators.shape[0], device=device)
    raw = torch.randn(1, 2, 4, 4, device=device)
    sigma0 = raw @ raw.transpose(-1, -2) + 0.5 * torch.eye(4, device=device)
    weight = torch.randn(1, 2, 2, 4, 4, device=device)

    outputs = []
    gradients = []
    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        sigma = sigma0.clone().requires_grad_(True)
        transport = build_belief_transport(
            phi,
            group,
            transport_mode="flat",
            compact_phi_block_transport=compact_enabled,
        )
        covariance = transport_covariance(transport, sigma, diagonal_out=False)
        outputs.append(covariance)
        gradients.append(torch.autograd.grad((covariance * weight).sum(), (phi, sigma)))

    assert torch.allclose(outputs[1], outputs[0], atol=2e-5, rtol=1e-5)
    for legacy, compact in zip(gradients[0], gradients[1]):
        assert torch.allclose(compact, legacy, atol=5e-5, rtol=1e-5)


def test_compact_phi_block_transport_matches_legacy_autocast_dtype_and_vjp() -> None:
    torch.manual_seed(23)
    group = get_group("block_glk")(4, 2)
    phi0 = 0.03 * torch.randn(1, 3, group.generators.shape[0])
    mu0 = torch.randn(1, 3, 4)
    sigma0 = torch.rand(1, 3, 4) + 0.5
    weight = torch.randn(1, 3, 3, 4)
    outputs = []
    gradients = []
    factor_dtypes = []
    factor_values = []

    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            transport = build_belief_transport(
                phi,
                group,
                transport_mode="flat",
                transport_mean_per_head=True,
                compact_phi_block_transport=compact_enabled,
            )
            mean = transport_mean(transport, mu0)
            covariance = transport_covariance(transport, sigma0, diagonal_out=True)
            loss = (mean * weight).sum() + 0.1 * (covariance * weight).sum()
        factors = transport.exp_blocks if compact_enabled else transport.exp_phi
        factor_dtypes.append(factors.dtype)
        factor_values.append(
            factors if compact_enabled else _equal_diag_blocks(factors, 2, 2))
        outputs.append((mean, covariance))
        gradients.append(torch.autograd.grad(loss, phi)[0])

    # Audit 2026-07-25 F2: the represented group element is never STORED below float32, on EITHER
    # route. Rounding the exponential (and its float64 inverse) back to the autocast dtype made
    # U U^-1 deviate from the identity like eps * cond(U) -- reaching O(1) near cond ~ 1/eps_bf16 --
    # and the vertex factor's cocycle property IS Regime-I flatness. This test's contract is dense/
    # compact PARITY, which is preserved; the parity now holds at float32 instead of at bfloat16.
    assert factor_dtypes == [torch.float32, torch.float32]
    assert torch.equal(factor_values[1], factor_values[0])
    for legacy, compact in zip(outputs[0], outputs[1]):
        assert torch.allclose(compact, legacy, atol=8e-3, rtol=1e-2)
    assert torch.allclose(gradients[1], gradients[0], atol=8e-3, rtol=1e-2)


def test_compact_phi_block_transport_keeps_float64_under_autocast() -> None:
    group = get_group("block_glk")(4, 2, dtype=torch.float64)
    phi = 0.03 * torch.randn(1, 3, group.generators.shape[0], dtype=torch.float64)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        legacy = build_belief_transport(phi, group, transport_mode="flat")
        compact = build_belief_transport(
            phi,
            group,
            transport_mode="flat",
            compact_phi_block_transport=True,
        )

    assert legacy.exp_phi.dtype == torch.float64
    assert compact.exp_blocks.dtype == torch.float64
    assert torch.equal(compact.exp_blocks, _equal_diag_blocks(legacy.exp_phi, 2, 2))


def test_compact_phi_block_transport_rejects_unmarked_same_shape_basis() -> None:
    canonical = get_group("block_glk")(4, 2)
    custom = GaugeGroup(
        name="block_glk",
        generators=canonical.generators.roll(1, dims=0),
        irrep_dims=list(canonical.irrep_dims),
        skew_symmetric=False,
    )
    phi = 0.03 * torch.randn(1, 3, custom.generators.shape[0])

    transport = build_belief_transport(
        phi,
        custom,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )

    assert isinstance(transport, FactoredTransport)


def test_phi_alignment_loss_accepts_compact_transport_and_matches_legacy_vjp() -> None:
    torch.manual_seed(29)
    group = get_group("block_glk")(4, 2)
    mu = torch.randn(1, 3, 4)
    sigma = torch.rand(1, 3, 4) + 0.5
    phi0 = 0.03 * torch.randn(1, 3, group.generators.shape[0])
    values = []
    gradients = []

    for compact_enabled in (False, True):
        phi = phi0.clone().requires_grad_(True)
        value = phi_alignment_loss(
            mu,
            sigma,
            phi,
            group,
            transport_mean_per_head=True,
            compact_phi_block_transport=compact_enabled,
        )
        values.append(value)
        gradients.append(torch.autograd.grad(value, phi)[0])

    assert torch.allclose(values[1], values[0], atol=2e-5, rtol=1e-5)
    assert torch.allclose(gradients[1], gradients[0], atol=5e-5, rtol=1e-5)


def test_compose_registry_default_call_does_not_require_compact_keyword() -> None:
    name = "_p1_legacy_compose_signature"

    @register_compose(name)
    def _legacy_compose(
        phi1:       torch.Tensor,
        phi2:       torch.Tensor,
        generators: torch.Tensor,

        *,
        order:      int,
        gram_pinv_: torch.Tensor,
        block_dims: list[int],
    ) -> torch.Tensor:
        del generators, order, gram_pinv_, block_dims
        return phi1 + phi2

    try:
        generators = get_group("block_glk")(4, 2).generators
        phi1 = torch.zeros(2, generators.shape[0])
        phi2 = torch.ones_like(phi1)
        out = compose_phi(
            phi1,
            phi2,
            generators,
            mode=name,
            gram_pinv_=torch.eye(generators.shape[0]),
            block_dims=[2, 2],
        )
        assert torch.equal(out, phi2)
    finally:
        _COMPOSE.pop(name, None)


def test_compact_phi_block_transport_keeps_bch_retraction_packed_and_matches_vjp(
    device:      torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = get_group("block_glk")(4, 2, device=device)
    phi = (0.02 * torch.randn(1, 3, group.generators.shape[0], device=device)).requires_grad_()
    delta = (0.01 * torch.randn_like(phi)).requires_grad_()
    cotangent = torch.randn_like(phi)

    legacy = retract_phi(
        phi,
        delta,
        group,
        mode="bch",
        compact_blocks=False,
    )
    legacy_grad = torch.autograd.grad(legacy, (phi, delta), cotangent)

    def _dense_embed_forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("canonical packed BCH retraction must not embed a dense KxK matrix")

    monkeypatch.setattr(lie_ops_module, "embed_phi", _dense_embed_forbidden)
    phi_compact = phi.detach().clone().requires_grad_()
    delta_compact = delta.detach().clone().requires_grad_()
    compact = retract_phi(
        phi_compact,
        delta_compact,
        group,
        mode="bch",
        compact_blocks=True,
    )
    compact_grad = torch.autograd.grad(compact, (phi_compact, delta_compact), cotangent)

    assert torch.allclose(compact, legacy, atol=2e-6, rtol=1e-5)
    assert torch.allclose(compact_grad[0], legacy_grad[0], atol=2e-6, rtol=1e-5)
    assert torch.allclose(compact_grad[1], legacy_grad[1], atol=2e-6, rtol=1e-5)


def test_projected_model_rejects_inner_phi_updates() -> None:
    with pytest.raises(ValueError, match="e_phi_lr"):
        test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                    n_layers=1, n_e_steps=1, e_phi_lr=0.01, pos_phi="none")


@pytest.mark.parametrize("route", ["glk", "tied", "cross", "single_head", "block_glk"])
def test_compact_phi_block_transport_routes_by_group_layout(route: str) -> None:
    if route == "glk":
        group = get_group("glk")(4)
    elif route == "tied":
        group = get_group("tied_block_glk")(4, 2)
    elif route == "cross":
        group = get_group("block_glk")(4, 2, cross_couplings=[(0, 1)])
    elif route == "single_head":
        group = get_group("block_glk")(4, 1)
    else:
        group = get_group("block_glk")(4, 2)
    phi = 0.05 * torch.randn(1, 3, group.generators.shape[0])

    transport = build_belief_transport(
        phi,
        group,
        transport_mode="flat",
        compact_phi_block_transport=True,
    )

    assert isinstance(transport, CompactFactoredTransport) is (route == "block_glk")


def _tiny_config() -> VFE3Config:
    return test_config(
        vocab_size=9,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        n_layers=1,
        n_e_steps=1,
        e_phi_lr=0.0,
        decode_mode="diagonal_chunked",
    )


def test_compact_phi_block_transport_reaches_shared_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(13)
    dense_model = VFEModel(_tiny_config()).eval()
    compact_model = VFEModel(_tiny_config()).eval()
    compact_model.load_state_dict(dense_model.state_dict())
    monkeypatch.setattr(dense_model, "_compact_phi_blocks_enabled", lambda: False)
    token_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # logits is None on the fused-CE branch by design and the default decode_mode is chunked
    # (audit 2026-08-06), so the logit parity is checked on the unfused no-targets path.
    dense_logits = dense_model(token_ids)
    _, dense_loss, _ = dense_model(token_ids, targets)
    dense_grad, = torch.autograd.grad(dense_loss, dense_model.prior_bank.phi_embed)

    seen: list[object] = []
    original = e_step_module.build_belief_transport

    def _spy(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(e_step_module, "build_belief_transport", _spy)
    compact_logits = compact_model(token_ids)
    _, compact_loss, _ = compact_model(token_ids, targets)
    compact_grad, = torch.autograd.grad(compact_loss, compact_model.prior_bank.phi_embed)

    assert sum(isinstance(item, CompactFactoredTransport) for item in seen) >= 2
    assert torch.allclose(compact_logits, dense_logits, atol=2e-5, rtol=1e-5)
    assert torch.allclose(compact_loss, dense_loss, atol=2e-5, rtol=1e-5)
    assert torch.allclose(compact_grad, dense_grad, atol=5e-5, rtol=1e-5)
