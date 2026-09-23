"""Full-Gaussian covariance transport keeps the SPD sandwich in float64."""

from __future__ import annotations

from typing import Optional
import warnings

import pytest
import torch

from vfe3.families.gaussian import FullGaussian
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    transport_covariance,
    transport_mean,
)
from vfe3.metrics import gauge_equivariance_residual


@pytest.fixture(autouse=True)
def _pin_fp64_congruence():
    r"""Pin the float64 congruence island for this file (audit 2026-08-06 C1).

    Every test here asserts what the FLOAT64 path does -- the sandwich retained in float64, its
    backward staying finite, the flag reaching the structural wrappers. None is about which policy
    happens to be the shipped default, so relying on that default silently retargeted the whole file
    at the float32 path the moment the default was flipped. Pinning states the intent and keeps the
    file meaningful under either default.
    """
    from vfe3.geometry.transport import (
        set_full_cov_congruence_certification,
        set_full_cov_congruence_precision,
    )

    previous = set_full_cov_congruence_precision("fp64")
    previous_certificate = set_full_cov_congruence_certification(True)
    yield
    set_full_cov_congruence_certification(previous_certificate)
    set_full_cov_congruence_precision(previous)


_OMEGA = torch.tensor(
    [[
        0.057695649564266205,
        0.016874510794878006,
        0.20264112949371338,
        0.07412116974592209,
    ], [
        -0.49864041805267334,
        -0.11252593994140625,
        -0.2768476605415344,
        0.1497870236635208,
    ], [
        -0.4640336036682129,
        -0.11606039106845856,
        -0.08934598416090012,
        0.28132152557373047,
    ], [
        0.4184989333152771,
        0.15523938834667206,
        0.12013706564903259,
        -0.3998108506202698,
    ]],
    dtype=torch.float32,
).reshape(1, 1, 4, 4)


def test_full_gaussian_transport_retains_spd_lost_by_float32_storage() -> None:
    sigma = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)

    stored = transport_covariance(_OMEGA, sigma, diagonal_out=False)
    retained = transport_covariance(
        _OMEGA,
        sigma,
        retain_full_precision=True,
        diagonal_out=False,
    )
    family_retained = FullGaussian.transport_dispersion(
        sigma,
        _OMEGA,
        diagonal_out=False,
    )

    stored_eigmin = torch.linalg.eigvalsh(stored.double())[..., 0]
    retained_eigmin = torch.linalg.eigvalsh(retained)[..., 0]
    high_precision_oracle = torch.einsum(
        "...ijkl,...jlm,...ijnm->...ijkn",
        _OMEGA.double(),
        sigma.double(),
        _OMEGA.double(),
    )
    rejected_cast = high_precision_oracle.to(sigma.dtype)
    rejected_cast_eigmin = torch.linalg.eigvalsh(rejected_cast.double())[..., 0]

    assert -1.0e-7 < float(rejected_cast_eigmin) < 0.0
    assert stored.dtype == torch.float64
    assert torch.equal(stored, high_precision_oracle)
    assert 0.0 < float(stored_eigmin) < 1.0e-7
    assert retained.dtype == torch.float64
    assert 0.0 < float(retained_eigmin) < 1.0e-7
    assert torch.equal(stored, retained)
    assert torch.equal(family_retained, retained)


def test_retained_transport_has_finite_backward_and_float32_public_kl() -> None:
    omega = _OMEGA.clone().requires_grad_(True)
    sigma = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).requires_grad_(True)
    mu = torch.zeros(1, 4, dtype=torch.float32)

    transported_mu = transport_mean(omega, mu)
    transported_sigma = FullGaussian.transport_dispersion(
        sigma,
        omega,
        diagonal_out=False,
    )
    transported = FullGaussian.from_transported(
        transported_mu,
        transported_sigma,
        sigma,
    )
    divergence = transported.renyi_closed_form(
        transported,
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )
    grad_omega, grad_sigma = torch.autograd.grad(divergence.sum(), (omega, sigma))

    assert transported_sigma.dtype == torch.float64
    assert divergence.dtype == torch.float32
    assert torch.isfinite(divergence).all()
    assert torch.isfinite(grad_omega).all()
    assert torch.isfinite(grad_sigma).all()


def test_genuine_float64_full_gaussian_still_returns_float64() -> None:
    mu = torch.zeros(1, 4, dtype=torch.float64)
    sigma = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4)

    divergence = FullGaussian(mu, sigma).renyi_closed_form(
        FullGaussian(mu, sigma),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )

    assert divergence.dtype == torch.float64


def test_genuine_float64_covariance_with_float32_mean_returns_float64() -> None:
    mu = torch.zeros(1, 4, dtype=torch.float32)
    sigma = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4)

    divergence = FullGaussian(mu, sigma).renyi_closed_form(
        FullGaussian(mu, sigma),
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )

    assert divergence.dtype == torch.float64


def test_transported_float64_mean_with_float32_source_covariance_returns_float64() -> None:
    mu = torch.zeros(1, 4, dtype=torch.float64)
    source_sigma = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    retained_sigma = source_sigma.double()
    transported = FullGaussian.from_transported(mu, retained_sigma, source_sigma)

    divergence = transported.renyi_closed_form(
        transported,
        alpha=1.0,
        kl_max=100.0,
        eps=1.0e-6,
    )

    assert divergence.dtype == torch.float64


def test_internal_transport_public_dtype_survives_views_and_family_combinators() -> None:
    sigma = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    mu = torch.zeros(1, 4, dtype=torch.float32)
    transported = FullGaussian.from_transported(
        transport_mean(_OMEGA, mu)[0],
        FullGaussian.transport_dispersion(sigma, _OMEGA, diagonal_out=False)[0],
        sigma,
    )

    variants = (
        transported,
        transported.block(0, 2),
        transported.broadcast_over_keys(),
        FullGaussian.stack([transported, transported], dim=0),
    )
    for belief in variants:
        divergence = belief.renyi_closed_form(
            belief,
            alpha=1.0,
            kl_max=100.0,
            eps=1.0e-6,
        )
        assert belief.sigma.dtype == torch.float64
        assert divergence.dtype == torch.float32


def test_full_precision_flag_reaches_structural_transport_wrappers() -> None:
    sigma = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    identity = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    factored = FactoredTransport(_OMEGA[0], identity, [4])
    compact = CompactFactoredTransport(
        _OMEGA[0].unsqueeze(-3),
        identity.unsqueeze(-3),
        4,
    )
    rope = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    transports = (
        factored,
        compact,
        RopeTransport(_OMEGA, rope, on_cov=False, insertion="left"),
        RopeTransport(_OMEGA, rope, on_cov=True, insertion="left"),
    )

    for omega in transports:
        retained = transport_covariance(
            omega,
            sigma,
            retain_full_precision=True,
            diagonal_out=False,
        )
        assert retained.dtype == torch.float64
        assert float(torch.linalg.eigvalsh(retained)[..., 0]) > 0.0


def test_gauge_equivariance_metric_uses_full_family_transport_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"transport": 0, "factory": 0}
    transport_impl = FullGaussian.transport_dispersion.__func__
    factory_impl = FullGaussian.from_transported.__func__

    def transport_spy(
        cls:        type[FullGaussian],
        dispersion: torch.Tensor,
        omega:      object,

        *,
        diagonal_out: Optional[bool] = None,
    ) -> torch.Tensor:
        calls["transport"] += 1
        return transport_impl(
            cls,
            dispersion,
            omega,
            diagonal_out=diagonal_out,
        )

    def factory_spy(
        cls:               type[FullGaussian],
        mu:                torch.Tensor,
        dispersion:        torch.Tensor,
        source_dispersion: torch.Tensor,
    ) -> FullGaussian:
        calls["factory"] += 1
        return factory_impl(cls, mu, dispersion, source_dispersion)

    monkeypatch.setattr(FullGaussian, "transport_dispersion", classmethod(transport_spy))
    monkeypatch.setattr(FullGaussian, "from_transported", classmethod(factory_spy))

    group = get_group("block_glk")(4, 2)
    mu = torch.zeros(2, 4, dtype=torch.float32)
    sigma = torch.eye(4, dtype=torch.float32).repeat(2, 1, 1)
    dense = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(2, 2, 1, 1)
    blocks = torch.eye(2, dtype=torch.float32).reshape(1, 1, 2, 2).repeat(2, 2, 1, 1)
    compact = CompactFactoredTransport(blocks, blocks.clone(), 4)

    for omega in (dense, compact):
        gauge_equivariance_residual(
            mu,
            sigma,
            omega,
            group,
            n_samples=1,
            seed=0,
        )

    assert calls["transport"] > 0
    assert calls["factory"] > 0
