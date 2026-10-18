"""Opt-in conditioning-triggered float64 source recovery in the projected scorers (2026-09-23).

A float32 projected covariance whose zero-jitter Cholesky SUCCEEDS but whose certificate
eps_32 ||G||_F ||G^{-1}||_F exceeds 1e-4 is rebuilt from its source in float64 only under
``projected_factorization="gram_certified"``. A covariance whose Cholesky FAILS is always rebuilt.
"""
from typing import Callable, Iterator

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vfe3.relational_scoring import (
    _cholesky_status, _strict_cholesky, projected_condition_recovery, projected_gaussian_kl,
    projected_moments, projected_noisy_query_cost, reset_source_recovery_elements,
    set_projected_condition_recovery, source_recovery_elements,
)


@pytest.fixture
def condition_recovery() -> Iterator[Callable[[bool], bool]]:
    previous = projected_condition_recovery()
    yield set_projected_condition_recovery
    set_projected_condition_recovery(previous)


def _ill_conditioned(device: torch.device) -> tuple:
    r"""33 query tokens; query token 0 has cond(G) = 1e4, SPD at float32 but over the certificate."""
    generator = torch.Generator().manual_seed(5)
    mu_q, mu_k = (torch.randn(1, 33, 1, 2, generator=generator).to(device) for _ in range(2))
    sigma_q, sigma_k = (torch.ones(1, 33, 1, 2, device=device) for _ in range(2))
    sigma_q[0, 0, 0, 1] = 1e-4
    inverse = torch.eye(2, device=device).expand(1, 33, 1, 2, 2)
    S = torch.eye(2, device=device)[None]
    aq, Gq = projected_moments(mu_q, sigma_q, inverse, S)
    ak, Gk = projected_moments(mu_k, sigma_k, inverse, S)
    return (aq, Gq, ak, Gk, (sigma_q, inverse, S, mu_q), (sigma_k, inverse, S, mu_k),
            (mu_q, sigma_q, mu_k, sigma_k, inverse, S))


def _reference(sources: tuple, cost: Callable) -> torch.Tensor:
    mu_q, sigma_q, mu_k, sigma_k, inverse, S = (x.double() for x in sources)
    aq, Gq = projected_moments(mu_q, sigma_q, inverse, S)
    ak, Gk = projected_moments(mu_k, sigma_k, inverse, S)
    return cost(aq, Gq, ak, Gk)


def test_config_disabled_validated_and_active_on_supported_routes() -> None:
    from tests.test_relational_integration import _cfg
    assert _cfg(projected_factorization="gram").projected_factorization == "gram"
    with pytest.raises(ValueError, match="projected_factorization"):
        _cfg(projected_factorization=True)
    assert "projected_factorization" not in _cfg().inactive_fields()


@pytest.mark.parametrize("enabled", [True, False])
def test_model_installs_the_config_policy(condition_recovery: Callable[[bool], bool],
                                          enabled: bool) -> None:
    from tests.test_relational_integration import _cfg
    from vfe3.model.model import VFEModel

    condition_recovery(not enabled)
    VFEModel(_cfg(projected_factorization="gram_certified" if enabled else "gram"))
    assert projected_condition_recovery() is enabled


def test_off_skips_the_certificate_inverse(condition_recovery: Callable[[bool], bool],
                                           device: torch.device) -> None:
    class _Ops(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.names: list = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    _, Gq, *_ = _ill_conditioned(device)
    for enabled in (True, False):
        condition_recovery(enabled)
        ops = _Ops()
        with ops:
            _, failed, screen, status = _cholesky_status(Gq.detach())
        assert ("cholesky_inverse" in ops.names) is enabled
        assert bool(screen.any()) is enabled
        assert status.tolist()[2] is enabled                      # suspicious_any
        assert not bool(failed.any())


@pytest.mark.parametrize("cost", [projected_gaussian_kl, projected_noisy_query_cost])
def test_ill_conditioned_entry_keeps_its_float32_factor_when_off(
        condition_recovery: Callable[[bool], bool], device: torch.device, cost: Callable) -> None:
    aq, Gq, ak, Gk, source_q, source_k, sources = _ill_conditioned(device)
    reference = _reference(sources, cost)
    for enabled in (False, True):
        condition_recovery(enabled)
        recovered = _strict_cholesky(Gq.detach(), source_q)[1]
        assert (recovered is not None) is enabled
        reset_source_recovery_elements()
        score = cost(aq, Gq, ak, Gk, projection_q=source_q, projection_k=source_k)
        assert (source_recovery_elements() > 0) is enabled
        assert score.dtype == torch.float32 and torch.isfinite(score).all()
        # cond(G) = 1e4 is diagonal here, so the float32 factor is exact up to rounding.
        torch.testing.assert_close(score.double(), reference, rtol=1e-4, atol=1e-4)


def test_failed_factorization_still_recovers_when_off(condition_recovery: Callable[[bool], bool],
                                                       device: torch.device) -> None:
    from tests.test_relational_source_factorization import _inputs

    condition_recovery(False)
    mu, sigma, inverse, S = _inputs(device)
    identity = torch.eye(3, device=device)[None]
    aq, Gq = projected_moments(mu, sigma, inverse, S)
    ak, Gk = projected_moments(mu, sigma, inverse, identity)
    assert torch.linalg.cholesky_ex(Gq)[1].ne(0).any()           # rounded Gram lost SPD
    reset_source_recovery_elements()
    score = projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=(sigma, inverse, S),
                                  projection_k=(sigma, inverse, identity))
    assert source_recovery_elements() > 0
    aq64, Gq64 = projected_moments(mu.double(), sigma.double(), inverse.double(), S.double())
    ak64, Gk64 = projected_moments(mu.double(), sigma.double(), inverse.double(), identity.double())
    torch.testing.assert_close(score.double(), projected_gaussian_kl(aq64, Gq64, ak64, Gk64),
                               rtol=2e-5, atol=2e-5)
