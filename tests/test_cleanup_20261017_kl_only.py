r"""The divergence seam is KL-only (audit 2026-10-16 F93, cleanup 2026-10-17).

The retired ``divergence_family`` / ``renyi_order`` fields left the KL, the alpha = 1 Renyi divergence,
as the only configured divergence, so the squared Hellinger, Bhattacharyya and Jeffreys functionals and
the alpha != 1 blend kernels were removed. The Renyi names keep their ``alpha`` keyword for signature
stability; any order outside the KL switch :math:`|\alpha - 1| < 10^{-6}` must raise. Tiny CPU tensors.
"""

import pytest
import torch

import vfe3.divergence as divergence
from vfe3.families.base import _KL_ALPHA_SWITCH, _renyi_from_log_partition, kl, renyi, renyi_per_coord
from vfe3.families.gaussian import BlockGaussian, DiagonalGaussian, FullGaussian

_REMOVED = ("squared_hellinger", "bhattacharyya", "jeffreys", "bhattacharyya_per_coord", "jeffreys_per_coord")
_NON_KL_ORDERS = (0.5, 1.0 + 10.0 * _KL_ALPHA_SWITCH, 1.0 - 10.0 * _KL_ALPHA_SWITCH, 2.0, 0.0, -1.0)


def _pair(family: str) -> tuple:
    g = torch.Generator().manual_seed(0)
    K = 4
    mu_q = torch.randn(2, K, generator=g)
    mu_p = torch.randn(2, K, generator=g)
    if family == "diagonal":
        return (DiagonalGaussian(mu_q, torch.rand(2, K, generator=g) + 0.5),
                DiagonalGaussian(mu_p, torch.rand(2, K, generator=g) + 0.5))
    A = torch.randn(2, K, K, generator=g)
    S = A @ A.transpose(-1, -2) + torch.eye(K)
    cls = FullGaussian if family == "full" else BlockGaussian
    if family == "block":
        mask = torch.block_diag(torch.ones(2, 2), torch.ones(2, 2)).bool()
        S = torch.where(mask, S, torch.zeros_like(S))
    return cls(mu_q, S), cls(mu_p, 0.5 * S + torch.eye(K))


def test_removed_functionals_are_gone_from_the_divergence_seam() -> None:
    for name in _REMOVED:
        assert not hasattr(divergence, name), name
        assert name not in divergence.__all__, name
    assert divergence.divergence_functionals() == ("renyi",)
    assert divergence.divergence_functionals_per_coord() == ("renyi",)
    for name in ("squared_hellinger", "bhattacharyya", "jeffreys"):
        with pytest.raises(KeyError):
            divergence.get_functional(name)


@pytest.mark.parametrize("family", ["diagonal", "full", "block"])
@pytest.mark.parametrize("alpha", _NON_KL_ORDERS)
def test_every_renyi_entry_point_refuses_a_non_kl_order(family: str, alpha: float) -> None:
    q, p = _pair(family)
    calls = (
        lambda: renyi(q, p, alpha=alpha),
        lambda: renyi_per_coord(q, p, alpha=alpha),
        lambda: q.renyi_closed_form(p, alpha=alpha),
        lambda: q.renyi_per_coord(p, alpha=alpha),
        lambda: _renyi_from_log_partition(q, p, alpha=alpha, kl_max=100.0, eps=1e-6),
    )
    for call in calls:
        with pytest.raises(ValueError, match=r"only the KL divergence \(alpha = 1\)"):
            call()


@pytest.mark.parametrize("family", ["diagonal", "full", "block"])
def test_orders_inside_the_kl_switch_are_the_kl(family: str) -> None:
    q, p = _pair(family)
    reference = kl(q, p)
    for alpha in (1.0, 1.0 + 0.5 * _KL_ALPHA_SWITCH, 1.0 - 0.5 * _KL_ALPHA_SWITCH):
        assert torch.equal(renyi(q, p, alpha=alpha), reference)
