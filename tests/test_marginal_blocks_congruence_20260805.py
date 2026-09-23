r"""Cross-head off-block elimination in the compact full congruence (audit 2026-08-05).

``BeliefParams.coupling_energy`` forwards its ``irrep_dims`` to ``transport_dispersion`` as
``marginal_blocks``: a promise that the transported dispersion's ONLY consumer is the
``pairwise_energy`` below it, which with more than one irrep block reads exactly
``block(h*d, (h+1)*d)`` per head and never an off-block. A compact-factored transport may then
skip the cross-head congruence blocks (H^2 -> H block products).

Pins: (a) the promise is honored -- the fast energy equals the full congruence's energy, on a
block-diagonal covariance AND on a general one whose cross-head blocks are genuinely nonzero;
(b) the cross-head blocks really are dropped, so the flag is doing something rather than being
inert; (c) a MISALIGNED partition straddling a head boundary fails closed to the full congruence;
(d) ``irrep_dims`` None / single-block fails closed, because there ``pairwise_energy`` reads
full-K; (e) a DIRECT ``transport_dispersion`` caller -- the gauge-equivariance certificate in
metrics and the model's sandwich diagnostic -- does not pass the promise and keeps every
off-block.
"""

import pytest
import torch

from vfe3.families import get_family
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    _compact_factored_full_covariance,
    _reads_only_head_marginals,
    transport_covariance,
)

H, d = 2, 5
K = H * d
B, N = 2, 6
KW = dict(alpha=1.0, kl_max=8.0 * K, eps=1e-6,
          divergence_family="renyi", diagonal_out=False)


def _omega(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = 0.15 * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    a = a - a.transpose(-1, -2)                    # skew => exp is invertible, exp(-a) its inverse
    return CompactFactoredTransport(
        torch.matrix_exp(a).float(), torch.matrix_exp(-a).float(), K)


def _spd(seed, block_diagonal):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(B, N, K, K, generator=g, dtype=torch.float64)
    s = a @ a.transpose(-1, -2) + K * torch.eye(K, dtype=torch.float64)
    if block_diagonal:
        m = torch.zeros(K, K, dtype=torch.float64)
        for h in range(H):
            m[h * d:(h + 1) * d, h * d:(h + 1) * d] = 1.0
        s = s * m
    return s.float()


def _mu(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, K, generator=g)


def _energy(monkeypatch, *, irrep_dims, sigma, honor):
    r"""coupling_energy with the fast route either honored or forced off at the geometry seam."""
    if not honor:
        monkeypatch.setattr(
            "vfe3.geometry.transport._reads_only_head_marginals",
            lambda *a, **k: False)
    return get_family("gaussian_full").coupling_energy(
        _mu(1), sigma, _mu(2), sigma, _omega(), irrep_dims=irrep_dims, **KW)


# -- (0) the promise actually reaches the geometry --------------------------------------------


def test_promise_propagates_from_coupling_energy_to_the_congruence(monkeypatch):
    r"""Without this the parity pins below would pass vacuously: they compare a fast route that
    never fired against a full route, and agree because both are the full route.

    Both seams are spied, because there are now two ways to honor the promise (audit 2026-08-06
    B2): ``_compact_factored_full_covariance(blocks_only=True)`` computes the head blocks and then
    scatters them into a dense zero tensor, while ``_compact_head_block_congruence`` -- reached
    through ``transport_covariance_head_blocks`` -- returns them in the block layout and skips the
    scatter. ``FullGaussian.coupling_energy`` takes the second; asserting on the FIRST alone is
    exactly how this test went stale.
    """
    import vfe3.geometry.transport as transport_mod

    fired: list[str] = []
    real_dense = transport_mod._compact_factored_full_covariance
    real_blocks = transport_mod._compact_head_block_congruence

    def dense_spy(factored, sigma, blocks_only=False):
        fired.append("scatter" if blocks_only else "full")
        return real_dense(factored, sigma, blocks_only=blocks_only)

    def blocks_spy(factored, sigma):
        fired.append("head_blocks")
        return real_blocks(factored, sigma)

    monkeypatch.setattr(transport_mod, "_compact_factored_full_covariance", dense_spy)
    monkeypatch.setattr(transport_mod, "_compact_head_block_congruence", blocks_spy)
    fam, sigma = get_family("gaussian_full"), _spd(3, block_diagonal=True)

    fam.coupling_energy(_mu(1), sigma, _mu(2), sigma, _omega(),
                        irrep_dims=[d] * H, **KW)
    assert fired == ["head_blocks"], (
        "per-head coupling_energy must reach the head-block congruence, and must NOT also build "
        "the dense congruence")

    fired.clear()
    fam.coupling_energy(_mu(1), sigma, _mu(2), sigma, _omega(), irrep_dims=None, **KW)
    assert fired == ["full"], "full-K coupling_energy reads off-blocks and must not fire it"

    fired.clear()
    fam.transport_dispersion(sigma, _omega(), diagonal_out=False)
    assert fired == ["full"], "a direct transport_dispersion caller makes no promise"


# -- (a) the promise is honored, on both covariance structures --------------------------------


@pytest.mark.parametrize("block_diagonal", [True, False])
def test_marginal_blocks_energy_matches_full_congruence(monkeypatch, block_diagonal):
    sigma = _spd(3, block_diagonal)
    with monkeypatch.context() as m:
        ref = _energy(m, irrep_dims=[d] * H, sigma=sigma, honor=False)
    fast = _energy(monkeypatch, irrep_dims=[d] * H, sigma=sigma, honor=True)
    # The head-diagonal output blocks are the SAME algebra (Omega_h Sigma^(h,h) Omega_h^T); any
    # difference is float64 reassociation, orders under the fp32 public cast of the energy.
    assert torch.allclose(fast, ref, atol=1e-6, rtol=1e-5)


def test_general_covariance_really_has_cross_head_mass():
    r"""Guards (a)'s second case: a block-diagonal sigma would make it vacuously true."""
    s = _spd(3, block_diagonal=False)
    off = s.clone()
    for h in range(H):
        off[..., h * d:(h + 1) * d, h * d:(h + 1) * d] = 0.0
    assert off.abs().max() > 1.0


# -- (b) the fast route is not inert ----------------------------------------------------------


def test_cross_head_blocks_are_dropped_and_head_blocks_kept():
    om, sigma = _omega(), _spd(3, block_diagonal=False).double()
    ref = _compact_factored_full_covariance(om, sigma, blocks_only=False)
    fast = _compact_factored_full_covariance(om, sigma, blocks_only=True)
    head = torch.zeros(K, K, dtype=torch.bool)
    for h in range(H):
        head[h * d:(h + 1) * d, h * d:(h + 1) * d] = True

    assert ref[..., ~head].abs().max() > 1.0        # the full route populates the off-blocks
    assert fast[..., ~head].abs().max() == 0.0      # the fast route leaves them exactly zero
    assert torch.allclose(fast[..., head], ref[..., head], atol=1e-9, rtol=1e-9)


# -- (c) / (d) misaligned and unpartitioned reads fail closed ---------------------------------


def test_misaligned_partition_falls_back_to_full_congruence(monkeypatch):
    r"""[3, 7] against two 5-wide heads: the second slice straddles the boundary at 5 and DOES
    read the cross-head blocks, so the fast route must not fire."""
    sigma = _spd(3, block_diagonal=False)
    with monkeypatch.context() as m:
        ref = _energy(m, irrep_dims=[3, 7], sigma=sigma, honor=False)
    fast = _energy(monkeypatch, irrep_dims=[3, 7], sigma=sigma, honor=True)
    assert torch.equal(fast, ref)                   # byte-identical: same code path


@pytest.mark.parametrize("blocks", [None, [d] * H, [3, 7], [K], [d] * (H + 1)])
def test_predicate_admits_only_the_transports_own_partition(blocks):
    expected = blocks == [d] * H
    assert _reads_only_head_marginals(_omega(), blocks) is expected


# -- (e) direct callers keep the full congruence ----------------------------------------------


def test_direct_transport_dispersion_caller_keeps_every_off_block():
    r"""metrics.gauge_equivariance_residual and the model sandwich diagnostic call
    transport_dispersion WITHOUT the promise; the off-blocks must survive for them."""
    om, sigma = _omega(), _spd(3, block_diagonal=False)
    head = torch.zeros(K, K, dtype=torch.bool)
    for h in range(H):
        head[h * d:(h + 1) * d, h * d:(h + 1) * d] = True

    default = get_family("gaussian_full").transport_dispersion(sigma, om, diagonal_out=False)
    assert default[..., ~head].abs().max() > 1.0

    # ... and the geometry seam's own default is equally conservative.
    assert transport_covariance(om, sigma, diagonal_out=False)[..., ~head].abs().max() > 1.0
