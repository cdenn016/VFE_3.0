r"""Exact-congruence Gaussian family: registration, exactness, route agreement, and gauge liveness.

Every model here is TINY (K < 6) and CPU-bound per CLAUDE.md.
"""

import pytest
import torch

from vfe3.families import get_family
from vfe3.families.base import BeliefParams
from vfe3.families.exact_congruence import ExactCongruenceDiagonalGaussian
from vfe3.families.gaussian import DiagonalGaussian
from vfe3.geometry.lie_ops import _from_equal_diag_blocks
from vfe3.geometry.transport import CompactFactoredTransport, FactoredTransport


def _dense_kl(
    mu_q:    torch.Tensor,             # (K,) query mean
    sigma_q: torch.Tensor,             # (K, K) query covariance
    mu_p:    torch.Tensor,             # (K,) prior mean
    sigma_p: torch.Tensor,             # (K, K) prior covariance
) -> torch.Tensor:                     # () KL(q || p)
    r"""Dense reference KL for two nondegenerate Gaussians (float64)."""
    k = mu_q.shape[-1]
    precision_p = torch.linalg.inv(sigma_p)
    delta = (mu_p - mu_q).unsqueeze(-1)
    return 0.5 * (
        torch.trace(precision_p @ sigma_q)
        + (delta.T @ precision_p @ delta).squeeze()
        - k
        + torch.logdet(sigma_p)
        - torch.logdet(sigma_q)
    )


def _cocycle(
    n_tokens:  int,
    n_blocks:  int,
    block_dim: int,

    *,
    scale:     float = 0.6,
) -> tuple:
    r"""A certified flat cocycle in all three representations plus its dense pairwise operator."""
    blocks = torch.linalg.matrix_exp(
        torch.randn(n_tokens, n_blocks, block_dim, block_dim, dtype=torch.float64) * scale)
    inv_blocks = torch.linalg.inv(blocks)
    K = n_blocks * block_dim
    vertex = _from_equal_diag_blocks(blocks, K)                       # (N, K, K) U_i
    vertex_inv = _from_equal_diag_blocks(inv_blocks, K)               # (N, K, K) U_j^{-1}
    factored = FactoredTransport(
        exp_phi=vertex, exp_neg_phi=vertex_inv,
        irrep_dims=[block_dim] * n_blocks, same_frame_flat_cocycle=True)
    compact = CompactFactoredTransport(blocks, inv_blocks, K, same_frame_flat_cocycle=True)
    dense = torch.einsum("ikl,jlm->ijkm", vertex, vertex_inv)         # (N, N, K, K)
    return factored, compact, dense


def test_family_is_registered_and_diagonal() -> None:
    r"""The family is reachable by its config key and declares a diagonal dispersion."""
    family = get_family("gaussian_diagonal_exact")
    assert family is ExactCongruenceDiagonalGaussian
    assert family.cov_kind == "diagonal"
    assert issubclass(family, DiagonalGaussian)
    assert issubclass(family, BeliefParams)


def test_default_coupling_seam_matches_the_inlined_transport_triple() -> None:
    r"""The base seam is byte-identical to the transport + pairwise_energy composition it replaced."""
    from vfe3.free_energy import pairwise_energy
    from vfe3.geometry.transport import transport_covariance, transport_mean

    torch.manual_seed(0)
    n, k = 4, 4
    omega = torch.linalg.matrix_exp(torch.randn(n, n, k, k, dtype=torch.float64) * 0.4)
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.5

    mu_t = transport_mean(omega, mu)
    sigma_t = transport_covariance(omega, sigma, diagonal_out=True)
    expected = pairwise_energy(
        DiagonalGaussian(mu, sigma),
        DiagonalGaussian.from_transported(mu_t, sigma_t, sigma),
        alpha=1.0, kl_max=100.0, eps=1e-6, divergence_family="renyi", irrep_dims=[2, 2],
    )
    actual = DiagonalGaussian.coupling_energy(
        mu, sigma, mu, sigma, omega,
        alpha=1.0, kl_max=100.0, eps=1e-6, divergence_family="renyi", irrep_dims=[2, 2],
        diagonal_out=True,
    )
    assert torch.equal(actual, expected)


def test_exact_energy_equals_the_dense_congruence_kl() -> None:
    r"""The exact energy IS KL(q_i || Omega_ij q_j) for the full pushforward covariance.

    The transport here is a general invertible operator, NOT a cocycle, so this pins the
    reference route on its own terms.
    """
    torch.manual_seed(0)
    n, k = 4, 4
    omega = torch.linalg.matrix_exp(torch.randn(n, n, k, k, dtype=torch.float64) * 0.5)
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4

    energy = ExactCongruenceDiagonalGaussian.coupling_energy(
        mu, sigma, mu, sigma, omega, kl_max=1e9, eps=1e-12)
    assert energy.shape == (n, n)
    for i in range(n):
        for j in range(n):
            transport = omega[i, j]
            reference = _dense_kl(
                mu[i], torch.diag(sigma[i]),
                transport @ mu[j], transport @ torch.diag(sigma[j]) @ transport.T,
            )
            assert energy[i, j].item() == pytest.approx(reference.item(), rel=1e-9, abs=1e-9)


def test_truncated_energy_is_not_the_congruence_kl() -> None:
    r"""The default family's energy differs from the exact one, so the toggle is not cosmetic.

    Guards against a change that silently makes the two paths identical (which would mean the
    exact route had stopped inverting the transport).
    """
    torch.manual_seed(0)
    n, k = 4, 4
    omega = torch.linalg.matrix_exp(torch.randn(n, n, k, k, dtype=torch.float64) * 0.5)
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4

    kwargs = dict(kl_max=1e9, eps=1e-12)
    exact = ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, omega, **kwargs)
    truncated = DiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, omega, **kwargs)
    assert (exact - truncated).abs().max().item() > 1.0


def test_factored_and_compact_routes_match_the_dense_reference() -> None:
    r"""The cocycle transpose fast route reproduces the explicit-inverse reference route.

    For Omega_ij = U_i U_j^{-1} the inverse is the pair transpose Omega_ji, so the fast route may
    reuse the forward transport machinery; this pins that identity end to end.
    """
    torch.manual_seed(0)
    n, n_blocks, block_dim = 5, 2, 3
    factored, compact, dense = _cocycle(n, n_blocks, block_dim)
    k = n_blocks * block_dim
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4
    kwargs = dict(kl_max=1e9, eps=1e-12, irrep_dims=[block_dim] * n_blocks)

    reference = ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, dense, **kwargs)
    for transport in (factored, compact):
        fast = ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, transport, **kwargs)
        assert fast.shape == (n_blocks, n, n)
        assert (fast - reference).abs().max().item() == pytest.approx(0.0, abs=1e-9)


def test_per_head_energy_is_the_dense_per_block_congruence_kl() -> None:
    r"""Each head's energy is the exact congruence KL over that irrep block's coordinates."""
    torch.manual_seed(0)
    n, n_blocks, block_dim = 4, 2, 2
    factored, _, dense = _cocycle(n, n_blocks, block_dim)
    k = n_blocks * block_dim
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4

    energy = ExactCongruenceDiagonalGaussian.coupling_energy(
        mu, sigma, mu, sigma, factored,
        kl_max=1e9, eps=1e-12, irrep_dims=[block_dim] * n_blocks)
    for h in range(n_blocks):
        block = slice(h * block_dim, (h + 1) * block_dim)
        for i in range(n):
            for j in range(n):
                transport = dense[i, j][block, block]
                reference = _dense_kl(
                    mu[i][block], torch.diag(sigma[i][block]),
                    transport @ mu[j][block],
                    transport @ torch.diag(sigma[j][block]) @ transport.T,
                )
                assert energy[h, i, j].item() == pytest.approx(reference.item(), rel=1e-8, abs=1e-8)


def test_self_pairs_are_structurally_zero_under_a_cocycle() -> None:
    r"""Omega_ii = I, so the exact self-pair energy is EXACTLY zero, as the truncated form is.

    The saturation mask and the softmax both depend on that structural zero.
    """
    torch.manual_seed(0)
    n, n_blocks, block_dim = 5, 2, 2
    factored, compact, _ = _cocycle(n, n_blocks, block_dim)
    k = n_blocks * block_dim
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4

    for transport in (factored, compact):
        energy = ExactCongruenceDiagonalGaussian.coupling_energy(
            mu, sigma, mu, sigma, transport,
            kl_max=1e9, eps=1e-12, irrep_dims=[block_dim] * n_blocks)
        assert energy.diagonal(dim1=-2, dim2=-1).abs().max().item() == 0.0


def test_identity_transport_collapses_onto_the_default_family() -> None:
    r"""With no gauge (Omega = I) there is nothing to truncate, so both families agree exactly.

    This is the gauge_transport='off' limit: it pins that the exact route adds no offset of its
    own (the determinant correction vanishes and the pullback is the identity).
    """
    torch.manual_seed(0)
    n, n_blocks, block_dim = 4, 2, 2
    k = n_blocks * block_dim
    identity = torch.eye(k, dtype=torch.float64).expand(n, k, k).contiguous()
    factored = FactoredTransport(
        exp_phi=identity, exp_neg_phi=identity.clone(),
        irrep_dims=[block_dim] * n_blocks, same_frame_flat_cocycle=True)
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4
    kwargs = dict(kl_max=1e9, eps=1e-12, irrep_dims=[block_dim] * n_blocks)

    exact = ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, factored, **kwargs)
    truncated = DiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, factored, **kwargs)
    assert (exact - truncated).abs().max().item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("route", ["factored", "compact", "dense"])
def test_reduced_precision_transport_is_promoted_for_the_determinant(route: str) -> None:
    r"""bfloat16 beliefs and transport (the AMP path) produce a finite fp32 energy.

    ``torch.linalg`` rejects bfloat16 outright, so the log-determinant and the dense inverse are
    promoted to fp32; the determinant enters the energy ADDITIVELY, so a bfloat16 evaluation would
    also be numerically wrong rather than merely imprecise. Regression pin for the AMP crash in
    the model channel's gamma coupling, which runs under the caller's autocast context.
    """
    torch.manual_seed(0)
    n, n_blocks, block_dim = 4, 2, 2
    k = n_blocks * block_dim
    blocks = torch.linalg.matrix_exp(torch.randn(n, n_blocks, block_dim, block_dim) * 0.4)
    inv_blocks = torch.linalg.inv(blocks)
    vertex = _from_equal_diag_blocks(blocks, k).bfloat16()
    vertex_inv = _from_equal_diag_blocks(inv_blocks, k).bfloat16()
    transports = {
        "factored": FactoredTransport(
            exp_phi=vertex, exp_neg_phi=vertex_inv,
            irrep_dims=[block_dim] * n_blocks, same_frame_flat_cocycle=True),
        "compact": CompactFactoredTransport(
            blocks.bfloat16(), inv_blocks.bfloat16(), k, same_frame_flat_cocycle=True),
        "dense": torch.einsum("ikl,jlm->ijkm", vertex, vertex_inv),
    }
    mu = torch.randn(n, k).bfloat16()
    sigma = (torch.rand(n, k) + 0.4).bfloat16()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        energy = ExactCongruenceDiagonalGaussian.coupling_energy(
            mu, sigma, mu, sigma, transports[route],
            kl_max=1e9, eps=1e-6, irrep_dims=[block_dim] * n_blocks)
    assert energy.dtype == torch.float32
    assert torch.isfinite(energy).all()


def test_noncanonical_divergence_and_uninvertible_transport_fail_closed() -> None:
    r"""The family raises rather than silently returning the truncated energy.

    The pullback identity is specific to KL, and the transpose fast route needs a certified
    cocycle; both are hard errors so no config can quietly get the default energy back.
    """
    torch.manual_seed(0)
    n, k = 4, 4
    omega = torch.linalg.matrix_exp(torch.randn(n, n, k, k, dtype=torch.float64) * 0.3)
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.4

    with pytest.raises(ValueError, match="KL only"):
        ExactCongruenceDiagonalGaussian.coupling_energy(
            mu, sigma, mu, sigma, omega, alpha=0.5)
    with pytest.raises(ValueError, match="KL only"):
        ExactCongruenceDiagonalGaussian.coupling_energy(
            mu, sigma, mu, sigma, omega, divergence_family="jeffreys")

    uncertified = FactoredTransport(
        exp_phi=torch.linalg.matrix_exp(torch.randn(n, k, k, dtype=torch.float64) * 0.3),
        exp_neg_phi=torch.linalg.matrix_exp(torch.randn(n, k, k, dtype=torch.float64) * 0.3),
        irrep_dims=[2, 2], same_frame_flat_cocycle=False)
    with pytest.raises(TypeError, match="certified same-frame flat cocycle"):
        ExactCongruenceDiagonalGaussian.coupling_energy(
            mu, sigma, mu, sigma, uncertified, irrep_dims=[2, 2])


def test_structural_helpers_preserve_the_subclass() -> None:
    r"""block/broadcast_over_keys/stack keep the exact-congruence family, not the parent."""
    torch.manual_seed(0)
    n, k = 2, 4
    params = ExactCongruenceDiagonalGaussian(
        torch.randn(n, k, dtype=torch.float64),
        torch.rand(n, k, dtype=torch.float64) + 0.5,
    )
    assert isinstance(params.block(0, 2), ExactCongruenceDiagonalGaussian)
    assert isinstance(params.broadcast_over_keys(), ExactCongruenceDiagonalGaussian)
    assert isinstance(
        ExactCongruenceDiagonalGaussian.stack([params, params]), ExactCongruenceDiagonalGaussian)
    assert params.block(0, 2).coordinate_dim() == 2


def test_gauge_table_keeps_its_gradient_under_the_exact_family() -> None:
    r"""End-to-end pin of the point of this family: phi stays live in the belief coupling.

    Unlike ``gaussian_frame_diagonal`` (where the relative frame cancels and phi_embed receives no
    gradient at all), the exact congruence keeps the gauge in the energy. The oracle route needs
    ``oracle_unroll_grad=True`` for the unrolled E-step tangent to reach the prior tables, so both
    settings are pinned: the toggle is the difference between a live and a frozen gauge here.
    """
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    grads = {}
    for family in ("gaussian_diagonal_exact", "gaussian_frame_diagonal"):
        for unroll in (False, True):
            cfg = VFE3Config(
                embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
                n_layers=1, n_e_steps=2, family=family, e_step_update="gradient",
                lambda_alpha_mode="constant", max_steps=1, oracle_unroll_grad=unroll,
            )
            torch.manual_seed(0)
            model = VFEModel(cfg)
            token_ids = torch.randint(0, 11, (2, 6))
            targets = torch.randint(0, 11, (2, 6))
            _, loss, _ = model(token_ids, targets)
            loss.backward()
            grads[(family, unroll)] = model.prior_bank.phi_embed.grad

    assert grads[("gaussian_diagonal_exact", True)] is not None
    assert grads[("gaussian_diagonal_exact", True)].norm() > 0.0
    # The detached oracle tangent freezes phi; this is the config footgun the toggle docs name.
    assert grads[("gaussian_diagonal_exact", False)] is None
    # The frame family cancels the gauge out of the energy, so no estimator revives it.
    assert grads[("gaussian_frame_diagonal", True)] is None
    assert grads[("gaussian_frame_diagonal", False)] is None
