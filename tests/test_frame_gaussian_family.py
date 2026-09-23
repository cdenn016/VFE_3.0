r"""Frame-intrinsic Gaussian family: registration, transport identities, and the exact-KL claim.

Every model here is TINY (K < 6) and CPU-bound per CLAUDE.md.
"""

import math

import pytest
import torch

from vfe3.families import get_family
from vfe3.families.base import BeliefParams
from vfe3.families.frame_gaussian import FrameDiagonalGaussian
from vfe3.families.gaussian import DiagonalGaussian


def _gaussian_kl(
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


def test_family_is_registered_and_diagonal() -> None:
    r"""The family is reachable by its config key and declares a diagonal dispersion."""
    family = get_family("gaussian_frame_diagonal")
    assert family is FrameDiagonalGaussian
    assert family.cov_kind == "diagonal"
    assert issubclass(family, BeliefParams)


def test_default_family_location_transport_is_unchanged() -> None:
    r"""The new seam is byte-identical to transport_mean for a family that does not override it."""
    from vfe3.geometry.transport import transport_mean

    torch.manual_seed(0)
    n, k = 3, 4
    omega = torch.randn(n, n, k, k, dtype=torch.float64)
    mu = torch.randn(n, k, dtype=torch.float64)
    assert torch.equal(DiagonalGaussian.transport_location(mu, omega), transport_mean(omega, mu))


def _certified_flat_cocycle(n: int, k: int) -> object:
    r"""A transport container that certifies Omega_ij = U_i U_j^{-1}, the family's precondition."""
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import build_factored_transport

    group = get_group("block_glk")(k, 2)                        # K=k, 2 equal blocks
    phi = 0.1 * torch.randn(n, group.generators.shape[0])   # group generators are fp32
    return build_factored_transport(phi, group)


def test_frame_transports_are_query_broadcasts() -> None:
    r"""Both frame-intrinsic transports drop the query index: entry [i, j] equals the input at j."""
    torch.manual_seed(0)
    n, k = 3, 4
    omega = _certified_flat_cocycle(n, k)                          # values unused; certification is not
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.5

    mu_t = FrameDiagonalGaussian.transport_location(mu, omega)
    sigma_t = FrameDiagonalGaussian.transport_dispersion(sigma, omega)
    assert mu_t.shape == (n, n, k)
    assert sigma_t.shape == (n, n, k)
    for i in range(n):
        assert torch.equal(mu_t[i], mu)
        assert torch.equal(sigma_t[i], sigma)


def test_frame_energy_equals_dense_full_covariance_kl() -> None:
    r"""The diagonal energy on frame coordinates IS the dense congruence KL.

    Builds Sigma_i = U_i diag(sigma_i) U_i^T, transports the sender by the Regime-I coboundary
    Omega = U_i U_j^{-1}, and checks the dense KL against the family's diagonal closed form on
    the frame-intrinsic coordinates a = U^{-1} mu.
    """
    torch.manual_seed(0)
    k = 4
    for scale in (0.3, 1.0, 2.0):
        frame_i = torch.linalg.matrix_exp(torch.randn(k, k, dtype=torch.float64) * scale / k ** 0.5)
        frame_j = torch.linalg.matrix_exp(torch.randn(k, k, dtype=torch.float64) * scale / k ** 0.5)
        sigma_i = torch.rand(k, dtype=torch.float64) + 0.2
        sigma_j = torch.rand(k, dtype=torch.float64) + 0.2
        mu_i = torch.randn(k, dtype=torch.float64)
        mu_j = torch.randn(k, dtype=torch.float64)

        cov_i = frame_i @ torch.diag(sigma_i) @ frame_i.T
        cov_j = frame_j @ torch.diag(sigma_j) @ frame_j.T
        omega = frame_i @ torch.linalg.inv(frame_j)

        dense = _gaussian_kl(mu_i, cov_i, omega @ mu_j, omega @ cov_j @ omega.T)

        a_i = torch.linalg.solve(frame_i, mu_i)
        a_j = torch.linalg.solve(frame_j, mu_j)
        frame_kl = FrameDiagonalGaussian(a_i, sigma_i).renyi_closed_form(
            FrameDiagonalGaussian(a_j, sigma_j), alpha=1.0, kl_max=1e9, eps=1e-12,
        )
        assert frame_kl.item() == pytest.approx(dense.item(), rel=1e-9, abs=1e-9)


def test_frame_energy_is_independent_of_the_relative_frame() -> None:
    r"""Omega cancels: re-gauging both agents leaves the frame-intrinsic energy unchanged.

    This pins the documented CONSEQUENCE of the family (the gauge leaves the pair energy), so a
    future change that silently reintroduces frame dependence fails here.
    """
    torch.manual_seed(0)
    k = 3
    a_i = torch.randn(k, dtype=torch.float64)
    a_j = torch.randn(k, dtype=torch.float64)
    sigma_i = torch.rand(k, dtype=torch.float64) + 0.2
    sigma_j = torch.rand(k, dtype=torch.float64) + 0.2

    reference = FrameDiagonalGaussian(a_i, sigma_i).renyi_closed_form(
        FrameDiagonalGaussian(a_j, sigma_j), alpha=1.0, kl_max=1e9, eps=1e-12,
    )
    for _ in range(3):
        omega = torch.linalg.matrix_exp(torch.randn(k, k, dtype=torch.float64) * 0.7)
        energy = FrameDiagonalGaussian(a_i, sigma_i).renyi_closed_form(
            FrameDiagonalGaussian(a_j, sigma_j), alpha=1.0, kl_max=1e9, eps=1e-12,
        )
        assert torch.equal(energy, reference)
        assert omega.shape == (k, k)                      # the frame exists but never enters


def test_block_and_stack_preserve_the_subclass() -> None:
    r"""Structural helpers return the frame family, not the fixed-basis parent."""
    torch.manual_seed(0)
    n, k = 2, 4
    params = FrameDiagonalGaussian(
        torch.randn(n, k, dtype=torch.float64),
        torch.rand(n, k, dtype=torch.float64) + 0.5,
    )
    assert isinstance(params.block(0, 2), FrameDiagonalGaussian)
    assert isinstance(params.broadcast_over_keys(), FrameDiagonalGaussian)
    assert isinstance(FrameDiagonalGaussian.stack([params, params]), FrameDiagonalGaussian)
    assert params.block(0, 2).coordinate_dim() == 2


def test_frame_family_leaves_the_gauge_table_without_gradient() -> None:
    r"""End-to-end pin of the documented CONSEQUENCE: under this family phi is inert.

    Because the relative frame cancels from the pair energy and both transports are the identity,
    nothing on the belief path consumes ``phi_embed`` and it receives no gradient, while the
    fixed-basis family does. This is the ablation semantics the toggle exists to expose; if a
    future change reconnects phi, this test fails and the docstring must be revisited.
    """
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel

    grads = {}
    for family in ("gaussian_diagonal", "gaussian_frame_diagonal"):
        cfg = VFE3Config(
            embed_dim=4, n_heads=2, vocab_size=11, max_seq_len=6, batch_size=2,
            n_layers=1, n_e_steps=2, family=family, e_step_update="gradient",
            lambda_alpha_mode="constant", max_steps=1,
        )
        torch.manual_seed(0)
        model = VFEModel(cfg)
        token_ids = torch.randint(0, 11, (2, 6))
        targets = torch.randint(0, 11, (2, 6))
        _, loss, _ = model(token_ids, targets)
        loss.backward()
        grads[family] = model.prior_bank.phi_embed.grad

    assert grads["gaussian_diagonal"] is not None
    assert grads["gaussian_diagonal"].norm() > 0.0
    assert grads["gaussian_frame_diagonal"] is None


def test_entropy_matches_the_diagonal_parent() -> None:
    r"""Frame coordinates obey the same diagonal exponential-family algebra."""
    torch.manual_seed(0)
    k = 4
    mu = torch.randn(k, dtype=torch.float64)
    sigma = torch.rand(k, dtype=torch.float64) + 0.5
    expected = 0.5 * (torch.log(sigma) + math.log(2.0 * math.pi * math.e)).sum()
    assert FrameDiagonalGaussian(mu, sigma).entropy().item() == pytest.approx(expected.item())


def test_frame_family_refuses_a_transport_whose_frames_do_not_cancel() -> None:
    r"""Fail closed off the coboundary (audit 2026-07-26 A-01).

    The frame-intrinsic identity `U_i^-1 Omega_ij U_j = I` holds only for `Omega_ij = U_i U_j^-1`.
    Under a learned connection the edge factor sits between the frames and the residual is not the
    identity, so returning the flat answer silently discards the whole connection -- measured as a
    bit-identical pair energy and `connection_W.grad is None`. An uncertified transport must raise
    rather than answer.
    """
    torch.manual_seed(0)
    n, k = 3, 4
    dense = torch.randn(n, n, k, k, dtype=torch.float64)           # certifies nothing
    mu = torch.randn(n, k, dtype=torch.float64)
    sigma = torch.rand(n, k, dtype=torch.float64) + 0.5

    for seam, argument in ((FrameDiagonalGaussian.transport_location, mu),
                           (FrameDiagonalGaussian.transport_dispersion, sigma)):
        with pytest.raises(TypeError, match="same-frame flat cocycle"):
            seam(argument, dense)
