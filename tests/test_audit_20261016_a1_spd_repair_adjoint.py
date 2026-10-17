r"""Audit 2026-10-16 F39 and F99: the SPD repair and the spectral maps differentiate exactly.

F39. ``_certify_public_spd`` rebuilds a failing row as :math:`V\,\mathrm{clip}(\Lambda; \epsilon + c, \sigma_{\max} - c)\,V^T`.
Differentiating that rebuild through a gap-damped eigendecomposition zeroed the in-cluster term of the adjoint,
so inside a degenerate unclipped eigen-cluster the gradient disagreed with finite differences (autograd 1.16
against 2.0 on the dossier's probe). The rebuild's adjoint is the Daleckii-Krein (Loewner) form
:math:`V\,(D \circ V^T \bar G V)\,V^T` with :math:`D_{ij} = (f(\lambda_i) - f(\lambda_j))/(\lambda_i - \lambda_j)` and the
repeated-eigenvalue limit :math:`f'(\lambda)`, which is exact at a degenerate spectrum.

F99. The two spectral-map Functions call the plain ``safe_eigh`` in their forwards; their backwards were already
the exact Loewner adjoint, so the damping and its gap-eps reduction were dead work.
"""
import torch

import vfe3.geometry.retraction as retraction

F64 = torch.float64


def _degenerate_cluster(seed: int = 0):
    r"""Sigma = Q diag(5e-7, 5, 5, 5) Q^T (floor-repaired row) and an in-cluster direction E = q1 q2^T + q2 q1^T."""
    generator = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=F64))
    lam = torch.tensor([5e-7, 5.0, 5.0, 5.0], dtype=F64)
    sigma = (Q * lam) @ Q.T
    q1, q2 = Q[:, 1], Q[:, 2]
    E = torch.outer(q1, q2) + torch.outer(q2, q1)
    return sigma, E


def _readout(t: torch.Tensor, sigma: torch.Tensor, E: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    repaired = retraction._certify_public_spd((sigma + t * E)[None], eps=1e-6, sigma_max=1e6)[0]
    return (repaired * W).sum()


def test_repair_gradient_inside_a_degenerate_unclipped_cluster_matches_finite_differences() -> None:
    sigma, E = _degenerate_cluster()
    retraction.reset_public_spd_repair_elements()
    t = torch.zeros((), dtype=F64, requires_grad=True)
    _readout(t, sigma, E, E).backward()
    assert retraction.public_spd_repair_elements() == 1                      # the repair branch ran
    h = 1e-6
    fd = (_readout(torch.tensor(h, dtype=F64), sigma, E, E)
          - _readout(torch.tensor(-h, dtype=F64), sigma, E, E)) / (2.0 * h)
    assert abs(float(fd) - 2.0) < 1e-6                                       # the Loewner value, 2 <E, E>/2
    assert abs(float(t.grad) - float(fd)) < 1e-6


def test_repair_gradcheck_on_a_floor_and_ceiling_row() -> None:
    r"""Full-matrix gradcheck on a separated spectrum with one eigenvalue below the floor and one above the ceiling."""
    generator = torch.Generator().manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=F64))
    lam = torch.tensor([-0.4, 0.8, 1.7, 4.5], dtype=F64)
    matrix = ((Q * lam) @ Q.T).requires_grad_(True)

    def repaired(m: torch.Tensor) -> torch.Tensor:
        return retraction._certify_public_spd(0.5 * (m + m.T)[None], eps=1e-3, sigma_max=3.0)[0]

    assert torch.autograd.gradcheck(repaired, (matrix,), eps=1e-7, atol=1e-6, rtol=1e-5)


def test_repair_gradient_through_the_clearance_is_kept() -> None:
    r"""Both eigenvalues of diag(-2, 0) sit below eps = 1; the trace then moves only through the clearance."""
    boundary = torch.diag(torch.tensor([-2.0, 0.0], dtype=F64)).requires_grad_(True)
    out = retraction._certify_public_spd(boundary, eps=1.0, sigma_max=100.0)
    out.trace().backward()
    # c = factor 8 K eps_64 max|A_ij| on the rung that certifies, so d(2 (eps + c))/dA_00 = -2 factor 16 eps_64.
    rung_gradients = [-2.0 * factor * 16.0 * torch.finfo(F64).eps for factor in (2.0 ** -10, 2.0 ** -6, 2.0 ** -3, 1.0)]
    assert any(abs(float(boundary.grad[0, 0]) - value) <= 1e-12 * abs(value) for value in rung_gradients)
    assert float(boundary.grad[0, 1]) == float(boundary.grad[1, 0]) == float(boundary.grad[1, 1]) == 0.0


def test_spectral_maps_skip_the_damped_eigendecomposition(monkeypatch) -> None:
    calls = []
    original = retraction.safe_eigh

    def spy(matrix: torch.Tensor, **kwargs) -> tuple:
        calls.append(tuple(matrix.shape))
        return original(matrix, **kwargs)

    monkeypatch.setattr(retraction, "safe_eigh", spy)
    assert not hasattr(retraction, "_rel_gap_eps") and not hasattr(retraction, "_eigh_damped")
    A = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(1), dtype=F64)
    A = A @ A.transpose(-1, -2) + torch.eye(4, dtype=F64)
    A = 0.5 * (A + A.transpose(-1, -2))                                       # exactly symmetric: the map's own symmetrization is then exact
    root, inv_root = retraction._symmetric_spectral_map_pair(A, "sqrt_floor", "inv_sqrt_floor", lower=1e-6)
    exponential = retraction._symmetric_spectral_map(A, "exp_bounded", lower=-50.0, upper=50.0)
    reference = torch.linalg.eigh(A)
    assert calls == [(3, 4, 4), (3, 4, 4)]                                    # one eigh per Function, chunk-safe
    V, w = reference.eigenvectors, reference.eigenvalues
    torch.testing.assert_close(root, (V * w.sqrt().unsqueeze(-2)) @ V.transpose(-1, -2), rtol=0, atol=0)
    torch.testing.assert_close(exponential, (V * w.exp().unsqueeze(-2)) @ V.transpose(-1, -2), rtol=0, atol=0)
    torch.testing.assert_close(inv_root @ root, torch.eye(4, dtype=F64).expand(3, 4, 4), rtol=0, atol=1e-12)


def test_spectral_pair_adjoint_matches_the_stock_eigh_backward_on_a_separated_spectrum() -> None:
    r"""Independent oracle for the Loewner adjoint (replaces the retired damped-eigh comparison)."""
    generator = torch.Generator().manual_seed(0)
    A = torch.randn(3, 5, 5, generator=generator, dtype=F64)
    A = A @ A.transpose(-1, -2) + torch.diag_embed(torch.arange(1.0, 6.0, dtype=F64).expand(3, 5))
    A = 0.5 * (A + A.transpose(-1, -2))
    W1 = torch.randn(3, 5, 5, generator=generator, dtype=F64)
    W2 = torch.randn(3, 5, 5, generator=generator, dtype=F64)
    a1 = A.clone().requires_grad_(True)
    root, inv_root = retraction._symmetric_spectral_map_pair(a1, "sqrt_floor", "inv_sqrt_floor", lower=1e-6)
    ((root * W1).sum() + (inv_root * W2).sum()).backward()
    a2 = A.clone().requires_grad_(True)
    w, V = torch.linalg.eigh(0.5 * (a2 + a2.transpose(-1, -2)))
    stock_root = (V * w.sqrt().unsqueeze(-2)) @ V.transpose(-1, -2)
    stock_inv = (V * w.rsqrt().unsqueeze(-2)) @ V.transpose(-1, -2)
    ((stock_root * W1).sum() + (stock_inv * W2).sum()).backward()
    torch.testing.assert_close(a1.grad, a2.grad, rtol=1e-9, atol=1e-11)


def test_retract_spd_full_gradcheck_at_the_isotropic_init() -> None:
    r"""At Sigma = I (fully degenerate) the exact adjoint is finite and matches finite differences."""
    delta = 0.3 * torch.randn(4, 4, generator=torch.Generator().manual_seed(2), dtype=F64)
    delta = 0.5 * (delta + delta.T)
    sigma = torch.eye(4, dtype=F64).requires_grad_(True)

    def retracted(s: torch.Tensor) -> torch.Tensor:
        return retraction.retract_spd_full(0.5 * (s + s.T)[None], delta[None], eps=1e-6, sigma_max=10.0)[0]

    assert torch.autograd.gradcheck(retracted, (sigma,), eps=1e-7, atol=1e-6, rtol=1e-5)


def test_retraction_gradient_on_a_ceiling_cluster_matches_finite_differences() -> None:
    r"""Two eigenvalues projected onto sigma_max form a represented cluster that the strict ceiling certificate repairs.

    The damped eigendecomposition gave a directional derivative of -0.12002 on this row against a central
    difference of -0.11953; the exact adjoint matches it.
    """
    generator = torch.Generator().manual_seed(0)
    A = torch.randn(6, 4, 4, generator=generator, dtype=F64)
    sigma = (A @ A.transpose(-1, -2) / 4 + 0.3 * torch.eye(4, dtype=F64))[3]
    D = torch.randn(6, 4, 4, generator=generator, dtype=F64)
    D = (0.5 * (D + D.transpose(-1, -2)))[3]
    W = torch.randn(6, 4, 4, generator=generator, dtype=F64)[3]
    E = torch.randn(4, 4, generator=generator, dtype=F64)
    E = 0.5 * (E + E.T)

    def readout(s: torch.Tensor) -> torch.Tensor:
        s = 0.5 * (s + s.T)
        out = retraction.retract_spd_full(s[None], D[None], step_size=0.7, trust_region=2.0, eps=1e-6, sigma_max=3.0)
        return (out[0] * W).sum()

    retraction.reset_public_spd_repair_elements()
    s = sigma.clone().requires_grad_(True)
    readout(s).backward()
    assert retraction.public_spd_ceiling_contact_elements() == 1
    h = 1e-6
    fd = (readout(sigma + h * E) - readout(sigma - h * E)) / (2.0 * h)
    assert abs(float((s.grad * E).sum()) - float(fd)) < 1e-7 * abs(float(fd))
