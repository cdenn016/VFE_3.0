r"""Belief-Fisher metric of the stored frame factors (derivation 2026-09-28, Track 2).

The frame factor :math:`U` of occurrence :math:`i` enters the reference belief
:math:`r_i = U_i^{-1} \cdot q_i = \mathcal N(\xi_i, C_i)`, :math:`\xi = U^{-1}\mu`,
:math:`C = U^{-1}\Sigma U^{-\top}`. Pulling the Gaussian Fisher-Rao metric back along
:math:`U \mapsto U e^{X}` gives

.. math::
    g_U(X, X) = \|Z\eta\|^2 + 2\|\mathrm{sym}\,Z\|_F^2, \qquad Z = L^{-1} X L,
    \quad \eta = L^{-1}\xi, \quad L L^\top = C,

whose kernel is the Lie algebra of the stabilizer of :math:`r`. The covariant completion
:math:`g + \lambda\|\mathrm{asym}\,Z\|_F^2` (the family :math:`(\lambda_s, \lambda_a) = (0, \lambda)`)
has the Frobenius-self-adjoint operator

.. math::
    \mathcal M(X) = C^{-1} X (c_1 C + \xi\xi^\top) + c_2 X^\top, \qquad
    c_1 = 1 + \lambda/2, \quad c_2 = 1 - \lambda/2,

a single Kronecker product at the default :math:`\lambda = 2`. For a fixed belief the metric is
right-invariant, so in the spatial generator :math:`W = \delta U\,U^{-1}` of a token-type factor it
is the same formula at the local table row :math:`(\mu_v, \mathrm{diag}\,s_v)`, independent of the
position. Positional factors see the batch column of middle-fiber beliefs
:math:`E_{v_b}^{-1} \cdot q_{v_b}` and are solved densely.

Every function here is pure torch, batched, and expects float64 inputs with autocast disabled.
Blocks are ``(..., d, d)`` head blocks; matrices are vectorized row-major,
:math:`\mathrm{vec}(A X B) = (A \otimes B^\top)\,\mathrm{vec}\,X`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class FisherRowStatistics:
    r"""Belief statistics that define the Fisher metric of ``R`` active rows of one frame table.

    ``kind='type'`` rows carry the layer-0 table row :math:`(\mu_v, s_v)` per head block;
    ``kind='pos'`` rows carry the pooled operator of their batch column. ``count`` is
    :math:`n_v` (type) or :math:`B_{\rm tot}` (pos) and ``total`` the scored-token count
    :math:`N`, so the count-normalized direction carries :math:`N/\mathrm{count}`.
    """
    kind:            str                                   # "type" | "pos"
    count:           torch.Tensor                          # (R,) float64
    total:           torch.Tensor                          # () float64 scored tokens
    local_mean:      Optional[torch.Tensor] = None         # (R, H, d) type only
    local_var:       Optional[torch.Tensor] = None         # (R, H, d) type only
    pooled_operator: Optional[torch.Tensor] = None         # (R, H, d*d, d*d) pos only


def completion_coefficients(rotation_weight: float) -> Tuple[float, float]:
    r"""Return :math:`(c_1, c_2) = (1 + \lambda/2, 1 - \lambda/2)` for :math:`(\lambda_s, \lambda_a) = (0, \lambda)`."""
    return 1.0 + 0.5 * rotation_weight, 1.0 - 0.5 * rotation_weight


def completed_fisher_operator(
    X:  torch.Tensor,                  # (..., d, d)
    xi: torch.Tensor,                  # (..., d) reference mean
    C:  torch.Tensor,                  # (..., d, d) reference covariance

    *,
    rotation_weight: float,
) -> torch.Tensor:                     # (..., d, d)
    r"""Apply :math:`\mathcal M(X) = C^{-1} X (c_1 C + \xi\xi^\top) + c_2 X^\top`."""
    c1, c2 = completion_coefficients(rotation_weight)
    N = c1 * C + xi[..., :, None] * xi[..., None, :]
    return torch.linalg.solve(C, X @ N) + c2 * X.transpose(-1, -2)


def fisher_spectrum_bounds(
    a: torch.Tensor,                   # (...) Mahalanobis norm mu^T Sigma^{-1} mu
    d: int,

    *,
    rotation_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:   # (...) lambda_min, (...) lambda_max
    r"""Extreme generalized eigenvalues of the completed metric relative to :math:`\mathcal P(X) = C^{-1} X C`.

    The spectrum is :math:`\{2 + a\}` (once), :math:`\{\tfrac12(2c_1 + a \pm \sqrt{a^2 + 4c_2^2})\}`
    (:math:`d - 1` times each), :math:`\{2\}` (:math:`d(d-1)/2` times) and :math:`\{\lambda\}`
    (:math:`(d-1)(d-2)/2` times). Only members with positive multiplicity enter the extremes. The
    maximum uses every member, which corrects the spec's ``cond_P`` for :math:`\lambda > 2`; the
    minimum also keeps the shear value 2, which the spec's ``min(lam, minus)`` omits and which is the
    minimum whenever :math:`\lambda > 2`.
    """
    c1, c2 = completion_coefficients(rotation_weight)
    root = torch.sqrt(a.square() + 4.0 * c2 * c2)
    members = [2.0 + a]
    if d >= 2:
        members += [0.5 * (2.0 * c1 + a - root), 0.5 * (2.0 * c1 + a + root),
                    torch.full_like(a, 2.0)]
    if d >= 3:
        members.append(torch.full_like(a, float(rotation_weight)))
    stacked = torch.stack(members)                                         # (m, ...)
    return stacked.amin(dim=0), stacked.amax(dim=0)


def fisher_type_solve(
    G_W:   torch.Tensor,               # (R, H, d, d) spatial covector
    mu:    torch.Tensor,               # (R, H, d) local table mean
    s:     torch.Tensor,               # (R, H, d) local table variance
    scale: torch.Tensor,               # (R,) count normalization N / n_v

    *,
    rotation_weight: float,
) -> torch.Tensor:                     # (R, H, d, d) spatial direction W
    r"""Solve :math:`\mathcal M(W) = \mathrm{scale}\cdot G_W` at the local row :math:`(\mu, S = \mathrm{diag}\,s)` in :math:`O(d^2)`.

    At :math:`\lambda = 2` the operator is the Kronecker product :math:`S^{-1} W (2S + \mu\mu^\top)`,
    and Sherman-Morrison gives

    .. math::
        W = \frac{\mathrm{scale}}{2}\Bigl(Q - \frac{(Q\mu)(S^{-1}\mu)^\top}{2 + a}\Bigr),
        \qquad Q_{jk} = \frac{s_j}{s_k}(G_W)_{jk}, \quad a = \mu^\top S^{-1}\mu.

    For general :math:`\lambda`, whiten with :math:`L = S^{1/2}`: :math:`R = S^{1/2} G_W S^{-1/2}`,
    :math:`\eta = S^{-1/2}\mu`, and solve :math:`c_1 Z + Z\eta\eta^\top + c_2 Z^\top = R` by the
    Woodbury identity with :math:`\mathcal L_0^{-1}(Y) = \mathrm{sym}\,Y/2 + \mathrm{asym}\,Y/\lambda`,
    :math:`\alpha = \tfrac12(\tfrac12 + \tfrac1\lambda)`, :math:`\beta = \tfrac12(\tfrac12 - \tfrac1\lambda)`:

    .. math::
        b = \mathcal L_0^{-1}(R)\eta, \qquad
        w = \frac{b - \beta\eta(\eta^\top b)/(1 + a(\alpha + \beta))}{1 + a\alpha}, \qquad
        Z = \mathcal L_0^{-1}(R) - \mathcal L_0^{-1}(w\eta^\top),

    then :math:`W = \mathrm{scale}\cdot S^{1/2} Z S^{-1/2}`. Both forms are basis-free and valid at
    :math:`\mu = 0`.
    """
    scale = scale[:, None, None, None]
    if rotation_weight == 2.0:
        Q = G_W * (s[..., :, None] / s[..., None, :])                      # (R, H, d, d)
        u = mu / s                                                         # (R, H, d) S^{-1} mu
        a = (mu * u).sum(dim=-1)                                           # (R, H)
        Q_mu = (Q @ mu[..., None])[..., 0]                                 # (R, H, d)
        correction = Q_mu[..., :, None] * u[..., None, :] / (2.0 + a)[..., None, None]
        return 0.5 * scale * (Q - correction)

    lam = float(rotation_weight)
    r = s.sqrt()
    ratio = r[..., :, None] / r[..., None, :]                              # (R, H, d, d) sqrt(s_j/s_k)
    R = G_W * ratio                                                        # whitened covector
    eta = mu / r                                                           # (R, H, d)
    a = eta.square().sum(dim=-1)[..., None]                                # (R, H, 1)

    def inverse_base(Y: torch.Tensor) -> torch.Tensor:
        Y_t = Y.transpose(-1, -2)
        return 0.25 * (Y + Y_t) + 0.5 * (Y - Y_t) / lam

    alpha = 0.5 * (0.5 + 1.0 / lam)
    beta = 0.5 * (0.5 - 1.0 / lam)
    Y0 = inverse_base(R)
    b = (Y0 @ eta[..., None])[..., 0]                                      # (R, H, d)
    eta_b = (eta * b).sum(dim=-1, keepdim=True)                            # (R, H, 1)
    w = (b - beta * eta * eta_b / (1.0 + a * (alpha + beta))) / (1.0 + a * alpha)
    Z = Y0 - inverse_base(w[..., :, None] * eta[..., None, :])
    return scale * Z * ratio


def middle_fiber_moments(
    X:  torch.Tensor,                  # (T, H, d, d) type chart blocks, E = exp(X)
    mu: torch.Tensor,                  # (T, H, d) local table mean
    s:  torch.Tensor,                  # (T, H, d) local table variance
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # xi0 (T,H,d), C0 (T,H,d,d), C0^{-1}
    r"""Moments of :math:`r_v^0 = E_v^{-1}\cdot q_v`: :math:`\xi^0 = E^{-1}\mu`, :math:`C^0 = E^{-1} S E^{-\top}`, :math:`(C^0)^{-1} = E^\top S^{-1} E`."""
    E = torch.linalg.matrix_exp(X)
    E_inv = torch.linalg.matrix_exp(-X)
    xi0 = (E_inv @ mu[..., None])[..., 0]
    C0 = (E_inv * s[..., None, :]) @ E_inv.transpose(-1, -2)
    C0_inv = (E.transpose(-1, -2) / s[..., None, :]) @ E
    return (xi0, 0.5 * (C0 + C0.transpose(-1, -2)),
            0.5 * (C0_inv + C0_inv.transpose(-1, -2)))


def pooled_pos_operator(
    xi0:    torch.Tensor,              # (R, B, H, d) middle-fiber means of the batch column
    C0:     torch.Tensor,              # (R, B, H, d, d)
    C0_inv: torch.Tensor,              # (R, B, H, d, d)

    *,
    rotation_weight: float,
) -> torch.Tensor:                     # (R, H, d*d, d*d)
    r"""Pooled positional operator :math:`\mathcal M_p = \frac1B\sum_b C_b^{-1}\otimes N_b + c_2 K`.

    :math:`N_b = c_1 C_b + \xi_b\xi_b^\top` and :math:`K` is the row-major commutation matrix
    :math:`K_{(ij),(kl)} = \delta_{il}\delta_{jk}` (:math:`K\,\mathrm{vec}\,X = \mathrm{vec}\,X^\top`).
    The operator is not a Kronecker product (its rearrangement rank is the number of distinct
    Gaussians in the column), so it is formed densely.
    """
    c1, c2 = completion_coefficients(rotation_weight)
    rows, batch, n_blocks, d = xi0.shape
    N = c1 * C0 + xi0[..., :, None] * xi0[..., None, :]                    # (R, B, H, d, d)
    M = torch.einsum("rbhik,rbhjl->rhijkl", C0_inv, N) / batch             # (R, H, d, d, d, d)
    M = M.reshape(rows, n_blocks, d * d, d * d)
    if c2 != 0.0:
        eye = torch.eye(d, dtype=M.dtype, device=M.device)
        commutation = torch.einsum("il,jk->ijkl", eye, eye).reshape(d * d, d * d)
        M = M + c2 * commutation
    return M


def jacobi_cholesky_solve(
    M: torch.Tensor,                   # (..., n, n) symmetric positive definite
    g: torch.Tensor,                   # (..., n)

    *,
    residual_max: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # x (..., n), ok (...), residual (...)
    r"""Solve :math:`M x = g` by a Jacobi-scaled float64 Cholesky factorization with a residual gate.

    With :math:`D = \mathrm{diag}(M)^{-1/2}`, factor :math:`D M D = L L^\top` and set
    :math:`x = D (L L^\top)^{-1} D g`. A solve is ``ok`` when the factorization succeeds, :math:`x` is
    finite, and the normwise backward error :math:`\|Mx - g\| / (\|M\|_F\|x\| + \|g\|) \le`
    ``residual_max``. The Frobenius norm (:math:`\|M\|_2 \le \|M\|_F \le \sqrt n\,\|M\|_2`) avoids an
    SVD per operator on the hot path. Failed solves return :math:`x = 0`.
    """
    diagonal = torch.diagonal(M, dim1=-2, dim2=-1)
    D = diagonal.clamp_min(torch.finfo(M.dtype).tiny).rsqrt()             # (..., n)
    scaled = D[..., :, None] * M * D[..., None, :]
    L, info = torch.linalg.cholesky_ex(scaled)
    x = D * torch.cholesky_solve((D * g)[..., None], L)[..., 0]
    residual = (torch.linalg.vector_norm((M @ x[..., None])[..., 0] - g, dim=-1)
                / (torch.linalg.matrix_norm(M) * torch.linalg.vector_norm(x, dim=-1)
                   + torch.linalg.vector_norm(g, dim=-1)).clamp_min(torch.finfo(M.dtype).tiny))
    ok = (info == 0) & (diagonal > 0).all(dim=-1) & torch.isfinite(x).all(dim=-1) & (
        residual <= residual_max)
    return torch.where(ok[..., None], x, torch.zeros_like(x)), ok, residual


def stabilizer_leak(
    R:   torch.Tensor,                 # (..., d, d) whitened covector
    eta: torch.Tensor,                 # (..., d) whitened mean
) -> torch.Tensor:                     # (...)
    r"""Share of a whitened covector in the Fisher kernel: :math:`\|P_\perp\,\mathrm{asym}(R)\,P_\perp\|_F / \|R\|_F`.

    :math:`P_\perp = I - nn^\top` with :math:`n = \eta/|\eta|` (:math:`P_\perp = I` at
    :math:`\eta = 0`) projects onto the stabilizer :math:`\{Z: \mathrm{sym}\,Z = 0,\ Z\eta = 0\}`, where
    the undamped Fisher vanishes and only the rotation weight sets the step.
    """
    norm = torch.linalg.vector_norm(eta, dim=-1, keepdim=True)
    n = torch.where(norm > 0, eta / norm.clamp_min(torch.finfo(eta.dtype).tiny),
                    torch.zeros_like(eta))
    eye = torch.eye(eta.shape[-1], dtype=eta.dtype, device=eta.device)
    P_perp = eye - n[..., :, None] * n[..., None, :]
    asym = 0.5 * (R - R.transpose(-1, -2))
    kernel_part = P_perp @ asym @ P_perp
    return (torch.linalg.matrix_norm(kernel_part)
            / torch.linalg.matrix_norm(R).clamp_min(torch.finfo(R.dtype).tiny))


def type_row_spectrum(
    mu: torch.Tensor,                  # (R, H, d)
    s:  torch.Tensor,                  # (R, H, d)

    *,
    rotation_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # a, lambda_min, cond_P, each (R, H)
    r"""Per-block :math:`a = \mu^\top S^{-1}\mu` and :math:`\lambda_{\min}`, :math:`\mathrm{cond}_{\mathcal P}` of the completed metric."""
    a = (mu.square() / s).sum(dim=-1)
    lam_min, lam_max = fisher_spectrum_bounds(a, mu.shape[-1], rotation_weight=rotation_weight)
    return a, lam_min, lam_max / lam_min


def type_metric_residual(
    W:     torch.Tensor,               # (R, H, d, d) solved direction
    G_W:   torch.Tensor,               # (R, H, d, d)
    mu:    torch.Tensor,               # (R, H, d)
    s:     torch.Tensor,               # (R, H, d)
    scale: torch.Tensor,               # (R,)

    *,
    rotation_weight: float,
) -> torch.Tensor:                     # (R,) max over blocks
    r"""Relative residual :math:`\|\mathcal M(W) - \mathrm{scale}\,G_W\|_F / \|\mathrm{scale}\,G_W\|_F` at the local row."""
    c1, c2 = completion_coefficients(rotation_weight)
    N = c1 * torch.diag_embed(s) + mu[..., :, None] * mu[..., None, :]
    applied = (W / s[..., :, None]) @ N + c2 * W.transpose(-1, -2)
    target = scale[:, None, None, None] * G_W
    return (torch.linalg.matrix_norm(applied - target)
            / torch.linalg.matrix_norm(target).clamp_min(torch.finfo(W.dtype).tiny)).amax(dim=-1)


def whitened_condition(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:   # (...) min, cond
    r"""Smallest eigenvalue and condition number of the Jacobi-scaled SPD matrix :math:`DMD`."""
    D = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(torch.finfo(M.dtype).tiny).rsqrt()
    eigenvalues = torch.linalg.eigvalsh(D[..., :, None] * M * D[..., None, :])
    lam_min = eigenvalues[..., 0]
    return lam_min, eigenvalues[..., -1] / lam_min.clamp_min(torch.finfo(M.dtype).tiny)

