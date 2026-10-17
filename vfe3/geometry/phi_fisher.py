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

a single Kronecker product at the default :math:`\lambda = 2`. That ``'rotation'`` completion also raises
the metric of every Fisher-visible rotation (one with :math:`Z\eta \ne 0`) from :math:`a` toward
:math:`a + 2\lambda`. The ``'stabilizer'`` completion :math:`g + \lambda\|P_\perp\,\mathrm{asym}(Z)\,P_\perp\|_F^2`,
:math:`P_\perp = I - nn^\top`, :math:`n = \eta/|\eta|` (:math:`P_\perp = I` at :math:`\eta = 0`), penalizes only the
kernel: :math:`Z \mapsto P_\perp\,\mathrm{asym}(Z)\,P_\perp` is the Frobenius-orthogonal projector :math:`\Pi` onto
:math:`\{Z : \mathrm{sym}\,Z = 0,\ Z\eta = 0\}`, so in whitened coordinates the solve is
:math:`F^+ R + \Pi R/\lambda`, the pseudo-inverse belief-Fisher natural gradient on the Fisher-visible subspace plus
a kernel step that vanishes as :math:`\lambda \to \infty` (audit 2026-10-16, F50). It is implemented for the
token-type rows only. For a fixed belief the metric is
right-invariant, so in the spatial generator :math:`W = \delta U\,U^{-1}` of a token-type factor it
is the same formula at the local table row :math:`(\mu_v, \mathrm{diag}\,s_v)`, independent of the
position. Positional factors see the batch column of middle-fiber beliefs
:math:`E_{v_b}^{-1} \cdot q_{v_b}` and are solved densely.

Every function here is pure torch, batched, and expects float64 inputs with autocast disabled.
Blocks are ``(..., d, d)`` head blocks; matrices are vectorized row-major,
:math:`\mathrm{vec}(A X B) = (A \otimes B^\top)\,\mathrm{vec}\,X`.
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch


@dataclass(frozen=True)
class FisherRowStatistics:
    r"""Belief statistics that define the Fisher metric of ``R`` active rows of one frame table.

    ``kind='type'`` rows carry the layer-0 table row :math:`(\mu_v, s_v)` per head block;
    ``kind='pos'`` rows carry the pooled operator of their batch column, either dense
    (``pooled_operator``) or as the compact moments it is built from (``pooled_moments``), from which
    :func:`pooled_operator_chunks` forms it a row chunk at a time. ``count`` is
    :math:`n_v` (type) or :math:`B_{\rm tot}` (pos) and ``total`` the scored-token count
    :math:`N`, so the count-normalized direction carries :math:`N/\mathrm{count}`.
    """
    kind:            str                                   # "type" | "pos"
    count:           torch.Tensor                          # (R,) float64
    total:           torch.Tensor                          # () float64 scored tokens
    local_mean:      Optional[torch.Tensor] = None         # (R, H, d) type only
    local_var:       Optional[torch.Tensor] = None         # (R, H, d) type only
    pooled_operator: Optional[torch.Tensor] = None         # (R, H, d*d, d*d) pos only
    # pos only: (xi0 (T, H, d), C0 (T, H, d, d), C0_inv (T, H, d, d), gather (R, B)) of the T distinct
    # column types; row r's operator pools the types gather[r].
    pooled_moments:  Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None


# Byte budget of one chunk of dense positional operators (and their pooling intermediates). The full
# (R, H, d^2, d^2) float64 stack is 4 GiB per table at R = 128, H = 4, d = 32 (audit 2026-10-08 P2-22).
POOLED_OPERATOR_CHUNK_BYTES: int = 256 * 2**20


# Completions of the frame Fisher (``m_phi_fisher_completion``): lambda on every rotation, or on the stabilizer only.
FISHER_COMPLETIONS: Tuple[str, ...] = ("rotation", "stabilizer")


def _check_completion(completion: str) -> str:
    r"""Return ``completion`` if it is one of :data:`FISHER_COMPLETIONS`, else raise."""
    if completion not in FISHER_COMPLETIONS:
        raise ValueError(f"Fisher completion must be one of {FISHER_COMPLETIONS}, got {completion!r}")
    return completion


def completion_coefficients(rotation_weight: float) -> Tuple[float, float]:
    r"""Return :math:`(c_1, c_2) = (1 + \lambda/2, 1 - \lambda/2)` for :math:`(\lambda_s, \lambda_a) = (0, \lambda)`."""
    return 1.0 + 0.5 * rotation_weight, 1.0 - 0.5 * rotation_weight


def completed_fisher_operator(
    X:  torch.Tensor,                  # (..., d, d)
    xi: torch.Tensor,                  # (..., d) reference mean
    C:  torch.Tensor,                  # (..., d, d) reference covariance

    *,
    rotation_weight: float,
    completion:      str = "rotation",
) -> torch.Tensor:                     # (..., d, d)
    r"""Apply the completed metric operator in spatial coordinates.

    ``'rotation'``: :math:`\mathcal M(X) = C^{-1} X (c_1 C + \xi\xi^\top) + c_2 X^\top`. ``'stabilizer'``:

    .. math::
        \mathcal M(X) = C^{-1} X (C + \xi\xi^\top) + X^\top
            + \frac{\lambda}{2}\bigl(A X B - D X^\top D\bigr),

    the pullback of :math:`\lambda\,\Pi` through :math:`Z = L^{-1} X L`, with :math:`u = C^{-1}\xi`,
    :math:`a = \xi^\top u`, :math:`A = C^{-1} - uu^\top/a`, :math:`B = C - \xi\xi^\top/a`,
    :math:`D = I - u\xi^\top/a`; at :math:`\xi = 0` it is the ``'rotation'`` operator. Both are basis-free
    (independent of the factor :math:`L` of :math:`C`).
    """
    if _check_completion(completion) == "rotation":
        c1, c2 = completion_coefficients(rotation_weight)
        N = c1 * C + xi[..., :, None] * xi[..., None, :]
        return torch.linalg.solve(C, X @ N) + c2 * X.transpose(-1, -2)
    lam = float(rotation_weight)
    u = torch.linalg.solve(C, xi[..., :, None])[..., 0]                    # (..., d) C^{-1} xi
    a = (xi * u).sum(dim=-1)                                               # (...)
    inv_a = torch.where(a > 0, 1.0 / torch.where(a > 0, a, torch.ones_like(a)),
                        torch.zeros_like(a))[..., None, None]              # 0 where xi = 0
    xi_xi = xi[..., :, None] * xi[..., None, :]
    B = C - inv_a * xi_xi
    D = torch.eye(xi.shape[-1], dtype=X.dtype, device=X.device) - inv_a * (u[..., :, None] * xi[..., None, :])
    XB = X @ B
    AXB = torch.linalg.solve(C, XB) - inv_a * u[..., :, None] * (u[..., None, :] @ XB)
    X_t = X.transpose(-1, -2)
    return (torch.linalg.solve(C, X @ (C + xi_xi)) + X_t
            + 0.5 * lam * (AXB - D @ X_t @ D))


def fisher_spectrum_bounds(
    a: torch.Tensor,                   # (...) Mahalanobis norm mu^T Sigma^{-1} mu
    d: int,

    *,
    rotation_weight: float,
    completion:      str = "rotation",
) -> Tuple[torch.Tensor, torch.Tensor]:   # (...) lambda_min, (...) lambda_max
    r"""Extreme generalized eigenvalues of the completed metric relative to :math:`\mathcal P(X) = C^{-1} X C`.

    The spectrum is :math:`\{2 + a\}` (once), :math:`\{\tfrac12(2c_1 + a \pm \sqrt{a^2 + 4c_2^2})\}`
    (:math:`d - 1` times each), :math:`\{2\}` (:math:`d(d-1)/2` times) and :math:`\{\lambda\}`
    (:math:`(d-1)(d-2)/2` times). Only members with positive multiplicity enter the extremes. The
    maximum uses every member, which corrects the spec's ``cond_P`` for :math:`\lambda > 2`; the
    minimum also keeps the shear value 2, which the spec's ``min(lam, minus)`` omits and which is the
    minimum whenever :math:`\lambda > 2`.

    Under ``completion='stabilizer'`` the Fisher-visible pairs keep the uncompleted values
    :math:`\tfrac12(2 + a \pm \sqrt{a^2 + 4})` (:math:`c_1 = c_2 = 1`), the smaller of which is about
    :math:`a/2` for small :math:`a`; at :math:`a = 0` the stabilizer is every rotation and the smaller
    member is :math:`\lambda`.
    """
    if _check_completion(completion) == "stabilizer":
        root = torch.sqrt(a.square() + 4.0)
        members = [2.0 + a]
        if d >= 2:
            minus = torch.where(a > 0, 0.5 * (2.0 + a - root), torch.full_like(a, float(rotation_weight)))
            members += [minus, 0.5 * (2.0 + a + root), torch.full_like(a, 2.0)]
        if d >= 3:
            members.append(torch.full_like(a, float(rotation_weight)))
        stacked = torch.stack(members)                                     # (m, ...)
        return stacked.amin(dim=0), stacked.amax(dim=0)
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
    completion:      str = "rotation",
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

    ``completion='stabilizer'`` solves :math:`Z\eta\eta^\top + 2\,\mathrm{sym}Z + \lambda P_\perp\mathrm{asym}(Z)P_\perp = R`
    blockwise in the frame :math:`n = \eta/|\eta|`, :math:`a = |\eta|^2`: with :math:`R_{\perp\perp} = P_\perp R P_\perp`,
    :math:`r_{\perp n} = P_\perp R n`, :math:`r_{n\perp} = P_\perp R^\top n`, :math:`r_{nn} = n^\top R n`,

    .. math::
        Z_{\perp\perp} = \tfrac12\,\mathrm{sym}R_{\perp\perp} + \tfrac1\lambda\,\mathrm{asym}R_{\perp\perp}, \qquad
        z_{\perp n} = \frac{r_{\perp n} - r_{n\perp}}{a}, \qquad
        z_{n\perp} = r_{n\perp} - z_{\perp n}, \qquad
        z_{nn} = \frac{r_{nn}}{2 + a},

    and :math:`Z = Z_{\perp\perp} + z_{\perp n}n^\top + n z_{n\perp}^\top + z_{nn}nn^\top`. At :math:`\mu = 0`,
    :math:`P_\perp = I` and the solve is the ``'rotation'`` one. The visible mean rotations carry metric
    eigenvalue about :math:`a/2`, so the step along them grows like :math:`1/a` as :math:`\mu \to 0`.
    """
    scale = scale[:, None, None, None]
    if _check_completion(completion) == "stabilizer":
        return scale * _stabilizer_whitened_solve(G_W, mu, s, float(rotation_weight))
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


def _stabilizer_whitened_solve(
    G_W: torch.Tensor,                 # (R, H, d, d) spatial covector
    mu:  torch.Tensor,                 # (R, H, d) local table mean
    s:   torch.Tensor,                 # (R, H, d) local table variance
    lam: float,
) -> torch.Tensor:                     # (R, H, d, d) S^{1/2} Z S^{-1/2}, before the count scale
    r"""The ``'stabilizer'`` branch of :func:`fisher_type_solve` (block formulas in its docstring)."""
    r = s.sqrt()
    ratio = r[..., :, None] / r[..., None, :]                              # (R, H, d, d) sqrt(s_j/s_k)
    R = G_W * ratio                                                        # whitened covector
    eta = mu / r                                                           # (R, H, d)
    a = eta.square().sum(dim=-1)                                           # (R, H)
    visible = a > 0
    n = torch.where(visible[..., None], eta / a.sqrt().clamp_min(torch.finfo(a.dtype).tiny)[..., None],
                    torch.zeros_like(eta))                                 # (R, H, d); 0 where eta = 0
    R_n = (R @ n[..., None])[..., 0]                                       # (R, H, d) R n
    R_t_n = (R.transpose(-1, -2) @ n[..., None])[..., 0]                   # (R, H, d) R^T n
    r_nn = (n * R_n).sum(dim=-1)                                           # (R, H)
    r_pn = R_n - r_nn[..., None] * n                                       # P_perp R n
    r_np = R_t_n - r_nn[..., None] * n                                     # P_perp R^T n
    eye = torch.eye(eta.shape[-1], dtype=R.dtype, device=R.device)
    P = eye - n[..., :, None] * n[..., None, :]                            # (R, H, d, d)
    R_pp = P @ R @ P
    R_pp_t = R_pp.transpose(-1, -2)
    Z = 0.25 * (R_pp + R_pp_t) + 0.5 * (R_pp - R_pp_t) / lam              # (R, H, d, d)
    z_pn = (r_pn - r_np) / torch.where(visible, a, torch.ones_like(a))[..., None]
    z_np = r_np - z_pn
    z_nn = r_nn / (2.0 + a)
    Z = (Z + z_pn[..., :, None] * n[..., None, :] + n[..., :, None] * z_np[..., None, :]
         + z_nn[..., None, None] * (n[..., :, None] * n[..., None, :]))
    return Z * ratio


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


def pooled_operator_chunks(
    statistics: FisherRowStatistics,

    *,
    rotation_weight: float,
    chunk_bytes:     Optional[int] = None,             # None: POOLED_OPERATOR_CHUNK_BYTES
) -> Iterator[Tuple[int, int, torch.Tensor]]:             # (start, stop, (stop-start, H, d*d, d*d))
    r"""Yield the positional operators of ``statistics`` in row chunks of at most ``chunk_bytes``.

    A dense ``pooled_operator`` is sliced. From ``pooled_moments`` each chunk is formed by
    :func:`pooled_pos_operator` on its own rows only, so peak memory is one chunk's operators and
    pooling intermediates instead of the whole table's; the operators are bitwise those of a single
    call, because each row's operator depends only on its own batch column.
    """
    chunk_bytes = POOLED_OPERATOR_CHUNK_BYTES if chunk_bytes is None else int(chunk_bytes)
    if statistics.pooled_operator is not None:
        operator = statistics.pooled_operator
        rows, n_blocks, n, _ = operator.shape
        step = max(1, chunk_bytes // max(1, n_blocks * n * n * operator.element_size()))
        for start in range(0, rows, step):
            yield start, min(rows, start + step), operator[start:start + step]
        return
    if statistics.pooled_moments is None:
        raise ValueError("positional Fisher statistics need pooled_operator or pooled_moments")
    xi0, C0, C0_inv, gather = statistics.pooled_moments
    rows, batch = gather.shape
    n_blocks, d = xi0.shape[-2:]
    # Per row: the (H, d^2, d^2) operator and its einsum output, plus three gathered (B, H, d, d) inputs.
    per_row = (2 * n_blocks * d ** 4 + 3 * batch * n_blocks * d * d) * C0.element_size()
    step = max(1, chunk_bytes // per_row)
    for start in range(0, rows, step):
        index = gather[start:start + step]                                  # (r, B)
        yield start, min(rows, start + step), pooled_pos_operator(
            xi0[index], C0[index], C0_inv[index], rotation_weight=rotation_weight)


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
    completion:      str = "rotation",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # a, lambda_min, cond_P, each (R, H)
    r"""Per-block :math:`a = \mu^\top S^{-1}\mu` and :math:`\lambda_{\min}`, :math:`\mathrm{cond}_{\mathcal P}` of the completed metric."""
    a = (mu.square() / s).sum(dim=-1)
    lam_min, lam_max = fisher_spectrum_bounds(a, mu.shape[-1], rotation_weight=rotation_weight,
                                              completion=completion)
    return a, lam_min, lam_max / lam_min


def type_metric_residual(
    W:     torch.Tensor,               # (R, H, d, d) solved direction
    G_W:   torch.Tensor,               # (R, H, d, d)
    mu:    torch.Tensor,               # (R, H, d)
    s:     torch.Tensor,               # (R, H, d)
    scale: torch.Tensor,               # (R,)

    *,
    rotation_weight: float,
    completion:      str = "rotation",
) -> torch.Tensor:                     # (R,) max over blocks
    r"""Relative residual :math:`\|\mathcal M(W) - \mathrm{scale}\,G_W\|_F / \|\mathrm{scale}\,G_W\|_F` at the local row.

    The ``'stabilizer'`` operator is applied through :func:`completed_fisher_operator` in spatial
    coordinates, independently of the solver's whitened block formulas.
    """
    if _check_completion(completion) == "stabilizer":
        applied = completed_fisher_operator(W, mu, torch.diag_embed(s), rotation_weight=rotation_weight,
                                            completion=completion)
    else:
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

