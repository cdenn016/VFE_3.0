r"""Batched principal matrix square root, logarithm, and logarithm Frechet derivative.

Pure torch, batched over ``(..., n, n)``. The caller runs these in float64 with autocast disabled;
nothing here raises on a per-element numerical failure. Each kernel returns a failure mask instead,
and failed elements come back NaN.

  * :func:`sqrtm_denman_beavers` -- scaled product-form Denman-Beavers iteration (Higham 2008,
    *Functions of Matrices*, Sec. 6.3).
  * :func:`logm_principal` -- inverse scaling and squaring: square roots until
    :math:`\|A^{1/2^s} - I\|_1 \le \theta`, then the 8-point Gauss-Legendre quadrature of
    :math:`\log(I + E) = \int_0^1 E (I + tE)^{-1} dt`, which equals the [8/8] Pade approximant
    (Higham 2008, Thm 11.11).
  * :func:`logm_frechet_pair` -- the same ladder run on block upper-triangular pairs
    :math:`(P_0, P_1) \leftrightarrow \begin{pmatrix} P_0 & P_1 \\ 0 & P_0 \end{pmatrix}`, whose
    principal log is :math:`\begin{pmatrix} \log A & L_{\log}(A, E) \\ 0 & \log A \end{pmatrix}`
    (Mathias 1996; Higham 2008, Thm 3.6), without forming ``2n x 2n`` matrices.
  * :func:`principal_strip_certificate` -- :math:`\max|\mathrm{Im}\,\lambda(X)| < \pi - \delta`,
    by Bendixson's bound with an exact eigenvalue test on the blocks it cannot certify.

GPU sync discipline: one host read per square root decides whether any element needs another
root; the Denman-Beavers loop runs a fixed iteration budget with no per-iteration reads.
"""

import math
from typing import Tuple

import torch


# 8-point Gauss-Legendre nodes and weights on [0, 1] (Golub-Welsch).
_GL_NODES: Tuple[float, ...] = (
    0.0198550717512319, 0.1016667612931866, 0.2372337950418356, 0.4082826787521751,
    0.5917173212478251, 0.7627662049581644, 0.8983332387068135, 0.9801449282487680,
)
_GL_WEIGHTS: Tuple[float, ...] = (
    0.0506142681451882, 0.1111905172266873, 0.1568533229389437, 0.1813418916891811,
    0.1813418916891812, 0.1568533229389436, 0.1111905172266873, 0.0506142681451883,
)
# Scaling is applied only while the iterate is far from convergence.
_DB_SCALING_THRESHOLD: float = 1e-2


def _eye_like(A: torch.Tensor) -> torch.Tensor:          # (..., n, n)
    n = A.shape[-1]
    return torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)


def _norm1(A: torch.Tensor) -> torch.Tensor:              # (...) max column sum
    return torch.linalg.matrix_norm(A, ord=1)


def _all_finite(A: torch.Tensor) -> torch.Tensor:         # (...) bool over the last two dims
    return torch.isfinite(A).all(dim=-1).all(dim=-1)


def _db_scale(M0: torch.Tensor, live: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Determinantal scaling :math:`\mu = |\det M|^{-1/(2n)}` while ``dist`` exceeds 1e-2, else 1."""
    n = M0.shape[-1]
    _, logabsdet = torch.linalg.slogdet(M0)
    mu = torch.exp(-logabsdet / (2 * n))
    use = live & (dist > _DB_SCALING_THRESHOLD) & torch.isfinite(mu)
    return torch.where(use, mu, torch.ones_like(mu))[..., None, None]


def sqrtm_denman_beavers(
    A: torch.Tensor,                                      # (..., n, n)

    *,
    tol:      float = 1e-14,
    max_iter: int   = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:                   # (..., n, n) root, (...) failed
    r"""Principal square root by the scaled product-form Denman-Beavers iteration.

    With :math:`M_0 = Y_0 = A` and :math:`\mu_k = |\det M_k|^{-1/(2n)}`,

    .. math::
        Y_{k+1} = \tfrac12 \mu_k Y_k (I + \mu_k^{-2} M_k^{-1}), \qquad
        M_{k+1} = \tfrac12 \bigl(I + \tfrac12 (\mu_k^2 M_k + \mu_k^{-2} M_k^{-1})\bigr),

    so :math:`Y_k \to A^{1/2}` and :math:`M_k \to I` when :math:`\mathrm{spec}(A)` avoids the
    closed negative real axis. An element is frozen once :math:`\|M_k - I\|_1 \le` ``tol``. It is
    failed if an inverse is singular (``inv_ex`` info), a value is nonfinite, or it has not
    converged after ``max_iter`` iterations. The budget is fixed, so there is no host read.
    """
    eye = _eye_like(A)
    M = A.clone()
    Y = A.clone()
    batch = A.shape[:-2]
    failed = ~_all_finite(A)
    done = torch.zeros(batch, dtype=torch.bool, device=A.device)
    for _ in range(max_iter):
        dist = _norm1(M - eye)
        done = done | (dist <= tol)
        live = ~(done | failed)
        M_inv, info = torch.linalg.inv_ex(M)
        bad = (info != 0) | ~_all_finite(M_inv)
        failed = failed | (bad & live)
        live = live & ~bad
        M_inv = torch.where(bad[..., None, None], eye, M_inv)
        mu = _db_scale(M, live, dist)
        Y_next = 0.5 * mu * (Y @ (eye + M_inv / mu**2))
        M_next = 0.5 * (eye + 0.5 * (mu**2 * M + M_inv / mu**2))
        Y = torch.where(live[..., None, None], Y_next, Y)
        M = torch.where(live[..., None, None], M_next, M)
    failed = failed | ~(_norm1(M - eye) <= tol) | ~_all_finite(Y)
    return Y, failed


def logm_principal(
    A: torch.Tensor,                                      # (..., n, n)

    *,
    theta:     float = 0.25,
    max_roots: int   = 40,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:     # Log A, (...) failed, (...) roots
    r"""Principal logarithm by inverse scaling and squaring.

    Square roots are taken per element while :math:`\|A^{1/2^s} - I\|_1 > \theta`; then with
    :math:`E = A^{1/2^s} - I`,

    .. math::
        \mathrm{Log}\,A = 2^s \sum_{j=1}^{8} w_j (I + t_j E)^{-1} E .

    Failed elements (a failed root, too many roots, or a nonfinite value) return NaN.
    """
    shape = A.shape
    n = shape[-1]
    A_flat = A.reshape(-1, n, n).clone()                  # (B, n, n)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    roots = torch.zeros(A_flat.shape[0], dtype=torch.long, device=A.device)
    failed = ~_all_finite(A_flat)
    for _ in range(max_roots):
        need = (_norm1(A_flat - eye) > theta) & ~failed
        if not bool(need.any()):                          # one host read per root
            break
        index = need.nonzero(as_tuple=True)[0]
        root, bad = sqrtm_denman_beavers(A_flat[index])
        A_flat[index] = torch.where(bad[:, None, None], A_flat[index], root)
        failed[index] = failed[index] | bad
        roots[index] += (~bad).long()
    failed = failed | ~(_norm1(A_flat - eye) <= theta)
    E = torch.where(failed[:, None, None], torch.zeros_like(A_flat), A_flat - eye)
    L = torch.zeros_like(E)
    for t_j, w_j in zip(_GL_NODES, _GL_WEIGHTS):
        L = L + w_j * torch.linalg.solve(eye + t_j * E, E)
    L = L * torch.exp2(roots.to(dtype=A.dtype))[:, None, None]
    failed = failed | ~_all_finite(L)
    L = torch.where(failed[:, None, None], torch.full_like(L, float("nan")), L)
    return L.reshape(shape), failed.reshape(shape[:-2]), roots.reshape(shape[:-2])


def _pair_inverse(
    P0: torch.Tensor,
    P1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``[[P0, P1], [0, P0]]^{-1} = [[P0^{-1}, -P0^{-1} P1 P0^{-1}], [0, P0^{-1}]]``."""
    I0, info = torch.linalg.inv_ex(P0)
    return I0, -(I0 @ P1 @ I0), info


def _pair_product(
    P0: torch.Tensor,
    P1: torch.Tensor,
    Q0: torch.Tensor,
    Q1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""``(P0, P1)(Q0, Q1) = (P0 Q0, P0 Q1 + P1 Q0)`` for block upper-triangular pairs."""
    return P0 @ Q0, P0 @ Q1 + P1 @ Q0


def _sqrtm_denman_beavers_pair(
    A0: torch.Tensor,                                     # (B, n, n)
    A1: torch.Tensor,                                     # (B, n, n)

    *,
    tol:      float,
    max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:     # root pair, (B,) failed
    r"""Denman-Beavers on the pair; :math:`\det T = (\det P_0)^2`, so :math:`\mu` reads :math:`P_0`."""
    eye = _eye_like(A0)
    M0, M1 = A0.clone(), A1.clone()
    Y0, Y1 = A0.clone(), A1.clone()
    failed = ~(_all_finite(A0) & _all_finite(A1))
    done = torch.zeros_like(failed)
    for _ in range(max_iter):
        dist = _norm1(M0 - eye) + _norm1(M1)
        done = done | (dist <= tol)
        live = ~(done | failed)
        N0, N1, info = _pair_inverse(M0, M1)
        bad = (info != 0) | ~_all_finite(N0) | ~_all_finite(N1)
        failed = failed | (bad & live)
        live = live & ~bad
        N0 = torch.where(bad[:, None, None], eye, N0)
        N1 = torch.where(bad[:, None, None], torch.zeros_like(N1), N1)
        mu = _db_scale(M0, live, dist)
        T0, T1 = _pair_product(Y0, Y1, eye + N0 / mu**2, N1 / mu**2)
        M0_next = 0.5 * (eye + 0.5 * (mu**2 * M0 + N0 / mu**2))
        M1_next = 0.25 * (mu**2 * M1 + N1 / mu**2)
        step = live[:, None, None]
        Y0 = torch.where(step, 0.5 * mu * T0, Y0)
        Y1 = torch.where(step, 0.5 * mu * T1, Y1)
        M0 = torch.where(step, M0_next, M0)
        M1 = torch.where(step, M1_next, M1)
    failed = (failed | ~(_norm1(M0 - eye) + _norm1(M1) <= tol)
              | ~_all_finite(Y0) | ~_all_finite(Y1))
    return Y0, Y1, failed


def logm_frechet_pair(
    A: torch.Tensor,                                      # (..., n, n)
    E: torch.Tensor,                                      # (..., n, n) direction

    *,
    theta:           float = 0.25,
    direction_scale: float = 1e-2,
    tol:             float = 1e-14,
    max_iter:        int   = 12,
    max_roots:       int   = 40,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:     # Log A, L_log(A, E), (...) failed
    r"""Principal log and its Frechet derivative from the block-triangular pair.

    Every primary function of :math:`T = \begin{pmatrix} A & E \\ 0 & A \end{pmatrix}` is
    :math:`\begin{pmatrix} f(A) & L_f(A, E) \\ 0 & f(A) \end{pmatrix}`, so the inverse scaling and
    squaring ladder of :func:`logm_principal` runs on pairs with

    .. math::
        (P_0, P_1)^{-1} = (P_0^{-1}, -P_0^{-1} P_1 P_0^{-1}), \qquad
        (P_0, P_1)(Q_0, Q_1) = (P_0 Q_0, P_0 Q_1 + P_1 Q_0).

    By linearity the direction is scaled to :math:`\|E\|_1 =` ``direction_scale`` :math:`\theta`
    so that it does not add square roots, and the result is scaled back. Failed elements return
    NaN in both outputs.
    """
    shape = A.shape
    n = shape[-1]
    A0 = A.reshape(-1, n, n).clone()                      # (B, n, n)
    E_flat = E.reshape(-1, n, n)
    scale = (_norm1(E_flat) / (direction_scale * theta)).clamp_min(torch.finfo(A.dtype).tiny)
    A1 = (E_flat / scale[:, None, None]).clone()          # (B, n, n), ||A1||_1 <= direction_scale theta
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    roots = torch.zeros(A0.shape[0], dtype=torch.long, device=A.device)
    failed = ~(_all_finite(A0) & _all_finite(A1))
    for _ in range(max_roots):
        need = (_norm1(A0 - eye) + _norm1(A1) > theta) & ~failed
        if not bool(need.any()):                          # one host read per root
            break
        index = need.nonzero(as_tuple=True)[0]
        Y0, Y1, bad = _sqrtm_denman_beavers_pair(A0[index], A1[index], tol=tol, max_iter=max_iter)
        keep = bad[:, None, None]
        A0[index] = torch.where(keep, A0[index], Y0)
        A1[index] = torch.where(keep, A1[index], Y1)
        failed[index] = failed[index] | bad
        roots[index] += (~bad).long()
    failed = failed | ~(_norm1(A0 - eye) + _norm1(A1) <= theta)
    zero = failed[:, None, None]
    E0 = torch.where(zero, torch.zeros_like(A0), A0 - eye)
    E1 = torch.where(zero, torch.zeros_like(A1), A1)
    L0 = torch.zeros_like(E0)
    L1 = torch.zeros_like(E1)
    for t_j, w_j in zip(_GL_NODES, _GL_WEIGHTS):
        I0, I1, _ = _pair_inverse(eye + t_j * E0, t_j * E1)
        R0, R1 = _pair_product(I0, I1, E0, E1)
        L0 = L0 + w_j * R0
        L1 = L1 + w_j * R1
    power = torch.exp2(roots.to(dtype=A.dtype))[:, None, None]
    L0 = L0 * power
    L1 = L1 * power * scale[:, None, None]
    failed = failed | ~_all_finite(L0) | ~_all_finite(L1)
    nan = failed[:, None, None]
    L0 = torch.where(nan, torch.full_like(L0, float("nan")), L0)
    L1 = torch.where(nan, torch.full_like(L1, float("nan")), L1)
    return L0.reshape(shape), L1.reshape(shape), failed.reshape(shape[:-2])


def principal_strip_certificate(
    X: torch.Tensor,                                      # (..., n, n) real

    *,
    margin: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:                   # (...) certified, (...) Bendixson margin
    r"""Certify :math:`\max_k |\mathrm{Im}\,\lambda_k(X)| < \pi - \delta` for :math:`\delta` = ``margin``.

    Bendixson (1902): every eigenvalue of a real :math:`X` satisfies
    :math:`|\mathrm{Im}\,\lambda| \le \|\mathrm{skew}\,X\|_2 \le \|\mathrm{skew}\,X\|_F` with
    :math:`\mathrm{skew}\,X = (X - X^\top)/2`, so :math:`\|\mathrm{skew}\,X\|_F < \pi - \delta`
    certifies on the device. Matrices it does not certify go to an exact ``eigvals`` test (one host
    read, and a sync on CUDA; rare at initialization scale). Nonfinite matrices are not certified.

    The second output is the Bendixson margin :math:`\pi - \|\mathrm{skew}\,X\|_F` (negative when
    the bound exceeds :math:`\pi`), a diagnostic that does not depend on the eigenvalue test.
    """
    limit = math.pi - margin
    bound = torch.linalg.matrix_norm(0.5 * (X - X.transpose(-1, -2)))   # (...)
    finite = _all_finite(X)
    certified = (bound < limit) & finite
    uncertified = finite & ~certified
    if bool(uncertified.any()):
        n = X.shape[-1]
        flat_certified = certified.reshape(-1).clone()
        index = uncertified.reshape(-1).nonzero(as_tuple=True)[0]
        eigenvalues = torch.linalg.eigvals(X.reshape(-1, n, n)[index])   # (k, n) complex
        flat_certified[index] = eigenvalues.imag.abs().amax(dim=-1) < limit
        certified = flat_certified.reshape(certified.shape)
    return certified, math.pi - bound
