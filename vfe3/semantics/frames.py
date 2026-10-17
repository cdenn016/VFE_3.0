r"""Gauge-frame (``phi``) semantics: batched float64 Lie/frame math and the ``phi`` section.

A token's gauge frame is the block-diagonal group element

.. math::  U_v = \exp(X_v), \qquad X_v = \sum_a \phi_{v,a} G_a ,

and the frame realized at sequence position ``p`` is :math:`U_{v,p} = e^{X_v} e^{Y_p}` with the
positional algebra element :math:`Y_p`. The trained model has exact global redundancies (package
spec): (i) a split :math:`e^{X_v} \to e^{X_v} k`, :math:`e^{Y} \to k^{-1} e^{Y}` with one global
block-GL ``k``; (ii) a global right ``h`` on :math:`e^{Y}` compensated in the relational operators;
(iii) a global left positive diagonal ``D``: :math:`\mu \to D\mu`, :math:`\mathrm{var} \to D^2
\mathrm{var}`, :math:`e^{X} \to D e^{X}` (whether (iii) is exact for a given configuration is decided
by :func:`vfe3.semantics.common.diagonal_redundancy_status`; a LayerNorm, affine or not, breaks it).
No function of a single raw :math:`\phi_v` is invariant, so every per-token frame statistic here is
computed in the gauge-fixed chart

.. math::  \hat U_v = D_0\, e^{X_v}\, m^{-1} D_0^{-1}, \qquad
           \log d_{0,k} = -\tfrac12\, \overline{\log \mathrm{var}_{\cdot k}}^{\,\mathrm{ref}},

with ``m`` the bi-invariant (exponential) barycenter of the reference frames. ``U_hat`` is invariant
under (i) and (iii) when ``m`` is the fixed point (up to the barycenter residual otherwise); statistics of the raw coordinates are labeled as statements about the trained
trivialization. For ``transport_mode="flat"`` the transport :math:`\Omega_{ij} = U_i U_j^{-1}`
telescopes, so holonomy is identically the identity and is never plotted.

All matrix work is float64 torch on the CPU, batched over tokens and chunked over rows.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.semantics.common import (
    OKABE_ITO,
    AnalogyQuestion,
    AnalysisContext,
    SectionResult,
    apply_style,
    diagonal_redundancy_status,
    markdown_table,
    panel_label,
    save_figure,
    write_table,
)

Tensor = torch.Tensor

_F64 = torch.float64
_CHUNK = 4096                 # matrices per batched linear-algebra call (bounds LAPACK workspace)
_BRANCH_TOL = 1e-8            # |arg lambda| >= pi - tol counts as on the closed negative real axis
_ISS_RADIUS = 0.25            # inverse scaling and squaring stops once ||B - I||_F < 0.25
_ISS_TERMS = 18               # atanh series terms: ||Z||_2 <= 0.143 => truncation < 1e-29
_PAR_MIN_ROWS = 64           # smallest per-worker share of a batch worth a thread (see _pmap)

# run_phi_section budgets; every key is overridable through ctx.options
PHI_DEFAULTS: Dict[str, Any] = {
    "frame_ref_count":          4096,   # reference set for the barycenter and D0 (top frequency)
    "frame_rerank":             100,    # group-analogy rerank pool
    "frame_rerank_questions":   500,    # benchmark questions reranked (seeded sample; curated: all)
    "frame_null":               500,    # relation-transport permutation-null draws
    "frame_commutator_pairs":   500,    # random token pairs for commutator / BCH diagnostics
    "frame_max_positions":      None,   # positional analysis window (None -> cfg.max_seq_len)
    "frame_positional_null":    200,    # position-permutation null draws of the relative-position R^2
}

# Stable per-step streams np.random.default_rng([seed, _PHI_STREAM, id]) of run_phi_section, so each
# stochastic step draws the same numbers whichever others ran, failed or changed budget (as in
# vectors._STEP_STREAM; audit 2026-10-16 F73). Ids are nonzero: trailing zero words do not change a
# SeedSequence pool, so [seed, _PHI_STREAM, 0] would collide with [seed, _PHI_STREAM].
_PHI_STREAM = 0x70E1
_PHI_STEP_STREAM: Dict[str, int] = {"positional": 1, "commutators_raw": 2, "commutators_canonical": 3,
                                    "group_analogies_benchmark": 4, "relation_transports": 5}


# ----------------------------------------------------------------------------------------------
# Small tensor helpers
# ----------------------------------------------------------------------------------------------

def _t(x: Any) -> Tensor:
    r"""float64 CPU tensor view/copy of ``x`` (numpy arrays are shared, not copied, when possible)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=_F64)
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float64)))


@contextmanager
def _one_intra_op_thread():
    r"""Pin torch's intra-op pool to one thread for the duration (restored afterwards)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield prev
    finally:
        torch.set_num_threads(prev)


def _pmap(fn: Callable[..., Any], *xs: Tensor) -> Any:
    r"""Apply a batched op ``fn`` to row blocks of ``xs`` (split on dim 0) on a thread pool.

    Torch runs a batch of small ``eig``/``eigvals`` calls on one thread, and a batch of small
    ``solve`` calls slows down under intra-op threading (per-matrix LAPACK work on 32 x 32 blocks is
    too small to share). The batch is therefore split into ``k = min(T, n / _PAR_MIN_ROWS)`` row
    blocks, with ``T`` the current torch thread count, and each block runs on its own worker with the
    intra-op pool pinned to one thread. Every matrix goes through the same LAPACK call as in the
    unsplit batch, so the result does not depend on ``k``. ``fn`` returns a tensor or a tuple of
    tensors, concatenated back along dim 0.
    """
    n = int(xs[0].shape[0])
    with _one_intra_op_thread() as threads:
        k = max(1, min(int(threads), n // _PAR_MIN_ROWS))
        if k == 1:
            return fn(*xs)
        bounds = [round(i * n / k) for i in range(k + 1)]
        with ThreadPoolExecutor(max_workers=k) as ex:
            parts = list(ex.map(lambda i: fn(*(x[bounds[i]:bounds[i + 1]] for x in xs)), range(k)))
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([p[j] for p in parts]) for j in range(len(parts[0])))
    return torch.cat(parts)


def _fro(A: Tensor) -> Tensor:
    r"""Frobenius norm over the last two dims: (..., d, d) -> (...)."""
    return torch.linalg.matrix_norm(A, ord="fro")


def _expm(X: Tensor, chunk: int = _CHUNK) -> Tensor:
    r"""Chunked ``torch.matrix_exp`` of a (n, d, d) batch (float64)."""
    X = _t(X)
    if X.dim() == 2:
        return torch.matrix_exp(X)
    if X.shape[0] <= chunk:
        return _pmap(torch.matrix_exp, X)
    out = torch.empty_like(X)
    for s in range(0, X.shape[0], chunk):
        out[s:s + chunk] = _pmap(torch.matrix_exp, X[s:s + chunk])
    return out


def _eigvals(A: Tensor, chunk: int = _CHUNK) -> Tensor:
    r"""Chunked complex eigenvalues of a (n, d, d) real batch -> (n, d) complex128."""
    out = torch.empty(A.shape[:-1], dtype=torch.complex128)
    for s in range(0, A.shape[0], chunk):
        out[s:s + chunk] = _pmap(torch.linalg.eigvals, A[s:s + chunk])
    return out


def _summary(values: Any) -> Dict[str, Any]:
    r"""Quantile summary of the finite entries of ``values`` (count of non-finite reported)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    out: Dict[str, Any] = {"n": int(finite.sum()), "n_nonfinite": int((~finite).sum())}
    if finite.any():
        v = x[finite]
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        out.update(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), q05=float(q[0]),
                   q25=float(q[1]), median=float(q[2]), q75=float(q[3]), q95=float(q[4]),
                   max=float(v.max()))
    return out


def _spearman(x: Any, y: Any) -> Dict[str, Any]:
    r"""Spearman rank correlation on the jointly finite entries: ``{rho, p, n}``."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return {"rho": None, "p": None, "n": n}
    rho, p = spearmanr(x[keep], y[keep])
    return {"rho": float(rho), "p": float(p), "n": n}


# ----------------------------------------------------------------------------------------------
# Matrix exponential / logarithm
# ----------------------------------------------------------------------------------------------


def _relative_residual(L: Tensor, A: Tensor) -> Tensor:
    r"""Reconstruction residual :math:`r = \|e^{L} - A\|_F / \|A\|_F` per matrix (NaN if L is not finite)."""
    finite = torch.isfinite(L).flatten(-2).all(-1)
    r = torch.full(A.shape[:-2], float("nan"), dtype=_F64)
    if finite.any():
        r[finite] = _fro(_expm(L[finite]) - A[finite]) / _fro(A[finite])
    return r


def _logm_eig(A: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Eigen-decomposition log :math:`L = V \log\Lambda\, V^{-1}` (real part) of a (m, d, d) batch.

    Returns ``(L, on_cut)``; ``on_cut`` flags matrices with an eigenvalue on the closed negative
    real axis (``|arg lambda| >= pi - tol`` or ``lambda == 0``), which have no real principal log.
    Rows whose eigenvector matrix is singular come back non-finite.
    """
    lam, V = _pmap(torch.linalg.eig, A)                            # (m, d), (m, d, d) complex
    on_cut = ((lam.abs() == 0) | (lam.angle().abs() >= math.pi - _BRANCH_TOL)).any(-1)
    rhs = V * torch.log(lam).unsqueeze(-2)                         # V diag(log lambda)
    Lc, info = _pmap(lambda V_, r_: torch.linalg.solve_ex(V_, r_, left=False), V, rhs)   # X V = V diag(log lambda)
    L = Lc.real.contiguous()
    L[info != 0] = float("nan")
    return L, on_cut


def _sqrtm_db(B: Tensor, *, max_iter: int = 100, tol: float = 1e-12) -> Tuple[Tensor, Tensor]:
    r"""Principal square root by the determinant-scaled product-form Denman--Beavers iteration.

    Higham (2008), *Functions of Matrices*, eqs. (6.17)/(6.29): with :math:`M_0 = Y_0 = B` and
    :math:`\mu_k = |\det M_k|^{-1/(2d)}`,

    .. math::
        Y_{k+1} = \tfrac12 \mu_k Y_k (I + \mu_k^{-2} M_k^{-1}), \qquad
        M_{k+1} = \tfrac12 \big(I + \tfrac12(\mu_k^2 M_k + \mu_k^{-2} M_k^{-1})\big),

    so :math:`Y_k \to B^{1/2}` and :math:`M_k \to I`. Returns ``(Y, converged)`` with convergence
    declared at :math:`\|M_k - I\|_F \le` ``tol``.
    """
    d = B.shape[-1]
    eye = torch.eye(d, dtype=_F64)
    M, Y = B.clone(), B.clone()
    conv = torch.zeros(B.shape[0], dtype=torch.bool)
    for _ in range(max_iter):
        Minv, info = torch.linalg.inv_ex(M)
        mu = torch.exp(-torch.linalg.slogdet(M).logabsdet / (2 * d))[:, None, None]    # (m, 1, 1)
        Y = 0.5 * mu * (Y @ (eye + Minv / mu ** 2))
        M = 0.5 * (eye + 0.5 * (mu ** 2 * M + Minv / mu ** 2))
        conv = (_fro(M - eye) <= tol) & (info == 0)
        if bool(conv.all()):
            break
    conv &= torch.isfinite(Y).flatten(-2).all(-1)
    return Y, conv


def _logm_iss(A: Tensor, *, max_sqrt: int = 64) -> Tuple[Tensor, Tensor]:
    r"""Principal log by inverse scaling and squaring (Kenney--Laub; Higham 2008, sec. 11.5).

    Take :math:`s` principal square roots until :math:`\|B - I\|_F < 1/4`, :math:`B = A^{1/2^s}`, then

    .. math::
        \log A = 2^s \log B, \qquad \log B = 2\,\mathrm{atanh}(Z) = 2\sum_{j \ge 0}
        \frac{Z^{2j+1}}{2j+1}, \qquad Z = (B - I)(B + I)^{-1},

    truncated after ``_ISS_TERMS`` terms (:math:`\|Z\|_2 \le 0.143`). Returns ``(L, converged)``;
    non-converged rows are NaN. ``A`` must have no eigenvalue on the closed negative real axis.
    """
    m, d = A.shape[0], A.shape[-1]
    eye = torch.eye(d, dtype=_F64)
    B = A.clone()
    s = torch.zeros(m, dtype=_F64)
    conv = torch.ones(m, dtype=torch.bool)
    for _ in range(max_sqrt):
        need = conv & (_fro(B - eye) >= _ISS_RADIUS)
        if not bool(need.any()):
            break
        root, ok = _sqrtm_db(B[need])
        B[need] = root
        s[need] += 1.0
        conv[need] = ok
    conv &= _fro(B - eye) < _ISS_RADIUS                             # NaN -> False
    B[~conv] = eye                                                  # keep the series finite; rows NaN'd below
    Z = torch.linalg.solve(B + eye, B - eye, left=False)            # (B - I)(B + I)^{-1}
    Z2 = Z @ Z
    term, S = Z.clone(), Z.clone()
    for j in range(1, _ISS_TERMS):
        term = term @ Z2
        S = S + term / (2 * j + 1)
    L = 2.0 * S * (2.0 ** s)[:, None, None]
    L[~conv] = float("nan")
    return L, conv


def logm_real(
    A: Any,                  # (..., d, d) real matrices

    *,
    tol:        float = 1e-9,
    accept_tol: float = 1e-6,
    chunk:      int   = _CHUNK,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Batched principal real matrix logarithm with reconstruction check.

    Primary path: :math:`A = V\Lambda V^{-1}`, :math:`L = \mathrm{Re}\, V \log\Lambda\, V^{-1}`
    (principal branch per eigenvalue), residual :math:`r = \|e^{L} - A\|_F / \|A\|_F`. Rows with
    :math:`r >` ``tol`` (defective or ill-conditioned eigenvectors) are recomputed by inverse scaling
    and squaring (:func:`_logm_iss`), keeping whichever candidate reconstructs ``A`` better.

    Returns ``(L, residual, ok)`` with shapes ``(..., d, d)``, ``(...)``, ``(...)``. ``ok`` holds iff
    a real principal log exists (no eigenvalue on the closed negative real axis) and the final
    residual is :math:`\le` ``accept_tol``; rows with ``ok = False`` have ``L = NaN`` (never a faked
    log). ``residual`` is NaN for matrices on the branch cut.
    """
    A = _t(A)
    batch_shape, d = A.shape[:-2], A.shape[-1]
    A2 = A.reshape(-1, d, d)
    n = A2.shape[0]
    L = torch.full_like(A2, float("nan"))
    res = torch.full((n,), float("nan"), dtype=_F64)
    ok = torch.zeros(n, dtype=torch.bool)
    for s in range(0, n, chunk):
        Ac = A2[s:s + chunk]
        Lc, on_cut = _logm_eig(Ac)
        rc = _relative_residual(Lc, Ac)
        rc[on_cut] = float("nan")
        retry = (~on_cut) & ~(rc <= tol)
        if bool(retry.any()):
            Li, conv = _logm_iss(Ac[retry])
            ri = _relative_residual(Li, Ac[retry])
            ri[~conv] = float("nan")
            re = rc[retry]
            use = conv & (torch.isnan(re) | (ri < re))
            Lc[retry] = torch.where(use[:, None, None], Li, Lc[retry])
            rc[retry] = torch.where(use, ri, re)
        okc = (~on_cut) & (rc <= accept_tol)
        Lc[~okc] = float("nan")
        L[s:s + chunk], res[s:s + chunk], ok[s:s + chunk] = Lc, rc, okc
    return L.reshape(A.shape), res.reshape(batch_shape), ok.reshape(batch_shape)


def bi_invariant_mean(
    U: Any,                  # (n, d, d) group elements

    *,
    max_iter:     int              = 30,
    tol:          float            = 1e-10,
    log_U:        Optional[Any]    = None,     # (n, d, d) known logs of U (init only), e.g. X when U = e^X
    max_iter_cap: Optional[int]    = None,     # hard cap of the adaptive extension (None -> 4 max_iter)
) -> Tuple[Tensor, Dict[str, Any]]:
    r"""Exponential (bi-invariant) barycenter of group elements (Pennec & Arsigny 2013).

    Solves :math:`\sum_a \log(m^{-1} U_a) = 0` by the fixed-point iteration

    .. math::  m \leftarrow m\, \exp\Big(\tfrac1n \sum_a \log(m^{-1} U_a)\Big),

    initialized at :math:`\exp(\overline{\log U_a})` (:math:`= \exp(\bar X)` when ``log_U = X``).
    The fixed point is equivariant, :math:`\mathrm{mean}(g U h) = g\, \mathrm{mean}(U)\, h`, because
    :math:`\log(h^{-1} A h) = h^{-1} \log(A) h`; the initialization and intermediate iterates are
    not. Elements whose :math:`m^{-1} U_a` has no real principal log are left out of the average (a
    conjugation-invariant condition) and counted.

    The iteration runs ``max_iter`` steps and is then extended, up to ``max_iter_cap`` steps, for as
    long as each step still contracts the residual by at least a factor 0.9 (a slowly converging
    iteration is finished rather than abandoned; a stagnating one stops at ``max_iter``).

    Returns ``(m, info)``; ``info`` has ``iterations``, ``residual`` (:math:`\|\overline{\log(m^{-1}U)}\|_F`
    at the returned ``m``), ``converged``, ``extended`` (ran past ``max_iter``), ``n_log_failed``,
    ``history`` and ``returned`` (``"fixed_point"``, or ``"best_iterate"`` -- the iterate with the
    smallest residual, which may be the initialization -- when the iteration stopped unconverged).
    """
    cap = max(int(max_iter), 4 * int(max_iter) if max_iter_cap is None else int(max_iter_cap))
    U = _t(U)
    if U.dim() != 3 or U.shape[0] == 0:
        raise ValueError(f"bi_invariant_mean expects a non-empty (n, d, d) batch; got {tuple(U.shape)}")
    if log_U is None:
        L0, _, ok0 = logm_real(U)
        if not bool(ok0.any()):
            raise ValueError("bi_invariant_mean: no element has a real principal log")
        init = L0[ok0].mean(0)
    else:
        init = _t(log_U).mean(0)
    m = torch.matrix_exp(init)
    history: List[float] = []
    best_norm, best_m, best_failed = float("inf"), m, 0
    converged, n_failed, it = False, 0, 0
    for it in range(cap + 1):
        Lr, _, okr = logm_real(_pmap(lambda U_: torch.linalg.solve(m, U_), U))   # log(m^{-1} U_a)
        n_failed = int((~okr).sum())
        if not bool(okr.any()):
            break
        step = Lr[okr].mean(0)
        norm = float(_fro(step))
        history.append(norm)
        if norm < best_norm:
            best_norm, best_m, best_failed = norm, m, n_failed
        if norm <= tol:
            converged = True
            break
        contracting = len(history) >= 2 and history[-1] <= 0.9 * history[-2]
        if it >= cap or (it >= max_iter and not contracting):
            break
        m = m @ torch.matrix_exp(step)
    info = {"iterations": it, "residual": best_norm, "converged": converged, "extended": it > max_iter,
            "n_log_failed": best_failed if not converged else n_failed, "history": history,
            "returned": "fixed_point" if converged else "best_iterate"}
    return (m if converged else best_m), info


# ----------------------------------------------------------------------------------------------
# Gauge-fixed chart
# ----------------------------------------------------------------------------------------------

@dataclass
class CanonicalFrames:
    r"""Gauge-fixed frame chart :math:`\hat U_v = D_0 e^{X_v} m^{-1} D_0^{-1}` and its log :math:`L_v`.

    ``U_hat[b]`` / ``L[b]`` are (n, d_b, d_b) float64 per block; ``ok[r]`` holds when every block of
    row ``r`` has a real principal log. ``ref_ids`` and ``block_slices`` record the reference set and
    the coordinate range of each block.
    """

    ids:          np.ndarray                 # (n,) int64 token ids
    log_d0:       np.ndarray                 # (K,) unit normalization log d0_k = -1/2 mean_ref log var_k
    m:            List[Tensor]               # per block (d_b, d_b) barycenter of the raw reference frames
    U_hat:        List[Tensor]               # per block (n, d_b, d_b) D0 e^{X_v} m^{-1} D0^{-1}
    L:            List[Tensor]               # per block (n, d_b, d_b) logm_real(U_hat)
    ok:           np.ndarray                 # (n,) bool
    diagnostics:  Dict[str, Any]
    ref_ids:      np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    block_slices: List[slice] = field(default_factory=list)
    residual:     List[np.ndarray] = field(default_factory=list)   # per block (n,) logm reconstruction residual

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        self._row_of: Dict[int, int] = {int(t): r for r, t in enumerate(self.ids)}

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])

    def rows(self, token_ids: Sequence[int]) -> np.ndarray:
        r"""Row index of each token id, ``-1`` where absent."""
        return np.array([self._row_of.get(int(t), -1) for t in token_ids], dtype=np.int64)

    def log_chart(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""``(ids, X)``: flattened canonical logs :math:`[\mathrm{vec}\, L^b_v]_b` of the ``ok`` rows,
        (n_ok, sum_b d_b^2) float64 -- Frobenius-Euclidean coordinates of the gauge-fixed chart."""
        keep = torch.from_numpy(self.ok)
        flat = torch.cat([Lb[keep].reshape(int(keep.sum()), -1) for Lb in self.L], dim=1)
        return self.ids[self.ok], flat.numpy()


def canonical_frames(
    run:         Any,          # SemanticRun (phi_blocks, block_slices)
    ids:         Sequence[int],
    ref_ids:     Sequence[int],
    log_var_ref: Any,          # (n_ref, K) log prior variances of ref_ids

    *,
    max_iter:     int           = 30,
    tol:          float         = 1e-10,
    chunk:        int           = _CHUNK,
    max_iter_cap: Optional[int] = None,
) -> CanonicalFrames:
    r"""Gauge-fixed chart :math:`\hat U_v = D_0\, e^{X_v}\, m^{-1} D_0^{-1}` per block, and :math:`L_v = \log \hat U_v`.

    :math:`D_0 = \mathrm{diag}(e^{\log d_0})`, :math:`\log d_{0,k} = -\tfrac12 \mathrm{mean}_{v \in \mathrm{ref}}
    \log \mathrm{var}_{vk}`; :math:`m` = :func:`bi_invariant_mean` of the raw reference frames per block.
    Invariance: under :math:`U_v \to U_v k` the barycenter maps to :math:`m k`; under
    :math:`(\log\mathrm{var} \to \log\mathrm{var} + 2\log D,\ U_v \to D U_v)` it maps to :math:`D m` and
    :math:`D_0 \to D_0 D^{-1}`; in both cases :math:`\hat U_v` is unchanged. On the reference set the
    barycenter of :math:`\hat U` is the identity. Both statements hold for the fixed point of the
    barycenter iteration; when a block returns its best iterate instead, they hold only up to
    :math:`O(r)` with ``r`` the barycenter residual (``diagnostics["barycenter_residual_max"]``).
    Exact identity per block: :math:`\mathrm{tr}\, L_v = \mathrm{tr}\, X_v - \log\det m`
    (recorded as a diagnostic).
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    ref_ids = np.asarray(ref_ids, dtype=np.int64).reshape(-1)
    lv_ref = np.asarray(log_var_ref, dtype=np.float64)
    if ids.size == 0 or ref_ids.size == 0:
        raise ValueError("canonical_frames needs non-empty ids and ref_ids")
    if lv_ref.ndim != 2 or lv_ref.shape[0] != ref_ids.size:
        raise ValueError(f"log_var_ref must be (n_ref, K) with n_ref={ref_ids.size}; got {lv_ref.shape}")
    log_d0 = -0.5 * lv_ref.mean(axis=0)                                   # (K,)
    slices = list(run.block_slices)
    n = ids.size

    X_ref = run.phi_blocks(ref_ids)
    m_list: List[Tensor] = []
    bary: List[Dict[str, Any]] = []
    for b, sl in enumerate(slices):
        Xr = _t(X_ref[b])                                                 # (n_ref, d, d)
        m_b, info = bi_invariant_mean(_expm(Xr), max_iter=max_iter, tol=tol, log_U=Xr,
                                      max_iter_cap=max_iter_cap)
        m_list.append(m_b)
        bary.append(info)
    del X_ref

    d0 = [torch.exp(_t(log_d0[sl])) for sl in slices]                     # per block (d,)
    m_inv = [torch.linalg.inv(m_b) for m_b in m_list]
    U_hat = [torch.empty(n, sl.stop - sl.start, sl.stop - sl.start, dtype=_F64) for sl in slices]
    tr_X = np.empty((n, len(slices)))
    for s in range(0, n, chunk):
        Xc = run.phi_blocks(ids[s:s + chunk])
        for b in range(len(slices)):
            Xb = _t(Xc[b])                                                # (c, d, d)
            tr_X[s:s + chunk, b] = torch.diagonal(Xb, dim1=-2, dim2=-1).sum(-1).numpy()
            U_hat[b][s:s + chunk] = d0[b][:, None] * (torch.matrix_exp(Xb) @ m_inv[b]) / d0[b][None, :]

    L_list: List[Tensor] = []
    res_list: List[np.ndarray] = []
    ok = np.ones(n, dtype=bool)
    blocks_diag: List[Dict[str, Any]] = []
    for b in range(len(slices)):
        L_b, res_b, ok_b = logm_real(U_hat[b], chunk=chunk)
        L_list.append(L_b)
        res_list.append(res_b.numpy())
        ok &= ok_b.numpy()
        tr_L = torch.diagonal(L_b, dim1=-2, dim2=-1).sum(-1).numpy()
        logdet_m = float(torch.linalg.slogdet(m_list[b]).logabsdet)
        ident = np.abs(tr_L - (tr_X[:, b] - logdet_m))
        blocks_diag.append({
            "d": int(U_hat[b].shape[-1]),
            "barycenter": {k: v for k, v in bary[b].items()},
            "logm_residual": _summary(res_b.numpy()),
            "logm_ok_fraction": float(ok_b.double().mean()),
            "trace_identity_max_abs_error": float(np.nanmax(ident)) if np.isfinite(ident).any() else None,
        })
    diagnostics = {
        "n": int(n), "n_ref": int(ref_ids.size), "n_ok": int(ok.sum()),
        "ok_fraction": float(ok.mean()), "log_d0": _summary(log_d0), "blocks": blocks_diag,
        "barycenter_converged_all": all(bool(x["converged"]) for x in bary),
        "barycenter_residual_max": max(float(x["residual"]) for x in bary),
    }
    return CanonicalFrames(ids=ids, log_d0=log_d0, m=m_list, U_hat=U_hat, L=L_list, ok=ok,
                           diagnostics=diagnostics, ref_ids=ref_ids, block_slices=slices, residual=res_list)


def invariant_scalars(cf: CanonicalFrames) -> Dict[str, np.ndarray]:
    r"""Per-token scalars of the gauge-fixed chart (each invariant under redundancies (i) and (iii)).

    * ``trace`` :math:`= \sum_b \log\det \hat U^b_v = \sum_b \mathrm{tr}\, L^b_v` (from ``slogdet``, so
      defined for every row); ``trace_block`` (n, B).
    * ``stretch`` :math:`= \log \sigma_{\max}(\hat U_v) - \log \sigma_{\min}(\hat U_v)` over all blocks
      (log condition number); ``stretch_block`` (n, B).
    * ``rms_log_sv`` :math:`= \sqrt{\mathrm{mean}\, (\log\sigma_i)^2}`.
    * ``rotation`` :math:`= \sqrt{\mathrm{mean}\, (\arg\lambda_i)^2}` over the eigenvalues of :math:`\hat U_v`.
    * ``sym_norm`` / ``skew_norm`` :math:`= \|\tfrac12(L + L^T)\|_F` / :math:`\|\tfrac12(L - L^T)\|_F`
      and ``log_norm`` :math:`= \|L\|_F` (NaN where ``ok`` is False).
    """
    n, B = cf.n, len(cf.U_hat)
    trace_block = np.empty((n, B))
    stretch_block = np.empty((n, B))
    sv_max = np.full(n, -np.inf)
    sv_min = np.full(n, np.inf)
    sq_log_sv = np.zeros(n)
    sq_arg = np.zeros(n)
    n_eig = 0
    sym2, skew2 = np.zeros(n), np.zeros(n)
    for b, (Uh, Lb) in enumerate(zip(cf.U_hat, cf.L)):
        trace_block[:, b] = torch.linalg.slogdet(Uh).logabsdet.numpy()
        log_sv = np.empty((n, Uh.shape[-1]))
        for s in range(0, n, _CHUNK):
            log_sv[s:s + _CHUNK] = torch.log(torch.linalg.svdvals(Uh[s:s + _CHUNK])).numpy()
        stretch_block[:, b] = log_sv.max(1) - log_sv.min(1)
        sv_max = np.maximum(sv_max, log_sv.max(1))
        sv_min = np.minimum(sv_min, log_sv.min(1))
        sq_log_sv += (log_sv ** 2).sum(1)
        sq_arg += (_eigvals(Uh).angle().numpy() ** 2).sum(1)
        n_eig += Uh.shape[-1]
        sym2 += (_fro(0.5 * (Lb + Lb.transpose(-1, -2))) ** 2).numpy()
        skew2 += (_fro(0.5 * (Lb - Lb.transpose(-1, -2))) ** 2).numpy()
    return {
        "trace": trace_block.sum(1), "trace_block": trace_block,
        "stretch": sv_max - sv_min, "stretch_block": stretch_block,
        "rms_log_sv": np.sqrt(sq_log_sv / n_eig), "rotation": np.sqrt(sq_arg / n_eig),
        "sym_norm": np.sqrt(sym2), "skew_norm": np.sqrt(skew2), "log_norm": np.sqrt(sym2 + skew2),
    }


# ----------------------------------------------------------------------------------------------
# Commutator / BCH diagnostics
# ----------------------------------------------------------------------------------------------

def _commutator_stats(Xa: Sequence[Tensor], Xb: Sequence[Tensor]) -> Dict[str, Any]:
    r"""Commutator ratio and BCH gap for paired algebra elements, per block and pooled.

    .. math::
        c = \frac{\|[X_a, X_b]\|_F}{\|X_a\|_F \|X_b\|_F}, \qquad
        g = \frac{\|\log(e^{X_a} e^{-X_b}) - (X_a - X_b)\|_F}{\|X_a - X_b\|_F}.

    Pooled values use the block-diagonal Frobenius norms (sums of squares over blocks).
    """
    comm2, na2, nb2, gap2, diff2 = 0.0, 0.0, 0.0, 0.0, 0.0
    ratio_block, gap_block = [], []
    for A, Bm in zip(Xa, Xb):
        C = A @ Bm - Bm @ A
        c2, a2, b2 = _fro(C) ** 2, _fro(A) ** 2, _fro(Bm) ** 2
        Lg, _, _ = logm_real(_expm(A) @ _expm(-Bm))
        D = A - Bm
        g2, d2 = _fro(Lg - D) ** 2, _fro(D) ** 2
        ratio_block.append(torch.sqrt(c2 / (a2 * b2)).numpy())
        gap_block.append(torch.sqrt(g2 / d2).numpy())
        comm2, na2, nb2, gap2, diff2 = comm2 + c2, na2 + a2, nb2 + b2, gap2 + g2, diff2 + d2
    return {
        "ratio": torch.sqrt(comm2 / (na2 * nb2)).numpy(),                # (n_pairs,)
        "ratio_block": np.stack(ratio_block, axis=1),                    # (n_pairs, B)
        "bch_gap": torch.sqrt(gap2 / diff2).numpy(),                     # (n_pairs,), NaN if log failed
        "bch_gap_block": np.stack(gap_block, axis=1),
    }


def _random_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    r"""``n_pairs`` uniformly random ordered pairs of distinct row indices in ``range(n)``."""
    if n < 2:
        raise ValueError("need at least two rows for pairs")
    a = rng.integers(0, n, size=n_pairs)
    b = (a + rng.integers(1, n, size=n_pairs)) % n                    # b != a, uniform over the rest
    return a, b


def commutator_diagnostics(
    run:     Any,
    ids:     Sequence[int],
    rng:     np.random.Generator,
    n_pairs: int = 500,
) -> Dict[str, Any]:
    r"""Raw-chart diagnostics of the token algebra elements (statements about the trained trivialization).

    Over ``n_pairs`` random token pairs: the commutator ratio
    :math:`\|[X_a,X_b]\|_F / (\|X_a\|_F\|X_b\|_F)` and the BCH gap
    :math:`\|\log(e^{X_a}e^{-X_b}) - (X_a - X_b)\|_F / \|X_a - X_b\|_F`, per block and pooled; plus the
    distribution of :math:`\|X_v\|_F` per block over all ``ids``. Small values mean raw-coordinate
    arithmetic approximates group arithmetic to first order (BCH).
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    norms: List[np.ndarray] = []
    for s in range(0, ids.size, _CHUNK):
        Xc = run.phi_blocks(ids[s:s + _CHUNK])
        norms.append(np.stack([_fro(_t(Xb)).numpy() for Xb in Xc], axis=1))
    norm_block = np.concatenate(norms, axis=0)                          # (n, B)
    a, b = _random_pairs(ids.size, n_pairs, rng)
    Xa = [_t(x) for x in run.phi_blocks(ids[a])]
    Xb = [_t(x) for x in run.phi_blocks(ids[b])]
    stats = _commutator_stats(Xa, Xb)
    return {
        "chart": "raw (trained trivialization)",
        "n_pairs": int(n_pairs),
        "norm_block": norm_block, "norm": np.sqrt((norm_block ** 2).sum(1)),
        **stats,
        "summary": {
            "norm": _summary(np.sqrt((norm_block ** 2).sum(1))),
            "norm_block": [_summary(norm_block[:, k]) for k in range(norm_block.shape[1])],
            "ratio": _summary(stats["ratio"]), "bch_gap": _summary(stats["bch_gap"]),
            "ratio_block": [_summary(stats["ratio_block"][:, k]) for k in range(norm_block.shape[1])],
            "bch_gap_block": [_summary(stats["bch_gap_block"][:, k]) for k in range(norm_block.shape[1])],
        },
    }


# ----------------------------------------------------------------------------------------------
# Reference content
# ----------------------------------------------------------------------------------------------

def _zca(Z: np.ndarray, eps_rel: float) -> np.ndarray:
    r"""ZCA whitening of the rows: :math:`(z - \bar z)\, V (\Lambda + \epsilon)^{-1/2} V^T`,
    :math:`\epsilon = ` ``eps_rel`` :math:`\cdot \mathrm{mean}\,\lambda` of the row covariance."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    C = Zc.T @ Zc / max(Zc.shape[0], 1)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)
    W = (V / np.sqrt(lam + eps_rel * lam.mean())) @ V.T
    return Zc @ W


def reference_content(
    run:      Any,
    ids:      Sequence[int],
    position: int,

    *,
    eps_rel: float = 1e-6,
) -> np.ndarray:
    r"""Reference content :math:`\xi_{v,p} = e^{-Y_p} e^{-X_v} \mu_v` per block, then per-block ZCA whitening.

    :math:`U_{v,p}^{-1} \mu_v` is exactly invariant under redundancies (i) and (iii) and maps to
    :math:`h^{-1}\xi` under (ii); ZCA whitening over the rows turns the global linear map into an
    orthogonal one, so the returned (n, K) float64 rows are invariant up to a per-block orthogonal map
    (exactly for ``eps_rel = 0``). Coordinates outside every frame block (none for block_glk) are not
    transported but are whitened as one extra group, so the global-diagonal redundancy acts on them
    orthogonally too. ``Y_p = 0`` when the run has no positional frames.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    xi = np.array(run.mu(ids), dtype=np.float64)                           # (n, K)
    Y = run.pos_phi_blocks(np.array([int(position)], dtype=np.int64))
    covered = np.zeros(xi.shape[1], dtype=bool)
    for s in range(0, ids.size, _CHUNK):
        Xc = run.phi_blocks(ids[s:s + _CHUNK])
        for b, sl in enumerate(run.block_slices):
            v = torch.matrix_exp(-_t(Xc[b])) @ _t(xi[s:s + _CHUNK, sl])[..., None]   # (c, d, 1)
            if Y is not None:
                v = torch.matrix_exp(-_t(Y[b][0])) @ v
            xi[s:s + _CHUNK, sl] = v[..., 0].numpy()
    for sl in run.block_slices:
        covered[sl] = True
        xi[:, sl] = _zca(xi[:, sl], eps_rel)
    if not covered.all():
        xi[:, ~covered] = _zca(xi[:, ~covered], eps_rel)
    return xi


# ----------------------------------------------------------------------------------------------
# Group analogies and relation transports
# ----------------------------------------------------------------------------------------------

def _rank_metrics(ranks: Sequence[Optional[int]]) -> Dict[str, Any]:
    r"""Accuracy (rank 1), hits@5 and MRR; ``None`` ranks count as misses with reciprocal rank 0."""
    r = np.array([np.inf if x is None else x for x in ranks], dtype=np.float64)
    if r.size == 0:
        return {"n": 0, "acc": None, "hits5": None, "mrr": None}
    return {"n": int(r.size), "acc": float((r == 1).mean()), "hits5": float((r <= 5).mean()),
            "mrr": float(np.where(np.isfinite(r), 1.0 / r, 0.0).mean())}


def group_analogies(
    cf:        CanonicalFrames,
    questions: Sequence[AnalogyQuestion],

    *,
    topk:                 int                           = 5,
    rerank:               int                           = 100,
    max_rerank_questions: Optional[int]                 = None,
    rng:                  Optional[np.random.Generator] = None,
    chunk:                int                           = 256,
) -> Dict[str, Any]:
    r"""Frame analogies ``a : a* :: b : ?`` in the gauge-fixed chart.

    Prediction :math:`P = \hat U_{a^*} \hat U_a^{-1} \hat U_b` per block (the relation transport
    :math:`\hat U_{a^*}\hat U_a^{-1}` applied to ``b``). Retrieval over all ``cf`` rows except
    ``a, a*, b`` by the chart distance :math:`\sum_b \|P^b - \hat U^b_x\|_F^2` (computed from inner
    products), then the top ``rerank`` candidates are re-scored by the relative residual
    :math:`\sum_b \|\log(P^b (\hat U^b_x)^{-1})\|_F^2`. ONLY-B baseline (Linzen 2016): chart distance to
    :math:`\hat U_b` with the same exclusions. Rerank ranks exist only when ``b*`` is in the pool
    (otherwise a miss; its MRR is truncated at ``rerank``). ``max_rerank_questions`` reranks a seeded
    random subset (all when None or when there are fewer questions); ``chart_reranked`` repeats the
    chart metrics on exactly that subset, so it pairs with ``rerank`` question by question.

    Also returns the BCH gap between the group prediction and the additive log-chart analogy,
    :math:`\|\log P - (L_{a^*} - L_a + L_b)\|_F / \|\log P\|_F` (block-pooled).
    """
    rows = [cf.rows([q.a, q.a_star, q.b, q.b_star]) for q in questions]
    keep = [i for i, r in enumerate(rows) if (r >= 0).all() and len(set(r.tolist())) == 4]
    out: Dict[str, Any] = {"n_questions": len(questions), "n_covered": len(keep), "questions": []}
    if not keep:
        out.update(overall={}, per_category={}, bch_gap=_summary([]))
        return out
    R = np.stack([rows[i] for i in keep])                                  # (Q, 4) rows a, a*, b, b*
    cats = [questions[i].category for i in keep]
    Q, n = R.shape[0], cf.n
    flat = [Uh.reshape(n, -1) for Uh in cf.U_hat]                          # views (n, d_b^2)
    sq = sum((f ** 2).sum(1) for f in flat)                                # (n,)

    rerank = int(min(rerank, n - 3))
    rr_sel = np.ones(Q, dtype=bool)
    if max_rerank_questions is not None and Q > max_rerank_questions:
        gen = rng if rng is not None else np.random.default_rng(0)
        rr_sel[:] = False
        rr_sel[gen.choice(Q, size=int(max_rerank_questions), replace=False)] = True

    # ONLY-B scores for the unique b rows: score(x) = 2 <U_b, U_x> - ||U_x||^2 (= -dist^2 + const)
    ub, ub_inv = np.unique(R[:, 2], return_inverse=True)
    S_b = sum(2.0 * f[torch.from_numpy(ub)] @ f.T for f in flat) - sq[None, :]   # (u, n)

    rank_chart = np.zeros(Q, dtype=np.int64)
    rank_only_b = np.zeros(Q, dtype=np.int64)
    rank_rerank: List[Optional[int]] = [None] * Q
    in_pool: List[Optional[bool]] = [None] * Q                             # b* among the rerank pool
    top_chart = np.zeros((Q, topk), dtype=np.int64)
    top_rerank: List[Optional[List[int]]] = [None] * Q
    bch = np.full(Q, np.nan)
    for s in range(0, Q, chunk):
        Rc = R[s:s + chunk]
        q = Rc.shape[0]
        ra, rs, rb, rt = (torch.from_numpy(Rc[:, j]) for j in range(4))
        P = [Uh[rs] @ _pmap(torch.linalg.solve, Uh[ra], Uh[rb]) for Uh in cf.U_hat]   # (q, d, d)
        score = sum(2.0 * Pb.reshape(q, -1) @ f.T for Pb, f in zip(P, flat)) - sq[None, :]   # (q, n)
        excl = torch.from_numpy(Rc[:, :3])
        score.scatter_(1, excl, float("-inf"))
        target = score.gather(1, rt[:, None])
        rank_chart[s:s + q] = (1 + (score > target).sum(1)).numpy()
        top_chart[s:s + q] = torch.topk(score, topk, dim=1).indices.numpy()

        sb = S_b[torch.from_numpy(ub_inv[s:s + q])].clone()
        sb.scatter_(1, excl, float("-inf"))
        rank_only_b[s:s + q] = (1 + (sb > sb.gather(1, rt[:, None])).sum(1)).numpy()

        # BCH gap: log P vs the additive log-chart analogy
        logP = [logm_real(Pb)[0] for Pb in P]
        num = sum(_fro(lp - (Lb[rs] - Lb[ra] + Lb[rb])) ** 2 for lp, Lb in zip(logP, cf.L))
        den = sum(_fro(lp) ** 2 for lp in logP)
        bch[s:s + q] = torch.sqrt(num / den).numpy()

        sel = np.flatnonzero(rr_sel[s:s + q])
        if sel.size and rerank > 0:
            ts = torch.from_numpy(sel)
            pool = torch.topk(score[ts], rerank, dim=1).indices                     # (qs, R)
            resid = torch.zeros(pool.shape, dtype=_F64)
            for Pb, Uh in zip(P, cf.U_hat):
                Mx = _pmap(lambda A_, B_: torch.linalg.solve(A_, B_, left=False),
                           Uh[pool], Pb[ts][:, None].expand(-1, rerank, -1, -1))
                Lx, _, okx = logm_real(Mx)                                          # log(P U_x^{-1})
                resid += torch.where(okx, _fro(Lx) ** 2, torch.full_like(resid, float("inf")))
            order = torch.argsort(resid, dim=1)
            for j, i in enumerate(sel):
                pj = pool[j].numpy()
                top_rerank[s + i] = [int(x) for x in pj[order[j, :topk].numpy()]]
                hit = np.flatnonzero(pj == Rc[i, 3])
                in_pool[s + i] = bool(hit.size)
                if hit.size and np.isfinite(float(resid[j, hit[0]])):
                    rank_rerank[s + i] = int(1 + (resid[j] < resid[j, hit[0]]).sum())

    idx = cf.ids
    for k in range(Q):
        out["questions"].append({
            "category": cats[k], "a": int(idx[R[k, 0]]), "a_star": int(idx[R[k, 1]]),
            "b": int(idx[R[k, 2]]), "b_star": int(idx[R[k, 3]]),
            "rank_chart": int(rank_chart[k]), "rank_only_b": int(rank_only_b[k]),
            "reranked": bool(rr_sel[k]), "rank_rerank": rank_rerank[k], "in_rerank_pool": in_pool[k],
            "top_chart": [int(idx[x]) for x in top_chart[k]],
            "top_rerank": None if top_rerank[k] is None else [int(idx[x]) for x in top_rerank[k]],
            "bch_gap": float(bch[k]),
        })

    def _block(mask: np.ndarray) -> Dict[str, Any]:
        sel = np.flatnonzero(mask & rr_sel)
        rr = [rank_rerank[k] for k in sel]
        return {"chart": _rank_metrics(rank_chart[mask].tolist()),
                "chart_reranked": _rank_metrics(rank_chart[sel].tolist()),      # paired with "rerank"
                "only_b": _rank_metrics(rank_only_b[mask].tolist()),
                "rerank": {**_rank_metrics(rr), "in_pool": float(np.mean([bool(in_pool[k]) for k in sel]))
                           if sel.size else None},
                "bch_gap_median": float(np.nanmedian(bch[mask])) if np.isfinite(bch[mask]).any() else None}

    cat_arr = np.array(cats)
    out["overall"] = _block(np.ones(Q, dtype=bool))
    out["per_category"] = {c: _block(cat_arr == c) for c in sorted(set(cats))}
    out["bch_gap"] = _summary(bch)
    out["rerank_pool"] = rerank
    out["n_reranked"] = int(rr_sel.sum())
    return out


def _derangement(m: int, rng: np.random.Generator) -> np.ndarray:
    r"""Uniform random permutation of ``range(m)`` without fixed points (rejection sampling, m >= 2)."""
    while True:
        p = rng.permutation(m)
        if not (p == np.arange(m)).any():
            return p


def relation_transports(
    cf:        CanonicalFrames,
    relations: Mapping[str, Sequence[Tuple[int, int]]],
    rng:       np.random.Generator,
    n_null:    int = 500,

    *,
    min_pairs:   int = 3,
    max_samples: int = 20000,
) -> Dict[str, Any]:
    r"""Relation transports :math:`T_i = \hat U_{a^*_i} \hat U_{a_i}^{-1}` and their consistency.

    For each relation with at least ``min_pairs`` resolved pairs, all cross transports
    :math:`T_{ij} = \hat U_{a^*_j}\hat U_{a_i}^{-1}` are formed once (the true pairs are the diagonal)
    and logged. Consistency = mean pairwise cosine of the flattened :math:`\log T_i` (block-pooled);
    the null draws ``n_null`` derangements :math:`\pi` (same words, permuted partners
    :math:`(a_i, a^*_{\pi(i)})`) and reports ``z = (obs - mean)/std`` and the one-sided permutation
    ``p = (1 + #\{null \ge obs\})/(1 + n_null)``. RotatE-style spectra: eigen-phases
    :math:`\arg\lambda` and :math:`\log|\lambda|` of the true :math:`T_i` versus the off-diagonal
    (random-partner) transports, pooled over blocks (``max_samples`` cap each).
    """
    out: Dict[str, Any] = {}
    for name, pairs in relations.items():
        rows = [(int(a), int(s)) for a, s in (cf.rows([p[0], p[1]]) for p in pairs) if a >= 0 and s >= 0 and a != s]
        rows = list(dict.fromkeys(rows))
        m = len(rows)
        if m < min_pairs:
            out[name] = {"status": "skipped", "n_pairs": m,
                         "reason": f"fewer than {min_pairs} resolved pairs in the frame chart"}
            continue
        ra = torch.tensor([r[0] for r in rows])
        rs = torch.tensor([r[1] for r in rows])
        vecs, ok, lam = [], torch.ones(m, m, dtype=torch.bool), []
        for Uh in cf.U_hat:
            d = Uh.shape[-1]
            T = _pmap(lambda A_, B_: torch.linalg.solve(A_, B_, left=False),
                      Uh[ra][:, None].expand(m, m, d, d), Uh[rs][None, :].expand(m, m, d, d))   # T_ij = U_{a*_j} U_{a_i}^{-1}
            LT, _, okT = logm_real(T)
            vecs.append(LT.reshape(m, m, -1))
            ok &= okT
            lam.append(_eigvals(T.reshape(m * m, d, d)).reshape(m, m, d))
        V = torch.cat(vecs, dim=-1).reshape(m * m, -1)                           # (m^2, F), NaN rows = failed logs
        norms = torch.linalg.vector_norm(V, dim=1, keepdim=True)
        unit = torch.where(torch.isfinite(norms) & (norms > 0), V / norms, torch.zeros_like(V))
        G = (unit @ unit.T).numpy()                                              # (m^2, m^2) cosines
        okf = ok.reshape(-1).numpy()

        def _consistency(perm: np.ndarray) -> float:
            idx = np.arange(m) * m + perm
            idx = idx[okf[idx]]
            k = idx.size
            if k < 2:
                return float("nan")
            C = G[np.ix_(idx, idx)]
            return float((C.sum() - np.trace(C)) / (k * (k - 1)))

        obs = _consistency(np.arange(m))
        null = np.array([_consistency(_derangement(m, rng)) for _ in range(int(n_null))])
        null_f = null[np.isfinite(null)]
        mu0 = float(null_f.mean()) if null_f.size else float("nan")
        sd0 = float(null_f.std(ddof=1)) if null_f.size > 1 else float("nan")
        lam_all = torch.cat(lam, dim=-1)                                         # (m, m, sum d)
        diag = torch.eye(m, dtype=torch.bool)
        true_l, rand_l = lam_all[diag].reshape(-1), lam_all[~diag].reshape(-1)

        def _cap(z: Tensor) -> Tensor:
            if z.numel() <= max_samples:
                return z
            return z[torch.from_numpy(rng.choice(z.numel(), size=max_samples, replace=False))]

        true_l, rand_l = _cap(true_l), _cap(rand_l)
        diag_idx = np.arange(m) * m + np.arange(m)
        out[name] = {
            "status": "ok", "n_pairs": m, "n_ok": int(okf[diag_idx].sum()),
            "consistency": obs, "null_mean": mu0, "null_std": sd0,
            "z": (obs - mu0) / sd0 if sd0 and np.isfinite(sd0) and sd0 > 0 else None,
            "p": float((1 + np.sum(null_f >= obs)) / (1 + null_f.size)) if null_f.size else None,
            "log_norm_mean": float(np.nanmean(torch.linalg.vector_norm(V, dim=1).numpy()[diag_idx])),
            "phase_true": true_l.angle().numpy(), "log_mod_true": torch.log(true_l.abs()).numpy(),
            "phase_random": rand_l.angle().numpy(), "log_mod_random": torch.log(rand_l.abs()).numpy(),
        }
    return out


# ----------------------------------------------------------------------------------------------
# Positional frames
# ----------------------------------------------------------------------------------------------

def _grouped_r2(F: np.ndarray, groups: np.ndarray) -> float:
    r"""Variance-weighted :math:`R^2 = 1 - \sum_f \mathrm{SS}_{\mathrm{within}}(f) / \sum_f \mathrm{SS}_{\mathrm{total}}(f)`
    of the features ``F`` (N, n_f) grouped by the integer labels ``groups`` (N,)."""
    g = np.unique(groups, return_inverse=True)[1]
    cnt = np.bincount(g).astype(np.float64)
    means = np.stack([np.bincount(g, weights=F[:, j]) for j in range(F.shape[1])], axis=1) / cnt[:, None]
    within = ((F - means[g]) ** 2).sum()
    total = ((F - F.mean(axis=0)) ** 2).sum()
    return float(1.0 - within / total) if total > 0 else float("nan")


def _ss_columns(F: Tensor, labels: Tensor, n_groups: int) -> Tuple[Tensor, Tensor]:
    r"""Per-column within-group and total sums of squares of ``F`` (N, n_f) grouped by ``labels`` (N,):

    .. math::  \mathrm{SS}_{\mathrm{within}}(f) = \sum_i F_{if}^2 - \sum_g \frac{(\sum_{i \in g} F_{if})^2}{n_g},
               \qquad \mathrm{SS}_{\mathrm{total}}(f) = \sum_i F_{if}^2 - \frac{(\sum_i F_{if})^2}{N}.

    Both are invariant under a constant shift of each column, so callers pass roughly centered ``F``.
    """
    cnt = torch.bincount(labels, minlength=n_groups).to(F.dtype)                    # (G,)
    gs = torch.zeros(n_groups, F.shape[1], dtype=F.dtype).index_add_(0, labels, F)  # (G, n_f) group sums
    sq = (F * F).sum(0)                                                             # (n_f,)
    nz = cnt > 0
    return sq - (gs[nz] ** 2 / cnt[nz, None]).sum(0), sq - F.sum(0) ** 2 / F.shape[0]


def _r2_from_columns(within: Tensor, total: Tensor, cols: Sequence[slice]) -> np.ndarray:
    r"""``[pooled, block_0, ..., block_{B-1}]`` variance-weighted :math:`R^2` from per-column sums of squares."""
    out = np.full(1 + len(cols), np.nan)
    for j, sl in enumerate([slice(None)] + list(cols)):
        t = float(total[sl].sum())
        if t > 0:
            out[j] = 1.0 - float(within[sl].sum()) / t
    return out


def _permuted_pair_rows(perm: np.ndarray, mm: np.ndarray, nn: np.ndarray, n_all: int) -> np.ndarray:
    r"""Rows of the stacked features ``[F(R); F(R^{-1})]`` that hold the relabeled pairs.

    Relabeling the positional frames by ``perm`` (:math:`V'_p = V_{\pi(p)}`) gives
    :math:`R'_{mn} = V_{\pi(m)} V_{\pi(n)}^{-1}`: the stored pair :math:`(\pi(m), \pi(n))` when
    :math:`\pi(m) > \pi(n)`, else the inverse :math:`R_{\pi(n)\pi(m)}^{-1}`. The pair :math:`(h, l)`,
    :math:`h > l`, is row :math:`h(h-1)/2 + l` of ``np.tril_indices(P, k=-1)``; inverses sit ``n_all``
    rows further down.
    """
    a, b = perm[mm], perm[nn]
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return hi * (hi - 1) // 2 + lo + n_all * (a < b)


def _positional_r2_null(
    F2:      Tensor,               # (2 N_all, n_f) features of R_mn (rows < N_all), then of R_mn^{-1}
    mm:      np.ndarray,           # (N,) tested pairs, m > n
    nn:      np.ndarray,
    movable: np.ndarray,           # positions permuted among themselves (all others fixed)
    P:       int,
    cols:    Sequence[slice],      # feature columns of each block
    n_perm:  int,
    rng:     np.random.Generator,
) -> Dict[str, np.ndarray]:
    r"""Position-permutation null of :math:`R^2_{\mathrm{offset}}` and :math:`R^2_{\mathrm{absolute}}`.

    Each draw relabels the positions in ``movable`` uniformly at random, gathers the features of the
    relabeled pairs from ``F2`` (:func:`_permuted_pair_rows`; no new eigendecomposition) and
    recomputes both groupings with the labels of the tested pairs held fixed. The draw keeps the set
    of positional frames and destroys any dependence on position order, so it is the null for both
    groupings; the i.i.d. value :math:`(G-1)/(N-1)` does not apply to the grouping by :math:`m`,
    whose pairs share the frame :math:`V_m`. Returns ``{"offset", "absolute"}`` arrays of shape
    (n_perm, 1 + B): pooled, then per block.
    """
    n_all = F2.shape[0] // 2
    lab_off, lab_abs = torch.from_numpy(mm - nn), torch.from_numpy(np.ascontiguousarray(mm))
    out = {"offset": np.empty((n_perm, 1 + len(cols))), "absolute": np.empty((n_perm, 1 + len(cols)))}
    perm = np.arange(P)
    for r in range(n_perm):
        perm[movable] = movable[rng.permutation(movable.size)]
        Fp = F2.index_select(0, torch.from_numpy(_permuted_pair_rows(perm, mm, nn, n_all)))   # (N, n_f)
        for key, lab in (("offset", lab_off), ("absolute", lab_abs)):
            out[key][r] = _r2_from_columns(*_ss_columns(Fp, lab, P), cols)
    return out


def _null_summary(obs: float, null: np.ndarray) -> Dict[str, Any]:
    r"""Permutation-null summary of an observed statistic: ``n_perm``, ``mean``, ``std``, ``q05``, ``q95``,
    ``z = (obs - mean)/std`` and the one-sided ``p = (1 + #{null >= obs})/(1 + n_perm)`` (a 1e-12
    guard counts rounding ties as ties)."""
    v = np.asarray(null, dtype=np.float64)
    v = v[np.isfinite(v)]
    out: Dict[str, Any] = {"n_perm": int(v.size), "mean": None, "std": None, "q05": None, "q95": None,
                           "z": None, "p": None}
    if not v.size or not np.isfinite(obs):
        return out
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    out.update(mean=mean, std=sd, q05=float(np.quantile(v, 0.05)), q95=float(np.quantile(v, 0.95)),
               z=(obs - mean) / sd if np.isfinite(sd) and sd > 0 else None,
               p=float((1 + np.sum(v >= obs - 1e-12)) / (1 + v.size)))
    return out


def _by_offset(values: np.ndarray, offset: np.ndarray, keep: np.ndarray, offs: np.ndarray) -> Tuple[np.ndarray, ...]:
    r"""Mean, 25th and 75th percentile of ``values[keep]`` per offset (NaN for empty offsets)."""
    out = np.full((3, offs.size), np.nan)
    for j, o in enumerate(offs):
        v = values[keep & (offset == o)]
        if v.size:
            out[:, j] = (v.mean(), np.quantile(v, 0.25), np.quantile(v, 0.75))
    return out[0], out[1], out[2]


def positional_frames(
    run:           Any,
    max_positions: Optional[int] = None,

    *,
    outlier_z: float                         = 5.0,
    n_perm:    int                           = 200,
    rng:       Optional[np.random.Generator] = None,
) -> Optional[Dict[str, Any]]:
    r"""Relative-position structure of the positional frames :math:`V_p = e^{Y_p}`.

    For every pair :math:`m > n`, :math:`R_{mn} = V_m V_n^{-1}`; only its conjugacy class is
    meaningful (redundancies (i)/(ii) conjugate it), so the features are the sorted
    :math:`\log|\lambda|` and sorted :math:`|\arg\lambda|` of :math:`R_{mn}` per block. Relative-position
    test: :math:`R^2_{\mathrm{offset}} = 1 - \mathbb{E}[\text{within-offset var}]/\text{total var}`
    (variance-weighted over features; grouping by :math:`m - n`) and :math:`R^2_{\mathrm{absolute}}`
    (grouping by :math:`m`), each against its own position-permutation null (``n_perm`` draws of
    :func:`_positional_r2_null`; ``null`` and ``per_block[*]["null_offset" | "null_absolute"]`` hold
    mean, sd, 5/95 % quantiles, ``z`` and the one-sided ``p``). A one-parameter subgroup
    :math:`Y_p = pA` gives :math:`R^2_{\mathrm{offset}} = 1`. The permuted features need no new
    eigendecomposition: :math:`R_{nm} = R_{mn}^{-1}` has eigenvalues :math:`1/\lambda`, so its sorted
    :math:`\log|\lambda|` is the negated reversed vector and its sorted :math:`|\arg\lambda|` is
    unchanged. ``r2_chance`` :math:`= (G-1)/(N-1)` is kept only as the analytic i.i.d. reference of
    the offset grouping; it is not a null for the grouping by :math:`m`. Spectral log-norm
    :math:`s_{mn} = \sqrt{\sum_i (\log|\lambda_i|)^2 + (\arg\lambda_i)^2}` (block-pooled; symmetric,
    zero diagonal) is returned as a (P, P) matrix. None if the run has no positional frames.

    Positions whose raw :math:`\|Y_p\|_F` exceeds the median by more than ``outlier_z`` robust
    standard deviations (1.4826 MAD) are listed in ``outlier_positions``, and the ``*_interior``
    entries repeat the test on the pairs that touch none of them (a variance-weighted :math:`R^2` is
    otherwise dominated by a single outlying position), against a null that permutes only the
    interior positions (``null_interior``). Both versions are always reported.
    """
    if getattr(run.cfg, "pos_phi", "none") == "none":
        return None
    P = int(max_positions if max_positions is not None else run.cfg.max_seq_len)
    Y_blocks = run.pos_phi_blocks(np.arange(P, dtype=np.int64))
    if Y_blocks is None:
        return None
    gen = rng if rng is not None else np.random.default_rng(0)
    mm, nn = np.tril_indices(P, k=-1)                                        # m > n
    offset = mm - nn
    y_norm = np.sqrt(sum((_fro(_t(Yb)) ** 2).numpy() for Yb in Y_blocks))   # (P,) raw chart
    med = float(np.median(y_norm))
    mad = 1.4826 * float(np.median(np.abs(y_norm - med)))
    outlier = (y_norm - med > outlier_z * mad) if mad > 0 else np.zeros(P, dtype=bool)
    interior = ~(outlier[mm] | outlier[nn])
    s2 = np.zeros(mm.size)
    feats, feats_inv, cols, per_block, s_block = [], [], [], [], []
    c0 = 0
    for Yb in Y_blocks:
        V = _expm(_t(Yb))                                                    # (P, d, d)
        Vinv = torch.linalg.inv(V)
        lam = torch.empty(mm.size, V.shape[-1], dtype=torch.complex128)
        for s in range(0, mm.size, _CHUNK):
            sl = slice(s, s + _CHUNK)
            lam[sl] = _pmap(torch.linalg.eigvals, V[torch.from_numpy(mm[sl])] @ Vinv[torch.from_numpy(nn[sl])])
        log_mod = torch.log(lam.abs()).numpy()
        phase = lam.angle().numpy()
        lm_sorted, ph_sorted = np.sort(log_mod, axis=1), np.sort(np.abs(phase), axis=1)
        Fb = np.concatenate([lm_sorted, ph_sorted], axis=1)                  # (N, 2d) features of R_mn
        feats.append(Fb)
        feats_inv.append(np.concatenate([-lm_sorted[:, ::-1], ph_sorted], axis=1))   # of R_mn^{-1}
        cols.append(slice(c0, c0 + Fb.shape[1]))
        c0 += Fb.shape[1]
        sb2 = (log_mod ** 2 + phase ** 2).sum(1)
        s2 += sb2
        s_block.append(np.sqrt(sb2))
        per_block.append({"r2_offset": _grouped_r2(Fb, offset), "r2_absolute": _grouped_r2(Fb, mm)})
    F = np.concatenate(feats, axis=1)
    s = np.sqrt(s2)
    S = np.zeros((P, P))
    S[mm, nn] = s
    S[nn, mm] = s
    offs = np.arange(1, P)
    everything = np.ones(mm.size, dtype=bool)
    s_mean, s_q25, s_q75 = _by_offset(s, offset, everything, offs)
    si_mean, si_q25, si_q75 = _by_offset(s, offset, interior, offs)
    n_int = int(interior.sum())
    g_int = int(np.unique(offset[interior]).size) if n_int else 0
    ok_int = n_int > g_int + 1
    out: Dict[str, Any] = {
        "positions": P, "n_pairs": int(mm.size),
        "r2_offset": _grouped_r2(F, offset), "r2_absolute": _grouped_r2(F, mm),
        "r2_chance": float((P - 2) / (mm.size - 1)) if mm.size > 1 else float("nan"),
        "per_block": per_block,
        "outlier_positions": np.flatnonzero(outlier), "outlier_z": float(outlier_z),
        "n_pairs_interior": n_int,
        "r2_offset_interior": _grouped_r2(F[interior], offset[interior]) if ok_int else float("nan"),
        "r2_absolute_interior": _grouped_r2(F[interior], mm[interior]) if ok_int else float("nan"),
        "r2_chance_interior": float((g_int - 1) / (n_int - 1)) if ok_int else float("nan"),
        "null": None, "null_interior": None,
        "spectral_norm": S,
        "offsets": offs,
        "s_mean_by_offset": s_mean, "s_q25_by_offset": s_q25, "s_q75_by_offset": s_q75,
        "s_mean_by_offset_interior": si_mean, "s_q25_by_offset_interior": si_q25,
        "s_q75_by_offset_interior": si_q75,
        "s_block_mean_by_offset_interior": np.stack([_by_offset(sb, offset, interior, offs)[0] for sb in s_block]),
        "y_norm": y_norm,
    }
    if n_perm > 0 and mm.size > 1:
        # rows [0, N): features of R_mn; rows [N, 2N): of R_mn^{-1}; one shift per column (SS unchanged)
        F2 = torch.from_numpy(np.concatenate([F, np.concatenate(feats_inv, axis=1)]) - F.mean(axis=0))
        del feats, feats_inv
        draws = _positional_r2_null(F2, mm, nn, np.arange(P), P, cols, int(n_perm), gen)
        out["null"] = {key: _null_summary(out[f"r2_{key}"], draws[key][:, 0]) for key in ("offset", "absolute")}
        for b, pb in enumerate(per_block):
            for key in ("offset", "absolute"):
                pb[f"null_{key}"] = _null_summary(pb[f"r2_{key}"], draws[key][:, 1 + b])
        if not outlier.any():
            out["null_interior"] = out["null"]                               # interior = all pairs
        elif ok_int:
            draws = _positional_r2_null(F2, mm[interior], nn[interior], np.flatnonzero(~outlier), P, cols,
                                        int(n_perm), gen)
            out["null_interior"] = {key: _null_summary(out[f"r2_{key}_interior"], draws[key][:, 0])
                                    for key in ("offset", "absolute")}
    return out


# ----------------------------------------------------------------------------------------------
# Routing covariance (called by the sigma section)
# ----------------------------------------------------------------------------------------------

def routing_covariance_summary(run: Any, ctx: AnalysisContext, ids: Sequence[int]) -> Dict[str, Any]:
    r"""Frame-induced (routing) covariance :math:`C^b_v = e^{-X^b_v}\, \mathrm{diag}(\mathrm{var}^b_v)\, e^{-X^{bT}_v}`.

    Exact identity per block: :math:`\log\det C^b_v = \sum_{k \in b} \log\mathrm{var}_{vk} - 2\,\mathrm{tr}\, X^b_v`,
    split into the :math:`\Sigma`-part and the frame part; the variance of :math:`\log\det C` is decomposed
    into :math:`\mathrm{Var}(\Sigma)`, :math:`\mathrm{Var}(\text{frame})` and :math:`2\,\mathrm{Cov}` shares.
    Also: Spearman of each part with log count, the eigen-spread :math:`\log(\lambda_{\max}/\lambda_{\min})`
    of :math:`C^b_v`, and the relative gap :math:`\|C_v - \bar c_v (U_v^T U_v)^{-1}\|_F / \|C_v\|_F`,
    :math:`\bar c_v` = mean variance in the block (block-pooled Frobenius norms).

    Gauge status: :math:`\log\det C` is invariant up to one global constant per block (so its
    variances and rank correlations are invariant); the eigen-spread and the gap are measured in the
    trained trivialization (as in the v2 manuscript measurement).

    Returned arrays: ``ids``, ``log_det_C``, ``sigma_part``, ``frame_part`` (n,), their ``*_block``
    versions (n, B), ``log_cond_block`` (n, B), ``rel_gap`` (n,), ``rel_gap_block`` (n, B); plus
    ``identity_max_abs_error``, ``variance_shares``, ``spearman_log_count``, ``summary``, ``note``.
    """
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    n, slices = ids.size, list(run.block_slices)
    B = len(slices)
    sigma_b, frame_b, logdet_b = np.empty((n, B)), np.empty((n, B)), np.empty((n, B))
    cond_b, gap_num_b, cnorm_b = np.empty((n, B)), np.empty((n, B)), np.empty((n, B))
    for s in range(0, n, _CHUNK):
        sl_rows = slice(s, s + _CHUNK)
        lv = np.asarray(run.log_var(ids[sl_rows]), dtype=np.float64)
        Xc = run.phi_blocks(ids[sl_rows])
        for b, sl in enumerate(slices):
            Xb = _t(Xc[b])
            Einv = torch.matrix_exp(-Xb)                                            # e^{-X} = U^{-1}
            lvb = _t(lv[:, sl])
            var = torch.exp(lvb)                                                     # (c, d)
            C = (Einv * var[:, None, :]) @ Einv.transpose(-1, -2)                   # (c, d, d) SPD
            sigma_b[sl_rows, b] = lvb.sum(1).numpy()
            frame_b[sl_rows, b] = (-2.0 * torch.diagonal(Xb, dim1=-2, dim2=-1).sum(-1)).numpy()
            logdet_b[sl_rows, b] = torch.linalg.slogdet(C).logabsdet.numpy()
            ev = torch.linalg.eigvalsh(C)
            cond_b[sl_rows, b] = torch.log(ev[:, -1] / ev[:, 0]).numpy()
            cbar = var.mean(1)
            dev = (Einv * (var - cbar[:, None])[:, None, :]) @ Einv.transpose(-1, -2)  # C - cbar U^{-1}U^{-T}
            gap_num_b[sl_rows, b] = (_fro(dev) ** 2).numpy()
            cnorm_b[sl_rows, b] = (_fro(C) ** 2).numpy()
    sigma_part, frame_part = sigma_b.sum(1), frame_b.sum(1)
    log_det = sigma_part + frame_part
    ident = float(np.max(np.abs(logdet_b - (sigma_b + frame_b))))
    tot = np.var(log_det)
    shares = ({"sigma": float(np.var(sigma_part) / tot), "frame": float(np.var(frame_part) / tot),
               "cross": float(2.0 * np.cov(sigma_part, frame_part, bias=True)[0, 1] / tot)}
              if tot > 0 else {"sigma": None, "frame": None, "cross": None})
    lc = ctx.log_count(ids)
    spear = ({"log_det_C": _spearman(log_det, lc), "sigma_part": _spearman(sigma_part, lc),
              "frame_part": _spearman(frame_part, lc)} if lc is not None else None)
    rel_gap = np.sqrt(gap_num_b.sum(1) / cnorm_b.sum(1))
    return {
        "ids": ids, "log_det_C": log_det, "sigma_part": sigma_part, "frame_part": frame_part,
        "sigma_part_block": sigma_b, "frame_part_block": frame_b, "log_cond_block": cond_b,
        "rel_gap": rel_gap, "rel_gap_block": np.sqrt(gap_num_b / cnorm_b),
        "identity_max_abs_error": ident, "variance_shares": shares, "spearman_log_count": spear,
        "summary": {"log_det_C": _summary(log_det), "sigma_part": _summary(sigma_part),
                    "frame_part": _summary(frame_part), "rel_gap": _summary(rel_gap),
                    "log_cond": [_summary(cond_b[:, k]) for k in range(B)]},
        "note": ("log det C = sum log var - 2 tr X is exact and gauge-invariant up to one global constant "
                 "per block; eigen-spread and the gap to cbar (U^T U)^{-1} are trained-trivialization "
                 "quantities."
                 + ("" if lc is not None else " Counts unavailable: no frequency correlations.")),
    }


# ----------------------------------------------------------------------------------------------
# Section driver, figures and tables
# ----------------------------------------------------------------------------------------------

def _opt(ctx: AnalysisContext, key: str) -> Any:
    return (ctx.options or {}).get(key, PHI_DEFAULTS[key])


def _diagonal_note(status: str, reason: str) -> str:
    r"""Report sentence on what ``D0`` is, given :func:`vfe3.semantics.common.diagonal_redundancy_status`."""
    if status == "exact":
        return (f"The global positive-diagonal rescaling is an exact redundancy of this configuration ({reason}), "
                "so D0 is a gauge fixing.")
    if status == "broken":
        return (f"The global positive-diagonal rescaling is not a redundancy of this model ({reason}); D0 is a "
                "documented unit normalization, not a gauge fixing, and invariance under that rescaling is a "
                "property of the chart, not of the model.")
    return (f"The global positive-diagonal rescaling is assumed, not verified, to be a redundancy of this "
            f"configuration ({reason}); D0 is a gauge fixing only under that assumption.")


def _barycenter_qualifier(cf: CanonicalFrames) -> str:
    r"""``""`` when every block's barycenter is the fixed point, else the :math:`O(r)` qualification of the
    chart's invariance statements (``r`` = largest barycenter residual)."""
    if cf.diagnostics["barycenter_converged_all"]:
        return ""
    return f" only up to O(r), r = {cf.diagnostics['barycenter_residual_max']:.1e},"


def _rerank_rank_text(q: Mapping[str, Any], pool: int) -> str:
    r"""Log-rerank rank of one question's answer, or why it has none."""
    if q["rank_rerank"] is not None:
        return str(q["rank_rerank"])
    if not q["reranked"]:
        return "not reranked"
    if q.get("in_rerank_pool") is False:
        return f"outside top-{pool} pool"
    return "no real log for the answer"


def _r2_vs_null(obs: Optional[float], nd: Optional[Mapping[str, Any]]) -> str:
    r"""``"0.054 (null 0.010, q95 0.011, p = 0.00498)"``; the bare value when there is no null."""
    if obs is None or not np.isfinite(obs):
        return "–"
    if not nd or nd.get("mean") is None:
        return f"{obs:.3f}"
    return f"{obs:.3f} (null {nd['mean']:.3f}, q95 {nd['q95']:.3f}, p = {nd['p']:.3g})"


def _fmt3(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.3f}"


def _find_curated(ga: Optional[Dict[str, Any]], ctx: AnalysisContext,
                  words: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    r"""The curated question ``words[0]:words[1]::words[2]:?`` (matched on display strings), if covered."""
    if not ga:
        return None
    for q in ga["questions"]:
        if (ctx.display(q["a"]), ctx.display(q["a_star"]), ctx.display(q["b"])) == words:
            return q
    return None


def _hist_step(ax: Any, values: Any, bins: Any, label: str, color: str) -> None:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size:
        ax.hist(v, bins=bins, histtype="step", density=True, label=label, color=color, linewidth=1.2)


def _plot_overview(ctx: AnalysisContext, cf: CanonicalFrames, comm: Dict[str, Any],
                   comm_canon: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    B = len(cf.U_hat)
    dims = sorted({int(Uh.shape[-1]) for Uh in cf.U_hat})
    color = [OKABE_ITO[b % len(OKABE_ITO)] for b in range(B)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)

    ax = axes[0, 0]
    nb = comm["norm_block"]
    bins = np.linspace(0.0, float(np.nanmax(nb)) * 1.05, 50)
    for b in range(B):
        _hist_step(ax, nb[:, b], bins, f"block {b}", color[b])
    ax.set_xlabel(r"$\|X_v^b\|_F$ (raw chart)")
    ax.set_ylabel("density")
    ax.set_title("Token algebra-element norms")
    ax.legend(ncol=1 if B <= 4 else 2)
    panel_label(ax, "a")

    ax = axes[0, 1]
    rb, rc = comm["ratio_block"].reshape(-1), comm_canon["ratio_block"].reshape(-1)
    hi = float(np.nanmax(np.concatenate([rb, rc, [np.sqrt(2.0 / dims[0])]]))) * 1.1
    bins = np.linspace(0.0, hi, 50)
    _hist_step(ax, rb, bins, "raw chart", OKABE_ITO[0])
    _hist_step(ax, rc, bins, "canonical chart", OKABE_ITO[1])
    for j, d in enumerate(dims):
        ax.axvline(np.sqrt(2.0 / d), color="0.4", linestyle="--", linewidth=0.8,
                   label=r"iid Gaussian $\sqrt{2/d}$" if j == 0 else None)
    ax.set_xlabel(r"$\|[X_a,X_b]\|_F / (\|X_a\|_F\|X_b\|_F)$ per block")
    ax.set_title("Commutator ratio (random pairs)")
    ax.legend(loc="upper left")
    panel_label(ax, "b")

    ax = axes[0, 2]
    gb, gc = comm["bch_gap_block"].reshape(-1), comm_canon["bch_gap_block"].reshape(-1)
    both = np.concatenate([gb, gc])
    both = both[np.isfinite(both)]
    bins = np.linspace(0.0, float(both.max()) * 1.1 if both.size and both.max() > 0 else 1.0, 50)
    _hist_step(ax, gb, bins, "raw chart", OKABE_ITO[0])
    _hist_step(ax, gc, bins, "canonical chart", OKABE_ITO[1])
    ax.set_xlabel(r"$\|\log(e^{X_a}e^{-X_b}) - (X_a - X_b)\|_F / \|X_a - X_b\|_F$")
    ax.set_title("BCH gap of additive frame arithmetic")
    ax.legend()
    panel_label(ax, "c")

    ax = axes[1, 0]
    r = np.concatenate(cf.residual) if cf.residual else np.zeros(0)
    r = np.log10(np.clip(r[np.isfinite(r)], 1e-17, None))
    if r.size:
        ax.hist(r, bins=40, color=OKABE_ITO[2])
    ax.set_xlabel(r"$\log_{10}\, \|e^{L} - \hat U\|_F / \|\hat U\|_F$")
    ax.set_ylabel("token blocks")
    ax.set_title(f"Canonical log chart (ok fraction {cf.diagnostics['ok_fraction']:.4f})")
    panel_label(ax, "d")

    ax = axes[1, 1]
    for b, bd in enumerate(cf.diagnostics["blocks"]):
        h = np.asarray(bd["barycenter"]["history"], dtype=np.float64)
        if h.size:
            ax.semilogy(np.arange(h.size), np.clip(h, 1e-18, None), marker="o", markersize=3,
                        color=color[b], label=f"block {b}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\overline{\log(m^{-1}U_a)}\|_F$")
    ax.set_title(f"Barycenter convergence ($n_{{ref}}$ = {cf.ref_ids.size})")
    ax.legend(ncol=1 if B <= 4 else 2)
    panel_label(ax, "e")

    ax = axes[1, 2]
    nL = np.stack([_fro(Lb).numpy() for Lb in cf.L], axis=1)
    finite = nL[np.isfinite(nL)]
    bins = np.linspace(0.0, float(finite.max()) * 1.05 if finite.size else 1.0, 50)
    for b in range(B):
        _hist_step(ax, nL[:, b], bins, f"block {b}", color[b])
    ax.set_xlabel(r"$\|L_v^b\|_F$ (canonical chart)")
    ax.set_ylabel("density")
    ax.set_title("Gauge-fixed log-frame norms")
    panel_label(ax, "f")
    return save_figure(fig, ctx, "phi", "overview")


_SCALAR_LABELS: Tuple[Tuple[str, str], ...] = (
    ("trace",      r"$\log\det \hat U_v$"),
    ("stretch",    r"stretch $\log(\sigma_{\max}/\sigma_{\min})$"),
    ("rms_log_sv", r"RMS $\log\sigma$"),
    ("rotation",   r"rotation RMS $|\arg\lambda|$"),
    ("sym_norm",   r"$\|\mathrm{sym}\, L\|_F$"),
    ("skew_norm",  r"$\|\mathrm{skew}\, L\|_F$"),
)


def _plot_invariants_vs_frequency(ctx: AnalysisContext, scalars: Dict[str, np.ndarray],
                                  log_count: np.ndarray, spear: Dict[str, Dict[str, Any]]) -> List[Path]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    edges = np.quantile(log_count, np.linspace(0.0, 1.0, 13))
    for k, (ax, (key, label)) in enumerate(zip(axes.ravel(), _SCALAR_LABELS)):
        y = np.asarray(scalars[key], dtype=np.float64)
        keep = np.isfinite(y)
        ax.hexbin(log_count[keep], y[keep], gridsize=40, bins="log", cmap="viridis", mincnt=1, linewidths=0)
        centers, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = keep & (log_count >= lo) & (log_count <= hi)
            if sel.sum() >= 5:
                centers.append(0.5 * (lo + hi))
                med.append(float(np.median(y[sel])))
        ax.plot(centers, med, color=OKABE_ITO[3], linewidth=1.6, marker="o", markersize=3,
                label="binned median")
        rho = spear[key]["rho"]
        ax.set_title(label + ("" if rho is None else f"   $\\rho_s$ = {rho:.2f}"))
        ax.set_xlabel(r"$\log(1 + \mathrm{count})$")
        panel_label(ax, "abcdef"[k])
    axes[0, 0].legend(loc="best")
    fig.suptitle("Gauge-fixed frame invariants vs training frequency", fontsize=10)
    return save_figure(fig, ctx, "phi", "invariants_vs_frequency")


_GA_METHODS: Tuple[Tuple[str, str], ...] = (
    ("chart", "group (chart), all questions"), ("chart_reranked", "group (chart), reranked subset"),
    ("rerank", "group (log rerank), reranked subset"), ("only_b", "ONLY-B baseline, all questions"))


def _plot_group_analogies(ctx: AnalysisContext, ga: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    cats = list(ga["per_category"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(3.6, 0.32 * len(cats) + 1.6)),
                             constrained_layout=True, gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})
    h = 0.2
    ax = axes[0]
    y = np.arange(len(cats))
    for j, (key, label) in enumerate(_GA_METHODS):
        vals = [ga["per_category"][c][key]["acc"] for c in cats]
        ax.barh(y + (j - 1.5) * h, [np.nan if v is None else v for v in vals], height=h,
                color=OKABE_ITO[j], label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={ga['per_category'][c]['chart']['n']}; "
                        f"{ga['per_category'][c]['rerank']['n']} reranked)" for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("top-1 accuracy (query words excluded)")
    ax.set_title("Group analogies per category")
    panel_label(ax, "a")

    ax = axes[1]
    stats = ("acc", "hits5", "mrr")
    x = np.arange(len(stats))
    for j, (key, label) in enumerate(_GA_METHODS):
        vals = [ga["overall"][key][s] for s in stats]
        ax.bar(x + (j - 1.5) * h, [np.nan if v is None else v for v in vals], width=h,
               color=OKABE_ITO[j], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["acc@1", "hits@5", "MRR"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Overall (all: n = {ga['n_covered']})\nreranked subset: n = {ga['overall']['rerank']['n']}, "
                 f"pool {ga['rerank_pool']}")
    ax.legend(loc="upper right", fontsize=7)
    panel_label(ax, "b")

    ax = axes[2]
    g = np.array([q["bch_gap"] for q in ga["questions"]], dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size:
        ax.hist(g, bins=40, color=OKABE_ITO[4])
        ax.axvline(float(np.median(g)), color="0.2", linestyle="--", linewidth=0.9, label="median")
        ax.legend()
    ax.set_xlabel(r"$\|\log P - (L_{a^*} - L_a + L_b)\|_F / \|\log P\|_F$")
    ax.set_ylabel("questions")
    ax.set_title("BCH gap: group vs additive log-chart analogy")
    panel_label(ax, "c")
    return save_figure(fig, ctx, "phi", "group_analogies")


def _plot_relation_transports(ctx: AnalysisContext, rel: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    names = [k for k, v in rel.items() if v.get("status") == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, max(3.2, 0.3 * len(names) + 1.4)),
                             constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    ax = axes[0]
    y = np.arange(len(names))
    null_m = np.array([rel[k]["null_mean"] for k in names], dtype=np.float64)
    null_s = np.array([rel[k]["null_std"] for k in names], dtype=np.float64)
    ax.errorbar(null_m, y, xerr=2.0 * null_s, fmt="|", color="0.5", label=r"null mean $\pm 2$ sd")
    ax.scatter([rel[k]["consistency"] for k in names], y, color=OKABE_ITO[0], zorder=3, s=18,
               label="relation pairs")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} (n={rel[k]['n_pairs']})" for k in names])
    ax.invert_yaxis()
    ax.set_xlabel(r"mean pairwise cosine of $\log T_i$")
    ax.set_title("Relation-transport consistency")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    def _pool(key: str) -> np.ndarray:
        parts = [np.asarray(rel[k][key], dtype=np.float64) for k in names]
        return np.concatenate(parts) if parts else np.zeros(0)

    ax = axes[1]
    ph = np.abs(np.concatenate([_pool("phase_random"), _pool("phase_true")]))
    ph = ph[np.isfinite(ph)]
    lim = min(np.pi, 1.05 * float(ph.max())) if ph.size and ph.max() > 0 else np.pi
    bins = np.linspace(-lim, lim, 61)
    _hist_step(ax, _pool("phase_random"), bins, "random partners", "0.4")
    _hist_step(ax, _pool("phase_true"), bins, "relation pairs", OKABE_ITO[0])
    ax.set_xlabel(r"eigen-phase $\arg\lambda(T)$")
    ax.set_ylabel("density")
    ax.set_title("Transport eigen-phases (pooled)")
    ax.legend()
    panel_label(ax, "b")

    ax = axes[2]
    lm = np.concatenate([_pool("log_mod_random"), _pool("log_mod_true")])
    lm = lm[np.isfinite(lm)]
    bins = np.linspace(float(lm.min()), float(lm.max()), 61) if lm.size and np.ptp(lm) > 0 else 20
    _hist_step(ax, _pool("log_mod_random"), bins, "random partners", "0.4")
    _hist_step(ax, _pool("log_mod_true"), bins, "relation pairs", OKABE_ITO[0])
    ax.set_xlabel(r"$\log|\lambda(T)|$")
    ax.set_title("Transport eigen-moduli (pooled)")
    panel_label(ax, "c")
    return save_figure(fig, ctx, "phi", "relation_transports")


def _plot_positional(ctx: AnalysisContext, pos: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0, 0.9, 0.9]})
    S = pos["spectral_norm"]
    off_diag = S[~np.eye(S.shape[0], dtype=bool)]
    vmax = float(np.quantile(off_diag, 0.99)) if off_diag.size else None
    outl = [int(p) for p in pos["outlier_positions"]]
    excl = (f"excluding {len(outl)} outlier position{'s' if len(outl) > 1 else ''} (panel d)"
            if outl else "all pairs")

    ax = axes[0]
    im = ax.imshow(S, origin="lower", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.85, extend="max", label=r"$s_{mn}$ (clipped at 99th pct)")
    ax.set_xlabel("position $n$")
    ax.set_ylabel("position $m$")
    ax.set_title(r"Spectral log-norm of $e^{Y_m}e^{-Y_n}$")
    panel_label(ax, "a")

    ax = axes[1]
    o = pos["offsets"]
    ax.fill_between(o, pos["s_q25_by_offset_interior"], pos["s_q75_by_offset_interior"], color=OKABE_ITO[0],
                    alpha=0.25, linewidth=0)
    ax.plot(o, pos["s_mean_by_offset_interior"], color=OKABE_ITO[0], linewidth=1.5, label="pooled mean (IQR)")
    for b, curve in enumerate(pos["s_block_mean_by_offset_interior"]):
        ax.plot(o, curve, color=OKABE_ITO[(b + 1) % len(OKABE_ITO)], linewidth=0.8, alpha=0.8,
                label=f"block {b}")
    ax.set_xlabel("offset $m - n$")
    ax.set_ylabel(r"$s_{mn}$")
    ax.set_title(f"Spectral log-norm vs offset\n({excl})")
    ax.legend(fontsize=7, ncol=2)
    panel_label(ax, "b")

    ax = axes[2]
    null = pos.get("null") or {}
    labels = ["pooled"] + [f"block {b}" for b in range(len(pos["per_block"]))]
    r_off = [pos["r2_offset"]] + [x["r2_offset"] for x in pos["per_block"]]
    r_abs = [pos["r2_absolute"]] + [x["r2_absolute"] for x in pos["per_block"]]
    n_off = [null.get("offset")] + [x.get("null_offset") for x in pos["per_block"]]
    n_abs = [null.get("absolute")] + [x.get("null_absolute") for x in pos["per_block"]]
    if outl:
        null_int = pos.get("null_interior") or {}
        labels.insert(1, "pooled,\ninterior")
        r_off.insert(1, pos["r2_offset_interior"])
        r_abs.insert(1, pos["r2_absolute_interior"])
        n_off.insert(1, null_int.get("offset"))
        n_abs.insert(1, null_int.get("absolute"))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, r_off, width=0.4, color=OKABE_ITO[0], label=r"$R^2_{\mathrm{offset}}$ (by $m - n$)")
    ax.bar(x + 0.2, r_abs, width=0.4, color=OKABE_ITO[1], label=r"$R^2_{\mathrm{absolute}}$ (by $m$)")
    top = [v for v in r_off + r_abs if v is not None and np.isfinite(v)]
    band_label = "position-permutation null\n(mean, 5–95%)"
    for xs, nulls in ((x - 0.2, n_off), (x + 0.2, n_abs)):
        for xi, nd in zip(xs, nulls):
            if not nd or nd.get("mean") is None:
                continue
            ax.vlines(xi, nd["q05"], nd["q95"], color="0.15", linewidth=1.2, label=band_label)
            ax.hlines(nd["mean"], xi - 0.12, xi + 0.12, color="0.15", linewidth=1.2)
            top.append(nd["q95"])
            band_label = None
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    hi = max(top) if top else 1.0
    ax.set_ylim(0, 1.0 if hi <= 0 or hi > 0.85 else 1.15 * hi)
    ax.set_ylabel(r"$R^2$ of spectral features")
    ax.set_title("Relative-position test")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    panel_label(ax, "c")

    ax = axes[3]
    ax.plot(np.arange(pos["positions"]), pos["y_norm"], color=OKABE_ITO[2], linewidth=1.0)
    if outl:
        ax.scatter(outl, pos["y_norm"][outl], color=OKABE_ITO[3], s=14, zorder=3, label="outlier positions")
        ax.legend()
    ax.set_xlabel("position $p$")
    ax.set_ylabel(r"$\|Y_p\|_F$ (raw chart)")
    ax.set_title("Positional algebra-element norm")
    panel_label(ax, "d")
    return save_figure(fig, ctx, "phi", "positional")


def _curated_table(ctx: AnalysisContext, ga: Dict[str, Any]) -> str:
    rows = []
    for q in ga["questions"]:
        rows.append({
            "category": q["category"],
            "analogy": f"{ctx.display(q['a'])}:{ctx.display(q['a_star'])}::{ctx.display(q['b'])}:?",
            "answer": ctx.display(q["b_star"]),
            "rank (chart)": q["rank_chart"], "rank (log rerank)": q["rank_rerank"],
            "rank (ONLY-B)": q["rank_only_b"],
            "top-5 (chart)": ", ".join(ctx.display(t) for t in q["top_chart"]),
            "top-5 (log rerank)": ("–" if q["top_rerank"] is None
                                   else ", ".join(ctx.display(t) for t in q["top_rerank"])),
            "BCH gap": q["bch_gap"],
        })
    return markdown_table(rows)


def _category_table(ga: Dict[str, Any]) -> str:
    rows = []
    for c, v in list(ga["per_category"].items()) + [("overall", ga["overall"])]:
        rows.append({"category": c, "n": v["chart"]["n"], "acc (chart)": v["chart"]["acc"],
                     "hits@5 (chart)": v["chart"]["hits5"], "MRR (chart)": v["chart"]["mrr"],
                     "n reranked": v["rerank"]["n"], "acc (chart, reranked)": v["chart_reranked"]["acc"],
                     "acc (log rerank)": v["rerank"]["acc"],
                     "acc (ONLY-B)": v["only_b"]["acc"], "BCH gap median": v["bch_gap_median"]})
    return markdown_table(rows)


def _relation_table(rel: Dict[str, Any]) -> str:
    rows = []
    for k, v in rel.items():
        ok = v.get("status") == "ok"
        rows.append({"relation": k, "pairs": v["n_pairs"],
                     "consistency": v["consistency"] if ok else None,
                     "null mean": v["null_mean"] if ok else None, "null sd": v["null_std"] if ok else None,
                     "z": v["z"] if ok else None, "p": v["p"] if ok else None,
                     "mean ||log T||_F": v["log_norm_mean"] if ok else None})
    return markdown_table(rows)


def _ga_metrics(ga: Optional[Dict[str, Any]], *, keep_questions: bool) -> Optional[Dict[str, Any]]:
    if ga is None:
        return None
    out = {k: ga.get(k) for k in ("n_questions", "n_covered", "overall", "per_category", "bch_gap",
                                  "rerank_pool", "n_reranked")}
    if keep_questions:
        out["questions"] = ga["questions"]
    return out


def _relation_metrics(v: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: x for k, x in v.items() if not k.startswith(("phase_", "log_mod_"))}
    if v.get("status") == "ok":
        out.update(abs_phase_true=_summary(np.abs(v["phase_true"])),
                   abs_phase_random=_summary(np.abs(v["phase_random"])),
                   log_mod_true=_summary(v["log_mod_true"]), log_mod_random=_summary(v["log_mod_random"]))
    return out


def run_phi_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    r"""The ``phi`` section: gauge-fixed frame chart, invariants, commutator/BCH diagnostics, group
    analogies, relation transports and positional frames; figures under ``figures/phi``, tables under
    ``tables/phi``. Skipped (with a note) when the run's frames are inactive or the analysis vocabulary
    has fewer than two candidate tokens. Stores the :class:`CanonicalFrames` in
    ``ctx.cache["canonical_frames"]`` for the report's ``phi_log`` view. Budgets come from
    ``ctx.options`` (keys of :data:`PHI_DEFAULTS`).

    A sub-step that raises is recorded in ``metrics["failed_steps"]`` as ``{"step", "error"}`` (error
    ``"Type: message"``) and in the notes; the sub-steps that do not depend on it still run.
    """
    t_start = time.perf_counter()
    res = SectionResult(name="phi")
    if not run.frames_active:
        res.status = "skipped"
        res.notes.append("Gauge frames are inactive for this run (identity token frames, or the unlearned random "
                         "frames of gauge_transport='frozen'; the run notes give the reason), so there is no phi "
                         "geometry to analyze.")
        res.seconds = time.perf_counter() - t_start
        return res
    ids = np.asarray(ctx.candidate_ids, dtype=np.int64)
    if ids.size < 2:
        res.status = "skipped"
        res.notes.append(f"The analysis vocabulary has {ids.size} candidate token(s) (the whole-word mask keeps "
                         "almost nothing for this tokenizer and dataset), so there is no frame geometry to analyze.")
        res.seconds = time.perf_counter() - t_start
        return res
    apply_style()
    timings: Dict[str, float] = {}
    failed: List[Dict[str, str]] = []

    def step(name: str, fn: Callable[[], Any]) -> Any:
        r"""Run one sub-step; if it raises, record ``{"step", "error"}`` and a note and return None."""
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:                  # recorded; the independent sub-steps still run
            err = f"{type(exc).__name__}: {exc}"
            err = err if len(err) <= 300 else err[:297] + "..."
            failed.append({"step": name, "error": err})
            res.notes.append(f"Sub-step {name} failed ({err}); the sub-steps that do not depend on it still ran.")
            return None
        finally:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0

    def rng(step_name: str) -> np.random.Generator:
        return np.random.default_rng([int(ctx.seed), _PHI_STREAM, _PHI_STEP_STREAM[step_name]])

    n_ref = int(min(int(_opt(ctx, "frame_ref_count")), ids.size))
    ref_ids = ids[:n_ref]
    cfg = run.cfg
    d_status, d_reason = diagonal_redundancy_status(cfg)

    cf = step("canonical_frames", lambda: canonical_frames(run, ids, ref_ids, run.log_var(ref_ids)))
    if cf is not None:
        ctx.cache["canonical_frames"] = cf
        qual = _barycenter_qualifier(cf)
        res.notes.append(
            f"Per-token frame statistics use the gauge-fixed chart U_hat = D0 e^X m^-1 D0^-1 (m = bi-invariant "
            f"barycenter of the {n_ref} most frequent candidate frames, D0 from their mean log-variance), which is "
            f"invariant{qual} under the split redundancy e^X -> e^X k and the global positive-diagonal rescaling. "
            "Norms, commutators and BCH gaps of the raw coordinates describe the trained trivialization only.")
        if qual:
            res.notes.append(
                f"The frame barycenter did not reach its tolerance in every block even after the adaptive "
                f"extension; the best iterate was used, so every invariance statement about U_hat, its logs and the "
                f"phi_log view holds only up to O(r), r = {cf.diagnostics['barycenter_residual_max']:.1e} "
                "(see metrics chart.blocks[*].barycenter).")
        if cf.diagnostics["n_ok"] < cf.n:
            res.notes.append(f"{cf.n - cf.diagnostics['n_ok']} of {cf.n} gauge-fixed frames have no real principal "
                             "log (eigenvalue on the closed negative real axis) and are excluded from log-chart "
                             "statistics.")
    res.notes.append(_diagonal_note(d_status, d_reason))
    res.notes.append("flat frame transport: Omega_ij = U_i U_j^-1 telescopes, so its loop holonomy is identically "
                     "the identity; no frame holonomy is computed or plotted."
                     + ("" if getattr(cfg, "content_link_edges", "off") in (None, "off") else
                        f" The content links G_ij of content_link_edges={cfg.content_link_edges!r} are a separate "
                        "connection whose triangle holonomy is not the identity; this report does not measure it."))
    if getattr(run, "phi_frozen", False):
        res.notes.append("phi is not updated in context: each realized frame is the token's lookup frame "
                         "times the positional right factor e^{Y_p}.")

    scalars = step("invariant_scalars", lambda: invariant_scalars(cf)) if cf is not None else None
    log_count = ctx.log_count(cf.ids) if cf is not None else None
    spear = ({k: _spearman(scalars[k], log_count) for k, _ in _SCALAR_LABELS}
             if scalars is not None and log_count is not None else None)

    n_pairs = int(_opt(ctx, "frame_commutator_pairs"))
    comm = step("commutators_raw", lambda: commutator_diagnostics(run, ids, rng("commutators_raw"), n_pairs))

    def _canonical_commutators() -> Dict[str, Any]:
        ok_rows = np.flatnonzero(cf.ok)
        pa, pb = _random_pairs(ok_rows.size, n_pairs, rng("commutators_canonical"))
        ta, tb = torch.from_numpy(ok_rows[pa]), torch.from_numpy(ok_rows[pb])
        return _commutator_stats([Lb[ta] for Lb in cf.L], [Lb[tb] for Lb in cf.L])

    comm_canon = step("commutators_canonical", _canonical_commutators) if cf is not None else None

    rerank = int(_opt(ctx, "frame_rerank"))
    ga_bench = (step("group_analogies_benchmark",
                     lambda: group_analogies(cf, ctx.analogies, rerank=rerank,
                                             max_rerank_questions=_opt(ctx, "frame_rerank_questions"),
                                             rng=rng("group_analogies_benchmark")))
                if cf is not None and ctx.analogies else None)
    ga_cur = (step("group_analogies_curated", lambda: group_analogies(cf, ctx.curated_analogies, rerank=rerank))
              if cf is not None and ctx.curated_analogies else None)
    if ga_bench is not None and ga_bench["n_reranked"] < ga_bench["n_covered"]:
        res.notes.append(f"Log-residual rerank evaluated on a seeded sample of {ga_bench['n_reranked']} of "
                         f"{ga_bench['n_covered']} covered benchmark questions; the chart accuracy is reported both "
                         "on all covered questions and on exactly that sample, and only the latter pairs with the "
                         "rerank accuracy.")

    rel = ((step("relation_transports", lambda: relation_transports(cf, ctx.relations, rng("relation_transports"),
                                                                    int(_opt(ctx, "frame_null"))))
            if cf is not None and ctx.relations else None) or {})

    n_perm = int(_opt(ctx, "frame_positional_null"))
    pos = step("positional_frames",
               lambda: positional_frames(run, _opt(ctx, "frame_max_positions"), n_perm=n_perm,
                                         rng=rng("positional")))
    if pos is None and not any(f["step"] == "positional_frames" for f in failed):
        res.notes.append("No positional frames (pos_phi='none'): the realized frame is the token frame.")
    elif pos is not None:
        res.notes.append(
            f"Relative-position R2 by offset m - n and by absolute position m are each compared with a "
            f"position-permutation null ({n_perm} relabelings of the positions; the interior variant permutes only "
            "the non-outlier positions). The i.i.d. value (G-1)/(N-1) is not a null for the grouping by m, whose "
            "pairs share the frame V_m; it is kept in metrics (r2_chance) only as the analytic reference of the "
            "offset grouping.")
        if pos["outlier_positions"].size:
            res.notes.append(f"Positions {pos['outlier_positions'].tolist()} have outlying ||Y_p||_F (> "
                             f"{pos['outlier_z']:g} robust sd above the median); the relative-position R2 is "
                             "reported for all pairs and for the pairs that avoid these positions.")

    def _write(name: str, text: str) -> None:
        write_table(ctx, "phi", name, text)
        res.tables[name] = text

    if cf is not None and comm is not None and comm_canon is not None:
        res.figures += step("figure_overview", lambda: _plot_overview(ctx, cf, comm, comm_canon)) or []
    if spear is not None:
        res.figures += step("figure_invariants_vs_frequency",
                            lambda: _plot_invariants_vs_frequency(ctx, scalars, log_count, spear)) or []
    elif scalars is not None:
        res.notes.append("Training counts unavailable: invariants-vs-frequency figure and correlations skipped.")
    if ga_bench is not None and ga_bench["n_covered"] > 0:
        res.figures += step("figure_group_analogies", lambda: _plot_group_analogies(ctx, ga_bench)) or []
        step("table_group_analogies_benchmark", lambda: _write("group_analogies_benchmark", _category_table(ga_bench)))
    elif cf is not None and (ga_bench is not None or not ctx.analogies):
        res.notes.append("No benchmark analogy question is covered by the frame chart; group-analogy figure "
                         "skipped.")
    if ga_cur is not None and ga_cur["n_covered"] > 0:
        step("table_group_analogies_curated", lambda: _write("group_analogies_curated", _curated_table(ctx, ga_cur)))
    if any(v.get("status") == "ok" for v in rel.values()):
        res.figures += step("figure_relation_transports", lambda: _plot_relation_transports(ctx, rel)) or []
        step("table_relation_transports", lambda: _write("relation_transports", _relation_table(rel)))
    elif cf is not None and (rel or not ctx.relations):
        res.notes.append("No relation has three resolved pairs in the frame chart; relation-transport figure "
                         "skipped.")
    if pos is not None:
        res.figures += step("figure_positional", lambda: _plot_positional(ctx, pos)) or []

    ratio_block_median = float(np.nanmedian(comm["ratio_block"])) if comm is not None else None
    positional = None
    if pos is not None:
        positional = {k: pos[k] for k in ("positions", "n_pairs", "r2_offset", "r2_absolute", "r2_chance",
                                          "null", "per_block", "outlier_positions", "outlier_z", "n_pairs_interior",
                                          "r2_offset_interior", "r2_absolute_interior", "r2_chance_interior",
                                          "null_interior", "s_mean_by_offset", "s_mean_by_offset_interior")}
        positional["y_norm"] = _summary(pos["y_norm"])
    res.metrics = {
        "chart": cf.diagnostics if cf is not None else None,
        "n_tokens": int(ids.size), "n_ref": n_ref,
        "diagonal_redundancy": {"status": d_status, "reason": d_reason},
        "invariants": ({k: {"summary": _summary(scalars[k]),
                            "spearman_log_count": None if spear is None else spear[k]}
                        for k, _ in _SCALAR_LABELS} if scalars is not None else None),
        "trace_block": ([_summary(scalars["trace_block"][:, b]) for b in range(scalars["trace_block"].shape[1])]
                        if scalars is not None else None),
        "commutator_raw": ({**comm["summary"], "ratio_block_median": ratio_block_median,
                            "iid_gaussian_ratio": [float(np.sqrt(2.0 / (sl.stop - sl.start)))
                                                   for sl in run.block_slices]}
                           if comm is not None else None),
        "commutator_canonical": ({"ratio": _summary(comm_canon["ratio"]),
                                  "bch_gap": _summary(comm_canon["bch_gap"]),
                                  "ratio_block_median": float(np.nanmedian(comm_canon["ratio_block"]))}
                                 if comm_canon is not None else None),
        "group_analogies": {"benchmark": _ga_metrics(ga_bench, keep_questions=False),
                            "curated": _ga_metrics(ga_cur, keep_questions=True)},
        "relation_transports": {k: _relation_metrics(v) for k, v in rel.items()},
        "positional": positional,
        "options": {k: _opt(ctx, k) for k in PHI_DEFAULTS},
        "timings": timings,
        "failed_steps": failed,
    }

    head: Dict[str, Any] = {}
    if comm is not None:
        head["phi: median ||X_v^b||_F (raw chart)"] = float(np.median(comm["norm_block"]))
    if cf is not None:
        head["phi: canonical log-chart ok fraction"] = cf.diagnostics["ok_fraction"]
    if comm is not None:
        head["phi: median per-block commutator ratio (raw; iid-Gaussian ref sqrt(2/d))"] = ratio_block_median
        head["phi: median BCH gap (raw chart)"] = comm["summary"]["bch_gap"].get("median")
    if spear is not None:
        head["phi: rho_s(log det U_hat, log count)"] = spear["trace"]["rho"]
        head["phi: rho_s(rotation, log count)"] = spear["rotation"]["rho"]
    if ga_bench is not None and ga_bench["n_covered"] > 0:
        ov = ga_bench["overall"]
        head["phi: group analogy acc, all covered questions: chart / ONLY-B"] = (
            f"{_fmt3(ov['chart']['acc'])} / {_fmt3(ov['only_b']['acc'])} (n = {ov['chart']['n']})")
        if ov["rerank"]["n"]:
            head["phi: group analogy acc, reranked subset: chart / log rerank"] = (
                f"{_fmt3(ov['chart_reranked']['acc'])} / {_fmt3(ov['rerank']['acc'])} "
                f"(n = {ov['rerank']['n']}, same questions)")
    q = _find_curated(ga_cur, ctx, ("man", "woman", "king"))
    if q is not None:
        head["phi: man:woman::king:? answer rank (chart / log rerank)"] = (
            f"{q['rank_chart']} / {_rerank_rank_text(q, ga_cur['rerank_pool'])}")
    ok_rel = [v for v in rel.values() if v.get("status") == "ok"]
    if ok_rel:
        head["phi: relations with consistent transports (p < 0.01)"] = (
            f"{sum(1 for v in ok_rel if v['p'] is not None and v['p'] < 0.01)} / {len(ok_rel)}")
    if pos is not None:
        key = "phi: positional R2 by offset / by absolute m (position-permutation null mean, q95, p)"
        null = pos["null"] or {}
        head[key] = (f"{_r2_vs_null(pos['r2_offset'], null.get('offset'))} / "
                     f"{_r2_vs_null(pos['r2_absolute'], null.get('absolute'))}")
        if pos["outlier_positions"].size:
            null = pos["null_interior"] or {}
            head[f"{key}, excluding {pos['outlier_positions'].size} outlier positions"] = (
                f"{_r2_vs_null(pos['r2_offset_interior'], null.get('offset'))} / "
                f"{_r2_vs_null(pos['r2_absolute_interior'], null.get('absolute'))}")
    res.headline = head
    res.seconds = time.perf_counter() - t_start
    return res
