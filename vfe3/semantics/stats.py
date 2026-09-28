r"""Generic, array-level statistics for the semantic battery.

Every function takes plain arrays (rows = tokens), computes in float64, and returns a float or a
JSON-able dict. Nothing here knows about vocabularies, views or checkpoints; :mod:`vectors` and the
other sections compose these on their own rows.

Conventions. ``X`` is ``(n, D)``. "Centered" means the column mean over the ``n`` rows is removed.
The covariance spectrum ``lambda`` is the list of eigenvalues of the centered scatter matrix
``Xc^T Xc`` (the common ``1/(n-1)`` factor cancels in every ratio below). Random quantities take an
explicit ``np.random.Generator``; no function seeds itself.

References: Roy & Vetterli (EUSIPCO 2007, effective rank); Rudman et al. (Findings ACL 2022,
IsoScore); Ethayarajh (EMNLP 2019, anisotropy); Facco et al. (Sci. Rep. 7:12140, 2017, TwoNN);
Kornblith et al. (ICML 2019, CKA) with the unbiased HSIC of Song et al. (JMLR 2012); Radovanovic
et al. (JMLR 2010, hubness); Timkey & van Schijndel (EMNLP 2021, rogue dimensions).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------------------------
# Array helpers
# ----------------------------------------------------------------------------------------------

def _f64(X: Any) -> np.ndarray:
    r"""``X`` as a finite float64 ``(n, D)`` array (raises otherwise)."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D (n, D) array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("array contains non-finite values")
    return array


def _centered(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def unit_rows(X: np.ndarray) -> np.ndarray:
    r"""Rows divided by their L2 norm; a zero row has no direction and raises."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if (norms <= 0.0).any():
        raise ValueError(f"{int((norms <= 0.0).sum())} zero-norm row(s): cosine is undefined")
    return X / norms


def centered_eigenvalues(X: Any) -> np.ndarray:
    r"""Eigenvalues ``lambda_1 >= ... >= lambda_D >= 0`` of the centered scatter ``Xc^T Xc``.

    ``lambda_i = s_i^2`` with ``s_i`` the singular values of ``Xc``. Computed from whichever Gram
    matrix is smaller (``D x D`` or ``n x n``) and zero-padded to length ``D``.
    """
    Xc = _centered(_f64(X))                                            # (n, D)
    n, D = Xc.shape
    gram = Xc.T @ Xc if D <= n else Xc @ Xc.T                          # (min(n,D), min(n,D))
    lam = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)
    out = np.zeros(D, dtype=np.float64)
    out[: min(D, lam.size)] = lam[:D]
    return out


def _require_spectrum(lam: np.ndarray) -> np.ndarray:
    if lam.sum() <= 0.0:
        raise ValueError("rows have zero variance: the spectrum is empty")
    return lam


def effective_rank_from_eigenvalues(lam: np.ndarray) -> float:
    r"""``exp(H(p))`` with ``p_i = s_i / sum_j s_j`` and ``s_i = sqrt(lambda_i)`` (Roy & Vetterli 2007)."""
    s = np.sqrt(_require_spectrum(np.asarray(lam, dtype=np.float64)))
    p = s / s.sum()
    p = p[p > 0.0]
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio_from_eigenvalues(lam: np.ndarray) -> float:
    r"""``(sum_i lambda_i)^2 / sum_i lambda_i^2``."""
    lam = _require_spectrum(np.asarray(lam, dtype=np.float64))
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def isoscore_from_eigenvalues(lam: np.ndarray) -> float:
    r"""IsoScore of a variance vector along the principal axes (Rudman et al. 2022, Alg. 1).

    With ``d`` the ambient dimension and ``Sigma_D`` the PCA variance vector,

    .. math::
        \hat\Sigma_D = \sqrt d\,\Sigma_D/\lVert\Sigma_D\rVert,\quad
        \delta = \frac{\lVert\hat\Sigma_D - \mathbf 1\rVert}{\sqrt{2(d-\sqrt d)}},\quad
        k = \frac{(d - \delta^2(d-\sqrt d))^2}{d},\quad
        \mathrm{IsoScore} = \frac{k-1}{d-1},

    which is 1 for an isotropic cloud and 0 when all variance lies on one axis.
    """
    lam = _require_spectrum(np.asarray(lam, dtype=np.float64))
    d = lam.size
    if d < 2:
        raise ValueError("IsoScore needs at least two dimensions")
    root = math.sqrt(d)
    v_hat = root * lam / np.linalg.norm(lam)
    delta = np.linalg.norm(v_hat - 1.0) / math.sqrt(2.0 * (d - root))
    k = (d - delta ** 2 * (d - root)) ** 2 / d
    return float((k - 1.0) / (d - 1.0))


def explained_variance_from_eigenvalues(lam: np.ndarray, k: int = 50) -> np.ndarray:
    r"""The first ``k`` explained-variance ratios ``lambda_i / sum_j lambda_j``."""
    lam = _require_spectrum(np.asarray(lam, dtype=np.float64))
    return (lam / lam.sum())[:k]


def knn(X: Any, k: int, metric: str = "cosine", *, chunk: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    r"""Exact ``k`` nearest neighbors of every row among the other rows (self excluded).

    ``metric="cosine"`` ranks by cosine similarity of the rows as given (they are L2-normalized
    here, not centered); ``"euclidean"`` ranks by distance, and the returned distances of the
    selected neighbors are recomputed from explicit differences (no Gram cancellation).
    Returns ``(idx, score)``, both ``(n, k)``, nearest first; ``score`` is the cosine or the distance.
    Work is chunked over query rows, so memory is ``O(chunk * n)``.
    """
    X = _f64(X)
    n = X.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k={k} must lie in [1, n-1] for n={n}")
    if metric == "cosine":
        Z = unit_rows(X)
    elif metric == "euclidean":
        Z = X
        sq = (X * X).sum(axis=1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    idx = np.empty((n, k), dtype=np.int64)
    score = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = np.arange(start, stop)
        G = Z[start:stop] @ Z.T                                         # (c, n)
        if metric == "cosine":
            key = -G                                                    # smaller key = nearer
        else:
            key = sq[start:stop, None] - 2.0 * G + sq[None, :]
        key[rows - start, rows] = np.inf
        part = np.argpartition(key, k - 1, axis=1)[:, :k]               # (c, k) unordered
        if metric == "cosine":
            vals = -np.take_along_axis(key, part, axis=1)
        else:
            vals = np.empty(part.shape, dtype=np.float64)
            for j in range(k):                                          # (c, D) per column
                vals[:, j] = np.linalg.norm(X[part[:, j]] - X[start:stop], axis=1)
        order = np.argsort(-vals if metric == "cosine" else vals, axis=1, kind="stable")
        idx[start:stop] = np.take_along_axis(part, order, axis=1)
        score[start:stop] = np.take_along_axis(vals, order, axis=1)
    return idx, score


def _label_codes(labels: Sequence[Hashable]) -> Tuple[np.ndarray, list]:
    names, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes.reshape(-1).astype(np.int64), [n.item() if hasattr(n, "item") else n for n in names]


# ----------------------------------------------------------------------------------------------
# Spectrum-based summaries
# ----------------------------------------------------------------------------------------------

def effective_rank(X: Any) -> float:
    r"""Effective rank of the centered rows: ``exp(-sum_i p_i log p_i)``, ``p_i = s_i/sum_j s_j``."""
    return effective_rank_from_eigenvalues(centered_eigenvalues(X))


def participation_ratio(X: Any) -> float:
    r"""``(sum lambda)^2 / sum lambda^2`` of the covariance eigenvalues."""
    return participation_ratio_from_eigenvalues(centered_eigenvalues(X))


def isoscore(X: Any) -> float:
    r"""IsoScore (Rudman et al. 2022) of the rows; see :func:`isoscore_from_eigenvalues`."""
    return isoscore_from_eigenvalues(centered_eigenvalues(X))


def pca_spectrum(X: Any, k: int = 50) -> np.ndarray:
    r"""Explained-variance ratios of the first ``k`` principal components of the centered rows."""
    return explained_variance_from_eigenvalues(centered_eigenvalues(X), k)


# ----------------------------------------------------------------------------------------------
# Cosine-geometry summaries
# ----------------------------------------------------------------------------------------------

def anisotropy(X: Any, rng: Optional[np.random.Generator] = None, n_pairs: int = 20000,
               center: bool = False) -> float:
    r"""Mean cosine similarity over distinct pairs of rows (Ethayarajh 2019).

    Computed EXACTLY over all ordered pairs ``i != j`` from unit rows ``z_i``:

    .. math:: \overline{\cos} = \frac{\lVert\sum_i z_i\rVert^2 - n}{n(n-1)},

    which is the expectation that a random-pair estimate samples, so ``rng``/``n_pairs`` are not
    used (kept for interface stability). ``center=True`` removes the row mean first.
    """
    X = _f64(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError("need at least two rows")
    Z = unit_rows(_centered(X) if center else X)
    total = Z.sum(axis=0)
    return float((total @ total - n) / (n * (n - 1)))


def mean_vector_ratio(X: Any) -> float:
    r"""``||mean_i x_i|| / mean_i ||x_i||`` (1 for identical rows, 0 for a symmetric cloud)."""
    X = _f64(X)
    mean_norm = float(np.linalg.norm(X, axis=1).mean())
    if mean_norm <= 0.0:
        raise ValueError("all rows are zero")
    return float(np.linalg.norm(X.mean(axis=0)) / mean_norm)


def rogue_dimensions(X: Any, rng: Optional[np.random.Generator] = None, n_pairs: int = 20000,
                     top: int = 5) -> Dict[str, Any]:
    r"""Per-dimension contribution to the mean pairwise cosine (Timkey & van Schijndel 2021).

    With unit rows ``z_i``, ``CC_k = E_{i != j}[z_{ik} z_{jk}]``, computed exactly as
    ``((sum_i z_ik)^2 - sum_i z_ik^2) / (n(n-1))``; ``sum_k CC_k`` is the mean cosine. The
    ``rng``/``n_pairs`` arguments are unused (exact expectation). Also reports the mean cosine
    after z-scoring every dimension, the post-processing Timkey & van Schijndel recommend.
    """
    X = _f64(X)
    n, D = X.shape
    if n < 2:
        raise ValueError("need at least two rows")
    Z = unit_rows(X)
    CC = (Z.sum(axis=0) ** 2 - (Z * Z).sum(axis=0)) / (n * (n - 1))   # (D,)
    mean_cos = float(CC.sum())
    order = np.argsort(-CC, kind="stable")[: min(top, D)]
    std = X.std(axis=0)
    usable = std > 0.0
    zscored = (X[:, usable] - X[:, usable].mean(axis=0)) / std[usable]
    return {
        "mean_cosine":          mean_cos,
        "mean_cosine_zscored":  anisotropy(zscored) if usable.any() else float("nan"),
        "top_dims":             order,
        "top_contribution":     CC[order],
        "top_share":            CC[order] / mean_cos if mean_cos != 0.0 else np.full(order.size, np.nan),
        "top_share_total":      float(CC[order].sum() / mean_cos) if mean_cos != 0.0 else float("nan"),
        "n":                    n,
    }


# ----------------------------------------------------------------------------------------------
# Intrinsic dimension
# ----------------------------------------------------------------------------------------------

def twonn_dimension(X: Any, rng: np.random.Generator, max_points: int = 5000) -> Dict[str, Any]:
    r"""TwoNN intrinsic dimension (Facco et al. 2017).

    For each point the ratio ``mu_i = r_2/r_1`` of its second to first nearest-neighbor Euclidean
    distance is Pareto(``d``) distributed, ``F(mu) = 1 - mu^{-d}``. Reported:

    * ``mle`` -- ``d = N / sum_i log mu_i``;
    * ``fit`` -- the Facco estimator: least-squares slope through the origin of
      ``-log(1 - F_emp(mu))`` against ``log mu`` over the smallest 90% of ``mu``.

    Exact duplicate rows are dropped first (they give ``r_1 = 0``); at most ``max_points`` distinct
    rows are used (uniform subsample from ``rng``).
    """
    X = _f64(X)
    distinct = np.unique(X, axis=0)
    n_duplicates = X.shape[0] - distinct.shape[0]
    if distinct.shape[0] > max_points:
        distinct = distinct[rng.choice(distinct.shape[0], size=max_points, replace=False)]
    N = distinct.shape[0]
    if N < 10:
        raise ValueError(f"TwoNN needs at least 10 distinct points, got {N}")
    _, dist = knn(distinct, 2, "euclidean")
    mu = dist[:, 1] / dist[:, 0]                                        # (N,) >= 1
    log_mu = np.log(mu)
    mle = float(N / log_mu.sum()) if log_mu.sum() > 0.0 else float("inf")
    sorted_log_mu = np.sort(log_mu)
    F = np.arange(1, N + 1, dtype=np.float64) / N
    keep = int(math.floor(0.9 * N))
    x = sorted_log_mu[:keep]
    y = -np.log1p(-F[:keep])
    fit = float((x @ y) / (x @ x)) if (x @ x) > 0.0 else float("inf")
    return {"mle": mle, "fit": fit, "n": N, "n_duplicates_dropped": int(n_duplicates)}


# ----------------------------------------------------------------------------------------------
# Representation similarity
# ----------------------------------------------------------------------------------------------

def _debiased_dot(xty: float, ssr_x: np.ndarray, ssr_y: np.ndarray, sq_x: float, sq_y: float,
                  n: int) -> float:
    r"""Unbiased HSIC numerator (Song et al. 2012) in feature space, as in Kornblith's reference code.

    For column-centered ``X, Y`` with ``K = X X^T``, ``L = Y Y^T`` and zeroed diagonals,
    ``n(n-3) HSIC_u = ||X^T Y||_F^2 - n/(n-2) <diag K, diag L> + ||X||_F^2 ||Y||_F^2/((n-1)(n-2))``.
    """
    return float(xty - n / (n - 2.0) * float(ssr_x @ ssr_y) + sq_x * sq_y / ((n - 1.0) * (n - 2.0)))


def linear_cka(X: Any, Y: Any, debiased: bool = False) -> float:
    r"""Linear CKA between two row-matched representations (Kornblith et al. 2019).

    .. math:: \mathrm{CKA}(X,Y) = \frac{\lVert Y_c^\top X_c\rVert_F^2}
              {\lVert X_c^\top X_c\rVert_F\,\lVert Y_c^\top Y_c\rVert_F},

    with column-centered ``X_c, Y_c``. Invariant to orthogonal transforms and isotropic scaling of
    either side. ``debiased=True`` replaces each HSIC by the unbiased estimator of Song et al.
    (2012), which removes the upward bias of CKA when ``n`` is not much larger than the widths.
    """
    X = _centered(_f64(X))
    Y = _centered(_f64(Y))
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"row counts differ: {n} vs {Y.shape[0]}")
    xty = float(np.linalg.norm(X.T @ Y) ** 2)
    xtx = float(np.linalg.norm(X.T @ X) ** 2)
    yty = float(np.linalg.norm(Y.T @ Y) ** 2)
    if not debiased:
        denom = math.sqrt(xtx * yty)
        if denom <= 0.0:
            raise ValueError("a representation has zero variance")
        return xty / denom
    if n < 4:
        raise ValueError("debiased CKA needs at least 4 rows")
    ssr_x, ssr_y = (X * X).sum(axis=1), (Y * Y).sum(axis=1)
    sq_x, sq_y = float(ssr_x.sum()), float(ssr_y.sum())
    num = _debiased_dot(xty, ssr_x, ssr_y, sq_x, sq_y, n)
    den_x = _debiased_dot(xtx, ssr_x, ssr_x, sq_x, sq_x, n)
    den_y = _debiased_dot(yty, ssr_y, ssr_y, sq_y, sq_y, n)
    if den_x <= 0.0 or den_y <= 0.0:
        raise ValueError("debiased self-HSIC is not positive (too few rows for this width)")
    return num / math.sqrt(den_x * den_y)


# ----------------------------------------------------------------------------------------------
# Resampling statistics
# ----------------------------------------------------------------------------------------------

def spearman_ci(x: Any, y: Any, rng: np.random.Generator, n_boot: int = 1000) -> Dict[str, Any]:
    r"""Spearman ``rho`` with its two-sided p-value and a percentile bootstrap 95% CI over pairs."""
    from scipy import stats as sps

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite values in Spearman inputs")
    n = x.size
    if n < 3:
        raise ValueError(f"Spearman needs at least 3 pairs, got {n}")
    result = sps.spearmanr(x, y)
    rho, p = float(result.statistic), float(result.pvalue)
    boots = []
    for start in range(0, n_boot, 200):
        b = min(200, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        rx = sps.rankdata(x[idx], axis=1)
        ry = sps.rankdata(y[idx], axis=1)
        rx -= rx.mean(axis=1, keepdims=True)
        ry -= ry.mean(axis=1, keepdims=True)
        den = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            boots.append(np.where(den > 0.0, (rx * ry).sum(axis=1) / den, np.nan))
    boot = np.concatenate(boots)
    valid = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(valid, [2.5, 97.5]) if valid.size else (np.nan, np.nan))
    return {"rho": rho, "p": p, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n),
            "n_boot_valid": int(valid.size)}


def bootstrap_mean_ci(values: Any, rng: np.random.Generator, n_boot: int = 1000) -> Dict[str, Any]:
    r"""Mean of ``values`` with a percentile bootstrap 95% CI."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("no values")
    if not np.isfinite(v).all():
        raise ValueError("non-finite values")
    means = np.concatenate([
        v[rng.integers(0, v.size, size=(min(500, n_boot - s), v.size))].mean(axis=1)
        for s in range(0, n_boot, 500)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(v.size)}


def wilson_interval(k: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    r"""Wilson score interval (95% by default) for a binomial proportion ``k / n`` (Wilson 1927).

    .. math::
        \frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n},
        \qquad \hat p = k/n.

    Unlike a percentile bootstrap it does not collapse to a point at ``k = 0`` (upper bound
    ``z^2/(n + z^2)``) or ``k = n``.
    """
    if n <= 0:
        raise ValueError(f"Wilson interval needs n >= 1, got {n}")
    p = float(k) / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    lo = 0.0 if k <= 0 else max(0.0, center - half)                    # exact endpoints, no round-off
    hi = 1.0 if k >= n else min(1.0, center + half)
    return lo, hi


def unit_bootstrap_weights(unit_a: Any, unit_b: Any, n_units: int, rng: np.random.Generator,
                           n_draws: int) -> np.ndarray:
    r"""Bootstrap weights of items built from one or two resampling units (crossed-unit bootstrap).

    Each draw resamples the ``n_units`` units with replacement, giving multiplicities ``c_u`` with
    ``sum_u c_u = n_units``. Item ``q`` made of the two distinct units ``(u_q, v_q)`` gets

    .. math:: w_q = c_{u_q}\, c_{v_q},

    the number of ordered pairs of distinct drawn copies that re-form it; an item made of one unit
    (``v_q < 0``) gets ``w_q = c_{u_q}``. The weighted mean ``sum_q w_q h_q / sum_q w_q`` is the
    statistic recomputed on the items formed by the resampled units (for word analogies: the
    questions formed by the resampled word pairs). Returns ``(n_draws, Q)`` float64.
    """
    unit_a = np.asarray(unit_a, dtype=np.int64).reshape(-1)
    unit_b = np.asarray(unit_b, dtype=np.int64).reshape(-1)
    if unit_a.shape != unit_b.shape:
        raise ValueError(f"unit arrays differ in shape: {unit_a.shape} vs {unit_b.shape}")
    if n_units < 1 or (unit_a.size and (unit_a.min() < 0 or max(unit_a.max(), unit_b.max()) >= n_units)):
        raise ValueError(f"unit ids must lie in [0, {n_units}) (unit_b may be -1)")
    if (unit_a == unit_b).any():
        raise ValueError("an item's two units must be distinct")
    draws = rng.integers(0, n_units, size=(n_draws, n_units))
    offset = (np.arange(n_draws, dtype=np.int64) * n_units)[:, None]
    counts = np.bincount((draws + offset).ravel(), minlength=n_draws * n_units)
    counts = counts.reshape(n_draws, n_units).astype(np.float64)                 # (n_draws, n_units)
    second = np.where(unit_b >= 0, counts[:, np.maximum(unit_b, 0)], 1.0)       # (n_draws, Q)
    return counts[:, unit_a] * second


# ----------------------------------------------------------------------------------------------
# Label-based cluster quality
# ----------------------------------------------------------------------------------------------

def knn_label_purity(X: Any, labels: Sequence[Hashable], k: int = 10, metric: str = "cosine",
                     rng: Optional[np.random.Generator] = None, n_perm: int = 200) -> Dict[str, Any]:
    r"""kNN label purity ``P@k`` with a label-permutation null.

    ``purity_i`` is the fraction of the ``k`` nearest neighbors of row ``i`` (self excluded) that
    share its label; the statistic is the mean over rows. The null keeps the neighbor graph and
    shuffles labels ``n_perm`` times; ``z = (obs - mean_null)/std_null`` and the one-sided
    ``p = (1 + #\{null >= obs\})/(1 + n_perm)``. ``chance`` is the exact expected purity under random
    labeling, ``sum_c n_c(n_c - 1) / (n(n-1))`` (approximately ``sum_c p_c^2``).
    """
    X = _f64(X)
    codes, names = _label_codes(labels)
    n = X.shape[0]
    if codes.size != n:
        raise ValueError(f"{codes.size} labels for {n} rows")
    if len(names) < 2:
        raise ValueError("kNN purity needs at least two labels")
    if n_perm > 0 and rng is None:
        raise ValueError("an rng is required for the permutation null")
    k_eff = min(k, n - 1)
    idx, _ = knn(X, k_eff, metric)
    per_row = (codes[idx] == codes[:, None]).mean(axis=1)               # (n,)
    obs = float(per_row.mean())
    counts = np.bincount(codes, minlength=len(names)).astype(np.float64)
    chance = float((counts * (counts - 1.0)).sum() / (n * (n - 1.0)))
    null = np.empty(n_perm, dtype=np.float64)
    for t in range(n_perm):
        perm = rng.permutation(codes)
        null[t] = (perm[idx] == perm[:, None]).mean()
    null_mean = float(null.mean()) if n_perm else float("nan")
    null_std = float(null.std(ddof=1)) if n_perm > 1 else float("nan")
    z = (obs - null_mean) / null_std if n_perm > 1 and null_std > 0.0 else float("nan")
    p = float((1 + (null >= obs).sum()) / (1 + n_perm)) if n_perm else float("nan")
    return {
        "purity":       obs,
        "k":            int(k_eff),
        "n":            int(n),
        "n_labels":     len(names),
        "per_label":    {str(name): float(per_row[codes == c].mean()) for c, name in enumerate(names)},
        "per_label_n":  {str(name): int(counts[c]) for c, name in enumerate(names)},
        "chance":       chance,
        "null_mean":    null_mean,
        "null_std":     null_std,
        "z":            float(z),
        "p":            p,
        "n_perm":       int(n_perm),
        "metric":       metric,
    }


def silhouette(X: Any, labels: Sequence[Hashable], metric: str = "cosine") -> Dict[str, Any]:
    r"""Silhouette ``s(i) = (b_i - a_i)/max(a_i, b_i)`` in the ORIGINAL space (Rousseeuw 1987).

    Uses :func:`sklearn.metrics.silhouette_samples` with ``metric``; returns the mean and the
    per-label means. Never apply this to 2-D display coordinates.
    """
    from sklearn.metrics import silhouette_samples

    X = _f64(X)
    codes, names = _label_codes(labels)
    if not 2 <= len(names) <= X.shape[0] - 1:
        raise ValueError(f"silhouette needs 2 <= n_labels <= n-1, got {len(names)} labels, n={X.shape[0]}")
    s = silhouette_samples(X, codes, metric=metric)
    return {
        "silhouette": float(s.mean()),
        "per_label":  {str(name): float(s[codes == c].mean()) for c, name in enumerate(names)},
        "n":          int(X.shape[0]),
        "metric":     metric,
    }


def hubness(X: Any, k: int = 10, metric: str = "cosine") -> Dict[str, Any]:
    r"""Hubness of the ``k``-NN graph (Radovanovic et al. 2010).

    ``N_k(x)`` counts how often row ``x`` appears among the ``k`` nearest neighbors of the other
    rows (``sum_x N_k = n k``). Reports the skewness of ``N_k`` (population moments), its maximum,
    the anti-hub fraction ``P(N_k = 0)``, and the top-10 hub rows.
    """
    from scipy import stats as sps

    X = _f64(X)
    n = X.shape[0]
    k_eff = min(k, n - 1)
    idx, _ = knn(X, k_eff, metric)
    Nk = np.bincount(idx.reshape(-1), minlength=n)
    top = np.argsort(-Nk, kind="stable")[: min(10, n)]
    return {
        "k":                 int(k_eff),
        "skewness":          float(sps.skew(Nk)) if Nk.std() > 0 else 0.0,
        "max_Nk":            int(Nk.max()),
        "antihub_fraction":  float((Nk == 0).mean()),
        "top_rows":          top,
        "top_Nk":            Nk[top],
        "n":                 int(n),
        "metric":            metric,
    }


# ----------------------------------------------------------------------------------------------
# Frequency probe
# ----------------------------------------------------------------------------------------------

def frequency_probe(X: Any, log_count: Any, rng: np.random.Generator, folds: int = 5,
                    max_features: int = 512) -> Dict[str, Any]:
    r"""Cross-validated ridge ``R^2`` of ``log count`` from the rows, with a shuffled-target control.

    The rows are centered and divided by ONE global scale (their RMS entry), so the ridge penalty
    ``alpha ||w||^2`` is isotropic in the view's own inner product (no per-feature standardization,
    which would whiten the rows and put every noise direction on an equal footing with the signal);
    :class:`sklearn.linear_model.RidgeCV` picks ``alpha in 10^{[-3, 5]}`` by efficient LOO inside
    each of ``folds`` seeded folds. The control refits on a permutation of the target (same folds),
    so ``r2_mean - control_r2_mean`` is the frequency information above chance (the selectivity
    logic of Hewitt & Liang 2019). When ``D > max_features`` the rows are first projected onto
    their top principal components (target-free; ridge is rotation invariant, so this is a pure
    subspace truncation), reported as ``pca_components``.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_score

    X = _centered(_f64(X))
    y = np.asarray(log_count, dtype=np.float64).reshape(-1)
    if y.size != X.shape[0]:
        raise ValueError(f"{y.size} targets for {X.shape[0]} rows")
    if X.shape[0] < 2 * folds:
        raise ValueError(f"need at least {2 * folds} rows for {folds}-fold CV")
    pca_components = None
    n, D = X.shape
    if D > max_features:
        # top principal-component scores from the smaller Gram, target-free
        if D <= n:
            _, V = np.linalg.eigh(X.T @ X)
            X = X @ V[:, ::-1][:, :max_features]
        else:
            w, U = np.linalg.eigh(X @ X.T)
            keep = min(max_features, n - 1)
            top = np.argsort(-w)[:keep]
            X = U[:, top] * np.sqrt(np.clip(w[top], 0.0, None))
        pca_components = int(X.shape[1])
    rms = float(np.sqrt((X * X).mean()))
    if rms <= 0.0:
        raise ValueError("rows have zero variance")
    X = X / rms
    seed = int(rng.integers(0, 2 ** 31 - 1))
    split = KFold(n_splits=folds, shuffle=True, random_state=seed)
    model = RidgeCV(alphas=np.logspace(-3, 5, 17))
    r2 = cross_val_score(model, X, y, cv=split, scoring="r2")
    control = cross_val_score(model, X, rng.permutation(y), cv=split, scoring="r2")
    return {
        "r2_mean":          float(r2.mean()),
        "r2_std":           float(r2.std(ddof=1)),
        "r2_folds":         r2,
        "control_r2_mean":  float(control.mean()),
        "control_r2_folds": control,
        "folds":            int(folds),
        "n":                int(n),
        "pca_components":   pca_components,
    }
