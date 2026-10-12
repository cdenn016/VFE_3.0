r"""Diagonal-Gaussian geometry of the token prior bank (and of a Gaussian decode bank).

Array level (float64, NumPy broadcasting over leading axes, the last axis is the coordinate
``k = 1..K``): every function takes means ``mu`` and log-variances ``lv = log sigma^2`` of two
diagonal Gaussians ``N_a = N(mu_a, diag e^{lv_a})`` and ``N_b`` and returns one value per
broadcast pair. Pairwise matrices come from ``f(mu[:, None], lv[:, None], mu[None], lv[None])``.

* :func:`kl_diag`, :func:`kl_one_vs_all` -- Kullback-Leibler divergence,
* :func:`jeffreys_diag` -- symmetrized KL,
* :func:`w2_sq_diag` -- squared 2-Wasserstein distance (exact for commuting covariances),
* :func:`bhattacharyya_diag`, :func:`hellinger_sq_diag`,
* :func:`fisher_rao_diag` -- Fisher-Rao geodesic distance inside the diagonal family,
* :func:`log_expected_likelihood`, :func:`distribution_cosine` -- the probability-product kernel
  of Vilnis & McCallum (2015) and its normalized form,
* :func:`w2_chart`, :func:`sigma_units` -- the Euclidean W2 chart ``(mu, sqrt v)`` and the D0
  unit normalization ``log d0_k = -1/2 mean_ref lv_k``.

:func:`run_sigma_section` measures what the covariance table carries (dispersion, frequency and
token-class dependence, variance-sorted neighbors, KL-asymmetry entailment, Gaussian similarity
scorers on SimLex/WordSim, table geometry, and the frame-induced routing covariance when frames
are active) and writes the ``sigma/*`` figures and tables.

Gauge status. Under the global positive-diagonal rescaling ``mu -> D mu, lv -> lv + 2 log D``
the divergences KL, Jeffreys, Bhattacharyya, Fisher-Rao and the distribution cosine are invariant
(they are invariant under any common affine map); W2 and raw cosine are not, so the section
evaluates them in D0 units, where they are invariant as well. Whether the rescaling is a symmetry
of the model is decided by :func:`vfe3.semantics.common.diagonal_redundancy_status`; with a
LayerNorm (affine or not) it is not, and D0 is only a documented unit convention. ``CV(tr Sigma)``
is reported in the trained coordinates (not invariant under the rescaling) and in D0 units.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vfe3.semantics.common import (
    OKABE_ITO,
    AnalysisContext,
    SectionResult,
    apply_style,
    diagonal_redundancy_status,
    markdown_table,
    panel_label,
    save_figure,
    write_table,
)


_SECTION = "sigma"
_LOG_2 = math.log(2.0)
_LOG_2PI = math.log(2.0 * math.pi)
_CV_GATE = 0.10                     # pre-registered calibration gate CV(tr Sigma) > 0.10
_AT_INIT_TOL = 1e-4                 # |lv - lv_init| below this on every coordinate = still at init
_N_NEIGHBORS = 20                   # Vilnis & McCallum (2015) Fig. 2 style neighbor lists
_N_KL_NEIGHBORS = 10
_MIN_PAIRS = 10                     # minimum resolved pairs for a similarity benchmark
_MIN_ENTAILMENT = 5                 # minimum resolved (hyponym, hypernym) pairs
_N_BOOT_CLUSTER = 2000              # cluster-bootstrap resamples (entailment, over hypernym groups)
_HEADLINE_SCORER = "neg_jeffreys"   # pre-registered headline Gaussian scorer (symmetric)
_FAILED_STEP: Dict[str, Any] = {"status": "failed", "reason": "raised; see metrics['failed_steps']"}


# ----------------------------------------------------------------------------------------------
# Array-level diagonal-Gaussian geometry
# ----------------------------------------------------------------------------------------------

def _f64(*arrays: Any) -> Tuple[np.ndarray, ...]:
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


def _log_cosh(x: np.ndarray) -> np.ndarray:
    r"""Overflow- and cancellation-free :math:`\log\cosh x`.

    :math:`\log\cosh x = \log(1 + 2\sinh^2(x/2))` for moderate :math:`|x|` (exact near 0) and
    :math:`|x| + \log(1 + e^{-2|x|}) - \log 2` for large :math:`|x|`.
    """
    ax = np.abs(x)
    small = ax < 20.0
    with np.errstate(over="ignore"):
        near = np.log1p(2.0 * np.square(np.sinh(0.5 * np.where(small, ax, 0.0))))
    far = ax + np.log1p(np.exp(-2.0 * ax)) - _LOG_2
    return np.where(small, near, far)


def kl_diag(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K) log-variances of N_a
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) KL(N_a || N_b)
    r"""Kullback-Leibler divergence of diagonal Gaussians.

    .. math::
        \mathrm{KL}(N_a\,\|\,N_b) = \tfrac12 \sum_k \Big[\frac{v_{ak}}{v_{bk}}
        + \frac{(\mu_{ak}-\mu_{bk})^2}{v_{bk}} - 1 - \log\frac{v_{ak}}{v_{bk}}\Big],

    evaluated as :math:`\tfrac12\sum_k[\mathrm{expm1}(d_k) - d_k + \Delta\mu_k^2 e^{-lv_{bk}}]`,
    :math:`d_k = lv_{ak} - lv_{bk}`, so identical inputs give exactly zero.
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    d = lv_a - lv_b
    per = np.expm1(d) - d + np.square(mu_a - mu_b) * np.exp(-lv_b)
    return 0.5 * per.sum(axis=-1)


def kl_one_vs_all(
    mu_q: np.ndarray,                    # (K,) or (m, K) query means
    lv_q: np.ndarray,                    # (K,) or (m, K) query log-variances
    MU:   np.ndarray,                    # (n, K) table means
    LV:   np.ndarray,                    # (n, K) table log-variances
) -> np.ndarray:                         # (n,) or (m, n): KL(q || p_x) for every table row x
    r"""KL from one (or ``m``) query Gaussians to every row of a table, by matrix products.

    With precisions :math:`P_{xk} = e^{-LV_{xk}}`,

    .. math::
        2\,\mathrm{KL}(q\,\|\,p_x) = \sum_k P_{xk} v_{qk} + \sum_k P_{xk}\mu_{qk}^2
        - 2\sum_k P_{xk} MU_{xk}\mu_{qk} + \sum_k P_{xk} MU_{xk}^2
        + \sum_k LV_{xk} - \sum_k lv_{qk} - K,

    i.e. three ``(n, K) @ (K,)`` products and two row sums, no loop over the table. Rounding can
    leave values of order ``1e-12`` below zero; they are floored at 0 (KL is non-negative).
    """
    mu_q, lv_q, MU, LV = _f64(mu_q, lv_q, MU, LV)
    single = mu_q.ndim == 1
    mu_q = np.atleast_2d(mu_q)                                  # (m, K)
    lv_q = np.atleast_2d(lv_q)                                  # (m, K)
    K = MU.shape[1]
    P = np.exp(-LV)                                             # (n, K) table precisions
    PM = P * MU                                                 # (n, K)
    row_const = (PM * MU).sum(axis=1) + LV.sum(axis=1) - K      # (n,)
    two_kl = (np.exp(lv_q) + np.square(mu_q)) @ P.T             # (m, n)
    two_kl -= 2.0 * (mu_q @ PM.T)
    two_kl += row_const[None, :] - lv_q.sum(axis=1)[:, None]
    kl = np.maximum(0.5 * two_kl, 0.0)
    return kl[0] if single else kl


def jeffreys_diag(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) KL(a||b) + KL(b||a)
    r"""Jeffreys divergence (symmetrized KL) of diagonal Gaussians.

    .. math::
        J = \sum_k \Big[2\sinh^2\tfrac{d_k}{2}
        + \tfrac12 \Delta\mu_k^2\big(e^{-lv_{ak}} + e^{-lv_{bk}}\big)\Big],
        \qquad d_k = lv_{ak} - lv_{bk}

    (the log-determinant terms cancel; :math:`\cosh d - 1 = 2\sinh^2(d/2)`).
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    d = lv_a - lv_b
    per = 2.0 * np.square(np.sinh(0.5 * d)) \
        + 0.5 * np.square(mu_a - mu_b) * (np.exp(-lv_a) + np.exp(-lv_b))
    return per.sum(axis=-1)


def w2_sq_diag(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) squared 2-Wasserstein distance
    r"""Squared 2-Wasserstein distance, exact for commuting (here diagonal) covariances.

    .. math::
        W_2^2 = \|\mu_a - \mu_b\|^2 + \|\sqrt{v_a} - \sqrt{v_b}\|^2,

    with :math:`\sqrt{v_a}-\sqrt{v_b} = e^{lv_b/2}\,\mathrm{expm1}((lv_a-lv_b)/2)`.
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    d_sigma = np.exp(0.5 * lv_b) * np.expm1(0.5 * (lv_a - lv_b))
    return (np.square(mu_a - mu_b) + np.square(d_sigma)).sum(axis=-1)


def bhattacharyya_diag(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) Bhattacharyya distance D_B
    r"""Bhattacharyya distance :math:`D_B = -\log\int\sqrt{p_a p_b}` of diagonal Gaussians.

    .. math::
        D_B = \sum_k\Big[\frac{\Delta\mu_k^2}{4(v_{ak}+v_{bk})}
        + \tfrac12\log\frac{(v_{ak}+v_{bk})/2}{\sqrt{v_{ak}v_{bk}}}\Big]
        = \sum_k\Big[\frac{\Delta\mu_k^2}{4(v_{ak}+v_{bk})} + \tfrac12\log\cosh\tfrac{d_k}{2}\Big].
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    inv_sum = np.exp(-np.logaddexp(lv_a, lv_b))                 # 1 / (v_a + v_b)
    per = 0.25 * np.square(mu_a - mu_b) * inv_sum + 0.5 * _log_cosh(0.5 * (lv_a - lv_b))
    return per.sum(axis=-1)


def hellinger_sq_diag(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) squared Hellinger distance in [0, 1]
    r"""Squared Hellinger distance :math:`H^2 = 1 - \int\sqrt{p_a p_b} = 1 - e^{-D_B}`."""
    return -np.expm1(-bhattacharyya_diag(mu_a, lv_a, mu_b, lv_b))


def fisher_rao_diag(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) Fisher-Rao geodesic distance
    r"""Fisher-Rao distance within the family of diagonal Gaussians.

    The Fisher metric of :math:`N(\mu,\sigma^2)` is :math:`ds^2 = (d\mu^2 + 2\,d\sigma^2)/\sigma^2`,
    i.e. :math:`\sqrt2\times` the Poincare half plane in :math:`(\mu/\sqrt2, \sigma)`; the diagonal
    family is the Riemannian product of K such planes, so

    .. math::
        d_k = \sqrt2\,\operatorname{arccosh}\Big(1 + \frac{\Delta\mu_k^2/2
        + (\sigma_{ak}-\sigma_{bk})^2}{2\sigma_{ak}\sigma_{bk}}\Big), \qquad
        d = \Big(\sum_k d_k^2\Big)^{1/2}.

    Exact inside the diagonal family and an upper bound on the full-covariance Fisher-Rao
    distance. The argument is evaluated as
    :math:`x_k = \Delta\mu_k^2 e^{-(lv_{ak}+lv_{bk})/2}/4 + 2\sinh^2(d_k/4)` and
    :math:`\operatorname{arccosh}(1+x) = \log(1 + x + \sqrt{x(x+2)})` (stable for small x).
    Special cases: equal means give :math:`\sqrt2|\log(\sigma_b/\sigma_a)|`; equal variances give
    :math:`2\sqrt2\,\operatorname{arcsinh}(|\Delta\mu|/(2\sqrt2\sigma))`.
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    x = 0.25 * np.square(mu_a - mu_b) * np.exp(-0.5 * (lv_a + lv_b)) \
        + 2.0 * np.square(np.sinh(0.25 * (lv_a - lv_b)))
    d_k = math.sqrt(2.0) * np.log1p(x + np.sqrt(x * (x + 2.0)))
    return np.sqrt(np.square(d_k).sum(axis=-1))


def log_expected_likelihood(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)
) -> np.ndarray:                         # (...) log of the probability-product kernel
    r"""Log expected-likelihood kernel (Vilnis & McCallum 2015).

    .. math::
        \log\!\int N_a(x) N_b(x)\,dx = \log N(\mu_a;\mu_b, \Sigma_a+\Sigma_b)
        = -\tfrac12\sum_k\Big[\log(v_{ak}+v_{bk}) + \frac{\Delta\mu_k^2}{v_{ak}+v_{bk}}
        + \log 2\pi\Big].
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    log_sum = np.logaddexp(lv_a, lv_b)
    per = log_sum + np.square(mu_a - mu_b) * np.exp(-log_sum) + _LOG_2PI
    return -0.5 * per.sum(axis=-1)


def distribution_cosine(
    mu_a: np.ndarray,                    # (..., K)
    lv_a: np.ndarray,                    # (..., K)
    mu_b: np.ndarray,                    # (..., K)
    lv_b: np.ndarray,                    # (..., K)

    *,
    log: bool = False,
) -> np.ndarray:                         # (...) cosine in (0, 1], or its log when log=True
    r"""Normalized expected-likelihood kernel ("distribution cosine", Vilnis & McCallum 2015).

    .. math::
        \cos(N_a, N_b) = \frac{E(a,b)}{\sqrt{E(a,a)E(b,b)}}, \qquad
        \log\cos = -\tfrac12\sum_k\Big[\log\cosh\tfrac{d_k}{2}
        + \frac{\Delta\mu_k^2}{v_{ak}+v_{bk}}\Big].

    In high dimension the cosine itself underflows; ``log=True`` returns the log (use it for
    rank statistics).
    """
    mu_a, lv_a, mu_b, lv_b = _f64(mu_a, lv_a, mu_b, lv_b)
    per = _log_cosh(0.5 * (lv_a - lv_b)) + np.square(mu_a - mu_b) * np.exp(-np.logaddexp(lv_a, lv_b))
    log_cos = -0.5 * per.sum(axis=-1)
    return log_cos if log else np.exp(log_cos)


def sigma_units(
    mu:          np.ndarray,             # (n, K) means
    log_var_ref: np.ndarray,             # (n_ref, K) log-variances of the reference rows
) -> Tuple[np.ndarray, np.ndarray]:      # (mu * d0 (n, K), log_d0 (K,))
    r"""D0 unit chart: :math:`\log d_{0,k} = -\tfrac12\,\mathrm{mean}_{v\in\mathrm{ref}}\,lv_{vk}`.

    Returns :math:`\mu\,d_0` and :math:`\log d_0`. Under the global rescaling
    :math:`\mu\to D\mu`, :math:`lv\to lv + 2\log D` one has :math:`\log d_0\to\log d_0-\log D`, so
    :math:`\mu d_0` and :math:`lv + 2\log d_0` are invariant (a gauge fixing when the rescaling is a
    symmetry of the model, a unit convention otherwise).
    """
    mu, log_var_ref = _f64(mu, log_var_ref)
    log_d0 = -0.5 * log_var_ref.mean(axis=0)                   # (K,)
    return mu * np.exp(log_d0), log_d0


def w2_chart(
    mu:      np.ndarray,                 # (n, K)
    log_var: np.ndarray,                 # (n, K)
    log_d0:  Optional[np.ndarray] = None,  # (K,) optional D0 units
) -> np.ndarray:                         # (n, 2K) [mu d0, sqrt(v) d0]
    r"""Euclidean W2 chart :math:`(\mu, \sqrt v)`: :math:`\|z_a - z_b\|^2 = W_2^2(N_a, N_b)`.

    With ``log_d0`` both halves are expressed in D0 units, :math:`(\mu d_0, \sqrt v\, d_0)`.
    """
    mu, log_var = _f64(mu, log_var)
    d0 = np.ones(mu.shape[-1]) if log_d0 is None else np.exp(np.asarray(log_d0, dtype=np.float64))
    return np.concatenate([mu * d0, np.exp(0.5 * log_var) * d0], axis=-1)


# ----------------------------------------------------------------------------------------------
# Small statistics helpers
# ----------------------------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    from vfe3.semantics import stats
    return dict(stats.spearman_ci(np.asarray(x, dtype=np.float64),
                                  np.asarray(y, dtype=np.float64), rng))


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Pearson correlation of matching rows of ``a`` and ``b`` (NaN for constant rows)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    return float(_row_pearson(rankdata(x)[None], rankdata(y)[None])[0])


def _paired_spearman_delta(
    score: np.ndarray,                   # (m,) candidate scorer
    ref:   np.ndarray,                   # (m,) reference scorer
    human: np.ndarray,                   # (m,) human ratings

    *,
    rng:    np.random.Generator,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    r"""Paired bootstrap of :math:`\Delta\rho = \rho_s(\mathrm{score}, h) - \rho_s(\mathrm{ref}, h)`.

    Pairs are resampled jointly, so the two correlations share every resample; two-sided
    :math:`p = 2\min(P^*(\Delta\le0), P^*(\Delta\ge0))`.
    """
    from scipy.stats import rankdata
    m = int(human.size)
    point = _spearman_rho(score, human) - _spearman_rho(ref, human)
    idx = rng.integers(0, m, size=(n_boot, m))                  # (B, m) resampled pair indices
    rh = rankdata(human[idx], axis=1)
    deltas = _row_pearson(rankdata(score[idx], axis=1), rh) - _row_pearson(rankdata(ref[idx], axis=1), rh)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"delta": point, "ci_lo": None, "ci_hi": None, "p": None, "n": m, "n_boot": 0}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean())))
    return {"delta": point, "ci_lo": float(lo), "ci_hi": float(hi), "p": p, "n": m,
            "n_boot": int(deltas.size)}


def _attempt(failed: List[Dict[str, str]], notes: List[str], step: str, fn: Callable[[], Any],
             default: Any) -> Any:
    r"""Run one sub-step; if it raises, record ``{'step', 'error'}`` in ``failed`` (error as
    ``'Type: message'``), add a note and return ``default`` so the rest of the section continues."""
    try:
        return fn()
    except Exception as exc:                           # recorded; the section continues
        error = f"{type(exc).__name__}: {exc}"
        failed.append({"step": step, "error": error})
        notes.append(f"{step} failed ({error}); the rest of the section continues")
        return default


def _cluster_bootstrap(
    correct: np.ndarray,                 # (m, n) 0/1 outcome rows (one per method) over n pairs
    cluster: np.ndarray,                 # (n,) cluster label of each pair
    rng:     np.random.Generator,

    *,
    n_boot: int = _N_BOOT_CLUSTER,
) -> np.ndarray:                         # (B, m) resampled accuracies
    r"""Cluster bootstrap of pair accuracies: resample the :math:`G` clusters with replacement.

    With cluster sizes :math:`n_c` and hit counts :math:`k_{ic}` of method :math:`i`, one resample
    draws :math:`c_1,\dots,c_G` uniformly with replacement and gives the pair-weighted accuracy

    .. math::
        \hat a_i^* = \sum_{g=1}^G k_{i c_g} \Big/ \sum_{g=1}^G n_{c_g},

    so pairs that share a cluster (and hence a hypernym's :math:`\log\det\Sigma` and count) are kept
    together; the methods share every resample, which pairs their comparison.
    """
    _, inv = np.unique(cluster, return_inverse=True)
    G = int(inv.max()) + 1
    n_c = np.bincount(inv, minlength=G).astype(np.float64)                        # (G,)
    k_c = np.stack([np.bincount(inv, weights=np.asarray(row, dtype=np.float64), minlength=G)
                    for row in correct])                                          # (m, G)
    idx = rng.integers(0, G, size=(n_boot, G))                                    # (B, G)
    return k_c[:, idx].sum(axis=2).T / n_c[idx].sum(axis=1)[:, None]              # (B, m)


def _bootstrap_interval(boot: np.ndarray, null: float, n_clusters: int) -> Dict[str, Any]:
    r"""Percentile 95% interval of ``boot`` and the two-sided bootstrap p-value against ``null``
    :math:`a_0`,

    .. math::
        p = \min\Big(1,\ 2\min\Big(\frac{1 + \#\{\hat a^*\le a_0\}}{B+1},
        \frac{1 + \#\{\hat a^*\ge a_0\}}{B+1}\Big)\Big),

    which never reports :math:`p = 0` from :math:`B` resamples. With fewer than two clusters the
    resamples are all identical, so no interval is reported.
    """
    if n_clusters < 2:
        return {"ci_lo": None, "ci_hi": None, "p": None, "n_clusters": n_clusters, "n_boot": 0}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    B = boot.size
    p = min(1.0, 2.0 * min((1 + int((boot <= null).sum())) / (B + 1),
                           (1 + int((boot >= null).sum())) / (B + 1)))
    return {"ci_lo": float(lo), "ci_hi": float(hi), "p": p, "n_clusters": n_clusters,
            "n_boot": int(boot.size)}


def _cluster_proportion(ok: np.ndarray, boot: np.ndarray, n_clusters: int) -> Dict[str, Any]:
    r"""Accuracy of ``ok`` with its cluster-bootstrap 95% interval and bootstrap p against 1/2."""
    k, n = int(ok.sum()), int(ok.size)
    interval = _bootstrap_interval(boot, 0.5, n_clusters)
    return {"acc": k / n, "k": k, "n": n, "ci_lo": interval["ci_lo"], "ci_hi": interval["ci_hi"],
            "p_vs_chance": interval["p"], "n_clusters": n_clusters,
            "ci_method": "cluster bootstrap over hypernym groups"}


def _binned_medians(x: np.ndarray, y: np.ndarray, n_bins: int = 12) -> List[Dict[str, Any]]:
    r"""Median and interquartile range of ``y`` in quantile bins of ``x``."""
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return []
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        sel = which == b
        if not sel.any():
            continue
        q25, q50, q75 = np.quantile(y[sel], [0.25, 0.5, 0.75])
        rows.append({"x_median": float(np.median(x[sel])), "y_median": float(q50),
                     "y_q25": float(q25), "y_q75": float(q75), "n": int(sel.sum())})
    return rows


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""Residual of the ordinary least-squares fit :math:`y \approx a + b x`."""
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _kruskal(groups: Sequence[np.ndarray]) -> Dict[str, Any]:
    from scipy.stats import kruskal
    usable = [g for g in groups if g.size >= 2]
    if len(usable) < 2 or np.ptp(np.concatenate(usable)) == 0.0:
        return {"H": None, "p": None, "n_groups": len(usable)}
    test = kruskal(*usable)
    return {"H": float(test.statistic), "p": float(test.pvalue), "n_groups": len(usable)}


def _init_log_var(cfg: Any) -> float:
    r"""Log of the covariance diagonal the family assigns to the initial slot value ``sigma_init``.

    The stored slot starts at ``log(sigma_init)``; the dispersion ``sigma_init`` maps to a variance
    through the family hook (the Gaussian families: itself).
    """
    import torch
    from vfe3.families.base import get_family       # importing the package registers the families

    family = get_family(str(cfg.family))
    dispersion = torch.full((1, 1), float(cfg.sigma_init), dtype=torch.float64)
    if family.cov_kind == "full":
        dispersion = torch.diag_embed(dispersion)               # (1, 1, 1)
    variance = family.covariance_diagonal(dispersion)
    return float(torch.log(variance).reshape(-1)[0])


# ----------------------------------------------------------------------------------------------
# One bank (prior or Gaussian decode)
# ----------------------------------------------------------------------------------------------

_SCORERS: Tuple[Tuple[str, str], ...] = (
    ("cos_mu",          "cos(μ), D0 units"),
    ("cos_mu_raw",      "cos(μ), raw chart"),
    ("neg_euclid_mu",   "−‖Δμ‖, D0 units"),
    ("neg_w2",          "−W2, D0 units"),
    ("neg_jeffreys",    "−Jeffreys"),
    ("neg_fisher_rao",  "−Fisher–Rao"),
    ("log_dist_cosine", "log distribution cosine"),
    ("neg_kl_12",       "−KL(w1‖w2)"),
    ("neg_kl_21",       "−KL(w2‖w1)"),
)
# Sigma-free references: cos(mu) and the Euclidean mean distance. With a near-constant Sigma every
# divergence reduces to a monotone function of ||Delta mu||, so a Gaussian scorer that beats cos(mu)
# but not -||Delta mu|| reflects Euclidean-vs-angular geometry of mu, not information in Sigma.
_REFERENCE_SCORERS = ("cos_mu", "cos_mu_raw", "neg_euclid_mu")


def _centered_unit(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.where(norms > 0.0, norms, 1.0)


def _dispersion(lv: np.ndarray, log_d0: np.ndarray, logdet: np.ndarray) -> Dict[str, Any]:
    r"""CV of :math:`\mathrm{tr}\,\Sigma`, log-determinant and per-coordinate summaries.

    ``cv_trace`` (and the gate) is computed in the trained coordinates, which the global rescaling
    :math:`lv \to lv + 2\log D` changes; ``cv_trace_d0`` uses
    :math:`\mathrm{tr}\,\Sigma^{D_0} = \sum_k e^{lv_k + 2\log d_{0,k}}`, which it leaves invariant.
    ``logdet`` is :math:`\log\det\Sigma_v` per row (the row sum of ``lv`` for a diagonal table).
    """
    trace = np.exp(lv).sum(axis=1)                              # (n,) trained coordinates
    trace_d0 = np.exp(lv + 2.0 * log_d0).sum(axis=1)            # (n,) D0 units
    q = [0.01, 0.25, 0.5, 0.75, 0.99]
    cv = float(trace.std() / trace.mean())
    return {
        "cv_trace": cv,
        "cv_trace_d0": float(trace_d0.std() / trace_d0.mean()),
        "gate_cv_gt_0.10": bool(cv > _CV_GATE),
        "trace_mean": float(trace.mean()),
        "logdet_mean": float(logdet.mean()),
        "logdet_std": float(logdet.std()),
        "logdet_quantiles": dict(zip([str(v) for v in q], np.quantile(logdet, q).tolist())),
        "log_var_quantiles": dict(zip([str(v) for v in q], np.quantile(lv, q).tolist())),
        "per_coordinate_token_std_mean": float(lv.std(axis=0).mean()),
    }


def _init_fractions(lv_all: np.ndarray, lv_init: float, cand_mask: np.ndarray,
                    counts: Optional[np.ndarray]) -> Dict[str, Any]:
    at_init = np.all(np.abs(lv_all - lv_init) < _AT_INIT_TOL, axis=1)   # (V,)
    out: Dict[str, Any] = {"log_var_init": lv_init, "tolerance": _AT_INIT_TOL,
                           "frac_at_init_candidates": float(at_init[cand_mask].mean()),
                           "frac_at_init_all": float(at_init.mean())}
    if counts is not None:
        zero = counts == 0
        out["n_count0"] = int(zero.sum())
        out["frac_at_init_count0"] = float(at_init[zero].mean()) if zero.any() else None
        out["frac_at_init_count_pos"] = float(at_init[~zero].mean()) if (~zero).any() else None
    return out


def _table_geometry(mu_d0: np.ndarray, lv: np.ndarray) -> Dict[str, Any]:
    r"""Effective rank / PCA spectrum of the centered log-variance table and CKA(mu, lv).

    ``global_scale_share`` is the fraction of the centered table's variance along the uniform
    direction :math:`\mathbf 1/\sqrt K` (a per-token overall scale); ``pc1_uniform_alignment`` is
    :math:`|\langle \mathrm{PC}_1, \mathbf 1/\sqrt K\rangle|`.
    """
    from vfe3.semantics import stats
    K = lv.shape[1]
    lvc = lv - lv.mean(axis=0, keepdims=True)
    total = float(np.square(lvc).sum())
    u = np.full(K, 1.0 / math.sqrt(K))
    _, _, Vt = np.linalg.svd(lvc, full_matrices=False)
    return {
        "effective_rank": float(stats.effective_rank(lv)),
        "pca_spectrum": np.asarray(stats.pca_spectrum(lv, k=min(50, K)), dtype=np.float64),
        "global_scale_share": float(np.square(lvc @ u).sum() / total) if total > 0 else None,
        "pc1_uniform_alignment": float(abs(Vt[0] @ u)),
        "cka_mu_logvar": float(stats.linear_cka(mu_d0, lv)),
    }


def _variance_sorted_neighbors(ctx: AnalysisContext, ids: np.ndarray, row_of: Dict[int, int],
                               mu: np.ndarray, lv: np.ndarray, mu_d0: np.ndarray,
                               logdet: np.ndarray) -> List[Dict[str, Any]]:
    r"""Top cosine neighbors (centered, D0 units) of each probe, re-sorted by log det Sigma.

    Also lists the nearest rows by :math:`\mathrm{KL}(\pi_{probe}\,\|\,\pi_x)`.
    """
    probes = [int(t) for t in ctx.probe_ids if int(t) in row_of]
    if not probes:
        return []
    rows_p = np.array([row_of[t] for t in probes], dtype=np.int64)
    Xn = _centered_unit(mu_d0)                                  # (n, K)
    S = Xn[rows_p] @ Xn.T                                       # (P, n) cosines
    S[np.arange(rows_p.size), rows_p] = -np.inf
    KL = kl_one_vs_all(mu[rows_p], lv[rows_p], mu, lv)          # (P, n)
    KL[np.arange(rows_p.size), rows_p] = np.inf
    z = (logdet - logdet.mean()) / (logdet.std() if logdet.std() > 0 else 1.0)
    k = min(_N_NEIGHBORS, ids.size - 1)
    k_kl = min(_N_KL_NEIGHBORS, ids.size - 1)
    out = []
    for i, t in enumerate(probes):
        top = np.argpartition(-S[i], k - 1)[:k]
        top = top[np.argsort(logdet[top], kind="stable")]       # narrowest -> broadest
        top_kl = np.argsort(KL[i], kind="stable")[:k_kl]
        out.append({
            "probe": ctx.display(t), "probe_id": t, "probe_logdet_z": float(z[rows_p[i]]),
            "neighbors": [{"id": int(ids[r]), "word": ctx.display(int(ids[r])),
                           "cos": float(S[i, r]), "logdet_z": float(z[r])} for r in top],
            "kl_neighbors": [{"id": int(ids[r]), "word": ctx.display(int(ids[r])),
                              "kl": float(KL[i, r])} for r in top_kl],
        })
    return out


def _entailment(ctx: AnalysisContext, row_of: Dict[int, int], mu: np.ndarray, lv: np.ndarray,
                logdet: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    r"""Hypernymy direction from KL asymmetry, against frequency and log-det baselines.

    A pair (hyponym h, hypernym H) is called correctly when
    :math:`\mathrm{KL}(\pi_h\|\pi_H) < \mathrm{KL}(\pi_H\|\pi_h)` (the specific density fits inside
    the general one; Vilnis & McCallum 2015, Brazinskas et al. 2018). Baselines: the more frequent
    word is the hypernym; the word with larger :math:`\log\det\Sigma` is the hypernym. Ties count as
    errors. Pairs that share a hypernym reuse its :math:`\log\det\Sigma_H` and count, so they are
    not independent: every interval, the p-values against 1/2 and the paired KL-minus-frequency
    difference come from a cluster bootstrap over hypernym groups (:func:`_cluster_bootstrap`).
    """
    pairs = [(int(h), int(H)) for h, H in ctx.hypernyms if int(h) in row_of and int(H) in row_of]
    if len(pairs) < _MIN_ENTAILMENT:
        return {"status": "skipped", "n": len(pairs),
                "reason": f"fewer than {_MIN_ENTAILMENT} resolved (hyponym, hypernym) pairs"}
    rh = np.array([row_of[h] for h, _ in pairs], dtype=np.int64)
    rH = np.array([row_of[H] for _, H in pairs], dtype=np.int64)
    hyper = np.array([H for _, H in pairs], dtype=np.int64)     # (n,) cluster label per pair
    n_clusters = int(np.unique(hyper).size)
    kl_hH = kl_diag(mu[rh], lv[rh], mu[rH], lv[rH])             # KL(hypo || hyper)
    kl_Hh = kl_diag(mu[rH], lv[rH], mu[rh], lv[rh])             # KL(hyper || hypo)
    ok_kl = kl_hH < kl_Hh
    ok_logdet = logdet[rH] > logdet[rh]
    ok_freq: Optional[np.ndarray] = None
    if ctx.counts is not None:
        c = ctx.counts.astype(np.int64)
        ok_freq = c[hyper] > c[[h for h, _ in pairs]]
    methods = [ok_kl, ok_logdet] + ([ok_freq] if ok_freq is not None else [])
    boot = _cluster_bootstrap(np.stack(methods), hyper, rng)    # (B, 2 or 3)
    out: Dict[str, Any] = {
        "status": "ok", "n": len(pairs), "n_clusters": n_clusters,
        "kl_asymmetry": _cluster_proportion(ok_kl, boot[:, 0], n_clusters),
        "logdet_baseline": _cluster_proportion(ok_logdet, boot[:, 1], n_clusters),
        "kl_agrees_with_logdet": float((ok_kl == ok_logdet).mean()),
    }
    if ok_freq is not None:
        out["frequency_baseline"] = _cluster_proportion(ok_freq, boot[:, 2], n_clusters)
        out["kl_vs_frequency_discordant"] = {"kl_only": int((ok_kl & ~ok_freq).sum()),
                                             "frequency_only": int((~ok_kl & ok_freq).sum())}
        out["kl_minus_frequency"] = {
            "delta": float(ok_kl.mean() - ok_freq.mean()),
            **_bootstrap_interval(boot[:, 0] - boot[:, 2], 0.0, n_clusters),
            "method": "paired cluster bootstrap over hypernym groups"}
    else:
        out["frequency_baseline"] = {"status": "skipped", "reason": "training counts unavailable"}
    per_hyper: Dict[str, Dict[str, Any]] = {}
    for H in sorted({H for _, H in pairs}):
        sel = np.array([p[1] == H for p in pairs])
        entry = {"n": int(sel.sum()), "acc_kl": float(ok_kl[sel].mean())}
        if ok_freq is not None:
            entry["acc_frequency"] = float(ok_freq[sel].mean())
        per_hyper[ctx.display(H)] = entry
    out["per_hypernym"] = per_hyper
    out["pairs"] = [{
        "hyponym": ctx.display(h), "hypernym": ctx.display(H),
        "kl_hypo_hyper": float(kl_hH[i]), "kl_hyper_hypo": float(kl_Hh[i]),
        "logdet_hypo": float(logdet[rh[i]]), "logdet_hyper": float(logdet[rH[i]]),
        "count_hypo": int(ctx.counts[h]) if ctx.counts is not None else None,
        "count_hyper": int(ctx.counts[H]) if ctx.counts is not None else None,
        "kl_correct": bool(ok_kl[i]),
        "frequency_correct": bool(ok_freq[i]) if ok_freq is not None else None,
    } for i, (h, H) in enumerate(pairs)]
    return out


def _pair_scores(r1: np.ndarray, r2: np.ndarray, mu: np.ndarray, mu_d0: np.ndarray,
                 lv_d0: np.ndarray) -> Dict[str, np.ndarray]:
    r"""Every scorer of :data:`_SCORERS` on row pairs (higher = more similar)."""
    Xd = _centered_unit(mu_d0)
    Xr = _centered_unit(mu)
    a = (mu_d0[r1], lv_d0[r1], mu_d0[r2], lv_d0[r2])
    return {
        "cos_mu": (Xd[r1] * Xd[r2]).sum(axis=1),
        "cos_mu_raw": (Xr[r1] * Xr[r2]).sum(axis=1),
        "neg_euclid_mu": -np.linalg.norm(mu_d0[r1] - mu_d0[r2], axis=1),
        "neg_w2": -np.sqrt(w2_sq_diag(*a)),
        "neg_jeffreys": -jeffreys_diag(*a),
        "neg_fisher_rao": -fisher_rao_diag(*a),
        "log_dist_cosine": distribution_cosine(*a, log=True),
        "neg_kl_12": -kl_diag(*a),
        "neg_kl_21": -kl_diag(a[2], a[3], a[0], a[1]),
    }


def _similarity(ctx: AnalysisContext, row_of: Dict[int, int], mu: np.ndarray, mu_d0: np.ndarray,
                lv_d0: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    r"""Spearman of each scorer with human ratings; paired :math:`\Delta\rho` vs cos(mu) for every
    scorer and, for the Sigma-using scorers, also vs the Sigma-free control :math:`-\|\Delta\mu\|`.

    Identical-word pairs (e.g. WordSim's "tiger tiger") are dropped, as in the vector battery, so
    both score the same pairs. The headline scorer is pre-registered (:data:`_HEADLINE_SCORER`,
    the symmetric :math:`-J`); ``best_gaussian`` is the post-hoc maximum over the Sigma-using
    scorers and is labeled as not adjusted for selection.
    """
    out: Dict[str, Any] = {}
    for bench, triples in ctx.similarity.items():
        n_identical = sum(1 for a, b, _ in triples if int(a) == int(b))
        kept = [(row_of[int(a)], row_of[int(b)], float(h)) for a, b, h in triples
                if int(a) != int(b) and int(a) in row_of and int(b) in row_of]
        if len(kept) < _MIN_PAIRS:
            out[bench] = {"status": "skipped", "n": len(kept), "n_identical_dropped": n_identical,
                          "reason": f"fewer than {_MIN_PAIRS} resolved pairs"}
            continue
        r1 = np.array([k[0] for k in kept], dtype=np.int64)
        r2 = np.array([k[1] for k in kept], dtype=np.int64)
        human = np.array([k[2] for k in kept], dtype=np.float64)
        scores = _pair_scores(r1, r2, mu, mu_d0, lv_d0)
        per: Dict[str, Any] = {}
        for key, label in _SCORERS:
            entry = _spearman(scores[key], human, rng)
            entry["label"] = label
            if key != "cos_mu":
                entry["delta_vs_cos_mu"] = _paired_spearman_delta(scores[key], scores["cos_mu"],
                                                                  human, rng=rng)
            if key not in _REFERENCE_SCORERS:
                entry["delta_vs_euclid_mu"] = _paired_spearman_delta(
                    scores[key], scores["neg_euclid_mu"], human, rng=rng)
            per[key] = entry
        gaussian = [k for k, _ in _SCORERS if k not in _REFERENCE_SCORERS
                    and per[k].get("rho") is not None and np.isfinite(per[k]["rho"])]
        best = max(gaussian, key=lambda k: per[k]["rho"]) if gaussian else None
        out[bench] = {"status": "ok", "n": len(kept), "n_listed": len(triples),
                      "n_identical_dropped": n_identical, "scorers": per,
                      "headline_scorer": _HEADLINE_SCORER, "best_gaussian": best,
                      "best_gaussian_note": f"maximum over {len(gaussian)} scorers, "
                                            "not adjusted for selection"}
    return out


def _ok_benches(sim: Dict[str, Any]) -> List[str]:
    r"""Benchmarks of a similarity result that were scored (a failed step leaves no entries)."""
    return [b for b, v in sim.items() if isinstance(v, dict) and v.get("status") == "ok"]


def _token_classes(run: Any, ctx: AnalysisContext, logdet_all: np.ndarray,
                   cand_mask: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    r"""log det Sigma per token class over all valid frequent tokens (not only whole words);
    ``logdet_all`` is (V,) :math:`\log\det\Sigma_v`."""
    codes, names = run.vocab.token_classes()
    codes = np.asarray(codes)
    if ctx.counts is not None:
        min_count = int(ctx.options.get("min_count", 100))
        mask = np.asarray(run.vocab.valid_text, dtype=bool) & (ctx.counts >= max(min_count, 1))
        population = f"valid-text tokens with training count >= {max(min_count, 1)}"
    else:
        mask = cand_mask
        population = "candidate words (training counts unavailable)"
    groups: Dict[str, np.ndarray] = {}
    summary: Dict[str, Any] = {}
    for c, name in enumerate(names):
        vals = logdet_all[mask & (codes == c)]
        if vals.size == 0:
            continue
        groups[name] = vals
        q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        summary[name] = {"n": int(vals.size), "median": float(q50), "q25": float(q25),
                         "q75": float(q75)}
    return ({"population": population, "n": int(mask.sum()), "per_class": summary,
             "kruskal": _kruskal(list(groups.values()))}, groups)


def _fields(ctx: AnalysisContext, row_of: Dict[int, int], logdet: np.ndarray,
            log_count: Optional[np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    r"""log det Sigma by semantic field, raw and after regressing out log count (OLS)."""
    resid = _residualize(logdet, log_count) if log_count is not None else None
    groups: Dict[str, np.ndarray] = {}
    groups_resid: Dict[str, np.ndarray] = {}
    summary: Dict[str, Any] = {}
    for name, members in ctx.lexicon.items():
        rows = np.array([row_of[int(t)] for t in members if int(t) in row_of], dtype=np.int64)
        if rows.size == 0:
            continue
        groups[name] = logdet[rows]
        entry = {"n": int(rows.size), "median": float(np.median(logdet[rows]))}
        if resid is not None:
            groups_resid[name] = resid[rows]
            entry["median_freq_residual"] = float(np.median(resid[rows]))
        summary[name] = entry
    out = {"per_field": summary, "kruskal": _kruskal(list(groups.values()))}
    out["kruskal_freq_residual"] = (_kruskal(list(groups_resid.values())) if resid is not None
                                    else {"status": "skipped", "reason": "training counts unavailable"})
    return out, groups


def _bank_battery(run: Any, ctx: AnalysisContext, mu_fn: Callable[[np.ndarray], np.ndarray],
                  lv_fn: Callable[[np.ndarray], np.ndarray], lv_init: float,
                  rng: np.random.Generator, *, bank: str, failed: List[Dict[str, str]],
                  notes: List[str],
                  logdet_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""Items 1-6 of the Sigma section for one bank. Returns (metrics, arrays for figures).

    Loading the tables and the dispersion summary are required; every later item that raises is
    recorded in ``failed`` (step ``"<bank>: <item>"``) and replaced by a failed-status entry.
    ``logdet_fn`` gives (n,) :math:`\log\det\Sigma_v` for a covariance that is not diagonal (the
    per-head ``block_cholesky`` blocks); None takes the row sum of ``lv_fn``.
    """
    ids = np.asarray(ctx.candidate_ids, dtype=np.int64)
    row_of = {int(t): r for r, t in enumerate(ids)}
    V = int(run.vocab.size)
    mu = np.asarray(mu_fn(ids), dtype=np.float64)              # (n, K)
    lv_all = np.asarray(lv_fn(np.arange(V, dtype=np.int64)), dtype=np.float64)   # (V, K)
    lv = lv_all[ids]                                            # (n, K)
    logdet_all = (lv_all.sum(axis=1) if logdet_fn is None
                  else np.asarray(logdet_fn(np.arange(V, dtype=np.int64)), dtype=np.float64))   # (V,)
    if not (np.isfinite(mu).all() and np.isfinite(lv_all).all() and np.isfinite(logdet_all).all()):
        raise ValueError("non-finite values in the mean, log-variance or log-determinant table")
    mu_d0, log_d0 = sigma_units(mu, lv)
    lv_d0 = lv + 2.0 * log_d0                                   # (n, K) D0-unit log-variances
    logdet = logdet_all[ids]                                    # (n,)
    log_count = ctx.log_count(ids)
    cand_mask = np.zeros(V, dtype=bool)
    cand_mask[ids] = True

    metrics: Dict[str, Any] = {"n_candidates": int(ids.size), "log_d0": log_d0,
                               "dispersion": _dispersion(lv, log_d0, logdet),
                               "init": _init_fractions(lv_all, lv_init, cand_mask, ctx.counts)}
    def step(name: str, fn: Callable[[], Any], default: Any) -> Any:
        return _attempt(failed, notes, f"{bank}: {name}", fn, default)

    binned: List[Dict[str, Any]] = []
    if log_count is not None:
        binned = step("binned log det medians", lambda: _binned_medians(log_count, logdet), [])
        metrics["frequency"] = step("frequency", lambda: {
            "logdet_vs_log_count": _spearman(logdet, log_count, rng),
            "trace_vs_log_count": _spearman(np.exp(lv).sum(axis=1), log_count, rng),
            "binned_medians": binned,
        }, dict(_FAILED_STEP))
    else:
        metrics["frequency"] = {"status": "skipped", "reason": "training counts unavailable"}
    metrics["classes"], class_groups = step(
        "token classes", lambda: _token_classes(run, ctx, logdet_all, cand_mask), (dict(_FAILED_STEP), {}))
    metrics["fields"], field_groups = step(
        "semantic fields", lambda: _fields(ctx, row_of, logdet, log_count), (dict(_FAILED_STEP), {}))
    metrics["neighbors"] = step(
        "variance-sorted neighbors",
        lambda: _variance_sorted_neighbors(ctx, ids, row_of, mu, lv, mu_d0, logdet), [])
    metrics["entailment"] = step(
        "entailment", lambda: _entailment(ctx, row_of, mu, lv, logdet, rng), dict(_FAILED_STEP))
    metrics["similarity"] = step(
        "similarity", lambda: _similarity(ctx, row_of, mu, mu_d0, lv_d0, rng), dict(_FAILED_STEP))
    metrics["table_geometry"] = step(
        "table geometry", lambda: _table_geometry(mu_d0, lv), dict(_FAILED_STEP))
    arrays = {"lv": lv, "logdet": logdet, "log_count": log_count, "binned": binned,
              "lv_init": lv_init, "class_groups": class_groups, "field_groups": field_groups}
    return metrics, arrays


# ----------------------------------------------------------------------------------------------
# Figures and tables
# ----------------------------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.3f}"


def _fmt_ci(entry: Dict[str, Any], key: str = "rho") -> str:
    return f"{_fmt(entry.get(key))} [{_fmt(entry.get('ci_lo'))}, {_fmt(entry.get('ci_hi'))}]"


def _plot_overview(run: Any, ctx: AnalysisContext, m: Dict[str, Any], arr: Dict[str, Any],
                   title: str, name: str) -> List[Path]:
    import matplotlib.pyplot as plt
    lv, logdet, lv_init = arr["lv"], arr["logdet"], arr["lv_init"]
    disp, geo = m["dispersion"], m["table_geometry"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    ax.hist(lv.ravel(), bins=80, color=OKABE_ITO[0])
    lo, hi = np.quantile(lv, [0.001, 0.999])
    if lo - 0.25 * (hi - lo) <= lv_init <= hi + 0.25 * (hi - lo):
        ax.axvline(lv_init, color=OKABE_ITO[3], ls="--", lw=1.0, label=r"init $\log\sigma^2_0$")
        ax.legend(loc="upper left")
    else:                                   # keep the histogram readable; state the init value
        ax.text(0.02, 0.95, f"init log σ²₀ = {lv_init:.3f}\n(outside the shown range)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=OKABE_ITO[3])
    ax.set_xlabel(r"$\log\sigma^2_{vk}$ (all coordinates)")
    ax.set_ylabel("count")
    panel_label(ax, "a")
    ax = axes[0, 1]
    ax.hist(logdet, bins=60, color=OKABE_ITO[2])
    gate = "pass" if disp["gate_cv_gt_0.10"] else "fail"
    ax.text(0.98, 0.95, f"CV(tr Σ) = {disp['cv_trace']:.3f} (trained coords)\n"
            f"CV(tr Σ) = {disp['cv_trace_d0']:.3f} (D0 units)\ngate CV > 0.10 (trained): {gate}\n"
            f"at init: {m['init']['frac_at_init_candidates']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel(r"$\log\det\Sigma_v$")
    ax.set_ylabel("tokens")
    panel_label(ax, "b")
    ax = axes[1, 0]
    q25, q50, q75 = np.quantile(lv, [0.25, 0.5, 0.75], axis=0)
    k = np.arange(lv.shape[1])
    ax.fill_between(k, q25, q75, color=OKABE_ITO[5], alpha=0.5, lw=0, label="IQR over tokens")
    ax.plot(k, q50, color=OKABE_ITO[0], lw=1.0, label="median")
    for s in getattr(run, "block_slices", [])[1:]:
        ax.axvline(s.start - 0.5, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("coordinate k")
    ax.set_ylabel(r"$\log\sigma^2_{vk}$")
    ax.legend(loc="best")
    panel_label(ax, "c")
    ax = axes[1, 1]
    if geo.get("status") == "failed":
        ax.text(0.5, 0.5, "table geometry failed\n(see failed_steps)", ha="center", va="center",
                transform=ax.transAxes)
    else:
        spec = np.asarray(geo["pca_spectrum"], dtype=np.float64)[:20]
        ax.bar(np.arange(1, spec.size + 1), spec, color=OKABE_ITO[4])
        share = geo["global_scale_share"]
        text = f"erank = {geo['effective_rank']:.1f}\nCKA(μ, log σ²) = {geo['cka_mu_logvar']:.3f}"
        if share is not None:
            text += f"\nuniform-scale share = {share:.3f}"
        ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel("principal component of centered log σ² table")
    ax.set_ylabel("explained variance ratio")
    panel_label(ax, "d")
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, ctx, _SECTION, name)


def _plot_frequency_and_classes(ctx: AnalysisContext, m: Dict[str, Any], arr: Dict[str, Any],
                                title: str, name: str) -> List[Path]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    if arr["log_count"] is not None:
        hb = ax.hexbin(arr["log_count"], arr["logdet"], gridsize=40, mincnt=1, bins="log",
                       cmap="viridis", linewidths=0)
        fig.colorbar(hb, ax=ax, label="tokens (log)")
        b = arr["binned"]
        if b:
            xm = [r["x_median"] for r in b]
            ax.plot(xm, [r["y_median"] for r in b], color=OKABE_ITO[3], lw=1.2, label="bin median")
            ax.fill_between(xm, [r["y_q25"] for r in b], [r["y_q75"] for r in b],
                            color=OKABE_ITO[3], alpha=0.2, lw=0)
            ax.legend(loc="best")
        if "logdet_vs_log_count" in m["frequency"]:
            ax.set_title(f"ρ = {_fmt_ci(m['frequency']['logdet_vs_log_count'])}", fontsize=8)
        ax.set_xlabel("log(1 + training count)")
    else:
        ax.text(0.5, 0.5, "training counts unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel(r"$\log\det\Sigma_v$")
    panel_label(ax, "a")
    ax = axes[1]
    groups = arr["class_groups"]
    if groups:
        names = list(groups)
        ax.boxplot([groups[n] for n in names], orientation="horizontal",
                   tick_labels=[f"{n} ({groups[n].size})" for n in names], showfliers=False,
                   medianprops={"color": OKABE_ITO[3]})
        ax.set_xlabel(r"$\log\det\Sigma_v$")
        ax.set_title(m["classes"]["population"], fontsize=7)
    panel_label(ax, "b")
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, ctx, _SECTION, name)


def _plot_fields(ctx: AnalysisContext, m: Dict[str, Any], arr: Dict[str, Any], title: str,
                 name: str) -> List[Path]:
    import matplotlib.pyplot as plt
    groups = arr["field_groups"]
    order = sorted(groups, key=lambda n: float(np.median(groups[n])))
    fig, ax = plt.subplots(figsize=(5.0, 0.22 * len(order) + 1.2))
    ax.boxplot([groups[n] for n in order], orientation="horizontal",
               tick_labels=[f"{n} ({groups[n].size})" for n in order], showfliers=False,
               medianprops={"color": OKABE_ITO[3]})
    jitter = np.random.default_rng(ctx.seed)
    for i, n in enumerate(order, start=1):
        ax.scatter(groups[n], i + jitter.uniform(-0.15, 0.15, groups[n].size), s=4,
                   color=OKABE_ITO[0], alpha=0.6, lw=0)
    kw, kr = m["fields"]["kruskal"], m["fields"]["kruskal_freq_residual"]

    def fmt(t: Dict[str, Any]) -> str:
        return "–" if t.get("H") is None else f"H = {t['H']:.1f}, p = {t['p']:.2g}"
    ax.set_title(f"Kruskal–Wallis: {fmt(kw)}; frequency-residualized: {fmt(kr)}", fontsize=7)
    ax.set_xlabel(r"$\log\det\Sigma_v$")
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, ctx, _SECTION, name)


def _plot_entailment(ctx: AnalysisContext, ent: Dict[str, Any], title: str,
                     name: str) -> List[Path]:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    bars = [("KL asymmetry", ent["kl_asymmetry"]), ("log det baseline", ent["logdet_baseline"])]
    if "acc" in ent["frequency_baseline"]:
        bars.insert(1, ("frequency baseline", ent["frequency_baseline"]))
    x = np.arange(len(bars))
    acc = np.array([b[1]["acc"] for b in bars])
    # a percentile bootstrap interval need not contain the point estimate; clip the bars
    err = np.array([[0.0 if b[1]["ci_lo"] is None else max(b[1]["acc"] - b[1]["ci_lo"], 0.0) for b in bars],
                    [0.0 if b[1]["ci_hi"] is None else max(b[1]["ci_hi"] - b[1]["acc"], 0.0) for b in bars]])
    ax.bar(x, acc, yerr=err, color=[OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2]][:len(bars)], capsize=3)
    ax.axhline(0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xticks(x, [b[0] for b in bars], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(f"n = {ent['n']} pairs in {ent['n_clusters']} hypernym groups;\n"
                 "cluster-bootstrap 95% CI", fontsize=8)
    panel_label(ax, "a")
    ax = axes[1]
    pairs = ent["pairs"]
    d_logdet = np.array([p["logdet_hyper"] - p["logdet_hypo"] for p in pairs])
    d_kl = np.array([p["kl_hyper_hypo"] - p["kl_hypo_hyper"] for p in pairs])
    ok = np.array([p["kl_correct"] for p in pairs])
    ax.scatter(d_logdet[ok], d_kl[ok], s=10, color=OKABE_ITO[2], label="KL correct")
    ax.scatter(d_logdet[~ok], d_kl[~ok], s=10, color=OKABE_ITO[3], label="KL wrong")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\log\det\Sigma_{hyper} - \log\det\Sigma_{hypo}$")
    ax.set_ylabel(r"KL(hyper‖hypo) − KL(hypo‖hyper)")
    ax.legend(loc="best")
    panel_label(ax, "b")
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, ctx, _SECTION, name)


def _plot_similarity(ctx: AnalysisContext, sim: Dict[str, Any], title: str,
                     name: str) -> List[Path]:
    import matplotlib.pyplot as plt
    benches = _ok_benches(sim)
    fig, axes = plt.subplots(1, len(benches), figsize=(3.6 * len(benches), 3.0), squeeze=False)
    for j, bench in enumerate(benches):
        ax = axes[0, j]
        per = sim[bench]["scorers"]
        keys = [k for k, _ in _SCORERS]
        y = np.arange(len(keys))[::-1]
        for yi, key in zip(y, keys):
            e = per[key]
            rho = e.get("rho")
            if rho is None or not np.isfinite(rho):
                continue
            d = e.get("delta_vs_cos_mu", {})
            sig = (d.get("ci_lo") is not None and (d["ci_lo"] > 0 or d["ci_hi"] < 0))
            color = OKABE_ITO[7] if key in _REFERENCE_SCORERS else OKABE_ITO[0]
            # a percentile bootstrap interval need not contain the point estimate; clip the bars
            ax.errorbar(rho, yi, xerr=[[max(rho - e["ci_lo"], 0.0)], [max(e["ci_hi"] - rho, 0.0)]], fmt="o",
                        color=color, mfc=color if (sig or key == "cos_mu") else "white", ms=4,
                        capsize=2)
        if per["cos_mu"].get("rho") is not None and np.isfinite(per["cos_mu"]["rho"]):
            ax.axvline(per["cos_mu"]["rho"], color="0.5", ls="--", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_yticks(y, [label for _, label in _SCORERS])
        for tick, key in zip(ax.get_yticklabels(), keys):
            if key == _HEADLINE_SCORER:
                tick.set_fontweight("bold")
        ax.set_xlabel("Spearman ρ with human ratings")
        ax.set_title(f"{bench} (n = {sim[bench]['n']})", fontsize=8)
        panel_label(ax, "abcdefgh"[j])
    fig.suptitle(title + "\n(filled: paired Δρ vs cos(μ) CI excludes 0; bold: pre-registered "
                 "headline scorer)", fontsize=8)
    fig.tight_layout()
    return save_figure(fig, ctx, _SECTION, name)


def _neighbors_table(rows: List[Dict[str, Any]]) -> str:
    out = []
    for r in rows:
        out.append({
            "probe": r["probe"], "log det z": r["probe_logdet_z"],
            f"top-{len(r['neighbors'])} cos neighbors, narrow → broad (log det z)":
                ", ".join(f"{n['word']} ({n['logdet_z']:+.1f})" for n in r["neighbors"]),
            "nearest by KL(probe‖x)": ", ".join(n["word"] for n in r["kl_neighbors"]),
        })
    return markdown_table(out, digits=2)


def _entailment_table(ent: Dict[str, Any]) -> str:
    rows = [{"hyponym": p["hyponym"], "hypernym": p["hypernym"],
             "KL(hypo‖hyper)": p["kl_hypo_hyper"], "KL(hyper‖hypo)": p["kl_hyper_hypo"],
             "log det hypo": p["logdet_hypo"], "log det hyper": p["logdet_hyper"],
             "count hypo": p["count_hypo"], "count hyper": p["count_hyper"],
             "KL correct": "yes" if p["kl_correct"] else "no",
             "freq. correct": ("–" if p["frequency_correct"] is None
                               else "yes" if p["frequency_correct"] else "no")}
            for p in ent["pairs"]]
    keys = ("acc", "ci_lo", "ci_hi", "n", "n_clusters")
    summary = [{"method": "KL asymmetry", **{k: ent["kl_asymmetry"][k] for k in keys}},
               {"method": "log det baseline", **{k: ent["logdet_baseline"][k] for k in keys}}]
    if "acc" in ent["frequency_baseline"]:
        summary.insert(1, {"method": "frequency baseline", **{k: ent["frequency_baseline"][k] for k in keys}})
    text = markdown_table(summary)
    diff = ent.get("kl_minus_frequency")
    if diff is not None:
        text += "\n" + markdown_table([{"comparison": "KL asymmetry − frequency baseline",
                                        **{k: diff[k] for k in ("delta", "ci_lo", "ci_hi", "p", "n_clusters")}}])
    text += (f"\nIntervals and p-values: cluster bootstrap over the {ent['n_clusters']} hypernym groups "
             f"(B = {_N_BOOT_CLUSTER}), since pairs sharing a hypernym are not independent.\n")
    return text + "\n" + markdown_table(rows)


def _similarity_table(sim: Dict[str, Any]) -> str:
    rows = []
    ok = _ok_benches(sim)
    for bench, v in sim.items():
        if bench not in ok:
            rows.append({"benchmark": bench, "scorer": "–", "n": v.get("n"), "role": v.get("reason")})
            continue
        best = v.get("best_gaussian")
        for key, label in _SCORERS:
            e = v["scorers"][key]
            d = e.get("delta_vs_cos_mu", {})
            de = e.get("delta_vs_euclid_mu", {})
            role = ("headline (pre-registered)" if key == _HEADLINE_SCORER
                    else "Σ-free reference" if key in _REFERENCE_SCORERS else "")
            if key == best:
                role = (role + "; " if role else "") + f"best ({v['best_gaussian_note']})"
            rows.append({"benchmark": bench, "scorer": label, "role": role, "n": e.get("n"),
                         "rho": e.get("rho"), "ci_lo": e.get("ci_lo"), "ci_hi": e.get("ci_hi"),
                         "Δρ vs cos(μ)": d.get("delta"), "Δ ci_lo": d.get("ci_lo"),
                         "Δ ci_hi": d.get("ci_hi"), "Δρ vs −‖Δμ‖": de.get("delta"),
                         "Δ' ci_lo": de.get("ci_lo"), "Δ' ci_hi": de.get("ci_hi")})
    return markdown_table(rows, columns=["benchmark", "scorer", "role", "n", "rho", "ci_lo", "ci_hi",
                                         "Δρ vs cos(μ)", "Δ ci_lo", "Δ ci_hi", "Δρ vs −‖Δμ‖",
                                         "Δ' ci_lo", "Δ' ci_hi"])


# ----------------------------------------------------------------------------------------------
# Routing covariance (frames)
# ----------------------------------------------------------------------------------------------

def _routing_covariance(run: Any, ctx: AnalysisContext, result: SectionResult,
                        failed: List[Dict[str, str]]) -> Dict[str, Any]:
    r"""Frame-induced routing covariance :math:`C_v^b = e^{-X_v^b}\,\mathrm{diag}(v_v^b)\,e^{-X_v^{b\top}}`.

    Delegates the matrix work to :func:`vfe3.semantics.frames.routing_covariance_summary` and plots
    the exact decomposition :math:`\log\det C_v^b = \sum_{k\in b} lv_{vk} - 2\,\mathrm{tr}\,X_v^b`
    into its Sigma part and frame part, their Spearman correlations with log count, and the relative
    gap :math:`\|C_v - \bar c_v (U_v^\top U_v)^{-1}\|_F / \|C_v\|_F`.
    """
    if not run.frames_active:
        note = ("frames inactive (identity token frames, or the unlearned random frames of "
                "gauge_transport='frozen'; the run notes give the reason): no frame decomposition of the "
                "routing covariance")
        result.notes.append(note)
        return {"status": "skipped", "reason": note}
    ids = np.asarray(ctx.candidate_ids, dtype=np.int64)

    def summary() -> Dict[str, Any]:
        from vfe3.semantics import frames
        return frames.routing_covariance_summary(run, ctx, ids)

    s = _attempt(failed, result.notes, "routing covariance", summary, None)
    if s is None:                                       # recorded; the Sigma battery above survives
        return {"status": "failed", "error": failed[-1]["error"]}
    ctx.cache["routing_covariance"] = s
    result.figures.extend(_attempt(failed, result.notes, "routing covariance figure",
                                   lambda: _plot_routing_covariance(ctx, s), []))
    if s.get("note"):
        result.notes.append(f"routing covariance: {s['note']}")
    kept = ("identity_max_abs_error", "variance_shares", "spearman_log_count", "summary", "note")
    return {"status": "ok", "n": int(np.asarray(s["ids"]).size), **{k: s[k] for k in kept}}


def _plot_routing_covariance(ctx: AnalysisContext, s: Dict[str, Any]) -> List[Path]:
    r"""(a) Sigma part vs frame part of log det C; (b) their Spearman with log count; (c) relative
    gap to :math:`\bar c (U^\top U)^{-1}`; (d) eigen-spread :math:`\log(\lambda_{max}/\lambda_{min})`
    of :math:`C_v^b` per block."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax = axes[0, 0]
    hb = ax.hexbin(s["sigma_part"], s["frame_part"], gridsize=40, mincnt=1, bins="log",
                   cmap="viridis", linewidths=0)
    fig.colorbar(hb, ax=ax, label="tokens (log)")
    sh = s["variance_shares"]
    if sh.get("sigma") is not None:
        ax.set_title(f"Var(log det C) shares: Σ {sh['sigma']:.2f}, frame {sh['frame']:.2f}, "
                     f"2Cov {sh['cross']:.2f}", fontsize=7)
    ax.set_xlabel(r"$\Sigma$ part $\sum_k \log\sigma^2_{vk}$")
    ax.set_ylabel(r"frame part $-2\,\mathrm{tr}\,X_v$")
    panel_label(ax, "a")
    ax = axes[0, 1]
    sp = s["spearman_log_count"]
    if sp is not None:
        keys = [("log_det_C", "log det C"), ("sigma_part", "Σ part"), ("frame_part", "frame part")]
        rho = [np.nan if sp[k]["rho"] is None else sp[k]["rho"] for k, _ in keys]
        ax.bar(np.arange(3), rho, color=[OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2]])
        ax.set_xticks(np.arange(3), [lab for _, lab in keys])
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel("Spearman ρ with log count")
    else:
        ax.text(0.5, 0.5, "training counts unavailable", ha="center", va="center",
                transform=ax.transAxes)
    panel_label(ax, "b")
    ax = axes[1, 0]
    gap = np.asarray(s["rel_gap"], dtype=np.float64)
    gap = gap[np.isfinite(gap)]
    ax.hist(gap, bins=60, color=OKABE_ITO[4])
    if gap.size:
        ax.axvline(float(np.median(gap)), color=OKABE_ITO[3], ls="--", lw=1.0,
                   label=f"median {np.median(gap):.3f}")
        ax.legend(loc="best")
    ax.set_xlabel(r"$\|C_v - \bar c_v (U_v^\top U_v)^{-1}\|_F / \|C_v\|_F$")
    ax.set_ylabel("tokens")
    panel_label(ax, "c")
    ax = axes[1, 1]
    cond = np.asarray(s["log_cond_block"], dtype=np.float64)          # (n, B)
    for b in range(cond.shape[1]):
        vals = cond[:, b][np.isfinite(cond[:, b])]
        ax.hist(vals, bins=50, histtype="step", color=OKABE_ITO[b % len(OKABE_ITO)], label=f"block {b}")
    ax.set_xlabel(r"$\log(\lambda_{\max}/\lambda_{\min})$ of $C_v^b$")
    ax.set_ylabel("tokens")
    if cond.shape[1] <= 8:
        ax.legend(loc="best")
    panel_label(ax, "d")
    fig.suptitle(r"Frame-induced routing covariance $C_v = U_v^{-1}\Sigma_v U_v^{-\top}$ (per head block)")
    fig.tight_layout()
    return save_figure(fig, ctx, _SECTION, "routing_covariance")


# ----------------------------------------------------------------------------------------------
# Section
# ----------------------------------------------------------------------------------------------

def _bank_outputs(run: Any, ctx: AnalysisContext, result: SectionResult, m: Dict[str, Any],
                  arr: Dict[str, Any], bank_label: str, prefix: str,
                  failed: List[Dict[str, str]]) -> None:
    r"""Figures and tables of one bank; skips are recorded in ``result.notes``, and a figure or
    table that raises is recorded in ``failed`` without stopping the others."""

    def emit_figure(stem: str, fn: Callable[[], List[Path]]) -> None:
        result.figures.extend(_attempt(failed, result.notes, f"{bank_label}: {stem} figure", fn, []))

    def emit_table(stem: str, fn: Callable[[], str]) -> None:
        def write() -> None:
            text = fn()
            write_table(ctx, _SECTION, f"{prefix}{stem}", text)
            result.tables[f"{prefix}{stem}"] = text
        _attempt(failed, result.notes, f"{bank_label}: {stem} table", write, None)

    emit_figure("overview", lambda: _plot_overview(run, ctx, m, arr, f"{bank_label}: covariance table",
                                                   f"{prefix}overview"))
    emit_figure("frequency_and_classes", lambda: _plot_frequency_and_classes(
        ctx, m, arr, f"{bank_label}: log det Σ vs frequency and token class",
        f"{prefix}frequency_and_classes"))
    if arr["field_groups"]:
        emit_figure("fields", lambda: _plot_fields(ctx, m, arr, f"{bank_label}: log det Σ by semantic field",
                                                   f"{prefix}fields"))
    elif m["fields"].get("status") != "failed":
        result.notes.append(f"{bank_label}: no semantic-field words among the candidates; "
                            "fields figure skipped")
    if m["neighbors"]:
        emit_table("variance_sorted_neighbors", lambda: _neighbors_table(m["neighbors"]))
    elif not any(f["step"] == f"{bank_label}: variance-sorted neighbors" for f in failed):
        result.notes.append(f"{bank_label}: no probe words among the candidates; neighbor table skipped")
    ent = m["entailment"]
    if ent.get("status") == "ok":
        emit_figure("entailment", lambda: _plot_entailment(
            ctx, ent, f"{bank_label}: hypernymy from KL asymmetry", f"{prefix}entailment"))
        emit_table("entailment", lambda: _entailment_table(ent))
    elif ent.get("status") != "failed":
        result.notes.append(f"{bank_label}: entailment skipped ({ent.get('reason')})")
    sim = m["similarity"]
    if _ok_benches(sim):
        emit_figure("gaussian_similarity", lambda: _plot_similarity(
            ctx, sim, f"{bank_label}: Gaussian similarity scorers", f"{prefix}gaussian_similarity"))
        emit_table("gaussian_similarity", lambda: _similarity_table(sim))
    elif sim.get("status") != "failed":
        result.notes.append(f"{bank_label}: no similarity benchmark with >= {_MIN_PAIRS} resolved pairs")


def _headline(m: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    r"""Headline numbers of one bank.

    The similarity headline quotes the pre-registered scorer :math:`-J` (Jeffreys) against cos(mu)
    and the Sigma-free control :math:`-\|\Delta\mu\|`, with the paired :math:`\Delta\rho` of
    :math:`-J` against that control; the best of the Sigma-using scorers is listed separately as a
    maximum not adjusted for selection.
    """
    disp = m["dispersion"]
    h: Dict[str, Any] = {f"{prefix}CV(tr Σ): trained coordinates / D0 units":
                         f"{_fmt(disp['cv_trace'])} / {_fmt(disp['cv_trace_d0'])}"}
    freq = m["frequency"]
    if "logdet_vs_log_count" in freq:
        h[f"{prefix}ρ(log det Σ, log count)"] = freq["logdet_vs_log_count"]["rho"]
    ent = m["entailment"]
    if ent.get("status") == "ok":
        fb = ent["frequency_baseline"]
        h[f"{prefix}entailment acc: KL / frequency baseline"] = (
            f"{ent['kl_asymmetry']['acc']:.3f} / " + (f"{fb['acc']:.3f}" if "acc" in fb else "–")
            + f" (n = {ent['n']} pairs, {ent['n_clusters']} hypernyms)")
    sim = m["similarity"]
    ok = _ok_benches(sim)
    bench = "simlex999" if "simlex999" in ok else (ok[0] if ok else None)
    if bench is not None:
        per = sim[bench]["scorers"]
        h[f"{prefix}{bench}: −Jeffreys ρ / cos(μ) ρ / −‖Δμ‖ ρ"] = (
            f"{_fmt(per[_HEADLINE_SCORER]['rho'])} / {_fmt(per['cos_mu']['rho'])} / "
            f"{_fmt(per['neg_euclid_mu']['rho'])} (n = {sim[bench]['n']})")
        h[f"{prefix}{bench}: Δρ(−Jeffreys − (−‖Δμ‖)), paired bootstrap"] = _fmt_ci(
            per[_HEADLINE_SCORER].get("delta_vs_euclid_mu", {}), "delta")
        best = sim[bench]["best_gaussian"]
        if best is not None:
            h[f"{prefix}{bench}: best Gaussian ρ ({sim[bench]['best_gaussian_note']})"] = (
                f"{per[best]['label']} {_fmt(per[best]['rho'])}")
    return h


def run_sigma_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    r"""Covariance-table semantics of the prior bank (and of a Gaussian decode bank).

    Items: (1) dispersion (CV of :math:`\mathrm{tr}\,\Sigma` against the pre-registered gate 0.10,
    log-det statistics, the per-coordinate log-variance histogram with the initial value, the
    fraction of rows still at initialization); (2) frequency (Spearman of :math:`\log\det\Sigma`
    with log count, binned medians, token classes, semantic fields); (3) variance-sorted
    neighbors; (4) hypernymy from KL asymmetry with frequency and log-det baselines (cluster
    bootstrap over hypernym groups); (5) Gaussian similarity scorers on the similarity benchmarks
    with paired :math:`\Delta\rho` vs cos(mu) and vs :math:`-\|\Delta\mu\|`, headlined by the
    pre-registered :math:`-J`; (6) table geometry (effective rank, PCA spectrum,
    CKA(mu, log sigma^2)); (7) the frame-induced routing covariance when frames are active.
    Sub-steps that raise are recorded in ``metrics['failed_steps']`` and the rest continues.
    """
    apply_style()
    result = SectionResult(name=_SECTION)
    ids = np.asarray(ctx.candidate_ids, dtype=np.int64)
    if ids.size < 3:
        result.status = "skipped"
        result.notes.append(f"only {ids.size} candidate tokens; the Sigma section needs at least 3")
        return result
    rng = np.random.default_rng(ctx.seed)
    cfg = run.cfg
    family = str(cfg.family)
    lv_init = _init_log_var(cfg)
    redundancy, redundancy_reason = diagonal_redundancy_status(cfg)
    failed: List[Dict[str, str]] = []
    metrics: Dict[str, Any] = {"family": family, "sigma_init": float(cfg.sigma_init),
                               "log_var_init": lv_init, "sigma_frozen": run.sigma_frozen,
                               "decode_kind": run.decode_kind,
                               "diagonal_redundancy": {"status": redundancy, "reason": redundancy_reason},
                               "banks": {}, "failed_steps": failed}

    if run.sigma_frozen is True:
        result.notes.append("Σ is frozen at encode on this route (every layer's Σ is the token prior "
                            "Σ_v): contextual Σ analyses are vacuous; everything here is a property "
                            "of the static table")
    elif run.sigma_frozen is None:
        result.notes.append("whether Σ is frozen at encode could not be determined for this config")
    if not family.startswith("gaussian"):
        result.notes.append(f"family {family}: the Gaussian divergences are evaluated on the "
                            "moment-matched diagonal Gaussians (same mean and covariance diagonal)")
    blocks = bool(getattr(run, "prior_covariance_blocks", False))
    metrics["prior_covariance"] = {
        "table": "block_cholesky" if blocks else "diagonal",
        "log_det": "exact, from the per-head blocks" if blocks else "exact (diagonal table)",
        "divergences": ("diagonal marginals N(mu_v, diag C_v): the within-head correlations are dropped"
                        if blocks else "exact (diagonal table)")}
    if blocks:                                   # audit 2026-10-10 #20
        result.notes.append("prior_covariance_table='block_cholesky': log det Σ (dispersion, frequency, token "
                            "classes, semantic fields, neighbor ordering and the log-det entailment baseline) is "
                            "computed exactly from the per-head blocks C_v^(h). The divergences (KL, Jeffreys, "
                            "Fisher–Rao, W2 and the distribution cosine of the similarity scorers, the KL-asymmetry "
                            "entailment test and the KL neighbors) use the diagonal marginals N(μ_v, diag C_v), "
                            "which drop the within-head correlations, so they are not the divergences of the "
                            "stored Gaussians; tr Σ needs only the diagonal and is exact")
    invariance = ("KL, Jeffreys, Fisher–Rao and the distribution cosine are invariant under a common "
                  "positive-diagonal rescaling μ → Dμ, log σ² → log σ² + 2 log D of the table; W2 and "
                  "cos(μ) are evaluated in D0 units (log d0_k = −½ mean over candidates of log σ²_k), "
                  "where they are invariant too")
    if redundancy == "exact":
        invariance += (f". The rescaling is an exact redundancy of this run ({redundancy_reason}), so "
                       "these scorers are gauge invariant and D0 is a gauge fixing")
    elif redundancy == "broken":
        invariance += (f". The rescaling is not a symmetry of this run ({redundancy_reason}), so D0 is "
                       "a unit convention, not a gauge fixing, and the raw coordinates are the ones "
                       "the model reads")
    else:
        invariance += (f". That the rescaling is a redundancy of this run is assumed, not verified "
                       f"({redundancy_reason}); D0 is a gauge fixing only under that assumption")
    result.notes.append(invariance)
    if run.decode_kind == "linear":
        result.notes.append("decode is linear (logits = μ Wᵀ + b + κ log π): Σ never reaches the "
                            "logits and enters only through the attention routing")
    elif run.decode_kind != "gaussian":
        result.notes.append(f"decode kind {run.decode_kind!r}: no decode-bank covariance analysis")

    m, arr = _bank_battery(run, ctx, run.mu, run.log_var, lv_init, rng, bank="Prior bank",
                           failed=failed, notes=result.notes, logdet_fn=run.log_det if blocks else None)
    metrics["banks"]["prior"] = m
    disp = m["dispersion"]
    result.notes.append(f"CV(tr Σ) = {disp['cv_trace']:.3f} is computed in the trained coordinates "
                        "(tr Σ is not invariant under the global positive-diagonal rescaling) and the "
                        f"pre-registered gate is applied to it; in D0 units CV(tr Σ) = "
                        f"{disp['cv_trace_d0']:.3f} (metrics key cv_trace_d0)")
    _bank_outputs(run, ctx, result, m, arr, "Prior bank", "", failed)
    result.headline.update(_attempt(failed, result.notes, "Prior bank: headline",
                                    lambda: _headline(m, ""), {}))

    if run.decode_kind == "gaussian":
        def decode_bank() -> Dict[str, Any]:
            tied = (np.array_equal(run.decode_mu(ids), run.mu(ids))
                    and np.array_equal(run.decode_log_var(ids), run.log_var(ids)))
            if tied:
                result.notes.append("the Gaussian decode bank equals the prior bank on the candidates "
                                    "(tied); its battery would repeat the prior bank's and is skipped")
                return {"status": "skipped", "reason": "tied to the prior bank"}
            md, arrd = _bank_battery(run, ctx, run.decode_mu, run.decode_log_var, lv_init, rng,
                                     bank="Decode bank", failed=failed, notes=result.notes)
            md["tau_eff"] = float(run.tau_eff())
            _bank_outputs(run, ctx, result, md, arrd, "Decode bank", "decode_", failed)
            result.headline.update(_attempt(failed, result.notes, "Decode bank: headline",
                                            lambda: _headline(md, "decode "), {}))
            return md

        metrics["banks"]["decode"] = _attempt(failed, result.notes, "Decode bank", decode_bank,
                                              dict(_FAILED_STEP))

    metrics["routing_covariance"] = _routing_covariance(run, ctx, result, failed)
    result.metrics = metrics
    return result
