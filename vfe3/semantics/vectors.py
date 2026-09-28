r"""The standard word-embedding battery on one :class:`~vfe3.semantics.common.VectorView`.

A view is a token-indexed table of vectors (prior means, decode rows, frame log-coordinates, ...).
This module runs, on any view, the protocols that word-embedding and transformer-interpretability
papers use for static token tables:

* nearest neighbors;
* word analogies ``a : a* :: b : b*`` with 3CosAdd (Mikolov, Yih & Zweig 2013), 3CosMul and
  PairDirection (Levy & Goldberg 2014) and the ONLY-B / IGNORE-A baselines plus the "honest"
  no-exclusion protocol (Linzen 2016), with 95% CIs that resample word pairs (the questions of a
  category share their pairs) and full-benchmark coverage denominators;
* relation-offset consistency against a re-pairing null (Fournier, Dupoux & Dunbar 2020),
  leave-one-out projections against a random-pair baseline (Park, Choe & Veitch 2024, Fig. 2)
  and the top-PC share of offsets (Bolukbasi et al. 2016);
* concept directions under the Euclidean and the causal (whitened) inner product
  (Park, Choe & Veitch 2024, Thm 3.4);
* similarity benchmarks (Spearman with bootstrap CI);
* semantic-field clustering (silhouette in the original space, kNN purity with a permutation null)
  and an exploratory k-means of the frequent vocabulary;
* semantic projection onto antonym axes (Grand et al. 2022);
* global geometry (anisotropy, IsoScore, effective rank, TwoNN, hubness, rogue dimensions, frequency
  probe) from :mod:`vfe3.semantics.stats`;

and writes the per-view figures/tables and the cross-view comparison figures.

Preprocessing (:func:`prepare`). A ``"cosine"`` view is optionally centered (row mean over the
view removed) and then L2-normalized, so inner products of prepared rows are cosines. A
``"euclidean"`` view is used raw: its Euclidean distance is the meaningful quantity (e.g. the
diagonal-Gaussian W2 chart). Prepared rows are float64.

Output locations: per-view figures under ``figures/embeddings/<view.name>/`` and tables under
``tables/embeddings/<view.name>/``; cross-view figures under ``figures/embeddings/compare/``.
Figure-level skip reasons that have no battery to attach to are appended to
``ctx.cache["vectors_notes"]``.
"""

from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vfe3.semantics import stats
from vfe3.semantics.common import (
    OKABE_ITO,
    AnalogyQuestion,
    AnalysisContext,
    VectorView,
    apply_style,
    markdown_table,
    panel_label,
    save_figure,
    write_table,
)


ANALOGY_METHODS: Tuple[str, ...] = ("3cosadd", "3cosmul", "pairdir", "only_b", "ignore_a")
COSINE_ONLY_METHODS: Tuple[str, ...] = ("3cosmul", "pairdir")
METHOD_LABELS: Dict[str, str] = {
    "3cosadd": "3CosAdd", "3cosmul": "3CosMul", "pairdir": "PairDirection",
    "only_b": "ONLY-B", "ignore_a": "IGNORE-A",
}
_COSMUL_EPS = 1e-3                       # Levy & Goldberg (2014)
_PAIRDIR_EPS = 1e-12                     # floor under ||x - b||^2 for unit rows
_QUESTION_CHUNK = 256                    # questions scored per (chunk, n) block
_BOOT_CHUNK = 250                        # bootstrap draws per (chunk, Q_category) weight block
_PROBE_MAX_FEATURES = 512                # frequency probe runs on at most this many principal components
_MARKERS = ("o", "s", "^", "D", "v", "P")
_SECTION = "embeddings"
_COMPARE_SECTION = "embeddings/compare"

# Stable per-step stream ids, so each stochastic step draws the same numbers whichever others ran.
_STEP_STREAM = {"analogy_benchmark": 1, "offsets": 2, "similarity": 3, "fields": 4,
                "clusters": 5, "geometry": 6}


# ----------------------------------------------------------------------------------------------
# Preprocessing and small helpers
# ----------------------------------------------------------------------------------------------

def _prepare_array(X: np.ndarray, metric: str, center: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if metric == "euclidean":
        return X.copy()
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return stats.unit_rows(X)


def prepare(view: VectorView, center: bool) -> np.ndarray:
    r"""Rows of ``view`` ready for its metric, float64 ``(n, D)``.

    ``"cosine"``: ``x_r <- (x_r - center*mean) / ||x_r - center*mean||`` so that the inner product
    of prepared rows is the cosine similarity. ``"euclidean"``: the raw rows.
    """
    return _prepare_array(view.X, view.metric, center)


def _rng(ctx: AnalysisContext, step: str) -> np.random.Generator:
    return np.random.default_rng([int(ctx.seed), _STEP_STREAM[step]])


def _note(ctx: AnalysisContext, battery: Optional[Dict[str, Any]], text: str) -> None:
    if battery is not None:
        battery.setdefault("notes", []).append(text)
    else:
        ctx.cache.setdefault("vectors_notes", []).append(text)


def _label(ctx: AnalysisContext, token_id: int) -> str:
    r"""Display label safe for Matplotlib text (``$`` would start mathtext)."""
    return ctx.display(int(token_id)).replace("$", r"\$")


def _ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") not in ("failed", "skipped")


def _row_pairs(view: VectorView, pairs: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    r"""Row indices of the ``(a, a_star)`` pairs whose two tokens are in the view (duplicates dropped)."""
    seen = set()
    ra, rb, kept = [], [], []
    for a, b in pairs:
        a, b = int(a), int(b)
        if a == b or (a, b) in seen:
            continue
        rows = view.rows([a, b])
        if (rows >= 0).all():
            seen.add((a, b))
            ra.append(rows[0])
            rb.append(rows[1])
            kept.append((a, b))
    return np.asarray(ra, dtype=np.int64), np.asarray(rb, dtype=np.int64), kept


def _lexicon_rows(view: VectorView, lexicon: Mapping[str, Sequence[int]]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    r"""Rows and field labels of lexicon words present in the view (first field wins for a repeat)."""
    rows, labels, owner = [], [], {}
    repeated = 0
    for field_name, token_ids in lexicon.items():
        for t in token_ids:
            t = int(t)
            if t in owner:
                repeated += int(owner[t] != field_name)
                continue
            owner[t] = field_name
            r = view.rows([t])[0]
            if r >= 0:
                rows.append(r)
                labels.append(field_name)
    return np.asarray(rows, dtype=np.int64), labels, {"repeated_tokens": repeated}


# ----------------------------------------------------------------------------------------------
# Nearest neighbors
# ----------------------------------------------------------------------------------------------

def _topk_for_rows(Xp: np.ndarray, query_rows: np.ndarray, k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    r"""Top-``k`` rows for each query row (self excluded): cosine descending or distance ascending."""
    G = Xp[query_rows] @ Xp.T                                           # (q, n)
    if metric == "cosine":
        key = -G
    else:
        sq = (Xp * Xp).sum(axis=1)
        key = sq[query_rows, None] - 2.0 * G + sq[None, :]
    key[np.arange(query_rows.size), query_rows] = np.inf
    k = min(k, Xp.shape[0] - 1)
    part = np.argpartition(key, k - 1, axis=1)[:, :k]
    if metric == "cosine":
        vals = -np.take_along_axis(key, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
    else:
        vals = np.linalg.norm(Xp[part] - Xp[query_rows][:, None, :], axis=2)
        order = np.argsort(vals, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(vals, order, axis=1)


def nearest_neighbors(view: VectorView, query_ids: Sequence[int], k: int = 10,
                      center: bool = True) -> Dict[int, List[Tuple[int, float]]]:
    r"""``query id -> [(neighbor id, score), ...]`` for queries present in the view.

    Score is the cosine of prepared rows (descending) for a cosine view and the Euclidean distance
    (ascending) for a Euclidean view. The query itself is excluded.
    """
    rows = view.rows(query_ids)
    ordered = list(dict.fromkeys(int(r) for r in rows if r >= 0))
    if not ordered or view.n < 2:
        return {}
    query_rows = np.asarray(ordered, dtype=np.int64)
    idx, vals = _topk_for_rows(prepare(view, center), query_rows, k, view.metric)
    return {int(view.ids[q]): [(int(view.ids[j]), float(v)) for j, v in zip(idx[i], vals[i])]
            for i, q in enumerate(query_rows)}


# ----------------------------------------------------------------------------------------------
# Analogies
# ----------------------------------------------------------------------------------------------

def _covered_questions(view: VectorView, questions: Sequence[AnalogyQuestion]) -> Tuple[List[int], np.ndarray, int]:
    r"""Indices of questions whose four ids are in the view and non-degenerate, their rows (Q, 4)."""
    if not questions:
        return [], np.zeros((0, 4), dtype=np.int64), 0
    flat = [t for q in questions for t in (q.a, q.a_star, q.b, q.b_star)]
    R = view.rows(flat).reshape(-1, 4)
    degenerate = np.array([q.b_star in (q.a, q.a_star, q.b) for q in questions])
    keep = (R >= 0).all(axis=1) & ~degenerate
    return [int(i) for i in np.flatnonzero(keep)], R[keep], int((degenerate & (R >= 0).all(axis=1)).sum())


def _method_scores(method: str, metric: str, Sa: np.ndarray, Ss: np.ndarray, Sb: np.ndarray,
                   g: Dict[str, np.ndarray], sq: Optional[np.ndarray]) -> np.ndarray:
    r"""Candidate scores (B, n) for one method (row = question, column = candidate); higher is better.

    Cosine view (unit rows, ``S_* = <x, *>``):
    3CosAdd ``cos(x,a*) - cos(x,a) + cos(x,b)``; 3CosMul ``c(x,a*) c(x,b)/(c(x,a) + 10^{-3})`` with
    ``c = (cos+1)/2``; PairDirection

    .. math:: \cos(x-b, a^*-a) = \frac{S_{a^*} - S_a - (b\cdot a^* - b\cdot a)}
              {\sqrt{\max(2-2S_b,\epsilon)}\,\sqrt{\max(2-2\,a\cdot a^*,\epsilon)}};

    ONLY-B ``cos(x,b)``; IGNORE-A ``cos(x,a*) + cos(x,b)``. Euclidean view: 3CosAdd is
    ``-||x - (a* - a + b)||^2`` up to a per-question constant, ONLY-B ``-||x-b||^2`` and IGNORE-A
    ``-(||x-a*||^2 + ||x-b||^2)``, all from inner products.
    """
    if metric == "cosine":
        if method == "3cosadd":
            return Ss - Sa + Sb
        if method == "3cosmul":
            return (0.5 * (Ss + 1.0)) * (0.5 * (Sb + 1.0)) / (0.5 * (Sa + 1.0) + _COSMUL_EPS)
        if method == "pairdir":
            num = Ss - Sa - (g["b_as"] - g["b_a"])[:, None]
            den = (np.sqrt(np.maximum(2.0 - 2.0 * Sb, _PAIRDIR_EPS))
                   * np.sqrt(np.maximum(2.0 - 2.0 * g["a_as"], _PAIRDIR_EPS))[:, None])
            return num / den
        if method == "only_b":
            return Sb.copy()
        if method == "ignore_a":
            return Ss + Sb
    else:
        norms = sq[None, :]
        if method == "3cosadd":
            return 2.0 * (Ss - Sa + Sb) - norms
        if method == "only_b":
            return 2.0 * Sb - norms
        if method == "ignore_a":
            return 2.0 * (Ss + Sb) - 2.0 * norms
    raise ValueError(f"method {method!r} is not defined for metric {metric!r}")


def analogy_scores(view: VectorView, questions: Sequence[AnalogyQuestion], *, center: bool = True,
                   exclude_queries: bool = True, methods: Sequence[str] = ANALOGY_METHODS,
                   topk: int = 5) -> Dict[str, Any]:
    r"""Rank of ``b*`` (1 = correct) and the top-``k`` ids under each method, per covered question.

    All candidates are the view's rows. One matmul ``S = X_p[U] X_p^T`` (``U`` = unique query
    words) serves every method; each method is then a gather over ``S``. With
    ``exclude_queries`` (the standard protocol) ``a, a*, b`` are removed from the candidates. The
    "honest" protocol of Linzen (2016) -- 3CosAdd WITHOUT exclusion -- is always reported under
    ``"honest"``: accuracy and the fractions of questions whose top-1 is ``b``, ``a*`` or ``a``.
    Questions with a word outside the view, or with ``b*`` in ``{a, a*, b}``, are not scored.
    Ranks count candidates with a strictly higher score (ties resolved in favor of ``b*``).
    """
    unknown = [m for m in methods if m not in ANALOGY_METHODS]
    if unknown:
        raise ValueError(f"unknown analogy methods {unknown}")
    run = [m for m in methods if not (view.metric == "euclidean" and m in COSINE_ONLY_METHODS)]
    skipped = {m: "cosine-only method; view metric is euclidean" for m in methods if m not in run}
    covered, R, n_degenerate = _covered_questions(view, questions)
    out: Dict[str, Any] = {
        "n_total": len(questions), "n_covered": len(covered), "n_degenerate": n_degenerate,
        "covered_index": np.asarray(covered, dtype=np.int64),
        "questions": [questions[i] for i in covered], "exclude_queries": bool(exclude_queries),
        "center": bool(center), "metric": view.metric, "methods": {}, "skipped_methods": skipped,
        "honest": None,
    }
    Q = len(covered)
    if Q == 0:
        return out
    Xp = prepare(view, center)
    sq = None
    if view.metric == "euclidean":
        Xp = Xp - Xp.mean(axis=0, keepdims=True)                        # distance-preserving
        sq = (Xp * Xp).sum(axis=1)
    n = Xp.shape[0]
    topk = min(topk, n)
    U, inverse = np.unique(R[:, :3], return_inverse=True)
    rows_of = inverse.reshape(Q, 3)                                     # (Q, 3) rows into S
    S = Xp[U] @ Xp.T                                                    # (|U|, n), row-contiguous
    GU = S[:, U]                                                        # (|U|, |U|) query Gram
    ca, cs, cb = rows_of[:, 0], rows_of[:, 1], rows_of[:, 2]
    g = {"b_as": GU[cb, cs], "b_a": GU[cb, ca], "a_as": GU[ca, cs]}
    # (method, exclude) passes; the honest pass is 3CosAdd without exclusion
    passes = [(m, exclude_queries) for m in run]
    honest_key = ("3cosadd", False)
    if honest_key not in passes:
        passes.append(honest_key)
    ranks = {key: np.empty(Q, dtype=np.int64) for key in passes}
    tops = {key: np.empty((Q, topk), dtype=np.int64) for key in passes}
    for start in range(0, Q, _QUESTION_CHUNK):
        sl = slice(start, min(start + _QUESTION_CHUNK, Q))
        B = sl.stop - sl.start
        rows_b = np.arange(B)
        Sa, Ss, Sb = S[ca[sl]], S[cs[sl]], S[cb[sl]]                    # (B, n) each
        gb = {key: value[sl] for key, value in g.items()}
        for key in passes:
            method, exclude = key
            score = _method_scores(method, view.metric, Sa, Ss, Sb, gb, sq)   # (B, n)
            if exclude:
                for j in range(3):
                    score[rows_b, R[sl, j]] = -np.inf
            target = score[rows_b, R[sl, 3]]                            # (B,)
            ranks[key][sl] = 1 + (score > target[:, None]).sum(axis=1)
            if topk == 1:
                tops[key][sl, 0] = view.ids[np.argmax(score, axis=1)]
                continue
            part = np.argpartition(score, n - topk, axis=1)[:, n - topk:]   # (B, topk) unordered
            order = np.argsort(-np.take_along_axis(score, part, axis=1), axis=1, kind="stable")
            tops[key][sl] = view.ids[np.take_along_axis(part, order, axis=1)]
    for method in run:
        out["methods"][method] = {"rank": ranks[(method, exclude_queries)], "topk": tops[(method, exclude_queries)]}
    honest_rank, honest_top = ranks[honest_key], tops[honest_key]
    ids = view.ids[R]                                                   # (Q, 4)
    top1 = honest_top[:, 0]
    out["honest"] = {
        "rank": honest_rank, "top1": top1,
        "acc": float((honest_rank == 1).mean()),
        "top1_is_b": float((top1 == ids[:, 2]).mean()),
        "top1_is_a_star": float((top1 == ids[:, 1]).mean()),
        "top1_is_a": float((top1 == ids[:, 0]).mean()),
        "top1_is_b_or_a_star": float(((top1 == ids[:, 2]) | (top1 == ids[:, 1])).mean()),
    }
    return out


def _resampling_units(questions: Sequence[AnalogyQuestion]) -> Dict[str, Dict[str, Any]]:
    r"""Word-pair resampling units of analogy questions, per category.

    Within a category a question ``a : a* :: b : b*`` is formed from the ordered word pairs
    ``(a, a*)`` and ``(b, b*)``; the Google set pairs the word pairs of each category with each
    other, so every pair recurs in many questions and shares its hit status across them. A question
    both of whose pairs occur in another question of its category gets those two pair units. A
    question with a pair that occurs in no other question (no pair structure, e.g. a hand-built or
    random list) cannot be attributed to shared pairs and is its own unit (``unit_b = -1``).
    Returns ``category -> {"index", "unit_a", "unit_b", "n_units", "n_pairs", "n_question_units"}``;
    ``index`` holds the positions of the category's questions and ``n_pairs`` its distinct word pairs.
    """
    by_category: Dict[str, List[int]] = {}
    for position, q in enumerate(questions):
        by_category.setdefault(str(q.category), []).append(position)
    plan: Dict[str, Dict[str, Any]] = {}
    for category, index in by_category.items():
        keys = [((int(questions[i].a), int(questions[i].a_star)), (int(questions[i].b), int(questions[i].b_star)))
                for i in index]
        occurrences: Dict[Tuple[int, int], int] = {}
        for pair_a, pair_b in keys:
            for key in (pair_a, pair_b):
                occurrences[key] = occurrences.get(key, 0) + 1
        pair_id: Dict[Tuple[int, int], int] = {}
        unit_a = np.empty(len(index), dtype=np.int64)
        unit_b = np.empty(len(index), dtype=np.int64)
        single: List[int] = []
        for r, (pair_a, pair_b) in enumerate(keys):
            if occurrences[pair_a] >= 2 and occurrences[pair_b] >= 2:
                unit_a[r] = pair_id.setdefault(pair_a, len(pair_id))
                unit_b[r] = pair_id.setdefault(pair_b, len(pair_id))
            else:
                single.append(r)
        for j, r in enumerate(single):
            unit_a[r], unit_b[r] = len(pair_id) + j, -1
        plan[category] = {"index": np.asarray(index, dtype=np.int64), "unit_a": unit_a, "unit_b": unit_b,
                          "n_units": len(pair_id) + len(single), "n_pairs": len(occurrences),
                          "n_question_units": len(single)}
    return plan


def _benchmark_totals(totals: Optional[Mapping[str, Any]], given: Mapping[str, int]) -> Tuple[int, Dict[str, int]]:
    r"""Overall and per-category question totals: full benchmark from ``totals``, else the questions given."""
    if totals is None:
        return int(sum(given.values())), dict(given)
    per_category: Dict[str, int] = {}
    for category, entry in (totals.get("per_category") or {}).items():
        per_category[str(category)] = int(entry["n_total"] if isinstance(entry, Mapping) else entry)
    for category, count in given.items():
        per_category.setdefault(category, int(count))
    return int(totals.get("n_total", sum(per_category.values()))), per_category


def _interval(observed: float, hits: float, draws: np.ndarray, n: int, n_units: int, binary: bool,
              resampling: str) -> Dict[str, Any]:
    r"""Point value with its 95% interval: percentile of the unit-bootstrap draws, or Wilson at 0 / all hits."""
    record: Dict[str, Any] = {"mean": float(observed), "n": int(n), "n_boot_valid": int(draws.size)}
    if binary and (hits == 0 or hits == n):
        lo, hi = stats.wilson_interval(n_units if hits == n else 0, n_units)
        unit = "pairs" if resampling == "pairs" else "units"
        record.update(ci_lo=lo, ci_hi=hi, ci_method=f"Wilson on {n_units} {unit}")
    elif draws.size:
        lo, hi = np.percentile(draws, [2.5, 97.5])
        record.update(ci_lo=float(lo), ci_hi=float(hi), ci_method=f"bootstrap over {resampling}")
    else:
        record.update(ci_lo=float("nan"), ci_hi=float("nan"), ci_method="none (no valid bootstrap draw)")
    return record


def analogy_benchmark(view: VectorView, questions: Sequence[AnalogyQuestion], rng: np.random.Generator,
                      *, center: bool = True, n_boot: int = 1000,
                      totals: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    r"""Accuracy, hits@5 and MRR with word-pair bootstrap 95% CIs, per method and category.

    Groups: ``overall``, ``semantic`` (categories not starting with ``gram``), ``syntactic``
    (``gram*``) and every category. The honest (no-exclusion) 3CosAdd numbers are reported per group.

    Intervals resample word pairs, not questions (:func:`_resampling_units`): per draw and per
    category the pairs are drawn with replacement (multiplicities ``c_p``) and question
    ``q = (i, j)`` is weighted ``w_q = c_i c_j`` (:func:`stats.unit_bootstrap_weights`), so

    .. math:: \mathrm{acc}^{*} = \frac{\sum_c \sum_{q \in c} w_q h_q}{\sum_c \sum_{q \in c} w_q}

    sums weights and hits over the group's categories within the same draw (a stratified bootstrap;
    a pair listed in two categories counts as two units). Every method and group uses the same
    draws. A 0/1 statistic whose observed hit count is 0 or all questions gets the Wilson interval
    on the number of resampling units instead of the degenerate percentile interval.

    ``totals`` (``ctx.coverage["google_analogies"]``: ``n_total`` and ``per_category`` entries with
    ``n_total``) sets the full-benchmark denominators; without it ``n_total`` counts the questions
    given. Per group: ``n`` = ``n_questions`` covered by this view, ``n_pairs`` distinct word pairs,
    ``n_units`` resampling units, ``n_resolved`` questions given, ``n_total`` and
    ``coverage = n / n_total``.
    """
    res = analogy_scores(view, questions, center=center, exclude_queries=True, topk=1)
    covered = res["questions"]
    Q = len(covered)
    columns: List[Tuple[str, str]] = []
    values: List[np.ndarray] = []
    for method, data in res["methods"].items():
        rank = data["rank"].astype(np.float64)
        for stat, value in (("acc", rank == 1), ("hits5", rank <= 5), ("mrr", 1.0 / rank)):
            columns.append((method, stat))
            values.append(np.asarray(value, dtype=np.float64))
    if Q:
        columns.append(("honest", "acc"))
        values.append((res["honest"]["rank"] == 1).astype(np.float64))
    H = np.stack(values, axis=1) if Q else np.zeros((0, 0))            # (Q, M) per-question statistics
    binary = np.array([stat != "mrr" for _, stat in columns], dtype=bool)

    plan = _resampling_units(covered)
    boot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for category in sorted(plan):
        p = plan[category]
        Hc = H[p["index"]]                                              # (Q_c, M)
        w_sum = np.empty(n_boot)
        h_sum = np.empty((n_boot, H.shape[1]))
        for start in range(0, n_boot, _BOOT_CHUNK):
            stop = min(start + _BOOT_CHUNK, n_boot)
            W = stats.unit_bootstrap_weights(p["unit_a"], p["unit_b"], p["n_units"], rng, stop - start)  # (b, Q_c)
            w_sum[start:stop] = W.sum(axis=1)
            h_sum[start:stop] = W @ Hc
        boot[category] = (w_sum, h_sum)

    given: Dict[str, int] = {}
    for q in questions:
        given[str(q.category)] = given.get(str(q.category), 0) + 1
    n_total_all, per_category_total = _benchmark_totals(totals, given)
    categories = sorted(set(given) | set(per_category_total))
    ids = np.array([[q.a, q.a_star, q.b] for q in covered], dtype=np.int64).reshape(-1, 3)

    def summarize(members: Sequence[str], n_total: int) -> Dict[str, Any]:
        present = [c for c in members if c in plan]
        n = int(sum(plan[c]["index"].size for c in present))
        n_units = int(sum(plan[c]["n_units"] for c in present))
        n_single = int(sum(plan[c]["n_question_units"] for c in present))
        resampling = "pairs" if n_single == 0 else ("questions" if n_single == n else "pairs and questions")
        record: Dict[str, Any] = {
            "n": n, "n_questions": n, "n_pairs": int(sum(plan[c]["n_pairs"] for c in present)),
            "n_units": n_units, "n_question_units": n_single, "resampling": resampling,
            "n_resolved": int(sum(given.get(c, 0) for c in members)), "n_total": int(n_total),
            "coverage": n / n_total if n_total else float("nan"), "methods": {},
        }
        if n == 0:
            return record
        index = np.concatenate([plan[c]["index"] for c in present])
        observed = H[index].mean(axis=0)                                # (M,)
        hits = H[index].sum(axis=0)
        w = sum(boot[c][0] for c in present)                            # (B,)
        h = sum(boot[c][1] for c in present)                            # (B, M)
        valid = w > 0
        ratio = h[valid] / w[valid, None]                               # (B_valid, M)
        intervals = {key: _interval(observed[k], hits[k], ratio[:, k], n, n_units, bool(binary[k]), resampling)
                     for k, key in enumerate(columns)}
        for method in res["methods"]:
            record["methods"][method] = {stat: intervals[(method, stat)] for stat in ("acc", "hits5", "mrr")}
        top1 = res["honest"]["top1"][index]
        sub = ids[index]
        record["honest"] = {
            "acc": intervals[("honest", "acc")],
            "top1_is_b": float((top1 == sub[:, 2]).mean()),
            "top1_is_a_star": float((top1 == sub[:, 1]).mean()),
            "top1_is_a": float((top1 == sub[:, 0]).mean()),
            "top1_is_b_or_a_star": float(((top1 == sub[:, 2]) | (top1 == sub[:, 1])).mean()),
        }
        return record

    semantic = [c for c in categories if not c.startswith("gram")]
    syntactic = [c for c in categories if c.startswith("gram")]
    overall = summarize(categories, n_total_all)
    n_single = overall["n_question_units"]
    note = (f"95% CIs resample the {overall['n_pairs']} word pairs within categories (question (i, j) "
            "weighted c_i c_j); Wilson intervals on the unit count where a 0/1 statistic has 0 or all hits")
    if n_single:
        note += (f"; {n_single} of {Q} covered questions have a word pair that occurs in no other question of "
                 "their category and are resampled as single questions")
    return {
        "n_total": n_total_all, "n_resolved": len(questions), "n_covered": res["n_covered"],
        "n_degenerate": res["n_degenerate"], "n_pairs": overall["n_pairs"],
        "coverage": overall["coverage"],
        "totals_source": "full benchmark" if totals is not None else "questions given",
        "resampling": overall["resampling"], "resampling_note": note, "n_boot": int(n_boot),
        "methods": list(res["methods"].keys()), "skipped_methods": res["skipped_methods"],
        "overall": overall,
        "semantic": summarize(semantic, sum(per_category_total.get(c, 0) for c in semantic)),
        "syntactic": summarize(syntactic, sum(per_category_total.get(c, 0) for c in syntactic)),
        "categories": {c: summarize([c], per_category_total.get(c, 0)) for c in categories},
    }


def analogy_examples(view: VectorView, curated: Sequence[AnalogyQuestion], *, center: bool = True,
                     topk: int = 5) -> List[Dict[str, Any]]:
    r"""One row per curated analogy: per-method top-``k`` ids and the rank of the answer.

    Rows for questions not covered by the view carry ``covered=False`` and no method results.
    """
    res = analogy_scores(view, curated, center=center, exclude_queries=True, topk=topk)
    position = {int(i): r for r, i in enumerate(res["covered_index"])}
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(curated):
        row: Dict[str, Any] = {"category": q.category, "a": int(q.a), "a_star": int(q.a_star),
                               "b": int(q.b), "b_star": int(q.b_star), "covered": i in position,
                               "methods": {}}
        if i in position:
            r = position[i]
            for method, data in res["methods"].items():
                row["methods"][method] = {"rank": int(data["rank"][r]),
                                          "topk": [int(t) for t in data["topk"][r]]}
            row["honest_top1"] = int(res["honest"]["top1"][r])
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# Relation offsets and concept directions
# ----------------------------------------------------------------------------------------------

def _cyclic_derangement(m: int, rng: np.random.Generator) -> np.ndarray:
    r"""A uniformly random cyclic permutation of ``range(m)`` (no fixed points)."""
    sigma = rng.permutation(m)
    pi = np.empty(m, dtype=np.int64)
    pi[sigma] = sigma[(np.arange(m) + 1) % m]
    return pi


def _offset_stats_from_gram(Gd: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    r"""OCS, LOO projections, LOO cosines and top-PC share from the offset Gram ``G_ij = <d_i, d_j>``.

    OCS is the mean pairwise cosine ``mean_{i != j} G_ij / (||d_i|| ||d_j||)``. The LOO projection
    of ``d_i`` on the unit mean of the others is ``(r_i - G_ii) / ||t - d_i||`` with ``r_i = sum_j G_ij``,
    ``||t - d_i||^2 = sum G - 2 r_i + G_ii``. The top-PC share is ``lambda_max(G)/tr G`` (uncentered
    PCA of the offsets, i.e. of the ``+/- d_i/2`` pair deviations of Bolukbasi et al.).
    """
    m = Gd.shape[0]
    norms = np.sqrt(np.clip(np.diag(Gd), 0.0, None))
    if (norms <= 0.0).any():
        raise ValueError("a relation offset has zero length")
    C = Gd / np.outer(norms, norms)
    ocs = float((C.sum() - m) / (m * (m - 1)))
    r = Gd.sum(axis=1)
    rest = np.sqrt(np.clip(Gd.sum() - 2.0 * r + np.diag(Gd), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = np.where(rest > 0.0, (r - np.diag(Gd)) / rest, np.nan)
    share = float(np.linalg.eigvalsh(Gd)[-1] / np.trace(Gd))
    return ocs, proj, proj / norms, share


def offset_consistency(view: VectorView, relations: Mapping[str, Sequence[Tuple[int, int]]],
                       rng: np.random.Generator, n_perm: int = 500, center: bool = True) -> Dict[str, Any]:
    r"""Consistency of the offsets ``delta_i = x_{a*_i} - x_{a_i}`` within each relation (>= 3 pairs).

    * ``ocs`` -- Offset Concentration Score: mean pairwise cosine of the offsets
      (Fournier, Dupoux & Dunbar 2020);
    * re-pairing null -- the same with partners re-paired by a random cyclic derangement,
      ``delta'_i = x_{a*_{pi(i)}} - x_{a_i}``; ``z`` and one-sided ``p = (1 + #\{null >= obs\})/(1 + n_perm)``;
    * ``loo_proj`` -- projection of each offset on the unit mean of the other offsets
      (Park, Choe & Veitch 2024, Fig. 2), with the pooled re-paired projections (``null_loo_proj``);
    * random-pair baseline (Park, Choe & Veitch 2024, Fig. 2) -- ``m`` random pairs ``(r_k, s_k)``
      of distinct rows of the view, ``e_k = x_{s_k} - x_{r_k}``, projected on the leave-one-out
      direction of the ``k``-th real pair, :math:`e_k\cdot t_{-k}/\lVert t_{-k}\rVert` with
      :math:`t_{-k} = \sum_{j\ne k}\delta_j` (``random_loo_proj``), and on the unit mean direction
      :math:`t/\lVert t\rVert` (``random_mean_proj_*``);
    * ``top_pc_share`` -- variance share of the first principal direction of the offsets
      (Bolukbasi et al. 2016), and its re-pairing null mean.

    The re-pairing null keeps the two word SETS, hence their mean offset: it tests whether the
    specific pairing carries a consistent offset (the pairing-consistency logic of Fournier et al.),
    not whether the ``a*`` words sit apart from the ``a`` words as groups, so re-paired LOO
    projections of a relation with a large group offset stay near the real ones. The random-pair
    baseline is the one that tests whether the direction is specific to the relation. OCS and the
    re-pairing null are computed from ``m x m`` Gram blocks (no ``D``-dimensional work per draw).
    """
    Xp = prepare(view, center)
    out: Dict[str, Any] = {"relations": {}, "skipped": {}, "n_perm": int(n_perm), "center": bool(center)}
    for name, pairs in relations.items():
        ra, rb, kept = _row_pairs(view, pairs)
        m = len(kept)
        if m < 3:
            out["skipped"][name] = f"{m} resolved pair(s) in view (need >= 3)"
            continue
        A, B = Xp[ra], Xp[rb]                                           # (m, D) each
        G_AA, G_BB, G_AB = A @ A.T, B @ B.T, A @ B.T                    # (m, m) each

        def gram(pi: np.ndarray) -> np.ndarray:
            # <B_pi(i) - A_i, B_pi(j) - A_j>
            return G_BB[np.ix_(pi, pi)] - G_AB[:, pi].T - G_AB[:, pi] + G_AA

        identity = np.arange(m)
        ocs, proj, loo_cos, share = _offset_stats_from_gram(gram(identity))
        null_ocs = np.empty(n_perm)
        null_share = np.empty(n_perm)
        null_proj: List[np.ndarray] = []
        for t in range(n_perm):
            o, p, _, s = _offset_stats_from_gram(gram(_cyclic_derangement(m, rng)))
            null_ocs[t], null_share[t] = o, s
            if t < 50:
                null_proj.append(p)
        null_std = float(null_ocs.std(ddof=1)) if n_perm > 1 else float("nan")
        pooled = np.concatenate(null_proj) if null_proj else np.zeros(0)
        # random-pair baseline: m random pairs of distinct view rows on the real LOO / mean directions
        d = B - A                                                       # (m, D) real offsets
        total = d.sum(axis=0)                                           # (D,)
        first = rng.integers(0, Xp.shape[0], size=m)
        second = (first + rng.integers(1, Xp.shape[0], size=m)) % Xp.shape[0]   # distinct from first
        e = Xp[second] - Xp[first]                                      # (m, D) random offsets
        rest = total[None, :] - d                                       # (m, D) LOO sums t_{-k}
        rest_norm = np.linalg.norm(rest, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            random_loo = np.where(rest_norm > 0.0, (e * rest).sum(axis=1) / rest_norm, np.nan)
        total_norm = float(np.linalg.norm(total))
        random_mean = e @ (total / total_norm) if total_norm > 0.0 else np.full(m, np.nan)
        out["relations"][name] = {
            "n_pairs": m, "pairs": kept, "ocs": ocs,
            "null_mean": float(null_ocs.mean()) if n_perm else float("nan"), "null_std": null_std,
            "z": float((ocs - null_ocs.mean()) / null_std) if n_perm > 1 and null_std > 0 else float("nan"),
            "p": float((1 + (null_ocs >= ocs).sum()) / (1 + n_perm)) if n_perm else float("nan"),
            "loo_proj": proj, "loo_cos": loo_cos, "loo_proj_mean": float(np.nanmean(proj)),
            "loo_proj_sd": float(np.nanstd(proj, ddof=1)),
            "null_loo_proj": pooled,
            "null_loo_proj_mean": float(np.nanmean(pooled)) if pooled.size else float("nan"),
            "n_random_pairs": m, "random_loo_proj": random_loo,
            "random_loo_proj_mean": float(np.nanmean(random_loo)),
            "random_loo_proj_sd": float(np.nanstd(random_loo, ddof=1)),
            "random_mean_proj_mean": float(np.nanmean(random_mean)),
            "random_mean_proj_sd": float(np.nanstd(random_mean, ddof=1)),
            "top_pc_share": share,
            "null_top_pc_share_mean": float(null_share.mean()) if n_perm else float("nan"),
        }
    return out


def _whitener(X: np.ndarray) -> Tuple[np.ndarray, int]:
    r"""``Cov(X)^{-1/2}`` restricted to the numerically nonzero eigenvalues (``> 1e-10 lambda_max``)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)                           # (D, D)
    w, V = np.linalg.eigh(cov)
    keep = w > 1e-10 * max(float(w.max()), 0.0)
    if not keep.any():
        raise ValueError("covariance is zero; cannot whiten")
    Vk = V[:, keep]
    return (Vk / np.sqrt(w[keep])) @ Vk.T, int(keep.sum())


def concept_direction_cosines(view: VectorView, relations: Mapping[str, Sequence[Tuple[int, int]]],
                              *, whiten: bool) -> Dict[str, Any]:
    r"""Cosine matrix between the mean offset directions of the relations (>= 2 pairs each).

    Direction of relation ``W``: ``gamma_W = mean_i (x_{a*_i} - x_{a_i})`` on the RAW rows.
    ``whiten=False`` compares them with the Euclidean inner product. ``whiten=True`` uses the causal
    inner product of Park, Choe & Veitch (2024), ``<u, v>_C = u^T Cov(x)^{-1} v`` with the covariance
    over the view's rows, i.e. cosines after mapping ``u -> Cov^{-1/2} u``. For an unembedding view
    this is their estimate; for other views it is the same whitening applied as a chart normalization.
    When ``Cov`` is singular the pseudo-inverse square root on its range is used and
    ``whitening_rank < dim`` records it.
    """
    X = view.X.astype(np.float64)                                      # (n, D)
    names, directions, n_pairs, skipped = [], [], [], {}
    for name, pairs in relations.items():
        ra, rb, kept = _row_pairs(view, pairs)
        if len(kept) < 2:
            skipped[name] = f"{len(kept)} resolved pair(s) in view (need >= 2)"
            continue
        names.append(name)
        n_pairs.append(len(kept))
        directions.append((X[rb] - X[ra]).mean(axis=0))
    out: Dict[str, Any] = {"whiten": bool(whiten), "relations": names, "n_pairs": n_pairs,
                           "skipped": skipped, "dim": int(view.dim), "whitening_rank": None}
    if len(names) < 2:
        out.update(status="skipped", reason=f"{len(names)} relation(s) with >= 2 pairs (need >= 2)")
        return out
    Dm = np.stack(directions)                                           # (R, D)
    if whiten:
        W, rank = _whitener(X)
        Dm = Dm @ W                                                     # W symmetric
        out["whitening_rank"] = rank
    Dm = stats.unit_rows(Dm)
    cos = Dm @ Dm.T                                                     # (R, R)
    off = cos[~np.eye(len(names), dtype=bool)]
    out.update(cosine=cos, mean_abs_offdiag=float(np.abs(off).mean()))
    return out


# ----------------------------------------------------------------------------------------------
# Similarity, fields, clusters, projections
# ----------------------------------------------------------------------------------------------

def view_similarity_scores(view: VectorView, pairs: Sequence[Tuple[int, int, float]],
                           center: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    r"""Model scores and human scores for the pairs whose two words are in the view.

    Cosine of prepared rows for a cosine view; negative Euclidean distance for a Euclidean view.
    """
    keep = [(int(a), int(b), float(h)) for a, b, h in pairs if int(a) != int(b)
            and view.has(a) and view.has(b)]
    if not keep:
        return np.zeros(0), np.zeros(0)
    Xp = prepare(view, center)
    r1 = view.rows([a for a, _, _ in keep])
    r2 = view.rows([b for _, b, _ in keep])
    if view.metric == "cosine":
        scores = (Xp[r1] * Xp[r2]).sum(axis=1)
    else:
        scores = -np.linalg.norm(Xp[r1] - Xp[r2], axis=1)
    return scores, np.array([h for _, _, h in keep], dtype=np.float64)


def similarity_benchmark(scores: np.ndarray, human: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    r"""Spearman ``rho`` between model and human similarity with a bootstrap 95% CI over pairs."""
    return stats.spearman_ci(scores, human, rng)


def field_clustering(view: VectorView, lexicon: Mapping[str, Sequence[int]], rng: np.random.Generator,
                     k: int = 10, n_perm: int = 200) -> Dict[str, Any]:
    r"""How well the view separates the semantic fields of the lexicon.

    Restricted to lexicon words in the view, in fields with >= 2 such words. Reports the silhouette
    in the original space (cosine on prepared rows for a cosine view, Euclidean otherwise) and the
    kNN label purity with its permutation null (:func:`stats.knn_label_purity`), overall and per
    field. Cluster labels are the external fields, never clusters fitted on the same rows.
    """
    rows, labels, info = _lexicon_rows(view, lexicon)
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    small = {f for f, c in counts.items() if c < 2}
    keep = np.array([lab not in small for lab in labels], dtype=bool)
    rows = rows[keep]
    labels = [lab for lab, kk in zip(labels, keep) if kk]
    out: Dict[str, Any] = {"n_words": int(rows.size), "n_fields": len(set(labels)),
                           "fields_dropped": sorted(small), **info}
    if len(set(labels)) < 2:
        out.update(status="skipped", reason=f"{len(set(labels))} field(s) with >= 2 words in view")
        return out
    Xp = prepare(view, center=True)[rows]
    metric = "cosine" if view.metric == "cosine" else "euclidean"
    sil = stats.silhouette(Xp, labels, metric=metric)
    pur = stats.knn_label_purity(Xp, labels, k=k, metric=metric, rng=rng, n_perm=n_perm)
    out.update(silhouette=sil, purity=pur, per_field={
        f: {"n": pur["per_label_n"][f], "purity": pur["per_label"][f], "silhouette": sil["per_label"][f]}
        for f in sorted(set(labels))})
    return out


def cluster_vocabulary(view: VectorView, top_ids: Sequence[int], k_clusters: int = 40,
                       rng: Optional[np.random.Generator] = None, n_central: int = 12) -> Dict[str, Any]:
    r"""Exploratory k-means (``n_init=4``, seeded) on the prepared rows of the frequent words.

    Each cluster reports its size and the ``n_central`` members nearest its centroid. With
    ``rng=None`` the k-means seed is 0; the seed actually used is returned.
    """
    from sklearn.cluster import KMeans

    rows = view.rows(top_ids)
    rows = rows[rows >= 0]
    if rows.size < 2:
        return {"status": "skipped", "reason": f"{rows.size} of the top words in view"}
    Xp = prepare(view, center=True)[rows]
    k = min(int(k_clusters), int(rows.size))
    seed = int(rng.integers(0, 2 ** 31 - 1)) if rng is not None else 0
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(Xp)
    clusters = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        dist = np.linalg.norm(Xp[members] - km.cluster_centers_[c], axis=1)
        central = members[np.argsort(dist, kind="stable")[:n_central]]
        clusters.append({"size": int(members.size), "central_ids": [int(t) for t in view.ids[rows[central]]]})
    clusters.sort(key=lambda c: -c["size"])
    for i, c in enumerate(clusters):
        c["cluster"] = i
    return {"k": k, "n": int(rows.size), "seed": seed, "inertia": float(km.inertia_), "clusters": clusters}


def semantic_projection(view: VectorView, axis: Mapping[str, Any], center: bool = True) -> Dict[str, Any]:
    r"""Projection of words onto an antonym axis (Grand et al. 2022).

    ``axis`` = ``{"pairs": [(neg_id, pos_id), ...], "words": [ids], "neg_label": str,
    "pos_label": str}`` (resolved token ids). The axis direction is the unit mean of
    ``x_pos - x_neg`` over the pairs in the view (prepared rows); each word's score is its inner
    product with it. ``loo_pair_accuracy`` is the fraction of pairs whose positive pole scores
    above its negative pole on the axis built from the OTHER pairs (a held-out validity check).
    """
    ra, rb, kept = _row_pairs(view, [(int(n), int(p)) for n, p in axis["pairs"]])
    if not kept:
        return {"status": "skipped", "reason": "no axis pair in view"}
    Xp = prepare(view, center)
    diff = Xp[rb] - Xp[ra]                                              # (m, D) pos - neg
    u = diff.mean(axis=0)
    u_norm = float(np.linalg.norm(u))
    if u_norm <= 0.0:
        raise ValueError("axis direction has zero length")
    u = u / u_norm
    loo = None
    if len(kept) >= 2:
        total = diff.sum(axis=0)
        hits = []
        for i in range(len(kept)):
            rest = total - diff[i]
            hits.append(float(diff[i] @ rest) > 0.0)
        loo = float(np.mean(hits))
    words = [int(w) for w in axis.get("words", []) if view.has(w)]
    wrows = view.rows(words)
    scores = Xp[wrows] @ u if words else np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    return {
        "n_pairs": len(kept), "n_words": len(words),
        "neg_label": axis.get("neg_label", "neg"), "pos_label": axis.get("pos_label", "pos"),
        "words": [(words[i], float(scores[i])) for i in order],
        "pair_words": [{"neg": a, "pos": b, "neg_score": float(Xp[r0] @ u), "pos_score": float(Xp[r1] @ u)}
                       for (a, b), r0, r1 in zip(kept, ra, rb)],
        "separation": float((Xp[rb] @ u).mean() - (Xp[ra] @ u).mean()),
        "loo_pair_accuracy": loo,
    }


# ----------------------------------------------------------------------------------------------
# Global geometry
# ----------------------------------------------------------------------------------------------

def _attempt(store: Dict[str, Any], key: str, fn: Callable[[], Any], notes: List[str]) -> None:
    r"""Run one analysis; on failure store a ``failed`` record with the reason and note it."""
    try:
        store[key] = fn()
    except Exception as exc:  # recorded, not swallowed: status + error + note
        store[key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-2000:]}
        notes.append(f"{key} failed: {type(exc).__name__}: {exc}")


def global_geometry(view: VectorView, log_count: Optional[np.ndarray], rng: np.random.Generator,
                    *, max_points: int = 8000) -> Dict[str, Any]:
    r"""Anisotropy, isotropy, dimension, hubness, rogue dimensions and frequency confounds.

    Rows: the ``max_points`` most frequent (by ``log_count``; the first rows in view order when
    counts are unavailable). Raw rows: anisotropy (raw and centered), mean-vector ratio, IsoScore,
    effective rank, participation ratio, PCA spectrum (top 50), rogue dimensions, frequency probe,
    Spearman of the first five PC scores (eigenvector signs are arbitrary, so only ``|rho|`` is
    meaningful) and of ``||x||`` with log count. Prepared rows (the
    battery's metric): TwoNN and hubness (k = 10). For views wider than 512 the frequency probe
    uses the top 512 principal components (``pca_components`` in its record).
    """
    notes: List[str] = []
    n_all = view.n
    if log_count is not None:
        log_count = np.asarray(log_count, dtype=np.float64).reshape(-1)
        if log_count.size != n_all:
            raise ValueError(f"log_count has {log_count.size} entries for {n_all} rows")
        sel = np.argsort(-log_count, kind="stable")[:max_points]
        selection = "top-frequency"
    else:
        sel = np.arange(min(max_points, n_all))
        selection = "view order (counts unavailable)"
        notes.append("counts unavailable: geometry rows are the first rows in view order; "
                     "frequency analyses skipped")
    X = view.X[sel].astype(np.float64)                                  # (m, D)
    lc = log_count[sel] if log_count is not None else None
    m, D = X.shape
    out: Dict[str, Any] = {"n": int(m), "n_view": int(n_all), "dim": int(D), "selection": selection,
                           "notes": notes}
    Xc = X - X.mean(axis=0, keepdims=True)
    gram = Xc.T @ Xc if D <= m else Xc @ Xc.T
    w, V = np.linalg.eigh(gram)
    w, V = np.clip(w[::-1], 0.0, None), V[:, ::-1]
    lam = np.zeros(D)
    lam[: min(D, w.size)] = w[:D]
    n_pc = min(5, int((w > 0).sum()))
    pcs = (Xc @ V[:, :n_pc]) if D <= m else V[:, :n_pc] * np.sqrt(w[:n_pc])   # (m, n_pc) PC scores

    _attempt(out, "anisotropy_raw", lambda: stats.anisotropy(X), notes)
    _attempt(out, "anisotropy_centered", lambda: stats.anisotropy(X, center=True), notes)
    _attempt(out, "mean_vector_ratio", lambda: stats.mean_vector_ratio(X), notes)
    _attempt(out, "isoscore", lambda: stats.isoscore_from_eigenvalues(lam), notes)
    _attempt(out, "effective_rank", lambda: stats.effective_rank_from_eigenvalues(lam), notes)
    _attempt(out, "participation_ratio", lambda: stats.participation_ratio_from_eigenvalues(lam), notes)
    _attempt(out, "pca_spectrum", lambda: stats.explained_variance_from_eigenvalues(lam, 50), notes)
    Xp = _prepare_array(X, view.metric, center=True)
    _attempt(out, "twonn", lambda: stats.twonn_dimension(Xp, rng), notes)
    _attempt(out, "hubness", lambda: stats.hubness(Xp, k=10, metric=view.metric), notes)
    if _ok(out.get("hubness")):
        out["hubness"]["top_ids"] = view.ids[sel][out["hubness"]["top_rows"]]
    _attempt(out, "rogue_dimensions", lambda: stats.rogue_dimensions(X), notes)
    if lc is not None:
        # wide views: probe on the top principal components already computed above (target-free)
        n_probe = min(_PROBE_MAX_FEATURES, int((w > 0).sum()))
        if D > _PROBE_MAX_FEATURES:
            X_probe = (Xc @ V[:, :n_probe]) if D <= m else V[:, :n_probe] * np.sqrt(w[:n_probe])
            _attempt(out, "frequency_probe",
                     lambda: {**stats.frequency_probe(X_probe, lc, rng), "pca_components": n_probe}, notes)
        else:
            _attempt(out, "frequency_probe", lambda: stats.frequency_probe(X, lc, rng), notes)
        _attempt(out, "pc_log_count_spearman",
                 lambda: [stats.spearman_ci(pcs[:, i], lc, rng) for i in range(n_pc)], notes)
        _attempt(out, "norm_log_count_spearman",
                 lambda: stats.spearman_ci(np.linalg.norm(X, axis=1), lc, rng), notes)
    return out


# ----------------------------------------------------------------------------------------------
# Battery
# ----------------------------------------------------------------------------------------------

def _google_totals(ctx: AnalysisContext) -> Optional[Mapping[str, Any]]:
    r"""``ctx.coverage["google_analogies"]`` when it carries full-benchmark totals, else ``None``."""
    google = ctx.coverage.get("google_analogies") if isinstance(ctx.coverage, Mapping) else None
    if isinstance(google, Mapping) and google.get("available") and "per_category" in google:
        return google
    return None


def _failed_steps(battery: Mapping[str, Any]) -> List[Dict[str, str]]:
    r"""``{"step", "error"}`` of every failed step of a battery, including failed sub-steps (``geometry/twonn``)."""
    found: List[Dict[str, str]] = []
    for key, value in battery.items():
        if not isinstance(value, Mapping):
            continue
        if value.get("status") == "failed":
            found.append({"step": str(key), "error": str(value.get("error", "?"))})
            continue
        for sub, inner in value.items():
            if isinstance(inner, Mapping) and inner.get("status") == "failed":
                found.append({"step": f"{key}/{sub}", "error": str(inner.get("error", "?"))})
    return found


def run_vector_battery(view: VectorView, ctx: AnalysisContext) -> Dict[str, Any]:
    r"""Run every analysis of this module on ``view`` with the resources of ``ctx``.

    Returns one JSON-able dict (after :func:`common.to_jsonable`) with keys ``view``, ``neighbors``,
    ``analogy_benchmark``, ``analogy_examples``, ``offsets``, ``directions_raw``,
    ``directions_whitened``, ``similarity`` (per benchmark), ``fields``, ``clusters``,
    ``projections`` (per axis), ``geometry``, ``coverage``, ``failed_steps``, ``notes`` and
    ``seconds``. A failing analysis is stored as ``{"status": "failed", "error": ...}``, noted, and
    listed in ``failed_steps`` as ``{"step", "error"}`` (sub-steps as ``"geometry/twonn"``); an
    inapplicable one is ``{"status": "skipped", "reason": ...}``. The analogy benchmark uses the
    full-benchmark denominators of ``ctx.coverage["google_analogies"]`` when present. Every
    stochastic step draws from ``np.random.default_rng([ctx.seed, step])``.
    Options read: ``geometry_max_points`` (8000), ``cluster_top_words`` (5000), ``n_clusters`` (40).
    """
    notes: List[str] = []
    seconds: Dict[str, float] = {}
    battery: Dict[str, Any] = {
        "view": {"name": view.name, "label": view.label, "metric": view.metric, "n": view.n,
                 "dim": view.dim, "note": view.note, "group": view.group},
        "notes": notes, "seconds": seconds,
    }

    def timed(key: str, fn: Callable[[], Any]) -> None:
        start = time.perf_counter()
        _attempt(battery, key, fn, notes)
        seconds[key] = time.perf_counter() - start

    def skipped(reason: str) -> Dict[str, Any]:
        return {"status": "skipped", "reason": reason}

    timed("neighbors", lambda: {int(q): [[int(t), s] for t, s in nn] for q, nn in
                                nearest_neighbors(view, ctx.probe_ids, k=10).items()})
    totals = _google_totals(ctx)
    timed("analogy_benchmark", lambda: analogy_benchmark(view, ctx.analogies, _rng(ctx, "analogy_benchmark"),
                                                         totals=totals)
          if ctx.analogies else skipped("no benchmark analogy questions resolved"))
    timed("analogy_examples", lambda: analogy_examples(view, ctx.curated_analogies)
          if ctx.curated_analogies else skipped("no curated analogies resolved"))
    timed("offsets", lambda: offset_consistency(view, ctx.relations, _rng(ctx, "offsets"))
          if ctx.relations else skipped("no relations resolved"))
    timed("directions_raw", lambda: concept_direction_cosines(view, ctx.relations, whiten=False)
          if ctx.relations else skipped("no relations resolved"))
    timed("directions_whitened", lambda: concept_direction_cosines(view, ctx.relations, whiten=True)
          if ctx.relations else skipped("no relations resolved"))

    def similarity() -> Dict[str, Any]:
        rng = _rng(ctx, "similarity")
        result: Dict[str, Any] = {}
        for bench, pairs in ctx.similarity.items():
            scores, human = view_similarity_scores(view, pairs)
            if scores.size < 3:
                result[bench] = {**skipped(f"{scores.size} pair(s) in view"), "n_pairs_total": len(pairs)}
                continue
            result[bench] = {**similarity_benchmark(scores, human, rng), "n_pairs_total": len(pairs)}
        return result

    timed("similarity", similarity)
    timed("fields", lambda: field_clustering(view, ctx.lexicon, _rng(ctx, "fields"))
          if ctx.lexicon else skipped("no lexicon resolved"))
    top = np.asarray(ctx.candidate_ids, dtype=np.int64)[: int(ctx.options.get("cluster_top_words", 5000))]
    timed("clusters", lambda: cluster_vocabulary(view, top, int(ctx.options.get("n_clusters", 40)),
                                                 _rng(ctx, "clusters")))
    timed("projections", lambda: {name: semantic_projection(view, axis) for name, axis in ctx.axes.items()})
    timed("geometry", lambda: global_geometry(
        view, ctx.log_count(view.ids), _rng(ctx, "geometry"),
        max_points=int(ctx.options.get("geometry_max_points", 8000))))

    lex_rows, _, _ = _lexicon_rows(view, ctx.lexicon)
    battery["coverage"] = {
        "n_rows": view.n,
        "probe_words": int(sum(view.has(t) for t in ctx.probe_ids)),
        "probe_words_total": len(ctx.probe_ids),
        "analogies": _covered_questions(view, ctx.analogies)[1].shape[0],
        "analogies_resolved": len(ctx.analogies),
        "analogies_total": int(totals["n_total"]) if totals is not None else len(ctx.analogies),
        "curated_analogies": _covered_questions(view, ctx.curated_analogies)[1].shape[0],
        "curated_analogies_total": len(ctx.curated_analogies),
        "lexicon_words": int(lex_rows.size),
        "lexicon_words_total": int(sum(len(v) for v in ctx.lexicon.values())),
        "relations": {name: len(_row_pairs(view, pairs)[2]) for name, pairs in ctx.relations.items()},
        "similarity": {bench: int(sum(int(a) != int(b) and view.has(a) and view.has(b) for a, b, _ in pairs))
                       for bench, pairs in ctx.similarity.items()},
        "similarity_total": {bench: len(pairs) for bench, pairs in ctx.similarity.items()},
    }
    battery["failed_steps"] = _failed_steps(battery)
    return battery


# ----------------------------------------------------------------------------------------------
# Figure helpers
# ----------------------------------------------------------------------------------------------

def _place_labels(ax: Any, xy: np.ndarray, texts: Sequence[str], *, obstacles: Optional[np.ndarray] = None,
                  fontsize: float = 6.5, color: str = "#222222", marker_pt: float = 2.5) -> int:
    r"""Greedy non-overlapping labels: each label tries 8 offsets and is dropped if all collide.

    A label may not overlap a label already placed, leave the axes, or cover any point of
    ``obstacles`` (data coordinates; a square of half-width ``marker_pt`` points around each).
    Axis limits must be final before calling. Returns the number of labels placed.
    """
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    half = marker_pt * fig.dpi / 72.0
    points = ax.transData.transform(np.asarray(obstacles if obstacles is not None else xy, dtype=np.float64))
    blocked = [Bbox.from_extents(px - half, py - half, px + half, py + half) for px, py in points]
    placed: List[Any] = []
    offsets = ((3, 3), (3, -3), (-3, 3), (-3, -3), (5, 0), (-5, 0), (0, 5), (0, -5))
    count = 0
    for (x, y), text in zip(xy, texts):
        for dx, dy in offsets:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                              ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                              va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                              fontsize=fontsize, color=color)
            box = ann.get_window_extent(renderer)
            inside = (box.x0 >= frame.x0 and box.x1 <= frame.x1 and box.y0 >= frame.y0 and box.y1 <= frame.y1)
            if inside and not any(box.overlaps(b) for b in placed) and not any(box.overlaps(b) for b in blocked):
                placed.append(box.expanded(1.04, 1.10))
                count += 1
                break
            ann.remove()
    return count


def _axes_extent(fig: Any) -> Tuple[float, float]:
    r"""(bottom, top) of the union of all axes' tight boxes (titles, labels, panel letters), in figure fraction."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible() and ax.axison]
    height = fig.bbox.height
    return min(b.y0 for b in boxes) / height, max(b.y1 for b in boxes) / height


def _suptitle(fig: Any, text: str, *, x: float = 0.5, ha: str = "center", fontsize: float = 9.0) -> None:
    r"""Figure title placed just above the measured extent of every axes (never overlapping them)."""
    _, top = _axes_extent(fig)
    fig.suptitle(text, x=x, y=top + 0.015, ha=ha, va="bottom", fontsize=fontsize)


def _footnote(fig: Any, text: str, *, width: int = 110) -> None:
    r"""Explanatory note placed just below the measured extent of every axes (math spans never split)."""
    import re
    import textwrap

    protected = re.sub(r"\$[^$]*\$", lambda match: match.group(0).replace(" ", "\x00"), text)
    wrapped = textwrap.fill(protected, width=width, break_on_hyphens=False).replace("\x00", " ")
    bottom, _ = _axes_extent(fig)
    fig.text(0.01, bottom - 0.015, wrapped, ha="left", va="top", fontsize=6.5, color="#444444")


def _style_for(index: int) -> Tuple[str, str]:
    return OKABE_ITO[index % len(OKABE_ITO)], _MARKERS[(index // len(OKABE_ITO)) % len(_MARKERS)]


def _pca2(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""2-D PCA scores and the explained-variance ratios of the two components."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ratio = (s ** 2) / max(float((s ** 2).sum()), 1e-300)
    return Xc @ Vt[:2].T, ratio[:2]


def _set_limits(ax: Any, xy: np.ndarray, pad: float = 0.08) -> None:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    ax.set_xlim(lo[0] - pad * span[0], hi[0] + pad * span[0])
    ax.set_ylim(lo[1] - pad * span[1], hi[1] + pad * span[1])


def _label_subset(ctx: AnalysisContext, token_ids: np.ndarray, labels: Sequence[str], per_field: int,
                  limit: int) -> np.ndarray:
    r"""Indices of up to ``per_field`` most frequent tokens per field (at most ``limit`` in total)."""
    counts = ctx.counts
    chosen: List[int] = []
    for field_name in dict.fromkeys(labels):
        members = np.array([i for i, lab in enumerate(labels) if lab == field_name])
        if counts is not None:
            members = members[np.argsort(-counts[token_ids[members]], kind="stable")]
        chosen.extend(members[:per_field].tolist())
    return np.asarray(chosen[:limit], dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# Per-view figures and tables
# ----------------------------------------------------------------------------------------------

def plot_view_embedding(view: VectorView, ctx: AnalysisContext, battery: Dict[str, Any]) -> List[Path]:
    r"""PCA and UMAP maps of the lexicon words, colored by semantic field.

    Both maps use the prepared rows (centered + normalized for cosine views). UMAP runs through
    :func:`vfe3.viz.figures.umap_embed` (subprocess-isolated, ``seed=ctx.seed``). If UMAP is
    disabled (``ctx.options["umap"] is False``) or raises, only the PCA panel is drawn and the
    reason is written into the figure title and appended to ``battery["notes"]``. Cluster-quality
    numbers are from ``battery["fields"]`` (original space), never from the 2-D maps.
    """
    import matplotlib.pyplot as plt

    rows, labels, _ = _lexicon_rows(view, ctx.lexicon)
    if rows.size < 3:
        _note(ctx, battery, f"embedding map skipped: {rows.size} lexicon word(s) in view {view.name}")
        return []
    apply_style()
    Xp = prepare(view, center=True)[rows]
    pca_xy, ratio = _pca2(Xp)
    umap_xy, umap_error = None, None
    if not ctx.options.get("umap", True):
        umap_error = "UMAP disabled by options"
    else:
        from vfe3.viz import figures as viz_figures
        try:
            umap_xy = np.asarray(viz_figures.umap_embed(Xp, n_neighbors=min(15, rows.size - 1),
                                                        seed=int(ctx.seed)), dtype=np.float64)
        except Exception as exc:
            umap_error = f"UMAP failed: {type(exc).__name__}: {exc}"
    if umap_error is not None:
        _note(ctx, battery, f"{view.name}: {umap_error}")

    fields = list(dict.fromkeys(labels))
    style = {f: _style_for(i) for i, f in enumerate(fields)}
    token_ids = view.ids[rows]
    lab_idx = _label_subset(ctx, token_ids, labels, per_field=2, limit=40)
    panels = [("PCA", pca_xy, f"PC1 ({100 * ratio[0]:.1f}%)", f"PC2 ({100 * ratio[1]:.1f}%)")]
    if umap_xy is not None:
        panels.append(("UMAP", umap_xy, "UMAP 1", "UMAP 2"))
    two = len(panels) == 2
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 if two else 5.0, 3.4), squeeze=False)
    fig.subplots_adjust(left=0.07 if two else 0.1, right=0.74 if two else 0.63, bottom=0.14, top=0.78,
                        wspace=0.3)
    for p, (name, xy, xl, yl) in enumerate(panels):
        ax = axes[0, p]
        for f in fields:
            mask = np.array([lab == f for lab in labels])
            color, marker = style[f]
            ax.scatter(xy[mask, 0], xy[mask, 1], s=12, c=color, marker=marker, alpha=0.85,
                       linewidths=0.3, edgecolors="white", label=f.replace("_", " "))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(name)
        _set_limits(ax, xy)
        panel_label(ax, "ab"[p])
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="center left", bbox_to_anchor=(0.75 if two else 0.64, 0.46),
               ncol=1 if len(fields) <= 16 else 2, fontsize=6.5, markerscale=1.2, handletextpad=0.3,
               columnspacing=0.8, title="semantic field", title_fontsize=7)
    for p, (_, xy, _, _) in enumerate(panels):
        _place_labels(axes[0, p], xy[lab_idx], [_label(ctx, t) for t in token_ids[lab_idx]], obstacles=xy)
    quality = ""
    fields_result = battery.get("fields")
    if _ok(fields_result) and "purity" in fields_result:
        pur, sil = fields_result["purity"], fields_result["silhouette"]
        quality = (f"silhouette {sil['silhouette']:.3f} · kNN purity@{pur['k']} {pur['purity']:.3f} "
                   f"(null {pur['null_mean']:.3f}, z = {pur['z']:.1f}) · original space")
    title = f"{view.label}: {rows.size} lexicon words, {len(fields)} fields"
    if quality:
        title += "\n" + quality
    if umap_error is not None:
        title += "\n" + umap_error[:160]
    _suptitle(fig, title, x=0.02, ha="left", fontsize=8.5)
    return save_figure(fig, ctx, f"{_SECTION}/{view.name}", "map")


_OFFSET_PRIORITY = ("gender", "capital", "comparative", "past", "plural", "nationality",
                    "superlative", "gerund", "adverb", "antonym")


def plot_offsets(view: VectorView, ctx: AnalysisContext, relations: Mapping[str, Sequence[Tuple[int, int]]],
                 *, battery: Optional[Dict[str, Any]] = None, max_panels: int = 6,
                 max_pairs: int = 10) -> List[Path]:
    r"""Mikolov-style offset panels: per relation a 2-D PCA of its words with arrows ``a -> a*``.

    The PCA is fit on that relation's words (prepared rows). Relations with >= 3 pairs in the view
    are ranked by :data:`_OFFSET_PRIORITY` keywords; at most ``max_pairs`` most frequent pairs are
    drawn. With ``battery`` the panel titles carry the offset-consistency z-score.
    """
    import matplotlib.pyplot as plt

    usable = []
    for name, pairs in relations.items():
        ra, rb, kept = _row_pairs(view, pairs)
        if len(kept) >= 3:
            usable.append((name, ra, rb, kept))
    if not usable:
        _note(ctx, battery, f"offset figure skipped: no relation with >= 3 pairs in view {view.name}")
        return []

    def priority(item: Tuple[str, Any, Any, Any]) -> Tuple[int, str]:
        name = item[0].lower()
        hits = [i for i, key in enumerate(_OFFSET_PRIORITY) if key in name]
        return (hits[0] if hits else len(_OFFSET_PRIORITY), name)

    usable = sorted(usable, key=priority)[:max_panels]
    apply_style()
    Xp = prepare(view, center=True)
    ncols = min(3, len(usable))
    nrows = int(math.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.45 * ncols + 0.3, 2.45 * nrows + 0.5), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.86, wspace=0.18, hspace=0.38)
    offsets_result = battery.get("offsets") if battery is not None else None
    for i, (name, ra, rb, kept) in enumerate(usable):
        ax = axes[i // ncols, i % ncols]
        if ctx.counts is not None and len(kept) > max_pairs:
            order = np.argsort(-ctx.counts[[a for a, _ in kept]], kind="stable")[:max_pairs]
        else:
            order = np.arange(min(max_pairs, len(kept)))
        ra_s, rb_s = ra[order], rb[order]
        xy, _ = _pca2(Xp[np.concatenate([ra_s, rb_s])])
        m = ra_s.size
        src, dst = xy[:m], xy[m:]
        for j in range(m):
            ax.annotate("", xy=dst[j], xytext=src[j],
                        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8, shrinkA=2, shrinkB=2,
                                        mutation_scale=7))
        ax.scatter(src[:, 0], src[:, 1], s=14, c=OKABE_ITO[0], zorder=3, label="a")
        ax.scatter(dst[:, 0], dst[:, 1], s=14, c=OKABE_ITO[1], marker="s", zorder=3, label="a*")
        _set_limits(ax, xy, pad=0.18)
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"{name.replace('_', ' ')} ({len(kept)} pairs)"
        if _ok(offsets_result) and name in offsets_result.get("relations", {}):
            rec = offsets_result["relations"][name]
            title += f"\nOCS {rec['ocs']:.2f}, z = {rec['z']:.1f}"
        ax.set_title(title, fontsize=8)
        ids = [kept[j][0] for j in order] + [kept[j][1] for j in order]
        _place_labels(ax, xy, [_label(ctx, t) for t in ids], fontsize=6)
    for j in range(len(usable), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["source a", "target a*"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    _suptitle(fig, f"{view.label}: relation offsets (2-D PCA per relation, prepared rows)")
    return save_figure(fig, ctx, f"{_SECTION}/{view.name}", "offsets")


def write_view_tables(view: VectorView, ctx: AnalysisContext, battery: Dict[str, Any]) -> List[Path]:
    r"""Markdown tables for one view: neighbors, analogy examples, analogy benchmark, offsets, clusters."""
    section = f"{_SECTION}/{view.name}"
    written: List[Path] = []
    score_name = "cosine" if view.metric == "cosine" else "distance"

    neighbors = battery.get("neighbors")
    if isinstance(neighbors, dict) and neighbors.get("status") != "failed":
        rows = [{"word": ctx.display(int(q)),
                 f"nearest neighbors ({score_name})": ", ".join(f"{ctx.display(int(t))} {s:.2f}" for t, s in nn)}
                for q, nn in neighbors.items()]
        written.append(write_table(ctx, section, "neighbors", f"# {view.label}: nearest neighbors\n\n"
                                   + markdown_table(rows)))

    examples = battery.get("analogy_examples")
    if isinstance(examples, list):
        methods = [m for m in ANALOGY_METHODS if any(m in r["methods"] for r in examples)]
        rows = []
        for r in examples:
            row = {"analogy": f"{ctx.display(r['a'])} : {ctx.display(r['a_star'])} :: {ctx.display(r['b'])} : ?",
                   "answer": ctx.display(r["b_star"]), "category": r["category"]}
            for m in methods:
                if not r["covered"]:
                    row[METHOD_LABELS[m]] = "not in view"
                elif m in r["methods"]:
                    d = r["methods"][m]
                    row[METHOD_LABELS[m]] = ", ".join(ctx.display(t) for t in d["topk"]) + f" (rank {d['rank']})"
            rows.append(row)
        written.append(write_table(ctx, section, "analogy_examples",
                                   f"# {view.label}: famous analogies (top-5, query words excluded)\n\n"
                                   + markdown_table(rows)))

    bench = battery.get("analogy_benchmark")
    if _ok(bench):
        rows = []
        groups = [("overall", bench["overall"]), ("semantic", bench["semantic"]),
                  ("syntactic", bench["syntactic"])] + sorted(bench["categories"].items())
        for name, g in groups:
            row = {"group": name, "n / total": f"{g['n']} / {g['n_total']}",
                   "coverage": g.get("coverage"), "pairs": g.get("n_pairs")}
            for m in bench["methods"]:
                row[METHOD_LABELS[m]] = g["methods"][m]["acc"]["mean"] if g["n"] else None
            if g["n"] and "3cosadd" in g["methods"]:
                acc = g["methods"]["3cosadd"]["acc"]
                row["3CosAdd 95% CI"] = (f"[{acc['ci_lo']:.3f}, {acc['ci_hi']:.3f}]"
                                         if _is_number(acc["ci_lo"]) and _is_number(acc["ci_hi"]) else None)
                row["3CosAdd hits@5"] = g["methods"]["3cosadd"]["hits5"]["mean"]
                row["3CosAdd MRR"] = g["methods"]["3cosadd"]["mrr"]["mean"]
            if g["n"]:
                row["honest acc"] = g["honest"]["acc"]["mean"]
                row["honest top-1 = b"] = g["honest"]["top1_is_b"]
            rows.append(row)
        columns = ["group", "n / total", "coverage", "pairs"] + [METHOD_LABELS[m] for m in bench["methods"]]
        columns += [c for c in ("3CosAdd 95% CI", "3CosAdd hits@5", "3CosAdd MRR", "honest acc", "honest top-1 = b")
                    if any(c in r for r in rows)]
        source = ("questions in the full Google benchmark" if bench.get("totals_source") == "full benchmark"
                  else "questions resolved to candidate tokens")
        written.append(write_table(
            ctx, section, "analogy_benchmark",
            f"# {view.label}: analogy benchmark (accuracy; query words excluded unless 'honest')\n\n"
            f"n: questions covered by this view; total: {source}; coverage = n / total; pairs: distinct word "
            f"pairs forming the covered questions. {bench.get('resampling_note', '')}.\n\n"
            + markdown_table(rows, columns)))

    offsets = battery.get("offsets")
    if _ok(offsets):
        rows = [{"relation": name, "pairs": r["n_pairs"], "OCS": r["ocs"], "OCS re-paired": r["null_mean"],
                 "z": r["z"], "p": r["p"], "LOO proj": r["loo_proj_mean"], "re-paired": r["null_loo_proj_mean"],
                 "random pairs": r.get("random_loo_proj_mean"), "random pairs sd": r.get("random_loo_proj_sd"),
                 "top-PC share": r["top_pc_share"], "top-PC re-paired": r["null_top_pc_share_mean"]}
                for name, r in offsets["relations"].items()]
        written.append(write_table(
            ctx, section, "offsets",
            f"# {view.label}: relation-offset consistency\n\n"
            "re-paired: partners permuted within the relation by random cyclic derangements (Fournier, Dupoux & "
            "Dunbar 2020; keeps the two word sets, hence their group offset). LOO proj: projection of each "
            "offset on the unit mean of the other offsets; random pairs: the same projection for as many random "
            "pairs of the view's words (mean and sd; the baseline of Park, Choe & Veitch 2024, Fig. 2).\n\n"
            + markdown_table(rows)))

    clusters = battery.get("clusters")
    if _ok(clusters):
        rows = [{"cluster": c["cluster"], "size": c["size"],
                 "central members": ", ".join(ctx.display(t) for t in c["central_ids"])}
                for c in clusters["clusters"]]
        written.append(write_table(ctx, section, "clusters",
                                   f"# {view.label}: k-means clusters of the {clusters['n']} most frequent "
                                   f"words (k = {clusters['k']}, seed {clusters['seed']})\n\n" + markdown_table(rows)))
    return written


# ----------------------------------------------------------------------------------------------
# Cross-view comparison figures
# ----------------------------------------------------------------------------------------------

_KEY_VIEW_PRIORITY = ("decode_W_causal", "decode_natural_causal", "decode_W", "decode_mu",
                      "mu_raw", "mu_sigma_units", "xi_ref", "w2_chart")
# Concept-direction panels: channel of each known view and the preference order within channels.
_DIRECTION_CHANNELS: Dict[str, str] = {
    "decode_W_causal": "decode", "decode_natural_causal": "decode", "decode_W": "decode", "decode_mu": "decode",
    "phi_log": "phi", "phi_raw": "phi", "xi_ref": "xi", "mu_raw": "mu", "mu_sigma_units": "mu",
}
_DIRECTION_PREFERENCE: Tuple[str, ...] = ("decode_W_causal", "decode_natural_causal", "decode_W", "decode_mu",
                                          "phi_log", "phi_raw", "xi_ref", "mu_raw", "mu_sigma_units")

Results = Mapping[str, Dict[str, Any]]
Views = Mapping[str, VectorView]


def _get(result: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(result, dict) or key not in result:
            return None
        result = result[key]
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


def _headline_view(results: Results, ctx: AnalysisContext) -> Optional[str]:
    requested = ctx.options.get("headline_view")
    if requested in results:
        return requested
    for name in _KEY_VIEW_PRIORITY:
        if name in results:
            return name
    return next(iter(results), None)


def _dot_row_axes(ax: Any, labels: Sequence[str]) -> None:
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)


def _panel_letters(axes: Sequence[Any]) -> None:
    if len(axes) > 1:
        for ax, letter in zip(axes, "abcdefgh"):
            panel_label(ax, letter)


def _compare_analogy_categories(results: Results, views: Views, ctx: AnalysisContext,
                                order: List[str]) -> Optional[Any]:
    r"""Heatmap group x view of 3CosAdd accuracy, each cell annotated ``acc`` over ``(ONLY-B)``."""
    import matplotlib.pyplot as plt

    usable = [v for v in order if _ok(results[v].get("analogy_benchmark"))]
    cats: List[str] = []
    for v in usable:
        for c in results[v]["analogy_benchmark"]["categories"]:
            if c not in cats:
                cats.append(c)
    if not cats:
        return None
    cats = sorted(cats, key=lambda c: (str(c).startswith("gram"), c))
    groups = [("overall", ("overall",)), ("semantic", ("semantic",)), ("syntactic", ("syntactic",))]
    groups += [(c, ("categories", c)) for c in cats]
    headline = _headline_view(results, ctx)
    ref = headline if headline in usable else usable[0]
    A = np.full((len(groups), len(usable)), np.nan)                    # 3CosAdd accuracy
    B = np.full((len(groups), len(usable)), np.nan)                    # ONLY-B accuracy
    for c_index, v in enumerate(usable):
        for r_index, (_, path) in enumerate(groups):
            g = _get(results[v]["analogy_benchmark"], *path)
            if not g or not g.get("n"):
                continue
            for M, method in ((A, "3cosadd"), (B, "only_b")):
                value = _get(g, "methods", method, "acc", "mean")
                if _is_number(value):
                    M[r_index, c_index] = value
    labels = []
    for name, path in groups:
        g = _get(results[ref]["analogy_benchmark"], *path) or {}
        labels.append(f"{name} ({g.get('n', 0)}/{g.get('n_total', 0)})")
    fig, ax = plt.subplots(figsize=(0.62 * len(usable) + 3.4, 0.34 * len(groups) + 1.9))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([views[v].label for v in usable], rotation=40, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.axhline(2.5, color="white", lw=2.5)
    for (r, c), value in np.ndenumerate(A):
        if np.isfinite(value):
            text = f"{value:.2f}" + (f"\n({B[r, c]:.2f})" if np.isfinite(B[r, c]) else "")
            ax.text(c, r, text, ha="center", va="center", fontsize=5.5, linespacing=1.0,
                    color="white" if value < 0.6 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="3CosAdd accuracy")
    _suptitle(fig, "Google analogy accuracy: 3CosAdd (ONLY-B)")
    _footnote(fig, "Cell: 3CosAdd accuracy with the query words excluded, in parentheses the ONLY-B baseline "
                   "(nearest neighbor of b; Linzen 2016). Row labels: questions covered by "
                   f"{views[ref].label} / questions in the benchmark. Grey: no covered question.", width=90)
    return fig


def _compare_analogy_methods(results: Results, views: Views, ctx: AnalysisContext,
                             order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    usable = [v for v in order if _ok(results[v].get("analogy_benchmark"))]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fig.subplots_adjust(left=0.09, right=0.72, bottom=0.16, top=0.92)
    width = 0.8 / len(usable)
    for i, v in enumerate(usable):
        color, _ = _style_for(i)
        overall = results[v]["analogy_benchmark"]["overall"]
        for m_index, m in enumerate(ANALOGY_METHODS):
            x = m_index - 0.4 + width * (i + 0.5)
            acc = _get(overall, "methods", m, "acc")
            if acc is None:
                ax.text(x, 0.01, "n/a", rotation=90, ha="center", va="bottom", fontsize=6, color=color)
                continue
            ax.bar(x, acc["mean"], width=width * 0.92, color=color, label=views[v].label)
            # a percentile interval of the pair-bootstrap ratio need not contain the point value
            ax.errorbar(x, acc["mean"], yerr=[[max(acc["mean"] - acc["ci_lo"], 0.0)],
                                              [max(acc["ci_hi"] - acc["mean"], 0.0)]],
                        color="#333333", lw=0.7, capsize=1.5)
    ax.set_xticks(range(len(ANALOGY_METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in ANALOGY_METHODS])
    ax.set_ylabel("overall accuracy (95% CI)")
    ax.set_ylim(0, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(0.73, 0.5), fontsize=7)
    bench = results[usable[0]]["analogy_benchmark"]
    n, n_total = bench["n_covered"], bench["n_total"]
    share = f" ({100.0 * n / n_total:.0f}%)" if n_total else ""
    _suptitle(fig, "Analogy accuracy by method (query words excluded)")
    _footnote(fig, f"{n} of {n_total} benchmark questions covered{share} for {views[usable[0]].label}; 95% CIs "
                   "resample word pairs. n/a: 3CosMul and PairDirection are cosine-only and are not defined for "
                   "Euclidean views.", width=100)
    return fig


def _compare_similarity(results: Results, views: Views, ctx: AnalysisContext,
                        order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    benches: List[str] = []
    for v in order:
        for b, r in (_get(results[v], "similarity") or {}).items():
            if _ok(r) and "rho" in r and b not in benches:
                benches.append(b)
    if not benches:
        return None
    fig, axes = plt.subplots(1, len(benches), figsize=(2.6 * len(benches) + 1.6, 0.3 * len(order) + 1.2),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.3 if len(benches) == 1 else 0.2, right=0.9, bottom=0.25, top=0.85, wspace=0.25)
    for b_index, b in enumerate(benches):
        ax = axes[0, b_index]
        for i, v in enumerate(order):
            r = _get(results[v], "similarity", b)
            if not (_ok(r) and "rho" in r):
                continue
            color, marker = _style_for(i)
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, lw=1.0)
            ax.plot(r["rho"], i, marker, color=color, ms=4.5)
            ax.text(1.02, i, f"n={r['n']}", transform=ax.get_yaxis_transform(), fontsize=6, va="center",
                    ha="left", color="#555555", clip_on=False)
        ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xlabel(r"Spearman $\rho$ (95% CI)")
        ax.set_title(b, fontsize=9)
    _panel_letters(axes[0])
    _dot_row_axes(axes[0, 0], [views[v].label for v in order])
    _suptitle(fig, "Word-similarity benchmarks (pairs covered by each view)")
    return fig


def _compare_fields(results: Results, views: Views, ctx: AnalysisContext, order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    usable = [v for v in order if _ok(results[v].get("fields")) and "purity" in results[v]["fields"]]
    if not usable:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 0.3 * len(usable) + 1.3), sharey=True)
    fig.subplots_adjust(left=0.26, right=0.97, bottom=0.25, top=0.85, wspace=0.12)
    for i, v in enumerate(usable):
        color, marker = _style_for(i)
        pur = results[v]["fields"]["purity"]
        sil = results[v]["fields"]["silhouette"]
        if _is_number(pur["null_std"]):
            lo, hi = pur["null_mean"] - 2 * pur["null_std"], pur["null_mean"] + 2 * pur["null_std"]
            axes[0].fill_betweenx([i - 0.3, i + 0.3], lo, hi, color="#cccccc", lw=0, zorder=0)
        axes[0].plot(pur["chance"], i, "|", color="#555555", ms=9, mew=1.0)
        axes[0].plot(pur["purity"], i, marker, color=color, ms=5)
        axes[1].plot(sil["silhouette"], i, marker, color=color, ms=5)
    k = results[usable[0]]["fields"]["purity"]["k"]
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_xlabel(f"kNN field purity@{k}")
    axes[0].set_title("kNN purity", fontsize=9)
    axes[1].axvline(0, color="#bbbbbb", lw=0.6)
    axes[1].set_xlabel("silhouette")
    axes[1].set_title("silhouette", fontsize=9)
    _dot_row_axes(axes[0], [views[v].label for v in usable])
    _panel_letters(axes)
    _suptitle(fig, "Semantic-field structure (original space)")
    _footnote(fig, "Grey band: label-permutation null mean ± 2 sd; tick: exact chance purity. Silhouette uses "
                   "the view's metric on prepared rows; field labels are external (no fitted clusters).",
              width=105)
    return fig


def _compare_geometry(results: Results, views: Views, ctx: AnalysisContext, order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    usable = [v for v in order if _ok(results[v].get("geometry"))]
    if not usable:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 0.3 * len(usable) + 1.3), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.25, top=0.85, wspace=0.18)
    cosines: List[float] = [0.0]
    dims: List[float] = [1.0]
    for i, v in enumerate(usable):
        color, marker = _style_for(i)
        g = results[v]["geometry"]
        if _is_number(g.get("effective_rank")):
            axes[0].plot(g["effective_rank"] / g["dim"], i, marker, color=color, ms=5)
        if _is_number(g.get("isoscore")):
            axes[1].plot(g["isoscore"], i, marker, color=color, ms=5)
        for key, fill in (("anisotropy_raw", color), ("anisotropy_centered", "none")):
            if _is_number(g.get(key)):
                axes[2].plot(g[key], i, marker, color=color, mfc=fill, ms=5)
                cosines.append(float(g[key]))
        if _ok(g.get("twonn")):
            for key, fill in (("mle", color), ("fit", "none")):
                if _is_number(g["twonn"][key]):
                    axes[3].plot(g["twonn"][key], i, marker, color=color, mfc=fill, ms=5)
                    dims.append(float(g["twonn"][key]))
    axes[0].set_xlim(0.0, 1.02)
    axes[1].set_xlim(0.0, 1.02)
    axes[2].set_xlim(min(-0.05, min(cosines) - 0.05), max(0.05, max(cosines) + 0.05))
    axes[3].set_xlim(0.0, 1.1 * max(dims))
    axes[2].axvline(0, color="#bbbbbb", lw=0.6)
    for ax, title in zip(axes, ("eff. rank / D", "IsoScore", "mean cosine", "TwoNN dim.")):
        ax.set_title(title, fontsize=9)
    _dot_row_axes(axes[0], [f"{views[v].label} (D={views[v].dim})" for v in usable])
    _panel_letters(axes)
    _suptitle(fig, "Global geometry of the most frequent rows")
    _footnote(fig, "Mean cosine: filled raw, open centered rows (Ethayarajh 2019). TwoNN: filled maximum "
                   "likelihood, open linear fit (Facco et al. 2017). IsoScore: Rudman et al. (2022).", width=115)
    return fig


def _heatmap(ax: Any, M: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], *,
             vmax: float, annotate: bool, fmt: str = "{:.1f}") -> Any:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    if annotate:
        for (r, c), value in np.ndenumerate(M):
            if np.isfinite(value):
                ax.text(c, r, fmt.format(value), ha="center", va="center", fontsize=5.5,
                        color="white" if abs(value) > 0.6 * vmax else "#222222")
    return im


def _compare_offsets(results: Results, views: Views, ctx: AnalysisContext, order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    rels: List[str] = []
    for v in order:
        for name in (_get(results[v], "offsets", "relations") or {}):
            if name not in rels:
                rels.append(name)
    if not rels:
        return None
    Z = np.full((len(rels), len(order)), np.nan)
    for c, v in enumerate(order):
        for r, name in enumerate(rels):
            rec = _get(results[v], "offsets", "relations", name)
            if rec is not None and _is_number(rec["z"]):
                Z[r, c] = rec["z"]
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    fig, ax = plt.subplots(figsize=(0.55 * len(order) + 2.8, 0.3 * len(rels) + 1.5))
    im = _heatmap(ax, Z, [r.replace("_", " ") for r in rels], [views[v].label for v in order],
                  vmax=max(vmax, 1e-9), annotate=Z.size <= 300)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="OCS z-score")
    _suptitle(fig, "Relation-offset consistency")
    _footnote(fig, "z of the offset concentration score against partners re-paired by random cyclic "
                   "derangements (Fournier et al. 2020); grey: fewer than 3 pairs in the view.", width=70)
    return fig


def _direction_panels(usable: Sequence[str], views: Views, headline: Optional[str], limit: int = 4) -> List[str]:
    r"""Up to ``limit`` views for the whitened concept-direction panels, one per channel.

    Views related by a global invertible linear map ``x -> A x`` have identical whitened cosines
    (``(A C A^T)^{-1/2} A d`` is an orthogonal image of ``C^{-1/2} d``), e.g. ``decode_W`` and
    ``decode_W_causal`` or ``mu_raw`` and ``mu_sigma_units``, so at most one view per channel of
    :data:`_DIRECTION_CHANNELS` is drawn: the headline view first, then decode, phi (``phi_log``
    before ``phi_raw``), ``xi_ref``, mu, then any other channel (``VectorView.group``) in view order.
    """
    ordered = ([headline] if headline in usable else []) + [v for v in _DIRECTION_PREFERENCE if v in usable]
    ordered += [v for v in usable if v not in ordered]
    chosen: List[str] = []
    channels = set()
    for v in ordered:
        channel = _DIRECTION_CHANNELS.get(v, views[v].group or v)
        if v in chosen or channel in channels:
            continue
        chosen.append(v)
        channels.add(channel)
        if len(chosen) == limit:
            break
    return chosen


def _compare_directions(results: Results, views: Views, ctx: AnalysisContext,
                        order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    usable = [v for v in order if _ok(_get(results[v], "directions_whitened"))
              and "cosine" in results[v]["directions_whitened"]]
    if not usable:
        return None
    key = _direction_panels(usable, views, _headline_view(results, ctx))
    fig, axes = plt.subplots(1, len(key), figsize=(2.3 * len(key) + 1.5, 3.0), squeeze=False)
    fig.subplots_adjust(left=0.16 if len(key) > 1 else 0.3, right=0.88, bottom=0.3, top=0.88, wspace=0.12)
    im = None
    for i, v in enumerate(key):
        d = results[v]["directions_whitened"]
        names = [n.replace("_", " ") for n in d["relations"]]
        C = np.asarray(d["cosine"], dtype=np.float64)
        im = _heatmap(axes[0, i], C, names if i == 0 else [""] * len(names), names, vmax=1.0,
                      annotate=len(names) <= 12, fmt="{:.2f}")
        axes[0, i].set_title(f"{views[v].label}\nrank {d.get('whitening_rank')}/{d['dim']}", fontsize=8)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02, label="cosine")
    _suptitle(fig, r"Concept-direction cosines under the causal inner product $\mathrm{Cov}^{-1}$")
    _footnote(fig, "Mean offset direction per relation, compared after whitening by the covariance of the "
                   "view's rows (Park, Choe & Veitch 2024); rank = number of covariance eigenvalues kept. One view "
                   "per channel: views related by a global linear map (decode W raw and causal, μ raw and in D0 "
                   "units) have identical whitened cosines.", width=110)
    return fig


def _compare_scree(results: Results, views: Views, ctx: AnalysisContext, order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    usable = [v for v in order if isinstance(_get(results[v], "geometry", "pca_spectrum"), (list, np.ndarray))]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    fig.subplots_adjust(left=0.12, right=0.66, bottom=0.16, top=0.9)
    for i, v in enumerate(usable):
        color, marker = _style_for(i)
        spec = np.asarray(results[v]["geometry"]["pca_spectrum"], dtype=np.float64)
        spec = spec[spec > 0]
        ax.plot(np.arange(1, spec.size + 1), spec, color=color, marker=marker, ms=2.5, lw=1.0,
                label=views[v].label, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained-variance ratio")
    fig.legend(loc="center left", bbox_to_anchor=(0.67, 0.5), fontsize=7)
    _suptitle(fig, "PCA spectra of the centered raw rows (first 50 components)")
    return fig


def _compare_frequency(results: Results, views: Views, ctx: AnalysisContext,
                       order: List[str]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    usable = [v for v in order if _ok(_get(results[v], "geometry", "frequency_probe"))]
    if not usable:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 0.3 * len(usable) + 1.3), sharey=True)
    fig.subplots_adjust(left=0.26, right=0.97, bottom=0.25, top=0.85, wspace=0.12)
    for i, v in enumerate(usable):
        color, marker = _style_for(i)
        g = results[v]["geometry"]
        probe = g["frequency_probe"]
        axes[0].plot(probe["r2_mean"], i, marker, color=color, ms=5)
        axes[0].plot(probe["control_r2_mean"], i, marker, mfc="none", color=color, ms=5)
        pcs = g.get("pc_log_count_spearman")
        if isinstance(pcs, list) and pcs:
            r = pcs[0]                                              # PC sign is arbitrary: fold to |rho|
            lo, hi = sorted((abs(r["ci_lo"]), abs(r["ci_hi"])))
            if r["ci_lo"] < 0.0 < r["ci_hi"]:
                lo, hi = 0.0, max(abs(r["ci_lo"]), abs(r["ci_hi"]))
            axes[1].plot([lo, hi], [i, i], color=color, lw=1.0)
            axes[1].plot(abs(r["rho"]), i, marker, color=color, ms=5)
        norm = g.get("norm_log_count_spearman")
        if _ok(norm) and "rho" in norm:
            axes[1].plot(norm["rho"], i, "x", color=color, ms=5)
    axes[0].axvline(0, color="#bbbbbb", lw=0.6)
    axes[1].axvline(0, color="#bbbbbb", lw=0.6)
    axes[1].set_xlim(-1.05, 1.05)
    axes[0].set_xlabel(r"cross-validated $R^2$")
    axes[0].set_title("ridge probe of log count", fontsize=9)
    axes[1].set_xlabel(r"Spearman $\rho$ with log count")
    axes[1].set_title("PC1 and norm", fontsize=9)
    _dot_row_axes(axes[0], [views[v].label for v in usable])
    _panel_letters(axes)
    _suptitle(fig, "Frequency confound")
    _footnote(fig, "(a) filled: real target, open: shuffled-target control (same folds). (b) dot with 95% CI: "
                   r"$|\rho|$ of the first principal-component score (its sign is arbitrary); cross: signed "
                   r"$\rho$ of the row norm $\Vert x\Vert$.", width=105)
    return fig


def _compare_projections(results: Results, views: Views, ctx: AnalysisContext, headline: str,
                         max_words: int = 24) -> Optional[Any]:
    import matplotlib.pyplot as plt

    projections = _get(results[headline], "projections") or {}
    axes_ok = [(name, p) for name, p in projections.items() if _ok(p) and p.get("n_words", 0) > 0][:4]
    if not axes_ok:
        return None
    rows = min(max_words, max(p["n_words"] for _, p in axes_ok))
    fig, axes = plt.subplots(1, len(axes_ok), figsize=(1.9 * len(axes_ok) + 1.0, 0.16 * rows + 1.3),
                             squeeze=False)
    fig.subplots_adjust(left=0.14 if len(axes_ok) > 1 else 0.3, right=0.97, bottom=0.2, top=0.86, wspace=0.9)
    for i, (name, p) in enumerate(axes_ok):
        ax = axes[0, i]
        words = p["words"]
        if len(words) > max_words:
            half = max_words // 2
            words = words[:half] + words[-half:]
        ax.scatter([s for _, s in words], range(len(words)), s=10, c=OKABE_ITO[5], zorder=3)
        pos = [w["pos_score"] for w in p["pair_words"]]
        neg = [w["neg_score"] for w in p["pair_words"]]
        ax.axvspan(min(neg), max(neg), color=OKABE_ITO[3], alpha=0.15, lw=0)
        ax.axvspan(min(pos), max(pos), color=OKABE_ITO[0], alpha=0.15, lw=0)
        _dot_row_axes(ax, [_label(ctx, t) for t, _ in words])
        ax.tick_params(axis="y", labelsize=6)
        ax.axvline(0, color="#bbbbbb", lw=0.6)
        loo = p.get("loo_pair_accuracy")
        ax.set_title(f"{name.replace('_', ' ')}" + (f" (LOO {loo:.2f})" if loo is not None else ""), fontsize=8)
        ax.set_xlabel(f"{p['neg_label']} $\\leftrightarrow$ {p['pos_label']}", fontsize=7)
    _panel_letters(axes[0])
    _suptitle(fig, f"{views[headline].label}: semantic projection")
    _footnote(fig, "Shaded: score range of the negative (vermillion) and positive (blue) pole words; LOO: "
                   "fraction of pole pairs ordered correctly by the axis built from the other pairs "
                   "(Grand et al. 2022).", width=60 if len(axes_ok) == 1 else 100)
    return fig


def plot_views_comparison(results: Dict[str, Dict[str, Any]], views: Dict[str, VectorView],
                          ctx: AnalysisContext) -> List[Path]:
    r"""Cross-view comparison figures under ``figures/embeddings/compare/``.

    (a) ``analogy_categories`` -- heatmap (overall, semantic, syntactic and every Google category)
    x view of 3CosAdd accuracy, each cell annotated with the ONLY-B baseline; (b)
    ``analogy_methods`` -- overall accuracy per method and view; (c)
    ``similarity`` -- benchmark Spearman rho with CI; (d) ``fields`` -- kNN purity with the
    permutation null band and silhouette; (e) ``geometry`` -- effective rank / D, IsoScore,
    anisotropy, TwoNN; (f) ``offsets`` -- OCS z heatmap relation x view; (g) ``directions`` --
    whitened concept-direction cosines for up to four views of distinct channels
    (:func:`_direction_panels`); (h) ``scree``; (i) ``frequency``
    -- probe R^2 vs control and rho(PC1, log count); (j) ``projections_<view>`` -- semantic
    projections for the headline view (``ctx.options["headline_view"]`` or the first key view).
    A figure without data is skipped and the reason appended to ``ctx.cache["vectors_notes"]``; a
    figure whose builder raises is noted there and recorded as ``{"step": "compare.<name>", "error"}``
    in ``ctx.cache["vectors_failed_steps"]``, and the other figures are still drawn.
    """
    import matplotlib.pyplot as plt

    apply_style()
    order = [name for name in results if name in views]
    if not order:
        _note(ctx, None, "view comparison skipped: no results")
        return []
    headline = _headline_view(results, ctx)
    builders: List[Tuple[str, Callable[[], Optional[Any]]]] = [
        (name, lambda fn=fn: fn(results, views, ctx, order)) for name, fn in (
            ("analogy_categories", _compare_analogy_categories),
            ("analogy_methods", _compare_analogy_methods),
            ("similarity", _compare_similarity),
            ("fields", _compare_fields),
            ("geometry", _compare_geometry),
            ("offsets", _compare_offsets),
            ("directions", _compare_directions),
            ("scree", _compare_scree),
            ("frequency", _compare_frequency))]
    if headline is not None:
        builders.append((f"projections_{headline}", lambda: _compare_projections(results, views, ctx, headline)))
    written: List[Path] = []
    for name, build in builders:
        open_before = set(plt.get_fignums())
        try:
            fig = build()
            if fig is None:
                _note(ctx, None, f"comparison figure '{name}' skipped: no usable results")
                continue
            written.extend(save_figure(fig, ctx, _COMPARE_SECTION, name))
        except Exception as exc:  # recorded: note + failed step; the other figures still run
            for number in set(plt.get_fignums()) - open_before:
                plt.close(number)
            error = f"{type(exc).__name__}: {exc}"
            ctx.cache.setdefault("vectors_failed_steps", []).append({"step": f"compare.{name}", "error": error})
            _note(ctx, None, f"comparison figure '{name}' failed: {error}")
    return written
