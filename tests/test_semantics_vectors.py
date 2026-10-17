r"""Tests for :mod:`vfe3.semantics.stats` and :mod:`vfe3.semantics.vectors` on tiny synthetic data.

Every expectation is either an exact identity (planted parallelograms, orthonormal spectra,
brute-force references) or a large-margin statistical separation (separable clusters vs permuted
labels) with fixed seeds.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from vfe3.semantics import stats, vectors
from vfe3.semantics.common import OKABE_ITO, AnalogyQuestion, AnalysisContext, VectorView, to_jsonable


# ----------------------------------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------------------------------

class _FakeVocab:
    def display(self, token_id: int) -> str:
        return f"w{int(token_id)}"


def _ctx(tmp_path: Path, n: int, **overrides: Any) -> AnalysisContext:
    fields: Dict[str, Any] = dict(
        out_dir=tmp_path, figures_dir=tmp_path / "figures", tables_dir=tmp_path / "tables",
        vocab=_FakeVocab(), candidate_ids=np.arange(n, dtype=np.int64),
        counts=(np.arange(n, 0, -1) * 10).astype(np.int64), lexicon={}, relations={}, analogies=[],
        curated_analogies=[], similarity={}, hypernyms=[], axes={}, probe_ids=[], ambiguous_ids=[],
        coverage={}, seed=0, options={"umap": True})
    fields.update(overrides)
    return AnalysisContext(**fields)


def _centered_orthonormal(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    r"""(n, k) orthonormal columns that each sum to zero (so centering leaves them unchanged)."""
    A = rng.standard_normal((n, k))
    A -= A.mean(axis=0)
    Q, _ = np.linalg.qr(A)
    return Q


def _orthonormal(D: int, k: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q


def _brute_ranks(Xp: np.ndarray, q: np.ndarray, method: str, metric: str, exclude: bool) -> tuple:
    r"""Reference rank of b* and the set of top-1 rows (exact ties kept) by explicit formulas."""
    a, s, b, t = (Xp[i] for i in q)
    scores = np.empty(Xp.shape[0])
    for r, x in enumerate(Xp):
        if metric == "cosine":
            cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            if method == "3cosadd":
                scores[r] = cos(x, s) - cos(x, a) + cos(x, b)
            elif method == "3cosmul":
                c = lambda u, v: (cos(u, v) + 1) / 2
                scores[r] = c(x, s) * c(x, b) / (c(x, a) + 1e-3)
            elif method == "pairdir":
                scores[r] = cos(x - b, s - a) if np.linalg.norm(x - b) > 1e-9 else 0.0
            elif method == "only_b":
                scores[r] = cos(x, b)
            elif method == "ignore_a":
                scores[r] = cos(x, s) + cos(x, b)
        else:
            if method == "3cosadd":
                scores[r] = -np.sum((x - (s - a + b)) ** 2)
            elif method == "only_b":
                scores[r] = -np.sum((x - b) ** 2)
            elif method == "ignore_a":
                scores[r] = -np.sum((x - s) ** 2) - np.sum((x - b) ** 2)
    if exclude:
        scores[q[:3]] = -np.inf
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    return 1 + int((scores > scores[q[3]]).sum()), set((best + 1000).tolist())


def _random_questions(n: int, count: int, rng: np.random.Generator, category: str = "cat") -> List[AnalogyQuestion]:
    out = []
    for _ in range(count):
        a, s, b, t = rng.choice(n, size=4, replace=False)
        out.append(AnalogyQuestion(category, int(a), int(s), int(b), int(t)))
    return out


# ----------------------------------------------------------------------------------------------
# stats.py
# ----------------------------------------------------------------------------------------------

def test_spectrum_summaries_are_exact_on_planted_spectra() -> None:
    rng = np.random.default_rng(0)
    n, D = 60, 6
    Q = _centered_orthonormal(n, 4, rng)
    V = _orthonormal(D, 4, rng)
    equal = Q @ np.diag([3.0, 3.0, 3.0, 3.0]) @ V.T                    # 4 equal singular values
    assert stats.effective_rank(equal) == pytest.approx(4.0, abs=1e-6)
    assert stats.participation_ratio(equal) == pytest.approx(4.0, abs=1e-6)
    spectrum = stats.pca_spectrum(equal, k=6)
    np.testing.assert_allclose(spectrum, [0.25, 0.25, 0.25, 0.25, 0.0, 0.0], atol=1e-10)
    # IsoScore by the closed form for variance vector (1,1,1,1,0,0) in d = 6
    d, root = 6, math.sqrt(6)
    v_hat = root * np.array([1, 1, 1, 1, 0, 0]) / 2.0
    delta = np.linalg.norm(v_hat - 1) / math.sqrt(2 * (d - root))
    k = (d - delta ** 2 * (d - root)) ** 2 / d
    assert stats.isoscore(equal) == pytest.approx((k - 1) / (d - 1), abs=1e-9)
    rank_one = Q[:, :1] @ V[:, :1].T * 5.0
    assert stats.effective_rank(rank_one) == pytest.approx(1.0, abs=1e-6)
    assert stats.participation_ratio(rank_one) == pytest.approx(1.0, abs=1e-6)
    assert stats.isoscore(rank_one) == pytest.approx(0.0, abs=1e-6)
    isotropic = _centered_orthonormal(n, 5, rng) @ _orthonormal(5, 5, rng).T * 2.0
    assert stats.isoscore(isotropic) == pytest.approx(1.0, abs=1e-9)
    assert stats.isoscore(isotropic + 7.0) == pytest.approx(1.0, abs=1e-9)   # translation-free


def test_isoscore_separates_isotropic_from_anisotropic_gaussians() -> None:
    rng = np.random.default_rng(1)
    iso = rng.standard_normal((4000, 8))
    aniso = iso * np.array([10.0, 1, 1, 1, 0.1, 0.1, 0.1, 0.1])
    assert stats.isoscore(iso) > 0.95
    assert stats.isoscore(aniso) < 0.2


def test_anisotropy_is_the_exact_all_pairs_mean_cosine() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((25, 4)) + np.array([1.0, 0.0, 0.5, 0.0])
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = Z @ Z.T
    brute = (C.sum() - np.trace(C)) / (25 * 24)
    assert stats.anisotropy(X) == pytest.approx(brute, abs=1e-12)
    shifted = rng.standard_normal((500, 6)) + 10.0
    assert stats.anisotropy(shifted) > 0.9
    assert abs(stats.anisotropy(shifted, center=True)) < 0.05
    assert stats.mean_vector_ratio(np.ones((5, 3))) == pytest.approx(1.0)
    sym = rng.standard_normal((10, 3))
    assert stats.mean_vector_ratio(np.vstack([sym, -sym])) == pytest.approx(0.0, abs=1e-12)


def test_rogue_dimensions_find_the_offset_dimension() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 8))
    X[:, 3] += 20.0
    rogue = stats.rogue_dimensions(X)
    assert rogue["top_dims"][0] == 3
    assert rogue["top_share"][0] > 0.95
    assert rogue["mean_cosine"] == pytest.approx(stats.anisotropy(X), abs=1e-12)
    assert abs(rogue["mean_cosine_zscored"]) < 0.05


def test_twonn_recovers_known_intrinsic_dimension() -> None:
    rng = np.random.default_rng(4)
    plane = rng.uniform(size=(2000, 2)) @ rng.standard_normal((2, 10))
    five = rng.standard_normal((2000, 5)) @ rng.standard_normal((5, 12))
    r2 = stats.twonn_dimension(plane, rng)
    r5 = stats.twonn_dimension(five, rng)
    assert r2["mle"] == pytest.approx(2.0, abs=0.25) and r2["fit"] == pytest.approx(2.0, abs=0.25)
    assert r5["mle"] == pytest.approx(5.0, abs=0.8) and r5["fit"] == pytest.approx(5.0, abs=0.8)
    with_dups = np.vstack([plane[:300], plane[:10]])
    assert stats.twonn_dimension(with_dups, rng)["n_duplicates_dropped"] == 10
    capped = stats.twonn_dimension(plane, rng, max_points=500)
    assert capped["n"] == 500


def test_knn_matches_brute_force_for_both_metrics() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((37, 5))
    for metric in ("cosine", "euclidean"):
        idx, score = stats.knn(X, 4, metric, chunk=8)
        if metric == "cosine":
            Z = X / np.linalg.norm(X, axis=1, keepdims=True)
            S = Z @ Z.T
            np.fill_diagonal(S, -np.inf)
            ref = np.argsort(-S, axis=1)[:, :4]
            np.testing.assert_allclose(score, np.take_along_axis(S, ref, axis=1), atol=1e-12)
        else:
            D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(D2, np.inf)
            ref = np.argsort(D2, axis=1)[:, :4]
            np.testing.assert_allclose(score, np.sqrt(np.take_along_axis(D2, ref, axis=1)), atol=1e-12)
        np.testing.assert_array_equal(idx, ref)


def _song_hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))
    one = np.ones(n)
    return float((np.trace(K @ L) + (one @ K @ one) * (one @ L @ one) / ((n - 1) * (n - 2))
                  - 2.0 / (n - 2) * (one @ K @ L @ one)) / (n * (n - 3)))


def test_linear_cka_invariances_and_debiased_reference() -> None:
    rng = np.random.default_rng(6)
    X = rng.standard_normal((40, 5))
    Qm = _orthonormal(5, 5, rng)
    Y = 3.0 * X @ Qm + 7.0
    assert stats.linear_cka(X, Y) == pytest.approx(1.0, abs=1e-10)
    assert stats.linear_cka(X, Y, debiased=True) == pytest.approx(1.0, abs=1e-10)
    W = rng.standard_normal((40, 6))
    Xc, Wc = X - X.mean(0), W - W.mean(0)
    K, L = Xc @ Xc.T, Wc @ Wc.T
    ref = _song_hsic(K, L) / math.sqrt(_song_hsic(K, K) * _song_hsic(L, L))
    assert stats.linear_cka(X, W, debiased=True) == pytest.approx(ref, abs=1e-10)
    # the unbiased estimator is translation invariant: uncentered Grams give the same value
    Ku, Lu = X @ X.T, (W + 3.0) @ (W + 3.0).T
    ref_u = _song_hsic(Ku, Lu) / math.sqrt(_song_hsic(Ku, Ku) * _song_hsic(Lu, Lu))
    assert ref_u == pytest.approx(ref, abs=1e-8)
    biased = stats.linear_cka(X, W)
    ref_biased = np.linalg.norm(Wc.T @ Xc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Wc.T @ Wc))
    assert biased == pytest.approx(ref_biased, abs=1e-12)
    assert abs(stats.linear_cka(X, W, debiased=True)) < biased


def test_spearman_and_bootstrap_intervals() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(50)
    r = stats.spearman_ci(x, x ** 3, rng, n_boot=300)
    assert r["rho"] == pytest.approx(1.0) and r["ci_lo"] == pytest.approx(1.0) and r["n"] == 50
    noisy = stats.spearman_ci(x, x + rng.standard_normal(50), rng, n_boot=300)
    assert noisy["ci_lo"] <= noisy["rho"] <= noisy["ci_hi"]
    const = stats.bootstrap_mean_ci(np.full(20, 0.3), rng, n_boot=200)
    assert const["mean"] == pytest.approx(0.3) and const["ci_lo"] == pytest.approx(0.3)
    assert const["ci_hi"] == pytest.approx(0.3)


def test_knn_purity_separable_clusters_vs_permuted_labels() -> None:
    rng = np.random.default_rng(8)
    centers = rng.standard_normal((4, 6)) * 10.0
    X = np.repeat(centers, 15, axis=0) + rng.standard_normal((60, 6))
    labels = np.repeat(["a", "b", "c", "d"], 15)
    good = stats.knn_label_purity(X, labels, k=5, metric="euclidean", rng=rng, n_perm=200)
    assert good["purity"] == pytest.approx(1.0)
    assert good["z"] > 5 and good["p"] == pytest.approx(1 / 201)
    assert good["chance"] == pytest.approx(4 * 15 * 14 / (60 * 59))
    shuffled = stats.knn_label_purity(X, rng.permutation(labels), k=5, metric="euclidean", rng=rng, n_perm=200)
    assert shuffled["purity"] < 0.5 and shuffled["p"] > 0.01
    sil = stats.silhouette(X, labels, metric="euclidean")
    assert sil["silhouette"] > 0.8 and set(sil["per_label"]) == {"a", "b", "c", "d"}
    with pytest.raises(ValueError, match="rng"):
        stats.knn_label_purity(X, labels, k=5, rng=None, n_perm=10)


def test_hubness_counts_sum_to_nk() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((50, 4))
    h = stats.hubness(X, k=5)
    idx, _ = stats.knn(X, 5, "cosine")
    Nk = np.bincount(idx.ravel(), minlength=50)
    assert Nk.sum() == 250
    assert h["max_Nk"] == Nk.max() and h["antihub_fraction"] == pytest.approx((Nk == 0).mean())


def test_frequency_probe_detects_planted_frequency_and_pca_branch() -> None:
    rng = np.random.default_rng(10)
    lc = rng.uniform(0, 8, size=200)
    X = rng.standard_normal((200, 5))
    X[:, 0] = 3.0 * lc + 0.1 * rng.standard_normal(200)
    probe = stats.frequency_probe(X, lc, rng)
    assert probe["r2_mean"] > 0.95 and probe["control_r2_mean"] < 0.1 and probe["pca_components"] is None
    wide = np.hstack([X, 0.1 * rng.standard_normal((200, 30))])
    probe_wide = stats.frequency_probe(wide, lc, rng, max_features=4)
    assert probe_wide["pca_components"] == 4 and probe_wide["r2_mean"] > 0.95


# ----------------------------------------------------------------------------------------------
# vectors.py: preprocessing, neighbors, analogies
# ----------------------------------------------------------------------------------------------

def test_prepare_centers_and_normalizes_only_cosine_views() -> None:
    rng = np.random.default_rng(11)
    X = rng.standard_normal((10, 3)) + 5.0
    cos_view = VectorView("c", "c", np.arange(10), X)
    P = vectors.prepare(cos_view, center=True)
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-12)
    Xc = X.astype(np.float32).astype(np.float64) - X.astype(np.float32).astype(np.float64).mean(0)
    np.testing.assert_allclose(P, Xc / np.linalg.norm(Xc, axis=1, keepdims=True), atol=1e-12)
    euc_view = VectorView("e", "e", np.arange(10), X, metric="euclidean")
    np.testing.assert_allclose(vectors.prepare(euc_view, center=True), X.astype(np.float32), atol=0)


def test_nearest_neighbors_match_brute_force() -> None:
    rng = np.random.default_rng(12)
    X = rng.standard_normal((30, 4))
    view = VectorView("v", "v", np.arange(100, 130), X)
    nn = vectors.nearest_neighbors(view, [105, 110, 999], k=3)
    assert set(nn) == {105, 110}
    P = vectors.prepare(view, center=True)
    S = P @ P.T
    np.fill_diagonal(S, -np.inf)
    for q in (105, 110):
        ref = np.argsort(-S[q - 100])[:3] + 100
        assert [t for t, _ in nn[q]] == ref.tolist()


@pytest.mark.parametrize("metric,center,exclude", [
    ("cosine", True, True), ("cosine", False, True), ("cosine", True, False), ("euclidean", False, True),
])
def test_analogy_scores_match_brute_force(metric: str, center: bool, exclude: bool) -> None:
    rng = np.random.default_rng(13)
    n = 40
    view = VectorView("v", "v", np.arange(n) + 1000, rng.standard_normal((n, 6)), metric=metric)
    qs = _random_questions(n, 12, rng)
    qs_ids = [AnalogyQuestion(q.category, q.a + 1000, q.a_star + 1000, q.b + 1000, q.b_star + 1000) for q in qs]
    res = vectors.analogy_scores(view, qs_ids, center=center, exclude_queries=exclude, topk=3)
    expected_methods = vectors.ANALOGY_METHODS if metric == "cosine" else ("3cosadd", "only_b", "ignore_a")
    assert tuple(res["methods"]) == expected_methods
    if metric == "euclidean":
        assert set(res["skipped_methods"]) == {"3cosmul", "pairdir"}
    Xp = vectors.prepare(view, center)
    for method in expected_methods:
        for i, q in enumerate(qs):
            rank, top = _brute_ranks(Xp, np.array([q.a, q.a_star, q.b, q.b_star]), method, metric, exclude)
            assert res["methods"][method]["rank"][i] == rank, (method, i)
            assert res["methods"][method]["topk"][i, 0] in top, (method, i)
    for i, q in enumerate(qs):
        rank, top = _brute_ranks(Xp, np.array([q.a, q.a_star, q.b, q.b_star]), "3cosadd", metric, False)
        assert res["honest"]["rank"][i] == rank and res["honest"]["top1"][i] in top
    top1_only = vectors.analogy_scores(view, qs_ids, center=center, exclude_queries=exclude, topk=1)
    for method in expected_methods:
        np.testing.assert_array_equal(top1_only["methods"][method]["rank"], res["methods"][method]["rank"])
        for i, q in enumerate(qs):
            _, top = _brute_ranks(Xp, np.array([q.a, q.a_star, q.b, q.b_star]), method, metric, exclude)
            assert top1_only["methods"][method]["topk"][i, 0] in top


def test_planted_parallelogram_is_retrieved_exactly() -> None:
    rng = np.random.default_rng(14)
    n, D = 50, 10
    X = rng.standard_normal((n, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    t = X[2] - X[0] + X[1]                                              # b* = unit(a* - a + b)
    X[3] = t / np.linalg.norm(t)
    q = AnalogyQuestion("gender", 0, 1, 2, 3)                           # a=0, a*=1, b=2 -> b*=3
    view = VectorView("v", "v", np.arange(n), X)
    res = vectors.analogy_scores(view, [q], center=False)
    assert res["methods"]["3cosadd"]["rank"][0] == 1
    assert res["methods"]["3cosadd"]["topk"][0, 0] == 3
    # Euclidean: an exact raw parallelogram has distance 0 to the target, with or without centering
    Y = rng.standard_normal((n, D)) * 3.0 + 1.0
    Y[3] = Y[1] - Y[0] + Y[2]
    evw = VectorView("e", "e", np.arange(n), Y, metric="euclidean")
    eres = vectors.analogy_scores(evw, [q], center=True)
    assert eres["methods"]["3cosadd"]["rank"][0] == 1


def test_analogy_benchmark_groups_coverage_and_degenerate_questions() -> None:
    rng = np.random.default_rng(15)
    n = 30
    view = VectorView("v", "v", np.arange(n), rng.standard_normal((n, 5)))
    qs = (_random_questions(n, 6, rng, "capital-common-countries")
          + _random_questions(n, 4, rng, "gram3-comparative")
          + [AnalogyQuestion("gram3-comparative", 0, 1, 2, 999),       # not in view
             AnalogyQuestion("capital-common-countries", 0, 1, 2, 1)])  # degenerate (b* = a*)
    bench = vectors.analogy_benchmark(view, qs, rng, n_boot=100)
    assert bench["n_total"] == 12 and bench["n_covered"] == 10 and bench["n_degenerate"] == 1
    assert bench["semantic"]["n"] == 6 and bench["syntactic"]["n"] == 4 and bench["overall"]["n"] == 10
    assert bench["syntactic"]["n_total"] == 5 and bench["semantic"]["n_total"] == 7
    ranks = vectors.analogy_scores(view, qs)["methods"]["3cosadd"]["rank"]
    assert bench["overall"]["methods"]["3cosadd"]["acc"]["mean"] == pytest.approx(float((ranks == 1).mean()))
    assert bench["overall"]["methods"]["3cosadd"]["mrr"]["mean"] == pytest.approx(float((1.0 / ranks).mean()))
    assert set(bench["categories"]) == {"capital-common-countries", "gram3-comparative"}
    # random questions share no word pair: every question is its own resampling unit, and it says so
    assert bench["resampling"] == "questions" and bench["overall"]["n_question_units"] == 10
    assert "resampled as single questions" in bench["resampling_note"]
    assert bench["n_total"] == 12 and bench["totals_source"] == "questions given"
    totals = {"n_total": 100, "per_category": {"capital-common-countries": {"n_total": 50},
                                               "gram3-comparative": {"n_total": 30}, "currency": 20}}
    full = vectors.analogy_benchmark(view, qs, rng, n_boot=100, totals=totals)
    assert full["n_total"] == 100 and full["n_resolved"] == 12 and full["coverage"] == pytest.approx(0.1)
    assert full["semantic"]["n_total"] == 70 and full["syntactic"]["n_total"] == 30
    assert full["categories"]["currency"]["n"] == 0 and full["categories"]["currency"]["n_total"] == 20
    assert full["categories"]["gram3-comparative"]["n_resolved"] == 5
    examples = vectors.analogy_examples(view, qs[-2:] + qs[:1])
    assert [e["covered"] for e in examples] == [False, False, True]
    assert set(examples[2]["methods"]) == set(vectors.ANALOGY_METHODS)
    assert len(examples[2]["methods"]["3cosadd"]["topk"]) == 5


def _crossed_only_b_world(rng: np.random.Generator) -> tuple:
    r"""Euclidean view with crossed pair categories whose ONLY-B hit depends on the b-pair alone.

    Pair ``p`` has words ``w_p`` and ``w*_p``. A good pair has ``w*_p`` next to ``w_p``; a bad pair has a
    decoy word nearer to ``w_p`` than ``w*_p``. In question ``(i, j)`` ONLY-B returns the nearest word to
    ``w_j`` other than ``w_i, w*_i, w_j``, so it hits iff pair ``j`` is good.
    """
    D, rows, questions = 8, [], []

    def category(name: str, good: List[bool]) -> None:
        ids = []
        for is_good in good:
            w = rng.standard_normal(D) * 6.0
            u = rng.standard_normal((2, D))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            star = w + (0.1 if is_good else 0.6) * u[0]
            rows.extend([w, star])
            ids.append((len(rows) - 2, len(rows) - 1))
            if not is_good:
                rows.append(w + 0.05 * u[1])                             # decoy, nearer than w*
        questions.extend(AnalogyQuestion(name, ids[i][0], ids[i][1], ids[j][0], ids[j][1])
                         for i in range(len(ids)) for j in range(len(ids)) if i != j)

    category("family", [True, False] * 5)                                # 10 pairs, 90 questions
    category("gram8-plural", [False] * 6)                                # 6 pairs, 30 questions, no hit
    view = VectorView("e", "e", np.arange(len(rows)), np.array(rows), metric="euclidean")
    return view, questions


def test_analogy_intervals_resample_word_pairs_and_use_wilson_at_zero_hits() -> None:
    rng = np.random.default_rng(30)
    view, qs = _crossed_only_b_world(rng)
    bench = vectors.analogy_benchmark(view, qs, rng, n_boot=2000)
    family = bench["categories"]["family"]
    assert family["n"] == 90 and family["n_pairs"] == 10 and family["n_units"] == 10
    assert family["resampling"] == "pairs" and family["n_question_units"] == 0
    only_b = family["methods"]["only_b"]["acc"]
    assert only_b["mean"] == pytest.approx(0.5) and only_b["ci_method"] == "bootstrap over pairs"
    ranks = vectors.analogy_scores(view, [q for q in qs if q.category == "family"])["methods"]["only_b"]["rank"]
    question_level = stats.bootstrap_mean_ci(ranks == 1, rng, 2000)
    # hits are shared by the 9 questions of each b-pair: the pair interval is several times wider
    assert only_b["ci_hi"] - only_b["ci_lo"] > 2.0 * (question_level["ci_hi"] - question_level["ci_lo"])
    zero = bench["categories"]["gram8-plural"]["methods"]["only_b"]["acc"]
    z2 = 1.959963984540054 ** 2
    assert zero["mean"] == 0.0 and zero["ci_lo"] == 0.0 and zero["ci_method"] == "Wilson on 6 pairs"
    assert zero["ci_hi"] == pytest.approx(z2 / (6 + z2))
    overall = bench["overall"]
    assert overall["n"] == 120 and overall["n_pairs"] == 16 and overall["resampling"] == "pairs"
    assert overall["methods"]["only_b"]["acc"]["mean"] == pytest.approx(45 / 120)
    lo, hi = overall["methods"]["only_b"]["acc"]["ci_lo"], overall["methods"]["only_b"]["acc"]["ci_hi"]
    assert 0.0 < lo < 45 / 120 < hi < 1.0
    assert overall["honest"]["acc"]["ci_method"].startswith(("bootstrap", "Wilson"))


def test_wilson_interval_and_unit_bootstrap_weights() -> None:
    z2 = 1.959963984540054 ** 2
    assert stats.wilson_interval(0, 18) == (0.0, pytest.approx(z2 / (18 + z2)))
    assert stats.wilson_interval(18, 18) == (pytest.approx(18 / (18 + z2)), 1.0)
    lo, hi = stats.wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0) and 0.2 < lo < 0.5 < hi < 0.8
    k, B = 5, 300
    counts = stats.unit_bootstrap_weights(np.arange(k), -np.ones(k), k, np.random.default_rng(3), B)
    assert counts.shape == (B, k) and (counts.sum(axis=1) == k).all()
    I, J = np.array([(i, j) for i in range(k) for j in range(k) if i != j]).T
    W = stats.unit_bootstrap_weights(I, J, k, np.random.default_rng(3), B)
    np.testing.assert_array_equal(W, counts[:, I] * counts[:, J])
    np.testing.assert_array_equal(W.sum(axis=1), k ** 2 - (counts ** 2).sum(axis=1))
    with pytest.raises(ValueError, match="distinct"):
        stats.unit_bootstrap_weights([0, 1], [0, 2], 3, np.random.default_rng(0), 2)


# ----------------------------------------------------------------------------------------------
# vectors.py: offsets, directions, similarity, fields, clusters, projections, geometry
# ----------------------------------------------------------------------------------------------

def test_offset_consistency_planted_relation_vs_random_pairs() -> None:
    rng = np.random.default_rng(16)
    n, D = 60, 12
    X = rng.standard_normal((n, D))
    delta = 3.0 * rng.standard_normal(D)
    for i in range(10):                                                 # rows 30+i = rows i + delta
        X[30 + i] = X[i] + delta + 0.05 * rng.standard_normal(D)
    view = VectorView("e", "e", np.arange(n), X, metric="euclidean")
    relations = {"planted": [(i, 30 + i) for i in range(10)],
                 "random": [(int(a), int(b)) for a, b in rng.choice(np.arange(40, 60), size=(8, 2), replace=False)],
                 "tiny": [(0, 1), (2, 3)]}
    replay = np.random.default_rng()
    replay.bit_generator.state = rng.bit_generator.state               # same stream as the call below
    res = vectors.offset_consistency(view, relations, rng, n_perm=200)
    planted, random = res["relations"]["planted"], res["relations"]["random"]
    assert planted["ocs"] > 0.99 and planted["z"] > 5 and planted["p"] == pytest.approx(1 / 201)
    assert planted["top_pc_share"] > 0.99 and planted["null_top_pc_share_mean"] < 0.9
    assert planted["loo_proj_mean"] > 0.95 * np.linalg.norm(delta)
    assert planted["null_mean"] < 0.9                                  # re-pairing adds x_pi(i) - x_i noise
    assert planted["effect_size"] == pytest.approx((planted["ocs"] - planted["null_mean"]) / (1.0 - planted["null_mean"]))
    assert planted["effect_size"] > 0.9 and abs(random["effect_size"]) < 0.5
    assert math.isnan(vectors._effect_size(0.5, 1.0)) and vectors._effect_size(0.3, 0.3) == 0.0
    assert random["p"] > 0.01
    # random-pair baseline (Park et al. Fig. 2): m random word pairs on the real LOO directions
    assert planted["n_random_pairs"] == 10 and len(planted["random_loo_proj"]) == 10
    for _ in range(200):                                                # replay the planted relation's draws
        replay.permutation(10)
    first = replay.integers(0, n, size=10)
    second = (first + replay.integers(1, n, size=10)) % n
    Xr = view.X.astype(np.float64)
    offsets = Xr[30:40] - Xr[:10]
    e = Xr[second] - Xr[first]
    expected = [e[k] @ (offsets.sum(0) - offsets[k]) / np.linalg.norm(offsets.sum(0) - offsets[k]) for k in range(10)]
    np.testing.assert_allclose(planted["random_loo_proj"], expected, atol=1e-10)
    assert planted["random_mean_proj_mean"] == pytest.approx(
        float(np.mean(e @ offsets.sum(0) / np.linalg.norm(offsets.sum(0)))), abs=1e-10)
    assert planted["loo_proj_mean"] > planted["random_loo_proj_mean"] + 3.0 * planted["random_loo_proj_sd"] / math.sqrt(10)
    assert "tiny" in res["skipped"]
    Xs = view.X.astype(np.float64)                                      # the float32 rows the view stores
    d = Xs[[30 + i for i in range(10)]] - Xs[:10]
    U = d / np.linalg.norm(d, axis=1, keepdims=True)
    C = U @ U.T
    assert planted["ocs"] == pytest.approx((C.sum() - 10) / 90, abs=1e-12)


def test_concept_directions_whitening_is_the_causal_inner_product() -> None:
    rng = np.random.default_rng(17)
    n, D = 6000, 4
    A = np.array([[1.0, 0.9, 0.0, 0.0], [0.0, 0.45, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    X = rng.standard_normal((n, D)) @ A.T
    e1, e2 = A @ np.eye(D)[0], A @ np.eye(D)[1]
    extra, rel1, rel2 = [], [], []
    for i in range(6):
        extra.append(X[i] + e1)
        rel1.append((i, n + len(extra) - 1))
        extra.append(X[10 + i] + e2)
        rel2.append((10 + i, n + len(extra) - 1))
    view = VectorView("v", "v", np.arange(n + len(extra)), np.vstack([X, np.array(extra)]))
    rels = {"r1": rel1, "r2": rel2, "single": [(0, 1)]}
    raw = vectors.concept_direction_cosines(view, rels, whiten=False)
    wht = vectors.concept_direction_cosines(view, rels, whiten=True)
    assert raw["relations"] == ["r1", "r2"] and "single" in raw["skipped"]
    assert raw["cosine"][0, 1] == pytest.approx(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2)), abs=1e-5)
    assert abs(wht["cosine"][0, 1]) < 0.05 and wht["whitening_rank"] == D


def test_similarity_scores_and_benchmark() -> None:
    rng = np.random.default_rng(18)
    view = VectorView("v", "v", np.arange(20), rng.standard_normal((20, 4)))
    P = vectors.prepare(view, center=True)
    pairs = [(i, i + 1, float(P[i] @ P[i + 1])) for i in range(0, 18, 2)] + [(0, 99, 1.0), (3, 3, 1.0)]
    scores, human = vectors.view_similarity_scores(view, pairs)
    assert scores.size == 9
    np.testing.assert_allclose(scores, human, atol=1e-6)
    bench = vectors.similarity_benchmark(scores, human, rng)
    assert bench["rho"] == pytest.approx(1.0) and bench["n"] == 9


def test_field_clustering_separable_vs_permuted_fields() -> None:
    rng = np.random.default_rng(19)
    dirs = _orthonormal(8, 3, rng).T * 5.0
    X = np.vstack([np.repeat(dirs, 8, axis=0) + 0.3 * rng.standard_normal((24, 8)),
                   rng.standard_normal((6, 8))])
    view = VectorView("v", "v", np.arange(30), X)
    lexicon = {"f0": list(range(0, 8)), "f1": list(range(8, 16)), "f2": list(range(16, 24)),
               "lonely": [24], "absent": [500, 501]}
    res = vectors.field_clustering(view, lexicon, rng, k=5)
    assert res["n_words"] == 24 and res["n_fields"] == 3 and set(res["fields_dropped"]) == {"lonely"}
    assert res["purity"]["purity"] == pytest.approx(1.0) and res["purity"]["p"] < 0.01
    assert res["silhouette"]["silhouette"] > 0.5
    perm = rng.permutation(24)
    shuffled = {"f0": perm[:8].tolist(), "f1": perm[8:16].tolist(), "f2": perm[16:].tolist()}
    res_perm = vectors.field_clustering(view, shuffled, rng, k=5)
    assert res_perm["purity"]["purity"] < 0.6 and res_perm["purity"]["p"] > 0.01


def test_cluster_vocabulary_recovers_planted_clusters() -> None:
    rng = np.random.default_rng(20)
    dirs = _orthonormal(6, 3, rng).T * 5.0
    X = np.repeat(dirs, 10, axis=0) + 0.2 * rng.standard_normal((30, 6))
    view = VectorView("v", "v", np.arange(30), X)
    res = vectors.cluster_vocabulary(view, np.arange(30), k_clusters=3, rng=rng, n_central=4)
    assert res["k"] == 3 and sum(c["size"] for c in res["clusters"]) == 30
    for c in res["clusters"]:
        assert c["size"] == 10 and len({t // 10 for t in c["central_ids"]}) == 1


def test_semantic_projection_orders_words_along_planted_axis() -> None:
    rng = np.random.default_rng(21)
    D = 6
    u = np.eye(D)[0]
    X = np.zeros((20, D))
    for i in range(4):                                                  # pairs (neg, pos) = (c - u, c + u)
        c = np.concatenate([[0.0], rng.standard_normal(D - 1)])
        X[2 * i], X[2 * i + 1] = c - u, c + u
    t = np.linspace(-2, 2, 12)
    for j in range(12):
        X[8 + j] = np.concatenate([[t[j]], rng.standard_normal(D - 1)])
    view = VectorView("v", "v", np.arange(20), X, metric="euclidean")
    axis = {"pairs": [(2 * i, 2 * i + 1) for i in range(4)], "words": list(range(8, 20)) + [77],
            "neg_label": "small", "pos_label": "big"}
    res = vectors.semantic_projection(view, axis)
    assert [w for w, _ in res["words"]] == list(range(19, 7, -1))
    np.testing.assert_allclose([s for _, s in res["words"]], t[::-1], atol=1e-6)
    assert res["loo_pair_accuracy"] == 1.0 and res["separation"] == pytest.approx(2.0, abs=1e-6)
    assert res["n_words"] == 12


def test_global_geometry_reports_and_detects_frequency() -> None:
    rng = np.random.default_rng(22)
    n = 300
    lc = np.log1p(np.arange(n, 0, -1) * 5.0)
    X = rng.standard_normal((n, 6))
    X[:, 0] = 20.0 * lc
    view = VectorView("v", "v", np.arange(n), X)
    g = vectors.global_geometry(view, lc, rng, max_points=200)
    assert g["n"] == 200 and g["selection"] == "top-frequency" and not g["notes"]
    for key in ("anisotropy_raw", "anisotropy_centered", "mean_vector_ratio", "isoscore", "effective_rank",
                "participation_ratio", "pca_spectrum", "twonn", "hubness", "rogue_dimensions",
                "frequency_probe", "pc_log_count_spearman", "norm_log_count_spearman"):
        assert key in g and not (isinstance(g[key], dict) and g[key].get("status") == "failed"), key
    assert g["frequency_probe"]["r2_mean"] > 0.9
    assert abs(g["pc_log_count_spearman"][0]["rho"]) > 0.95
    wide = VectorView("w", "w", np.arange(n), np.hstack([X, 0.1 * rng.standard_normal((n, 600))]))
    gw = vectors.global_geometry(wide, lc, rng, max_points=200)      # D = 606 > 512 and n < D
    assert gw["frequency_probe"]["pca_components"] == 199 and gw["frequency_probe"]["r2_mean"] > 0.9
    top = vectors.global_geometry(view, None, rng, max_points=50)
    assert top["n"] == 50 and "frequency_probe" not in top and top["notes"]


# ----------------------------------------------------------------------------------------------
# Battery, figures, tables
# ----------------------------------------------------------------------------------------------

def _synthetic_world(tmp_path: Path) -> tuple:
    r"""A 90-token world: 3 fields, a planted gender relation and analogies, one axis."""
    rng = np.random.default_rng(23)
    n, D = 90, 8
    dirs = _orthonormal(D, 3, rng).T * 4.0
    X = rng.standard_normal((n, D))
    X[:30] += np.repeat(dirs, 10, axis=0)
    gender = rng.standard_normal(D) * 2.0
    for i in range(8):
        X[40 + i] = X[50 + i] + gender + 0.1 * rng.standard_normal(D)
    lexicon = {"animals": list(range(0, 10)), "colors": list(range(10, 20)), "numbers": list(range(20, 30))}
    relations = {"gender": [(50 + i, 40 + i) for i in range(8)],
                 "country_capital": [(60 + i, 70 + i) for i in range(6)]}
    analogies = [AnalogyQuestion("family", 50 + i, 40 + i, 50 + j, 40 + j) for i in range(8) for j in range(8) if i != j]
    analogies += [AnalogyQuestion("gram1-adjective-to-adverb", 60 + i, 70 + i, 61 + i, 71 + i) for i in range(5)]
    curated = analogies[:3] + [AnalogyQuestion("capital", 60, 70, 61, 1234)]
    similarity = {"simlex999": [(i, i + 1, float(rng.uniform(0, 10))) for i in range(0, 40, 2)]}
    axes = {"size": {"pairs": [(20 + i, 10 + i) for i in range(4)], "words": list(range(0, 10)),
                     "neg_label": "small", "pos_label": "big"}}
    google = {"available": True, "n_total": 230, "n_resolved": 61,
              "per_category": {"family": {"n_total": 120, "n_resolved": 56},
                               "gram1-adjective-to-adverb": {"n_total": 80, "n_resolved": 5},
                               "currency": {"n_total": 30, "n_resolved": 0}}}
    ctx = _ctx(tmp_path, n, lexicon=lexicon, relations=relations, analogies=analogies, curated_analogies=curated,
               similarity=similarity, axes=axes, probe_ids=[0, 10, 50, 999], coverage={"google_analogies": google},
               options={"umap": True, "cluster_top_words": 60, "n_clusters": 4, "geometry_max_points": 80})
    views = {"mu_raw": VectorView("mu_raw", "mu (raw)", np.arange(n), X, group="mu"),
             "w2_chart": VectorView("w2_chart", "W2 chart", np.arange(n), X * 0.5 + 1.0, metric="euclidean",
                                    group="sigma")}
    return ctx, views


def _statuses(obj: Any) -> List[str]:
    found: List[str] = []
    if isinstance(obj, dict):
        if obj.get("status") == "failed":
            found.append(obj.get("error", "?"))
        for v in obj.values():
            found.extend(_statuses(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_statuses(v))
    return found


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    ctx, views = _synthetic_world(tmp_path_factory.mktemp("semantics_vectors"))
    results = {name: vectors.run_vector_battery(view, ctx) for name, view in views.items()}
    return ctx, views, results


def test_run_vector_battery_is_complete_and_json_serializable(world: tuple) -> None:
    ctx, views, results = world
    for name, battery in results.items():
        assert _statuses(battery) == [], (name, battery["notes"])
        for key in ("view", "neighbors", "analogy_benchmark", "analogy_examples", "offsets", "directions_raw",
                    "directions_whitened", "similarity", "fields", "clusters", "projections", "geometry",
                    "coverage", "failed_steps", "notes", "seconds"):
            assert key in battery, key
        assert battery["failed_steps"] == []
        json.dumps(to_jsonable(battery), allow_nan=False)
        assert battery["coverage"]["probe_words"] == 3 and battery["coverage"]["curated_analogies"] == 3
        bench = battery["analogy_benchmark"]                           # full-benchmark denominators
        assert bench["n_total"] == 230 and bench["n_resolved"] == 61 and bench["totals_source"] == "full benchmark"
        assert bench["categories"]["family"]["n_total"] == 120 and bench["categories"]["currency"]["n"] == 0
        assert bench["semantic"]["n_total"] == 150 and bench["syntactic"]["n_total"] == 80
        assert bench["overall"]["coverage"] == pytest.approx(61 / 230)
        assert battery["coverage"]["analogies_total"] == 230 and battery["coverage"]["analogies_resolved"] == 61
    mu = results["mu_raw"]
    assert mu["backend"] == {"requested": "cpu", "device": None, "path": "numpy float64"}
    for group in ("overall", "semantic", "syntactic"):                 # macro beside micro
        macro = mu["analogy_benchmark"][group]["macro"]
        cats = macro["category_n"]
        expected = np.mean([mu["analogy_benchmark"]["categories"][c]["methods"]["3cosadd"]["acc"]["mean"] for c in cats])
        assert macro["methods"]["3cosadd"]["acc"]["mean"] == pytest.approx(expected)
        assert mu["analogy_benchmark"][group]["largest_category"] in cats
    strat = mu["fields"]["purity"]["frequency_stratified"]             # counts known: both purity nulls
    assert mu["fields"]["purity"]["null"] == "both" and strat["n_strata"] == 10 and strat["n_perm"] == 200
    for f, rec in mu["fields"]["per_field"].items():
        assert rec["chance"] == pytest.approx((rec["n"] - 1) / (mu["fields"]["n_words"] - 1))
        assert rec["purity_excess"] == pytest.approx((rec["purity"] - rec["chance"]) / (1 - rec["chance"]))
    gender = mu["offsets"]["relations"]["gender"]
    assert gender["effect_size"] == pytest.approx((gender["ocs"] - gender["null_mean"]) / (1 - gender["null_mean"]))
    for key in ("directions_raw", "directions_whitened"):
        d = mu[key]
        assert d["n_null"] == vectors.DIRECTION_NULL_DRAWS and d["frequency_matched"]
        excess = np.asarray(d["excess"], dtype=np.float64)
        assert np.isnan(np.diag(excess)).all()
        off = ~np.eye(excess.shape[0], dtype=bool)
        np.testing.assert_allclose(excess[off], (np.asarray(d["cosine"]) - np.asarray(d["null_mean"]))[off],
                                   atol=1e-12)
    assert mu["offsets"]["relations"]["gender"]["z"] > 3
    assert mu["fields"]["purity"]["purity"] > 0.75 and mu["fields"]["purity"]["z"] > 3   # max 9/10 at k=10
    assert mu["analogy_benchmark"]["semantic"]["methods"]["3cosadd"]["acc"]["mean"] > 0.5
    assert set(results["w2_chart"]["analogy_benchmark"]["skipped_methods"]) == {"3cosmul", "pairdir"}
    again = vectors.run_vector_battery(views["mu_raw"], ctx)          # seeded: bitwise reproducible
    assert json.dumps(to_jsonable({k: v for k, v in again.items() if k != "seconds"}), allow_nan=False) == \
        json.dumps(to_jsonable({k: v for k, v in mu.items() if k != "seconds"}), allow_nan=False)


def test_plot_view_embedding_surfaces_umap_failure(world: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.viz.figures as viz_figures

    ctx, views, results = world
    battery = {**results["mu_raw"], "notes": []}

    def broken(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("numba exploded")

    titles: List[str] = []
    real_save = vectors.save_figure

    def capture(fig: Any, *args: Any, **kwargs: Any) -> list:
        titles.append(fig._suptitle.get_text())
        titles.append(str(len(fig.axes)))
        return real_save(fig, *args, **kwargs)

    monkeypatch.setattr(viz_figures, "umap_embed", broken)
    monkeypatch.setattr(vectors, "save_figure", capture)
    paths = vectors.plot_view_embedding(views["mu_raw"], ctx, battery)
    assert {p.suffix for p in paths} == {".png", ".pdf"} and all(p.exists() for p in paths)
    assert "numba exploded" in titles[0] and titles[1] == "1"
    assert any("numba exploded" in note for note in battery["notes"])


def test_plot_view_embedding_passes_prepared_rows_to_umap(world: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.viz.figures as viz_figures

    ctx, views, results = world
    seen: Dict[str, Any] = {}

    def fake(features: np.ndarray, **kwargs: Any) -> np.ndarray:
        seen["X"], seen["kwargs"] = np.asarray(features), kwargs
        return np.asarray(features)[:, :2] * 3.0

    monkeypatch.setattr(viz_figures, "umap_embed", fake)
    battery = {**results["mu_raw"], "notes": []}
    paths = vectors.plot_view_embedding(views["mu_raw"], ctx, battery)
    assert len(paths) == 2 and battery["notes"] == []
    expected = vectors.prepare(views["mu_raw"], center=True)[:30]
    np.testing.assert_allclose(seen["X"], expected)
    assert seen["kwargs"]["seed"] == ctx.seed


def test_offsets_figure_tables_and_comparison(world: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, views, results = world
    offsets = vectors.plot_offsets(views["mu_raw"], ctx, ctx.relations, battery=results["mu_raw"])
    assert len(offsets) == 2 and all(p.exists() for p in offsets)
    tables = vectors.write_view_tables(views["mu_raw"], ctx, results["mu_raw"])
    names = {p.stem for p in tables}
    assert names == {"neighbors", "analogy_examples", "analogy_benchmark", "offsets", "clusters"}
    text = (Path(ctx.tables_dir) / "embeddings" / "mu_raw" / "analogy_examples.md").read_text(encoding="utf-8")
    assert "not in view" in text and "w50 : w40 :: w51 : ?" in text
    bench_text = (Path(ctx.tables_dir) / "embeddings" / "mu_raw" / "analogy_benchmark.md").read_text(encoding="utf-8")
    assert "| family | 56 / 120 |" in bench_text and "coverage" in bench_text and "pairs" in bench_text
    assert "| currency | 0 / 30 |" in bench_text and "full Google benchmark" in bench_text
    offsets_text = (Path(ctx.tables_dir) / "embeddings" / "mu_raw" / "offsets.md").read_text(encoding="utf-8")
    assert "| LOO proj | re-paired | random pairs |" in offsets_text and "| effect size |" in offsets_text
    assert "3CosAdd macro" in bench_text and "largest category (share)" in bench_text
    figures: Dict[str, Any] = {}
    real_save = vectors.save_figure

    def capture(fig: Any, context: Any, section: str, name: str) -> list:
        figures[name] = fig
        return real_save(fig, context, section, name)

    monkeypatch.setattr(vectors, "save_figure", capture)
    ctx.cache.pop("vectors_failed_steps", None)
    written = vectors.plot_views_comparison(results, views, ctx)
    stems = {p.stem for p in written}
    assert {"analogy_categories", "analogy_methods", "similarity", "fields", "field_purity", "geometry", "offsets",
            "directions", "scree", "frequency", "projections_mu_raw"} <= stems
    assert all(p.exists() for p in written) and not ctx.cache.get("vectors_failed_steps")
    order = list(results)
    # (a) analogy heatmap: color = 3CosAdd - ONLY-B, cells "acc\n(ONLY-B)", micro and macro rows
    ax = figures["analogy_categories"].axes[0]
    shown = np.ma.filled(ax.images[0].get_array(), np.nan)
    assert any("\n(" in t.get_text() for t in ax.texts)
    for c, v in enumerate(order):
        overall = results[v]["analogy_benchmark"]["overall"]
        for r, g in enumerate((overall, overall["macro"])):             # rows: overall, overall macro
            margin = g["methods"]["3cosadd"]["acc"]["mean"] - g["methods"]["only_b"]["acc"]["mean"]
            assert shown[r, c] == pytest.approx(margin)
    assert [t.get_text() for t in ax.get_yticklabels()][1].startswith("overall macro")
    # (b) offsets heatmap: color = effect size, one dot per cell with p < 0.05
    ax = figures["offsets"].axes[0]
    shown = np.ma.filled(ax.images[0].get_array(), np.nan)
    rels = list(results[order[0]]["offsets"]["relations"])
    n_dots = 0
    for c, v in enumerate(order):
        for r, name in enumerate(rels):
            rec = results[v]["offsets"]["relations"][name]
            assert shown[r, c] == pytest.approx(rec["effect_size"])
            n_dots += int(rec["p"] < vectors.OFFSET_ALPHA)
    # filled (Holm-adjusted) and open (unadjusted only) dots together mark every p < 0.05 (audit 2026-10-16 F71)
    assert sum(len(line.get_xdata()) for line in ax.lines) == n_dots > 0
    # (c) directions: observed minus null mean, masked (grey) where |z| < 2
    ax = figures["directions"].axes[0]
    by_label = {views[v].label: v for v in order}
    d = results[by_label[ax.get_title().split(",")[0]]]["directions_whitened"]
    shown = np.ma.filled(ax.images[0].get_array(), np.nan)
    excess, z = np.asarray(d["excess"], dtype=np.float64), np.asarray(d["z"], dtype=np.float64)
    loud = np.abs(np.nan_to_num(z)) >= vectors.DIRECTION_Z
    np.testing.assert_allclose(shown[loud], excess[loud])
    assert np.isnan(shown[~loud]).all()
    # (d) fields: both null bands per view; (e) per-field heatmap of (P_f - c_f)/(1 - c_f)
    assert len(figures["fields"].axes[0].collections) == 2 * len(order)
    ax = figures["field_purity"].axes[0]
    shown = np.ma.filled(ax.images[0].get_array(), np.nan)
    fields = list(results[order[0]]["fields"]["per_field"])
    for c, v in enumerate(order):
        for r, f in enumerate(fields):
            assert shown[r, c] == pytest.approx(results[v]["fields"]["per_field"][f]["purity_excess"])
    assert [t.get_text() for t in ax.get_yticklabels()][0].endswith("(10)")


def test_direction_panels_pick_one_view_per_channel() -> None:
    groups = {"mu_raw": "mu", "mu_sigma_units": "mu", "w2_chart": "sigma", "xi_ref": "mu", "log_var": "sigma",
              "phi_raw": "phi", "phi_log": "phi", "decode_W": "decode", "decode_W_causal": "decode"}
    views = {name: VectorView(name, name, np.arange(2), np.eye(2), group=g) for name, g in groups.items()}
    names = list(groups)
    assert vectors._direction_panels(names, views, "decode_W_causal") == ["decode_W_causal", "phi_log", "xi_ref",
                                                                           "mu_raw"]
    assert vectors._direction_panels(names, views, "decode_W")[0] == "decode_W"
    fewer = [n for n in names if n not in ("phi_log", "xi_ref")]
    assert vectors._direction_panels(fewer, views, "decode_W_causal") == ["decode_W_causal", "phi_raw", "mu_raw",
                                                                          "w2_chart"]


def test_failed_steps_are_recorded_and_the_rest_continues(world: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, views, results = world

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    def tiny(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("tiny")

    monkeypatch.setattr(vectors, "cluster_vocabulary", boom)
    monkeypatch.setattr(stats, "twonn_dimension", tiny)
    battery = vectors.run_vector_battery(views["mu_raw"], ctx)
    assert {"step": "clusters", "error": "RuntimeError: boom"} in battery["failed_steps"]
    assert {"step": "geometry/twonn", "error": "ValueError: tiny"} in battery["failed_steps"]
    assert len(battery["failed_steps"]) == 2 and _statuses(battery["analogy_benchmark"]) == []
    monkeypatch.setattr(vectors, "_compare_scree", boom)
    ctx.cache.pop("vectors_failed_steps", None)
    written = vectors.plot_views_comparison(results, views, ctx)
    assert ctx.cache.pop("vectors_failed_steps") == [{"step": "compare.scree", "error": "RuntimeError: boom"}]
    assert "frequency" in {p.stem for p in written} and "scree" not in {p.stem for p in written}
    assert any("'scree' failed" in note for note in ctx.cache.get("vectors_notes", []))


# ----------------------------------------------------------------------------------------------
# Compute backend: the torch path (run on the CPU through stats.TORCH_CPU) against numpy float64
# ----------------------------------------------------------------------------------------------

def test_torch_device_resolution_fallback_and_precision_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    assert stats.torch_device(None) is None and stats.torch_device("cpu") is None
    assert stats.torch_device(stats.TORCH_CPU) == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        stats.torch_device("cuda:0")
    notes: List[str] = []                                               # the battery falls back to numpy and says so
    assert vectors._battery_device(SimpleNamespace(device="cuda"), notes) is None
    assert len(notes) == 1 and "numpy float64" in notes[0]
    assert vectors._battery_device(SimpleNamespace(device=stats.TORCH_CPU), notes) == stats.TORCH_CPU
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")                          # a process that enabled TF32
    try:
        with stats.ieee_fp32_matmul():
            assert torch.get_float32_matmul_precision() == "highest"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_torch_knn_hubness_and_purity_match_numpy(metric: str) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(40)
    X = rng.standard_normal((53, 6))
    idx, score = stats.knn(X, 5, metric, chunk=16)
    idx_t, score_t = stats.knn(X, 5, metric, chunk=16, device=stats.TORCH_CPU)
    np.testing.assert_array_equal(idx_t, idx)
    np.testing.assert_allclose(score_t, score, rtol=1e-5, atol=1e-5)
    assert score_t.dtype == np.float64
    h = stats.hubness(X, k=5, metric=metric)
    h_t = stats.hubness(X, k=5, metric=metric, device=stats.TORCH_CPU)
    assert json.dumps(to_jsonable(h_t)) == json.dumps(to_jsonable(h))
    labels = np.repeat(["a", "b", "c"], [20, 20, 13])
    lc = rng.uniform(0.0, 5.0, size=53)
    kwargs = dict(k=5, metric=metric, n_perm=50, null="both", log_count=lc)
    p = stats.knn_label_purity(X, labels, rng=np.random.default_rng(1), **kwargs)
    p_t = stats.knn_label_purity(X, labels, rng=np.random.default_rng(1), device=stats.TORCH_CPU, **kwargs)
    assert json.dumps(to_jsonable(p_t)) == json.dumps(to_jsonable(p))  # same graph, same draws
    if metric == "euclidean":
        t = stats.twonn_dimension(X, np.random.default_rng(2))
        t_t = stats.twonn_dimension(X, np.random.default_rng(2), device=stats.TORCH_CPU)
        assert t_t["n"] == t["n"] and t_t["mle"] == pytest.approx(t["mle"], rel=1e-5)
        assert t_t["fit"] == pytest.approx(t["fit"], rel=1e-5)


def test_torch_scatter_eigh_whitener_and_cka_match_numpy() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(41)
    X = rng.standard_normal((30, 7)) @ np.diag([5.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.01]) + 2.0
    Xc = X - X.mean(axis=0)
    for side in ("features", "samples"):
        w, V, used = stats.scatter_eigh(X, side=side)
        w_t, V_t, used_t = stats.scatter_eigh(X, side=side, device=stats.TORCH_CPU)
        assert used == used_t == side and w_t.dtype == np.float64
        np.testing.assert_allclose(w_t, w, atol=1e-10 * w.max())
        keep = w > 1e-8 * w.max()                                       # distinct nonzero eigenvalues: same vectors up to sign
        np.testing.assert_allclose(np.abs((V_t[:, keep] * V[:, keep]).sum(axis=0)), 1.0, atol=1e-8)
    np.testing.assert_allclose(stats.scatter_eigh(X)[0], np.linalg.eigvalsh(Xc.T @ Xc), atol=1e-9)
    assert stats.scatter_eigh(rng.standard_normal((8, 20)))[2] == "samples"
    P, rank = vectors._whitener(X)
    P_t, rank_t = vectors._whitener(X, stats.TORCH_CPU)
    assert rank == rank_t == 7
    np.testing.assert_allclose(P_t @ P_t.T, P @ P.T, rtol=1e-8, atol=1e-10 * np.abs(P @ P.T).max())
    np.testing.assert_allclose(P @ P.T, np.linalg.inv(Xc.T @ Xc / 29.0), rtol=1e-6)   # full rank: Cov^{-1}
    Y = rng.standard_normal((30, 4)) + X[:, :4]
    for debiased in (False, True):
        ref = stats.linear_cka(X, Y, debiased)
        assert stats.linear_cka(X, Y, debiased, device=stats.TORCH_CPU) == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("metric,center,exclude", [
    ("cosine", True, True), ("cosine", False, False), ("euclidean", False, True),
])
def test_torch_analogy_scores_neighbors_and_benchmark_match_numpy(metric: str, center: bool, exclude: bool) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    n = 64
    view = VectorView("v", "v", np.arange(n) + 500, rng.standard_normal((n, 7)), metric=metric)
    qs = [AnalogyQuestion(q.category, q.a + 500, q.a_star + 500, q.b + 500, q.b_star + 500)
          for q in _random_questions(n, 300, rng)]                      # > _QUESTION_CHUNK: two chunks
    ref = vectors.analogy_scores(view, qs, center=center, exclude_queries=exclude, topk=4)
    got = vectors.analogy_scores(view, qs, center=center, exclude_queries=exclude, topk=4, device=stats.TORCH_CPU)
    assert list(got["methods"]) == list(ref["methods"])
    for method in ref["methods"]:
        np.testing.assert_array_equal(got["methods"][method]["rank"], ref["methods"][method]["rank"])
        if method == "ignore_a" and not exclude:
            # a* and b tie exactly (score 1 + cos(a*, b)), so either may fill a slot: identify them
            a_star = np.array([q.a_star for q in qs])[:, None]
            b = np.array([q.b for q in qs])[:, None]
            merged = [np.sort(np.where(top == b, a_star, top), axis=1)
                      for top in (got["methods"][method]["topk"], ref["methods"][method]["topk"])]
            np.testing.assert_array_equal(merged[0], merged[1])
        else:
            np.testing.assert_array_equal(got["methods"][method]["topk"], ref["methods"][method]["topk"])
    np.testing.assert_array_equal(got["honest"]["rank"], ref["honest"]["rank"])
    np.testing.assert_array_equal(got["honest"]["top1"], ref["honest"]["top1"])
    probes = [500, 510, 520, 999]
    nn = vectors.nearest_neighbors(view, probes, k=5)
    nn_t = vectors.nearest_neighbors(view, probes, k=5, device=stats.TORCH_CPU)
    assert set(nn_t) == set(nn) == {500, 510, 520}
    for q in nn:
        assert [t for t, _ in nn_t[q]] == [t for t, _ in nn[q]]
        np.testing.assert_allclose([s for _, s in nn_t[q]], [s for _, s in nn[q]], rtol=1e-5, atol=1e-5)
    bench = vectors.analogy_benchmark(view, qs, np.random.default_rng(1), n_boot=50)
    bench_t = vectors.analogy_benchmark(view, qs, np.random.default_rng(1), n_boot=50, device=stats.TORCH_CPU)
    assert json.dumps(to_jsonable(bench_t)) == json.dumps(to_jsonable(bench))


def test_battery_torch_path_matches_the_numpy_path(world: tuple) -> None:
    pytest.importorskip("torch")
    ctx, views, results = world
    ctx_t = dataclasses.replace(ctx, device=stats.TORCH_CPU, cache={})
    got, ref = vectors.run_vector_battery(views["mu_raw"], ctx_t), results["mu_raw"]
    assert got["backend"]["device"] == stats.TORCH_CPU and got["backend"]["path"].startswith("torch")
    assert got["failed_steps"] == [] and _statuses(got) == []
    for key in ("analogy_benchmark", "analogy_examples", "offsets", "fields", "clusters", "similarity"):
        assert json.dumps(to_jsonable(got[key])) == json.dumps(to_jsonable(ref[key])), key
    for q, row in ref["neighbors"].items():
        assert [t for t, _ in got["neighbors"][q]] == [t for t, _ in row]
    for key in ("directions_raw", "directions_whitened"):
        for field in ("cosine", "null_mean", "excess"):
            np.testing.assert_allclose(np.asarray(got[key][field], dtype=np.float64),
                                       np.asarray(ref[key][field], dtype=np.float64), atol=1e-8)
    g, g_ref = got["geometry"], ref["geometry"]
    for key in ("effective_rank", "isoscore", "participation_ratio"):
        assert g[key] == pytest.approx(g_ref[key], rel=1e-9)
    assert g["twonn"]["mle"] == pytest.approx(g_ref["twonn"]["mle"], rel=1e-5)
    assert json.dumps(to_jsonable(g["hubness"])) == json.dumps(to_jsonable(g_ref["hubness"]))


# ----------------------------------------------------------------------------------------------
# Frequency-stratified purity null
# ----------------------------------------------------------------------------------------------

def test_frequency_strata_are_equal_count_log_count_bins() -> None:
    lc = np.array([5.0, 1.0, 3.0, 3.0, 9.0, 0.0, 2.0, 7.0, 4.0, 6.0, 8.0, 1.5])
    s = stats.frequency_strata(lc, 4)
    np.testing.assert_array_equal(np.bincount(s), [3, 3, 3, 3])
    assert s[np.argmin(lc)] == 0 and s[np.argmax(lc)] == 3
    assert (np.diff(s[np.argsort(lc, kind="stable")]) >= 0).all()      # rarer rows never in a higher bin
    assert stats.frequency_strata(lc[:3], 10).tolist() == [2, 0, 1]      # fewer rows than bins: one row each
    with pytest.raises(ValueError, match="non-finite"):
        stats.frequency_strata([1.0, np.nan])


def test_stratified_chance_is_the_exact_within_stratum_permutation_mean() -> None:
    codes = np.array([0, 1, 0, 1, 1, 0, 1])
    strata = np.array([0, 0, 0, 1, 1, 1, 1])
    idx = np.array([[1, 2], [0, 3], [4, 0], [6, 5], [3, 2], [4, 6], [5, 1]])
    pos = [np.flatnonzero(strata == s) for s in (0, 1)]
    values = []
    for p0 in itertools.permutations(pos[0]):                           # all 3! x 4! within-stratum permutations
        for p1 in itertools.permutations(pos[1]):
            perm = codes.copy()
            perm[pos[0]], perm[pos[1]] = codes[list(p0)], codes[list(p1)]
            values.append((perm[idx] == perm[:, None]).mean())
    assert stats._stratified_chance(codes, idx, strata, 2) == pytest.approx(float(np.mean(values)), abs=1e-12)


def test_frequency_stratified_null_absorbs_frequency_coded_neighbors() -> None:
    rng = np.random.default_rng(43)
    n = 200
    lc = np.sort(rng.uniform(0.0, 10.0, n))
    X = np.column_stack([lc, 0.01 * rng.standard_normal(n)])            # neighbors = words of similar frequency
    frequent = rng.random(n) < 1.0 / (1.0 + np.exp(-1.5 * (lc - 5.0)))
    labels = np.where(frequent, "frequent", "rare")                     # the two fields differ only in frequency
    kwargs = dict(k=5, metric="euclidean", n_perm=300)
    both = stats.knn_label_purity(X, labels, rng=np.random.default_rng(0), null="both", log_count=lc, **kwargs)
    perm = stats.knn_label_purity(X, labels, rng=np.random.default_rng(0), **kwargs)
    for key in ("purity", "null_mean", "null_std", "z", "p", "chance", "per_label", "per_label_chance"):
        assert both[key] == perm[key], key                              # the permutation null is drawn first
    assert both["null"] == "both" and perm["null"] == "permutation"
    assert both["purity"] > 0.8 and both["z"] > 10                      # the uniform null calls this structure
    strat = both["frequency_stratified"]
    assert strat["n_strata"] == 10 and strat["n_perm"] == 300
    assert abs(strat["z"]) < 2 and strat["p"] > 0.05                    # frequency alone reaches it
    assert abs(strat["null_mean"] - strat["chance"]) < 4 * strat["null_std"] / math.sqrt(300)
    alone = stats.knn_label_purity(X, labels, rng=np.random.default_rng(0), null="frequency_stratified",
                                   log_count=lc, **kwargs)
    assert alone["null"] == "frequency_stratified" and alone["chance"] == strat["chance"]
    assert alone["n_strata"] == 10 and abs(alone["z"]) < 2
    with pytest.raises(ValueError, match="log_count"):
        stats.knn_label_purity(X, labels, rng=rng, null="both", **kwargs)
    with pytest.raises(ValueError, match="log counts"):
        stats.knn_label_purity(X, labels, rng=rng, null="both", log_count=lc[:-1], **kwargs)
    with pytest.raises(ValueError, match="unknown null"):
        stats.knn_label_purity(X, labels, rng=rng, null="uniform", **kwargs)


def test_field_clustering_reports_both_nulls_and_per_field_excess() -> None:
    rng = np.random.default_rng(19)
    dirs = _orthonormal(8, 3, rng).T * 5.0
    X = np.vstack([np.repeat(dirs, 8, axis=0) + 0.3 * rng.standard_normal((24, 8)), rng.standard_normal((6, 8))])
    view = VectorView("v", "v", np.arange(30), X)
    lexicon = {"f0": list(range(0, 8)), "f1": list(range(8, 16)), "f2": list(range(16, 24))}
    lc = rng.permutation(np.log1p(np.arange(30, 0, -1).astype(np.float64)))   # frequency unrelated to the fields
    with_counts = vectors.field_clustering(view, lexicon, np.random.default_rng(5), k=5, log_count=lc)
    without = vectors.field_clustering(view, lexicon, np.random.default_rng(5), k=5)
    for key in ("purity", "null_mean", "null_std", "z", "p"):
        assert with_counts["purity"][key] == without["purity"][key]
    assert without["purity"]["frequency_stratified"]["status"] == "skipped"
    strat = with_counts["purity"]["frequency_stratified"]
    assert strat["n_strata"] == 10 and strat["p"] < 0.05                 # separable fields beat the stratified null too
    ordered = vectors.field_clustering(view, lexicon, np.random.default_rng(5), k=5,
                                       log_count=np.log1p(np.arange(30, 0, -1).astype(np.float64)))
    assert ordered["purity"]["frequency_stratified"]["p"] > 0.05           # fields ordered by frequency: absorbed
    for f, rec in with_counts["per_field"].items():
        assert rec["n"] == 8 and rec["chance"] == pytest.approx(7 / 23)
        assert rec["purity_excess"] == pytest.approx((rec["purity"] - 7 / 23) / (1 - 7 / 23))
    with pytest.raises(ValueError, match="view rows"):
        vectors.field_clustering(view, lexicon, rng, k=5, log_count=lc[:10])


# ----------------------------------------------------------------------------------------------
# Macro-averaged analogy accuracy, concept-direction null, view styles, heatmap collapse
# ----------------------------------------------------------------------------------------------

def test_analogy_macro_average_weights_categories_equally() -> None:
    rng = np.random.default_rng(30)
    view, qs = _crossed_only_b_world(rng)
    bench = vectors.analogy_benchmark(view, qs, rng, n_boot=500)
    overall = bench["overall"]
    assert overall["methods"]["only_b"]["acc"]["mean"] == pytest.approx(45 / 120)   # micro: 90 + 30 questions
    macro = overall["macro"]
    assert macro["n_categories"] == 2 and macro["category_n"] == {"family": 90, "gram8-plural": 30}
    only_b = macro["methods"]["only_b"]["acc"]
    assert only_b["mean"] == pytest.approx(0.25) and only_b["ci_lo"] < 0.25 < only_b["ci_hi"]
    assert only_b["ci_method"] == "bootstrap over pairs"
    assert overall["largest_category"] == "family" and overall["largest_category_share"] == pytest.approx(0.75)
    for m in bench["methods"]:
        for stat in ("acc", "hits5", "mrr"):
            expected = np.mean([bench["categories"][c]["methods"][m][stat]["mean"] for c in ("family", "gram8-plural")])
            assert macro["methods"][m][stat]["mean"] == pytest.approx(expected)
    assert macro["honest"]["acc"]["mean"] == pytest.approx(
        np.mean([bench["categories"][c]["honest"]["acc"]["mean"] for c in ("family", "gram8-plural")]))
    assert bench["semantic"]["macro"]["methods"]["only_b"]["acc"]["mean"] == pytest.approx(0.5)
    zero = bench["syntactic"]["macro"]["methods"]["only_b"]["acc"]
    assert zero["mean"] == 0.0 and zero["ci_method"] == "Wilson on 6 pairs"
    assert "macro" not in bench["categories"]["family"]


def test_direction_null_removes_the_shared_source_cosine() -> None:
    rng = np.random.default_rng(44)
    n, D, m = 1500, 16, 12
    X = rng.standard_normal((n, D))
    X[:m] += 1.5 * rng.standard_normal(D)                               # shared sources with a common part
    delta = 3.0 * rng.standard_normal(D)
    c1, c2 = 200 + 110 * np.arange(m), 255 + 110 * np.arange(m)         # planted targets, > 50 ranks apart
    X[c1] = X[20:20 + m] + delta + 0.3 * rng.standard_normal((m, D))
    X[c2] = X[40:40 + m] + delta + 0.3 * rng.standard_normal((m, D))
    rels = {"u1": [(i, 100 + i) for i in range(m)],                     # shared sources, random targets:
            "u2": [(i, 150 + i) for i in range(m)],                     # no coupling, but -mean(x_a) is common
            "c1": [(20 + i, int(c1[i])) for i in range(m)],             # distinct sources, one shared offset
            "c2": [(40 + i, int(c2[i])) for i in range(m)]}
    view = VectorView("v", "v", np.arange(n), X)
    lc = np.log1p(np.arange(n, 0, -1).astype(np.float64))              # frequency rank = row index
    for whiten in (False, True):
        out = vectors.concept_direction_cosines(view, rels, whiten=whiten, rng=np.random.default_rng(7), log_count=lc)
        assert out["n_null"] == 200 and out["frequency_matched"] and "matched frequency" in out["null"]
        assert out["cosine"][0, 1] > 0.8 and out["null_mean"][0, 1] > 0.8       # the shared-source effect
        assert abs(out["z"][0, 1]) < vectors.DIRECTION_Z and not out["significant"][0, 1]
        assert abs(out["excess"][0, 1]) < 0.05
        assert out["significant"][2, 3] and out["excess"][2, 3] > 0.5            # a real common offset survives
        assert np.isnan(np.diag(out["excess"])).all() and np.isnan(np.diag(out["z"])).all()
        np.testing.assert_array_equal(out["significant"], out["significant"].T)
    plain = vectors.concept_direction_cosines(view, rels, whiten=False)
    assert "excess" not in plain and "null_mean" not in plain           # no rng: no null (and no extra draws)
    uniform = vectors.concept_direction_cosines(view, rels, whiten=False, rng=np.random.default_rng(7), n_null=20)
    assert not uniform["frequency_matched"] and "uniformly random" in uniform["null"]


def test_matched_replacements_stay_in_the_frequency_window_and_avoid_the_target() -> None:
    rng = np.random.default_rng(45)
    n, h = 100, 5
    lc = rng.permutation(np.log1p(np.arange(n, dtype=np.float64)))
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(-lc, kind="stable")] = np.arange(n)
    targets = np.array([int(np.argmin(rank)), int(np.argmax(rank)), int(np.flatnonzero(rank == 50)[0])])
    R = vectors._matched_replacements(targets, n, lc, rng, 3000, h)
    assert R.shape == (3000, 3) and (R != targets[None, :]).all()
    for j, t in enumerate(targets):
        lo = int(np.clip(rank[t] - h, 0, n - (2 * h + 1)))              # window shifted inward at the ends
        got = rank[R[:, j]]
        assert got.min() >= lo and got.max() <= lo + 2 * h
        assert np.unique(R[:, j]).size == 2 * h                         # every other slot of the window is used
    U = vectors._matched_replacements(targets, n, None, rng, 500, h)
    assert (U != targets[None, :]).all() and U.min() >= 0 and U.max() < n


def test_view_style_gives_every_known_view_a_distinct_family_coded_style() -> None:
    styles = {name: vectors.view_style(name) for name in vectors.VIEW_STYLE}
    assert len({(s["color"], s["marker"], s["filled"]) for s in styles.values()}) == len(styles)
    for name, s in styles.items():
        assert s["color"] == vectors.FAMILY_COLORS[vectors.VIEW_STYLE[name]["family"]]
        assert (s["face"] == "white") != s["filled"]
    assert styles["mu_raw"]["filled"] and not styles["mu_sigma_units"]["filled"]
    assert styles["decode_W"]["color"] == styles["decode_W_causal"]["color"] != styles["mu_raw"]["color"]
    assert styles["phi_raw"]["filled"] and not styles["phi_log"]["filled"]
    layer = VectorView("layer_3", "layer 3", np.arange(2), np.eye(2), group="contextual")
    assert vectors.view_style("layer_3", layer) == vectors.view_style("layer_3", layer)   # stable
    assert vectors.view_style("layer_3", layer)["color"] == vectors.FAMILY_COLORS["contextual"]
    assert vectors.view_style("mystery")["color"] in {OKABE_ITO[5], OKABE_ITO[6], OKABE_ITO[7]}


def test_analogy_category_heatmap_collapses_all_zero_views(world: tuple) -> None:
    import matplotlib.pyplot as plt

    ctx, views, results = world
    zero = copy.deepcopy(results["w2_chart"])
    bench = zero["analogy_benchmark"]
    records = [bench[g] for g in ("overall", "semantic", "syntactic")] + list(bench["categories"].values())
    for rec in records + [r["macro"] for r in records if "macro" in r]:
        acc = _get_path(rec, "methods", "3cosadd", "acc")
        if acc is not None:
            acc["mean"] = 0.0
    fig = vectors._compare_analogy_categories({"mu_raw": results["mu_raw"], "w2_chart": zero}, views, ctx,
                                              ["mu_raw", "w2_chart"])
    try:
        assert fig.axes[0].images[0].get_array().shape[1] == 1           # only mu raw in the main heatmap
        assert fig.axes[1].get_xticklabels()[0].get_text() == "1 view"
        assert any("Collapsed" in t.get_text() and "W2 chart" in t.get_text().replace("\n", " ")
                   for t in fig.texts)
    finally:
        plt.close(fig)


def _get_path(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node
