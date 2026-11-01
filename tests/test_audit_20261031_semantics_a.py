r"""Audit 2026-10-31, package semantics-a: the semantic statistics and the vector battery.

Each test pins one finding (lane id in the section heading): the W2 and Fisher-Rao readings of the
``w2_chart`` and ``log_var`` views on the frame-diagonal family and the block-Cholesky table
(semantics-stats-1), integer ties of the kNN-purity null (semantics-stats-2), the +1-corrected paired
Delta-rho p (semantics-stats-4), shared Spearman resamples (semantics-stats-7), Wilson and Beta intervals of
small and boundary analogy categories (semantics-vectors-1), the word-disjoint LOO pair accuracy
(semantics-vectors-2), the exact re-pairing null of small relations (semantics-vectors-3), the LayerNorm of
the reference-coordinate block (semantics-vectors-4), contraction stems (semantics-vectors-5), the prepared
rows computed once per view (semantics-vectors-6), the similarity notes (semantics-vectors-7), the
``benchmark_dir`` option (semantics-vectors-8), the stale mixer sentence (semantics-vectors-9) and the
similarity totals (semantics-vectors-10). Tiny CPU models and synthetic arrays only.
"""
from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import tests.test_semantics_core as sc
import tests.test_semantics_gaussians as tg
import tests.test_semantics_vectors as tv
from tests.profiles import reference_k8_config, reference_k8_values, tiny_config
from vfe3.config import VFE3Config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.model.model import VFEModel
from vfe3.semantics import frames as F
from vfe3.semantics import gaussians as G
from vfe3.semantics import report as R
from vfe3.semantics import resources, stats, vectors
from vfe3.semantics.common import AnalogyQuestion, VectorView, diagonal_redundancy_status, to_jsonable
from vfe3.semantics.run import CORE_DEFAULT_OPTIONS, prepare_context
from vfe3.semantics.vocab import custom_vocab, vocab_for_run


@pytest.fixture(autouse=True)
def _low_dpi_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The sections' style at 40 dpi with one font (glyph measurement dominates the figure time)."""
    from vfe3.semantics import common

    original = common.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"], "font.sans-serif": ["DejaVu Sans"]})

    for module in (common, G, vectors):
        monkeypatch.setattr(module, "apply_style", low_dpi_style)


def _options(**over: Any) -> Dict[str, Any]:
    return {**R.DEFAULT_OPTIONS, "min_count": 1, "threads": 1, **over}


# ----------------------------------------------------------------------------------------------
# semantics-stats-1: w2_chart and log_var views on the frame-diagonal family and the block-Cholesky table
# ----------------------------------------------------------------------------------------------

def _frame_diagonal_run(tmp_path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = reference_k8_config(family="gaussian_frame_diagonal", predictive_readout="kernel",
                                  block_mlp_mode="coordinate", block_mlp_covariance="passthrough",
                                  decode_mode="diagonal_chunked", norm_type_block="layernorm", untie_decode_bank=True,
                                  pos_phi="none", vocab_size=sc.V)
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():                                      # frames far from orthogonal, distinct variances
        bank = model.prior_bank
        bank.phi_embed.copy_(0.4 * torch.randn(bank.phi_embed.shape, generator=g))
        bank.sigma_log_embed.copy_(torch.randn(bank.sigma_log_embed.shape, generator=g))
        bank.mu_embed.copy_(torch.randn(bank.mu_embed.shape, generator=g))
    return sc._load(sc._save_run(tmp_path, "frame_diag", cfg, model), cache_dir=tmp_path / "empty")


def _block_cholesky_run(tmp_path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = tiny_config(embed_dim=6, n_heads=2, vocab_size=sc.V, max_seq_len=5, n_layers=1,
                          family="gaussian_block", decode_mode="full_chunked", untie_decode_bank=True,
                          block_coordinates="reference", decode_coordinates="reference",
                          norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                          block_mlp_mode="coordinate", content_link_chart="exponential", pos_phi="none",
                          gauge_transport="off", prior_covariance_table="block_cholesky")
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():                                      # off-diagonal within-head covariance
        model.prior_bank.sigma_chol_embed.add_(0.6 * torch.randn(model.prior_bank.sigma_chol_embed.shape, generator=g))
        model.prior_bank.mu_embed.copy_(torch.randn(model.prior_bank.mu_embed.shape, generator=g))
    return sc._load(sc._save_run(tmp_path, "chol", cfg, model), cache_dir=tmp_path / "empty")


def _bures_sq(A: np.ndarray, B: np.ndarray) -> float:
    r"""Squared Bures distance :math:`\mathrm{tr}A + \mathrm{tr}B - 2\,\mathrm{tr}(A^{1/2}BA^{1/2})^{1/2}`."""
    from scipy import linalg

    root = linalg.sqrtm(A).real
    return float(np.trace(A) + np.trace(B) - 2.0 * np.trace(linalg.sqrtm(root @ B @ root).real))


def test_frame_diagonal_views_drop_the_wasserstein_and_fisher_rao_readings(tmp_path: Path) -> None:
    run = _frame_diagonal_run(tmp_path)
    assert run.reference_variances is True
    ctx = prepare_context(run, tmp_path / "out", options=_options(views=("mu_sigma_units", "w2_chart", "log_var")))
    views = R.build_views(run, ctx)
    assert set(views) == {"mu_sigma_units", "log_var"}
    skip = [n for n in ctx.cache["view_notes"] if n.startswith("view w2_chart skipped")]
    assert len(skip) == 1 and "gaussian_frame_diagonal" in skip[0] and "not applicable" in skip[0]
    lv = views["log_var"]
    assert lv.label == "log c (reference)"
    assert "not Fisher–Rao distances of the stored priors" in lv.note
    assert "the Fisher–Rao distance is" not in lv.note
    np.testing.assert_allclose(lv.X, run.log_var(ctx.candidate_ids), rtol=1e-6)
    units = views["mu_sigma_units"].note
    assert "reference variances" in units and "not invariant under it" in units


def test_block_cholesky_views_chart_the_diagonal_marginals_and_say_so(tmp_path: Path) -> None:
    run = _block_cholesky_run(tmp_path)
    assert run.prior_covariance_blocks is True
    ctx = prepare_context(run, tmp_path / "out", options=_options(views=("w2_chart", "log_var")))
    views = R.build_views(run, ctx)
    assert set(views) == {"w2_chart", "log_var"}
    w2 = views["w2_chart"]
    assert "diagonal marginals N(μ_v, diag C_v)" in w2.note and "not between the stored priors" in w2.note
    assert "between the diagonal priors" not in w2.note
    assert "Fisher–Rao distance between those marginals, not between the stored priors" in views["log_var"].note
    ids = ctx.candidate_ids
    mu, lv = run.mu(ids), run.log_var(ids)
    log_d0 = -0.5 * lv.mean(axis=0)
    d0 = np.exp(log_d0)
    X = w2.X.astype(np.float64)
    C = torch.cat(list(run._block_cholesky_chunks(ids))).numpy()      # (n, H, d, d) stored head blocks
    from scipy.linalg import block_diag

    gaps: List[float] = []
    for a, b in ((0, 1), (2, 5), (3, 7), (4, 9)):
        chart = float(np.sum((X[a] - X[b]) ** 2))
        marginal = float(G.w2_sq_diag(mu[a] * d0, lv[a] + 2.0 * log_d0, mu[b] * d0, lv[b] + 2.0 * log_d0))
        assert chart == pytest.approx(marginal, rel=1e-5)               # what the note now claims
        D = np.diag(d0)
        exact = float(np.sum((d0 * (mu[a] - mu[b])) ** 2)) + _bures_sq(D @ block_diag(*C[a]) @ D,
                                                                        D @ block_diag(*C[b]) @ D)
        gaps.append(abs(chart - exact) / exact)
    assert max(gaps) > 1e-3                                             # not the W2 of the stored priors


def test_diagonal_table_keeps_the_wasserstein_and_fisher_rao_notes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    mu_t, lv_t = rng.normal(size=(sc.V, 4)), rng.normal(size=(sc.V, 4))
    run = SimpleNamespace(cfg=SimpleNamespace(decode_mode="other", max_seq_len=8), frames_active=False,
                          decode_kind="unsupported", mu=lambda ids: mu_t[np.asarray(ids)],
                          log_var=lambda ids: lv_t[np.asarray(ids)])
    ctx = tv._ctx(tmp_path, 12, options={"views": ("w2_chart", "log_var", "mu_sigma_units")})
    views = R.build_views(run, ctx)
    assert views["w2_chart"].label == "W2 chart (μ, √v)"
    assert "the 2-Wasserstein distance between the diagonal priors in D0 units" in views["w2_chart"].note
    assert views["log_var"].label == "log σ²"
    assert "for equal means the Fisher–Rao distance is ‖Δ log σ²‖/√2" in views["log_var"].note
    assert "Invariant under μ → Dμ, log σ² → log σ² + 2 log D" in views["mu_sigma_units"].note


# ----------------------------------------------------------------------------------------------
# semantics-stats-2: the kNN-purity null compares integer agreeing-edge counts
# ----------------------------------------------------------------------------------------------

def test_complete_neighbor_graphs_have_permutation_p_one_and_no_z() -> None:
    rng = np.random.default_rng(3)
    for n in range(4, 12):
        for sizes in ((n // 2, n - n // 2), (n // 3, n // 3, n - 2 * (n // 3))):
            if min(sizes) < 1:
                continue
            labels = [f"f{i}" for i, s in enumerate(sizes) for _ in range(s)]
            X = rng.standard_normal((n, 3))
            res = stats.knn_label_purity(X, labels, k=10, metric="euclidean", rng=np.random.default_rng(n),
                                         n_perm=200, null="both", log_count=rng.uniform(0.0, 5.0, size=n))
            assert res["k"] == n - 1                                     # every other row is a neighbor
            assert res["p"] == 1.0 and math.isnan(res["z"]) and res["null_std"] == 0.0, (n, sizes)
            assert res["purity"] == res["null_mean"]
            strat = res["frequency_stratified"]
            assert strat["p"] == 1.0 and math.isnan(strat["z"]) and strat["null_std"] == 0.0


def test_a_draw_that_reproduces_the_observed_count_is_a_tie() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((14, 4))
    labels = list(rng.choice(["a", "b", "c"], size=14))
    res = stats.knn_label_purity(X, labels, k=3, metric="cosine", rng=np.random.default_rng(9), n_perm=300)
    codes, _ = stats._label_codes(labels)
    idx, _ = stats.knn(stats._f64(X), 3, "cosine")
    obs = int((codes[idx] == codes[:, None]).sum())
    replay = np.random.default_rng(9)
    draws = []
    for _ in range(300):
        perm = replay.permutation(codes)
        draws.append(int((perm[idx] == perm[:, None]).sum()))
    draws = np.array(draws)
    assert (draws == obs).any()                                          # ties occur in this draw set
    assert res["p"] == (1 + int((draws >= obs).sum())) / 301
    assert res["purity"] == obs / (14 * 3)
    assert res["null_mean"] == pytest.approx(draws.mean() / 42, rel=1e-15)
    assert res["z"] == pytest.approx((obs - draws.mean()) / draws.std(ddof=1), rel=1e-12)


# ----------------------------------------------------------------------------------------------
# semantics-stats-4: the paired Delta-rho bootstrap p is never 0
# ----------------------------------------------------------------------------------------------

def test_paired_delta_p_is_floored_at_two_over_b_plus_one() -> None:
    rng = np.random.default_rng(11)
    m = 200
    human = rng.normal(size=m)
    strong = human + 0.05 * rng.normal(size=m)
    unrelated = rng.normal(size=m)
    out = G._paired_spearman_delta(strong, unrelated, human, rng=np.random.default_rng(3), n_boot=1000)
    assert out["n_boot"] == 1000 and out["ci_lo"] > 0.0                 # every resample favors the strong scorer
    assert out["p"] == 2.0 / 1001
    assert stats.holm_adjust([out["p"]] * 8)[0] == pytest.approx(16.0 / 1001)
    assert G._delta_summary(0.1, np.full(9, 0.2), 5)["p"] == pytest.approx(0.2)
    assert G._delta_summary(0.0, np.zeros(9), 5)["p"] == 1.0
    assert G._delta_summary(0.0, np.array([np.nan]), 5)["p"] is None


# ----------------------------------------------------------------------------------------------
# semantics-stats-7: every Gaussian scorer of a benchmark shares one set of pair resamples
# ----------------------------------------------------------------------------------------------

def test_shared_resamples_reproduce_spearman_ci_for_one_scorer() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x + rng.normal(size=60)
    boot = G._bootstrap_spearman_rhos({"s": x}, y, np.random.default_rng(9), n_boot=1000)
    assert G._spearman_summary(x, y, boot["s"]) == stats.spearman_ci(x, y, np.random.default_rng(9))


def test_paired_delta_on_shared_resamples_matches_the_previous_draws() -> None:
    from scipy.stats import rankdata

    rng = np.random.default_rng(1)
    m = 80
    human = rng.normal(size=m)
    score, ref = human + rng.normal(size=m), human + 2.0 * rng.normal(size=m)
    old = np.random.default_rng(4)                                       # the previous implementation, verbatim
    idx = old.integers(0, m, size=(1000, m))
    rh = rankdata(human[idx], axis=1)
    deltas = G._row_pearson(rankdata(score[idx], axis=1), rh) - G._row_pearson(rankdata(ref[idx], axis=1), rh)
    deltas = deltas[np.isfinite(deltas)]
    new = G._paired_spearman_delta(score, ref, human, rng=np.random.default_rng(4))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    assert (new["ci_lo"], new["ci_hi"], new["n_boot"]) == (float(lo), float(hi), deltas.size)
    B = deltas.size
    assert new["p"] == min(1.0, 2.0 * min((1 + int((deltas <= 0).sum())) / (B + 1),
                                          (1 + int((deltas >= 0).sum())) / (B + 1)))


def test_similarity_ranks_each_scorer_once_per_resample_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scipy.stats as sps

    run = tg._FakeRun(frames_active=False)
    ctx = tg._context(run, tmp_path)
    ids = np.asarray(ctx.candidate_ids, dtype=np.int64)
    row_of = {int(t): r for r, t in enumerate(ids)}
    mu, lv = run.mu(ids), run.log_var(ids)
    mu_d0, log_d0 = G.sigma_units(mu, lv)
    calls: List[Any] = []
    real = sps.rankdata

    def counting(a: Any, *args: Any, **kwargs: Any) -> Any:
        if np.ndim(a) == 2:
            calls.append(np.shape(a))
        return real(a, *args, **kwargs)

    monkeypatch.setattr(sps, "rankdata", counting)
    out = G._similarity(ctx, row_of, mu, mu_d0, lv + 2.0 * log_d0, np.random.default_rng(0))
    assert out["simlex999"]["status"] == "ok" and out["wordsim353"]["status"] == "skipped"
    assert len(calls) == 5 * (1 + len(G._SCORERS))                       # B = 1000 in blocks of 200 (was 60 passes)
    again = G._similarity(ctx, row_of, mu, mu_d0, lv + 2.0 * log_d0, np.random.default_rng(0))
    assert json.dumps(to_jsonable(again)) == json.dumps(to_jsonable(out))  # reproducible for a fixed seed
    per = out["simlex999"]["scorers"]
    for key, _ in G._SCORERS:
        assert per[key]["n_boot_valid"] == 1000 and per[key]["ci_lo"] <= per[key]["ci_hi"]
        if key != "cos_mu":
            assert per[key]["delta_vs_cos_mu"]["p"] >= 2.0 / 1001


def test_shared_resample_intervals_agree_in_distribution_with_per_scorer_draws() -> None:
    rng = np.random.default_rng(2)
    m = 300
    human = rng.normal(size=m)
    scores = {"a": human + rng.normal(size=m), "b": human + 3.0 * rng.normal(size=m)}
    boot = G._bootstrap_spearman_rhos(scores, human, np.random.default_rng(10), n_boot=2000)
    for key, s in scores.items():
        shared = G._spearman_summary(s, human, boot[key])
        alone = stats.spearman_ci(s, human, np.random.default_rng(20), n_boot=2000)
        assert shared["rho"] == alone["rho"]
        assert abs(shared["ci_lo"] - alone["ci_lo"]) < 0.03 and abs(shared["ci_hi"] - alone["ci_hi"]) < 0.03


# ----------------------------------------------------------------------------------------------
# semantics-vectors-1: small and boundary analogy categories
# ----------------------------------------------------------------------------------------------

def _two_pair_world() -> tuple:
    r"""The audit probe's world: 'big' (8 planted pairs, every question answered) and 'tiny' (2 pairs, 1 of 2)."""
    rng = np.random.default_rng(1)
    n, D = 120, 16
    X = 5.0 * rng.standard_normal((n, D))
    offset = 3.0 * rng.standard_normal(D)
    for i in range(8):
        X[30 + i] = X[10 + i] + offset + 1.5 * rng.standard_normal(D)
    questions = [AnalogyQuestion("big", 10 + i, 30 + i, 10 + j, 30 + j) for i in range(8) for j in range(8) if i != j]
    delta = 4.0 * rng.standard_normal(D)
    X[70] = X[60] + delta
    X[71] = X[61] + delta + 0.05 * rng.standard_normal(D)
    X[80] = X[71] - X[61] + X[60]                                        # exactly the (Q, P) target: 1 hit of 2
    questions += [AnalogyQuestion("tiny", 60, 70, 61, 71), AnalogyQuestion("tiny", 61, 71, 60, 70)]
    return VectorView("v", "v", np.arange(n), X, metric="euclidean"), questions


def test_a_two_pair_category_gets_a_wilson_interval_and_feeds_the_macro() -> None:
    view, questions = _two_pair_world()
    res = vectors.analogy_benchmark(view, questions, np.random.default_rng(2), n_boot=1000)
    tiny = res["categories"]["tiny"]
    assert tiny["n_units"] == 2
    acc = tiny["methods"]["3cosadd"]["acc"]
    lo, hi = stats.wilson_interval(1, 2)
    assert acc["mean"] == 0.5 and acc["ci_method"] == "Wilson on 2 pairs"
    assert (acc["ci_lo"], acc["ci_hi"]) == (pytest.approx(lo), pytest.approx(hi)) and acc["ci_hi"] > acc["ci_lo"]
    assert tiny["methods"]["3cosadd"]["mrr"]["ci_method"].startswith("none (fewer than 3")
    big = res["categories"]["big"]["methods"]["3cosadd"]["acc"]
    assert big["mean"] == 1.0 and big["ci_method"] == "Wilson on 8 pairs"
    macro = res["overall"]["macro"]["methods"]["3cosadd"]["acc"]
    assert macro["mean"] == pytest.approx(0.75)
    assert macro["ci_lo"] < 0.75 < macro["ci_hi"] and macro["ci_hi"] - macro["ci_lo"] > 0.2   # was [0.75, 0.75]
    assert macro["ci_method"].startswith("bootstrap over pairs") and "2 categories" in macro["ci_method"]
    assert macro["n_boot_valid"] == 1000                                 # the small category no longer voids draws
    assert math.isnan(res["overall"]["macro"]["methods"]["3cosadd"]["mrr"]["ci_lo"])
    json.dumps(to_jsonable(res), allow_nan=False)


def _only_b_world(n_family: int, n_zero: int) -> tuple:
    r"""Euclidean world whose ONLY-B hit depends on the b-pair alone (as ``tv._crossed_only_b_world``)."""
    rng = np.random.default_rng(31)
    D, rows, questions = 8, [], []

    def category(name: str, good: List[bool]) -> None:
        ids = []
        for is_good in good:
            w = rng.standard_normal(D) * 6.0
            u = rng.standard_normal((2, D))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            rows.extend([w, w + (0.1 if is_good else 0.6) * u[0]])
            ids.append((len(rows) - 2, len(rows) - 1))
            if not is_good:
                rows.append(w + 0.05 * u[1])                             # decoy, nearer than w*
        questions.extend(AnalogyQuestion(name, ids[i][0], ids[i][1], ids[j][0], ids[j][1])
                         for i in range(len(ids)) for j in range(len(ids)) if i != j)

    category("family", [True, False] * (n_family // 2))
    category("gram8-plural", [False] * n_zero)
    return VectorView("e", "e", np.arange(len(rows)), np.array(rows), metric="euclidean"), questions


def test_the_macro_interval_carries_a_zero_hit_category() -> None:
    view, questions = _only_b_world(40, 6)
    res = vectors.analogy_benchmark(view, questions, np.random.default_rng(7), n_boot=1000)
    family = res["categories"]["family"]["methods"]["only_b"]["acc"]
    zero = res["categories"]["gram8-plural"]["methods"]["only_b"]["acc"]
    assert family["mean"] == pytest.approx(0.5) and zero["mean"] == 0.0 and zero["ci_method"] == "Wilson on 6 pairs"
    macro = res["overall"]["macro"]["methods"]["only_b"]["acc"]
    assert macro["mean"] == pytest.approx(0.25) and macro["ci_lo"] < 0.25 < macro["ci_hi"]
    width = macro["ci_hi"] - macro["ci_lo"]
    assert width >= stats.wilson_interval(0, 6)[1] / 2                  # the lane's bound; about 0.155 before
    assert width > 0.5 * (family["ci_hi"] - family["ci_lo"]) + 0.03     # wider than the varying category alone


# ----------------------------------------------------------------------------------------------
# semantics-vectors-2: the LOO pair accuracy holds out the pairs that share a pole word
# ----------------------------------------------------------------------------------------------

def test_loo_pair_accuracy_is_chance_on_noise_for_every_shipped_axis() -> None:
    shipped = resources.load_relations()["axes"]
    words = sorted({w for axis in shipped.values() for pair in axis["pairs"] for w in pair})
    token = {w: i for i, w in enumerate(words)}
    axes = {name: {"pairs": [(token[a], token[b]) for a, b in axis["pairs"]], "words": []}
            for name, axis in shipped.items()}
    rng = np.random.default_rng(0)
    draws = 300
    loo: Dict[str, List[float]] = {name: [] for name in axes}
    for _ in range(draws):
        view = VectorView("noise", "noise", np.arange(len(words)), rng.standard_normal((len(words), 64)))
        for name, axis in axes.items():
            loo[name].append(vectors.semantic_projection(view, axis)["loo_pair_accuracy"])
    for name, values in loo.items():
        v = np.asarray(values, dtype=np.float64)
        se = v.std(ddof=1) / math.sqrt(draws)
        assert abs(v.mean() - 0.5) < 3.0 * se + 1e-12, (name, v.mean(), se)   # size read 0.71 before the fix
    size = vectors.semantic_projection(VectorView("noise", "noise", np.arange(len(words)),
                                                  rng.standard_normal((len(words), 64))), axes["size"])
    # small:big shares 'small' with small:large and 'big' with little:big, so its held-out axis uses 4 pairs
    assert size["loo_axis_pairs"] == [4, 5, 6, 5, 6, 6, 6] and size["loo_n_pairs"] == 7


def test_loo_pair_accuracy_skips_a_pair_with_no_word_disjoint_partner() -> None:
    X = np.random.default_rng(1).standard_normal((4, 5))
    view = VectorView("v", "v", np.arange(4), X, metric="euclidean")
    res = vectors.semantic_projection(view, {"pairs": [(0, 1), (0, 2), (0, 3)], "words": []})
    assert res["loo_pair_accuracy"] is None and res["loo_n_pairs"] == 0 and res["loo_axis_pairs"] == []


# ----------------------------------------------------------------------------------------------
# semantics-vectors-3: the re-pairing null enumerates every m-cycle of a small relation
# ----------------------------------------------------------------------------------------------

def _single_cycles(m: int) -> List[tuple]:
    out = []
    for perm in itertools.permutations(range(m)):
        seen, i = {0}, perm[0]
        while i != 0:
            seen.add(i)
            i = perm[i]
        if len(seen) == m:
            out.append(perm)
    return out


def test_all_cyclic_derangements_are_exactly_the_m_cycles() -> None:
    for m in (3, 4, 5, 6):
        rows = vectors._all_cyclic_derangements(m)
        assert rows.shape == (math.factorial(m - 1), m)
        assert sorted(map(tuple, rows.tolist())) == sorted(_single_cycles(m))


def test_small_relations_report_the_exact_re_pairing_p() -> None:
    rng = np.random.default_rng(16)
    n, D = 40, 12
    X = rng.standard_normal((n, D))
    delta = 3.0 * rng.standard_normal(D)
    for i in range(8):
        X[20 + i] = X[i] + delta + 0.05 * rng.standard_normal(D)
    view = VectorView("e", "e", np.arange(n), X, metric="euclidean")
    rels = {f"m{m}": [(i, 20 + i) for i in range(m)] for m in (3, 4, 5, 8)}
    res = vectors.offset_consistency(view, rels, np.random.default_rng(0), n_perm=500)
    Xs = view.X.astype(np.float64)
    for m in (3, 4, 5):
        r = res["relations"][f"m{m}"]
        assert r["null_method"] == "exact" and r["n_null"] == math.factorial(m - 1)
        assert r["p"] == pytest.approx(1.0 / (math.factorial(m - 1) + 1))  # planted: no cycle reaches the OCS
        ocs = []
        for perm in _single_cycles(m):                                   # brute force over every m-cycle
            d = Xs[[20 + perm[i] for i in range(m)]] - Xs[:m]
            U = d / np.linalg.norm(d, axis=1, keepdims=True)
            ocs.append(((U @ U.T).sum() - m) / (m * (m - 1)))
        assert r["null_mean"] == pytest.approx(float(np.mean(ocs)), abs=1e-12)
        assert r["null_std"] == pytest.approx(float(np.std(ocs)), abs=1e-12)
    assert res["relations"]["m3"]["p"] >= 1.0 / 3.0                     # was 0.002 with 500 draws
    r8 = res["relations"]["m8"]
    assert r8["null_method"] == "monte_carlo" and r8["n_null"] == 500 and r8["p"] == pytest.approx(1.0 / 501)


# ----------------------------------------------------------------------------------------------
# semantics-vectors-4 (and semantics-contextual-4): the LayerNorm of the reference-coordinate block
# ----------------------------------------------------------------------------------------------

def test_reference_coordinate_layernorm_is_not_reported_broken() -> None:
    raw = reference_k8_values()
    local = VFE3Config(**raw)
    assert (local.norm_type_block, local.block_coordinates) == ("layernorm", "local")
    status, reason = diagonal_redundancy_status(local)
    assert status == "broken" and reason.startswith("a LayerNorm (affine or not) normalizes")   # unchanged
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        switched = VFE3Config(**{**raw, **fully_gauge_covariant_overrides(raw)})
    assert (switched.norm_type_block, switched.block_coordinates) == ("layernorm", "reference")
    assert gauge_covariance_failures(switched) == []
    status, reason = diagonal_redundancy_status(switched)
    assert status == "assumed" and "LayerNorm conjugated by the frame" in reason
    assert R._d_clause(switched).startswith("assumed, not verified") and "not an exact symmetry" not in R._d_clause(switched)
    assert "assumed, not verified" in F._diagonal_note(status, reason)
    probed = SimpleNamespace(gauge_group="block_glk", attention_mode="projected_predictive",
                             norm_type_block="layernorm", block_coordinates="reference")
    assert diagonal_redundancy_status(probed) == (
        "assumed", "not probed for a LayerNorm conjugated by the frame (block_coordinates='reference'; covariant "
                   "by construction, not measured)")


# ----------------------------------------------------------------------------------------------
# semantics-vectors-5: contraction stems and pieces
# ----------------------------------------------------------------------------------------------

def test_gpt2_contraction_stems_leave_the_word_mask_and_are_function_words() -> None:
    encoding = sc._gpt2_encoding()
    vocab = vocab_for_run(None, {"tokenizer_encoding": "gpt2", "dataset": "wikitext-103"}, 50257)
    assert encoding.encode_ordinary(" ca n't") == [1275, 299, 470] and encoding.encode_ordinary(" wo n't")[0] == 24486
    mask = vocab.word_mask()
    codes, names = vocab.token_classes()
    for token in (299, 1275, 24486):                                     # " n", " ca", " wo"
        assert not mask[token] and names[codes[token]] == "function word", vocab.strings[token]
    assert names[codes[470]] == "function word"                          # "'t"
    king, do = encoding.encode_ordinary(" king")[0], encoding.encode_ordinary(" do")[0]
    assert mask[king] and names[codes[king]] == "content word"
    assert mask[do] and names[codes[do]] == "function word"


def test_word_level_contraction_pieces() -> None:
    vocab = custom_vocab(["<unk>", "ca", "wo", "n't", "'s", "can", "Ca"])
    assert vocab.word_mask().tolist() == [False, False, False, False, False, True, True]
    codes, names = vocab.token_classes()
    assert [names[c] for c in codes[1:]] == ["function word"] * 5 + ["capitalized word"]
    other = custom_vocab(["ca", "wo"], english=False)
    assert other.word_mask().tolist() == [True, True]


# ----------------------------------------------------------------------------------------------
# semantics-vectors-6: one prepare() per view for the battery and its figures
# ----------------------------------------------------------------------------------------------

def test_battery_and_per_view_figures_prepare_the_view_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, views = tv._synthetic_world(tmp_path)
    ctx = dataclasses.replace(ctx, options={**ctx.options, "umap": False})
    view = views["mu_raw"]
    calls: List[bool] = []
    real = vectors._prepare_array

    def counting(X: Any, metric: str, center: bool) -> np.ndarray:
        if X is view.X:
            calls.append(bool(center))
        return real(X, metric, center)

    monkeypatch.setattr(vectors, "_prepare_array", counting)
    for _, v in vectors.views_with_prepared_rows({"mu_raw": view}):
        battery = vectors.run_vector_battery(v, ctx)
        vectors.plot_view_embedding(v, ctx, battery)
        vectors.plot_offsets(v, ctx, ctx.relations, battery=battery)
        np.testing.assert_array_equal(v._prepared[True], real(v.X, v.metric, True))   # never written
    assert calls == [True] and view._prepared is None                    # 10 full-view prepares before
    assert battery["failed_steps"] == []


def test_cached_battery_equals_the_uncached_battery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, views = tv._synthetic_world(tmp_path)
    cached = vectors.run_vector_battery(views["mu_raw"], ctx)
    monkeypatch.setattr(vectors, "prepared_rows_cache", lambda view: contextlib.nullcontext(view))
    uncached = vectors.run_vector_battery(views["mu_raw"], ctx)

    def text(battery: Dict[str, Any]) -> str:
        return json.dumps(to_jsonable({k: v for k, v in battery.items() if k != "seconds"}), allow_nan=False)

    assert text(cached) == text(uncached)


# ----------------------------------------------------------------------------------------------
# semantics-vectors-7, -9, -10: notes and totals
# ----------------------------------------------------------------------------------------------

def test_similarity_records_state_the_independent_pair_bootstrap_and_the_benchmark_total(tmp_path: Path) -> None:
    ctx, views = tv._synthetic_world(tmp_path)
    resolved = len(ctx.similarity["simlex999"])
    coverage = {**ctx.coverage, "similarity": {"simlex999": {"available": True, "n_total": 999, "n_resolved": resolved}}}
    battery = vectors.run_vector_battery(views["w2_chart"], dataclasses.replace(ctx, coverage=coverage))
    sim = battery["similarity"]["simlex999"]
    assert (sim["n_pairs_resolved"], sim["n_pairs_total"]) == (resolved, 999)
    assert "independent" in sim["ci_method"] and sim["ci_method"] == vectors.SIMILARITY_CI_METHOD
    assert battery["coverage"]["similarity_resolved"] == {"simlex999": resolved}
    assert battery["coverage"]["similarity_total"] == {"simlex999": 999}
    plain = vectors.run_vector_battery(views["w2_chart"], ctx)          # no benchmark coverage: the resolved count
    assert plain["similarity"]["simlex999"]["n_pairs_total"] == resolved
    assert "independent" in R._FIGURE_CAPTIONS["similarity"]
    table = G._similarity_table({"simlex999": {"status": "skipped", "n": 3, "reason": "few pairs"}})
    assert "treats the pairs as independent" in table


def test_natural_parameters_docstring_names_no_mixer() -> None:
    assert "mixer" not in R._natural_parameters.__doc__


# ----------------------------------------------------------------------------------------------
# semantics-vectors-8: the benchmark_dir option
# ----------------------------------------------------------------------------------------------

def test_benchmark_dir_feeds_both_loaders_and_the_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "gensim_test_data_dir", lambda: None)   # as on an interpreter without gensim
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / resources.GOOGLE_ANALOGY_FILE).write_text(": capital-world\nFrance Paris Germany Berlin\n", encoding="utf-8")
    (bench / resources.SIMILARITY_FILES["simlex999"]).write_text(
        "word1\tword2\tSimLex999\nking\tqueen\t8.0\ndog\tcat\t5.0\nred\tblue\t3.0\n", encoding="utf-8")
    run = SimpleNamespace(vocab=custom_vocab(sc.STRINGS), counts=sc.COUNTS, counts_source="unigram_log_prior")
    assert R.DEFAULT_OPTIONS["benchmark_dir"] is None and CORE_DEFAULT_OPTIONS["benchmark_dir"] is None
    ctx = prepare_context(run, tmp_path / "out", options={"threads": 1, "min_count": 1, "benchmark_dir": str(bench)})
    google = ctx.coverage["google_analogies"]
    assert google["available"] and google["n_total"] == 1 and len(ctx.analogies) == 1
    assert google["source"] == str(bench / resources.GOOGLE_ANALOGY_FILE)
    simlex = ctx.coverage["similarity"]["simlex999"]
    assert simlex["available"] and simlex["n_total"] == 3 and simlex["source"] == str(bench / "simlex999.txt")
    assert len(ctx.similarity["simlex999"]) == 3
    assert ctx.coverage["similarity"]["wordsim353"] == {"available": False, "n_total": 0, "n_resolved": 0,
                                                        "source": None}
    assert any(n.startswith("similarity benchmark wordsim353 unavailable (not in benchmark_dir")
               for n in ctx.coverage["notes"])
    default = prepare_context(run, tmp_path / "out2", options={"threads": 1, "min_count": 1})
    assert default.coverage["google_analogies"]["available"] is False and not default.similarity
    assert any("gensim test data not found" in n for n in default.coverage["notes"])
