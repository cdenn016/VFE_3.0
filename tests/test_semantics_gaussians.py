r"""Tests for :mod:`vfe3.semantics.gaussians` (tiny synthetic arrays only, CPU)."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from scipy import integrate, linalg, stats as sps

from vfe3.semantics import common, gaussians as G


def _random_pair(rng: np.random.Generator, shape=(7, 5), scale: float = 1.0):
    mu_a = rng.normal(0.0, scale, shape)
    mu_b = rng.normal(0.0, scale, shape)
    lv_a = rng.normal(0.0, 1.0, shape)
    lv_b = rng.normal(0.0, 1.0, shape)
    return mu_a, lv_a, mu_b, lv_b


def _torch_normal(mu: np.ndarray, lv: np.ndarray) -> torch.distributions.Distribution:
    return torch.distributions.Independent(
        torch.distributions.Normal(torch.tensor(mu), torch.tensor(np.exp(0.5 * lv))), 1)


# ----------------------------------------------------------------------------------------------
# Divergences against independent references
# ----------------------------------------------------------------------------------------------

def test_kl_matches_torch_distributions() -> None:
    rng = np.random.default_rng(0)
    mu_a, lv_a, mu_b, lv_b = _random_pair(rng)
    ref = torch.distributions.kl_divergence(_torch_normal(mu_a, lv_a), _torch_normal(mu_b, lv_b))
    np.testing.assert_allclose(G.kl_diag(mu_a, lv_a, mu_b, lv_b), ref.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(G.kl_diag(mu_a, lv_a, mu_a, lv_a), 0.0)


def test_kl_pairwise_broadcast_matches_torch() -> None:
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(4, 3))
    lv = rng.normal(size=(4, 3))
    M = G.kl_diag(mu[:, None], lv[:, None], mu[None], lv[None])          # (4, 4)
    for i in range(4):
        for j in range(4):
            ref = torch.distributions.kl_divergence(_torch_normal(mu[i], lv[i]), _torch_normal(mu[j], lv[j]))
            assert M[i, j] == pytest.approx(float(ref), rel=1e-10, abs=1e-12)


def test_kl_one_vs_all_equals_pairwise() -> None:
    rng = np.random.default_rng(2)
    MU = rng.normal(0.0, 2.0, (30, 6))
    LV = rng.normal(0.0, 1.5, (30, 6))
    one = G.kl_one_vs_all(MU[3], LV[3], MU, LV)
    np.testing.assert_allclose(one, G.kl_diag(MU[3], LV[3], MU, LV), rtol=1e-9, atol=1e-9)
    assert one[3] == pytest.approx(0.0, abs=1e-10)
    assert one.shape == (30,)
    many = G.kl_one_vs_all(MU[:5], LV[:5], MU, LV)
    assert many.shape == (5, 30)
    np.testing.assert_allclose(many, G.kl_diag(MU[:5, None], LV[:5, None], MU[None], LV[None]),
                               rtol=1e-9, atol=1e-9)
    assert (many >= 0.0).all()


def test_jeffreys_is_symmetrized_kl() -> None:
    rng = np.random.default_rng(3)
    a = _random_pair(rng)
    J = G.jeffreys_diag(*a)
    np.testing.assert_allclose(J, G.kl_diag(*a) + G.kl_diag(a[2], a[3], a[0], a[1]), rtol=1e-10)
    np.testing.assert_allclose(J, G.jeffreys_diag(a[2], a[3], a[0], a[1]), rtol=1e-12)
    np.testing.assert_array_equal(G.jeffreys_diag(a[0], a[1], a[0], a[1]), 0.0)


def test_w2_matches_bures_formula_after_common_rotation() -> None:
    r"""Diagonal closed form == general Bures W2 for (R S_a R^T, R S_b R^T) (common orthogonal R)."""
    rng = np.random.default_rng(4)
    K = 4
    R, _ = np.linalg.qr(rng.normal(size=(K, K)))
    for _ in range(5):
        mu_a, lv_a, mu_b, lv_b = (rng.normal(size=K) for _ in range(4))
        Sa = R @ np.diag(np.exp(lv_a)) @ R.T
        Sb = R @ np.diag(np.exp(lv_b)) @ R.T
        root_a = linalg.sqrtm(Sa).real
        cross = linalg.sqrtm(root_a @ Sb @ root_a).real
        bures = np.trace(Sa + Sb - 2.0 * cross)
        ref = float(np.sum((R @ (mu_a - mu_b)) ** 2) + bures)
        assert G.w2_sq_diag(mu_a, lv_a, mu_b, lv_b) == pytest.approx(ref, rel=1e-8)


def test_bhattacharyya_and_hellinger_references() -> None:
    rng = np.random.default_rng(5)
    K = 3
    mu_a, lv_a, mu_b, lv_b = (rng.normal(size=K) for _ in range(4))
    Sa, Sb = np.diag(np.exp(lv_a)), np.diag(np.exp(lv_b))
    S = 0.5 * (Sa + Sb)
    d = mu_a - mu_b
    ref = d @ np.linalg.solve(S, d) / 8.0 + 0.5 * math.log(
        np.linalg.det(S) / math.sqrt(np.linalg.det(Sa) * np.linalg.det(Sb)))
    assert G.bhattacharyya_diag(mu_a, lv_a, mu_b, lv_b) == pytest.approx(ref, rel=1e-10)
    # 1-D Hellinger by quadrature: H^2 = 1 - int sqrt(p_a p_b)
    m1, v1, m2, v2 = 0.3, 0.5, -0.4, 2.0
    bc, _ = integrate.quad(lambda x: math.sqrt(sps.norm.pdf(x, m1, math.sqrt(v1))
                                               * sps.norm.pdf(x, m2, math.sqrt(v2))), -40, 40)
    h2 = G.hellinger_sq_diag(np.array([m1]), np.log([v1]), np.array([m2]), np.log([v2]))
    assert float(h2) == pytest.approx(1.0 - bc, abs=1e-9)
    assert float(G.hellinger_sq_diag(mu_a, lv_a, mu_a, lv_a)) == 0.0


def test_expected_likelihood_and_distribution_cosine() -> None:
    rng = np.random.default_rng(6)
    K = 3
    mu_a, lv_a, mu_b, lv_b = (rng.normal(size=K) for _ in range(4))
    ref = sps.multivariate_normal(mean=mu_b, cov=np.diag(np.exp(lv_a) + np.exp(lv_b))).logpdf(mu_a)
    assert G.log_expected_likelihood(mu_a, lv_a, mu_b, lv_b) == pytest.approx(ref, rel=1e-10)
    # 1-D quadrature of the probability-product kernel
    m1, v1, m2, v2 = 0.2, 0.7, 1.1, 0.3
    el, _ = integrate.quad(lambda x: sps.norm.pdf(x, m1, math.sqrt(v1)) * sps.norm.pdf(x, m2, math.sqrt(v2)),
                           -30, 30)
    got = G.log_expected_likelihood(np.array([m1]), np.log([v1]), np.array([m2]), np.log([v2]))
    assert float(got) == pytest.approx(math.log(el), rel=1e-8)
    # normalized kernel
    lab = G.log_expected_likelihood(mu_a, lv_a, mu_b, lv_b)
    laa = G.log_expected_likelihood(mu_a, lv_a, mu_a, lv_a)
    lbb = G.log_expected_likelihood(mu_b, lv_b, mu_b, lv_b)
    log_cos = G.distribution_cosine(mu_a, lv_a, mu_b, lv_b, log=True)
    assert log_cos == pytest.approx(lab - 0.5 * (laa + lbb), rel=1e-10)
    assert G.distribution_cosine(mu_a, lv_a, mu_b, lv_b) == pytest.approx(math.exp(log_cos), rel=1e-12)
    assert G.distribution_cosine(mu_a, lv_a, mu_a, lv_a) == 1.0
    assert 0.0 < G.distribution_cosine(mu_a, lv_a, mu_b, lv_b) < 1.0


# ----------------------------------------------------------------------------------------------
# Fisher-Rao
# ----------------------------------------------------------------------------------------------

def _fr_1d(m1: float, s1: float, m2: float, s2: float) -> float:
    return float(G.fisher_rao_diag(np.array([m1]), np.array([2 * math.log(s1)]),
                                   np.array([m2]), np.array([2 * math.log(s2)])))


def test_fisher_rao_special_cases() -> None:
    for s_a, s_b in [(0.3, 2.0), (1.0, 1.0 + 1e-7), (5.0, 0.01)]:
        assert _fr_1d(0.7, s_a, 0.7, s_b) == pytest.approx(math.sqrt(2) * abs(math.log(s_b / s_a)),
                                                           rel=1e-9, abs=1e-15)
    for dmu, s in [(0.5, 0.3), (1e-6, 2.0), (40.0, 0.7)]:
        ref = 2.0 * math.sqrt(2.0) * math.asinh(abs(dmu) / (2.0 * math.sqrt(2.0) * s))
        assert _fr_1d(1.0, s, 1.0 + dmu, s) == pytest.approx(ref, rel=1e-9)


def _fisher_length(mu: np.ndarray, sigma: np.ndarray) -> float:
    r"""Length of a polyline in the metric ds^2 = (dmu^2 + 2 dsigma^2)/sigma^2 (midpoint rule)."""
    dmu, ds = np.diff(mu), np.diff(sigma)
    s_mid = 0.5 * (sigma[1:] + sigma[:-1])
    return float(np.sum(np.sqrt(dmu ** 2 + 2.0 * ds ** 2) / s_mid))


def test_fisher_rao_equals_numerically_integrated_geodesic_length() -> None:
    r"""The half-plane geodesic (a semicircle in (mu/sqrt 2, sigma)) has the closed-form length,
    and a perturbed path with the same endpoints is longer."""
    m1, s1, m2, s2 = -0.8, 0.4, 1.3, 1.1
    x1, x2 = m1 / math.sqrt(2), m2 / math.sqrt(2)
    c = ((x2 ** 2 + s2 ** 2) - (x1 ** 2 + s1 ** 2)) / (2 * (x2 - x1))
    R = math.hypot(x1 - c, s1)
    t1, t2 = math.atan2(s1, x1 - c), math.atan2(s2, x2 - c)
    t = np.linspace(t1, t2, 200001)
    mu_path = math.sqrt(2) * (c + R * np.cos(t))
    sigma_path = R * np.sin(t)
    closed = _fr_1d(m1, s1, m2, s2)
    assert _fisher_length(mu_path, sigma_path) == pytest.approx(closed, rel=1e-6)
    bump = 0.05 * np.sin(np.linspace(0.0, math.pi, t.size))
    assert _fisher_length(mu_path, sigma_path + bump) > closed
    straight = np.linspace(0.0, 1.0, 200001)
    assert _fisher_length(m1 + (m2 - m1) * straight, s1 + (s2 - s1) * straight) > closed


def test_fisher_rao_metric_axioms_and_product_structure() -> None:
    rng = np.random.default_rng(7)
    a, b, c = (rng.normal(size=(200, 4)) for _ in range(3))
    la, lb, lc = (rng.normal(0.0, 1.5, size=(200, 4)) for _ in range(3))
    dab = G.fisher_rao_diag(a, la, b, lb)
    np.testing.assert_allclose(dab, G.fisher_rao_diag(b, lb, a, la), rtol=1e-12)
    np.testing.assert_array_equal(G.fisher_rao_diag(a, la, a, la), 0.0)
    assert (dab > 0).all()
    dbc = G.fisher_rao_diag(b, lb, c, lc)
    dac = G.fisher_rao_diag(a, la, c, lc)
    assert (dac <= dab + dbc + 1e-12).all()
    per = np.stack([G.fisher_rao_diag(a[:, k:k + 1], la[:, k:k + 1], b[:, k:k + 1], lb[:, k:k + 1])
                    for k in range(4)], axis=1)
    np.testing.assert_allclose(dab, np.sqrt((per ** 2).sum(axis=1)), rtol=1e-12)


def test_same_mean_fisher_rao_is_log_variance_distance_over_sqrt2() -> None:
    rng = np.random.default_rng(8)
    mu = rng.normal(size=(5, 6))
    la, lb = rng.normal(size=(5, 6)), rng.normal(size=(5, 6))
    np.testing.assert_allclose(G.fisher_rao_diag(mu, la, mu, lb),
                               np.linalg.norm(la - lb, axis=1) / math.sqrt(2), rtol=1e-10)


# ----------------------------------------------------------------------------------------------
# Charts and gauge behavior
# ----------------------------------------------------------------------------------------------

def test_divergences_invariant_under_global_diagonal_rescaling() -> None:
    rng = np.random.default_rng(9)
    a = _random_pair(rng, shape=(10, 4))
    log_D = rng.normal(size=4)
    b = (a[0] * np.exp(log_D), a[1] + 2 * log_D, a[2] * np.exp(log_D), a[3] + 2 * log_D)
    for f in (G.kl_diag, G.jeffreys_diag, G.bhattacharyya_diag, G.fisher_rao_diag):
        np.testing.assert_allclose(f(*b), f(*a), rtol=1e-9)
    np.testing.assert_allclose(G.distribution_cosine(*b, log=True),
                               G.distribution_cosine(*a, log=True), rtol=1e-9)
    assert not np.allclose(G.w2_sq_diag(*b), G.w2_sq_diag(*a))


def test_w2_chart_and_sigma_units() -> None:
    rng = np.random.default_rng(10)
    mu, lv = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    Z = G.w2_chart(mu, lv)
    D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
    np.testing.assert_allclose(D ** 2, G.w2_sq_diag(mu[:, None], lv[:, None], mu[None], lv[None]),
                               rtol=1e-10, atol=1e-12)
    mu_d0, log_d0 = G.sigma_units(mu, lv)
    np.testing.assert_allclose(log_d0, -0.5 * lv.mean(axis=0))
    np.testing.assert_allclose(G.w2_chart(mu, lv, log_d0),
                               np.concatenate([mu_d0, np.exp(0.5 * lv + log_d0)], axis=1))
    # D0 units remove a global positive-diagonal rescaling
    log_D = rng.normal(size=3)
    mu2, lv2 = mu * np.exp(log_D), lv + 2 * log_D
    mu2_d0, log2_d0 = G.sigma_units(mu2, lv2)
    np.testing.assert_allclose(mu2_d0, mu_d0, rtol=1e-12)
    np.testing.assert_allclose(G.w2_chart(mu2, lv2, log2_d0), G.w2_chart(mu, lv, log_d0), rtol=1e-12)


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------

def test_paired_spearman_delta_detects_better_scorer() -> None:
    rng = np.random.default_rng(11)
    human = rng.normal(size=200)
    good = human + 0.1 * rng.normal(size=200)
    bad = human + 3.0 * rng.normal(size=200)
    out = G._paired_spearman_delta(good, bad, human, rng=rng, n_boot=300)
    assert out["delta"] > 0 and out["ci_lo"] > 0 and out["p"] < 0.05
    same = G._paired_spearman_delta(good, good, human, rng=rng, n_boot=50)
    assert same["delta"] == 0.0 and same["ci_lo"] == 0.0 and same["ci_hi"] == 0.0


def test_init_log_var_is_family_aware() -> None:
    assert G._init_log_var(SimpleNamespace(family="gaussian_diagonal", sigma_init=15.0)) == pytest.approx(math.log(15.0))
    assert G._init_log_var(SimpleNamespace(family="gaussian_full", sigma_init=3.0)) == pytest.approx(math.log(3.0))
    assert G._init_log_var(SimpleNamespace(family="laplace_diagonal", sigma_init=2.0)) == pytest.approx(math.log(8.0))


# ----------------------------------------------------------------------------------------------
# run_sigma_section on a fake run exposing exactly the spec accessors
# ----------------------------------------------------------------------------------------------

_CLASS_NAMES = ["punctuation", "number", "function word", "content word", "capitalized word",
                "subword", "other"]


class _FakeVocab:
    def __init__(self, V: int) -> None:
        self.kind = "custom"
        self.size = V
        self.strings = [f" w{i}" if i % 5 else f"s{i}" for i in range(V)]
        self.valid_text = np.ones(V, dtype=bool)
        self.valid_text[-1] = False

    def token_classes(self):
        codes = np.where(np.arange(self.size) % 5 == 0, 5, 3)
        codes[:3] = 2
        return codes.astype(np.int64), list(_CLASS_NAMES)

    def display(self, token_id: int) -> str:
        return self.strings[int(token_id)].strip()


class _FakeRun:
    r"""Minimal stand-in for ``SemanticRun`` (only the accessors the Sigma section reads)."""

    def __init__(self, *, V: int = 90, K: int = 4, seed: int = 0, decode_kind: str = "linear",
                 frames_active: bool = True, sigma_init: float = 1.5) -> None:
        rng = np.random.default_rng(seed)
        self.cfg = SimpleNamespace(family="gaussian_diagonal", sigma_init=sigma_init, eps=1e-6)
        self.vocab = _FakeVocab(V)
        self.K, self.n_layers, self.n_heads = K, 2, 2
        self.block_slices = [slice(0, K // 2), slice(K // 2, K)]
        self.frames_active = frames_active
        self.decode_kind = decode_kind
        self.sigma_frozen = True
        self.phi_frozen = True
        self._mu = rng.normal(size=(V, K))
        self._lv = math.log(sigma_init) + rng.normal(0.0, 0.3, size=(V, K))
        self._lv[[0, V - 2]] = math.log(sigma_init)              # rows still at init
        self._phi = rng.normal(0.0, 0.3, size=(V, 2, K // 2, K // 2))
        self._dmu = rng.normal(size=(V, K))
        self._dlv = rng.normal(0.0, 0.5, size=(V, K))
        counts = rng.integers(5, 5000, size=V).astype(np.int64)
        counts[[V - 2, V - 3]] = 0
        self.counts = counts
        self.counts_source = "unigram_log_prior"

    def broaden(self, rows, amount: float) -> None:
        self._lv[rows] += amount

    def mu(self, ids): return self._mu[np.asarray(ids)].copy()
    def log_var(self, ids): return self._lv[np.asarray(ids)].copy()
    def phi_blocks(self, ids): return [self._phi[np.asarray(ids), b] for b in range(2)]
    def decode_mu(self, ids): return self._dmu[np.asarray(ids)].copy()
    def decode_log_var(self, ids): return self._dlv[np.asarray(ids)].copy()
    def tau_eff(self) -> float: return 0.5


def _context(run: _FakeRun, tmp_path: Path, *, with_counts: bool = True) -> common.AnalysisContext:
    counts = run.counts if with_counts else None
    word = np.array([i % 5 != 0 and run.vocab.valid_text[i] for i in range(run.vocab.size)])
    ids = np.flatnonzero(word & (run.counts >= 10))
    ids = ids[np.argsort(-run.counts[ids], kind="stable")]
    rng = np.random.default_rng(3)
    hypo = [int(t) for t in ids[:8]]
    hyper = [int(t) for t in ids[8:10]]
    run.broaden(hyper, 1.0)                                      # hypernyms are broad
    pairs = rng.choice(ids, size=(24, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    mu, lv = run.mu(pairs.ravel()).reshape(-1, 2, run.K), run.log_var(pairs.ravel()).reshape(-1, 2, run.K)
    human = -G.jeffreys_diag(mu[:, 0], lv[:, 0], mu[:, 1], lv[:, 1]) + 0.1 * rng.normal(size=len(pairs))
    out = tmp_path / "semantics"
    return common.AnalysisContext(
        out_dir=out, figures_dir=out / "figures", tables_dir=out / "tables", vocab=run.vocab,
        candidate_ids=ids, counts=counts,
        lexicon={"alpha": [int(t) for t in ids[10:16]], "beta": [int(t) for t in ids[16:22]],
                 "empty": [int(run.vocab.size - 1)]},
        relations={}, analogies=[], curated_analogies=[],
        similarity={"simlex999": [(int(a), int(b), float(h)) for (a, b), h in zip(pairs, human)],
                    "wordsim353": [(int(ids[0]), int(ids[1]), 5.0)]},
        hypernyms=[(h, hyper[i % 2]) for i, h in enumerate(hypo)],
        axes={}, probe_ids=[int(ids[0]), int(ids[5]), int(run.vocab.size - 1)], ambiguous_ids=[],
        coverage={}, seed=0, options={"min_count": 10})


def _assert_jsonable(result: common.SectionResult) -> None:
    json.dumps(common.to_jsonable(result.metrics), allow_nan=False)
    json.dumps(common.to_jsonable(result.headline), allow_nan=False)


def _written(result: common.SectionResult) -> set:
    return {p.name for p in result.figures if p.exists()}


@pytest.fixture(autouse=True)
def _fast_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Keep the section's style but save at 40 dpi with one small font (the multilingual font
    stack's glyph measurement and PDF subsetting dominate the runtime of these tests)."""
    original = G.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"],
                             "font.sans-serif": ["DejaVu Sans"]})

    monkeypatch.setattr(G, "apply_style", low_dpi_style)


def _fake_frames(monkeypatch: pytest.MonkeyPatch, summary_fn) -> None:
    module = types.ModuleType("vfe3.semantics.frames")
    module.routing_covariance_summary = summary_fn
    import vfe3.semantics as package
    monkeypatch.setitem(sys.modules, "vfe3.semantics.frames", module)
    monkeypatch.setattr(package, "frames", module, raising=False)


def _schema_summary(run: _FakeRun, ctx: common.AnalysisContext, ids) -> Dict[str, Any]:
    r"""Stand-in with the documented return schema of frames.routing_covariance_summary."""
    ids = np.asarray(ids)
    lv = run.log_var(ids)
    blocks = run.phi_blocks(ids)
    sig_b = np.stack([lv[:, sl].sum(1) for sl in run.block_slices], axis=1)
    frm_b = np.stack([-2.0 * np.trace(X, axis1=1, axis2=2) for X in blocks], axis=1)
    rng = np.random.default_rng(0)
    lc = ctx.log_count(ids)
    sp = {k: {"rho": float(sps.spearmanr(v, lc).statistic), "p": 0.5, "n": ids.size}
          for k, v in (("log_det_C", sig_b.sum(1) + frm_b.sum(1)), ("sigma_part", sig_b.sum(1)),
                       ("frame_part", frm_b.sum(1)))}
    return {"ids": ids, "log_det_C": sig_b.sum(1) + frm_b.sum(1), "sigma_part": sig_b.sum(1),
            "frame_part": frm_b.sum(1), "sigma_part_block": sig_b, "frame_part_block": frm_b,
            "log_cond_block": rng.uniform(0, 2, size=sig_b.shape), "rel_gap": rng.uniform(0, 0.2, ids.size),
            "rel_gap_block": rng.uniform(0, 0.2, size=sig_b.shape), "identity_max_abs_error": 0.0,
            "variance_shares": {"sigma": 0.5, "frame": 0.4, "cross": 0.1}, "spearman_log_count": sp,
            "summary": {"rel_gap": {"median": 0.1}}, "note": "schema stand-in"}


def test_sigma_section_prior_bank_frames_inactive(tmp_path: Path) -> None:
    run = _FakeRun(frames_active=False)
    ctx = _context(run, tmp_path)
    result = G.run_sigma_section(run, ctx)
    assert result.status == "ok"
    names = _written(result)
    for stem in ("overview", "frequency_and_classes", "fields", "entailment", "gaussian_similarity"):
        assert {f"{stem}.png", f"{stem}.pdf"} <= names
    assert not any(n.startswith("decode_") or n.startswith("routing") for n in names)
    for table in ("variance_sorted_neighbors", "entailment", "gaussian_similarity"):
        assert (ctx.tables_dir / "sigma" / f"{table}.md").exists() and table in result.tables
    _assert_jsonable(result)
    m = result.metrics["banks"]["prior"]
    # dispersion and init bookkeeping
    lv = run.log_var(ctx.candidate_ids)
    tr = np.exp(lv).sum(axis=1)
    assert m["dispersion"]["cv_trace"] == pytest.approx(tr.std() / tr.mean())
    assert m["init"]["n_count0"] == 2 and m["init"]["frac_at_init_count0"] == 0.5
    assert m["dispersion"]["cv_trace_d0"] > 0
    # entailment: hypernyms were broadened, so KL asymmetry and the log-det baseline are perfect;
    # hypernyms are the rarer words here, so the frequency baseline is always wrong
    ent = m["entailment"]
    assert ent["n"] == 8 and ent["kl_asymmetry"]["acc"] == 1.0 and ent["logdet_baseline"]["acc"] == 1.0
    assert ent["n_clusters"] == 2 and ent["kl_asymmetry"]["n_clusters"] == 2
    assert ent["kl_asymmetry"]["ci_method"].startswith("cluster bootstrap")
    assert ent["kl_asymmetry"]["ci_lo"] == ent["kl_asymmetry"]["ci_hi"] == 1.0
    assert ent["frequency_baseline"]["acc"] == 0.0
    assert ent["kl_minus_frequency"]["delta"] == 1.0 and ent["kl_minus_frequency"]["ci_lo"] == 1.0
    assert "p" not in ent["kl_vs_frequency_discordant"]
    # similarity: the human score is -Jeffreys + small noise; tiny benchmark is skipped with a reason
    sim = m["similarity"]
    scorers = sim["simlex999"]["scorers"]
    assert scorers["neg_jeffreys"]["rho"] > 0.9
    assert "delta_vs_euclid_mu" in scorers["neg_jeffreys"] and "delta_vs_euclid_mu" not in scorers["cos_mu_raw"]
    assert sim["simlex999"]["headline_scorer"] == "neg_jeffreys"
    assert sim["simlex999"]["best_gaussian"] not in G._REFERENCE_SCORERS
    assert sim["simlex999"]["best_gaussian_note"] == "maximum over 6 scorers, not adjusted for selection"
    assert sim["simlex999"]["n_identical_dropped"] == 0
    assert sim["wordsim353"]["status"] == "skipped"
    assert "headline (pre-registered)" in result.tables["gaussian_similarity"]
    assert "cluster bootstrap over the 2 hypernym groups" in result.tables["entailment"]
    assert result.metrics["failed_steps"] == []
    # a SimpleNamespace cfg is outside the probed class: the invariance note says "assumed"
    assert result.metrics["diagonal_redundancy"]["status"] == "assumed"
    assert any("assumed, not verified" in n for n in result.notes)
    assert any("trained coordinates" in n and "cv_trace_d0" in n for n in result.notes)
    # neighbors: only probes among the candidates, 20 neighbors capped by the candidate count
    assert [r["probe_id"] for r in m["neighbors"]] == [int(ctx.candidate_ids[0]), int(ctx.candidate_ids[5])]
    logdets = [n["logdet_z"] for n in m["neighbors"][0]["neighbors"]]
    assert logdets == sorted(logdets)
    assert result.metrics["routing_covariance"]["status"] == "skipped"
    assert any("frames inactive" in n for n in result.notes)
    assert any("linear" in n for n in result.notes) and any("frozen" in n for n in result.notes)
    assert "CV(tr Σ): trained coordinates / D0 units" in result.headline
    assert "ρ(log det Σ, log count)" in result.headline
    jeff = scorers["neg_jeffreys"]
    assert result.headline["simlex999: −Jeffreys ρ / cos(μ) ρ / −‖Δμ‖ ρ"].startswith(f"{jeff['rho']:.3f} / ")
    assert result.headline["simlex999: Δρ(−Jeffreys − (−‖Δμ‖)), paired bootstrap"] == G._fmt_ci(
        jeff["delta_vs_euclid_mu"], "delta")
    assert any("not adjusted for selection" in k for k in result.headline)


def test_sigma_section_without_counts_and_with_gaussian_decode(tmp_path: Path) -> None:
    run = _FakeRun(decode_kind="gaussian", frames_active=False, seed=1)
    ctx = _context(run, tmp_path, with_counts=False)
    result = G.run_sigma_section(run, ctx)
    assert result.status == "ok"
    names = _written(result)
    assert {"decode_overview.png", "decode_entailment.pdf", "decode_gaussian_similarity.png"} <= names
    prior, decode = result.metrics["banks"]["prior"], result.metrics["banks"]["decode"]
    assert prior["frequency"]["status"] == "skipped"
    assert prior["entailment"]["frequency_baseline"]["status"] == "skipped"
    assert prior["classes"]["population"].startswith("candidate words")
    assert decode["tau_eff"] == 0.5
    assert any(k.startswith("decode ") for k in result.headline)
    _assert_jsonable(result)


def test_sigma_section_tied_gaussian_decode_is_not_repeated(tmp_path: Path) -> None:
    run = _FakeRun(decode_kind="gaussian", frames_active=False, seed=2)
    run.decode_mu, run.decode_log_var = run.mu, run.log_var
    result = G.run_sigma_section(run, _context(run, tmp_path))
    assert result.metrics["banks"]["decode"]["status"] == "skipped"
    assert not any(p.name.startswith("decode_") for p in result.figures)


def test_sigma_section_routing_covariance_with_patched_frames(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Any] = []

    def summary(run, ctx, ids):
        calls.append(np.asarray(ids))
        return _schema_summary(run, ctx, ids)

    _fake_frames(monkeypatch, summary)
    run = _FakeRun(frames_active=True, seed=3)
    ctx = _context(run, tmp_path)
    result = G.run_sigma_section(run, ctx)
    assert len(calls) == 1 and np.array_equal(calls[0], ctx.candidate_ids)
    assert {"routing_covariance.png", "routing_covariance.pdf"} <= _written(result)
    rc = result.metrics["routing_covariance"]
    assert rc["status"] == "ok" and rc["n"] == ctx.candidate_ids.size
    assert "rel_gap" not in rc and "sigma_part_block" not in rc      # per-token arrays stay out of metrics
    assert ctx.cache["routing_covariance"]["note"] == "schema stand-in"
    _assert_jsonable(result)


def test_sigma_section_records_a_failing_frames_call(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(run, ctx, ids):
        raise RuntimeError("synthetic frames failure")

    _fake_frames(monkeypatch, broken)
    run = _FakeRun(frames_active=True, seed=4)
    result = G.run_sigma_section(run, _context(run, tmp_path))
    assert result.status == "ok"
    rc = result.metrics["routing_covariance"]
    assert rc["status"] == "failed" and rc["error"] == "RuntimeError: synthetic frames failure"
    assert result.metrics["failed_steps"] == [{"step": "routing covariance",
                                               "error": "RuntimeError: synthetic frames failure"}]
    assert any("routing covariance failed" in n for n in result.notes)
    assert "overview.png" in _written(result)
    _assert_jsonable(result)


def test_sigma_section_records_failed_sub_steps_and_continues(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):
        raise ValueError("synthetic step failure")

    monkeypatch.setattr(G, "_entailment", broken)
    monkeypatch.setattr(G, "_similarity", broken)
    monkeypatch.setattr(G, "_table_geometry", broken)
    run = _FakeRun(frames_active=False, seed=6)
    result = G.run_sigma_section(run, _context(run, tmp_path))
    assert result.status == "ok"
    steps = {f["step"]: f["error"] for f in result.metrics["failed_steps"]}
    assert steps == {"Prior bank: entailment": "ValueError: synthetic step failure",
                     "Prior bank: similarity": "ValueError: synthetic step failure",
                     "Prior bank: table geometry": "ValueError: synthetic step failure"}
    m = result.metrics["banks"]["prior"]
    assert m["entailment"]["status"] == m["similarity"]["status"] == m["table_geometry"]["status"] == "failed"
    assert sum("synthetic step failure" in n for n in result.notes) == 3
    assert not any("entailment skipped" in n or "no similarity benchmark" in n for n in result.notes)
    # the other items and figures still ran (the overview figure tolerates the missing geometry)
    assert {"overview.png", "frequency_and_classes.png", "fields.png"} <= _written(result)
    assert "entailment" not in result.tables and "gaussian_similarity" not in result.tables
    assert "CV(tr Σ): trained coordinates / D0 units" in result.headline
    assert not any("Jeffreys" in k for k in result.headline)
    _assert_jsonable(result)


def test_decode_bank_failure_is_recorded(tmp_path: Path) -> None:
    run = _FakeRun(decode_kind="gaussian", frames_active=False, seed=7)

    def broken(ids):
        raise KeyError("decode table missing")

    run.decode_mu = broken
    result = G.run_sigma_section(run, _context(run, tmp_path))
    assert result.status == "ok"
    assert result.metrics["banks"]["decode"]["status"] == "failed"
    assert [f["step"] for f in result.metrics["failed_steps"]] == ["Decode bank"]
    assert result.metrics["failed_steps"][0]["error"].startswith("KeyError:")
    assert result.metrics["banks"]["prior"]["entailment"]["status"] == "ok"
    _assert_jsonable(result)


@pytest.mark.parametrize("overrides, status, phrase", [
    ({"norm_type_block": "layernorm", "layernorm_affine": False}, "broken", "not a symmetry of this run"),
    ({"gauge_group": "block_glk", "attention_mode": "projected_predictive", "norm_type_block": "none",
      "norm_type_final": "none"}, "exact", "exact redundancy of this run"),
])
def test_invariance_note_follows_diagonal_redundancy_status(tmp_path: Path, overrides: Dict[str, Any],
                                                            status: str, phrase: str) -> None:
    run = _FakeRun(frames_active=False, seed=8)
    for key, value in overrides.items():
        setattr(run.cfg, key, value)
    assert common.diagonal_redundancy_status(run.cfg)[0] == status
    result = G.run_sigma_section(run, _context(run, tmp_path))
    assert result.metrics["diagonal_redundancy"]["status"] == status
    notes = [n for n in result.notes if n.startswith("KL, Jeffreys")]
    assert len(notes) == 1 and phrase in notes[0]


def test_identical_word_pairs_are_dropped_like_the_vector_battery(tmp_path: Path) -> None:
    from vfe3.semantics import vectors
    run = _FakeRun(frames_active=False, seed=9)
    ctx = _context(run, tmp_path)
    t = int(ctx.candidate_ids[3])
    ctx.similarity["simlex999"] = list(ctx.similarity["simlex999"]) + [(t, t, 10.0)]
    result = G.run_sigma_section(run, ctx)
    sim = result.metrics["banks"]["prior"]["similarity"]["simlex999"]
    assert sim["n_identical_dropped"] == 1 and sim["n"] == sim["n_listed"] - 1
    view = common.VectorView(name="mu", label="μ", ids=ctx.candidate_ids, X=run.mu(ctx.candidate_ids))
    scores, _ = vectors.view_similarity_scores(view, ctx.similarity["simlex999"])
    assert scores.size == sim["n"]


def test_headline_quotes_the_pre_registered_scorer_not_the_best() -> None:
    def entry(rho: float, label: str) -> Dict[str, Any]:
        return {"rho": rho, "label": label,
                "delta_vs_euclid_mu": {"delta": rho - 0.1, "ci_lo": rho - 0.2, "ci_hi": rho}}

    scorers = {key: entry(0.1, label) for key, label in G._SCORERS}
    scorers["neg_jeffreys"] = entry(0.30, "−Jeffreys")
    scorers["neg_kl_12"] = entry(0.35, "−KL(w1‖w2)")
    m = {"dispersion": {"cv_trace": 0.2, "cv_trace_d0": 0.1}, "frequency": {"status": "skipped"},
         "entailment": {"status": "skipped"},
         "similarity": {"simlex999": {"status": "ok", "n": 50, "scorers": scorers,
                                      "best_gaussian": "neg_kl_12",
                                      "best_gaussian_note": "maximum over 6 scorers, not adjusted for selection"}}}
    h = G._headline(m, "")
    assert h["CV(tr Σ): trained coordinates / D0 units"] == "0.200 / 0.100"
    assert h["simlex999: −Jeffreys ρ / cos(μ) ρ / −‖Δμ‖ ρ"] == "0.300 / 0.100 / 0.100 (n = 50)"
    assert h["simlex999: Δρ(−Jeffreys − (−‖Δμ‖)), paired bootstrap"] == "0.200 [0.100, 0.300]"
    assert h["simlex999: best Gaussian ρ (maximum over 6 scorers, not adjusted for selection)"] == \
        "−KL(w1‖w2) 0.350"


# ----------------------------------------------------------------------------------------------
# Cluster bootstrap (entailment)
# ----------------------------------------------------------------------------------------------

def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    cluster = np.array([7, 7, 7, 3, 3, 9])                      # sizes: 3 -> 2, 7 -> 3, 9 -> 1
    correct = np.array([[1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]])
    boot = G._cluster_bootstrap(correct, cluster, np.random.default_rng(5), n_boot=6)
    idx = np.random.default_rng(5).integers(0, 3, size=(6, 3))  # same draws; codes sorted 3, 7, 9
    members = {0: [3, 4], 1: [0, 1, 2], 2: [5]}
    for b in range(6):
        rows = [r for c in idx[b] for r in members[int(c)]]
        np.testing.assert_allclose(boot[b], correct[:, rows].mean(axis=1), rtol=1e-12)


def test_cluster_bootstrap_widens_intervals_for_correlated_pairs() -> None:
    r"""10 hypernym groups of 10 pairs, each group all right or all wrong: the effective sample size
    is 10 groups, so the interval is far wider than a Wilson interval on 100 independent pairs."""
    cluster = np.repeat(np.arange(10), 10)
    ok = np.repeat(np.arange(10) % 2 == 0, 10)
    boot = G._cluster_bootstrap(ok[None], cluster, np.random.default_rng(0))[:, 0]
    entry = G._cluster_proportion(ok, boot, 10)
    assert entry["acc"] == 0.5 and entry["n"] == 100
    wilson = sps.binomtest(50, 100).proportion_ci(method="wilson")
    assert entry["ci_hi"] - entry["ci_lo"] > 2.0 * (wilson.high - wilson.low)
    single = G._cluster_proportion(ok[:10], boot, 1)
    assert single["ci_lo"] is None and single["p_vs_chance"] is None


def test_sigma_section_with_real_frames_summary(tmp_path: Path) -> None:
    r"""Integration with the real frames.routing_covariance_summary on the tiny fake run."""
    run = _FakeRun(frames_active=True, seed=5)
    ctx = _context(run, tmp_path)
    result = G.run_sigma_section(run, ctx)
    rc = result.metrics["routing_covariance"]
    assert rc["status"] == "ok", rc.get("error")
    assert rc["identity_max_abs_error"] < 1e-8
    assert "routing_covariance.png" in _written(result)
    _assert_jsonable(result)
