r"""Report orchestration of :mod:`vfe3.semantics`: views, ``analyze_run`` end to end, failure isolation.

Every model is tiny (a vocabulary of about 90 real English words so the curated resources and the
Google/SimLex benchmarks partially resolve, K=4, two heads, two layers) and saved to a temporary run
directory. Figures are written at 40 dpi in DejaVu Sans to keep the end-to-end tests short.
"""

import importlib
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel
from vfe3.semantics import report as R
from vfe3.semantics.common import SectionResult
from vfe3.semantics.gaussians import w2_sq_diag
from vfe3.semantics.run import SemanticRun, load_semantic_run, prepare_context
from vfe3.semantics.vocab import custom_vocab


WORDS = [
    "the", "of", "and", "man", "woman", "king", "queen", "boy", "girl", "father", "mother", "brother",
    "sister", "son", "daughter", "husband", "wife", "he", "she", "his", "her", "prince", "princess",
    "France", "Paris", "Germany", "Berlin", "Italy", "Rome", "Japan", "Tokyo", "Spain", "Madrid",
    "England", "London", "Russia", "Moscow", "China", "Beijing", "Greece", "Athens", "French", "German",
    "Italian", "dog", "cat", "horse", "cow", "dogs", "cats", "red", "blue", "green", "yellow", "one", "two",
    "three", "four", "January", "February", "March", "Monday", "Tuesday", "good", "bad", "big", "small",
    "bigger", "smaller", "hot", "cold", "run", "walk", "water", "music", "war", "science", "church",
    "city", "love", "computer", "river", "bank", "light", "animal", "color", "country",
]
STRINGS = ["<unk>", ",", ".", "3", *WORDS]
V = len(STRINGS)
COUNTS = np.random.default_rng(0).integers(20, 400, size=V)
COUNTS[0] = 0                                                         # <unk>: the zero-count token
LINEAR_VIEWS = {"mu_raw", "mu_sigma_units", "w2_chart", "xi_ref", "log_var", "phi_raw", "phi_log",
                "decode_W", "decode_W_causal"}
E2E_VIEWS = ("mu_raw", "w2_chart", "xi_ref", "phi_log", "decode_W_causal")   # one per chart family
SMALL = {"min_count": 1, "threads": 1, "umap": False, "frame_ref_count": 32, "geometry_max_points": 200,
         "cluster_top_words": 60, "n_clusters": 4, "frame_null": 20, "frame_commutator_pairs": 20,
         "frame_rerank": 10, "frame_rerank_questions": 20, "contextual_tokens": 64, "occurrence_tokens": 256,
         "occurrence_per_word": 2, "context_len": 6}


@pytest.fixture(autouse=True)
def _fast_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The sections' style at 40 dpi with one font (glyph measurement and PDF subsetting of the
    multilingual font stack dominate the figure time)."""
    from vfe3.semantics import common, frames, gaussians, vectors

    original = common.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"],
                             "font.sans-serif": ["DejaVu Sans"]})

    modules = [common, frames, gaussians, vectors]
    contextual = sys.modules.get("vfe3.semantics.contextual")
    if contextual is not None:
        modules.append(contextual)
    for module in modules:
        if hasattr(module, "apply_style"):
            monkeypatch.setattr(module, "apply_style", low_dpi_style)


def _tiny_run(root: Path, name: str = "tiny", **over) -> Tuple[Path, VFEModel]:
    options = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=2, e_phi_lr=0.0,
                   attention_mode="projected_predictive")
    options.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(**options)
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    bank = model.prior_bank
    with torch.no_grad():                                              # move the tables off their init
        bank.sigma_log_embed.normal_(0.0, 0.3)
        bank.phi_embed.normal_(0.0, 0.3)
        if hasattr(bank, "decode_sigma_log_embed"):
            bank.decode_sigma_log_embed.normal_(0.0, 0.3)
            bank.decode_mu_embed.normal_()
        bank.set_unigram_log_prior(torch.from_numpy(COUNTS).float())
    run_dir = root / name
    run_dir.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "config": serialize_config(cfg),
                "selection_data_identity": None}, run_dir / "best_model.pt")
    (run_dir / "config.json").write_text(
        json.dumps({"config": serialize_config(cfg), "dataset": "wikitext-103"}), encoding="utf-8")
    return run_dir, model


def _load(run_dir: Path) -> SemanticRun:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_semantic_run(run_dir, vocab=custom_vocab(STRINGS))


def _context(run: SemanticRun, out: Path, **over):
    return prepare_context(run, out, options={**R.DEFAULT_OPTIONS, **SMALL, **over})


def _analyze(run_dir: Path, out: Path, **over) -> Tuple[Path, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = R.analyze_run(run_dir, out_dir=out, options={**SMALL, **over}, vocab=custom_vocab(STRINGS))
    return path, json.loads((out / "metrics.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------------------------

def test_build_views_linear_decode_charts(tmp_path):
    run_dir, _ = _tiny_run(tmp_path)
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out")
    views = R.build_views(run, ctx)
    assert set(views) == LINEAR_VIEWS
    ids = ctx.candidate_ids
    for name, view in views.items():
        assert view.note, name
        if name != "phi_log":
            np.testing.assert_array_equal(view.ids, ids)
    mu, lv = run.mu(ids), run.log_var(ids)
    log_d0 = -0.5 * lv.mean(axis=0)
    np.testing.assert_allclose(views["mu_sigma_units"].X, mu * np.exp(log_d0), rtol=1e-6)
    # Euclidean distance in the W2 chart is the 2-Wasserstein distance in D0 units
    X = views["w2_chart"].X.astype(np.float64)
    mu_d0, lv_d0 = mu * np.exp(log_d0), lv + 2.0 * log_d0
    for a, b in ((0, 1), (3, 17), (5, 40)):
        np.testing.assert_allclose(np.sum((X[a] - X[b]) ** 2),
                                   w2_sq_diag(mu_d0[a], lv_d0[a], mu_d0[b], lv_d0[b]), rtol=1e-5)
    np.testing.assert_allclose(views["log_var"].X, lv, rtol=1e-6)
    np.testing.assert_allclose(views["phi_raw"].X, run.phi(ids), rtol=1e-6)
    # the causal decode view is white over the candidates
    Z = views["decode_W_causal"].X.astype(np.float64)
    np.testing.assert_allclose(np.cov(Z, rowvar=False), np.eye(4), atol=1e-5)
    # phi_log: flattened canonical logs of the ok rows (reference set = frame_ref_count most frequent)
    from vfe3.semantics import frames
    ref = ids[:32]
    cf = frames.canonical_frames(run, ids, ref, run.log_var(ref))
    ids_ok, L = cf.log_chart()
    np.testing.assert_array_equal(views["phi_log"].ids, ids_ok)
    np.testing.assert_allclose(views["phi_log"].X, L, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(views["xi_ref"].X, frames.reference_content(run, ids, 4), rtol=1e-5, atol=1e-6)


def test_causal_rows_are_invariant_under_global_linear_maps():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 5)) + rng.normal(size=5)
    Z, rank = R._causal_rows(W)
    assert rank == 5
    A = rng.normal(size=(5, 5)) + 3 * np.eye(5)                        # any invertible global map
    Z2, _ = R._causal_rows(W @ A)
    np.testing.assert_allclose(Z @ Z.T, Z2 @ Z2.T, atol=1e-8)         # same causal inner products
    Zs, rank_s = R._causal_rows(np.concatenate([W, W[:, :1]], axis=1))   # singular covariance
    assert rank_s == 5
    np.testing.assert_allclose(Zs @ Zs.T, Z @ Z.T, atol=1e-8)
    with pytest.raises(ValueError):
        R._causal_rows(np.ones((10, 3)))


def test_natural_parameters_linearize_the_kl_decode(tmp_path):
    run_dir, model = _tiny_run(tmp_path, decode_mode="diagonal_chunked", untie_decode_bank=True)
    run = _load(run_dir)
    eta = R._natural_parameters(run, np.arange(V))                     # (V, 2K)
    rng = np.random.default_rng(1)
    residuals = []
    lam, logits = [], []
    for _ in range(3):
        mu, var = rng.normal(size=4), np.exp(0.3 * rng.normal(size=4))
        with torch.no_grad():
            out = model.prior_bank.decode(torch.tensor(mu, dtype=torch.float32).view(1, 1, 4),
                                          torch.tensor(var, dtype=torch.float32).view(1, 1, 4))
        lam.append(np.concatenate([mu, mu ** 2 + var]))
        logits.append(out.view(-1).double().numpy())
    for i in (1, 2):
        # logits(q) - <lambda(q), eta_v> must not depend on v beyond a q-independent c_v
        r = (logits[i] - logits[0]) - eta @ (lam[i] - lam[0])          # (V,)
        residuals.append(np.ptp(r) / np.ptp(logits[i]))
    assert max(residuals) < 1e-4
    ctx = _context(run, tmp_path / "out")
    views = R.build_views(run, ctx)
    assert {"decode_mu", "decode_natural_causal"} <= set(views)
    assert not {"decode_W", "decode_W_causal"} & set(views)


def test_head_evidence_mixer_skips_the_natural_parameter_view(tmp_path):
    run_dir, _ = _tiny_run(tmp_path, decode_mode="diagonal_chunked", untie_decode_bank=True,
                           use_priorbank_head_evidence_mixer=True)
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out", views=("decode_mu", "decode_natural_causal"))
    views = R.build_views(run, ctx)
    assert set(views) == {"decode_mu"}
    assert any("decode_natural_causal skipped" in n and "head-evidence" in n for n in ctx.cache["view_notes"])
    assert not ctx.cache["view_errors"]


def test_empty_candidates_build_no_view(tmp_path):
    run_dir, _ = _tiny_run(tmp_path)
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out")
    ctx.candidate_ids = np.zeros(0, dtype=np.int64)
    assert R.build_views(run, ctx) == {}
    assert any("candidate set is empty" in n for n in ctx.cache["view_notes"])
    res = R._embeddings_section(run, ctx)
    assert res.status == "skipped"
    assert R._phi_section(run, ctx).status == "skipped"


def test_view_notes_follow_the_diagonal_status_and_barycenter(tmp_path):
    from vfe3.semantics import frames

    run_dir, _ = _tiny_run(tmp_path, norm_type_block="layernorm")
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out", views=("mu_raw", "w2_chart", "phi_log", "decode_W"))
    ids = ctx.candidate_ids
    cf = frames.canonical_frames(run, ids, ids[:32], run.log_var(ids[:32]))
    cf.diagnostics["barycenter_converged_all"] = False                  # as if max_iter was exhausted
    cf.diagnostics["blocks"][0]["barycenter"]["residual"] = 3.0e-4
    cf.diagnostics["barycenter_residual_max"] = 3.0e-4
    ctx.cache["canonical_frames"] = cf
    views = R.build_views(run, ctx)
    for name in ("mu_raw", "w2_chart", "decode_W"):
        assert "not an exact symmetry of this run because a LayerNorm" in views[name].note, name
    assert "up to O(r), r = 3.0e-04" in views["phi_log"].note
    assert ctx.cache["phi_log_chart"] == {"barycenter_converged_all": False, "barycenter_residual_max": 3.0e-4}


def _fake_run(**cfg) -> SimpleNamespace:
    base = dict(norm_type_block="none", norm_type_final="none", gauge_group="block_glk",
                attention_mode="projected_predictive", use_block_mlp=True, block_mlp_mode="coordinate",
                transport_mode="regular_representation")
    return SimpleNamespace(cfg=SimpleNamespace(**{**base, **cfg}), sigma_frozen=False, phi_frozen=False,
                           frames_active=True)


def test_gauge_paragraph_states_measured_only_for_the_probed_class():
    exact = R._gauge_paragraph(_fake_run())
    assert "were measured to leave the logits unchanged to fp32 round-off for the configuration class of this" in exact
    assert "assumed" not in exact and "If (iii)" not in exact
    broken = R._gauge_paragraph(_fake_run(norm_type_block="layernorm"))
    assert "only for projected-predictive block-GL runs" in broken
    assert "(i) and (ii) are assumed, not verified" in broken
    assert "(iii) is not an exact symmetry here because a LayerNorm" in broken
    group = R._gauge_paragraph(_fake_run(gauge_group="so_n"))
    assert "not an exact symmetry here because D e^X leaves the gauge group" in group
    assumed = R._gauge_paragraph(_fake_run(use_head_mixer=True))
    assert "(iii) is assumed, not verified (not probed for use_head_mixer=True)" in assumed
    assert "for the configuration class of this run" not in assumed
    chart = {"barycenter_converged_all": False, "blocks": [{"barycenter": {"residual": 2.5e-3}}]}
    assert "only up to O(r), r = 2.5e-03" in R._gauge_paragraph(_fake_run(), chart)
    assert "up to O(r)" not in R._gauge_paragraph(_fake_run(), {"barycenter_converged_all": True, "blocks": []})


def _battery(acc: float, only_b: float, purity: float = 0.2, rho: float = 0.1, n_covered: int = 10162) -> dict:
    ci = lambda m: {"mean": m, "ci_lo": m - 0.01, "ci_hi": m + 0.01}           # noqa: E731
    group = {"methods": {"3cosadd": {"acc": ci(acc)}, "only_b": {"acc": ci(only_b)}}}
    return {"analogy_benchmark": {"n_covered": n_covered, "n_total": n_covered, "overall": group,
                                  "semantic": group, "syntactic": group},
            "fields": {"purity": {"purity": purity, "null_mean": 0.03, "z": 12.0, "k": 10, "n": 700}},
            "similarity": {"simlex999": {"rho": rho, "ci_lo": rho - 0.06, "ci_hi": rho + 0.06, "n": 900}}}


def test_analogy_headline_selects_by_margin_over_only_b():
    batteries = {"phi_raw": _battery(0.0453, 0.0449, purity=0.84, rho=0.39),
                 "decode_W": _battery(0.0451, 0.0119, purity=0.79, rho=0.44),
                 "mu_raw": _battery(0.001, 0.007, purity=0.10, rho=0.05)}
    meta = {"phi_raw": {"label": "φ raw", "group": "phi"}, "decode_W": {"label": "decode W", "group": "decode"},
            "mu_raw": {"label": "μ raw", "group": "mu"}}
    res = SimpleNamespace(metrics={"batteries": batteries, "views": meta})
    ctx = SimpleNamespace(coverage={"google_analogies": {"n_total": 19544, "n_resolved": 10162}})
    text = R._embeddings_summary(res, ctx)
    assert "10162/19544 (52.0%)" in text
    assert "decode W view, 3CosAdd" in text and "+0.033 over ONLY-B is the largest" in text
    assert "The highest 3CosAdd accuracy, 0.045" in text and "φ raw view" in text
    assert "does not exceed its ONLY-B baseline for the μ raw (0.0010 vs 0.0070) view" in text
    assert "maximum over 3 views, not adjusted for selection" in text
    views = {n: SimpleNamespace(label=m["label"]) for n, m in meta.items()}
    head = R._embeddings_headline(batteries, views)
    assert head["best Google 3CosAdd − ONLY-B margin (view)"].startswith("0.033 (decode W")
    assert head["best Google 3CosAdd accuracy (view)"].startswith("0.045 (φ raw")
    glance = R._at_a_glance({"embeddings": SectionResult(name="embeddings", metrics=res.metrics)})
    assert "μ views (μ raw): " in glance and "3CosAdd does not exceed ONLY-B in any of these views" in glance
    assert "decode views (decode W): " in glance and "exceeds ONLY-B by up to 0.0332 (0.0451 vs 0.0119" in glance
    assert "Σ views: none analyzed." in glance and "SimLex-999 ρ up to 0.050 [-0.010, 0.110]; its 95% CI includes 0" in glance


def test_at_a_glance_contextual_sentence_uses_the_final_layer():
    layer = lambda l, pur, a, ob: {"layer": l, "fields": {"purity": pur, "null_mean": 0.05, "z": 4.0},   # noqa: E731
                                   "analogies": {"n_covered": 50, "overall": {"3cosadd": {"mean": a},
                                                                              "only_b": {"mean": ob}}}}
    sem = {"n_words": 680, "layers": [layer(0, 0.3, 0.4, 0.1), layer(1, 0.2, 0.2, 0.25)],
           "static_mu": {"fields": {"purity": 0.31}, "analogies": {"overall": {"3cosadd": {"mean": 0.41},
                                                                             "only_b": {"mean": 0.1}}}}}
    text = R._at_a_glance({"contextual": SectionResult(name="contextual", metrics={"layerwise_semantics": sem})})
    assert "final layer 1" in text and "680 words, retrieval among those words only" in text
    assert "purity 0.200" in text and "3CosAdd 0.200 against ONLY-B 0.250 on 50 questions, so 3CosAdd does not" in text
    assert "static prior μ of the same words gives purity 0.310 and 3CosAdd 0.410 against ONLY-B 0.100" in text
    skipped = R._at_a_glance({"contextual": SectionResult(name="contextual", metrics={
        "layerwise_semantics": {"status": "skipped", "reason": "only 2 words have occurrences"}})})
    assert "layerwise_semantics skipped: only 2 words have occurrences" in skipped


def test_failures_aggregate_failed_steps_and_failed_substeps():
    results = {
        "contextual": SectionResult(name="contextual", metrics={
            "occurrences": {"status": "failed", "error": "ValueError: not symmetric", "traceback": "..."},
            "sigma": {"status": "skipped", "reason": "Σ does not move"}}),
        "phi": SectionResult(name="phi", metrics={"failed_steps": [{"step": "positional", "error": "KeyError: 'x'"}]}),
        "embeddings": SectionResult(name="embeddings", metrics={
            "batteries": {"mu_raw": {"geometry": {"status": "failed", "error": "Traceback\nRuntimeError: boom"},
                                     "failed_steps": [{"step": "fields/purity", "error": "ValueError: y"}],
                                     "output_errors": {"map": "Traceback\nOSError: disk"}}},
            "failed_views": {}}),
    }
    text = "\n".join(R._failure_lines(results))
    assert "section **contextual**, step `occurrences` failed: `ValueError: not symmetric`" in text
    assert "section contextual, step sigma skipped: Σ does not move" in text
    assert "section **phi**, step `positional` failed: `KeyError: 'x'`" in text
    assert "view **mu_raw**, step `geometry` failed: `RuntimeError: boom`" in text
    assert "view **mu_raw**, step `fields/purity` failed" in text and "`output_errors/map` failed: `OSError: disk`" in text


def test_phi_summary_quotes_permutation_nulls_and_the_paired_rerank_subset():
    overall = {"chart": {"acc": 0.05}, "rerank": {"acc": 0.06}, "only_b": {"acc": 0.01}, "chart_reranked": {"acc": 0.04}}
    res = SectionResult(name="phi", metrics={
        "chart": {"n": 10, "n_ref": 5, "ok_fraction": 1.0, "barycenter_converged_all": False,
                  "barycenter_residual_max": 2.0e-3},
        "group_analogies": {"benchmark": {"n_covered": 100, "n_reranked": 20, "overall": overall}},
        "positional": {"r2_offset": 0.5, "r2_absolute": 0.3, "r2_chance": 0.01, "outlier_positions": [],
                       "null": {"offset": {"mean": 0.07, "q95": 0.09, "p": 0.005},
                                "absolute": {"mean": 0.35, "q95": 0.42, "p": 0.7}}}})
    text = R._phi_summary(res)
    assert "largest residual 2.0e-03" in text and "0.040 by chart distance on the same questions" in text
    assert "0.300 (position-permutation null mean 0.350, 95th percentile 0.420, p = 0.700)" in text
    assert "expected by chance" not in text


def test_headline_values_never_print_none_and_methods_state_provenance():
    table = R._kv_table({"rank chart / rerank": "132 / None", "missing": None})
    assert "None" not in table and "132 / –" in table
    ctx = SimpleNamespace(vocab=SimpleNamespace(english=True), coverage={"candidates": {"prefix_filter": {
        "applied": True, "max_continuation_fraction": 0.9, "n_removed": 3633}}})
    text = R._candidates_sentences(SimpleNamespace(counts_source="train_cache_bincount"), ctx)
    assert "bincount of the full cached training stream" in text
    assert "0.9" in text and "3633 tokens" in text and "curated and benchmark words are exempt" in text
    none = R._candidates_sentences(SimpleNamespace(counts_source="unavailable"),
                                   SimpleNamespace(vocab=SimpleNamespace(english=True), coverage={}))
    assert "Training counts are unavailable" in none and "Byte-pair" not in none


def test_view_selection_and_option_validation(tmp_path):
    run_dir, _ = _tiny_run(tmp_path)
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out", views=("mu_raw", "decode_mu"))
    views = R.build_views(run, ctx)
    assert set(views) == {"mu_raw"}
    assert any("decode_mu" in note for note in ctx.cache["view_notes"])
    ctx.options["views"] = ("no_such_view",)
    with pytest.raises(ValueError):
        R.build_views(run, ctx)
    with pytest.raises(ValueError):
        R._resolve_options({"sections": ("embeddings", "nope")})
    with pytest.raises(ValueError):
        R._resolve_options({"views": ("mu_raw", "nope")})


def test_json_ready_maps_tensors_sets_and_unknowns():
    unknown = []
    out = R._json_ready({"t": torch.ones(2), "s": {3, 1}, "c": np.array([1 + 2j]), "sl": slice(0, 4),
                         "o": object()}, unknown)
    assert isinstance(out["t"], np.ndarray) and out["s"] == [1, 3] and out["sl"] == [0, 4]
    assert isinstance(out["c"][0], str) and isinstance(out["o"], str)
    assert len(unknown) == 2 and all(":" in u for u in unknown)


# ----------------------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------------------

def test_analyze_run_end_to_end(tmp_path, monkeypatch):
    from vfe3.semantics import frames, vectors

    phi_section, battery = frames.run_phi_section, vectors.run_vector_battery

    def phi_with_failed_step(run, ctx):                                # a caught, recorded sub-step failure
        res = phi_section(run, ctx)
        res.metrics.setdefault("failed_steps", []).append({"step": "planted_phi_step",
                                                           "error": "RuntimeError: planted phi step"})
        return res

    def battery_with_failed_step(view, ctx):
        out = battery(view, ctx)
        if view.name == "w2_chart":
            out.setdefault("failed_steps", []).append({"step": "planted_view_step", "error": "ValueError: planted"})
        return out

    monkeypatch.setattr(frames, "run_phi_section", phi_with_failed_step)
    monkeypatch.setattr(vectors, "run_vector_battery", battery_with_failed_step)
    run_dir, _ = _tiny_run(tmp_path)
    sections = ["embeddings", "sigma", "phi"]
    try:
        importlib.import_module("vfe3.semantics.contextual")        # optional section module
    except ImportError:
        pass
    else:
        sections.append("contextual")
        stream = torch.from_numpy(np.random.default_rng(2).integers(1, V, size=4096))
        monkeypatch.setattr(SemanticRun, "token_stream", lambda self, split: stream)
    out = tmp_path / "semantics"
    path, metrics = _analyze(run_dir, out, sections=tuple(sections), views=E2E_VIEWS)
    assert path == out / "REPORT.md" and path.is_file()
    for name in ("embeddings", "sigma", "phi"):
        record = metrics["sections"][name]
        assert record["status"] == "ok", record["error"]
        assert record["seconds"] is not None and record["figures"]
        for rel in record["figures"]:
            assert (out / rel).is_file(), rel
    if "contextual" in sections:
        assert metrics["sections"]["contextual"]["status"] in ("ok", "skipped", "failed")
    emb = metrics["sections"]["embeddings"]
    assert set(emb["metrics"]["batteries"]) == set(E2E_VIEWS) and not emb["metrics"]["failed_views"]
    assert emb["metrics"]["headline_view"] == "decode_W_causal"
    assert any(f.startswith("embeddings/compare/") for f in emb["figures"])
    assert (out / "tables" / "embeddings" / "mu_raw" / "neighbors.md").is_file()
    assert metrics["run"]["K"] == 4 and metrics["run"]["decode_kind"] == "linear"
    assert metrics["options"]["min_count"] == 1 and metrics["timings"]["total"] > 0
    assert metrics["coverage"]["curated_analogies"]["n_resolved"] > 0
    assert metrics["run"]["diagonal_redundancy_status"] == "exact"
    assert "diagonal_redundancy_exact" not in metrics["run"]
    text = path.read_text(encoding="utf-8")
    headings = ("## Model facts", "## At a glance", "## What is and is not gauge-meaningful", "## Headline results",
                "## Section summaries", "## Figures", "## Tables", "## Methods and references",
                "## Failures and skips", "## Timings")
    for heading in headings:
        assert heading in text, heading
    assert [text.index(h) for h in headings] == sorted(text.index(h) for h in headings)
    assert "summary could not be generated" not in text
    assert "μ (raw)" in text and "decode W (causal)" in text
    assert "(rank of queen:" in text                                    # man:woman::king:? row resolved
    assert "](embeddings/compare/" in text and "](sigma/" in text and "](phi/" in text
    failures = text[text.index("## Failures and skips"):]
    assert "No section, view or battery step failed" not in failures
    assert "section **phi**, step `planted_phi_step` failed: `RuntimeError: planted phi step`" in failures
    assert "view **w2_chart**, step `planted_view_step` failed: `ValueError: planted`" in failures
    glance = text[text.index("## At a glance"):text.index("## What is and is not gauge-meaningful")]
    assert "- μ views (μ (raw), ξ ref. content" in glance and "- decode views (decode W (causal)): " in glance
    assert "- Σ views (W2 chart" in glance and "- φ views (log Û (canonical)): " in glance
    headline = text[text.index("## Headline results"):text.index("## Section summaries")]
    assert "None" not in headline
    google_total = metrics["coverage"]["google_analogies"]["n_total"]
    if google_total:
        assert f"/{google_total} (" in headline                         # coverage against the full benchmark
    assert "not adjusted for selection" in text
    assert "were measured to leave the logits unchanged to fp32 round-off for the configuration class" in text
    assert "add-one unigram table" in text                               # count provenance of this run


def test_failures_are_recorded_and_the_run_continues(tmp_path, monkeypatch):
    from vfe3.semantics import gaussians, vectors
    import vfe3.semantics as package

    def failing_sigma(run, ctx):
        raise RuntimeError("planted sigma failure")

    battery = vectors.run_vector_battery

    def failing_view(view, ctx):
        if view.name == "mu_sigma_units":
            raise RuntimeError("planted view failure")
        return battery(view, ctx)

    monkeypatch.setattr(gaussians, "run_sigma_section", failing_sigma)
    monkeypatch.setattr(vectors, "run_vector_battery", failing_view)
    monkeypatch.setitem(sys.modules, "vfe3.semantics.contextual", None)   # the lazy import must fail
    monkeypatch.delattr(package, "contextual", raising=False)
    run_dir, _ = _tiny_run(tmp_path)
    out = tmp_path / "semantics"
    path, metrics = _analyze(run_dir, out, sections=("embeddings", "sigma", "phi", "contextual"),
                             views=("mu_raw", "mu_sigma_units"))
    sections = metrics["sections"]
    assert sections["sigma"]["status"] == "failed"
    assert "Traceback" in sections["sigma"]["error"] and "planted sigma failure" in sections["sigma"]["error"]
    assert sections["contextual"]["status"] == "failed"
    assert "import of vfe3.semantics.contextual halted" in sections["contextual"]["error"]
    assert sections["phi"]["status"] == "ok" and sections["embeddings"]["status"] == "ok"
    emb = sections["embeddings"]["metrics"]
    assert set(emb["batteries"]) == {"mu_raw"}
    assert "planted view failure" in emb["failed_views"]["mu_sigma_units"]
    text = path.read_text(encoding="utf-8")
    assert "section **sigma** failed" in text and "planted sigma failure" in text
    assert "section **contextual** failed" in text and "view **mu_sigma_units** failed" in text
