r"""Across-seed aggregation of the semantic analysis (:mod:`vfe3.semantics.multiseed`).

Unit tests of the headline table, the data-order caveat, the configuration check and the two
stability metrics (their invariance class and their chance levels), and an end-to-end aggregation of
three tiny seeds (K=4, two heads, two layers, the 90-word vocabulary of ``test_semantics_report``)
analyzed with the quick preset. Figures are written at 40 dpi in DejaVu Sans.
"""
from tests.profiles import test_config

import itertools
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
import torch

from tests.test_semantics_report import COUNTS, SMALL, STRINGS, V
from vfe3.config import serialize_config
from vfe3.model.model import VFEModel
from vfe3.semantics import multiseed as M
from vfe3.semantics import report as R
from vfe3.semantics import stats
from vfe3.semantics.vocab import custom_vocab


@pytest.fixture(autouse=True)
def _fast_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The package style at 40 dpi with one font, in every module that draws here."""
    from vfe3.semantics import common, frames, gaussians, vectors

    original = common.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"],
                             "font.sans-serif": ["DejaVu Sans"]})

    for module in (common, frames, gaussians, vectors, M):
        if hasattr(module, "apply_style"):
            monkeypatch.setattr(module, "apply_style", low_dpi_style)


# ----------------------------------------------------------------------------------------------
# Tiny seeds
# ----------------------------------------------------------------------------------------------

def _seed_run(root: Path, name: str, seed: int, data_seed: Optional[int], **over) -> Path:
    r"""A tiny trained-looking run whose tables are drawn from ``seed`` (and a provenance.json)."""
    options = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=2,
                   attention_mode="projected_predictive", seed=seed,
                   norm_type_block="none", layernorm_affine=False,           # pinned: independent of
                   skip_belief_sigma_update=False)                           # the config defaults
    options.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(**options)
        torch.manual_seed(seed)
        model = VFEModel(cfg).eval()
    bank = model.prior_bank
    with torch.no_grad():
        bank.sigma_log_embed.normal_(0.0, 0.3)
        bank.phi_embed.normal_(0.0, 0.3)
        bank.set_unigram_log_prior(torch.from_numpy(COUNTS).float())
    run_dir = root / name
    run_dir.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "config": serialize_config(cfg),
                "selection_data_identity": None}, run_dir / "best_model.pt")
    (run_dir / "config.json").write_text(
        json.dumps({"config": serialize_config(cfg), "dataset": "wikitext-103"}), encoding="utf-8")
    if data_seed is not None:
        (run_dir / "provenance.json").write_text(json.dumps({"seed": seed, "data_seed": data_seed}), encoding="utf-8")
    return run_dir


def _quick_analysis(run_dir: Path, **over) -> Path:
    small = {k: v for k, v in SMALL.items() if k not in ("umap", "geometry_max_points", "frame_rerank_questions")}
    options = {**small, "preset": "quick", "sections": ("embeddings",), "views": ("mu_raw", "decode_W_causal"),
               "geometry_max_points": 100, **over}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R.analyze_run(run_dir, vocab=custom_vocab(STRINGS), options=options)
    return run_dir / "semantics"


# ----------------------------------------------------------------------------------------------
# Headline table, data order, configuration
# ----------------------------------------------------------------------------------------------

def _metrics(acc: float, only_b: float, purity: float, macro: Optional[float], ce: Optional[float]) -> Dict:
    ci = lambda m: {"mean": m, "ci_lo": m - 0.01, "ci_hi": m + 0.01}   # noqa: E731
    overall = {"methods": {"3cosadd": {"acc": ci(acc)}, "only_b": {"acc": ci(only_b)}}}
    if macro is not None:
        overall["macro"] = {"methods": {"3cosadd": {"acc": ci(macro)}, "only_b": {"acc": ci(only_b / 2)}}}
    battery = {"analogy_benchmark": {"overall": overall},
               "fields": {"purity": {"purity": purity, "null_mean": 0.03, "z": 10 * purity}},
               "similarity": {"simlex999": {"rho": purity / 2}, "wordsim353": {"rho": purity / 3}}}
    contextual = {} if ce is None else {"logit_lens": {"ce": [{"mean": 9.0}, {"mean": ce}], "unigram_ce": {"mean": 7.0}}}
    return {"schema": "vfe3.semantics.report/1",
            "sections": {"embeddings": {"metrics": {"batteries": {"mu_raw": battery, "decode_W_causal": battery},
                                                    "views": {"mu_raw": {"label": "μ raw"}}}},
                         "sigma": {"metrics": {"banks": {"prior": {"dispersion": {"cv_trace": 0.1 + acc}}}}},
                         "contextual": {"metrics": contextual}}}


def test_headline_metrics_summaries_and_missing_values():
    labels = ["seed 0", "seed 1", "seed 2"]
    metrics = [_metrics(0.5, 0.1, 0.8, 0.4, 3.5), _metrics(0.6, 0.2, 0.7, 0.3, 3.4), _metrics(0.7, 0.1, 0.9, None, None)]
    table = M.headline_metrics(metrics, labels)
    acc = table["view/mu_raw/google_3cosadd"]
    assert acc["values"] == {"seed 0": 0.5, "seed 1": 0.6, "seed 2": 0.7} and acc["label"] == "μ raw: Google 3CosAdd (micro)"
    s = acc["summary"]
    assert s["n"] == 3 and s["mean"] == pytest.approx(0.6) and s["sd"] == pytest.approx(0.1)
    assert (s["min"], s["max"]) == (0.5, 0.7)
    margin = table["view/decode_W_causal/google_margin"]["summary"]
    assert margin["mean"] == pytest.approx((0.4 + 0.4 + 0.6) / 3)
    macro = table["view/mu_raw/google_margin_macro"]
    assert macro["missing"] == ["seed 2"] and macro["summary"]["n"] == 2
    assert macro["values"]["seed 0"] == pytest.approx(0.4 - 0.05)
    assert table["view/mu_raw/field_purity_z"]["values"]["seed 1"] == pytest.approx(7.0)
    assert table["view/mu_raw/wordsim353_rho"]["values"]["seed 2"] == pytest.approx(0.3)
    assert table["sigma/cv_trace"]["summary"]["mean"] == pytest.approx(0.7)
    ce = table["contextual/lens_ce_final"]
    assert ce["values"] == {"seed 0": 3.5, "seed 1": 3.4, "seed 2": None} and ce["summary"]["sd"] == pytest.approx(
        np.std([3.5, 3.4], ddof=1))
    assert "phi/group_analogy_chart_acc" not in table                  # no seed reports it
    assert M.summarize([1.0, None, float("nan")]) == {"n": 1, "mean": 1.0, "sd": None, "min": 1.0, "max": 1.0}
    assert M.common_views(metrics) == ["mu_raw", "decode_W_causal"]    # report.VIEW_NAMES order


def test_data_order_status_states_the_caveat():
    labels = ["seed 0", "seed 1", "seed 2"]
    shared = M.data_order_status([3, 3, 3], [0, 1, 2], labels)
    assert shared["status"] == "shared" and "lower bound" in shared["caveat"] and "data seed 3" in shared["caveat"]
    assert shared["data_seed_equals_seed"] is False
    distinct = M.data_order_status([0, 1, 2], [0, 1, 2], labels)
    assert distinct["status"] == "distinct" and "includes data-order variance" in distinct["caveat"]
    assert distinct["data_seed_equals_seed"] is True
    assert M.data_order_status([3, 3, 4], [0, 1, 2], labels)["status"] == "partially_shared"
    unknown = M.data_order_status([3, None, 3], [0, 1, 2], labels)
    assert unknown["status"] == "unknown" and "seed 1" in unknown["caveat"]


def test_config_homogeneity_exempts_only_the_seeds():
    base = {"K": 4, "lr": 1e-3, "seed": 0, "data_seed": 3}
    same = M.config_homogeneity([base, {**base, "seed": 1, "data_seed": 4}], ["a", "b"])
    assert same["homogeneous"] and same["differences"] == {}
    differ = M.config_homogeneity([base, {**base, "seed": 1, "lr": 2e-3}, {**base, "extra": True}], ["a", "b", "c"])
    assert not differ["homogeneous"]
    assert differ["differences"]["lr"] == {"a": 1e-3, "b": 2e-3, "c": 1e-3}
    assert differ["differences"]["extra"]["a"] == "<absent>"
    opts = M.option_differences([{"threads": 1, "device": "cpu", "min_count": 5, "drop_unknown_config_fields": True},
                                 {"threads": 8, "device": "cuda", "min_count": 6}], ["a", "b"])
    assert opts == {"min_count": {"a": 5, "b": 6}}


def test_semantics_dir_lookup_order_and_parent_discovery(tmp_path):
    metrics = json.dumps({"schema": "vfe3.semantics.report/1"})
    run = tmp_path / "parent" / "run_a"
    for sub in ("figures", "figures/semantics"):
        (run / sub).mkdir(parents=True, exist_ok=True)
        (run / sub / "metrics.json").write_text(metrics, encoding="utf-8")
    (run / "figures" / "semantics" / "metrics.json").write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    assert M.find_semantics_dir(run) == run / "figures"                # a non-semantic metrics.json is skipped
    (run / "figures" / "semantics" / "metrics.json").write_text(metrics, encoding="utf-8")
    assert M.find_semantics_dir(run) == run / "figures" / "semantics"  # the training report's analysis
    (run / "semantics").mkdir()
    (run / "semantics" / "metrics.json").write_text(metrics, encoding="utf-8")
    assert M.find_semantics_dir(run) == run / "semantics"
    (tmp_path / "parent" / "not_a_run").mkdir()
    assert M.discover_runs(tmp_path / "parent") == [run]
    with pytest.raises(FileNotFoundError):
        M.discover_runs([run, tmp_path / "parent" / "not_a_run"])


# ----------------------------------------------------------------------------------------------
# Stability metrics
# ----------------------------------------------------------------------------------------------

def _cka(Xa: np.ndarray, Xb: np.ndarray, perm=None):
    ta, tb = (torch.as_tensor(X - X.mean(axis=0), dtype=torch.float32) for X in (Xa, Xb))
    return M.cka_pair(ta, tb, M._cka_self_terms(ta), M._cka_self_terms(tb), perm=perm)


def test_stability_metrics_are_invariant_under_each_seeds_trivialization():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6)) @ rng.normal(size=(6, 6))
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Y = 2.5 * X @ Q + rng.normal(size=6)                                # c X Q + 1 t^T
    Y_wide = np.concatenate([X @ Q, np.zeros((300, 3))], axis=1) * 0.3  # another width, same Gram matrix
    nn_x = M.knn_indices(X, 10)
    for other in (Y, Y_wide):
        assert M.knn_jaccard(nn_x, M.knn_indices(other, 10)).min() == 1.0
        biased, debiased = _cka(X, other)
        assert biased == pytest.approx(1.0, abs=1e-5) and debiased == pytest.approx(1.0, abs=1e-5)
    Z = rng.normal(size=(300, 6))                                       # unrelated rows
    biased, debiased = _cka(X, Z)
    assert biased == pytest.approx(stats.linear_cka(X, Z), abs=1e-5)
    assert debiased == pytest.approx(stats.linear_cka(X, Z, debiased=True), abs=1e-5)
    assert abs(debiased) < 0.05 and 0.0 < biased < 0.1
    perm = rng.permutation(300)
    pb, pd = _cka(X, Y, perm=perm)
    assert pb == pytest.approx(stats.linear_cka(X, Y[perm]), abs=1e-5) and abs(pd) < 0.05
    # the neighbor sets exclude the word itself and are the k largest cosines of the centered rows
    Xc = X - X.mean(axis=0)
    U = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    np.testing.assert_array_equal(np.sort(nn_x, axis=1), np.sort(np.argsort(-S, axis=1)[:, :10], axis=1))


def test_jaccard_chance_matches_enumeration():
    n, k = 7, 3
    subsets = list(itertools.combinations(range(n - 1), k))
    exact = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a in subsets for b in subsets])
    assert M.jaccard_chance(n, k) == pytest.approx(exact, rel=1e-12)
    np.testing.assert_allclose(M.knn_jaccard(np.array([[0, 1, 2]]), np.array([[2, 3, 4]])), [1 / 5])


# ----------------------------------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------------------------------

def test_aggregate_seeds_end_to_end(tmp_path):
    parent = tmp_path / "study"
    for seed in range(3):
        _quick_analysis(_seed_run(parent, f"seed_{seed}", seed, data_seed=3))
    _seed_run(parent, "seed_9_not_analyzed", 9, data_seed=3)
    out = tmp_path / "multiseed"
    md_path = M.aggregate_seeds(parent, out_dir=out, vocab=custom_vocab(STRINGS), log=lambda msg: None)
    assert md_path == out / M.MARKDOWN_NAME
    payload = json.loads((out / M.JSON_NAME).read_text(encoding="utf-8"))
    assert [r["label"] for r in payload["runs"]] == ["seed 0", "seed 1", "seed 2"]
    assert all(r["preset"] == "quick" for r in payload["runs"])
    assert payload["config_homogeneity"]["homogeneous"] is True
    assert payload["data_order"]["status"] == "shared"
    assert any("seed_9_not_analyzed" in n for n in payload["notes"])
    acc = payload["metrics"]["view/decode_W_causal/google_3cosadd"]
    assert set(acc["values"]) == {"seed 0", "seed 1", "seed 2"} and acc["summary"]["n"] == 3
    assert "view/xi_ref/google_3cosadd" not in payload["metrics"]      # views were restricted
    stab = payload["stability"]
    assert all(src.startswith(R.CANDIDATE_IDS_FILE) for src in stab["candidate_sources"].values())
    for channel in M.STABILITY_CHANNELS:
        rec = stab["channels"][channel]
        assert rec["status"] == "ok", rec
        for key in ("jaccard", "cka_biased", "cka_debiased"):
            A = np.array(rec[key], dtype=np.float64)
            assert A.shape == (3, 3) and np.allclose(A, A.T) and np.allclose(np.diag(A), 1.0)
        J = np.array(rec["jaccard"])
        assert ((J >= 0) & (J <= 1)).all() and 0 < rec["jaccard_chance"] < 1
        assert rec["gauge_note"]
    assert stab["channels"]["decode_causal"]["view"] == "decode_W_causal"
    assert "exact redundancy" in stab["channels"]["mu_raw"]["gauge_note"]
    for name in ("headline_views", "headline_sections", "stability"):
        if name == "headline_sections" and not any(e["group"] != "view" for e in payload["metrics"].values()):
            continue
        assert (out / "figures" / f"{name}.png").is_file() and (out / "figures" / f"{name}.pdf").is_file()
    assert payload["figure_errors"] == {}
    text = md_path.read_text(encoding="utf-8")
    assert payload["data_order"]["caveat"] in text and "lower bound" in text
    assert "## Cross-seed stability" in text and "decode_W_causal" in text and "Jaccard chance" in text


def test_identical_seeds_are_perfectly_stable_and_pre_preset_analyses_resolve(tmp_path):
    run_a = _seed_run(tmp_path, "run_a", 0, data_seed=None)
    _quick_analysis(run_a)
    run_b = tmp_path / "run_b"
    shutil.copytree(run_a, run_b)
    (run_b / "semantics").rename(run_b / "figures")                   # a pre-preset analysis layout
    (run_b / "figures" / R.CANDIDATE_IDS_FILE).unlink()
    out = tmp_path / "multiseed"
    M.aggregate_seeds([run_a, run_b], out_dir=out, vocab=custom_vocab(STRINGS), log=lambda msg: None)
    payload = json.loads((out / M.JSON_NAME).read_text(encoding="utf-8"))
    assert [r["label"] for r in payload["runs"]] == ["run_a", "run_b"]  # shared seed: labeled by folder
    assert any("share a seed" in n for n in payload["notes"])
    assert payload["data_order"]["status"] == "unknown"
    stab = payload["stability"]
    assert "rebuilt" in stab["candidate_sources"]["run_b"]
    for channel in M.STABILITY_CHANNELS:
        rec = stab["channels"][channel]
        assert rec["status"] == "ok", rec
        assert rec["jaccard"][0][1] == 1.0
        assert rec["cka_biased"][0][1] == pytest.approx(1.0, abs=1e-6)
        assert rec["cka_debiased"][0][1] == pytest.approx(1.0, abs=1e-6)
        assert abs(rec["cka_debiased_row_permuted"][0][1]) < 0.5
    assert payload["metrics"]["view/mu_raw/field_purity"]["summary"]["sd"] == 0.0


def test_retired_config_fields_are_dropped_when_the_analysis_did(tmp_path):
    runs = []
    for seed in range(2):
        run_dir = _seed_run(tmp_path, f"seed_{seed}", seed, data_seed=seed)
        bundle = torch.load(run_dir / "best_model.pt", weights_only=True)
        bundle["config"] = {**bundle["config"], "emission_mode": "none"}
        torch.save(bundle, run_dir / "best_model.pt")
        _quick_analysis(run_dir, drop_unknown_config_fields=True, sections=("sigma",))
        runs.append(run_dir)
    with pytest.raises(ValueError, match="unknown field"):
        M.load_seed(runs[0], vocab=custom_vocab(STRINGS), drop_unknown_config_fields=False)
    out = tmp_path / "multiseed"
    M.aggregate_seeds(runs, out_dir=out, vocab=custom_vocab(STRINGS), stability=False, log=lambda msg: None)
    payload = json.loads((out / M.JSON_NAME).read_text(encoding="utf-8"))
    assert payload["data_order"]["status"] == "distinct" and payload["data_order"]["data_seed_equals_seed"]
    assert payload["stability"] == {"status": "skipped", "reason": "stability=False"}
    dropped = [n for n in payload["notes"] if "no longer declares" in n]
    assert len(dropped) == 2 and all("emission_mode='none'" in n for n in dropped)
    assert "Stability was skipped" in (out / M.MARKDOWN_NAME).read_text(encoding="utf-8")
