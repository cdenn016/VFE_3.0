r"""Audit 2026-10-16, lane C2 (semantic analysis): F70, F71, F72, F73.

Tiny models (vocabulary of about 90 words, K=4, two heads, at most two layers) and fake runs only;
nothing is trained, everything runs on the CPU.

* F70: ``decode_mode='expected_likelihood_chunked'`` reads the same Gaussian decode bank
  :math:`(m_v, t_v)` at :math:`\tau_\mathrm{eff}` as the KL decoders. Its decode kind is
  ``"gaussian_likelihood"``: the decode accessors, the ``decode_mu`` view and the decode-bank
  battery apply; only the natural-parameter view (linear in :math:`\lambda(q)` for the KL decode
  alone) is skipped.
* F71: Holm-adjusted p-values (family-wise error under any dependence) are reported next to the raw
  ones for each family: the relation transports of the phi section, the relations of an offsets view,
  the relation pairs of a concept-direction view and the scorers of a similarity comparison; the
  offsets heatmap fills only the cells that survive the adjustment.
* F72: a word-level run's vocabulary sidecar is read from the ``cache_dir`` its token streams use.
* F73: every stochastic sub-step of the phi and Sigma sections draws from its own seeded stream, so a
  budget change or skip in one step leaves the draws of the others unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_semantics_report import _context, _load, _tiny_run
from vfe3.semantics import report as R


# ----------------------------------------------------------------------------------------------
# F70: the expected-likelihood decoder is a Gaussian-bank decode kind
# ----------------------------------------------------------------------------------------------

def test_expected_likelihood_decode_reads_the_gaussian_bank(tmp_path: Path) -> None:
    run_dir, model = _tiny_run(tmp_path, decode_mode="expected_likelihood_chunked", untie_decode_bank=True)
    run = _load(run_dir)
    assert run.decode_kind == "gaussian_likelihood"
    V = run.vocab_size
    ids = list(range(V))
    bank = model.prior_bank
    m, log_t = run.decode_mu(ids), run.decode_log_var(ids)              # (V, K) each
    np.testing.assert_array_equal(m, bank.decode_mu_embed.detach().double().numpy())
    assert abs(run.tau_eff() - float(bank._tau_eff().detach())) < 1e-6
    # The accessors reproduce the expected-likelihood logits
    # -1/(2 tau_eff) sum_k [(mu_q - m_v)^2 / (sigma_q + t_v) + log(sigma_q + t_v)] + kappa log pi_v.
    rng = np.random.default_rng(4)
    mu_q, var_q = rng.normal(size=4), np.exp(rng.normal(size=4) * 0.3)
    s = var_q + np.exp(log_t)                                            # (V, K) convolved variances
    want = (-0.5 * ((mu_q - m) ** 2 / s + np.log(s)).sum(-1) / run.tau_eff()
            + run.cfg.unigram_kappa * run.unigram_log_prior())
    with torch.no_grad():
        got = bank.decode(torch.tensor(mu_q, dtype=torch.float32).view(1, 1, 4),
                          torch.tensor(var_q, dtype=torch.float32).view(1, 1, 4))
    np.testing.assert_allclose(got.view(-1).double().numpy(), want, rtol=1e-4, atol=1e-3)
    with pytest.raises(ValueError):
        run.decode_W(ids)


def test_expected_likelihood_views_keep_decode_mu_and_skip_only_the_natural_view(tmp_path: Path) -> None:
    run_dir, _ = _tiny_run(tmp_path, decode_mode="expected_likelihood_chunked", untie_decode_bank=True)
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out", views=("mu_raw", "decode_mu", "decode_natural_causal"))
    views = R.build_views(run, ctx)
    assert "decode_mu" in views and "decode_natural_causal" not in views
    np.testing.assert_array_equal(views["decode_mu"].X, run.decode_mu(ctx.candidate_ids))
    assert "expected-likelihood" in views["decode_mu"].note
    skipped = [n for n in ctx.cache["view_notes"] if "decode_natural_causal" in n]
    assert len(skipped) == 1 and "not linear" in skipped[0]
    assert R.preset_views("quick", "gaussian_likelihood") == R.QUICK_VIEWS


def test_expected_likelihood_sigma_section_analyzes_the_decode_bank(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_semantics_gaussians import _FakeRun, _context as sigma_context
    from vfe3.semantics import gaussians as G

    original = G.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"],
                             "font.sans-serif": ["DejaVu Sans"]})

    monkeypatch.setattr(G, "apply_style", low_dpi_style)
    run = _FakeRun(decode_kind="gaussian_likelihood", frames_active=False, seed=1)
    result = G.run_sigma_section(run, sigma_context(run, tmp_path))
    assert result.status == "ok"
    decode = result.metrics["banks"]["decode"]
    assert decode["tau_eff"] == 0.5 and "similarity" in decode
    assert not any("no decode-bank covariance analysis" in n for n in result.notes)


# ----------------------------------------------------------------------------------------------
# F72: the word-level vocabulary sidecar comes from the run's cache_dir
# ----------------------------------------------------------------------------------------------

def test_word_level_vocab_sidecar_follows_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_semantics_report import STRINGS, _tiny_run as tiny
    from vfe3.data import datasets
    from vfe3.data.word_vocab import WordVocabulary, vocabulary_path
    from vfe3.semantics.run import load_semantic_run
    from vfe3.semantics.vocab import vocab_for_run

    default, custom = tmp_path / "default_cache", tmp_path / "custom_cache"
    default.mkdir()
    custom.mkdir()
    monkeypatch.setattr(datasets, "default_cache_dir", lambda: default)
    dataset = "wikitext-103-word"
    words = list(STRINGS)
    words[0] = "<unk>"
    WordVocabulary(words).save(vocabulary_path(dataset, tag="word", cache_dir=custom))
    # the default cache holds a different table of the same size, which only the size check would pass
    WordVocabulary(["<unk>", *[f"other{i}" for i in range(len(words) - 1)]]).save(
        vocabulary_path(dataset, tag="word", cache_dir=default))

    vocab = vocab_for_run(dataset, None, len(words), cache_dir=custom)
    assert vocab.kind == "word" and vocab.word_id("king") == words.index("king")

    run_dir, _ = tiny(tmp_path, decode_mode="linear")
    identity = {"dataset": dataset, "split": "validation", "tokenizer_encoding": "word"}
    bundle = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    bundle["selection_data_identity"] = identity
    torch.save(bundle, run_dir / "best_model.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = load_semantic_run(run_dir, cache_dir=custom)
    assert run.vocab.word_id("king") == words.index("king")
    assert run.vocab.word_id("other1") is None


# ----------------------------------------------------------------------------------------------
# F73: each stochastic sub-step of the phi and Sigma sections draws from its own stream
# ----------------------------------------------------------------------------------------------

def _phi_draws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **options) -> dict:
    r"""The relation-transport null and the rerank-subset generator state of one phi section run."""
    import copy

    from tests.test_semantics_frames import _section_setup
    from vfe3.semantics import frames as F

    seen: dict = {}
    original = F.group_analogies

    def recording(cf, questions, **kw):
        rng = kw.get("rng")
        if rng is not None:
            seen["rerank_draw"] = copy.deepcopy(rng).integers(0, 1 << 62, size=4).tolist()
        return original(cf, questions, **kw)

    monkeypatch.setattr(F, "group_analogies", recording)
    monkeypatch.setattr(F, "apply_style", lambda: None)
    run, ctx = _section_setup(tmp_path, **options)
    res = F.run_phi_section(run, ctx)
    assert res.status == "ok" and res.metrics["failed_steps"] == []
    planted = res.metrics["relation_transports"]["planted"]
    seen.update(null_mean=planted["null_mean"], null_std=planted["null_std"], p=planted["p"])
    return seen


def test_phi_rerank_subset_and_transport_null_do_not_depend_on_the_commutator_budget(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _phi_draws(tmp_path / "a", monkeypatch, frame_commutator_pairs=40)
    b = _phi_draws(tmp_path / "b", monkeypatch, frame_commutator_pairs=17)
    assert a == b


def test_sigma_similarity_draws_do_not_depend_on_the_frequency_step(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_semantics_gaussians import _FakeRun, _context as sigma_context
    from vfe3.semantics import gaussians as G

    monkeypatch.setattr(G, "apply_style", lambda: None)
    monkeypatch.setattr(G, "_bank_outputs", lambda *args, **kwargs: None)   # figures are not under test

    def similarity(with_counts: bool) -> dict:
        run = _FakeRun(decode_kind="gaussian", frames_active=False, seed=5)
        result = G.run_sigma_section(run, sigma_context(run, tmp_path / str(with_counts), with_counts=with_counts))
        assert result.metrics["failed_steps"] == []
        banks = result.metrics["banks"]
        return {bank: banks[bank]["similarity"]["simlex999"]["scorers"] for bank in ("prior", "decode")}

    with_counts, without = similarity(True), similarity(False)
    # the frequency step draws only with counts; the similarity bootstraps of both banks are unchanged
    assert with_counts == without


# ----------------------------------------------------------------------------------------------
# F71: Holm-adjusted p-values next to the raw ones, per family of tests
# ----------------------------------------------------------------------------------------------

def test_holm_adjust_is_the_step_down_procedure() -> None:
    from vfe3.semantics.stats import holm_adjust

    got = holm_adjust([0.01, 0.04, 0.03, None, float("nan"), 0.005])
    # family of the m = 4 finite p: sorted 0.005, 0.01, 0.03, 0.04 -> 4p, 3p, 2p, 1p = 0.02, 0.03, 0.06, 0.04,
    # then the running maximum 0.02, 0.03, 0.06, 0.06
    assert got[3] is None and got[4] is None
    np.testing.assert_allclose([got[0], got[1], got[2], got[5]], [0.03, 0.06, 0.06, 0.02], rtol=1e-12)
    assert holm_adjust([0.6, 0.7]) == [1.0, 1.0] and holm_adjust([]) == []
    assert holm_adjust([0.2]) == [0.2]


def test_phi_relation_transports_report_holm_adjusted_p(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_semantics_frames import _section_setup
    from vfe3.semantics import frames as F
    from vfe3.semantics.stats import holm_adjust

    monkeypatch.setattr(F, "apply_style", lambda: None)
    run, ctx = _section_setup(tmp_path)
    ctx.relations["half"] = [(i, i + 12) for i in range(6)]
    res = F.run_phi_section(run, ctx)
    rel = res.metrics["relation_transports"]
    ok = [k for k, v in rel.items() if v["status"] == "ok"]
    assert ok == ["planted", "half"]
    want = holm_adjust([rel[k]["p"] for k in ok])
    assert [rel[k]["p_holm"] for k in ok] == pytest.approx(want)
    key = "phi: relations with consistent transports (Holm-adjusted p < 0.01 over the 2 relations)"
    assert res.headline[key] == f"{sum(p < 0.01 for p in want)} / 2"
    # frame_null = 50: the smallest attainable Holm-adjusted p is 2/51 > 0.01, which the report says
    assert any("2/(1 + 50)" in n for n in res.notes)
    assert "p (Holm)" in res.tables["relation_transports"]


def test_offsets_and_directions_carry_holm_adjusted_p() -> None:
    from scipy.stats import norm

    from vfe3.semantics import vectors
    from vfe3.semantics.common import VectorView
    from vfe3.semantics.stats import holm_adjust

    rng = np.random.default_rng(16)
    n, D = 80, 12
    X = rng.standard_normal((n, D))
    delta = 3.0 * rng.standard_normal(D)
    for i in range(10):
        X[30 + i] = X[i] + delta + 0.05 * rng.standard_normal(D)
    view = VectorView("e", "e", np.arange(n), X)
    relations = {"planted": [(i, 30 + i) for i in range(10)],
                 "random": [(int(a), int(b)) for a, b in rng.choice(np.arange(40, 60), size=(8, 2), replace=False)],
                 "other": [(int(a), int(b)) for a, b in rng.choice(np.arange(60, 80), size=(8, 2), replace=False)],
                 "tiny": [(0, 1), (2, 3)]}
    off = vectors.offset_consistency(view, relations, np.random.default_rng(1), n_perm=100)
    names = list(off["relations"])
    want = holm_adjust([off["relations"][k]["p"] for k in names])
    assert [off["relations"][k]["p_holm"] for k in names] == pytest.approx(want)

    out = vectors.concept_direction_cosines(view, relations, whiten=False, rng=np.random.default_rng(7), n_null=50)
    R = len(out["relations"])
    iu = np.triu_indices(R, k=1)
    p_normal = 2.0 * norm.sf(np.abs(out["z"][iu]))
    np.testing.assert_allclose(out["p_normal"][iu], p_normal, rtol=1e-12)
    np.testing.assert_allclose(out["p_holm"][iu], holm_adjust(list(p_normal)), rtol=1e-12)
    np.testing.assert_array_equal(out["p_holm"], out["p_holm"].T)
    assert out["n_significant_pairs_holm"] == int((np.asarray(holm_adjust(list(p_normal))) < 0.05).sum())
    assert out["n_significant_pairs_holm"] <= out["n_significant_pairs"]


def test_offset_heatmap_fills_only_holm_significant_cells(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from vfe3.semantics import vectors

    def rec(p: float) -> dict:
        return {"ocs": 0.5, "null_mean": 0.1, "effect_size": 0.4, "p": p}

    # one view, four relations: raw p < 0.05 for three; Holm keeps 0.001 * 4 and 0.01 * 3 below 0.05 only
    results = {"v": {"offsets": {"relations": {"a": rec(0.001), "b": rec(0.01), "c": rec(0.04), "d": rec(0.5)}}}}
    ctx = SimpleNamespace(options={})
    fig = vectors._compare_offsets(results, {"v": SimpleNamespace(label="v")}, ctx, ["v"])
    ax = fig.axes[0]
    marks = {line.get_label(): len(line.get_xdata()) for line in ax.lines}
    assert marks == {"holm": 2, "raw": 1}


def test_gaussian_similarity_reports_holm_over_the_scorers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_semantics_gaussians import _FakeRun, _context as sigma_context
    from vfe3.semantics import gaussians as G
    from vfe3.semantics.stats import holm_adjust

    monkeypatch.setattr(G, "apply_style", lambda: None)
    monkeypatch.setattr(G, "_bank_outputs", lambda *args, **kwargs: None)
    run = _FakeRun(frames_active=False, seed=2)
    result = G.run_sigma_section(run, sigma_context(run, tmp_path))
    per = result.metrics["banks"]["prior"]["similarity"]["simlex999"]["scorers"]
    for field in ("delta_vs_cos_mu", "delta_vs_euclid_mu"):
        keys = [k for k, _ in G._SCORERS if field in per[k]]
        want = holm_adjust([per[k][field]["p"] for k in keys])
        assert [per[k][field]["p_holm"] for k in keys] == pytest.approx(want), field
