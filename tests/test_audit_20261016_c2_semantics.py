r"""Audit 2026-10-16, lane C2 (semantic analysis): F70, F71, F72, F73.

Tiny models (vocabulary of about 90 words, K=4, two heads, at most two layers) and fake runs only;
nothing is trained, everything runs on the CPU.

* F70: ``decode_mode='expected_likelihood_chunked'`` reads the same Gaussian decode bank
  :math:`(m_v, t_v)` at :math:`\tau_\mathrm{eff}` as the KL decoders. Its decode kind is
  ``"gaussian_likelihood"``: the decode accessors, the ``decode_mu`` view and the decode-bank
  battery apply; only the natural-parameter view (linear in :math:`\lambda(q)` for the KL decode
  alone) is skipped.
* F72: a word-level run's vocabulary sidecar is read from the ``cache_dir`` its token streams use.
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
