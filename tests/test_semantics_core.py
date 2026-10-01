r"""Core of :mod:`vfe3.semantics`: vocabularies, run loading, table accessors, resources, context.

Every model here is tiny (vocabulary 24, K=4, two heads, one layer) and saved to a temporary run
directory, so the loader is exercised end to end on CPU without any real checkpoint.
"""
from tests.profiles import test_config

import json
from dataclasses import asdict, replace
from types import SimpleNamespace
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel
from vfe3.semantics.common import AnalysisContext
from vfe3.semantics.resources import (
    load_fields,
    load_google_analogies,
    load_relations,
    load_similarity,
    resolve_words,
)
from vfe3.semantics.run import (
    counts_from_unigram_log_prior,
    load_semantic_run,
    prepare_context,
)
from vfe3.semantics.vocab import TOKEN_CLASS_NAMES, custom_vocab, vocab_for_run


STRINGS = ["<unk>", "king", "queen", "man", "woman", "France", "Paris", "Germany", "Berlin", "dog",
           "dogs", "cat", "cats", "red", "blue", "one", "two", "the", ",", "Hello", "xyz1", "ing",
           "3", "zebra"]
COUNTS = np.array([0, 50, 40, 60, 55, 30, 25, 28, 22, 20, 8, 18, 3, 45, 12, 100, 90, 500, 400, 7,
                   6, 9, 11, 2], dtype=np.int64)
V = len(STRINGS)
ID = {s: i for i, s in enumerate(STRINGS)}


def _gpt2_encoding():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception as exc:                                           # offline without the BPE cache
        pytest.skip(f"gpt2 encoding unavailable: {exc}")


def _tiny_model(**over) -> "tuple[VFE3Config, VFEModel]":
    options = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, e_phi_lr=0.0)
    options.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(**options)
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    with torch.no_grad():                                              # move tables off their init
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        if hasattr(model.prior_bank, "decode_sigma_log_embed"):
            model.prior_bank.decode_sigma_log_embed.normal_(0.0, 0.5)
            model.prior_bank.decode_mu_embed.normal_()
        model.prior_bank.decode_log_scale.fill_(0.3)
        if cfg.decode_unigram_prior:
            model.prior_bank.set_unigram_log_prior(torch.from_numpy(COUNTS).float())
    return cfg, model


def _save_run(root: Path, name: str, cfg: VFE3Config, model: VFEModel, *,
              selection_identity: Optional[dict] = None, dataset: str = "wikitext-103") -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "config": serialize_config(cfg),
                "selection_data_identity": selection_identity}, run_dir / "best_model.pt")
    (run_dir / "config.json").write_text(
        json.dumps({"config": serialize_config(cfg), "dataset": dataset}), encoding="utf-8")
    return run_dir


def _load(run_dir: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_semantic_run(run_dir, vocab=custom_vocab(STRINGS), **kwargs)


def _embed_blocks(model: VFEModel, coords: torch.Tensor, slices: Sequence[slice]) -> "list[np.ndarray]":
    X = model.group.embed_algebra(coords.to(torch.float64), dtype=torch.float64)   # (n, K, K)
    return [X[:, s, s].numpy() for s in slices]


# ----------------------------------------------------------------------------------------------
# Vocabularies
# ----------------------------------------------------------------------------------------------

def test_custom_vocab_word_policy_classes_and_display():
    vocab = custom_vocab(STRINGS)
    assert vocab.kind == "custom" and vocab.size == V
    assert vocab.word_id("king") == ID["king"]
    assert vocab.word_id("King") is None                                # exact, no case fallback
    assert vocab.word_id("<unk>") is None and vocab.word_id("absent") is None
    mask = vocab.word_mask()
    assert mask[ID["king"]] and mask[ID["Hello"]] and not mask[ID[","]] and not mask[ID["xyz1"]]
    assert not mask[ID["<unk>"]]
    codes, names = vocab.token_classes()
    assert names == list(TOKEN_CLASS_NAMES)
    cls = {s: names[codes[ID[s]]] for s in ("the", "king", "Hello", ",", "3", "xyz1", "<unk>")}
    assert cls == {"the": "function word", "king": "content word", "Hello": "capitalized word",
                   ",": "punctuation", "3": "number", "xyz1": "other", "<unk>": "other"}
    assert vocab.display(ID["king"]) == "king" and vocab.display(0) == "<0>"
    long = custom_vocab(["a" * 30, "x\ny"])
    assert long.display(0) == "a" * 15 + "…" and long.display(1) == "x\\ny"


def test_non_english_vocab_classes_are_all_other():
    vocab = custom_vocab(STRINGS, english=False)
    codes, names = vocab.token_classes()
    assert set(names[c] for c in codes) == {"other"}


def test_gpt2_vocab_single_token_policy():
    encoding = _gpt2_encoding()
    vocab = vocab_for_run(None, {"tokenizer_encoding": "gpt2", "dataset": "wikitext-103"}, 50257)
    assert vocab.kind == "gpt2" and vocab.size == 50257 and vocab.english
    king = encoding.encode_ordinary(" king")[0]
    assert vocab.word_id("king") == king and vocab.strings[king] == " king"
    assert vocab.word_id("King") == encoding.encode_ordinary(" King")[0] != king
    assert vocab.word_id("swam") is None                                # " swam" is two tokens
    bare = encoding.encode_ordinary("king")
    assert len(bare) == 1 and vocab.display(bare[0]) == "·king"         # continuation subword
    assert vocab.display(king) == "king"
    assert not vocab.valid_text.all()                                   # byte fragments exist
    invalid = int(np.flatnonzero(~vocab.valid_text)[0])
    assert vocab.display(invalid) == f"<{invalid}>"
    mask = vocab.word_mask()
    assert mask[king] and not mask[bare[0]]
    codes, names = vocab.token_classes()
    probe = {" the": "function word", " ,": "punctuation", " King": "capitalized word",
             " king": "content word", " 1990": "number"}
    for text, expected in probe.items():
        ids = encoding.encode_ordinary(text)
        assert len(ids) == 1 and names[codes[ids[0]]] == expected, text
    with pytest.raises(ValueError):
        vocab_for_run(None, {"tokenizer_encoding": "gpt2"}, 50000)      # size mismatch
    with pytest.raises(ValueError):
        vocab_for_run(None, None, 50257)                                # tokenizer unknown


def test_word_level_vocab_from_sidecar(tmp_path, monkeypatch):
    from vfe3.data import datasets
    from vfe3.data.word_vocab import WordVocabulary, vocabulary_path

    monkeypatch.setattr(datasets, "default_cache_dir", lambda: tmp_path)
    WordVocabulary(["<unk>", "the", "king", ","]).save(vocabulary_path("wikitext-103-word", tag="word"))
    vocab = vocab_for_run("wikitext-103-word", None, 4)
    assert vocab.kind == "word"
    assert vocab.word_id("king") == 2 and vocab.word_id("<unk>") is None
    assert vocab.word_mask().tolist() == [False, True, True, False]


# ----------------------------------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------------------------------

def test_counts_inversion_matches_the_smoothing_formula():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5000, size=500)
    counts[:7] = 0
    counts[7] = 2_000_000                                               # a Zipf head
    u = np.log((counts + 1.0) / (counts.sum() + counts.size)).astype(np.float32)
    recovered, diagnostics = counts_from_unigram_log_prior(u)
    assert np.array_equal(recovered, counts)
    assert diagnostics["n_zero"] == int((counts == 0).sum())
    assert diagnostics["max_abs_log_roundtrip_error"] < 1e-5
    shifted = counts + 3                                                # rarest count is 3, not 0
    u_shifted = np.log((shifted + 1.0) / (shifted.sum() + shifted.size))
    with pytest.raises(ValueError):
        counts_from_unigram_log_prior(u_shifted)
    with pytest.raises(ValueError):
        counts_from_unigram_log_prior(np.zeros(10))                     # unset table


# ----------------------------------------------------------------------------------------------
# Run loading and table accessors
# ----------------------------------------------------------------------------------------------

def test_load_semantic_run_linear_decode_tables(tmp_path):
    cfg, model = _tiny_model(decode_mode="linear")
    run = _load(_save_run(tmp_path, "linear", cfg, model))
    assert (run.K, run.n_layers, run.n_heads, run.n_gen) == (4, 1, 2, 8)
    assert run.block_slices == [slice(0, 2), slice(2, 4)]
    assert run.frames_active and run.phi_frozen and run.decode_kind == "linear"
    assert run.counts_source == "unigram_log_prior" and np.array_equal(run.counts, COUNTS)
    assert run.dataset == "wikitext-103"                               # from config.json
    ids = [3, 1, 17, 1]
    pb = model.prior_bank
    np.testing.assert_array_equal(run.mu(ids), pb.mu_embed.detach()[ids].double().numpy())
    with torch.no_grad():
        encoded = pb.encode(torch.tensor([ids]))
    np.testing.assert_allclose(run.log_var(ids), torch.log(encoded.sigma[0].double()).numpy(), atol=1e-6)
    np.testing.assert_array_equal(run.phi(ids), pb.phi_embed.detach()[ids].double().numpy())
    for got, want in zip(run.phi_blocks(ids), _embed_blocks(model, pb.phi_embed.detach()[ids], run.block_slices)):
        assert got.shape == (4, 2, 2)
        np.testing.assert_allclose(got, want, atol=1e-12)
    positions = [0, 5, 2]
    Y = model._pos_phi_right(torch.zeros(6, run.n_gen)).detach()        # (N, n_gen)
    for got, want in zip(run.pos_phi_blocks(positions), _embed_blocks(model, Y[positions], run.block_slices)):
        np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_array_equal(run.decode_W(ids), pb.output_proj_weight.detach()[ids].double().numpy())
    np.testing.assert_array_equal(run.decode_b(ids), pb.output_proj_bias.detach()[ids].double().numpy())
    np.testing.assert_allclose(run.unigram_log_prior(), pb.unigram_log_prior.double().numpy())
    with pytest.raises(ValueError):
        run.decode_mu(ids)
    rebuilt = run.build_model()
    tokens = torch.tensor([[1, 3, 5, 7, 9, 11]])
    with torch.no_grad():
        _, want_logits = model.forward_beliefs(tokens, return_logits=True)
        _, got_logits = rebuilt.forward_beliefs(tokens, return_logits=True)
    torch.testing.assert_close(got_logits, want_logits)


def test_gaussian_decode_tables_reproduce_kl_logits(tmp_path):
    cfg, model = _tiny_model(decode_mode="diagonal_chunked", untie_decode_bank=True)
    run = _load(_save_run(tmp_path, "kl", cfg, model))
    assert run.decode_kind == "gaussian"
    all_ids = list(range(V))
    m, log_t = run.decode_mu(all_ids), run.decode_log_var(all_ids)     # (V, K)
    np.testing.assert_array_equal(m, model.prior_bank.decode_mu_embed.detach().double().numpy())
    assert abs(run.tau_eff() - float(model.prior_bank._tau_eff().detach())) < 1e-6   # fp32 model
    rng = np.random.default_rng(1)
    mu_q, var_q = rng.normal(size=4), np.exp(rng.normal(size=4) * 0.3)
    t = np.exp(log_t)
    kl = 0.5 * ((var_q + (mu_q - m) ** 2) / t + log_t - np.log(var_q) - 1.0).sum(-1)     # (V,)
    want = -kl / run.tau_eff() + cfg.unigram_kappa * run.unigram_log_prior()
    with torch.no_grad():
        got = model.prior_bank.decode(torch.tensor(mu_q, dtype=torch.float32).view(1, 1, 4),
                                      torch.tensor(var_q, dtype=torch.float32).view(1, 1, 4))
    np.testing.assert_allclose(got.view(-1).double().numpy(), want, rtol=1e-4, atol=1e-3)
    with pytest.raises(ValueError):
        run.decode_W(all_ids)


def test_log_var_full_and_standalone_laplace_accessor(tmp_path):
    ids = [2, 4, 6, 8]
    cfg_full, model_full = _tiny_model(family="gaussian_full", decode_mode="full_chunked")
    run_full = _load(_save_run(tmp_path, "full", cfg_full, model_full))
    with torch.no_grad():
        sigma_full = model_full.prior_bank.encode(torch.tensor([ids])).sigma[0]  # (n, K, K) diagonal
    np.testing.assert_allclose(
        run_full.log_var(ids), torch.log(torch.diagonal(sigma_full, dim1=-2, dim2=-1).double()).numpy(),
        atol=1e-6)
    run_lap = replace(run_full, cfg=SimpleNamespace(
        **{**asdict(cfg_full), "family": "laplace_diagonal"}))
    b = torch.exp(run_lap.state["prior_bank.sigma_log_embed"][ids].double())
    np.testing.assert_allclose(run_lap.log_var(ids), torch.log(2.0 * b * b).numpy(), atol=1e-6)


def test_counts_from_training_cache_and_token_stream(tmp_path):
    from vfe3.data.datasets import cache_path, cache_source_identity

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    train = torch.tensor([1, 1, 2, 17, 17, 17, 5, 23], dtype=torch.int64)
    validation = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
    torch.save(train, cache_path("wikitext-2", "train", cache_dir=cache_dir))
    torch.save(validation, cache_path("wikitext-2", "validation", cache_dir=cache_dir))
    identity = {"dataset": "wikitext-2", "split": "validation",
                "source": {"format": "pt",
                           "sha256": cache_source_identity("wikitext-2", "validation",
                                                           cache_dir=cache_dir)["sha256"]}}
    cfg, model = _tiny_model(decode_unigram_prior=False)
    run = _load(_save_run(tmp_path, "bincount", cfg, model, selection_identity=identity,
                          dataset="ignored"), cache_dir=cache_dir)
    assert run.dataset == "wikitext-2"                                  # selection identity wins
    assert run.counts_source == "train_cache_bincount"
    np.testing.assert_array_equal(run.counts, np.bincount(train.numpy(), minlength=V))
    assert torch.equal(run.token_stream("validation"), validation)
    assert torch.equal(run.token_stream("train"), train)


def test_counts_unavailable_without_prior_or_cache(tmp_path):
    cfg, model = _tiny_model(decode_unigram_prior=False)
    run = _load(_save_run(tmp_path, "nocounts", cfg, model), cache_dir=tmp_path / "empty")
    assert run.counts is None and run.counts_source == "unavailable"
    assert any("unavailable" in note for note in run.notes)
    ctx = prepare_context(run, tmp_path / "out", options={"threads": 1})
    assert np.array_equal(np.sort(ctx.candidate_ids), np.flatnonzero(run.vocab.word_mask()))
    assert any("counts unavailable" in note for note in ctx.coverage["notes"])


# ----------------------------------------------------------------------------------------------
# Resources and context
# ----------------------------------------------------------------------------------------------

def test_curated_resources_are_well_formed_single_gpt2_tokens():
    fields = load_fields()
    assert len(fields) >= 22
    for name, words in fields.items():
        assert name in ("months", "weekdays") or 15 <= len(words) <= 35, name
    relations = load_relations()
    assert all(len(pairs) >= 8 for pairs in relations["relations"].values())
    assert len(relations["axes"]) >= 3 and len(relations["probe_words"]) == 24
    words = {w for ws in fields.values() for w in ws}
    words |= {w for pairs in relations["relations"].values() for pair in pairs for w in pair}
    words |= {w for q in relations["curated_analogies"] for w in q[:4]}
    words |= set(relations["hypernyms"]) | {w for ws in relations["hypernyms"].values() for w in ws}
    words |= {w for axis in relations["axes"].values() for pair in axis["pairs"] for w in pair}
    words |= {w for axis in relations["axes"].values() for w in axis["words"]}
    words |= set(relations["probe_words"]) | set(relations["ambiguous_words"])
    encoding = _gpt2_encoding()
    split = sorted(w for w in words if len(encoding.encode_ordinary(" " + w)) != 1)
    assert split == []


def test_benchmark_parsers(tmp_path):
    questions = tmp_path / "questions-words.txt"
    questions.write_text(": capital\nAthens Greece Paris France\n\n: gram3\nbig bigger small smaller\n",
                         encoding="utf-8")
    assert load_google_analogies(questions) == [
        ("capital", "Athens", "Greece", "Paris", "France"),
        ("gram3", "big", "bigger", "small", "smaller")]
    simlex = tmp_path / "simlex.txt"
    simlex.write_text("# comment\nword1\tword2\tscore\nold\tnew\t1.58\nsmart\tintelligent\t9.2\n",
                      encoding="utf-8")
    assert load_similarity("simlex999", simlex) == [("old", "new", 1.58), ("smart", "intelligent", 9.2)]
    with pytest.raises(ValueError):
        load_similarity("men3000", simlex)
    resolved, missing = resolve_words(custom_vocab(STRINGS), ["king", "cats", "nope", "king"],
                                      {ID["king"]})
    assert resolved == {"king": ID["king"]} and missing == ["cats", "nope"]


def test_prepare_context_candidates_and_forced_resources(tmp_path):
    cfg, model = _tiny_model()
    run = _load(_save_run(tmp_path, "ctx", cfg, model))
    ctx = prepare_context(run, tmp_path / "semantics",
                          options={"min_count": 5, "max_candidates": 3, "seed": 1, "threads": 1})
    assert isinstance(ctx, AnalysisContext)
    assert ctx.figures_dir.is_dir() and ctx.tables_dir.is_dir()
    assert (ctx.seed, ctx.threads, ctx.device) == (1, 1, "cpu")
    assert ctx.candidate_ids[:3].tolist() == [ID["the"], ID["one"], ID["two"]]
    candidates = set(ctx.candidate_ids.tolist())
    for excluded in ("cats", "zebra", "<unk>", ",", "xyz1"):           # floor, non-word, unk
        assert ID[excluded] not in candidates, excluded
    tail = ctx.counts[ctx.candidate_ids[3:]]
    assert np.all(np.diff(tail) <= 0)                                   # forced tail by frequency
    assert ctx.lexicon["numbers"] == [ID["one"], ID["two"]]
    assert set(ctx.lexicon["cities"]) == {ID["Paris"], ID["Berlin"]}
    assert (ID["man"], ID["woman"]) in ctx.relations["gender"]
    assert (ID["France"], ID["Paris"]) in ctx.relations["country_capital"]
    assert (ID["dog"], ID["dogs"]) in ctx.relations["singular_plural"]
    assert (ID["cat"], ID["cats"]) not in ctx.relations["singular_plural"]
    assert ("man", "woman", "king", "queen") in {
        tuple(STRINGS[i] for i in q[1:]) for q in ctx.curated_analogies}
    assert ID["king"] in ctx.probe_ids
    resource_ids = {i for ids in ctx.lexicon.values() for i in ids}
    resource_ids |= {i for pairs in ctx.relations.values() for pair in pairs for i in pair}
    resource_ids |= {i for q in (*ctx.analogies, *ctx.curated_analogies) for i in q[1:]}
    resource_ids |= {i for rows in ctx.similarity.values() for row in rows for i in row[:2]}
    assert resource_ids <= candidates
    coverage = ctx.coverage
    assert coverage["candidates"]["n_top"] == 3
    assert coverage["candidates"]["n_forced"] == len(ctx.candidate_ids) - 3
    assert coverage["relations"]["per_relation"]["singular_plural"]["missing"].count("cat:cats") == 1
    assert coverage["fields"]["n_resolved"] == sum(len(v) for v in ctx.lexicon.values())


class _FakeBPEEncoding:
    r"""Single-token lookup standing in for a tiktoken encoding (``" word"`` -> its id, else two ids)."""

    def __init__(self, strings: Sequence[str]) -> None:
        self._lookup = {s: i for i, s in enumerate(strings)}

    def encode_ordinary(self, text: str) -> "list[int]":
        token = self._lookup.get(text)
        return [token] if token is not None else [0, 0]


def test_prefix_filter_applies_without_counts(tmp_path, monkeypatch):
    from vfe3.semantics.vocab import Vocab

    strings = ["<|endoftext|>", " Squ", "irrel", " king", " queen", " man", " woman", " the", ".", " cat", "s"]
    vocab = Vocab("gpt2", strings, np.ones(len(strings), dtype=bool), english=True,
                  encoding=_FakeBPEEncoding(strings))
    cfg, model = _tiny_model(vocab_size=len(strings), decode_unigram_prior=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = load_semantic_run(_save_run(tmp_path, "bpe", cfg, model), vocab=vocab, cache_dir=tmp_path / "empty")
    assert run.counts is None
    # " Squ" is always followed by "irrel" and " cat" by "s" (both prefixes); " king" never is.
    stream = torch.tensor([1, 2] * 20 + [9, 10] * 5 + [3, 4, 5, 6, 7, 8, 3, 7], dtype=torch.long)
    monkeypatch.setattr(run, "token_stream", lambda split: stream)
    ctx = prepare_context(run, tmp_path / "out", options={"threads": 1})
    record = ctx.coverage["candidates"]["prefix_filter"]
    assert record["applied"] is True and record["n_removed"] == 2
    assert sorted(record["examples"]) == ["Squ", "cat"]
    candidates = set(ctx.candidate_ids.tolist())
    assert 1 not in candidates and {3, 4, 5, 6, 7} <= candidates
    assert 9 in ctx.lexicon["animals"] and 9 in candidates             # " cat" is a resource word: exempt
    assert any("counts unavailable" in note for note in ctx.coverage["notes"])


def test_non_english_candidates_are_valid_text_tokens(tmp_path):
    cfg, model = _tiny_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = load_semantic_run(_save_run(tmp_path, "ja", cfg, model), vocab=custom_vocab(STRINGS, english=False))
    ctx = prepare_context(run, tmp_path / "out", options={"min_count": 5, "threads": 1})
    expected = {i for i, s in enumerate(STRINGS) if s != "<unk>" and COUNTS[i] >= 5}
    assert set(ctx.candidate_ids.tolist()) == expected                  # no whole-word mask: ",", "3" kept
    assert ctx.candidate_ids[:2].tolist() == [ID["the"], ID[","]]
    assert ctx.lexicon == {} and ctx.analogies == [] and ctx.similarity == {}
    assert any("non-English" in note for note in ctx.coverage["notes"])
    assert ctx.coverage["candidates"]["prefix_filter"]["applied"] is False


def test_continuation_fraction_flags_bpe_prefix_pieces() -> None:
    from vfe3.semantics.run import continuation_fraction

    vocab = custom_vocab([" Squ", "irrel", " king", " the", ".", " cat", "alog"])
    # " Squ" is always followed by the continuation "irrel"; " cat" once in two; " king" never.
    stream = torch.tensor([0, 1, 2, 3, 5, 6, 5, 4, 0, 1, 2, 4], dtype=torch.long)
    frac = continuation_fraction(stream, vocab)
    assert frac.shape == (7,)
    assert frac[0] == 1.0 and frac[2] == 0.0 and frac[5] == 0.5
    assert frac[4] == 0.0                                    # "." is followed by " Squ" (spaced)
