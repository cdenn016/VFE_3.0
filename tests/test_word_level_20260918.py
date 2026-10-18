r"""Word-level WikiText-103 tokenization: vocabulary, cache, and loader wiring.

The word-level setting is a second tokenizer alongside the GPT-2 BPE caches, so these contracts
pin three things. The tokenization is the corpus's own whitespace split with no normalization. The
vocabulary puts the corpus's literal ``<unk>`` marker at id 0 and maps any unseen word onto that
same id -- it is not a synthetic extra slot, which is why building it cannot produce a duplicate.
And the loader reports word caches as word-level everywhere it previously assumed a tiktoken
encoding, while the GPT-2 and cl100k paths keep their exact prior behavior. The published stream
(``wikitext-103-word-eos``, 2026-09-28) appends one ``<eos>`` per row, fixes it at id 1 and is held to
the published vocabulary and token counts; the no-``<eos>`` dataset keeps its exact prior behavior.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

import build_wikitext103_word_cache as builder
import vfe3.data.datasets as datasets_mod
from vfe3.data.datasets import (
    _load_word_vocabulary,
    _recorded_source_chars,
    _tokenizer_tag,
    cache_path,
    get_tiktoken_byte_decoder,
    get_tiktoken_decoder,
    is_word_level,
    tiktoken_encoding_name,
    tokenizer_vocab_size,
)
from vfe3.data.word_vocab import (
    REFERENCE_N_TOKENS_EOS,
    REFERENCE_VOCAB_SIZE,
    REFERENCE_VOCAB_SIZE_EOS,
    WORD_EOS,
    WORD_UNK,
    WordVocabulary,
    appends_eos,
    count_tokens,
    reference_vocab_size,
    tokenize_row,
    vocabulary_from_counts,
    vocabulary_size_for_min_count,
    vocabulary_path,
)

WORD_DATASET = "wikitext-103-word"


# ---- tokenization ----------------------------------------------------------------------------

def test_tokenize_row_is_the_corpus_whitespace_split() -> None:
    """Punctuation stays its own token and casing is preserved: no normalization is applied."""
    assert tokenize_row(" Homarus gammarus , known as the European lobster . ") == [
        "Homarus", "gammarus", ",", "known", "as", "the", "European", "lobster", "."]
    assert tokenize_row("  ") == []
    assert tokenize_row("") == []
    assert tokenize_row(" = Section = \n") == ["=", "Section", "="]


def test_tokenize_row_collapses_internal_whitespace() -> None:
    assert tokenize_row("a\t\tb\nc") == ["a", "b", "c"]


# ---- vocabulary ------------------------------------------------------------------------------

def test_unk_marker_occupies_id_zero_rather_than_duplicating() -> None:
    """The corpus contains a literal <unk>; it becomes the unk id instead of a second entry."""
    counts = Counter({WORD_UNK: 441_959, "the": 100, "cat": 9})
    words = vocabulary_from_counts(counts)
    assert words == [WORD_UNK, "the", "cat"]
    assert words.count(WORD_UNK) == 1


def test_unseen_words_and_the_marker_both_map_to_id_zero() -> None:
    vocabulary = WordVocabulary(vocabulary_from_counts(Counter({WORD_UNK: 5, "the": 4})))
    assert vocabulary.unk_id == 0
    assert vocabulary.encode(["the", "<unk>", "never-seen"]) == [1, 0, 0]


def test_frequency_order_with_deterministic_tie_break() -> None:
    """Equal counts order lexicographically, so ids are a function of the corpus, not of hashing."""
    counts = Counter({"b": 7, "a": 7, "c": 7, "rare": 1})
    forward = vocabulary_from_counts(counts)
    backward = vocabulary_from_counts(Counter(dict(reversed(list(counts.items())))))
    assert forward == backward == [WORD_UNK, "a", "b", "c", "rare"]


def test_min_count_drops_rare_types_but_never_the_unk_id() -> None:
    counts = Counter({WORD_UNK: 1, "common": 50, "rare": 1})
    assert vocabulary_from_counts(counts, min_count=0) == [WORD_UNK, "common", "rare"]
    assert vocabulary_from_counts(counts, min_count=1) == [WORD_UNK, "common"]
    assert vocabulary_size_for_min_count(counts, min_count=1) == 2


def test_min_count_rejects_negative() -> None:
    with pytest.raises(ValueError, match="min_count"):
        vocabulary_from_counts(Counter({"a": 1}), min_count=-1)


def test_decode_renders_the_unk_id_as_its_marker() -> None:
    vocabulary = WordVocabulary([WORD_UNK, "a", "b"])
    assert vocabulary.decode([0, 1, 2]) == "<unk> a b"


def test_vocabulary_round_trips_through_json(tmp_path: Path) -> None:
    vocabulary = WordVocabulary([WORD_UNK, "a", "b"], min_count=3)
    path = tmp_path / "vocab.json"
    vocabulary.save(path)
    reloaded = WordVocabulary.load(path)
    assert reloaded.words == vocabulary.words
    assert reloaded.min_count == 3
    assert reloaded.vocab_size == 3


def test_vocabulary_load_fails_closed_on_a_bad_payload(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"schema_version": 999, "words": ["<unk>"], "vocab_size": 1}))
    with pytest.raises(ValueError, match="schema_version"):
        WordVocabulary.load(path)
    path.write_text(json.dumps({"schema_version": 1, "words": ["<unk>"], "vocab_size": 2}))
    with pytest.raises(ValueError, match="vocab_size"):
        WordVocabulary.load(path)


def test_vocabulary_rejects_a_table_not_starting_with_unk() -> None:
    with pytest.raises(ValueError, match="must start with"):
        WordVocabulary(["a", "b"])


# ---- loader wiring ---------------------------------------------------------------------------

def test_word_dataset_reports_word_level_everywhere() -> None:
    assert _tokenizer_tag(WORD_DATASET) == "word"
    assert is_word_level(WORD_DATASET) is True
    assert is_word_level("wikitext-103") is False
    assert tiktoken_encoding_name(WORD_DATASET) == "word"
    assert cache_path(WORD_DATASET, "train").name == "wikitext-103-word_train_word_tokens.pt"


def test_existing_tokenizer_paths_are_unchanged() -> None:
    assert _tokenizer_tag("wikitext-103") == "tiktoken"
    assert tokenizer_vocab_size("wikitext-103") == 50257
    assert tiktoken_encoding_name("wikitext-103") == "gpt2"
    assert _tokenizer_tag("wiki-ja") == "tiktoken_cl100k"
    assert tokenizer_vocab_size("wiki-ja") == 100277
    assert tiktoken_encoding_name("wiki-ja") == "cl100k_base"


def test_word_tokenizer_has_no_byte_decoder() -> None:
    assert get_tiktoken_byte_decoder(WORD_DATASET) is None


def test_word_vocab_size_falls_back_to_the_reference_without_a_build(tmp_path: Path,
                                                                     monkeypatch) -> None:
    """Before any build the bound is the reference size rather than a KeyError."""
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    assert datasets_mod.tokenizer_vocab_size(WORD_DATASET) == REFERENCE_VOCAB_SIZE


def test_word_vocab_size_reads_the_built_sidecar(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    WordVocabulary([WORD_UNK, "a", "b", "c"]).save(
        vocabulary_path(WORD_DATASET, tag="word", cache_dir=tmp_path))
    assert datasets_mod.tokenizer_vocab_size(WORD_DATASET) == 4


def test_word_decoder_round_trips_ids_to_words(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    WordVocabulary([WORD_UNK, "a", "b"]).save(
        vocabulary_path(WORD_DATASET, tag="word", cache_dir=tmp_path))
    decode = get_tiktoken_decoder(WORD_DATASET)
    assert decode is not None
    assert decode([1, 2, 0]) == "a b <unk>"


def test_word_decoder_is_none_when_the_vocabulary_is_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    assert get_tiktoken_decoder(WORD_DATASET) is None


def test_vocabulary_lookup_is_memoized_and_absent_is_not_an_error(tmp_path: Path,
                                                                  monkeypatch) -> None:
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    assert _load_word_vocabulary(WORD_DATASET, "word", cache_dir=tmp_path) is None
    WordVocabulary([WORD_UNK, "a"]).save(
        vocabulary_path(WORD_DATASET, tag="word", cache_dir=tmp_path))
    first = _load_word_vocabulary(WORD_DATASET, "word", cache_dir=tmp_path)
    second = _load_word_vocabulary(WORD_DATASET, "word", cache_dir=tmp_path)
    assert first is not None and first is second


# ---- recorded source characters ---------------------------------------------------------------

def test_recorded_source_chars_requires_a_matching_token_count(tmp_path: Path) -> None:
    """A sidecar describing a different stream is not evidence about this one."""
    target = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    target.write_bytes(b"")
    sidecar = Path(str(target) + ".meta.json")
    sidecar.write_text(json.dumps({"n_tokens": 10, "n_chars_source": 50}), encoding="utf-8")
    assert _recorded_source_chars("wikitext-103", "train", 10, cache_dir=tmp_path) == 50
    assert _recorded_source_chars("wikitext-103", "train", 11, cache_dir=tmp_path) is None


def test_recorded_source_chars_absent_or_malformed_is_none(tmp_path: Path) -> None:
    assert _recorded_source_chars("wikitext-103", "train", 10, cache_dir=tmp_path) is None
    target = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    target.write_bytes(b"")
    Path(str(target) + ".meta.json").write_text("{not json", encoding="utf-8")
    assert _recorded_source_chars("wikitext-103", "train", 10, cache_dir=tmp_path) is None


# ---- builder ---------------------------------------------------------------------------------

_FAKE_ROWS = {
    ("train", 0): [" the cat sat ", " the cat , the dog . "],
    ("validation", 0): [" the cat "],
    ("test", 0): [" dog . "],
}


def _fake_read_rows(file_name: str):
    """Serve synthetic rows keyed by (split, shard index) instead of hitting the hub."""
    split, index = file_name.split("|")
    return list(_FAKE_ROWS[(split, int(index))])


@pytest.fixture()
def fake_corpus(monkeypatch):
    monkeypatch.setattr(
        builder, "SPLIT_FILES",
        {"train": ("train|0",), "validation": ("validation|0",), "test": ("test|0",)})
    monkeypatch.setattr(builder, "_read_rows", _fake_read_rows)


def test_builder_writes_a_cache_the_loader_reads_back(tmp_path: Path,
                                                      monkeypatch, fake_corpus) -> None:
    """End to end over synthetic rows: ids, sidecars, and the vocabulary sidecar all agree."""
    summary = builder.build(min_count=1, cache_dir=tmp_path)

    # min_count=1 keeps only types seen more than once: 'the' (3) and 'cat' (2), plus <unk>.
    vocabulary = WordVocabulary.load(vocabulary_path(builder.DATASET, tag="word", cache_dir=tmp_path))
    assert vocabulary.words[0] == WORD_UNK
    assert vocabulary.vocab_size == 3
    assert set(vocabulary.words) == {WORD_UNK, "the", "cat"}

    tokens = torch.load(cache_path(builder.DATASET, "train", suffix="pt", cache_dir=tmp_path),
                        weights_only=True)
    assert tokens.dtype == torch.int64 and tokens.dim() == 1
    expected = vocabulary.encode(tokenize_row(" the cat sat ") + tokenize_row(" the cat , the dog . "))
    assert tokens.tolist() == expected
    assert summary["splits"]["train"]["n_tokens"] == len(expected)

    meta = json.loads(Path(str(
        cache_path(builder.DATASET, "train", suffix="pt", cache_dir=tmp_path)) + ".meta.json"
    ).read_text())
    assert meta["n_tokens"] == len(expected)
    assert meta["vocab_size"] == vocabulary.vocab_size
    assert meta["n_chars_source"] == len(" the cat sat ") + len(" the cat , the dog . ")
    assert meta["tokens_per_char"] == pytest.approx(len(expected) / meta["n_chars_source"])


def test_builder_refuses_to_overwrite_without_force(tmp_path: Path,
                                                    monkeypatch, fake_corpus) -> None:
    builder.build(min_count=1, cache_dir=tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        builder.build(min_count=1, cache_dir=tmp_path)
    builder.build(min_count=1, cache_dir=tmp_path, force=True)


def test_builder_reference_build_asserts_the_measured_vocabulary_size(
        tmp_path: Path, monkeypatch, fake_corpus) -> None:
    """At the default cutoff the build must reproduce the pinned revision's type count."""
    with pytest.raises(SystemExit, match="reference vocabulary mismatch"):
        builder.build(min_count=0, cache_dir=tmp_path)


def test_builder_meta_and_loader_vocab_bound_agree(tmp_path: Path,
                                                  monkeypatch, fake_corpus) -> None:
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    builder.build(min_count=1, cache_dir=tmp_path)
    assert datasets_mod.tokenizer_vocab_size(builder.DATASET) == 3


# ---- the published <eos> stream (wikitext-103-word-eos) ----------------------------------------

EOS_DATASET = "wikitext-103-word-eos"


def test_eos_rows_end_in_eos_and_blank_rows_count_once() -> None:
    """Every row, blank rows included, contributes exactly one <eos>; the default is unchanged."""
    assert tokenize_row(" the cat ", eos=True) == ["the", "cat", WORD_EOS]
    assert tokenize_row("", eos=True) == [WORD_EOS]
    assert tokenize_row(" \n", eos=True) == [WORD_EOS]
    assert tokenize_row(" the cat ") == ["the", "cat"]
    counts = count_tokens(["a b", "", " a "], eos=True)
    assert counts == Counter({"a": 2, "b": 1, WORD_EOS: 3})


def test_eos_vocabulary_fixes_unk_then_eos_then_frequency_order() -> None:
    """<eos> is the most frequent type here, yet it sits at id 1, never inside the frequency order."""
    counts = Counter({WORD_EOS: 50, "the": 9, "a": 9, WORD_UNK: 4, "rare": 1})
    words = vocabulary_from_counts(counts, eos=True)
    assert words == [WORD_UNK, WORD_EOS, "a", "the", "rare"]
    assert vocabulary_from_counts(counts, min_count=5, eos=True) == [WORD_UNK, WORD_EOS, "a", "the"]
    assert vocabulary_size_for_min_count(counts, eos=True) == 5
    assert vocabulary_size_for_min_count(counts, min_count=5, eos=True) == 4
    # Without eos the marker would be an ordinary counted type (the no-<eos> stream never has one).
    assert vocabulary_from_counts(Counter({"the": 2, WORD_UNK: 1}))[:2] == [WORD_UNK, "the"]


def test_eos_word_vocabulary_ids_encode_and_round_trip(tmp_path: Path) -> None:
    vocabulary = WordVocabulary([WORD_UNK, WORD_EOS, "the", "cat"], eos=True)
    assert vocabulary.eos_id == 1 and vocabulary.unk_id == 0
    assert vocabulary.encode_row(" the dog ") == [2, 0, 1]
    assert vocabulary.encode_row("") == [1]
    path = tmp_path / "vocab.json"
    vocabulary.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["eos"] is True
    reloaded = WordVocabulary.load(path)
    assert reloaded.eos is True and reloaded.words == vocabulary.words
    with pytest.raises(ValueError, match="id 1"):
        WordVocabulary([WORD_UNK, "the", WORD_EOS], eos=True)


def test_no_eos_vocabulary_file_layout_is_unchanged(tmp_path: Path) -> None:
    """A table without <eos> writes no 'eos' key, so the existing sidecar format is untouched."""
    path = tmp_path / "vocab.json"
    WordVocabulary([WORD_UNK, "a"]).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "tokenizer", "min_count", "vocab_size", "words"}
    assert WordVocabulary.load(path).eos is False and WordVocabulary.load(path).eos_id is None
    payload["eos"] = "yes"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="eos"):
        WordVocabulary.load(path)


def test_eos_dataset_is_a_word_dataset_with_the_published_reference() -> None:
    assert appends_eos(EOS_DATASET) and not appends_eos(WORD_DATASET)
    assert is_word_level(EOS_DATASET) and _tokenizer_tag(EOS_DATASET) == "word"
    assert tiktoken_encoding_name(EOS_DATASET) == "word"
    assert cache_path(EOS_DATASET, "test").name == "wikitext-103-word-eos_test_word_tokens.pt"
    assert reference_vocab_size(EOS_DATASET) == REFERENCE_VOCAB_SIZE_EOS == 267_735
    assert reference_vocab_size(WORD_DATASET) == REFERENCE_VOCAB_SIZE == 267_734
    assert REFERENCE_N_TOKENS_EOS == {"train": 103_227_021, "validation": 217_646, "test": 245_569}


def test_eos_vocab_size_falls_back_to_the_published_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: tmp_path)
    assert datasets_mod.tokenizer_vocab_size(EOS_DATASET) == 267_735
    assert datasets_mod.tokenizer_vocab_size(WORD_DATASET) == 267_734


_FAKE_EOS_ROWS = {
    ("train", 0): ["", " = A = ", " the cat sat ", "", " the cat , the dog . "],
    ("validation", 0): ["", " the cat "],
    ("test", 0): [" dog . ", ""],
}


def _fake_eos_rows(file_name: str):
    split, index = file_name.split("|")
    return list(_FAKE_EOS_ROWS[(split, int(index))])


@pytest.fixture()
def fake_eos_corpus(monkeypatch):
    monkeypatch.setattr(
        builder, "SPLIT_FILES",
        {"train": ("train|0",), "validation": ("validation|0",), "test": ("test|0",)})
    monkeypatch.setattr(builder, "_read_rows", _fake_eos_rows)


def _expected_ids(vocabulary: WordVocabulary, rows) -> list:
    return [i for row in rows for i in vocabulary.encode(tokenize_row(row, eos=True))]


def test_eos_builder_appends_eos_to_every_row_and_the_loader_reads_it(
        tmp_path: Path, monkeypatch, fake_eos_corpus) -> None:
    summary = builder.build(min_count=1, cache_dir=tmp_path, dataset=EOS_DATASET)
    vocabulary = WordVocabulary.load(vocabulary_path(EOS_DATASET, tag="word", cache_dir=tmp_path))
    assert vocabulary.eos and vocabulary.words[:2] == (WORD_UNK, WORD_EOS)
    # min_count=1 keeps types seen more than once: 'the' (3), 'cat' (2), '=' (2) and '<eos>'.
    assert set(vocabulary.words) == {WORD_UNK, WORD_EOS, "the", "cat", "="}
    for split in ("train", "validation", "test"):
        rows = _FAKE_EOS_ROWS[(split, 0)]
        tokens = torch.load(cache_path(EOS_DATASET, split, suffix="pt", cache_dir=tmp_path),
                            weights_only=True)
        assert tokens.tolist() == _expected_ids(vocabulary, rows)
        assert int((tokens == 1).sum()) == len(rows)                   # one <eos> per row
        n_words = sum(len(tokenize_row(row)) for row in rows)
        assert summary["splits"][split]["n_tokens"] == n_words + len(rows)
        meta = json.loads(Path(str(cache_path(EOS_DATASET, split, suffix="pt", cache_dir=tmp_path))
                               + ".meta.json").read_text())
        assert meta["eos"] is True and meta["n_tokens"] == n_words + len(rows)
    loader = datasets_mod.make_dataloader(EOS_DATASET, "test", 2, 1, shuffle=False, drop_last=False,
                                          cache_dir=tmp_path, vocab_size=vocabulary.vocab_size)
    assert loader.dataset.data_identity["tokenizer_vocab_size"] == vocabulary.vocab_size
    # The no-<eos> dataset built from the same rows is a separate cache without <eos> or an 'eos' key.
    builder.build(min_count=1, cache_dir=tmp_path, dataset=WORD_DATASET)
    plain = json.loads(Path(str(cache_path(WORD_DATASET, "train", suffix="pt", cache_dir=tmp_path))
                            + ".meta.json").read_text())
    assert "eos" not in plain
    assert WORD_EOS not in WordVocabulary.load(
        vocabulary_path(WORD_DATASET, tag="word", cache_dir=tmp_path)).words


def test_eos_reference_build_checks_vocabulary_and_every_split_before_writing(
        tmp_path: Path, monkeypatch, fake_eos_corpus) -> None:
    """At min_count=0 the <eos> build must reproduce the reference vocabulary AND token counts."""
    with pytest.raises(SystemExit, match="reference vocabulary mismatch"):
        builder.build(min_count=0, cache_dir=tmp_path, dataset=EOS_DATASET)
    counts = {split: sum(len(tokenize_row(row, eos=True)) for row in _FAKE_EOS_ROWS[(split, 0)])
              for split in ("train", "validation", "test")}
    n_types = len(count_tokens(_FAKE_EOS_ROWS[("train", 0)], eos=True))   # <eos> among them
    monkeypatch.setattr(builder, "reference_vocab_size", lambda dataset: n_types + 1)   # + <unk>
    monkeypatch.setattr(builder, "REFERENCE_N_TOKENS_EOS", {**counts, "test": counts["test"] + 1})
    with pytest.raises(SystemExit, match="token-count mismatch.*test="):
        builder.build(min_count=0, cache_dir=tmp_path, dataset=EOS_DATASET)
    assert not list(tmp_path.iterdir())                                 # refused before any write
    monkeypatch.setattr(builder, "REFERENCE_N_TOKENS_EOS", counts)
    summary = builder.build(min_count=0, cache_dir=tmp_path, dataset=EOS_DATASET)
    assert {split: entry["n_tokens"] for split, entry in summary["splits"].items()} == counts
    assert summary["vocab_size"] == n_types + 1


def test_builder_rejects_a_non_word_dataset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not a word dataset"):
        builder.build(cache_dir=tmp_path, dataset="wikitext-103")


def test_loader_rejects_an_eos_cache_read_against_a_table_without_eos(
        tmp_path: Path, monkeypatch, fake_eos_corpus) -> None:
    builder.build(min_count=1, cache_dir=tmp_path, dataset=EOS_DATASET)
    sidecar = vocabulary_path(EOS_DATASET, tag="word", cache_dir=tmp_path)
    table = WordVocabulary.load(sidecar)
    WordVocabulary([WORD_UNK, *table.words[2:], "extra"], min_count=1).save(sidecar)   # same size
    with pytest.raises(ValueError, match="eos"):
        datasets_mod.make_dataloader(EOS_DATASET, "test", 2, 1, shuffle=False, drop_last=False,
                                     cache_dir=tmp_path)
