r"""Canonical word-level tokenization and vocabulary for WikiText-103.

The word-level setting is the one the benchmark was introduced with (Merity et al., *Pointer
Sentinel Mixture Models*, 2016, arXiv:1609.07843). The corpus variant consumed here is the
pre-tokenized ``wikitext-103-v1`` text, whose rows already carry punctuation as separate
whitespace-delimited tokens (``" the European lobster , is "``). Tokenizing it is therefore a
whitespace split with no further normalization: no lowercasing, no number handling, no punctuation
stripping. Casing and punctuation are part of the vocabulary, as in the reference setup.

THE CORPUS CARRIES ITS OWN ``<unk>``. The release replaced rare words with a literal ``<unk>``
token, which is why the reference reports a 0.4% OOV rate: measured on the pinned revision, 0.4357%
of training tokens, 0.7747% of validation, and 1.0331% of test are that literal marker. It is
therefore not a synthetic id bolted onto the table -- it OCCUPIES id 0, and out-of-vocabulary words
encountered at evaluation map onto the same id. No other type is discarded by this module.

VOCABULARY SIZE (measured, not assumed). On the pinned revision the training split holds 267,734
distinct whitespace tokens *including* that ``<unk>`` marker, so the vocabulary is 267,734 ids and
every validation/test token is already in it (OOV exactly 0.0% beyond the corpus's own markers).
Merity et al. quote 267,735 for their training split; the pinned Hugging Face revision differs from
the original release by one type, and this module records the MEASURED value rather than the quoted
one so a preprocessing drift fails loudly. A ``count > 3`` cutoff -- often cited as the definition
of the vocabulary -- actually yields only 224,727 types here and introduces real OOV
(0.068% validation, 0.129% test), so it is not the reference setting; it remains available as a
size knob through ``min_count``.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The single out-of-vocabulary id. Fixed at 0 so every word id is a stable positive integer.
WORD_UNK = "<unk>"

#: Keep types with count STRICTLY GREATER than this. 0 keeps every observed type, which is the
#: reference word-level vocabulary (see the module docstring); raise it to trade OOV for a
#: smaller ``phi_embed`` table. The ``<unk>`` id is kept regardless of the cutoff.
DEFAULT_MIN_COUNT = 0

#: Vocabulary size the reference word-level build must reproduce on the pinned corpus revision:
#: the 267,734 distinct whitespace tokens of the training split, one of which is ``<unk>`` and
#: occupies id 0. (Merity et al. quote 267,735 for the original release; this revision differs by
#: one type. The measured value is what a build is held to, so drift cannot pass silently.)
REFERENCE_VOCAB_SIZE = 267_734

#: Vocabulary-file schema, so a future layout change is detectable rather than mis-parsed.
VOCAB_SCHEMA_VERSION = 1


def tokenize_row(text: str) -> List[str]:
    r"""Whitespace-split one ``wikitext-103-v1`` text row into word tokens.

    ``str.split()`` with no argument collapses runs of whitespace and drops leading/trailing
    whitespace, so a blank row yields no tokens and a section header (``" = Homarus gammarus = "``)
    yields the ``=`` markers as ordinary words -- the same treatment the reference tokenization
    gives them.
    """
    return text.split()


def count_tokens(rows: Iterable[str]) -> Counter:
    r"""Count word-type occurrences over an iterable of text rows.

    Streams row by row so the training split's ~103M tokens never materialize as a Python list.
    """
    counts: Counter = Counter()
    for row in rows:
        counts.update(tokenize_row(row))
    return counts


def vocabulary_from_counts(
    counts: Counter,
    *,
    min_count: int = DEFAULT_MIN_COUNT,
) -> List[str]:
    r"""Build the id-to-word list from training-split counts, ``<unk>`` first.

    The order is ``<unk>`` (id 0) followed by every OTHER type whose count exceeds ``min_count``,
    sorted by descending frequency and then lexicographically. Two details are load-bearing. The
    corpus's own ``<unk>`` marker is a counted type, so it is excluded from the frequency ordering
    and placed at id 0 instead of appearing twice; a frequency rank would still leave it inside the
    table, but the unk id has to be fixed because out-of-vocabulary words map onto it. And the
    lexicographic secondary key makes the ids a pure function of the corpus rather than of
    ``Counter`` iteration order, so two builds of the same revision agree byte for byte.
    """
    if min_count < 0:
        raise ValueError(f"min_count must be >= 0, got {min_count}")
    kept = ((word, n) for word, n in counts.items() if word != WORD_UNK and n > min_count)
    ordered = sorted(kept, key=lambda item: (-item[1], item[0]))
    return [WORD_UNK, *(word for word, _ in ordered)]


def vocabulary_size_for_min_count(counts: Counter, *, min_count: int = DEFAULT_MIN_COUNT) -> int:
    r"""Number of ids :func:`vocabulary_from_counts` would produce, without materializing the list."""
    return 1 + sum(1 for word, n in counts.items() if word != WORD_UNK and n > min_count)


class WordVocabulary:
    r"""An id-to-word table with encode/decode, loadable from the cache sidecar.

    ``encode`` maps an unseen word to :data:`WORD_UNK` rather than raising: the 0.4% OOV rate is a
    property of the benchmark, not an error condition. ``decode`` renders ``<unk>`` back as the
    literal marker, so a decoded stream is a faithful view of the ids (it is not, and cannot be, a
    reconstruction of the original words).
    """

    def __init__(self, words: Sequence[str], *, min_count: int = DEFAULT_MIN_COUNT) -> None:
        if not words or words[0] != WORD_UNK:
            raise ValueError(f"vocabulary must start with {WORD_UNK!r}, got {words[:1]!r}")
        if len(set(words)) != len(words):
            raise ValueError("vocabulary contains duplicate words")
        self.words: Tuple[str, ...] = tuple(words)
        self.min_count = int(min_count)
        self._ids: Dict[str, int] = {word: index for index, word in enumerate(self.words)}

    @property
    def vocab_size(self) -> int:
        """Number of ids, ``<unk>`` included."""
        return len(self.words)

    @property
    def unk_id(self) -> int:
        return self._ids[WORD_UNK]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        r"""Map word tokens to ids, sending every word outside the vocabulary to ``<unk>``."""
        lookup = self._ids
        return [lookup.get(word, 0) for word in tokens]

    def encode_row(self, text: str) -> List[int]:
        r"""Tokenize and encode one raw text row."""
        return self.encode(tokenize_row(text))

    def decode(self, ids: Sequence[int]) -> str:
        r"""Render ids as space-joined words, ``<unk>`` included as its literal marker."""
        return " ".join(self.words[int(i)] for i in ids)

    def row_token_counts(self, rows: Iterable[str]) -> Tuple[int, int]:
        r"""Token and OOV counts over ``rows`` under this vocabulary, for build-time reporting."""
        n_tokens = 0
        n_oov = 0
        for row in rows:
            tokens = tokenize_row(row)
            n_tokens += len(tokens)
            n_oov += sum(1 for word in tokens if word not in self._ids)
        return n_tokens, n_oov

    def save(self, path: Path) -> None:
        """Write the vocabulary as JSON, with the schema and cutoff it was built at."""
        payload = {
            "schema_version": VOCAB_SCHEMA_VERSION,
            "tokenizer": "word",
            "min_count": self.min_count,
            "vocab_size": self.vocab_size,
            "words": list(self.words),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WordVocabulary":
        """Read a vocabulary written by :meth:`save`, failing closed on a layout mismatch."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"word vocabulary {path} must be a JSON object")
        version = payload.get("schema_version")
        if version != VOCAB_SCHEMA_VERSION:
            raise ValueError(
                f"word vocabulary {path} has schema_version={version!r}, "
                f"expected {VOCAB_SCHEMA_VERSION}")
        words = payload.get("words")
        if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
            raise ValueError(f"word vocabulary {path} must carry a list of strings under 'words'")
        if payload.get("vocab_size") != len(words):
            raise ValueError(
                f"word vocabulary {path} declares vocab_size={payload.get('vocab_size')!r} "
                f"but lists {len(words)} words")
        return cls(words, min_count=int(payload.get("min_count", DEFAULT_MIN_COUNT)))


def vocabulary_path(dataset: str, *, tag: str, cache_dir: Optional[Path] = None) -> Path:
    r"""Path of the vocabulary sidecar for ``dataset``'s word cache.

    Lives in the cache directory (not the per-run artifacts) because the cache's ids are
    meaningless without it, and both the decoder and the bits-per-character normalizer read it.
    """
    from vfe3.data.datasets import default_cache_dir  # local import avoids an import cycle

    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return root / f"{dataset}_{tag}_vocab.json"
