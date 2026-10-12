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
A ``count > 3`` cutoff -- often cited as the definition of the vocabulary -- actually yields only
224,727 types here and introduces real OOV (0.068% validation, 0.129% test), so it is not the
reference setting; it remains available as a size knob through ``min_count``.

THE PUBLISHED FIGURES COUNT ONE ``<eos>`` PER LINE. Merity et al. quote a vocabulary of 267,735 and
103,227,021 / 217,646 / 245,569 train / validation / test tokens. The pinned revision is not a
different text: its rows are the lines of the release, and appending one ``<eos>`` to every row,
blank rows included, gives exactly those numbers (1,801,350 / 3,760 / 4,358 rows added to the
whitespace counts 101,425,671 / 213,886 / 241,211, and one more type). Two datasets therefore exist.
``wikitext-103-word`` is the whitespace stream without ``<eos>`` (vocabulary 267,734); its cache and
the runs trained on it are unchanged. ``wikitext-103-word-eos`` is the published stream: every row
ends in ``<eos>``, ``<unk>`` keeps id 0, ``<eos>`` is fixed at id 1, and the other types follow in
frequency order, so its every-target evaluation is comparable with published WikiText-103 results.
"""

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The single out-of-vocabulary id. Fixed at 0 so every word id is a stable positive integer.
WORD_UNK = "<unk>"

#: End-of-line marker of the published stream, appended to every row (blank rows included) by the
#: ``<eos>`` datasets and fixed at id 1 in their vocabulary.
WORD_EOS = "<eos>"

#: The whitespace stream without ``<eos>`` and the published stream with one ``<eos>`` per row.
WORD_DATASET = "wikitext-103-word"
WORD_EOS_DATASET = "wikitext-103-word-eos"

#: Word datasets whose rows end in :data:`WORD_EOS`.
EOS_WORD_DATASETS = frozenset({WORD_EOS_DATASET})

#: Keep types with count STRICTLY GREATER than this. 0 keeps every observed type, which is the
#: reference word-level vocabulary (see the module docstring); raise it to trade OOV for a
#: smaller ``phi_embed`` table. The ``<unk>`` (and ``<eos>``) ids are kept regardless of the cutoff.
DEFAULT_MIN_COUNT = 0

#: Vocabulary size the reference ``wikitext-103-word`` build must reproduce on the pinned corpus
#: revision: the 267,734 distinct whitespace tokens of the training split, one of which is ``<unk>``
#: and occupies id 0. The published 267,735 is this plus ``<eos>`` (:data:`REFERENCE_VOCAB_SIZE_EOS`).
REFERENCE_VOCAB_SIZE = 267_734

#: Vocabulary size of the reference ``wikitext-103-word-eos`` build: 267,734 types plus ``<eos>``,
#: the figure Merity et al. publish.
REFERENCE_VOCAB_SIZE_EOS = 267_735

#: Token counts of the reference ``wikitext-103-word-eos`` build, the published WikiText-103 sizes:
#: the whitespace tokens of each split plus one ``<eos>`` per row.
REFERENCE_N_TOKENS_EOS: Dict[str, int] = {
    "train": 103_227_021,
    "validation": 217_646,
    "test": 245_569,
}

#: Vocabulary-file schema, so a future layout change is detectable rather than mis-parsed.
VOCAB_SCHEMA_VERSION = 1


def appends_eos(dataset: str) -> bool:
    r"""Whether the word dataset ``dataset`` ends every row in :data:`WORD_EOS`."""
    return dataset in EOS_WORD_DATASETS


def reference_vocab_size(dataset: str) -> int:
    r"""Vocabulary size a reference (``min_count = 0``) build of the word dataset must reproduce."""
    return REFERENCE_VOCAB_SIZE_EOS if appends_eos(dataset) else REFERENCE_VOCAB_SIZE


def _reserved_words(eos: bool) -> Tuple[str, ...]:
    r"""Ids fixed ahead of the frequency order: ``<unk>`` (id 0) and, with ``eos``, ``<eos>`` (id 1)."""
    return (WORD_UNK, WORD_EOS) if eos else (WORD_UNK,)


def tokenize_row(text: str, *, eos: bool = False) -> List[str]:
    r"""Whitespace-split one ``wikitext-103-v1`` text row into word tokens.

    ``str.split()`` with no argument collapses runs of whitespace and drops leading/trailing
    whitespace, so a blank row yields no tokens and a section header (``" = Homarus gammarus = "``)
    yields the ``=`` markers as ordinary words -- the same treatment the reference tokenization
    gives them. With ``eos`` the row's tokens are followed by :data:`WORD_EOS`, so a blank row
    yields exactly ``["<eos>"]``: one end-of-line per line of the release.
    """
    tokens = text.split()
    if eos:
        tokens.append(WORD_EOS)
    return tokens


def count_tokens(rows: Iterable[str], *, eos: bool = False) -> Counter:
    r"""Count word-type occurrences over an iterable of text rows (with ``eos``: one ``<eos>`` each).

    Streams row by row so the training split's ~103M tokens never materialize as a Python list.
    """
    counts: Counter = Counter()
    for row in rows:
        counts.update(tokenize_row(row, eos=eos))
    return counts


def vocabulary_from_counts(
    counts: Counter,
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    eos: bool = False,
) -> List[str]:
    r"""Build the id-to-word list from training-split counts, ``<unk>`` first.

    The order is ``<unk>`` (id 0), then with ``eos`` the end-of-line marker ``<eos>`` (id 1),
    followed by every OTHER type whose count exceeds ``min_count``, sorted by descending frequency
    and then lexicographically. Two details are load-bearing. The corpus's own ``<unk>`` marker is a
    counted type, so it is excluded from the frequency ordering and placed at id 0 instead of
    appearing twice; a frequency rank would still leave it inside the table, but the unk id has to
    be fixed because out-of-vocabulary words map onto it (``<eos>`` is handled the same way, so its
    id does not depend on the line count). And the lexicographic secondary key makes the ids a pure
    function of the corpus rather than of ``Counter`` iteration order, so two builds of the same
    revision agree byte for byte.
    """
    if min_count < 0:
        raise ValueError(f"min_count must be >= 0, got {min_count}")
    reserved = _reserved_words(eos)
    kept = ((word, n) for word, n in counts.items() if word not in reserved and n > min_count)
    ordered = sorted(kept, key=lambda item: (-item[1], item[0]))
    return [*reserved, *(word for word, _ in ordered)]


def vocabulary_size_for_min_count(
    counts: Counter,
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    eos: bool = False,
) -> int:
    r"""Number of ids :func:`vocabulary_from_counts` would produce, without materializing the list."""
    reserved = _reserved_words(eos)
    return len(reserved) + sum(1 for word, n in counts.items() if word not in reserved and n > min_count)


class WordVocabulary:
    r"""An id-to-word table with encode/decode, loadable from the cache sidecar.

    ``encode`` maps an unseen word to :data:`WORD_UNK` rather than raising: the 0.4% OOV rate is a
    property of the benchmark, not an error condition. ``decode`` renders ``<unk>`` back as the
    literal marker, so a decoded stream is a faithful view of the ids (it is not, and cannot be, a
    reconstruction of the original words). With ``eos`` the table is that of an ``<eos>`` dataset:
    ``<eos>`` must sit at id 1 and :meth:`encode_row` ends every row in it.
    """

    def __init__(
        self,
        words: Sequence[str],
        *,
        min_count: int = DEFAULT_MIN_COUNT,
        eos: bool = False,
    ) -> None:
        if not words or words[0] != WORD_UNK:
            raise ValueError(f"vocabulary must start with {WORD_UNK!r}, got {words[:1]!r}")
        if type(eos) is not bool:
            raise TypeError(f"eos must be a bool, got {eos!r}")
        if eos and (len(words) < 2 or words[1] != WORD_EOS):
            raise ValueError(f"an <eos> vocabulary must hold {WORD_EOS!r} at id 1, got {words[1:2]!r}")
        if len(set(words)) != len(words):
            raise ValueError("vocabulary contains duplicate words")
        self.words: Tuple[str, ...] = tuple(words)
        self.min_count = int(min_count)
        self.eos = eos
        self._ids: Dict[str, int] = {word: index for index, word in enumerate(self.words)}

    @property
    def vocab_size(self) -> int:
        """Number of ids, ``<unk>`` (and ``<eos>``) included."""
        return len(self.words)

    @property
    def unk_id(self) -> int:
        return self._ids[WORD_UNK]

    @property
    def eos_id(self) -> Optional[int]:
        """Id of ``<eos>`` (1) for an ``<eos>`` vocabulary, else None."""
        return self._ids[WORD_EOS] if self.eos else None

    def encode(self, tokens: Sequence[str]) -> List[int]:
        r"""Map word tokens to ids, sending every word outside the vocabulary to ``<unk>``."""
        lookup = self._ids
        return [lookup.get(word, 0) for word in tokens]

    def encode_row(self, text: str) -> List[int]:
        r"""Tokenize and encode one raw text row (ending in ``<eos>`` for an ``<eos>`` vocabulary)."""
        return self.encode(tokenize_row(text, eos=self.eos))

    def decode(self, ids: Sequence[int]) -> str:
        r"""Render ids as space-joined words, ``<unk>`` included as its literal marker."""
        return " ".join(self.words[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        """Write the vocabulary as JSON, with the schema and cutoff it was built at.

        The write is atomic: the payload goes to a sibling temporary file that ``os.replace``
        publishes, so a reader never observes a partially written table and a failed write leaves
        any previous sidecar untouched. An ``<eos>`` vocabulary records ``"eos": true``; a table
        without it keeps the original layout byte for byte.
        """
        payload = {
            "schema_version": VOCAB_SCHEMA_VERSION,
            "tokenizer": "word",
            "min_count": self.min_count,
            "vocab_size": self.vocab_size,
            **({"eos": True} if self.eos else {}),
            "words": list(self.words),
        }
        target = Path(path)
        handle, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

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
        eos = payload.get("eos", False)
        if type(eos) is not bool:
            raise ValueError(f"word vocabulary {path} has a non-boolean 'eos' entry {eos!r}")
        return cls(words, min_count=int(payload.get("min_count", DEFAULT_MIN_COUNT)), eos=eos)


def vocabulary_path(dataset: str, *, tag: str, cache_dir: Optional[Path] = None) -> Path:
    r"""Path of the vocabulary sidecar for ``dataset``'s word cache.

    Lives in the cache directory (not the per-run artifacts) because the cache's ids are
    meaningless without it, and both the decoder and the bits-per-character normalizer read it.
    """
    from vfe3.data.datasets import default_cache_dir  # local import avoids an import cycle

    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return root / f"{dataset}_{tag}_vocab.json"
