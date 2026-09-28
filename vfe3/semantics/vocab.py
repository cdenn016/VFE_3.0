r"""Analysis vocabularies for :mod:`vfe3.semantics`.

A :class:`Vocab` maps model token ids to display strings and resolves English words to token ids
under a strict single-token policy: for a BPE tokenizer (GPT-2, cl100k) a word ``w`` resolves only
when ``" " + w``, exactly as written, encodes to ONE token -- no case fallback and no bare-form
fallback, because in a pre-tokenized corpus the bare form is a word-internal piece (``"king"``
occurs 1,176 times in wikitext-103 training against 11,107 for ``" king"``) and a case change can
change the meaning (``" Turkey"`` vs ``" turkey"``). A word-level vocabulary resolves by exact,
case-sensitive lookup, with ``<unk>`` unresolved.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Token classes returned by :meth:`Vocab.token_classes`, in code order.
TOKEN_CLASS_NAMES: Tuple[str, ...] = (
    "punctuation", "number", "function word", "content word", "capitalized word", "subword", "other",
)
_CLASS_CODE: Dict[str, int] = {name: code for code, name in enumerate(TOKEN_CLASS_NAMES)}

# English function words (copied from vfe3/viz/figures.py ``_STOPWORDS``, including its BPE
# contraction fragments). Matched on the lower-cased word.
FUNCTION_WORDS = frozenset((
    "a an the this that these those some any all each every no none "
    "i you he she it we they me him her us them my your his its our their mine yours hers ours theirs "
    "is are was were be been being am do does did doing have has had having "
    "will would shall should can could may might must ought "
    "of in on at by for with about against between into through during before after "
    "above below to from up down out off over under again further then once here there "
    "and or but nor so yet because as if while although though unless until than "
    "not only own same too very just more most other such "
    "what which who whom whose where when why how "
    "s t re ve ll d m"
).split())

# Datasets whose text is not English: English word resources and token classes do not apply.
NON_ENGLISH_DATASETS = frozenset(("wiki-ja", "wiki-ar"))

_BPE_KINDS = ("gpt2", "cl100k_base")
_WORD_UNK = "<unk>"
_DISPLAY_MAX_CHARS = 16


def _is_number(core: str) -> bool:
    r"""Whether ``core`` is a digit string, allowing ``,`` and ``.`` separators (``1,000``, ``3.5``)."""
    digits = core.replace(",", "").replace(".", "")
    return bool(digits) and digits.isdigit()


class Vocab:
    r"""Per-id strings of a model vocabulary plus the word-resolution policy of its tokenizer.

    ``kind`` is ``"gpt2"`` or ``"cl100k_base"`` (BPE), ``"word"`` (word-level cache vocabulary) or
    ``"custom"`` (an explicit string list, resolved like a word vocabulary). ``strings[t]`` is the
    decoded text of id ``t`` (BPE: ``decode_single_token_bytes`` decoded as UTF-8 with replacement);
    ``valid_text[t]`` is False when that byte string is not valid UTF-8 on its own or the id has no
    bytes (BPE), or when the id is ``<unk>`` (word / custom). ``english`` is False for datasets whose
    text is not English (wiki-ja, wiki-ar), for which the English token classes are not meaningful.
    """

    def __init__(
        self,
        kind:       str,
        strings:    Sequence[str],
        valid_text: np.ndarray,

        *,
        english:    bool = True,
        encoding:   Optional[object] = None,          # tiktoken.Encoding for BPE kinds
    ) -> None:
        if kind not in (*_BPE_KINDS, "word", "custom"):
            raise ValueError(f"unknown vocabulary kind {kind!r}")
        if kind in _BPE_KINDS and encoding is None:
            raise ValueError(f"a {kind!r} vocabulary needs its tiktoken encoding")
        self.kind: str = kind
        self.strings: List[str] = list(strings)
        self.size: int = len(self.strings)
        self.valid_text: np.ndarray = np.asarray(valid_text, dtype=bool).reshape(-1)
        if self.valid_text.shape != (self.size,):
            raise ValueError(f"valid_text must have shape ({self.size},), got {self.valid_text.shape}")
        self.english: bool = bool(english)
        self._encoding = encoding
        self._lookup: Dict[str, int] = {}
        if kind not in _BPE_KINDS:
            for token_id, text in enumerate(self.strings):
                self._lookup.setdefault(text, token_id)
        self._word_mask: Optional[np.ndarray] = None
        self._classes: Optional[np.ndarray] = None

    @property
    def is_bpe(self) -> bool:
        return self.kind in _BPE_KINDS

    def __len__(self) -> int:
        return self.size

    def word_id(self, word: str) -> Optional[int]:
        r"""Token id of ``word`` under the single-token policy, else ``None``.

        BPE: ``" " + word`` must encode (as ordinary text, no special tokens) to exactly one token.
        Word / custom: exact lookup; ``<unk>`` and absent words give ``None``.
        """
        if not word or word != word.strip():
            return None
        if self.is_bpe:
            ids = self._encoding.encode_ordinary(" " + word)
            return int(ids[0]) if len(ids) == 1 else None
        token_id = self._lookup.get(word)
        if token_id is None or not self.valid_text[token_id]:
            return None
        return int(token_id)

    def word_mask(self) -> np.ndarray:
        r"""(V,) bool: whole-word alphabetic tokens.

        BPE: the string starts with a space and the rest is non-empty and alphabetic (ASCII or
        Unicode letters). Word / custom: the string is alphabetic. Invalid-text ids are excluded.
        """
        if self._word_mask is None:
            if self.is_bpe:
                mask = [len(s) > 1 and s[0] == " " and s[1:].isalpha() for s in self.strings]
            else:
                mask = [s.isalpha() for s in self.strings]
            self._word_mask = np.asarray(mask, dtype=bool) & self.valid_text
        return self._word_mask

    def _classify(self, text: str, valid: bool) -> str:
        if not valid or not text or text.isspace():
            return "other"
        core = text.strip()
        if _is_number(core):
            return "number"
        if not any(ch.isalnum() for ch in core):
            return "punctuation"
        if self.is_bpe:
            if text[0] == " " and text[1:].isalpha():
                word = text[1:]
            elif text[0] != " " and text.isalpha():
                return "subword"                            # continuation piece of a word
            else:
                return "other"
        else:
            if not text.isalpha():
                return "other"
            word = text
        if word.lower() in FUNCTION_WORDS:
            return "function word"
        if word[0].isupper():
            return "capitalized word"
        return "content word"

    def token_classes(self) -> Tuple[np.ndarray, List[str]]:
        r"""(V,) int class codes into :data:`TOKEN_CLASS_NAMES`, and the class names.

        Classes are read from each token's STRING: punctuation (no letters or digits), number,
        function word (lower-cased word in :data:`FUNCTION_WORDS`), capitalized word, content word,
        subword (BPE continuation piece: letters without a leading space), other. When
        ``english`` is False every id is ``"other"`` (the English classes are not meaningful).
        """
        if self._classes is None:
            if not self.english:
                codes = np.full(self.size, _CLASS_CODE["other"], dtype=np.int64)
            else:
                codes = np.fromiter(
                    (_CLASS_CODE[self._classify(s, bool(v))]
                     for s, v in zip(self.strings, self.valid_text)),
                    dtype=np.int64, count=self.size)
            self._classes = codes
        return self._classes, list(TOKEN_CLASS_NAMES)

    def display(self, token_id: int) -> str:
        r"""Short label for figures and tables.

        Whole words lose their leading space, BPE continuation subwords get a leading ``·``,
        ``\n``/``\t`` are escaped, labels are cut to 16 characters, and invalid text is ``<id>``.
        """
        token_id = int(token_id)
        if not 0 <= token_id < self.size:
            raise IndexError(f"token id {token_id} outside vocabulary of size {self.size}")
        if not self.valid_text[token_id]:
            return f"<{token_id}>"
        text = self.strings[token_id]
        if self.is_bpe:
            if len(text) > 1 and text[0] == " " and not text[1:].isspace():
                text = text[1:]
            elif text and text[0].isalnum():
                text = "·" + text
        text = text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        if not text.strip():
            text = "␣" * max(1, len(text))
        if len(text) > _DISPLAY_MAX_CHARS:
            text = text[:_DISPLAY_MAX_CHARS - 1] + "…"
        return text


def _bpe_vocab(encoding_name: str, *, english: bool) -> Vocab:
    r"""Build a BPE :class:`Vocab` from the named tiktoken encoding (ids ``0 .. n_vocab - 1``)."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    strings: List[str] = []
    valid = np.zeros(encoding.n_vocab, dtype=bool)
    for token_id in range(encoding.n_vocab):
        try:
            raw = encoding.decode_single_token_bytes(token_id)
        except KeyError:                                    # an id with no bytes (cl100k gaps)
            strings.append("")
            continue
        try:
            strings.append(raw.decode("utf-8"))
            valid[token_id] = True
        except UnicodeDecodeError:                          # a byte fragment of a multi-byte char
            strings.append(raw.decode("utf-8", errors="replace"))
    return Vocab(encoding_name, strings, valid, english=english, encoding=encoding)


def vocab_for_run(
    dataset:            Optional[str],
    selection_identity: Optional[dict],
    vocab_size:         int,
) -> Vocab:
    r"""The analysis vocabulary of a run.

    The tokenizer is read, in priority order, from ``selection_identity["tokenizer_encoding"]``
    (the checkpoint's data identity) and from the dataset name via
    :func:`vfe3.data.datasets.tiktoken_encoding_name`. A word-level vocabulary is loaded from the
    cache sidecar ``vocabulary_path(dataset, tag="word")``. Raises when the tokenizer cannot be
    determined or when the vocabulary size differs from the model's ``vocab_size``.
    """
    identity = selection_identity or {}
    dataset = dataset or identity.get("dataset")
    encoding_name = identity.get("tokenizer_encoding")
    if encoding_name is None:
        if dataset is None:
            raise ValueError("cannot determine the tokenizer: no selection identity and no dataset")
        from vfe3.data.datasets import tiktoken_encoding_name

        encoding_name = tiktoken_encoding_name(dataset)
    english = (dataset not in NON_ENGLISH_DATASETS) if dataset is not None else encoding_name == "gpt2"
    if encoding_name in _BPE_KINDS:
        vocab = _bpe_vocab(encoding_name, english=english)
    elif encoding_name == "word":
        if dataset is None:
            raise ValueError("a word-level vocabulary needs the dataset name to locate its sidecar")
        from vfe3.data.word_vocab import WordVocabulary, vocabulary_path

        table = WordVocabulary.load(vocabulary_path(dataset, tag="word"))
        words = list(table.words)
        valid = np.asarray([w != _WORD_UNK for w in words], dtype=bool)
        vocab = Vocab("word", words, valid, english=english)
    else:
        raise ValueError(f"unsupported tokenizer encoding {encoding_name!r}")
    if vocab.size != int(vocab_size):
        raise ValueError(
            f"{encoding_name!r} vocabulary has {vocab.size} ids but the model table has "
            f"{int(vocab_size)} rows; the run's tokenizer is not the one inferred")
    return vocab


def custom_vocab(strings: Sequence[str], *, english: bool = True) -> Vocab:
    r"""A ``"custom"`` vocabulary over explicit per-id strings (tests, non-standard runs).

    Words resolve by exact lookup (first occurrence wins); ``"<unk>"`` is invalid text.
    """
    strings = [str(s) for s in strings]
    valid = np.asarray([s != _WORD_UNK for s in strings], dtype=bool)
    return Vocab("custom", strings, valid, english=english)
