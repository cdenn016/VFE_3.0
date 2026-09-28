r"""Word-normalized test perplexity after a BPE/word text audit (``tab:placeholder_calibration``).

The word-normalized perplexity of a subword model is

.. math::
    \mathrm{PPL}_{\mathrm{word}} = \exp\Bigl(\frac{1}{W}\sum_{t} \ell_t\Bigr),

the summed test negative log-likelihood over every scored BPE target divided by the number :math:`W` of
word-level test tokens. It is meaningful only when (a) the BPE test stream and the word-level test
stream encode the same text, and (b) every BPE target of the stream is scored exactly once.

(a) :func:`text_audit` decodes the BPE stream to text (the dataset's byte decoder) and compares it with
the word-level text after one normalization applied to both: all whitespace removed (line layout and
the spacing of detokenized punctuation carry no words), the WikiText markers ``@-@``, ``@,@``, ``@.@``
read as ``-``, ``,``, ``.``, and each ``<unk>`` of the word-level text matched as a wildcard for one
original word of at most :data:`MAX_UNK_CHARS` characters (the word-level release replaced rare words
by ``<unk>``; a BPE stream built from the raw variant carries the words). The result is the fraction of
word-level characters matched in order, the fraction of BPE characters accounted for, and the verdict
``same_text`` (both exactly one).

(b) :func:`target_coverage` checks that the scored target positions are exactly :math:`1, \dots, T-1`
for a :math:`T`-token stream (the first token is never a target).

:math:`W` depends on a counting convention, both reported (:func:`word_counts`):
``words_plus_eos_per_line`` counts the whitespace-delimited words (each ``<unk>`` one word, so the count
is the same for the raw and ``<unk>`` variants) plus one ``<eos>`` per line of the split, blank lines
included, which reproduces the published WikiText-103 test size (245,569 = 241,211 words + 4,358 lines)
and is the headline; ``words_only`` counts the words without ``<eos>`` (241,211, the vfe3
``wikitext-103-word`` cache). The word-level text is the pinned ``Salesforce/wikitext``
``wikitext-103-v1`` parquet from the local Hugging Face cache (the revision the word cache was built
from), or a file supplied as ``word_stream_path`` (parquet with a ``text`` column, or a text file with
one row per line such as ``wiki.test.tokens``). WikiText-2 shares WikiText-103's test text, which the
audit checks rather than assumes.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

WORD_DATASET_ID: str = "Salesforce/wikitext"
WORD_DATASET_CONFIG: str = "wikitext-103-v1"
WORD_DATASET_REVISION: str = "6231e49f19a707241d6b84d9cff60a3a86b85a85"     # build_wikitext103_word_cache.py
WORD_SPLIT_FILES: Dict[str, str] = {
    "test": f"{WORD_DATASET_CONFIG}/test-00000-of-00001.parquet",
    "validation": f"{WORD_DATASET_CONFIG}/validation-00000-of-00001.parquet",
}
WIKITEXT_PREFIXES: Tuple[str, ...] = ("wikitext-103", "wikitext-2")
MARKERS: Tuple[Tuple[str, str], ...] = (("@-@", "-"), ("@,@", ","), ("@.@", "."))
UNK: str = "<unk>"
MAX_UNK_CHARS: int = 64                   # longest original word a single <unk> may stand for
RESYNC_PIECE_CHARS: int = 32              # piece length of the in-order match after a failed segment
MISMATCH_EXAMPLES: int = 5
HEADLINE_CONVENTION: str = "words_plus_eos_per_line"

_WHITESPACE = re.compile(r"\s+")


# ----------------------------------------------------------------------------------------------
# The word-level stream
# ----------------------------------------------------------------------------------------------

@dataclass
class WordStream:
    r"""The rows (lines) of a word-level split, in stored order."""

    rows:   List[str] = field(repr=False)
    source: str = ""
    split:  str = "test"

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_blank_rows(self) -> int:
        return sum(1 for row in self.rows if not row.strip())

    @property
    def n_words(self) -> int:
        return sum(len(row.split()) for row in self.rows)

    @property
    def text(self) -> str:
        return "".join(self.rows)


def _read_rows(path: Path) -> List[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [str(row) for row in pq.read_table(path, columns=["text"]).column("text").to_pylist()]
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def load_word_stream(
    split: str = "test",

    *,
    path: "str | Path | None" = None,
) -> Tuple[Optional[WordStream], Optional[str]]:
    r"""``(stream, None)`` or ``(None, reason)``: the supplied file, else the pinned parquet, offline."""
    if path is not None:
        source = Path(path)
        if not source.is_file():
            return None, f"word stream unavailable: {source} does not exist"
        return WordStream(_read_rows(source), str(source), split), None
    name = WORD_SPLIT_FILES.get(split)
    if name is None:
        return None, f"word stream unavailable: no pinned word-level file for split {split!r}"
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None, "word stream unavailable: huggingface_hub is not installed and no word_stream_path was given"
    cached = try_to_load_from_cache(WORD_DATASET_ID, name, revision=WORD_DATASET_REVISION, repo_type="dataset")
    if not isinstance(cached, str) or not Path(cached).is_file():
        return None, (f"word stream unavailable: {WORD_DATASET_ID}@{WORD_DATASET_REVISION[:7]}/{name} is not in "
                      "the local Hugging Face cache (supply word_stream_path)")
    return WordStream(_read_rows(Path(cached)), f"{WORD_DATASET_ID}@{WORD_DATASET_REVISION}/{name}", split), None


def word_counts(stream: WordStream) -> Dict[str, Dict[str, Any]]:
    r"""The word-level token count under each convention, with its rule."""
    n_words, n_rows = stream.n_words, stream.n_rows
    return {
        "words_plus_eos_per_line": {
            "count": n_words + n_rows,
            "rule": (f"{n_words:,} whitespace-delimited words (each <unk> one word) plus one <eos> per line, "
                     f"{n_rows:,} lines including {stream.n_blank_rows:,} blank: the published WikiText-103 "
                     "test size and the word-level language-model convention")},
        "words_only": {
            "count": n_words,
            "rule": f"{n_words:,} whitespace-delimited words, no <eos> (the vfe3 wikitext-103-word cache)"},
    }


# ----------------------------------------------------------------------------------------------
# Text audit
# ----------------------------------------------------------------------------------------------

def normalize_for_audit(text: str) -> str:
    r"""Remove all whitespace and read the WikiText ``@x@`` markers as their characters."""
    text = _WHITESPACE.sub("", text)
    for marker, char in MARKERS:
        text = text.replace(marker, char)
    return text


def _match_pieces(segment: str, text: str, cursor: int, limit: int) -> Tuple[int, int]:
    r"""In-order match of ``segment``'s pieces in ``text[cursor:limit]``; ``(matched chars, new cursor)``."""
    matched = 0
    for start in range(0, len(segment), RESYNC_PIECE_CHARS):
        piece = segment[start:start + RESYNC_PIECE_CHARS]
        found = text.find(piece, cursor, limit)
        if found >= 0:
            matched += len(piece)
            cursor = found + len(piece)
    return matched, cursor


def text_audit(
    bpe_text:  str,
    word_text: str,

    *,
    max_unk_chars: int = MAX_UNK_CHARS,
) -> Dict[str, Any]:
    r"""Whether the two texts are the same under the audit normalization (module docstring).

    The normalized word text is split at its ``<unk>`` markers into literal segments; each segment is
    located in order in the normalized BPE text, starting between one and ``max_unk_chars`` characters
    per preceding wildcard after the previous match. A segment that is not found there is matched
    piecewise (:data:`RESYNC_PIECE_CHARS`) to count its characters and resynchronize, and recorded as a
    mismatch example.
    """
    B = normalize_for_audit(bpe_text)
    W = normalize_for_audit(word_text)
    segments = W.split(UNK)
    n_unk = len(segments) - 1
    word_chars = len(W) - len(UNK) * n_unk
    matched = wildcard_chars = cursor = pending = mismatched_segments = 0
    examples: List[Dict[str, Any]] = []
    for k, segment in enumerate(segments):
        if k > 0:
            pending += 1
        if not segment:
            continue
        lo = cursor + pending
        hi = cursor + pending * max_unk_chars
        found = B.find(segment, lo, hi + len(segment)) if pending else (cursor if B.startswith(segment, cursor) else -1)
        if found >= 0:
            wildcard_chars += found - cursor
            matched += len(segment)
            cursor = found + len(segment)
        else:
            mismatched_segments += 1
            if len(examples) < MISMATCH_EXAMPLES:
                examples.append({"segment": k, "bpe_offset": cursor, "word_text": segment[:80],
                                 "bpe_text": B[cursor:cursor + 80]})
            limit = cursor + 2 * len(segment) + pending * max_unk_chars + 1024
            got, cursor = _match_pieces(segment, B, cursor, limit)
            matched += got
        pending = 0
    if pending:                                          # the text ends in <unk>: the tail is the wildcard
        tail = len(B) - cursor
        if pending <= tail <= pending * max_unk_chars:
            wildcard_chars += tail
            cursor = len(B)
    unmatched_bpe = len(B) - matched - wildcard_chars
    same = matched == word_chars and unmatched_bpe == 0
    return {
        "same_text": bool(same),
        "char_match_fraction": matched / word_chars if word_chars else float("nan"),
        "bpe_chars_accounted_fraction": (matched + wildcard_chars) / len(B) if B else float("nan"),
        "word_chars": int(word_chars), "bpe_chars": int(len(B)), "matched_chars": int(matched),
        "unk_wildcards": int(n_unk), "unk_wildcard_chars": int(wildcard_chars),
        "unmatched_bpe_chars": int(unmatched_bpe), "mismatched_segments": int(mismatched_segments),
        "mismatch_examples": examples,
        "raw_counts": {
            "bpe": {"chars": len(bpe_text), "newlines": bpe_text.count("\n"), "unk": bpe_text.count(UNK),
                    **{marker: bpe_text.count(marker) for marker, _ in MARKERS}},
            "word": {"chars": len(word_text), "newlines": word_text.count("\n"), "unk": word_text.count(UNK),
                     **{marker: word_text.count(marker) for marker, _ in MARKERS}},
        },
        "normalization": {
            "whitespace": "all whitespace removed before comparison (line layout and detokenized spacing "
                          "carry no words)",
            "markers": "WikiText @-@, @,@, @.@ read as -, ',', '.' on both sides",
            "unk": f"each <unk> of the word-level text matches one original word of 1-{max_unk_chars} characters",
        },
    }


# ----------------------------------------------------------------------------------------------
# Coverage and perplexity
# ----------------------------------------------------------------------------------------------

def target_coverage(target_pos: np.ndarray, n_tokens: int) -> Dict[str, Any]:
    r"""Whether the scored targets are exactly the stream positions :math:`1, \dots, T-1`."""
    target_pos = np.asarray(target_pos, dtype=np.int64)
    expected = max(0, int(n_tokens) - 1)
    unique = np.unique(target_pos)
    in_range = unique[(unique >= 1) & (unique <= expected)]
    complete = (target_pos.size == expected and unique.size == expected and in_range.size == expected)
    return {"complete": bool(complete), "n_scored": int(target_pos.size), "expected": expected,
            "n_unique": int(unique.size), "duplicates": int(target_pos.size - unique.size),
            "missing": int(expected - in_range.size), "out_of_range": int(unique.size - in_range.size)}


def word_normalized_perplexity(
    nll:    np.ndarray,                           # (n,) per-target NLL in nats
    counts: Mapping[str, Mapping[str, Any]],      # word_counts(...)
) -> Dict[str, Any]:
    r"""Summed NLL, BPE perplexity, and the word-normalized perplexity under every convention."""
    nll = np.asarray(nll, dtype=np.float64)
    total = float(nll.sum())
    out: Dict[str, Any] = {"sum_nll": total, "n_targets": int(nll.size),
                           "bpe_ce": total / nll.size if nll.size else float("nan"),
                           "bpe_ppl": math.exp(total / nll.size) if nll.size else float("nan"),
                           "conventions": {}}
    for name, entry in counts.items():
        words = int(entry["count"])
        out["conventions"][name] = {"count": words, "nats_per_word": total / words,
                                    "ppl": math.exp(total / words), "bpe_targets_per_word": nll.size / words}
    return out


def calibration_for_run(
    dataset: str,
    tokens:  Optional["Any"],                     # (T,) the run's identity-checked BPE test stream
    passes:  Mapping[str, Tuple[np.ndarray, np.ndarray]],   # checkpoint -> (nll, target_pos)

    *,
    seq_len:             int,
    stride:              int,
    headline_checkpoint: str = "final_model.pt",
    word_stream_path:    "str | Path | None" = None,
    decode_bytes:        Optional[Callable[[Sequence[int]], bytes]] = None,
    word_stream:         Optional[WordStream] = None,
    steps:               Optional[Mapping[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    r"""The calibration row for every checkpoint with a test pass; skipped with a reason when undefined."""
    from vfe3.data.datasets import is_word_level, tiktoken_encoding_name

    base: Dict[str, Any] = {"dataset": dataset, "context": {"seq_len": int(seq_len), "stride": int(stride)},
                            "headline_checkpoint": headline_checkpoint, "headline_convention": HEADLINE_CONVENTION}
    if decode_bytes is None:
        if is_word_level(dataset):
            return {**base, "status": "not_applicable",
                    "reason": "word-level run: its perplexity is already per word-level token"}
        if not dataset.startswith(WIKITEXT_PREFIXES):
            return {**base, "status": "not_applicable",
                    "reason": f"no word-level reference stream is defined for dataset {dataset!r} (WikiText only)"}
        from vfe3.data.datasets import get_tiktoken_byte_decoder

        decode_bytes = get_tiktoken_byte_decoder(dataset)
        if decode_bytes is None:
            return {**base, "status": "not_applicable", "reason": f"no byte decoder for dataset {dataset!r}"}
        base["tokenizer"] = f"tiktoken {tiktoken_encoding_name(dataset)}"
    if tokens is None:
        return {**base, "status": "skipped", "reason": "the test stream was not loaded ('test' not in splits)"}
    if not passes:
        return {**base, "status": "skipped", "reason": "no successful test pass"}
    if word_stream is None:
        word_stream, reason = load_word_stream("test", path=word_stream_path)
        if word_stream is None:
            return {**base, "status": "skipped", "reason": reason}
    ids = tokens.reshape(-1).tolist()                                           # torch or numpy (T,)
    raw = decode_bytes(ids)
    bpe_text = raw.decode("utf-8", errors="replace")
    audit = text_audit(bpe_text, word_stream.text)
    audit["bpe_utf8_replacements"] = bpe_text.count("�")
    counts = word_counts(word_stream)
    n_tokens = len(ids)
    rows: Dict[str, Any] = {}
    for checkpoint, (nll, target_pos) in passes.items():
        coverage = target_coverage(target_pos, n_tokens)
        entry: Dict[str, Any] = {"coverage": coverage, "step": (steps or {}).get(checkpoint)}
        if coverage["complete"]:
            entry.update(word_normalized_perplexity(nll, counts))
            entry["status"] = "ok" if audit["same_text"] else "audit_failed"
        else:
            entry["status"] = "incomplete_coverage"
            entry["reason"] = (f"{coverage['n_scored']:,} scored targets, {coverage['missing']:,} of "
                               f"{coverage['expected']:,} stream targets missing: run the full test split")
        rows[checkpoint] = entry
    headline = rows.get(headline_checkpoint) or next(iter(rows.values()))
    return {**base, "status": headline["status"], "reason": headline.get("reason"),
            "word_stream": {"source": word_stream.source, "n_rows": word_stream.n_rows,
                            "n_blank_rows": word_stream.n_blank_rows, "n_words": word_stream.n_words},
            "word_counts": counts, "audit": audit, "checkpoints": rows, "n_bpe_tokens": n_tokens,
            "note": "the first BPE token of the stream has no context and is not a target"}


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------

def calibration_table(result: Mapping[str, Any]) -> str:
    r"""Markdown of ``tab:placeholder_calibration`` (evaluated row), the conventions and the audit."""
    status = result.get("status")
    if status not in ("ok", "audit_failed", "incomplete_coverage"):
        return f"Word-normalized perplexity {status}: {result.get('reason')}.\n"
    context = result["context"]
    context_text = f"{context['seq_len']} tokens, stride {context['stride']}"
    tokenizer = result.get("tokenizer", "BPE")
    head = result["checkpoints"].get(result["headline_checkpoint"]) or next(iter(result["checkpoints"].values()))
    conv = (head.get("conventions") or {}).get(result["headline_convention"], {})
    value = f"{conv['ppl']:.2f}" if conv else "–"
    if status == "audit_failed":
        value += " (TEXT AUDIT FAILED)"
    lines = ["| Model and source | Tokenization | Context | Word-normalized test PPL |", "|---|---|---|---|",
             f"| Evaluated realization ({result['headline_checkpoint']}) | {tokenizer} | {context_text} | {value} |",
             "", "| Weights | Step | Convention | Word-level tokens | Word-normalized PPL | BPE PPL | "
                 "BPE targets per word | Status |", "|---|---|---|---|---|---|---|---|"]
    for checkpoint, entry in result["checkpoints"].items():
        if not entry.get("conventions"):
            lines.append(f"| {checkpoint} | {entry.get('step')} | – | – | – | – | – | {entry['status']} |")
            continue
        for name, c in entry["conventions"].items():
            lines.append(f"| {checkpoint} | {entry.get('step')} | {name} | {c['count']:,} | {c['ppl']:.4f} | "
                         f"{entry['bpe_ppl']:.4f} | {c['bpe_targets_per_word']:.4f} | {entry['status']} |")
    audit = result["audit"]
    lines += ["", "Conventions: " + "; ".join(f"{k}: {v['rule']}" for k, v in result["word_counts"].items()) + ".",
              f"Text audit ({result['word_stream']['source']}): same text = {audit['same_text']}; "
              f"{100 * audit['char_match_fraction']:.4f}% of {audit['word_chars']:,} word-level characters matched "
              f"in order; {100 * audit['bpe_chars_accounted_fraction']:.4f}% of {audit['bpe_chars']:,} BPE characters "
              f"accounted for, {audit['unk_wildcards']:,} <unk> wildcards covering {audit['unk_wildcard_chars']:,} "
              f"characters; {audit['mismatched_segments']} mismatched segments. Normalization: "
              + "; ".join(audit["normalization"].values()) + "."]
    if status == "incomplete_coverage":
        lines.append(f"Coverage incomplete: {head.get('reason')}.")
    return "\n".join(lines) + "\n"


def write_calibration_outputs(result: Mapping[str, Any], out_dir: Path) -> List[Path]:
    r"""``tables/calibration.md`` and ``csv/calibration.csv``."""
    out_dir = Path(out_dir)
    table = out_dir / "tables" / "calibration.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(calibration_table(result), encoding="utf-8")
    written = [table]
    rows = [{"checkpoint": checkpoint, "step": entry.get("step"), "convention": name, "word_tokens": c["count"],
             "word_ppl": c["ppl"], "nats_per_word": c["nats_per_word"], "bpe_ppl": entry["bpe_ppl"],
             "sum_nll": entry["sum_nll"], "n_targets": entry["n_targets"], "status": entry["status"]}
            for checkpoint, entry in (result.get("checkpoints") or {}).items()
            for name, c in (entry.get("conventions") or {}).items()]
    if rows:
        path = out_dir / "csv" / "calibration.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written
