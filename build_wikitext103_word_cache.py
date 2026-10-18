r"""Build the WORD-level WikiText-103 token cache consumed by ``vfe3.data.datasets``.

Produces, in the shared cache directory, for ``DATASET`` (``wikitext-103-word`` or
``wikitext-103-word-eos``):

  ``{DATASET}_{train,validation,test}_word_tokens.pt``   int64 id streams
  ``..._tokens.pt.meta.json``                            per-split build provenance
  ``{DATASET}_word_vocab.json``                          the id-to-word table
  ``..._tokens.pt.provenance.json``                      per-split tokenizer-provenance manifest

The run is source-bound: the corpus revision is pinned, and the tokenizer code is hashed into the
sidecar, so a cache that was produced by different code or a different revision is detectable. Each
split's metadata also records the SHA-256 of the vocabulary in id order (``vocabulary_sha256``), which
the loader checks against the sidecar, so a same-size reordered table is refused; and each split gets a
``.provenance.json`` manifest binding its payload digest and tokenizer identity, which the loader
reports as ``tokenizer_provenance_status='manifest_verified'``.

Vocabulary (see :mod:`vfe3.data.word_vocab`): every whitespace token type observed in the TRAIN
split, ordered by descending frequency then lexicographically, with ``<unk>`` reserved at id 0.
For ``wikitext-103-word`` that is 267,734 ids (the 267,734 observed types, ``<unk>`` among them).
``wikitext-103-word-eos`` appends one ``<eos>`` to every row, blank rows included, and fixes it at
id 1; that is the published WikiText-103 stream, vocabulary 267,735 and 103,227,021 / 217,646 /
245,569 tokens. A reference build asserts the vocabulary size (and, for the ``<eos>`` stream, the
three token counts) before anything is written, so a preprocessing drift fails loudly rather than
silently re-indexing every embedding table. ``MIN_COUNT`` raises the cutoff to trade OOV for a
smaller ``phi_embed``.

Click-to-run: edit the constants below (``DATASET``, ``MIN_COUNT``, ``CACHE_DIR``, ``FORCE``) and run
the file. ``MIN_COUNT = DEFAULT_MIN_COUNT`` is the reference build; ``MIN_COUNT = 3`` a capped
vocabulary.
Every split target is checked for existence before anything is written, so a refused rerun leaves
the existing caches and their vocabulary untouched. Each split's metadata and token cache are
published atomically, and the vocabulary sidecar is published after every cache, so an interrupted
build can leave caches whose recorded ``vocab_size``/``min_count``/``vocabulary_sha256`` disagree with
the sidecar -- which the loader rejects -- but never a sidecar that silently relabels caches encoded
with a different table. A split's earlier provenance manifest is removed before its new payload is
published and a new one is written after the sidecar, so an interrupted build leaves a cache unverified
rather than bound to a manifest of the payload it replaced.

The corpus is fetched from the Hugging Face hub at the pinned revision on first run (about 500 MB
of parquet) and re-read from the local hub cache afterwards.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from vfe3.data.datasets import cache_path, default_cache_dir, publish_cache_provenance, word_vocabulary_sha256
from vfe3.data.word_vocab import (
    DEFAULT_MIN_COUNT,
    REFERENCE_N_TOKENS_EOS,
    WORD_DATASET,
    WORD_EOS_DATASET,
    WordVocabulary,
    appends_eos,
    count_tokens,
    reference_vocab_size,
    tokenize_row,
    vocabulary_from_counts,
    vocabulary_path,
)

# ---- click-to-run settings --------------------------------------------------------------------
DATASET: str = WORD_DATASET               # "wikitext-103-word" | "wikitext-103-word-eos" (published stream)
MIN_COUNT: int = DEFAULT_MIN_COUNT        # keep types occurring STRICTLY MORE than this many times
CACHE_DIR: Optional[Path] = None          # None -> ~/.cache/tokenized_cache
FORCE: bool = False                       # True overwrites existing token caches

WORD_DATASETS = (WORD_DATASET, WORD_EOS_DATASET)
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-v1"
DATASET_REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"

# Declared parquet order per split: the token stream is the concatenation of these objects in this
# order, then their text rows in stored order (the same text_row_policy the sibling caches record).
SPLIT_FILES: Dict[str, Tuple[str, ...]] = {
    "train": (
        f"{DATASET_CONFIG}/train-00000-of-00002.parquet",
        f"{DATASET_CONFIG}/train-00001-of-00002.parquet",
    ),
    "validation": (f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",),
    "test": (f"{DATASET_CONFIG}/test-00000-of-00001.parquet",),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenization_code_sha256() -> str:
    r"""Hash the tokenizer code so a cache records which implementation produced it."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in (here / "build_wikitext103_word_cache.py", here / "vfe3" / "data" / "word_vocab.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _download(file_name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        DATASET_ID, file_name, repo_type="dataset", revision=DATASET_REVISION))


def _read_rows(file_name: str) -> List[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(_download(file_name), columns=["text"])
    return table.column("text").to_pylist()


def _count_split_tokens(split: str, *, eos: bool = False) -> "tuple[int, int, int, object]":
    r"""Count tokens and characters for ``split`` one parquet object at a time.

    Returns ``(n_rows, n_tokens, n_chars, counts)``; with ``eos`` every row contributes one
    ``<eos>``. Streaming per object keeps the peak memory at one shard's rows: the full training
    split is 1.8M rows and ~535M characters.
    """
    from collections import Counter

    counts: Counter = Counter()
    n_rows = n_tokens = n_chars = 0
    for file_name in SPLIT_FILES[split]:
        rows = _read_rows(file_name)
        n_rows += len(rows)
        n_chars += sum(len(row) for row in rows)
        shard_counts = count_tokens(rows, eos=eos)
        counts += shard_counts
        n_tokens += sum(shard_counts.values())
        del rows
    return n_rows, n_tokens, n_chars, counts


def _encode_split(split: str, vocabulary: WordVocabulary) -> "tuple[np.ndarray, int, int]":
    r"""Encode ``split`` into one flat int64 array in a single pass, shard by shard (rows end in
    ``<eos>`` when the vocabulary is an ``<eos>`` vocabulary).

    Returns ``(ids, n_rows, n_chars)``. Each row is tokenized once and its ids appended to the shard's
    int64 buffer, so the shard's rows are read and tokenized once (the former sizing pass tokenized
    every row twice; audit 2026-10-17 item 79). Each parquet object becomes its own array,
    concatenated at the end, so nothing holds more than one shard's rows plus the finished stream.
    """
    from array import array

    eos = vocabulary.eos
    shards = []
    n_rows = n_chars = 0
    for file_name in SPLIT_FILES[split]:
        rows = _read_rows(file_name)
        n_rows += len(rows)
        buffer = array("q")
        for row in rows:
            n_chars += len(row)
            tokens = tokenize_row(row, eos=eos)
            if tokens:
                buffer.extend(vocabulary.encode(tokens))
        # A writable zero-copy view: the array keeps the buffer alive after the name is dropped.
        shards.append(np.frombuffer(buffer, dtype=np.int64) if len(buffer) else np.empty(0, dtype=np.int64))
        del rows, buffer
    ids = shards[0] if len(shards) == 1 else np.concatenate(shards)
    return ids, n_rows, n_chars


def _atomic_publish(target: Path, write: Callable[[Path], None]) -> None:
    r"""Write through ``write`` to a sibling temporary file, then ``os.replace`` it onto ``target``."""
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(handle)
    try:
        write(Path(temporary))
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build(
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    cache_dir: Path | None = None,
    force: bool = False,
    dataset: Optional[str] = None,
) -> Dict[str, object]:
    r"""Build the vocabulary and every split's token cache; return a provenance summary.

    ``dataset`` (default: the module-level ``DATASET``) is ``wikitext-103-word`` or
    ``wikitext-103-word-eos``; the latter ends every row in ``<eos>`` (id 1).
    """
    dataset = DATASET if dataset is None else dataset
    if dataset not in WORD_DATASETS:
        raise SystemExit(f"DATASET={dataset!r} is not a word dataset; expected one of {WORD_DATASETS}")
    eos = appends_eos(dataset)
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Refuse BEFORE writing anything: a rerun at a different cutoff must not replace the vocabulary
    # sidecar that the existing caches' ids are defined against.
    targets = {split: cache_path(dataset, split, suffix="pt", cache_dir=root) for split in SPLIT_FILES}
    existing = [str(target) for target in targets.values() if target.exists()]
    if existing and not force:
        raise SystemExit(f"{', '.join(existing)} already exists; set FORCE = True to rebuild")

    print(f"[1/3] counting training tokens (dataset={dataset}, min_count={min_count}) ...", flush=True)
    n_train_rows, n_train_tokens, n_train_chars, counts = _count_split_tokens("train", eos=eos)
    # (rows, tokens, characters) of each split counted so far: the encode pass checks against them instead of
    # recounting (the training split was tokenized four times before; audit 2026-10-17 item 79).
    split_counts: Dict[str, Tuple[int, int, int]] = {"train": (n_train_rows, n_train_tokens, n_train_chars)}
    print(f"      train rows={n_train_rows:,} tokens={n_train_tokens:,} "
          f"chars={n_train_chars:,} types={len(counts):,}", flush=True)

    words = vocabulary_from_counts(counts, min_count=min_count, eos=eos)
    vocabulary = WordVocabulary(words, min_count=min_count, eos=eos)
    expected_vocab = reference_vocab_size(dataset)
    if min_count == DEFAULT_MIN_COUNT and vocabulary.vocab_size != expected_vocab:
        raise SystemExit(
            f"reference vocabulary mismatch: built {vocabulary.vocab_size} ids, expected "
            f"{expected_vocab}. The corpus revision or the tokenization changed; refusing to "
            f"write a cache whose ids would not mean what the benchmark's ids mean."
        )
    if eos and min_count == DEFAULT_MIN_COUNT:
        # The published stream's sizes are a property of every split, not only of the vocabulary:
        # check all three before anything is written (validation and test are small to recount).
        measured = {"train": n_train_tokens}
        for split in SPLIT_FILES:
            if split != "train":
                split_counts[split] = _count_split_tokens(split, eos=True)[:3]
                measured[split] = split_counts[split][1]
        wrong = {split: (measured[split], REFERENCE_N_TOKENS_EOS[split]) for split in measured
                 if measured[split] != REFERENCE_N_TOKENS_EOS.get(split)}
        if wrong:
            raise SystemExit(
                "reference token-count mismatch (measured, expected): "
                + ", ".join(f"{split}={pair}" for split, pair in sorted(wrong.items()))
                + ". The corpus revision or the <eos> policy changed; refusing to write the cache.")
    print(f"      vocabulary: {vocabulary.vocab_size:,} ids "
          f"({len(counts):,} observed types, min_count={min_count})", flush=True)

    vocab_file = vocabulary_path(dataset, tag="word", cache_dir=root)

    code_sha = _tokenization_code_sha256()
    summary: Dict[str, object] = {
        "dataset": dataset,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "min_count": min_count,
        "vocab_size": vocabulary.vocab_size,
        "vocabulary_file": vocab_file.name,
        "tokenization_code_sha256": code_sha,
        "splits": {},
    }

    print("[2/3] encoding splits ...", flush=True)
    for split in SPLIT_FILES:
        target = targets[split]
        meta_target = Path(str(target) + ".meta.json")
        ids, n_rows, n_chars_source = _encode_split(split, vocabulary)
        if split in split_counts:
            counted_rows, counted_tokens, counted_chars = split_counts[split]
            if (ids.size, n_rows, n_chars_source) != (counted_tokens, counted_rows, counted_chars):
                raise AssertionError(
                    f"{split}: encoded {ids.size} ids over {n_rows} rows and {n_chars_source} characters but "
                    f"counted {counted_tokens} tokens over {counted_rows} rows and {counted_chars} characters")
        meta = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "split": split,
            "tokenizer": "word",
            "min_count": min_count,
            "vocab_size": vocabulary.vocab_size,
            "n_tokens": int(ids.size),
            "n_chars_source": int(n_chars_source),
            "tokens_per_char": (float(ids.size) / n_chars_source) if n_chars_source else None,
            "tokenization_code_sha256": code_sha,
            # The table in id order: the loader refuses a sidecar of the same size in another order.
            "vocabulary_sha256": word_vocabulary_sha256(vocabulary.words),
            "tokenizer_config_sha256": hashlib.sha256(json.dumps(
                {"tokenizer": "word", "min_count": min_count, "vocab_size": vocabulary.vocab_size,
                 **({"eos": True} if eos else {})},
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            # Recorded only by <eos> builds (the no-<eos> metadata carries no "eos" key); the
            # loader requires the vocabulary sidecar to agree on it.
            **({"eos": True} if eos else {}),
        }
        # Metadata first, then the ids: until the vocabulary is replaced below, a new cache is always
        # visible to the loader as recorded vocab_size/min_count disagreeing with the old sidecar.
        meta_text = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        _atomic_publish(meta_target, lambda path: path.write_text(meta_text, encoding="utf-8"))
        # The manifest of the payload being replaced goes first; the new one follows the sidecar below.
        Path(str(target) + ".provenance.json").unlink(missing_ok=True)
        _atomic_publish(target, lambda path: torch.save(torch.from_numpy(ids), path))
        summary["splits"][split] = {
            "rows": n_rows,
            "n_tokens": int(ids.size),
            "n_chars_source": int(n_chars_source),
            "tokens_per_char": meta["tokens_per_char"],
            "path": str(target),
        }
        print(f"      {split:11s} rows={n_rows:>9,} tokens={ids.size:>11,} "
              f"chars={n_chars_source:>11,} tpc={meta['tokens_per_char']:.6f}", flush=True)

    # The vocabulary is published (atomically, see WordVocabulary.save) after every cache it defines is
    # in place; the provenance manifests read its vocab_size, so they follow it.
    vocabulary.save(vocab_file)
    for split in SPLIT_FILES:
        summary["splits"][split]["provenance"] = str(publish_cache_provenance(dataset, split, cache_dir=root))
    print(f"[3/3] wrote {vocab_file} and the provenance manifests; done", flush=True)
    return summary


def main() -> int:
    r"""Click-to-run entry point: build ``DATASET`` with the module-level ``MIN_COUNT``/``CACHE_DIR``/``FORCE``."""
    summary = build(min_count=MIN_COUNT, cache_dir=CACHE_DIR, force=FORCE, dataset=DATASET)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
