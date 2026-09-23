r"""Build the WORD-level WikiText-103 token cache consumed by ``vfe3.data.datasets``.

Produces, in the shared cache directory:

  ``wikitext-103-word_{train,validation,test}_word_tokens.pt``   int64 id streams
  ``..._tokens.pt.meta.json``                                    per-split build provenance
  ``wikitext-103-word_word_vocab.json``                          the id-to-word table

The run is source-bound: the corpus revision is pinned, and the tokenizer code is hashed into the
sidecar, so a cache that was produced by different code or a different revision is detectable.

Vocabulary (see :mod:`vfe3.data.word_vocab`): every whitespace token type observed in the TRAIN
split, ordered by descending frequency then lexicographically, with ``<unk>`` reserved at id 0.
That reproduces the benchmark's quoted 267,735 (= 267,734 observed types + ``<unk>``); the builder
asserts the count so a preprocessing drift fails loudly rather than silently re-indexing every
embedding table. ``--min-count`` raises the cutoff to trade OOV for a smaller ``phi_embed``.

Usage::

    python build_wikitext103_word_cache.py                 # reference build
    python build_wikitext103_word_cache.py --min-count 3   # capped vocabulary

The corpus is fetched from the Hugging Face hub at the pinned revision on first run (about 500 MB
of parquet) and re-read from the local hub cache afterwards.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from vfe3.data.datasets import cache_path, default_cache_dir
from vfe3.data.word_vocab import (
    DEFAULT_MIN_COUNT,
    REFERENCE_VOCAB_SIZE,
    WordVocabulary,
    count_tokens,
    tokenize_row,
    vocabulary_from_counts,
    vocabulary_path,
)

DATASET = "wikitext-103-word"
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


def _count_split_tokens(split: str) -> "tuple[int, int, int, object]":
    r"""Count tokens and characters for ``split`` one parquet object at a time.

    Returns ``(n_rows, n_tokens, n_chars, counts)``. Streaming per object keeps the peak memory at
    one shard's rows: the full training split is 1.8M rows and ~535M characters.
    """
    from collections import Counter

    counts: Counter = Counter()
    n_rows = n_tokens = n_chars = 0
    for file_name in SPLIT_FILES[split]:
        rows = _read_rows(file_name)
        n_rows += len(rows)
        n_chars += sum(len(row) for row in rows)
        shard_counts = count_tokens(rows)
        counts += shard_counts
        n_tokens += sum(shard_counts.values())
        del rows
    return n_rows, n_tokens, n_chars, counts


def _encode_split(split: str, vocabulary: WordVocabulary) -> np.ndarray:
    r"""Encode ``split`` into one flat int64 array, shard by shard.

    Each parquet object becomes its own array, concatenated at the end, so nothing holds more than
    one shard's rows plus the finished stream.
    """
    shards = []
    for file_name in SPLIT_FILES[split]:
        rows = _read_rows(file_name)
        n_tokens = sum(len(tokenize_row(row)) for row in rows)
        out = np.empty(n_tokens, dtype=np.int64)
        cursor = 0
        for row in rows:
            tokens = tokenize_row(row)
            if tokens:
                ids = vocabulary.encode(tokens)
                out[cursor:cursor + len(ids)] = ids
                cursor += len(ids)
        if cursor != n_tokens:
            raise AssertionError(f"encoded {cursor} ids but sized the array for {n_tokens}")
        shards.append(out)
        del rows
    return shards[0] if len(shards) == 1 else np.concatenate(shards)


def build(
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Dict[str, object]:
    r"""Build the vocabulary and every split's token cache; return a provenance summary."""
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] counting training tokens (min_count={min_count}) ...", flush=True)
    n_train_rows, n_train_tokens, n_train_chars, counts = _count_split_tokens("train")
    print(f"      train rows={n_train_rows:,} tokens={n_train_tokens:,} "
          f"chars={n_train_chars:,} types={len(counts):,}", flush=True)

    words = vocabulary_from_counts(counts, min_count=min_count)
    vocabulary = WordVocabulary(words, min_count=min_count)
    if min_count == DEFAULT_MIN_COUNT and vocabulary.vocab_size != REFERENCE_VOCAB_SIZE:
        raise SystemExit(
            f"reference vocabulary mismatch: built {vocabulary.vocab_size} ids, expected "
            f"{REFERENCE_VOCAB_SIZE}. The corpus revision or the tokenization changed; refusing to "
            f"write a cache whose ids would not mean what the benchmark's ids mean."
        )
    print(f"      vocabulary: {vocabulary.vocab_size:,} ids "
          f"({len(counts):,} observed types, min_count={min_count})", flush=True)

    vocab_file = vocabulary_path(DATASET, tag="word", cache_dir=root)
    vocabulary.save(vocab_file)
    print(f"      wrote {vocab_file}", flush=True)

    code_sha = _tokenization_code_sha256()
    summary: Dict[str, object] = {
        "dataset": DATASET,
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
        target = cache_path(DATASET, split, suffix="pt", cache_dir=root)
        meta_target = Path(str(target) + ".meta.json")
        if target.exists() and not force:
            raise SystemExit(f"{target} already exists; pass --force to rebuild")
        n_rows, n_tokens, n_chars_source, _counts = _count_split_tokens(split)
        ids = _encode_split(split, vocabulary)
        if ids.size != n_tokens:
            raise AssertionError(f"{split}: encoded {ids.size} ids but counted {n_tokens}")
        torch.save(torch.from_numpy(ids), target)
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
            "tokenizer_config_sha256": hashlib.sha256(json.dumps(
                {"tokenizer": "word", "min_count": min_count, "vocab_size": vocabulary.vocab_size},
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
        meta_target.write_text(
            json.dumps(meta, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        summary["splits"][split] = {
            "rows": n_rows,
            "n_tokens": int(ids.size),
            "n_chars_source": int(n_chars_source),
            "tokens_per_char": meta["tokens_per_char"],
            "path": str(target),
        }
        print(f"      {split:11s} rows={n_rows:>9,} tokens={ids.size:>11,} "
              f"chars={n_chars_source:>11,} tpc={meta['tokens_per_char']:.6f}", flush=True)

    print("[3/3] done", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help="keep types occurring STRICTLY MORE than this many times "
                             f"(default {DEFAULT_MIN_COUNT} = the reference full vocabulary)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="cache directory (default: ~/.cache/tokenized_cache)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing token caches")
    args = parser.parse_args(argv)
    summary = build(min_count=args.min_count, cache_dir=args.cache_dir, force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
