r"""Build the GPT-2 BPE WikiText-103 token cache by its reconstructed recipe, or check an existing cache against it.

The ``wikitext-103`` caches that every reported run reads (``wikitext-103_{train,validation,test}_tiktoken_tokens.pt``
under ``~/.cache/tokenized_cache``) were built outside this repository and carry no provenance manifest, so the loader
records them as ``tokenizer_provenance_status = "filename_inferred_unverified"``. Their recipe was reconstructed on
2026-10-21 (README, data section): take each split's rows of the Hugging Face ``Salesforce/wikitext`` raw text
(``wikitext-103-raw-v1``) in stored order, drop the empty rows, join the remaining rows (each ends in its own newline)
with ``"\n\n"``, replace the markers ``" @-@ "``, ``" @,@ "`` and ``" @.@ "`` by ``"-"``, ``","`` and ``"."``, and
encode with tiktoken's ``gpt2`` encoding (``encode_ordinary``), with no ``<|endoftext|>`` between rows or articles.
This script applies that recipe at a pinned revision of the raw text.

The recipe was reconstructed against the ``<unk>``-masked ``wikitext-103-v1`` text, where a few lines differ in a
whitespace detail, and has not been confirmed on the raw text byte for byte. So by default the script writes nothing
over an existing cache: when ``DATASET``'s caches exist and ``FORCE`` is False, each split is encoded and compared
token by token with the existing stream, and the report says whether the recipe reproduces it.
``PUBLISH_MANIFEST = True`` then writes the tokenizer-provenance manifest of every split
(:func:`vfe3.data.datasets.publish_cache_provenance`), but only when all three splits match. Publishing a manifest,
like a rebuild, changes the source identity that checkpoints and selected models bind, so do either only when no resume
of an existing checkpoint is pending.

A build (no cache for ``DATASET`` yet, or ``FORCE``) writes, per split, the ids in ``TOKEN_DTYPE`` (int32 by default:
half the page cache of the existing int64 caches; the loader reads either), a ``.pt.meta.json`` with the token, row and
source-character counts, the recipe and the source revision, and the provenance manifest. The encoding runs over
segments of about ``SEGMENT_CHARS`` characters cut only before a row that starts with one space and a non-space
character. The GPT-2 pre-tokenizer always ends a token before such a row (the whitespace run before it stops short of
its leading space), so the concatenated segment encodings equal the encoding of the whole text.

The raw text is fetched from the Hugging Face hub at the pinned revision on first run (about 300 MB of parquet) and
re-read from the local hub cache afterwards.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from vfe3.data.datasets import cache_path, default_cache_dir, load_cached_tokens, publish_cache_provenance
from vfe3.data.datasets import tiktoken_encoding_name

# ---- click-to-run settings --------------------------------------------------------------------
DATASET: str = "wikitext-103"             # a GPT-2 BPE dataset name of vfe3.data.datasets ("wikitext-103")
CACHE_DIR: Optional[Path] = None          # None -> ~/.cache/tokenized_cache
FORCE: bool = False                       # True overwrites existing caches (new payload digests: no resume across it)
PUBLISH_MANIFEST: bool = False            # True | False: publish the manifests of existing caches when every split matches
TOKEN_DTYPE: str = "int32"                # "int32" (new builds) | "int64" (the dtype of the existing caches)

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "6231e49f19a707241d6b84d9cff60a3a86b85a85"

# Declared parquet order per split; the stream is these objects' text rows in stored order.
SPLIT_FILES: Dict[str, Tuple[str, ...]] = {
    "train": (
        f"{DATASET_CONFIG}/train-00000-of-00002.parquet",
        f"{DATASET_CONFIG}/train-00001-of-00002.parquet",
    ),
    "validation": (f"{DATASET_CONFIG}/validation-00000-of-00001.parquet",),
    "test": (f"{DATASET_CONFIG}/test-00000-of-00001.parquet",),
}

# The WikiText number and hyphen markers and their merged forms, replaced in this order.
MARKERS: Tuple[Tuple[str, str], ...] = ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."))
ROW_JOINT = "\n\n"
TOKEN_DTYPES = ("int32", "int64")
SEGMENT_CHARS = 1 << 20
RECIPE = ("rows in stored order, empty rows dropped, joined with '\\n\\n'; ' @-@ ', ' @,@ ', ' @.@ ' replaced by "
          "'-', ',', '.'; tiktoken gpt2 encode_ordinary, no <|endoftext|>")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(file_name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET_ID, file_name, repo_type="dataset", revision=DATASET_REVISION))


def _read_rows(file_name: str) -> List[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(_download(file_name), columns=["text"])
    return table.column("text").to_pylist()


def _split_rows(split: str) -> Iterator[str]:
    r"""The rows of ``split``, one parquet object at a time (peak memory: one shard's rows)."""
    for file_name in SPLIT_FILES[split]:
        yield from _read_rows(file_name)


def _merge_markers(text: str) -> str:
    for marker, replacement in MARKERS:
        text = text.replace(marker, replacement)
    return text


def recipe_text(rows: Iterable[str]) -> str:
    r"""The text the recipe encodes: the non-empty ``rows`` joined with ``"\n\n"``, markers merged.

    A marker holds no newline, so it never spans a joint, and merging the markers row by row gives the same text.
    """
    return _merge_markers(ROW_JOINT.join(row for row in rows if row))


def _starts_pretoken(row: str) -> bool:
    r"""Whether the GPT-2 pre-tokenizer starts a token at ``row`` when a newline precedes it.

    Before ``" X"`` (one space, then a non-space ``X``) the pattern ``\s+(?!\S)`` ends the preceding whitespace run
    one character short of the space, and ``" X"`` opens the next token, so no token crosses into the row.
    ``str.isspace`` accepts every Unicode White_Space character, so the test errs toward not cutting.
    """
    return len(row) >= 2 and row[0] == " " and not row[1].isspace()


def encode_rows(
    rows:          Iterable[str],

    *,
    segment_chars: int = SEGMENT_CHARS,
) -> Tuple[np.ndarray, int]:
    r"""``(ids, n_chars)`` of :func:`recipe_text` of ``rows`` under the gpt2 encoding: int64 ids and the text length.

    The text is encoded in segments of at least ``segment_chars`` characters, each cut before a row that
    :func:`_starts_pretoken` admits, so the concatenated ids equal the encoding of the whole text.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("gpt2")
    pieces: List[np.ndarray] = []
    n_chars = 0
    segment: List[str] = []
    size = 0

    def flush(text: str) -> None:
        nonlocal n_chars
        pieces.append(np.asarray(encoding.encode_ordinary(text), dtype=np.int64))
        n_chars += len(text)

    for row in rows:
        if not row:
            continue
        text = _merge_markers(row)
        if segment and size >= segment_chars and _starts_pretoken(text):
            flush(ROW_JOINT.join(segment) + ROW_JOINT)
            segment, size = [], 0
        segment.append(text)
        size += len(text) + len(ROW_JOINT)
    if segment:
        flush(ROW_JOINT.join(segment))
    ids = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
    return ids, n_chars


def _code_sha256() -> str:
    r"""Hash this script, so a cache records which implementation of the recipe produced it."""
    return _sha256_file(Path(__file__).resolve())


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


def _comparison(existing: torch.Tensor, ids: np.ndarray) -> Dict[str, object]:
    r"""How the recipe's ids compare with an existing stream: equal, or the first index where they part."""
    old = np.asarray(existing.numpy(), dtype=np.int64)
    common = min(old.size, ids.size)
    unequal = np.flatnonzero(old[:common] != ids[:common])
    first = int(unequal[0]) if unequal.size else (None if old.size == ids.size else common)
    return {
        "status": "matches_existing" if first is None else "differs_from_existing",
        "n_tokens": int(ids.size),
        "existing_n_tokens": int(old.size),
        "first_difference": first,
        "differing_positions": int(unequal.size) if old.size == ids.size else None,
    }


def build(
    *,
    cache_dir:        Optional[Path] = None,
    force:            bool = False,
    dataset:          Optional[str] = None,
    publish_manifest: bool = False,
    token_dtype:      Optional[str] = None,
) -> Dict[str, object]:
    r"""Build ``dataset``'s three caches by the recipe, or compare existing caches with it; return a summary.

    Without ``force``, an existing cache of any split selects the comparison: nothing is written, and with
    ``publish_manifest`` the manifests are published when every split matches its existing stream.
    """
    dataset = DATASET if dataset is None else dataset
    if tiktoken_encoding_name(dataset) != "gpt2":
        raise SystemExit(f"DATASET={dataset!r} is not a GPT-2 BPE dataset (encoding "
                         f"{tiktoken_encoding_name(dataset)!r})")
    token_dtype = TOKEN_DTYPE if token_dtype is None else token_dtype
    if token_dtype not in TOKEN_DTYPES:
        raise ValueError(f"token_dtype must be one of {TOKEN_DTYPES}, got {token_dtype!r}")
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    targets = {split: cache_path(dataset, split, suffix="pt", cache_dir=root) for split in SPLIT_FILES}
    existing = {split for split in SPLIT_FILES
                if targets[split].exists() or cache_path(dataset, split, suffix="bin", cache_dir=root).exists()}
    compare = bool(existing) and not force
    code_sha = _code_sha256()
    summary: Dict[str, object] = {
        "dataset": dataset, "mode": "compare" if compare else "build", "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG, "dataset_revision": DATASET_REVISION, "recipe": RECIPE,
        "token_dtype": token_dtype, "code_sha256": code_sha, "splits": {}, "manifests_published": False,
    }
    for split in SPLIT_FILES:
        print(f"[{split}] encoding by the recipe ...", flush=True)
        n_rows = 0

        def counted(rows: Iterable[str]) -> Iterator[str]:
            nonlocal n_rows
            for row in rows:
                n_rows += bool(row)
                yield row

        ids, n_chars = encode_rows(counted(_split_rows(split)))
        if compare:
            if split not in existing:
                record: Dict[str, object] = {"status": "missing", "n_tokens": int(ids.size)}
            else:
                record = _comparison(load_cached_tokens(dataset, split, cache_dir=root), ids)
            summary["splits"][split] = {**record, "n_rows": n_rows, "n_chars_source": n_chars}
            print(f"      {split}: {record['status']} ({record['n_tokens']:,} tokens)", flush=True)
            continue
        if token_dtype == "int32" and ids.size and int(ids.max()) >= 2 ** 31:
            raise ValueError("GPT-2 ids exceed int32")                    # unreachable for a 50,257-id vocabulary
        stored = ids.astype(np.int32) if token_dtype == "int32" else ids
        target = targets[split]
        meta = {
            "schema_version": 1, "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION, "split": split, "tokenizer": "tiktoken",
            "tokenizer_encoding": "gpt2", "token_dtype": token_dtype, "n_rows": n_rows, "n_tokens": int(ids.size),
            "n_chars_source": int(n_chars), "tokens_per_char": (float(ids.size) / n_chars) if n_chars else None,
            "recipe": RECIPE, "code_sha256": code_sha,
        }
        meta_text = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        _atomic_publish(Path(str(target) + ".meta.json"), lambda path: path.write_text(meta_text, encoding="utf-8"))
        # The manifest of the payload being replaced goes first; the new one is published after the payload.
        Path(str(target) + ".provenance.json").unlink(missing_ok=True)
        _atomic_publish(target, lambda path: torch.save(torch.from_numpy(stored), path))
        manifest = publish_cache_provenance(dataset, split, cache_dir=root)
        summary["splits"][split] = {"status": "written", "n_rows": n_rows, "n_tokens": int(ids.size),
                                    "n_chars_source": int(n_chars), "path": str(target), "provenance": str(manifest)}
        print(f"      {split}: wrote {ids.size:,} tokens over {n_rows:,} rows ({n_chars:,} characters)", flush=True)
    if compare and publish_manifest:
        if all(summary["splits"][split]["status"] == "matches_existing" for split in SPLIT_FILES):
            for split in SPLIT_FILES:
                summary["splits"][split]["provenance"] = str(publish_cache_provenance(dataset, split, cache_dir=root))
            summary["manifests_published"] = True
        else:
            print("      manifests not published: not every split matches its existing stream", flush=True)
    return summary


def main() -> int:
    r"""Click-to-run entry point: build or compare ``DATASET`` with the module-level settings."""
    summary = build(cache_dir=CACHE_DIR, force=FORCE, dataset=DATASET, publish_manifest=PUBLISH_MANIFEST,
                    token_dtype=TOKEN_DTYPE)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
