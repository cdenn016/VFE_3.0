r"""Tokenized-dataset loading for VFE_3.0 (reads a pre-tokenized cache; no tokenization).

The cache holds 1-D token-id streams under
``{cache_dir}/{dataset}_{split}_{tokenizer}_tokens.{pt|bin}``:
  .pt   a torch int64 tensor (torch.load), e.g. wikitext-103 (gpt2), wiki-ja (cl100k).
  .bin  a raw int32 memmap with an ``n_tokens`` sidecar ``.meta.json``, e.g. wiki-en.
``TokenWindows`` slices the stream into causal-LM (input, target) windows:
``input = tokens[i:i+L]``, ``target = tokens[i+1:i+L+1]``. No neural code, no CLI.
"""

import codecs
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vfe3.run_archive import _sha256_file  # noqa: F401  (re-exported: ablation.py imports it from here)

_CL100K_DATASETS = ("wiki-ja", "wiki-en", "wiki-ar")

# Datasets whose cache holds WORD-level ids (whitespace tokens, not BPE). Their vocabulary is a
# sidecar JSON beside the cache (vfe3.data.word_vocab) rather than a fixed published table, so the
# bound used by validate_token_range is read from the table the ids were actually produced from.
# "wikitext-103-word-eos" is the published stream (one <eos> per row, vocabulary 267,735); the
# no-<eos> "wikitext-103-word" cache is a separate dataset and is unchanged.
_WORD_DATASETS = ("wikitext-103-word", "wikitext-103-word-eos")

# Integer dtypes an on-disk token cache may hold. Uncapped loads keep any of these mapped in their
# native width (no corpus-sized int64 copy); a capped load owns an int64 clone. Used by the cache
# loader and TokenWindows to reject bool / floating / complex caches before they index a table.
SUPPORTED_TOKEN_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


def default_cache_dir() -> Path:
    """The shared tokenized cache: ``~/.cache/tokenized_cache``."""
    return Path.home() / ".cache" / "tokenized_cache"


def _tokenizer_tag(dataset: str) -> str:
    """Cache tokenizer tag: word for the word-level corpora, cl100k for multilingual, GPT-2 else."""
    if dataset in _WORD_DATASETS:
        return "word"
    return "tiktoken_cl100k" if dataset in _CL100K_DATASETS else "tiktoken"


_TOKENIZER_VOCAB_SIZE = {"tiktoken": 50257, "tiktoken_cl100k": 100277}
_TIKTOKEN_ENCODING_NAME = {"tiktoken": "gpt2", "tiktoken_cl100k": "cl100k_base", "word": "word"}
_CACHE_PROVENANCE_SCHEMA_VERSION = 1


def is_word_level(dataset: str) -> bool:
    """Whether ``dataset``'s cache holds word-level ids rather than BPE ids."""
    return _tokenizer_tag(dataset) == "word"


def tiktoken_encoding_name(dataset: str) -> str:
    """Return the encoding name used to build ``dataset``'s cache.

    Word-level caches report ``"word"``: they have no BPE encoding, and naming one would misstate
    the cache's provenance. Every other dataset reports its tiktoken encoding.
    """
    return _TIKTOKEN_ENCODING_NAME[_tokenizer_tag(dataset)]


def tokenizer_vocab_size(dataset: str, *, cache_dir: Optional[Path] = None) -> int:
    r"""Return the vocabulary bound for the tokenizer used by ``dataset``'s cache.

    BPE caches use a fixed published encoding, so their bound is a constant. A word-level cache's
    vocabulary is built from the training split at a chosen frequency cutoff, so its bound is read
    from the vocabulary sidecar that the ids were produced from; the reference size is returned
    only when that sidecar is absent (the corpus was never built), which keeps this callable during
    provenance bookkeeping without a cache present.
    """
    tag = _tokenizer_tag(dataset)
    if tag == "word":
        from vfe3.data.word_vocab import reference_vocab_size

        vocabulary = _load_word_vocabulary(dataset, tag, cache_dir=cache_dir)
        return reference_vocab_size(dataset) if vocabulary is None else vocabulary.vocab_size
    return _TOKENIZER_VOCAB_SIZE[tag]


_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+")


def _validate_path_component(
    name: str,
    role: str,
) -> None:
    """Reject names that are not safe, non-special single path components."""
    if name in (".", "..") or _SAFE_COMPONENT_RE.fullmatch(name) is None:
        raise ValueError(
            f"{role} {name!r} is not a safe path component: use only letters, digits, "
            f"'.', '_', '-' (no path separators, drive prefixes, or special dot names)"
        )


def cache_path(
    dataset:    str,
    split:      str,

    *,
    suffix:     str            = "pt",
    cache_dir:  Optional[Path] = None,
) -> Path:
    """Build the cache file path for ``dataset``/``split`` with extension ``suffix``."""
    _validate_path_component(dataset, "dataset")
    _validate_path_component(split,   "split")
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return root / f"{dataset}_{split}_{_tokenizer_tag(dataset)}_tokens.{suffix}"


def _resolved_binary_cache_paths(
    source: Path,
) -> Tuple[Path, Path, Path]:
    """Resolve a binary payload and derive its sidecars from that one target path."""
    payload = source.resolve(strict=True)
    return (
        payload,
        Path(str(payload) + ".meta.json"),
        Path(str(payload) + ".provenance.json"),
    )


def _binary_cache_metadata(
    path: Path,
) -> Tuple[Dict[str, object], np.dtype]:
    r"""Read and validate a binary-cache sidecar before any memory mapping.

    A binary token stream is valid only when the sidecar exists, ``n_tokens`` is an exact positive
    integer, the declared dtype is a supported integer token dtype, and the file contains exactly
    ``n_tokens * dtype.itemsize`` bytes. Both truncation and extension fail closed: neither is a
    legitimate alternate view of the declared logical corpus.
    """
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"binary token cache metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"binary token cache metadata {meta_path} must be a JSON object")
    n_tokens = meta.get("n_tokens")
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens <= 0:
        raise ValueError(
            f"binary token cache metadata {meta_path} n_tokens must be a positive integer")
    try:
        dtype = np.dtype(meta.get("dtype", "int32"))
    except TypeError as exc:
        raise TypeError(
            f"binary token cache metadata {meta_path} has invalid dtype {meta.get('dtype')!r}") from exc
    _require_supported_numpy_token_dtype(dtype, source=path)
    actual_bytes = int(path.stat().st_size)
    expected_bytes = int(n_tokens) * int(dtype.itemsize)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"binary token cache {path} file bytes={actual_bytes}; expected exactly "
            f"{expected_bytes} from n_tokens={n_tokens} and dtype={dtype}")
    canonical_meta = dict(meta)
    canonical_meta["dtype"] = dtype.name
    return canonical_meta, dtype


def _expected_cache_provenance(
    dataset:        str,
    split:          str,
    payload_sha256: str,

    *,
    cache_dir:      Path,
) -> Dict[str, object]:
    r"""The tokenizer-provenance manifest that binds one cache payload (see :func:`_cache_tokenizer_provenance`)."""
    return {
        "schema_version": _CACHE_PROVENANCE_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "tokenizer_tag": _tokenizer_tag(dataset),
        "tokenizer_encoding": tiktoken_encoding_name(dataset),
        "tokenizer_vocab_size": tokenizer_vocab_size(dataset, cache_dir=cache_dir),
        "payload_sha256": payload_sha256,
    }


def publish_cache_provenance(
    dataset:   str,
    split:     str,

    *,
    cache_dir: Optional[Path] = None,
) -> Path:
    r"""Write the ``<cache>.provenance.json`` manifest of ``dataset``/``split``'s current payload; return its path.

    The payload resolves as in :func:`cache_source_identity` (the ``.pt`` cache first, then the ``.bin``
    memmap), and the manifest binds its SHA-256 digest, the dataset, split, tokenizer tag, encoding and
    vocabulary size, so the loader reports ``tokenizer_provenance_status='manifest_verified'`` for it. A
    word cache reads its vocabulary size from the sidecar, so publish the manifest after the sidecar. The
    file is replaced atomically. Publishing a manifest changes the cache's source identity, which runs bind.
    """
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if pt.exists():
        payload = pt.resolve()
    elif binp.exists():
        payload = _resolved_binary_cache_paths(binp)[0]
    else:
        raise FileNotFoundError(f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}")
    manifest = _expected_cache_provenance(dataset, split, _sha256_file(payload), cache_dir=payload.parent)
    target = Path(str(payload) + ".provenance.json")
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def _cache_tokenizer_provenance(
    path:           Path,
    dataset:       str,
    split:         str,
    payload_sha256: str,
) -> Dict[str, object]:
    r"""Validate an optional exact tokenizer-provenance manifest for one cache payload.

    Legacy caches remain readable, but their tokenizer identity is labeled as filename-inferred and
    unverified. A present ``<cache>.provenance.json`` must bind the exact payload digest, dataset,
    split, tokenizer tag, encoding, and vocabulary size; malformed or mismatched manifests fail closed.
    """
    manifest_path = Path(str(path) + ".provenance.json")
    if not manifest_path.exists():
        return {
            "tokenizer_provenance_status": "filename_inferred_unverified",
            "tokenizer_provenance": None,
            "tokenizer_provenance_sha256": None,
        }
    if not manifest_path.is_file():
        raise ValueError(f"token cache provenance manifest must be a regular file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"token cache provenance manifest is unreadable: {manifest_path}") from exc
    required = {
        "schema_version",
        "dataset",
        "split",
        "tokenizer_tag",
        "tokenizer_encoding",
        "tokenizer_vocab_size",
        "payload_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError(f"token cache provenance manifest has an unsupported schema: {manifest_path}")
    expected = _expected_cache_provenance(dataset, split, payload_sha256, cache_dir=path.parent)
    differing = sorted(key for key in required if manifest.get(key) != expected[key])
    if differing:
        raise ValueError(
            f"token cache provenance manifest disagrees with the cache for field(s) {differing}"
        )
    return {
        "tokenizer_provenance_status": "manifest_verified",
        "tokenizer_provenance": dict(manifest),
        "tokenizer_provenance_sha256": _sha256_file(manifest_path),
    }


def cache_source_identity(
    dataset:   str,
    split:     str = "validation",

    *,
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Return the tokenizer, format, byte size, and SHA-256 identity of one cache source.

    Resolves the concrete cache file for ``dataset``/``split`` -- the ``.pt`` tensor cache first,
    then the ``.bin`` memmap with its ``.meta.json`` sidecar -- and returns a mapping that binds
    reuse to the EXACT bytes on disk: the resolved cache ``format``, byte ``size_bytes``, the
    streamed ``sha256`` digest, the tokenizer tag, and (for ``.bin``) the parsed sidecar ``meta``
    plus its own ``meta_sha256`` digest, so a corpus edit that leaves the binary untouched but
    changes ``n_tokens`` is still caught. Raises ``FileNotFoundError`` when neither cache exists
    (the caller treats an unhashable source as "forbid reuse").

    Every call re-hashes the source bytes. Filesystem stat tuples are not content identities: an
    in-place writer can preserve path, size, timestamps, and inode while changing the corpus. Run
    code that owns an immutable loaded dataset may retain the returned contract on that dataset,
    but a fresh filesystem identity request never trusts prior stat metadata.
    """
    tokenizer_tag = _tokenizer_tag(dataset)
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        resolved = pt.resolve()
        stat = resolved.stat()
        payload_sha256 = _sha256_file(resolved)
        identity: Dict[str, object] = {
            "format":        "pt",
            "tokenizer_tag": tokenizer_tag,
            "size_bytes":    int(stat.st_size),
            "sha256":        payload_sha256,
            "meta":          None,
            "meta_sha256":   None,
        }
        identity.update(_cache_tokenizer_provenance(
            resolved, dataset, split, payload_sha256,
        ))
        return identity

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        resolved, meta_path, _provenance_path = _resolved_binary_cache_paths(binp)
        meta, _ = _binary_cache_metadata(resolved)
        stat = resolved.stat()
        payload_sha256 = _sha256_file(resolved)
        identity = {
            "format":        "bin",
            "tokenizer_tag": tokenizer_tag,
            "size_bytes":    int(stat.st_size),
            "sha256":        payload_sha256,
            "meta":          meta,
            "meta_sha256":   _sha256_file(meta_path),
        }
        identity.update(_cache_tokenizer_provenance(
            resolved, dataset, split, payload_sha256,
        ))
        return identity

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
    )


def _require_supported_token_dtype(
    tokens: torch.Tensor,
    source: Path,
) -> None:
    if tokens.dtype not in SUPPORTED_TOKEN_DTYPES:
        names = ", ".join(sorted(str(dtype) for dtype in SUPPORTED_TOKEN_DTYPES))
        raise TypeError(f"token cache {source} has dtype {tokens.dtype}; expected one of {names}")


def _require_supported_numpy_token_dtype(dtype: np.dtype, source: Path) -> None:
    try:
        torch_dtype = torch.from_numpy(np.empty((0,), dtype=dtype)).dtype
    except TypeError as exc:
        raise TypeError(f"token cache {source} has unsupported dtype {dtype}") from exc
    if torch_dtype not in SUPPORTED_TOKEN_DTYPES:
        names = ", ".join(sorted(str(item) for item in SUPPORTED_TOKEN_DTYPES))
        raise TypeError(f"token cache {source} has dtype {dtype}; expected one of {names}")


def _validated_token_limit(
    name:  str,
    value: Optional[int],
) -> Optional[int]:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{name} must be a positive plain integer or None, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive plain integer or None, got {value!r}")
    return value


def load_cached_tokens(
    dataset:    str,
    split:      str            = "validation",

    *,
    cache_dir:  Optional[Path] = None,
    limit:      Optional[int]  = None,
) -> torch.Tensor:
    """Load the 1-D token-id stream for ``dataset``/``split``.

    Tries the ``.pt`` (torch.load) cache, then the ``.bin`` memmap (size and dtype from the
    ``.meta.json`` sidecar). An uncapped load keeps the corpus mapped in its NATIVE integer
    dtype (any of :data:`SUPPORTED_TOKEN_DTYPES`) -- no corpus-sized int64 copy is allocated;
    the caller (``TokenWindows``) converts to ``torch.long`` before indexing. A ``limit`` yields
    an OWNED ``torch.long`` clone of exactly the first ``limit`` tokens, sliced before any int64
    materialization. The dtype is validated before either branch can slice or cast, so bool /
    floating / complex caches are rejected rather than silently reinterpreted as token ids.
    Raises FileNotFoundError if neither cache is present.
    """
    limit = _validated_token_limit("limit", limit)
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        tokens = torch.load(pt, weights_only=True, mmap=True)
        _require_supported_token_dtype(tokens, source=pt)
        if tokens.dim() != 1 or not tokens.is_contiguous():
            raise ValueError(f"token cache {pt} must be a contiguous 1-D tensor")
        if limit is not None:
            return tokens[:limit].clone().to(torch.long)
        return tokens

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        resolved, _meta_path, _provenance_path = _resolved_binary_cache_paths(binp)
        meta, dtype = _binary_cache_metadata(resolved)
        n = int(meta["n_tokens"])
        mm = np.memmap(resolved, dtype=dtype, mode="r", shape=(n,))
        if limit is not None:
            capped = torch.from_numpy(np.asarray(mm[:limit]))
            return capped.clone() if capped.dtype == torch.long else capped.to(torch.long)
        tokens = torch.from_numpy(np.asarray(mm))
        _require_supported_token_dtype(tokens, source=resolved)
        return tokens

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
    )


def word_vocabulary_sha256(words: Sequence[str]) -> str:
    r"""SHA-256 of a word table in id order (UTF-8 JSON list), binding a word cache to its exact id map."""
    return hashlib.sha256(json.dumps(list(words), ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _require_word_cache_matches_vocabulary(
    dataset:   str,
    split:     str,

    *,
    cache_dir: Optional[Path] = None,
) -> None:
    r"""Fail closed when a word cache's build metadata disagrees with the vocabulary sidecar.

    A word cache's ids mean something only relative to the vocabulary they were encoded with. The
    builder records that vocabulary's ``vocab_size`` and ``min_count`` in the cache's
    ``.pt.meta.json``; the loader resolves ids through the separate vocabulary sidecar. When the
    metadata records either field, the sidecar must exist and agree on it, otherwise the cache would
    be silently decoded, bounded, and identity-bound against a different table. BPE caches, and word
    caches whose metadata records neither field, are unaffected.
    """
    if not is_word_level(dataset):
        return
    meta_path = Path(str(cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)) + ".meta.json")
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"word cache metadata {meta_path} is unreadable") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"word cache metadata {meta_path} must be a JSON object")
    # "eos" is recorded only by <eos> builds, so the no-<eos> caches are checked exactly as before.
    recorded = {key: meta[key] for key in ("vocab_size", "min_count", "eos") if key in meta}
    if not recorded:
        return
    from vfe3.data.word_vocab import vocabulary_path

    vocabulary = _load_word_vocabulary(dataset, _tokenizer_tag(dataset), cache_dir=cache_dir)
    sidecar = vocabulary_path(dataset, tag=_tokenizer_tag(dataset), cache_dir=cache_dir)
    if vocabulary is None:
        raise FileNotFoundError(
            f"word cache {meta_path.name} records {recorded} but its vocabulary sidecar {sidecar} is "
            "absent; rebuild the word cache")
    actual = {"vocab_size": vocabulary.vocab_size, "min_count": vocabulary.min_count,
              "eos": vocabulary.eos}
    # A cache whose metadata records the table digest is bound to the exact id order, which catches a
    # same-size reordered table (audit 2026-10-08 gap-data-lifecycle#6). Older metadata lacks the key.
    if "vocabulary_sha256" in meta:
        recorded["vocabulary_sha256"] = meta["vocabulary_sha256"]
        actual["vocabulary_sha256"] = word_vocabulary_sha256(vocabulary.words)
    differing = sorted(key for key in recorded if recorded[key] != actual[key])
    if differing:
        encoded_with = {key: recorded[key] for key in differing}
        sidecar_has = {key: actual[key] for key in differing}
        raise ValueError(
            f"word cache {meta_path.name} was encoded with {encoded_with} but the vocabulary "
            f"sidecar {sidecar} has {sidecar_has}; the cache ids would be read against a different "
            "table. Rebuild the word cache (FORCE = True in build_wikitext103_word_cache.py)")


def _load_identity_bound_tokens(
    dataset:   str,
    split:     str,

    *,
    cache_dir: Optional[Path] = None,
    limit:     Optional[int]  = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    r"""Load tokens only when the exact source identity is stable across the load."""
    _require_word_cache_matches_vocabulary(dataset, split, cache_dir=cache_dir)
    identity_before = cache_source_identity(dataset, split, cache_dir=cache_dir)
    tokens = (
        load_cached_tokens(dataset, split, cache_dir=cache_dir)
        if limit is None
        else load_cached_tokens(dataset, split, cache_dir=cache_dir, limit=limit)
    )
    if limit is None and identity_before["format"] == "bin":
        tokens = tokens.clone()
    identity_after = cache_source_identity(dataset, split, cache_dir=cache_dir)
    if identity_before != identity_after:
        differing = sorted(
            key for key in set(identity_before) | set(identity_after)
            if identity_before.get(key) != identity_after.get(key)
        )
        raise RuntimeError(
            f"token cache for {dataset!r}/{split!r} changed while loading "
            f"(identity fields: {', '.join(differing)})"
        )
    return tokens, identity_before


def cached_token_count(
    dataset:    str,
    split:      str            = "validation",

    *,
    cache_dir:  Optional[Path] = None,
) -> int:
    """Return the cached token count without materializing the token stream."""
    pt = cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)
    if pt.exists():
        return int(torch.load(pt, weights_only=True, mmap=True).numel())

    binp = cache_path(dataset, split, suffix="bin", cache_dir=cache_dir)
    if binp.exists():
        resolved, _meta_path, _provenance_path = _resolved_binary_cache_paths(binp)
        meta, _ = _binary_cache_metadata(resolved)
        return int(meta["n_tokens"])

    raise FileNotFoundError(
        f"no tokenized cache for {dataset!r}/{split!r}: tried {pt} and {binp}"
    )


def validate_token_range(
    tokens:     torch.Tensor,        # (T,) 1-D token-id stream

    vocab_size: int,

    *,
    dataset:    str,
    cache_dir:  Optional[Path] = None,
) -> None:
    """Raise ValueError unless every token id satisfies ``0 <= id < vocab_size``."""
    if tokens.numel() == 0:
        return
    lo, hi = int(tokens.min()), int(tokens.max())
    if lo < 0:
        raise ValueError(
            f"dataset {dataset!r} (tokenizer {_tokenizer_tag(dataset)!r}) contains negative token ids "
            f"down to {lo}; repair or rebuild the tokenized cache before loading it"
        )
    if hi >= vocab_size:
        required_vocab_size = max(hi + 1, tokenizer_vocab_size(dataset, cache_dir=cache_dir))
        raise ValueError(
            f"dataset {dataset!r} (tokenizer {_tokenizer_tag(dataset)!r}) has token ids in "
            f"[{lo}, {hi}] but vocab_size={vocab_size}: set vocab_size >= {required_vocab_size}"
        )


def _recorded_source_chars(
    dataset:    str,
    split:      str,
    n_tokens:   int,
    *,
    cache_dir:  Optional[Path] = None,
) -> Optional[int]:
    r"""The source character count recorded beside a ``.pt`` cache, or None.

    Returns the value only when the sidecar exists, is well formed, and its ``n_tokens`` equals the
    stream actually loaded: a sidecar describing a different stream is not evidence about this one.
    """
    sidecar = Path(str(cache_path(dataset, split, suffix="pt", cache_dir=cache_dir)) + ".meta.json")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("n_tokens") != n_tokens:
        return None
    n_chars = payload.get("n_chars_source")
    if type(n_chars) is not int or n_chars <= 0:
        return None
    return n_chars


_WORD_VOCABULARY_CACHE: dict = {}


def _load_word_vocabulary(
    dataset:   str,
    tag:       str,
    *,
    cache_dir: Optional[Path] = None,
):
    r"""Load ``dataset``'s word vocabulary sidecar, or None when it is absent or unreadable.

    Memoized on the file's size and mtime so a rebuilt vocabulary is picked up rather than served
    stale. A malformed vocabulary raises rather than returning None: an unreadable table means the
    ids cannot be interpreted, which is a provenance failure, not a missing-feature condition.
    """
    from vfe3.data.word_vocab import WordVocabulary, vocabulary_path

    path = vocabulary_path(dataset, tag=tag, cache_dir=cache_dir)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _WORD_VOCABULARY_CACHE:
        _WORD_VOCABULARY_CACHE[key] = WordVocabulary.load(path)
    return _WORD_VOCABULARY_CACHE[key]


def get_tiktoken_decoder(
    dataset:   str,
    *,
    cache_dir: Optional[Path] = None,
) -> 'Optional[Callable[[Sequence[int]], str]]':
    """A ``decode(token_ids) -> str`` for ``dataset``'s tokenizer, or None if unavailable.

    Uses the SAME tokenizer the cache was built with (cl100k for wiki-ja/wiki-en/wiki-ar, GPT-2
    otherwise, word-level vocabulary for the ``-word`` corpora, matching :func:`_tokenizer_tag`), so
    generated ids map back to text consistently. Lazy-imports tiktoken and returns None when
    tiktoken is absent or the dataset has no real tokenizer (the synthetic period-3 anchor), letting
    the caller treat None as "no sample text".

    A word-level cache decodes through the vocabulary sidecar in the selected cache directory;
    it returns None when that sidecar is absent.
    """
    if dataset == "synthetic-period3":
        return None
    tag = _tokenizer_tag(dataset)
    if tag == "word":
        vocabulary = _load_word_vocabulary(dataset, tag, cache_dir=cache_dir)
        if vocabulary is None:
            return None
        return lambda ids: vocabulary.decode([int(t) for t in ids])
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding(_TIKTOKEN_ENCODING_NAME[tag])
    return lambda ids: enc.decode([int(t) for t in ids])


def get_tiktoken_byte_decoder(
    dataset: str,
) -> 'Optional[Callable[[Sequence[int]], bytes]]':
    r"""A bounded ``decode_bytes(token_ids)`` callable for the cache tokenizer, or ``None``.

    Byte decoding plus an incremental UTF-8 decoder lets :func:`tokens_per_char` preserve the exact
    whole-stream codepoint semantics even when one Unicode scalar spans tokens on opposite sides of
    a bounded token chunk.
    """
    if dataset == "synthetic-period3":
        return None
    if _tokenizer_tag(dataset) == "word":
        return None                                # word ids carry no byte-level composition
    try:
        import tiktoken
    except ImportError:
        return None
    enc = tiktoken.get_encoding(tiktoken_encoding_name(dataset))
    return lambda ids: enc.decode_bytes([int(token) for token in ids])


_TOKENS_PER_CHAR_CACHE: dict = {}


def tokens_per_char(
    dataset:      str,
    split:        str,

    *,
    chunk_tokens: int            = 64 * 1024,

    cache_dir:    Optional[Path] = None,
) -> Optional[float]:
    r"""Corpus constant ``n_tokens / n_unicode_codepoints`` for ``dataset``/``split``, or None.

    The bits-per-character correction factor: ``BPC = (CE / ln 2) * tokens_per_char`` turns the
    model's bits-per-TOKEN into true bits-per-CHARACTER, so PPL/BPC compare across tokenizers and
    languages (gpt2 vs cl100k; English vs Japanese/Arabic, where a token spans ~3 codepoints).
    Computed by decoding the split's cached token stream once with its OWN tokenizer and counting
    Unicode codepoints incrementally in bounded token chunks. The tiktoken path decodes bytes and
    carries UTF-8 state across chunk boundaries, so a codepoint split across two BPE tokens is counted
    exactly once. Returns None when the dataset has no real tokenizer (the synthetic anchor),
    tiktoken is absent, or the cache is missing. Memoization includes the exact cache
    source identity, so replacing bytes or metadata in place cannot serve stale normalization.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    try:
        tokens, source_identity = _load_identity_bound_tokens(
            dataset, split, cache_dir=cache_dir)
    except FileNotFoundError:
        return None
    n_tokens = int(tokens.numel())
    # A word-level cache carries the EXACT source character count in its build sidecar, because a
    # word stream cannot be decoded back to the source spacing (joining tokens with single spaces
    # drops each row's newline and leading space -- about 0.7% of the denominator). Prefer the
    # recorded count there, and only when it describes the very stream loaded. A BPE cache keeps
    # decoding: its tokens round-trip to the source text, so the decoded count is self-verifying and
    # a stale sidecar cannot silently rescale every reported BPC.
    recorded_chars = (
        _recorded_source_chars(dataset, split, n_tokens, cache_dir=cache_dir)
        if is_word_level(dataset) else None
    )
    key = (
        dataset,
        split,
        str(cache_dir) if cache_dir is not None else None,
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")),
        recorded_chars,
    )
    if key in _TOKENS_PER_CHAR_CACHE:
        return _TOKENS_PER_CHAR_CACHE[key]
    if recorded_chars is not None:
        tpc = (n_tokens / recorded_chars) if recorded_chars > 0 else None
        _TOKENS_PER_CHAR_CACHE[key] = tpc
        return tpc
    decode_bytes = get_tiktoken_byte_decoder(dataset)
    decode_text = None if decode_bytes is not None else get_tiktoken_decoder(dataset, cache_dir=cache_dir)
    if decode_bytes is None and decode_text is None:
        return None
    n_chars = 0
    if decode_bytes is not None:
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for start in range(0, n_tokens, chunk_tokens):
            token_chunk = tokens[start:start + chunk_tokens].tolist()
            n_chars += len(utf8.decode(decode_bytes(token_chunk), final=False))
        n_chars += len(utf8.decode(b"", final=True))
    else:
        assert decode_text is not None
        for start in range(0, n_tokens, chunk_tokens):
            token_chunk = tokens[start:start + chunk_tokens].tolist()
            n_chars += len(decode_text(token_chunk))
    tpc = (n_tokens / n_chars) if n_chars > 0 else None
    _TOKENS_PER_CHAR_CACHE[key] = tpc
    return tpc


class TokenWindows(Dataset):
    r"""Causal-LM windows over a 1-D token stream: item ``i`` -> (input_L, target_L).

    ``pad_final`` selects the EVALUATION contract: cover every transition of the stream exactly
    once, padding the final window and marking any position that must not be scored with the
    ``-100`` ignore target. Training leaves it off and simply walks the stream by ``stride``.

    STRIDE UNDER THE EVAL CONTRACT. With the default ``stride == seq_len`` the windows are
    disjoint, so the first target of every window is predicted from a single token of context, the
    second from two, and so on: the reported cross-entropy is inflated by that context starvation,
    and the inflation grows as ``seq_len`` shrinks. Setting ``stride < seq_len`` gives the standard
    sliding-window evaluation instead -- each window still runs on its full ``seq_len`` of context,
    but only its final ``stride`` targets are scored and the overlapping prefix is masked to
    ``-100``, so every transition is counted exactly ONCE and at ``seq_len - stride`` tokens of
    context or better. The masked prefix is context, not padding: ``train.evaluate`` runs the model
    on the whole window and weights it by the count of non-ignored targets, so the masked positions
    inform the prediction without entering the average. This is the recipe in HuggingFace's
    fixed-length-model perplexity documentation; it costs ``seq_len / stride`` times more forward
    passes and always reports a LOWER perplexity than the disjoint windows, so the two are different
    metrics and must not be mixed inside one comparison.

    Window ``i`` starts at ``i * stride`` and scores transitions ``[max(0, i*stride + seq_len -
    stride), i*stride + seq_len)``, which tile the stream contiguously from zero; ``n`` is the
    smallest count whose last window reaches the final transition. At ``stride == seq_len`` every
    formula below reduces to the disjoint case exactly (the mask covers zero positions and
    ``n == ceil(transitions / seq_len)``), so the default path is byte-identical.

    TRAINING-WINDOW ORIGIN. ``window_offset_mode='fixed'`` (a new dataset, and every evaluation stream)
    starts window ``i`` at ``i * stride``. ``'per_epoch_random'`` (:meth:`use_per_epoch_offsets`, which
    :func:`vfe3.train.train` applies to every training window dataset) starts window ``i`` at
    :math:`o + i s` with an origin :math:`o \in [0, b)` that :meth:`draw_window_offset` draws once per
    epoch. With :math:`T` tokens, window length :math:`L` and stride :math:`s`, the origin bound is
    :math:`b = \min(s, T - L)`, so every origin leaves at least one full window: :math:`b = s` once
    :math:`T \ge L + s`, and a shorter stream (a smoke run with a small token budget,
    :math:`L + 1 \le T < L + s`) draws its origin from the :math:`T - L` that fit instead of being refused
    (audit 2026-10-12 round 2, item 16). The window count is the one that fits every origin,

    .. math::
        n = \Big\lfloor \frac{T - 1 - L - (b - 1)}{s} \Big\rfloor + 1,

    so the epoch length does not depend on the draw. At :math:`s = L` an epoch scores the transitions
    :math:`[o, o + nL)` exactly once; the :math:`o` transitions before the origin and the
    :math:`T - 1 - o - nL < 2L` after the last window (the tail) are not scored in that epoch, and a
    different draw moves them. Under ``'fixed'`` nothing is drawn.
    """

    def __init__(
        self,
        tokens:    torch.Tensor,           # (T,) 1-D token-id stream
        seq_len:   int,

        *,
        pad_final: bool          = False,  # eval: cover every transition once, ignore target = -100

        stride:    Optional[int] = None,   # None -> non-overlapping seq_len stride
    ) -> None:
        if tokens.dim() != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}")
        if tokens.dtype not in SUPPORTED_TOKEN_DTYPES:
            raise ValueError(f"tokens must have integer dtype, got {tokens.dtype}")
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.pad_final = pad_final
        if self.seq_len <= 0 or self.stride <= 0:
            raise ValueError("seq_len and stride must be positive")
        # A stride ABOVE seq_len would skip transitions between consecutive windows, which no
        # masking can recover, so the exactly-once contract still rejects it. Below seq_len the
        # windows overlap and the prefix mask below restores exactly-once.
        if self.pad_final and self.stride > self.seq_len:
            raise ValueError(
                f"pad_final needs stride <= seq_len so consecutive windows leave no gap; got "
                f"stride={self.stride}, seq_len={self.seq_len}")
        # Positions of every window after the first that repeat the previous window's transitions.
        self.overlap_prefix = (self.seq_len - self.stride) if self.pad_final else 0
        self.window_offset_mode = "fixed"
        self.window_offset = 0
        if self.pad_final:
            transitions = self.tokens.numel() - 1
            if transitions <= 0:
                raise ValueError("a padded evaluation stream must contain at least two tokens")
            # Window i ends at transition i*stride + seq_len; take the fewest windows reaching the
            # last one. At stride == seq_len this is exactly ceil(transitions / seq_len) as before.
            remaining = transitions - self.seq_len
            self.n = max(1, -(-remaining // self.stride) + 1)
            self._fixed_n = self.n
            return
        usable = self.tokens.numel() - seq_len - 1
        if usable < 0:
            raise ValueError(f"stream of {self.tokens.numel()} too short for seq_len={seq_len}")
        self.n = usable // self.stride + 1
        self._fixed_n = self.n

    def use_per_epoch_offsets(self) -> None:
        r"""Switch to per-epoch random origins (see the class docstring); the origin resets to 0.

        A training mode: an evaluation stream (``pad_final``) keeps its fixed origin.
        """
        if self.pad_final:
            raise ValueError("evaluation windows (pad_final=True) keep a fixed origin")
        slack = self.tokens.numel() - self.seq_len - 1          # the largest origin with one full window
        self.offset_bound = min(self.stride, slack + 1)         # origins o in [0, offset_bound)
        usable = slack - (self.offset_bound - 1)
        self.window_offset_mode, self.window_offset, self.n = "per_epoch_random", 0, usable // self.stride + 1

    def draw_window_offset(self, generator: Optional[torch.Generator]) -> int:
        r"""Draw this epoch's origin :math:`o \sim \mathcal{U}\{0, \dots, b - 1\}`, :math:`b = \min(s, T - L)`, from
        ``generator`` (None: the global CPU RNG) and set it."""
        if self.window_offset_mode == "fixed":
            raise RuntimeError("fixed-origin windows draw no offset")
        self.window_offset = int(torch.randint(self.offset_bound, (1,), generator=generator).item())
        return self.window_offset

    def at_origin(self) -> "TokenWindows":
        r"""A fixed-origin view of these windows: same stream and stride, origin 0, and the ``'fixed'`` mode's
        window count (a ``'per_epoch_random'`` dataset fits one window fewer whenever the stride's slack
        does not fit; audit 2026-10-09 round 2, item 34)."""
        view = copy.copy(self)
        view.window_offset_mode, view.window_offset, view.n = "fixed", 0, self._fixed_n
        return view

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self.n
        if idx < 0 or idx >= self.n:
            raise IndexError(idx)
        start = self.window_offset + idx * self.stride
        end = start + self.seq_len
        window = self.tokens[start:end + 1].to(torch.long)
        if self.pad_final and window.numel() < self.seq_len + 1:
            n_real = max(int(window.numel()) - 1, 0)
            inputs = torch.zeros(self.seq_len, dtype=torch.long)
            targets = torch.full((self.seq_len,), -100, dtype=torch.long)
            if n_real:
                inputs[:n_real] = window[:-1]
                targets[:n_real] = window[1:]
        else:
            inputs, targets = window[:-1], window[1:]
        if self.overlap_prefix and idx > 0:
            # Clone first: ``.to(torch.long)`` is a no-op for an int64 stream, so ``targets`` is a
            # VIEW into self.tokens and masking in place would corrupt the corpus for every later
            # window. Window 0 keeps its full target row -- it is the only one whose prefix is not
            # a repeat, and dropping it would leave the first seq_len-stride transitions unscored.
            targets = targets.clone()
            targets[:self.overlap_prefix] = -100
        return inputs, targets


def make_dataloader(
    dataset:     str,
    split:       str,
    seq_len:     int,
    batch_size:  int,

    *,
    shuffle:    bool = True,
    drop_last:  bool = True,

    stride:     Optional[int]             = None,
    cache_dir:  Optional[Path]            = None,
    max_tokens: Optional[int]             = None,   # cap the stream (fast smoke runs)
    vocab_size: Optional[int]             = None,   # check token ids fit cfg.vocab_size-sized tables
    generator:  Optional[torch.Generator] = None,   # fix shuffle order independent of global RNG
) -> DataLoader:
    """Build a DataLoader of causal-LM windows from the cached ``dataset``/``split``.

    ``shuffle`` / ``drop_last`` default to the TRAIN regime (shuffle the stream, drop the partial
    last batch). Evaluation must pass ``shuffle=False, drop_last=False`` so validation/test are a
    stable corpus measurement that reads the WHOLE split (the token-weighted CE in
    ``train.evaluate`` is order-independent, but a dropped tail and a randomly-varying drawn subset
    are not -- see _select_loader). ``max_tokens`` is applied while loading; when supplied,
    ``vocab_size`` rejects cached ids that cannot index the model's vocabulary-sized tables."""
    for name, value in (("shuffle", shuffle), ("drop_last", drop_last)):
        if type(value) is not bool:
            raise TypeError(
                f"{name} must be a plain bool, got {type(value).__name__}: {value!r}"
            )

    max_tokens = _validated_token_limit("max_tokens", max_tokens)
    tokens, source_identity = _load_identity_bound_tokens(
        dataset, split, cache_dir=cache_dir, limit=max_tokens)
    if vocab_size is not None:
        validate_token_range(tokens, vocab_size, dataset=dataset, cache_dir=cache_dir)
    ds = TokenWindows(
        tokens,
        seq_len,
        stride=stride,
        pad_final=(not shuffle and not drop_last),
    )
    ds.cache_dir = cache_dir
    ds.data_identity = {
        "schema_version":       2,
        "dataset":              dataset,
        "split":                split,
        "tokenizer_tag":        _tokenizer_tag(dataset),
        "tokenizer_encoding":   tiktoken_encoding_name(dataset),
        "tokenizer_vocab_size": tokenizer_vocab_size(dataset, cache_dir=cache_dir),
        "model_vocab_size":     (int(vocab_size) if vocab_size is not None else None),
        "max_tokens":           max_tokens,
        "source":               source_identity,
    }
    # pin_memory only when a CUDA device exists (it would pin host pages uselessly on a CPU-only
    # box). With pinned host buffers the per-step .to(device, non_blocking=True) H2D copy in
    # train()/evaluate() can overlap compute; num_workers stays 0 (the dataset is an in-memory
    # tensor slice, so worker IPC would cost more than it saves).
    # generator=None (default) keeps the RandomSampler drawing each epoch permutation from the GLOBAL
    # RNG -- byte-identical to the historic behavior. Passing a seeded generator fixes the shuffle
    # order independent of the global RNG (used by the multi-seed variance floor so the data order is
    # shared across seeds while model-init RNG still varies; see train_vfe3.DATA_SEED / EXP-1).
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available(), generator=generator)
