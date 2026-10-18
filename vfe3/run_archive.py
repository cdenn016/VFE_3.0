r"""Source-tree archive and tokenizer manifest for a run directory (audit 2026-09-28, Section 2).

A SHA plus a dirty-tree fingerprint (``provenance.json``) names a source generation but cannot
recover its bytes. This module stores the bytes themselves: when a run directory is created,
:func:`write_source_archive` zips the executable source tree -- every ``vfe3/**/*.py``, every
repository-root ``*.py`` driver, and ``pyproject.toml`` -- into ``<run_dir>/source_archive.zip``
and records its hash, the Git HEAD, and the list of dirty files. At finalization
:func:`verify_source_archive` re-hashes the tree; on drift it writes a second archive
(``source_archive_end.zip``), so both generations a drifted run may have mixed are recoverable.

Hashes. ``tree_sha256`` is a content hash of the archived files,

.. math::
    h = \mathrm{SHA256}\Big(\big\Vert_{f \in F \text{ (sorted)}}
        \ \ell(p_f)\, p_f\, \ell(b_f)\, b_f\Big),

with :math:`p_f` the POSIX relative path, :math:`b_f` the raw file bytes (no line-ending
normalization: the archive stores what executed), and :math:`\ell(\cdot)` an 8-byte big-endian
length prefix. ``sha256`` is the hash of the zip file itself, which is deterministic for fixed
content because every member carries a fixed timestamp and the members are written in sorted
order. Drift is decided on ``tree_sha256``, which does not depend on the zlib build.

:func:`tokenizer_manifest` records the tokenizer a run's token ids were produced with: for
tiktoken encodings the library version, encoding name, vocabulary size and a SHA-256 of the
mergeable ranks (plus special tokens); for the word-level corpora the SHA-256 of the vocabulary
JSON sidecar. Everything here is best-effort bookkeeping: failures are returned as recorded
errors and never raise into training or finalization.
"""

import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ARCHIVE_NAME:     str = "source_archive.zip"        # written when the run directory is created
ARCHIVE_END_NAME: str = "source_archive_end.zip"    # written at finalization only on drift
_ZIP_DATE_TIME:   Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)   # zip epoch: deterministic bytes
_MAX_DIRTY_FILES: int = 2000                         # cap on the recorded dirty-path list
_GIT_TIMEOUT_S:   float = 10.0


def repository_root() -> Path:
    r"""The directory holding the ``vfe3`` package (the repository root the drivers live in)."""
    return Path(__file__).resolve().parent.parent


def source_files(
    root: Optional[Path] = None,
) -> List[Path]:
    r"""Sorted absolute paths of the archived sources: ``vfe3/**/*.py``, root ``*.py``, ``pyproject.toml``."""
    repo = repository_root() if root is None else Path(root).resolve()
    package = repo / "vfe3"
    files = {path for path in package.rglob("*.py") if "__pycache__" not in path.parts}
    files |= {path for path in repo.glob("*.py") if path.is_file()}
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        files.add(pyproject)
    return sorted(files, key=lambda path: path.relative_to(repo).as_posix())


def _read_tree(
    root: Path,
) -> List[Tuple[str, bytes]]:
    """Read every archived source once as ``(relative POSIX path, raw bytes)`` in sorted order."""
    return [(path.relative_to(root).as_posix(), path.read_bytes()) for path in source_files(root)]


def _tree_sha256(
    entries: Sequence[Tuple[str, bytes]],
) -> str:
    r"""Length-prefixed content hash of ``(path, bytes)`` pairs (see the module docstring)."""
    digest = hashlib.sha256()
    for relative, content in entries:
        encoded = relative.encode("utf-8")
        for payload in (encoded, content):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def source_tree_sha256(
    root: Optional[Path] = None,
) -> Tuple[str, int]:
    r"""Return ``(tree_sha256, n_files)`` for the current on-disk source tree."""
    repo = repository_root() if root is None else Path(root).resolve()
    entries = _read_tree(repo)
    return _tree_sha256(entries), len(entries)


def _sha256_file(
    path: Path,
) -> str:
    r"""Stream ``path`` in fixed 1 MiB blocks and return its SHA-256 hex digest.

    Reads the file in bounded blocks (never materializing the whole corpus in memory) so hashing a
    multi-hundred-megabyte tokenized cache stays within a fixed working set. The one copy, shared by
    :mod:`vfe3.data.datasets` (audit 2026-10-17 item 85).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head_and_dirty_files(
    root: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Return ``{git_head, dirty_files[, dirty_files_truncated][, git_error]}`` from bounded probes.

    ``dirty_files`` lists every path ``git status --porcelain=v1 --untracked-files=all`` reports
    (modified, staged, deleted, renamed and untracked), capped at ``_MAX_DIRTY_FILES`` entries.
    """
    repo = repository_root() if root is None else Path(root).resolve()
    record: Dict[str, object] = {"git_head": None, "dirty_files": None}
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise FileNotFoundError("git executable was not found on PATH")
        from vfe3.run_artifacts import _git_environment     # function-local: run_artifacts imports this module
        env = _git_environment(git_executable)

        def _git(*args: str) -> bytes:
            return subprocess.check_output(
                [git_executable, "-c", "core.fsmonitor=false",
                 "-c", f"safe.directory={repo.as_posix()}", *args],
                cwd=str(repo), env=env, stderr=subprocess.DEVNULL, timeout=_GIT_TIMEOUT_S)

        record["git_head"] = _git("rev-parse", "HEAD").decode("ascii").strip()
        status = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        fields = status.split(b"\0")
        dirty: List[str] = []
        index = 0
        while index < len(fields):
            entry = fields[index]
            index += 1
            if len(entry) < 4:
                continue
            code, path = entry[:2], entry[3:]
            dirty.append(os.fsdecode(path))
            if code[:1] in (b"R", b"C"):          # rename/copy: the next field is the source path
                index += 1
        record["dirty_files"] = dirty[:_MAX_DIRTY_FILES]
        if len(dirty) > _MAX_DIRTY_FILES:
            record["dirty_files_truncated"] = len(dirty)
    except Exception as exc:                      # noqa: BLE001 - provenance is best-effort
        record["git_head"] = None
        record["dirty_files"] = None
        record["git_error"] = repr(exc)
    return record


def write_source_archive(
    run_dir: 'str | Path',

    *,
    name:     str            = ARCHIVE_NAME,
    root:     Optional[Path] = None,
    git_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    r"""Zip the source tree into ``run_dir/name`` and return its provenance record.

    The record carries ``path`` (relative to ``run_dir``), ``sha256`` (of the zip bytes),
    ``tree_sha256`` (content hash, used for drift), ``n_files``, ``n_bytes`` (uncompressed),
    ``git_head`` and ``dirty_files``. The zip is published atomically (temporary sibling plus
    ``os.replace``). Any failure returns ``{"available": False, "error": ...}`` instead of raising.
    """
    run_dir = Path(run_dir)
    repo = repository_root() if root is None else Path(root).resolve()
    final = run_dir / name
    tmp = run_dir / f".{name}.{os.getpid()}.tmp"
    try:
        entries = _read_tree(repo)
        if not entries:
            raise RuntimeError(f"no source files found under {repo}")
        run_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, content in entries:
                info = zipfile.ZipInfo(relative, date_time=_ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, content)
        os.replace(tmp, final)
        record: Dict[str, object] = {
            "available":   True,
            "path":        name,
            "sha256":      _sha256_file(final),
            "tree_sha256": _tree_sha256(entries),
            "n_files":     len(entries),
            "n_bytes":     int(sum(len(content) for _, content in entries)),
        }
        record.update(git_head_and_dirty_files(repo) if git_info is None else dict(git_info))
        return record
    except Exception as exc:                      # noqa: BLE001 - never fail run construction
        try:
            tmp.unlink()
        except OSError:
            pass
        return {"available": False, "path": None, "error": repr(exc)}


def verify_source_archive(
    run_dir:      'str | Path',
    start_record: Optional[Dict[str, object]],

    *,
    root: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Re-hash the source tree at finalization and archive it again when it drifted.

    Returns ``{"tree_sha256_end", "drift", "source_archive_end"}``. ``drift`` is None when the
    start record is unavailable (nothing to compare), otherwise ``tree_sha256_end != tree_sha256``.
    On drift ``source_archive_end`` is the :func:`write_source_archive` record of
    ``ARCHIVE_END_NAME``; without drift it is None.
    """
    repo = repository_root() if root is None else Path(root).resolve()
    try:
        tree_end, n_files = source_tree_sha256(repo)
    except Exception as exc:                      # noqa: BLE001
        return {"tree_sha256_end": None, "drift": None, "source_archive_end": None,
                "error": repr(exc)}
    start_hash = (start_record or {}).get("tree_sha256")
    drift = None if not isinstance(start_hash, str) else tree_end != start_hash
    end_record = (write_source_archive(run_dir, name=ARCHIVE_END_NAME, root=repo)
                  if drift else None)
    return {"tree_sha256_end": tree_end, "n_files_end": n_files, "drift": drift,
            "source_archive_end": end_record}


def _tiktoken_ranks_sha256(
    encoding: object,
) -> Tuple[str, str]:
    r"""SHA-256 over the encoding's token table in rank order; returns ``(digest, method)``.

    Preferred: the mergeable ranks ``(bytes, rank)`` sorted by rank, then the special tokens
    ``(name, id)`` sorted by id, each length-prefixed. Fallback when the private table is absent:
    ``decode_single_token_bytes(i)`` for every id ``i < n_vocab`` (ids that do not decode hash as
    an empty payload plus a marker byte).
    """
    digest = hashlib.sha256()
    ranks = getattr(encoding, "_mergeable_ranks", None)
    if isinstance(ranks, dict) and ranks:
        for token_bytes, rank in sorted(ranks.items(), key=lambda item: item[1]):
            digest.update(len(token_bytes).to_bytes(4, "big"))
            digest.update(bytes(token_bytes))
            digest.update(int(rank).to_bytes(8, "big"))
        specials = getattr(encoding, "_special_tokens", None) or {}
        for text, rank in sorted(specials.items(), key=lambda item: item[1]):
            payload = str(text).encode("utf-8")
            digest.update(b"special")
            digest.update(len(payload).to_bytes(4, "big"))
            digest.update(payload)
            digest.update(int(rank).to_bytes(8, "big"))
        return digest.hexdigest(), "mergeable_ranks_sorted_by_rank_plus_special_tokens"
    for token_id in range(int(encoding.n_vocab)):
        try:
            payload = bytes(encoding.decode_single_token_bytes(token_id))
            digest.update(b"\x01")
        except Exception:                         # noqa: BLE001 - undecodable id
            payload = b""
            digest.update(b"\x00")
        digest.update(len(payload).to_bytes(4, "big"))
        digest.update(payload)
    return digest.hexdigest(), "decode_single_token_bytes_by_id"


def tokenizer_manifest(
    tokenizer_tag: Optional[str],

    *,
    dataset:   Optional[str]  = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Describe the tokenizer behind ``tokenizer_tag`` (``tiktoken`` / ``tiktoken_cl100k`` / ``word``).

    Keys: ``kind``, ``encoding``, ``library_version``, ``vocab_size`` and ``ranks_sha256`` (BPE) or
    ``vocab_sha256`` plus ``vocab_path`` (word level). An unknown tag, a missing library or a missing
    vocabulary sidecar yields ``status='unavailable'`` with a reason; nothing raises.
    """
    record: Dict[str, object] = {"kind": tokenizer_tag, "encoding": None, "library_version": None,
                                 "vocab_size": None, "status": "unavailable"}
    try:
        if tokenizer_tag in ("tiktoken", "tiktoken_cl100k"):
            import tiktoken
            name = "gpt2" if tokenizer_tag == "tiktoken" else "cl100k_base"
            encoding = tiktoken.get_encoding(name)
            digest, method = _tiktoken_ranks_sha256(encoding)
            record.update(kind="tiktoken", encoding=name,
                          library_version=getattr(tiktoken, "__version__", None),
                          vocab_size=int(encoding.n_vocab), ranks_sha256=digest,
                          ranks_hash_method=method, status="recorded")
        elif tokenizer_tag == "word":
            if dataset is None:
                raise ValueError("a word-level manifest needs the dataset name")
            from vfe3.data.datasets import tokenizer_vocab_size
            from vfe3.data.word_vocab import vocabulary_path
            path = vocabulary_path(dataset, tag="word", cache_dir=cache_dir)
            if not path.is_file():
                raise FileNotFoundError(f"word vocabulary sidecar not found at {path}")
            record.update(kind="word", encoding="word", vocab_sha256=_sha256_file(path),
                          vocab_path=str(path),
                          vocab_size=int(tokenizer_vocab_size(dataset, cache_dir=cache_dir)),
                          status="recorded")
        else:
            record["reason"] = f"no tokenizer manifest for tokenizer_tag={tokenizer_tag!r}"
    except Exception as exc:                      # noqa: BLE001 - provenance is best-effort
        record["status"] = "unavailable"
        record["reason"] = repr(exc)
    return record
