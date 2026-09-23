"""Install the canonical skill into one or three managed global mirrors.

The transaction is failure-atomic for cooperating installers. It is not
observation-atomic or crash-atomic across the three independent filesystems.
It also does not preserve ACLs, permissions, or empty-directory identity.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


TARGET_NAME = "rigorous-theory-search"
_CACHE_DIRECTORIES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class InstallError(RuntimeError):
    """Raised when installation cannot proceed without violating safety."""


class _CommittedCleanupError(InstallError):
    """Raised after all live mirrors commit but backup cleanup fails."""


@dataclass(frozen=True)
class _Mirror:
    label: str
    home: Path
    skills: Path
    target: Path
    canonical_target: str


@dataclass(frozen=True)
class _FilesystemIdentity:
    device: int
    inode: int
    file_type: int


@dataclass(frozen=True)
class _TargetPrecondition:
    existed: bool
    identity: _FilesystemIdentity | None
    root_stamp: tuple[int, int, int, int, int] | None
    manifest: dict[str, dict[str, object]] | None


@dataclass
class _TransactionMirror:
    mirror: _Mirror
    precondition: _TargetPrecondition | None = None
    stage: Path | None = None
    stage_identity: _FilesystemIdentity | None = None
    backup: Path | None = None
    backup_identity: _FilesystemIdentity | None = None
    promoted: bool = False
    promoted_identity: _FilesystemIdentity | None = None


@dataclass
class _OwnedLock:
    path: Path
    token: bytes
    identity: _FilesystemIdentity
    complete: bool = False


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stat_stamp(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _filesystem_identity(value: os.stat_result) -> _FilesystemIdentity:
    return _FilesystemIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        file_type=stat.S_IFMT(value.st_mode),
    )


def _unsafe_entry_reason(path: Path, value: os.stat_result) -> str | None:
    if stat.S_ISLNK(value.st_mode):
        return "symlink"
    if getattr(value, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE:
        return "reparse point"
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None and is_junction(path):
        return "junction"
    return None


def _check_safe_entry(path: Path, value: os.stat_result, label: str) -> None:
    reason = _unsafe_entry_reason(path, value)
    if reason is not None:
        raise InstallError(f"unsafe {label} at {path}: {reason}")


def _identity_at(path: Path, label: str) -> _FilesystemIdentity:
    if not _lexists(path):
        raise InstallError(f"missing {label}: {path}")
    try:
        value = path.lstat()
    except OSError as exc:
        raise InstallError(f"cannot inspect {label} {path}: {exc}") from exc
    _check_safe_entry(path, value, label)
    return _filesystem_identity(value)


def _require_identity(
    path: Path,
    expected: _FilesystemIdentity,
    label: str,
) -> None:
    actual = _identity_at(path, label)
    if actual != expected:
        raise InstallError(f"{label} identity changed and was preserved: {path}")


def _inspect_existing_components(path: Path, label: str) -> Path:
    absolute = _absolute(path)
    parts = absolute.parts
    if not parts:
        raise InstallError(f"invalid empty {label} path")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        if not _lexists(current):
            break
        try:
            value = current.lstat()
        except OSError as exc:
            raise InstallError(f"cannot inspect {label} component {current}: {exc}") from exc
        _check_safe_entry(current, value, f"{label} component")
        if index < len(parts) - 1 and not stat.S_ISDIR(value.st_mode):
            raise InstallError(f"{label} component is not a directory: {current}")
    return absolute


def _require_directory(path: Path, label: str) -> os.stat_result:
    absolute = _inspect_existing_components(path, label)
    if not _lexists(absolute):
        raise InstallError(f"missing {label}: {absolute}")
    try:
        value = absolute.lstat()
    except OSError as exc:
        raise InstallError(f"cannot inspect {label} {absolute}: {exc}") from exc
    _check_safe_entry(absolute, value, label)
    if not stat.S_ISDIR(value.st_mode):
        raise InstallError(f"{label} is not a directory: {absolute}")
    return value


def _validate_ignored_tree(root: Path) -> None:
    try:
        entries = sorted(os.scandir(root), key=lambda item: item.name)
    except OSError as exc:
        raise InstallError(f"cannot scan ignored cache {root}: {exc}") from exc
    for entry in entries:
        path = root / entry.name
        try:
            value = path.lstat()
        except OSError as exc:
            raise InstallError(f"cannot inspect cache entry {path}: {exc}") from exc
        _check_safe_entry(path, value, "cache entry")
        if stat.S_ISDIR(value.st_mode):
            _validate_ignored_tree(path)
        elif not stat.S_ISREG(value.st_mode):
            raise InstallError(f"unsafe special cache entry at {path}")


def _hash_regular_file(path: Path, before: os.stat_result) -> dict[str, object]:
    _check_safe_entry(path, before, "managed file")
    if not stat.S_ISREG(before.st_mode):
        raise InstallError(f"managed entry is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _stat_stamp(opened) != _stat_stamp(before):
                raise InstallError(f"managed file changed while opening: {path}")
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except InstallError:
        raise
    except OSError as exc:
        raise InstallError(f"cannot hash managed file {path}: {exc}") from exc
    try:
        after = path.lstat()
    except OSError as exc:
        raise InstallError(f"managed file changed while hashing: {path}: {exc}") from exc
    if _stat_stamp(after) != _stat_stamp(before) or size != before.st_size:
        raise InstallError(f"managed file changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size": size}


def _sorted_tree_entries(directory: Path) -> list[os.DirEntry[str]]:
    try:
        return sorted(os.scandir(directory), key=lambda item: item.name)
    except OSError as exc:
        raise InstallError(f"cannot scan tree directory {directory}: {exc}") from exc


def _confirm_file_unchanged(path: Path, before: os.stat_result) -> None:
    try:
        after = path.lstat()
    except OSError as exc:
        raise InstallError(f"managed file changed while hashing: {path}: {exc}") from exc
    if _stat_stamp(after) != _stat_stamp(before):
        raise InstallError(f"managed file changed while hashing: {path}")


def _scan_tree_entry(
    path: Path,
    root: Path,
    manifest: dict[str, dict[str, object]],
) -> None:
    try:
        before = path.lstat()
    except OSError as exc:
        raise InstallError(f"cannot inspect tree entry {path}: {exc}") from exc
    _check_safe_entry(path, before, "tree entry")
    if stat.S_ISDIR(before.st_mode):
        if path.name in _CACHE_DIRECTORIES:
            _validate_ignored_tree(path)
        else:
            _visit_tree(path, root, manifest)
        return
    if not stat.S_ISREG(before.st_mode):
        raise InstallError(f"unsafe special tree entry at {path}")
    if path.suffix.lower() in _CACHE_SUFFIXES:
        return
    relative = path.relative_to(root).as_posix()
    manifest[relative] = _hash_regular_file(path, before)
    _confirm_file_unchanged(path, before)


def _visit_tree(
    directory: Path,
    root: Path,
    manifest: dict[str, dict[str, object]],
) -> None:
    for entry in _sorted_tree_entries(directory):
        _scan_tree_entry(directory / entry.name, root, manifest)


def _scan_tree(root: Path, *, require_skill: bool) -> dict[str, dict[str, object]]:
    absolute = _absolute(root)
    _require_directory(absolute, "tree root")
    manifest: dict[str, dict[str, object]] = {}
    _visit_tree(absolute, absolute, manifest)
    ordered = dict(sorted(manifest.items()))
    if require_skill and "SKILL.md" not in ordered:
        raise InstallError(f"tree root lacks a real SKILL.md: {absolute}")
    return ordered


def _snapshot_target(target: Path) -> _TargetPrecondition:
    if not _lexists(target):
        return _TargetPrecondition(
            existed=False,
            identity=None,
            root_stamp=None,
            manifest=None,
        )
    try:
        before = target.lstat()
    except OSError as exc:
        raise InstallError(f"cannot inspect managed target {target}: {exc}") from exc
    _check_safe_entry(target, before, "managed target")
    if not stat.S_ISDIR(before.st_mode):
        raise InstallError(f"managed target is not a directory: {target}")
    manifest = _scan_tree(target, require_skill=False)
    try:
        after = target.lstat()
    except OSError as exc:
        raise InstallError(f"managed target changed while snapshotting: {target}: {exc}") from exc
    if _stat_stamp(after) != _stat_stamp(before):
        raise InstallError(f"managed target changed while snapshotting: {target}")
    return _TargetPrecondition(
        existed=True,
        identity=_filesystem_identity(before),
        root_stamp=_stat_stamp(before),
        manifest=manifest,
    )


def _record_target_preconditions(states: list[_TransactionMirror]) -> None:
    for state in states:
        state.precondition = _snapshot_target(state.mirror.target)


def _assert_target_precondition(state: _TransactionMirror) -> None:
    if state.precondition is None:
        raise InstallError("internal error: target precondition was not recorded")
    current = _snapshot_target(state.mirror.target)
    if current != state.precondition:
        raise InstallError(
            f"target changed during staging and was preserved: {state.mirror.target}"
        )


def _recheck_target_preconditions(states: list[_TransactionMirror]) -> None:
    for state in states:
        _assert_target_precondition(state)


def build_manifest(source: Path) -> dict[str, dict[str, object]]:
    """Return the sorted size and SHA-256 manifest for one real skill tree."""

    return _scan_tree(Path(source), require_skill=True)


def _canonical_key(path: Path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _paths_overlap(first: str, second: str) -> bool:
    try:
        common = os.path.normcase(os.path.commonpath((first, second)))
    except ValueError:
        return False
    return common == first or common == second


def _validate_optional_directory(path: Path, label: str) -> None:
    absolute = _inspect_existing_components(path, label)
    if not _lexists(absolute):
        return
    value = absolute.lstat()
    _check_safe_entry(absolute, value, label)
    if not stat.S_ISDIR(value.st_mode):
        raise InstallError(f"{label} is not a directory: {absolute}")


def _validate_mirror_layout(
    source_root: Path,
    supplied_homes: Sequence[tuple[str, Path]],
) -> tuple[_Mirror, ...]:
    source = _inspect_existing_components(Path(source_root), "source root")
    source_key = _canonical_key(source)
    mirrors: list[_Mirror] = []
    for label, supplied_home in supplied_homes:
        home = _absolute(Path(supplied_home))
        _validate_optional_directory(home, f"{label} home")
        skills = home / "skills"
        _validate_optional_directory(skills, f"{label} skills directory")
        canonical_skills = _canonical_key(skills)
        canonical_target = os.path.normcase(
            os.path.join(canonical_skills, TARGET_NAME)
        )
        if os.path.dirname(canonical_target) != canonical_skills:
            raise InstallError(
                f"{label} target escapes its skills directory: {canonical_target}"
            )
        mirrors.append(
            _Mirror(
                label=label,
                home=home,
                skills=skills,
                target=skills / TARGET_NAME,
                canonical_target=canonical_target,
            )
        )

    targets = [mirror.canonical_target for mirror in mirrors]
    if len(set(targets)) != len(targets):
        raise InstallError("duplicate canonical installation targets")
    for index, first in enumerate(mirrors):
        if _paths_overlap(source_key, first.canonical_target):
            raise InstallError(
                f"source/destination overlap: {source} and {first.target}"
            )
        for second in mirrors[index + 1 :]:
            if _paths_overlap(first.canonical_target, second.canonical_target):
                raise InstallError(
                    "installation target overlap: "
                    f"{first.target} and {second.target}"
                )
    return tuple(mirrors)


def _validate_layout(
    source_root: Path,
    agents_home: Path,
    claude_home: Path,
    codex_home: Path,
) -> tuple[_Mirror, ...]:
    return _validate_mirror_layout(
        source_root,
        (
            ("agents", agents_home),
            ("claude", claude_home),
            ("codex", codex_home),
        ),
    )


def _validate_canonical_layout(
    source_root: Path,
    agents_home: Path,
) -> tuple[_Mirror, ...]:
    return _validate_mirror_layout(
        source_root,
        (("agents", agents_home),),
    )


def _preflight_existing_targets(mirrors: tuple[_Mirror, ...]) -> None:
    for mirror in mirrors:
        if _lexists(mirror.target):
            _scan_tree(mirror.target, require_skill=False)


def _manifest_diagnostics(
    label: str,
    target: Path,
    expected: dict[str, dict[str, object]],
) -> list[str]:
    if not _lexists(target):
        return [f"{label}: missing target: {target}"]
    try:
        actual = _scan_tree(target, require_skill=False)
    except InstallError as exc:
        return [f"{label}: invalid or unsafe target: {exc}"]
    diagnostics: list[str] = []
    for relative in sorted(expected):
        if relative not in actual:
            diagnostics.append(f"{label}: missing managed file: {relative}")
        elif actual[relative] != expected[relative]:
            diagnostics.append(f"{label}: content mismatch: {relative}")
    for relative in sorted(set(actual) - set(expected)):
        diagnostics.append(f"{label}: stale managed file: {relative}")
    return diagnostics


def _diagnostics(
    mirrors: tuple[_Mirror, ...],
    expected: dict[str, dict[str, object]],
) -> list[str]:
    diagnostics: list[str] = []
    for mirror in mirrors:
        diagnostics.extend(
            _manifest_diagnostics(mirror.label, mirror.target, expected)
        )
    return sorted(diagnostics)


def verify_install(
    source_root: Path,
    agents_home: Path,
    claude_home: Path,
    codex_home: Path,
) -> list[str]:
    """Return sorted mirror drift diagnostics without mutating any path."""

    expected = build_manifest(Path(source_root))
    mirrors = _validate_layout(
        Path(source_root),
        Path(agents_home),
        Path(claude_home),
        Path(codex_home),
    )
    return _diagnostics(mirrors, expected)


def verify_canonical_install(
    source_root: Path,
    agents_home: Path,
) -> list[str]:
    """Return canonical agents-mirror drift without mutating any path."""

    expected = build_manifest(Path(source_root))
    mirrors = _validate_canonical_layout(
        Path(source_root),
        Path(agents_home),
    )
    return _diagnostics(mirrors, expected)


def _mkdir_owned(path: Path, label: str, created: list[Path]) -> None:
    if _lexists(path):
        _validate_optional_directory(path, label)
        return
    parent = path.parent
    _require_directory(parent, f"parent of {label}")
    try:
        path.mkdir()
    except FileExistsError:
        _validate_optional_directory(path, label)
    except OSError as exc:
        raise InstallError(f"cannot create {label} {path}: {exc}") from exc
    else:
        created.append(path)


def _ensure_skills_directories(
    mirrors: tuple[_Mirror, ...],
    created: list[Path],
) -> None:
    for mirror in mirrors:
        _mkdir_owned(mirror.home, f"{mirror.label} home", created)
        _mkdir_owned(
            mirror.skills,
            f"{mirror.label} skills directory",
            created,
        )


def _lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.install.lock")


def _stage_path(target: Path, nonce: str) -> Path:
    return target.with_name(f".{target.name}.stage-{nonce}")


def _backup_path(target: Path, nonce: str) -> Path:
    return target.with_name(f".{target.name}.backup-{nonce}")


def _release_locks(locks: list[_OwnedLock]) -> list[str]:
    errors: list[str] = []
    for owned in reversed(locks):
        try:
            if not _lexists(owned.path):
                errors.append(f"owned lock disappeared: {owned.path}")
                continue
            value = owned.path.lstat()
            _check_safe_entry(owned.path, value, "owned lock")
            if not stat.S_ISREG(value.st_mode):
                raise InstallError(f"owned lock is not regular: {owned.path}")
            if _filesystem_identity(value) != owned.identity:
                raise InstallError(f"owned lock identity changed: {owned.path}")
            if owned.complete and owned.path.read_bytes() != owned.token:
                raise InstallError(f"owned lock content changed: {owned.path}")
            owned.path.unlink()
        except (InstallError, OSError) as exc:
            errors.append(str(exc))
    return sorted(errors)


def _write_lock_token(descriptor: int, token: bytes) -> None:
    written = 0
    while written < len(token):
        count = os.write(descriptor, token[written:])
        if count <= 0:
            raise InstallError("installation lock write made no progress")
        written += count


def _create_owned_lock(
    lock: Path,
    token: bytes,
    locks: list[_OwnedLock],
) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise InstallError(f"installation lock already exists: {lock}") from exc
    except OSError as exc:
        raise InstallError(f"cannot acquire installation lock {lock}: {exc}") from exc
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise InstallError(f"installation lock is not regular: {lock}")
        owned = _OwnedLock(
            path=lock,
            token=token,
            identity=_filesystem_identity(value),
        )
        locks.append(owned)
        _write_lock_token(descriptor, token)
        owned.complete = True
    finally:
        os.close(descriptor)


def _acquire_locks(
    mirrors: tuple[_Mirror, ...],
    nonce: str,
) -> list[_OwnedLock]:
    locks: list[_OwnedLock] = []
    token = f"rigorous-theory-search:{nonce}\n".encode()
    try:
        for mirror in sorted(mirrors, key=lambda item: item.canonical_target):
            lock = _lock_path(mirror.target)
            if _lexists(lock):
                raise InstallError(f"installation lock already exists: {lock}")
            _create_owned_lock(lock, token, locks)
    except BaseException as exc:
        cleanup_errors = _release_locks(locks)
        if not isinstance(exc, Exception):
            _add_interruption_cleanup_note(exc, cleanup_errors)
            raise
        detail = str(exc)
        if cleanup_errors:
            detail += "; lock cleanup errors: " + "; ".join(cleanup_errors)
        raise InstallError(detail) from exc
    return locks


def _copy_managed_file(
    source: Path,
    destination: Path,
    expected: dict[str, object],
) -> None:
    try:
        before = source.lstat()
    except OSError as exc:
        raise InstallError(f"source changed before copy: {source}: {exc}") from exc
    if _hash_regular_file(source, before) != expected:
        raise InstallError(f"source changed before copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as read_handle, destination.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
    except OSError as exc:
        raise InstallError(f"cannot copy {source} to {destination}: {exc}") from exc
    try:
        after = source.lstat()
    except OSError as exc:
        raise InstallError(f"source changed during copy: {source}: {exc}") from exc
    if _stat_stamp(after) != _stat_stamp(before):
        raise InstallError(f"source changed during copy: {source}")


def _create_stage(
    source_root: Path,
    target: Path,
    manifest: dict[str, dict[str, object]],
    nonce: str,
) -> Path:
    stage = _stage_path(target, nonce)
    if _lexists(stage):
        raise InstallError(f"staging path already exists: {stage}")
    _require_directory(target.parent, "staging parent")
    try:
        stage.mkdir()
        for relative in sorted(manifest):
            source = _absolute(source_root) / Path(relative)
            destination = stage.joinpath(*relative.split("/"))
            _copy_managed_file(source, destination, manifest[relative])
        actual = build_manifest(stage)
        if actual != manifest:
            raise InstallError(f"staging manifest mismatch: {stage}")
    except BaseException as exc:
        cleanup_errors: list[str] = []
        try:
            _remove_owned_tree(stage)
        except (InstallError, OSError) as cleanup_exc:
            cleanup_errors.append(str(cleanup_exc))
        if not isinstance(exc, Exception):
            _add_interruption_cleanup_note(exc, cleanup_errors)
            raise
        detail = str(exc)
        if cleanup_errors:
            detail += "; staging cleanup errors: " + "; ".join(cleanup_errors)
        raise InstallError(detail) from exc
    return stage


def _move_target_to_backup(
    target: Path,
    backup: Path,
    expected_identity: _FilesystemIdentity,
) -> _FilesystemIdentity:
    _require_identity(target, expected_identity, "managed target")
    if _lexists(backup):
        raise InstallError(f"backup path already exists: {backup}")
    try:
        os.rename(target, backup)
    except OSError as exc:
        raise InstallError(f"cannot move target {target} to backup {backup}: {exc}") from exc
    _require_identity(backup, expected_identity, "installation backup")
    return expected_identity


def _promote_stage(stage: Path, target: Path) -> _FilesystemIdentity:
    _require_directory(stage, "staging tree")
    if _lexists(target):
        raise InstallError(f"target appeared before promotion and was preserved: {target}")
    try:
        os.rename(stage, target)
    except OSError as exc:
        raise InstallError(f"cannot promote stage {stage} to {target}: {exc}") from exc
    return _identity_at(target, "promoted target")


def _demote_target(
    target: Path,
    stage: Path,
    expected_identity: _FilesystemIdentity,
) -> None:
    _require_identity(target, expected_identity, "promoted target")
    if _lexists(stage):
        raise InstallError(f"rollback staging path is occupied: {stage}")
    try:
        os.rename(target, stage)
    except OSError as exc:
        raise InstallError(f"cannot demote target {target} to {stage}: {exc}") from exc
    _require_identity(stage, expected_identity, "demoted staging tree")


def _restore_backup(
    backup: Path,
    target: Path,
    expected_identity: _FilesystemIdentity,
) -> None:
    _require_identity(backup, expected_identity, "installation backup")
    if _lexists(target):
        raise InstallError(
            f"concurrently created target preserved beside backup: {target}, {backup}"
        )
    try:
        os.rename(backup, target)
    except OSError as exc:
        raise InstallError(f"cannot restore backup {backup} to {target}: {exc}") from exc
    _require_identity(target, expected_identity, "restored target")


def _collect_removal_entries(root: Path) -> tuple[list[Path], list[Path]]:
    _require_directory(root, "transaction-owned tree")
    files: list[Path] = []
    directories: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise InstallError(f"cannot scan owned tree {directory}: {exc}") from exc
        for entry in entries:
            path = directory / entry.name
            value = path.lstat()
            _check_safe_entry(path, value, "transaction-owned entry")
            if stat.S_ISDIR(value.st_mode):
                visit(path)
                directories.append(path)
            elif stat.S_ISREG(value.st_mode):
                files.append(path)
            else:
                raise InstallError(f"unsafe special transaction-owned entry: {path}")

    visit(root)
    return files, directories


def _remove_owned_tree(root: Path) -> None:
    if not _lexists(root):
        return
    files, directories = _collect_removal_entries(root)
    for path in files:
        value = path.lstat()
        _check_safe_entry(path, value, "transaction-owned entry")
        if not stat.S_ISREG(value.st_mode):
            raise InstallError(f"owned file changed type before removal: {path}")
        path.unlink()
    for path in directories:
        value = path.lstat()
        _check_safe_entry(path, value, "transaction-owned directory")
        if not stat.S_ISDIR(value.st_mode):
            raise InstallError(f"owned directory changed type before removal: {path}")
        path.rmdir()
    root.rmdir()


def _assert_manifest(
    root: Path,
    expected: dict[str, dict[str, object]],
    label: str,
) -> None:
    if build_manifest(root) != expected:
        raise InstallError(f"{label} manifest mismatch: {root}")


def _require_backup_ownership(state: _TransactionMirror) -> None:
    if state.backup is None or state.backup_identity is None:
        raise InstallError("backup ownership was not established; backup preserved")
    _require_identity(state.backup, state.backup_identity, "installation backup")
    if state.precondition is None:
        raise InstallError("target precondition missing; backup preserved")
    if _snapshot_target(state.backup) != state.precondition:
        raise InstallError(f"installation backup changed and was preserved: {state.backup}")


def _require_promoted_ownership(
    state: _TransactionMirror,
    expected: dict[str, dict[str, object]],
) -> None:
    if state.promoted_identity is None:
        raise InstallError("promoted target ownership was not established; target preserved")
    _require_identity(
        state.mirror.target,
        state.promoted_identity,
        "promoted target",
    )
    _assert_manifest(state.mirror.target, expected, "promoted target")


def _require_stage_ownership(
    state: _TransactionMirror,
    expected: dict[str, dict[str, object]],
) -> None:
    if state.stage is None or state.stage_identity is None:
        raise InstallError("staging ownership was not established; stage preserved")
    _require_identity(state.stage, state.stage_identity, "staging tree")
    _assert_manifest(state.stage, expected, "staging tree")


def _rollback_promotions(
    states: list[_TransactionMirror],
    expected: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    for state in reversed(states):
        if not state.promoted or state.stage is None:
            continue
        try:
            _require_promoted_ownership(state, expected)
            if state.promoted_identity is None:
                raise InstallError("internal error: missing promoted target identity")
            _demote_target(
                state.mirror.target,
                state.stage,
                state.promoted_identity,
            )
            state.promoted = False
        except (InstallError, OSError) as exc:
            errors.append(str(exc))
    return errors


def _rollback_backups(states: list[_TransactionMirror]) -> list[str]:
    errors: list[str] = []
    for state in reversed(states):
        if state.backup is None or not _lexists(state.backup):
            continue
        try:
            _require_backup_ownership(state)
            if state.backup_identity is None:
                raise InstallError("internal error: missing backup identity")
            _restore_backup(
                state.backup,
                state.mirror.target,
                state.backup_identity,
            )
        except (InstallError, OSError) as exc:
            errors.append(str(exc))
    return errors


def _rollback_stages(
    states: list[_TransactionMirror],
    expected: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    for state in reversed(states):
        if state.stage is None or not _lexists(state.stage):
            continue
        try:
            _require_stage_ownership(state, expected)
            _remove_owned_tree(state.stage)
        except (InstallError, OSError) as exc:
            errors.append(str(exc))
    return errors


def _rollback(
    states: list[_TransactionMirror],
    expected: dict[str, dict[str, object]],
) -> list[str]:
    errors = _rollback_promotions(states, expected)
    errors.extend(_rollback_backups(states))
    errors.extend(_rollback_stages(states, expected))
    return sorted(errors)


def _remove_created_directories(created: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in reversed(created):
        try:
            if _lexists(path):
                value = path.lstat()
                _check_safe_entry(path, value, "transaction-created directory")
                if not stat.S_ISDIR(value.st_mode):
                    raise InstallError(
                        f"transaction-created path changed type: {path}"
                    )
                path.rmdir()
        except (InstallError, OSError) as exc:
            errors.append(str(exc))
    return sorted(errors)


def _raise_install_failure(
    original: Exception,
    rollback_errors: list[str],
) -> None:
    detail = str(original)
    if rollback_errors:
        detail += "; rollback errors: " + "; ".join(sorted(rollback_errors))
    raise InstallError(detail) from original


def _add_interruption_cleanup_note(
    original: BaseException,
    cleanup_errors: list[str],
    *,
    committed: bool = False,
) -> None:
    if not cleanup_errors and not committed:
        return
    if committed:
        note = "installation committed before interruption; installed targets were retained"
        if cleanup_errors:
            note += "; cleanup errors: " + "; ".join(sorted(cleanup_errors))
    else:
        note = "installation interrupted; cleanup errors: " + "; ".join(
            sorted(cleanup_errors)
        )
    add_note = getattr(original, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = [*getattr(original, "__notes__", ()), note]
    setattr(original, "__notes__", notes)


def _prepare_stages(
    states: list[_TransactionMirror],
    source: Path,
    expected: dict[str, dict[str, object]],
    nonce: str,
) -> None:
    for state in states:
        state.stage = _stage_path(state.mirror.target, nonce)
        created_stage = _create_stage(
            source,
            state.mirror.target,
            expected,
            nonce,
        )
        if created_stage != state.stage:
            raise InstallError(
                f"stage escaped its deterministic sibling path: {created_stage}"
            )
        state.stage_identity = _identity_at(created_stage, "staging tree")
    if build_manifest(source) != expected:
        raise InstallError("source changed after staging and before live mutation")


def _backup_targets(states: list[_TransactionMirror], nonce: str) -> None:
    for state in states:
        _assert_target_precondition(state)
        precondition = state.precondition
        if precondition is None:
            raise InstallError("internal error: target precondition was not recorded")
        if not precondition.existed:
            continue
        if precondition.identity is None:
            raise InstallError("internal error: existing target identity is missing")
        state.backup = _backup_path(state.mirror.target, nonce)
        _move_target_to_backup(
            state.mirror.target,
            state.backup,
            precondition.identity,
        )
        if _snapshot_target(state.backup) != precondition:
            raise InstallError(
                f"installation backup changed and was preserved: {state.backup}"
            )
        state.backup_identity = precondition.identity


def _promote_stages(
    states: list[_TransactionMirror],
    expected: dict[str, dict[str, object]],
) -> None:
    for state in states:
        _require_stage_ownership(state, expected)
        if state.stage is None or state.stage_identity is None:
            raise InstallError("internal error: missing staged tree ownership")
        state.promoted_identity = _promote_stage(
            state.stage,
            state.mirror.target,
        )
        state.promoted = True
        if state.promoted_identity != state.stage_identity:
            raise InstallError(
                f"promoted target identity mismatch: {state.mirror.target}"
            )
        _assert_manifest(state.mirror.target, expected, state.mirror.label)


def _verify_final_state(
    source: Path,
    mirrors: tuple[_Mirror, ...],
    expected: dict[str, dict[str, object]],
) -> None:
    if build_manifest(source) != expected:
        raise InstallError("source changed before final verification")
    diagnostics = _diagnostics(mirrors, expected)
    if diagnostics:
        raise InstallError("; ".join(diagnostics))


def _remove_backups_after_commit(states: list[_TransactionMirror]) -> None:
    try:
        for state in states:
            if state.backup is not None and _lexists(state.backup):
                _require_backup_ownership(state)
                _remove_owned_tree(state.backup)
    except Exception as exc:
        raise _CommittedCleanupError(str(exc)) from exc


def _execute_install_transaction(
    source: Path,
    mirrors: tuple[_Mirror, ...],
    states: list[_TransactionMirror],
    expected: dict[str, dict[str, object]],
    nonce: str,
) -> None:
    _prepare_stages(states, source, expected, nonce)
    _recheck_target_preconditions(states)
    _backup_targets(states, nonce)
    _promote_stages(states, expected)
    _verify_final_state(source, mirrors, expected)


def _raise_committed_cleanup(
    original: Exception,
    cleanup_errors: list[str],
) -> None:
    detail = f"installation committed, cleanup incomplete: {original}"
    if cleanup_errors:
        detail += "; cleanup errors: " + "; ".join(sorted(cleanup_errors))
    raise InstallError(detail) from original


def _run_install_transaction(
    source: Path,
    mirrors: tuple[_Mirror, ...],
    expected: dict[str, dict[str, object]],
) -> None:
    nonce = uuid.uuid4().hex
    states = [_TransactionMirror(mirror=mirror) for mirror in mirrors]
    created: list[Path] = []
    locks: list[_OwnedLock] = []
    committed = False
    try:
        _ensure_skills_directories(mirrors, created)
        locks = _acquire_locks(mirrors, nonce)
        _preflight_existing_targets(mirrors)
        _record_target_preconditions(states)
        _execute_install_transaction(source, mirrors, states, expected, nonce)
        committed = True
        _remove_backups_after_commit(states)
    except _CommittedCleanupError as exc:
        _raise_committed_cleanup(exc, _release_locks(locks))
    except Exception as exc:
        cleanup_errors = _rollback(states, expected)
        cleanup_errors.extend(_release_locks(locks))
        cleanup_errors.extend(_remove_created_directories(created))
        _raise_install_failure(exc, cleanup_errors)
    except BaseException as exc:
        if committed:
            cleanup_errors = _release_locks(locks)
            _add_interruption_cleanup_note(
                exc,
                cleanup_errors,
                committed=True,
            )
            raise
        cleanup_errors = _rollback(states, expected)
        cleanup_errors.extend(_release_locks(locks))
        cleanup_errors.extend(_remove_created_directories(created))
        _add_interruption_cleanup_note(exc, cleanup_errors)
        raise
    cleanup_errors = _release_locks(locks)
    if cleanup_errors:
        _raise_committed_cleanup(
            InstallError("owned lock release failed"),
            cleanup_errors,
        )


def install(
    source_root: Path,
    agents_home: Path,
    claude_home: Path,
    codex_home: Path,
) -> None:
    """Install one source tree into agents, Claude, and Codex homes."""

    source = _absolute(Path(source_root))
    expected = build_manifest(source)
    mirrors = _validate_layout(
        source,
        Path(agents_home),
        Path(claude_home),
        Path(codex_home),
    )
    _preflight_existing_targets(mirrors)
    if not _diagnostics(mirrors, expected):
        return
    _run_install_transaction(source, mirrors, expected)


def install_canonical(
    source_root: Path,
    agents_home: Path,
) -> None:
    """Install one source tree into the shared agents home."""

    source = _absolute(Path(source_root))
    expected = build_manifest(source)
    mirrors = _validate_canonical_layout(source, Path(agents_home))
    _preflight_existing_targets(mirrors)
    if not _diagnostics(mirrors, expected):
        return
    _run_install_transaction(source, mirrors, expected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install rigorous-theory-search into the shared agents home."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--agents-home", type=Path, required=True)
    parser.add_argument("--claude-home", type=Path)
    parser.add_argument("--codex-home", type=Path)
    parser.add_argument("--mirrored", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit-path installer CLI and return its process status."""

    arguments = _parser().parse_args(argv)
    try:
        runtime_homes = (arguments.claude_home, arguments.codex_home)
        if arguments.mirrored and any(home is None for home in runtime_homes):
            raise InstallError(
                "--mirrored requires --claude-home and --codex-home"
            )
        if not arguments.mirrored and any(
            home is not None for home in runtime_homes
        ):
            raise InstallError(
                "--claude-home and --codex-home require --mirrored"
            )
        mirrored = arguments.mirrored
        if arguments.check:
            if mirrored:
                diagnostics = verify_install(
                    arguments.source_root,
                    arguments.agents_home,
                    arguments.claude_home,
                    arguments.codex_home,
                )
            else:
                diagnostics = verify_canonical_install(
                    arguments.source_root,
                    arguments.agents_home,
                )
            for diagnostic in diagnostics:
                print(diagnostic, file=sys.stderr)
            return 1 if diagnostics else 0
        if mirrored:
            install(
                arguments.source_root,
                arguments.agents_home,
                arguments.claude_home,
                arguments.codex_home,
            )
        else:
            install_canonical(
                arguments.source_root,
                arguments.agents_home,
            )
    except (InstallError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
