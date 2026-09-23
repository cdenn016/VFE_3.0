"""Deterministic filesystem-safe names for user- and experiment-supplied labels."""

import hashlib
import os
import re
import stat
import unicodedata
from pathlib import Path


_WINDOWS_RESERVED_PATH_STEMS = {
    "aux",
    "clock$",
    "con",
    "conin$",
    "conout$",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
    *(f"com{index}" for index in "¹²³"),
    *(f"lpt{index}" for index in "¹²³"),
}


def portable_path_component_key(
    value: str,

    *,
    field: str = "path component",
) -> str:
    r"""Validate one portable regular-file name and return its alias-comparison key.

    The validation is deliberately independent of the host Python version and operating system.
    In particular, it implements the Windows device-name rules that ``os.path.isreserved`` only
    exposes on newer Python versions, so manifests written on Linux cannot later escape or alias
    a cell directory when consumed on Windows.
    """
    invalid_windows_chars = '<>:"/\\|?*'
    if type(value) is str:
        try:
            utf8_bytes = len(value.encode("utf-8"))
            utf16_units = len(value.encode("utf-16-le")) // 2
        except UnicodeError:
            utf8_bytes = utf16_units = 256
    else:
        utf8_bytes = utf16_units = 256
    invalid = (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or len(value) > 255
        or utf8_bytes > 255
        or utf16_units > 255
        or any(ord(char) < 32 for char in value)
        or any(char in invalid_windows_chars for char in value)
        or value[-1] in {" ", "."}
    )
    if not invalid:
        stem = value.split(".", maxsplit=1)[0].rstrip(" ").casefold()
        invalid = stem in _WINDOWS_RESERVED_PATH_STEMS
    if invalid:
        raise ValueError(f"{field} must be a safe single path component, got {value!r}")
    return unicodedata.normalize("NFC", value).casefold()


def path_is_reparse_point(path: Path) -> bool:
    """Return whether ``path`` is a symlink, junction, or other Windows reparse point."""
    path = Path(path)
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None and isjunction(path):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(reparse_flag and attributes & reparse_flag)


def prepare_owned_output_child(
    root: Path,
    name: str,

    *,
    role: str,
) -> Path:
    """Create or validate one real direct-child output directory under ``root``."""
    portable_path_component_key(name, field=f"{role} directory name")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"{role} output root is not a directory: {root}")

    child = root / name
    if path_is_reparse_point(child):
        raise ValueError(f"{role} directory may not be a symlink, junction, or reparse point")
    if child.exists() and not child.is_dir():
        raise ValueError(f"{role} path exists but is not a directory: {child}")
    child.mkdir(parents=False, exist_ok=True)
    if path_is_reparse_point(child):
        raise ValueError(f"{role} directory became a symlink, junction, or reparse point")
    if not child.is_dir() or child.resolve().parent != root.resolve():
        raise ValueError(f"{role} directory resolves outside its output root")
    return child


def filesystem_slug(
    value: str,

    *,
    fallback: str = "artifact",
) -> str:
    r"""Return one safe path component plus a stable collision-disambiguating hash.

    The readable prefix uses one allowlist across every artifact producer. The suffix binds the
    original value, so distinct inputs remain distinct even when their readable prefixes collide.
    """
    if not isinstance(value, str):
        raise TypeError(f"filesystem_slug value must be str, got {type(value).__name__}")
    if not isinstance(fallback, str) or not fallback:
        raise ValueError("filesystem_slug fallback must be a nonempty string")
    safe_fallback = re.sub(r"[^A-Za-z0-9._-]", "_", fallback).strip("._")
    if not safe_fallback:
        raise ValueError("filesystem_slug fallback must contain a filesystem-safe character")
    prefix = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._") or safe_fallback
    prefix = prefix[:120]
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    slug = f"{prefix}__{suffix}"
    try:
        portable_path_component_key(slug, field="filesystem slug")
    except ValueError:
        slug = f"_{prefix[:119]}__{suffix}"
        portable_path_component_key(slug, field="filesystem slug")
    return slug
