r"""The scaling cell contract, shared by its writer (``scaling.py``) and its reader (``scaling_analysis.py``).

``scaling.run_cell`` publishes, beside each cell's ``summary.json``, a ``scaling_cell.json`` record that binds the
cell to its effective configuration, corpus and code. The record's canonical SHA-256 without its own digest field
(:func:`reuse_contract_sha256`) is stored in the record under :data:`REUSE_DIGEST_FIELD` and in ``summary.json``
under :data:`SUMMARY_DIGEST_FIELD`, and ``config_sha256`` is the digest of the serialized configuration
(:func:`config_sha256`). ``scaling._cell_resume_status`` reuses a cell, and ``scaling_analysis`` harvests one, only
through this binding, so the schema version, the field names, the split tuple and the validation rules are
declared here once; a change in one script alone made the other reject every cell (audit 2026-10-31, A356).

A cell written since October 31, 2026 also records the package code digest of the run
(:data:`PACKAGE_CODE_FIELD`, ``RunArtifacts.code_identity_sha256``, which ``provenance.json`` publishes as
``code_identity_sha256``), and its ``code_identity`` is the Git snapshot ``provenance.json`` publishes, so the two
records describe one snapshot (audit 2026-10-31, A050). Older cells carry neither binding and are read as before.
"""
import json
from collections.abc import Mapping
from typing import Any, Dict, Optional, Set

from vfe3.artifact_utils import canonical_json_sha256

SCALING_CELL_SCHEMA_VERSION = 2
REUSE_DIGEST_FIELD = "reuse_contract_sha256"
SUMMARY_DIGEST_FIELD = "scaling_reuse_contract_sha256"
PACKAGE_CODE_FIELD = "package_code_sha256"
SOURCE_SPLITS = ("train", "validation", "test")
# The Git fields of a code identity; provenance.json publishes the same three at its top level.
GIT_IDENTITY_FIELDS = ("git_sha", "git_dirty", "git_dirty_fingerprint")


def _detached_json(value: object) -> Any:
    """A strict-JSON round trip of ``value`` (no NaN or infinity), so the result is what a reader sees."""
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def is_sha256(value: object) -> bool:
    """Whether ``value`` is one lowercase hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def reuse_contract_sha256(cellmeta: Mapping[str, object]) -> str:
    """Digest a scaling-cell reuse contract without its self-referential digest field."""
    contract = dict(cellmeta)
    contract.pop(REUSE_DIGEST_FIELD, None)
    return canonical_json_sha256(contract)


def config_sha256(cfg: Any) -> str:
    r"""Digest of ``cfg``'s strict-JSON serialized form, the ``config_sha256`` a scaling cell records.

    ``serialize_config`` output passes through ``json.dumps(default=str)`` first, exactly as ``run_cell`` and
    ``_cell_resume_status`` have always done, so a non-finite ``kl_max`` is digested as its string form.
    """
    from vfe3.config import serialize_config

    return canonical_json_sha256(json.loads(json.dumps(serialize_config(cfg), default=str)))


def accepted_config_sha256(
    raw_saved_config: Mapping[str, object],    # config.json's "config" mapping as saved
    migrated:         Any,                     # the VFE3Config config_from_serialized built from it
) -> Set[str]:
    r"""The ``config_sha256`` values that bind a saved cell to its ``config.json``.

    A cell written before a schema migration (a retired field or value the loader migrates) hashed the saved form,
    so either the saved bytes or their migrated form may carry the digest; equality of the effective configs is
    checked separately by each caller.
    """
    return {config_sha256(migrated), canonical_json_sha256(dict(raw_saved_config))}


def digest_binding_failure(
    cellmeta:                Mapping[str, object],
    summary:                 Mapping[str, object],
    accepted_config_digests: Set[str],
) -> Optional[str]:
    """Why ``scaling_cell.json`` and ``summary.json`` are not one bound generation, or None when they are."""
    saved_digest = cellmeta.get(REUSE_DIGEST_FIELD)
    if (not isinstance(saved_digest, str)
            or len(saved_digest) != 64
            or summary.get(SUMMARY_DIGEST_FIELD) != saved_digest
            or cellmeta.get("config_sha256") not in accepted_config_digests):
        return "reuse digest or config_sha256 mismatch"
    return None


def validated_code_identity(value: object) -> Dict[str, object]:
    """Return one detached, usable Git identity (HEAD, dirty flag and dirty-tree fingerprint)."""
    if not isinstance(value, Mapping):
        raise TypeError("scaling code identity must be a mapping")
    detached = _detached_json(dict(value))
    git_sha = detached.get("git_sha")
    git_dirty = detached.get("git_dirty")
    fingerprint = detached.get("git_dirty_fingerprint")
    if not isinstance(git_sha, str) or not git_sha or type(git_dirty) is not bool:
        raise ValueError("scaling code identity is unavailable")
    if ((git_dirty and (not isinstance(fingerprint, str) or not fingerprint))
            or (not git_dirty and fingerprint is not None)):
        raise ValueError("scaling code identity has an inconsistent dirty-tree fingerprint")
    return detached


def is_valid_code_identity(value: object) -> bool:
    """Whether :func:`validated_code_identity` accepts ``value``."""
    try:
        validated_code_identity(value)
    except (TypeError, ValueError):
        return False
    return True


def provenance_code_identity(provenance: Mapping[str, object]) -> Dict[str, object]:
    """The Git identity ``provenance.json`` publishes at its top level (the run-start snapshot)."""
    return {field: provenance.get(field) for field in GIT_IDENTITY_FIELDS}


def validated_data_source_identities(value: object) -> Dict[str, Dict[str, object]]:
    """Return a detached, JSON-safe identity for exactly the splits a scaling cell consumes."""
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_SPLITS):
        raise ValueError(
            "scaling data source identities must contain exactly train, validation, and test")
    normalized: Dict[str, Dict[str, object]] = {}
    for split in SOURCE_SPLITS:
        source = value[split]
        if not isinstance(source, Mapping):
            raise TypeError(f"{split} source identity must be a mapping")
        detached = _detached_json(dict(source))
        if (not isinstance(detached, dict)
                or not isinstance(detached.get("format"), str)
                or not detached["format"]
                or type(detached.get("size_bytes")) is not int
                or detached["size_bytes"] < 0
                or not isinstance(detached.get("sha256"), str)
                or not detached["sha256"]):
            raise ValueError(f"{split} source identity is incomplete")
        normalized[split] = detached
    return normalized


def package_code_binding_failure(
    cellmeta:   Mapping[str, object],
    provenance: Mapping[str, object],
) -> Optional[str]:
    r"""Why the package code digest of ``scaling_cell.json`` is not the one ``provenance.json`` publishes, or None.

    A cell without :data:`PACKAGE_CODE_FIELD` (written before October 31, 2026) has no such binding and passes.
    """
    if PACKAGE_CODE_FIELD not in cellmeta:
        return None
    recorded = cellmeta[PACKAGE_CODE_FIELD]
    if not is_sha256(recorded) or provenance.get("code_identity_sha256") != recorded:
        return "package code digest differs between scaling_cell and provenance"
    return None
