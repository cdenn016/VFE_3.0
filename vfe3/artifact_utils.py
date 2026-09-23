"""Stable artifact encodings shared by producers and readers."""

import hashlib
import json


def canonical_json_sha256(value: object) -> str:
    """SHA-256 of finite JSON with the persisted scaling contract's exact encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
