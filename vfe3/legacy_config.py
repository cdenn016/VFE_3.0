r"""Load a serialized config written by an older code line whose toggles have since been deleted.

``vfe3.config.config_from_serialized`` refuses any field this build does not declare. Runs trained
before a toggle was retired therefore cannot be reopened for analysis. :func:`lenient_config_from_serialized`
drops those fields (keeping the historical keys ``config_from_serialized`` itself migrates) and
returns them, so the caller can record what was ignored. Dropping a toggle whose non-default value
changed the forward pass changes the model silently; callers that evaluate should confirm that the
reloaded model reproduces the run's recorded loss.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Mapping, Tuple

from vfe3.config import VFE3Config, config_from_serialized

# Keys that are not VFE3Config fields but that config_from_serialized migrates itself.
MIGRATED_HISTORICAL_KEYS = frozenset({
    "_allow_legacy_dense_decode", "grad_clip_per_role", "exp_fp64_mode",
    "relational_mode", "relational_readout_mode",
})


def lenient_config_from_serialized(
    payload: Mapping[str, Any],

    *,
    source: str,
) -> Tuple[VFE3Config, Dict[str, Any]]:
    """Build a config after dropping fields this build does not declare; return (config, dropped)."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"serialized config from {source} must be a mapping")
    known = {field.name for field in fields(VFE3Config)} | MIGRATED_HISTORICAL_KEYS
    dropped = {str(key): value for key, value in payload.items() if key not in known}
    kept = {key: value for key, value in payload.items() if key in known}
    return config_from_serialized(kept, source=source), dropped
