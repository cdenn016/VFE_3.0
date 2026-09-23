"""Plotting-free validation helpers for loading one finalized run."""

import json
from pathlib import Path
from typing import Mapping, Optional

import torch

from vfe3.config import VFE3Config, config_from_serialized
from vfe3.run_artifacts import (
    _normalized_data_identity,
    _validate_best_model_mapping,
    _validate_live_figure_snapshot_mapping,
    semantic_config_fingerprint,
)


def load_run_config(run_dir: Path) -> 'tuple[VFE3Config, str]':
    r"""Rebuild ``(cfg, dataset)`` from one RunArtifacts ``config.json``."""
    path = run_dir / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("config"), Mapping):
        raise ValueError(f"run metadata {path} has no config mapping")
    raw_config = data["config"]
    stored_fingerprint = data.get("config_fingerprint")
    if (stored_fingerprint is not None
            and stored_fingerprint != semantic_config_fingerprint(raw_config)):
        raise ValueError(f"run metadata {path} has a config fingerprint mismatch")
    cfg = config_from_serialized(raw_config, source=str(path))
    dataset = data.get("dataset", "")
    if not isinstance(dataset, str):
        raise ValueError(f"run metadata {path} has a non-string dataset")
    return cfg, dataset


def load_best_model_state(
    path:                             Path,
    cfg:                              VFE3Config,
    expected_model_state:             Mapping[str, torch.Tensor],
    expected_code_identity:           str,
    expected_selection_data_identity: Mapping[str, object],

    *,
    map_location: object,
) -> Mapping[str, torch.Tensor]:
    """Return an owned model state after the unified selected-bundle validation boundary."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    try:
        validated = _validate_best_model_mapping(
            payload,
            cfg,
            expected_model_state,
            f"offline best-model bundle at {path}",
            expected_code_identity=expected_code_identity,
            expected_selection_data_identity=expected_selection_data_identity,
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return validated["model_state"]


def load_figure_model_state(
    path:                             Path,
    cfg:                              VFE3Config,
    expected_model_state:             Mapping[str, torch.Tensor],
    expected_code_identity:           str,
    expected_selection_data_identity: Optional[Mapping[str, object]],

    *,
    map_location: object,
) -> Mapping[str, torch.Tensor]:
    """Load either a selected bundle or the exact nonselected live-figure snapshot schema."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    try:
        if isinstance(payload, Mapping) and "artifact_kind" in payload:
            return _validate_live_figure_snapshot_mapping(
                payload,
                cfg,
                expected_model_state,
                expected_code_identity,
                path,
            )
        if expected_selection_data_identity is None:
            raise RuntimeError(
                f"offline selected-model loading requires a trusted validation-data identity "
                f"before loading {path}")
        validated = _validate_best_model_mapping(
            payload,
            cfg,
            expected_model_state,
            f"offline best-model bundle at {path}",
            expected_code_identity=expected_code_identity,
            expected_selection_data_identity=expected_selection_data_identity,
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return validated["model_state"]


def _load_run_provenance(run_dir: Path) -> 'tuple[Path, Mapping[str, object]]':
    """Load the independent run-provenance mapping or fail closed."""
    path = run_dir / "provenance.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"offline model loading requires readable provenance at {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"offline model loading requires mapping provenance at {path}")
    return path, payload


def load_run_figure_contract(
    run_dir: Path,
) -> 'tuple[str, Optional[Mapping[str, object]]]':
    """Load trusted code identity plus an optional selected-data identity for figure replay."""
    path, payload = _load_run_provenance(run_dir)
    code_identity = payload.get("code_identity_sha256")
    if (not isinstance(code_identity, str) or len(code_identity) != 64
            or any(character not in "0123456789abcdef" for character in code_identity)):
        raise RuntimeError(
            f"offline model loading requires a trusted executable-code identity in {path}")
    selection_identity = payload.get("selection_data_identity")
    if selection_identity is None:
        return code_identity, None
    try:
        normalized_identity = _normalized_data_identity(selection_identity)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"offline model loading found an invalid validation-data identity in {path}") from exc
    return code_identity, normalized_identity


def load_run_selection_contract(
    run_dir: Path,
) -> 'tuple[str, Mapping[str, object]]':
    """Load trusted selected-model identities from the run's independent provenance record."""
    code_identity, selection_identity = load_run_figure_contract(run_dir)
    if selection_identity is None:
        raise RuntimeError(
            f"offline selected-model loading requires a trusted validation-data identity in "
            f"{run_dir / 'provenance.json'}")
    return code_identity, selection_identity
