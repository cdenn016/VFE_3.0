r"""Run artifacts for VFE_3.0 training: the persistence + reporting layer.

A training run produces a self-contained directory::

    run_dir/
      config.json        full VFE3Config + run metadata (n_params, dataset, device, timestamp)
      metrics.csv        one row per periodic eval (step, train_loss, lr, val_ce/ppl/bpc, diagnostics)
      checkpoints/
        step_<N>.pt      resumable {step, model_state, optimizer_state, config}
      best_model.pt      {model_state, config, config_fingerprint} at the lowest validation PPL
      test_results.json  end-of-run TEST-split eval on the reloaded best checkpoint
      summary.json       headline numbers (best_val_ppl, test_ppl, wall_time, ...)
      loss_curve.png     training cross-entropy trajectory
      val_ppl.png        validation perplexity trajectory (log-y, best marked)
      holonomy.png / gauge_trace_spread.png   gauge-geometry diagnostics
      free_energy_decomposition.png   per-token F budget snapshot + early/mid/late evolution
      free_energy_relationship.png     final-iterate objective vs validation CE

``RunArtifacts`` is OPT-IN: ``train`` only touches it when an instance is passed, so the silent
path (``artifacts=None``) writes nothing and is unchanged. ``finalize_run`` reloads the best-val
checkpoint, scores the held-out test split, and writes the summary + figures. Figure generation
is best-effort (a plotting/dependency error is logged, never fatal) so the numeric results
survive a viz problem.
"""

import copy
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch

from vfe3.contracts import is_conditional_filtering
from vfe3.config import (
    VFE3Config,
    config_from_serialized,
    serialize_config,
)
from vfe3.contracts import DataState, DataStateBuffer, ExecutableBuildMetadata
from vfe3.model.block_mlp import (
    block_mlp_build_metadata,
    get_block_mlp_registration,
)
from vfe3.process_utils import run_process_tree
from vfe3.runtime import deterministic_state

if TYPE_CHECKING:                                        # forward ref only: train imports RunArtifacts
    from vfe3.train import TrainingTerminalState         # at top level, so a runtime import here cycles


def _relational_artifact_fields(cfg: VFE3Config) -> Dict[str, object]:
    """Describe exact marginal routing and its separate conditional mean update."""
    if cfg.relational_mode == "off":
        return {}
    H, d = cfg.n_heads, cfg.d_head
    r, s = d, (None if cfg.relational_identity_content else d)
    input_dim = cfg.embed_dim if cfg.relational_input_scope == "full" else d
    predictive = cfg.relational_readout_mode == "predictive_content"
    output_dim = cfg.embed_dim if cfg.relational_output_scope == "full" else d
    layer_counts = {"score": 2*H*input_dim*r,
                    "content": 0 if s is None else H*s*(input_dim+output_dim)}
    counts = {role: count * cfg.n_layers for role, count in layer_counts.items()}
    return {"relational_operators": {
        "attention_mode": cfg.attention_mode,
        "routing": cfg.relational_routing,
        "n_layers": cfg.n_layers, "layer_parameter_sharing": "independent",
        "per_layer_parameter_counts": layer_counts,
        "shape_scope": "per_layer",
        "inference_semantics": ("relational_predictive_content" if predictive
                                else "relational_projected_conditional_filtering"),
        "readout_mode": cfg.relational_readout_mode,
        "output_scope": cfg.relational_output_scope,
        "mean_update": ("input_plus_predictive_mean" if cfg.attention_residual else
                        "replace_with_predictive_mean" if predictive else "conditional_fisher_step"),
        "mode": cfg.relational_mode,
        "input_scope": cfg.relational_input_scope,
        "input_dimension": input_dim,
        "head_output_dimension": output_dim,
        "score_parameterization": "free_full_row_rank",
        "score_initialization": "independent_orthonormal_rows",
        "score_role_parameter_counts": {"q": cfg.n_layers*H*input_dim*r, "k": cfg.n_layers*H*input_dim*r},
        "score_rank_per_head": r, "content_rank_per_head": s,
        "identity_content": cfg.relational_identity_content,
        "content_mode": "identity" if s is None else "full_width",
        "score_shape": [H, r, input_dim],
        "content_shapes": None if s is None else [[H, s, input_dim], [H, output_dim, s]],
        "parameter_counts": counts, "total_parameters": sum(counts.values()),
        "learning_rates": {
            role: (cfg.m_p_mu_lr if getattr(cfg, f"m_relational_{role}_lr") is None
                   else getattr(cfg, f"m_relational_{role}_lr")) if counts[role] else None
            for role in counts},
        "score_covariance_contract": "exact_full_reduced_covariance_from_local_gaussian",
        "covariance_update": (getattr(cfg, "predictive_covariance", "passthrough") if predictive
                              else "not_applicable"),
        "content_covariance_contract": ("exact_predictive_kernel_moment_match_plus_covariant_floor"
                                        if predictive and getattr(cfg, "predictive_covariance",
                                                                  "passthrough") == "moment_matched"
                                        else "frozen_receiver_bookkeeping_not_predictive_covariance" if predictive
                                        else "legacy_transported_diagonal_variance_mean_only_content"
                                        if cfg.family == "gaussian_diagonal"
                                        else "full_gaussian_transported_head_marginals_mean_only_content"),
        "self_coupling_coordinate_contract": (
            "ordered_conditional_kl_stored_coordinate_order"
            if cfg.family == "gaussian_full" and cfg.lambda_alpha_mode == "state_dependent_per_coord"
            else "family_default"),
        "factorization_policy": "zero_jitter_no_covariance_repairs",
        "factorization_fallback": "failed_entries_source_square_root_qr_fp64",
        "temperature_convention": "kappa_beta_times_sqrt_ambient_head_dim",
        "routing_state": "pre_update", "state_destination": "receiver_belief_mean",
        "source_beliefs": "live_pre_update" if predictive else "detached_pre_update",
        "outer_operator_and_frame_graph": "live",
        "canonical_shared_objective": False,
        "gradient_clipping": "shares_existing_mu_role_and_global_budgets",
    }}


def _conditional_artifact_fields(cfg: VFE3Config) -> Dict[str, object]:
    # The query/key frame pair is gone, so the conditional block carries no frame-role record.
    return _relational_artifact_fields(cfg)


def _conditional_unavailable_probe(probe: str, cfg: Optional[VFE3Config] = None) -> Dict[str, object]:
    """Publish an explicit unavailable record instead of running an incompatible depth/objective probe."""
    return {
        "available": False,
        "probe": probe,
        "inference_semantics": ("relational_predictive_content"
                                if cfg is not None and cfg.relational_readout_mode == "predictive_content"
                                else "relational_projected_conditional_filtering"
                                if cfg is not None and cfg.relational_mode != "off"
                                else "legacy_mean_field"),
        "reason": ("only the configured single readout is supported; inference-depth sweeps are unavailable"
                   if "depth" in probe or probe == "zero_e_steps_counterfactual"
                   else "routing and readout do not define the legacy shared stationary free-energy objective"),
        "points": [],
        "stages": {},
    }


def _require_nonnegative_int(value: object, field: str) -> int:
    """Return an exact nonnegative integer cursor; reject coercible lookalikes."""
    if type(value) is not int or value < 0:
        raise ValueError(f"data_state {field} must be a non-negative integer")
    return value


_DATA_IDENTITY_SCHEMA_VERSION = 2
_DATA_IDENTITY_FIELDS = {
    "schema_version",
    "dataset",
    "split",
    "tokenizer_tag",
    "tokenizer_encoding",
    "tokenizer_vocab_size",
    "model_vocab_size",
    "max_tokens",
    "source",
    "iterator",
}
_DATA_SOURCE_IDENTITY_FIELDS = {
    "format",
    "tokenizer_tag",
    "size_bytes",
    "sha256",
    "meta",
    "meta_sha256",
}
_BINARY_TOKEN_DTYPE_BYTES = {
    "uint8":  1,
    "int8":   1,
    "int16":  2,
    "int32":  4,
    "int64":  8,
}
_TENSOR_TOKEN_DTYPE_BYTES = {
    f"torch.{name}": itemsize for name, itemsize in _BINARY_TOKEN_DTYPE_BYTES.items()
}
_DATA_ITERATOR_IDENTITY_FIELDS = {
    "dataset_type",
    "seq_len",
    "stride",
    "pad_final",
    "n_windows",
    "batch_size",
    "drop_last",
    "sampler",
    "sampler_replacement",
    "sampler_num_samples",
}
_LIVE_FIGURE_SNAPSHOT_KIND = "live_figure_snapshot"
_LIVE_FIGURE_SNAPSHOT_SCHEMA_VERSION = 1


def _require_identity_sha256(value: object, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)):
        raise ValueError(f"data_state data_identity {field} must be a 64-digit SHA-256 hex digest")
    return value


def _package_code_identity() -> str:
    r"""Return the current content identity for the package and supported root execution drivers.

    A run captures this digest once when its artifact owner is constructed. Generated files, tests,
    documentation, Git metadata, and output directories are excluded, so artifact publication cannot
    change the identity. Recomputing at each run boundary is required for long-lived Spyder kernels:
    editing a source file must not reuse the previous run's identity.
    """
    root = Path(__file__).resolve().parent
    repository = root.parent
    digest = hashlib.sha256()
    package_sources = {
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    root_drivers = {
        repository / name
        for name in (
            "ablation.py",
            "compare_vocab_figures.py",
            "make_figures.py",
            "multiseed_analysis.py",
            "scaling.py",
            "scaling_analysis.py",
            "train_vfe3.py",
        )
        if (repository / name).is_file()
    }
    sources = sorted(package_sources | root_drivers)
    if not sources:
        raise RuntimeError("training code identity found no executable Python sources")
    for path in sources:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        try:
            content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        except OSError as exc:
            raise RuntimeError(
                f"training code identity cannot read {path.relative_to(root.parent)!s}") from exc
        for payload in (relative, content):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _require_identity_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"data_state data_identity {field} must be a positive integer")
    return value


def _normalized_data_identity(
    value: object,
) -> Dict[str, object]:
    r"""Return an owned JSON-normalized data contract or fail closed.

    JSON normalization prevents a caller from mutating nested checkpoint identity state after the
    save boundary and gives equality one canonical set of scalar/container semantics.
    """
    if not isinstance(value, Mapping):
        raise ValueError("data_state data_identity must be a mapping")
    try:
        normalized = json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ValueError("data_state data_identity must contain only JSON-compatible values") from exc
    if not isinstance(normalized, dict):
        raise ValueError("data_state data_identity must normalize to a mapping")
    missing = sorted(_DATA_IDENTITY_FIELDS - normalized.keys())
    if missing:
        raise ValueError(f"data_state data_identity is missing required field(s) {missing}")
    schema_version = normalized.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _DATA_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"data_state data_identity schema_version must be {_DATA_IDENTITY_SCHEMA_VERSION}")
    for field in ("dataset", "split"):
        if not isinstance(normalized.get(field), str) or not normalized[field]:
            raise ValueError(f"data_state data_identity {field} must be a nonempty string")
    tokenizer_tag = normalized.get("tokenizer_tag")
    tokenizer_encoding = normalized.get("tokenizer_encoding")
    tokenizer_vocab_size = normalized.get("tokenizer_vocab_size")
    if tokenizer_tag is not None and (not isinstance(tokenizer_tag, str) or not tokenizer_tag):
        raise ValueError(
            "data_state data_identity tokenizer_tag must be a nonempty string or null")
    if (tokenizer_encoding is not None
            and (not isinstance(tokenizer_encoding, str) or not tokenizer_encoding)):
        raise ValueError(
            "data_state data_identity tokenizer_encoding must be a nonempty string or null")
    if tokenizer_vocab_size is not None:
        _require_identity_positive_int(tokenizer_vocab_size, "tokenizer_vocab_size")
    _require_identity_positive_int(normalized.get("model_vocab_size"), "model_vocab_size")
    max_tokens = normalized.get("max_tokens")
    if max_tokens is not None:
        _require_identity_positive_int(max_tokens, "max_tokens")

    iterator = normalized.get("iterator")
    if not isinstance(iterator, dict):
        raise ValueError("data_state data_identity iterator must be a mapping")
    missing_iterator = sorted(_DATA_ITERATOR_IDENTITY_FIELDS - iterator.keys())
    if missing_iterator:
        raise ValueError(
            f"data_state data_identity iterator is missing required field(s) {missing_iterator}")
    if not isinstance(iterator.get("dataset_type"), str) or not iterator["dataset_type"]:
        raise ValueError(
            "data_state data_identity iterator dataset_type must be a nonempty string")
    for field in ("seq_len", "stride", "n_windows", "batch_size", "sampler_num_samples"):
        _require_identity_positive_int(iterator.get(field), f"iterator {field}")
    for field in ("pad_final", "drop_last"):
        if not isinstance(iterator.get(field), bool):
            raise ValueError(f"data_state data_identity iterator {field} must be a boolean")
    sampler = iterator.get("sampler")
    replacement = iterator.get("sampler_replacement")
    if sampler not in {"random", "sequential"}:
        raise ValueError(
            "data_state data_identity iterator sampler must be 'random' or 'sequential'")
    if sampler == "random":
        if not isinstance(replacement, bool):
            raise ValueError(
                "data_state data_identity iterator random sampler replacement must be a boolean")
    elif replacement is not None:
        raise ValueError(
            "data_state data_identity iterator sequential sampler replacement must be null")

    source = normalized.get("source")
    if not isinstance(source, dict):
        raise ValueError("data_state data_identity source must be a mapping")
    missing_source = sorted(_DATA_SOURCE_IDENTITY_FIELDS - source.keys())
    if missing_source:
        raise ValueError(
            f"data_state data_identity source is missing required field(s) {missing_source}")
    source_format = source.get("format")
    if source_format not in {"pt", "bin", "tensor"}:
        raise ValueError(
            "data_state data_identity source format must be one of 'pt', 'bin', or 'tensor'")
    source_tokenizer = source.get("tokenizer_tag")
    if source_format in {"pt", "bin"}:
        if not isinstance(source_tokenizer, str) or not source_tokenizer:
            raise ValueError(
                "data_state data_identity source tokenizer_tag must be a nonempty string")
        if (not isinstance(tokenizer_tag, str) or not tokenizer_tag
                or not isinstance(tokenizer_encoding, str) or not tokenizer_encoding):
            raise ValueError(
                "data_state data_identity file sources require tokenizer tag and encoding")
        _require_identity_positive_int(tokenizer_vocab_size, "tokenizer_vocab_size")
    elif source_tokenizer is not None and (
            not isinstance(source_tokenizer, str) or not source_tokenizer):
        raise ValueError(
            "data_state data_identity source tokenizer_tag must be a nonempty string or null")
    if source_tokenizer != tokenizer_tag:
        raise ValueError(
            "data_state data_identity tokenizer_tag must agree with source tokenizer_tag")
    size_bytes = _require_identity_positive_int(source.get("size_bytes"), "source size_bytes")
    _require_identity_sha256(source.get("sha256"), "source sha256")

    tokenizer_provenance_status = source.get("tokenizer_provenance_status")
    tokenizer_provenance = source.get("tokenizer_provenance")
    tokenizer_provenance_sha256 = source.get("tokenizer_provenance_sha256")
    if tokenizer_provenance_status is not None:
        if tokenizer_provenance_status not in {
                "manifest_verified", "filename_inferred_unverified"}:
            raise ValueError(
                "data_state source tokenizer_provenance_status is unsupported")
        if tokenizer_provenance_status == "filename_inferred_unverified":
            if tokenizer_provenance is not None or tokenizer_provenance_sha256 is not None:
                raise ValueError(
                    "unverified tokenizer provenance must not carry a manifest identity")
        else:
            if not isinstance(tokenizer_provenance, dict):
                raise ValueError("verified tokenizer provenance requires a manifest mapping")
            _require_identity_sha256(
                tokenizer_provenance_sha256,
                "source tokenizer_provenance_sha256",
            )
            expected_manifest = {
                "schema_version": 1,
                "dataset": normalized["dataset"],
                "split": normalized["split"],
                "tokenizer_tag": tokenizer_tag,
                "tokenizer_encoding": tokenizer_encoding,
                "tokenizer_vocab_size": tokenizer_vocab_size,
                "payload_sha256": source["sha256"],
            }
            if tokenizer_provenance != expected_manifest:
                raise ValueError(
                    "verified tokenizer provenance disagrees with the data identity")

    meta = source.get("meta")
    meta_sha256 = source.get("meta_sha256")
    if source_format == "pt":
        if meta is not None or meta_sha256 is not None:
            raise ValueError(
                "data_state data_identity pt source metadata fields must be null")
    else:
        if not isinstance(meta, dict):
            raise ValueError(
                f"data_state data_identity {source_format} source meta must be a mapping")
        missing_meta = sorted({"n_tokens", "dtype"} - meta.keys())
        if missing_meta:
            raise ValueError(
                f"data_state data_identity source meta is missing required field(s) {missing_meta}")
        n_tokens = _require_identity_positive_int(
            meta.get("n_tokens"), "source meta n_tokens")
        dtype = meta.get("dtype")
        dtype_bytes = (
            _BINARY_TOKEN_DTYPE_BYTES.get(dtype)
            if source_format == "bin"
            else _TENSOR_TOKEN_DTYPE_BYTES.get(dtype)
        )
        if dtype_bytes is None:
            raise ValueError(
                f"data_state data_identity source meta dtype {dtype!r} is unsupported")
        expected_bytes = n_tokens * dtype_bytes
        if size_bytes != expected_bytes:
            raise ValueError(
                "data_state data_identity source size_bytes must equal "
                f"n_tokens * dtype.itemsize ({expected_bytes})")
        if source_format == "bin":
            _require_identity_sha256(meta_sha256, "source meta_sha256")
        elif meta_sha256 is not None:
            raise ValueError(
                "data_state data_identity tensor source meta_sha256 must be null")
    return normalized


def semantic_config_fingerprint(
    config: Mapping[str, Any],
) -> str:
    """Return the stable SHA-256 fingerprint of a normalized semantic config mapping."""
    normalized = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _phi_chart_norm_route(
    model: torch.nn.Module,
    cfg:   object,
) -> Optional[str]:
    """Persist the active phi factor-bound norm route and pullback default radius."""
    configured_radius = getattr(cfg, "phi_mstep_max_matrix_norm", None)
    pullback_group = getattr(cfg, "m_phi_update_mode", "adamw") == "pullback_group"
    if configured_radius is None and not pullback_group:
        return None
    route = getattr(model.group, "phi_norm_route", None)
    route_name = route() if route is not None else "dense_fallback"
    if pullback_group:
        radius = 5.0 if configured_radius is None else float(configured_radius)
        return f"{route_name}:factor_radius={radius}"
    return route_name


def _decode_tau_eff(
    model: torch.nn.Module,
    cfg:   object,
) -> Optional[float]:
    r"""Realized prior-bank decode temperature, or ``None`` when no KL-to-bank decode is active.

    .. math:: \tau_{\mathrm{eff}} = \tau \exp(-\mathrm{clamp}(s, -3, 3)),

    with ``tau = decode_tau`` and ``s = decode_log_scale``. The scale is trainable unless the run
    froze it, so the persisted ``decode_tau`` is only the initialization of the temperature the
    decode ``logits = -KL / tau_eff`` actually used."""
    prior_bank = getattr(model, "prior_bank", None)
    if prior_bank is None or getattr(cfg, "decode_mode", "linear") == "linear":
        return None
    with torch.no_grad():
        return float(prior_bank._tau_eff())


def _atomic_replace(
    final: Path,                         # destination (the artifact name readers load)
    tmp:   Path,                         # same-directory temp file, already fully written

    *,
    delay:   float = 0.2,
    retries: int   = 5,
) -> None:
    r"""Atomically publish ``tmp`` over ``final`` via ``os.replace`` (same-volume rename).

    Same-directory temp + ``os.replace`` makes the name switch atomic on one volume. The temporary
    file is flushed with ``fsync`` before publication; on POSIX the containing directory is also
    flushed after the rename. Windows has no portable directory-fsync primitive, so it provides the
    flushed-file atomic replacement guarantee but not a claim of directory-entry durability across
    sudden power loss. Retries with backoff on ``PermissionError`` -- Windows can hold a
    transient open-handle lock on the destination (this host has hit it on ``best_model.pt``) --
    and re-raises any other error (and the last ``PermissionError``) so a real failure is never
    swallowed. On pre-publication raising paths the orphaned ``tmp`` is best-effort deleted first;
    between retries it must survive because it is the source of the next replace. A post-rename
    POSIX directory-flush error propagates after publication and cannot re-enter that retry loop."""
    def _cleanup_tmp() -> None:
        try:                                             # cleanup failure must not mask the original error
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
    except Exception:
        _cleanup_tmp()
        raise
    for i in range(retries):
        try:
            os.replace(tmp, final)
            break
        except PermissionError:
            if i == retries - 1:
                _cleanup_tmp()
                raise
            time.sleep(delay)
        except Exception:
            _cleanup_tmp()
            raise
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(final.parent, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


_TEMP_CLEANUP_RETRIES = 5                        # mirrors _atomic_replace's Windows-lock backoff
_TEMP_CLEANUP_DELAY   = 0.2


@contextmanager
def _unique_sibling_temp(final: Path) -> Iterator[Path]:
    r"""Yield a uniquely reserved same-directory temporary path for one artifact writer.

    ``mkstemp`` performs the reservation atomically, so concurrent writers to the same final
    artifact never share a temporary filename.  The caller must publish with
    :func:`_atomic_replace`; this context also removes an unconsumed temporary after any failure.

    Cleanup RETRIES rather than taking a single shot (audit 2026-07-27): a writer that failed
    mid-save can still hold the handle while this unwinds, and Windows refuses to unlink an open
    file, so one attempt loses a race it wins moments later. The incident that motivated this
    abandoned a 2.48 GB checkpoint temp that was freely deletable minutes afterward; with a failed
    save now non-fatal to training, an unretried cleanup would leak that much per checkpoint
    interval and fill the disk the run is still writing to.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        dir=str(final.parent),
        prefix=f".{final.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(raw_path)
    try:
        yield tmp
    finally:
        for i in range(_TEMP_CLEANUP_RETRIES):
            try:
                tmp.unlink(missing_ok=True)
                break
            except OSError:                              # cleanup failure must not mask the original error
                if i == _TEMP_CLEANUP_RETRIES - 1:
                    break
                time.sleep(_TEMP_CLEANUP_DELAY)


@contextmanager
def _unique_sibling_output_temp(final: Path) -> Iterator[Path]:
    r"""Yield a reserved sibling whose suffix preserves renderer format inference."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp{final.suffix}" if final.suffix else ".tmp"
    fd, raw_path = tempfile.mkstemp(
        dir=str(final.parent),
        prefix=f".{final.stem}.",
        suffix=suffix,
    )
    os.close(fd)
    tmp = Path(raw_path)
    try:
        yield tmp
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json_atomic(final: Path, obj: object) -> Path:
    """Serialize JSON through a unique same-directory temporary and atomically publish it."""
    final = Path(final)
    with _unique_sibling_temp(final) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        _atomic_replace(final, tmp)
    return final


_CONTROLLED_FIGURE_BANK_TOKENS = 16_384
_FINAL_FIGURE_TIMEOUT_SECONDS = 2 * 60 * 60
_ATTENTION_FIGURE_TIMEOUT_SECONDS = 15 * 60


def _snapshot_report_batches(
    loader: Iterable,
) -> List[object]:
    """Copy an actual-shape-bounded report population to CPU for a fresh plotting process."""
    batches: List[object] = []
    captured_tokens = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            tokens = batch[0]
        elif isinstance(batch, torch.Tensor):
            tokens = batch
        else:
            raise TypeError(
                "figure report loader must yield a tensor or a tuple/list containing tensors"
            )
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise TypeError("figure report token batches must be rank-two tensors")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("figure report token batches must have positive dimensions")
        remaining = _CONTROLLED_FIGURE_BANK_TOKENS - captured_tokens
        if remaining <= 0:
            break
        batch_rows, sequence_length = tokens.shape
        if sequence_length > remaining:
            row_limit, sequence_limit = 1, remaining
        else:
            row_limit = min(batch_rows, max(1, remaining // max(sequence_length, 1)))
            sequence_limit = sequence_length

        def _copy(item: object) -> object:
            if not isinstance(item, torch.Tensor):
                return item
            if item.ndim >= 2 and item.shape[:2] == tokens.shape[:2]:
                return item[:row_limit, :sequence_limit].detach().cpu().clone()
            return item.detach().cpu().clone()

        copied = tuple(_copy(item) for item in batch) if isinstance(batch, (tuple, list)) else _copy(batch)
        batches.append(copied)
        captured_tokens += row_limit * sequence_limit
        if captured_tokens >= _CONTROLLED_FIGURE_BANK_TOKENS:
            break
    return batches


def _run_figures_isolated(
    model:     torch.nn.Module,
    artifacts: 'RunArtifacts',
    losses:    Optional[List[float]],
    logger:    logging.Logger,

    *,
    generate_publication:       bool,
    generate_embedding_figures: bool               = False,
    report_loader:              Optional[Iterable] = None,
) -> bool:
    r"""Generate every end-of-run figure in a disposable process.

    Anaconda NumPy and PyTorch can load distinct copies of Intel OpenMP.  A plotting-time NumPy or
    SciPy kernel may then call the native runtime's fatal ``abort`` path, which Python exception
    handlers cannot contain.  The compatibility setting is therefore scoped to this child only:
    training numerics and the long-lived Spyder kernel never inherit it.  A child failure returns
    ``False`` after all numeric artifacts have already been published.
    """
    if type(generate_embedding_figures) is not bool:
        raise ValueError("generate_embedding_figures must be an exact boolean")
    run_dir = artifacts.run_dir.resolve()
    model_device = next(model.parameters()).device
    report_batches: Optional[List[object]] = None
    if generate_publication and report_loader is not None:
        report_batches = _snapshot_report_batches(report_loader)

    with _unique_sibling_temp(run_dir / "figure_request.json") as request_path:
        with _unique_sibling_temp(run_dir / "figure_report_batches.pt") as batches_path:
            with _unique_sibling_temp(run_dir / "figure_live_model.pt") as model_path:
                if report_batches is not None:
                    torch.save(report_batches, batches_path)
                live_model_path: Optional[str] = None
                if generate_publication:
                    # Replay the caller's evaluated state even when an earlier best exists.
                    config = serialize_config(artifacts.cfg)
                    torch.save({
                        "artifact_kind":        _LIVE_FIGURE_SNAPSHOT_KIND,
                        "schema_version":       _LIVE_FIGURE_SNAPSHOT_SCHEMA_VERSION,
                        "model_state":          model.state_dict(),
                        "config":               config,
                        "config_fingerprint":   semantic_config_fingerprint(config),
                        "code_identity_sha256": artifacts.code_identity_sha256,
                    }, model_path)
                    live_model_path = str(model_path)
                request = {
                    "run_dir":             str(run_dir),
                    "losses":              ([float(value) for value in losses] if losses else None),
                    "generate_publication": bool(generate_publication),
                    "report_batches_path": (str(batches_path) if report_batches is not None else None),
                    "model_bundle_path":   live_model_path,
                    "device":              str(model_device),
                    "max_tokens":          _CONTROLLED_FIGURE_BANK_TOKENS,
                    "generate_embedding_figures": generate_embedding_figures,
                }
                request_path.write_text(json.dumps(request), encoding="utf-8")

                environment = os.environ.copy()
                environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                environment["VFE3_FIGURE_REQUEST"] = str(request_path)
                environment["PYTHONUNBUFFERED"] = "1"
                command = [sys.executable, "-m", "vfe3.viz.figure_worker"]
                repo_root = Path(__file__).resolve().parent.parent
                offloaded = generate_publication and model_device.type == "cuda"
                try:
                    if offloaded:
                        model.to("cpu")
                        torch.cuda.empty_cache()
                    completed = run_process_tree(
                        command,
                        cwd=str(repo_root),
                        env=environment,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=_FINAL_FIGURE_TIMEOUT_SECONDS,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "isolated figure process exceeded %d seconds; numeric results are saved",
                        _FINAL_FIGURE_TIMEOUT_SECONDS,
                    )
                    return False
                except (OSError, subprocess.SubprocessError) as exc:
                    logger.warning(
                        "isolated figure process could not start (%s); numeric results are saved", exc
                    )
                    return False
                finally:
                    if offloaded:
                        model.to(model_device)

    if completed.stdout.strip():
        logger.info("isolated figure process:\n%s", completed.stdout.rstrip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        logger.warning(
            "isolated figure process exited with code %d (%s); numeric results are saved",
            completed.returncode,
            detail[-4000:],
        )
        return False
    if completed.stderr.strip():
        logger.info("isolated figure process diagnostics:\n%s", completed.stderr.rstrip())
    return True


def _run_attention_maps_isolated(
    maps:    torch.Tensor,
    run_dir: Path,
    logger:  logging.Logger,

    *,
    channel: str,
    step:    int,
) -> Optional[List[Path]]:
    r"""Render periodic attention maps in a child that may safely suffer a native plotting abort."""
    try:
        tensor = maps.detach().cpu() if isinstance(maps, torch.Tensor) else torch.as_tensor(maps)
        if channel == "estep":
            # (3, L, H, N, N): the E-step's own pattern, the converged re-score, and the prior
            # alone, stacked so one worker call renders the heatmaps AND the distance profile that
            # compares them (audit 2026-07-26 R-3). Written to attention_estep/, never to
            # attention/ -- the shipped maps are left exactly as they are.
            if tensor.ndim != 5 or tensor.shape[0] != 3:
                raise ValueError(
                    f"estep maps must be (3, L, H, N, N); got {tuple(tensor.shape)}"
                )
            expected = [
                run_dir / "attention_estep" / f"step_{step}_layer{layer}_head{head}.png"
                for layer in range(tensor.shape[1])
                for head in range(tensor.shape[2])
            ] + [run_dir / "attention_estep" / f"step_{step}_profile.png"]
        elif channel == "beta":
            if tensor.ndim == 2:
                tensor = tensor[None, None]
            elif tensor.ndim == 3:
                tensor = tensor[None]
            if tensor.ndim != 4:
                raise ValueError(
                    f"attention maps must be (L, H, N, N); got {tuple(tensor.shape)}"
                )
            expected = [
                run_dir / "attention" / f"step_{step}_layer{layer}_head{head}.png"
                for layer in range(tensor.shape[0])
                for head in range(tensor.shape[1])
            ]
        else:
            raise ValueError(f"unsupported attention channel {channel!r}")

        with _unique_sibling_temp(run_dir / f"attention_{channel}_request.json") as request_path:
            with _unique_sibling_temp(run_dir / f"attention_{channel}_maps.pt") as maps_path:
                torch.save(tensor, maps_path)
                request_path.write_text(json.dumps({
                    "mode":      "attention",
                    "run_dir":   str(run_dir.resolve()),
                    "maps_path": str(maps_path),
                    "channel":   channel,
                    "step":      int(step),
                }), encoding="utf-8")
                environment = os.environ.copy()
                environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                environment["VFE3_FIGURE_REQUEST"] = str(request_path)
                environment["PYTHONUNBUFFERED"] = "1"
                completed = run_process_tree(
                    [sys.executable, "-m", "vfe3.viz.figure_worker"],
                    cwd=str(Path(__file__).resolve().parent.parent),
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=_ATTENTION_FIGURE_TIMEOUT_SECONDS,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "no stderr"
            logger.warning(
                "%s attention-map worker exited with code %d (%s); training continues",
                channel,
                completed.returncode,
                detail[-4000:],
            )
            return None
        return [path for path in expected if path.is_file()]
    except Exception as exc:
        logger.warning(
            "%s attention-map figure at step %d failed (%s); training continues",
            channel,
            step,
            exc,
        )
        return None


def _selection_semantic_config(
    config: 'VFE3Config | Mapping[str, object]',
) -> Dict[str, object]:
    r"""Project a config down to the fields that determine the SELECTED weights.

    Model selection depends on architecture, family/transport/decode, optimizer/schedule, and every
    objective weight -- but NOT on resume bookkeeping/policy (``resume_from``,
    ``trust_resume_checkpoint``) or output policy/cadence (``log_interval``,
    ``checkpoint_interval``, ``generate_embedding_figures``). Comparing this projection lets
    a cross-run resume carry otherwise-identical selected weights even when the resumed run changed
    its resume path or figure/log cadence, while every architecture/objective difference still
    invalidates the bundle.

    A live :class:`VFE3Config` starts from ``asdict`` directly. A SERIALIZED mapping is first checked
    key-by-key against the current fields -- an unknown newer field FAILS CLOSED rather than being
    silently ignored by a permissive loader -- and is then passed through strict typed migration so
    a genuinely older mapping acquires the current defaults for any
    field it predates."""
    if isinstance(config, VFE3Config):
        normalized = asdict(config)
    elif isinstance(config, Mapping):
        try:
            normalized = asdict(config_from_serialized(
                config,
                source="best-model selection compatibility",
            ))
        except ValueError as exc:
            raise ValueError(
                f"stored config blocks inherited model selection: {exc}"
            ) from exc
    else:
        raise TypeError(
            f"_selection_semantic_config expects a VFE3Config or mapping, got {type(config).__name__}")
    for key in (
        "resume_from",
        "trust_resume_checkpoint",
        "log_interval",
        "checkpoint_interval",
        "generate_embedding_figures",
    ):
        normalized.pop(key, None)
    return normalized


class _BestModelContractMismatch(RuntimeError):
    """A valid selected bundle was measured under a different live selection contract."""


def _validate_best_model_mapping(
    bundle:               object,
    cfg:                  Optional[VFE3Config],
    expected_model_state: Mapping[str, torch.Tensor],
    context:              str,

    *,
    expected_code_identity:           Optional[str] = None,
    expected_selection_data_identity: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return an owned selected state after validating the complete artifact contract."""
    legacy_fields = {"model_state", "config", "config_fingerprint"}
    current_fields = legacy_fields | {"code_identity_sha256", "selection_data_identity"}
    if not isinstance(bundle, Mapping) or set(bundle) not in (legacy_fields, current_fields):
        raise RuntimeError(f"{context} is not a semantic best-model mapping")
    saved_config = bundle["config"]
    if not isinstance(saved_config, Mapping):
        raise RuntimeError(f"{context} has a non-mapping config")
    if (cfg is not None
            and semantic_config_fingerprint(_selection_semantic_config(saved_config))
            != semantic_config_fingerprint(_selection_semantic_config(cfg))):
        raise _BestModelContractMismatch(
            f"{context} does not match the active selection config")
    saved_state = bundle["model_state"]
    validated_state = _validate_checkpoint_model_state(
        saved_state,
        expected_model_state,
        context,
        artifact_class=None,
    )
    validated = dict(bundle)
    if set(bundle) == current_fields:
        code_identity = bundle["code_identity_sha256"]
        if (not isinstance(code_identity, str) or len(code_identity) != 64
                or any(character not in "0123456789abcdef" for character in code_identity)):
            raise RuntimeError(f"{context} has an invalid executable-code identity")
        selection_identity = bundle["selection_data_identity"]
        if selection_identity is not None:
            try:
                validated["selection_data_identity"] = _normalized_data_identity(selection_identity)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{context} has an invalid validation-data identity") from exc
    if expected_code_identity is not None:
        if set(bundle) != current_fields:
            raise _BestModelContractMismatch(
                f"{context} has no executable-code identity")
        if validated["code_identity_sha256"] != expected_code_identity:
            raise _BestModelContractMismatch(
                f"{context} does not match the active executable-code identity")
    if expected_selection_data_identity is not None:
        if set(bundle) != current_fields:
            raise _BestModelContractMismatch(
                f"{context} has no validation-data identity")
        try:
            live_selection_identity = _normalized_data_identity(
                expected_selection_data_identity)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{context} received an invalid active validation-data identity") from exc
        saved_selection_identity = validated["selection_data_identity"]
        if saved_selection_identity is None:
            raise _BestModelContractMismatch(
                f"{context} has no validation-data identity")
        if saved_selection_identity != live_selection_identity:
            raise _BestModelContractMismatch(
                f"{context} does not match the active validation-data identity")
    validated["model_state"] = {
        key: tensor.detach().clone()
        for key, tensor in validated_state.items()
    }
    return validated


def _validate_live_figure_snapshot_mapping(
    bundle:                 object,
    cfg:                    VFE3Config,
    expected_model_state:   Mapping[str, torch.Tensor],
    expected_code_identity: str,
    path:                   Path,
) -> Mapping[str, torch.Tensor]:
    """Return owned live weights after validating the exact nonselected figure contract."""
    fields = {
        "artifact_kind",
        "schema_version",
        "model_state",
        "config",
        "config_fingerprint",
        "code_identity_sha256",
    }
    context = f"live figure snapshot at {path}"
    if not isinstance(bundle, Mapping) or set(bundle) != fields:
        raise RuntimeError(f"{context} does not have the exact live-snapshot schema")
    if bundle["artifact_kind"] != _LIVE_FIGURE_SNAPSHOT_KIND:
        raise RuntimeError(f"{context} has an invalid artifact kind")
    if type(bundle["schema_version"]) is not int or (
            bundle["schema_version"] != _LIVE_FIGURE_SNAPSHOT_SCHEMA_VERSION):
        raise RuntimeError(f"{context} has an unsupported schema version")
    saved_config = bundle["config"]
    if not isinstance(saved_config, Mapping):
        raise RuntimeError(f"{context} has a non-mapping config")
    try:
        effective_saved_config = asdict(config_from_serialized(
            saved_config,
            source="live figure snapshot config",
        ))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if (semantic_config_fingerprint(effective_saved_config)
            != semantic_config_fingerprint(asdict(cfg))):
        raise RuntimeError(f"{context} does not match the active full configuration")
    code_identity = bundle["code_identity_sha256"]
    if (not isinstance(code_identity, str) or len(code_identity) != 64
            or any(character not in "0123456789abcdef" for character in code_identity)):
        raise RuntimeError(f"{context} has an invalid executable-code identity")
    if code_identity != expected_code_identity:
        raise RuntimeError(f"{context} does not match the active executable-code identity")
    validated_state = _validate_checkpoint_model_state(
        bundle["model_state"],
        expected_model_state,
        path,
        artifact_class="live figure snapshot",
    )
    return {
        key: tensor.detach().clone()
        for key, tensor in validated_state.items()
    }


def _read_best_model_bundle(
    path:                             Path,
    cfg:                              VFE3Config,
    expected_model_state:             Mapping[str, torch.Tensor],
    map_location:                     'str | torch.device',
    expected_code_identity:           str,
    expected_selection_data_identity: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    r"""Safe-load ``best_model.pt`` and validate it as a portable selected-weights bundle, NONMUTATING.

    Loaded with ``weights_only=True`` (the bundle is only tensors, an ``asdict`` config, and a
    fingerprint string). Fails closed unless the exact current five-field schema passes its
    semantic selection config, model-state structure and finiteness, executable-code identity, and
    validation-data identity checks. A bundle with a null stored data identity may remain
    structurally readable for legacy checkpoint persistence, but it cannot supply cross-run selection or
    headline finalization when a live identity is expected. Neither live configuration nor model state
    is mutated. The returned model-state mapping owns cloned tensors, so later caller mutation cannot
    alter the validated source bundle."""
    bundle = torch.load(path, map_location=map_location, weights_only=True)
    return _validate_best_model_mapping(
        bundle,
        cfg,
        expected_model_state,
        f"best-model bundle at {path}",
        expected_code_identity=expected_code_identity,
        expected_selection_data_identity=expected_selection_data_identity,
    )


def _validate_checkpoint_model_state(
    saved_state:          object,
    expected_model_state: Mapping[str, torch.Tensor],
    artifact:             'str | Path',

    *,
    artifact_class: Optional[str] = "checkpoint",
) -> Mapping[str, torch.Tensor]:
    """Validate a checkpoint model state completely before any live tensor is copied."""
    context = f"{artifact_class} {artifact}" if artifact_class is not None else str(artifact)
    if not isinstance(saved_state, Mapping):
        raise RuntimeError(f"{context} has a non-mapping model_state")
    if set(saved_state) != set(expected_model_state):
        raise RuntimeError(f"{context} model_state keys do not match the live model")
    for key, expected in expected_model_state.items():
        actual = saved_state[key]
        if not isinstance(actual, torch.Tensor):
            raise RuntimeError(f"{context} model_state entry {key!r} is not a tensor")
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or actual.layout != expected.layout):
            raise RuntimeError(
                f"{context} model_state entry {key!r} has an incompatible "
                f"shape/dtype/layout")
        if ((actual.is_floating_point() or actual.is_complex())
                and not bool(torch.isfinite(actual).all().item())):
            raise RuntimeError(
                f"{context} model_state entry {key!r} contains nonfinite values")
    return saved_state


def _validated_saved_successful_updates(
    saved_groups: object,
    saved_step:   int,
) -> Optional[int]:
    """Validate the all-or-none accepted-update clock on serialized optimizer groups."""
    if not isinstance(saved_groups, list) or not all(
            isinstance(group, Mapping) for group in saved_groups):
        raise RuntimeError("checkpoint optimizer_state param_groups must be a list of mappings")
    present = ["successful_updates" in group for group in saved_groups]
    if any(present) and not all(present):
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates must be present on every group or none")
    if not any(present):
        return None
    values = [group["successful_updates"] for group in saved_groups]
    if any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates must be exact non-negative integers")
    if len(set(values)) != 1:
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates must agree across all groups")
    count = values[0]
    if count > saved_step:
        raise RuntimeError(
            "checkpoint optimizer_state successful_updates cannot exceed checkpoint step")
    return count


def _validate_optimizer_state(
    saved_state: object,
    optimizer:   torch.optim.Optimizer,
    saved_step:  int,
) -> Mapping[str, object]:
    """Preflight optimizer topology and populated per-parameter slots without mutation."""
    if not isinstance(saved_state, Mapping):
        raise RuntimeError("checkpoint optimizer_state must be a mapping when optimizer is supplied")
    saved_groups = saved_state.get("param_groups")
    saved_slots = saved_state.get("state")
    has_phi_group_differential_mode = hasattr(optimizer, "_phi_group_differential_mode")
    expected_top_level = {"state", "param_groups"} | (
        {"phi_group_differential_mode"} if has_phi_group_differential_mode else set())
    if set(saved_state) != expected_top_level:
        raise RuntimeError("checkpoint optimizer_state has an invalid top-level schema")
    successful_updates = _validated_saved_successful_updates(saved_groups, saved_step)
    maximum_slot_step = successful_updates if successful_updates is not None else saved_step
    if not isinstance(saved_slots, Mapping):
        raise RuntimeError("checkpoint optimizer_state state must be a mapping")
    if len(saved_groups) != len(optimizer.param_groups):
        raise RuntimeError("checkpoint optimizer_state parameter-group count mismatch")

    parameter_by_id: Dict[int, Tuple[torch.Tensor, Mapping[str, object]]] = {}
    for saved_group, live_group in zip(saved_groups, optimizer.param_groups):
        saved_ids = saved_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(saved_ids, list) or not isinstance(live_parameters, list):
            raise RuntimeError("checkpoint optimizer_state group params must be lists")
        if len(saved_ids) != len(live_parameters):
            raise RuntimeError("checkpoint optimizer_state parameter topology mismatch")
        for parameter_id, parameter in zip(saved_ids, live_parameters):
            if type(parameter_id) is not int or parameter_id in parameter_by_id:
                raise RuntimeError("checkpoint optimizer_state parameter ids are invalid")
            parameter_by_id[parameter_id] = (parameter, live_group)

    for parameter_id, state in saved_slots.items():
        if type(parameter_id) is not int or parameter_id not in parameter_by_id:
            raise RuntimeError("checkpoint optimizer_state contains an unknown parameter id")
        if not isinstance(state, Mapping):
            raise RuntimeError("checkpoint optimizer_state parameter slot must be a mapping")
        if not state:
            continue
        parameter, group = parameter_by_id[parameter_id]
        keys = set(state)
        if bool(group.get("pullback_group", False)):
            raise RuntimeError(
                "checkpoint optimizer_state contains unsupported parameter slots for a stateless "
                "pullback-group parameter"
            )
        elif keys & {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            expected_keys = {"step", "exp_avg", "exp_avg_sq"}
            if bool(group.get("amsgrad", False)):
                expected_keys.add("max_exp_avg_sq")
            if keys != expected_keys:
                raise RuntimeError("checkpoint optimizer_state has an invalid AdamW slot schema")
            step = state["step"]
            valid_step = (
                isinstance(step, torch.Tensor)
                and step.device.type == "cpu"
                and step.dtype == torch.float32
                and step.layout == torch.strided
                and step.ndim == 0
                and bool(torch.isfinite(step))
                and float(step) >= 0.0
                and float(step).is_integer()
                and float(step) <= maximum_slot_step
            )
            if not valid_step:
                raise RuntimeError("checkpoint optimizer_state AdamW step is invalid")
            tensor_slots = ["exp_avg", "exp_avg_sq"]
            if bool(group.get("amsgrad", False)):
                tensor_slots.append("max_exp_avg_sq")
            for name in tensor_slots:
                value = state[name]
                if (not isinstance(value, torch.Tensor)
                        or value.device.type != "cpu"
                        or value.shape != parameter.shape
                        or value.dtype != parameter.dtype
                        or value.layout != parameter.layout
                        or not bool(torch.isfinite(value).all())):
                    raise RuntimeError(
                        f"checkpoint optimizer_state {name} is incompatible with its parameter")
                if name in {"exp_avg_sq", "max_exp_avg_sq"} and bool((value < 0).any()):
                    raise RuntimeError(
                        f"checkpoint optimizer_state {name} must be nonnegative")
        else:
            raise RuntimeError("checkpoint optimizer_state contains unsupported parameter slots")
    return saved_state


_OPTIMIZER_PARAMETER_MANIFEST_SCHEMA_VERSION = 2


def _optimizer_parameter_name_shape_groups(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> List[List[Dict[str, object]]]:
    """Describe live optimizer parameters without comparing checkpoint-local numeric IDs."""
    name_by_identity: Dict[int, str] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        # Weight tying can intentionally register one Parameter under multiple module paths.
        # PyTorch's canonical named_parameters() convention is the first registration name.
        name_by_identity.setdefault(identity, name)

    seen: set[int] = set()
    parameter_groups: List[List[Dict[str, object]]] = []
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise RuntimeError("optimizer parameter group params must be a list")
        entries: List[Dict[str, object]] = []
        for parameter in parameters:
            identity = id(parameter)
            name = name_by_identity.get(identity)
            if name is None:
                raise RuntimeError("optimizer contains a parameter not owned by the model")
            if identity in seen:
                raise RuntimeError("optimizer contains a parameter more than once")
            seen.add(identity)
            entries.append({"name": name, "shape": list(parameter.shape)})
        parameter_groups.append(entries)
    return parameter_groups


def _optimizer_parameter_manifest(
    model:           torch.nn.Module,
    optimizer:       torch.optim.Optimizer,
    optimizer_state: Mapping[str, object],
) -> Dict[str, object]:
    """Record checkpoint-local slot IDs with canonical names/shapes in exact group order."""
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list):
        raise RuntimeError("optimizer state param_groups must be a list")
    name_shape_groups = _optimizer_parameter_name_shape_groups(model, optimizer)
    if len(saved_groups) != len(name_shape_groups):
        raise RuntimeError("optimizer state parameter-group count mismatch")

    seen_ids: set[int] = set()
    parameter_groups: List[List[Dict[str, object]]] = []
    for saved_group, name_shape_group in zip(saved_groups, name_shape_groups):
        if not isinstance(saved_group, Mapping):
            raise RuntimeError("optimizer state parameter group must be a mapping")
        parameter_ids = saved_group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(name_shape_group):
            raise RuntimeError("optimizer state parameter topology mismatch")
        entries: List[Dict[str, object]] = []
        for parameter_id, name_shape in zip(parameter_ids, name_shape_group):
            if type(parameter_id) is not int or parameter_id < 0 or parameter_id in seen_ids:
                raise RuntimeError("optimizer state parameter ids are invalid")
            seen_ids.add(parameter_id)
            entries.append({"parameter_id": parameter_id, **name_shape})
        parameter_groups.append(entries)
    return {
        "schema_version": _OPTIMIZER_PARAMETER_MANIFEST_SCHEMA_VERSION,
        "parameter_groups": parameter_groups,
    }


def _validate_serialized_optimizer_parameter_manifest(
    manifest: object,
) -> Mapping[str, object]:
    """Validate the manifest schema; checkpoints written before 2026-09-27 also carry a sha256."""
    fields = {"schema_version", "parameter_groups"}
    if not isinstance(manifest, Mapping) or set(manifest) not in (fields, fields | {"sha256"}):
        raise RuntimeError("checkpoint optimizer parameter manifest has an invalid schema")
    version = manifest["schema_version"]
    groups = manifest["parameter_groups"]
    if type(version) is not int or version != _OPTIMIZER_PARAMETER_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("checkpoint optimizer parameter manifest has an unsupported schema version")
    if not isinstance(groups, list):
        raise RuntimeError("checkpoint optimizer parameter manifest parameter_groups must be a list")
    for group in groups:
        if not isinstance(group, list):
            raise RuntimeError("checkpoint optimizer parameter manifest group must be a list")
        for entry in group:
            if (not isinstance(entry, Mapping)
                    or set(entry) != {"parameter_id", "name", "shape"}):
                raise RuntimeError("checkpoint optimizer parameter manifest entry is invalid")
            parameter_id = entry["parameter_id"]
            name = entry["name"]
            shape = entry["shape"]
            if type(parameter_id) is not int or parameter_id < 0:
                raise RuntimeError("checkpoint optimizer parameter manifest id is invalid")
            if not isinstance(name, str) or not name:
                raise RuntimeError("checkpoint optimizer parameter manifest name is invalid")
            if (not isinstance(shape, list)
                    or any(type(dimension) is not int or dimension < 0 for dimension in shape)):
                raise RuntimeError("checkpoint optimizer parameter manifest shape is invalid")
    return manifest


def _validate_optimizer_parameter_manifest(
    manifest:  object,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Mapping[str, object]:
    """Validate the manifest schema and its name/shape/group ordering against the live optimizer."""
    validated = _validate_serialized_optimizer_parameter_manifest(manifest)
    groups = validated["parameter_groups"]

    manifest_name_shapes = [
        [{"name": entry["name"], "shape": entry["shape"]} for entry in group]
        for group in groups
    ]
    if manifest_name_shapes != _optimizer_parameter_name_shape_groups(model, optimizer):
        raise RuntimeError(
            "checkpoint optimizer parameter manifest does not match the live optimizer")
    return validated


def _validate_realized_optimizer_state(
    optimizer:          torch.optim.Optimizer,
    saved_step:         int,
    successful_updates: Optional[int],
) -> None:
    """Validate optimizer tensors after load placement onto their live parameters."""
    maximum_slot_step = successful_updates if successful_updates is not None else saved_step
    owner_by_identity: Dict[int, Tuple[torch.Tensor, Mapping[str, object]]] = {}
    for group in optimizer.param_groups:
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise RuntimeError("realized optimizer group params must be a list")
        for parameter in parameters:
            identity = id(parameter)
            if identity in owner_by_identity:
                raise RuntimeError("realized optimizer contains a duplicate parameter")
            owner_by_identity[identity] = (parameter, group)

    for parameter, state in optimizer.state.items():
        owned = owner_by_identity.get(id(parameter))
        if owned is None:
            raise RuntimeError("realized optimizer state has an unknown parameter")
        parameter, group = owned
        if not isinstance(state, Mapping):
            raise RuntimeError("realized optimizer parameter slot must be a mapping")
        if not state:
            continue
        keys = set(state)
        if bool(group.get("pullback_group", False)):
            raise RuntimeError("realized optimizer has state for a stateless pullback parameter")
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if bool(group.get("amsgrad", False)):
            expected_keys.add("max_exp_avg_sq")
        if keys != expected_keys:
            raise RuntimeError("realized optimizer has an invalid AdamW slot schema")
        step = state["step"]
        expected_step_device = (
            parameter.device
            if bool(group.get("fused", False)) or bool(group.get("capturable", False))
            else torch.device("cpu")
        )
        valid_step = (
            isinstance(step, torch.Tensor)
            and step.device == expected_step_device
            and step.dtype == torch.float32
            and step.layout == torch.strided
            and step.ndim == 0
            and bool(torch.isfinite(step))
            and float(step) >= 0.0
            and float(step).is_integer()
            and float(step) <= maximum_slot_step
        )
        if not valid_step:
            raise RuntimeError("realized optimizer AdamW step is invalid")
        tensor_slots = ["exp_avg", "exp_avg_sq"]
        if bool(group.get("amsgrad", False)):
            tensor_slots.append("max_exp_avg_sq")
        for name in tensor_slots:
            value = state[name]
            if (not isinstance(value, torch.Tensor)
                    or value.device != parameter.device
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or value.layout != parameter.layout
                    or not bool(torch.isfinite(value).all())):
                raise RuntimeError(f"realized optimizer {name} is incompatible with its parameter")
            if name in {"exp_avg_sq", "max_exp_avg_sq"} and bool((value < 0).any()):
                raise RuntimeError(f"realized optimizer {name} must be nonnegative")


def _optimizer_state_with_live_group_metadata(
    saved_state:        Mapping[str, object],
    optimizer:          torch.optim.Optimizer,
    successful_updates: Optional[int],
) -> Dict[str, object]:
    """Prepare a private load copy whose placement flags come from the live optimizer."""
    # Avoid duplicating the potentially multi-gigabyte CPU moment bundle. PyTorch constructs the
    # realized state mapping during load; only the small group metadata needs an owned copy here.
    state_to_load = dict(saved_state)
    saved_groups = saved_state["param_groups"]
    live_groups: List[Dict[str, object]] = []
    for saved_group, current_group in zip(saved_groups, optimizer.param_groups):
        group = copy.deepcopy({key: value for key, value in current_group.items() if key != "params"})
        group["params"] = list(saved_group["params"])
        if successful_updates is not None:
            group["successful_updates"] = successful_updates
        live_groups.append(group)
    state_to_load["param_groups"] = live_groups
    return state_to_load


def _validate_scaler_state(
    saved_state: object,
    scaler:      'torch.amp.GradScaler',
) -> Mapping[str, object]:
    if not isinstance(saved_state, Mapping):
        raise RuntimeError("checkpoint scaler_state is required for an enabled GradScaler")
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if set(saved_state) != required:
        raise RuntimeError("checkpoint scaler_state has an invalid schema")
    scale = saved_state["scale"]
    growth = saved_state["growth_factor"]
    backoff = saved_state["backoff_factor"]
    interval = saved_state["growth_interval"]
    tracker = saved_state["_growth_tracker"]
    if (not isinstance(scale, Real) or isinstance(scale, bool)
            or not math.isfinite(float(scale)) or float(scale) <= 0.0):
        raise RuntimeError("checkpoint scaler_state scale must be a finite positive real")
    if (not isinstance(growth, Real) or isinstance(growth, bool)
            or not math.isfinite(float(growth)) or float(growth) <= 1.0):
        raise RuntimeError("checkpoint scaler_state growth_factor must be greater than one")
    if (not isinstance(backoff, Real) or isinstance(backoff, bool)
            or not math.isfinite(float(backoff)) or not 0.0 < float(backoff) < 1.0):
        raise RuntimeError("checkpoint scaler_state backoff_factor must lie in (0, 1)")
    if type(interval) is not int or interval <= 0:
        raise RuntimeError("checkpoint scaler_state growth_interval must be a positive integer")
    if type(tracker) is not int or not 0 <= tracker < interval:
        raise RuntimeError(
            "checkpoint scaler_state _growth_tracker must lie in [0, growth_interval)")
    return saved_state


def _validate_rng_state(
    rng_state: object,
) -> Mapping[str, object]:
    if not isinstance(rng_state, Mapping) or "cpu" not in rng_state or "cuda" not in rng_state:
        raise RuntimeError("checkpoint rng_state must contain cpu and cuda states")
    cpu = rng_state["cpu"]
    if (not isinstance(cpu, torch.Tensor) or cpu.device.type != "cpu"
            or cpu.dtype != torch.uint8):
        raise RuntimeError("checkpoint rng_state cpu state must be a CPU ByteTensor")
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except RuntimeError as exc:
        raise RuntimeError("checkpoint rng_state cpu state is invalid") from exc
    cuda = rng_state["cuda"]
    if torch.cuda.is_available() and cuda is None:
        raise RuntimeError("checkpoint rng_state must contain every active CUDA RNG state")
    if cuda is not None:
        if not isinstance(cuda, (list, tuple)):
            raise RuntimeError("checkpoint rng_state cuda states must be a sequence or null")
        if torch.cuda.is_available() and len(cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint rng_state CUDA state count does not match the active device count")
        for index, state in enumerate(cuda):
            if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
                    or state.dtype != torch.uint8):
                raise RuntimeError("checkpoint rng_state CUDA states must be CPU ByteTensors")
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                try:
                    torch.Generator(device=f"cuda:{index}").set_state(state)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"checkpoint rng_state CUDA state {index} is invalid") from exc
    return rng_state


def _validate_generator_state(
    state: object,
    field: str,
) -> torch.Tensor:
    if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
            or state.dtype != torch.uint8):
        raise RuntimeError(f"checkpoint {field} must be a CPU ByteTensor")
    try:
        torch.Generator(device="cpu").set_state(state)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint {field} is invalid") from exc
    return state


def _validate_epoch_generator_state(
    state:         object,
    data_identity: Mapping[str, object],
) -> Optional[torch.Tensor]:
    """Validate the epoch-start generator state against the iterator's sampler semantics."""
    iterator = data_identity.get("iterator")
    if not isinstance(iterator, Mapping):
        raise RuntimeError("checkpoint data identity iterator must be a mapping")
    sampler = iterator.get("sampler")
    if sampler == "random":
        return _validate_generator_state(state, "data_state epoch_start_generator_state")
    if sampler == "sequential":
        if state is not None:
            raise RuntimeError(
                "checkpoint sequential data_state requires a null epoch generator state")
        return None
    raise RuntimeError("checkpoint data identity carries an unsupported sampler")


def _publish_best_model_bundle(
    bundle:               Mapping[str, object],
    expected_model_state: Mapping[str, torch.Tensor],
    artifacts:            'RunArtifacts',
) -> None:
    r"""Revalidate ``bundle`` against the live run and atomically publish it as the run's ``best_model.pt``.

    Writes through a uniquely reserved same-directory temporary and ``os.replace`` so no reader sees a partial
    bundle; revalidates the exact bytes it writes (a byte-for-byte round-trip through
    :func:`_read_best_model_bundle`) so a corrupt in-memory bundle fails closed before it is published.
    The bundle is NEVER loaded into the live training model -- publication only makes the selection
    checkpoint reachable in the new run directory for later finalization."""
    with _unique_sibling_temp(artifacts.best_path) as tmp:
        torch.save(dict(bundle), tmp)
        _read_best_model_bundle(
            tmp,
            artifacts.cfg,
            expected_model_state,
            "cpu",
            artifacts.code_identity_sha256,
            artifacts.selection_data_identity,
        )
        _atomic_replace(artifacts.best_path, tmp)


class RunArtifacts:
    r"""Owns a run directory and its incremental metrics, reports, and optional model files.

    Contract (m25): each instance owns a FRESH run_dir. ``__init__`` (re)writes config.json and the
    first ``log_metrics`` publishes a newly written metrics.csv, so aiming a new instance at a
    populated dir would replace it -- but no path does: resume builds a new timestamped run_dir and
    restores state from a checkpoint FILE via ``load_checkpoint``, never reusing a dir in place."""

    def __init__(
        self,
        run_dir:   'str | Path',
        cfg:       VFE3Config,
        model:     torch.nn.Module,

        *,
        dataset:             str                  = "",
        device:              'str | torch.device' = "cpu",
        save_model_artifacts: bool                 = True,
        timestamp:           Optional[str]        = None,
    ) -> None:
        if type(save_model_artifacts) is not bool:
            raise TypeError("save_model_artifacts must be an exact boolean")
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.save_model_artifacts = save_model_artifacts
        if self.save_model_artifacts:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.best_path = self.run_dir / "best_model.pt"
        self.final_path = self.run_dir / "final_model.pt"
        self.cfg = cfg                                       # kept for figure scaling (lambda_beta) + guards
        # Import-time digest of the code this process actually executes, kept beside the (possibly
        # newer) on-disk digest so provenance can flag a mixed-generation process.
        self.process_code_identity_sha256 = _PROCESS_PACKAGE_CODE_IDENTITY
        self.code_identity_sha256 = _verified_process_code_identity()
        self.run_start_git_identity = dict(_git_code_identity())
        self.selection_data_identity: Optional[Dict[str, object]] = None
        self.scheduler_base_lrs: List[float] = []
        self.scheduler_group_roles: List[str] = []

        self.best_val_ppl: float = float("inf")
        self.best_step: Optional[int] = None
        self.history: List[Dict[str, float]] = []          # in-memory copy of the CSV rows (for figures)
        self._fieldnames: Optional[List[str]] = None

        config_record = {
            "config":    serialize_config(cfg),
            # Structural inert set (field -> reason) BESIDE the config mapping, never inside it:
            # every reader reloads and fingerprints only record["config"], so this key cannot
            # change config reload, semantic fingerprints, or selection projections.
            "inactive_fields": dict(cfg.inactive_fields()),
            "n_params":  int(sum(p.numel() for p in model.parameters())),
            "dataset":   dataset,
            "device":    str(device),
            "timestamp": timestamp,
            "save_model_artifacts": self.save_model_artifacts,
            **_conditional_artifact_fields(cfg),
        }
        executable_build = getattr(model, "executable_build", None)
        if isinstance(executable_build, ExecutableBuildMetadata):
            config_record["executable_build"] = asdict(executable_build)
        self.save_json("config.json", config_record)

    def bind_selection_data_identity(self, identity: Mapping[str, object]) -> None:
        """Bind every selected-weight comparison in this run to one validation data contract."""
        normalized = _normalized_data_identity(identity)
        if (self.selection_data_identity is not None
                and self.selection_data_identity != normalized):
            raise RuntimeError(
                "one RunArtifacts instance cannot compare model selections across validation data")
        self.selection_data_identity = normalized

    def save_json(self, name: str, obj: dict) -> Path:
        r"""Write ``obj`` as pretty JSON to ``run_dir/name`` (non-serializable -> str).

        Atomic: written to a unique same-directory temporary then published via ``os.replace``, so a crash
        mid-write can never leave a truncated/partial JSON at the final name."""
        candidate         = Path(name)
        windows_candidate = PureWindowsPath(name)
        if (not name or name in {".", ".."} or "/" in name or "\\" in name
                or candidate.name != name or candidate.is_absolute() or windows_candidate.drive):
            raise ValueError(f"artifact name must be a regular bare filename, got {name!r}")
        return _write_json_atomic(self.run_dir / name, obj)

    def log_metrics(self, row: Dict[str, float]) -> None:
        r"""Atomically publish ``metrics.csv`` with one additional rectangular row.

        The column set is fixed by the first row; later rows must share those keys so the CSV
        stays rectangular (the training loop emits a homogeneous row each periodic eval).

        NaN cells are written to the file as BLANK (empty string), so an eval-cadence column
        (val_*, generalization_gap, the held-out probes) -- NaN on the denser log-interval rows
        between evals -- shows an empty cell rather than a repeated value or a literal "nan".
        The IN-MEMORY ``self.history`` keeps the raw NaN float so
        the figure pass (which filters on ``math.isfinite``) is unaffected."""
        fieldnames = list(row) if self._fieldnames is None else list(self._fieldnames)
        if set(row) != set(fieldnames):
            missing = sorted(set(fieldnames) - set(row))
            extra = sorted(set(row) - set(fieldnames))
            raise ValueError(
                f"metrics row field set changed (missing={missing}, extra={extra})"
            )
        candidate_history = [*self.history, dict(row)]
        with _unique_sibling_temp(self.csv_path) as tmp:
            with tmp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for history_row in candidate_history:
                    writer.writerow({
                        key: ("" if isinstance(value, float) and math.isnan(value) else value)
                        for key, value in history_row.items()
                    })
            _atomic_replace(self.csv_path, tmp)
        self._fieldnames = fieldnames
        self.history.append(dict(row))                          # update only after durable publication

    def maybe_save_best(self, step: int, model: torch.nn.Module, val_ppl: float) -> bool:
        r"""Record a new minimum and, when enabled, save its config-bound weights.

        Atomic (same-dir tmp + ``os.replace``): a crash or Windows lock mid-save can never leave a
        corrupt/unreadable ``best_model.pt`` where a good one stood."""
        if val_ppl < self.best_val_ppl:
            self.best_val_ppl = float(val_ppl)
            self.best_step = int(step)
            if not self.save_model_artifacts:
                return True
            config = serialize_config(self.cfg)
            bundle = {
                "model_state":        model.state_dict(),
                "config":             config,
                "config_fingerprint": semantic_config_fingerprint(config),
                "code_identity_sha256": self.code_identity_sha256,
                "selection_data_identity": self.selection_data_identity,
            }
            with _unique_sibling_temp(self.best_path) as tmp:
                torch.save(bundle, tmp)
                _atomic_replace(self.best_path, tmp)
            return True
        return False

    def save_attention_maps(
        self,
        step:   int,
        maps:   torch.Tensor,                 # (L, H, N, N) per-layer per-head attention
        logger: Optional[logging.Logger] = None,
    ) -> Optional[List[Path]]:
        r"""Best-effort attention heatmaps for one periodic eval: one figure per (layer, head).

        Writes ``attention/step_<N>_layer<l>_head<h>.png`` per (layer, head) -- a LOG-scaled beta
        heatmap (see :func:`vfe3.viz.figures.plot_attention_heatmap`) on a color scale shared
        across panels so heads/layers stay comparable. Mirrors ``_save_figures``: a plotting or
        dependency error is logged and swallowed (never fatal to the run), and each figure is
        closed so ~30 evals do not leak figures. Returns the paths written, or None on failure.
        """
        return _run_attention_maps_isolated(
            maps,
            self.run_dir,
            logger or logging.getLogger(__name__),
            channel="beta",
            step=step,
        )

    def save_estep_attention_maps(
        self,
        step:   int,
        maps:   'Optional[torch.Tensor]',     # (3, L, H, N, N) entry / converged / prior, or None
        logger: Optional[logging.Logger] = None,
    ) -> Optional[List[Path]]:
        r"""The attention the E-step ACTUALLY used, written beside the shipped maps.

        Writes ``attention_estep/step_<N>_layer<l>_head<h>.png`` (the ENTRY-belief beta -- the
        pattern the E-step's first iteration aggregated with) plus one
        ``attention_estep/step_<N>_profile.png`` per step: row-mean beta against key distance
        ``|i - j|``, with the converged re-score and the attention prior overlaid on the same axes.

        The shipped ``attention/`` heatmaps score at each block's OUTPUT belief, after the E-step
        has moved it and after ``head_mixer``/``block_norm`` -- a post-hoc re-score, not the
        pattern that produced the aggregation (audit 2026-07-26 R-3, measured max row difference
        1.0 on a trained checkpoint). Both are kept: the converged map is what ``diagnostics``
        reports, so it stays the reference, and this is the mechanism.

        The profile panel is the one that makes positional pathology legible -- a rotary schedule
        whose kernel revives at long lag shows up as off-diagonal bumps here that are invisible in
        a log-scaled heatmap, and a beta that has collapsed onto its prior shows up as the two
        curves lying on top of each other. ``maps`` is None -> no-op. Errors are logged and
        swallowed, never fatal to the run.
        """
        if maps is None:
            return None
        return _run_attention_maps_isolated(
            maps,
            self.run_dir,
            logger or logging.getLogger(__name__),
            channel="estep",
            step=step,
        )

    def save_checkpoint(
        self,
        step:      int,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg:       VFE3Config,

        *,
        scaler:                     Optional['torch.amp.GradScaler'] = None,
        data_state:                  Optional[DataState]               = None,
    ) -> Path:
        r"""Write a resumable ``checkpoints/step_<N>.pt`` (model + optimizer + RNG + config + step).

        ``load_checkpoint`` reads this back to continue training: ``model_state`` and
        ``optimizer_state`` restore the weights, AdamW momentum, and successful-update scheduler
        clock; ``rng_state`` restores the
        CPU (and CUDA) generators for reproducible continuation, and ``step`` is the number of
        completed M-steps so the resumed run rebuilds the cosine ``LambdaLR`` at the right point.
        ``scaler`` (audit 2026-06-09 IE3): an ENABLED fp16 GradScaler's state (current scale +
        growth counters) is bundled so a resumed fp16 run does not restart at the init scale
        65536 and re-converge by skipped steps; a disabled/None scaler stores None.
        ``best_val_ppl``/``best_step`` (audit 2026-07-01 C2): the model-selection state is bundled
        so a resumed run reports the run-wide best, not just the continuation's best. The write is
        atomic (same-dir tmp + ``os.replace``) so a crash never leaves a corrupt ``step_<N>.pt``.
        ``data_state`` records the current epoch's cursor and data/iterator identity. Its optional
        generator state is cloned for random sampling; deterministic sequential sampling stores null.
        """
        if not self.save_model_artifacts:
            raise RuntimeError("model-artifact persistence is disabled for this run")
        step = _require_nonnegative_int(step, "step")
        path = self.ckpt_dir / f"step_{step}.pt"
        rng_state = {
            "cpu":  torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        saved_data_state = None
        if data_state is not None:
            batches_consumed = _require_nonnegative_int(
                data_state["batches_consumed"], "batches_consumed")
            epoch = _require_nonnegative_int(data_state["epoch"], "epoch")
            generator_state = data_state["epoch_start_generator_state"]
            if generator_state is not None and not isinstance(generator_state, torch.Tensor):
                raise ValueError(
                    "data_state epoch_start_generator_state must be a tensor or null")
            data_identity = _normalized_data_identity(data_state.get("data_identity"))
            generator_state = _validate_epoch_generator_state(generator_state, data_identity)
            saved_data_state = {
                "epoch_start_generator_state": (generator_state.clone()
                                                  if generator_state is not None else None),
                "batches_consumed":            batches_consumed,
                "epoch":                       epoch,
                "data_identity":               data_identity,
            }
        # Portable best-model selection state (PB-03): a finite best_val_ppl means best_model.pt IS the
        # selected checkpoint, so its validated bundle is embedded here and travels with the checkpoint
        # across a cross-run-directory resume (older bundles carried only the scalar, whose file was left
        # behind). A finite best scalar without a readable, matching best_model.pt is an integrity error
        # and MUST prevent checkpoint publication; no validation best -> explicit empty selection state.
        if math.isfinite(float(self.best_val_ppl)):
            if not self.best_path.is_file():
                raise RuntimeError(
                    f"finite best_val_ppl ({float(self.best_val_ppl)}) but no readable best_model.pt "
                    f"at {self.best_path}; cannot embed a portable best-model bundle")
            best_model_bundle: Optional[Dict[str, object]] = _read_best_model_bundle(
                self.best_path,
                cfg,
                model.state_dict(),
                "cpu",
                self.code_identity_sha256,
                self.selection_data_identity,
            )
            saved_best_val_ppl = float(self.best_val_ppl)
            saved_best_step    = self.best_step
        else:
            best_model_bundle  = None
            saved_best_val_ppl = float("inf")
            saved_best_step    = None
        optimizer_state = optimizer.state_dict()
        optimizer_parameter_manifest = _optimizer_parameter_manifest(
            model, optimizer, optimizer_state,
        )
        with _unique_sibling_temp(path) as tmp:
            torch.save({
                "step":            step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer_state,
                "optimizer_parameter_manifest": optimizer_parameter_manifest,
                "rng_state":       rng_state,
                "config":          serialize_config(cfg),
                "code_identity_sha256": self.code_identity_sha256,
                "git_identity":    _git_code_identity(),
                "selection_data_identity": self.selection_data_identity,
                "scaler_state":    (scaler.state_dict()
                                    if scaler is not None and scaler.is_enabled() else None),
                "best_val_ppl":    saved_best_val_ppl,
                "best_step":       saved_best_step,
                "best_model_bundle": best_model_bundle,
                "data_state":      saved_data_state,
            }, tmp)
            _atomic_replace(path, tmp)
        return path


def _restore_best_selection(
    ckpt:                 Mapping[str, object],
    checkpoint_path:      Path,
    artifacts:            'RunArtifacts',
    expected_model_state: Mapping[str, torch.Tensor],
    validated_selection:  Optional[Mapping[str, object]],
) -> None:
    r"""Publish the owned selected snapshot captured before any live checkpoint restoration."""
    import warnings

    ckpt_best_ppl = ckpt.get("best_val_ppl")
    had_finite_best = ckpt_best_ppl is not None and math.isfinite(float(ckpt_best_ppl))

    if validated_selection is None:
        artifacts.best_val_ppl = float("inf")
        artifacts.best_step = None
        if had_finite_best:
            warnings.warn(
                f"resume from {checkpoint_path.name} carried a selected model measured under a "
                f"different or unbound code/config/validation-data contract; raw training state "
                f"was restored, but model selection restarts from this run.",
                UserWarning,
                stacklevel=3,
            )
        return

    _publish_best_model_bundle(validated_selection, expected_model_state, artifacts)
    artifacts.best_val_ppl = float(ckpt_best_ppl)
    artifacts.best_step    = ckpt.get("best_step")
    if not artifacts.best_path.is_file():
        raise RuntimeError(
            "best-model bundle is not reachable after publication into the resumed run")


def _preflight_best_selection(
    ckpt:                 Mapping[str, object],
    checkpoint_path:      Path,
    cfg:                  Optional[VFE3Config],
    expected_model_state: Mapping[str, torch.Tensor],
    map_location:         'str | torch.device',
    saved_step:           int,
    expected_code_identity: str,
    expected_selection_data_identity: Optional[Mapping[str, object]],
) -> Optional[Dict[str, object]]:
    """Return one owned selected snapshot validated before live restoration, or ``None``."""
    if "best_val_ppl" not in ckpt or "best_step" not in ckpt:
        raise RuntimeError("checkpoint best-model selection metadata is missing")
    best_ppl = ckpt["best_val_ppl"]
    best_step = ckpt["best_step"]
    embedded_present = "best_model_bundle" in ckpt
    embedded = ckpt.get("best_model_bundle")
    if (not isinstance(best_ppl, Real) or isinstance(best_ppl, bool)
            or math.isnan(float(best_ppl)) or float(best_ppl) == float("-inf")):
        raise RuntimeError("checkpoint best-model selection PPL must be finite or positive infinity")
    selected = math.isfinite(float(best_ppl))
    candidate: Optional[Dict[str, object]] = None
    if selected:
        if float(best_ppl) <= 0.0:
            raise RuntimeError("checkpoint best-model selection PPL must be positive")
        if type(best_step) is not int or best_step < 0 or best_step > saved_step:
            raise RuntimeError(
                "checkpoint best-model selection step must be a non-negative integer no later "
                "than the checkpoint step")
        try:
            if embedded_present:
                if not isinstance(embedded, Mapping):
                    raise RuntimeError(
                        "checkpoint best-model selection has no reachable embedded weight bundle")
                candidate = _validate_best_model_mapping(
                    embedded,
                    cfg,
                    expected_model_state,
                    f"checkpoint {checkpoint_path} best-model selection bundle",
                    expected_code_identity=expected_code_identity,
                    expected_selection_data_identity=expected_selection_data_identity,
                )
            else:
                sibling = checkpoint_path.parent.parent / "best_model.pt"
                if not sibling.is_file():
                    raise RuntimeError(
                        "checkpoint best-model selection has no reachable legacy sibling weights")
                bundle = torch.load(sibling, map_location=map_location, weights_only=True)
                candidate = _validate_best_model_mapping(
                    bundle,
                    cfg,
                    expected_model_state,
                    f"checkpoint {checkpoint_path} legacy best-model selection bundle",
                    expected_code_identity=expected_code_identity,
                    expected_selection_data_identity=expected_selection_data_identity,
                )
        except _BestModelContractMismatch:
            return None
    elif best_step is not None or embedded is not None:
        raise RuntimeError(
            "checkpoint empty best-model selection must use best_step=None and no weight bundle")
    if not selected or candidate is None:
        return None
    if expected_selection_data_identity is None:
        return None
    checkpoint_code_identity = ckpt.get("code_identity_sha256")
    checkpoint_selection_identity = ckpt.get("selection_data_identity")
    if (checkpoint_code_identity != candidate.get("code_identity_sha256")
            or checkpoint_selection_identity is None):
        return None
    try:
        if (_normalized_data_identity(checkpoint_selection_identity)
                != _normalized_data_identity(candidate["selection_data_identity"])):
            return None
    except (TypeError, ValueError):
        return None
    return candidate


def _preflight_resume_config(
    saved_config: object,
    cfg:          VFE3Config,
) -> Tuple[dict, List[str]]:
    """Validate serialized config keys and compute deterministic drift before any live mutation."""
    if not isinstance(saved_config, Mapping):
        raise RuntimeError("checkpoint config must be a mapping when cfg is supplied")
    if any(not isinstance(key, str) for key in saved_config):
        raise RuntimeError("checkpoint config keys must be strings")
    try:
        parsed = config_from_serialized(saved_config, source="checkpoint resume config")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    raw_config = dict(saved_config)
    effective = asdict(parsed)
    current = asdict(cfg)
    drift = sorted(
        key for key in (set(effective) | set(current))
        if key not in ("resume_from", "trust_resume_checkpoint")
        and effective.get(key) != current.get(key)
    )
    return raw_config, drift


def load_checkpoint(
    path:      'str | Path',
    model:     torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,

    *,
    map_location:         'Optional[str | torch.device]'   = None,
    max_step:             Optional[int]                    = None,
    restore_rng:          bool                             = True,
    scaler:               Optional['torch.amp.GradScaler'] = None,
    cfg:                  Optional[VFE3Config]             = None,
    artifacts:            'Optional[RunArtifacts]'         = None,
    data_state:                  Optional[DataStateBuffer]          = None,
    expected_data_identity: Optional[Mapping[str, object]]   = None,
    expected_selection_data_identity: Optional[Mapping[str, object]] = None,
    expected_steps_per_epoch: Optional[int]                  = None,
) -> int:
    r"""Restore a ``save_checkpoint`` bundle into ``model`` (and optionally ``optimizer``); return the saved step.

    This is the LOAD half of the resumable checkpoint. It always restores the model weights;
    it restores the AdamW optimizer state (momentum buffers + per-parameter step counts) when an
    ``optimizer`` is supplied, then reapplies that optimizer's current non-parameter group metadata
    so the current config remains authoritative. The CPU/CUDA RNG is restored when ``restore_rng`` is
    set and the bundle carries it (older checkpoints simply skip absent RNG fields). The returned integer is the number of
    completed M-steps; ``train(resume_from=...)`` uses it to rebuild the cosine ``LambdaLR`` at the
    saved step and to start the loop from there.


    ``scaler`` (audit 2026-06-09 IE3): when given AND the bundle carries a saved scaler state,
    the fp16 GradScaler's scale/growth counters are restored (bundles written before the scaler
    was persisted, or written from a non-fp16 run, simply skip the step). ``cfg`` (audit IE4):
    when given, the CURRENT config is compared against the bundle's saved config and any
    differing fields are warned about -- strict ``load_state_dict`` already catches
    shape-changing divergence, but shape-preserving semantic drift (LR schedule, n_e_steps,
    e_*_lr, ...) would otherwise pass silently. ``artifacts`` (audit 2026-07-01 C2, PB-03): when
    given, the portable best-model selection state is restored into it -- a modern checkpoint's
    embedded, validated ``best_model_bundle`` (or, for a legacy checkpoint, the sibling
    ``<old_run>/best_model.pt``) is published into the resumed run's directory and the
    ``best_val_ppl``/``best_step`` scalars are set; when no reachable bundle exists the selection state
    resets to ``inf``/``None`` with one warning, so an unreachable best scalar is never retained. When a
    mutable ``data_state`` mapping is supplied, it is filled from the bundled iterator cursor.
    Supplying ``expected_data_identity`` requests exact data resume: a legacy bundle without
    ``data_state`` or a bundle whose identity differs is rejected before any mutable state is restored.

    The bundle is loaded with ``weights_only=True`` by default, which refuses to execute arbitrary
    pickle reductions: our bundle carries only tensors, an ``asdict`` config dict, and RNG tensors
    (no custom classes), so it loads safely under it (matching
    ``test_run_artifacts.py::test_save_checkpoint_is_loadable``). A bundle that fails the safe load
    (e.g. an older format) RAISES unless ``cfg.trust_resume_checkpoint`` is set, which falls back to
    the legacy ``weights_only=False`` load -- only use that for a checkpoint you trust, since that
    path can execute arbitrary code embedded in the pickle.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    if map_location is None:
        map_location = next(model.parameters()).device
    # A checkpoint mixes parameter tensors with CPU-only state: torch CPU RNG, DataLoader
    # iterator states, and non-fused/non-capturable Adam step scalars. Loading the whole
    # bundle onto a CUDA target corrupts those storage contracts before validation. Validate from a
    # CPU snapshot; model.load_state_dict and optimizer.load_state_dict perform the deliberate
    # parameter-slot transfer to the live model device below.
    checkpoint_load_location = "cpu"
    trust = bool(getattr(cfg, "trust_resume_checkpoint", False))
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location=checkpoint_load_location,
            weights_only=True,
        )
    except Exception as exc:                                    # safe load rejected a non-tensor object
        if not trust:
            raise RuntimeError(
                f"checkpoint {Path(path).name} could not be loaded under the safe weights_only=True "
                f"path ({type(exc).__name__}: {exc}). If you trust this file, set "
                f"trust_resume_checkpoint=True to allow the legacy weights_only=False load (which can "
                f"execute arbitrary code embedded in the pickle)."
            ) from exc
        ckpt = torch.load(
            checkpoint_path,
            map_location=checkpoint_load_location,
            weights_only=False,
        )
    if not isinstance(ckpt, Mapping):
        raise RuntimeError("checkpoint payload must be a mapping")
    saved_step = _require_nonnegative_int(ckpt.get("step"), "step")
    if max_step is not None:
        max_step = _require_nonnegative_int(max_step, "max_step")
        if saved_step > max_step:
            raise ValueError(
                f"checkpoint step {saved_step} exceeds requested n_steps={max_step} "
                f"(max_step={max_step})")
    saved_data_state = ckpt.get("data_state")
    saved_data_identity: Optional[Dict[str, object]] = None
    if saved_data_state is None and expected_data_identity is not None:
        raise RuntimeError(
            "checkpoint is missing data_state required for exact resume; refusing to restore "
            "model, optimizer, cursor, or RNG state")
    if saved_data_state is not None:
        if not isinstance(saved_data_state, Mapping):
            raise RuntimeError("checkpoint data_state must be a mapping")
        saved_batches_consumed = _require_nonnegative_int(
            saved_data_state["batches_consumed"], "batches_consumed")
        saved_epoch = _require_nonnegative_int(saved_data_state["epoch"], "epoch")
        saved_generator_state = saved_data_state.get("epoch_start_generator_state")
        if (saved_generator_state is not None
                and not isinstance(saved_generator_state, torch.Tensor)):
            raise RuntimeError(
                "checkpoint data_state epoch_start_generator_state must be a tensor or null")
        if "data_identity" not in saved_data_state:
            raise RuntimeError(
                "checkpoint data_state is missing its data identity contract; exact resume is unsafe")
        saved_data_identity = _normalized_data_identity(saved_data_state["data_identity"])
        if expected_data_identity is None:
            raise RuntimeError(
                "checkpoint carries a data cursor, but no live expected data identity was supplied")
        live_data_identity = _normalized_data_identity(expected_data_identity)
        if saved_data_identity != live_data_identity:
            differing = sorted(
                key for key in (saved_data_identity.keys() | live_data_identity.keys())
                if saved_data_identity.get(key) != live_data_identity.get(key)
            )
            raise RuntimeError(
                f"checkpoint data identity mismatch for field(s) {differing}; refusing to restore "
                "model, optimizer, cursor, or RNG state")
        saved_generator_state = _validate_epoch_generator_state(
            saved_generator_state, live_data_identity)
        if expected_steps_per_epoch is None:
            iterator_identity = live_data_identity["iterator"]
            n_samples = iterator_identity["sampler_num_samples"]
            batch_size = iterator_identity["batch_size"]
            expected_steps_per_epoch = (
                n_samples // batch_size
                if iterator_identity["drop_last"]
                else (n_samples + batch_size - 1) // batch_size
            )
        if type(expected_steps_per_epoch) is not int or expected_steps_per_epoch <= 0:
            raise RuntimeError("exact resume requires a positive integer steps-per-epoch")
        if saved_batches_consumed > expected_steps_per_epoch:
            raise RuntimeError(
                "checkpoint batches_consumed exceeds the live loader epoch length")
        expected_step = saved_epoch * expected_steps_per_epoch + saved_batches_consumed
        if saved_step != expected_step:
            raise RuntimeError(
                f"checkpoint step {saved_step} is inconsistent with epoch {saved_epoch} and "
                f"batches_consumed {saved_batches_consumed} for "
                f"{expected_steps_per_epoch} steps per epoch")
    raw_model_state = ckpt.get("model_state")
    live_model_state = model.state_dict()
    saved_model_state = _validate_checkpoint_model_state(
        raw_model_state, live_model_state, checkpoint_path)
    saved_config = None
    config_drift: List[str] = []
    resume_cfg = cfg if cfg is not None else (artifacts.cfg if artifacts is not None else None)
    if resume_cfg is not None:
        saved_config, config_drift = _preflight_resume_config(
            ckpt.get("config"), resume_cfg,
        )
    else:
        raw_checkpoint_config = ckpt.get("config")
        if not isinstance(raw_checkpoint_config, Mapping):
            raise RuntimeError("checkpoint config must be a mapping for exact resume")
        if any(not isinstance(key, str) for key in raw_checkpoint_config):
            raise RuntimeError("checkpoint config keys must be strings")
        try:
            config_from_serialized(raw_checkpoint_config, source="checkpoint exact resume config")
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        saved_config = dict(raw_checkpoint_config)
    if ckpt.get("ema_state") is not None:
        raise RuntimeError(
            "checkpoint carries retired non-null ema_state; exact resume requires the removed EMA "
            "runtime. Import model weights explicitly to start a new non-EMA run"
        )
    expected_code_identity = (
        artifacts.code_identity_sha256
        if artifacts is not None
        else _verified_process_code_identity()
    )
    saved_optimizer_state = None
    successful_updates = None
    if optimizer is not None:
        checkpoint_optimizer_state = ckpt.get("optimizer_state")
        if "optimizer_parameter_manifest" in ckpt:
            _validate_optimizer_parameter_manifest(
                ckpt.get("optimizer_parameter_manifest"),
                model,
                optimizer,
            )
        else:
            saved_code_identity = ckpt.get("code_identity_sha256")
            if (resume_cfg is None or config_drift
                    or not isinstance(saved_code_identity, str)
                    or saved_code_identity != expected_code_identity):
                raise RuntimeError(
                    "legacy optimizer checkpoint has no parameter manifest and cannot establish an "
                    "unchanged source/config execution contract; load model weights without an "
                    "optimizer or restart with a fresh optimizer"
                )
        saved_optimizer_state = _validate_optimizer_state(
            checkpoint_optimizer_state, optimizer, saved_step,
        )
        successful_updates = _validated_saved_successful_updates(
            saved_optimizer_state["param_groups"], saved_step)
    saved_scaler_state = ckpt.get("scaler_state")
    if scaler is not None and scaler.is_enabled():
        saved_scaler_state = _validate_scaler_state(saved_scaler_state, scaler)
    saved_rng_state = None
    if restore_rng:
        saved_rng_state = _validate_rng_state(ckpt.get("rng_state"))
    active_selection_cfg = cfg if cfg is not None else (
        artifacts.cfg if artifacts is not None else None)
    if expected_selection_data_identity is None and artifacts is not None:
        expected_selection_data_identity = artifacts.selection_data_identity
    validated_selection = _preflight_best_selection(
        ckpt,
        checkpoint_path,
        active_selection_cfg,
        model.state_dict(),
        map_location,
        saved_step,
        expected_code_identity,
        expected_selection_data_identity,
    )
    if optimizer is not None:
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        state_to_load = _optimizer_state_with_live_group_metadata(
            saved_optimizer_state, optimizer, successful_updates,
        )
        try:
            optimizer.load_state_dict(state_to_load)
            _validate_realized_optimizer_state(optimizer, saved_step, successful_updates)
        except Exception:
            optimizer.load_state_dict(optimizer_snapshot)
            raise
        del optimizer_snapshot
    model.load_state_dict(saved_model_state)
    if scaler is not None and scaler.is_enabled():
        scaler.load_state_dict(saved_scaler_state)
    # Portable best-val model-selection state (PB-03, extends audit 2026-07-01 C2): restore
    # best_val_ppl/best_step AND make the selected weights reachable in the resumed run's directory,
    # so a cross-run_dir resume no longer restores best metadata whose best_model.pt is missing. The
    # scalar is never retained without reachable weights (audit m26). By precedence:
    #   (1) a modern checkpoint's embedded best_model_bundle is validated and published into the new run;
    #   (2) a legacy checkpoint (no such field) validates <old_run>/best_model.pt and publishes it;
    #   (3) neither -> the selection state resets to empty, and any unreachable finite scalar is dropped.
    if artifacts is not None:
        _restore_best_selection(
            ckpt,
            checkpoint_path,
            artifacts,
            model.state_dict(),
            validated_selection,
        )
    if resume_cfg is not None:
        if config_drift:
            import warnings
            other_drift = [field for field in config_drift if field != "seed"]
            message = (
                f"resume config drift: the checkpoint at {Path(path).name} was written under a "
                f"different config for field(s) {config_drift}"
            )
            if other_drift:
                message += (
                    f"; {other_drift} use the CURRENT values (weights/optimizer load strictly, but "
                    f"semantic knobs are not restored from the bundle)"
                )
            if "seed" in config_drift:
                # The seed acts only through initialization and the random stream. Initialization
                # is overwritten by the restored weights, so the new seed reaches the run only
                # through an RNG state that is not restored.
                message += (
                    "; 'seed' is NOT applied: the checkpoint's weights and RNG state are restored, "
                    "so the continuation follows the checkpoint's random stream and the new seed has "
                    "no effect on it"
                    if restore_rng and saved_rng_state is not None else
                    "; 'seed' is only partly applied: the checkpoint's weights are restored (the new "
                    "seed's initialization is discarded) and only the unrestored global RNG stream "
                    "derives from the current seed"
                )
            warnings.warn(message + ".", UserWarning, stacklevel=2)
    if restore_rng:
        rng = saved_rng_state
        # RNG tensors must be CPU ByteTensors regardless of map_location (set_rng_state asserts this).
        torch.set_rng_state(rng["cpu"].cpu() if hasattr(rng["cpu"], "cpu") else rng["cpu"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    if data_state is not None:
        data_state.clear()
        if saved_data_state is not None:
            data_state.update({
                "epoch_start_generator_state": saved_data_state["epoch_start_generator_state"],
                "batches_consumed":            saved_batches_consumed,
                "epoch":                       saved_epoch,
                "data_identity":               saved_data_identity,
            })
    if artifacts is not None:
        saved_code_identity = ckpt.get("code_identity_sha256")
        code_identity_match = (
            saved_code_identity == expected_code_identity
            if isinstance(saved_code_identity, str)
            else None
        )
        saved_git_identity = ckpt.get("git_identity")
        current_git_identity = dict(artifacts.run_start_git_identity)
        if isinstance(saved_git_identity, Mapping):
            git_identity_status, git_identity_drift = _git_identity_comparison(
                saved_git_identity,
                current_git_identity,
            )
            saved_git_record: Optional[Dict[str, object]] = dict(saved_git_identity)
        else:
            git_identity_status, git_identity_drift = "unverifiable", None
            saved_git_record = None
        artifacts.save_json("resume_provenance.json", {
            "schema_version":                  1,
            "resume_mode":                     "raw_state_resume",
            "checkpoint_path":                 str(checkpoint_path.resolve()),
            "saved_step":                      saved_step,
            "config_restore_policy":           "current_config_authoritative",
            "config_drift_fields":             list(config_drift),
            "saved_config_fingerprint":        (
                semantic_config_fingerprint(saved_config)
                if isinstance(saved_config, Mapping) else None
            ),
            "current_config_fingerprint":      (
                semantic_config_fingerprint(serialize_config(resume_cfg))
                if resume_cfg is not None else None
            ),
            "saved_code_identity_sha256":      saved_code_identity,
            "current_code_identity_sha256":    expected_code_identity,
            "code_identity_match":             code_identity_match,
            "saved_git_identity":              saved_git_record,
            "current_git_identity":            current_git_identity,
            "git_identity_status":             git_identity_status,
            "git_identity_drift":              git_identity_drift,
        })
    return saved_step


def _git_environment(
    git_executable: str,
) -> Dict[str, str]:
    r"""Minimal noninteractive environment for bounded Git provenance probes."""
    env = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": str(Path(git_executable).resolve().parent),
    }
    for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _git_code_identity(
    root: Optional[Path] = None,
) -> Dict[str, object]:
    r"""Return HEAD plus an exact dirty-tree fingerprint, or a persisted probe error."""
    repo = Path(__file__).resolve().parent.parent if root is None else Path(root).resolve()
    identity: Dict[str, object] = {
        "git_sha":               None,
        "git_dirty":             None,
        "git_dirty_fingerprint": None,
    }
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise FileNotFoundError("git executable was not found on PATH")
        env = _git_environment(git_executable)

        def _git(*args: str) -> bytes:
            return subprocess.check_output(
                [git_executable,
                 "-c", "core.fsmonitor=false",
                 "-c", f"safe.directory={repo.as_posix()}",
                 *args],
                cwd=str(repo),
                env=env,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )

        identity["git_sha"] = _git("rev-parse", "HEAD").decode("ascii").strip()
        status = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        identity["git_dirty"] = bool(status)
        if status:
            diff = _git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")
            untracked = _git("ls-files", "--others", "--exclude-standard", "-z")
            digest = hashlib.sha256()
            digest.update(b"status\0")
            digest.update(status)
            digest.update(b"\0diff\0")
            digest.update(diff)
            digest.update(b"\0untracked\0")
            for raw_name in (name for name in untracked.split(b"\0") if name):
                path = repo / os.fsdecode(raw_name)
                digest.update(raw_name)
                digest.update(b"\0")
                digest.update(bytes.fromhex(_sha256_file_content(path)))
            identity["git_dirty_fingerprint"] = digest.hexdigest()
    except Exception as exc:
        identity["git_sha"] = None
        identity["git_dirty"] = None
        identity["git_dirty_fingerprint"] = None
        identity["git_error"] = repr(exc)
    return identity


def _complete_git_identity(value: Mapping[str, object]) -> bool:
    """Return whether a Git probe produced a comparable source-tree identity."""
    git_sha = value.get("git_sha")
    dirty = value.get("git_dirty")
    fingerprint = value.get("git_dirty_fingerprint")
    sha_is_valid = (
        isinstance(git_sha, str)
        and len(git_sha) in (40, 64)
        and all(character in "0123456789abcdefABCDEF" for character in git_sha)
    )
    fingerprint_is_valid = (
        fingerprint is None
        if dirty is False
        else (
            isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and all(character in "0123456789abcdefABCDEF" for character in fingerprint)
        )
    )
    return (
        sha_is_valid
        and type(dirty) is bool
        and fingerprint_is_valid
        and "git_error" not in value
    )


def _git_identity_comparison(
    run_start: Mapping[str, object],
    final:     Mapping[str, object],
) -> Tuple[str, Optional[bool]]:
    """Classify two bounded Git probes as stable, drifted, or unverifiable."""
    if not (_complete_git_identity(run_start) and _complete_git_identity(final)):
        return "unverifiable", None
    fields = ("git_sha", "git_dirty", "git_dirty_fingerprint")
    drifted = any(run_start.get(field) != final.get(field) for field in fields)
    return ("drifted" if drifted else "stable"), drifted


def _sha256_file_content(
    path: Path,

    *,
    chunk_bytes: int = 1024 * 1024,
) -> str:
    """Hash a file without materializing it in memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor_content(
    tokens: torch.Tensor,

    *,
    chunk_tokens: int = 128 * 1024,
) -> str:
    r"""Hash token values canonically as int64 using bounded device-to-host chunks."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    flat = tokens.detach().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_tokens):
        chunk = flat[start:start + chunk_tokens].to(device="cpu", dtype=torch.long).contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


_PROCESS_PACKAGE_CODE_IDENTITY = _package_code_identity()


# STRICT_CODE_IDENTITY: when True, a source edit made after this process imported the package is a
# hard error, and every ablation cell trained under a different source generation is refused for
# reuse. That is the reproducibility-pure path: a run is labeled only with a generation it provably
# ran under. It is default OFF because the digest covers whole-file content across the package and
# the root drivers, so an edit that cannot affect a trained number (a figure driver, an unrelated
# module, a checkout that rewrites bytes) still invalidates a finished ablation cohort and forces it
# to retrain. With gating off the digest is still recorded descriptively in provenance; it simply no
# longer aborts a run or authorizes destroying a completed cell.
STRICT_CODE_IDENTITY: bool = False


def strict_code_identity_enabled() -> bool:
    """Read the gating toggle live so a test can monkeypatch the module attribute."""
    return bool(STRICT_CODE_IDENTITY)


def _verified_process_code_identity() -> str:
    r"""Return the import-time identity, or under strict gating fail when disk source drifts.

    Python keeps imported module objects and bytecode alive after an editor changes their files.
    Relabeling such a mixed process with the new disk digest would be false provenance, so under
    ``STRICT_CODE_IDENTITY`` a new run in a long-lived Spyder kernel stops until that kernel restarts
    and imports one coherent source generation. With gating off the current disk digest is returned
    instead: provenance still records what the source looked like at run start, a warning is logged,
    and ``RunArtifacts`` keeps the import-time digest so ``_write_provenance`` reports
    ``source_identity_status='mixed_generation'`` rather than ``'stable'``.
    """
    current = _package_code_identity()
    if current != _PROCESS_PACKAGE_CODE_IDENTITY:
        if strict_code_identity_enabled():
            raise RuntimeError(
                "vfe3 source changed after this Python process imported the package; restart the "
                "Spyder kernel before starting another run"
            )
        # Non-strict: keep running, but never silently. RunArtifacts also keeps the import-time
        # digest so provenance reports source_identity_status='mixed_generation', not 'stable'.
        logging.getLogger(__name__).warning(
            "vfe3 source changed after this process imported the package: the on-disk code "
            "identity is recorded, but the process may execute an older, mixed source generation"
        )
        return current
    return _PROCESS_PACKAGE_CODE_IDENTITY


def _require_token_chunk_in_vocab(
    chunk:      torch.Tensor,
    vocab_size: int,
) -> None:
    """Reject invalid token IDs before bincount can allocate from an out-of-vocabulary maximum."""
    if chunk.numel() == 0:
        return
    minimum, maximum = torch.aminmax(chunk)
    if int(minimum) < 0 or int(maximum) >= vocab_size:
        raise ValueError(
            f"training token id is outside [0, {vocab_size}); validate the cache first"
        )


def _scan_token_content(
    tokens: torch.Tensor,

    *,
    vocab_size:   int,
    chunk_tokens: int = 128 * 1024,
) -> Tuple[str, torch.Tensor]:
    r"""Compute the canonical digest and unigram counts in one bounded corpus pass."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    flat = tokens.detach().reshape(-1)
    digest = hashlib.sha256()
    counts = torch.zeros(vocab_size, dtype=torch.long, device="cpu")
    for start in range(0, flat.numel(), chunk_tokens):
        chunk = flat[start:start + chunk_tokens].to(device="cpu", dtype=torch.long).contiguous()
        digest.update(chunk.numpy().tobytes())
        _require_token_chunk_in_vocab(chunk, vocab_size)
        partial = torch.bincount(chunk, minlength=vocab_size)
        counts.add_(partial)
    return digest.hexdigest(), counts


def _loader_token_content_summary(
    loader: Optional[Iterable],

    *,
    vocab_size: Optional[int] = None,
) -> 'Tuple[Optional[str], Optional[int], Optional[torch.Tensor]]':
    r"""Return cached corpus identity and optional counts, scanning both together when needed."""
    dataset = getattr(loader, "dataset", None)
    tokens = getattr(dataset, "tokens", None)
    if tokens is None:
        return None, None, None
    cached_sha = getattr(dataset, "_vfe3_token_content_sha256", None)
    cached_counts = getattr(dataset, "_vfe3_token_content_counts", None)
    cached_vocab_size = getattr(dataset, "_vfe3_token_content_counts_vocab_size", None)
    counts_match = (
        vocab_size is not None
        and isinstance(cached_counts, torch.Tensor)
        and cached_counts.device.type == "cpu"
        and cached_counts.dtype == torch.long
        and cached_counts.ndim == 1
        and cached_counts.numel() == vocab_size
        and cached_vocab_size == vocab_size
    )
    if cached_sha is None and vocab_size is not None and not counts_match:
        cached_sha, cached_counts = _scan_token_content(tokens, vocab_size=vocab_size)
        setattr(dataset, "_vfe3_token_content_sha256", cached_sha)
        setattr(dataset, "_vfe3_token_content_counts", cached_counts)
        setattr(dataset, "_vfe3_token_content_counts_vocab_size", vocab_size)
        counts_match = True
    else:
        if cached_sha is None:
            cached_sha = _sha256_tensor_content(tokens)
            setattr(dataset, "_vfe3_token_content_sha256", cached_sha)
        if vocab_size is not None and not counts_match:
            cached_counts = _bincount_token_chunks(tokens, vocab_size=vocab_size)
            setattr(dataset, "_vfe3_token_content_counts", cached_counts)
            setattr(dataset, "_vfe3_token_content_counts_vocab_size", vocab_size)
            counts_match = True
    return (
        str(cached_sha),
        int(tokens.numel()),
        cached_counts if counts_match else None,
    )


def _loader_token_content_identity(
    loader: Optional[Iterable],
) -> 'Tuple[Optional[str], Optional[int]]':
    r"""Return one immutable loader split's canonical token digest and count.

    ``ablation.get_loader`` reuses the same ``TokenWindows`` instance across cells. Cache the
    canonical int64 digest on that dataset after its first bounded hash so every later finalization
    reuses the exact value instead of rereading the unchanged mapped corpus.
    """
    digest, count, _ = _loader_token_content_summary(loader)
    return digest, count


def _bincount_token_chunks(
    tokens: torch.Tensor,

    *,
    vocab_size:   int,
    chunk_tokens: int = 128 * 1024,
) -> torch.Tensor:
    r"""Accumulate exact training-token counts with a bounded host-int64 working chunk."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    flat = tokens.detach().reshape(-1)
    counts = torch.zeros(vocab_size, dtype=torch.long, device="cpu")
    for start in range(0, flat.numel(), chunk_tokens):
        chunk = flat[start:start + chunk_tokens].to(device="cpu", dtype=torch.long)
        _require_token_chunk_in_vocab(chunk, vocab_size)
        partial = torch.bincount(chunk, minlength=vocab_size)
        counts.add_(partial)
    return counts


def _write_provenance(
    artifacts: RunArtifacts,
    cfg:       VFE3Config,
    model:     torch.nn.Module,
    logger:    logging.Logger,

    *,
    train_loader:  Optional[Iterable] = None,
    val_loader:    Optional[Iterable] = None,
    test_loader:   Optional[Iterable] = None,
    data_seed:     Optional[int]      = None,
    max_tokens:    Optional[int]      = None,
    tokenizer_tag: Optional[str]      = None,
) -> None:
    r"""Write code, environment, per-split data, and data-order provenance best-effort."""

    run_start_git_identity = dict(artifacts.run_start_git_identity)
    final_git_identity = _git_code_identity()
    git_identity_status, git_identity_drift = _git_identity_comparison(
        run_start_git_identity,
        final_git_identity,
    )
    final_code_identity_sha256: Optional[str]
    final_code_identity_error: Optional[str] = None
    try:
        final_code_identity_sha256 = _package_code_identity()
    except (OSError, RuntimeError) as exc:
        final_code_identity_sha256 = None
        final_code_identity_error = repr(exc)
    code_identity_drift = (
        None
        if final_code_identity_sha256 is None
        else final_code_identity_sha256 != artifacts.code_identity_sha256
    )
    if code_identity_drift is True or git_identity_drift is True:
        source_identity_status = "drifted"
        source_identity_drift: Optional[bool] = True
    elif code_identity_drift is False and git_identity_drift is False:
        source_identity_status = "stable"
        source_identity_drift = False
    else:
        source_identity_status = "unverifiable"
        source_identity_drift = None
    # Non-strict code identity records the on-disk digest at run start. When that differs from the
    # digest this process imported, the executed code is an older (possibly mixed) generation, so
    # the run-boundary comparison above cannot certify it as 'stable'.
    process_code_identity = getattr(artifacts, "process_code_identity_sha256", None)
    process_code_identity_match = (
        None if process_code_identity is None
        else process_code_identity == artifacts.code_identity_sha256
    )
    if process_code_identity_match is False:
        source_identity_status = "mixed_generation"

    prov: Dict[str, object] = {
        "seed":                cfg.seed,
        "deterministic_state": deterministic_state(),
        "n_params":            int(sum(p.numel() for p in model.parameters())),
        "torch_version":       torch.__version__,
        "cuda_version":        torch.version.cuda,
        "device_name":         (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "data_seed":           (int(data_seed) if data_seed is not None else None),
        "max_tokens":          (int(max_tokens) if max_tokens is not None else None),
        "tokenizer_tag":       tokenizer_tag,
        "code_identity_sha256":       artifacts.code_identity_sha256,
        "process_code_identity_sha256": process_code_identity,
        "process_code_identity_match":  process_code_identity_match,
        "selection_data_identity":    artifacts.selection_data_identity,
        "final_code_identity_sha256": final_code_identity_sha256,
        "code_identity_drift":        code_identity_drift,
        "run_start_git_identity":     run_start_git_identity,
        "final_git_identity":         final_git_identity,
        "git_identity_status":        git_identity_status,
        "git_identity_drift":         git_identity_drift,
        "source_identity_status":     source_identity_status,
        "source_identity_drift":      source_identity_drift,
    }
    if final_code_identity_error is not None:
        prov["final_code_identity_error"] = final_code_identity_error
    # Backward-compatible top-level Git fields now identify the source tree at RUN START, which is
    # the execution identity consumed by existing multiseed and scaling cohort readers. The final
    # probe above exists to expose mid-run source drift rather than silently relabeling the run.
    prov.update(run_start_git_identity)
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        sha_key = f"{split}_data_sha256"
        n_key = f"{split}_data_n_tokens"
        prov[sha_key], prov[n_key] = None, None
        try:
            # Hash the CONTENT, not the storage: TokenWindows may hold the stream in its native
            # cache dtype (int32 memmap) or int64 (capped load). Normalize to int64 once, then reuse
            # that immutable split digest across every ablation cell sharing the loader.
            if split == "train":
                prov[sha_key], prov[n_key], _ = _loader_token_content_summary(
                    loader,
                    vocab_size=int(cfg.vocab_size),
                )
            else:
                prov[sha_key], prov[n_key] = _loader_token_content_identity(loader)
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, MemoryError) as exc:
            # Best-effort provenance, narrowed to the realistic hash-path failures (audit
            # 2026-07-12 N2): exotic loader/dtype/allocation errors must not crash finalize, but
            # the failure is LOGGED -- previously a bare `except Exception: pass` silently
            # recorded null data hashes, indistinguishable from a loader without corpus tokens --
            # while a programming error (NameError/KeyError/...) now surfaces instead of
            # masquerading as a missing corpus.
            logger.warning("%s-split provenance data hash failed (%s); recording null", split, exc)
    # Backward-compatible held-out aliases consumed by existing scaling-analysis artifacts.
    prov["data_sha256"] = prov["test_data_sha256"]
    prov["data_n_tokens"] = prov["test_data_n_tokens"]
    artifacts.save_json("provenance.json", prov)
    if source_identity_status == "mixed_generation":
        logger.warning(
            "the process executed code imported before a source edit; provenance records both the "
            "import-time and on-disk code identities (source_identity_status='mixed_generation')"
        )
    elif source_identity_status == "drifted":
        logger.warning(
            "source identity drifted during the run; provenance records both boundaries"
        )
    elif source_identity_status == "unverifiable":
        logger.warning(
            "source identity could not be verified at both run boundaries; provenance is marked"
        )
    logger.info(
        "wrote provenance.json (git_sha=%s dirty=%s source_identity=%s)",
        prov.get("git_sha"),
        prov.get("git_dirty"),
        source_identity_status,
    )


@contextmanager
def _diagnostic_eval_mode(model: torch.nn.Module) -> Iterator[None]:
    """Evaluate a diagnostic while preserving every module's incoming training flag."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


@torch.no_grad()
def _calibration_and_strata(
    corpus_counts: torch.Tensor,             # (V,) training-corpus unigram counts

    model:         torch.nn.Module,
    test_loader:   Iterable,
    device:        torch.device,

    *,
    max_batches: int = 20,
    n_bins:      int = 15,
) -> Dict[str, object]:
    r"""Decode calibration (ECE + reliability curve) and corpus-frequency-stratified CE over the test
    split. The decode is non-standard (KL-to-prior Mahalanobis or mu @ W^T with Sigma feeding the
    logit scale), so a mis-scaled ``decode_log_scale`` can leave PPL acceptable while the probability
    mass is wrong -- PPL alone cannot catch it. Bucket cutoffs are quantiles over the positive-count
    token types in the complete training corpus; sampled target duplication cannot move them, and
    evaluation targets unseen in training are rare. The aggregated values remain sampled held-out CE.
    The strata expose prior-table tail stagnation. Off-graph; capped at ``max_batches``."""
    with _diagnostic_eval_mode(model):
        import torch.nn.functional as F

        confs, corrects, nats, tgts = [], [], [], []
        for i, (tok, tgt) in enumerate(test_loader):
            tok, tgt = tok.to(device), tgt.to(device)
            logits = model(tok)                                     # (B, N, V) inference path
            lp = logits.reshape(-1, logits.shape[-1]).float()
            t = tgt.reshape(-1)
            valid = t != -100
            lp, t = lp[valid], t[valid]
            prob = torch.softmax(lp, dim=-1)
            p_max, pred = prob.max(dim=-1)
            confs.append(p_max)
            corrects.append((pred == t).float())
            nats.append(F.cross_entropy(lp, t, reduction="none"))
            tgts.append(t)
            if i + 1 >= max_batches:
                break
        if not confs:
            return {}
        conf, corr = torch.cat(confs), torch.cat(corrects)
        nat, tg = torch.cat(nats), torch.cat(tgts)

        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device)
        ece, rel = 0.0, []
        for b in range(n_bins):                                     # expected calibration error (15-bin)
            m = (conf > edges[b]) & (conf <= edges[b + 1])
            if m.any():
                acc, cf, w = corr[m].mean(), conf[m].mean(), m.float().mean()
                ece += float(w * (acc - cf).abs())
                rel.append({"conf": float(cf), "acc": float(acc), "frac": float(w)})
        if corpus_counts.ndim != 1:
            raise ValueError("corpus_counts must be a one-dimensional training-corpus bincount")
        counts = corpus_counts.to(device=tg.device)
        if int(tg.max()) >= counts.numel():
            raise ValueError("corpus_counts does not cover every sampled evaluation target")
        positive_counts = counts[counts > 0].float()
        if positive_counts.numel() == 0:
            q1, q2 = 0.0, 0.0
        else:
            quantiles = positive_counts.new_tensor([1 / 3, 2 / 3])
            q1, q2 = torch.quantile(positive_counts, quantiles).tolist()
        tok_count = counts[tg].float()                              # training-corpus count of each target
        seen = tok_count > 0
        strata = {}
        for name, mask in (("rare", (~seen) | (tok_count <= q1)),
                           ("mid", seen & (tok_count > q1) & (tok_count <= q2)),
                           ("frequent", seen & (tok_count > q2))):
            strata[name] = float(nat[mask].mean()) if mask.any() else float("nan")
        return {"ece": ece, "reliability": rel, "overall_ce": float(nat.mean()),
                "corpus_freq_strata_ce": strata}


def _fd_gradient_check(
    model:       torch.nn.Module,
    test_loader: Iterable,
    device:      torch.device,

    *,
    n_coords:    int   = 4,
    fd_eps:      float = 1e-3,
) -> float:
    r"""Worst relative error between autograd-of-CE and a central finite difference on a few DECODE
    coordinates (``output_proj_weight``, else the decode log-scale) -- a parameter whose gradient does
    NOT pass through the E-step belief adjoint (which the default kernel/oracle route detaches), so a
    healthy model reads ~1e-4 and a broken decode adjoint spikes far above it. (Probing ``mu_embed``
    instead would sit at the detached-oracle's ~10-25% plateau with no headroom to flag a real bug.)
    The CLAUDE.md-mandated FD-vs-autograd check. Best-effort on one tiny batch; restores every coord."""
    with _diagnostic_eval_mode(model):
        batch = next(iter(test_loader))
        tok, tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
        tok = tok[:2].to(device)
        tgt = tgt[:2].to(device)
        pb = model.prior_bank
        p = pb.output_proj_weight if getattr(pb, "output_proj_weight", None) is not None else pb.decode_log_scale
        model.zero_grad(set_to_none=True)
        _, loss, _ = model(tok, tgt)
        loss.backward()
        if p.grad is None:                                          # decode param severed under this config
            model.zero_grad(set_to_none=True)
            return float("nan")
        flat, gflat = p.detach().view(-1), p.grad.detach().view(-1).clone()
        # Probe the LARGEST-gradient coords, not random ones: a random coord usually has near-zero
        # gradient where the central difference is dominated by fp rounding (a spurious large rel error),
        # so this checks the gradient where its signal actually dominates -- where a real adjoint bug shows.
        idx = torch.topk(gflat.abs(), min(n_coords, gflat.numel())).indices.tolist()
        worst = 0.0
        with torch.no_grad():
            for j in idx:
                orig = float(flat[j])
                # try/finally (audit 2026-07-12 N1): `flat` is a storage-sharing view of the LIVE
                # decode parameter, and the caller catches broadly -- a forward that raises between
                # the +/-eps writes must not leave the parameter perturbed for subsequent probes.
                try:
                    flat[j] = orig + fd_eps
                    _, lp, _ = model(tok, tgt)
                    flat[j] = orig - fd_eps
                    _, lm, _ = model(tok, tgt)
                finally:
                    flat[j] = orig
                fd = (float(lp) - float(lm)) / (2.0 * fd_eps)
                ana = float(gflat[j])
                if not math.isfinite(fd) or not math.isfinite(ana):
                    model.zero_grad(set_to_none=True)
                    return float("nan")
                worst = max(worst, abs(fd - ana) / max(abs(fd), abs(ana), 1e-8))
        model.zero_grad(set_to_none=True)
        return worst


def _write_research_artifacts(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,
    train_loader: Optional[Iterable],
    test_loader: Optional[Iterable],
    device:      torch.device,
    logger:      logging.Logger,
) -> None:
    r"""Best-effort ``research.json``: decode calibration (ECE) + frequency-stratified loss + the FD
    gradient-check residual. Each probe is independently guarded so one failure never blocks the
    others or the saved numeric results."""
    if test_loader is None:
        return
    out: Dict[str, object] = {}
    try:
        _, _, corpus_counts = _loader_token_content_summary(
            train_loader,
            vocab_size=int(cfg.vocab_size),
        )
        if corpus_counts is None:
            raise ValueError("training loader dataset does not expose corpus tokens")
        calibration = _calibration_and_strata(corpus_counts, model, test_loader, device)
        out.update(calibration)
        if calibration.get("reliability") is not None:
            out["reliability_split"] = "test"
    except Exception as exc:
        logger.warning("calibration/strata probe failed (%s); skipped", exc)
    try:
        out["fd_gradient_worst_rel_error"] = _fd_gradient_check(model, test_loader, device)
        logger.info("FD gradient-check worst rel error: %.2e", out["fd_gradient_worst_rel_error"])
    except Exception as exc:
        logger.warning("FD gradient-check failed (%s); skipped", exc)
    # B1/EXP-3 Sigma_q calibration headline: Spearman rho(tr Sigma_q, CE) and the across-token
    # spread gate CV(tr Sigma_q) > 0.10 (below it the covariance channel is inert -- reported as
    # such, NOT miscoded as "decode doesn't matter"). Off-graph; capped at a few batches.
    # On a route whose covariance is frozen at encode, tr Sigma is the input token's table variance
    # tr Sigma(x_i), a function of the token id alone, so rho pairs input-token variance with
    # next-token CE (a type-level association) and the CV gate cannot certify belief uncertainty:
    # the statistics are kept under that label and the gate is withheld (None, with a reason).
    try:
        from vfe3.viz.extract import belief_ce_bank
        from vfe3 import metrics as _cal_metrics
        bank = belief_ce_bank(model, test_loader, device=device, max_batches=10)
        tr = bank["tr_sigma"]
        if tr.numel() >= 2:
            frozen_covariance = _cal_metrics.covariance_frozen_at_encode(cfg)
            out["sigma_trace_source"] = (
                "input_token_variance" if frozen_covariance else "belief_covariance")
            out["sigma_ce_spearman"] = _cal_metrics.spearman_rho(tr, bank["ce"])
            out["sigma_trace_cv"] = _cal_metrics.cv(tr)
            if frozen_covariance:
                out["sigma_trace_cv_gate_pass"] = None
                out["sigma_trace_cv_gate_reason"] = (
                    "covariance frozen at encode (projected attention_mode without "
                    "predictive_covariance='moment_matched', or skip_belief_sigma_update=True, "
                    "no delta_full covariance seam): tr Sigma is "
                    "the input-token variance tr Sigma(x_i), not a context-dependent belief "
                    "uncertainty, so the CV>0.10 calibration gate is withheld")
                logger.info("input-token variance vs next-token CE: rho(trSigma(x_i),CE)=%.3f "
                            "CV(trSigma(x_i))=%.3f (calibration gate withheld: covariance frozen "
                            "at encode)", out["sigma_ce_spearman"], out["sigma_trace_cv"])
            else:
                out["sigma_trace_cv_gate_pass"] = bool(out["sigma_trace_cv"] > 0.10)
                logger.info("Sigma_q calibration: rho(trSigma,CE)=%.3f CV(trSigma)=%.3f (gate>0.10: %s)",
                            out["sigma_ce_spearman"], out["sigma_trace_cv"],
                            out["sigma_trace_cv_gate_pass"])
    except Exception as exc:
        logger.warning("Sigma_q calibration probe failed (%s); skipped", exc)
    if out:
        artifacts.save_json("research.json", out)


# Lookup tables indexed by token id (prior bank) or absolute position (model). Their sum is the
# "embedding" parameter count that Kaplan et al. (2020, arXiv:2001.08361) exclude from N.
_TOKEN_TABLE_PARAMETERS: Tuple[Tuple[str, str], ...] = (
    ("prior_bank", "mu_embed"),
    ("prior_bank", "sigma_log_embed"),
    ("prior_bank", "phi_embed"),
    ("prior_bank", "output_proj_weight"),
    ("prior_bank", "output_proj_bias"),
    ("prior_bank", "decode_mu_embed"),
    ("prior_bank", "decode_sigma_log_embed"),
    ("", "pos_phi_free"),
)


def _token_table_parameter_count(model: torch.nn.Module) -> Optional[int]:
    r"""Parameters held in per-token (and absolute-position) lookup tables, or ``None`` if unknown.

    ``n_non_embedding_params = n_params - n_token_table_params`` is the context-processing count a
    scaling fit should also be reported against: at the scaling widths the vocabulary-indexed
    ``phi``/``mu``/``sigma``/decode tables are ~99% of ``n_params``."""
    if getattr(model, "prior_bank", None) is None:
        return None
    total = 0
    for owner_name, name in _TOKEN_TABLE_PARAMETERS:
        owner = getattr(model, owner_name) if owner_name else model
        parameter = getattr(owner, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            total += int(parameter.numel())
    return total


def _cost_model_fields(
    model:       torch.nn.Module,
    cfg:         VFE3Config,

    n_params:    int,
    tokens_seen: int,

    *,
    wall_time:   Optional[float] = None,
) -> Dict[str, object]:
    r"""Structural axes + a faithful compute proxy for the scaling frontier (extends scaling_point).

    The ``6ND`` rule (``6 * n_params * tokens_seen``) is LOOSE here: ``n_params`` is dominated by the
    vocab-size gauge/prior tables (``phi_embed`` is ``V * n_gen``), but only the active tokens' rows
    participate per step, while the decode reads all ``V`` rows every forward. So this records
    (a) the structural axes that set the real per-token work, (b) ``active_params_per_token`` -- the
    honest working set (decode-bound, ~``K``, NOT ``phi``/``n_gen``-bound, the mirror image of
    ``n_params`` being ``n_gen``-dominated), and (c) a transparent analytic FLOP proxy assembled from
    those drivers (order-1 constants; for a calibrated frontier use ``wall_time`` or a profiler).
    ``wall_time`` on a fixed GPU is the empirical ground truth the analytic constants calibrate
    against. ``n_gen`` / ``n_blocks`` are read from the GROUP OBJECT so they track ``cross_couplings``,
    bracket closure, and the ``so_n``/``sp_n`` decoupling of ``n_gen`` from ``K``.
    """
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    n_gen = int(model.group.generator_shape[0])
    n_blocks = max(1, len(model.group.irrep_dims))
    d_head = K / n_blocks                                        # representative block dim
    # ACTIVE params per token: the executable encoder looks up one routed mean/variance row plus its
    # phi row. The decoder then reads EITHER its routed prior-bank rows (2VK) OR the linear output
    # matrix (VK) and optional bias. Base token priors and every decode vocabulary prior are
    # diagonal.
    token_row = 2 * K + n_gen
    executable_build = getattr(model, "executable_build", None)
    block_mlp_build = (
        executable_build.block_mlp
        if isinstance(executable_build, ExecutableBuildMetadata)
        else block_mlp_build_metadata(
            enabled=bool(getattr(cfg, "use_block_mlp", False)),
            mode=getattr(cfg, "block_mlp_mode", "coordinate"),
            covariance=getattr(cfg, "block_mlp_covariance", "passthrough"),
            expansion=int(getattr(cfg, "block_mlp_expansion", 4)),
            activation=getattr(cfg, "block_mlp_activation", "gelu"),
            dropout=float(getattr(cfg, "block_mlp_dropout", 0.0)),
            covariance_floor=float(getattr(cfg, "block_mlp_covariance_floor", 1e-4)),
            embed_dim=K,
            irrep_dims=model.group.irrep_dims,
            n_layers=int(cfg.n_layers),
        )
    )
    block_mlp_active = block_mlp_build.enabled
    block_mlp_params = block_mlp_build.parameter_count
    if cfg.decode_mode != "linear":
        decode_readout = 2 * V * K
    else:
        decode_readout = V * K + (V if cfg.decode_bias else 0)
    active = token_row + decode_readout
    active += block_mlp_params
    # Transparent analytic FLOP proxy. Per token: decode over all V (2VK) and L*T belief E-step
    # iterations. Each E-step iteration has O(N) attention energy (2NK) plus O(N) transport
    # application (2N*d_head^2). Constants are O(1); this is a proxy, not a calibrated count.
    L, T, N = int(cfg.n_layers), int(cfg.n_e_steps), int(cfg.max_seq_len)
    fpt_decode         = 2.0 * V * K
    estep_kernel       = 2.0 * N * K + 2.0 * N * d_head * d_head
    belief_estep       = L * T * estep_kernel
    fpt_estep          = belief_estep
    fpt_block_mlp = block_mlp_build.flops_per_token if block_mlp_active else 0.0
    est_flops_analytic = (fpt_decode + fpt_estep + fpt_block_mlp) * float(tokens_seen)
    out: Dict[str, object] = {
        "embed_dim":               K,
        "n_heads":                 int(cfg.n_heads),
        "n_blocks":                n_blocks,
        "n_gen":                   n_gen,
        "n_layers":                L,
        "n_e_steps":               T,
        "max_seq_len":             N,
        "batch_size":              int(cfg.batch_size),
        "diagonal_covariance":     bool(cfg.diagonal_covariance),
        "gauge_group":             cfg.gauge_group,
        "decode_mode":             cfg.decode_mode,
        "vocab_size":              V,
        "n_learnable_params":      int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "active_params_per_token": int(active),
        "est_flops_analytic":      est_flops_analytic,
        "flops_per_token_decode":  fpt_decode,
        "flops_per_token_estep":   fpt_estep,
        "block_mlp_params":         int(block_mlp_params),
        "flops_per_token_block_mlp": fpt_block_mlp,
        "device_name":             (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "amp_dtype":               cfg.amp_dtype,
    }
    token_table_params = _token_table_parameter_count(model)
    out["n_token_table_params"] = token_table_params
    out["n_non_embedding_params"] = (
        None if token_table_params is None else int(n_params) - token_table_params)
    conditional_fields = _conditional_artifact_fields(cfg)
    if conditional_fields:
        out.update(conditional_fields)
        out["active_params_per_token"] += sum(fields["total_parameters"] for fields in conditional_fields.values())
        # The existing analytic proxy counts the legacy hand kernel, not the new inner autograd.
        out["est_flops_analytic"] = None
        out["flops_per_token_estep"] = None
        out["est_flops_analytic_unavailable_reason"] = "conditional routing/content cost is not modeled"
    if wall_time is not None and tokens_seen > 0:
        out["wall_time_s"]         = float(wall_time)
        out["wall_time_per_token"] = float(wall_time) / float(tokens_seen)
        out["wall_time_per_step"]  = float(wall_time) / max(1, int(cfg.max_steps))
    return out


def _estep_iterate_evidence(
    *,
    free_energy: Iterable[float],
    halted: bool,
    fixed_point_residual: Optional[float],
    fixed_point_tolerance: Optional[float],
) -> Dict[str, object]:
    """Gate scientific iterate labels on explicit mechanical evidence."""
    values = [float(value) for value in free_energy]
    descent = len(values) >= 2 and all(
        current <= previous for previous, current in zip(values, values[1:]))
    fixed = bool(
        halted and fixed_point_residual is not None and fixed_point_tolerance is not None
        and float(fixed_point_residual) <= float(fixed_point_tolerance)
    )
    return {
        "iterate_label": "final_iterate",
        "descent_evidence": descent,
        "convergence_evidence": fixed,
        "fixed_point_evidence": fixed,
    }


def _estep_endpoint_artifact(evidence: Mapping[str, object]) -> Dict[str, str]:
    """Name the endpoint-delta artifact without claiming unproven iterate behavior."""
    title = "E-step endpoint delta"
    if evidence.get("descent_evidence"):
        title += " (descent established)"
    return {"filename": "estep_endpoint_delta.png", "title": title}




def _scheduler_metadata(
    *,
    min_lr: float,
    min_lr_frac: float,
    base_lrs: Iterable[float] = (),
    group_roles: Iterable[str] = (),
) -> Dict[str, object]:
    """Describe the executable per-group floor, including frozen zero-base groups."""
    bases = [float(value) for value in base_lrs]
    roles = [str(value) for value in group_roles]
    if roles and len(roles) != len(bases):
        raise ValueError("scheduler group roles and base learning rates must have equal length")
    if not roles:
        roles = [f"group_{index}" for index in range(len(bases))]
    groups = []
    for index, (role, base) in enumerate(zip(roles, bases)):
        if base < 0.0 or not math.isfinite(base):
            raise ValueError(f"scheduler group {index} base LR must be finite and nonnegative")
        frozen = base == 0.0
        groups.append({
            "index": index,
            "role": role,
            "base_lr": base,
            "frozen": frozen,
            "absolute_floor": 0.0 if frozen else float(min_lr),
            "fractional_floor": 0.0 if frozen else float(min_lr_frac) * base,
            "effective_floor": (0.0 if frozen else max(float(min_lr), float(min_lr_frac) * base)),
        })
    return {
        "kind": "warmup_half_cosine_with_floor",
        "absolute_floor": float(min_lr),
        "fractional_floor": float(min_lr_frac),
        "zero_base_groups_remain_frozen": True,
        "floor_description": (
            "0 for group_base_lr == 0; otherwise "
            "max(absolute_floor, fractional_floor * group_base_lr)"
        ),
        "groups": groups,
    }


def finalize_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,

    *,
    final_validation: Optional[Mapping[str, Optional[float]]] = None,
    tokens_per_char: Optional[float]           = None,  # None -> BPC unavailable; BPT remains defined
    train_loader:    Optional[Iterable]       = None,
    val_loader:      Optional[Iterable]       = None,
    test_loader:     Optional[Iterable]       = None,
    losses:          Optional[List[float]]    = None,
    data_seed:       Optional[int]            = None,
    max_tokens:      Optional[int]            = None,
    tokenizer_tag:   Optional[str]            = None,
    device:          Optional[torch.device]   = None,
    wall_time:       Optional[float]          = None,
    logger:          Optional[logging.Logger] = None,
) -> Dict[str, object]:
    r"""Score the fully trained live weights on TEST and persist their metrics and figures.

    ``best_model.pt`` remains the validation-selected artifact. Finalization never reloads it:
    ``final_model.pt``, headline test metrics, research probes, and figures share the live state.
    The launcher supplies its already-computed final validation without another evaluation pass.
    """
    from vfe3.train import (                                     # local import avoids an import cycle
        evaluate,
    )

    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = next(model.parameters()).device

    final_step = getattr(artifacts, "final_step", None)
    final_val_fields = {
        f"final_val_{key}": value for key, value in (final_validation or {}).items()
    }
    if final_validation is not None:
        artifacts.save_json("final_validation.json", {"step": final_step, **final_validation})
        if final_step is not None:
            artifacts.maybe_save_best(final_step, model, float(final_validation["ppl"]))
    # Reachability guard (PB-03), the same invariant finalize_validation_run and save_checkpoint
    # enforce: finite best-model metadata is published below (test_results.json, summary.json), so
    # it must name weights that exist. Checked after the final validation had its chance to save.
    if (artifacts.save_model_artifacts
            and math.isfinite(float(artifacts.best_val_ppl))
            and not artifacts.best_path.is_file()):
        raise RuntimeError("finite best-model metadata has no reachable weights")
    if artifacts.save_model_artifacts:
        config = serialize_config(artifacts.cfg)
        with _unique_sibling_temp(artifacts.final_path) as tmp:
            torch.save({
                "artifact_kind":        _LIVE_FIGURE_SNAPSHOT_KIND,
                "schema_version":       _LIVE_FIGURE_SNAPSHOT_SCHEMA_VERSION,
                "model_state":          model.state_dict(),
                "config":               config,
                "config_fingerprint":   semantic_config_fingerprint(config),
                "code_identity_sha256": artifacts.code_identity_sha256,
            }, tmp)
            _atomic_replace(artifacts.final_path, tmp)
    logger.info("Testing final training weights (step %s)", final_step)

    conditional_enabled = is_conditional_filtering(model)
    results: Dict[str, object] = {
        "diagnostics": {},
        **_conditional_artifact_fields(cfg),
    }                                                          # mixes scalar headlines and nested diagnostics
    if test_loader is not None:
        m = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
        results.update({
            "test_ce":             m["ce"],
            "test_ppl":            m["ppl"],
            "test_bits_per_token": m["bits_per_token"],
            "test_bpc":            m["bpc"],
            "expected_targets":    m["expected_targets"],
            "scored_targets":      m["scored_targets"],
            "excluded_targets":    m["excluded_targets"],
            "test_expected_targets": m["expected_targets"],
            "test_scored_targets":   m["scored_targets"],
            "test_excluded_targets": m["excluded_targets"],
        })
        logger.info(
            "Test (held-out) | CE: %.4f | PPL: %.2f | BPT: %.4f | BPC: %s",
            m["ce"], m["ppl"], m["bits_per_token"],
            (f"{float(m['bpc']):.4f}" if m["bpc"] is not None else "unavailable"),
        )
    best_val_ppl = artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None
    decode_tau_eff = _decode_tau_eff(model, cfg)
    results.update({"best_val_ppl": best_val_ppl, "best_step": artifacts.best_step,
                    "reloaded_best": False, "evaluation_weights": "final",
                    "evaluated_step": final_step, **final_val_fields,
                    "decode_tau_eff": decode_tau_eff})

    # Held-out inference-depth counterfactual, always attempted when test metrics are available. This does not replace any selected headline
    # metric: it is published only under the diagnostic namespace after a complete successful probe.
    if conditional_enabled:
        results["diagnostics"]["zero_e_steps_counterfactual"] = _conditional_unavailable_probe("zero_e_steps_counterfactual", cfg=model.cfg)
        results["diagnostics"]["legacy_reduced_free_energy"] = _conditional_unavailable_probe("legacy_reduced_free_energy", cfg=model.cfg)
    if (not conditional_enabled and test_loader is not None
            and results.get("test_ce") is not None):
        _saved_ne = model.cfg.n_e_steps
        try:
            model.cfg.n_e_steps = 0
            m0 = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
            counterfactual_ce = float(m0["ce"])
            ce_delta = counterfactual_ce - float(results["test_ce"])
            diagnostics = results["diagnostics"]
            if not isinstance(diagnostics, dict):
                raise RuntimeError("finalization diagnostics must be a mapping")
            diagnostics["zero_e_steps_counterfactual"] = {
                "kind":                 "held_out_inference_depth_counterfactual",
                "split":                "test",
                "configured_depth":     int(_saved_ne),
                "counterfactual_depth": 0,
                "counterfactual_ce":    counterfactual_ce,
                "ce_delta_vs_headline": ce_delta,
            }
            logger.info(
                "Zero-E-step held-out counterfactual (CE@0 - CE@%d): %.4f",
                _saved_ne,
                ce_delta,
            )
        except Exception as exc:
            logger.warning("zero-E-step held-out counterfactual failed (%s); skipped", exc)
        finally:
            model.cfg.n_e_steps = _saved_ne

    # EXP-5 (C2): the converged final E-step free energy PER TOKEN -- the E-step's OWN target-blind
    # functional value (free_energy_value sums F over the N tokens; divide by N). Persisted so a
    # cross-arm reader (scaling_analysis) can describe the final-F/CE relationship across an
    # n_e_steps sweep -- the structural non-Neal-Hinton EM prediction (the E-step serves a distinct
    # functional, not the likelihood). Off-graph, best-effort, on a fixed test batch (sequence 0).
    if test_loader is not None and not conditional_enabled:
        try:
            from vfe3.viz.extract import e_step_belief_trace
            _b = next(iter(test_loader))
            _tok = (_b[0] if isinstance(_b, (tuple, list)) else _b).to(device)
            _tr = e_step_belief_trace(model, _tok)              # n_iter defaults to cfg.n_e_steps
            results["estep_final_iterate_f_per_token"] = float(_tr["free_energy"][-1]) / max(1, int(_tok.shape[1]))
            results["estep_final_f_per_token"] = results["estep_final_iterate_f_per_token"]  # compatibility alias
            results["estep_iterate_evidence"] = _estep_iterate_evidence(
                free_energy=[float(value) for value in _tr["free_energy"]],
                halted=False, fixed_point_residual=None, fixed_point_tolerance=None,
            )
            logger.info("Final E-step iterate F/token: %.4f", results["estep_final_iterate_f_per_token"])
        except Exception as exc:
            logger.warning("estep final-F probe failed (%s); skipped", exc)

    depth_loader = val_loader if val_loader is not None else test_loader
    # Mechanism records are standard outputs. The sole figure-policy switch controls only the
    # embedding/clustering family downstream in the isolated report worker; it never suppresses
    # numeric probes or ordinary publication figures. Batch extraction remains best-effort so an
    # exhausted or malformed loader cannot invalidate the already-computed headline metrics.
    _depth_tokens = _depth_targets = None
    if depth_loader is not None:
        try:
            _batch = next(iter(depth_loader))
            if not isinstance(_batch, (tuple, list)) or len(_batch) < 2:
                raise ValueError("mechanism diagnostics require a (tokens, targets) batch")
            _depth_tokens = _batch[0].to(device)
            _depth_targets = _batch[1].to(device)
        except Exception as exc:
            logger.warning("diagnostic batch unavailable (%s); mechanism probes skipped", exc)
    if _depth_tokens is not None:
        try:
            _depth_record = collect_estep_depth_sensitivity(
                model,
                _depth_tokens,
                _depth_targets,
                depths=(0, 1, 2, 3, 5, 8),
            )
            artifacts.save_json("estep_depth_sensitivity.json", _depth_record)
        except Exception as exc:
            logger.warning("estep depth-sensitivity probe failed (%s); skipped", exc)
        _character_record = None
        _beta_record = None
        _context_record = None
        try:
            _character_record = collect_estep_character(
                model, _depth_tokens, depths=(1, 2, 3, 5, 8))
            artifacts.save_json("estep_character.json", _character_record)
        except Exception as exc:
            logger.warning("estep character probe failed (%s); skipped", exc)
        try:
            _beta_record = collect_beta_channel_decomposition(model, _depth_tokens, _depth_targets)
            artifacts.save_json("beta_channel_decomposition.json", _beta_record)
        except Exception as exc:
            logger.warning("beta channel decomposition failed (%s); skipped", exc)
        try:
            _context_record = collect_context_sensitivity(model, _depth_tokens)
            artifacts.save_json("context_sensitivity.json", _context_record)
        except Exception as exc:
            logger.warning("context-sensitivity probe failed (%s); skipped", exc)
        _log_mechanism_diagnostics(logger, _character_record, _beta_record, _context_record)
    if depth_loader is not None:
        try:
            _phi_batch = next(iter(depth_loader))
            _phi_tokens = (
                _phi_batch[0] if isinstance(_phi_batch, (tuple, list)) else _phi_batch
            ).to(device)
            _phi_record = collect_phi_numerics(model, _phi_tokens)
            artifacts.save_json("phi_numerics.json", _phi_record)
        except Exception as exc:
            logger.warning("phi numerical-reference probe failed (%s); skipped", exc)
    artifacts.save_json("test_results.json", results)

    # Reproducibility provenance (git SHA / data hash / versions) + a scaling-law data point -- the
    # externally-grounded records a config-only artifact omits (identical config.json can come from
    # different code and data, and a single run carries no (N, tokens, FLOPs, loss) frontier point).
    _write_provenance(
        artifacts,
        cfg,
        model,
        logger,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        data_seed=data_seed,
        max_tokens=max_tokens,
        tokenizer_tag=tokenizer_tag,
    )
    n_params = int(sum(p.numel() for p in model.parameters()))
    tokens_seen = int(cfg.max_steps) * int(cfg.batch_size) * int(cfg.max_seq_len)
    # scaling-law data point: the 6ND FLOP proxy is LOOSE for a no-NN E-step model, so record the
    # inputs too (a cross-run frontier can be re-fit offline with the right cost model). The
    # _cost_model_fields block adds the structural axes + active-params-per-token + a faithful
    # analytic proxy so each point is standalone; best-effort, never blocks the saved numbers.
    scaling_point: Dict[str, object] = {
        "n_params":             n_params,
        "tokens_seen":          tokens_seen,
        "est_flops_6ND":        6 * n_params * tokens_seen,
        "test_ce":              results.get("test_ce"),
        "test_ppl":             results.get("test_ppl"),
        "test_bits_per_token":  results.get("test_bits_per_token"),
        "test_bpc":             results.get("test_bpc"),
        "expected_targets":      results.get("expected_targets"),
        "scored_targets":        results.get("scored_targets"),
        "excluded_targets":      results.get("excluded_targets"),
    }
    try:
        scaling_point.update(_cost_model_fields(model, cfg, n_params, tokens_seen, wall_time=wall_time))
    except Exception as exc:
        logger.warning("cost-model fields failed (%s); scaling_point keeps the 6ND proxy only", exc)
    # "n_steps" is the CONFIGURED budget. The realized counts sit beside it rather than replacing it
    # (audit 2026-08-06 F5): the LR scheduler clocks on accepted updates, so a run that rejected a
    # third of its steps trained on a different schedule than its config claims, and sweep arms with
    # different skip rates are not comparable. Absent on a train() call that had no artifacts.
    _realized = dict(getattr(artifacts, "realized_updates", None) or {})
    artifacts.save_json("summary.json", {
        **_conditional_artifact_fields(cfg),
        "n_steps":      cfg.max_steps,
        **_realized,
        "n_params":     n_params,
        "best_val_ppl": best_val_ppl,
        "best_step":    artifacts.best_step,
        "reloaded_best": False,
        "evaluation_weights": "final",
        "evaluated_step": final_step,
        **final_val_fields,
        "test_ppl":     results.get("test_ppl"),
        "test_ce":      results.get("test_ce"),
        "test_bits_per_token": results.get("test_bits_per_token"),
        "test_bpc":     results.get("test_bpc"),
        "expected_targets": results.get("expected_targets"),
        "scored_targets": results.get("scored_targets"),
        "excluded_targets": results.get("excluded_targets"),
        "test_expected_targets": results.get("test_expected_targets"),
        "test_scored_targets": results.get("test_scored_targets"),
        "test_excluded_targets": results.get("test_excluded_targets"),
        "diagnostics":  results["diagnostics"],
        "estep_final_iterate_f_per_token": results.get("estep_final_iterate_f_per_token"),
        "estep_final_f_per_token": results.get("estep_final_f_per_token"),
        "estep_iterate_evidence": results.get("estep_iterate_evidence"),
        "scheduler": _scheduler_metadata(
            min_lr=cfg.min_lr,
            min_lr_frac=cfg.min_lr_frac,
            base_lrs=getattr(artifacts, "scheduler_base_lrs", ()),
            group_roles=getattr(artifacts, "scheduler_group_roles", ()),
        ),
        "final_train_loss": (losses[-1] if losses else None),
        "wall_time_s":  wall_time,
        "decode_mode":     cfg.decode_mode,
        "decode_tau_eff":  decode_tau_eff,
        "use_head_mixer":  cfg.use_head_mixer,
        "phi_chart_norm_route": _phi_chart_norm_route(model, cfg),
        "scaling_point":   scaling_point,
    })

    # Pure-path certificate: the config toggles for the principal gauge / decode / free-energy purity
    # axes (flat gauge, canonical F, prior-bank decode, no head mixer, ...), plus the converged-state
    # stress metrics that say whether the numerical guards stayed inert. A REPORT of where the run sits,
    # not a judgment that any toggle is wrong (toggles are changed intentionally). Best-effort.
    try:
        artifacts.save_json("pure_path_report.json", _pure_path_report(
            cfg,
            artifacts.history,
            executable_build=model.executable_build,
        ))
    except Exception as exc:
        logger.warning("pure-path report failed (%s); skipped", exc)

    # Research records and standard publication figures are always attempted. The downstream
    # embedding switch adds only the vfe3.semantics analysis (figures/semantics/) of best_model.pt.
    try:
        _write_research_artifacts(model, artifacts, cfg, train_loader, test_loader, device, logger)
    except Exception as exc:
        logger.warning("research artifact preparation failed (%s); skipped", exc)

    # Plotting is a separate process because Anaconda NumPy/SciPy and PyTorch may initialize
    # incompatible Intel OpenMP copies. A native abort then kills only the disposable worker, not
    # the long-lived Spyder kernel. The worker also drives the model-replay publication set when
    # enabled, using the exact bounded test batches snapshotted from this invocation.
    try:
        _run_figures_isolated(
            model,
            artifacts,
            losses,
            logger,
            generate_publication=True,
            generate_embedding_figures=cfg.generate_embedding_figures,
            report_loader=test_loader,
        )
    except Exception as exc:
        logger.warning("isolated figure preparation failed (%s); numeric results are saved", exc)
    return results


@torch.no_grad()
def collect_estep_depth_sensitivity(
    model:   torch.nn.Module,
    tokens:  torch.Tensor,
    targets: torch.Tensor,
    depths:  Iterable[int],
) -> Dict[str, object]:
    r"""Evaluate current weights at several inference depths on one fixed batch.

    This is a sensitivity probe, not a retrained-depth comparison. Model mode, configured depths,
    and global CPU/CUDA RNG state are restored exactly before return.

    ``cfg.n_e_steps`` is read by the belief E-step (``model/block.py``); ``points`` sweeps that
    BELIEF loop and records the depth each point ran at.

    ``free_energy_per_token`` comes from ``e_step_belief_trace``, which uses SEQUENCE 0 only, while
    ``ce`` is over every sequence in ``tokens``; the key is duplicated as
    ``free_energy_per_token_seq0`` to keep that explicit, and ``n_sequences`` records the CE sample.
    """
    if is_conditional_filtering(model):
        return _conditional_unavailable_probe("collect_estep_depth_sensitivity", cfg=model.cfg)
    import torch.nn.functional as F
    from vfe3.viz.extract import e_step_belief_trace

    requested = list(depths)
    if any(type(depth) is not int or depth < 0 for depth in requested):
        raise ValueError(f"depths must contain nonnegative integers, got {requested!r}")
    ordered = sorted(set(requested))
    trained_depth = int(model.cfg.n_e_steps)
    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)

    def _score(belief_depth: int) -> Dict[str, float | int]:
        model.cfg.n_e_steps = belief_depth
        logits = model(tokens)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )
        trace = e_step_belief_trace(model, tokens, n_iter=belief_depth)
        free_energy = float(trace["free_energy"][-1]) / max(1, int(tokens.shape[1]))
        return {
            "depth":                      belief_depth,   # legacy key: the swept axis of `points`
            "belief_depth":               belief_depth,
            "ce":                         float(ce),
            "free_energy_per_token":      free_energy,
            "free_energy_per_token_seq0": free_energy,
        }

    points: List[Dict[str, float | int]] = []
    try:
        model.eval()
        # Belief-loop sweep: `n_e_steps` is the depth the weights were trained with.
        for depth in ordered:
            points.append(_score(depth))
    finally:
        model.cfg.n_e_steps = trained_depth
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "trained_depth": trained_depth,
        "n_sequences": int(tokens.shape[0]),
        "interpretation": (
            "current-weight depth sensitivity at fixed weights; no depth other than the trained "
            "depth was retrained. 'points' varies the BELIEF E-step; free_energy_per_token is "
            "sequence 0 only; ce is over all n_sequences."
        ),
        "points": points,
    }


def _log_mechanism_diagnostics(
    logger:    'logging.Logger',
    character: Optional[Dict[str, object]],
    beta:      Optional[Dict[str, object]],
    context:   Optional[Dict[str, object]],
) -> None:
    r"""Print the mechanism headlines to the run log so they are read, not buried in JSON.

    Every number here answers one question about where the model's capacity actually is. A probe
    that did not run prints nothing rather than a zero.
    """
    logger.info("---- mechanism diagnostics ----")
    if character is not None and character.get("points"):
        trained = int(character["trained_depth"])
        rows = {int(p["belief_depth"]): p for p in character["points"]}
        at_trained = rows.get(trained) or next(iter(rows.values()))
        deepest = rows[max(rows)]
        if at_trained.get("pair_precision_share") is not None:
            logger.info(
                "E-step fusion at trained depth %d: PAIR (attention) %.3f / PRIOR (residual) %.3f",
                trained, at_trained["pair_precision_share"], at_trained["prior_precision_share"])
        if at_trained.get("rel_displacement") is not None:
            logger.info(
                "Belief displacement ||dmu||/||mu_p||: %.4f at depth %d -> %.4f at depth %d",
                at_trained["rel_displacement"], trained,
                deepest["rel_displacement"], int(deepest["belief_depth"]))
        if deepest.get("cos_dir_vs_step1") is not None:
            logger.info(
                "Trajectory straightness cos(dir_%d, dir_1) = %+.4f; step 1 takes %.0f%% of it",
                int(deepest["belief_depth"]), deepest["cos_dir_vs_step1"],
                100.0 * (deepest.get("step1_share_of_displacement") or 0.0))
        if float(character.get("recompute_max_abs_err") or 0.0) > 1e-4:
            logger.warning(
                "E-step character replica error %.3e is NONZERO -- the probe no longer matches the "
                "kernel fusion; treat its precision shares as stale",
                character["recompute_max_abs_err"])
    if beta is not None and beta.get("available"):
        deltas = beta["delta_ce"]
        logger.info(
            "Attention channels (delta CE, nats): positional prior %+.4f vs content energy %+.4f",
            deltas["positional_channel"], deltas["content_channel"])
    if context is not None and context.get("stages"):
        logger.info(
            "Context injection (relative belief shift under prefix randomization): %s",
            ", ".join(f"{name} {value:.4f}" for name, value in context["stages"].items()))
    logger.info("-------------------------------")


@torch.no_grad()
def collect_estep_character(
    model:  torch.nn.Module,
    tokens: torch.Tensor,
    depths: Iterable[int],
) -> Dict[str, object]:
    r"""Is the belief E-step convergent inference, or one aggregation with a residual?

    Three questions, one fixed batch, current weights (2026-07-25):

    1. **How far does the belief move?** ``rel_displacement`` is
       :math:`\lVert\mu^{*}_d-\mu_p\rVert/\lVert\mu_p\rVert` at each depth d. A no-op reads 0.
    2. **Is the trajectory a straight line?** ``cos_dir_vs_step1`` is the cosine between the total
       displacement at depth d and the displacement after ONE step. Near +1 means every later step
       pushes the same way, so depth buys magnitude rather than direction.
    3. **How is the fixed point weighted?** ``mm_exact`` fuses
       :math:`\mu^{*}=(a\mu_p/s_p+\sum_j w_{ij}\mu_t/s_t)/P`. ``pair_precision_share`` is the
       transported-neighbor (ATTENTION) mass of ``P`` and ``prior_precision_share`` its complement
       (the RESIDUAL). Measured 2026-07-25 at 0.190 (K=20) and 0.298 (K=300), i.e. the converged
       belief is a convex blend dominated by its own prior, not a deep computation.

    SELF-VALIDATING. The split is recomputed from the same public helpers the kernel uses and then
    checked against the kernel's own returned ``mu_star``; ``recompute_max_abs_err`` is that residual
    and is 0.0 when the replica is exact. If the kernel's fusion is ever changed without updating
    this probe the residual grows, so the number is reported rather than asserted -- a drifted probe
    is visible in the artifact instead of silently wrong.

    Only the ``mm_exact`` kernel route is decomposed. On any other ``e_step_update`` the displacement
    and cosine are still reported and the shares are ``None`` (``precision_split_available=False``);
    nothing is fabricated for a route whose fusion does not exist.

    That last sentence was a docstring CLAIM the code did not honor until 2026-07-26: the displacement
    window was populated only inside the ``mm_exact`` spy, so a ``e_step_update='gradient'`` checkpoint
    returned ``None`` for every field, not just the shares. It surfaced while re-measuring the numbers
    B-01 invalidated -- the surviving K=20 checkpoint runs the gradient route, so the probe reported
    nothing at all about it. The block-level window below closes that: ``vfe_block`` receives the
    belief prior and returns the converged belief whatever the inner route is, so displacement and
    cosine are now genuinely route-independent. The ``mm_exact`` window stays PRIMARY where it exists,
    because it is what the published mm numbers were measured against; the block window is a fallback
    used only when no belief fusion was recorded.

    That fallback then read the block's RETURN value, which is post-``head_mixer``/``cg_coupling``/
    norm, while the ``mm_exact`` window reads ``mu_star`` before them -- so the two windows measured
    different quantities and the published ``rel_displacement`` was not comparable across routes
    (audit 2026-07-26 B-11). On the K=20 ``mm_exact`` checkpoint at ``use_head_mixer=True`` the same
    forward pass reads 0.213 on the mm window and 4.579 on the block's return, because the mixer
    alone moves the belief 4.43x its own norm -- and 98.5% of that motion is a scalar gain of 5.33
    (cosine 0.985), i.e. magnitude carrying no inferential content. The block window now ends at
    ``capture['converged']``, the pre-transform belief ``vfe_block`` already stashes, so both windows
    measure the E-step and ``displacement_window`` records which one produced each point.
    """
    if is_conditional_filtering(model):
        requested = list(depths)
        if any(type(depth) is not int or depth < 1 for depth in requested):
            raise ValueError(f"depths must contain integers >= 1, got {requested!r}")
        snapshot = model.build_diagnostic_snapshot(tokens)
        mu_p, mu_q = snapshot.trace_states[0].mu, snapshot.trace_states[-1].mu
        denominator = float(mu_p.norm())
        return {
            "available": True, "inference_semantics": model.conditional_semantics,
            "trained_depth": 1, "depth_sweep_available": False,
            "depth_sweep_reason": "only the configured single readout is supported",
            "precision_split_available": False,
            "points": [{"belief_depth": 1, "rel_displacement": (
                float((mu_q - mu_p).norm()) / denominator if denominator > 0 else None),
                "pair_precision_share": None, "cos_dir_vs_step1": None,
                "displacement_window": "captured_pre_transform_readout"}],
        }
    from vfe3.alpha_i import alpha_gradient_coefficient, alpha_is_per_coord
    from vfe3.free_energy import self_divergence_for_alpha
    from vfe3.gradients import kernels as kernels_module
    from vfe3.inference import e_step as e_step_module
    from vfe3.model import stack as stack_module

    requested = list(depths)
    if any(type(depth) is not int or depth < 1 for depth in requested):
        raise ValueError(f"depths must contain integers >= 1, got {requested!r}")
    ordered = sorted(set(requested))
    trained_depth = int(model.cfg.n_e_steps)
    was_training = bool(model.training)

    # `inference/e_step.py` binds mm_exact_update at import, so patching `kernels` alone leaves the
    # live call site pointing at the original and the probe silently records nothing. Patch every
    # module that holds a reference, and restore each from its own saved value.
    original_mm = kernels_module.mm_exact_update
    holders = [module for module in (kernels_module, e_step_module)
               if getattr(module, "mm_exact_update", None) is original_mm]
    original_contract = kernels_module._pair_contract
    contract_out: List[torch.Tensor] = []
    recorded: List[Dict[str, object]] = []

    # LAYER TAGGING (audit 2026-07-26 B-01): tag every call with the layer it belongs to, then read
    # LAYER 0 only -- one E-step of one layer is the object this probe is named for.
    frame = {"layer": 0}
    original_block = stack_module.vfe_block

    def _block_spy(*args, **kwargs):
        # Route-independent displacement window. vfe_block(belief, mu_p, sigma_p, ...) takes the
        # prior positionally at the stack call site, so the ANCHOR exists on every e_step_update --
        # unlike the mm_exact fusion, which the gradient route never calls.
        #
        # The CONVERGED end must be the E-step's own output, NOT the block's return value (audit
        # 2026-07-26 B-11). vfe_block applies head_mixer / cg_coupling / block_norm AFTER the E-step,
        # so `out.mu` is post-transform while the mm spy reads mu_star pre-transform: the two windows
        # then measure different quantities and rel_displacement is silently incomparable across
        # routes. Measured on the K=20 mm_exact checkpoint at use_head_mixer=True, one batch of 64:
        # the mm window reads 0.213 and the block's return reads 4.579 for the SAME forward pass,
        # because the mixer alone moves the belief 4.43x its own norm -- and that motion is 98.5%
        # a scalar gain (cos 0.985, gain 5.33), so it is magnitude carrying no inferential content.
        # `capture['converged']` is the pre-transform belief vfe_block already stashes for the M-step
        # self-coupling, which is exactly the mm window's endpoint; inject a capture when the caller
        # passed none (an empty mapping selects the same branches None does).
        record = (frame["layer"] == 0 and len(args) >= 2)
        if record and kwargs.get("capture") is None:
            kwargs["capture"] = {}
        out = original_block(*args, **kwargs)
        if record:
            mu_prior = args[1]
            converged = (kwargs.get("capture") or {}).get("converged")
            mu_converged = getattr(converged, "mu", None)
            if isinstance(mu_prior, torch.Tensor) and isinstance(mu_converged, torch.Tensor):
                block_window["mu_p"] = mu_prior.detach()
                block_window["mu_star"] = mu_converged.detach()
        frame["layer"] += 1                              # count layers as they COMPLETE
        return out

    def _contract_spy(w, x, irrep_dims):
        out = original_contract(w, x, irrep_dims)
        contract_out.append(out.detach())
        return out

    def _mm_spy(mu, sigma, mu_p, sigma_p, omega, **kw):
        contract_out.clear()
        # The kernel's reduced-precision island re-enters this same module global with fp32 inputs
        # (kernels.py:571-573). Recording the OUTER bf16 frame too would double every entry and
        # recompute the replica from reduced-precision moments against fp32 contractions, tripping
        # the staleness warning on a correct probe (audit 2026-07-26 B-05). Let only the inner
        # fp32 invocation record.
        if torch.is_autocast_enabled(mu.device.type):
            return original_mm(mu, sigma, mu_p, sigma_p, omega, **kw)
        mu_star, sigma_star = original_mm(mu, sigma, mu_p, sigma_p, omega, **kw)
        if len(contract_out) < 2:                      # a route that skipped the pair contraction
            return mu_star, sigma_star
        pair_prec, pair_mean = contract_out[0], contract_out[1]
        eps = kw.get("eps", 1e-6)
        kl_max = kw.get("kl_max", 100.0)
        mode = kw.get("lambda_alpha_mode", "constant")
        family = kernels_module.get_family(kw.get("family", "gaussian_diagonal"))
        self_divergence = self_divergence_for_alpha(
            family(mu, sigma), family(mu_p, sigma_p), alpha=1.0, kl_max=kl_max, eps=eps,
            divergence_family=kw.get("divergence_family", "renyi"), lambda_alpha_mode=mode)
        coefficient = alpha_gradient_coefficient(
            self_divergence, value=kw.get("value", 1.0),
            b0=kw.get("b0", 1.0), c0=kw.get("c0", 1.0), mode=mode)
        if alpha_is_per_coord(mode):
            raw_self = kernels_module._raw_diag_kl_per_coord(mu, sigma, mu_p, sigma_p, eps=eps)
            self_mask = (raw_self < kl_max).to(mu.dtype)
        else:
            coefficient = coefficient.unsqueeze(-1)
            raw_self = kernels_module._raw_diag_kl(mu, sigma, mu_p, sigma_p, eps=eps)
            self_mask = (raw_self < kl_max).to(mu.dtype).unsqueeze(-1)
        a = self_mask * coefficient                                    # (..., N, 1) or (..., N, K)
        sp = sigma_p.clamp(min=eps)
        prior_prec = a / sp
        precision = prior_prec + pair_prec
        replica = torch.where(precision <= eps, mu,
                              (a * mu_p / sp + pair_mean) / precision.clamp(min=eps))
        entry: Dict[str, object] = {
            "layer":      int(frame["layer"]),
            "prior_mass": float(prior_prec.sum()),
            "pair_mass":  float(pair_prec.sum()),
            "err":        float((replica - mu_star).abs().max()),
        }
        # Retain the two (B, N, K) tensors ONLY for the window actually read (audit 2026-07-26
        # B-06). Keeping every call's pair held 2 x n_layers x n_e_steps live device tensors while
        # two were ever used, which OOMs at production batch size with optimizer state resident --
        # and the OOM is swallowed by the caller's `except Exception`.
        if entry["layer"] == 0:
            if window["first_mu_p"] is None:
                window["first_mu_p"] = mu_p.detach()
            window["last_mu_star"] = mu_star.detach()
        recorded.append(entry)
        return mu_star, sigma_star

    points: List[Dict[str, object]] = []
    directions: Dict[int, torch.Tensor] = {}
    window: Dict[str, Optional[torch.Tensor]] = {"first_mu_p": None, "last_mu_star": None}
    block_window: Dict[str, Optional[torch.Tensor]] = {"mu_p": None, "mu_star": None}
    split_available = False
    max_err = 0.0
    realized_belief_calls: Dict[int, int] = {}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    try:
        model.eval()
        for module in holders:
            module.mm_exact_update = _mm_spy
        kernels_module._pair_contract = _contract_spy
        stack_module.vfe_block = _block_spy
        for depth in ordered:
            model.cfg.n_e_steps = depth
            recorded.clear()
            frame["layer"] = 0
            window["first_mu_p"] = window["last_mu_star"] = None
            block_window["mu_p"] = block_window["mu_star"] = None
            model(tokens)
            belief0 = [call for call in recorded if call["layer"] == 0]
            realized_belief_calls[depth] = len(belief0)
            # The mm_exact fusion window is primary; the block capture covers every other route.
            # Both now end at the E-step output, so rel_displacement means the same thing on either
            # (audit 2026-07-26 B-11). The window is published per point rather than assumed.
            if belief0 and window["first_mu_p"] is not None:
                anchor, converged = window["first_mu_p"], window["last_mu_star"]
                measured_window = "mm_exact_fusion"
            else:
                anchor, converged = block_window["mu_p"], block_window["mu_star"]
                measured_window = "block_capture"
            if anchor is None or converged is None or float(anchor.norm()) == 0.0:
                points.append({"belief_depth": depth, "n_belief_calls": len(belief0),
                               "rel_displacement": None, "displacement_window": None,
                               "pair_precision_share": None, "prior_precision_share": None})
                continue
            displacement = converged - anchor
            directions[depth] = displacement.flatten()
            point: Dict[str, object] = {
                "belief_depth":          depth,
                "n_belief_calls":        len(belief0),
                "rel_displacement":      float(displacement.norm() / anchor.norm()),
                "displacement_window":   measured_window,
                "pair_precision_share":  None,
                "prior_precision_share": None,
            }
            # Shares exist only where the mm_exact fusion does; nothing is fabricated otherwise.
            if belief0 and window["first_mu_p"] is not None:
                split_available = True
                first = belief0[0]
                max_err = max(max_err, max(float(call["err"]) for call in recorded))
                prior_mass, pair_mass = float(first["prior_mass"]), float(first["pair_mass"])
                total = prior_mass + pair_mass
                if total > 0.0:
                    point["pair_precision_share"] = pair_mass / total
                    point["prior_precision_share"] = prior_mass / total
            points.append(point)
    finally:
        for module in holders:
            module.mm_exact_update = original_mm
        kernels_module._pair_contract = original_contract
        stack_module.vfe_block = original_block
        model.cfg.n_e_steps = trained_depth
        model.train(was_training)
        torch.set_rng_state(cpu_rng)                     # audit 2026-07-26 B-09: match the sibling
        if cuda_rng is not None:                         # probes' RNG contract
            torch.cuda.set_rng_state_all(cuda_rng)

    step1 = directions.get(min(ordered)) if directions else None
    for point in points:
        vector = directions.get(int(point["belief_depth"]))
        if step1 is None or vector is None:
            point["cos_dir_vs_step1"] = None
            point["step1_share_of_displacement"] = None
            continue
        point["cos_dir_vs_step1"] = float(
            torch.nn.functional.cosine_similarity(vector, step1, dim=0))
        norm = float(vector.norm())
        point["step1_share_of_displacement"] = (float(step1.norm()) / norm) if norm > 0.0 else None

    return {
        "trained_depth":              trained_depth,
        "e_step_update":              str(model.cfg.e_step_update),
        "measured_layer":             0,
        "n_sequences":                int(tokens.shape[0]),
        "precision_split_available":  split_available,
        "recompute_max_abs_err":      max_err,
        "interpretation": (
            "current-weight E-step character on one fixed batch, measured on LAYER 0 only (audit "
            "2026-07-26 B-01 tags every call with its layer so later layers are excluded). "
            "n_belief_calls is the realized fixed-depth iteration count. rel_displacement is "
            "||mu*_d - mu_p|| / ||mu_p|| where mu*_d is the E-step output BEFORE the block's "
            "head_mixer / cg_coupling / norm, on either window -- displacement_window names which "
            "one produced the point (audit 2026-07-26 B-11; before that fix the non-mm_exact window "
            "ended at the block's RETURN value, so a post-transform number was published alongside "
            "pre-transform mm_exact numbers and the routes were not comparable). cos_dir_vs_step1 "
            "near +1 means depth adds magnitude, not "
            "direction. pair_precision_share is the transported-neighbor (attention) mass of the "
            "mm_exact fused precision and prior_precision_share its residual complement; both are "
            "None off the mm_exact route. recompute_max_abs_err is the probe's replica error against "
            "the kernel's own mu_star and must stay at 0."
        ),
        "points": points,
    }


@torch.no_grad()
def collect_beta_channel_decomposition(
    model:   torch.nn.Module,
    tokens:  torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, object]:
    r"""Which channel of the attention weight carries the prediction: position or content?

    The belief attention weight is :math:`\beta_{ij}=\mathrm{softmax}_j(\log\pi_{ij}-E_{ij}/\tau)`,
    where :math:`\log\pi` is the positional/precision prior and :math:`E_{ij}` is the
    belief-coupling energy -- the only content-dependent channel. Three arms isolate them:

    ============ ============================================ ==================================
    arm          beta                                         reads
    ============ ============================================ ==================================
    ``full``     ``softmax(log_prior - E/tau)``                as trained; reproduces the baseline
    ``no_energy``  ``softmax(log_prior)``                      the CONTENT channel's worth
    ``no_prior``   ``softmax(-E/tau)``                         the POSITIONAL channel's worth
    ============ ============================================ ==================================

    Measured 2026-07-25 on the K=300 baseline: flattening the prior cost 0.612 nats against 0.210
    for ablating the whole coupling energy, so the positional prior carried roughly three times the
    content channel. ``full`` reproducing the headline CE is the harness check -- if it drifts, the
    patch missed the live softmax and the other two arms mean nothing.

    Patching the ``attention_weights`` binding inside ``gradients.kernels`` isolates the belief beta
    exactly. Both belief routes read that binding -- ``belief_gradients`` and ``mm_exact_update``
    each call it -- so this works under either ``e_step_update``. ``available`` records whether any
    live softmax was actually intercepted; when it is False the arms are omitted rather than
    reported as numbers that describe nothing.

    ``no_prior`` preserves ``-inf`` entries of the log-prior so the causal / no-self MASK survives
    while its graded positional shape is flattened. A configuration carrying no attention log-prior
    at all therefore leaves ``no_prior`` equal to ``full``, correctly reporting that this model has
    no positional channel to ablate.
    """
    import torch.nn.functional as F
    from vfe3 import free_energy as free_energy_module
    from vfe3.gradients import kernels as kernels_module
    from vfe3.gradients import qkv as qkv_module
    from vfe3.inference import e_step as e_step_module

    was_training = bool(model.training)
    # Patch EVERY module that binds `attention_weights`, definition site included (audit
    # 2026-07-26 B-02). Patching only `kernels` left `inference/e_step.py`'s own import live: under
    # the phi substep called the unablated function, so all three arms ran the true beta while the
    # record still reported `available: True`.
    original = free_energy_module.attention_weights
    binders = [module for module in (free_energy_module, kernels_module, e_step_module, qkv_module)
               if getattr(module, "attention_weights", None) is original]
    mode = {"arm": "full"}

    def _patched(energy, *, tau=1.0, log_prior=None):
        if mode["arm"] == "no_energy":
            return original(torch.zeros_like(energy), tau=tau, log_prior=log_prior)
        if mode["arm"] == "no_prior":
            keep = None if log_prior is None else torch.where(
                torch.isneginf(log_prior), log_prior, torch.zeros_like(log_prior))
            return original(energy, tau=tau, log_prior=keep)      # keep the causal/no-self support
        return original(energy, tau=tau, log_prior=log_prior)

    arms: Dict[str, Optional[float]] = {}
    intercepted = {"n": 0}

    def _counting(energy, *, tau=1.0, log_prior=None):
        intercepted["n"] += 1
        return _patched(energy, tau=tau, log_prior=log_prior)

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    try:
        model.eval()
        for module in binders:
            module.attention_weights = _counting
        for arm in ("full", "no_energy", "no_prior"):
            mode["arm"] = arm
            logits = model(tokens)
            arms[arm] = float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-100,
            ))
    finally:
        for module in binders:
            module.attention_weights = original
        model.train(was_training)
        torch.set_rng_state(cpu_rng)                     # audit 2026-07-26 B-09
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    available = intercepted["n"] > 0
    baseline = arms.get("full")
    return {
        "available":     available,
        "e_step_update": str(model.cfg.e_step_update),
        "n_sequences":   int(tokens.shape[0]),
        "ce":            arms if available else {},
        "delta_ce": ({
            "content_channel":    (arms["no_energy"] - baseline) if available else None,
            "positional_channel": (arms["no_prior"] - baseline) if available else None,
        } if available else {}),
        "interpretation": (
            "delta_ce.content_channel is the cost of ablating the belief-coupling energy from beta "
            "(what CONTENT is worth); delta_ce.positional_channel is the cost of flattening the "
            "attention log-prior on its causal support (what POSITION is worth). available=False "
            "means no live beta softmax was intercepted -- the active e_step_update does not use "
            "the kernel route, and the arms carry no meaning."
        ),
    }


@torch.no_grad()
def collect_context_sensitivity(
    model:  torch.nn.Module,
    tokens: torch.Tensor,

    *,
    n_replicates: int = 4,
    seed:         int = 0,
) -> Dict[str, object]:
    r"""Does the belief at the final position depend on its prefix, and where does that enter?

    Randomizes every token before the last, holds the last token fixed, and measures the relative
    :math:`L^2` displacement of the final position's belief mean. A pure per-token lookup reads
    EXACTLY 0; anything above that is context the forward pass injected. The belief trace this
    reads is SEQUENCE 0 only, so ``tokens`` supplies one measured sequence however large the batch.

    Two stages separate the inference loop from the lookup. ``raw_prior`` (inference off) is the
    control and must read 0; ``belief_estep`` adds the belief E-step on top, so the increment is the
    context the fitted inference injects.

    This standard mechanism record costs ``n_replicates`` extra forward passes per stage.
    """
    conditional = is_conditional_filtering(model)
    from vfe3.viz.extract import e_step_belief_trace

    if type(n_replicates) is not int or n_replicates < 1:
        raise ValueError(f"n_replicates must be an int >= 1, got {n_replicates!r}")
    trained_depth = int(model.cfg.n_e_steps)
    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)

    vocab = int(model.cfg.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _final_mean(batch: torch.Tensor, initial: bool = False) -> torch.Tensor:
        trace = e_step_belief_trace(model, batch, n_iter=int(model.cfg.n_e_steps))
        return torch.as_tensor(trace["mu"][0 if initial else -1])[-1].flatten().float()

    stages: Dict[str, Optional[float]] = {}
    try:
        model.eval()
        readout_name = ("predictive_content" if conditional and
                        model.cfg.relational_readout_mode == "predictive_content" else "belief_estep")
        variants = [("raw_prior", 0), (readout_name, trained_depth)]
        for name, belief_depth in variants:
            if not conditional:
                model.cfg.n_e_steps = belief_depth
            reference = _final_mean(tokens, initial=conditional and belief_depth == 0)
            shifts = []
            for _ in range(n_replicates):
                perturbed = tokens.clone()
                if perturbed.shape[1] > 1:
                    perturbed[:, :-1] = torch.randint(
                        0, vocab, perturbed[:, :-1].shape, generator=generator, dtype=perturbed.dtype
                    ).to(perturbed.device)
                moved = _final_mean(perturbed, initial=conditional and belief_depth == 0)
                denominator = float(reference.norm())
                shifts.append(float((moved - reference).norm()) / denominator
                              if denominator > 0.0 else 0.0)
            stages[name] = sum(shifts) / len(shifts)
    finally:
        model.cfg.n_e_steps = trained_depth
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    return {
        "available":          True,
        "n_replicates":       n_replicates,
        "seed":               int(seed),
        "stages":             stages,
        "interpretation": (
            "mean relative L2 displacement of the FINAL position's belief mean when the prefix is "
            "randomized and the final token held fixed. raw_prior is the control and reads 0 for a "
            "pure per-token lookup; belief_estep adds the belief loop, so the increment is the "
            "context the fitted inference injects."
        ),
    }


@torch.no_grad()
def collect_phi_numerics(
    model:  torch.nn.Module,
    tokens: torch.Tensor,

    *,
    max_tokens: int = 8,
) -> Dict[str, object]:
    r"""Collect sampled active-frame numerics and flatness references off the hot path."""
    from vfe3 import metrics

    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else None)
    try:
        model.eval()
        selected = tokens[:1, :max_tokens]
        belief, _ = model.forward_beliefs(selected)
        phi = belief.phi[0]
        block_dims = list(model.group.irrep_dims)
        record: Dict[str, object] = {
            "sample_tokens": int(phi.shape[0]),
            "runtime_dtype": str(phi.dtype),
            "reference_dtype": "torch.float64",
            "composition": "group_product",
        }
        record["chart"] = metrics.phi_chart_statistics(
            phi,
            model.group,
            block_dims=block_dims,
        )
        if belief.right_phi is not None:
            record["right_chart"] = metrics.phi_chart_statistics(
                belief.right_phi[0],
                model.group,
                block_dims=block_dims,
            )
        if model.cfg.transport_mode == "flat":
            record["flatness"] = metrics.flatness_reference_statistics(
                phi,
                model.group,
                right_phi=belief.right_phi,
                max_triangles=8,
                block_dims=block_dims,
            )
        return record
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _restore_rng_state(
    rng_state: Mapping[str, object],
) -> None:
    r"""Restore the captured CPU (and any available CUDA) global RNG states.

    Mirrors ``load_checkpoint``'s RNG restore: the CPU state must be a CPU ByteTensor and the CUDA
    per-device states are set only when CUDA is available (a CPU-only host silently skips the CUDA leg).
    """
    cpu = rng_state.get("cpu")
    if cpu is not None:
        torch.set_rng_state(cpu.cpu() if hasattr(cpu, "cpu") else cpu)
    cuda = rng_state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in cuda])


@torch.no_grad()
def finalize_validation_run(
    model:       torch.nn.Module,
    artifacts:   RunArtifacts,
    cfg:         VFE3Config,
    val_loader:  Iterable,

    *,
    tokens_per_char: Optional[float]           = None,
    train_loader:    Optional[Iterable]       = None,
    losses:          Optional[List[float]]    = None,
    data_seed:       Optional[int]            = None,
    max_tokens:      Optional[int]            = None,
    tokenizer_tag:   Optional[str]            = None,
    device:          Optional[torch.device]   = None,
    wall_time:       Optional[float]          = None,
    logger:          Optional[logging.Logger] = None,
    terminal_state:  "Optional[TrainingTerminalState]" = None,
) -> Dict[str, object]:
    r"""Score validation, save terminal artifacts, and never open a test split.

    The validation-only sibling of :func:`finalize_run` (PB-02): invoked ONCE as a ``train`` terminal
    callback so even a default ablation cell (log/eval interval above ``max_steps``,
    ``checkpoint_interval=0``) publishes a complete resumable artifact set. Model selection and every
    reported number use VALIDATION data only -- the split is labeled ``selection_split="validation"``
    and ``summary.json`` never carries ``test_ce``/``test_ppl``/``test_bpc``.

    Successful sequence: evaluate ``val_loader`` once, append a terminal metrics row, publish the
    selected best weights (``best_model.pt``), write ``validation_results.json``, collect VALIDATION-only
    provenance and the pure-path report, and render the history-only figures. A ``finally`` block then
    strictly reloads ``terminal_state.raw_model_state`` and
    restores the captured CPU/CUDA RNG. After a successful validation the resumable terminal checkpoint
    is written from the restored model plus its matching optimizer/scaler/private-RNG/data cursor.
    ``summary.json`` is written only after that checkpoint exists. A validation/checkpoint failure
    restores the captured weights/RNG and re-raises,
    publishing no summary (and, via the caller, no success contract).

    Returns the ablation merge mapping with both ``final_val_bits_per_token`` and nullable
    ``final_val_bpc`` alongside the validation/selection/checkpoint fields.
    ``primary_val_ppl`` is the minimum of the finite run-wide best and the final validation PPL (or the
    final value when no earlier best exists); after ``maybe_save_best`` it equals the selected finite best.
    """
    from vfe3.train import (                                     # local import avoids an import cycle
        _loader_data_identity,
        evaluate,
    )

    logger = logger or logging.getLogger(__name__)
    if device is None:
        device = next(model.parameters()).device
    completed_step = _require_nonnegative_int(
        terminal_state.step if terminal_state is not None else cfg.max_steps,
        "terminal step",
    )
    artifacts.bind_selection_data_identity(_loader_data_identity(val_loader, cfg.vocab_size))

    # Reachability guard (PB-03): entering with FINITE best metadata (a resumed periodic best) requires
    # a reachable best_model.pt. A recomputed ablation cell instead enters with INFINITE metadata even
    # when a stale file survives on disk, so that file is ignored -- terminal validation replaces it
    # below rather than the finalizer silently selecting the previous cell's weights. This finalizer
    # never loads a preexisting best into the live model before scoring terminal validation.
    if (artifacts.save_model_artifacts
            and math.isfinite(float(artifacts.best_val_ppl))
            and not artifacts.best_path.is_file()):
        raise RuntimeError("finite best-model metadata has no reachable weights")

    n_params = int(sum(p.numel() for p in model.parameters()))
    final_train_loss = (float(losses[-1]) if losses else None)

    try:
        metrics = evaluate(model, val_loader, tokens_per_char=tokens_per_char, device=device)
        final_ce = float(metrics["ce"])
        final_ppl = float(metrics["ppl"])
        final_bits_per_token = float(metrics["bits_per_token"])
        final_bpc = (float(metrics["bpc"]) if metrics["bpc"] is not None else None)
        target_counts = {
            "expected_targets": int(metrics["expected_targets"]),
            "scored_targets": int(metrics["scored_targets"]),
            "excluded_targets": int(metrics["excluded_targets"]),
        }

        # Terminal metrics row: an empty history adopts the compact terminal schema. An established
        # training history retains its exact rectangular field set, but every non-terminal field is
        # blanked rather than republished under the terminal step.
        terminal_validation_fields = {
            "step":       completed_step,
            "val_ce":     final_ce,
            "val_ppl":    final_ppl,
            "val_bits_per_token": final_bits_per_token,
            "val_bpc":    (final_bpc if final_bpc is not None else float("nan")),
        }
        terminal_row = {
            "train_loss": float(losses[-1]) if losses else float("nan"),
            **terminal_validation_fields,
        }
        if artifacts._fieldnames is not None:
            established_fields = list(artifacts._fieldnames)
            missing_terminal_fields = sorted(
                set(terminal_validation_fields) - set(established_fields)
            )
            if missing_terminal_fields:
                raise ValueError(
                    "established metrics schema cannot represent terminal validation fields "
                    f"{missing_terminal_fields}"
                )
            terminal_row = {
                field: float("nan")
                for field in established_fields
            } | terminal_validation_fields
        # Run-wide best BEFORE the terminal save: primary is the better of the finite periodic best and
        # the final validation, so after maybe_save_best it equals the selected finite best.
        prior_best = artifacts.best_val_ppl
        artifacts.log_metrics(terminal_row)
        artifacts.maybe_save_best(completed_step, model, final_ppl)
        # After the terminal save the selection is live and MUST be reachable: maybe_save_best either
        # atomically replaced any stale file with the terminal weights (a recomputed cell) or left an
        # already-reachable earlier best (guarded above). Fail closed before publishing the success
        # contract so a stale-contract rerun never selects the previous cell's weights.
        if (not math.isfinite(float(artifacts.best_val_ppl))
                or (artifacts.save_model_artifacts and not artifacts.best_path.is_file())):
            raise RuntimeError("finite best-model metadata has no reachable weights")
        primary_val_ppl = (min(prior_best, final_ppl) if math.isfinite(prior_best) else final_ppl)
        best_val_ppl = (artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None)

        artifacts.save_json("validation_results.json", {
            "selection_split": "validation",
            "val_ce":          final_ce,
            "val_ppl":         final_ppl,
            "val_bits_per_token": final_bits_per_token,
            "val_bpc":         final_bpc,
            **target_counts,
            "val_expected_targets": target_counts["expected_targets"],
            "val_scored_targets": target_counts["scored_targets"],
            "val_excluded_targets": target_counts["excluded_targets"],
            "primary_val_ppl": float(primary_val_ppl),
            "best_val_ppl":    best_val_ppl,
            "best_step":       artifacts.best_step,
        })

        # Reproducibility provenance -- VALIDATION ONLY (test_loader=None): the ablation finalizer must
        # never open a test split.
        _write_provenance(
            artifacts, cfg, model, logger,
            train_loader=train_loader, val_loader=val_loader, test_loader=None,
            data_seed=data_seed, max_tokens=max_tokens, tokenizer_tag=tokenizer_tag,
        )
        try:
            artifacts.save_json("pure_path_report.json", _pure_path_report(
                cfg,
                artifacts.history,
                executable_build=model.executable_build,
            ))
        except Exception as exc:
            logger.warning("pure-path report failed (%s); skipped", exc)
        try:
            _run_figures_isolated(
                model,
                artifacts,
                losses,
                logger,
                generate_publication=False,
                generate_embedding_figures=cfg.generate_embedding_figures,
            )
        except Exception as exc:
            logger.warning(
                "isolated figure preparation failed (%s); validation results are saved",
                exc,
            )
    finally:
        # Strict raw-state reload + RNG restore, on BOTH the success and failure paths: the terminal
        # checkpoint below pairs the captured model with its optimizer moments, and the global RNG is
        # rewound to its captured post-final-step value.
        if terminal_state is not None:
            model.load_state_dict(terminal_state.raw_model_state)
            _restore_rng_state(terminal_state.rng_state)

    # Reached only after a successful validation pass. Persistent runs write the resumable terminal
    # checkpoint from the restored model; lightweight runs proceed directly to summary.json.
    terminal_checkpoint: Optional[str] = None
    if terminal_state is not None and artifacts.save_model_artifacts:
        checkpoint_path = artifacts.save_checkpoint(
            completed_step, model, terminal_state.optimizer, cfg,
            scaler=terminal_state.scaler,
            data_state=terminal_state.data_state,
        )
        terminal_checkpoint = str(checkpoint_path)

    figures_written = sorted(p.name for p in artifacts.run_dir.glob("*.png") if p.is_file())
    artifacts.save_json("summary.json", {
        **_conditional_artifact_fields(cfg),
        "selection_split":     "validation",
        "primary_val_ppl":     float(primary_val_ppl),
        "final_val_ce":        final_ce,
        "final_val_ppl":       final_ppl,
        "final_val_bits_per_token": final_bits_per_token,
        "final_val_bpc":       final_bpc,
        **target_counts,
        "val_expected_targets": target_counts["expected_targets"],
        "val_scored_targets": target_counts["scored_targets"],
        "val_excluded_targets": target_counts["excluded_targets"],
        "best_val_ppl":        best_val_ppl,
        "best_step":           artifacts.best_step,
        "n_steps":             completed_step,
        "n_params":            n_params,
        "final_train_loss":    final_train_loss,
        "wall_time_s":         (float(wall_time) if wall_time is not None else None),
        "terminal_checkpoint": terminal_checkpoint,
        "save_model_artifacts": artifacts.save_model_artifacts,
        "figures_written":     figures_written,
        "phi_chart_norm_route": _phi_chart_norm_route(model, cfg),
        "decode_tau_eff":      _decode_tau_eff(model, cfg),
    })

    return {
        "primary_val_ppl":     float(primary_val_ppl),
        "final_val_ppl":       final_ppl,
        "final_val_ce":        final_ce,
        "final_val_bits_per_token": final_bits_per_token,
        "final_val_bpc":       final_bpc,
        **target_counts,
        "val_expected_targets": target_counts["expected_targets"],
        "val_scored_targets": target_counts["scored_targets"],
        "val_excluded_targets": target_counts["excluded_targets"],
        "best_val_ppl":        best_val_ppl,
        "best_step":           artifacts.best_step,
        "final_train_loss":    final_train_loss,
        "n_params":            n_params,
        "terminal_checkpoint": terminal_checkpoint,
        "save_model_artifacts": artifacts.save_model_artifacts,
    }


def _pure_path_report(
    cfg: VFE3Config,
    history: List[Dict],
    *,
    executable_build: Optional[ExecutableBuildMetadata] = None,
) -> Dict:
    r"""Where a run sits relative to the theoretically pure path: the toggle states that define it plus
    the converged-state stress metrics that say whether the numerical guards stayed inert.

    This is a report, not a rejection gate: diagnostic GL-breaking and noncausal configurations
    remain executable and are labeled by independent certificate facets. ``on_pure_path`` preserves
    the legacy free-energy/decode schema. ``on_gauge_pure_path`` is the executable gauge facet and
    includes the active transport, norm, parameterization, clipping, temperature, encoder,
    HeadMixer, and immutable BlockMLP seams. ``on_causal_lm_path`` comes independently from both
    active attention-prior registrations. ``transport_exactness_status`` is affirmative only when
    complete runtime evidence warrants it; absent evidence is ``unknown``. The conjunctive
    ``on_theory_pure_path`` combines the explicitly enumerated ``theory_flags``.

    ``pure_flags`` intentionally retains its historical meaning and does not enumerate every
    default-off learned-scalar toggle. The covariance and E-step facets (``full_sigma_update``,
    ``gradient_e_step_update``, ``fixed_covariance_surrogate``, ``spd_retraction_*``) are derived
    from ``attention_mode`` and the branch the E-step executes, and read ``'not_applicable'`` when
    that route runs no such step. ``converged_stress`` reads the last finite value of each guard or
    flatness-residual column (``None`` if absent)."""
    def _last(key: str) -> Optional[float]:
        for r in reversed(history):
            v = r.get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
        return None
    from vfe3.geometry.groups import declared_invariant_families
    from vfe3.geometry.norms import get_norm_registration
    from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM, get_transport_registration
    from vfe3.attention_prior import get_prior_registration

    invariant_families = declared_invariant_families(cfg.gauge_group)
    family_group_invariant = cfg.family in invariant_families
    transport_registration = get_transport_registration(cfg.transport_mode)
    priorbank_head_evidence_mixer = bool(
        getattr(cfg, "use_priorbank_head_evidence_mixer", False)
    )
    head_mixer_compatibility = getattr(cfg, "head_mixer_compatibility", None)
    if head_mixer_compatibility is None:
        head_mixer_compatibility = VFE3Config.head_mixer_compatibility.fget(cfg)
    head_mixer_gauge_compatible = getattr(cfg, "head_mixer_gauge_compatible", None)
    if head_mixer_gauge_compatible is None:
        head_mixer_gauge_compatible = VFE3Config.head_mixer_gauge_compatible.fget(cfg)
    if executable_build is not None:
        block_mlp_build = executable_build.block_mlp
        block_mlp_active = block_mlp_build.enabled
        block_mlp_mode = block_mlp_build.mode
        block_mlp_covariance = block_mlp_build.covariance
        block_mlp_expansion = block_mlp_build.expansion
        block_mlp_activation = block_mlp_build.activation
        block_mlp_dropout = block_mlp_build.dropout
        block_mlp_covariance_floor = block_mlp_build.covariance_floor
        block_mlp_structural_mode = block_mlp_build.report_label
        block_mlp_covariance_contract = block_mlp_build.covariance_report_label
        block_mlp_intertwiner_compatible = block_mlp_build.intertwiner_compatible
    else:
        block_mlp_active = bool(getattr(cfg, "use_block_mlp", False))
        block_mlp_mode = getattr(cfg, "block_mlp_mode", "coordinate")
        block_mlp_covariance = getattr(cfg, "block_mlp_covariance", "passthrough")
        block_mlp_expansion = int(getattr(cfg, "block_mlp_expansion", 4))
        block_mlp_activation = getattr(cfg, "block_mlp_activation", "gelu")
        block_mlp_dropout = float(getattr(cfg, "block_mlp_dropout", 0.0))
        block_mlp_covariance_floor = float(getattr(cfg, "block_mlp_covariance_floor", 1e-4))
        registration = get_block_mlp_registration(block_mlp_mode)
        block_mlp_structural_mode = (
            registration.report_label(block_mlp_covariance)
            if block_mlp_active else "disabled"
        )
        block_mlp_covariance_contract = (
            "not_applicable" if not block_mlp_active else
            "delta_full_plus_covariant_floor"
            if block_mlp_covariance == "delta_full" else block_mlp_covariance
        )
        block_mlp_intertwiner_compatible = (
            registration.intertwiner_compatible if block_mlp_active else True
        )
    runtime_exactness_key = transport_registration.runtime_exactness_key
    if runtime_exactness_key is None:
        # No runtime exactness certificate is registered for this transport: the flat cocycle is
        # exact by construction (no edge factor to approximate), so there is nothing to observe.
        regime_ii_covariant_exact = True
        regime_ii_covariant_exactness = "not_applicable"
    elif not family_group_invariant:
        regime_ii_covariant_exact = False
        regime_ii_covariant_exactness = "diagonal_projection_approximation"
    else:
        observations = [row.get(runtime_exactness_key) for row in history]
        finite_observations = [
            value for value in observations
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
        complete = bool(observations) and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in observations
        )
        regime_ii_covariant_exact = complete and all(value >= 0.5 for value in observations)
        regime_ii_covariant_exactness = (
            "exact_valid_spd_factorization" if regime_ii_covariant_exact else
            "jitter_recovered_approximation" if any(value < 0.5 for value in finite_observations) else
            "unknown_missing_runtime_evidence"
        )
    if runtime_exactness_key is None:
        transport_exactness_status = "not_applicable"
    elif not transport_registration.gauge_equivariant or not family_group_invariant:
        transport_exactness_status = "approximate"
    elif regime_ii_covariant_exactness == "unknown_missing_runtime_evidence":
        transport_exactness_status = "unknown"
    else:
        transport_exactness_status = "exact" if regime_ii_covariant_exact else "approximate"

    # Covariance / E-step facets come from the EXECUTED branch of e_step_iteration, not raw toggles:
    #   transported           -> the configured updater ('gradient' or 'mm_exact'); the covariance
    #                            moves unless skip_belief_sigma_update, and only the gradient updater
    #                            applies the affine SPD retraction (mm_exact blends precisions);
    #   projected_conditional -> a relational-filtering gradient step on the mean only (no sigma arm);
    #   projected_predictive  -> the predictive readout replaces the mean; no E-step descent runs.
    # Projected routes keep sigma = belief.sigma, so their covariance is frozen at encode whatever
    # skip_belief_sigma_update says, except that predictive_covariance='moment_matched' assigns the
    # predictive kernel's covariance (a readout: no retraction). A facet with no executed
    # counterpart is 'not_applicable'.
    from vfe3.inference.e_step import canonical_e_step_update
    attention_mode = getattr(cfg, "attention_mode", "transported")
    transported = attention_mode == "transported"
    e_step_updater = canonical_e_step_update(getattr(cfg, "e_step_update", "gradient"))
    predictive_covariance_updated = (
        attention_mode == "projected_predictive"
        and getattr(cfg, "predictive_covariance", "passthrough") == "moment_matched")
    covariance_updated = (transported and not cfg.skip_belief_sigma_update) or predictive_covariance_updated
    gradient_e_step_update: 'bool | str' = (
        e_step_updater == "gradient" if transported
        else True if attention_mode == "projected_conditional"
        else "not_applicable"
    )
    spd_retraction_applies = (transported and not cfg.skip_belief_sigma_update
                              and e_step_updater == "gradient")

    sigma_max = getattr(cfg, "sigma_max", 10.0)
    spd_trust_region = getattr(cfg, "e_sigma_q_trust", 5.0)
    spd_retraction_trust_projection_active: 'bool | str' = (
        spd_trust_region is not None and float(spd_trust_region) > 0.0)
    # The affine SPD route calls an exp_bounded kernel with fixed [-50, 50] bounds.
    spd_retraction_exponential_clipping_active: 'bool | str' = True
    spd_route_parts = ["airm"]
    if spd_retraction_trust_projection_active:
        spd_route_parts.append("trust_projected")
    if spd_retraction_exponential_clipping_active:
        spd_route_parts.append("exp_clipped")
    if sigma_max is not None:
        spd_route_parts.append("spectral_cap")
    spd_retraction_route = "_".join(spd_route_parts)
    spd_retraction_exact: 'bool | str' = len(spd_route_parts) == 1
    if not spd_retraction_applies:
        spd_retraction_route = "not_applicable"
        spd_retraction_exact = "not_applicable"
        spd_retraction_trust_projection_active = "not_applicable"
        spd_retraction_exponential_clipping_active = "not_applicable"

    beta_prior = get_prior_registration(getattr(cfg, "beta_attention_prior", "causal"))
    causal_flags = {
        "beta_prior_causal": beta_prior.is_causal_for_next_token_scoring(cfg),
        "inference_updates_causal": cfg.gradient_mode == "filtering",
    }
    on_causal_lm_path = all(causal_flags.values())

    block_norm_name = getattr(cfg, "norm_type_block", "none")
    final_norm_name = getattr(cfg, "norm_type_final", "none")
    block_norm_registration = get_norm_registration(block_norm_name)
    final_norm_registration = get_norm_registration(final_norm_name)
    transport_chart_max_norm = getattr(cfg, "transport_chart_max_norm", None)

    conditional_fields = _conditional_artifact_fields(cfg)
    pure_flags = {
        "canonical_attention_entropy": bool(cfg.include_attention_entropy) and not conditional_fields,
        "flat_transport":              cfg.transport_mode == "flat",
        "constant_lambda_alpha":       cfg.lambda_alpha_mode == "constant",
        "prior_bank_decode":           cfg.decode_mode != "linear",
        "no_head_mixer":               not cfg.use_head_mixer,
        "no_priorbank_head_evidence_mixer": not priorbank_head_evidence_mixer,
        # Executed-route facet: False on projected routes, where sigma stays at encode, unless
        # predictive_covariance='moment_matched' assigns the predictive kernel's covariance.
        "full_sigma_update":           covariance_updated,
        "no_block_mlp":                not block_mlp_active,
        # The E-STEP UPDATE RULE axis (audit 2026-07-26 D-03). 'mm_exact' is the closed-form
        # minimizer of the STRICT-PAIR-MASKED surrogate, which excludes the structural E_ii = 0
        # self-pairs and is therefore not a majorizer of the canonical frozen-attention objective
        # (see the e_step_update note and D-01). Without this key a run whose E-step descends that
        # surrogate published on_pure_path: True with no record of the substitution anywhere: the
        # only trace, state_dependent_alpha_majorizer, is False at the pure lambda_alpha_mode=
        # 'constant', so the ledger asserted purity it had not checked. 'not_applicable' on
        # projected_predictive, where no E-step descent runs; like any non-False facet it does not
        # lower on_pure_path (that route is already off it through canonical_attention_entropy).
        "gradient_e_step_update":      gradient_e_step_update,
    }
    # Second, INDEPENDENT purity axis (audit 2026-07-01 F8): the gauge / model-channel path. Keyed
    # on pos_rotation itself rather than the RoPE sub-toggles (rope_full_gauge / rope_on_value),
    # which are inert while RoPE is off -- those are reported in config_toggles for transparency.
    gauge_flags = {
        "learned_gauge_transport":   cfg.gauge_transport == "on",
        "transport_gauge_equivariant": transport_registration.gauge_equivariant,
        "no_positional_rotation":    cfg.pos_rotation == "none",
        "family_group_invariant":    family_group_invariant,
        "head_mixer_intertwiner_compatible": head_mixer_gauge_compatible,
        "block_mlp_intertwiner_compatible": block_mlp_intertwiner_compatible,
        "block_norm_gauge_equivariant": block_norm_registration.gauge_equivariant,
        "final_norm_gauge_equivariant": final_norm_registration.gauge_equivariant,
        "no_fixed_coordinate_spectral_cap": sigma_max is None,
        "no_phi_mstep_chart_cap": getattr(cfg, "phi_mstep_max_matrix_norm", None) is None,
        "no_pullback_trust_region": getattr(cfg, "m_phi_update_mode", "adamw") != "pullback_group",
        "no_transport_exponential_clipping": (
            transport_chart_max_norm is not None
            and float(transport_chart_max_norm) <= float(TRANSPORT_CLAMP_MAX_NORM)
        ),
        "no_e_step_phi_retraction_clipping": float(getattr(cfg, "e_phi_lr", 0.0)) == 0.0,
        "no_additive_encoder_control": getattr(cfg, "encode_mode", "per_token") != "per_token_additive",
    }
    on_gauge_pure_path = all(gauge_flags.values())
    theory_flags = {
        **pure_flags,
        "gauge_pure_path": on_gauge_pure_path,
        "causal_lm_path": on_causal_lm_path,
        "transport_exact_when_applicable": transport_exactness_status in {"exact", "not_applicable"},
    }
    return {
        **conditional_fields,
        "on_pure_path":       all(pure_flags.values()),
        "pure_flags":         pure_flags,
        "gauge_flags":        gauge_flags,
        "on_gauge_pure_path": on_gauge_pure_path,
        "causal_flags":       causal_flags,
        "on_causal_lm_path": on_causal_lm_path,
        "transport_exactness_status": transport_exactness_status,
        "theory_flags":       theory_flags,
        "on_theory_pure_path": all(theory_flags.values()),
        "config_toggles": {
            "include_attention_entropy":    bool(cfg.include_attention_entropy),
            "transport_mode":               cfg.transport_mode,
            "lambda_alpha_mode":            cfg.lambda_alpha_mode,
            "lambda_beta":                  float(cfg.lambda_beta),
            "decode_mode":                  cfg.decode_mode,
            "use_head_mixer":               bool(cfg.use_head_mixer),
            "use_priorbank_head_evidence_mixer": priorbank_head_evidence_mixer,
            "encode_mode":                  getattr(cfg, "encode_mode", "per_token"),
            "beta_attention_prior":         getattr(cfg, "beta_attention_prior", "causal"),
            "use_block_mlp":                block_mlp_active,
            "block_mlp_mode":               block_mlp_mode,
            "block_mlp_covariance":         block_mlp_covariance,
            "block_mlp_expansion":          block_mlp_expansion,
            "block_mlp_covariance_floor":   block_mlp_covariance_floor,
            "block_mlp_activation":         block_mlp_activation,
            "block_mlp_dropout":            block_mlp_dropout,
            "block_mlp_lr_resolved":        float(
                getattr(cfg, "m_block_mlp_lr", None)
                if getattr(cfg, "m_block_mlp_lr", None) is not None else getattr(cfg, "m_p_mu_lr", 0.0)),
            "block_mlp_structural_mode":    block_mlp_structural_mode,
            "block_mlp_covariance_contract": block_mlp_covariance_contract,
            "priorbank_head_evidence_role": (
                "decoder_kl_irrep_weights" if priorbank_head_evidence_mixer else "inactive"
            ),
            "head_mixer_compatibility":      head_mixer_compatibility,
            "head_mixer_gauge_compatible":   head_mixer_gauge_compatible,
            "gauge_transport":              cfg.gauge_transport,
            "pos_rotation":                 cfg.pos_rotation,
            "rope_full_gauge":              bool(cfg.rope_full_gauge),
            "rope_on_value":                bool(cfg.rope_on_value),
            # WHICH RoPE transport regime the pair energy actually used (audit 2026-07-26 A-02),
            # recorded in the same spirit as the transport exactness block. Means-only rotates the
            # MEAN by R_i Omega_ij R_j^T while the covariance keeps the bare Omega_ij, so the
            # transported pair is the pushforward of the key under no single operator and
            # E_ij = D(q_i || Omega_ij# q_j) does not hold literally -- deliberate (R Sigma R^T is
            # dense, hence rope_full_gauge=True requiring gaussian_full) but previously unrecorded.
            "rope_pair_energy_exactness":   (
                "not_applicable" if cfg.pos_rotation == "none"
                else "exact_rope_congruence" if cfg.rope_full_gauge
                else "means_only_unrotated_covariance"
            ),
            "skip_belief_sigma_update":      bool(cfg.skip_belief_sigma_update),
            "predictive_covariance":         getattr(cfg, "predictive_covariance", "passthrough"),
            "predictive_covariance_floor":   float(getattr(cfg, "predictive_covariance_floor", 1e-4)),
            "gauge_group":                   cfg.gauge_group,
            "family":                        cfg.family,
            "group_invariant_families":      list(invariant_families),
            # The update rule and its damping, so a reader can tell WHICH objective the E-step
            # descended and how far it stepped toward the surrogate's minimizer (audit D-03).
            "e_step_update":                 getattr(cfg, "e_step_update", "gradient"),
            "mm_damping":                    float(getattr(cfg, "mm_damping", 1.0)),
            # Truthful fixed-surrogate ledger (C6): these derived booleans expose when the run
            # intentionally freezes a state-dependent quantity rather than following its full
            # joint objective. Defaults are False, preserving the pure path. The covariance is
            # frozen on projected routes (unless predictive_covariance='moment_matched') as well
            # as under skip_belief_sigma_update.
            "fixed_covariance_surrogate":   not covariance_updated,
            "attention_mode":               attention_mode,
            "state_dependent_alpha_majorizer": (
                getattr(cfg, "e_step_update", "gradient") == "mm_exact"
                and cfg.lambda_alpha_mode in ("state_dependent", "state_dependent_per_coord")
            ),
            # regime_ii_covariant under gaussian_diagonal is a CONTROLLED APPROXIMATION (the
            # diagonal cone is not closed under GL congruence Omega Sigma Omega^T -- audit C5),
            # so a diagonal covariant run is never reported as exact Route B.
            "regime_ii_covariant_exact":    regime_ii_covariant_exact,
            "regime_ii_covariant_exactness": regime_ii_covariant_exactness,
            "spd_retraction_route":         spd_retraction_route,
            "spd_retraction_exact":         spd_retraction_exact,
            "spd_retraction_trust_projection_active": spd_retraction_trust_projection_active,
            "spd_retraction_exponential_clipping_active": spd_retraction_exponential_clipping_active,
            "norm_type_block":              block_norm_name,
            "norm_type_final":              final_norm_name,
            "layernorm_affine":             bool(getattr(cfg, "layernorm_affine", False)),
            "m_phi_update_mode":            getattr(cfg, "m_phi_update_mode", "adamw"),
            "m_phi_group_trust_radius":      float(getattr(cfg, "m_phi_group_trust_radius", 0.1)),
            "phi_mstep_max_matrix_norm":     getattr(cfg, "phi_mstep_max_matrix_norm", None),
            "transport_chart_max_norm":      transport_chart_max_norm,
            "e_phi_lr":                     float(getattr(cfg, "e_phi_lr", 0.0)),
            # Covariance class of the ACTIVE transport (audit C7), owned by its complete registry
            # record. An unregistered mode fails closed instead of inventing report metadata.
            "transport_covariance_class":   transport_registration.covariance_class,
        },
        "converged_stress": {k: _last(k) for k in (
            "guard_sigma_floor_frac", "guard_sigma_ceil_frac", "guard_energy_klmax_frac",
            "guard_selfdiv_klmax_frac", "nonfinite_frac", "renyi_band_frac",
            "cocycle_residual", "holonomy_wilson", "gauge_invariant_spread")},
    }


def _save_figures(
    artifacts: RunArtifacts,
    losses:    Optional[List[float]],
    logger:    logging.Logger,
) -> None:
    r"""Best-effort publication figures from the logged history (no model re-run)."""
    try:
        from vfe3.viz import figures as raw_figs

        class _SafePyplot:
            r"""Delegate pyplot operations while making a failed renderer's ``None`` close a no-op."""

            def __getattr__(self, name: str) -> object:
                return getattr(raw_figs.plt, name)

            def close(self, figure: object = None) -> None:
                if figure is not None:
                    raw_figs.plt.close(figure)

        class _IsolatedFigures:
            r"""Proxy plot calls so one failed renderer cannot abort or leak into the next one."""

            def __getattr__(self, name: str) -> object:
                if name == "plt":
                    return _SafePyplot()
                value = getattr(raw_figs, name)
                if not name.startswith("plot_") or not callable(value):
                    return value

                def _isolated(*args: object, **kwargs: object) -> object:
                    before = set(raw_figs.plt.get_fignums())
                    try:
                        path_value = kwargs.get("path")
                        if path_value is None:
                            return value(*args, **kwargs)
                        final_path = Path(path_value)
                        with _unique_sibling_output_temp(final_path) as temporary_path:
                            isolated_kwargs = dict(kwargs)
                            isolated_kwargs["path"] = str(temporary_path)
                            figure = value(*args, **isolated_kwargs)
                            if (not temporary_path.is_file()
                                    or temporary_path.stat().st_size == 0):
                                raise RuntimeError(
                                    f"figure {name!r} did not write a nonempty temporary output"
                                )
                            _atomic_replace(final_path, temporary_path)
                            return figure
                    except Exception as exc:
                        for number in set(raw_figs.plt.get_fignums()) - before:
                            raw_figs.plt.close(number)
                        logger.warning("figure %s failed (%s); remaining figures continue", name, exc)
                        return None

                return _isolated

        figs = _IsolatedFigures()
        figs.set_publication_style()
        run = artifacts.run_dir

        def _aligned(key: str) -> tuple:
            r"""Aligned (step, value) for a history column, dropping pre-first-eval NaN rows."""
            xs, ys = [], []
            for i, r in enumerate(artifacts.history):
                if key in r and math.isfinite(r[key]):
                    xs.append(r.get("step", i))
                    ys.append(r[key])
            return xs, ys

        def _hist_subset(keys: tuple) -> Optional[Dict]:
            r"""A ``{step, key: [...]}`` history dict over ``keys`` present (finite on >= 1 row), each a
            full-length column with NaN where missing so an eval-cadence key keeps its step alignment and
            the dashboard masks it per series. Returns None when no key is present (caller skips)."""
            present = [k for k in keys
                       if any(k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])
                              for r in artifacts.history)]
            if not present:
                return None
            cols: Dict = {"step": [r.get("step", i) for i, r in enumerate(artifacts.history)]}
            for k in present:
                cols[k] = [float(r[k]) if (k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k]))
                           else float("nan") for r in artifacts.history]
            return cols

        readout_keys = tuple({key for row in artifacts.history for key in row
                              if key.startswith("relational_")})
        if readout_keys:
            readout_history = _hist_subset(readout_keys + (
                "attn_entropy", "attn_entropy_min", "effective_rank", "belief_cond_median"))
            if readout_history is not None:
                fig = figs.plot_readout_diagnostics(
                    readout_history, path=str(run / "readout_diagnostics.png"))
                figs.plt.close(fig)

        if losses:
            # losses is one entry per optimizer step, so the 1-based step index IS the x-axis.
            n = len(losses)
            steps_per_epoch = next(
                (
                    int(r["steps_per_epoch"])
                    for r in artifacts.history
                    if isinstance(r.get("steps_per_epoch"), (int, float))
                    and math.isfinite(float(r["steps_per_epoch"]))
                    and int(r["steps_per_epoch"]) > 0
                ),
                0,
            )
            epoch_boundaries = (
                list(range(steps_per_epoch, n + 1, steps_per_epoch))
                if steps_per_epoch
                else None
            )
            fig = figs.plot_trajectory(
                losses, list(range(1, n + 1)), ylabel="train CE (nats/token)",
                title="Training cross-entropy", color=figs._CB[0],
                smooth=max(25, n // 240), annotate_final=True,
                epoch_boundaries=epoch_boundaries,
                path=str(run / "loss_curve.png"))
            figs.plt.close(fig)
        sx, sy = _aligned("val_ppl")
        if sy:
            fig = figs.plot_trajectory(
                sy, sx, ylabel="validation perplexity", title="Validation perplexity",
                color=figs._CB[1], logy=True, smooth=max(5, len(sy) // 80), annotate="min",
                path=str(run / "val_ppl.png"))
            figs.plt.close(fig)
        # Gauge-geometry trajectories (diagnostics tier): curvature proxy + gauge-trace spread.
        hx, hy = _aligned("holonomy_deviation")
        if hy:
            # Heavy-tailed (median ~1e-3, rare spikes ~1e3): log y + a median reference; NOT smoothed,
            # so the curvature spikes survive.
            flat_transport = getattr(getattr(artifacts, "cfg", None), "transport_mode", "flat") == "flat"
            fig = figs.plot_trajectory(
                hy, hx,
                ylabel=("numerical closure residual" if flat_transport
                         else r"$\langle\|H_{ijk}-I\|_F\rangle$"),
                title=("Numerical closure of nominally flat transport" if flat_transport
                       else "Holonomy curvature (frame-dependent Frobenius)"),
                color=figs._CB[2],
                logy=True, median_line=True, annotate="max",
                path=str(run / "holonomy.png"))
            figs.plt.close(fig)
        gx, gy = _aligned("gauge_trace_spread")
        if gy:
            fig = figs.plot_trajectory(
                gy, gx, ylabel=r"std $\log|\det\Omega|$", title="Gauge trace spread",
                color=figs._CB[3], smooth=max(5, len(gy) // 60), annotate_final=True,
                path=str(run / "gauge_trace_spread.png"))
            figs.plt.close(fig)
        # Optimization and endpoint trends (history-only; no model re-run): the pre-clip gradient
        # norm, belief-covariance conditioning, and the per-eval E-step endpoint delta.
        nx, ny = _aligned("grad_norm")
        if ny:
            fig = figs.plot_trajectory(
                ny, nx, ylabel=r"$\|\nabla\|_2$ (pre-clip)", title="Gradient norm",
                color=figs._CB[5 % len(figs._CB)], logy=True, smooth=max(5, len(ny) // 80),
                annotate="max", path=str(run / "grad_norm.png"))
            figs.plt.close(fig)
        # M-step per-role gradient-norm decomposition (mu / sigma / phi): the parameter-learning
        # channels the aggregate grad_norm.png folds together. Columns logged by train_step (aggregated
        # by each optimizer group's "role" tag, so the live tables are attributed correctly under any
        # config); present only on a run that captured step_metrics, so gate on their presence (the CSV
        # stays rectangular per run). A SEPARATE figure from the aggregate, same pre-clip magnitudes.
        gd_keys = ("grad_norm_mu", "grad_norm_sigma", "grad_norm_phi")
        gd_present = [k for k in gd_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if gd_present:
            gd_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in gd_present)]
            if gd_rows:
                hist_gd = {"step": [r.get("step", i) for i, r in enumerate(gd_rows)],
                           **{k: [r[k] for r in gd_rows] for k in gd_present}}
                fig = figs.plot_grad_norm_decomposition(hist_gd, path=str(run / "grad_norm_decomposition.png"))
                figs.plt.close(fig)
        # E-step belief-gradient decomposition (mu / sigma / phi): the INFERENCE analogue of the M-step
        # figure above -- ||grad F|| over the belief tuple per inner-loop component, logged by train_step
        # from model.forward's estep_grad_out. Accumulated runs prefer the explicitly named arithmetic
        # microbatch means; single-batch runs retain the historical column names. Same presence gate;
        # independent of the M-step columns, so build its own row set.
        eg_mean_keys = (
            "estep_grad_norm_mu_microbatch_mean",
            "estep_grad_norm_sigma_microbatch_mean",
            "estep_grad_norm_phi_microbatch_mean",
        )
        eg_keys = (eg_mean_keys if any(any(k in r for k in eg_mean_keys)
                                      for r in artifacts.history)
                   else ("estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi"))
        eg_present = [k for k in eg_keys
                      if any(k in r and math.isfinite(r[k]) for r in artifacts.history)]
        if eg_present:
            eg_rows = [r for r in artifacts.history
                       if all(k in r and math.isfinite(r[k]) for k in eg_present)]
            if eg_rows:
                hist_eg = {"step": [r.get("step", i) for i, r in enumerate(eg_rows)],
                           **{k: [r[k] for r in eg_rows] for k in eg_present}}
                fig = figs.plot_estep_grad_norm_decomposition(hist_eg, path=str(run / "estep_grad_norm_decomposition.png"))
                figs.plt.close(fig)
        cx, cy = _aligned("belief_cond_median")
        if cy:
            fig = figs.plot_trajectory(
                cy, cx, ylabel=r"median $\lambda_{\max}/\lambda_{\min}$",
                title="Belief covariance conditioning", color=figs._CB[6 % len(figs._CB)],
                logy=True, smooth=max(5, len(cy) // 80), annotate="max",
                path=str(run / "belief_condition.png"))
            figs.plt.close(fig)
        ex, ey = _aligned("estep_f_drop")
        if ey:
            _endpoint_artifact = _estep_endpoint_artifact({})
            fig = figs.plot_trajectory(
                ey, ex, ylabel=r"$F_{\mathrm{end}}-F_{\mathrm{start}}$ (inner E-step)",
                title=_endpoint_artifact["title"], color=figs._CB[2 % len(figs._CB)],
                median_line=True, path=str(run / _endpoint_artifact["filename"]))
            figs.plt.close(fig)
        # Free-energy figures: the per-token budget decomposition (snapshot + early/mid/late evolution)
        # and, separately, the final-iterate objective/CE relationship over training. Both need every plotted term
        # finite, so rows before the first eval (NaN val_*) are dropped.
        fe_keys = ("self_coupling", "belief_coupling", "attention_entropy", "val_ce")
        fe_rows = [r for r in artifacts.history
                   if all(k in r and math.isfinite(r[k]) for k in fe_keys)]
        if fe_rows:
            cfg = getattr(artifacts, "cfg", None)
            hist = {"step": [r.get("step", i) for i, r in enumerate(fe_rows)],
                    **{k: [r[k] for r in fe_rows] for k in fe_keys}}
            # Scale the coupling terms by the static config lambda_beta scalar.
            lam = getattr(cfg, "lambda_beta", 1.0)
            iae = getattr(cfg, "include_attention_entropy", True)
            fig = figs.plot_free_energy_decomposition(
                hist, lambda_beta=lam, include_attention_entropy=iae,
                divergence_family=getattr(cfg, "divergence_family", "renyi"),
                path=str(run / "free_energy_decomposition.png"))
            figs.plt.close(fig)
            fig = figs.plot_free_energy_relationship(
                hist, lambda_beta=lam, include_attention_entropy=iae,
                divergence_family=getattr(cfg, "divergence_family", "renyi"),
                path=str(run / "free_energy_relationship.png"))
            figs.plt.close(fig)
        # Geometry / SPD / Fisher health dashboard (history-only): the gauge, belief-spectrum, guard,
        # and numerical-safety scalars diagnostics() logs to metrics.csv but that no standard figure
        # surfaced. The panels self-gate, so a run missing a column simply drops that panel.
        hist_geom = _hist_subset((
            "holonomy_wilson", "cocycle_residual", "holonomy_deviation",
            "gauge_invariant_spread", "gauge_head_logdet_spread", "phi_norm_mean", "phi_norm_std",
            "phi_matrix_norm_p95", "phi_matrix_norm_p99", "phi_matrix_norm_max",
            "phi_exp_clamp_frac", "phi_exp_scale_min",
            "vertex_cond_median", "vertex_cond_p95", "vertex_cond_p99",
            "phi_chart_projected_fraction", "phi_chart_projection_scale_min",
            "phi_chart_projection_ms",
            "belief_cond_p95", "belief_cond_max", "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
            "fisher_trace_mean", "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
            "guard_energy_klmax_frac", "guard_selfdiv_klmax_frac", "nonfinite_frac", "renyi_band_frac",
            "attn_entropy_min", "attn_entropy_collapsed_heads"))
        if hist_geom:
            fig = figs.plot_geometry_health(
                hist_geom,
                transport_mode=getattr(getattr(artifacts, "cfg", None), "transport_mode", "flat"),
                family=getattr(getattr(artifacts, "cfg", None), "family", None),
                path=str(run / "geometry_health.png"),
            )
            figs.plt.close(fig)
        # E-step inference-quality dashboard: the inner-loop F-drop, the nondecreasing fraction, and the
        # last-iter belief residuals -- the E-step evidence the single estep_f_drop curve does not show.
        hist_estep = _hist_subset((
            "estep_f_drop", "estep_f_nondecreasing_frac",
            "estep_r_mu_last", "estep_r_sigma_last", "estep_r_phi_last",
            "estep_fp_kl", "estep_fp_mu_rms", "estep_fp_sigma_rms", "estep_fp_phi_rms",
            "estep_target_gap", "estep_beta_js", "estep_alpha_rms_delta"))
        if hist_estep:
            fig = figs.plot_estep_quality(hist_estep, path=str(run / "estep_quality.png"))
            figs.plt.close(fig)
        # Held-out validation-sanity dashboard: the per-eval probes (_val_diagnostics) that were CSV-only
        # -- generalization gap, positional loss, causal/attention sanity, and the held-out geometry.
        hist_val = _hist_subset((
            "generalization_gap", "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
            "val_future_leakage", "val_row_sum_error", "val_pos_content_r2",
            "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
            "val_holonomy_wilson", "val_cocycle_residual", "val_gauge_invariant_spread",
            "val_belief_cond_p95", "val_fisher_trace_mean", "val_guard_sigma_floor_frac",
            "val_guard_sigma_ceil_frac", "val_guard_energy_klmax_frac",
            "val_phi_norm_mean", "val_phi_norm_std", "val_phi_matrix_norm_p95",
            "val_phi_matrix_norm_p99", "val_phi_matrix_norm_max", "val_phi_exp_clamp_frac",
            "val_phi_exp_scale_min", "val_vertex_cond_p99"))
        if hist_val:
            fig = figs.plot_validation_sanity(
                hist_val,
                family=getattr(getattr(artifacts, "cfg", None), "family", None),
                path=str(run / "validation_sanity.png"),
            )
            figs.plt.close(fig)
        # Pullback-group optimizer dashboard: ridge-direction reshaping, certified conditioning,
        # accepted factor scale/chart health, per-role weight norms, and gradient-to-weight ratio.
        hist_opt = _hist_subset((
            "phi_ridge_direction_cosine_mean",
            "phi_pullback_damped_gen_cond_median", "phi_pullback_damped_gen_cond_max",
            "phi_group_trust_scale_mean", "phi_group_trust_scale_min",
            "phi_group_active_rows", "phi_group_chart_norm_max",
            "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
            "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))
        if hist_opt:
            fig = figs.plot_optimizer_geometry(hist_opt, path=str(run / "optimizer_geometry.png"))
            figs.plt.close(fig)
    except Exception as exc:                                    # never let a plot kill a finished run
        logger.warning("figure generation failed (%s); numeric results are still saved", exc)
