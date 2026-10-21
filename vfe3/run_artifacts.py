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
      loss_curve.png     per-step training-loss trajectory (the optimizer objective, metrics.csv's
                         train_loss) at the absolute training step
      val_ppl.png        validation perplexity trajectory (log-y, best marked)
      free_energy_decomposition.png   per-token F budget snapshot + early/mid/late evolution
      free_energy_relationship.png     final-iterate objective vs validation CE
      canonical_free_energy_{decomposition,relationship}.png   the same on the projected routes under
                         diagnose_canonical_free_energy: canonical F at the captured beliefs (a diagnostic)

``RunArtifacts`` is OPT-IN: ``train`` only touches it when an instance is passed, so the silent
path (``artifacts=None``) writes nothing and is unchanged. ``finalize_run`` reloads the best-val
checkpoint, scores the held-out test split, and writes the summary + figures. Figure generation
is best-effort (a plotting/dependency error is logged, never fatal) so the numeric results
survive a viz problem.
"""

import ast
import copy
import csv
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple,
    Union,
)

import torch

from vfe3.contracts import (
    VFE_MODEL_FAMILY,
    is_vfe_config,
    is_vfe_model,
    model_family_of,
)
from vfe3.config import (
    VFE3Config,
    config_from_serialized,
    serialize_config,
)
from vfe3.contracts import DataState, DataStateBuffer, ExecutableBuildMetadata
from vfe3.gauge_optim import FRAME_CONDITION_GROUPS, bound_frame_condition_if_configured_
from vfe3.model.minimal_agent import READOUT_COVARIANCES
from vfe3.model.block_mlp import (
    block_mlp_build_metadata,
    get_block_mlp_registration,
)
from vfe3.metrics_schema import normalize_metric_row, split_spike_rows
from vfe3.process_utils import run_process_tree
from vfe3.runtime import deterministic_state, ieee_fp32_matmul

if TYPE_CHECKING:                                        # forward ref only: train imports RunArtifacts
    from vfe3.train import TrainingTerminalState         # at top level, so a runtime import here cycles


def _serialize_run_config(cfg: Any) -> Dict[str, object]:
    r"""Serialize a run config of any model family (the VFE path is :func:`vfe3.config.serialize_config`).

    A config of another family (``cfg.model_family``, e.g. :class:`vfe3.baseline.BaselineConfig`)
    serializes itself; its mapping carries the ``model_family`` tag every loader dispatches on.
    """
    if is_vfe_config(cfg):
        return serialize_config(cfg)
    return cfg.serialize()


def _config_from_run_serialized(
    payload: Mapping[str, object],

    *,
    source:  str,
) -> Any:
    r"""Strictly rebuild a run config from its serialized mapping, dispatching on ``model_family``.

    An untagged mapping is a VFE config (:func:`vfe3.config.config_from_serialized`, unchanged);
    ``'conventional_transformer'`` builds :class:`vfe3.baseline.BaselineConfig`. Any other tag, or a
    malformed baseline mapping, raises ``ValueError`` like the VFE loader.
    """
    family = model_family_of(payload) if isinstance(payload, Mapping) else VFE_MODEL_FAMILY
    if family == VFE_MODEL_FAMILY:
        return config_from_serialized(payload, source=source)
    from vfe3.baseline import MODEL_FAMILY as BASELINE_FAMILY, BaselineConfig
    if family != BASELINE_FAMILY:
        raise ValueError(f"{source}: unknown model_family {family!r}")
    return BaselineConfig.from_serialized(payload, source=source)


def _relational_artifact_fields(cfg: VFE3Config) -> Dict[str, object]:
    """Describe full-width projected routing and its configured receiver update."""
    if not is_vfe_config(cfg):                        # another model family has no relational route
        return {}
    H, K = cfg.n_heads, cfg.embed_dim
    # Namespace configs without the property (tests, old records) have equal heads.
    dims = getattr(cfg, "head_dims", None) or [K // H] * H
    full_link = cfg.attention_mode == "projected_gradient"
    # Every head reads and writes the full state: sum_h d_h K = K^2 per map whatever the partition.
    layer_counts = {"score": 2*K*K, "content": 2*K*K}
    if len(set(dims)) == 1:                       # equal heads: (H, d, K) stacks
        d = dims[0]
        rank: object = d
        score_shape: object = [H, d, K]
        content_shapes: object = [[H, d, K], [H, K, d]]
    else:                                         # mixed so_n tower: one (d_h, K) map per head
        rank = list(dims)
        score_shape = [[d, K] for d in dims]
        content_shapes = [[[d, K] for d in dims], [[K, d] for d in dims]]
    edges = getattr(cfg, "content_link_edges", "off")
    edge_fields: Dict[str, object] = {}
    if edges != "off":
        # Edge factor maps (content_link_edges): W_B (K r x K) and W_A (K r x K) or the shared basis A (K x r).
        r = cfg.content_link_edge_rank
        layer_counts["content"] += K * r * K + (K * r * K if edges == "lowrank" else K * r)
        edge_fields = {"content_link_edges": edges, "content_link_edge_rank": r}
        if getattr(cfg, "content_link_latent", False):
            # Latent receiver factors (MAP inference per context): no parameters, only the inference settings.
            edge_fields.update({name: getattr(cfg, name) for name in (
                "content_link_latent", "content_link_latent_steps", "content_link_latent_sigma",
                "content_link_latent_damping", "content_link_loop_weight", "content_link_loop_window")})
    # predictive_readout='minimal_agent' (vfe3/model/minimal_agent.py): S_Q (K^2) and channel_log_noise (H) score,
    # T (K^2) is both the content and the tied key; no raw_score_k and no O (the receiving map is derived).
    minimal_agent = getattr(cfg, "predictive_readout", "kernel") == "minimal_agent"
    agent_fields: Dict[str, object] = {}
    if minimal_agent:
        layer_counts = {"score": K*K + H, "content": K*K}
        content_shapes = [[H, dims[0], K]]
        agent_fields = {"predictive_readout": "minimal_agent",
                        "minimal_agent_inference": cfg.minimal_agent_inference,
                        "minimal_agent_channel_noise_init": float(cfg.minimal_agent_channel_noise),
                        "key_tied_to_content": True}
    tied = getattr(cfg, "tie_comparison_maps", False)      # S_K := S_Q: no raw_score_k
    if tied:
        layer_counts["score"] -= K*K
    if getattr(cfg, "learn_attention_temperature", False):  # routing_log_temperature (H,) per layer
        layer_counts["score"] += H
    noisy_key = not minimal_agent and getattr(cfg, "relational_routing", "gaussian_kl") == "noisy_key"
    key_noise_fields: Dict[str, object] = {}
    if noisy_key:                                           # routing_key_log_noise (H,) per layer
        layer_counts["score"] += H
        key_noise_fields = {"relational_key_noise_init": float(cfg.relational_key_noise)}
    counts = {role: count * cfg.n_layers for role, count in layer_counts.items()}
    return {"relational_operators": {
        "attention_mode": cfg.attention_mode,
        "routing": ((f"minimal_agent_{cfg.minimal_agent_inference}_source_posterior" if minimal_agent
                     else cfg.relational_routing)),
        "n_layers": cfg.n_layers, "layer_parameter_sharing": "independent",
        "per_layer_parameter_counts": layer_counts,
        "shape_scope": "per_layer",
        "inference_semantics": ("relational_full_link_frozen_filtering" if full_link
                                else f"relational_minimal_agent_{cfg.minimal_agent_inference}" if minimal_agent
                                else "relational_predictive_content"),
        "readout_mode": cfg.relational_readout_mode,
        "output_scope": "full",
        "mean_update": (("receiver_closed_form_natural_parameter_update"
                         if cfg.relational_receiver_update == "closed_form" else
                         "receiver_profiled_closed_form_natural_parameter_update"
                         if cfg.relational_receiver_update == "closed_form_profiled" else "receiver_fisher_step")
                        if full_link else
                        "minimal_agent_posterior_mean" if minimal_agent else
                        "input_plus_predictive_mean" if cfg.attention_residual else
                        "replace_with_predictive_mean"),
        "mode": cfg.relational_mode,
        "input_scope": "full",
        "input_dimension": K,
        "head_output_dimension": K,
        "score_parameterization": "free_full_row_rank",
        "score_initialization": ("query_equal_to_sending_block_projector" if minimal_agent
                                 else "tied_orthonormal_rows_key_equals_query" if tied
                                 else "independent_orthonormal_rows"),
        "score_role_parameter_counts": ({"q": cfg.n_layers*K*K, "k": 0, "channel_log_noise": cfg.n_layers*H}
                                        if minimal_agent else {"q": cfg.n_layers*K*K, "k": 0 if tied else cfg.n_layers*K*K}),
        "score_rank_per_head": rank, "content_rank_per_head": rank,
        "identity_content": False,
        "content_mode": "full_width",
        "content_link_chart": None if minimal_agent else cfg.content_link_chart,
        **edge_fields,
        **agent_fields,
        **key_noise_fields,
        "score_shape": score_shape,
        "content_shapes": content_shapes,
        "parameter_counts": counts, "total_parameters": sum(counts.values()),
        "learning_rates": {
            # Unbound, so a duck-typed config (tests pass namespaces) resolves through the same rule.
            role: VFE3Config.resolved_hyperparameter(cfg, f"m_relational_{role}_lr") for role in counts},
        "score_covariance_contract": ("exact_full_reduced_covariance_from_local_gaussian_plus_channel_noise"
                                      if minimal_agent else "exact_full_reduced_covariance_from_local_gaussian"),
        "receiver_noise": cfg.relational_receiver_noise if full_link else None,
        "receiver_measurement": cfg.relational_receiver_measurement if full_link else None,
        "receiver_update": cfg.relational_receiver_update if full_link else None,
        "receiver_damping": (cfg.relational_receiver_damping
                             if full_link and cfg.relational_receiver_update != "euler" else None),
        "covariance_update": (("passthrough" if cfg.skip_belief_sigma_update else
                               "fisher_step" if cfg.relational_receiver_update == "euler" else "closed_form")
                              if full_link else getattr(cfg, "predictive_covariance", "passthrough")),
        "content_covariance_contract": ("value_space_channel_likelihood_plus_channel_noise" if full_link else
                                        (("minimal_agent_mean_field_label_spread_corrected_plus_prior_floor"
                                          if getattr(cfg, "predictive_covariance", "passthrough") == "spread_corrected"
                                          else "minimal_agent_posterior_moment_match_plus_covariant_floor")
                                         if minimal_agent
                                         # The head-block family stores the head blocks of U_i(C^att + f I)U_i^T: the
                                         # exact M-projection onto block-diagonal Gaussians, covariant under the
                                         # per-head block group (audit 2026-10-09 round 2, #10).
                                         else "head_block_moment_projection_of_predictive_kernel_plus_covariant_floor"
                                         if cfg.family == "gaussian_block"
                                         # The diagonal family stores diag of U_i(C^att + f I)U_i^T in local
                                         # coordinates: a moment projection, neither exact nor covariant.
                                         else "diagonal_local_moment_projection_of_predictive_kernel_plus_floor"
                                         if cfg.family != "gaussian_full"
                                         else "exact_predictive_kernel_moment_match_plus_covariant_floor")
                                        if getattr(cfg, "predictive_covariance", "passthrough") in READOUT_COVARIANCES
                                        else "frozen_receiver_bookkeeping_not_predictive_covariance"),
        "self_coupling_coordinate_contract": (
            "ordered_conditional_kl_stored_coordinate_order"
            if cfg.family == "gaussian_full" and cfg.lambda_alpha_mode == "state_dependent_per_coord"
            else "family_default"),
        "factorization_policy": "zero_jitter_no_covariance_repairs",
        # The minimal agent passes no source tuples, so no source-root recovery exists. A float32 layer call with a
        # failed factorization (or, under a certified projected_factorization, an ADF query covariance that fails the
        # certificate) is recomputed whole in float64; a float64 failure raises. Its mean field also keeps the
        # Gram-route certificate of projected_gaussian_kl (a whole-tensor float64 refactor of the KL factors).
        "factorization_fallback": (("fp32_failure_or_certificate_layer_fp64_recompute"
                                    + ("" if cfg.minimal_agent_inference == "exact"
                                       else "_kl_certificate_fp64_refactor")) if minimal_agent
                                   else "failed_entries_source_square_root_qr_fp64"),
        "temperature_convention": ((f"unit_temperature_{'exact' if cfg.minimal_agent_inference == 'exact' else 'mean_field'}"
                                    "_posterior_no_clamp") if minimal_agent
                                   else "unit_temperature_learned_key_noise" if noisy_key
                                   else "kappa_beta_times_sqrt_ambient_head_dim"),
        # Exact minimal agent: head h routes on the receiver moments left by heads 1..h-1 (ADF), sources pre-update.
        "routing_state": ("episode_initial" if full_link
                          else "pre_update_sources_adf_receiver"
                          if minimal_agent and cfg.minimal_agent_inference == "exact" else "pre_update"),
        "state_destination": (
            "receiver_belief_mean_and_covariance"
            if (full_link and not getattr(cfg, "skip_belief_sigma_update", False))
            or (not full_link and getattr(cfg, "predictive_covariance", "passthrough") in READOUT_COVARIANCES)
            else "receiver_belief_mean"),
        "source_beliefs": "live_episode_initial" if full_link else "live_pre_update",
        "outer_operator_and_frame_graph": "live",
        "canonical_shared_objective": False,
        "gradient_clipping": "shares_existing_mu_role_and_global_budgets",
    }}


def _conditional_artifact_fields(cfg: VFE3Config) -> Dict[str, object]:
    # The query/key frame pair is gone, so the conditional block carries no frame-role record.
    return _relational_artifact_fields(cfg)


def _count_finalize_failure(
    artifacts: "RunArtifacts",
    name:      str,
) -> None:
    r"""Count one failed finalize-time probe in ``artifacts.diagnostics_failures``, which summary.json publishes
    beside the training-time failures (audit 2026-10-12 #15)."""
    failures = getattr(artifacts, "diagnostics_failures", None)
    if failures is None:
        failures = artifacts.diagnostics_failures = {}
    failures[name] = failures.get(name, 0) + 1


def _non_vfe_skipped_analyses(model: torch.nn.Module) -> Dict[str, str]:
    r"""The VFE-only finalize analyses a model of another family skips, each with its reason."""
    family = model_family_of(model)
    return {
        "zero_e_steps_counterfactual": f"model_family={family!r} has no E-step whose depth could be set to zero",
        "estep_final_f": f"model_family={family!r} has no E-step free energy",
        "mechanism_probes": (f"model_family={family!r}: the E-step depth, character, beta-channel and "
                             "context-sensitivity probes read the VFE belief stack"),
        "phi_numerics": f"model_family={family!r} has no gauge frames",
        "pure_path_report": f"model_family={family!r}: the pure-path toggles are VFE3Config fields",
        "research_artifacts": f"model_family={family!r}: the research records read the prior bank",
        "figures": (f"model_family={family!r}: the figure worker rebuilds a VFE3Config and VFEModel; "
                    "curves come from metrics.csv"),
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


_DECLARATIVE_DRIVER_NAME = re.compile(r"^(?:config|fully_gauge_covariant|[A-Z][A-Z0-9_]*)$")


def _is_declarative_driver_statement(statement: ast.stmt) -> bool:
    r"""Whether a top-level driver statement only assigns click-to-run settings.

    The launchers keep their live settings in a top-level ``config`` dictionary, the lowercase launcher
    toggle ``fully_gauge_covariant`` (``train_vfe3.py``, ``ablation.py``), and public UPPER_CASE names
    (``CONFIG``, ``SWEEPS``, ``DATA_SEED``, ``RUN_DIR`` ...), including ``config[...] = ...`` item
    assignments. Those values are bound by the run's own config fingerprint and contracts, not by the
    executable-code identity, so editing them (for example adding ``resume_from``) is not code drift. The
    toggle acts through the gauge overrides it writes into the config, which the fingerprint binds, and a
    refusal gate that computes nothing; ``ablation.py``'s runner hash lists it as declarative too (audit
    2026-10-16 F78).
    """
    if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return False
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]

    def _root(target: ast.expr) -> Optional[str]:
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    roots = [_root(target) for target in targets]
    return bool(roots) and all(root is not None and _DECLARATIVE_DRIVER_NAME.match(root) for root in roots)


def _driver_logic_bytes(source: bytes) -> bytes:
    r"""Executable logic of a root driver: its AST without top-level declarative settings.

    A driver that does not parse (a half-saved edit) falls back to its newline-normalized bytes, so
    the identity still changes with it.
    """
    normalized = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    try:
        tree = ast.parse(normalized.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return normalized
    tree.body = [statement for statement in tree.body if not _is_declarative_driver_statement(statement)]
    return ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")


def _package_code_identity() -> str:
    r"""Return the current content identity for the package and supported root execution drivers.

    A run captures this digest once when its artifact owner is constructed. Generated files, tests,
    documentation, Git metadata, and output directories are excluded, so artifact publication cannot
    change the identity. Recomputing at each run boundary is required for long-lived Spyder kernels:
    editing a source file must not reuse the previous run's identity. Package sources are hashed
    byte for byte (newline-normalized); root drivers contribute only their executable logic
    (:func:`_driver_logic_bytes`), so a resume or sweep edit of a launcher's settings does not discard
    the checkpoint's best-model selection (audit 2026-10-08 P2-2). The drivers that train runs,
    ``train_baseline.py`` and ``paper_arms.py`` included, are all listed (gap-data-lifecycle#8).
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
            "train_baseline.py",
            "paper_arms.py",
        )
        if (repository / name).is_file()
    }
    sources = sorted(package_sources | root_drivers)
    if not sources:
        raise RuntimeError("training code identity found no executable Python sources")
    for path in sources:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"training code identity cannot read {path.relative_to(root.parent)!s}") from exc
        content = (_driver_logic_bytes(content) if path in root_drivers
                   else content.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
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


# A float field is annotated float, Optional[float], or the string annotation 'float | List[float]'.
_FLOAT_FIELD_TYPES: Tuple[object, ...] = (float, Optional[float], "float | List[float]")
_FLOAT_CONFIG_FIELDS: Optional[frozenset] = None


def _float_config_fields() -> frozenset:
    r"""Names of the ``VFE3Config`` fields whose scalar value is a float (cached)."""
    global _FLOAT_CONFIG_FIELDS
    if _FLOAT_CONFIG_FIELDS is None:
        from dataclasses import fields as dataclass_fields
        _FLOAT_CONFIG_FIELDS = frozenset(
            field.name for field in dataclass_fields(VFE3Config)
            if field.type in _FLOAT_FIELD_TYPES)
    return _FLOAT_CONFIG_FIELDS


def semantic_config_fingerprint(
    config: Mapping[str, Any],
) -> str:
    """Return the stable SHA-256 fingerprint of a normalized semantic config mapping.

    A top-level key that names a float-typed ``VFE3Config`` field hashes an integer value as the
    equal float, so ``rope_fraction=1`` and ``rope_fraction=1.0`` fingerprint alike (audit 2026-10-08
    P2-4). Records that spelled such a field as a float keep their fingerprint.
    """
    float_fields = _float_config_fields()
    normalized = json.dumps(
        {key: (float(value) if key in float_fields and isinstance(value, int) and not isinstance(value, bool)
               else value)
         for key, value in dict(config).items()},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _migrated_config_fingerprint(
    config: Mapping[str, Any],
) -> str:
    """Fingerprint a saved config in current serialized form, or raw when it no longer loads."""
    try:
        return semantic_config_fingerprint(_serialize_run_config(
            _config_from_run_serialized(config, source="resume provenance fingerprint")))
    except (TypeError, ValueError):
        return semantic_config_fingerprint(config)


def _phi_chart_norm_route(
    model: torch.nn.Module,
    cfg:   object,
) -> Optional[str]:
    """Persist the active phi factor-bound norm route and pullback default radius."""
    configured_radius = getattr(cfg, "phi_mstep_max_matrix_norm", None)
    pullback_group = getattr(cfg, "m_phi_update_mode", "adamw") in ("pullback_group", "fisher_group")
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

    with ``tau = decode_tau`` and ``s = decode_log_scale``. The scale is frozen in this build, but a
    historical run with the retired ``decode_temperature_learned=True`` trained it, so the persisted
    ``decode_tau`` is only the initialization of the temperature its decode
    ``logits = -KL / tau_eff`` actually used."""
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


def _losses_start_step(
    artifacts: 'RunArtifacts',
    losses:    Optional[List[float]],
    logger:    logging.Logger,
) -> int:
    r"""Absolute optimizer steps completed before ``losses[0]``.

    ``train`` appends one loss per step of ``range(start_step, n_steps)`` and records
    ``final_step = n_steps``, so a resumed run's continuation list satisfies
    ``start_step = final_step - len(losses)`` and ``losses[k]`` belongs to the 1-based step
    ``start_step + k + 1`` (the ``step`` column of ``metrics.csv``). A caller without a recorded
    ``final_step`` (or an inconsistent one) falls back to 0, the from-scratch axis.
    """
    if not losses:
        return 0
    final_step = getattr(artifacts, "final_step", None)
    if type(final_step) is not int:
        return 0
    start_step = final_step - len(losses)
    if start_step < 0:
        logger.warning(
            "final_step %d is shorter than the %d-step loss history; loss_curve.png uses steps 1..n",
            final_step, len(losses))
        return 0
    return start_step


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
                    config = _serialize_run_config(artifacts.cfg)
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
                    "losses_start_step":   _losses_start_step(artifacts, losses, logger),
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
        if channel == "beta":
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
    elif not is_vfe_config(config) and is_dataclass(config) and not isinstance(config, type):
        normalized = asdict(config)                   # a live config of another model family
    elif isinstance(config, Mapping):
        try:
            normalized = asdict(_config_from_run_serialized(
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
        "checkpoint_keep_last",
        "checkpoint_at_end",
        "block_variance_at_log",
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
        effective_saved_config = asdict(_config_from_run_serialized(
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


def _row_step_valid(
    value:             object,
    parameter:         torch.Tensor,
    device:            torch.device,
    maximum_slot_step: int,
) -> bool:
    r"""Whether ``value`` is a row-lazy table's per-row step count :math:`k_v` (``m_phi_sparse_rows``): shape ``(V,)``,
    the parameter's dtype, on ``device``, finite non-negative integers no larger than the optimizer step clock."""
    return (isinstance(value, torch.Tensor)
            and value.device == device
            and value.dtype == parameter.dtype
            and value.layout == torch.strided
            and value.shape == parameter.shape[:1]
            and bool(torch.isfinite(value).all())
            and bool((value >= 0).all())
            and bool((value == value.round()).all())
            and bool((value <= maximum_slot_step).all()))


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
            # m_phi_sparse_rows: the row-lazy table also carries its per-row counts (absent in a dense checkpoint,
            # which the row-lazy optimizer resumes at k_v = step).
            row_lazy = bool(group.get("sparse_rows", False))
            if keys != expected_keys and not (row_lazy and keys == expected_keys | {"row_step"}):
                raise RuntimeError("checkpoint optimizer_state has an invalid AdamW slot schema")
            if "row_step" in keys and not _row_step_valid(state["row_step"], parameter, torch.device("cpu"),
                                                          maximum_slot_step):
                raise RuntimeError("checkpoint optimizer_state row_step is incompatible with its parameter")
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


_RETIRED_DECODE_TEMPERATURE = "prior_bank.decode_log_scale"


def _drop_retired_decode_temperature(
    manifest:        object,
    optimizer_state: object,
    optimizer:       torch.optim.Optimizer,
    model:           torch.nn.Module,
) -> Tuple[object, object]:
    r"""Remove a never-stepped retired learned decode temperature from a saved optimizer.

    Runs saved under the retired ``decode_temperature_learned=True`` grouped
    ``prior_bank.decode_log_scale`` in the optimizer; this build freezes it. When its saved AdamW slot
    is empty the temperature was never stepped (for example a linear decode, whose readout discards
    it), so dropping the entry, and any group it leaves empty, reproduces the saved optimizer exactly.
    A populated slot means the run trained the temperature, and resuming would silently stop
    training it, so that resume is refused as retired. Malformed inputs are returned unchanged for the
    strict validators that follow.
    """
    if (not isinstance(manifest, Mapping) or not isinstance(optimizer_state, Mapping)
            or not isinstance(manifest.get("parameter_groups"), list)
            or not isinstance(optimizer_state.get("param_groups"), list)
            or not isinstance(optimizer_state.get("state"), Mapping)
            or len(manifest["parameter_groups"]) != len(optimizer_state["param_groups"])):
        return manifest, optimizer_state
    live_names = {name for group in _optimizer_parameter_name_shape_groups(model, optimizer)
                  for name in (entry["name"] for entry in group)}
    if _RETIRED_DECODE_TEMPERATURE in live_names:
        return manifest, optimizer_state
    retired_ids = [entry.get("parameter_id")
                   for group in manifest["parameter_groups"] if isinstance(group, list)
                   for entry in group
                   if isinstance(entry, Mapping) and entry.get("name") == _RETIRED_DECODE_TEMPERATURE]
    if not retired_ids:
        return manifest, optimizer_state
    slots = optimizer_state["state"]
    if any(slots.get(parameter_id) for parameter_id in retired_ids):
        raise RuntimeError(
            "checkpoint optimizer trained the retired learned decode temperature "
            "(decode_temperature_learned=True): prior_bank.decode_log_scale is frozen in this build, "
            "so resuming would silently stop training it. Evaluate the checkpoint (its temperature "
            "is replayed exactly) or start a fresh optimizer from its weights")
    retired = set(retired_ids)
    kept_manifest: List[object] = []
    kept_groups: List[object] = []
    for manifest_group, saved_group in zip(manifest["parameter_groups"], optimizer_state["param_groups"]):
        if not isinstance(manifest_group, list) or not isinstance(saved_group, Mapping):
            return manifest, optimizer_state
        entries = [entry for entry in manifest_group
                   if not (isinstance(entry, Mapping) and entry.get("parameter_id") in retired)]
        params = saved_group.get("params")
        if isinstance(params, list):
            params = [parameter_id for parameter_id in params if parameter_id not in retired]
        if not entries and manifest_group:
            continue                                   # the group held only the temperature
        kept_manifest.append(entries)
        kept_groups.append({**saved_group, "params": params})
    manifest = {**manifest, "parameter_groups": kept_manifest}
    optimizer_state = {**optimizer_state, "param_groups": kept_groups,
                       "state": {key: value for key, value in slots.items() if key not in retired}}
    return manifest, optimizer_state


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
        row_lazy = bool(group.get("sparse_rows", False))
        if keys != expected_keys and not (row_lazy and keys == expected_keys | {"row_step"}):
            raise RuntimeError("realized optimizer has an invalid AdamW slot schema")
        if "row_step" in keys and not _row_step_valid(state["row_step"], parameter, parameter.device,
                                                      maximum_slot_step):
            raise RuntimeError("realized optimizer row_step is incompatible with its parameter")
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
    if torch.cuda.device_count() > 0 and cuda is None:
        raise RuntimeError("checkpoint rng_state must contain every active CUDA RNG state")
    if cuda is not None:
        if not isinstance(cuda, (list, tuple)):
            raise RuntimeError("checkpoint rng_state cuda states must be a sequence or null")
        if torch.cuda.device_count() > 0 and len(cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint rng_state CUDA state count does not match the active device count")
        for index, state in enumerate(cuda):
            if (not isinstance(state, torch.Tensor) or state.device.type != "cpu"
                    or state.dtype != torch.uint8):
                raise RuntimeError("checkpoint rng_state CUDA states must be CPU ByteTensors")
            if index < torch.cuda.device_count():
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


_CHECKPOINT_NUMERICS_KEYS: Tuple[str, ...] = (
    "frame_projection_count", "frame_projection_total", "frame_projection_since_step")


def _checkpoint_numerics_counters(
    value: object,
) -> Optional[Dict[str, int]]:
    r"""The checkpoint's cumulative numerics counts as ``{key: int >= 0}``, or None.

    The frame-projection counts are kept all or none; each run total of
    ``vfe3.train.RUN_TOTAL_COUNTER_KEYS`` is kept when it is a non-negative int, so a resumed run's
    summary and metrics report whole-run totals. Tolerant by design: these are diagnostics, so a
    missing or malformed record (an older checkpoint, a hand-edited file) yields None rather than
    failing a save or a resume.
    """
    from vfe3.train import RUN_TOTAL_COUNTER_KEYS

    if not isinstance(value, Mapping):
        return None
    frame = {key: value.get(key) for key in _CHECKPOINT_NUMERICS_KEYS}
    out: Dict[str, int] = (dict(frame) if all(type(count) is int and count >= 0 for count in frame.values())
                           else {})
    out.update({key: value[key] for key in RUN_TOTAL_COUNTER_KEYS
                if type(value.get(key)) is int and value[key] >= 0})
    return out or None


def _checkpoint_divergence_state(
    value: object,
) -> Optional[Dict[str, object]]:
    r"""The checkpoint's divergence-monitor record, validated by ``vfe3.train._checkpoint_divergence_state``
    (None when missing or malformed: a diagnostic, so it never fails a save or a resume)."""
    from vfe3.train import _checkpoint_divergence_state as validated

    return validated(value)


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
        # In-memory copy of the CSV rows (for figures); cells are normalized floats and ints, and text in the str
        # columns such as row_kind (vfe3.metrics_schema.normalize_metric_cell).
        self.history: List[Dict[str, Union[float, int, str]]] = []
        self._fieldnames: Optional[List[str]] = None

        config_record = {
            "config":    _serialize_run_config(cfg),
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

        # Training-time diagnostic handoff (audit 2026-09-28), filled by train() and read by
        # finalize_run: fixed training-sample CE by evaluated step, per-name diagnostic failure
        # counts, and the optimizer-state footprint.
        self.dataset = dataset
        self.train_sample_by_step: Dict[int, float] = {}
        self.diagnostics_failures: Dict[str, int] = {}
        self.optimizer_state_bytes: Optional[Dict[str, int]] = None
        # S1 panel D counters (2026-09-28): summary.json "numerics_counters" (set by train()), the
        # cumulative frame-projection counts a checkpoint carries (numerics_counters_state), and
        # those a resumed checkpoint carried in (resumed_numerics_counters, set by load_checkpoint).
        self.numerics_counters: Optional[Dict[str, object]] = None
        self.numerics_counters_state: Optional[Dict[str, int]] = None
        self.resumed_numerics_counters: Optional[Dict[str, int]] = None
        # Cumulative wall clock a checkpoint carries (set by train()) and the value a resumed
        # checkpoint carried in (set by load_checkpoint), so metrics.csv wall_clock_s continues.
        self.wall_clock_state: Optional[float] = None
        self.resumed_wall_clock_s: Optional[float] = None
        # Divergence-monitor host state a checkpoint carries (set by train()) and the validated record a
        # resumed checkpoint carried in (set by load_checkpoint), so spike detection continues exactly.
        self.divergence_state: Optional[Dict[str, object]] = None
        self.resumed_divergence_state: Optional[Dict[str, object]] = None
        # The run a resume continues (set by load_checkpoint; None from scratch). This directory's
        # metrics.csv, diagnostics and figures start after saved_step, while best_step is run-wide, so
        # summary.json names where the earlier history lives (audit 2026-10-16 F123).
        self.resume_source: Optional[Dict[str, object]] = None
        # Source bytes at run start: vfe3/**/*.py, root *.py drivers and pyproject.toml, zipped with
        # their hash, the Git HEAD and the dirty-file list. Best-effort: a failure is recorded in
        # the returned mapping (and later in provenance.json), never raised.
        from vfe3.run_archive import write_source_archive
        self.source_archive: Dict[str, object] = write_source_archive(self.run_dir)

    def append_diagnostic_rows(
        self,
        name:       str,
        fieldnames: Sequence[str],
        rows:       Iterable[Mapping[str, object]],
    ) -> Path:
        r"""Append long-format rows to ``run_dir/diagnostics/<name>`` (header written once).

        NaN cells are written blank, as in ``metrics.csv``. A header that differs from
        ``fieldnames`` raises rather than mixing two schemas in one file, except that a file whose
        header is ``fieldnames`` with trailing columns missing (one written before those columns were
        added, as ``block_variance.csv`` before its upper quantiles) is first rewritten atomically to
        the full header with the missing cells blank, so a resumed run keeps one schema per file.
        """
        if not name or "/" in name or "\\" in name or Path(name).name != name:
            raise ValueError(f"diagnostic file name must be a bare filename, got {name!r}")
        directory = self.run_dir / "diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        fields = list(fieldnames)
        if path.is_file() and path.stat().st_size > 0:
            with path.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header is not None and header != fields and header == fields[:len(header)]:
                with path.open("r", newline="", encoding="utf-8") as handle:
                    existing = list(csv.DictReader(handle))
                with _unique_sibling_temp(path) as tmp:
                    with open(tmp, "w", newline="", encoding="utf-8") as handle:
                        migrated = csv.DictWriter(handle, fieldnames=fields, restval="")
                        migrated.writeheader()
                        migrated.writerows(existing)
                    _atomic_replace(path, tmp)
                header = fields
            if header != fields:
                raise ValueError(f"{path} has header {header}, expected {fields}")
            write_header = False
        else:
            write_header = True
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: ("" if isinstance(value, float) and math.isnan(value) else value)
                    for key, value in row.items()
                })
        return path

    def save_map_snapshot(
        self,
        step:  int,
        state: Mapping[str, torch.Tensor],
    ) -> Path:
        r"""Write ``run_dir/map_snapshots/step_<N:07d>.pt = {"step": N, "state": state}`` atomically."""
        directory = self.run_dir / "map_snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"step_{int(step):07d}.pt"
        with _unique_sibling_temp(final) as tmp:
            torch.save({"step": int(step), "state": dict(state)}, tmp)
            _atomic_replace(final, tmp)
        return final

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

    def log_metrics(self, row: Mapping[str, object]) -> None:
        r"""Atomically publish ``metrics.csv`` with one additional rectangular row.

        The column set is fixed by the first row; later rows must share those keys so the CSV
        stays rectangular (the training loop emits a homogeneous row each periodic eval).

        Every cell first passes :func:`vfe3.metrics_schema.normalize_metric_cell` (audit 2026-10-17
        item 31): one-element tensors and NumPy scalars become Python numbers, a bool becomes 0 or 1,
        ``None`` becomes NaN, and a string in a column the schema declares numeric, a multi-element
        array or any other object raises ``TypeError`` before anything is written. The file and the
        in-memory history hold the same normalized values.

        NaN cells are written to the file as BLANK (empty string), so an eval-cadence column
        (val_*, generalization_gap, the held-out probes) -- NaN on the denser log-interval rows
        between evals -- shows an empty cell rather than a repeated value or a literal "nan".
        The IN-MEMORY ``self.history`` keeps the raw NaN float so
        the figure pass (which filters on ``math.isfinite``) is unaffected."""
        row = normalize_metric_row(row)
        fieldnames = list(row) if self._fieldnames is None else list(self._fieldnames)
        if set(row) != set(fieldnames):
            missing = sorted(set(fieldnames) - set(row))
            extra = sorted(set(row) - set(fieldnames))
            raise ValueError(
                f"metrics row field set changed (missing={missing}, extra={extra})"
            )
        candidate_history = [*self.history, dict(row)]
        try:
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
        except PermissionError as exc:
            # A reader holding metrics.csv open on Windows blocks the replace past its retries (audit
            # 2026-10-08 P1-6). Every publish rewrites the whole history, so the row is kept in memory
            # and reaches the file at the next publish that is not blocked; the run continues.
            logging.getLogger(__name__).warning(
                "metrics.csv publish blocked (%s); the row is kept and published with the next one", exc)
        self._fieldnames = fieldnames
        self.history.append(dict(row))

    def maybe_save_best(
        self,
        step:    int,
        model:   torch.nn.Module,
        val_ppl: float,

        *,
        background: bool = False,
    ) -> bool:
        r"""Record a new minimum and, when enabled, save its config-bound weights.

        Atomic (same-dir tmp + ``os.replace``): a crash or Windows lock mid-save can never leave a
        corrupt/unreadable ``best_model.pt`` where a good one stood.

        ``background`` (audit 2026-10-17 item 68): the weights are copied to host memory here, so the bundle is the
        state at this call, and ``torch.save`` plus the atomic replace run on a worker thread while training
        continues. ``best_val_ppl`` and ``best_step`` advance only when the file has been replaced, as on the
        synchronous path; a failed save is logged and counted under ``diagnostics_failures["best_model_save"]``.
        Every reader of the record or the file waits first (:meth:`wait_background`; this method,
        :meth:`save_checkpoint` and the finalizers do). Returns True when a save was published or scheduled."""
        self.wait_best_save()
        if val_ppl < self.best_val_ppl:
            if not self.save_model_artifacts:
                self.best_val_ppl = float(val_ppl)
                self.best_step = int(step)
                return True
            config = _serialize_run_config(self.cfg)
            state = (model.state_dict() if not background else
                     {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()})
            bundle = {
                "model_state":        state,
                "config":             config,
                "config_fingerprint": semantic_config_fingerprint(config),
                "code_identity_sha256": self.code_identity_sha256,
                "selection_data_identity": self.selection_data_identity,
            }
            if background:
                worker = threading.Thread(target=self._publish_best, args=(bundle, int(step), float(val_ppl), True),
                                          name=f"best_model_save_step_{int(step)}", daemon=True)
                self._best_save_thread = worker
                worker.start()
                return True
            return self._publish_best(bundle, int(step), float(val_ppl), False)
        return False

    def _publish_best(
        self,
        bundle:     Dict[str, object],
        step:       int,
        val_ppl:    float,
        background: bool,
    ) -> bool:
        r"""Write ``bundle`` to ``best_model.pt`` (same-directory temporary and atomic replace) and advance the record."""
        try:
            with _unique_sibling_temp(self.best_path) as tmp:
                torch.save(bundle, tmp)
                _atomic_replace(self.best_path, tmp)
        except PermissionError as exc:
            # A reader holding best_model.pt open on Windows (audit 2026-10-08 P1-6): the record is
            # not advanced, so best_model.pt and best_val_ppl stay consistent and the next eval
            # that beats the published best retries the save.
            logging.getLogger(__name__).warning(
                "best_model.pt publish at step %d blocked (%s); the previous best is kept", step, exc)
            return False
        except Exception as exc:
            if not background:
                raise
            logging.getLogger(__name__).warning(
                "(best_model.pt save at step %d failed: %s); training continues, the next improving eval retries",
                step, exc)
            failures = getattr(self, "diagnostics_failures", None)
            if isinstance(failures, dict):
                failures["best_model_save"] = failures.get("best_model_save", 0) + 1
            return False
        self.best_val_ppl = float(val_ppl)
        self.best_step = int(step)
        return True

    def wait_best_save(self) -> None:
        r"""Block until a background ``best_model.pt`` save (:meth:`maybe_save_best`) has finished."""
        worker = getattr(self, "_best_save_thread", None)
        if worker is not None:
            worker.join()
            self._best_save_thread = None

    def submit_figure_work(
        self,
        label: str,
        work:  Callable[[], object],
    ) -> None:
        r"""Run ``work`` (an isolated figure subprocess launch) on a worker thread so training does not wait for it.

        The child process keeps its isolation (:func:`_run_attention_maps_isolated`); the thread only waits for it.
        Jobs run one at a time in submission order, so at most one figure child runs beside training (audit
        2026-10-17 item 68). :meth:`wait_background` reaps them. A failure is logged, never raised.
        """
        queue = getattr(self, "_figure_jobs", None)
        if queue is None:
            queue = self._figure_jobs = []
        previous = queue[-1] if queue else None

        def run() -> None:
            if previous is not None:
                previous.join()
            try:
                work()
            except Exception as exc:                              # noqa: BLE001 - figures are best-effort
                logging.getLogger(__name__).warning("background figure job %s failed (%s)", label, exc)

        worker = threading.Thread(target=run, name=f"figure_{label}", daemon=True)
        queue.append(worker)
        worker.start()

    def wait_background(self) -> None:
        r"""Reap every background job of this run: the ``best_model.pt`` save and the figure workers."""
        self.wait_best_save()
        for worker in list(getattr(self, "_figure_jobs", None) or ()):
            worker.join()
        self._figure_jobs = []

    def prune_checkpoints(
        self,
        keep_last: Optional[int],
    ) -> List[Path]:
        r"""Delete all but the ``keep_last`` highest-step ``checkpoints/step_<N>.pt`` of this run directory.

        ``None`` keeps every checkpoint (``checkpoint_keep_last``; audit 2026-10-17 item 48). Only files named
        ``step_<N>.pt`` in this run's own ``checkpoints/`` are considered. Returns the removed paths.
        """
        if keep_last is None or not self.ckpt_dir.is_dir():
            return []
        found: List[Tuple[int, Path]] = []
        for path in self.ckpt_dir.glob("step_*.pt"):
            stem = path.stem[len("step_"):]
            if stem.isdigit():
                found.append((int(stem), path))
        found.sort()
        removed: List[Path] = []
        for _, path in found[:max(0, len(found) - int(keep_last))]:
            try:
                path.unlink()
                removed.append(path)
            except OSError as exc:
                logging.getLogger(__name__).warning("could not prune checkpoint %s (%s)", path, exc)
        return removed

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
        self.wait_best_save()                        # the embedded best-model bundle reads best_model.pt
        step = _require_nonnegative_int(step, "step")
        path = self.ckpt_dir / f"step_{step}.pt"
        rng_state = {
            "cpu":  torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.device_count() > 0 else None,
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
            if "window_offset" in data_state:            # per-epoch window origins only
                saved_data_state["window_offset"] = _require_nonnegative_int(
                    data_state["window_offset"], "window_offset")
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
                "config":          _serialize_run_config(cfg),
                "code_identity_sha256": self.code_identity_sha256,
                "git_identity":    _git_code_identity(),
                "selection_data_identity": self.selection_data_identity,
                "scaler_state":    (scaler.state_dict()
                                    if scaler is not None and scaler.is_enabled() else None),
                "best_val_ppl":    saved_best_val_ppl,
                "best_step":       saved_best_step,
                "best_model_bundle": best_model_bundle,
                "data_state":      saved_data_state,
                # Cumulative S1 panel D frame-projection counts (flat str -> int, or None); a
                # diagnostic, never read by the resume contract itself.
                "numerics_counters": _checkpoint_numerics_counters(
                    getattr(self, "numerics_counters_state", None)),
                "wall_clock_s":    getattr(self, "wall_clock_state", None),
                # Divergence-monitor EMAs, warmups, cooldown and open interval with the skip counters
                # (vfe3.train._divergence_checkpoint_record, or None); a diagnostic like the counters above.
                "divergence_monitor": _checkpoint_divergence_state(getattr(self, "divergence_state", None)),
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
        parsed = _config_from_run_serialized(saved_config, source="checkpoint resume config")
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
            _config_from_run_serialized(raw_checkpoint_config, source="checkpoint exact resume config")
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
            checkpoint_manifest, checkpoint_optimizer_state = _drop_retired_decode_temperature(
                ckpt.get("optimizer_parameter_manifest"), checkpoint_optimizer_state,
                optimizer, model,
            )
            _validate_optimizer_parameter_manifest(
                checkpoint_manifest,
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
    # A bounded run's checkpoint is a fixed point of the frame condition bound, so this is a bitwise no-op
    # there; it bounds the frames of any other state before step 0 or evaluation (audit round 2, M4).
    bound_frame_condition_if_configured_(model)
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
        if rng.get("cuda") is not None and torch.cuda.device_count() > 0:
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
            if saved_data_state.get("window_offset") is not None:
                data_state["window_offset"] = _require_nonnegative_int(
                    saved_data_state["window_offset"], "window_offset")
    if artifacts is not None:
        # Cumulative S1 panel D frame-projection counts (None for an older checkpoint); train()
        # continues counting from them so a resumed run's columns stay cumulative since step 0.
        artifacts.resumed_numerics_counters = _checkpoint_numerics_counters(
            ckpt.get("numerics_counters"))
        saved_wall_clock = ckpt.get("wall_clock_s")
        artifacts.resumed_wall_clock_s = (
            float(saved_wall_clock) if isinstance(saved_wall_clock, (int, float))
            and not isinstance(saved_wall_clock, bool) and math.isfinite(saved_wall_clock) else None)
        # Divergence-monitor state (None for an older checkpoint or a malformed record); train() continues it.
        artifacts.resumed_divergence_state = _checkpoint_divergence_state(ckpt.get("divergence_monitor"))
        resolved_checkpoint = checkpoint_path.resolve()
        artifacts.resume_source = {
            # <run_dir>/checkpoints/step_<N>.pt; None for a checkpoint kept outside a run directory.
            "source_run_dir":  (str(resolved_checkpoint.parent.parent)
                                if resolved_checkpoint.parent.name == "checkpoints" else None),
            "checkpoint_path": str(resolved_checkpoint),
            "saved_step":      saved_step,
        }
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
            # Fingerprint the saved mapping after loader migration, so a pre-migration checkpoint
            # (e.g. relational_mode/scope keys before PR #32) with no semantic drift matches.
            "saved_config_fingerprint":        (
                _migrated_config_fingerprint(saved_config)
                if isinstance(saved_config, Mapping) else None
            ),
            "current_config_fingerprint":      (
                semantic_config_fingerprint(_serialize_run_config(resume_cfg))
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
    # The probe ignores the system and global config files, but the checkout's line-ending policy
    # lives there (Git for Windows sets core.autocrlf=true system-wide). Without it every CRLF working
    # file reads as modified until an ordinary `git status` refreshes the index, so git_dirty and the
    # fingerprint would track index stat freshness instead of content (audit 2026-10-08 P2-28).
    for index, (key, value) in enumerate(_git_line_ending_config(git_executable)):
        env[f"GIT_CONFIG_KEY_{index}"] = key
        env[f"GIT_CONFIG_VALUE_{index}"] = value
        env["GIT_CONFIG_COUNT"] = str(index + 1)
    return env


_GIT_LINE_ENDING_KEYS: Tuple[str, ...] = ("core.autocrlf", "core.eol")


@functools.lru_cache(maxsize=None)
def _git_line_ending_config(
    git_executable: str,
) -> Tuple[Tuple[str, str], ...]:
    r"""Effective system/global ``core.autocrlf`` and ``core.eol`` (global wins), read once per process.

    Repository-local values still apply inside the probe, which disables only the system and global
    files. A failed read contributes nothing, which is the previous probe behavior.
    """
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    found: Dict[str, str] = {}
    for scope in ("--system", "--global"):
        for key in _GIT_LINE_ENDING_KEYS:
            try:
                completed = subprocess.run(
                    [git_executable, "config", scope, "--get", key],
                    env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                continue
            value = completed.stdout.decode("utf-8", "replace").strip()
            if completed.returncode == 0 and value:
                found[key] = value
    return tuple((key, found[key]) for key in _GIT_LINE_ENDING_KEYS if key in found)


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


def _execution_device_fields(model: torch.nn.Module) -> Dict[str, object]:
    r"""Identify the model's execution device without querying unused CUDA hardware."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    return {
        "device":       str(device),
        "device_type":  device.type,
        "device_index": device.index,
        "device_name":  torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
    }


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
        **_execution_device_fields(model),
        "data_seed":           (int(data_seed) if data_seed is not None else None),
        # The training-window origin ('per_epoch_random' for every TokenWindows run trained now), so
        # vfe3.experiment.saved_train_window_mode reads it without the repository history (audit 2026-10-16 F59).
        "train_window_mode":   getattr(getattr(train_loader, "dataset", None), "window_offset_mode", None),
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
    # Source bytes (audit 2026-09-28): the run-start archive record, the tree re-hashed now, and a
    # second archive (source_archive_end) only when the tree drifted during the run. Best-effort.
    try:
        from vfe3.run_archive import verify_source_archive
        start_archive = getattr(artifacts, "source_archive", None)
        start_archive = start_archive if isinstance(start_archive, dict) else None
        prov["source_archive"] = start_archive
        check = verify_source_archive(artifacts.run_dir, start_archive)
        prov["source_archive_drift"] = check.get("drift")
        prov["source_tree_sha256_end"] = check.get("tree_sha256_end")
        if check.get("error") is not None:
            prov["source_archive_check_error"] = check["error"]
        if check.get("drift"):
            prov["source_archive_end"] = check.get("source_archive_end")
            logger.warning("source tree changed during the run; archived the finalize-time tree as "
                           "source_archive_end.zip beside the run-start source_archive.zip")
    except Exception as exc:                                  # never block finalize on bookkeeping
        prov["source_archive_check_error"] = repr(exc)
    # Tokenizer manifest: library version, encoding, vocabulary size and a hash of the token table.
    try:
        from vfe3.run_archive import tokenizer_manifest
        prov["tokenizer_manifest"] = tokenizer_manifest(
            tokenizer_tag,
            dataset=(getattr(artifacts, "dataset", None) or None),
            cache_dir=getattr(getattr(train_loader, "dataset", None), "cache_dir", None),
        )
    except Exception as exc:
        prov["tokenizer_manifest"] = {"kind": tokenizer_tag, "status": "unavailable",
                                      "reason": repr(exc)}
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
    record:      Optional[Dict[str, Any]] = None,
) -> float:
    r"""Worst relative error between autograd-of-CE and a central finite difference on a few DECODE
    coordinates: ``output_proj_weight`` under linear decode, else the untied decode tables
    ``decode_mu_embed`` / ``decode_sigma_log_embed`` of a prior-bank decode, whose gradient reaches the
    loss only through the decode, so a healthy model reads ~1e-4 and a broken decode adjoint spikes far
    above it. A tied prior-bank decode has no decode-only trainable table (``decode_log_scale`` is frozen),
    so the check returns NaN there. The probed parameter, or the reason for a NaN, is written to ``record``
    (``fd_gradient_probe``, ``fd_gradient_nan_reason``; audit 2026-10-07, F86). Best-effort on one tiny
    batch; restores every coord."""
    if record is None:
        record = {}
    with _diagnostic_eval_mode(model):
        batch = next(iter(test_loader))
        tok, tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
        tok = tok[:2].to(device)
        tgt = tgt[:2].to(device)
        pb = model.prior_bank
        candidates = [(name, getattr(pb, name, None)) for name in
                      ("output_proj_weight", "decode_mu_embed", "decode_sigma_log_embed")]
        candidates = [(name, q) for name, q in candidates if q is not None and q.requires_grad]
        if not candidates:
            record["fd_gradient_nan_reason"] = ("no decode-only trainable parameter (tied prior-bank decode; "
                                                "decode_log_scale is frozen)")
            return float("nan")
        name, p = candidates[0]
        record["fd_gradient_probe"] = f"prior_bank.{name}"
        model.zero_grad(set_to_none=True)
        _, loss, _ = model(tok, tgt)
        loss.backward()
        if p.grad is None:                                          # decode param severed under this config
            model.zero_grad(set_to_none=True)
            record["fd_gradient_nan_reason"] = f"prior_bank.{name} received no gradient"
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
                    record["fd_gradient_nan_reason"] = "non-finite finite difference or autograd value"
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
        out["fd_gradient_worst_rel_error"] = _fd_gradient_check(model, test_loader, device, record=out)
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
    ("prior_bank", "sigma_chol_embed"),                 # prior_covariance_table='block_cholesky'
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


def _measured_forward_flops(
    model:       torch.nn.Module,
    cfg:         VFE3Config,
    tokens_seen: int,

    *,
    device:      Optional[torch.device] = None,
) -> Dict[str, object]:
    r"""Measured forward FLOPs per token of the model's belief pipeline and decode (audit 2026-10-20 item 48).

    The analytic proxy of :func:`_cost_model_fields` does not model the projected routing and content route, so
    ``est_flops_analytic`` is None there. This counts instead: one sequence of ``max_seq_len`` seeded tokens runs
    through ``forward_beliefs`` under :class:`torch.utils.flop_counter.FlopCounterMode`, once with the logits and once
    without, in eval mode, without gradients and with the RNG streams restored. The diagonal-moments kernel is
    switched to its torch route for the count, since the counter does not see Triton kernels, and restored. The
    counter counts matrix-product-class operations (``mm``, ``bmm``, ``addmm``, ``baddbmm``, convolutions,
    attention); elementwise work, reductions and factorizations are not counted, so the numbers are a lower bound
    on the forward work and are dominated by the GEMMs that set it on this model.

    ``flops_per_token_forward_measured`` (decode included), ``flops_per_token_beliefs_measured`` (the stack, no
    decode), ``flops_per_token_decode_measured`` (their difference), the top counted operators of the full forward,
    and ``est_train_flops_measured_3x`` :math:`3 F_{\mathrm{fwd}} D`, the forward count times the usual factor of
    three for forward plus backward over the run's ``tokens_seen`` :math:`D` (the E-step backward and the table
    optimizers are not separately modeled).
    """
    from torch.utils.flop_counter import FlopCounterMode
    from vfe3.relational_scoring import set_diagonal_moments_kernel

    device = device if device is not None else next(model.parameters()).device
    n = int(cfg.max_seq_len)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, int(cfg.vocab_size), (1, n), generator=generator).to(device)
    was_training = model.training
    rng_devices = [device] if device.type == "cuda" else []
    previous_kernel = set_diagonal_moments_kernel("torch")
    try:
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            model.eval()
            with FlopCounterMode(display=False) as with_decode:
                model.forward_beliefs(tokens, return_logits=True)
            with FlopCounterMode(display=False) as without_decode:
                model.forward_beliefs(tokens, return_logits=False)
    finally:
        set_diagonal_moments_kernel(previous_kernel)
        model.train(was_training)
    total = float(with_decode.get_total_flops()) / n
    beliefs = float(without_decode.get_total_flops()) / n
    by_op = with_decode.get_flop_counts().get("Global", {})
    top = sorted(((str(op), float(value) / n) for op, value in by_op.items()), key=lambda item: -item[1])[:8]
    return {
        "flops_per_token_forward_measured": total,
        "flops_per_token_beliefs_measured": beliefs,
        "flops_per_token_decode_measured": total - beliefs,
        "flops_measured_top_ops": {name: value for name, value in top},
        "flops_measured_seq_len": n,
        "flops_measured_scope": ("torch.utils.flop_counter matrix-product operators of one forward_beliefs "
                                 "sequence (eval, no grad); a lower bound on the forward work"),
        "est_train_flops_measured_3x": 3.0 * total * float(tokens_seen),
    }


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
    bracket closure, and the ``so_n`` decoupling of ``n_gen`` from ``K``.
    """
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    n_gen = int(model.group.generator_shape[0])
    n_blocks = max(1, len(model.group.irrep_dims))
    # Mean squared block dim: d_head^2 = (K / B)^2 for equal blocks, sum_b d_b^2 / B for a mixed irrep tower.
    block_dim_sq = sum(int(d) * int(d) for d in model.group.irrep_dims) / n_blocks
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
            covariance_layout=("full" if getattr(cfg, "family", None) in ("gaussian_full", "gaussian_block")
                               else "diagonal"),                                   # head blocks: dense storage
        )
    )
    block_mlp_active = block_mlp_build.enabled
    block_mlp_params = block_mlp_build.parameter_count
    if cfg.decode_mode != "linear":
        decode_readout = 2 * V * K
    else:
        decode_readout = V * K + V                                    # output_proj_weight, output_proj_bias
    active = token_row + decode_readout
    active += block_mlp_params
    # Audit 2026-09-28: every token also reads one row of the learned positional-frame table
    # (pos_phi_free is (max_seq_len, n_gen)) and the block normalization affine; both were
    # omitted (K160: +5,120 + 320 = 11,182,897). Parameters are deduplicated by identity, so a
    # normalization shared across layers counts once.
    pos_phi_free = getattr(model, "pos_phi_free", None)
    if (isinstance(pos_phi_free, torch.nn.Parameter) and pos_phi_free.dim() >= 1
            and pos_phi_free.shape[0] > 0):
        active += int(pos_phi_free.numel() // pos_phi_free.shape[0])
    norm_parameters: Dict[int, int] = {}
    norm_module = getattr(model, "block_norm", None)
    if isinstance(norm_module, torch.nn.Module):
        for parameter in norm_module.parameters():
            norm_parameters[id(parameter)] = int(parameter.numel())
    active += sum(norm_parameters.values())
    # Transparent analytic FLOP proxy. Per token: decode over all V and L*T belief E-step
    # iterations. The prior-bank decode is one (S, 2K) x (2K, V) product in its expanded form (the mean and the
    # second-moment features against the bank's (V, 2K) right factor; vfe3/kernels/kl_decode.py), so 4VK forward
    # FLOPs; the linear decode is (S, K) x (K, V), 2VK (audit 2026-10-20 item 48: 2VK understated the KL decode by 2). Each E-step iteration has O(N) attention energy (2NK) plus O(N) transport
    # application (2N times the mean squared block dim, d_head^2 for equal blocks). Constants are O(1); this is a proxy, not a calibrated count.
    L, T, N = int(cfg.n_layers), int(cfg.n_e_steps), int(cfg.max_seq_len)
    fpt_decode         = (2.0 if cfg.decode_mode == "linear" else 4.0) * V * K
    estep_kernel       = 2.0 * N * K + 2.0 * N * block_dim_sq
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
        "gauge_tying":             getattr(model.group, "tying", None),
        "decode_mode":             cfg.decode_mode,
        "vocab_size":              V,
        "n_learnable_params":      int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "active_params_per_token": int(active),
        "est_flops_analytic":      est_flops_analytic,
        "flops_per_token_decode":  fpt_decode,
        "flops_per_token_estep":   fpt_estep,
        "block_mlp_params":         int(block_mlp_params),
        "flops_per_token_block_mlp": fpt_block_mlp,
        **_execution_device_fields(model),
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
        out["est_flops_analytic_unavailable_reason"] = "projected routing/content cost is not modeled"
    if wall_time is not None and tokens_seen > 0:
        out["wall_time_s"]         = float(wall_time)
        out["wall_time_per_token"] = float(wall_time) / float(tokens_seen)
        out["wall_time_per_step"]  = float(wall_time) / max(1, int(tokens_seen) // (int(cfg.batch_size)
                                                                                   * int(cfg.max_seq_len)))
    return out




def _scheduler_metadata(
    *,
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
            "fractional_floor": 0.0 if frozen else float(min_lr_frac) * base,
            "effective_floor": 0.0 if frozen else float(min_lr_frac) * base,
        })
    return {
        "kind": "warmup_half_cosine_with_floor",
        "fractional_floor": float(min_lr_frac),
        "zero_base_groups_remain_frozen": True,
        "floor_description": (
            "0 for group_base_lr == 0; otherwise "
            "fractional_floor * group_base_lr"
        ),
        "groups": groups,
    }


def _resolved_alibi_slopes(cfg: VFE3Config) -> Optional[List[float]]:
    r"""Per-head ALiBi slopes the attention prior applies, or None for a non-ALiBi prior.

    The Press et al. schedule scaled by ``alibi_slope`` (:func:`vfe3.attention_prior._press_slopes`):
    for a power-of-two :math:`H`, :math:`s_h = s_0\, 2^{-8h/H}`, :math:`h = 1, \dots, H`; otherwise
    the nearest lower power-of-two schedule followed by the even-indexed entries of the next one.
    """
    name = getattr(cfg, "beta_attention_prior", None)
    if not isinstance(name, str) or "alibi" not in name:
        return None
    from vfe3.attention_prior import _press_slopes
    slopes = _press_slopes(int(cfg.n_heads), float(cfg.alibi_slope), "cpu", torch.float64)  # (H,)
    return [float(value) for value in slopes.tolist()]


def _best_weight_and_train_sample_evaluation(
    model:        torch.nn.Module,
    artifacts:    RunArtifacts,
    cfg:          VFE3Config,
    logger:       logging.Logger,

    *,
    final_step:      Optional[int],
    final_test:      Mapping[str, object],
    train_loader:    Optional[Iterable],
    val_loader:      Optional[Iterable],
    test_loader:     Optional[Iterable],
    tokens_per_char: Optional[float],
    device:          torch.device,
) -> Dict[str, object]:
    r"""Test CE at the best-validation weights and fixed training-sample CE at final and best weights.

    The headline stays the final weights (``evaluation_weights = "final"``); this adds
    ``test_best_{ce,ppl,step}`` for the validation-selected weights and
    ``train_sample_ce_{final,best}`` on the fixed training sample of
    :func:`vfe3.train._build_fixed_train_sample` (the one scored during training). A value already
    measured during training at the same step is reused. The best bundle is read with the
    run's selection contract (config, code identity, validation-data identity), loaded into the
    model, and the final state is restored bitwise from a host copy in a ``finally`` block. CPU and
    CUDA RNG are forked, so later finalize probes draw the same numbers as before.
    """
    from vfe3.train import TRAIN_SAMPLE_MAX_TARGETS, TRAIN_SAMPLE_SEED, _build_fixed_train_sample, evaluate

    out: Dict[str, object] = {
        "test_best_ce": None, "test_best_ppl": None, "test_best_step": artifacts.best_step,
        "test_best_status": None,
        "train_sample_ce_final": None, "train_sample_ce_best": None,
        # The training-time sample is capped at TRAIN_SAMPLE_MAX_TARGETS scored targets (32,768 since PR #75);
        # the post-hoc evaluator keeps its own larger cap, so the two train_sample CEs need not agree.
        "train_sample_max_targets": int(TRAIN_SAMPLE_MAX_TARGETS),
        "train_sample_seed": int(TRAIN_SAMPLE_SEED),
        "train_sample_scored_targets": None,
    }
    by_step = getattr(artifacts, "train_sample_by_step", None) or {}
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        sample = None
        if train_loader is not None:
            try:
                sample, reason = _build_fixed_train_sample(train_loader, val_loader, cfg)
                if sample is None:
                    out["train_sample_unavailable_reason"] = reason
                else:
                    out["train_sample_scored_targets"] = (
                        len(sample.indices) * int(sample.scored_targets_per_window))
            except Exception as exc:
                out["train_sample_unavailable_reason"] = repr(exc)
        if final_step is not None and final_step in by_step:
            out["train_sample_ce_final"] = float(by_step[final_step])
        elif sample is not None:
            try:
                out["train_sample_ce_final"] = float(evaluate(model, sample, device=device)["ce"])
            except Exception as exc:
                logger.warning("fixed training-sample CE at the final weights failed (%s)", exc)
        best_step = artifacts.best_step
        if best_step is None:
            out["test_best_status"] = "no validation-selected weights"
        elif final_step is not None and best_step == final_step:
            out.update(test_best_ce=final_test.get("test_ce"), test_best_ppl=final_test.get("test_ppl"),
                       train_sample_ce_best=out["train_sample_ce_final"],
                       test_best_status="best weights are the final weights")
        elif not artifacts.save_model_artifacts or not artifacts.best_path.is_file():
            out["test_best_status"] = "best_model.pt was not saved (save_model_artifacts=False)"
        else:
            final_state = {name: tensor.detach().to("cpu", copy=True)
                           for name, tensor in model.state_dict().items()}
            try:
                bundle = _read_best_model_bundle(
                    artifacts.best_path, artifacts.cfg, model.state_dict(), "cpu",
                    artifacts.code_identity_sha256, artifacts.selection_data_identity)
                model.load_state_dict(bundle["model_state"], strict=True)
                del bundle
                if test_loader is not None:
                    best = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
                    out.update(test_best_ce=float(best["ce"]), test_best_ppl=float(best["ppl"]),
                               test_best_bits_per_token=float(best["bits_per_token"]),
                               test_best_status="evaluated")
                    logger.info("Test at best-validation weights (step %s) | CE: %.4f | PPL: %.2f",
                                best_step, best["ce"], best["ppl"])
                else:
                    out["test_best_status"] = "no test loader"
                if best_step in by_step:
                    out["train_sample_ce_best"] = float(by_step[best_step])
                elif sample is not None:
                    out["train_sample_ce_best"] = float(
                        evaluate(model, sample, device=device)["ce"])
            except Exception as exc:
                out["test_best_status"] = f"failed: {exc!r}"
                logger.warning("test at best-validation weights failed (%s); headline unaffected", exc)
            finally:
                model.load_state_dict(final_state, strict=True)       # bitwise restore of the final weights
                del final_state
    return out


def _quantile_summary(values: Sequence[float]) -> Optional[Dict[str, float]]:
    r"""``n``, mean and the 5/25/50/75/95% quantiles (linear interpolation) of finite values."""
    finite = [float(value) for value in values
              if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return None
    data = torch.tensor(finite, dtype=torch.float64)
    levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    q05, q25, q50, q75, q95 = torch.quantile(data, levels).tolist()
    return {"n": len(finite), "mean": float(data.mean()),
            "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95}


def _resources_record(
    model:     torch.nn.Module,
    artifacts: RunArtifacts,
    cfg:       VFE3Config,

    *,
    device:    torch.device,
    wall_time: Optional[float],
) -> Dict[str, object]:
    r"""``resources.json``: parameter breakdown, optimizer/model bytes, peak memory, throughput.

    ``parameter_breakdown`` comes from :func:`vfe3.run_resources.parameter_breakdown` (imported
    lazily; a failure is recorded, not raised). ``optimizer_state_bytes`` is measured in
    :func:`vfe3.train.train` after the last step. ``peak_gpu_memory_mb`` is the maximum of the
    per-log-window ``peak_mem_mb`` column (train() resets the allocator peak at every log row) and
    the allocator peak since the last reset. Throughput quantiles are over the logged windows of
    ``tokens_per_s`` (pipeline, includes data loading) and ``train_step_tokens_per_s`` on the log and
    eval rows; the windows that contain an evaluation are included, so the pipeline quantiles bracket
    both, and divergence-spike rows (:mod:`vfe3.metrics_schema`) are left out.
    """
    record: Dict[str, object] = {
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                        else "cpu"),
        "device_type": device.type,
        "amp_dtype":   getattr(cfg, "amp_dtype", None),
        "wall_time_s": (float(wall_time) if wall_time is not None else None),
    }
    try:
        from vfe3.run_resources import parameter_breakdown
        record["parameter_breakdown"] = parameter_breakdown(dict(model.named_parameters()), cfg)
    except Exception as exc:
        record["parameter_breakdown"] = None
        record["parameter_breakdown_error"] = repr(exc)
    record["optimizer_state_bytes"] = getattr(artifacts, "optimizer_state_bytes", None)
    record["model_state_bytes"] = int(sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()))
    history = list(getattr(artifacts, "history", []) or [])
    peaks = [float(row["peak_mem_mb"]) for row in history
             if isinstance(row.get("peak_mem_mb"), (int, float))
             and math.isfinite(float(row["peak_mem_mb"]))]
    if device.type == "cuda":
        peaks.append(torch.cuda.max_memory_allocated(device) / 1e6)
    record["peak_gpu_memory_mb"] = max(peaks) if peaks else None
    # Divergence-spike rows close a short pipeline window around one anomalous step; their throughput is not
    # a sample of the regular windows, so the quantiles read the log and eval rows only (audit 2026-10-17 item 5).
    regular, _ = split_spike_rows(history, cfg=cfg)
    record["throughput_tokens_per_s"] = {
        column: _quantile_summary([row.get(column) for row in regular])
        for column in ("tokens_per_s", "train_step_tokens_per_s")
    }
    seen = [row.get("tokens_seen") for row in history
            if isinstance(row.get("tokens_seen"), (int, float))]
    record["tokens_seen"] = int(seen[-1]) if seen else None
    if wall_time is not None and wall_time > 0:
        record["device_hours"] = float(wall_time) / 3600.0
    return record


def _cumulative_wall_time(
    artifacts: "RunArtifacts",
    wall_time: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    r"""Return ``(run-total wall time, wall time carried in by the resumed checkpoint)``.

    The launchers time ``train()`` in their own process, so a resumed run's ``wall_time`` covers only the
    continuation; the checkpoint's carried wall clock (the same offset metrics.csv ``wall_clock_s`` uses)
    is added so summary.json and resources.json report the whole run (audit 2026-10-10, item 51).
    """
    resumed = getattr(artifacts, "resumed_wall_clock_s", None)
    if wall_time is None or resumed is None:
        return wall_time, resumed
    return float(wall_time) + float(resumed), float(resumed)


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
    # Background best-model saves and figure workers of the training loop finish before anything reads them
    # (audit 2026-10-17 item 68).
    wait = getattr(artifacts, "wait_background", None)
    if callable(wait):
        wait()
    if device is None:
        device = next(model.parameters()).device
    wall_time, resumed_wall_time = _cumulative_wall_time(artifacts, wall_time)

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
        config = _serialize_run_config(artifacts.cfg)
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

    # Another model family (the conventional baseline) runs the model-agnostic parts of finalize
    # (test at final and best weights, training sample, provenance, summary, resources) and skips the
    # VFE-only probes, recording why under "skipped_analyses".
    vfe_model = is_vfe_model(model)
    skipped_analyses: Dict[str, str] = {} if vfe_model else _non_vfe_skipped_analyses(model)
    # No "diagnostics" key: it was always an empty dict and nothing wrote into it (audit 2026-10-20 item 73); a
    # reader of an older test_results.json or summary.json finds it there and empty.
    results: Dict[str, object] = {
        **_conditional_artifact_fields(cfg),
    }                                                          # mixes scalar headlines and nested records
    if test_loader is not None:
        m = evaluate(model, test_loader, tokens_per_char=tokens_per_char, device=device)
        # A padded evaluation stream of T tokens holds T - 1 transitions, each scored exactly once,
        # so test_scored_targets == test_stream_transitions checks the window protocol.
        test_dataset = getattr(test_loader, "dataset", None)
        stream_transitions = (int(test_dataset.tokens.numel()) - 1
                              if getattr(test_dataset, "pad_final", False) else None)
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
            "test_overlap_context_targets": m["overlap_context_targets"],
            "test_padding_targets":         m["padding_targets"],
            "test_stream_transitions":      stream_transitions,
        })
        logger.info(
            "Test (held-out) | CE: %.4f | PPL: %.2f | BPT: %.4f | BPC: %s",
            m["ce"], m["ppl"], m["bits_per_token"],
            (f"{float(m['bpc']):.4f}" if m["bpc"] is not None else "unavailable"),
        )
    # Test at the best-validation weights and the fixed training-sample CE at final and best weights
    # (audit 2026-09-28). Published in summary.json only; the live final weights are restored
    # bitwise before any later probe or figure reads the model.
    try:
        best_weight_fields = _best_weight_and_train_sample_evaluation(
            model, artifacts, cfg, logger,
            final_step=final_step, final_test=results, train_loader=train_loader,
            val_loader=val_loader, test_loader=test_loader, tokens_per_char=tokens_per_char,
            device=device)
    except Exception as exc:
        logger.warning("best-weight / training-sample finalize evaluation failed (%s)", exc)
        best_weight_fields = {"test_best_ce": None, "test_best_ppl": None,
                              "test_best_step": artifacts.best_step,
                              "test_best_status": f"failed: {exc!r}",
                              "train_sample_ce_final": None, "train_sample_ce_best": None}
    best_val_ppl = artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None
    decode_tau_eff = _decode_tau_eff(model, cfg)
    # Test at the validation-selected weights beside the final-weights headline in test_results.json, as in
    # summary.json (audit 2026-10-20 item 7); the headline test_ce / test_ppl stay the final weights.
    results.update({key: best_weight_fields.get(key) for key in (
        "test_best_ce", "test_best_ppl", "test_best_bits_per_token", "test_best_step", "test_best_status")})
    if test_loader is not None and results.get("test_best_ppl") is not None and results.get("test_ppl") is not None:
        logger.info("Test perplexity: %.2f at the final weights (step %s), %.2f at the validation-selected weights "
                    "(step %s)", float(results["test_ppl"]), final_step, float(results["test_best_ppl"]),
                    results.get("test_best_step"))
    results.update({"best_val_ppl": best_val_ppl, "best_step": artifacts.best_step,
                    "reloaded_best": False, "evaluation_weights": "final",
                    "evaluated_step": final_step, **final_val_fields,
                    "decode_tau_eff": decode_tau_eff})

    # No inference-depth counterfactual, depth-sensitivity sweep or legacy reduced free energy is recorded:
    # the projected routes run one configured readout or receiver episode, so there is no depth to vary
    # and no shared stationary free energy (audit 2026-10-17 item 74).
    depth_loader = val_loader if val_loader is not None else test_loader
    # Mechanism records are standard outputs. The sole figure-policy switch controls only the
    # embedding/clustering family downstream in the isolated report worker; it never suppresses
    # numeric probes or ordinary publication figures. Batch extraction remains best-effort so an
    # exhausted or malformed loader cannot invalidate the already-computed headline metrics.
    _depth_tokens = _depth_targets = None
    if depth_loader is not None and vfe_model:
        try:
            _batch = next(iter(depth_loader))
            if not isinstance(_batch, (tuple, list)) or len(_batch) < 2:
                raise ValueError("mechanism diagnostics require a (tokens, targets) batch")
            _depth_tokens = _batch[0].to(device)
            _depth_targets = _batch[1].to(device)
        except Exception as exc:
            logger.warning("diagnostic batch unavailable (%s); mechanism probes skipped", exc)
            _count_finalize_failure(artifacts, "mechanism_probe_batch")
    if _depth_tokens is not None:
        _character_record = None
        _beta_record = None
        _context_record = None
        try:
            _character_record = collect_estep_character(model, _depth_tokens)
            artifacts.save_json("estep_character.json", _character_record)
        except Exception as exc:
            logger.warning("estep character probe failed (%s); skipped", exc)
            _count_finalize_failure(artifacts, "estep_character")
        try:
            _beta_record = collect_beta_channel_decomposition(model, _depth_tokens, _depth_targets)
            artifacts.save_json("beta_channel_decomposition.json", _beta_record)
        except Exception as exc:
            logger.warning("beta channel decomposition failed (%s); skipped", exc)
            _count_finalize_failure(artifacts, "beta_channel_decomposition")
        try:
            _context_record = collect_context_sensitivity(model, _depth_tokens)
            artifacts.save_json("context_sensitivity.json", _context_record)
        except Exception as exc:
            logger.warning("context-sensitivity probe failed (%s); skipped", exc)
            _count_finalize_failure(artifacts, "context_sensitivity")
        _log_mechanism_diagnostics(logger, _character_record, _beta_record, _context_record)
    if depth_loader is not None and vfe_model:
        try:
            _phi_batch = next(iter(depth_loader))
            _phi_tokens = (
                _phi_batch[0] if isinstance(_phi_batch, (tuple, list)) else _phi_batch
            ).to(device)
            _phi_record = collect_phi_numerics(model, _phi_tokens)
            artifacts.save_json("phi_numerics.json", _phi_record)
        except Exception as exc:
            logger.warning("phi numerical-reference probe failed (%s); skipped", exc)
            _count_finalize_failure(artifacts, "phi_numerics")
    if skipped_analyses:
        results["model_family"] = model_family_of(model)
        results["skipped_analyses"] = dict(skipped_analyses)
        logger.info("model_family=%r: VFE-only finalize analyses skipped: %s",
                    model_family_of(model), ", ".join(sorted(skipped_analyses)))
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
    # Trainable count: parameters frozen by design (requires_grad=False, e.g. the zero token frames
    # under gauge_transport='off') are run constants and are excluded (resources.json lists them).
    n_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # "n_steps" is the CONFIGURED budget. The realized counts sit beside it rather than replacing it
    # (audit 2026-08-06 F5): the LR scheduler clocks on accepted updates, so a run that rejected a
    # third of its steps trained on a different schedule than its config claims, and sweep arms with
    # different skip rates are not comparable. Absent on a train() call that had no artifacts.
    _realized = dict(getattr(artifacts, "realized_updates", None) or {})
    # Tokens of the steps actually run (attempted_steps; cfg.max_steps without a realized record).
    steps_run = int(_realized.get("attempted_steps", cfg.max_steps))
    tokens_seen = steps_run * int(cfg.batch_size) * int(cfg.max_seq_len)
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
        if vfe_model:
            scaling_point.update(_cost_model_fields(model, cfg, n_params, tokens_seen, wall_time=wall_time))
            try:                                       # the projected route's measured cost (audit 2026-10-20 #48)
                scaling_point.update(_measured_forward_flops(model, cfg, tokens_seen, device=device))
            except Exception as flop_exc:
                logger.warning("measured forward FLOPs failed (%s); scaling_point keeps the analytic fields",
                               flop_exc)
        else:
            scaling_point.update(model.cost_model_fields(n_params, tokens_seen, wall_time=wall_time))
    except Exception as exc:
        logger.warning("cost-model fields failed (%s); scaling_point keeps the 6ND proxy only", exc)
    artifacts.save_json("summary.json", {
        **_conditional_artifact_fields(cfg),
        "n_steps":      cfg.max_steps,
        **_realized,
        "n_params":     n_params,
        "n_params_trainable": n_params_trainable,
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
        "test_overlap_context_targets": results.get("test_overlap_context_targets"),
        "test_padding_targets": results.get("test_padding_targets"),
        "test_stream_transitions": results.get("test_stream_transitions"),
        # No estep_final_iterate_f_per_token / estep_final_f_per_token / estep_iterate_evidence: nothing on the
        # projected routes produces an E-step iterate sequence, so those keys were always null (audit 2026-10-16
        # F86).
        "scheduler": _scheduler_metadata(
            min_lr_frac=cfg.min_lr_frac,
            base_lrs=getattr(artifacts, "scheduler_base_lrs", ()),
            group_roles=getattr(artifacts, "scheduler_group_roles", ()),
        ),
        "final_train_loss": (losses[-1] if losses else None),
        "wall_time_s":  wall_time,
        "wall_time_resumed_s": resumed_wall_time,     # carried in by a resumed checkpoint (in wall_time_s)
        # A resumed run's metrics.csv starts after resume_source.saved_step; earlier rows (and a best_step at or
        # before it) are in resume_source.source_run_dir. None from scratch.
        "resume_source": getattr(artifacts, "resume_source", None),
        "decode_mode":     cfg.decode_mode,
        "decode_tau_eff":  decode_tau_eff,
        # The expected-likelihood CE kernel that last ran ('torch' | 'triton'; None when that decoder never
        # ran): CUDA tensors run the fused Triton kernel, CPU tensors the chunked torch kernel (audit
        # 2026-10-10 round 2, #42; selector removed 2026-10-12).
        "expected_likelihood_ce_kernel": getattr(getattr(model, "prior_bank", None),
                                                 "realized_expected_likelihood_kernel", None),
        # The diagonal_chunked / full_chunked CE kernel that last ran ('torch' | 'triton'; None when neither
        # ran): 'triton' needs decode_ce_triton=True and CUDA tensors (2026-10-12).
        "decode_ce_kernel": getattr(getattr(model, "prior_bank", None), "realized_decode_ce_kernel", None),
        "phi_chart_norm_route": _phi_chart_norm_route(model, cfg),
        "scaling_point":   scaling_point,
        # Audit 2026-09-28: test at the validation-selected weights beside the final-weight
        # headline, the fixed training-sample CE at both, the per-head ALiBi slopes the prior
        # applied, and the per-name count of training-time diagnostics that failed (and were
        # skipped without affecting training).
        **best_weight_fields,
        "alibi_slopes":    _resolved_alibi_slopes(cfg),
        "diagnostics_failures": dict(sorted(
            (getattr(artifacts, "diagnostics_failures", None) or {}).items())),
        # S1 panel D counters (2026-09-28; see vfe3.train._numerics_counters_summary): final
        # cumulative counts, the last evaluation's cap/factorization counts, and skip reasons.
        "numerics_counters": getattr(artifacts, "numerics_counters", None),
        # Another model family: its tag and the VFE-only analyses finalize skipped, with reasons.
        **({"model_family": model_family_of(model), "skipped_analyses": dict(skipped_analyses)}
           if not vfe_model else {}),
    })
    try:
        artifacts.save_json("resources.json", _resources_record(
            model, artifacts, cfg, device=device, wall_time=wall_time))
    except Exception as exc:
        logger.warning("resources.json failed (%s); skipped", exc)
    if not vfe_model:
        return results

    # Pure-path certificate: the config toggles for the principal gauge / decode / free-energy purity
    # axes (flat gauge, canonical F, prior-bank decode, ...), plus the converged-state
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
    if beta is not None and beta.get("available"):
        deltas = beta["delta_ce"]
        if beta.get("energy_includes_rope_position"):
            # Under RoPE the routing cost carries position too; the rotation-only arm is a separate,
            # non-additive ablation (audit 2026-10-16, F25).
            rope = deltas.get("rope_position_channel")
            logger.info(
                "Attention channels (delta CE, nats): positional prior %+.4f vs routing cost "
                "(content + RoPE position) %+.4f%s",
                deltas["positional_channel"], deltas["content_channel"],
                (f" vs RoPE position (rotation only) {rope:+.4f}" if rope is not None else ""))
        else:
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
) -> Dict[str, object]:
    r"""How far the configured projected readout or receiver episode moves the belief.

    ``rel_displacement`` is :math:`\lVert\mu_q-\mu_p\rVert/\lVert\mu_p\rVert` between the first
    and last captured states of one diagnostic snapshot (pre-transform: before CG, norm
    and MLP). Only the configured depth is measured, so ``depth_sweep_available`` is False and
    ``cos_dir_vs_step1`` is None.
    """
    snapshot = model.build_diagnostic_snapshot(tokens)
    mu_p, mu_q = snapshot.trace_states[0].mu, snapshot.trace_states[-1].mu
    denominator = float(mu_p.norm())
    return {
        "available": True, "inference_semantics": model.conditional_semantics,
        "trained_depth": len(snapshot.trace_states) - 1, "depth_sweep_available": False,
        "depth_sweep_reason": ("only the configured receiver episode is supported"
                               if model.cfg.relational_readout_mode == "full_link_update"
                               else "only the configured single readout is supported"),
        "points": [{"belief_depth": len(snapshot.trace_states) - 1, "rel_displacement": (
            float((mu_q - mu_p).norm()) / denominator if denominator > 0 else None),
            "cos_dir_vs_step1": None,
            "displacement_window": "captured_pre_transform_readout"}],
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
    belief-coupling energy -- the only content-dependent channel. Three arms isolate them, and a fourth
    (``no_rope``, below) runs under RoPE:

    ============ ============================================ ==================================
    arm          beta                                         reads
    ============ ============================================ ==================================
    ``full``     ``softmax(log_prior - E/tau)``                as trained; reproduces the baseline
    ``no_energy``  ``softmax(log_prior)``                      the CONTENT channel's worth (plus
                                                               RoPE position under a rope_mode)
    ``no_prior``   ``softmax(-E/tau)``                         the POSITIONAL channel's worth
    ``no_rope``    ``full`` with the rotation :math:`R_i = I`  RoPE position (rope_mode != 'none')
    ============ ============================================ ==================================

    Measured 2026-07-25 on the K=300 baseline: flattening the prior cost 0.612 nats against 0.210
    for ablating the whole coupling energy, so the positional prior carried roughly three times the
    content channel. ``full`` reproducing the headline CE is the harness check -- if it drifts, the
    patch missed the live softmax and the other two arms mean nothing.

    Patching every ``attention_weights`` binding isolates the live projected router on both
    predictive and full-link gradient attention. ``available`` records whether any
    live softmax was actually intercepted; when it is False the arms are omitted rather than
    reported as numbers that describe nothing.

    ``no_prior`` preserves ``-inf`` entries of the log-prior so the causal / no-self MASK survives
    while its graded positional shape is flattened. A configuration carrying no attention log-prior
    at all therefore leaves ``no_prior`` equal to ``full``, correctly reporting that this model has
    no positional channel to ablate.

    Under a ``rope_mode`` other than ``'none'`` position also enters through the routing cost itself (the rotated
    comparison moments, or the rotation folded into the frame), so ``no_energy`` then removes RoPE
    position together with content. A fourth arm ``no_rope`` keeps the cost and the prior and removes
    only the rotation; ``delta_ce.rope_position_channel`` reports it and ``energy_includes_rope_position``
    flags the reading (audit 2026-10-10, item 5). Under a comparison ``rope_mode`` the arm sets each
    operator's comparison rotation to :math:`R_i = I` and keeps everything else, including the commutant
    projection :math:`P(G)` of ``rope_mode='comparison_invariant'``.
    """
    import torch.nn.functional as F
    from vfe3 import free_energy as free_energy_module
    from vfe3.gradients import full_link as full_link_module
    from vfe3.inference import e_step as e_step_module

    was_training = bool(model.training)
    # Patch EVERY module that binds `attention_weights`, definition site included (audit
    # 2026-07-26 B-02); the predictive readout imports it from the definition site at call time.
    original = free_energy_module.attention_weights
    binders = [module for module in (free_energy_module, e_step_module, full_link_module)
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

    cfg = getattr(model, "cfg", None)
    rope_active = getattr(cfg, "rope_mode", "none") != "none"
    operators = list(getattr(model, "relational_operators_by_layer", None) or [])
    comparison_ops = [ops for ops in operators if getattr(ops, "comparison_rope_base", None) is not None]

    def _identity_rotation(n: int, r: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.eye(r, device=device, dtype=dtype).expand(n, r, r).contiguous()        # (N, r, r) R_i = I

    def _set_rope(enabled: bool) -> None:
        # Comparison-space RoPE lives on each layer's operators; frame-space RoPE is the model's rotation.
        # The comparison arm keeps comparison_rope_base, which also gates the comparison_invariant commutant
        # projection P(G), and replaces each operator's rotation R_i by the identity, so only the rotation
        # is removed (audit 2026-10-10 round 2, #19).
        for ops in comparison_ops:
            if enabled:
                ops.__dict__.pop("comparison_rope_rotation", None)
            else:
                ops.__dict__["comparison_rope_rotation"] = _identity_rotation
        if enabled:
            model.__dict__.pop("_rope_rotation", None)
        elif getattr(cfg, "rope_mode", "none") in ("frame", "frame_value"):
            model._rope_rotation = lambda n, device: None

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.device_count() > 0 else None)
    try:
        model.eval()
        for module in binders:
            module.attention_weights = _counting
        for arm in ("full", "no_energy", "no_prior") + (("no_rope",) if rope_active else ()):
            mode["arm"] = "full" if arm == "no_rope" else arm
            _set_rope(arm != "no_rope")
            logits = model(tokens)
            arms[arm] = float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-100,
            ))
    finally:
        _set_rope(True)
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
        "n_sequences":   int(tokens.shape[0]),
        "ce":            arms if available else {},
        "delta_ce": ({
            "content_channel":    (arms["no_energy"] - baseline) if available else None,
            "positional_channel": (arms["no_prior"] - baseline) if available else None,
            **({"rope_position_channel": arms["no_rope"] - baseline} if rope_active else {}),
        } if available else {}),
        "energy_includes_rope_position": rope_active,
        "interpretation": (
            "delta_ce.content_channel is the cost of ablating the belief-coupling energy from beta "
            "(what CONTENT is worth, plus RoPE position when energy_includes_rope_position is true); "
            "delta_ce.positional_channel is the cost of flattening the attention log-prior on its causal "
            "support (what the PRIOR's position is worth); delta_ce.rope_position_channel, present under "
            "RoPE, is the cost of removing only the rotation. available=False means no live beta softmax "
            "was intercepted, and the arms carry no meaning."
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
    control and must read 0; ``belief_estep`` (``predictive_content`` on the predictive route) reads the
    final belief after every layer and the final norm, so the increment is the context the whole stack
    injects. ``layer0_readout`` keeps the layer-0 attention output the record measured before (the
    E-step trace is captured at layer 0 only; audit 2026-10-10, item 34).

    This standard mechanism record costs ``n_replicates`` extra forward passes per stage.
    """
    from vfe3.viz.extract import e_step_belief_trace

    if type(n_replicates) is not int or n_replicates < 1:
        raise ValueError(f"n_replicates must be an int >= 1, got {n_replicates!r}")
    was_training = bool(model.training)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.device_count() > 0 else None)

    vocab = int(model.cfg.vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def _trace_mean(batch: torch.Tensor, initial: bool = False) -> torch.Tensor:
        trace = e_step_belief_trace(model, batch, n_iter=int(model.cfg.n_e_steps))
        return torch.as_tensor(trace["mu"][0 if initial else -1])[-1].flatten().float()

    def _final_belief_mean(batch: torch.Tensor) -> torch.Tensor:
        belief, _ = model.forward_beliefs(batch, return_logits=False)
        return belief.mu[0, -1].detach().flatten().float()           # sequence 0, final position

    stages: Dict[str, Optional[float]] = {}
    try:
        model.eval()
        readout_name = ("predictive_content"
                        if model.cfg.relational_readout_mode == "predictive_content" else "belief_estep")
        # Depth 0 reads the captured initial state of the configured forward (no depth sweep).
        variants = [("raw_prior", lambda batch: _trace_mean(batch, initial=True)),
                    (readout_name, _final_belief_mean),
                    ("layer0_readout", _trace_mean)]
        for name, final_mean in variants:
            reference = final_mean(tokens)
            shifts = []
            for _ in range(n_replicates):
                perturbed = tokens.clone()
                if perturbed.shape[1] > 1:
                    perturbed[:, :-1] = torch.randint(
                        0, vocab, perturbed[:, :-1].shape, generator=generator, dtype=perturbed.dtype
                    ).to(perturbed.device)
                moved = final_mean(perturbed)
                denominator = float(reference.norm())
                shifts.append(float((moved - reference).norm()) / denominator
                              if denominator > 0.0 else 0.0)
            stages[name] = sum(shifts) / len(shifts)
    finally:
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
            "pure per-token lookup; belief_estep / predictive_content reads the final belief after "
            "every layer, so the increment is the context the stack injects; layer0_readout is the "
            "same measure at layer 0's attention output."
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
                if torch.cuda.device_count() > 0 else None)
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
            # right_phi is (N, n_gen) for one sequence; only a batched (B, N, n_gen) chart takes [0]
            # (audit 2026-10-07 F66).
            right_chart = belief.right_phi[0] if belief.right_phi.dim() == 3 else belief.right_phi
            record["right_chart"] = metrics.phi_chart_statistics(
                right_chart,
                model.group,
                block_dims=block_dims,
            )
        # The fp32 leg runs under IEEE fp32 GEMMs so "fp32" names the arithmetic, not a TF32 policy
        # (audit 2026-10-09); the fp64 leg is unaffected by the fp32 matmul policy.
        record["process_fp32_matmul_precision"] = torch.get_float32_matmul_precision()
        with ieee_fp32_matmul():
            record["flatness_fp32_matmul_precision"] = torch.get_float32_matmul_precision()
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
    if cuda is not None and torch.cuda.device_count() > 0:
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
    publishing no summary (and, via the caller, no success contract). When ``terminal_state.final_validation``
    carries the training loop's validation of the same final weights on this loader and ``tokens_per_char``,
    those metrics are reused and the loop's row for that step stands in for the terminal row.

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
    wait = getattr(artifacts, "wait_background", None)       # background saves and figure workers (item 68)
    if callable(wait):
        wait()
    if device is None:
        device = next(model.parameters()).device
    wall_time, resumed_wall_time = _cumulative_wall_time(artifacts, wall_time)
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
    n_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    final_train_loss = (float(losses[-1]) if losses else None)
    vfe_model = is_vfe_model(model)

    # The loop already scored these weights on this loader at the final step and wrote that step's row: reuse
    # it rather than evaluating twice and logging a second row for the same step (audit 2026-10-12 #12).
    loop_validation = getattr(terminal_state, "final_validation", None)
    reuse_loop_validation = (loop_validation is not None
                             and loop_validation["val_loader"] is val_loader
                             and loop_validation["tokens_per_char"] == tokens_per_char)
    try:
        metrics = (dict(loop_validation["metrics"]) if reuse_loop_validation
                   else evaluate(model, val_loader, tokens_per_char=tokens_per_char, device=device))
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
        if not reuse_loop_validation:
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
        if vfe_model:                   # VFE-only reports (see _non_vfe_skipped_analyses)
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
        # Realized-update counts and training-time diagnostic failures, as finalize_run publishes them
        # (train() sets both before the terminal callback; audit 2026-10-12 round 2, item 7).
        **dict(getattr(artifacts, "realized_updates", None) or {}),
        "n_params":            n_params,
        "n_params_trainable":  n_params_trainable,
        "final_train_loss":    final_train_loss,
        "wall_time_s":         (float(wall_time) if wall_time is not None else None),
        "wall_time_resumed_s": resumed_wall_time,     # carried in by a resumed checkpoint (in wall_time_s)
        "resume_source":       getattr(artifacts, "resume_source", None),   # as in finalize_run
        "terminal_checkpoint": terminal_checkpoint,
        "save_model_artifacts": artifacts.save_model_artifacts,
        "figures_written":     figures_written,
        "phi_chart_norm_route": _phi_chart_norm_route(model, cfg),
        "decode_tau_eff":      _decode_tau_eff(model, cfg),
        "diagnostics_failures": dict(sorted(
            (getattr(artifacts, "diagnostics_failures", None) or {}).items())),
        "numerics_counters":   getattr(artifacts, "numerics_counters", None),
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
        # The realized-update counts, so arms with different skip rates are flagged in every comparison
        # table, not only in summary.json (audit 2026-10-17 item 29). None without a realized record.
        "skipped_steps":       dict(getattr(artifacts, "realized_updates", None) or {}).get("skipped_steps"),
        "accepted_update_frac": dict(getattr(artifacts, "realized_updates", None) or {}).get(
            "accepted_update_frac"),
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
    decoder coordinate, and immutable BlockMLP seams. ``on_causal_lm_path`` comes
    independently from both active attention-prior registrations. ``transport_exactness_status`` is affirmative only when
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
    from vfe3.geometry.transport import get_transport_registration
    from vfe3.attention_prior import get_prior_registration

    invariant_families = declared_invariant_families(cfg.gauge_group)
    # gaussian_frame_diagonal stores c_i = diag(U_i^{-1} Sigma_i U_i^{-T}), invariant under the passive action of every
    # group (the frame carries the action), so the family is covariant whatever the group declares.
    from vfe3.families import get_family as _family_of
    family_group_invariant = (cfg.family in invariant_families
                              or bool(getattr(_family_of(cfg.family), "frame_diagonal_covariance", False)))
    transport_registration = get_transport_registration("flat")       # the only projected transport
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
    if block_mlp_active and block_mlp_covariance == "delta_full" and (
            getattr(cfg, "norm_type_block", "none") == "layernorm"):
        # Through the block LayerNorm the floor is f D Sigma D, D = diag(gamma)/s: post-norm units, not the
        # covariant f Sigma (audit 2026-10-09 round 2, #13). In reference coordinates the composite acts on
        # C = U^-1 Sigma U^-T and is mapped back (audit 2026-10-10 round 2, item 51).
        block_mlp_covariance_contract = (
            "delta_full_through_reference_layernorm_plus_post_norm_scaled_floor"
            if getattr(cfg, "block_coordinates", "local") == "reference"
            else "delta_full_through_layernorm_plus_post_norm_scaled_floor")
    if block_mlp_active and block_mlp_covariance != "passthrough" and cfg.family == "gaussian_block":
        # The block's covariance is read through the head-block M-projection (vfe3.model.block._head_block_readout).
        block_mlp_covariance_contract = f"head_block_projection_of_{block_mlp_covariance_contract}"
    if block_mlp_active and block_mlp_covariance == "delta_full" and cfg.family == "gaussian_diagonal":
        # The diagonal family keeps diag(J Sigma J^T) + floor, the M-projection of the dense pushforward (audit
        # 2026-10-10, item 9; vfe3.model.block_mlp._diagonal_delta_covariance).
        block_mlp_covariance_contract = f"diagonal_projection_of_{block_mlp_covariance_contract}"
    runtime_exactness_key = transport_registration.runtime_exactness_key
    if runtime_exactness_key is None:
        # No runtime exactness certificate is registered for this transport: the flat cocycle is
        # exact by construction (no edge factor to approximate), so there is nothing to observe.
        transport_runtime_exact = True
        transport_runtime_exactness = "not_applicable"
    elif not family_group_invariant:
        transport_runtime_exact = False
        transport_runtime_exactness = "diagonal_projection_approximation"
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
        transport_runtime_exact = complete and all(value >= 0.5 for value in observations)
        transport_runtime_exactness = (
            "exact_valid_spd_factorization" if transport_runtime_exact else
            "jitter_recovered_approximation" if any(value < 0.5 for value in finite_observations) else
            "unknown_missing_runtime_evidence"
        )
    if runtime_exactness_key is None:
        transport_exactness_status = "not_applicable"
    elif not transport_registration.gauge_equivariant or not family_group_invariant:
        transport_exactness_status = "approximate"
    elif transport_runtime_exactness == "unknown_missing_runtime_evidence":
        transport_exactness_status = "unknown"
    else:
        transport_exactness_status = "exact" if transport_runtime_exact else "approximate"

    # Predictive attention assigns its configured readout covariance; projected_gradient
    # executes Fisher receiver updates unless its covariance update is disabled.
    attention_mode = cfg.attention_mode
    receiver_gradient = attention_mode == "projected_gradient"
    predictive_covariance_updated = (
        attention_mode == "projected_predictive"
        and getattr(cfg, "predictive_covariance", "passthrough") in READOUT_COVARIANCES)
    covariance_updated = (receiver_gradient and not cfg.skip_belief_sigma_update) or predictive_covariance_updated
    # The receiver always descends its own energy by Fisher steps or the closed-form update.
    gradient_e_step_update: 'bool | str' = True if receiver_gradient else "not_applicable"
    # The closed-form receiver sets the covariance from natural parameters, Sigma = c Lambda^{-1}; it
    # runs no AIRM retraction, trust projection or exponential, and only its diagonal branch clamps
    # at sigma_max (audit 2026-10-07 F72).
    closed_form_receiver = getattr(cfg, "relational_receiver_update", "closed_form") in (
        "closed_form", "closed_form_profiled")
    sigma_step = receiver_gradient and not cfg.skip_belief_sigma_update
    spd_retraction_applies = sigma_step and not closed_form_receiver

    sigma_max = getattr(cfg, "sigma_max", 10.0)
    sigma_cap_consumer_active = sigma_step and (not closed_form_receiver or cfg.diagonal_covariance)
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
    if sigma_step and closed_form_receiver:
        spd_retraction_route = ("closed_form_natural_parameter_spectral_cap"
                                if sigma_cap_consumer_active and sigma_max is not None
                                else "closed_form_natural_parameter")
        spd_retraction_exact = spd_retraction_route == "closed_form_natural_parameter"
        spd_retraction_trust_projection_active = "not_applicable"
        spd_retraction_exponential_clipping_active = "not_applicable"
    elif not spd_retraction_applies:
        spd_retraction_route = "not_applicable"
        spd_retraction_exact = "not_applicable"
        spd_retraction_trust_projection_active = "not_applicable"
        spd_retraction_exponential_clipping_active = "not_applicable"

    beta_prior = get_prior_registration(getattr(cfg, "beta_attention_prior", "causal"))
    causal_flags = {
        "beta_prior_causal": beta_prior.is_causal_for_next_token_scoring(cfg),
        "inference_updates_causal": True,              # projected inference is filtering-only
    }
    on_causal_lm_path = all(causal_flags.values())

    block_norm_name = getattr(cfg, "norm_type_block", "none")
    block_norm_registration = get_norm_registration(block_norm_name)
    transport_chart_max_norm = float(cfg.transport_chart_max_norm)
    reference_blocks = getattr(cfg, "block_coordinates", "local") == "reference"

    # Readers of the self term alpha_i D(q_i||p_i) + R(alpha_i). Under state_dependent_per_coord a family with
    # conditional_kl_coordinates (gaussian_full, gaussian_block) supplies the ordered conditional KLs D^(k), whose
    # sum is frame invariant but whose terms move under a per-head GL element; the diagonal family's marginal KLs
    # are invariant under the diagonal maps it admits (its GL closure is family_group_invariant). The canonical
    # diagnostic reads the self term but trains nothing, so it is recorded without gating the gauge-pure path.
    from vfe3.alpha_i import alpha_is_per_coord
    from vfe3.families import get_family
    model_self_coupling_consumers = [name for name, live in (
        ("projected_gradient_receiver", cfg.attention_mode == "projected_gradient"),
        ("mstep_self_coupling", float(getattr(cfg, "mstep_self_coupling_weight", 0.0)) > 0.0)) if live]
    self_coupling_consumers = model_self_coupling_consumers + (
        ["canonical_free_energy_diagnostic"] if getattr(cfg, "diagnose_canonical_free_energy", False) else [])
    per_coord_conditional_kl = (alpha_is_per_coord(cfg.lambda_alpha_mode)
                                and bool(getattr(get_family(cfg.family), "conditional_kl_coordinates", False)))
    comparison_rope = cfg.rope_mode in ("comparison", "comparison_invariant")
    # comparison_invariant scores the commutant projection P(G) of the kernel router's comparison covariance:
    # only the projected means rotate (vfe3.gradients.relational.relational_score).
    rope_commutant_covariance = (
        cfg.rope_mode == "comparison_invariant"
        and getattr(cfg, "relational_routing", "gaussian_kl") != "mean_distance")

    conditional_fields = _conditional_artifact_fields(cfg)
    pure_flags = {
        "canonical_attention_entropy": not conditional_fields,
        "flat_transport":              True,
        "prior_bank_decode":           cfg.decode_mode != "linear",
        # The prior-bank logit is the exact -KL(q*||p_v)/tau only when it reads the belief's own covariance.
        "decode_reads_belief_covariance": getattr(cfg, "decode_covariance", "belief") == "belief",
        # Executed-route facet: gradient updates sigma unless skipped; predictive readout
        # keeps it at encode unless moment_matched assigns the predictive covariance.
        "full_sigma_update":           covariance_updated,
        "no_block_mlp":                not block_mlp_active,
        # The E-step update-rule axis (audit 2026-07-26 D-03): True on the receiver route, whose
        # only updater is gradient descent of its own energy; 'not_applicable' on
        # projected_predictive, where no E-step descent runs.
        "gradient_e_step_update":      gradient_e_step_update,
    }
    # Second, INDEPENDENT purity axis (audit 2026-07-01 F8): the gauge / model-channel path. Keyed
    # on rope_mode itself; the RoPE base and fraction are inert while RoPE is off and are reported in
    # config_toggles for transparency.
    gauge_flags = {
        "learned_gauge_transport":   cfg.gauge_transport == "on",
        "transport_gauge_equivariant": transport_registration.gauge_equivariant,
        # RoPE is folded on the right of the realized frame (U_i R_i^T), the same right-factor
        # structure as pos_phi, so both projected routes stay covariant (audit 2026-10-01 F11).
        "positional_rotation_gauge_covariant": (
            cfg.rope_mode == "none" or getattr(cfg, "rope_insertion", "right") == "right"),
        "family_group_invariant":    family_group_invariant,
        # block_coordinates='reference' conjugates the block norm and block MLP by the
        # realized frame, x -> U_i N(U_i^{-1} x), which is covariant for any registered map
        # (audit 2026-10-07 F18). The canonical_frame MLP is the same dressing U_i F(U_i^{-1} x) applied to the MLP
        # alone, covariant under the per-token action U_i -> g_i U_i (audit 2026-10-10, item 42).
        "block_mlp_intertwiner_compatible": (block_mlp_intertwiner_compatible or reference_blocks
                                             or (block_mlp_active and block_mlp_mode == "canonical_frame")),
        "block_norm_gauge_equivariant": block_norm_registration.gauge_equivariant or reference_blocks,
        "no_fixed_coordinate_spectral_cap": (
            sigma_max is None if sigma_cap_consumer_active else "not_applicable"),
        # The chart-norm ball and the frame condition bound both rewrite frame rows after a step (round 2, item 14).
        "no_phi_mstep_chart_cap": (getattr(cfg, "phi_mstep_max_matrix_norm", None) is None
                                   and (getattr(cfg, "phi_frame_condition_max", None) is None
                                        or getattr(cfg, "gauge_group", None) not in FRAME_CONDITION_GROUPS
                                        or getattr(cfg, "gauge_transport", "on") != "on")),
        "no_pullback_trust_region": getattr(cfg, "m_phi_update_mode", "adamw") not in (
            "pullback_group", "fisher_group"),
        # The transport exponential is never rescaled: its chart radius is fail-closed (2026-10-02).
        "no_transport_exponential_clipping": True,
        "no_additive_encoder_control": getattr(cfg, "encode_mode", "per_token") != "per_token_additive",
        "decoder_gauge_compatible": getattr(cfg, "decode_coordinates", "local") == "reference",
        # False when a per-coordinate alpha over ordered conditional KLs has a live model consumer (audit 2026-10-10
        # round 2, #16); vfe3.viz.extract.projected_gauge_equivariance witnesses it against an independent prior.
        "self_coupling_frame_invariant": not (per_coord_conditional_kl and model_self_coupling_consumers),
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
            "lambda_alpha_mode":            cfg.lambda_alpha_mode,
            "self_coupling_consumers":      self_coupling_consumers,
            "lambda_beta":                  float(cfg.lambda_beta),
            "decode_mode":                  cfg.decode_mode,
            "decode_covariance":            getattr(cfg, "decode_covariance", "belief"),
            "decode_coordinates":           getattr(cfg, "decode_coordinates", "local"),
            "block_coordinates":            getattr(cfg, "block_coordinates", "local"),
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
            "block_norm_covariance":        getattr(cfg, "block_norm_covariance", "passthrough"),
            "block_norm_trace_cap":         bool(getattr(cfg, "block_norm_trace_cap", True)),
            "gauge_transport":              cfg.gauge_transport,
            "rope_mode":                    cfg.rope_mode,
            # Which RoPE regime the routing score used (audit 2026-07-26 A-02). A comparison rope_mode
            # folds nothing into the frame: it rotates the projected moments by R_i after the comparison
            # map, so the pair cost depends on positions through j - i (audit 2026-10-08 C#6). Under
            # rope_mode='comparison_invariant' only the means rotate and the covariance is the
            # commutant projection P(G), with R_i P(G) R_i^T = P(G) (audit 2026-10-10 round 2, #33).
            "rope_pair_energy_exactness":   (
                "not_applicable" if cfg.rope_mode == "none"
                else "comparison_space_rotated_means_commutant_projected_covariance"
                if rope_commutant_covariance
                else "comparison_space_rotated_projected_moments"
                if comparison_rope
                else "frame_folded_both_moments"   # projected routes push both moments through U_i R_i^T
            ),
            "rope_fraction":                (None if cfg.rope_mode == "none"
                                             else float(cfg.rope_fraction)),
            "rope_base":                    (None if cfg.rope_mode == "none"
                                             else float(cfg.rope_base)),
            "skip_belief_sigma_update":      bool(cfg.skip_belief_sigma_update),
            "predictive_covariance":         getattr(cfg, "predictive_covariance", "passthrough"),
            "predictive_covariance_floor":   float(getattr(cfg, "predictive_covariance_floor", 1e-4)),
            "predictive_covariance_floor_mode": getattr(cfg, "predictive_covariance_floor_mode", "prior"),
            "predictive_covariance_ceiling": getattr(cfg, "predictive_covariance_ceiling", None),
            "gauge_group":                   cfg.gauge_group,
            "gauge_tying":                   getattr(cfg, "gauge_tying", None),
            "family":                        cfg.family,
            "group_invariant_families":      list(invariant_families),
            # Truthful fixed-surrogate ledger (C6): these derived booleans expose when the run
            # intentionally freezes a state-dependent quantity rather than following its full
            # joint objective. Defaults are False, preserving the pure path. The covariance is
            # frozen on projected routes (unless predictive_covariance='moment_matched') as well
            # as under skip_belief_sigma_update.
            "fixed_covariance_surrogate":   not covariance_updated,
            "attention_mode":               attention_mode,
            # The transport's runtime exactness certificate: not_applicable for the flat transport (no
            # registered runtime_exactness_key, exact by construction). A registered certificate under
            # gaussian_diagonal is a CONTROLLED APPROXIMATION (the diagonal cone is not closed under GL
            # congruence Omega Sigma Omega^T -- audit C5), so such a run is never reported as exact. Named
            # after the transport, not a retired regime (audit 2026-10-16 F94).
            "transport_runtime_exact":      transport_runtime_exact,
            "transport_runtime_exactness":  transport_runtime_exactness,
            "spd_retraction_route":         spd_retraction_route,
            "spd_retraction_exact":         spd_retraction_exact,
            "spd_retraction_trust_projection_active": spd_retraction_trust_projection_active,
            "spd_retraction_exponential_clipping_active": spd_retraction_exponential_clipping_active,
            "norm_type_block":              block_norm_name,
            "layernorm_affine":             bool(getattr(cfg, "layernorm_affine", False)),
            "m_phi_update_mode":            getattr(cfg, "m_phi_update_mode", "adamw"),
            "m_phi_group_trust_radius":      float(getattr(cfg, "m_phi_group_trust_radius", 0.1)),
            "m_phi_group_retraction":        getattr(cfg, "m_phi_group_retraction", "chart"),
            "m_phi_fisher_rotation_weight":  float(getattr(cfg, "m_phi_fisher_rotation_weight", 2.0)),
            # fisher_group's completed metric (audit 2026-10-16 F50); a record without it ran "rotation".
            "m_phi_fisher_completion":       getattr(cfg, "m_phi_fisher_completion", "rotation"),
            "phi_mstep_max_matrix_norm":     getattr(cfg, "phi_mstep_max_matrix_norm", None),
            "transport_chart_max_norm":      transport_chart_max_norm,
            "frame_pair_escalation_tol":     getattr(cfg, "frame_pair_escalation_tol", None),
            "projected_factorization":       getattr(cfg, "projected_factorization", "gram"),
            # Covariance class of the ACTIVE transport (audit C7), owned by its complete registry
            # record. An unregistered mode fails closed instead of inventing report metadata.
            "transport_covariance_class":   transport_registration.covariance_class,
        },
        "converged_stress": {k: _last(k) for k in (
            "guard_sigma_floor_frac", "guard_sigma_ceil_frac", "nonfinite_frac",
            "gauge_invariant_spread")},
    }


# A metrics.csv cell as the figure pass sees it: a number, or text such as row_kind or relational_semantics.
HistoryValue = Union[float, int, str]


class FigureArtifacts(Protocol):
    r"""What :func:`_save_figures` reads from its artifacts: a :class:`RunArtifacts` during training, or the
    ``SimpleNamespace(run_dir, cfg, history)`` the figure worker rebuilds from ``metrics.csv`` (whose ``cfg``
    can itself be a namespace of raw saved values)."""

    run_dir: Path
    cfg:     object
    history: List[Dict[str, HistoryValue]]


def _finite_cell(value: object) -> bool:
    r"""Whether a history cell is a finite real number (not a bool, a text cell or a blank NaN)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _save_figures(
    artifacts: FigureArtifacts,
    losses:    Optional[List[float]],
    logger:    logging.Logger,

    *,
    losses_start_step: int = 0,
) -> None:
    r"""Best-effort publication figures from the logged history (no model re-run).

    ``losses_start_step`` is the absolute number of optimizer steps completed before ``losses[0]``
    (nonzero on a resumed run, whose list holds only the continuation), so ``loss_curve.png`` shares
    the absolute ``step`` axis of every history trajectory.
    """
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

            def __init__(self, module: object = raw_figs) -> None:
                self._module = module

            def __getattr__(self, name: str) -> object:
                if name == "plt":
                    return _SafePyplot()
                value = getattr(self._module, name)
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
                        if isinstance(exc, raw_figs.FigureSkipped):
                            logger.info("figure %s skipped (%s)", name, exc)
                            return None
                        logger.warning("figure %s failed (%s); remaining figures continue", name, exc)
                        return None

                return _isolated

        figs = _IsolatedFigures()
        figs.set_publication_style()
        run = artifacts.run_dir
        try:
            route_skips = raw_figs.route_figure_skip_reasons(getattr(artifacts, "cfg", None))
        except Exception as exc:                               # no gates beats no figures
            logger.warning("route figure gates unavailable (%s); emitting every figure", exc)
            route_skips = {}
        for name, reason in route_skips.items():            # route-trivial: fixed by the config, not news
            logger.debug("figure %s skipped (%s)", name, reason)

        # Divergence-spike rows are forced between the log and eval rows (vfe3.metrics_schema): every
        # trend below reads the regular rows only, and the gradient-norm, optimizer-geometry and numerics
        # figures overlay the spike rows as markers (audit 2026-10-17 item 5).
        history, spike_rows = split_spike_rows(artifacts.history, cfg=getattr(artifacts, "cfg", None))

        def _aligned(key: str) -> Tuple[List[float], List[float]]:
            r"""Aligned (step, value) for a history column over the regular rows, dropping rows where it is
            not a finite number (pre-first-eval NaN, a blank or a text cell)."""
            xs: List[float] = []
            ys: List[float] = []
            for i, r in enumerate(history):
                if _finite_cell(r.get(key)):
                    xs.append(float(r.get("step", i)))
                    ys.append(float(r[key]))
            return xs, ys

        def _subset(rows: Sequence[Mapping[str, HistoryValue]], keys: Sequence[str]) -> Optional[Dict]:
            r"""A ``{step, key: [...]}`` dict over the ``keys`` finite on at least one of ``rows``, each a
            full-length column with NaN where missing so an eval-cadence key keeps its step alignment and
            the dashboard masks it per series. None when no key is present (caller skips)."""
            present = [k for k in keys if any(_finite_cell(r.get(k)) for r in rows)]
            if not present:
                return None
            cols: Dict = {"step": [r.get("step", i) for i, r in enumerate(rows)]}
            for k in present:
                cols[k] = [float(r[k]) if _finite_cell(r.get(k)) else float("nan") for r in rows]
            return cols

        def _hist_subset(keys: tuple) -> Optional[Dict]:
            r""":func:`_subset` over the regular (non-spike) history rows."""
            return _subset(history, keys)

        def _spike_subset(keys: tuple) -> Optional[Dict]:
            r""":func:`_subset` over the divergence-spike rows, for the marker overlays."""
            return _subset(spike_rows, keys) if spike_rows else None

        readout_keys = tuple({key for row in history for key in row
                              if key.startswith("relational_")})
        if readout_keys:
            # The canonical-F panels read the diagnose_canonical_free_energy columns (NaN otherwise, so
            # _hist_subset drops them), and their "F at the layer input" series is the per-layer figure's
            # "total", which metrics.csv writes as inner_alignment_energy_total (audit 2026-10-16 F82).
            readout_history = _hist_subset(readout_keys + (
                "attn_entropy", "attn_entropy_min", "effective_rank", "belief_cond_median",
                "belief_cond_median_layer_max",
                "self_coupling", "self_divergence", "belief_coupling", "attention_entropy",
                "inner_alignment_energy_total", "canonical_total_output",
                *(key for key, _ in raw_figs._CANONICAL_ROUTER_SERIES)))
            if readout_history is not None and "inner_alignment_energy_total" in readout_history:
                readout_history["total"] = readout_history.pop("inner_alignment_energy_total")
            if readout_history is not None:
                try:
                    from vfe3.metrics import covariance_frozen_at_encode
                    frozen = bool(covariance_frozen_at_encode(artifacts.cfg))
                except Exception:                              # config-free artifacts: assume live Sigma
                    frozen = False
                from types import SimpleNamespace

                from vfe3.train import _residual_update_is_additive
                fig = figs.plot_readout_diagnostics(
                    readout_history,
                    residual_ratio=raw_figs.read_long_table(run / "diagnostics" / "residual_ratio.csv"),
                    covariance_frozen=frozen,
                    residual_additive=_residual_update_is_additive(SimpleNamespace(cfg=artifacts.cfg)),
                    path=str(run / "readout_diagnostics.png"))
                figs.plt.close(fig)

        if losses:
            # losses is one entry per optimizer step from losses_start_step on, so losses[k] belongs to
            # the 1-based absolute step losses_start_step + k + 1 (metrics.csv's step column).
            n = len(losses)
            first = int(losses_start_step)
            steps_per_epoch = next(
                (
                    int(r["steps_per_epoch"])
                    for r in history
                    if isinstance(r.get("steps_per_epoch"), (int, float))
                    and math.isfinite(float(r["steps_per_epoch"]))
                    and int(r["steps_per_epoch"]) > 0
                ),
                0,
            )
            epoch_boundaries = (                               # absolute multiples inside (first, first + n]
                list(range(steps_per_epoch * (first // steps_per_epoch + 1), first + n + 1,
                           steps_per_epoch))
                if steps_per_epoch
                else None
            )
            # The series is the optimizer objective (train_loss): CE plus any active second-order decode,
            # self-coupling, CG-energy or EM term, so it is labelled as the loss rather than as CE.
            fig = figs.plot_trajectory(
                losses, list(range(first + 1, first + n + 1)), ylabel="train loss (nats/token)",
                title="Training loss", color=figs._CB[0],
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
        # Optimization and endpoint trends (history-only; no model re-run): the pre-clip gradient
        # norm, belief-covariance conditioning, and the per-eval E-step endpoint delta.
        nx, ny = _aligned("grad_norm")
        if ny:
            spike_norm = _spike_subset(("grad_norm",))
            fig = figs.plot_trajectory(
                ny, nx, ylabel=r"$\|\nabla\|_2$ (pre-clip)", title="Gradient norm",
                color=figs._CB[5 % len(figs._CB)], logy=True, smooth=max(5, len(ny) // 80),
                markers=((spike_norm["step"], spike_norm["grad_norm"]) if spike_norm else None),
                path=str(run / "grad_norm.png"))
            figs.plt.close(fig)
        # M-step per-role gradient-norm decomposition (mu / sigma / phi): the parameter-learning
        # channels the aggregate grad_norm.png folds together. Columns logged by train_step (aggregated
        # by each optimizer group's "role" tag, so the live tables are attributed correctly under any
        # config); present only on a run that captured step_metrics, so gate on their presence (the CSV
        # stays rectangular per run). A SEPARATE figure from the aggregate, same pre-clip magnitudes.
        # Per-named-group columns when logged (NaN-padded per series), else the legacy role columns.
        hist_gd = _hist_subset(raw_figs.GRAD_NORM_DECOMPOSITION_COLUMNS)
        if hist_gd:
            fig = figs.plot_grad_norm_decomposition(
                hist_gd, spikes=_spike_subset(raw_figs.GRAD_NORM_DECOMPOSITION_COLUMNS),
                path=str(run / "grad_norm_decomposition.png"))
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
                                      for r in history)
                   else ("estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi"))
        eg_present = [k for k in eg_keys
                      if any(_finite_cell(r.get(k)) for r in history)]
        if eg_present and "estep_grad_norm_decomposition" not in route_skips:
            eg_rows = [r for r in history
                       if all(_finite_cell(r.get(k)) for k in eg_present)]
            if eg_rows:
                hist_eg = {"step": [r.get("step", i) for i, r in enumerate(eg_rows)],
                           **{k: [r[k] for r in eg_rows] for k in eg_present}}
                fig = figs.plot_estep_grad_norm_decomposition(hist_eg, path=str(run / "estep_grad_norm_decomposition.png"))
                figs.plt.close(fig)
        cx, cy = _aligned("belief_cond_median")
        if cy and "belief_condition" not in route_skips:
            fig = figs.plot_trajectory(
                cy, cx, ylabel=r"median $\lambda_{\max}/\lambda_{\min}$",
                title="Belief covariance conditioning", color=figs._CB[4 % len(figs._CB)],
                logy=True, smooth=max(5, len(cy) // 80),
                path=str(run / "belief_condition.png"))
            figs.plt.close(fig)
        # Free-energy figures: the per-token budget decomposition (snapshot + early/mid/late evolution)
        # and, separately, the final-iterate objective/CE relationship over training. Both need every plotted term
        # finite, so rows before the first eval (NaN val_*) are dropped.
        fe_terms = ("self_coupling", "belief_coupling", "attention_entropy")

        def _finite_rows(keys: tuple) -> List[Dict]:
            r"""History rows on which every column of ``keys`` is a finite number."""
            return [r for r in history if all(_finite_cell(r.get(k)) for k in keys)]

        def _canonical_rows(rows: List[Dict]) -> bool:
            # On the projected routes the F columns are finite only under diagnose_canonical_free_energy, which
            # writes canonical_beta_tv beside them (relational_* columns mark the route): the canonical F at the
            # captured beliefs, which nothing there optimizes. Its figures carry no E-step or co-descent reading
            # and plot the self-divergence apart from R(alpha) (audit 2026-10-10 round 2, #17).
            return any("canonical_beta_tv" in r or "relational_legacy_global_f_available" in r for r in rows)

        fe_rows = _finite_rows(fe_terms + ("val_ce",))
        held_out_rows = _finite_rows(tuple(f"val_{k}" for k in fe_terms) + ("val_ce",))
        # The canonical figures correlate F with val_ce, so they read the held-out val_* F columns of the same
        # evaluation rows when the run logged them (the validation split val_ce is scored on), and otherwise the
        # log step's training-batch columns, naming the population either way (audit 2026-10-16 F83).
        if held_out_rows and _canonical_rows(held_out_rows):
            population, rows, prefix = "held_out", held_out_rows, "val_"
        elif fe_rows and _canonical_rows(fe_rows):
            population, rows, prefix = "training_batch", fe_rows, ""
        else:
            population, rows, prefix = None, fe_rows, ""
        if rows:
            cfg = getattr(artifacts, "cfg", None)
            hist = {"step": [r.get("step", i) for i, r in enumerate(rows)],
                    **{k: [r[prefix + k] for r in rows] for k in fe_terms},
                    "val_ce": [r["val_ce"] for r in rows]}
            # Scale the coupling terms by the static config lambda_beta scalar.
            lam = getattr(cfg, "lambda_beta", 1.0)
            if population is not None:
                for key in ("self_divergence",) + tuple(k for k, _ in raw_figs._CANONICAL_ROUTER_SERIES):
                    column = [float(r[prefix + key]) if isinstance(r.get(prefix + key), (int, float))
                              else float("nan") for r in rows]
                    if any(math.isfinite(v) for v in column):
                        hist[key] = column
                fig = figs.plot_canonical_free_energy_decomposition(
                    hist, lambda_beta=lam, population=population,
                    path=str(run / "canonical_free_energy_decomposition.png"))
                figs.plt.close(fig)
                fig = figs.plot_canonical_free_energy_relationship(
                    hist, lambda_beta=lam, population=population,
                    path=str(run / "canonical_free_energy_relationship.png"))
                figs.plt.close(fig)
            else:
                fig = figs.plot_free_energy_decomposition(
                    hist, lambda_beta=lam, path=str(run / "free_energy_decomposition.png"))
                figs.plt.close(fig)
                fig = figs.plot_free_energy_relationship(
                    hist, lambda_beta=lam, path=str(run / "free_energy_relationship.png"))
                figs.plt.close(fig)
        # Geometry / SPD / Fisher health dashboard (history-only): the gauge, belief-spectrum, guard,
        # and numerical-safety scalars diagnostics() logs to metrics.csv but that no standard figure
        # surfaced. The panels self-gate, so a run missing a column simply drops that panel.
        hist_geom = _hist_subset(raw_figs.GEOMETRY_HEALTH_COLUMNS)
        if hist_geom:
            fig = figs.plot_geometry_health(
                hist_geom,
                family=getattr(getattr(artifacts, "cfg", None), "family", None),
                path=str(run / "geometry_health.png"),
            )
            figs.plt.close(fig)
        # Held-out validation-sanity dashboard: the per-eval probes (_val_diagnostics) that were CSV-only
        # -- generalization gap, positional loss, causal/attention sanity, and the held-out geometry.
        hist_val = _hist_subset(raw_figs.VALIDATION_SANITY_COLUMNS)
        if hist_val:
            fig = figs.plot_validation_sanity(
                hist_val,
                family=getattr(getattr(artifacts, "cfg", None), "family", None),
                path=str(run / "validation_sanity.png"),
            )
            figs.plt.close(fig)
        # Pullback-group optimizer dashboard: ridge-direction reshaping, certified conditioning,
        # accepted factor scale/chart health, per-role weight norms, and gradient-to-weight ratio.
        hist_opt = _hist_subset(raw_figs.OPTIMIZER_GEOMETRY_COLUMNS)
        if hist_opt:
            fig = figs.plot_optimizer_geometry(
                hist_opt, spikes=_spike_subset(raw_figs.OPTIMIZER_GEOMETRY_COLUMNS),
                path=str(run / "optimizer_geometry.png"))
            figs.plt.close(fig)
        # Numerics counters, the decode variance margin and the per-layer block variances (audit 2026-10-17
        # item 64). Each is drawn only when its inputs exist, so older runs skip both without a warning.
        from vfe3.viz import diagnostic_figures as raw_diag
        diag = _IsolatedFigures(raw_diag)
        hist_num = _hist_subset(raw_diag.NUMERICS_PANEL_COLUMNS)
        if hist_num:
            fig = diag.plot_numerics_history(
                hist_num, spikes=_spike_subset(raw_diag.NUMERICS_PANEL_COLUMNS),
                path=str(run / "numerics_history.png"))
            figs.plt.close(fig)
        block_variance = raw_diag.read_block_variance(run / "diagnostics" / "block_variance.csv")
        if block_variance is not None:
            fig = diag.plot_block_variance_history(block_variance, path=str(run / "block_variance_history.png"))
            figs.plt.close(fig)
    except Exception as exc:                                    # never let a plot kill a finished run
        logger.warning("figure generation failed (%s); numeric results are still saved", exc)
