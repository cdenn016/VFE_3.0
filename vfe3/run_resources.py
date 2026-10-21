r"""Parameter, optimizer-state and throughput accounting of a vfe3 run, by parameter role.

:func:`parameter_breakdown` assigns every trainable parameter to one of four sections by its NAME:

* ``per_type_tables``: the vocabulary-indexed tables of the prior bank (``mu_embed``,
  ``sigma_log_embed``, ``phi_embed`` and the decode tables ``output_proj_weight``,
  ``output_proj_bias``, ``decode_mu_embed``, ``decode_sigma_log_embed``, and any other prior-bank
  parameter whose leading axis is the vocabulary), and the per-type block-MLP hidden bias
  ``per_type_hidden_bias`` when present;
* ``positional_frames``: the positional frame generators ``pos_phi_free``;
* ``context_processing``: the relational comparison maps ``raw_score_q``/``raw_score_k`` and the
  minimal-agent channel noise ``channel_log_noise`` (``relational_score``), the content maps ``T``/``O`` and the
  ``edge_*`` factor maps of
  ``content_link_edges`` (``relational_content``), the block MLPs
  (``block_mlp``), the block and final normalizations (``norm``), and the opt-in CG
  coupling when present;
* ``other``: everything else (for example the scalar ``decode_log_scale``).

A conventional-transformer baseline (``cfg.model_family == 'conventional_transformer'``) is
classified by :func:`vfe3.baseline.baseline_parameter_role` into the same sections and role names.

Parameters frozen by design (``requires_grad=False``: the token and positional frames under
``gauge_transport='off'`` or ``'frozen'``, and ``decode_log_scale`` unless a historical run record
carries ``decode_temperature_learned=True``) are constants of the run, not trained parameters: they are
listed in the separate ``frozen`` section and excluded from the four sections, ``total`` and the
parameters read per token (``total_including_frozen`` adds them back).

Buffers are not parameters: a state dict passed in place of ``named_parameters()`` has its known
buffers (:data:`BUFFER_LEAVES`) removed and listed under ``excluded_buffers``.

Parameters read per token count what one scored token touches: one row of every per-type table the
encoder looks up, every row of the tables the decoder reads (the softmax normalizes over the whole
vocabulary, so a linear decoder reads all of ``W_out`` and ``b_out``), one positional row, and every
context-processing parameter,

.. math::
    P_{\mathrm{read}} = \sum_{t\in\mathrm{enc}} \frac{|t|}{V} + \sum_{t\in\mathrm{dec}} |t|
        + \frac{|Y|}{N_{\max}} + P_{\mathrm{ctx}}.

:func:`optimizer_state_bytes` sizes the optimizer state of a training checkpoint from tensor
metadata only (memory-mapped, no tensor data is read), and :func:`throughput_summary` reads the
logged throughput, memory and wall time of a run directory.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import torch

from vfe3.contracts import VFE_MODEL_FAMILY, model_family_of

if TYPE_CHECKING:
    from vfe3.config import VFE3Config


# Leaf names of persistent buffers that can appear in a model state dict but are not parameters.
BUFFER_LEAVES: Tuple[str, ...] = ("unigram_log_prior",)

# Prior-bank tables the ENCODER reads one row of per token.
ENCODE_TABLE_LEAVES: Tuple[str, ...] = ("mu_embed", "sigma_log_embed", "phi_embed")

# Prior-bank decode tables (vocabulary rows), grouped under per_type_tables["decode"].
DECODE_TABLE_LEAVES: Tuple[str, ...] = (
    "output_proj_weight", "output_proj_bias", "decode_mu_embed", "decode_sigma_log_embed")

SECTIONS: Tuple[str, ...] = ("per_type_tables", "positional_frames", "context_processing", "other")

# Module-wrapper prefixes stripped before a name is classified.
_WRAPPER_PREFIXES: Tuple[str, ...] = ("_orig_mod.", "module.")

# Metrics columns read by throughput_summary.
_THROUGHPUT_COLUMNS: Tuple[str, ...] = (
    "tokens_per_s", "train_step_tokens_per_s", "pipeline_tokens_per_s")
# Allocator and clock telemetry read by throughput_summary (vfe3.train.DEVICE_TELEMETRY_COLUMNS; audit 2026-10-20
# item 20). Absent in older runs, where every summary entry is None.
_TELEMETRY_COLUMNS: Tuple[str, ...] = (
    "cuda_mem_allocated_mb", "cuda_mem_reserved_mb", "cuda_mem_reserved_peak_mb", "cuda_mem_inactive_split_mb",
    "cuda_alloc_retries", "cuda_device_allocs", "cuda_device_frees", "cuda_ooms",
    "gpu_sm_clock_mhz", "gpu_temperature_c", "gpu_power_w")


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def canonical_name(name: str) -> str:
    r"""``name`` without module-wrapper prefixes (:data:`_WRAPPER_PREFIXES`)."""
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name, stripped = name[len(prefix):], True
    return name


def parameter_role(
    name:  str,
    shape: Sequence[int] = (),

    *,
    vocab_size: Optional[int] = None,
    model_family: str = VFE_MODEL_FAMILY,
) -> Tuple[str, str]:
    r"""Return ``(section, role)`` of one parameter from its dotted state-dict ``name``.

    ``shape`` and ``vocab_size`` only matter for a prior-bank parameter outside the known table
    names: it is a per-type table when its leading axis is the vocabulary, else ``other``. The
    per-type block-MLP hidden bias ``per_type_hidden_bias`` (``VFE3Config.per_type_hidden_bias``, a
    :math:`V \times n_{\mathrm{gen}}` table split over layers) is a per-type table. Wrapper
    prefixes (``_orig_mod.`` from ``torch.compile``, ``module.`` from data parallelism) are ignored.
    ``model_family`` other than ``"vfe"`` dispatches to that family's role map
    (:func:`vfe3.baseline.baseline_parameter_role` for the conventional transformer).
    """
    name = canonical_name(name)
    leaf = _leaf(name)
    if model_family != VFE_MODEL_FAMILY:
        from vfe3.baseline import MODEL_FAMILY as _BASELINE_FAMILY, baseline_parameter_role
        if model_family != _BASELINE_FAMILY:
            raise ValueError(f"unknown model_family {model_family!r}")
        return baseline_parameter_role(name)
    if name == "per_type_hidden_bias" or name.startswith("per_type_hidden_bias."):
        # V x n_gen per-type bias on the block-MLP hidden pre-activation (split over layers).
        return "per_type_tables", "per_type_hidden_bias"
    if name.startswith("prior_bank."):
        if leaf in ENCODE_TABLE_LEAVES:
            return "per_type_tables", leaf
        if leaf in DECODE_TABLE_LEAVES:
            return "per_type_tables", "decode"
        if vocab_size is not None and len(shape) >= 1 and int(shape[0]) == int(vocab_size):
            return "per_type_tables", leaf
        return "other", leaf
    if leaf == "pos_phi_free":
        return "positional_frames", "pos_phi"
    if name.startswith("relational_operators"):
        # Mixed head sizes store per-run lists: the attribute is the segment before the run index.
        attribute = name.rsplit(".", 2)[-2] if leaf.isdigit() else leaf
        if attribute in ("raw_score_q", "raw_score_k", "channel_log_noise",   # channel_log_noise: minimal agent
                         "routing_log_temperature", "routing_key_log_noise"):
            return "context_processing", "relational_score"
        if attribute in ("T", "O") or attribute.startswith("edge_"):   # edge_*: content_link_edges factor maps
            return "context_processing", "relational_content"
        return "other", leaf
    if name.startswith("block_mlps."):
        return "context_processing", "block_mlp"
    if name.startswith("block_norm."):
        return "context_processing", "norm"
    if name.startswith("cg_coupling."):
        return "context_processing", "cg_coupling"
    return "other", leaf


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    r"""``key`` of a config object, namespace or serialized mapping (``default`` when absent)."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def frozen_parameter_names(
    cfg:   Any,
    names: Iterable[str],
) -> List[str]:
    r"""The parameters in ``names`` that ``cfg`` freezes by design (``requires_grad=False`` when built).

    For a state dict, whose tensors carry no ``requires_grad``, this reproduces what the built model
    freezes: ``prior_bank.phi_embed`` and ``pos_phi_free`` under ``gauge_transport`` ``'off'`` or
    ``'frozen'``, and ``prior_bank.decode_log_scale`` unless ``cfg`` is a historical record carrying the
    retired ``decode_temperature_learned=True`` (the temperature was trained then). The
    conventional-transformer baseline freezes nothing. ``cfg`` may be a config, a namespace or a
    serialized mapping; returned names are canonical (wrapper prefixes stripped).
    """
    if model_family_of(cfg) != VFE_MODEL_FAMILY:
        return []
    wanted = set()
    if _cfg_value(cfg, "gauge_transport", "on") in ("off", "frozen"):
        wanted |= {"prior_bank.phi_embed", "pos_phi_free"}
    if _cfg_value(cfg, "decode_temperature_learned", False) is not True:
        wanted.add("prior_bank.decode_log_scale")
    return [canonical_name(name) for name in names if canonical_name(name) in wanted]


def _decoder_full_tables(names: Iterable[str], cfg: "VFE3Config") -> List[str]:
    r"""Names of the per-type tables every decoded token reads IN FULL (all V rows)."""
    names = list(names)
    if model_family_of(cfg) != VFE_MODEL_FAMILY:                        # the baseline's linear decoder
        return [n for n in names if _leaf(n) in ("output_proj_weight", "output_proj_bias")]
    if cfg.decode_mode == "linear":
        wanted = ("output_proj_weight", "output_proj_bias")
    elif any(_leaf(n) == "decode_mu_embed" for n in names):
        wanted = ("decode_mu_embed", "decode_sigma_log_embed")          # untied Gaussian decode bank
    else:
        wanted = ("mu_embed", "sigma_log_embed")                        # tied Gaussian decode bank
    return [n for n in names if n.startswith("prior_bank.") and _leaf(n) in wanted]


def parameter_breakdown(
    named_params: Mapping[str, torch.Tensor],
    cfg:          "VFE3Config",

    *,
    frozen:       Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    r"""Counts and shares of the trainable parameters by role, and the parameters read per token.

    ``named_params`` is ``dict(model.named_parameters())`` or a model state dict (known buffers are
    dropped). Only shapes are read, so memory-mapped or meta tensors are fine. A parameter is frozen
    when it is an ``nn.Parameter`` with ``requires_grad=False`` (``named_parameters()`` input) or when
    its name is in ``frozen`` (pass :func:`frozen_parameter_names` for a state dict, whose tensors
    carry no ``requires_grad``). Frozen parameters are listed under ``frozen`` and counted in no other
    section, in neither ``total`` nor the parameters read per token. Returns::

        {"per_type_tables":    {"mu_embed", "sigma_log_embed", "phi_embed", "decode", ..., "total", "share"},
         "positional_frames":  {"pos_phi", "total", "share"},
         "context_processing": {"relational_score", "relational_content", "block_mlp", "norm", ..., "total", "share"},
         "other":              {<leaf>: n, ..., "total", "share"},
         "frozen":             {<name>: n, ..., "total"},
         "total", "total_including_frozen", "parameters_read_per_token", "read_per_token_components",
         "roles_by_name", "excluded_buffers", "definition", "model_family"}
    """
    V = int(cfg.vocab_size)
    family = model_family_of(cfg)
    frozen_names = {canonical_name(name) for name in (frozen or ())}
    sections: Dict[str, Dict[str, int]] = {name: {} for name in SECTIONS}
    frozen_section: Dict[str, int] = {}
    roles_by_name: Dict[str, Dict[str, Any]] = {}
    excluded: List[str] = []
    for name, tensor in named_params.items():
        name = canonical_name(name)
        if _leaf(name) in BUFFER_LEAVES:
            excluded.append(name)
            continue
        shape = tuple(int(s) for s in tensor.shape)
        numel = int(math.prod(shape))
        section, role = parameter_role(name, shape, vocab_size=V, model_family=family)
        is_frozen = name in frozen_names or (
            isinstance(tensor, torch.nn.Parameter) and not tensor.requires_grad)
        if is_frozen:
            frozen_section[name] = frozen_section.get(name, 0) + numel
        else:
            sections[section][role] = sections[section].get(role, 0) + numel
        roles_by_name[name] = {"section": section, "role": role, "numel": numel, "shape": list(shape),
                               "frozen": bool(is_frozen)}
    if "context_processing" in sections:
        for role in ("relational_score", "relational_content", "block_mlp", "norm"):
            sections["context_processing"].setdefault(role, 0)
    default_tables = (("mu_embed", "sigma_log_embed", "phi_embed", "decode") if family == VFE_MODEL_FAMILY
                      else ("token_embed", "decode"))
    for role in default_tables:
        sections["per_type_tables"].setdefault(role, 0)
    sections["positional_frames"].setdefault("pos_phi", 0)

    total = sum(sum(roles.values()) for roles in sections.values())
    out: Dict[str, Any] = {}
    for section, roles in sections.items():
        section_total = sum(roles.values())
        out[section] = {**roles, "total": section_total,
                        "share": (section_total / total) if total else float("nan")}
    out["frozen"] = {**frozen_section, "total": sum(frozen_section.values())}
    out["total"] = total
    out["total_including_frozen"] = total + out["frozen"]["total"]

    # Parameters read per token (trainable parameters only; frozen tables are run constants).
    trainable = {name: entry for name, entry in roles_by_name.items() if not entry["frozen"]}
    decode_full = _decoder_full_tables(trainable, cfg)
    encode_rows = 0
    decode_rows = 0
    for name, entry in trainable.items():
        if entry["section"] != "per_type_tables":
            continue
        if name in decode_full:
            decode_rows += entry["numel"]
        elif _leaf(name) not in DECODE_TABLE_LEAVES:
            encode_rows += entry["numel"] // max(1, entry["shape"][0])     # one row of the table
    positional_row = sum(entry["numel"] // max(1, entry["shape"][0])
                         for entry in trainable.values() if entry["section"] == "positional_frames")
    decode_scalars = 0
    if cfg.decode_mode != "linear":
        decode_scalars = sum(entry["numel"] for name, entry in trainable.items()
                             if _leaf(name) == "decode_log_scale")
    context = out["context_processing"]["total"]
    components = {
        "encode_table_rows":        encode_rows,
        "decode_tables_full":       decode_rows,
        "decode_scalars":           decode_scalars,
        "positional_row":           positional_row,
        "context_processing":       context,
        "decode_tables_read_in_full": decode_full,
    }
    out["parameters_read_per_token"] = encode_rows + decode_rows + decode_scalars + positional_row + context
    out["read_per_token_components"] = components
    out["roles_by_name"] = roles_by_name
    out["excluded_buffers"] = excluded
    out["model_family"] = family
    if family == VFE_MODEL_FAMILY:
        out["definition"] = (
            "per-type tables are the vocabulary-indexed prior-bank tables (token frames, prior means and "
            "log-variances, decoder weights and biases); context processing is the relational comparison "
            "and content maps, block MLPs and normalizations; parameters read per token are one row of "
            "each encoder table, every row of the decoder tables (the softmax normalizes over the "
            "vocabulary), one positional-frame row, and all context-processing parameters; frozen "
            "(requires_grad=False) tables are run constants, listed apart and counted nowhere else")
    else:
        out["definition"] = (
            "conventional transformer: per-type tables are the token table, the per-type read map when "
            "present and the decoder weights and bias; context processing is W_Q, W_K "
            "(relational_score), W_V, W_O (relational_content), the MLPs (block_mlp) and every "
            "normalization (norm); parameters read per token are one row of the token table and of the "
            "read map, every row of the decoder, and all context-processing parameters")
    return out


def exclude_frozen_from_breakdown(
    breakdown: Mapping[str, Any],
    cfg:       Any,
) -> Optional[Dict[str, Any]]:
    r"""Recompute a breakdown recorded before the ``frozen`` section existed, or None when not needed.

    Returns None when ``breakdown`` already has a ``frozen`` section, carries no ``roles_by_name``,
    or ``cfg`` freezes nothing it lists (:func:`frozen_parameter_names`). Otherwise the breakdown is
    rebuilt from the recorded shapes with those parameters frozen.
    """
    if "frozen" in breakdown or not isinstance(breakdown.get("roles_by_name"), Mapping):
        return None
    roles = breakdown["roles_by_name"]
    frozen = frozen_parameter_names(cfg, roles.keys())
    if not frozen:
        return None
    shapes = {name: torch.empty([int(s) for s in entry.get("shape", [])], device="meta")
              for name, entry in roles.items()}
    return parameter_breakdown(shapes, cfg, frozen=frozen)


# ----------------------------------------------------------------------------------------------
# Optimizer state
# ----------------------------------------------------------------------------------------------

def _tensor_bytes(obj: Any) -> int:
    r"""Total bytes of every tensor nested in ``obj`` (metadata only; mmap-safe)."""
    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, Mapping):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def optimizer_state_bytes(
    checkpoint: Union[str, Path, Mapping[str, Any]],
) -> Dict[str, Any]:
    r"""Byte breakdown of the optimizer state in a training checkpoint.

    ``checkpoint`` is a path (loaded with ``mmap=True``; only tensor metadata is read) or an already
    loaded mapping with ``optimizer_state`` (``{"state", "param_groups"}``) and, when present,
    ``optimizer_parameter_manifest`` (parameter id to name). Returns total bytes, bytes by state slot
    (``exp_avg``, ``exp_avg_sq``, ``step``), by optimizer group role tag, and by parameter section and
    role (via :func:`parameter_role`), together with the bytes of the other checkpoint members.
    """
    path: Optional[Path] = None
    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    else:
        bundle = checkpoint
    state = bundle.get("optimizer_state")
    if not isinstance(state, Mapping) or "state" not in state:
        raise ValueError("checkpoint has no optimizer_state with a 'state' entry")
    names: Dict[int, str] = {}
    manifest = bundle.get("optimizer_parameter_manifest")
    if isinstance(manifest, Mapping):
        for group in manifest.get("parameter_groups", []):
            for entry in group:
                names[int(entry["parameter_id"])] = str(entry["name"])
    group_role: Dict[int, str] = {}
    for index, group in enumerate(state.get("param_groups", [])):
        for pid in group.get("params", []):
            group_role[int(pid)] = str(group.get("role", f"group_{index}"))
    cfg_like = bundle.get("config") if isinstance(bundle.get("config"), Mapping) else {}
    vocab_size = cfg_like.get("vocab_size")

    by_slot: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    by_section: Dict[str, int] = {}
    by_role: Dict[str, int] = {}
    total = 0
    for pid, slots in state["state"].items():
        pid = int(pid)
        name = names.get(pid)
        if name is not None:
            section, role = parameter_role(name, vocab_size=vocab_size,
                                           model_family=model_family_of(cfg_like))
        else:
            section, role = "unnamed", f"parameter_{pid}"
        for slot, value in slots.items():
            n_bytes = _tensor_bytes(value)
            total += n_bytes
            by_slot[slot] = by_slot.get(slot, 0) + n_bytes
            by_group[group_role.get(pid, "unassigned")] = by_group.get(group_role.get(pid, "unassigned"), 0) + n_bytes
            by_section[section] = by_section.get(section, 0) + n_bytes
            key = f"{section}.{role}"
            by_role[key] = by_role.get(key, 0) + n_bytes
    members = {key: _tensor_bytes(value) for key, value in bundle.items() if key != "optimizer_state"}
    return {
        "checkpoint":           str(path) if path is not None else None,
        "file_bytes":           (int(os.path.getsize(path)) if path is not None else None),
        "step":                 (int(bundle["step"]) if isinstance(bundle.get("step"), int) else None),
        "optimizer_state_bytes": total,
        "by_slot":              by_slot,
        "by_group_role":        by_group,
        "by_parameter_section": by_section,
        "by_parameter_role":    by_role,
        "n_parameters_with_state": len(state["state"]),
        "other_member_bytes":   {k: v for k, v in members.items() if v},
    }


# ----------------------------------------------------------------------------------------------
# Throughput, memory, wall time
# ----------------------------------------------------------------------------------------------

def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    finite = sorted(v for v in values if isinstance(v, float) and math.isfinite(v))
    if not finite:
        return {"n": 0, "min": None, "q10": None, "median": None, "q90": None, "max": None, "mean": None}

    def q(p: float) -> float:
        position = p * (len(finite) - 1)
        lo, hi = int(math.floor(position)), int(math.ceil(position))
        return finite[lo] + (finite[hi] - finite[lo]) * (position - lo)

    return {"n": len(finite), "min": finite[0], "q10": q(0.1), "median": q(0.5), "q90": q(0.9),
            "max": finite[-1], "mean": sum(finite) / len(finite)}


def _read_metric_columns(path: Path, columns: Sequence[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {c: [] for c in columns}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for c in columns:
                raw = row.get(c)
                try:
                    out[c].append(float(raw) if raw not in (None, "") else float("nan"))
                except ValueError:
                    out[c].append(float("nan"))
    return out


def throughput_summary(run_dir: Union[str, Path]) -> Dict[str, Any]:
    r"""Throughput quantiles (tokens/s), peak memory, wall time and device of a run directory.

    Reads ``metrics.csv`` (``tokens_per_s``, ``train_step_tokens_per_s``, ``pipeline_tokens_per_s``,
    ``peak_mem_mb``, ``train_step_ms_mean``, ``step``; the throughput quantiles leave out divergence-spike
    rows, counted in ``spike_rows_excluded``; and the allocator and clock telemetry of ``_TELEMETRY_COLUMNS``,
    summarized under ``"device_telemetry"`` as the first and last finite value and the quantiles over the
    non-spike rows, so a throughput decline can be set beside allocator retries and the SM clock),
    ``summary.json`` (``wall_time_s``) and
    ``provenance.json`` (``device_name``); a ``resources.json`` written at finalize, when present, is
    returned under ``"resources_json"``. The training-only time estimate is
    :math:`\sum_r \bar t_r \Delta s_r`, the logged mean step time times the steps in each logging
    interval; wall time also includes evaluation and artifact writing.
    """
    run_dir = Path(run_dir)
    out: Dict[str, Any] = {"run_dir": str(run_dir)}
    metrics = run_dir / "metrics.csv"
    if metrics.is_file():
        cols = _read_metric_columns(
            metrics, ("step", "peak_mem_mb", "train_step_ms_mean", *_THROUGHPUT_COLUMNS, *_TELEMETRY_COLUMNS))
        # Quantiles over the log and eval rows only: a divergence-spike row closes a short window around one
        # anomalous step (vfe3.metrics_schema; audit 2026-10-17 item 5). The time estimate below keeps every
        # row, since the rows' windows partition the run.
        from vfe3.metrics_schema import metrics_csv_spike_mask
        spike = metrics_csv_spike_mask(metrics)
        out["throughput_tokens_per_s"] = {
            c: _quantiles([v for v, is_spike in zip(cols[c], spike) if not is_spike])
            for c in _THROUGHPUT_COLUMNS}
        out["spike_rows_excluded"] = int(sum(spike))
        telemetry: Dict[str, Any] = {}
        for c in _TELEMETRY_COLUMNS:
            kept = [v for v, is_spike in zip(cols[c], spike) if not is_spike and math.isfinite(v)]
            telemetry[c] = {"first": kept[0] if kept else None, "last": kept[-1] if kept else None,
                            **_quantiles(kept)}
        out["device_telemetry"] = telemetry
        peak = [v for v in cols["peak_mem_mb"] if math.isfinite(v)]
        out["peak_mem_mb"] = max(peak) if peak else None
        steps = cols["step"]
        ms = cols["train_step_ms_mean"]
        seconds = 0.0
        previous = 0.0
        counted = 0
        for step, step_ms in zip(steps, ms):
            if math.isfinite(step) and math.isfinite(step_ms):
                seconds += step_ms * 1e-3 * max(0.0, step - previous)
                previous = step
                counted += 1
        out["train_only_hours_estimate"] = seconds / 3600.0 if counted else None
    else:
        out["throughput_tokens_per_s"] = None
        out["spike_rows_excluded"] = None
        out["device_telemetry"] = None
        out["peak_mem_mb"] = None
        out["train_only_hours_estimate"] = None
    summary = run_dir / "summary.json"
    wall = None
    if summary.is_file():
        wall = json.loads(summary.read_text(encoding="utf-8")).get("wall_time_s")
    out["wall_time_s"] = wall
    out["wall_time_hours"] = (float(wall) / 3600.0) if isinstance(wall, (int, float)) else None
    provenance = run_dir / "provenance.json"
    out["device_name"] = (json.loads(provenance.read_text(encoding="utf-8")).get("device_name")
                          if provenance.is_file() else None)
    resources = run_dir / "resources.json"
    out["resources_json"] = (json.loads(resources.read_text(encoding="utf-8"))
                             if resources.is_file() else None)
    return out
