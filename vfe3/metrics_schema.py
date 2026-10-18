r"""The ``metrics.csv`` schema: row kinds, the column registry readers rely on, and the cell rule.

Three things live here so that writers and readers share one definition (audit 2026-10-17, items 5
and 31).

Row kinds. ``metrics.csv`` carries three kinds of row: ``log`` rows at every ``log_interval`` step,
``eval`` rows at every ``eval_interval`` step, and ``spike`` rows that a reported divergence spike
forces between them (``vfe3.train``, ``forced_row``). A spike row carries the pre-clip role norms and
the spike attribution's group norms of one anomalous step, so a reader that treats it as a regular
sample plots the spikes as the trend, withholds across-seed curves at the spike steps and overstates
throughput. Runs written after the ``row_kind`` column was added declare the kind; older runs are
classified by :func:`infer_row_kind` from the step and the intervals of the run's config. Every
reader goes through :func:`row_kind_of` (one row) or :func:`row_kinds` (a table).

Column registry. :data:`METRIC_COLUMNS` declares, for each column a figure, table or analysis reads,
its cadence and dtype (:class:`MetricColumn`). The reader tuples themselves stay beside their
readers; ``tests/test_audit_20261017_l4_contracts.py`` checks every reader tuple against this
registry and reports the written columns no reader uses.

Cell rule. :func:`normalize_metric_cell` is what ``RunArtifacts.log_metrics`` applies to each cell:
one-element tensors and NumPy scalars are converted with ``.item()``, a bool becomes the integer 0 or 1,
``None`` becomes NaN (a blank cell), a string is accepted only in a column the registry does not
declare numeric, and anything else is rejected. Without it a tensor cell was written as
``tensor(0.5000)`` and a bool as ``True``, which every reader then parsed as NaN with no warning.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

# ------------------------------------------------------------------------------------------------
# Row kinds
# ------------------------------------------------------------------------------------------------

ROW_KIND_COLUMN: str = "row_kind"
ROW_KIND_LOG: str = "log"
ROW_KIND_EVAL: str = "eval"
ROW_KIND_SPIKE: str = "spike"
ROW_KINDS: Tuple[str, ...] = (ROW_KIND_LOG, ROW_KIND_EVAL, ROW_KIND_SPIKE)

RowKind = Literal["log", "eval", "spike"]
Intervals = Tuple[Optional[int], Optional[int]]          # (log_interval, eval_interval), None when off


def _positive_int(value: object) -> Optional[int]:
    r"""``value`` as a positive integer, or None when it is absent, falsy, non-integral or not a number."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)                                   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0 or number != int(number):
        return None
    return int(number)


def config_intervals(cfg: object) -> Intervals:
    r"""``(log_interval, eval_interval)`` of a config object, namespace or serialized mapping.

    ``config.json`` stores the config under ``"config"``; both the whole record and the inner mapping
    are accepted. A falsy, absent or malformed interval is None (that cadence wrote no rows).
    """
    if cfg is None:
        return None, None
    if isinstance(cfg, Mapping):
        inner = cfg.get("config")
        source: Mapping[str, object] = inner if isinstance(inner, Mapping) else cfg
        return _positive_int(source.get("log_interval")), _positive_int(source.get("eval_interval"))
    return (_positive_int(getattr(cfg, "log_interval", None)),
            _positive_int(getattr(cfg, "eval_interval", None)))


def run_config_intervals(run_dir: Union[str, Path]) -> Intervals:
    r"""``(log_interval, eval_interval)`` from ``run_dir/config.json``; ``(None, None)`` when unreadable."""
    path = Path(run_dir) / "config.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    return config_intervals(record) if isinstance(record, Mapping) else (None, None)


def _finite_number(value: object) -> Optional[float]:
    r"""A finite float from a number or a numeric string; None for blanks, text and non-finite values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)                                   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _on_cadence(step: float, interval: Optional[int]) -> bool:
    return interval is not None and step == int(step) and int(step) % interval == 0


def infer_row_kind(
    row: Mapping[str, object],

    *,
    log_interval:  Optional[int],
    eval_interval: Optional[int],
) -> RowKind:
    r"""Classify a row that carries no ``row_kind``.

    A row is ``spike`` when it carries ``divergence_spike == 1`` and its step is a multiple of neither
    interval (a spike on a log or eval step reuses that regular row); otherwise it is ``eval`` on a
    multiple of ``eval_interval`` and ``log`` elsewhere. With no ``divergence_spike`` column every row is
    regular, as on runs that predate forced rows.
    """
    step = _finite_number(row.get("step"))
    if step is None:
        return ROW_KIND_LOG
    flagged = _finite_number(row.get("divergence_spike")) == 1.0
    on_log, on_eval = _on_cadence(step, log_interval), _on_cadence(step, eval_interval)
    if flagged and not on_log and not on_eval:
        return ROW_KIND_SPIKE
    return ROW_KIND_EVAL if on_eval else ROW_KIND_LOG


def row_kind_of(
    row: Mapping[str, object],

    *,
    log_interval:  Optional[int],
    eval_interval: Optional[int],
) -> RowKind:
    r"""The declared ``row_kind`` of ``row`` when it is one of :data:`ROW_KINDS`, else the inferred one."""
    declared = row.get(ROW_KIND_COLUMN)
    if isinstance(declared, str) and declared.strip() in ROW_KINDS:
        return declared.strip()                                 # type: ignore[return-value]
    return infer_row_kind(row, log_interval=log_interval, eval_interval=eval_interval)


def _gcd_of_regular_steps(rows: Sequence[Mapping[str, object]]) -> Optional[int]:
    r"""The gcd of the steps of rows not flagged as spikes, the fallback log cadence of a run with no config."""
    steps = []
    for row in rows:
        step = _finite_number(row.get("step"))
        if step is None or step != int(step) or step <= 0:
            continue
        if _finite_number(row.get("divergence_spike")) == 1.0:
            continue
        steps.append(int(step))
    if not steps:
        return None
    return reduce(math.gcd, steps)


def row_kinds(
    rows: Sequence[Mapping[str, object]],

    *,
    cfg:       object                    = None,
    intervals: Optional[Intervals]       = None,
) -> List[RowKind]:
    r"""The kind of every row of a metrics table (declared, else inferred).

    The intervals come from ``intervals`` when given, else from ``cfg`` (a config, namespace or
    ``config.json`` mapping). When neither interval is known and some row lacks a declared kind, the
    log cadence falls back to the gcd of the steps of the rows not flagged ``divergence_spike``: the
    regular rows sit on multiples of the log and eval intervals, so their gcd is a multiple of
    ``gcd(log_interval, eval_interval)`` and a forced row off that lattice is still recognized.
    """
    log_interval, eval_interval = intervals if intervals is not None else config_intervals(cfg)
    needs_inference = any(not (isinstance(row.get(ROW_KIND_COLUMN), str)
                               and str(row.get(ROW_KIND_COLUMN)).strip() in ROW_KINDS) for row in rows)
    if needs_inference and log_interval is None and eval_interval is None:
        log_interval = _gcd_of_regular_steps(rows)
    return [row_kind_of(row, log_interval=log_interval, eval_interval=eval_interval) for row in rows]


def regular_rows(
    rows: Sequence[Mapping[str, object]],

    *,
    cfg:       object              = None,
    intervals: Optional[Intervals] = None,
) -> List[Mapping[str, object]]:
    r"""The rows of ``rows`` that are not ``spike`` rows, in order."""
    kinds = row_kinds(rows, cfg=cfg, intervals=intervals)
    return [row for row, kind in zip(rows, kinds) if kind != ROW_KIND_SPIKE]


def split_spike_rows(
    rows: Sequence[Mapping[str, object]],

    *,
    cfg:       object              = None,
    intervals: Optional[Intervals] = None,
) -> Tuple[List[Mapping[str, object]], List[Mapping[str, object]]]:
    r"""``(regular rows, spike rows)`` of a metrics table, each in file order."""
    kinds = row_kinds(rows, cfg=cfg, intervals=intervals)
    regular = [row for row, kind in zip(rows, kinds) if kind != ROW_KIND_SPIKE]
    spikes = [row for row, kind in zip(rows, kinds) if kind == ROW_KIND_SPIKE]
    return regular, spikes


def metrics_csv_spike_mask(
    path: Union[str, Path],

    *,
    cfg:       object              = None,
    intervals: Optional[Intervals] = None,
) -> List[bool]:
    r"""One flag per data row of the ``metrics.csv`` at ``path``: True on a ``spike`` row.

    The intervals default to the ``config.json`` beside the file when neither ``cfg`` nor
    ``intervals`` is given.
    """
    path = Path(path)
    if intervals is None and cfg is None:
        intervals = run_config_intervals(path.parent)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: row.get(key) for key in ("step", "divergence_spike", ROW_KIND_COLUMN)}
                for row in csv.DictReader(handle)]
    return [kind == ROW_KIND_SPIKE for kind in row_kinds(rows, cfg=cfg, intervals=intervals)]


# ------------------------------------------------------------------------------------------------
# Column registry
# ------------------------------------------------------------------------------------------------

Cadence = Literal["row", "eval", "diagnostic", "cumulative"]
Dtype = Literal["float", "count", "flag", "str"]

CADENCES: Tuple[str, ...] = ("row", "eval", "diagnostic", "cumulative")
DTYPES: Tuple[str, ...] = ("float", "count", "flag", "str")


@dataclass(frozen=True)
class MetricColumn:
    r"""One ``metrics.csv`` column a reader relies on.

    ``cadence`` is ``"row"`` (a value on every log, eval and spike row), ``"eval"`` (finite only on
    eval rows, blank between), ``"diagnostic"`` (from the pre-step diagnostics pass: finite on log and
    eval rows, blank on spike rows, which collect none) or ``"cumulative"`` (a running total since step 0,
    on every row). ``dtype`` is ``"float"``, ``"count"`` (a nonnegative integer stored as a number),
    ``"flag"`` (0 or 1) or ``"str"``. ``toggle`` names the config setting that makes the column finite
    when it is conditional, else None.
    """

    name:    str
    cadence: Cadence
    dtype:   Dtype
    toggle:  Optional[str] = None


_PARAMETER_GROUPS: Tuple[str, ...] = (
    "mu_embed", "sigma_log_embed", "phi_embed", "pos_phi", "decode",
    "relational_score", "relational_content", "block_mlp", "norm", "per_type_table", "other",
)
_ROLES: Tuple[str, ...] = ("mu", "sigma", "phi")


def _declare(
    cadence: Cadence,
    dtype:   Dtype,
    names:   Iterable[str],

    *,
    toggle:  Optional[str] = None,
) -> List[MetricColumn]:
    return [MetricColumn(name, cadence, dtype, toggle) for name in names]


_PULLBACK = "m_phi_update_mode pullback group (optimizer pullback_group)"
_CHART = "phi_mstep_max_matrix_norm set"

_REGISTRY: List[MetricColumn] = [
    # Row structure and the divergence monitor.
    *_declare("row", "count", ("step", "tokens_seen")),
    *_declare("row", "str", (ROW_KIND_COLUMN,)),
    *_declare("row", "flag", ("divergence_spike",)),
    # Training objective and gradient health.
    *_declare("row", "float", ("train_ce", "grad_norm",
                               *(f"grad_norm_{role}" for role in _ROLES))),
    *_declare("row", "float", tuple(f"weight_norm_{role}" for role in _ROLES)),
    *_declare("row", "float", tuple(f"grad_norm_group_{group}" for group in _PARAMETER_GROUPS)),
    *_declare("row", "float", tuple(f"param_norm_group_{group}" for group in _PARAMETER_GROUPS)),
    *_declare("row", "float", tuple(f"update_ratio_group_{group}" for group in _PARAMETER_GROUPS)),
    *_declare("row", "float", ("sigma_log_mean", "sigma_log_sd")),
    *_declare("row", "count", ("steps_per_epoch",)),
    # Resources.
    *_declare("row", "float", ("tokens_per_s", "train_step_tokens_per_s", "pipeline_tokens_per_s",
                               "train_step_ms_mean", "peak_mem_mb")),
    # Validation (eval rows only).
    *_declare("eval", "float", ("val_ce", "val_ppl", "val_bits_per_token", "val_bpc",
                                "generalization_gap", "generalization_gap_fixed", "train_sample_ce",
                                "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
                                "val_future_leakage", "val_row_sum_error", "val_pos_content_r2",
                                "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
                                "val_gauge_invariant_spread", "val_belief_cond_p95", "val_fisher_trace_mean",
                                "val_guard_sigma_floor_frac", "val_guard_sigma_ceil_frac",
                                "val_phi_norm_mean", "val_phi_norm_std",
                                "val_phi_matrix_norm_p95", "val_phi_matrix_norm_p99",
                                "val_phi_matrix_norm_max", "val_vertex_cond_p99")),
    *_declare("eval", "float", ("val_self_coupling", "val_belief_coupling", "val_attention_entropy",
                                "val_self_divergence"),
              toggle="diagnose_canonical_free_energy"),
    *_declare("eval", "count", ("cap_active_count", "cap_entry_count",
                                "factorization_fallback_count", "factorization_count")),
    # Pre-step diagnostics (blank on spike rows).
    *_declare("diagnostic", "float", ("attn_entropy", "self_coupling", "self_divergence", "belief_coupling",
                                      "attention_entropy", "inner_alignment_energy_total", "effective_rank",
                                      "gauge_invariant_spread", "gauge_head_logdet_spread",
                                      "gauge_head_phase_abs_mean", "belief_circularity_mean",
                                      "phi_norm_mean", "phi_norm_std",
                                      "phi_matrix_norm_p95", "phi_matrix_norm_p99", "phi_matrix_norm_max",
                                      "pos_phi_matrix_norm_p95", "pos_phi_matrix_norm_max",
                                      "vertex_cond_median", "vertex_cond_p95", "vertex_cond_p99",
                                      "belief_cond_median", "belief_cond_p95", "belief_cond_max",
                                      "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
                                      "fisher_trace_mean", "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
                                      "nonfinite_frac", "attn_entropy_min", "attn_entropy_collapsed_heads",
                                      "relational_score_nonfinite_frac")),
    *_declare("row", "float", ("phi_chart_projected_fraction", "phi_chart_projection_scale_min",
                               "phi_chart_projection_ms"), toggle=_CHART),
    *_declare("row", "float", ("phi_ridge_direction_cosine_mean", "phi_pullback_damped_gen_cond_median",
                               "phi_pullback_damped_gen_cond_max", "phi_group_trust_scale_mean",
                               "phi_group_trust_scale_min", "phi_group_chart_norm_max",
                               "phi_group_backtracks_mean", "phi_group_backtracks_max"), toggle=_PULLBACK),
    *_declare("row", "count", ("phi_group_active_rows",), toggle=_PULLBACK),
    # Decode margin (NaN under decode_mode = "linear").
    *_declare("row", "float", ("decode_sigma_v_min",)),
    # Cumulative numerics counters.
    *_declare("cumulative", "count", ("frame_projection_count", "frame_projection_total",
                                      "skipped_updates", "attempted_updates",
                                      "layernorm_trace_cap_elements", "covariance_ceiling_elements",
                                      "covariance_escalation_elements", "gemm_fallback_slices",
                                      "source_recovery_elements")),
]

METRIC_COLUMNS: Dict[str, MetricColumn] = {}
for _column in _REGISTRY:
    if _column.name in METRIC_COLUMNS:
        raise RuntimeError(f"metrics column {_column.name!r} is declared twice")
    if _column.cadence not in CADENCES or _column.dtype not in DTYPES:
        raise RuntimeError(f"metrics column {_column.name!r} has an undeclared cadence or dtype")
    METRIC_COLUMNS[_column.name] = _column
del _column


def columns_with_cadence(cadence: Cadence) -> Tuple[str, ...]:
    r"""Registered column names of one cadence, in declaration order."""
    return tuple(name for name, column in METRIC_COLUMNS.items() if column.cadence == cadence)


def unregistered_columns(columns: Iterable[str]) -> List[str]:
    r"""The names in ``columns`` that :data:`METRIC_COLUMNS` does not declare, sorted."""
    return sorted({name for name in columns if name not in METRIC_COLUMNS})


# ------------------------------------------------------------------------------------------------
# Cell rule
# ------------------------------------------------------------------------------------------------

MetricCell = Union[float, int, str]


def normalize_metric_cell(key: str, value: object) -> MetricCell:
    r"""The value ``log_metrics`` writes for one cell, or ``TypeError`` for a cell it cannot write faithfully.

    ``float``/``int``: unchanged. ``bool``: the integer 0 or 1. ``None``: NaN (a blank cell). A
    one-element tensor or NumPy scalar: its ``.item()``, normalized again. ``str``: unchanged unless the
    registry declares ``key`` numeric. Anything else (a multi-element array, a container, an object)
    raises, since its ``str()`` would read back as NaN.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return float("nan")
    if isinstance(value, str):
        declared = METRIC_COLUMNS.get(key)
        if declared is not None and declared.dtype != "str":
            raise TypeError(f"metrics column {key!r} is declared {declared.dtype}; got the string {value!r}")
        return value
    item = getattr(value, "item", None)
    if callable(item):
        numel = getattr(value, "numel", None)
        size = numel() if callable(numel) else getattr(value, "size", 1)
        if size != 1:
            raise TypeError(f"metrics column {key!r} got a {type(value).__name__} of {size} elements")
        return normalize_metric_cell(key, item())
    raise TypeError(f"metrics column {key!r} got an unwritable {type(value).__name__} cell")


def normalize_metric_row(row: Mapping[str, object]) -> Dict[str, MetricCell]:
    r"""``row`` with :func:`normalize_metric_cell` applied to every cell, key order kept."""
    return {key: normalize_metric_cell(key, value) for key, value in row.items()}
