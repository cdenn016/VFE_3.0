r"""Which per-type table carries the memorized component? An evaluation-only shrinkage probe (2026-10-21).

The protocol-matched gap :math:`\mathrm{CE}_{\rm val} - \mathrm{CE}_{\rm train\,sample}` of a long multi-epoch run
lies within token-frequency buckets, keyed on rare and mid-frequency input types, so it is stored in the
vocabulary-indexed tables (``docs/research/2026-10-20-ppl-improvement-levers.md``, stage 0). This probe asks which
table. For one table at a time it moves the rows of rarely seen types toward a neutral target,

.. math::
    \theta_v \leftarrow t + \lambda_v(\theta_v - t),\qquad
    \lambda_v = \begin{cases} \mathbf 1[n_v \ge c] & \text{reset} \\ n_v/(n_v + c) & \text{shrink,} \end{cases}

with :math:`n_v` the type's count in the training stream, and scores the run's fixed training sample and held-out
text with the run's own evaluator on the same windows. A table whose rare rows hold memorized training continuations
loses more on the training sample than on held-out text, so the memorization score

.. math::
    M = \Delta\mathrm{CE}_{\rm train\,sample} - \Delta\mathrm{CE}_{\rm validation}

is large and positive for it; a table whose rows hold transferable knowledge raises both alike (:math:`M \approx 0`).
Because validation can recover at most what the training sample loses beyond it, :math:`M` also bounds what
regularizing that table could gain.

Targets: the frame table ``phi_embed`` stores Lie-algebra coordinates, so :math:`t = 0` is the identity frame. Every
other table (``mu_embed``, ``sigma_log_embed``, ``decode_mu_embed``, ``decode_sigma_log_embed``) is moved toward its
occurrence-weighted mean row :math:`t = \sum_v n_v\theta_v / \sum_v n_v`, a typical row of the training text. Each arm
restores the modified rows before the next one, so every arm differs from the loaded checkpoint in exactly one table
(or, in the ``all`` arms, in every listed table). Nothing is trained and no run artifact is modified; the report is a
new JSON file in the run directory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.checkpoint_average import PROTECTED_RUN_FILES, _write_json
from vfe3.cpu_linalg import batched_cpu_lu_guard

# Tables the probe can move, and the target each is moved toward ("zero" = identity frame, "mean" = the
# occurrence-weighted mean row).
TABLE_TARGETS: Dict[str, str] = {
    "phi_embed":              "zero",
    "mu_embed":               "mean",
    "sigma_log_embed":        "mean",
    "decode_mu_embed":        "mean",
    "decode_sigma_log_embed": "mean",
}


@dataclass
class TableShrinkageOptions:
    r"""Options of :func:`run_table_shrinkage` (the click-to-run ``table_shrinkage_probe.py`` passes its ``CONFIG``)."""

    checkpoint:                 str = "final_model.pt"
    tables:                     Sequence[str] = tuple(TABLE_TARGETS)
    thresholds:                 Sequence[int] = (10, 100, 1000)
    mode:                       str = "reset"           # "reset" (n_v < c -> target) | "shrink" (n_v/(n_v + c))
    all_tables_arm:             bool = True             # also move every listed table at once, per threshold
    splits:                     Sequence[str] = ("train_sample", "validation")   # "test" may be added
    max_windows:                Optional[int] = 1000    # per split; None -> every window
    device:                     str = "auto"
    precision:                  str = "run"
    batch_size:                 Optional[int] = None
    allow_stream_mismatch:      bool = False
    drop_unknown_config_fields: bool = True
    cache_dir:                  Optional[str] = None
    out_name:                   str = "table_shrinkage_probe.json"


def row_weights(counts: torch.Tensor, threshold: int, mode: str) -> torch.Tensor:
    r"""Per-type keep factors :math:`\lambda_v` (V,) in float64: an indicator under ``reset``, :math:`n/(n+c)` else."""
    counts = counts.to(torch.float64)
    if mode == "reset":
        return (counts >= float(threshold)).to(torch.float64)
    if mode == "shrink":
        return counts / (counts + float(threshold))
    raise ValueError(f"mode must be 'reset' or 'shrink', got {mode!r}")


def table_target(table: torch.Tensor, name: str, counts: torch.Tensor) -> torch.Tensor:
    r"""The row (n_cols,) that ``name`` is moved toward: zero for frames, else the occurrence-weighted mean row."""
    if TABLE_TARGETS[name] == "zero":
        return torch.zeros(table.shape[1], dtype=torch.float64, device=table.device)
    weights = counts.to(device=table.device, dtype=torch.float64)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("the training stream has no tokens")
    return (weights @ table.detach().to(torch.float64)) / total


class _MovedRows:
    r"""Move the rows of one table toward its target in place, and restore them exactly on exit."""

    def __init__(self, table: torch.nn.Parameter, target: torch.Tensor, keep: torch.Tensor) -> None:
        self.table, self.target, self.keep = table, target, keep

    def __enter__(self) -> int:
        rows = torch.nonzero(self.keep < 1.0).flatten().to(self.table.device)               # (R,) moved rows
        self.rows = rows
        self.saved = self.table.data.index_select(0, rows).clone()                           # (R, n_cols)
        lam = self.keep.to(device=self.table.device)[rows].unsqueeze(1)                       # (R, 1)
        moved = self.target.unsqueeze(0) + lam * (self.saved.to(torch.float64) - self.target.unsqueeze(0))
        self.table.data.index_copy_(0, rows, moved.to(self.table.dtype))
        return int(rows.numel())

    def __exit__(self, *exc: Any) -> None:
        self.table.data.index_copy_(0, self.rows, self.saved)


def run_table_shrinkage(
    run_dir: "str | Path",

    *,
    options: Optional[TableShrinkageOptions] = None,
    log:     Callable[[str], None] = print,
) -> Dict[str, Any]:
    r"""Score the baseline and every (table, threshold) arm on the same windows; write and return the report."""
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc import streams as st

    options = options or TableShrinkageOptions()
    run_dir = Path(run_dir)
    if Path(options.out_name).name != options.out_name or not options.out_name.endswith(".json") \
            or options.out_name in PROTECTED_RUN_FILES:
        raise ValueError(f"out_name must be a new .json file name in the run directory, not one of "
                         f"{sorted(PROTECTED_RUN_FILES)}; got {options.out_name!r}")
    unknown = [name for name in options.tables if name not in TABLE_TARGETS]
    if unknown:
        raise ValueError(f"unknown tables {unknown}; choose from {sorted(TABLE_TARGETS)}")
    if not options.thresholds or any(int(c) < 1 for c in options.thresholds):
        raise ValueError("thresholds must be positive integers")
    bad_splits = [s for s in options.splits if s not in ("train_sample", "validation", "test")]
    if bad_splits or "train_sample" not in options.splits:
        raise ValueError("splits must include 'train_sample' and may add 'validation' and 'test'; "
                         f"got {list(options.splits)}")
    started = time.perf_counter()

    device = ev.resolve_device(options.device)
    loaded = ev.load_model(run_dir, options.checkpoint, device=device, precision=options.precision,
                           drop_unknown_config_fields=options.drop_unknown_config_fields)
    model = loaded.model
    bank = model.prior_bank
    info = st.load_run_info(run_dir, options.checkpoint, drop_unknown_config_fields=options.drop_unknown_config_fields)
    batch_size = int(options.batch_size or info.cfg.batch_size)
    cache_dir = Path(options.cache_dir) if options.cache_dir else None

    tables: Dict[str, torch.nn.Parameter] = {}
    skipped: List[str] = []
    for name in options.tables:
        table = getattr(bank, name, None)
        if isinstance(table, torch.nn.Parameter) and table.dim() == 2 and table.shape[0] == int(info.cfg.vocab_size):
            tables[name] = table
        else:
            skipped.append(name)
    if not tables:
        raise ValueError(f"none of {list(options.tables)} is a (V, n) table of this model")

    train_tokens, train_record = st.resolve_stream(info, "train", cache_dir=cache_dir,
                                                   allow_mismatch=options.allow_stream_mismatch)
    counts = torch.bincount(train_tokens.flatten().to(torch.int64), minlength=int(info.cfg.vocab_size)).cpu()

    datasets: Dict[str, Tuple[Any, np.ndarray]] = {}
    streams: Dict[str, Any] = {}
    val_targets: Optional[int] = None
    for split in ("validation", "test"):
        if split not in options.splits and not (split == "validation" and val_targets is None):
            continue
        tokens, record = st.resolve_stream(info, split, cache_dir=cache_dir,
                                           allow_mismatch=options.allow_stream_mismatch)
        if split == "validation":
            val_targets = int(tokens.numel()) - 1             # the training loop sizes its sample to validation
        if split in options.splits:
            dataset = st.eval_windows(tokens, info.seq_len, info.eval_stride)
            indices = st.evenly_spaced(len(dataset), options.max_windows)
            record["protocol"] = {"n_windows": len(dataset), "n_scored_windows": int(indices.size)}
            datasets[split], streams[split] = (dataset, indices), record
    sample = st.fixed_train_sample(train_tokens, info.seq_len, info.eval_stride, int(val_targets or 0))
    sample_indices = st.evenly_spaced(len(sample), options.max_windows)
    train_record["protocol"] = {"n_windows": len(sample), "n_scored_windows": int(sample_indices.size),
                                "rule": "vfe3.posthoc.streams.fixed_train_sample sized to the validation targets"}
    datasets["train_sample"], streams["train_sample"] = (sample, sample_indices), train_record
    order = [s for s in ("train_sample", "validation", "test") if s in datasets]

    def score(label: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for split in order:
            dataset, indices = datasets[split]
            with batched_cpu_lu_guard():
                out[split] = float(ev.protocol_evaluate(model, dataset, indices, batch_size=batch_size,
                                                        device=device)["ce"])
        log(f"[shrinkage] {label}: " + " | ".join(f"{s} CE {out[s]:.4f}" for s in order))
        return out

    total_tokens = float(counts.sum())
    targets = {name: table_target(table, name, counts) for name, table in tables.items()}
    report: Dict[str, Any] = {
        "run_dir": str(run_dir), "checkpoint": options.checkpoint, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": options.mode, "thresholds": [int(c) for c in options.thresholds], "tables": list(tables),
        "skipped_tables": skipped, "targets": {name: TABLE_TARGETS[name] for name in tables},
        "protocol": {"seq_len": info.seq_len, "eval_stride": info.eval_stride, "batch_size": batch_size,
                     "max_windows": options.max_windows},
        "device": str(device), "precision": options.precision, "notes": list(loaded.notes),
        "streams": streams, "baseline": {}, "arms": [],
        "definition": ("memorization_score = delta_ce[train_sample] - delta_ce[validation], deltas against the "
                       "baseline on the same windows; positive means the moved rows held training-specific fit"),
    }
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), torch.no_grad():
        baseline = score("baseline")
        report["baseline"] = baseline
        for threshold in options.thresholds:
            keep = row_weights(counts, int(threshold), options.mode)                          # (V,) float64
            moved_mass = float(((1.0 - keep) * counts.to(torch.float64)).sum()) / total_tokens
            groups = [(name, [name]) for name in tables]
            if options.all_tables_arm and len(tables) > 1:
                groups.append(("all", list(tables)))
            for label, names in groups:
                contexts = [_MovedRows(tables[n], targets[n], keep) for n in names]
                n_rows = [ctx.__enter__() for ctx in contexts]
                try:
                    ce = score(f"{label} c={threshold}")
                finally:
                    for ctx in reversed(contexts):
                        ctx.__exit__(None, None, None)
                delta = {s: ce[s] - baseline[s] for s in ce}
                report["arms"].append({
                    "table": label, "threshold": int(threshold), "rows_moved": int(n_rows[0]),
                    "train_token_mass_moved": moved_mass, "ce": ce, "delta_ce": delta,
                    "memorization_score": delta["train_sample"] - delta.get("validation", float("nan")),
                })
        restored = score("baseline (restored)")
        report["baseline_restored_matches"] = all(restored[s] == baseline[s] for s in baseline)
    report["seconds"] = time.perf_counter() - started
    out_path = run_dir / options.out_name
    _write_json(out_path, report)
    report["out_path"] = str(out_path)
    log(f"[shrinkage] wrote {out_path}")
    return report


def summary_table(report: Dict[str, Any]) -> str:
    r"""A plain-text table of the arms, sorted by memorization score (largest first)."""
    lines = [f"{'table':<24}{'c':>6}{'rows':>8}{'mass':>8}{'dCE train':>11}{'dCE val':>10}{'M':>9}"]
    for arm in sorted(report["arms"], key=lambda a: -(a["memorization_score"]
                                                      if a["memorization_score"] == a["memorization_score"] else -1e9)):
        d = arm["delta_ce"]
        lines.append(f"{arm['table']:<24}{arm['threshold']:>6}{arm['rows_moved']:>8}"
                     f"{arm['train_token_mass_moved']:>8.4f}{d['train_sample']:>11.4f}"
                     f"{d.get('validation', float('nan')):>10.4f}{arm['memorization_score']:>9.4f}")
    return "\n".join(lines)
