r"""Score checkpoints of several runs under one shared evaluation protocol (2026-10-21).

Each run records perplexity under its own protocol: windows of its ``max_seq_len`` :math:`N`, advanced by its
``eval_stride`` :math:`s`. Every window after a split's first scores its last :math:`s` targets, each with at least
:math:`N - s + 1` tokens of context; the first window scores all :math:`N` of its targets, the first :math:`N - s` of
them with fewer (from one token up). Two runs trained at different :math:`(N, s)` therefore report numbers that differ
by the context they were scored with as well as by the model. This module re-scores any number of checkpoints on the
same validation and test windows at one chosen :math:`(N, s)`, with :func:`vfe3.train.evaluate` on the identity-checked
streams (:mod:`vfe3.posthoc.streams`), so the differences that remain are the models', provided the entries scored the
same targets of the same streams: each entry records its streams (dataset, content SHA-256, token count), its scored
targets and its realized precision per split, and :func:`comparison_table` omits the difference of an entry whose
streams or scored targets differ from the first entry's.

Every protocol scores each target of a split exactly once (the padded final window covers the tail), so the target
set is the same for every :math:`(N, s)`; only the context changes. Shorter windows than a model was trained at are
always admissible here; longer ones are refused for a learned positional frame table, which has one row per trained
position. Nothing is trained and no run artifact is modified; the report is one new JSON file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from vfe3.checkpoint_average import _write_json
from vfe3.cpu_linalg import batched_cpu_lu_guard


@dataclass
class ProtocolRescoreOptions:
    r"""Options of :func:`run_protocol_rescore` (the click-to-run ``rescore_protocol.py`` passes its ``CONFIG``)."""

    entries:                    Sequence[Tuple[str, str]] = field(default_factory=tuple)   # (run_dir, checkpoint)
    seq_len:                    int = 128
    eval_stride:                int = 32
    splits:                     Sequence[str] = ("validation", "test")
    device:                     str = "auto"
    precision:                  str = "run"
    batch_size:                 Optional[int] = None    # None -> each run's cfg.batch_size
    max_windows:                Optional[int] = None    # None -> full splits; an int -> evenly spaced subset
    allow_stream_mismatch:      bool = False
    drop_unknown_config_fields: bool = True
    cache_dir:                  Optional[str] = None


def check_protocol(seq_len: int, eval_stride: int) -> None:
    r"""Raise ``ValueError`` unless :math:`1 \le s \le N`."""
    if int(seq_len) < 2 or int(eval_stride) < 1 or int(eval_stride) > int(seq_len):
        raise ValueError(f"need seq_len >= 2 and 1 <= eval_stride <= seq_len, got N={seq_len}, s={eval_stride}")


# Fields of an entry's split record that must agree with the first entry's for their CE difference to compare models.
COMPARABLE_FIELDS: Tuple[str, ...] = ("dataset", "content_sha256", "n_tokens", "scored_targets")
# Stream fields of vfe3.posthoc.streams.resolve_stream copied into each entry's split record.
_STREAM_FIELDS: Tuple[str, ...] = ("dataset", "format", "file_sha256", "content_sha256", "n_tokens",
                                   "tokenizer_provenance_status")


def protocol_context(seq_len: int, eval_stride: int) -> Dict[str, int]:
    r"""The context the padded protocol windows give their scored targets.

    Window :math:`i \ge 1` scores the targets at window positions :math:`N - s, \dots, N - 1`, whose contexts are
    :math:`N - s + 1, \dots, N` tokens, so every target scored after the first window has at least
    :math:`N - s + 1`. Window 0 scores all :math:`N` of its targets, with contexts :math:`1, \dots, N`, so
    :math:`N - s` of them fall below that minimum.
    """
    N, s = int(seq_len), int(eval_stride)
    return {"min_context_after_first_window": N - s + 1, "first_window_targets_below_min_context": N - s}


def run_protocol_rescore(
    out_path: "str | Path",

    *,
    options: ProtocolRescoreOptions,
    log:     Callable[[str], None] = print,
) -> Dict[str, Any]:
    r"""Score every ``(run_dir, checkpoint)`` entry at ``(seq_len, eval_stride)``; write and return the report."""
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc import streams as st

    check_protocol(options.seq_len, options.eval_stride)
    if not options.entries:
        raise ValueError("no (run_dir, checkpoint) entries to score")
    bad = [s for s in options.splits if s not in ("validation", "test")]
    if bad or not options.splits:
        raise ValueError(f"splits must be drawn from ('validation', 'test'); got {list(options.splits)}")
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; choose a new out_name")
    started = time.perf_counter()
    device = ev.resolve_device(options.device)
    cache_dir = Path(options.cache_dir) if options.cache_dir else None
    N, s = int(options.seq_len), int(options.eval_stride)
    report: Dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {"seq_len": N, "eval_stride": s, **protocol_context(N, s), "max_windows": options.max_windows},
        "device": str(device), "precision": options.precision, "results": [],
    }
    for run_dir, checkpoint in options.entries:
        run_dir = Path(run_dir)
        info = st.load_run_info(run_dir, checkpoint, drop_unknown_config_fields=options.drop_unknown_config_fields)
        trained_N = int(info.cfg.max_seq_len)
        if N > trained_N and getattr(info.cfg, "pos_phi", "none") == "learned":
            raise ValueError(f"{run_dir.name}: seq_len {N} exceeds the {trained_N} positions of its learned "
                             f"positional frame table")
        loaded = ev.load_model(run_dir, checkpoint, device=device, precision=options.precision,
                               drop_unknown_config_fields=options.drop_unknown_config_fields)
        batch_size = int(options.batch_size or info.cfg.batch_size)
        entry: Dict[str, Any] = {
            "run_dir": str(run_dir), "run": run_dir.name, "checkpoint": checkpoint,
            "trained_protocol": {"seq_len": trained_N, "eval_stride": int(info.eval_stride)},
            "batch_size": batch_size, "notes": list(loaded.notes), "splits": {},
            # The arithmetic this entry was scored in: the autocast dtype applied, and the process fp32 matmul
            # policy the model constructor pinned.
            "realized_amp_dtype": loaded.amp_dtype,
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),
            "recorded": {key: info.summary.get(key) for key in (
                "best_val_ppl", "best_step", "final_val_ppl", "test_ppl", "test_ce", "test_best_ppl",
                "test_best_step") if key in info.summary},
        }
        for split in options.splits:
            tokens, record = st.resolve_stream(info, split, cache_dir=cache_dir,
                                               allow_mismatch=options.allow_stream_mismatch)
            dataset = st.eval_windows(tokens, N, s)
            indices = st.evenly_spaced(len(dataset), options.max_windows)
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), batched_cpu_lu_guard():
                result = ev.protocol_evaluate(loaded.model, dataset, indices, batch_size=batch_size, device=device)
            entry["splits"][split] = {
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in result.items()},
                "n_windows": len(dataset), "n_scored_windows": int(indices.size),
                "stream_status": record.get("status"),
                **{key: record.get(key) for key in _STREAM_FIELDS},
            }
            log(f"[rescore] {run_dir.name}/{checkpoint} {split} at N={N}, stride={s}: "
                f"CE {result['ce']:.4f} | PPL {result['ppl']:.3f}")
        report["results"].append(entry)
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["seconds"] = time.perf_counter() - started
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, report)
    report["out_path"] = str(out_path)
    log(f"[rescore] wrote {out_path}")
    return report


def comparison_table(report: Dict[str, Any]) -> str:
    r"""Plain-text table of every entry's CE and PPL per split, with the CE difference to the first entry.

    The difference is printed as ``n/a`` when any of :data:`COMPARABLE_FIELDS` (the stream's dataset, content
    SHA-256 and token count, and the scored targets) differs from the first entry's for that split; a note under
    the table names the fields.
    """
    results: List[Dict[str, Any]] = report["results"]
    splits = list(results[0]["splits"]) if results else []
    header = f"{'run / checkpoint':<58}" + "".join(f"{sp + ' CE':>15}{'PPL':>9}{'dCE':>9}" for sp in splits)
    lines = [header]
    notes: List[str] = []
    for entry in results:
        cells = ""
        name = entry["run"] + " / " + entry["checkpoint"]
        for sp in splits:
            ce = entry["splits"][sp]["ce"]
            ref = results[0]["splits"][sp]["ce"]
            differing = [key for key in COMPARABLE_FIELDS
                         if entry["splits"][sp].get(key) != results[0]["splits"][sp].get(key)]
            delta = f"{'n/a':>9}" if differing else f"{ce - ref:>9.4f}"
            cells += f"{ce:>15.4f}{entry['splits'][sp]['ppl']:>9.3f}{delta}"
            if differing:
                notes.append(f"  {name} {sp}: dCE omitted, {', '.join(differing)} differ from the first entry")
        lines.append(f"{name[:57]:<58}{cells}")
    return "\n".join(lines + notes)
