r"""Score checkpoints of several runs under one shared evaluation protocol (2026-10-21).

Each run records perplexity under its own protocol: windows of its ``max_seq_len`` :math:`N`, advanced by its
``eval_stride`` :math:`s`, each window scoring its last :math:`s` targets, so every scored target sees at least
:math:`N - s` tokens of context. Two runs trained at different :math:`(N, s)` therefore report numbers that differ by
the context they were scored with as well as by the model. This module re-scores any number of checkpoints on the
same validation and test windows at one chosen :math:`(N, s)`, with :func:`vfe3.train.evaluate` on the identity-checked
streams (:mod:`vfe3.posthoc.streams`), so the differences that remain are the models'.

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
        "protocol": {"seq_len": N, "eval_stride": s, "min_context": N - s, "max_windows": options.max_windows},
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
    r"""Plain-text table of every entry's CE and PPL per split, with the CE difference to the first entry."""
    results: List[Dict[str, Any]] = report["results"]
    splits = list(results[0]["splits"]) if results else []
    header = f"{'run / checkpoint':<58}" + "".join(f"{sp + ' CE':>15}{'PPL':>9}{'dCE':>9}" for sp in splits)
    lines = [header]
    for entry in results:
        cells = ""
        for sp in splits:
            ce = entry["splits"][sp]["ce"]
            ref = results[0]["splits"][sp]["ce"]
            cells += f"{ce:>15.4f}{entry['splits'][sp]['ppl']:>9.3f}{ce - ref:>9.4f}"
        lines.append(f"{(entry['run'] + ' / ' + entry['checkpoint'])[:57]:<58}{cells}")
    return "\n".join(lines)
