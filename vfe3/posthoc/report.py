r"""Orchestration of the post-hoc analyses of one run: :func:`run_posthoc`.

For the final and best-validation weights and each requested split (validation, test, the fixed
training-split sample) one per-target pass (:func:`vfe3.posthoc.evaluate.nll_pass`) is run under the
run's own evaluation protocol (``cfg.max_seq_len``, ``cfg.eval_stride``, ``cfg.batch_size``) on the
identity-checked streams of :mod:`vfe3.posthoc.streams`. Each pass is cross-checked against
:func:`vfe3.train.evaluate` on the same windows and, on a full split, against the numbers the run
recorded (the reproduction check; fields dropped from a legacy config are recorded beside it). The
final-weight validation pass also feeds the KL-cap and attention observers, and it is the baseline of
the M5 interventions, each of which re-runs the same windows under a reversible patch. A checkpoint
whose config and model state are bitwise identical to the final weights' is not evaluated again: its
passes, residual and covariance entries are the final weights', marked ``aliased_to`` and listed
under ``checkpoint_aliases``, with the reproduction check redone against its own recorded numbers.

The same passes carry the v2:899 residual observer (every checkpoint on ``residual_split``, with the
per-layer self-weights), the final-weight ``routing_split`` pass is handed to the M4 decomposition of
:mod:`vfe3.posthoc.routing` (imported lazily, skipped with the reason when missing or failing), each
loaded checkpoint gets the v2:872 covariance analysis, and the full test passes give the
word-normalized perplexity of ``tab:placeholder_calibration`` after the text audit. Every section that
is not defined for the run's route is recorded with its reason rather than failing the report.

Outputs under ``out_dir`` (default ``<run_dir>/posthoc``): ``posthoc.json``; ``tables/*.md``
(M5, ``tab:relational_run``, ``tab:placeholder_resources``, KL cap, attention statistics, map
invariants, residual ratios, covariance gap and temperature, calibration); ``csv/*.csv`` (and the
per-pair covariance ``.npz``); ``figures/*.png|pdf`` (including CE by context length);
``nll/*.npz`` per-target dumps with article ids.
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import streams as st

# |CE(replay) - CE(recorded)| tolerance (nats) of the reproduction check: the arithmetic of the run's
# own evaluation (its amp dtype on CUDA and its fp32 matmul policy), and a float32 replay of a
# mixed-precision evaluation or of a reduced matmul policy (TF32 under 'high', bf16 under 'medium').
REPRODUCTION_TOL_MATCHED: float = 1e-3
REPRODUCTION_TOL_FP32_REPLAY: float = 5e-3
# Windows re-run after the intervention arms to confirm every patch was undone.
RESTORATION_CHECK_WINDOWS: int = 2
SPLIT_KEYS: Tuple[str, ...] = ("validation", "test", "train_sample")


@dataclass
class PosthocOptions:
    r"""What :func:`run_posthoc` computes. The defaults are the full manuscript set."""

    device:                     str = "auto"              # "auto" -> cuda when available
    precision:                  str = "run"               # "run": the run's amp dtype on CUDA; "fp32"
    checkpoints:                Tuple[str, ...] = ("final_model.pt", "best_model.pt")
    splits:                     Tuple[str, ...] = SPLIT_KEYS
    batch_size:                 Optional[int] = None      # None -> cfg.batch_size (the run's loaders)
    max_windows:                Optional[int] = None      # None -> the full split; n -> evenly spaced subset
    window_indices:             Optional[Sequence[int]] = None   # explicit validation/test windows
    cross_check:                bool = True               # also run vfe3.train.evaluate on the same windows
    interventions:              Optional[Sequence[str]] = None   # None -> rows 1-12; () -> none
    intervention_checkpoint:    str = "final_model.pt"
    intervention_split:         str = "validation"
    klcap_splits:               Tuple[str, ...] = ("validation", "test")   # at the final weights
    attention_split:            Optional[str] = "validation"               # at the final weights
    maps:                       bool = True
    resources:                  bool = True
    latency_batches:            int = 20                  # timed forward batches (0 disables)
    figures:                    bool = True
    save_nll:                   bool = True
    bootstrap_resamples:        int = ev.BOOTSTRAP_RESAMPLES
    allow_stream_mismatch:      bool = False
    drop_unknown_config_fields: bool = False
    cache_dir:                  Optional[str] = None
    residual_split:             Optional[str] = "validation"   # v2:899 per checkpoint; None disables
    covariance:                 bool = True               # v2:872 C_i gap and hidden temperature per checkpoint
    covariance_split:           str = "validation"        # source of the occurrence-weighted (type, position) sample
    calibration:                bool = True               # tab:placeholder_calibration (needs full test passes)
    word_stream_path:           Optional[str] = None      # word-level test text; None -> pinned HF cache, offline
    routing:                    bool = True               # M4 via vfe3.posthoc.routing (lazily imported)
    routing_split:              Optional[str] = "validation"   # at the final weights


def baseline_posthoc_options(
    options: PosthocOptions,
) -> Tuple[PosthocOptions, Dict[str, str]]:
    r"""The preset for a conventional-transformer run: ``options`` with the VFE-only analyses off.

    Kept: the per-target passes, the ``vfe3.train.evaluate`` cross-check, the reproduction check,
    the bootstrap intervals, CE by context, calibration, resources, latency and the ALiBi slopes.
    Turned off, each with the reason returned in the second element (recorded in ``posthoc.json``
    under ``skipped_analyses``): the M5 frame/role interventions, the KL-cap observer, the
    attention-statistics and residual observers (they read the projected-router capture), the
    routing-covariance section, the M4 routing decomposition and the relational map invariants.
    """
    skipped: Dict[str, str] = {}
    changes: Dict[str, Any] = {}
    if options.interventions is None or len(options.interventions) > 0:
        changes["interventions"] = ()
        skipped["interventions"] = "the M5 rows remove frame roles of the VFE predictor"
    if options.klcap_splits:
        changes["klcap_splits"] = ()
        skipped["klcap"] = "the routing-cost cap exists only on the VFE projected router"
    if options.attention_split is not None:
        changes["attention_split"] = None
        skipped["attention"] = "the attention-statistics observer reads the VFE router capture"
    if options.residual_split is not None:
        changes["residual_split"] = None
        skipped["residual"] = "the residual observer reads the VFE belief-stack capture"
    if options.covariance:
        changes["covariance"] = False
        skipped["covariance"] = "there is no routing covariance C_i without frames"
    if options.routing:
        changes["routing"] = False
        skipped["routing"] = "the M4 decomposition splits the VFE routing energy"
    if options.maps:
        changes["maps"] = False
        skipped["maps"] = "the map invariants read the VFE relational operators"
    return (dataclasses.replace(options, **changes) if changes else options), skipped


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Optional[Path]:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    return path


def _stem(checkpoint: str) -> str:
    return checkpoint.rsplit(".", 1)[0].replace("_model", "")


def _checkpoint_step(info: st.RunInfo, checkpoint: str) -> Optional[int]:
    summary = info.summary
    key = "n_steps" if checkpoint.startswith("final") else "best_step" if checkpoint.startswith("best") else None
    step = summary.get(key) if key is not None else None
    return int(step) if step is not None else None


def _latest_training_checkpoint(run_dir: Path) -> Optional[Path]:
    directory = run_dir / "checkpoints"
    if not directory.is_dir():
        return None
    found = []
    for path in directory.glob("step_*.pt"):
        try:
            found.append((int(path.stem.split("_", 1)[1]), path))
        except ValueError:
            continue
    return max(found)[1] if found else None


def _recorded_reference(info: st.RunInfo, checkpoint: str, split: str) -> Optional[Dict[str, Any]]:
    r"""The CE and scored-target count the run recorded for (checkpoint, split), if any.

    The validation count is ``final_val_scored_targets`` in a :func:`finalize_run` summary and
    ``val_scored_targets`` in a :func:`finalize_validation_run` (ablation-cell) summary.
    """
    s = info.summary
    val_scored = s.get("final_val_scored_targets", s.get("val_scored_targets"))
    weights = s.get("evaluation_weights", "final")
    if split == "validation" and checkpoint.startswith("final") and s.get("final_val_ce") is not None:
        return {"source": "summary.json final_val_ce", "ce": float(s["final_val_ce"]),
                "scored_targets": val_scored, "kind": "final evaluation"}
    if split == "test" and s.get("test_ce") is not None and checkpoint.startswith(weights):
        return {"source": "summary.json test_ce", "ce": float(s["test_ce"]),
                "scored_targets": s.get("test_scored_targets"), "kind": "final evaluation"}
    if split == "test" and checkpoint.startswith("best") and s.get("test_best_ce") is not None:
        return {"source": "summary.json test_best_ce", "ce": float(s["test_best_ce"]),
                "scored_targets": s.get("test_scored_targets"), "kind": "final evaluation"}
    if split == "validation" and checkpoint.startswith("best") and s.get("best_val_ppl"):
        return {"source": "summary.json log(best_val_ppl)", "ce": math.log(float(s["best_val_ppl"])),
                "scored_targets": val_scored, "kind": "in-training evaluation"}
    if split == "train_sample":
        key = "train_sample_ce_final" if checkpoint.startswith("final") else "train_sample_ce_best"
        if s.get(key) is not None:
            return {"source": f"summary.json {key}", "ce": float(s[key]), "scored_targets": None,
                    "kind": "in-training evaluation"}
    return None


def _replay_matches_run(loaded: Any, cfg: Any, device: torch.device) -> bool:
    r"""Whether the replay ran the arithmetic of the run's own evaluation (the tight reproduction tolerance).

    Both the autocast dtype and the fp32 matmul policy must be the run's: ``precision='fp32'`` replaces
    a reduced policy (``'high'``, TF32) by ``'highest'`` (:func:`vfe3.posthoc.evaluate.load_model`), and an
    IEEE fp32 replay of a TF32 evaluation is an fp32 replay, not matched arithmetic (audit 2026-10-16
    F68). A run that left the policy unset (``None``, the card's policy as found) matches only a replay
    that also leaves it unset. A mixed-precision run matches only a replay on CUDA in its amp dtype.
    """
    same_amp = loaded.amp_dtype == cfg.amp_dtype and (cfg.amp_dtype is None or device.type == "cuda")
    same_matmul = (getattr(loaded.cfg, "fp32_matmul_precision", None)
                   == getattr(cfg, "fp32_matmul_precision", None))
    return bool(same_amp and same_matmul)


def _reproduction(
    result:    Dict[str, Any],
    reference: Optional[Dict[str, Any]],
    *,
    full:      bool,
    matched:   bool,
) -> Dict[str, Any]:
    if reference is None:
        return {"status": "no_reference"}
    if not full:
        return {"status": "subset", "reference": reference}
    tol = REPRODUCTION_TOL_MATCHED if matched else REPRODUCTION_TOL_FP32_REPLAY
    delta = result["ce"] - reference["ce"]
    targets_ok = (reference.get("scored_targets") is None
                  or int(reference["scored_targets"]) == int(result["n_scored"]))
    return {"status": "ok" if (abs(delta) <= tol and targets_ok) else "mismatch",
            "reference": reference, "delta_ce": delta, "tolerance": tol,
            "arithmetic": "matched" if matched else "fp32 replay", "scored_targets_ok": targets_ok}


def _subsample_reproduction(
    result:    ev.TargetNLL,
    indices:   np.ndarray,
    windows:   np.ndarray,
    reference: Optional[Dict[str, Any]],
    *,
    matched:   bool,
) -> Dict[str, Any]:
    r"""The reproduction check restricted to the windows ``windows`` (the population the run logged).

    The reported pass statistic keeps every window of the larger post-hoc sample; the check compares
    the logged value with the mean NLL of the targets in ``windows`` only, and is ``"subset"`` unless
    every one of those windows was evaluated.
    """
    keep = np.isin(result.window, windows)
    n_scored = int(keep.sum())
    ce = float(result.nll[keep].mean()) if n_scored else float("nan")
    out = _reproduction({"ce": ce, "n_scored": n_scored}, reference,
                        full=bool(np.isin(windows, indices).all()), matched=matched)
    out.update(ce=ce, n_scored=n_scored, n_windows=int(windows.size),
               sample="the training loop's fixed sample (a subset of the windows of this pass)")
    return out


def _identical_model_states(path_a: Path, path_b: Path) -> bool:
    r"""Whether two checkpoint bundles carry the same config and a bitwise-identical ``model_state``.

    Tensors are compared byte for byte (dtype, shape and raw storage), so ``-0.0`` against ``0.0`` or a
    NaN payload counts as a difference and the second checkpoint is evaluated on its own.
    """
    a = torch.load(path_a, map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(path_b, map_location="cpu", weights_only=True, mmap=True)
    state_a, state_b = a.get("model_state"), b.get("model_state")
    if not (isinstance(state_a, Mapping) and isinstance(state_b, Mapping)) or a.get("config") != b.get("config"):
        return False
    if list(state_a) != list(state_b):
        return False
    for key, x in state_a.items():
        y = state_b[key]
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            if x != y:
                return False
        elif (x.dtype != y.dtype or x.shape != y.shape
              or not torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8))):
            return False
    return True


def _checkpoint_aliases(run_dir: Path, options: "PosthocOptions", final_ckpt: str) -> Dict[str, str]:
    r"""``{checkpoint: final_ckpt}`` for each requested checkpoint after ``final_ckpt`` identical to it.

    Such a checkpoint is not evaluated again: its passes and sections are the final weights' (audit
    2026-10-10 #30). The intervention checkpoint is never aliased while the M5 rows are requested,
    since the rows run on the loaded model.
    """
    checkpoints = list(options.checkpoints)
    if final_ckpt not in checkpoints or not (run_dir / final_ckpt).is_file():
        return {}
    interventions = options.interventions is None or len(options.interventions) > 0
    aliases: Dict[str, str] = {}
    for checkpoint in checkpoints[checkpoints.index(final_ckpt) + 1:]:
        if checkpoint == final_ckpt or not (run_dir / checkpoint).is_file():
            continue
        if interventions and checkpoint == options.intervention_checkpoint:
            continue
        if _identical_model_states(run_dir / final_ckpt, run_dir / checkpoint):
            aliases[checkpoint] = final_ckpt
    return aliases


def _alias_checkpoint(
    report:     Dict[str, Any],
    info:       st.RunInfo,
    checkpoint: str,
    target:     str,

    *,
    kept:                 Mapping[str, Tuple[ev.TargetNLL, np.ndarray, bool]],
    matched:              bool,
    reproduction_windows: Mapping[str, np.ndarray],
    test_passes:          Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    r"""Record ``checkpoint``'s passes and sections as ``target``'s (bitwise-identical model state).

    Every pass entry of ``target`` is copied with ``checkpoint``, its step and ``aliased_to`` and its
    reproduction check redone against the references recorded for ``checkpoint``; the residual and
    covariance sections and the test pass handed to the calibration are ``target``'s, relabeled alike.
    ``kept`` holds ``target``'s per-split ``(result, indices, full)``.
    """
    step = _checkpoint_step(info, checkpoint)
    for entry in [p for p in report["passes"] if p["checkpoint"] == target]:
        alias = dict(entry, checkpoint=checkpoint, step=step, aliased_to=target)
        split = entry["split"]
        if entry.get("status") == "ok" and split in kept:
            result, indices, full = kept[split]
            reference = _recorded_reference(info, checkpoint, split)
            if split in reproduction_windows:
                alias["reproduction"] = _subsample_reproduction(
                    result, indices, reproduction_windows[split], reference, matched=matched)
            else:
                alias["reproduction"] = _reproduction(alias, reference, full=full, matched=matched)
        report["passes"].append(alias)
    for section in ("residual", "covariance"):
        if target in report[section]:
            report[section][checkpoint] = dict(report[section][target], checkpoint=checkpoint,
                                               stem=_stem(checkpoint), step=step, aliased_to=target)
    if target in test_passes:
        test_passes[checkpoint] = test_passes[target]
    report["checkpoint_aliases"][checkpoint] = {
        "aliased_to": target,
        "reason": (f"model_state and config are bitwise identical to {target}: its passes, residual and "
                   "covariance sections are reused (reproduction checked against this checkpoint's "
                   "own references), not evaluated again")}


def _units_for(result: ev.TargetNLL, split: str, stream_units: Optional[np.ndarray]) -> np.ndarray:
    r"""Resampling unit of every scored target: its article (validation/test), else its window."""
    if split != "train_sample" and stream_units is not None:
        return stream_units[result.target_pos]
    return result.window.copy()


def _align(
    base: ev.TargetNLL,
    arm:  ev.TargetNLL,
) -> Tuple[np.ndarray, np.ndarray, int]:
    r"""``(base_index, arm_index, n_dropped)`` of the targets both passes scored, paired."""
    if np.array_equal(base.target_pos, arm.target_pos):
        index = np.arange(base.n)
        return index, index, 0
    common, i_base, i_arm = np.intersect1d(base.target_pos, arm.target_pos, return_indices=True)
    return i_base, i_arm, base.n + arm.n - 2 * int(common.size)


def measure_latency(
    model:      torch.nn.Module,
    dataset:    Any,
    batch_size: int,
    device:     torch.device,
    n_batches:  int,
    *,
    n_warmup:   int = 2,
) -> Dict[str, Any]:
    r"""Wall time of the inference forward with logits at the declared batch and context."""
    indices = np.arange(min(len(dataset), batch_size * (n_batches + n_warmup)), dtype=np.int64)
    times: List[float] = []
    n_tokens = 0
    with torch.no_grad():
        for k, (_, inputs, _) in enumerate(ev.iter_batches(dataset, indices, batch_size)):
            tokens = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model.forward_beliefs(tokens, return_logits=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if k >= n_warmup:
                times.append(time.perf_counter() - started)
                n_tokens += int(tokens.numel())
    if not times:
        return {"status": "not_measured"}
    arr = np.asarray(times)
    return {"status": "ok", "batch_size": int(batch_size), "context": int(dataset.seq_len),
            "n_batches": int(arr.size), "seconds_per_batch_median": float(np.median(arr)),
            "seconds_per_batch_mean": float(arr.mean()),
            "tokens_per_s": float(n_tokens / arr.sum()), "device": str(device)}


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "cpu"


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 4) -> str:
    from vfe3.semantics.common import format_value

    return format_value(value, digits)


def interventions_table(rows: Sequence[Mapping[str, Any]], *, n_units: Optional[int], unit: str,
                        n_targets: Optional[int], subset: bool) -> str:
    r"""Markdown of Table M5 (``tab:placeholder_interventions``), rows 1-12.

    An arm that scored a different target set than the baseline (a full-covariance decode can exclude
    degenerate positions per pass) is paired on the targets both passes scored: its CE and
    :math:`\Delta\mathrm{CE}` are over that aligned subset, so :math:`\mathrm{CE}_{\mathrm{arm}} -
    \mathrm{CE}_{\mathrm{base}}` with the baseline row's all-target CE need not equal the printed
    :math:`\Delta\mathrm{CE}`. Such a row's CE carries a dagger, and a note gives its target count and the
    baseline CE on the aligned subset (audit 2026-10-16 F69).
    """
    lines = ["| Frozen intervention on every validation target | CE | $\\Delta$CE | 95% interval |",
             "|---|---|---|---|"]
    aligned: List[str] = []
    for r in sorted(rows, key=lambda r: int(r["row"])):
        if r.get("status") == "ok":
            interval = (f"[{_fmt(r.get('lo'))}, {_fmt(r.get('hi'))}]" if r["row"] != 1
                        else f"[{_fmt(r.get('ce_lo'))}, {_fmt(r.get('ce_hi'))}] (CE)")
            delta = "0" if r["row"] == 1 else _fmt(r.get("delta_ce"))
            dropped = int(r.get("n_targets_dropped") or 0)
            mark = "†" if dropped else ""
            lines.append(f"| {r['label']} | {_fmt(r.get('ce'))}{mark} | {delta} | {interval} |")
            if dropped:
                aligned.append(f"† Row {r['row']}: scored on {r.get('n_targets')} of the baseline's {n_targets} "
                               f"targets ({dropped} dropped: scored by only one of the two passes); its CE and "
                               f"$\\Delta$CE are on these aligned targets, where the baseline CE is "
                               f"{_fmt(r.get('ce_base_aligned'))}.")
        else:
            lines.append(f"| {r['label']} | {r.get('status')} | {r.get('status')} | {r.get('status')} |")
    note = (f"\nCE in nats per scored target over {n_targets} targets; intervals are 2.5/97.5 percentiles of a "
            f"paired {unit} bootstrap over {n_units} {unit}s")
    if subset:
        note += "; SUBSET of the validation split (not the manuscript number)"
    return "\n".join(lines) + note + ".\n" + "".join(f"\n{line}\n" for line in aligned)


def relational_run_table(passes: Sequence[Mapping[str, Any]], info: st.RunInfo) -> str:
    r"""Markdown of ``tab:relational_run`` (recorded rows and the post-hoc rows)."""
    s = info.summary
    lines = ["| Evaluation | Step | Perplexity | CE (nats) | 95% interval (CE) | Source |",
             "|---|---|---|---|---|---|"]
    if s.get("best_val_ppl") is not None:
        lines.append(f"| Best recorded validation | {s.get('best_step')} | {float(s['best_val_ppl']):.4f} | "
                     f"{math.log(float(s['best_val_ppl'])):.6f} | – | recorded |")
    labels = {("final", "validation"): "Final validation", ("final", "test"): "Test, final weights",
              ("best", "test"): "Test, best-validation weights",
              ("best", "validation"): "Validation, best-validation weights (replay)",
              ("final", "train_sample"): "Training-split sample, evaluation protocol",
              ("best", "train_sample"): "Training-split sample, evaluation protocol (best weights)"}
    order = [("best", "validation"), ("final", "validation"), ("final", "test"), ("best", "test"),
             ("final", "train_sample"), ("best", "train_sample")]
    by_key = {(_stem(p["checkpoint"]), p["split"]): p for p in passes if p.get("status") == "ok"}
    for key in order:
        p = by_key.get(key)
        if p is None:
            continue
        ci = p.get("ce_interval", {})
        source = "post-hoc" + (" (subset)" if p.get("subset") else "")
        lines.append(f"| {labels[key]} | {p.get('step')} | {math.exp(p['ce']):.4f} | {p['ce']:.6f} | "
                     f"[{_fmt(ci.get('lo'))}, {_fmt(ci.get('hi'))}] | {source} |")
    return "\n".join(lines) + "\n"


def resources_table(resources: Mapping[str, Any]) -> str:
    r"""Markdown of ``tab:placeholder_resources`` for this run."""
    pb = resources.get("parameter_breakdown") or {}
    opt = resources.get("optimizer_state") or {}
    thr = resources.get("throughput") or {}
    lat = resources.get("inference_latency") or {}

    def count(section: str) -> str:
        entry = pb.get(section) or {}
        if not entry:
            return "–"
        return f"{int(entry['total']):,} ({100.0 * float(entry['share']):.2f}%)"

    peak = thr.get("peak_mem_mb")
    opt_bytes = opt.get("optimizer_state_bytes")
    tps = ((thr.get("throughput_tokens_per_s") or {}).get("tokens_per_s") or {})
    hours = thr.get("wall_time_hours")
    rows = [
        ("Per-type tables (token frames, prior means and covariances, decoder)", count("per_type_tables")),
        ("Positional frame generators", count("positional_frames")),
        ("Context-processing parameters (interface maps, MLP, normalization)", count("context_processing")),
        ("Parameters read per token", f"{int(pb['parameters_read_per_token']):,}" if pb else "–"),
        ("Optimizer state and peak device memory",
         (f"{opt_bytes / 2**30:.2f} GiB optimizer state" if opt_bytes else "– optimizer state")
         + (f"; peak {peak / 1000:.2f} GB" if peak else "; peak –")),   # peak_mem_mb is bytes / 1e6
        ("Training throughput and device-hours",
         (f"median {tps['median']:,.0f} tokens/s" if tps.get("median") else "–")
         + (f"; {hours:.2f} device-hours wall" if hours else "")
         + (f" ({thr.get('device_name')})" if thr.get("device_name") else "")),
        ("Inference latency at declared batch and context",
         (f"{1000 * lat['seconds_per_batch_median']:.1f} ms per batch of {lat['batch_size']} x "
          f"{lat['context']} ({lat.get('device_name')}, {lat.get('precision')})")
         if lat.get("status") == "ok" else "–"),
    ]
    total = f"{int(pb['total']):,}" if pb else "–"
    lines = ["| Parameter role or resource | This run |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines) + f"\n\nTotal parameters: {total}.\n"


def klcap_table(summary: Mapping[str, Any]) -> str:
    r"""Markdown of the cap activation per population and layer."""
    lines = [f"KL cost cap kappa = {summary.get('kl_max')}. Populations: "
             + "; ".join(f"{k}: {v}" for k, v in (summary.get("populations") or {}).items()) + ".", "",
             "| Population | Layer | Capped entries | Fraction | Mean beta mass on capped entries per row "
             "| Max beta on a capped entry | Capped at window start | Max uncapped causal cost | Windows with a cap |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in summary.get("per_layer", []):
        lines.append(f"| {r['population']} | {r['layer']} | {r['n_capped']:,} / {r['n_entries']:,} | "
                     f"{_fmt(r['frac_capped'], 6)} | {_fmt(r['capped_beta_mass_mean'], 6)} | "
                     f"{_fmt(r['max_beta_capped'])} | {r['capped_at_window_start']:,} | "
                     f"{_fmt(r['max_cost_uncapped'], 1)} | {len(r['windows_with_cap'])} |")
    return "\n".join(lines) + "\n"


def attention_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| Layer | Head | Entropy / log(#sources) | Mean distance | Self-weight | Prior overlap | "
             "Min entropy row (entropy, stream pos) | Rows |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def cell(key: str, digits: int = 3) -> str:
            return f"{_fmt(r.get(key), digits)} [{_fmt(r.get(key + '_lo'), digits)}, {_fmt(r.get(key + '_hi'), digits)}]"
        lines.append(f"| {r['layer']} | {r['head']} | {cell('entropy_norm')} | {cell('distance_mean', 1)} | "
                     f"{cell('self_weight')} | {cell('prior_overlap')} | "
                     f"{_fmt(r.get('min_entropy_norm'))}, {r.get('min_target_stream_pos', '–')} | {r['n_rows']:,} |")
    return "\n".join(lines) + "\n"


def map_invariants_table(rows: Sequence[Mapping[str, Any]]) -> str:
    r"""Per layer, the range over heads of the S1 panel B quantities."""
    lines = ["| Layer | Common scale | Relative gains (min-max) | Median principal angle (deg) | "
             "sigma_max(O T) | Own-block share S_Q / S_K / T / O |", "|---|---|---|---|---|---|"]
    for layer in sorted({int(r["layer"]) for r in rows}):
        mine = [r for r in rows if int(r["layer"]) == layer]

        def span(key: str, spec: str = ".3g") -> str:
            values = [float(r[key]) for r in mine if r.get(key) is not None and np.isfinite(r[key])]
            return f"{min(values):{spec}}–{max(values):{spec}}" if values else "–"
        shares = " / ".join(span(f"{m}_own_block_share", ".2f") for m in ("S_Q", "S_K", "T", "O"))
        gains = [v for r in mine for v in (r["gain_min"], r["gain_max"])]
        lines.append(f"| {layer} | {span('scale_rms')} | {min(gains):.3g}–{max(gains):.3g} | "
                     f"{span('angle_median_deg', '.1f')} | {span('M_sigma_max')} | {shares} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------------------

def _datasets(
    info:     st.RunInfo,
    options:  PosthocOptions,
    log:      Callable[[str], None],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Optional[np.ndarray]], Dict[str, torch.Tensor],
           Dict[str, np.ndarray]]:
    r"""``(datasets, stream records, per-position units, tokens, reproduction windows)`` of the requested splits.

    The reproduction windows of ``train_sample`` index the windows of the post-hoc sample that the
    training loop scored as its own (smaller) fixed sample, the population of the logged
    ``train_sample_ce``.
    """
    cache_dir = Path(options.cache_dir) if options.cache_dir else None
    datasets: Dict[str, Any] = {}
    records: Dict[str, Dict[str, Any]] = {}
    units: Dict[str, Optional[np.ndarray]] = {}
    tokens_by_split: Dict[str, torch.Tensor] = {}
    reproduction_windows: Dict[str, np.ndarray] = {}
    needed = set(options.splits)
    if "train_sample" in needed:
        needed.add("validation")                     # the sample is sized to the validation targets
    for split in ("validation", "test"):
        if split not in needed:
            continue
        tokens, record = st.resolve_stream(info, split, cache_dir=cache_dir,
                                           allow_mismatch=options.allow_stream_mismatch)
        dataset = st.eval_windows(tokens, info.seq_len, info.eval_stride)
        record["protocol"] = {"seq_len": info.seq_len, "stride": info.eval_stride, "n_windows": len(dataset)}
        identity = info.selection_identity or {}
        iterator = identity.get("iterator") if identity.get("split") == split else None
        if isinstance(iterator, Mapping):
            record["protocol"]["recorded_iterator"] = dict(iterator)
            record["protocol"]["iterator_ok"] = (
                int(iterator.get("seq_len", -1)) == info.seq_len
                and int(iterator.get("stride", -1)) == info.eval_stride
                and int(iterator.get("n_windows", -1)) == len(dataset))
            if not record["protocol"]["iterator_ok"] and not options.allow_stream_mismatch:
                raise st.StreamIdentityError(
                    f"{split}: evaluation windows {record['protocol']} differ from the recorded iterator {iterator}")
        position_units, unit_info = st.resampling_units(tokens, info.dataset,
                                                        vocab_size=int(info.cfg.vocab_size))
        record["units"] = {k: v for k, v in unit_info.items() if k != "titles"}
        record["units"]["titles"] = unit_info.get("titles", [])
        datasets[split], records[split], units[split] = dataset, record, position_units
        tokens_by_split[split] = tokens
        log(f"[posthoc] {split}: {record['n_tokens']:,} tokens, {len(dataset)} windows, "
            f"{unit_info.get('n_units')} {unit_info.get('unit')}s, identity {record['status']}")
    if "train_sample" in options.splits:
        tokens, record = st.resolve_stream(info, "train", cache_dir=cache_dir,
                                           allow_mismatch=options.allow_stream_mismatch)
        n_val = int(datasets["validation"].tokens.numel()) - 1 if "validation" in datasets else st.TRAIN_SAMPLE_MAX_TARGETS
        sample = st.fixed_train_sample(tokens, info.seq_len, info.eval_stride, n_val)
        training_windows, training_max_targets = st.training_loop_sample_windows(sample, tokens, n_val)
        reproduction_windows["train_sample"] = training_windows
        record["protocol"] = {"seq_len": info.seq_len, "stride": info.eval_stride, "n_windows": len(sample),
                              "seed": st.TRAIN_SAMPLE_SEED, "max_targets": st.TRAIN_SAMPLE_MAX_TARGETS,
                              "requested_targets": n_val, "scored_targets": len(sample) * info.eval_stride,
                              "window_starts_sha256": hashlib.sha256(
                                  np.asarray(sample.starts, dtype=np.int64).tobytes()).hexdigest(),
                              "reproduction_sample": {
                                  "rule": "the windows the training loop scored (vfe3.train."
                                          "TRAIN_SAMPLE_MAX_TARGETS cap, a prefix of the same seeded "
                                          "permutation); the logged train_sample_ce is checked on these only",
                                  "max_targets": training_max_targets,
                                  "n_windows": int(training_windows.size),
                                  "scored_targets": int(training_windows.size) * info.eval_stride,
                                  "window_starts_sha256": hashlib.sha256(np.asarray(
                                      sample.starts[training_windows], dtype=np.int64).tobytes()).hexdigest()}}
        record["units"] = {"unit": "window", "rule": "each sampled training window is one resampling unit"}
        datasets["train_sample"], records["train_sample"], units["train_sample"] = sample, record, None
        log(f"[posthoc] train sample: {len(sample)} windows of the {record['n_tokens']:,}-token train "
            f"stream ({training_windows.size} of them the training loop's sample), identity {record['status']}")
    return datasets, records, units, tokens_by_split, reproduction_windows


def _indices(split: str, dataset: Any, options: PosthocOptions) -> np.ndarray:
    if split != "train_sample" and options.window_indices is not None:
        chosen = np.asarray(sorted({int(i) for i in options.window_indices if 0 <= int(i) < len(dataset)}),
                            dtype=np.int64)
        return chosen
    return st.evenly_spaced(len(dataset), options.max_windows)


def run_posthoc(
    run_dir: "str | Path",

    *,
    out_dir: "str | Path | None" = None,
    options: Optional[PosthocOptions] = None,
    log:     Callable[[str], None] = print,
) -> Dict[str, Any]:
    r"""Run every requested analysis on ``run_dir`` and write ``posthoc.json``, tables and figures."""
    from vfe3.posthoc import interventions as iv
    from vfe3.posthoc.attention import AttentionStatsObserver
    from vfe3.posthoc.klcap import KLCapObserver
    from vfe3.posthoc.residual import ResidualObserver
    from vfe3.semantics.common import write_json

    options = options or PosthocOptions()
    started = time.perf_counter()
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "posthoc"
    for sub in ("tables", "csv", "figures", "nll"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = ev.resolve_device(options.device)
    final_ckpt = next((c for c in options.checkpoints if c.startswith("final")), options.checkpoints[0])
    info = st.load_run_info(run_dir, final_ckpt, drop_unknown_config_fields=options.drop_unknown_config_fields)
    cfg = info.cfg
    from vfe3.contracts import is_vfe_config, model_family_of

    skipped_analyses: Dict[str, str] = {}
    if not is_vfe_config(cfg):
        options, skipped_analyses = baseline_posthoc_options(options)
        log(f"[posthoc] model_family={model_family_of(cfg)!r}: VFE-only analyses skipped: "
            f"{', '.join(sorted(skipped_analyses)) or 'none'}")
    batch_size = int(options.batch_size or cfg.batch_size)
    log(f"[posthoc] {run_dir.name}: K={cfg.embed_dim} L={cfg.n_layers} H={cfg.n_heads} "
        f"seq_len={info.seq_len} stride={info.eval_stride} batch={batch_size} device={device} "
        f"precision={options.precision}")
    if info.dropped_config_fields:
        log(f"[posthoc] dropped {len(info.dropped_config_fields)} config fields this build does not declare: "
            f"{sorted(info.dropped_config_fields)}")

    recorded_options = dataclasses.asdict(options)
    if options.window_indices is not None:
        recorded_options["window_indices"] = [int(i) for i in options.window_indices]
    report: Dict[str, Any] = {
        "run_dir": str(run_dir), "out_dir": str(out), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "options": recorded_options, "device": str(device), "device_name": _device_name(device),
        "torch_version": torch.__version__, "batch_size": batch_size,
        "protocol": {"seq_len": info.seq_len, "eval_stride": info.eval_stride},
        "dropped_config_fields": info.dropped_config_fields,
        "passes": [], "interventions": None, "klcap": {}, "attention": None, "maps": {},
        "resources": None, "alibi_slopes": None, "figures": [], "tables": {}, "notes": [],
        "residual": {}, "covariance": {}, "calibration": None, "routing": None, "ce_by_context": None,
        "checkpoint_aliases": {},
    }
    if not is_vfe_config(cfg):
        report["model_family"] = model_family_of(cfg)
        report["skipped_analyses"] = skipped_analyses

    datasets, records, stream_units, tokens_by_split, reproduction_windows = _datasets(info, options, log)
    report["streams"] = records

    intervention_rows: List[Dict[str, Any]] = []
    attention_observer: Optional[AttentionStatsObserver] = None
    latency: Optional[Dict[str, Any]] = None
    test_passes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}          # checkpoint -> (nll, target_pos)
    context_series: Dict[str, List[Dict[str, float]]] = {}
    if not (options.routing and options.routing_split):
        report["routing"] = {"status": "disabled", "reason": "options.routing is off or routing_split is None"}
    elif options.routing_split not in datasets or options.routing_split not in options.splits:
        report["routing"] = {"status": "skipped", "reason": f"split {options.routing_split!r} was not evaluated"}
    if not options.calibration:
        report["calibration"] = {"status": "disabled", "reason": "options.calibration is off"}
    if options.residual_split is None:
        report["notes"].append("residual ratios disabled (residual_split=None)")
    elif options.residual_split not in options.splits:
        report["notes"].append(f"residual ratios skipped: split {options.residual_split!r} is not evaluated")
    if not options.covariance:
        report["notes"].append("routing-covariance gap and hidden temperature disabled (covariance=False)")
    aliases = _checkpoint_aliases(run_dir, options, final_ckpt)
    kept: Dict[str, Tuple[ev.TargetNLL, np.ndarray, bool]] = {}     # final_ckpt's passes, for the aliases
    kept_matched = False
    for checkpoint in options.checkpoints:
        if not (run_dir / checkpoint).is_file():
            report["notes"].append(f"{checkpoint} not found; skipped")
            continue
        if checkpoint in aliases:
            _alias_checkpoint(report, info, checkpoint, aliases[checkpoint], kept=kept, matched=kept_matched,
                              reproduction_windows=reproduction_windows, test_passes=test_passes)
            report["notes"].append(f"{checkpoint}: {report['checkpoint_aliases'][checkpoint]['reason']}")
            log(f"[posthoc] {checkpoint}: model state identical to {aliases[checkpoint]}; its results are reused")
            continue
        loaded = ev.load_model(run_dir, checkpoint, device=device, precision=options.precision,
                               drop_unknown_config_fields=options.drop_unknown_config_fields)
        model = loaded.model
        matched = _replay_matches_run(loaded, cfg, device)
        report["notes"].extend(f"{checkpoint}: {n}" for n in loaded.notes)
        step = _checkpoint_step(info, checkpoint)
        is_final = checkpoint == final_ckpt
        if is_final:
            kept_matched = matched
            report["alibi_slopes"] = ev.alibi_slopes(model, info.seq_len)
            if options.latency_batches > 0 and "validation" in datasets:
                latency = measure_latency(model, datasets["validation"], batch_size, device, options.latency_batches)
                latency.update(device_name=_device_name(device), precision=loaded.amp_dtype or "fp32")
        for split in options.splits:
            dataset = datasets.get(split)
            if dataset is None:
                continue
            indices = _indices(split, dataset, options)
            full = indices.size == len(dataset)
            observers: List[Any] = []
            klcap_observer = None
            if is_final and split in options.klcap_splits:
                # The minimal-agent posterior applies no cost clamp: only a non-finite cost counts (kl_max = inf).
                klcap_observer = KLCapObserver(math.inf if getattr(cfg, "predictive_readout", "kernel")
                                               == "minimal_agent" else float(cfg.kl_max),
                                               getattr(cfg, "routing_kl_max_reference", "absolute"))
                observers.append(klcap_observer)
            if is_final and split == options.attention_split:
                attention_observer = AttentionStatsObserver(stream_units.get(split))
                observers.append(attention_observer)
            residual_observer: Optional[ResidualObserver] = None
            self_weight_observer: Optional[AttentionStatsObserver] = None
            if options.residual_split == split:
                residual_observer = ResidualObserver(model)
                observers.append(residual_observer)
                if getattr(model, "relational_operators", None) is None:
                    self_weight_observer = None                   # no projected router: no beta_pre
                elif is_final and split == options.attention_split:
                    self_weight_observer = attention_observer
                else:
                    self_weight_observer = AttentionStatsObserver(stream_units.get(split))
                    observers.append(self_weight_observer)
            log(f"[posthoc] {checkpoint} / {split}: {indices.size} of {len(dataset)} windows")
            entry: Dict[str, Any] = {"checkpoint": checkpoint, "step": step, "split": split,
                                     "n_windows": int(indices.size), "n_windows_total": len(dataset),
                                     "subset": not full, "precision": loaded.amp_dtype or "fp32",
                                     "fp32_matmul_precision": torch.get_float32_matmul_precision()}
            try:
                result = ev.nll_pass(model, dataset, indices, batch_size=batch_size, device=device,
                                     observers=observers)
            except Exception as exc:                          # recorded; the other passes continue
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                report["passes"].append(entry)
                log(f"[posthoc]   FAILED: {entry['error']}")
                if residual_observer is not None:
                    report["residual"][checkpoint] = {"status": "failed", "reason": "the evaluation pass failed",
                                                      "error": entry["error"]}
                if is_final and options.routing and split == options.routing_split:
                    report["routing"] = {"status": "skipped", "reason": "the evaluation pass of its split failed"}
                continue
            if residual_observer is not None:
                report["residual"][checkpoint] = _residual_section(
                    residual_observer, self_weight_observer, checkpoint=checkpoint, step=step, split=split,
                    subset=not full, n_windows=int(indices.size))
            if split == "test":
                test_passes[checkpoint] = (result.nll, result.target_pos)
            if is_final and aliases:
                kept[split] = (result, indices, full)
            if is_final and split in ("validation", "test"):
                from vfe3.posthoc.figures import ce_by_context

                context_series[f"{split}, {_stem(checkpoint)} weights"] = ce_by_context(result.context, result.nll)
            units = _units_for(result, split, stream_units.get(split))
            entry.update(status="ok", ce=result.ce, n_scored=result.n, expected=result.expected,
                         excluded=result.excluded, seconds=result.seconds,
                         ce_interval=ev.ce_interval(result.nll, units, n_resamples=options.bootstrap_resamples))
            if options.cross_check:
                protocol = ev.protocol_evaluate(model, dataset, indices, batch_size=batch_size, device=device)
                delta = result.ce - float(protocol["ce"])
                entry["cross_check"] = {"evaluate_ce": float(protocol["ce"]),
                                        "evaluate_scored_targets": protocol.get("scored_targets"),
                                        "delta_ce": delta, "tolerance": ev.CROSS_CHECK_TOL,
                                        "ok": abs(delta) <= ev.CROSS_CHECK_TOL
                                        and int(protocol.get("scored_targets") or -1) == result.n}
            reference = _recorded_reference(info, checkpoint, split)
            if split in reproduction_windows:
                entry["reproduction"] = _subsample_reproduction(
                    result, indices, reproduction_windows[split], reference, matched=matched)
            else:
                entry["reproduction"] = _reproduction(entry, reference, full=full, matched=matched)
            if options.save_nll:
                path = out / "nll" / f"{_stem(checkpoint)}_{split}.npz"
                ev.save_nll(path, result, units)
                entry["nll_file"] = str(path)
            if klcap_observer is not None:
                report["klcap"][split] = {"summary": klcap_observer.summary(), "rows": klcap_observer.rows(),
                                          "checkpoint": checkpoint, "subset": not full}
            report["passes"].append(entry)
            log(f"[posthoc]   CE {result.ce:.6f} (PPL {math.exp(result.ce):.4f}) over {result.n:,} targets "
                f"in {result.seconds:.1f}s; reproduction {entry['reproduction']['status']}"
                + (f"; evaluate() delta {entry['cross_check']['delta_ce']:+.2e}" if "cross_check" in entry else ""))

            if checkpoint == options.intervention_checkpoint and split == options.intervention_split \
                    and (options.interventions is None or len(options.interventions) > 0):
                intervention_rows = _run_interventions(
                    iv, model, dataset, indices, result, units, batch_size=batch_size, device=device,
                    keys=options.interventions, n_resamples=options.bootstrap_resamples, log=log,
                    save_to=(out / "nll" / f"interventions_{split}.npz") if options.save_nll else None)
                report["interventions"] = {
                    "checkpoint": checkpoint, "split": split, "subset": not full,
                    "n_targets": result.n, "unit": records[split]["units"]["unit"],
                    "n_units": int(np.unique(units).size), "rows": intervention_rows}
            if is_final and options.routing and split == options.routing_split:
                report["routing"] = _routing(model, dataset, indices, batch_size=batch_size, device=device,
                                             out=out, checkpoint=checkpoint, split=split, subset=not full, log=log)
        if options.covariance:
            report["covariance"][checkpoint] = _covariance(
                model, datasets.get(options.covariance_split), tokens_by_split.get(options.covariance_split),
                device=device, checkpoint=checkpoint, step=step, split=options.covariance_split, log=log)
        if attention_observer is not None and is_final:
            report["attention"] = {"checkpoint": checkpoint, "split": options.attention_split,
                                   "rows": attention_observer.rows(), "layers": attention_observer.layer_rows()}
            _attach_min_entropy_context(report["attention"]["rows"], tokens_by_split.get(options.attention_split),
                                        info.dataset)
        del model, loaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report["reproduction"] = _reproduction_summary(report, info)
    if options.calibration:
        report["calibration"] = _calibration(info, options, tokens_by_split.get("test"), test_passes,
                                             final_ckpt=final_ckpt, log=log)
    if context_series:
        report["ce_by_context"] = {"series": context_series,
                                   "definition": "mean NLL of the scored targets per context length (tokens up "
                                                 "to and including the query position)"}

    if options.maps:
        report["maps"] = _maps(run_dir, options.checkpoints, cfg, out, options.figures, report)
    if options.resources:
        report["resources"] = _resources(run_dir, final_ckpt, cfg, latency)

    _write_outputs(report, info, out, options)
    report["seconds"] = time.perf_counter() - started
    write_json(out / "posthoc.json", report)
    log(f"[posthoc] wrote {out / 'posthoc.json'} in {report['seconds']:.1f}s")
    return report


def _run_interventions(
    iv:          Any,
    model:       torch.nn.Module,
    dataset:     Any,
    indices:     np.ndarray,
    base:        ev.TargetNLL,
    units:       np.ndarray,

    *,
    batch_size:  int,
    device:      torch.device,
    keys:        Optional[Sequence[str]],
    n_resamples: int,
    log:         Callable[[str], None],
    save_to:     Optional[Path],
) -> List[Dict[str, Any]]:
    r"""Rows 1-12 of M5 on the baseline's windows, paired with the baseline per target (inapplicable rows recorded)."""
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {"baseline": base.nll}
    base_interval = ev.ce_interval(base.nll, units, n_resamples=n_resamples)
    # Restoration reference: the check windows scored alone BEFORE any arm, so the re-run after the arms
    # has the same batch composition (under CUDA autocast a different batch shape can pick different
    # kernels and move per-target NLL by ~1e-2 without any patch leaking).
    check = indices[:RESTORATION_CHECK_WINDOWS]
    reference = ev.nll_pass(model, dataset, check, batch_size=batch_size, device=device)
    for intervention in iv.by_key(keys):
        row: Dict[str, Any] = {"row": intervention.row, "key": intervention.key, "label": intervention.label}
        if intervention.key == "baseline":
            row.update(status="ok", ce=base.ce, delta_ce=0.0, lo=0.0, hi=0.0, se=0.0,
                       ce_lo=base_interval["lo"], ce_hi=base_interval["hi"], n_targets=base.n)
            rows.append(row)
            continue
        started = time.perf_counter()
        try:
            with intervention.apply(model) as record:
                arm = ev.nll_pass(model, dataset, indices, batch_size=batch_size, device=device)
            i_base, i_arm, dropped = _align(base, arm)
            base_nll, arm_nll = base.nll[i_base], arm.nll[i_arm]
            interval = ev.paired_delta_interval(arm_nll, base_nll, units[i_base], n_resamples=n_resamples)
            row.update(status="ok", ce=float(arm_nll.mean()), delta_ce=interval["point"], lo=interval["lo"],
                       hi=interval["hi"], se=interval["se"], n_targets=int(arm_nll.size),
                       n_targets_dropped=dropped, ce_base_aligned=float(base_nll.mean()), record=dict(record))
            if dropped == 0:
                arrays[intervention.key] = arm.nll
        except iv.NotApplicable as exc:
            row.update(status="not_applicable", reason=str(exc))
        except Exception as exc:                              # recorded; the other rows continue
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        log(f"[posthoc]   M5 row {intervention.row} {intervention.key}: {row['status']}"
            + (f" dCE {row['delta_ce']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]" if row["status"] == "ok" else
               f" ({row.get('reason') or row.get('error')})"))
    # Every patch must have been undone: the check windows reproduce their pre-arm reference.
    again = ev.nll_pass(model, dataset, check, batch_size=batch_size, device=device)
    restored = (float(np.max(np.abs(again.nll - reference.nll)))
                if again.n and again.n == reference.n else (0.0 if again.n == reference.n else float("inf")))
    rows.append({"row": 0, "key": "restoration_check", "label": "baseline re-run after all arms",
                 "status": "ok" if restored <= ev.CROSS_CHECK_TOL else "failed",
                 "max_abs_nll_change": restored, "n_targets": again.n})
    for row_index, key, label in iv.NOT_IMPLEMENTED:
        if keys is None:
            rows.append({"row": row_index, "key": key, "label": label, "status": "not implemented",
                         "reason": iv.NOT_IMPLEMENTED_REASON})
    if save_to is not None:
        np.savez_compressed(save_to, target_pos=base.target_pos, token=base.token, window=base.window,
                            context=base.context, unit=np.asarray(units, dtype=np.int64),
                            **{f"nll_{k}": v for k, v in arrays.items()})
    return rows


def _attach_min_entropy_context(rows: List[Dict[str, Any]], tokens: Optional[torch.Tensor], dataset: str,
                                n_context: int = 8) -> None:
    r"""Add the decoded text before each minimum-entropy query and its most-weighted source token."""
    if tokens is None:
        return
    try:
        from vfe3.data.datasets import get_tiktoken_byte_decoder
        decode = get_tiktoken_byte_decoder(dataset)
    except Exception:
        decode = None
    if decode is None:
        return
    flat = tokens.reshape(-1)
    for row in rows:
        q = row.get("min_query_stream_pos")
        s = row.get("min_argmax_source_stream_pos")
        if q is None:
            continue
        lo = max(0, int(q) - n_context + 1)
        row["min_context_text"] = decode(flat[lo:int(q) + 1].tolist()).decode("utf-8", errors="replace")
        if s is not None:
            row["min_source_token_text"] = decode([int(flat[int(s)])]).decode("utf-8", errors="replace")


def _residual_section(
    observer:             Any,
    self_weight_observer: Optional[Any],

    *,
    checkpoint: str,
    step:       Optional[int],
    split:      str,
    subset:     bool,
    n_windows:  int,
) -> Dict[str, Any]:
    r"""The v2:899 section of one pass: residual quantiles and the per-layer self-weights."""
    from vfe3.posthoc.residual import self_weight_summary

    section = observer.result()
    section.update(checkpoint=checkpoint, stem=_stem(checkpoint), step=step, split=split, subset=subset,
                   n_windows=n_windows)
    if self_weight_observer is None:
        section["self_weight"] = {"status": "unavailable",
                                  "reason": "no projected router on this route: no routing weights beta"}
    elif not self_weight_observer.n_batches:
        section["self_weight"] = {"status": "unavailable", "reason": "no routing-weight captures reached the observer"}
    else:
        section["self_weight"] = self_weight_summary(self_weight_observer.rows(n_resamples=0),
                                                     self_weight_observer.layer_rows())
    return section


def _plain(obj: Any) -> Any:
    r"""``obj`` with torch tensors converted to lists (for ``posthoc.json``)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _routing(
    model:      torch.nn.Module,
    dataset:    Any,
    indices:    np.ndarray,

    *,
    batch_size: int,
    device:     torch.device,
    out:        Path,
    checkpoint: str,
    split:      str,
    subset:     bool,
    log:        Callable[[str], None],
) -> Dict[str, Any]:
    r"""M4 (``fig:placeholder_routing_decomposition``) through :mod:`vfe3.posthoc.routing`, imported lazily."""
    base = {"checkpoint": checkpoint, "split": split, "subset": subset, "n_windows": int(indices.size)}
    try:
        from vfe3.posthoc import routing as rt
    except Exception as exc:                                  # missing module or a failing import
        log(f"[posthoc] M4 routing decomposition skipped: {type(exc).__name__}: {exc}")
        return {**base, "status": "skipped", "reason": f"vfe3.posthoc.routing unavailable ({type(exc).__name__}: {exc})"}
    started = time.perf_counter()
    try:
        result = rt.routing_decomposition(model, dataset, indices, batch_size=batch_size, device=device)
        paths = [str(p) for p in rt.write_routing_outputs(result, out)]
    except Exception as exc:                                  # recorded; the report continues
        log(f"[posthoc] M4 routing decomposition FAILED: {type(exc).__name__}: {exc}")
        return {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - started}
    log(f"[posthoc] M4 routing decomposition written ({len(paths)} files)")
    status = result.get("status", "ok") if isinstance(result, Mapping) else "ok"
    reason = result.get("reason") if isinstance(result, Mapping) else None
    # The per-row samples live in m4_routing_samples.npz (write_routing_outputs), not in posthoc.json.
    summary = {k: v for k, v in result.items() if k != "samples"} if isinstance(result, Mapping) else result
    samples_file = next((p for p in paths if p.endswith("m4_routing_samples.npz")), None)
    return {**base, "status": status, "reason": reason, "result": _plain(summary), "outputs": paths,
            "samples_file": samples_file, "seconds": time.perf_counter() - started}


def _covariance(
    model:         torch.nn.Module,
    dataset:       Optional[Any],
    stream_tokens: Optional[torch.Tensor],

    *,
    device:     torch.device,
    checkpoint: str,
    step:       Optional[int],
    split:      str,
    log:        Callable[[str], None],
) -> Dict[str, Any]:
    r"""The v2:872 section at one set of weights (each part isolated inside :func:`covariance_analysis`)."""
    from vfe3.posthoc.covariance import covariance_analysis

    started = time.perf_counter()
    try:
        result = covariance_analysis(model, dataset, stream_tokens=stream_tokens, device=device)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result.update(checkpoint=checkpoint, stem=_stem(checkpoint), step=step, split=split,
                  seconds=time.perf_counter() - started)
    rc = result.get("routing_covariance") or {}
    ht = result.get("hidden_temperature") or {}
    log(f"[posthoc] {checkpoint} covariance: C_i gap {rc.get('status')}"
        + (f" (c={rc['c']:.4g}, median Frobenius {rc['median_frobenius_gap']:.4g}, median log-det "
           f"{rc['median_logdet_gap']:.4g})" if rc.get("status") == "ok" else f" ({rc.get('reason') or rc.get('error')})")
        + f"; hidden temperature {ht.get('status')}"
        + (f" {ht['factor']:.4g}" if ht.get("status") == "ok" else ""))
    return result


def _calibration(
    info:        st.RunInfo,
    options:     PosthocOptions,
    tokens:      Optional[torch.Tensor],
    test_passes: Mapping[str, Tuple[np.ndarray, np.ndarray]],

    *,
    final_ckpt: str,
    log:        Callable[[str], None],
) -> Dict[str, Any]:
    r"""``tab:placeholder_calibration`` from the full test passes (skipped with a reason otherwise)."""
    from vfe3.posthoc.calibration import calibration_for_run

    weights = str(info.summary.get("evaluation_weights") or "final")
    headline = next((c for c in options.checkpoints if c.startswith(weights)), final_ckpt)
    try:
        result = calibration_for_run(
            info.dataset, tokens, test_passes, seq_len=info.seq_len, stride=info.eval_stride,
            headline_checkpoint=headline, word_stream_path=options.word_stream_path,
            steps={c: _checkpoint_step(info, c) for c in test_passes})
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    head = (result.get("checkpoints") or {}).get(headline) or {}
    conv = (head.get("conventions") or {}).get(result.get("headline_convention") or "", {})
    log(f"[posthoc] word-normalized test PPL: {result.get('status')}"
        + (f" {conv['ppl']:.3f} ({conv['count']:,} word-level tokens, text audit same_text="
           f"{result['audit']['same_text']})" if conv else f" ({result.get('reason') or result.get('error')})"))
    return result


def _reproduction_summary(report: Mapping[str, Any], info: st.RunInfo) -> Dict[str, Any]:
    checks = []
    for p in report["passes"]:
        rep = p.get("reproduction") or {}
        if rep.get("status") in ("ok", "mismatch"):
            checks.append({"checkpoint": p["checkpoint"], "split": p["split"], "status": rep["status"],
                           "ce": rep.get("ce", p["ce"]), "reference": rep["reference"], "delta_ce": rep["delta_ce"],
                           "tolerance": rep["tolerance"], "arithmetic": rep["arithmetic"]})
    final_checks = [c for c in checks if c["checkpoint"].startswith("final")]
    return {
        "dropped_config_fields": info.dropped_config_fields,
        "checks": checks,
        "final_weights_reproduced": (all(c["status"] == "ok" for c in final_checks) if final_checks else None),
        "note": ("a dropped config field that changed the forward pass would show as a final-weight "
                 "reproduction mismatch" if info.dropped_config_fields else None),
    }


def _maps(run_dir: Path, checkpoints: Sequence[str], cfg: Any, out: Path, figures: bool,
          report: Dict[str, Any]) -> Dict[str, Any]:
    from vfe3.posthoc import maps as mp
    from vfe3.posthoc import figures as fg

    if len(set(getattr(cfg, "head_dims", None) or [0])) > 1:
        return {"status": "not_applicable", "reason": "heads of different sizes (a mixed so_n irrep tower) are not supported by this analysis yet"}
    result: Dict[str, Any] = {}
    for checkpoint in checkpoints:
        path = run_dir / checkpoint
        if not path.is_file():
            continue
        state, _ = mp.load_map_state(path)
        rows = mp.map_invariants(state, cfg)
        norms = mp.head_block_norms(state, cfg)
        long_rows = []
        for layer, per_map in norms.items():
            for name, matrix in per_map.items():
                shares = mp.block_energy_shares(matrix)
                for (h, b), value in np.ndenumerate(matrix):
                    long_rows.append({"layer": layer, "map": name, "head": h, "block": b,
                                      "norm": float(value), "share": float(shares[h, b])})
        stem = _stem(checkpoint)
        _write_csv(out / "csv" / f"map_invariants_{stem}.csv", rows)
        eig_rows = mp.link_eigenvalue_rows(state, cfg)
        _write_csv(out / "csv" / f"link_eigenvalues_{stem}.csv", eig_rows)
        _write_csv(out / "csv" / f"head_block_norms_{stem}.csv", long_rows)
        result[checkpoint] = {"invariants": rows}
        if figures:
            report["figures"] += [str(p) for p in fg.plot_head_block_norms(norms, out / "figures",
                                                                           f"s2a_head_block_shares_{stem}")]
            report["figures"] += [str(p) for p in fg.plot_map_invariants(rows, out / "figures",
                                                                         f"s1b_map_invariants_{stem}")]
            if eig_rows:
                report["figures"] += [str(p) for p in fg.plot_link_spectrum(eig_rows, out / "figures",
                                                                            f"s1b_link_spectrum_{stem}")]
    snapshots = run_dir / "map_snapshots"
    if mp.snapshot_files(snapshots):
        curves = mp.snapshot_curves(snapshots, cfg)
        _write_csv(out / "csv" / "map_invariant_curves.csv", curves)
        result["snapshot_curves"] = {"n_snapshots": len(mp.snapshot_files(snapshots)), "n_rows": len(curves)}
        if figures and curves:
            report["figures"] += [str(p) for p in fg.plot_map_invariant_curves(curves, out / "figures")]
    else:
        result["snapshot_curves"] = {"n_snapshots": 0,
                                     "note": "no map_snapshots/ in the run directory (runs before 2026-09-28)"}
    return result


def _resources(run_dir: Path, checkpoint: str, cfg: Any, latency: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from vfe3.posthoc.maps import load_map_state
    from vfe3.run_resources import (frozen_parameter_names, optimizer_state_bytes, parameter_breakdown,
                                    throughput_summary)

    out: Dict[str, Any] = {}
    state, _ = load_map_state(run_dir / checkpoint)
    breakdown = parameter_breakdown(state, cfg, frozen=frozen_parameter_names(cfg, state.keys()))
    out["parameter_breakdown"] = {k: v for k, v in breakdown.items() if k != "roles_by_name"}
    latest = _latest_training_checkpoint(run_dir)
    if latest is not None:
        try:
            out["optimizer_state"] = optimizer_state_bytes(latest)
        except Exception as exc:
            out["optimizer_state"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    out["throughput"] = throughput_summary(run_dir)
    out["inference_latency"] = latency or {"status": "not_measured"}
    return out


def _write_outputs(report: Dict[str, Any], info: st.RunInfo, out: Path, options: PosthocOptions) -> None:
    from vfe3.posthoc import figures as fg

    tables: Dict[str, str] = {}
    iv_report = report.get("interventions")
    if iv_report:
        rows = [r for r in iv_report["rows"] if r["row"] > 0]
        tables["interventions"] = interventions_table(
            rows, n_units=iv_report["n_units"], unit=iv_report["unit"], n_targets=iv_report["n_targets"],
            subset=iv_report["subset"])
        _write_csv(out / "csv" / "interventions.csv", iv_report["rows"])
        if options.figures:
            report["figures"] += [str(p) for p in fg.plot_interventions(rows, out / "figures",
                                                                        unit=iv_report["unit"])]
    if report["passes"]:
        tables["relational_run"] = relational_run_table(report["passes"], info)
        _write_csv(out / "csv" / "passes.csv", [
            {k: v for k, v in p.items() if not isinstance(v, dict)} | {
                "ce_lo": (p.get("ce_interval") or {}).get("lo"), "ce_hi": (p.get("ce_interval") or {}).get("hi")}
            for p in report["passes"]])
    if report.get("resources"):
        tables["resources"] = resources_table(report["resources"])
    for split, entry in (report.get("klcap") or {}).items():
        tables[f"klcap_{split}"] = klcap_table(entry["summary"])
        _write_csv(out / "csv" / f"klcap_{split}.csv", entry["rows"])
    if report.get("attention"):
        tables["attention_stats"] = attention_table(report["attention"]["rows"])
        _write_csv(out / "csv" / "attention_stats.csv", report["attention"]["rows"])
        if options.figures:
            report["figures"] += [str(p) for p in fg.plot_attention_stats(report["attention"]["rows"],
                                                                          out / "figures")]
    for checkpoint, entry in (report.get("maps") or {}).items():
        if isinstance(entry, dict) and "invariants" in entry:
            tables[f"map_invariants_{_stem(checkpoint)}"] = map_invariants_table(entry["invariants"])
    for name, text in tables.items():
        path = out / "tables" / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        report["tables"][name] = str(path)
    _write_gap_outputs(report, out, options)


def _write_gap_outputs(report: Dict[str, Any], out: Path, options: PosthocOptions) -> None:
    r"""Tables, CSVs and figures of the residual, covariance, calibration and context sections.

    A failing writer is recorded in ``notes``; the per-pair covariance arrays are written to
    ``csv/covariance_pairs_<stem>.npz`` and then dropped from the report (``posthoc.json``).
    """
    from vfe3.posthoc import calibration as cal
    from vfe3.posthoc import covariance as cv
    from vfe3.posthoc import figures as fg
    from vfe3.posthoc import residual as rs

    def guarded(label: str, writer: Callable[[], Sequence[Path]]) -> None:
        try:
            paths = writer()
        except Exception as exc:                              # recorded; the other outputs are written
            report["notes"].append(f"{label} outputs failed: {type(exc).__name__}: {exc}")
            return
        for path in map(Path, paths):
            if path.suffix == ".md":
                report["tables"][path.stem] = str(path)
            elif path.suffix in (".png", ".pdf"):
                report["figures"].append(str(path))

    for checkpoint, section in (report.get("residual") or {}).items():
        if section.get("status") == "ok":
            guarded(f"residual {checkpoint}",
                    lambda s=section: rs.write_residual_outputs(s, out, figures=options.figures))
    for checkpoint, section in list((report.get("covariance") or {}).items()):
        guarded(f"covariance {checkpoint}",
                lambda s=section: cv.write_covariance_outputs(s, out, figures=options.figures))
        report["covariance"][checkpoint] = cv.without_arrays(section)
    if report.get("calibration"):
        guarded("calibration", lambda: cal.write_calibration_outputs(report["calibration"], out))
    routing_outputs = (report.get("routing") or {}).get("outputs") or []
    guarded("routing", lambda: routing_outputs)                   # written by vfe3.posthoc.routing itself
    series = (report.get("ce_by_context") or {}).get("series") or {}
    if series:
        rows = [dict(r, series=label) for label, entries in series.items() for r in entries]
        guarded("ce_by_context", lambda: [p for p in [_write_csv(out / "csv" / "ce_by_context.csv", rows)] if p]
                + (fg.plot_ce_by_context(series, out / "figures") if options.figures else []))
