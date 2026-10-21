r"""Uniform checkpoint averaging of one run, scored under the run's own evaluation protocol (audit 2026-10-20 item 8).

Given :math:`n \ge 2` model bundles of one run (``best_model.pt``, ``final_model.pt``, ``checkpoints/step_*.pt``),
the averaged state is

.. math::
    \bar\theta = \frac1n\sum_{c=1}^{n}\theta_c,

taken entry by entry over every floating tensor of the state dict (parameters and floating buffers), accumulated in
float64 in the listed order and cast back to the tensor's dtype. A non-floating tensor (an integer or boolean buffer)
cannot be averaged and is taken from the last listed checkpoint. The bundles must share the serialized config (the
resume bookkeeping fields ``resume_from`` and ``trust_resume_checkpoint`` aside) and the key set, shape and dtype of
every tensor; anything else raises before a model is built.

The averaged weights are scored with :func:`vfe3.train.evaluate` on the run's identity-checked validation and test
streams (:func:`vfe3.posthoc.streams.resolve_stream`), at the run's ``max_seq_len`` and ``eval_stride`` with the
padded final window, as the run's loaders score them (:func:`vfe3.posthoc.evaluate.protocol_evaluate`). The result is
written to ``<run_dir>/averaged_eval.json`` (or another new file); no existing artifact of the run is modified. On
the CPU every batched LU-family call runs under one thread (:mod:`vfe3.cpu_linalg`, audit 2026-10-20 item 28).

Polyak and Juditsky (1992) and Izmailov et al. (UAI 2018) give the rationale: a uniform average of late iterates
cancels the zero-mean noise of a learning-rate-dominated tail while keeping its drift.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.cpu_linalg import batched_cpu_lu_guard

# Serialized config fields that record how a run was started, not what it computes; a resumed run's step checkpoints
# carry them while its best and final bundles may not.
RESUME_BOOKKEEPING_FIELDS: Tuple[str, ...] = ("resume_from", "trust_resume_checkpoint")


# JSON artifacts the training run itself writes; the averaged report never replaces them.
PROTECTED_RUN_FILES: Tuple[str, ...] = ("config.json", "summary.json", "provenance.json", "test_results.json",
                                        "final_validation.json", "resources.json", "research.json",
                                        "pure_path_report.json", "prototype_calibration.json")


@dataclass
class CheckpointAverageOptions:
    r"""Options of :func:`run_checkpoint_average` (the click-to-run ``average_checkpoints.py`` passes its ``CONFIG``)."""

    checkpoints:                Sequence[str] = ("best_model.pt", "final_model.pt")   # paths relative to run_dir
    splits:                     Sequence[str] = ("validation", "test")
    device:                     str = "auto"            # "auto" -> cuda when available, else cpu
    precision:                  str = "run"             # "run" (the run's amp dtype on CUDA) or "fp32"
    batch_size:                 Optional[int] = None    # None -> the run's cfg.batch_size
    max_windows:                Optional[int] = None    # None -> full splits; an int -> evenly spaced subset
    evaluate_inputs:            bool = False            # also score every input checkpoint on the same windows
    allow_stream_mismatch:      bool = False
    drop_unknown_config_fields: bool = False
    cache_dir:                  Optional[str] = None    # None -> the run's loaders' default cache
    out_name:                   str = "averaged_eval.json"


def load_model_bundle(path: "str | Path") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    r"""``(model_state, serialized config)`` of a best, final or step checkpoint, loaded on the CPU (weights only)."""
    bundle = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("model_state"), Mapping):
        raise ValueError(f"{path} is not a model bundle with a 'model_state'")
    if not isinstance(bundle.get("config"), Mapping):
        raise ValueError(f"{path} carries no serialized 'config'")
    return dict(bundle["model_state"]), dict(bundle["config"])


def check_compatible(
    states:  Sequence[Mapping[str, torch.Tensor]],
    configs: Sequence[Mapping[str, Any]],
    names:   Sequence[str],
) -> None:
    r"""Raise ``ValueError`` unless every bundle has the first one's config, keys, shapes and dtypes."""
    if len(states) < 2 or len(states) != len(configs) or len(states) != len(names):
        raise ValueError(f"averaging needs at least two checkpoints, got {len(states)}")

    def semantic(config: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in config.items() if key not in RESUME_BOOKKEEPING_FIELDS}

    reference_config = semantic(configs[0])
    reference = states[0]
    for name, state, config in zip(names[1:], states[1:], configs[1:]):
        other = semantic(config)
        differing = sorted(key for key in set(reference_config) | set(other)
                           if reference_config.get(key) != other.get(key))
        if differing:
            raise ValueError(f"{name} was written under a different config than {names[0]}: {differing}")
        if set(state) != set(reference):
            raise ValueError(f"{name} and {names[0]} hold different tensors: "
                             f"{sorted(set(state) ^ set(reference))}")
        for key, tensor in reference.items():
            if state[key].shape != tensor.shape or state[key].dtype != tensor.dtype:
                raise ValueError(f"{key}: {name} has {tuple(state[key].shape)} {state[key].dtype}, "
                                 f"{names[0]} has {tuple(tensor.shape)} {tensor.dtype}")


def average_model_states(states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    r"""The uniform average :math:`\frac1n\sum_c\theta_c` of every floating tensor; others from the last state.

    The sum runs in float64 in the listed order and is divided by :math:`n` before the cast back to each tensor's
    dtype, so averaging :math:`n` identical states returns them exactly.
    """
    if len(states) < 1:
        raise ValueError("no states to average")
    n = len(states)
    out: Dict[str, torch.Tensor] = {}
    for key, last in states[-1].items():
        if torch.is_floating_point(last):
            total = torch.zeros(last.shape, dtype=torch.float64)
            for state in states:
                total += state[key].detach().to(device="cpu", dtype=torch.float64)
            out[key] = (total / n).to(last.dtype)
        else:
            out[key] = last.detach().to("cpu").clone()
    return out


def run_checkpoint_average(
    run_dir: "str | Path",

    *,
    options: Optional[CheckpointAverageOptions] = None,
    log:     Callable[[str], None] = print,
) -> Dict[str, Any]:
    r"""Average ``options.checkpoints`` of ``run_dir``, score validation and test, write the JSON; return the report."""
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc import streams as st

    options = options or CheckpointAverageOptions()
    run_dir = Path(run_dir)
    out_path = run_dir / options.out_name
    if Path(options.out_name).name != options.out_name or not options.out_name.endswith(".json")             or options.out_name in PROTECTED_RUN_FILES:
        raise ValueError(f"out_name must be a new .json file name in the run directory, not one of the run's own "
                         f"artifacts {sorted(PROTECTED_RUN_FILES)}; got {options.out_name!r}")
    started = time.perf_counter()
    paths = [run_dir / name for name in options.checkpoints]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoints not found: {missing}")
    states, configs = zip(*(load_model_bundle(path) for path in paths))
    names = [str(name) for name in options.checkpoints]
    check_compatible(states, configs, names)
    averaged = average_model_states(states)
    steps = {name: _bundle_step(path) for name, path in zip(names, paths)}
    del states
    log(f"[average] {run_dir.name}: uniform average of {', '.join(names)}")

    device = ev.resolve_device(options.device)
    # The model is built from the last listed bundle's config (all configs agree) and loaded with the average.
    loaded = ev.load_model(run_dir, names[-1], device=device, precision=options.precision,
                           drop_unknown_config_fields=options.drop_unknown_config_fields)
    model = loaded.model
    info = st.load_run_info(run_dir, names[-1], drop_unknown_config_fields=options.drop_unknown_config_fields)
    batch_size = int(options.batch_size or info.cfg.batch_size)
    cache_dir = Path(options.cache_dir) if options.cache_dir else None
    report: Dict[str, Any] = {
        "run_dir": str(run_dir), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoints": names, "checkpoint_steps": steps, "n_checkpoints": len(names),
        "averaging": ("uniform mean of every floating tensor of the state dict (parameters and floating buffers), "
                      "float64 accumulation; non-floating tensors from the last listed checkpoint"),
        "protocol": {"seq_len": info.seq_len, "eval_stride": info.eval_stride, "batch_size": batch_size,
                     "max_windows": options.max_windows},
        "device": str(device), "precision": options.precision, "notes": list(loaded.notes),
        "streams": {}, "averaged": {}, "inputs": {},
        "recorded": {key: info.summary.get(key) for key in (
            "best_val_ppl", "best_step", "final_val_ce", "final_val_ppl", "test_ce", "test_ppl",
            "test_best_ce", "test_best_ppl", "test_best_step") if key in info.summary},
    }
    datasets: Dict[str, Tuple[Any, np.ndarray]] = {}
    for split in options.splits:
        tokens, record = st.resolve_stream(info, split, cache_dir=cache_dir,
                                           allow_mismatch=options.allow_stream_mismatch)
        dataset = st.eval_windows(tokens, info.seq_len, info.eval_stride)
        indices = st.evenly_spaced(len(dataset), options.max_windows)
        record["protocol"] = {"n_windows": len(dataset), "n_scored_windows": int(indices.size)}
        report["streams"][split] = record
        datasets[split] = (dataset, indices)

    def score(label: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for split, (dataset, indices) in datasets.items():
            with batched_cpu_lu_guard():
                result = ev.protocol_evaluate(model, dataset, indices, batch_size=batch_size, device=device)
            results[split] = {key: (float(value) if isinstance(value, (int, float)) else value)
                              for key, value in result.items()}
            log(f"[average] {label} {split}: CE {result['ce']:.4f} | PPL {result['ppl']:.3f}")
        return results

    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        model.load_state_dict(averaged, strict=True)                  # copied onto the model's device
        report["averaged"] = score("averaged")
        if options.evaluate_inputs:
            for name, path in zip(names, paths):
                state, _ = load_model_bundle(path)
                model.load_state_dict(state, strict=True)
                del state
                report["inputs"][name] = score(name)
    report["seconds"] = time.perf_counter() - started
    _write_json(out_path, report)
    report["out_path"] = str(out_path)
    log(f"[average] wrote {out_path}")
    return report


def _bundle_step(path: Path) -> Optional[int]:
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    step = bundle.get("step") if isinstance(bundle, Mapping) else None
    return int(step) if isinstance(step, int) and not isinstance(step, bool) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (np.integer, np.floating)):
            return plain(value.item())
        return value

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plain(payload), indent=2), encoding="utf-8")
    tmp.replace(path)
