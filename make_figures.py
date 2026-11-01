r"""Click-to-run: generate the single-run publication figures for a trained run.

Edit ``CONFIG`` below and run. Points at a run directory written by ``train_vfe3.py``
(``config.json`` + ``summary.json`` + ``best_model.pt``), rebuilds the trained model from the final
weights (``final_model.pt``, or ``best_model.pt`` for older runs without one), drives the
:mod:`vfe3.viz.extract` runners + :mod:`vfe3.metrics` measurements, writes model-replay figures to
``<run_dir>/figures/``, and rebuilds persisted-data history/probe figures at ``<run_dir>/``. With
``run_dir=None`` the most recent run under ``RUN_ROOT`` is used. Run selection checks the files
only; the figure worker performs the full checkpoint validation when it loads the weights.

Training's ``finalize_run`` already auto-runs the publication set. This driver re-runs the
model-replay set and every history/probe figure supported by the saved run. The single
``generate_embedding_figures`` switch adds only the slow semantic-geometry analysis of
``best_model.pt`` (:func:`vfe3.semantics.report.analyze_run` at report budgets, written to
``<run_dir>/figures/semantics/``; ``analyze_semantics.py`` runs the full budgets); standard figures
and data are always attempted.
Historical strict-opt-out runs do not retain the complete per-step loss list needed to reproduce the
original full-resolution ``loss_curve.png``; their cadence-sampled metrics remain available.
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import math
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Deque, List, Mapping, Optional, Sequence, Tuple

import torch

from vfe3.process_utils import spawn_process_tree
from vfe3.run_artifacts import _unique_sibling_temp

RUN_ROOT = "vfe3_runs"
_FIGURE_TIMEOUT_SECONDS = 2 * 60 * 60
_WORKER_CLEANUP_SECONDS = 10.0          # bound on releasing the worker's process tree and joining its reader
_WORKER_OUTPUT_TAIL_LINES = 400         # output lines a failure message can quote

CONFIG = {
    "run_dir":       None,                                  # None -> newest run under RUN_ROOT
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "split":         "validation",                          # split the representative batch is drawn from
    "max_sequences": 256,                                   # report-batch population of the model-replay figures
    "n_e_steps":     None,                                  # ignored (no retained figure reads it); None | positive int
    "generate_embedding_figures": False,                    # slow semantic-geometry analysis (vfe3.semantics)
}


def _validated_run_dir(path: Path) -> Path:
    r"""Return one finalized, checkpoint-backed run directory or raise a precise error."""
    required = ("config.json", "summary.json", "best_model.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"run directory {path} is incomplete; missing {', '.join(missing)}"
        )
    # The figure worker replays final_model.pt when present (vfe3.viz.report.generate_figures);
    # check that file, plus best_model.pt, which the semantic analysis reads.
    checkpoints = ["best_model.pt"]
    if (path / "final_model.pt").is_file():
        checkpoints.append("final_model.pt")
    for name in checkpoints:
        try:
            with open(path / name, "rb") as handle:
                readable = bool(handle.read(1))
        except OSError as exc:
            raise ValueError(f"run directory {path} has unreadable {name}") from exc
        if not readable:
            raise ValueError(f"run directory {path} has an empty {name}")
    config_payload = None
    summary_payload = None
    for name in ("config.json", "summary.json"):
        try:
            payload = json.loads((path / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"run directory {path} has unreadable {name}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"run directory {path} has non-mapping {name}")
        if name == "config.json":
            config_payload = payload
        else:
            summary_payload = payload
    if not isinstance(config_payload.get("config") if config_payload is not None else None, dict):
        raise ValueError(f"run directory {path} config.json has no config mapping")
    if (type(summary_payload.get("n_steps")) is not int
            or summary_payload["n_steps"] < 0
            or type(summary_payload.get("n_params")) is not int
            or summary_payload["n_params"] <= 0):
        raise ValueError(
            f"run directory {path} summary.json has no valid completion counts"
        )
    completion_values = (
        summary_payload.get("test_ce"),
        summary_payload.get("final_val_ce"),
        summary_payload.get("best_val_ppl"),
    )
    if not any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in completion_values):
        raise ValueError(f"run directory {path} summary.json has no finite completion metric")
    return path


def _newest_run(root: str) -> Path:
    r"""Find the newest finalized checkpoint-backed run anywhere under ``root``."""
    candidates = [
        config_path.parent
        for config_path in Path(root).rglob("config.json")
        if all((config_path.parent / name).is_file()
               for name in ("config.json", "summary.json", "best_model.pt"))
    ]
    candidates.sort(
        key=lambda path: (
            max((path / name).stat().st_mtime_ns
                for name in ("config.json", "summary.json", "best_model.pt")),
            str(path.resolve()),
        ),
        reverse=True,
    )
    for candidate in candidates:
        try:
            return _validated_run_dir(candidate)
        except (FileNotFoundError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(
            f"no finalized runs (config.json + summary.json + best_model.pt) under {root!r}; "
            "train one first"
        )
    raise FileNotFoundError(
        f"no valid finalized run under {root!r}; candidate checkpoints failed integrity validation"
    )


def _log_worker_line(line: str) -> None:
    r"""Forward one line of the figure worker's output to this driver's log as it arrives."""
    logging.getLogger(__name__).info("%s", line)


def _run_figure_worker(
    command: Sequence[str],

    *,
    cwd:     str,
    env:     Mapping[str, str],
    timeout: float,
    on_line: Callable[[str], None],
) -> Tuple[int, str]:
    r"""Run the figure worker in a contained process tree, handing each output line to ``on_line`` as it arrives.

    The worker's stdout and stderr share one pipe that a reader thread drains line by line, so a click-to-run report
    shows its progress while it runs, not only after the child exits (audit 2026-10-31 viz-pipeline-12). Returns the
    exit code and the last lines of output, which a failure message quotes. A worker that outlives ``timeout``
    seconds is terminated with its descendants and ``subprocess.TimeoutExpired`` is raised; an interruption also
    terminates the tree.
    """
    tree = spawn_process_tree(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    process = tree.process
    if process.stdin is not None:                  # the containment gate is written; the worker reads no input
        process.stdin.close()
    tail: Deque[str] = deque(maxlen=_WORKER_OUTPUT_TAIL_LINES)

    def drain() -> None:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            tail.append(line)
            try:
                on_line(line)
            except Exception:                      # a failing sink never stalls the pipe
                pass

    reader = threading.Thread(target=drain, name="figure-worker-output", daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout)
    finally:
        # After a normal exit this releases the containment job (and any descendant still alive); after a timeout or
        # an interruption it terminates the tree.
        tree.terminate(reason="figure worker finished or was interrupted", timeout=_WORKER_CLEANUP_SECONDS)
        reader.join(timeout=_WORKER_CLEANUP_SECONDS)
        if process.stdout is not None and not reader.is_alive():
            process.stdout.close()
    return int(returncode), "\n".join(tail)


def _generate_figures_isolated(
    run_dir: Path,

    *,
    device:        str,
    split:         str,
    max_sequences: Optional[int],
    n_e_steps:     Optional[int],
    generate_embedding_figures: bool,
) -> List[Path]:
    r"""Run the entire report stack in a disposable child and validate its published outputs.

    ``run_dir`` is resolved to an absolute path first: the child runs with the repository root as its
    working directory, so a relative path would resolve there, and the worker's ownership check of the
    result file would fail whenever this process was launched elsewhere (audit 2026-10-17 item 67). The
    child's output is logged line by line while it runs (:func:`_run_figure_worker`). ``n_e_steps`` is
    forwarded and ignored by the worker: no retained figure reads it.
    """
    if type(generate_embedding_figures) is not bool:
        raise TypeError("generate_embedding_figures must be bool")
    run_dir = Path(run_dir).resolve()
    with _unique_sibling_temp(run_dir / "figure_request.json") as request_path:
        with _unique_sibling_temp(run_dir / "figure_result.json") as result_path:
            request_path.write_text(json.dumps({
                "mode":          "report",
                "run_dir":       str(run_dir),
                "result_path":   str(result_path),
                "device":        device,
                "split":         split,
                "max_sequences": max_sequences,
                "n_e_steps":     n_e_steps,
                "generate_embedding_figures": generate_embedding_figures,
            }), encoding="utf-8")
            environment = os.environ.copy()
            environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            environment["VFE3_FIGURE_REQUEST"] = str(request_path)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"          # the streamed lines are decoded as UTF-8
            try:
                returncode, output_tail = _run_figure_worker(
                    [sys.executable, "-m", "vfe3.viz.figure_worker"],
                    cwd=str(Path(__file__).resolve().parent),
                    env=environment,
                    timeout=_FIGURE_TIMEOUT_SECONDS,
                    on_line=_log_worker_line,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"isolated figure generation exceeded {_FIGURE_TIMEOUT_SECONDS} seconds"
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"isolated figure generation could not start: {exc}") from exc
            if returncode != 0:
                detail = output_tail.strip() or "no output"
                raise RuntimeError(
                    f"isolated figure generation exited with code {returncode}: "
                    f"{detail[-4000:]}"
                )
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("isolated figure generation returned no valid result") from exc
            raw_paths = payload.get("paths") if isinstance(payload, Mapping) else None
            if not isinstance(raw_paths, list) or any(not isinstance(value, str) for value in raw_paths):
                raise RuntimeError("isolated figure generation returned an invalid path list")
            figure_root = (run_dir / "figures").resolve()
            paths: List[Path] = []
            for raw_path in raw_paths:
                path = Path(raw_path).resolve(strict=True)
                try:
                    path.relative_to(figure_root)
                except ValueError as exc:
                    raise RuntimeError(f"figure worker published outside its figure directory: {path}") from exc
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"figure worker reported an empty output: {path}")
                paths.append(path)
            if len(paths) != len(set(paths)):
                raise RuntimeError("figure worker reported duplicate outputs")
            return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = (
        _validated_run_dir(Path(CONFIG["run_dir"]))
        if CONFIG["run_dir"]
        else _newest_run(RUN_ROOT)
    )
    print(f"\nVFE_3.0 figure generation\n  run_dir: {run_dir}\n  device:  {CONFIG['device']}")
    paths = _generate_figures_isolated(
        run_dir,
        device=str(CONFIG["device"]),
        split=CONFIG["split"],
        max_sequences=CONFIG["max_sequences"],
        n_e_steps=CONFIG["n_e_steps"],
        generate_embedding_figures=CONFIG["generate_embedding_figures"],
    )
    print(f"\nwrote {len(paths)} model-replay figures to {run_dir / 'figures'}")
    print(f"rebuilt persisted-data history/probe figures under {run_dir}")


if __name__ == "__main__":
    main()
