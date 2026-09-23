r"""Click-to-run: generate the single-run publication figures for a trained run.

Edit ``CONFIG`` below and run. Points at a run directory written by ``train_vfe3.py``
(``config.json`` + ``best_model.pt``), rebuilds the trained model, drives the
:mod:`vfe3.viz.extract` runners + :mod:`vfe3.metrics` measurements, writes model-replay figures to
``<run_dir>/figures/``, and rebuilds persisted-data history/probe figures at ``<run_dir>/``. With
``run_dir=None`` the most recent run under ``RUN_ROOT`` is used.

Training's ``finalize_run`` already auto-runs the publication set. This driver re-runs the
model-replay set and every history/probe figure supported by the saved run. The single
``generate_embedding_figures`` switch controls only the slow UMAP/category-separation family;
standard figures and data are always attempted.
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
from pathlib import Path
from typing import List, Mapping, Optional

import torch

from vfe3.process_utils import run_process_tree
from vfe3.run_artifacts import _unique_sibling_temp
from vfe3.viz.run_loading import (
    load_best_model_state,
    load_run_config,
    load_run_selection_contract,
)

RUN_ROOT = "vfe3_runs"
_FIGURE_TIMEOUT_SECONDS = 2 * 60 * 60

CONFIG = {
    "run_dir":       None,                                  # None -> newest run under RUN_ROOT
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "split":         "validation",                          # split the representative batch is drawn from
    "max_sequences": 256,                                   # belief-bank size for the UMAP triptych
    "n_e_steps":     None,                                  # E-step trace length (None -> trained cfg.n_e_steps)
    "generate_embedding_figures": False,                    # slow UMAP/category-separation family
}


def _validated_run_dir(path: Path) -> Path:
    r"""Return one finalized, checkpoint-backed run directory or raise a precise error."""
    required = ("config.json", "summary.json", "best_model.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"run directory {path} is incomplete; missing {', '.join(missing)}"
        )
    if (path / "best_model.pt").stat().st_size == 0:
        raise ValueError(f"run directory {path} has an empty best_model.pt")
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
    try:
        from vfe3.model.model import VFEModel

        cfg, _dataset = load_run_config(path)
        expected_code_identity, expected_selection_data_identity = (
            load_run_selection_contract(path))
        with torch.random.fork_rng(devices=[]):
            validation_model = VFEModel(cfg)
            model_state = load_best_model_state(
                path / "best_model.pt",
                cfg,
                validation_model.state_dict(),
                expected_code_identity,
                expected_selection_data_identity,
                map_location="cpu",
            )
            validation_model.load_state_dict(model_state, strict=True)
    except Exception as exc:
        raise ValueError(f"run directory {path} has an invalid best_model.pt") from exc
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


def _generate_figures_isolated(
    run_dir: Path,

    *,
    device:        str,
    split:         str,
    max_sequences: Optional[int],
    n_e_steps:     Optional[int],
    generate_embedding_figures: bool,
) -> List[Path]:
    r"""Run the entire report stack in a disposable child and validate its published outputs."""
    if type(generate_embedding_figures) is not bool:
        raise TypeError("generate_embedding_figures must be bool")
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
            try:
                completed = run_process_tree(
                    [sys.executable, "-m", "vfe3.viz.figure_worker"],
                    cwd=str(Path(__file__).resolve().parent),
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=_FIGURE_TIMEOUT_SECONDS,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"isolated figure generation exceeded {_FIGURE_TIMEOUT_SECONDS} seconds"
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError(f"isolated figure generation could not start: {exc}") from exc
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "no stderr"
                raise RuntimeError(
                    f"isolated figure generation exited with code {completed.returncode}: "
                    f"{detail[-4000:]}"
                )
            if completed.stdout.strip():
                logging.getLogger(__name__).info(completed.stdout.rstrip())
            if completed.stderr.strip():
                logging.getLogger(__name__).info(
                    "isolated figure process diagnostics:\n%s",
                    completed.stderr.rstrip(),
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
