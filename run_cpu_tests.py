"""Run the ordinary CPU pytest lanes with bounded parallelism.

Run this file under ``$CUDA_PYTHON`` (``C:/anaconda/python.exe``): ``CPU_ENVIRONMENT`` hides CUDA from
the lanes, and that interpreter has triton, which the Triton interpreter tests of the live decode kernel
need. Under an interpreter without triton those tests skip, so the runner refuses to start unless
``ALLOW_MISSING_TRITON`` is set (audit 2026-10-17 item 40).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

from check_junit import read_junit_counts
from vfe3.process_utils import run_process_tree


FAST_WORKERS = 12
SLOW_WORKERS = 3
RUN_LANES = ("fast", "slow")
LANE_TIMEOUT_SECONDS = {
    "fast": 2 * 60 * 60,
    "slow": 8 * 60 * 60,
}

CPU_ENVIRONMENT = {
    "VFE3_TEST_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

# Click-to-run opt-out: True runs the lanes under an interpreter without triton, accepting that the
# Triton interpreter tests below skip and the live decode kernel's gradients go unchecked.
ALLOW_MISSING_TRITON = False                    # False (default) | True
TRITON_INTERPRETER_TESTS = (
    "tests/test_kl_decode_triton_20261012.py",
    "tests/test_expected_likelihood_kernel_20261010.py",
    "tests/test_audit_20261016_a2b_decode_bias_reduction.py",
    "tests/test_audit_20261016_a2b_decode_once_differentiable.py",
    "tests/test_audit_20261016_a2b_el_kernel_small_tau.py",
)


def triton_preflight(
    allow_missing: bool,
    triton_found:  bool,
) -> "str | None":
    r"""The refusal message when triton is absent and not explicitly allowed, else ``None``."""
    if triton_found or allow_missing:
        return None
    listed = "\n".join(f"  {path}" for path in TRITON_INTERPRETER_TESTS)
    return (f"triton is not importable under {sys.executable}; the Triton interpreter tests would "
            f"skip:\n{listed}\nRun under $CUDA_PYTHON (C:/anaconda/python.exe) or set "
            "ALLOW_MISSING_TRITON = True to run without them.")


def resolve_cpu_workers(
    configured_workers: int,
    logical_cpu_count:  int | None,
) -> int:
    """Validate a fixed worker count against the available logical CPUs."""
    if isinstance(configured_workers, bool) or not isinstance(configured_workers, int):
        raise TypeError("configured worker count must be a non-bool integer")
    if configured_workers <= 0:
        raise ValueError("configured worker count must be positive")
    if (
        isinstance(logical_cpu_count, bool)
        or not isinstance(logical_cpu_count, int)
        or logical_cpu_count <= 0
    ):
        raise ValueError("available logical CPU count must be a positive integer")
    if configured_workers > logical_cpu_count:
        raise ValueError(
            f"configured worker count {configured_workers} exceeds "
            f"{logical_cpu_count} available logical CPUs"
        )
    return configured_workers


def build_cpu_environment(parent_environment: Mapping[str, str]) -> dict[str, str]:
    """Copy the parent environment and apply CPU-lane overrides."""
    child_environment = dict(parent_environment)
    child_environment.update(CPU_ENVIRONMENT)
    return child_environment


def preflight_cpu_lane_workers(logical_cpu_count: int | None) -> None:
    """Validate every configured CPU lane against one topology snapshot."""
    for lane, configured_workers in (
        ("fast", FAST_WORKERS),
        ("slow", SLOW_WORKERS),
    ):
        try:
            resolve_cpu_workers(configured_workers, logical_cpu_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{lane} lane worker configuration is invalid: {exc}") from exc


def build_cpu_lane_command(
    lane:              str,
    junit_path:        Path,
    logical_cpu_count: int | None,
) -> list[str]:
    """Build one fixed CPU-lane pytest command."""
    if lane == "fast":
        workers = resolve_cpu_workers(FAST_WORKERS, logical_cpu_count)
        lane_arguments = [
            "-n",
            str(workers),
            "--dist",
            "loadscope",
            "-m",
            "not slow and not cuda and not external",
        ]
    elif lane == "slow":
        workers = resolve_cpu_workers(SLOW_WORKERS, logical_cpu_count)
        lane_arguments = [
            "--runslow",
            "-n",
            str(workers),
            "--dist",
            "loadgroup",
            "-m",
            "slow and not cuda and not external",
        ]
    else:
        raise ValueError(f"unknown CPU test lane {lane!r}")

    return [
        sys.executable,
        "-m",
        "pytest",
        *lane_arguments,
        f"--junitxml={junit_path}",
    ]


def run_lane(
    lane:              str,
    environment:       Mapping[str, str],
    logical_cpu_count: int | None,
) -> int:
    """Run one lane in a fresh subprocess and validate its JUnit result."""
    handle = tempfile.NamedTemporaryFile(
        prefix=f"vfe3-{lane}-",
        suffix=".xml",
        dir=tempfile.gettempdir(),
        delete=False,
    )
    junit_path = Path(handle.name)
    handle.close()
    junit_path.unlink(missing_ok=True)

    try:
        command = build_cpu_lane_command(lane, junit_path, logical_cpu_count)
        try:
            completed = run_process_tree(
                command,
                env=dict(environment),
                timeout=LANE_TIMEOUT_SECONDS[lane],
            )
        except subprocess.TimeoutExpired as exc:
            print(
                f"{lane}: pytest subprocess exceeded {LANE_TIMEOUT_SECONDS[lane]} seconds: {exc}",
                file=sys.stderr,
            )
            return 1
        except OSError as exc:
            print(f"{lane}: pytest subprocess failed: {exc}", file=sys.stderr)
            return 1

        if lane == "slow" and completed.returncode == 5:
            # pytest exit 5: no test carries the slow marker, so the lane has nothing to run.
            print(f"{lane}: no tests selected")
            return 0
        try:
            counts = read_junit_counts(junit_path)
        except (OSError, ElementTree.ParseError, ValueError) as exc:
            print(f"{lane}: invalid JUnit result: {exc}", file=sys.stderr)
            return int(completed.returncode) or 1

        print(
            f"{lane}: tests={counts['tests']} failures={counts['failures']} "
            f"errors={counts['errors']} skipped={counts['skipped']}"
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        if counts["failures"] or counts["errors"]:
            return 1
        return 0
    finally:
        junit_path.unlink(missing_ok=True)


def main() -> int:
    """Run the configured CPU lanes in order, stopping at the first failure."""
    logical_cpu_count = os.cpu_count()
    try:
        preflight_cpu_lane_workers(logical_cpu_count)
    except (TypeError, ValueError) as exc:
        print(f"CPU lane preflight failed: {exc}", file=sys.stderr)
        return 1
    refusal = triton_preflight(ALLOW_MISSING_TRITON, importlib.util.find_spec("triton") is not None)
    if refusal is not None:
        print(f"CPU lane preflight failed: {refusal}", file=sys.stderr)
        return 1

    child_environment = build_cpu_environment(os.environ)
    for lane in RUN_LANES:
        status = run_lane(lane, child_environment, logical_cpu_count)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
