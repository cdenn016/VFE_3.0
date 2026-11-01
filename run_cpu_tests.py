"""Run the ordinary CPU pytest lanes with bounded parallelism.

The parallel lanes need pytest-xdist and run under the interpreter that launches this file. The tests that need
triton carry the ``triton`` marker, which ``tests/pytest_policy.py`` derives from the test sources: every
``tests/test_*.py`` that runs the Triton interpreter or tests for triton (``TRITON_INTERPRETER_TESTS``, discovered,
not listed; audit 2026-10-31 test-infrastructure-1). When the launching interpreter has triton the fast lane runs
them with the rest. Otherwise the fast lane deselects them and a final serial ``triton`` lane runs them under
``TRITON_PYTHON`` (``$CUDA_PYTHON``, default ``C:/anaconda/python.exe``) with ``CPU_ENVIRONMENT`` hiding CUDA. A lane
whose interpreter has triton sets ``VFE3_REQUIRE_TRITON=1``, so a test that skips for want of triton fails there
rather than passing as a skip. If no interpreter with triton is available the run fails unless
``ALLOW_MISSING_TRITON`` is set (audit 2026-10-17 item 40). Every lane prints its skips grouped by reason (audit
2026-10-31 test-infrastructure-8).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

from check_junit import read_junit_counts, read_junit_skip_reasons
from tests.pytest_policy import REQUIRE_TRITON_ENV, triton_test_files
from vfe3.process_utils import run_process_tree


FAST_WORKERS = 12
# The former slow lane was removed (audit 2026-10-20 item 22): no test carried the slow marker, so it never ran one.
RUN_LANES = ("fast",)
LANE_TIMEOUT_SECONDS = {
    "fast": 2 * 60 * 60,
    "triton": 2 * 60 * 60,
}
CPU_LANE_MARKERS = "not cuda and not external"
# The skip reasons a lane prints, most frequent first (the rest are summed in one line).
SKIP_SUMMARY_REASONS = 20

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
TRITON_PYTHON = os.environ.get("CUDA_PYTHON", "C:/anaconda/python.exe")   # interpreter with triton for the triton lane
# The test files whose tests carry the triton marker, discovered from their sources by tests/pytest_policy.py.
TRITON_INTERPRETER_TESTS = triton_test_files()
TRITON_LANE_MARKERS = f"triton and {CPU_LANE_MARKERS}"


def triton_preflight(
    allow_missing: bool,
    triton_found:  bool,
) -> "str | None":
    r"""The refusal message when no interpreter with triton is available and not explicitly allowed, else ``None``."""
    if triton_found or allow_missing:
        return None
    listed = "\n".join(f"  {path}" for path in TRITON_INTERPRETER_TESTS)
    return (f"triton is importable neither under {sys.executable} nor under TRITON_PYTHON={TRITON_PYTHON}; the "
            f"Triton interpreter tests would not run:\n{listed}\nPoint $CUDA_PYTHON at an interpreter with triton "
            "or set ALLOW_MISSING_TRITON = True to run without them.")


def local_interpreter_has_triton() -> bool:
    r"""Whether the interpreter running this file can import triton."""
    return importlib.util.find_spec("triton") is not None


def interpreter_has_triton(python: str) -> bool:
    r"""Whether ``python`` exists and can import triton (``sys.executable`` is checked in process)."""
    if os.path.normcase(os.path.abspath(python)) == os.path.normcase(os.path.abspath(sys.executable)):
        return importlib.util.find_spec("triton") is not None
    if not Path(python).is_file():
        return False
    probe = subprocess.run([python, "-c", "import importlib.util,sys; sys.exit(importlib.util.find_spec('triton') is None)"],
                           capture_output=True, timeout=120)
    return probe.returncode == 0


def build_triton_lane_command(python: str, junit_path: Path) -> list[str]:
    r"""The serial triton lane: the ``triton``-marked tests of ``TRITON_INTERPRETER_TESTS`` under ``python``, no xdist."""
    return [python, "-m", "pytest", "-p", "no:cacheprovider", "-m", TRITON_LANE_MARKERS, *TRITON_INTERPRETER_TESTS,
            f"--junitxml={junit_path}"]


def triton_lane_environment(environment: Mapping[str, str]) -> dict[str, str]:
    r"""``environment`` for a lane whose interpreter has triton: a test that skips for want of triton fails."""
    return {**environment, REQUIRE_TRITON_ENV: "1"}


def skip_summary(reasons: Mapping[str, int], limit: int = SKIP_SUMMARY_REASONS) -> list[str]:
    r"""Lines that list a lane's skips by reason, most frequent first; reasons beyond ``limit`` are summed."""
    ordered = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
    lines = [f"  {count:5d} x {reason}" for reason, count in ordered[:limit]]
    rest = sum(count for _, count in ordered[limit:])
    if rest:
        lines.append(f"  {rest:5d} x (other reasons)")
    return lines


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
    ):
        try:
            resolve_cpu_workers(configured_workers, logical_cpu_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{lane} lane worker configuration is invalid: {exc}") from exc


def build_cpu_lane_command(
    lane:              str,
    junit_path:        Path,
    logical_cpu_count: int | None,

    *,
    exclude_triton:    bool = False,          # True when a separate triton lane runs the triton-marked tests
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
            f"{CPU_LANE_MARKERS} and not triton" if exclude_triton else CPU_LANE_MARKERS,
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

    *,
    exclude_triton:    bool = False,          # build_cpu_lane_command(exclude_triton=...)
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
        command = (build_triton_lane_command(TRITON_PYTHON, junit_path) if lane == "triton"
                   else build_cpu_lane_command(lane, junit_path, logical_cpu_count, exclude_triton=exclude_triton))
        timeout = LANE_TIMEOUT_SECONDS.get(lane, LANE_TIMEOUT_SECONDS["fast"])
        try:
            completed = run_process_tree(
                command,
                env=dict(environment),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            print(
                f"{lane}: pytest subprocess exceeded {timeout} seconds: {exc}",
                file=sys.stderr,
            )
            return 1
        except OSError as exc:
            print(f"{lane}: pytest subprocess failed: {exc}", file=sys.stderr)
            return 1

        try:
            counts = read_junit_counts(junit_path)
            skips = read_junit_skip_reasons(junit_path)
        except (OSError, ElementTree.ParseError, ValueError) as exc:
            print(f"{lane}: invalid JUnit result: {exc}", file=sys.stderr)
            return int(completed.returncode) or 1

        print(
            f"{lane}: tests={counts['tests']} failures={counts['failures']} "
            f"errors={counts['errors']} skipped={counts['skipped']}"
        )
        if skips:
            print(f"{lane}: skips by reason:")
            for line in skip_summary(skips):
                print(line)
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
    local_triton = local_interpreter_has_triton()
    separate_triton = (not local_triton) and interpreter_has_triton(TRITON_PYTHON)
    refusal = triton_preflight(ALLOW_MISSING_TRITON, local_triton or separate_triton)
    if refusal is not None:
        print(f"CPU lane preflight failed: {refusal}", file=sys.stderr)
        return 1

    child_environment = build_cpu_environment(os.environ)
    # The launching interpreter has triton: its lanes run the triton-marked tests and fail a triton skip. Otherwise
    # they leave those tests to the serial triton lane, which runs them under TRITON_PYTHON with the same rule.
    lane_environment = triton_lane_environment(child_environment) if local_triton else child_environment
    for lane in RUN_LANES:
        status = run_lane(lane, lane_environment, logical_cpu_count, exclude_triton=separate_triton)
        if status != 0:
            return status
    if separate_triton:
        return run_lane("triton", triton_lane_environment(child_environment), logical_cpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
