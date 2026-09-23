"""Run the ordinary CPU pytest lanes with bounded parallelism."""

import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

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

_COUNT_FIELDS = ("tests", "failures", "errors", "skipped")


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


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_junit_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    root_name = _local_name(root.tag)
    if root_name not in ("testsuite", "testsuites"):
        raise ValueError(f"unsupported JUnit root element {root_name!r}")

    suites = [element for element in root.iter() if _local_name(element.tag) == "testsuite"]
    if not suites:
        raise ValueError("JUnit XML contains no testsuite elements")

    counts = {field: 0 for field in _COUNT_FIELDS}
    for suite in suites:
        for field in _COUNT_FIELDS:
            if field not in suite.attrib:
                raise ValueError(f"JUnit testsuite lacks the {field!r} count")
            try:
                value = int(suite.attrib[field])
            except ValueError as exc:
                raise ValueError(f"invalid JUnit {field!r} count") from exc
            if value < 0:
                raise ValueError(f"JUnit {field!r} count must be nonnegative")
            counts[field] += value

    if counts["failures"] + counts["errors"] + counts["skipped"] > counts["tests"]:
        raise ValueError("JUnit failure/error/skip counts exceed the test count")
    if counts["tests"] == 0:
        raise ValueError("JUnit XML contains no executed tests")
    return counts


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

        try:
            counts = _read_junit_counts(junit_path)
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

    child_environment = build_cpu_environment(os.environ)
    for lane in RUN_LANES:
        status = run_lane(lane, child_environment, logical_cpu_count)
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
