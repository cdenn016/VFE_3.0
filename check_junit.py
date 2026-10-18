"""Shared machine-readable pytest accounting for click-to-run verification scripts."""

from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional, Sequence, AbstractSet
from xml.etree import ElementTree


_COUNT_FIELDS = ("tests", "failures", "errors", "skipped")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_junit_counts(path: Path) -> Dict[str, int]:
    """Return aggregate pytest counts from either supported JUnit root form.

    Raises ``ValueError`` on a malformed report and on one that records zero executed tests, so an
    empty run can never read as a pass. The one JUnit reader, shared by ``run_cpu_tests.py`` and
    ``check_gpu_tests.py`` (audit 2026-10-17 item 85).
    """
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
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid JUnit {field!r} count") from exc
            if value < 0:
                raise ValueError(f"JUnit {field!r} count must be nonnegative")
            counts[field] += value

    counts["passes"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if counts["passes"] < 0:
        raise ValueError("JUnit failure/error/skip counts exceed the test count")
    if counts["tests"] == 0:
        raise ValueError("JUnit XML contains no executed tests")
    return counts


def junit_is_exact_all_pass(
    counts: Mapping[str, int],

    *,
    expected_tests: int,
) -> bool:
    """Return whether JUnit proves the exact expected cohort passed in full."""
    if (
        isinstance(expected_tests, bool)
        or not isinstance(expected_tests, int)
        or expected_tests <= 0
    ):
        return False

    fields = ("tests", "passes", "failures", "errors", "skipped")
    try:
        values = {field: counts[field] for field in fields}
    except KeyError:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values.values()
    ):
        return False
    if values["tests"] != sum(values[field] for field in fields[1:]):
        return False
    return (
        values["tests"] == expected_tests
        and values["passes"] == expected_tests
        and values["failures"] == 0
        and values["errors"] == 0
        and values["skipped"] == 0
    )


def run_pytest_junit(
    args: Sequence[str],

    *,
    prefix: str,
    required_node_keys: Optional[AbstractSet[str]] = None,
) -> tuple[int, Dict[str, int]]:
    """Run pytest with a temporary JUnit report, return its exit code and derived counts."""
    cli_args = [str(arg) for arg in args]
    if any(arg.startswith(("--junitxml", "--junit-xml")) for arg in cli_args):
        raise ValueError("run_pytest_junit owns the temporary --junitxml argument")

    handle = NamedTemporaryFile(prefix=prefix, suffix=".xml", delete=False)
    junit_path = Path(handle.name)
    handle.close()
    try:
        import pytest

        class SelectedCohort:
            count: int = 0

            def pytest_collection_finish(self, session: pytest.Session) -> None:
                self.count = len(session.items)
                keys = {item.nodeid.replace("\\", "/").split("/")[-1].split("[")[0]
                        for item in session.items}
                missing = set(required_node_keys or ()) - keys
                if missing:
                    raise pytest.UsageError(f"required test coverage was not collected: {sorted(missing)}")

        cohort = SelectedCohort()
        plugins = [cohort] if required_node_keys is not None else []
        code = int(pytest.main([*cli_args, f"--junitxml={junit_path}"], plugins=plugins))
        counts = read_junit_counts(junit_path)
        if required_node_keys is not None:
            counts["collected"] = cohort.count
        return code, counts
    finally:
        junit_path.unlink(missing_ok=True)
