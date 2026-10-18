r"""CPU runner preflight and the shared JUnit reader (audit 2026-10-17 items 40 and 85). No pytest subprocess."""

from pathlib import Path

import pytest

import check_junit
import run_cpu_tests


def _junit(path: Path, tests: int, failures: int = 0) -> Path:
    path.write_text(
        f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8")
    return path


def test_shared_junit_reader_refuses_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no executed tests"):
        check_junit.read_junit_counts(_junit(tmp_path / "empty.xml", 0))
    counts = check_junit.read_junit_counts(_junit(tmp_path / "one.xml", 3, failures=1))
    assert counts["tests"] == 3 and counts["failures"] == 1 and counts["passes"] == 2


def test_cpu_runner_uses_the_shared_reader() -> None:
    assert run_cpu_tests.read_junit_counts is check_junit.read_junit_counts
    assert not hasattr(run_cpu_tests, "_read_junit_counts")


def test_triton_preflight_refuses_without_triton_unless_opted_out() -> None:
    message = run_cpu_tests.triton_preflight(False, False)
    assert message is not None and "ALLOW_MISSING_TRITON" in message
    for path in run_cpu_tests.TRITON_INTERPRETER_TESTS:
        assert path in message
        assert (Path(run_cpu_tests.__file__).parent / path).is_file(), path
    assert run_cpu_tests.triton_preflight(True, False) is None
    assert run_cpu_tests.triton_preflight(False, True) is None
    assert run_cpu_tests.ALLOW_MISSING_TRITON is False


def test_triton_lane_runs_the_interpreter_tests_serially_under_triton_python(tmp_path: Path) -> None:
    command = run_cpu_tests.build_triton_lane_command("C:/x/python.exe", tmp_path / "t.xml")
    assert command[:3] == ["C:/x/python.exe", "-m", "pytest"]
    assert "-n" not in command
    for path in run_cpu_tests.TRITON_INTERPRETER_TESTS:
        assert path in command
    assert run_cpu_tests.interpreter_has_triton(str(tmp_path / "missing.exe")) is False
