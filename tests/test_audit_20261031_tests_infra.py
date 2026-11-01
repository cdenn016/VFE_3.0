r"""Test-lane infrastructure, audit of October 31, 2026 (remediation package tests-infra).

* A052 (test-infrastructure-1): the triton lane of ``run_cpu_tests.py`` listed six files by hand and missed five
  later Triton-interpreter test files, whose tests skipped in every full CPU run under an interpreter without triton.
  The ``triton`` marker is now derived from the test sources (``tests.pytest_policy``), the runner gives the marked
  tests to an interpreter with triton, and a lane whose interpreter has triton fails a triton skip.
* A133 (test-infrastructure-2): the CUDA-skip guard read the substring "cuda" of the report text, so a strict xfail
  whose message mentions CUDA failed and a skip for "a GPU" passed unnoticed.
* A134 (test-infrastructure-3): ``train()`` and ``seed_everything`` left the uninitialized-memory fill, the
  deterministic-algorithms mode, the cuDNN flags and the thread count changed for every later test.
* A237 (test-infrastructure-4): the device-portable scorer, diagonal-moments, frame-reuse and projection-numerics
  tests are mirrored onto the CUDA lane.
* A379 (test-infrastructure-5): every test starts from seeded global generators.
* A380 (test-infrastructure-6): the inert ``xdist_group`` marks are gone.
* A381 (test-infrastructure-8): every lane prints its skips by reason, and the timeout message names the timeout the
  lane ran with.

No test here runs a lane; the pytest subprocesses only collect or run synthetic files of a few lines.
"""
import ast
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

import check_junit
import run_cpu_tests
from tests import pytest_policy
from tests.conftest import model_process_globals_restored
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

_ROOT = Path(__file__).resolve().parents[1]
# The five Triton-interpreter test files of October 25 and 28 that the hand-kept list of October 18 missed.
_FORMERLY_OMITTED = (
    "tests/test_audit_20261025_cue_result.py",
    "tests/test_audit_20261025_numerics.py",
    "tests/test_audit_20261028_memory_checkpoint.py",
    "tests/test_cue_result_triton_20261025.py",
    "tests/test_perf_bitwise_20261028.py",
)
_FORMERLY_LISTED = (
    "tests/test_kl_decode_triton_20261012.py",
    "tests/test_expected_likelihood_kernel_20261010.py",
    "tests/test_audit_20261016_a2b_decode_bias_reduction.py",
    "tests/test_audit_20261016_a2b_decode_once_differentiable.py",
    "tests/test_audit_20261016_a2b_el_kernel_small_tau.py",
    "tests/test_diag_moments_triton_20261018.py",
)
# The interpreter's environment variable and the "triton unavailable" exit-code name of the interpreter tests, spelled
# in two parts so that this file does not match tests.pytest_policy.TRITON_SOURCE_PATTERN itself.
_INTERPRETER_VARIABLE = "TRITON" + "_INTERPRET"
_UNAVAILABLE_EXIT_CODE = "_NO_" + "TRITON"


def _pytest(*args: str, env: Optional[Dict[str, str]] = None, cwd: Path = _ROOT) -> subprocess.CompletedProcess:
    environment = dict(os.environ, VFE3_TEST_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
    environment.pop(pytest_policy.REQUIRE_TRITON_ENV, None)
    environment.update(env or {})
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
                          cwd=cwd, env=environment, capture_output=True, text=True, timeout=600)


# --------------------------------------------------------------------------------------------- A052


def test_every_test_file_that_runs_the_interpreter_is_in_the_triton_lane() -> None:
    listed = set(run_cpu_tests.TRITON_INTERPRETER_TESTS)
    assert set(_FORMERLY_OMITTED) | set(_FORMERLY_LISTED) <= listed
    for path in sorted((_ROOT / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        # Independent of the policy's pattern: the interpreter's variable or the "triton unavailable" exit code.
        runs_interpreter = any(token in source.replace(f"{_INTERPRETER_VARIABLE}ER", "")
                               for token in (_INTERPRETER_VARIABLE, _UNAVAILABLE_EXIT_CODE))
        if runs_interpreter:
            assert f"tests/{path.name}" in listed, path.name
    for name in listed:
        assert (_ROOT / name).is_file(), name
    assert f"tests/{Path(__file__).name}" not in listed


def test_the_triton_marker_partitions_the_interpreter_files_between_the_lanes() -> None:
    ordinary = "tests/test_cpu_runner_20261018.py"
    files = [*_FORMERLY_OMITTED, ordinary]

    def collected(markers: str) -> set:
        completed = _pytest("-o", "addopts=", "--collect-only", "-q", "-m", markers, *files)
        assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
        return {line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if "::" in line}

    triton = collected(run_cpu_tests.TRITON_LANE_MARKERS)
    fast = collected(f"{run_cpu_tests.CPU_LANE_MARKERS} and not triton")
    assert {node.split("::")[0] for node in triton} == set(_FORMERLY_OMITTED)
    assert {node.split("::")[0] for node in fast} == {ordinary}
    for node in ("tests/test_perf_bitwise_20261028.py::"
                 "test_fused_memory_planes_need_no_initialization_and_unused_grads_stay_unmaterialized",
                 "tests/test_audit_20261028_memory_checkpoint.py::"
                 "test_f01_interpreted_triton_keeps_the_off_axis_variance_and_the_factor_gradient",
                 "tests/test_audit_20261025_numerics.py::"
                 "test_expected_likelihood_kernel_recovers_after_an_all_negative_infinite_tile",
                 "tests/test_audit_20261025_cue_result.py::test_fused_route_f01_and_f02_under_the_triton_interpreter",
                 "tests/test_cue_result_triton_20261025.py::test_kernels_compile_for_sm120"):
        assert node in triton, node


def test_a_triton_skip_fails_only_in_a_lane_that_requires_triton(tmp_path: Path) -> None:
    probe = tmp_path / "test_probe_triton_skip.py"
    probe.write_text("import pytest\n\n"
                     "def test_needs_triton():\n    pytest.skip('triton unavailable in this interpreter')\n\n"
                     "def test_other_skip():\n    pytest.skip('unrelated reason')\n", encoding="utf-8")
    required = _pytest("-p", "tests.pytest_policy", "-q", "-rfs", str(probe),
                       env={pytest_policy.REQUIRE_TRITON_ENV: "1"})
    assert required.returncode == 1, required.stdout + required.stderr
    assert "1 failed, 1 skipped" in required.stdout, required.stdout
    assert pytest_policy.REQUIRE_TRITON_ENV in required.stdout
    plain = _pytest("-p", "tests.pytest_policy", "-q", "-rfs", str(probe))
    assert plain.returncode == 0 and "2 skipped" in plain.stdout, plain.stdout + plain.stderr


@pytest.mark.parametrize("local, separate, allow, expected", [
    # The CPU lane's interpreter lacks triton, TRITON_PYTHON has it: the fast lane leaves the triton-marked tests to
    # the serial triton lane, which fails a triton skip.
    (False, True, False, [("fast", None, True), ("triton", "1", False)]),
    # The launching interpreter has triton: one lane runs everything and fails a triton skip.
    (True, False, False, [("fast", "1", False)]),
    # No interpreter has triton and the user opted out: the triton tests run, and skip, in the fast lane.
    (False, False, True, [("fast", None, False)]),
])
def test_the_runner_gives_the_triton_tests_to_an_interpreter_with_triton(
        monkeypatch: pytest.MonkeyPatch, local: bool, separate: bool, allow: bool,
        expected: List[Tuple[str, Optional[str], bool]]) -> None:
    calls: List[Tuple[str, Optional[str], bool]] = []

    def run_lane(lane: str, environment: Dict[str, str], logical_cpu_count: Any, *, exclude_triton: bool = False
                 ) -> int:
        calls.append((lane, environment.get(pytest_policy.REQUIRE_TRITON_ENV), exclude_triton))
        return 0

    monkeypatch.setattr(run_cpu_tests, "run_lane", run_lane)
    monkeypatch.setattr(run_cpu_tests, "preflight_cpu_lane_workers", lambda count: None)
    monkeypatch.setattr(run_cpu_tests, "local_interpreter_has_triton", lambda: local)
    monkeypatch.setattr(run_cpu_tests, "interpreter_has_triton", lambda python: separate)
    monkeypatch.setattr(run_cpu_tests, "ALLOW_MISSING_TRITON", allow)
    monkeypatch.delenv(pytest_policy.REQUIRE_TRITON_ENV, raising=False)
    assert run_cpu_tests.main() == 0
    assert calls == expected


def test_the_lane_commands_select_by_the_triton_marker(tmp_path: Path) -> None:
    fast = run_cpu_tests.build_cpu_lane_command("fast", tmp_path / "f.xml", 64, exclude_triton=True)
    assert "not cuda and not external and not triton" in fast
    assert run_cpu_tests.CPU_LANE_MARKERS in run_cpu_tests.build_cpu_lane_command("fast", tmp_path / "f.xml", 64)
    triton = run_cpu_tests.build_triton_lane_command("C:/x/python.exe", tmp_path / "t.xml")
    assert triton[triton.index("-m", triton.index("pytest")) + 1] == "triton and not cuda and not external"
    assert set(run_cpu_tests.TRITON_INTERPRETER_TESTS) <= set(triton) and "-n" not in triton
    assert run_cpu_tests.triton_lane_environment({"A": "b"}) == {"A": "b", pytest_policy.REQUIRE_TRITON_ENV: "1"}


# --------------------------------------------------------------------------------------------- A133


def test_the_cuda_skip_guard_leaves_expected_failures_alone_and_reads_gpu_wording(tmp_path: Path) -> None:
    probe = tmp_path / "test_probe_skip_wording.py"
    probe.write_text(
        "import pytest\n\n"
        "@pytest.mark.xfail(strict=True, reason='a known defect')\n"
        "def test_expected_failure_mentions_cuda():\n    assert False, 'the CUDA kernel disagrees'\n\n"
        "def test_needs_a_gpu():\n    pytest.skip('requires a GPU')\n\n"
        "def test_needs_cusolver():\n    pytest.skip('cusolver batched eigensolver unavailable')\n\n"
        "def test_needs_cuda():\n    pytest.skip('requires CUDA')\n\n"
        "def test_other_skip():\n    pytest.skip('unrelated reason')\n", encoding="utf-8")
    completed = _pytest("-p", "tests.pytest_policy", "-q", "-rfsx", str(probe))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "3 failed, 1 skipped, 1 xfailed" in completed.stdout, completed.stdout


# --------------------------------------------------------------------------------------------- A134


def _process_state() -> Dict[str, Any]:
    return {"fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "num_threads": torch.get_num_threads()}


def _train_and_reseed(before: Dict[str, Any]) -> None:
    r"""One tiny ``train()`` step (which turns the uninitialized-memory fill off), then ``seed_everything`` and a thread
    count that move every other flag away from ``before``."""
    from vfe3.runtime import seed_everything
    from vfe3.train import train

    torch.utils.deterministic.fill_uninitialized_memory = True
    cfg = tiny_config(max_steps=1, warmup_steps=0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len))
    train(model, [(tokens, tokens)], cfg, n_steps=1, log_interval=0, eval_interval=0, generate_samples=False)
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    # A worker may start this test with the fill already off (state set at collection or by a wider-scoped fixture
    # outlives the per-test restore), so move it away from ``before`` explicitly.
    torch.utils.deterministic.fill_uninitialized_memory = not before["fill_uninitialized_memory"]
    deterministic = not before["deterministic_algorithms"]
    seed_everything(0, deterministic=deterministic)
    torch.use_deterministic_algorithms(deterministic, warn_only=not before["warn_only"])
    torch.backends.cudnn.deterministic = not before["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = not before["cudnn_benchmark"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.set_num_threads(2 if before["num_threads"] == 1 else 1)
    after = _process_state()
    assert {key for key in before if before[key] == after[key]} == set(), after


def test_training_and_reseeding_leave_no_process_state_behind() -> None:
    before = _process_state()
    with model_process_globals_restored():
        _train_and_reseed(before)
    assert _process_state() == before


# The same through the autouse fixture: the second test runs after the first in this module and must start from the
# state the first one started from.
_PAIR: Dict[str, Dict[str, Any]] = {}


def test_pair_first_trains_and_reseeds() -> None:
    _PAIR["before"] = _process_state()
    _train_and_reseed(_PAIR["before"])


def test_pair_second_starts_from_the_restored_process_state() -> None:
    if "before" not in _PAIR:
        pytest.skip("checks the state test_pair_first_trains_and_reseeds left, which did not run in this session")
    assert _process_state() == _PAIR["before"]


# --------------------------------------------------------------------------------------------- A237


def _device_tests(name: str) -> set:
    tree = ast.parse((_ROOT / "tests" / name).read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            and "device" in {argument.arg for argument in node.args.args}}


class _Item:
    r"""The parts of a collected item the collection policy reads."""

    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid
        self.path = _ROOT / nodeid.split("::")[0]
        self.markers: List[Any] = []

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)


def test_the_scorer_moments_frame_reuse_and_projection_device_tests_run_on_the_cuda_lane(
        monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("test_audit_20261010_scoring.py", "test_frame_reuse_20260906.py",
                 "test_round2_projection_numerics_20261009.py"):
        tests = _device_tests(name)
        assert tests
        assert {f"{name}::{test}" for test in tests} <= pytest_policy.CUDA_MIRROR_TESTS, name
    required = pytest_policy.required_cuda_node_keys()
    for key in pytest_policy.CUDA_MIRROR_TESTS:
        name, test = key.split("::")
        assert test in _device_tests(name), key                    # an existing test that takes the device fixture
        assert key in required
    node = "tests/test_audit_20261010_scoring.py::test_factor_route_tile_runs_ieee_fp32"
    for lane, marked in (("cpu", False), ("cuda", True)):
        monkeypatch.setenv("VFE3_TEST_DEVICE", lane)
        item = _Item(node)
        pytest_policy.pytest_collection_modifyitems(SimpleNamespace(), [item])
        assert any(marker.name == "cuda" for marker in item.markers) is marked, lane


# --------------------------------------------------------------------------------------------- A379


def test_every_test_starts_from_the_seeded_global_generators() -> None:
    assert torch.equal(torch.rand(4), torch.rand(4, generator=torch.Generator().manual_seed(0)))
    assert np.array_equal(np.random.rand(3), np.random.RandomState(0).rand(3))
    assert random.random() == random.Random(0).random()


# --------------------------------------------------------------------------------------------- A380


def test_no_test_carries_an_xdist_group_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not hasattr(pytest_policy, "RESOURCE_GROUPS")
    markers = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "xdist_group" not in markers and '"triton: ' in markers
    for lane in ("cpu", "cuda"):
        monkeypatch.setenv("VFE3_TEST_DEVICE", lane)
        items = [_Item("tests/test_cuda_training_lane_20261010.py::test_cuda_evaluate_matches_cpu_evaluate"),
                 _Item("tests/test_divergence.py::test_safe_kl_clamp_bounds_and_nan"),
                 _Item("tests/test_predictive_multilayer.py::test_local_projected_config_training_step[cuda]")]
        pytest_policy.pytest_collection_modifyitems(SimpleNamespace(), items)
        assert all(marker.name != "xdist_group" for item in items for marker in item.markers)


# --------------------------------------------------------------------------------------------- A381


_JUNIT = ('<testsuites><testsuite tests="4" failures="0" errors="0" skipped="3">'
          '<testcase classname="a" name="t1"><skipped type="pytest.skip" message="triton unavailable"/></testcase>'
          '<testcase classname="a" name="t2"><skipped type="pytest.skip" message="triton unavailable"/></testcase>'
          '<testcase classname="a" name="t3"><skipped type="pytest.skip" message="cache absent&#10;detail"/>'
          '</testcase><testcase classname="a" name="t4"/></testsuite></testsuites>')


def test_junit_skips_are_counted_by_reason(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(_JUNIT, encoding="utf-8")
    assert check_junit.read_junit_skip_reasons(path) == {"triton unavailable": 2, "cache absent": 1}
    assert run_cpu_tests.skip_summary({"a": 1, "b": 3, "c": 2}, limit=2) == [
        "      3 x b", "      2 x c", "      1 x (other reasons)"]


def test_a_lane_prints_its_skips_and_the_timeout_it_ran_with(monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture) -> None:
    def finished(command: List[str], *, env: Dict[str, str], timeout: float) -> subprocess.CompletedProcess:
        junit = next(argument for argument in command if argument.startswith("--junitxml="))
        Path(junit.split("=", 1)[1]).write_text(_JUNIT, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_cpu_tests, "run_process_tree", finished)
    assert run_cpu_tests.run_lane("fast", {}, 64) == 0
    out = capsys.readouterr().out
    assert "fast: skips by reason:" in out and "2 x triton unavailable" in out and "1 x cache absent" in out

    seen: Dict[str, float] = {}

    def timed_out(command: List[str], *, env: Dict[str, str], timeout: float) -> subprocess.CompletedProcess:
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(run_cpu_tests, "run_process_tree", timed_out)
    assert run_cpu_tests.run_lane("triton", {}, 64) == 1
    err = capsys.readouterr().err
    assert seen["timeout"] == run_cpu_tests.LANE_TIMEOUT_SECONDS["triton"]
    assert f"triton: pytest subprocess exceeded {seen['timeout']} seconds" in err and "None" not in err
