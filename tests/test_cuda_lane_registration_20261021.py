r"""Every CUDA-only test is selected by the CUDA runner, and the empty slow lane is gone (audit 2026-10-20 item 22)."""
import os
import subprocess
import sys
from pathlib import Path

import run_cpu_tests
from tests import pytest_policy


_ROOT = Path(__file__).resolve().parents[1]

_FORMERLY_UNROUTED = (
    "tests/test_sync_free_matrix_exp_20261012.py::test_cuda_frame_exponential_has_no_host_sync",
    "tests/test_audit_20261016_a2b_decode_bias_reduction.py::"
    "test_cuda_fused_bias_gradient_matches_the_torch_route_under_high_precision",
    "tests/test_predictive_multilayer.py::test_local_projected_config_training_step[cuda]",
)


def _pytest(*args: str, cwd: Path = _ROOT) -> subprocess.CompletedProcess:
    env = dict(os.environ, VFE3_TEST_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def test_formerly_unrouted_cuda_tests_are_registered_and_required() -> None:
    required = pytest_policy.required_cuda_node_keys()
    for nodeid in _FORMERLY_UNROUTED:
        assert pytest_policy.is_registered_cuda_test(nodeid), nodeid
        assert pytest_policy.node_key(nodeid) in required, nodeid
    # The CPU parametrization stays in the CPU lane.
    assert not pytest_policy.is_registered_cuda_test(
        "tests/test_predictive_multilayer.py::test_local_projected_config_training_step[cpu]")


def test_cuda_marker_selects_the_formerly_unrouted_tests() -> None:
    files = sorted({nodeid.split("::")[0] for nodeid in _FORMERLY_UNROUTED})
    completed = _pytest("-o", "addopts=", "--collect-only", "-q", "-m", "cuda", *files)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    selected = {line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if "::" in line}
    assert set(_FORMERLY_UNROUTED) <= selected, completed.stdout
    assert "tests/test_predictive_multilayer.py::test_local_projected_config_training_step[cpu]" not in selected


def test_an_unregistered_cuda_skip_fails_outside_the_cuda_lane(tmp_path: Path) -> None:
    probe = tmp_path / "test_probe_unregistered_cuda.py"
    probe.write_text(
        "import pytest\n\n"
        "def test_needs_gpu():\n    pytest.skip('requires CUDA')\n\n"
        "def test_other_skip():\n    pytest.skip('unrelated reason')\n",
        encoding="utf-8")
    completed = _pytest("-p", "tests.pytest_policy", "-q", "-rfs", str(probe))
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "1 failed, 1 skipped" in completed.stdout, completed.stdout
    assert "CUDA_TESTS" in completed.stdout


def test_the_cpu_runner_has_no_slow_lane(tmp_path: Path) -> None:
    assert run_cpu_tests.RUN_LANES == ("fast",)
    assert not hasattr(run_cpu_tests, "SLOW_WORKERS")
    command = run_cpu_tests.build_cpu_lane_command("fast", tmp_path / "f.xml", 64)
    assert "--runslow" not in command and "not cuda and not external" in command
    assert not hasattr(pytest_policy, "SLOW_TESTS")
