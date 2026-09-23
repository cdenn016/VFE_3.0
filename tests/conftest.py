import os
import sys
from collections.abc import Mapping, MutableMapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator

# MKL and the OpenMP runtime torch ships (libiomp) coexist here, and that duplicate-runtime state
# makes Intel's LAPACK abort the ENTIRE interpreter on otherwise ordinary calls: np.polyfit on two
# weighted points (vfe3/viz/figures.py _fit_power_law) and scipy's gaussian_kde bandwidth estimator
# (plot_belief_umap) both aborted mid-suite, killing a full run with no pytest summary at all -- so
# the failures it had already found were never reported. The repository's own drivers (train_vfe3.py,
# ablation.py, scaling.py, make_figures.py) already set this, each behind VFE3_ALLOW_DUPLICATE_OPENMP=1.
# A test run that aborts reports nothing, so the suite takes it unconditionally; the setting only
# changes the Intel runtime's tolerance of the duplicate, not any computed value. setdefault, so an
# explicit KMP_DUPLICATE_LIB_OK in the environment still wins.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_REQUESTED_DEVICE_NAME = os.environ.get("VFE3_TEST_DEVICE", "cpu")
_CUBLAS_WORKSPACE_CONFIG_WAS_PRESENT = "CUBLAS_WORKSPACE_CONFIG" in os.environ
_PREVIOUS_CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def _requests_cuda(name: str) -> bool:
    return name.partition(":")[0].lower() == "cuda"


def _validate_cuda_preimport(
    requested_device_name: str,
    environment:           Mapping[str, str],
    loaded_modules:        Mapping[str, object],
) -> None:
    if not _requests_cuda(requested_device_name):
        return
    if (
        "torch" in loaded_modules
        and environment.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before importing torch "
            "for CUDA tests"
        )


_validate_cuda_preimport(_REQUESTED_DEVICE_NAME, os.environ, sys.modules)
if _requests_cuda(_REQUESTED_DEVICE_NAME):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pytest
import torch


@contextmanager
def _deterministic_cuda_policy(
    torch_module:          Any,
    requested_device_name: str,
) -> Iterator[None]:
    requested_device = torch_module.device(requested_device_name)
    if requested_device.type != "cuda":
        yield
        return

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before importing torch "
            "for CUDA tests"
        )

    algorithms_enabled = torch_module.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch_module.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch_module.backends.cudnn.deterministic
    cudnn_benchmark = torch_module.backends.cudnn.benchmark
    matmul = torch_module.backends.cuda.matmul
    cudnn = torch_module.backends.cudnn
    modern_tf32 = hasattr(matmul, "fp32_precision") and hasattr(
        cudnn,
        "fp32_precision",
    )
    if modern_tf32:
        matmul_tf32 = matmul.fp32_precision
        cudnn_tf32 = cudnn.fp32_precision
    else:
        matmul_tf32 = matmul.allow_tf32
        cudnn_tf32 = cudnn.allow_tf32

    try:
        torch_module.use_deterministic_algorithms(True, warn_only=False)
        cudnn.deterministic = True
        cudnn.benchmark = False
        if modern_tf32:
            matmul.fp32_precision = "ieee"
            cudnn.fp32_precision = "ieee"
        else:
            matmul.allow_tf32 = False
            cudnn.allow_tf32 = False
        yield
    finally:
        torch_module.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        cudnn.deterministic = cudnn_deterministic
        cudnn.benchmark = cudnn_benchmark
        if modern_tf32:
            matmul.fp32_precision = matmul_tf32
            cudnn.fp32_precision = cudnn_tf32
        else:
            matmul.allow_tf32 = matmul_tf32
            cudnn.allow_tf32 = cudnn_tf32


def _start_deterministic_cuda_policy(
    torch_module:          Any,
    requested_device_name: str,
) -> AbstractContextManager[None]:
    """Enter the process CUDA policy before pytest begins collection."""
    policy_context = _deterministic_cuda_policy(torch_module, requested_device_name)
    policy_context.__enter__()
    return policy_context


class _DeterministicCudaPolicyLifecycle:
    def __init__(
        self,
        policy_context:        AbstractContextManager[None],
        environment:           MutableMapping[str, str],
        requested_device_name: str,
        cublas_was_present:    bool,
        previous_cublas_value: str | None,
    ) -> None:
        self._policy_context = policy_context
        self._environment = environment
        self._requested_device_name = requested_device_name
        self._cublas_was_present = cublas_was_present
        self._previous_cublas_value = previous_cublas_value
        self._closed = False

    def close(self) -> None:
        """Close and restore process state once across all pytest exit paths."""
        if self._closed:
            return
        self._closed = True
        try:
            self._policy_context.__exit__(None, None, None)
        finally:
            if _requests_cuda(self._requested_device_name):
                if (
                    self._cublas_was_present
                    and self._previous_cublas_value is not None
                ):
                    self._environment["CUBLAS_WORKSPACE_CONFIG"] = (
                        self._previous_cublas_value
                    )
                else:
                    self._environment.pop("CUBLAS_WORKSPACE_CONFIG", None)


_DETERMINISTIC_CUDA_POLICY_CONTEXT = _start_deterministic_cuda_policy(
    torch,
    _REQUESTED_DEVICE_NAME,
)
_DETERMINISTIC_CUDA_POLICY_LIFECYCLE = _DeterministicCudaPolicyLifecycle(
    _DETERMINISTIC_CUDA_POLICY_CONTEXT,
    os.environ,
    _REQUESTED_DEVICE_NAME,
    _CUBLAS_WORKSPACE_CONFIG_WAS_PRESENT,
    _PREVIOUS_CUBLAS_WORKSPACE_CONFIG,
)

@pytest.fixture(scope="session", autouse=True)
def deterministic_cuda_policy() -> Iterator[None]:
    try:
        yield
    finally:
        _DETERMINISTIC_CUDA_POLICY_LIFECYCLE.close()


def pytest_unconfigure(config: pytest.Config) -> None:
    del config
    _DETERMINISTIC_CUDA_POLICY_LIFECYCLE.close()


@pytest.fixture
def device() -> torch.device:
    # Tests are device-agnostic; default CPU for portability.
    # Set VFE3_TEST_DEVICE=cuda to run on the GPU.
    name = os.environ.get("VFE3_TEST_DEVICE", "cpu")
    resolved_device = torch.device(name)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return resolved_device
