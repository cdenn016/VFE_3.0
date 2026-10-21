r"""Click-to-run for the tests that need the GPU.

Runs every test selected by the canonical ``cuda`` marker, including CUDA-only tests and the
ordinary numerical contracts mirrored onto CUDA by the collection policy. The former t3
learnability-gate stage was removed: the helpers it imported from ``tests/test_train.py`` were
deleted in PR #32.

No CLI args::

    python check_gpu_tests.py
"""

import os

os.environ["VFE3_TEST_DEVICE"] = "cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import time
from pathlib import Path

import torch

from check_junit import junit_is_exact_all_pass, run_pytest_junit
from tests.pytest_policy import required_cuda_node_keys


REQUIRED_CUDA_TEST_KEYS = required_cuda_node_keys()
# Each run's JUnit report (record_property parity errors and kernel timings) is kept here (gitignored; audit
# 2026-10-20 item 15).
REPORT_DIR = Path(__file__).resolve().parent / ".verification" / "gpu_tests"


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available in this environment -- these tests NEED the GPU.")
        print("Run this on the RTX 5090 (and make sure torch was installed with CUDA support).")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}\n")

    print("CUDA marker lane -- canonical CUDA-only and mirrored tests:")
    report = REPORT_DIR / f"junit_cuda_{time.strftime('%Y%m%d_%H%M%S')}.xml"
    cuda_code, cuda_counts = run_pytest_junit(
        ["-m", "cuda", "-v", "-s", "-p", "no:cacheprovider"],
        prefix="vfe3-gpu-cuda-",
        required_node_keys=REQUIRED_CUDA_TEST_KEYS,
        keep_report=report,
    )
    print(f"JUnit report with the recorded parity errors and timings kept at {report}")
    cuda_ok = cuda_code == 0 and junit_is_exact_all_pass(
        cuda_counts,
        expected_tests=cuda_counts["collected"],
    )

    bar = "=" * 64
    print("\n" + bar)
    cuda_status = "GREEN" if cuda_ok else f"FAIL (exit {cuda_code})"
    print(
        f"GPU CHECK: CUDA {cuda_status} "
        f"({cuda_counts['passes']} passed, {cuda_counts['skipped']} skipped, "
        f"{cuda_counts['failures']} failed, {cuda_counts['errors']} errors)"
    )
    print(bar)
    return 0 if cuda_ok else 1


if __name__ == "__main__":
    sys.exit(main())
