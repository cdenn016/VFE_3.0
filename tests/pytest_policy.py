"""Collection policy for the test lanes: --runslow and the slow, cuda, external and xdist_group markers.

Registered as a pytest plugin by the repository-root ``conftest.py`` (``pytest_plugins``), so its
hooks run for every invocation. It must not import torch: it loads before ``tests/conftest.py`` sets
the CUDA workspace environment that has to precede the first torch import.
"""
import os

import pytest


SLOW_TESTS: frozenset[str] = frozenset()

CUDA_TESTS: frozenset[str] = frozenset({
    # cuSOLVER batched-eigensolver ceiling guards, which skip without CUDA (audit 2026-10-10, item 50).
    'test_eigvalsh_batch_ceiling_20260807.py::test_safe_eigvalsh_survives_a_batch_above_the_ceiling',
    # The training loop on the device: fused AdamW, autocast, transfers, CUDA RNG restore, evaluation.
    'test_cuda_training_lane_20261010.py::test_cuda_resume_matches_the_uninterrupted_run',
    'test_cuda_training_lane_20261010.py::test_cuda_evaluate_matches_cpu_evaluate',
    # Fused Triton expected-likelihood CE against the float64 CPU torch route, reproducibility and timing
    # (2026-10-10).
    'test_expected_likelihood_kernel_20261010.py::test_cuda_triton_route_matches_cpu_torch_and_is_reproducible',
    # Fused Triton KL-decode CE (decode_ce_triton) against the float64 torch route, bitwise logits, reproducibility,
    # memory and timing (2026-10-12).
    'test_kl_decode_triton_20261012.py::test_cuda_fused_route_matches_float64_like_the_torch_route',
    # The amp_dtype reduced-precision probe under CUDA autocast's operation lists (2026-10-12).
    'test_amp_routes_20261012.py::test_cuda_autocast_probe_matches_the_rule',
    # The tiny live launcher route on CUDA (fused KL-decode CE, fixed-schedule frame exponential) against a
    # float64 CPU copy (audit 2026-10-17 item 26).
    'test_cuda_live_route_20261018.py::test_cuda_live_route_matches_a_float64_cpu_copy',
    # Fused Triton diagonal comparison moments (relational_diag_moments_kernel) against the float64 CPU torch route on
    # a live-shaped layer and on the tiny live profile moved onto the diagonal comparison route (audit 2026-10-17 item 20).
    'test_diag_moments_triton_20261018.py::test_cuda_fused_moments_match_float64_like_the_torch_route',
    'test_diag_moments_triton_20261018.py::test_cuda_live_route_with_fused_moments_matches_a_float64_cpu_copy',
})

CUDA_MIRROR_TESTS: frozenset[str] = frozenset({
    'test_divergence.py::test_safe_kl_clamp_bounds_and_nan',
    'test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior',
    'test_retraction.py::test_full_retraction_stays_spd',
})

EXTERNAL_TESTS: frozenset[str] = frozenset()

RESOURCE_GROUPS: dict[str, str] = {
    **{key: "cuda" for key in CUDA_TESTS},
}


def node_key(nodeid: str) -> str:
    """Return the file/function policy key for a collected pytest node ID."""
    tail = nodeid.replace("\\", "/").split("/")[-1]
    return tail.split("[")[0]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the slow figure/artifact integration tests (skipped by default)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="slow integration test; pass --runslow to run it")
    mirror_cuda = os.environ.get("VFE3_TEST_DEVICE", "cpu").partition(":")[0].strip().lower() == "cuda"

    for item in items:
        key = node_key(item.nodeid)
        if key in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)
        if key in CUDA_TESTS or (mirror_cuda and key in CUDA_MIRROR_TESTS):
            item.add_marker(pytest.mark.cuda)
        if key in EXTERNAL_TESTS:
            item.add_marker(pytest.mark.external)
        if key in RESOURCE_GROUPS:
            item.add_marker(pytest.mark.xdist_group(name=RESOURCE_GROUPS[key]))
        elif mirror_cuda and key in CUDA_MIRROR_TESTS:
            item.add_marker(pytest.mark.xdist_group(name="cuda"))
        if key in SLOW_TESTS and not run_slow:
            item.add_marker(skip_slow)
