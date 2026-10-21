"""Collection policy for the test lanes: the cuda, external and xdist_group markers.

Registered as a pytest plugin by the repository-root ``conftest.py`` (``pytest_plugins``), so its
hooks run for every invocation. It must not import torch: it loads before ``tests/conftest.py`` sets
the CUDA workspace environment that has to precede the first torch import.

The former slow lane was removed (audit 2026-10-20 item 22): no test carried the marker, so it always ran nothing.
"""
import os

import pytest


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
    # The same at the reference layer shape (K = 160, H = 5, 32 x 128 tokens) with a forward plus backward timing of
    # both kernels (audit 2026-10-20 item 15).
    'test_diag_moments_triton_20261018.py::test_cuda_fused_moments_parity_and_timing_at_the_reference_shape',
    # Selected by no runner before audit 2026-10-20 item 22: the frame exponential's host-sync count and the fused
    # decode bias gradient under 'high' precision.
    'test_sync_free_matrix_exp_20261012.py::test_cuda_frame_exponential_has_no_host_sync',
    'test_audit_20261016_a2b_decode_bias_reduction.py::test_cuda_fused_bias_gradient_matches_the_torch_route_under_high_precision',
})

# CUDA parametrizations of tests whose other parameters run in the CPU lane, keyed with the parameter id
# (audit 2026-10-20 item 22).
CUDA_PARAM_TESTS: frozenset[str] = frozenset({
    'test_predictive_multilayer.py::test_local_projected_config_training_step[cuda]',
})

CUDA_MIRROR_TESTS: frozenset[str] = frozenset({
    'test_divergence.py::test_safe_kl_clamp_bounds_and_nan',
    'test_free_energy.py::test_free_energy_entropy_exact_for_deep_finite_prior',
    'test_retraction.py::test_full_retraction_stays_spd',
})

EXTERNAL_TESTS: frozenset[str] = frozenset()

RESOURCE_GROUPS: dict[str, str] = {
    **{key: "cuda" for key in CUDA_TESTS},
    **{tail: "cuda" for tail in CUDA_PARAM_TESTS},
}


def node_tail(nodeid: str) -> str:
    """Return the file/function tail of a collected pytest node ID, parameter id included."""
    return nodeid.replace("\\", "/").split("/")[-1]


def node_key(nodeid: str) -> str:
    """Return the file/function policy key for a collected pytest node ID."""
    return node_tail(nodeid).split("[")[0]


def required_cuda_node_keys() -> frozenset[str]:
    """Node keys the CUDA runner must collect: every CUDA-only, CUDA-parametrized and mirrored test."""
    return CUDA_TESTS | CUDA_MIRROR_TESTS | frozenset(node_key(tail) for tail in CUDA_PARAM_TESTS)


def cuda_lane_requested() -> bool:
    """Whether this process runs the CUDA lane (``VFE3_TEST_DEVICE=cuda``)."""
    return os.environ.get("VFE3_TEST_DEVICE", "cpu").partition(":")[0].strip().lower() == "cuda"


def is_registered_cuda_test(nodeid: str) -> bool:
    """Whether a node is a registered CUDA-only test, a registered CUDA parametrization or a CUDA mirror."""
    key = node_key(nodeid)
    return key in CUDA_TESTS or key in CUDA_MIRROR_TESTS or node_tail(nodeid) in CUDA_PARAM_TESTS


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    mirror_cuda = cuda_lane_requested()

    for item in items:
        key = node_key(item.nodeid)
        tail = node_tail(item.nodeid)
        if key in CUDA_TESTS or tail in CUDA_PARAM_TESTS or (mirror_cuda and key in CUDA_MIRROR_TESTS):
            item.add_marker(pytest.mark.cuda)
        if key in EXTERNAL_TESTS:
            item.add_marker(pytest.mark.external)
        if key in RESOURCE_GROUPS:
            item.add_marker(pytest.mark.xdist_group(name=RESOURCE_GROUPS[key]))
        elif tail in RESOURCE_GROUPS:
            item.add_marker(pytest.mark.xdist_group(name=RESOURCE_GROUPS[tail]))
        elif mirror_cuda and key in CUDA_MIRROR_TESTS:
            item.add_marker(pytest.mark.xdist_group(name="cuda"))


def _skip_reason(report: pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> object:
    """Fail an unregistered test that skips for want of CUDA outside the CUDA lane (audit 2026-10-20 item 22).

    Such a test is selected by no runner: the CPU lane skips it and ``check_gpu_tests.py`` runs only ``-m cuda``.
    Registering it in ``CUDA_TESTS`` (or ``CUDA_PARAM_TESTS``) marks it ``cuda`` and moves it to the CUDA runner.
    """
    outcome = yield
    report = outcome.get_result()
    if (not report.skipped or cuda_lane_requested() or is_registered_cuda_test(item.nodeid)
            or "cuda" not in _skip_reason(report).lower()):
        return
    reason = _skip_reason(report)
    report.outcome = "failed"
    report.longrepr = (f"{item.nodeid} skips for want of CUDA ({reason}) but is registered in neither CUDA_TESTS "
                       "nor CUDA_PARAM_TESTS of tests/pytest_policy.py, so no runner executes it")
