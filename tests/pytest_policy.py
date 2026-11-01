"""Collection policy for the test lanes: the cuda, external and triton markers.

Registered as a pytest plugin by the repository-root ``conftest.py`` (``pytest_plugins``), so its
hooks run for every invocation. It must not import torch: it loads before ``tests/conftest.py`` sets
the CUDA workspace environment that has to precede the first torch import.

The former slow lane was removed (audit 2026-10-20 item 22): no test carried the marker, so it always ran nothing.
The ``xdist_group`` serialization marks were removed as well (audit 2026-10-31 test-infrastructure-6): pytest-xdist
honors them only under ``--dist loadgroup``, which no runner uses, and every marked test was a CUDA test the CPU lane
deselects while the GPU lane runs serially.

The ``triton`` marker is derived, not listed (audit 2026-10-31 test-infrastructure-1): every test in a
``tests/test_*.py`` file whose source runs the Triton interpreter or tests for triton (:data:`TRITON_SOURCE_PATTERN`)
carries it, so ``run_cpu_tests.py`` can give those files to an interpreter with triton. A lane that sets
``VFE3_REQUIRE_TRITON=1`` fails every test that skips for want of triton instead of reporting a skip.
"""
import os
import re
from functools import lru_cache
from pathlib import Path

import pytest


CUDA_TESTS: frozenset[str] = frozenset({
    # cuSOLVER batched-eigensolver ceiling guards, which skip without CUDA (audit 2026-10-10, item 50).
    'test_eigvalsh_batch_ceiling_20260807.py::test_safe_eigvalsh_survives_a_batch_above_the_ceiling',
    # The training loop on the device: fused AdamW, autocast, transfers, CUDA RNG restore, evaluation.
    'test_cuda_training_lane_20261010.py::test_cuda_resume_matches_the_uninterrupted_run',
    'test_cuda_training_lane_20261010.py::test_cuda_evaluate_matches_cpu_evaluate',
    # The step timer's phase marks on the device record events and read nothing (audit 2026-10-31 perf-live-path-5).
    'test_audit_20261031_train_loop.py::test_cuda_phase_marks_add_no_host_synchronization',
    # Fused Triton expected-likelihood CE against the float64 CPU torch route, reproducibility and timing
    # (2026-10-10).
    'test_expected_likelihood_kernel_20261010.py::test_cuda_triton_route_matches_cpu_torch_and_is_reproducible',
    # Fused Triton KL-decode CE (decode_ce_triton) against the float64 torch route, bitwise logits, reproducibility,
    # memory and timing (2026-10-12).
    'test_kl_decode_triton_20261012.py::test_cuda_fused_route_matches_float64_like_the_torch_route',
    'test_cue_result_triton_20261025.py::test_cuda_kernels_match_a_float64_reference_and_record_timing',
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
    # The frame condition bound at the live head shape under 'high' precision: no float64 recomputation, no repeated
    # eigendecomposition of rows inside the bound, a fixed number of host reads (2026-10-21).
    'test_frame_bound_cost_20261021.py::test_cuda_frame_bound_at_the_live_shape',
    # weight_ema_device='cpu' with CUDA parameters: pinned staging, non-blocking copies and the worker-thread lerp
    # against a hand EMA (audit 2026-10-31, train-step-5).
    'test_audit_20261031_optimizers.py::test_cuda_weight_ema_host_route_matches_a_hand_ema',
    # The local-learner test's filter, local M-step and backprop step on CUDA agree with the CPU (2026-10-21).
    'test_local_learner_20261021.py::test_cuda_filter_and_learners_match_cpu',
    # The fused KL-decode CE with the frozen temperature the training step runs (NEED_TAU=False) on an odd row count
    # against the float64 torch route (audit 2026-10-31 A221).
    'test_audit_20261031_model_assembly.py::'
    'test_a221_cuda_fused_route_with_a_frozen_temperature_matches_float64_like_the_torch_route',
    # The pair-KL expansion's grid-free backward against the former IEEE fp32 recompute under the TF32 policy
    # (audit 2026-10-31 attention-scoring-1).
    'test_audit_20261031_scoring.py::test_cuda_expansion_function_is_bitwise_the_former_recompute_under_tf32_policy',
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
    # The device-portable float32 scorer, diagonal-moments, frame-reuse and projection-numerics contracts, which bear on
    # the live CUDA route under fp32_matmul_precision='high' (audit 2026-10-31 test-infrastructure-4).
    'test_audit_20261010_scoring.py::test_gram_precision_forward_is_bitwise_the_unfused_ops',
    'test_audit_20261010_scoring.py::test_gram_precision_gradcheck',
    'test_audit_20261010_scoring.py::test_gram_precision_backward_matches_the_autograd_chain',
    'test_audit_20261010_scoring.py::test_gram_precision_failed_entries_receive_zero_gradient',
    'test_audit_20261010_scoring.py::test_healthy_scorer_forward_is_the_unfused_gemm_bitwise',
    'test_audit_20261010_scoring.py::test_healthy_scorer_gradients_match_the_float64_direct_kl',
    'test_audit_20261010_scoring.py::test_healthy_key_backward_runs_no_triangular_solve',
    'test_audit_20261010_scoring.py::test_tied_covariance_is_factored_once',
    'test_audit_20261010_scoring.py::test_tied_factor_route_builds_one_source_root',
    'test_audit_20261010_scoring.py::test_diagonal_moments_match_the_reference_formula',
    'test_audit_20261010_scoring.py::test_diagonal_moments_float32_stay_within_rounding_of_float64',
    'test_audit_20261010_scoring.py::test_factor_route_tile_runs_ieee_fp32',
    'test_audit_20261010_scoring.py::test_recovered_pairs_reach_the_pinned_factor_tile',
    'test_frame_reuse_20260906.py::test_coordinate_norm_avoids_matrix',
    'test_frame_reuse_20260906.py::test_norm_fallback_retains_reduction',
    'test_frame_reuse_20260906.py::test_unique_factors_values_gradients_hvp',
    'test_frame_reuse_20260906.py::test_unique_exp_work_is_shared',
    'test_frame_reuse_20260906.py::test_unique_errors_precede_deduplication',
    'test_frame_reuse_20260906.py::test_unique_table_directional_finite_difference',
    'test_frame_reuse_20260906.py::test_builtin_model_gate',
    'test_frame_reuse_20260906.py::test_reuse_extreme_token_counts',
    'test_frame_reuse_20260906.py::test_unique_factors_under_active_bf16_autocast',
    'test_round2_projection_numerics_20261009.py::test_gemm_estimate_routes_the_spread_slice_to_direct_tiles',
    'test_round2_projection_numerics_20261009.py::test_estimate_majorizes_the_gemm_error_on_the_audit_sweep',
    'test_round2_projection_numerics_20261009.py::test_fallback_adds_no_host_synchronization',
    'test_round2_projection_numerics_20261009.py::test_fallback_gradient_matches_float64',
    'test_round2_projection_numerics_20261009.py::test_projected_moments_tiles_recompute_at_ieee',
    'test_round2_projection_numerics_20261009.py::test_projected_root_factor_tiles_recompute_at_ieee',
    'test_round2_projection_numerics_20261009.py::test_head_block_projection_matches_dense',
    'test_round2_projection_numerics_20261009.py::test_head_block_projection_rejects_bad_declarations',
    'test_round2_projection_numerics_20261009.py::test_head_block_projection_never_materializes_the_broadcast_covariance',
})

EXTERNAL_TESTS: frozenset[str] = frozenset()

# A test file runs the Triton interpreter (or tests for triton) when its source names the interpreter's environment
# variable, carries the subprocess exit code that means "triton unavailable", or skips on a failed triton import.
TRITON_SOURCE_PATTERN: re.Pattern[str] = re.compile(
    r"\bTRITON_INTERPRET\b|\b_NO_TRITON\b|importorskip\(\s*['\"]triton['\"]\s*\)")
# The environment variable a lane sets to "1" when its interpreter has triton: a triton skip then fails.
REQUIRE_TRITON_ENV = "VFE3_REQUIRE_TRITON"
_TRITON_SKIP_REASON: re.Pattern[str] = re.compile(r"\btriton\b", re.IGNORECASE)
# A skip reason that names the GPU (not only the word cuda): requires a GPU, cusolver, cublas or cudnn.
_CUDA_SKIP_REASON: re.Pattern[str] = re.compile(r"cuda|cusolver|cublas|cudnn|\bgpus?\b", re.IGNORECASE)
TESTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def is_triton_test_file(path: str) -> bool:
    r"""Whether the test file at ``path`` runs the Triton interpreter or tests for triton (its source matches
    :data:`TRITON_SOURCE_PATTERN`)."""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    return TRITON_SOURCE_PATTERN.search(source) is not None


def triton_test_files(tests_dir: Path = TESTS_DIR) -> tuple[str, ...]:
    r"""The ``tests/test_*.py`` files that carry the ``triton`` marker, as ``tests/<name>`` paths, sorted."""
    return tuple(f"{tests_dir.name}/{path.name}" for path in sorted(tests_dir.glob("test_*.py"))
                 if is_triton_test_file(str(path)))


def triton_required() -> bool:
    r"""Whether this lane's interpreter has triton, so that a triton skip must fail (``VFE3_REQUIRE_TRITON=1``)."""
    return os.environ.get(REQUIRE_TRITON_ENV, "").strip() == "1"


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
        if is_triton_test_file(str(item.path)):
            item.add_marker(pytest.mark.triton)


def _skip_reason(report: pytest.TestReport) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> object:
    """Fail a skip that hides a test no runner would execute.

    An unregistered test that skips for want of CUDA outside the CUDA lane is selected by no runner (audit 2026-10-20
    item 22): the CPU lane skips it and ``check_gpu_tests.py`` runs only ``-m cuda``. Registering it in ``CUDA_TESTS``
    (or ``CUDA_PARAM_TESTS``) marks it ``cuda`` and moves it to the CUDA runner. A test that skips for want of triton
    in a lane that requires it (``VFE3_REQUIRE_TRITON=1``, set by ``run_cpu_tests.py`` for an interpreter with triton)
    fails too (audit 2026-10-31 test-infrastructure-1). An expected failure (``xfail``) is not a skip and is left alone,
    and the GPU wording is matched by name (cuda, gpu, cusolver, cublas, cudnn), not by the substring "cuda" alone
    (audit 2026-10-31 test-infrastructure-2).
    """
    outcome = yield
    report = outcome.get_result()
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = _skip_reason(report)
    if triton_required() and _TRITON_SKIP_REASON.search(reason):
        report.outcome = "failed"
        report.longrepr = (f"{item.nodeid} skips for want of triton ({reason}) in a lane that requires it "
                           f"({REQUIRE_TRITON_ENV}=1): the interpreter of this lane must import triton")
        return
    if cuda_lane_requested() or is_registered_cuda_test(item.nodeid) or not _CUDA_SKIP_REASON.search(reason):
        return
    report.outcome = "failed"
    report.longrepr = (f"{item.nodeid} skips for want of CUDA ({reason}) but is registered in neither CUDA_TESTS "
                       "nor CUDA_PARAM_TESTS of tests/pytest_policy.py, so no runner executes it")
