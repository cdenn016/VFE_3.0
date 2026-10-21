r"""The ``relational_operators`` contract strings follow the configuration (audit 2026-10-20 item 25).

``config.json``, ``pure_path_report.json`` and ``test_results.json`` carry ``gradient_clipping``,
``score_covariance_contract``, ``factorization_fallback`` and ``content_covariance_contract``; before this fix they
were constants that misstated the clip policy, the routing covariance, the factorization fallback and the
frame-diagonal projection on the reference route. Tiny configs only; nothing is built or run.
"""

from typing import Any, Dict

from tests.profiles import tiny_config
from vfe3.run_artifacts import _relational_artifact_fields


def _record(**overrides: Any) -> Dict[str, Any]:
    return _relational_artifact_fields(tiny_config(**overrides))["relational_operators"]


def test_default_kernel_route_strings() -> None:
    record = _record()
    assert record["gradient_clipping"] == "shares_mu_role_l2_bound"
    assert record["score_covariance_contract"] == "exact_full_reduced_covariance_from_local_gaussian"
    assert record["factorization_fallback"] == "failed_entries_source_root_shifted_cholesky_qr3_fp32_then_qr_fp64"
    assert record["content_covariance_contract"] == "diagonal_local_moment_projection_of_predictive_kernel_plus_floor"


def test_reference_like_route_strings_differ_and_are_truthful() -> None:
    # Group-scoped clipping, comparison-invariant RoPE at rope_fraction 1 (diagonal P(G), Gram factorization, so the
    # closed-form diagonal scorer), and the frame-diagonal family's reference-coordinate projection.
    record = _record(grad_clip_scope="group", rope_mode="comparison_invariant", rope_fraction=1.0,
                     family="gaussian_frame_diagonal")
    assert record["gradient_clipping"] == "own_l2_bound_per_optimizer_group"
    assert record["score_covariance_contract"] == "diagonal_rope_commutant_projection_of_reduced_covariance"
    assert record["factorization_fallback"] == "none_closed_form_diagonal_comparison_route"
    assert record["content_covariance_contract"] == (
        "reference_diagonal_moment_projection_of_predictive_kernel_plus_covariant_floor")
    default = _record()
    for key in ("gradient_clipping", "score_covariance_contract", "factorization_fallback",
                "content_covariance_contract"):
        assert record[key] != default[key], key


def test_other_branches() -> None:
    assert _record(grad_clip=None)["gradient_clipping"] == "off"
    certified = _record(rope_mode="comparison_invariant", projected_factorization="gram_certified")
    assert certified["factorization_fallback"] == (
        "certificate_flagged_or_failed_entries_commutant_root_shifted_cholesky_qr3_fp32_then_qr_fp64")
    assert _record(projected_factorization="cholesky_qr2_certified")["factorization_fallback"] == (
        "certificate_flagged_or_failed_entries_source_root_qr_fp64")
    noisy_key = _record(relational_routing="noisy_key")
    assert noisy_key["score_covariance_contract"] == (
        "exact_full_reduced_covariance_from_local_gaussian_plus_learned_key_noise")
    blind = _record(relational_routing="mean_distance")
    assert blind["score_covariance_contract"] == "none_mean_distance_reads_no_covariance"
    assert blind["factorization_fallback"] == "none_mean_distance_has_no_factorization"
