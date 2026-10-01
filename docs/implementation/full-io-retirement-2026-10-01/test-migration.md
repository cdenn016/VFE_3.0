# Test migration for fixed full input/output attention

Date: 2026-10-01. Scope: the approved retirement of transported and projected-conditional model attention, with projected-predictive and projected-gradient attention retained at full input/output width. This records implementation choices; it does not make a new mathematical or performance claim.

The explicit shared profile now selects `projected_gradient` with the existing supported inference settings, constant alpha, and full-width operators. Predictive fixtures continue to select `projected_predictive` where their assertions depend on a single predictive content readout. Tests no longer inject removed `relational_input_scope`, `relational_output_scope`, or `relational_identity_content` constructor fields. Both retained routes keep independent Q/K projections and learned T/O content maps. Config-default independence remains explicit in `tests/profiles.py`.

Tests of a retired model architecture were removed or replaced with rejection checks. Their old objectives and numerical checksums were not reinterpreted as results for another model. This includes conditional-only relational forces and qkv imports, bare transported E-step execution, MM model iteration, inner frame updates, and detached or straight-through projected inference. Direct transport, divergence, Gaussian geometry, Fisher-gradient, retraction, and MM mathematical primitives remain where they do not require an unsupported model configuration. `tests/test_relational_filtering.py` remains as a shared deterministic Gaussian/frame fixture for retained full-map numerical checks.

The retained relational score tests still compare with direct projected Gaussian divergences. The full-link gradient tests still compare with a literal full-link Gaussian KL objective and finite differences. Their numerical tolerances were not relaxed. Predictive covariance, centering, source factorization, and no-self identity-start behavior remain applicable and retained. The removed `apply_content` API check was replaced by its supported predictive-content calculation with identity self-routing beta, keeping the value assertion.

The fixed-width migration tests check both retained modes, full-width projection shapes and counts, removal of constructor scope fields, explicit serialized full/full compatibility, and serialized head-scope rejection. The compatibility fixture strictly loads matched state before comparing model state; it does not compare independently randomized model instances. Root-owned `tests/test_attention_retirement.py` separately exercises the migration contract.

Optimizer, checkpoint, counters, train-loop wiring, model gradient, and supported readout diagnostics are preserved using explicit supported configurations where their assertions apply. The receiver-cost capture test reads receiver costs from the snapshot/diagnostic capture that actually owns them; it does not revive retired conditional iteration traces. Conditional-only identity-content and transported ablation arms are excluded from the current architecture. Persisted synthetic scaling fixtures use supported two-head block-GL(K) metadata without building a model.

## CPU verification selection

All tests are run by the coordinating agent with `C:/anaconda/python.exe`, CPU, one thread, and small model dimensions. This worker did not launch a separate test pass. The coordinating agent owns the focused JUnit results and the one broader default CPU pass.

The source precheck looked for `VFE3Config` calls bypassing the explicit profiles, production-size model literals, and long training loops. The launcher's direct constructor test monkeypatches its entry config to V=6, K=4, H=2, N=4, B=2. Three tests create actual GPT-2 vocabulary models (V=50,257) and should be excluded from this bounded CPU pass: `test_train.py::test_training_smoke_on_real_wikitext2_if_present`, `test_train.py::test_generate_samples_false_is_silent_at_real_vocab`, and `test_train.py::test_auto_default_sample_decoder_emits_at_gpt2_vocab`. Their existing 1-30 step workloads fall outside the tiny-model CPU policy. Large vocabulary/sequence values in report memory guards are metadata-only and use a scalar fake module. Reporting scaling K=8/16/32 values are persisted metadata-only. The belief initialization spread test uses V=512 for initialization statistics without a forward or training loop; its model-table tests can be excluded under a strict single-digit-vocabulary selection.

At the source precheck, `tests/pytest_policy.py` had empty slow and external test sets, so the default slow option alone did not exclude the actual large-vocabulary training tests. Explicit exclusions are required. This is a selection observation, not a timing benchmark.

## Broader-pass fixture repairs

The broader CPU pass exposed additional legacy fixture choices after the shared profile moved to projected-gradient attention. Generic model, optimizer, mixed-precision, evaluation-stride, block-transform, and generation contracts now use supported Gaussian KL, two-head block-GL(K), fixed-frame settings. These changes select an applicable operating point; they do not claim equivalence to the retired model objective. Block and stack unit tests receive explicit full-width operators and a realized frame, and chunked-decoder twin comparisons consume the same-forward model belief state.

Non-KL decoder tests in the family-chunked decode, workspace, and canonical-dispatch modules keep their original order 0.5 and registered-functional numerical contracts. They now construct standalone `PriorBank` primitives without an unsupported `VFE3Config` or a projected model. No numeric tolerances were relaxed. The canonical dispatch module respects `VFE3_TEST_DEVICE`, whose default is CPU, instead of automatically selecting CUDA.

Canonical-content gauge/projected encoders remain available as standalone PriorBank and coordinate-transform primitives, but their model admission/materialization paths are outside the retained attention envelope. Model-only canonical materialization, single-head, old MM, and estimator-comparison tests were retired. Direct encoder/family/decoder registry rejection, pushforward/pullback gradchecks, frame validation, dense/fused decoder numerics, and bank initialization remain exercised. New model configuration rejection checks state that both retained attention modes reject canonical encoder selectors.

The same distinction applies to SO(N), Clebsch-Gordan, isotypic mixer, exotic gauge-group, and Laplace tests: standalone mathematical builders, equivariance checks, family divergences, finite differences, and direct bank decoding remain; unsupported end-to-end model cases were retired. Model config tests now reject the unsupported group/family/functionals explicitly instead of accepting them through old attention modes.

Receiver diagnostics are checked through their current capture contract: finite frozen-receiver costs and covariance summaries, with legacy global-F availability set to zero. Active head-mixer/head-evidence and outer-phi report assertions remain, and their missing production diagnostics were repaired by the coordinating agent rather than deleting those assertions. This worker performed syntax validation of its 28 affected modules and did not launch a second broader suite.

The affected-file repair pass exposed four residual fixtures. The trained strided-evaluation witness now uses two-head block-GL(K), preserving its original inequality. The dense generic family-workspace witness constructs a standalone full Gaussian PriorBank at its original Renyi order 0.5. The coordinate-KL rejection test distinguishes the direct family API's KL(alpha=1) error from the projected config's renyi_order error. The direct Laplace decoder test checks the actual decoder registration's family-consistency metadata and finite value/gradient behavior; it does not assert a PriorBank-constructor validation that does not exist. This worker did not run these residual tests separately.

## Removed or replaced test definitions

The inventory below compares current definitions with the fetched implementation baseline. Missing definitions may be replaced by rejection or supported-contract tests in the same file.

### `tests/test_canonical_content_gauge.py`

- `test_canonical_config_accepts_unigram_base_rate_and_names_its_separate_role`
- `test_canonical_model_has_no_supervised_phi_gradient_and_emits_one_provenance_notice`

### `tests/test_canonical_content_projected.py`

- `test_projected_config_accepts_full_head_evidence_after_pullback`
- `test_projected_config_accepts_supported_decode_position_and_tie_variants`
- `test_projected_config_rejects_each_incompatible_cross_field_pair`
- `test_projected_fixed_depth_reuses_one_transport_across_all_layers`
- `test_projected_manual_decode_matches_canonical_full_oracle`
- `test_projected_materialize_expands_only_compact_vertex_factors`
- `test_projected_materialize_uses_one_shared_factored_frame_and_preserves_q0_equals_p`
- `test_projected_model_dense_fused_training_threads_context_and_gradients`
- `test_projected_model_materialization_preserves_float32_under_bf16_autocast`
- `test_projected_phi_gradient_reaches_canonical_tables_and_realized_frames`
- `test_projected_single_head_groups_request_factored_vertex_transport`

### `tests/test_cg.py`

- `test_cg_model_step0_byte_identical_and_trains`

### `tests/test_cleanup_config.py`

- `test_historical_attention_pair_migrates`

### `tests/test_congruence_cond_guard_toggle_20260727.py`

- `test_default_model_forward_never_runs_the_conditioning_svd`
- `test_enabled_model_forward_runs_the_conditioning_svd`

### `tests/test_diagnostics.py`

- `test_diagnostics_registry_dispatch_substitutes_attention_entropy`

### `tests/test_divergence.py`

- `test_config_accepts_squared_hellinger_and_rejects_unknown`
- `test_model_forward_under_new_divergence`
- `test_model_forward_under_squared_hellinger`

### `tests/test_e_step.py`

- `test_decoupled_learning_rates_freeze_components`
- `test_e_step_iteration_spd_affine_default_is_byte_identical`
- `test_e_step_iteration_transport_flat_default_is_byte_identical`
- `test_e_step_runs_n_iter_and_returns_trajectory`
- `test_filtering_step_descends_F_filt`
- `test_fixed_seed_regression`
- `test_iteration_keeps_sigma_positive_and_shapes`
- `test_keys_freezing_bites_on_stepped_belief`
- `test_oracle_unroll_grad_toggle_changes_prior_gradient`
- `test_phi_step_descends_global_F_with_beliefs_frozen`
- `test_smoothing_loop_decreases_F_overall`
- `test_smoothing_step_descends_global_F`

### `tests/test_estep_fixed_point_reporting_20260715.py`

- `test_depth_sensitivity_marks_trained_depth_and_restores_state`
- `test_one_step_ahead_residual_is_distinct_from_configured_last_step`

### `tests/test_extract.py`

- `test_e_step_belief_trace_captures_trajectory`

### `tests/test_extract_forward_fidelity.py`

- `test_projected_rope_detached_replay_matches_production_snapshot`

### `tests/test_forward_kl_uniqueness.py`

- `test_forward_kl_converges_to_geometric_mean_fixed_point`
- `test_geometric_mean_gap_is_kl_specific`
- `test_non_kl_divergence_breaks_the_geometric_mean_form`

### `tests/test_frame_gaussian_family.py`

- `test_frame_family_leaves_the_gauge_table_without_gradient`

### `tests/test_frame_pair_escalation_20260923.py`

- `test_escalated_diagnostic_bank_keeps_float64_frames`

### `tests/test_full_covariance.py`

- `test_full_covariance_e_step_keeps_sigma_spd_and_symmetric`
- `test_full_covariance_reduces_to_diagonal_at_identity_transport`

### `tests/test_fullcov_alpha_roadmap_2026_06_13.py`

- `test_fullcov_unroll_oracle_grad_is_finite`
- `test_fullcov_unroll_oracle_grad_reaches_gauge_frame_only_when_enabled`
- `test_per_coord_bhattacharyya_and_jeffreys_construct_squared_hellinger_still_rejected`
- `test_per_coord_bhattacharyya_forward_backward_runs`
- `test_state_dependent_per_position_nonrenyi_forward_backward`
- `test_warn_state_dependent_bounded_divergence_b0`

### `tests/test_gauge_groups.py`

- `test_full_model_logits_invariant_under_global_gauge`

### `tests/test_gauge_optim.py`

- `test_default_adamw_one_step_is_byte_identical_to_golden`
- `test_multi_head_rope_forward_reuses_pair_stats_and_matches_recompute_oracle`
- `test_single_head_forward_matches_recompute_dense_low_level_oracle`

### `tests/test_hardcoded_exact_fastpaths_20260718.py`

- `test_diagnostic_free_energy_forwards_per_head_mean_transport`

### `tests/test_head_mixer_isotypic.py`

- `test_so_n_mixed_tower_model_constructs_with_mixer_and_trains`

### `tests/test_laplace_family.py`

- `test_laplace_model_trains_end_to_end`

### `tests/test_layernorm_affine.py`

- `test_block_affine_frozen_under_detach`
- `test_block_affine_trains_under_straight_through`
- `test_final_affine_trains_under_any_estimator`

### `tests/test_link_spectrum_20260928.py`

- `test_conditional_update_route_emits_nan_where_undefined`

### `tests/test_mimo_remediation.py`

- `test_scaling_rejects_inert_mup_before_creating_model`

### `tests/test_mm_exact_full_covariance_20260807.py`

- `Test1_DiagonalPathByteIdentical::test_e_step_iteration_mm_exact_diagonal_byte_identical`
- `Test2_ConfigConstruction::test_gaussian_full_mm_exact_multi_head_accepted_at_config_time`
- `Test2_ConfigConstruction::test_gaussian_full_mm_exact_single_block_constructs`
- `Test3_ModelForwardBackward::test_forward_backward_all_grads_finite`
- `Test5_FixedPointAgreement::test_converged_gradient_belief_is_a_near_fixed_point_of_mm_exact`
- `Test7_MMDampingSemantics::test_diagonal_path_convention_matches`

### `tests/test_model.py`

- `test_attention_maps_single_block_group_has_unit_head_axis`
- `test_b0_c0_threaded_into_state_dependent_alpha`
- `test_block_runs_e_step_and_preserves_shapes`
- `test_detached_e_step_with_phi_runs_and_reaches_decode_priors`
- `test_diagnostics_includes_gauge_geometry_probes`
- `test_mass_phi_penalizes_phi_inside_the_e_step`
- `test_model_runs_under_cross_coupled_block_glk`
- `test_model_runs_under_sp_gauge_group`
- `test_model_runs_under_tied_block_glk`
- `test_phi_e_step_actually_moves_phi`
- `test_phi_retraction_bch_is_reached_by_the_stack`
- `test_stack_handoff_updates_prior_across_blocks`

### `tests/test_mu_trust_region.py`

- `test_e_step_trust_binds_changes_mu`
- `test_e_step_trust_none_is_current_behavior`

### `tests/test_p1_compact_phi_block_transport_20260711.py`

- `test_model_threads_automatic_route_into_live_phi_bch_retraction`

### `tests/test_p3_pairwise_stats_reuse_20260711.py`

- `test_p3_production_forwards_true_to_the_q_consumer`

### `tests/test_perf_equivalence.py`

- `test_model_forward_matches_frozen_oracle`

### `tests/test_posthoc_gaps_20260928.py`

- `test_run_posthoc_transported_route_records_reasons`

### `tests/test_precision_factor_route_20260924.py`

- `test_factor_route_config_validated_and_inactive_on_transported`

### `tests/test_predictive_full_scope.py`

- `test_full_predictive_config_roundtrip_and_conditional_scopes`
- `test_predictive_config_rejects_legacy_head_architecture`
- `test_predictive_model_wiring_does_not_read_conditional_scope_overrides`
- `test_predictive_operator_defaults_match_existing_full_initialization`
- `test_predictive_operator_rejects_explicit_head_architecture`

### `tests/test_projected_condition_recovery_20260923.py`

- `test_config_disabled_validated_and_inactive_on_transported`

### `tests/test_projected_gradient.py`

- `test_receiver_noise_keeps_existing_positional_operator_arguments`

### `tests/test_projected_prior_bank_decode.py`

- `test_projected_filtering_skip_false_does_not_typeerror`
- `test_transported_linear_still_constructs`

### `tests/test_relational_families.py`

- `test_full_gaussian_active_per_coordinate_alpha`
- `test_gaussian_conditional_outer_finite_difference`

### `tests/test_relational_filtering.py`

- `test_inner_force_matches_frozen_cost`
- `test_outer_parameter_finite_difference`
- `test_source_beliefs_detached_and_no_grad_evaluation`

### `tests/test_relational_full_input.py`

- `test_full_input_conditional_outer_gradient_finite_difference`
- `test_full_input_contains_head_local_scores_and_values`
- `test_invalid_input_scope_rejected`
- `test_unspecified_scopes_score_like_the_explicit_mode_defaults`

### `tests/test_relational_integration.py`

- `test_disabled_mode_has_no_parameters_or_rng_effect`
- `test_snapshot_reports_actual_conditional_semantics`

### `tests/test_relational_predictive.py`

- `test_invalid_predictive_combinations_are_rejected`

### `tests/test_relational_scoring.py`

- `test_operator_initialization_identity_content_and_compact_frames`

### `tests/test_remediation_20260923_A.py`

- `test_additive_encoder_admitted_on_projected_routes`
- `test_additive_encoder_contamination_warns_on_transported`
- `test_conditional_route_keeps_family_live`
- `test_conditional_route_marks_only_its_inert_knobs`
- `test_frozen_sigma_channel_is_marked_and_bit_inert`
- `test_lambda_alpha_sweeps_apply_local_two_e_step_requirement`
- `test_renyi_order_inert_under_order_free_functionals`
- `test_self_coupling_inert_at_one_iteration_one_layer`
- `test_transported_frame_diagonal_drops_positional_channels`
- `test_windowed_prior_keeps_attention_window_live`

### `tests/test_remediation_20260923_C.py`

- `test_estep_bch_residual_gate_reaches_compose_bch`
- `test_estep_phi_step_no_longer_collapses_large_frames`
- `test_phi_objective_detach_follows_gradient_mode`
- `test_transported_phi_estep_does_not_leak_future_tokens`

### `tests/test_remediation_20260923_d.py`

- `test_conditional_values_use_the_value_gauge`
- `test_depth_scaled_output_init_is_rejected_without_learned_output_maps`
- `test_transported_diagnostics_score_the_e_step_state_against_its_live_prior`

### `tests/test_report.py`

- `test_generate_figures_forwards_model_owned_full_gaussian_policy`

### `tests/test_reporting_additions.py`

- `test_cost_model_counts_the_belief_estep_once_per_layer_forward`

### `tests/test_retraction.py`

- `test_full_cov_e_step_isotropic_init_finite_backward`

### `tests/test_semantics_core.py`

- `test_log_var_full_and_laplace_families`

### `tests/test_son_irreps.py`

- `test_config_so_n_accepts_and_coerces_json_pairs`
- `test_config_so_n_alibi_requires_head_count_match`
- `test_config_so_n_dim_sum_mismatch_rejected`
- `test_config_so_n_per_head_kappa_keyed_to_block_count`
- `test_config_so_n_rejects_per_block_phi_preconditioners`
- `test_config_so_n_requires_both_fields`
- `test_model_runs_under_so_n_irrep_tower`

### `tests/test_straight_through.py`

- `test_config_accepts_the_three_modes`
- `test_config_mode_passthrough_is_canonical`
- `test_config_rejects_unknown_mode`
- `test_straight_through_forward_byte_identical_to_unroll`
- `test_straight_through_grad_differs_from_unroll`
- `test_straight_through_grad_flows_to_encode_tables`

### `tests/test_train.py`

- `test_random_stream_does_not_clear_cutover_anchor`
- `test_training_decreases_loss_on_structured_stream`

### `tests/test_training_counters_20260928.py`

- `test_transported_route_leaves_route_counters_nan`

### `tests/test_viz_fixes_20260928.py`

- `test_route_gates_transported_live_route_keeps_estep_and_sigma_figures`
- `test_route_gates_zero_estep_and_frozen_transported`
