r"""Introduction-time values of the ``VFE3Config`` fields, frozen for :func:`vfe3.config.config_from_serialized`.

A config written by :func:`vfe3.config.serialize_config` carries every field, so a serialized record lacks a field
only when it predates that field, and it then ran the field's value at introduction, which is fixed forever. The
loader fills such a field from these tables (after the explicit ``_HISTORICAL_DEFAULT_PINS`` rows and era rules of
:func:`vfe3.config.historical_default_pins`, which take precedence), so changing a ``VFE3Config`` default never
changes how an older record reloads and needs no bookkeeping at the time of the change.

``SNAPSHOT_CAA0114`` is every field default at caa0114 (2026-10-17), copied from the snapshot committed with audit
2026-10-17 item 24 (``tests/data/vfe3config_defaults_caa0114.json``, 395a430) and checked against the ``VFE3Config``
source of caa0114. A field present then whose default had already changed after its introduction carries a pin row,
which overrides the snapshot. ``INTRODUCED_AFTER_CAA0114`` holds the fields added after the snapshot at their
introduction values, read from the source of the commit that added each. A field added later than both tables is
filled with the current default, which is its introduction value until that default first changes; do not edit
these tables when a default changes.
"""
from typing import Any, Mapping

SNAPSHOT_CAA0114: Mapping[str, Any] = {
    'alibi_slope':                           1.0,
    'amp_dtype':                             'bf16',
    'attention_mode':                        'projected_predictive',
    'attention_residual':                    True,
    'attention_window':                      128,
    'b0':                                    1.0,
    'batch_size':                            64,
    'beta_attention_prior':                  'causal',
    'block_coordinates':                     'local',
    'block_mlp_activation':                  'gelu',
    'block_mlp_cavi_steps':                  1,
    'block_mlp_covariance':                  'passthrough',
    'block_mlp_covariance_floor':            0.0001,
    'block_mlp_dropout':                     0.0,
    'block_mlp_em_weight':                   0.0,
    'block_mlp_expansion':                   8,
    'block_mlp_inference':                   'mean_field',
    'block_mlp_mode':                        'coordinate',
    'block_mlp_template_init':               'absolute',
    'block_mlp_template_init_pull':          0.09,
    'block_mlp_template_init_variance':      10.0,
    'block_norm_covariance':                 'passthrough',
    'block_norm_trace_cap':                  True,
    'c0':                                    1.0,
    'cg_covariance_floor':                   1e-06,
    'cg_covariance_mode':                    'passthrough',
    'cg_energy_weight':                      0.0,
    'checkpoint_interval':                   15000,
    'content_link_chart':                    'exponential',
    'content_link_edge_rank':                4,
    'content_link_edges':                    'off',
    'content_link_latent':                   False,
    'content_link_latent_damping':           1.0,
    'content_link_latent_sigma':             0.1,
    'content_link_latent_steps':             1,
    'content_link_loop_weight':              0.0,
    'content_link_loop_window':              1,
    'decode_av_precision':                   'fp32',
    'decode_ce_checkpoint':                  'auto',
    'decode_ce_triton':                      False,
    'decode_chunk_size':                     8192,
    'decode_coordinates':                    'local',
    'decode_covariance':                     'belief',
    'decode_logits_amp':                     False,
    'decode_mode':                           'linear',
    'decode_mu_weight_decay':                None,
    'decode_ranking_fp64_escalation':        False,
    'decode_second_order':                   False,
    'decode_tau':                            1.0,
    'decode_unigram_prior':                  True,
    'deterministic':                         True,
    'diagnose_agent_elbo':                   False,
    'diagnose_canonical_free_energy':        False,
    'e_mu_q_trust':                          2,
    'e_q_mu_lr':                             0.5,
    'e_q_sigma_lr':                          0.015,
    'e_sigma_q_trust':                       5.0,
    'embed_dim':                             64,
    'encode_mode':                           'per_token',
    'eps':                                   1e-06,
    'eval_interval':                         1500,
    'eval_stride':                           None,
    'family':                                'gaussian_diagonal',
    'fp32_matmul_precision':                 'highest',
    'frame_pair_escalation_tol':             None,
    'gauge_group':                           'block_glk',
    'gauge_transport':                       'on',
    'gauge_tying':                           None,
    'generate_embedding_figures':            False,
    'grad_accum_steps':                      1,
    'grad_clip':                             1.0,
    'grad_clip_scope':                       'role',
    'group_n':                               None,
    'irrep_max_rank':                        None,
    'irrep_min_rank':                        1,
    'irrep_spec':                            None,
    'irrep_spec_seed':                       0,
    'kappa_beta':                            1.0,
    'kl_max':                                None,
    'lambda_alpha_mode':                     'state_dependent_per_coord',
    'lambda_beta':                           1.0,
    'layernorm_affine':                      False,
    'learn_attention_temperature':           False,
    'log_interval':                          100,
    'm_block_mlp_lr':                        None,
    'm_decode_mu_lr':                        0.025,
    'm_decode_sigma_lr':                     0.001,
    'm_layernorm_lr':                        None,
    'm_p_mu_lr':                             0.025,
    'm_p_sigma_lr':                          0.0025,
    'm_per_type_hidden_bias_lr':             None,
    'm_phi_fisher_completion':               'rotation',
    'm_phi_fisher_rotation_weight':          2.0,
    'm_phi_group_differential_mode':         'series',
    'm_phi_group_log_fallback':              'series',
    'm_phi_group_retraction':                'chart',
    'm_phi_group_trust_radius':              0.1,
    'm_phi_lr':                              0.015,
    'm_phi_update_mode':                     'adamw',
    'm_pos_phi_lr':                          0.0035,
    'm_relational_content_lr':               0.005,
    'm_relational_score_lr':                 0.005,
    'm_table_fisher_lr':                     0.1,
    'm_table_fisher_trust_radius':           0.1,
    'm_table_update_mode':                   'adamw',
    'max_seq_len':                           128,
    'max_steps':                             15000,
    'min_lr_frac':                           0.01,
    'minimal_agent_cavi_steps':              1,
    'minimal_agent_channel_noise':           1.0,
    'minimal_agent_inference':               'mean_field',
    'mstep_self_coupling_weight':            0.0,
    'mu_init_std':                           0.065,
    'mu_weight_decay':                       0,
    'n_e_steps':                             1,
    'n_heads':                               2,
    'n_layers':                              1,
    'norm_type_block':                       'layernorm',
    'pair_kl_gemm_tol':                      None,
    'per_type_hidden_bias':                  False,
    'phi_frame_condition_max':               None,
    'phi_mstep_max_matrix_norm':             None,
    'phi_scale':                             0.035,
    'phi_weight_decay':                      0.065,
    'pos_phi':                               'learned',
    'pos_phi_project_slk':                   False,
    'pos_phi_scale':                         0.02,
    'predictive_covariance':                 'moment_matched',
    'predictive_covariance_ceiling':         None,
    'predictive_covariance_checkpoint':      False,
    'predictive_covariance_floor':           0.0001,
    'predictive_covariance_floor_mode':      'prior',
    'predictive_covariance_fp64_escalation': False,
    'predictive_covariance_precision':       'fp32',
    'predictive_readout':                    'kernel',
    'prior_covariance_table':                'diagonal',
    'prior_handoff_rho':                     1.0,
    'prior_handoff_sigma':                   0.0,
    'projected_factorization':               'gram',
    'relational_key_noise':                  1.0,
    'relational_receiver_damping':           1.0,
    'relational_receiver_measurement':       'inverse',
    'relational_receiver_noise':             1.0,
    'relational_receiver_update':            'closed_form',
    'relational_routing':                    'gaussian_kl',
    'relational_tile_mib':                   32,
    'resume_from':                           None,
    'rope_base':                             100.0,
    'rope_fraction':                         1.0,
    'rope_mode':                             'none',
    'routing_kl_max_reference':              'absolute',
    'seed':                                  6,
    'sigma_init':                            15,
    'sigma_max':                             100.0,
    'sigma_weight_decay':                    0,
    'skip_belief_sigma_update':              False,
    'tie_comparison_maps':                   False,
    'transport_chart_max_norm':              50.0,
    'trust_resume_checkpoint':               False,
    'unigram_kappa':                         1.0,
    'untie_decode_bank':                     True,
    'use_block_mlp':                         True,
    'use_cg_coupling':                       False,
    'val_diagnostic_batches':                1,
    'vocab_size':                            50257,
    'warmup_steps':                          100,
    'weight_decay':                          0.05,
}

INTRODUCED_AFTER_CAA0114: Mapping[str, Any] = {
    'max_failed_step_skips':                 None,            # e4bc4a5 (2026-10-18)
    'max_consecutive_skips':                 None,            # e4bc4a5 (2026-10-18)
    'loss_spike_skip_nats':                  None,            # e4bc4a5 (2026-10-18)
    'checkpoint_keep_last':                  None,            # e4bc4a5 (2026-10-18)
    'checkpoint_at_end':                     False,           # e4bc4a5 (2026-10-18)
    'block_variance_at_log':                 False,           # e4bc4a5 (2026-10-18)
    'predictive_covariance_spread_squares':  False,           # 9693a9a (2026-10-18)
    'predictive_covariance_symmetric_pairs': False,           # 9693a9a (2026-10-18)
    'self_kl_max':                           None,            # 9693a9a (2026-10-18)
    'block_norm_cap_moment_fallback':        False,           # 9693a9a (2026-10-18)
    'relational_diag_moments_kernel':        'torch',         # b396e2a (2026-10-18)
    'weight_ema_decay':                      None,            # bfb86bf (2026-10-21)
    'weight_ema_start_frac':                 0.5,             # bfb86bf (2026-10-21)
    'weight_ema_update_interval':            1,               # bfb86bf (2026-10-21)
    'weight_ema_device':                     'same',          # bfb86bf (2026-10-21)
    'm_phi_sparse_rows':                     False,           # 1f94d7d (2026-10-21)
    'm_decode_update_mode':                  'adamw',         # 1f94d7d (2026-10-21)
    'relational_score_weight_decay':         0.0,             # 11db517 (2026-10-21)
    'block_memory_regression_rank':          None,            # cue_result_memory (2026-10-24)
    'block_memory_result_noise_ratio':       0.01,            # cue_result_memory (2026-10-24)
    'block_mlp_tie_layers':                  False,           # tied block MLP across depth (2026-10-25)
    'block_memory_regime_transition':        'off',           # regime belief through depth (2026-10-25)
    'block_memory_regime_beta':              1.0,             # regime belief through depth (2026-10-25)
    'block_memory_regime_stay':              0.9,             # regime belief through depth (2026-10-25)
    'block_memory_tile_mib':                 32,              # cue_result_memory tile budget (2026-10-25)
    'block_memory_triton':                   False,           # fused cue/result Triton route (2026-10-25)
}

FIELD_INTRODUCTION_VALUES: Mapping[str, Any] = {**SNAPSHOT_CAA0114, **INTRODUCED_AFTER_CAA0114}
