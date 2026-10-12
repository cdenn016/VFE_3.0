"""Explicit test-owned configurations, independent of click-to-run settings.

Each constructor argument is supplied from this checked-in mapping. Keep behavioral
requirements beside their tests as overrides; never regenerate this mapping from
experiment defaults at import time.
"""
from copy import deepcopy
from typing import Any

from vfe3.config import VFE3Config


# Explicit full-link baseline, with tiny dimensions and figure generation disabled.
_CONFIG_VALUES: dict[str, Any] = {
    'vocab_size': 7,
    'embed_dim': 4,
    'max_seq_len': 3,
    'eval_stride': None,
    'train_window_offset': 'fixed',
    'batch_size': 2,
    'max_steps': 15000,
    'warmup_steps': 100,
    'n_layers': 1,
    'n_e_steps': 1,
    'n_heads': 2,
    'seed': 6,
    'deterministic': True,
    'fill_uninitialized_memory': True,
    'log_interval': 100,
    'eval_interval': 1500,
    'checkpoint_interval': 15000,
    'resume_from': None,
    'trust_resume_checkpoint': False,
    'family': 'gaussian_diagonal',
    'prior_covariance_table': 'diagonal',
    'mu_init_std': 0.02,
    'sigma_init': 1.0,
    'phi_scale': 0.01,
    'pos_phi_scale': 0.02,
    'encode_mode': 'per_token',
    'decode_mode': 'linear',
    'decode_coordinates': 'local',
    'decode_covariance': 'belief',
    'decode_tau': 1.0,
    'decode_unigram_prior': True,
    'unigram_kappa': 1.0,
    'decode_bias': True,
    'untie_decode_bank': True,
    'decode_chunk_size': 8192,
    'decode_ce_checkpoint': 'auto',
    'expected_likelihood_kernel': 'torch',
    'decode_logits_amp': False,
    'decode_second_order': False,
    'decode_ranking_fp64_escalation': True,
    'gauge_group': 'block_glk',
    'gauge_transport': 'on',
    'group_n': None,
    'irrep_spec': None,
    'irrep_basis': 'canonical',
    'irrep_spec_seed': 0,
    'irrep_min_rank': 1,
    'irrep_max_rank': None,
    'gauge_tying': None,
    'use_cg_coupling': False,
    'cg_covariance_mode': 'passthrough',
    'cg_covariance_floor': 1e-06,
    'cg_energy_weight': 0.0,
    'pos_phi': 'learned',
    'bch_residual_max': None,
    'pos_phi_project_slk': False,
    'pos_rotation': 'none',
    'rope_base': 100.0,
    'rope_fraction': 1.0,
    'rope_space': 'frame',
    'rope_comparison_covariance': 'full',
    'rope_on_value': True,
    'attention_mode': 'projected_gradient',
    'attention_residual': False,
    'relational_output_init': 'identity',
    'content_link_chart': 'affine',
    'content_link_edges': 'off',
    'content_link_edge_rank': 4,
    'content_link_latent': False,
    'content_link_latent_steps': 1,
    'content_link_latent_sigma': 0.1,
    'content_link_latent_damping': 1.0,
    'content_link_loop_weight': 0.0,
    'content_link_loop_window': 1,
    'relational_receiver_noise': 1.0,
    'relational_receiver_law': 'ambient_alignment',
    'relational_receiver_measurement': 'adjoint',
    'relational_receiver_update': 'euler',
    'relational_receiver_damping': 1.0,
    'relational_routing': 'gaussian_kl',
    'tie_comparison_maps': False,
    'predictive_readout': 'kernel',
    'minimal_agent_inference': 'exact',
    'minimal_agent_cavi_steps': 1,
    'minimal_agent_channel_noise': 1.0,
    'predictive_covariance': 'passthrough',
    'predictive_covariance_floor': 0.0001,
    'predictive_covariance_checkpoint': False,
    'lambda_alpha_mode': 'constant',
    'b0': 1.0,
    'c0': 1.0,
    'lambda_beta': 1.0,
    'lambda_alpha': 1.0,
    'mass_phi': 0.0,
    'mstep_self_coupling_weight': 0.0,
    'kappa_beta': 1.0,
    'learn_attention_temperature': False,
    'beta_attention_prior': 'causal_alibi',
    'alibi_slope': 1.0,
    'attention_window': 128,
    't5_num_buckets': 32,
    't5_max_distance': 128,
    't5_learnable_bias': False,
    'norm_type_block': 'layernorm',
    'norm_type_final': 'none',
    'block_coordinates': 'local',
    'prior_handoff_rho': 1.0,
    'prior_handoff_sigma': 0.0,
    'use_block_mlp': True,
    'block_mlp_mode': 'coordinate',
    'block_mlp_covariance': 'passthrough',
    'block_mlp_activation': 'gelu',
    'block_mlp_post_norm': False,
    'block_mlp_autocast': True,
    'block_mlp_covariance_floor': 0.0001,
    'block_mlp_expansion': 8,
    'block_mlp_inference': 'mean_field',
    'block_mlp_cavi_steps': 1,
    'block_mlp_em_weight': 0.0,
    'block_mlp_template_init_variance': 10.0,
    'block_mlp_template_init': 'absolute',
    'block_mlp_template_init_pull': 0.09,
    'block_mlp_dropout': 0.0,
    'm_block_mlp_lr': None,
    'per_type_hidden_bias': False,
    'm_per_type_hidden_bias_lr': None,
    'layernorm_affine': True,
    'block_norm_covariance': 'passthrough',
    'm_layernorm_lr': None,
    'e_q_mu_lr': 0.5,
    'e_q_sigma_lr': 0.015,
    'skip_belief_sigma_update': True,
    'm_p_mu_lr': 0.025,
    'm_p_sigma_lr': 0.0025,
    'm_phi_lr': 0.015,
    'm_decode_mu_lr': None,
    'm_decode_sigma_lr': None,
    'm_pos_phi_lr': None,
    'm_relational_score_lr': None,
    'm_relational_content_lr': None,
    'm_phi_update_mode': 'adamw',
    'm_table_update_mode': 'adamw',
    'm_table_fisher_lr': 0.1,
    'm_table_fisher_trust_radius': 0.1,
    'm_phi_group_trust_radius': 0.1,
    'm_phi_group_retraction': 'chart',
    'm_phi_fisher_rotation_weight': 2.0,
    'weight_decay': 0.05,
    'sigma_weight_decay': None,
    'mu_weight_decay': None,
    'decode_mu_weight_decay': None,
    'phi_weight_decay': 0.065,
    'eps': 1e-06,
    'kl_max': 100.0,
    'routing_kl_max_reference': 'absolute',
    'pair_kl_gemm_tol': None,
    'sigma_max': 100.0,
    'e_mu_q_trust': 2,
    'e_sigma_q_trust': 5.0,
    'grad_clip': 1.0,
    'grad_clip_scope': 'role',
    'divergence_abort': False,
    'amp_dtype': None,
    'fp32_matmul_precision': 'highest',
    'matmul_precision_retry': False,
    'full_cov_kl_precision': 'fp32_escalate',
    'projected_cov_condition_recovery': False,
    'projected_factor_route': 'gram',
    'projected_gram_condition_tol': 0.0001,
    'projected_factor_condition_tol': 0.0003,
    'projected_cov_recovery_precision': 'fp64',
    'mu_trust_cholesky_rounds': 3,
    'decode_av_precision': 'fp64',
    'transport_chart_max_norm': 20.0,
    'frame_pair_fp64_escalation': False,
    'frame_pair_condition_tol': 0.0001,
    'phi_mstep_max_matrix_norm': None,
    'phi_frame_condition_max': None,
    'val_diagnostic_batches': 1,
    'diagnose_canonical_free_energy': False,
    'grad_accum_steps': 1,
    'min_lr_frac': 0.01,
    'generate_embedding_figures': False,
}


def config_values(**overrides: Any) -> dict[str, Any]:
    """Return a fresh complete configuration dictionary with local overrides."""
    values = deepcopy({**_CONFIG_VALUES, **overrides})
    return values


def test_config(**overrides: Any) -> VFE3Config:
    """Construct the real validated config using only test-owned arguments."""
    return VFE3Config(**config_values(**overrides))


# pytest must not collect the factory when test modules import it.
test_config.__test__ = False


def tiny_config(**overrides: Any) -> VFE3Config:
    """Minimal profile for relational and runtime contracts."""
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                  n_layers=1, n_e_steps=1, norm_type_block='none',
                  norm_type_final='none', use_block_mlp=False, layernorm_affine=False,
                  decode_bias=False, decode_unigram_prior=False, untie_decode_bank=False,
                  beta_attention_prior='causal', fp32_matmul_precision=None,
                  attention_mode='projected_predictive', predictive_covariance='moment_matched',
                  attention_residual=True,
                  decode_ranking_fp64_escalation=False, decode_av_precision='fp32',
                  e_mu_q_trust=None, block_mlp_expansion=2,
                  generate_embedding_figures=False,
                  skip_belief_sigma_update=False)
    values.update(overrides)
    return test_config(**values)


def make_relational_operators(*args: Any, **overrides: Any) -> Any:
    """RelationalOperators whose eight required config-mirroring keywords come from the test baseline.

    ``attention_residual``, ``predictive_covariance``, ``receiver_law``, ``receiver_update``,
    ``content_link_chart``, ``minimal_agent_inference``, ``kl_max_reference`` and
    ``rope_comparison_covariance`` have no constructor default; explicit keyword arguments override the
    baseline values.
    """
    from vfe3.model.relational_operators import RelationalOperators
    values = _CONFIG_VALUES
    baseline = dict(attention_residual=values['attention_residual'],
                    predictive_covariance=values['predictive_covariance'],
                    receiver_law=values['relational_receiver_law'],
                    receiver_update=values['relational_receiver_update'],
                    content_link_chart=values['content_link_chart'],
                    minimal_agent_inference=values['minimal_agent_inference'],
                    kl_max_reference=values['routing_kl_max_reference'],
                    rope_comparison_covariance=values['rope_comparison_covariance'])
    return RelationalOperators(*args, **{**baseline, **overrides})
