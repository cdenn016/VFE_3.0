"""Explicit test-owned configurations, independent of click-to-run settings.

Each constructor argument is supplied from this checked-in mapping. Keep behavioral
requirements beside their tests as overrides; never regenerate this mapping from
experiment defaults at import time.
The opt-in live profile is the exception: it reads the launcher dictionary of
``train_vfe3.py`` on purpose, so that it follows the launcher's route.
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
    'batch_size': 2,
    'max_steps': 15000,
    'warmup_steps': 100,
    'n_layers': 1,
    'n_e_steps': 1,
    'n_heads': 2,
    'seed': 6,
    'deterministic': True,
    'log_interval': 100,
    'eval_interval': 1500,
    'checkpoint_interval': 15000,
    'checkpoint_keep_last': None,
    'checkpoint_at_end': False,
    'block_variance_at_log': False,
    'max_failed_step_skips': None,
    'max_consecutive_skips': None,
    'loss_spike_skip_nats': None,
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
    'untie_decode_bank': True,
    'decode_chunk_size': 8192,
    'decode_ce_checkpoint': 'auto',
    'decode_ce_triton': False,
    'decode_logits_amp': False,
    'decode_second_order': False,
    'decode_ranking_fp64_escalation': True,
    'gauge_group': 'block_glk',
    'gauge_transport': 'on',
    'group_n': None,
    'irrep_spec': None,
    'irrep_spec_seed': 0,
    'irrep_min_rank': 1,
    'irrep_max_rank': None,
    'gauge_tying': None,
    'use_cg_coupling': False,
    'cg_covariance_mode': 'passthrough',
    'cg_covariance_floor': 1e-06,
    'cg_energy_weight': 0.0,
    'pos_phi': 'learned',
    'pos_phi_project_slk': False,
    'rope_mode': 'none',
    'rope_base': 100.0,
    'rope_fraction': 1.0,
    'attention_mode': 'projected_gradient',
    'attention_residual': False,
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
    'relational_receiver_measurement': 'adjoint',
    'relational_receiver_update': 'euler',
    'relational_receiver_damping': 1.0,
    'relational_routing': 'gaussian_kl',
    'tie_comparison_maps': False,
    'relational_key_noise': 1.0,
    'predictive_readout': 'kernel',
    'minimal_agent_inference': 'exact',
    'minimal_agent_cavi_steps': 1,
    'minimal_agent_channel_noise': 1.0,
    'predictive_covariance': 'passthrough',
    'predictive_covariance_floor': 0.0001,
    'predictive_covariance_floor_mode': 'prior',
    'predictive_covariance_ceiling': None,
    'predictive_covariance_checkpoint': False,
    'predictive_covariance_fp64_escalation': False,
    'predictive_covariance_precision': 'fp32',
    'predictive_covariance_spread_squares': False,
    'predictive_covariance_symmetric_pairs': False,
    'lambda_alpha_mode': 'state_dependent',
    'b0': 1.0,
    'c0': 1.0,
    'lambda_beta': 1.0,
    'mstep_self_coupling_weight': 0.0,
    'self_kl_max': None,
    'kappa_beta': 1.0,
    'learn_attention_temperature': False,
    'beta_attention_prior': 'causal_alibi',
    'alibi_slope': 1.0,
    'attention_window': 128,
    'norm_type_block': 'layernorm',
    'block_coordinates': 'local',
    'prior_handoff_rho': 1.0,
    'prior_handoff_sigma': 0.0,
    'use_block_mlp': True,
    'block_mlp_mode': 'coordinate',
    'block_mlp_covariance': 'passthrough',
    'block_mlp_activation': 'gelu',
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
    'block_norm_trace_cap': True,
    'block_norm_cap_moment_fallback': False,
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
    'm_decode_update_mode': 'adamw',
    'm_phi_sparse_rows': False,
    'm_phi_group_trust_radius': 0.1,
    'm_phi_group_retraction': 'chart',
    'm_phi_group_log_fallback': 'series',
    'm_phi_group_differential_mode': 'series',
    'm_phi_fisher_rotation_weight': 2.0,
    'm_phi_fisher_completion': 'rotation',
    'weight_decay': 0.05,
    'sigma_weight_decay': None,
    'mu_weight_decay': None,
    'decode_mu_weight_decay': None,
    'phi_weight_decay': 0.065,
    'eps': 1e-06,
    'kl_max': 100.0,
    'routing_kl_max_reference': 'absolute',
    'pair_kl_gemm_tol': None,
    'relational_tile_mib': 32,
    'relational_diag_moments_kernel': 'torch',
    'sigma_max': 100.0,
    'e_mu_q_trust': 2,
    'e_sigma_q_trust': 5.0,
    'grad_clip': 1.0,
    'grad_clip_scope': 'role',
    'amp_dtype': None,
    'fp32_matmul_precision': 'highest',
    'projected_factorization': 'gram',
    'decode_av_precision': 'fp64',
    'transport_chart_max_norm': 20.0,
    'frame_pair_escalation_tol': None,
    'phi_mstep_max_matrix_norm': None,
    'phi_frame_condition_max': None,
    'val_diagnostic_batches': 1,
    'diagnose_canonical_free_energy': False,
    'diagnose_agent_elbo': False,
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
                  use_block_mlp=False, layernorm_affine=False,
                  decode_unigram_prior=False, untie_decode_bank=False,
                  beta_attention_prior='causal', fp32_matmul_precision=None,
                  attention_mode='projected_predictive', predictive_covariance='moment_matched',
                  attention_residual=True,
                  decode_ranking_fp64_escalation=False, decode_av_precision='fp32',
                  e_mu_q_trust=None, block_mlp_expansion=2,
                  generate_embedding_figures=False,
                  skip_belief_sigma_update=False)
    values.update(overrides)
    return test_config(**values)


# The live route of the train_vfe3.py launcher at tiny scale. Every key the launcher's ``config`` dictionary sets is
# taken from it, so the route switches (readout, MLP mode, family, decode, routing, prior, rope, norms, residual,
# alpha form, ...) follow the launcher; _LIVE_TINY then shrinks the shape and run fields to a CPU unit test. A field
# the launcher omits runs at its VFE3Config default, which _LIVE_OMITTED pins here as a test-owned value wherever
# the legacy baseline differs (test_live_route_profile checks every omitted field against its default). The legacy
# baseline above (projected_gradient, Euler, state-dependent alpha) stays the default profile; this one is opt-in.
_LIVE_OMITTED: dict[str, Any] = {
    'decode_av_precision': 'fp32',
    'decode_ranking_fp64_escalation': False,
    'skip_belief_sigma_update': False,
    'transport_chart_max_norm': 50.0,
}

_LIVE_TINY: dict[str, Any] = {
    'vocab_size': 7,
    'embed_dim': 8,                   # heads of 4: p-RoPE at rope_fraction >= 0.5 still rotates a pair
    'n_heads': 2,
    'max_seq_len': 4,
    'eval_stride': None,              # the launcher's stride exceeds the tiny context
    'batch_size': 2,
    'n_layers': 2,
    'max_steps': 10,
    'warmup_steps': 1,
    'block_mlp_expansion': 2,
    'kl_max': None,                   # None: the launcher's 8 K at the tiny K; a launcher number scales (live_tiny)
    'amp_dtype': None,                # CPU unit tests run in fp32
    'log_interval': 0,                # run-only fields: no unit test reaches the logging or saving of train()
    'eval_interval': 0,
    'checkpoint_interval': 0,
    'generate_embedding_figures': False,
}


def launcher_values() -> dict[str, Any]:
    """Return a fresh copy of the ``config`` dictionary of ``train_vfe3.py`` (imported on first use)."""
    import train_vfe3
    return deepcopy(train_vfe3.config)


def live_kl_max(launcher: dict[str, Any]) -> "float | None":
    r"""The tiny profile's ``kl_max``: :math:`\mathrm{kl\_max}_{\mathrm{launcher}}K_{\mathrm{tiny}}/K_{\mathrm{launcher}}`.

    The divergence clamp scales with the belief dimension (its default is :math:`8K`), so the tiny profile keeps the
    launcher's nats per dimension and the clamp binds in the same regime. ``None`` (the :math:`8K` default) stays
    ``None`` (audit 2026-10-17, item 38).
    """
    kl_max = launcher.get("kl_max")
    if kl_max is None:
        return None
    return float(kl_max) * _LIVE_TINY['embed_dim'] / launcher["embed_dim"]


def live_tiny(launcher: dict[str, Any]) -> dict[str, Any]:
    """``_LIVE_TINY`` with ``kl_max`` scaled from ``launcher`` (:func:`live_kl_max`)."""
    return {**_LIVE_TINY, 'kl_max': live_kl_max(launcher)}


def live_values(**overrides: Any) -> dict[str, Any]:
    """Return a fresh complete live-route configuration dictionary with local overrides."""
    launcher = launcher_values()
    return deepcopy({**_CONFIG_VALUES, **_LIVE_OMITTED, **launcher, **live_tiny(launcher), **overrides})


def live_config(**overrides: Any) -> VFE3Config:
    """Construct the validated launcher route at tiny scale."""
    return VFE3Config(**live_values(**overrides))


live_config.__test__ = False


def make_relational_operators(*args: Any, **overrides: Any) -> Any:
    """RelationalOperators whose config-mirroring keywords come from the test baseline.

    ``attention_residual``, ``predictive_covariance``, ``receiver_update``,
    ``content_link_chart``, ``minimal_agent_inference``, ``kl_max_reference`` and
    ``rope_comparison_covariance`` have no constructor default; ``receiver_measurement`` mirrors the
    baseline's ``"adjoint"`` rather than the constructor default. Explicit keyword arguments override the
    baseline values (the comparison covariance is ``"full"``, the operator spelling of every rope_mode but
    ``"comparison_invariant"``).
    """
    from vfe3.model.relational_operators import RelationalOperators
    values = _CONFIG_VALUES
    baseline = dict(attention_residual=values['attention_residual'],
                    predictive_covariance=values['predictive_covariance'],
                    receiver_update=values['relational_receiver_update'],
                    receiver_measurement=values['relational_receiver_measurement'],
                    content_link_chart=values['content_link_chart'],
                    minimal_agent_inference=values['minimal_agent_inference'],
                    kl_max_reference=values['routing_kl_max_reference'],
                    rope_comparison_covariance='full')
    return RelationalOperators(*args, **{**baseline, **overrides})


# PriorBank's config-mirroring keywords are required (audit 2026-10-17, item 55). Unit tests that build a bank directly
# take them from this test-owned baseline, the constructor defaults before the keywords became required, so each test
# states only the keywords it exercises.
_PRIOR_BANK_BASELINE: dict[str, Any] = {
    'mu_init_std': 0.02,
    'sigma_init': 1.0,
    'phi_scale': 0.01,
    'decode_tau': 1.0,
    'eps': 1e-6,
    'family': 'gaussian_diagonal',
    'encode_mode': 'per_token',
    'decode_mode': 'diagonal_chunked',
    'decode_chunk_size': 8192,
    'decode_av_precision': None,
    'decode_ce_checkpoint': 'auto',
    'decode_ce_triton': False,
    'decode_ranking_fp64_escalation': False,
    'unigram_kappa': 1.0,
    'decode_unigram_prior': False,
    'untie_decode_bank': False,
    'decode_coordinates': 'local',
    'prior_covariance_table': 'diagonal',
}


def make_prior_bank(*args: Any, **overrides: Any) -> Any:
    """A ``PriorBank`` whose required config-mirroring keywords come from ``_PRIOR_BANK_BASELINE``."""
    from vfe3.model.prior_bank import PriorBank
    return PriorBank(*args, **{**_PRIOR_BANK_BASELINE, **overrides})
