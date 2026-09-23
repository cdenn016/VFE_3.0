"""Explicit tiny profiles shared by runtime and relational contract tests."""
from vfe3.config import VFE3Config


def tiny_config(**overrides: object) -> VFE3Config:
    """Pin behavior independently of user-edited experiment defaults."""
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                  n_layers=1, n_e_steps=1, e_phi_lr=0., norm_type_block='none',
                  norm_type_final='none', use_block_mlp=False, layernorm_affine=False,
                  decode_bias=False, decode_unigram_prior=False, untie_decode_bank=False,
                  beta_attention_prior='causal', fp32_matmul_precision=None,
                  decode_ranking_fp64_escalation=False, decode_av_precision='fp32',
                  e_mu_q_trust=None, block_mlp_expansion=2, transport_clamp_monitor=False,
                  generate_embedding_figures=False, exp_fp64_mode='dim',
                  exp_fp64_norm_threshold=15., full_cov_congruence_certification=False,
                  skip_belief_sigma_update=False)
    values.update(overrides)
    return VFE3Config(**values)
