# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 12:28:43 2026

@author: chris and christine
"""

BEST_K=320_n_head_config={
    
    
    m_p_mu_lr                 = 0.009,
    m_p_sigma_lr              = 0.001,
    m_phi_lr                  = 0.0015,

    
    
    m_relational_score_lr     = 0.01,   # None inherits m_p_mu_lr; 0 freezes both score maps
    m_relational_content_lr   = 0.01,   # None inherits m_p_mu_lr; 0 freezes T/O
    


    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.04,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.025,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)


    m_head_mixer_lr           = 0.025,     #  post-belief Schur HeadMixer parameters
    m_block_mlp_lr            = 0.001,     
    
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 15,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.035,      # std
    pos_phi_scale             = 0.02,      # learned-table init scale AND frozen per-position step

    pos_phi_project_slk       = False,      # per-block trace projection (det Omega = 1)
    
    relational_mode                  = "projected_filtering",  # "off" | "projected_filtering"

    qk_frame_mode                    = "tied",   # "independent": separate Q/K frames for routing AND content
    
    relational_input_scope           = "full",         # each head reads all K coordinates; "head" is legacy behavior

    relational_readout_mode          = "predictive_content",  # Transformers:"predictive_content", legacy: "conditional_update"
    relational_output_scope          = "full",                  # transformers: "full": requires predictive_content, legacy: 'head'
    #dont use headmixer for transformer settings
    
    }





































SINGLE_ATTENTION_config_K100 = dict(
    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)

    embed_dim                 = 100,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 4,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 128,                 # N, context length
    eval_stride               = 8,

    batch_size                = 64,
    max_steps                 = 15000,

    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations
    s_e_step_n_iter           = None,                # None = same as n_e_steps

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,
    
    fp32_matmul_precision     = "highest",       # None (leave as-is) | "highest" | "high" | "medium"
    
    learned_beta_prior_residual  = False,     # learned bounded B_ij from fixed pair-energy features
    
    #################################
    # belief family
    #################################


    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full"

    

    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 15,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.035,      # std
    pos_phi_scale             = 0.02,      # learned-table init scale AND frozen per-position step


    #################################
    #        Encode/Decode          #
    #################################
    use_prior_bank                    = False,   # True: KL-to-prior decode (pure path). False: linear projection
                                              
        
    decode_bias                       = True,     # only if use_prior_bank = False
    use_head_mixer                    = True,      # opt-in Schur-commutant head mixer (needs >=2 blocks: block_glk heads OR an so_n irrep tower with per-isotypic-component mixing);
    use_priorbank_head_evidence_mixer = False,
                                         
    
    decode_mode                       = 'linear',  #"full_chunked", "diagonal_chunked", "expected_likelihood_chunked", "full", "family", "family_chunked"   
    encode_mode                       = "per_token",   #"per_token_additive"
    
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.008,  #prior_bank = True temperature
    
    
    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,      
    block_mlp_expansion       = 10,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.00,
    m_block_mlp_lr            = 0.001,     # 0.0025

    # (counts train-loop iterations, 1:1 with optimizer steps INCLUDING under grad_accum_steps>1,
    # which chunks intra-step -- see vfe3/train.py::train_step docstring; diverges only when a step's
    # update is dropped by the NaN/Inf skip_step guard, see spec Sec.4)
    omega_compact_storage        = False,     # opt-in compact (V,H,d,d)/(V,d,d) block storage (equal-block groups)
    omega_reorth_every            = 0 ,        # SO-group re-orthogonalization cadence in M-steps (0 = off)
    phi_reflection               = "off",      # phi-path det<0 via R*exp(phi): 'off' (default) | 'init_seed' | 'metropolis'
    phi_metropolis_temperature   = 1.0,        # T in bounded phi Metropolis accept min(1, exp(-dF/T)); >0
    phi_metropolis_every         = 1,        # successful-update cadence for bounded phi Metropolis proposals; >=1
    phi_metropolis_proposals     = 10,       # bounded phi proposals per firing update; >=1
    
    
    

    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    s_frame_mode              = "tied",       # "tied" | "phi_tilde" (independent model-channel gauge frame)


    

    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    transport_chart_max_norm  = None,
    phi_mstep_max_matrix_norm = 12,

    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor

    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    phi_retract_mode          = "bch",                # "euclidean" | "bch"
    spd_retract_mode          = "spd_affine",         # SPD covariance retraction (registry: "spd_affine" | "log_euclidean")


    gauge_group               = "block_glk",    # "block_glk" | "so_n" (so_n awaits reconnection to the projected routes)

    group_n                   = None,                # so_n only; must stay None for block_glk
    irrep_spec                = None,                # so_n only; must stay None for block_glk

    use_cg_coupling           = False,               # so_n only: CG cross-type coupling
    cg_covariance_mode        = "passthrough",       # "passthrough" (means-only, pure) | "delta_full"
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF)

    ####################################
    # Non-Flat Connection - Regime II
    ####################################

    cocycle_relaxation        = 1.0,        # regime_ii homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    link_alpha                = 1.0,        # regime_ii_link only
    link_soft_cap             = 6.0,        # regime_ii_link only
    close_basis               = False,
   
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope"
    pos_phi_compose           = "group_product",     # composition chart: "bch" | "euclidean" | "group_product"
   

    rope_base                 = 100.0,               # rotary frequency base
    rope_insertion            = "right",             # "right" (PURE) | "left" (LEGACY)
    rope_on_value             = False,

    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "constant",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",                   # "constant" | "state_dependent"

    b0                         = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                         = 1.0,                 # state-dependent alpha shape (numerator)

    lambda_alpha               = 1,          # constant self-coupling value
    lambda_h                   = 0.0,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF)


    b0_h                       = 1.0,        # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h                       = 1.0,        # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,       # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)


    ##################################################
    #              Attention Energy:
    # lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j)
    ##################################################

    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)
    lambda_gamma               = 0,       # model-channel coupling (0 = OFF)

    
    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################

    kappa_beta                   = 1,        # tau = kappa * sqrt(d_head); kappa=1 -> Vaswani temperature
    kappa_gamma                  = 1,        # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)

    learnable_kappa_beta         = False,    # learn per-head kappa_beta = exp(log_kappa_beta)
    learnable_kappa_gamma        = False,    # learn per-head kappa_gamma

    beta_attention_prior         = "causal_alibi_noself",
    
    gamma_attention_prior        = "causal_alibi_noself",
    
    
    

    alibi_slope                  = 1,
    attention_window             = 128,
    

    t5_num_buckets               = 32,
    t5_max_distance              = 128,
    t5_learnable_bias            = False,    # learn the per-bucket T5 bias table b_{i-j}

     learned_beta_prior_residual  = False,        # learned bounded B_ij from fixed pair-energy features    
     gamma_as_beta_prior          = False,        # fold DETACHED gamma posterior into beta's prior; needs lambda_gamma > 0
     gamma_prior_weight           = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
     lambda_twohop                = 0.0,         # two-hop coupling F2 (0 = OFF)   
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
   
    e_q_mu_lr                 = 1,
    e_q_sigma_lr              = 0.001,
    

   
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################

    r_update_mode             = "gradient",          # "gradient" (AdamW M-step) | "barycenter"
    prior_source              = "token",     # belief prior p_i: "token" or "model_channel"
    learnable_r               = False,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = False,

    e_s_mu_lr                 = 0.65,
    e_s_sigma_lr              = 0.01,

    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################

    m_p_mu_lr                 = 0.005,
    m_p_sigma_lr              = 0.005,
    m_phi_lr                  = 0.00275,

    m_s_phi_lr                = 0.007,


    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.005,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.025,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)

    
    m_head_evidence_lr        = 0.001,     #  PriorBank-native KL irrep evidence weights
    m_head_mixer_lr           = 0.075,     #  post-belief Schur HeadMixer parameters
    
    
    min_lr_frac               = 0.01,
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine           = False,
    norm_type_block            = "none",             # "none" | "mahalanobis"
    norm_type_final            = "none",             # "none" | "mahalanobis"

    prior_handoff_rho          = 1,                  # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma        = 0.1,                # sigma damping in [0,1] (0.0 = frozen at embedding)

    #################################
    #        Numerical Safety
    #################################

    e_mu_q_trust               = 2,
    e_sigma_q_trust            = 1.0,
    sigma_max                  = 100.0,

    #################################
    #         Misc/Logging
    #################################
    amp_dtype                  = 'bf16',    # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32

    log_interval               = 100,       # console log every N steps (0 = off)
    eval_interval              = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 15000,     # save a resumable checkpoint every N steps (0 = off)
    eval_max_batches           = None,
    

    pos_phi_project_slk        = False,      # per-block trace projection (det Omega = 1)
    
    decode_av_precision        = "fp64",     # not in run: "fp32" | "fp64"



    # --- E-step update rule ---

    # --- randomized-depth E-step (recurrent-depth recipe) ---
    randomize_e_steps          = False,       # training forwards sample T ~ Uniform{e_steps_min..e_steps_max}
    e_steps_min                = 1,
    e_steps_max                = 3,
    e_steps_backprop_last      = 0,           # truncated backprop (0 = OFF)
    e_step_halt_tol            = None,        # eval halting (None = OFF)

    # --- decode / objective ---
    decode_unigram_prior       = False,       # add kappa*log pi_v (corpus unigram) to decode logits
    unigram_kappa              = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    untie_decode_bank          = False,       # use_prior_bank=True only: decode reads its OWN cloned (V,K) tables
    z_loss_weight              = 0,           # z-loss on the decode partition (0 = OFF)


    # --- attention / coupling ---
    gamma_as_beta_prior        = False,        # fold DETACHED gamma posterior into beta's prior; needs lambda_gamma > 0
    gamma_prior_weight         = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop              = 0.0,         # two-hop coupling F2 (0 = OFF)
    query_adaptive_tau         = False,       # per-query tau_i = tau_h (1 + c tr_h Sigma_i / d_h), detached
    query_tau_c                = 1.0,         # strength c >= 0 (read only when query_adaptive_tau=True)


    emission_mode              = "off",       # "off" | "shared" | "separate"
    emission_weight            = 0.0,         # 0.0 reproduces the "off" path byte-identically


     # --- training mechanics ---
     grad_clip                 = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role
     grad_clip_per_role        = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
     skip_belief_sigma_update  = True,        # skip the belief-channel sigma E-step update

     # --- compute reclamation (exactness-preserving perf; default OFF) ---
     exp_fp64_mode             = "norm",       # "dim" | "norm"
    
     share_refine_s_transport  = True,        # build the flat transport ONCE per forward
     compile_pair_kernel       = False,       # torch.compile the closed-form pair kernel

    
     
     decode_ranking_fp64_escalation       = True,    # False: keep chunked CE streamed in fp32.
                                                # True: strict score-interval audit; any uncertain
                                                # winner recomputes the complete (B,N,V) decode in
                                                # differentiable fp64 (~3.1 GiB logits here).
                                              
     

     full_cov_kl_precision                = "fp32_escalate",  # "fp64" | "fp32_escalate"

     mu_trust_cholesky_rounds             = 3,           

     
     
     emit_expensive_diagnostics           = True,
     check_parameter_motion               = True,
     parameter_motion_rel_tol             = 1e-06,
     evaluate_zero_e_steps_counterfactual = True,
     generate_figures                     = True,
     force_large_figures                  = False,

)


SINGLE_ATTN_CONFIG_k20 = dict(
    

    
    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)

    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 128,                 # N, context length
    eval_stride               = 8,

    batch_size                = 64,
    max_steps                 = 15000,

    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations
    s_e_step_n_iter           = None,                # None = same as n_e_steps

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,
    
       
    #################################
    # belief family
    #################################


    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full"

    
    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 15,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.035,      # std
    pos_phi_scale             = 0.02,      # learned-table init scale AND frozen per-position step


    #################################
    #        Encode/Decode          #
    #################################
    decode_bias                       = True,     # only for decode_mode='linear'
    use_head_mixer                    = True,      # opt-in Schur-commutant head mixer (needs >=2 blocks: block_glk heads OR an so_n irrep tower with per-isotypic-component mixing);
    use_priorbank_head_evidence_mixer = False,
                                         
    
    decode_mode                       = 'linear',  # prior-bank: diagonal_chunked/full_chunked/expected_likelihood_chunked/family_chunked
    encode_mode                       = "per_token",   #"per_token_additive" is ~1.5x faster at K=80 currently unablated
                                                       # "canonical_content_projected" REQUIRES decode_mode = 'full_*" and full gauissians
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.008,  #prior_bank = True temperature
    
    
    decode_unigram_prior              = True,       # add kappa*log pi_v (corpus unigram) to decode logits
    unigram_kappa                     = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    untie_decode_bank                 = False,       # prior-bank modes only: decode reads its OWN cloned (V,K) tables
    
    
    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,      
    block_mlp_expansion       = 10,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.00,
    


    #################################
    #     Observation Term          #
    #################################
    emission_mode              = "off",       # "off" | "shared" | "separate"
    emission_weight            = 0.0,         # 0.0 reproduces the "off" path byte-identically
    
    
    
    # (counts train-loop iterations, 1:1 with optimizer steps INCLUDING under grad_accum_steps>1,
    # which chunks intra-step -- see vfe3/train.py::train_step docstring; diverges only when a step's
    # update is dropped by the NaN/Inf skip_step guard, see spec Sec.4)
    omega_compact_storage        = False,     # opt-in compact (V,H,d,d)/(V,d,d) block storage (equal-block groups)
    omega_reorth_every            = 0 ,        # SO-group re-orthogonalization cadence in M-steps (0 = off)
    phi_reflection               = "off",      # phi-path det<0 via R*exp(phi): 'off' (default) | 'init_seed' | 'metropolis'
    phi_metropolis_temperature   = 1.0,        # T in bounded phi Metropolis accept min(1, exp(-dF/T)); >0
    phi_metropolis_every         = 1,        # successful-update cadence for bounded phi Metropolis proposals; >=1
    phi_metropolis_proposals     = 10,       # bounded phi proposals per firing update; >=1
    
    
    #################################
    #     Rank Reduction/Subspace   #
    #################################
    
    
    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    


    phi_retract_mode          = "bch", # "euclidean" | "bch"
    pos_phi_project_slk       = False,      # per-block trace projection (det Omega = 1)
    
    gauge_group               = "block_glk",    # "block_glk" | "so_n" (so_n awaits reconnection to the projected routes)

    group_n                   = None,                # so_n only; must stay None for block_glk
    irrep_spec                = None,                # so_n only; must stay None for block_glk

    use_cg_coupling           = False,               # so_n only: CG cross-type coupling
    cg_covariance_mode        = "passthrough",       # "passthrough" (means-only, pure) | "delta_full"
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF)


    ####################################
    # Non-Flat Connection - Regime II
    ####################################
   

    cocycle_relaxation        = 1.0,        # regime_ii homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    link_alpha                = 1.0,        # regime_ii_link only
    link_soft_cap             = 6.0,        # regime_ii_link only
   
    
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope"
   
    rope_base                 = 100.0,               # rotary frequency base
        
    rope_on_value             = False,


    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "state_dependent_per_coord",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "state_dependent_per_coord",                   # "constant" | "state_dependent"

    b0   = 1.0,                                # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0   = 1.0,                                 # state-dependent alpha shape (numerator)

    b0_h = 1.0,                              # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h = 1.0,                              # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    
    kappa_beta                   = 1,        # tau = kappa * sqrt(d_head)
    kappa_gamma                  = 1,        # tau_gamma = kappa_gamma*sqrt(d_head)

    beta_attention_prior         = "causal_alibi_noself",    
    gamma_attention_prior        = "causal_alibi_noself",
    
    
    alibi_slope                  = 1,
    attention_window             = 128,

    learned_beta_prior_residual  = False,        # learned bounded B_ij from fixed pair-energy features    
    gamma_as_beta_prior          = False,        # fold DETACHED gamma posterior into beta's prior; needs lambda_gamma > 0
    gamma_prior_weight           = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop                = 0.0,         # two-hop coupling F2 (0 = OFF)
  
    
    ##########################################################
    #              VFE Hyper-parameters
    #  Sum_i alpha_i * KL(q_i||p_i) + 
    #  lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) + ...
    ##########################################################

    lambda_alpha               = 1,          # constant self-coupling value
    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)
    
    
    lambda_gamma               = 0,       # model-channel coupling (0 = OFF)
    lambda_h                   = 0.0,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF)
    
    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,       # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)
   
    
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
    
    e_q_mu_lr                 = 1,
    e_q_sigma_lr              = 0.01,
    

   
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################
    s_e_step                  = False,
    prior_source              = "token",     # belief prior p_i: "token" or "model_channel"
    s_frame_mode              = "tied",       # "tied" | "phi_tilde" (independent model-channel gauge frame)
                                                # "tied" means the same as 'phi'

    e_s_mu_lr                 = 0.65,
    e_s_sigma_lr              = 0.01,


    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    phi_mstep_max_matrix_norm = 12,
    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor



    m_p_mu_lr                 = 0.03,
    m_p_sigma_lr              = 0.01,
    m_phi_lr                  = 0.01,

    m_s_phi_lr                = 0.007,


    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.005,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.025,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)

    m_head_mixer_lr           = 0.075,     #  post-belief Schur HeadMixer parameters
    m_block_mlp_lr            = 0.001,     
    
    m_head_evidence_lr        = None,      # active only with use_priorbank_head_evidence_mixer
    
    
    min_lr_frac               = 0.01,
   
    
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine           = False,
    norm_type_block            = "none",             # "none" | "mahalanobis"
    norm_type_final            = "none",             # "none" | "mahalanobis"

    prior_handoff_rho          = 1,                  # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma        = 0.1,                # sigma damping in [0,1] (0.0 = frozen at embedding)


    #################################
    #         Misc/Logging
    #################################

    log_interval               = 100,       # console log every N steps (0 = off)
    eval_interval              = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 15000,     # save a resumable checkpoint every N steps (0 = off)


    #################################
    #        Numerics
    #################################
    
    sigma_max                            = 100.0,
    e_mu_q_trust                         = 2,
    e_sigma_q_trust                      = 1.0,
   
   
    amp_dtype                            = 'bf16',        # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
    fp32_matmul_precision                = "highest",       # None (leave as-is) | "highest" | "high" | "medium"  
    
    grad_clip                            = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role
    grad_clip_per_role                   = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
    
    exp_fp64_mode                        = "norm",        # "dim" | "norm" norm ~1.5x faster at K=80
    transport_chart_max_norm             = 20,

    full_cov_kl_precision                = "fp32_escalate",  # "fp64" | "fp32_escalate"

    mu_trust_cholesky_rounds             = 3,           


    # Working precision of the decode's a_v reduction. a_v accumulates
    # terms of size 1/sigma_v and the result is divided by 2*tau (0.016 live), so the fp32 error is
    # amplified twice. Measured max |a_v - fp64| at K=20: 5.2e-6 at sigma_v=4, 2.5e-3 at 1e-2,
    # 2.2e-1 at 1e-4 -- the last being ~12 logits. NOTE the audit filed this as a cancellation
    # defect in the expanded square and proposed differencing before squaring; that was measured and
    # buys only ~1.8x, while the SAME algebra in float64 buys nine decades. It is a precision
    # problem, not a factorization problem. sigma_log_embed trains freely against only the eps=1e-6
    # floor, so this is harmless at init and about one order of magnitude of shrinkage from
    # material; watch decode_sigma_v_min. Default "fp32" is bit-identical to every run on disk.
    decode_av_precision                  = "fp32",     #  "fp32" | "fp64"


    # Strict decoder ranking guard. When enabled, analytic
    # fp32 decoders construct conservative score intervals and, if ANY row's winning interval
    # overlaps a competitor, recompute the complete (B,N,V) decode in differentiable float64.
    # That is an accuracy diagnostic, not a production-safe fallback: at B=64, N=128, V=50,257
    # the promoted logits alone occupy about 3.1 GiB before backward intermediates. Keep it
    # opt-in so the default chunked route remains streamed and bounded.
    decode_ranking_fp64_escalation       = False,             
     
     
     
                                                 
      
    skip_belief_sigma_update             = True,        # skip the belief-channel sigma E-step update
    compile_pair_kernel                  = False,       # torch.compile the closed-form pair kernel 
    
    check_parameter_motion               = True,
    evaluate_zero_e_steps_counterfactual = True,
    generate_embedding_figures           = False,
)


DIAGONAL_Dual_atten_config = dict(

    # ---- pulled from vfe3_runs/data/138.35_diag-optimized/config.json ----
    #      wikitext-103, cuda, 13,142,722 params, 2026-07-28T14:15:09
    #      Keys the run predates (block-MLP, head-mixer/evidence LRs, decode_av_precision,
    #      full-cov precision knobs, safe_cholesky_*) are marked "not in run".

   #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)

    embed_dim                 = 100,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 4,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 128,                 # N, context length
    eval_stride               = 8,

    batch_size                = 64,
    max_steps                 = 15000,

    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations
    s_e_step_n_iter           = None,                # None = same as n_e_steps

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,
    
    fp32_matmul_precision     = None,       # None (leave as-is) | "highest" | "high" | "medium"
    #################################
    # belief family
    #################################


    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full"

    

    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 15,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.035,      # std
    pos_phi_scale             = 0.02,      # learned-table init scale AND frozen per-position step


    #################################
    #        Encode/Decode          #
    #################################
    use_prior_bank                    = False,   # True: KL-to-prior decode (pure path). False: linear projection
                                              
        
    decode_bias                       = True,     # only if use_prior_bank = False
    use_head_mixer                    = True,      # opt-in Schur-commutant head mixer (needs >=2 blocks: block_glk heads OR an so_n irrep tower with per-isotypic-component mixing);
    use_priorbank_head_evidence_mixer = False,
                                         
    
    decode_mode                       = 'diagonal_chunked',  #"full_chunked", "diagonal_chunked", "expected_likelihood_chunked", "full", "family", "family_chunked"   
    encode_mode                       = "per_token",   #"per_token_additive"
    
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.008,  #prior_bank = True temperature
    
    
    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,      
    block_mlp_expansion       = 10,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.00,
    m_block_mlp_lr            = 0.001,     # 0.0025



    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    s_frame_mode              = "tied",       # "tied" | "phi_tilde" (independent model-channel gauge frame)


    

    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    transport_chart_max_norm  = None,
    phi_mstep_max_matrix_norm = 12,

    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor

    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    phi_retract_mode          = "bch",                # "euclidean" | "bch"
    spd_retract_mode          = "spd_affine",         # SPD covariance retraction (registry: "spd_affine" | "log_euclidean")


    gauge_group               = "block_glk",    # "block_glk" | "so_n" (so_n awaits reconnection to the projected routes)

    group_n                   = None,                # so_n only; must stay None for block_glk
    irrep_spec                = None,                # so_n only; must stay None for block_glk

    use_cg_coupling           = False,               # so_n only: CG cross-type coupling
    cg_covariance_mode        = "passthrough",       # "passthrough" (means-only, pure) | "delta_full"
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF)

    ####################################
    # Non-Flat Connection - Regime II
    ####################################

    cocycle_relaxation        = 1.0,        # regime_ii homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    link_alpha                = 1.0,        # regime_ii_link only
    link_soft_cap             = 6.0,        # regime_ii_link only
    close_basis               = False,
   
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope"
    pos_phi_compose           = "group_product",     # composition chart: "bch" | "euclidean" | "group_product"
   

    rope_base                 = 100.0,               # rotary frequency base
    rope_insertion            = "right",             # "right" (PURE) | "left" (LEGACY)
    rope_on_value             = False,

    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "constant",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",                   # "constant" | "state_dependent"

    b0                         = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                         = 1.0,                 # state-dependent alpha shape (numerator)

    lambda_alpha               = 1,          # constant self-coupling value
    lambda_h                   = 0.0,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF)


    b0_h                       = 1.0,        # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h                       = 1.0,        # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,       # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)


    ##################################################
    #              Attention Energy:
    # lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j)
    ##################################################

    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)
    lambda_gamma               = 0,       # model-channel coupling (0 = OFF)


    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################

    kappa_beta                   = 1,        # tau = kappa * sqrt(d_head); kappa=1 -> Vaswani temperature
    kappa_gamma                  = 1,        # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)

    learnable_kappa_beta         = False,    # learn per-head kappa_beta = exp(log_kappa_beta)
    learnable_kappa_gamma        = False,    # learn per-head kappa_gamma

    beta_attention_prior         = "causal_alibi_noself",
    learned_beta_prior_residual  = False,     # learned bounded B_ij from fixed pair-energy features
    
    gamma_attention_prior        = "causal_alibi_noself",

    alibi_slope                  = 1,
    attention_window             = 128,
    

    t5_num_buckets               = 32,
    t5_max_distance              = 128,
    t5_learnable_bias            = False,    # learn the per-bucket T5 bias table b_{i-j}

    
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
   
    e_q_mu_lr                 = 1,
    e_q_sigma_lr              = 0.001,
    

   
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################

    r_update_mode             = "gradient",          # "gradient" (AdamW M-step) | "barycenter"
    prior_source              = "token",     # belief prior p_i: "token" or "model_channel"
    learnable_r               = False,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = False,

    e_s_mu_lr                 = 0.65,
    e_s_sigma_lr              = 0.01,

    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################

    m_p_mu_lr                 = 0.005,
    m_p_sigma_lr              = 0.005,
    m_phi_lr                  = 0.00275,

    m_s_phi_lr                = 0.007,


    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.005,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.025,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)

    
    m_head_evidence_lr        = 0.001,     #  PriorBank-native KL irrep evidence weights
    m_head_mixer_lr           = 0.075,     #  post-belief Schur HeadMixer parameters
    
    
    min_lr_frac               = 0.01,
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine           = False,
    norm_type_block            = "none",             # "none" | "mahalanobis"
    norm_type_final            = "none",             # "none" | "mahalanobis"

    prior_handoff_rho          = 1,                  # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma        = 0.1,                # sigma damping in [0,1] (0.0 = frozen at embedding)

    #################################
    #        Numerical Safety
    #################################

    e_mu_q_trust               = 1,
    e_sigma_q_trust            = 1.0,
    sigma_max                  = 100.0,

    #################################
    #         Misc/Logging
    #################################
    amp_dtype                  = 'bf16',    # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32

    log_interval               = 100,       # console log every N steps (0 = off)
    eval_interval              = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 15000,     # save a resumable checkpoint every N steps (0 = off)
    eval_max_batches           = None,
    

    pos_phi_project_slk        = False,      # per-block trace projection (det Omega = 1)
    
    decode_av_precision        = "fp32",     # not in run: "fp32" | "fp64"



    # --- E-step update rule ---

    # --- randomized-depth E-step (recurrent-depth recipe) ---
    randomize_e_steps          = False,       # training forwards sample T ~ Uniform{e_steps_min..e_steps_max}
    e_steps_min                = 1,
    e_steps_max                = 3,
    e_steps_backprop_last      = 0,           # truncated backprop (0 = OFF)
    e_step_halt_tol            = None,        # eval halting (None = OFF)

    # --- decode / objective ---
    decode_unigram_prior       = False,       # add kappa*log pi_v (corpus unigram) to decode logits
    unigram_kappa              = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    untie_decode_bank          = False,       # use_prior_bank=True only: decode reads its OWN cloned (V,K) tables
    z_loss_weight              = 0,           # z-loss on the decode partition (0 = OFF)


    # --- attention / coupling ---
    gamma_as_beta_prior        = False,        # fold DETACHED gamma posterior into beta's prior; needs lambda_gamma > 0
    gamma_prior_weight         = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop              = 0.0,         # two-hop coupling F2 (0 = OFF)
    query_adaptive_tau         = False,       # per-query tau_i = tau_h (1 + c tr_h Sigma_i / d_h), detached
    query_tau_c                = 1.0,         # strength c >= 0 (read only when query_adaptive_tau=True)


    emission_mode              = "off",       # "off" | "shared" | "separate"
    emission_weight            = 0.0,         # 0.0 reproduces the "off" path byte-identically


     # --- training mechanics ---
     grad_clip                 = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role
     grad_clip_per_role        = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
     skip_belief_sigma_update  = True,        # skip the belief-channel sigma E-step update

     # --- compute reclamation (exactness-preserving perf; default OFF) ---
     exp_fp64_mode             = "dim",       # "dim" | "norm"
    
     share_refine_s_transport  = True,        # build the flat transport ONCE per forward
     compile_pair_kernel       = False,       # torch.compile the closed-form pair kernel

     full_cov_kl_condition_escalation     = False,   # False: production fast path, reuse the existing
                                                # KL Cholesky and promote only on factorization failure.
                                                # True: run a second spectral certificate across the
                                                # full pair grid and promote all rows if any row fails;
                                                # substantially slower at B=64, N=128, gaussian_full.
     
     decode_ranking_fp64_escalation       = False,    # False: keep chunked CE streamed in fp32.
                                                # True: strict score-interval audit; any uncertain
                                                # winner recomputes the complete (B,N,V) decode in
                                                # differentiable fp64 (~3.1 GiB logits here).
                                              
     

     full_cov_kl_precision                = "fp32_escalate",  # "fp64" | "fp32_escalate"

     mu_trust_cholesky_rounds             = 3,           

     
     
     emit_expensive_diagnostics           = True,
     check_parameter_motion               = True,
     parameter_motion_rel_tol             = 1e-06,
     evaluate_zero_e_steps_counterfactual = True,
     generate_figures                     = True,
     force_large_figures                  = False,

)





SO(N)_config = dict(    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)
    
    embed_dim                 = 48,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 8,                   # = number of irrep blocks (alibi priors require this)
    
    max_seq_len               = 128,                 # N, context length
    eval_stride               = None,
    
    batch_size                = 64,
    max_steps                 = 15000,
    
    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1 ,                   # T, E-step inner iterations
    s_e_step_n_iter           = None,              #None = same as n_e_steps
    
    seed                      = 6,
    warmup_steps              = 100,
    
    #################################
    # belief family
    #################################
    

    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full"
                                                     # "gaussian_frame_diagonal" (default OFF): covariance diagonal in the agent's OWN fiber frame,
                                                     # Sigma_i = U_i diag(sigma_i) U_i^T, which IS closed under GL(K) -- so the pair energy is the EXACT
                                                     # full-covariance divergence at diagonal cost instead of the diagonal-of-sandwich truncation.
                                                     # CONSEQUENCE: for the Regime-I coboundary the relative frame cancels, so Omega leaves the pair
                                                     # energy and the gauge no longer shapes the belief coupling. A DIFFERENT model, not a refinement.
                                                     # REQUIRES e_step_update="gradient": mm_exact is a hand kernel registered only for
                                                     # "gaussian_diagonal", so this family routes to the autograd oracle (slower per step).
                                                     # NOTE phi_embed receives NO gradient under this family (nothing on the belief path reads it),
                                                     # so it stays at init and decays under phi_weight_decay -- compare against gauge_transport="off".
                                                     #
                                                     # "gaussian_diagonal_exact" (default OFF): the SAME diagonal Gaussian as "gaussian_diagonal"
                                                     # (same self-divergence, decode, retraction); only the pair energy differs. Instead of pushing
                                                     # the key FORWARD and truncating Omega diag(s_j) Omega^T to its diagonal, it pulls the QUERY BACK
                                                     # by Omega^{-1} (a divergence is invariant under a common invertible pushforward), which leaves
                                                     # the second argument exactly diagonal and needs the first only through its diagonal and its
                                                     # determinant. The energy is then the EXACT congruence KL at the same cost, and unlike
                                                     # "gaussian_frame_diagonal" the gauge STAYS in the coupling (phi keeps its gradient).
                                                     # REQUIRES divergence_family="renyi" + renyi_order=1.0: the pullback identity is KL-specific
                                                     #   (a Renyi blend of the dense pushforward with a diagonal covariance does not reduce). Raises.
                                                     # REQUIRES e_step_update="gradient": no hand kernel is registered for this family, so it routes
                                                     #   to the autograd oracle (slower per step); mm_exact is rejected at config time.
                                                     # REQUIRES oracle_unroll_grad=True under e_step_gradient="unroll": the oracle otherwise returns a
                                                     #   DETACHED tangent and phi_embed receives no gradient -- silently a frozen-gauge run.
                                                     # TRANSPORT: flat vertex-factored cocycle (the fast route inverts by the pair transpose
                                                     #   Omega_ij^{-1} = Omega_ji) or a dense pairwise Omega (the O(N^2 K^3) reference route).
                                                     #   regime_ii / direct-link / RoPE-wrapped transports raise rather than silently truncating.
    
    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 4,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.06,      # std
    pos_phi_scale             = 0.02,                # learned-table init scale AND frozen per-position step
    
    
    #################################
    #        Encode/Decode          #
    #################################
    decode_bias               = True,     # only if use_prior_bank = False
    use_head_mixer            = True,      # opt-in Schur-commutant head mixer (needs >=2 blocks: block_glk heads OR an so_n irrep tower with per-isotypic-component mixing);
                                           # breaks strict equivariance under block_glk (exact at init); EXACT under tied_block_glk (full-cov)
    
    use_block_mlp             = True,      # opt-in coordinate mean-only residual MLP; not gauge-pure
    block_mlp_expansion       = 4,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.01,
    m_block_mlp_lr            = 0.004,      # None inherits m_p_mu_lr when the optional MLP is active
    
    
    
    use_priorbank_head_evidence_mixer = False,      
    
    
    use_prior_bank            = False,               # True: KL-to-prior decode (pure path). False: linear projection
                                                     # mu->logits ablation (encode stays on the prior bank)
    decode_tau                = 0.008,
    decode_mode               = 'diagonal_chunked',  #"full_chunked", "diagonal_chunked", "expected_likelihood_chunked", "full", "family", "family_chunked" (family/family_chunked: divergence-consistent KL-to-prior decode, use_prior_bank=True)
    encode_mode               = "per_token",   #"per_token_additive"
    
    
    
    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly: forces phi_scale=0, pos_phi='none',
                                              #     e_phi_lr=m_phi_lr=0; needs transport_mode='flat' + pos_rotation='none')
                                              #   | "frozen" (random fixed frame: e_phi_lr=m_phi_lr=0, phi_scale kept).
                                              #   NOT transport_mode (flat vs regime_ii). docs/hypotheses/2026-06-21-hypotheses.md
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    s_frame_mode              = "tied",       # "tied" | "phi_tilde" (independent model-channel gauge frame)
    
    
    omega_retract_mode        = "lie_exp",  # omega_direct group-manifold retraction: 'lie_exp' | 'cayley'
    omega_reflection           = "off",      # omega_direct det<0 seeding: 'off' (det>0 only) | 'init_seed' | 'metropolis'
    omega_metropolis_temperature  = 1.0 ,   # T in the metropolis det-sign accept min(1, exp(-dF/T)); >0
    omega_metropolis_every        = 100,       # cadence in optimizer steps for the metropolis det-sign sweep; >=1
   
    # (counts train-loop iterations, 1:1 with optimizer steps INCLUDING under grad_accum_steps>1,
    # which chunks intra-step -- see vfe3/train.py::train_step docstring; diverges only when a step's
    # update is dropped by the NaN/Inf skip_step guard, see spec Sec.4)
    omega_compact_storage        = False,     # opt-in compact (V,H,d,d)/(V,d,d) block storage (equal-block groups)
    omega_reorth_every            = 0 ,        # SO-group re-orthogonalization cadence in M-steps (0 = off)
    phi_reflection               = "off",      # phi-path det<0 via R*exp(phi): 'off' (default) | 'init_seed' | 'metropolis'; reuses omega_metropolis_temperature/every
    
    
    
    
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    transport_chart_max_norm  = None, 
    phi_mstep_max_matrix_norm = 12,
    
    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor
    
    phi_precond_mode          = "killing",             # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"
                                                       # so_n/sp_n: the per-block modes are REJECTED (one tied phi drives every
                                                       # block, so the generators do not partition) -- ambient Killing instead
                                                       # needs e_phi_lr>0
    
    phi_retract_mode          = "bch",                # "euclidean" | "bch"
    spd_retract_mode          = "spd_affine",         # SPD covariance retraction (registry: "spd_affine" | "log_euclidean")

    
    gauge_group               = "so_n",         # "block_glk" | "so_n" (so_n awaits reconnection to the projected routes)
                                                     # tied_block_glk: one shared GL(d) frame across heads (kron(I_n, gl(d)))

    # so_n / sp_n irrep towers (heads = irreps). Structure group SO(group_n) / Sp(group_n) with
    # group_n DECOUPLED from embed_dim; irrep_spec = [(label, mult), ...] blocks laid out in order,
    # block dims summing to embed_dim. Labels: so_n 'l<p>' = symmetric-traceless rank-p irrep
    # (group_n=3: spin-p, dim 2p+1); sp_n 'sym<p>' = Sym^p of the defining rep (dim C(2m+p-1, p)).
    # One shared per-token phi drives EVERY block (TIED gauge; n_gen = dim of the algebra), and
    # unequal block dims get per-head tau_h = kappa_h*sqrt(d_h). Both REQUIRED for so_n/sp_n,
    # must stay None for every other group. CONSTRAINTS for these groups: phi_precond_mode must
    # be "none"/"clip"/"killing" (the per-block modes are rejected -- tied generators do not
    # partition per block); use_head_mixer mixes per isotypic component (equal-mult towers mix copies; mults-one towers get scalar gains); alibi-family priors need
    # n_heads == number of blocks.
    # embed_dim=20 examples:
    #   so_n: group_n=3, irrep_spec=[("l2", 4)]                            # 4 equal spin-2 heads (mixer OK)
    #   so_n: group_n=3, irrep_spec=[("l0",1),("l1",1),("l3",1),("l4",1)]  # spins 0,1,3,4 = 1+3+7+9 (unequal: mixer = per-head scalar gains)
    #   sp_n: group_n=4, irrep_spec=[("sym2", 2)]                          # 2 equal Sym^2(R^4) heads, dim 10 each
    
    group_n                   = 3,                   # so_n/sp_n only: N of SO(N) / 2m of Sp(2m)  -> SO(3), n_gen = 3
    irrep_spec                = [("l0", 1),
                                 ("l1", 2),          # so_n/sp_n only: [(label, mult), ...]; dims sum == embed_dim
                                 ("l2", 2),          # SO(3) spin tower: 3 + 3 + 5 + 9 = 20 = embed_dim, 4 blocks
                                 ("l4", 1),
                                 ("l5", 2)],         #   heads: two spin-1 (vector) copies, one spin-2, one spin-4.
                                                     # WHY this spread: (a) UNEQUAL block dims 3/3/5/9 exercise the
                                                     #   per-head tau_h = kappa_h*sqrt(d_h) path (a single-dim tower
                                                     #   collapses to one scalar temperature);
                                                     # (b) the l1 MULTIPLICITY-2 isotypic component gives the head
                                                     #   mixer a genuine 2x2 Schur-commutant block to learn, while
                                                     #   l2 and l4 (mult 1) get scalar gains -- both mixer regimes
                                                     #   live in ONE run;
                                                     # (c) spin 4 is the widest block that still fits under K=20 with
                                                     #   room for lower ranks, so the tower spans a 3x dimension range
                                                     #   under ONE shared tied phi (all four blocks rotate together).

    use_cg_coupling           = False,               # so_n only: CG cross-type coupling (bilinear, exactly
                                                     # equivariant, means-only sigma; zero-init path weights)
    cg_covariance_mode        = "passthrough",       # CG covariance pushforward: "passthrough" (means-only, pure) |
                                                     # "delta_full" (delta-method sigma_out=sym(J Sigma J^T); needs family="gaussian_full")
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF; >0 adds once
                                                     # cg_energy_weight*mean_layers(mean_tokens D(q_post||q_pre)); needs use_cg_coupling=True)

    ####################################
    # Non-Flat Connection - Regime II
    ####################################
                                           
    cocycle_relaxation        =   1.0,        # regime_ii / regime_ii_covariant homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    close_basis               = False,
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope" (block-diagonal positional rotation folded into transport)
    pos_phi_compose           = "group_product",     # composition chart: "bch" | "euclidean" |"group_product"
               
    
    rope_base                 = 100.0,               # rotary frequency base
    rope_insertion            = "right",             # "right" (PURE: frame V_i = U_i R_i^T, so Omega_ij = U_i R(theta_j - theta_i) U_j^-1 --
                                                     #   the relative rotation sits BETWEEN query and key content, per GL(K)_attention.tex and
                                                     #   Su et al. Eq 16. Exactly (i-j)-dependent for any phi, and exactly gauge covariant.)
                                                     # "left" (LEGACY, what shipped pre-2026-07-26: Omega_ij = R_i U_i U_j^-1 R_j^T, rotations
                                                     #   OUTSIDE the content operator, so the transport is conjugated by the query's ABSOLUTE
                                                     #   angle. Measured on the K=20 rope run: 69% of the pair-energy spread is absolute-position
                                                     #   contamination and gauge covariance fails at 5.6. Kept only to reproduce old runs.)
    rope_on_value             = False,
    
    ######################################
    #                Self Energy:  
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "state_dependent_per_coord",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "state_dependent_per_coord",  # "constant" | "state_dependent" (lambda_h*=c0_h/(b0_h+KL); +R_h)
    
    b0                         = 1.0,                 # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0                         = 1.0,                 # state-dependent alpha shape (numerator)
       
    lambda_alpha               = 1,          # constant self-coupling value
    lambda_h                   = 0.25,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF; >0 creates s/r tables)
    
    
    b0_h                       = 1.0,        # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h                       = 1.0,        # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,      # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF) 0.25
    
    
    ##################################################
    #              Attention Energy: 
    # lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) 
    ##################################################
    
    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)    
    lambda_gamma               = 0.75,       # model-channel coupling (0 = OFF; >0 creates s tables, predictively inert by default)
         

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    
    kappa_beta                   = 1, #[1, 0.5],        # tau = kappa * sqrt(d_head); kappa=1 -> Vaswani temperature
    kappa_gamma                  = 1, #[1, 0.5],        # model-channel temperature tau_gamma = kappa_gamma*sqrt(d_head)

    learnable_kappa_beta         = False,       # learn per-head kappa_beta = exp(log_kappa_beta), init from kappa_beta above
                                             # (t5-exception family; freezes under detach/straight_through E-step)
    learnable_kappa_gamma        = False,       # learn per-head kappa_gamma (trains under any estimator on the scored
                                             # lambda_gamma>0 path; under s_e_step needs an 'unroll' E-step)

    beta_attention_prior         = "causal_alibi_noself",        # "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"
    gamma_attention_prior        = "causal_alibi_noself",        # model-channel prior pi^s_ij (same 7 keys): "uniform" | "causal" | "alibi" | "causal_alibi" | "windowed" | "causal_windowed" | "t5_relative_bias"

    alibi_slope                  = 1,

    t5_learnable_bias            = False,           # learn the per-bucket T5 bias table b_{i-j} (sanctioned NN exception, default OFF; needs a t5_relative_bias channel)

    precision_weighted_attention = False,        # down-weight high-variance keys: fold detached -log(b0 + tr Sigma_j)
                                                 # into the attention prior (diagnostic; OFF = position-only prior)
    precision_attention_b0       = 2.0,          # b0 in the per-key reliability -log(b0 + tr Sigma_j); > 0
    precision_attention_per_head = False,        # per-key reliability PER HEAD (trace over each block's coords) vs
                                                 # global (all K); needs precision_weighted_attention=True
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
    e_q_mu_lr                 = 0.9,
    e_q_sigma_lr              = 0.001,
    
    
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################
    
    r_update_mode             = "gradient",          # "gradient" (AdamW M-step; correct under s_e_step) | "barycenter" (closed-form forward-KL centroid of s; exact M-step in the scored s_e_step=False regime)
    prior_source              = "token",    # belief prior p_i: "token" or "model_channel"
    learnable_r               = False,               # un-freeze hyper-prior centroid r (empirical-Bayes)
    s_e_step                  = False,
    
    e_s_mu_lr                 = 0.85,
    e_s_sigma_lr              = 0.1,
    
    #################################
    #    Embedding/Priors M-step 
    #        Learning Rates
    #################################
        
    m_p_mu_lr                 = 0.015,     
    m_p_sigma_lr              = 0.01,     
    m_phi_lr                  = 0.01,
    
    m_s_phi_lr                = 0.007,  


    m_head_evidence_lr        = 0.001,     # PriorBank-native KL irrep evidence weights
    m_head_mixer_lr           = 0.001,     # post-belief Schur HeadMixer parameters       
    
    weight_decay              = 0.02,   
    phi_weight_decay          = 0.03,   
    sigma_weight_decay        = 0.01,           # AdamW decay for log-variance tables (None = inherit weight_decay;
                                             # 0.0 exempts sigma from the unintended log-sigma->0 pull)
    mu_weight_decay           = None,            # AdamW decay for the MEAN-role tables: mu_embed, s_mu_embed,
                                             # decode_mu_embed, output_proj_weight (None = inherit weight_decay).
                                             # Decoupled decay reaches EVERY row of a live embedding table on
                                             # every step, so rare rows are crushed faster than their gather
                                             # gradient restores them (measured: zero-count rows at norm 0.000,
                                             # count 1-15 BELOW init). 0.0 exempts the mean sector; set None to
                                             # restore the pre-2026-07-26 inherit-the-global behavior.

    min_lr                    = 0,       # absolute cosine-decay LR floor (0.0 = pure cosine)
    min_lr_frac               = 0.01,    # proportional LR floor, max(min_lr, frac*base); OFF
    
    #################################
    #     Layer Normalization 
    #        and Hand-Off
    #################################
    
    layernorm_affine          = False,
    norm_type_block           = "none",             # "none" | "mahalanobis"
    norm_type_final           = "none",              # "none" | "mahalanobis"
    
    prior_handoff_rho         = 1,                 # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma       = 0.1,                 # sigma damping in [0,1] (0.0 = frozen at embedding)
    
    #################################
    #        Numerical Safety
    #################################
    
    e_mu_q_trust              = None,
    e_sigma_q_trust           = 10.0,
    sigma_max                 = None,
    
    #################################
    #         Misc/Logging
    #################################     
    amp_dtype                 = None,      # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
        
    log_interval              = 100,       # console log every N steps (0 = off)
    eval_interval             = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval       = 15000,     # save a resumable checkpoint every N steps (0 = off)

    pos_phi_project_slk       = True,       # per-block trace projection (det Omega = 1)
    decode_av_precision       = "fp32",     # "fp32" | "fp64"



    # --- E-step update rule ---

    

    # --- decode / objective ---
    decode_unigram_prior      = False,       # add kappa*log pi_v (corpus unigram, data statistic) to decode logits
    unigram_kappa             = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)
    
    # decode_mode "expected_likelihood_chunked" is also new: sigma-aware Gaussian-convolution readout
    # log N(mu_q; mu_v, Sigma_q + Sigma_v) - select it above under use_prior_bank=True.
    untie_decode_bank         = False,       # use_prior_bank=True only: decode reads its OWN cloned (V,K) tables
    
   

    # --- attention / coupling ---
    gamma_as_beta_prior       = False,        # fold DETACHED gamma posterior into beta's prior (h->s->p->q);
                                             # needs lambda_gamma > 0
    gamma_prior_weight        = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop             = 0.0,         # two-hop coupling F2 = lam2 sum_ik (beta@beta)_ik KL_ik (0 = OFF;
                                             # exact composed transport, effective depth 2 at L=1)
   

    #   "off"      -- the PURE path and the default: no data term, byte-identical to before.
    #   "shared"   -- reuse the decode table (output_proj_weight). Whitepaper-faithful, no new
    #                 parameters, but V3 decodes position t against x_{t+1} while the emission pulls
    #                 toward x_t, so one linear map carries both roles with no nonlinear head between.
    #   "separate" -- own (V, K) table. Removes that competition at the cost of decoupling the factor
    #                 from the decoder that actually scores the prediction.
    # Both live modes are fixed-basis linear maps and therefore NOT gauge-equivariant, the same
    # footprint the linear decode already carries; "off" keeps the gauge-pure path.
    emission_mode    = "off",       # "off" | "shared" | "separate"
    emission_weight  = 0.0,            # 0.0 reproduces the "off" path byte-identically


     # --- training mechanics ---
     grad_clip                 = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role; None/0.0 disables
     grad_clip_per_role        = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
                                              # (global is phi-dominated and silently rescales other roles)
     skip_belief_sigma_update  = False,        # skip the belief-channel sigma E-step update (dead-compute ablation
                                              # for linear-decode configs; user asserts sigma has no consumer)

     # --- compute reclamation (exactness-preserving perf; default OFF) ---
     exp_fp64_mode             = "dim",       # "dim" (long-standing: fp64 when block dim >= 20) | "norm" (fp64 only
                                              # when clamped ||M||_F >= exp_fp64_norm_threshold; d_head=25 blocks
                                              # currently run fp64 PERMANENTLY under "dim")
     share_refine_s_transport  = True,        # build the flat transport ONCE per forward, share s-refine + belief
                                              # E-step (+ all layers); valid on flat/e_phi_lr=0/no-rope configs
     compile_pair_kernel       = False,       # torch.compile the closed-form pair kernel (eager fallback + warn)

     full_cov_kl_precision     = "fp32_escalate",  # "fp64" | "fp32_escalate"  (family="gaussian_full" only)
                                              # "fp64" = the historical unconditional float64 island.
                                              # "fp32_escalate" runs the (B,N,N,K,K) pair KL in float32 and
                                              # recomputes in float64 ONLY when a Cholesky actually fails --
                                              # data-keyed, not a blanket downcast. Verified single-pass through
                                              # cond(Sigma)~1e6, escalates at ~1e9; post-softmax attention moves
                                              # <=2e-4 on adversarial synthetic spectra, 4e-6 on real trained ones.
                                              

     mu_trust_cholesky_rounds = 3,

     emit_expensive_diagnostics           = True,    
     generate_figures                     = True,      # OFF: strict opt-out for all finalization plots, plot-only
                                                       # probes/model replays, and per-eval attention/gamma heatmaps.
                                                       # True re-enables; make_figures.py later rebuilds the replayable
                                                       # model-report, saved-probe, and persisted-history set.
     
)  
    
    
    
    
    
    
    
    
    
    
    
    
FULL_GAUSSIAN_config = dict(
    

    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)

    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 128,                 # N, context length
    eval_stride               = 8,

    batch_size                = 64,
    max_steps                 = 15000,

    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations
    s_e_step_n_iter           = None,                # None = same as n_e_steps

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,
    
       
    #################################
    # belief family
    #################################


    family                    = "gaussian_full", # "gaussian_diagonal" | "gaussian_full"

    
    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.001,   #0.065,   
    sigma_init                = 2.5,        
    phi_scale                 = 0.06,     
    pos_phi_scale             = 0.02,       
    

    #################################
    #        Encode/Decode          #
    #################################
    decode_bias                       = False,     # only for decode_mode='linear'
    use_head_mixer                    = False,      # opt-in Schur-commutant head mixer (needs >=2 blocks: block_glk heads OR an so_n irrep tower with per-isotypic-component mixing);
    use_priorbank_head_evidence_mixer = False,
                                         
    
    decode_mode                       = 'full_chunked',  # prior-bank: diagonal_chunked/full_chunked/expected_likelihood_chunked/family_chunked
    encode_mode                       = "per_token",   #"per_token_additive" is ~1.5x faster at K=80 currently unablated
                                                       # "canonical_content_projected" REQUIRES decode_mode = 'full_*" and full gauissians
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.01,  #prior_bank = True temperature
    
    
    decode_unigram_prior              = True,       # add kappa*log pi_v (corpus unigram) to decode logits
    unigram_kappa                     = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    untie_decode_bank                 = True,       # prior-bank modes only: decode reads its OWN cloned (V,K) tables
    
    
    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,      
    block_mlp_expansion       = 4,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.01,
    


    #################################
    #     Observation Term          #
    #################################
    emission_mode              = "off",       # "off" | "shared" | "separate"
    emission_weight            = 0.0,         # 0.0 reproduces the "off" path byte-identically
    
    
    
    # (counts train-loop iterations, 1:1 with optimizer steps INCLUDING under grad_accum_steps>1,
    # which chunks intra-step -- see vfe3/train.py::train_step docstring; diverges only when a step's
    # update is dropped by the NaN/Inf skip_step guard, see spec Sec.4)
    omega_compact_storage        = False,     # opt-in compact (V,H,d,d)/(V,d,d) block storage (equal-block groups)
    omega_reorth_every            = 0 ,        # SO-group re-orthogonalization cadence in M-steps (0 = off)
    phi_reflection               = "off",      # phi-path det<0 via R*exp(phi): 'off' (default) | 'init_seed' | 'metropolis'
    phi_metropolis_temperature   = 1.0,        # T in bounded phi Metropolis accept min(1, exp(-dF/T)); >0
    phi_metropolis_every         = 1,        # successful-update cadence for bounded phi Metropolis proposals; >=1
    phi_metropolis_proposals     = 10,       # bounded phi proposals per firing update; >=1
    
    
    #################################
    #     Rank Reduction/Subspace   #
    #################################
    
    
    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)
    gauge_parameterization    = "phi",        # "phi" | "omega_direct" (omega_direct: live-rejected, no belief source)
    


    phi_retract_mode          = "bch", # "euclidean" | "bch"
    pos_phi_project_slk       = True,      # per-block trace projection (det Omega = 1)
    
    gauge_group               = "block_glk",    # "block_glk" | "so_n" (so_n awaits reconnection to the projected routes)

    group_n                   = None,                # so_n only; must stay None for block_glk
    irrep_spec                = None,                # so_n only; must stay None for block_glk

    use_cg_coupling           = False,               # so_n only: CG cross-type coupling
    cg_covariance_mode        = "passthrough",       # "passthrough" (means-only, pure) | "delta_full"
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF)


    ####################################
    # Non-Flat Connection - Regime II
    ####################################
   

    cocycle_relaxation        = 1.0,        # regime_ii homotopy: 0.0 -> flat, 1.0 -> fully relaxed (ignored by flat)
    link_alpha                = 1.0,        # regime_ii_link only
    link_soft_cap             = 6.0,        # regime_ii_link only
   
    
    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope"
   
    rope_base                 = 100.0,               # rotary frequency base
        
    rope_on_value             = False,


    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "constant",  # "constant" | "state_dependent" | "state_dependent_per_coord"
    lambda_h_mode              = "constant",                   # "constant" | "state_dependent"

    b0   = 1.0,                                # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0   = 1.0,                                 # state-dependent alpha shape (numerator)

    b0_h = 1.0,                              # state-dependent lambda_h shape: lambda_h* = c0_h/(b0_h + KL(s||r))
    c0_h = 1.0,                              # state-dependent lambda_h shape (numerator); max precision c0_h/b0_h

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    
    kappa_beta                   = 1,        # tau = kappa * sqrt(d_head)
    kappa_gamma                  = 1,        # tau_gamma = kappa_gamma*sqrt(d_head)

    beta_attention_prior         = "causal_alibi_noself",    
    gamma_attention_prior        = "causal_alibi_noself",
    
    
    alibi_slope                  = 1,
    attention_window             = 128,

    learned_beta_prior_residual  = False,        # learned bounded B_ij from fixed pair-energy features    
    gamma_as_beta_prior          = False,        # fold DETACHED gamma posterior into beta's prior; needs lambda_gamma > 0
    gamma_prior_weight           = 0.5,         # mixture weight w in [0,1]: pi = (1-w) softmax(B) + w gamma
    lambda_twohop                = 0.0,         # two-hop coupling F2 (0 = OFF)
  
    
    ##########################################################
    #              VFE Hyper-parameters
    #  Sum_i alpha_i * KL(q_i||p_i) + 
    #  lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) + ...
    ##########################################################

    lambda_alpha               = 1,          # constant self-coupling value
    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)
    
    
    lambda_gamma               = 0,       # model-channel coupling (0 = OFF)
    lambda_h                   = 0.0,       # hyper-prior weight lambda_h * mean_i KL(s_i||r) (0 = OFF)
    
    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,       # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)
   
    
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    
    
    e_q_mu_lr                 = 0.3,
    e_q_sigma_lr              = 0.025,
    

   
    ####################################
    #       Model E-step LR's
    #      If s_e_step = True
    # and prior_source = 'model_channel'
    ####################################
    s_e_step                  = False,
    prior_source              = "token",     # belief prior p_i: "token" or "model_channel"
    s_frame_mode              = "tied",       # "tied" | "phi_tilde" (independent model-channel gauge frame)
                                                # "tied" means the same as 'phi'

    e_s_mu_lr                 = 0.65,
    e_s_sigma_lr              = 0.01,


    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    phi_mstep_max_matrix_norm = 12,
    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor



    m_p_mu_lr                 = 0.004,       
    m_p_sigma_lr              = 0.002,     
    m_phi_lr                  = 0.0025,    

    m_s_phi_lr                = 0.007,


    weight_decay              = 0.02,
    phi_weight_decay          = 0.00,    
    sigma_weight_decay        = 0.000,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.000,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)

    m_head_mixer_lr           = 0.001,     #  post-belief Schur HeadMixer parameters
    m_block_mlp_lr            = 0.004,     
    
    m_head_evidence_lr        = 0.001,      # active only with use_priorbank_head_evidence_mixer
    
    
    min_lr_frac               = 0.01,
   
    
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine           = False,
    norm_type_block            = "none",             # "none" | "mahalanobis"
    norm_type_final            = "none",             # "none" | "mahalanobis"

    prior_handoff_rho          = 1,                  # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma        = 0.1,                # sigma damping in [0,1] (0.0 = frozen at embedding)


    #################################
    #         Misc/Logging
    #################################

    log_interval               = 100,       # console log every N steps (0 = off)
    eval_interval              = 1500,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 15000,     # save a resumable checkpoint every N steps (0 = off)


    #################################
    #        Numerics
    #################################
    
    sigma_max                            = 100.0,
    e_mu_q_trust                         = 1,
    e_sigma_q_trust                      = 1.0,
   
   
    amp_dtype                            = None,        # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
    fp32_matmul_precision                = None,       # None (leave as-is) | "highest" | "high" | "medium"  
    
    grad_clip                            = 1.0,         # gradient clip: global L2 norm unless grad_clip_per_role
    grad_clip_per_role                   = True,        # clip grads per role (mu/sigma/phi) instead of one global norm
    
    exp_fp64_mode                        = "dim",        # "dim" | "norm" norm ~1.5x faster at K=80
    transport_chart_max_norm             = None,

   
    full_cov_kl_precision                = "fp32_escalate",  # "fp64" | "fp32_escalate"

    mu_trust_cholesky_rounds             = 3,           


    # Working precision of the decode's a_v reduction. a_v accumulates
    # terms of size 1/sigma_v and the result is divided by 2*tau (0.016 live), so the fp32 error is
    # amplified twice. Measured max |a_v - fp64| at K=20: 5.2e-6 at sigma_v=4, 2.5e-3 at 1e-2,
    # 2.2e-1 at 1e-4 -- the last being ~12 logits. NOTE the audit filed this as a cancellation
    # defect in the expanded square and proposed differencing before squaring; that was measured and
    # buys only ~1.8x, while the SAME algebra in float64 buys nine decades. It is a precision
    # problem, not a factorization problem. sigma_log_embed trains freely against only the eps=1e-6
    # floor, so this is harmless at init and about one order of magnitude of shrinkage from
    # material; watch decode_sigma_v_min. Default "fp32" is bit-identical to every run on disk.
    decode_av_precision                  = "fp32",     #  "fp32" | "fp64"


    # Strict decoder ranking guard. When enabled, analytic
    # fp32 decoders construct conservative score intervals and, if ANY row's winning interval
    # overlaps a competitor, recompute the complete (B,N,V) decode in differentiable float64.
    # That is an accuracy diagnostic, not a production-safe fallback: at B=64, N=128, V=50,257
    # the promoted logits alone occupy about 3.1 GiB before backward intermediates. Keep it
    # opt-in so the default chunked route remains streamed and bounded.
    decode_ranking_fp64_escalation       = False,             
     
     
                                                 
      
    skip_belief_sigma_update             = False,        # skip the belief-channel sigma E-step update
    compile_pair_kernel                  = False,       # torch.compile the closed-form pair kernel 
    
    check_parameter_motion               = True,
    evaluate_zero_e_steps_counterfactual = True,
    generate_embedding_figures           = False,
)