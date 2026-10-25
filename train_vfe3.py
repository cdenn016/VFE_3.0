r"""Click-to-run training entry for the VFE_3.0 transformer.

Edit the ``config`` dict below, pick a ``DATASET``,
then run ``python train_vfe3.py``. There is no CLI arg parsing.

The ``config`` dict exposes the commonly-tuned ``VFE3Config`` toggles, grouped exactly as
in ``vfe3/config.py``; each registry-backed ``*_mode`` / ``*_family`` / ``*_group`` field
lists its valid keys inline. Any ``VFE3Config`` field omitted here simply takes its dataclass
default -- add it to this dict to tune it. ``DATASET`` picks the cached corpus under
``~/.cache/tokenized_cache`` and ``vocab_size`` follows it by default, so click-to-run works out of
the box for the BPE corpora and for the word-level one. ``MAX_TOKENS`` caps the training stream
for fast smoke runs.

A missing tokenized cache raises ``FileNotFoundError`` rather than substituting toy data:
held-out numbers are never silently computed on a synthetic stream and mislabeled as the
real corpus. Build the corpus cache first: the BPE caches are prebuilt, and the word-level one is
produced by ``build_wikitext103_word_cache.py``.

A full ``max_steps`` run over the whole training split -- 101.4M word tokens, or 116.8M gpt2
tokens -- is a real (not smoke) job: run it on the CUDA interpreter (the RTX 5090), or drop
``MAX_TOKENS`` / ``max_steps`` for a quick slice on CPU.
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Third-party toolchain noise, filtered BEFORE torch is imported (triton emits its
# "Failed to find cuobjdump/nvdisasm" UserWarnings while torch is loading, and a warning
# already emitted cannot be un-emitted). Disassembly tools only -- absent on the Windows pip
# wheel, unused by this project, and irrelevant to numerics or throughput. See vfe3/quiet.py.
from vfe3.quiet import silence_toolchain_warnings
silence_toolchain_warnings()

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import (
    _tokenizer_tag,
    _validate_path_component,
    cached_token_count,
    make_dataloader,
    tokenizer_vocab_size as _tokenizer_vocab_size,
)
from vfe3.runtime import seed_everything
# The shared run-directory helpers (audit 2026-10-16 F96): atomic reservation (base_2, base_3, ... on a collision)
# and the rename by test perplexity.
from vfe3.experiment import rename_run_by_test_ppl, reserve_directory as _reserve_directory
from vfe3.train import _fmt_tau, evaluate, train





# Cached tokenized corpus (gpt2/tiktoken -> vocab_size 50257). Caches live in
# ~/.cache/tokenized_cache; a missing cache raises (no synthetic substitution).
#   "wikitext-103" | "wikitext-2"                -> gpt2 BPE,     vocab = 50257
#   "wiki-en" | "wiki-ja" | "wiki-ar"            -> cl100k BPE,   vocab = 100277
#   "wikitext-103-word"                          -> WORD-level,   vocab = 267734
# The word corpus is built by `python build_wikitext103_word_cache.py` (whitespace tokens, every
# training type kept, the corpus's own <unk> marker at id 0). The size is MEASURED for the pinned
# revision, not the 267735 quoted for the original release. NOTE it is a much larger vocabulary:
# phi_embed alone costs V * n_gen parameters, so at the shipped K=20/H=2 shape the model grows
# ~5.3x (13.1M -> 69.9M). Lower it by setting MIN_COUNT in build_wikitext103_word_cache.py and
# rebuilding, which drops rare types.
DATASET = "wikitext-103"

# Cap the *training* stream for fast smoke runs (the validation split is always read
# in full -- it is small). None = the full corpus (116.8M tokens for wikitext-103).
MAX_TOKENS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seeds: the only place the model seed is set (the config dictionary carries none). Click-to-run
# launches NUM_RUNS independent runs back-to-back, run i using SEEDS[i]; each run is fully independent
# (its own model, RNG, and artifacts dir -- the seed is in the run-folder label so they never collide).
# NUM_RUNS=1 runs SEEDS[0]. Example: NUM_RUNS=3, SEEDS=[3, 64, 23] trains all three seeds.
NUM_RUNS = 1
SEEDS    = [6, 54, 271]       # 54 is best seed so far; must list at least NUM_RUNS seeds

# DATA_SEED (EXP-1 variance floor): when set to an int, the TRAIN loader's shuffle order is fixed to
# this seed via an explicit generator, INDEPENDENT of the per-run model seed -- so a multi-seed run
# shares ONE batch order across seeds and the measured across-seed SD is init+optimization variance,
# NOT data-order noise. None seeds the train generator from each run's cfg.seed instead (a per-seed
# batch order with a resumable cursor, mirroring ablation._effective_data_seed).
DATA_SEED = 3    

# Where each run's artifacts go: vfe3_runs/<timestamp>_<label>/ while training (config.json,
# metrics.csv, checkpoints/, best_model.pt, test_results.json, summary.json, *.png), renamed to
# vfe3_runs/<test_ppl>_<label>/ (timestamp dropped) at finalize. None disables persistence.
RUN_ROOT = "vfe3_runs"

# True: run the fully gauge-covariant model. The config below is left as written; at launch the few keys that
# break gauge covariance are overridden (vfe3.gauge_covariant: frame transport on, a family that carries
# g Sigma g^T, block and decoder in reference coordinates, no fixed-coordinate caps or trust regions, total-KL
# self term) and the run is refused unless every gauge flag of the pure-path report holds. The content links
# (Lie-algebra L_ij), routing, readout, predictive covariance floor (both floor modes are covariant under the
# local frame change), decode covariance and learning rates are not touched. The overrides are logged and listed
# in the run's config.md. False: the config exactly as written.
fully_gauge_covariant = False

config = dict(
       
    #################################
    #            Training
    #################################
   
    vocab_size                       = _tokenizer_vocab_size(DATASET),   # follows DATASET's tokenizer (gpt2 50257 / cl100k 100277 / word 267735)
                                                                  
    embed_dim                        = 128,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                          = 4,                   # = number of irrep blocks (alibi priors require this)
    
    max_seq_len                      = 128,                 # N, context length
    eval_stride                      = 32,
    
    batch_size                       = 64,
    max_steps                        = 15000,
     
    min_lr_frac                      = 0.01,        # float in [0, 1]: LR floor as a fraction of each group's base LR (0 = decay to 0)
   
    
    n_layers                         = 4,                   # L, number of blocks
    n_e_steps                        = 1,                   # T, E-step inner iterations

    warmup_steps                     = 100,         # int >= 0: linear warmup steps before the cosine decay
    
    #################################
    #         Misc/Logging
    #################################

    log_interval                     = 500,       # console log every N steps (0 = off)
    eval_interval                    = 3000,      # periodic validation every N steps (0 = off)
    checkpoint_interval              = 15000,     # save a resumable checkpoint every N steps (0 = off) 
   
    checkpoint_keep_last             = 1,      # None (keep every checkpoint) | int >= 1 (keep the newest N)
    checkpoint_at_end                = True,     # True | False (also save a resumable checkpoint after the last step)
    block_variance_at_log            = True,     # True | False (also write diagnostics/block_variance.csv at log steps)
       
    loss_spike_skip_nats             = None,        # None (off) | float > 0 (reject an update whose loss exceeds the monitor's loss EMA by this many nats)
    
    kl_max                           = 128,
    #################################
    #           Modes               #
    #################################
    
    family                           = "gaussian_diagonal",     # "gaussian_diagonal" | "gaussian_full" | "gaussian_block" (per-head SPD blocks; equal head-block groups, kernel readout) | "gaussian_frame_diagonal" (U diag(c) U^T, diagonal in reference coordinates; kernel readout, coordinate MLP, linear/diagonal_chunked decode)
    decode_mode                      = 'diagonal_chunked',       # "linear" | "diagonal_chunked" | "full_chunked" | "expected_likelihood_chunked" (the *_chunked modes decode through the prior bank)
    prior_covariance_table           = "diagonal",               # "diagonal" | "block_cholesky" (frame-free (xi, C) table: gaussian_block/full, gauge_transport="off", untie_decode_bank)
    decode_covariance                = "belief",                 # "belief" (stack's final Sigma_q) | "encoded" (token's encoded covariance) | "constant" (sigma_init I); prior-bank decode only
    
    content_link_chart               = "exponential",        # "affine" Oh'=Oh | "first_order" Oh'=T^-1+Oh | "exponential" Oh'=T^-1+Oh phi_1(TOh)=e^{OhT}T^-1        
    attention_mode                   = "projected_predictive",  # "projected_predictive" | "projected_gradient"
    predictive_covariance            = 'moment_matched',        # "passthrough"  #TODO passthrough 
                                                                # "moment_matched" (kernel: exact moment-matched; minimal_agent: CAVI posterior) 
                                                                 # "spread_corrected" (minimal_agent mean_field: plus the label spread)
            
    encode_mode                      = "per_token",   # "per_token" | "per_token_additive" (additive only when phi carries no position)
    
    relational_routing               = "gaussian_kl",           # "gaussian_kl" | "noisy_query" | "noisy_key" | "mean_distance" (noisy_key: kernel readout, temperature 1)
    relational_receiver_measurement  = "inverse",    # "inverse" O'^-1, the dual measurement (VFE3Config default) | "adjoint" O'^T (projected_gradient only)
    relational_receiver_update       = "closed_form",           # "closed_form" (exact frozen-context update) | "closed_form_profiled" (state-dependent alpha: the profiled stationary point) | "euler" (Fisher step-projected_gradient only)
       
    predictive_readout               = "kernel",         # "kernel" | "minimal_agent" (projected_predictive only)
    minimal_agent_inference          = "mean_field",            # "exact" | "mean_field" (minimal_agent only)
    minimal_agent_cavi_steps         = 1,                       # mean_field only: coordinate-ascent sweeps (1 = labels at the input moments)
   
    block_coordinates                = "local",       # "local" | "reference" (LN and MLP applied as U Phi(U^-1 .))
    decode_coordinates               = "local",       # "local" | "reference" (reference: decode_mode "linear" | "diagonal_chunked" | "full_chunked")
             
    minimal_agent_channel_noise      = 1.0,         # initial channel noise n_h > 0
    relational_key_noise             = 1,         # relational_routing="noisy_key" only: initial per-head key noise n_h > 0 (learned; routes at temperature 1)
    predictive_covariance_floor      = 1e-4,        # moment_matched / spread_corrected only: f, the floor's scale (floor_mode "prior": f * Sigma_p, reference-invariant; "relative": f tr(C)/K U U^T)
    predictive_covariance_ceiling    = None,        # None (off) | c > floor, e.g. 100: Loewner ceiling Sigma^+ <= c * Sigma_p (a clamp; counted in metrics.csv)
     
    relational_receiver_damping      = 1.0,         # closed_form only: rho in (0, 1]; 1 = the frozen minimizer    
    relational_receiver_noise        = 1.0,         # channel floor nu I_d, nu > 0 (projected_gradient only)
     
    tie_comparison_maps              = False,           # S_K := S_Q (kernel readout): D_ii = 0, flat-alignment routing
    decode_second_order              = False,         # linear decode, diagonal family, local coords: train on CE(mu) + 1/2 tr(Sigma Cov_p(w)); pushes Sigma to its floor, ablation arm first
    
    attention_residual               = False,      #ignored with minimal_agent
    #################################
    #     Optional Block MLP        #
    #################################
    
    use_block_mlp                    = True,
    block_mlp_tie_layers             = False,          # True | False: one block MLP shared by every layer (depth as time; labels persist across layers)
    per_type_hidden_bias             = False,          # per-type (V x n_gen) bias on the MLP hidden layer; coordinate MLP, projected route
     
    block_mlp_activation             = "gelu",         # "gelu" | "silu" | "relu" (not read by gaussian_template / prototype_agents)   
    block_mlp_mode                   = "coordinate",   # "coordinate" | "prototype_agents" | "cue_result_memory" (exact reference route; gaussian_full dense or gaussian_frame_diagonal elementwise) | "canonical_frame" | "gaussian_template" | "gauge_gate"
    block_mlp_covariance             = "passthrough",   # "passthrough" | "posterior" (prototype_agents; required by cue_result_memory) | "delta_full" (block norm "none"/"layernorm", no post-norm)  | gaussian_template: "passthrough" only
    
      
    block_mlp_inference              = "mean_field",   # "mean_field" | "exact" (prototype_agents only)
    block_mlp_template_init          = "working_variance",      # "absolute" | "working_variance" (lambda_0, nu from one batch; prototype_agents only)
    block_mlp_cavi_steps             = 1,              # mean_field only: coordinate-ascent sweeps (1 = labels at the input moments)
    block_mlp_em_weight              = 0.0,            # prototype_agents only: EM M-step term on the templates (nu, lambda, pi); 0 = off
    block_memory_regression_rank     = None,           # cue_result_memory only: None (dense F_a = I + D_a) | int r in [1, K] (F_a = I + U_a V_a^T)
    block_memory_result_noise_ratio  = 0.01,           # cue_result_memory only: initial result noise Q_a / L_a > 0
    
    block_mlp_expansion              = 8,               # K*block_mlp_expansion = number of templates
    block_mlp_template_init_variance = 10.0,            # initial template variance lambda_0 > 0 (absolute init only) 
    block_mlp_template_init_pull     = 0.09,            # working_variance only: initial pull p in (0, 1)
    block_mlp_dropout                = 0.00,
    block_mlp_covariance_floor       = 1e-4,            # delta_full only: Sigma+ = J Sigma J^T + floor Sigma
    
  
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################
    
    block_norm_trace_cap             = True,               # True | False: cap the LayerNorm covariance pushforward at the sphere bound K max(gamma^2) (False: uncapped delta method)
    layernorm_affine                 = True,
    norm_type_block                  = "layernorm",        # "none" | "layernorm" | "mahalanobis" - often useful
    block_norm_covariance            = "passthrough",      # "passthrough" | "pushforward" | "pushforward_moment" (the same at the effective scale s^2 + tr(Pi Sigma Pi)/K with a span floor; a moment estimate at every Sigma/s^2)
    
    block_norm_cap_moment_fallback   = False,              # True | False: capped 'pushforward' only: where the cap binds, return the 'pushforward_moment' estimate instead of the capped delta method
   
    #################################
    #        Initialization
    #################################
       
    mu_init_std                      = 0.065,     # std of the random mean table mu_embed
    sigma_init                       = 15,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                        = 0.035,      # std
    pos_phi_scale                    = 0.02,      # learned-table init scale AND frozen per-position step
    
    decode_tau                       = 0.1,       # tau =1 principled bayes 0.01 works well for some reason
    unigram_kappa                    = 1,         # tempering on log pi_v (1.0 = exact Bayes class prior)
    decode_chunk_size                = 8192,
   
    decode_unigram_prior             = True,       # add kappa*log pi_v (corpus unigram) to decode logits    
    untie_decode_bank                = True,       # prior-bank modes only: decode reads its OWN cloned (V,K) tables
      
    #################################
    #          Gauge Group
    #################################
    gauge_transport                  = "on",           # "on" | "off" | "frozen" (random fixed frame)
    gauge_group                      = "block_glk",    # "block_glk" | "so_n" | "su_n" | "block_so" | "block_co" | "block_sl" | "block_borel" | "block_unipotent" | "block_glmc" (block_glmc: even d_head)
    gauge_tying                      = "per_head",     # None (group default) | "per_head" | "shared" (shared: block_glk, so_n, su_n only)
    
    group_n                          = None,                # so_n / su_n only: the N of SO(N) (>= 2) or SU(N) (>= 3); None for every other group
    irrep_spec                       = None,                # so_n: [(label, mult), ...] (labels "l<p>", "w<p>", "wp<m>", "wm<m>") | "random"; su_n: [("pq<p>_<q>", mult), ...] (p >= q; SU(3) dims pq1_0 6, pq1_1 8, pq2_0 12); None otherwise
    irrep_max_rank                   = None,                # irrep_spec="random" only: None (every rank that fits) | int >= irrep_min_rank
   
    use_cg_coupling                  = False,               # True | False: so_n / su_n with gauge_tying="shared" only, Clebsch-Gordan coupling between irrep blocks
    cg_covariance_mode               = "passthrough",       # "passthrough" | "delta_full" (J Sigma J^T + cg_covariance_floor Sigma; gaussian_full); use_cg_coupling only
    cg_covariance_floor              = 1e-6,                # float > 0: delta_full floor scale
    cg_energy_weight                 = 0.0,                 # float >= 0: post-CG moment divergence in the loss (> 0 needs use_cg_coupling)
   
    ####################################
    #       Positional Encoding
    #    gauge-frame PE (pos_phi): exact right factor exp(X_i) exp(Y_n)
    #     gauge-RoPE (rope_mode)
    ####################################

    pos_phi                          = "none",     # "none" | "learned" | "frozen"
    rope_mode                        = "comparison_invariant",  # "none" | "frame" | "frame_value" 
                                                 #"comparison" | "comparison_invariant" (rotation-invariant comparison)
    rope_base                        = 100,
    rope_fraction                    = 1,       # p-RoPE: fraction of each head's pairs rotated (fastest first); 1.0 = RoPE
    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    
    lambda_alpha_mode                = "state_dependent",  # "state_dependent" | "state_dependent_per_coord" (not gauge covariant)
    mstep_self_coupling_weight       = 0.0,      # float >= 0: alpha_hat of the M-step term alpha_hat * mean_i alpha_i D(q_i*||p_i), gradient through q*; 0.0 = off
    self_kl_max                      = None,     # None | float > 0 (float("inf") allowed): divergence clamp of the M-step self term; None = kl_max
    b0                               = 1.0,      # float | K-list (state_dependent_per_coord only): alpha_i = c0 / (b0 + D)
    c0                               = 1.0,      # float | K-list (state_dependent_per_coord only): alpha_i = c0 / (b0 + D)
  
    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    lambda_beta                      = 1.0,        # belief-coupling block weight (1.0 = pure F)
    kappa_beta                       = 1,        # tau = kappa * sqrt(d_head)
    learn_attention_temperature      = False,    # learn a per-layer, per-head log-temperature offset (starts at tau above)
    
    beta_attention_prior             = 'causal_alibi',    # "causal" | "causal_noself" | "causal_alibi" | "causal_alibi_noself" | "causal_windowed"
     
    #################################
    #         Belief E-step
    #################################    
    
    e_q_mu_lr                        = 0.2,
    e_q_sigma_lr                     = 0.01,
     
    e_mu_q_trust                     = 2,
    e_sigma_q_trust                  = 1.0,
     
    #################################
    #        M-step Settings  
    #################################
    
    m_phi_update_mode                = "adamw",        # "adamw" | "pullback_group" | "fisher_group" (belief-Fisher natural gradient)
    m_phi_group_retraction           = "chart",        # "chart" | "group_log" (exact Log(U e^{-eta xi}); pullback_group/fisher_group only)
    m_phi_group_log_fallback         = "series",       # "series" (certified series direction; structure-constant budget, block_glk d_head <= 16) | "eigenbasis" (off-strip blocks by the eigenbasis Psi inverse, kappa_2(U) > 1e3 rows take no step; no width budget); pullback_group with group_log only
    m_phi_group_differential_mode    = "series",       # "series" (certified power series) | "spectral" (Daleckii-Krein closed form, series fallback); pullback_group only
    
    m_phi_sparse_rows                = False,          # False (dense AdamW) | True (row-lazy AdamW on phi_embed: rows absent from the step take no moment update, step or decay; per-row bias correction; adamw only)
    phi_mstep_max_matrix_norm        = None,           # None (off) | float in (0, transport_chart_max_norm): interior radius for phi after each M-step (adamw projects rows back; pullback_group rejects the step)
    
    m_phi_group_trust_radius         = 0.1,       # pullback_group/fisher_group only: float > 0, bound on ||eta s xi||_F per row (exact under group_log; the chart route's 1e-6 residual certificate caps the step near 1e-2 off the identity)
    m_phi_fisher_rotation_weight     = 2.0,       # fisher_group only: completion weight lambda (under "rotation" it also damps Fisher-visible rotations)
    m_phi_fisher_completion          = "rotation",  # fisher_group only: "rotation" (lambda on every rotation) | "stabilizer" (lambda on the belief's stabilizer only: the belief-Fisher natural gradient on the visible subspace; not with pos_phi="learned")

    m_table_update_mode              = "fisher",   # "adamw" | "fisher" (natural-gradient M-step on mu_embed / sigma_log_embed; linear or untied decode)
    m_table_fisher_lr                = 0.1,       # fisher only: natural step size eta (the AdamW rates m_p_mu_lr / m_p_sigma_lr schedule it and set decay)
    m_table_fisher_trust_radius      = 0.5,       # fisher only: per-row Fisher-norm bound (KL <= r^2/2 per step to second order; decay outside)
    m_decode_update_mode             = "adamw",   # "adamw" | "fisher" (natural-gradient M-step on the untied decode bank at m_table_fisher_lr / m_table_fisher_trust_radius; untie_decode_bank=True, prior-bank decode)

    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################

    m_p_mu_lr                        = 0.005,     #0.025
    m_p_sigma_lr                     = 0.001,       
    m_phi_lr                         = 0.00075,     #0.003
    
    
    #################################
    #      Other M-Step LR's
    #################################
      
    m_pos_phi_lr                     = 0.003,      #0.0025?
     
    m_decode_mu_lr                   = 0.005,       # 0.025?
    m_decode_sigma_lr                = 0.001,       # 0.001
      
    m_relational_score_lr            = 0.01,        # 0.005 0.01
    m_relational_content_lr          = 0.0025,      # 0.005   0.0025
      
    m_block_mlp_lr                   = 0.005,      #0.0075?
    m_layernorm_lr                   = 0.005,      # 0.005
    
    m_per_type_hidden_bias_lr        = 0.0025,    
    
    #################################
    #        Weight Decays
    #################################
    weight_decay                     = 0.01,
    phi_weight_decay                 = 0.1,    
    sigma_weight_decay               = 0,      
    mu_weight_decay                  = 0,       

    decode_mu_weight_decay           = 0,      # decode-side means only (untied decode bank / linear readout); None = mu_weight_decay; try ~1e-3 to bound rare-type rows
    relational_score_weight_decay    = 0.0,    # float >= 0: decoupled AdamW decay of the query/key comparison maps only (raw_score_q, raw_score_k); 0 = undecayed, as before

    #################################
    #   Numerics/Speed
    #################################
    weight_ema_decay                 = None,         # None (off) | float in (0, 1): EMA of the trainable weights, scored at every eval and at the end
    weight_ema_start_frac            = 0.5,          # float in [0, 1]: the EMA starts at this fraction of max_steps
    weight_ema_update_interval       = 1,            # int >= 1: accepted updates between EMA updates (decay d^k per update)
    weight_ema_device                = "same",       # "same" (shadow on the parameters' device) | "cpu" (host memory; pair with interval > 1)
    
    amp_dtype                        = None,         # None (fp32) | "bf16" | "fp16". 
    
    grad_clip_scope                  = "group",      # "role" | "group" (one L2 clip per optimizer role, or per optimizer group)
    fp32_matmul_precision            = "high",       # None (leave as-is) | "highest" | "high" | "medium" ("high" permits TF32 on the decode/energy GEMMs)
    relational_diag_moments_kernel   = "triton",     # "torch" | "triton": diagonal comparison route (comparison_invariant) on CUDA: fused Triton diag(G) kernels, no materialized source root    
    decode_ce_checkpoint             = "auto",       # "auto" | "auto_aggregate" (saves memory with recompute)        
    decode_av_precision              = "fp32",   
    projected_factorization          = "gram",       # "gram" | "gram_certified" | "cholesky_qr2_certified" 
    
    routing_kl_max_reference         = "query_shifted",  # "absolute" | "row_min" | "query_shifted" (kernel router: kl_max caps D_ij, the gap D_ij - min_j D_ij, or D_ij less its query-only row constant)
    predictive_covariance_floor_mode = "max",            # "prior" | "relative" | "max" (the larger of the two)
    
    pair_kl_gemm_tol                 = None,         # None (off) | nats, e.g. 1e-3: rescore pair-KL GEMM slices whose error estimate exceeds it (~+45% step at K=128) 
    relational_tile_mib              = 512,          # int >= 1 (MiB): scorer and routed-spread tile budget; larger dispatches fewer kernels, more transient memory (A/B on CUDA before raising)
    phi_frame_condition_max          = 20,         # None (off) | kappa_max > 1, e.g. 20 or 1e4: after each step bound cond_2(U_i) <= kappa_max (adamw; block_glk/block_glmc/block_sl; inert for block_so/block_co/so_n/su_n)
    
    frame_pair_escalation_tol        = None,         # None (off) | tol, e.g. 1e-4: redo ill-conditioned frame pairs in fp64 (costly)
    decode_ranking_fp64_escalation   = False, 
    decode_logits_amp                = True,         # True | False: linear decode with amp_dtype set only: logits matmul under amp_dtype (CE stays fp32)
    decode_ce_triton                 = True,         # True | False: diagonal_chunked / full_chunked on CUDA: fused Triton chunk kernels (fp32-rounding differences)
    
    predictive_covariance_spread_squares  = False,   # True | False: moment_matched kernel readout: routed spread as a nonnegative sum of squares after the sandwich (no edge links)
    predictive_covariance_symmetric_pairs = False,   # True | False: moment_matched kernel readout, blockwise body: head pairs h <= g only, off-diagonal pairs doubled (no edge links)
   
    diagnose_canonical_free_energy        = False,   # log the canonical free energy (gauge-KL routing) at the captured beliefs
    diagnose_agent_elbo                   = False,   # True | False: per-layer ELBO, evidence and gap at eval (minimal_agent, prototype_agents); never in the loss
   
    deterministic                         = False,
    generate_embedding_figures            = False,
    
    predictive_covariance_checkpoint      = False,       # moment_matched only: recompute the (B,N,K,K) update in backward; bit-identical.
    predictive_covariance_fp64_escalation = False,       # True | False: moment_matched kernel readout: redo the failing sequences of a non-PD fp32 update in fp64 (counted in covariance_escalation_elements)
    predictive_covariance_precision       = "fp32",      # "fp32" | "fp64": moment_matched kernel readout: run the whole covariance update in float64 from fp32 beliefs (removes kappa(U)^2 rounding; costly)
    
   
)



def _banner(model, cfg: VFE3Config, dataset: str, device: str, n_steps: int,
            train_loader=None, full_corpus_tokens: 'int | None' = None) -> str:
    from vfe3.train import _fmt_decode_lrs, _fmt_mixer_lr, _fmt_table_lrs, coverage_lines, parameter_report
    rep = parameter_report(model, device=device)
    bar = "=" * 64
    cov = (coverage_lines(train_loader, n_steps, dataset, full_corpus_tokens=full_corpus_tokens)
           if train_loader is not None else [])
    live_note = (f" ({rep['live']:,} live, {rep['dead']:,} dead)"
                 if rep["probed"] and rep["dead"] else "")
    dead_line = ([" dead under config (no grad): "
                  + ", ".join(n.replace("prior_bank.", "") for n in rep["dead_names"])]
                 if rep["probed"] and rep["dead_names"] else [])
    # Live parameters whose init-probe gradient is identically zero (audit 2026-10-16 F110), e.g. the first layer of
    # a zero-initialized gate: benign at init, but named so a parameter that never moves is not mistaken for live.
    zero_names = rep.get("zero_grad_names", [])
    zero_line = ([" zero gradient at the init probe (counted live): "
                  + ", ".join(n.replace("prior_bank.", "") for n in zero_names)]
                 if rep["probed"] and zero_names else [])
    probe_note = "" if rep["probed"] else "  [live/dead probe failed]"
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {rep['total']:,} params{live_note}{probe_note} | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  "
        f"heads={len(model.group.irrep_dims)}  "  # runtime attention heads = irrep blocks
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        *dead_line,
        *zero_line,
        f" M-LRs: {_fmt_table_lrs(cfg)}  "
        f"phi={cfg.m_phi_lr}",
        f" decoder M-LRs: {_fmt_decode_lrs(cfg)}",
        f" auxiliary M-LRs: "
        f"block_mlp={_fmt_mixer_lr(cfg.m_block_mlp_lr, cfg.m_p_mu_lr)}",
        f" VFE: lambda_alpha_mode={cfg.lambda_alpha_mode}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}",
        f" seed={cfg.seed}",
        bar,
    ])


def _validated_data_seed() -> Optional[int]:
    if DATA_SEED is not None and (type(DATA_SEED) is not int or DATA_SEED < 0):
        raise ValueError(
            f"DATA_SEED must be None or an exact non-negative integer, got {DATA_SEED!r}")
    return DATA_SEED


def _effective_data_seed(cfg: VFE3Config) -> int:
    """Return DATA_SEED, or the run's own ``cfg.seed`` when DATA_SEED is None."""
    data_seed = _validated_data_seed()
    return cfg.seed if data_seed is None else data_seed


def _select_loader(
    dataset: str,
    cfg:     VFE3Config,

    *,
    split:   str = "train",
) -> DataLoader:
    r"""Loader for ``dataset``/``split``. A missing cache raises ``FileNotFoundError``.

    ``MAX_TOKENS`` caps only the train split (smoke runs); the small validation/test splits are
    always read in full. The loader never substitutes synthetic data for a missing real corpus --
    that would silently compute held-out numbers on a toy stream and mislabel them as the corpus.

    Split-aware loader semantics: only TRAIN shuffles and drops the partial last batch; VALIDATION
    and TEST read the whole split in deterministic order (shuffle=False, drop_last=False) so the
    held-out metric is a stable corpus measurement, not a randomly-varying ~97% subset.

    ``cfg.eval_stride`` applies to VALIDATION and TEST only. None (the default) leaves the disjoint
    ``stride == max_seq_len`` windows untouched; a smaller value selects the sliding-window
    evaluation, whose overlapping prefixes are masked so every transition is still scored exactly
    once (see :class:`~vfe3.data.datasets.TokenWindows`). Train is never strided -- it shuffles and
    drops its tail, so the exactly-once contract does not apply to it.
    """
    is_train = (split == "train")
    cap = MAX_TOKENS if is_train else None
    # The TRAIN shuffle always draws from an explicit generator, so train() can checkpoint and restore
    # its data cursor: DATA_SEED fixes one order across seeds (EXP-1), None gives each run its own
    # cfg.seed order. Val/test do not shuffle, so a generator is moot there.
    gen = torch.Generator().manual_seed(_effective_data_seed(cfg)) if is_train else None
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=is_train, drop_last=is_train, max_tokens=cap,
                           stride=(None if is_train else cfg.eval_stride),
                           vocab_size=cfg.vocab_size, generator=gen)


def _run_label(cfg: VFE3Config, dataset: str) -> str:
    r"""Descriptive run label ``<dataset>_K<embed_dim>_<group>[_linear][_mix][_mlp][_cross]_s<seed>`` (no
    timestamp, no PPL).

    The stable part of the run-folder name: ``_run_dir`` prefixes it with a timestamp while the run is
    in progress, and ``_rename_run_by_test_ppl`` swaps that prefix for held-out test perplexity at finalize. The
    ``_s<seed>`` suffix keeps a multi-seed launch's run folders distinct (and identifiable by seed).
    """
    _validate_path_component(dataset, "dataset")
    tags = (("_linear" if cfg.decode_mode == "linear" else "")
            + ("_mlp" if cfg.use_block_mlp else ""))
    return f"{dataset}_K{cfg.embed_dim}_{cfg.gauge_group}{tags}_s{cfg.seed}"




def _run_dir(
    cfg:     VFE3Config,
    dataset: str,

    *,
    run_root: 'str | None' = None,
) -> 'str | None':
    r"""Reserve ``<run_root>/<timestamp>_<label>/`` (None when persistence is disabled).

    The timestamp keeps concurrent runs from colliding while training; ``_rename_run_by_test_ppl`` drops it in
    favor of the held-out test perplexity once finalization completes.
    """
    selected_root = RUN_ROOT if run_root is None else run_root
    if selected_root is None:
        return None
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{_run_label(cfg, dataset)}"
    return str(_reserve_directory(Path(selected_root), base))


def _rename_run_by_test_ppl(
    run_dir:  str,                       # in-progress timestamped run directory
    label:    str,                       # descriptive part to keep (see _run_label)
    test_ppl: 'float | None',            # held-out test perplexity (None / non-finite -> no rename)

    logger:   logging.Logger,
) -> str:
    r"""Rename ``run_dir`` to ``vfe3_runs/<test_ppl:.2f>_<label>/`` so runs sort by held-out test perplexity.

    Within a run nothing is selected on test: the test evaluation scores the final training weights
    (no best-val reload), and the name is applied after it. Across runs the name exposes the test
    perplexity, so any recipe chosen by comparing run names was chosen on the test split. The
    folder is created with a timestamp prefix because the test PPL is unavailable before
    finalization; finalization swaps that prefix for the test PPL and drops the timestamp. Returns the
    new path -- or the original unchanged when the PPL is missing/non-finite (the timestamped name is
    then the only stable handle) or when the OS refuses the move (an open handle / locked directory --
    the numeric results are already on disk, so a failed rename is logged, never fatal). A name clash
    gets a ``_2``, ``_3``, ... suffix so an existing run is never clobbered.
    """
    return str(rename_run_by_test_ppl(Path(run_dir), label, test_ppl, logger))


def _run_once(
    seed:   int,
    logger: logging.Logger,

    *,
    run_root: 'str | None' = None,
) -> None:
    r"""One full, independent training run at ``seed`` (build -> train -> val -> test/finalize).

    Builds a fresh ``VFE3Config`` from ``config`` with ``seed`` overridden, seeds the RNG, and runs the
    complete pipeline into its own seed-labeled artifacts dir. Called once per resolved seed by
    :func:`main`, so a multi-seed launch yields one independent, comparable run folder per seed.
    """
    import time

    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts, finalize_run

    overrides = {}
    if fully_gauge_covariant:
        from vfe3.gauge_covariant import (describe_fully_gauge_covariant_overrides, fully_gauge_covariant_overrides,
                                          gauge_covariance_failures)
        overrides = fully_gauge_covariant_overrides(config)
    cfg = VFE3Config(**{**config, **overrides, "seed": seed})   # the run's seed, from SEEDS
    if fully_gauge_covariant:
        failures = gauge_covariance_failures(cfg)
        if failures:
            raise ValueError(f"fully_gauge_covariant=True left these gauge flags false: {failures}")
        logger.info("fully_gauge_covariant=True: overriding %s",
                    describe_fully_gauge_covariant_overrides(config, overrides)
                    or "nothing (the config is already fully gauge covariant)")
    # Always shown, unlike the env-gated ConfigNotice: every setting this launcher names explicitly
    # that no active path reads under the resolved config, with the structural reason.
    inert = {name: reason for name, reason in cfg.inactive_fields().items() if name in config}
    if inert:
        logger.info("Inactive launcher settings under this config (read by no active path):\n%s",
                    "\n".join(f"  - {name}={getattr(cfg, name)!r}: {reason}"
                              for name, reason in inert.items()))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(DEVICE)
    train_loader = _select_loader(DATASET, cfg, split="train")
    val_loader = _select_loader(DATASET, cfg, split="validation")

    # Bits-per-CHARACTER correction so PPL/BPC compare across tokenizers and languages (gpt2 vs
    # cl100k; en/ja/ar -- a cl100k token spans ~3 Japanese codepoints). tokens_per_char =
    # n_tokens/n_codepoints from the held-out stream; None (synthetic / no tiktoken / cache absent)
    # -> BPC unavailable, while evaluate publishes the always-defined bits_per_token separately.
    from vfe3.data.datasets import tokens_per_char as _tokens_per_char
    val_tpc = _tokens_per_char(DATASET, "validation")
    # The end-of-run test split opens here, with the train and validation splits, so a missing test cache fails
    # before training and before a run directory exists (audit 2026-10-16 F122). A sequential loader draws no RNG.
    persist = (RUN_ROOT if run_root is None else run_root) is not None
    test_loader = _select_loader(DATASET, cfg, split="test") if persist else None
    test_tpc = _tokens_per_char(DATASET, "test") if persist else None

    # Run-artifacts directory (config.json, metrics.csv, checkpoints/, best_model.pt, figures).
    # None disables persistence (RUN_ROOT = None); the synthetic fallback also runs unsaved-free.
    run_dir = _run_dir(cfg, DATASET, run_root=run_root)
    artifacts = None
    if run_dir is not None:
        from datetime import datetime
        artifacts = RunArtifacts(run_dir, cfg, model, dataset=DATASET, device=DEVICE,
                                 timestamp=datetime.now().isoformat(timespec="seconds"))
        # config.md: the config dict above, verbatim, plus the values it does not show as literals.
        # Best effort: it re-reads this file from disk, which a later seed of a multi-seed launch may
        # find mid-edit, so a failure is logged and the run continues (audit 2026-10-08).
        try:
            from vfe3.config_markdown import write_config_markdown
            write_config_markdown(run_dir, __file__, {key: getattr(cfg, key) for key in [*config, "seed"]},
                                  header={"DATASET": DATASET, "data seed": _effective_data_seed(cfg),
                                          "fully_gauge_covariant": fully_gauge_covariant,
                                          "fully_gauge_covariant overrides": overrides})
        except Exception as exc:
            logger.warning("config.md was not written (%s: %s); the run continues", type(exc).__name__, exc)
        logger.info("Saving run artifacts to %s", run_dir)

    # Full uncapped corpus size for the "stream is X% of full" banner line -- only computed when
    # MAX_TOKENS actually caps the train stream (the default None loads the whole corpus, so no cap line).
    full_corpus_tokens = None
    if MAX_TOKENS is not None:
        try:
            full_corpus_tokens = cached_token_count(DATASET, "train")
        except FileNotFoundError:
            full_corpus_tokens = None
    logger.info(_banner(model, cfg, DATASET, DEVICE, cfg.max_steps,
                        train_loader=train_loader, full_corpus_tokens=full_corpus_tokens))
    # Reseed AFTER model construction so every later global-RNG draw does NOT depend on the
    # config-dependent amount of global RNG VFEModel(cfg) consumes at init. The train shuffle has
    # its own explicit generator (see _select_loader); the remaining global draws inside train()
    # would otherwise differ from ablation.py for an identical config+seed (model init itself is
    # already identical). Mirrors ablation.run_single's post-build reseed so the two entry points
    # reproduce each other.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    t0 = time.perf_counter()
    final_validation: Dict[str, Optional[float]] = {}
    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=torch.device(DEVICE),
        logger=logger,
        artifacts=artifacts,
        _final_validation_out=final_validation,
        sample_dataset=DATASET,          # audit 2026-07-27: key the sample tokenizer, do not guess
    )
    wall = time.perf_counter() - t0

    if final_validation:
        # Preserve the omitted pass's CPU base-seed draw. The handoff admits only the
        # built-in zero-worker sequential TokenWindows loader, with no RNG during iteration.
        iter(val_loader)
        m = final_validation
    else:
        m = evaluate(model, val_loader, tokens_per_char=val_tpc, device=torch.device(DEVICE))
    logger.info("=" * 64)
    logger.info(                                          # val-only summary; CE is the loss (no separate train loss here)
        "Final (val) | CE: %.4f | PPL: %.1f | BPT: %.4f | BPC: %s",
        m["ce"], m["ppl"], m["bits_per_token"],
        (f"{float(m['bpc']):.4f}" if m["bpc"] is not None else "unavailable"),
    )

    # End-of-run held-out TEST evaluation on the final training weights, plus summary + figures, on the
    # dataset's test split opened at launch (a missing cache raised there -- no synthetic substitution).
    if artifacts is not None:
        results = finalize_run(
            model,
            artifacts,
            cfg,
            final_validation=m,
            tokens_per_char=test_tpc,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            losses=losses,
            data_seed=_effective_data_seed(cfg),
            max_tokens=MAX_TOKENS,
            tokenizer_tag=_tokenizer_tag(DATASET),
            device=torch.device(DEVICE),
            wall_time=wall,
            logger=logger,
        )
        run_dir = _rename_run_by_test_ppl(
            run_dir, _run_label(cfg, DATASET), results.get("test_ppl"), logger)
        logger.info("Artifacts written to %s", run_dir)


def _resolve_seeds(
    run_config: Dict[str, object],
    seeds:      Sequence[int],
    num_runs:   int,
) -> List[int]:
    """Return the first ``num_runs`` entries of ``SEEDS``, the single source of the model seed."""
    if isinstance(num_runs, bool) or not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError(f"NUM_RUNS must be a positive integer, got {num_runs!r}")
    if "seed" in run_config:
        raise ValueError("the config dictionary must not set 'seed'; list the run seeds in SEEDS")
    # A resumed run restores the checkpoint's weights, optimizer, RNG and data cursor, so every
    # seed would continue the SAME run while its folder carries a different seed label. The
    # resolved list always has NUM_RUNS entries.
    if run_config.get("resume_from") is not None and num_runs > 1:
        raise ValueError(
            "resume_from continues one checkpointed run and cannot be combined with more than one "
            f"seed (NUM_RUNS={num_runs}); set NUM_RUNS=1")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("SEEDS must contain exact non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"SEEDS must be unique, got {list(seeds)!r}")
    if len(seeds) < num_runs:
        raise ValueError(
            f"SEEDS lists {len(seeds)} seed(s) but NUM_RUNS={num_runs}; provide at least NUM_RUNS seeds")
    return list(seeds[:num_runs])


def main() -> None:
    r"""Click-to-run entry: train ``NUM_RUNS`` independent seeds back-to-back (default: one run on
    ``SEEDS[0]``). Run i uses ``SEEDS[i]``; each run is fully independent with its own seed-labeled
    artifacts directory. A multi-seed launch owns one atomically reserved group containing its request
    manifest and one comparable run folder per seed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train_vfe3")
    _validated_data_seed()                                  # before reserving any run/manifest directory
    seeds = _resolve_seeds(config, seeds=SEEDS, num_runs=NUM_RUNS)
    run_root = None
    request_manifest = None
    request_path = None
    if RUN_ROOT is not None and len(seeds) > 1:
        from datetime import datetime
        from vfe3.run_artifacts import _write_json_atomic
        group = _reserve_directory(
            Path(RUN_ROOT),
            f"multiseed_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        run_root = str(group)
        request_path = group / "multiseed_request.json"
        request_manifest = {
            "schema_version": 1,
            "status": "pending",
            "seeds": seeds,
            "cells": [{"seed": seed, "status": "pending"} for seed in seeds],
        }
        _write_json_atomic(request_path, request_manifest)
        logger.info("Multi-seed run group: %s", group)
    if request_manifest is not None and request_path is not None:
        request_manifest["status"] = "running"
        _write_json_atomic(request_path, request_manifest)
    for i, s in enumerate(seeds):
        if len(seeds) > 1:
            logger.info("\n%s\n# Run %d/%d  (seed=%d)\n%s", "#" * 64, i + 1, len(seeds), int(s), "#" * 64)
        try:
            _run_once(int(s), logger, run_root=run_root)
        except Exception as exc:
            if request_manifest is not None and request_path is not None:
                request_manifest["status"] = "failed"
                request_manifest["cells"][i].update({
                    "status": "failed",
                    "error": str(exc),
                })
                _write_json_atomic(request_path, request_manifest)
            raise
        if request_manifest is not None and request_path is not None:
            request_manifest["cells"][i]["status"] = "complete"
            request_manifest["status"] = "complete" if i == len(seeds) - 1 else "running"
            _write_json_atomic(request_path, request_manifest)


if __name__ == "__main__":
    main()
