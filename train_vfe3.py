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

# Multi-seed training: launch NUM_RUNS independent runs back-to-back on click-to-run, run i using
# SEEDS[i]. Each run is fully independent (its own model, RNG, and artifacts dir -- the seed is in the
# run-folder label so they never collide). NUM_RUNS=1 with SEEDS=[] keeps the single-run path on the
# config `seed` above, unchanged. Example: NUM_RUNS=3, SEEDS=[3, 64, 23] trains all three seeds.
NUM_RUNS = 1
SEEDS    = 6,#[6,23,54,66,122]        # 54 is best seed so far; must list at least NUM_RUNS seeds when NUM_RUNS > 1

# DATA_SEED (EXP-1 variance floor): when set to an int, the TRAIN loader's shuffle order is fixed to
# this seed via an explicit generator, INDEPENDENT of the per-run model seed -- so a multi-seed run
# shares ONE batch order across seeds and the measured across-seed SD is init+optimization variance,
# NOT data-order noise. None seeds the train generator from each run's cfg.seed instead (a per-seed
# batch order with a resumable cursor, mirroring ablation._effective_data_seed).
DATA_SEED = 3

# Where each run's artifacts go: vfe3_runs/<timestamp>_<label>/ while training (config.json,
# metrics.csv, checkpoints/, best_model.pt, test_results.json, summary.json, *.png), renamed to
# vfe3_runs/<val_ppl>_<label>/ (timestamp dropped) at finalize. None disables persistence.
RUN_ROOT = "vfe3_runs"

config = dict(
    

   
    #################################
    #            Training
    #################################

   
    vocab_size                = _tokenizer_vocab_size(DATASET),   # follows DATASET's tokenizer
                                                                  # (gpt2 50257 / cl100k 100277 /
                                                                  #  word 267735); override to narrow

    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 128,                 # N, context length
    eval_stride               = 32,

    batch_size                = 32,
    max_steps                 = 30000,

    n_layers                  = 2,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,
    
    gradient_mode             = "filtering",   #"smoothing"
    
    # Attention execution and projected input/output scopes.
    attention_mode                   = "projected_predictive",  # "transported" | "projected_conditional" | "projected_predictive"

    relational_input_scope           = "full",         # each head reads all K coordinates; "head" is legacy behavior

    relational_output_scope          = "full",                  # transformers: "full": requires predictive_content, legacy: 'head'
    
    attention_residual               = True,   # predictive S/T/O: add the incoming mean
    relational_routing               = "gaussian_kl",  # "gaussian_kl" | "noisy_query": exact posterior N(a_i; b_j, V_Q + V_K), predictive only
    predictive_covariance            = "passthrough",  # "passthrough" (Sigma kept at encode) | "moment_matched" (exact predictive-kernel covariance), predictive only
    predictive_covariance_floor      = 1e-4,           # moment_matched only: covariant floor floor * (tr C_i / K) I in reference coordinates
   
       
    
    #################################
    # f-divergence and e/m family
    #################################

    divergence_family         = "renyi",   # "renyi", "squared_hellinger","bhattacharyya", "jeffreys",
    renyi_order               = 1.0,       # Renyi order (1.0 -> KL)

    family                    = "gaussian_full", # "gaussian_diagonal" | "gaussian_full" | "laplace_diagonal" | "gaussian_frame_diagonal"
                                                     # | "gaussian_diagonal_exact" (single covariance toggle; diagonal_covariance is derived)

     
    #################################
    #        Initialization
    #################################
    mu_init_std               = 0.065,     # std of the random mean table mu_embed
    sigma_init                = 15,         # constant initial coordinate variance (sigma_log = log of this)
    phi_scale                 = 0.035,      # std
    pos_phi_scale             = 0.02,      # learned-table init scale AND frozen per-position step

    pos_phi_project_slk       = False,      # per-block trace projection (det Omega = 1)
    
    
    #################################
    #        Encode/Decode          #
    #################################
    decode_bias                       = True,     # only for decode_mode='linear'
    
    use_head_mixer                    = False,   # opt-in Schur-commutant head mixer (needs >=2 equal blocks (block_glk/tied_block_glk) OR a labeled irrep tower (so_n/sp_n: per-isotypic-component mixing; mults-one towers get scalar gains));
    use_priorbank_head_evidence_mixer = False,
                                         
    
    decode_mode                       = 'full_chunked',  # prior-bank: diagonal_chunked/full_chunked/expected_likelihood_chunked/family_chunked
    encode_mode                       = "per_token",   #"per_token_additive" is ~1.5x faster at K=80 currently unablated
                                                       # "canonical_content_projected" requires gaussian_diagonal input and a full decoder
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.008,       #prior_bank = True temperature
    
    unigram_kappa                     = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    
    decode_unigram_prior              = True,       # add kappa*log pi_v (corpus unigram) to decode logits    
    untie_decode_bank                 = True,       # prior-bank modes only: decode reads its OWN cloned (V,K) tables
    
    decode_coordinates                = "local",       # "local" | "reference"
    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,      
    block_mlp_mode            = "coordinate",   # "coordinate" | "gauge_gate" | "canonical_frame" | "gaussian_template" (template routing, passthrough covariance)
    block_mlp_covariance      = "passthrough",  # "passthrough" | "delta_full" (needs gaussian_full; not for gaussian_template)
    block_mlp_covariance_floor = 1e-4,          # delta_full only: Sigma+ = J Sigma J^T + floor Sigma
    block_mlp_expansion       = 8,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.00,
    per_type_hidden_bias      = False,          # per-type (V x n_gen) bias on the MLP hidden layer; coordinate MLP, projected route


    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)

    gauge_group               = "block_glk",    # "glk" | "block_glk" | "tied_block_glk" | "so_k" | "sp" | "so_n" | "sp_n"

    group_n                   = None,                # so_n/sp_n only; must stay None for every other group
    irrep_spec                = None,                # so_n/sp_n only; must stay None for every other group

    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    pos_rotation              = "none",              # "none" | "rope"
   
    rope_base                 = 100.0,               # rotary frequency base
        
    rope_full_gauge           = False,               # rotate the covariance sandwich too (REQUIRES family="gaussian_full")
    rope_on_value             = False,


    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "constant",  # "constant" | "state_dependent" | "state_dependent_per_coord"

    b0   = 1.0,                                # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0   = 1.0,                                 # state-dependent alpha shape (numerator)

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################
    
    kappa_beta                   = 1,        # tau = kappa * sqrt(d_head)

    beta_attention_prior         = "causal_alibi",


    alibi_slope                  = 1,
    attention_window             = 128,

  
    
    ##########################################################
    #              VFE Hyper-parameters
    #  Sum_i alpha_i * KL(q_i||p_i) + 
    #  lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) + ...
    ##########################################################

    lambda_alpha               = 1,          # constant self-coupling value
    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)


    # Further Regularizers
    mass_phi                   = 0.0,       # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight = 0.0,       # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)
   
    
    #################################
    #         Belief E-step
    #         Learning Rates
    #################################
    e_step_update             = "gradient",  # "gradient" (pure current path) | "mm_exact" (closed-form MM
                                            # coordinate minimizer at frozen beta)
        
    
    e_q_mu_lr                 = 0.9,
    e_q_sigma_lr              = 0.01,
    
    e_phi_lr                  = 0.00,

   
    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    transport_chart_max_norm  = 18,
    phi_mstep_max_matrix_norm = 12,
    m_phi_group_trust_radius  = 0.1,         


    m_p_mu_lr                 = 0.005,
    m_p_sigma_lr              = 0.001,
    m_phi_lr                  = 0.0015,
    
    m_pos_phi_lr              = 0.001,      # None inherits m_phi_lr; own M-step rate for the LEARNED
                                           # positional frame table (active only at pos_phi='learned')


    #################################
    #        
    #        Weight Decays
    #        
    #################################
    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.04,      
    mu_weight_decay           = 0.025,     


    m_relational_score_lr     = 0.01,   # None inherits m_p_mu_lr; 0 freezes both query/key score maps
    m_relational_content_lr   = 0.0025,   # None inherits m_p_mu_lr; 0 freezes T/O
    

    
    m_block_mlp_lr            = 0.001,
    m_per_type_hidden_bias_lr = None,      # None inherits the declared m_phi_lr; active only with per_type_hidden_bias
    m_layernorm_lr            = 0.05,      # None inherits m_p_mu_lr; own M-step rate for the
                                           


    m_head_mixer_lr           = 0.025,     #  post-belief Schur HeadMixer parameters
    m_head_evidence_lr        = None,      # active only with use_priorbank_head_evidence_mixer

    
    min_lr_frac               = 0.01,
   
    
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine          = True,
    norm_type_block           = "layernorm",        # "none" | "mahalanobis" |"layernorm"
    norm_type_final           = "none",             # "none" | "mahalanobis" |"layernorm"
    block_coordinates         = "local",            # "local" | "reference" (LN and MLP applied as U Phi(U^-1 .))
    
   
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
    mu_trust_mode                        = "ball",
   
   
    amp_dtype                            = 'bf16',        # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
    fp32_matmul_precision                = "highest",          # None (leave as-is) | "highest" | "high" | "medium" ("high" permits TF32 on the decode/energy GEMMs)
    
    grad_clip                            = 1.0,         # L2 gradient clip applied independently per optimizer role
    
    exp_fp64_norm_threshold              = 20.0,          # "norm" mode threshold
    

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
     
    # DEFAULT guard is sign-only: recompute in float64 iff the working-dtype congruence went
    # negative -- one amin reduction the floor needs regardless. Positivity is necessary but not
    # sufficient (cancellation destroys accuracy before it destroys sign), so this toggle adds the
    # conditioning proxy cond(exp_phi)^4 * eps > 1e-3, which bounds the quadratic form's backward
    # error (Higham 2002, 3.1). It is OPT-IN because it is not free: it runs svdvals on the vertex
    # blocks and then a bool() on the result, i.e. an SVD launch plus a GPU->CPU sync, on the
    # HEALTHY path of every covariance transport.                                         
    congruence_cond_escalation           = False, #at K=80 4 heads ~  2x-3x slower                        
     
     
    # Strict post-congruence certificate (audit remediation M13). When enabled, factor every
    # transported pair covariance and promote the complete pair field to float64 if any matrix is
    # uncertified. At B=64, N=128, K=20 this checks 1,048,576 separate 20x20 matrices and the
    # promoted field is about 3.1 GiB. Keep this opt-in for accuracy investigations; False retains
    # the selected working precision without the additional certificate or fallback recompute.
    full_cov_congruence_certification    = False,  # False: no per-pair Cholesky certificate or
                                                 # whole-field fp64 retry. True: strict M13 audit
                                                 # over ~1.05M 20x20 matrices per call at B64/N128.
                                                 
      
    skip_belief_sigma_update             = False,        # skip the belief-channel sigma E-step update
    compile_pair_kernel                  = False,       # torch.compile the closed-form pair kernel 
    
    transport_clamp_monitor              = False,
    generate_embedding_figures           = True,
)

# kl_max is the numerical safety-net clamp on EVERY divergence (KL(q||p), pairwise energy),
# next to eps -- NOT an operating ceiling. Diagonal KL is a sum over K coords (~0.8 nats/coord trained),
# so the K-INDEPENDENT 100.0 default binds for ~100% of tokens at large K (K* ~ 126). Scale it with K
# so it binds only on genuine NaN/inf/Cholesky blowups; F is provably kl_max-independent below the
# ceiling (safe_kl_clamp is the identity there). See docs/2026-06-21-edits.md.
config["kl_max"] = 8 * config["embed_dim"]








def _banner(model, cfg: VFE3Config, dataset: str, device: str, n_steps: int,
            train_loader=None, full_corpus_tokens: 'int | None' = None) -> str:
    from vfe3.train import _fmt_mixer_lr, coverage_lines, parameter_report
    rep = parameter_report(model, device=device)
    bar = "=" * 64
    cov = (coverage_lines(train_loader, n_steps, dataset, full_corpus_tokens=full_corpus_tokens)
           if train_loader is not None else [])
    live_note = (f" ({rep['live']:,} live, {rep['dead']:,} dead)"
                 if rep["probed"] and rep["dead"] else "")
    dead_line = ([" dead under config (no grad): "
                  + ", ".join(n.replace("prior_bank.", "") for n in rep["dead_names"])]
                 if rep["probed"] and rep["dead_names"] else [])
    head_evidence_lr = (
        _fmt_mixer_lr(cfg.m_head_evidence_lr, cfg.m_p_mu_lr)
        if cfg.use_priorbank_head_evidence_mixer else "inactive"
    )
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {rep['total']:,} params{live_note} | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  "
        f"heads={len(model.group.irrep_dims)}  "  # runtime attention heads = irrep blocks (cross_couplings -> 1)
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        *dead_line,
        f" M-LRs: mu={cfg.m_p_mu_lr}  sigma={cfg.m_p_sigma_lr}  "
        f"phi={cfg.m_phi_lr}",
        f" auxiliary M-LRs: head_evidence={head_evidence_lr}  "
        f"head_mixer={_fmt_mixer_lr(cfg.m_head_mixer_lr, cfg.m_p_mu_lr)}  "
        f"block_mlp={_fmt_mixer_lr(cfg.m_block_mlp_lr, cfg.m_p_mu_lr)}",
        f" VFE: lambda_alpha={cfg.lambda_alpha}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}  mass_phi={cfg.mass_phi}",
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
            + ("_mix" if cfg.use_head_mixer else "")
            + ("_mlp" if cfg.use_block_mlp else "")
            + ("_cross" if cfg.cross_couplings else ""))
    return f"{dataset}_K{cfg.embed_dim}_{cfg.gauge_group}{tags}_s{cfg.seed}"


def _reserve_directory(root: Path, base: str) -> Path:
    """Atomically reserve ``root/base`` and retry deterministic numeric suffixes on collision."""
    root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = root / (base if suffix == 1 else f"{base}_{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


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

    Model selection remains validation-only; this is a naming convenience applied AFTER the held-out
    test evaluation of the reloaded best-val checkpoint, so no test metric enters selection. The
    folder is created with a timestamp prefix because the test PPL is unavailable before
    finalization; finalization swaps that prefix for the test PPL and drops the timestamp. Returns the
    new path -- or the original unchanged when the PPL is missing/non-finite (the timestamped name is
    then the only stable handle) or when the OS refuses the move (an open handle / locked directory --
    the numeric results are already on disk, so a failed rename is logged, never fatal). A name clash
    gets a ``_2``, ``_3``, ... suffix so an existing run is never clobbered.
    """
    import math
    from pathlib import Path

    src = Path(run_dir)
    if test_ppl is None or not math.isfinite(test_ppl) or not src.exists():
        return run_dir
    dst = src.parent / f"{test_ppl:.2f}_{label}"
    i = 2
    while dst.exists():
        dst = src.parent / f"{test_ppl:.2f}_{label}_{i}"
        i += 1
    try:
        src.rename(dst)
    except OSError as exc:                                # open handle / locked dir -> keep run, log it
        logger.warning("could not rename run dir to %s (%s); kept %s", dst.name, exc, src.name)
        return run_dir
    logger.info("Renamed run dir -> %s", dst.name)
    return str(dst)


# Compatibility aliases for callers that imported an older boundary. The PPL argument is now the
# held-out TEST perplexity, not validation.
_rename_run_by_ppl = _rename_run_by_test_ppl
_rename_run_by_validation_ppl = _rename_run_by_test_ppl


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

    cfg = VFE3Config(**{**config, "seed": seed})         # per-run seed override (config `seed` is the default)
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

    # Run-artifacts directory (config.json, metrics.csv, checkpoints/, best_model.pt, figures).
    # None disables persistence (RUN_ROOT = None); the synthetic fallback also runs unsaved-free.
    run_dir = _run_dir(cfg, DATASET, run_root=run_root)
    artifacts = None
    if run_dir is not None:
        from datetime import datetime
        artifacts = RunArtifacts(run_dir, cfg, model, dataset=DATASET, device=DEVICE,
                                 timestamp=datetime.now().isoformat(timespec="seconds"))
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

    # End-of-run held-out TEST evaluation on the final training weights, plus summary +
    # figures, on the dataset's test split (a missing cache raises -- no synthetic substitution).
    if artifacts is not None:
        test_loader = _select_loader(DATASET, cfg, split="test")
        test_tpc = _tokens_per_char(DATASET, "test")
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
    """Resolve click-to-run seeds without silently overriding a one-run config seed."""
    if isinstance(num_runs, bool) or not isinstance(num_runs, int) or num_runs <= 0:
        raise ValueError(f"NUM_RUNS must be a positive integer, got {num_runs!r}")
    config_seed = run_config.get("seed")
    if type(config_seed) is not int or config_seed < 0:
        raise ValueError(f"config seed must be an exact non-negative integer, got {config_seed!r}")
    # A resumed run restores the checkpoint's weights, optimizer, RNG and data cursor, so every
    # seed would continue the SAME run while its folder carries a different seed label. The
    # resolved list always has NUM_RUNS entries.
    if run_config.get("resume_from") is not None and num_runs > 1:
        raise ValueError(
            "resume_from continues one checkpointed run and cannot be combined with more than one "
            f"seed (NUM_RUNS={num_runs}); set NUM_RUNS=1")
    if seeds:
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("SEEDS must contain exact non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"SEEDS must be unique, got {list(seeds)!r}")
        if len(seeds) < num_runs:
            raise ValueError(
                f"SEEDS lists {len(seeds)} seed(s) but NUM_RUNS={num_runs}; "
                "provide at least NUM_RUNS seeds"
            )
        if num_runs == 1 and seeds[0] != config_seed:
            raise ValueError(
                f"SEEDS[0]={seeds[0]} conflicts with config seed={config_seed}; "
                "make the one-run seed values agree"
            )
        return list(seeds[:num_runs])
    if num_runs != 1:
        raise ValueError(f"NUM_RUNS={num_runs} > 1 but SEEDS is empty; list one seed per run in SEEDS")
    return [config_seed]


def main() -> None:
    r"""Click-to-run entry: train ``NUM_RUNS`` independent seeds back-to-back (default: one run on the
    config ``seed``). Run i uses ``SEEDS[i]``; each run is fully independent with its own seed-labeled
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
