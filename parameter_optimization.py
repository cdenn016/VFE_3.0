r"""Click-to-run joint hyperparameter optimization for the VFE_3.0 transformer.

Edit the ``config`` model/training dictionary and ``SEARCH_CONFIG`` controls below, then click
Run (or run ``python parameter_optimization.py``). No edit to ablation.py is required.
Every candidate changes all enabled parameters together, so the search does not promote one
one-at-a-time winner into the baseline before choosing the next parameter. Candidate generation
is deterministic for a fixed sobol_seed and the existing ablation runner owns training,
validation-only selection, artifacts, and resume validation.
After aggregation, the launcher also writes sweep-level PNG and PDF summaries under
``<output_dir>/figures``, including ranked pairwise interaction strengths and heatmaps.
Optimization cells retain metrics and reports but do not write model weights or checkpoints.

n_trials is the total number of candidate configurations, including the untouched baseline.
If enabled parameters require a different regime, a second search_baseline anchor is included
with only those prerequisites changed. Bounds are continuous and may use linear or logarithmic
sampling; values defines an explicit discrete domain.
"""

import copy
import csv
import json
import logging
import math
import statistics
import time
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

import ablation
from vfe3.config import VFE3Config

logger = logging.getLogger("parameter_optimization")
_JOINT_SWEEP_NAME = "parameter_optimization"
_MAX_DRAW_MULTIPLIER = 64
_SAVE_MODEL_ARTIFACTS = False


# =============================================================================
# CLICK-TO-RUN CONFIGURATION
# =============================================================================
# Edit this dictionary to set the complete model and training operating point.
# It is local to this launcher: parameter_optimization.py never requires a baseline edit in
# ablation.py or train_vfe3.py.

# This is a partial override dictionary. Delete any entry to inherit its VFE3Config default.
config: Dict[str, Any] = dict(

    #################################
    #            Training
    #################################
    # Mean-only Q/K/V conditional filtering (independent opt-in maps).

    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)

    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 128,                 # N, context length
    eval_stride               = 8,

    batch_size                = 64,
    max_steps                 = 15000,

    n_layers                  = 1,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations

    seed                      = 6,
    warmup_steps              = 100,
   

    #################################
    # f-divergence and e/m family
    #################################

    divergence_family         = "renyi",   # "renyi", "squared_hellinger","bhattacharyya", "jeffreys",
    renyi_order               = 1.0,       # Renyi order (1.0 -> KL)

    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full" | "laplace_diagonal" | "gaussian_frame_diagonal"
                                                     # | "gaussian_diagonal_exact" (single covariance toggle; diagonal_covariance is derived)
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
    use_head_mixer                    = True,      # opt-in Schur-commutant head mixer (needs >=2 equal blocks (block_glk/tied_block_glk) OR a labeled irrep tower (so_n/sp_n: per-isotypic-component mixing; mults-one towers get scalar gains));
    use_priorbank_head_evidence_mixer = False,


    decode_mode                       = 'linear',  # prior-bank: diagonal_chunked/full_chunked/expected_likelihood_chunked/family_chunked
    encode_mode                       = "per_token",   #"per_token_additive" is ~1.5x faster at K=80 currently unablated
                                                       # "canonical_content_projected" requires gaussian_diagonal input and a full decoder
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.008,  #prior_bank = True temperature
    decode_ce_checkpoint              = "auto",  # "auto" | "always" | "never"


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
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)




    pos_phi_project_slk       = False,      # per-block trace projection (det Omega = 1)

    gauge_group               = "block_glk",    # "glk" | "block_glk" | "tied_block_glk" | "so_k" | "sp" | "so_n" | "sp_n"

    group_n                   = None,                # so_n/sp_n only; must stay None for every other group
    irrep_spec                = None,                # so_n/sp_n only; must stay None for every other group

    use_cg_coupling           = False,               # so_n/sp_n only: CG cross-type coupling
    cg_covariance_mode        = "passthrough",       # "passthrough" (means-only, pure) | "delta_full"
    cg_covariance_floor       = 1e-6,
    cg_energy_weight          = 0.0,                 # CG moment-energy regularizer (0.0 = OFF)


    ####################################
    # Connection Regime
    ####################################

    transport_mode            = "flat",     # "flat" (the phi-cocycle): the ONLY registered connection regime

    cross_couplings           = None,       # off-block GL(K) head pairs e.g. [(0, 1)]; block_glk only


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
    lambda_alpha_mode          = "state_dependent_per_coord",  # "constant" | "state_dependent" | "state_dependent_per_coord"

    b0   = 1.0,                                # state-dependent alpha shape: alpha* = c0/(b0 + D)
    c0   = 1.0,                                 # state-dependent alpha shape (numerator)

    ########################################
    #     Attention Belief Settings
    #            & Temperatures
    ########################################

    kappa_beta                   = 1,        # tau = kappa * sqrt(d_head)

    beta_attention_prior         = "causal_alibi_noself",



    alibi_slope                  = 1,
    


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
    gradient_mode             = "filtering",  # "filtering" | "smoothing"
    include_attention_entropy = True,
    e_step_gradient           = "unroll",     # "unroll" | "implicit" | "first_order" | "detach"
    oracle_unroll_grad        = True,
    e_step_mu_precond         = "fisher",     # "fisher" | "raw"
    mm_damping                = 1,           # mm_exact damping eta in (0,1]; 1.0 = exact minimizer
   

    e_q_mu_lr                 = 1,
    e_q_sigma_lr              = 0.01,

    e_phi_lr                  = 0.00,


    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group"
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    phi_mstep_max_matrix_norm = 12,
    
    


    m_p_mu_lr                 = 0.03,
    m_p_sigma_lr              = 0.01,
    m_phi_lr                  = 0.01,


    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.005,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.025,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)

    

    m_head_mixer_lr           = 0.075,     #  post-belief Schur HeadMixer parameters
    m_block_mlp_lr            = 0.001,
    
    m_head_evidence_lr        = None,      # active only with use_priorbank_head_evidence_mixer
   


    min_lr                    = 0,
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

    log_interval               = 15001,       # console log every N steps (0 = off)
    eval_interval              = 15001,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 15001,     # save a resumable checkpoint every N steps (0 = off)
    

    #################################
    #        Numerics
    #################################
    
    e_mu_q_trust                         = 2,
    e_sigma_q_trust                      = 1.0,
    sigma_max                            = 100.0,
    mu_trust_mode                        = "ball",


    amp_dtype                            = 'bf16',        # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
    fp32_matmul_precision                = "highest",       # None (leave as-is) | "highest" | "high" | "medium"

    grad_clip                            = 1.0,         # L2 gradient clip applied independently per optimizer role

    exp_fp64_norm_threshold              = 20.0,          # "norm" mode threshold
    transport_chart_max_norm             = 20,


    full_cov_kl_precision                = "fp32_escalate",  # "fp64" | "fp32_escalate"
    full_cov_congruence_precision        = "fp32_escalate",  #  "fp64" | "fp32_escalate"

    safe_cholesky_jitter_mode            = "relative",        #"absolute" | "relative"
    mu_trust_cholesky_rounds             = 3,

    
    decode_av_precision                  = "fp64",     #  "fp32" | "fp64"   
    decode_ranking_fp64_escalation       = True, 
    congruence_cond_escalation           = False, #at K=80 4 heads ~  2x slower
    full_cov_congruence_certification    = True,  

    skip_belief_sigma_update             = True,        # skip the belief-channel sigma E-step update
    
)

# kl_max is the numerical safety-net clamp on EVERY divergence, scaled with K exactly as in
# train_vfe3.py (``config["kl_max"] = 8 * config["embed_dim"]``) so the two entry points share one
# objective under an identical config -- the K-independent 100.0 dataclass default would silently
# differ from train_vfe3's 160 at K=20. A sweep that changes embed_dim must override kl_max itself
# (the K=28 multi-arm below already sets kl_max=224).
config["kl_max"] = 8 * config["embed_dim"]


# =============================================================================
# JOINT-SEARCH CONTROLS AND LOCAL PARAMETER DOMAINS
# =============================================================================
# Presence in this dictionary makes a parameter available; enabled=True includes it in this run.
# Use exactly one of bounds=[low, high] or values=[...]. A disabled entry may leave both as None.
# Prerequisites are fixed for every sampled candidate and validated for cross-parameter conflicts.
SEARCH_CONFIG: Dict[str, Any] = {
    "list_only":  False,
    "device":     "auto",
    "dataset":    "wikitext-103",
    "max_tokens": None,
    "max_steps":  None,

    # Total configurations, including baseline/search_baseline anchors.
    "n_trials":   64,
    "sobol_seed": 642,

    # One seed screens candidates. Use [6, 23, 64] for a joint multi-seed confirmation run.
    "seeds":      [6],
    "resume":     True,
    "output_dir": "vfe3_parameter_optimization_results",

    "parameters": {
        
        
        
        
        
        "e_q_mu_lr": {
            "enabled": True, "bounds": [0.01, 2], "values": None, "scale": "log",
            "requires": {"e_step_update": "gradient"},
            "description": "E-step natural-gradient step size for mu_q.",
        },
        "m_phi_lr": {
            "enabled": True, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "M-step learning rate for gauge-frame parameters.",
        },
        "m_p_mu_lr": {
            "enabled": True, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "M-step learning rate for prior-bank means.",
        },
        "m_p_sigma_lr": {
            "enabled": True, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "M-step learning rate for prior-bank variances.",
        },
       
        
        
        "m_block_mlp_lr": {
            "enabled": True, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {"use_block_mlp": True},
            "description": "M-step learning rate for block-MLP parameters.",
        },
        "m_head_mixer_lr": {
            "enabled": False, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "M-step learning rate for the post-belief head mixer.",
        },
        
        
        
        
        
        
        
        
        "phi_weight_decay": {
            "enabled": False, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "Weight decay on phi.",
        },
        "weight_decay": {
            "enabled": False, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "AdamW weight decay.",
        },
        "mu_weight_decay": {
            "enabled": False, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "AdamW weight decay on mean parameters.",
        },
        "sigma_weight_decay": {
            "enabled": False, "bounds": [0.0005, 0.2], "values": None, "scale": "log",
            "requires": {},
            "description": "AdamW weight decay on log-variance parameters.",
        },
        
        
        
        
        
        
        "phi_scale": {
            "enabled": True, "bounds": [0.0005, 2], "values": None, "scale": "log",
            "requires": {},
            "description": "Initialization standard deviation of phi_embed.",
        },
        "mu_init_std": {
            "enabled": True, "bounds": [0.0005, 2], "values": None, "scale": "log",
            "requires": {},
            "description": "Initialization standard deviation of mu_embed.",
        },
        "pos_phi_scale": {
            "enabled": True, "bounds": [0.0005, 2], "values": None, "scale": "log",
            "requires": {"pos_phi": "learned"},
            "description": "Initialization scale of the learned positional phi table.",
        },
        "sigma_init": {
            "enabled": True, "bounds": [0.1, 10.0], "values": None, "scale": "log",
            "requires": {},
            "description": "Initial coordinate variance of the prior table.",
        },
        
        
        
        
        
        
       
        
        
        "unigram_kappa": {
            "enabled": False, "bounds": None, "values": [0.1, 0.3, 0.4, 0.5, 0.7, 0.85, 1], "scale": "linear",
            "requires": {},
            "description": "Decode unigram-prior temperature.",
        },
        
        
        
        
        
        "lambda_alpha": {
            "enabled": False, "bounds": None, "values": [0.0, 0.5, 1.0, 5.0],
            "scale": "linear",
            "requires": {"lambda_alpha_mode": "constant", "n_e_steps": 1},
            "description": "Constant self-coupling value.",
        },
        "lambda_beta": {
            "enabled": False, "bounds": [0.0, 5.0], "values": None, "scale": "linear",
            "requires": {},
            "description": "Belief-coupling block weight.",
        },
        
        
        
        
        "e_q_sigma_lr": {
            "enabled": False, "bounds": [0.0, 0.1], "values": None, "scale": "linear",
            "requires": {"e_step_update": "gradient"},
            "description": "E-step retraction step size for sigma_q.",
        },
        "decode_tau": {
            "enabled": False, "bounds": None, "values": [0.005, 0.01, 0.025, 0.05],
            "scale": "linear", "requires": {"decode_mode": "diagonal_chunked"},
            "description": "KL-to-prior decode temperature.",
        },
        "kappa_beta": {
            "enabled": False, "bounds": [0.1, 5.0], "values": None, "scale": "linear",
            "requires": {},
            "description": "Belief-attention temperature multiplier.",
        },
        "mm_damping": {
            "enabled": False, "bounds": None, "values": [0.5, 0.75, 0.85, 1.0],
            "scale": "linear", "requires": {"e_step_update": "mm_exact"},
            "description": "MM-exact E-step damping.",
        },
        # SWEEP_ORDER currently contains this field but ablation.SWEEPS has no range for it.
        # Supply bounds or values here before enabling it; no edit to ablation.py is needed.
        "m_head_evidence_lr": {
            "enabled": False, "bounds": None, "values": None, "scale": "log",
            "requires": {"use_priorbank_head_evidence_mixer": True},
            "description": "M-step learning rate for prior-bank head-evidence logits.",
        },
        
        
        
        
        
        "block_mlp_expansion": {
            "enabled": False, "bounds": None, "values": [4, 6, 8, 10, 14], "scale": "linear",
            "requires": {"use_block_mlp": True},
            "description": "Block-MLP hidden-width multiplier.",
        },
        "block_mlp_dropout": {
            "enabled": False, "bounds": None, "values": [0.0, 0.001, 0.01, 0.1],
            "scale": "linear", "requires": {"use_block_mlp": True},
            "description": "Block-MLP residual-branch dropout probability.",
        },
        
        
        
        
        
        
        "sigma_max": {
            "enabled": False, "bounds": None, "values": [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            "scale": "linear", "requires": {},
            "description": "SPD variance ceiling.",
        },
        
        
        
        
        
        
        
        "mstep_self_coupling_weight": {
            "enabled": False, "bounds": [0.1, 1.0], "values": None, "scale": "linear",
            "requires": {},
            "description": "M-step self-coupling weight.",
        },
        "mass_phi": {
            "enabled": False, "bounds": [0.0, 0.4], "values": None, "scale": "linear",
            "requires": {},
            "description": "Gauge-frame mass coefficient.",
        },
    },
}


def _config_field_names() -> set[str]:
    """Return the live VFE3Config field surface."""
    return {field.name for field in dataclass_fields(VFE3Config)}


def _resolved_model_config(model_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the full effective config after applying ``VFE3Config`` defaults."""
    if not isinstance(model_config, Mapping):
        raise ValueError("config must be a mapping")
    config_fields = _config_field_names()
    unknown = sorted(set(model_config) - config_fields)
    if unknown:
        raise ValueError(f"config contains unknown VFE3Config fields: {unknown}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = VFE3Config(**copy.deepcopy(dict(model_config)))
    return asdict(cfg)


def _validate_model_config(model_config: Mapping[str, Any]) -> None:
    """Validate a partial local config without reading defaults from ablation.py."""
    _resolved_model_config(model_config)


@contextmanager
def _ablation_baseline_scope(
    model_config: Mapping[str, Any],
) -> Iterator[None]:
    """Temporarily make the local model config the ablation runner's baseline."""
    resolved = _resolved_model_config(model_config)
    original = ablation.BASELINE_CONFIG
    ablation.BASELINE_CONFIG = resolved
    try:
        yield
    finally:
        ablation.BASELINE_CONFIG = original


def _is_finite_number(value: object) -> bool:
    """Whether value is a finite non-boolean integer or float."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_parameter_domain(name: str, spec: Mapping[str, Any]) -> None:
    """Validate one local continuous or discrete search domain."""
    enabled = spec.get("enabled")
    if type(enabled) is not bool:
        raise ValueError(f"parameter {name!r} enabled must be a bool")
    bounds = spec.get("bounds")
    values = spec.get("values")
    if bounds is not None and values is not None:
        raise ValueError(f"parameter {name!r} must define bounds or values, not both")
    if enabled and bounds is None and values is None:
        raise ValueError(
            f"parameter {name!r} is enabled but has no local bounds or values in SEARCH_CONFIG"
        )
    scale = spec.get("scale")
    if scale not in {"linear", "log"}:
        raise ValueError(f"parameter {name!r} scale must be 'linear' or 'log'")
    if bounds is not None:
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or not all(_is_finite_number(value) for value in bounds)
        ):
            raise ValueError(f"parameter {name!r} bounds must be two finite numbers")
        low, high = float(bounds[0]), float(bounds[1])
        if low >= high:
            raise ValueError(f"parameter {name!r} bounds must satisfy low < high")
        if scale == "log" and low <= 0.0:
            raise ValueError(f"parameter {name!r} log bounds must be strictly positive")
    if values is not None:
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise ValueError(f"parameter {name!r} values must be a list or tuple")
        serialized = [json.dumps(value, sort_keys=True, default=str) for value in values]
        if len(values) < 2:
            raise ValueError(f"parameter {name!r} values must contain at least two choices")
        if len(serialized) != len(set(serialized)):
            raise ValueError(f"parameter {name!r} values must not contain duplicates")


def _validate_parameter_catalog(parameters: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate the self-contained local parameter catalog."""
    if not isinstance(parameters, Mapping):
        raise ValueError("SEARCH_CONFIG['parameters'] must be a mapping")
    config_fields = _config_field_names()
    for name, spec in parameters.items():
        if name not in config_fields:
            raise ValueError(f"search parameter {name!r} is not a VFE3Config field")
        if not isinstance(spec, Mapping):
            raise ValueError(f"parameter {name!r} specification must be a mapping")
        _validate_parameter_domain(name, spec)
        requires = spec.get("requires")
        if not isinstance(requires, Mapping):
            raise ValueError(f"parameter {name!r} requires must be a mapping")
        unknown_requires = [field for field in requires if field not in config_fields]
        if unknown_requires:
            raise ValueError(
                f"parameter {name!r} has unknown prerequisite fields: {unknown_requires}"
            )


def _validated_seeds(value: object) -> List[int]:
    """Return a non-empty ordered list of unique exact integer seeds."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("SEARCH_CONFIG['seeds'] must be a non-empty list or tuple")
    seeds = list(value)
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("SEARCH_CONFIG['seeds'] must contain non-negative exact integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("SEARCH_CONFIG['seeds'] must not contain duplicates")
    return seeds


def _validated_n_trials(value: object) -> int:
    """Return an exact positive total candidate count."""
    if type(value) is not int or value <= 0:
        raise ValueError("SEARCH_CONFIG['n_trials'] must be an exact positive integer")
    return value


def _enabled_parameter_specs(
    parameters: Mapping[str, Mapping[str, Any]],
) -> List[tuple[str, Mapping[str, Any]]]:
    """Return enabled entries in their local dictionary order."""
    return [
        (name, spec)
        for name, spec in parameters.items()
        if spec["enabled"]
    ]


def _fixed_requirements(
    enabled_specs: Sequence[tuple[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Merge fixed prerequisites and reject incompatible joint spaces."""
    sampled_names = {name for name, _spec in enabled_specs}
    requirements: Dict[str, Any] = {}
    owners: Dict[str, str] = {}
    for name, spec in enabled_specs:
        for field, value in spec["requires"].items():
            if field in sampled_names:
                raise ValueError(
                    f"parameter {name!r} requires sampled parameter {field!r}={value!r}; "
                    "disable one of them or remove the fixed prerequisite"
                )
            if field in requirements and requirements[field] != value:
                raise ValueError(
                    f"conflicting prerequisite for {field!r}: "
                    f"{owners[field]!r} requires {requirements[field]!r}, "
                    f"{name!r} requires {value!r}"
                )
            requirements[field] = value
            owners[field] = name
    return requirements


def _parameter_from_unit(unit: float, spec: Mapping[str, Any]) -> Any:
    r"""Map one Sobol coordinate to a local domain.

    Linear bounds use x = low + u(high-low); logarithmic bounds use
    x = exp(log(low) + u(log(high)-log(low))).
    """
    if not 0.0 <= unit <= 1.0:
        raise ValueError(f"Sobol coordinate must be in [0, 1], got {unit!r}")
    values = spec.get("values")
    if values is not None:
        index = min(int(unit * len(values)), len(values) - 1)
        return values[index]
    low, high = (float(value) for value in spec["bounds"])
    if spec["scale"] == "log":
        return math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
    return low + unit * (high - low)


def _candidate_config(
    overrides: Mapping[str, Any],
    model_config: Mapping[str, Any],

    *,
    seed:      int,
    max_steps: Optional[int],
) -> Dict[str, Any]:
    """Return a validated full VFE3Config mapping for one candidate."""
    candidate = _resolved_model_config(model_config)
    candidate.update(overrides)
    candidate["checkpoint_interval"] = 0
    candidate["seed"] = seed
    if max_steps is not None:
        candidate["max_steps"] = max_steps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resolved = VFE3Config(**candidate)
    return asdict(resolved)


def _candidate_fingerprint(
    overrides: Mapping[str, Any],
    model_config: Mapping[str, Any],

    *,
    seed:      int,
    max_steps: Optional[int],
) -> str:
    """Return the semantic full-config identity used to remove duplicate candidates."""
    return ablation.semantic_config_fingerprint(
        _candidate_config(
            overrides,
            model_config,
            seed=seed,
            max_steps=max_steps,
        )
    )


def generate_candidates(
    search_config: Mapping[str, Any],
    model_config:  Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate a deterministic, validated, semantically unique joint Sobol design."""
    local_model_config = _resolved_model_config(
        config if model_config is None else model_config
    )
    parameters = search_config.get("parameters")
    _validate_parameter_catalog(parameters)
    enabled_specs = _enabled_parameter_specs(parameters)
    if not enabled_specs:
        raise ValueError("enable at least one SEARCH_CONFIG['parameters'] entry")
    n_trials = _validated_n_trials(search_config.get("n_trials"))
    sobol_seed = search_config.get("sobol_seed")
    if type(sobol_seed) is not int or sobol_seed < 0:
        raise ValueError("SEARCH_CONFIG['sobol_seed'] must be a non-negative exact integer")
    seeds = _validated_seeds(search_config.get("seeds"))
    max_steps = search_config.get("max_steps")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("SEARCH_CONFIG['max_steps'] must be None or an exact positive integer")

    requirements = _fixed_requirements(enabled_specs)
    changed_requirements = {
        field: value
        for field, value in requirements.items()
        if local_model_config.get(field) != value
    }
    anchors: List[Dict[str, Any]] = [{"label": "baseline", "overrides": {}}]
    if changed_requirements:
        anchors.append({
            "label": "search_baseline",
            "overrides": copy.deepcopy(changed_requirements),
        })
    if n_trials <= len(anchors):
        raise ValueError(
            f"SEARCH_CONFIG['n_trials']={n_trials} leaves no sampled candidate after "
            f"{len(anchors)} required anchor(s)"
        )

    seen: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    for anchor in anchors:
        fingerprint = _candidate_fingerprint(
            anchor["overrides"],
            local_model_config,
            seed=seeds[0],
            max_steps=max_steps,
        )
        if fingerprint not in seen:
            seen.add(fingerprint)
            candidates.append(anchor)

    engine = torch.quasirandom.SobolEngine(
        dimension=len(enabled_specs),
        scramble=True,
        seed=sobol_seed,
    )
    max_draws = max(n_trials * _MAX_DRAW_MULTIPLIER, 256)
    invalid_errors: Dict[str, int] = {}
    draws = 0
    while len(candidates) < n_trials and draws < max_draws:
        point = engine.draw(1, dtype=torch.float64)[0]  # (D,) one joint configuration
        draws += 1
        overrides = copy.deepcopy(changed_requirements)
        for coordinate, (name, spec) in zip(point.tolist(), enabled_specs):
            overrides[name] = _parameter_from_unit(float(coordinate), spec)
        try:
            fingerprint = _candidate_fingerprint(
                overrides,
                local_model_config,
                seed=seeds[0],
                max_steps=max_steps,
            )
        except (NotImplementedError, TypeError, ValueError) as exc:
            message = str(exc)
            invalid_errors[message] = invalid_errors.get(message, 0) + 1
            continue
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append({
            "label": f"joint_{len(candidates):04d}",
            "overrides": overrides,
        })

    if len(candidates) != n_trials:
        invalid_summary = "; ".join(
            f"{count}x {message}" for message, count in invalid_errors.items()
        )
        raise ValueError(
            f"generated only {len(candidates)} unique valid configurations after "
            f"{draws} Sobol draws; reduce n_trials or enlarge the domains"
            + (f"; rejected: {invalid_summary}" if invalid_summary else "")
        )
    return candidates


def _joint_sweep(
    candidates: Sequence[Mapping[str, Any]],
    seeds:      Sequence[int],
) -> Dict[str, Any]:
    """Return the ephemeral multi-arm sweep consumed by ablation.run_sweep."""
    sweep: Dict[str, Any] = {
        "description": (
            "Joint Sobol hyperparameter search generated by parameter_optimization.py; "
            "selection uses validation perplexity only."
        ),
        "configs": [
            {"label": candidate["label"], **copy.deepcopy(candidate["overrides"])}
            for candidate in candidates
        ],
    }
    if len(seeds) > 1:
        sweep["seeds"] = list(seeds)
    return sweep


def _result_base_label(
    result:     Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    """Resolve one single- or multi-seed result label to its candidate label."""
    label = result.get("label")
    if not isinstance(label, str):
        return None
    labels = {str(candidate["label"]) for candidate in candidates}
    if label in labels:
        return label
    seed = result.get("seed")
    if type(seed) is int:
        suffix = f"__s{seed}"
        if label.endswith(suffix) and label[:-len(suffix)] in labels:
            return label[:-len(suffix)]
    return None


def _leaderboard_rows(
    candidates: Sequence[Mapping[str, Any]],
    results:    Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate validation perplexity by candidate and rank finite means first."""
    grouped: Dict[str, List[float]] = {
        str(candidate["label"]): [] for candidate in candidates
    }
    for result in results:
        base_label = _result_base_label(result, candidates)
        value = result.get("primary_val_ppl")
        if (
            base_label is not None
            and _is_finite_number(value)
            and math.isfinite(float(value))
        ):
            grouped[base_label].append(float(value))

    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        label = str(candidate["label"])
        values = grouped[label]
        mean = statistics.fmean(values) if values else math.inf
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append({
            "rank":                     0,
            "label":                    label,
            "mean_primary_val_ppl":     mean,
            "std_primary_val_ppl":      std,
            "best_primary_val_ppl":     min(values) if values else math.inf,
            "worst_primary_val_ppl":    max(values) if values else math.inf,
            "successful_seeds":         len(values),
            "overrides_json":           json.dumps(
                candidate["overrides"], sort_keys=True, separators=(",", ":")
            ),
        })
    rows.sort(key=lambda row: (row["mean_primary_val_ppl"], row["label"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _write_leaderboard(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write the compact candidate-level validation leaderboard."""
    fieldnames = [
        "rank",
        "label",
        "mean_primary_val_ppl",
        "std_primary_val_ppl",
        "best_primary_val_ppl",
        "worst_primary_val_ppl",
        "successful_seeds",
        "overrides_json",
    ]
    with ablation._unique_sibling_temp(path) as temporary:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        ablation._atomic_replace(path, temporary)


def _unit_parameter_value(
    value: object,
    spec:  Mapping[str, Any],
) -> Optional[float]:
    r"""Map one sampled value to its search coordinate $u \in [0, 1]$."""
    values = spec.get("values")
    if values is not None:
        domain = list(values)
        try:
            index = domain.index(value)
        except ValueError:
            return None
        return 0.5 if len(domain) == 1 else index / (len(domain) - 1)

    bounds = spec.get("bounds")
    if bounds is None or not _is_finite_number(value):
        return None
    lo, hi = (float(bounds[0]), float(bounds[1]))
    numeric = float(value)
    if spec.get("scale") == "log":
        if numeric <= 0.0:
            return None
        return (math.log(numeric) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (numeric - lo) / (hi - lo)


def _interaction_rows(
    rows:          Sequence[Mapping[str, Any]],
    candidates:    Sequence[Mapping[str, Any]],
    search_config: Mapping[str, Any],
    n_bins:        int                           = 4,
) -> List[Dict[str, Any]]:
    r"""Rank pairwise departures from an additive median response in PPL units.

    Each enabled parameter is mapped to its normalized search coordinate and binned.
    For populated cell $(i, j)$, the interaction residual is

    $$r_{ij} = \tilde y_{ij} - (\tilde y_{i\cdot} + \tilde y_{\cdot j} - \tilde y),$$

    where each tilde denotes a median. The reported strength is the cell-count-weighted
    root-mean-square residual. This is a descriptive projection of the sampled search,
    not a causal effect estimate.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    parameters = search_config["parameters"]
    enabled_names = [name for name, spec in parameters.items() if spec["enabled"]]
    row_by_label = {
        str(row["label"]): row
        for row in rows
        if _is_finite_number(row.get("mean_primary_val_ppl"))
        and math.isfinite(float(row["mean_primary_val_ppl"]))
    }
    sampled: List[tuple[Dict[str, int], float]] = []
    for candidate in candidates:
        label = str(candidate["label"])
        overrides = candidate.get("overrides", {})
        if label not in row_by_label or not all(name in overrides for name in enabled_names):
            continue
        bins: Dict[str, int] = {}
        for name in enabled_names:
            unit = _unit_parameter_value(overrides[name], parameters[name])
            if unit is None or not math.isfinite(unit):
                break
            bounded = min(1.0, max(0.0, unit))
            bins[name] = min(n_bins - 1, int(math.floor(n_bins * bounded)))
        if len(bins) == len(enabled_names):
            sampled.append((bins, float(row_by_label[label]["mean_primary_val_ppl"])))

    interactions: List[Dict[str, Any]] = []
    for x_index, parameter_x in enumerate(enabled_names):
        for parameter_y in enabled_names[x_index + 1:]:
            cell_values: Dict[tuple[int, int], List[float]] = {}
            x_values: Dict[int, List[float]] = {}
            y_values: Dict[int, List[float]] = {}
            all_values: List[float] = []
            for bins, value in sampled:
                x_bin = bins[parameter_x]
                y_bin = bins[parameter_y]
                cell_values.setdefault((x_bin, y_bin), []).append(value)
                x_values.setdefault(x_bin, []).append(value)
                y_values.setdefault(y_bin, []).append(value)
                all_values.append(value)
            if not all_values:
                continue

            overall_median = float(statistics.median(all_values))
            cell_medians: List[List[Optional[float]]] = [
                [None] * n_bins for _ in range(n_bins)
            ]
            cell_counts = [[0] * n_bins for _ in range(n_bins)]
            cell_residuals: List[List[Optional[float]]] = [
                [None] * n_bins for _ in range(n_bins)
            ]
            weighted_squared_residual = 0.0
            total_count = 0
            maximum_absolute_residual = 0.0
            for (x_bin, y_bin), values in cell_values.items():
                cell_median = float(statistics.median(values))
                additive = (
                    float(statistics.median(x_values[x_bin]))
                    + float(statistics.median(y_values[y_bin]))
                    - overall_median
                )
                residual = cell_median - additive
                count = len(values)
                cell_medians[y_bin][x_bin] = cell_median
                cell_counts[y_bin][x_bin] = count
                cell_residuals[y_bin][x_bin] = residual
                weighted_squared_residual += count * residual * residual
                total_count += count
                maximum_absolute_residual = max(maximum_absolute_residual, abs(residual))

            interactions.append({
                "rank":                         0,
                "parameter_x":                  parameter_x,
                "parameter_y":                  parameter_y,
                "interaction_rms_ppl":          math.sqrt(
                    weighted_squared_residual / total_count
                ),
                "max_abs_interaction_ppl":      maximum_absolute_residual,
                "populated_cells":              len(cell_values),
                "total_cells":                  n_bins * n_bins,
                "n_candidates":                 total_count,
                "_cell_medians":                cell_medians,
                "_cell_counts":                 cell_counts,
                "_cell_residuals":              cell_residuals,
            })

    interactions.sort(key=lambda item: (
        -float(item["interaction_rms_ppl"]),
        str(item["parameter_x"]),
        str(item["parameter_y"]),
    ))
    for rank, interaction in enumerate(interactions, start=1):
        interaction["rank"] = rank
    return interactions


def _parameter_axis_value(
    value: object,
    spec:  Mapping[str, Any],
) -> tuple[Optional[float], Optional[List[str]]]:
    """Return a numeric plotting coordinate and optional categorical tick labels."""
    if _is_finite_number(value):
        return float(value), None
    values = spec.get("values")
    if values is None:
        return None, None
    domain = list(values)
    try:
        index = domain.index(value)
    except ValueError:
        return None, None
    return float(index), [str(item) for item in domain]


def _save_sweep_figure(
    figure:    Any,
    output_dir: Path,
    stem:       str,
) -> List[str]:
    """Atomically publish one sweep figure as a 300-DPI PNG and vector PDF."""
    written: List[str] = []
    figure_dir = output_dir / "figures"
    for suffix in (".png", ".pdf"):
        final_path = figure_dir / f"{stem}{suffix}"
        with ablation._unique_sibling_temp(final_path) as temporary:
            figure.savefig(
                temporary,
                format=suffix.lstrip("."),
                dpi=300,
                bbox_inches="tight",
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"figure renderer wrote no data for {final_path}")
            ablation._atomic_replace(final_path, temporary)
        written.append(final_path.relative_to(output_dir).as_posix())
    return written


def _write_interaction_table(
    output_dir:   Path,
    interactions: Sequence[Mapping[str, Any]],
) -> str:
    """Atomically write the ranked pairwise interaction summary."""
    fieldnames = [
        "rank",
        "parameter_x",
        "parameter_y",
        "interaction_rms_ppl",
        "max_abs_interaction_ppl",
        "populated_cells",
        "total_cells",
        "n_candidates",
    ]
    final_path = output_dir / "figures" / "interaction_strength.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with ablation._unique_sibling_temp(final_path) as temporary:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(interactions)
        ablation._atomic_replace(final_path, temporary)
    return final_path.relative_to(output_dir).as_posix()


def _write_sweep_figures(
    output_dir:    Path,
    rows:          Sequence[Mapping[str, Any]],
    candidates:    Sequence[Mapping[str, Any]],
    search_config: Mapping[str, Any],
    model_config:  Mapping[str, Any],
) -> tuple[List[str], List[str]]:
    """Write aggregate optimization figures and analysis tables."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from vfe3.viz.figures import set_publication_style

    set_publication_style()
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    gray = "#777777"
    row_by_label = {
        str(row["label"]): row
        for row in rows
        if _is_finite_number(row.get("mean_primary_val_ppl"))
        and math.isfinite(float(row["mean_primary_val_ppl"]))
    }
    if not row_by_label:
        raise ValueError("cannot generate sweep figures without finite validation perplexities")

    parameters = search_config["parameters"]
    enabled_names = [name for name, spec in parameters.items() if spec["enabled"]]
    sampled = [
        (candidate, row_by_label[str(candidate["label"])])
        for candidate in candidates
        if str(candidate["label"]) in row_by_label
        and all(name in candidate.get("overrides", {}) for name in enabled_names)
    ]
    finite_rows = sorted(
        row_by_label.values(),
        key=lambda row: (float(row["mean_primary_val_ppl"]), str(row["label"])),
    )
    baseline_row = row_by_label.get("baseline")
    baseline_ppl = (
        float(baseline_row["mean_primary_val_ppl"])
        if baseline_row is not None
        else None
    )
    written: List[str] = []

    overview, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    rank_axis, order_axis = axes
    ranks = np.arange(1, len(finite_rows) + 1)
    ranked_ppl = np.asarray(
        [float(row["mean_primary_val_ppl"]) for row in finite_rows], dtype=float
    )
    rank_axis.plot(ranks, ranked_ppl, color=gray, linewidth=1.0, alpha=0.75)
    rank_axis.scatter(ranks, ranked_ppl, color=blue, s=18, alpha=0.8, zorder=3)
    rank_axis.scatter([1], [ranked_ppl[0]], color=orange, marker="*", s=105,
                      label=f"Best: {finite_rows[0]['label']}", zorder=4)
    if baseline_ppl is not None:
        rank_axis.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.2,
                          label=f"Baseline: {baseline_ppl:.3f}")
    rank_axis.set_xlabel("Validation rank")
    rank_axis.set_ylabel("Validation perplexity")
    rank_axis.set_title("Ranked candidate performance")
    rank_axis.legend(frameon=False, fontsize=8)

    ordered = [
        row_by_label[str(candidate["label"])]
        for candidate in candidates
        if str(candidate["label"]) in row_by_label
    ]
    order_x = np.arange(len(ordered))
    order_y = np.asarray(
        [float(row["mean_primary_val_ppl"]) for row in ordered], dtype=float
    )
    order_axis.scatter(order_x, order_y, color=blue, s=18, alpha=0.75,
                       label="Candidate")
    order_axis.plot(order_x, np.minimum.accumulate(order_y), color=orange, linewidth=1.8,
                    label="Best so far")
    if baseline_ppl is not None:
        order_axis.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.2)
    order_axis.set_xlabel("Candidate order (Sobol; nonadaptive)")
    order_axis.set_ylabel("Validation perplexity")
    order_axis.set_title("Search coverage")
    order_axis.legend(frameon=False, fontsize=8)
    for panel, axis in zip(("A", "B"), axes):
        axis.text(-0.12, 1.06, panel, transform=axis.transAxes, fontweight="bold",
                  va="top")
    overview.suptitle(f"Joint parameter optimization (n={len(finite_rows)} candidates)")
    overview.tight_layout()
    try:
        written.extend(_save_sweep_figure(overview, output_dir, "optimization_overview"))
    finally:
        plt.close(overview)

    n_parameters = len(enabled_names)
    n_columns = min(4, max(1, n_parameters))
    n_rows = max(1, math.ceil(n_parameters / n_columns))
    responses, response_axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.5 * n_columns, 2.9 * n_rows),
        squeeze=False,
    )
    flat_axes = list(response_axes.flat)
    for index, name in enumerate(enabled_names):
        axis = flat_axes[index]
        spec = parameters[name]
        points: List[tuple[float, float, float, str]] = []
        categorical_labels: Optional[List[str]] = None
        for candidate, row in sampled:
            raw_value = candidate["overrides"][name]
            x_value, labels = _parameter_axis_value(raw_value, spec)
            if x_value is None:
                continue
            categorical_labels = labels or categorical_labels
            points.append((
                x_value,
                float(row["mean_primary_val_ppl"]),
                float(row["std_primary_val_ppl"]),
                str(candidate["label"]),
            ))
        if points:
            x_values = np.asarray([point[0] for point in points], dtype=float)
            y_values = np.asarray([point[1] for point in points], dtype=float)
            y_errors = np.asarray([point[2] for point in points], dtype=float)
            axis.errorbar(x_values, y_values, yerr=y_errors, fmt="none", ecolor=blue,
                          alpha=0.18, linewidth=0.7)
            axis.scatter(x_values, y_values, color=blue, s=16, alpha=0.55)
            if spec.get("values") is not None:
                centers = sorted(set(x_values))
                medians = [float(np.median(y_values[x_values == center])) for center in centers]
            else:
                order = np.argsort(x_values)
                groups = np.array_split(order, min(4, len(order)))
                centers = [float(np.median(x_values[group])) for group in groups if len(group)]
                medians = [float(np.median(y_values[group])) for group in groups if len(group)]
            axis.plot(centers, medians, color="black", marker="o", markersize=3,
                      linewidth=1.25, label="Binned median")
            best_point = min(points, key=lambda point: (point[1], point[3]))
            axis.scatter([best_point[0]], [best_point[1]], color=orange, marker="*",
                         s=85, zorder=4)
        if baseline_ppl is not None:
            axis.axhline(baseline_ppl, color="black", linestyle="--", linewidth=1.0,
                         alpha=0.75)
        baseline_value = model_config.get(name)
        baseline_x, baseline_labels = _parameter_axis_value(baseline_value, spec)
        categorical_labels = baseline_labels or categorical_labels
        if baseline_x is not None:
            axis.axvline(baseline_x, color=green, linestyle=":", linewidth=1.2)
        if spec.get("scale") == "log" and categorical_labels is None:
            axis.set_xscale("log")
        if categorical_labels is not None:
            axis.set_xticks(range(len(categorical_labels)), categorical_labels, rotation=30,
                            ha="right")
        axis.set_xlabel(name)
        if index % n_columns == 0:
            axis.set_ylabel("Validation perplexity")
        axis.set_title(name.replace("_", " "))
        if index < 26:
            axis.text(-0.13, 1.07, chr(ord("A") + index), transform=axis.transAxes,
                      fontweight="bold", va="top")
    for axis in flat_axes[n_parameters:]:
        axis.set_visible(False)
    responses.suptitle(
        "Marginal parameter responses (descriptive joint-search projections)",
        y=1.02,
    )
    responses.legend(
        handles=[
            Line2D([], [], color=blue, marker="o", linestyle="none", markersize=4,
                   label="Candidates"),
            Line2D([], [], color="black", marker="o", linewidth=1.25, markersize=3,
                   label="Binned median"),
            Line2D([], [], color=orange, marker="*", linestyle="none", markersize=8,
                   label="Best candidate"),
            Line2D([], [], color=green, linestyle=":", linewidth=1.2,
                   label="Baseline parameter"),
            Line2D([], [], color="black", linestyle="--", linewidth=1.0,
                   label="Baseline PPL"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    responses.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    try:
        written.extend(_save_sweep_figure(responses, output_dir, "parameter_responses"))
    finally:
        plt.close(responses)

    profiles, profile_axis = plt.subplots(
        figsize=(max(6.6, 0.9 * max(1, n_parameters)), 4.6)
    )
    sampled_by_ppl = sorted(
        sampled,
        key=lambda pair: (
            float(pair[1]["mean_primary_val_ppl"]),
            str(pair[0]["label"]),
        ),
    )
    top_count = min(len(sampled_by_ppl), max(1, math.ceil(0.1 * len(sampled_by_ppl))))
    x_positions = np.arange(n_parameters)
    for candidate, row in sampled_by_ppl[:top_count]:
        units = [
            _unit_parameter_value(candidate["overrides"][name], parameters[name])
            for name in enabled_names
        ]
        if all(unit is not None for unit in units):
            profile_axis.plot(x_positions, units, color=blue, alpha=0.28, linewidth=1.0)
    if sampled_by_ppl:
        best_candidate, best_row = sampled_by_ppl[0]
        best_units = [
            _unit_parameter_value(best_candidate["overrides"][name], parameters[name])
            for name in enabled_names
        ]
        if all(unit is not None for unit in best_units):
            profile_axis.plot(x_positions, best_units, color=orange, linewidth=2.4,
                              marker="o", markersize=4,
                              label=f"Best: {best_candidate['label']} "
                                    f"({float(best_row['mean_primary_val_ppl']):.3f})")
    baseline_units = [
        _unit_parameter_value(model_config.get(name), parameters[name])
        for name in enabled_names
    ]
    if baseline_ppl is not None and all(unit is not None for unit in baseline_units):
        profile_axis.plot(x_positions, baseline_units, color="black", linestyle="--",
                          linewidth=1.8, marker="s", markersize=3,
                          label=f"Baseline ({baseline_ppl:.3f})")
    profile_axis.set_xticks(x_positions, enabled_names, rotation=30, ha="right")
    profile_axis.set_ylim(-0.04, 1.04)
    profile_axis.set_ylabel("Normalized search coordinate")
    profile_axis.set_title(f"Top {top_count} sampled configurations")
    profile_axis.legend(frameon=False, fontsize=8)
    profiles.tight_layout()
    try:
        written.extend(_save_sweep_figure(profiles, output_dir, "top_candidate_profiles"))
    finally:
        plt.close(profiles)

    interactions = _interaction_rows(rows, candidates, search_config)
    tables = [_write_interaction_table(output_dir, interactions)]
    top_interactions = interactions[:6]
    n_interactions = len(top_interactions)
    interaction_columns = min(3, max(1, n_interactions))
    interaction_rows = max(1, math.ceil(n_interactions / interaction_columns))
    interaction_figure, interaction_axes = plt.subplots(
        interaction_rows,
        interaction_columns,
        figsize=(4.15 * interaction_columns, 3.65 * interaction_rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_interaction_axes = list(interaction_axes.flat)
    active_interaction_axes = flat_interaction_axes[:n_interactions]
    if top_interactions:
        residual_limit = max(
            abs(float(residual))
            for interaction in top_interactions
            for row in interaction["_cell_residuals"]
            for residual in row
            if residual is not None
        )
        residual_limit = max(residual_limit, 1.0e-12)
        image = None
        bin_labels = ["low", "mid-low", "mid-high", "high"]
        for index, interaction in enumerate(top_interactions):
            axis = flat_interaction_axes[index]
            residuals = np.asarray([
                [np.nan if value is None else float(value) for value in row]
                for row in interaction["_cell_residuals"]
            ])
            image = axis.imshow(
                residuals,
                cmap="PuOr_r",
                vmin=-residual_limit,
                vmax=residual_limit,
                origin="lower",
                aspect="equal",
            )
            for y_bin, row in enumerate(interaction["_cell_medians"]):
                for x_bin, median in enumerate(row):
                    if median is None:
                        continue
                    count = int(interaction["_cell_counts"][y_bin][x_bin])
                    residual = float(interaction["_cell_residuals"][y_bin][x_bin])
                    color = "white" if abs(residual) > 0.55 * residual_limit else "black"
                    axis.text(
                        x_bin,
                        y_bin,
                        f"{float(median):.2f}\n(n={count})",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=7,
                    )
            axis.set_xticks(range(4), bin_labels, rotation=30, ha="right")
            axis.set_yticks(range(4), bin_labels)
            axis.set_xlabel(str(interaction["parameter_x"]))
            axis.set_ylabel(str(interaction["parameter_y"]))
            axis.set_title(
                f"{interaction['parameter_x']} x {interaction['parameter_y']}\n"
                f"interaction RMS = {float(interaction['interaction_rms_ppl']):.3f} PPL",
                fontsize=9,
            )
            axis.text(
                -0.15,
                1.08,
                chr(ord("A") + index),
                transform=axis.transAxes,
                fontweight="bold",
                va="top",
            )
        for axis in flat_interaction_axes[n_interactions:]:
            axis.set_visible(False)
        if image is not None:
            colorbar = interaction_figure.colorbar(
                image,
                ax=active_interaction_axes,
                shrink=0.84,
                pad=0.02,
            )
            colorbar.set_label("Interaction residual (PPL vs additive projection)")
    else:
        flat_interaction_axes[0].axis("off")
        flat_interaction_axes[0].text(
            0.5,
            0.5,
            "Enable at least two parameters to estimate pairwise interactions.",
            ha="center",
            va="center",
        )
    interaction_figure.suptitle(
        "Strongest pairwise parameter interactions\n"
        "Cells show median validation PPL and sample count; color shows nonadditive residual",
        fontsize=11,
    )
    try:
        written.extend(_save_sweep_figure(
            interaction_figure,
            output_dir,
            "parameter_interactions",
        ))
    finally:
        plt.close(interaction_figure)

    return written, tables


def _try_write_sweep_figures(
    output_dir:    Path,
    rows:          Sequence[Mapping[str, Any]],
    candidates:    Sequence[Mapping[str, Any]],
    search_config: Mapping[str, Any],
    model_config:  Mapping[str, Any],
) -> tuple[str, List[str], List[str], Optional[str]]:
    """Write sweep reports, returning a recorded failure instead of losing run results."""
    try:
        figures, tables = _write_sweep_figures(
            output_dir,
            rows,
            candidates,
            search_config,
            model_config,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("sweep figure generation failed: %s", error)
        return "failed", [], [], error
    return "complete", figures, tables, None


def _print_parameter_catalog(parameters: Mapping[str, Mapping[str, Any]]) -> None:
    """Print every self-contained domain without requiring the user to open ablation.py."""
    print("\nJoint-search parameter catalog:\n")
    print(f"  {'enabled':<8}{'parameter':<30}domain | prerequisites")
    print("-" * 110)
    for name, spec in parameters.items():
        if spec.get("bounds") is not None:
            domain = f"{spec['scale']} bounds {spec['bounds']}"
        elif spec.get("values") is not None:
            domain = f"values {spec['values']}"
        else:
            domain = "UNSET"
        requires = spec.get("requires") or {}
        print(
            f"  {str(spec['enabled']):<8}{name:<30}{domain}"
            f" | {json.dumps(requires, sort_keys=True)}"
        )


def _print_leaderboard(rows: Sequence[Mapping[str, Any]], *, limit: int = 10) -> None:
    """Print the best finite candidate aggregates."""
    print("\nValidation leaderboard:\n")
    print(f"  {'rank':>4}  {'label':<22}{'mean PPL':>12}{'std':>12}{'seeds':>8}")
    print("-" * 62)
    for row in rows[:limit]:
        mean = float(row["mean_primary_val_ppl"])
        std = float(row["std_primary_val_ppl"])
        print(
            f"  {int(row['rank']):>4}  {str(row['label']):<22}"
            f"{mean:>12.6f}{std:>12.6f}{int(row['successful_seeds']):>8}"
        )


def _device_from_config(value: object) -> torch.device:
    """Resolve the click-to-run device setting."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(value, str):
        raise ValueError("SEARCH_CONFIG['device'] must be 'auto' or a torch device string")
    return torch.device(value)


def _manifest(
    search_config: Mapping[str, Any],
    model_config:  Mapping[str, Any],
    candidates:    Sequence[Mapping[str, Any]],
    seeds:         Sequence[int],
) -> Dict[str, Any]:
    """Return the immutable candidate-generation record for this invocation."""
    effective_model_config = _resolved_model_config(model_config)
    enabled = [
        name
        for name, spec in search_config["parameters"].items()
        if spec["enabled"]
    ]
    baseline = _candidate_config(
        {},
        effective_model_config,
        seed=seeds[0],
        max_steps=search_config.get("max_steps"),
    )
    return {
        "schema_version":       3,
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
        "method":               "scrambled_sobol_joint",
        "n_trials":             len(candidates),
        "sobol_seed":           search_config["sobol_seed"],
        "training_seeds":       list(seeds),
        "enabled_parameters":   enabled,
        "parameter_order":      list(search_config["parameters"]),
        "baseline_fingerprint": ablation.semantic_config_fingerprint(baseline),
        "model_config":         ablation._jsonable(
            effective_model_config
        ),
        "search_config":        ablation._jsonable(
            copy.deepcopy(dict(search_config))
        ),
        "candidates":           ablation._jsonable(copy.deepcopy(list(candidates))),
        "save_model_artifacts": _SAVE_MODEL_ARTIFACTS,
    }


def main() -> int:
    """Generate the joint design, run it through ablation.py, and publish a leaderboard."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_config = _resolved_model_config(config)
    parameters = SEARCH_CONFIG.get("parameters")
    _validate_parameter_catalog(parameters)
    if SEARCH_CONFIG.get("list_only") is True:
        _print_parameter_catalog(parameters)
        return 0
    if type(SEARCH_CONFIG.get("resume")) is not bool:
        raise ValueError("SEARCH_CONFIG['resume'] must be a bool")
    dataset = SEARCH_CONFIG.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("SEARCH_CONFIG['dataset'] must be a non-empty string")
    max_tokens = SEARCH_CONFIG.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("SEARCH_CONFIG['max_tokens'] must be None or an exact positive integer")

    seeds = _validated_seeds(SEARCH_CONFIG.get("seeds"))
    candidates = generate_candidates(SEARCH_CONFIG, model_config)
    device = _device_from_config(SEARCH_CONFIG.get("device"))
    output_dir = Path(SEARCH_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation._write_json_atomic(
        output_dir / "optimization_manifest.json",
        _manifest(SEARCH_CONFIG, model_config, candidates, seeds),
    )

    print(
        f"\nVFE_3.0 joint parameter optimization"
        f"\n  device:      {device}"
        f"\n  dataset:     {dataset}"
        f"\n  output:      {output_dir}"
        f"\n  candidates:  {len(candidates)}"
        f"\n  seeds:       {seeds}"
        f"\n  parameters:  "
        + ", ".join(
            name for name, spec in parameters.items() if spec["enabled"]
        )
    )

    if _JOINT_SWEEP_NAME in ablation.SWEEPS:
        raise ValueError(
            f"temporary sweep name {_JOINT_SWEEP_NAME!r} already exists in ablation.SWEEPS"
        )
    sweep = _joint_sweep(candidates, seeds)
    ablation.SWEEPS[_JOINT_SWEEP_NAME] = sweep
    try:
        with _ablation_baseline_scope(model_config):
            ablation.validate_sweeps([_JOINT_SWEEP_NAME])
            results = ablation.run_sweep(
                _JOINT_SWEEP_NAME,
                output_dir,
                dataset=dataset,
                device=device,
                seed=seeds[0],
                resume=SEARCH_CONFIG["resume"],
                save_model_artifacts=_SAVE_MODEL_ARTIFACTS,
                max_tokens=max_tokens,
                max_steps=SEARCH_CONFIG.get("max_steps"),
            )
    finally:
        if ablation.SWEEPS.get(_JOINT_SWEEP_NAME) is sweep:
            del ablation.SWEEPS[_JOINT_SWEEP_NAME]

    rows = _leaderboard_rows(candidates, results)
    _write_leaderboard(output_dir / "leaderboard.csv", rows)
    expected_results = len(candidates) * len(seeds)
    complete = len(results) == expected_results
    figure_status, figures, tables, figure_error = _try_write_sweep_figures(
        output_dir,
        rows,
        candidates,
        SEARCH_CONFIG,
        model_config,
    )
    ablation._write_json_atomic(output_dir / "optimization_status.json", {
        "schema_version":   4,
        "status":           "complete" if complete else "incomplete",
        "expected_results": expected_results,
        "admitted_results": len(results),
        "figure_status":    figure_status,
        "figures":          figures,
        "tables":           tables,
        "figure_error":     figure_error,
        "save_model_artifacts": _SAVE_MODEL_ARTIFACTS,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _print_leaderboard(rows)
    if not complete:
        logger.error(
            "optimization incomplete: admitted %d of %d requested seed-cells",
            len(results),
            expected_results,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
