r"""Click-to-run hyperparameter ablation/sweep runner for the VFE_3.0 transformer.

Sweeps one (or several) ``VFE3Config`` field(s) around the operating point defined in
``train_vfe3.py``. There is no CLI arg parsing (project policy): edit the ``CONFIG`` dict
at the bottom, pick a sweep, then run ``python ablation.py``.

Three sweep shapes are supported, all declared in the ``SWEEPS`` registry:

  * single-field  -- vary ONE field across an explicit ``values`` list or an arithmetic
    ``range = [start, stop, step]`` (one-at-a-time ablation around the baseline);
  * multi-arm     -- a ``configs`` list of named arms, each a dict of field overrides,
    for categorical comparisons whose arms differ in more than one field (e.g. a
    full-covariance arm that flips ``family`` and the per-coordinate alpha form together).
  * parameter-grid -- a Cartesian ``parameter_grid`` whose valid candidates are counted
    exactly and reduced to one closest configuration per embedding width under the click-run
    target and relative-tolerance settings.

The baseline is the self-contained ``BASELINE_CONFIG`` dict below -- a full ``VFE3Config`` toggle
set kept deliberately separate from ``train_vfe3.py`` so an ablation can pin its own (fast)
operating point. It tracks train_vfe3.py's K=20 / block_glk point; keep the two in sync by hand
when the training operating point moves. ``kl_max`` is derived as ``8 * embed_dim`` below exactly
as in train_vfe3.py, and ``DATA_SEED`` mirrors ``train_vfe3.DATA_SEED``, so an identical config
trains on the identical batch order and reproduces train_vfe3.py run-for-run. Each run gets a self-contained
``RunArtifacts`` directory (``config.json``, ``metrics.csv``, ``best_model.pt``, figures)
nested under its sweep, plus an ``ablation_result.json`` headline used for resume and the
sweep-level leaderboard.

``CONFIG['max_steps']`` sets the cosine horizon; ``CONFIG['stop_after_steps']`` can
cap a screening run earlier without compressing that schedule. For example, 15000
and 5000 train through step 5000 on a 15000-step schedule. With model artifacts
enabled, the terminal checkpoint retains that horizon for explicit continuation.

Model selection here is VALIDATION-ONLY (``best_val_ppl``): the held-out test split is NOT
scored per cell (that would leak the test set into selection and cost a full extra eval per
run). To get the test number for the winning configuration, copy its fields into
``train_vfe3.py`` and run that -- ``train_vfe3.py`` calls ``finalize_run`` for the test eval.

Three guards make this safe for VFE_3.0's strict config surface:

  1. every swept field name is checked against the real ``VFE3Config`` dataclass fields at
     startup, so a typo aborts loudly instead of being silently dropped (which would make
     every run identical and read as "this field does not matter");
  2. the data loader is rebuilt whenever a swept field changes ``dataset`` / ``max_seq_len``
     / ``batch_size`` (a memoised factory keyed on those), so a ``batch_size`` sweep does
     not silently reuse the wrong loader;
  3. a config-construction failure (a cross-field violation caught by
     ``VFE3Config.__post_init__``) is tagged ``error_kind = "config"`` and kept DISTINCT
     from a training crash (``"train"``), so a mis-specified cell is not silently bucketed
     as ``ppl = inf``.
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

import ast
import copy
import csv
import gc
import hashlib
import itertools
import json
import logging
import math
import random
import shutil
import stat
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from vfe3.config import VFE3Config, config_notices_shown
from vfe3.data.datasets import (
    _sha256_file,
    _tokenizer_tag,
    cache_source_identity,
    make_dataloader,
    tokens_per_char,
)
from vfe3.metrics import (
    attention_entropy,
    gauge_equivariance_residual,
    guard_saturation,
    head_mixer_gauge_residual,
    rank_one_residual,
)
from vfe3.model.model import VFEModel
from vfe3.path_utils import filesystem_slug, portable_path_component_key
from vfe3.process_utils import run_process_tree
from vfe3.run_artifacts import (
    RunArtifacts,
    _atomic_replace,
    _pure_path_report,
    _package_code_identity,
    _PROCESS_PACKAGE_CODE_IDENTITY,
    _unique_sibling_temp,
    _verified_process_code_identity,
    _write_json_atomic,
    finalize_validation_run,
    semantic_config_fingerprint,
    strict_code_identity_enabled,
)
from vfe3.runtime import seed_everything
from vfe3.train import coverage_lines, evaluate, train
from vfe3.viz.extract import (
    across_layer_belief_trace,
    attention_entropy_cov_gap,
    converged_state,
    per_unit_eval_nats,
)
logger = logging.getLogger("ablation")
_ABLATION_FIGURE_TIMEOUT_SECONDS = 2 * 60 * 60
_RESERVED_SWEEP_NAMES = {"figures", "__sensitivity__"}
_RESERVED_SWEEP_KEYS = {
    portable_path_component_key(name, field="reserved sweep name")
    for name in _RESERVED_SWEEP_NAMES
}
_DECLARATIVE_ABLATION_NAMES = {
    "BASELINE_CONFIG",
    "CONFIG",
    "DATA_SEED",
    "SWEEPS",
    "SWEEP_ORDER",
    "QKV_FACTORIAL_BASELINE",
    "QKV_FACTORIAL_SOURCE",
    "QKV_FACTORIAL_DATA",
}


def _assignment_root_name(target: ast.expr) -> Optional[str]:
    """Return the root name assigned by one top-level assignment target."""
    while isinstance(target, (ast.Attribute, ast.Subscript)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _is_declarative_ablation_statement(statement: ast.stmt) -> bool:
    """Whether a top-level statement changes only registry/config data bound elsewhere."""
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return any(
            _assignment_root_name(target) in _DECLARATIVE_ABLATION_NAMES
            for target in targets
        )
    return False


def _ablation_runner_source_sha256() -> str:
    r"""Hash executable runner logic while excluding separately contracted declarations."""
    source_path = Path(__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    tree.body = [
        statement
        for statement in tree.body
        if not _is_declarative_ablation_statement(statement)
    ]
    payload = ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_PROCESS_ABLATION_RUNNER_SHA256 = _ablation_runner_source_sha256()


def _verified_ablation_runner_source_sha256() -> str:
    """Return the import-time runner identity or, under strict gating, fail after runner edits."""
    current = _ablation_runner_source_sha256()
    if current != _PROCESS_ABLATION_RUNNER_SHA256:
        if strict_code_identity_enabled():
            raise RuntimeError(
                "ablation runner logic changed after this Python process imported it; restart the "
                "Spyder kernel before starting another sweep"
            )
        return current
    return _PROCESS_ABLATION_RUNNER_SHA256


# Sentinel digests published in place of the real source hashes when code-identity gating is off.
# They are constant, so every contract built by any invocation compares equal in its code_identity
# field no matter what the source did between runs: a finished cell is never re-run (and therefore
# never deleted) because an unrelated file changed, the pre-sweep snapshot cannot abort a sweep, and
# the terminal snapshot can never differ from the invocation snapshot, so successful markers are
# never downgraded. The field keeps its validated two-digest shape, so contract schemas, persisted
# fixtures, and the required-key lists are untouched.
_GATING_DISABLED_PACKAGE_SHA256 = hashlib.sha256(
    b"vfe3 package code identity gating disabled").hexdigest()
_GATING_DISABLED_RUNNER_SHA256 = hashlib.sha256(
    b"ablation runner code identity gating disabled").hexdigest()


def _git_code_identity() -> Dict[str, object]:
    r"""Return the exact imported-package and runner-logic identity for ablation contracts.

    The legacy name remains as a monkeypatch seam. Sweep values, order, baseline, output policy, and
    data seed are excluded here because their effective values are already bound explicitly by the
    aggregation or per-cell contract. This lets an edit that only adds a sweep value tack onto the
    prior compatible cohort without treating the declaration edit itself as model-code drift.

    Under ``STRICT_CODE_IDENTITY`` these are the real source digests, so any source generation change
    breaks cohort compatibility. With gating off (the default) constant sentinels are returned and
    source drift no longer invalidates a cohort; see the sentinel definitions above.
    """
    if not strict_code_identity_enabled():
        return {
            "package_code_sha256":    _GATING_DISABLED_PACKAGE_SHA256,
            "ablation_runner_sha256": _GATING_DISABLED_RUNNER_SHA256,
        }
    return {
        "package_code_sha256":  _verified_process_code_identity(),
        "ablation_runner_sha256": _verified_ablation_runner_source_sha256(),
    }


# DATA_SEED (EXP-1 variance floor), mirroring train_vfe3.DATA_SEED: when set to an int, the TRAIN
# loader's shuffle order is fixed to this seed via an explicit generator, INDEPENDENT of the
# per-cell model seed. None keeps the legacy behavior (shuffle drawn from the global RNG, which
# run_single's post-build reseed pins to cfg.seed). Keep this EQUAL to train_vfe3.DATA_SEED so the
# two entry points train on the same batch order and reproduce each other under an identical config.
DATA_SEED: Optional[int] = 3

# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # The default run is ONE contiguous flow: train each sweep, then (per sweep) write its CSV,
    # print its analysis table, and save its PPL figure, then the next sweep; after all sweeps,
    # the cross-sweep comparison plot + best-per-sweep summary. A later re-run with a different
    # value list TACKS its new cells onto the existing per-sweep figure (union of cell markers).
    # Set list_only=True to instead just print the sweep registry and exit (no training).
    "list_only":   False,

    # One sweep name, or None -> every sweep in SWEEP_ORDER.
    # New attention suite: "new_attention_features" (8 arms x NEW_ATTENTION_SEEDS).
    # Smaller subsets: "relational_features", "cross_block_features".
    "sweep":       None,   #"new_attention_features" 8-arms,  shorter: "relational_features" 6-arms


    # 'auto' picks CUDA when present (the RTX 5090), else CPU.
    "device":      "auto",

    # Dataset for every run in the session (NOT a VFE3Config field; the loader seam).
    #   "wikitext-103" | "wikitext-2" | "wiki-en" | "wiki-ja"
    "dataset":     "wikitext-103",

    # Cap the TRAIN stream for fast sweeps (validation is always read in full). None = full.
    "max_tokens":  None, #11684031,

    # Override every run's cosine horizon (None = use BASELINE_CONFIG['max_steps']).
    "max_steps":   None,

    # Optional absolute stopping step, capped at max_steps; None runs the full horizon.
    # Example: max_steps=15000, stop_after_steps=5000 keeps the 15000-step cosine.
    "stop_after_steps": 1875,    

    "seed":        6,

    # Used only by a sweep with a ``parameter_grid``. The selector retains one closest valid
    # architecture per embed_dim and fails before training unless at least two widths fit.
    "target_n_params":               30_000_000,
    "max_param_relative_deviation": 0.02,

    # Skip cells that already wrote ablation_result.json (idempotent reruns / crash recovery).
    # Reuse no longer depends on the source generation: vfe3.run_artifacts.STRICT_CODE_IDENTITY is
    # default False, so editing the package or a driver (or a checkout that rewrites it) cannot
    # invalidate a finished cell. Set that flag True for a reproducibility-strict cohort instead.
    "resume":      True,

    "output_dir":  "vfe3_ablation_results",   # "vfe3_ablation_results_attention_20260913"
}


# =============================================================================
# BASELINE CONFIG  -- self-contained operating point: EVERY VFE3Config toggle.
# =============================================================================


BASELINE_CONFIG: Dict[str, Any] = dict(
    
    #################################
    #            Training
    #################################
    vocab_size                = 50257,               # gpt2/tiktoken vocab (REQUIRED for wikitext-*/wiki-*  100277)

    embed_dim                 = 20,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 2,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 512,                 # N, context length
    eval_stride               = 128,

    batch_size                = 64,
    max_steps                 = 3750,

    n_layers                  = 2,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations
   

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,
    
    # Attention execution and projected input/output scopes.
    attention_mode                   = "projected_predictive",  # "transported" | "projected_conditional" | "projected_predictive"

    relational_input_scope           = "full",         # each head reads all K coordinates; "head" is legacy behavior

    relational_output_scope          = "full",                  # transformers: "full": requires predictive_content, legacy: 'head'
    attention_residual               = True,   # predictive S/T/O: add the incoming mean
    
    
    
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
    phi_scale                 = 0.01,      # std
    pos_phi_scale             = 0.05,      # learned-table init scale AND frozen per-position step

    pos_phi_project_slk       = False,      # per-block trace projection (det Omega = 1)
    
    
    #################################
    #        Encode/Decode          #
    #################################
    decode_bias                       = True,     # only for decode_mode='linear'
    
    use_head_mixer                    = False,   # opt-in Schur-commutant head mixer (needs >=2 equal blocks (block_glk/tied_block_glk) OR a labeled irrep tower (so_n/sp_n: per-isotypic-component mixing; mults-one towers get scalar gains));
    use_priorbank_head_evidence_mixer = False,
                                         
    
    decode_mode                       = 'linear',  # prior-bank: diagonal_chunked/full_chunked/expected_likelihood_chunked/family_chunked
    encode_mode                       = "per_token",   #"per_token_additive" is ~1.5x faster at K=80 currently unablated
                                                       # "canonical_content_projected" requires gaussian_diagonal input and a full decoder
    decode_chunk_size                 = 8192,
    decode_ce_checkpoint              = "auto",     # historical gate; "auto_aggregate" saves memory with recompute
    decode_ce_stable_reduction        = False,      # True avoids large-score CE cancellation; adds tensor passes
    decode_tau                        = 0.008,       #prior_bank = True temperature
    
    unigram_kappa                     = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)

    
    decode_unigram_prior              = True,       # add kappa*log pi_v (corpus unigram) to decode logits    
    untie_decode_bank                 = False,       # prior-bank modes only: decode reads its OWN cloned (V,K) tables

    decode_coordinates                = "local",       # "local" | "reference"
    
    
    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,      
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
    lambda_alpha_mode          = "state_dependent_per_coord",  # "constant" | "state_dependent" | "state_dependent_per_coord"

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
    e_step_mu_precond         = "fisher",  
    mm_damping                = 1,           # mm_exact damping eta in (0,1]; 1.0 = exact minimizer
    
    
    e_q_mu_lr                 = 0.9,
    e_q_sigma_lr              = 0.01,
    
    e_phi_lr                  = 0.00,

   
    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group" | "fisher_group" (belief-Fisher natural gradient)
    phi_precond_mode          = "pullback_per_block",  # "none" | "clip" | "killing" | "killing_per_block" | "pullback" | "pullback_per_block"

    transport_chart_max_norm  = 18,
    phi_mstep_max_matrix_norm = 12,
    m_phi_group_trust_radius  = 0.1,          # embedded Frobenius bound on the group factor
    m_phi_group_retraction    = "chart",      # "chart" | "group_log" (exact Log(U e^{-eta xi}); pullback_group/fisher_group only)
    m_phi_fisher_rotation_weight = 2.0,       # fisher_group only: completion weight on belief-invisible rotations


    m_p_mu_lr                 = 0.005,
    m_p_sigma_lr              = 0.001,
    m_decode_mu_lr            = None,       # None inherits m_p_mu_lr
    m_decode_sigma_lr         = None,       # None inherits m_p_sigma_lr
    m_phi_lr                  = 0.0015,
    

    weight_decay              = 0.01,
    phi_weight_decay          = 0.05,    
    sigma_weight_decay        = 0.04,      # AdamW decay for log-variance tables (None = inherit weight_decay)
    mu_weight_decay           = 0.025,      # AdamW decay for the MEAN-role tables (None = inherit weight_decay)


    m_relational_score_lr     = 0.01,   # None inherits m_p_mu_lr; 0 freezes both query/key score maps
    m_relational_content_lr   = 0.0025,   # None inherits m_p_mu_lr; 0 freezes T/O
    m_layernorm_lr            = 0.05,      # None inherits m_p_mu_lr; own M-step rate for the
    m_pos_phi_lr              = 0.001,      # None inherits m_phi_lr; own M-step rate for the LEARNED
                                           # positional frame table (active only at pos_phi='learned')

   
    m_block_mlp_lr            = 0.001,
    
    m_head_mixer_lr           = 0.025,     #  post-belief Schur HeadMixer parameters
    m_head_evidence_lr        = None,      # active only with use_priorbank_head_evidence_mixer

    
    m_per_type_hidden_bias_lr = None,      # None inherits the declared m_phi_lr; active only with per_type_hidden_bias
    min_lr_frac               = 0.01,
   
    
    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine          = True,
    norm_type_block           = "layernorm",             # "none" | "mahalanobis" |"layernorm"
    norm_type_final           = "none",             # "none" | "mahalanobis" |"layernorm"
    block_coordinates         = "local",            # "local" | "reference" (LN and MLP applied as U Phi(U^-1 .))
  

    #################################
    #         Misc/Logging
    #################################

    log_interval               = 100000,       # console log every N steps (0 = off)
    eval_interval              = 150000,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 150000,     # save a resumable checkpoint every N steps (0 = off)


    #################################
    #        Numerics
    #################################
    
    sigma_max                            = 100.0,
    e_mu_q_trust                         = 2,
    e_sigma_q_trust                      = 1.0,
    mu_trust_mode                        = "ball",
   
   
    amp_dtype                            = 'bf16',        # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
    fp32_matmul_precision                = "highest",       # None (leave as-is) | "highest" | "high" | "medium"  
    
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
    decode_av_precision                  = "fp64",     #  "fp32" | "fp64"


    # Strict decoder ranking guard. When enabled, analytic
    # fp32 decoders construct conservative score intervals and, if ANY row's winning interval
    # overlaps a competitor, recompute the complete (B,N,V) decode in differentiable float64.
    # That is an accuracy diagnostic, not a production-safe fallback: at B=64, N=128, V=50,257
    # the promoted logits alone occupy about 3.1 GiB before backward intermediates. Keep it
    # opt-in so the default chunked route remains streamed and bounded.
    decode_ranking_fp64_escalation       = True,             
     
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
    full_cov_congruence_certification    = True,  # False: no per-pair Cholesky certificate or
                                                 # whole-field fp64 retry. True: strict M13 audit
                                                 # over ~1.05M 20x20 matrices per call at B64/N128.
                                                 
      
    skip_belief_sigma_update             = True,        # skip the belief-channel sigma E-step update
    compile_pair_kernel                  = False,       # torch.compile the closed-form pair kernel 
    
    transport_clamp_monitor              = True,
    generate_embedding_figures           = False,
)

# kl_max is the numerical safety-net clamp on EVERY divergence, scaled with K exactly as in
# train_vfe3.py (``config["kl_max"] = 8 * config["embed_dim"]``) so the two entry points share one
# objective under an identical config -- the K-independent 100.0 dataclass default would silently
# differ from train_vfe3's 160 at K=20. A sweep that changes embed_dim must override kl_max itself
# (the K=28 multi-arm below already sets kl_max=224).
BASELINE_CONFIG["kl_max"] = 8 * BASELINE_CONFIG["embed_dim"]


# Fields deliberately NOT swept, with the reason:
#   vocab_size             fixed by the dataset
#   encode_mode            only implemented registry entries are selectable
#   divergence_family      only 'renyi' is registered (renyi_order is its live knob)
#   seed                   set per run from CONFIG['seed'] (the runner reseeds each cell)
#   max_steps              cosine horizon, set via CONFIG['max_steps']
#   stop_after_steps       optional earlier stopping step (runner-only, not swept)
#   log/eval/checkpoint_interval   bookkeeping, not model behavior
# Dense decoder modes remain private legacy/oracle registrations and are not swept for fresh runs.


# Which sweeps run (and in what order) when CONFIG["sweep"] is None. This is a CURATED subset of
# the full SWEEPS registry above (every key in SWEEPS is also runnable on its own via
# CONFIG["sweep"]="<name>"); add or remove names to shape a session. Cheap-to-expensive is a good
# ordering for a single GPU. Set CONFIG["list_only"]=True (with sweep=None) to print every sweep.
SWEEP_ORDER: List[str] = [
#
 #  "m_layernorm_lr",
  # "m_pos_phi_lr", 

  # "m_relational_content_lr",
  # "m_relational_lr_tied",
   
  
   
 # "pos_phi_scale",
  "phi_scale",   
    "mu_init_std",
    
    
  
  # "m_p_mu_lr",  
   "m_phi_lr",    
     
   "m_p_sigma_lr",
  
  
   "m_block_mlp_lr",
  "m_relational_score_lr",
   
  # "block_mlp_expansion",
 
    "phi_weight_decay",
    "weight_decay",  
    "mu_weight_decay",
   "sigma_weight_decay",
   
   
    
    #  "sigma_init",

     # "e_q_mu_lr",
  
 # "mstep_self_coupling_weight",
 #  "mass_phi",   
  
   
  # "block_mlp_dropout",
 
   
 #   "m_head_mixer_lr",
   #"unigram_kappa",
   
 
  # "lambda_alpha",
  # "lambda_beta",
   
  # "e_q_sigma_lr",
   
  #  "decode_tau",
  #  "kappa_beta",
  #

   # "mm_damping",

  # "sigma_max",
      
   
]


SWEEPS: Dict[str, Dict[str, Any]] = {

    
    "e_q_mu_lr": {
       "description": "E-step natural-gradient step size for mu_q",
       "param": "e_q_mu_lr",
       "range": [0.8, 1.1, 0.1],
       "scale": "linear",
       "requires": {"e_step_update": "gradient"},   
    },
   
    "e_q_sigma_lr": {
       "description": "E-step retraction step size for sigma_q",
       "param": "e_q_sigma_lr", "range": [0.0000, 0.1, 0.005],
       "requires": {"e_step_update": "gradient"},   
    },
   

    "m_p_mu_lr": {
        "description": "M-step LR for the prior-bank means",
        "param": "m_p_mu_lr", 
        "values": [0.0025, 0.005, 0.0075, 0.009, 0.012],
        "scale": "log",
    },
    
   "m_phi_lr": {
        "description": "M-step LR for the gauge-frame parameters (phi)",
        "param": "m_phi_lr",
        "values": [0.001, 0.0015, 0.0025, 0.005],
        "scale": "linear",
    },

   # Two parameters that were previously FORCED onto another component's rate. Each sweep includes
   # the inherited value as an arm, so that arm reproduces the pre-decoupling baseline exactly and
   # the decoupling itself is the measured contrast -- not a confound with a rate change.
   "m_pos_phi_lr": {
        "description": "M-step LR for the learned POSITIONAL frame table (the 0.0015 arm is the "
                       "inherited m_phi_lr baseline)",
        "param": "m_pos_phi_lr",
        "values": [0.001, 0.0025, 0.005, 0.0075],
        "scale": "linear",
        "requires": {"pos_phi": "learned"},
    },

   "m_layernorm_lr": {
        "description": "M-step LR for the LayerNorm affine gamma/beta (the 0.009 arm is the "
                       "inherited m_p_mu_lr baseline)",
        "param": "m_layernorm_lr",
        "values": [0.05],
        "scale": "linear",
        "requires": {"layernorm_affine": True, "norm_type_block": "layernorm"},
    },
    
   
    "m_p_sigma_lr": {
        "description": "M-step LR for the prior-bank variances",
        "param": "m_p_sigma_lr", 
        "values": [0.0005, 0.001, 0.0025, 0.005],
        "scale": "linear",
    },
    "m_decode_mu_lr": {
        "description": "M-step LR for untied decoder means or linear projection",
        "param": "m_decode_mu_lr",
        "values": [0.0005, 0.00075, 0.001, 0.0025],
        "scale": "linear",
    },
    "m_decode_sigma_lr": {
        "description": "M-step LR for untied decoder log variances",
        "param": "m_decode_sigma_lr",
        "values": [0.0005, 0.00075, 0.001, 0.0025],
        "scale": "linear",
        "requires": {"untie_decode_bank": True},
    },
    
    
    "m_relational_score_lr": {
        "description": "M-step LR for independent relational Q/K score maps (0 freezes)",
        "param": "m_relational_score_lr",
        "values": [0.001, 0.0025, 0.005, 0.0075, 0.01],
        "scale": "linear",
        "requires": {
            "attention_mode": "projected_predictive",
        },
    },

    "m_relational_content_lr": {
        "description": "M-step LR for learned relational T/O content maps (0 freezes)",
        "param": "m_relational_content_lr",
        "values": [0.001, 0.0025, 0.005, 0.0075, 0.01],
        "scale": "linear",
        "requires": {
            "attention_mode": "projected_predictive",
        },
    },

    "m_relational_lr_tied": {
        "description": "Q/K score-map and T/O content-map M-step LRs swept together at equal values",
        "configs": [
            {"label": f"m_relational_lr_tied={lr}",
             "m_relational_score_lr": lr, "m_relational_content_lr": lr}
            for lr in (0.0015, 0.003, 0.005, 0.01, 0.02)
        ],
        "requires": {
            "attention_mode": "projected_predictive",
        },
    },
    
    
     "m_block_mlp_lr": {
         "description": "M-step LR for the block MLP parameters",
         "param": "m_block_mlp_lr",
         "values": [0.0005, 0.001, 0.0025, 0.005],  
         "scale": "log",
         "requires": {"use_block_mlp": True},
     },


    "m_head_mixer_lr": {
        "description": "M-step LR for the head mixer",
        "param": "m_head_mixer_lr", 
        "log_range": [0.001, 0.1, 10],  
        "scale": "log",
    },
    
    
    "weight_decay": {
        "description": "AdamW weight decay",
        "param": "weight_decay", 
        "values": [0.005, 0.01, 0.025, 0.05],  
        "scale": "log",
    },

    "phi_weight_decay":{
        "description": "weight decay on phi",
        "param": "phi_weight_decay",  
        "values": [0.01, 0.025, 0.05, 0.1],  
        "scale": "log",
        },


    "mu_weight_decay": {
        "description": "AdamW weight decay",
        "param": "mu_weight_decay",  
        "log_range": [0.001, 0.1, 5],  
        "scale": "log",
    }, 
    
    
    "sigma_weight_decay": {  
        "description": "AdamW weight decay on the log-variance (sigma) tables",
        "param": "sigma_weight_decay",  
        "log_range": [0.001, 0.1, 5],  
        "scale": "log",
    },
    
    
    # === belief-table init scales (PriorBank) ===============================
    "mu_init_std": {
        "description": "init std of the prior mean table mu_embed ~ N(0, std^2)",
        "param": "mu_init_std", 
        "values": [0.001, 0.01, 0.025, 0.05, 0.1],
        "scale": "log",
    },
    
    "sigma_init": {
        "description": "constant initial coordinate variance of the prior table (>0)",
        "param": "sigma_init", 
        "log_range": [0.01, 50, 20],
    },
    
    "phi_scale": {
        "description": "init std of the gauge-frame table phi_embed ~ N(0, std^2)",
        "param": "phi_scale", 
        "values": [0.001, 0.05, 0.1],
        "scale": "log",
    },
    
    
    "pos_phi_scale": {
        "description": "learned pos_phi table init scale",
        "param": "pos_phi_scale", 
        "values": [0.001, 0.01, 0.025, 0.05, 0.1],
        "scale": "log",
        "requires": {"pos_phi": "learned"},
    },    
    
    
    "block_mlp_expansion": {
        "description": "block MLP hidden-width multiplier",
        "param": "block_mlp_expansion",
        "values": [4, 8, 25, 50],
        "requires": {"use_block_mlp": True},
    },

    "block_mlp_activation": {
        "description": "block MLP pointwise activation",
        "param": "block_mlp_activation",
        "values": ["gelu", "silu", "relu"],
        "requires": {"use_block_mlp": True},
    },

    "block_mlp_dropout": {
        "description": "block MLP residual-branch dropout probability",
        "param": "block_mlp_dropout",
        "log_range": [0.0001, 0.2, 20],
        "scale": "log",
        "requires": {"use_block_mlp": True},
    },

   
     "mass_phi": {  # D1 / EXP-8: the regime knob (NOT phi_weight_decay, which is hard-zeroed under
         # pullback group descent). The pullback advantage is predicted to shrink as mass_phi rises (the frame-
         # norm penalty pulls phi toward 0, where ad_phi -> 0 and the pullback metric -> I).
         "description": "mass_phi ",
         "param": "mass_phi", 
         "log_range": [0.0001, 2, 20],
         "scale": "log",
     },
     
    
     "mstep_self_coupling_weight": {
         "description": "M-step self-coupling term alpha_hat * sum_i KL(q_i*||p_i)",
         "param": "mstep_self_coupling_weight", 
         "log_range": [0.0001, 2, 20],
         "scale": "log",
     },


    "lambda_alpha": {
        "description": "constant self-coupling value (lambda_alpha_mode=constant)",
        # audit 2026-08-05: n_e_steps >= 2 is REQUIRED for this sweep to measure anything. At
        # n_e_steps=1 forward_beliefs hands the E-step its own encode output as the prior, so
        # q0 == p exactly and BOTH self-term derivatives are identically zero -- measured
        # bit-identical logits at lambda_alpha 0.0 vs 7.5, i.e. seven identical runs. Same reason
        # lambda_alpha_mode / b0 / c0 are inert at depth 1.
        "param": "lambda_alpha", "values": [0, 0.5, 1, 5],
        "requires": {"lambda_alpha_mode": "constant", "n_e_steps": 2},
    },
    
    
    "lambda_beta": {
        "description": "belief-coupling block weight (1.0 = pure F)",
        "param": "lambda_beta", "range": [0.0, 5, 0.05],
    },
    
    
    "kappa_beta": {
       "description": "attention temperature tau = kappa * sqrt(d_head)",
       "param": "kappa_beta", "range": [0.25, 3, 0.25],
    },
    
    "decode_tau": {
        "description": "KL-to-prior decode temperature",
        "param": "decode_tau", "values": [0.005, 0.01, 0.025, 0.05],
        "requires": {"decode_mode": "diagonal_chunked"},
    },
    

    "unigram_kappa": {
       "description": "decode unigram prior temperature.",
       "param": "unigram_kappa", "log_range": [0.001, 2, 10],
    },
      
    
    "sigma_max": {  # E1 / EXP-20: does the SPD variance ceiling ever bind? (read guard_sigma_ceil_frac)
        "description": "SPD variance ceiling sigma_max (binding vs slack) [E1/EXP-20]",
        # audit 2026-08-06 F20: a single-value grid is not a comparison. The point of this sweep is
        # whether the ceiling ever BINDS, so it needs points either side of the baseline 100.
        "param": "sigma_max", "values": [1, 5, 10, 20, 50, 100],
    },
    
    
    "mm_damping": {  # MM-exact E-step damping eta in (0,1]; 'requires' pins e_step_update='mm_exact' so the
        # damped coordinate-minimizer step is actually taken every cell (1.0 = full exact minimizer).
        "description": "MM-exact E-step damping eta in (0,1] (requires mm_exact E-step)",
        "param": "mm_damping", "values": [0.5, 0.75, 0.85,  1],
        "requires": {"e_step_update": "mm_exact"},
    },

    
    "warmup_steps": {  # LR warmup length before the cosine decay (0 = no warmup, straight into cosine).
        "description": "LR warmup steps before cosine decay",
        "param": "warmup_steps", "values": [50, 150, 200, 1500],
    },
    
    
   "min_lr_frac": {  
       "description": "min_lr_frac ",
       "param": "min_lr_frac", "values": [0.0, 0.001, 0.01, 0.1],
       
   },
    
   
    # === free-energy coupling ==============================================
    
    

    "block_mlp_mode": {
        "description": "post-block residual structure: coordinate, strict invariant gate, or frame-fixed",
        "configs": [
            {
                "label": "block_mlp_mode=coordinate",
                "use_block_mlp": True, "block_mlp_mode": "coordinate",
            },
            {
                "label": "block_mlp_mode=gauge_gate",
                "use_block_mlp": True, "block_mlp_mode": "gauge_gate",
                "family": "gaussian_full", "decode_mode": "full_chunked",
            },
            {
                "label": "block_mlp_mode=canonical_frame",
                "use_block_mlp": True, "block_mlp_mode": "canonical_frame",
            },
        ],
    },

    "block_mlp_covariance": {
        "description": "post-block covariance contract: pass through or delta-method full pushforward",
        "configs": [
            {
                "label": "block_mlp_covariance=passthrough",
                "use_block_mlp": True, "block_mlp_covariance": "passthrough",
            },
            {
                "label": "block_mlp_covariance=delta_full",
                "use_block_mlp": True, "block_mlp_covariance": "delta_full",
                "family": "gaussian_full", "decode_mode": "full_chunked",
            },
        ],
    },

   
}


# Opt-in attention ablations. Keep BASELINE_CONFIG and SWEEP_ORDER unchanged.
# Select CONFIG["sweep"] = "new_attention_features" or one of the four subsets below.
# Ranks track the declared baseline width; other training/data settings remain live.
NEW_ATTENTION_SEEDS = [6]  # use [6] for a one-seed screen
_ATTENTION_REQUIRES = {
    "family": "gaussian_diagonal", "divergence_family": "renyi", "renyi_order": 1.0,
    "include_attention_entropy": True, "gradient_mode": "filtering",
    "e_step_update": "gradient", "e_step_gradient": "unroll", "oracle_unroll_grad": True,
    "n_layers": 1, "n_e_steps": 1, "skip_belief_sigma_update": True, "e_phi_lr": 0.0,
    "transport_mode": "flat", "gauge_group": "block_glk",
    "gauge_transport": "on", "pos_rotation": "none",
    "encode_mode": "per_token", "decode_mode": "linear",
    "compile_pair_kernel": False,
    "cross_couplings": None, "use_priorbank_head_evidence_mixer": False,
    "attention_mode": "projected_predictive", "relational_input_scope": "full",
    "relational_output_scope": "head",
    "m_relational_score_lr": None, "m_relational_content_lr": None,
    "relational_identity_content": False,
    "use_head_mixer": True,
}
_ATTENTION_RELATIONAL_OFF = {"attention_mode": "transported"}
_ATTENTION_CROSS_CONTROL = {
    **_ATTENTION_RELATIONAL_OFF, "use_head_mixer": False,
    "beta_attention_prior": "causal",
    # The residual belongs to projected predictive attention. The transported control
    # disables it explicitly instead of inheriting the projected baseline's setting.
    "attention_residual": False,
}
_ATTENTION_CROSS_PAIRS = [
    (i, j) for i in range(BASELINE_CONFIG["n_heads"])
    for j in range(BASELINE_CONFIG["n_heads"]) if i != j
]
_ATTENTION_ARMS = {
    "relational_features": [
        {"label": "relational_reference"},
        {"label": "relational_head_input", "relational_input_scope": "head"},
        # Dropping the learned content maps is defined only on the conditional-update readout;
        # predictive_content requires learned content maps, and the residual in turn requires
        # predictive_content, so this arm pins both of its own.
        {"label": "relational_identity_content", "relational_identity_content": True,
         "attention_mode": "projected_conditional", "attention_residual": False},
        {"label": "relational_no_head_mixer", "use_head_mixer": False},
        {"label": "relational_frozen_score", "m_relational_score_lr": 0.0},
        {"label": "relational_frozen_content", "m_relational_content_lr": 0.0},
    ],
    "cross_block_features": [
        {**_ATTENTION_CROSS_CONTROL, "label": f"cross_block_{mode}_tied",
         "cross_couplings": _ATTENTION_CROSS_PAIRS if mode == "dense" else None}
        for mode in ("control", "dense")
    ],
}
for _attention_name, _attention_arms in _ATTENTION_ARMS.items():
    SWEEPS[_attention_name] = {
        "description": "Attention feature controls; see docs/research/2026-09-13-relational-implementation/attention-ablations.md",
        "requires": _ATTENTION_REQUIRES,
        "seeds": NEW_ATTENTION_SEEDS,
        "schedule_seed": 20260913,
        "configs": _attention_arms,
    }
SWEEPS["new_attention_features"] = {
    "description": "All 8 attention arms: relational features and cross-block controls",
    "requires": _ATTENTION_REQUIRES,
    "seeds": NEW_ATTENTION_SEEDS,
    "schedule_seed": 20260913,
    "configs": [arm for arms in _ATTENTION_ARMS.values() for arm in arms],
}


SWEEPS["component_ablation_forest"] = {
    "description": "paired-token component ablation forest",
    "configs": [
        {"label": "baseline"},
        {"label": "head_mixer_off", "use_head_mixer": False},
    ],
    "paired_token_bootstrap": True,
    "forest_baseline_label": "baseline",
}


# =============================================================================
# FIELD VALIDATION  -- guard #1: a typo'd field name aborts loudly.
# =============================================================================
_VFE3_FIELDS = {f.name for f in dataclass_fields(VFE3Config)}


def _swept_field_names(sweep: Dict[str, Any]) -> List[str]:
    r"""Every VFE3Config field a sweep touches through its values, arms, grid, or requirements."""
    names: List[str] = list(sweep.get("requires", {}).keys())
    if "parameter_grid" in sweep:
        grid = sweep.get("parameter_grid")
        if isinstance(grid, Mapping):
            names.extend(str(field) for field in grid)
    elif "configs" in sweep:
        for arm in sweep["configs"]:
            names.extend(k for k in arm if k != "label")
    elif "param" in sweep:
        names.append(sweep["param"])
    return names


def _validated_sweep_name(value: object) -> str:
    """Return one safe, non-reserved output-directory component for a sweep."""
    if not isinstance(value, str):
        raise ValueError(f"sweep name is unsafe or reserved: {value!r}")
    key = portable_path_component_key(value, field="sweep name")
    if key in _RESERVED_SWEEP_KEYS:
        raise ValueError(f"sweep name is unsafe or reserved: {value!r}")
    return value


def _validated_target_n_params(value: object) -> int:
    """Return one exact positive learned-parameter target."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"target_n_params must be an exact positive integer, got {value!r}")
    return value


def _validated_param_relative_deviation(value: object) -> float:
    """Return one finite relative parameter-budget tolerance in [0, 1)."""
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(
            "max_param_relative_deviation must be a finite float in [0, 1), "
            f"got {value!r}"
        )
    return value


def _parameter_grid_overrides(sweep: Mapping[str, object]) -> List[Dict[str, Any]]:
    r"""Expand one declared parameter grid in stable order, deriving width-scaled ``kl_max``."""
    grid = sweep.get("parameter_grid")
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("parameter-matched sweep needs a non-empty parameter_grid mapping")
    match_by = sweep.get("match_by")
    if match_by != "embed_dim":
        raise ValueError(
            "parameter-matched sweep currently requires match_by='embed_dim', "
            f"got {match_by!r}"
        )
    if match_by not in grid:
        raise ValueError("parameter_grid must include its match_by field 'embed_dim'")

    fields: List[str] = []
    value_lists: List[List[Any]] = []
    for field, raw_values in grid.items():
        if not isinstance(field, str) or field not in _VFE3_FIELDS:
            raise ValueError(f"parameter_grid field is not a VFE3Config field: {field!r}")
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ValueError(f"parameter_grid[{field!r}] must be a non-empty list or tuple")
        values = list(raw_values)
        serialized = [json.dumps(value, sort_keys=True, default=str) for value in values]
        if len(serialized) != len(set(serialized)):
            raise ValueError(f"parameter_grid[{field!r}] must not contain duplicate values")
        fields.append(field)
        value_lists.append(values)

    requires = sweep.get("requires", {})
    if not isinstance(requires, Mapping):
        raise ValueError("parameter-matched sweep requires must be a mapping")
    overrides: List[Dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        candidate = dict(requires)
        candidate.update(dict(zip(fields, values)))
        if "embed_dim" in candidate and "kl_max" not in candidate:
            embed_dim = candidate["embed_dim"]
            if type(embed_dim) is not int:
                raise ValueError(
                    f"parameter_grid embed_dim must be an exact integer, got {embed_dim!r}"
                )
            candidate["kl_max"] = 8 * embed_dim
        overrides.append(candidate)
    return overrides


def _realized_n_params_for_overrides(overrides: Mapping[str, object]) -> int:
    """Construct one CPU model and return its exact realized learned-parameter count."""
    cfg = VFE3Config(**_cell_cfg_dict(dict(overrides), seed=0))
    model = VFEModel(cfg)
    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    finally:
        del model
        gc.collect()


def _parameter_candidate_label(overrides: Mapping[str, object], fields: Sequence[str]) -> str:
    """Return one deterministic, human-readable label in declared grid-field order."""
    return "__".join(f"{field}={overrides[field]}" for field in fields)


def _parameter_match_selection(sweep_name: str) -> Dict[str, Any]:
    r"""Select one closest realized-parameter candidate per requested embedding width."""
    sweep_name = _validated_sweep_name(sweep_name)
    sweep = SWEEPS[sweep_name]
    grid = sweep.get("parameter_grid")
    if not isinstance(grid, Mapping):
        raise ValueError(f"sweep {sweep_name!r} is not parameter matched")
    target = _validated_target_n_params(CONFIG.get("target_n_params"))
    tolerance = _validated_param_relative_deviation(
        CONFIG.get("max_param_relative_deviation")
    )
    fields = [str(field) for field in grid]
    match_by = str(sweep.get("match_by"))
    candidates = _parameter_grid_overrides(sweep)

    requested_widths = list(grid[match_by])
    valid_by_width: Dict[object, List[Dict[str, Any]]] = {
        width: [] for width in requested_widths
    }
    rejected: List[Dict[str, Any]] = []
    for candidate_index, overrides in enumerate(candidates):
        label = _parameter_candidate_label(overrides, fields)
        try:
            VFE3Config(**_cell_cfg_dict(overrides, seed=0))
            n_params = _realized_n_params_for_overrides(overrides)
        except (NotImplementedError, TypeError, ValueError) as exc:
            rejected.append({
                "label":      label,
                "overrides":  _jsonable(overrides),
                "reason":     "config",
                "error":      str(exc),
            })
            continue
        difference = n_params - target
        relative_deviation = abs(difference) / target
        record = {
            "label":                    label,
            "overrides":                _jsonable(overrides),
            "n_params":                 n_params,
            "target_n_params":          target,
            "param_difference":         difference,
            "param_relative_deviation": relative_deviation,
            "candidate_index":          candidate_index,
        }
        valid_by_width.setdefault(overrides[match_by], []).append(record)

    selected: List[Dict[str, Any]] = []
    closest_by_width: Dict[object, Dict[str, Any]] = {}
    for width, records in valid_by_width.items():
        if not records:
            continue
        closest = min(
            records,
            key=lambda record: (
                record["param_relative_deviation"],
                record["candidate_index"],
            ),
        )
        closest_by_width[width] = closest
        if closest["param_relative_deviation"] <= tolerance:
            selected.append(closest)
        for record in records:
            if record is closest and record in selected:
                continue
            rejected.append({
                **record,
                "reason": ("outside_tolerance" if record is closest else "not_closest"),
                "error":  None,
            })

    if len(selected) < 2:
        details = "; ".join(
            (
                f"{match_by}={width}: n_params={closest_by_width[width]['n_params']:,}, "
                "relative_deviation="
                f"{closest_by_width[width]['param_relative_deviation']:.6f}"
                if width in closest_by_width
                else f"{match_by}={width}: no valid candidates"
            )
            for width in requested_widths
        )
        raise ValueError(
            f"parameter-matched sweep retained {len(selected)} width(s), but at least 2 are "
            f"required within relative tolerance {tolerance:.6f}; closest rejected candidates: "
            f"{details}"
        )

    # Candidate-grid axes are search knobs, not necessarily training contrasts. Check only
    # the retained arms so inactive candidates do not masquerade as selected comparisons.
    selected_overrides = [record["overrides"] for record in selected]
    varying = {key for key in set().union(*(set(arm) for arm in selected_overrides))
               if len({repr(arm.get(key)) for arm in selected_overrides}) > 1}
    inactive = [set(VFE3Config(**_cell_cfg_dict(arm, seed=0)).inactive_fields())
                for arm in selected_overrides]
    always_dead = sorted(varying & set.intersection(*inactive))
    if always_dead:
        raise ValueError("parameter-matched selected contrast(s) " + ", ".join(always_dead)
                         + " are INERT on EVERY selected arm")

    for record in selected:
        record.pop("candidate_index", None)
    for record in rejected:
        record.pop("candidate_index", None)
    return {
        "target_n_params":                 target,
        "max_param_relative_deviation":   tolerance,
        "match_by":                        match_by,
        "parameter_grid":                  _jsonable(dict(grid)),
        "selected":                        selected,
        "rejected":                        rejected,
    }


def _sweep_output_scope(sweep_name: str) -> str:
    """Return the direct output child for one logical sweep and parameter-budget invocation."""
    sweep_name = _validated_sweep_name(sweep_name)
    sweep = SWEEPS[sweep_name]
    if "parameter_grid" not in sweep:
        return sweep_name
    target = _validated_target_n_params(CONFIG.get("target_n_params"))
    tolerance = _validated_param_relative_deviation(
        CONFIG.get("max_param_relative_deviation")
    )
    tolerance_text = repr(tolerance).replace(".", "p")
    return f"{sweep_name}_N{target}_rtol{tolerance_text}"


def _parameter_budget_record_matches(
    result: Mapping[str, object],
    record: Mapping[str, object],
) -> bool:
    """Whether a persisted marker carries the exact selected parameter-budget identity."""
    try:
        return bool(
            type(result.get("n_params")) is int
            and result["n_params"] == record["n_params"]
            and type(result.get("target_n_params")) is int
            and result["target_n_params"] == record["target_n_params"]
            and type(result.get("param_difference")) is int
            and result["param_difference"] == record["param_difference"]
            and type(result.get("param_relative_deviation")) is float
            and result["param_relative_deviation"] == record["param_relative_deviation"]
        )
    except (KeyError, TypeError):
        return False


def _parameter_records_by_label(value: object) -> Optional[Dict[str, Mapping[str, object]]]:
    """Validate and index the selected rows in optional sweep-level budget metadata."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("parameter_match metadata must be a mapping or null")
    target = _validated_target_n_params(value.get("target_n_params"))
    tolerance = _validated_param_relative_deviation(
        value.get("max_param_relative_deviation")
    )
    selected = value.get("selected")
    if not isinstance(selected, list) or len(selected) < 2:
        raise ValueError("parameter_match metadata must contain at least two selected rows")
    records: Dict[str, Mapping[str, object]] = {}
    for record in selected:
        if not isinstance(record, Mapping):
            raise ValueError("parameter_match selected rows must be mappings")
        label = record.get("label")
        if not isinstance(label, str) or not label or label in records:
            raise ValueError("parameter_match selected labels must be unique non-empty strings")
        if not _parameter_budget_record_matches(record, record):
            raise ValueError(f"parameter_match selected row {label!r} has invalid budget fields")
        n_params = record["n_params"]
        difference = record["param_difference"]
        relative_deviation = record["param_relative_deviation"]
        if (
            n_params <= 0
            or record["target_n_params"] != target
            or difference != n_params - target
            or relative_deviation != abs(difference) / target
            or relative_deviation > tolerance
        ):
            raise ValueError(
                f"parameter_match selected row {label!r} has inconsistent budget fields"
            )
        records[label] = record
    return records


def validate_sweeps(sweep_names: List[str], *, require_construction: bool = True) -> None:
    r"""Abort unless every named arm references real fields and constructs against the baseline.

    VFE3Config(**cfg) would silently ignore an unknown kwarg only if it were dropped first;
    here a bad name would instead raise a TypeError mid-run (or, worse under a dict-merge
    that pre-filtered, vanish and make every cell identical). Catching it once at startup
    turns a subtle "this parameter has no effect" result into an immediate, named error.

    Two classes of check live here, and only the first is baseline-independent:

      * name / field / label validity -- ALWAYS fatal, registry-wide. A typo'd or aliasing
        field name is wrong under every baseline, which is the guarantee described above.
      * arm CONSTRUCTION against the current ``BASELINE_CONFIG`` -- fatal only when
        ``require_construction=True``, i.e. for the sweeps this session will actually run.

    The registry deliberately spans mutually exclusive regimes (diagonal vs full covariance,
    kernel vs autograd-oracle belief gradient, prior-bank vs linear decode, flat vs regime-II
    transport), so NO single baseline can construct every registered arm at once. Requiring
    that of the whole registry made an edit to one baseline field abort the session over
    sweeps that were never scheduled -- e.g. ``family='gaussian_full'`` alone invalidates the
    ``mm_exact`` arms of 'estep_depth_damping'/'mm_damping' (kernel-route only), the diagonal
    arm of 'covariance', and every off-KL 'renyi_order'
    arm. Those sweeps each carry the baseline they need; they are reported as a warning here
    and remain runnable on their own once the baseline is set for them.
    """
    portable_names: Dict[str, str] = {}
    offenders: List[Tuple[str, str]] = []
    for name in sweep_names:
        validated = _validated_sweep_name(name)
        key = portable_path_component_key(validated, field="sweep name")
        prior = portable_names.get(key)
        if prior is not None and prior != validated:
            raise ValueError(f"sweep names {prior!r} and {validated!r} alias on a portable filesystem")
        portable_names[key] = validated
        sweep = SWEEPS[name]
        scale = sweep.get("scale", "linear")
        if scale not in {"linear", "log"}:
            raise ValueError(
                f"sweep {name!r} scale must be 'linear' or 'log', got {scale!r}"
            )
        if not any(key in sweep for key in ("configs", "param", "parameter_grid")):
            raise ValueError(
                f"sweep {name!r} declares none of 'param'/'values', 'configs', or "
                "'parameter_grid'"
            )
        if "param" in sweep:
            values = _sweep_values(sweep)
            if scale == "log" and any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
                for value in values
            ):
                raise ValueError(
                    f"sweep {name!r} with scale='log' requires strictly positive finite numeric values"
                )
        for field in _swept_field_names(sweep):
            if field not in _VFE3_FIELDS:
                offenders.append((name, field))
    if offenders:
        lines = "\n".join(f"  sweep {s!r}: {f!r} is not a VFE3Config field" for s, f in offenders)
        raise ValueError(
            "ablation SWEEPS reference field(s) that do not exist on VFE3Config "
            f"(typo? renamed?):\n{lines}"
        )
    construction_errors: List[Tuple[str, str, str]] = []
    for name in sweep_names:
        if "parameter_grid" in SWEEPS[name]:
            valid_candidates = 0
            for index, overrides in enumerate(_parameter_grid_overrides(SWEEPS[name])):
                try:
                    VFE3Config(**{**_resolved_sweep_baseline(), **overrides})
                    valid_candidates += 1
                except (TypeError, ValueError):
                    pass
            if valid_candidates == 0:
                construction_errors.append((
                    name,
                    "parameter_grid",
                    "no candidate combination constructs a valid VFE3Config",
                ))
            continue
        runs = make_run_overrides(name)
        labels = [label for label, _overrides in runs]
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError(f"sweep {name!r} has a non-string or empty label")
        duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
        if duplicate_labels:
            raise ValueError(f"sweep {name!r} has duplicate expanded label(s): {duplicate_labels}")
        # Keys whose value actually DIFFERS across this sweep's arms -- the sweep's contrast. For a
        # 'param' sweep that is the swept parameter; for a 'configs' sweep it is whatever the arm
        # dicts disagree on. Everything else in an arm dict is a constant REGIME setting shared by
        # every arm (e.g. cg_coupling pins phi_precond_mode='killing' on both arms), and an inert
        # regime setting is not a broken measurement -- only an inert CONTRAST is.
        varying_keys = {
            key for key in {k for _label, ov in runs for k in ov}
            if len({repr(ov.get(key)) for _label, ov in runs}) > 1
        }
        inert_per_arm: List[set] = []
        for label, overrides in runs:
            cfg = _resolved_sweep_baseline()
            cfg.update(overrides)
            try:
                # Activity is structural, including default-valued fields; warnings are display only.
                resolved = VFE3Config(**cfg)
                inert_per_arm.append(set(resolved.inactive_fields()) & varying_keys)
            except (TypeError, ValueError) as exc:
                construction_errors.append((name, label, str(exc)))
        if inert_per_arm and len(inert_per_arm) == len(runs):
            # A contrast may be live in only one arm. Reject it only when every arm is inactive.
            always_dead = sorted(set.intersection(*inert_per_arm))
            if always_dead:
                construction_errors.append((
                    name, "all arms",
                    f"swept parameter(s) {', '.join(always_dead)} are INERT on EVERY arm -- each "
                    f"arm would reproduce the baseline number. Add the enabling setting(s) to the "
                    f"sweep's 'requires'.",
                ))
    if construction_errors:
        if require_construction:
            lines = "\n".join(
                f"  sweep {sweep!r}, arm {label!r}: {error}"
                for sweep, label, error in construction_errors
            )
            raise ValueError(f"ablation arm construction failed:\n{lines}")
        # Advisory pass over the unscheduled remainder: one line per sweep, not per arm, so a
        # baseline that legitimately excludes a whole regime does not bury the run banner.
        by_sweep: Dict[str, List[str]] = {}
        first_error: Dict[str, str] = {}
        for sweep, label, error in construction_errors:
            by_sweep.setdefault(sweep, []).append(label)
            first_error.setdefault(sweep, error)
        summary = "\n".join(
            f"  {sweep}: {len(labels)} arm(s) -- {first_error[sweep]}"
            for sweep, labels in by_sweep.items()
        )
        # Hidden by default, alongside the config inertness notices (VFE3_CONFIG_NOTICES=1 or
        # vfe3.config.set_config_notices(True) to expand). The one-line COUNT still goes out at
        # INFO even when hidden, and deliberately so: this block is not console noise about a
        # value that does nothing, it is the record that N registered sweeps are being DROPPED
        # from the session. Silently scheduling 12 sweeps when the registry lists 20 is the exact
        # failure the dead-sweep guard exists to prevent, so the FACT stays visible even when the
        # per-sweep detail does not.
        if config_notices_shown():
            logger.warning(
                "%d registered sweep(s) do not construct against the current BASELINE_CONFIG and "
                "are NOT scheduled this session; each needs its own baseline regime:\n%s",
                len(by_sweep), summary,
            )
        else:
            logger.info(
                "%d registered sweep(s) not scheduled (incompatible with BASELINE_CONFIG); "
                "set VFE3_CONFIG_NOTICES=1 for the per-sweep reason.",
                len(by_sweep),
            )


# =============================================================================
# RUN-CONFIG EXPANSION
# =============================================================================

def _expand_range(spec: List[Union[int, float]]) -> List[Union[int, float]]:
    r"""Expand a ``[start, stop, step]`` range into an explicit inclusive list."""
    if len(spec) != 3:
        raise ValueError(f"'range' must be [start, stop, step], got {spec!r}")
    start, stop, step = spec
    if step == 0:
        raise ValueError("'range' step must be non-zero")
    if start != stop and (stop - start) * step < 0:          # sign-mismatched -> empty sweep otherwise
        raise ValueError(
            f"'range' step sign disagrees with [start, stop] direction: {spec!r}"
        )
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in spec)
    values: List[Union[int, float]] = []
    tol = abs(step) * 1e-9
    n = int(round((stop - start) / step))
    for i in range(n + 2):
        v = start + i * step
        if (step > 0 and v > stop + tol) or (step < 0 and v < stop - tol):
            break
        values.append(v if all_int else round(v, 10))
    if not values:
        raise ValueError(f"'range' expanded to no values: {spec!r}")
    return values


def _expand_log_range(spec: List[Union[int, float]]) -> List[float]:
    r"""Expand [start, stop, count] into inclusive geometrically spaced values."""
    if len(spec) != 3:
        raise ValueError(f"'log_range' must be [start, stop, count], got {spec!r}")
    start, stop, count = spec
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, (int, float))
        or not isinstance(stop, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(stop))
        or start <= 0
        or stop <= 0
    ):
        raise ValueError(f"'log_range' endpoints must be strictly positive finite numbers: {spec!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"'log_range' count must be an integer >= 2, got {count!r}")

    log_start = math.log(float(start))
    log_stop = math.log(float(stop))
    values = [
        float(f"{math.exp(log_start + i * (log_stop - log_start) / (count - 1)):.12g}")
        for i in range(count)
    ]
    values[0] = float(start)
    values[-1] = float(stop)
    return values


def _sweep_values(sweep: Dict[str, Any]) -> List[Any]:
    sources = [key for key in ("values", "range", "log_range") if key in sweep]
    if len(sources) != 1:
        if not sources:
            raise KeyError(
                f"single-field sweep must define exactly one of 'values', 'range', or 'log_range': {sweep!r}"
            )
        raise ValueError(
            f"single-field sweep must define exactly one of 'values', 'range', or 'log_range'; "
            f"got {sources!r}"
        )
    if sources[0] == "values":
        return list(sweep["values"])
    if sources[0] == "range":
        return _expand_range(sweep["range"])
    return _expand_log_range(sweep["log_range"])


def sweep_n_runs(sweep: Dict[str, Any]) -> int:
    if "parameter_grid" in sweep:
        name = next((key for key, value in SWEEPS.items() if value is sweep), None)
        if name is None:
            raise ValueError("parameter-matched sweep is not registered in SWEEPS")
        n_cells = len(_parameter_match_selection(name)["selected"])
    else:
        n_cells = len(sweep["configs"]) if "configs" in sweep else len(_sweep_values(sweep))
    return n_cells * len(sweep.get("seeds") or [None])         # x seeds when the sweep is multi-seeded


def make_run_overrides(sweep_name: str) -> List[Tuple[str, Dict[str, Any]]]:
    r"""(label, overrides) pairs for a sweep; ``requires`` is folded into every override dict.

    The returned ``overrides`` is the FULL set of field changes for that cell (prerequisites
    first, then the swept field/arm), so the caller merges one dict onto the baseline.
    """
    sweep = SWEEPS[sweep_name]
    if "parameter_grid" in sweep:
        return [
            (str(record["label"]), dict(record["overrides"]))
            for record in _parameter_match_selection(sweep_name)["selected"]
        ]
    requires = sweep.get("requires", {})
    runs: List[Tuple[str, Dict[str, Any]]] = []
    if "configs" in sweep:
        for arm in sweep["configs"]:
            arm = dict(arm)
            label = arm.pop("label")
            runs.append((label, {**requires, **arm}))
    else:
        param = sweep["param"]
        for value in _sweep_values(sweep):
            overrides = {**requires, param: value}
            runs.append((f"{param}={value}", overrides))
    return runs


def _resolved_sweep_baseline() -> Dict[str, Any]:
    r"""Return the one broadly compatible baseline used before applying every sweep arm.

    ``BASELINE_CONFIG`` remains the user's explicit click-to-run declaration. Resolve the common
    sweep baseline once before expansion so labels and arm overrides state only their contrast.
    """
    resolved = copy.deepcopy(BASELINE_CONFIG)
    return resolved


# =============================================================================
# LOADERS  -- guard #2: memoised on the fields that actually change the stream.
# =============================================================================
_LOADER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def get_loader(
    dataset:     str,
    seq_len:     int,
    batch_size:  int,
    split:       str,

    *,
    max_tokens:  Optional[int] = None,
    vocab_size:  Optional[int] = None,
    eval_stride: Optional[int] = None,
) -> Any:
    r"""DataLoader for ``dataset``/``split``. A missing cache raises ``FileNotFoundError``.

    Memoized on ``(dataset, seq_len, batch_size, split, cap, vocab_size, stride)`` so runs that do not
    change those reuse the already-loaded immutable dataset and its attached byte identity, while a
    sweep over ``batch_size`` / ``max_seq_len`` / ``vocab_size`` / ``eval_stride`` correctly builds a
    distinct, matching loader. ``max_tokens`` caps only the train split (validation is always full).
    The loader never substitutes synthetic data for a missing real corpus -- that would mislabel
    synthetic numbers as a corpus measurement.

    ``eval_stride`` mirrors ``train_vfe3._select_loader``: it applies to validation/test only and is
    ignored for train (which shuffles and drops its tail, so the exactly-once eval contract that
    makes a stride meaningful does not apply). It MUST stay in the memo key -- a sweep over
    ``eval_stride`` would otherwise be served the first arm's loader and every arm would silently
    report the same windows.
    """
    data_seed_override = _validated_data_seed_override()
    cap = max_tokens if split == "train" else None
    stride = None if split == "train" else eval_stride
    key = (dataset, seq_len, batch_size, split, cap, vocab_size, stride)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    # Split-aware loader semantics, mirroring train_vfe3._select_loader: only the train stream is
    # shuffled and tail-dropped; validation/test read the WHOLE split in a stable order so the
    # held-out PPL is a full-corpus measurement (make_dataloader defaults to the train regime, so
    # the eval flags must be passed explicitly here). The TRAIN shuffle order is fixed to DATA_SEED
    # when set (an explicit generator, as in train_vfe3._select_loader); None -> no generator ->
    # legacy global-RNG shuffle, pinned to cfg.seed by run_single's post-build reseed. The cached
    # generator's state advances across cells; run_single re-pins it to DATA_SEED per cell.
    gen = (
        torch.Generator().manual_seed(data_seed_override)
        if split == "train" and data_seed_override is not None else None
    )
    loader = make_dataloader(dataset, split, seq_len, batch_size,
                             shuffle=(split == "train"), drop_last=(split == "train"),
                             max_tokens=cap, stride=stride, vocab_size=vocab_size, generator=gen)
    _LOADER_CACHE[key] = loader
    return loader


# =============================================================================
# SINGLE-RUN EXECUTOR
# =============================================================================

def _cell_cfg_dict(
    overrides:  Dict[str, Any],

    *,
    seed:       int,
    max_steps:  Optional[int] = None,
) -> Dict[str, Any]:
    r"""The exact kwargs dict a cell's VFE3Config is built from (baseline + overrides + run knobs).

    Single source of truth for cell construction, shared by ``run_single`` and the resume
    staleness check so the cached-config comparison is faithful.
    """
    d = _resolved_sweep_baseline()
    d.update(overrides)
    # eval_stride is bounded by max_seq_len -- config.__post_init__ rejects a stride above it -- and
    # the baseline ships 32. Any arm that shrinks max_seq_len below that (a context-length sweep, or
    # a test shrinking the model to a toy) would otherwise be an INVALID cell and be silently
    # bucketed as error_kind='config'. An arm whose context is shorter than the stride has no
    # meaningful sliding window at all, so it drops to the disjoint (None) default rather than being
    # clamped to a stride that would re-score nearly every window.
    _stride, _seq = d.get("eval_stride"), d.get("max_seq_len")
    if _stride is not None and _seq is not None and _stride > _seq:
        d["eval_stride"] = None
    d["checkpoint_interval"] = 0                             # no per-cell step_N.pt blowup
    d["seed"] = _require_exact_seed(seed, "seed")
    if max_steps is not None:
        d["max_steps"] = int(max_steps)
    return d


def _validated_stop_after_steps(stop_after_steps: Optional[int]) -> Optional[int]:
    """Validate the optional absolute stopping step independently of the cosine horizon."""
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps <= 0):
        raise ValueError("stop_after_steps must be an exact positive integer or None")
    return stop_after_steps


def _gauge_reporting_fields(
    cfg: VFE3Config,
    *,
    history: Optional[List[Dict[str, object]]] = None,
    executable_build=None,
) -> Dict[str, object]:
    """Return the executable scientific-certificate fields carried by an ablation row."""
    report = _pure_path_report(
        cfg,
        ([] if history is None else history),
        executable_build=executable_build,
    )
    toggles = report["config_toggles"]
    gauge_flags = report["gauge_flags"]
    return {
        "head_mixer_compatibility":          cfg.head_mixer_compatibility,
        "head_mixer_gauge_compatible":       bool(cfg.head_mixer_gauge_compatible),
        "block_mlp_structural_mode":         str(toggles["block_mlp_structural_mode"]),
        "block_mlp_covariance_contract":     str(toggles["block_mlp_covariance_contract"]),
        "block_mlp_intertwiner_compatible":  bool(gauge_flags["block_mlp_intertwiner_compatible"]),
        "on_gauge_pure_path":                bool(report["on_gauge_pure_path"]),
        "on_causal_lm_path":                 bool(report["on_causal_lm_path"]),
        "transport_exactness_status":         str(report["transport_exactness_status"]),
        "on_theory_pure_path":                bool(report["on_theory_pure_path"]),
    }

_EXECUTION_DERIVED_REPORT_FIELDS = frozenset({
    "block_mlp_structural_mode",
    "block_mlp_covariance_contract",
    "block_mlp_intertwiner_compatible",
    "on_gauge_pure_path",
    "transport_exactness_status",
    "on_theory_pure_path",
})
_EXACTNESS_STATUSES = frozenset({"exact", "approximate", "not_applicable", "unknown"})


def _merge_gauge_reporting_defaults(
    result: Dict[str, object],
    expected_fields: Mapping[str, object],
) -> None:
    """Fill absent report fields without replacing persisted execution-derived evidence."""
    for key, value in expected_fields.items():
        result.setdefault(key, value)


def _execution_derived_reporting_fields_are_valid(result: Mapping[str, object]) -> bool:
    """Fail closed on missing or malformed persisted execution-derived certificate fields."""
    if not _EXECUTION_DERIVED_REPORT_FIELDS <= result.keys():
        return False
    return (
        isinstance(result["block_mlp_structural_mode"], str)
        and isinstance(result["block_mlp_covariance_contract"], str)
        and type(result["block_mlp_intertwiner_compatible"]) is bool
        and type(result["on_gauge_pure_path"]) is bool
        and result["transport_exactness_status"] in _EXACTNESS_STATUSES
        and type(result["on_theory_pure_path"]) is bool
    )


@torch.no_grad()
def _cell_diagnostics(
    model:   VFEModel,
    cfg:     VFE3Config,
    loader:  Any,
    device:  torch.device,
) -> Dict[str, Any]:
    r"""Per-cell converged-state diagnostics for the gauge / entropy / equivariance experiments.

    OPT-IN: only runs when a sweep declares ``collect_diagnostics: True``, so the default tuning
    sweeps pay nothing. One faithful converged-state replay of validation sequence 0
    (``viz.extract.converged_state``) feeds the scalars the experiments read out of
    ``sweep_results.csv``:

      * ``attn_entropy``        mean attention-row entropy H(beta)                       [C1/EXP-4]
      * ``omega_identity_dev``  max |Omega_ij - I| over off-diagonal pairs (confirms     [A1/EXP-2]
                                gauge_transport='off' gives Omega = I to float eps)
      * ``builder_resid``       median builder-break gauge residual of the head mixer    [A2/EXP-9]
                                (eps under the tied gauge, grows under the untied one)
      * ``gauge_resid_in/out``  median in-/out-of-group congruence residual of the       [A1/EXP-2,
                                converged attention operator                              A3/EXP-10]

    Each probe is isolated: a failure (e.g. OOM on the O(N^2 K^2) Omega at large K) is logged and
    drops that scalar, never crashing the cell. The dominant cost is building Omega inside
    ``converged_state``; the experiment sweeps run at the baseline K=20, where it is cheap.
    """
    out: Dict[str, Any] = {}
    try:
        batch = next(iter(loader))
        token_ids = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        cstate = converged_state(model, token_ids)
    except Exception as exc:                                  # converged-state replay failed wholesale
        logger.warning("  [diagnostics: converged_state failed] %s", exc)
        return out

    def _probe(name: str, fn) -> None:
        try:
            out[name] = float(fn())
        except Exception as exc:
            logger.warning("  [diagnostics: %s skipped] %s", name, exc)

    _probe("attn_entropy", lambda: attention_entropy(cstate["beta"]))

    def _omega_identity_dev() -> torch.Tensor:
        omega = cstate["omega"]
        if not isinstance(omega, torch.Tensor):
            to_dense = getattr(omega, "to_dense_omega", None)
            if not callable(to_dense):
                raise TypeError(
                    "omega identity diagnostics require a tensor or an explicit dense converter"
                )
            omega = to_dense()
        if not isinstance(omega, torch.Tensor) or omega.ndim < 4:
            raise TypeError("omega identity diagnostics require (..., N, N, K, K) transport")
        n, k = omega.shape[0], omega.shape[-1]
        off = ~torch.eye(n, dtype=torch.bool, device=omega.device)
        eye = torch.eye(k, device=omega.device, dtype=omega.dtype)
        return (omega[off] - eye).abs().max()
    _probe("omega_identity_dev", _omega_identity_dev)

    # Pass the diagonal/full flag explicitly (cfg.diagonal_covariance) rather than letting the
    # metrics infer it from shape -- a diagonal sigma is (N, K), which is ambiguous with a full
    # (N, K, K) whenever the sequence length N happens to equal K.
    diag = cfg.diagonal_covariance
    if model.head_mixer is not None:
        def _builder_resid() -> torch.Tensor:
            r = head_mixer_gauge_residual(cstate["mu"], cstate["sigma"], model.head_mixer, model.group,
                                          diagonal=diag)
            return torch.cat([r["mu_residual"], r["sigma_residual"]]).median()
        _probe("builder_resid", _builder_resid)

    try:
        res = gauge_equivariance_residual(
            cstate["mu"], cstate["sigma"], cstate["omega"], model.group,
            kappa=cfg.kappa_beta, renyi_order=cfg.renyi_order, kl_max=cfg.kl_max,
            eps=cfg.eps, divergence_family=cfg.divergence_family, diagonal=diag)
        out["gauge_resid_in"]  = float(res["energy_in_group"].median())
        out["gauge_resid_out"] = float(res["energy_out_group"].median())
    except Exception as exc:
        logger.warning("  [diagnostics: gauge_resid skipped] %s", exc)

    # Dong rank-one residual r(X) of the converged per-token mean cloud (F2/EXP-7). The final-layer
    # scalar is a CSV column (a cheap rank-collapse readout for every diagnostics sweep); the full
    # per-layer curve is one extra deep replay and rides in the cell JSON only (not CSV) for the
    # rank-residual depth-overlay figure (_plot_rank_collapse).
    _probe("rank_resid", lambda: rank_one_residual(cstate["mu"]))
    try:
        curve = across_layer_belief_trace(model, token_ids)["rank_one_residual"]
        out["rank_resid_by_layer"] = [float(x) for x in curve]
    except Exception as exc:
        logger.warning("  [diagnostics: rank_resid_by_layer skipped] %s", exc)

    # -tau^{-1} Cov_beta(E, dE) attention-entropy gradient gap (C1/EXP-4): the magnitude of the
    # belief gradient the canonical entropy term adds over the surrogate, on the converged belief.
    # Gated to L=1: the extractor builds the single-block operating point, so it would report a
    # misleading layer-1 value on the L>1 cells of other sweeps (gauge_transport L2, rho_handoff L4).
    if cfg.n_layers == 1:
        _probe("cov_gap", lambda: attention_entropy_cov_gap(model, token_ids)["cov_gap"])

    # kl_max saturation fraction of the converged pairwise energy (B2/EXP-12): for Renyi alpha>1 the
    # non-PD blend pins E_ij at kl_max with zero gradient, so this fraction climbs in the alpha>1 tail
    # and explains a non-monotone H(beta)-vs-alpha curve.
    _probe("energy_klmax_frac", lambda: guard_saturation(
        cstate["sigma"], cstate["energy"], cstate["self_div"], diagonal=diag,
        eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max)["energy_klmax_frac"])
    return out


@torch.no_grad()
def _eval_at_growing_n(
    model:   Any,
    cfg:     VFE3Config,
    dataset: str,
    device:  torch.device,

    *,
    requested_lengths: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    r"""H1/EXP-13: held-out CE at sequence lengths from the trained ``max_seq_len`` outward.

    Re-windows the validation split at each N (anchor ``max_seq_len``, then 1.5/2/3/4x) via
    ``get_loader`` and scores the SAME trained model. Offset attention priors (alibi / t5, functions
    of |i-j|) and the causal mask rebuild at runtime N and extrapolate; the absolute schemes
    (``pos_phi='learned'`` table -- now clamped past the table; RoPE) degrade. Each N is isolated: a
    too-short split or any failure remains in the returned domain record and makes the enclosing
    cell incomplete when that point is mandatory. Batch size scales inversely with N to hold the
    approximate token load fixed. Returns one status record for every requested N."""
    base = int(cfg.max_seq_len)
    n_list = (
        _default_extrapolation_lengths(base)
        if requested_lengths is None
        else _validated_extrapolation_lengths(requested_lengths, "requested_lengths")
    )
    out: List[Dict[str, Any]] = []
    for n in n_list:
        effective_batch_size = max(1, int(cfg.batch_size) * base // n)
        try:
            # Same eval contract as the cell's own validation metric, so the CE-vs-N curve and the
            # headline PPL are the same measurement at different N. cfg.eval_stride <= max_seq_len <= n
            # holds for every extrapolation length, so the stride stays valid as N grows.
            loader = get_loader(dataset, n, effective_batch_size, "validation",
                                vocab_size=cfg.vocab_size, eval_stride=cfg.eval_stride)
            m = evaluate(model, loader, device=device)
            ce = float(m["ce"])
            ppl = float(m["ppl"])
            if not math.isfinite(ce) or not math.isfinite(ppl) or ppl <= 0.0:
                raise ValueError(f"nonfinite or nonpositive extrapolation metrics: ce={ce}, ppl={ppl}")
            out.append({
                "n":                    n,
                "status":               "success",
                "ce":                   ce,
                "ppl":                  ppl,
                "effective_batch_size": effective_batch_size,
            })
        except Exception as exc:                              # short split / OOM -> retained failed point
            logger.warning("  [extrapolation eval N=%d skipped] %s", n, exc)
            out.append({
                "n":                    n,
                "status":               "failed",
                "failure_reason":       f"{type(exc).__name__}: {exc}",
                "effective_batch_size": effective_batch_size,
            })
    return out


def run_single(
    label:       str,
    overrides:   Dict[str, Any],
    run_dir:     Path,

    *,
    dataset:                str,
    device:                 torch.device,
    seed:                   int,
    collect_diagnostics:    bool          = False,
    collect_extrapolation:  bool          = False,
    paired_token_bootstrap: bool          = False,
    save_model_artifacts:   bool          = True,
    max_tokens:             Optional[int] = None,
    max_steps:              Optional[int] = None,
    stop_after_steps:       Optional[int] = None,
    extrapolation_lengths:  Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    r"""Build a fresh model from baseline+overrides, train it, and score validation.

    Returns a headline dict with ``primary_val_ppl`` (= min of any periodic best and the
    final validation PPL) and bookkeeping. A cross-field config violation is caught and
    returned as ``error_kind = "config"`` (not raised), keeping it distinct from a training
    crash; the headline is ``inf`` either way so it sorts to the bottom of the leaderboard.
    """
    save_model_artifacts = _require_exact_bool(
        save_model_artifacts,
        "save_model_artifacts",
    )
    diagnostic_flags = _validated_diagnostic_flags({
        "collect_diagnostics":    collect_diagnostics,
        "collect_extrapolation":  collect_extrapolation,
        "paired_token_bootstrap": paired_token_bootstrap,
    })
    collect_diagnostics = diagnostic_flags["collect_diagnostics"]
    collect_extrapolation = diagnostic_flags["collect_extrapolation"]
    paired_token_bootstrap = diagnostic_flags["paired_token_bootstrap"]
    requested_extrapolation_lengths = (
        None
        if extrapolation_lengths is None
        else _validated_extrapolation_lengths(
            extrapolation_lengths,
            "extrapolation_lengths",
        )
    )
    if not collect_extrapolation and requested_extrapolation_lengths:
        raise ValueError("extrapolation_lengths needs collect_extrapolation=True")

    cfg_dict = _cell_cfg_dict(overrides, seed=seed, max_steps=max_steps)
    try:
        cfg = VFE3Config(**cfg_dict)
        stop_after_steps = _validated_stop_after_steps(stop_after_steps)
        n_steps = cfg.max_steps if stop_after_steps is None else min(cfg.max_steps, stop_after_steps)
    except (ValueError, NotImplementedError, TypeError) as exc:
        logger.warning("  [config rejected] %s: %s", label, exc)
        return {"label": label, "error_kind": "config", "error": str(exc),
                "primary_val_ppl": float("inf"), "seed": seed,
                "overrides": _jsonable(overrides)}

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train",
                              max_tokens=max_tokens, vocab_size=cfg.vocab_size)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation",
                              vocab_size=cfg.vocab_size, eval_stride=cfg.eval_stride)
    loaded_data_sources = _loader_ablation_source_identities(
        train_loader,
        val_loader,
        dataset=dataset,
        max_tokens=max_tokens,
    )

    # Bits-per-CHARACTER correction for val BPC, mirroring train_vfe3. When normalization is
    # unavailable, BPC stays null and the separately named bits-per-token metric remains defined.
    val_tpc = tokens_per_char(dataset, "validation")

    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(
        run_dir,
        cfg,
        model,
        dataset=dataset,
        device=device,
        save_model_artifacts=save_model_artifacts,
    )

    # Reproducible, ORDER-INDEPENDENT data stream. Model construction above consumes a
    # config-dependent amount of RNG, and a cached loader's shuffle (its own generator, or the
    # global RNG for make_dataloader) otherwise advances across runs -- so without this reseed
    # the same config would see different batches depending on its position in the sweep, and
    # the comparison would be confounded by data order. Reseeding here, after the model is built,
    # pins every cell to the same batch sequence regardless of order.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    for loader, is_train in ((train_loader, True), (val_loader, False)):  # synthetic loaders carry their own generator
        if getattr(loader, "generator", None) is not None:
            # The train loader's DATA_SEED generator (get_loader) is re-pinned to DATA_SEED, not
            # cfg.seed: its state advanced across cells (memoised loader), and train_vfe3 hands
            # each run a FRESH generator at DATA_SEED -- reseeding here reproduces that exactly.
            loader.generator.manual_seed(
                _effective_data_seed(cfg.seed) if is_train else cfg.seed)

    print(f"    K={cfg.embed_dim} heads={len(model.group.irrep_dims)} group={cfg.gauge_group} "
          f"family={cfg.family} | steps={n_steps} cosine_horizon={cfg.max_steps} "
          f"batch={cfg.batch_size} | {n_params:,} params")
    for _cov in coverage_lines(train_loader, n_steps, dataset):
        print(f"   {_cov}")

    # Terminal artifact set (PB-02): a VALIDATION-ONLY finalizer runs once as a train() callback --
    # immediately after the final optimizer step -- so even a default cell (log/eval interval above
    # max_steps, checkpoint_interval=0) publishes terminal metrics, validation results, summary,
    # provenance, and figures WITHOUT ever scoring a test split. Default-on model persistence also
    # writes best_model.pt and a resumable terminal checkpoint; lightweight optimization cells omit
    # both. The returned merge mapping carries the validation headline and artifact policy. Periodic
    # log/eval/checkpoint cadence is not relied on for this post-final-step artifact set.
    terminal_result: Dict[str, Any] = {}
    train_start = time.perf_counter()

    def _terminal_callback(state, callback_losses):
        terminal_result.update(finalize_validation_run(
            model, artifacts, cfg, val_loader,
            tokens_per_char=val_tpc,
            train_loader=train_loader,
            losses=callback_losses,
            data_seed=_effective_data_seed(cfg.seed),
            max_tokens=max_tokens,
            tokenizer_tag=_tokenizer_tag(dataset),
            device=device,
            wall_time=time.perf_counter() - train_start,
            logger=logger,
            terminal_state=state,
        ))

    train(
        model, train_loader, cfg,
        n_steps=n_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=device,
        logger=logger,
        artifacts=artifacts,
        generate_samples=False,                              # pure silent path: no sample text
        terminal_callback=_terminal_callback,
    )

    result: Dict[str, Any] = {
        "label":                label,
        "error_kind":           None,
        "n_params":             n_params,
        "n_steps":              n_steps,
        "max_steps":            cfg.max_steps,
        "stop_after_steps":     stop_after_steps,
        "seed":                 int(cfg.seed),
        "overrides":            _jsonable(overrides),
        "max_tokens":           (int(max_tokens) if max_tokens is not None else None),
        "_loaded_data_sources": loaded_data_sources,
        "save_model_artifacts": save_model_artifacts,
    }
    result.update(_gauge_reporting_fields(
        cfg, history=artifacts.history, executable_build=model.executable_build,
    ))
    result.update(terminal_result)                           # primary/final/best/terminal_checkpoint headline

    # PB-07 opt-in: publish the per-token validation nats (the paired within-run bootstrap the
    # component-ablation forest consumes) as an atomic sibling artifact, and record its SHA-256, size,
    # numel, and dtype in the marker. Written BEFORE run_sweep publishes the reuse contract and
    # completion marker, so the post-contract validator checks a file that already exists on disk. The
    # stable, unshuffled validation loader plus the shared seed make the vector position-aligned
    # across arms. This per-unit extraction is the ONLY added validation replay; aggregate final
    # validation stays owned by the terminal callback above.
    token_identity: Optional[Dict[str, Any]] = None
    if paired_token_bootstrap:
        per_token = per_unit_eval_nats(model, val_loader, device=device)["per_token_nats"]
        final_path = run_dir / "val_token_nats.pt"
        with _unique_sibling_temp(final_path) as tmp_path:
            torch.save(per_token.detach().cpu(), tmp_path)
            _atomic_replace(final_path, tmp_path)
        token_identity = {
            "path":       final_path.name,
            "sha256":     _sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "numel":      int(per_token.numel()),
            "dtype":      str(per_token.dtype),
        }
    result["paired_token_bootstrap"] = paired_token_bootstrap
    result["val_token_nats_path"] = (
        token_identity["path"] if token_identity is not None else None
    )
    result["val_token_nats_sha256"] = (
        token_identity["sha256"] if token_identity is not None else None
    )
    result["val_token_nats_size_bytes"] = (
        token_identity["size_bytes"] if token_identity is not None else None
    )
    result["val_token_nats_numel"] = (
        token_identity["numel"] if token_identity is not None else None
    )
    result["val_token_nats_dtype"] = (
        token_identity["dtype"] if token_identity is not None else None
    )

    if collect_diagnostics:                                  # opt-in converged-state probes (S2)
        result.update(_cell_diagnostics(model, cfg, val_loader, device))
    if collect_extrapolation:                                # opt-in growing-N eval (H1/EXP-13)
        result["extrap_ce"] = _eval_at_growing_n(
            model,
            cfg,
            dataset,
            device,
            requested_lengths=requested_extrapolation_lengths,
        )
    return result


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    r"""Coerce override values (e.g. tuples in cross_couplings) to JSON-friendly forms."""
    return json.loads(json.dumps(d, default=str))


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# SWEEP DRIVER
# =============================================================================
_CSV_COLUMNS = [
    "sweep", "label", "error_kind", "primary_val_ppl", "final_val_ppl",
    "final_val_ce", "final_val_bits_per_token", "final_val_bpc", "best_val_ppl", "final_train_loss",
    "n_params", "target_n_params", "param_difference", "param_relative_deviation",
    "n_steps", "max_steps", "stop_after_steps",
    "head_mixer_compatibility", "head_mixer_gauge_compatible",
    "block_mlp_structural_mode", "block_mlp_covariance_contract",
    "block_mlp_intertwiner_compatible", "on_gauge_pure_path", "on_causal_lm_path",
    "transport_exactness_status", "on_theory_pure_path",
    # opt-in per-cell converged-state diagnostics (S2; empty unless the sweep sets collect_diagnostics)
    "attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in", "gauge_resid_out",
    "rank_resid", "cov_gap", "energy_klmax_frac",
    # opt-in paired-token artifact identity (PB-07; empty unless the sweep sets paired_token_bootstrap)
    "paired_token_bootstrap", "val_token_nats_path", "val_token_nats_sha256",
    "val_token_nats_size_bytes", "val_token_nats_numel", "val_token_nats_dtype",
    "wall_time_s", "seed", "error",
]

_DIAGNOSTIC_RESULT_KEYS = {
    "attn_entropy", "omega_identity_dev", "builder_resid", "gauge_resid_in", "gauge_resid_out",
    "rank_resid", "rank_resid_by_layer", "cov_gap", "energy_klmax_frac",
}
_BASE_REQUIRED_DIAGNOSTIC_KEYS = {
    "attn_entropy",
    "energy_klmax_frac",
    "gauge_resid_in",
    "gauge_resid_out",
    "omega_identity_dev",
    "rank_resid",
}


_CELL_CONTRACT_SCHEMA_VERSION = 5
_SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION = 6
_ABLATION_CELL_OWNER_SCHEMA_VERSION = 1
_ABLATION_CELL_OWNER_FILENAME = "ablation_cell_owner.json"
_SWEEP_DIAGNOSTIC_FLAG_KEYS = {
    "collect_diagnostics",
    "collect_extrapolation",
    "paired_token_bootstrap",
}


def _require_exact_seed(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact non-negative integer, got {value!r}")
    return value


def _require_exact_bool(value: object, field: str) -> bool:
    """Return one exact Boolean without accepting numeric or string truthiness surrogates."""
    if type(value) is not bool:
        raise TypeError(f"{field} must be an exact boolean, got {value!r}")
    return value


def _validated_sweep_seeds(sweep: Mapping[str, object], fallback_seed: object) -> List[int]:
    """Resolve one sweep's seed axis without bool/fraction/string coercion or duplicate cells."""
    fallback = _require_exact_seed(fallback_seed, "seed")
    if "seeds" not in sweep:
        return [fallback]
    raw = sweep["seeds"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("sweep['seeds'] must be a non-empty list or tuple of unique integers")
    seeds = [_require_exact_seed(seed, "sweep['seeds'] entry") for seed in raw]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"sweep['seeds'] must be unique, got {seeds!r}")
    return seeds


def _validated_data_seed_override() -> Optional[int]:
    """Validate the optional global train-loader seed without coercion."""
    if DATA_SEED is None:
        return None
    return _require_exact_seed(DATA_SEED, "DATA_SEED")


def _effective_data_seed(fallback_seed: object) -> int:
    """Return the validated override or one exact per-cell fallback seed."""
    override = _validated_data_seed_override()
    return override if override is not None else _require_exact_seed(fallback_seed, "seed")


_ABLATION_SOURCE_SPLITS = ("train", "validation")


def _validated_ablation_source_identities(
    value: object,
) -> Dict[str, Dict[str, object]]:
    """Return a detached, JSON-safe identity for exactly the splits an ablation cell consumes."""
    if not isinstance(value, Mapping) or set(value) != set(_ABLATION_SOURCE_SPLITS):
        raise ValueError(
            "ablation data source identities must contain exactly train and validation")
    normalized: Dict[str, Dict[str, object]] = {}
    for split in _ABLATION_SOURCE_SPLITS:
        source = value[split]
        if not isinstance(source, Mapping):
            raise TypeError(f"{split} source identity must be a mapping")
        detached = json.loads(json.dumps(
            dict(source), sort_keys=True, ensure_ascii=False, allow_nan=False))
        if (not isinstance(detached, dict)
                or not isinstance(detached.get("format"), str)
                or not detached["format"]
                or type(detached.get("size_bytes")) is not int
                or detached["size_bytes"] < 0
                or not isinstance(detached.get("sha256"), str)
                or not detached["sha256"]):
            raise ValueError(f"{split} source identity is incomplete")
        normalized[split] = detached
    return normalized


def _ablation_source_identities(
    dataset:   str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Hash each corpus split once for a shared pre-sweep filesystem snapshot."""
    return _validated_ablation_source_identities({
        split: cache_source_identity(dataset, split, cache_dir=cache_dir)
        for split in _ABLATION_SOURCE_SPLITS
    })


def _loader_ablation_source_identities(
    train_loader: object,
    val_loader:   object,

    *,
    dataset:    str,
    max_tokens: Optional[int],
) -> Optional[Dict[str, Dict[str, object]]]:
    r"""Snapshot the identities attached to the immutable datasets actually used by one cell.

    ``make_dataloader`` binds every loaded tensor to the source identity that remained stable across
    its load. A custom loader without that contract remains usable by ``run_single`` directly, but
    the sweep receives ``None`` and therefore refuses to publish a reusable success contract.
    """
    sources: Dict[str, object] = {}
    try:
        for split, loader, expected_cap in (
            ("train", train_loader, max_tokens),
            ("validation", val_loader, None),
        ):
            data_identity = getattr(getattr(loader, "dataset", None), "data_identity", None)
            if (not isinstance(data_identity, Mapping)
                    or data_identity.get("schema_version") != 2
                    or data_identity.get("dataset") != dataset
                    or data_identity.get("split") != split
                    or data_identity.get("max_tokens") != expected_cap):
                raise ValueError(f"{split} loader data identity is unavailable or mismatched")
            sources[split] = data_identity.get("source")
        return _validated_ablation_source_identities(sources)
    except (TypeError, ValueError):
        logger.warning("  [loaded corpus identity unavailable -> reuse contract forbidden]")
        return None


def _validated_ablation_code_identity(value: object) -> Dict[str, object]:
    """Return one detached, usable source identity for an ablation invocation."""
    if not isinstance(value, Mapping):
        raise TypeError("ablation code identity must be a mapping")
    detached = json.loads(json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False))
    if set(detached) == {"package_code_sha256", "ablation_runner_sha256"}:
        for field in ("package_code_sha256", "ablation_runner_sha256"):
            digest = detached[field]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError(f"ablation code identity {field} is not a SHA-256 digest")
        return detached

    # Legacy shape remains accepted for persisted-fixture and direct-API compatibility. A current
    # production contract uses the two source digests above, so a legacy cohort never equals it.
    git_sha = detached.get("git_sha")
    git_dirty = detached.get("git_dirty")
    fingerprint = detached.get("git_dirty_fingerprint")
    if not isinstance(git_sha, str) or not git_sha or type(git_dirty) is not bool:
        raise ValueError("ablation code identity is unavailable")
    if ((git_dirty and (not isinstance(fingerprint, str) or not fingerprint))
            or (not git_dirty and fingerprint is not None)):
        raise ValueError("ablation code identity has an inconsistent dirty-tree fingerprint")
    return detached


def _validated_diagnostic_flags(value: object) -> Dict[str, bool]:
    """Validate the three invocation-wide artifact requests bound into every cell contract."""
    if not isinstance(value, Mapping) or set(value) != _SWEEP_DIAGNOSTIC_FLAG_KEYS:
        raise ValueError(
            "ablation diagnostic flags must contain exactly collect_diagnostics, "
            "collect_extrapolation, and paired_token_bootstrap"
        )
    return {
        key: _require_exact_bool(value[key], key)
        for key in sorted(_SWEEP_DIAGNOSTIC_FLAG_KEYS)
    }


def _sweep_diagnostic_request(sweep: Mapping[str, object]) -> Dict[str, bool]:
    """Validate raw sweep flags before experiment construction or artifact publication."""
    return _validated_diagnostic_flags({
        key: sweep.get(key, False)
        for key in _SWEEP_DIAGNOSTIC_FLAG_KEYS
    })


def _validated_required_diagnostic_keys(value: object) -> List[str]:
    """Return a unique, sorted list of diagnostic marker fields required for completion."""
    if not isinstance(value, (list, tuple)):
        raise TypeError("required diagnostic keys must be a list or tuple")
    keys = list(value)
    if (any(not isinstance(key, str) or key not in _DIAGNOSTIC_RESULT_KEYS for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("required diagnostic keys must be unique known result fields")
    return sorted(keys)


def _validated_min_extrapolation_points(value: object) -> int:
    """Return the exact non-negative number of finite distinct-N extrapolation points required."""
    if type(value) is not int or value < 0:
        raise ValueError("min_extrapolation_points must be an exact non-negative integer")
    return value


def _validated_extrapolation_lengths(value: object, field: str) -> List[int]:
    """Return a sorted exact positive requested-length domain without duplicates."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list or tuple of exact positive integers")
    lengths = list(value)
    if any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError(f"{field} must contain exact positive integers")
    if len(lengths) != len(set(lengths)):
        raise ValueError(f"{field} must not contain duplicate sequence lengths")
    return sorted(lengths)


def _default_extrapolation_lengths(base: object) -> List[int]:
    """Return the established base, 1.5x, 2x, 3x, and 4x growing-N domain."""
    if type(base) is not int or base <= 0:
        raise ValueError("extrapolation base length must be an exact positive integer")
    return sorted({base} | {int(round(base * m)) for m in (1.5, 2.0, 3.0, 4.0)})


def _validated_extrapolation_request(
    requested: object,
    mandatory: object,

    *,
    collect_extrapolation: bool,
) -> Tuple[List[int], List[int]]:
    """Validate the requested domain and its mandatory subset under one exact collection flag."""
    collect = _require_exact_bool(collect_extrapolation, "collect_extrapolation")
    requested_lengths = _validated_extrapolation_lengths(
        requested,
        "extrapolation_lengths",
    )
    mandatory_lengths = _validated_extrapolation_lengths(
        mandatory,
        "mandatory_extrapolation_lengths",
    )
    if collect:
        if len(requested_lengths) < 2:
            raise ValueError("collect_extrapolation requires at least two requested lengths")
        if not mandatory_lengths:
            raise ValueError("collect_extrapolation requires at least one mandatory length")
        if not set(mandatory_lengths).issubset(requested_lengths):
            raise ValueError("mandatory extrapolation lengths must be requested")
    elif requested_lengths or mandatory_lengths:
        raise ValueError("unrequested extrapolation cannot declare sequence lengths")
    return requested_lengths, mandatory_lengths


def _sweep_extrapolation_request(
    sweep: Mapping[str, object],
    diagnostic_flags: Mapping[str, bool],
) -> Tuple[List[int], List[int]]:
    """Resolve one sweep's exact growing-N domain and mandatory completion points."""
    flags = _validated_diagnostic_flags(diagnostic_flags)
    if not flags["collect_extrapolation"]:
        requested_raw = sweep.get("extrapolation_lengths", ())
        mandatory_raw = sweep.get("mandatory_extrapolation_lengths", ())
    else:
        requires = sweep.get("requires", {})
        if not isinstance(requires, Mapping):
            raise TypeError("sweep['requires'] must be a mapping")
        base = requires.get("max_seq_len", BASELINE_CONFIG["max_seq_len"])
        requested_raw = sweep.get(
            "extrapolation_lengths",
            _default_extrapolation_lengths(base),
        )
        requested_preview = _validated_extrapolation_lengths(
            requested_raw,
            "extrapolation_lengths",
        )
        mandatory_raw = sweep.get(
            "mandatory_extrapolation_lengths",
            ((max(requested_preview),) if requested_preview else ()),
        )
    return _validated_extrapolation_request(
        requested_raw,
        mandatory_raw,
        collect_extrapolation=flags["collect_extrapolation"],
    )


def _validated_seed_design(value: object) -> List[int]:
    """Return the sorted, unique model-seed panel used for cross-sweep comparisons."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("ablation seed design must be a non-empty list or tuple")
    seeds = [_require_exact_seed(seed, "seed_design") for seed in value]
    if len(seeds) != len(set(seeds)):
        raise ValueError("ablation seed design must not contain duplicates")
    return sorted(seeds)


def _baseline_semantic_config_fingerprint(
    max_steps: Optional[int],
    baseline_overrides: Optional[Mapping[str, object]] = None,
) -> str:
    """Fingerprint the effective unswept operating point independently of model seed."""
    cfg = VFE3Config(**_cell_cfg_dict(dict(baseline_overrides or {}), seed=0, max_steps=max_steps))
    return semantic_config_fingerprint(asdict(cfg))


def _sweep_required_outputs(
    sweep: Mapping[str, object],
    diagnostic_flags: Mapping[str, bool],
) -> Tuple[List[str], int, List[int], List[int]]:
    """Resolve the output-completeness profile declared by one sweep."""
    flags = _validated_diagnostic_flags(diagnostic_flags)
    collect_diagnostics = flags["collect_diagnostics"]
    extra = sweep.get("required_diagnostics", ())
    required = _validated_required_diagnostic_keys(extra)
    if collect_diagnostics:
        required = sorted(set(required) | _BASE_REQUIRED_DIAGNOSTIC_KEYS)
    elif required:
        raise ValueError("required_diagnostics needs collect_diagnostics=True")

    collect_extrapolation = flags["collect_extrapolation"]
    default_points = 2 if collect_extrapolation else 0
    minimum = _validated_min_extrapolation_points(
        sweep.get("min_extrapolation_points", default_points)
    )
    if collect_extrapolation and minimum < 2:
        raise ValueError("collect_extrapolation requires at least two extrapolation points")
    if not collect_extrapolation and minimum:
        raise ValueError("min_extrapolation_points needs collect_extrapolation=True")
    requested_lengths, mandatory_lengths = _sweep_extrapolation_request(sweep, flags)
    return required, minimum, requested_lengths, mandatory_lengths


def _requested_outputs_are_complete(
    result: Mapping[str, object],

    *,
    required_diagnostic_keys: Sequence[str],
    min_extrapolation_points: int,
    requested_extrapolation_lengths: Optional[Sequence[int]] = None,
    mandatory_extrapolation_lengths: Optional[Sequence[int]] = None,
) -> bool:
    """Whether one result contains every finite output promised by its sweep contract."""
    try:
        required = _validated_required_diagnostic_keys(required_diagnostic_keys)
        minimum = _validated_min_extrapolation_points(min_extrapolation_points)
    except (TypeError, ValueError):
        return False
    for key in required:
        value = result.get(key)
        if key == "rank_resid_by_layer":
            if not isinstance(value, list) or len(value) < 2:
                return False
            try:
                if any(not math.isfinite(float(item)) for item in value):
                    return False
            except (TypeError, ValueError):
                return False
            continue
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    if minimum == 0:
        return True
    curve = result.get("extrap_ce")
    if not isinstance(curve, list):
        return False
    try:
        requested = _validated_extrapolation_lengths(
            ([point.get("n") for point in curve] if requested_extrapolation_lengths is None
             else requested_extrapolation_lengths),
            "requested_extrapolation_lengths",
        )
        mandatory = _validated_extrapolation_lengths(
            ((max(requested),) if mandatory_extrapolation_lengths is None and requested else
             (() if mandatory_extrapolation_lengths is None else mandatory_extrapolation_lengths)),
            "mandatory_extrapolation_lengths",
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if len(requested) < minimum or not requested or not set(mandatory).issubset(requested):
        return False
    strict_domain_record = requested_extrapolation_lengths is not None
    seen_n = set()
    successful_n = set()
    for point in curve:
        if not isinstance(point, Mapping):
            return False
        n = point.get("n")
        if type(n) is not int or n <= 0 or n in seen_n or n not in requested:
            return False
        seen_n.add(n)
        effective_batch_size = point.get("effective_batch_size")
        if strict_domain_record and effective_batch_size is None:
            return False
        if effective_batch_size is not None and (
                type(effective_batch_size) is not int or effective_batch_size <= 0):
            return False
        if strict_domain_record and "status" not in point:
            return False
        status = point.get("status", "success")
        if status == "failed":
            reason = point.get("failure_reason")
            if not isinstance(reason, str) or not reason:
                return False
            continue
        if status != "success":
            return False
        try:
            ce = float(point["ce"])
            ppl = float(point["ppl"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(ce) or not math.isfinite(ppl) or ppl <= 0.0:
            return False
        successful_n.add(n)
    return (
        seen_n == set(requested)
        and len(successful_n) >= max(2, minimum)
        and max(requested) in successful_n
        and set(mandatory).issubset(successful_n)
    )


def _validated_sweep_aggregation_contract(value: object) -> Dict[str, object]:
    """Return one normalized invocation contract for cross-cell scientific aggregation."""
    if not isinstance(value, Mapping):
        raise TypeError("sweep aggregation contract must be a mapping")
    required = {
        "schema_version",
        "baseline_semantic_config_fingerprint",
        "seed_design",
        "dataset",
        "device",
        "tokenizer_tag",
        "data_seed_override",
        "max_tokens",
        "max_steps",
        "source_identities",
        "code_identity",
        "diagnostic_flags",
        "required_diagnostic_keys",
        "min_extrapolation_points",
        "requested_extrapolation_lengths",
        "mandatory_extrapolation_lengths",
        "save_model_artifacts",
    }
    # Optional extension preserves the contract identity of existing uncapped runs.
    if set(value) not in (required, required | {"stop_after_steps"}):
        raise ValueError("sweep aggregation contract has missing or unknown fields")
    if value.get("schema_version") != _SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported sweep aggregation contract schema")
    baseline_fingerprint = value.get("baseline_semantic_config_fingerprint")
    if (not isinstance(baseline_fingerprint, str) or len(baseline_fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in baseline_fingerprint)):
        raise ValueError("ablation baseline semantic config fingerprint must be a SHA-256 digest")
    seed_design = _validated_seed_design(value.get("seed_design"))
    dataset = value.get("dataset")
    device = value.get("device")
    tokenizer_tag = value.get("tokenizer_tag")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("sweep aggregation dataset must be a non-empty string")
    if not isinstance(device, str) or not device:
        raise ValueError("sweep aggregation device must be a non-empty string")
    if not isinstance(tokenizer_tag, str) or not tokenizer_tag:
        raise ValueError("sweep aggregation tokenizer_tag must be a non-empty string")

    data_seed_override = value.get("data_seed_override")
    if data_seed_override is not None:
        data_seed_override = _require_exact_seed(data_seed_override, "data_seed_override")
    max_tokens = value.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
        raise ValueError("sweep aggregation max_tokens must be an exact non-negative integer or null")
    stop_after_steps = _validated_stop_after_steps(value.get("stop_after_steps"))
    max_steps = value.get("max_steps")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("sweep aggregation max_steps must be an exact positive integer or null")
    save_model_artifacts = _require_exact_bool(
        value.get("save_model_artifacts"),
        "save_model_artifacts",
    )

    diagnostic_flags = _validated_diagnostic_flags(value.get("diagnostic_flags"))
    required_diagnostic_keys = _validated_required_diagnostic_keys(
        value.get("required_diagnostic_keys")
    )
    min_extrapolation_points = _validated_min_extrapolation_points(
        value.get("min_extrapolation_points")
    )
    requested_extrapolation_lengths, mandatory_extrapolation_lengths = (
        _validated_extrapolation_request(
            value.get("requested_extrapolation_lengths"),
            value.get("mandatory_extrapolation_lengths"),
            collect_extrapolation=diagnostic_flags["collect_extrapolation"],
        )
    )
    if diagnostic_flags["collect_diagnostics"] != bool(required_diagnostic_keys):
        raise ValueError("diagnostic collection and required keys are inconsistent")
    if diagnostic_flags["collect_extrapolation"]:
        if min_extrapolation_points < 2:
            raise ValueError("extrapolation collection requires at least two points")
    elif min_extrapolation_points != 0:
        raise ValueError("unrequested extrapolation cannot require output points")

    return {
        "schema_version":     _SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION,
        "baseline_semantic_config_fingerprint": baseline_fingerprint.lower(),
        "seed_design":        seed_design,
        "dataset":            dataset,
        "device":             device,
        "tokenizer_tag":      tokenizer_tag,
        "data_seed_override": data_seed_override,
        "max_tokens":         max_tokens,
        "max_steps":          max_steps,
        **({"stop_after_steps": stop_after_steps} if stop_after_steps is not None else {}),
        "source_identities":  _validated_ablation_source_identities(
            value.get("source_identities")
        ),
        "code_identity":      _validated_ablation_code_identity(value.get("code_identity")),
        "diagnostic_flags":   diagnostic_flags,
        "required_diagnostic_keys": required_diagnostic_keys,
        "min_extrapolation_points": min_extrapolation_points,
        "requested_extrapolation_lengths": requested_extrapolation_lengths,
        "mandatory_extrapolation_lengths": mandatory_extrapolation_lengths,
        "save_model_artifacts": save_model_artifacts,
    }


def _sweep_aggregation_contract(
    dataset:          str,
    diagnostic_flags: Mapping[str, bool],

    *,
    data_seed_override: Optional[int],
    max_tokens:        Optional[int],
    max_steps:         Optional[int],
    seed_design:       Sequence[int],
    source_identities: Mapping[str, object],
    code_identity:     Mapping[str, object],
    save_model_artifacts: bool = True,
    device:            str = "cpu",
    stop_after_steps:  Optional[int] = None,
    required_diagnostic_keys: Optional[Sequence[str]] = None,
    min_extrapolation_points: Optional[int]           = None,
    requested_extrapolation_lengths: Optional[Sequence[int]] = None,
    mandatory_extrapolation_lengths: Optional[Sequence[int]] = None,
    baseline_overrides: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Build the common invocation identity under which unlike labels, values, and seeds compare."""
    flags = _validated_diagnostic_flags(diagnostic_flags)
    required_keys = _validated_required_diagnostic_keys(
        (sorted(_BASE_REQUIRED_DIAGNOSTIC_KEYS) if flags["collect_diagnostics"] else ())
        if required_diagnostic_keys is None else required_diagnostic_keys
    )
    minimum = _validated_min_extrapolation_points(
        (2 if flags["collect_extrapolation"] else 0)
        if min_extrapolation_points is None else min_extrapolation_points
    )
    requested_default = (
        _default_extrapolation_lengths(BASELINE_CONFIG["max_seq_len"])
        if flags["collect_extrapolation"] else []
    )
    requested_raw = (
        requested_default if requested_extrapolation_lengths is None
        else requested_extrapolation_lengths
    )
    requested_preview = _validated_extrapolation_lengths(
        requested_raw,
        "requested_extrapolation_lengths",
    )
    requested, mandatory = _validated_extrapolation_request(
        requested_preview,
        ((max(requested_preview),) if mandatory_extrapolation_lengths is None and requested_preview
         else (() if mandatory_extrapolation_lengths is None
               else mandatory_extrapolation_lengths)),
        collect_extrapolation=flags["collect_extrapolation"],
    )
    return _validated_sweep_aggregation_contract({
        "schema_version":     _SWEEP_AGGREGATION_CONTRACT_SCHEMA_VERSION,
        "baseline_semantic_config_fingerprint": _baseline_semantic_config_fingerprint(max_steps, baseline_overrides),
        "seed_design":        list(seed_design),
        "dataset":            dataset,
        "device":             device,
        "tokenizer_tag":      _tokenizer_tag(dataset),
        "data_seed_override": data_seed_override,
        "max_tokens":         max_tokens,
        "max_steps":          max_steps,
        **({"stop_after_steps": stop_after_steps} if stop_after_steps is not None else {}),
        "source_identities":  source_identities,
        "code_identity":      code_identity,
        "diagnostic_flags":   flags,
        "required_diagnostic_keys": required_keys,
        "min_extrapolation_points": minimum,
        "requested_extrapolation_lengths": requested,
        "mandatory_extrapolation_lengths": mandatory,
        "save_model_artifacts": _require_exact_bool(
            save_model_artifacts,
            "save_model_artifacts",
        ),
    })


def _cell_contract(
    cfg:              VFE3Config,
    dataset:          str,
    diagnostic_flags: Mapping[str, bool],

    *,
    data_seed:         int,
    device:            str                          = "cpu",
    max_tokens:        Optional[int]                  = None,
    stop_after_steps:  Optional[int]                  = None,
    cache_dir:         Optional[Path]                 = None,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
    save_model_artifacts: bool                       = True,
    required_diagnostic_keys: Optional[Sequence[str]] = None,
    min_extrapolation_points: Optional[int]           = None,
    requested_extrapolation_lengths: Optional[Sequence[int]] = None,
    mandatory_extrapolation_lengths: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    r"""Build the versioned contract that authorizes reuse of one ablation cell.

    The contract binds a cached cell to the full identity a bare ``param=value`` label omits, so
    resume fails closed after ANY of them drifts (audit 2026-07-12 PB-01): the semantic-config
    fingerprint (the imported baseline, not just the swept field), the session dataset / data seed /
    token cap / tokenizer tag (loader seams that never land in ``config.json``), the per-split corpus
    identity (byte size + streamed SHA-256, so a rebuilt or re-tokenized cache is caught even at the
    same label), the repository code identity (HEAD plus a dirty-tree fingerprint), and the requested
    diagnostic/extrapolation collections. ``cache_source_identity`` raises when a split's cache is
    absent, which the caller converts into "forbid reuse" rather than a stale hit.
    """
    stop_after_steps = _validated_stop_after_steps(stop_after_steps)
    data_seed = _require_exact_seed(data_seed, "data_seed")
    if not isinstance(device, str) or not device:
        raise ValueError("ablation cell device must be a non-empty string")
    sources = (
        _ablation_source_identities(dataset, cache_dir=cache_dir)
        if source_identities is None
        else _validated_ablation_source_identities(source_identities)
    )
    code = (
        _validated_ablation_code_identity(_git_code_identity())
        if code_identity is None
        else _validated_ablation_code_identity(code_identity)
    )
    flags = _validated_diagnostic_flags(diagnostic_flags)
    required_keys = _validated_required_diagnostic_keys(
        (sorted(_BASE_REQUIRED_DIAGNOSTIC_KEYS) if flags["collect_diagnostics"] else ())
        if required_diagnostic_keys is None else required_diagnostic_keys
    )
    minimum = _validated_min_extrapolation_points(
        (2 if flags["collect_extrapolation"] else 0)
        if min_extrapolation_points is None else min_extrapolation_points
    )
    requested_default = (
        _default_extrapolation_lengths(int(cfg.max_seq_len))
        if flags["collect_extrapolation"] else []
    )
    requested_raw = (
        requested_default if requested_extrapolation_lengths is None
        else requested_extrapolation_lengths
    )
    requested_preview = _validated_extrapolation_lengths(
        requested_raw,
        "requested_extrapolation_lengths",
    )
    requested, mandatory = _validated_extrapolation_request(
        requested_preview,
        ((max(requested_preview),) if mandatory_extrapolation_lengths is None and requested_preview
         else (() if mandatory_extrapolation_lengths is None
               else mandatory_extrapolation_lengths)),
        collect_extrapolation=flags["collect_extrapolation"],
    )
    if flags["collect_diagnostics"] != bool(required_keys):
        raise ValueError("diagnostic collection and required keys are inconsistent")
    if flags["collect_extrapolation"] != (minimum >= 2):
        raise ValueError("extrapolation collection and minimum output points are inconsistent")
    return {
        "schema_version":              _CELL_CONTRACT_SCHEMA_VERSION,
        "semantic_config_fingerprint": semantic_config_fingerprint(asdict(cfg)),
        "dataset":                     dataset,
        "device":                      device,
        "data_seed":                   data_seed,
        "max_tokens":                  int(max_tokens) if max_tokens is not None else None,
        **({"stop_after_steps": stop_after_steps} if stop_after_steps is not None else {}),
        "tokenizer_tag":               _tokenizer_tag(dataset),
        "train_source":                sources["train"],
        "validation_source":           sources["validation"],
        "code_identity":               code,
        "diagnostic_flags":            flags,
        "required_diagnostic_keys":    required_keys,
        "min_extrapolation_points":    minimum,
        "requested_extrapolation_lengths": requested,
        "mandatory_extrapolation_lengths": mandatory,
        "save_model_artifacts": _require_exact_bool(
            save_model_artifacts,
            "save_model_artifacts",
        ),
    }


def _expected_cell_contract_from_aggregation(
    result:               Mapping[str, object],
    aggregation_contract: Mapping[str, object],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Reconstruct one adjacent row's exact cell contract under the current common cohort."""
    aggregation = _validated_sweep_aggregation_contract(aggregation_contract)
    overrides = result.get("overrides")
    if not isinstance(overrides, Mapping):
        raise TypeError("ablation result overrides must be a mapping")
    seed = _require_exact_seed(result.get("seed"), "ablation result seed")
    if seed not in aggregation["seed_design"]:
        raise ValueError("ablation result seed is outside the sweep aggregation seed design")
    cfg = VFE3Config(**_cell_cfg_dict(
        dict(overrides),
        seed=seed,
        max_steps=aggregation["max_steps"],
    ))
    data_seed_override = aggregation["data_seed_override"]
    contract = _cell_contract(
        cfg,
        str(aggregation["dataset"]),
        aggregation["diagnostic_flags"],
        data_seed=(seed if data_seed_override is None else int(data_seed_override)),
        device=str(aggregation["device"]),
        max_tokens=aggregation["max_tokens"],
        stop_after_steps=aggregation.get("stop_after_steps"),
        source_identities=aggregation["source_identities"],
        code_identity=aggregation["code_identity"],
        required_diagnostic_keys=aggregation["required_diagnostic_keys"],
        min_extrapolation_points=int(aggregation["min_extrapolation_points"]),
        requested_extrapolation_lengths=aggregation["requested_extrapolation_lengths"],
        mandatory_extrapolation_lengths=aggregation["mandatory_extrapolation_lengths"],
        save_model_artifacts=_require_exact_bool(
            aggregation["save_model_artifacts"],
            "save_model_artifacts",
        ),
    )
    contract["tokenizer_tag"] = aggregation["tokenizer_tag"]
    return contract, _gauge_reporting_fields(cfg)


def _gauge_purity_summary(results: List[Dict[str, Any]]) -> Dict[str, object]:
    """Summarize row-level gauge classifications without collapsing mixed sweeps into a pure claim."""
    classifications = {
        str(result["label"]): str(result["head_mixer_compatibility"])
        for result in results
    }
    mlp_modes = {
        str(result["label"]): str(result.get("block_mlp_structural_mode", "unavailable"))
        for result in results
    }
    mlp_covariance = {
        str(result["label"]): str(result.get("block_mlp_covariance_contract", "unavailable"))
        for result in results
    }
    mlp_intertwiner_compatible = {
        str(result["label"]): bool(result.get("block_mlp_intertwiner_compatible", False))
        for result in results
    }
    return {
        "classifications_by_label": classifications,
        "block_mlp_structural_modes_by_label": mlp_modes,
        "block_mlp_covariance_contracts_by_label": mlp_covariance,
        "block_mlp_intertwiner_compatible_by_label": mlp_intertwiner_compatible,
        "contains_independent_head_nonintertwiner": any(
            value == "independent_head_nonintertwiner"
            for value in classifications.values()
        ),
        "contains_coordinate_mean_only_nonintertwiner": any(
            value == "coordinate_mean_only_nonintertwiner"
            for value in mlp_modes.values()
        ),
        "all_rows_on_gauge_pure_path": bool(results) and all(
            result.get("on_gauge_pure_path") is True for result in results
        ),
    }


def _expected_cell_contract_or_none(
    overrides: Mapping[str, object],
    dataset: str,
    diagnostic_flags: Mapping[str, bool],
    *,
    seed:              int,
    device:            str                              = "cpu",
    max_steps:         Optional[int]                  = None,
    stop_after_steps:  Optional[int]                  = None,
    max_tokens:        Optional[int]                  = None,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
    save_model_artifacts: bool                       = True,
    required_diagnostic_keys: Optional[Sequence[str]] = None,
    min_extrapolation_points: Optional[int]           = None,
    requested_extrapolation_lengths: Optional[Sequence[int]] = None,
    mandatory_extrapolation_lengths: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, object]]:
    r"""Build the reuse contract inside the per-cell failure boundary, else forbid reuse.

    Wraps both ``VFE3Config`` construction and the corpus source hashing so a rejected config or a
    missing/corrupt cache raises here rather than in the sweep loop: the exception is logged and
    ``None`` is returned, which only forbids cache reuse (``run_single`` then executes inside its own
    isolation and records the actual config/data error without aborting later cells). The data-order
    seed recorded in the contract is the effective TRAIN-loader seed -- ``DATA_SEED`` when set (shared
    across per-model seeds), else the cell seed -- matching ``run_single``'s post-build reseed.
    """
    try:
        cfg = VFE3Config(**_cell_cfg_dict(dict(overrides), seed=seed, max_steps=max_steps))
        data_seed = _require_exact_seed(
            DATA_SEED if DATA_SEED is not None else seed,
            "DATA_SEED" if DATA_SEED is not None else "seed",
        )
        return _cell_contract(
            cfg,
            dataset,
            diagnostic_flags,
            data_seed=data_seed,
            device=device,
            max_tokens=max_tokens,
            stop_after_steps=stop_after_steps,
            source_identities=source_identities,
            code_identity=code_identity,
            required_diagnostic_keys=required_diagnostic_keys,
            min_extrapolation_points=min_extrapolation_points,
            requested_extrapolation_lengths=requested_extrapolation_lengths,
            mandatory_extrapolation_lengths=mandatory_extrapolation_lengths,
            save_model_artifacts=save_model_artifacts,
        )
    except Exception as exc:                                  # unbuildable config / unhashable corpus
        logger.warning("  [contract unavailable -> reuse forbidden] %s", exc)
        return None


def _terminal_checkpoint_location(
    run_dir: Path,
    marker:  Mapping[str, object],
) -> Optional[Tuple[Path, str]]:
    """Return the owned terminal checkpoint's resolved path and run-relative path, or ``None``."""
    raw_path = marker.get("terminal_checkpoint")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    root = run_dir.resolve()
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [raw, run_dir / raw]
    for candidate in candidates:
        try:
            if candidate.is_symlink() or _path_is_junction(candidate):
                continue
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root)
            if not relative.parts or not resolved.is_file():
                continue
            if any(part in {"", ".", ".."} for part in relative.parts):
                continue
            return resolved, relative.as_posix()
        except (OSError, ValueError):
            continue
    return None


def _terminal_checkpoint_identity(
    run_dir: Path,
    marker:  Mapping[str, object],
) -> Optional[Dict[str, object]]:
    """Return the owned terminal checkpoint's recorded identity, or ``None`` if unresolvable."""
    location = _terminal_checkpoint_location(run_dir, marker)
    if location is None:
        return None
    resolved, relpath = location
    try:
        return {
            "terminal_checkpoint_relpath":    relpath,
            "terminal_checkpoint_sha256":     _sha256_file(resolved),
            "terminal_checkpoint_size_bytes": resolved.stat().st_size,
        }
    except OSError:
        return None


def _terminal_checkpoint_is_current(
    run_dir: Path,
    marker:  Mapping[str, object],
) -> bool:
    """Revalidate a published terminal checkpoint's owned location on every reuse."""
    location = _terminal_checkpoint_location(run_dir, marker)
    return location is not None and marker.get("terminal_checkpoint_relpath") == location[1]


def _model_artifacts_are_current(
    run_dir: Path,
    marker:  Mapping[str, object],

    *,
    save_model_artifacts: bool,
) -> bool:
    """Validate either the retained-model bundle or its explicit absence."""
    required = _require_exact_bool(save_model_artifacts, "save_model_artifacts")
    if marker.get("save_model_artifacts") is not required:
        return False
    if required:
        return _terminal_checkpoint_is_current(run_dir, marker)
    if marker.get("terminal_checkpoint") is not None:
        return False
    if any(marker.get(key) is not None for key in (
        "terminal_checkpoint_relpath",
        "terminal_checkpoint_sha256",
        "terminal_checkpoint_size_bytes",
    )):
        return False
    if (run_dir / "best_model.pt").exists():
        return False
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        if (checkpoint_dir.is_symlink()
                or _path_is_junction(checkpoint_dir)
                or any(checkpoint_dir.iterdir())):
            return False
    return True


def _cell_is_current(
    run_dir:           Path,
    expected_contract: Mapping[str, object],
) -> bool:
    r"""Return true only for a successful cell with an exactly matching contract.

    Fails closed on anything short of a completed success bound to ``expected_contract``: a legacy
    directory carrying only the old ``ablation_result.json`` success marker and no
    ``cell_contract.json`` re-runs, as does a marker that is unreadable, non-mapping, failed, or
    lacks a finite terminal validation PPL. The published contract must parse, be a mapping, carry
    the current schema version, and equal the freshly rebuilt ``expected_contract`` field for field
    (ordinary nested mapping equality) -- any code, corpus, tokenizer, dataset, seed, token-cap,
    semantic-config, or diagnostic-flag drift leaves the two unequal and forbids reuse.
    """
    try:
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    except Exception:                                        # no/unreadable success marker -> re-run
        return False
    if not isinstance(marker, Mapping):                      # parseable but not an object -> re-run
        return False
    if marker.get("status") != "success" or marker.get("error_kind") is not None:
        return False
    try:
        expected_diagnostic_flags = _validated_diagnostic_flags(
            expected_contract["diagnostic_flags"]
        )
        marker_diagnostic_flags = _validated_diagnostic_flags({
            key: marker[key]
            for key in _SWEEP_DIAGNOSTIC_FLAG_KEYS
        })
    except (KeyError, TypeError, ValueError):
        return False
    if marker_diagnostic_flags != expected_diagnostic_flags:
        return False
    expected_fingerprint = semantic_config_fingerprint(dict(expected_contract))
    if marker.get("cell_contract_fingerprint") != expected_fingerprint:
        return False
    try:
        terminal_ppl = float(marker["final_val_ppl"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(terminal_ppl):
        return False
    if not _requested_outputs_are_complete(
        marker,
        required_diagnostic_keys=expected_contract.get("required_diagnostic_keys", ()),
        min_extrapolation_points=expected_contract.get("min_extrapolation_points", 0),
        requested_extrapolation_lengths=expected_contract.get(
            "requested_extrapolation_lengths", ()),
        mandatory_extrapolation_lengths=expected_contract.get(
            "mandatory_extrapolation_lengths", ()),
    ):
        return False
    try:
        save_model_artifacts = _require_exact_bool(
            expected_contract["save_model_artifacts"],
            "save_model_artifacts",
        )
    except (KeyError, TypeError, ValueError):
        return False
    if not _model_artifacts_are_current(
        run_dir,
        marker,
        save_model_artifacts=save_model_artifacts,
    ):
        return False
    contract_path = run_dir / "cell_contract.json"
    if not contract_path.exists():                           # legacy dir / never-published contract
        return False
    try:
        loaded = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:                                        # truncated / corrupt contract -> re-run
        return False
    if not isinstance(loaded, Mapping):
        return False
    if loaded.get("schema_version") != _CELL_CONTRACT_SCHEMA_VERSION:
        return False
    return (dict(loaded) == dict(expected_contract)
            and semantic_config_fingerprint(dict(loaded)) == expected_fingerprint)


def _paired_token_artifact_is_current(run_dir: Path, *, required: bool) -> bool:
    r"""Return true only when the requested paired-token artifact is present and matches its marker.

    The cell contract's sorted ``diagnostic_flags`` binds the REQUEST (was paired_token_bootstrap
    asked for); this post-contract validator checks the requested ARTIFACT's presence and tensor
    schema so a missing, nonfinite, or schema-drifted ``val_token_nats.pt`` forces a re-run rather
    than a cache hit. When the sweep did not request the artifact (``required`` false) there is
    nothing to check, so it returns true immediately. Otherwise the completion marker must record a
    real ``paired_token_bootstrap`` true, ``val_token_nats_path == "val_token_nats.pt"``, and the
    SHA-256 / tensor length / dtype fields, the file must exist, and a ``weights_only`` safe load must
    yield a finite, nonempty, one-dimensional tensor whose numel and string dtype equal the marker.
    Any missing, malformed, or nonexact field fails closed.
    """
    if not required:
        return True
    try:
        marker = json.loads((run_dir / "ablation_result.json").read_text(encoding="utf-8"))
    except Exception:                                        # no/unreadable marker -> re-run
        return False
    if not isinstance(marker, Mapping):
        return False
    if marker.get("paired_token_bootstrap") is not True:
        return False
    if marker.get("val_token_nats_path") != "val_token_nats.pt":
        return False
    expected_sha   = marker.get("val_token_nats_sha256")
    expected_numel = marker.get("val_token_nats_numel")
    expected_dtype = marker.get("val_token_nats_dtype")
    if not (isinstance(expected_sha, str) and expected_sha):
        return False
    path = run_dir / "val_token_nats.pt"
    if not path.is_file():
        return False
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.ndim != 1 or tensor.numel() == 0:
        return False
    if int(tensor.numel()) != expected_numel or str(tensor.dtype) != expected_dtype:
        return False
    return bool(torch.isfinite(tensor).all())


def _sanitize(label: str) -> str:
    r"""A filesystem-safe single path component (no separators, parent tokens, or drive colon).

    The char-replace is lossy ('a=b', 'a b', 'a/b' all map to 'a_b'), so a stable short hash of
    the RAW label is appended: distinct labels get distinct run dirs, while the map stays
    deterministic in the label so the resume [CACHED] path finds the same dir on re-run.
    """
    return filesystem_slug(label)


def _write_sweep_csv(sweep_dir: Path, results: List[Dict[str, Any]]) -> None:
    r"""Atomically replace ``sweep_results.csv`` with one complete fixed-column frame."""
    final_path = sweep_dir / "sweep_results.csv"
    with _unique_sibling_temp(final_path) as temporary_path:
        with open(temporary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for result in results:
                writer.writerow({key: result.get(key, "") for key in _CSV_COLUMNS})
        _atomic_replace(final_path, temporary_path)


def _collect_sweep_results(
    sweep_dir: Path,

    *,
    aggregation_contract: Optional[Mapping[str, object]] = None,
) -> List[Dict[str, Any]]:
    r"""Collect only successful cells whose full contracts belong to one invocation cohort.

    Labels and swept values may differ. Every row is nevertheless reconstructed from the cohort's
    common code, corpus, tokenizer, token budget, diagnostic requests, and baseline plus that row's
    own overrides and seed. A base label is retained only with exactly one admitted row for every
    seed in the cohort's declared panel. Missing, malformed, fingerprint-mismatched, adjacent, or
    seed-incomplete contracts are excluded before CSV publication, plotting, or winner selection.
    """
    try:
        meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
    except Exception:
        meta = None
    if aggregation_contract is None:
        try:
            aggregation_contract = meta["aggregation_contract"]
        except Exception:
            return []
    try:
        aggregation = _validated_sweep_aggregation_contract(aggregation_contract)
    except (TypeError, ValueError):
        return []
    try:
        parameter_records = _parameter_records_by_label(
            meta.get("parameter_match") if isinstance(meta, Mapping) else None
        )
    except (TypeError, ValueError):
        return []

    results: List[Dict[str, Any]] = []
    for marker in sorted(sweep_dir.glob("*/ablation_result.json")):
        run_dir = marker.parent
        if not _cell_dir_is_owned(sweep_dir, run_dir):
            continue
        try:
            result = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:                                       # unreadable marker -> skip
            continue
        if not isinstance(result, Mapping):
            continue
        label = result.get("label")
        if (not isinstance(label, str) or not label
                or result.get("sweep") != sweep_dir.name
                or run_dir.name != _sanitize(label)):
            continue
        seed = result.get("seed")
        if (type(seed) is not int
                or not _ablation_cell_owner_is_exact(
                    run_dir, sweep_dir.name, label, seed)):
            continue
        if parameter_records is not None:
            parameter_record = parameter_records.get(_base_label(label))
            if (parameter_record is None
                    or not _parameter_budget_record_matches(result, parameter_record)):
                continue
        if result.get("status") != "success" or result.get("error_kind") is not None:
            continue
        try:
            terminal_ppl = float(result["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(terminal_ppl):
            continue
        try:
            expected_contract, gauge_fields = _expected_cell_contract_from_aggregation(
                result,
                aggregation,
            )
        except Exception:
            continue
        if not _cell_is_current(run_dir, expected_contract):
            continue
        if not _paired_token_artifact_is_current(
            run_dir,
            required=bool(aggregation["diagnostic_flags"]["paired_token_bootstrap"]),
        ):
            continue
        if (
            not _execution_derived_reporting_fields_are_valid(result)
            or any(
                key not in _EXECUTION_DERIVED_REPORT_FIELDS and result.get(key) != value
                for key, value in gauge_fields.items()
            )
        ):
            continue
        results.append(dict(result))

    required_seeds = set(aggregation["seed_design"])
    panels: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        panels.setdefault(_base_label(str(result["label"])), []).append(result)
    complete_bases = {
        base
        for base, members in panels.items()
        if (
            len(members) == len(required_seeds)
            and {member["seed"] for member in members} == required_seeds
        )
    }
    return [
        result
        for result in results
        if _base_label(str(result["label"])) in complete_bases
    ]


def _path_is_junction(path: Path) -> bool:
    """Return whether ``path`` is any Windows reparse point, including pre-3.12 junctions."""
    probe = getattr(os.path, "isjunction", None)
    if probe is not None and probe(path):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(reparse_flag and attributes & reparse_flag)


def _cell_dir_is_owned(sweep_dir: Path, run_dir: Path) -> bool:
    """Return true only for a real direct child directory that cannot redirect cleanup elsewhere."""
    try:
        return bool(
            run_dir.parent == sweep_dir
            and run_dir.is_dir()
            and not run_dir.is_symlink()
            and not _path_is_junction(run_dir)
            and run_dir.resolve().parent == sweep_dir.resolve()
        )
    except OSError:
        return False


def _prepare_owned_output_child(root: Path, name: str, *, role: str) -> Path:
    """Create or validate one real direct-child directory under an explicit output root."""
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError(f"{role} name must be one non-empty path component")
    component = Path(name)
    if component.name != name or component.is_absolute() or component.drive:
        raise ValueError(f"{role} name must be one safe path component: {name!r}")

    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"{role} output root is not a directory: {root}")
    child = root / name
    if child.is_symlink() or _path_is_junction(child):
        raise ValueError(f"{role} directory may not be a symlink, junction, or reparse point")
    if child.exists() and not child.is_dir():
        raise ValueError(f"{role} path exists but is not a directory: {child}")
    child.mkdir(parents=False, exist_ok=True)
    return child


def _ablation_cell_owner_payload(
    sweep_name: str,
    label:      str,
    seed:       int,
) -> Dict[str, object]:
    """Return the exact versioned identity that authorizes destructive cell cleanup."""
    if type(sweep_name) is not str or not sweep_name:
        raise ValueError("ablation cell ownership sweep must be a nonempty string")
    if type(label) is not str or not label:
        raise ValueError("ablation cell ownership label must be a nonempty string")
    return {
        "schema_version": _ABLATION_CELL_OWNER_SCHEMA_VERSION,
        "sweep":          sweep_name,
        "label":          label,
        "seed":           _require_exact_seed(seed, "ablation cell ownership seed"),
    }


def _read_regular_json_mapping(path: Path, *, role: str) -> Dict[str, object]:
    """Read one direct regular, non-reparse JSON object or fail without following a redirect."""
    if path.is_symlink() or _path_is_junction(path) or not path.is_file():
        raise ValueError(f"{role} must be a regular non-reparse file")
    try:
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ValueError(f"{role} resolves outside its cell directory")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must contain a JSON object")
    return dict(value)


def _ablation_cell_owner_is_exact(
    run_dir:    Path,
    sweep_name: str,
    label:      str,
    seed:       int,
) -> bool:
    """Return true only when the regular ownership sentinel has the exact requested identity."""
    try:
        expected = _ablation_cell_owner_payload(sweep_name, label, seed)
        owner = _read_regular_json_mapping(
            run_dir / _ABLATION_CELL_OWNER_FILENAME,
            role="ablation cell ownership sentinel",
        )
    except ValueError:
        return False
    return bool(
        set(owner) == set(expected)
        and type(owner.get("schema_version")) is int
        and type(owner.get("sweep")) is str
        and type(owner.get("label")) is str
        and type(owner.get("seed")) is int
        and owner == expected
    )


def _authorize_ablation_cell_cleanup(
    run_dir:    Path,
    sweep_name: str,
    label:      str,
    seed:       int,
) -> Path:
    r"""Validate existing ownership or promote one exact legacy marker before any overwrite.

    An empty cell receives a fresh sentinel. A nonempty cell must already carry that exact sentinel,
    except that a pre-sentinel generation may be promoted once from its regular result marker when
    label, seed, and any present sweep field match this invocation exactly. Malformed or mismatched
    state never authorizes cleanup.
    """
    expected = _ablation_cell_owner_payload(sweep_name, label, seed)
    owner_path = run_dir / _ABLATION_CELL_OWNER_FILENAME
    children = list(run_dir.iterdir())
    if not children:
        _write_json_atomic(owner_path, expected)
        return owner_path

    owner_present = (
        owner_path.exists()
        or owner_path.is_symlink()
        or _path_is_junction(owner_path)
    )
    if owner_present:
        if not _ablation_cell_owner_is_exact(run_dir, sweep_name, label, seed):
            raise ValueError(
                "ablation cell ownership sentinel does not match the requested sweep, label, and seed"
            )
        return owner_path

    marker_path = run_dir / "ablation_result.json"
    try:
        legacy = _read_regular_json_mapping(
            marker_path,
            role="legacy ablation cell ownership marker",
        )
    except ValueError as exc:
        raise ValueError(
            "nonempty ablation cell has no valid ownership sentinel or promotable legacy marker"
        ) from exc
    if (
        legacy.get("label") != label
        or type(legacy.get("seed")) is not int
        or legacy.get("seed") != seed
        or ("sweep" in legacy and legacy.get("sweep") != sweep_name)
    ):
        raise ValueError(
            "legacy ablation cell ownership marker does not match the requested sweep, label, and seed"
        )
    _write_json_atomic(owner_path, expected)
    return owner_path


def _start_owned_cell_generation(
    sweep_dir: Path,
    run_dir:   Path,

    *,
    sweep_name: str,
    label:      str,
    seed:       int,
) -> None:
    r"""Invalidate the old marker, then empty only this owned cell directory before recomputation.

    A matching resume never calls this helper. Before any existing file is overwritten, a versioned
    sentinel must bind the directory to this exact sweep, label, and seed; one regular legacy result
    marker may establish that ownership once. Writing the ``running`` marker then makes a partial
    cleanup or process crash non-reportable. Every other direct child except the sentinel is removed,
    preventing optional artifacts, figures, and checkpoints from a prior owned generation from
    surviving beside the new one. Symlinks and junctions are removed as links and never traversed.
    """
    if run_dir.parent != sweep_dir:
        raise ValueError("ablation cell directory must be a direct child of its sweep directory")
    if run_dir.exists() and (run_dir.is_symlink() or _path_is_junction(run_dir)):
        raise ValueError("ablation cell directory may not be a symlink or junction")
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError("ablation cell path exists but is not a directory")
    run_dir.mkdir(parents=False, exist_ok=True)

    owner_path = _authorize_ablation_cell_cleanup(
        run_dir,
        sweep_name,
        label,
        seed,
    )
    marker = run_dir / "ablation_result.json"
    _write_json_atomic(marker, {
        "status": "running",
        "sweep":  sweep_name,
        "label":  label,
        "seed":   seed,
    })
    for child in list(run_dir.iterdir()):
        if child in {marker, owner_path}:
            continue
        if child.is_symlink():
            child.unlink()
        elif _path_is_junction(child):
            child.rmdir()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _invalidate_code_drifted_cells(
    sweep_dir: Path,
    cells:     List[Tuple[str, Dict[str, Any], int]],
    error:     str,
) -> None:
    """Atomically downgrade every successful current-invocation marker after terminal code drift."""
    for label, _overrides, _seed in cells:
        run_dir = sweep_dir / _sanitize(label)
        marker_path = run_dir / "ablation_result.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(marker, Mapping) or marker.get("status") != "success":
            continue
        failed = dict(marker)
        failed.update({
            "status":                    "failed",
            "error_kind":                "code_identity_drift",
            "error":                     error,
            "cell_contract_fingerprint": None,
        })
        _write_json_atomic(marker_path, failed)
        try:
            (run_dir / "cell_contract.json").unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("  [%s] failed to remove drifted cell contract: %s", label, exc)


def _qkv_code_identity() -> Dict[str, object]:
    """Require the actual imported source generation for this experiment only."""
    package = _package_code_identity()
    runner = _ablation_runner_source_sha256()
    if package != _PROCESS_PACKAGE_CODE_IDENTITY or runner != _PROCESS_ABLATION_RUNNER_SHA256:
        raise RuntimeError("QKV experiment source changed after import; restart this Python process")
    return {"package_code_sha256": package, "ablation_runner_sha256": runner}


def _qkv_schedule(
    runs:          Sequence[Tuple[str, Dict[str, Any]]],
    seeds:         Sequence[int],
    schedule_seed: int,
) -> List[Tuple[str, Dict[str, Any], int]]:
    """Permute arms within complete seed blocks without consuming training RNG."""
    rng = random.Random(_require_exact_seed(schedule_seed, "schedule_seed"))
    cells = []
    for seed in seeds:
        block = list(runs)
        rng.shuffle(block)
        cells.extend((f"{label}__s{seed}", overrides, seed) for label, overrides in block)
    return cells


def run_sweep(
    sweep_name:  str,
    output_dir:  Path,

    *,
    dataset:     str,
    device:      torch.device,
    seed:        int,
    resume:      bool,
    save_model_artifacts: bool          = True,
    max_tokens:  Optional[int] = None,
    max_steps:   Optional[int] = None,
    stop_after_steps: Optional[int] = None,
    collected_out: Optional[List[Dict[str, Any]]] = None,   # out-param: the collected cohort, even when incomplete
) -> List[Dict[str, Any]]:
    r"""Run every cell of one sweep; per-cell failures are isolated so the sweep completes.

    Returns the collected cohort only for a complete sweep and ``[]`` otherwise. ``collected_out``
    receives the collected cohort in either case, for callers that rank partial results.
    """
    stop_after_steps = _validated_stop_after_steps(stop_after_steps)
    sweep_name = _validated_sweep_name(sweep_name)
    save_model_artifacts = _require_exact_bool(
        save_model_artifacts,
        "save_model_artifacts",
    )
    sweep = SWEEPS[sweep_name]
    qkv_experiment = sweep_name == "learned_qkv_factorial"
    
    code_identity_snapshot = _qkv_code_identity if qkv_experiment else _git_code_identity
    diagnostic_flags = _sweep_diagnostic_request(sweep)
    cell_seeds = _validated_sweep_seeds(sweep, seed)
    data_seed_override = _validated_data_seed_override()
    sweep_scope = _sweep_output_scope(sweep_name)
    sweep_dir = _prepare_owned_output_child(output_dir, sweep_scope, role="ablation sweep")
    try:
        parameter_match = (
            _parameter_match_selection(sweep_name)
            if "parameter_grid" in sweep else None
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        error = f"parameter-matched sweep setup failed: {exc}"
        logger.error("  [%s]", error)
        _write_json_atomic(sweep_dir / "sweep_meta.json", {
            "sweep_name":              sweep_name,
            "description":             sweep["description"],
            "n_runs":                  0,
            "n_successful_requested":  0,
            "failed_requested_labels": [],
            "dataset":                 dataset,
            "device":                  str(device),
            "seed":                    seed,
            "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
            "status":                  "incomplete",
            "error":                   error,
            "aggregation_contract":    None,
            "gauge_purity":            _gauge_purity_summary([]),
            "parameter_match": {
                "target_n_params": CONFIG.get("target_n_params"),
                "max_param_relative_deviation": CONFIG.get(
                    "max_param_relative_deviation"
                ),
                "match_by":       sweep.get("match_by"),
                "parameter_grid": _jsonable(sweep.get("parameter_grid")),
                "selected":       [],
                "rejected":       [],
                "error":          str(exc),
            },
        })
        _write_sweep_csv(sweep_dir, [])
        return []
    runs = (
        [
            (str(record["label"]), dict(record["overrides"]))
            for record in parameter_match["selected"]
        ]
        if parameter_match is not None else make_run_overrides(sweep_name)
    )
    run_labels = [label for label, _overrides in runs]
    if any(not isinstance(label, str) or not label for label in run_labels):
        raise ValueError(f"sweep {sweep_name!r} has a non-string or empty label")
    duplicate_labels = sorted({label for label in run_labels if run_labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(
            f"sweep {sweep_name!r} has duplicate expanded label(s): {duplicate_labels}"
        )
    collect_diagnostics = diagnostic_flags["collect_diagnostics"]
    collect_extrapolation = diagnostic_flags["collect_extrapolation"]
    paired_token_bootstrap = diagnostic_flags["paired_token_bootstrap"]
    (
        required_diagnostic_keys,
        min_extrapolation_points,
        requested_extrapolation_lengths,
        mandatory_extrapolation_lengths,
    ) = _sweep_required_outputs(sweep, diagnostic_flags)
    # Multi-seed (I1/EXP-1): a sweep may declare ``seeds`` to replicate every cell across seeds for an
    # across-seed error bar. Each (cell, seed) gets its own ``{label}__s{seed}`` run dir and result row
    # (the seed also lives in the existing ``seed`` column), so the across-seed aggregate is a plain
    # group-by on the base label. A sweep WITHOUT ``seeds`` keeps the single-seed label/run-dir exactly,
    # so every existing sweep is byte-identical.
    multiseed = "seeds" in sweep
    cells = [((f"{label}__s{s}" if multiseed else label), overrides, s)
             for (label, overrides) in runs for s in cell_seeds]
    if "schedule_seed" in sweep:
        cells = _qkv_schedule(runs, cell_seeds, sweep["schedule_seed"])
    parameter_record_by_label: Dict[str, Mapping[str, object]] = {}
    if parameter_match is not None:
        base_records = {
            str(record["label"]): record
            for record in parameter_match["selected"]
        }
        parameter_record_by_label = {
            (f"{label}__s{cell_seed}" if multiseed else label): record
            for label, record in base_records.items()
            for cell_seed in cell_seeds
        }

    report_metadata = {
        "stop_after_steps":      stop_after_steps,
        "paired_token_bootstrap": paired_token_bootstrap,
        "required_diagnostic_keys": required_diagnostic_keys,
        "min_extrapolation_points": min_extrapolation_points,
        "requested_extrapolation_lengths": requested_extrapolation_lengths,
        "mandatory_extrapolation_lengths": mandatory_extrapolation_lengths,
        "forest_baseline_label":  sweep.get("forest_baseline_label"),
        "grid_x":                 sweep.get("grid_x"),
        "grid_y":                 sweep.get("grid_y"),
        "grid_x_values":          sweep.get("grid_x_values"),
        "grid_y_values":          sweep.get("grid_y_values"),
        "grid_baseline":          (list(sweep["grid_baseline"])
                                   if sweep.get("grid_baseline") is not None else None),
        "parameter_match":        _jsonable(parameter_match) if parameter_match is not None else None,
        "save_model_artifacts":   save_model_artifacts,
    }
    running_meta = {
        "sweep_name":              sweep_name,
        "description":             sweep["description"],
        "n_runs":                  len(cells),
        "n_successful_requested":  0,
        "failed_requested_labels": [label for label, _overrides, _seed in cells],
        "dataset":                 dataset,
        "device":                  str(device),
        "seed":                    (cell_seeds if multiseed else seed),
        "timestamp":               time.strftime("%Y-%m-%d %H:%M:%S"),
        "status":                  "running",
        "error":                   None,
        "aggregation_contract":    None,
        "gauge_purity":            _gauge_purity_summary([]),
        **report_metadata,
    }
    # Invalidate any prior complete view before even the invocation identity probes run. Cell
    # artifacts remain available for exact-contract resume, but a crash cannot leave the old
    # metadata/CSV pair externally visible as the outcome of this new invocation.
    _write_json_atomic(sweep_dir / "sweep_meta.json", running_meta)
    _write_sweep_csv(sweep_dir, [])

    try:
        invocation_code_identity = _validated_ablation_code_identity(code_identity_snapshot())
    except Exception as exc:
        error = f"pre-sweep code identity unavailable: {exc}"
        logger.error("  [%s -> sweep aborted]", error)
        running_meta.update({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status":    "incomplete",
            "error":     error,
        })
        _write_json_atomic(sweep_dir / "sweep_meta.json", running_meta)
        return []
    contract_code_identity: Mapping[str, object] = invocation_code_identity

    print(f"\n{'=' * 70}\nSWEEP: {sweep_name} ({len(cells)} runs"
          f"{f' = {len(runs)} cells x {len(cell_seeds)} seeds' if multiseed else ''})"
          f"\n  {sweep['description']}"
          f"\n  Output: {sweep_dir}{'  [resume ON]' if resume else ''}\n{'=' * 70}")

    try:
        sweep_source_identities: Optional[Dict[str, Dict[str, object]]] = (
            _ablation_source_identities(dataset))
    except Exception as exc:                                  # missing/corrupt/unreadable source
        logger.warning("  [pre-sweep corpus identity unavailable -> reuse forbidden] %s", exc)
        sweep_source_identities = None
    # An explicit empty mapping tells the contract builder that the invocation-owned snapshot failed.
    # It must not fall back to per-cell filesystem hashing; every cell then fails closed at publication.
    contract_source_identities: Mapping[str, object] = (
        sweep_source_identities if sweep_source_identities is not None else {})
    try:
        aggregation_contract: Optional[Dict[str, object]] = (
            _sweep_aggregation_contract(
                dataset,
                diagnostic_flags,
                data_seed_override=data_seed_override,
                max_tokens=max_tokens,
                max_steps=max_steps,
                stop_after_steps=stop_after_steps,
                seed_design=cell_seeds,
                source_identities=contract_source_identities,
                code_identity=contract_code_identity,
                save_model_artifacts=save_model_artifacts,
                device=str(device),
                required_diagnostic_keys=required_diagnostic_keys,
                min_extrapolation_points=min_extrapolation_points,
                requested_extrapolation_lengths=requested_extrapolation_lengths,
                mandatory_extrapolation_lengths=mandatory_extrapolation_lengths,
                
            )
            if sweep_source_identities is not None else None
        )
    except Exception as exc:
        logger.warning("  [aggregation contract unavailable -> sweep incomplete] %s", exc)
        aggregation_contract = None
    running_meta["aggregation_contract"] = aggregation_contract
    
    _write_json_atomic(sweep_dir / "sweep_meta.json", running_meta)
    results: List[Dict[str, Any]] = []
    for i, (label, overrides, cell_seed) in enumerate(cells):
        run_dir = sweep_dir / _sanitize(label)
        marker = run_dir / "ablation_result.json"
        contract_path = run_dir / "cell_contract.json"

        # Reuse is authorized only by a versioned contract that binds this cell to its code identity,
        # per-split corpus hashes, and semantic config (audit 2026-07-12 PB-01). The expected contract
        # is built inside the per-cell failure boundary for every recomputation, not only resume.
        # An unbuildable config or unhashable corpus returns None, which forbids both reuse and later
        # success publication because the generation active at training start is unknown.
        expected_contract = _expected_cell_contract_or_none(
            overrides, dataset, diagnostic_flags, seed=cell_seed,
            device=str(device),
            max_steps=max_steps, max_tokens=max_tokens,
            stop_after_steps=stop_after_steps,
            source_identities=contract_source_identities,
            code_identity=contract_code_identity,
            save_model_artifacts=save_model_artifacts,
            required_diagnostic_keys=required_diagnostic_keys,
            min_extrapolation_points=min_extrapolation_points,
            requested_extrapolation_lengths=requested_extrapolation_lengths,
            mandatory_extrapolation_lengths=mandatory_extrapolation_lengths)
        expected_gauge_fields: Dict[str, object] = {}
        try:
            if aggregation_contract is None:
                raise ValueError("invocation aggregation contract is unavailable")
            cohort_cell_contract, expected_gauge_fields = _expected_cell_contract_from_aggregation(
                {"overrides": overrides, "seed": cell_seed},
                aggregation_contract,
            )
            if expected_contract is None or cohort_cell_contract != expected_contract:
                raise ValueError("cell contract does not match the invocation aggregation contract")
        except Exception as exc:
            logger.warning("  [contract unavailable -> reuse forbidden] %s", exc)
            expected_contract = None
        if resume and marker.exists():
            try:
                cached_result = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                cached_result = None
            parameter_record = parameter_record_by_label.get(label)
            parameter_budget_current = (
                parameter_record is None
                or (
                    isinstance(cached_result, Mapping)
                    and _parameter_budget_record_matches(cached_result, parameter_record)
                )
            )
            # The contract binds the request (diagnostic_flags carries paired_token_bootstrap); a
            # separate post-contract validator checks the requested artifact's presence and schema, so a
            # missing or drifted val_token_nats.pt forbids reuse even when the contract still matches.
            if (isinstance(cached_result, Mapping)
                    and parameter_budget_current
                    and _cell_dir_is_owned(sweep_dir, run_dir)
                    and _ablation_cell_owner_is_exact(
                        run_dir, sweep_scope, label, cell_seed)
                    and expected_contract is not None
                    and _cell_is_current(run_dir, expected_contract)
                    and _paired_token_artifact_is_current(run_dir, required=paired_token_bootstrap)):
                print(f"\n--- {i + 1}/{len(cells)}: {label}  [CACHED] ---")
                cached_result = dict(cached_result)
                _merge_gauge_reporting_defaults(cached_result, expected_gauge_fields)
                cached_result["sweep"] = sweep_scope
                cached_result["label"] = label
                cached_result["seed"] = cell_seed
                cached_result["overrides"] = _jsonable(overrides)
                parameter_record = parameter_record_by_label.get(label)
                if parameter_record is not None:
                    cached_result.update({
                        "target_n_params":          parameter_record["target_n_params"],
                        "param_difference":         parameter_record["param_difference"],
                        "param_relative_deviation": parameter_record["param_relative_deviation"],
                    })
                _write_json_atomic(marker, cached_result)
                results.append(cached_result)
                continue
            print(f"\n--- {i + 1}/{len(cells)}: {label}  [contract changed -> re-running] ---")
        else:
            print(f"\n--- {i + 1}/{len(cells)}: {label} ---")
        t0 = time.perf_counter()
        generation_started = False
        try:
            _start_owned_cell_generation(
                sweep_dir,
                run_dir,
                sweep_name=sweep_scope,
                label=label,
                seed=cell_seed,
            )
        except Exception as exc:
            logger.exception("sweep %s / %s setup failed", sweep_name, label)
            result = {"label": label, "error_kind": "setup", "error": str(exc),
                      "primary_val_ppl": float("inf"), "seed": int(cell_seed),
                      "overrides": _jsonable(overrides)}
        else:
            generation_started = True
            try:
                result = run_single(label, overrides, run_dir, dataset=dataset, device=device,
                                    seed=cell_seed, collect_diagnostics=collect_diagnostics,
                                    collect_extrapolation=collect_extrapolation,
                                    paired_token_bootstrap=paired_token_bootstrap,
                                    save_model_artifacts=save_model_artifacts,
                                    extrapolation_lengths=requested_extrapolation_lengths,
                                    max_tokens=max_tokens, max_steps=max_steps,
                                    stop_after_steps=stop_after_steps)
            except Exception as exc:                         # one training crash must not kill the sweep
                logger.exception("sweep %s / %s training crashed", sweep_name, label)
                result = {"label": label, "error_kind": "train", "error": str(exc),
                          "primary_val_ppl": float("inf"), "seed": int(cell_seed),
                          "overrides": _jsonable(overrides)}
        finally:
            _cleanup()

        loaded_sources_raw = result.pop("_loaded_data_sources", None)
        try:
            loaded_source_identities = _validated_ablation_source_identities(loaded_sources_raw)
        except (TypeError, ValueError):
            loaded_source_identities = None
        result.setdefault("error_kind", None)
        result["label"] = label
        result["seed"] = int(cell_seed)
        result["overrides"] = _jsonable(overrides)
        result["save_model_artifacts"] = save_model_artifacts
        parameter_record = parameter_record_by_label.get(label)
        if parameter_record is not None:
            expected_n_params = int(parameter_record["n_params"])
            if result.get("error_kind") is None and result.get("n_params") != expected_n_params:
                result["error_kind"] = "parameter_count_drift"
                result["error"] = (
                    f"selected realized count was {expected_n_params}, but run_single built "
                    f"{result.get('n_params')!r} parameters"
                )
            result.update({
                "target_n_params":          parameter_record["target_n_params"],
                "param_difference":         parameter_record["param_difference"],
                "param_relative_deviation": parameter_record["param_relative_deviation"],
            })
        reporting_fields = expected_gauge_fields or {
            "head_mixer_compatibility":         "unavailable",
            "head_mixer_gauge_compatible":      False,
            "block_mlp_structural_mode":        "unavailable",
            "block_mlp_covariance_contract":    "unavailable",
            "block_mlp_intertwiner_compatible": False,
            "on_gauge_pure_path":               False,
            "on_causal_lm_path":                False,
            "transport_exactness_status":        "unknown",
            "on_theory_pure_path":               False,
        }
        if result.get("error_kind") is None:
            _merge_gauge_reporting_defaults(result, reporting_fields)
        else:
            result.update(reporting_fields)
        result["collect_diagnostics"] = diagnostic_flags["collect_diagnostics"]
        result["collect_extrapolation"] = diagnostic_flags["collect_extrapolation"]
        # The request flag always lands in the marker; the artifact identity fields default to null on
        # any path (crash, or a sweep that did not request the artifact) so the CSV/adapters stay whole.
        result["paired_token_bootstrap"] = diagnostic_flags["paired_token_bootstrap"]
        result.setdefault("val_token_nats_path", None)
        result.setdefault("val_token_nats_sha256", None)
        result.setdefault("val_token_nats_size_bytes", None)
        result.setdefault("val_token_nats_numel", None)
        result.setdefault("val_token_nats_dtype", None)
        try:
            terminal_ppl = float(result["final_val_ppl"])
        except (KeyError, TypeError, ValueError):
            terminal_ppl = float("inf")
        result["final_val_ppl"] = terminal_ppl
        successful = result["error_kind"] is None and math.isfinite(terminal_ppl)

        # Requested collections are terminal artifacts. Every declared diagnostic must be finite,
        # and extrapolation must contain the contracted number of distinct finite sequence lengths.
        if successful and not _requested_outputs_are_complete(
                result,
                required_diagnostic_keys=required_diagnostic_keys,
                min_extrapolation_points=min_extrapolation_points,
                requested_extrapolation_lengths=requested_extrapolation_lengths,
                mandatory_extrapolation_lengths=mandatory_extrapolation_lengths):
            logger.warning("  [%s] requested outputs incomplete -> cell marked failed", label)
            successful = False

        # A retained terminal checkpoint's path, size, and SHA-256 are recorded; reuse revalidates its
        # location. Lightweight optimizer cells instead bind the explicit absence of both best weights
        # and checkpoint files.
        if successful and save_model_artifacts:
            checkpoint_identity = _terminal_checkpoint_identity(run_dir, result)
            if checkpoint_identity is None:
                logger.warning("  [%s] terminal checkpoint is missing or unowned -> cell marked failed", label)
                successful = False
            else:
                result.update(checkpoint_identity)

        # A successful recomputation is cached only when its contract can be rebuilt now. Never
        # publish the pre-run resume contract: code or corpus identity can drift while training. If
        # a pre-run contract was available, require exact agreement with the post-run rebuild;
        # otherwise convert the cell to a failed result rather than mix two generations.
        contract = None
        if successful:
            if loaded_source_identities is None:
                logger.warning(
                    "  [%s] loaded corpus identity unavailable -> cell marked failed", label)
                successful = False
            else:
                post_run_contract = _expected_cell_contract_or_none(
                    overrides, dataset, diagnostic_flags, seed=cell_seed,
                    device=str(device),
                    max_steps=max_steps, max_tokens=max_tokens,
                    stop_after_steps=stop_after_steps,
                    source_identities=loaded_source_identities,
                    code_identity=contract_code_identity,
                    save_model_artifacts=save_model_artifacts,
                    required_diagnostic_keys=required_diagnostic_keys,
                    min_extrapolation_points=min_extrapolation_points,
                    requested_extrapolation_lengths=requested_extrapolation_lengths,
                    mandatory_extrapolation_lengths=mandatory_extrapolation_lengths)
                if expected_contract is None or post_run_contract is None:
                    successful = False
                elif dict(post_run_contract) != dict(expected_contract):
                    logger.warning(
                        "  [%s] contract drifted during recomputation -> cell marked failed", label)
                    successful = False
                else:
                    contract = post_run_contract
        if successful and not _model_artifacts_are_current(
                run_dir,
                result,
                save_model_artifacts=save_model_artifacts):
            logger.warning(
                "  [%s] model-artifact policy changed before publication -> cell marked failed",
                label,
            )
            successful = False
            contract = None
        result["status"] = "success" if successful else "failed"
        result["cell_contract_fingerprint"] = (
            semantic_config_fingerprint(contract)
            if successful and contract is not None else None
        )
        result["sweep"] = sweep_scope
        result["wall_time_s"] = time.perf_counter() - t0

        # Publish the contract atomically (same-dir tmp + os.replace) BEFORE the completion marker, so
        # a reader that sees the success marker always sees a fully-written contract; a failed cell
        # publishes no contract. Neither file is written after a training exception.
        if generation_started and successful and contract is not None:
            _write_json_atomic(contract_path, contract)
        if generation_started:
            _write_json_atomic(marker, result)
        results.append(result)

        ppl = result["primary_val_ppl"]
        tag = f" [{result['error_kind'].upper()}]" if result.get("error_kind") else ""
        print(f"\n\n  -> val PPL {ppl:.3f}{tag}  ({result['wall_time_s']:.0f}s)\n")
        if i == 0 and len(cells) > 1:
            est = result["wall_time_s"] * len(cells)
            print(f"  ** ~{est / 60:.0f} min estimated for the full {len(cells)}-run sweep")

        # Keep the CSV whole AND accumulated after each cell: write the union of every persisted
        # marker (this cell, the rest of this run, and any prior run's cells) so the tacked-on
        # frame is live even mid-sweep.
        _write_sweep_csv(sweep_dir, _collect_sweep_results(
            sweep_dir,
            aggregation_contract=(aggregation_contract or {}),
        ))

    try:
        terminal_code_identity = _validated_ablation_code_identity(code_identity_snapshot())
        terminal_code_identity_error = None
    except Exception as exc:
        terminal_code_identity = None
        terminal_code_identity_error = f"terminal code identity snapshot failed: {exc}"
    if terminal_code_identity_error is not None:
        code_identity_error = terminal_code_identity_error
    else:
        code_identity_error = (
            None if terminal_code_identity == invocation_code_identity
            else "code identity drifted during the ablation invocation"
        )
    if code_identity_error is not None:
        _invalidate_code_drifted_cells(sweep_dir, cells, code_identity_error)

    # The final frame is the one compatible cohort only. A requested label counts as successful
    # exactly when its current marker, contract, optional paired artifact, and gauge disclosure all
    # passed that collector; historical survivors cannot make an incomplete invocation complete.
    union = _collect_sweep_results(
        sweep_dir,
        aggregation_contract=(aggregation_contract or {}),
    )
    collected_labels = {str(result["label"]) for result in union}
    failed_requested_labels = [
        label for label, _overrides, _cell_seed in cells
        if label not in collected_labels
    ]
    if code_identity_error is not None:
        sweep_error = code_identity_error
    elif aggregation_contract is None:
        sweep_error = "sweep aggregation contract was unavailable"
    elif failed_requested_labels:
        sweep_error = (
            f"{len(failed_requested_labels)} of {len(cells)} requested cells failed or lacked "
            "a current contract: " + ", ".join(failed_requested_labels)
        )
    else:
        sweep_error = None
    sweep_status = "complete" if sweep_error is None else "incomplete"
    gauge_purity = _gauge_purity_summary(union)

    # Publish the complete accumulated frame before the terminal metadata. A reader that observes
    # status="complete" therefore cannot see a truncated or previous-generation CSV.
    _write_sweep_csv(sweep_dir, union)
    _write_json_atomic(sweep_dir / "sweep_meta.json", {
        "sweep_name":  sweep_name,
        "description": sweep["description"],
        "n_runs":      len(cells),
        "n_successful_requested": len(cells) - len(failed_requested_labels),
        "failed_requested_labels": failed_requested_labels,
        "dataset":     dataset,
        "device":      str(device),
        "seed":        (cell_seeds if multiseed else seed),
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "status":      sweep_status,
        "error":       sweep_error,
        "aggregation_contract": aggregation_contract,
        "gauge_purity": gauge_purity,
        # PB-07 report metadata (null for ordinary sweeps): lets the ablation-forest / joint-LR-grid
        # adapters operate after a process restart, when only the persisted sweep view survives.
        **report_metadata,
    })

    if collected_out is not None:
        collected_out[:] = union
    if sweep_error is not None:
        print(f"\nSWEEP INCOMPLETE: {sweep_name}  ->  {sweep_error}")
        return []
    finished = [r for r in union if _as_float(r.get("primary_val_ppl")) < float("inf")]
    if finished:
        best = min(finished, key=lambda r: _as_float(r.get("primary_val_ppl")))
        print(f"\nSWEEP COMPLETE: {sweep_name}  ->  best {best['label']} "
              f"(val PPL {_as_float(best['primary_val_ppl']):.3f})")
    else:
        print(f"\nSWEEP COMPLETE: {sweep_name}  ->  no successful run")
    return union


# =============================================================================
# ANALYSIS  (reads sweep_results.csv; no model re-run)
# =============================================================================

def _read_sweep_csv(sweep_dir: Path) -> List[Dict[str, Any]]:
    path = sweep_dir / "sweep_results.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("inf")


def _base_label(label: str) -> str:
    r"""Strip the ``__s<seed>`` suffix a multi-seed sweep appends, so the seeded cells of one config
    collapse to one base label (``a2_on__s0`` / ``a2_on__s1`` -> ``a2_on``). A non-seeded label is
    returned unchanged."""
    base, _, tail = str(label).rpartition("__s")
    return base if (base and tail.lstrip("-").isdigit()) else str(label)


def _seed_aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""Group finished cells by base label into across-seed statistics: n, and the mean / SD / CV of the
    primary val PPL. A single-seed base reports n=1 with SD = CV = NaN: its seed variance is unknown,
    not zero. Sorted by mean PPL; bases with no finite PPL are dropped. The honest endpoint for
    reading a sub-3-percent ablation effect against seed noise."""
    groups: Dict[str, List[float]] = {}
    for r in rows:
        ppl = _as_float(r.get("primary_val_ppl"))
        if ppl < float("inf"):
            groups.setdefault(_base_label(r.get("label", "")), []).append(ppl)
    agg: List[Dict[str, Any]] = []
    for base, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        agg.append({"label": base, "n": n, "mean": mean, "sd": sd,
                    "cv": (sd / mean) if mean > 0 else float("nan")})
    return sorted(agg, key=lambda d: d["mean"])


def _aggregate_cells(
    cells:      List[Dict[str, Any]],
    key_fields: 'tuple | list',
    avg_fields: 'tuple | list',
) -> List[Dict[str, Any]]:
    r"""Collapse per-seed cells that share the same ``key_fields`` into one cell whose ``avg_fields`` are
    the across-seed means (non-finite values dropped; empty -> nan); non-averaged fields inherit the first
    member. Used so a multi-seed sweep's ``__s<seed>`` cells aggregate into one point/bar before the
    headline ablation figures plot them, instead of being skipped or read off one arbitrary seed
    (audit 2026-07-06 M4)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault(tuple(c.get(k) for k in key_fields), []).append(c)
    out: List[Dict[str, Any]] = []
    for members in groups.values():
        agg = dict(members[0])
        for f in avg_fields:
            vals = [float(m[f]) for m in members
                    if isinstance(m.get(f), (int, float))
                    and float(m[f]) == float(m[f]) and abs(float(m[f])) != float("inf")]
            agg[f] = (sum(vals) / len(vals)) if vals else float("nan")
        out.append(agg)
    return out


def _print_ablation_rows(rows: List[Dict[str, Any]], *, parameter_matched: bool) -> None:
    """Print one reachable ablation table with the complete structural gauge label."""
    if parameter_matched:
        print(f"{'label':<34}{'val PPL':>12}{'params':>12}{'budget dev':>12}  gauge classification")
        print("-" * 116)
    else:
        print(f"{'label':<34}{'val PPL':>12}{'params':>12}  gauge classification")
        print("-" * 104)
    for row in rows:
        ppl = "inf" if row.get("_ppl", _as_float(row.get("primary_val_ppl"))) == float("inf") else f"{row.get('_ppl', _as_float(row.get('primary_val_ppl'))):.3f}"
        params = f"{int(_as_float(row.get('n_params'))):,}" if row.get("n_params") not in ("", None) else "-"
        gauge = _sensitivity_gauge_label(row)
        if parameter_matched:
            relative = _as_float(row.get("param_relative_deviation"))
            deviation = "-" if relative == float("inf") else f"{relative * 100:.2f}%"
            print(f"{row['label']:<34}{ppl:>12}{params:>12}{deviation:>12}  {gauge}")
        else:
            print(f"{row['label']:<34}{ppl:>12}{params:>12}  {gauge}")


def analyze_sweep(sweep_dir: Path) -> None:
    rows = _read_sweep_csv(sweep_dir)
    if not rows:
        print(f"No results in {sweep_dir}")
        return
    for r in rows:
        r["_ppl"] = _as_float(r.get("primary_val_ppl"))
    rows.sort(key=lambda r: r["_ppl"])

    print(f"\n{'=' * 70}\nANALYSIS: {sweep_dir.name}\n{'=' * 70}")
    parameter_matched = any(r.get("target_n_params") not in ("", None) for r in rows)
    _print_ablation_rows(rows, parameter_matched=parameter_matched)

    finished = [r for r in rows if r["_ppl"] < float("inf")]
    if len(finished) > 1:
        best = finished[0]["_ppl"]
        print(f"\nrelative to best ({finished[0]['label']}):")
        for r in finished:
            print(f"  {r['label']:<34}{(r['_ppl'] - best) / best * 100:+.1f}%")

    # Across-seed aggregation (only when a base label carries multiple seeds): n / mean / SD / CV of the
    # val PPL per config, so a sub-3-percent effect is read against the seed noise rather than as a
    # single-seed point win. Native here rather than left advisory.
    agg = _seed_aggregate(rows)
    if any(a["n"] > 1 for a in agg):
        print(f"\nacross-seed aggregation ({sum(a['n'] for a in agg)} runs -> {len(agg)} cells):")
        print(f"{'label':<34}{'n':>4}{'mean PPL':>12}{'SD':>10}{'CV%':>8}")
        print("-" * 68)
        for a in agg:
            print(f"{a['label']:<34}{a['n']:>4}{a['mean']:>12.3f}{a['sd']:>10.3f}{a['cv'] * 100:>8.1f}")


def _cross_sweep_cohort_identity(
    aggregation_contract: Mapping[str, object],
) -> Dict[str, object]:
    r"""Project a sweep contract to fields that make validation-PPL comparisons commensurate."""
    contract = _validated_sweep_aggregation_contract(aggregation_contract)
    identity = {
        key: contract[key]
        for key in (
            "schema_version",
            "baseline_semantic_config_fingerprint",
            "seed_design",
            "dataset",
            "device",
            "tokenizer_tag",
            "data_seed_override",
            "max_tokens",
            "max_steps",
            "source_identities",
            "code_identity",
        )
    }
    if "stop_after_steps" in contract:
        identity["stop_after_steps"] = contract["stop_after_steps"]
    return identity


def _sweep_matches_cohort(
    sweep_dir:       Path,
    cohort_identity: Mapping[str, object],
) -> bool:
    """Whether one complete persisted sweep belongs to an exact comparison cohort."""
    try:
        meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
        return (
            isinstance(meta, Mapping)
            and meta.get("status") == "complete"
            and _cross_sweep_cohort_identity(meta["aggregation_contract"])
            == dict(cohort_identity)
        )
    except Exception:
        return False


def summarize_sweeps(
    output_dir: Path,

    *,
    cohort_identity: Optional[Mapping[str, object]] = None,
) -> None:
    r"""Cross-sweep comparison table: the best (lowest val PPL) cell of every persisted sweep.

    Printed once after all sweeps in a run (the per-sweep tables come from ``analyze_sweep`` as
    each sweep finishes). Scans EVERY sweep dir under ``output_dir`` so earlier-session sweeps
    are included, not just this run's.
    """
    print(f"\n{'=' * 70}\nBEST PER SWEEP  ({output_dir})\n{'=' * 70}")
    sweep_dirs = [
        directory
        for directory in sorted(output_dir.iterdir())
        if (
            directory.is_dir()
            and (directory / "sweep_results.csv").exists()
            and cohort_identity is not None
            and _sweep_matches_cohort(directory, cohort_identity)
        )
    ]
    if not sweep_dirs:
        print("No completed sweeps found in the current comparison cohort.")
        return
    print(f"{'sweep':<24}{'best config':<30}{'val PPL':>10}  gauge classification")
    print("-" * 104)
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        best = min(rows, key=lambda r: _as_float(r.get("primary_val_ppl")))
        gauge = _sensitivity_gauge_label(best)
        print(f"{d.name:<24}{best['label']:<30}{_as_float(best['primary_val_ppl']):>10.3f}  {gauge}")


# =============================================================================
# PLOTS
# =============================================================================

def _plt_or_none() -> Any:
    r"""Import matplotlib and apply the publication style once; return the module or ``None``.

    Plotting is best-effort -- a headless box or a missing matplotlib must never abort a sweep --
    so a failure prints once and returns ``None``, and the caller silently skips the figure.
    ``set_publication_style`` is idempotent, so repeated calls across per-sweep figures are fine.
    """
    try:
        import matplotlib.pyplot as plt
        from vfe3.viz.figures import set_publication_style
        set_publication_style()
        return plt
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"plotting unavailable ({exc}); skipping figure")
        return None


def _gauge_disclosure_text(gauge_purity: object) -> str:
    """Return the explicit gauge/structural disclosure printed on every ablation figure."""
    if not isinstance(gauge_purity, Mapping):
        return "Gauge classification: unavailable"
    classifications = gauge_purity.get("classifications_by_label")
    values = (
        sorted({str(value) for value in classifications.values()})
        if isinstance(classifications, Mapping) else []
    )
    value_text = ", ".join(values) if values else "unavailable"
    if gauge_purity.get("contains_coordinate_mean_only_nonintertwiner") is True:
        return (
            "Gauge classification: coordinate_mean_only_nonintertwiner present; "
            "block MLP is mean-only with covariance passthrough, not a Gaussian pushforward; "
            "this figure is not gauge-pure"
        )
    if gauge_purity.get("contains_independent_head_nonintertwiner") is True:
        return (
            "Gauge classification: independent_head_nonintertwiner present; "
            "this figure is not gauge-pure"
        )
    if gauge_purity.get("all_rows_on_gauge_pure_path") is True:
        return f"Gauge classification: all rows gauge-pure; head mixer(s): {value_text}"
    return f"Gauge classification: not all rows are gauge-pure; head mixer(s): {value_text}"


def _sweep_gauge_purity(sweep_dir: Path) -> object:
    """Read one sweep's persisted gauge-purity summary without trusting plot input rows."""
    try:
        metadata = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return metadata.get("gauge_purity") if isinstance(metadata, Mapping) else None


def _annotate_gauge_disclosure(fig: Any, gauge_purity: object) -> None:
    """Add a visible, figure-level gauge disclosure independent of individual axes."""
    fig.text(
        0.5,
        0.01,
        _gauge_disclosure_text(gauge_purity),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#b22222",
    )


def _save_ablation_figure(fig: Any, out: Path, sweep_dir: Path, plt: Any) -> None:
    """Annotate, save, and close one legacy ablation figure under the common disclosure policy."""
    _annotate_gauge_disclosure(fig, _sweep_gauge_purity(sweep_dir))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _numeric_sweep_axis(rows: Sequence[Mapping[str, object]]) -> Optional[List[float]]:
    """Parse a single-field numeric axis, withholding compound parameter-budget labels."""
    if any(row.get("target_n_params") not in ("", None) for row in rows):
        return None
    numeric: List[float] = []
    for row in rows:
        try:
            numeric.append(float(str(row["label"]).split("=")[-1]))
        except (KeyError, TypeError, ValueError):
            return None
    return numeric


def _plot_one_sweep(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>.png`` from the sweep's ACCUMULATED CSV (so a tacked-on re-run shows
    every point). Numeric ``param=value`` labels -> line plot, x-sorted by value; categorical arms
    -> bar plot, sorted by PPL. Both layouts print the best label and PPL. No-op when the sweep has
    no finished cell or matplotlib is absent.
    """
    rows = [r for r in _read_sweep_csv(sweep_dir) if _as_float(r.get("primary_val_ppl")) < float("inf")]
    if not rows:
        return
    plt = _plt_or_none()
    if plt is None:
        return

    labels = [r["label"] for r in rows]
    ppls = [_as_float(r["primary_val_ppl"]) for r in rows]

    # Numeric param=value labels -> line plot; categorical arms -> sorted bar plot.
    numeric = _numeric_sweep_axis(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if numeric is not None:
        order = sorted(range(len(numeric)), key=lambda k: numeric[k])
        ax.plot([numeric[k] for k in order], [ppls[k] for k in order], "o-", lw=2, ms=7)
        ax.set_xscale(SWEEPS.get(sweep_dir.name, {}).get("scale", "linear"))
        ax.set_xlabel(sweep_dir.name)
        ax.set_ylabel("validation PPL")            # numeric line plot: PPL on the y-axis
    else:
        order = sorted(range(len(ppls)), key=lambda k: ppls[k])
        vals = [ppls[k] for k in order]
        ax.barh(range(len(order)), vals,
                color=["#2ca02c" if j == 0 else "#1f77b4" for j in range(len(order))])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[k] for k in order])
        ax.invert_yaxis()
        ax.set_xlabel("validation PPL")            # horizontal bars: PPL on the x-axis (m30)
        # Zoom the value axis to a padded [min, max] window (not [0, max]) so near-equal PPLs are
        # visually distinguishable instead of all reading as full-length bars, and annotate each bar
        # with its value so the truncated bars are not misread.
        lo, hi = min(vals), max(vals)
        if hi > lo:
            span = hi - lo
            ax.set_xlim(lo - 0.25 * span, hi + 0.30 * span)
        for j, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", xy=(v, j), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, clip_on=False)
    best_index = min(range(len(ppls)), key=ppls.__getitem__)
    ax.text(
        0.98,
        0.98,
        f"Best: {labels[best_index]}\nPPL: {ppls[best_index]:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_title(sweep_dir.name)
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}.png"
    _save_ablation_figure(fig, out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_seed_aggregate(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_seed_aggregate.png`` -- across-seed mean val PPL +/- 1 SD per base label
    (the seed-grouped error bars the per-row table flattens). A no-op unless some base label has >= 2
    seeds, so it is safe to call after every sweep; a single-seed sweep draws nothing."""
    agg = sorted(_seed_aggregate(_read_sweep_csv(sweep_dir)), key=lambda d: d["mean"])
    if not any(a["n"] > 1 for a in agg):
        return
    plt = _plt_or_none()
    if plt is None:
        return
    y = list(range(len(agg)))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(agg) + 1.5))
    multi = [i for i, a in enumerate(agg) if a["n"] > 1]
    single = [i for i, a in enumerate(agg) if a["n"] == 1]
    ax.errorbar([agg[i]["mean"] for i in multi], multi, xerr=[agg[i]["sd"] for i in multi],
                fmt="o", color="#0072B2", ecolor="#888888", capsize=3, label="n >= 2: mean $\\pm$ 1 SD")
    if single:                                   # unknown seed variance: no error bar, distinct marker
        ax.plot([agg[i]["mean"] for i in single], single, linestyle="none", marker="D",
                markerfacecolor="none", color="#D55E00", label="n = 1: single seed, SD unknown")
        ax.legend(loc="best", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a['label']} (n={a['n']})" for a in agg])
    ax.invert_yaxis()
    ax.set_xlabel("validation PPL (across-seed mean $\\pm$ 1 SD)")
    ax.set_title(f"{sweep_dir.name}: seed-grouped aggregation")
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_seed_aggregate.png"
    _save_ablation_figure(fig, out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_rank_collapse(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_rank_collapse.png`` -- the F2/EXP-7 Dong r(X)-vs-depth overlay, one
    line per arm -- from the per-cell ``rank_resid_by_layer`` curves persisted in each
    ``ablation_result.json`` (off ``collect_diagnostics``; see ``_cell_diagnostics``). A no-op (writes
    nothing) unless at least one finished cell carries a >=2-layer rank curve, so it is safe to call
    after every sweep -- only the deep collect_diagnostics sweeps (rho_handoff) produce the curve."""
    arms: Dict[str, Any] = {}
    for r in _collect_sweep_results(sweep_dir):
        curve = r.get("rank_resid_by_layer")
        if (isinstance(curve, list) and len(curve) >= 2
                and _as_float(r.get("primary_val_ppl")) < float("inf")):
            arms[str(r.get("label", "?"))] = curve
    if not arms:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_rank_residual_by_depth
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"rank-collapse figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_rank_collapse.png"
    fig = plot_rank_residual_by_depth(arms)
    _save_ablation_figure(fig, out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_attention_entropy(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write the C1/EXP-4 figures -- the canonical-vs-surrogate PPL gap (grouped by kappa) and the
    -tau^{-1} Cov_beta gradient-gap magnitude vs kappa -- from the per-cell ``include_attention_entropy``
    / ``kappa_beta`` overrides and the ``cov_gap`` diagnostic. A no-op unless >= 2 finished cells carry
    an ``include_attention_entropy`` override, so it is safe to call after every sweep (only the
    attention_entropy sweep produces these)."""
    cells_ppl, cells_gap = [], []
    for r in _collect_sweep_results(sweep_dir):
        ov = r.get("overrides", {}) or {}
        if "include_attention_entropy" not in ov or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        kap = _as_float(ov.get("kappa_beta", BASELINE_CONFIG.get("kappa_beta")))
        if kap >= float("inf"):                                  # non-scalar kappa (per-head list) -> skip
            continue
        ent = bool(ov["include_attention_entropy"])
        cells_ppl.append({"include_attention_entropy": ent, "kappa": kap,
                          "ppl": _as_float(r.get("primary_val_ppl"))})
        if _as_float(r.get("cov_gap")) < float("inf"):
            cells_gap.append({"include_attention_entropy": ent, "kappa": kap,
                              "cov_gap": _as_float(r.get("cov_gap"))})
    cells_ppl = _aggregate_cells(cells_ppl, ("include_attention_entropy", "kappa"), ("ppl",))   # M4
    cells_gap = _aggregate_cells(cells_gap, ("include_attention_entropy", "kappa"), ("cov_gap",))
    if len(cells_ppl) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_cov_gap_vs_kappa, plot_entropy_ppl_gap
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"attention-entropy figures unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_ppl_gap.png"
    _save_ablation_figure(plot_entropy_ppl_gap(cells_ppl), out, sweep_dir, plt)
    print(f"  figure -> {out}")
    if len(cells_gap) >= 2:
        out = fig_dir / f"{sweep_dir.name}_cov_gap.png"
        _save_ablation_figure(plot_cov_gap_vs_kappa(cells_gap), out, sweep_dir, plt)
        print(f"  figure -> {out}")


def _compatible_cell_dirs(sweep_dir: Path) -> List[Path]:
    """Return only cell directories admitted by the persisted aggregation contract."""
    return [
        sweep_dir / _sanitize(str(result["label"]))
        for result in _collect_sweep_results(sweep_dir)
    ]


def _plot_wallclock_convergence(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Write ``figures/<sweep>_wallclock_convergence.png`` -- the D1/EXP-8 per-wall-clock convergence
    overlay (val PPL vs cumulative wall time, one line per arm, with steps/wall-to-target annotated) --
    from each cell's ``metrics.csv`` eval rows (val_ppl + wall_clock_s). A no-op unless >= 2 cells carry
    >= 2 eval points, so it is safe to call after every sweep (the gauge M-step sweeps populate it)."""
    arms: List[Dict[str, Any]] = []
    for run_dir in _compatible_cell_dirs(sweep_dir):
        cell = run_dir / "metrics.csv"
        if not cell.is_file():
            continue
        steps, ppls, walls = [], [], []
        try:
            with open(cell, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    vp, wc = _as_float(r.get("val_ppl")), _as_float(r.get("wall_clock_s"))
                    if vp < float("inf") and wc < float("inf"):     # eval rows carry both
                        steps.append(_as_float(r.get("step")))
                        ppls.append(vp)
                        walls.append(wc)
        except Exception:                                           # unreadable metrics.csv -> skip cell
            continue
        if len(ppls) >= 2:
            arms.append({"label": cell.parent.name, "step": steps, "val_ppl": ppls, "wall_clock_s": walls})
    if len(arms) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_wallclock_convergence
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"wallclock-convergence figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_wallclock_convergence.png"
    _save_ablation_figure(plot_wallclock_convergence(arms), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_gauge_transport(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A1/EXP-2 gauge ON/frozen/OFF(Omega=I) grouped-bar (val PPL by depth, omega_identity_dev
    annotated) from the gauge_transport cells. No-op unless >= 2 cells label as <mode>_<depth> with
    mode in {on,off,frozen}."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        parts = _base_label(str(r.get("label", ""))).split("_")   # strip __s<seed> before parsing (M4)
        if len(parts) != 2 or parts[0] not in ("on", "off", "frozen"):
            continue
        if _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        cells.append({"mode": parts[0], "depth": parts[1],
                      "ppl": _as_float(r.get("primary_val_ppl")),
                      "omega_dev": _as_float(r.get("omega_identity_dev"))})
    cells = _aggregate_cells(cells, ("mode", "depth"), ("ppl", "omega_dev"))   # collapse seeds (M4)
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_gauge_transport_bars
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"gauge-transport figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_gauge_bars.png"
    _save_ablation_figure(plot_gauge_transport_bars(cells), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_cg_coupling(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A3/EXP-10 combined PPL + median equivariance-residual bar from the cg_coupling cells (gated on
    the use_cg_coupling override). No-op unless >= 2 such cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        ov = r.get("overrides", {}) or {}
        if "use_cg_coupling" not in ov or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        cells.append({"label": str(r.get("label", "")), "ppl": _as_float(r.get("primary_val_ppl")),
                      "resid": _as_float(r.get("gauge_resid_in"))})
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_ppl_equivariance_bars
    except Exception as exc:
        print(f"cg-coupling figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_ppl_equiv.png"
    _save_ablation_figure(plot_ppl_equivariance_bars(cells), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_kappa_dispersion(sweep_dir: Path, fig_dir: Path) -> None:
    r"""H2/EXP-11 PPL vs per-head temperature dispersion std(kappa_beta), read from each cell's
    kappa_beta LIST override (the scalar-kappa sweeps carry no list -> skipped). No-op unless >= 2
    list-valued cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        kb = (r.get("overrides", {}) or {}).get("kappa_beta")
        if not isinstance(kb, (list, tuple)) or len(kb) < 2:
            continue
        if _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        try:
            vals = [float(x) for x in kb]
        except (TypeError, ValueError):
            continue
        m = sum(vals) / len(vals)
        disp = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5      # population std of the kappa list
        cells.append({"label": str(r.get("label", "")), "dispersion": disp,
                      "ppl": _as_float(r.get("primary_val_ppl"))})
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_kappa_dispersion
    except Exception as exc:
        print(f"kappa-dispersion figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_kappa_dispersion.png"
    _save_ablation_figure(plot_kappa_dispersion(cells), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_gauge_residual_drift(sweep_dir: Path, fig_dir: Path) -> None:
    r"""A2/EXP-9 builder-break gauge residual vs step (tied vs untied) from each cell's metrics.csv
    ``val_builder_resid`` eval series. No-op unless >= 2 cells carry >= 2 eval points."""
    arms: List[Dict[str, Any]] = []
    for run_dir in _compatible_cell_dirs(sweep_dir):
        cell = run_dir / "metrics.csv"
        if not cell.is_file():
            continue
        steps, res = [], []
        try:
            with open(cell, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rv = _as_float(row.get("val_builder_resid"))
                    if rv < float("inf"):                              # eval rows carry the residual
                        steps.append(_as_float(row.get("step")))
                        res.append(rv)
        except Exception:                                             # unreadable metrics.csv -> skip
            continue
        if len(res) >= 2:
            arms.append({"label": cell.parent.name, "step": steps, "resid": res})
    if len(arms) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_gauge_residual_drift
    except Exception as exc:
        print(f"gauge-residual-drift figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_residual_drift.png"
    _save_ablation_figure(plot_gauge_residual_drift(arms), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_mu_precond(sweep_dir: Path, fig_dir: Path) -> None:
    r"""B3/EXP-14 PPL-vs-n_e_steps figure, Fisher vs raw mean preconditioner, from the
    fisher_mu_precond cells (label '<precond>_T<n_e_steps>'). No-op unless >= 2 such cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        lab = _base_label(str(r.get("label", "")))   # strip __s<seed> before parsing (M4)
        if "_T" not in lab or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        pre, _, t = lab.partition("_T")
        if pre not in ("fisher", "raw"):
            continue
        try:
            n_e = int(t)
        except ValueError:
            continue
        cells.append({"precond": pre, "n_e_steps": n_e, "ppl": _as_float(r.get("primary_val_ppl"))})
    cells = _aggregate_cells(cells, ("precond", "n_e_steps"), ("ppl",))   # collapse seeds (M4)
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_mu_precond
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"mu-precond figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_mu_precond.png"
    _save_ablation_figure(plot_mu_precond(cells), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_renyi_saturation(sweep_dir: Path, fig_dir: Path) -> None:
    r"""B2/EXP-12 H(beta)-vs-alpha + kl_max-saturation-vs-alpha figure from the renyi_order cells
    (label 'renyi_order=<alpha>', with attn_entropy + energy_klmax_frac diagnostics). No-op unless
    >= 2 such cells finished."""
    cells: List[Dict[str, Any]] = []
    for r in _collect_sweep_results(sweep_dir):
        lab = str(r.get("label", ""))
        if not lab.startswith("renyi_order=") or _as_float(r.get("primary_val_ppl")) >= float("inf"):
            continue
        try:
            alpha = float(lab.split("=")[-1])
        except ValueError:
            continue
        cells.append({"alpha": alpha, "attn_entropy": _as_float(r.get("attn_entropy")),
                      "energy_klmax_frac": _as_float(r.get("energy_klmax_frac"))})
    if len(cells) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_renyi_saturation
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"renyi-saturation figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_renyi_saturation.png"
    _save_ablation_figure(plot_renyi_saturation(cells), out, sweep_dir, plt)
    print(f"  figure -> {out}")


def _plot_pos_extrapolation(sweep_dir: Path, fig_dir: Path) -> None:
    r"""H1/EXP-13 CE-vs-N extrapolation overlay (one line per positional scheme) from each cell's
    ``extrap_ce`` curve (persisted by run_single under collect_extrapolation). No-op unless >= 2 cells
    carry a >= 2-point curve. The train length (the smallest evaluated N = max_seq_len) is marked."""
    arms: Dict[str, Any] = {}
    all_n: List[float] = []
    for r in _collect_sweep_results(sweep_dir):
        curve = r.get("extrap_ce")
        if isinstance(curve, list) and len(curve) >= 2:
            arms[str(r.get("label", "?"))] = curve
            all_n += [float(p["n"]) for p in curve if isinstance(p, dict) and "n" in p]
    if len(arms) < 2:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    try:
        from vfe3.viz.figures import plot_pos_extrapolation
    except Exception as exc:                                  # plotting is best-effort, never fatal
        print(f"pos-extrapolation figure unavailable ({exc}); skipping")
        return
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / f"{sweep_dir.name}_extrapolation.png"
    _save_ablation_figure(
        plot_pos_extrapolation(arms, train_n=(min(all_n) if all_n else None)),
        out,
        sweep_dir,
        plt,
    )
    print(f"  figure -> {out}")


def _sensitivity_gauge_label(row: Mapping[str, object]) -> str:
    """Return the complete structural gauge label used by cross-sweep bars."""
    head_mixer = str(row.get("head_mixer_compatibility") or "unavailable")
    block_mlp = str(row.get("block_mlp_structural_mode") or "unavailable")
    return f"head_mixer={head_mixer}; block_mlp={block_mlp}"


def _plot_sensitivity(
    output_dir: Path,
    fig_dir:    Path,

    *,
    cohort_identity: Optional[Mapping[str, object]] = None,
) -> None:
    r"""Cross-sweep comparison: a PPL-range (worst - best) bar per sweep, sorted by sensitivity.

    Made once after all sweeps. Scans EVERY persisted sweep under ``output_dir`` (not just this
    run's), matching the per-sweep figures' accumulated view.
    """
    sweep_dirs = [
        directory
        for directory in sorted(output_dir.iterdir())
        if (
            directory.is_dir()
            and (directory / "sweep_results.csv").exists()
            and cohort_identity is not None
            and _sweep_matches_cohort(directory, cohort_identity)
        )
    ]
    sensitivity: List[Tuple[str, float, str, str]] = []      # sweep, PPL range, best label, gauge class
    gauge_summaries: List[Mapping[str, object]] = []
    for d in sweep_dirs:
        rows = [r for r in _read_sweep_csv(d) if _as_float(r.get("primary_val_ppl")) < float("inf")]
        if not rows:
            continue
        gauge_summary = _sweep_gauge_purity(d)
        if isinstance(gauge_summary, Mapping):
            gauge_summaries.append(gauge_summary)
        ppls = [_as_float(r["primary_val_ppl"]) for r in rows]
        best = min(rows, key=lambda r: _as_float(r["primary_val_ppl"]))
        sensitivity.append((
            d.name,
            max(ppls) - min(ppls),
            best["label"],
            _sensitivity_gauge_label(best),
        ))
    if not sensitivity:
        return
    plt = _plt_or_none()
    if plt is None:
        return
    sensitivity.sort(key=lambda t: t[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(sensitivity))))
    ax.barh(range(len(sensitivity)), [s[1] for s in sensitivity], color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(sensitivity)))
    ax.set_yticklabels([f"{s[0]}\n(best: {s[2]}; gauge: {s[3]})" for s in sensitivity])
    ax.invert_yaxis()
    ax.set_xlabel("validation PPL range (worst - best)")
    ax.set_title("hyperparameter sensitivity")
    fig.tight_layout()
    classifications = {
        f"{index}:{label}": value
        for index, summary in enumerate(gauge_summaries)
        for label, value in (
            summary.get("classifications_by_label", {}).items()
            if isinstance(summary.get("classifications_by_label"), Mapping) else ()
        )
    }
    _annotate_gauge_disclosure(fig, {
        "classifications_by_label": classifications,
        "contains_independent_head_nonintertwiner": any(
            summary.get("contains_independent_head_nonintertwiner") is True
            for summary in gauge_summaries
        ),
        "contains_coordinate_mean_only_nonintertwiner": any(
            summary.get("contains_coordinate_mean_only_nonintertwiner") is True
            for summary in gauge_summaries
        ),
        "all_rows_on_gauge_pure_path": bool(gauge_summaries) and all(
            summary.get("all_rows_on_gauge_pure_path") is True
            for summary in gauge_summaries
        ),
    })
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "sensitivity_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"comparison figure -> {out}")


def emit_registered_figures(
    specs:      Any,
    context:    Mapping[str, object],
    output_dir: Path,
) -> List[Path]:
    r"""Compatibility wrapper whose Matplotlib-backed implementation is imported only on demand."""
    from vfe3.viz.specs import emit_registered_figures as _emit_registered_figures

    return _emit_registered_figures(specs, context, output_dir)


def _render_sweep_figures(sweep_dir: Path, fig_dir: Path) -> None:
    r"""Render one complete sweep's registered and legacy figures inside a plotting worker."""
    from vfe3.viz.specs import FigureSpec
    from vfe3.viz.sweep_adapters import ablation_forest_kwargs, lr_grid_heatmap_kwargs

    meta_path = sweep_dir / "sweep_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or meta.get("status") != "complete":
        raise ValueError(f"sweep {sweep_dir.name!r} has no complete metadata")
    rows = _collect_sweep_results(sweep_dir)
    for row in rows:
        row["_cell_dir"] = str(sweep_dir / _sanitize(str(row["label"])))
    report_context = {
        "sweep_dir":      sweep_dir,
        "rows":           rows,
        "gauge_purity":   meta.get("gauge_purity"),
        "baseline_label": meta.get("forest_baseline_label"),
        "grid_x":         meta.get("grid_x"),
        "grid_y":         meta.get("grid_y"),
        "grid_x_values":  meta.get("grid_x_values"),
        "grid_y_values":  meta.get("grid_y_values"),
        "baseline":       tuple(meta["grid_baseline"]) if meta.get("grid_baseline") else None,
    }
    specs = []
    if meta.get("paired_token_bootstrap") is True:
        specs.append(FigureSpec(
            "ablation_forest",
            f"{sweep_dir.name}_ablation_forest.png",
            lambda ctx: ablation_forest_kwargs(
                ctx["sweep_dir"],
                ctx["baseline_label"],
                admitted_rows=ctx["rows"],
            ),
            postprocess=lambda fig, ctx: _annotate_gauge_disclosure(
                fig, ctx["gauge_purity"]
            ),
        ))
    if meta.get("grid_x") and meta.get("grid_y"):
        specs.append(FigureSpec(
            "lr_grid_heatmap",
            f"{sweep_dir.name}_lr_grid_heatmap.png",
            lambda ctx: lr_grid_heatmap_kwargs(
                ctx["rows"],
                ctx["grid_x"],
                ctx["grid_y"],
                ctx["grid_x_values"],
                ctx["grid_y_values"],
                ctx["baseline"],
            ),
            postprocess=lambda fig, ctx: _annotate_gauge_disclosure(
                fig, ctx["gauge_purity"]
            ),
        ))
    emit_registered_figures(specs, report_context, fig_dir)
    _plot_one_sweep(sweep_dir, fig_dir)
    _plot_seed_aggregate(sweep_dir, fig_dir)
    _plot_rank_collapse(sweep_dir, fig_dir)
    _plot_attention_entropy(sweep_dir, fig_dir)
    _plot_wallclock_convergence(sweep_dir, fig_dir)
    _plot_gauge_transport(sweep_dir, fig_dir)
    _plot_cg_coupling(sweep_dir, fig_dir)
    _plot_kappa_dispersion(sweep_dir, fig_dir)
    _plot_gauge_residual_drift(sweep_dir, fig_dir)
    _plot_pos_extrapolation(sweep_dir, fig_dir)
    _plot_renyi_saturation(sweep_dir, fig_dir)
    _plot_mu_precond(sweep_dir, fig_dir)


def _run_ablation_figures_isolated(
    output_dir: Path,

    *,
    scope:           str,
    invalidate:      bool                          = False,
    cohort_identity: Optional[Mapping[str, object]] = None,
) -> bool:
    r"""Render one ablation figure scope in a disposable process with child-only OMP policy."""
    output_dir = output_dir.resolve()
    with _unique_sibling_temp(output_dir / "ablation_figure_request.json") as request_path:
        request_path.write_text(json.dumps({
            "mode":    "ablation",
            "run_dir": str(output_dir),
            "scope":   scope,
            "invalidate": invalidate,
            "cohort_identity": (
                dict(cohort_identity) if cohort_identity is not None else None
            ),
        }), encoding="utf-8")
        environment = os.environ.copy()
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        environment["VFE3_FIGURE_REQUEST"] = str(request_path)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            completed = run_process_tree(
                [sys.executable, "-m", "vfe3.viz.figure_worker"],
                cwd=str(Path(__file__).resolve().parent),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_ABLATION_FIGURE_TIMEOUT_SECONDS,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "isolated ablation figure process exceeded %d seconds; numeric sweep results are saved",
                _ABLATION_FIGURE_TIMEOUT_SECONDS,
            )
            return False
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "isolated ablation figure process could not start (%s); numeric sweep results are saved",
                exc,
            )
            return False
    if completed.stdout.strip():
        logger.info("isolated ablation figure process:\n%s", completed.stdout.rstrip())
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "no stderr"
        logger.warning(
            "isolated ablation figure process exited with code %d (%s); numeric sweep results are saved",
            completed.returncode,
            detail[-4000:],
        )
        return False
    if completed.stderr.strip():
        logger.info(
            "isolated ablation figure process diagnostics:\n%s",
            completed.stderr.rstrip(),
        )
    return True


# =============================================================================
# MAIN  (click-to-run; edit CONFIG above)
# =============================================================================

def _write_ablation_run_status(
    output_dir: Path,

    *,
    status:                str,
    requested_sweeps:      List[str],
    incomplete_sweeps:     List[str],
    failed_figure_scopes:  List[str],
) -> None:
    """Atomically publish the terminal completeness of one click-to-run invocation."""
    _write_json_atomic(output_dir / "ablation_run_status.json", {
        "schema_version":       1,
        "status":               status,
        "requested_sweeps":     list(requested_sweeps),
        "incomplete_sweeps":    list(dict.fromkeys(incomplete_sweeps)),
        "failed_figure_scopes": list(dict.fromkeys(failed_figure_scopes)),
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(CONFIG["output_dir"])

    if CONFIG["list_only"]:
        # Every registered sweep (CONFIG["sweep"]=None) or just the named one; an asterisk marks
        # those in the curated SWEEP_ORDER that a None-sweep run would execute.
        names = sorted(SWEEPS) if CONFIG["sweep"] is None else [CONFIG["sweep"]]
        active = set(SWEEP_ORDER)
        print(f"\nRegistered sweeps ({len(names)} shown; * = in SWEEP_ORDER):\n")
        print(f"  {'name':<28}{'runs':>5}  description")
        print("-" * 90)
        for name in names:
            s = SWEEPS[name]
            mark = "*" if name in active else " "
            print(f"{mark} {name:<28}{sweep_n_runs(s):>5}  {s['description']}")
        print(f"\n{len(SWEEPS)} sweeps registered; SWEEP_ORDER runs {len(SWEEP_ORDER)} "
              f"({sum(sweep_n_runs(SWEEPS[n]) for n in SWEEP_ORDER)} runs).")
        return 0

    # ---- contiguous run: per sweep { train -> analyze table -> PPL figure }, then comparison ----
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])

    sweep_names = SWEEP_ORDER if CONFIG["sweep"] is None else [CONFIG["sweep"]]
    for name in sweep_names:
        if name not in SWEEPS:
            raise ValueError(f"unknown sweep {name!r}; choose from {sorted(SWEEPS)}")
    validate_sweeps(sweep_names)                             # scheduled arms MUST construct upfront
    validate_sweeps(list(SWEEPS), require_construction=False)  # rest of registry: names/fields only

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nVFE_3.0 ablation suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seed:    {CONFIG['seed']}"
          f"\n  sweeps:  {', '.join(sweep_names)}")

    current_cohort: Optional[Dict[str, object]] = None
    incomplete_sweeps: List[str] = []
    failed_figure_scopes: List[str] = []
    _write_ablation_run_status(
        output_dir,
        status="running",
        requested_sweeps=list(sweep_names),
        incomplete_sweeps=incomplete_sweeps,
        failed_figure_scopes=failed_figure_scopes,
    )
    if not _run_ablation_figures_isolated(
            output_dir,
            scope="__sensitivity__",
            invalidate=True):
        failed_figure_scopes.append("__sensitivity__:invalidate")
        incomplete_sweeps.extend(sweep_names)
        logger.error(
            "could not invalidate prior sensitivity figures; refusing to publish a mixed generation"
        )
        _write_ablation_run_status(
            output_dir,
            status="incomplete",
            requested_sweeps=list(sweep_names),
            incomplete_sweeps=incomplete_sweeps,
            failed_figure_scopes=failed_figure_scopes,
        )
        return 1
    for name in sweep_names:
        sweep_scope = _sweep_output_scope(name)
        if not _run_ablation_figures_isolated(output_dir, scope=sweep_scope, invalidate=True):
            failed_figure_scopes.append(f"{sweep_scope}:invalidate")
            incomplete_sweeps.append(name)
            logger.error(
                "could not invalidate prior %r figures; refusing to run beside stale output", name
            )
            continue
        run_sweep(name, output_dir, dataset=CONFIG["dataset"], device=device,
                  seed=CONFIG["seed"], resume=CONFIG["resume"],
                  max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"],
                  stop_after_steps=CONFIG.get("stop_after_steps"))
        sweep_dir = output_dir / sweep_scope

        try:
            meta = json.loads((sweep_dir / "sweep_meta.json").read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("sweep %s has no readable completion metadata: %s", name, exc)
            incomplete_sweeps.append(name)
            continue
        if not isinstance(meta, Mapping) or meta.get("status") != "complete":
            logger.error(
                "sweep %s is incomplete; later requested sweeps will still run", name)
            incomplete_sweeps.append(name)
            continue
        try:
            sweep_cohort = _cross_sweep_cohort_identity(meta["aggregation_contract"])
        except Exception as exc:
            logger.error("sweep %s has invalid comparison metadata: %s", name, exc)
            incomplete_sweeps.append(name)
            continue
        if current_cohort is None:
            current_cohort = sweep_cohort
        elif sweep_cohort != current_cohort:
            logger.error("sweep %s belongs to a different comparison cohort", name)
            incomplete_sweeps.append(name)
            continue
        analyze_sweep(sweep_dir)                            # this sweep's numeric table (accumulated)
        if not _run_ablation_figures_isolated(output_dir, scope=sweep_scope):
            failed_figure_scopes.append(f"{sweep_scope}:render")

    # ---- after all sweeps: the cross-sweep comparison ----
    if current_cohort is not None:
        if not _run_ablation_figures_isolated(
                output_dir,
                scope="__sensitivity__",
                cohort_identity=current_cohort):
            failed_figure_scopes.append("__sensitivity__:render")
        summarize_sweeps(output_dir, cohort_identity=current_cohort)
    if incomplete_sweeps:
        logger.error(
            "incomplete sweeps withheld from analysis: %s",
            ", ".join(incomplete_sweeps),
        )
    status = "complete" if not incomplete_sweeps and not failed_figure_scopes else "incomplete"
    _write_ablation_run_status(
        output_dir,
        status=status,
        requested_sweeps=list(sweep_names),
        incomplete_sweeps=incomplete_sweeps,
        failed_figure_scopes=failed_figure_scopes,
    )
    if failed_figure_scopes:
        logger.error(
            "requested figure scopes failed: %s",
            ", ".join(failed_figure_scopes),
        )
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
