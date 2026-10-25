r"""Click-to-run PARAMETER-scaling experiment runner for the VFE_3.0 transformer.

Scaling experiments are with respect to NUMBER OF PARAMETERS. This runner loops a size grid x a
seed list, training each (size, seed) cell into its own self-contained ``RunArtifacts`` directory and
calling ``finalize_run`` so EVERY point carries the canonical held-out TEST cross-entropy and the
enriched ``scaling_point`` block (n_params, n_gen, active-params-per-token, FLOP proxies, wall-clock).
There is no CLI arg parsing (project policy): edit the ``CONFIG`` dict and the active ``ROUTES`` at the
bottom. Preview with ``dry_run=True``, then set it False to train.
Aggregate + fit + plot afterwards with ``scaling_analysis.py``.

WHY A PARAMETER AXIS IS SUBTLE HERE (read before picking a grid). The token-prior pure path has
``mu_embed (V,K)``, ``sigma_log_embed (V,K)``, ``phi_embed (V,n_gen)``, and a scalar
(prior_bank.py), so ``N = 2*V*K + V*n_gen + 1`` with ``V=50257``. An untied decoder adds a second
``2*V*K`` table pair. ``phi_embed = V*n_gen`` usually DOMINATES. ``n_gen`` is set by the gauge group:
for ``block_glk`` it is ``K^2/n_heads`` (so FEWER/larger
blocks = MORE params -- the opposite sign of a standard transformer); the ``so_n`` tower decouples
``n_gen`` from ``K`` entirely. Three consequences:
the gauge group / n_heads is a FIRST-CLASS parameter lever; growing ``embed_dim`` moves ``N`` on two
fronts (linear ``2VK`` + quadratic ``V*n_gen``); and depth adds parameters when
relational maps or block MLPs are enabled.
Inner iteration count changes inference work while reusing parameters.

The baseline operating point is the self-contained ``config`` dict IN THIS FILE (it is NOT imported
from ``train_vfe3.py`` -- edit ``config`` below to set what every scaling cell trains around). ``BASELINE``
is bound to it right after the dict is built. Each cell overrides only the scale knob(s). Equal-token
budget: ``max_steps`` / ``batch_size`` /
``max_seq_len`` are held fixed across the parameter routes, so ``tokens_seen`` is constant and the fitted
exponent is the equal-data exponent. A missing tokenized cache raises ``FileNotFoundError`` (no
synthetic substitution); build the corpus cache first (see ``vfe3/data``).
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import gc
from vfe3.artifact_utils import canonical_json_sha256 as _canonical_json_sha256
import json
import logging
import math
import stat
import time
import warnings
from collections.abc import Mapping
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vfe3.config import _INHERITED_HYPERPARAMETERS, VFE3Config, config_from_serialized, serialize_config
from vfe3.data.datasets import (
    _tokenizer_tag,
    cache_source_identity,
    tokenizer_vocab_size as _tokenizer_vocab_size,
    make_dataloader,
    tokens_per_char as _tokens_per_char,
)
from vfe3.metric_contracts import perplexity_matches_ce
from vfe3.model.block_mlp import get_block_mlp_registration
from vfe3.model.model import VFEModel, build_group
from vfe3.path_utils import portable_path_component_key
from vfe3.run_artifacts import RunArtifacts, _git_code_identity, _write_json_atomic, finalize_run
from vfe3.runtime import restored_fp32_matmul_precision, seed_everything
from vfe3.train import train

logger = logging.getLogger("scaling")

_SCALING_CELL_SCHEMA_VERSION = 2
_SCALING_REUSE_DIGEST_FIELD = "reuse_contract_sha256"
_SCALING_OWNER_SCHEMA_VERSION = 1
_SCALING_OWNER_FILENAME = "scaling_cell_owner.json"
_SCALING_SUMMARY_DIGEST_FIELD = "scaling_reuse_contract_sha256"


# =============================================================================
# CLICK-TO-RUN KNOBS  -- edit, then run.
# =============================================================================
CONFIG: Dict[str, Any] = {
    # Standard routes: width, depth, width_depth, train_steps. Grids are in SWEEPS below.
    # Historical gauge/block and muP routes remain available in ROUTES.
    "routes": ["width"],
    "dry_run": True,             # print the full plan, without loading data or building a model

    # Seeds per cell. Graduated budget is sensible (more seeds at the cheap small end); the simplest
    # honest default is one shared list applied to every cell -- trim/extend per your compute budget.
    "seeds":      [6, 64, 23],
    "data_seed":  3,             # fixed order across model seeds; "model" = per-seed; None = global RNG

    "device":     "auto",                                   # 'auto' -> CUDA (RTX 5090) else CPU

    # Dataset for every run (NOT a VFE3Config field; the loader seam). Held-out CE is comparable across
    # sizes only within one tokenizer/corpus.
    "dataset":    "wikitext-103",              # "wikitext-103" | "wiki-ja" | "wiki-en" | "wiki-ar"

    # Cap the TRAIN stream for fast scaling passes (validation/test always read in full). None = full.
    "max_tokens": None,

    # Override every run's max_steps (None = use the local `config` max_steps below). HOLD THIS FIXED across the
    # parameter routes for an equal-token budget (so tokens_seen is constant and the exponent is clean).
    "max_steps":  None,

    # Protect verified completed cells with the same effective config and corpus. Prior-code records
    # are skipped with explicit provenance labeling; partial cells are restarted.
    "resume":     True,

    # Per-cell router cap for routes that vary the head dimension d_head = embed_dim / n_heads: a number c sets
    # kl_max = c * d_head in every cell that does not set kl_max itself, so the per-dimension cap is the same in
    # every cell; None keeps today's caps (BASELINE's kl_max, 8 * embed_dim when None).
    "kl_max_per_head_dim": None,  # None | a positive number of nats per head dimension

    # Under resume, a cell whose earlier attempt crashed or ended non-finite is reported, not retrained, so
    # the fit is not conditioned on eventual success; True retrains it (every attempt is archived).
    "retry_failed": False,       # False | True

    "output_dir": "vfe3_scaling_results/standard",
}

# Width keeps the baseline head count and all other architecture toggles. Choose widths
# divisible by n_heads (and even per-head widths with RoPE). Depth adds layer-owned
# relational maps and MLPs when enabled. Step sweeps hold model size and batch/context fixed.
SWEEPS: Dict[str, List[int]] = {
    "widths": [20, 40, 80, 120, 160],   # five sizes: the offset law needs five for R^2/CI (four for the plain law)
    "depths": [1, 2, 4, 6],
    "train_steps": [10000, 30000, 100000],
}

# Independent baseline, initialized from train_vfe3.py on September 22, 2026.
# Edit freely here; later launcher edits are NOT imported into scaling runs.
config = dict(



    #################################
    #            Training
    #################################


    vocab_size                = _tokenizer_vocab_size(CONFIG["dataset"]),   # follows DATASET's tokenizer
                                                                  # (gpt2 50257 / cl100k 100277 /
                                                                  #  word 267735); override to narrow

    embed_dim                 = 160,                  # K, total belief dim (must be divisible by n_heads)
    n_heads                   = 5,                   # = number of irrep blocks (alibi priors require this)

    max_seq_len               = 256,                 # N, context length
    eval_stride               = 128,

    batch_size                = 8,
    max_steps                 = 240000,

    n_layers                  = 6,                   # L, number of blocks
    n_e_steps                 = 1,                   # T, E-step inner iterations

    seed                      = 6,
    warmup_steps              = 100,
    deterministic             = True,


    # Attention execution; all interfaces have full input/output.
    attention_mode                   = "projected_predictive",  # "projected_predictive" | "projected_gradient"
    relational_receiver_measurement  = "inverse",               # "inverse" O'^-1, the dual measurement (VFE3Config default) | "adjoint" O'^T (projected_gradient only)



    attention_residual               = True,   # predictive S/T/O: add the incoming mean



    #################################
    # belief family
    #################################


    family                    = "gaussian_diagonal", # "gaussian_diagonal" | "gaussian_full" | "gaussian_block" (per-head SPD blocks; equal head-block groups, kernel readout)


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



    decode_mode                       = 'linear',  # prior-bank: diagonal_chunked/full_chunked/expected_likelihood_chunked
    encode_mode                       = "per_token",   # "per_token" | "per_token_additive" (additive only when phi carries no position; ~1.5x faster at K=80, currently unablated)
    decode_chunk_size                 = 8192,
    decode_tau                        = 0.008,       #prior_bank = True temperature

    unigram_kappa                     = 1.0,         # tempering on log pi_v (1.0 = exact Bayes class prior)


    decode_unigram_prior              = True,       # add kappa*log pi_v (corpus unigram) to decode logits
    untie_decode_bank                 = False,       # prior-bank modes only: decode reads its OWN cloned (V,K) tables


    #################################
    #     Optional Block MLP        #
    #################################
    use_block_mlp             = True,
    block_mlp_expansion       = 8,
    block_mlp_activation      = "gelu",
    block_mlp_dropout         = 0.00,


    #################################
    #          Gauge Group
    #################################
    gauge_transport           = "on",         # gauge-frame ABLATION (A1/EXP-2): "on" (pure, learned frame)
                                              #   | "off" (Omega=I exactly) | "frozen" (random fixed frame)

    gauge_group               = "block_glk",    # "block_glk" | "so_n" | "su_n" | "block_so" | "block_co" | "block_sl" | "block_borel" | "block_unipotent" | "block_glmc" (block_glmc: even d_head)

    group_n                   = None,                # so_n only; must stay None for block_glk
    irrep_spec                = None,                # so_n only; must stay None for block_glk

    ####################################
    #       Positional Encoding
    #    gauge-frame PE (pos_phi): exact right factor exp(X_i) exp(Y_n)
    #     gauge-RoPE (rope_mode)
    ####################################

    pos_phi                   = "learned",           # "none" (pure path) | "learned" | "frozen"
    rope_mode                 = "none",              # "none" | "frame" | "frame_value" | "comparison" | "comparison_invariant"

    rope_base                 = 100.0,               # rotary frequency base



    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################
    lambda_alpha_mode          = "state_dependent_per_coord",  # "state_dependent" | "state_dependent_per_coord"

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

    lambda_beta                = 1.0,        # belief-coupling block weight (1.0 = pure F)


    # Further Regularizers
    mstep_self_coupling_weight = 0.0,       # alpha_hat * sum_i KL(q_i*||p_i) M-step term (0 = OFF)


    #################################
    #         Belief E-step
    #         Learning Rates
    #################################


    e_q_mu_lr                 = 0.9,
    e_q_sigma_lr              = 0.01,



    #################################
    #    Embedding/Priors M-step
    #        Learning Rates
    #################################
    m_phi_update_mode         = "adamw",      # "adamw" | "pullback_group" | "fisher_group" (belief-Fisher natural gradient)

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
    m_layernorm_lr            = 0.05,      # None inherits m_p_mu_lr; own M-step rate for the





    min_lr_frac               = 0.01,


    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    layernorm_affine          = True,
    norm_type_block           = "layernorm",        # "none" | "mahalanobis" |"layernorm"


    #################################
    #         Misc/Logging
    #################################

    log_interval               = 1200,       # console log every N steps (0 = off)
    eval_interval              = 24000,      # periodic validation every N steps (0 = off)
    checkpoint_interval        = 60000,     # save a resumable checkpoint every N steps (0 = off)


    #################################
    #        Numerics
    #################################

    sigma_max                            = 100.0,
    e_mu_q_trust                         = 2,
    e_sigma_q_trust                      = 1.0,


    kl_max                               = None,          # None (8 * embed_dim, per cell) | nats; a number stays fixed across widths
    amp_dtype                            = 'bf16',        # None=fp32 | 'bf16' , 'fp16'. Sigma must be at least fp32
    fp32_matmul_precision                = "highest",       # None (leave as-is) | "highest" | "high" | "medium"

    grad_clip                            = 1.0,         # L2 gradient clip applied independently per optimizer role



    # Working precision of the decode's a_v reduction. a_v accumulates
    # terms of size 1/sigma_v and the result is divided by 2*tau (0.016 live), so the fp32 error is
    # amplified twice. Measured max |a_v - fp64| at K=20: 5.2e-6 at sigma_v=4, 2.5e-3 at 1e-2,
    # 2.2e-1 at 1e-4 -- the last being ~12 logits. NOTE the audit filed this as a cancellation
    # defect in the expanded square and proposed differencing before squaring; that was measured and
    # buys only ~1.8x, while the SAME algebra in float64 buys nine decades. It is a precision
    # problem, not a factorization problem. sigma_log_embed trains freely against only the eps=1e-6
    # floor, so this is harmless at init and about one order of magnitude of shrinkage from
    # material; watch the decode_sigma_v_min column of metrics.csv. Default "fp32" is bit-identical
    # to every run on disk.
    decode_av_precision                  = "fp64",     #  "fp32" | "fp64"


    # Strict decoder ranking guard. When enabled, analytic
    # fp32 decoders construct conservative score intervals and, if ANY row's winning interval
    # overlaps a competitor, recompute the complete (B,N,V) decode in differentiable float64.
    # That is an accuracy diagnostic, not a production-safe fallback: at B=64, N=128, V=50,257
    # the promoted logits alone occupy about 3.1 GiB before backward intermediates. Keep it
    # opt-in so the default chunked route remains streamed and bounded.
    decode_ranking_fp64_escalation       = True,





    skip_belief_sigma_update             = True,        # skip the belief-channel sigma E-step update

    generate_embedding_figures           = True,
)

# The scaling operating point IS the local ``config`` dict above (self-contained; NOT train_vfe3.py).
# Edit ``config`` to change what every cell trains around; each ROUTE cell overrides only its scale knob(s).
BASELINE: Dict[str, Any] = config


# =============================================================================
# ROUTE BUILDERS  -- each returns a list of cells; a cell is
#   {"label", "route", "scale_knob", "overrides": {VFE3Config field: value, ...}}.
# Every cell's overrides must independently satisfy VFE3Config.__post_init__ (n_heads | embed_dim,
# alibi priors need n_heads == n_blocks, ...); a cell that
# violates a cross-field constraint is caught at construction and recorded as a config-error point
# (never crashes the grid), but the defaults below are pre-satisfied so they construct cleanly.
# =============================================================================

def build_standard_routes(settings: Dict[str, List[int]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build independent width, depth, joint-size, and training-step grids from BASELINE."""
    for name in ("widths", "depths", "train_steps"):
        values = settings[name]
        if (not values or any(type(value) is not int or value <= 0 for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"SWEEPS[{name!r}] must contain unique positive integers")
    widths, depths, steps = (settings[name] for name in ("widths", "depths", "train_steps"))
    return {
        "width": [{"label": f"K{k}", "route": "width", "scale_knob": "embed_dim",
                   "overrides": {"embed_dim": k}} for k in widths],
        "depth": [{"label": f"L{depth}", "route": "depth", "scale_knob": "n_layers",
                   "overrides": {"n_layers": depth}} for depth in depths],
        "width_depth": [{"label": f"K{k}_L{depth}", "route": "width_depth",
                         "scale_knob": "embed_dim+n_layers",
                         "overrides": {"embed_dim": k, "n_layers": depth}}
                        for k in widths for depth in depths],
        "train_steps": [{"label": f"steps{step}", "route": "train_steps", "scale_knob": "max_steps",
                         "overrides": {"max_steps": step}} for step in steps],
    }


def route_grow_k(embed_dims: List[int], n_heads: int = 4) -> List[Dict[str, Any]]:
    r"""Grow N by widening embed_dim at a FIXED block_glk head count (route A). Mixed linear+quadratic
    token-prior route: 2VK grows linearly, phi_embed = V*K^2/n_heads quadratically. n_heads stays equal to the
    block count so the baseline causal_alibi prior remains valid."""
    # kl_max scales WITH k (audit 2026-07-27): BASELINE's kl_max=None resolves to 8*k in each cell.
    return [{"label": f"K{k}", "route": "grow_K", "scale_knob": "embed_dim",
             "overrides": {"embed_dim": k, "n_heads": n_heads, "gauge_group": "block_glk"}}
            for k in embed_dims]


_VFE3_DEFAULTS = {f.name: f.default for f in _dc_fields(VFE3Config)}


def _baseline_value(key: str) -> float:
    r"""Operating-point value of a VFE3Config float field: the local ``config`` dict (BASELINE) if it
    sets it, else the dataclass default. A rate left at None resolves through its parent in
    ``_INHERITED_HYPERPARAMETERS`` (the rule ``VFE3Config.resolved_hyperparameter`` applies), so the muP
    route scales the rate the anchor cell actually trains with, without hard-coding the operating point."""
    value = BASELINE.get(key, _VFE3_DEFAULTS.get(key))
    while value is None:
        key = _INHERITED_HYPERPARAMETERS[key]
        value = BASELINE.get(key, _VFE3_DEFAULTS.get(key))
    return float(value)


# M-step groups whose matrices have fan-in proportional to K at fixed n_heads: the query/key score maps
# raw_score_q/k (n, d, K), the content maps T (n, d, K) and O (n, K, d), the block MLP Linear(K, mK) and
# Linear(mK, K) (hidden weights), and the linear readout output_proj_weight (V, K) (output weights).
_MUP_HIDDEN_OUTPUT_RATES: Tuple[str, ...] = (
    "m_relational_score_lr", "m_relational_content_lr", "m_block_mlp_lr", "m_decode_mu_lr")


def route_grow_k_mup(embed_dims: List[int], n_heads: int = 4, anchor_k: int = 20) -> List[Dict[str, Any]]:
    r"""Width-stability route with the muP Adam learning-rate rule for the hidden and output matrices
    (F1/EXP-6; Yang et al. 2022, arXiv:2203.03466, Table 3).

    For each K it emits a matched PAIR at fixed ``n_heads``: a ``K{k}_fixed`` arm at the baseline rates
    (the width-fixed control) and a ``K{k}_mup`` arm that multiplies every rate in
    ``_MUP_HIDDEN_OUTPUT_RATES`` by :math:`K_0/K`, Table 3's Adam rate :math:`\propto 1/\mathrm{fan\_in}`
    for hidden and output weights. Table 3 holds the input tables width-independent, so the mean table's
    rate ``m_p_mu_lr`` and init ``mu_init_std``, and likewise ``m_p_sigma_lr``, ``m_phi_lr`` and
    ``m_pos_phi_lr``, stay at the baseline, as does the vector affine rate ``m_layernorm_lr`` (the
    "biases" column). ``e_q_mu_lr`` is an E-step step size with no muP rule and is not varied. The
    arm is the muP rate rule only, not full muP: the readout init variance (xavier_uniform,
    :math:`2/(K+V)`, against muP's :math:`1/\mathrm{fan\_in}^2`) and the attention temperature
    :math:`\tau = \kappa \sqrt{d_\mathrm{head}}` (Transformer muP uses :math:`1/d`; whether the KL score
    needs the analogue is open) are unchanged. BOTH arms run at kl_max = 8*K per cell (BASELINE's
    kl_max=None), which avoids the frozen-kl_max width confound. Anchored at K=anchor_k, where the factor
    is 1 and the two arms coincide. Scoped to this route so the other routes are untouched. The fitted
    exponent should be compared against embed_dim, not n_params (which is K^2-dominated by phi_embed)."""
    base_rates = {key: _baseline_value(key) for key in _MUP_HIDDEN_OUTPUT_RATES}
    cells: List[Dict[str, Any]] = []
    for k in embed_dims:
        common = {"embed_dim": k, "n_heads": n_heads, "gauge_group": "block_glk"}
        cells.append({"label": f"K{k}_fixed", "route": "grow_K_mup", "scale_knob": "embed_dim",
                      "overrides": dict(common)})
        w = anchor_k / k                                     # muP width ratio K_0/K (1 at the anchor)
        cells.append({"label": f"K{k}_mup", "route": "grow_K_mup", "scale_knob": "embed_dim",
                      "overrides": {**common, **{key: rate * w for key, rate in base_rates.items()}}})
    return cells


def route_blocksize(embed_dim: int, n_heads_list: List[int]) -> List[Dict[str, Any]]:
    r"""Grow N by SHRINKING the head count at fixed K (route B): block_glk n_gen = K^2/n_heads, so
    fewer/larger blocks -> more params. n_heads == n_blocks keeps causal_alibi valid."""
    return [{"label": f"K{embed_dim}_h{h}", "route": "blocksize", "scale_knob": "n_heads",
             "overrides": {"embed_dim": embed_dim, "n_heads": h, "gauge_group": "block_glk"}}
            for h in n_heads_list]


def route_grow_k_fixed_block(embed_dims: List[int], block: int) -> List[Dict[str, Any]]:
    r"""Grow N by widening K at a FIXED gauge block size GL(``block``): n_heads = K/block scales WITH K
    (every head is one GL(block) frame). Because block_glk has n_gen = K^2/n_heads = block*K, phi_embed
    = V*block*K grows LINEARLY in K here -- contrast ``route_grow_k`` (fixed n_heads, so the block grows
    and n_gen ~ K^2). kl_max = 8*K per cell (BASELINE's kl_max=None). K must be a
    positive multiple of ``block``; non-multiples are skipped with a warning."""
    cells: List[Dict[str, Any]] = []
    for k in embed_dims:
        if k <= 0 or k % block != 0:
            logger.warning("  [skip] grow_K_GL%d: K=%d is not a positive multiple of block=%d", block, k, block)
            continue
        h = k // block
        ov: Dict[str, Any] = {"embed_dim": k, "n_heads": h, "gauge_group": "block_glk"}
        cells.append({"label": f"K{k}_GL{block}", "route": f"grow_K_GL{block}",
                      "scale_knob": "embed_dim", "overrides": ov})
    return cells


def route_vary_block_fixed_k(
    embed_dim:   int,
    blocks:      List[int],

    *,
    gauge_group:     str                     = "block_glk",
    tag:             Optional[str]           = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    r"""Fixed K, vary the gauge block size GL(``b``): n_heads = K/b, so block_glk n_gen = K^2/n_heads =
    b*K. FEWER/LARGER blocks (bigger b) = MORE params (n_gen up) -- the opposite sign of a standard
    transformer, and the parameter axis here. This is the ``route_blocksize`` idea written in GL(b) terms
    (b=K/n_heads). kl_max = 8*K (BASELINE's kl_max=None; constant, as K is fixed across the route). Each b must
    divide K; non-divisors are skipped with a warning.

    ``gauge_group`` selects the structure group per cell (default ``block_glk``, the untied per-head
    GL(d_head)). ``tag`` overrides the internal cell route
    tag (default ``f'blocks_K{embed_dim}'``); give a distinct tag to a variant route (different budget or
    group) so ``scaling_analysis`` keeps its points separate from the base ``blocks_K48`` run."""
    cells: List[Dict[str, Any]] = []
    for b in blocks:
        if b <= 0 or embed_dim % b != 0:
            logger.warning("  [skip] blocks_K%d: block=%d does not divide K=%d", embed_dim, b, embed_dim)
            continue
        h = embed_dim // b
        ov: Dict[str, Any] = {"embed_dim": embed_dim, "n_heads": h, "gauge_group": gauge_group}
        if extra_overrides:
            ov.update(extra_overrides)
        cells.append({"label": f"K{embed_dim}_GL{b}", "route": tag or f"blocks_K{embed_dim}",
                      "scale_knob": "n_heads", "overrides": ov})
    return cells



def route_inference_t(n_e_steps_list: List[int]) -> List[Dict[str, Any]]:
    r"""FLAT-N inference-compute axis: more E-step inner iterations T at constant params. route tagged
    'inference' so the analyzer plots it on the inference-capacity figure, NEVER the L(N) curve."""
    return [{"label": f"T{t}", "route": "inference", "scale_knob": "n_e_steps",
             "overrides": {"n_e_steps": t}} for t in n_e_steps_list]


def route_inference_l(n_layers_list: List[int]) -> List[Dict[str, Any]]:
    """Legacy depth grid; layer-owned maps and MLPs can add parameters."""
    return [{"label": f"L{n}", "route": "depth", "scale_knob": "n_layers",
             "overrides": {"n_layers": n}} for n in n_layers_list]


# ============================ ROUTE MENU (how to set up a run) ===============================
# A ROUTE is one way of moving N (number of params). Pick which ones run in CONFIG["routes"] above;
# edit each route's grid HERE (the call args). The predicted n_params is printed per cell before any
# training so you can size a grid to the GPU first. Geometric (~2x) spacing in N gives the cleanest fit.
#
#   PARAMETER routes (plotted on the L(N) = test_CE-vs-N power-law curve):
#     grow_K_GL10   GROW K at a FIXED block GL(10): K=10,20,...; n_heads=K/10; n_gen=10*K (LINEAR in K).
#     blocks_K48    FIXED K=48, VARY the block GL(b): b=48,24,12,8,6; n_heads=K/b; n_gen=b*K
#                   (bigger block = FEWER heads = MORE params -- e.g. GL(48) is the largest model here).
#     grow_K        grow K at FIXED n_heads=4 (so the block grows with K; n_gen ~ K^2/4, quadratic).
#     grow_K_mup    grow_K + a twin per K with the muP Adam rate rule K_0/K on the hidden and output
#                   matrices (Yang et al. 2022, Table 3; F1/EXP-6 width-stability), kl_max=8*K.
#     blocksize     FIXED K=64, vary n_heads in {8,4,2} (= block GL(8),GL(16),GL(32)); same idea as
#                   blocks_K48 but parameterized by n_heads instead of GL(b).
#   INFERENCE routes (FLAT N -- plotted on a SEPARATE inference-capacity figure, NEVER on L(N)):
#     infer_T       n_e_steps in {1,2,4,8} at constant params.
#     infer_L       legacy depth alias; now tagged depth because per-layer modules add parameters.
#
# To add your own: call a route builder with a new grid and give it a key; add that key to CONFIG["routes"].
ROUTES: Dict[str, List[Dict[str, Any]]] = {
    **build_standard_routes(SWEEPS),
    "grow_K_GL10":   route_grow_k_fixed_block([20, 30, 40, 50, 60, 80, 100], block=10),
    
    "blocks_K48":    route_vary_block_fixed_k(48, [3, 6, 8, 12, 24]),
    # blocks_K48 follow-up (S1 window GL(3)..GL(24)) at the current BASELINE batch_size=64 => 491.52M
    # tokens/run, the MATCHED budget that removes the 2x Chinchilla D-slice confound vs grow_K_GL10.
    # Distinct keys/tags so scaling_analysis keeps these points separate from the 245.76M blocks_K48 run.
    "blocks_K48_2x":      route_vary_block_fixed_k(48, [3, 6, 8, 12, 24], tag="blocks_K48_2x"),
    
    # Arm 2a: the non-gauge capacity control. Keep block_glk so n_gen (= 48*b) matches each gauge cell,
    # but encode NON-structurally (encode_mode='per_token_additive' + pos_phi='none'): the learned
    # (V, n_gen) phi table drives an additive mean shift through a frozen readout, and Omega = I. Tests
    # whether raw phi-table capacity, minus the gl(g) structure, reproduces the S1 curve.
    "blocks_K48_ctrl_2x": route_vary_block_fixed_k(
        48, [3, 6, 8, 12, 24], tag="blocks_K48_ctrl_2x",
        extra_overrides={"encode_mode": "per_token_additive", "pos_phi": "none"}),
    
    "grow_K":        route_grow_k([20, 40, 60, 80, 100, 120], n_heads=4),
    "grow_K_mup":    route_grow_k_mup([20, 40, 60, 80, 100], n_heads=4, anchor_k=20),  # F1/EXP-6 (fixed vs muP rate)
    "blocksize":     route_blocksize(64, [8, 4, 2]),
    "infer_T":       route_inference_t([1, 2, 4, 8]),
    "infer_L":       route_inference_l([1, 2, 4, 6]),
}





# =============================================================================
# PARAMETER PREDICTION  -- size a grid to the GPU before committing to long runs.
# =============================================================================

def _scaling_knob_report(cfg: VFE3Config, name: str) -> Dict[str, object]:
    """Report a configured knob without inventing an effect on an inactive path."""
    if name == "embed_dim+n_layers":
        values = {key: getattr(cfg, key) for key in ("embed_dim", "n_layers")}
        return {"name": name, "active": True, "configured_value": values, "effective_value": values}
    is_named_field = isinstance(name, str)
    configured = getattr(cfg, name, None) if is_named_field else None
    active = is_named_field and not (
        name == "decode_tau" and cfg.decode_mode == "linear"
    )
    return {
        "name": name,
        "active": active,
        "configured_value": configured,
        "effective_value": configured if active else None,
    }


def predict_n_params(cfg: VFE3Config) -> Tuple[int, int]:
    r"""Predicted total ``n_params`` and ``n_gen`` for ``cfg``, by building only the (cheap) gauge group
    and summing the prior-table sizes per ``PriorBank`` (prior_bank.py). Exact on the pure path; CG / learnable-scalar tables remain outside this sizing model."""
    group = build_group(cfg)
    n_gen = int(group.generators.shape[0])
    V, K = int(cfg.vocab_size), int(cfg.embed_dim)
    n = V * n_gen + 1                                        # phi_embed, decode_log_scale
    n += V * K                                               # token-prior mean table
    n += V * K                                               # token-prior variance table
    if cfg.decode_mode == "linear":
        n += V * K                                          # output_proj_weight
        n += V                                              # output_proj_bias
    elif cfg.untie_decode_bank:
        n += 2 * V * K                                      # independent routed decode tables
    if cfg.pos_phi == "learned":
        n += int(cfg.max_seq_len) * n_gen                   # pos_phi_free
    if cfg.use_block_mlp:
        registration = get_block_mlp_registration(cfg.block_mlp_mode)
        n += registration.parameter_count(
            embed_dim=K,
            irrep_dims=group.irrep_dims,
            expansion=int(cfg.block_mlp_expansion),
            n_layers=1 if getattr(cfg, "block_mlp_tie_layers", False) else int(cfg.n_layers),   # tied: counted once
            regression_rank=getattr(cfg, "block_memory_regression_rank", None),    # cue_result_memory only
        )
    if cfg.layernorm_affine and cfg.norm_type_block == "layernorm":
        n += 2 * K
    n += int(cfg.n_layers) * 4 * K * K                       # full-width Q/K/T/O maps
    return n, n_gen


# =============================================================================
# LOADERS  -- memoised on the fields that actually change the stream (mirrors ablation.get_loader).
# =============================================================================
_LOADER_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _effective_data_seed(model_seed: int) -> Optional[int]:
    """Resolve fixed, per-model, or global-RNG training order without changing model seeds."""
    setting = CONFIG.get("data_seed", "model")
    if setting == "model":
        return _require_scaling_seed(model_seed)
    if setting is None:
        return None
    return _require_scaling_seed(setting, "data_seed")


def _require_scaling_seed(value: object, field: str = "seed") -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact nonnegative integer, got {value!r}")
    return value


def get_loader(
    dataset:    str,
    seq_len:    int,
    batch_size: int,
    split:      str,

    *,
    data_seed:   Optional[int] = None,
    max_tokens:  Optional[int] = None,
    vocab_size:  Optional[int] = None,
    eval_stride: Optional[int] = None,
) -> Any:
    r"""Split-aware DataLoader for ``dataset``/``split`` (a missing cache raises ``FileNotFoundError``).

    Memoized on ``(dataset, seq_len, batch_size, split, cap, vocab_size, data_seed, stride)`` so the
    corpus loads once across cells with the same data and sampling contract.
    Only the train stream shuffles / drops the partial last batch; validation/test read the whole split
    in a stable order so the held-out metric is a full-corpus measurement. ``max_tokens`` caps the train
    split only. No synthetic substitution for a missing real corpus.

    ``eval_stride`` mirrors ``train_vfe3._select_loader`` and ``ablation.get_loader``: it selects the
    sliding-window held-out evaluation for validation/test and is ignored for train. It belongs in the
    memo key for the same reason the other fields do -- a scaling route that varied it would otherwise
    be served the first cell's loader."""
    cap = max_tokens if split == "train" else None
    seeded = split == "train" and data_seed is not None
    validated_data_seed = _require_scaling_seed(data_seed, "data_seed") if seeded else None
    stride = None if split == "train" else eval_stride
    key = (dataset, seq_len, batch_size, split, cap, vocab_size,
           validated_data_seed, stride)
    if key not in _LOADER_CACHE:
        generator = torch.Generator().manual_seed(validated_data_seed) if seeded else None
        _LOADER_CACHE[key] = make_dataloader(dataset, split, seq_len, batch_size,
                                             shuffle=(split == "train"), drop_last=(split == "train"),
                                             max_tokens=cap, stride=stride, vocab_size=vocab_size,
                                             generator=generator)
    return _LOADER_CACHE[key]


# =============================================================================
# SINGLE-CELL EXECUTOR  -- one independent (size, seed) run (replicates _run_once's body).
# =============================================================================

_EMBEDDING_FIGURE_OVERRIDE_LOGGED: List[bool] = []


def _cell_cfg_dict(overrides: Dict[str, Any], seed: int, max_steps: Optional[int]) -> Dict[str, Any]:
    r"""The exact kwargs a cell's VFE3Config is built from: baseline + overrides + run knobs. Single
    source of truth, shared by ``run_cell`` and the resume staleness check."""
    d = copy.deepcopy(dict(BASELINE))
    d.update(overrides)
    d["seed"] = _require_scaling_seed(seed)
    d["checkpoint_interval"] = 0                             # no per-cell step_N.pt blowup
    # Skip only the slow semantic-geometry analysis. A per-cell override is honored; the launcher's
    # BASELINE value is not, which is logged once per process (audit 2026-10-07 F42).
    if "generate_embedding_figures" not in overrides:
        if d.get("generate_embedding_figures") and not _EMBEDDING_FIGURE_OVERRIDE_LOGGED:
            _EMBEDDING_FIGURE_OVERRIDE_LOGGED.append(True)
            logging.getLogger(__name__).warning(
                "scaling cells skip the semantic-geometry analysis: generate_embedding_figures=True "
                "in BASELINE is set to False per cell (override it per cell to keep it)")
        d["generate_embedding_figures"] = False
    if max_steps is not None:
        d["max_steps"] = int(max_steps)
    per_dim = CONFIG.get("kl_max_per_head_dim")
    if per_dim is not None and "kl_max" not in overrides:
        d["kl_max"] = _per_head_dim_kl_max(d, per_dim)
    return d


def _per_head_dim_kl_max(cell: Mapping[str, Any], per_dim: object) -> float:
    r"""The router cap :math:`\mathrm{kl\_max} = c \cdot d_\mathrm{head}` with :math:`d_\mathrm{head} = K / H`, of one cell.

    The routes that vary the head dimension at a fixed cap (``blocks_K48``: kl_max = 8K with
    :math:`d_\mathrm{head}` = 3 to 24; ``grow_K_GL{b}``: kl_max = 8K at fixed :math:`d_\mathrm{head}`) give
    their cells different per-dimension caps, so whether a cell can lose a head to the clamp depends on the
    cell (audit 2026-10-17 item 10). ``c`` is in nats per head dimension.
    """
    if isinstance(per_dim, bool) or not isinstance(per_dim, (int, float)) or not float(per_dim) > 0.0:
        raise ValueError(f"CONFIG['kl_max_per_head_dim'] must be None or a positive number, got {per_dim!r}")
    embed_dim = int(cell.get("embed_dim", _VFE3_DEFAULTS.get("embed_dim")))
    n_heads = int(cell.get("n_heads", _VFE3_DEFAULTS.get("n_heads")))
    if n_heads <= 0 or embed_dim % n_heads:
        raise ValueError(f"kl_max_per_head_dim needs n_heads dividing embed_dim, got K={embed_dim}, H={n_heads}")
    return float(per_dim) * (embed_dim // n_heads)


def _dead_router_heads(run_dir: Path) -> Dict[str, Any]:
    r"""Dead (layer, head) routers of one cell, read from ``diagnostics/cap_clamped_rows.csv`` when present.

    At the last recorded step, a head with more than ``DEAD_ROUTER_CLAMPED_ROW_FRAC`` of its scored rows fully
    clamped at kl_max routes those rows by the attention prior alone, with zero router gradient (the
    threshold the training loop warns on). Returns ``available`` (False without the file), ``step``,
    ``dead_heads`` (count), ``dead`` ([layer, head] pairs) and ``heads_recorded`` (audit 2026-10-17 item 10).
    """
    import csv

    from vfe3.train import DEAD_ROUTER_CLAMPED_ROW_FRAC

    path = run_dir / "diagnostics" / "cap_clamped_rows.csv"
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return {"available": False, "step": None, "dead_heads": None, "dead": [], "heads_recorded": 0}
    parsed = []
    for record in records:
        try:
            parsed.append((int(float(record["step"])), int(float(record["layer"])), int(float(record["head"])),
                           float(record["clamped_rows"]), float(record["rows"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not parsed:
        return {"available": False, "step": None, "dead_heads": None, "dead": [], "heads_recorded": 0}
    last = max(step for step, *_ in parsed)
    at_last = [row for row in parsed if row[0] == last]
    dead = sorted([layer, head] for _step, layer, head, clamped, rows in at_last
                  if rows > 0 and clamped / rows > DEAD_ROUTER_CLAMPED_ROW_FRAC)
    return {"available": True, "step": last, "dead_heads": len(dead), "dead": dead,
            "heads_recorded": len(at_last)}


def _current_code_identity() -> Dict[str, object]:
    """Current repository code identity used to validate a persisted scaling cell."""
    return _git_code_identity(Path(__file__).resolve().parent)


def _validated_scaling_code_identity(value: object) -> Dict[str, object]:
    """Return one detached, usable Git identity for a scaling invocation."""
    if not isinstance(value, Mapping):
        raise TypeError("scaling code identity must be a mapping")
    detached = json.loads(json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False))
    git_sha = detached.get("git_sha")
    git_dirty = detached.get("git_dirty")
    fingerprint = detached.get("git_dirty_fingerprint")
    if not isinstance(git_sha, str) or not git_sha or type(git_dirty) is not bool:
        raise ValueError("scaling code identity is unavailable")
    if ((git_dirty and (not isinstance(fingerprint, str) or not fingerprint))
            or (not git_dirty and fingerprint is not None)):
        raise ValueError("scaling code identity has an inconsistent dirty-tree fingerprint")
    return detached


_SCALING_SOURCE_SPLITS = ("train", "validation", "test")


def _validated_data_source_identities(value: object) -> Dict[str, Dict[str, object]]:
    """Return a detached, JSON-safe identity for exactly the splits scaling consumes."""
    if not isinstance(value, Mapping) or set(value) != set(_SCALING_SOURCE_SPLITS):
        raise ValueError(
            "scaling data source identities must contain exactly train, validation, and test")
    normalized: Dict[str, Dict[str, object]] = {}
    for split in _SCALING_SOURCE_SPLITS:
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


def _data_source_identities(dataset: str) -> Dict[str, Dict[str, object]]:
    """Current cached corpus identities for every split consumed by a scaling cell."""
    return _validated_data_source_identities({
        split: cache_source_identity(dataset, split)
        for split in _SCALING_SOURCE_SPLITS
    })


def _loader_data_source_identities(
    train_loader: object,
    val_loader:   object,
    test_loader:  object,

    *,
    dataset:    str,
    max_tokens: Optional[int],
) -> Dict[str, Dict[str, object]]:
    """Return source identities from the immutable loader datasets actually used by one cell."""
    sources: Dict[str, object] = {}
    for split, loader, expected_cap in (
        ("train", train_loader, max_tokens),
        ("validation", val_loader, None),
        ("test", test_loader, None),
    ):
        data_identity = getattr(getattr(loader, "dataset", None), "data_identity", None)
        if (not isinstance(data_identity, Mapping)
                or data_identity.get("schema_version") != 2
                or data_identity.get("dataset") != dataset
                or data_identity.get("split") != split
                or data_identity.get("max_tokens") != expected_cap):
            raise RuntimeError(f"{split} loader data identity is unavailable or mismatched")
        sources[split] = data_identity.get("source")
    return _validated_data_source_identities(sources)


def _scaling_reuse_contract_sha256(cellmeta: Mapping[str, object]) -> str:
    """Digest a scaling-cell reuse contract without its self-referential digest field."""
    contract = dict(cellmeta)
    contract.pop(_SCALING_REUSE_DIGEST_FIELD, None)
    return _canonical_json_sha256(contract)


def _cached_scaling_metrics(summary: Mapping[str, object]) -> Dict[str, object]:
    """Return the internally consistent metric payload served by the cached-cell path."""
    scaling_point = summary.get("scaling_point")
    if not isinstance(scaling_point, Mapping):
        return {}
    required = ("n_params", "test_ce", "test_ppl", "test_bits_per_token", "test_bpc")
    if any(key not in summary or key not in scaling_point for key in required):
        return {}
    if any(summary[key] != scaling_point[key] for key in required):
        return {}
    return {key: scaling_point[key] for key in required}


def _cell_resume_status(
    run_dir: Path,
    cfg:     VFE3Config,
    dataset: str,

    max_tokens: Optional[int] = None,

    *,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
) -> Optional[str]:
    r"""Classify one verified completed cell for restart handling.

    ``current`` means the effective config, corpus, and code identity all match this invocation.
    ``recorded_prior_code`` means the internally bound result has the same effective config and corpus
    but was produced under another code identity; restart protects and skips that completed record,
    while downstream analysis retains the saved provenance and can reject mixed-code aggregation.
    ``None`` means the cell is missing, partial, corrupt, or belongs to another effective experiment.
    Historical configs are compared only after the repository's strict serialized-config migration,
    so retired inert fields do not force a completed cell to be overwritten."""
    if (not (run_dir / "summary.json").exists()
            or not (run_dir / "config.json").exists()
            or (run_dir / "scaling_failure.json").exists()):
        return None
    try:
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        built = json.loads(json.dumps(serialize_config(cfg), default=str))
    except Exception:
        return None
    if (not isinstance(saved, Mapping) or not isinstance(summary, Mapping)
            or saved.get("dataset") != dataset
            or not isinstance(saved.get("config"), Mapping)):
        return None
    raw_saved_config = dict(saved["config"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            migrated = config_from_serialized(
                raw_saved_config,
                source=str(run_dir / "config.json"),
            )
        saved_effective = json.loads(json.dumps(serialize_config(migrated), default=str))
    except (TypeError, ValueError):
        return None
    if saved_effective != built:
        return None
    try:
        cellmeta = json.loads((run_dir / "scaling_cell.json").read_text(encoding="utf-8"))
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        if not isinstance(cellmeta, Mapping) or not isinstance(provenance, Mapping):
            return None
    except Exception:
        return None
    if cellmeta.get("schema_version") != _SCALING_CELL_SCHEMA_VERSION:
        return None
    if cellmeta.get("dataset") != dataset:
        return None
    if cellmeta.get("data_seed", cfg.seed) != _effective_data_seed(cfg.seed):
        return None
    try:
        # The cell binds the config.json it wrote. A cell written before a schema migration (a retired
        # field or value the loader migrates) hashed the saved form, so either the saved bytes or their
        # migrated form may carry the digest; equality of effective configs is checked above.
        saved_config_sha256 = {_canonical_json_sha256(saved_effective),
                               _canonical_json_sha256(raw_saved_config)}
    except (TypeError, ValueError):
        return None
    saved_contract_sha256 = cellmeta.get(_SCALING_REUSE_DIGEST_FIELD)
    if (not isinstance(saved_contract_sha256, str)
            or len(saved_contract_sha256) != 64
            or summary.get(_SCALING_SUMMARY_DIGEST_FIELD) != saved_contract_sha256
            or cellmeta.get("config_sha256") not in saved_config_sha256):
        return None
    if _scaling_result_status(_cached_scaling_metrics(summary)) != "complete":
        return None
    if cellmeta.get("max_tokens", None) != (int(max_tokens) if max_tokens is not None else None):
        return None
    saved_sources = cellmeta.get("data_sources")
    if not isinstance(saved_sources, Mapping):
        return None
    try:
        recorded = _validated_scaling_code_identity(cellmeta.get("code_identity"))
        current_sources = (
            _data_source_identities(dataset)
            if source_identities is None
            else _validated_data_source_identities(source_identities)
        )
        current = (
            _validated_scaling_code_identity(_current_code_identity())
            if code_identity is None
            else _validated_scaling_code_identity(code_identity)
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if dict(saved_sources) != current_sources:
        return None
    if any(provenance.get(key) != value for key, value in recorded.items()):
        return None
    return "current" if recorded == current else "recorded_prior_code"


def _cell_is_current(
    run_dir: Path,
    cfg:     VFE3Config,
    dataset: str,

    max_tokens: Optional[int] = None,

    *,
    source_identities: Optional[Mapping[str, object]] = None,
    code_identity:     Optional[Mapping[str, object]] = None,
) -> bool:
    """Return whether a completed cell matches the current effective experiment and code."""
    return _cell_resume_status(
        run_dir,
        cfg,
        dataset,
        max_tokens=max_tokens,
        source_identities=source_identities,
        code_identity=code_identity,
    ) == "current"


def _path_is_reparse_point(path: Path) -> bool:
    """Return whether ``path`` is a symlink, junction, or other filesystem reparse point."""
    junction_probe = getattr(os.path, "isjunction", lambda _path: False)
    if path.is_symlink() or junction_probe(path):
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _trusted_scaling_output_dir(path: Path) -> Path:
    """Create and return a real output directory reached without any reparse-point ancestor."""
    absolute = Path(os.path.abspath(path))
    existing_chain = [candidate for candidate in reversed(absolute.parents) if candidate.exists()]
    if absolute.exists():
        existing_chain.append(absolute)
    for candidate in existing_chain:
        if _path_is_reparse_point(candidate):
            raise ValueError(f"scaling output path crosses a symlink or junction: {candidate}")
    absolute.mkdir(parents=True, exist_ok=True)
    if not absolute.is_dir():
        raise ValueError("scaling output path must be a real directory")
    return absolute.resolve(strict=True)


def _trusted_scaling_run_dir(
    output_dir: Path,
    route:      str,
    label:      str,
    seed:       int,
) -> Path:
    """Return one real direct-child cell tree contained by ``output_dir``."""
    seed = _require_scaling_seed(seed)
    root = _trusted_scaling_output_dir(output_dir)
    components = (route, label, f"s{seed}")
    current = root
    for component in components:
        portable_path_component_key(component)
        candidate = current / component
        if candidate.exists() and (
                _path_is_reparse_point(candidate) or not candidate.is_dir()):
            raise ValueError(f"unsafe scaling cell path: {candidate}")
        candidate.mkdir(exist_ok=True)
        current = candidate.resolve(strict=True)
    try:
        current.relative_to(root)
    except ValueError as exc:
        raise ValueError("scaling cell escaped its owned output tree") from exc
    _authorize_scaling_cell(current, route, label, seed)
    return current


def _archive_scaling_design(design_path: Path) -> Optional[Path]:
    """Move a prior invocation manifest into append-only history before creating another."""
    if not design_path.exists():
        if design_path.is_symlink() or _path_is_reparse_point(design_path):
            raise ValueError("scaling design path may not be a symlink or reparse point")
        return None
    if (_path_is_reparse_point(design_path) or not design_path.is_file()
            or design_path.resolve(strict=True).parent != design_path.parent.resolve(strict=True)):
        raise ValueError("scaling design path must be a direct regular file")
    archive_root = design_path.parent / ".invocations"
    if archive_root.exists():
        if _path_is_reparse_point(archive_root) or not archive_root.is_dir():
            raise ValueError("scaling invocation archive must be a real directory")
    else:
        archive_root.mkdir()
    for index in range(1, 1_000_000):
        archived = archive_root / f"scaling_design__invocation_{index:04d}.json"
        if archived.exists() or archived.is_symlink() or _path_is_reparse_point(archived):
            continue
        design_path.rename(archived)
        return archived
    raise RuntimeError("scaling invocation archive exhausted its numeric namespace")


def _scaling_owner_payload(route: str, label: str, seed: int) -> Dict[str, object]:
    """Return the exact identity that authorizes scaling-cell attempt archival."""
    portable_path_component_key(route, field="scaling owner route")
    portable_path_component_key(label, field="scaling owner label")
    return {
        "schema_version": _SCALING_OWNER_SCHEMA_VERSION,
        "route":          route,
        "label":          label,
        "seed":           _require_scaling_seed(seed),
    }


def _scaling_owner_is_exact(
    value:    Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    """Whether a decoded owner has exactly the versioned identity schema and Python types."""
    return bool(
        set(value) == set(expected)
        and type(value.get("schema_version")) is int
        and type(value.get("route")) is str
        and type(value.get("label")) is str
        and type(value.get("seed")) is int
        and value == expected
    )


def _read_regular_scaling_json(path: Path, *, role: str) -> Dict[str, object]:
    """Read a direct regular, non-reparse JSON object without following a redirect."""
    if path.is_symlink() or _path_is_reparse_point(path) or not path.is_file():
        raise ValueError(f"{role} must be a regular non-reparse file")
    try:
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ValueError(f"{role} resolves outside its scaling cell")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must contain a JSON object")
    return dict(value)


def _authorize_scaling_cell(run_dir: Path, route: str, label: str, seed: int) -> Path:
    r"""Validate exact ownership or promote one exact regular legacy cell marker once."""
    expected = _scaling_owner_payload(route, label, seed)
    owner_path = run_dir / _SCALING_OWNER_FILENAME
    children = list(run_dir.iterdir())
    if not children:
        _write_json_atomic(owner_path, expected)
        return owner_path

    owner_present = (
        owner_path.exists()
        or owner_path.is_symlink()
        or _path_is_reparse_point(owner_path)
    )
    if owner_present:
        owner = _read_regular_scaling_json(
            owner_path,
            role="scaling cell ownership sentinel",
        )
        if not _scaling_owner_is_exact(owner, expected):
            raise ValueError(
                "scaling cell ownership sentinel does not match route, label, and seed"
            )
        return owner_path

    legacy_path = run_dir / "scaling_cell.json"
    try:
        legacy = _read_regular_scaling_json(
            legacy_path,
            role="legacy scaling cell ownership marker",
        )
    except ValueError as exc:
        raise ValueError(
            "nonempty scaling cell has no valid ownership sentinel or promotable legacy marker"
        ) from exc
    if (
        legacy.get("route") != route
        or legacy.get("label") != label
        or type(legacy.get("seed")) is not int
        or legacy.get("seed") != seed
    ):
        raise ValueError(
            "legacy scaling cell ownership marker does not match route, label, and seed"
        )
    _write_json_atomic(owner_path, expected)
    return owner_path


def _archive_scaling_attempt(
    run_dir: Path,

    *,
    route: str,
    label: str,
    seed:  int,
) -> Path:
    r"""Move one owned cell attempt intact into its append-only archive.

    A fresh ``RunArtifacts`` instance must never target a populated directory because its writers
    replace canonical filenames. Archiving the entire directory first preserves every runner-owned
    artifact and every user-added file without interpreting or deleting either class.
    """
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    if _path_is_reparse_point(run_dir):
        raise ValueError("scaling cell directory may not be a symlink or junction")
    if not run_dir.is_dir():
        raise ValueError("scaling cell path exists but is not a directory")
    _authorize_scaling_cell(run_dir, route, label, seed)
    archive_root = run_dir.parent / ".attempts"
    if archive_root.exists():
        if _path_is_reparse_point(archive_root) or not archive_root.is_dir():
            raise ValueError("scaling attempt archive must be a real directory")
    else:
        archive_root.mkdir()
    for index in range(1, 1_000_000):
        archived = archive_root / f"s{seed}__attempt_{index:04d}"
        if archived.exists() or archived.is_symlink() or _path_is_reparse_point(archived):
            continue
        run_dir.rename(archived)
        return archived
    raise RuntimeError("scaling attempt archive exhausted its numeric namespace")


# Failure kinds a run produces itself; under resume they are terminal unless CONFIG["retry_failed"] is set
# (audit 2026-10-17 item 28). Config and path failures are recomputed on every invocation.
_SCALING_TERMINAL_FAILURE_KINDS = frozenset({"train", "nonfinite"})


def _terminal_scaling_failure(
    run_dir: Path,
    cfg:     VFE3Config,

    *,
    route:   str,
    label:   str,
    seed:    int,
) -> Optional[Dict[str, Any]]:
    r"""The recorded run failure of this exact cell, or None.

    The cell's ``scaling_failure.json`` must name the route, label and seed and carry a run-produced
    ``error_kind`` (:data:`_SCALING_TERMINAL_FAILURE_KINDS`); when the attempt wrote ``scaling_cell.json``,
    its ``config_sha256`` must equal ``cfg``'s, so a changed cell is retrained rather than reported.
    """
    try:
        failure = json.loads((run_dir / "scaling_failure.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(failure, dict)
            or failure.get("error_kind") not in _SCALING_TERMINAL_FAILURE_KINDS
            or failure.get("route") != route or failure.get("label") != label
            or failure.get("seed") != int(seed)):
        return None
    try:
        cellmeta = json.loads((run_dir / "scaling_cell.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cellmeta = None
    if isinstance(cellmeta, Mapping):
        current = _canonical_json_sha256(json.loads(json.dumps(serialize_config(cfg), default=str)))
        if cellmeta.get("config_sha256") != current:
            return None
    return failure


def _scaling_attempt_history(run_dir: Path, seed: int) -> Dict[str, Any]:
    r"""Every attempt of one cell: the archived ones under ``.attempts`` and the current directory.

    Each attempt reports ``status``, ``error_kind`` and ``error`` from its ``scaling_result.json`` (or
    ``scaling_failure.json``); an archived attempt with neither was interrupted.
    """
    def read(directory: Path) -> Optional[Dict[str, Any]]:
        for name in ("scaling_result.json", "scaling_failure.json"):
            try:
                record = json.loads((directory / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(record, dict):
                return record
        return None

    archive_root = run_dir.parent / ".attempts"
    directories = (sorted(archive_root.glob(f"s{int(seed)}__attempt_*")) if archive_root.is_dir() else [])
    attempts: List[Dict[str, Any]] = []
    for directory in [*directories, run_dir]:
        record = read(directory)
        if record is None:
            if directory is run_dir:
                continue
            record = {"status": "interrupted", "error_kind": None,
                      "error": "the attempt published no scaling_result.json"}
        attempts.append({"attempt": len(attempts) + 1, "status": record.get("status"),
                         "error_kind": record.get("error_kind"), "error": record.get("error"),
                         "archive": None if directory is run_dir else directory.name})
    failed = [a for a in attempts if a["status"] not in ("complete", None)]
    return {"n_attempts": len(attempts), "n_failed": len(failed), "attempts": attempts}


def _scaling_cell_has_payload(run_dir: Path) -> bool:
    """Return whether a cell contains anything beyond its ownership sentinel."""
    return run_dir.is_dir() and any(
        path.name != _SCALING_OWNER_FILENAME for path in run_dir.iterdir()
    )


@restored_fp32_matmul_precision()         # a None cell must not inherit the previous cell's pinned matmul policy
def run_cell(
    cell:       Dict[str, Any],
    run_dir:    Path,
    seed:       int,

    *,
    dataset:            str,
    device:             torch.device,
    max_tokens:         Optional[int]                  = None,
    max_steps:          Optional[int]                  = None,
    source_identities:  Optional[Mapping[str, object]] = None,
    code_identity:      Optional[Mapping[str, object]] = None,
) -> Dict[str, Any]:
    r"""Build a fresh model from baseline+overrides, train it, score the held-out TEST split via
    ``finalize_run``, and return a harvest dict. A cross-field config violation is caught and returned
    as ``error_kind='config'`` (not raised), keeping it distinct from a training crash."""
    seed = _require_scaling_seed(seed)
    invocation_sources = (
        None if source_identities is None
        else _validated_data_source_identities(source_identities)
    )
    invocation_code_identity = (
        None if code_identity is None
        else _validated_scaling_code_identity(code_identity)
    )
    label = cell["label"]
    cfg_dict = _cell_cfg_dict(cell["overrides"], seed, max_steps)
    try:
        cfg = VFE3Config(**cfg_dict)
        _validate_cell_controls(cfg, cell, max_steps)
    except (ValueError, NotImplementedError, TypeError) as exc:
        preserve_existing = _scaling_cell_has_payload(run_dir)
        logger.warning("  [config rejected] %s: %s", label, exc)
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": "config", "error": str(exc), "seed": seed,
                "preserve_existing": preserve_existing,
                "test_ce": None, "test_ppl": None, "test_bits_per_token": None,
                "test_bpc": None, "n_params": None}

    resume_status: Optional[str] = None
    if CONFIG["resume"]:
        if _cell_is_current(
            run_dir, cfg, dataset, max_tokens=max_tokens,
            source_identities=invocation_sources,
            code_identity=invocation_code_identity,
        ):
            resume_status = "current"
        else:
            resume_status = _cell_resume_status(
                run_dir, cfg, dataset, max_tokens=max_tokens,
                source_identities=invocation_sources,
                code_identity=invocation_code_identity,
            )
    if resume_status is not None:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        metrics = _cached_scaling_metrics(summary)
        marker = "CACHED" if resume_status == "current" else "RECORDED"
        suffix = (
            ""
            if resume_status == "current"
            else "  (complete prior-code result preserved; provenance retained)"
        )
        print(f"    [{marker}] {label} s{seed}  test_ce={metrics['test_ce']}  "
              f"N={metrics['n_params']}{suffix}")
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": None, "seed": seed, "cached": True,
                "cache_status": resume_status, "kl_max": float(cfg.kl_max),
                "router_dead_heads": _dead_router_heads(run_dir),
                **metrics}
    terminal = (None if not CONFIG["resume"] or CONFIG.get("retry_failed", False) else
                _terminal_scaling_failure(run_dir, cfg, route=cell["route"], label=label, seed=seed))
    if terminal is not None:
        print(f"    [FAILED earlier, not retried] {label} s{seed}  {terminal.get('error_kind')}: "
              f"{terminal.get('error')}  (set retry_failed=True to retrain)")
        return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
                "error_kind": terminal.get("error_kind"), "error": terminal.get("error"), "seed": seed,
                "preserve_existing": True, "terminal_failure": True,
                "test_ce": None, "test_ppl": None, "test_bits_per_token": None,
                "test_bpc": None, "n_params": None}

    if _scaling_cell_has_payload(run_dir):
        archived = _archive_scaling_attempt(
            run_dir,
            route=cell["route"],
            label=label,
            seed=seed,
        )
        print(f"    [ARCHIVED] {label} s{seed} previous attempt preserved at {archived}")
        run_dir.mkdir()
        _authorize_scaling_cell(run_dir, cell["route"], label, seed)

    pred_n, n_gen = predict_n_params(cfg)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    actual_n = int(sum(p.numel() for p in model.parameters()))
    gap = "" if actual_n == pred_n else f"  (predicted {pred_n:,}; +{actual_n - pred_n:,} small modules)"
    print(f"    {label} s{seed} | K={cfg.embed_dim} h={cfg.n_heads} {cfg.gauge_group} "
          f"n_gen={n_gen} | N={actual_n:,}{gap} | steps={cfg.max_steps}")

    data_seed = _effective_data_seed(cfg.seed)
    train_loader = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "train",
                              data_seed=data_seed, max_tokens=max_tokens, vocab_size=cfg.vocab_size)
    val_loader   = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "validation",
                              vocab_size=cfg.vocab_size, eval_stride=cfg.eval_stride)
    test_loader  = get_loader(dataset, cfg.max_seq_len, cfg.batch_size, "test",
                              vocab_size=cfg.vocab_size, eval_stride=cfg.eval_stride)

    if invocation_sources is not None:
        loaded_sources = _loader_data_source_identities(
            train_loader,
            val_loader,
            test_loader,
            dataset=dataset,
            max_tokens=max_tokens,
        )
        if loaded_sources != invocation_sources:
            raise RuntimeError(
                "scaling corpus identity drifted between the invocation snapshot and loader build")
    else:
        # Direct API callers retain the historical behavior: take one fresh source snapshot for this
        # standalone cell. ``main`` always supplies its invocation-owned shared snapshot instead.
        loaded_sources = _data_source_identities(dataset)
    cell_code_identity = (
        _validated_scaling_code_identity(_current_code_identity())
        if invocation_code_identity is None else invocation_code_identity
    )

    # Order-INDEPENDENT data stream: model build consumed config-dependent RNG, so reseed AFTER it and
    # re-seed each loader's generator so every cell sees the same batch sequence regardless of grid
    # position (per-seed variance is then init/optimization variance, not a data-order artifact).
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    for loader in (train_loader, val_loader, test_loader):
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(data_seed if loader is train_loader else cfg.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device,
                             timestamp=datetime.now().isoformat(timespec="seconds"))
    # Cell provenance the per-run config.json does not carry: which ROUTE / scale knob produced this N.
    cellmeta: Dict[str, object] = {
        "schema_version": _SCALING_CELL_SCHEMA_VERSION,
        "label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
        "scale_knob_report": _scaling_knob_report(cfg, cell["scale_knob"]),
        "overrides": {key: serialize_config(cfg)[key] for key in cell["overrides"]},
        "predicted_n_params": pred_n, "n_gen": n_gen, "seed": seed,
        "max_tokens": (int(max_tokens) if max_tokens is not None else None),
        "dataset": dataset,
        "data_seed": data_seed,
        "config_sha256": _canonical_json_sha256(
            json.loads(json.dumps(serialize_config(cfg), default=str))),
        "code_identity": cell_code_identity,
        "data_sources": loaded_sources,
    }
    reuse_contract_sha256 = _scaling_reuse_contract_sha256(cellmeta)
    cellmeta[_SCALING_REUSE_DIGEST_FIELD] = reuse_contract_sha256
    artifacts.save_json("scaling_cell.json", cellmeta)

    val_tpc = _tokens_per_char(dataset, "validation")
    test_tpc = _tokens_per_char(dataset, "test")
    t0 = time.perf_counter()
    losses = train(model, train_loader, cfg, n_steps=cfg.max_steps,
                   grad_clip=cfg.grad_clip,
                   log_interval=cfg.log_interval, eval_interval=cfg.eval_interval,
                   val_loader=val_loader, tokens_per_char=val_tpc, device=device,
                   logger=logger, artifacts=artifacts, generate_samples=False)
    wall = time.perf_counter() - t0
    results = finalize_run(
        model,
        artifacts,
        cfg,
        tokens_per_char=test_tpc,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        losses=losses,
        data_seed=data_seed,
        max_tokens=max_tokens,
        tokenizer_tag=_tokenizer_tag(dataset),
        device=device,
        wall_time=wall,
        logger=logger,
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise RuntimeError("scaling summary must be a JSON object")
    bound_summary = dict(summary)
    bound_summary[_SCALING_SUMMARY_DIGEST_FIELD] = reuse_contract_sha256
    _write_json_atomic(summary_path, bound_summary)
    return {"label": label, "route": cell["route"], "scale_knob": cell["scale_knob"],
            "error_kind": None, "seed": int(cfg.seed), "cached": False,
            "test_ce": results.get("test_ce"), "test_ppl": results.get("test_ppl"),
            "test_bits_per_token": results.get("test_bits_per_token"),
            "test_bpc": results.get("test_bpc"),
            "n_params": actual_n, "n_gen": n_gen, "wall_time_s": wall,
            "kl_max": float(cfg.kl_max), "router_dead_heads": _dead_router_heads(run_dir)}


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# MAIN  (click-to-run; edit CONFIG / ROUTES above)
# =============================================================================

def _validate_cell_controls(
    cfg: VFE3Config,
    cell: Dict[str, Any],
    max_steps: Optional[int],
) -> None:
    """Reject sweep axes erased by a different active update or global budget.

    A cell is refused when one of its overrides is inert under the cell's own resolved config
    (``VFE3Config.inactive_fields``, the rule ``ablation.py`` applies to its arms), since that axis would
    not change what the cell computes. A control arm without an inert override is admitted.
    """
    inactive = cfg.inactive_fields()
    inert = sorted(set(cell["overrides"]) & set(inactive))
    if inert:
        detail = "; ".join(f"{name} ({inactive[name]})" for name in inert)
        raise ValueError(f"cell {cell['label']!r} overrides inert field(s), so its sweep axis would "
                         f"not change the run: {detail}")
    if cell["scale_knob"] == "max_steps" and max_steps is not None:
        raise ValueError("set CONFIG['max_steps']=None for the train_steps sweep; "
                         "a global override would erase its budget axis")


def validate_routes(route_names: Optional[List[str]] = None) -> None:
    """Validate selected cells and their effective overrides before execution."""
    errors: List[Tuple[str, str, str]] = []
    for route_name in ROUTES if route_names is None else route_names:
        for cell in ROUTES[route_name]:
            try:
                cfg = VFE3Config(**_cell_cfg_dict(cell["overrides"], 0, CONFIG["max_steps"]))
                _validate_cell_controls(cfg, cell, CONFIG["max_steps"])
            except (TypeError, ValueError) as exc:
                errors.append((route_name, str(cell["label"]), str(exc)))
    if errors:
        detail = "\n".join(f"  route {route!r}, arm {label!r}: {error}"
                           for route, label, error in errors)
        raise ValueError(f"scaling route construction failed:\n{detail}")


def preview_routes(route_names: List[str], seeds: List[int]) -> None:
    """Print effective dimensions, estimated parameters, and token exposure without models/data."""
    print("Scaling plan (parameter estimates; exact counts are saved during training)")
    print("route / cell | K | heads | layers | steps | est. params | tokens/run")
    runs = 0
    for name in route_names:
        for cell in ROUTES[name]:
            cfg = VFE3Config(**_cell_cfg_dict(cell["overrides"], seeds[0], CONFIG["max_steps"]))
            params, _ = predict_n_params(cfg)
            tokens = cfg.max_steps * cfg.batch_size * cfg.max_seq_len
            print(f"{name}/{cell['label']} | {cfg.embed_dim} | {cfg.n_heads} | {cfg.n_layers} | "
                  f"{cfg.max_steps:,} | {params:,} | {tokens:,}")
            runs += len(seeds)
    print(f"{runs} runs; seeds={seeds}; data_seed={CONFIG.get('data_seed', 'model')!r}")
    print("dry_run=True: no training or output directories; set it False to execute.")


def _validated_scaling_seeds(raw_seeds: object) -> List[int]:
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
        raise ValueError("CONFIG['seeds'] must be a non-empty list or tuple of unique integers")
    if any(type(seed) is not int for seed in raw_seeds):
        raise ValueError("CONFIG['seeds'] must contain exact integers")
    seeds = list(raw_seeds)
    if any(seed < 0 for seed in seeds):
        raise ValueError("CONFIG['seeds'] must contain nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"CONFIG['seeds'] must be unique, got {seeds!r}")
    return seeds


def _scaling_path_component_key(value: str, *, field: str) -> str:
    """Validate one portable directory name and return its collision key."""
    return portable_path_component_key(value, field=field)


def _validated_scaling_routes(raw_routes: object) -> List[str]:
    if not isinstance(raw_routes, (list, tuple)) or not raw_routes:
        raise ValueError("CONFIG['routes'] must be a non-empty list or tuple of unique route names")
    if any(not isinstance(name, str) or not name for name in raw_routes):
        raise ValueError("CONFIG['routes'] must contain nonempty strings")
    route_names = list(raw_routes)
    if len(set(route_names)) != len(route_names):
        raise ValueError(f"CONFIG['routes'] must be unique, got {route_names!r}")
    route_keys = [
        _scaling_path_component_key(name, field="scaling route name")
        for name in route_names
    ]
    if len(set(route_keys)) != len(route_keys):
        raise ValueError(
            f"CONFIG['routes'] must not contain filesystem aliases, got {route_names!r}"
        )
    for name in route_names:
        if name not in ROUTES:
            raise ValueError(f"unknown route {name!r}; choose from {sorted(ROUTES)}")
        cells = ROUTES[name]
        if not cells:
            raise ValueError(f"route {name!r} must contain at least one scaling cell")
        labels = [cell.get("label") if isinstance(cell, Mapping) else None for cell in cells]
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError(f"route {name!r} cell labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError(f"route {name!r} cell labels must be unique, got {labels!r}")
        label_keys = [
            _scaling_path_component_key(label, field=f"route {name!r} cell label")
            for label in labels
        ]
        if len(set(label_keys)) != len(label_keys):
            raise ValueError(
                f"route {name!r} cell labels must not contain filesystem aliases, "
                f"got {labels!r}"
            )
    return route_names


def _scaling_design(route_names: List[str], seeds: List[int]) -> Dict[str, Any]:
    cells = []
    for route_name in route_names:
        for cell in ROUTES[route_name]:
            for seed in seeds:
                cells.append({
                    "route": route_name,
                    "label": cell["label"],
                    "seed": seed,
                    "scale_knob": cell["scale_knob"],
                    "run_dir": f"{route_name}/{cell['label']}/s{seed}",
                    "status": "pending",
                    "effective_config": serialize_config(VFE3Config(
                        **_cell_cfg_dict(cell["overrides"], seed, CONFIG["max_steps"]))),
                    "data_seed": _effective_data_seed(seed),
                })
    return {
        "schema_version": 1,
        "routes": list(route_names),
        "seeds": list(seeds),
        "status": "pending",
        "cells": cells,
    }


def _scaling_result_status(result: Mapping[str, Any]) -> str:
    if result.get("error_kind") is not None:
        return "failed"

    def _finite_positive(value: object) -> bool:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(float(value)) and float(value) > 0.0)

    if type(result.get("n_params")) is not int or result["n_params"] <= 0:
        return "nonfinite"
    required_metrics = ("test_ce", "test_ppl", "test_bits_per_token", "test_bpc")
    if any(key not in result for key in required_metrics):
        return "nonfinite"
    test_ce = result["test_ce"]
    test_ppl = result["test_ppl"]
    test_bits = result["test_bits_per_token"]
    test_bpc = result["test_bpc"]
    if not all(_finite_positive(value) for value in (test_ce, test_bits)):
        return "nonfinite"
    if not (isinstance(test_ppl, (int, float)) and not isinstance(test_ppl, bool)
            and float(test_ppl) > 0.0 and not math.isnan(float(test_ppl))):
        return "nonfinite"
    if test_bpc is not None and not _finite_positive(test_bpc):
        return "nonfinite"
    expected_bits = float(test_ce) / math.log(2.0)
    if (not perplexity_matches_ce(test_ce, test_ppl)
            or not math.isclose(float(test_bits), expected_bits, rel_tol=1e-9, abs_tol=1e-12)):
        return "nonfinite"
    return "complete"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    route_names = _validated_scaling_routes(CONFIG["routes"])
    seeds = _validated_scaling_seeds(CONFIG["seeds"])
    validate_routes(route_names)
    for seed in seeds:
        _effective_data_seed(seed)
    if CONFIG.get("dry_run", False):
        preview_routes(route_names, seeds)
        return 0
    try:
        invocation_code_identity = _validated_scaling_code_identity(
            _current_code_identity())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("scaling code identity snapshot failed before execution: %s", exc)
        return 1
    try:
        invocation_sources = _validated_data_source_identities(
            _data_source_identities(CONFIG["dataset"]))
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("scaling source identity snapshot failed before execution: %s", exc)
        return 1
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if CONFIG["device"] == "auto" else torch.device(CONFIG["device"]))
    try:
        output_dir = _trusted_scaling_output_dir(Path(CONFIG["output_dir"]))
    except (OSError, ValueError) as exc:
        logger.error("unsafe scaling output directory: %s", exc)
        return 1

    design = _scaling_design(route_names, seeds)
    design_path = output_dir / "scaling_design.json"
    try:
        archived_design = _archive_scaling_design(design_path)
        if archived_design is not None:
            print(f"  [ARCHIVED] previous scaling design preserved at {archived_design}")
        _write_json_atomic(design_path, design)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("scaling design publication refused to overwrite prior data: %s", exc)
        return 1
    design_cells = {
        (cell["route"], cell["label"], int(cell["seed"])): cell
        for cell in design["cells"]
    }
    incomplete = False
    n_cells = sum(len(ROUTES[n]) for n in route_names)
    print(f"\nVFE_3.0 parameter-scaling suite\n  device:  {device}\n  dataset: {CONFIG['dataset']}"
          f"\n  output:  {output_dir}\n  seeds:   {seeds}\n  routes:  {', '.join(route_names)}"
          f"\n  total:   {n_cells} cells x {len(seeds)} seeds = {n_cells * len(seeds)} runs")

    for name in route_names:
        cells = ROUTES[name]
        print(f"\n{'=' * 70}\nROUTE: {name}  ({len(cells)} cells x {len(seeds)} seeds)\n{'=' * 70}")
        for cell in cells:
            for seed in seeds:
                try:
                    run_dir = _trusted_scaling_run_dir(
                        output_dir, name, cell["label"], int(seed)
                    )
                    res = run_cell(cell, run_dir, int(seed), dataset=CONFIG["dataset"], device=device,
                                   max_tokens=CONFIG["max_tokens"], max_steps=CONFIG["max_steps"],
                                   source_identities=invocation_sources,
                                   code_identity=invocation_code_identity)
                except Exception as exc:                     # a training crash must not kill the suite
                    logger.exception("route %s / %s s%d crashed", name, cell["label"], seed)
                    res = {"label": cell["label"], "route": name, "error_kind": "train",
                           "error": str(exc), "seed": int(seed), "test_ce": None,
                           "test_ppl": None, "test_bits_per_token": None,
                           "test_bpc": None, "n_params": None}
                finally:
                    _cleanup()
                status = _scaling_result_status(res)
                published = dict(res)
                published["status"] = status
                if status == "nonfinite":
                    if published.get("error_kind") is None:
                        published["error_kind"] = "nonfinite"
                    if published.get("error") is None:
                        published["error"] = (
                            "scaling result is incomplete, non-finite, non-positive, or inconsistent")
                persist_result = not res.get("cached") and not res.get("preserve_existing")
                if persist_result:
                    try:
                        run_dir = _trusted_scaling_run_dir(
                            output_dir, name, cell["label"], int(seed)
                        )
                        _write_json_atomic(run_dir / "scaling_result.json", published)
                        if status != "complete":
                            _write_json_atomic(run_dir / "scaling_failure.json", published)
                            incomplete = True
                    except (OSError, ValueError) as exc:
                        published["status"] = "failed"
                        published["error_kind"] = "path"
                        published["error"] = str(exc)
                        status = "failed"
                        incomplete = True
                        logger.error(
                            "refusing per-cell publication outside the owned scaling tree: %s", exc
                        )
                elif status != "complete":
                    incomplete = True
                manifest_cell = design_cells[(name, cell["label"], int(seed))]
                try:
                    history = _scaling_attempt_history(
                        _trusted_scaling_run_dir(output_dir, name, cell["label"], int(seed)), int(seed))
                except (OSError, ValueError):
                    history = {"n_attempts": None, "n_failed": None, "attempts": []}
                manifest_cell.update({
                    "status": status,
                    "error_kind": published.get("error_kind"),
                    "error": published.get("error"),
                    "cache_status": published.get("cache_status"),
                    "terminal_failure": bool(published.get("terminal_failure")),
                    "kl_max": published.get("kl_max"),
                    "router_dead_heads": published.get("router_dead_heads"),
                    "n_attempts": history["n_attempts"],
                    "n_failed_attempts": history["n_failed"],
                    "attempts": history["attempts"],
                })
                design["status"] = "incomplete" if incomplete else "running"
                _trusted_scaling_output_dir(output_dir)
                _write_json_atomic(design_path, design)
                if status == "complete" and not res.get("cached"):
                    print(f"      -> test_ce={res.get('test_ce')}  ppl={res.get('test_ppl')}  "
                          f"({res.get('wall_time_s', 0.0):.0f}s)")
                dead = res.get("router_dead_heads") or {}
                if dead.get("dead_heads"):
                    print(f"      -> {dead['dead_heads']} dead router head(s) at kl_max={res.get('kl_max')}: "
                          f"{dead['dead']} (cap_clamped_rows.csv, step {dead['step']})")

    invocation_errors: List[str] = []
    try:
        terminal_sources = _validated_data_source_identities(
            _data_source_identities(CONFIG["dataset"]))
        if terminal_sources != invocation_sources:
            invocation_errors.append(
                "data source identities drifted during the scaling invocation")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        invocation_errors.append(f"terminal data source identity snapshot failed: {exc}")
    try:
        terminal_code_identity = _validated_scaling_code_identity(_current_code_identity())
        if terminal_code_identity != invocation_code_identity:
            invocation_errors.append(
                "code identity drifted during the scaling invocation")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        invocation_errors.append(f"terminal code identity snapshot failed: {exc}")
    if invocation_errors:
        incomplete = True
        design["error"] = "; ".join(invocation_errors)

    design["status"] = "incomplete" if incomplete else "complete"
    design["retry_failed"] = bool(CONFIG.get("retry_failed", False))
    _trusted_scaling_output_dir(output_dir)
    _write_json_atomic(design_path, design)
    from scaling_analysis import print_design_attempts

    print_design_attempts(design)              # failed and repeated attempts (audit 2026-10-17 item 28)
    if incomplete:
        print(f"\nROUTES INCOMPLETE. Inspect {design_path} and per-cell scaling_failure.json files.")
        return 1
    print(f"\nALL ROUTES COMPLETE. Aggregate + fit + plot with:  python scaling_analysis.py"
          f"  (reads {output_dir}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
