"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly, then run.
Every registry-backed seam is selected here by name (belief family, encode/decode mode, alpha
form, attention prior, norm), so a variant swaps without editing call sites.

The projected routes fix the divergence (KL), the connection (flat), the gradient roles
(filtering), the E-step estimator (unroll), the E-step update (one Fisher step or the closed-form
receiver), the mean preconditioner (Fisher) and the mean trust region (Mahalanobis ball). Those
former selectors are not fields; ``config_from_serialized`` migrates or refuses them on load.
"""

import math
import os
import warnings
from dataclasses import MISSING, asdict, dataclass, fields
from numbers import Real
from typing import Any, List, Literal, Mapping, Optional, Tuple


class ConfigNotice(UserWarning):
    r"""An advisory that a setting is INERT -- read by no active path under this config.

    Separate from the other config warnings ON PURPOSE. These say "this value does nothing here",
    which matters when you are composing a config and is pure noise on every run afterwards. The
    correctness advisories -- a detached E-step freezing a parameter, a diagonal family silently
    truncating a congruence, an auto-enabled override -- stay ordinary ``UserWarning`` and are NOT
    suppressed by the switch below, because those change what the run MEANS rather than what it
    ignores.

    Subclasses ``UserWarning``, so anything already filtering or asserting on ``UserWarning``
    (such as ``pytest.warns``) keeps matching without changes. Sweep validation reads
    ``inactive_fields()`` directly.
    """


# Console visibility of ConfigNotice. Default HIDDEN: on a settled config these fire on every single
# construction -- every ablation arm, every worker -- and say nothing new after the first read.
# Suppression is a DISPLAY filter, not a removal: ``warnings.catch_warnings`` with an explicit
# ``simplefilter`` still sees them. Sweep admission uses the structured activity map,
# independently of this display setting.
#
# Enable with ``VFE3_CONFIG_NOTICES=1`` in the environment, or ``set_config_notices(True)`` before
# constructing the config.
_CONFIG_NOTICES_SHOWN: bool = os.environ.get(
    "VFE3_CONFIG_NOTICES", "").strip().lower() in ("1", "true", "yes", "on")


def set_config_notices(shown: bool) -> bool:
    r"""Show or hide :class:`ConfigNotice` on the console; returns the previous setting."""
    global _CONFIG_NOTICES_SHOWN
    previous = _CONFIG_NOTICES_SHOWN
    _CONFIG_NOTICES_SHOWN = bool(shown)
    _apply_config_notice_filter()
    return previous


def config_notices_shown() -> bool:
    r"""Whether :class:`ConfigNotice` advisories currently reach the console."""
    return _CONFIG_NOTICES_SHOWN


def _apply_config_notice_filter() -> None:
    r"""Install the display filter for :class:`ConfigNotice` matching the current setting."""
    warnings.filterwarnings(
        "default" if _CONFIG_NOTICES_SHOWN else "ignore", category=ConfigNotice)


_apply_config_notice_filter()

# Seams with a live registry (lambda_alpha_mode, attention priors, norms, positional frames) are
# validated against that registry in __post_init__ via tuple(sorted(_REGISTRY)), so a newly
# registered variant is a valid config value WITHOUT editing this validator (the add-by-registering
# modularity contract). Positional
# phi always uses exact group-product composition. Encode/decode modes likewise use their
# registries. The static tuples below are the remaining seams WITHOUT a registry.
_VALID_GAUGE_TRANSPORT     = ("on", "off", "frozen")
# Supported structure groups on the projected routes (so_n takes group_n and irrep_spec, one irrep
# block per head, blocks may differ in size). The retired names are refused.
_RETIRED_GAUGE_GROUPS      = frozenset({"glk", "tied_block_glk", "so_k", "sp", "sp_n"})
# Decoders decode_coordinates='reference' admits: linear reads xi = U^{-1} mu; the full Gaussian-KL
# kernel reads (xi, C); the diagonal Gaussian-KL kernel reads (xi, diag C), exact up to a per-position
# constant against a diagonal bank.
_REFERENCE_DECODE_MODES    = ("linear", "diagonal_chunked", "full_chunked")
# Retired dense decode spellings and the chunked kernels they migrate to on load (same logits; the
# dense helpers remain private oracles in vfe3.model.prior_bank).
_DENSE_DECODE_MIGRATIONS   = {"diagonal": "diagonal_chunked", "full": "full_chunked", "family": "family_chunked"}


class _ConfigTypeError(TypeError, ValueError):
    """Configuration type error compatible with legacy ``ValueError`` callers."""


@dataclass
class VFE3Config:
    """The single authoritative configuration surface for VFE_3.0.

    Fields are grouped in 19 numbered sections: run and training loop; logging and checkpoints; model
    shape; belief family and initialization; encode/decode; gauge group; positional encoding; attention
    routing; predictive readout; content links; projected-gradient receiver; free-energy weights;
    normalization and handoff; block MLP; E-step; frame M-step; learning rates; weight decay; numerics.
    Registry-backed fields accept every registered key. Validation runs once in ``__post_init__``.
    """

    # Layout: one banner per group. A categorical field's options are listed on the first comment line
    # above it ("a" | "b" | ...), followed by one line per option where the name is not self-explanatory.
    # "only:" notes name the settings under which a field is read; elsewhere it is inert (ConfigNotice).

    ##################################################
    #   1. Run, data and training loop
    ##################################################

    vocab_size:                       int             = 50257
    
    max_seq_len:                      int             = 128          # N, context length
    batch_size:                       int             = 64
    
    max_steps:                        int             = 15000        # cosine-schedule horizon
    warmup_steps:                     int             = 100
   
    min_lr_frac:                      float           = 0.01         # LR floor as a fraction of each group's base LR (0 = decay to 0)
    
    grad_accum_steps:                 int             = 1            # microbatches per optimizer step (1 = no accumulation)
    grad_clip:                        Optional[float] = 1.0          # per-role L2 clip after accumulation; None or 0 = off

    seed:                             int             = 6
    
    # deterministic only: fill uninitialized allocations with NaN so stray reads are visible. False
    # skips the fill kernels (about 13.5 ms per K=64 step on the RTX 5090; loss and gradients identical).
    fill_uninitialized_memory:        bool            = True
    deterministic:                    bool            = True         # torch/cuDNN determinism (TF32: see fp32_matmul_precision)
    
    ##################################################
    #   2. Logging, evaluation and checkpoints
    ##################################################

    log_interval:                     int             = 100          # console log every N steps (0 = off)
    eval_interval:                    int             = 1500         # validation every N steps (0 = off)
    checkpoint_interval:              int             = 15000        # resumable checkpoint every N steps (0 = off)
    val_diagnostic_batches:           int             = 1             # validation batches the val_* probes average over
    
    eval_stride:                      Optional[int]   = None         # held-out window stride <= max_seq_len; None = disjoint windows
               
    generate_embedding_figures:       bool            = False        # slow semantic-geometry figures at finalize

    # Resume from a checkpoints/step_<N>.pt (weights, optimizer, RNG, schedule). None = train from scratch.
    # An explicit train(resume_from=...) argument takes precedence.
    resume_from:                      Optional[str]   = None
    
    # Allow the unsafe weights_only=False load when the safe load of a (legacy) bundle fails. Only for
    # checkpoints you trust: that path can execute code embedded in the pickle.
    trust_resume_checkpoint:          bool            = False

    ##################################################
    #   3. Model shape
    ##################################################

    embed_dim:                        int             = 64           # K, belief dimension
    n_heads:                          int             = 2            # head blocks (irrep-tower groups: one irrep block per head)
    n_layers:                         int             = 1            # L, number of blocks
    n_e_steps:                        int             = 1            # T, E-step iterations per block (0 = zero-depth control, warns)

    ##################################################
    #   4. Belief family and table initialization
    ##################################################

    # "gaussian_diagonal" | "gaussian_full" | "laplace_diagonal"
    # The belief covariance structure (read diagonal_covariance, never a second flag). 
    family:                           str             = "gaussian_diagonal"

    # Token tables: mu_v ~ N(0, mu_init_std^2), every variance = sigma_init (> 0), phi_v ~ N(0, phi_scale^2).
    mu_init_std:                      float           = 0.065
    sigma_init:                       float           = 15
    phi_scale:                        float           = 0.035
    pos_phi_scale:                    float           = 0.02         # pos_phi: learned-table init scale | frozen per-position step

    ##################################################
    #   5. Encode and decode
    ##################################################

    # "per_token" | "per_token_additive"
    #   per_token           table lookup (mu_v, sigma_v, phi_v) as the token belief
    #   per_token_additive  control: the same phi table used additively, not as a frame (no position in phi)
    encode_mode:                      str             = "per_token"

    # "linear" | "diagonal_chunked" | "full_chunked" | "family_chunked" | "expected_likelihood_chunked"
    #   linear                       learned (V, K) readout of mu_q (+ bias); ignores sigma
    #   diagonal_chunked             -KL(q || p_v)/tau against the diagonal prior bank
    #   full_chunked                 -KL(q || p_v)/tau with a full q against the diagonal bank
    #   family_chunked               -D(q || p_v)/tau through the configured family (any covariance rank)
    #   expected_likelihood_chunked  log N(mu_q; mu_v, Sigma_q + Sigma_v), diagonal only
    # Every prior-bank mode streams the vocabulary in chunks and never forms (B, N, V) in training.
    decode_mode:                      str             = "linear"

    # "local" | "reference"
    #   local      decode the local moments (mu_i, Sigma_i)
    #   reference  decode xi_i = U_i^-1 mu_i, C_i = U_i^-1 Sigma_i U_i^-T through the attention frame.
    #              only: a projected attention_mode, pos_rotation="none", decode_mode "linear" |
    #              "diagonal_chunked" | "full_chunked"
    decode_coordinates:               str             = "local"

    decode_unigram_prior:             bool            = True         # add unigram_kappa * log pi_v (corpus unigram) to the logits
    decode_bias:                      bool            = True         # linear only: learned per-vocab bias (zero init, no decay)
    untie_decode_bank:                bool            = True         # prior-bank only: decode reads its own copy of the (V, K) tables

    decode_tau:                       float           = 1.0          # prior-bank decode temperature (1 = Bayes); not read by linear
    unigram_kappa:                    float           = 1.0          # tempering of log pi_v (1 = exact Bayes class prior)
    
    # Vocabulary entries per chunk of the fused CE kernels. The family_chunked full-covariance route
    # narrows its slice to a fixed byte budget and treats this as an upper bound. Values are unchanged.
    decode_chunk_size:                int             = 8192

    # "auto" | "auto_aggregate"
    #   auto            checkpoint the decoder by per-chunk memory accounting (2 GiB threshold)
    #   auto_aggregate  account the whole vocabulary, including fp64 scores (saves more memory)
    decode_ce_checkpoint:             str             = "auto"  

    # decode_mode="linear" with amp_dtype set only: run the logits matmul x W^T under amp_dtype
    # autocast (bf16/fp16 operands, fp32 accumulate); bias, logsumexp and the cross-entropy stay fp32.
    decode_logits_amp:                bool            = False

    # "fp64" | "fp32"
    # Prior-bank Gaussian decoders only: precision of the a_v reduction (sum of 1/sigma_v terms). fp32 loses digits once
    # sigma_v falls far below 1; watch decode_sigma_v_min.
    decode_av_precision:              str             = "fp32"

    # "fp64" | "fp32_escalate" | "fp32_escalate_cond"
    # family="gaussian_full" with family_chunked only: precision of the full-covariance pair KL.
    #   fp64                unconditional float64
    #   fp32_escalate       float32, recomputed in float64 when the Cholesky certificate fails
    #   fp32_escalate_cond  fp32_escalate plus a pivot-ratio conditioning trigger
    full_cov_kl_precision:            str             = "fp32_escalate_cond"
    
    # Prior-bank Gaussian decoders only: recompute an uncertain float32 decode argmax in float64 for the
    # candidates whose score interval reaches the winner's (the fused training CE is unaffected).
    decode_ranking_fp64_escalation:   bool            = False
    
    ##################################################
    #   6. Gauge group
    ##################################################

    # "block_glk" | "so_n" | "block_so" | "block_co" | "block_glmc"
    #   block_glk   GL(d_head) per head
    #   so_n        SO(group_n) acting through the irrep blocks of irrep_spec (one block per head)
    #   block_so    SO(d_head) per head (rotations)
    #   block_co    CO(d_head) = R+ x SO(d_head) per head (rotation and a positive gain)
    #   block_glmc  GL(m, C) per head realified on d_head = 2m (even); complex coordinate k = pair (2k, 2k+1)
    gauge_group:                      str             = "block_glk"

    # None | "per_head" | "shared"
    #   None      the group's native tying (block_glk "per_head", so_n "shared")
    #   per_head  an independent group element per head
    #   shared    one element acting on every head. only: block_glk, so_n. block_glk "shared" refuses
    #             pullback_group, fisher_group and group_log; so_n "per_head" refuses
    #             use_cg_coupling
    gauge_tying:                      Optional[str]   = None

    # "on" | "off" | "frozen"
    #   on      the frame is learned as phi_scale / m_phi_lr / pos_phi specify (pure path)
    #   off     Omega = I exactly: coerces phi_scale=0, pos_phi="none", m_phi_lr=0; needs pos_rotation="none"
    #           or rope_space="comparison" (which rotates the routing moments, never the frame)
    #   frozen  a random fixed frame: keeps phi_scale, coerces m_phi_lr=0 (same parameter count as "on")
    gauge_transport:                  str             = "on"

    # so_n only (None for every other group). group_n is the N of SO(N), an int >= 2, decoupled from K.
    # irrep_spec: [(label, mult), ...] | "random". Labels: "l<p>" symmetric-traceless rank p (N=3: spin
    # p, dim 2p+1), "w<p>" exterior power, "wp<m>" / "wm<m>" self-dual / anti-self-dual halves at
    # m = N/2 even. Blocks are laid out in order, one per head, and their dims must sum to embed_dim.
    # "random" draws exactly n_heads irreps summing to embed_dim (ranks in [irrep_min_rank,
    # irrep_max_rank], None = all that fit) with irrep_spec_seed, and the saved config keeps the drawn list.
    # JSON list-of-lists pairs are accepted and coerced to tuples.
    group_n:                            Optional[int]   = None
    irrep_spec: 'Optional[List[Tuple[str, int]] | str]' = None 
    irrep_max_rank:                     Optional[int]   = None
    
    irrep_spec_seed:                    int             = 0
    irrep_min_rank:                     int             = 1            # 1 skips the gauge-invariant trivial irrep l0
    
    # "canonical" | "eigh"
    # so_n only: basis of each irrep block.
    #   canonical  deterministic; for group_n=3 any spin in closed form
    #   eigh       the original tensor-power construction (rank <= 6); old so_n configs load as eigh
    irrep_basis:                      str             = "canonical"

    # so_n with gauge_tying="shared" only: Clebsch-Gordan coupling between irrep blocks on the means
    # (exactly equivariant; learned scalar path weights, zero init).
    use_cg_coupling:                  bool            = False
    
    # "passthrough" | "delta_full"
    # use_cg_coupling only: covariance after the CG update.
    #   passthrough  covariance untouched
    #   delta_full   first-order pushforward J Sigma J^T + cg_covariance_floor * Sigma (family="gaussian_full")
    cg_covariance_mode:               str             = "passthrough"
    cg_covariance_floor:              float           = 1e-6
    cg_energy_weight:                 float           = 0.0          # > 0 adds the post-CG moment divergence to the loss (needs use_cg_coupling)

    ##################################################
    #   7. Positional encoding
    ##################################################

    # "none" | "learned" | "frozen"
    # Positional Lie-algebra element composed into the token frame (exact group product).
    #   none     no positional frame (pure path)
    #   learned  learned (max_seq_len, n_gen) table
    #   frozen   parameter-free i * pos_phi_scale on one generator axis; on the projected routes it is folded into
    #            the frame before the learned maps, so routing is not relative (relative: pos_rotation='rope' with
    #            rope_space='comparison')
    pos_phi:                          str             = "learned"
    pos_rotation:                     str             = "none"       # "none" | "rope" block-diagonal rotation R(theta) folded 
                                                                     # into each frame on the right, V_i = U_i R_i^T.
    rope_on_value:                    bool            = False        # rope only: rotate the content frame too (False: routing only)
    pos_phi_project_slk:              bool            = False        # pos_phi only: per-block trace projection (det = 1)
    bch_residual_max:                 Optional[float] = None         # manifold phi M-step only: group-product residual gate, min(1e-6, value); None = 1e-6 (tightens only)
    
    rope_base:                        float           = 100.0        # rope only: frequency base
    # rope only: p-RoPE fraction p in (0, 1] (Barbero et al., ICLR 2025): only the floor(p d / 2) fastest
    # pairs of each block rotate, at their full-RoPE frequencies; the slower pairs stay position-free. 1.0 = RoPE.
    rope_fraction:                    float           = 1.0
    # "frame" | "comparison"
    # rope only: where the rotation acts.
    #   frame       folded into each frame, V_i = U_i R_i^T, before the learned maps (absolute-position routing)
    #   comparison  applied to each head's projected query and key moments after the learned maps, so the
    #               routing cost depends on positions only through i - j (kernel readout: routing only;
    #               minimal_agent: the record channel R_{j-i}, routing and content by exact conditioning)
    rope_space:                       str             = "frame"
    
    ##################################################
    #   8. Attention routing
    ##################################################

    # "projected_predictive" | "projected_gradient"
    #   projected_predictive  the predictive readout replaces the belief (section 9)
    #   projected_gradient    a frozen-context receiver update (section 11)
    attention_mode:                   str             = "projected_predictive"

    # Temperature scale: tau_h = kappa_beta * sqrt(d_h); a list (one per head) gives per-head temperatures.
    # Not read by predictive_readout="minimal_agent".
    kappa_beta:                 'float | List[float]' = 1.0

    # "causal" | "causal_noself" | "causal_alibi" | "causal_alibi_noself" | "causal_windowed"
    # | "t5_relative_bias" | "uniform" | "alibi" | "windowed"
    # Log-prior over sources. The last three are not causal (they see future tokens).
    #   *_noself           masks the self key
    #   *alibi             adds -slope_h |i - j| (Press et al.)
    #   *windowed          -inf outside a band of attention_window
    #   t5_relative_bias   T5 bucketed relative-position bias
    #   uniform            zero log-bias
    beta_attention_prior:             str             = "causal"
    
    # alibi priors only: BASE multiplier; the realized head slopes are alibi_slope * 2^(-8h/H), so an
    # n_heads sweep at fixed alibi_slope rescales the prior.
    alibi_slope:                      float           = 1.0
    attention_window:                 int             = 128          # windowed priors only: band half-width
    t5_num_buckets:                   int             = 32           # t5_relative_bias only: bucket count
    t5_max_distance:                  int             = 128          # t5_relative_bias only: log-bucketing horizon
    t5_learnable_bias:                bool            = False        # t5_relative_bias only: learn the bucket table

    # "gaussian_kl" | "noisy_query" | "mean_distance"
    # projected_predictive with predictive_readout="kernel" only: routing cost.
    #   gaussian_kl    KL between the projected query and key Gaussians
    #   noisy_query    the query mean treated as a noisy measurement (GL(K) v2, Remark 9)
    #   mean_distance  ||a_i - b_j||^2 / 2 of the projected means: the covariance-blind control
    relational_routing:               str             = "gaussian_kl"

    ##################################################
    #   9. Predictive readout (projected_predictive)
    ##################################################

    # "kernel" | "minimal_agent"
    #   kernel         content U_i sum_h O'_h sum_j beta_ij^h T_h xi_j
    #   minimal_agent  each receiver conditions exactly on a noisy comparison record with the key tied to
    #                  the content (S_K = T); no O, routing is the exact source posterior (temperature 1;
    #                  relational_routing, kappa_beta and kl_max are not read). Needs equal heads,
    #                  content_link_edges="off" and not (frame-space pos_rotation="rope" with rope_on_value=False);
    #                  comparison-space RoPE is the known record channel R_{j-i} (exact conditioning).
    predictive_readout:               str             = "kernel"

    # "exact" | "mean_field"
    # minimal_agent only.
    #   exact       each head's exact posterior; heads combined by assumed-density filtering
    #   mean_field  KL labels at the input moments and an information-form fusion (no between-source spread)
    minimal_agent_inference:          str             = "mean_field"
    
    # "passthrough" | "moment_matched"
    #   passthrough     keep the incoming covariance
    #   moment_matched  the exact moment-matched covariance of the readout
    predictive_covariance:            str             = "moment_matched"
    
    predictive_covariance_floor:      float           = 1e-4         # moment_matched with kernel only: floor * (tr C_i / K) I
    minimal_agent_channel_noise:      float           = 1.0          # minimal_agent only: initial channel noise n_h > 0 (learned)
   
    # moment_matched only: recompute the covariance update in backward instead of storing its
    # intermediates (identical values and gradients; saves memory at large K).
    predictive_covariance_checkpoint: bool            = False
    attention_residual:               bool            = True      # kernel only: new mean = mu_i + content (False: content only)

    ##################################################
    #   10. Content links (O, T)
    ##################################################

    # "identity" | "depth_scaled"
    # predictive_readout="kernel" only (as are content_link_chart and attention_residual).
    #   identity      initial content scale 1
    #   depth_scaled  initial O times 1/sqrt(2 n_layers)
    relational_output_init:           str             = "identity"

    # "affine" | "first_order" | "exponential"
    # Receiving-map chart on the stored (O, T); every chart starts at the same content.
    #   affine       O' = O (O = 0 is zero content)
    #   first_order  O' = T^-1 + O (O = 0 is flat transport)
    #   exponential  O' = T^-1 + O phi_1(T O) = e^{O T} T^-1 (O = 0 is flat transport)
    content_link_chart:               str             = "exponential"

    # "off" | "lowrank" | "pure_gauge"
    # Edge-dependent links G_ij (only: content_link_chart="exponential", projected_predictive, equal heads).
    #   off         one shared exponential link
    #   lowrank     curved: G_ij = g_i e^L exp(A_i B_j^T), rank content_link_edge_rank - EXPENSIVE
    #   pure_gauge  flat control: G_ij = g_i e^L g_j^-1, g_i = exp(A B_i^T)
    content_link_edges:               str             = "off"
    content_link_edge_rank:           int             = 4            # edges on only: factor rank r, 1 <= r <= embed_dim

    # content_link_edges="lowrank" only: the receiver factor A_i becomes a per-context MAP estimate with
    # prior N(a(xi_i), sigma_L^2 I), found by damped fixed-point steps. Stable while sigma_L^2 times the
    # curvature stays below about 1; watch val_relational_latent_energy_final. EXPENSIVE
    content_link_latent:              bool            = False
    content_link_latent_steps:        int             = 1            # fixed-point steps >= 1
    content_link_latent_sigma:        float           = 0.1          # prior scale sigma_L > 0
    content_link_latent_damping:      float           = 1.0          # damping rho in (0, 1]
    content_link_loop_weight:         float           = 0.0          # lambda_W >= 0 loop prior (> 0 needs content_link_latent)
    content_link_loop_window:         Optional[int]   = 1            # loop window >= 1; None = every loop (O(N^3) per layer)

    ##################################################
    #   11. Receiver (projected_gradient)
    ##################################################

    # "channel_likelihood" | "ambient_alignment"
    #   channel_likelihood  expected likelihood of O_h^T xi_i against the d-dim message T_h xi_j
    #   ambient_alignment   sum of KLs to K-dim noisy laws
    relational_receiver_law:          str             = "channel_likelihood"

    # "closed_form" | "euler"
    #   closed_form  exact frozen-context natural-parameter update, damped by relational_receiver_damping
    #   euler        explicit Fisher step with e_q_mu_lr / e_q_sigma_lr and the trust regions
    relational_receiver_update:       str             = "closed_form"
    relational_receiver_damping:      float           = 1.0          # closed_form only: rho in (0, 1]; 1 = the minimizer
    relational_receiver_noise:        float           = 1.0          # channel floor (channel_likelihood) | K-dim noise (ambient_alignment)

    ##################################################
    #   12. Free-energy weights
    ##################################################

    # "constant" | "state_dependent" | "state_dependent_per_coord"
    # Self-coupling alpha_i on KL(q_i || p_i).
    #   constant                   alpha = lambda_alpha
    #   state_dependent            alpha_i = c0 / (b0 + D(q_i || p_i))
    #   state_dependent_per_coord  the same per coordinate (b0, c0 may be K-lists)
    lambda_alpha_mode:                str                   = "state_dependent_per_coord"
    lambda_alpha:                     float                 = 1.0          # constant only
    b0:                               'float | List[float]' = 1.0
    c0:                               'float | List[float]' = 1.0

    lambda_beta:                      float                 = 1.0          # weight of the whole belief-coupling block (1 = pure F)
    
    mass_phi:                         float                 = 0.0          # (mass_phi / 2) ||phi||^2 penalty
    mstep_self_coupling_weight:       float                 = 0.0          # M-step sum_i alpha_i D(q_i* || p_i); 0 = off

    ##################################################
    #   13. Normalization and block handoff
    ##################################################

    # "none" | "layernorm" | "mahalanobis"
    #   none         identity (gauge-pure)
    #   layernorm    standard LayerNorm over the belief dim (not gauge-equivariant)
    #   mahalanobis  gauge-invariant rescale
    # norm_type_block runs inside every block; norm_type_final once before decode.
    norm_type_block:                  str             = "layernorm"
    norm_type_final:                  str             = "none"
    layernorm_affine:                 bool            = False        # layernorm only: learned gamma/beta (init 1/0)

    # "local" | "reference"
    #   local      the block norm, block MLP and final norm act on mu_i
    #   reference  they act on xi_i = U_i^-1 mu_i and map back, mu -> U_i Phi(U_i^-1 mu). only:
    #              projected_predictive, pos_rotation="none", a "none"/"layernorm" norm, and a passthrough
    #              coordinate / gaussian_template MLP or a prototype_agents MLP
    block_coordinates:                str             = "local"

    # Prior handoff to the next block, convex blends: p_next = (1 - rho) p + rho q.
    prior_handoff_rho:                float           = 1.0          # mean: 1 = full flow, 0 = frozen priors
    prior_handoff_sigma:              float           = 0.0          # variance: 0 = frozen at the embedding

    ##################################################
    #   14. Block MLP
    ##################################################

    use_block_mlp:                    bool            = True         # residual feed-forward unit after each block

    # "coordinate" | "gauge_gate" | "canonical_frame" | "gaussian_template" | "prototype_agents"
    #   coordinate         standard MLP on the mean coordinates
    #   gauge_gate         gauge-equivariant gate built from Mahalanobis invariants
    #   canonical_frame    ordinary MLP in the realized frame-fixed coordinates
    #   gaussian_template  softmax over template KLs (covariance kept)
    #   prototype_agents   minimal agent model with M = block_mlp_expansion * K template agents
    block_mlp_mode:                   str             = "coordinate"

    # "passthrough" | "delta_full" | "posterior"
    #   passthrough  covariance untouched (every mode)
    #   delta_full   J Sigma J^T + block_mlp_covariance_floor * Sigma (coordinate, gauge_gate, canonical_frame)
    #   posterior    the template posterior covariance (prototype_agents)
    block_mlp_covariance:             str             = "passthrough"
   
    # "gelu" | "silu" | "relu"
    # Hidden activation; not read by gaussian_template or prototype_agents.
    block_mlp_activation:             str             = "gelu"

    # Block norm reapplied to the post-MLP mean, so the next router reads a normalized mean (audit
    # 2026-10-08 P1-5). False keeps norm -> MLP. only: block_coordinates="local", a block norm, a block MLP
    block_mlp_post_norm:              bool            = False
       
    # "mean_field" | "exact"
    # prototype_agents only: template posterior.
    #   mean_field  KL-label responsibilities and the power posterior
    #   exact       exact mixture posterior (dense or block covariance needs a Cholesky per token/template)
    block_mlp_inference:              str             = "mean_field"
    
    # Per-type (V, n_gen) table added to the first hidden pre-activations of each layer's MLP (zero init).
    # only: projected attention, use_block_mlp with a coordinate or canonical_frame MLP.
    per_type_hidden_bias:             bool            = False        # TODO ablate then remove
    
    block_mlp_expansion:              int             = 8            # hidden width (or template count) = expansion * K
    block_mlp_dropout:                float           = 0.0
    block_mlp_covariance_floor:       float           = 1e-4         # delta_full only
    block_mlp_template_init_variance: float           = 10.0         # prototype_agents only: initial template variance > 0
    # "absolute" | "working_variance"
    # prototype_agents only: template initialization.
    #   absolute          nu ~ U(+-K^-1/2), lambda = block_mlp_template_init_variance (the absolute variance)
    #   working_variance  at a fresh start, per layer from one training batch: nu = sampled block-input means,
    #                     lambda_0 = c (1 - p)/p with c the mean working variance there and p the target pull
    block_mlp_template_init:          str             = "absolute"
    block_mlp_template_init_pull:     float           = 0.09         # working_variance only: initial pull p in (0, 1)

    ##################################################
    #   15. E-step (belief updates)
    ##################################################

    # projected_gradient with relational_receiver_update="euler" only (this group's first four fields).
    # Fisher step sizes:
    e_q_mu_lr:                        float           = 0.5
    e_q_sigma_lr:                     float           = 0.015
    
    # Trust regions: the mean step is projected onto the Mahalanobis ball of this many standard
    # deviations (None = no projection); the sigma step onto one shared tangent ball.
    e_mu_q_trust:                     Optional[float] = 2
    e_sigma_q_trust:                  float           = 5.0
    mu_trust_cholesky_rounds:         int             = 3            # jitter rounds before the mean trust whitening falls back to diagonal
    
    skip_belief_sigma_update:         bool            = False        # projected_gradient only: skip the sigma E-step

    ##################################################
    #   16. Frame (phi) M-step
    ##################################################

    # "adamw" | "pullback_group" | "fisher_group"
    #   adamw           AdamW on the stored phi tables
    #   pullback_group  stateless Frobenius-pullback direction plus a right-group retraction (every registered group;
    #                   block-separable generators per block, shared-tied or so_n frames as one full block)
    #   fisher_group    belief-Fisher natural gradient on the frame (untied block_glk, per_token)
    m_phi_update_mode:                str             = "adamw"

    # "chart" | "group_log"
    # pullback_group / fisher_group only: retraction of the phi step.
    #   chart      additive chart step with product-residual backtracking
    #   group_log  exact right product Log(exp(phi) exp(-eta s xi)) (untied block_glk)
    m_phi_group_retraction:           str             = "chart"
        
    # Interior radius for phi after each M-step, < transport_chart_max_norm (AdamW projects rows back;
    # pullback_group rejects the step). None: nothing bounds ||phi|| under adamw.
    phi_mstep_max_matrix_norm:        Optional[float] = None
    
    m_phi_group_trust_radius:         float           = 0.1          # pullback_group / fisher_group only: step bound
    m_phi_fisher_rotation_weight:     float           = 2.0          # fisher_group only: completion weight lambda on every rotation; raises Fisher-visible rotation metrics a -> a + 2 lambda
    
    ##################################################
    #   17. Learning rates (M-step)
    ##################################################

    m_p_mu_lr:                        float           = 0.025        # mean tables
    m_p_sigma_lr:                     float           = 0.0025       # log-variance tables
    m_phi_lr:                         float           = 0.015        # frame table phi_embed
    
    m_pos_phi_lr:                     Optional[float] = 0.0035       # learned positional frame; None = m_phi_lr
    
    m_decode_mu_lr:                   Optional[float] = 0.025        # untied decode means / linear readout; None = m_p_mu_lr
    m_decode_sigma_lr:                Optional[float] = 0.001        # untied decode log variances (not linear); None = m_p_sigma_lr
    
    m_relational_score_lr:            Optional[float] = 0.005        # query/key maps; None = m_p_mu_lr
    m_relational_content_lr:          Optional[float] = 0.005        # content maps (O, T); None = m_p_mu_lr
    
    m_block_mlp_lr:                   Optional[float] = None         # block MLP; None = m_p_mu_lr
    m_layernorm_lr:                   Optional[float] = None         # layernorm affine; None = m_p_mu_lr
    m_per_type_hidden_bias_lr:        Optional[float] = None         # per_type_hidden_bias; None = the declared m_phi_lr

    ##################################################
    #   18. Weight decay
    ##################################################

    weight_decay:                     float           = 0.05         # AdamW default for every group without its own value
    
    # Mean tables (mu_embed, decode means, linear readout). Decay reaches every row every step, rare
    # tokens included. None = weight_decay; 0 = exempt.
    mu_weight_decay:                  Optional[float] = 0
    
    # Log-variance tables. Decay pulls sigma toward 1 against sigma_init. None = weight_decay; 0 = exempt.
    sigma_weight_decay:               Optional[float] = 0
    
    # Frame tables (phi_embed, pos_phi_free) under adamw; sets a frame-norm ceiling (0 = none).
    phi_weight_decay:                 float           = 0.065

    ##################################################
    #   19. Numerics and precision
    ##################################################

    eps:                              float           = 1e-6
    kl_max:                           float           = 100.0        # safety clamp on every divergence (launchers scale it with K)
    # "absolute" | "row_min": what kl_max clamps in the projected kernel router (audit 2026-10-08 P1-4)
    #   absolute  D_ij; a row whose every admissible cost exceeds kl_max routes uniformly with zero gradient
    #   row_min   D_ij - min_{j admissible} D_ij; beta is the exact softmax wherever no gap exceeds kl_max
    routing_kl_max_reference:         str             = "absolute"
    sigma_max:                        Optional[float] = 100.0        # variance ceiling; None = none
    transport_chart_max_norm:         float           = 50.0         # fail-closed ||M||_F radius of every transport exponential

    # None | "bf16" | "fp16"
    # Mixed precision on CUDA. None = full fp32 (no autocast). Decode, CE and matrix functions stay fp32.
    # bf16 needs no GradScaler; fp16 training does.
    amp_dtype:                        Optional[str]   = "bf16"

    # None | "highest" | "high" | "medium"
    #   None     leave the card's TF32 policy as is
    #   highest  no TF32 (full fp32, reproducible)
    #   high     TF32 (or bf16x3) allowed
    #   medium   bf16 allowed
    fp32_matmul_precision:            Optional[str]   = "highest"
    # Under a reduced fp32_matmul_precision ("medium"/"high"): when a training step raises before its
    # optimizer update, replay it one rung stricter, "medium" -> "high" -> "highest" (IEEE fp32),
    # instead of stopping the run. A failure that persists at "highest" is raised as before.
    # Not used with fp16 (GradScaler state).
    matmul_precision_retry:           bool            = False

    # Redo a frame producer call in float64 when its float32 frame pair is ill-conditioned
    # (eps_32 kappa_F(U) > frame_pair_condition_tol). Costly (about 7x per escalated step).
    frame_pair_fp64_escalation:       bool            = False
    
    # Also rebuild projected covariances that factor in float32 but fail eps_32 ||G||_F ||G^-1||_F <= tol
    # (failed factorizations are always rebuilt). Costly on CUDA; with minimal_agent it also screens the
    # ADF query covariances.
    projected_cov_condition_recovery: bool            = False
       
    # "fp64" | "fp32"
    # Precision of the projected source recovery.
    #   fp64  float64 QR, and float64 for every pair touching a recovered entry
    #   fp32  float32 shifted CholeskyQR3 (falls back to fp64 when that fails)
    projected_cov_recovery_precision: str             = "fp32"

    # "gram" | "cholesky_qr2"
    # Factor route of the projected Gaussian-KL scorer.
    #   gram          factor the float32 Gram X X^T (error O(eps kappa^2))
    #   cholesky_qr2  factor X by CholeskyQR2 (error O(eps kappa); slower unless recovery trips)
    projected_factor_route:           str             = "gram"
    
    projected_gram_condition_tol:     float           = 1e-4         # recovery only, gram route
    projected_factor_condition_tol:   float           = 3e-4         # recovery only, cholesky_qr2 route
    frame_pair_condition_tol:         float           = 1e-4         # escalation only: kappa_F > 838.9 at 1e-4
    
    def __post_init__(self) -> None:
        # Every plain bool/int field is gated here on its exact type (bool is not an int, and
        # "False" is not a bool), so the later validators need not repeat the type test.
        for field in fields(self):
            if field.type in (bool, int):
                value = getattr(self, field.name)
                if type(value) is not field.type:
                    raise _ConfigTypeError(
                        f"{field.name} must be a plain {field.type.__name__}, got "
                        f"{type(value).__name__}: {value!r}"
                    )

        # The model architecture uses full-state projected interfaces in every layer.
        if self.attention_mode in ("transported", "projected_conditional"):
            raise ValueError(
                f"retired attention mode {self.attention_mode!r}; full-state architecture "
                "requires explicit migration to projected_predictive or projected_gradient")
        _require(self.attention_mode, ("projected_predictive", "projected_gradient"), "attention_mode")
        # Validated against the group registry, so a newly registered group needs no edit here
        # (audit 2026-10-07 F39).
        from vfe3.geometry.groups import _GROUPS as _registered_groups
        _supported_groups = tuple(sorted(set(_registered_groups) - _RETIRED_GAUGE_GROUPS))
        if self.gauge_group in _RETIRED_GAUGE_GROUPS:
            raise ValueError(f"gauge_group={self.gauge_group!r} is retired; supported groups are "
                             f"{_supported_groups}")
        _require(self.gauge_group, _supported_groups, "gauge_group")
        full_link = self.attention_mode == "projected_gradient"
        # attention_residual and predictive_covariance are predictive-readout knobs; under
        # projected_gradient they are inactive (inactive_fields), not errors.
        _require(self.relational_routing, ("gaussian_kl", "noisy_query", "mean_distance"), "relational_routing")
        if self.relational_routing != "gaussian_kl" and full_link:
            raise ValueError(f"relational_routing={self.relational_routing!r} requires attention_mode='projected_predictive'")
        from vfe3.model.minimal_agent import MINIMAL_AGENT_INFERENCE, PREDICTIVE_READOUTS
        _require(self.predictive_readout, PREDICTIVE_READOUTS, "predictive_readout")
        _require(self.minimal_agent_inference, MINIMAL_AGENT_INFERENCE, "minimal_agent_inference")
        if (isinstance(self.minimal_agent_channel_noise, bool)
                or not isinstance(self.minimal_agent_channel_noise, Real)
                or not math.isfinite(float(self.minimal_agent_channel_noise))
                or float(self.minimal_agent_channel_noise) <= 0.0):
            raise ValueError("minimal_agent_channel_noise must be a finite real value > 0, "
                             f"got {self.minimal_agent_channel_noise!r}")
        _require(self.predictive_covariance, ("passthrough", "moment_matched"), "predictive_covariance")
        if (isinstance(self.predictive_covariance_floor, bool)
                or not isinstance(self.predictive_covariance_floor, Real)
                or not math.isfinite(float(self.predictive_covariance_floor))
                or float(self.predictive_covariance_floor) <= 0.0):
            raise ValueError(
                f"predictive_covariance_floor must be a finite real value > 0, "
                f"got {self.predictive_covariance_floor!r}")
        if (isinstance(self.relational_receiver_noise, bool)
                or not isinstance(self.relational_receiver_noise, Real)
                or not math.isfinite(float(self.relational_receiver_noise))
                or self.relational_receiver_noise <= 0.0):
            raise ValueError("relational_receiver_noise must be a finite positive reference variance")
        _require(self.relational_receiver_law, ("channel_likelihood", "ambient_alignment"),
                 "relational_receiver_law")
        _require(self.relational_receiver_update, ("closed_form", "euler"), "relational_receiver_update")
        if (isinstance(self.relational_receiver_damping, bool)
                or not isinstance(self.relational_receiver_damping, Real)
                or not math.isfinite(float(self.relational_receiver_damping))
                or not 0.0 < self.relational_receiver_damping <= 1.0):
            raise ValueError("relational_receiver_damping must lie in (0, 1]")
        if full_link and self.n_e_steps < 1:
            raise ValueError("attention_mode='projected_gradient' requires n_e_steps >= 1; "
                             "n_e_steps=0 would remove attention from every layer")
        # A state-dependent alpha* = c0/(b0 + D) is positive (c0 > 0), so only constant alpha can
        # leave the receiver without mass or the channel law without its entropy term.
        constant_alpha = self.lambda_alpha_mode == "constant"
        if (full_link and constant_alpha and self.relational_receiver_law == "ambient_alignment"
                and isinstance(self.lambda_alpha, Real) and self.lambda_alpha <= 0
                and isinstance(self.lambda_beta, Real) and self.lambda_beta <= 0):
            raise ValueError("projected_gradient needs lambda_alpha > 0 or lambda_beta > 0: "
                             "otherwise the receiver energy has no mass")
        if (full_link and constant_alpha and self.relational_receiver_law == "channel_likelihood"
                and not (isinstance(self.lambda_alpha, Real) and self.lambda_alpha > 0)):
            raise ValueError("relational_receiver_law='channel_likelihood' requires lambda_alpha > 0: "
                             "the prior divergence carries the receiver's only entropy term")
        # The projected support envelope. The divergence (KL), connection (flat), gradient roles
        # (filtering), estimator (unroll) and mean preconditioner (Fisher) are fixed, not fields.
        incompatible = []
        if not full_link and self.n_e_steps != 1:
            incompatible.append(f"n_e_steps={self.n_e_steps!r} (requires 1)")
        # The receiver profiles the closed-form alpha forms (projected_gradient.tex, profiled
        # self-coupling); per coordinate, KL(q||p) separates only on the diagonal family.
        if full_link and self.lambda_alpha_mode not in (
                "constant", "state_dependent", "state_dependent_per_coord"):
            incompatible.append(f"lambda_alpha_mode={self.lambda_alpha_mode!r} (requires 'constant', "
                                "'state_dependent' or 'state_dependent_per_coord')")
        if (full_link and self.lambda_alpha_mode == "state_dependent_per_coord"
                and self.family != "gaussian_diagonal"):
            incompatible.append(f"lambda_alpha_mode='state_dependent_per_coord' with family={self.family!r} "
                                "(requires gaussian_diagonal)")
        if self.encode_mode not in ("per_token", "per_token_additive"):
            incompatible.append(
                f"encode_mode={self.encode_mode!r} (requires 'per_token' or 'per_token_additive')")
        if self.family not in ("gaussian_diagonal", "gaussian_full"):
            incompatible.append(f"family={self.family!r} (requires gaussian_diagonal or gaussian_full)")
        if self.n_heads < 2:
            incompatible.append(f"n_heads={self.n_heads!r} (requires at least 2)")
        if incompatible:
            raise ValueError(
                f"attention_mode={self.attention_mode!r} has unsupported projected_filtering configuration: "
                + "; ".join(incompatible))

        # numerics
        if not (math.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and positive, got {self.eps}")
        if not self.kl_max > 0.0:
            raise ValueError(f"kl_max must be positive (infinity disables the ceiling), got {self.kl_max}")
        if not (math.isfinite(self.rope_base) and self.rope_base > 0.0):
            raise ValueError(f"rope_base must be finite and positive, got {self.rope_base}")
        if (isinstance(self.rope_fraction, bool) or not isinstance(self.rope_fraction, Real)
                or not 0.0 < float(self.rope_fraction) <= 1.0):
            raise ValueError(f"rope_fraction must be a real value in (0, 1], got {self.rope_fraction!r}")
        if not (math.isfinite(self.alibi_slope) and self.alibi_slope >= 0.0):
            raise ValueError(f"alibi_slope must be finite and nonnegative, got {self.alibi_slope}")
        for name in ("mu_init_std", "phi_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be >= 0 and finite, got {value}")
        if self.pos_phi != "none" and not math.isfinite(self.pos_phi_scale):
            raise ValueError(
                f"pos_phi_scale must be finite when pos_phi is active, got {self.pos_phi_scale}")
        if not (
            math.isfinite(self.m_phi_group_trust_radius)
            and self.m_phi_group_trust_radius > 0.0
        ):
            raise ValueError(
                "m_phi_group_trust_radius must be finite and positive, got "
                f"{self.m_phi_group_trust_radius}"
            )
        _require(self.m_phi_group_retraction, ("chart", "group_log"), "m_phi_group_retraction")
        if not (
            math.isfinite(self.m_phi_fisher_rotation_weight)
            and self.m_phi_fisher_rotation_weight > 0.0
        ):
            raise ValueError(
                "m_phi_fisher_rotation_weight must be finite and positive, got "
                f"{self.m_phi_fisher_rotation_weight}"
            )
        # sigma_max caps the covariance in the SPD retractions; a nonfinite or sub-eps cap would push
        # the clamped covariance below eps / negative / NaN. None stays permissive (no cap).
        if self.sigma_max is not None and not (
                math.isfinite(self.sigma_max) and self.sigma_max > self.eps):
            raise ValueError(
                f"sigma_max must be None or finite and > eps ({self.eps}), got {self.sigma_max}"
            )
        if not (math.isfinite(self.e_sigma_q_trust) and self.e_sigma_q_trust > 0.0):
            raise ValueError(
                f"e_sigma_q_trust must be finite and positive, got {self.e_sigma_q_trust}"
            )
        if self.phi_mstep_max_matrix_norm is not None and not (
            math.isfinite(self.phi_mstep_max_matrix_norm)
            and self.phi_mstep_max_matrix_norm > 0.0
        ):
            raise ValueError(
                "phi_mstep_max_matrix_norm must be None or finite and positive, got "
                f"{self.phi_mstep_max_matrix_norm}"
            )
        if (isinstance(self.transport_chart_max_norm, bool)
                or not isinstance(self.transport_chart_max_norm, Real)
                or not math.isfinite(self.transport_chart_max_norm)
                or self.transport_chart_max_norm <= 0.0):
            raise ValueError(
                "transport_chart_max_norm must be finite and positive, got "
                f"{self.transport_chart_max_norm!r}"
            )
        if (self.phi_mstep_max_matrix_norm is not None
                and not self.phi_mstep_max_matrix_norm < self.transport_chart_max_norm):
            raise ValueError(
                "phi_mstep_max_matrix_norm must be strictly below transport_chart_max_norm (the "
                "M-step interior radius lies inside the fail-closed transport chart), got "
                f"{self.phi_mstep_max_matrix_norm} >= {self.transport_chart_max_norm}"
            )
        if self.bch_residual_max is not None and not (
            math.isfinite(self.bch_residual_max) and self.bch_residual_max > 0.0
        ):
            raise ValueError(
                "bch_residual_max must be None or finite and positive, got "
                f"{self.bch_residual_max!r}"
            )

        # The declared phi step size, read BEFORE the coercion below can zero it: the per-type hidden-bias
        # table (per_type_hidden_bias) inherits it when m_per_type_hidden_bias_lr is None.
        declared_m_phi_lr = self.m_phi_lr
        # gauge_transport meta-toggle: coerce the gauge-frame fields UP FRONT so every downstream
        # validation and the optimizer/table wiring see the resolved state. 'on' is a no-op.
        _require(self.gauge_transport, _VALID_GAUGE_TRANSPORT, "gauge_transport")
        if self.gauge_transport == "off":
            # Omega = exp(phi_i) exp(-phi_j) = I needs a ZERO frame AND no other transport modifier, so
            # a frame-space positional rotation must also be off (else Omega != I and the cell is mislabeled).
            # Comparison-space RoPE rotates the projected routing moments after the maps and leaves Omega = I.
            if self.pos_rotation != "none" and self.rope_space != "comparison":
                raise ValueError(
                    f"gauge_transport='off' requires Omega=I, but pos_rotation={self.pos_rotation!r} folds "
                    f"a positional rotation into the transport. Set pos_rotation='none'."
                )
            self.phi_scale    = 0.0
            self.pos_phi      = "none"
            self.m_phi_lr     = 0.0
            import warnings
            warnings.warn(
                "gauge_transport='off': forcing the gauge frame to the identity (phi_scale=0.0, "
                "pos_phi='none', m_phi_lr=0.0), so "
                "the transport is identity and frozen.",
                UserWarning, stacklevel=2,
            )
        elif self.gauge_transport == "frozen":
            if self.phi_scale <= 0.0:
                raise ValueError(
                    f"gauge_transport='frozen' needs a nonzero random frame (phi_scale>0), got "
                    f"phi_scale={self.phi_scale}; use gauge_transport='off' for the identity frame."
                )
            self.m_phi_lr = 0.0
            import warnings
            warnings.warn(
                f"gauge_transport='frozen': freezing the random gauge frame at init "
                f"(phi_scale={self.phi_scale}, m_phi_lr=0.0); "
                "the frame is fixed, not learned.",
                UserWarning, stacklevel=2,
            )

        # model structure
        for name in ("vocab_size", "embed_dim", "max_seq_len", "n_heads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        # Domain is >= 0, not >= 1: two probes deliberately drive the field to 0 by direct assignment
        # (the depth-sensitivity sweep and the zero-E-step counterfactual readout), so the declared
        # domain has to describe what the code actually does. Zero is a coherent state, but as a USER
        # setting it silently disables belief inference, so it warns rather than passing quietly.
        if self.n_e_steps < 0:
            raise ValueError(f"n_e_steps must be >= 0, got {self.n_e_steps}")
        if self.n_e_steps == 0:
            # Local import: later blocks in this method do `import warnings`, which makes the name
            # function-local throughout and leaves the module-level binding unreachable here.
            import warnings
            warnings.warn(
                "n_e_steps=0 runs NO belief E-step iteration: the belief stays at its prior and the "
                "decode reads that prior out directly. This is the zero-depth counterfactual the "
                "probes measure, not a trained inference path -- set n_e_steps >= 1 to run inference.",
                UserWarning,
                stacklevel=2,
            )
        # eval_stride above max_seq_len would skip transitions between consecutive held-out windows,
        # which no target mask can recover, so the exactly-once eval contract rejects it. Equal to
        # max_seq_len is the disjoint default spelled out.
        if self.eval_stride is not None and (
                type(self.eval_stride) is not int
                or self.eval_stride < 1
                or self.eval_stride > self.max_seq_len):
            raise ValueError(
                f"eval_stride must be None (disjoint windows) or an int in [1, max_seq_len="
                f"{self.max_seq_len}], got {self.eval_stride!r}")
        # Registration traits of the group (vfe3.geometry.groups.GroupTraits): an irrep-tower group
        # (algebra set, e.g. so_n) takes its heads from irrep_spec, possibly of different sizes;
        # every other group splits embed_dim into n_heads equal blocks.
        from vfe3.geometry.groups import group_traits
        _traits = group_traits(self.gauge_group)
        _tower = _traits.algebra is not None
        if not _tower and self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by n_heads={self.n_heads}"
            )
        if not _tower and self.embed_dim // self.n_heads < _traits.min_head_dim:
            raise ValueError(
                f"gauge_group={self.gauge_group!r} needs head blocks of size >= {_traits.min_head_dim}; "
                f"embed_dim={self.embed_dim}, n_heads={self.n_heads} gives {self.embed_dim // self.n_heads}"
            )
        if not _tower and (self.embed_dim // self.n_heads) % _traits.head_dim_multiple != 0:
            raise ValueError(
                f"gauge_group={self.gauge_group!r} needs head blocks whose size is a multiple of "
                f"{_traits.head_dim_multiple}; embed_dim={self.embed_dim}, n_heads={self.n_heads} gives "
                f"{self.embed_dim // self.n_heads}"
            )

        # Frame tying resolves to the group's native value when unset.
        if self.gauge_tying is None:
            self.gauge_tying = _traits.native_tying
        _require(self.gauge_tying, _traits.tyings, "gauge_tying")

        # gauge seam. Irrep tower: validate group_n + irrep_spec at construction (labels
        # resolve, block dims sum to embed_dim) so a bad spec fails here with the computed dims.
        if _tower:
            if self.group_n is None or self.irrep_spec is None:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} requires both group_n and irrep_spec "
                    f"([(label, mult), ...]); got group_n={self.group_n}, "
                    f"irrep_spec={self.irrep_spec}"
                )
            if type(self.group_n) is not int or self.group_n < 2:
                raise ValueError(f"group_n must be an int >= 2, got {self.group_n!r}")
            from vfe3.geometry.irreps import IRREP_BASES, check_irrep_constructible, irrep_dim
            _require(self.irrep_basis, IRREP_BASES, "irrep_basis")
            if isinstance(self.irrep_spec, str):
                if self.irrep_spec != "random":
                    raise ValueError(f"irrep_spec must be a list of (label, mult) pairs or 'random', "
                                     f"got {self.irrep_spec!r}")
                if type(self.irrep_spec_seed) is not int:
                    raise ValueError(f"irrep_spec_seed must be an int, got {self.irrep_spec_seed!r}")
                from vfe3.geometry.irreps import random_irrep_spec
                self.irrep_spec = random_irrep_spec(
                    self.group_n, self.embed_dim, self.n_heads, algebra=_traits.algebra,
                    basis=self.irrep_basis, seed=self.irrep_spec_seed,
                    min_rank=self.irrep_min_rank, max_rank=self.irrep_max_rank)
            if not isinstance(self.irrep_spec, list) or not self.irrep_spec:
                raise ValueError(
                    f"irrep_spec must be a non-empty list of (label, mult) pairs, got "
                    f"{self.irrep_spec!r}"
                )
            # JSON has no tuple type; coerce reloaded list pairs to tuples.
            self.irrep_spec = [tuple(p) if isinstance(p, list) else p for p in self.irrep_spec]
            _block_dims: List[int] = []
            for _entry in self.irrep_spec:
                if (not isinstance(_entry, tuple) or len(_entry) != 2
                        or not isinstance(_entry[0], str)
                        or type(_entry[1]) is not int or _entry[1] < 1):
                    raise ValueError(
                        "each irrep_spec entry must be a "
                        "(label: str, mult: plain int >= 1) pair, "
                        f"got {_entry!r}"
                    )
                _d = irrep_dim(self.group_n, algebra=_traits.algebra, label=_entry[0])
                check_irrep_constructible(self.group_n, algebra=_traits.algebra, label=_entry[0],
                                          basis=self.irrep_basis)
                _block_dims.extend([_d] * _entry[1])
            if sum(_block_dims) != self.embed_dim:
                raise ValueError(
                    f"irrep_spec blocks {_block_dims} sum to {sum(_block_dims)} != "
                    f"embed_dim={self.embed_dim} (group_n={self.group_n})"
                )
            # The projected routes read the frame through the operators' head partition: each head
            # is one gauge block (relational_score slices the dense inverse frame into those head
            # blocks), so n_heads is the number of irrep blocks. Blocks may differ in size.
            if len(_block_dims) != self.n_heads:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} needs n_heads equal to the number of irrep blocks (one gauge "
                    f"block per head): irrep_spec gives {len(_block_dims)} blocks {_block_dims}, "
                    f"n_heads={self.n_heads}"
                )
        elif self.group_n is not None or self.irrep_spec is not None:
            raise ValueError(
                f"group_n/irrep_spec are consumed only by irrep-tower groups (e.g. 'so_n'); got "
                f"gauge_group={self.gauge_group!r}"
            )
        if self.use_cg_coupling and not _tower:
            raise ValueError(
                f"use_cg_coupling requires an irrep-labeled tower group (e.g. 'so_n'); got "
                f"gauge_group={self.gauge_group!r}"
            )
        if self.use_cg_coupling and self.gauge_tying != "shared":
            raise ValueError(
                "use_cg_coupling requires gauge_tying='shared': a Clebsch-Gordan intertwiner commutes "
                "with the diagonal action of ONE group element on the coupled blocks, and an untied "
                "so_n acts on each block by its own element"
            )
        # CG moment-closure. 'delta_full' needs a full-covariance Gaussian family (the dense
        # J Sigma J^T cannot be stored diagonally). The q-only moment regularizer weight must be a
        # finite non-negative scalar, and any positive weight needs the CG coupling live.
        if self.cg_covariance_mode not in ("passthrough", "delta_full"):
            raise ValueError(
                f"cg_covariance_mode must be 'passthrough' or 'delta_full'; got "
                f"{self.cg_covariance_mode!r}"
            )
        if self.cg_covariance_mode == "delta_full" and self.family != "gaussian_full":
            raise ValueError(
                f"cg_covariance_mode='delta_full' needs a full-covariance Gaussian family "
                f"(family='gaussian_full'); the delta-method J Sigma J^T is dense and a "
                f"diagonal/non-Gaussian family cannot carry it (got family={self.family!r})."
            )
        if not (math.isfinite(self.cg_covariance_floor) and self.cg_covariance_floor > 0.0):
            raise ValueError(
                "cg_covariance_floor must be finite and strictly positive; got "
                f"{self.cg_covariance_floor!r}")

        if not math.isfinite(self.cg_energy_weight) or self.cg_energy_weight < 0.0:
            raise ValueError(
                f"cg_energy_weight must be finite and >= 0; got {self.cg_energy_weight!r}"
            )
        if self.cg_energy_weight > 0.0 and not self.use_cg_coupling:
            raise ValueError(
                f"cg_energy_weight={self.cg_energy_weight} > 0 requires use_cg_coupling=True "
                f"(the moment-energy regularizer reads the CG coupling's per-layer pushforward)."
            )
        # belief family. ``family`` is the SINGLE covariance-structure toggle: a registry key whose
        # declared cov_kind ('diagonal'|'full') is the one source of truth for the diagonal-vs-full
        # path, exposed read-only as the derived ``diagonal_covariance`` property.
        from vfe3.divergence import divergence_families, family_cov_kind
        _require(self.family, divergence_families(), "family")
        family_is_diagonal = family_cov_kind(self.family) == "diagonal"
        from vfe3.families.base import get_family
        _family_cls = get_family(self.family)

        # free-energy coupling
        for _name in ("kappa_beta",):
            _v = getattr(self, _name)
            if isinstance(_v, (list, tuple)):
                if _tower:
                    # heads = irrep blocks (possibly unequal dims): one kappa entry per block
                    # (irrep_spec already validated/coerced in the gauge seam above).
                    _n_blocks = sum(m for _, m in self.irrep_spec)
                    if len(_v) != _n_blocks:
                        raise ValueError(
                            f"{_name} list must have one entry per irrep block; gauge_group="
                            f"{self.gauge_group!r} has {_n_blocks} blocks, got {len(_v)}")
                elif len(_v) != self.n_heads:
                    raise ValueError(
                        f"{_name} list must have length n_heads={self.n_heads}, got {len(_v)}")
                if any(not math.isfinite(x) or x <= 0.0 for x in _v):
                    raise ValueError(f"{_name} entries must be finite and > 0, got {_v}")
            else:
                if not math.isfinite(_v) or _v <= 0.0:
                    raise ValueError(f"{_name} must be finite and positive, got {_v}")
        if not math.isfinite(self.mass_phi) or self.mass_phi < 0.0:
            raise ValueError(f"mass_phi must be >= 0 and finite, got {self.mass_phi}")
        if (not math.isfinite(self.mstep_self_coupling_weight)
                or self.mstep_self_coupling_weight < 0.0):
            raise ValueError(
                "mstep_self_coupling_weight must be >= 0 and finite, got "
                f"{self.mstep_self_coupling_weight}"
            )
        if not math.isfinite(self.lambda_beta) or self.lambda_beta < 0.0:
            raise ValueError(f"lambda_beta must be >= 0 and finite, got {self.lambda_beta}")
        # attention priors validated against the prior REGISTRY (add-by-registering), so a newly
        # registered prior is a valid config value without editing this validator.
        from vfe3.attention_prior import _PRIORS
        # prior-shape knobs (threaded into model._attention_log_prior; audit P9): the windowed
        # band half-width and the T5 bucketing parameters must be positive ints.
        for name in ("attention_window", "t5_max_distance"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an int >= 1, got {getattr(self, name)!r}")
        # t5_num_buckets needs >= 2: max_exact = num_buckets // 2, so num_buckets=1 -> max_exact=0 ->
        # division by zero in the log-bucketing.
        if not isinstance(self.t5_num_buckets, int) or self.t5_num_buckets < 2:
            raise ValueError(f"t5_num_buckets must be an int >= 2, got {self.t5_num_buckets!r}")
        # The log-bucketing divides by log(max_distance / max_exact), max_exact = num_buckets // 2.
        # max_distance == max_exact gives log(1) = 0 -> garbage .long() bucket index; max_distance <
        # max_exact gives a NEGATIVE index that silently reads the wrong end of the bias table. Require
        # strict max_distance > num_buckets // 2 so every bucket index stays in [0, num_buckets).
        if self.t5_max_distance <= self.t5_num_buckets // 2:
            raise ValueError(
                f"t5_max_distance must be > t5_num_buckets // 2 (={self.t5_num_buckets // 2}) so the "
                f"T5 log-bucketing denominator log(t5_max_distance / (t5_num_buckets // 2)) is "
                f"positive; got t5_max_distance={self.t5_max_distance}, t5_num_buckets="
                f"{self.t5_num_buckets}."
            )
        for name in ("b0", "c0"):
            _v = getattr(self, name)
            if isinstance(_v, (list, tuple)):
                # A (K,) per-coordinate b0/c0 shapes the PER-COORDINATE alpha*_k = c0_k/(b0_k + D_k);
                # every other alpha form carries a per-position scalar D, against which a (K,) list
                # either crashes at the first forward or (when K == N) silently mis-broadcasts.
                from vfe3.alpha_i import alpha_is_per_coord
                if not alpha_is_per_coord(self.lambda_alpha_mode):
                    raise ValueError(
                        f"{name} list (per-coordinate) requires a per-coordinate alpha form "
                        f"(lambda_alpha_mode='state_dependent_per_coord'); got "
                        f"lambda_alpha_mode={self.lambda_alpha_mode!r}"
                    )
                if len(_v) != self.embed_dim:
                    raise ValueError(
                        f"{name} list must have length embed_dim={self.embed_dim}, got {len(_v)}")
                if any(not math.isfinite(x) or x <= 0.0 for x in _v):
                    raise ValueError(f"{name} entries must be finite and > 0, got {_v}")
            else:
                if not math.isfinite(_v) or _v <= 0.0:
                    raise ValueError(f"{name} must be finite and positive, got {_v}")
        if not math.isfinite(self.sigma_init) or self.sigma_init <= 0.0:
            raise ValueError(
                f"sigma_init must be positive and finite (log is taken), got {self.sigma_init}")
        # lambda_alpha_mode validated against the alpha-form REGISTRY (add-by-registering).
        from vfe3.alpha_i import _ALPHAS
        _require(self.lambda_alpha_mode, tuple(sorted(_ALPHAS)), "lambda_alpha_mode")
        if not (math.isfinite(self.lambda_alpha) and self.lambda_alpha >= 0.0):
            raise ValueError(
                f"lambda_alpha must be finite and nonnegative under lambda_alpha_mode='constant', "
                f"got {self.lambda_alpha}"
            )
        # Full Gaussians expose an ordered conditional KL decomposition, not a sum of marginal KLs.
        from vfe3.alpha_i import alpha_is_per_coord
        if (alpha_is_per_coord(self.lambda_alpha_mode) and not family_is_diagonal
                and not getattr(_family_cls, "conditional_kl_coordinates", False)):
            raise ValueError(
                "per-coordinate full-covariance self-coupling requires a family with "
                "conditional KL coordinates")

        # attention
        _require(self.beta_attention_prior, tuple(sorted(_PRIORS)), "beta_attention_prior")

        # E-step
        for name in ("e_q_mu_lr", "e_q_sigma_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be >= 0 and finite, got {value}")
        from vfe3.gauge_optim import _PHI_UPDATE_POLICIES, get_phi_update_policy
        _require(
            self.m_phi_update_mode,
            tuple(sorted(_PHI_UPDATE_POLICIES)),
            "m_phi_update_mode",
        )
        phi_update_policy = get_phi_update_policy(self.m_phi_update_mode)
        if (
            self.encode_mode == "per_token_additive"
            and self.m_phi_update_mode == "pullback_group"
        ):
            raise ValueError(
                "encode_mode='per_token_additive' is incompatible with "
                "m_phi_update_mode='pullback_group' because the additive encoder does not "
                "provide the group-chart state required by the pullback update"
            )
        if phi_update_policy.requires_fisher_statistics:
            # Belief-Fisher phi direction (derivation 2026-09-28, Track 2). The metric is read at the
            # layer-0 table row (mu_v, diag s_v) in closed form, per head block, so the tables must
            # hold LOCAL moments whose layer-0 covariance is diag(s) exactly: per_token encoding and
            # a Gaussian family that encodes Sigma = diag(bounded_variance(sigma_log)).
            # The whitened closed forms need untied block_glk (the metric is block-diagonal across
            # heads only for block-diagonal local covariance) and the full registered chart.
            fisher_problems = []
            if self.gauge_group != "block_glk":
                fisher_problems.append(f"gauge_group='block_glk'; got {self.gauge_group!r}")
            elif self.gauge_tying != "per_head":
                fisher_problems.append(f"gauge_tying='per_head'; got {self.gauge_tying!r}")
            if self.encode_mode != "per_token":
                fisher_problems.append(f"encode_mode='per_token'; got {self.encode_mode!r}")
            if self.pos_phi not in ("none", "learned", "frozen"):
                fisher_problems.append(f"pos_phi in ('none', 'learned', 'frozen'); got {self.pos_phi!r}")
            if self.pos_phi_project_slk:
                fisher_problems.append("pos_phi_project_slk=False (the metric uses the full chart)")
            chart_problem = self._chart_ordering_problem()
            if chart_problem is not None:
                fisher_problems.append(chart_problem)
            if fisher_problems:
                raise ValueError(
                    "m_phi_update_mode='fisher_group' requires " + "; ".join(fisher_problems))
        if phi_update_policy.requires_pullback_geometry:
            from vfe3.geometry.groups import get_group
            group_builder = get_group(self.gauge_group)
            capability = getattr(group_builder, "pullback_capability", None)
            if capability is None:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} registers no pullback capability; phase one "
                    "requires a built-in bracket-closed per-block pullback direction"
                )
            if self.gauge_group == "block_glk" and self.d_head > 12:
                raise ValueError(
                    "pullback_group with gauge_group='block_glk' requires d_head <= 12; got "
                    f"d_head={self.d_head}"
                )
            if self.pos_phi_project_slk:
                raise ValueError(
                    "pullback_group does not support pos_phi_project_slk=True because the stored "
                    "factor metric is defined on the full registered chart"
                )
            chart_problem = self._chart_ordering_problem()
            if chart_problem is not None:
                raise ValueError(f"pullback_group requires {chart_problem}")
        if (phi_update_policy.requires_manifold_optimizer and self.m_phi_group_retraction == "group_log"
                and (self.gauge_group != "block_glk" or self.gauge_tying != "per_head")):
            raise ValueError(
                "m_phi_group_retraction='group_log' reads the chart as per-head d x d blocks and "
                "requires untied block_glk; "
                f"got gauge_group={self.gauge_group!r}, gauge_tying={self.gauge_tying!r}"
            )

        from vfe3.model.positional_phi import _POS_PHI
        _require(self.pos_phi, tuple(sorted(_POS_PHI)), "pos_phi")
        from vfe3.geometry.rope import _POS_ROTATIONS
        _require(self.pos_rotation, tuple(sorted(_POS_ROTATIONS)), "pos_rotation")
        # The retained projected routes fold RoPE into the realized frame on the right,
        # V_i = U_i R_i^T (rope_insertion is fixed to "right"). Under U_i -> g_i U_i this gives
        # V_i -> g_i V_i, the same right-factor structure as pos_phi's exp(Y_i), so routing and
        # content stay exactly covariant (audit 2026-10-01 F11: residual ~6e-16). The former
        # gauge-breaking warning quoted a probe of the retired left insertion and was removed.
        # RoPE's adjacent-pair rotation R is orthogonal but generally NOT in the irrep image
        # rho(SO(N)) inside GL(K), so R Omega R^T leaves the SELECTED structure group
        # (GL(K)-congruence invariance survives).
        _require(self.rope_space, ("frame", "comparison"), "rope_space")
        if self.pos_rotation == "rope":
            # p-RoPE rotates floor(p d / 2) pairs of each block (frame space: each irrep block; comparison
            # space: each head's comparison block, also of the head size); none rotated is NoPE, not p-RoPE.
            empty = sorted({d for d in self.head_dims if d >= 2 and int(float(self.rope_fraction) * d // 2) == 0})
            if empty:
                raise ValueError(f"rope_fraction={self.rope_fraction} rotates no pair of blocks of size {empty}; "
                                 "raise it or use pos_rotation='none'")
        if self.pos_rotation == "rope" and self.rope_space == "comparison":
            problems = []
            if self.attention_mode != "projected_predictive":
                problems.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive')")
            # minimal_agent: the rotation is a known record channel R_{j-i}, applied by exact conditioning.
            if self.rope_on_value:
                problems.append("rope_on_value=True (the comparison-space rotation is not a frame rotation)")
            if problems:
                raise ValueError("rope_space='comparison' rotates the projected routing moments after the learned "
                                 "maps; got " + "; ".join(problems))
        if self.pos_rotation == "rope" and self.rope_space == "frame" and not _traits.contains_pair_rotations:
            import warnings
            warnings.warn(
                f"pos_rotation='rope' with gauge_group={self.gauge_group!r} leaves the structure "
                f"group: the coordinate-pair rotation R(theta) is generally not in the group's "
                f"image in GL(K), so R Omega R^T is no longer in the selected group. The "
                f"GL(K)-congruence divergence invariance still holds (R is orthogonal), so the "
                f"model runs; only the structure-group claim is dropped.",
                UserWarning,
                stacklevel=2,
            )
        # The full-covariance KL precision spellings are inlined rather than imported from
        # vfe3.families.gaussian (importing a family module inside __post_init__ perturbs import
        # order). set_full_cov_kl_precision validates the same set.
        if self.full_cov_kl_precision not in ("fp64", "fp32_escalate", "fp32_escalate_cond"):
            raise ValueError(
                "full_cov_kl_precision must be one of ('fp64', 'fp32_escalate', "
                f"'fp32_escalate_cond'), got {self.full_cov_kl_precision!r}")
        if self.projected_factor_route not in ("gram", "cholesky_qr2"):
            raise ValueError(
                "projected_factor_route must be one of ('gram', 'cholesky_qr2'), "
                f"got {self.projected_factor_route!r}")
        _require(self.projected_cov_recovery_precision, ("fp64", "fp32"),
                 "projected_cov_recovery_precision")
        for name in ("frame_pair_condition_tol", "projected_gram_condition_tol",
                     "projected_factor_condition_tol"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0.0 < value < 1.0):
                raise ValueError(f"{name} must be a finite number in (0, 1), got {value!r}")
        if self.mu_trust_cholesky_rounds < 0:
            raise ValueError(
                "mu_trust_cholesky_rounds must be a nonnegative int, got "
                f"{self.mu_trust_cholesky_rounds!r}")
        if self.val_diagnostic_batches < 1:
            raise ValueError(
                "val_diagnostic_batches must be an int >= 1, got "
                f"{self.val_diagnostic_batches!r}")
        if self.decode_av_precision not in ("fp32", "fp64"):
            raise ValueError(
                "decode_av_precision must be one of ('fp32', 'fp64'), "
                f"got {self.decode_av_precision!r}")
        if self.fp32_matmul_precision is not None and self.fp32_matmul_precision not in (
                "highest", "high", "medium"):
            raise ValueError(
                "fp32_matmul_precision must be None (leave the card as-is) or one of "
                f"('highest', 'high', 'medium'), got {self.fp32_matmul_precision!r}")
        # decode / encode
        if not math.isfinite(self.decode_tau) or self.decode_tau <= 0.0:
            raise ValueError(f"decode_tau must be finite and positive, got {self.decode_tau}")
        _require(self.relational_output_init, ("identity", "depth_scaled"), "relational_output_init")
        from vfe3.model.relational_operators import CONTENT_LINK_CHARTS
        _require(self.content_link_chart, CONTENT_LINK_CHARTS, "content_link_chart")
        from vfe3.model.edge_links import EDGE_MODES
        _require(self.content_link_edges, EDGE_MODES, "content_link_edges")
        if self.content_link_edges != "off":
            edge_problems = []
            if not 1 <= self.content_link_edge_rank <= self.embed_dim:
                edge_problems.append(f"content_link_edge_rank={self.content_link_edge_rank!r} "
                                     f"(requires 1 <= rank <= embed_dim={self.embed_dim})")
            if self.content_link_chart != "exponential":
                edge_problems.append(f"content_link_chart={self.content_link_chart!r} (requires 'exponential': "
                                     "the edge links multiply the shared exponential link)")
            if self.attention_mode != "projected_predictive":
                edge_problems.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive': "
                                     "the projected_gradient receiver reads the shared content maps only)")
            if len(set(self.head_dims)) > 1:
                edge_problems.append(f"head sizes {self.head_dims} (requires equal heads; a mixed so_n irrep "
                                     "tower is not supported)")
            if edge_problems:
                raise ValueError(f"content_link_edges={self.content_link_edges!r} is unsupported here: "
                                 + "; ".join(edge_problems))
        # Latent receiver factors and their loop prior (plan Section 3.4); the same checks as RelationalOperators.
        from vfe3.model.relational_operators import _validate_latent_links
        _validate_latent_links(self.content_link_edges, self.content_link_latent, self.content_link_latent_steps,
                               self.content_link_latent_sigma, self.content_link_latent_damping,
                               self.content_link_loop_weight, self.content_link_loop_window)
        # The minimal-agent readout (plan Section 2.1); the same checks as RelationalOperators where it can see them.
        if self.predictive_readout == "minimal_agent":
            agent_problems = []
            if self.attention_mode != "projected_predictive":
                agent_problems.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive': the "
                                      "projected_gradient receiver reads the content maps O)")
            if len(set(self.head_dims)) > 1:
                agent_problems.append(f"head sizes {self.head_dims} (requires equal heads; a mixed so_n irrep tower is "
                                      "not supported)")
            if self.content_link_edges != "off":
                agent_problems.append(f"content_link_edges={self.content_link_edges!r} (requires 'off': the edge links "
                                      "multiply the content factors O, which the minimal agent derives)")
            if self.pos_rotation == "rope" and self.rope_space == "frame" and not self.rope_on_value:
                agent_problems.append("frame-space pos_rotation='rope' with rope_on_value=False (the tied key is "
                                      "the content, so both are read in one frame; or use rope_space='comparison')")
            if agent_problems:
                raise ValueError("predictive_readout='minimal_agent' is unsupported here: " + "; ".join(agent_problems))
        # decode_mode / encode_mode validated against the LIVE decode/encode REGISTRIES (not static
        # literals) so a newly registered kernel is config-selectable without editing this validator.
        from vfe3.model.prior_bank import _DECODERS, _ENCODERS
        _require(self.decode_mode, tuple(sorted(_DECODERS)), "decode_mode")
        # decode_mode sets the covariance RANK(s) the prior-bank kernel accepts, via its resolved
        # DecodeRegistration.covariance_kinds: 'diagonal*' -> {'diagonal'}, 'full*' -> {'full'},
        # 'family*' -> both. The configured family's cov_kind must be a MEMBER of that set, else the
        # mismatch is a shape RuntimeError at the first forward. The linear registration discards
        # sigma, so these rank checks apply only to non-linear registrations.
        decode_registration = _DECODERS[self.decode_mode]
        family_kind = family_cov_kind(self.family)
        if (self.decode_mode != "linear"
                and family_kind not in decode_registration.covariance_kinds):
            raise ValueError(
                f"decode_mode={self.decode_mode!r} is rank-incompatible with family={self.family!r}: "
                f"it scores covariance kinds {sorted(decode_registration.covariance_kinds)} but the "
                f"family is {family_kind!r}. Pair a matching decode_mode (a full/diagonal kernel "
                f"with a full/diagonal family, or 'family_chunked' which scores both), or "
                f"select decode_mode='linear'."
            )
        if self.decode_chunk_size < 1:
            raise ValueError(f"decode_chunk_size must be >= 1, got {self.decode_chunk_size}")
        _require(self.decode_ce_checkpoint, ("auto", "auto_aggregate"), "decode_ce_checkpoint")
        # decode_bias (linear decode only) and decode_tau (prior-bank decode only) are reported by
        # inactive_fields(), so the default-guarded notice below covers them and untouched
        # defaults stay silent.
        # Capability validation: the fast diagonal/full kernels hardcode the Gaussian KL readout, so a
        # family without the Gaussian POINTWISE algebra must select the family-consistent
        # 'family_chunked' decode, which scores logits through the configured family.
        if (self.decode_mode != "linear" and not _family_cls.gaussian_pointwise_algebra
                and not decode_registration.family_consistent):
            raise ValueError(
                f"prior-bank decode with family={self.family!r} requires a family-consistent "
                f"decode_mode ('family_chunked'); decode_mode={self.decode_mode!r} is a fixed "
                f"Gaussian KL kernel. Set decode_mode='family_chunked' or 'linear'.")
        # Full-covariance compute discarded at the decode boundary: linear decode reads out
        # logits = mu_q @ W^T and DISCARDS sigma. The E-step still evolves a (B,N,K,K) covariance (it
        # shapes the mean trajectory, so the result is correct, not wasted) but that covariance never
        # reaches the output. Warn so the combination is a deliberate choice; the covariance reaches
        # the logits only under a full prior-bank decode.
        if not family_is_diagonal and self.decode_mode == "linear":
            import warnings
            warnings.warn(
                f"family={self.family!r} (full covariance) with decode_mode='linear': the linear "
                f"decode logits = mu_q @ W^T discards the converged (B,N,K,K) covariance at the output "
                f"boundary (it still shapes the E-step mean trajectory, so the result is correct). For "
                f"the covariance to reach the logits use decode_mode='full_chunked' "
                f"(the KL-to-prior decode).",
                UserWarning,
                stacklevel=2,
            )
        # encode_mode is validated against the live encoder registry.
        _require(self.encode_mode, tuple(sorted(_ENCODERS)), "encode_mode")
        # 'per_token_additive' is the Omega = I capacity control: the token frame is zero, but a
        # positional frame (pos_phi's right factor, or the gauge-RoPE fold) still composes into
        # U_i, so Omega_ij = U_i U_j^{-1} != I and the control is contaminated. Checked after the
        # gauge_transport coercion, which supplies pos_phi='none' under 'off'. Comparison-space RoPE does
        # not compose into U_i.
        if (self.encode_mode == "per_token_additive"
                and (self.pos_phi != "none"
                     or (self.pos_rotation != "none" and self.rope_space != "comparison"))):
            detail = (
                f"encode_mode='per_token_additive' with pos_phi={self.pos_phi!r}, "
                f"pos_rotation={self.pos_rotation!r}: the positional frame keeps Omega_ij != I, "
                "so the additive encoder is not the Omega = I capacity control; set "
                "pos_phi='none' and pos_rotation='none'"
            )
            raise ValueError(detail + " on the projected routes")
        # decode_mode is the single decode gate. Linear selects the learned (V,K) projection and
        # discards sigma; every other admitted mode reads out against the prior bank. Encode and the
        # free-energy self-coupling stay on PriorBank in both cases.
        # handoff (both blends must be convex so the prior stays on the SPD cone:
        # sigma_p_next = (1-rho_s) sigma_p + rho_s sigma_q stays > 0 iff rho_s in [0,1])
        if not (0.0 <= self.prior_handoff_rho <= 1.0):
            raise ValueError(f"prior_handoff_rho must be in [0,1], got {self.prior_handoff_rho}")
        if not (0.0 <= self.prior_handoff_sigma <= 1.0):
            raise ValueError(f"prior_handoff_sigma must be in [0,1], got {self.prior_handoff_sigma}")
        # normalization validated against the norm REGISTRY (add-by-registering). Local import
        # avoids a config <- norms cycle.
        from vfe3.geometry.norms import _NORMS
        _valid_norms = tuple(sorted(_NORMS))
        _require(self.norm_type_block, _valid_norms, "norm_type_block")
        _require(self.norm_type_final, _valid_norms, "norm_type_final")
        _require(self.routing_kl_max_reference, ("absolute", "row_min"), "routing_kl_max_reference")
        if not isinstance(self.block_mlp_post_norm, bool):
            raise ValueError(f"block_mlp_post_norm must be a bool, got {self.block_mlp_post_norm!r}")
        if self.block_mlp_post_norm and (self.block_coordinates != "local" or self.norm_type_block == "none"
                                         or not self.use_block_mlp):
            raise ValueError("block_mlp_post_norm=True needs block_coordinates='local', a block norm "
                             "(norm_type_block != 'none') and use_block_mlp=True")

        if self.block_mlp_expansion < 1:
            raise ValueError(
                "block_mlp_expansion must be an integer >= 1, "
                f"got {self.block_mlp_expansion!r}"
            )
        from vfe3.model.block_mlp import get_block_mlp_registration
        block_mlp_registration = get_block_mlp_registration(self.block_mlp_mode)
        if self.block_mlp_covariance not in block_mlp_registration.covariance_kinds:
            raise ValueError(
                f"block_mlp_covariance for mode {self.block_mlp_mode!r} must be one of "
                f"{sorted(block_mlp_registration.covariance_kinds)}, got "
                f"{self.block_mlp_covariance!r}"
            )
        if (isinstance(self.block_mlp_covariance_floor, bool)
                or not isinstance(self.block_mlp_covariance_floor, Real)
                or not math.isfinite(float(self.block_mlp_covariance_floor))
                or float(self.block_mlp_covariance_floor) <= 0.0):
            raise ValueError(
                "block_mlp_covariance_floor must be a finite real value > 0, "
                f"got {self.block_mlp_covariance_floor!r}"
            )
        if self.use_block_mlp and self.block_mlp_covariance == "delta_full" and self.family != "gaussian_full":
            raise ValueError(
                "block_mlp_covariance='delta_full' needs a full-covariance Gaussian family "
                "(family='gaussian_full'): the dense J Sigma J^T cannot be represented by a "
                f"diagonal/non-Gaussian family (got family={self.family!r})."
            )
        # The invariant Mahalanobis gate reads s_b = mu_b^T Sigma_bb^{-1} mu_b per irrep block. With a
        # passthrough covariance it is exact on the diagonal family (Sigma_bb = diag sigma_b gives
        # s_b = sum_k mu_k^2 / sigma_k), where it equals the full-family gate on diag_embed(sigma);
        # other families are not admitted.
        if (self.use_block_mlp
                and block_mlp_registration.gauge_class == "invariant_scalar_intertwiner"
                and self.family not in ("gaussian_full", "gaussian_diagonal")):
            raise ValueError(
                "block_mlp_mode='gauge_gate' needs a Gaussian family whose block covariance it can "
                "invert exactly (family='gaussian_full', or 'gaussian_diagonal' with "
                f"block_mlp_covariance='passthrough'); got family={self.family!r}."
            )
        if self.block_mlp_activation not in ("gelu", "silu", "relu"):
            raise ValueError(f"unknown block_mlp_activation {self.block_mlp_activation!r}")
        if (isinstance(self.block_mlp_dropout, bool)
                or not isinstance(self.block_mlp_dropout, Real)
                or not math.isfinite(float(self.block_mlp_dropout))
                or not 0.0 <= float(self.block_mlp_dropout) < 1.0):
            raise ValueError(
                "block_mlp_dropout must be a finite real value in [0, 1), "
                f"got {self.block_mlp_dropout!r}"
            )
        # prototype_agents knobs (validated in every mode; inactive_fields reports them off that mode).
        _require(self.block_mlp_inference, ("mean_field", "exact"), "block_mlp_inference")
        if (isinstance(self.block_mlp_template_init_variance, bool)
                or not isinstance(self.block_mlp_template_init_variance, Real)
                or not math.isfinite(float(self.block_mlp_template_init_variance))
                or float(self.block_mlp_template_init_variance) <= 0.0):
            raise ValueError(
                "block_mlp_template_init_variance must be a finite real value > 0, "
                f"got {self.block_mlp_template_init_variance!r}"
            )
        _require(self.block_mlp_template_init, ("absolute", "working_variance"), "block_mlp_template_init")
        if (isinstance(self.block_mlp_template_init_pull, bool)
                or not isinstance(self.block_mlp_template_init_pull, Real)
                or not 0.0 < float(self.block_mlp_template_init_pull) < 1.0):
            raise ValueError(
                "block_mlp_template_init_pull must be a real value in (0, 1), "
                f"got {self.block_mlp_template_init_pull!r}")

        # Reference coordinates (M2 arms: LN and MLP in reference coordinates; the covariant decoder).
        # Both conjugate by the realized relational frame U_i, which is defined, fixed through the
        # stack, and equal to the content frame on the projected routes without frame-space RoPE
        # (comparison-space RoPE rotates only the routing moments). Everything else fails closed.
        _require(self.block_coordinates, ("local", "reference"), "block_coordinates")
        _require(self.decode_coordinates, ("local", "reference"), "decode_coordinates")
        frame_requirements = []
        if self.pos_rotation != "none" and self.rope_space != "comparison":
            frame_requirements.append(
                f"pos_rotation={self.pos_rotation!r} (RoPE folds a rotation into the routing frame, "
                "so the content/decode frame is not U_i)")
        if self.block_coordinates == "reference":
            incompatible = list(frame_requirements)
            if self.attention_mode != "projected_predictive":
                incompatible.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive')")
            for name in ("norm_type_block", "norm_type_final"):
                if getattr(self, name) not in ("none", "layernorm"):
                    incompatible.append(f"{name}={getattr(self, name)!r} (requires 'none' or 'layernorm')")
            if self.use_cg_coupling:
                incompatible.append("use_cg_coupling=True (acts in local coordinates before the norm)")
            if self.use_block_mlp and self.block_mlp_mode not in ("coordinate", "gaussian_template",
                                                                  "prototype_agents"):
                incompatible.append(
                    f"block_mlp_mode={self.block_mlp_mode!r} (requires 'coordinate', 'gaussian_template' or "
                    "'prototype_agents'; canonical_frame is already conjugated and gauge_gate is already "
                    "covariant)")
            # 'posterior' (prototype_agents only, by the registry) maps C^+ back as U C^+ U^T.
            if self.use_block_mlp and self.block_mlp_covariance not in ("passthrough", "posterior"):
                incompatible.append(f"block_mlp_covariance={self.block_mlp_covariance!r} (requires "
                                    "'passthrough', or 'posterior' with prototype_agents)")
            if incompatible:
                raise ValueError(
                    "block_coordinates='reference' conjugates the block by the realized relational frame "
                    "and is defined only where that frame is; got " + "; ".join(incompatible))
        if self.decode_coordinates == "reference":
            incompatible = list(frame_requirements)
            if self.decode_mode not in _REFERENCE_DECODE_MODES:
                incompatible.append(
                    f"decode_mode={self.decode_mode!r} (requires one of {_REFERENCE_DECODE_MODES}: the "
                    "Gaussian-KL reduction to (xi, C) is exact only for these kernels)")
            if incompatible:
                raise ValueError(
                    "decode_coordinates='reference' decodes the reference moments through the realized "
                    "relational frame and is defined only where that frame is; got " + "; ".join(incompatible))

        # Per-type hidden bias on the block-MLP pre-activation (capacity-bearing per-type table).
        if self.per_type_hidden_bias:
            incompatible = []
            if not self.use_block_mlp:
                incompatible.append("use_block_mlp=False")
            elif self.block_mlp_mode not in ("coordinate", "canonical_frame"):
                incompatible.append(
                    f"block_mlp_mode={self.block_mlp_mode!r} (requires a coordinate hidden layer: "
                    "'coordinate' or 'canonical_frame')")
            if incompatible:
                raise ValueError(
                    "per_type_hidden_bias=True adds the per-type table to the block MLP's fc1 hidden "
                    "pre-activation; got " + ", ".join(incompatible))
            if self.m_per_type_hidden_bias_lr is None:
                self.m_per_type_hidden_bias_lr = declared_m_phi_lr

        # M-step / training
        if self.e_mu_q_trust is not None and not (
            math.isfinite(self.e_mu_q_trust) and self.e_mu_q_trust > 0.0
        ):
            raise ValueError(
                f"e_mu_q_trust must be finite and > 0 or None, got {self.e_mu_q_trust}")
        for name in ("m_p_mu_lr", "m_p_sigma_lr", "m_phi_lr", "weight_decay", "phi_weight_decay", "min_lr_frac"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        # Optional absolute rates: None inherits, otherwise a finite real >= 0 (never a bool).
        for name in ("m_relational_score_lr", "m_relational_content_lr",
                     "m_pos_phi_lr", "m_layernorm_lr", "m_decode_mu_lr",
                     "m_decode_sigma_lr", "m_block_mlp_lr", "m_per_type_hidden_bias_lr"):
            v = getattr(self, name)
            if v is not None and (
                    isinstance(v, bool) or not isinstance(v, Real)
                    or not math.isfinite(v) or v < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 or None, got {v!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}")
        if self.eval_interval < 0:
            raise ValueError(f"eval_interval must be >= 0, got {self.eval_interval}")
        if self.checkpoint_interval < 0:
            raise ValueError(f"checkpoint_interval must be >= 0, got {self.checkpoint_interval}")
        # max_steps=0 is an empty training loop and warmup_steps<0 is meaningless; warmup_steps=0
        # stays legal. (The plain-int type is gated at the top of __post_init__.)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be an int >= 1, got {self.max_steps!r}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be an int >= 0, got {self.warmup_steps!r}")
        if self.resume_from is not None and not isinstance(self.resume_from, str):
            raise ValueError(
                f"resume_from must be None or a path string to a checkpoints/step_<N>.pt, got "
                f"{type(self.resume_from).__name__}"
            )
        # amp_dtype: None (default, OFF) = pure fp32 / no autocast; 'bf16'/'fp16' enable autocast.
        # fp16 is accepted for FORWARD/inference; fp16 TRAINING still needs a GradScaler in the M-step,
        # so it is not enforced-rejected here (that would also block the fp16 inference path that
        # tests/test_amp.py pins).
        _require(self.amp_dtype, (None, "bf16", "fp16"), "amp_dtype")

        if not math.isfinite(self.unigram_kappa) or self.unigram_kappa < 0.0:
            raise ValueError(
                f"unigram_kappa must be finite and >= 0, got {self.unigram_kappa}")
        for name in ("sigma_weight_decay", "mu_weight_decay"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 or None, got {value}")

        # Activity is structural, independent of values and warning visibility. Notices only
        # display settings that differ from their defaults; sweep validation uses the full map.
        defaults = {field.name: field.default for field in fields(self) if field.default is not MISSING}
        inactive = self.inactive_fields()
        notices = [
            f"{name}={getattr(self, name)!r} ({reason})"
            for name, reason in inactive.items()
            if getattr(self, name) != defaults.get(name)
        ]
        if notices:
            import warnings
            warnings.warn(
                "inert configuration setting(s) -- these values are read by no active path, so "
                "changing them will have no effect:\n  - " + "\n  - ".join(notices),
                ConfigNotice, stacklevel=2,
            )

        # A phi update policy whose optimizer-group metadata overrides weight_decay replaces the
        # configured phi_weight_decay AFTER train.py sets it, and phi_weight_decay has no consumer in
        # run_artifacts -- so config.json would record a decay the optimizer never used.
        if self.phi_weight_decay != 0.0:
            from vfe3.gauge_optim import get_phi_update_policy
            _meta = get_phi_update_policy(self.m_phi_update_mode).optimizer_group_metadata
            if "weight_decay" in _meta and _meta["weight_decay"] != self.phi_weight_decay:
                import warnings
                warnings.warn(
                    f"phi_weight_decay={self.phi_weight_decay} is OVERRIDDEN to "
                    f"{_meta['weight_decay']} by the m_phi_update_mode={self.m_phi_update_mode!r} "
                    "policy metadata, for every phi-role group (phi_embed, pos_phi_free). "
                    "The configured value is what gets persisted, so treat the "
                    "recorded phi_weight_decay as nominal for this run.",
                    UserWarning, stacklevel=2,
                )

        predictive_moment_matched = (self.attention_mode == "projected_predictive"
                                     and self.predictive_covariance == "moment_matched")
        if self.decode_mode != "linear" and not predictive_moment_matched and (
            self.skip_belief_sigma_update or self.attention_mode == "projected_predictive"
        ):
            import warnings
            frozen_by = (
                "skip_belief_sigma_update=True"
                if self.skip_belief_sigma_update
                else f"attention_mode={self.attention_mode!r} (projected routes keep covariance)"
            )
            warnings.warn(
                f"prior-bank decode with frozen belief sigma ({frozen_by}): the KL decode READS "
                "sigma_q, so a frozen sigma changes predictions (sigma stays at encode). "
                "skip_belief_sigma_update is meant for configs where q's sigma is dead "
                "(linear decode, no norms); passthrough predictive readout keeps sigma.",
                UserWarning, stacklevel=2,
            )
        if self.grad_clip is not None:
            if (
                isinstance(self.grad_clip, bool)
                or not isinstance(self.grad_clip, (int, float))
                or not math.isfinite(float(self.grad_clip))
                or float(self.grad_clip) < 0.0
            ):
                raise ValueError(
                    "grad_clip must be None or a finite real value >= 0; "
                    f"got {self.grad_clip!r}"
                )

    def inactive_fields(self) -> dict[str, str]:
        """Return inactive config field names and reasons, including default-valued fields.

        The same structural result drives display notices and sweep admission. Consumers
        must not infer activity from whether a warning happens to be visible.
        """
        from vfe3.model.prior_bank import PriorBank, _DECODERS, _decode_family_chunked
        from vfe3.divergence import get_functional, renyi
        from vfe3.families.base import get_family
        from vfe3.families.gaussian import FullGaussian

        inactive: dict[str, str] = {}

        def mark(names: tuple[str, ...], reason: str) -> None:
            inactive.update(dict.fromkeys(names, reason))

        from vfe3.geometry.groups import group_traits
        if group_traits(self.gauge_group).algebra is None:
            mark(("irrep_basis", "irrep_spec_seed", "irrep_min_rank", "irrep_max_rank"),
                 "only read by irrep-tower groups (gauge_group='so_n')")
        from vfe3.gauge_optim import get_phi_update_policy
        phi_update_policy = get_phi_update_policy(self.m_phi_update_mode)
        if not phi_update_policy.requires_manifold_optimizer:
            mark(("m_phi_group_trust_radius", "m_phi_group_retraction"),
                 "only read by the manifold phi optimizer "
                 "(m_phi_update_mode='pullback_group' or 'fisher_group')")
        if not phi_update_policy.requires_manifold_optimizer:
            mark(("bch_residual_max",),
                 "only read by the manifold phi optimizer's group-product gate")
        if self.beta_attention_prior not in ("alibi", "causal_alibi", "causal_alibi_noself"):
            mark(("alibi_slope",), "only read by an alibi attention prior")
        if self.beta_attention_prior != "t5_relative_bias":
            mark(("t5_num_buckets", "t5_max_distance", "t5_learnable_bias"),
                 "only read by beta_attention_prior='t5_relative_bias'")
        if not self.use_cg_coupling:
            mark(("cg_covariance_mode", "cg_covariance_floor", "cg_energy_weight"),
                 "only read when use_cg_coupling=True")
        elif self.cg_covariance_mode == "passthrough":
            mark(("cg_covariance_floor",), "only read by cg_covariance_mode='delta_full'")
        if not phi_update_policy.requires_fisher_statistics:
            mark(("m_phi_fisher_rotation_weight",), "only read by m_phi_update_mode='fisher_group'")
        minimal_agent = self.predictive_readout == "minimal_agent"
        if minimal_agent:
            mark(("projected_factor_route",),
                 "predictive_readout='minimal_agent' passes no source tuples to its factorizations, so the "
                 "cholesky_qr2 factor route is never selected")
        elif self.relational_routing == "noisy_query":
            mark(("projected_factor_route",),
                 "the noisy-query cost factors per-pair sums G_q + G_k and keeps the Gram route")
        if self.decode_mode != "linear" or self.amp_dtype is None:
            mark(("decode_logits_amp",), "only read by decode_mode='linear' with amp_dtype set")
        if self.fp32_matmul_precision == "highest" or self.amp_dtype == "fp16":
            mark(("matmul_precision_retry",),
                 "only read under a reduced fp32_matmul_precision and without the fp16 GradScaler")
        if not self.deterministic:
            mark(("fill_uninitialized_memory",),
                 "torch fills uninitialized memory only under deterministic algorithms")
        gram_status = (self.projected_factor_route == "gram"
                       or self.relational_routing == "noisy_query" or minimal_agent)
        if not (self.projected_cov_condition_recovery and gram_status):
            mark(("projected_gram_condition_tol",),
                 "only read by the Gram-route certificate under projected_cov_condition_recovery=True")
        if not (self.projected_cov_condition_recovery and not gram_status):
            mark(("projected_factor_condition_tol",),
                 "only read by the cholesky_qr2 certificate under projected_cov_condition_recovery=True")
        if self.gauge_transport == "off":
            mark(("frame_pair_fp64_escalation", "frame_pair_condition_tol"),
                 "gauge_transport='off' fixes every frame to the identity (kappa_F = d), so the "
                 "certificate cannot trip")
        elif not self.frame_pair_fp64_escalation:
            mark(("frame_pair_condition_tol",), "only read when frame_pair_fp64_escalation=True")
        if self.attention_mode != "projected_predictive":
            mark(("predictive_covariance", "predictive_covariance_floor", "attention_residual",
                  "predictive_covariance_checkpoint"),
                 "only read by attention_mode='projected_predictive': the gradient receiver's covariance "
                 "comes from its receiver update, and its self-information enters through "
                 "alpha KL(q||p) and the closed-form damping")
        elif self.predictive_covariance == "passthrough":
            # The minimal agent checkpoints its whole readout (mean included), so only the floor is inert there.
            mark(("predictive_covariance_floor",) if minimal_agent
                 else ("predictive_covariance_floor", "predictive_covariance_checkpoint"),
                 "only read by predictive_covariance='moment_matched'")
        if self.attention_mode != "projected_predictive":
            mark(("predictive_readout",), "only read by attention_mode='projected_predictive'")
        if minimal_agent:
            posterior = ("exact source posterior" if self.minimal_agent_inference == "exact"
                         else "mean-field source posterior")
            mark(("relational_routing", "content_link_chart", "attention_residual", "relational_output_init",
                  "kappa_beta"),
                 f"predictive_readout='minimal_agent' routes by the {posterior} at temperature 1 (no "
                 "routing-cost selector, no kappa_beta temperature) and derives the receiving map and the "
                 "precision-weighted residual (no content factor O, so no chart, initial output scale or residual flag)")
            # The readout applies no clamp (relational.py returns before safe_kl_clamp); the remaining
            # readers are the M-step self-coupling and the CG moment energy.
            if self.mstep_self_coupling_weight == 0.0 and self.cg_energy_weight == 0.0:
                mark(("kl_max",),
                     "the minimal-agent posterior applies no divergence clamp, decoders use kl_max=inf, and "
                     "no M-step self-coupling or CG energy term is enabled")
            # Without the certificate the minimal agent pre-replaces failed factors before projected
            # scoring, so no source recovery (the only reader of the recovery precision) can occur.
            if not self.projected_cov_condition_recovery:
                mark(("projected_cov_recovery_precision",),
                     "the minimal agent recomputes failed layers itself; without "
                     "projected_cov_condition_recovery no source recovery runs")
            # Every contraction on this combination runs in an autocast-disabled island (the minimal-agent
            # update, the prototype posteriors, LayerNorm, the transport exponentials and the linear
            # decode/CE) or is elementwise, so the forward computes in float32 whatever amp_dtype says.
            # Checked bit-for-bit on CPU autocast (tests/test_audit_20261007_opus_fixes.py); not run on CUDA.
            if (self.amp_dtype is not None and self.decode_mode == "linear"
                    and (not self.use_block_mlp or self.block_mlp_mode == "prototype_agents")
                    and self.norm_type_block in ("none", "layernorm")
                    and self.norm_type_final in ("none", "layernorm")
                    and self.mstep_self_coupling_weight == 0.0 and not self.use_cg_coupling
                    and not self.decode_logits_amp):
                mark(("amp_dtype",),
                     "effective precision float32: on minimal_agent with a prototype_agents or absent block "
                     "MLP, a none/layernorm norm and linear decode, every contraction runs with autocast "
                     "disabled")
        else:
            mark(("minimal_agent_inference", "minimal_agent_channel_noise"),
                 "only read by predictive_readout='minimal_agent'")
        if self.content_link_edges == "off":
            mark(("content_link_edge_rank",), "only read when content_link_edges is 'lowrank' or 'pure_gauge'")
        if self.content_link_edges != "lowrank":
            mark(("content_link_latent",), "only read when content_link_edges='lowrank'")
        if not self.content_link_latent:
            mark(("content_link_latent_steps", "content_link_latent_sigma", "content_link_latent_damping",
                  "content_link_loop_weight"), "only read when content_link_latent=True")
        if self.content_link_loop_weight == 0.0:
            mark(("content_link_loop_window",), "only read when content_link_loop_weight > 0")
        if self.attention_mode != "projected_gradient":
            mark(("relational_receiver_noise", "relational_receiver_law", "relational_receiver_update",
                  "relational_receiver_damping"), "only read by projected_gradient receiving laws")
        elif self.relational_receiver_update == "closed_form":
            mark(("e_q_mu_lr", "e_q_sigma_lr", "e_mu_q_trust", "e_sigma_q_trust"),
                 "relational_receiver_update='closed_form' applies the exact frozen-context update; "
                 "its step control is relational_receiver_damping")
        else:
            mark(("relational_receiver_damping",), "only read by relational_receiver_update='closed_form'")
        gradient_controls = ("e_q_mu_lr", "e_q_sigma_lr", "e_mu_q_trust", "e_sigma_q_trust")
        if self.attention_mode == "projected_predictive":
            mark(("skip_belief_sigma_update",),
                 "projected routes run no sigma E-step: predictive_content keeps or moment-matches "
                 "covariance by predictive_covariance")
        if self.attention_mode == "projected_predictive":
            mark(gradient_controls, "predictive_content replaces the mean (and, under "
                 "predictive_covariance='moment_matched', the covariance) by readout, not a gradient step")
            if self.mstep_self_coupling_weight == 0.0:
                mark(("lambda_alpha_mode", "lambda_alpha", "b0", "c0"),
                     "predictive_content with mstep_self_coupling_weight=0 has no self-coupling consumer")
        if not self.use_block_mlp:
            mark(("block_mlp_mode", "block_mlp_covariance", "block_mlp_covariance_floor",
                  "block_mlp_expansion", "block_mlp_activation", "block_mlp_dropout", "m_block_mlp_lr"),
                 "only read when use_block_mlp=True")
        elif self.block_mlp_mode == "gaussian_template":
            mark(("block_mlp_activation", "block_mlp_covariance_floor"),
                 "gaussian_template gates by softmax over template KLs and keeps covariance")
        elif self.block_mlp_mode == "prototype_agents":
            mark(("block_mlp_activation", "block_mlp_covariance_floor"),
                 "prototype_agents has no hidden activation, and its posterior covariance is exact, "
                 "with no floor")
        if not (self.use_block_mlp and self.block_mlp_mode == "prototype_agents"):
            mark(("block_mlp_inference", "block_mlp_template_init_variance", "block_mlp_template_init",
                  "block_mlp_template_init_pull"),
                 "only read by block_mlp_mode='prototype_agents' with use_block_mlp=True")
        elif self.block_mlp_template_init == "working_variance":
            mark(("block_mlp_template_init_variance",),
                 "block_mlp_template_init='working_variance' sets lambda_0 from the measured working variance")
        else:
            mark(("block_mlp_template_init_pull",), "only read by block_mlp_template_init='working_variance'")
        if self.pos_phi != "learned":
            mark(("m_pos_phi_lr",), "only pos_phi='learned' creates the learnable positional table")
        if "layernorm" not in (self.norm_type_block, self.norm_type_final):
            mark(("layernorm_affine",), "only read by a 'layernorm' block or final norm seam")
        if not (self.layernorm_affine and "layernorm" in (self.norm_type_block, self.norm_type_final)):
            mark(("m_layernorm_lr",), "requires layernorm_affine and a layernorm seam")
        if not self.decode_unigram_prior:
            mark(("unigram_kappa",), "needs decode_unigram_prior=True")
        if self.lambda_alpha_mode == "constant":
            mark(("b0", "c0"), "only read by a state_dependent lambda_alpha_mode")
        else:
            mark(("lambda_alpha",), "only read by lambda_alpha_mode='constant'")
        if self.pos_rotation == "none":
            mark(("rope_on_value", "rope_base", "rope_space", "rope_fraction"), "only read by pos_rotation='rope'")
        if self.attention_mode != "projected_predictive" or self.predictive_readout != "kernel":
            mark(("routing_kl_max_reference",), "only read by the projected predictive kernel router")
        if self.pos_phi == "none":
            mark(("pos_phi_project_slk",), "only read by a non-'none' pos_phi")
        if (self.beta_attention_prior in ("windowed", "causal_windowed")
                and self.attention_window >= self.max_seq_len - 1):
            mark(("attention_window",), "window covers the context and is identical to plain causal")
        if self.n_layers == 1:
            mark(("prior_handoff_rho", "prior_handoff_sigma"), "no next layer consumes the handoff")
        registration = _DECODERS[self.decode_mode]
        if not registration.supports_chunked:
            mark(("decode_ce_checkpoint",), "the selected decoder has no checkpointed chunked CE kernel")
        canonical_family_dispatch = (
            self.decode_mode == "family_chunked"
            and registration.callable is _decode_family_chunked
            and registration.fused_ce is PriorBank.decode_ce_family_chunked
            and get_family(self.family) is FullGaussian
            and get_functional("renyi") is renyi
            and self.full_cov_kl_precision == "fp32_escalate"
        )
        if registration.family_consistent and not canonical_family_dispatch:
            mark(("decode_av_precision",), "generic family-consistent decoding does not call _decode_av")

        # The route-specific rules below were confirmed by tiny launcher-derived CPU probes: loss and
        # every parameter gradient stay torch.equal under an override of the marked knob
        # (tests/test_remediation_20260923_A.py). The self-coupling rule is exact in real
        # arithmetic; where an autograd oracle executes it (full-covariance families) the
        # override leaves only a floating-point roundoff residue in the gradients.
        if self.decode_mode == "linear":
            mark(("m_decode_sigma_lr",), "linear decode has no independent variance table")
            mark(("decode_av_precision", "decode_ranking_fp64_escalation"),
                 "read only by the prior-bank Gaussian KL decoders and their ranking guard")
            mark(("decode_tau",), "read only by the prior-bank KL decode")
            mark(("untie_decode_bank",),
                 "read only by the KL-to-bank decode; the linear decode is untied by construction")
        else:
            if not self.untie_decode_bank:
                mark(("m_decode_mu_lr", "m_decode_sigma_lr"),
                     "shared encode/decode tables use the encoder rates")
            mark(("decode_bias",), "linear decode only; the prior-bank KL decoders create no output bias")
        if self.beta_attention_prior not in ("windowed", "causal_windowed"):
            mark(("attention_window",), "only the windowed/causal_windowed priors read the band half-width")
        if self.gauge_transport in ("off", "frozen"):
            mark(("m_pos_phi_lr", "phi_weight_decay"),
                 f"gauge_transport={self.gauge_transport!r} sets requires_grad=False on the frame "
                 "tables, so no phi-role step or decay applies")
        if self.attention_mode == "projected_predictive" or self.skip_belief_sigma_update:
            mark(("e_q_sigma_lr", "e_sigma_q_trust", "sigma_max"),
                 "no sigma E-step runs on projected routes or under skip_belief_sigma_update "
                 "(a moment-matched predictive covariance is a readout with its own floor), so no "
                 "sigma step, trust region, or SPD ceiling runs")
        elif (self.relational_receiver_update == "closed_form" and self.family == "gaussian_full"):
            mark(("sigma_max",),
                 "the closed-form full-covariance receiver sets Sigma = c Lambda^{-1} with no SPD ceiling "
                 "(only the diagonal family clamps at sigma_max)")
        if self.attention_mode == "projected_predictive":
            mark(("lambda_beta",), "predictive_content weights no belief-coupling block")
        if self.decode_mode != "family_chunked" and not (
                self.family == "gaussian_full"
                and (self.mstep_self_coupling_weight != 0.0 or self.cg_energy_weight != 0.0)):
            mark(("full_cov_kl_precision",),
                 "projected scoring has its own factor policy; only the family_chunked decoder "
                 "evaluates the pair divergence")
        if "e_mu_q_trust" in inactive or self.e_mu_q_trust is None or self.diagonal_covariance:
            mark(("mu_trust_cholesky_rounds",),
                 "only an active mean trust region on a full covariance factors sigma_q")
        # stack._can_omit_prior_handoff at config level: predictive_content never reads the prior
        # and no M-step prior consumer is enabled.
        if (self.attention_mode == "projected_predictive" and self.mstep_self_coupling_weight == 0.0
                and self.cg_energy_weight == 0.0):
            mark(("prior_handoff_rho", "prior_handoff_sigma"),
                 "predictive_content never reads the prior and no M-step prior consumer is enabled")
        # The stack enters the E-step with q = p; one gradient iteration in one layer therefore
        # evaluates the self-coupling at D(q||p) = 0, where its force and derivatives vanish.
        # The closed-form receiver jumps to the frozen minimizer, which alpha sets, so the rule holds
        # only for a gradient step (audit 2026-10-07 F08/F23/F25/F71).
        if (self.n_e_steps == 1 and self.n_layers == 1 and self.mstep_self_coupling_weight == 0.0
                and not (self.attention_mode == "projected_gradient"
                         and self.relational_receiver_update == "closed_form")):
            mark(("lambda_alpha_mode", "lambda_alpha", "b0", "c0"),
                 "the single gradient E-step iteration starts at q = p, where the self-coupling "
                 "force vanishes")
        if (self.attention_mode == "projected_predictive" and self.family == "gaussian_full"
                and self.encode_mode in ("per_token", "per_token_additive")
                and self.predictive_covariance == "passthrough"
                and self.block_coordinates == "local" and self.decode_coordinates == "local"
                and not (self.use_block_mlp and self.block_mlp_covariance != "passthrough")
                and not (self.use_cg_coupling and self.cg_covariance_mode == "delta_full")):
            # A passthrough gauge gate no longer distinguishes the families: the diagonal gate on
            # (mu, sigma) is the full gate on (mu, diag_embed sigma), and nothing densifies sigma.
            mark(("family",),
                 "value-identical to gaussian_diagonal under the rank-matched decoder on this "
                 "route: stored covariance is a diagonal token lookup and no seam densifies it; "
                 "full covariance requires block_mlp_covariance='delta_full', "
                 "predictive_covariance='moment_matched', or attention_mode='projected_gradient'")
        # Reference coordinates. With identically-I frames xi = mu and C = Sigma exactly, so both
        # selectors are no-ops; without a block norm, final norm or block MLP nothing is conjugated.
        identity_frames = (self.gauge_transport == "off"
                           or (self.encode_mode == "per_token_additive" and self.pos_phi == "none"))
        if identity_frames:
            mark(("block_coordinates", "decode_coordinates"),
                 "the realized frames are identically I (gauge_transport='off', or per_token_additive "
                 "without positional frames), so reference and local coordinates coincide")
        elif (self.norm_type_block == "none" and self.norm_type_final == "none"
              and not self.use_block_mlp):
            mark(("block_coordinates",), "no block norm, final norm or block MLP to conjugate")
        if not self.per_type_hidden_bias:
            mark(("m_per_type_hidden_bias_lr",), "only read when per_type_hidden_bias=True")
        else:
            # The per-type table's optimizer group is decayed by phi_weight_decay even when the
            # frame tables are frozen (gauge_transport 'off'/'frozen').
            inactive.pop("phi_weight_decay", None)
        return inactive

    @property
    def relational_mode(self) -> str:
        """Internal compatibility view; attention_mode is the only writable selector."""
        return "projected_filtering"

    @property
    def relational_readout_mode(self) -> str:
        """Internal executor selection derived from attention_mode."""
        if self.attention_mode == "projected_gradient":
            return "full_link_update"
        return "predictive_content"

    @property
    def tau(self) -> float:
        """Per-head softmax-temperature convenience value tau = kappa * sqrt(d_head).

        NOTE: this is a config-level convenience (used for logging). The ACTIVE attention
        temperature is computed group-aware by ``free_energy.attention_tau(kappa, group.irrep_dims)``,
        which keys off the dimension the energy accumulates over -- the gauge-irrep BLOCK, sqrt(d_head)
        for block_glk. This property equals the active tau only for an equal-block group whose block
        size is d_head (block_glk). kappa=1 recovers the Vaswani sqrt(d_k) temperature over the
        energy dimension.
        When kappa is a per-head list, returns the mean tau (for logging only).
        """
        dims = self.head_dims
        if len(set(dims)) > 1:                       # mixed so_n tower: mean of kappa_h sqrt(d_h)
            kappas = (list(self.kappa_beta) if isinstance(self.kappa_beta, (list, tuple))
                      else [self.kappa_beta] * len(dims))
            return sum(k * d ** 0.5 for k, d in zip(kappas, dims)) / len(dims)
        k = float(sum(self.kappa_beta) / len(self.kappa_beta)) if isinstance(self.kappa_beta, (list, tuple)) else self.kappa_beta
        dimension = self.d_head
        return k * (dimension ** 0.5)

    @property
    def diagonal_covariance(self) -> bool:
        """Whether the belief covariance is diagonal -- DERIVED from ``family``.

        ``family`` is the single covariance-structure toggle; this returns its declared
        ``cov_kind == 'diagonal'`` (gaussian_diagonal -> True, gaussian_full -> False). Read-only:
        switch the covariance structure by setting ``family``, never this. Consumers that want a
        fast boolean (e.g. PriorBank) read it once at construction.
        """
        from vfe3.divergence import family_cov_kind
        return family_cov_kind(self.family) == "diagonal"

    @property
    def rope_insertion(self) -> Literal["right"]:
        """Fixed positional composition in the vertex frame."""
        return "right"

    @property
    def head_dims(self) -> List[int]:
        """Per-head belief dimensions: the irrep block sizes for an irrep tower (so_n), ``[K // n_heads] * n_heads`` otherwise."""
        from vfe3.geometry.groups import group_traits
        algebra = group_traits(self.gauge_group).algebra
        if algebra is not None:
            from vfe3.geometry.irreps import irrep_dim
            return [irrep_dim(self.group_n, algebra=algebra, label=label)
                    for label, mult in self.irrep_spec for _ in range(mult)]
        return [self.embed_dim // self.n_heads] * self.n_heads

    @property
    def head_offsets(self) -> List[int]:
        """First coordinate of each head, ``[0, d_0, d_0 + d_1, ...]`` (length n_heads)."""
        offsets = [0]
        for d in self.head_dims[:-1]:
            offsets.append(offsets[-1] + d)
        return offsets

    @property
    def d_head(self) -> int:
        """Per-head belief dimension K // n_heads; raises for a mixed irrep tower (read ``head_dims``)."""
        if len(set(self.head_dims)) > 1:
            raise ValueError(f"d_head is undefined for heads of different sizes {self.head_dims}; "
                             f"read head_dims")
        return self.embed_dim // self.n_heads

    def _chart_ordering_problem(self) -> Optional[str]:
        r"""Describe a violated manifold-phi chart ordering, or return None when it holds.

        The manifold phi policies need the M-step factor radius strictly inside the transport chart,
        ``phi_mstep_max_matrix_norm (or 5.0) < transport_chart_max_norm``.
        """
        factor_radius = (self.phi_mstep_max_matrix_norm
                         if self.phi_mstep_max_matrix_norm is not None else 5.0)
        if factor_radius < self.transport_chart_max_norm:
            return None
        return (f"a transport_chart_max_norm above the factor radius {factor_radius}; got "
                f"{self.transport_chart_max_norm!r}")


def serialize_config(config: VFE3Config) -> dict[str, Any]:
    """Serialize a config to its plain JSON-ready mapping of public fields."""
    payload = asdict(config)
    if math.isinf(config.kl_max):
        payload["kl_max"] = "inf"  # Strict JSON has no Infinity number.
    return payload


# Retired selectors pinned by the projected envelope: a saved value equal to the pinned one is
# dropped silently, any other value is refused.
_RETIRED_PINNED_FIELDS: Mapping[str, object] = {
    "divergence_family": "renyi", "renyi_order": 1.0, "include_attention_entropy": True,
    "gradient_mode": "filtering", "e_step_update": "gradient", "e_step_gradient": "unroll",
    "oracle_unroll_grad": True, "e_phi_lr": 0.0, "transport_mode": "flat",
}
# Retired fields no projected route read: any saved value is dropped silently.
_RETIRED_INERT_FIELDS = frozenset({
    "mm_damping", "full_cov_kl_backend", "full_cov_congruence_precision",
    "full_cov_congruence_certification", "congruence_cond_escalation", "rope_full_gauge",
    # 2026-10-02 transport-exponential unification: the clamp monitor watched the retired clamp;
    # the norm threshold changed only the interior precision of float32-stored exponentials.
    # (frame_pair_fp64_escalation, retired the same day, is a live opt-in field again since
    # 2026-10-04; a saved value loads as saved, and a config without it loads the default False.)
    "transport_clamp_monitor", "exp_fp64_norm_threshold",
    # 2026-10-05 head-mixer removal: the two mixer rates were read only by the retired mixers.
    "m_head_mixer_lr", "m_head_evidence_lr",
    # 2026-10-07 (PR #69): pullback_group was the only reader, and it required the group's registered
    # pullback_capability, which the M-step now dispatches directly; every other value was ignored or refused.
    "phi_precond_mode",
})
# Retired fields with a config-dependent rule in _migrate_retired_fields.
_RETIRED_CONDITIONAL_FIELDS = frozenset({
    "cross_couplings", "mu_trust_mode", "e_step_mu_precond", "m_decode_lr", "_allow_legacy_dense_decode",
    "decode_temperature_learned", "safe_cholesky_jitter_mode",
    # 2026-10-02 numerical-policy hardcoding.
    "decode_ce_stable_reduction", "decode_ranking_refinement", "predictive_covariance_centering",
    "min_lr",
    # 2026-10-05: the Schur head mixer and the PriorBank head-evidence mixer were removed.
    "use_head_mixer", "use_priorbank_head_evidence_mixer",
})
# Every serialized key config_from_serialized migrates or refuses that is not a VFE3Config field.
RETIRED_CONFIG_KEYS = frozenset(
    set(_RETIRED_PINNED_FIELDS) | _RETIRED_INERT_FIELDS | _RETIRED_CONDITIONAL_FIELDS
    | {"grad_clip_per_role", "exp_fp64_mode", "compile_pair_kernel", "relational_mode",
       "relational_readout_mode", "relational_input_scope", "relational_output_scope",
       "relational_identity_content"})
_RETIRED_FAMILIES = frozenset({"gaussian_frame_diagonal"})
_MIGRATED_FAMILIES = {"gaussian_diagonal_exact": "gaussian_diagonal"}
_RETIRED_ENCODE_MODES = frozenset({"canonical_content_gauge", "canonical_content_projected"})


def _same_serialized_value(value: object, expected: object) -> bool:
    """Exact equality for a pinned value, without letting bool stand in for a number."""
    if isinstance(expected, bool):
        return type(value) is bool and value is expected
    if isinstance(expected, float):
        return type(value) in (int, float) and value == expected
    return value == expected


def _migrate_retired_fields(
    payload: dict[str, Any],

    *,
    source: str,
) -> None:
    r"""Pop every retired field from ``payload`` in place; migrate it or refuse it.

    A retired selector is accepted silently when its saved value is the one the projected routes
    pinned, or when the field was inert for this config; any other value changed a forward this
    build no longer computes and is refused as retired. Value renames (dense decode modes,
    ``gaussian_diagonal_exact``, ``m_decode_lr``) are rewritten to their surviving equivalents.
    Must run after the attention selector and receiver-law defaults are resolved.
    """
    def refuse(name: str, value: object, detail: str = "") -> None:
        raise ValueError(f"serialized config from {source} requests retired {name}={value!r}"
                         f"{detail}; continuation requires explicit migration")

    for name in ("include_attention_entropy", "oracle_unroll_grad"):
        if name in payload:
            payload[name] = _normalize_serialized_boolean(payload[name], name, source=source)
    for name, pinned in _RETIRED_PINNED_FIELDS.items():
        if name in payload:
            value = payload.pop(name)
            if not _same_serialized_value(value, pinned):
                refuse(name, value, f" (only {pinned!r} survives)")
    for name in _RETIRED_INERT_FIELDS:
        payload.pop(name, None)
    payload.pop("_allow_legacy_dense_decode", None)

    if "cross_couplings" in payload:
        value = payload.pop("cross_couplings")
        if value not in (None, []):
            refuse("cross_couplings", value)
    attention_mode = payload.get("attention_mode")
    gradient_route = attention_mode == "projected_gradient"
    if "mu_trust_mode" in payload:
        value = payload.pop("mu_trust_mode")
        trust_read = (gradient_route
                      and payload.get("relational_receiver_update", "closed_form") == "euler"
                      and payload.get("e_mu_q_trust", VFE3Config.e_mu_q_trust) is not None)
        if value != "ball" and trust_read:
            refuse("mu_trust_mode", value, " (the mean trust region is the Mahalanobis ball)")
    if "e_step_mu_precond" in payload:
        value = payload.pop("e_step_mu_precond")
        if value != "fisher" and gradient_route:
            refuse("e_step_mu_precond", value, " (the receiver mean step is the Fisher natural gradient)")

    for name in ("use_head_mixer", "use_priorbank_head_evidence_mixer"):
        if name in payload:
            if _normalize_serialized_boolean(payload.pop(name), name, source=source):
                refuse(name, True, " (the head mixers were removed 2026-10-05)")

    decode_mode = payload.get("decode_mode")
    if isinstance(decode_mode, str) and decode_mode in _DENSE_DECODE_MIGRATIONS:
        payload["decode_mode"] = _DENSE_DECODE_MIGRATIONS[decode_mode]
    if "m_decode_lr" in payload:
        rate = payload.pop("m_decode_lr")
        if rate is not None:
            # Historical default: the decode temperature was learned unless the run froze it.
            learned = _normalize_serialized_boolean(
                payload.get("decode_temperature_learned", True),
                "decode_temperature_learned", source=source)
            linear = payload.get("decode_mode", VFE3Config.decode_mode) == "linear"
            if learned and not linear:
                refuse("m_decode_lr", rate, " (it also set the learned decode temperature rate)")
            # Linear decode has no variance table, so only its mean-role rate is migrated.
            for name in ("m_decode_mu_lr",) if linear else ("m_decode_mu_lr", "m_decode_sigma_lr"):
                if payload.get(name) is None:
                    payload[name] = rate

    if "transport_chart_max_norm" in payload and payload["transport_chart_max_norm"] is None:
        # The chart radius is required since 2026-10-02. None meant "no fail-closed bound, only the
        # surrogate clamp at 20"; it loads as the clamp's radius, so a frame the clamp would have
        # rescaled now raises instead.
        # A literal, not the class default: the historical clamp radius must not follow later
        # edits of the default.
        payload["transport_chart_max_norm"] = 20.0
    family = payload.get("family")
    if isinstance(family, str):
        if family in _RETIRED_FAMILIES:
            refuse("family", family)
        payload["family"] = _MIGRATED_FAMILIES.get(family, family)
    if "decode_temperature_learned" in payload:
        # Either value loads: decode_log_scale is restored from the weights and then held frozen, so
        # evaluation replays the saved temperature exactly. Resuming the optimizer of a run that
        # actually stepped the temperature is refused in run_artifacts.load_checkpoint.
        _normalize_serialized_boolean(
            payload.pop("decode_temperature_learned"), "decode_temperature_learned", source=source)
    if "safe_cholesky_jitter_mode" in payload:
        value = payload.pop("safe_cholesky_jitter_mode")
        if value != "relative":
            from vfe3.divergence import family_cov_kind
            try:
                diagonal = family_cov_kind(payload.get("family", VFE3Config.family)) == "diagonal"
            except Exception:
                diagonal = False
            if not diagonal:
                refuse("safe_cholesky_jitter_mode", value,
                       " (safe_cholesky's ridge is relative; only a diagonal family was inert to it)")
    # 2026-10-02 numerical-policy hardcoding. Both values of each of these loaded fields evaluate
    # the same mathematical quantity; the surviving arithmetic differs from a saved False/"full"
    # only at rounding level (max-centered CE reduction; float32 CE under the candidate-refined
    # ranking guard instead of a float64 promotion; mean-shifted predictive spread).
    for name in ("decode_ce_stable_reduction", "predictive_covariance_centering"):
        if name in payload:
            _normalize_serialized_boolean(payload.pop(name), name, source=source)
    if "decode_ranking_refinement" in payload:
        value = payload.pop("decode_ranking_refinement")
        if value not in ("full", "candidates"):
            refuse("decode_ranking_refinement", value, " (only 'full' or 'candidates' was valid)")
    if payload.get("decode_ce_checkpoint") in ("always", "never"):
        # A forced checkpoint policy only traded memory for backward recomputation.
        payload["decode_ce_checkpoint"] = "auto"
    if "min_lr" in payload:
        value = payload.pop("min_lr")
        if type(value) not in (int, float) or value != 0:
            refuse("min_lr", value, " (only 0 survives; a nonzero absolute floor changes the LR schedule)")
    encode_mode = payload.get("encode_mode")
    if isinstance(encode_mode, str) and encode_mode in _RETIRED_ENCODE_MODES:
        refuse("encode_mode", encode_mode)
    gauge_group = payload.get("gauge_group")
    if isinstance(gauge_group, str) and gauge_group in _RETIRED_GAUGE_GROUPS:
        refuse("gauge_group", gauge_group)


def config_from_serialized(
    payload: Mapping[str, Any],

    *,
    source: str,
) -> VFE3Config:
    """Build a config from a serialized mapping, refusing any field this build does not declare.

    Historical per-role clipping and norm precision migrate without changing behavior.
    Retired global clipping and dimension precision are refused. Historical attention pairs
    translate to one canonical selector; conflicting old/new selectors are refused. Retired
    selectors (see :func:`_migrate_retired_fields`) are dropped when pinned or inert and refused
    otherwise.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"serialized config from {source} must be a mapping")
    payload = dict(payload)
    if "grad_clip_per_role" in payload:
        per_role = _normalize_serialized_boolean(
            payload.pop("grad_clip_per_role"), "grad_clip_per_role", source=source)
        if not per_role:
            raise ValueError(f"serialized config from {source} requests retired global gradient clipping "
                             "(grad_clip_per_role=False); continuation requires an explicit policy change")
    if "exp_fp64_mode" in payload and payload.pop("exp_fp64_mode") != "norm":
        raise ValueError(f"serialized config from {source} requests a retired exp_fp64_mode; "
                         "only historical 'norm' is accepted without a numerical-policy change")
    # compile_pair_kernel was a pure wall-clock toggle: the eager and compiled pair kernels are
    # value-identical, so a recorded value cannot change a continuation's math. Drop it silently.
    payload.pop("compile_pair_kernel", None)
    # so_n configs saved before irrep_basis existed were built in the eigh basis (2026-10-05); the
    # learned tables live in that basis, so the reload must rebuild it rather than the new default.
    if payload.get("gauge_group") == "so_n" and "irrep_basis" not in payload:
        payload["irrep_basis"] = "eigh"
    if "relational_mode" in payload or "relational_readout_mode" in payload:
        mode = payload.pop("relational_mode", "off")
        readout = payload.pop("relational_readout_mode", "conditional_update")
        legacy_modes = {
            ("off", "conditional_update"): "transported",
            ("projected_filtering", "conditional_update"): "projected_conditional",
            ("projected_filtering", "predictive_content"): "projected_predictive",
        }
        try:
            attention_mode = legacy_modes[(mode, readout)]  # "projected_predictive" | "projected_gradient"
        except (KeyError, TypeError):
            raise ValueError(f"serialized config from {source} has an invalid historical attention pair "
                             f"({mode!r}, {readout!r})") from None
        if "attention_mode" in payload and payload["attention_mode"] != attention_mode:
            raise ValueError(f"serialized config from {source} has conflicting attention selectors")
        payload["attention_mode"] = attention_mode
    for name in ("relational_input_scope", "relational_output_scope"):
        if name in payload:
            scope = payload.pop(name)
            if scope not in ("full", None):
                raise ValueError(
                    f"serialized config from {source} requests retired head architecture "
                    f"({name}={scope!r}); head architecture requires explicit migration")
    if "relational_identity_content" in payload:
        identity = _normalize_serialized_boolean(
            payload.pop("relational_identity_content"), "relational_identity_content", source=source)
        if identity:
            raise ValueError(
                f"serialized config from {source} requests retired identity content; "
                "identity content requires explicit migration")
    if "attention_mode" not in payload:
        # A payload with no selector predates the attention-mode field; its historical default was
        # the retired transported route, so loading it as projected_predictive would silently
        # change the forward (audit 2026-10-01 F14).
        raise ValueError(f"serialized config from {source} names no attention selector; it predates "
                         "the projected routes and requires explicit migration")
    # Configs saved before the receiver-law fields keep the PR #30 receiver they ran.
    payload.setdefault("relational_receiver_law", "ambient_alignment")
    payload.setdefault("relational_receiver_update", "euler")
    _migrate_retired_fields(payload, source=source)
    config_fields = {field.name: field for field in fields(VFE3Config)}
    unknown = sorted(str(key) for key in payload if key not in config_fields)
    if unknown:
        raise ValueError(
            f"serialized config from {source} contains unknown field(s) {unknown}"
        )
    bool_fields = {
        name for name, field in config_fields.items()
        if field.type is bool or field.type == Optional[bool]
    }
    values = {
        name: (
            _normalize_serialized_boolean(
                value,
                name,
                source=source,
                optional_bool=config_fields[name].type == Optional[bool],
            )
            if name in bool_fields else value
        )
        for name, value in payload.items()
    }
    if values.get("kl_max") == "inf":
        values["kl_max"] = math.inf
    return VFE3Config(**values)


def _normalize_serialized_boolean(
    value: object,
    name:  str,

    *,
    source:        str,
    optional_bool: bool = False,
) -> 'bool | None':
    """Normalize the exact boolean spellings admitted by serialized configs."""
    if type(value) is bool or (value is None and optional_bool):
        return value
    if isinstance(value, str) and value.casefold() in ("false", "true"):
        return value.casefold() == "true"
    raise ValueError(
        f"serialized boolean field {name!r} from {source} must be a bool or the exact spelling "
        f"'true'/'false' (case-insensitive), got {value!r}"
    )


def _require(value: Optional[str], valid: Tuple[object, ...], name: str) -> None:
    """Raise ValueError unless ``value`` is one of ``valid``."""
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
