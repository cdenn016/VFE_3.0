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
# modularity contract). phi_precond_mode is validated against phi_preconditioner._PRECOND. Positional
# phi always uses exact group-product composition. Encode/decode modes likewise use their
# registries. The static tuples below are the remaining seams WITHOUT a registry.
_VALID_GAUGE_TRANSPORT     = ("on", "off", "frozen")
# Supported structure groups on the projected routes: block_glk, and so_n (with group_n and
# irrep_spec, one equal irrep block per head). The others are retired.
_SUPPORTED_GAUGE_GROUPS    = ("block_glk", "so_n")
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

    Fields are grouped: numerics, model structure, gauge seam, belief family,
    free-energy coupling, attention, E-step, decode/encode, cross-block handoff,
    normalization, and M-step / training. Each ``*_mode`` / ``*_family`` field is a
    registry key. Validation runs once in ``__post_init__`` in field-declaration order.
    """

    #################################
    #        Model structure
    #################################

    vocab_size:                int   = 50257
    embed_dim:                 int   = 64           # K (total belief dimension)
    max_seq_len:               int   = 128          # N (context length)

    # Held-out evaluation window stride. None = disjoint windows (stride == max_seq_len).
    # Must be <= max_seq_len: a larger stride skips transitions no target mask can recover.
    eval_stride:       Optional[int] = None

    batch_size:                int   = 64
    max_steps:                 int   = 15000
    warmup_steps:              int   = 100

    n_layers:                  int   = 1            # L (number of blocks)
    n_e_steps:                 int   = 1            # T (E-step iterations; 0 = zero-depth counterfactual, warns)
    n_heads:                   int   = 2

    seed:                      int   = 6
    deterministic:             bool  = True         # pins torch/cuDNN determinism; does NOT pin TF32 (see fp32_matmul_precision)
    # Under deterministic=True torch fills every uninitialized allocation with NaN (or the integer
    # maximum) so a read of uninitialized memory is visible and reproducible. False skips those fill
    # kernels: about 2,800 fewer kernels and 13.5 ms per K=64 step on the RTX 5090, with bit-identical
    # loss and gradients, at the cost of that guard. Inactive when deterministic=False.
    fill_uninitialized_memory: bool  = True

    log_interval:              int   = 100          # console log every N steps (0 = off)
    eval_interval:             int   = 1500         # periodic validation every N steps (0 = off)
    checkpoint_interval:       int   = 15000        # save a resumable checkpoint every N steps (0 = off)

    # Opt-in resume (default None = OFF = the pure from-scratch path). Set to a checkpoints/step_<N>.pt
    # to restore weights, AdamW state, RNG and the per-group cosine LambdaLR at step N, then continue.
    # An explicit train(resume_from=...) argument takes precedence over this field.
    resume_from:               Optional[str] = None

    # Resume bundles load with torch.load(weights_only=True). If that safe load fails on a (e.g.
    # legacy-format) bundle, resume RAISES unless this is True, permitting the legacy
    # weights_only=False load -- only for a checkpoint you trust, since that path can execute
    # arbitrary code embedded in the pickle.
    trust_resume_checkpoint:   bool          = False

    #################################
    #         Belief family
    #################################

    # Belief covariance STRUCTURE (registry key), the single source of truth for diagonal-vs-full.
    # Read the derived read-only ``diagonal_covariance`` property, never a second flag. The
    # divergence is KL (Renyi order 1) on every projected route.
    family:                    str   = "gaussian_diagonal"   # "gaussian_diagonal" | "gaussian_full"

    #################################
    #         Initialization
    #################################

    # Belief-table init (PriorBank.__init__): mu_v ~ N(0, mu_init_std^2); every coordinate variance is
    # the constant sigma_init (stored as log(sigma_init), not randomly spread); phi_v ~ N(0, phi_scale^2).
    # mu_init_std and phi_scale may be 0; sigma_init must be > 0 (the log is taken).
    mu_init_std:               float = 0.02         # std of the random mean table mu_embed
    sigma_init:                float = 1.0          # constant initial coordinate variance (sigma_log = log of this)
    phi_scale:                 float = 0.01         # std of the random gauge-frame table phi_embed
    pos_phi_scale:             float = 0.02         # learned-table init scale AND frozen per-position step

    #################################
    #         Encode / Decode
    #################################

    # Encoder (registry key).
    encode_mode:               str   = "per_token"   # "per_token" | "per_token_additive"

    # Decoder (registry key). "linear" is the learned (V,K) readout logits = mu_q @ W^T (+ bias),
    # discarding sigma. Every other mode reads out against the prior bank with a chunked kernel that
    # streams the vocabulary and never materializes (B,N,V) in training.
    decode_mode:               str   = "linear"      # "linear" | "diagonal_chunked" | "full_chunked"
                                                     # | "family_chunked" | "expected_likelihood_chunked"

    # Coordinates the decoder reads. "local" (default) decodes the local moments (mu_i, Sigma_i).
    # "reference" decodes the reference moments xi_i = U_i^{-1} mu_i and C_i = U_i^{-1} Sigma_i U_i^{-T}
    # with the same realized relational frame the attention reads: linear logits W_out xi_i + b
    # (+ kappa log p_uni), a full Gaussian decode KL(N(xi_i, C_i) || p_v), and a diagonal Gaussian decode
    # of diag C_i (equal to the full-C KL up to a per-position constant that cancels in the softmax).
    # Applies on the fused training CE and the dense logits path alike. Requires a projected
    # attention_mode, pos_rotation='none', and decode_mode in linear/diagonal*/full*.
    decode_coordinates:        str   = "local"       # "local" | "reference"

    decode_tau:                float = 1.0           # KL-to-prior decode temperature; inert under decode_mode="linear"
    # decode_log_scale is always frozen, so tau_eff == decode_tau exactly (the learned temperature,
    # decode_temperature_learned, was retired 2026-10-02).

    # Unigram log-prior decode (Bayes-correct class prior): adds kappa*log pi_v to the decode logits,
    # pi_v the corpus unigram frequency -- a fixed DATA statistic, not a learned parameter. train()
    # sets the table from the training stream; unset (all zeros) it is a warned no-op. Applies to
    # every decode path (bank KL and linear).
    decode_unigram_prior:      bool  = True
    unigram_kappa:             float = 1.0           # tempering on log pi_v (1.0 = exact Bayes rule)
    decode_bias:               bool  = True          # linear decode only: learned per-vocab log-unigram bias
                                                     # on logits = mu_q @ W^T + b (zero-init, weight-decay-free)

    # Untied decode bank (prior-bank modes only): decode reads its OWN (V,K) mu/sigma tables, cloned
    # from the encode tables at init (step-0 byte-identical) and trained separately. Isolates the
    # tied-vs-untied confound; still a KL-to-priors decode.
    untie_decode_bank:         bool  = True

    # Vocabulary-chunk width for the fused CE kernels -- the training-path memory win that never
    # materializes the (B,N,V) logit tensor. Read by every decode_ce_*_chunked kernel plus the two
    # logits kernels. The unit is vocabulary entries, but an entry is not equally expensive
    # everywhere: the (B,N,Vc) kernels pay one scalar per entry, while a full family on the
    # family-consistent route pays a (K,K) functional workspace per entry twice over. That route
    # therefore re-reads this value in its own units and narrows its slice to a fixed byte budget;
    # the number here is an upper bound for it, not the width it uses. Narrowing is value- and
    # gradient-identical (the vocabulary reduction is a logsumexp over independent slices).
    decode_chunk_size:         int   = 8192

    # Checkpointing trades backward recomputation for retained decoder memory.
    # "auto" keeps historical per-chunk accounting for analytic/expected/linear CE.
    # Generic family CE already uses aggregate retention. "auto_aggregate" explicitly opts
    # into whole-vocabulary accounting, including fp64 scores and expected 2K worksets.
    # Both automatic policies use a 2 GiB threshold. The forced "always"/"never" policies are
    # PriorBank-internal (tests); a saved config carrying either loads as "auto".
    decode_ce_checkpoint:       str  = "auto"  # "auto" | "auto_aggregate"

    # Strict decoder ranking guard. When enabled, analytic fp32 decoders construct conservative score
    # intervals for the logits decode. It resolves an uncertain ranking by candidate refinement
    # (optimization_plan.md WS5): only the candidates whose float32 interval reaches the winner's, in only the uncertain rows,
    # are recomputed in float64, giving the same argmax as a full float64 recompute at
    # O(candidates * K). The fused training CE never reads the ranking and streams in float32.
    decode_ranking_fp64_escalation:             bool = True

    #################################
    #          Gauge Group
    #################################

    # Structure group of the flat phi-cocycle Omega_ij = exp(phi_i) exp(-phi_j). block_glk: one
    # independent GL(d_head) frame per head. so_n: one shared SO(group_n) element acting through
    # the irrep blocks of irrep_spec, which on the projected routes must be n_heads equal blocks.
    gauge_group:               str   = "block_glk"   # "block_glk" | "so_n"

    # Ablation meta-toggle over the GL(K) gauge FRAME (whether the per-token frame exists at all).
    #   "on"     (default, PURE PATH) the frame is whatever phi_scale / m_phi_lr / pos_phi specify
    #            (learned when m_phi_lr > 0); byte-identical to omitting the toggle.
    #   "off"    Omega_ij = exp(phi_i) exp(-phi_j) = I EXACTLY, by coercing phi_scale=0.0,
    #            pos_phi='none', m_phi_lr=0.0 -- and REQUIRING pos_rotation='none', since a
    #            positional rotation would make Omega != I (raises).
    #   "frozen" a RANDOM but FIXED frame: keeps phi_scale>0, coerces m_phi_lr=0.0.
    # __post_init__ applies the coercion and warns. NB 'off' drops the pos_phi_free table, so to match
    # parameter counts against an 'on' baseline set pos_phi='none' in 'on' too, or compare against the
    # equal-parameter 'frozen' arm. The projected routes BUILD their frames from phi, so all three
    # arms are reachable.
    gauge_transport:           str   = "on"          # "on" | "off" | "frozen"

    # SO(N) irrep-tower group ('so_n') only. The structure group is SO(group_n) with group_n
    # DECOUPLED from embed_dim. The embedding carries the direct sum of the irreps in irrep_spec --
    # (label, multiplicity) pairs laid out contiguously in order; heads = irrep blocks, which the
    # projected routes require to be n_heads blocks of equal dim d_head. Labels 'l<p>' are the symmetric-traceless rank-p tensor irreps (group_n=3: the
    # spin-p tower, dim 2p+1). sum(mult * dim) must equal embed_dim. One shared per-token phi
    # (n_gen = dim of the algebra) drives every block -- a TIED gauge. Both fields are consumed ONLY
    # by so_n (rejected otherwise).
    group_n:                      Optional[int] = None

    # The annotation reflects the POST-coercion type: __post_init__ also accepts the JSON/TOML
    # round-trip list-of-lists form ([["l1", 3], ...]) and coerces each pair to a tuple.
    irrep_spec: Optional[List[Tuple[str, int]]] = None

    # CG cross-type coupling (opt-in, default off; so_n only): bilinear Clebsch-Gordan
    # between-block update on the means, exactly equivariant for any weights; sigma untouched
    # (means-only phase). NEURAL-NETWORK EXCEPTION (sanctioned, default-off): learned scalar path
    # weights, zero-init (step 0 byte-identical).
    use_cg_coupling:                   bool = False

    # CG covariance pushforward (opt-in; default 'passthrough' = the means-only phase, sigma
    # untouched). 'delta_full' applies the delta-method Gaussian moment closure sigma_out =
    # sym(J Sigma J^T), with J the exact analytic Jacobian of the bilinear CG map at the current
    # mean -- a FIRST-ORDER pushforward, NOT an exact distributional image. It requires a
    # full-covariance Gaussian family (family='gaussian_full'), since J Sigma J^T is dense.
    cg_covariance_mode:        str = "passthrough"   # "passthrough" | "delta_full"
    cg_covariance_floor:       float = 1e-6          # delta_full: J Sigma J^T + floor * Sigma

    # CG moment-energy regularizer weight (opt-in; default 0.0 = off, everything untouched). When
    # > 0 the outer objective adds ONCE cg_energy_weight * mean_layers(mean_tokens(D(q_post||q_pre))),
    # the post-CG moment divergence of the belief against its pre-CG value -- a q-only regularizer
    # that never reweights the canonical q/p/s/h hierarchy. Requires use_cg_coupling=True.
    cg_energy_weight:          float            = 0.0

    ####################################
    #       Positional Encoding
    #    BCH gauge-frame PE (pos_phi)
    #     gauge-RoPE (pos_rotation)
    ####################################

    # Group-product positional encoding: a per-position Lie-algebra element pos_phi_i composed into
    # the token gauge frame before transport. "learned" owns a model parameter table
    # (max_seq_len, n_gen); "frozen" is the parameter-free i*pos_phi_scale on one axis. The
    # theoretically PURE no-composition path is "none".
    pos_phi:                   str             = "learned"   # "learned" | "frozen" | "none"
    bch_residual_max:          Optional[float] = None        # fail-closed BCH/group-product relative residual
    pos_phi_project_slk:       bool            = False       # per-block trace projection (det Omega = 1)

    # Gauge-RoPE (default-off): a block-diagonal positional rotation R(theta) folded into each
    # realized frame on the right, V_i = U_i R(theta_i)^T, which carries both moments on the
    # projected routes.
    pos_rotation:              str   = "none"      # "none" | "rope"
    rope_base:                 float = 100.0       # rotary frequency base

    # True (default) = the coherent single-gauge path: the gauge-RoPE rotation feeds BOTH the
    # routing frame and the content/value frame. False keeps routing on U_i R_i^T and moves content
    # on U_i -- RoPE's "position-dependent attention, position-independent values" asymmetry.
    # Inert unless pos_rotation='rope'.
    rope_on_value:             bool  = True         # False -> value aggregation uses the un-rotated base

    #################################
    #          Methods
    #################################

    # Full-state projected attention uses predictive content or frozen-receiver gradients.
    # Projection ranks still follow the gauge block dimensions.
    # Projected attention requires the explicit support envelope validated below.
    attention_mode:                      str           = "projected_predictive" # "projected_predictive" | "projected_gradient"
    attention_residual:                  bool          = False                  # requires predictive_content
    relational_output_init:              str           = "identity"             # "identity" | "depth_scaled" (initial O times 1/sqrt(2 n_layers))

    # Receiving-map chart of the content link on the stored (O, T) (attention_unified.tex, "The tiled
    # message order by order"): "affine" O' = O (current head sum); "first_order" O' = T^{-1} + O (flat
    # head-routed transport plus the head sum); "exponential" O' = T^{-1} + O phi_1(TO) = e^{OT} T^{-1}
    # (Option A shared-generator exponential link, O'T = e^{OT}). Every chart starts at the same content
    # s B_ij (s from relational_output_init). O = 0 means zero content (affine) or flat transport
    # (first_order, exponential); the content group has no weight decay. Both projected routes read O'.
    content_link_chart:                  str           = "affine"              # "affine" | "first_order" | "exponential"
    relational_receiver_noise:           float         = 1.0                   # projected_gradient: Q_h = variance * I (K-dim reference law under ambient, d-dim channel floor under channel)
    # projected_gradient receiver energy and update (projected_gradient.tex). "channel_likelihood" scores
    # O_h^T xi_i against the d-dim message T_h xi_j as an expected likelihood (entropy counted once, no
    # complement pinning); "ambient_alignment" is the PR #30 sum of KLs to K-dim noisy laws. "closed_form"
    # applies the exact frozen-context natural-parameter update with damping rho in (0, 1] (rho = 1 is the
    # minimizer); "euler" is the PR #30 explicit Fisher step governed by e_q_mu_lr / e_q_sigma_lr and trusts.
    relational_receiver_law:             str           = "channel_likelihood"  # "channel_likelihood" | "ambient_alignment"
    relational_receiver_update:          str           = "closed_form"         # "closed_form" | "euler"
    relational_receiver_damping:         float         = 1.0                   # closed_form only: rho in (0, 1]
    relational_routing:                  str           = "gaussian_kl"          # "gaussian_kl" | "noisy_query" (projected_predictive only)
    # Belief covariance after the predictive readout. "passthrough" keeps the incoming covariance
    # (a per-token-type routing state); "moment_matched" replaces it by the exact moment-matched
    # covariance of the predictive kernel (GL(K) v2 Theorem thm:probabilistic_status (2)): source
    # covariances pushed through M_h = O_h T_h, cross-head terms, between-source spread, the
    # residual persistence term when attention_residual=True, and the covariant reference floor
    # floor * (tr C_i / K) I (local floor * (tr C_i / K) U_i U_i^T).
    predictive_covariance:               str           = "passthrough"          # "passthrough" | "moment_matched" (projected_predictive only)
    predictive_covariance_floor:         float         = 1e-4                   # Q~_i = floor * (tr C_i / K) I, keeps Sigma^+ positive definite and well conditioned
    # Recompute the moment-matched covariance update in backward instead of saving its dense
    # (B, N, K, K) intermediates. Values and gradients are bit-identical. At K=256, H=8, B=16 it
    # took peak memory from 46.3 to 23.0 GB and the step from 12.0 s (the Windows driver silently
    # spilling to shared system memory instead of raising OOM) to 0.83 s, measured on the dense
    # update before the blockwise route (2026-10-04) removed its K x K intermediates. Worth it at large K.
    predictive_covariance_checkpoint:    bool          = False

    ######################################
    #                Self Energy:
    #        Sum_i alpha_i * KL(q_i||p_i)
    ######################################

    # Self-coupling form (registry key). The default-and-pure no-NN forms are closed-form functions
    # of the self-divergence D, with no learned parameters.
    lambda_alpha_mode:         str   = "constant"   # "constant" | "state_dependent" | "state_dependent_per_coord"

    b0:                        'float | List[float]' = 1.0   # state-dependent alpha shape: alpha* = c0/(b0 + D); list -> (K,) per-coord
    c0:                        'float | List[float]' = 1.0   # state-dependent alpha shape (numerator); list -> (K,) per-coord

    ##########################################################
    #              VFE Hyper-parameters
    #  Sum_i alpha_i * KL(q_i||p_i) +
    #  lambda_beta*Sum_i beta_ij * KL(q_i||Omega_ij q_j) + ...
    ##########################################################

    # Weight on the ENTIRE belief-coupling block of F -- sum_ij [ beta_ij E_ij +
    # tau beta_ij log(beta_ij/pi_ij) ] -- relative to the alpha self-coupling and the likelihood.
    # 1.0 = the canonical/pure F (byte-identical). It scales the POST-softmax block (NOT the energy
    # inside the softmax), so beta = softmax(-E/tau) is unchanged.
    lambda_beta:                float = 1.0
    lambda_alpha:               float = 1.0          # constant self-coupling value
    mass_phi:                   float = 0.0          # (mass_phi/2) ||phi||^2 penalty
    mstep_self_coupling_weight: float = 0.0          # M-step sum_i alpha_i D(q_i*||p_i) (0 = OFF)

    ########################################
    #     Attention Belief/Model Settings
    #            & Temperatures
    ########################################

    kappa_beta: 'float | List[float]' = 1.0   # sharpness; list (len n_heads, or n irrep blocks) -> per-head tau

    beta_attention_prior:      str   = "causal_alibi"  # "causal_alibi" | "causal_alibi_noself" | "alibi" | "causal"
                                                       # | "causal_noself" | "causal_windowed" | "windowed"
                                                       # | "uniform" | "t5_relative_bias"

    # BASE MULTIPLIER on the Press schedule 2^(-8h/H), NOT a slope: the REALIZED per-head slopes are
    # alibi_slope * 2^(-8h/H). Consequence for experiment design: an n_heads sweep at FIXED
    # alibi_slope silently rescales the positional prior (head-0 span 3.94/0.246 nats at H=2, and up
    # to 180x across an H sweep), so pin the realized head-0 slope, not this base. Kept named
    # alibi_slope: renaming it is a config-schema change that would break every config.json on disk.
    alibi_slope:               float = 1.0           # BASE multiplier for alibi/causal_alibi priors
    attention_window:          int   = 128           # band half-width for the windowed/causal_windowed priors

    t5_num_buckets:            int   = 32            # t5_relative_bias: relative-position bucket count
    t5_max_distance:           int   = 128           # t5_relative_bias: log-bucketing horizon (beyond -> last bucket)
    t5_learnable_bias:         bool  = False         # t5_relative_bias: learn the per-bucket bias table
                                                     #   (sanctioned no-NN exception, default OFF)

    #################################
    #     Layer Normalization
    #        and Hand-Off
    #################################

    # Normalization (registry key). "none" (identity) is the gauge-pure default; "mahalanobis" is a
    # gauge-invariant rescale; "layernorm" is the standard parameter-free transformer LayerNorm over
    # the belief dim -- OPT-IN and NON-gauge-equivariant, since centering and per-coordinate variance
    # in the fixed basis do not commute with g in GL(K): LN(g mu) != g LN(mu). Set
    # layernorm_affine=True for the learned gamma/beta variant. 'block' runs per block inside the
    # E-step; 'final' runs once post-stack before decode.
    norm_type_block:            str   = "layernorm"   # "layernorm" | "mahalanobis" | "none"
    norm_type_final:            str   = "none"        # "layernorm" | "mahalanobis" | "none"

    # Coordinates the block norm, the block MLP and the final norm act in. "local" (default) applies
    # them to the local mean mu_i. "reference" applies them to the reference mean xi_i = U_i^{-1} mu_i
    # and maps back, mu -> U_i Phi(U_i^{-1} mu), with ONE conjugation per block (LN and MLP together) by
    # the realized relational frame U_i = exp(X_{x_i}) exp(Y_i); a Gaussian-template MLP reads
    # diag C_i, C_i = U_i^{-1} Sigma_i U_i^{-T}. This is the training counterpart of M5 row 11. Frame
    # actions run with autocast off (float32, float64 for an escalated frame); the MLP keeps the
    # ambient autocast. Requires attention_mode='projected_predictive', pos_rotation='none', a
    # 'none'/'layernorm' norm, and a passthrough 'coordinate'/'gaussian_template' MLP.
    block_coordinates:          str   = "local"       # "local" | "reference"

    # Cross-block belief handoff (mu_q -> mu_p). Both blends must be convex so the prior stays on the
    # SPD cone: sigma_p_next = (1-rho_s) sigma_p + rho_s sigma_q stays > 0 iff rho_s in [0,1].
    prior_handoff_rho:          float = 1.0           # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma:        float = 0.0           # sigma damping (0.0 = frozen at embedding)

    # Optional residual feed-forward unit applied after each belief block.
    use_block_mlp:              bool  = True
    block_mlp_mode:             str   = "coordinate"  # "coordinate" | "gauge_gate" | "canonical_frame" | "gaussian_template" (passthrough only)
    block_mlp_covariance:       str   = "passthrough" # "passthrough" | "delta_full"
    block_mlp_activation:       str   = "gelu"        # "gelu" | "silu" | "relu"

    block_mlp_covariance_floor: float = 1e-4         # covariant SPD floor: Sigma+ = J Sigma J^T + floor Sigma
    block_mlp_expansion:        int   = 8
    block_mlp_dropout:          float = 0.0

    m_block_mlp_lr:   Optional[float] = None

    # Per-type hidden bias (capacity-bearing per-type table): an nn.Parameter E of shape (V, n_gen),
    # n_gen the gauge-generator count (the token-frame table's width), zero-initialized so step 0 equals
    # the model without it. E[x_i] is gathered once per forward and split over the layers into widths
    # w_l = floor(n_gen / L) (+1 on the first n_gen mod L layers); layer l adds its slice to the first
    # w_l hidden pre-activation units of the block MLP, h^(l) = W_1 u + b_1 + E^(l)[x_i]. It enters
    # after the norm and inside the nonlinearity at every layer, so it is not absorbable into mu_v.
    # Requires a projected attention_mode, use_block_mlp with a 'coordinate' or 'canonical_frame' MLP,
    # and w_l <= its hidden width.
    per_type_hidden_bias:       bool  = False
    # Step size of that table; None resolves to the DECLARED m_phi_lr (read before gauge_transport
    # 'off'/'frozen' coerces m_phi_lr to 0), the phi-table hyperparameter it mirrors. Its decay is
    # phi_weight_decay.
    m_per_type_hidden_bias_lr:  Optional[float] = None

    # SANCTIONED LEARNED-SCALAR EXCEPTION (default ON; inert, no params, warned, unless a norm seam is
    # "layernorm"). Adds a learned per-feature affine gamma/beta to any "layernorm" seam:
    # mu_norm = gamma*LN(mu) + beta, exactly torch.nn.LayerNorm(elementwise_affine=True) on the belief
    # mean. gamma inits to 1 and beta to 0, so a learnable model is BYTE-IDENTICAL to the
    # parameter-free "layernorm" at construction. gamma/beta are raw nn.Parameter diagonal scale/shift
    # tables (shared across blocks for the block seam, a separate pair for the final seam), NOT an
    # nn.Linear/MLP -- no hidden network -- and they stay non-gauge-equivariant exactly as the
    # parameter-free "layernorm" does. As the BLOCK norm the affine is applied to the belief VALUE
    # inside the stack, so it trains under 'unroll' AND 'straight_through' and is frozen only by the
    # fully-detached E-step; as the FINAL norm it is post-stack and trains under any estimator.
    # Grouped in build_optimizer at m_p_mu_lr, weight_decay=0, role='mu'.
    layernorm_affine:          bool  = True
    m_layernorm_lr:  Optional[float] = None

    #################################
    #
    #             E-step
    #
    #################################

    # Receiver step sizes of the projected_gradient 'euler' update. The mean step is the Fisher
    # natural gradient nat_mu = Sigma grad_mu, differentiated through the unrolled receiver with
    # each source row frozen (filtering). The frame is fixed inside the stack (no E-step phi step).
    e_q_mu_lr:                 float = 0.5
    e_q_sigma_lr:              float = 0.015

    phi_precond_mode:             str   = "none"      # "none" | "clip" | "killing" | "killing_per_block"
                                                      # | "pullback" | "pullback_per_block"

    # Skip the belief-channel sigma E-step update (gradient + retraction) entirely. OPT-IN dead-compute
    # ablation for configs where q's sigma has no consumer (linear decode, no norms); NOT coerced from
    # the config -- the user asserts deadness. Warns under a prior-bank decode, which reads sigma_q.
    skip_belief_sigma_update:  bool  = True

    #################################
    #             M-step
    #################################

    m_p_mu_lr:                 float = 0.025
    m_p_sigma_lr:              float = 0.0025
    m_phi_lr:                  float = 0.015

    # Independent rates for untied decode means (or linear weight/bias) and log variances.
    # None inherits the corresponding encoder rate. Shared encode/decode tables retain encoder
    # rates. These fields do not change temperature. A historical combined m_decode_lr migrates
    # into whichever of the two is None on load.
    m_decode_mu_lr:            Optional[float] = None
    m_decode_sigma_lr:         Optional[float] = None

    # Optional absolute M-step rate for the LEARNED POSITIONAL frame table (pos_phi='learned'), which
    # otherwise rides m_phi_lr with phi_embed. The two tables compose into ONE clamped frame and so
    # share phi_weight_decay, but they are different objects: phi_embed is a (V, n_gen) codebook
    # gathered by token id with heavy-tailed row visitation, while pos_phi_free is (max_seq_len,
    # n_gen) with every row live every step -- their per-row gradient magnitudes are not comparable.
    # Their init scales are already separate (phi_scale vs pos_phi_scale), so the step size need not
    # coincide either. None inherits m_phi_lr, reproducing the historical yoked behavior byte-for-byte.
    m_pos_phi_lr:               Optional[float] = None

    m_relational_score_lr:      Optional[float] = None  # None inherits m_p_mu_lr.
    m_relational_content_lr:    Optional[float] = None  # None inherits m_p_mu_lr.

    # Stored phi-factor M-step policy (registry key). "adamw" is the default and preserves the
    # established optimizer topology. "pullback_group" opts into a stateless strict pullback
    # direction followed by a local right-group retraction; the trust radius bounds the
    # learning-rate-weighted right factor in the embedded Frobenius norm. The strict path is
    # intentionally limited by validation below.
    m_phi_update_mode:         str       = "adamw"    # "adamw" | "pullback_group" | "fisher_group"
    m_phi_group_trust_radius:  float     = 0.1
    # Retraction of the manifold phi step. "chart" is the additive chart step phi - eta v_phi with
    # its product-residual backtracking (the historical route, unchanged). "group_log" is the exact
    # right product phi_new = Log(exp(phi) exp(-eta s xi)) with the closed-form U-space pullback
    # direction (series direction off the principal strip); it halves only on log, residual,
    # descent or chart-bound failure (derivation 2026-09-28, Track 1). Untied block_glk only.
    m_phi_group_retraction:    str       = "chart"    # "chart" | "group_log"
    # "fisher_group" (derivation 2026-09-28, Track 2) replaces the pullback direction by the belief-
    # Fisher natural gradient: the Fisher-Rao metric of the reference belief U^{-1} q pulled back to
    # the frame, at the layer-0 table rows, count-normalized (N_scored / n_v), and completed on its
    # kernel (the stabilizer rotations) by this weight: M = g + lam ||asym Z||^2. lam = 2 gives the
    # rotations the Fisher's own shear weight and makes the metric the Kronecker product
    # C^{-1} X (2C + xi xi^T). It composes with either retraction; "group_log" is recommended, since
    # the chart route's 1e-6 product certificate caps steps near 7e-3 per block at trained norms.
    m_phi_fisher_rotation_weight: float  = 2.0

    #################################
    #        Weight Decays
    #################################

    weight_decay:              float = 0.05

    # SEPARATE AdamW weight decay for the log-variance tables (sigma_log_embed, s_sigma_log,
    # decode_log_scale). None = inherit the global weight_decay -- the long-standing behavior, which
    # decays log sigma toward 0, i.e. an unintended lognormal prior pinning sigma to 1 against the
    # configured sigma_init. Set 0.0 to exempt the sigma sector.
    sigma_weight_decay:        Optional[float] = None

    # SEPARATE AdamW weight decay for the MEAN-role capacity tables (mu_embed, s_mu_embed,
    # decode_mu_embed, output_proj_weight) -- the exact mirror of sigma_weight_decay above. None =
    # inherit the global weight_decay. A dead TENSOR escapes AdamW decay because
    # zero_grad(set_to_none=True) leaves p.grad = None, but a rarely-visited ROW inside a live tensor
    # does not: the embedding gather produces a dense gradient, so decay is applied to all V rows on
    # every step whether or not the token appeared. Measured on the K=300 wikitext-103 run, that
    # drives the zero-count rows of the mean tables to norm 0.000 and the count-1..15 rows BELOW
    # random initialization. Set 0.0 to exempt the mean sector; the frequent rows keep their dense
    # likelihood gradient either way.
    mu_weight_decay:           Optional[float] = None

    # SEPARATE AdamW weight decay for the gauge-frame coordinate tables (phi_embed, learned
    # pos_phi_free), default 0.065. Decoupled decay on phi sets an LR-invariant frame-norm ceiling
    # (|phi*| ~ E[normalized-grad]/wd) that pulls the transport exp(phi.G) toward the identity; this
    # knob decouples that from the belief-table weight_decay so it can be swept (set 0 for full
    # gauge-frame protection). The pullback-group policy supplies weight_decay=0 through its registry
    # metadata, so this field remains specific to the AdamW phi route.
    phi_weight_decay:          float = 0.065

    #################################
    #        Numerics
    #################################

    eps:                       float = 1e-6
    kl_max:                    float = 100.0
    sigma_max:       Optional[float] = 100.0

    # E-step MEAN trust region. When set to a float, every per-iteration mean step
    # delta_mu = e_q_mu_lr*nat_mu is projected onto the Mahalanobis ball of this many standard
    # deviations, delta_mu^T Sigma^{-1} delta_mu <= trust^2. None reproduces the bare
    # mu = mu - e_q_mu_lr*nat_mu bit-for-bit.
    e_mu_q_trust:    Optional[float] = 2
    e_sigma_q_trust:           float = 5.0       # diag L2 / full Frobenius: one shared whole-tangent ball

    # L2 clipping threshold applied independently to each optimizer role after accumulation.
    # None or zero disables clipping.
    grad_clip:       Optional[float] = 1.0

    # Opt-in mixed precision for CUDA throughput (RTX 5090). None (default) = OFF = the pure fp32
    # path: NO autocast context is entered anywhere in the forward, so the loss/logits are
    # byte-identical to the no-AMP build. 'bf16' / 'fp16' wrap the E-step / belief pipeline in
    # torch.autocast. The cancellation-sensitive decode matmul (_decode_diagonal) AND the
    # cross-entropy stay fp32 even when AMP is on (their inputs are .float()-ed under an
    # autocast(enabled=False) island), as do the existing matrix_exp / SPD-retraction islands. TF32 is
    # intentionally NOT used: its 10-bit mantissa worsens the decode's catastrophic cancellation and
    # breaks the atol-1e-3 decode pin. bf16 needs no GradScaler and is the recommended default for the
    # 5090; fp16 TRAINING would need a GradScaler in the M-step (deferred follow-up).
    amp_dtype:                 Optional[str] = None       # None | "bf16" | "fp16"

    # fp32 matmul / TF32 policy. `deterministic` above does NOT constrain TF32, so without this the
    # run proceeds under whatever policy the card defaulted to while the tests pin their own -- every
    # numeric tolerance would then be asserted under a policy the training run neither stated nor
    # logged. RECORDING is unconditional and already live; SETTING is this knob, and None means
    # "leave the card as it is" (the historical behavior), so no run changes unless you ask.
    # "highest" disables TF32 (full fp32 mantissa, slowest, most reproducible); "high"/"medium"
    # permit TF32/bf16x3 reductions.
    fp32_matmul_precision: Optional[str] = "highest"       # None (leave as-is) | "highest" | "high" | "medium"

    # Float64 escalation of the stored frame pair (U, U^-1) (audit 2026-09-23 #12). On, a producer
    # call whose float32 pair has max_b kappa_F(U_b) eps_32 > frame_pair_condition_tol is redone in
    # float64 as a whole, and every relational consumer of that frame (projected scorer, predictive
    # content, moment-matched covariance) then runs in float64, forward and backward. OPT-IN: it was
    # unconditional from 2026-10-02 to 2026-10-04 and measured ~7x per escalated step on the RTX 5090
    # (1.53 s vs 0.21 s at K=64, B=32; seed 54, where the learned position-0 frame drifted to
    # kappa_F ~ 445, escalated on 61 of 64 batches by step 8000). Off, the stored pair keeps its
    # inverse residual <= eps_32 kappa_F(U) (about 2e-4 there) and the loss moved by <= 9e-5 nats;
    # CanonicalFrameContext still fails closed above min(64 K eps kappa_inf, 1e-2).
    frame_pair_fp64_escalation: bool  = False
    frame_pair_condition_tol:   float = 1e-4        # escalate when eps_32 max kappa_F > tol, i.e. kappa_F > 838.9

    # Working precision of the FULL-covariance pair KL in the family_chunked decoder (family=
    # "gaussian_full" only). "fp64" is the historical unconditional float64 island.
    # "fp32_escalate" evaluates in float32 and recomputes in float64 only when a Cholesky fails.
    full_cov_kl_precision:     str   = "fp32_escalate"   # "fp32_escalate" | "fp64" | "fp32_escalate_cond"

    # Conditioning-triggered float64 source recovery in the projected attention scorers. A float32
    # projected covariance G whose zero-jitter Cholesky FAILS is always rebuilt from its source
    # (X = S U^-1 Sigma^1/2, QR in float64). With this on, entries that factor but fail the
    # certificate eps_32 ||G||_F ||G^-1||_F <= 1e-4 are rebuilt too. It is OPT-IN because it is not
    # free: torch.linalg.qr on CUDA factors the flagged matrices one at a time. At B=64, N=128, K=20,
    # H=2 on the launcher route it flagged 8,019 of 16,384 query entries by step 250 and cost
    # ~550 ms per scorer call forward+backward against 11 ms without it, taking the RTX 5090 step
    # from 0.11 s to 0.60-0.79 s. Off, a flagged entry keeps its float32 factor (relative error
    # O(kappa eps_32) in its scores) and the certificate's two batched inverses are skipped.
    projected_cov_condition_recovery: bool = False

    # Certificate thresholds of that recovery (read only with projected_cov_condition_recovery=True).
    # Gram route: eps_32 ||G||_F ||G^-1||_F > tol (kappa_F(G) > 838.9 at 1e-4). Factor route
    # ("cholesky_qr2"): eps_32 kappa_1-estimate(C) > tol on the CholeskyQR2 factor (> 2517 at 3e-4).
    projected_gram_condition_tol:   float = 1e-4
    projected_factor_condition_tol: float = 3e-4

    # Working precision of the projected source recovery (failed float32 Cholesky factors, plus the
    # certificate-flagged ones above). "fp64" rebuilds them by float64 QR, promotes the factor tensor
    # to float64 and evaluates every pair touching a recovered entry in float64 triangular-solve tiles;
    # those tiles, not the QR, cost ~230 ms per recovering step at K=64 (seed 54). "fp32" rebuilds
    # them by float32 shifted CholeskyQR3 of the source root and keeps factor and tiles in the
    # working dtype: whitened factor error <= c eps_32 kappa_2(X) (measured <= 9.1e-4 on every
    # seed-54 failure, all at position 0), the same as float32 Householder QR of the same root. A
    # call whose float32 retry fails (kappa_2(X) beyond ~1/eps_32) takes the "fp64" path.
    projected_cov_recovery_precision: str = "fp64"      # "fp64" | "fp32"

    # Factor route of the projected Gaussian-KL scorer (optimization_plan.md WS1). "gram" factors the
    # rounded float32 Gram G = fl(X X^T), X = S U^-1 Sigma^1/2, so its factor, precision and
    # log-determinant carry error O(eps kappa(X)^2). "cholesky_qr2" factors X itself by CholeskyQR2:
    # the same Gaussian and the same KL in exact arithmetic, with error O(eps kappa(X)); the recovery
    # certificate above becomes eps kappa_1(C) > 3e-4 on the factor, and a failed CholeskyQR2 retries
    # shifted CholeskyQR3 in float32 before float64. Measured on the saved K=64/K=128 runs (Phase 0,
    # 2026-09-24): max KL error / tau 2.5e-4 / 4.6e-5 (gram) vs 2.0e-4 / 3.3e-5 (cholesky_qr2) against
    # a float64 oracle; the float32 pair-stage GEMM tr(P_k G_q) dominates what remains. With
    # condition recovery on, "cholesky_qr2" recovered no entry where "gram" recovered 127,112 (K=64),
    # but it raised CUDA kernel time per training step from 216 to 362 ms (K=64, B=64).
    # The noisy-query cost factors per-pair sums G_q + G_k and keeps the Gram route.
    projected_factor_route: str = "gram"     # "gram" | "cholesky_qr2"

    # safe_cholesky's jitter ladder is always RELATIVE (scaled by max(1, |diagonal_mean(M)|)); the
    # absolute ridge selector safe_cholesky_jitter_mode was retired 2026-10-02.

    # Jitter-escalation rounds for the mu-trust-region whitening Cholesky. At 0 (the historical
    # value) one marginally non-PD sigma_q silently drops that belief to a diagonal whitening that is
    # NOT GL-equivariant, masked in by torch.where. A positive value lets the ladder rescue it and
    # keep it on the equivariant path; guard_mu_trust_fallback in the run report says whether it fires.
    mu_trust_cholesky_rounds: int = 3

    # Working precision of the decode's a_v reduction. a_v accumulates terms of size 1/sigma_v and
    # the result is divided by 2*tau (0.016 live), so the fp32 error is amplified twice. Measured max
    # |a_v - fp64| at K=20: 5.2e-6 at sigma_v=4, 2.5e-3 at 1e-2, 2.2e-1 at 1e-4 -- the last being
    # ~12 logits. It is a precision problem, not a factorization problem (differencing before
    # squaring buys only ~1.8x, while the same algebra in float64 buys nine decades). sigma_log_embed
    # trains freely against only the eps=1e-6 floor, so this is harmless at init and about one order
    # of magnitude of shrinkage from matter; watch decode_sigma_v_min.
    decode_av_precision: str = "fp64"     # "fp64" | "fp32"

    # Fail-closed chart radius of every production transport exponential: a raw Lie-algebra matrix
    # with ||M||_F above it raises ValueError instead of being exponentiated. It is the one domain
    # radius of the transport; the former surrogate Frobenius clamp (rescale to ||M||_F = 20) is
    # retired, and 20.0 is that clamp's radius. Applies to every group, so_n included.
    transport_chart_max_norm:  float = 20.0

    # Optional post-M-step interior radius, strictly below transport_chart_max_norm when set. AdamW
    # projects rows back along their algebra ray after the optimizer step. The pullback-group route
    # instead treats this value as an in-optimizer candidate bound and rejects atomically; when None,
    # that route uses the fixed factor radius 5.0.
    #
    # When this is None under m_phi_update_mode='adamw', NOTHING bounds ||phi|| in the M-step: train.py
    # gates the chart projection on this field being set, and the 'adamw' phi policy registers no
    # manifold hook. A row that drifts past transport_chart_max_norm then aborts the next transport
    # build (fail-closed). Archived runs show phi_matrix_norm_median rising 0.86 -> ~7.4 with
    # vertex_cond_max reaching ~4e3, and transport conditioning grows with ||phi||.
    phi_mstep_max_matrix_norm: Optional[float] = None

    # How many validation batches the held-out val_* probes average over. At 1 -- the historical
    # value, and bit-identical -- EVERY val_* column is a statistic of one fixed 64-token sequence:
    # _val_diagnostics takes the first batch of a shuffle=False loader, so the same sequence is
    # scored at every eval of every run. That is a sample size of one standing behind every held-out
    # number the project reports, including estep_fnondecreasing_frac, which at n_e_steps=1 is a
    # single Bernoulli draw. Raising this costs one extra held-out forward per batch per eval and is
    # the cheapest way to make the val_* columns mean what they appear to mean; the attention FIGURES
    # still come from the first sequence either way.
    val_diagnostic_batches:               int = 1  # try 64

    # Accumulate gradients over N microbatches before an optimizer step, for a larger effective batch
    # without the memory of one big forward. Each pulled batch is split into N equal chunks along the
    # batch axis, each backed (loss / N) into .grad, then a single clip + optimizer.step() +
    # scheduler.step() fires at the boundary. Default 1 = current single-step behavior
    # (byte-identical: no chunking, no divide).
    grad_accum_steps:          int   = 1

    # Cosine-decay floor.
    min_lr_frac:               float = 0.01    # fractional floor: no group's LR decays below
    #                                          min_lr_frac * its OWN base LR, preserving the
    #                                          m_mu : m_sigma : m_phi base ratios into the tail.
    #                                          0.0 is the pure half-cosine-to-zero path.

    generate_embedding_figures:     bool = True


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
        if self.gauge_group in _RETIRED_GAUGE_GROUPS:
            raise ValueError(f"gauge_group={self.gauge_group!r} is retired; supported groups are "
                             f"{_SUPPORTED_GAUGE_GROUPS}")
        _require(self.gauge_group, _SUPPORTED_GAUGE_GROUPS, "gauge_group")
        full_link = self.attention_mode == "projected_gradient"
        # attention_residual and predictive_covariance are predictive-readout knobs; under
        # projected_gradient they are inactive (inactive_fields), not errors.
        _require(self.relational_routing, ("gaussian_kl", "noisy_query"), "relational_routing")
        if self.relational_routing == "noisy_query" and full_link:
            raise ValueError("relational_routing='noisy_query' requires attention_mode='projected_predictive'")
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
            # the positional rotation must also be off (else Omega != I and the cell is mislabeled).
            if self.pos_rotation != "none":
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
        if self.embed_dim % self.n_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by n_heads={self.n_heads}"
            )

        # gauge seam. so_n irrep tower: validate group_n + irrep_spec at construction (labels
        # resolve, block dims sum to embed_dim) so a bad spec fails here with the computed dims.
        if self.gauge_group == "so_n":
            if self.group_n is None or self.irrep_spec is None:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} requires both group_n and irrep_spec "
                    f"([(label, mult), ...]); got group_n={self.group_n}, "
                    f"irrep_spec={self.irrep_spec}"
                )
            if type(self.group_n) is not int or self.group_n < 2:
                raise ValueError(f"group_n must be an int >= 2, got {self.group_n!r}")
            if not isinstance(self.irrep_spec, list) or not self.irrep_spec:
                raise ValueError(
                    f"irrep_spec must be a non-empty list of (label, mult) pairs, got "
                    f"{self.irrep_spec!r}"
                )
            # JSON has no tuple type; coerce reloaded list pairs to tuples.
            self.irrep_spec = [tuple(p) if isinstance(p, list) else p for p in self.irrep_spec]
            from vfe3.geometry.irreps import irrep_dim
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
                _d = irrep_dim(self.group_n, algebra="so", label=_entry[0])
                _block_dims.extend([_d] * _entry[1])
            if sum(_block_dims) != self.embed_dim:
                raise ValueError(
                    f"irrep_spec blocks {_block_dims} sum to {sum(_block_dims)} != "
                    f"embed_dim={self.embed_dim} (group_n={self.group_n})"
                )
            # The projected routes read the frame through the operators' head partition: each head
            # is one gauge block, so the irrep blocks must be n_heads equal blocks of size d_head
            # (relational_score slices the dense inverse frame into those head blocks).
            if _block_dims != [self.embed_dim // self.n_heads] * self.n_heads:
                raise ValueError(
                    f"gauge_group='so_n' on the projected routes needs n_heads={self.n_heads} equal "
                    f"irrep blocks of size d_head={self.embed_dim // self.n_heads} (one gauge block "
                    f"per head); irrep_spec gives blocks {_block_dims}"
                )
            # One shared phi drives EVERY block, so the generators do not partition per block:
            # the per-block phi preconditioners are undefined.
            if self.phi_precond_mode in ("killing_per_block", "pullback_per_block"):
                raise ValueError(
                    f"phi_precond_mode={self.phi_precond_mode!r} is incompatible with "
                    f"gauge_group={self.gauge_group!r}: the shared so generators act in every "
                    f"irrep block, so a per-block metric is undefined. Use 'none', 'clip', or "
                    f"the ambient 'killing'."
                )
            # ALiBi-family priors are built with H = n_heads while the energy head axis is the number
            # of irrep blocks; require they agree so the (H,N,N) prior aligns with the energy.
            for _pname in ("beta_attention_prior",):
                if (getattr(self, _pname) in ("alibi", "causal_alibi", "causal_alibi_noself")
                        and self.n_heads != len(_block_dims)):
                    raise ValueError(
                        f"{_pname}={getattr(self, _pname)!r} builds an (n_heads, N, N) prior but "
                        f"gauge_group={self.gauge_group!r} has {len(_block_dims)} irrep blocks; "
                        f"set n_heads={len(_block_dims)} or use a headless prior "
                        f"(uniform/causal/...)."
                    )
        elif self.group_n is not None or self.irrep_spec is not None:
            raise ValueError(
                f"group_n/irrep_spec are consumed only by gauge_group 'so_n'; got "
                f"gauge_group={self.gauge_group!r}"
            )
        if self.use_cg_coupling and self.gauge_group != "so_n":
            raise ValueError(
                f"use_cg_coupling requires the irrep-labeled tower group 'so_n'; got "
                f"gauge_group={self.gauge_group!r}"
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
                if self.gauge_group == "so_n":
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
        if (self.lambda_alpha_mode == "constant"
                and not (math.isfinite(self.lambda_alpha) and self.lambda_alpha >= 0.0)):
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
        from vfe3.geometry.phi_preconditioner import _PRECOND
        _require(self.phi_precond_mode, tuple(sorted(_PRECOND)), "phi_precond_mode")
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
            if capability != self.phi_precond_mode:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} has registered pullback capability "
                    f"{capability!r}, which does not certify phi_precond_mode="
                    f"{self.phi_precond_mode!r}; phase one requires a built-in bracket-closed "
                    "full/per-block pullback pairing"
                )
            if self.gauge_group != "block_glk":
                raise ValueError(
                    "pullback_group phase one accepts only untied block_glk; "
                    f"got gauge_group={self.gauge_group!r}"
                )
            if self.d_head > 12:
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
        if self.pos_rotation == "rope" and self.gauge_group == "so_n":
            import warnings
            warnings.warn(
                f"pos_rotation='rope' with gauge_group={self.gauge_group!r} leaves the structure "
                f"group: the coordinate-pair rotation R(theta) is generally not in the irrep image "
                f"of SO(N), so R Omega R^T is no longer in the selected group. The "
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
        # FAIL CLOSED rather than falling back. The right insertion folds R into the VERTEX frame, so
        # it needs a transport that carries one; a dense Omega exposes only the composed operator.
        # Silently reverting to 'left' there would put the defective composition (69% absolute-position
        # contamination, broken gauge covariance) on a config that asked for the pure path.
        if self.pos_rotation == "rope":
            from vfe3.geometry.groups import get_group
            _escapes = "Use a group carrying a per-vertex gauge chart."
            # The flat FACTORED builder -- the thing that exposes the vertex frame -- is selected only
            # for a block-diagonal group with MORE THAN ONE block, so the group alone settles it.
            _builder = get_group(self.gauge_group)
            if self.gauge_group == "so_n":       # irrep blocks validated in the gauge seam above
                _dims = _block_dims
            else:
                try:                             # builders differ: only the block groups take n_heads
                    _dims = _builder(K=self.embed_dim, n_heads=self.n_heads).irrep_dims
                except TypeError:
                    _dims = _builder(K=self.embed_dim).irrep_dims
            if len(_dims) < 2:
                raise ValueError(
                    f"rope_insertion='right' needs a group whose "
                    f"transport factors per block, but gauge_group={self.gauge_group!r} has a "
                    f"single irrep block ({_dims}), so the flat transport is built as a dense "
                    f"pairwise Omega with no per-vertex frame to fold into. {_escapes}")
        # decode / encode
        if not math.isfinite(self.decode_tau) or self.decode_tau <= 0.0:
            raise ValueError(f"decode_tau must be finite and positive, got {self.decode_tau}")
        _require(self.relational_output_init, ("identity", "depth_scaled"), "relational_output_init")
        from vfe3.model.relational_operators import CONTENT_LINK_CHARTS
        _require(self.content_link_chart, CONTENT_LINK_CHARTS, "content_link_chart")
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
        # gauge_transport coercion, which supplies pos_phi='none' under 'off'.
        if (self.encode_mode == "per_token_additive"
                and (self.pos_phi != "none" or self.pos_rotation != "none")):
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

        # Reference coordinates (M2 arms: LN and MLP in reference coordinates; the covariant decoder).
        # Both conjugate by the realized relational frame U_i, which is defined, fixed through the
        # stack, and equal to the content frame on the projected routes without RoPE. Everything else
        # fails closed.
        _require(self.block_coordinates, ("local", "reference"), "block_coordinates")
        _require(self.decode_coordinates, ("local", "reference"), "decode_coordinates")
        frame_requirements = []
        if self.pos_rotation != "none":
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
            if self.use_block_mlp and self.block_mlp_mode not in ("coordinate", "gaussian_template"):
                incompatible.append(
                    f"block_mlp_mode={self.block_mlp_mode!r} (requires 'coordinate' or 'gaussian_template'; "
                    "canonical_frame is already conjugated and gauge_gate is already covariant)")
            if self.use_block_mlp and self.block_mlp_covariance != "passthrough":
                incompatible.append(f"block_mlp_covariance={self.block_mlp_covariance!r} (requires 'passthrough')")
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

        from vfe3.gauge_optim import get_phi_update_policy
        phi_update_policy = get_phi_update_policy(self.m_phi_update_mode)
        if not phi_update_policy.requires_manifold_optimizer:
            mark(("m_phi_group_trust_radius", "m_phi_group_retraction"),
                 "only read by the manifold phi optimizer "
                 "(m_phi_update_mode='pullback_group' or 'fisher_group')")
        if not phi_update_policy.requires_fisher_statistics:
            mark(("m_phi_fisher_rotation_weight",), "only read by m_phi_update_mode='fisher_group'")
        if self.relational_routing == "noisy_query":
            mark(("projected_factor_route",),
                 "the noisy-query cost factors per-pair sums G_q + G_k and keeps the Gram route")
        if not self.deterministic:
            mark(("fill_uninitialized_memory",),
                 "torch fills uninitialized memory only under deterministic algorithms")
        gram_status = (self.projected_factor_route == "gram"
                       or self.relational_routing == "noisy_query")
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
            mark(("predictive_covariance_floor", "predictive_covariance_checkpoint"),
                 "only read by predictive_covariance='moment_matched'")
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
            mark(("rope_on_value", "rope_base"), "only read by pos_rotation='rope'")
        if self.pos_phi == "none":
            mark(("pos_phi_project_slk",), "only read by a non-'none' pos_phi")
        if (self.beta_attention_prior in ("windowed", "causal_windowed")
                and self.attention_window >= self.max_seq_len - 1):
            mark(("attention_window",), "window covers the context and is identical to plain causal")
        if self.n_layers == 1:
            mark(("prior_handoff_rho", "prior_handoff_sigma"), "no next layer consumes the handoff")
        if self.m_phi_update_mode != "pullback_group":
            mark(("phi_precond_mode",), "reached only by m_phi_update_mode='pullback_group'")
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
        if self.attention_mode == "projected_predictive":
            mark(("lambda_beta",), "predictive_content weights no belief-coupling block")
        if self.decode_mode != "family_chunked":
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
        if (self.n_e_steps == 1 and self.n_layers == 1 and self.mstep_self_coupling_weight == 0.0):
            mark(("lambda_alpha_mode", "lambda_alpha", "b0", "c0"),
                 "the single gradient E-step iteration starts at q = p, where the self-coupling "
                 "force vanishes")
        if (self.attention_mode == "projected_predictive" and self.family == "gaussian_full"
                and self.encode_mode in ("per_token", "per_token_additive")
                and self.predictive_covariance == "passthrough"
                and self.block_coordinates == "local" and self.decode_coordinates == "local"
                and not (self.use_block_mlp and self.block_mlp_covariance != "passthrough")):
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
    def phi_retract_mode(self) -> Literal["bch"]:
        """Fixed E-step phi composition, excluded from serialized configuration."""
        return "bch"

    @property
    def rope_insertion(self) -> Literal["right"]:
        """Fixed positional composition in the vertex frame."""
        return "right"

    @property
    def d_head(self) -> int:
        """Per-head belief dimension K // n_heads."""
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
    if "relational_mode" in payload or "relational_readout_mode" in payload:
        mode = payload.pop("relational_mode", "off")
        readout = payload.pop("relational_readout_mode", "conditional_update")
        legacy_modes = {
            ("off", "conditional_update"): "transported",
            ("projected_filtering", "conditional_update"): "projected_conditional",
            ("projected_filtering", "predictive_content"): "projected_predictive",
        }
        try:
            attention_mode = legacy_modes[(mode, readout)]
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
