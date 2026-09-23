"""Configuration for VFE_3.0. Single dataclass, single validation block.

No CLI parsing (project policy: click-to-run). Edit fields directly, then run.
Every registry-backed seam is selected here by name (divergence, gauge group,
encode/decode mode, alpha form, attention prior, norm, gradient mode), so a
variant swaps without editing call sites.
"""

import math
import os
import warnings
from dataclasses import MISSING, InitVar, asdict, dataclass, fields
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

# Seams with a live registry (gauge_group, lambda_alpha_mode, attention priors, norms; alongside
# transport/retraction/positional/divergence) are validated against that registry in __post_init__
# via tuple(sorted(_REGISTRY)), so a newly registered variant is a valid config value WITHOUT
# editing this validator (the add-by-registering modularity contract). phi_precond_mode is
# validated against phi_preconditioner._PRECOND. E-step phi always uses BCH; positional phi always uses exact group-product composition.
# Encode/decode modes likewise use their registries. The linear registration is public; dense
# decode registrations are retained only for typed legacy replay and private oracles. The static
# tuples below are the remaining seams WITHOUT a registry.
_VALID_GRADIENT_MODES      = ("filtering", "smoothing")
_VALID_E_STEP_GRADIENTS    = ("unroll", "straight_through", "detach")
_VALID_GAUGE_TRANSPORT     = ("on", "off", "frozen")
_DENSE_DECODE_MODES        = frozenset({"diagonal", "full", "family"})


class _ConfigTypeError(TypeError, ValueError):
    """Configuration type error compatible with legacy ``ValueError`` callers."""


@dataclass
class VFE3Config:
    """The single authoritative configuration surface for VFE_3.0.

    Fields are grouped: numerics, divergence seam, model structure, gauge seam,
    belief family, free-energy coupling, attention, E-step, decode/encode, cross-
    block handoff, normalization, and M-step / training. Each ``*_mode`` /
    ``*_family`` / ``*_group`` field is a registry key. Validation runs once in
    ``__post_init__`` in field-declaration order.
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
    #   f-divergence and e/m family
    #################################

    # Divergence FUNCTIONAL (registry key). renyi_order selects the member and is IGNORED by
    # non-alpha functionals (e.g. squared_hellinger).
    divergence_family:         str   = "renyi"        # "renyi" | "bhattacharyya" | "jeffreys" | "squared_hellinger"
    renyi_order:               float = 1.0            # Renyi order (alpha = 1 -> KL)

    # Belief covariance STRUCTURE (registry key), the single source of truth for diagonal-vs-full.
    # Read the derived read-only ``diagonal_covariance`` property, never a second flag.
    family:                    str   = "gaussian_diagonal"   # "gaussian_diagonal" | "gaussian_full"
                                                             # | "gaussian_diagonal_exact" | "gaussian_frame_diagonal"
                                                             # | "laplace_diagonal"

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
                                                     # | "canonical_content_gauge" | "canonical_content_projected"

    # Decoder (registry key). "linear" is the learned (V,K) readout logits = mu_q @ W^T (+ bias),
    # discarding sigma. Every other mode reads out against the prior bank; 'diagonal'/'full'/'family'
    # are DENSE legacy/oracle kernels rejected for fresh training (see _allow_legacy_dense_decode) --
    # use their *_chunked counterparts, which stream the vocabulary and never materialize (B,N,V).
    decode_mode:               str   = "linear"      # "linear" | "diagonal_chunked" | "full_chunked"
                                                     # | "family_chunked" | "expected_likelihood_chunked"
                                                     # | (dense legacy, opt-in) "diagonal" | "full" | "family"

    decode_tau:                float = 1.0           # KL-to-prior decode temperature; inert under decode_mode="linear"
    decode_temperature_learned: bool = True          # False freezes decode_log_scale, so tau_eff == decode_tau exactly

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

    # Gradient checkpointing for the decode_ce_*_chunked fused-CE kernels. "always" reproduces the
    # pre-2026-08-07 unconditional behavior (recompute the chunk in backward whenever grad is live).
    # "never" keeps the chunk workspace live for backward and pays no recompute. "auto" checkpoints
    # only when the (B,N,chunk_width) tensor the chunk would otherwise save, at its own dtype
    # itemsize, exceeds a fixed 2 GiB budget -- a constant rather than a live device-memory query, so
    # the decision is reproducible run-to-run and degrades to the same value on CPU. Inert when the
    # active decode route has no fused_ce registration ('full'/'family').
    decode_ce_checkpoint:      str   = "auto"        # "auto" | "always" | "never"

    # Strict decoder ranking guard. When enabled, analytic fp32 decoders construct conservative score
    # intervals and, if ANY row's winning interval overlaps a competitor, recompute the complete
    # (B,N,V) decode in differentiable float64. That is an accuracy diagnostic, not a production-safe
    # fallback: at B=64, N=128, V=50257 the promoted logits alone occupy ~3.1 GiB before backward
    # intermediates. Keep it opt-in so the default chunked route remains streamed and bounded.
    decode_ranking_fp64_escalation:             bool = True
    _allow_legacy_dense_decode:        InitVar[bool] = False

    #################################
    #          Gauge Group
    #################################

    gauge_group:               str   = "block_glk"   # "block_glk" | "tied_block_glk" | "glk" | "so_k" | "sp"
                                                     # | "so_n" | "sp_n"

    # Ablation meta-toggle over the GL(K) gauge FRAME. DISTINCT from transport_mode below: that
    # decides the connection's flatness, this decides whether the per-token frame exists at all.
    #   "on"     (default, PURE PATH) the frame is whatever phi_scale / e_phi_lr / m_phi_lr / pos_phi
    #            specify (learned when m_phi_lr > 0); byte-identical to omitting the toggle.
    #   "off"    Omega_ij = exp(phi_i) exp(-phi_j) = I EXACTLY, by coercing phi_scale=0.0,
    #            pos_phi='none', e_phi_lr=0.0, m_phi_lr=0.0 -- and REQUIRING pos_rotation='none',
    #            since a positional rotation would make Omega != I (raises).
    #   "frozen" a RANDOM but FIXED frame: keeps phi_scale>0, coerces e_phi_lr=m_phi_lr=0.0.
    # __post_init__ applies the coercion and warns. NB 'off' drops the pos_phi_free table, so to match
    # parameter counts against an 'on' baseline set pos_phi='none' in 'on' too, or compare against the
    # equal-parameter 'frozen' arm. All three arms are reachable under
    # relational_mode='projected_filtering', whose conditional path BUILDS its transport from the
    # frame rather than bypassing it.
    gauge_transport:           str   = "on"          # "on" | "off" | "frozen"

    # Connection REGIME (registry key). Only the flat phi-cocycle is registered: Omega_ij =
    # exp(phi_i) exp(-phi_j), a pure no-NN coboundary with no learned edge factor and no stateful
    # transport parameter.
    transport_mode:            str   = "flat"        # "flat"

    # Cross-head GL(K) coupling: a list of directed (head_a, head_b) index pairs that add off-block
    # generators (and a genuinely larger-than-direct-sum subalgebra under the builder's bracket
    # closure) to the gauge basis. None = the current block-diagonal GL(d_head)^H gauge. Only a
    # group whose builder accepts the kwarg (block_glk) supports it.
    cross_couplings: Optional[List[Tuple[int, int]]] = None

    # SO(N)/Sp(2m) irrep-tower groups ('so_n' / 'sp_n') only. The structure group is SO(group_n)
    # (resp. Sp(group_n), group_n = 2m even) with group_n DECOUPLED from embed_dim. The embedding
    # carries the direct sum of the irreps in irrep_spec -- (label, multiplicity) pairs laid out
    # contiguously in order; heads = irrep blocks (possibly UNEQUAL dims). so_n labels 'l<p>' are the
    # symmetric-traceless rank-p tensor irreps (group_n=3: the spin-p tower, dim 2p+1); sp_n labels
    # 'sym<p>' are Sym^p of the defining rep (dim C(2m+p-1, p)). sum(mult * dim) must equal embed_dim.
    # One shared per-token phi (n_gen = dim of the algebra) drives every block -- a TIED gauge. Both
    # fields are consumed ONLY by these two groups (rejected otherwise).
    group_n:                      Optional[int] = None

    # The annotation reflects the POST-coercion type: __post_init__ also accepts the JSON/TOML
    # round-trip list-of-lists form ([["l1", 3], ...]) and coerces each pair to a tuple.
    irrep_spec: Optional[List[Tuple[str, int]]] = None

    # CG cross-type coupling (opt-in, default off; so_n/sp_n only): bilinear Clebsch-Gordan
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

    #################################
    #          Head Mixers
    #################################

    # Head mixer (opt-in, default off): a learned Schur-commutant matrix mixing the equal-size
    # gauge-irrep blocks (under block_glk: the n_heads heads) of the converged belief. Identity init
    # -> step-0 bit-identical to off. Breaks strict gauge equivariance under block_glk's untied
    # per-block gauge (exact at init, deviates as the mixer drifts); the tied_block_glk group
    # restores EXACT equivariance on the full-covariance path. Needs a group with >= 2 blocks
    # (block_glk / tied_block_glk heads, or an so_n/sp_n tower); a single-block group is rejected at
    # config validation, and a head-block group with n_heads < 2 auto-disables the mixer with a warning.
    use_head_mixer:                    bool = False

    # PriorBank-native per-gauge-block evidence calibration: one zero-initialized scalar per block,
    # restricted to the built-in canonical KL decode, where its evidence interpretation is fixed.
    # A separate opt-in from the Schur-commutant post-E-step head mixer above.
    use_priorbank_head_evidence_mixer: bool = False

    # Optional absolute M-step rates for the two semantically distinct mixer parameter families.
    # None preserves the historical behavior by inheriting m_p_mu_lr at optimizer construction.
    m_head_evidence_lr:        Optional[float] = None
    m_head_mixer_lr:           Optional[float] = None

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

    # Gauge-RoPE (default-off): a block-diagonal positional rotation R(theta) folded into the
    # transport (Omega^RoPE_ij = R(theta_i) Omega_ij R(theta_j)^T). Means-only by default;
    # rope_full_gauge=True also rotates the covariance sandwich and REQUIRES full covariance.
    pos_rotation:              str   = "none"      # "none" | "rope"
    rope_base:                 float = 100.0       # rotary frequency base

    rope_full_gauge:           bool  = False       # rotate covariance too (needs family="gaussian_full")

    # True (default) = the coherent single-gauge path: the gauge-RoPE rotation feeds BOTH the
    # attention score and the value aggregation. False factors the transport into an attention gauge
    # and a value gauge (GL(K)_attention.tex:1909) -- RoPE's "position-dependent attention,
    # position-independent values" asymmetry. Decoupled breaks beta's coupling-sum stationarity, so
    # the belief gradient routes to the autograd oracle (no closed-form kernel). Inert unless
    # pos_rotation='rope'.
    rope_on_value:             bool  = True         # False -> value aggregation uses the un-rotated base

    #################################
    #          Methods
    #################################

    # One attention selector chooses transported inference, conditional projected updates, or
    # predictive projected content. Projection ranks still follow the gauge block dimensions.
    # Projected attention requires the explicit support envelope validated below.
    attention_mode:                      str           = "transported"          # "transported" | "projected_conditional" | "projected_predictive"
    relational_input_scope:              str           = "head"                 # "head" | "full"
    relational_output_scope:             str           = "head"                 # "head" | "full"
    attention_residual:                  bool          = False                  # requires predictive_content
    relational_output_init:              str           = "identity"             # "identity" | "depth_scaled" (initial O times 1/sqrt(2 n_layers))

    relational_identity_content:         bool          = False                  # predictive_content requires learned content maps
    relational_routing:                  str           = "gaussian_kl"          # "gaussian_kl" | "noisy_query" (projected_predictive only)

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
    # inside the softmax), so beta = softmax(-E/tau) is unchanged and the analytic kernel stays
    # envelope-consistent with the autograd oracle.
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

    # Cross-block belief handoff (mu_q -> mu_p). Both blends must be convex so the prior stays on the
    # SPD cone: sigma_p_next = (1-rho_s) sigma_p + rho_s sigma_q stays > 0 iff rho_s in [0,1].
    prior_handoff_rho:          float = 1.0           # 1.0 = full flow; 0.0 = priors frozen
    prior_handoff_sigma:        float = 0.0           # sigma damping (0.0 = frozen at embedding)

    # Optional residual feed-forward unit applied after each belief block.
    use_block_mlp:              bool  = True
    block_mlp_mode:             str   = "coordinate"  # "coordinate" | "gauge_gate" | "canonical_frame"
    block_mlp_covariance:       str   = "passthrough" # "passthrough" | "delta_full"
    block_mlp_activation:       str   = "gelu"        # "gelu" | "silu" | "relu"

    block_mlp_covariance_floor: float = 1e-4         # covariant SPD floor: Sigma+ = J Sigma J^T + floor Sigma
    block_mlp_expansion:        int   = 8
    block_mlp_dropout:          float = 0.0

    m_block_mlp_lr:   Optional[float] = None

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

    e_q_mu_lr:                 float = 0.5
    e_q_sigma_lr:              float = 0.015
    e_phi_lr:                  float = 0.0

    # E-step MEAN preconditioner. 'fisher' (default, pure) descends the Fisher natural gradient
    # nat_mu = Sigma*grad_mu; 'raw' descends the raw Euclidean grad_mu. The sigma retraction is
    # unchanged either way, so this isolates the MEAN preconditioner.
    e_step_mu_precond:         str   = "fisher"       # "fisher" | "raw"

    # E-step update rule (registry key). 'gradient' (default, pure): one natural-gradient step per
    # inner iteration. 'mm_exact': the closed-form coordinate minimizer of the beta-frozen,
    # strict-pair-masked diagonal-KL surrogate F_hat = sum_i a_i KL(q_i||p_i)
    # + lambda_beta sum_ij beta0_ij KL(q_i||Omega_ij q_j) -- precisions add, means fuse
    # precision-weighted (harmonic-barycenter sigma*) at the same O(N^2 K) cost. The strict pair mask
    # excludes the structural E_ii = 0 self-pairs, so the surrogate is NOT a majorizer of the
    # canonical frozen-attention objective and the update carries NO majorization /
    # monotone-F-descent / exact self-consistent-argmin guarantee -- it is a one-step fusion toward
    # the mask-selected target. KERNEL-ROUTE ONLY (filtering + gaussian_diagonal + renyi order 1 +
    # attention entropy); rejected otherwise in __post_init__.
    e_step_update:             str   = "gradient"     # "gradient" | "mm_exact"

    # Damped mm_exact step theta <- (1-eta) theta + eta theta* (natural-coordinate mirror descent);
    # 1.0 = the full exact coordinate minimizer. Read only under e_step_update='mm_exact'.
    mm_damping:                float = 1.0

    # E-step backward estimator (manuscript Algorithm 1).
    #   'unroll' (default)  differentiate through the inner trajectory. Keeps the OWN-ROW
    #                       second-order term d^2F/dq_i^2 but NOT the cross-row d^2F/dq_i dq_j:
    #                       under the default gradient_mode='filtering' the key slot is a detached
    #                       copy, so it is absent from the outer backward too. The detach is
    #                       intended and documented (the mean-field coordinate-ascent default, with
    #                       'smoothing' the registered alternative).
    #   'straight_through'  each inner update computes its tangent DETACHED but rebuilds the belief
    #                       grad-connected to the previous belief (mu_next = mu_prev +
    #                       delta.detach()), so d belief_next/d belief_prev = I flows WITHOUT the
    #                       second-order term. Forward VALUE is byte-identical to 'unroll'; only the
    #                       BACKWARD differs.
    #   'detach'            the whole E-step under torch.no_grad -- no E-step gradient at all.
    e_step_gradient:              str   = "unroll"    # "unroll" | "straight_through" | "detach"
    phi_precond_mode:             str   = "none"      # "none" | "clip" | "killing" | "killing_per_block"
                                                      # | "pullback" | "pullback_per_block"

    # Opt-in (default ON here): make the autograd belief-gradient ORACLE -- served for the non-kernel
    # families: gradient_mode='smoothing', family='gaussian_full', renyi_order != 1 -- return a
    # create_graph (differentiable) gradient under e_step_gradient='unroll', so the
    # unrolled-through-inference signal reaches the prior tables, matching the closed-form kernel.
    # NUMERICS: this builds a SECOND-ORDER graph through the E-step. The SINGLE training backward is
    # FINITE for all the non-kernel families, full covariance included: the SPD retractions route
    # their eigendecomposition through the gap-regularized damped eigh (whose Lorentzian-damped
    # adjoint is finite on the degenerate isotropic Sigma = I default, where stock eigh backward is
    # 100% NaN), and the full-covariance KL/entropy use safe_cholesky. The residual caveat is HIGHER
    # order only: a genuine SECOND backward (double-grad / Hessian-vector product) can still NaN at
    # large Renyi order (renyi_order >= 2), so do not rely on this toggle for > 1st-order autograd.
    oracle_unroll_grad:        bool  = True

    # Gradient taken through the belief roles.
    #   filtering   Only each token's query/receiver role: key/source beliefs are detached, so each
    #               position updates from earlier beliefs without modifying those earlier beliefs
    #               through its own row (the source beliefs act as a frozen snapshot).
    #   smoothing   BOTH query and key roles on shared belief tensors: earlier beliefs also receive
    #               gradients from later positions that use them as keys, i.e. the complete coupled
    #               scalar is differentiated. Use with oracle_unroll_grad=True.
    gradient_mode:                str   = "filtering"   # "filtering" | "smoothing"
    include_attention_entropy:    bool  = True          # canonical (True) vs surrogate (False)

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
    m_phi_update_mode:         str       = "adamw"    # "adamw" | "pullback_group"
    m_phi_group_trust_radius:  float     = 0.1

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

    # E-step MEAN trust region (default OFF = current unbounded update). When set to a float, every
    # per-iteration mean step delta_mu = e_q_mu_lr*nat_mu is clamped in sigma-whitened units to at
    # most this many standard deviations (mu_trust_mode='box' per-coord, 'ball' = 2-norm Mahalanobis
    # ball). None reproduces the bare mu = mu - e_q_mu_lr*nat_mu bit-for-bit.
    e_mu_q_trust:    Optional[float] = 2
    e_sigma_q_trust:           float = 5.0       # diag L2 / full Frobenius: one shared whole-tangent ball

    mu_trust_mode:             str   = "ball"          # "ball" | "box" (consulted only when e_mu_q_trust is not None)

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

    # Matrix exponential precision is norm-based. The config threshold is passed explicitly;
    # low-level geometry helpers retain their separate default threshold of 5.
    exp_fp64_norm_threshold:   float = 20.0            # upcast when max ||M||_F >= this

    # Second trigger for the diagonal-congruence float64 escalation. The DEFAULT guard is sign-only:
    # recompute in float64 iff the working-dtype congruence went negative -- one amin reduction the
    # floor needs regardless. Positivity is necessary but not sufficient (cancellation destroys
    # accuracy before it destroys sign), so this toggle adds the conditioning proxy
    # cond(exp_phi)^4 * eps > 1e-3, which bounds the quadratic form's backward error (Higham 2002,
    # 3.1). It is OPT-IN because it is not free: it runs svdvals on the vertex blocks and then a
    # bool() on the result -- an SVD launch plus a GPU->CPU sync -- on the HEALTHY path of every
    # covariance transport (measured 85 -> 126 ms/step at K=20, with no value change in the
    # well-conditioned case). Turn it on where the accuracy actually bites: large K, large ||phi||,
    # ill-conditioned frames.
    congruence_cond_escalation: bool = False         # True -> also escalate on the conditioning proxy

    # Working precision of the FULL-covariance pair KL (family="gaussian_full" only; the diagonal
    # family already keys its island on operand dtypes). "fp64" is the historical unconditional
    # float64 island, so leaving it alone is byte-identical to the pre-2026-08-05 build.
    # "fp32_escalate" evaluates the (B,N,N,K,K) grid in float32 and recomputes in float64 only when a
    # Cholesky actually fails
    full_cov_kl_precision:     str   = "fp32_escalate"   # "fp32_escalate" | "fp64" | "fp32_escalate_cond"

    # Working precision of the full-covariance CONGRUENCE Omega Sigma Omega^T. Distinct from
    # full_cov_kl_precision above: that governs the pair DIVERGENCE, this governs the transport
    # sandwich that feeds it. At the live shape it is the largest stage in the step -- 1.573 GFLOP of
    # float64 over ~1.57 GB, 0.83 FLOP/byte, on a card whose float64 rate is ~1/64 of float32.
    # "fp32_escalate" contracts in float32 and redoes a batch in float64 only if the float32 result
    # lost finiteness; the congruence SQUARES cond(Omega), so it is a real accuracy trade at high
    # ||phi||. A float64 SOURCE contracts in float64 under either setting.
    full_cov_congruence_precision: str = "fp32_escalate"      # "fp32_escalate" | "fp64"

    # Strict post-congruence certificate. When enabled, factor every transported pair covariance and
    # promote the complete pair field to float64 if any matrix is uncertified. At B=64, N=128, K=20
    # this checks 1,048,576 separate 20x20 matrices and the promoted field is about 3.1 GiB. Keep
    # this opt-in for accuracy investigations; False retains the selected working precision without
    # the additional certificate or fallback recompute.
    full_cov_congruence_certification: bool = False

    # Scaling of safe_cholesky's jitter ladder. The ridge eps*10^t is ABSOLUTE by default, so at an
    # eigenvalue on the eps=1e-6 SPD floor the t=0 ridge DOUBLES it and shifts logdet by exactly
    # log 2 per floored direction, while at sigma_max=100 the same ridge is a 1e-8 relative no-op.
    # "relative" scales by diagonal_mean(M) so every element gets the same relative perturbation.
    # The two coincide exactly when diagonal_mean(M) == 1, and the ladder only fires above cond ~1e8.
    safe_cholesky_jitter_mode: str = "relative"      # "relative" | "absolute"

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

    # Optional fail-closed chart-validity domain for every production transport exponential.
    # None preserves the existing permissive clamp behavior; a positive finite bound rejects the
    # raw Lie-algebra matrix before the numerical clamp can silently change the modeled operator.
    transport_chart_max_norm:  Optional[float] = None

    # Optional post-M-step chart guard. AdamW projects rows back along their algebra ray after the
    # optimizer step. The pullback-group route instead treats this value as an in-optimizer candidate
    # bound and rejects atomically; when None, that route uses the fixed factor radius 5.0.
    #
    # When this is None under m_phi_update_mode='adamw' with e_phi_lr=0.0, NOTHING bounds ||phi||:
    # train.py gates the chart projection on this field being set, the 'adamw' phi policy registers
    # no manifold hook, and retract_phi's own cap (max_norm=5.0) is reachable only from the E-step
    # phi substep. TRANSPORT_CLAMP_MAX_NORM bounds the EXPONENTIATED operator, not the parameter.
    # Archived runs show phi_matrix_norm_median rising 0.86 -> ~7.4 with vertex_cond_max reaching
    # ~4e3, and transport conditioning grows with ||phi||; 5.0 (the GL retraction's own radius) is a
    # reasonable bound. Left at None because it is the shipped default and every value the
    # pullback_group validator admits (5.0 < bound < 20.0) could hard-abort a run whose observed
    # ||M||_F already reaches 18. Deliberately NOT a construction warning: the condition is true for
    # a bare VFE3Config(), and this project keeps a clean config warning-free. Runtime detection
    # belongs in train.py's _warn_phi_transport_clamp.
    phi_mstep_max_matrix_norm: Optional[float] = None

    # Opt-in transport diagnostic: forward clamp_monitor=True into every production
    # stable_matrix_exp_pair transport build, emitting a RuntimeWarning EVERY time the hard
    # Frobenius clamp (TRANSPORT_CLAMP_MAX_NORM = 20.0) fires -- the returned factor is then a
    # surrogate exp(M*scale), not exp(M). Costs a tensor-reduction host sync per build; keep OFF on
    # the training hot path. The FIRST occurrence is reported even with this off, so leaving it off
    # no longer means silence.
    transport_clamp_monitor:             bool = False

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

    # Cosine-decay floors, combined as max(min_lr, min_lr_frac * base).
    min_lr:                    float = 0       # absolute floor: no group's LR decays below this;
    #                                          0.0 recovers the pure half-cosine-to-zero.
    min_lr_frac:               float = 0.01    # fractional floor: no group's LR decays below
    #                                          min_lr_frac * its OWN base LR, preserving the
    #                                          m_mu : m_sigma : m_phi base ratios into the tail.
    #                                          0.0 (with min_lr=0) is the pure half-cosine-to-zero path.

    # Regional torch.compile on the closed-form pair kernel (static shapes; matrix_exp/eigh stay
    # outside). Falls back to eager with a warning when the backend is unavailable (e.g. no triton).
    compile_pair_kernel:            bool = False
    generate_embedding_figures:     bool = True
    # Exact common-frame KL scoring for eligible flat full-Gaussian transports.
    # Unsupported paths retain the pairwise implementation; default preserves existing runs.
    full_cov_kl_backend:           str   = "common_frame"  # "common_frame" | "pairwise"

    


    def __post_init__(
        self,
        _allow_legacy_dense_decode: bool,
    ) -> None:
        for name, value in (
            ("_allow_legacy_dense_decode", _allow_legacy_dense_decode),
        ):
            if type(value) is not bool:
                raise _ConfigTypeError(f"{name} must be a plain bool")
        # An InitVar is not a dataclass field, so keep the admission on the instance for
        # serialize_config to persist (it shadows the class-level default False).
        self._allow_legacy_dense_decode = _allow_legacy_dense_decode
        for field in fields(self):
            if field.type in (bool, int):
                value = getattr(self, field.name)
                if type(value) is not field.type:
                    raise _ConfigTypeError(
                        f"{field.name} must be a plain {field.type.__name__}, got "
                        f"{type(value).__name__}: {value!r}"
                    )

        # Validate the declared relational envelope before downstream selector coercions.
        _require(self.attention_mode, ("transported", "projected_conditional", "projected_predictive"), "attention_mode")
        _require(self.relational_input_scope, ("head", "full"), "relational_input_scope")
        if type(self.attention_residual) is not bool:
            raise ValueError("attention_residual must be a bool")
        if self.attention_residual and self.attention_mode != "projected_predictive":
            raise ValueError("attention_residual requires attention_mode='projected_predictive'")
        _require(self.relational_output_scope, ("head", "full"), "relational_output_scope")
        if self.relational_output_scope == "full" and self.attention_mode != "projected_predictive":
            raise ValueError("relational_output_scope='full' requires predictive_content")
        if self.attention_mode == "projected_predictive" and self.relational_identity_content:
            raise ValueError("predictive_content requires learned content maps")
        _require(self.relational_routing, ("gaussian_kl", "noisy_query"), "relational_routing")
        if self.relational_routing == "noisy_query" and self.attention_mode != "projected_predictive":
            raise ValueError("relational_routing='noisy_query' requires attention_mode='projected_predictive'")
        for name in ("m_relational_score_lr", "m_relational_content_lr"):
            rate = getattr(self, name)
            if rate is not None and (
                    isinstance(rate, bool) or not isinstance(rate, Real)
                    or not math.isfinite(rate) or rate < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 or None, got {rate!r}")
        if self.attention_mode != "transported":
            requirements = {
                "divergence_family": "renyi",
                "renyi_order": 1.0, "include_attention_entropy": True,
                "gradient_mode": "filtering", "e_step_update": "gradient",
                "e_step_gradient": "unroll", "oracle_unroll_grad": True,
                "n_e_steps": 1,
                "e_phi_lr": 0.0, "transport_mode": "flat", "gauge_group": "block_glk",
            }
            # encode_mode admits 'per_token_additive' beside 'per_token': the additive encoder
            # zeroes the token frame and moves the same learned (V, n_gen) phi table into a mean
            # shift, which is the capacity-matched Omega = I control for a gauge on/off contrast on
            # this route. The projected executors read only the realized frame and the encoded
            # moments, so no step depends on the encoder beyond them. Its positional requirement
            # (pos_phi='none', pos_rotation='none') is enforced after the gauge_transport
            # coercion below, which can itself supply pos_phi='none'.
            # skip_belief_sigma_update is deliberately ABSENT from the envelope above: it is INERT
            # on every projected route, not incompatible. 'projected_predictive' replaces the mean
            # and keeps covariance (e_step.py predictive branch assigns sigma = belief.sigma and
            # never reads the flag); 'projected_conditional' filtering returns grad_sigma=None by
            # signature (relational.py:85), so there is no sigma arm to skip. The previous pin to
            # True only kept skip=False from reaching the float*None retraction; the None guard in
            # e_step_iteration covers that now. Inertness is reported by inactive_fields.
            # decode_mode is deliberately ABSENT from the envelope above: the prior-bank KL
            # decoders (diagonal_chunked / full_chunked / family_chunked) are live on this route.
            # The fused CE path already branches on active_decode_mode != 'linear' (model.py), and
            # family/cov-kind pairing for 'gaussian_full' + 'full_chunked' is enforced by the
            # decoder registration and the rank checks below. Pinning 'linear' was a support
            # default with no stated mathematical obstruction -- and the linear readout discards
            # Sigma_q, which is exactly what a full-covariance family should not do at decode.
            # compile_pair_kernel is deliberately ABSENT from the envelope above: it is INERT here,
            # not incompatible. Its only consumer is the closed-form pair kernel inside
            # belief_gradients (kernels.py:669), which this mode never reaches -- with relational
            # operators present (and 'projected_filtering' always builds them) the E-step takes
            # either relational_predictive_readout or compute_relational_filtering_gradients
            # (e_step.py:1125-1149), and belief_gradients sits in the else branch below both. The
            # knob was previously REJECTED, which turned a no-op into a config error and blocked a
            # click-to-run config that is otherwise in-envelope; the inertness advisory at the end of
            # this method now reports it instead. Kept out of the envelope rather than pinned False
            # so the field stays sweepable without tripping the sweep all-arms-inert guard.
            # gauge_transport is deliberately ABSENT from the envelope above: it is live in this
            # path and is an ablation cell the click-to-run
            # config must be able to reach. The frame is not bypassed here -- the conditional path
            # builds its shared transport from it (_build_shared_flat_transport ->
            # _canonical_frame_context, model.py:1265-1270), so gauge_transport='off' is the
            # Omega_ij=I arm of the A1/EXP-2 ablation and its own coercion supplies the rest of the
            # contract; and the learned partner residual folds into the effective log prior this
            # E-step descends (_effective_beta_log_prior, model.py:1286-1295), which the residual
            # test measures as a nonzero gradient on beta_prior_residual_weight.
            incompatible = [
                f"{name}={getattr(self, name)!r} (requires {expected!r})"
                for name, expected in requirements.items()
                if getattr(self, name) != expected
            ]
            if self.encode_mode not in ("per_token", "per_token_additive"):
                incompatible.append(
                    f"encode_mode={self.encode_mode!r} (requires 'per_token' or 'per_token_additive')")
            if self.family not in ("gaussian_diagonal", "gaussian_full",
                                   "gaussian_diagonal_exact", "gaussian_frame_diagonal"):
                incompatible.append(f"family={self.family!r} (requires a supported Gaussian family)")
            # Predictive content composes through the existing stack; conditional updates
            # retain their original single-layer support envelope.
            if self.attention_mode == "projected_conditional" and self.n_layers != 1:
                incompatible.append(f"n_layers={self.n_layers!r} (requires 1 for conditional_update)")
            if self.n_heads < 2:
                incompatible.append(f"n_heads={self.n_heads!r} (requires at least 2)")
            elif self.embed_dim % self.n_heads:
                incompatible.append(f"embed_dim={self.embed_dim!r} (requires equal heads)")
            if self.cross_couplings:
                incompatible.append(f"cross_couplings={self.cross_couplings!r} (requires none)")
            if incompatible:
                raise ValueError(
                    f"attention_mode={self.attention_mode!r} has unsupported projected_filtering configuration: "
                    + "; ".join(incompatible)
                )

        # encode_mode='canonical_content_gauge' stores the token tables as canonical, frame-intrinsic
        # (a_v, s_v) coordinates; its exactness needs the whole flat-cocycle contract, so reject the
        # incompatible fields here rather than let a downstream validator report a secondary symptom.
        if self.encode_mode == "canonical_content_gauge":
            incompatible = []
            if self.family != "gaussian_frame_diagonal":
                incompatible.append(f"family={self.family!r}")
            if self.decode_mode not in ("diagonal", "diagonal_chunked"):
                incompatible.append(f"decode_mode={self.decode_mode!r}")
            if incompatible:
                raise ValueError(
                    "encode_mode='canonical_content_gauge' requires "
                    "family='gaussian_frame_diagonal', "
                    "decode_mode='diagonal'/'diagonal_chunked'; got "
                    + ", ".join(incompatible)
                )

        # Practical canonical-content control: the token tables are intrinsic diagonal moments,
        # materialized once into the realized token-plus-position frame before the diagonal E-step.
        # The final query is pulled back to a full canonical covariance at decode, so the otherwise
        # rank-incompatible full decoder is intentional and mandatory for this one named mode.
        if self.encode_mode == "canonical_content_projected":
            incompatible = []
            if self.family != "gaussian_diagonal":
                incompatible.append(f"family={self.family!r}")
            if self.e_phi_lr != 0.0:
                incompatible.append(f"e_phi_lr={self.e_phi_lr!r}")
            if self.decode_mode not in ("full", "full_chunked"):
                incompatible.append(f"decode_mode={self.decode_mode!r}")
            if incompatible:
                raise ValueError(
                    "encode_mode='canonical_content_projected' requires "
                    "family='gaussian_diagonal', "
                    "e_phi_lr=0.0, decode_mode='full'/'full_chunked'; got "
                    + ", ".join(incompatible)
                )

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
        if self.transport_chart_max_norm is not None and not (
            math.isfinite(self.transport_chart_max_norm)
            and self.transport_chart_max_norm > 0.0
        ):
            raise ValueError(
                "transport_chart_max_norm must be None or finite and positive, got "
                f"{self.transport_chart_max_norm}"
            )
        if self.bch_residual_max is not None and not (
            math.isfinite(self.bch_residual_max) and self.bch_residual_max > 0.0
        ):
            raise ValueError(
                "bch_residual_max must be None or finite and positive, got "
                f"{self.bch_residual_max!r}"
            )

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
            self.e_phi_lr     = 0.0
            self.m_phi_lr     = 0.0
            import warnings
            warnings.warn(
                "gauge_transport='off': forcing the gauge frame to the identity (phi_scale=0.0, "
                "pos_phi='none', e_phi_lr=0.0, m_phi_lr=0.0), so "
                "the transport is identity and frozen.",
                UserWarning, stacklevel=2,
            )
        elif self.gauge_transport == "frozen":
            if self.phi_scale <= 0.0:
                raise ValueError(
                    f"gauge_transport='frozen' needs a nonzero random frame (phi_scale>0), got "
                    f"phi_scale={self.phi_scale}; use gauge_transport='off' for the identity frame."
                )
            self.e_phi_lr = 0.0
            self.m_phi_lr = 0.0
            import warnings
            warnings.warn(
                f"gauge_transport='frozen': freezing the random gauge frame at init "
                f"(phi_scale={self.phi_scale}, e_phi_lr=0.0, m_phi_lr=0.0); "
                "the frame is fixed, not learned.",
                UserWarning, stacklevel=2,
            )

        # divergence seam: divergence_family is the FUNCTIONAL (f-divergence) registry key, distinct
        # from `family` (the covariance-structure kernel). renyi_order is IGNORED by non-alpha
        # functionals (e.g. squared_hellinger). Validated against the functional REGISTRY, not a
        # hardcoded list, so a newly registered functional is config-selectable without editing here.
        from vfe3.divergence import divergence_functionals
        _require(self.divergence_family, divergence_functionals(), "divergence_family")
        if not (math.isfinite(self.renyi_order) and self.renyi_order > 0.0):
            raise ValueError(
                f"renyi_order must be finite and positive, got {self.renyi_order}"
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

        # gauge seam, validated against the group REGISTRY (not a hardcoded literal) so a newly
        # registered group is a valid config value without editing this validator.
        from vfe3.geometry.groups import _GROUPS
        _require(self.gauge_group, tuple(sorted(_GROUPS)), "gauge_group")
        # Sp(2m,R) lives in even dimension K = 2m; reject an odd embed_dim here, with a clear
        # message, rather than letting generate_sp raise mid-build.
        if self.gauge_group == "sp" and self.embed_dim % 2 != 0:
            raise ValueError(
                f"gauge_group='sp' (Sp(2m,R)) requires an EVEN embed_dim (K=2m), got "
                f"embed_dim={self.embed_dim}"
            )
        # so_n / sp_n irrep towers: validate group_n + irrep_spec at construction (labels resolve,
        # block dims sum to embed_dim) so a bad spec fails here with the computed dims.
        if self.gauge_group in ("so_n", "sp_n"):
            _algebra = "so" if self.gauge_group == "so_n" else "sp"
            if self.group_n is None or self.irrep_spec is None:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} requires both group_n and irrep_spec "
                    f"([(label, mult), ...]); got group_n={self.group_n}, "
                    f"irrep_spec={self.irrep_spec}"
                )
            if not isinstance(self.group_n, int) or self.group_n < 2:
                raise ValueError(f"group_n must be an int >= 2, got {self.group_n!r}")
            if self.gauge_group == "sp_n" and self.group_n % 2 != 0:
                raise ValueError(
                    f"gauge_group='sp_n' (Sp(2m,R)) requires an EVEN group_n (= 2m), got "
                    f"group_n={self.group_n}"
                )
            if not isinstance(self.irrep_spec, list) or not self.irrep_spec:
                raise ValueError(
                    f"irrep_spec must be a non-empty list of (label, mult) pairs, got "
                    f"{self.irrep_spec!r}"
                )
            # JSON has no tuple type; coerce reloaded list pairs to tuples (the cross_couplings
            # round-trip pattern above).
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
                _d = irrep_dim(self.group_n, algebra=_algebra, label=_entry[0])
                _block_dims.extend([_d] * _entry[1])
            if sum(_block_dims) != self.embed_dim:
                raise ValueError(
                    f"irrep_spec blocks {_block_dims} sum to {sum(_block_dims)} != "
                    f"embed_dim={self.embed_dim} (group_n={self.group_n})"
                )
            # One shared phi drives EVERY block, so the generators do not partition per block:
            # the per-block phi preconditioners are undefined (the tied_block_glk footprint).
            if self.phi_precond_mode in ("killing_per_block", "pullback_per_block"):
                raise ValueError(
                    f"phi_precond_mode={self.phi_precond_mode!r} is incompatible with "
                    f"gauge_group={self.gauge_group!r}: the shared so/sp generators act in every "
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
                f"group_n/irrep_spec are consumed only by gauge_group 'so_n'/'sp_n'; got "
                f"gauge_group={self.gauge_group!r}"
            )
        if self.use_cg_coupling and self.gauge_group not in ("so_n", "sp_n"):
            raise ValueError(
                f"use_cg_coupling requires an irrep-labeled tower group ('so_n'/'sp_n'); got "
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
        # The head mixer needs >= 2 gauge blocks to mix. Two single-block cases differ:
        #   (1) a head-block group (block_glk/tied_block_glk) with n_heads < 2 is a single-HEAD
        #       artifact of a head/K sweep -- nothing to mix -- so AUTO-DISABLE and warn, letting
        #       use_head_mixer=True stay set across a sweep;
        #   (2) a genuinely single-BLOCK group (glk/so_k/sp) cannot mix regardless of head count, a
        #       static misconfiguration, so RAISE. so_n/sp_n mix per isotypic component.
        if (self.use_head_mixer
                and self.gauge_group in ("block_glk", "tied_block_glk")
                and self.n_heads < 2):
            import warnings
            warnings.warn(
                f"use_head_mixer=True with gauge_group={self.gauge_group!r} and n_heads={self.n_heads} "
                f"yields a single gauge block (nothing to mix); auto-disabling use_head_mixer. Set "
                f"n_heads >= 2 to use the head mixer.",
                UserWarning,
                stacklevel=2,
            )
            self.use_head_mixer = False
        if self.use_head_mixer and self.gauge_group in ("glk", "so_k", "sp"):
            raise ValueError(
                f"use_head_mixer=True needs >= 2 gauge blocks to mix, but "
                f"gauge_group={self.gauge_group!r} is a single-block group. Use "
                f"block_glk/tied_block_glk with n_heads >= 2, or an so_n/sp_n irrep tower."
            )
        if self.use_priorbank_head_evidence_mixer:
            if self.decode_mode == "linear":
                raise ValueError(
                    "use_priorbank_head_evidence_mixer=True requires a prior-bank decode_mode."
                )
            if ((self.gauge_group in ("block_glk", "tied_block_glk") and self.n_heads < 2)
                    or self.gauge_group in ("glk", "so_k", "sp")
                    or (self.gauge_group in ("so_n", "sp_n") and len(_block_dims) < 2)):
                raise ValueError(
                    "use_priorbank_head_evidence_mixer=True requires at least two gauge blocks. "
                    f"gauge_group={self.gauge_group!r} does not provide a multi-block PriorBank."
                )
        # transport_mode selects the connection REGIME, validated against the transport REGISTRY so a
        # newly registered regime is a valid config value without editing this validator.
        from vfe3.geometry.transport import _TRANSPORTS
        _require(self.transport_mode, tuple(sorted(_TRANSPORTS)), "transport_mode")
        # cross_couplings (off-block GL(K) head coupling) is supported only by a group builder that
        # accepts the kwarg, detected by inspecting the registered builder's signature rather than a
        # hardcoded name list. Each pair must be a distinct in-range directed head pair.
        if self.cross_couplings is not None:
            import inspect
            from vfe3.geometry.groups import get_group
            builder = get_group(self.gauge_group)
            if "cross_couplings" not in inspect.signature(builder).parameters:
                raise ValueError(
                    f"cross_couplings is not supported by gauge_group={self.gauge_group!r} (its "
                    f"builder does not accept the kwarg); use 'block_glk', or leave cross_couplings "
                    f"None"
                )
            if not isinstance(self.cross_couplings, list):
                raise ValueError(
                    f"cross_couplings must be a list of (int, int) head pairs, got "
                    f"{type(self.cross_couplings).__name__}"
                )
            # JSON has no tuple type, so a config.json reloaded by viz.report._load_config hands
            # back list pairs; coerce them to tuples so the round-trip rebuild does not trip the
            # isinstance(pair, tuple) gate below (and downstream builders see a consistent type).
            self.cross_couplings = [tuple(p) if isinstance(p, list) else p
                                    for p in self.cross_couplings]
            for pair in self.cross_couplings:
                if (not isinstance(pair, tuple) or len(pair) != 2
                        or not all(isinstance(x, int) for x in pair)):
                    raise ValueError(
                        f"each cross_couplings entry must be an (int, int) head pair, got {pair!r}"
                    )
                a, b = pair
                if a == b:
                    raise ValueError(f"cross_couplings self-coupling ({a},{a}) not allowed (a != b)")
                if not (0 <= a < self.n_heads and 0 <= b < self.n_heads):
                    raise ValueError(
                        f"cross_couplings head indices ({a},{b}) out of range [0, {self.n_heads})"
                    )
            # Cross-coupling's off-block generators destroy the per-head direct sum, so the builder
            # reports a SINGLE irrep block [K]: n_heads no longer sets the runtime attention-head
            # count and the energy is one full-K block. Fail FAST on combinations that would otherwise
            # crash late at model construction, and warn on the semantic shift.
            import warnings
            for _pname in ("beta_attention_prior",):
                if getattr(self, _pname) in ("alibi", "causal_alibi", "causal_alibi_noself"):
                    raise ValueError(
                        f"{_pname}={getattr(self, _pname)!r} is incompatible with cross_couplings: an "
                        f"alibi prior builds an (n_heads, N, N) per-head bias, but cross-coupled "
                        f"block_glk collapses irrep_dims to a single block [K] (one runtime attention "
                        f"head). Use a headless prior (uniform/causal/causal_windowed/...) with "
                        f"cross_couplings."
                    )
            if self.use_head_mixer:
                raise ValueError(
                    "use_head_mixer=True is incompatible with cross_couplings: cross-coupled block_glk "
                    "collapses irrep_dims to a single block [K], and the head mixer needs >= 2 blocks "
                    "to mix. Disable use_head_mixer or remove cross_couplings."
                )
            if self.use_priorbank_head_evidence_mixer:
                raise ValueError(
                    "use_priorbank_head_evidence_mixer=True is incompatible with cross_couplings: "
                    "cross-coupled block_glk collapses irrep_dims to a single block [K], but head "
                    "evidence requires at least two gauge blocks. Disable the mixer or remove "
                    "cross_couplings."
                )
            for _name in ("kappa_beta",):
                if isinstance(getattr(self, _name), (list, tuple)):
                    raise ValueError(
                        f"{_name} per-head list is incompatible with cross_couplings: the cross-coupled "
                        f"group has a single irrep block [K], not n_heads blocks. Use a scalar {_name}."
                    )
            if self.family == "gaussian_diagonal":
                warnings.warn(
                    "cross_couplings with family='gaussian_diagonal' is an APPROXIMATION: the off-block "
                    "GL(K) congruence g*Sigma*g^T is not diagonal, so the diagonal readout is not "
                    "gauge-invariant. family='gaussian_full' (diagonal_covariance=False) is the exact "
                    "congruence path.",
                    UserWarning, stacklevel=2,
                )
            if self.n_heads > 1:
                warnings.warn(
                    f"cross_couplings collapses block_glk to a single irrep block [K]: n_heads="
                    f"{self.n_heads} no longer sets the attention-head count (runtime is 1 head) and "
                    f"the softmax temperature shifts from kappa*sqrt(d_head) to kappa*sqrt(K). This is "
                    f"the coherent single-block cross-coupled gauge; n_heads only partitions the "
                    f"generators.",
                    UserWarning, stacklevel=2,
                )

        # belief family. ``family`` is the SINGLE covariance-structure toggle: a registry key whose
        # declared cov_kind ('diagonal'|'full') is the one source of truth for the diagonal-vs-full
        # path, exposed read-only as the derived ``diagonal_covariance`` property. ``divergence_family``
        # is the SEPARATE functional seam and is NOT forced equal to family.
        from vfe3.divergence import divergence_families, family_cov_kind
        _require(self.family, divergence_families(), "family")
        family_is_diagonal = family_cov_kind(self.family) == "diagonal"
        # Family SCOPE gate: a family whose coupling energy is defined only on part of the config
        # space declares that scope on its class; reject an unsupported pairing HERE rather than at the
        # first forward, where ablation.py catches config errors only around VFE3Config(**cfg_dict) and
        # a mid-forward raise is misfiled as error_kind="train". Both checks read the family and
        # transport REGISTRIES, so a new registration is covered without editing this validator.
        from vfe3.families.base import get_family
        _family_cls = get_family(self.family)
        if _family_cls.requires_kl_divergence and (
                self.divergence_family != "renyi" or self.renyi_order != 1.0):
            raise ValueError(
                f"family={self.family!r} implements its coupling energy for KL only "
                f"(divergence_family='renyi', renyi_order=1.0); got "
                f"divergence_family={self.divergence_family!r}, renyi_order={self.renyi_order}. "
                f"Use family='gaussian_diagonal' for a noncanonical divergence."
            )
        if _family_cls.transport_requirement != "any":
            from vfe3.geometry.transport import get_transport_registration
            _registration = get_transport_registration(self.transport_mode)
            if not _registration.satisfies(_family_cls.transport_requirement):
                raise ValueError(
                    f"family={self.family!r} needs a "
                    f"{_family_cls.transport_requirement!r} pairwise transport, but "
                    f"transport_mode={self.transport_mode!r} provides "
                    f"{_registration.pair_transport_kind!r}. Use a family "
                    f"whose coupling energy consumes this transport (e.g. 'gaussian_diagonal')."
                )

        # free-energy coupling
        for _name in ("kappa_beta",):
            _v = getattr(self, _name)
            if isinstance(_v, (list, tuple)):
                if self.gauge_group in ("so_n", "sp_n"):
                    # heads = irrep blocks (possibly unequal dims): one kappa entry per block
                    # (irrep_spec already validated/coerced in the gauge seam above).
                    _n_blocks = sum(m for _, m in self.irrep_spec)
                    if len(_v) != _n_blocks:
                        raise ValueError(
                            f"{_name} list must have one entry per irrep block; gauge_group="
                            f"{self.gauge_group!r} has {_n_blocks} blocks, got {len(_v)}")
                elif self.gauge_group not in ("block_glk", "tied_block_glk"):
                    raise ValueError(
                        f"{_name} list (per-head) requires an equal-block group "
                        f"(block_glk/tied_block_glk); got gauge_group={self.gauge_group!r}")
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
        for name in ("mu_init_std", "phi_scale"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        # lambda_alpha_mode validated against the alpha-form REGISTRY (add-by-registering).
        from vfe3.alpha_i import _ALPHAS
        _require(self.lambda_alpha_mode, tuple(sorted(_ALPHAS)), "lambda_alpha_mode")
        if (self.lambda_alpha_mode == "constant"
                and not (math.isfinite(self.lambda_alpha) and self.lambda_alpha >= 0.0)):
            raise ValueError(
                f"lambda_alpha must be finite and nonnegative under lambda_alpha_mode='constant', "
                f"got {self.lambda_alpha}"
            )
        # Full Gaussians expose an ordered conditional KL decomposition, not a
        # sum of marginal KLs. General Renyi orders have no such chain rule.
        from vfe3.alpha_i import alpha_is_per_coord
        if alpha_is_per_coord(self.lambda_alpha_mode) and not family_is_diagonal:
            conditional_kl = getattr(_family_cls, "conditional_kl_coordinates", False)
            supported = (self.divergence_family == "jeffreys" or
                         (self.divergence_family == "renyi" and self.renyi_order == 1.0))
            if not conditional_kl or not supported:
                raise ValueError(
                    "per-coordinate full-covariance self-coupling requires a family with "
                    "conditional KL coordinates and divergence_family='renyi', renyi_order=1 "
                    "(KL), or 'jeffreys'")
        # The per-coordinate self-divergence exists only for a divergence that DECOMPOSES
        # coordinate-wise on a diagonal Gaussian: Renyi/KL and the two divergences AFFINE in it
        # (Bhattacharyya = 0.5 D_{1/2}, Jeffreys = KL + KL_rev). squared_hellinger is excluded
        # (H^2 = 1 - exp(-D_{1/2}/2) is a nonlinear transform of the SUMMED divergence).
        from vfe3.divergence import has_per_coord_functional, divergence_functionals_per_coord
        if alpha_is_per_coord(self.lambda_alpha_mode) and not has_per_coord_functional(self.divergence_family):
            raise ValueError(
                f"lambda_alpha_mode={self.lambda_alpha_mode!r} needs a per-coordinate self-divergence, which is "
                f"implemented only for divergences that decompose coordinate-wise on a diagonal "
                f"Gaussian ({divergence_functionals_per_coord()}); got "
                f"divergence_family={self.divergence_family!r} (e.g. 'squared_hellinger' does not "
                f"decompose). Use a decomposable divergence or a per-position lambda_alpha_mode."
            )
        # Calibration footgun: the state-dependent envelope alpha* = c0/(b0 + D) is correct for ANY
        # divergence, but its RANGE collapses when D is BOUNDED. squared_hellinger has D = H^2 in
        # [0,1), so the alpha* ratio max/min = 1 + 1/b0 -- at the default b0 = 1 only ~2x, versus
        # ~1 + kl_max/b0 for the unbounded KL, leaving the adaptive coupling nearly inert. Set
        # b0 ~ Dmax/10 (~0.1) to restore a wide range. Only squared_hellinger is bounded-small.
        if (self.lambda_alpha_mode in ("state_dependent", "state_dependent_per_coord")
                and self.divergence_family == "squared_hellinger"):
            _b0_vals = self.b0 if isinstance(self.b0, (list, tuple)) else [self.b0]
            if min(_b0_vals) >= 1.0:
                import warnings
                warnings.warn(
                    f"lambda_alpha_mode={self.lambda_alpha_mode!r} with divergence_family='squared_hellinger' and "
                    f"b0={self.b0}: squared Hellinger is bounded (H^2 in [0,1)), so the state-dependent "
                    f"alpha* = c0/(b0 + D) spans only a ~{1.0 + 1.0/min(_b0_vals):.1f}x range and the "
                    f"adaptive self-coupling is nearly constant. Set b0 ~ 0.1 (Dmax/10) for a wide "
                    f"alpha* range, or use an unbounded divergence (renyi/KL).",
                    UserWarning,
                    stacklevel=2,
                )

        # attention
        _require(self.beta_attention_prior, tuple(sorted(_PRIORS)), "beta_attention_prior")

        # E-step
        for name in ("e_q_mu_lr", "e_q_sigma_lr", "e_phi_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be >= 0 and finite, got {value}")
        _require(self.gradient_mode, _VALID_GRADIENT_MODES, "gradient_mode")
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
            if self.gauge_group == "glk":
                if self.embed_dim > 12:
                    raise ValueError(
                        "pullback_group with gauge_group='glk' requires embed_dim <= 12; got "
                        f"embed_dim={self.embed_dim}"
                    )
            elif self.gauge_group == "block_glk":
                if self.cross_couplings is not None:
                    raise ValueError(
                        "pullback_group requires untied block_glk without cross_couplings"
                    )
                if self.d_head > 12:
                    raise ValueError(
                        "pullback_group with gauge_group='block_glk' requires d_head <= 12; got "
                        f"d_head={self.d_head}"
                    )
            else:
                raise ValueError(
                    "pullback_group phase one accepts only built-in glk or untied block_glk; "
                    f"got gauge_group={self.gauge_group!r}"
                )
            if self.pos_phi_project_slk:
                raise ValueError(
                    "pullback_group does not support pos_phi_project_slk=True because the stored "
                    "factor metric is defined on the full registered chart"
                )
            factor_radius = (
                self.phi_mstep_max_matrix_norm
                if self.phi_mstep_max_matrix_norm is not None
                else 5.0
            )
            from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM
            if self.transport_chart_max_norm is None or not (
                factor_radius
                < self.transport_chart_max_norm
                < TRANSPORT_CLAMP_MAX_NORM
            ):
                raise ValueError(
                    "pullback_group requires an explicit transport_chart_max_norm satisfying "
                    f"{factor_radius} < transport_chart_max_norm < "
                    f"{TRANSPORT_CLAMP_MAX_NORM}; got {self.transport_chart_max_norm!r}"
                )

        from vfe3.model.positional_phi import _POS_PHI
        _require(self.pos_phi, tuple(sorted(_POS_PHI)), "pos_phi")
        from vfe3.geometry.rope import _POS_ROTATIONS
        _require(self.pos_rotation, tuple(sorted(_POS_ROTATIONS)), "pos_rotation")
        # Gauge-RoPE breaks GLOBAL gauge equivariance for EVERY group: the position-fixed
        # R(theta_i) do not commute with a global gauge element g (probe: beta residual 0.44-0.61
        # under rope vs 7e-6 without), including rope_full_gauge=True. The group-specific warnings
        # below cover only the structure-group departures and the means-only incoherence, so this one
        # is unconditional. The equivariant positional path is pos_phi or pos_rotation='none'.
        if self.pos_rotation == "rope":
            import warnings
            warnings.warn(
                "pos_rotation='rope' is a deliberate residual gauge-FIXING layer: the position-fixed "
                "rotations R(theta_i) do not commute with a global gauge element, so gauge-RoPE "
                "breaks the global gauge equivariance of F/beta for every group (probe: beta "
                "residual 0.44-0.61 under rope vs 7e-6 without). The equivariant positional paths "
                "are pos_phi='frozen'/'learned' (composed into phi) or pos_rotation='none'.",
                UserWarning,
                stacklevel=2,
            )
        if self.rope_full_gauge and self.diagonal_covariance:
            raise ValueError(
                "rope_full_gauge=True rotates the covariance sandwich (R Sigma R^T), which the "
                "diagonal-covariance approximation cannot carry; set family='gaussian_full'."
            )
        # RoPE rotates ADJACENT coordinate pairs (2k, 2k+1) while Sp(2m) pairs coordinate i with m+i
        # (J = [[0,I],[-I,0]]), so the rope-wrapped transport R Omega R^T leaves the symplectic group.
        # R is orthogonal, so each per-pair divergence stays a well-defined GL(K)-congruence value.
        # Warn (not error): the symplectic property is not consumed downstream.
        if self.pos_rotation == "rope" and self.gauge_group == "sp":
            import warnings
            warnings.warn(
                "pos_rotation='rope' with gauge_group='sp' leaves the symplectic group: RoPE rotates "
                "adjacent pairs (2k, 2k+1) while Sp(2m) pairs i with m+i, so R Omega R^T is no longer "
                "in Sp(2m). The GL(K)-congruence divergence invariance still holds (R is orthogonal), "
                "so the model runs; only the structure-group claim is dropped.",
                UserWarning,
                stacklevel=2,
            )
        # Same structural fact for the irrep-tower groups: RoPE's adjacent-pair rotation R is
        # orthogonal but generally NOT in the irrep image rho(SO(N))/rho(Sp(2m)) inside GL(K), so
        # R Omega R^T leaves the SELECTED structure group (GL(K)-congruence invariance survives).
        if self.pos_rotation == "rope" and self.gauge_group in ("so_n", "sp_n"):
            import warnings
            warnings.warn(
                f"pos_rotation='rope' with gauge_group={self.gauge_group!r} leaves the structure "
                f"group: the coordinate-pair rotation R(theta) is generally not in the irrep image "
                f"of SO(N)/Sp(2m), so R Omega R^T is no longer in the selected group. The "
                f"GL(K)-congruence divergence invariance still holds (R is orthogonal), so the "
                f"model runs; only the structure-group claim is dropped.",
                UserWarning,
                stacklevel=2,
            )
        # Means-only RoPE (the default pairing) transports mu under R Omega R^T but Sigma under the
        # UN-rotated Omega -- an affine-incoherent operator pair that does not preserve the Mahalanobis
        # form (probe: invariant drift 0.18 -> 0.74, while rope_full_gauge=True preserves it).
        # Available as a deliberate cheap approximation; warn so the pairing is explicit.
        # The full-covariance KL precision spellings are inlined rather than imported from
        # vfe3.families.gaussian (importing a family module inside __post_init__ perturbs import
        # order). set_full_cov_kl_precision validates the same set.
        if self.full_cov_kl_backend not in ("pairwise", "common_frame"):
            raise ValueError(
                "full_cov_kl_backend must be one of ('pairwise', 'common_frame'), "
                f"got {self.full_cov_kl_backend!r}")
        if self.full_cov_kl_precision not in ("fp64", "fp32_escalate", "fp32_escalate_cond"):
            raise ValueError(
                "full_cov_kl_precision must be one of ('fp64', 'fp32_escalate', "
                f"'fp32_escalate_cond'), got {self.full_cov_kl_precision!r}")
        if not isinstance(self.decode_ranking_fp64_escalation, bool):
            raise ValueError(
                "decode_ranking_fp64_escalation must be bool, got "
                f"{self.decode_ranking_fp64_escalation!r}")
        if self.full_cov_congruence_precision not in ("fp64", "fp32_escalate"):
            raise ValueError(
                "full_cov_congruence_precision must be one of ('fp64', 'fp32_escalate'), "
                f"got {self.full_cov_congruence_precision!r}")
        if not isinstance(self.full_cov_congruence_certification, bool):
            raise ValueError(
                "full_cov_congruence_certification must be bool, got "
                f"{self.full_cov_congruence_certification!r}")
        if self.safe_cholesky_jitter_mode not in ("absolute", "relative"):
            raise ValueError(
                "safe_cholesky_jitter_mode must be one of ('absolute', 'relative'), "
                f"got {self.safe_cholesky_jitter_mode!r}")
        if (not isinstance(self.mu_trust_cholesky_rounds, int)
                or isinstance(self.mu_trust_cholesky_rounds, bool)
                or self.mu_trust_cholesky_rounds < 0):
            raise ValueError(
                "mu_trust_cholesky_rounds must be a nonnegative int, got "
                f"{self.mu_trust_cholesky_rounds!r}")
        if (not isinstance(self.val_diagnostic_batches, int)
                or isinstance(self.val_diagnostic_batches, bool)
                or self.val_diagnostic_batches < 1):
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
            # for a block-diagonal group with MORE THAN ONE equal block; a single-block group (glk,
            # so_k) falls to the dense pairwise Omega, so the group alone settles it.
            _builder = get_group(self.gauge_group)
            try:                                 # builders differ: only the block groups take n_heads
                _dims = _builder(K=self.embed_dim, n_heads=self.n_heads).irrep_dims
            except TypeError:
                _dims = _builder(K=self.embed_dim).irrep_dims
            if len(_dims) < 2:
                raise ValueError(
                    f"rope_insertion='right' needs a group whose "
                    f"transport factors per block, but gauge_group={self.gauge_group!r} has a "
                    f"single irrep block ({_dims}), so the flat transport is built as a dense "
                    f"pairwise Omega with no per-vertex frame to fold into. {_escapes}")
        if self.pos_rotation == "rope" and not self.rope_full_gauge:
            import warnings
            warnings.warn(
                "pos_rotation='rope' with rope_full_gauge=False rotates the transported MEANS but "
                "not the covariance sandwich -- an affine-incoherent transport pair (Mahalanobis "
                "invariants are not preserved). The coherent gauge-RoPE is rope_full_gauge=True "
                "with a full-covariance family.",
                UserWarning,
                stacklevel=2,
            )
        # The per-block preconditioners build a per-HEAD metric and need generators that partition per
        # block (block_glk's independent gl(d_head) per head). The tied gauge's shared generators
        # kron(I_n, gl(d_head)) each act on EVERY block, so no per-block partition exists; reject BOTH
        # variants here. The ambient 'killing'/'pullback', 'clip' and 'none' are unaffected.
        if (self.gauge_group == "tied_block_glk"
                and self.phi_precond_mode in ("killing_per_block", "pullback_per_block")):
            raise ValueError(
                f"phi_precond_mode={self.phi_precond_mode!r} is incompatible with gauge_group="
                "'tied_block_glk': the shared kron(I_n, gl(d)) generators do not partition per head, "
                "so the per-block metric is undefined. Use 'none', 'clip', or the ambient "
                "'killing'/'pullback' (the ambient 'pullback' is feasible only up to K<=12 -- "
                "its structure-constants build is O(K^6))."
            )

        # decode / encode
        if not math.isfinite(self.decode_tau) or self.decode_tau <= 0.0:
            raise ValueError(f"decode_tau must be finite and positive, got {self.decode_tau}")
        if not self.decode_temperature_learned and self.decode_mode == "linear":
            raise ValueError(
                "decode_temperature_learned=False freezes decode_log_scale, which only the prior-bank "
                "decoders read; decode_mode='linear' discards tau_eff")
        _require(self.relational_output_init, ("identity", "depth_scaled"), "relational_output_init")
        if self.relational_output_init == "depth_scaled" and (
                self.attention_mode == "transported" or self.relational_identity_content):
            raise ValueError(
                "relational_output_init='depth_scaled' scales the learned relational output map O, "
                "which exists only on the projected attention routes with learned content maps")
        # decode_mode / encode_mode validated against the LIVE decode/encode REGISTRIES (not static
        # literals) so a newly registered kernel is config-selectable without editing this validator.
        # Dense kernels remain registered as exact legacy/oracle implementations but fresh public
        # configs reject them; chunked counterparts are the supported training routes.
        from vfe3.model.prior_bank import (
            _DECODERS,
            _ENCODERS,
            has_builtin_canonical_content_encoder,
            has_builtin_canonical_content_family,
            has_builtin_head_evidence_decoder,
            has_builtin_projected_full_family,
            has_builtin_projected_full_decoder,
        )
        _require(self.decode_mode, tuple(sorted(_DECODERS)), "decode_mode")
        if self.decode_mode in _DENSE_DECODE_MODES and not _allow_legacy_dense_decode:
            raise ValueError(
                f"decode_mode={self.decode_mode!r} is a dense legacy/oracle decoder and is not "
                "admitted for fresh training; select its chunked counterpart"
            )
        # decode_mode sets the covariance RANK(s) the prior-bank kernel accepts, via its resolved
        # DecodeRegistration.covariance_kinds: 'diagonal*' -> {'diagonal'}, 'full*' -> {'full'},
        # 'family*' -> both. The configured family's cov_kind must be a MEMBER of that set, else the
        # mismatch is a shape RuntimeError at the first forward. The linear registration discards
        # sigma, so these rank checks apply only to non-linear registrations.
        decode_registration = _DECODERS[self.decode_mode]
        if (self.decode_mode != "linear"
                and decode_registration.family_consistent
                and self.divergence_family == "renyi"
                and self.renyi_order > 1.0):
            raise ValueError(
                "family-consistent prior-bank Renyi decoding requires renyi_order <= 1 "
                "because alpha > 1 cannot guarantee a finite score across the entire prior bank; "
                f"got renyi_order={self.renyi_order}.")

        if (
            self.encode_mode in ("canonical_content_gauge", "canonical_content_projected")
            and not has_builtin_canonical_content_encoder(self.encode_mode)
        ):
            raise ValueError(
                f"encode_mode={self.encode_mode!r} requires its built-in encoder registration "
                "and callable identities; registry overrides are not eligible."
            )
        if (
            self.encode_mode == "canonical_content_gauge"
            and not has_builtin_canonical_content_family(self.encode_mode)
        ):
            raise ValueError(
                "encode_mode='canonical_content_gauge' requires the import-time "
                "gaussian_frame_diagonal family identity; registry overrides are not eligible."
            )
        if (
            self.encode_mode == "canonical_content_projected"
            and not has_builtin_canonical_content_family(self.encode_mode)
        ):
            raise ValueError(
                "encode_mode='canonical_content_projected' requires the import-time "
                "gaussian_diagonal family identity used by its canonical table; registry "
                "overrides are not eligible."
            )
        if (
            self.encode_mode == "canonical_content_projected"
            and not has_builtin_projected_full_family()
        ):
            raise ValueError(
                "encode_mode='canonical_content_projected' requires the import-time "
                "gaussian_full family identity used by its analytic full decoder; registry "
                "overrides are not eligible."
            )
        if (
            self.encode_mode == "canonical_content_projected"
            and not has_builtin_projected_full_decoder(self.decode_mode)
        ):
            raise ValueError(
                "encode_mode='canonical_content_projected' requires the built-in analytic "
                "'full'/'full_chunked' decode registration and callable identities; "
                "registry overrides are not eligible."
            )
        if self.use_priorbank_head_evidence_mixer:
            canonical_modes = {
                "gaussian_diagonal": ("diagonal", "diagonal_chunked"),
                "gaussian_full": ("full", "full_chunked"),
            }
            if self.divergence_family != "renyi" or self.renyi_order != 1.0:
                raise ValueError(
                    "use_priorbank_head_evidence_mixer=True requires the built-in canonical KL "
                    "functional (divergence_family='renyi', renyi_order=1.0)."
                )
            allowed_modes = canonical_modes.get(self.family)
            projected_full_decode = (
                self.encode_mode == "canonical_content_projected"
                and self.family == "gaussian_diagonal"
                and self.decode_mode in ("full", "full_chunked")
            )
            if ((allowed_modes is None or self.decode_mode not in allowed_modes)
                    and not projected_full_decode):
                raise ValueError(
                    "use_priorbank_head_evidence_mixer=True requires a built-in canonical KL decode: "
                    "family='gaussian_diagonal' with decode_mode='diagonal'/'diagonal_chunked', or "
                    "family='gaussian_full' with decode_mode='full'/'full_chunked'; the sole rank "
                    "exception is encode_mode='canonical_content_projected', which pulls its "
                    "gaussian_diagonal query back to full covariance before full decoding. "
                    f"Got family={self.family!r}, decode_mode={self.decode_mode!r}."
                )
            from vfe3.families.base import _FAMILIES, _FUNCTIONALS, renyi
            from vfe3.families.gaussian import DiagonalGaussian, FullGaussian
            canonical_families = {
                "gaussian_diagonal": DiagonalGaussian,
                "gaussian_full": FullGaussian,
            }
            if (
                _FAMILIES.get(self.family) is not canonical_families[self.family]
                or _FUNCTIONALS.get("renyi") is not renyi
                or not has_builtin_head_evidence_decoder(self.decode_mode)
            ):
                raise ValueError(
                    "use_priorbank_head_evidence_mixer=True requires built-in canonical registry "
                    "identities for its Gaussian family, the renyi functional, and its KL decoder; "
                    "registry overrides are not eligible."
                )
        family_kind = family_cov_kind(self.family)
        if (self.decode_mode != "linear"
                and self.encode_mode != "canonical_content_projected"
                and family_kind not in decode_registration.covariance_kinds):
            raise ValueError(
                f"decode_mode={self.decode_mode!r} is rank-incompatible with family={self.family!r}: "
                f"it scores covariance kinds {sorted(decode_registration.covariance_kinds)} but the "
                f"family is {family_kind!r}. Pair a matching decode_mode (a 'full'/'diagonal' kernel "
                f"with a full/diagonal family, or 'family'/'family_chunked' which score both), or "
                f"select decode_mode='linear'."
            )
        if self.decode_chunk_size < 1:
            raise ValueError(f"decode_chunk_size must be >= 1, got {self.decode_chunk_size}")
        _require(self.decode_ce_checkpoint, ("auto", "always", "never"), "decode_ce_checkpoint")
        # decode_bias (linear decode only) and decode_tau (prior-bank decode only) are reported by
        # inactive_fields(), so the default-guarded notice below covers them and untouched
        # defaults stay silent.
        # Capability validation: the optimized 'diagonal'/'full' kernels hardcode a gaussian alpha=1
        # KL readout and ignore divergence_family/renyi_order, so a non-Gaussian family or a
        # noncanonical divergence would read the belief out under the WRONG geometry. Such a config on
        # a prior-bank route MUST select a family-consistent decode kernel
        # ('family'/'family_chunked'), which scores logits through the configured family AND
        # divergence. Canonical gaussian/renyi/alpha=1 configs keep their fast modes.
        if self.decode_mode != "linear":
            noncanonical = self.renyi_order != 1.0 or self.divergence_family != "renyi"
            # What the fast decode kernels assume is the Gaussian POINTWISE readout, which a family
            # overriding only its coupling/transport seams still satisfies.
            non_gaussian = not _family_cls.gaussian_pointwise_algebra
            if (noncanonical or non_gaussian) and not decode_registration.family_consistent:
                raise ValueError(
                    f"prior-bank decode with family={self.family!r}/"
                    f"divergence_family={self.divergence_family!r}/renyi_order={self.renyi_order} "
                    f"requires a family-consistent decode_mode ('family' or 'family_chunked') that "
                    f"reads logits out under the configured divergence; decode_mode="
                    f"{self.decode_mode!r} is a fixed gaussian alpha=1 KL kernel and would project "
                    f"the belief through the wrong geometry (the argmax can flip). Set "
                    f"decode_mode='family'/'family_chunked', or decode_mode='linear' (which is "
                    f"family-agnostic)."
                )
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
            if self.attention_mode != "transported":
                raise ValueError(detail + " on the projected routes")
            import warnings
            warnings.warn(detail + " for a capacity-matched gauge contrast.", UserWarning,
                          stacklevel=2)
        if self.encode_mode == "canonical_content_gauge":
            import warnings
            warnings.warn(
                "encode_mode='canonical_content_gauge' is frame-intrinsic: under the exact flat "
                "cocycle, token phi cancels from the supervised belief/decode path, so its "
                "Gaussian divergence is exact and phi_embed receives no supervised gradient unless "
                "another objective explicitly acts on phi. An optional additive unigram base-rate "
                "bias kappa*log(pi_v) is applied after that divergence and is outside the gauge/"
                "covariance map.",
                UserWarning,
                stacklevel=2,
            )
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

        if type(self.block_mlp_expansion) is not int or self.block_mlp_expansion < 1:
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
        if self.use_block_mlp and (
                block_mlp_registration.gauge_class == "invariant_scalar_intertwiner"
                or self.block_mlp_covariance == "delta_full"
        ) and self.family != "gaussian_full":
            raise ValueError(
                "block_mlp_mode='gauge_gate' and block_mlp_covariance='delta_full' need a "
                "full-covariance Gaussian family (family='gaussian_full'); invariant "
                "Mahalanobis gates and dense J Sigma J^T cannot be represented by a "
                f"diagonal/non-Gaussian family (got family={self.family!r})."
            )
        if self.use_block_mlp and block_mlp_registration.requires_frame_context:
            incompatible = []
            if self.e_phi_lr != 0.0:
                incompatible.append(f"e_phi_lr={self.e_phi_lr!r}")
            if self.pos_rotation == "rope":
                if self.rope_insertion != "right":
                    incompatible.append(f"rope_insertion={self.rope_insertion!r}")
                if not self.rope_full_gauge:
                    incompatible.append("rope_full_gauge=False")
                if not self.rope_on_value:
                    incompatible.append("rope_on_value=False")
            if incompatible:
                raise ValueError(
                    "block_mlp_mode='canonical_frame' requires a fixed factored flat frame: "
                    "e_phi_lr=0, and active RoPE must use right insertion with "
                    "rope_full_gauge=True and rope_on_value=True; got "
                    + ", ".join(incompatible)
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

        # M-step / training
        # e_step_gradient is the single E-step backward-estimator selector.
        _require(self.e_step_gradient, _VALID_E_STEP_GRADIENTS, "e_step_gradient")
        if self.use_block_mlp and self.e_step_gradient == "detach":
            raise ValueError(
                "use_block_mlp=True requires a gradient-carrying E-step; "
                "set e_step_gradient='unroll'."
            )
        # Both 'straight_through' and 'detach' sever the per-iteration E-step tangent, so a learnable
        # parameter whose only loss path IS that tangent receives no gradient and silently freezes.
        # phi_embed is the primary such parameter: the gauge frame's only loss path is Omega(phi)
        # inside the E-step tangent whenever mass_phi == 0, so those estimators freeze it (measured
        # |grad| 8.2e-03 under 'unroll' versus None under 'straight_through'). Warn, non-breaking --
        # 'unroll' is the default that trains it. pos_phi='learned' also freezes here but is the
        # DEFAULT and is warned at the MODEL level instead, keeping a default config silent.
        _phi_frame_is_tangent_only = (self.phi_scale != 0.0 and self.mass_phi == 0.0)
        if self.e_step_gradient in ("straight_through", "detach") and _phi_frame_is_tangent_only:
            import warnings
            warnings.warn(
                f"e_step_gradient={self.e_step_gradient!r} severs the per-iteration E-step tangent, so a "
                "learnable parameter that enters the loss only through it (prior_bank.phi_embed -- the "
                "gauge-frame table that generates every transport "
                "operator -- whenever mass_phi=0) "
                "receives NO gradient and stays frozen. Use e_step_gradient='unroll' (the default) to "
                "train these.",
                UserWarning,
                stacklevel=2,
            )
        # A SECOND freeze route: under e_step_gradient='unroll' the closed-form kernel keeps the
        # unrolled E-step signal live to the prior tables, but the autograd ORACLE fallback (served for
        # every NON-kernel family) returns a DETACHED tangent unless oracle_unroll_grad=True,
        # truncating it. The kernel covers exactly gradient_mode=='filtering' AND
        # family=='gaussian_diagonal' AND divergence_family=='renyi' AND renyi_order==1.0 AND
        # include_attention_entropy; anything else routes to the oracle. The decoupled value gauge
        # (pos_rotation='rope' + rope_on_value=False) also forces the oracle route, so this warning
        # must include it or it silently misses a frozen E-step parameter there.
        from vfe3.gradients.kernels import uses_kernel_route
        # route="gradient": this predicate asks whether belief_gradients' GRADIENT route serves the
        # closed-form kernel -- deliberately decoupled from the mm_exact closed-form capability, which a
        # family can register independently. Registering a family's mm_exact fusion therefore cannot
        # change any warning gated on _routes_to_oracle below.
        _routes_to_oracle = not uses_kernel_route(
            renyi_order=self.renyi_order,
            gradient_mode=self.gradient_mode,
            family=self.family,
            divergence_family=self.divergence_family,
            include_attention_entropy=self.include_attention_entropy,
            transport_mode=self.transport_mode,
            decoupled_value_gauge=(self.pos_rotation == "rope" and not self.rope_on_value),
            route="gradient",
        )
        # Hoisted for the unroll-truncation warning just below: mm_exact_update never calls
        # belief_gradients/the oracle at all (its own graph stays live under 'unroll' regardless of
        # oracle_unroll_grad), so that warning's "routes to the DETACHED oracle" claim is a false
        # positive under e_step_update='mm_exact' and must be excluded from the predicate.
        from vfe3.inference.e_step import _E_STEP_UPDATE_ALIASES, canonical_e_step_update as _canon
        _require(self.e_step_update, tuple(sorted(_E_STEP_UPDATE_ALIASES)), "e_step_update")
        canonical_e_step_update = _canon(self.e_step_update)
        if (
            self.e_step_gradient == "unroll"
            and not self.oracle_unroll_grad
            and _routes_to_oracle
            and canonical_e_step_update != "mm_exact"
            and (
                self.pos_phi == "learned"
                # t5_bias's only gradient path is likewise the E-step tangent (audit 2026-07-05 m9);
                # the predicate mirrors the model's parameter-creation gate (model.py: created only
                # when a t5_relative_bias channel is active).
                or (self.t5_learnable_bias
                    and self.beta_attention_prior == "t5_relative_bias")
                # phi_embed was missing from this disjunction too (F13): on the oracle route with
                # oracle_unroll_grad=False the detached tangent severs its only loss path, so with
                # pos_phi='none' plus a full-covariance or entropy-suppressed config NO config-time
                # warning fired at all while phi_embed.grad was None.
                or _phi_frame_is_tangent_only
            )
        ):
            import warnings
            warnings.warn(
                "e_step_gradient='unroll' but this family/gradient_mode routes the belief gradient to "
                "the autograd ORACLE (NOT the closed-form kernel: that covers only "
                "gradient_mode='filtering' + family='gaussian_diagonal' + divergence_family='renyi' + "
                "renyi_order=1.0 + include_attention_entropy=True), which returns a DETACHED tangent while "
                "oracle_unroll_grad=False. A learnable parameter that enters the loss only through the "
                "E-step tangent (prior_bank.phi_embed -- the gauge-frame table generating every "
                "transport operator -- whenever mass_phi=0, "
                "pos_phi_free under pos_phi='learned', "
                "t5_bias under t5_learnable_bias with an active t5_relative_bias channel, "
                "therefore receives NO gradient and stays frozen. Set "
                "oracle_unroll_grad=True to make the oracle return a differentiable (unrolled) gradient.",
                UserWarning,
                stacklevel=2,
            )
        if self.e_mu_q_trust is not None and not (
            math.isfinite(self.e_mu_q_trust) and self.e_mu_q_trust > 0.0
        ):
            raise ValueError(
                f"e_mu_q_trust must be finite and > 0 or None, got {self.e_mu_q_trust}")
        if self.mu_trust_mode not in ("box", "ball"):
            raise ValueError(f"mu_trust_mode must be 'box' or 'ball', got {self.mu_trust_mode!r}")
        if self.e_step_mu_precond not in ("fisher", "raw"):
            raise ValueError(f"e_step_mu_precond must be 'fisher' or 'raw', got {self.e_step_mu_precond!r}")
        for name in ("m_p_mu_lr", "m_p_sigma_lr", "m_phi_lr", "weight_decay", "phi_weight_decay", "min_lr", "min_lr_frac"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        for name in ("m_head_evidence_lr", "m_head_mixer_lr",
                     "m_pos_phi_lr", "m_layernorm_lr"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 or None, got {v}")
        v = self.m_block_mlp_lr
        if v is not None and (
                isinstance(v, bool) or not isinstance(v, Real)
                or not math.isfinite(v) or v < 0.0):
            raise ValueError(f"m_block_mlp_lr must be finite and >= 0 or None, got {v}")
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
        # max_steps / warmup_steps feed range(start_step, n_steps) and the scheduler: a float
        # TypeErrors late in range(), max_steps=0 is an empty training loop, warmup_steps<0 is
        # meaningless. `type(...) is not int` rejects float AND bool; warmup_steps=0 stays legal.
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError(f"max_steps must be an int >= 1, got {self.max_steps!r}")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be an int >= 0, got {self.warmup_steps!r}")
        # Security/behavior-relevant bools must be REAL bools, not truthy coercions:
        # load_checkpoint reads bool(cfg.trust_resume_checkpoint), so the string "False" would coerce to
        # True and enable the unsafe weights_only=False fallback. `type(x) is not bool` (not isinstance)
        # rejects the int/str footguns. Deliberately not swept across every bool field.
        for _bname in ("trust_resume_checkpoint", "generate_embedding_figures"):
            _bval = getattr(self, _bname)
            if type(_bval) is not bool:
                raise ValueError(f"{_bname} must be a bool, got {type(_bval).__name__}: {_bval!r}")
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

        # Registry-driven, like every other *_mode field in this validator: the key set AND the alias
        # collapse both live in vfe3/inference/e_step.py. (e_step_update was validated, and
        # canonical_e_step_update computed, earlier -- ahead of the unroll-truncation warning.)
        if canonical_e_step_update == "mm_exact":
            # The closed-form MM minimizer needs a family that registers the mm_exact CAPABILITY --
            # decoupled from the GRADIENT-kernel capability _routes_to_oracle checks, since a family can
            # register one closed form without the other (e.g. gaussian_full's mm_exact fusion with no
            # analytic gradient kernel). Same structural predicate, route="mm_exact".
            _mm_exact_eligible = uses_kernel_route(
                renyi_order=self.renyi_order,
                gradient_mode=self.gradient_mode,
                family=self.family,
                divergence_family=self.divergence_family,
                include_attention_entropy=self.include_attention_entropy,
                transport_mode=self.transport_mode,
                decoupled_value_gauge=(self.pos_rotation == "rope" and not self.rope_on_value),
                route="mm_exact",
            )
            if not _mm_exact_eligible:
                from vfe3.gradients.kernels import mm_exact_families
                raise ValueError(
                    "e_step_update='mm_exact' needs a family that registers the mm_exact closed-form "
                    "capability under gradient_mode='filtering' + divergence_family='renyi' + "
                    "renyi_order=1.0 + include_attention_entropy=True, belief-independent transport, "
                    f"coupled value gauge; this config does not qualify (registered mm_exact "
                    f"families: {mm_exact_families()}); it routes the belief gradient to the "
                    "autograd oracle (or has no oracle fallback at all for a closed-form minimizer)."
                )
            # gaussian_full's fusion decomposes exactly per gauge-irrep block WHENEVER sigma_p is
            # block-diagonal, which prior_bank._encode_prior_sigma guarantees (torch.diag_embed). The
            # static check that used to rule out a multi-head config here is retired; kernels.py
            # verifies the ACTUAL sigma_p tensor on every call.
            if not (0.0 < self.mm_damping <= 1.0):
                raise ValueError(f"mm_damping must be in (0, 1], got {self.mm_damping}")
            if self.lambda_alpha_mode in ("state_dependent", "state_dependent_per_coord"):
                import warnings
                warnings.warn(
                    f"e_step_update='mm_exact' with lambda_alpha_mode={self.lambda_alpha_mode!r} "
                    "computes the frozen-alpha minimizer of the strict-pair-masked surrogate "
                    "(NOT a majorizer of the canonical objective; see the e_step_update note); the "
                    f"iteration takes a step toward it using mm_damping={self.mm_damping} (a damped "
                    "step for values below 1.0, the full step at 1.0). It is not one-step exact for "
                    "the profiled state-dependent-alpha objective.",
                    UserWarning,
                    stacklevel=2,
                )
        if not math.isfinite(self.unigram_kappa) or self.unigram_kappa < 0.0:
            raise ValueError(
                f"unigram_kappa must be finite and >= 0, got {self.unigram_kappa}")
        if (not math.isfinite(self.exp_fp64_norm_threshold)
                or self.exp_fp64_norm_threshold <= 0.0):
            raise ValueError(
                "exp_fp64_norm_threshold must be finite and > 0, got "
                f"{self.exp_fp64_norm_threshold}"
            )
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

        if self.decode_mode != "linear" and (
            self.skip_belief_sigma_update or self.attention_mode != "transported"
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
                "(linear decode, no norms); projected routes freeze sigma regardless of that flag.",
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
        from vfe3.inference.e_step import canonical_e_step_update
        from vfe3.model.prior_bank import PriorBank, _DECODERS, _decode_family_chunked
        from vfe3.divergence import get_functional, renyi
        from vfe3.families.base import get_family
        from vfe3.families.gaussian import FullGaussian

        inactive: dict[str, str] = {}

        def mark(names: tuple[str, ...], reason: str) -> None:
            inactive.update(dict.fromkeys(names, reason))

        if self.m_phi_update_mode != "pullback_group":
            mark(("m_phi_group_trust_radius",), "only read by m_phi_update_mode='pullback_group'")
        if self.attention_mode == "transported":
            mark(("relational_identity_content", "m_relational_score_lr", "m_relational_content_lr",
                  "relational_routing"),
                 "only read by projected attention; transported attention creates no relational parameters")
        else:
            mark(("compile_pair_kernel",), "projected attention bypasses the belief-gradient pair kernel")
        gradient_controls = ("e_q_mu_lr", "e_q_sigma_lr", "e_step_mu_precond",
                             "e_mu_q_trust", "mu_trust_mode", "e_sigma_q_trust")
        if self.attention_mode != "transported":
            mark(("skip_belief_sigma_update",),
                 "projected routes keep covariance: predictive_content does not estimate it and "
                 "relational filtering returns no sigma gradient")
        if self.attention_mode == "projected_predictive":
            mark(gradient_controls, "predictive_content replaces the mean and keeps covariance")
            if self.mstep_self_coupling_weight == 0.0:
                mark(("lambda_alpha_mode", "lambda_alpha", "b0", "c0"),
                     "predictive_content with mstep_self_coupling_weight=0 has no self-coupling consumer")
        if not self.use_block_mlp:
            mark(("block_mlp_mode", "block_mlp_covariance", "block_mlp_covariance_floor",
                  "block_mlp_expansion", "block_mlp_activation", "block_mlp_dropout", "m_block_mlp_lr"),
                 "only read when use_block_mlp=True")
        if canonical_e_step_update(self.e_step_update) == "mm_exact":
            mark(gradient_controls, "unread under e_step_update='mm_exact'; its step control is mm_damping")
        else:
            mark(("mm_damping",), "only read by e_step_update='mm_exact'")
        if not self.use_priorbank_head_evidence_mixer:
            mark(("m_head_evidence_lr",), "only read when use_priorbank_head_evidence_mixer=True")
        if self.pos_phi != "learned":
            mark(("m_pos_phi_lr",), "only pos_phi='learned' creates the learnable positional table")
        if not (self.layernorm_affine and "layernorm" in (self.norm_type_block, self.norm_type_final)):
            mark(("m_layernorm_lr",), "requires layernorm_affine and a layernorm seam")
        if not self.decode_unigram_prior:
            mark(("unigram_kappa",), "needs decode_unigram_prior=True")
        if self.lambda_alpha_mode == "constant":
            mark(("b0", "c0"), "only read by a state_dependent lambda_alpha_mode")
        if self.pos_rotation == "none":
            mark(("rope_on_value", "rope_base", "rope_full_gauge"), "only read by pos_rotation='rope'")
        if self.pos_phi == "none":
            mark(("pos_phi_project_slk",), "only read by a non-'none' pos_phi")
        if (self.beta_attention_prior in ("windowed", "causal_windowed")
                and self.attention_window >= self.max_seq_len - 1):
            mark(("attention_window",), "window covers the context and is identical to plain causal")
        if self.n_layers == 1:
            mark(("prior_handoff_rho", "prior_handoff_sigma"), "no next layer consumes the handoff")
        if self.e_phi_lr == 0.0 and self.m_phi_update_mode != "pullback_group":
            mark(("phi_precond_mode",), "reached only by e_phi_lr>0 or m_phi_update_mode='pullback_group'")
        registration = _DECODERS[self.decode_mode]
        if not registration.supports_chunked:
            mark(("decode_ce_checkpoint",), "the selected decoder has no checkpointed chunked CE kernel")
        canonical_family_dispatch = (
            self.decode_mode == "family_chunked"
            and registration.callable is _decode_family_chunked
            and registration.fused_ce is PriorBank.decode_ce_family_chunked
            and get_family(self.family) is FullGaussian
            and get_functional(self.divergence_family) is renyi
            and type(self.renyi_order) in (int, float) and self.renyi_order == 1.0
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
            mark(("decode_tau",), "read only by the prior-bank KL decode")
            mark(("untie_decode_bank",),
                 "read only by the KL-to-bank decode; the linear decode is untied by construction")
        else:
            mark(("decode_bias",), "linear decode only; the prior-bank KL decoders create no output bias")
        if self.beta_attention_prior not in ("windowed", "causal_windowed"):
            mark(("attention_window",), "only the windowed/causal_windowed priors read the band half-width")
        from vfe3.families.base import bhattacharyya, jeffreys, squared_hellinger
        if get_functional(self.divergence_family) in (bhattacharyya, jeffreys, squared_hellinger):
            mark(("renyi_order",), "this order-free divergence functional fixes its own Renyi order")
        if self.gauge_transport in ("off", "frozen"):
            mark(("m_pos_phi_lr", "phi_weight_decay"),
                 f"gauge_transport={self.gauge_transport!r} sets requires_grad=False on the frame "
                 "tables, so no phi-role step or decay applies")
        if self.attention_mode != "transported" or self.skip_belief_sigma_update:
            mark(("e_q_sigma_lr", "e_sigma_q_trust", "sigma_max"),
                 "the belief covariance is frozen on projected routes and under "
                 "skip_belief_sigma_update, so no sigma step, trust region, or SPD ceiling runs")
        if self.attention_mode != "transported":
            mark(("full_cov_kl_backend",),
                 "projected scoring never calls the transported full-covariance pair backend")
        if self.attention_mode == "projected_predictive":
            mark(("lambda_beta",), "predictive_content weights no belief-coupling block")
            mark(("full_cov_congruence_precision", "full_cov_congruence_certification",
                  "congruence_cond_escalation"),
                 "predictive_content never transports a covariance by congruence")
        if "e_mu_q_trust" in inactive or self.e_mu_q_trust is None or self.diagonal_covariance:
            mark(("mu_trust_cholesky_rounds",),
                 "only an active mean trust region on a full covariance factors sigma_q")
        # stack._can_omit_prior_handoff at config level: predictive_content never reads the prior
        # and no M-step prior consumer is enabled.
        if (self.attention_mode == "projected_predictive" and self.mstep_self_coupling_weight == 0.0
                and self.cg_energy_weight == 0.0 and self.e_step_update == "gradient"
                and self.e_phi_lr == 0.0):
            mark(("prior_handoff_rho", "prior_handoff_sigma"),
                 "predictive_content never reads the prior and no M-step prior consumer is enabled")
        # The stack enters the E-step with q = p; one gradient iteration in one layer therefore
        # evaluates the self-coupling at D(q||p) = 0, where its force and derivatives vanish.
        if (canonical_e_step_update(self.e_step_update) == "gradient" and self.n_e_steps == 1
                and self.n_layers == 1 and self.mstep_self_coupling_weight == 0.0):
            mark(("lambda_alpha_mode", "lambda_alpha", "b0", "c0"),
                 "the single gradient E-step iteration starts at q = p, where the self-coupling "
                 "force vanishes")
        # Every exponentiated block norm is bounded by its frame's chart norm, so the fp64 island
        # (block norm >= threshold) is unreachable once the threshold exceeds the chart bound.
        if (self.transport_chart_max_norm is not None
                and self.exp_fp64_norm_threshold > self.transport_chart_max_norm):
            mark(("exp_fp64_norm_threshold",),
                 "transport_chart_max_norm rejects a frame before any block norm reaches this "
                 "threshold, so the float64 exponential island is unreachable")
        # Transported gaussian_frame_diagonal is frame-intrinsic: the flat coboundary cancels the
        # whole vertex frame, including the RoPE fold and the positional right factor.
        if self.attention_mode == "transported" and self.family == "gaussian_frame_diagonal":
            mark(("pos_rotation", "rope_base", "rope_on_value", "pos_phi", "pos_phi_scale",
                  "pos_phi_project_slk", "m_pos_phi_lr"),
                 "transported gaussian_frame_diagonal cancels the vertex frame, including the "
                 "RoPE fold and the positional right factor")
        from vfe3.model.block_mlp import get_block_mlp_registration
        if (self.attention_mode == "projected_predictive" and self.family == "gaussian_full"
                and self.encode_mode in ("per_token", "per_token_additive")
                and not self.use_head_mixer and not self.rope_full_gauge
                and not (self.use_block_mlp and (
                    self.block_mlp_covariance != "passthrough"
                    or get_block_mlp_registration(self.block_mlp_mode).gauge_class
                    == "invariant_scalar_intertwiner"))):
            mark(("family",),
                 "value-identical to gaussian_diagonal under the rank-matched decoder on this "
                 "route: stored covariance is a diagonal token lookup and no seam densifies it; "
                 "full covariance requires block_mlp_covariance='delta_full', use_head_mixer=True, "
                 "or attention_mode='transported'")
        return inactive

    @property
    def relational_mode(self) -> str:
        """Internal compatibility view; attention_mode is the only writable selector."""
        return "off" if self.attention_mode == "transported" else "projected_filtering"

    @property
    def relational_readout_mode(self) -> str:
        """Internal executor selection derived from attention_mode."""
        return "predictive_content" if self.attention_mode == "projected_predictive" else "conditional_update"

    @property
    def tau(self) -> float:
        """Per-head softmax-temperature convenience value tau = kappa * sqrt(d_head).

        NOTE: this is a config-level convenience (used for logging). The ACTIVE attention
        temperature is computed group-aware by ``free_energy.attention_tau(kappa, group.irrep_dims)``,
        which keys off the dimension the energy accumulates over -- the gauge-irrep BLOCK: sqrt(K)
        for a single-block group (glk/so_k/sp, irrep_dims=[K]) and sqrt(d_head) for per-head
        multi-block (block_glk). This property equals the active tau only for an equal-block group
        whose block size is d_head (the default block_glk); on a single-block group it understates it
        by sqrt(n_heads). kappa=1 recovers the Vaswani sqrt(d_k) temperature over the energy dimension.
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
    def head_mixer_gauge_compatible(self) -> bool:
        r"""Whether the selected mixer route is an intertwiner for the selected gauge action.

        ``block_glk`` assigns an independent frame to each head, so a trainable cross-head mixer
        is not in that action's commutant. Tied/isotypic mixing is exact for full covariance;
        diagonal dispersion projection is exact only for diagonal gauges. This is metadata only:
        the established opt-in ``block_glk`` baseline remains selectable and the default remains
        mixer-off.
        """
        compatibility = VFE3Config.head_mixer_compatibility.fget(self)
        return compatibility in (
            "disabled",
            "tied_intertwiner",
            "isotypic_intertwiner",
        )

    @property
    def head_mixer_compatibility(self) -> str:
        r"""Truthful label for artifact/config reporting of the selected mixer route."""
        if not self.use_head_mixer:
            return "disabled"
        if self.gauge_group == "block_glk":
            return "independent_head_nonintertwiner"
        from vfe3.divergence import family_cov_kind
        if family_cov_kind(self.family) == "diagonal":
            prefix = "tied" if self.gauge_group == "tied_block_glk" else "isotypic"
            return f"{prefix}_diagonal_projection_nonintertwiner"
        if self.gauge_group == "tied_block_glk":
            return "tied_intertwiner"
        return "isotypic_intertwiner"

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


def serialize_config(config: VFE3Config) -> dict[str, Any]:
    """Serialize a config to its plain JSON-ready mapping of public fields."""
    payload = asdict(config)
    if math.isinf(config.kl_max):
        payload["kl_max"] = "inf"  # Strict JSON has no Infinity number.
    # asdict omits the InitVar. Persist it only when set, so every ordinary config serializes and
    # fingerprints exactly as before while an admitted dense-decode config survives a round trip.
    if getattr(config, "_allow_legacy_dense_decode", False):
        payload["_allow_legacy_dense_decode"] = True
    return payload


def config_from_serialized(
    payload: Mapping[str, Any],

    *,
    source: str,
) -> VFE3Config:
    """Build a config from a serialized mapping, refusing any field this build does not declare.

    Historical per-role clipping and norm precision migrate without changing behavior.
    Retired global clipping and dimension precision are refused. Historical attention pairs
    translate to one canonical selector; conflicting old/new selectors are refused.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"serialized config from {source} must be a mapping")
    payload = dict(payload)
    allow_legacy_dense_decode = _normalize_serialized_boolean(
        payload.pop("_allow_legacy_dense_decode", False), "_allow_legacy_dense_decode",
        source=source)
    if "grad_clip_per_role" in payload:
        per_role = _normalize_serialized_boolean(
            payload.pop("grad_clip_per_role"), "grad_clip_per_role", source=source)
        if not per_role:
            raise ValueError(f"serialized config from {source} requests retired global gradient clipping "
                             "(grad_clip_per_role=False); continuation requires an explicit policy change")
    if "exp_fp64_mode" in payload and payload.pop("exp_fp64_mode") != "norm":
        raise ValueError(f"serialized config from {source} requests a retired exp_fp64_mode; "
                         "only historical 'norm' is accepted without a numerical-policy change")
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
    return VFE3Config(**values, _allow_legacy_dense_decode=allow_legacy_dense_decode)


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
