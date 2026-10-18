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
from typing import Any, Callable, List, Literal, Mapping, Optional, Tuple


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
_DENSE_DECODE_MIGRATIONS   = {"diagonal": "diagonal_chunked", "full": "full_chunked"}
# Retired generic family decode ("family" dense, "family_chunked" fused; 2026-10-12): the validator
# admitted it only for the Gaussian families at Renyi order 1, where it scores the same -KL(q || p_v)/tau
# as the fast kernel of the family's covariance rank (float64 loss differences of 2e-16 and 0).
_FAMILY_DECODE_MIGRATIONS  = {"gaussian_diagonal": "diagonal_chunked", "gaussian_full": "full_chunked"}
# Projected Gaussian-KL factorizations (see VFE3Config.projected_factorization).
_PROJECTED_FACTORIZATIONS  = ("gram", "gram_certified", "cholesky_qr2_certified")
# Gauge-RoPE placements (see VFE3Config.rope_mode): frame space folds R_i into the frame, comparison space
# rotates the projected routing moments after the learned maps.
_FRAME_ROPE_MODES          = ("frame", "frame_value")
_COMPARISON_ROPE_MODES     = ("comparison", "comparison_invariant")
_ROPE_MODES                = ("none",) + _FRAME_ROPE_MODES + _COMPARISON_ROPE_MODES


class _ConfigTypeError(TypeError, ValueError):
    """Configuration type error compatible with legacy ``ValueError`` callers."""


# Messages of the diagonal-comparison-covariance advisory already shown in this process (one warning each).
_DIAGONAL_COMPARISON_WARNED: set = set()


def _warn_diagonal_comparison_covariance(
    head_dims:   List[int],
    fraction:    float,
    base:        float,
    max_seq_len: int,
) -> None:
    r"""Warn once per message that ``comparison_invariant`` makes the router's comparison covariance diagonal.

    With every coordinate of a head's comparison block of size d but at most one in a rotating plane, the
    commutant projection P(G) is one isotropic variance per plane (audit 2026-10-10 round 2, item 48). Plane k
    turns by :math:`(N - 1)\,\mathrm{base}^{-2k/d}` across the largest in-window offset N - 1, and the
    message counts the planes that turn by less than 0.5 rad there.
    """
    slow = []
    for d in sorted(set(head_dims)):
        planes = int(fraction * d // 2)
        count = sum((max_seq_len - 1) * base ** (-2.0 * k / d) < 0.5 for k in range(planes))
        slow.append(f"{count} of {planes} planes for heads of size {d}")
    message = (f"rope_mode='comparison_invariant' with rope_fraction={fraction:g}: every comparison "
               "coordinate but at most one lies in a rotating plane, so the router's comparison covariance P(G) is "
               "diagonal, one isotropic variance per plane, and routing reads no covariance between comparison "
               f"coordinates. At rope_base={base:g} and max_seq_len={max_seq_len}, " + ", ".join(slow)
               + f" rotate by less than 0.5 rad across the largest offset {max_seq_len - 1}, so they carry "
               "almost no relative position.")
    if message not in _DIAGONAL_COMPARISON_WARNED:
        _DIAGONAL_COMPARISON_WARNED.add(message)
        warnings.warn(message, UserWarning, stacklevel=3)


# The optional rates and decays for which None inherits another hyperparameter, child -> parent. This is the
# one place the inheritance rule lives (audit 2026-10-12 C7): build_optimizer, the run artifacts and the inert-field
# rules read it through VFE3Config.resolved_hyperparameter / inherits_hyperparameter. The fields keep None, so a
# saved config and its fingerprint are unchanged. m_per_type_hidden_bias_lr inherits the DECLARED m_phi_lr, which
# __post_init__ writes into the field before gauge_transport can coerce m_phi_lr to 0.
_INHERITED_HYPERPARAMETERS: Mapping[str, str] = {
    "m_decode_mu_lr":            "m_p_mu_lr",
    "m_decode_sigma_lr":         "m_p_sigma_lr",
    "m_block_mlp_lr":            "m_p_mu_lr",
    "m_layernorm_lr":            "m_p_mu_lr",
    "m_relational_score_lr":     "m_p_mu_lr",
    "m_relational_content_lr":   "m_p_mu_lr",
    "m_pos_phi_lr":              "m_phi_lr",
    "m_per_type_hidden_bias_lr": "m_phi_lr",
    "mu_weight_decay":           "weight_decay",
    "sigma_weight_decay":        "weight_decay",
    "decode_mu_weight_decay":    "mu_weight_decay",
}


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
    grad_clip:                        Optional[float] = 1.0          # L2 clip per grad_clip_scope set after accumulation; None or 0 = off
    # "role" | "group"
    # The parameter sets grad_clip bounds, each by its own L2 norm (groups marked table_fisher are never clipped).
    #   role   one set per optimizer role (mu / sigma / phi), the historical behavior
    #   group  one set per optimizer group, so one exploding group cannot set the coefficient of the others
    grad_clip_scope:                  str             = "role"
    # Failed-step recovery (audit 2026-10-17 item 16). None keeps the historical abort: a step whose strictest
    # precision rung raises before any optimizer update ends the run. N >= 1: up to N such steps over the run are
    # skipped instead (gradients zeroed, no optimizer or scheduler step, the batch saved under
    # diagnostics/failed_steps/, counted in metrics.csv failed_step_skips), and a failed evaluation reports NaN
    # validation with no best-model update; the (N+1)-th failure aborts as before.
    max_failed_step_skips:            Optional[int]   = None
    # Abort after this many consecutive rejected updates of any kind (non-finite, failed, loss spike) through the
    # crash path (diagnostics/crash.json and a resumable checkpoint). None = never abort on consecutive skips.
    max_consecutive_skips:            Optional[int]   = None
    # Reject the update of a step whose training loss exceeds the divergence monitor's loss EMA by more than this
    # many nats (after its 50-step warmup; audit 2026-10-17 item 17), read from the host loss the step already
    # transfers. Rejections are counted in metrics.csv loss_spike_rejected_updates. None = off.
    loss_spike_skip_nats:             Optional[float] = None

    seed:                             int             = 6
    
    deterministic:                    bool            = True         # torch/cuDNN determinism (TF32: see fp32_matmul_precision)
    
    ##################################################
    #   2. Logging, evaluation and checkpoints
    ##################################################

    log_interval:                     int             = 100          # console log every N steps (0 = off)
    eval_interval:                    int             = 1500         # validation every N steps (0 = off)
    checkpoint_interval:              int             = 15000        # resumable checkpoint every N steps (0 = off)
    checkpoint_keep_last:             Optional[int]   = None         # keep the newest N step checkpoints; None = keep all
    checkpoint_at_end:                bool            = False        # also write a resumable checkpoint after the last step
    block_variance_at_log:            bool            = False        # also write diagnostics/block_variance.csv at log steps
    val_diagnostic_batches:           int             = 1             # validation batches the val_* probes average over
    # Log-step diagnostic: the canonical free energy at each layer's captured beliefs,
    # sum_i [alpha_i KL(q_i||p_i) + R(alpha_i)] + lambda_beta sum_h sum_ij [beta*_ij D^h_ij + tau beta*_ij log(beta*_ij/pi_ij)]
    # with D^h_ij the head-marginal KL of q_i and Omega_ij q_j and the canonical router beta* = softmax(log pi - D/tau).
    # beta* includes the attention prior pi (causal mask, ALiBi), the RoPE rotation of either rope_mode space (comparison:
    # each head's reference moments rotated by R_i, full rotated covariance) and the router's temperature tau (1 under
    # minimal_agent); D_ii = 0 = B_ii puts its row maximum on the diagonal. Fills self_coupling, self_divergence,
    # belief_coupling, attention_entropy and total (per-token means), canonical_beta_tv, canonical_self_weight
    # (mean beta*_ii) and, at each layer's converged output, canonical_total_output and canonical_total_delta
    # (F_out - F_in). O(B N K^3 + B N^2 sum_h d_h^2) per diagnostic window (O(B N H d^3) on head-block groups)
    diagnose_canonical_free_energy:   bool            = False
    # Eval-cadence diagnostic (audit 2026-10-12 T1): per layer, the evidence lower bound of the minimal agent's
    # posterior (predictive_readout='minimal_agent') and of each prototype_agents block, the evidence of the declared
    # record model (ADF over heads for the minimal agent, exact for the prototypes) and their gap, in nats per token.
    # The minimal agent also logs the exact evidence (NaN unless its heads are uncoupled) and a 64-sample
    # importance-sampled evidence under q with its gap (audit 2026-10-12 round 2, item 12 and M1), and the effective
    # sample size fraction (sum w)^2/(n sum w^2) of its weights (audit 2026-10-16, F47).
    # Logged as val_relational_minimal_agent_{elbo,adf_evidence,adf_elbo_gap,evidence,is_evidence,is_elbo_gap,is_ess}
    # (last layer) and in
    # diagnostics/minimal_agent_elbo.csv and diagnostics/prototype_elbo.csv; computed without autograd, never in the
    # loss (the evidence is unbounded above as the maps and the noise shrink). The mean field also runs the exact
    # path once per diagnostic forward for the evidence.
    diagnose_agent_elbo:              bool            = False

    eval_stride:                     Optional[int]   = None         # held-out window stride <= max_seq_len; None = disjoint windows
               
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

    # "gaussian_diagonal" | "gaussian_full" | "gaussian_block"
    # The belief covariance structure (read diagonal_covariance, never a second flag).
    #   gaussian_block  one SPD d_head x d_head block per head, stored in the dense (K, K) layout. only:
    #                   projected_predictive, predictive_readout="kernel", equal head blocks (block_glk, any
    #                   block_* group or an equal-block so_n / su_n tower; any tying), content_link_edges
    #                   "off", decode_mode "linear" | "diagonal_chunked" | "full_chunked"; a block MLP covariance
    #                   is read through its head-block projection (posterior needs block_mlp_inference="mean_field")
    family:                           str             = "gaussian_diagonal"

    # "diagonal" | "block_cholesky"
    #   diagonal        per-type log variances (V, K)
    #   block_cholesky  frame-free per-type head blocks C_v = L_v L_v^T + eps I, L_v = diag(exp(l_v/2))(I + N_v),
    #                   stored (V, H, d(d+1)/2): l_v (log-variance units) on the diagonal slots, N_v below;
    #                   with mu_embed as xi_v it is the learned-frame reference arm without frames.
    #                   only: family "gaussian_block" | "gaussian_full", gauge_transport="off", block_glk,
    #                   projected_predictive, content_link_edges="off", untie_decode_bank=True or a linear decode
    prior_covariance_table:           str             = "diagonal"

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

    # "linear" | "diagonal_chunked" | "full_chunked" | "expected_likelihood_chunked"
    #   linear                       learned (V, K) readout of mu_q (+ bias); ignores sigma
    #   diagonal_chunked             -KL(q || p_v)/tau against the diagonal prior bank
    #   full_chunked                 -KL(q || p_v)/tau with a full q against the diagonal bank
    #   expected_likelihood_chunked  log N(mu_q; mu_v, Sigma_q + Sigma_v), diagonal only
    # Every prior-bank mode streams the vocabulary in chunks and never forms (B, N, V) in training.
    decode_mode:                      str             = "linear"

    # "local" | "reference"
    #   local      decode the local moments (mu_i, Sigma_i)
    #   reference  decode xi_i = U_i^-1 mu_i, C_i = U_i^-1 Sigma_i U_i^-T through the attention frame.
    #              only: a projected attention_mode, rope_mode "none" or a comparison mode, decode_mode "linear" |
    #              "diagonal_chunked" | "full_chunked"
    decode_coordinates:               str             = "local"

    decode_unigram_prior:             bool            = True         # add unigram_kappa * log pi_v (corpus unigram) to the logits
    # decode_mode="linear" only: train on the second-order (delta-method) expected log-loss
    # E_q[-log p_y] ~ CE(mu) + 1/2 tr(Sigma Cov_p(w)) instead of the plug-in CE(mu), at one extra (V, 2K)
    # GEMM per vocab chunk. Training mode only; evaluation keeps the plug-in predictive CE. Requires
    # family="gaussian_diagonal" and decode_coordinates="local". Its sigma gradient 1/2 Var_p(w_k) >= 0
    # pushes Sigma toward its floor: run it as an ablation arm first.
    decode_second_order:              bool            = False
    untie_decode_bank:                bool            = True         # prior-bank only: decode reads its own copy of the (V, K) tables

    decode_tau:                       float           = 1.0          # prior-bank decode temperature (1 = Bayes); not read by linear
    unigram_kappa:                    float           = 1.0          # tempering of log pi_v (1 = exact Bayes class prior)

    # "belief" | "encoded" | "constant": the covariance the prior-bank decoder reads (not read by linear)
    #   belief    the stack's final Sigma_q (the exact KL(q* || p_v))
    #   encoded   the token's encoded covariance, the stack's input (passthrough for the decoder only)
    #   constant  sigma_init I, in the decoder's coordinates: under decode_coordinates="reference" the decoder reads
    #             sigma_init I in reference coordinates, so the logits are frame-invariant (audit 2026-10-17, item 25)
    # The decode logit -KL(q || p_v)/tau contains tr(Sigma_v^-1 Sigma_q)/(2 tau), so under a moment-matched
    # covariance every jump of the Sigma_q the stack builds reaches the cross-entropy (2026-10-11 K=128 spikes).
    decode_covariance:                str             = "belief"
    
    # Vocabulary entries per chunk of the fused CE kernels. Values do not depend on the width.
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
    # sigma_v falls far below 1; watch the decode_sigma_v_min column of metrics.csv (min decode prior variance of the
    # table the decoder scores against; NaN under decode_mode="linear").
    decode_av_precision:              str             = "fp32"

    # Prior-bank Gaussian decoders only: recompute an uncertain float32 decode argmax in float64 for the
    # candidates whose score interval reaches the winner's (the fused training CE is unaffected).
    decode_ranking_fp64_escalation:   bool            = False

    # decode_mode="diagonal_chunked" | "full_chunked" on CUDA tensors only: run each vocabulary chunk's score assembly
    # (bias, clamp, temperature, unigram bias), online log-sum-exp, target gather and their backward in fused Triton
    # kernels (vfe3/kernels/kl_decode.py); the cross-term GEMMs stay cuBLAS. Same formula and logits, rounding-level
    # differences in the log-sum-exp and the backward sums. CPU tensors always run the torch loop.
    decode_ce_triton:                 bool            = False

    ##################################################
    #   6. Gauge group
    ##################################################

    # "block_glk" | "so_n" | "su_n" | "block_so" | "block_co" | "block_sl" | "block_borel" | "block_unipotent"
    # | "block_glmc"
    #   block_glk   GL(d_head) per head
    #   so_n        SO(group_n) acting through the irrep blocks of irrep_spec (one block per head)
    #   su_n        SU(group_n) acting through the irrep blocks of irrep_spec (one block per head)
    #   block_so    SO(d_head) per head (rotations)
    #   block_co    CO(d_head) = R+ x SO(d_head) per head (rotation and a positive gain)
    #   block_sl    SL(d_head) per head (det = 1, so pos_phi_project_slk is the identity)
    #   block_borel B+(d_head) per head: upper triangular, positive diagonal (no rotations: RoPE leaves it)
    #   block_unipotent  N+(d_head) per head: upper triangular, unit diagonal (det = 1; no rotations)
    #   block_glmc  GL(m, C) per head realified on d_head = 2m (even); complex coordinate k = pair (2k, 2k+1)
    gauge_group:                      str             = "block_glk"

    # None | "per_head" | "shared"
    #   None      the group's native tying (block_glk "per_head", so_n and su_n "shared")
    #   per_head  an independent group element per head
    #   shared    one element acting on every head. only: block_glk, so_n, su_n. block_glk "shared" refuses
    #             fisher_group and the group_log retraction (pullback_group runs with the chart retraction);
    #             so_n and su_n "per_head" refuse use_cg_coupling
    gauge_tying:                      Optional[str]   = None

    # "on" | "off" | "frozen"
    #   on      the frame is learned as phi_scale / m_phi_lr / pos_phi specify (pure path)
    #   off     Omega = I exactly: coerces phi_scale=0, pos_phi="none", m_phi_lr=0; needs rope_mode="none"
    #           or a comparison mode (which rotates the routing moments, never the frame)
    #   frozen  a random fixed frame: keeps phi_scale, coerces m_phi_lr=0 (same parameter count as "on")
    gauge_transport:                  str             = "on"

    # so_n and su_n only (None for every other group). group_n is the N of SO(N) (int >= 2) or SU(N)
    # (int >= 3), decoupled from K. irrep_spec: [(label, mult), ...] | "random" (so_n only).
    # so_n labels: "l<p>" symmetric-traceless rank p (N=3: spin p, dim 2p+1), "w<p>" exterior power,
    # "wp<m>" / "wm<m>" self-dual / anti-self-dual halves at m = N/2 even.
    # su_n labels: "pq<p>_<q>" (p >= q >= 0, p + q <= 6): highest weight p w_1 + q w_{N-1}, which covers every
    # SU(3) irrep up to conjugation but for N >= 4 only those with vanishing inner Dynkin labels (not Lambda^2 C^4).
    # Real dimension D for p = q (real type) and 2D for p != q (complex type, realified),
    # D = C(p+N-1, p) C(q+N-1, q) - C(p+N-2, p-1) C(q+N-2, q-1); e.g. SU(3): pq1_0 6, pq1_1 8 (adjoint),
    # pq2_0 12, pq2_2 27. pq1_1-only towers are PSU(N).
    # Blocks are laid out in order, one per head, and their dims must sum to embed_dim.
    # "random" draws exactly n_heads irreps summing to embed_dim (ranks in [irrep_min_rank,
    # irrep_max_rank], None = all that fit) with irrep_spec_seed, and the saved config keeps the drawn list.
    # JSON list-of-lists pairs are accepted and coerced to tuples.
    group_n:                            Optional[int]   = None
    irrep_spec: 'Optional[List[Tuple[str, int]] | str]' = None 
    irrep_max_rank:                     Optional[int]   = None
    
    irrep_spec_seed:                    int             = 0
    irrep_min_rank:                     int             = 1            # 1 skips the gauge-invariant trivial irrep l0
    
    # so_n or su_n with gauge_tying="shared" only: Clebsch-Gordan coupling between irrep blocks on the means
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
    #            the frame before the learned maps, so routing is not relative (relative: a comparison rope_mode)
    pos_phi:                          str             = "learned"
    # "none" | "frame" | "frame_value" | "comparison" | "comparison_invariant"
    # Gauge-RoPE: the block-diagonal positional rotation R(theta_i).
    #   none                  no rotation
    #   frame                 folded into each routing frame on the right, V_i = U_i R_i^T, before the learned maps
    #                         (absolute-position routing); the content frame stays U_i (routing only)
    #   frame_value           frame, and the content frame is rotated too
    #   comparison            applied to each head's projected query and key moments after the learned maps, so the
    #                         routing cost depends on positions only through i - j (kernel readout: routing only;
    #                         minimal_agent: the record channel R_{j-i}, routing and content by exact conditioning).
    #                         The kernel router scores the comparison covariance R_i G R_i^T (as run): the trace and
    #                         log-det terms of every non-self pair grow with the anisotropy of G, which can self-lock a
    #                         head (audit 2026-10-09 round 2, addendum). only: projected_predictive
    #   comparison_invariant  comparison, with G projected onto the commutant of the RoPE torus (one isotropic variance
    #                         per rotating plane, the full non-rotating block, no cross-covariance), so
    #                         R_i P(G) R_i^T = P(G) and only the mean term carries i - j; scored on the Gram route (no
    #                         source-root factor route), or in closed form with no factorization when P(G) is diagonal
    #                         (rope_fraction = 1) and projected_factorization="gram"
    #                         (vfe3.gradients.relational._diagonal_comparison_route). only: the kernel readout (not
    #                         minimal_agent)
    rope_mode:                        str             = "none"
    pos_phi_project_slk:              bool            = False        # pos_phi only: per-block trace projection (det = 1)
    
    rope_base:                        float           = 100.0        # rope_mode != "none" only: frequency base
    # rope_mode != "none" only: p-RoPE fraction p in (0, 1] (Barbero et al., ICLR 2025): only the floor(p d / 2) fastest
    # pairs of each block rotate, at their full-RoPE frequencies; the slower pairs stay position-free. 1.0 = RoPE.
    rope_fraction:                    float           = 1.0
    
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
    # Learn the routing temperature per layer and head: tau_{l,h} = kappa_h sqrt(d_h) exp(3 tanh(delta_{l,h} / 3))
    # with delta = 0 at init (routing_log_temperature, (H,) per layer; no weight decay, score learning rate).
    # The KL cost is invariant under (S_Q, S_K) -> (A S_Q, A S_K), so no map can set the sharpness. Cools only
    # below the kl_max cap: a fully capped absolute-reference row stays at the prior pi. Not with minimal_agent.
    learn_attention_temperature:      bool            = False

    # "causal" | "causal_noself" | "causal_alibi" | "causal_alibi_noself" | "causal_windowed"
    # Causal log-prior over sources.
    #   *_noself           masks the self key
    #   causal_alibi*      adds -slope_h (i - j) (Press et al.)
    #   causal_windowed    -inf more than attention_window keys back
    beta_attention_prior:             str             = "causal"
    
    # alibi priors only: BASE multiplier on the Press et al. (ICLR 2022) head slopes: alibi_slope * 2^(-8h/H),
    # h = 1..H, for a power-of-two H; otherwise the 2^(-8h/H') schedule of the largest power of two H' < H, then the
    # first H - H' slopes 2^(-8h/(2H')) at odd h (H = 5: 1/4, 1/16, 1/64, 1/256, 1/2; vfe3.attention_prior). An
    # n_heads sweep at fixed alibi_slope rescales the prior.
    alibi_slope:                      float           = 1.0
    attention_window:                 int             = 128          # causal_windowed only: band width

    # "gaussian_kl" | "noisy_query" | "noisy_key" | "mean_distance" (noisy_key: see relational_key_noise below)
    # projected_predictive with predictive_readout="kernel" only: routing cost.
    #   gaussian_kl    KL between the projected query and key Gaussians
    #   noisy_query    the query mean treated as a noisy measurement (GL(K) v2, Remark 9)
    #   mean_distance  ||a_i - b_j||^2 / 2 of the projected means: the covariance-blind control
    relational_routing:               str             = "gaussian_kl"
    # projected_predictive kernel only: tie the key comparison map to the query map, S_K := S_Q (no raw_score_k),
    # so D_ii = 0 and each head scores D^h_ij = KL((S_h)_* r_i || (S_h)_* r_j) between the reference beliefs pushed
    # forward by the one rank-d map S_h, which is at most KL(q_i || (Omega_ij)_* q_j) and equals it only when S_h
    # is sufficient for the pair (GL(K) v2, flat alignment).
    tie_comparison_maps:              bool            = False
    # relational_routing="noisy_key" only (audit 2026-10-12, T6): the initial per-head key noise n_h > 0 (learned as
    # routing_key_log_noise, (H,) per layer). The router scores D_ij = KL(N(a_i, V_Q,i) || N(b_j, V_K,j + n_h I)) at
    # temperature exactly 1 (kappa_beta is not read; learn_attention_temperature is refused), so the routing
    # objective is the KL of the declared joint pi_ij N(y; b_j, V_K,j + n_h I) (GL(K) v2, Eq. mixture_free_energy).
    relational_key_noise:             float           = 1.0

    ##################################################
    #   9. Predictive readout (projected_predictive)
    ##################################################

    # "kernel" | "minimal_agent"
    #   kernel         content U_i sum_h O'_h sum_j beta_ij^h T_h xi_j
    #   minimal_agent  each receiver conditions exactly on a noisy comparison record with the key tied to
    #                  the content (S_K = T); no O, routing is the exact source posterior (temperature 1;
    #                  relational_routing, kappa_beta and kl_max are not read). Needs equal heads,
    #                  content_link_edges="off" and not rope_mode="frame";
    #                  comparison-space RoPE is the known record channel R_{j-i} (exact conditioning).
    predictive_readout:               str             = "kernel"

    # "exact" | "mean_field"
    # minimal_agent only.
    #   exact       each head's exact posterior; heads combined by assumed-density filtering
    #   mean_field  KL labels at the input moments and an information-form fusion (no between-source spread)
    minimal_agent_inference:          str             = "mean_field"
    minimal_agent_cavi_steps:         int             = 1            # mean_field only: label/Gaussian coordinate-ascent sweeps >= 1 (1 = labels at the input moments)
    
    # "passthrough" | "moment_matched" | "spread_corrected"
    #   passthrough       keep the incoming covariance
    #   moment_matched    the readout's covariance plus floor * Sigma_p: kernel, the exact moment-matched covariance of
    #                     the predictive kernel; minimal_agent exact, the exact posterior's; minimal_agent mean_field,
    #                     the CAVI posterior (C^-1 + sum_h S_h^T Lambda_h S_h)^-1 without the between-source spread, so
    #                     with equal source covariances it does not depend on the context (audit 2026-10-12, item 9)
    #   spread_corrected  minimal_agent mean_field only: P + sum_h P S_h^T R'_h S_h P plus floor * Sigma_p, the
    #                     linearized label spread R'_h = sum_j beta_j d_j d_j^T (audit 2026-10-12 T2)
    predictive_covariance:            str             = "moment_matched"
    
    predictive_covariance_floor:      float           = 1e-4         # moment_matched only (kernel and minimal_agent readouts): floor * Sigma_p, the token's local prior covariance
    # "prior" | "relative"
    # The moment-matched covariance floor (scaled by predictive_covariance_floor = f).
    #   prior     f Sigma_p, the token's local prior covariance: frame-free and exactly reference-invariant (since 2026-10-12)
    #   relative  f (tr C_i / K) U_i U_i^T with C_i = U_i^{-1} Sigma_i U_i^{-T}: the earlier floor, relative to the input
    #             covariance; changes under a non-orthogonal common reference change (kept for testing). It is
    #             covariant under the local action U_i -> g_i U_i (tr C_i is invariant). Its bound is relative only: in
    #             reference coordinates it adds f (tr C_i / K) I >= f lambda_max(C_i) / K, so it bounds the output
    #             spectrum below by f/K times the input's own scale and has no absolute lower bound; a contraction of
    #             tr C_i in one layer passes into the next layer's floor, so the bound compounds over depth (a
    #             per-layer trace ratio r gives r^L, audit 2026-10-17 item 1). In local coordinates the floor's top
    #             eigenvalue reaches up to f (tr Sigma_i / K) kappa(U_i)^2 for isotropic Sigma_i, an inflation of up to
    #             about f kappa^2 relative to the scale of Sigma_i on ill-conditioned frames (kappa(U_i) the frame
    #             condition number)
    #   max       the larger of the two: elementwise max on the diagonal family, the whitened spectral max
    #             f Sigma_p^{1/2} max(W, I) Sigma_p^{1/2} (W the relative floor whitened by f Sigma_p) on full layouts;
    #             keeps the relative floor's scaling with the absolute bound f Sigma_p (audit 2026-10-17 item 1)
    predictive_covariance_floor_mode: str             = "prior"
    # moment_matched / spread_corrected only: Loewner ceiling Sigma^+ <= c * Sigma_p (vfe3.model.covariance_ceiling),
    # the counterpart of the floor; a clamp where it binds, counted in metrics.csv covariance_ceiling_elements. None = off.
    predictive_covariance_ceiling:    Optional[float] = None
    minimal_agent_channel_noise:      float           = 1.0          # minimal_agent only: initial channel noise n_h > 0 (learned)
   
    # moment_matched only: recompute the covariance update in backward instead of storing its
    # intermediates (identical values and gradients; saves memory at large K).
    predictive_covariance_checkpoint: bool            = False
    # moment_matched, kernel readout only: recompute the failing sequences of a float32 update in float64 when its
    # output is not positive definite (diag(U C^att U^T) amplifies float32 rounding by about kappa(U)^2), counted in covariance_escalation_elements;
    # one host read per layer call. A float64 failure raises.
    predictive_covariance_fp64_escalation: bool       = False
    # "fp32" | "fp64"
    # moment_matched, kernel readout only: compute precision of the covariance update.
    #   fp32  the beliefs' precision
    #   fp64  the whole update (C_i, pooling, sandwich, floor, ceiling) in float64 from float32 beliefs, cast back:
    #         removes the kappa(U)^2 rounding the sign-only escalation misses (audit 2026-10-16 F7); costly
    predictive_covariance_precision:  str             = "fp32"
    # moment_matched, kernel readout only (audit 2026-10-17 item 12): add the routed spread after the sandwich as the
    # nonnegative sum of squares sum_h sum_j beta_ij^h (U_i O'_h (v_j^h - m_i^h))^2 (outer products on full layouts)
    # instead of through the head-pair pooled covariance, where diag(U C^att U^T) cancels it; equal in exact
    # arithmetic, about N K^2 multiply-adds per token. Not with content_link_edges.
    predictive_covariance_spread_squares: bool        = False
    # moment_matched, kernel readout only (audit 2026-10-17 item 19): form the head-pair source and pooled
    # covariances for h <= g only and double the off-diagonal pairs (diag(U Y U^T) = diag(U Y^T U^T)); equal up to
    # rounding, about H(H-1)/2 fewer pair contractions. Blockwise route (block frames, equal heads), not with
    # content_link_edges.
    predictive_covariance_symmetric_pairs: bool       = False
    attention_residual:               bool            = True      # kernel only: new mean = mu_i + content (False: content only)

    ##################################################
    #   10. Content links (O, T)
    ##################################################

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
    #   pure_gauge  coboundary control: G_ij = g_i e^L g_j^-1, g_i = exp(A B_i^T); not flat unless e^{3L} = I
    #               (the triangle holonomy G_ij G_jk G_ki = g_i e^{3L} g_i^-1 conjugates the shared link)
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

    # The receiver law is the channel likelihood: the expected likelihood of the measurement P_h xi_i
    # against the d-dim message T_h xi_j, with the receiver entropy counted once.

    # "inverse" | "adjoint"
    # The receiving measurement P_h xi_i of xi_i on head h.
    #   inverse  (default, the theoretically consistent choice) the head-h rows of O'^-1, the dual
    #            measurement: invariant under (U h, T h, S h, h^-1 O'), and the frozen-context minimizer
    #            is the kernel mean as alpha -> 0 with equal channel precisions
    #   adjoint  O'_h^T xi_i, equal to the inverse only for orthogonal O'; kept for charts too
    #            ill-conditioned for the inverse's float64 certificate
    relational_receiver_measurement:  str             = "inverse"

    # "closed_form" | "closed_form_profiled" | "euler"
    #   closed_form           exact frozen-context natural-parameter update, damped by relational_receiver_damping
    #   closed_form_profiled  under lambda_alpha_mode="state_dependent": the limit of iterated closed_form (MM)
    #                         steps, a stationary point of c0 log(b0 + D) + G, in one scalar root solve per
    #                         receiver (not per_coord)
    #   euler                 explicit Fisher step with e_q_mu_lr / e_q_sigma_lr and the trust regions
    relational_receiver_update:       str             = "closed_form"
    # closed_form(_profiled) only: rho in (0, 1]; rho = 1 takes the full step. Under closed_form that is the
    # minimizer with alpha frozen at the current iterate, one MM step, so extra E-steps keep moving the belief;
    # under closed_form_profiled it is the profiled fixed point, which extra E-steps leave unchanged.
    relational_receiver_damping:      float           = 1.0
    relational_receiver_noise:        float           = 1.0          # channel floor nu I_d, nu > 0

    ##################################################
    #   12. Free-energy weights
    ##################################################

    # "state_dependent" | "state_dependent_per_coord"
    # Self-coupling alpha_i on KL(q_i || p_i).
    #   state_dependent            alpha_i = c0 / (b0 + D(q_i || p_i)), frame-invariant
    #   state_dependent_per_coord  the same per coordinate (b0, c0 may be K-lists)
    lambda_alpha_mode:                str                   = "state_dependent_per_coord"
    b0:                               'float | List[float]' = 1.0
    c0:                               'float | List[float]' = 1.0

    lambda_beta:                      float                 = 1.0          # weight of the whole belief-coupling block (1 = pure F)
    
    mstep_self_coupling_weight:       float                 = 0.0          # M-step sum_i alpha_i D(q_i* || p_i); 0 = off
    # None | float > 0 (inf allowed): the divergence clamp of the M-step self-coupling term D(q_i* || p_i); None uses
    # kl_max, the router's clamp, under which the term saturates (zero gradient) in the collapse regime (audit
    # 2026-10-17 item 33)
    self_kl_max:                      Optional[float]       = None

    ##################################################
    #   13. Normalization and block handoff
    ##################################################

    # "none" | "layernorm" | "mahalanobis"
    #   none         identity (gauge-pure)
    #   layernorm    standard LayerNorm over the belief dim (not gauge-equivariant)
    #   mahalanobis  gauge-invariant rescale
    # norm_type_block runs inside every block; no norm runs between the stack and the decoder.
    norm_type_block:                  str             = "layernorm"
    layernorm_affine:                 bool            = False        # layernorm only: learned gamma/beta (init 1/0)

    # "passthrough" | "pushforward" | "pushforward_moment"
    # Covariance through the LayerNorm block norm (norm_type_block="layernorm").
    #   passthrough  the norms map the mean only; Sigma stays in pre-norm units
    #   pushforward  J_LN Sigma J_LN^T + block_mlp_covariance_floor * D Sigma D + rho g g^T, D = diag(gamma)/s, with
    #                rho g g^T the radial second-order variance; its diagonal on the diagonal family (O(K) per
    #                token). In reference coordinates (gaussian_full / gaussian_block only) it acts on C = U^-1 Sigma U^-T.
    #                At the block norm inert under block_mlp_covariance="delta_full", whose Jacobian composes the norm's
    #   pushforward_moment  the same expression at the effective scale s_eff^2 = s^2 + tr(Pi Sigma Pi)/K (the input's
    #                expected LayerNorm variance) plus a span floor f (T/K) D W W^T D on span{1, z}: a moment estimate
    #                of the LayerNorm output at every Sigma/s^2, where 'pushforward' (the delta method at s, capped at
    #                the sphere bound) keeps the delta method's radial shape (audit 2026-10-16 F3, F2)
    block_norm_covariance:            str             = "passthrough"
    # Scale the LayerNorm covariance pushforward (both pushforward modes, and the delta_full composite's norm stage)
    # by min(1, K max(gamma^2) / tr) at the sphere bound (audit 2026-10-12 item 1). False skips the cap: the
    # uncapped delta method, which can exceed the bound by orders of magnitude once Sigma is large against s^2.
    block_norm_trace_cap:             bool            = True
    # block_norm_covariance='pushforward' with the trace cap only (audit 2026-10-17 item 8): on the tokens where the
    # cap binds return the moment estimate of 'pushforward_moment' instead of the capped delta method, whose shape
    # is off by up to about 10^3 per coordinate on anisotropic input. False keeps the capped delta method everywhere.
    block_norm_cap_moment_fallback:   bool            = False

    # "local" | "reference"
    #   local      the block norm and block MLP act on mu_i
    #   reference  they act on xi_i = U_i^-1 mu_i and map back, mu -> U_i Phi(U_i^-1 mu). only:
    #              projected_predictive, rope_mode "none" or a comparison mode, a "none"/"layernorm" norm, and a passthrough
    #              coordinate / gaussian_template MLP, a prototype_agents MLP, or (gaussian_full / gaussian_block)
    #              a delta_full coordinate MLP
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
    #   delta_full   J Sigma J^T + block_mlp_covariance_floor * Sigma (coordinate, gauge_gate, canonical_frame);
    #                its diagonal on gaussian_diagonal (coordinate, canonical_frame)
    #   posterior    the template posterior covariance (prototype_agents)
    block_mlp_covariance:             str             = "passthrough"
   
    # "gelu" | "silu" | "relu"
    # Hidden activation; not read by gaussian_template or prototype_agents.
    block_mlp_activation:             str             = "gelu"

    # "mean_field" | "exact"
    # prototype_agents only: template posterior.
    #   mean_field  KL-label responsibilities and the power posterior
    #   exact       exact mixture posterior (dense or block covariance needs a Cholesky per token/template)
    block_mlp_inference:              str             = "mean_field"
    block_mlp_cavi_steps:             int             = 1            # mean_field only: label/Gaussian coordinate-ascent sweeps >= 1 (1 = labels at the input moments)
    block_mlp_em_weight:              float           = 0.0          # prototype_agents only: weight of the EM M-step term on (nu, lambda, pi); 0 = off
    
    # Per-type (V, n_gen) table added to the first hidden pre-activations of each layer's MLP (zero init).
    # only: projected attention, use_block_mlp with a coordinate or canonical_frame MLP.
    per_type_hidden_bias:             bool            = False        # TODO ablate then remove
    
    block_mlp_expansion:              int             = 8            # hidden width (or template count) = expansion * K
    block_mlp_dropout:                float           = 0.0
    block_mlp_covariance_floor:       float           = 1e-4         # delta_full and a block_norm_covariance pushforward
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
    #   chart      additive chart step with product-residual backtracking (fixed 1e-6 gate; it caps the step at
    #              trained norms, see m_phi_group_trust_radius)
    #   group_log  exact right product Log(exp(phi) exp(-eta s xi)) (untied block_glk)
    m_phi_group_retraction:           str             = "chart"

    # "series" | "eigenbasis"
    # pullback_group with group_log only: direction of the rows the closed form xi = U^{-1} G_bar does not serve
    # (a block off the principal strip, max_h kappa_2(U_h) > 1e3, or a Frechet-log failure).
    #   series      the certified strict Frobenius-pullback direction; it builds the dense structure constants,
    #               whose budget refuses block_glk from d_head = 17 on
    #   eigenbasis  off-strip blocks invert Psi_L(ad_{X^T}) in the eigenbasis of the chart block (exact where Psi is
    #               invertible); rows with max_h kappa_2(U_h) > 1e3 or a failed eigenbasis gate take no step. No
    #               structure constants, so no head-width budget. The gate (shared with fisher_group) is tight at
    #               d_head = 32, where it rejected every generic off-strip block of a probe (README)
    m_phi_group_log_fallback:         str             = "series"

    # "series" | "spectral"
    # pullback_group only: evaluation of the trivialized exponential differentials Psi_R / Psi_L in the strict
    # Frobenius-pullback direction (vfe3/geometry/phi_preconditioner.py; under group_log only on the rows off the
    # closed form). Checkpointed with the optimizer state; a resumed run keeps the saved route.
    #   series    truncated power series with a geometric-majorant tail certificate (orders 8 to 128)
    #   spectral  exact Daleckii-Krein closed form from one eigendecomposition of the chart block; a call whose
    #             eigenbasis fails the conditioning or imaginary-residue guard uses the series instead
    m_phi_group_differential_mode:    str             = "series"

    # Interior radius for phi after each M-step, < transport_chart_max_norm (AdamW projects rows back;
    # pullback_group rejects the step). None: nothing bounds ||phi|| under adamw.
    phi_mstep_max_matrix_norm:        Optional[float] = None

    # Condition-number ceiling kappa_max > 1 on every composed frame head block U_i = e^X e^Y, applied at
    # construction, on checkpoint load and after each accepted adamw step (audit 2026-10-12 T4; round 2, M4).
    # Each factor's head block keeps its skew part and trace while the spread of its traceless symmetric part is
    # bounded: a learned positional row (pos_phi="learned") by log(kappa_max) / 2, then the token frame by
    # log(kappa_max) minus the realized positional spread (learned or "frozen"), so cond_2(U_i) <= kappa_max
    # (Omega_ij <= kappa_max^2). A frozen positional factor whose spread alone reaches log(kappa_max) is refused
    # at model construction. block_glk / block_glmc / block_sl; inert (cond = 1) for block_so, block_co, so_n,
    # su_n; refused for block_borel and block_unipotent. None: no bound.
    phi_frame_condition_max:          Optional[float] = None

    # pullback_group / fisher_group only: upper bound R on each row's right-factor step ||eta s xi||_F, the embedded
    # Frobenius norm of the body generator xi (under fisher_group xi = U^{-1} W U, so R bounds a Frobenius norm, not
    # the Fisher norm or per-step KL that m_table_fisher_trust_radius bounds). group_log realizes R exactly; so does
    # the chart retraction whenever R is below its certificate cap. On the chart retraction the fixed 1e-6
    # product-residual certificate is quadratic in the step and binds first away from the identity frame: it halves
    # a trust-saturated step to a cap of about 4e-2 at ||phi||_F = 0.05, 9e-3 at 1 and 7e-3 at 1.7 (d = 6;
    # docs/derivations/2026-09-28-phi-natgrad.tex, Table tab:cap), so values above the cap change nothing there.
    # The optimizer reports the halvings as phi_group_backtracks_mean / phi_group_backtracks_max.
    m_phi_group_trust_radius:        float           = 0.1
    m_phi_fisher_rotation_weight:     float           = 2.0          # fisher_group only: completion weight lambda; under m_phi_fisher_completion='rotation' on every rotation, raising Fisher-visible rotation metrics a -> a + 2 lambda

    # "rotation" | "stabilizer"
    # fisher_group only: completion of the belief-Fisher frame metric g(Z) = ||Z eta||^2 + 2||sym Z||_F^2, whose kernel
    # is the stabilizer of the belief (whitened Z = L^{-1} X L, eta = L^{-1} xi).
    #   rotation    g + lambda ||asym Z||_F^2 on every rotation; it also raises the metric of a Fisher-visible rotation
    #               e n^T - n e^T (unit e orthogonal to n below) from a to a + 2 lambda (a = |eta|^2), so no lambda gives
    #               the belief-Fisher natural gradient there
    #   stabilizer  g + lambda ||P asym(Z) P||_F^2, P = I - n n^T, n = eta/|eta|: lambda only on the kernel, so the
    #               whitened step is the pseudo-inverse belief-Fisher natural gradient on the Fisher-visible
    #               subspace plus Pi R / lambda on the kernel (vfe3/geometry/phi_fisher.py). Token-type rows only:
    #               refused with pos_phi='learned' (the pooled positional metric has no stabilizer-only form yet)
    m_phi_fisher_completion:          str             = "rotation"

    # "adamw" | "fisher"
    # M-step update of the type-prior tables mu_embed and sigma_log_embed, p_v = N(mu_v, diag s_v), s_v = e^{l_v}.
    #   adamw   AdamW at m_p_mu_lr / m_p_sigma_lr
    #   fisher  natural-gradient step preconditioned by (n_v/N) diag(1/s_v, 1/2), the Fisher metric of the
    #           prior family p_v counted once per occurrence (the exact Hessian of the self-coupling term only;
    #           a chosen preconditioner for the CE): (mu_v, l_v) -= eta (N/n_v) (s_v grad_mu, 2 grad_l), each
    #           row's step bounded in Fisher norm by m_table_fisher_trust_radius and never gradient-clipped; the
    #           AdamW rates schedule it (step times lr_t/lr_0) and set the decoupled decay, nothing else. only: a
    #           decoder that does not read these tables (decode_mode="linear" or untie_decode_bank=True) and
    #           prior_covariance_table="diagonal"
    m_table_update_mode:              str             = "adamw"
    m_table_fisher_lr:                float           = 0.1          # fisher only: natural step size eta > 0
    m_table_fisher_trust_radius:      float           = 0.1          # fisher only: per-row Fisher-norm bound r (KL <= r^2/2 per step to second order; decay outside the bound)

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

    # Separate decode-side mean tables only (untied decode_mu_embed, linear output_proj_weight): decoupled
    # AdamW decay theta <- theta (1 - lr_t lambda), which is not an L2 penalty or a Gaussian-prior MAP
    # (Loshchilov and Hutter 2019). Against a mean Adam-normalized update u_bar it settles at
    # |theta| = |u_bar| / lambda. The rows of types that are rarely or never targets receive only the softmax
    # push away from the queries, which nothing else opposes, and the KL logit is unbounded below in their
    # distance (audit 2026-10-10, item 2). None = mu_weight_decay; 0 = exempt.
    decode_mu_weight_decay:           Optional[float] = None

    # Log-variance tables. Decay pulls sigma toward 1 against sigma_init. None = weight_decay; 0 = exempt.
    sigma_weight_decay:               Optional[float] = 0
    
    # Frame tables (phi_embed, pos_phi_free) under adamw; sets a frame-norm ceiling (0 = none).
    phi_weight_decay:                 float           = 0.065

    ##################################################
    #   19. Numerics and precision
    ##################################################

    eps:                              float           = 1e-6
    # safety clamp on every divergence; None resolves at construction to 8 * embed_dim, the launchers' scaling,
    # and the resolved number is what the config stores and saves (audit 2026-10-12 C6)
    kl_max:                           Optional[float] = None
    # "absolute" | "row_min" | "query_shifted": what kl_max clamps in the kernel router, projected_predictive and
    # projected_gradient (audit 2026-10-08 P1-4)
    #   absolute       D_ij; a row whose every admissible cost exceeds kl_max routes by the attention prior pi alone
    #                  (uniform under a causal mask, the ALiBi weights under causal_alibi) with zero cost gradient
    #   row_min        D_ij - min_{j admissible} D_ij; beta is the exact softmax wherever no gap exceeds kl_max
    #   query_shifted  D_ij - c_i, clamped from above only, with c_i the query-only row constant of the cost
    #                  (-(r + log det G_i)/2 for the Gaussian KL, -log det G_i / 2 for noisy_query, 0 for
    #                  mean_distance), which the softmax ignores; beta equals the absolute route's wherever nothing
    #                  is capped, and shrinking query variances no longer saturate the clamp (audit 2026-10-17 item 2)
    routing_kl_max_reference:         str             = "absolute"
    # None | tolerance in nats (suggested 1e-3): Gaussian-KL kernel router only. A (batch, head) slice whose
    # a-priori error estimate of the centered pair-KL GEMM exceeds it is rescored by the direct-difference
    # tiles (audit 2026-10-09 round 2 #7). Opt-in: about +45% step time on the K=128 base on the dense route; the
    # diagonal comparison route keeps its closed form with its own O(N r) estimate and one host read per layer
    # (audit 2026-10-17 item 11).
    pair_kl_gemm_tol:                 Optional[float] = None
    # int >= 1. Byte budget, in MiB, of one tile of the relational scorer (projected moments, source-root factor,
    # GEMM-fallback direct pair tiles, mean-distance rows) and of the routed spread's query-row tiles. Larger tiles
    # dispatch fewer kernels, until one tile holds the whole loop, and hold more transient memory; values change only
    # by float32 reduction order, and loops above 32 MiB stay checkpointed. A projection tile holds at most
    # 32 * (relational_tile_mib // 32) tokens (audit 2026-10-17 item 6). Keep 32 until a CUDA A/B confirms a gain
    # (audit 2026-10-16 F26).
    relational_tile_mib:              int             = 32
    sigma_max:                        Optional[float] = 100.0        # variance ceiling; None = none
    transport_chart_max_norm:         float           = 50.0         # fail-closed ||M||_F radius of every transport exponential

    # None | "bf16" | "fp16"
    # Mixed precision on CUDA. None = full fp32 (no autocast). Decode, CE and matrix functions stay fp32.
    # bf16 needs no GradScaler; fp16 training does. Its only reduced-precision consumers are the linear
    # decode's logits matmul (decode_logits_amp) and the CG coupling; elsewhere inactive_fields() marks it.
    amp_dtype:                        Optional[str]   = "bf16"

    # None | "highest" | "high" | "medium"
    #   None     leave the card's TF32 policy as is
    #   highest  no TF32 (full fp32, reproducible)
    #   high     TF32 (or bf16x3) allowed
    #   medium   bf16 allowed
    fp32_matmul_precision:            Optional[str]   = "highest"

    # None (off) | tol in (0, 1)
    # Redo a frame producer call in float64 when its float32 frame pair is ill-conditioned,
    # eps_32 kappa_F(U) > tol (1e-4: kappa_F > 838.9). Costly (about 7x per escalated step).
    frame_pair_escalation_tol:        Optional[float] = None
    
    # "gram" | "gram_certified" | "cholesky_qr2_certified"
    # Factorization of the projected Gaussian-KL scorer. Failed float32 factorizations are always rebuilt
    # from source where the scorer has source tuples; a Gram-route recovery runs in float32 shifted
    # CholeskyQR3 and falls back to float64 QR when that fails.
    #   gram                    factor the float32 Gram X X^T (error O(eps kappa^2))
    #   gram_certified          gram, and also rebuild entries that factor but fail
    #                           eps_32 ||G||_F ||G^-1||_F <= 1e-4 (costly on CUDA; with minimal_agent it
    #                           also screens the ADF query covariances; under comparison_invariant with a
    #                           diagonal P(G) it keeps the dense Gram route in place of the diagonal one, as does
    #                           cholesky_qr2_certified)
    #   cholesky_qr2_certified  factor X by CholeskyQR2 (error O(eps kappa)) and rebuild entries failing
    #                           eps_32 kappa_1(C) <= 3e-4, or failing outright, in float64 (QR of the
    #                           float64 source root; their pairs leave the GEMM for float64 triangular-solve
    #                           tiles); the Gram route serves minimal_agent, noisy_query and comparison_invariant,
    #                           which have no source-root factor route (comparison_invariant does pass its R_i S
    #                           source tuples, but a commutant projection P(XX^T) has no CholeskyQR2 root factor),
    #                           so there cholesky_qr2_certified acts as gram_certified: float32 rebuilds from source
    #                           with float64 QR only on failure
    projected_factorization:          str             = "gram"
    
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
        from vfe3.relational_scoring import ROUTING_COSTS as _routing_costs   # the registry, as for groups
        _require(self.relational_routing, tuple(_routing_costs), "relational_routing")
        if self.relational_routing != "gaussian_kl" and full_link:
            raise ValueError(f"relational_routing={self.relational_routing!r} requires attention_mode='projected_predictive'")
        from vfe3.model.minimal_agent import MINIMAL_AGENT_INFERENCE, PREDICTIVE_READOUTS
        _require(self.predictive_readout, PREDICTIVE_READOUTS, "predictive_readout")
        _require(self.minimal_agent_inference, MINIMAL_AGENT_INFERENCE, "minimal_agent_inference")
        for name in ("minimal_agent_cavi_steps", "block_mlp_cavi_steps"):
            steps = getattr(self, name)
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                raise ValueError(f"{name} must be an int >= 1, got {steps!r}")
        if (isinstance(self.minimal_agent_channel_noise, bool)
                or not isinstance(self.minimal_agent_channel_noise, Real)
                or not math.isfinite(float(self.minimal_agent_channel_noise))
                or float(self.minimal_agent_channel_noise) <= 0.0):
            raise ValueError("minimal_agent_channel_noise must be a finite real value > 0, "
                             f"got {self.minimal_agent_channel_noise!r}")
        if (isinstance(self.relational_key_noise, bool)
                or not isinstance(self.relational_key_noise, Real)
                or not math.isfinite(float(self.relational_key_noise))
                or float(self.relational_key_noise) <= 0.0):
            raise ValueError(f"relational_key_noise must be a finite real value > 0, got {self.relational_key_noise!r}")
        from vfe3.model.minimal_agent import PREDICTIVE_COVARIANCES
        _require(self.predictive_covariance, PREDICTIVE_COVARIANCES, "predictive_covariance")
        if (self.predictive_covariance == "spread_corrected" and self.attention_mode == "projected_predictive"
                and (self.predictive_readout != "minimal_agent" or self.minimal_agent_inference != "mean_field")):
            raise ValueError("predictive_covariance='spread_corrected' is the minimal agent's mean-field covariance "
                             "with the linearized label spread; it requires predictive_readout='minimal_agent' and "
                             "minimal_agent_inference='mean_field' (the kernel readout and the exact posterior "
                             "already carry the between-source spread; use 'moment_matched')")
        if not isinstance(self.predictive_covariance_fp64_escalation, bool):
            raise ValueError("predictive_covariance_fp64_escalation must be bool, got "
                             f"{self.predictive_covariance_fp64_escalation!r}")
        from vfe3.model.relational_operators import PREDICTIVE_COVARIANCE_PRECISIONS
        _require(self.predictive_covariance_precision, PREDICTIVE_COVARIANCE_PRECISIONS,
                 "predictive_covariance_precision")
        if self.predictive_covariance_floor_mode not in ("prior", "relative", "max"):
            raise ValueError(f"predictive_covariance_floor_mode must be 'prior', 'relative' or 'max', "
                             f"got {self.predictive_covariance_floor_mode!r}")
        for name in ("predictive_covariance_spread_squares", "predictive_covariance_symmetric_pairs"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
            if getattr(self, name) and self.content_link_edges != "off":
                raise ValueError(f"{name} reads the head-pair pooled covariance without edge links; "
                                 "it requires content_link_edges='off'")
        if (isinstance(self.predictive_covariance_floor, bool)
                or not isinstance(self.predictive_covariance_floor, Real)
                or not math.isfinite(float(self.predictive_covariance_floor))
                or float(self.predictive_covariance_floor) <= 0.0):
            raise ValueError(
                f"predictive_covariance_floor must be a finite real value > 0, "
                f"got {self.predictive_covariance_floor!r}")
        if self.predictive_covariance_ceiling is not None and (
                isinstance(self.predictive_covariance_ceiling, bool)
                or not isinstance(self.predictive_covariance_ceiling, Real)
                or not math.isfinite(float(self.predictive_covariance_ceiling))
                or float(self.predictive_covariance_ceiling) <= float(self.predictive_covariance_floor)):
            raise ValueError(
                f"predictive_covariance_ceiling must be None or a finite real value above predictive_covariance_floor "
                f"({self.predictive_covariance_floor}), got {self.predictive_covariance_ceiling!r}")
        if (isinstance(self.relational_receiver_noise, bool)
                or not isinstance(self.relational_receiver_noise, Real)
                or not math.isfinite(float(self.relational_receiver_noise))
                or self.relational_receiver_noise <= 0.0):
            raise ValueError("relational_receiver_noise must be a finite positive reference variance")
        _require(self.relational_receiver_measurement, ("adjoint", "inverse"), "relational_receiver_measurement")
        _require(self.relational_receiver_update, ("closed_form", "closed_form_profiled", "euler"),
                 "relational_receiver_update")
        if (isinstance(self.relational_receiver_damping, bool)
                or not isinstance(self.relational_receiver_damping, Real)
                or not math.isfinite(float(self.relational_receiver_damping))
                or not 0.0 < self.relational_receiver_damping <= 1.0):
            raise ValueError("relational_receiver_damping must lie in (0, 1]")
        if full_link and self.n_e_steps < 1:
            raise ValueError("attention_mode='projected_gradient' requires n_e_steps >= 1; "
                             "n_e_steps=0 would remove attention from every layer")
        # The projected support envelope. The divergence (KL), connection (flat), gradient roles
        # (filtering), estimator (unroll) and mean preconditioner (Fisher) are fixed, not fields.
        incompatible = []
        if not full_link and self.n_e_steps != 1:
            incompatible.append(f"n_e_steps={self.n_e_steps!r} (requires 1)")
        # The receiver profiles the closed-form alpha forms (projected_gradient.tex, profiled
        # self-coupling); per coordinate, KL(q||p) separates only on the diagonal family.
        if (full_link and self.lambda_alpha_mode == "state_dependent_per_coord"
                and self.family != "gaussian_diagonal"):
            incompatible.append(f"lambda_alpha_mode='state_dependent_per_coord' with family={self.family!r} "
                                "(requires gaussian_diagonal)")
        # Per coordinate, mu*(alpha) = Lambda(alpha)^{-1} eta(alpha) couples the coordinate fixed points.
        if (full_link and self.relational_receiver_update == "closed_form_profiled"
                and self.lambda_alpha_mode == "state_dependent_per_coord"):
            incompatible.append("relational_receiver_update='closed_form_profiled' with lambda_alpha_mode="
                                "'state_dependent_per_coord' (requires 'state_dependent')")
        if self.encode_mode not in ("per_token", "per_token_additive"):
            incompatible.append(
                f"encode_mode={self.encode_mode!r} (requires 'per_token' or 'per_token_additive')")
        if self.family not in ("gaussian_diagonal", "gaussian_full", "gaussian_block"):
            incompatible.append(f"family={self.family!r} (requires gaussian_diagonal, gaussian_full or "
                                "gaussian_block)")
        if self.n_heads < 2:
            incompatible.append(f"n_heads={self.n_heads!r} (requires at least 2)")
        if incompatible:
            raise ValueError(
                f"attention_mode={self.attention_mode!r} has unsupported projected_filtering configuration: "
                + "; ".join(incompatible))

        # numerics
        if not (math.isfinite(self.eps) and self.eps > 0.0):
            raise ValueError(f"eps must be finite and positive, got {self.eps}")
        if self.kl_max is None:
            self.kl_max = float(8 * self.embed_dim)
        if not self.kl_max > 0.0:
            raise ValueError(f"kl_max must be positive (infinity disables the ceiling), got {self.kl_max}")
        if self.self_kl_max is not None and (isinstance(self.self_kl_max, bool)
                                             or not isinstance(self.self_kl_max, Real)
                                             or not self.self_kl_max > 0.0):
            raise ValueError(f"self_kl_max must be None or positive (infinity disables the clamp), "
                             f"got {self.self_kl_max!r}")
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
        if self.pos_phi != "none" and not (math.isfinite(self.pos_phi_scale) and self.pos_phi_scale >= 0.0):
            raise ValueError(
                f"pos_phi_scale must be finite and >= 0 when pos_phi is active, got {self.pos_phi_scale}")
        if not (
            math.isfinite(self.m_phi_group_trust_radius)
            and self.m_phi_group_trust_radius > 0.0
        ):
            raise ValueError(
                "m_phi_group_trust_radius must be finite and positive, got "
                f"{self.m_phi_group_trust_radius}"
            )
        _require(self.m_phi_group_retraction, ("chart", "group_log"), "m_phi_group_retraction")
        _require(self.m_phi_group_log_fallback, ("series", "eigenbasis"), "m_phi_group_log_fallback")
        _require(self.m_phi_group_differential_mode, ("series", "spectral"), "m_phi_group_differential_mode")
        if not (
            math.isfinite(self.m_phi_fisher_rotation_weight)
            and self.m_phi_fisher_rotation_weight > 0.0
        ):
            raise ValueError(
                "m_phi_fisher_rotation_weight must be finite and positive, got "
                f"{self.m_phi_fisher_rotation_weight}"
            )
        _require(self.m_phi_fisher_completion, ("rotation", "stabilizer"), "m_phi_fisher_completion")
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
            if self.rope_mode in _FRAME_ROPE_MODES:
                raise ValueError(
                    f"gauge_transport='off' requires Omega=I, but rope_mode={self.rope_mode!r} folds "
                    f"a positional rotation into the transport. Set rope_mode='none' or a comparison mode."
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
            from vfe3.geometry.irreps import check_irrep_constructible, irrep_dim
            if isinstance(self.irrep_spec, str):
                if self.irrep_spec == "random" and _traits.algebra != "so":
                    raise ValueError(
                        f"irrep_spec='random' draws so(N) towers only; gauge_group={self.gauge_group!r} needs an "
                        "explicit [(label, mult), ...] list")
                if self.irrep_spec != "random":
                    raise ValueError(f"irrep_spec must be a list of (label, mult) pairs or 'random', "
                                     f"got {self.irrep_spec!r}")
                if type(self.irrep_spec_seed) is not int:
                    raise ValueError(f"irrep_spec_seed must be an int, got {self.irrep_spec_seed!r}")
                from vfe3.geometry.irreps import random_irrep_spec
                self.irrep_spec = random_irrep_spec(
                    self.group_n, self.embed_dim, self.n_heads, algebra=_traits.algebra,
                    seed=self.irrep_spec_seed,
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
                check_irrep_constructible(self.group_n, algebra=_traits.algebra, label=_entry[0])
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
        # Head-block family (gaussian_block): exact only where every producer keeps one SPD block per head --
        # the kernel predictive readout's moment-matched head-block projection (or passthrough), mean-only
        # block maps, compact head-block frames, and decoders that score a full-layout covariance
        # against the diagonal bank exactly (full_chunked) or up to a per-position constant (diagonal_chunked).
        if _family_cls.head_block_covariance:
            block_problems = []
            if self.attention_mode != "projected_predictive":
                block_problems.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive')")
            if self.predictive_readout != "kernel":
                block_problems.append(f"predictive_readout={self.predictive_readout!r} (requires 'kernel')")
            # Any group declaring the head-block family has every element block-diagonal over its heads, so its
            # frames are compact head blocks when the heads are equal (GaugeGroup.head_block_generators).
            from vfe3.geometry.groups import declared_invariant_families
            if ("gaussian_block" not in declared_invariant_families(self.gauge_group)
                    or len(set(self.head_dims)) != 1):
                block_problems.append(f"gauge_group={self.gauge_group!r} with head sizes {self.head_dims} (requires "
                                      "equal head blocks of a group that preserves them: block_glk, any block_* "
                                      "group or an equal-block so_n / su_n tower)")
            if self.content_link_edges != "off":
                block_problems.append(f"content_link_edges={self.content_link_edges!r} (requires 'off')")
            if self.decode_mode not in ("linear", "diagonal_chunked", "full_chunked"):
                block_problems.append(f"decode_mode={self.decode_mode!r} (requires 'linear', 'diagonal_chunked' "
                                      "or 'full_chunked')")
            # delta_full is read through the head-block M-projection of J Sigma J^T + floor
            # (vfe3.model.block._head_block_readout); the mean-field prototype posterior
            # P = (C^-1 + diag D)^-1 of a block-diagonal C is block-diagonal itself; the exact posterior's
            # between-template spread is dense, so it would leave the family.
            if (self.use_block_mlp and self.block_mlp_covariance == "posterior"
                    and self.block_mlp_inference != "mean_field"):
                block_problems.append(f"block_mlp_covariance='posterior' with block_mlp_inference="
                                      f"{self.block_mlp_inference!r} (requires 'mean_field': the exact posterior's "
                                      "between-template spread leaves the head-block family)")
            if self.use_cg_coupling:
                block_problems.append("use_cg_coupling=True")
            if block_problems:
                raise ValueError(f"family={self.family!r} keeps one SPD covariance block per head and is defined "
                                 "only where every covariance producer preserves that; got " + "; ".join(block_problems))
        _require(self.prior_covariance_table, ("diagonal", "block_cholesky"), "prior_covariance_table")
        # Frame-free (xi, C) prior table: with identity frames (gauge_transport='off') it represents exactly what
        # the learned-frame reference arm reads, C_v = U_v^-1 diag(sigma_v) U_v^-T as one SPD block per head.
        if self.prior_covariance_table == "block_cholesky":
            table_problems = []
            if self.family not in ("gaussian_block", "gaussian_full"):
                table_problems.append(f"family={self.family!r} (requires 'gaussian_block' or 'gaussian_full': a full "
                                      "SPD block per head)")
            if self.gauge_transport != "off":
                table_problems.append(f"gauge_transport={self.gauge_transport!r} (requires 'off': the table "
                                      "replaces the frames)")
            if self.gauge_group != "block_glk":
                table_problems.append(f"gauge_group={self.gauge_group!r} (requires 'block_glk')")
            if self.attention_mode != "projected_predictive":
                table_problems.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive')")
            if self.content_link_edges != "off":
                table_problems.append(f"content_link_edges={self.content_link_edges!r} (requires 'off')")
            if self.encode_mode != "per_token":
                table_problems.append(f"encode_mode={self.encode_mode!r} (requires 'per_token')")
            if self.decode_mode != "linear" and not self.untie_decode_bank:
                table_problems.append("untie_decode_bank=False with a prior-bank decode (the decode bank keeps "
                                      "its own diagonal tables; requires untie_decode_bank=True)")
            if table_problems:
                raise ValueError("prior_covariance_table='block_cholesky' is the frame-free reference-arm table and "
                                 "is defined only there; got " + "; ".join(table_problems))

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
        # prior-shape knob (threaded into model._attention_log_prior; audit P9): the causal_windowed
        # band width must be a positive int.
        if not isinstance(self.attention_window, int) or self.attention_window < 1:
            raise ValueError(f"attention_window must be an int >= 1, got {self.attention_window!r}")
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
            if self.m_phi_fisher_completion == "stabilizer" and self.pos_phi == "learned":
                raise ValueError(
                    "m_phi_fisher_completion='stabilizer' is implemented for the token-type frame rows only; "
                    "pos_phi='learned' also steps the positional frames, whose pooled Fisher has no "
                    "stabilizer-only completion yet. Use m_phi_fisher_completion='rotation', or pos_phi='none' "
                    "or 'frozen'")
        if phi_update_policy.requires_pullback_geometry:
            from vfe3.geometry.groups import get_group
            group_builder = get_group(self.gauge_group)
            capability = getattr(group_builder, "pullback_capability", None)
            if capability is None:
                raise ValueError(
                    f"gauge_group={self.gauge_group!r} registers no pullback capability; phase one "
                    "requires a built-in bracket-closed per-block pullback direction"
                )
            # The eigenbasis fallback of group_log never builds structure constants (audit 2026-10-16, F40).
            structure_problem = (None if self._pullback_uses_eigenbasis_fallback()
                                 else self._pullback_structure_problem())
            if structure_problem is not None:
                raise ValueError(f"pullback_group {structure_problem}")
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
        if self.phi_frame_condition_max is not None:
            from vfe3.gauge_optim import FRAME_CONDITION_GROUPS, FRAME_CONDITION_INERT_GROUPS
            if (isinstance(self.phi_frame_condition_max, bool)
                    or not isinstance(self.phi_frame_condition_max, Real)
                    or not math.isfinite(self.phi_frame_condition_max)
                    or self.phi_frame_condition_max <= 1.0):
                raise ValueError("phi_frame_condition_max must be None or a finite real > 1, got "
                                 f"{self.phi_frame_condition_max!r}")
            frame_condition_problems = []
            if self.gauge_group not in FRAME_CONDITION_GROUPS + FRAME_CONDITION_INERT_GROUPS:
                frame_condition_problems.append(
                    f"gauge_group in {FRAME_CONDITION_GROUPS} (implemented) or {FRAME_CONDITION_INERT_GROUPS} "
                    f"(cond = 1, inert); got {self.gauge_group!r}")
            if phi_update_policy.requires_manifold_optimizer:
                frame_condition_problems.append(
                    f"m_phi_update_mode='adamw' (the manifold optimizer keeps its own chart); got "
                    f"{self.m_phi_update_mode!r}")
            if self.encode_mode != "per_token":
                frame_condition_problems.append(
                    f"encode_mode='per_token' (per_token_additive does not use phi_embed as a frame); got "
                    f"{self.encode_mode!r}")
            if frame_condition_problems:
                raise ValueError("phi_frame_condition_max requires " + "; ".join(frame_condition_problems))

        from vfe3.model.positional_phi import _POS_PHI
        _require(self.pos_phi, tuple(sorted(_POS_PHI)), "pos_phi")
        _require(self.rope_mode, _ROPE_MODES, "rope_mode")
        # The retained projected routes fold RoPE into the realized frame on the right,
        # V_i = U_i R_i^T (rope_insertion is fixed to "right"). Under U_i -> g_i U_i this gives
        # V_i -> g_i V_i, the same right-factor structure as pos_phi's exp(Y_i), so routing and
        # content stay exactly covariant (audit 2026-10-01 F11: residual ~6e-16). The former
        # gauge-breaking warning quoted a probe of the retired left insertion and was removed.
        # RoPE's adjacent-pair rotation R is orthogonal but generally NOT in the irrep image
        # rho(SO(N)) inside GL(K), so R Omega R^T leaves the SELECTED structure group
        # (GL(K)-congruence invariance survives).
        if self.rope_mode != "none":
            # p-RoPE rotates floor(p d / 2) pairs of each block (frame space: each irrep block; comparison
            # space: each head's comparison block, also of the head size); none rotated is NoPE, not p-RoPE.
            empty = sorted({d for d in self.head_dims if d >= 2 and int(float(self.rope_fraction) * d // 2) == 0})
            if empty:
                raise ValueError(f"rope_fraction={self.rope_fraction} rotates no pair of blocks of size {empty}; "
                                 "raise it or use rope_mode='none'")
        # minimal_agent: the comparison rotation is a known record channel R_{j-i}, applied by exact conditioning.
        if self.rope_mode in _COMPARISON_ROPE_MODES and self.attention_mode != "projected_predictive":
            raise ValueError(f"rope_mode={self.rope_mode!r} rotates the projected routing moments after the learned "
                             f"maps; got attention_mode={self.attention_mode!r} (requires 'projected_predictive')")
        if self.rope_mode == "comparison_invariant" and self.predictive_readout == "minimal_agent":
            raise ValueError("rope_mode='comparison_invariant' projects the kernel router's comparison "
                             "covariance; predictive_readout='minimal_agent' applies the rotation as a record channel "
                             "by exact conditioning and does not read it (use 'comparison')")
        if (self.rope_mode == "comparison_invariant"
                and all(d - 2 * int(float(self.rope_fraction) * d // 2) <= 1 for d in self.head_dims)):
            _warn_diagonal_comparison_covariance(self.head_dims, float(self.rope_fraction), float(self.rope_base),
                                                 self.max_seq_len)
        if self.rope_mode in _FRAME_ROPE_MODES and not _traits.contains_pair_rotations:
            import warnings
            warnings.warn(
                f"rope_mode={self.rope_mode!r} with gauge_group={self.gauge_group!r} leaves the structure "
                f"group: the coordinate-pair rotation R(theta) is generally not in the group's "
                f"image in GL(K), so R Omega R^T is no longer in the selected group. The "
                f"GL(K)-congruence divergence invariance still holds (R is orthogonal), so the "
                f"model runs; only the structure-group claim is dropped.",
                UserWarning,
                stacklevel=2,
            )
        _require(self.projected_factorization, _PROJECTED_FACTORIZATIONS, "projected_factorization")
        value = self.frame_pair_escalation_tol
        if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                  or not math.isfinite(value) or not 0.0 < value < 1.0):
            raise ValueError(f"frame_pair_escalation_tol must be None or a finite number in (0, 1), got {value!r}")
        if not isinstance(self.diagnose_canonical_free_energy, bool):
            raise ValueError(f"diagnose_canonical_free_energy must be a bool, got "
                             f"{self.diagnose_canonical_free_energy!r}")
        if not isinstance(self.diagnose_agent_elbo, bool):
            raise ValueError(f"diagnose_agent_elbo must be a bool, got {self.diagnose_agent_elbo!r}")
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
            if self.rope_mode == "frame":
                agent_problems.append("rope_mode='frame' (the tied key is the content, so both are read in one "
                                      "frame; use 'frame_value' or 'comparison')")
            if agent_problems:
                raise ValueError("predictive_readout='minimal_agent' is unsupported here: " + "; ".join(agent_problems))
        for _name in ("learn_attention_temperature", "tie_comparison_maps"):
            if not isinstance(getattr(self, _name), bool):
                raise ValueError(f"{_name} must be a bool, got {getattr(self, _name)!r}")
        if self.learn_attention_temperature and self.predictive_readout == "minimal_agent":
            raise ValueError("learn_attention_temperature=True needs a kernel router: predictive_readout='minimal_agent' "
                             "routes by the source posterior at temperature 1")
        if (self.learn_attention_temperature and self.relational_routing == "noisy_key"
                and self.predictive_readout == "kernel"):
            raise ValueError("learn_attention_temperature=True is incompatible with relational_routing='noisy_key', "
                             "which routes at temperature exactly 1 (the learned key noise n_h sets the sharpness)")
        if self.tie_comparison_maps and (self.attention_mode != "projected_predictive"
                                         or self.predictive_readout != "kernel"):
            raise ValueError("tie_comparison_maps=True requires attention_mode='projected_predictive' with "
                             f"predictive_readout='kernel' (got {self.attention_mode!r}, {self.predictive_readout!r})")
        if self.tie_comparison_maps and not self.beta_attention_prior.endswith("_noself"):
            import warnings
            warnings.warn(
                f"tie_comparison_maps=True with beta_attention_prior={self.beta_attention_prior!r}: the prior admits "
                "the self key, and the tie makes D_ii = KL(q_i || q_i) = 0 every row's minimum, so "
                "routing_kl_max_reference='row_min' coincides with the absolute clamp and routing can collapse onto "
                "self (audit 2026-10-09 round 2). The self-masking priors 'causal_noself' and "
                "'causal_alibi_noself' avoid this.",
                UserWarning,
                stacklevel=2,
            )
        # decode_mode / encode_mode validated against the LIVE decode/encode REGISTRIES (not static
        # literals) so a newly registered kernel is config-selectable without editing this validator.
        from vfe3.model.prior_bank import _DECODERS, _ENCODERS
        _require(self.decode_mode, tuple(sorted(_DECODERS)), "decode_mode")
        # decode_mode sets the covariance RANK(s) the prior-bank kernel accepts, via its resolved
        # DecodeRegistration.covariance_kinds: 'diagonal*' -> {'diagonal'}, 'full*' -> {'full'}.
        # The configured family's cov_kind must be a MEMBER of that set, else the
        # mismatch is a shape RuntimeError at the first forward. The linear registration discards
        # sigma, so these rank checks apply only to non-linear registrations.
        decode_registration = _DECODERS[self.decode_mode]
        family_kind = family_cov_kind(self.family)
        # gaussian_block + diagonal_chunked is admitted above: the bank reads diag(Sigma_q), exact up to a
        # per-position constant (PriorBank._diagonal_decoder_dispersion).
        if (self.decode_mode != "linear"
                and family_kind not in decode_registration.covariance_kinds
                and not (_family_cls.head_block_covariance and self.decode_mode == "diagonal_chunked")):
            raise ValueError(
                f"decode_mode={self.decode_mode!r} is rank-incompatible with family={self.family!r}: "
                f"it scores covariance kinds {sorted(decode_registration.covariance_kinds)} but the "
                f"family is {family_kind!r}. Pair a matching decode_mode (a full/diagonal kernel "
                f"with a full/diagonal family), or select decode_mode='linear'."
            )
        if self.decode_chunk_size < 1:
            raise ValueError(f"decode_chunk_size must be >= 1, got {self.decode_chunk_size}")
        _require(self.decode_ce_checkpoint, ("auto", "auto_aggregate"), "decode_ce_checkpoint")
        # decode_tau (prior-bank decode only) is reported by inactive_fields(), so the
        # default-guarded notice below covers it and an untouched default stays silent.
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
                     or self.rope_mode in _FRAME_ROPE_MODES)):
            detail = (
                f"encode_mode='per_token_additive' with pos_phi={self.pos_phi!r}, "
                f"rope_mode={self.rope_mode!r}: the positional frame keeps Omega_ij != I, "
                "so the additive encoder is not the Omega = I capacity control; set "
                "pos_phi='none' and rope_mode='none' (or a comparison mode)"
            )
            raise ValueError(detail + " on the projected routes")
        # The additive encoder's learned capacity is the phi table itself, which gauge_transport 'off'
        # zeroes and freezes and 'frozen' freezes at its random init (audit 2026-10-08 hotspot-3#5).
        if self.encode_mode == "per_token_additive" and self.gauge_transport != "on":
            raise ValueError(
                f"encode_mode='per_token_additive' with gauge_transport={self.gauge_transport!r}: the "
                "additive encoder trains the phi table, which this setting freezes ('off' also zeroes it); "
                "use gauge_transport='on'")
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
        _require(self.routing_kl_max_reference, ("absolute", "row_min", "query_shifted"), "routing_kl_max_reference")
        if self.pair_kl_gemm_tol is not None and not (isinstance(self.pair_kl_gemm_tol, (int, float))
                                                      and not isinstance(self.pair_kl_gemm_tol, bool)
                                                      and self.pair_kl_gemm_tol > 0):
            raise ValueError(f"pair_kl_gemm_tol must be None or a positive number, got {self.pair_kl_gemm_tol!r}")
        if type(self.relational_tile_mib) is not int or self.relational_tile_mib < 1:
            raise ValueError(f"relational_tile_mib must be an int >= 1 (MiB), got {self.relational_tile_mib!r}")

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
        # The diagonal family reads the pushforward through its M-projection diag(J Sigma J^T) (audit 2026-10-10,
        # item 9), as the attention update already projects its dense covariance; gauge_gate's delta_full
        # Jacobian needs the full layout.
        if (self.use_block_mlp and self.block_mlp_covariance == "delta_full"
                and self.family not in ("gaussian_full", "gaussian_block", "gaussian_diagonal")):
            raise ValueError(
                "block_mlp_covariance='delta_full' needs a Gaussian family (family='gaussian_full', "
                "'gaussian_block' through its head-block projection, or 'gaussian_diagonal' through its "
                f"diagonal projection); got family={self.family!r}."
            )
        if (self.use_block_mlp and self.block_mlp_covariance == "delta_full"
                and self.family == "gaussian_diagonal" and self.block_mlp_mode == "gauge_gate"):
            raise ValueError("block_mlp_mode='gauge_gate' with block_mlp_covariance='delta_full' needs the "
                             "full-covariance layout (family='gaussian_full' or 'gaussian_block')")
        _require(self.block_norm_covariance, ("passthrough", "pushforward", "pushforward_moment"),
                 "block_norm_covariance")
        if type(self.block_norm_trace_cap) is not bool:
            raise ValueError(f"block_norm_trace_cap must be a bool, got {self.block_norm_trace_cap!r}")
        if type(self.block_norm_cap_moment_fallback) is not bool:
            raise ValueError("block_norm_cap_moment_fallback must be a bool, got "
                             f"{self.block_norm_cap_moment_fallback!r}")
        if self.block_norm_covariance in ("pushforward", "pushforward_moment") and self.norm_type_block == "layernorm":
            # The block norm pushes Sigma; in reference coordinates it acts on the dense C = U^-1 Sigma U^-T and
            # maps back as U C' U^T, which the full and head-block families keep (audit 2026-10-10 round 2, item 51).
            # Only a LayerNorm block norm reads the field: under 'none' or 'mahalanobis' it is inert (inactive_fields)
            # and either value is accepted (audit 2026-10-16 F5).
            problems = []
            if self.block_coordinates != "local" and self.family not in ("gaussian_full", "gaussian_block"):
                problems.append(f"block_coordinates='local', or 'reference' with family 'gaussian_full' or "
                                f"'gaussian_block' (got {self.block_coordinates!r} with {self.family!r})")
            if self.family not in ("gaussian_full", "gaussian_block", "gaussian_diagonal"):
                problems.append(f"a Gaussian family (got {self.family!r})")
            if problems:
                raise ValueError(f"block_norm_covariance={self.block_norm_covariance!r} needs " + "; ".join(problems))
        if (isinstance(self.block_mlp_em_weight, bool) or not isinstance(self.block_mlp_em_weight, Real)
                or not math.isfinite(float(self.block_mlp_em_weight)) or float(self.block_mlp_em_weight) < 0.0):
            raise ValueError(f"block_mlp_em_weight must be a finite real value >= 0, got {self.block_mlp_em_weight!r}")
        # In local coordinates the block norm maps the mean only and passes Sigma through, so the
        # pushforward must compose the norm Jacobian, J = J_mlp J_norm (vfe3.model.block_mlp.
        # block_norm_jacobian). That closed form exists for 'layernorm' (plain or affine); 'mahalanobis'
        # reads Sigma and has none here.
        if self.use_block_mlp and self.block_mlp_covariance == "delta_full" and self.norm_type_block not in (
                "none", "layernorm"):
            raise ValueError(
                "block_mlp_covariance='delta_full' needs norm_type_block 'none' or 'layernorm': the "
                "pushforward composes the norm Jacobian, which is implemented for LayerNorm only "
                f"(got norm_type_block={self.norm_type_block!r})."
            )
        # The invariant Mahalanobis gate reads s_b = mu_b^T Sigma_bb^{-1} mu_b per irrep block. With a
        # passthrough covariance it is exact on the diagonal family (Sigma_bb = diag sigma_b gives
        # s_b = sum_k mu_k^2 / sigma_k), where it equals the full-family gate on diag_embed(sigma);
        # gaussian_block stores its head blocks densely, so the full-layout gate reads them exactly;
        # other families are not admitted.
        if (self.use_block_mlp
                and block_mlp_registration.gauge_class == "invariant_scalar_intertwiner"
                and self.family not in ("gaussian_full", "gaussian_block", "gaussian_diagonal")):
            raise ValueError(
                "block_mlp_mode='gauge_gate' needs a Gaussian family whose block covariance it can "
                "invert exactly (family='gaussian_full' or 'gaussian_block', or 'gaussian_diagonal' with "
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
        _require(self.decode_covariance, ("belief", "encoded", "constant"), "decode_covariance")
        frame_requirements = []
        if self.rope_mode in _FRAME_ROPE_MODES:
            frame_requirements.append(
                f"rope_mode={self.rope_mode!r} (RoPE folds a rotation into the routing frame, "
                "so the content/decode frame is not U_i)")
        if self.block_coordinates == "reference":
            incompatible = list(frame_requirements)
            if self.attention_mode != "projected_predictive":
                incompatible.append(f"attention_mode={self.attention_mode!r} (requires 'projected_predictive')")
            if self.norm_type_block not in ("none", "layernorm"):
                incompatible.append(f"norm_type_block={self.norm_type_block!r} (requires 'none' or 'layernorm')")
            if self.use_cg_coupling:
                incompatible.append("use_cg_coupling=True (acts in local coordinates before the norm)")
            if self.use_block_mlp and self.block_mlp_mode not in ("coordinate", "gaussian_template",
                                                                  "prototype_agents"):
                incompatible.append(
                    f"block_mlp_mode={self.block_mlp_mode!r} (requires 'coordinate', 'gaussian_template' or "
                    "'prototype_agents'; canonical_frame is already conjugated and gauge_gate is already "
                    "covariant)")
            # 'posterior' (prototype_agents only, by the registry) maps C^+ back as U C^+ U^T; a delta_full
            # coordinate MLP pushes the dense C and maps it back the same way, kept by the full and head-block
            # families (audit 2026-10-10 round 2, item 51).
            covariant_delta_full = (self.block_mlp_covariance == "delta_full"
                                    and self.family in ("gaussian_full", "gaussian_block"))
            if (self.use_block_mlp and self.block_mlp_covariance not in ("passthrough", "posterior")
                    and not covariant_delta_full):
                incompatible.append(f"block_mlp_covariance={self.block_mlp_covariance!r} (requires "
                                    "'passthrough', 'posterior' with prototype_agents, or 'delta_full' with "
                                    "family 'gaussian_full' or 'gaussian_block')")
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
        if self.decode_second_order and self.decode_mode == "linear":
            incompatible = []
            if self.family != "gaussian_diagonal":
                incompatible.append(
                    f"family={self.family!r} (requires 'gaussian_diagonal': the full-covariance term "
                    "1/2 tr(Sigma W^T diag(p) W) needs a per-position (K, K) moment, K times the logits GEMM)")
            if self.decode_coordinates != "local":
                incompatible.append(
                    f"decode_coordinates={self.decode_coordinates!r} (requires 'local': the reference "
                    "covariance U^-1 Sigma U^-T is not diagonal)")
            if incompatible:
                raise ValueError(
                    "decode_second_order=True adds the diagonal delta-method term to the linear decode CE; "
                    "got " + "; ".join(incompatible))

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
        _require(self.m_table_update_mode, ("adamw", "fisher"), "m_table_update_mode")
        for name in ("m_table_fisher_lr", "m_table_fisher_trust_radius"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v)) or v <= 0.0:
                raise ValueError(f"{name} must be a finite real value > 0, got {v!r}")
        if self.m_table_update_mode == "fisher":
            # The count-weighted metric (n_v/N) G_v is the Fisher of the gathered prior rows only: a decoder
            # that reads these tables sends every row a softmax gradient, including rows with n_v = 0.
            table_problems = []
            if self.decode_mode != "linear" and not self.untie_decode_bank:
                table_problems.append("a decoder that does not read the prior tables (decode_mode='linear' or "
                                      "untie_decode_bank=True)")
            if self.prior_covariance_table != "diagonal":
                table_problems.append(f"prior_covariance_table='diagonal' (got {self.prior_covariance_table!r})")
            if table_problems:
                raise ValueError("m_table_update_mode='fisher' needs " + "; ".join(table_problems))
        if self.e_mu_q_trust is not None and not (
            math.isfinite(self.e_mu_q_trust) and self.e_mu_q_trust > 0.0
        ):
            raise ValueError(
                f"e_mu_q_trust must be finite and > 0 or None, got {self.e_mu_q_trust}")
        for name in ("m_p_mu_lr", "m_p_sigma_lr", "m_phi_lr", "weight_decay", "phi_weight_decay", "min_lr_frac"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        if self.min_lr_frac > 1.0:      # a floor above the base rate would raise every group's peak lr
            raise ValueError(f"min_lr_frac must be <= 1, got {self.min_lr_frac}")
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
        for name in ("checkpoint_keep_last", "max_failed_step_skips", "max_consecutive_skips"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be None or an int >= 1, got {value!r}")
        if self.loss_spike_skip_nats is not None and (
                isinstance(self.loss_spike_skip_nats, bool) or not isinstance(self.loss_spike_skip_nats, Real)
                or not math.isfinite(float(self.loss_spike_skip_nats)) or float(self.loss_spike_skip_nats) <= 0.0):
            raise ValueError(f"loss_spike_skip_nats must be None or a finite real value > 0, "
                             f"got {self.loss_spike_skip_nats!r}")
        # max_steps=0 is an empty training loop and warmup_steps<0 is meaningless; warmup_steps=0
        # stays legal. A warmup longer than the run never reaches the peak rate, so the cosine
        # schedule would silently train at a fraction of it. (The plain-int type is gated at the top
        # of __post_init__.)
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be an int >= 1, got {self.max_steps!r}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be an int >= 0, got {self.warmup_steps!r}")
        if self.warmup_steps > self.max_steps:
            raise ValueError(f"warmup_steps ({self.warmup_steps}) must not exceed max_steps ({self.max_steps})")
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
        for name in ("sigma_weight_decay", "mu_weight_decay", "decode_mu_weight_decay"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 or None, got {value}")

        # Activity is structural, independent of values and warning visibility. Notices only
        # display settings that differ from their defaults; sweep validation uses the full map.
        defaults = {field.name: field.default for field in fields(self) if field.default is not MISSING}
        defaults["kl_max"] = float(8 * self.embed_dim)       # what the None default resolved to
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
        # The commutant-projected comparison route reads the factorization toggles differently from the other routes
        # (vfe3.gradients.relational._diagonal_comparison_route); a notice, since gram and certified still differ here
        # (audit 2026-10-16 F9).
        if (self.attention_mode == "projected_predictive" and self.predictive_readout == "kernel"
                and self.relational_routing == "gaussian_kl" and self.rope_mode == "comparison_invariant"):
            route_notes = []
            if self.projected_factorization == "cholesky_qr2_certified":
                route_notes.append("projected_factorization='cholesky_qr2_certified' acts as 'gram_certified' on this "
                                   "route: the commutant projection P(XX^T) has no source-root factor route")
            from vfe3.geometry.groups import group_traits
            if group_traits(self.gauge_group).algebra is None:            # equal heads of embed_dim // n_heads
                from vfe3.geometry.rope import rope_commutant_is_diagonal, rope_rotated_pairs
                d_head = self.embed_dim // self.n_heads
                if (rope_commutant_is_diagonal(d_head, rope_rotated_pairs(d_head, self.rope_fraction))
                        and self.projected_factorization != "gram"):
                    # pair_kl_gemm_tol keeps the diagonal scorer, with its own fallback (audit 2026-10-17 item 11).
                    route_notes.append(
                        f"P(G) is diagonal (rope_fraction={self.rope_fraction!r}), but projected_factorization="
                        f"{self.projected_factorization!r} bypasses "
                        "the closed-form diagonal comparison scorer for the dense Gram route")
            if route_notes:
                import warnings
                warnings.warn("rope_mode='comparison_invariant': " + "; ".join(route_notes), ConfigNotice,
                              stacklevel=2)

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

        from vfe3.model.minimal_agent import READOUT_COVARIANCES
        predictive_moment_matched = (self.attention_mode == "projected_predictive"
                                     and self.predictive_covariance in READOUT_COVARIANCES)
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
        _require(self.grad_clip_scope, ("role", "group"), "grad_clip_scope")

    def inactive_fields(self) -> dict[str, str]:
        """Return inactive config field names and reasons, including default-valued fields.

        The same structural result drives display notices and sweep admission. Consumers
        must not infer activity from whether a warning happens to be visible.
        """
        from vfe3.model.prior_bank import _DECODERS

        inactive: dict[str, str] = {}

        def mark(names: tuple[str, ...], reason: str) -> None:
            inactive.update(dict.fromkeys(names, reason))

        from vfe3.geometry.groups import group_traits
        if group_traits(self.gauge_group).algebra is None:
            mark(("irrep_spec_seed", "irrep_min_rank", "irrep_max_rank"),
                 "only read by irrep-tower groups (gauge_group='so_n')")
        from vfe3.gauge_optim import get_phi_update_policy
        phi_update_policy = get_phi_update_policy(self.m_phi_update_mode)
        if not phi_update_policy.requires_manifold_optimizer:
            mark(("m_phi_group_trust_radius", "m_phi_group_retraction"),
                 "only read by the manifold phi optimizer "
                 "(m_phi_update_mode='pullback_group' or 'fisher_group')")
        if not phi_update_policy.requires_pullback_geometry:
            mark(("m_phi_group_differential_mode",),
                 "only read by the Frobenius-pullback direction (m_phi_update_mode='pullback_group')")
        elif self._pullback_uses_eigenbasis_fallback():
            mark(("m_phi_group_differential_mode",),
                 "only read by the series direction, which m_phi_group_log_fallback='eigenbasis' never calls")
        if not (phi_update_policy.requires_pullback_geometry and self.m_phi_group_retraction == "group_log"):
            mark(("m_phi_group_log_fallback",),
                 "only read by m_phi_update_mode='pullback_group' with m_phi_group_retraction='group_log'")
        if self.m_table_update_mode != "fisher":
            mark(("m_table_fisher_lr", "m_table_fisher_trust_radius"), "only read by m_table_update_mode='fisher'")
        else:
            # The table step reads its group rate only through lr_t/lr_0 (independent of the base rate; 0 freezes
            # the table) and the decoupled decay lr_t * lambda, so a table rate is inert when that decay is 0 and no
            # other optimizer group falls back to it (vfe3/train.py build_optimizer; audit 2026-10-10 round 2,
            # item 20). Under the fisher validation the decoder is linear or untied, so its means read
            # m_decode_mu_lr and an untied bank's log variances read m_decode_sigma_lr.
            layernorm_affine_live = self.layernorm_affine and self.norm_type_block == "layernorm"
            # Groups that read m_p_mu_lr: those whose own rate inherits it, plus CG path weights.
            mu_rate_fallback = (
                any(self.inherits_hyperparameter(name, "m_p_mu_lr")
                    for name, live in (("m_decode_mu_lr", True), ("m_block_mlp_lr", self.use_block_mlp),
                                       ("m_relational_score_lr", True), ("m_relational_content_lr", True),
                                       ("m_layernorm_lr", layernorm_affine_live))
                    if live)
                or self.use_cg_coupling)
            if self.resolved_hyperparameter("mu_weight_decay") == 0.0 and not mu_rate_fallback:
                mark(("m_p_mu_lr",), "m_table_update_mode='fisher' steps mu_embed at m_table_fisher_lr; its AdamW "
                     "rate only shapes the schedule (0 freezes the table), its decay is 0 and no other group "
                     "inherits it")
            sigma_rate_fallback = (self.decode_mode != "linear"
                                   and self.inherits_hyperparameter("m_decode_sigma_lr", "m_p_sigma_lr"))
            if self.resolved_hyperparameter("sigma_weight_decay") == 0.0 and not sigma_rate_fallback:
                mark(("m_p_sigma_lr",), "m_table_update_mode='fisher' steps sigma_log_embed at m_table_fisher_lr; "
                     "its AdamW rate only shapes the schedule (0 freezes the table), its decay is 0 and no other "
                     "group inherits it")
        if self.grad_clip is None or self.grad_clip == 0:
            mark(("grad_clip_scope",), "only read when grad_clip > 0")
        if self.checkpoint_interval == 0 and not self.checkpoint_at_end:
            mark(("checkpoint_keep_last",), "only read when a checkpoint is written (checkpoint_interval > 0 or "
                 "checkpoint_at_end=True)")
        if self.log_interval == 0 or self.eval_interval == 0:
            mark(("block_variance_at_log",), "only read at log steps of a run with evaluation (log_interval > 0 and "
                 "eval_interval > 0, which build the diagnostic windows)")
        if self.beta_attention_prior not in ("causal_alibi", "causal_alibi_noself"):
            mark(("alibi_slope",), "only read by an alibi attention prior")
        if not self.use_cg_coupling:
            mark(("cg_covariance_mode", "cg_covariance_floor", "cg_energy_weight"),
                 "only read when use_cg_coupling=True")
        elif self.cg_covariance_mode == "passthrough":
            mark(("cg_covariance_floor",), "only read by cg_covariance_mode='delta_full'")
        if not phi_update_policy.requires_fisher_statistics:
            mark(("m_phi_fisher_rotation_weight", "m_phi_fisher_completion"),
                 "only read by m_phi_update_mode='fisher_group'")
        minimal_agent = self.predictive_readout == "minimal_agent"
        if self.decode_mode != "linear" or self.amp_dtype is None:
            mark(("decode_logits_amp",), "only read by decode_mode='linear' with amp_dtype set")
        # The forward enters autocast, but every contraction runs in an autocast-disabled fp32/fp64 island
        # (transport exponentials, routing and pair-KL GEMMs, covariance updates, the block MLP's mean path,
        # norms, the minimal agent, the prototype posteriors, decode and CE) or is elementwise. The only
        # reduced-precision consumers are the linear decode's logits matmul under decode_logits_amp and the
        # CG coupling's moment contraction. A dispatch-mode dtype probe found no bf16 output and bitwise-equal
        # loss, gradients and eval logits on every other route tried, on CPU and CUDA (2026-10-12; pinned by
        # tests/test_amp_routes_20261012.py). fp16 is never marked: it also enables the GradScaler.
        if (self.amp_dtype == "bf16" and not self.use_cg_coupling
                and not (self.decode_mode == "linear" and self.decode_logits_amp)):
            mark(("amp_dtype",),
                 "effective precision float32: no contraction on this route runs under autocast (only "
                 "decode_logits_amp on the linear decode and use_cg_coupling consume amp_dtype)")
        if (self.attention_mode == "projected_predictive" and not minimal_agent
                and self.relational_routing == "mean_distance"):
            # The covariance-blind cost reads neither the projected covariances nor the source tuples,
            # so no factorization, certificate or source recovery runs on the routing path (audit
            # 2026-10-08 implementation-engineer#4).
            mark(("projected_factorization",),
                 "relational_routing='mean_distance' scores ||a_i - b_j||^2 / 2 and factors no projected "
                 "covariance")
        if self.gauge_transport == "off":
            mark(("frame_pair_escalation_tol",),
                 "gauge_transport='off' fixes every frame to the identity (kappa_F = d), so the "
                 "certificate cannot trip")
        if self.attention_mode != "projected_predictive":
            mark(("predictive_covariance", "predictive_covariance_floor", "predictive_covariance_floor_mode",
                  "predictive_covariance_ceiling", "attention_residual", "predictive_covariance_checkpoint",
                  "predictive_covariance_fp64_escalation", "predictive_covariance_precision",
                  "predictive_covariance_spread_squares", "predictive_covariance_symmetric_pairs"),
                 "only read by attention_mode='projected_predictive': the gradient receiver's covariance "
                 "comes from its receiver update, and its self-information enters through "
                 "alpha KL(q||p) and the closed-form damping")
        elif self.predictive_covariance == "passthrough":
            # The minimal agent checkpoints its whole readout (mean included), so only the floor is inert there.
            mark(("predictive_covariance_floor", "predictive_covariance_floor_mode", "predictive_covariance_ceiling",
                  "predictive_covariance_spread_squares", "predictive_covariance_symmetric_pairs")
                 if minimal_agent
                 else ("predictive_covariance_floor", "predictive_covariance_floor_mode", "predictive_covariance_ceiling",
                       "predictive_covariance_checkpoint", "predictive_covariance_fp64_escalation",
                       "predictive_covariance_precision", "predictive_covariance_spread_squares",
                       "predictive_covariance_symmetric_pairs"),
                 "only read by predictive_covariance='moment_matched' or 'spread_corrected'")
        elif minimal_agent:
            mark(("predictive_covariance_fp64_escalation", "predictive_covariance_precision",
                  "predictive_covariance_spread_squares", "predictive_covariance_symmetric_pairs"),
                 "only read by predictive_readout='kernel'; the minimal agent recomputes a float32 call in float64 "
                 "on its own readout status (factorization failures, the A^h certificate under a certified "
                 "projected_factorization, and the screen of its floored posterior covariance), unconditionally")
        if self.family == "gaussian_full":
            mark(("predictive_covariance_symmetric_pairs",),
                 "only read by the blockwise covariance body (gaussian_diagonal or gaussian_block on block frames); "
                 "gaussian_full takes the dense body")
        if self.mstep_self_coupling_weight == 0.0:
            mark(("self_kl_max",), "only read by the M-step self-coupling term (mstep_self_coupling_weight > 0)")
        if self.attention_mode != "projected_predictive":
            mark(("predictive_readout",), "only read by attention_mode='projected_predictive'")
        if not ((self.attention_mode == "projected_predictive" and minimal_agent)
                or (self.use_block_mlp and self.block_mlp_mode == "prototype_agents")):
            mark(("diagnose_agent_elbo",),
                 "only read by predictive_readout='minimal_agent' (attention_mode='projected_predictive') or a "
                 "block_mlp_mode='prototype_agents' block MLP")
        if minimal_agent:
            posterior = ("exact source posterior" if self.minimal_agent_inference == "exact"
                         else "mean-field source posterior")
            mark(("relational_routing", "content_link_chart", "attention_residual", "kappa_beta"),
                 f"predictive_readout='minimal_agent' routes by the {posterior} at temperature 1 (no "
                 "routing-cost selector, no kappa_beta temperature) and derives the receiving map and the "
                 "precision-weighted residual (no content factor O, so no chart or residual flag)")
            # The readout applies no clamp (relational.py returns before safe_kl_clamp); the remaining
            # readers are the M-step self-coupling, the CG moment energy and the self term of the canonical
            # free-energy diagnostic (audit 2026-10-10 round 2, item 31).
            if (self.mstep_self_coupling_weight == 0.0 and self.cg_energy_weight == 0.0
                    and not self.diagnose_canonical_free_energy):
                mark(("kl_max",),
                     "the minimal-agent posterior applies no divergence clamp, decoders use kl_max=inf, and "
                     "no M-step self-coupling or CG energy term is enabled; with diagnose_canonical_free_energy=True "
                     "it would feed that log-step diagnostic only")
        else:
            mark(("minimal_agent_inference", "minimal_agent_channel_noise", "minimal_agent_cavi_steps"),
                 "only read by predictive_readout='minimal_agent'")
        if self.predictive_readout == "minimal_agent" and self.minimal_agent_inference != "mean_field":
            mark(("minimal_agent_cavi_steps",), "only read by minimal_agent_inference='mean_field'")
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
            mark(("relational_receiver_noise", "relational_receiver_update",
                  "relational_receiver_damping", "relational_receiver_measurement"),
                 "only read by projected_gradient receiving laws")
        elif self.relational_receiver_update in ("closed_form", "closed_form_profiled"):
            mark(("e_q_mu_lr", "e_q_sigma_lr", "e_mu_q_trust", "e_sigma_q_trust"),
                 f"relational_receiver_update={self.relational_receiver_update!r} applies an exact "
                 "natural-parameter update; its step control is relational_receiver_damping")
        else:
            mark(("relational_receiver_damping",),
                 "only read by relational_receiver_update='closed_form' or 'closed_form_profiled'")
        gradient_controls = ("e_q_mu_lr", "e_q_sigma_lr", "e_mu_q_trust", "e_sigma_q_trust")
        if self.attention_mode == "projected_predictive":
            mark(("skip_belief_sigma_update",),
                 "projected routes run no sigma E-step: predictive_content keeps or moment-matches "
                 "covariance by predictive_covariance")
        if self.attention_mode == "projected_predictive":
            mark(gradient_controls, "predictive_content replaces the mean (and, under "
                 "predictive_covariance='moment_matched', the covariance) by readout, not a gradient step")
            # diagnose_canonical_free_energy reads the self-coupling form (audit 2026-10-10 round 2, item 31).
            if self.mstep_self_coupling_weight == 0.0 and not self.diagnose_canonical_free_energy:
                mark(("lambda_alpha_mode", "b0", "c0"),
                     "predictive_content with mstep_self_coupling_weight=0 has no self-coupling consumer; with "
                     "diagnose_canonical_free_energy=True they would feed that log-step diagnostic only")
        mlp_delta_full = self.use_block_mlp and self.block_mlp_covariance == "delta_full"
        # The norm pushforward reads the same floor (in post-norm units) as delta_full; only a LayerNorm pushes it.
        norm_pushes = (self.block_norm_covariance in ("pushforward", "pushforward_moment")
                       and self.norm_type_block == "layernorm")
        norm_floor = () if norm_pushes and not mlp_delta_full else ("block_mlp_covariance_floor",)
        if norm_pushes and mlp_delta_full:
            mark(("block_norm_covariance",), "block_mlp_covariance='delta_full' already composes the norm Jacobian")
        if not (self.norm_type_block == "layernorm" and self.block_norm_covariance == "pushforward"
                and self.block_norm_trace_cap and not mlp_delta_full):
            mark(("block_norm_cap_moment_fallback",),
                 "only read by a capped block_norm_covariance='pushforward' (block_norm_trace_cap=True, LayerNorm "
                 "block norm, no delta_full MLP)")
        if not (self.norm_type_block == "layernorm" and (norm_pushes or mlp_delta_full)):
            mark(("block_norm_trace_cap",), "only read by a LayerNorm covariance pushforward (block_norm_covariance "
                 "'pushforward' or 'pushforward_moment', or block_mlp_covariance='delta_full')")
        if not self.use_block_mlp:
            mark(("block_mlp_mode", "block_mlp_covariance", *norm_floor,
                  "block_mlp_expansion", "block_mlp_activation", "block_mlp_dropout", "m_block_mlp_lr"),
                 "only read when use_block_mlp=True")
        elif self.block_mlp_mode == "gaussian_template":
            mark(("block_mlp_activation", *norm_floor),
                 "gaussian_template gates by softmax over template KLs and keeps covariance")
        elif self.block_mlp_mode == "prototype_agents":
            mark(("block_mlp_activation", *norm_floor),
                 "prototype_agents has no hidden activation, and its posterior covariance is exact, "
                 "with no floor")
        elif self.block_mlp_covariance != "delta_full":
            mark(norm_floor, "only read by block_mlp_covariance='delta_full' or a block_norm_covariance pushforward")
        if not (self.use_block_mlp and self.block_mlp_mode == "prototype_agents"):
            mark(("block_mlp_inference", "block_mlp_template_init_variance", "block_mlp_template_init",
                  "block_mlp_template_init_pull", "block_mlp_cavi_steps", "block_mlp_em_weight"),
                 "only read by block_mlp_mode='prototype_agents' with use_block_mlp=True")
        else:
            if self.block_mlp_inference != "mean_field":
                mark(("block_mlp_cavi_steps",), "only read by block_mlp_inference='mean_field'")
            if self.block_mlp_template_init == "working_variance":
                mark(("block_mlp_template_init_variance",),
                     "block_mlp_template_init='working_variance' sets lambda_0 from the measured working variance")
            else:
                mark(("block_mlp_template_init_pull",), "only read by block_mlp_template_init='working_variance'")
        if self.pos_phi != "learned":
            mark(("m_pos_phi_lr",), "only pos_phi='learned' creates the learnable positional table")
        if self.norm_type_block != "layernorm":
            mark(("layernorm_affine",), "only read by a 'layernorm' block norm")
        if self.norm_type_block != "layernorm":
            # Under a LayerNorm in reference coordinates on gaussian_diagonal the field is read, and only
            # 'passthrough' is supported there, so it is not inert (audit 2026-10-16 F5).
            mark(("block_norm_covariance",), "only read by a 'layernorm' block norm")
        if not (self.layernorm_affine and self.norm_type_block == "layernorm"):
            mark(("m_layernorm_lr",), "requires layernorm_affine and a layernorm block norm")
        if not self.decode_unigram_prior:
            mark(("unigram_kappa",), "needs decode_unigram_prior=True")
        if self.rope_mode == "none":
            mark(("rope_base", "rope_fraction"), "only read by a rope_mode other than 'none'")
        if self.attention_mode == "projected_predictive" and self.predictive_readout != "kernel":
            mark(("routing_kl_max_reference",), "only read by the kernel router (minimal_agent applies no clamp)")
        if not (self.attention_mode == "projected_predictive" and self.predictive_readout == "kernel"
                and self.relational_routing in ("gaussian_kl", "noisy_key")):
            mark(("pair_kl_gemm_tol",), "only read by the kernel router's Gaussian-KL cost")
        if (self.attention_mode == "projected_predictive" and self.predictive_readout == "kernel"
                and self.relational_routing == "noisy_key"):
            mark(("kappa_beta",), "relational_routing='noisy_key' routes at temperature exactly 1: the learned key "
                                  "noise n_h sets the sharpness")
        else:
            mark(("relational_key_noise",), "only read by relational_routing='noisy_key' on the kernel router")
        if self.pos_phi == "none":
            mark(("pos_phi_project_slk", "pos_phi_scale"), "only read by a non-'none' pos_phi")
        if self.beta_attention_prior == "causal_windowed" and self.attention_window >= self.max_seq_len - 1:
            mark(("attention_window",), "window covers the context: 'causal_windowed' is identical to plain causal")
        if self.n_layers == 1:
            mark(("prior_handoff_rho", "prior_handoff_sigma"), "no next layer consumes the handoff")
        registration = _DECODERS[self.decode_mode]
        if not registration.supports_chunked:
            mark(("decode_ce_checkpoint",), "the selected decoder has no checkpointed chunked CE kernel")
        if self.decode_mode not in ("diagonal_chunked", "full_chunked"):
            mark(("decode_ce_triton",), "read only by the diagonal_chunked and full_chunked KL decode CE")
            # The linear and expected-likelihood decodes form no a_v reduction and run no ranking guard
            # (audit 2026-10-16 F36).
            mark(("decode_av_precision", "decode_ranking_fp64_escalation"),
                 "read only by the diagonal_chunked and full_chunked KL decoders (the a_v reduction) and their "
                 "ranking guard")

        # The route-specific rules below were confirmed by tiny launcher-derived CPU probes: loss and
        # every parameter gradient stay torch.equal under an override of the marked knob
        # (tests/test_remediation_20260923_A.py). The self-coupling rule is exact in real
        # arithmetic; where an autograd oracle executes it (full-covariance families) the
        # override leaves only a floating-point roundoff residue in the gradients.
        if self.decode_mode == "linear":
            mark(("m_decode_sigma_lr",), "linear decode has no independent variance table")
            mark(("decode_tau",), "read only by the prior-bank KL decode")
            mark(("decode_covariance",), "the linear readout reads no covariance")
            mark(("untie_decode_bank",),
                 "read only by the KL-to-bank decode; the linear decode is untied by construction")
        else:
            if not self.untie_decode_bank:
                mark(("m_decode_mu_lr", "m_decode_sigma_lr"),
                     "shared encode/decode tables use the encoder rates")
                mark(("decode_mu_weight_decay",),
                     "shared encode/decode tables: the decode means are mu_embed, decayed by mu_weight_decay")
            mark(("decode_second_order",),
                 "linear decode only; the prior-bank KL decoders already read Sigma_q")
        # Decay is read per optimizer group (vfe3/train.py build_optimizer). The global weight_decay reaches
        # only a group that sets no decay of its own: the block MLP and CG path weights, and the mean,
        # log-variance and decode-mean tables whose dedicated decay is None (audit 2026-10-10, item 13).
        if (not self.use_block_mlp and not self.use_cg_coupling
                and not any(self.inherits_hyperparameter(name, "weight_decay")
                            for name in ("mu_weight_decay", "sigma_weight_decay", "decode_mu_weight_decay"))):
            mark(("weight_decay",),
                 "every optimizer group sets its own decay (no block MLP or CG coupling, and "
                 "mu/sigma/decode-mean decays are set)")
        if "weight_decay" in phi_update_policy.optimizer_group_metadata:
            mark(("phi_weight_decay",),
                 f"m_phi_update_mode={self.m_phi_update_mode!r} replaces the frame-table decay "
                 "with its own optimizer-group value")
        if self.beta_attention_prior != "causal_windowed":
            mark(("attention_window",), "only the causal_windowed prior reads the band width")
        if self.gauge_transport in ("off", "frozen"):
            mark(("m_pos_phi_lr", "phi_weight_decay"),
                 f"gauge_transport={self.gauge_transport!r} sets requires_grad=False on the frame "
                 "tables, so no phi-role step or decay applies")
        from vfe3.gauge_optim import FRAME_CONDITION_INERT_GROUPS
        if self.gauge_transport in ("off", "frozen"):
            mark(("phi_frame_condition_max",),
                 f"gauge_transport={self.gauge_transport!r} freezes the frame tables, so no step is bounded")
        elif self.gauge_group in FRAME_CONDITION_INERT_GROUPS:
            mark(("phi_frame_condition_max",),
                 f"gauge_group={self.gauge_group!r} frames are orthogonal or conformal on every head "
                 "(cond_2 = 1), so there is nothing to bound")
        if self.attention_mode == "projected_predictive" or self.skip_belief_sigma_update:
            mark(("e_q_sigma_lr", "e_sigma_q_trust", "sigma_max"),
                 "no sigma E-step runs on projected routes or under skip_belief_sigma_update "
                 "(a moment-matched predictive covariance is a readout with its own floor), so no "
                 "sigma step, trust region, or SPD ceiling runs")
        elif (self.relational_receiver_update in ("closed_form", "closed_form_profiled")
              and self.family == "gaussian_full"):
            mark(("sigma_max",),
                 "the closed-form full-covariance receiver sets Sigma = c Lambda^{-1} with no SPD ceiling "
                 "(only the diagonal family clamps at sigma_max)")
        if self.attention_mode == "projected_predictive" and not self.diagnose_canonical_free_energy:
            mark(("lambda_beta",), "predictive_content weights no belief-coupling block; with "
                 "diagnose_canonical_free_energy=True it would feed that log-step diagnostic only")
        # stack._can_omit_prior_handoff at config level: predictive_content never reads the prior mean,
        # and no M-step prior consumer is enabled. The canonical diagnostic reads each layer's handed-off prior
        # (the capture keeps the handoff), so it is a consumer too (audit 2026-10-10 round 2, item 31). The
        # moment-matched and spread-corrected readouts floor at f * Sigma_p with the layer's prior covariance, so
        # there the variance handoff is read (audit 2026-10-12 round 2, item 8), but only under the prior floor or
        # a ceiling c * Sigma_p: the relative floor f (tr C_i / K) U_i U_i^T never reads Sigma_p (audit 2026-10-17,
        # item 23).
        if (self.attention_mode == "projected_predictive" and self.mstep_self_coupling_weight == 0.0
                and self.cg_energy_weight == 0.0 and not self.diagnose_canonical_free_energy):
            from vfe3.model.minimal_agent import READOUT_COVARIANCES
            floored = (self.predictive_covariance in READOUT_COVARIANCES
                       and (self.predictive_covariance_floor_mode in ("prior", "max")
                            or self.predictive_covariance_ceiling is not None))
            mark(("prior_handoff_rho",) if floored else ("prior_handoff_rho", "prior_handoff_sigma"),
                 "predictive_content never reads the prior mean"
                 + ("" if floored else " or, without a moment-matched covariance under the prior floor or a "
                    "ceiling, the prior covariance")
                 + ", and no M-step prior consumer is enabled; with diagnose_canonical_free_energy=True "
                 "they would feed that log-step diagnostic only")
        # The stack enters the E-step with q = p; one gradient iteration in one layer therefore
        # evaluates the self-coupling at D(q||p) = 0, where its force and derivatives vanish.
        # The closed-form receiver jumps to the frozen minimizer, which alpha sets, so the rule holds
        # only for a gradient step (audit 2026-10-07 F08/F23/F25/F71). The canonical diagnostic still
        # reads the self-coupling form (R(alpha) at D = 0, and D > 0 at the layer output).
        if (self.n_e_steps == 1 and self.n_layers == 1 and self.mstep_self_coupling_weight == 0.0
                and not self.diagnose_canonical_free_energy
                and not (self.attention_mode == "projected_gradient"
                         and self.relational_receiver_update in ("closed_form", "closed_form_profiled"))):
            mark(("lambda_alpha_mode", "b0", "c0"),
                 "the single gradient E-step iteration starts at q = p, where the self-coupling "
                 "force vanishes")
        # gaussian_block on this route is the same diagonal lookup in the dense layout (audit 2026-10-09
        # round 2, #20); the block_cholesky table stores a dense block per head, so it is excluded.
        if (self.attention_mode == "projected_predictive" and self.family in ("gaussian_full", "gaussian_block")
                and self.prior_covariance_table == "diagonal"
                and self.encode_mode in ("per_token", "per_token_additive")
                and self.predictive_covariance == "passthrough"
                and self.block_coordinates == "local" and self.decode_coordinates == "local"
                and not (self.use_block_mlp and self.block_mlp_covariance != "passthrough")
                and not (self.use_cg_coupling and self.cg_covariance_mode == "delta_full")
                and not (self.block_norm_covariance in ("pushforward", "pushforward_moment")
                         and self.norm_type_block == "layernorm")):
            # A passthrough gauge gate no longer distinguishes the families: the diagonal gate on
            # (mu, sigma) is the full gate on (mu, diag_embed sigma), and nothing densifies sigma.
            mark(("family",),
                 "value-identical to gaussian_diagonal under the rank-matched decoder on this "
                 "route: stored covariance is a diagonal token lookup and no seam densifies it; "
                 "full covariance requires block_mlp_covariance='delta_full', a block_norm_covariance pushforward, "
                 "predictive_covariance='moment_matched', or attention_mode='projected_gradient'")
        # Reference coordinates. With identically-I frames xi = mu and C = Sigma exactly, so both
        # selectors are no-ops; without a block norm or block MLP nothing is conjugated.
        identity_frames = (self.gauge_transport == "off"
                           or (self.encode_mode == "per_token_additive" and self.pos_phi == "none"))
        if identity_frames:
            mark(("block_coordinates", "decode_coordinates"),
                 "the realized frames are identically I (gauge_transport='off', or per_token_additive "
                 "without positional frames), so reference and local coordinates coincide")
        elif self.norm_type_block == "none" and not self.use_block_mlp:
            mark(("block_coordinates",),
                 "no block norm or block MLP to conjugate: reference and local coordinates "
                 "agree in exact arithmetic, but the reference block still runs the U(U^{-1} mu) round "
                 "trip, so the two differ at roundoff (not bit-identical)")
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

    def _pullback_uses_eigenbasis_fallback(self) -> bool:
        r"""Whether the pullback_group M-step runs without the strict series direction (audit 2026-10-16, F40).

        Under ``m_phi_group_retraction='group_log'`` with ``m_phi_group_log_fallback='eigenbasis'`` every row is
        either the closed form :math:`\xi = U^{-1}\bar G` or the eigenbasis inverse of :math:`\Psi_L`, so the
        structure-constant budget of :meth:`_pullback_structure_problem` does not apply.
        """
        from vfe3.gauge_optim import get_phi_update_policy
        try:
            policy = get_phi_update_policy(self.m_phi_update_mode)
        except (KeyError, ValueError):
            return False
        return (policy.requires_pullback_geometry and self.m_phi_group_retraction == "group_log"
                and self.m_phi_group_log_fallback == "eigenbasis")

    def _pullback_structure_problem(self) -> Optional[str]:
        r"""Why the strict pullback M-step cannot build this group's structure constants, or None.

        Mirrors the runtime budget of :func:`vfe3.geometry.phi_preconditioner._strict_structure_constants`
        (audit 2026-10-08, geometer#2). A group whose generators each act on one block prepares one
        (n_gen_b, d_b) basis per block; a tied group (``gauge_tying='shared'``) prepares one
        (n_gen, K) basis over the full width.
        """
        from vfe3.geometry.phi_preconditioner import STRICT_STRUCTURE_BYTES, strict_structure_bytes
        local_algebra = {"block_glk": lambda d: d * d, "block_so": lambda d: d * (d - 1) // 2,
                         "block_co": lambda d: d * (d - 1) // 2 + 1, "block_glmc": lambda d: d * d // 2,
                         "block_sl": lambda d: d * d - 1, "block_borel": lambda d: d * (d + 1) // 2,
                         "block_unipotent": lambda d: d * (d - 1) // 2}
        if self.gauge_group in ("so_n", "su_n"):
            n_alg = (int(self.group_n) * (int(self.group_n) - 1) // 2 if self.gauge_group == "so_n"
                     else int(self.group_n) ** 2 - 1)
            shapes = ([(n_alg, self.embed_dim)] if self.gauge_tying == "shared"
                      else [(n_alg, d) for d in sorted(set(self.head_dims)) if d > 1])   # 'l0' blocks carry none
        elif self.gauge_group in local_algebra:
            n_alg = local_algebra[self.gauge_group](self.d_head)
            shapes = [(n_alg, self.embed_dim if self.gauge_tying == "shared" else self.d_head)]
        else:
            return None
        for n_gen, K in shapes:
            need = strict_structure_bytes(n_gen, K)
            if need > STRICT_STRUCTURE_BYTES:
                return (f"needs {need / 2**20:.0f} MiB of structure-constant workspace for {n_gen} generators "
                        f"on a {K} x {K} block (gauge_group={self.gauge_group!r}, gauge_tying="
                        f"{self.gauge_tying!r}); the budget is {STRICT_STRUCTURE_BYTES / 2**20:.0f} MiB")
        return None

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

    def resolved_hyperparameter(self, name: str) -> float:
        r"""Effective value of a rate or decay: its own value, or, when None, its parent's (recursively).

        The parents are listed in ``_INHERITED_HYPERPARAMETERS``; a field outside it is returned as stored.
        """
        value = getattr(self, name)
        while value is None:
            name = _INHERITED_HYPERPARAMETERS[name]
            value = getattr(self, name)
        return value

    def inherits_hyperparameter(self, name: str, ancestor: str) -> bool:
        """True when ``name`` is None and its value resolves through ``ancestor``."""
        while getattr(self, name) is None:
            name = _INHERITED_HYPERPARAMETERS[name]
            if name == ancestor:
                return True
        return False

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
    if config.self_kl_max is not None and math.isinf(config.self_kl_max):
        payload["self_kl_max"] = "inf"
    if config.gauge_group == "so_n":
        # Basis marker (the irrep_basis field was removed 2026-10-12): config_from_serialized refuses an
        # so_n payload without it, which identifies a save from before 2026-10-05 built in the eigh basis.
        payload["irrep_basis"] = "canonical"
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
    # (frame_pair_fp64_escalation, retired the same day, was a live opt-in field again from 2026-10-04
    # until it merged into frame_pair_escalation_tol on 2026-10-12; a config without it loads as off.)
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
# The five projected-factorization fields merged into projected_factorization (2026-10-12).
_PROJECTED_FACTORIZATION_FIELDS = (
    "projected_cov_condition_recovery", "projected_factor_route", "projected_cov_recovery_precision",
    "projected_gram_condition_tol", "projected_factor_condition_tol")
# The four RoPE selectors merged into rope_mode (2026-10-12).
_ROPE_FIELDS = ("pos_rotation", "rope_space", "rope_on_value", "rope_comparison_covariance")
# Hardcoded on 2026-10-12: the paper's identity content init and the 1e-6 group-product residual gate;
# the frame-pair escalation switch and tolerance merged into frame_pair_escalation_tol.
_HARDCODED_FIELDS = ("relational_output_init", "bch_residual_max")
_FRAME_PAIR_FIELDS = ("frame_pair_fp64_escalation", "frame_pair_condition_tol")
# Removed 2026-10-12 (second pass): the final norm before decode is fixed at none, the block runs norm -> MLP
# with no post-MLP norm, and under amp_dtype the block MLP's mean path always runs in an fp32 island.
_NORM_MLP_FIELDS = ("norm_type_final", "block_mlp_post_norm", "block_mlp_autocast")
# Removed 2026-10-12 (second pass): the non-causal attention priors (each position saw future tokens).
_RETIRED_ATTENTION_PRIORS = frozenset({"uniform", "alibi", "windowed"})
# Runtime and diagnostic policies hardcoded/removed 2026-10-12 (second pass): no divergence abort, no NaN fill
# of uninitialized memory, a per-epoch random training-window origin and the matmul-precision replay ladder.
# None changes the model, so any saved value is dropped silently.
_RUNTIME_POLICY_FIELDS = ("divergence_abort", "fill_uninitialized_memory", "train_window_offset",
                          "matmul_precision_retry")
# Removed or hardcoded 2026-10-12 (second pass): the expected-likelihood CE kernel selector (CUDA tensors
# run the fused Triton kernel, CPU tensors the chunked torch kernel), the full-covariance pair-KL precision
# (fixed at the former "fp32_escalate_cond"), the so_n irrep basis (canonical only) and the linear decoder's
# bias toggle (the per-vocab bias is always on).
_NUMERICS_REMOVED_FIELDS = (
    "expected_likelihood_kernel", "full_cov_kl_precision", "irrep_basis", "decode_bias")
# removed 2026-10-12 (round-2 audit): the t5_relative_bias attention prior and its three fields, the mass_phi
# frame-norm penalty, and mu_trust_cholesky_rounds (now numerics.MU_TRUST_CHOLESKY_ROUNDS).
_T5_PRIOR_FIELDS = ("t5_num_buckets", "t5_max_distance", "t5_learnable_bias")
_ROUND2_REMOVED_FIELDS = (*_T5_PRIOR_FIELDS, "mass_phi", "mu_trust_cholesky_rounds")
# Every serialized key config_from_serialized migrates or refuses that is not a VFE3Config field.
RETIRED_CONFIG_KEYS = frozenset(
    set(_RETIRED_PINNED_FIELDS) | _RETIRED_INERT_FIELDS | _RETIRED_CONDITIONAL_FIELDS
    | {"grad_clip_per_role", "exp_fp64_mode", "compile_pair_kernel", "relational_mode",
       "relational_readout_mode", "relational_input_scope", "relational_output_scope",
       "relational_identity_content"}
    | set(_PROJECTED_FACTORIZATION_FIELDS) | set(_ROPE_FIELDS) | set(_HARDCODED_FIELDS)
    | set(_FRAME_PAIR_FIELDS) | set(_NORM_MLP_FIELDS)
    | set(_RUNTIME_POLICY_FIELDS) | set(_NUMERICS_REMOVED_FIELDS)
    # removed 2026-10-12 (second pass): the ambient_alignment receiver law and the constant alpha form
    | {"relational_receiver_law", "lambda_alpha"}
    | set(_ROUND2_REMOVED_FIELDS))
# laplace_diagonal (retired 2026-10-12) was registered but refused on both projected routes.
_RETIRED_FAMILIES = frozenset({"gaussian_frame_diagonal", "laplace_diagonal"})
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
    Must run after the attention selector and receiver-update defaults are resolved.
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
    for name in (*_RETIRED_INERT_FIELDS, *_RUNTIME_POLICY_FIELDS):
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
    if payload.get("decode_mode") in ("family", "family_chunked"):
        # The generic family decode ran only on a Gaussian family at Renyi order 1, where it equals the
        # fast kernel of the family's rank up to rounding (float64 loss differences of 2e-16 and 0).
        family = payload.get("family", VFE3Config.family)
        if family not in _FAMILY_DECODE_MIGRATIONS:
            refuse("decode_mode", payload["decode_mode"],
                   f" with family={family!r} (only gaussian_diagonal and gaussian_full migrate)")
        payload["decode_mode"] = _FAMILY_DECODE_MIGRATIONS[family]
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
    _migrate_projected_factorization(payload, source=source, refuse=refuse)
    _migrate_rope_mode(payload, source=source, refuse=refuse)
    _migrate_hardcoded_fields(payload, source=source, refuse=refuse)
    _migrate_norm_mlp_fields(payload, source=source, refuse=refuse)
    _migrate_numerics_removals(payload, source=source, refuse=refuse)
    _migrate_round2_removals(payload, refuse=refuse)


def _migrate_round2_removals(
    payload: dict[str, Any],

    *,
    refuse: Callable[..., None],
) -> None:
    r"""Drop the fields removed 2026-10-12 (round-2 audit), refusing the values that changed a forward or a loss.

    ``beta_attention_prior="t5_relative_bias"`` is refused; its three bucketing fields were read only by that
    prior, so they are dropped silently otherwise. ``mass_phi`` loads only as 0 (a positive value added
    :math:`\tfrac{m}{2}\|\phi\|^2` to the loss). ``mu_trust_cholesky_rounds`` is dropped: the ridge ladder is now
    the constant ``vfe3.numerics.MU_TRUST_CHOLESKY_ROUNDS`` and changed nothing where the round-0 factor succeeded.
    """
    if payload.get("beta_attention_prior") == "t5_relative_bias":
        refuse("beta_attention_prior", "t5_relative_bias", " (the T5 relative-position prior was removed 2026-10-12)")
    for name in _T5_PRIOR_FIELDS:
        payload.pop(name, None)
    if "mass_phi" in payload:
        value = payload.pop("mass_phi")
        if isinstance(value, bool) or not isinstance(value, Real) or value != 0:
            refuse("mass_phi", value, " (only 0 survives; the frame-norm penalty was removed 2026-10-12)")
    payload.pop("mu_trust_cholesky_rounds", None)


def _migrate_norm_mlp_fields(
    payload: dict[str, Any],

    *,
    source: str,
    refuse: Callable[..., None],
) -> None:
    r"""Drop the norm and block-MLP fields removed on 2026-10-12 (second pass), or refuse them, and refuse a
    removed attention prior.

    ``norm_type_final`` loads only as ``"none"``: any other final norm changed the decoded mean. ``block_mlp_post_norm``
    loads only as False: True re-normalized the post-MLP mean (validation refused it wherever it was not read).
    ``block_mlp_autocast`` was a precision policy, so any saved value is dropped. A non-causal
    ``beta_attention_prior`` (``uniform``, ``alibi``, ``windowed``) is refused.
    """
    if "norm_type_final" in payload:
        value = payload.pop("norm_type_final")
        if value != "none":
            refuse("norm_type_final", value, " (only 'none' survives; a final norm changed the decoded mean)")
    if "block_mlp_post_norm" in payload:
        value = _normalize_serialized_boolean(payload.pop("block_mlp_post_norm"), "block_mlp_post_norm",
                                              source=source)
        if value:
            refuse("block_mlp_post_norm", value, " (only False survives; the post-MLP norm changed the forward)")
    payload.pop("block_mlp_autocast", None)
    prior = payload.get("beta_attention_prior")
    if isinstance(prior, str) and prior in _RETIRED_ATTENTION_PRIORS:
        refuse("beta_attention_prior", prior, " (a non-causal prior; only causal priors survive)")


def _migrate_numerics_removals(
    payload: dict[str, Any],

    *,
    source: str,
    refuse: Callable[..., None],
) -> None:
    r"""Drop the fields removed on 2026-10-12 (second pass), refusing the two that changed a forward.

    ``expected_likelihood_kernel`` and ``full_cov_kl_precision`` chose how one quantity was computed
    (agreement to rounding), so any saved value is dropped. ``irrep_basis`` is dropped for every
    non-so_n config and for an so_n config saved as ``"canonical"``; an so_n config saved as ``"eigh"``,
    or without the field (saved before 2026-10-05, built in the eigh basis), is refused because its
    learned tables live in the eigh basis and would load against canonical frames. ``decode_bias`` is
    dropped when True or when the decoder is not linear (it was inert there); a linear decoder saved
    with ``decode_bias=False`` is refused, since its checkpoint carries no ``output_proj_bias``.
    """
    payload.pop("expected_likelihood_kernel", None)
    payload.pop("full_cov_kl_precision", None)
    so_n = payload.get("gauge_group") == "so_n"
    if so_n and payload.get("irrep_basis") != "canonical":
        basis = payload.get("irrep_basis")
        saved = "without irrep_basis" if basis is None else f"with irrep_basis={basis!r}"
        raise ValueError(
            f"serialized so_n config from {source} was saved {saved}; such configs were built in "
            "the eigh irrep basis, which was removed 2026-10-12. Its learned tables live in that "
            "basis and would load against the canonical frames, changing the forward; continuation "
            "requires explicit migration")
    payload.pop("irrep_basis", None)
    if "decode_bias" in payload:
        bias = _normalize_serialized_boolean(payload.pop("decode_bias"), "decode_bias", source=source)
        if not bias and payload.get("decode_mode", VFE3Config.decode_mode) == "linear":
            refuse("decode_bias", False, " (the linear decoder always carries its learned per-vocab "
                   "bias since 2026-10-12, and this checkpoint has no output_proj_bias)")


def _migrate_hardcoded_fields(
    payload: dict[str, Any],

    *,
    source: str,
    refuse: Callable[..., None],
) -> None:
    r"""Drop the fields hardcoded on 2026-10-12 and merge the frame-pair escalation pair.

    ``relational_output_init`` loads when ``"identity"`` or where it was inert (the minimal-agent readout has
    no content factor O); ``"depth_scaled"`` elsewhere is refused. ``bch_residual_max`` loads when ``None``,
    when at least 1e-6 (the gate is ``min(1e-6, value)``, so the gate is the same 1e-6), or where no manifold
    phi optimizer read it; a tighter gate is refused. ``frame_pair_fp64_escalation`` and
    ``frame_pair_condition_tol`` become ``frame_pair_escalation_tol``: the saved tolerance when the escalation
    was on, else ``None`` (an absent switch was off).
    """
    if "relational_output_init" in payload:
        value = payload.pop("relational_output_init")
        inert = (payload.get("attention_mode") == "projected_predictive"
                 and payload.get("predictive_readout", VFE3Config.predictive_readout) == "minimal_agent")
        if value not in ("identity", "depth_scaled"):
            refuse("relational_output_init", value, " (only 'identity' or 'depth_scaled' was valid)")
        if value != "identity" and not inert:
            refuse("relational_output_init", value, " (the initial content scale is fixed at the identity)")
    if "bch_residual_max" in payload:
        value = payload.pop("bch_residual_max")
        if value is not None:
            from vfe3.gauge_optim import get_phi_update_policy
            try:
                read = get_phi_update_policy(
                    payload.get("m_phi_update_mode", VFE3Config.m_phi_update_mode)).requires_manifold_optimizer
            except (KeyError, ValueError):
                read = True
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    or value <= 0.0):
                refuse("bch_residual_max", value, " (only None or a finite positive gate was valid)")
            if read and value < 1e-6:
                refuse("bch_residual_max", value, " (the group-product residual gate is fixed at 1e-6)")
    present = [name for name in _FRAME_PAIR_FIELDS if name in payload]
    if present:
        if "frame_pair_escalation_tol" in payload:
            raise ValueError(f"serialized config from {source} has conflicting frame-pair escalation "
                             f"selectors: frame_pair_escalation_tol and {present}")
        enabled = _normalize_serialized_boolean(
            payload.pop("frame_pair_fp64_escalation", False), "frame_pair_fp64_escalation", source=source)
        tol = payload.pop("frame_pair_condition_tol", 1e-4)
        payload["frame_pair_escalation_tol"] = tol if enabled else None


def _migrate_rope_mode(
    payload: dict[str, Any],

    *,
    source: str,
    refuse: Callable[..., None],
) -> None:
    r"""Merge ``pos_rotation``, ``rope_space``, ``rope_on_value`` and ``rope_comparison_covariance`` into ``rope_mode``.

    Every combination the old validator admitted maps to exactly one mode: ``pos_rotation="none"`` to ``none``
    (the other three were inert); frame space to ``frame`` or, with ``rope_on_value``, ``frame_value`` (the
    comparison covariance was inert); comparison space to ``comparison`` (covariance ``full``) or
    ``comparison_invariant`` (``rotation_invariant``). Comparison space with ``rope_on_value=True`` was refused
    and stays refused. An absent ``rope_space`` is the historical frame space (its 2026-10-07 pin row).
    """
    present = [name for name in _ROPE_FIELDS if name in payload]
    if not present:
        return
    if "rope_mode" in payload:
        raise ValueError(f"serialized config from {source} has conflicting RoPE selectors: rope_mode and {present}")
    rotation = payload.pop("pos_rotation", "none")
    space = payload.pop("rope_space", "frame")
    on_value = _normalize_serialized_boolean(payload.pop("rope_on_value", False), "rope_on_value", source=source)
    covariance = payload.pop("rope_comparison_covariance", "full")
    for name, value, valid in (("pos_rotation", rotation, ("none", "rope")),
                               ("rope_space", space, ("frame", "comparison")),
                               ("rope_comparison_covariance", covariance, ("full", "rotation_invariant"))):
        if value not in valid:
            refuse(name, value, f" (only {valid} was valid)")
    if rotation == "none":
        payload["rope_mode"] = "none"
    elif space == "frame":
        payload["rope_mode"] = "frame_value" if on_value else "frame"
    elif on_value:
        refuse("rope_on_value", True, " with rope_space='comparison' (never a valid combination)")
    else:
        payload["rope_mode"] = "comparison_invariant" if covariance == "rotation_invariant" else "comparison"


def _migrate_projected_factorization(
    payload: dict[str, Any],

    *,
    source: str,
    refuse: Callable[..., None],
) -> None:
    r"""Merge the five projected-factorization fields into ``projected_factorization`` (2026-10-12).

    ``(recovery, route)`` maps to ``gram`` for ``(False, gram)``, ``gram_certified`` for ``(True, gram)``
    and ``cholesky_qr2_certified`` for either ``cholesky_qr2`` tuple; an uncertified CholeskyQR2 run now
    also rebuilds the entries that fail the factor certificate. The tolerances are module constants at
    their defaults (1e-4 Gram, 3e-4 factor); a non-default tolerance under recovery is refused, and any
    tolerance without recovery was inert. The recovery precision is fixed at fp32 (shifted CholeskyQR3
    with a float64 fallback); a saved fp64 precision loads and replays a recovery with rounding-level
    differences, as the 2026-10-02 numerical-policy hardcoding did.
    """
    present = [name for name in _PROJECTED_FACTORIZATION_FIELDS if name in payload]
    if not present:
        return
    if "projected_factorization" in payload:
        raise ValueError(f"serialized config from {source} has conflicting projected-factorization "
                         f"selectors: projected_factorization and {present}")
    recovery = _normalize_serialized_boolean(
        payload.pop("projected_cov_condition_recovery", False), "projected_cov_condition_recovery",
        source=source)
    route = payload.pop("projected_factor_route", "gram")
    precision = payload.pop("projected_cov_recovery_precision", "fp64")
    tolerances = {name: payload.pop(name, default) for name, default in (
        ("projected_gram_condition_tol", 1e-4), ("projected_factor_condition_tol", 3e-4))}
    if route not in ("gram", "cholesky_qr2"):
        refuse("projected_factor_route", route, " (only 'gram' or 'cholesky_qr2' was valid)")
    if precision not in ("fp64", "fp32"):
        refuse("projected_cov_recovery_precision", precision, " (only 'fp64' or 'fp32' was valid)")
    if recovery:
        for name, default in (("projected_gram_condition_tol", 1e-4), ("projected_factor_condition_tol", 3e-4)):
            if not _same_serialized_value(tolerances[name], default):
                refuse(name, tolerances[name], f" (the certificate tolerance is fixed at {default!r})")
    payload["projected_factorization"] = (
        "cholesky_qr2_certified" if route == "cholesky_qr2"
        else "gram_certified" if recovery else "gram")


# Introduction-time values of fields whose default was later flipped, applied by config_from_serialized
# only when a serialized config lacks the field (it predates it). Add a row whenever a default flips.
#   content_link_chart                "affine" at 6e54aee (2026-10-03), now "exponential"
#   predictive_covariance             "passthrough" at 8b398f3 (2026-09-27), now "moment_matched"
#   m_decode_mu_lr, m_decode_sigma_lr None (inherit) at d04dfc4 (2026-09-30), now 0.025 / 0.001
#   projected_cov_recovery_precision  "fp64" at fed4164 (2026-10-04); merged into projected_factorization
#                                     on 2026-10-12, whose migration treats an absent precision as fp64
#   rope_space                        "frame" from eaa3a21 (2026-10-07); merged into rope_mode on 2026-10-12,
#                                     whose migration treats an absent rope_space as "frame"
#   kl_max                            100.0 until 2026-10-12, now None (8 * embed_dim)
#   block_norm_covariance             "passthrough" at 89eedfb (2026-10-10): a record without it ran the mean-only norm
#   relational_receiver_measurement   "adjoint" from 2026-10-10 until 2026-10-12, now "inverse"
#   decode_logits_amp                 False at 46a3a6c (2026-10-07)
#   decode_ce_triton                  False at 5e5a118 (2026-10-12)
#   routing_kl_max_reference          "absolute" at e0a3bb4 (2026-10-08)
_HISTORICAL_DEFAULT_PINS: dict[str, Any] = {
    "content_link_chart":               "affine",
    "predictive_covariance":            "passthrough",
    "m_decode_mu_lr":                   None,
    "m_decode_sigma_lr":                None,
    "kl_max":                           100.0,
    "relational_receiver_measurement":  "adjoint",
    "decode_logits_amp":                False,
    "decode_ce_triton":                 False,
    "routing_kl_max_reference":         "absolute",
    "block_norm_covariance":            "passthrough",
}


def historical_default_pins(
    payload: Mapping[str, Any],    # a serialized config, before migration
) -> dict[str, Any]:               # field -> the value config_from_serialized fills in for it
    r"""The values :func:`config_from_serialized` gives the fields ``payload`` lacks, other than today's defaults.

    Every row of ``_HISTORICAL_DEFAULT_PINS`` whose field is absent, and one era-keyed value. The field
    ``predictive_covariance_floor_mode`` (322843b, 2026-10-16, default ``"prior"``) selects a floor that had already
    changed at 5f5d484 (2026-10-12): before then the moment-matched readouts used the relative floor
    :math:`f\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top` (``"relative"``), after it :math:`f\Sigma_{p,i}`
    (``"prior"``). A record without the field is from either era, so no fixed pin is right. 5f5d484 also added
    ``phi_frame_condition_max``, which every later record carries (as ``null`` when off), so a record lacking both
    ran ``"relative"``, and one carrying ``phi_frame_condition_max`` keeps the default ``"prior"``.

    ``block_norm_trace_cap`` (3aeede1, 2026-10-17, default True) has the same era key. The LayerNorm pushforward
    entered uncapped at 89eedfb (2026-10-10) and the sphere-bound cap was added without a toggle at 5f5d484, so a
    record lacking both the toggle and ``phi_frame_condition_max`` ran the pushforward (and the ``delta_full``
    composite) uncapped and is pinned False; one carrying ``phi_frame_condition_max`` keeps the default True
    (audit 2026-10-17, item 51). Records older than 89eedfb run the pinned passthrough norm, where the cap is inert.
    """
    pins = {name: value for name, value in _HISTORICAL_DEFAULT_PINS.items() if name not in payload}
    if "predictive_covariance_floor_mode" not in payload and "phi_frame_condition_max" not in payload:
        pins["predictive_covariance_floor_mode"] = "relative"
    if "block_norm_trace_cap" not in payload and "phi_frame_condition_max" not in payload:
        pins["block_norm_trace_cap"] = False
    return pins


def _migrate_receiver_law_and_alpha(
    payload: dict[str, Any],

    *,
    source: str,
) -> None:
    r"""Drop ``relational_receiver_law`` and ``lambda_alpha`` (removed 2026-10-12, second pass).

    The receiver law is now always the channel likelihood. The law and ``relational_receiver_update`` were
    introduced together (2017ea3, 2026-10-02), so a payload with neither predates both and ran
    ``ambient_alignment``, while one with only the update was saved after this removal and ran the channel
    likelihood. Must run before the update default is filled in. A resolved ``ambient_alignment`` on
    ``projected_gradient`` is refused (the receiver forward changed); elsewhere the law was inert.
    ``lambda_alpha_mode="constant"`` is refused (the self-coupling forward changed); ``lambda_alpha`` was
    read only by that form, so it is dropped silently otherwise.
    """
    predates_law = "relational_receiver_update" not in payload
    law = payload.pop("relational_receiver_law", "ambient_alignment" if predates_law else "channel_likelihood")
    if law not in ("channel_likelihood", "ambient_alignment"):
        raise ValueError(f"serialized config from {source} has invalid relational_receiver_law={law!r}")
    if law == "ambient_alignment" and payload.get("attention_mode") == "projected_gradient":
        raise ValueError(
            f"serialized config from {source} requests retired relational_receiver_law='ambient_alignment' "
            "on projected_gradient (only the channel likelihood survives); continuation requires explicit "
            "migration")
    if payload.get("lambda_alpha_mode") == "constant":
        raise ValueError(
            f"serialized config from {source} requests retired lambda_alpha_mode='constant' (only the "
            "state-dependent forms survive); continuation requires explicit migration")
    payload.pop("lambda_alpha", None)


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
    _migrate_receiver_law_and_alpha(payload, source=source)
    payload.setdefault("relational_receiver_update", "euler")
    # Fields whose default changed after they were introduced: a config saved before the field existed
    # ran the introduction-time value, not today's default (audit 2026-10-08 P2-3), and the floor mode
    # follows its era marker (audit 2026-10-16 F1).
    payload.update(historical_default_pins(payload))
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
    if values.get("self_kl_max") == "inf":
        values["self_kl_max"] = math.inf
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
